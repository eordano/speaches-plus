#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>
#include "nv_kernels.h"

namespace {

constexpr int kBlockDim = 256;
constexpr int kRcRefusesShapeThePackedLayoutCannotTile = -3;
constexpr int kMarlinScaleHalfMult2Pow7 = 128;
constexpr float kMarlinMinScale1Over64 = 1.0f / 64.0f;
constexpr float kMarlinGlobalExponentBias2Pow119 = 6.646139e35f;
constexpr int kFlagScaleOverflowsHalf = 1;
constexpr int kFlagNonzeroScaleMappedToZero = 2;

#define NV_DX_UE4M3_SUBNORMAL_STEP 0.001953125f
#define NV_DX_UE4M3_MIN_NORMAL 0.015625f

__device__ __forceinline__ float dx_decode_e2m1(uint8_t nib) {
    static const float kE2M1[16] = {
         0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
    };
    return kE2M1[nib & 0xF];
}

__device__ __forceinline__ uint8_t dx_encode_e2m1(float x) {
    static const float kMag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    uint8_t sign = signbit(x) ? 0x8 : 0x0;
    float a = fabsf(x);
    uint8_t best = 0;
    float best_err = INFINITY;
    for (int i = 0; i < 8; ++i) {
        float err = fabsf(a - kMag[i]);
        if (err < best_err) {
            best_err = err;
            best = (uint8_t)i;
        }
    }
    return sign | best;
}

__device__ __forceinline__ float dx_decode_ue4m3(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_DX_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
}

__device__ __forceinline__ uint8_t dx_encode_ue4m3(float scale) {
    if (!isfinite(scale) || scale <= 0.f) return 0;
    float clamped = fminf(scale, 448.f);
    if (clamped < NV_DX_UE4M3_MIN_NORMAL) {
        int m = (int)roundf(clamped / NV_DX_UE4M3_SUBNORMAL_STEP);
        if (m <= 0) return 0;
        if (m <= 7) return (uint8_t)m;
        return 0x08;
    }
    float lg = log2f(clamped);
    int e = (int)floorf(lg);
    float mant_f = (clamped / exp2f((float)e)) - 1.f;
    int mant = (int)roundf(mant_f * 8.f);
    if (mant >= 8) {
        mant = 0;
        e += 1;
    }
    mant = mant < 0 ? 0 : (mant > 7 ? 7 : mant);
    int biased = e + 7;
    biased = biased < 1 ? 1 : (biased > 15 ? 15 : biased);
    uint8_t byte = (uint8_t)((biased << 3) | (mant & 0x07));
    return byte == 0x7F ? 0x7E : byte;
}

__device__ __forceinline__ int dx_swizzled_scale_src(int m, int kb, int k_blocks) {
    int k_tiles = (k_blocks + 3) / 4;
    int m_tile = m / 128;
    int d2 = (m / 32) & 3;
    int d3 = m & 31;
    int k_tile = kb / 4;
    int d5 = kb & 3;
    return ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
}

__global__ void dx_twin_requant_transposed_kmajor_kernel(
    const uint8_t* __restrict__ w_packed,
    const uint8_t* __restrict__ w_scales_swizzled,
    float alpha,
    int n_out,
    int k_in,
    uint32_t* __restrict__ kmajor_out,
    uint8_t* __restrict__ scales_linear_out
) {
    int blocks_per_twin_row = n_out / 16;
    long long total = (long long)k_in * blocks_per_twin_row;
    long long stride = (long long)gridDim.x * blockDim.x;
    int src_bytes_per_row = k_in / 2;
    int src_k_blocks = k_in / 16;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        int c = (int)(idx / blocks_per_twin_row);
        int b = (int)(idx % blocks_per_twin_row);
        float v[16];
        float amax = 0.f;
        int src_kb = c / 16;
        uint8_t shift = (uint8_t)((c & 1) * 4);
        for (int i = 0; i < 16; ++i) {
            int r = b * 16 + i;
            uint8_t byte = w_packed[(long long)r * src_bytes_per_row + (c >> 1)];
            float scale = dx_decode_ue4m3(w_scales_swizzled[dx_swizzled_scale_src(r, src_kb, src_k_blocks)]) * alpha;
            float x = __bfloat162float(__float2bfloat16(dx_decode_e2m1((byte >> shift) & 0xF) * scale));
            v[i] = x;
            amax = fmaxf(amax, fabsf(x));
        }
        float local_scale = (amax == 0.f) ? 1.f : amax / 6.f;
        uint8_t scale_byte = dx_encode_ue4m3(local_scale);
        float scale_decoded = dx_decode_ue4m3(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : 1.f / scale_decoded;
        uint8_t packed[8];
        for (int j = 0; j < 8; ++j) {
            float s0 = fminf(fmaxf(v[2 * j] * inv, -6.f), 6.f);
            float s1 = fminf(fmaxf(v[2 * j + 1] * inv, -6.f), 6.f);
            uint8_t lo = dx_encode_e2m1(s0);
            uint8_t hi = dx_encode_e2m1(s1);
            packed[j] = (uint8_t)((hi << 4) | (lo & 0x0F));
        }
        for (int h = 0; h < 2; ++h) {
            uint32_t word = (uint32_t)packed[4 * h] | ((uint32_t)packed[4 * h + 1] << 8) |
                            ((uint32_t)packed[4 * h + 2] << 16) | ((uint32_t)packed[4 * h + 3] << 24);
            kmajor_out[(long long)(b * 2 + h) * k_in + c] = word;
        }
        scales_linear_out[(long long)c * blocks_per_twin_row + b] = scale_byte;
    }
}

__global__ void dx_twin_scale_min_init_kernel(unsigned int* vmin_bits, int* flags, float* global_out) {
    *vmin_bits = 0x7f800000u;
    *flags = 0;
    *global_out = 0.f;
}

__global__ void dx_twin_scale_min_kernel(const uint8_t* __restrict__ scales_linear, long long count,
                                         unsigned int* vmin_bits) {
    long long stride = (long long)gridDim.x * blockDim.x;
    float local = INFINITY;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        float v = dx_decode_ue4m3(scales_linear[i]);
        if (v > 0.f) local = fminf(local, v);
    }
    if (local < INFINITY) atomicMin(vmin_bits, __float_as_uint(local));
}

__device__ __forceinline__ int dx_fold_from_vmin(unsigned int vmin_bits) {
    float vmin = __uint_as_float(vmin_bits);
    int fold = 0;
    if (isfinite(vmin)) {
        while (vmin * exp2f((float)fold) < kMarlinMinScale1Over64) fold += 1;
    }
    return fold;
}

__global__ void dx_twin_scales_to_marlin_kernel(
    const uint8_t* __restrict__ scales_linear,
    int n,
    int kb,
    const unsigned int* __restrict__ vmin_bits,
    float global_scale,
    uint8_t* __restrict__ bytes_e5m3_out,
    float* __restrict__ global_out,
    int* __restrict__ flags
) {
    int fold = dx_fold_from_vmin(*vmin_bits);
    float fold_mult = exp2f((float)fold);
    long long total = (long long)n * kb;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x; o < total; o += stride) {
        static const int kQuadPerm[4] = {0, 2, 1, 3};
        long long p = (o / 4) * 4 + kQuadPerm[o % 4];
        int idx = (int)(p % 64);
        long long t = (p / 64) * 64 + ((idx / 8) + 8 * (idx % 8));
        long long g = t / n;
        long long row = t % n;
        uint8_t src = scales_linear[row * kb + g];
        float v = dx_decode_ue4m3(src) * fold_mult;
        uint8_t out = 0;
        if (v > 0.f) {
            float scaled = v * (float)kMarlinScaleHalfMult2Pow7;
            if (!(scaled < 65504.f)) atomicOr(flags, kFlagScaleOverflowsHalf);
            unsigned short h = __half_as_ushort(__float2half_rn(scaled));
            unsigned short shifted = (unsigned short)(h << 1);
            out = (uint8_t)(shifted >> 8);
            if (out == 0) atomicOr(flags, kFlagNonzeroScaleMappedToZero);
        }
        bytes_e5m3_out[o] = out;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *global_out = global_scale / fold_mult * kMarlinGlobalExponentBias2Pow119;
    }
}

int dx_grid(long long total) {
    long long blocks = (total + kBlockDim - 1) / kBlockDim;
    if (blocks > (1 << 16)) blocks = 1 << 16;
    if (blocks < 1) blocks = 1;
    return (int)blocks;
}

}

extern "C" int nv_kernels_dx_twin_nvfp4_requant_transposed_kmajor(
    void* stream,
    const uint8_t* w_packed,
    const uint8_t* w_scales_swizzled,
    float alpha,
    int n_out,
    int k_in,
    uint32_t* kmajor_out,
    uint8_t* scales_linear_out
) {
    if (n_out <= 0 || k_in <= 0 || n_out % 16 != 0 || k_in % 16 != 0) {
        return kRcRefusesShapeThePackedLayoutCannotTile;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    long long total = (long long)k_in * (n_out / 16);
    dx_twin_requant_transposed_kmajor_kernel<<<dx_grid(total), kBlockDim, 0, s>>>(
        w_packed, w_scales_swizzled, alpha, n_out, k_in, kmajor_out, scales_linear_out);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_dx_twin_nvfp4_scales_to_marlin(
    void* stream,
    const uint8_t* scales_linear,
    int n,
    int kb,
    float global_scale,
    uint8_t* bytes_e5m3_out,
    float* global_out,
    unsigned int* vmin_scratch,
    int* flags_out
) {
    if (n <= 0 || kb <= 0 || ((long long)n * kb) % 64 != 0) {
        return kRcRefusesShapeThePackedLayoutCannotTile;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    long long total = (long long)n * kb;
    dx_twin_scale_min_init_kernel<<<1, 1, 0, s>>>(vmin_scratch, flags_out, global_out);
    dx_twin_scale_min_kernel<<<dx_grid(total), kBlockDim, 0, s>>>(scales_linear, total, vmin_scratch);
    dx_twin_scales_to_marlin_kernel<<<dx_grid(total), kBlockDim, 0, s>>>(
        scales_linear, n, kb, vmin_scratch, global_scale, bytes_e5m3_out, global_out, flags_out);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
