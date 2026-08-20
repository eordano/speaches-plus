#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

namespace {

constexpr int kBlockDim = 256;
constexpr int kRowLanes = 32;
constexpr int kGroups = kBlockDim / kRowLanes;

static_assert(kGroups <= kRowLanes, "cross-group reduction must fit in one lane group");

__device__ __forceinline__ float lane_group_sum32(float v) {
    #pragma unroll
    for (int offset = kRowLanes / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, kRowLanes);
    }
    return v;
}

__device__ __forceinline__ float decode_e2m1_dev(uint8_t nib) {

    static const float kE2M1[16] = {
         0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
    };
    return kE2M1[nib & 0xF];
}

__device__ __forceinline__ uint8_t encode_e2m1_dev(float x) {
    static const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    uint8_t sign = signbit(x) ? 0b1000 : 0;
    float a = fabsf(x);
    uint8_t best = 0;
    float best_err = INFINITY;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float err = fabsf(a - kE2M1[i]);
        if (err < best_err) {
            best_err = err;
            best = (uint8_t)i;
        }
    }
    return (uint8_t)(sign | best);
}

__device__ __forceinline__ uint8_t encode_ue4m3_dev(float scale) {
    if (!isfinite(scale) || scale <= 0.f) return 0;
    float clamped = fminf(scale, 448.f);
    if (clamped < 0.015625f) {
        int m = (int)roundf(clamped * 512.f);
        if (m <= 0) return 0;
        if (m <= 7) return (uint8_t)m;
        return 0x08;
    }
    int bin_exp = 0;
    float frac = frexpf(clamped, &bin_exp);
    int exp_v = bin_exp - 1;
    float mant_f = frac * 2.f - 1.f;
    int mant = (int)roundf(mant_f * 8.f);

    if (mant >= 8) { mant = 0; exp_v += 1; }
    if (mant < 0) mant = 0;
    if (mant > 7) mant = 7;
    int biased = exp_v + 7;
    if (biased < 1) biased = 1;
    if (biased > 15) biased = 15;
    uint8_t byte = ((uint8_t)biased << 3) | (uint8_t)(mant & 0x07);
    return (byte == 0x7F) ? 0x7E : byte;
}

__device__ __forceinline__ float decode_ue4m3_dev(uint8_t b) {
    int exp_v = ((int)(b >> 3) & 0x0F);
    float mant = (float)(b & 0x07);
    if (exp_v == 0) return mant * 0.001953125f;
    return (1.f + mant / 8.f) * ldexpf(1.f, exp_v - 7);
}

__device__ __forceinline__ int swizzled_scale_dst(int m, int kb, int k_blocks) {
    int k_tiles = (k_blocks + 3) / 4;
    int m_tile = m / 128;
    int d2 = (m / 32) & 3;
    int d3 = m & 31;
    int k_tile = kb / 4;
    int d5 = kb & 3;
    return ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
}

__global__ void nvfp4_quantize_row_bf16_kernel(
    const __hip_bfloat16* __restrict__ x,
    uint8_t* __restrict__ packed_out,
    uint8_t* __restrict__ scales_out,
    float stored_global,
    int K
) {
    int tid = threadIdx.x;
    int n_blocks = K / 16;
    float stored = (stored_global == 0.f || !isfinite(stored_global)) ? 1.f : stored_global;

    for (int kb = tid; kb < n_blocks; kb += kBlockDim) {
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v = __bfloat162float(x[kb * 16 + i]);
            vals[i] = v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        uint8_t scale_byte = encode_ue4m3_dev(stored * local_scale);
        float scale_decoded = decode_ue4m3_dev(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);

        uint8_t* dst = packed_out + kb * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v_lo = fminf(fmaxf(vals[2*i]     * inv, -6.f), 6.f);
            float v_hi = fminf(fmaxf(vals[2*i + 1] * inv, -6.f), 6.f);
            uint8_t lo = encode_e2m1_dev(v_lo);
            uint8_t hi = encode_e2m1_dev(v_hi);
            dst[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
        }

        scales_out[kb] = scale_byte;
    }
}

__global__ void nvfp4_gemv_bf16_kernel(
    const uint8_t* __restrict__ W_packed,
    const uint8_t* __restrict__ W_scales,
    const uint8_t* __restrict__ x_packed,
    const uint8_t* __restrict__ x_scales,
    __hip_bfloat16* __restrict__ y,
    float alpha,
    int N,
    int K
) {
    int n = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;
    int n_blocks = K / 16;
    const uint8_t* w_row = W_packed + (size_t)n * (K / 2);

    float acc = 0.f;
    for (int kb = tid; kb < n_blocks; kb += kBlockDim) {

        int w_scale_idx = swizzled_scale_dst(n, kb, n_blocks);
        float w_scale = decode_ue4m3_dev(W_scales[w_scale_idx]);
        float x_scale = decode_ue4m3_dev(x_scales[kb]);
        float block_scale = w_scale * x_scale;

        const uint8_t* w_block = w_row + kb * 8;
        const uint8_t* x_block = x_packed + kb * 8;
        float block_dot = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t wb = w_block[i];
            uint8_t xb = x_block[i];
            float w_lo = decode_e2m1_dev(wb & 0xF);
            float w_hi = decode_e2m1_dev((wb >> 4) & 0xF);
            float x_lo = decode_e2m1_dev(xb & 0xF);
            float x_hi = decode_e2m1_dev((xb >> 4) & 0xF);
            block_dot += w_lo * x_lo + w_hi * x_hi;
        }
        acc += block_scale * block_dot;
    }

    acc = lane_group_sum32(acc);

    __shared__ float smem[kGroups];
    int lane = tid & (kRowLanes - 1);
    int grp = tid / kRowLanes;
    if (lane == 0) smem[grp] = acc;
    __syncthreads();

    if (grp == 0) {
        acc = (lane < kGroups) ? smem[lane] : 0.f;
        acc = lane_group_sum32(acc);
        if (lane == 0) y[n] = __float2bfloat16(acc * alpha);
    }
}

}

extern "C" int nv_kernels_nvfp4_quantize_row_bf16(
    void* stream,
    const uint16_t* x,
    uint8_t* packed_out,
    uint8_t* scales_out,
    float stored_global,
    int K
) {
    if ((K & 15) != 0) return -1;
    hipStream_t s = (hipStream_t)stream;
    nvfp4_quantize_row_bf16_kernel<<<1, kBlockDim, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(x),
        packed_out,
        scales_out,
        stored_global,
        K
    );
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_nvfp4_gemv_bf16(
    void* stream,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const uint8_t* x_packed,
    const uint8_t* x_scales,
    uint16_t* y,
    float alpha,
    int N,
    int K
) {
    if (N <= 0 || K <= 0) return 0;
    if ((K & 15) != 0) return -1;
    hipStream_t s = (hipStream_t)stream;
    dim3 grid((unsigned)N);
    dim3 block(kBlockDim);
    nvfp4_gemv_bf16_kernel<<<grid, block, 0, s>>>(
        W_packed,
        W_scales,
        x_packed,
        x_scales,
        reinterpret_cast<__hip_bfloat16*>(y),
        alpha,
        N, K
    );
    return (int)hipGetLastError();
}
