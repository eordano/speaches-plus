
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#include "nv_kernels.h"

namespace {

template <int BLOCK>
__device__ inline float rq_block_sum(float v) {
    constexpr int kWarp = 32;
    constexpr int kWarps = BLOCK / kWarp;
    __shared__ float warp_sums[kWarps];
    __shared__ float total;
    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

__device__ __forceinline__ uint8_t rq_encode_e2m1(float x) {
    static const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    uint8_t sign = signbit(x) ? 0b1000 : 0;
    float a = fabsf(x);
    uint8_t best = 0;
    float best_err = INFINITY;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float err = fabsf(a - kE2M1[i]);
        if (err < best_err) { best_err = err; best = (uint8_t)i; }
    }
    return (uint8_t)(sign | best);
}

#define NV_UE4M3_MIN_NORMAL      0.015625f
#define NV_UE4M3_SUBNORMAL_STEP  0.001953125f

__device__ __forceinline__ uint8_t rq_encode_ue4m3(float scale) {
    if (!isfinite(scale) || scale <= 0.f) return 0;
    float clamped = fminf(scale, 448.f);
    if (clamped < NV_UE4M3_MIN_NORMAL) {
        int sub = (int)roundf(clamped / NV_UE4M3_SUBNORMAL_STEP);
        if (sub <= 0) return 0;
        if (sub <= 7) return (uint8_t)sub;
        return 0x08;
    }
    int e2;
    frexpf(clamped, &e2);
    int exp_v = e2 - 1;
    float mant_f = ldexpf(clamped, -exp_v) - 1.f;
    int mant = (int)roundf(mant_f * 8.f);
    if (mant < 0) mant = 0;
    if (mant > 7) { mant = 0; exp_v += 1; }
    int biased = exp_v + 7;
    if (biased < 1) biased = 1;
    if (biased > 15) biased = 15;
    uint8_t byte = ((uint8_t)biased << 3) | (uint8_t)(mant & 0x07);
    return (byte == 0x7F) ? 0x7E : byte;
}

__device__ __forceinline__ float rq_decode_ue4m3(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
}

__device__ __forceinline__ int rq_swizzled_scale_dst(int m, int kb, int k_blocks) {
    int k_tiles = (k_blocks + 3) / 4;
    int m_tile = m / 128;
    int d2 = (m / 32) & 3;
    int d3 = m & 31;
    int k_tile = kb / 4;
    int d5 = kb & 3;
    return ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
}

}

extern __shared__ __nv_bfloat16 rq_smem[];

__global__ void rmsnorm_quantize_nvfp4_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    uint8_t* __restrict__ packed_out,
    uint8_t* __restrict__ scales_out,
    float stored_global,
    float eps,
    int m_padded,
    int m_logical,
    int K
) {
    int row = blockIdx.x;
    if (row >= m_padded) return;
    bool real = (row < m_logical);

    float local = 0.f;
    if (real) {
        const __nv_bfloat16* xrow = x + (size_t)row * K;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            float v = __bfloat162float(xrow[i]);
            local += v * v;
        }
    }
    float sum = rq_block_sum<256>(local);
    float rms = rsqrtf(sum / (float)K + eps);

    if (real) {
        const __nv_bfloat16* xrow = x + (size_t)row * K;
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            float v = __bfloat162float(xrow[i]) * rms * __bfloat162float(weight[i]);
            rq_smem[i] = __float2bfloat16(v);
        }
    } else {
        for (int i = threadIdx.x; i < K; i += blockDim.x) {
            rq_smem[i] = __float2bfloat16(0.f);
        }
    }
    __syncthreads();

    int blocks_per_row = K / 16;
    uint8_t* prow = packed_out + (size_t)row * (K / 2);
    float stored = (stored_global == 0.f || !isfinite(stored_global)) ? 1.f : stored_global;
    for (int kb = threadIdx.x; kb < blocks_per_row; kb += blockDim.x) {
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v = __bfloat162float(rq_smem[kb * 16 + i]);
            vals[i] = v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        float stored_scale = stored * local_scale;
        uint8_t scale_byte = rq_encode_ue4m3(stored_scale);
        float scale_decoded = rq_decode_ue4m3(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);

        uint8_t* pblock = prow + kb * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v_lo = fminf(fmaxf(vals[2 * i]     * inv, -6.f), 6.f);
            float v_hi = fminf(fmaxf(vals[2 * i + 1] * inv, -6.f), 6.f);
            uint8_t lo = rq_encode_e2m1(v_lo);
            uint8_t hi = rq_encode_e2m1(v_hi);
            pblock[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
        }

        int dst = rq_swizzled_scale_dst(row, kb, blocks_per_row);
        scales_out[dst] = scale_byte;
    }
}

extern "C" int nv_kernels_rmsnorm_quantize_nvfp4_bf16(
    void* stream,
    const uint16_t* x_bf16,
    const uint16_t* weight_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    float stored_global,
    float eps,
    int m_padded,
    int m_logical,
    int K
) {
    if (m_padded <= 0 || K <= 0) return 0;
    if (m_logical < 0 || m_logical > m_padded) return -2;
    if ((K % 16) != 0) return -1;
    size_t smem = (size_t)K * sizeof(__nv_bfloat16);
    if (smem > 48 * 1024) return -3;
    cudaStream_t s = (cudaStream_t)stream;
    rmsnorm_quantize_nvfp4_bf16_kernel<<<m_padded, 256, smem, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<const __nv_bfloat16*>(weight_bf16),
        packed_out,
        scales_out_swizzled,
        stored_global,
        eps,
        m_padded,
        m_logical,
        K);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
