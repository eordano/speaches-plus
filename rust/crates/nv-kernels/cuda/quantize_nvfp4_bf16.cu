
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

#include "nv_kernels.h"

namespace {

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

#define NV_UE4M3_MIN_NORMAL      0.015625f
#define NV_UE4M3_SUBNORMAL_STEP  0.001953125f

__device__ __forceinline__ uint8_t encode_ue4m3_dev(float scale) {
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

__device__ __forceinline__ float decode_ue4m3_dev(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
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

}

__global__ void quantize_nvfp4_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ packed_out,
    uint8_t* __restrict__ scales_out,
    float stored_global,
    int m_padded,
    int m_logical,
    int K
) {
    int row = blockIdx.x;
    if (row >= m_padded) return;
    int blocks_per_row = K / 16;
    bool row_in_range = (row < m_logical);
    const __nv_bfloat16* xrow = row_in_range ? (x + (size_t)row * K) : nullptr;
    uint8_t* prow = packed_out + (size_t)row * (K / 2);

    float stored = (stored_global == 0.f || !isfinite(stored_global)) ? 1.f : stored_global;

    for (int kb = threadIdx.x; kb < blocks_per_row; kb += blockDim.x) {
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v = row_in_range ? __bfloat162float(xrow[kb * 16 + i]) : 0.f;
            vals[i] = v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        float stored_scale = stored * local_scale;
        uint8_t scale_byte = encode_ue4m3_dev(stored_scale);
        float scale_decoded = decode_ue4m3_dev(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);

        uint8_t* pblock = prow + kb * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v_lo = fminf(fmaxf(vals[2*i]     * inv, -6.f), 6.f);
            float v_hi = fminf(fmaxf(vals[2*i + 1] * inv, -6.f), 6.f);
            uint8_t lo = encode_e2m1_dev(v_lo);
            uint8_t hi = encode_e2m1_dev(v_hi);
            pblock[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
        }

        int dst = swizzled_scale_dst(row, kb, blocks_per_row);
        scales_out[dst] = scale_byte;
    }
}

__global__ void quantize_nvfp4_bf16_per_expert_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ packed_out,
    uint8_t* __restrict__ scales_out,
    const float* __restrict__ stored_globals,
    int m_per_expert,
    int m_total,
    int K,
    int row_mul
) {
    int row = blockIdx.x * row_mul;
    if (row >= m_total) return;
    int expert_idx = row / m_per_expert;
    float stored = stored_globals[expert_idx];
    if (stored == 0.f || !isfinite(stored)) stored = 1.f;
    int blocks_per_row = K / 16;
    const __nv_bfloat16* xrow = x + (size_t)row * K;
    uint8_t* prow = packed_out + (size_t)row * (K / 2);

    for (int kb = threadIdx.x; kb < blocks_per_row; kb += blockDim.x) {
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v = __bfloat162float(xrow[kb * 16 + i]);
            vals[i] = v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        float stored_scale = stored * local_scale;
        uint8_t scale_byte = encode_ue4m3_dev(stored_scale);
        float scale_decoded = decode_ue4m3_dev(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);

        uint8_t* pblock = prow + kb * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v_lo = fminf(fmaxf(vals[2*i]     * inv, -6.f), 6.f);
            float v_hi = fminf(fmaxf(vals[2*i + 1] * inv, -6.f), 6.f);
            uint8_t lo = encode_e2m1_dev(v_lo);
            uint8_t hi = encode_e2m1_dev(v_hi);
            pblock[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
        }

        int dst = swizzled_scale_dst(row, kb, blocks_per_row);
        scales_out[dst] = scale_byte;
    }
}

__global__ void silu_mul_quantize_nvfp4_bf16_per_expert_kernel(
    const __nv_bfloat16* __restrict__ y_gate,
    const __nv_bfloat16* __restrict__ y_up,
    uint8_t* __restrict__ packed_out,
    uint8_t* __restrict__ scales_out,
    const float* __restrict__ stored_globals,
    int m_per_expert,
    int m_total,
    int K,
    int row_mul
) {
    int row = blockIdx.x * row_mul;
    if (row >= m_total) return;
    int expert_idx = row / m_per_expert;
    float stored = stored_globals[expert_idx];
    if (stored == 0.f || !isfinite(stored)) stored = 1.f;
    int blocks_per_row = K / 16;
    const __nv_bfloat16* g_row = y_gate + (size_t)row * K;
    const __nv_bfloat16* u_row = y_up   + (size_t)row * K;
    uint8_t* prow = packed_out + (size_t)row * (K / 2);

    for (int kb = threadIdx.x; kb < blocks_per_row; kb += blockDim.x) {
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float g = __bfloat162float(g_row[kb * 16 + i]);
            float u = __bfloat162float(u_row[kb * 16 + i]);
            float silu_g = g / (1.f + expf(-g));
            float v = silu_g * u;
            vals[i] = v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        float stored_scale = stored * local_scale;
        uint8_t scale_byte = encode_ue4m3_dev(stored_scale);
        float scale_decoded = decode_ue4m3_dev(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);

        uint8_t* pblock = prow + kb * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v_lo = fminf(fmaxf(vals[2*i]     * inv, -6.f), 6.f);
            float v_hi = fminf(fmaxf(vals[2*i + 1] * inv, -6.f), 6.f);
            uint8_t lo = encode_e2m1_dev(v_lo);
            uint8_t hi = encode_e2m1_dev(v_hi);
            pblock[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
        }

        int dst = swizzled_scale_dst(row, kb, blocks_per_row);
        scales_out[dst] = scale_byte;
    }
}

extern "C" int nv_kernels_silu_mul_quantize_nvfp4_bf16_per_expert(
    void* stream,
    const uint16_t* y_gate_bf16,
    const uint16_t* y_up_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    const float* stored_globals,
    int m_per_expert,
    int m_total,
    int K
) {
    if (m_total <= 0 || K <= 0 || m_per_expert <= 0) return 0;
    if ((K % 16) != 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    silu_mul_quantize_nvfp4_bf16_per_expert_kernel<<<m_total, 256, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(y_gate_bf16),
        reinterpret_cast<const __nv_bfloat16*>(y_up_bf16),
        packed_out, scales_out_swizzled,
        stored_globals, m_per_expert, m_total, K, 1
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_silu_mul_quantize_nvfp4_bf16_per_expert_strided(
    void* stream,
    const uint16_t* y_gate_bf16,
    const uint16_t* y_up_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    const float* stored_globals,
    int m_per_expert,
    int m_total,
    int K
) {
    if (m_total <= 0 || K <= 0 || m_per_expert <= 0) return 0;
    if ((K % 16) != 0) return -1;
    if ((m_total % m_per_expert) != 0) return -3;
    int n_tiles = m_total / m_per_expert;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    silu_mul_quantize_nvfp4_bf16_per_expert_kernel<<<n_tiles, 256, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(y_gate_bf16),
        reinterpret_cast<const __nv_bfloat16*>(y_up_bf16),
        packed_out, scales_out_swizzled,
        stored_globals, m_per_expert, m_total, K, m_per_expert
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_quantize_nvfp4_bf16_per_expert(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    const float* stored_globals,
    int m_per_expert,
    int m_total,
    int K
) {
    if (m_total <= 0 || K <= 0 || m_per_expert <= 0) return 0;
    if ((K % 16) != 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    quantize_nvfp4_bf16_per_expert_kernel<<<m_total, 256, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        packed_out,
        scales_out_swizzled,
        stored_globals,
        m_per_expert, m_total, K, 1
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_quantize_nvfp4_bf16_per_expert_strided(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    const float* stored_globals,
    int m_per_expert,
    int m_total,
    int K
) {
    if (m_total <= 0 || K <= 0 || m_per_expert <= 0) return 0;
    if ((K % 16) != 0) return -1;
    if ((m_total % m_per_expert) != 0) return -3;
    int n_tiles = m_total / m_per_expert;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    quantize_nvfp4_bf16_per_expert_kernel<<<n_tiles, 256, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        packed_out,
        scales_out_swizzled,
        stored_globals,
        m_per_expert, m_total, K, m_per_expert
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_quantize_nvfp4_bf16_rows(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    float stored_global,
    int m_rows,
    int K
) {
    if (m_rows <= 0 || K <= 0) return 0;
    if ((K % 16) != 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(m_rows);
    dim3 block(256);
    quantize_nvfp4_bf16_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        packed_out,
        scales_out_swizzled,
        stored_global,
        m_rows, m_rows, K
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_quantize_nvfp4_bf16(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* packed_out,
    uint8_t* scales_out_swizzled,
    float stored_global,
    int m_padded,
    int m_logical,
    int K
) {
    if (m_padded <= 0 || K <= 0) return 0;
    if (m_logical < 0 || m_logical > m_padded) return -2;
    if ((K % 16) != 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(m_padded);
    dim3 block(256);
    quantize_nvfp4_bf16_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        packed_out,
        scales_out_swizzled,
        stored_global,
        m_padded, m_logical, K
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
