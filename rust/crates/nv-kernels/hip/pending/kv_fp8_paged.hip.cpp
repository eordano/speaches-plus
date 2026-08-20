#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

#include "nv_hip_wave.h"

namespace {

constexpr float kFp8E4m3Max = 448.0f;
constexpr int kMaxBlock = 512;
constexpr int kMaxWaves = kMaxBlock / nv_hip::kWave;

__device__ __forceinline__ uint8_t f32_to_e4m3_ocp(float v) {
    uint32_t x = __float_as_uint(v);
    uint32_t sign = (x >> 24) & 0x80u;
    x &= 0x7fffffffu;
    if (x > 0x7f800000u) return (uint8_t)(sign | 0x7fu);
    if (x >= 0x43e00000u) return (uint8_t)(sign | 0x7eu);
    int exp = (int)(x >> 23) - 127;
    uint32_t mant = x & 0x7fffffu;
    uint32_t out;
    if (exp >= -6) {
        uint32_t e = (uint32_t)(exp + 7);
        uint32_t m = mant >> 20;
        uint32_t rem = mant & 0xfffffu;
        if (rem > 0x80000u || (rem == 0x80000u && (m & 1u) != 0u)) {
            m += 1u;
            if (m == 8u) {
                m = 0u;
                e += 1u;
            }
        }
        if (e > 15u || (e == 15u && m >= 7u)) return (uint8_t)(sign | 0x7eu);
        out = (e << 3) | m;
    } else {
        int shift = 20 + (-6 - exp);
        if (shift > 31) {
            out = 0u;
        } else {
            uint32_t full = mant | 0x800000u;
            uint32_t m = full >> shift;
            uint32_t rem = full & ((1u << shift) - 1u);
            uint32_t half = 1u << (shift - 1);
            if (rem > half || (rem == half && (m & 1u) != 0u)) m += 1u;
            out = m;
        }
    }
    return (uint8_t)(sign | out);
}

__device__ __forceinline__ float e4m3_ocp_to_f32(uint8_t b) {
    uint32_t sign = ((uint32_t)b & 0x80u) << 24;
    uint32_t e = ((uint32_t)b >> 3) & 0xfu;
    uint32_t m = (uint32_t)b & 0x7u;
    if (e == 15u && m == 7u) return __uint_as_float(sign | 0x7fc00000u);
    if (e == 0u) {
        float f = (float)m * 0.001953125f;
        return (sign != 0u) ? -f : f;
    }
    return __uint_as_float(sign | ((e + 120u) << 23) | (m << 20));
}

__device__ __forceinline__ float kv_block_max(float v) {
    constexpr int kW = nv_hip::kWave;
    __shared__ float wave_max_s[kMaxWaves];
    const int lane = threadIdx.x & (kW - 1);
    const int wave = threadIdx.x / kW;
    v = nv_hip::wave_max<kW>(v);
    if (lane == 0) wave_max_s[wave] = v;
    __syncthreads();
    if (wave == 0) {
        const int n_waves = (int)((blockDim.x + kW - 1) / kW);
        float s = (lane < n_waves) ? wave_max_s[lane] : 0.0f;
        s = nv_hip::wave_max<kW>(s);
        if (lane == 0) wave_max_s[0] = s;
    }
    __syncthreads();
    return wave_max_s[0];
}

int kv_block_dim(int head_dim) {
    return nv_hip::wave_aligned_block(head_dim, kMaxBlock);
}

__device__ __forceinline__ int paged_slot(
    const int* __restrict__ block_table, int block_size, int logical
) {
    int blk = logical / block_size;
    int off = logical - blk * block_size;
    return block_table[blk] * block_size + off;
}

__global__ void quantize_kv_fp8_paged_kernel(
    const __hip_bfloat16* __restrict__ x_bf16,
    uint8_t* __restrict__ x_fp8_base,
    float* __restrict__ scales_base,
    const int* __restrict__ start_dev,
    const int* __restrict__ block_table,
    int block_size,
    int n_tokens,
    int n_kv,
    int head_dim
) {
    int kv_head = blockIdx.x;
    int token = blockIdx.y;
    if (token >= n_tokens || kv_head >= n_kv) return;

    int start = *start_dev;
    int logical = start + token;
    int slot = paged_slot(block_table, block_size, logical);
    int base_src = (token * n_kv + kv_head) * head_dim;
    int base_dst = (slot * n_kv + kv_head) * head_dim;
    int tid = threadIdx.x;

    float local_max = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = __bfloat162float(x_bf16[base_src + d]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    float amax = kv_block_max(local_max);

    float scale = (amax > 0.0f) ? (amax / kFp8E4m3Max) : 1.0f;
    float inv_scale = (amax > 0.0f) ? (kFp8E4m3Max / amax) : 1.0f;
    if (tid == 0) {
        scales_base[slot * n_kv + kv_head] = scale;
    }
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = __bfloat162float(x_bf16[base_src + d]);
        x_fp8_base[base_dst + d] = f32_to_e4m3_ocp(v * inv_scale);
    }
}

__global__ void dequantize_kv_fp8_paged_kernel(
    const uint8_t* __restrict__ x_fp8_base,
    const float* __restrict__ scales_base,
    __hip_bfloat16* __restrict__ x_bf16_out,
    const int* __restrict__ block_table,
    int block_size,
    int len,
    int n_kv,
    int head_dim
) {
    int kv_head = blockIdx.x;
    int token = blockIdx.y;
    if (token >= len || kv_head >= n_kv) return;

    int slot = paged_slot(block_table, block_size, token);
    int base_src = (slot * n_kv + kv_head) * head_dim;
    int base_dst = (token * n_kv + kv_head) * head_dim;
    float scale = scales_base[slot * n_kv + kv_head];
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = e4m3_ocp_to_f32(x_fp8_base[base_src + d]) * scale;
        x_bf16_out[base_dst + d] = __float2bfloat16(v);
    }
}

__global__ void copy_kv_block_fp8_kernel(
    const uint8_t* __restrict__ fp8_base,
    const float* __restrict__ scales_base,
    uint8_t* __restrict__ fp8_dst_base,
    float* __restrict__ scales_dst_base,
    int src_block,
    int dst_block,
    int block_size,
    int n_kv,
    int head_dim
) {
    int slot_in_block = blockIdx.x;
    int kv_head = blockIdx.y;
    if (slot_in_block >= block_size || kv_head >= n_kv) return;

    int src_slot = src_block * block_size + slot_in_block;
    int dst_slot = dst_block * block_size + slot_in_block;
    int src_base = (src_slot * n_kv + kv_head) * head_dim;
    int dst_base = (dst_slot * n_kv + kv_head) * head_dim;
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        fp8_dst_base[dst_base + d] = fp8_base[src_base + d];
    }
    if (tid == 0) {
        scales_dst_base[dst_slot * n_kv + kv_head] =
            scales_base[src_slot * n_kv + kv_head];
    }
}

}

extern "C" int nv_kernels_quantize_kv_fp8_paged(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* x_fp8_base,
    float* scales_base,
    const int* start_dev,
    const int* block_table,
    int block_size,
    int n_tokens,
    int n_kv,
    int head_dim
) {
    if (n_tokens <= 0 || n_kv <= 0 || head_dim <= 0 || block_size <= 0) return 0;
    hipStream_t s = static_cast<hipStream_t>(stream);
    int block = kv_block_dim(head_dim);
    dim3 grid(static_cast<unsigned>(n_kv), static_cast<unsigned>(n_tokens));
    quantize_kv_fp8_paged_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(x_bf16),
        x_fp8_base, scales_base, start_dev, block_table,
        block_size, n_tokens, n_kv, head_dim
    );
    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_dequantize_kv_fp8_paged(
    void* stream,
    const uint8_t* x_fp8_base,
    const float* scales_base,
    uint16_t* x_bf16_out,
    const int* block_table,
    int block_size,
    int len,
    int n_kv,
    int head_dim
) {
    if (len <= 0 || n_kv <= 0 || head_dim <= 0 || block_size <= 0) return 0;
    hipStream_t s = static_cast<hipStream_t>(stream);
    int block = kv_block_dim(head_dim);
    dim3 grid(static_cast<unsigned>(n_kv), static_cast<unsigned>(len));
    dequantize_kv_fp8_paged_kernel<<<grid, block, 0, s>>>(
        x_fp8_base, scales_base,
        reinterpret_cast<__hip_bfloat16*>(x_bf16_out),
        block_table, block_size, len, n_kv, head_dim
    );
    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_copy_kv_block_fp8(
    void* stream,
    const uint8_t* fp8_base,
    const float* scales_base,
    uint8_t* fp8_dst_base,
    float* scales_dst_base,
    int src_block,
    int dst_block,
    int block_size,
    int n_kv,
    int head_dim
) {
    if (block_size <= 0 || n_kv <= 0 || head_dim <= 0) return 0;
    if (src_block == dst_block) return 0;
    hipStream_t s = static_cast<hipStream_t>(stream);
    int block = kv_block_dim(head_dim);
    dim3 grid(static_cast<unsigned>(block_size), static_cast<unsigned>(n_kv));
    copy_kv_block_fp8_kernel<<<grid, block, 0, s>>>(
        fp8_base, scales_base, fp8_dst_base, scales_dst_base,
        src_block, dst_block, block_size, n_kv, head_dim
    );
    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : static_cast<int>(e);
}
