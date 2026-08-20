#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "nv_hip_wave.h"

__global__ void rope_bf16_kernel(
    __hip_bfloat16* __restrict__ q,
    __hip_bfloat16* __restrict__ k,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int32_t* __restrict__ positions,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    unsigned half_dim_u,
    unsigned per_token
) {
    unsigned lin = blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= per_token) return;

    unsigned head_u = lin / half_dim_u;
    unsigned pair_u = lin - head_u * half_dim_u;

    size_t token_idx = blockIdx.y;
    size_t head_idx = head_u;
    size_t pair_idx = pair_u;
    size_t half_dim = head_dim / 2;

    int32_t pos = positions[token_idx];
    const float* cos_row = cos_tbl + (size_t)pos * half_dim;
    const float* sin_row = sin_tbl + (size_t)pos * half_dim;

    float c = cos_row[pair_idx];
    float s = sin_row[pair_idx];

    bool is_q = head_idx < n_heads;
    bool is_k = (head_idx >= n_heads) && (head_idx < n_heads + n_kv_heads);

    if (is_q) {
        size_t base = (token_idx * n_heads + head_idx) * head_dim;
        float a = __bfloat162float(q[base + pair_idx]);
        float b = __bfloat162float(q[base + pair_idx + half_dim]);
        q[base + pair_idx]            = __float2bfloat16(a * c - b * s);
        q[base + pair_idx + half_dim] = __float2bfloat16(a * s + b * c);
    } else if (is_k) {
        size_t kv_head = head_idx - n_heads;
        size_t base = (token_idx * n_kv_heads + kv_head) * head_dim;
        float a = __bfloat162float(k[base + pair_idx]);
        float b = __bfloat162float(k[base + pair_idx + half_dim]);
        k[base + pair_idx]            = __float2bfloat16(a * c - b * s);
        k[base + pair_idx + half_dim] = __float2bfloat16(a * s + b * c);
    }
}

extern "C" int nv_kernels_rope_bf16(
    void* stream,
    uint16_t* q,
    uint16_t* k,
    const float* cos_tbl,
    const float* sin_tbl,
    const int32_t* positions,
    size_t batch,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim
) {
    hipStream_t s = (hipStream_t)stream;
    if (head_dim % 2 != 0) return -2;
    size_t half_dim = head_dim / 2;
    size_t n_all = n_heads + n_kv_heads;
    if (batch == 0 || n_all == 0 || half_dim == 0) return 0;
    size_t per_token = n_all * half_dim;
    if (per_token > 0xffffffffull) return -3;

    int block = nv_hip::wave_aligned_block(
        per_token < 256 ? (int)per_token : 256, 256);
    unsigned grid_x =
        (unsigned)((per_token + (size_t)block - 1) / (size_t)block);
    dim3 grid(grid_x, (unsigned)batch);
    rope_bf16_kernel<<<grid, dim3((unsigned)block), 0, s>>>(
        reinterpret_cast<__hip_bfloat16*>(q),
        reinterpret_cast<__hip_bfloat16*>(k),
        cos_tbl, sin_tbl, positions,
        n_heads, n_kv_heads, head_dim,
        (unsigned)half_dim, (unsigned)per_token);
    return (int)hipGetLastError();
}

__global__ void rope_bf16_oop_kernel(
    const __hip_bfloat16* __restrict__ q_in,
    const __hip_bfloat16* __restrict__ k_in,
    __hip_bfloat16* __restrict__ q_out,
    __hip_bfloat16* __restrict__ k_out,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int32_t* __restrict__ positions,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim,
    unsigned half_dim_u,
    unsigned per_token
) {
    unsigned lin = blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= per_token) return;

    unsigned head_u = lin / half_dim_u;
    unsigned pair_u = lin - head_u * half_dim_u;

    size_t token_idx = blockIdx.y;
    size_t head_idx = head_u;
    size_t pair_idx = pair_u;
    size_t half_dim = head_dim / 2;

    int32_t pos = positions[token_idx];
    const float* cos_row = cos_tbl + (size_t)pos * half_dim;
    const float* sin_row = sin_tbl + (size_t)pos * half_dim;

    float c = cos_row[pair_idx];
    float s = sin_row[pair_idx];

    bool is_q = head_idx < n_heads;
    bool is_k = (head_idx >= n_heads) && (head_idx < n_heads + n_kv_heads);

    if (is_q) {
        size_t base = (token_idx * n_heads + head_idx) * head_dim;
        float a = __bfloat162float(q_in[base + pair_idx]);
        float b = __bfloat162float(q_in[base + pair_idx + half_dim]);
        q_out[base + pair_idx]            = __float2bfloat16(a * c - b * s);
        q_out[base + pair_idx + half_dim] = __float2bfloat16(a * s + b * c);
    } else if (is_k) {
        size_t kv_head = head_idx - n_heads;
        size_t base = (token_idx * n_kv_heads + kv_head) * head_dim;
        float a = __bfloat162float(k_in[base + pair_idx]);
        float b = __bfloat162float(k_in[base + pair_idx + half_dim]);
        k_out[base + pair_idx]            = __float2bfloat16(a * c - b * s);
        k_out[base + pair_idx + half_dim] = __float2bfloat16(a * s + b * c);
    }
}

extern "C" int nv_kernels_rope_bf16_oop(
    void* stream,
    const uint16_t* q_in,
    const uint16_t* k_in,
    uint16_t* q_out,
    uint16_t* k_out,
    const float* cos_tbl,
    const float* sin_tbl,
    const int32_t* positions,
    size_t batch,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim
) {
    hipStream_t s = (hipStream_t)stream;
    if (head_dim % 2 != 0) return -2;
    size_t half_dim = head_dim / 2;
    size_t n_all = n_heads + n_kv_heads;
    if (batch == 0 || n_all == 0 || half_dim == 0) return 0;
    size_t per_token = n_all * half_dim;
    if (per_token > 0xffffffffull) return -3;

    int block = nv_hip::wave_aligned_block(
        per_token < 256 ? (int)per_token : 256, 256);
    unsigned grid_x =
        (unsigned)((per_token + (size_t)block - 1) / (size_t)block);
    dim3 grid(grid_x, (unsigned)batch);
    rope_bf16_oop_kernel<<<grid, dim3((unsigned)block), 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(q_in),
        reinterpret_cast<const __hip_bfloat16*>(k_in),
        reinterpret_cast<__hip_bfloat16*>(q_out),
        reinterpret_cast<__hip_bfloat16*>(k_out),
        cos_tbl, sin_tbl, positions,
        n_heads, n_kv_heads, head_dim,
        (unsigned)half_dim, (unsigned)per_token);
    return (int)hipGetLastError();
}
