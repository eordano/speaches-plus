
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "nvk_pdl.cuh"

__global__ void rope_bf16_kernel(
    __nv_bfloat16* __restrict__ q,
    __nv_bfloat16* __restrict__ k,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int32_t* __restrict__ positions,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim
) {
    size_t token_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t pair_idx = threadIdx.x;

    size_t half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

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
    cudaStream_t s = (cudaStream_t)stream;
    if (head_dim % 2 != 0) return -2;
    size_t half_dim = head_dim / 2;
    dim3 grid((unsigned)batch, (unsigned)(n_heads + n_kv_heads));
    dim3 block((unsigned)half_dim);
    rope_bf16_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(q),
        reinterpret_cast<__nv_bfloat16*>(k),
        cos_tbl, sin_tbl, positions,
        n_heads, n_kv_heads, head_dim);
    return (int)cudaGetLastError();
}

__global__ void rope_bf16_oop_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int32_t* __restrict__ positions,
    size_t n_heads,
    size_t n_kv_heads,
    size_t head_dim
) {
    NVK_PDL_PROLOG();

    size_t token_idx = blockIdx.x;
    size_t head_idx = blockIdx.y;
    size_t pair_idx = threadIdx.x;

    size_t half_dim = head_dim / 2;
    if (pair_idx >= half_dim) return;

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

    NVK_PDL_EPILOG();
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
    cudaStream_t s = (cudaStream_t)stream;
    if (head_dim % 2 != 0) return -2;
    size_t half_dim = head_dim / 2;
    dim3 grid((unsigned)batch, (unsigned)(n_heads + n_kv_heads));
    dim3 block((unsigned)half_dim);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, block, 0, s);
        cudaLaunchKernelEx(
            &cfg, rope_bf16_oop_kernel,
            reinterpret_cast<const __nv_bfloat16*>(q_in),
            reinterpret_cast<const __nv_bfloat16*>(k_in),
            reinterpret_cast<__nv_bfloat16*>(q_out),
            reinterpret_cast<__nv_bfloat16*>(k_out),
            cos_tbl, sin_tbl, positions,
            n_heads, n_kv_heads, head_dim);
    } else {
        rope_bf16_oop_kernel<<<grid, block, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(q_in),
            reinterpret_cast<const __nv_bfloat16*>(k_in),
            reinterpret_cast<__nv_bfloat16*>(q_out),
            reinterpret_cast<__nv_bfloat16*>(k_out),
            cos_tbl, sin_tbl, positions,
            n_heads, n_kv_heads, head_dim);
    }
    return (int)cudaGetLastError();
}
