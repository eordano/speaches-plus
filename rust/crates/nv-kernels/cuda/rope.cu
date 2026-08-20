#include <cuda_runtime.h>
#include "nv_kernels.h"

__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
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
        float a = q[base + pair_idx];
        float b = q[base + pair_idx + half_dim];
        q[base + pair_idx] = a * c - b * s;
        q[base + pair_idx + half_dim] = a * s + b * c;
    } else if (is_k) {
        size_t kv_head = head_idx - n_heads;
        size_t base = (token_idx * n_kv_heads + kv_head) * head_dim;
        float a = k[base + pair_idx];
        float b = k[base + pair_idx + half_dim];
        k[base + pair_idx] = a * c - b * s;
        k[base + pair_idx + half_dim] = a * s + b * c;
    }
}

extern "C" int nv_kernels_rope_f32(
    void* stream,
    float* q,
    float* k,
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
    rope_kernel<<<grid, block, 0, s>>>(q, k, cos_tbl, sin_tbl, positions,
                                       n_heads, n_kv_heads, head_dim);
    return (int)cudaGetLastError();
}
