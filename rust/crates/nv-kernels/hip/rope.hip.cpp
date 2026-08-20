#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include "nv_kernels.h"
#include "nv_hip_wave.h"

__global__ void rope_kernel(
    float* __restrict__ q,
    float* __restrict__ k,
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
    rope_kernel<<<grid, dim3((unsigned)block), 0, s>>>(
        q, k, cos_tbl, sin_tbl, positions,
        n_heads, n_kv_heads, head_dim,
        (unsigned)half_dim, (unsigned)per_token);
    return (int)hipGetLastError();
}
