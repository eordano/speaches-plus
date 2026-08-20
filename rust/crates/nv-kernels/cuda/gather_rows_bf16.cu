
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "nv_kernels.h"

__global__ void gather_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ src_idx,
    __nv_bfloat16* __restrict__ out,
    int m_total_padded,
    int hidden,
    int n_tokens,
    int row_mul
) {
    int r = blockIdx.x * row_mul;
    if (r >= m_total_padded) return;
    int s = src_idx[r];
    const __nv_bfloat16* src = nullptr;
    if (s >= 0 && s < n_tokens) {
        src = x + (size_t)s * hidden;
    }
    __nv_bfloat16* dst = out + (size_t)r * hidden;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        dst[h] = src ? src[h] : __float2bfloat16(0.f);
    }
}

extern "C" int nv_kernels_gather_rows_bf16(
    void* stream,
    const uint16_t* x_bf16,
    const int32_t* src_idx,
    uint16_t* out_bf16,
    int m_total_padded,
    int hidden,
    int n_tokens
) {
    if (m_total_padded <= 0 || hidden <= 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = 256;
    if (hidden < block) block = ((hidden + 31) / 32) * 32;
    if (block < 32) block = 32;
    gather_rows_bf16_kernel<<<m_total_padded, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        src_idx,
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        m_total_padded, hidden, n_tokens, 1
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_gather_rows_bf16_strided(
    void* stream,
    const uint16_t* x_bf16,
    const int32_t* src_idx,
    uint16_t* out_bf16,
    int m_total_padded,
    int hidden,
    int n_tokens,
    int row_stride
) {
    if (m_total_padded <= 0 || hidden <= 0) return 0;
    if (row_stride <= 0 || (m_total_padded % row_stride) != 0) return -3;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = 256;
    if (hidden < block) block = ((hidden + 31) / 32) * 32;
    if (block < 32) block = 32;
    int n_tiles = m_total_padded / row_stride;
    gather_rows_bf16_kernel<<<n_tiles, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        src_idx,
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        m_total_padded, hidden, n_tokens, row_stride
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
