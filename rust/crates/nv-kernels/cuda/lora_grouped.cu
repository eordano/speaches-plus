#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "nvk_grid.cuh"

#define LORA_BLOCK_M 16
#define LORA_BLOCK_N 16

__global__ void lora_shrink_kernel(
    const __nv_bfloat16* __restrict__ x,
    const unsigned long long* __restrict__ a_ptrs,
    float* __restrict__ buffer,
    const int32_t* __restrict__ token_indices_sorted,
    const int32_t* __restrict__ num_tokens_per_lora,
    const int32_t* __restrict__ lora_token_start_loc,
    const int32_t* __restrict__ active_lora_ids,
    int m,
    int rank,
    int k,
    long long a_d0_stride,
    float scale
) {
    int lora_idx = blockIdx.z;
    int lora_id = active_lora_ids[lora_idx];
    if (lora_id == -1) return;

    int group_size = num_tokens_per_lora[lora_idx];
    int cta_m_num = (m + LORA_BLOCK_M - 1) / LORA_BLOCK_M;
    int pid_m = blockIdx.x % cta_m_num;
    int pid_n = blockIdx.x / cta_m_num;
    int m_offset = pid_m * LORA_BLOCK_M;
    if (m_offset >= group_size) return;

    int mi = m_offset + threadIdx.y;
    int n = pid_n * LORA_BLOCK_N + threadIdx.x;
    if (mi >= group_size || n >= rank) return;

    int slice_id = blockIdx.y;
    int row = token_indices_sorted[lora_token_start_loc[lora_idx] + mi];

    const __nv_bfloat16* a =
        reinterpret_cast<const __nv_bfloat16*>(a_ptrs[slice_id])
        + (long long)lora_id * a_d0_stride
        + (long long)n * k;
    const __nv_bfloat16* xr = x + (size_t)row * k;

    float acc = 0.f;
    for (int kk = 0; kk < k; ++kk) {
        acc += __bfloat162float(xr[kk]) * __bfloat162float(a[kk]);
    }

    buffer[(size_t)slice_id * m * rank + (size_t)row * rank + n] = acc * scale;
}

__global__ void lora_expand_kernel(
    const float* __restrict__ buffer,
    const unsigned long long* __restrict__ b_ptrs,
    __nv_bfloat16* __restrict__ y,
    const int32_t* __restrict__ token_indices_sorted,
    const int32_t* __restrict__ num_tokens_per_lora,
    const int32_t* __restrict__ lora_token_start_loc,
    const int32_t* __restrict__ active_lora_ids,
    const int32_t* __restrict__ slice_n,
    const int32_t* __restrict__ slice_start,
    int m,
    int rank,
    int y_row_stride
) {
    int lora_idx = blockIdx.z;
    int lora_id = active_lora_ids[lora_idx];
    if (lora_id == -1) return;

    int group_size = num_tokens_per_lora[lora_idx];
    int cta_m_num = (m + LORA_BLOCK_M - 1) / LORA_BLOCK_M;
    int pid_m = blockIdx.x % cta_m_num;
    int pid_n = blockIdx.x / cta_m_num;
    int m_offset = pid_m * LORA_BLOCK_M;
    if (m_offset >= group_size) return;

    int slice_id = blockIdx.y;
    int curr_n = slice_n[slice_id];
    if (pid_n * LORA_BLOCK_N >= curr_n) return;

    int mi = m_offset + threadIdx.y;
    int n = pid_n * LORA_BLOCK_N + threadIdx.x;
    if (mi >= group_size || n >= curr_n) return;

    int row = token_indices_sorted[lora_token_start_loc[lora_idx] + mi];

    const __nv_bfloat16* b =
        reinterpret_cast<const __nv_bfloat16*>(b_ptrs[slice_id])
        + (long long)lora_id * curr_n * rank
        + (long long)n * rank;
    const float* br = buffer + (size_t)slice_id * m * rank + (size_t)row * rank;

    float acc = 0.f;
    for (int r = 0; r < rank; ++r) {
        acc += br[r] * __bfloat162float(b[r]);
    }

    __nv_bfloat16* yp = y + (size_t)row * y_row_stride + slice_start[slice_id] + n;
    *yp = __float2bfloat16(__bfloat162float(*yp) + acc);
}

extern "C" int nv_kernels_lora_shrink(
    void* stream,
    const uint16_t* x_bf16,
    const unsigned long long* a_ptrs,
    float* buffer,
    const int32_t* token_lora_mapping,
    const int32_t* token_indices_sorted,
    const int32_t* num_tokens_per_lora,
    const int32_t* lora_token_start_loc,
    const int32_t* active_lora_ids,
    int m,
    int rank,
    int k,
    int n_slices,
    int grid_loras,
    long long a_d0_stride,
    float scale
) {
    (void)token_lora_mapping;
    if (m <= 0 || rank <= 0 || k <= 0 || n_slices <= 0 || grid_loras <= 0) return -2;
    if (n_slices > 65535) return NVK_ERR_GRID_AXIS;
    if (grid_loras > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    size_t buf_bytes = (size_t)n_slices * m * rank * sizeof(float);
    cudaError_t e = cudaMemsetAsync(buffer, 0, buf_bytes, s);
    if (e != cudaSuccess) return (int)e;
    int cta_m_num = (m + LORA_BLOCK_M - 1) / LORA_BLOCK_M;
    int cta_n_num = (rank + LORA_BLOCK_N - 1) / LORA_BLOCK_N;
    dim3 grid(cta_m_num * cta_n_num, n_slices, grid_loras);
    dim3 block(LORA_BLOCK_N, LORA_BLOCK_M, 1);
    lora_shrink_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        a_ptrs,
        buffer,
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        m, rank, k, a_d0_stride, scale
    );
    e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_lora_expand(
    void* stream,
    const float* buffer,
    const unsigned long long* b_ptrs,
    uint16_t* y_bf16,
    const int32_t* token_lora_mapping,
    const int32_t* token_indices_sorted,
    const int32_t* num_tokens_per_lora,
    const int32_t* lora_token_start_loc,
    const int32_t* active_lora_ids,
    const int32_t* slice_n,
    const int32_t* slice_start,
    int m,
    int rank,
    int max_n,
    int n_slices,
    int grid_loras,
    int y_row_stride
) {
    (void)token_lora_mapping;
    if (m <= 0 || rank <= 0 || max_n <= 0 || n_slices <= 0 || grid_loras <= 0) return -2;
    if (n_slices > 65535) return NVK_ERR_GRID_AXIS;
    if (grid_loras > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int cta_m_num = (m + LORA_BLOCK_M - 1) / LORA_BLOCK_M;
    int cta_n_num = (max_n + LORA_BLOCK_N - 1) / LORA_BLOCK_N;
    dim3 grid(cta_m_num * cta_n_num, n_slices, grid_loras);
    dim3 block(LORA_BLOCK_N, LORA_BLOCK_M, 1);
    lora_expand_kernel<<<grid, block, 0, s>>>(
        buffer,
        b_ptrs,
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        slice_n,
        slice_start,
        m, rank, y_row_stride
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
