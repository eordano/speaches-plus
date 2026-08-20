#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "nvk_grid.cuh"

#define FUSED_MAX_RANK 64
#define FUSED_N_CHUNK 512
#define FUSED_WARPS 16

__global__ void lora_fused_kernel(
    const __nv_bfloat16* __restrict__ x,
    const unsigned long long* __restrict__ a_ptrs,
    const unsigned long long* __restrict__ b_ptrs,
    __nv_bfloat16* __restrict__ y,
    const int32_t* __restrict__ token_indices_sorted,
    const int32_t* __restrict__ num_tokens_per_lora,
    const int32_t* __restrict__ lora_token_start_loc,
    const int32_t* __restrict__ active_lora_ids,
    const int32_t* __restrict__ slice_n,
    const int32_t* __restrict__ slice_start,
    const long long* __restrict__ b_d0_stride,
    int m,
    int rank,
    int k,
    long long a_d0_stride,
    int win_off,
    int win_len,
    int y_row_stride,
    float scale
) {
    int lora_idx = blockIdx.z;
    int lora_id = active_lora_ids[lora_idx];
    if (lora_id == -1) return;

    int group_size = num_tokens_per_lora[lora_idx];
    int pid_m = blockIdx.x % m;
    int pid_n = blockIdx.x / m;
    if (pid_m >= group_size) return;

    int slice_id = blockIdx.y;
    int curr_n = slice_n[slice_id];
    int n_base = pid_n * FUSED_N_CHUNK;
    if (n_base >= curr_n) return;

    int s_start = slice_start[slice_id];
    int chunk_end = n_base + FUSED_N_CHUNK;
    if (chunk_end > curr_n) chunk_end = curr_n;
    if (s_start + chunk_end <= win_off || s_start + n_base >= win_off + win_len) return;

    int row = token_indices_sorted[lora_token_start_loc[lora_idx] + pid_m];

    __shared__ float h[FUSED_MAX_RANK];

    const __nv_bfloat16* a =
        reinterpret_cast<const __nv_bfloat16*>(a_ptrs[slice_id])
        + (long long)lora_id * a_d0_stride;
    const __nv_bfloat16* xr = x + (size_t)row * k;

    int lane = threadIdx.x;
    int wid = threadIdx.y;
    if ((k & 1) == 0) {
        int k2 = k >> 1;
        const __nv_bfloat162* xr2 = reinterpret_cast<const __nv_bfloat162*>(xr);
        for (int r = wid; r < rank; r += blockDim.y) {
            const __nv_bfloat162* ar2 = reinterpret_cast<const __nv_bfloat162*>(a + (long long)r * k);
            float acc = 0.f;
            for (int kk = lane; kk < k2; kk += 32) {
                float2 xv = __bfloat1622float2(xr2[kk]);
                float2 av = __bfloat1622float2(ar2[kk]);
                acc = fmaf(xv.x, av.x, acc);
                acc = fmaf(xv.y, av.y, acc);
            }
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) h[r] = acc * scale;
        }
    } else {
        for (int r = wid; r < rank; r += blockDim.y) {
            const __nv_bfloat16* ar = a + (long long)r * k;
            float acc = 0.f;
            for (int kk = lane; kk < k; kk += 32) {
                acc += __bfloat162float(xr[kk]) * __bfloat162float(ar[kk]);
            }
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) h[r] = acc * scale;
        }
    }
    __syncthreads();

    int tid = wid * blockDim.x + lane;
    int nl = n_base + tid;
    if (nl >= curr_n) return;
    int col = s_start + nl;
    if (col < win_off || col >= win_off + win_len) return;

    const __nv_bfloat16* brow =
        reinterpret_cast<const __nv_bfloat16*>(b_ptrs[slice_id])
        + (long long)lora_id * b_d0_stride[slice_id]
        + (long long)nl * rank;
    float acc = 0.f;
    if ((rank & 1) == 0) {
        const __nv_bfloat162* brow2 = reinterpret_cast<const __nv_bfloat162*>(brow);
        int r2 = rank >> 1;
        for (int r = 0; r < r2; ++r) {
            float2 bv = __bfloat1622float2(brow2[r]);
            acc = fmaf(h[2 * r], bv.x, acc);
            acc = fmaf(h[2 * r + 1], bv.y, acc);
        }
    } else {
        for (int r = 0; r < rank; ++r) {
            acc += h[r] * __bfloat162float(brow[r]);
        }
    }
    __nv_bfloat16* yp = y + (size_t)row * y_row_stride + (col - win_off);
    *yp = __float2bfloat16(__bfloat162float(*yp) + acc);
}

extern "C" int nv_kernels_lora_fused(
    void* stream,
    const uint16_t* x_bf16,
    const unsigned long long* a_ptrs,
    const unsigned long long* b_ptrs,
    uint16_t* y_bf16,
    const int32_t* token_indices_sorted,
    const int32_t* num_tokens_per_lora,
    const int32_t* lora_token_start_loc,
    const int32_t* active_lora_ids,
    const int32_t* slice_n,
    const int32_t* slice_start,
    const long long* b_d0_stride,
    int m,
    int rank,
    int k,
    int max_n,
    int n_slices,
    int grid_loras,
    long long a_d0_stride,
    int win_off,
    int win_len,
    int y_row_stride,
    float scale
) {
    if (m <= 0 || rank <= 0 || k <= 0 || max_n <= 0 || n_slices <= 0 || grid_loras <= 0)
        return -2;
    if (rank > FUSED_MAX_RANK) return -3;
    if (win_len <= 0 || win_off < 0) return -4;
    if (n_slices > 65535) return NVK_ERR_GRID_AXIS;
    if (grid_loras > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int cta_n_num = (max_n + FUSED_N_CHUNK - 1) / FUSED_N_CHUNK;
    dim3 grid((unsigned)(m * cta_n_num), (unsigned)n_slices, (unsigned)grid_loras);
    dim3 block(32, FUSED_WARPS, 1);
    lora_fused_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        a_ptrs,
        b_ptrs,
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        token_indices_sorted,
        num_tokens_per_lora,
        lora_token_start_loc,
        active_lora_ids,
        slice_n,
        slice_start,
        b_d0_stride,
        m, rank, k, a_d0_stride, win_off, win_len, y_row_stride, scale
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
