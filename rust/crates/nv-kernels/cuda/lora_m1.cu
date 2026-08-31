#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "nvk_grid.cuh"

#define LORA_M1_MAX_RANK 64
#define LORA_M1_SHRINK_WARPS 8
#define LORA_M1_EXPAND_BLOCK 256

__global__ void lora_m1_shrink_tiles_kernel(
    const __nv_bfloat16* __restrict__ x,
    const unsigned long long* __restrict__ a_ptrs,
    float* __restrict__ partials,
    int rank,
    int k,
    int k_tile,
    int n_tiles
) {
    int slice_id = blockIdx.y;
    int tile = blockIdx.x;
    int k0 = tile * k_tile;
    if (k0 >= k) {
        if (threadIdx.x == 0) {
            for (int r = threadIdx.y; r < rank; r += blockDim.y) {
                partials[((size_t)slice_id * n_tiles + tile) * rank + r] = 0.f;
            }
        }
        return;
    }
    int k1 = k0 + k_tile;
    if (k1 > k) k1 = k;
    const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(a_ptrs[slice_id]);
    int lane = threadIdx.x;
    for (int r = threadIdx.y; r < rank; r += blockDim.y) {
        const __nv_bfloat16* ar = a + (long long)r * k;
        float acc = 0.f;
        if (((k0 | k1) & 1) == 0) {
            const __nv_bfloat162* xr2 = reinterpret_cast<const __nv_bfloat162*>(x + k0);
            const __nv_bfloat162* ar2 = reinterpret_cast<const __nv_bfloat162*>(ar + k0);
            int span2 = (k1 - k0) >> 1;
            for (int kk = lane; kk < span2; kk += 32) {
                float2 xv = __bfloat1622float2(xr2[kk]);
                float2 av = __bfloat1622float2(ar2[kk]);
                acc = fmaf(xv.x, av.x, acc);
                acc = fmaf(xv.y, av.y, acc);
            }
        } else {
            for (int kk = k0 + lane; kk < k1; kk += 32) {
                acc += __bfloat162float(x[kk]) * __bfloat162float(ar[kk]);
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) {
            partials[((size_t)slice_id * n_tiles + tile) * rank + r] = acc;
        }
    }
}

__global__ void lora_m1_expand_tiles_kernel(
    const float* __restrict__ partials,
    const unsigned long long* __restrict__ b_ptrs,
    __nv_bfloat16* __restrict__ y,
    const int32_t* __restrict__ slice_n,
    const int32_t* __restrict__ slice_start,
    int rank,
    int n_tiles,
    int win_off,
    int win_len
) {
    int slice_id = blockIdx.y;
    __shared__ float h[LORA_M1_MAX_RANK];
    int t = threadIdx.x;
    if (t < rank) {
        float s = 0.f;
        for (int tile = 0; tile < n_tiles; ++tile) {
            s += partials[((size_t)slice_id * n_tiles + tile) * rank + t];
        }
        h[t] = s;
    }
    __syncthreads();
    int nl = blockIdx.x * LORA_M1_EXPAND_BLOCK + t;
    if (nl >= slice_n[slice_id]) return;
    int col = slice_start[slice_id] + nl;
    if (col < win_off || col >= win_off + win_len) return;
    const __nv_bfloat16* brow =
        reinterpret_cast<const __nv_bfloat16*>(b_ptrs[slice_id]) + (long long)nl * rank;
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
    __nv_bfloat16* yp = y + (col - win_off);
    *yp = __float2bfloat16(__bfloat162float(*yp) + acc);
}

extern "C" int nv_kernels_lora_m1_shrink_tiles(
    void* stream,
    const uint16_t* x_bf16,
    const unsigned long long* a_ptrs,
    float* partials,
    int n_slices,
    int rank,
    int k,
    int k_tile,
    int n_tiles
) {
    if (n_slices <= 0 || rank <= 0 || k <= 0 || k_tile <= 0 || n_tiles <= 0) return -2;
    if (rank > LORA_M1_MAX_RANK) return -3;
    if ((long long)k_tile * n_tiles < k) return -4;
    if (n_slices > 65535 || n_tiles > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)n_tiles, (unsigned)n_slices, 1);
    dim3 block(32, LORA_M1_SHRINK_WARPS, 1);
    lora_m1_shrink_tiles_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        a_ptrs,
        partials,
        rank,
        k,
        k_tile,
        n_tiles
    );
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_lora_m1_expand_tiles(
    void* stream,
    const float* partials,
    const unsigned long long* b_ptrs,
    uint16_t* y_bf16,
    const int32_t* slice_n,
    const int32_t* slice_start,
    int n_slices,
    int rank,
    int n_tiles,
    int max_n,
    int win_off,
    int win_len
) {
    if (n_slices <= 0 || rank <= 0 || n_tiles <= 0 || max_n <= 0) return -2;
    if (rank > LORA_M1_MAX_RANK) return -3;
    if (win_len <= 0 || win_off < 0) return -4;
    if (n_slices > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int chunks = (max_n + LORA_M1_EXPAND_BLOCK - 1) / LORA_M1_EXPAND_BLOCK;
    dim3 grid((unsigned)chunks, (unsigned)n_slices, 1);
    dim3 block(LORA_M1_EXPAND_BLOCK, 1, 1);
    lora_m1_expand_tiles_kernel<<<grid, block, 0, s>>>(
        partials,
        b_ptrs,
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        slice_n,
        slice_start,
        rank,
        n_tiles,
        win_off,
        win_len
    );
    return (int)cudaGetLastError();
}
