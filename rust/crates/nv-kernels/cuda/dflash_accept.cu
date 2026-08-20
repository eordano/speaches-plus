
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "nv_kernels.h"
#include "nvk_grid.cuh"

constexpr int kAccBlock = 256;
constexpr int kAccParts = 64;

__global__ void dflash_row_argmax_stage1(
    const float* __restrict__ logits,
    int vocab,
    float* __restrict__ part_val,
    int* __restrict__ part_idx
) {
    __shared__ float sval[kAccBlock];
    __shared__ int sidx[kAccBlock];
    int row = blockIdx.y;
    int tid = threadIdx.x;
    const float* rowp = logits + (size_t)row * vocab;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (int i = blockIdx.x * kAccBlock + tid; i < vocab; i += gridDim.x * kAccBlock) {
        float v = rowp[i];
        if (v > best || (v == best && i < bidx)) {
            best = v;
            bidx = i;
        }
    }
    sval[tid] = best;
    sidx[tid] = bidx;
    __syncthreads();
    for (int s = kAccBlock / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        part_val[(size_t)row * gridDim.x + blockIdx.x] = sval[0];
        part_idx[(size_t)row * gridDim.x + blockIdx.x] = sidx[0];
    }
}

__global__ void dflash_row_argmax_stage2(
    const float* __restrict__ part_val,
    const int* __restrict__ part_idx,
    int nparts,
    uint32_t* __restrict__ row_argmax
) {
    __shared__ float sval[kAccParts];
    __shared__ int sidx[kAccParts];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (tid < nparts) {
        sval[tid] = part_val[(size_t)row * nparts + tid];
        sidx[tid] = part_idx[(size_t)row * nparts + tid];
    } else {
        sval[tid] = -INFINITY;
        sidx[tid] = 0x7fffffff;
    }
    __syncthreads();
    for (int s = kAccParts / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float ov = sval[tid + s];
            int oi = sidx[tid + s];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        row_argmax[row] = (uint32_t)sidx[0];
    }
}

__global__ void dflash_accept_chain_kernel(
    const uint32_t* __restrict__ row_argmax,
    const uint32_t* __restrict__ drafts,
    uint32_t* __restrict__ out,
    int m
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int a = 0;
    while (a < m - 1 && row_argmax[a] == drafts[a]) a++;
    out[0] = (uint32_t)a;
    for (int j = 0; j < a; ++j) out[1 + j] = drafts[j];
    out[1 + a] = row_argmax[a];
}

extern "C" int nv_kernels_dflash_accept_f32(
    void* stream,
    const float* logits,
    const uint32_t* drafts,
    uint32_t* row_argmax,
    uint32_t* out,
    float* part_val,
    int* part_idx,
    int m,
    int vocab
) {
    if (m <= 0 || vocab <= 0) return -2;
    if (m > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid1(kAccParts, m);
    dflash_row_argmax_stage1<<<grid1, kAccBlock, 0, s>>>(logits, vocab, part_val, part_idx);
    dflash_row_argmax_stage2<<<m, kAccParts, 0, s>>>(part_val, part_idx, kAccParts, row_argmax);
    dflash_accept_chain_kernel<<<1, 1, 0, s>>>(row_argmax, drafts, out, m);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_dflash_accept_parts(void) { return kAccParts; }
