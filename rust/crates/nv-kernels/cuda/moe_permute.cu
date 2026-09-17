
#include <cuda_runtime.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include "nv_kernels.h"

namespace {

__global__ void moe_permute_count_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ counts,
    int total
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int e = topk_ids[t];
    atomicAdd(&counts[e], 1);
}

__global__ void moe_permute_scan_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ expert_offsets,
    int num_experts
) {
    if (threadIdx.x != 0) return;
    int32_t acc = 0;
    expert_offsets[0] = 0;
    for (int e = 0; e < num_experts; ++e) {
        acc += counts[e];
        expert_offsets[e + 1] = acc;
    }
}

__global__ void moe_permute_scan_padded_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ expert_offsets,
    int num_experts,
    int tile_rows,
    int padded_capacity
) {
    if (threadIdx.x != 0) return;
    int64_t acc = 0;
    expert_offsets[0] = 0;
    for (int e = 0; e < num_experts; ++e) {
        int64_t count = counts[e];
        int64_t padded = ((count + tile_rows - 1) / tile_rows) * tile_rows;
        int64_t next = acc + padded;
        if (next > padded_capacity || next > INT_MAX) {
            atomicOr(&expert_offsets[num_experts + 1], 8);
            next = padded_capacity;
        }
        expert_offsets[e + 1] = (int32_t)next;
        acc = next;
    }
}

__global__ void moe_permute_validate_count_padded_kernel(
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    int32_t* __restrict__ counts,
    int32_t* __restrict__ status,
    int total,
    int k,
    int num_experts
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int e = topk_ids[t];
    if (e < 0 || e >= num_experts) {
        atomicOr(status, 1);
        return;
    }
    if (!isfinite(topk_weights[t])) {
        atomicOr(status, 2);
    }
    int row_start = (t / k) * k;
    for (int prior = row_start; prior < t; ++prior) {
        if (topk_ids[prior] == e) {
            atomicOr(status, 4);
            break;
        }
    }
    atomicAdd(&counts[e], 1);
}

__global__ void moe_permute_assign_padded_kernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ cursors,
    int32_t* __restrict__ permuted_token_idx,
    int32_t* __restrict__ inv_perm,
    int32_t* __restrict__ status,
    int n_tokens,
    int k,
    int num_experts,
    int padded_capacity
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * k;
    if (t >= total || *status != 0) return;
    int e = topk_ids[t];
    if (e < 0 || e >= num_experts) return;
    int slot = atomicAdd(&cursors[e], 1);
    int64_t pos = (int64_t)expert_offsets[e] + slot;
    int64_t end = expert_offsets[e + 1];
    if (pos < 0 || pos >= end || pos >= padded_capacity) {
        atomicOr(status, 8);
        return;
    }
    permuted_token_idx[pos] = t / k;
    inv_perm[t] = (int32_t)pos;
}

__global__ void moe_permute_assign_kernel(
    const int32_t* __restrict__ topk_ids,
    const int32_t* __restrict__ expert_offsets,
    int32_t* __restrict__ cursors,
    int32_t* __restrict__ permuted_token_idx,
    int32_t* __restrict__ inv_perm,
    int n_tokens,
    int k
) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tokens * k;
    if (t >= total) return;
    int n = t / k;
    int e = topk_ids[t];
    int pos = expert_offsets[e] + atomicAdd(&cursors[e], 1);
    permuted_token_idx[pos] = n;
    inv_perm[t] = pos;
}

}

extern "C" int nv_kernels_moe_permute(
    void* stream,
    const int32_t* topk_ids,
    int32_t* permuted_token_idx,
    int32_t* expert_offsets,
    int32_t* inv_perm,
    int32_t* scratch_counts,
    int n_tokens,
    int k,
    int num_experts
) {
    if (n_tokens <= 0 || k <= 0 || num_experts <= 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int total = n_tokens * k;

    cudaError_t err = cudaMemsetAsync(scratch_counts, 0, sizeof(int32_t) * num_experts, s);
    if (err != cudaSuccess) return (int)err;

    const int block = 256;
    int grid = (total + block - 1) / block;
    moe_permute_count_kernel<<<grid, block, 0, s>>>(topk_ids, scratch_counts, total);
    err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    moe_permute_scan_kernel<<<1, 1, 0, s>>>(scratch_counts, expert_offsets, num_experts);
    err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    err = cudaMemsetAsync(scratch_counts, 0, sizeof(int32_t) * num_experts, s);
    if (err != cudaSuccess) return (int)err;
    moe_permute_assign_kernel<<<grid, block, 0, s>>>(
        topk_ids, expert_offsets, scratch_counts,
        permuted_token_idx, inv_perm, n_tokens, k);
    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}

extern "C" int nv_kernels_moe_permute_padded(
    void* stream,
    const int32_t* topk_ids,
    const float* topk_weights,
    int32_t* permuted_token_idx,
    int32_t* expert_offsets,
    int32_t* inv_perm,
    int32_t* scratch_counts,
    int n_tokens,
    int k,
    int num_experts,
    int tile_rows,
    int padded_capacity
) {
    if (n_tokens <= 0 || k <= 0 || num_experts <= 0) return 0;
    if (tile_rows <= 0 || padded_capacity <= 0) return -1;
    if (n_tokens > INT_MAX / k) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int total = n_tokens * k;

    cudaError_t err = cudaMemsetAsync(
        expert_offsets, 0, sizeof(int32_t) * (num_experts + 2), s);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemsetAsync(scratch_counts, 0, sizeof(int32_t) * num_experts, s);
    if (err != cudaSuccess) return (int)err;

    const int block = 256;
    int grid = (total + block - 1) / block;
    moe_permute_validate_count_padded_kernel<<<grid, block, 0, s>>>(
        topk_ids, topk_weights, scratch_counts,
        &expert_offsets[num_experts + 1], total, k, num_experts);
    err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    moe_permute_scan_padded_kernel<<<1, 1, 0, s>>>(
        scratch_counts, expert_offsets, num_experts, tile_rows, padded_capacity);
    err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;

    err = cudaMemsetAsync(
        permuted_token_idx, 0xff, sizeof(int32_t) * padded_capacity, s);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemsetAsync(inv_perm, 0xff, sizeof(int32_t) * total, s);
    if (err != cudaSuccess) return (int)err;
    err = cudaMemsetAsync(scratch_counts, 0, sizeof(int32_t) * num_experts, s);
    if (err != cudaSuccess) return (int)err;
    moe_permute_assign_padded_kernel<<<grid, block, 0, s>>>(
        topk_ids, expert_offsets, scratch_counts,
        permuted_token_idx, inv_perm, &expert_offsets[num_experts + 1],
        n_tokens, k, num_experts, padded_capacity);
    err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : (int)err;
}
