#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
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
    hipStream_t s = static_cast<hipStream_t>(stream);
    int total = n_tokens * k;

    hipError_t err = hipMemsetAsync(scratch_counts, 0, sizeof(int32_t) * num_experts, s);
    if (err != hipSuccess) return (int)err;

    const int block = 256;
    int grid = (total + block - 1) / block;
    moe_permute_count_kernel<<<grid, block, 0, s>>>(topk_ids, scratch_counts, total);
    err = hipGetLastError();
    if (err != hipSuccess) return (int)err;

    moe_permute_scan_kernel<<<1, 1, 0, s>>>(scratch_counts, expert_offsets, num_experts);
    err = hipGetLastError();
    if (err != hipSuccess) return (int)err;

    err = hipMemsetAsync(scratch_counts, 0, sizeof(int32_t) * num_experts, s);
    if (err != hipSuccess) return (int)err;
    moe_permute_assign_kernel<<<grid, block, 0, s>>>(
        topk_ids, expert_offsets, scratch_counts,
        permuted_token_idx, inv_perm, n_tokens, k);
    err = hipGetLastError();
    return (err == hipSuccess) ? 0 : (int)err;
}
