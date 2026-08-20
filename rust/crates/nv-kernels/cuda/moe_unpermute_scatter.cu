
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "nv_kernels.h"

__global__ void moe_unpermute_scatter_kernel(
    const __nv_bfloat16* __restrict__ y_sorted,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ inv_perm,
    float* __restrict__ y_acc,
    int n_tokens,
    int k,
    int H,
    int y_sorted_row_stride
) {
    int n = blockIdx.x;
    if (n >= n_tokens) return;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= H) return;

    float acc = 0.f;
    #pragma unroll 1
    for (int s = 0; s < k; ++s) {
        int slot = n * k + s;
        int sorted_row = inv_perm[slot];
        float w = topk_weights[slot];
        float v = __bfloat162float(y_sorted[(size_t)sorted_row * y_sorted_row_stride + h]);
        acc += w * v;
    }
    y_acc[(size_t)n * H + h] = acc;
}

__global__ void moe_unpermute_scatter_tail_kernel(
    const __nv_bfloat16* __restrict__ y_sorted,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ inv_perm,
    const float* __restrict__ shared_f32,
    const __nv_bfloat16* __restrict__ resid,
    __nv_bfloat16* __restrict__ out,
    int n_tokens,
    int k,
    int H,
    int y_sorted_row_stride
) {
    int n = blockIdx.x;
    if (n >= n_tokens) return;
    int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= H) return;

    float acc = 0.f;
    #pragma unroll 1
    for (int s = 0; s < k; ++s) {
        int slot = n * k + s;
        int sorted_row = inv_perm[slot];
        float w = topk_weights[slot];
        float v = __bfloat162float(y_sorted[(size_t)sorted_row * y_sorted_row_stride + h]);
        acc += w * v;
    }
    size_t i = (size_t)n * H + h;
    float t = (acc + shared_f32[i]) * 1.0f;
    __nv_bfloat16 fb = __float2bfloat16(t);
    float rv = __bfloat162float(resid[i]);
    out[i] = __float2bfloat16((rv + __bfloat162float(fb)) * 1.0f);
}

extern "C" int nv_kernels_moe_unpermute_scatter_tail(
    void* stream,
    const uint16_t* y_sorted_bf16,
    const float* topk_weights,
    const int32_t* inv_perm,
    const float* shared_f32,
    const uint16_t* resid_bf16,
    uint16_t* out_bf16,
    int n_tokens,
    int k,
    int hidden,
    int y_sorted_row_stride
) {
    if (n_tokens <= 0 || k <= 0 || hidden <= 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int tx = 256;
    dim3 grid(n_tokens, (hidden + tx - 1) / tx);
    dim3 block(tx);
    moe_unpermute_scatter_tail_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(y_sorted_bf16),
        topk_weights,
        inv_perm,
        shared_f32,
        reinterpret_cast<const __nv_bfloat16*>(resid_bf16),
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        n_tokens, k, hidden,
        y_sorted_row_stride
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_moe_unpermute_scatter(
    void* stream,
    const uint16_t* y_sorted_bf16,
    const float* topk_weights,
    const int32_t* inv_perm,
    float* y_acc_f32,
    int n_tokens,
    int k,
    int hidden,
    int y_sorted_row_stride
) {
    if (n_tokens <= 0 || k <= 0 || hidden <= 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int tx = 256;
    dim3 grid(n_tokens, (hidden + tx - 1) / tx);
    dim3 block(tx);
    moe_unpermute_scatter_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(y_sorted_bf16),
        topk_weights,
        inv_perm,
        y_acc_f32,
        n_tokens, k, hidden,
        y_sorted_row_stride
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
