#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "nv_kernels.h"

__global__ void moe_unpermute_scatter_kernel(
    const __hip_bfloat16* __restrict__ y_sorted,
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
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int tx = 256;
    dim3 grid(n_tokens, (hidden + tx - 1) / tx);
    dim3 block(tx);
    moe_unpermute_scatter_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(y_sorted_bf16),
        topk_weights,
        inv_perm,
        y_acc_f32,
        n_tokens, k, hidden,
        y_sorted_row_stride
    );
    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : (int)e;
}
