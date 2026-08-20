
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>
#include "nv_kernels.h"
#include "nvk_grid.cuh"

template <int K>
__global__ void depthwise_conv1d_silu_bf16_kernel_kfixed(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    int B, int C, int T
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int t = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const __nv_bfloat16* x_row = x + ((size_t)b * C + c) * T;
    const __nv_bfloat16* w_row = w + (size_t)c * K;

    float wv[K];
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        wv[k] = __bfloat162float(w_row[k]);
    }

    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        int src_t = t - (K - 1) + k;
        if (src_t >= 0) {
            acc += __bfloat162float(x_row[src_t]) * wv[k];
        }
    }
    float sig = 1.f / (1.f + expf(-acc));
    y[((size_t)b * C + c) * T + t] = __float2bfloat16(acc * sig);
}

__global__ void depthwise_conv1d_silu_bf16_kernel_generic(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ y,
    int B, int C, int T, int K
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int t = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const __nv_bfloat16* x_row = x + ((size_t)b * C + c) * T;
    const __nv_bfloat16* w_row = w + (size_t)c * K;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        int src_t = t - (K - 1) + k;
        if (src_t >= 0) {
            float xv = __bfloat162float(x_row[src_t]);
            float wv = __bfloat162float(w_row[k]);
            acc += xv * wv;
        }
    }
    float sig = 1.f / (1.f + expf(-acc));
    y[((size_t)b * C + c) * T + t] = __float2bfloat16(acc * sig);
}

extern "C" int nv_kernels_depthwise_conv1d_silu_bf16(
    void* stream,
    const uint16_t* x_bf16,
    const uint16_t* w_bf16,
    uint16_t* y_bf16,
    int B, int C, int T, int K
) {
    if (B <= 0 || C <= 0 || T <= 0 || K <= 0) return 0;
    if (C > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    int tile_t;
    if (T <= 32) tile_t = 32;
    else if (T <= 64) tile_t = 64;
    else if (T <= 128) tile_t = 128;
    else tile_t = 256;
    int tile_count = (T + tile_t - 1) / tile_t;
    if (tile_count > 65535) return NVK_ERR_GRID_AXIS;
    dim3 grid(B, C, tile_count);
    dim3 block(tile_t);

    auto x = reinterpret_cast<const __nv_bfloat16*>(x_bf16);
    auto w = reinterpret_cast<const __nv_bfloat16*>(w_bf16);
    auto y = reinterpret_cast<__nv_bfloat16*>(y_bf16);

    switch (K) {
        case 2:
            depthwise_conv1d_silu_bf16_kernel_kfixed<2>
                <<<grid, block, 0, s>>>(x, w, y, B, C, T);
            break;
        case 3:
            depthwise_conv1d_silu_bf16_kernel_kfixed<3>
                <<<grid, block, 0, s>>>(x, w, y, B, C, T);
            break;
        case 4:
            depthwise_conv1d_silu_bf16_kernel_kfixed<4>
                <<<grid, block, 0, s>>>(x, w, y, B, C, T);
            break;
        case 5:
            depthwise_conv1d_silu_bf16_kernel_kfixed<5>
                <<<grid, block, 0, s>>>(x, w, y, B, C, T);
            break;
        default:
            depthwise_conv1d_silu_bf16_kernel_generic
                <<<grid, block, 0, s>>>(x, w, y, B, C, T, K);
            break;
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
