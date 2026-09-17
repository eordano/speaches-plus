
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
#include "nv_kernels.h"
#include "nvk_grid.cuh"

static __device__ __forceinline__ float conv_weight(
    const __nv_bfloat16* w, const float*, int k, int
) {
    return __bfloat162float(w[k]);
}

static __device__ __forceinline__ float conv_weight(
    const uint8_t* w, const float* scales, int k, int c
) {
    __half_raw raw = __nv_cvt_fp8_to_halfraw(w[k], __NV_E4M3);
    float value;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(value) : "h"(raw.x));
    return __bfloat162float(__float2bfloat16(value * scales[c]));
}

template <int K, typename Weight>
__global__ void depthwise_conv1d_silu_bf16_kernel_kfixed(
    const __nv_bfloat16* __restrict__ x,
    const Weight* __restrict__ w,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ y,
    int B, int C, int T
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int t = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const __nv_bfloat16* x_row = x + ((size_t)b * C + c) * T;
    const Weight* w_row = w + (size_t)c * K;

    float wv[K];
    #pragma unroll
    for (int k = 0; k < K; ++k) {
        wv[k] = conv_weight(w_row, scales, k, c);
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

template <typename Weight>
__global__ void depthwise_conv1d_silu_bf16_kernel_generic(
    const __nv_bfloat16* __restrict__ x,
    const Weight* __restrict__ w,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ y,
    int B, int C, int T, int K
) {
    int b = blockIdx.x;
    int c = blockIdx.y;
    int t = blockIdx.z * blockDim.x + threadIdx.x;
    if (t >= T) return;

    const __nv_bfloat16* x_row = x + ((size_t)b * C + c) * T;
    const Weight* w_row = w + (size_t)c * K;

    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        int src_t = t - (K - 1) + k;
        if (src_t >= 0) {
            float xv = __bfloat162float(x_row[src_t]);
            float wv = conv_weight(w_row, scales, k, c);
            acc += xv * wv;
        }
    }
    float sig = 1.f / (1.f + expf(-acc));
    y[((size_t)b * C + c) * T + t] = __float2bfloat16(acc * sig);
}

template <typename Weight>
static int launch_depthwise_conv1d_silu(
    void* stream,
    const uint16_t* x_bf16,
    const Weight* w,
    const float* scales,
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
    int tile_count = 1 + (T - 1) / tile_t;
    if (tile_count > 65535) return NVK_ERR_GRID_AXIS;
    dim3 grid(B, C, tile_count);
    dim3 block(tile_t);

    auto x = reinterpret_cast<const __nv_bfloat16*>(x_bf16);
    auto y = reinterpret_cast<__nv_bfloat16*>(y_bf16);

    switch (K) {
        case 2:
            depthwise_conv1d_silu_bf16_kernel_kfixed<2>
                <<<grid, block, 0, s>>>(x, w, scales, y, B, C, T);
            break;
        case 3:
            depthwise_conv1d_silu_bf16_kernel_kfixed<3>
                <<<grid, block, 0, s>>>(x, w, scales, y, B, C, T);
            break;
        case 4:
            depthwise_conv1d_silu_bf16_kernel_kfixed<4>
                <<<grid, block, 0, s>>>(x, w, scales, y, B, C, T);
            break;
        case 5:
            depthwise_conv1d_silu_bf16_kernel_kfixed<5>
                <<<grid, block, 0, s>>>(x, w, scales, y, B, C, T);
            break;
        default:
            depthwise_conv1d_silu_bf16_kernel_generic
                <<<grid, block, 0, s>>>(x, w, scales, y, B, C, T, K);
            break;
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_depthwise_conv1d_silu_bf16(
    void* stream,
    const uint16_t* x_bf16,
    const uint16_t* w_bf16,
    uint16_t* y_bf16,
    int B, int C, int T, int K
) {
    return launch_depthwise_conv1d_silu(
        stream, x_bf16, reinterpret_cast<const __nv_bfloat16*>(w_bf16),
        nullptr, y_bf16, B, C, T, K);
}

extern "C" int nv_kernels_depthwise_conv1d_silu_fp8_weights(
    void* stream,
    const uint16_t* x_bf16,
    const uint8_t* w_fp8,
    const float* weight_scales,
    uint16_t* y_bf16,
    int B, int C, int T, int K
) {
    if (B <= 0 || C <= 0 || T <= 0 || K <= 0) return 0;
    if (!stream || !x_bf16 || !w_fp8 || !weight_scales || !y_bf16) return -3;
    if ((uint64_t)B * C > SIZE_MAX / ((uint64_t)T * sizeof(uint16_t))) return -3;
    return launch_depthwise_conv1d_silu(
        stream, x_bf16, w_fp8, weight_scales, y_bf16, B, C, T, K);
}
