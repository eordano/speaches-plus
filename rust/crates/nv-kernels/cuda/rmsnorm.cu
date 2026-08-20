#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "nv_kernels.h"
#include "nvk_pdl.cuh"

template <typename T>
__device__ inline float to_f32(T x);

template <>
__device__ inline float to_f32<float>(float x) { return x; }

template <>
__device__ inline float to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T from_f32(float x);

template <>
__device__ inline float from_f32<float>(float x) { return x; }

template <>
__device__ inline __nv_bfloat16 from_f32<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

template <int BLOCK>
__device__ inline float block_sum(float v) {
    constexpr int kWarp = 32;
    constexpr int kWarps = BLOCK / kWarp;
    __shared__ float warp_sums[kWarps];
    __shared__ float total;
    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

template <typename T, int BLOCK>
__global__ void rmsnorm_kernel(const T* __restrict__ x,
                               const T* __restrict__ weight,
                               T* __restrict__ y,
                               size_t hidden,
                               float eps) {
    size_t row = blockIdx.x;
    const T* row_x = x + row * hidden;
    T* row_y = y + row * hidden;

    NVK_PDL_PROLOG();

    float local = 0.f;
    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float v = to_f32<T>(row_x[i]);
        local += v * v;
    }
    float sum = block_sum<BLOCK>(local);
    float rms = rsqrtf(sum / (float)hidden + eps);

    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float v = to_f32<T>(row_x[i]) * rms * to_f32<T>(weight[i]);
        row_y[i] = from_f32<T>(v);
    }

    NVK_PDL_EPILOG();
}

extern "C" int nv_kernels_rmsnorm_f32(
    void* stream,
    const float* x,
    const float* weight,
    float* y,
    size_t batch,
    size_t hidden,
    float eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    constexpr int BLOCK = 256;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)batch), dim3(BLOCK), 0, s);
        cudaLaunchKernelEx(&cfg, rmsnorm_kernel<float, BLOCK>, x, weight, y, hidden, eps);
    } else {
        rmsnorm_kernel<float, BLOCK><<<(int)batch, BLOCK, 0, s>>>(x, weight, y, hidden, eps);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_rmsnorm_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* weight,
    uint16_t* y,
    size_t batch,
    size_t hidden,
    float eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    constexpr int BLOCK = 256;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)batch), dim3(BLOCK), 0, s);
        cudaLaunchKernelEx(
            &cfg, rmsnorm_kernel<__nv_bfloat16, BLOCK>,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<__nv_bfloat16*>(y),
            hidden,
            eps);
    } else {
        rmsnorm_kernel<__nv_bfloat16, BLOCK><<<(int)batch, BLOCK, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(weight),
            reinterpret_cast<__nv_bfloat16*>(y),
            hidden,
            eps);
    }
    return (int)cudaGetLastError();
}
