#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <math.h>
#include "nv_kernels.h"

template <typename T>
__device__ inline float silu_to_f32(T x);

template <>
__device__ inline float silu_to_f32<float>(float x) { return x; }

template <>
__device__ inline float silu_to_f32<__hip_bfloat16>(__hip_bfloat16 x) { return __bfloat162float(x); }

template <typename T>
__device__ inline T silu_from_f32(float x);

template <>
__device__ inline float silu_from_f32<float>(float x) { return x; }

template <>
__device__ inline __hip_bfloat16 silu_from_f32<__hip_bfloat16>(float x) { return __float2bfloat16(x); }

__device__ inline float silu_scalar(float x) {
    return x / (1.f + expf(-x));
}

template <typename T>
__global__ void silu_kernel(const T* __restrict__ x,
                            T* __restrict__ y,
                            size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = silu_to_f32<T>(x[idx]);
    y[idx] = silu_from_f32<T>(silu_scalar(v));
}

template <typename T>
__global__ void silu_mul_kernel(const T* __restrict__ x,
                                const T* __restrict__ gate,
                                T* __restrict__ y,
                                size_t n) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float a = silu_to_f32<T>(x[idx]);
    float b = silu_to_f32<T>(gate[idx]);
    y[idx] = silu_from_f32<T>(silu_scalar(a) * b);
}

extern "C" int nv_kernels_silu_f32(
    void* stream,
    const float* x,
    float* y,
    size_t n
) {
    hipStream_t s = (hipStream_t)stream;
    constexpr int BLOCK = 256;
    int grid = (int)((n + BLOCK - 1) / BLOCK);
    if (grid <= 0) return 0;
    silu_kernel<float><<<grid, BLOCK, 0, s>>>(x, y, n);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_silu_bf16(
    void* stream,
    const uint16_t* x,
    uint16_t* y,
    size_t n
) {
    hipStream_t s = (hipStream_t)stream;
    constexpr int BLOCK = 256;
    int grid = (int)((n + BLOCK - 1) / BLOCK);
    if (grid <= 0) return 0;
    silu_kernel<__hip_bfloat16><<<grid, BLOCK, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<__hip_bfloat16*>(y),
        n);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_silu_mul_f32(
    void* stream,
    const float* x,
    const float* gate,
    float* y,
    size_t n
) {
    hipStream_t s = (hipStream_t)stream;
    constexpr int BLOCK = 256;
    int grid = (int)((n + BLOCK - 1) / BLOCK);
    if (grid <= 0) return 0;
    silu_mul_kernel<float><<<grid, BLOCK, 0, s>>>(x, gate, y, n);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_silu_mul_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* gate,
    uint16_t* y,
    size_t n
) {
    hipStream_t s = (hipStream_t)stream;
    constexpr int BLOCK = 256;
    int grid = (int)((n + BLOCK - 1) / BLOCK);
    if (grid <= 0) return 0;
    silu_mul_kernel<__hip_bfloat16><<<grid, BLOCK, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<const __hip_bfloat16*>(gate),
        reinterpret_cast<__hip_bfloat16*>(y),
        n);
    return (int)hipGetLastError();
}
