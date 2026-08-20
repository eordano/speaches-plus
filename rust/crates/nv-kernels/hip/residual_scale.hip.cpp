#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <cmath>

namespace {

__global__ void residual_add_scale_bf16_kernel(
    const __hip_bfloat16* __restrict__ a,
    const __hip_bfloat16* __restrict__ b,
    __hip_bfloat16*        __restrict__ y,
    float scale,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __bfloat162float(a[idx]);
    float bv = __bfloat162float(b[idx]);
    y[idx] = __float2bfloat16((av + bv) * scale);
}

__global__ void scale_bf16_kernel(
    __hip_bfloat16* __restrict__ y,
    float scale,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(y[idx]);
    y[idx] = __float2bfloat16(v * scale);
}

__global__ void scale_out_bf16_kernel(
    const __hip_bfloat16* __restrict__ x,
    __hip_bfloat16*       __restrict__ y,
    float scale,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(x[idx]);
    y[idx] = __float2bfloat16(v * scale);
}

}

extern "C" int nv_kernels_residual_add_scale_bf16(
    void* stream,
    const uint16_t* a_bf16,
    const uint16_t* b_bf16,
    uint16_t* y_bf16,
    float scale,
    size_t n
) {
    if (n == 0) return 0;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int block = 256;
    const size_t grid = (n + block - 1) / block;
    if (grid > (size_t)0x7FFFFFFF) return -1;
    residual_add_scale_bf16_kernel<<<(unsigned)grid, block, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(a_bf16),
        reinterpret_cast<const __hip_bfloat16*>(b_bf16),
        reinterpret_cast<__hip_bfloat16*>(y_bf16),
        scale,
        n
    );
    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_scale_inplace_bf16(
    void* stream,
    uint16_t* y_bf16,
    float scale,
    size_t n
) {
    if (n == 0) return 0;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int block = 256;
    const size_t grid = (n + block - 1) / block;
    if (grid > (size_t)0x7FFFFFFF) return -1;
    scale_bf16_kernel<<<(unsigned)grid, block, 0, s>>>(
        reinterpret_cast<__hip_bfloat16*>(y_bf16),
        scale,
        n
    );
    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_scale_out_bf16(
    void* stream,
    const uint16_t* x_bf16,
    uint16_t* y_bf16,
    float scale,
    size_t n
) {
    if (n == 0) return 0;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int block = 256;
    const size_t grid = (n + block - 1) / block;
    if (grid > (size_t)0x7FFFFFFF) return -1;
    scale_out_bf16_kernel<<<(unsigned)grid, block, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(x_bf16),
        reinterpret_cast<__hip_bfloat16*>(y_bf16),
        scale,
        n
    );
    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : static_cast<int>(e);
}

__global__ void tanh_softcap_bf16_to_f32_kernel(
    const __hip_bfloat16* __restrict__ x,
    float*               __restrict__ y,
    float cap,
    float inv_cap,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(x[idx]);
    y[idx] = tanhf(v * inv_cap) * cap;
}

__global__ void cast_bf16_to_f32_kernel(
    const __hip_bfloat16* __restrict__ x,
    float*               __restrict__ y,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = __bfloat162float(x[idx]);
}

extern "C" int nv_kernels_tanh_softcap_bf16_to_f32(
    void* stream,
    const uint16_t* x_bf16,
    float* y_f32,
    float cap,
    size_t n
) {
    if (n == 0) return 0;
    hipStream_t s = static_cast<hipStream_t>(stream);
    const int block = 256;
    const size_t grid = (n + block - 1) / block;
    if (grid > (size_t)0x7FFFFFFF) return -1;
    if (cap > 0.f && std::isfinite(cap)) {
        const float inv_cap = 1.f / cap;
        tanh_softcap_bf16_to_f32_kernel<<<(unsigned)grid, block, 0, s>>>(
            reinterpret_cast<const __hip_bfloat16*>(x_bf16),
            y_f32, cap, inv_cap, n
        );
    } else {
        cast_bf16_to_f32_kernel<<<(unsigned)grid, block, 0, s>>>(
            reinterpret_cast<const __hip_bfloat16*>(x_bf16),
            y_f32, n
        );
    }
    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : static_cast<int>(e);
}
