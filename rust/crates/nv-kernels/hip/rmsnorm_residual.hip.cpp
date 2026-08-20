#include "hip/hip_runtime.h"
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include "nv_kernels.h"

template <int BLOCK>
__global__ void rmsnorm_residual_kernel_bf16(
    const __hip_bfloat16* __restrict__ x,
    __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ weight,
    __hip_bfloat16* __restrict__ out,
    size_t hidden,
    float eps
) {
    __shared__ float scratch[BLOCK];
    __shared__ float row_rms;

    size_t row = blockIdx.x;
    const __hip_bfloat16* row_x = x + row * hidden;
    __hip_bfloat16* row_res = residual + row * hidden;
    __hip_bfloat16* row_out = out + row * hidden;

    float local = 0.f;
    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float xv = __bfloat162float(row_x[i]);
        float rv = __bfloat162float(row_res[i]);
        float s = xv + rv;
        row_res[i] = __float2bfloat16(s);
        local += s * s;
    }
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_rms = rsqrtf(scratch[0] / (float)hidden + eps);
    }
    __syncthreads();

    float rms = row_rms;

    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float s = __bfloat162float(row_res[i]);
        float w = __bfloat162float(weight[i]);
        float v = s * rms * w;
        row_out[i] = __float2bfloat16(v);
    }
}

template <int BLOCK>
__global__ void rmsnorm_residual_kernel_f32(
    const float* __restrict__ x,
    float* __restrict__ residual,
    const float* __restrict__ weight,
    float* __restrict__ out,
    size_t hidden,
    float eps
) {
    __shared__ float scratch[BLOCK];
    __shared__ float row_rms;

    size_t row = blockIdx.x;
    const float* row_x = x + row * hidden;
    float* row_res = residual + row * hidden;
    float* row_out = out + row * hidden;

    float local = 0.f;
    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float s = row_x[i] + row_res[i];
        row_res[i] = s;
        local += s * s;
    }
    scratch[threadIdx.x] = local;
    __syncthreads();
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_rms = rsqrtf(scratch[0] / (float)hidden + eps);
    }
    __syncthreads();

    float rms = row_rms;
    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        row_out[i] = row_res[i] * rms * weight[i];
    }
}

extern "C" int nv_kernels_rmsnorm_residual_bf16(
    void* stream,
    const uint16_t* x,
    uint16_t* residual,
    const uint16_t* weight,
    uint16_t* out,
    size_t batch,
    size_t hidden,
    float eps
) {
    hipStream_t s = (hipStream_t)stream;
    constexpr int BLOCK = 256;
    rmsnorm_residual_kernel_bf16<BLOCK><<<(int)batch, BLOCK, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<__hip_bfloat16*>(residual),
        reinterpret_cast<const __hip_bfloat16*>(weight),
        reinterpret_cast<__hip_bfloat16*>(out),
        hidden, eps);
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_rmsnorm_residual_f32(
    void* stream,
    const float* x,
    float* residual,
    const float* weight,
    float* out,
    size_t batch,
    size_t hidden,
    float eps
) {
    hipStream_t s = (hipStream_t)stream;
    constexpr int BLOCK = 256;
    rmsnorm_residual_kernel_f32<BLOCK><<<(int)batch, BLOCK, 0, s>>>(
        x, residual, weight, out, hidden, eps);
    return (int)hipGetLastError();
}
