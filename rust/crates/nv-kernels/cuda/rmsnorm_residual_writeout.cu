#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "nv_kernels.h"

template <int BLOCK>
__global__ void rmsnorm_residual_writeout_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ res_in,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ res_out,
    __nv_bfloat16* __restrict__ out,
    size_t hidden,
    float eps
) {
    __shared__ float scratch[BLOCK];
    __shared__ float row_rms;

    size_t row = blockIdx.x;
    const __nv_bfloat16* row_x = x + row * hidden;
    const __nv_bfloat16* row_res_in = res_in + row * hidden;
    __nv_bfloat16* row_res_out = res_out + row * hidden;
    __nv_bfloat16* row_out = out + row * hidden;

    float local = 0.f;
    for (size_t i = threadIdx.x; i < hidden; i += BLOCK) {
        float xv = __bfloat162float(row_x[i]);
        float rv = __bfloat162float(row_res_in[i]);
        float s = xv + rv;
        row_res_out[i] = __float2bfloat16(s);
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
        float s = __bfloat162float(row_res_out[i]);
        float w = __bfloat162float(weight[i]);
        float v = s * rms * w;
        row_out[i] = __float2bfloat16(v);
    }
}

extern "C" int nv_kernels_rmsnorm_residual_writeout_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* res_in,
    const uint16_t* weight,
    uint16_t* res_out,
    uint16_t* out,
    size_t batch,
    size_t hidden,
    float eps
) {
    cudaStream_t s = (cudaStream_t)stream;
    constexpr int BLOCK = 256;
    rmsnorm_residual_writeout_kernel_bf16<BLOCK><<<(int)batch, BLOCK, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(res_in),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(res_out),
        reinterpret_cast<__nv_bfloat16*>(out),
        hidden, eps);
    return (int)cudaGetLastError();
}
