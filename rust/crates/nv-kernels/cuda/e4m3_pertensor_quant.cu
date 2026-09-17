#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
#include "nv_kernels.h"

namespace {

constexpr int kThreads = 256;
constexpr float kE4m3Max448 = 448.0f;

__global__ void e4m3_pertensor_amax_init_kernel(unsigned int* amax_bits) { *amax_bits = 0u; }

__global__ void e4m3_pertensor_amax_kernel(const __nv_bfloat16* __restrict__ x, long long n,
                                           unsigned int* amax_bits) {
    long long stride = (long long)gridDim.x * blockDim.x;
    float local = 0.f;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float a = fabsf(__bfloat162float(x[i]));
        if (isfinite(a)) local = fmaxf(local, a);
    }
    __shared__ float red[kThreads];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int s = kThreads / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicMax(amax_bits, __float_as_uint(red[0]));
}

__global__ void e4m3_pertensor_quant_kernel(const __nv_bfloat16* __restrict__ x, long long n,
                                            const unsigned int* __restrict__ amax_bits,
                                            uint8_t* __restrict__ q, float* __restrict__ scale_out) {
    float amax = __uint_as_float(*amax_bits);
    float scale = (amax > 0.f) ? amax / kE4m3Max448 : 1.f;
    float inv = (amax > 0.f) ? kE4m3Max448 / amax : 0.f;
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale_out = scale;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        float v = __bfloat162float(x[i]) * inv;
        q[i] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}

int grid_for(long long n) {
    long long blocks = (n + kThreads - 1) / kThreads;
    if (blocks > 4096) blocks = 4096;
    if (blocks < 1) blocks = 1;
    return (int)blocks;
}

}

extern "C" int nv_kernels_e4m3_pertensor_quant_two_segments(
    void* stream,
    const uint16_t* x0_bf16,
    const uint16_t* x1_bf16,
    long long n_each,
    uint8_t* q0_out,
    uint8_t* q1_out,
    float* scale_out,
    unsigned int* amax_scratch
) {
    if (n_each <= 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const __nv_bfloat16* x0 = reinterpret_cast<const __nv_bfloat16*>(x0_bf16);
    const __nv_bfloat16* x1 = reinterpret_cast<const __nv_bfloat16*>(x1_bf16);
    e4m3_pertensor_amax_init_kernel<<<1, 1, 0, s>>>(amax_scratch);
    e4m3_pertensor_amax_kernel<<<grid_for(n_each), kThreads, 0, s>>>(x0, n_each, amax_scratch);
    if (x1 != nullptr) e4m3_pertensor_amax_kernel<<<grid_for(n_each), kThreads, 0, s>>>(x1, n_each, amax_scratch);
    e4m3_pertensor_quant_kernel<<<grid_for(n_each), kThreads, 0, s>>>(x0, n_each, amax_scratch, q0_out, scale_out);
    if (x1 != nullptr) e4m3_pertensor_quant_kernel<<<grid_for(n_each), kThreads, 0, s>>>(x1, n_each, amax_scratch, q1_out, scale_out);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
