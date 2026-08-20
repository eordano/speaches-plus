#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include "nv_kernels.h"

__global__ void copy_cols_bf16_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int rows,
    int width,
    int64_t src_stride,
    int64_t dst_stride,
    int64_t src_off,
    int64_t dst_off
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)rows * width;
    if (i >= total) return;
    int64_t r = i / width;
    int64_t c = i - r * width;
    dst[r * dst_stride + dst_off + c] = src[r * src_stride + src_off + c];
}

extern "C" int nv_kernels_copy_cols_bf16(
    void* stream,
    const uint16_t* src,
    uint16_t* dst,
    int rows,
    int width,
    long long src_stride,
    long long dst_stride,
    long long src_off,
    long long dst_off
) {
    if (rows <= 0 || width <= 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    long long total = (long long)rows * width;
    const int tx = 256;
    long long bx = (total + tx - 1) / tx;
    copy_cols_bf16_kernel<<<(unsigned int)bx, tx, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(src),
        reinterpret_cast<__nv_bfloat16*>(dst),
        rows, width,
        (int64_t)src_stride, (int64_t)dst_stride,
        (int64_t)src_off, (int64_t)dst_off
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

__global__ void mul_sigmoid_rowgate_f32_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gate_logits,
    float* __restrict__ y,
    int rows,
    int hidden
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)rows * hidden;
    if (i >= total) return;
    int r = (int)(i / hidden);
    float g = 1.f / (1.f + expf(-gate_logits[r]));
    y[i] = x[i] * g;
}

extern "C" int nv_kernels_mul_sigmoid_rowgate_f32(
    void* stream,
    const float* x,
    const float* gate_logits,
    float* y,
    int rows,
    int hidden
) {
    if (rows <= 0 || hidden <= 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    long long total = (long long)rows * hidden;
    const int tx = 256;
    long long bx = (total + tx - 1) / tx;
    mul_sigmoid_rowgate_f32_kernel<<<(unsigned int)bx, tx, 0, s>>>(
        x, gate_logits, y, rows, hidden
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
