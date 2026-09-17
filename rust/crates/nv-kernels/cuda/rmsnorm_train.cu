#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <typename T> __device__ __forceinline__ float ld(const T* p);
template <> __device__ __forceinline__ float ld<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float ld<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ __forceinline__ void st(T* p, float v);
template <> __device__ __forceinline__ void st<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void st<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T>
__global__ void __launch_bounds__(kThreads) rmsnorm_train_fwd_kernel(
    const T* __restrict__ x,
    const float* __restrict__ w,
    T* __restrict__ y,
    float* __restrict__ rstd_out,
    long rows,
    int dim,
    float eps
) {
    long row = (long)blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= rows) return;
    int lane = threadIdx.x & 31;
    const T* xr = x + row * dim;
    float ss = 0.f;
    for (int j = lane; j < dim; j += 32) {
        float v = ld<T>(xr + j);
        ss += v * v;
    }
    ss = warp_sum(ss);
    float rstd = rsqrtf(ss / (float)dim + eps);
    if (lane == 0) rstd_out[row] = rstd;
    T* yr = y + row * dim;
    for (int j = lane; j < dim; j += 32) st<T>(yr + j, ld<T>(xr + j) * rstd * w[j]);
}

template <typename T>
__global__ void __launch_bounds__(kThreads) rmsnorm_train_bwd_kernel(
    const T* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ rstd_in,
    const T* __restrict__ g,
    T* __restrict__ dx,
    long rows,
    int dim
) {
    long row = (long)blockIdx.x * kWarps + (threadIdx.x >> 5);
    if (row >= rows) return;
    int lane = threadIdx.x & 31;
    const T* xr = x + row * dim;
    const T* gr = g + row * dim;
    float rstd = rstd_in[row];
    float dot = 0.f;
    for (int j = lane; j < dim; j += 32) dot += ld<T>(gr + j) * w[j] * ld<T>(xr + j) * rstd;
    dot = warp_sum(dot) / (float)dim;
    T* dr = dx + row * dim;
    for (int j = lane; j < dim; j += 32) {
        float xh = ld<T>(xr + j) * rstd;
        st<T>(dr + j, rstd * (ld<T>(gr + j) * w[j] - xh * dot));
    }
}

}

extern "C" int nv_kernels_rmsnorm_train_fwd(
    void* stream, const void* x, const float* w, void* y, float* rstd_out, long rows, int dim,
    float eps, int x_is_bf16) {
    if (rows <= 0 || dim <= 0) return 0;
    unsigned blocks = (unsigned)((rows + kWarps - 1) / kWarps);
    cudaStream_t s = (cudaStream_t)stream;
    if (x_is_bf16) {
        rmsnorm_train_fwd_kernel<__nv_bfloat16><<<blocks, kThreads, 0, s>>>(
            (const __nv_bfloat16*)x, w, (__nv_bfloat16*)y, rstd_out, rows, dim, eps);
    } else {
        rmsnorm_train_fwd_kernel<float><<<blocks, kThreads, 0, s>>>(
            (const float*)x, w, (float*)y, rstd_out, rows, dim, eps);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_rmsnorm_train_bwd(
    void* stream, const void* x, const float* w, const float* rstd, const void* g, void* dx,
    long rows, int dim, int x_is_bf16) {
    if (rows <= 0 || dim <= 0) return 0;
    unsigned blocks = (unsigned)((rows + kWarps - 1) / kWarps);
    cudaStream_t s = (cudaStream_t)stream;
    if (x_is_bf16) {
        rmsnorm_train_bwd_kernel<__nv_bfloat16><<<blocks, kThreads, 0, s>>>(
            (const __nv_bfloat16*)x, w, rstd, (const __nv_bfloat16*)g, (__nv_bfloat16*)dx, rows, dim);
    } else {
        rmsnorm_train_bwd_kernel<float><<<blocks, kThreads, 0, s>>>(
            (const float*)x, w, rstd, (const float*)g, (float*)dx, rows, dim);
    }
    return (int)cudaGetLastError();
}
