#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;

template <typename T> __device__ __forceinline__ float ld(const T* p);
template <> __device__ __forceinline__ float ld<float>(const float* p) { return *p; }
template <> __device__ __forceinline__ float ld<__nv_bfloat16>(const __nv_bfloat16* p) { return __bfloat162float(*p); }
template <typename T> __device__ __forceinline__ void st(T* p, float v);
template <> __device__ __forceinline__ void st<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void st<__nv_bfloat16>(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template <typename T, bool BWD>
__global__ void __launch_bounds__(kThreads) rope_train_kernel(
    const T* __restrict__ x,
    T* __restrict__ y,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const uint32_t* __restrict__ pos,
    long tokens,
    int heads,
    int head_dim,
    int rotary_dim,
    float scale
) {
    const long idx = (long)blockIdx.x * kThreads + threadIdx.x;
    const long total = tokens * heads * head_dim;
    if (idx >= total) return;
    const int d = (int)(idx % head_dim);
    const long th = idx / head_dim;
    const long t = th / heads;
    const int half = rotary_dim / 2;
    if (d >= rotary_dim) {
        y[idx] = x[idx];
        return;
    }
    if (d >= half) return;
    const long base = th * head_dim;
    const float c = cos_tbl[(size_t)pos[t] * half + d];
    const float s = sin_tbl[(size_t)pos[t] * half + d];
    float lo = ld<T>(x + base + d);
    float hi = ld<T>(x + base + half + d);
    float out_lo;
    float out_hi;
    if (BWD) {
        if (scale != 1.0f) {
            lo = __fmul_rn(lo, scale);
            hi = __fmul_rn(hi, scale);
        }
        out_lo = __fadd_rn(__fmul_rn(lo, c), __fmul_rn(hi, s));
        out_hi = __fsub_rn(__fmul_rn(hi, c), __fmul_rn(lo, s));
    } else {
        out_lo = __fsub_rn(__fmul_rn(lo, c), __fmul_rn(hi, s));
        out_hi = __fadd_rn(__fmul_rn(lo, s), __fmul_rn(hi, c));
        if (scale != 1.0f) {
            out_lo = __fmul_rn(out_lo, scale);
            out_hi = __fmul_rn(out_hi, scale);
        }
    }
    st<T>(y + base + d, out_lo);
    st<T>(y + base + half + d, out_hi);
}

template <bool BWD>
int launch_rope_train(void* stream, const void* x, void* y, const float* cos_tbl, const float* sin_tbl,
                      const uint32_t* pos, long tokens, int heads, int head_dim, int rotary_dim, float scale, int x_is_bf16) {
    if (tokens <= 0 || heads <= 0 || head_dim <= 0) return 0;
    if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;
    const long total = tokens * heads * head_dim;
    const unsigned blocks = (unsigned)((total + kThreads - 1) / kThreads);
    cudaStream_t s = (cudaStream_t)stream;
    if (x_is_bf16) {
        rope_train_kernel<__nv_bfloat16, BWD><<<blocks, kThreads, 0, s>>>(
            (const __nv_bfloat16*)x, (__nv_bfloat16*)y, cos_tbl, sin_tbl, pos, tokens, heads, head_dim, rotary_dim, scale);
    } else {
        rope_train_kernel<float, BWD><<<blocks, kThreads, 0, s>>>(
            (const float*)x, (float*)y, cos_tbl, sin_tbl, pos, tokens, heads, head_dim, rotary_dim, scale);
    }
    return (int)cudaGetLastError();
}

}

extern "C" int nv_kernels_rope_train_fwd(
    void* stream, const void* x, void* y, const float* cos_tbl, const float* sin_tbl, const uint32_t* pos,
    long tokens, int heads, int head_dim, int rotary_dim, float scale, int x_is_bf16) {
    return launch_rope_train<false>(stream, x, y, cos_tbl, sin_tbl, pos, tokens, heads, head_dim, rotary_dim, scale, x_is_bf16);
}

extern "C" int nv_kernels_rope_train_bwd(
    void* stream, const void* g, void* dx, const float* cos_tbl, const float* sin_tbl, const uint32_t* pos,
    long tokens, int heads, int head_dim, int rotary_dim, float scale, int x_is_bf16) {
    return launch_rope_train<true>(stream, g, dx, cos_tbl, sin_tbl, pos, tokens, heads, head_dim, rotary_dim, scale, x_is_bf16);
}
