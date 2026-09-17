#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kWarpsPerBlock = 8;
constexpr int kThreads = kWarpsPerBlock * 32;

__device__ __forceinline__ float warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ bool visible(int t_q, int j, int window) {
    return j <= t_q && (window <= 0 || t_q - j < window);
}

__global__ void attn_masked_softmax_rows_bf16_kernel(
    const float* __restrict__ s,
    __nv_bfloat16* __restrict__ p,
    long rows,
    int t,
    int window,
    float scale
) {
    long row = (long)blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) return;
    int lane = threadIdx.x & 31;
    int t_q = (int)(row % t);
    const float* sr = s + row * t;
    float m = -INFINITY;
    for (int j = lane; j < t; j += 32) {
        if (visible(t_q, j, window)) m = fmaxf(m, sr[j] * scale);
    }
    m = warp_max(m);
    float l = 0.f;
    for (int j = lane; j < t; j += 32) {
        if (visible(t_q, j, window)) l += __expf(sr[j] * scale - m);
    }
    l = warp_sum(l);
    float inv = 1.f / l;
    __nv_bfloat16* pr = p + row * t;
    for (int j = lane; j < t; j += 32) {
        float v = visible(t_q, j, window) ? __expf(sr[j] * scale - m) * inv : 0.f;
        pr[j] = __float2bfloat16(v);
    }
}

__global__ void attn_masked_softmax_bwd_rows_bf16_kernel(
    const float* __restrict__ s,
    const float* __restrict__ dp,
    const float* __restrict__ delta,
    __nv_bfloat16* __restrict__ p,
    __nv_bfloat16* __restrict__ ds,
    long rows,
    int t,
    int window,
    float scale
) {
    long row = (long)blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
    if (row >= rows) return;
    int lane = threadIdx.x & 31;
    int t_q = (int)(row % t);
    const float* sr = s + row * t;
    const float* dpr = dp + row * t;
    float m = -INFINITY;
    for (int j = lane; j < t; j += 32) {
        if (visible(t_q, j, window)) m = fmaxf(m, sr[j] * scale);
    }
    m = warp_max(m);
    float l = 0.f;
    for (int j = lane; j < t; j += 32) {
        if (visible(t_q, j, window)) l += __expf(sr[j] * scale - m);
    }
    l = warp_sum(l);
    float inv = 1.f / l;
    float d = delta[row];
    __nv_bfloat16* pr = p + row * t;
    __nv_bfloat16* dsr = ds + row * t;
    for (int j = lane; j < t; j += 32) {
        float pv = visible(t_q, j, window) ? __expf(sr[j] * scale - m) * inv : 0.f;
        pr[j] = __float2bfloat16(pv);
        dsr[j] = __float2bfloat16(pv * (dpr[j] - d) * scale);
    }
}

}

extern "C" int nv_kernels_attn_masked_softmax_rows_bf16(
    void* stream, const float* s, uint16_t* p, long rows, int t, int window, float scale) {
    if (rows <= 0 || t <= 0) return 0;
    unsigned blocks = (unsigned)((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    attn_masked_softmax_rows_bf16_kernel<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(
        s, reinterpret_cast<__nv_bfloat16*>(p), rows, t, window, scale);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_attn_masked_softmax_bwd_rows_bf16(
    void* stream, const float* s, const float* dp, const float* delta, uint16_t* p, uint16_t* ds,
    long rows, int t, int window, float scale) {
    if (rows <= 0 || t <= 0) return 0;
    unsigned blocks = (unsigned)((rows + kWarpsPerBlock - 1) / kWarpsPerBlock);
    attn_masked_softmax_bwd_rows_bf16_kernel<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(
        s, dp, delta, reinterpret_cast<__nv_bfloat16*>(p), reinterpret_cast<__nv_bfloat16*>(ds),
        rows, t, window, scale);
    return (int)cudaGetLastError();
}
