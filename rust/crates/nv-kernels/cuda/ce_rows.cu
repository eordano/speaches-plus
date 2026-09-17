#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kThreads = 256;

template <typename Op>
__device__ __forceinline__ float block_reduce_256(float v, float* red8, Op op, float init) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = op(v, __shfl_xor_sync(0xffffffffu, v, o));
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    if (lane == 0) red8[warp] = v;
    __syncthreads();
    float r = init;
    #pragma unroll
    for (int w = 0; w < 8; ++w) r = op(r, red8[w]);
    __syncthreads();
    return r;
}

struct MaxOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
};
struct SumOp {
    __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};

__global__ void __launch_bounds__(kThreads) ce_rows_bf16_fwd_dlogits_kernel(
    const __nv_bfloat16* __restrict__ logits,
    const uint32_t* __restrict__ targets,
    const float* __restrict__ weights,
    float* __restrict__ loss_rows,
    __nv_bfloat16* __restrict__ dlogits,
    int vocab
) {
    const long row = blockIdx.x;
    const __nv_bfloat16* lr = logits + row * (long)vocab;
    __shared__ float red8[8];
    float m = -INFINITY;
    for (int j = threadIdx.x; j < vocab; j += kThreads) m = fmaxf(m, __bfloat162float(lr[j]));
    m = block_reduce_256(m, red8, MaxOp(), -INFINITY);
    float s = 0.f;
    for (int j = threadIdx.x; j < vocab; j += kThreads) s += __expf(__bfloat162float(lr[j]) - m);
    s = block_reduce_256(s, red8, SumOp(), 0.f);
    const float lse = m + __logf(s);
    const uint32_t tgt = targets[row];
    const float w = weights[row];
    if (threadIdx.x == 0) {
        float xt = (tgt < (uint32_t)vocab) ? __bfloat162float(lr[tgt]) : lse;
        loss_rows[row] = w * (lse - xt);
    }
    __nv_bfloat16* dr = dlogits + row * (long)vocab;
    for (int j = threadIdx.x; j < vocab; j += kThreads) {
        float p = __expf(__bfloat162float(lr[j]) - lse);
        float g = w * (p - (((uint32_t)j == tgt) ? 1.f : 0.f));
        dr[j] = __float2bfloat16(g);
    }
}

__global__ void ce_rows_scale_dlogits_kernel(
    __nv_bfloat16* __restrict__ dlogits,
    const float* __restrict__ row_scale,
    long rows,
    int vocab
) {
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = rows * (long)vocab;
    if (idx >= total) return;
    long row = idx / vocab;
    float sc = row_scale[row];
    if (sc != 1.f) dlogits[idx] = __float2bfloat16(__bfloat162float(dlogits[idx]) * sc);
}

}

extern "C" int nv_kernels_ce_rows_bf16_fwd_dlogits(
    void* stream, const uint16_t* logits, const uint32_t* targets, const float* weights,
    float* loss_rows, uint16_t* dlogits, long rows, int vocab) {
    if (rows <= 0 || vocab <= 0) return 0;
    ce_rows_bf16_fwd_dlogits_kernel<<<(unsigned)rows, kThreads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(logits), targets, weights, loss_rows,
        reinterpret_cast<__nv_bfloat16*>(dlogits), vocab);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_ce_rows_scale_dlogits(
    void* stream, uint16_t* dlogits, const float* row_scale, long rows, int vocab) {
    if (rows <= 0 || vocab <= 0) return 0;
    long total = rows * (long)vocab;
    unsigned blocks = (unsigned)((total + 255) / 256);
    ce_rows_scale_dlogits_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<__nv_bfloat16*>(dlogits), row_scale, rows, vocab);
    return (int)cudaGetLastError();
}
