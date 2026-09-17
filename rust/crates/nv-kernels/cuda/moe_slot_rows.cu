#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kThreads = 256;

__global__ void moe_slot_weighted_sum_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const float* __restrict__ w,
    float* __restrict__ out,
    int tokens,
    int top_k,
    int hidden
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)tokens * hidden;
    if (idx >= total) return;
    int t = (int)(idx / hidden);
    int h = (int)(idx % hidden);
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        acc += w[(size_t)t * top_k + k] *
               __bfloat162float(rows[((size_t)t * top_k + k) * hidden + h]);
    }
    out[idx] = acc;
}

__global__ void moe_slot_weight_grad_kernel(
    const __nv_bfloat16* __restrict__ rows,
    const float* __restrict__ d_out,
    float* __restrict__ dw,
    int tokens,
    int top_k,
    int hidden
) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= tokens * top_k) return;
    int t = warp / top_k;
    const __nv_bfloat16* r = rows + (size_t)warp * hidden;
    const float* d = d_out + (size_t)t * hidden;
    float acc = 0.f;
    for (int h = lane; h < hidden; h += 32) acc += __bfloat162float(r[h]) * d[h];
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) dw[warp] = acc;
}

__global__ void moe_slot_scatter_grad_kernel(
    const float* __restrict__ d_out,
    const float* __restrict__ w,
    __nv_bfloat16* __restrict__ d_rows,
    int tokens,
    int top_k,
    int hidden
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)tokens * top_k * hidden;
    if (idx >= total) return;
    size_t row = idx / hidden;
    int h = (int)(idx % hidden);
    int t = (int)(row / top_k);
    d_rows[idx] = __float2bfloat16(w[row] * d_out[(size_t)t * hidden + h]);
}

__global__ void moe_slot_sum_pairs_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    float* __restrict__ out,
    int tokens,
    int top_k,
    int hidden
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)tokens * hidden;
    if (idx >= total) return;
    int t = (int)(idx / hidden);
    int h = (int)(idx % hidden);
    float acc = 0.f;
    for (int k = 0; k < top_k; ++k) {
        size_t r = ((size_t)t * top_k + k) * hidden + h;
        acc += __bfloat162float(a[r]) + __bfloat162float(b[r]);
    }
    out[idx] = acc;
}

}

extern "C" int nv_kernels_moe_slot_weighted_sum_bf16_f32(
    void* stream, const uint16_t* rows, const float* w, float* out, int tokens, int top_k, int hidden) {
    if (tokens <= 0 || top_k <= 0 || hidden <= 0) return 0;
    size_t total = (size_t)tokens * hidden;
    unsigned blocks = (unsigned)((total + kThreads - 1) / kThreads);
    moe_slot_weighted_sum_kernel<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(rows), w, out, tokens, top_k, hidden);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_moe_slot_weight_grad_bf16_f32(
    void* stream, const uint16_t* rows, const float* d_out, float* dw, int tokens, int top_k, int hidden) {
    if (tokens <= 0 || top_k <= 0 || hidden <= 0) return 0;
    size_t warps = (size_t)tokens * top_k;
    unsigned blocks = (unsigned)((warps * 32 + kThreads - 1) / kThreads);
    moe_slot_weight_grad_kernel<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(rows), d_out, dw, tokens, top_k, hidden);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_moe_slot_scatter_grad_f32_bf16(
    void* stream, const float* d_out, const float* w, uint16_t* d_rows, int tokens, int top_k, int hidden) {
    if (tokens <= 0 || top_k <= 0 || hidden <= 0) return 0;
    size_t total = (size_t)tokens * top_k * hidden;
    unsigned blocks = (unsigned)((total + kThreads - 1) / kThreads);
    moe_slot_scatter_grad_kernel<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(
        d_out, w, reinterpret_cast<__nv_bfloat16*>(d_rows), tokens, top_k, hidden);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_moe_slot_sum_pairs_bf16_f32(
    void* stream, const uint16_t* a, const uint16_t* b, float* out, int tokens, int top_k, int hidden) {
    if (tokens <= 0 || top_k <= 0 || hidden <= 0) return 0;
    size_t total = (size_t)tokens * hidden;
    unsigned blocks = (unsigned)((total + kThreads - 1) / kThreads);
    moe_slot_sum_pairs_kernel<<<blocks, kThreads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(a), reinterpret_cast<const __nv_bfloat16*>(b), out,
        tokens, top_k, hidden);
    return (int)cudaGetLastError();
}
