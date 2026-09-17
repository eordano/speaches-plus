#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace nvk_norm_row {

constexpr int kRmsnormRowReduceScratchFloats = 32;

template <int kNormBlock, int kThreads>
__device__ __forceinline__ float block_sum_over_the_first_norm_block_threads_so_a_wider_block_reduces_bitwise_like_the_norm_kernel(float v, float* s_red) {
    static_assert(kThreads >= kNormBlock && kNormBlock % 32 == 0, "the norm block must fit the launch block");
    constexpr int kW = 32;
    constexpr int kWarps = kNormBlock / kW;
    static_assert(kWarps + 1 <= kRmsnormRowReduceScratchFloats, "the reduce scratch holds the warp sums and the total");
    float* warp_sums = s_red;
    float* total = s_red + kWarps;
    int lane = threadIdx.x & (kW - 1);
    int warp = threadIdx.x / kW;
    #pragma unroll
    for (int o = kW / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0 && warp < kWarps) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) *total = s;
    }
    __syncthreads();
    float t = *total;
    __syncthreads();
    return t;
}

template <int kNormBlock, int kThreads>
__device__ __forceinline__ void rmsnorm_add_scale_row(
    const __nv_bfloat16* xr,
    const __nv_bfloat16* w,
    const __nv_bfloat16* rr,
    __nv_bfloat16* yr,
    float* rstd_out,
    const __nv_bfloat16* next_w,
    __nv_bfloat16* nr,
    float* s_red,
    float* s_of,
    int dim,
    float eps,
    float scale,
    float eps_next
) {
    const int tid = threadIdx.x;
    const bool active = tid < kNormBlock;
    float partial = 0.0f;
    if (active) {
        for (int d = tid; d < dim; d += kNormBlock) {
            float v = __bfloat162float(__ldcg(xr + d));
            partial += v * v;
        }
    }
    const bool stage = (nr != nullptr);
    float inv = rsqrtf(
        block_sum_over_the_first_norm_block_threads_so_a_wider_block_reduces_bitwise_like_the_norm_kernel<kNormBlock, kThreads>(partial, s_red) / dim + eps);
    float out_sq = 0.0f;
    if (active) {
        for (int d = tid; d < dim; d += kNormBlock) {
            float normed = __bfloat162float(__ldcg(xr + d)) * inv * __bfloat162float(w[d]);
            float out = (__bfloat162float(rr[d]) + normed) * scale;
            __nv_bfloat16 ob = __float2bfloat16(out);
            yr[d] = ob;
            float of = __bfloat162float(ob);
            if (stage) s_of[d] = of;
            out_sq += of * of;
        }
    }
    if (rstd_out != nullptr || nr != nullptr) {
        float total =
            block_sum_over_the_first_norm_block_threads_so_a_wider_block_reduces_bitwise_like_the_norm_kernel<kNormBlock, kThreads>(out_sq, s_red);
        float inv2 = rsqrtf(total / dim + eps_next);
        if (rstd_out != nullptr && tid == 0) rstd_out[0] = inv2;
        if (nr != nullptr && active) {
            for (int d = tid; d < dim; d += kNormBlock) {
                nr[d] = __float2bfloat16(s_of[d] * inv2 * __bfloat162float(next_w[d]));
            }
        }
    }
}

}
