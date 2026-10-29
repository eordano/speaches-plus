#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include "nv_kernels.h"

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float decode_ue4m3_branchless_dev(uint8_t b) {
    unsigned biased = (unsigned)(b >> 3) & 0x0Fu;
    unsigned mant = (unsigned)(b & 0x07u);
    float norm = __uint_as_float(((biased + 120u) << 23) | (mant << 20));
    return biased ? norm : (float)mant * 0.001953125f;
}

__device__ __forceinline__ float bf16_bits_to_f32(unsigned bits16) {
    return __uint_as_float(bits16 << 16);
}

constexpr int kSweptSmemCapIs48KiBSoTwoCtasShareAnSmWithoutTheDynamicSmemOptIn = 48 * 1024;
constexpr int kSweptChunkGranuleIs512ElementsSoEveryChunkStartKeepsBlockIndexCongruentToLaneMod32 = 512;

__device__ __forceinline__ unsigned e2m1x2_times_2e14_as_half2_bits(unsigned byte) {
    unsigned h2;
    asm("{ .reg .b8 t; cvt.u8.u32 t, %1; cvt.rn.f16x2.e2m1x2 %0, t; }" : "=r"(h2) : "r"(byte));
    __half2 v = __hmul2(*reinterpret_cast<const __half2*>(&h2), __float2half2_rn(6.103515625e-05f));
    return *reinterpret_cast<const unsigned*>(&v);
}

__device__ __forceinline__ void decode_block_e2m1_times_2e14(unsigned w0, unsigned w1, float* w) {
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        unsigned word = (b < 4) ? w0 : w1;
        unsigned h2 = e2m1x2_times_2e14_as_half2_bits((word >> (8 * (b & 3))) & 0xffu);
        float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h2));
        w[2 * b] = f.x;
        w[2 * b + 1] = f.y;
    }
}

__device__ __forceinline__ void unpack_bf16x16(uint4 x0, uint4 x1, float* xf) {
    unsigned xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        xf[2 * i] = bf16_bits_to_f32(xw[i] & 0xffffu);
        xf[2 * i + 1] = bf16_bits_to_f32(xw[i] >> 16);
    }
}

template <int MROWS, int WARPS, int RPW, int DEPTH>
__global__ void __launch_bounds__(WARPS * kWarpSize) nvfp4_gemv_bf16act_swept_kernel(
    const uint8_t* __restrict__ W_packed,
    const uint8_t* __restrict__ W_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    float alpha,
    int N,
    int K,
    int chunk_elems
) {
    extern __shared__ __align__(16) unsigned char
        x_chunk_split_in_half_blocks_so_a_warp_reads_it_bank_conflict_free[];
    uint4* xs = reinterpret_cast<uint4*>(x_chunk_split_in_half_blocks_so_a_warp_reads_it_bank_conflict_free);
    const uint4* xg = reinterpret_cast<const uint4*>(x);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0 = (blockIdx.x * WARPS + warp) * RPW;
    const int n_blocks = K / 16;
    const int chunk_blocks = chunk_elems / 16;

    float acc[MROWS][RPW];
    #pragma unroll
    for (int r = 0; r < MROWS; ++r)
        #pragma unroll
        for (int rr = 0; rr < RPW; ++rr) acc[r][rr] = 0.f;

    for (int c0 = 0; c0 < K; c0 += chunk_elems) {
        const int ce = min(chunk_elems, K - c0);
        const int cb = ce / 16;
        __syncthreads();
        for (int t = threadIdx.x; t < MROWS * cb * 2; t += WARPS * kWarpSize) {
            const int r = t / (cb * 2);
            const int e = t - r * (cb * 2);
            xs[(r * 2 + (e & 1)) * chunk_blocks + (e >> 1)] =
                xg[((size_t)r * K + c0) / 8 + e];
        }
        __syncthreads();
        const int kb0 = c0 / 16;
        const int kbe = kb0 + cb;
        uint2 wq[DEPTH][RPW];
        unsigned sb[DEPTH][RPW];
        #pragma unroll
        for (int d = 0; d < DEPTH; ++d) {
            const int kb = kb0 + lane + d * kWarpSize;
            #pragma unroll
            for (int rr = 0; rr < RPW; ++rr) {
                const int n = n0 + rr;
                const bool live = n < N && kb < kbe;
                wq[d][rr] = live ? *reinterpret_cast<const uint2*>(W_packed + (size_t)n * (K / 2) + (size_t)kb * 8)
                                 : make_uint2(0u, 0u);
                sb[d][rr] = live ? W_scales[(size_t)n * n_blocks + kb] : 0u;
            }
        }
        for (int kbg = kb0 + lane; kbg < kbe; kbg += DEPTH * kWarpSize) {
            uint2 wq_n[DEPTH][RPW];
            unsigned sb_n[DEPTH][RPW];
            #pragma unroll
            for (int d = 0; d < DEPTH; ++d) {
                const int kbn = kbg + (DEPTH + d) * kWarpSize;
                #pragma unroll
                for (int rr = 0; rr < RPW; ++rr) {
                    const int n = n0 + rr;
                    const bool live = n < N && kbn < kbe;
                    wq_n[d][rr] = live ? *reinterpret_cast<const uint2*>(W_packed + (size_t)n * (K / 2) + (size_t)kbn * 8)
                                       : make_uint2(0u, 0u);
                    sb_n[d][rr] = live ? W_scales[(size_t)n * n_blocks + kbn] : 0u;
                }
            }
            #pragma unroll
            for (int d = 0; d < DEPTH; ++d) {
                const int kb = kbg + d * kWarpSize;
                if (kb >= kbe) break;
                float w[RPW][16];
                float sc[RPW];
                #pragma unroll
                for (int rr = 0; rr < RPW; ++rr) {
                    decode_block_e2m1_times_2e14(wq[d][rr].x, wq[d][rr].y, w[rr]);
                    sc[rr] = decode_ue4m3_branchless_dev((uint8_t)sb[d][rr]) * 16384.f;
                }
                const int j = kb - kb0;
                #pragma unroll
                for (int r = 0; r < MROWS; ++r) {
                    float xf[16];
                    unpack_bf16x16(xs[(r * 2) * chunk_blocks + j], xs[(r * 2 + 1) * chunk_blocks + j], xf);
                    #pragma unroll
                    for (int rr = 0; rr < RPW; ++rr) {
                        float bd = 0.f;
                        #pragma unroll
                        for (int e = 0; e < 16; ++e) bd = fmaf(w[rr][e], xf[e], bd);
                        acc[r][rr] = fmaf(sc[rr], bd, acc[r][rr]);
                    }
                }
            }
            #pragma unroll
            for (int d = 0; d < DEPTH; ++d)
                #pragma unroll
                for (int rr = 0; rr < RPW; ++rr) {
                    wq[d][rr] = wq_n[d][rr];
                    sb[d][rr] = sb_n[d][rr];
                }
        }
    }

    #pragma unroll
    for (int r = 0; r < MROWS; ++r) {
        #pragma unroll
        for (int rr = 0; rr < RPW; ++rr) {
            float a = acc[r][rr];
            #pragma unroll
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                a += __shfl_xor_sync(0xffffffff, a, offset);
            }
            const int n = n0 + rr;
            if (lane == 0 && n < N) y[(size_t)r * N + n] = __float2bfloat16(a * alpha);
        }
    }
}

static inline int swept_chunk_elems(int K, int M) {
    const int granule = kSweptChunkGranuleIs512ElementsSoEveryChunkStartKeepsBlockIndexCongruentToLaneMod32;
    int cap = kSweptSmemCapIs48KiBSoTwoCtasShareAnSmWithoutTheDynamicSmemOptIn / (M * 2);
    cap -= cap % granule;
    if (K <= cap) return K;
    return cap;
}

template <int MROWS, int WARPS, int RPW, int DEPTH>
static int launch_swept(
    cudaStream_t s,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    float alpha,
    int N,
    int K
) {
    const int chunk = swept_chunk_elems(K, MROWS);
    const size_t smem = (size_t)MROWS * chunk * sizeof(__nv_bfloat16);
    dim3 grid((unsigned)((N + WARPS * RPW - 1) / (WARPS * RPW)));
    nvfp4_gemv_bf16act_swept_kernel<MROWS, WARPS, RPW, DEPTH><<<grid, WARPS * kWarpSize, smem, s>>>(
        W_packed, W_scales, x, y, alpha, N, K, chunk);
    return (int)cudaGetLastError();
}

template <int MROWS>
static int launch_swept_cfg(
    cudaStream_t s,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    float alpha,
    int N,
    int K,
    int warps,
    int rpw,
    int depth
) {
#define NVK_SWEPT(W, R, D) \
    if (warps == (W) && rpw == (R) && depth == (D)) \
        return launch_swept<MROWS, W, R, D>(s, W_packed, W_scales, x, y, alpha, N, K);
    NVK_SWEPT(16, 2, 1)
    NVK_SWEPT(16, 4, 1)
#undef NVK_SWEPT
    return -3;
}

}

extern "C" int nv_kernels_nvfp4_gemv_bf16act_swept_cfg(
    void* stream,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const uint16_t* x_bf16,
    uint16_t* y,
    float alpha,
    int N,
    int K,
    int M,
    int warps,
    int rpw,
    int depth
) {
    if (N <= 0 || K <= 0 || M <= 0) return 0;
    if ((K & 15) != 0) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    const __nv_bfloat16* xx = reinterpret_cast<const __nv_bfloat16*>(x_bf16);
    __nv_bfloat16* yy = reinterpret_cast<__nv_bfloat16*>(y);
    switch (M) {
        case 1: return launch_swept_cfg<1>(s, W_packed, W_scales, xx, yy, alpha, N, K, warps, rpw, depth);
        case 2: return launch_swept_cfg<2>(s, W_packed, W_scales, xx, yy, alpha, N, K, warps, rpw, depth);
        case 3: return launch_swept_cfg<3>(s, W_packed, W_scales, xx, yy, alpha, N, K, warps, rpw, depth);
        case 4: return launch_swept_cfg<4>(s, W_packed, W_scales, xx, yy, alpha, N, K, warps, rpw, depth);
        case 5: return launch_swept_cfg<5>(s, W_packed, W_scales, xx, yy, alpha, N, K, warps, rpw, depth);
        case 6: return launch_swept_cfg<6>(s, W_packed, W_scales, xx, yy, alpha, N, K, warps, rpw, depth);
        case 7: return launch_swept_cfg<7>(s, W_packed, W_scales, xx, yy, alpha, N, K, warps, rpw, depth);
        case 8: return launch_swept_cfg<8>(s, W_packed, W_scales, xx, yy, alpha, N, K, warps, rpw, depth);
        default: return -2;
    }
}

