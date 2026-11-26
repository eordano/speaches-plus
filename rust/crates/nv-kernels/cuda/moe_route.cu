
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>
#include "nv_kernels.h"
#include "nvk_pdl.cuh"

#define ROUTE_MAX_E 4096
#define ROUTE_MAX_K 32

namespace {

__device__ __forceinline__ uint32_t rgq_ord_f32(float v) {
    if (isnan(v)) return 1u;
    if (v == 0.f) v = 0.f;
    uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ uint64_t rgq_pack_key(float v, int idx) {
    return ((uint64_t)rgq_ord_f32(v) << 32) | (uint64_t)(0xFFFFFFFFu ^ (uint32_t)idx);
}

__device__ __forceinline__ uint8_t rgq_encode_e2m1(float x) {
    uint8_t sign = signbit(x) ? 0x08u : 0u;
    float a = fabsf(x);
    uint8_t c = (uint8_t)(a > 0.25f) + (uint8_t)(a > 0.75f) + (uint8_t)(a > 1.25f)
              + (uint8_t)(a > 1.75f) + (uint8_t)(a > 2.5f)  + (uint8_t)(a > 3.5f)
              + (uint8_t)(a > 5.0f);
    return sign | c;
}

#define NV_UE4M3_MIN_NORMAL      0.015625f
#define NV_UE4M3_SUBNORMAL_STEP  0.001953125f

__device__ __forceinline__ uint8_t rgq_encode_ue4m3(float scale) {
    if (!isfinite(scale) || scale <= 0.f) return 0;
    float clamped = fminf(scale, 448.f);
    uint8_t b = (uint8_t)__nv_cvt_float_to_fp8(clamped, __NV_SATFINITE, __NV_E4M3);
    return (b == 0x7Fu) ? 0x7Eu : (b & 0x7Fu);
}

__device__ __forceinline__ float rgq_decode_ue4m3(uint8_t b) {
    unsigned biased = (unsigned)(b >> 3) & 0x0Fu;
    unsigned mant = (unsigned)(b & 0x07u);
    float norm = __uint_as_float(((biased + 120u) << 23) | (mant << 20));
    return biased ? norm : (float)mant * NV_UE4M3_SUBNORMAL_STEP;
}

__device__ __forceinline__ int rgq_swizzled_scale_dst(int m, int kb, int k_blocks) {
    int k_tiles = (k_blocks + 3) / 4;
    int m_tile = m / 128;
    int d2 = (m / 32) & 3;
    int d3 = m & 31;
    int k_tile = kb / 4;
    int d5 = kb & 3;
    return ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
}

}

template <int ITEMS_PER_LANE>
__global__ void moe_route_topk_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    int n_tokens,
    int E,
    int K,
    int mode,
    float softcap,
    int norm_topk,
    float routed_scaling,
    int out_stride,
    int shared_tail_id
) {
    NVK_PDL_PROLOG();
    extern __shared__ float smem[];
    float* sel = smem;
    float* raw = smem + E;
    __shared__ float red_val[256];
    __shared__ int red_idx[256];
    __shared__ int win_ids[ROUTE_MAX_K];
    __shared__ float win_w[ROUTE_MAX_K];

    int n = blockIdx.x;
    if (n >= n_tokens) return;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    if (E <= ITEMS_PER_LANE * 32 && nthreads <= 256) {
        constexpr int kW = 32;
        const int lane = tid & (kW - 1);
        const int warp = tid >> 5;

        for (int e = tid; e < E; e += nthreads) {
            float x = logits[(size_t)n * E + e];
            if (softcap > 0.f) {
                x = softcap * tanhf(x / softcap);
            }
            if (mode == 1) {
                float s = 1.f / (1.f + expf(-x));
                raw[e] = s;
                sel[e] = s + (bias ? bias[e] : 0.f);
            } else {
                raw[e] = x;
                sel[e] = x;
            }
        }
        __syncthreads();
        if (warp != 0) return;

        float my[ITEMS_PER_LANE];
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_LANE; ++i) {
            int e = lane + i * kW;
            my[i] = (e < E) ? sel[e] : -INFINITY;
        }

        for (int j = 0; j < K; ++j) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
            #pragma unroll
            for (int i = 0; i < ITEMS_PER_LANE; ++i) {
                int e = lane + i * kW;
                if (e < E && (my[i] > bv || (my[i] == bv && e < bi))) {
                    bv = my[i];
                    bi = e;
                }
            }
            #pragma unroll
            for (int o = kW / 2; o > 0; o >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }

            if (lane == 0) {
                win_ids[j] = (bi < E) ? bi : -1;
                win_w[j] = (bi < E) ? raw[bi] : 0.f;
            }
            if (bi < E && (bi & (kW - 1)) == lane) {
                int slot = bi >> 5;
                #pragma unroll
                for (int i = 0; i < ITEMS_PER_LANE; ++i) {
                    if (i == slot) my[i] = -INFINITY;
                }
            }
        }

        if (lane == 0) {
            if (mode == 0) {
                float mx = -INFINITY;
                for (int j = 0; j < K; ++j) mx = fmaxf(mx, win_w[j]);
                float sum = 0.f;
                for (int j = 0; j < K; ++j) {
                    win_w[j] = expf(win_w[j] - mx);
                    sum += win_w[j];
                }
                for (int j = 0; j < K; ++j) win_w[j] /= sum;
            } else if (norm_topk) {
                float sum = 0.f;
                for (int j = 0; j < K; ++j) sum += win_w[j];
                if (sum > 0.f) {
                    for (int j = 0; j < K; ++j) win_w[j] /= sum;
                }
            }
            for (int j = 0; j < K; ++j) {
                topk_ids[(size_t)n * out_stride + j] = win_ids[j];
                topk_weights[(size_t)n * out_stride + j] = win_w[j] * routed_scaling;
            }
            if (shared_tail_id >= 0 && out_stride > K) {
                topk_ids[(size_t)n * out_stride + K] = shared_tail_id;
                topk_weights[(size_t)n * out_stride + K] = 1.f;
            }
        }
        return;
    }

    for (int e = tid; e < E; e += nthreads) {
        float x = logits[(size_t)n * E + e];
        if (softcap > 0.f) {
            x = softcap * tanhf(x / softcap);
        }
        if (mode == 1) {
            float s = 1.f / (1.f + expf(-x));
            raw[e] = s;
            sel[e] = s + (bias ? bias[e] : 0.f);
        } else {
            raw[e] = x;
            sel[e] = x;
        }
    }
    __syncthreads();

    for (int j = 0; j < K; ++j) {
        float best = -INFINITY;
        int best_i = -1;
        for (int e = tid; e < E; e += nthreads) {
            float v = sel[e];
            if (v > best || (v == best && e < best_i)) {
                best = v;
                best_i = e;
            }
        }
        red_val[tid] = best;
        red_idx[tid] = best_i;
        __syncthreads();
        for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                float ov = red_val[tid + stride];
                int oi = red_idx[tid + stride];
                if (ov > red_val[tid] ||
                    (ov == red_val[tid] && oi >= 0 && (red_idx[tid] < 0 || oi < red_idx[tid]))) {
                    red_val[tid] = ov;
                    red_idx[tid] = oi;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            int w = red_idx[0];
            win_ids[j] = w;
            win_w[j] = (w >= 0) ? raw[w] : 0.f;
            if (w >= 0) sel[w] = -INFINITY;
        }
        __syncthreads();
    }

    if (tid == 0) {
        if (mode == 0) {
            float mx = -INFINITY;
            for (int j = 0; j < K; ++j) mx = fmaxf(mx, win_w[j]);
            float sum = 0.f;
            for (int j = 0; j < K; ++j) {
                win_w[j] = expf(win_w[j] - mx);
                sum += win_w[j];
            }
            for (int j = 0; j < K; ++j) win_w[j] /= sum;
        } else if (norm_topk) {
            float sum = 0.f;
            for (int j = 0; j < K; ++j) sum += win_w[j];
            if (sum > 0.f) {
                for (int j = 0; j < K; ++j) win_w[j] /= sum;
            }
        }
        for (int j = 0; j < K; ++j) {
            topk_ids[(size_t)n * out_stride + j] = win_ids[j];
            topk_weights[(size_t)n * out_stride + j] = win_w[j] * routed_scaling;
        }
        if (shared_tail_id >= 0 && out_stride > K) {
            topk_ids[(size_t)n * out_stride + K] = shared_tail_id;
            topk_weights[(size_t)n * out_stride + K] = 1.f;
        }
    }
}

static void moe_route_topk_launch(
    cudaStream_t s, const float* logits, const float* bias, int32_t* topk_ids, float* topk_weights,
    int n_tokens, int E, int K, int mode, float softcap, int norm_topk, float routed_scaling,
    int out_stride, int shared_tail_id
) {
    const int tx = 256;
    size_t shmem = (size_t)2 * E * sizeof(float);
    if (E <= 256) {
        moe_route_topk_kernel<8><<<n_tokens, tx, shmem, s>>>(
            logits, bias, topk_ids, topk_weights, n_tokens, E, K, mode, softcap, norm_topk,
            routed_scaling, out_stride, shared_tail_id);
    } else {
        moe_route_topk_kernel<16><<<n_tokens, tx, shmem, s>>>(
            logits, bias, topk_ids, topk_weights, n_tokens, E, K, mode, softcap, norm_topk,
            routed_scaling, out_stride, shared_tail_id);
    }
}

extern "C" int nv_kernels_moe_route_topk(
    void* stream,
    const float* logits,
    const float* bias,
    int32_t* topk_ids,
    float* topk_weights,
    int n_tokens,
    int E,
    int K,
    int mode,
    float softcap,
    int norm_topk,
    float routed_scaling
) {
    if (n_tokens <= 0 || E <= 0 || K <= 0) return 0;
    if (E > ROUTE_MAX_E || K > ROUTE_MAX_K || K > E) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    moe_route_topk_launch(
        s, logits, bias, topk_ids, topk_weights, n_tokens, E, K, mode, softcap, norm_topk,
        routed_scaling, K, -1);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_moe_route_topk_shared_tail(
    void* stream,
    const float* logits,
    const float* bias,
    int32_t* topk_ids,
    float* topk_weights,
    int n_tokens,
    int E,
    int K,
    int mode,
    float softcap,
    int norm_topk,
    float routed_scaling,
    int shared_tail_id
) {
    if (n_tokens <= 0 || E <= 0 || K <= 0) return 0;
    if (E > ROUTE_MAX_E || K + 1 > ROUTE_MAX_K || K > E) return -2;
    if (shared_tail_id < 0) return -3;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    moe_route_topk_launch(
        s, logits, bias, topk_ids, topk_weights, n_tokens, E, K, mode, softcap, norm_topk,
        routed_scaling, K + 1, shared_tail_id);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

template <int ITEMS_PER_LANE>
__global__ void moe_route_gather_quant_small_m_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ globals_gu,
    const float* __restrict__ globals_dn,
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    float* __restrict__ gu_mini,
    float* __restrict__ dn_mini,
    uint8_t* __restrict__ x_fp4,
    uint8_t* __restrict__ x_sf,
    int E,
    int K,
    int mode,
    float softcap,
    int norm_topk,
    float routed_scaling,
    int shared_tail_id,
    int hidden,
    int min_tile
) {
    extern __shared__ float smem[];
    float* sel = smem;
    float* raw = smem + E;
    __shared__ int win_ids[ROUTE_MAX_K];
    __shared__ float win_w[ROUTE_MAX_K];
    __shared__ float s_stored[ROUTE_MAX_K];

    constexpr int kW = 32;
    int tid = threadIdx.x;
    const int lane = tid & (kW - 1);
    const int warp = tid >> 5;
    const int tiles = K + (shared_tail_id >= 0 ? 1 : 0);
    const int n = blockIdx.y;
    const bool writer_cta = blockIdx.x == 0;
    const int groups = hidden / 16;
    const bool fast_reg_x = (groups <= (int)blockDim.x);
    float reg_vals[16];
    float reg_amax = 0.f;
    if (fast_reg_x && tid < groups) {
        const uint4* xv4 = reinterpret_cast<const uint4*>(x + (size_t)n * hidden + tid * 16);
        uint4 v0 = __ldg(&xv4[0]);
        uint4 v1 = __ldg(&xv4[1]);
        const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
        const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 p0 = __bfloat1622float2(h0[i]);
            float2 p1 = __bfloat1622float2(h1[i]);
            reg_vals[2 * i] = p0.x;
            reg_vals[2 * i + 1] = p0.y;
            reg_vals[8 + 2 * i] = p1.x;
            reg_vals[8 + 2 * i + 1] = p1.y;
            reg_amax = fmaxf(reg_amax, fmaxf(fabsf(p0.x), fabsf(p0.y)));
            reg_amax = fmaxf(reg_amax, fmaxf(fabsf(p1.x), fabsf(p1.y)));
        }
    }
    NVK_PDL_PROLOG();

    if (K <= 16) {
        __shared__ uint64_t s_ckey[8 * 16];
        __shared__ float s_raw[512];
        int e0 = tid;
        int e1 = tid + 256;
        uint64_t k0 = 0ull;
        uint64_t k1 = 0ull;
        if (e0 < E) {
            float xv = __ldg(&logits[(size_t)n * E + e0]);
            if (softcap > 0.f) xv = softcap * tanhf(xv / softcap);
            float r0, my0;
            if (mode == 1) {
                r0 = 1.f / (1.f + expf(-xv));
                my0 = r0 + (bias ? __ldg(&bias[e0]) : 0.f);
            } else {
                r0 = xv;
                my0 = xv;
            }
            s_raw[e0] = r0;
            k0 = rgq_pack_key(my0, e0);
        }
        if (e1 < E) {
            float xv = __ldg(&logits[(size_t)n * E + e1]);
            if (softcap > 0.f) xv = softcap * tanhf(xv / softcap);
            float r1, my1;
            if (mode == 1) {
                r1 = 1.f / (1.f + expf(-xv));
                my1 = r1 + (bias ? __ldg(&bias[e1]) : 0.f);
            } else {
                r1 = xv;
                my1 = xv;
            }
            s_raw[e1] = r1;
            k1 = rgq_pack_key(my1, e1);
        }
        for (int j = 0; j < K; ++j) {
            uint64_t bk = (k0 >= k1) ? k0 : k1;
            #pragma unroll
            for (int o = kW / 2; o > 0; o >>= 1) {
                uint64_t ok = __shfl_xor_sync(0xffffffffu, bk, o);
                if (ok > bk) bk = ok;
            }
            if (lane == 0) s_ckey[warp * K + j] = bk;
            if (k0 == bk) k0 = 0ull;
            if (k1 == bk) k1 = 0ull;
        }
        __syncthreads();
        if (warp == 0) {
            int ncand = 8 * K;
            uint64_t ck[4];
            #pragma unroll
            for (int c = 0; c < 4; ++c) {
                int idx = lane + c * 32;
                ck[c] = (idx < ncand) ? s_ckey[idx] : 0ull;
            }
            for (int j = 0; j < K; ++j) {
                uint64_t bk = ck[0];
                if (ck[1] > bk) bk = ck[1];
                if (ck[2] > bk) bk = ck[2];
                if (ck[3] > bk) bk = ck[3];
                #pragma unroll
                for (int o = kW / 2; o > 0; o >>= 1) {
                    uint64_t ok = __shfl_xor_sync(0xffffffffu, bk, o);
                    if (ok > bk) bk = ok;
                }
                if (lane == 0) {
                    int bi = (bk == 0ull) ? 0x7fffffff : (int)(0xFFFFFFFFu ^ (uint32_t)(bk & 0xFFFFFFFFull));
                    int w = (bi < E) ? bi : -1;
                    win_ids[j] = w;
                    win_w[j] = (w >= 0) ? s_raw[w] : 0.f;
                }
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    if (ck[c] == bk) ck[c] = 0ull;
                }
            }
            if (lane == 0) {
                if (mode == 0) {
                    float mx = -INFINITY;
                    for (int j = 0; j < K; ++j) mx = fmaxf(mx, win_w[j]);
                    float sum = 0.f;
                    for (int j = 0; j < K; ++j) {
                        win_w[j] = expf(win_w[j] - mx);
                        sum += win_w[j];
                    }
                    for (int j = 0; j < K; ++j) win_w[j] /= sum;
                } else if (norm_topk) {
                    float sum = 0.f;
                    for (int j = 0; j < K; ++j) sum += win_w[j];
                    if (sum > 0.f) {
                        for (int j = 0; j < K; ++j) win_w[j] /= sum;
                    }
                }
                if (writer_cta) {
                    for (int j = 0; j < K; ++j) {
                        topk_ids[n * tiles + j] = win_ids[j];
                        topk_weights[n * tiles + j] = win_w[j] * routed_scaling;
                    }
                    if (shared_tail_id >= 0) {
                        topk_ids[n * tiles + K] = shared_tail_id;
                        topk_weights[n * tiles + K] = 1.f;
                    }
                }
            }
            __syncwarp();
            if (lane < tiles) {
                int id = (lane < K) ? win_ids[lane] : shared_tail_id;
                float g_gu = (id >= 0) ? __ldg(&globals_gu[id]) : 0.f;
                float g_dn = (id >= 0) ? __ldg(&globals_dn[id]) : 0.f;
                if (writer_cta) {
                    gu_mini[n * tiles + lane] = g_gu;
                    dn_mini[n * tiles + lane] = g_dn;
                }
                s_stored[lane] = (g_gu == 0.f || !isfinite(g_gu)) ? 1.f : g_gu;
            }
        }
        __syncthreads();
    } else {
        for (int e = tid; e < E; e += blockDim.x) {
            float xv = logits[(size_t)n * E + e];
            if (softcap > 0.f) {
                xv = softcap * tanhf(xv / softcap);
            }
            if (mode == 1) {
                float s = 1.f / (1.f + expf(-xv));
                raw[e] = s;
                sel[e] = s + (bias ? bias[e] : 0.f);
            } else {
                raw[e] = xv;
                sel[e] = xv;
            }
        }
        __syncthreads();

        __shared__ float s_wval[8];
        __shared__ int s_widx[8];
        __shared__ int s_pick;
        int e0 = tid;
        int e1 = tid + 256;
        float my0 = (e0 < E) ? sel[e0] : -INFINITY;
        float my1 = (e1 < E) ? sel[e1] : -INFINITY;
        for (int j = 0; j < K; ++j) {
            float bv = my0;
            int bi = (e0 < E) ? e0 : 0x7fffffff;
            if (my1 > bv || (my1 == bv && e1 < bi)) {
                bv = my1;
                bi = e1;
            }
            #pragma unroll
            for (int o = kW / 2; o > 0; o >>= 1) {
                float ov = __shfl_xor_sync(0xffffffffu, bv, o);
                int oi = __shfl_xor_sync(0xffffffffu, bi, o);
                if (ov > bv || (ov == bv && oi < bi)) {
                    bv = ov;
                    bi = oi;
                }
            }
            if (lane == 0) {
                s_wval[warp] = bv;
                s_widx[warp] = bi;
            }
            __syncthreads();
            if (warp == 0) {
                float wv = (lane < 8) ? s_wval[lane] : -INFINITY;
                int wi = (lane < 8) ? s_widx[lane] : 0x7fffffff;
                #pragma unroll
                for (int o = 4; o > 0; o >>= 1) {
                    float ov = __shfl_xor_sync(0xffffffffu, wv, o);
                    int oi = __shfl_xor_sync(0xffffffffu, wi, o);
                    if (ov > wv || (ov == wv && oi < wi)) {
                        wv = ov;
                        wi = oi;
                    }
                }
                if (lane == 0) {
                    int w = (wi < E) ? wi : -1;
                    win_ids[j] = w;
                    win_w[j] = (w >= 0) ? raw[w] : 0.f;
                    s_pick = w;
                }
            }
            __syncthreads();
            int picked = s_pick;
            if (e0 == picked) my0 = -INFINITY;
            if (e1 == picked) my1 = -INFINITY;
        }
        if (tid == 0) {
            if (mode == 0) {
                float mx = -INFINITY;
                for (int j = 0; j < K; ++j) mx = fmaxf(mx, win_w[j]);
                float sum = 0.f;
                for (int j = 0; j < K; ++j) {
                    win_w[j] = expf(win_w[j] - mx);
                    sum += win_w[j];
                }
                for (int j = 0; j < K; ++j) win_w[j] /= sum;
            } else if (norm_topk) {
                float sum = 0.f;
                for (int j = 0; j < K; ++j) sum += win_w[j];
                if (sum > 0.f) {
                    for (int j = 0; j < K; ++j) win_w[j] /= sum;
                }
            }
            if (writer_cta) {
                for (int j = 0; j < K; ++j) {
                    topk_ids[n * tiles + j] = win_ids[j];
                    topk_weights[n * tiles + j] = win_w[j] * routed_scaling;
                }
                if (shared_tail_id >= 0) {
                    topk_ids[n * tiles + K] = shared_tail_id;
                    topk_weights[n * tiles + K] = 1.f;
                }
            }
        }
        __syncthreads();

        if (tid < tiles) {
            int id = (tid < K) ? win_ids[tid] : shared_tail_id;
            float g_gu = (id >= 0) ? globals_gu[id] : 0.f;
            float g_dn = (id >= 0) ? globals_dn[id] : 0.f;
            if (writer_cta) {
                gu_mini[n * tiles + tid] = g_gu;
                dn_mini[n * tiles + tid] = g_dn;
            }
            s_stored[tid] = (g_gu == 0.f || !isfinite(g_gu)) ? 1.f : g_gu;
        }
        __syncthreads();
    }

    if (fast_reg_x) {
        if (tid < groups) {
            int kb = tid;
            float local_scale = (reg_amax == 0.f) ? 1.f : (reg_amax / 6.f);
            for (int t = blockIdx.x; t < tiles; t += gridDim.x) {
                float stored = s_stored[t];
                uint8_t scale_byte = rgq_encode_ue4m3(stored * local_scale);
                float scale_decoded = rgq_decode_ue4m3(scale_byte);
                float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);
                int row = (n * tiles + t) * min_tile;
                uint8_t* pblock = x_fp4 + (size_t)row * (hidden / 2) + kb * 8;
                uint8_t packed[8];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float v_lo = fminf(fmaxf(reg_vals[2 * i] * inv, -6.f), 6.f);
                    float v_hi = fminf(fmaxf(reg_vals[2 * i + 1] * inv, -6.f), 6.f);
                    uint8_t lo = rgq_encode_e2m1(v_lo);
                    uint8_t hi = rgq_encode_e2m1(v_hi);
                    packed[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
                }
                *reinterpret_cast<uint2*>(pblock) = *reinterpret_cast<const uint2*>(packed);
                x_sf[rgq_swizzled_scale_dst(row, kb, groups)] = scale_byte;
            }
        }
    } else {
        for (int gidx = tid + blockIdx.x * blockDim.x; gidx < tiles * groups;
             gidx += blockDim.x * gridDim.x) {
            int t = gidx / groups;
            int kb = gidx % groups;
            float vals[16];
            float amax = 0.f;
            const uint4* xv4 = reinterpret_cast<const uint4*>(x + (size_t)n * hidden + kb * 16);
            uint4 v0 = __ldg(&xv4[0]);
            uint4 v1 = __ldg(&xv4[1]);
            const __nv_bfloat162* h0 = reinterpret_cast<const __nv_bfloat162*>(&v0);
            const __nv_bfloat162* h1 = reinterpret_cast<const __nv_bfloat162*>(&v1);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 p0 = __bfloat1622float2(h0[i]);
                float2 p1 = __bfloat1622float2(h1[i]);
                vals[2 * i] = p0.x;
                vals[2 * i + 1] = p0.y;
                vals[8 + 2 * i] = p1.x;
                vals[8 + 2 * i + 1] = p1.y;
                amax = fmaxf(amax, fmaxf(fabsf(p0.x), fabsf(p0.y)));
                amax = fmaxf(amax, fmaxf(fabsf(p1.x), fabsf(p1.y)));
            }
            float stored = s_stored[t];
            float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
            float stored_scale = stored * local_scale;
            uint8_t scale_byte = rgq_encode_ue4m3(stored_scale);
            float scale_decoded = rgq_decode_ue4m3(scale_byte);
            float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);

            int row = (n * tiles + t) * min_tile;
            uint8_t* pblock = x_fp4 + (size_t)row * (hidden / 2) + kb * 8;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                float v_lo = fminf(fmaxf(vals[2 * i] * inv, -6.f), 6.f);
                float v_hi = fminf(fmaxf(vals[2 * i + 1] * inv, -6.f), 6.f);
                uint8_t lo = rgq_encode_e2m1(v_lo);
                uint8_t hi = rgq_encode_e2m1(v_hi);
                pblock[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
            }
            x_sf[rgq_swizzled_scale_dst(row, kb, groups)] = scale_byte;
        }
    }
    NVK_PDL_EPILOG();
}

extern "C" int nv_kernels_moe_route_gather_quant_small_m(
    void* stream,
    const float* logits,
    const float* bias,
    const uint16_t* x_bf16,
    const float* globals_gu,
    const float* globals_dn,
    int32_t* topk_ids,
    float* topk_weights,
    float* gu_mini,
    float* dn_mini,
    uint8_t* x_fp4,
    uint8_t* x_sf,
    int n_tokens,
    int E,
    int K,
    int mode,
    float softcap,
    int norm_topk,
    float routed_scaling,
    int shared_tail_id,
    int hidden,
    int min_tile,
    int quant_ctas
) {
    if (n_tokens <= 0 || E <= 0 || K <= 0 || hidden <= 0 || min_tile <= 0) return -1;
    if (E > 512 || K + 1 > ROUTE_MAX_K || K > E) return -2;
    if ((hidden % 16) != 0 || (min_tile % 128) != 0) return -3;
    if (n_tokens > 65535) return -4;
    if (quant_ctas <= 0 || quant_ctas > 64) return -5;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int tx = 256;
    size_t shmem = (K <= 16) ? 0u : ((size_t)2 * E * sizeof(float));
    dim3 grid((unsigned)quant_ctas, (unsigned)n_tokens);
    auto kern = (E <= 256) ? moe_route_gather_quant_small_m_kernel<8> : moe_route_gather_quant_small_m_kernel<16>;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3((unsigned)tx), shmem, s);
        cudaLaunchKernelEx(&cfg, kern,
            logits, bias,
            reinterpret_cast<const __nv_bfloat16*>(x_bf16),
            globals_gu, globals_dn,
            topk_ids, topk_weights, gu_mini, dn_mini,
            x_fp4, x_sf,
            E, K, mode, softcap, norm_topk, routed_scaling,
            shared_tail_id, hidden, min_tile);
    } else {
        kern<<<grid, tx, shmem, s>>>(
            logits, bias,
            reinterpret_cast<const __nv_bfloat16*>(x_bf16),
            globals_gu, globals_dn,
            topk_ids, topk_weights, gu_mini, dn_mini,
            x_fp4, x_sf,
            E, K, mode, softcap, norm_topk, routed_scaling,
            shared_tail_id, hidden, min_tile
        );
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

#define ROUTER_GEMV_MAX_SHARED_K 4096

template <int TM>
__global__ void __launch_bounds__(128) moe_router_gemv_sg_xstage_small_m_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wg,
    const __nv_bfloat16* __restrict__ wsg,
    float* __restrict__ logits_f32,
    __nv_bfloat16* __restrict__ sg_out,
    __nv_bfloat16* __restrict__ x_stage,
    int E,
    int K
) {
    __shared__ float s_red[TM][4];
    constexpr int kW = 32;
    constexpr int kDim = 128;
    int tid = threadIdx.x;
    int lane = tid & (kW - 1);
    int warp = tid >> 5;
    int n = blockIdx.x;
    int n_total = E + (wsg != nullptr ? 1 : 0);
    bool active = n < n_total;
    int Kv = K >> 3;
    const uint4* w4 = active ? reinterpret_cast<const uint4*>((n < E) ? (wg + (size_t)n * K) : wsg) : nullptr;
    uint4 wcur = (active && tid < Kv) ? __ldcs(&w4[tid]) : make_uint4(0u, 0u, 0u, 0u);
    NVK_PDL_PROLOG();

    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    if (x_stage != nullptr && blockIdx.x == 0) {
        int total_v = TM * Kv;
        uint4* xst4 = reinterpret_cast<uint4*>(x_stage);
        for (int i = tid; i < total_v; i += kDim) {
            xst4[i] = __ldg(&x4[i]);
        }
    }

    if (!active) {
        NVK_PDL_EPILOG();
        return;
    }

    float acc[TM];
    #pragma unroll
    for (int r = 0; r < TM; ++r) acc[r] = 0.0f;

    for (int v = tid; v < Kv; v += kDim) {
        int vn = v + kDim;
        uint4 wnxt = (vn < Kv) ? __ldcs(&w4[vn]) : make_uint4(0u, 0u, 0u, 0u);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&wcur);
        float wf[8];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 f = __bfloat1622float2(wp[j]);
            wf[2 * j] = f.x;
            wf[2 * j + 1] = f.y;
        }
        #pragma unroll
        for (int r = 0; r < TM; ++r) {
            uint4 xv = __ldg(&x4[(size_t)r * Kv + v]);
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&xv);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 xf = __bfloat1622float2(xp[j]);
                acc[r] += wf[2 * j] * xf.x + wf[2 * j + 1] * xf.y;
            }
        }
        wcur = wnxt;
    }

    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        #pragma unroll
        for (int offset = kW / 2; offset > 0; offset >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
        }
        if (lane == 0) s_red[r][warp] = acc[r];
    }
    __syncthreads();
    if (tid < TM) {
        float tot = s_red[tid][0] + s_red[tid][1] + s_red[tid][2] + s_red[tid][3];
        __nv_bfloat16 rounded = __float2bfloat16(tot);
        if (n < E) {
            logits_f32[(size_t)tid * E + n] = __bfloat162float(rounded);
        } else {
            sg_out[tid] = rounded;
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
static void launch_router_gemv_tm(
    cudaStream_t s,
    const __nv_bfloat16* x,
    const __nv_bfloat16* wg,
    const __nv_bfloat16* wsg,
    float* logits_f32,
    __nv_bfloat16* sg_out,
    __nv_bfloat16* x_stage,
    int E,
    int K,
    dim3 grid,
    size_t smem
) {
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(128), smem, s);
        cudaLaunchKernelEx(&cfg, moe_router_gemv_sg_xstage_small_m_kernel<TM>,
            x, wg, wsg, logits_f32, sg_out, x_stage, E, K);
    } else {
        moe_router_gemv_sg_xstage_small_m_kernel<TM><<<grid, dim3(128), smem, s>>>(
            x, wg, wsg, logits_f32, sg_out, x_stage, E, K);
    }
}

extern "C" int nv_kernels_moe_router_gemv_sg_xstage_small_m(
    void* stream,
    const uint16_t* x_bf16,
    const uint16_t* wg_bf16,
    const uint16_t* wsg_bf16,
    float* logits_f32,
    uint16_t* sg_out_bf16,
    uint16_t* x_stage_bf16,
    int m,
    int E,
    int K
) {
    if (m <= 0 || E <= 0 || K <= 0) return -1;
    if (m > 8 || (K & 7) != 0 || K > ROUTER_GEMV_MAX_SHARED_K) return -2;
    if ((wsg_bf16 == nullptr) != (sg_out_bf16 == nullptr)) return -3;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int n_total = E + (wsg_bf16 != nullptr ? 1 : 0);
    dim3 grid((unsigned)n_total);
    size_t smem = 0;
    const auto* x = reinterpret_cast<const __nv_bfloat16*>(x_bf16);
    const auto* wg = reinterpret_cast<const __nv_bfloat16*>(wg_bf16);
    const auto* wsg = reinterpret_cast<const __nv_bfloat16*>(wsg_bf16);
    auto* sgo = reinterpret_cast<__nv_bfloat16*>(sg_out_bf16);
    auto* xst = reinterpret_cast<__nv_bfloat16*>(x_stage_bf16);
    switch (m) {
        case 1: launch_router_gemv_tm<1>(s, x, wg, wsg, logits_f32, sgo, xst, E, K, grid, smem); break;
        case 2: launch_router_gemv_tm<2>(s, x, wg, wsg, logits_f32, sgo, xst, E, K, grid, smem); break;
        case 3: launch_router_gemv_tm<3>(s, x, wg, wsg, logits_f32, sgo, xst, E, K, grid, smem); break;
        case 4: launch_router_gemv_tm<4>(s, x, wg, wsg, logits_f32, sgo, xst, E, K, grid, smem); break;
        case 5: launch_router_gemv_tm<5>(s, x, wg, wsg, logits_f32, sgo, xst, E, K, grid, smem); break;
        case 6: launch_router_gemv_tm<6>(s, x, wg, wsg, logits_f32, sgo, xst, E, K, grid, smem); break;
        case 7: launch_router_gemv_tm<7>(s, x, wg, wsg, logits_f32, sgo, xst, E, K, grid, smem); break;
        case 8: launch_router_gemv_tm<8>(s, x, wg, wsg, logits_f32, sgo, xst, E, K, grid, smem); break;
        default: return -2;
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

__global__ void gather_f32_by_ids_kernel(
    const float* __restrict__ src,
    const int32_t* __restrict__ ids,
    float* __restrict__ dst,
    int n
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int e = ids[i];
    dst[i] = (e >= 0) ? src[e] : 0.f;
}

extern "C" int nv_kernels_gather_f32_by_ids(
    void* stream,
    const float* src,
    const int32_t* ids,
    float* dst,
    int n
) {
    if (n <= 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int tx = 128;
    gather_f32_by_ids_kernel<<<(n + tx - 1) / tx, tx, 0, s>>>(src, ids, dst, n);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
