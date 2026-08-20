
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include "nv_kernels.h"

#define ROUTE_MAX_E 4096
#define ROUTE_MAX_K 32

namespace {

__device__ __forceinline__ uint8_t rgq_encode_e2m1(float x) {
    static const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    uint8_t sign = signbit(x) ? 0b1000 : 0;
    float a = fabsf(x);
    uint8_t best = 0;
    float best_err = INFINITY;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float err = fabsf(a - kE2M1[i]);
        if (err < best_err) {
            best_err = err;
            best = (uint8_t)i;
        }
    }
    return (uint8_t)(sign | best);
}

#define RGQ_UE4M3_MIN_NORMAL      0.015625f
#define RGQ_UE4M3_SUBNORMAL_STEP  0.001953125f

__device__ __forceinline__ uint8_t rgq_encode_ue4m3(float scale) {
    if (!isfinite(scale) || scale <= 0.f) return 0;
    float clamped = fminf(scale, 448.f);
    if (clamped < RGQ_UE4M3_MIN_NORMAL) {
        int sub = (int)roundf(clamped / RGQ_UE4M3_SUBNORMAL_STEP);
        if (sub <= 0) return 0;
        if (sub <= 7) return (uint8_t)sub;
        return 0x08;
    }
    int e2;
    frexpf(clamped, &e2);
    int exp_v = e2 - 1;
    float mant_f = ldexpf(clamped, -exp_v) - 1.f;
    int mant = (int)roundf(mant_f * 8.f);
    if (mant < 0) mant = 0;
    if (mant > 7) { mant = 0; exp_v += 1; }
    int biased = exp_v + 7;
    if (biased < 1) biased = 1;
    if (biased > 15) biased = 15;
    uint8_t byte = ((uint8_t)biased << 3) | (uint8_t)(mant & 0x07);
    return (byte == 0x7F) ? 0x7E : byte;
}

__device__ __forceinline__ float rgq_decode_ue4m3(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * RGQ_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
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

    if (E <= nthreads && nthreads <= 256) {
        constexpr int kW = 32;
        const int lane = tid & (kW - 1);
        const int warp = tid >> 5;

        if (tid < E) {
            float x = logits[(size_t)n * E + tid];
            if (softcap > 0.f) {
                x = softcap * tanhf(x / softcap);
            }
            if (mode == 1) {
                float s = 1.f / (1.f + expf(-x));
                raw[tid] = s;
                sel[tid] = s + (bias ? bias[tid] : 0.f);
            } else {
                raw[tid] = x;
                sel[tid] = x;
            }
        }
        __syncthreads();
        if (warp != 0) return;

        float my[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int e = lane + i * kW;
            my[i] = (e < E) ? sel[e] : -INFINITY;
        }

        for (int j = 0; j < K; ++j) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
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
                for (int i = 0; i < 8; ++i) {
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
    const int tx = 256;
    size_t shmem = (size_t)2 * E * sizeof(float);
    moe_route_topk_kernel<<<n_tokens, tx, shmem, s>>>(
        logits, bias, topk_ids, topk_weights,
        n_tokens, E, K, mode, softcap, norm_topk, routed_scaling,
        K, -1
    );
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
    const int tx = 256;
    size_t shmem = (size_t)2 * E * sizeof(float);
    moe_route_topk_kernel<<<n_tokens, tx, shmem, s>>>(
        logits, bias, topk_ids, topk_weights,
        n_tokens, E, K, mode, softcap, norm_topk, routed_scaling,
        K + 1, shared_tail_id
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

__global__ void moe_route_gather_quant_m1_kernel(
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

    if (tid < E) {
        float xv = logits[tid];
        if (softcap > 0.f) {
            xv = softcap * tanhf(xv / softcap);
        }
        if (mode == 1) {
            float s = 1.f / (1.f + expf(-xv));
            raw[tid] = s;
            sel[tid] = s + (bias ? bias[tid] : 0.f);
        } else {
            raw[tid] = xv;
            sel[tid] = xv;
        }
    }
    __syncthreads();

    if (warp == 0) {
        float my[8];
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int e = lane + i * kW;
            my[i] = (e < E) ? sel[e] : -INFINITY;
        }
        for (int j = 0; j < K; ++j) {
            float bv = -INFINITY;
            int bi = 0x7fffffff;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
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
                for (int i = 0; i < 8; ++i) {
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
                topk_ids[j] = win_ids[j];
                topk_weights[j] = win_w[j] * routed_scaling;
            }
            if (shared_tail_id >= 0) {
                topk_ids[K] = shared_tail_id;
                topk_weights[K] = 1.f;
            }
        }
    }
    __syncthreads();

    if (tid < tiles) {
        int id = (tid < K) ? win_ids[tid] : shared_tail_id;
        float g_gu = (id >= 0) ? globals_gu[id] : 0.f;
        float g_dn = (id >= 0) ? globals_dn[id] : 0.f;
        gu_mini[tid] = g_gu;
        dn_mini[tid] = g_dn;
        s_stored[tid] = (g_gu == 0.f || !isfinite(g_gu)) ? 1.f : g_gu;
    }
    __syncthreads();

    int groups = hidden / 16;
    for (int gidx = tid; gidx < tiles * groups; gidx += blockDim.x) {
        int t = gidx / groups;
        int kb = gidx % groups;
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v = __bfloat162float(x[kb * 16 + i]);
            vals[i] = v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }
        float stored = s_stored[t];
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        float stored_scale = stored * local_scale;
        uint8_t scale_byte = rgq_encode_ue4m3(stored_scale);
        float scale_decoded = rgq_decode_ue4m3(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);

        int row = t * min_tile;
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

extern "C" int nv_kernels_moe_route_gather_quant_m1(
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
    if (E <= 0 || K <= 0 || hidden <= 0 || min_tile <= 0) return -1;
    if (E > 256 || K + 1 > ROUTE_MAX_K || K > E) return -2;
    if ((hidden % 16) != 0 || (min_tile % 128) != 0) return -3;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int tx = 256;
    size_t shmem = (size_t)2 * E * sizeof(float);
    moe_route_gather_quant_m1_kernel<<<1, tx, shmem, s>>>(
        logits, bias,
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        globals_gu, globals_dn,
        topk_ids, topk_weights, gu_mini, dn_mini,
        x_fp4, x_sf,
        E, K, mode, softcap, norm_topk, routed_scaling,
        shared_tail_id, hidden, min_tile
    );
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
