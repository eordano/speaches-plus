#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>
#include "nv_kernels.h"
#include "nvk_grid.cuh"
#include "nvk_pdl.cuh"
#include "nvk_smem_optin.cuh"

namespace {

int q4exp_device_cc_major() {
    static const int major = [] {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 0;
        int v = 0;
        if (cudaDeviceGetAttribute(&v, cudaDevAttrComputeCapabilityMajor, dev) != cudaSuccess) return 0;
        return v;
    }();
    return major;
}

__device__ __forceinline__ float bf16_round(float x) {
    return __bfloat162float(__float2bfloat16(x));
}

__device__ __forceinline__ float ld_bf16(const uint16_t* p) {
    return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(p));
}

__device__ __forceinline__ void st_bf16(uint16_t* p, float v) {
    *reinterpret_cast<__nv_bfloat16*>(p) = __float2bfloat16(v);
}

__device__ __forceinline__ void unpack8_bf16(uint4 w, float* f) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float2 p = __bfloat1622float2(h[i]);
        f[2 * i] = p.x;
        f[2 * i + 1] = p.y;
    }
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

__device__ __forceinline__ float sigmoidf_(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ float fp8_e4m3_to_f32(uint8_t b) {
    __half_raw h = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    return __half2float(__half(h));
}

__global__ void q4exp_ngram_ids_kernel(
    const uint32_t* __restrict__ tokens, const uint32_t* __restrict__ history,
    const int64_t* __restrict__ multipliers, const int64_t* __restrict__ sizes,
    const int64_t* __restrict__ offsets, uint32_t eos, int T, int ctx, int heads_per_ngram,
    int64_t* __restrict__ out
) {
    NVK_PDL_PROLOG();
    int t = blockIdx.x;
    int h = threadIdx.x;
    int n_heads = ctx * heads_per_ngram;
    if (t >= T || h >= n_heads) {
        NVK_PDL_EPILOG();
        return;
    }
    int ngram = 2 + h / heads_per_ngram;
    int64_t mixed = 0;
    for (int s = 0; s < ngram; ++s) {
        int64_t tok;
        int src = t - s;
        bool reset = false;
        for (int j = t - s; j < t; ++j) {
            uint32_t v = (j >= 0) ? tokens[j] : history[ctx + j];
            if (v == eos) reset = true;
        }
        if (reset) tok = (int64_t)eos;
        else tok = (int64_t)((src >= 0) ? tokens[src] : history[ctx + src]);
        int64_t term = (int64_t)((uint64_t)tok * (uint64_t)multipliers[s]);
        mixed = (s == 0) ? term : (mixed ^ term);
    }
    int64_t size = sizes[h];
    int64_t r = mixed % size;
    if (r < 0) r += size;
    out[(size_t)t * n_heads + h] = r + offsets[h];
    NVK_PDL_EPILOG();
}

__global__ void q4exp_ple_gather_kernel(
    const uint8_t* __restrict__ table, const int64_t* __restrict__ ids, float scale,
    int heads, int hd, uint16_t* __restrict__ out
) {
    NVK_PDL_PROLOG();
    int t = blockIdx.x;
    int h = blockIdx.y;
    int64_t row = ids[(size_t)t * heads + h];
    const uint8_t* src = table + (size_t)row * hd;
    uint16_t* dst = out + ((size_t)t * heads + h) * hd;
    for (int i = threadIdx.x * 4; i < hd; i += blockDim.x * 4) {
        uint32_t w = *reinterpret_cast<const uint32_t*>(src + i);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float v = bf16_round(fp8_e4m3_to_f32((uint8_t)(w >> (8 * j))));
            st_bf16(dst + i + j, bf16_round(v * scale));
        }
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_hc_norm_kernel(
    const uint16_t* __restrict__ x, const float* __restrict__ norm_w, uint16_t* __restrict__ xn_out,
    int hh, int nb, float eps
) {
    __shared__ float red[8][32];
    __shared__ float rstd[8];
    int m = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = blockDim.x >> 5;
    int group = hh / nb;
    const uint16_t* xr = x + (size_t)m * hh;
    float ss[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) ss[g] = 0.f;
    int n8 = hh / 8;
    for (int c = tid; c < n8; c += blockDim.x) {
        uint4 v = *reinterpret_cast<const uint4*>(xr + (size_t)c * 8);
        float f[8];
        unpack8_bf16(v, f);
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += f[j] * f[j];
        int g = (c * 8) / group;
#pragma unroll
        for (int gg = 0; gg < 8; ++gg) if (gg == g) ss[gg] += acc;
    }
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = warp_sum(ss[g]);
        if (lane == 0) red[g][warp] = v;
    }
    __syncthreads();
    if (tid < nb) {
        float tot = 0.f;
        for (int w = 0; w < nwarps; ++w) tot += red[tid][w];
        rstd[tid] = rsqrtf(tot / (float)group + eps);
    }
    __syncthreads();
    for (int c = tid; c < n8; c += blockDim.x) {
        uint4 v = *reinterpret_cast<const uint4*>(xr + (size_t)c * 8);
        float f[8];
        unpack8_bf16(v, f);
        int g = (c * 8) / group;
        float r = rstd[g];
#pragma unroll
        for (int j = 0; j < 8; ++j) st_bf16(xn_out + (size_t)m * hh + (size_t)c * 8 + j, f[j] * r * norm_w[c * 8 + j]);
    }
}

__global__ void q4exp_hc_down_kernel(
    const uint16_t* __restrict__ xn, const uint16_t* __restrict__ w_down,
    const uint16_t* __restrict__ w_inject, float* __restrict__ lora_out, float* __restrict__ inject_out,
    int hh, int nb, int rank
) {
    __shared__ float red[32];
    int m = blockIdx.y;
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const uint16_t* wr = (row < rank) ? (w_down + (size_t)row * hh) : (w_inject + (size_t)(row - rank) * hh);
    const uint16_t* xr = xn + (size_t)m * hh;
    float acc = 0.f;
    for (int i = tid * 8; i < hh; i += blockDim.x * 8) {
        uint4 wv = *reinterpret_cast<const uint4*>(wr + i);
        uint4 xv = *reinterpret_cast<const uint4*>(xr + i);
        float wf[8];
        float xf[8];
        unpack8_bf16(wv, wf);
        unpack8_bf16(xv, xf);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += wf[j] * xf[j];
    }
    acc = warp_sum(acc);
    if (lane == 0) red[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float tot = 0.f;
        int nw = blockDim.x >> 5;
        for (int w = 0; w < nw; ++w) tot += red[w];
        float v = bf16_round(tot) / (float)nb;
        if (row < rank) {
            lora_out[(size_t)m * rank + row] = bf16_round(v * sigmoidf_(v));
        } else {
            inject_out[(size_t)m * nb + (row - rank)] = 2.f * sigmoidf_(v);
        }
    }
}

__global__ void q4exp_hc_lora_inject_kernel(
    const uint16_t* __restrict__ d, const uint16_t* __restrict__ inj, uint16_t* __restrict__ lora,
    float* __restrict__ inject, int t, int rank, int nb
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int n_lora = t * rank;
    int n_inj = inj ? t * nb : 0;
    if (i < n_lora) {
        float v = ld_bf16(d + i) / (float)nb;
        st_bf16(lora + i, v * sigmoidf_(v));
    } else if (i < n_lora + n_inj) {
        int j = i - n_lora;
        float v = ld_bf16(inj + j) / (float)nb;
        inject[j] = 2.f * sigmoidf_(v);
    }
}

__global__ void q4exp_hc_gate_mix_kernel(
    const uint16_t* __restrict__ xn, const uint16_t* __restrict__ g, uint16_t* __restrict__ mixed,
    int t, int h, int nb
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * h) return;
    int m = i / h;
    int j = i - m * h;
    float out = 0.f;
    for (int b = 0; b < nb; ++b) {
        size_t o = ((size_t)m * nb + b) * h + j;
        out += sigmoidf_(ld_bf16(g + o)) * ld_bf16(xn + o);
    }
    st_bf16(mixed + i, out / (float)nb);
}

__global__ void q4exp_hc_up_mix_kernel(
    const float* __restrict__ lora, const uint16_t* __restrict__ w_up,
    const uint16_t* __restrict__ xn, uint16_t* __restrict__ mixed,
    int h, int nb, int rank
) {
    extern __shared__ float smem[];
    float* ls = smem;
    int m = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int hh = nb * h;
    for (int i = tid; i < rank; i += blockDim.x) ls[i] = lora[(size_t)m * rank + i];
    __syncthreads();
    int j = blockIdx.x * (blockDim.x >> 5) + warp;
    if (j >= h) return;
    float out = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint16_t* wr = w_up + (size_t)(b * h + j) * rank;
        float acc = 0.f;
        for (int i = lane * 8; i < rank; i += 256) {
            uint4 wv = *reinterpret_cast<const uint4*>(wr + i);
            float wf[8];
            unpack8_bf16(wv, wf);
#pragma unroll
            for (int q = 0; q < 8; ++q) acc += wf[q] * ls[i + q];
        }
        acc = warp_sum(acc);
        float g = sigmoidf_(bf16_round(acc));
        out += g * ld_bf16(xn + (size_t)m * hh + b * h + j);
    }
    if (lane == 0) st_bf16(mixed + (size_t)m * h + j, out / (float)nb);
}

__global__ void q4exp_hc_combine_kernel(
    uint16_t* __restrict__ hc, const uint16_t* __restrict__ y, const float* __restrict__ inject,
    int h, int nb
) {
    int m = blockIdx.y;
    int hh = nb * h;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < hh; i += gridDim.x * blockDim.x) {
        int b = i / h;
        int j = i - b * h;
        float v = ld_bf16(hc + (size_t)m * hh + i) + ld_bf16(y + (size_t)m * h + j) * inject[(size_t)m * nb + b];
        st_bf16(hc + (size_t)m * hh + i, v);
    }
}

__global__ void q4exp_qsa_compress_kernel(
    const uint16_t* __restrict__ raw_k, int raw_cap, int b0, const float* __restrict__ k_norm_w,
    const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    uint16_t* __restrict__ out, int ratio, int hd, int rot, float eps
) {
    extern __shared__ float smem[];
    float* v = smem;
    __shared__ float red[8];
    int b = b0 + blockIdx.x;
    int d = threadIdx.x;
    float acc = 0.f;
    for (int r = 0; r < ratio; ++r) {
        size_t row = ((size_t)b * ratio + r) % (size_t)raw_cap;
        acc += ld_bf16(raw_k + row * hd + d);
    }
    float x = bf16_round(acc / (float)ratio);
    float ss = warp_sum(x * x);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int w = 0; w < (hd >> 5); ++w) tot += red[w];
    float rstd = rsqrtf(tot / (float)hd + eps);
    v[d] = bf16_round(x * rstd * k_norm_w[d]);
    __syncthreads();
    int half = rot / 2;
    float o;
    if (d < half) {
        float c = cos_t[(size_t)(b * ratio) * half + d];
        float s = sin_t[(size_t)(b * ratio) * half + d];
        o = v[d] * c - v[d + half] * s;
    } else if (d < rot) {
        float c = cos_t[(size_t)(b * ratio) * half + (d - half)];
        float s = sin_t[(size_t)(b * ratio) * half + (d - half)];
        o = v[d] * c + v[d - half] * s;
    } else {
        o = v[d];
    }
    st_bf16(out + (size_t)b * hd + d, o);
}

template <int R, int KB>
__global__ void q4exp_qsa_scores_rows_kernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ cmp_k,
    const int32_t* __restrict__ nb_per_row, int m_rows, int nh, int hd, int nb_stride, float scale, float* __restrict__ scores) {
    extern __shared__ float4 q4exp_scores_smem4[];
    float* smem = reinterpret_cast<float*>(q4exp_scores_smem4);
    int m0 = blockIdx.y * R;
    int rows = min(R, m_rows - m0);
    int cta_b0 = blockIdx.x * blockDim.x * KB;
    int nb_max = 0;
    for (int r = 0; r < rows; ++r) {
        int nb = nb_per_row[m0 + r];
        if (nb > nb_max) nb_max = nb;
    }
    if (cta_b0 >= nb_max || cta_b0 >= nb_stride) return;
    for (int i = threadIdx.x; i < rows * nh * hd; i += blockDim.x) smem[i] = ld_bf16(q + (size_t)m0 * nh * hd + i);
    __syncthreads();
    int b0 = (blockIdx.x * blockDim.x + threadIdx.x) * KB;
    int nb_lim = min(nb_max, nb_stride);
    if (b0 >= nb_lim) return;
    int nbr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) nbr[r] = (r < rows) ? min(nb_per_row[m0 + r], nb_stride) : 0;
    float acc[KB][R][4];
#pragma unroll
    for (int kb = 0; kb < KB; ++kb)
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int h = 0; h < 4; ++h) acc[kb][r][h] = 0.f;
    for (int i = 0; i < hd; i += 8) {
        float kf[KB][8];
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) {
            int b = b0 + kb;
            uint4 kv = (b < nb_lim) ? __ldg(reinterpret_cast<const uint4*>(cmp_k + (size_t)b * hd + i)) : make_uint4(0, 0, 0, 0);
            unpack8_bf16(kv, kf[kb]);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r < rows) {
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    if (h < nh) {
                        const float4* qh = reinterpret_cast<const float4*>(smem + (r * nh + h) * hd + i);
                        float4 qa = qh[0], qb = qh[1];
                        float qv[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
                        for (int kb = 0; kb < KB; ++kb) {
#pragma unroll
                            for (int j = 0; j < 8; ++j) acc[kb][r][h] += qv[j] * kf[kb][j];
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int kb = 0; kb < KB; ++kb) {
        int b = b0 + kb;
        if (b >= nb_lim) break;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r >= rows) break;
            if (b >= nbr[r]) continue;
            float sv = 0.f;
#pragma unroll
            for (int h = 0; h < 4; ++h) if (h < nh) sv += fmaxf(acc[kb][r][h], 0.f);
            scores[(size_t)(m0 + r) * nb_stride + b] = sv * scale;
        }
    }
}

constexpr int QSA_SCORES_THREADS = 128;
constexpr int QSA_SCORES_HALF_WARPS = QSA_SCORES_THREADS / 16;
constexpr int QSA_SCORES_SMALL_M_MAX = 8;

template <int R_MAX, int KB>
__global__ void __launch_bounds__(QSA_SCORES_THREADS) q4exp_qsa_scores_hd128_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ cmp_k,
    const int32_t* __restrict__ nb_per_row, int m_rows, int nh, int nb_stride, float scale,
    float* __restrict__ scores
) {
    extern __shared__ uint4 q4exp_scores_qsmem[];
    __shared__ int s_nb[R_MAX];
    int m0 = blockIdx.y * R_MAX;
    int rows = min(R_MAX, m_rows - m0);
    int tid = threadIdx.x;
    int cta_b0 = blockIdx.x * (QSA_SCORES_HALF_WARPS * KB);
    NVK_PDL_PROLOG();
    int nb_max = 0;
    for (int r = 0; r < rows; ++r) {
        int nb = nb_per_row[m0 + r];
        if (nb > nb_max) nb_max = nb;
    }
    if (cta_b0 >= nb_max || cta_b0 >= nb_stride) {
        NVK_PDL_EPILOG();
        return;
    }
    if (tid < rows) s_nb[tid] = nb_per_row[m0 + tid];
    int nb_lim = min(nb_max, nb_stride);
    int q_u4 = rows * nh * 16;
    const uint4* q4 = reinterpret_cast<const uint4*>(q + (size_t)m0 * nh * 128);
    for (int i = tid; i < q_u4; i += QSA_SCORES_THREADS) q4exp_scores_qsmem[i] = q4[i];
    __syncthreads();
    int hw = tid >> 4;
    int sublane = tid & 15;
    int b0 = cta_b0 + hw * KB;
    if (b0 >= nb_lim) {
        NVK_PDL_EPILOG();
        return;
    }
    unsigned hmask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
    float kf[KB][8];
#pragma unroll
    for (int kb = 0; kb < KB; ++kb) {
        int b = b0 + kb;
        uint4 kv = (b < nb_lim) ? __ldcs(reinterpret_cast<const uint4*>(cmp_k + (size_t)b * 128 + sublane * 8))
                                : make_uint4(0u, 0u, 0u, 0u);
        unpack8_bf16(kv, kf[kb]);
    }
    for (int r = 0; r < rows; ++r) {
        int nb_r = min(s_nb[r], nb_stride);
        if (b0 >= nb_r) continue;
        float s[KB];
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) s[kb] = 0.f;
        for (int h = 0; h < nh; ++h) {
            uint4 qv = q4exp_scores_qsmem[(r * nh + h) * 16 + sublane];
            float qf[8];
            unpack8_bf16(qv, qf);
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                float d = 0.f;
#pragma unroll
                for (int j = 0; j < 8; ++j) d += qf[j] * kf[kb][j];
#pragma unroll
                for (int o = 8; o > 0; o >>= 1) d += __shfl_xor_sync(hmask, d, o, 16);
                s[kb] += fmaxf(d, 0.f);
            }
        }
        if (sublane == 0) {
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                int b = b0 + kb;
                if (b < nb_r) scores[(size_t)(m0 + r) * nb_stride + b] = s[kb] * scale;
            }
        }
    }
    NVK_PDL_EPILOG();
}

template <int R_MAX>
__global__ void __launch_bounds__(QSA_SCORES_THREADS) q4exp_qsa_scores_gen_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ cmp_k,
    const int32_t* __restrict__ nb_per_row, int m_rows, int nh, int hd, int nb_stride, float scale,
    float* __restrict__ scores
) {
    extern __shared__ uint4 q4exp_scores_qsmem[];
    __shared__ int s_nb[R_MAX];
    int m0 = blockIdx.y * R_MAX;
    int rows = min(R_MAX, m_rows - m0);
    int tid = threadIdx.x;
    int cta_b0 = blockIdx.x * QSA_SCORES_HALF_WARPS;
    int nb_max = 0;
    for (int r = 0; r < rows; ++r) {
        int nb = nb_per_row[m0 + r];
        if (nb > nb_max) nb_max = nb;
    }
    if (cta_b0 >= nb_max || cta_b0 >= nb_stride) return;
    if (tid < rows) s_nb[tid] = nb_per_row[m0 + tid];
    int nb_lim = min(nb_max, nb_stride);
    int hd8 = hd >> 3;
    int q_u4 = rows * nh * hd8;
    const uint4* q4 = reinterpret_cast<const uint4*>(q + (size_t)m0 * nh * hd);
    for (int i = tid; i < q_u4; i += QSA_SCORES_THREADS) q4exp_scores_qsmem[i] = q4[i];
    __syncthreads();
    int hw = tid >> 4;
    int sublane = tid & 15;
    int b = cta_b0 + hw;
    if (b >= nb_lim) return;
    unsigned hmask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
    const uint4* k4 = reinterpret_cast<const uint4*>(cmp_k + (size_t)b * hd);
    for (int r = 0; r < rows; ++r) {
        int nb_r = min(s_nb[r], nb_stride);
        if (b >= nb_r) continue;
        float s = 0.f;
        for (int h = 0; h < nh; ++h) {
            float d = 0.f;
            const uint4* qh4 = q4exp_scores_qsmem + (r * nh + h) * hd8;
            for (int c = sublane; c < hd8; c += 16) {
                float qf[8];
                float kf[8];
                unpack8_bf16(qh4[c], qf);
                unpack8_bf16(__ldg(&k4[c]), kf);
#pragma unroll
                for (int j = 0; j < 8; ++j) d += qf[j] * kf[j];
            }
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) d += __shfl_xor_sync(hmask, d, o, 16);
            s += fmaxf(d, 0.f);
        }
        if (sublane == 0) scores[(size_t)(m0 + r) * nb_stride + b] = s * scale;
    }
}

__device__ __forceinline__ uint32_t ordered_key(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

constexpr int TOPK_THREADS = 1024;
constexpr int QSA_TOPK_CLUSTER_BELOW_16_ROWS_ONE_ROW_PER_BLOCK_STARVES_THE_GPU = 16;
constexpr int QSA_SCORE_ROWS_PER_TILE_16_EACH_KEY_BLOCK_IS_READ_ONCE_PER_TILE = 16;

__device__ int block_scan_inclusive(int v, int* warp_sums) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int w = (lane < (TOPK_THREADS >> 5)) ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffffu, w, o);
            if (lane >= o) w += y;
        }
        warp_sums[lane] = w;
    }
    __syncthreads();
    int offset = (warp > 0) ? warp_sums[warp - 1] : 0;
    __syncthreads();
    return x + offset;
}

__global__ void q4exp_qsa_topk_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nb_per_row, int k, int nb_stride,
    int32_t* __restrict__ out_blocks, int32_t* __restrict__ out_count
) {
    __shared__ int hist[256];
    __shared__ int warp_sums[32];
    __shared__ int s_total;
    __shared__ uint32_t s_prefix;
    __shared__ int s_need;
    __shared__ int s_run_gt;
    __shared__ int s_run_eq;
    int m = blockIdx.x;
    int nb = nb_per_row[m];
    const float* row = scores + (size_t)m * nb_stride;
    int32_t* outp = out_blocks + (size_t)m * k;
    if (nb <= k) {
        for (int i = threadIdx.x; i < nb; i += blockDim.x) outp[i] = i;
        if (threadIdx.x == 0) out_count[m] = nb;
        return;
    }
    uint32_t prefix = 0;
    int need = k;
    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - 8 * pass;
        uint32_t mask = (pass == 0) ? 0u : (0xffffffffu << (32 - 8 * pass));
        for (int i = threadIdx.x; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        for (int i = threadIdx.x; i < nb; i += blockDim.x) {
            uint32_t key = ordered_key(row[i]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 0xffu], 1);
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0;
            int bin = 255;
            for (; bin >= 0; --bin) {
                int c = hist[bin];
                if (acc + c >= need) break;
                acc += c;
            }
            if (bin < 0) bin = 0;
            s_prefix = prefix | ((uint32_t)bin << shift);
            s_need = need - acc;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        __syncthreads();
    }
    uint32_t threshold = prefix;
    int need_eq = need;
    if (threadIdx.x == 0) {
        s_run_gt = 0;
        s_run_eq = 0;
    }
    __syncthreads();
    for (int base = 0; base < nb; base += blockDim.x) {
        int i = base + threadIdx.x;
        int is_gt = 0;
        int is_eq = 0;
        if (i < nb) {
            uint32_t key = ordered_key(row[i]);
            is_gt = key > threshold;
            is_eq = key == threshold;
        }
        int gt_incl = block_scan_inclusive(is_gt, warp_sums);
        if (threadIdx.x == blockDim.x - 1) s_total = gt_incl;
        __syncthreads();
        int gt_total = s_total;
        __syncthreads();
        int eq_incl = block_scan_inclusive(is_eq, warp_sums);
        if (threadIdx.x == blockDim.x - 1) s_total = eq_incl;
        __syncthreads();
        int eq_total = s_total;
        int run_gt = s_run_gt;
        int run_eq = s_run_eq;
        if (is_gt) {
            int slot = run_gt + gt_incl - 1;
            outp[slot] = i;
        } else if (is_eq) {
            int eq_rank = run_eq + eq_incl - 1;
            if (eq_rank < need_eq) outp[(k - need_eq) + eq_rank] = i;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            s_run_gt = run_gt + gt_total;
            s_run_eq = run_eq + eq_total;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out_count[m] = k;
}

__device__ __forceinline__ int q4exp_topk_block_exclusive_scan(int v, int* warp_sums, int* total) {
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
    if (lane == 31) warp_sums[warp] = x;
    __syncthreads();
    int nwarps = blockDim.x >> 5;
    if (warp == 0) {
        int w = (lane < nwarps) ? warp_sums[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffffu, w, o); if (lane >= o) w += y; }
        warp_sums[lane] = w;
    }
    __syncthreads();
    int excl = x - v + (warp > 0 ? warp_sums[warp - 1] : 0);
    if (total) *total = warp_sums[nwarps - 1];
    __syncthreads();
    return excl;
}

constexpr int QSA_TOPK_CLUSTER_CTAS_PER_ROW = 8;
constexpr int QSA_TOPK_CLUSTER_THREADS = 256;
constexpr int QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS = 32;

// Thread-block clusters need sm_90+. The package also builds sm_89, so the cluster kernel
// compiles to a trap there and the launcher picks the one-block-per-row kernel below at runtime.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 900
#define Q4EXP_CLUSTER_DIMS_8
#define Q4EXP_HAS_CLUSTERS 0
#else
#define Q4EXP_CLUSTER_DIMS_8 __cluster_dims__(8, 1, 1)
#define Q4EXP_HAS_CLUSTERS 1
#endif

static_assert(QSA_TOPK_CLUSTER_CTAS_PER_ROW == 8, "q4exp_qsa_topk_cluster_kernel takes its cluster size from Q4EXP_CLUSTER_DIMS_8");

#if Q4EXP_HAS_CLUSTERS
__device__ __forceinline__ void q4exp_topk_cluster_sync() {
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ int q4exp_topk_remote_ld(const int* local_ptr, unsigned rank) {
    unsigned la = (unsigned)__cvta_generic_to_shared(local_ptr), ra;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(ra) : "r"(la), "r"(rank));
    int v;
    asm volatile("ld.shared::cluster.s32 %0, [%1];\n" : "=r"(v) : "r"(ra) : "memory");
    return v;
}

#endif

__device__ __forceinline__ void q4exp_warp_hist_add(int* hist, int bin, bool valid) {
    int key = valid ? bin : -1;
    unsigned m = __match_any_sync(0xffffffffu, key);
    if (valid && (__ffs(m) - 1) == (threadIdx.x & 31)) {
        atomicAdd(&hist[bin], __popc(m));
    }
}

__global__ void Q4EXP_CLUSTER_DIMS_8 __launch_bounds__(QSA_TOPK_CLUSTER_THREADS) q4exp_qsa_topk_cluster_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nb_per_row, int k, int nb_stride,
    int32_t* __restrict__ out_blocks, int32_t* __restrict__ out_count) {
#if Q4EXP_HAS_CLUSTERS
    __shared__ uint32_t s_keys[QSA_TOPK_CLUSTER_THREADS * QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS];
    __shared__ int lhist[2][256];
    __shared__ int warp_sums[32];
    __shared__ uint32_t s_prefix;
    __shared__ int s_need;
    __shared__ int s_bin_cnt;
    __shared__ int s_tot[2];
    NVK_PDL_PROLOG();
    unsigned rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(rank));
    int m = blockIdx.y;
    int tid = threadIdx.x;
    int nb = nb_per_row[m];
    const float* row = scores + (size_t)m * nb_stride;
    int32_t* outp = out_blocks + (size_t)m * k;
    if (nb <= k) {
        if (rank == 0) {
            for (int i = tid; i < nb; i += blockDim.x) outp[i] = i;
            if (tid == 0) out_count[m] = nb;
        }
        NVK_PDL_EPILOG();
        return;
    }
    int span = (nb + QSA_TOPK_CLUSTER_CTAS_PER_ROW - 1) / QSA_TOPK_CLUSTER_CTAS_PER_ROW;
    int base = (int)rank * span;
    int end = min(nb, base + span);
    int span_len = max(0, end - base);
    if ((((uintptr_t)(row + base)) & 15u) == 0u) {
        int n4 = span_len >> 2;
        const float4* row4 = reinterpret_cast<const float4*>(row + base);
        for (int i = tid; i < n4; i += QSA_TOPK_CLUSTER_THREADS) {
            float4 v = __ldg(&row4[i]);
            s_keys[i * 4 + 0] = ordered_key(v.x);
            s_keys[i * 4 + 1] = ordered_key(v.y);
            s_keys[i * 4 + 2] = ordered_key(v.z);
            s_keys[i * 4 + 3] = ordered_key(v.w);
        }
        int rem_base = n4 << 2;
        if (rem_base + tid < span_len) {
            s_keys[rem_base + tid] = ordered_key(__ldg(&row[base + rem_base + tid]));
        }
    } else {
        for (int i = tid; i < span_len; i += QSA_TOPK_CLUSTER_THREADS) {
            s_keys[i] = ordered_key(__ldg(&row[base + i]));
        }
    }
    lhist[0][tid] = 0;
    lhist[1][tid] = 0;
    __syncthreads();
    int per = (span + QSA_TOPK_CLUSTER_THREADS - 1) / QSA_TOPK_CLUSTER_THREADS;
    int lo_loc = tid * per;
    int lo = base + lo_loc;
    int cnt = max(0, min(per, end - lo));
    uint32_t key[QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS];
#pragma unroll
    for (int j = 0; j < QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS; ++j) {
        key[j] = (j < cnt) ? s_keys[lo_loc + j] : 0u;
    }
    uint32_t prefix = 0;
    int need = k;
    uint32_t final_mask = 0xffffffffu;
    for (int pass = 0; pass < 4; ++pass) {
        int cur = pass & 1;
        int shift = 24 - 8 * pass;
        uint32_t mask = (pass == 0) ? 0u : (0xffffffffu << (32 - 8 * pass));
#pragma unroll
        for (int j = 0; j < QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS; ++j) {
            bool valid = (j < cnt) && ((key[j] & mask) == prefix);
            q4exp_warp_hist_add(lhist[cur], (int)((key[j] >> shift) & 0xffu), valid);
        }
        q4exp_topk_cluster_sync();
        int bin = 255 - tid;
        int c = 0;
#pragma unroll
        for (int r = 0; r < QSA_TOPK_CLUSTER_CTAS_PER_ROW; ++r) c += q4exp_topk_remote_ld(&lhist[cur][bin], r);
        if (pass < 3) lhist[cur ^ 1][tid] = 0;
        int lane = tid & 31, w = tid >> 5;
        int x = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
        if (lane == 31) warp_sums[w] = x;
        __syncthreads();
        int add = 0;
        for (int q = 0; q < w; ++q) add += warp_sums[q];
        int incl = x + add;
        int excl = incl - c;
        if (excl < need && incl >= need) {
            s_prefix = prefix | ((uint32_t)bin << shift);
            s_need = need - excl;
            s_bin_cnt = c;
        } else if (tid == 255 && incl < need) {
            s_prefix = prefix;
            s_need = need - incl;
            s_bin_cnt = 0;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        int bin_cnt = s_bin_cnt;
        if (bin_cnt > 0 && bin_cnt <= need) {
            final_mask = 0xffffffffu << shift;
            break;
        }
    }
    uint32_t threshold = prefix;
    int need_eq = need;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS; ++j) {
        if (j < cnt) {
            uint32_t km = key[j] & final_mask;
            gt += km > threshold;
            eq += km == threshold;
        }
    }
    int gt_tot, eq_tot;
    int gt_off = q4exp_topk_block_exclusive_scan(gt, warp_sums, &gt_tot);
    int eq_off = q4exp_topk_block_exclusive_scan(eq, warp_sums, &eq_tot);
    if (tid == 0) { s_tot[0] = gt_tot; s_tot[1] = eq_tot; }
    q4exp_topk_cluster_sync();
    int gt_base = 0, eq_base = 0;
    for (unsigned r = 0; r < rank; ++r) { gt_base += q4exp_topk_remote_ld(&s_tot[0], r); eq_base += q4exp_topk_remote_ld(&s_tot[1], r); }
    gt_off += gt_base;
    eq_off += eq_base;
#pragma unroll
    for (int j = 0; j < QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS; ++j) {
        if (j < cnt) {
            uint32_t km = key[j] & final_mask;
            if (km > threshold) outp[gt_off++] = lo + j;
            else if (km == threshold) { if (eq_off < need_eq) outp[(k - need_eq) + eq_off] = lo + j; ++eq_off; }
        }
    }
    if (rank == 0 && tid == 0) out_count[m] = k;
    q4exp_topk_cluster_sync();
    NVK_PDL_EPILOG();
#else
    __trap();
#endif
}

__global__ void __launch_bounds__(1024) q4exp_qsa_topk_fast_stream_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nb_per_row, int k, int nb_stride,
    int32_t* __restrict__ out_blocks, int32_t* __restrict__ out_count
) {
    __shared__ int hist[4096];
    __shared__ int warp_sums[32];
    __shared__ uint32_t s_prefix;
    __shared__ int s_need;
    __shared__ int s_bin_cnt;
    __shared__ int s_out_cnt;

    NVK_PDL_PROLOG();

    int m = blockIdx.x;
    int tid = threadIdx.x;
    int nb = nb_per_row[m];
    const float* row = scores + (size_t)m * nb_stride;
    int32_t* outp = out_blocks + (size_t)m * k;
    if (nb <= k) {
        for (int i = tid; i < nb; i += 1024) outp[i] = i;
        if (tid == 0) out_count[m] = nb;
        NVK_PDL_EPILOG();
        return;
    }

    int nb_round = (nb + 1023) & ~1023;

    #pragma unroll
    for (int i = 0; i < 4; ++i) hist[tid + i * 1024] = 0;
    __syncthreads();
    for (int idx = tid; idx < nb_round; idx += 1024) {
        bool valid = idx < nb;
        uint32_t ky = valid ? ordered_key(__ldg(&row[idx])) : 0u;
        q4exp_warp_hist_add(hist, (int)(ky >> 20), valid);
    }
    __syncthreads();

    int need = k;
    int bin_hi = 4095 - tid * 4;
    int c0 = hist[bin_hi];
    int c1 = hist[bin_hi - 1];
    int c2 = hist[bin_hi - 2];
    int c3 = hist[bin_hi - 3];
    int gsum = c0 + c1 + c2 + c3;
    int gexcl = q4exp_topk_block_exclusive_scan(gsum, warp_sums, nullptr);
    if (gexcl < need && gexcl + gsum >= need) {
        int run = gexcl;
        int cs[4] = {c0, c1, c2, c3};
        for (int s = 0; s < 4; ++s) {
            if (run + cs[s] >= need) {
                s_prefix = (uint32_t)(bin_hi - s) << 20;
                s_need = need - run;
                s_bin_cnt = cs[s];
                break;
            }
            run += cs[s];
        }
    }
    __syncthreads();

    uint32_t prefix = s_prefix;
    need = s_need;
    uint32_t mask = 0xFFF00000u;
    int bin_cnt = s_bin_cnt;

    if (bin_cnt > need) {
        hist[tid] = 0;
        __syncthreads();
        for (int idx = tid; idx < nb_round; idx += 1024) {
            bool valid = idx < nb;
            uint32_t ky = valid ? ordered_key(__ldg(&row[idx])) : 0u;
            bool hit = valid && ((ky & mask) == prefix);
            q4exp_warp_hist_add(hist, (int)((ky >> 10) & 0x3FFu), hit);
        }
        __syncthreads();
        int b = 1023 - tid;
        int c = hist[b];
        int pexcl = q4exp_topk_block_exclusive_scan(c, warp_sums, nullptr);
        if (pexcl < need && pexcl + c >= need) {
            s_prefix = prefix | ((uint32_t)b << 10);
            s_need = need - pexcl;
            s_bin_cnt = c;
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        mask = 0xFFFFFC00u;
        bin_cnt = s_bin_cnt;

        if (bin_cnt > need) {
            hist[tid] = 0;
            __syncthreads();
            for (int idx = tid; idx < nb_round; idx += 1024) {
                bool valid = idx < nb;
                uint32_t ky = valid ? ordered_key(__ldg(&row[idx])) : 0u;
                bool hit = valid && ((ky & mask) == prefix);
                q4exp_warp_hist_add(hist, (int)(ky & 0x3FFu), hit);
            }
            __syncthreads();
            int rb = 1023 - tid;
            int rc = hist[rb];
            int rexcl = q4exp_topk_block_exclusive_scan(rc, warp_sums, nullptr);
            if (rexcl < need && rexcl + rc >= need) {
                s_prefix = prefix | (uint32_t)rb;
                s_need = need - rexcl;
                s_bin_cnt = rc;
            }
            __syncthreads();
            prefix = s_prefix;
            need = s_need;
            mask = 0xFFFFFFFFu;
            bin_cnt = s_bin_cnt;
        }
    }

    if (bin_cnt <= need) {
        if (tid == 0) s_out_cnt = 0;
        __syncthreads();
        for (int idx = tid; idx < nb; idx += 1024) {
            uint32_t km = ordered_key(__ldg(&row[idx])) & mask;
            if (km >= prefix) {
                int p = atomicAdd(&s_out_cnt, 1);
                if (p < k) outp[p] = idx;
            }
        }
        if (tid == 0) out_count[m] = k;
    } else {
        int per = (nb + 1023) / 1024;
        int lo = tid * per;
        int hi = min(nb, lo + per);
        int gt = 0, eq = 0;
        for (int idx = lo; idx < hi; ++idx) {
            uint32_t km = ordered_key(__ldg(&row[idx])) & mask;
            if (km > prefix) ++gt;
            else if (km == prefix) ++eq;
        }
        int gt_off = q4exp_topk_block_exclusive_scan(gt, warp_sums, nullptr);
        int eq_off = q4exp_topk_block_exclusive_scan(eq, warp_sums, nullptr);
        for (int idx = lo; idx < hi; ++idx) {
            uint32_t km = ordered_key(__ldg(&row[idx])) & mask;
            if (km > prefix) outp[gt_off++] = idx;
            else if (km == prefix) {
                if (eq_off < need) outp[(k - need) + eq_off] = idx;
                ++eq_off;
            }
        }
        if (tid == 0) out_count[m] = k;
    }
    NVK_PDL_EPILOG();
}

constexpr int QSA_TOPK_FAST_THREADS = 512;
constexpr int QSA_TOPK_FAST_MAX_KEYS_PER_THREAD = 16;

__global__ void __launch_bounds__(QSA_TOPK_FAST_THREADS) q4exp_qsa_topk_fast_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nb_per_row, int k, int nb_stride,
    int32_t* __restrict__ out_blocks, int32_t* __restrict__ out_count
) {
    __shared__ int hist[4096];
    __shared__ int warp_sums[32];
    __shared__ uint32_t s_prefix;
    __shared__ int s_need;
    __shared__ int s_bin_cnt;

    NVK_PDL_PROLOG();

    int m = blockIdx.x;
    int tid = threadIdx.x;
    int nb = nb_per_row[m];
    const float* row = scores + (size_t)m * nb_stride;
    int32_t* outp = out_blocks + (size_t)m * k;
    if (nb <= k) {
        for (int i = tid; i < nb; i += QSA_TOPK_FAST_THREADS) outp[i] = i;
        if (tid == 0) out_count[m] = nb;
        NVK_PDL_EPILOG();
        return;
    }

    uint32_t key[QSA_TOPK_FAST_MAX_KEYS_PER_THREAD];
    #pragma unroll
    for (int j = 0; j < QSA_TOPK_FAST_MAX_KEYS_PER_THREAD; ++j) {
        int idx = tid + j * QSA_TOPK_FAST_THREADS;
        key[j] = (idx < nb) ? ordered_key(__ldg(&row[idx])) : 0u;
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) hist[tid + i * QSA_TOPK_FAST_THREADS] = 0;
    __syncthreads();
    #pragma unroll
    for (int j = 0; j < QSA_TOPK_FAST_MAX_KEYS_PER_THREAD; ++j) {
        int idx = tid + j * QSA_TOPK_FAST_THREADS;
        q4exp_warp_hist_add(hist, (int)(key[j] >> 20), idx < nb);
    }
    __syncthreads();

    int need = k;
    int group_sum = 0;
    int bin_hi = 4095 - tid * 8;
    #pragma unroll
    for (int s = 0; s < 8; ++s) group_sum += hist[bin_hi - s];
    int group_excl = q4exp_topk_block_exclusive_scan(group_sum, warp_sums, nullptr);
    if (group_excl < need && group_excl + group_sum >= need) {
        int run = group_excl;
        for (int s = 0; s < 8; ++s) {
            int b = bin_hi - s;
            int c = hist[b];
            if (run + c >= need) {
                s_prefix = (uint32_t)b << 20;
                s_need = need - run;
                s_bin_cnt = c;
                break;
            }
            run += c;
        }
    }
    __syncthreads();

    uint32_t prefix = s_prefix;
    need = s_need;
    uint32_t mask = 0xFFF00000u;
    int bin_cnt = s_bin_cnt;

    if (bin_cnt > need) {
        hist[tid] = 0;
        hist[tid + 512] = 0;
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < QSA_TOPK_FAST_MAX_KEYS_PER_THREAD; ++j) {
            int idx = tid + j * QSA_TOPK_FAST_THREADS;
            bool hit = (idx < nb) && ((key[j] & mask) == prefix);
            q4exp_warp_hist_add(hist, (int)((key[j] >> 10) & 0x3FFu), hit);
        }
        __syncthreads();
        int b0 = 1023 - tid * 2;
        int c0 = hist[b0];
        int c1 = hist[b0 - 1];
        int psum = c0 + c1;
        int pexcl = q4exp_topk_block_exclusive_scan(psum, warp_sums, nullptr);
        if (pexcl < need && pexcl + psum >= need) {
            if (pexcl + c0 >= need) {
                s_prefix = prefix | ((uint32_t)b0 << 10);
                s_need = need - pexcl;
                s_bin_cnt = c0;
            } else {
                s_prefix = prefix | ((uint32_t)(b0 - 1) << 10);
                s_need = need - (pexcl + c0);
                s_bin_cnt = c1;
            }
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        mask = 0xFFFFFC00u;
        bin_cnt = s_bin_cnt;

        if (bin_cnt > need) {
            hist[tid] = 0;
            hist[tid + 512] = 0;
            __syncthreads();
            #pragma unroll
            for (int j = 0; j < QSA_TOPK_FAST_MAX_KEYS_PER_THREAD; ++j) {
                int idx = tid + j * QSA_TOPK_FAST_THREADS;
                bool hit = (idx < nb) && ((key[j] & mask) == prefix);
                q4exp_warp_hist_add(hist, (int)(key[j] & 0x3FFu), hit);
            }
            __syncthreads();
            int r0 = 1023 - tid * 2;
            int rc0 = hist[r0];
            int rc1 = hist[r0 - 1];
            int rsum = rc0 + rc1;
            int rexcl = q4exp_topk_block_exclusive_scan(rsum, warp_sums, nullptr);
            if (rexcl < need && rexcl + rsum >= need) {
                if (rexcl + rc0 >= need) {
                    s_prefix = prefix | (uint32_t)r0;
                    s_need = need - rexcl;
                    s_bin_cnt = rc0;
                } else {
                    s_prefix = prefix | (uint32_t)(r0 - 1);
                    s_need = need - (rexcl + rc0);
                    s_bin_cnt = rc1;
                }
            }
            __syncthreads();
            prefix = s_prefix;
            need = s_need;
            mask = 0xFFFFFFFFu;
        }
    }

    int gt = 0, eq = 0;
    #pragma unroll
    for (int j = 0; j < QSA_TOPK_FAST_MAX_KEYS_PER_THREAD; ++j) {
        int idx = tid + j * QSA_TOPK_FAST_THREADS;
        if (idx < nb) {
            uint32_t km = key[j] & mask;
            if (km > prefix) ++gt;
            else if (km == prefix) ++eq;
        }
    }
    int gt_off = q4exp_topk_block_exclusive_scan(gt, warp_sums, nullptr);
    #pragma unroll
    for (int j = 0; j < QSA_TOPK_FAST_MAX_KEYS_PER_THREAD; ++j) {
        int idx = tid + j * QSA_TOPK_FAST_THREADS;
        if (idx < nb && (key[j] & mask) > prefix) {
            outp[gt_off++] = idx;
        }
    }
    int eq_base = k - need;
    if (s_bin_cnt <= need) {
        int eq_off = q4exp_topk_block_exclusive_scan(eq, warp_sums, nullptr);
        #pragma unroll
        for (int j = 0; j < QSA_TOPK_FAST_MAX_KEYS_PER_THREAD; ++j) {
            int idx = tid + j * QSA_TOPK_FAST_THREADS;
            if (idx < nb && (key[j] & mask) == prefix) {
                if (eq_off < need) outp[eq_base + eq_off] = idx;
                ++eq_off;
            }
        }
    } else {
        int rem = need;
        int written = 0;
        #pragma unroll
        for (int j = 0; j < QSA_TOPK_FAST_MAX_KEYS_PER_THREAD; ++j) {
            if (rem <= 0) break;
            int idx = tid + j * QSA_TOPK_FAST_THREADS;
            int is_eq = (idx < nb && (key[j] & mask) == prefix) ? 1 : 0;
            int tot_j = 0;
            int off_j = q4exp_topk_block_exclusive_scan(is_eq, warp_sums, &tot_j);
            if (is_eq && off_j < rem) {
                outp[eq_base + written + off_j] = idx;
            }
            written += tot_j;
            rem -= tot_j;
        }
    }
    if (tid == 0) out_count[m] = k;
    NVK_PDL_EPILOG();
}

__global__ void q4exp_qsa_expand_kernel(
    const int32_t* __restrict__ blocks, const int32_t* __restrict__ block_count,
    const int32_t* __restrict__ positions, int ratio, int k, int width,
    int32_t* __restrict__ out_idx, int32_t* __restrict__ out_count
) {
    NVK_PDL_PROLOG();
    int m = blockIdx.y;
    int pos = positions[m];
    int nblk = block_count[m];
    int expanded = nblk * ratio;
    int tail_start = ((pos + 1) / ratio) * ratio;
    int tail_count = pos + 1 - tail_start;
    int total = expanded + tail_count;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < width; c += gridDim.x * blockDim.x) {
        int v = -1;
        if (c < expanded) {
            v = blocks[(size_t)m * k + c / ratio] * ratio + (c % ratio);
        } else if (c < total) {
            v = tail_start + (c - expanded);
        }
        out_idx[(size_t)m * width + c] = v;
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) out_count[m] = total;
    NVK_PDL_EPILOG();
}

constexpr int ATTN_HD = 256;
constexpr int ATTN_HEADS_PER_WARP = 1;
constexpr int ATTN_WARPS = 12;
constexpr int ATTN_TOKENS_LOADED_AHEAD_OF_THE_ONLINE_SOFTMAX_CHAIN = 12;

__global__ void __launch_bounds__(384, 2) q4exp_attn_rows_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const int32_t* __restrict__ idx,
    const int32_t* __restrict__ count, int nq, int nkv, int splits, int idx_stride, float scale,
    float* __restrict__ partial
) {
    __shared__ uint4 s_k[2][12][32];
    __shared__ uint4 s_v[2][12][32];
    __shared__ int s_valid[2][12];
    NVK_PDL_PROLOG();
    constexpr float kLog2e = 1.4426950408889634f;
    constexpr float kLn2 = 0.6931471805599453f;
    float scale_log2 = scale * kLog2e;
    int split = blockIdx.x;
    int g = blockIdx.y;
    int m = blockIdx.z;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int group = nq / nkv;
    int n = count[m];
    int per = (n + splits - 1) / splits;
    int start = split * per;
    int end = min(n, start + per);
    const int32_t* row_idx = idx ? (idx + (size_t)m * idx_stride) : nullptr;
    float qf[8];
    float acc[8];
    int head = g * group + warp;
    uint4 qv = *reinterpret_cast<const uint4*>(q + ((size_t)m * nq + head) * ATTN_HD + lane * 8);
    unpack8_bf16(qv, qf);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        qf[j] *= scale_log2;
        acc[j] = 0.f;
    }
    float mx = -INFINITY;
    float l = 0.f;
    if (start < end) {
        int t_first = start + warp;
        int tok_w = (t_first < end) ? (row_idx ? __ldg(&row_idx[t_first]) : t_first) : -1;
        if (tok_w >= 0) {
            size_t base_kv = ((size_t)tok_w * nkv + g) * ATTN_HD + lane * 8;
            s_k[0][warp][lane] = __ldg(reinterpret_cast<const uint4*>(k_cache + base_kv));
            s_v[0][warp][lane] = __ldg(reinterpret_cast<const uint4*>(v_cache + base_kv));
        }
        if (lane == 0) s_valid[0][warp] = (tok_w >= 0) ? 1 : 0;
        __syncthreads();
        int cur = 0;
        for (int t0 = start; t0 < end; t0 += 12) {
            int nxt_t0 = t0 + 12;
            int tok_n = -1;
            uint4 nk = make_uint4(0u, 0u, 0u, 0u);
            uint4 nv = make_uint4(0u, 0u, 0u, 0u);
            if (nxt_t0 < end) {
                int tn = nxt_t0 + warp;
                tok_n = (tn < end) ? (row_idx ? __ldg(&row_idx[tn]) : tn) : -1;
                if (tok_n >= 0) {
                    size_t base_kv = ((size_t)tok_n * nkv + g) * ATTN_HD + lane * 8;
                    nk = __ldg(reinterpret_cast<const uint4*>(k_cache + base_kv));
                    nv = __ldg(reinterpret_cast<const uint4*>(v_cache + base_kv));
                }
            }
#pragma unroll
            for (int b = 0; b < 12; ++b) {
                if (s_valid[cur][b]) {
                    float kf[8];
                    unpack8_bf16(s_k[cur][b][lane], kf);
                    float s = qf[0]*kf[0] + qf[1]*kf[1] + qf[2]*kf[2] + qf[3]*kf[3]
                            + qf[4]*kf[4] + qf[5]*kf[5] + qf[6]*kf[6] + qf[7]*kf[7];
                    s = warp_sum(s);
                    float vf[8];
                    unpack8_bf16(s_v[cur][b][lane], vf);
                    if (s > mx) {
                        float corr = exp2f(mx - s);
                        l = l * corr + 1.f;
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc[j] = acc[j] * corr + vf[j];
                        mx = s;
                    } else {
                        float p = exp2f(s - mx);
                        l += p;
#pragma unroll
                        for (int j = 0; j < 8; ++j) acc[j] += p * vf[j];
                    }
                }
            }
            if (nxt_t0 < end) {
                int nxt = cur ^ 1;
                if (tok_n >= 0) {
                    s_k[nxt][warp][lane] = nk;
                    s_v[nxt][warp][lane] = nv;
                }
                if (lane == 0) s_valid[nxt][warp] = (tok_n >= 0) ? 1 : 0;
                __syncthreads();
                cur = nxt;
            }
        }
    }
    float* dst = partial + (((size_t)m * nq + head) * splits + split) * (ATTN_HD + 2);
    float2* dst2 = reinterpret_cast<float2*>(dst + lane * 8);
    dst2[0] = make_float2(acc[0], acc[1]);
    dst2[1] = make_float2(acc[2], acc[3]);
    dst2[2] = make_float2(acc[4], acc[5]);
    dst2[3] = make_float2(acc[6], acc[7]);
    if (lane == 0) {
        dst[ATTN_HD] = (mx == -INFINITY) ? -INFINITY : (mx * kLn2);
        dst[ATTN_HD + 1] = l;
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_store_rows_at_kernel(
    const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, const int32_t* __restrict__ pos,
    int t, int row_elems
) {
    NVK_PDL_PROLOG();
    int row = blockIdx.y;
    int p = pos[0] + row;
    const uint4* s4 = reinterpret_cast<const uint4*>(src + (size_t)row * row_elems);
    uint4* d4 = reinterpret_cast<uint4*>(dst + (size_t)p * row_elems);
    int n4 = row_elems / 8;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) d4[i] = s4[i];
    NVK_PDL_EPILOG();
}

__global__ void q4exp_decode_meta_kernel(
    const int32_t* __restrict__ pos, int t, int ratio, int32_t* __restrict__ positions,
    int32_t* __restrict__ nb_per_row
) {
    NVK_PDL_PROLOG();
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < t) {
        int p = pos[0] + m;
        positions[m] = p;
        nb_per_row[m] = (p + 1) / ratio;
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_history_push_kernel(
    uint32_t* __restrict__ history, const uint32_t* __restrict__ tokens, int ctx, int t
) {
    NVK_PDL_PROLOG();
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        uint32_t tmp[8];
        for (int i = 0; i < ctx; ++i) {
            int src = t - ctx + i;
            tmp[i] = (src >= 0) ? tokens[src] : history[ctx + src];
        }
        for (int i = 0; i < ctx; ++i) history[i] = tmp[i];
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_qsa_compress_devpos_kernel(
    const uint16_t* __restrict__ raw_k, int raw_cap, const int32_t* __restrict__ pos, int t,
    const float* __restrict__ k_norm_w, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    uint16_t* __restrict__ out, int ratio, int hd, int rot, float eps
) {
    extern __shared__ float smem[];
    float* v = smem;
    __shared__ float red[8];
    NVK_PDL_PROLOG();
    int p0 = pos[0];
    int b0 = p0 / ratio;
    int b_end = (p0 + t) / ratio;
    int b = b0 + blockIdx.x;
    if (b >= b_end) {
        NVK_PDL_EPILOG();
        return;
    }
    int d = threadIdx.x;
    float acc = 0.f;
    for (int r = 0; r < ratio; ++r) {
        size_t row = ((size_t)b * ratio + r) % (size_t)raw_cap;
        acc += ld_bf16(raw_k + row * hd + d);
    }
    float x = bf16_round(acc / (float)ratio);
    float ss = warp_sum(x * x);
    if ((d & 31) == 0) red[d >> 5] = ss;
    __syncthreads();
    float tot = 0.f;
    for (int w = 0; w < (hd >> 5); ++w) tot += red[w];
    float rstd = rsqrtf(tot / (float)hd + eps);
    float vn = bf16_round(x * rstd * __ldg(&k_norm_w[d]));
    if (d < rot) v[d] = vn;
    __syncthreads();
    int half = rot / 2;
    float o = vn;
    if (d < half) {
        float c = __ldg(&cos_t[(size_t)(b * ratio) * half + d]);
        float s = __ldg(&sin_t[(size_t)(b * ratio) * half + d]);
        o = vn * c - v[d + half] * s;
    } else if (d < rot) {
        float c = __ldg(&cos_t[(size_t)(b * ratio) * half + (d - half)]);
        float s = __ldg(&sin_t[(size_t)(b * ratio) * half + (d - half)]);
        o = vn * c + v[d - half] * s;
    }
    st_bf16(out + (size_t)b * hd + d, o);
    NVK_PDL_EPILOG();
}

__global__ void q4exp_embed_hc_kernel(
    const uint16_t* __restrict__ embed, const uint32_t* __restrict__ tokens, int h, int nb,
    uint16_t* __restrict__ out
) {
    NVK_PDL_PROLOG();
    int t = blockIdx.x;
    uint32_t tok = tokens[t];
    const uint4* src = reinterpret_cast<const uint4*>(embed + (size_t)tok * h);
    int n4 = h / 8;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        uint4 v = src[i];
        for (int b = 0; b < nb; ++b) {
            reinterpret_cast<uint4*>(out + ((size_t)t * nb + b) * h)[i] = v;
        }
    }
    NVK_PDL_EPILOG();
}

__device__ __forceinline__ float q4exp_block_rstd_once(float x, int n, float eps, float* red) {
    float ss = warp_sum(x * x);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
    int nw = (blockDim.x + 31) >> 5;
    for (int w = 0; w < nw; ++w) tot += red[w];
    return rsqrtf(tot / (float)n + eps);
}

__device__ __forceinline__ float q4exp_block_rstd(float x, int n, float eps, float* red) {
    float r = q4exp_block_rstd_once(x, n, eps, red);
    __syncthreads();
    return r;
}

__global__ void q4exp_attn_prep_kernel(
    const uint16_t* __restrict__ q_raw, const uint16_t* __restrict__ k_raw,
    const uint16_t* __restrict__ v_raw, const uint16_t* __restrict__ idx_raw,
    const float* __restrict__ q_norm_w, const float* __restrict__ k_norm_w,
    const float* __restrict__ iq_norm_w, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    const int32_t* __restrict__ pos_dev, int nq, int nkv, int hd, int inh, int ih, int rot, float eps,
    uint16_t* __restrict__ q_out, uint16_t* __restrict__ gate_out, uint16_t* __restrict__ k_cache,
    uint16_t* __restrict__ v_cache, uint16_t* __restrict__ raw_idx_cache, int raw_idx_cap, uint16_t* __restrict__ iq_out
) {
    extern __shared__ float smem[];
    float* v = smem;
    __shared__ float red[32];
    NVK_PDL_PROLOG();
    int m = blockIdx.y;
    int task = blockIdx.x;
    int pos = pos_dev[0] + m;
    int d = threadIdx.x;
    int half = rot >> 1;
    if (task < nq) {
        int h = task;
        const uint16_t* src = q_raw + ((size_t)m * nq + h) * 2 * hd;
        float x = ld_bf16(src + d);
        uint16_t gval = src[hd + d];
        float rstd = q4exp_block_rstd_once(x, hd, eps, red);
        float vn = bf16_round(x * rstd * __ldg(&q_norm_w[d]));
        if (d < rot) v[d] = vn;
        __syncthreads();
        float o = vn;
        if (d < half) {
            float c = __ldg(&cos_t[(size_t)pos * half + d]);
            float s = __ldg(&sin_t[(size_t)pos * half + d]);
            o = vn * c - v[d + half] * s;
        } else if (d < rot) {
            float c = __ldg(&cos_t[(size_t)pos * half + (d - half)]);
            float s = __ldg(&sin_t[(size_t)pos * half + (d - half)]);
            o = vn * c + v[d - half] * s;
        }
        st_bf16(q_out + ((size_t)m * nq + h) * hd + d, o);
        gate_out[((size_t)m * nq + h) * hd + d] = gval;
        NVK_PDL_EPILOG();
        return;
    }
    task -= nq;
    if (task < nkv) {
        int h = task;
        float x = ld_bf16(k_raw + ((size_t)m * nkv + h) * hd + d);
        uint16_t vval = v_raw[((size_t)m * nkv + h) * hd + d];
        float rstd = q4exp_block_rstd_once(x, hd, eps, red);
        float vn = bf16_round(x * rstd * __ldg(&k_norm_w[d]));
        if (d < rot) v[d] = vn;
        __syncthreads();
        float o = vn;
        if (d < half) {
            float c = __ldg(&cos_t[(size_t)pos * half + d]);
            float s = __ldg(&sin_t[(size_t)pos * half + d]);
            o = vn * c - v[d + half] * s;
        } else if (d < rot) {
            float c = __ldg(&cos_t[(size_t)pos * half + (d - half)]);
            float s = __ldg(&sin_t[(size_t)pos * half + (d - half)]);
            o = vn * c + v[d - half] * s;
        }
        st_bf16(k_cache + ((size_t)pos * nkv + h) * hd + d, o);
        v_cache[((size_t)pos * nkv + h) * hd + d] = vval;
        NVK_PDL_EPILOG();
        return;
    }
    task -= nkv;
    if (task < inh) {
        int h = task;
        float vn = 0.f;
        if (d < ih) {
            float x = ld_bf16(idx_raw + (size_t)m * (inh + 1) * ih + h * ih + d);
            float rstd = q4exp_block_rstd_once(x, ih, eps, red);
            vn = bf16_round(x * rstd * __ldg(&iq_norm_w[d]));
            if (d < rot) v[d] = vn;
        } else {
            q4exp_block_rstd_once(0.f, ih, eps, red);
        }
        __syncthreads();
        if (d < ih) {
            float o = vn;
            if (d < half) {
                float c = __ldg(&cos_t[(size_t)pos * half + d]);
                float s = __ldg(&sin_t[(size_t)pos * half + d]);
                o = vn * c - v[d + half] * s;
            } else if (d < rot) {
                float c = __ldg(&cos_t[(size_t)pos * half + (d - half)]);
                float s = __ldg(&sin_t[(size_t)pos * half + (d - half)]);
                o = vn * c + v[d - half] * s;
            }
            st_bf16(iq_out + ((size_t)m * inh + h) * ih + d, o);
        }
        NVK_PDL_EPILOG();
        return;
    }
    if (d < ih) {
        size_t slot = (size_t)pos % (size_t)raw_idx_cap;
        raw_idx_cache[slot * ih + d] = idx_raw[(size_t)m * (inh + 1) * ih + inh * ih + d];
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_ple_gate_kernel(
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ value, const uint16_t* __restrict__ hc,
    const float* __restrict__ norm_key_w, const float* __restrict__ norm_query_w,
    const float* __restrict__ norm_conv_w, int h, int nb, float eps,
    uint16_t* __restrict__ gv_out, float* __restrict__ gvn_out
) {
    __shared__ float red[32];
    __shared__ float s_gate;
    NVK_PDL_PROLOG();
    int t = blockIdx.y;
    int b = blockIdx.x;
    int hh = nb * h;
    size_t base = (size_t)t * hh + (size_t)b * h;
    float ssk = 0.f, ssq = 0.f;
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
        float k = ld_bf16(key + base + i);
        float q = ld_bf16(hc + base + i);
        ssk += k * k;
        ssq += q * q;
    }
    float rk = q4exp_block_rstd(sqrtf(ssk), 1, 0.f, red);
    float rq = q4exp_block_rstd(sqrtf(ssq), 1, 0.f, red);
    rk = rsqrtf((1.f / (rk * rk)) / (float)h + eps);
    rq = rsqrtf((1.f / (rq * rq)) / (float)h + eps);
    float dot = 0.f;
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
        float kn = bf16_round(ld_bf16(key + base + i) * rk * __ldg(&norm_key_w[b * h + i]));
        float qn = bf16_round(ld_bf16(hc + base + i) * rq * __ldg(&norm_query_w[b * h + i]));
        dot += kn * qn;
    }
    dot = warp_sum(dot);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = dot;
    __syncthreads();
    if (threadIdx.x == 0) {
        float g = 0.f;
        int nw = (blockDim.x + 31) >> 5;
        for (int w = 0; w < nw; ++w) g += red[w];
        g = g / sqrtf((float)h);
        float mag = sqrtf(fmaxf(fabsf(g), 1e-6f));
        float sg = (g > 0.f) ? mag : ((g < 0.f) ? -mag : 0.f);
        s_gate = sigmoidf_(sg);
    }
    __syncthreads();
    float sig = s_gate;
    float ssg = 0.f;
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
        float gvv = bf16_round(sig * ld_bf16(value + (size_t)t * h + i));
        st_bf16(gv_out + base + i, gvv);
        ssg += gvv * gvv;
    }
    float rg = q4exp_block_rstd(sqrtf(ssg), 1, 0.f, red);
    rg = rsqrtf((1.f / (rg * rg)) / (float)h + eps);
    for (int i = threadIdx.x; i < h; i += blockDim.x) {
        float gvv = ld_bf16(gv_out + base + i);
        gvn_out[base + i] = bf16_round(gvv * rg * __ldg(&norm_conv_w[b * h + i]));
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_ple_conv_kernel(
    const float* __restrict__ gvn, const uint16_t* __restrict__ gv, float* __restrict__ window,
    const float* __restrict__ conv_w, int t, int hh, int k, int dil, int win, int update_window,
    uint16_t* __restrict__ out
) {
    NVK_PDL_PROLOG();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < hh) {
        float full[64];
        for (int i = 0; i < win; ++i) full[i] = window[(size_t)i * hh + c];
        for (int r = 0; r < t; ++r) full[win + r] = gvn[(size_t)r * hh + c];
        for (int r = 0; r < t; ++r) {
            float acc = 0.f;
            for (int tap = 0; tap < k; ++tap) acc += __ldg(&conv_w[(size_t)c * k + tap]) * full[r + tap * dil];
            float o = acc * sigmoidf_(acc);
            st_bf16(out + (size_t)r * hh + c, ld_bf16(gv + (size_t)r * hh + c) + o);
        }
        if (update_window) {
            for (int i = 0; i < win; ++i) window[(size_t)i * hh + c] = full[t + i];
        }
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_ple_window_advance_kernel(
    const float* __restrict__ gvn, float* __restrict__ window, int t, int hh, int win
) {
    NVK_PDL_PROLOG();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < hh) {
        float full[64];
        for (int i = 0; i < win; ++i) full[i] = window[(size_t)i * hh + c];
        for (int r = 0; r < t; ++r) full[win + r] = gvn[(size_t)r * hh + c];
        for (int i = 0; i < win; ++i) window[(size_t)i * hh + c] = full[t + i];
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_rmsnorm_rows_kernel(
    const uint16_t* __restrict__ x, const float* __restrict__ w, uint16_t* __restrict__ out, int n,
    float eps
) {
    __shared__ float red[32];
    NVK_PDL_PROLOG();
    int m = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const uint16_t* xr = x + (size_t)m * n;
    uint16_t* orow = out + (size_t)m * n;
    float ss = 0.f;
    for (int i = tid; i < n; i += blockDim.x) {
        float v = ld_bf16(xr + i);
        ss += v * v;
    }
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
    int nw = blockDim.x >> 5;
    for (int i = 0; i < nw; ++i) tot += red[i];
    float rstd = rsqrtf(tot / (float)n + eps);
    for (int i = tid; i < n; i += blockDim.x) st_bf16(orow + i, ld_bf16(xr + i) * rstd * __ldg(&w[i]));
    NVK_PDL_EPILOG();
}

}  // namespace

extern "C" int nv_kernels_q4exp_ngram_ids(
    void* stream, const uint32_t* tokens, const uint32_t* history, const int64_t* multipliers,
    const int64_t* sizes, const int64_t* offsets, uint32_t eos, int t, int ctx, int heads_per_ngram,
    int64_t* out
) {
    if (t <= 0) return 0;
    int n_heads = ctx * heads_per_ngram;
    if (n_heads > 1024) return -2;
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)t), dim3((unsigned)n_heads), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_ngram_ids_kernel,
            tokens, history, multipliers, sizes, offsets, eos, t, ctx, heads_per_ngram, out);
    } else {
        q4exp_ngram_ids_kernel<<<t, n_heads, 0, s>>>(
            tokens, history, multipliers, sizes, offsets, eos, t, ctx, heads_per_ngram, out);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_ple_gather_bf16(
    void* stream, const uint8_t* table, const int64_t* ids, float scale, int t, int heads, int hd,
    uint16_t* out
) {
    if (t <= 0) return 0;
    if (hd % 4 != 0) return -2;
    dim3 grid(t, heads);
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(64), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_ple_gather_kernel, table, ids, scale, heads, hd, out);
    } else {
        q4exp_ple_gather_kernel<<<grid, 64, 0, s>>>(table, ids, scale, heads, hd, out);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_hc_norm_rows(
    void* stream, const uint16_t* x, const float* norm_w, uint16_t* xn_out, int m, int hh, int nb, float eps
) {
    if (m <= 0) return 0;
    if (hh % 2048 != 0 || hh % nb != 0 || nb > 8) return -2;
    q4exp_hc_norm_kernel<<<m, 256, 0, (cudaStream_t)stream>>>(x, norm_w, xn_out, hh, nb, eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_hc_lora_inject(
    void* stream, const uint16_t* d, const uint16_t* inj, uint16_t* lora, float* inject, int t, int rank, int nb
) {
    if (t <= 0) return 0;
    int n = t * rank + (inj ? t * nb : 0);
    q4exp_hc_lora_inject_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(d, inj, lora, inject, t, rank, nb);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_hc_gate_mix(
    void* stream, const uint16_t* xn, const uint16_t* g, uint16_t* mixed, int t, int h, int nb
) {
    if (t <= 0) return 0;
    int n = t * h;
    q4exp_hc_gate_mix_kernel<<<(n + 255) / 256, 256, 0, (cudaStream_t)stream>>>(xn, g, mixed, t, h, nb);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_hc_norm_down(
    void* stream, const uint16_t* x, const float* norm_w, const uint16_t* w_down,
    const uint16_t* w_inject, uint16_t* xn_out, float* lora_out, float* inject_out, int m, int hh,
    int nb, int rank, float eps
) {
    if (m <= 0) return 0;
    if (hh % 2048 != 0 || hh % nb != 0 || nb > 8) return -2;
    q4exp_hc_norm_kernel<<<m, 256, 0, (cudaStream_t)stream>>>(x, norm_w, xn_out, hh, nb, eps);
    int rows = rank + (w_inject ? nb : 0);
    dim3 grid(rows, m);
    q4exp_hc_down_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(xn_out, w_down, w_inject, lora_out, inject_out, hh, nb, rank);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_hc_down(
    void* stream, const uint16_t* xn, const uint16_t* w_down, const uint16_t* w_inject, float* lora_out,
    float* inject_out, int m, int hh, int nb, int rank
) {
    if (m <= 0) return 0;
    if (hh % 2048 != 0 || hh % nb != 0 || nb > 8) return -2;
    int rows = rank + (w_inject ? nb : 0);
    dim3 grid(rows, m);
    q4exp_hc_down_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(xn, w_down, w_inject, lora_out, inject_out, hh, nb, rank);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_hc_up_mix(
    void* stream, const float* lora, const uint16_t* w_up, const uint16_t* xn, uint16_t* mixed,
    int m, int h, int nb, int rank
) {
    if (m <= 0) return 0;
    if (rank % 8 != 0) return -2;
    dim3 grid((h + 7) / 8, m);
    size_t smem = (size_t)rank * sizeof(float);
    q4exp_hc_up_mix_kernel<<<grid, 256, smem, (cudaStream_t)stream>>>(lora, w_up, xn, mixed, h, nb, rank);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_hc_combine(
    void* stream, uint16_t* hc, const uint16_t* y, const float* inject, int m, int h, int nb
) {
    if (m <= 0) return 0;
    dim3 grid((nb * h + 255) / 256, m);
    q4exp_hc_combine_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(hc, y, inject, h, nb);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_qsa_compress(
    void* stream, const uint16_t* raw_k, int raw_cap, int b0, int nblocks, const float* k_norm_w,
    const float* cos_t, const float* sin_t, uint16_t* out, int ratio, int hd, int rot, float eps
) {
    if (nblocks <= 0) return 0;
    if (raw_cap <= 0 || hd % 32 != 0 || hd > 1024 || rot % 2 != 0 || rot > hd) return -2;
    size_t smem = (size_t)hd * sizeof(float);
    q4exp_qsa_compress_kernel<<<nblocks, hd, smem, (cudaStream_t)stream>>>(
        raw_k, raw_cap, b0, k_norm_w, cos_t, sin_t, out, ratio, hd, rot, eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_qsa_scores(
    void* stream, const uint16_t* q, const uint16_t* cmp_k, const int32_t* nb_per_row, int m, int nh,
    int hd, int nb_stride, float scale, float* scores
) {
    if (m <= 0 || nb_stride <= 0) return 0;
    if (nh <= 0 || nh > 4 || hd <= 0 || hd % 8 != 0) return -2;
    if (m >= QSA_SCORE_ROWS_PER_TILE_16_EACH_KEY_BLOCK_IS_READ_ONCE_PER_TILE) {
        constexpr int R = QSA_SCORE_ROWS_PER_TILE_16_EACH_KEY_BLOCK_IS_READ_ONCE_PER_TILE;
        int gy = (m + R - 1) / R;
        if (gy > NVK_MAX_GRID_Y) return NVK_ERR_GRID_AXIS;
        dim3 grid((nb_stride + 511) / 512, gy);
        size_t smem = (size_t)R * nh * hd * sizeof(float);
        q4exp_qsa_scores_rows_kernel<R, 2><<<grid, 256, smem, (cudaStream_t)stream>>>(
            q, cmp_k, nb_per_row, m, nh, hd, nb_stride, scale, scores);
        return (int)cudaGetLastError();
    }
    if (hd == 128) {
        constexpr int R = QSA_SCORES_SMALL_M_MAX;
        constexpr int KB = 4;
        constexpr int BPC = QSA_SCORES_HALF_WARPS * KB;
        int gy = (m + R - 1) / R;
        if (gy > NVK_MAX_GRID_Y) return NVK_ERR_GRID_AXIS;
        dim3 grid((nb_stride + BPC - 1) / BPC, gy);
        size_t smem = (size_t)min(m, R) * nh * 128 * sizeof(uint16_t);
        cudaStream_t s = (cudaStream_t)stream;
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(cfg, grid, dim3(QSA_SCORES_THREADS), smem, s);
            cudaLaunchKernelEx(&cfg, q4exp_qsa_scores_hd128_kernel<R, KB>,
                q, cmp_k, nb_per_row, m, nh, nb_stride, scale, scores);
        } else {
            q4exp_qsa_scores_hd128_kernel<R, KB><<<grid, QSA_SCORES_THREADS, smem, s>>>(
                q, cmp_k, nb_per_row, m, nh, nb_stride, scale, scores);
        }
        return (int)cudaGetLastError();
    }
    int r_tile = (m <= QSA_SCORES_SMALL_M_MAX) ? m : QSA_SCORES_SMALL_M_MAX;
    size_t smem = (size_t)r_tile * nh * hd * sizeof(uint16_t);
    if (smem <= 48 * 1024) {
        int gy = (m + QSA_SCORES_SMALL_M_MAX - 1) / QSA_SCORES_SMALL_M_MAX;
        if (gy > NVK_MAX_GRID_Y) return NVK_ERR_GRID_AXIS;
        dim3 grid((nb_stride + QSA_SCORES_HALF_WARPS - 1) / QSA_SCORES_HALF_WARPS, gy);
        q4exp_qsa_scores_gen_kernel<QSA_SCORES_SMALL_M_MAX><<<grid, QSA_SCORES_THREADS, smem, (cudaStream_t)stream>>>(
            q, cmp_k, nb_per_row, m, nh, hd, nb_stride, scale, scores);
        return (int)cudaGetLastError();
    }
    size_t smem1 = (size_t)nh * hd * sizeof(uint16_t);
    if (smem1 > 48 * 1024) return -2;
    if (m > NVK_MAX_GRID_Y) return NVK_ERR_GRID_AXIS;
    dim3 grid((nb_stride + QSA_SCORES_HALF_WARPS - 1) / QSA_SCORES_HALF_WARPS, m);
    q4exp_qsa_scores_gen_kernel<1><<<grid, QSA_SCORES_THREADS, smem1, (cudaStream_t)stream>>>(
        q, cmp_k, nb_per_row, m, nh, hd, nb_stride, scale, scores);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_qsa_topk(
    void* stream, const float* scores, const int32_t* nb_per_row, int m, int k, int nb_stride,
    int32_t* out_blocks, int32_t* out_count
) {
    if (m <= 0) return 0;
    const char* no_fast = getenv("NV_Q4EXP_TOPK_CLUSTER");
    bool use_fast = !(no_fast && no_fast[0] == '1' && no_fast[1] == '\0');
    if (use_fast && nb_stride <= QSA_TOPK_FAST_THREADS * QSA_TOPK_FAST_MAX_KEYS_PER_THREAD) {
        cudaStream_t s = (cudaStream_t)stream;
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(cfg, dim3((unsigned)m), dim3(QSA_TOPK_FAST_THREADS), 0, s);
            cudaLaunchKernelEx(&cfg, q4exp_qsa_topk_fast_kernel,
                scores, nb_per_row, k, nb_stride, out_blocks, out_count);
        } else {
            q4exp_qsa_topk_fast_kernel<<<m, QSA_TOPK_FAST_THREADS, 0, s>>>(
                scores, nb_per_row, k, nb_stride, out_blocks, out_count);
        }
        return (int)cudaGetLastError();
    }
    if (use_fast && nb_stride <= 16384) {
        cudaStream_t s = (cudaStream_t)stream;
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(cfg, dim3((unsigned)m), dim3(1024), 0, s);
            cudaLaunchKernelEx(&cfg, q4exp_qsa_topk_fast_stream_kernel,
                scores, nb_per_row, k, nb_stride, out_blocks, out_count);
        } else {
            q4exp_qsa_topk_fast_stream_kernel<<<m, 1024, 0, s>>>(
                scores, nb_per_row, k, nb_stride, out_blocks, out_count);
        }
        return (int)cudaGetLastError();
    }
    if (q4exp_device_cc_major() >= 9 && m < QSA_TOPK_CLUSTER_BELOW_16_ROWS_ONE_ROW_PER_BLOCK_STARVES_THE_GPU
        && nb_stride <= QSA_TOPK_CLUSTER_CTAS_PER_ROW * QSA_TOPK_CLUSTER_THREADS * QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS) {
        cudaStream_t s = (cudaStream_t)stream;
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(cfg, dim3(QSA_TOPK_CLUSTER_CTAS_PER_ROW, m), dim3(QSA_TOPK_CLUSTER_THREADS), 0, s);
            cudaLaunchKernelEx(&cfg, q4exp_qsa_topk_cluster_kernel,
                scores, nb_per_row, k, nb_stride, out_blocks, out_count);
        } else {
            q4exp_qsa_topk_cluster_kernel<<<dim3(QSA_TOPK_CLUSTER_CTAS_PER_ROW, m), QSA_TOPK_CLUSTER_THREADS, 0, s>>>(
                scores, nb_per_row, k, nb_stride, out_blocks, out_count);
        }
        return (int)cudaGetLastError();
    }
    q4exp_qsa_topk_kernel<<<m, TOPK_THREADS, 0, (cudaStream_t)stream>>>(
        scores, nb_per_row, k, nb_stride, out_blocks, out_count);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_qsa_expand(
    void* stream, const int32_t* blocks, const int32_t* block_count, const int32_t* positions, int m,
    int ratio, int k, int width, int32_t* out_idx, int32_t* out_count
) {
    if (m <= 0) return 0;
    dim3 grid((width + 255) / 256, m);
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_qsa_expand_kernel,
            blocks, block_count, positions, ratio, k, width, out_idx, out_count);
    } else {
        q4exp_qsa_expand_kernel<<<grid, 256, 0, s>>>(
            blocks, block_count, positions, ratio, k, width, out_idx, out_count);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_attn_rows(
    void* stream, const uint16_t* q, const uint16_t* k_cache, const uint16_t* v_cache,
    const int32_t* idx, const int32_t* count, int m, int nq, int nkv, int hd, int splits,
    int idx_stride, float scale, float* partial
) {
    if (m <= 0) return 0;
    if (hd != ATTN_HD || nq != nkv * ATTN_HEADS_PER_WARP * ATTN_WARPS || splits <= 0) return -2;
    if (nkv > NVK_MAX_GRID_Y || m > NVK_MAX_GRID_Z) return NVK_ERR_GRID_AXIS;
    dim3 grid(splits, nkv, m);
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(ATTN_WARPS * 32), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_attn_rows_kernel,
            q, k_cache, v_cache, idx, count, nq, nkv, splits, idx_stride, scale, partial);
    } else {
        q4exp_attn_rows_kernel<<<grid, ATTN_WARPS * 32, 0, s>>>(
            q, k_cache, v_cache, idx, count, nq, nkv, splits, idx_stride, scale, partial);
    }
    return (int)cudaGetLastError();
}

extern "C" void nvk_q4exp_get_i8_scratch(int** buf, float** sc);

__global__ void __launch_bounds__(128) q4exp_attn_combine_f2_kernel(
    const float* __restrict__ partial, const uint16_t* __restrict__ gate,
    int nq, int splits, uint16_t* __restrict__ out
) {
    __shared__ float s_w[128];
    __shared__ float s_red[4];
    __shared__ float s_red2[4];
    NVK_PDL_PROLOG();
    int head = blockIdx.x;
    int m = blockIdx.y;
    int sub = threadIdx.x;
    int d0 = sub * 2;
    const float* base = partial + ((size_t)m * nq + head) * splits * (ATTN_HD + 2);
    float num0 = 0.f;
    float num1 = 0.f;
    float inv_den = 0.f;
    if (splits <= 128) {
        int lane = sub & 31;
        int warp = sub >> 5;
        float sm = (sub < splits) ? __ldg(&base[(size_t)sub * (ATTN_HD + 2) + ATTN_HD]) : -INFINITY;
        float sl = (sub < splits) ? __ldg(&base[(size_t)sub * (ATTN_HD + 2) + ATTN_HD + 1]) : 0.f;
        float wm = warp_max(sm);
        if (lane == 0) s_red[warp] = wm;
        __syncthreads();
        float gm = fmaxf(fmaxf(s_red[0], s_red[1]), fmaxf(s_red[2], s_red[3]));
        float w = (sub < splits && sm != -INFINITY) ? expf(sm - gm) : 0.f;
        s_w[sub] = w;
        float wd = warp_sum(w * sl);
        if (lane == 0) s_red2[warp] = wd;
        __syncthreads();
        float den = s_red2[0] + s_red2[1] + s_red2[2] + s_red2[3];
        inv_den = (den > 0.f) ? (1.f / den) : 0.f;
        for (int s = 0; s < splits; ++s) {
            float ws = s_w[s];
            float2 pv = __ldg(reinterpret_cast<const float2*>(base + (size_t)s * (ATTN_HD + 2) + d0));
            num0 += ws * pv.x;
            num1 += ws * pv.y;
        }
    } else {
        float gm = -INFINITY;
        for (int s = 0; s < splits; ++s) gm = fmaxf(gm, __ldg(&base[(size_t)s * (ATTN_HD + 2) + ATTN_HD]));
        float den = 0.f;
        for (int s = 0; s < splits; ++s) {
            const float* p = base + (size_t)s * (ATTN_HD + 2);
            float sm = __ldg(&p[ATTN_HD]);
            if (sm == -INFINITY) continue;
            float w = expf(sm - gm);
            den += w * __ldg(&p[ATTN_HD + 1]);
            float2 pv = __ldg(reinterpret_cast<const float2*>(p + d0));
            num0 += w * pv.x;
            num1 += w * pv.y;
        }
        inv_den = (den > 0.f) ? (1.f / den) : 0.f;
    }
    float o0 = bf16_round(num0 * inv_den);
    float o1 = bf16_round(num1 * inv_den);
    size_t row_head_base = ((size_t)m * nq + head) * ATTN_HD;
    if (gate) {
        uint32_t g2 = *reinterpret_cast<const uint32_t*>(gate + row_head_base + d0);
        uint16_t gl = (uint16_t)(g2 & 0xFFFFu);
        uint16_t gh = (uint16_t)(g2 >> 16);
        o0 = o0 * sigmoidf_(ld_bf16(&gl));
        o1 = o1 * sigmoidf_(ld_bf16(&gh));
    }
    __nv_bfloat16 b0 = __float2bfloat16(o0);
    __nv_bfloat16 b1 = __float2bfloat16(o1);
    uint16_t u0, u1;
    memcpy(&u0, &b0, sizeof(u0));
    memcpy(&u1, &b1, sizeof(u1));
    *reinterpret_cast<uint32_t*>(out + row_head_base + d0) = (uint32_t)u0 | ((uint32_t)u1 << 16);
    NVK_PDL_EPILOG();
}

extern "C" int nv_kernels_q4exp_attn_combine(
    void* stream, const float* partial, const uint16_t* gate, int m, int nq, int hd, int splits,
    uint16_t* out
) {
    if (m <= 0) return 0;
    if (hd != ATTN_HD) return -2;
    if (m > NVK_MAX_GRID_Y) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid(nq, m);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(ATTN_HD / 2), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_attn_combine_f2_kernel, partial, gate, nq, splits, out);
    } else {
        q4exp_attn_combine_f2_kernel<<<grid, ATTN_HD / 2, 0, s>>>(partial, gate, nq, splits, out);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_store_rows_at(
    void* stream, const uint16_t* src, uint16_t* dst, const int32_t* pos, int t, int row_elems
) {
    if (t <= 0) return 0;
    if (row_elems % 8 != 0) return -2;
    dim3 grid((row_elems / 8 + 255) / 256, t);
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_store_rows_at_kernel, src, dst, pos, t, row_elems);
    } else {
        q4exp_store_rows_at_kernel<<<grid, 256, 0, s>>>(src, dst, pos, t, row_elems);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_decode_meta(
    void* stream, const int32_t* pos, int t, int ratio, int32_t* positions, int32_t* nb_per_row
) {
    if (t <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((t + 127) / 128));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(128), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_decode_meta_kernel, pos, t, ratio, positions, nb_per_row);
    } else {
        q4exp_decode_meta_kernel<<<grid, 128, 0, s>>>(pos, t, ratio, positions, nb_per_row);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_history_push(
    void* stream, uint32_t* history, const uint32_t* tokens, int ctx, int t
) {
    if (ctx <= 0 || ctx > 8 || t <= 0) return -2;
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3(1), dim3(32), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_history_push_kernel, history, tokens, ctx, t);
    } else {
        q4exp_history_push_kernel<<<1, 32, 0, s>>>(history, tokens, ctx, t);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_qsa_compress_devpos(
    void* stream, const uint16_t* raw_k, int raw_cap, const int32_t* pos, int t, const float* k_norm_w,
    const float* cos_t, const float* sin_t, uint16_t* out, int ratio, int hd, int rot, float eps
) {
    if (t <= 0) return 0;
    if (raw_cap <= 0 || hd % 32 != 0 || hd > 1024 || rot % 2 != 0 || rot > hd) return -2;
    int max_blocks = t / ratio + 1;
    size_t smem = (size_t)hd * sizeof(float);
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)max_blocks), dim3((unsigned)hd), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_qsa_compress_devpos_kernel,
            raw_k, raw_cap, pos, t, k_norm_w, cos_t, sin_t, out, ratio, hd, rot, eps);
    } else {
        q4exp_qsa_compress_devpos_kernel<<<max_blocks, hd, smem, s>>>(
            raw_k, raw_cap, pos, t, k_norm_w, cos_t, sin_t, out, ratio, hd, rot, eps);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_embed_hc(
    void* stream, const uint16_t* embed, const uint32_t* tokens, int t, int h, int nb, uint16_t* out
) {
    if (t <= 0) return 0;
    if (h % 8 != 0) return -2;
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)t), dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_embed_hc_kernel, embed, tokens, h, nb, out);
    } else {
        q4exp_embed_hc_kernel<<<t, 256, 0, s>>>(embed, tokens, h, nb, out);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_attn_prep(
    void* stream, const uint16_t* q_raw, const uint16_t* k_raw, const uint16_t* v_raw,
    const uint16_t* idx_raw, const float* q_norm_w, const float* k_norm_w, const float* iq_norm_w,
    const float* cos_t, const float* sin_t, const int32_t* pos_dev, int t, int nq, int nkv, int hd,
    int inh, int ih, int rot, float eps, uint16_t* q_out, uint16_t* gate_out, uint16_t* k_cache,
    uint16_t* v_cache, uint16_t* raw_idx_cache, int raw_idx_cap, uint16_t* iq_out
) {
    if (t <= 0) return 0;
    if (raw_idx_cap <= 0 || hd % 32 != 0 || hd > 1024 || ih > hd || ih % 32 != 0 || rot % 2 != 0 || rot > ih) return -2;
    if (t > NVK_MAX_GRID_Y) return NVK_ERR_GRID_AXIS;
    dim3 grid(nq + nkv + inh + 1, t);
    size_t smem = (size_t)hd * sizeof(float);
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3((unsigned)hd), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_attn_prep_kernel,
            q_raw, k_raw, v_raw, idx_raw, q_norm_w, k_norm_w, iq_norm_w, cos_t, sin_t, pos_dev, nq, nkv, hd,
            inh, ih, rot, eps, q_out, gate_out, k_cache, v_cache, raw_idx_cache, raw_idx_cap, iq_out);
    } else {
        q4exp_attn_prep_kernel<<<grid, hd, smem, s>>>(
            q_raw, k_raw, v_raw, idx_raw, q_norm_w, k_norm_w, iq_norm_w, cos_t, sin_t, pos_dev, nq, nkv, hd,
            inh, ih, rot, eps, q_out, gate_out, k_cache, v_cache, raw_idx_cache, raw_idx_cap, iq_out);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_ple_gate(
    void* stream, const uint16_t* key, const uint16_t* value, const uint16_t* hc, const float* norm_key_w,
    const float* norm_query_w, const float* norm_conv_w, int t, int h, int nb, float eps,
    uint16_t* gv_out, float* gvn_out
) {
    if (t <= 0) return 0;
    dim3 grid(nb, t);
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_ple_gate_kernel,
            key, value, hc, norm_key_w, norm_query_w, norm_conv_w, h, nb, eps, gv_out, gvn_out);
    } else {
        q4exp_ple_gate_kernel<<<grid, 256, 0, s>>>(
            key, value, hc, norm_key_w, norm_query_w, norm_conv_w, h, nb, eps, gv_out, gvn_out);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_ple_conv(
    void* stream, const float* gvn, const uint16_t* gv, float* window, const float* conv_w, int t,
    int hh, int k, int dil, int win, int update_window, uint16_t* out
) {
    if (t <= 0) return 0;
    if (win + t > 64 || (k - 1) * dil != win) return -2;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((hh + 255) / 256));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_ple_conv_kernel,
            gvn, gv, window, conv_w, t, hh, k, dil, win, update_window, out);
    } else {
        q4exp_ple_conv_kernel<<<grid, 256, 0, s>>>(
            gvn, gv, window, conv_w, t, hh, k, dil, win, update_window, out);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_ple_window_advance(
    void* stream, const float* gvn, float* window, int t, int hh, int win
) {
    if (t <= 0) return 0;
    if (win + t > 64) return -2;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((hh + 255) / 256));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_ple_window_advance_kernel, gvn, window, t, hh, win);
    } else {
        q4exp_ple_window_advance_kernel<<<grid, 256, 0, s>>>(gvn, window, t, hh, win);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_rmsnorm_rows(
    void* stream, const uint16_t* x, const float* w, uint16_t* out, int rows, int n, float eps
) {
    if (rows <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)rows), dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_rmsnorm_rows_kernel, x, w, out, n, eps);
    } else {
        q4exp_rmsnorm_rows_kernel<<<rows, 256, 0, s>>>(x, w, out, n, eps);
    }
    return (int)cudaGetLastError();
}

namespace {

constexpr int kQ4ExpFp8Warp = 32;
constexpr int kQ4ExpFp8Rows = 8;
constexpr int kQ4ExpFp8Block = kQ4ExpFp8Warp * kQ4ExpFp8Rows;
constexpr int kQ4ExpCombineNormChunksPerThread = 8;
constexpr int kQ4ExpTailMaxSlots = 16;

__device__ __forceinline__ void q4exp_deq_e4m3x16(const int4& raw, float* wf) {
    const __nv_fp8x2_storage_t* p2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p2[t], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
        wf[2 * t] = f.x;
        wf[2 * t + 1] = f.y;
    }
}

__device__ __forceinline__ void q4exp_deq_e4m3x4(uint32_t raw, float* wf) {
    const __nv_fp8x2_storage_t* p2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
    for (int t = 0; t < 2; ++t) {
        __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p2[t], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
        wf[2 * t] = f.x;
        wf[2 * t + 1] = f.y;
    }
}

__device__ __forceinline__ void q4exp_deq_e4m3x8(uint2 raw, float* out) {
    const __nv_fp8x2_storage_t* p2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p2[t], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
        out[2 * t] = f.x;
        out[2 * t + 1] = f.y;
    }
}

template <int TM>
__device__ __forceinline__ void q4exp_stage_rows_f32(const __nv_bfloat16* __restrict__ x, float2* xshf, int K) {
    const int rowpad = (K >> 4) * 9;
    const int kp = K >> 1;
    for (int idx = threadIdx.x; idx < TM * kp; idx += kQ4ExpFp8Block) {
        int j = idx / kp;
        int p = idx - j * kp;
        int k0 = 2 * p;
        xshf[j * rowpad + (p >> 3) * 9 + (p & 7)] = make_float2(
            __bfloat162float(x[(size_t)j * K + k0]),
            __bfloat162float(x[(size_t)j * K + k0 + 1]));
    }
}

template <int TM>
__global__ void q4exp_hc_down_fp8_kernel(
    const uint16_t* __restrict__ xn, const uint8_t* __restrict__ w_down, const float* __restrict__ rs_down,
    const uint16_t* __restrict__ w_inject, float* __restrict__ lora_out, float* __restrict__ inject_out,
    int hh, int nb, int rank
) {
    __shared__ float red[TM][8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float acc[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) acc[m] = 0.f;
    if (row < rank) {
        const int4* wr = reinterpret_cast<const int4*>(w_down + (size_t)row * hh);
        int Kv = hh >> 4;
        int4 wcur = (tid < Kv) ? __ldcs(&wr[tid]) : make_int4(0, 0, 0, 0);
        NVK_PDL_PROLOG();
        for (int v = tid; v < Kv; v += 256) {
            int vn = v + 256;
            int4 wnxt = (vn < Kv) ? __ldcs(&wr[vn]) : make_int4(0, 0, 0, 0);
            float wf[16];
            q4exp_deq_e4m3x16(wcur, wf);
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                const uint4* xr = reinterpret_cast<const uint4*>(xn + (size_t)m * hh + (size_t)v * 16);
                float xf0[8];
                float xf1[8];
                unpack8_bf16(xr[0], xf0);
                unpack8_bf16(xr[1], xf1);
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[m] += wf[j] * xf0[j] + wf[8 + j] * xf1[j];
            }
            wcur = wnxt;
        }
#pragma unroll
        for (int m = 0; m < TM; ++m) {
            float v = warp_sum(acc[m]);
            if (lane == 0) red[m][warp] = v;
        }
        __syncthreads();
        if (tid < TM) {
            float tot = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[tid][w];
            float v = bf16_round(tot * __ldg(&rs_down[row])) / (float)nb;
            lora_out[(size_t)tid * rank + row] = bf16_round(v * sigmoidf_(v));
        }
    } else {
        const uint4* wr = reinterpret_cast<const uint4*>(w_inject + (size_t)(row - rank) * hh);
        int Kv = hh >> 3;
        uint4 wcur = (tid < Kv) ? __ldcs(&wr[tid]) : make_uint4(0u, 0u, 0u, 0u);
        NVK_PDL_PROLOG();
        for (int v = tid; v < Kv; v += 256) {
            int vn = v + 256;
            uint4 wnxt = (vn < Kv) ? __ldcs(&wr[vn]) : make_uint4(0u, 0u, 0u, 0u);
            float wf[8];
            unpack8_bf16(wcur, wf);
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                uint4 xv = *reinterpret_cast<const uint4*>(xn + (size_t)m * hh + (size_t)v * 8);
                float xf[8];
                unpack8_bf16(xv, xf);
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[m] += wf[j] * xf[j];
            }
            wcur = wnxt;
        }
#pragma unroll
        for (int m = 0; m < TM; ++m) {
            float v = warp_sum(acc[m]);
            if (lane == 0) red[m][warp] = v;
        }
        __syncthreads();
        if (tid < TM) {
            float tot = 0.f;
#pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[tid][w];
            float v = bf16_round(tot) / (float)nb;
            inject_out[(size_t)tid * nb + (row - rank)] = 2.f * sigmoidf_(v);
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void q4exp_hc_up_mix_fp8_kernel(
    const float* __restrict__ lora, const uint8_t* __restrict__ w_up, const float* __restrict__ rs_up,
    const uint16_t* __restrict__ xn, uint16_t* __restrict__ mixed, int h, int nb, int rank
) {
    extern __shared__ float ls[];
    int tid = threadIdx.x;
    int hw = tid >> 4;
    int sub = tid & 15;
    int j = blockIdx.x * 8 + hw;
    bool active = j < h;
    unsigned hmask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
    const uint8_t* w0 = w_up + (size_t)(active ? j : 0) * rank;
    int4 p0a = (active && rank == 320) ? __ldcs(reinterpret_cast<const int4*>(w0 + sub * 16)) : make_int4(0, 0, 0, 0);
    uint32_t p0b = (active && rank == 320) ? __ldcs(reinterpret_cast<const uint32_t*>(w0 + 256 + sub * 4)) : 0u;
    NVK_PDL_PROLOG();
    for (int i = tid; i < TM * rank; i += 128) ls[i] = lora[i];
    __syncthreads();
    if (!active) {
        NVK_PDL_EPILOG();
        return;
    }
    int hh = nb * h;
    float out[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) out[m] = 0.f;
    for (int b = 0; b < nb; ++b) {
        float acc[TM];
#pragma unroll
        for (int m = 0; m < TM; ++m) acc[m] = 0.f;
        const uint8_t* wr = w_up + (size_t)(b * h + j) * rank;
        if (rank == 320) {
            int4 wa = (b == 0) ? p0a : __ldcs(reinterpret_cast<const int4*>(wr + sub * 16));
            uint32_t wb = (b == 0) ? p0b : __ldcs(reinterpret_cast<const uint32_t*>(wr + 256 + sub * 4));
            float wfa[16];
            float wfb[4];
            q4exp_deq_e4m3x16(wa, wfa);
            q4exp_deq_e4m3x4(wb, wfb);
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                const float4* la4 = reinterpret_cast<const float4*>(ls + m * 320 + sub * 16);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    float4 lv = la4[q];
                    acc[m] += wfa[4 * q] * lv.x + wfa[4 * q + 1] * lv.y + wfa[4 * q + 2] * lv.z + wfa[4 * q + 3] * lv.w;
                }
                float4 lb = *reinterpret_cast<const float4*>(ls + m * 320 + 256 + sub * 4);
                acc[m] += wfb[0] * lb.x + wfb[1] * lb.y + wfb[2] * lb.z + wfb[3] * lb.w;
            }
        } else {
            for (int i = sub * 16; i < rank; i += 256) {
                int4 wa = __ldcs(reinterpret_cast<const int4*>(wr + i));
                float wfa[16];
                q4exp_deq_e4m3x16(wa, wfa);
#pragma unroll
                for (int m = 0; m < TM; ++m) {
                    const float4* la4 = reinterpret_cast<const float4*>(ls + m * rank + i);
#pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        float4 lv = la4[q];
                        acc[m] += wfa[4 * q] * lv.x + wfa[4 * q + 1] * lv.y + wfa[4 * q + 2] * lv.z + wfa[4 * q + 3] * lv.w;
                    }
                }
            }
        }
        float rs = __ldg(&rs_up[b * h + j]);
#pragma unroll
        for (int m = 0; m < TM; ++m) {
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(hmask, acc[m], o, 16);
            if (sub == 0) {
                float g = sigmoidf_(bf16_round(acc[m] * rs));
                out[m] += g * ld_bf16(xn + (size_t)m * hh + b * h + j);
            }
        }
    }
    if (sub == 0) {
#pragma unroll
        for (int m = 0; m < TM; ++m) st_bf16(mixed + (size_t)m * h + j, out[m] / (float)nb);
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(256) q4exp_hc_up_mix_fp8_r320_nb4_kernel(
    const float* __restrict__ lora, const uint8_t* __restrict__ w_up, const float* __restrict__ rs_up,
    const uint16_t* __restrict__ xn, uint16_t* __restrict__ mixed, int h
) {
    extern __shared__ float ls[];
    __shared__ float s_part[TM][4][4];
    int tid = threadIdx.x;
    int sub = tid & 15;
    int hw = tid >> 4;
    int j_loc = hw & 3;
    int b = hw >> 2;
    int j = blockIdx.x * 4 + j_loc;
    bool active = j < h;
    unsigned hmask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
    int row_idx = b * h + j;
    const uint8_t* wr = w_up + (size_t)(active ? row_idx : 0) * 320;
    int4 wa = active ? __ldcs(reinterpret_cast<const int4*>(wr + sub * 16)) : make_int4(0, 0, 0, 0);
    uint32_t wb = active ? __ldcs(reinterpret_cast<const uint32_t*>(wr + 256 + sub * 4)) : 0u;
    float rs = active ? __ldg(&rs_up[row_idx]) : 0.f;
    NVK_PDL_PROLOG();
    for (int i = tid; i < TM * 320; i += 256) ls[i] = lora[i];
    __syncthreads();
    if (active) {
        float wfa[16], wfb[4];
        q4exp_deq_e4m3x16(wa, wfa);
        q4exp_deq_e4m3x4(wb, wfb);
        float acc[TM];
#pragma unroll
        for (int m = 0; m < TM; ++m) {
            float a = 0.f;
            const float4* la = reinterpret_cast<const float4*>(ls + m * 320 + sub * 16);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float4 v0 = la[q];
                a += wfa[4 * q] * v0.x + wfa[4 * q + 1] * v0.y + wfa[4 * q + 2] * v0.z + wfa[4 * q + 3] * v0.w;
            }
            float4 lb = *reinterpret_cast<const float4*>(ls + m * 320 + 256 + sub * 4);
            a += wfb[0] * lb.x + wfb[1] * lb.y + wfb[2] * lb.z + wfb[3] * lb.w;
            acc[m] = a;
        }
        int hh = 4 * h;
#pragma unroll
        for (int m = 0; m < TM; ++m) {
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(hmask, acc[m], o, 16);
            if (sub == 0) {
                float g = sigmoidf_(bf16_round(acc[m] * rs));
                s_part[m][j_loc][b] = g * ld_bf16(xn + (size_t)m * hh + row_idx);
            }
        }
    }
    __syncthreads();
    if (tid < TM * 4) {
        int m = tid >> 2;
        int jl = tid & 3;
        int jj = blockIdx.x * 4 + jl;
        if (jj < h) {
            float sum = s_part[m][jl][0] + s_part[m][jl][1] + s_part[m][jl][2] + s_part[m][jl][3];
            st_bf16(mixed + (size_t)m * h + jj, sum * 0.25f);
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void q4exp_hc_up_mix_fp8_r320_nb8_kernel(
    const float* __restrict__ lora, const uint8_t* __restrict__ w_up, const float* __restrict__ rs_up,
    const uint16_t* __restrict__ xn, uint16_t* __restrict__ mixed, int h
) {
    extern __shared__ float ls[];
    __shared__ float s_part[TM][4][8];
    int tid = threadIdx.x;
    int sub = tid & 7;
    int row_in_cta = tid >> 3;
    int j_loc = row_in_cta & 3;
    int b = row_in_cta >> 2;
    int j = blockIdx.x * 4 + j_loc;
    bool active = j < h;
    int row_idx = b * h + j;
    const uint8_t* wr = w_up + (size_t)(active ? row_idx : 0) * 320;
    int4 w0 = active ? __ldcs(reinterpret_cast<const int4*>(wr + sub * 16)) : make_int4(0, 0, 0, 0);
    int4 w1 = active ? __ldcs(reinterpret_cast<const int4*>(wr + 128 + sub * 16)) : make_int4(0, 0, 0, 0);
    uint2 w2 = active ? __ldcs(reinterpret_cast<const uint2*>(wr + 256 + sub * 8)) : make_uint2(0u, 0u);
    float rs = active ? __ldg(&rs_up[row_idx]) : 0.f;
    NVK_PDL_PROLOG();
    for (int i = tid; i < TM * 320; i += 256) ls[i] = lora[i];
    __syncthreads();
    if (active) {
        float wf0[16], wf1[16], wf2[8];
        q4exp_deq_e4m3x16(w0, wf0);
        q4exp_deq_e4m3x16(w1, wf1);
        q4exp_deq_e4m3x8(w2, wf2);
        float acc[TM];
#pragma unroll
        for (int m = 0; m < TM; ++m) {
            float a = 0.f;
            const float4* l0 = reinterpret_cast<const float4*>(ls + m * 320 + sub * 16);
            const float4* l1 = reinterpret_cast<const float4*>(ls + m * 320 + 128 + sub * 16);
            const float4* l2 = reinterpret_cast<const float4*>(ls + m * 320 + 256 + sub * 8);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                float4 v0 = l0[q];
                a += wf0[4 * q] * v0.x + wf0[4 * q + 1] * v0.y + wf0[4 * q + 2] * v0.z + wf0[4 * q + 3] * v0.w;
                float4 v1 = l1[q];
                a += wf1[4 * q] * v1.x + wf1[4 * q + 1] * v1.y + wf1[4 * q + 2] * v1.z + wf1[4 * q + 3] * v1.w;
            }
            float4 v2a = l2[0];
            float4 v2b = l2[1];
            a += wf2[0] * v2a.x + wf2[1] * v2a.y + wf2[2] * v2a.z + wf2[3] * v2a.w;
            a += wf2[4] * v2b.x + wf2[5] * v2b.y + wf2[6] * v2b.z + wf2[7] * v2b.w;
            acc[m] = a;
        }
        int hh = 8 * h;
#pragma unroll
        for (int m = 0; m < TM; ++m) {
#pragma unroll
            for (int o = 4; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o, 8);
            if (sub == 0) {
                float g = sigmoidf_(bf16_round(acc[m] * rs));
                s_part[m][j_loc][b] = g * ld_bf16(xn + (size_t)m * hh + row_idx);
            }
        }
    }
    __syncthreads();
    if (tid < TM * 4) {
        int m = tid >> 2;
        int jl = tid & 3;
        int jj = blockIdx.x * 4 + jl;
        if (jj < h) {
            float sum = 0.f;
#pragma unroll
            for (int bb = 0; bb < 8; ++bb) sum += s_part[m][jl][bb];
            st_bf16(mixed + (size_t)m * h + jj, sum * 0.125f);
        }
    }
    NVK_PDL_EPILOG();
}

__device__ __forceinline__ void q4exp_resolve_qkv_row(
    int g, int nq, int nk, int nv, int ni,
    const uint8_t* wq, const float* rsq, __nv_bfloat16* yq,
    const uint8_t* wk, const float* rsk, __nv_bfloat16* yk,
    const uint8_t* wv, const float* rsv, __nv_bfloat16* yv,
    const uint8_t* wi, const float* rsi, __nv_bfloat16* yi,
    int K, const uint8_t*& w_row, float& rs_val, __nv_bfloat16*& y_base, int& y_col, int& y_stride
) {
    if (g < nq) {
        w_row = wq + (size_t)g * K;
        rs_val = __ldg(&rsq[g]);
        y_base = yq;
        y_col = g;
        y_stride = nq;
    } else if (g < nq + nk) {
        int r = g - nq;
        w_row = wk + (size_t)r * K;
        rs_val = __ldg(&rsk[r]);
        y_base = yk;
        y_col = r;
        y_stride = nk;
    } else if (g < nq + nk + nv) {
        int r = g - nq - nk;
        w_row = wv + (size_t)r * K;
        rs_val = __ldg(&rsv[r]);
        y_base = yv;
        y_col = r;
        y_stride = nv;
    } else {
        int r = g - nq - nk - nv;
        w_row = wi + (size_t)r * K;
        rs_val = __ldg(&rsi[r]);
        y_base = yi;
        y_col = r;
        y_stride = ni;
    }
}

template <int TM>
__global__ void q4exp_fp8_qkv_idx_gemv_kernel(
    const uint8_t* __restrict__ wq, const float* __restrict__ rsq,
    const uint8_t* __restrict__ wk, const float* __restrict__ rsk,
    const uint8_t* __restrict__ wv, const float* __restrict__ rsv,
    const uint8_t* __restrict__ wi, const float* __restrict__ rsi,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ yq, __nv_bfloat16* __restrict__ yk,
    __nv_bfloat16* __restrict__ yv, __nv_bfloat16* __restrict__ yi,
    int nq, int nk, int nv, int ni, int K
) {
    extern __shared__ float2 xshf[];
    int total = nq + nk + nv + ni;
    int lane = threadIdx.x & (kQ4ExpFp8Warp - 1);
    int warp = threadIdx.x / kQ4ExpFp8Warp;
    int n0 = blockIdx.x * (2 * kQ4ExpFp8Rows) + warp;
    int n1 = n0 + kQ4ExpFp8Rows;
    bool has0 = n0 < total;
    bool has1 = n1 < total;
    const uint8_t* ra_row = wq;
    const uint8_t* rb_row = wq;
    float rs0 = 0.f;
    float rs1 = 0.f;
    __nv_bfloat16* y0_base = yq;
    __nv_bfloat16* y1_base = yq;
    int y0_col = 0, y0_stride = nq;
    int y1_col = 0, y1_stride = nq;
    if (has0) {
        q4exp_resolve_qkv_row(n0, nq, nk, nv, ni, wq, rsq, yq, wk, rsk, yk, wv, rsv, yv, wi, rsi, yi, K,
                              ra_row, rs0, y0_base, y0_col, y0_stride);
    }
    if (has1) {
        q4exp_resolve_qkv_row(n1, nq, nk, nv, ni, wq, rsq, yq, wk, rsk, yk, wv, rsv, yv, wi, rsi, yi, K,
                              rb_row, rs1, y1_base, y1_col, y1_stride);
    }
    const int4* wa = reinterpret_cast<const int4*>(ra_row);
    const int4* wb = reinterpret_cast<const int4*>(rb_row);
    int Kv = K >> 4;
    int4 ra = (has0 && lane < Kv) ? __ldcs(&wa[lane]) : make_int4(0, 0, 0, 0);
    int4 rb = (has1 && lane < Kv) ? __ldcs(&wb[lane]) : make_int4(0, 0, 0, 0);
    NVK_PDL_PROLOG();
    q4exp_stage_rows_f32<TM>(x, xshf, K);
    __syncthreads();
    if (!has0) {
        NVK_PDL_EPILOG();
        return;
    }
    const int rowpad = Kv * 9;
    float acc0[TM];
    float acc1[TM];
#pragma unroll
    for (int j = 0; j < TM; ++j) {
        acc0[j] = 0.f;
        acc1[j] = 0.f;
    }
    int v = lane;
    while (v < Kv) {
        int vn = v + kQ4ExpFp8Warp;
        int4 na = (vn < Kv) ? __ldcs(&wa[vn]) : make_int4(0, 0, 0, 0);
        int4 nb = (has1 && vn < Kv) ? __ldcs(&wb[vn]) : make_int4(0, 0, 0, 0);
        float wfa[16];
        float wfb[16];
        q4exp_deq_e4m3x16(ra, wfa);
        q4exp_deq_e4m3x16(rb, wfb);
        const float2* base = xshf + v * 9;
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const float2* xp = base + j * rowpad;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 xv = xp[p];
                acc0[j] += wfa[2 * p] * xv.x + wfa[2 * p + 1] * xv.y;
                acc1[j] += wfb[2 * p] * xv.x + wfb[2 * p + 1] * xv.y;
            }
        }
        v = vn;
        ra = na;
        rb = nb;
    }
#pragma unroll
    for (int j = 0; j < TM; ++j) {
#pragma unroll
        for (int offset = kQ4ExpFp8Warp / 2; offset > 0; offset >>= 1) {
            acc0[j] += __shfl_xor_sync(0xffffffff, acc0[j], offset);
            acc1[j] += __shfl_xor_sync(0xffffffff, acc1[j], offset);
        }
        if (lane == 0) {
            y0_base[(size_t)j * y0_stride + y0_col] = __float2bfloat16(acc0[j] * rs0);
            if (has1) y1_base[(size_t)j * y1_stride + y1_col] = __float2bfloat16(acc1[j] * rs1);
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__device__ __forceinline__ void q4exp_fp8_rows2_dot(
    const uint8_t* __restrict__ ra_row, const uint8_t* __restrict__ rb_row, bool has_b,
    const float2* xshf, int K, float* acc0, float* acc1
) {
    const int rowpad = (K >> 4) * 9;
    const int lane = threadIdx.x & (kQ4ExpFp8Warp - 1);
    const int4* wa = reinterpret_cast<const int4*>(ra_row);
    const int4* wb = reinterpret_cast<const int4*>(rb_row);
    int Kv = K / 16;
#pragma unroll
    for (int j = 0; j < TM; ++j) {
        acc0[j] = 0.0f;
        acc1[j] = 0.0f;
    }
    int v = lane;
    int4 ra = (v < Kv) ? __ldcs(&wa[v]) : make_int4(0, 0, 0, 0);
    int4 rb = (has_b && v < Kv) ? __ldcs(&wb[v]) : make_int4(0, 0, 0, 0);
    while (v < Kv) {
        int vn = v + kQ4ExpFp8Warp;
        int4 na = (vn < Kv) ? __ldcs(&wa[vn]) : make_int4(0, 0, 0, 0);
        int4 nb = (has_b && vn < Kv) ? __ldcs(&wb[vn]) : make_int4(0, 0, 0, 0);
        float wfa[16];
        float wfb[16];
        q4exp_deq_e4m3x16(ra, wfa);
        q4exp_deq_e4m3x16(rb, wfb);
        const float2* base = xshf + v * 9;
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const float2* xp = base + j * rowpad;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 xv = xp[p];
                acc0[j] += wfa[2 * p] * xv.x + wfa[2 * p + 1] * xv.y;
                acc1[j] += wfb[2 * p] * xv.x + wfb[2 * p + 1] * xv.y;
            }
        }
        v = vn;
        ra = na;
        rb = nb;
    }
#pragma unroll
    for (int j = 0; j < TM; ++j) {
#pragma unroll
        for (int offset = kQ4ExpFp8Warp / 2; offset > 0; offset >>= 1) {
            acc0[j] += __shfl_xor_sync(0xffffffff, acc0[j], offset);
            acc1[j] += __shfl_xor_sync(0xffffffff, acc1[j], offset);
        }
    }
}

}

template <int MAXC>
__global__ void q4exp_hc_combine_norm_kernel(
    const uint16_t* __restrict__ hc_in, const uint16_t* __restrict__ y, const float* __restrict__ inject,
    const float* __restrict__ norm_w, uint16_t* __restrict__ hc_out, uint16_t* __restrict__ xn_out,
    int hh, int h, int nb, float eps
) {
    __shared__ float red[8][32];
    __shared__ float rstd[8];
    int m = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nwarps = blockDim.x >> 5;
    int group = hh / nb;
    int n8 = hh / 8;
    const uint16_t* xr = hc_in + (size_t)m * hh;
    float f[MAXC][8];
#pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        int c = tid + i * blockDim.x;
        if (c < n8) {
            uint4 v = *reinterpret_cast<const uint4*>(xr + (size_t)c * 8);
            unpack8_bf16(v, f[i]);
        }
    }
    if (y != nullptr) {
#pragma unroll
        for (int i = 0; i < MAXC; ++i) {
            int c = tid + i * blockDim.x;
            if (c < n8) {
                int b = (c * 8) / h;
                int j0 = c * 8 - b * h;
                uint4 yv = *reinterpret_cast<const uint4*>(y + (size_t)m * h + j0);
                float yf[8];
                unpack8_bf16(yv, yf);
                float inj = inject[(size_t)m * nb + b];
                __align__(16) __nv_bfloat16 o[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float v = f[i][j] + yf[j] * inj;
                    o[j] = __float2bfloat16(v);
                    f[i][j] = __bfloat162float(o[j]);
                }
                *reinterpret_cast<uint4*>(hc_out + (size_t)m * hh + (size_t)c * 8) = *reinterpret_cast<const uint4*>(o);
            }
        }
    }
    float ss[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) ss[g] = 0.f;
#pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        int c = tid + i * blockDim.x;
        if (c < n8) {
            float acc = 0.f;
#pragma unroll
            for (int j = 0; j < 8; ++j) acc += f[i][j] * f[i][j];
            int g = (c * 8) / group;
#pragma unroll
            for (int gg = 0; gg < 8; ++gg) if (gg == g) ss[gg] += acc;
        }
    }
#pragma unroll
    for (int g = 0; g < 8; ++g) {
        float v = warp_sum(ss[g]);
        if (lane == 0) red[g][warp] = v;
    }
    __syncthreads();
    if (tid < nb) {
        float tot = 0.f;
        for (int w = 0; w < nwarps; ++w) tot += red[tid][w];
        rstd[tid] = rsqrtf(tot / (float)group + eps);
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < MAXC; ++i) {
        int c = tid + i * blockDim.x;
        if (c < n8) {
            int g = (c * 8) / group;
            float r = rstd[g];
            __align__(16) __nv_bfloat16 o[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16(f[i][j] * r * norm_w[c * 8 + j]);
            *reinterpret_cast<uint4*>(xn_out + (size_t)m * hh + (size_t)c * 8) = *reinterpret_cast<const uint4*>(o);
        }
    }
}

__global__ void Q4EXP_CLUSTER_DIMS_8 q4exp_hc_combine_norm_cluster_kernel(
    const uint16_t* __restrict__ hc_in, const uint16_t* __restrict__ y, const float* __restrict__ inject,
    const float* __restrict__ norm_w, uint16_t* __restrict__ hc_out, uint16_t* __restrict__ xn_out,
    int hh, int h, int nb, float eps
) {
#if Q4EXP_HAS_CLUSTERS
    __shared__ float accs[kQ4ExpCombineNormChunksPerThread][32];
    __shared__ int grp[kQ4ExpCombineNormChunksPerThread][32];
    __shared__ float red_all[8][8];
    __shared__ float rstd[8];
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    unsigned r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
    const int m = blockIdx.y;
    const int lane = threadIdx.x & 31;
    const int it = threadIdx.x >> 5;
    const int group = hh / nb;
    const int n8 = hh / 8;
    const int c = (int)r * 32 + lane + 256 * it;
    const bool valid = c < n8;
    float f[8];
    float wv[8];
    if (valid) {
        float4 w0 = __ldg(reinterpret_cast<const float4*>(norm_w + c * 8));
        float4 w1 = __ldg(reinterpret_cast<const float4*>(norm_w + c * 8 + 4));
        wv[0] = w0.x; wv[1] = w0.y; wv[2] = w0.z; wv[3] = w0.w;
        wv[4] = w1.x; wv[5] = w1.y; wv[6] = w1.z; wv[7] = w1.w;
    }
    NVK_PDL_PROLOG();
    if (valid) {
        uint4 v = *reinterpret_cast<const uint4*>(hc_in + (size_t)m * hh + (size_t)c * 8);
        unpack8_bf16(v, f);
        if (y != nullptr) {
            int b = (c * 8) / h;
            int j0 = c * 8 - b * h;
            uint4 yv = *reinterpret_cast<const uint4*>(y + (size_t)m * h + j0);
            float yf[8];
            unpack8_bf16(yv, yf);
            float inj = inject[(size_t)m * nb + b];
            __align__(16) __nv_bfloat16 o[8];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float vv = f[j] + yf[j] * inj;
                o[j] = __float2bfloat16(vv);
                f[j] = __bfloat162float(o[j]);
            }
            *reinterpret_cast<uint4*>(hc_out + (size_t)m * hh + (size_t)c * 8) = *reinterpret_cast<const uint4*>(o);
        }
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += f[j] * f[j];
        accs[it][lane] = acc;
        grp[it][lane] = (c * 8) / group;
    } else {
        grp[it][lane] = -1;
    }
    __syncthreads();
    asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    if (it == 0) {
        float ss[8];
#pragma unroll
        for (int g = 0; g < 8; ++g) ss[g] = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); ++i) {
            int g = grp[i][lane];
            if (g >= 0) {
                float a = accs[i][lane];
#pragma unroll
                for (int gg = 0; gg < 8; ++gg) if (gg == g) ss[gg] += a;
            }
        }
#pragma unroll
        for (int g = 0; g < 8; ++g) {
            float v = warp_sum(ss[g]);
            if (lane < 8 && g < nb) {
                unsigned local = (unsigned)__cvta_generic_to_shared(&red_all[g][r]);
                unsigned remote;
                asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(remote) : "r"(local), "r"(lane));
                asm volatile("st.shared::cluster.f32 [%0], %1;\n" :: "r"(remote), "f"(v) : "memory");
            }
        }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
    if (threadIdx.x < nb) {
        float tot = 0.f;
        for (int w = 0; w < 8; ++w) tot += red_all[threadIdx.x][w];
        rstd[threadIdx.x] = rsqrtf(tot / (float)group + eps);
    }
    __syncthreads();
    if (valid) {
        float rs = rstd[(c * 8) / group];
        __align__(16) __nv_bfloat16 o[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16(f[j] * rs * wv[j]);
        *reinterpret_cast<uint4*>(xn_out + (size_t)m * hh + (size_t)c * 8) = *reinterpret_cast<const uint4*>(o);
    }
    NVK_PDL_EPILOG();
#else
    __trap();
#endif
}

template <int TM>
__global__ void q4exp_fp8_gateup_silu_kernel(
    const uint8_t* __restrict__ wg, const float* __restrict__ rsg,
    const uint8_t* __restrict__ wu, const float* __restrict__ rsu,
    const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ act, int N, int K
) {
    extern __shared__ float2 xshf[];
    NVK_PDL_PROLOG();
    q4exp_stage_rows_f32<TM>(x, xshf, K);
    __syncthreads();
    int lane = threadIdx.x & (kQ4ExpFp8Warp - 1);
    int warp = threadIdx.x / kQ4ExpFp8Warp;
    int n = blockIdx.x * kQ4ExpFp8Rows + warp;
    if (n >= N) {
        NVK_PDL_EPILOG();
        return;
    }
    float acc0[TM];
    float acc1[TM];
    q4exp_fp8_rows2_dot<TM>(wg + (size_t)n * K, wu + (size_t)n * K, true, xshf, K, acc0, acc1);
    if (lane == 0) {
        float sg_ = __ldg(&rsg[n]);
        float su = __ldg(&rsu[n]);
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            float g = __bfloat162float(__float2bfloat16(acc0[j] * sg_));
            float u = __bfloat162float(__float2bfloat16(acc1[j] * su));
            float s = g / (1.0f + expf(-g));
            act[(size_t)j * N + n] = __float2bfloat16(__fmul_rn(s, u));
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void q4exp_fp8_down_tail_sub8_kernel(
    const uint8_t* __restrict__ wq, const float* __restrict__ rs, const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ y_down, long long slot_stride, const float* __restrict__ topk_w, int k,
    const __nv_bfloat16* __restrict__ sg, __nv_bfloat16* __restrict__ out, int N, int K
) {
    extern __shared__ float2 xshf[];
    const int tid = threadIdx.x;
    const int sub = tid & 7;
    const int sgrp = tid >> 3;
    const int n = blockIdx.x * 16 + sgrp;
    const bool active = n < N;
    const int Kv = K >> 4;
    const int rowpad = Kv * 9;
    const int4* wa = active ? reinterpret_cast<const int4*>(wq + (size_t)n * K) : nullptr;
    int4 ra = (active && sub < Kv) ? __ldcs(&wa[sub]) : make_int4(0, 0, 0, 0);
    const float rsn = active ? __ldg(&rs[n]) : 0.f;
    NVK_PDL_PROLOG();
    const int kp = K >> 1;
    for (int idx = tid; idx < TM * kp; idx += 128) {
        int j = idx / kp;
        int p = idx - j * kp;
        int k0 = 2 * p;
        xshf[j * rowpad + (p >> 3) * 9 + (p & 7)] = make_float2(
            __bfloat162float(act[(size_t)j * K + k0]),
            __bfloat162float(act[(size_t)j * K + k0 + 1]));
    }
    __syncthreads();
    if (!active) {
        NVK_PDL_EPILOG();
        return;
    }
    float acc[TM];
#pragma unroll
    for (int j = 0; j < TM; ++j) acc[j] = 0.f;
    int v = sub;
    while (v < Kv) {
        int vn = v + 8;
        int4 na = (vn < Kv) ? __ldcs(&wa[vn]) : make_int4(0, 0, 0, 0);
        float wfa[16];
        q4exp_deq_e4m3x16(ra, wfa);
        const float2* base = xshf + v * 9;
#pragma unroll
        for (int j = 0; j < TM; ++j) {
            const float2* xp = base + j * rowpad;
#pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 xv = xp[p];
                acc[j] += wfa[2 * p] * xv.x + wfa[2 * p + 1] * xv.y;
            }
        }
        v = vn;
        ra = na;
    }
#pragma unroll
    for (int j = 0; j < TM; ++j) {
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) acc[j] += __shfl_xor_sync(0xffffffffu, acc[j], o, 8);
    }
    if (sub < TM) {
        int j = sub;
        float s = __bfloat162float(__float2bfloat16(acc[j] * rsn));
        float gl = __bfloat162float(__ldg(&sg[j]));
        float gate = 1.f / (1.f + expf(-gl));
        float routed = 0.f;
#pragma unroll
        for (int q = 0; q < kQ4ExpTailMaxSlots; ++q) {
            if (q < k) {
                int slot = j * k + q;
                float wt = __ldg(&topk_w[slot]);
                float val = __bfloat162float(__ldg(&y_down[(size_t)slot * slot_stride + n]));
                routed += wt * val;
            }
        }
        out[(size_t)j * N + n] = __float2bfloat16(__fadd_rn(routed, __fmul_rn(s, gate)));
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void q4exp_fp8_down_tail_kernel(
    const uint8_t* __restrict__ wq, const float* __restrict__ rs, const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ y_down, long long slot_stride, const float* __restrict__ topk_w, int k,
    const __nv_bfloat16* __restrict__ sg, __nv_bfloat16* __restrict__ out, int N, int K
) {
    extern __shared__ float2 xshf[];
    NVK_PDL_PROLOG();
    q4exp_stage_rows_f32<TM>(act, xshf, K);
    __syncthreads();
    int lane = threadIdx.x & (kQ4ExpFp8Warp - 1);
    int warp = threadIdx.x / kQ4ExpFp8Warp;
    int n0 = blockIdx.x * (2 * kQ4ExpFp8Rows) + warp;
    int n1 = n0 + kQ4ExpFp8Rows;
    if (n0 >= N) {
        NVK_PDL_EPILOG();
        return;
    }
    bool has1 = n1 < N;
    float acc0[TM];
    float acc1[TM];
    q4exp_fp8_rows2_dot<TM>(wq + (size_t)n0 * K, wq + (size_t)n1 * K, has1, xshf, K, acc0, acc1);
    float rs0 = __ldg(&rs[n0]);
    float rs1 = has1 ? __ldg(&rs[n1]) : 0.0f;
#pragma unroll
    for (int j = 0; j < TM; ++j) {
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            if (lane != 2 * j + r) continue;
            if (r == 1 && !has1) continue;
            int n = r == 0 ? n0 : n1;
            float s = __bfloat162float(__float2bfloat16((r == 0 ? acc0[j] : acc1[j]) * (r == 0 ? rs0 : rs1)));
            float gl = __bfloat162float(sg[j]);
            float gate = 1.f / (1.f + expf(-gl));
            float wts[kQ4ExpTailMaxSlots];
            float vals[kQ4ExpTailMaxSlots];
#pragma unroll
            for (int q = 0; q < kQ4ExpTailMaxSlots; ++q) {
                if (q < k) {
                    int slot = j * k + q;
                    wts[q] = topk_w[slot];
                    vals[q] = __bfloat162float(y_down[(size_t)slot * slot_stride + n]);
                }
            }
            float routed = 0.f;
#pragma unroll
            for (int q = 0; q < kQ4ExpTailMaxSlots; ++q) {
                if (q < k) routed += wts[q] * vals[q];
            }
            out[(size_t)j * N + n] = __float2bfloat16(__fadd_rn(routed, __fmul_rn(s, gate)));
        }
    }
    NVK_PDL_EPILOG();
}

static int* g_q4exp_hc_xn_q8 = nullptr;
static float* g_q4exp_hc_xn_gsc = nullptr;

__global__ void __launch_bounds__(320) q4exp_hc_combine_norm_group_h2560_kernel(
    const uint16_t* __restrict__ hc_in, const uint16_t* __restrict__ y, const float* __restrict__ inject,
    const float* __restrict__ norm_w, uint16_t* __restrict__ hc_out, uint16_t* __restrict__ xn_out,
    int* __restrict__ xn_q8_out, float* __restrict__ xn_gsc_out,
    int hh, int nb, float eps
) {
    __shared__ float red[10];
    __shared__ float s_rstd;
    const int g = blockIdx.x;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int c = g * 320 + tid;
    float4 w0 = __ldg(reinterpret_cast<const float4*>(norm_w + (size_t)c * 8));
    float4 w1 = __ldg(reinterpret_cast<const float4*>(norm_w + (size_t)c * 8 + 4));
    NVK_PDL_PROLOG();
    float f[8];
    uint4 v = *reinterpret_cast<const uint4*>(hc_in + (size_t)m * hh + (size_t)c * 8);
    unpack8_bf16(v, f);
    if (y != nullptr) {
        float inj = __ldg(&inject[(size_t)m * nb + g]);
        uint4 yv = *reinterpret_cast<const uint4*>(y + (size_t)m * 2560 + (size_t)tid * 8);
        float yf[8];
        unpack8_bf16(yv, yf);
        __align__(16) __nv_bfloat16 o[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float vv = f[j] + yf[j] * inj;
            o[j] = __float2bfloat16(vv);
            f[j] = __bfloat162float(o[j]);
        }
        *reinterpret_cast<uint4*>(hc_out + (size_t)m * hh + (size_t)c * 8) = *reinterpret_cast<const uint4*>(o);
    }
    float ss = f[0]*f[0] + f[1]*f[1] + f[2]*f[2] + f[3]*f[3]
             + f[4]*f[4] + f[5]*f[5] + f[6]*f[6] + f[7]*f[7];
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = red[0] + red[1] + red[2] + red[3] + red[4]
                  + red[5] + red[6] + red[7] + red[8] + red[9];
        s_rstd = rsqrtf(tot * (1.0f / 2560.0f) + eps);
    }
    __syncthreads();
    float r = s_rstd;
    float fn[8];
    fn[0] = f[0] * r * w0.x;
    fn[1] = f[1] * r * w0.y;
    fn[2] = f[2] * r * w0.z;
    fn[3] = f[3] * r * w0.w;
    fn[4] = f[4] * r * w1.x;
    fn[5] = f[5] * r * w1.y;
    fn[6] = f[6] * r * w1.z;
    fn[7] = f[7] * r * w1.w;
    __align__(16) __nv_bfloat16 o[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16(fn[j]);
    *reinterpret_cast<uint4*>(xn_out + (size_t)m * hh + (size_t)c * 8) = *reinterpret_cast<const uint4*>(o);
    if (xn_q8_out != nullptr) {
        float mx = fmaxf(fmaxf(fmaxf(fabsf(fn[0]), fabsf(fn[1])), fmaxf(fabsf(fn[2]), fabsf(fn[3]))),
                         fmaxf(fmaxf(fabsf(fn[4]), fabsf(fn[5])), fmaxf(fabsf(fn[6]), fabsf(fn[7]))));
        mx = warp_max(mx);
        if (lane == 0) red[warp] = mx;
        __syncthreads();
        if (tid == 0) {
            float gmx = fmaxf(fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3])),
                              fmaxf(fmaxf(red[4], red[5]), fmaxf(fmaxf(red[6], red[7]), fmaxf(red[8], red[9]))));
            s_rstd = (gmx > 0.f) ? (127.f / gmx) : 0.f;
            xn_gsc_out[(size_t)m * nb + g] = gmx * (1.f / 127.f);
        }
        __syncthreads();
        float inv = s_rstd;
        int q[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            q[j] = __float2int_rn(fminf(fmaxf(fn[j] * inv, -127.f), 127.f)) & 0xFF;
        }
        int w_lo = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
        int w_hi = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
        *reinterpret_cast<int2*>(xn_q8_out + (size_t)m * (hh >> 2) + (size_t)c * 2) = make_int2(w_lo, w_hi);
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_hc_combine_norm_group_kernel(
    const uint16_t* __restrict__ hc_in, const uint16_t* __restrict__ y, const float* __restrict__ inject,
    const float* __restrict__ norm_w, uint16_t* __restrict__ hc_out, uint16_t* __restrict__ xn_out,
    int hh, int h, int nb, float eps
) {
    __shared__ float red[8];
    __shared__ float s_rstd;
    const int g = blockIdx.x;
    const int m = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int h8 = h >> 3;
    const int base_c = g * h8;
    float wv[4][8];
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        int loc = tid + it * 256;
        if (loc < h8) {
            int c = base_c + loc;
            float4 w0 = __ldg(reinterpret_cast<const float4*>(norm_w + (size_t)c * 8));
            float4 w1 = __ldg(reinterpret_cast<const float4*>(norm_w + (size_t)c * 8 + 4));
            wv[it][0] = w0.x; wv[it][1] = w0.y; wv[it][2] = w0.z; wv[it][3] = w0.w;
            wv[it][4] = w1.x; wv[it][5] = w1.y; wv[it][6] = w1.z; wv[it][7] = w1.w;
        }
    }
    NVK_PDL_PROLOG();
    float f[4][8];
    float ss = 0.f;
    float inj = (y != nullptr) ? __ldg(&inject[(size_t)m * nb + g]) : 0.f;
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        int loc = tid + it * 256;
        if (loc < h8) {
            int c = base_c + loc;
            uint4 v = *reinterpret_cast<const uint4*>(hc_in + (size_t)m * hh + (size_t)c * 8);
            unpack8_bf16(v, f[it]);
            if (y != nullptr) {
                uint4 yv = *reinterpret_cast<const uint4*>(y + (size_t)m * h + (size_t)loc * 8);
                float yf[8];
                unpack8_bf16(yv, yf);
                __align__(16) __nv_bfloat16 o[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    float vv = f[it][j] + yf[j] * inj;
                    o[j] = __float2bfloat16(vv);
                    f[it][j] = __bfloat162float(o[j]);
                }
                *reinterpret_cast<uint4*>(hc_out + (size_t)m * hh + (size_t)c * 8) = *reinterpret_cast<const uint4*>(o);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) ss += f[it][j] * f[it][j];
        }
    }
    ss = warp_sum(ss);
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    if (tid == 0) {
        float tot = red[0] + red[1] + red[2] + red[3] + red[4] + red[5] + red[6] + red[7];
        s_rstd = rsqrtf(tot / (float)h + eps);
    }
    __syncthreads();
    float r = s_rstd;
    #pragma unroll
    for (int it = 0; it < 4; ++it) {
        int loc = tid + it * 256;
        if (loc < h8) {
            int c = base_c + loc;
            __align__(16) __nv_bfloat16 o[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) o[j] = __float2bfloat16(f[it][j] * r * wv[it][j]);
            *reinterpret_cast<uint4*>(xn_out + (size_t)m * hh + (size_t)c * 8) = *reinterpret_cast<const uint4*>(o);
        }
    }
    NVK_PDL_EPILOG();
}

extern "C" int nv_kernels_q4exp_hc_combine_norm(
    void* stream, const uint16_t* hc_in, const uint16_t* y, const float* inject, const float* norm_w,
    uint16_t* hc_out, uint16_t* xn_out, int m, int hh, int h, int nb, float eps
) {
    if (m <= 0) return 0;
    if (hh % 2048 != 0 || hh % nb != 0 || nb > 8 || h % 8 != 0 || hh != nb * h) return -2;
    if (hh / 8 > 256 * kQ4ExpCombineNormChunksPerThread) return -3;
    if ((y == nullptr) != (hc_out == nullptr) || (y != nullptr && inject == nullptr)) return -4;
    if (m > 65535) return -5;
    cudaStream_t s = (cudaStream_t)stream;
    const char* use_cluster = getenv("NV_Q4EXP_HC_NORM_CLUSTER");
    bool force_cluster = use_cluster && use_cluster[0] == '1' && use_cluster[1] == '\0';
    if (!force_cluster && h == 2560) {
        dim3 grid((unsigned)nb, (unsigned)m);
        int* q8 = (g_q4exp_hc_xn_q8 != nullptr && m <= 8 && hh == 10240 && nb == 4) ? g_q4exp_hc_xn_q8 : nullptr;
        float* gsc = q8 ? g_q4exp_hc_xn_gsc : nullptr;
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(cfg, grid, dim3(320), 0, s);
            cudaLaunchKernelEx(&cfg, q4exp_hc_combine_norm_group_h2560_kernel,
                hc_in, y, inject, norm_w, hc_out, xn_out, q8, gsc, hh, nb, eps);
        } else {
            q4exp_hc_combine_norm_group_h2560_kernel<<<grid, 320, 0, s>>>(
                hc_in, y, inject, norm_w, hc_out, xn_out, q8, gsc, hh, nb, eps);
        }
        return (int)cudaGetLastError();
    }
    if (!force_cluster && h <= 8192) {
        dim3 grid((unsigned)nb, (unsigned)m);
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(cfg, grid, dim3(256), 0, s);
            cudaLaunchKernelEx(&cfg, q4exp_hc_combine_norm_group_kernel,
                hc_in, y, inject, norm_w, hc_out, xn_out, hh, h, nb, eps);
        } else {
            q4exp_hc_combine_norm_group_kernel<<<grid, 256, 0, s>>>(
                hc_in, y, inject, norm_w, hc_out, xn_out, hh, h, nb, eps);
        }
        return (int)cudaGetLastError();
    }
    if (q4exp_device_cc_major() < 9) {
        q4exp_hc_combine_norm_kernel<kQ4ExpCombineNormChunksPerThread><<<m, 256, 0, s>>>(
            hc_in, y, inject, norm_w, hc_out, xn_out, hh, h, nb, eps);
        return (int)cudaGetLastError();
    }
    int iters = (hh / 8 + 255) / 256;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3(8, m), dim3(32 * iters), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_hc_combine_norm_cluster_kernel,
            hc_in, y, inject, norm_w, hc_out, xn_out, hh, h, nb, eps);
    } else {
        q4exp_hc_combine_norm_cluster_kernel<<<dim3(8, m), 32 * iters, 0, s>>>(
            hc_in, y, inject, norm_w, hc_out, xn_out, hh, h, nb, eps);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_gateup_silu(cudaStream_t s, const uint8_t* wg, const float* rsg, const uint8_t* wu, const float* rsu,
                                    const uint16_t* x, uint16_t* act, int N, int K) {
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(float2);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(optin, (const void*)q4exp_fp8_gateup_silu_kernel<TM>, smem);
    if (orc != 0) return orc;
    dim3 grid((unsigned)((N + kQ4ExpFp8Rows - 1) / kQ4ExpFp8Rows));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(kQ4ExpFp8Block), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_fp8_gateup_silu_kernel<TM>,
            wg, rsg, wu, rsu, reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(act), N, K);
    } else {
        q4exp_fp8_gateup_silu_kernel<TM><<<grid, kQ4ExpFp8Block, smem, s>>>(
            wg, rsg, wu, rsu, reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(act), N, K);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_down_tail(cudaStream_t s, const uint8_t* wq, const float* rs, const uint16_t* act,
                                  const uint16_t* y_down, long long slot_stride, const float* topk_w, int k,
                                  const uint16_t* sg, uint16_t* out, int N, int K) {
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(float2);
    if (K <= 1024 && TM <= 8) {
        static DynamicSmemOptin optin_s8;
        int orc = raise_dynamic_smem_optin_never_lowering_it(optin_s8, (const void*)q4exp_fp8_down_tail_sub8_kernel<TM>, smem);
        if (orc != 0) return orc;
        int num_tiles = (N + 15) / 16;
    dim3 grid((unsigned)(num_tiles < 752 ? num_tiles : 752));
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(cfg, grid, dim3(128), smem, s);
            cudaLaunchKernelEx(&cfg, q4exp_fp8_down_tail_sub8_kernel<TM>,
                wq, rs, reinterpret_cast<const __nv_bfloat16*>(act), reinterpret_cast<const __nv_bfloat16*>(y_down), slot_stride,
                topk_w, k, reinterpret_cast<const __nv_bfloat16*>(sg), reinterpret_cast<__nv_bfloat16*>(out), N, K);
        } else {
            q4exp_fp8_down_tail_sub8_kernel<TM><<<grid, 128, smem, s>>>(
                wq, rs, reinterpret_cast<const __nv_bfloat16*>(act), reinterpret_cast<const __nv_bfloat16*>(y_down), slot_stride,
                topk_w, k, reinterpret_cast<const __nv_bfloat16*>(sg), reinterpret_cast<__nv_bfloat16*>(out), N, K);
        }
        return (int)cudaGetLastError();
    }
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(optin, (const void*)q4exp_fp8_down_tail_kernel<TM>, smem);
    if (orc != 0) return orc;
    dim3 grid((unsigned)((N + 2 * kQ4ExpFp8Rows - 1) / (2 * kQ4ExpFp8Rows)));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(kQ4ExpFp8Block), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_fp8_down_tail_kernel<TM>,
            wq, rs, reinterpret_cast<const __nv_bfloat16*>(act), reinterpret_cast<const __nv_bfloat16*>(y_down), slot_stride,
            topk_w, k, reinterpret_cast<const __nv_bfloat16*>(sg), reinterpret_cast<__nv_bfloat16*>(out), N, K);
    } else {
        q4exp_fp8_down_tail_kernel<TM><<<grid, kQ4ExpFp8Block, smem, s>>>(
            wq, rs, reinterpret_cast<const __nv_bfloat16*>(act), reinterpret_cast<const __nv_bfloat16*>(y_down), slot_stride,
            topk_w, k, reinterpret_cast<const __nv_bfloat16*>(sg), reinterpret_cast<__nv_bfloat16*>(out), N, K);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_hc_down_fp8(
    cudaStream_t s, const uint16_t* xn, const uint8_t* w_down, const float* rs_down,
    const uint16_t* w_inject, float* lora_out, float* inject_out, int hh, int nb, int rank
) {
    int rows = rank + (w_inject ? nb : 0);
    dim3 grid((unsigned)rows, 1);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_hc_down_fp8_kernel<TM>,
            xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
    } else {
        q4exp_hc_down_fp8_kernel<TM><<<grid, 256, 0, s>>>(
            xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_hc_up_mix_fp8(
    cudaStream_t s, const float* lora, const uint8_t* w_up, const float* rs_up,
    const uint16_t* xn, uint16_t* mixed, int h, int nb, int rank
) {
    size_t smem = (size_t)TM * rank * sizeof(float);
    const char* old_up = getenv("NV_Q4EXP_HC_UP_OLD");
    bool force_old = old_up && old_up[0] == '1' && old_up[1] == '\0';
    if (!force_old && rank == 320 && nb == 4) {
        dim3 grid((unsigned)((h + 3) / 4), 1);
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
            cudaLaunchKernelEx(&cfg, q4exp_hc_up_mix_fp8_r320_nb4_kernel<TM>,
                lora, w_up, rs_up, xn, mixed, h);
        } else {
            q4exp_hc_up_mix_fp8_r320_nb4_kernel<TM><<<grid, 256, smem, s>>>(
                lora, w_up, rs_up, xn, mixed, h);
        }
        return (int)cudaGetLastError();
    }
    if (!force_old && rank == 320 && nb == 8) {
        dim3 grid((unsigned)((h + 3) / 4), 1);
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
            cudaLaunchKernelEx(&cfg, q4exp_hc_up_mix_fp8_r320_nb8_kernel<TM>,
                lora, w_up, rs_up, xn, mixed, h);
        } else {
            q4exp_hc_up_mix_fp8_r320_nb8_kernel<TM><<<grid, 256, smem, s>>>(
                lora, w_up, rs_up, xn, mixed, h);
        }
        return (int)cudaGetLastError();
    }
    dim3 grid((unsigned)((h + 7) / 8), 1);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(128), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_hc_up_mix_fp8_kernel<TM>,
            lora, w_up, rs_up, xn, mixed, h, nb, rank);
    } else {
        q4exp_hc_up_mix_fp8_kernel<TM><<<grid, 128, smem, s>>>(
            lora, w_up, rs_up, xn, mixed, h, nb, rank);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_fp8_qkv_idx_gemv(
    cudaStream_t s,
    const uint8_t* wq, const float* rsq, const uint8_t* wk, const float* rsk,
    const uint8_t* wv, const float* rsv, const uint8_t* wi, const float* rsi,
    const uint16_t* x, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yi,
    int nq, int nk, int nv, int ni, int K
) {
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(float2);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(optin, (const void*)q4exp_fp8_qkv_idx_gemv_kernel<TM>, smem);
    if (orc != 0) return orc;
    int total = nq + nk + nv + ni;
    dim3 grid((unsigned)((total + 2 * kQ4ExpFp8Rows - 1) / (2 * kQ4ExpFp8Rows)));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(kQ4ExpFp8Block), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_fp8_qkv_idx_gemv_kernel<TM>,
            wq, rsq, wk, rsk, wv, rsv, wi, rsi,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk),
            reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yi),
            nq, nk, nv, ni, K);
    } else {
        q4exp_fp8_qkv_idx_gemv_kernel<TM><<<grid, kQ4ExpFp8Block, smem, s>>>(
            wq, rsq, wk, rsk, wv, rsv, wi, rsi,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk),
            reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yi),
            nq, nk, nv, ni, K);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_hc_down_fp8(
    void* stream, const uint16_t* xn, const uint8_t* w_down, const float* rs_down,
    const uint16_t* w_inject, float* lora_out, float* inject_out, int m, int hh, int nb, int rank
) {
    if (m <= 0) return 0;
    if (hh % 16 != 0 || hh % nb != 0 || nb > 8 || rank <= 0) return -2;
    cudaStream_t s = (cudaStream_t)stream;
    switch (m) {
        case 1: return q4exp_launch_hc_down_fp8<1>(s, xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
        case 2: return q4exp_launch_hc_down_fp8<2>(s, xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
        case 3: return q4exp_launch_hc_down_fp8<3>(s, xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
        case 4: return q4exp_launch_hc_down_fp8<4>(s, xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
        case 5: return q4exp_launch_hc_down_fp8<5>(s, xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
        case 6: return q4exp_launch_hc_down_fp8<6>(s, xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
        case 7: return q4exp_launch_hc_down_fp8<7>(s, xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
        case 8: return q4exp_launch_hc_down_fp8<8>(s, xn, w_down, rs_down, w_inject, lora_out, inject_out, hh, nb, rank);
        default: return -1;
    }
}

extern "C" int nv_kernels_q4exp_hc_up_mix_fp8(
    void* stream, const float* lora, const uint8_t* w_up, const float* rs_up,
    const uint16_t* xn, uint16_t* mixed, int m, int h, int nb, int rank
) {
    if (m <= 0) return 0;
    if (rank <= 0 || (rank % 16) != 0 || h <= 0 || nb <= 0 || nb > 8) return -2;
    if ((size_t)m * rank * sizeof(float) > 48 * 1024) return -3;
    cudaStream_t s = (cudaStream_t)stream;
    switch (m) {
        case 1: return q4exp_launch_hc_up_mix_fp8<1>(s, lora, w_up, rs_up, xn, mixed, h, nb, rank);
        case 2: return q4exp_launch_hc_up_mix_fp8<2>(s, lora, w_up, rs_up, xn, mixed, h, nb, rank);
        case 3: return q4exp_launch_hc_up_mix_fp8<3>(s, lora, w_up, rs_up, xn, mixed, h, nb, rank);
        case 4: return q4exp_launch_hc_up_mix_fp8<4>(s, lora, w_up, rs_up, xn, mixed, h, nb, rank);
        case 5: return q4exp_launch_hc_up_mix_fp8<5>(s, lora, w_up, rs_up, xn, mixed, h, nb, rank);
        case 6: return q4exp_launch_hc_up_mix_fp8<6>(s, lora, w_up, rs_up, xn, mixed, h, nb, rank);
        case 7: return q4exp_launch_hc_up_mix_fp8<7>(s, lora, w_up, rs_up, xn, mixed, h, nb, rank);
        case 8: return q4exp_launch_hc_up_mix_fp8<8>(s, lora, w_up, rs_up, xn, mixed, h, nb, rank);
        default: return -1;
    }
}

extern "C" int nv_kernels_q4exp_fp8_qkv_idx_gemv(
    void* stream,
    const uint8_t* wq, const float* rsq, const uint8_t* wk, const float* rsk,
    const uint8_t* wv, const float* rsv, const uint8_t* wi, const float* rsi,
    const uint16_t* x, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yi,
    int nq, int nk, int nv, int ni, int k, int m
) {
    if (m <= 0 || nq + nk + nv + ni <= 0) return 0;
    if ((k & 15) != 0 || (size_t)m * (k >> 4) * 9 * sizeof(float2) > (size_t)nvk_max_dynamic_smem_optin()) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    switch (m) {
        case 1: return q4exp_launch_fp8_qkv_idx_gemv<1>(s, wq, rsq, wk, rsk, wv, rsv, wi, rsi, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 2: return q4exp_launch_fp8_qkv_idx_gemv<2>(s, wq, rsq, wk, rsk, wv, rsv, wi, rsi, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 3: return q4exp_launch_fp8_qkv_idx_gemv<3>(s, wq, rsq, wk, rsk, wv, rsv, wi, rsi, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 4: return q4exp_launch_fp8_qkv_idx_gemv<4>(s, wq, rsq, wk, rsk, wv, rsv, wi, rsi, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 5: return q4exp_launch_fp8_qkv_idx_gemv<5>(s, wq, rsq, wk, rsk, wv, rsv, wi, rsi, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 6: return q4exp_launch_fp8_qkv_idx_gemv<6>(s, wq, rsq, wk, rsk, wv, rsv, wi, rsi, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 7: return q4exp_launch_fp8_qkv_idx_gemv<7>(s, wq, rsq, wk, rsk, wv, rsv, wi, rsi, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 8: return q4exp_launch_fp8_qkv_idx_gemv<8>(s, wq, rsq, wk, rsk, wv, rsv, wi, rsi, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        default: return -1;
    }
}

extern "C" int nv_kernels_q4exp_fp8_gateup_silu(
    void* stream, const uint8_t* wg, const float* rsg, const uint8_t* wu, const float* rsu,
    const uint16_t* x, uint16_t* act, int m, int n, int k
) {
    if (m <= 0 || n <= 0) return 0;
    if ((k & 15) != 0 || (size_t)m * (k >> 4) * 9 * sizeof(float2) > (size_t)nvk_max_dynamic_smem_optin()) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    switch (m) {
        case 1: return q4exp_launch_gateup_silu<1>(s, wg, rsg, wu, rsu, x, act, n, k);
        case 2: return q4exp_launch_gateup_silu<2>(s, wg, rsg, wu, rsu, x, act, n, k);
        case 3: return q4exp_launch_gateup_silu<3>(s, wg, rsg, wu, rsu, x, act, n, k);
        case 4: return q4exp_launch_gateup_silu<4>(s, wg, rsg, wu, rsu, x, act, n, k);
        case 5: return q4exp_launch_gateup_silu<5>(s, wg, rsg, wu, rsu, x, act, n, k);
        case 6: return q4exp_launch_gateup_silu<6>(s, wg, rsg, wu, rsu, x, act, n, k);
        default: return -1;
    }
}

extern "C" int nv_kernels_q4exp_fp8_down_tail(
    void* stream, const uint8_t* wq, const float* rs, const uint16_t* act, const uint16_t* y_down,
    long long slot_stride, const float* topk_w, int topk, const uint16_t* sg, uint16_t* out, int m, int n, int k
) {
    if (m <= 0 || n <= 0) return 0;
    if ((k & 15) != 0 || topk <= 0 || topk > kQ4ExpTailMaxSlots || (size_t)m * (k >> 4) * 9 * sizeof(float2) > (size_t)nvk_max_dynamic_smem_optin()) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    switch (m) {
        case 1: return q4exp_launch_down_tail<1>(s, wq, rs, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 2: return q4exp_launch_down_tail<2>(s, wq, rs, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 3: return q4exp_launch_down_tail<3>(s, wq, rs, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 4: return q4exp_launch_down_tail<4>(s, wq, rs, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 5: return q4exp_launch_down_tail<5>(s, wq, rs, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 6: return q4exp_launch_down_tail<6>(s, wq, rs, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        default: return -1;
    }
}

namespace {

constexpr int kQ4ExpArgmaxBlock = 256;
constexpr int kQ4ExpArgmaxParts = 128;

__global__ void q4exp_argmax_bf16_rows_stage1_kernel(
    const uint16_t* __restrict__ logits,
    int n,
    float* __restrict__ part_val,
    int* __restrict__ part_idx
) {
    __shared__ float sval[8];
    __shared__ int sidx[8];
    const uint16_t* row = logits + (size_t)blockIdx.y * n;
    int tid = threadIdx.x;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    int n8 = n >> 3;
    if ((((uintptr_t)row) & 15u) == 0) {
        const uint4* r4 = reinterpret_cast<const uint4*>(row);
        for (int c = blockIdx.x * kQ4ExpArgmaxBlock + tid; c < n8; c += gridDim.x * kQ4ExpArgmaxBlock) {
            uint4 w = r4[c];
            float f[8];
            unpack8_bf16(w, f);
            int base = c << 3;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                float v = f[j];
                int idx = base + j;
                if (!isnan(v) && (v > best || (v == best && idx < bidx))) {
                    best = v;
                    bidx = idx;
                }
            }
        }
        for (int i = (n8 << 3) + blockIdx.x * kQ4ExpArgmaxBlock + tid; i < n; i += gridDim.x * kQ4ExpArgmaxBlock) {
            float v = ld_bf16(row + i);
            if (!isnan(v) && (v > best || (v == best && i < bidx))) {
                best = v;
                bidx = i;
            }
        }
    } else {
        for (int i = blockIdx.x * kQ4ExpArgmaxBlock + tid; i < n; i += gridDim.x * kQ4ExpArgmaxBlock) {
            float v = ld_bf16(row + i);
            if (!isnan(v) && (v > best || (v == best && i < bidx))) {
                best = v;
                bidx = i;
            }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, bidx, o);
        if (ov > best || (ov == best && oi < bidx)) {
            best = ov;
            bidx = oi;
        }
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        sval[warp] = best;
        sidx[warp] = bidx;
    }
    __syncthreads();
    if (warp == 0) {
        best = (lane < 8) ? sval[lane] : -INFINITY;
        bidx = (lane < 8) ? sidx[lane] : 0x7fffffff;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, bidx, o);
            if (ov > best || (ov == best && oi < bidx)) {
                best = ov;
                bidx = oi;
            }
        }
        if (lane == 0) {
            part_val[(size_t)blockIdx.y * kQ4ExpArgmaxParts + blockIdx.x] = best;
            part_idx[(size_t)blockIdx.y * kQ4ExpArgmaxParts + blockIdx.x] = bidx;
        }
    }
}

__global__ void q4exp_argmax_bf16_rows_stage2_kernel(
    const float* __restrict__ part_val,
    const int* __restrict__ part_idx,
    uint32_t* __restrict__ out
) {
    __shared__ float sval[4];
    __shared__ int sidx[4];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float best = part_val[(size_t)row * kQ4ExpArgmaxParts + tid];
    int bidx = part_idx[(size_t)row * kQ4ExpArgmaxParts + tid];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffffu, best, o);
        int oi = __shfl_xor_sync(0xffffffffu, bidx, o);
        if (ov > best || (ov == best && oi < bidx)) {
            best = ov;
            bidx = oi;
        }
    }
    int lane = tid & 31;
    int warp = tid >> 5;
    if (lane == 0) {
        sval[warp] = best;
        sidx[warp] = bidx;
    }
    __syncthreads();
    if (warp == 0) {
        best = (lane < 4) ? sval[lane] : -INFINITY;
        bidx = (lane < 4) ? sidx[lane] : 0x7fffffff;
#pragma unroll
        for (int o = 2; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffffu, best, o);
            int oi = __shfl_xor_sync(0xffffffffu, bidx, o);
            if (ov > best || (ov == best && oi < bidx)) {
                best = ov;
                bidx = oi;
            }
        }
        if (lane == 0) {
            out[row] = (bidx == 0x7fffffff) ? 0u : (uint32_t)bidx;
        }
    }
}

}

extern "C" int nv_kernels_q4exp_argmax_bf16_rows(
    void* stream,
    const uint16_t* logits,
    int rows,
    int n,
    float* part_val,
    int* part_idx,
    uint32_t* out
) {
    if (rows <= 0 || n <= 0) return -1;
    if (rows > NVK_MAX_GRID_Y) return NVK_ERR_GRID_AXIS;
    if (logits == nullptr || part_val == nullptr || part_idx == nullptr || out == nullptr) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 g1(kQ4ExpArgmaxParts, (unsigned)rows);
    q4exp_argmax_bf16_rows_stage1_kernel<<<g1, kQ4ExpArgmaxBlock, 0, s>>>(
        logits, n, part_val, part_idx);
    q4exp_argmax_bf16_rows_stage2_kernel<<<(unsigned)rows, kQ4ExpArgmaxParts, 0, s>>>(
        part_val, part_idx, out);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_argmax_bf16_rows_parts(void) { return kQ4ExpArgmaxParts; }

namespace {

__device__ __forceinline__ float q4exp_decode_ue4m3(uint8_t b) {
    unsigned biased = (unsigned)(b >> 3) & 0x0Fu;
    unsigned mant = (unsigned)(b & 0x07u);
    float norm = __uint_as_float(((biased + 120u) << 23) | (mant << 20));
    return biased ? norm : (float)mant * 0.001953125f;
}

__device__ __forceinline__ void q4exp_nibbles8_to_i8(unsigned w, unsigned& lo, unsigned& hi) {
    constexpr unsigned kMag0to3 = 0x03020100u;
    constexpr unsigned kMag4to7 = 0x0C080604u;
    unsigned mag_sel = w & 0x77777777u;
    unsigned sgn_sel = (w >> 3) & 0x11111111u;
    unsigned mag_lo = __byte_perm(kMag0to3, kMag4to7, mag_sel);
    unsigned sgn_lo = __byte_perm(0x0000FF00u, 0u, sgn_sel);
    lo = __vsub4(mag_lo ^ sgn_lo, sgn_lo);
    unsigned mag_hi = __byte_perm(kMag0to3, kMag4to7, mag_sel >> 16);
    unsigned sgn_hi = __byte_perm(0x0000FF00u, 0u, sgn_sel >> 16);
    hi = __vsub4(mag_hi ^ sgn_hi, sgn_hi);
}

__device__ __forceinline__ unsigned long long q4exp_pack_argmax_key(float v, unsigned idx) {
    unsigned b = __float_as_uint(v);
    unsigned ord = (v >= 0.0f) ? (b | 0x80000000u) : ~b;
    return ((unsigned long long)ord << 32) | (unsigned long long)(idx ^ 0xFFFFFFFFu);
}

__global__ void __launch_bounds__(256) q4exp_lm_head_nvfp4_argmax_m1_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ scales,
    const uint16_t* __restrict__ x,
    unsigned long long* __restrict__ packed_best,
    int N,
    int K
) {
    extern __shared__ int xs_q8_words[];
    __shared__ float s_amax[8];
    __shared__ unsigned long long warp_best[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int kb_total = K >> 4;
    int num_tiles = (N + 15) >> 4;

    NVK_PDL_PROLOG();

    int n8 = K >> 3;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    float local_max = 0.f;
    for (int i = tid; i < n8; i += 256) {
        float f[8];
        unpack8_bf16(__ldg(&x4[i]), f);
#pragma unroll
        for (int j = 0; j < 8; ++j) local_max = fmaxf(local_max, fabsf(f[j]));
    }
    local_max = warp_max(local_max);
    if (lane == 0) s_amax[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < 8) ? s_amax[lane] : 0.f;
        m = warp_max(m);
        if (lane == 0) s_amax[0] = (m > 0.f) ? (127.f / m) : 0.f;
    }
    __syncthreads();
    float inv = s_amax[0];
    for (int i = tid; i < n8; i += 256) {
        float f[8];
        unpack8_bf16(__ldg(&x4[i]), f);
        int q[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            int qi = __float2int_rn(fminf(fmaxf(f[j] * inv, -127.f), 127.f));
            q[j] = qi & 0xFF;
        }
        xs_q8_words[2 * i] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
        xs_q8_words[2 * i + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
    }
    __syncthreads();

    unsigned long long my_best = 0ull;
    const int4* xs_i4 = reinterpret_cast<const int4*>(xs_q8_words);
    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        int n0 = tile * 16 + warp;
        int n1 = n0 + 8;
        bool has0 = n0 < N;
        bool has1 = n1 < N;
        if (!has0) continue;
        const uint2* wa = reinterpret_cast<const uint2*>(wq + (size_t)n0 * (K >> 1));
        const uint2* wb = reinterpret_cast<const uint2*>(wq + (size_t)(has1 ? n1 : n0) * (K >> 1));
        const uint8_t* sa_ptr = scales + (size_t)n0 * kb_total;
        const uint8_t* sb_ptr = scales + (size_t)(has1 ? n1 : n0) * kb_total;
        int kb = lane;
        uint2 ra = (kb < kb_total) ? __ldcs(&wa[kb]) : make_uint2(0u, 0u);
        uint2 rb = (has1 && kb < kb_total) ? __ldcs(&wb[kb]) : make_uint2(0u, 0u);
        uint8_t sca = (kb < kb_total) ? __ldcs(&sa_ptr[kb]) : 0u;
        uint8_t scb = (has1 && kb < kb_total) ? __ldcs(&sb_ptr[kb]) : 0u;
        float acc0 = 0.f;
        float acc1 = 0.f;
        while (kb < kb_total) {
            int kb_next = kb + 32;
            uint2 na = (kb_next < kb_total) ? __ldcs(&wa[kb_next]) : make_uint2(0u, 0u);
            uint2 nb = (has1 && kb_next < kb_total) ? __ldcs(&wb[kb_next]) : make_uint2(0u, 0u);
            uint8_t nsa = (kb_next < kb_total) ? __ldcs(&sa_ptr[kb_next]) : 0u;
            uint8_t nsb = (has1 && kb_next < kb_total) ? __ldcs(&sb_ptr[kb_next]) : 0u;
            int4 xv = xs_i4[kb];
            unsigned a0, a1, a2, a3;
            q4exp_nibbles8_to_i8(ra.x, a0, a1);
            q4exp_nibbles8_to_i8(ra.y, a2, a3);
            int d0 = __dp4a((int)a0, xv.x, 0);
            d0 = __dp4a((int)a1, xv.y, d0);
            d0 = __dp4a((int)a2, xv.z, d0);
            d0 = __dp4a((int)a3, xv.w, d0);
            acc0 += q4exp_decode_ue4m3(sca) * (float)d0;
            if (has1) {
                unsigned b0, b1, b2, b3;
                q4exp_nibbles8_to_i8(rb.x, b0, b1);
                q4exp_nibbles8_to_i8(rb.y, b2, b3);
                int d1 = __dp4a((int)b0, xv.x, 0);
                d1 = __dp4a((int)b1, xv.y, d1);
                d1 = __dp4a((int)b2, xv.z, d1);
                d1 = __dp4a((int)b3, xv.w, d1);
                acc1 += q4exp_decode_ue4m3(scb) * (float)d1;
            }
            kb = kb_next;
            ra = na;
            rb = nb;
            sca = nsa;
            scb = nsb;
        }
        acc0 = warp_sum(acc0);
        if (has1) acc1 = warp_sum(acc1);
        if (lane == 0) {
            unsigned long long p0 = q4exp_pack_argmax_key(acc0, (unsigned)n0);
            if (p0 > my_best) my_best = p0;
            if (has1) {
                unsigned long long p1 = q4exp_pack_argmax_key(acc1, (unsigned)n1);
                if (p1 > my_best) my_best = p1;
            }
        }
    }
    if (lane == 0) warp_best[warp] = my_best;
    __syncthreads();
    if (tid == 0) {
        unsigned long long blk = 0ull;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (warp_best[w] > blk) blk = warp_best[w];
        }
        if (blk > *packed_best) atomicMax(packed_best, blk);
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_unpack_packed_argmax_kernel(
    const unsigned long long* __restrict__ packed_best,
    uint32_t* __restrict__ out_id
) {
    NVK_PDL_PROLOG();
    unsigned long long p = *packed_best;
    out_id[0] = (p == 0ull) ? 0u : (((unsigned)(p & 0xFFFFFFFFull)) ^ 0xFFFFFFFFu);
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void q4exp_lm_head_fp8_argmax_mt_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ rs,
    const __nv_bfloat16* __restrict__ x,
    unsigned long long* __restrict__ packed_best,
    int N,
    int K
) {
    extern __shared__ float2 xshf[];
    __shared__ unsigned long long warp_best[TM][8];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int Kv = K >> 4;
    const int rowpad = Kv * 9;
    const int num_tiles = (N + 15) >> 4;

    NVK_PDL_PROLOG();

    q4exp_stage_rows_f32<TM>(x, xshf, K);
    __syncthreads();

    unsigned long long my_best[TM];
#pragma unroll
    for (int m = 0; m < TM; ++m) my_best[m] = 0ull;

    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        const int n0 = tile * 16 + warp;
        const int n1 = n0 + 8;
        const bool has0 = n0 < N;
        const bool has1 = n1 < N;
        if (!has0) continue;
        const int4* wa = reinterpret_cast<const int4*>(wq + (size_t)n0 * K);
        const int4* wb = reinterpret_cast<const int4*>(wq + (size_t)(has1 ? n1 : n0) * K);
        const float rs0 = __ldg(&rs[n0]);
        const float rs1 = has1 ? __ldg(&rs[n1]) : 0.f;
        int v = lane;
        int4 ra = (v < Kv) ? __ldcs(&wa[v]) : make_int4(0, 0, 0, 0);
        int4 rb = (has1 && v < Kv) ? __ldcs(&wb[v]) : make_int4(0, 0, 0, 0);
        float acc0[TM];
        float acc1[TM];
#pragma unroll
        for (int m = 0; m < TM; ++m) {
            acc0[m] = 0.f;
            acc1[m] = 0.f;
        }
        while (v < Kv) {
            int vn = v + 32;
            int4 na = (vn < Kv) ? __ldcs(&wa[vn]) : make_int4(0, 0, 0, 0);
            int4 nb = (has1 && vn < Kv) ? __ldcs(&wb[vn]) : make_int4(0, 0, 0, 0);
            float wfa[16];
            float wfb[16];
            q4exp_deq_e4m3x16(ra, wfa);
            if (has1) q4exp_deq_e4m3x16(rb, wfb);
            const float2* base = xshf + v * 9;
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                const float2* xp = base + m * rowpad;
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    float2 xv = xp[p];
                    acc0[m] += wfa[2 * p] * xv.x + wfa[2 * p + 1] * xv.y;
                    if (has1) acc1[m] += wfb[2 * p] * xv.x + wfb[2 * p + 1] * xv.y;
                }
            }
            v = vn;
            ra = na;
            rb = nb;
        }
#pragma unroll
        for (int m = 0; m < TM; ++m) {
            acc0[m] = warp_sum(acc0[m]);
            if (has1) acc1[m] = warp_sum(acc1[m]);
            if (lane == 0) {
                float v0 = __bfloat162float(__float2bfloat16(acc0[m] * rs0));
                unsigned long long p0 = q4exp_pack_argmax_key(v0, (unsigned)n0);
                if (p0 > my_best[m]) my_best[m] = p0;
                if (has1) {
                    float v1 = __bfloat162float(__float2bfloat16(acc1[m] * rs1));
                    unsigned long long p1 = q4exp_pack_argmax_key(v1, (unsigned)n1);
                    if (p1 > my_best[m]) my_best[m] = p1;
                }
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int m = 0; m < TM; ++m) warp_best[m][warp] = my_best[m];
    }
    __syncthreads();
    if (tid < TM) {
        unsigned long long blk = 0ull;
#pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (warp_best[tid][w] > blk) blk = warp_best[tid][w];
        }
        if (blk > packed_best[tid]) atomicMax(&packed_best[tid], blk);
    }
    NVK_PDL_EPILOG();
}

__global__ void q4exp_unpack_packed_argmax_mt_kernel(
    const unsigned long long* __restrict__ packed_best,
    uint32_t* __restrict__ out_ids,
    int m
) {
    NVK_PDL_PROLOG();
    int i = threadIdx.x;
    if (i < m) {
        unsigned long long p = packed_best[i];
        out_ids[i] = (p == 0ull) ? 0u : (((unsigned)(p & 0xFFFFFFFFull)) ^ 0xFFFFFFFFu);
    }
    NVK_PDL_EPILOG();
}

template <int TM>
static int q4exp_launch_lm_head_fp8_argmax_mt(
    cudaStream_t s, const uint8_t* wq, const float* rs, const __nv_bfloat16* x,
    uint64_t* packed_best, uint32_t* out_ids, int N, int K
) {
    size_t smem = (size_t)TM * (size_t)(K >> 4) * 9 * sizeof(float2);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(optin, (const void*)q4exp_lm_head_fp8_argmax_mt_kernel<TM>, smem);
    if (orc != 0) return orc;
    cudaError_t me = cudaMemsetAsync(packed_best, 0, (size_t)TM * sizeof(uint64_t), s);
    if (me != cudaSuccess) return (int)me;
    int num_tiles = (N + 15) / 16;
    dim3 grid((unsigned)(num_tiles < 752 ? num_tiles : 752));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_lm_head_fp8_argmax_mt_kernel<TM>,
            wq, rs, x, reinterpret_cast<unsigned long long*>(packed_best), N, K);
        NVK_PDL_ATTR(cfg2, dim3(1), dim3((unsigned)TM), 0, s);
        cudaLaunchKernelEx(&cfg2, q4exp_unpack_packed_argmax_mt_kernel,
            reinterpret_cast<const unsigned long long*>(packed_best), out_ids, TM);
    } else {
        q4exp_lm_head_fp8_argmax_mt_kernel<TM><<<grid, 256, smem, s>>>(
            wq, rs, x, reinterpret_cast<unsigned long long*>(packed_best), N, K);
        q4exp_unpack_packed_argmax_mt_kernel<<<1, TM, 0, s>>>(
            reinterpret_cast<const unsigned long long*>(packed_best), out_ids, TM);
    }
    return (int)cudaGetLastError();
}

__global__ void q4exp_quant_rows_to_i8_kernel(
    const uint16_t* __restrict__ src_bf16,
    uint8_t* __restrict__ dst_i8,
    float* __restrict__ dst_sc,
    const int32_t* __restrict__ pos_dev,
    int row_offset,
    int rows_per_tok,
    int row_elems
) {
    __shared__ float s_amax[8];
    NVK_PDL_PROLOG();
    int r = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int dst_row;
    if (pos_dev != nullptr && rows_per_tok < 0) {
        int ratio = -rows_per_tok;
        int p0 = pos_dev[0];
        int b0 = p0 / ratio;
        int b_end = (p0 + row_offset) / ratio;
        dst_row = b0 + r;
        if (dst_row >= b_end) {
            NVK_PDL_EPILOG();
            return;
        }
    } else {
        int base_row = pos_dev ? (pos_dev[0] * rows_per_tok) : row_offset;
        dst_row = base_row + r;
    }
    const uint16_t* sr = src_bf16 + (size_t)dst_row * row_elems;
    int8_t* dr = reinterpret_cast<int8_t*>(dst_i8) + (size_t)dst_row * row_elems;
    float v = (tid < row_elems) ? ld_bf16(sr + tid) : 0.f;
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) s_amax[warp] = amax;
    __syncthreads();
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        float m = (lane < nw) ? s_amax[lane] : 0.f;
#pragma unroll
        for (int o = 4; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
        if (lane == 0) s_amax[0] = m;
    }
    __syncthreads();
    float m = s_amax[0];
    float sc = (m > 0.f) ? (m * (1.f / 127.f)) : 1.f;
    float inv = (m > 0.f) ? (127.f / m) : 0.f;
    if (tid < row_elems) {
        dr[tid] = (int8_t)__float2int_rn(fminf(fmaxf(v * inv, -127.f), 127.f));
    }
    if (tid == 0) dst_sc[dst_row] = sc;
    NVK_PDL_EPILOG();
}

template <int R_MAX, int KB>
__global__ void __launch_bounds__(QSA_SCORES_THREADS) q4exp_qsa_scores_i8_hd128_kernel(
    const uint16_t* __restrict__ q,
    const uint8_t* __restrict__ cmp_k_i8,
    const float* __restrict__ cmp_k_sc,
    const int32_t* __restrict__ nb_per_row,
    int m_rows,
    int nh,
    int nb_stride,
    float scale,
    float* __restrict__ scores
) {
    __shared__ int2 s_q_i8[R_MAX * 4 * 16];
    __shared__ float s_q_sc[R_MAX * 4];
    __shared__ int s_nb[R_MAX];
    NVK_PDL_PROLOG();
    int m0 = blockIdx.y * R_MAX;
    int rows = min(R_MAX, m_rows - m0);
    int tid = threadIdx.x;
    int cta_b0_init = blockIdx.x * (QSA_SCORES_HALF_WARPS * KB);
    int nb_max = 0;
    for (int r = 0; r < rows; ++r) {
        int nb = nb_per_row[m0 + r];
        if (nb > nb_max) nb_max = nb;
    }
    int nb_lim = min(nb_max, nb_stride);
    if (cta_b0_init >= nb_lim) {
        NVK_PDL_EPILOG();
        return;
    }
    if (tid < rows) s_nb[tid] = nb_per_row[m0 + tid];
    int hw = tid >> 4;
    int sublane = tid & 15;
    unsigned hmask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
    int n_qvecs = rows * nh;
    for (int v = hw; v < n_qvecs; v += QSA_SCORES_HALF_WARPS) {
        const uint4* q4 = reinterpret_cast<const uint4*>(q + (size_t)(m0 * nh + v) * 128);
        float qf[8];
        unpack8_bf16(__ldg(&q4[sublane]), qf);
        float mx = 0.f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fabsf(qf[j]));
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(hmask, mx, o, 16));
        float inv = (mx > 0.f) ? (127.f / mx) : 0.f;
        if (sublane == 0) s_q_sc[v] = mx * (1.f / 127.f) * scale;
        int qi[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) qi[j] = __float2int_rn(fminf(fmaxf(qf[j] * inv, -127.f), 127.f)) & 0xFF;
        int w0 = qi[0] | (qi[1] << 8) | (qi[2] << 16) | (qi[3] << 24);
        int w1 = qi[4] | (qi[5] << 8) | (qi[6] << 16) | (qi[7] << 24);
        s_q_i8[v * 16 + sublane] = make_int2(w0, w1);
    }
    __syncthreads();
    int stride_b = (int)gridDim.x * (QSA_SCORES_HALF_WARPS * KB);
    int b0 = cta_b0_init + hw * KB;
    int2 ki[KB];
    float ksc[KB];
#pragma unroll
    for (int kb = 0; kb < KB; ++kb) {
        int b = b0 + kb;
        ki[kb] = (b < nb_lim) ? __ldcs(reinterpret_cast<const int2*>(cmp_k_i8 + (size_t)b * 128 + sublane * 8))
                              : make_int2(0, 0);
        ksc[kb] = (b < nb_lim) ? __ldg(&cmp_k_sc[b]) : 0.f;
    }
    bool aligned_out = (KB == 4) && ((b0 & 3) == 0) && ((nb_stride & 3) == 0);
    while (b0 < nb_lim) {
        int b0_n = b0 + stride_b;
        int2 ki_n[KB];
        float ksc_n[KB];
        if (b0_n < nb_lim) {
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) {
                int b = b0_n + kb;
                ki_n[kb] = (b < nb_lim) ? __ldcs(reinterpret_cast<const int2*>(cmp_k_i8 + (size_t)b * 128 + sublane * 8))
                                        : make_int2(0, 0);
                ksc_n[kb] = (b < nb_lim) ? __ldg(&cmp_k_sc[b]) : 0.f;
            }
        }
        for (int r = 0; r < rows; ++r) {
            int nb_r = min(s_nb[r], nb_stride);
            if (b0 >= nb_r) continue;
            float s[KB];
#pragma unroll
            for (int kb = 0; kb < KB; ++kb) s[kb] = 0.f;
            if (nh == 4) {
                #pragma unroll
                for (int h = 0; h < 4; ++h) {
                    int2 qv = s_q_i8[(r * 4 + h) * 16 + sublane];
                    float qsc = s_q_sc[r * 4 + h];
#pragma unroll
                    for (int kb = 0; kb < KB; ++kb) {
                        int d = __dp4a(qv.x, ki[kb].x, 0);
                        d = __dp4a(qv.y, ki[kb].y, d);
#pragma unroll
                        for (int o = 8; o > 0; o >>= 1) d += __shfl_down_sync(hmask, d, o, 16);
                        if (sublane == 0 && d > 0) s[kb] += (float)d * qsc;
                    }
                }
            } else {
                for (int h = 0; h < nh; ++h) {
                    int2 qv = s_q_i8[(r * nh + h) * 16 + sublane];
                    float qsc = s_q_sc[r * nh + h];
#pragma unroll
                    for (int kb = 0; kb < KB; ++kb) {
                        int d = __dp4a(qv.x, ki[kb].x, 0);
                        d = __dp4a(qv.y, ki[kb].y, d);
#pragma unroll
                        for (int o = 8; o > 0; o >>= 1) d += __shfl_down_sync(hmask, d, o, 16);
                        if (sublane == 0 && d > 0) s[kb] += (float)d * qsc;
                    }
                }
            }
            if (sublane == 0) {
                float* dst = scores + (size_t)(m0 + r) * nb_stride + b0;
                if (aligned_out && b0 + 4 <= nb_r) {
                    *reinterpret_cast<float4*>(dst) = make_float4(
                        s[0] * ksc[0], s[1] * ksc[1], s[2] * ksc[2], s[3] * ksc[3]);
                } else {
#pragma unroll
                    for (int kb = 0; kb < KB; ++kb) {
                        if (b0 + kb < nb_r) dst[kb] = s[kb] * ksc[kb];
                    }
                }
            }
        }
        b0 = b0_n;
#pragma unroll
        for (int kb = 0; kb < KB; ++kb) { ki[kb] = ki_n[kb]; ksc[kb] = ksc_n[kb]; }
    }
    NVK_PDL_EPILOG();
}

}

extern "C" int nv_kernels_q4exp_quant_rows_to_i8(
    void* stream, const uint16_t* src_bf16, uint8_t* dst_i8, float* dst_sc,
    const int32_t* pos_dev, int row_offset, int n_rows, int rows_per_tok, int row_elems
) {
    if (n_rows <= 0) return 0;
    if (row_elems <= 0 || row_elems > 256 || (row_elems & 31) != 0) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)n_rows), dim3((unsigned)row_elems), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_quant_rows_to_i8_kernel,
            src_bf16, dst_i8, dst_sc, pos_dev, row_offset, rows_per_tok, row_elems);
    } else {
        q4exp_quant_rows_to_i8_kernel<<<n_rows, row_elems, 0, s>>>(
            src_bf16, dst_i8, dst_sc, pos_dev, row_offset, rows_per_tok, row_elems);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_qsa_scores_i8_hd128(
    void* stream, const uint16_t* q, const uint8_t* cmp_k_i8, const float* cmp_k_sc,
    const int32_t* nb_per_row, int m, int nh, int nb_stride, float scale, float* scores
) {
    if (m <= 0 || nb_stride <= 0) return 0;
    if (nh <= 0 || nh > 4) return -2;
    constexpr int R = QSA_SCORES_SMALL_M_MAX;
    constexpr int KB = 4;
    constexpr int BPC = QSA_SCORES_HALF_WARPS * KB;
    int gy = (m + R - 1) / R;
    if (gy > NVK_MAX_GRID_Y) return NVK_ERR_GRID_AXIS;
    int gx = (nb_stride + BPC - 1) / BPC;
    if (gx > 376) gx = 376;
    dim3 grid(gx, gy);
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(QSA_SCORES_THREADS), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_qsa_scores_i8_hd128_kernel<R, KB>,
            q, cmp_k_i8, cmp_k_sc, nb_per_row, m, nh, nb_stride, scale, scores);
    } else {
        q4exp_qsa_scores_i8_hd128_kernel<R, KB><<<grid, QSA_SCORES_THREADS, 0, s>>>(
            q, cmp_k_i8, cmp_k_sc, nb_per_row, m, nh, nb_stride, scale, scores);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_lm_head_fp8_argmax_mt(
    void* stream,
    const uint8_t* wq,
    const float* rs,
    const uint16_t* x,
    uint64_t* packed_best,
    uint32_t* out_ids,
    int m,
    int n,
    int k
) {
    if (m <= 0 || n <= 0 || k <= 0 || (k & 15) != 0) return -1;
    if (wq == nullptr || rs == nullptr || x == nullptr || packed_best == nullptr || out_ids == nullptr) return -2;
    if ((size_t)m * (size_t)(k >> 4) * 9 * sizeof(float2) > (size_t)nvk_max_dynamic_smem_optin()) return -3;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const auto* xb = reinterpret_cast<const __nv_bfloat16*>(x);
    switch (m) {
        case 1: return q4exp_launch_lm_head_fp8_argmax_mt<1>(s, wq, rs, xb, packed_best, out_ids, n, k);
        case 2: return q4exp_launch_lm_head_fp8_argmax_mt<2>(s, wq, rs, xb, packed_best, out_ids, n, k);
        case 3: return q4exp_launch_lm_head_fp8_argmax_mt<3>(s, wq, rs, xb, packed_best, out_ids, n, k);
        case 4: return q4exp_launch_lm_head_fp8_argmax_mt<4>(s, wq, rs, xb, packed_best, out_ids, n, k);
        case 5: return q4exp_launch_lm_head_fp8_argmax_mt<5>(s, wq, rs, xb, packed_best, out_ids, n, k);
        case 6: return q4exp_launch_lm_head_fp8_argmax_mt<6>(s, wq, rs, xb, packed_best, out_ids, n, k);
        case 7: return q4exp_launch_lm_head_fp8_argmax_mt<7>(s, wq, rs, xb, packed_best, out_ids, n, k);
        case 8: return q4exp_launch_lm_head_fp8_argmax_mt<8>(s, wq, rs, xb, packed_best, out_ids, n, k);
        default: return -1;
    }
}

template <int TM>
__global__ void __launch_bounds__(256) q4exp_lm_head_nvfp4_argmax_mt_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ scales,
    const uint16_t* __restrict__ x,
    unsigned long long* __restrict__ packed_best,
    int N,
    int K
) {
    extern __shared__ int xs_q8_words[];
    __shared__ float s_amax[TM][8];
    __shared__ unsigned long long warp_best[TM][8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int kb_total = K >> 4;
    int row_words = K >> 2;
    int num_tiles = (N + 15) >> 4;

    NVK_PDL_PROLOG();

    int n8 = K >> 3;
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)m * K);
        float local_max = 0.f;
        for (int i = tid; i < n8; i += 256) {
            float f[8];
            unpack8_bf16(__ldg(&x4[i]), f);
            #pragma unroll
            for (int j = 0; j < 8; ++j) local_max = fmaxf(local_max, fabsf(f[j]));
        }
        local_max = warp_max(local_max);
        if (lane == 0) s_amax[m][warp] = local_max;
    }
    __syncthreads();
    if (warp < TM) {
        float mx = (lane < 8) ? s_amax[warp][lane] : 0.f;
        mx = warp_max(mx);
        if (lane == 0) s_amax[warp][0] = (mx > 0.f) ? (127.f / mx) : 0.f;
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)m * K);
        float inv = s_amax[m][0];
        int* dst = xs_q8_words + m * row_words;
        for (int i = tid; i < n8; i += 256) {
            float f[8];
            unpack8_bf16(__ldg(&x4[i]), f);
            int q[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int qi = __float2int_rn(fminf(fmaxf(f[j] * inv, -127.f), 127.f));
                q[j] = qi & 0xFF;
            }
            dst[2 * i] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
            dst[2 * i + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
        }
    }
    __syncthreads();

    unsigned long long my_best[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) my_best[m] = 0ull;

    const int4* xs_i4 = reinterpret_cast<const int4*>(xs_q8_words);
    for (int tile = blockIdx.x; tile < num_tiles; tile += gridDim.x) {
        int n0 = tile * 16 + warp;
        int n1 = n0 + 8;
        if (n0 >= N) continue;
        bool has1 = n1 < N;
        const uint2* wa = reinterpret_cast<const uint2*>(wq + (size_t)n0 * (K >> 1));
        const uint2* wb = has1 ? reinterpret_cast<const uint2*>(wq + (size_t)n1 * (K >> 1)) : wa;
        const uint8_t* sa_ptr = scales + (size_t)n0 * kb_total;
        const uint8_t* sb_ptr = has1 ? (scales + (size_t)n1 * kb_total) : sa_ptr;
        int kb = lane;
        uint2 ra = (kb < kb_total) ? __ldcs(&wa[kb]) : make_uint2(0u, 0u);
        uint2 rb = (has1 && kb < kb_total) ? __ldcs(&wb[kb]) : make_uint2(0u, 0u);
        uint8_t sca = (kb < kb_total) ? __ldcs(&sa_ptr[kb]) : 0u;
        uint8_t scb = (has1 && kb < kb_total) ? __ldcs(&sb_ptr[kb]) : 0u;
        float acc0[TM], acc1[TM];
        #pragma unroll
        for (int m = 0; m < TM; ++m) { acc0[m] = 0.f; acc1[m] = 0.f; }
        while (kb < kb_total) {
            int kb_next = kb + 32;
            uint2 na = (kb_next < kb_total) ? __ldcs(&wa[kb_next]) : make_uint2(0u, 0u);
            uint2 nb = (has1 && kb_next < kb_total) ? __ldcs(&wb[kb_next]) : make_uint2(0u, 0u);
            uint8_t nsa = (kb_next < kb_total) ? __ldcs(&sa_ptr[kb_next]) : 0u;
            uint8_t nsb = (has1 && kb_next < kb_total) ? __ldcs(&sb_ptr[kb_next]) : 0u;
            unsigned a0, a1, a2, a3;
            q4exp_nibbles8_to_i8(ra.x, a0, a1);
            q4exp_nibbles8_to_i8(ra.y, a2, a3);
            float fa = q4exp_decode_ue4m3(sca);
            unsigned b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
            float fb = 0.f;
            if (has1) {
                q4exp_nibbles8_to_i8(rb.x, b0, b1);
                q4exp_nibbles8_to_i8(rb.y, b2, b3);
                fb = q4exp_decode_ue4m3(scb);
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                int4 xv = xs_i4[m * kb_total + kb];
                int d0 = __dp4a((int)a0, xv.x, 0);
                d0 = __dp4a((int)a1, xv.y, d0);
                d0 = __dp4a((int)a2, xv.z, d0);
                d0 = __dp4a((int)a3, xv.w, d0);
                acc0[m] += fa * (float)d0;
                if (has1) {
                    int d1 = __dp4a((int)b0, xv.x, 0);
                    d1 = __dp4a((int)b1, xv.y, d1);
                    d1 = __dp4a((int)b2, xv.z, d1);
                    d1 = __dp4a((int)b3, xv.w, d1);
                    acc1[m] += fb * (float)d1;
                }
            }
            kb = kb_next;
            ra = na; rb = nb; sca = nsa; scb = nsb;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            acc0[m] = warp_sum(acc0[m]);
            if (has1) acc1[m] = warp_sum(acc1[m]);
            if (lane == 0) {
                unsigned long long p0 = q4exp_pack_argmax_key(acc0[m], (unsigned)n0);
                if (p0 > my_best[m]) my_best[m] = p0;
                if (has1) {
                    unsigned long long p1 = q4exp_pack_argmax_key(acc1[m], (unsigned)n1);
                    if (p1 > my_best[m]) my_best[m] = p1;
                }
            }
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int m = 0; m < TM; ++m) warp_best[m][warp] = my_best[m];
    }
    __syncthreads();
    if (tid < TM) {
        unsigned long long blk = 0ull;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (warp_best[tid][w] > blk) blk = warp_best[tid][w];
        }
        if (blk > packed_best[tid]) atomicMax(&packed_best[tid], blk);
    }
    NVK_PDL_EPILOG();
}

template <int TM>
static int q4exp_launch_lm_head_nvfp4_argmax_mt(
    cudaStream_t s, const uint8_t* wq, const uint8_t* scales, const uint16_t* x,
    uint64_t* packed_best, uint32_t* out_ids, int N, int K
) {
    size_t smem = (size_t)TM * (size_t)K;
    if (smem > 48 * 1024) return -3;
    cudaError_t me = cudaMemsetAsync(packed_best, 0, (size_t)TM * sizeof(uint64_t), s);
    if (me != cudaSuccess) return (int)me;
    int num_tiles = (N + 15) / 16;
    dim3 grid((unsigned)(num_tiles < 376 ? num_tiles : 376));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_lm_head_nvfp4_argmax_mt_kernel<TM>,
            wq, scales, x, reinterpret_cast<unsigned long long*>(packed_best), N, K);
        NVK_PDL_ATTR(cfg2, dim3(1), dim3((unsigned)TM), 0, s);
        cudaLaunchKernelEx(&cfg2, q4exp_unpack_packed_argmax_mt_kernel,
            reinterpret_cast<const unsigned long long*>(packed_best), out_ids, TM);
    } else {
        q4exp_lm_head_nvfp4_argmax_mt_kernel<TM><<<grid, 256, smem, s>>>(
            wq, scales, x, reinterpret_cast<unsigned long long*>(packed_best), N, K);
        q4exp_unpack_packed_argmax_mt_kernel<<<1, TM, 0, s>>>(
            reinterpret_cast<const unsigned long long*>(packed_best), out_ids, TM);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_lm_head_nvfp4_argmax_mt(
    void* stream,
    const uint8_t* wq,
    const uint8_t* scales,
    const uint16_t* x,
    uint64_t* packed_best,
    uint32_t* out_ids,
    int m,
    int n,
    int k
) {
    if (m <= 0 || n <= 0 || k <= 0 || (k & 15) != 0) return -1;
    if (wq == nullptr || scales == nullptr || x == nullptr || packed_best == nullptr || out_ids == nullptr) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: return q4exp_launch_lm_head_nvfp4_argmax_mt<1>(s, wq, scales, x, packed_best, out_ids, n, k);
        case 2: return q4exp_launch_lm_head_nvfp4_argmax_mt<2>(s, wq, scales, x, packed_best, out_ids, n, k);
        case 3: return q4exp_launch_lm_head_nvfp4_argmax_mt<3>(s, wq, scales, x, packed_best, out_ids, n, k);
        case 4: return q4exp_launch_lm_head_nvfp4_argmax_mt<4>(s, wq, scales, x, packed_best, out_ids, n, k);
        case 5: return q4exp_launch_lm_head_nvfp4_argmax_mt<5>(s, wq, scales, x, packed_best, out_ids, n, k);
        case 6: return q4exp_launch_lm_head_nvfp4_argmax_mt<6>(s, wq, scales, x, packed_best, out_ids, n, k);
        case 7: return q4exp_launch_lm_head_nvfp4_argmax_mt<7>(s, wq, scales, x, packed_best, out_ids, n, k);
        case 8: return q4exp_launch_lm_head_nvfp4_argmax_mt<8>(s, wq, scales, x, packed_best, out_ids, n, k);
        default: return -1;
    }
}

__device__ __forceinline__ uint8_t q4exp_encode_e2m1(float x) {
    uint8_t sign = signbit(x) ? 0x08u : 0u;
    float a = fabsf(x);
    uint8_t c = (uint8_t)(a > 0.25f) + (uint8_t)(a > 0.75f) + (uint8_t)(a > 1.25f)
              + (uint8_t)(a > 1.75f) + (uint8_t)(a > 2.5f)  + (uint8_t)(a > 3.5f)
              + (uint8_t)(a > 5.0f);
    return sign | c;
}

__device__ __forceinline__ uint8_t q4exp_encode_ue4m3(float scale) {
    if (!isfinite(scale) || scale <= 0.f) return 0u;
    float clamped = fminf(scale, 448.f);
    uint8_t b = (uint8_t)__nv_cvt_float_to_fp8(clamped, __NV_SATFINITE, __NV_E4M3);
    return (b == 0x7Fu) ? 0x7Eu : (b & 0x7Fu);
}

__global__ void q4exp_requant_fp8_to_nvfp4_kernel(
    const uint8_t* __restrict__ fp8_w,
    const float* __restrict__ fp8_rs,
    uint8_t* __restrict__ wq_out,
    uint8_t* __restrict__ sc_out,
    int N,
    int K,
    float stored_global
) {
    int kb_total = K >> 4;
    size_t total = (size_t)N * (size_t)kb_total;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = (size_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += step) {
        int n = (int)(idx / (size_t)kb_total);
        float rs = __ldg(&fp8_rs[n]);
        int4 raw = __ldg(reinterpret_cast<const int4*>(fp8_w) + idx);
        float v[16];
        q4exp_deq_e4m3x16(raw, v);
        float amax = 0.f;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            v[j] *= rs;
            amax = fmaxf(amax, fabsf(v[j]));
        }
        uint8_t sb = (amax > 0.f) ? q4exp_encode_ue4m3(stored_global * (amax * (1.f / 6.f))) : 0u;
        float sdec = q4exp_decode_ue4m3(sb);
        float inv = (sdec > 0.f) ? (stored_global / sdec) : 0.f;
        uint8_t packed[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            uint8_t lo = q4exp_encode_e2m1(fminf(fmaxf(v[2 * j] * inv, -6.f), 6.f));
            uint8_t hi = q4exp_encode_e2m1(fminf(fmaxf(v[2 * j + 1] * inv, -6.f), 6.f));
            packed[j] = (uint8_t)((hi << 4) | (lo & 0x0Fu));
        }
        *reinterpret_cast<uint2*>(wq_out + idx * 8) = *reinterpret_cast<const uint2*>(packed);
        sc_out[idx] = sb;
    }
}

template <int TM>
__device__ __forceinline__ void q4exp_stage_x_i8_smem(
    const uint16_t* __restrict__ x,
    int* xs_q8_words,
    float* x_scales,
    float s_amax[TM][8],
    int K
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int n8 = K >> 3;
    int row_words = K >> 2;
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)m * K);
        float local_max = 0.f;
        for (int i = tid; i < n8; i += 256) {
            float f[8];
            unpack8_bf16(__ldg(&x4[i]), f);
            #pragma unroll
            for (int j = 0; j < 8; ++j) local_max = fmaxf(local_max, fabsf(f[j]));
        }
        local_max = warp_max(local_max);
        if (lane == 0) s_amax[m][warp] = local_max;
    }
    __syncthreads();
    if (warp < TM) {
        float mx = (lane < 8) ? s_amax[warp][lane] : 0.f;
        mx = warp_max(mx);
        if (lane == 0) {
            s_amax[warp][0] = (mx > 0.f) ? (127.f / mx) : 0.f;
            x_scales[warp] = mx * (1.f / 127.f);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)m * K);
        float inv = s_amax[m][0];
        int* dst = xs_q8_words + m * row_words;
        for (int i = tid; i < n8; i += 256) {
            float f[8];
            unpack8_bf16(__ldg(&x4[i]), f);
            int q[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int qi = __float2int_rn(fminf(fmaxf(f[j] * inv, -127.f), 127.f));
                q[j] = qi & 0xFF;
            }
            dst[2 * i] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
            dst[2 * i + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
        }
    }
    __syncthreads();
}

static int* g_q4exp_i8_buf = nullptr;
static float* g_q4exp_i8_sc = nullptr;
static constexpr int kQ4ExpI8MaxM = 8;
static constexpr int kQ4ExpI8MaxK = 16384;

static void q4exp_ensure_i8_scratch() {
    if (g_q4exp_i8_buf != nullptr) return;
    int* b = nullptr;
    float* sc = nullptr;
    int* hq = nullptr;
    float* hs = nullptr;
    if (cudaMalloc(&b, (size_t)kQ4ExpI8MaxM * (size_t)(kQ4ExpI8MaxK >> 2) * sizeof(int)) == cudaSuccess
        && cudaMalloc(&sc, (size_t)kQ4ExpI8MaxM * sizeof(float)) == cudaSuccess
        && cudaMalloc(&hq, (size_t)kQ4ExpI8MaxM * 2560 * sizeof(int)) == cudaSuccess
        && cudaMalloc(&hs, (size_t)kQ4ExpI8MaxM * 4 * sizeof(float)) == cudaSuccess) {
        g_q4exp_i8_buf = b;
        g_q4exp_i8_sc = sc;
        g_q4exp_hc_xn_q8 = hq;
        g_q4exp_hc_xn_gsc = hs;
    } else {
        if (b) cudaFree(b);
        if (sc) cudaFree(sc);
        if (hq) cudaFree(hq);
        if (hs) cudaFree(hs);
    }
}

__global__ void __launch_bounds__(256) q4exp_rowquant_i8_kernel(
    const uint16_t* __restrict__ x,
    int* __restrict__ x_q8,
    float* __restrict__ x_scales,
    int K
) {
    __shared__ float s_amax[8];
    __shared__ float s_inv;
    int m = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int n8 = K >> 3;
    int row_words = K >> 2;
    NVK_PDL_PROLOG();
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)m * K);
    int* dst = x_q8 + m * row_words;
    if (n8 <= 768) {
        float rf[3][8];
        float mx = 0.f;
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            int i = tid + it * 256;
            if (i < n8) {
                unpack8_bf16(__ldg(&x4[i]), rf[it]);
                #pragma unroll
                for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fabsf(rf[it][j]));
            }
        }
        mx = warp_max(mx);
        if (lane == 0) s_amax[warp] = mx;
        __syncthreads();
        if (warp == 0) {
            float t = (lane < 8) ? s_amax[lane] : 0.f;
            t = warp_max(t);
            if (lane == 0) {
                s_inv = (t > 0.f) ? (127.f / t) : 0.f;
                x_scales[m] = t * (1.f / 127.f);
            }
        }
        __syncthreads();
        float inv = s_inv;
        #pragma unroll
        for (int it = 0; it < 3; ++it) {
            int i = tid + it * 256;
            if (i < n8) {
                int q[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    q[j] = __float2int_rn(fminf(fmaxf(rf[it][j] * inv, -127.f), 127.f)) & 0xFF;
                }
                int w0 = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
                int w1 = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
                *reinterpret_cast<int2*>(dst + 2 * i) = make_int2(w0, w1);
            }
        }
        NVK_PDL_EPILOG();
        return;
    }
    float mx = 0.f;
    for (int i = tid; i < n8; i += 256) {
        float f[8];
        unpack8_bf16(__ldg(&x4[i]), f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fabsf(f[j]));
    }
    mx = warp_max(mx);
    if (lane == 0) s_amax[warp] = mx;
    __syncthreads();
    if (warp == 0) {
        float t = (lane < 8) ? s_amax[lane] : 0.f;
        t = warp_max(t);
        if (lane == 0) {
            s_inv = (t > 0.f) ? (127.f / t) : 0.f;
            x_scales[m] = t * (1.f / 127.f);
        }
    }
    __syncthreads();
    float inv = s_inv;
    for (int i = tid; i < n8; i += 256) {
        float f[8];
        unpack8_bf16(__ldg(&x4[i]), f);
        int q[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            q[j] = __float2int_rn(fminf(fmaxf(f[j] * inv, -127.f), 127.f)) & 0xFF;
        }
        int w0 = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
        int w1 = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
        *reinterpret_cast<int2*>(dst + 2 * i) = make_int2(w0, w1);
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(128) q4exp_nvfp4_gemv_i8_mt_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ scales,
    const int* __restrict__ x_q8,
    const float* __restrict__ x_scales,
    __nv_bfloat16* __restrict__ y,
    float alpha_over_2g,
    int N,
    int K
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int n0 = blockIdx.x * 4 + warp;
    if (n0 >= N) return;
    int kb_total = K >> 4;
    const int4* x4_base = reinterpret_cast<const int4*>(x_q8);
    if ((K & 31) == 0) {
        int kp_total = K >> 5;
        const uint4* wa = reinterpret_cast<const uint4*>(wq + (size_t)n0 * (K >> 1));
        const uint16_t* sa = reinterpret_cast<const uint16_t*>(scales + (size_t)n0 * kb_total);
        uint4 ra = (lane < kp_total) ? __ldcs(&wa[lane]) : make_uint4(0u, 0u, 0u, 0u);
        uint16_t sp = (lane < kp_total) ? __ldcs(&sa[lane]) : 0u;
        NVK_PDL_PROLOG();
        float acc0[TM];
        #pragma unroll
        for (int m = 0; m < TM; ++m) acc0[m] = 0.f;
        int kp = lane;
        while (kp < kp_total) {
            int kpn = kp + 32;
            uint4 na = (kpn < kp_total) ? __ldcs(&wa[kpn]) : make_uint4(0u, 0u, 0u, 0u);
            uint16_t nsp = (kpn < kp_total) ? __ldcs(&sa[kpn]) : 0u;
            unsigned a0, a1, a2, a3, b0, b1, b2, b3;
            q4exp_nibbles8_to_i8(ra.x, a0, a1);
            q4exp_nibbles8_to_i8(ra.y, a2, a3);
            q4exp_nibbles8_to_i8(ra.z, b0, b1);
            q4exp_nibbles8_to_i8(ra.w, b2, b3);
            float fa = q4exp_decode_ue4m3((uint8_t)(sp & 0xFFu));
            float fb = q4exp_decode_ue4m3((uint8_t)(sp >> 8));
            int kb0 = kp * 2;
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                int4 xv0 = __ldg(&x4_base[m * kb_total + kb0]);
                int4 xv1 = __ldg(&x4_base[m * kb_total + kb0 + 1]);
                int d0 = __dp4a((int)a0, xv0.x, 0);
                d0 = __dp4a((int)a1, xv0.y, d0);
                d0 = __dp4a((int)a2, xv0.z, d0);
                d0 = __dp4a((int)a3, xv0.w, d0);
                int d1 = __dp4a((int)b0, xv1.x, 0);
                d1 = __dp4a((int)b1, xv1.y, d1);
                d1 = __dp4a((int)b2, xv1.z, d1);
                d1 = __dp4a((int)b3, xv1.w, d1);
                acc0[m] += fa * (float)d0 + fb * (float)d1;
            }
            kp = kpn;
            ra = na; sp = nsp;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            acc0[m] = warp_sum(acc0[m]);
            if (lane == 0) {
                float sc = __ldg(&x_scales[m]) * alpha_over_2g;
                y[(size_t)m * N + n0] = __float2bfloat16(acc0[m] * sc);
            }
        }
        NVK_PDL_EPILOG();
        return;
    }
    const uint2* wa = reinterpret_cast<const uint2*>(wq + (size_t)n0 * (K >> 1));
    const uint8_t* sa = scales + (size_t)n0 * kb_total;
    uint2 ra = (lane < kb_total) ? __ldcs(&wa[lane]) : make_uint2(0u, 0u);
    uint8_t sca = (lane < kb_total) ? __ldcs(&sa[lane]) : 0u;
    NVK_PDL_PROLOG();
    float acc0[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) acc0[m] = 0.f;
    int kb = lane;
    while (kb < kb_total) {
        int kbn = kb + 32;
        uint2 na = (kbn < kb_total) ? __ldcs(&wa[kbn]) : make_uint2(0u, 0u);
        uint8_t nsa = (kbn < kb_total) ? __ldcs(&sa[kbn]) : 0u;
        unsigned a0, a1, a2, a3;
        q4exp_nibbles8_to_i8(ra.x, a0, a1);
        q4exp_nibbles8_to_i8(ra.y, a2, a3);
        float fa = q4exp_decode_ue4m3(sca);
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            int4 xv = __ldg(&x4_base[m * kb_total + kb]);
            int d0 = __dp4a((int)a0, xv.x, 0);
            d0 = __dp4a((int)a1, xv.y, d0);
            d0 = __dp4a((int)a2, xv.z, d0);
            d0 = __dp4a((int)a3, xv.w, d0);
            acc0[m] += fa * (float)d0;
        }
        kb = kbn;
        ra = na; sca = nsa;
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        acc0[m] = warp_sum(acc0[m]);
        if (lane == 0) {
            float sc = __ldg(&x_scales[m]) * alpha_over_2g;
            y[(size_t)m * N + n0] = __float2bfloat16(acc0[m] * sc);
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(128) q4exp_nvfp4_dual_i8_mt_kernel(
    const uint8_t* __restrict__ wq_a, const uint8_t* __restrict__ sc_a, float alpha_a,
    const uint8_t* __restrict__ wq_b, const uint8_t* __restrict__ sc_b, float alpha_b,
    const int* __restrict__ x_q8, const float* __restrict__ x_scales,
    __nv_bfloat16* __restrict__ y_a, __nv_bfloat16* __restrict__ y_b,
    int Na, int Nb, int K
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int kb_total = K >> 4;
    int total = Na + Nb;
    int n0 = blockIdx.x * 4 + warp;
    if (n0 >= total) return;
    const uint8_t* w_row;
    const uint8_t* sa_ptr;
    float al0;
    __nv_bfloat16* y0_b;
    int y0_c, y0_s;
    if (n0 < Na) {
        w_row = wq_a + (size_t)n0 * (K >> 1);
        sa_ptr = sc_a + (size_t)n0 * kb_total; al0 = alpha_a; y0_b = y_a; y0_c = n0; y0_s = Na;
    } else {
        int r = n0 - Na;
        w_row = wq_b + (size_t)r * (K >> 1);
        sa_ptr = sc_b + (size_t)r * kb_total; al0 = alpha_b; y0_b = y_b; y0_c = r; y0_s = Nb;
    }
    const int4* x4_base = reinterpret_cast<const int4*>(x_q8);
    if ((K & 1023) == 0) {
        int kp_total = K >> 5;
        const uint4* wa = reinterpret_cast<const uint4*>(w_row);
        const uint16_t* sa = reinterpret_cast<const uint16_t*>(sa_ptr);
        uint4 ra = (lane < kp_total) ? __ldcs(&wa[lane]) : make_uint4(0u, 0u, 0u, 0u);
        uint16_t sp = (lane < kp_total) ? __ldcs(&sa[lane]) : 0u;
        NVK_PDL_PROLOG();
        float acc0[TM];
        #pragma unroll
        for (int m = 0; m < TM; ++m) acc0[m] = 0.f;
        int kp = lane;
        while (kp < kp_total) {
            int kpn = kp + 32;
            uint4 na = (kpn < kp_total) ? __ldcs(&wa[kpn]) : make_uint4(0u, 0u, 0u, 0u);
            uint16_t nsp = (kpn < kp_total) ? __ldcs(&sa[kpn]) : 0u;
            unsigned a0, a1, a2, a3, b0, b1, b2, b3;
            q4exp_nibbles8_to_i8(ra.x, a0, a1);
            q4exp_nibbles8_to_i8(ra.y, a2, a3);
            q4exp_nibbles8_to_i8(ra.z, b0, b1);
            q4exp_nibbles8_to_i8(ra.w, b2, b3);
            float fa = q4exp_decode_ue4m3((uint8_t)(sp & 0xFFu));
            float fb = q4exp_decode_ue4m3((uint8_t)(sp >> 8));
            int kb0 = kp * 2;
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                int4 xv0 = __ldg(&x4_base[m * kb_total + kb0]);
                int4 xv1 = __ldg(&x4_base[m * kb_total + kb0 + 1]);
                int d0 = __dp4a((int)a0, xv0.x, 0);
                d0 = __dp4a((int)a1, xv0.y, d0);
                d0 = __dp4a((int)a2, xv0.z, d0);
                d0 = __dp4a((int)a3, xv0.w, d0);
                int d1 = __dp4a((int)b0, xv1.x, 0);
                d1 = __dp4a((int)b1, xv1.y, d1);
                d1 = __dp4a((int)b2, xv1.z, d1);
                d1 = __dp4a((int)b3, xv1.w, d1);
                acc0[m] += fa * (float)d0 + fb * (float)d1;
            }
            kp = kpn;
            ra = na; sp = nsp;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            acc0[m] = warp_sum(acc0[m]);
            if (lane == 0) {
                float xs = __ldg(&x_scales[m]);
                y0_b[(size_t)m * y0_s + y0_c] = __float2bfloat16(acc0[m] * (xs * al0));
            }
        }
        NVK_PDL_EPILOG();
        return;
    }
    const uint2* wa = reinterpret_cast<const uint2*>(w_row);
    uint2 ra = (lane < kb_total) ? __ldcs(&wa[lane]) : make_uint2(0u, 0u);
    uint8_t sca = (lane < kb_total) ? __ldcs(&sa_ptr[lane]) : 0u;
    NVK_PDL_PROLOG();
    float acc0[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) acc0[m] = 0.f;
    int kb = lane;
    while (kb < kb_total) {
        int kbn = kb + 32;
        uint2 na = (kbn < kb_total) ? __ldcs(&wa[kbn]) : make_uint2(0u, 0u);
        uint8_t nsa = (kbn < kb_total) ? __ldcs(&sa_ptr[kbn]) : 0u;
        unsigned a0, a1, a2, a3;
        q4exp_nibbles8_to_i8(ra.x, a0, a1);
        q4exp_nibbles8_to_i8(ra.y, a2, a3);
        float fa = q4exp_decode_ue4m3(sca);
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            int4 xv = __ldg(&x4_base[m * kb_total + kb]);
            int d0 = __dp4a((int)a0, xv.x, 0);
            d0 = __dp4a((int)a1, xv.y, d0);
            d0 = __dp4a((int)a2, xv.z, d0);
            d0 = __dp4a((int)a3, xv.w, d0);
            acc0[m] += fa * (float)d0;
        }
        kb = kbn;
        ra = na; sca = nsa;
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        acc0[m] = warp_sum(acc0[m]);
        if (lane == 0) {
            float xs = __ldg(&x_scales[m]);
            y0_b[(size_t)m * y0_s + y0_c] = __float2bfloat16(acc0[m] * (xs * al0));
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(128) q4exp_nvfp4_qkv_idx_i8_mt_kernel(
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ scq, float aq,
    const uint8_t* __restrict__ wk, const uint8_t* __restrict__ sck, float ak,
    const uint8_t* __restrict__ wv, const uint8_t* __restrict__ scv, float av,
    const uint8_t* __restrict__ wi, const uint8_t* __restrict__ sci, float ai,
    const int* __restrict__ x_q8, const float* __restrict__ x_scales,
    __nv_bfloat16* __restrict__ yq, __nv_bfloat16* __restrict__ yk,
    __nv_bfloat16* __restrict__ yv, __nv_bfloat16* __restrict__ yi,
    int nq, int nk, int nv, int ni, int K
) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int kb_total = K >> 4;
    int total = nq + nk + nv + ni;
    int n0 = blockIdx.x * 4 + warp;
    if (n0 >= total) return;
    const uint8_t* w_row;
    const uint8_t* sa_ptr;
    float al0;
    __nv_bfloat16* y0_b;
    int y0_c, y0_s;
    if (n0 < nq) {
        w_row = wq + (size_t)n0 * (K >> 1);
        sa_ptr = scq + (size_t)n0 * kb_total; al0 = aq; y0_b = yq; y0_c = n0; y0_s = nq;
    } else if (n0 < nq + nk) {
        int r = n0 - nq;
        w_row = wk + (size_t)r * (K >> 1);
        sa_ptr = sck + (size_t)r * kb_total; al0 = ak; y0_b = yk; y0_c = r; y0_s = nk;
    } else if (n0 < nq + nk + nv) {
        int r = n0 - nq - nk;
        w_row = wv + (size_t)r * (K >> 1);
        sa_ptr = scv + (size_t)r * kb_total; al0 = av; y0_b = yv; y0_c = r; y0_s = nv;
    } else {
        int r = n0 - nq - nk - nv;
        w_row = wi + (size_t)r * (K >> 1);
        sa_ptr = sci + (size_t)r * kb_total; al0 = ai; y0_b = yi; y0_c = r; y0_s = ni;
    }
    const int4* x4_base = reinterpret_cast<const int4*>(x_q8);
    if ((K & 1023) == 0) {
        int kp_total = K >> 5;
        const uint4* wa = reinterpret_cast<const uint4*>(w_row);
        const uint16_t* sa = reinterpret_cast<const uint16_t*>(sa_ptr);
        uint4 ra = (lane < kp_total) ? __ldcs(&wa[lane]) : make_uint4(0u, 0u, 0u, 0u);
        uint16_t sp = (lane < kp_total) ? __ldcs(&sa[lane]) : 0u;
        NVK_PDL_PROLOG();
        float acc0[TM];
        #pragma unroll
        for (int m = 0; m < TM; ++m) acc0[m] = 0.f;
        int kp = lane;
        while (kp < kp_total) {
            int kpn = kp + 32;
            uint4 na = (kpn < kp_total) ? __ldcs(&wa[kpn]) : make_uint4(0u, 0u, 0u, 0u);
            uint16_t nsp = (kpn < kp_total) ? __ldcs(&sa[kpn]) : 0u;
            unsigned a0, a1, a2, a3, b0, b1, b2, b3;
            q4exp_nibbles8_to_i8(ra.x, a0, a1);
            q4exp_nibbles8_to_i8(ra.y, a2, a3);
            q4exp_nibbles8_to_i8(ra.z, b0, b1);
            q4exp_nibbles8_to_i8(ra.w, b2, b3);
            float fa = q4exp_decode_ue4m3((uint8_t)(sp & 0xFFu));
            float fb = q4exp_decode_ue4m3((uint8_t)(sp >> 8));
            int kb0 = kp * 2;
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                int4 xv0 = __ldg(&x4_base[m * kb_total + kb0]);
                int4 xv1 = __ldg(&x4_base[m * kb_total + kb0 + 1]);
                int d0 = __dp4a((int)a0, xv0.x, 0);
                d0 = __dp4a((int)a1, xv0.y, d0);
                d0 = __dp4a((int)a2, xv0.z, d0);
                d0 = __dp4a((int)a3, xv0.w, d0);
                int d1 = __dp4a((int)b0, xv1.x, 0);
                d1 = __dp4a((int)b1, xv1.y, d1);
                d1 = __dp4a((int)b2, xv1.z, d1);
                d1 = __dp4a((int)b3, xv1.w, d1);
                acc0[m] += fa * (float)d0 + fb * (float)d1;
            }
            kp = kpn;
            ra = na; sp = nsp;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            acc0[m] = warp_sum(acc0[m]);
            if (lane == 0) {
                float xs = __ldg(&x_scales[m]);
                y0_b[(size_t)m * y0_s + y0_c] = __float2bfloat16(acc0[m] * (xs * al0));
            }
        }
        NVK_PDL_EPILOG();
        return;
    }
    const uint2* wa = reinterpret_cast<const uint2*>(w_row);
    uint2 ra = (lane < kb_total) ? __ldcs(&wa[lane]) : make_uint2(0u, 0u);
    uint8_t sca = (lane < kb_total) ? __ldcs(&sa_ptr[lane]) : 0u;
    NVK_PDL_PROLOG();
    float acc0[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) acc0[m] = 0.f;
    int kb = lane;
    while (kb < kb_total) {
        int kbn = kb + 32;
        uint2 na = (kbn < kb_total) ? __ldcs(&wa[kbn]) : make_uint2(0u, 0u);
        uint8_t nsa = (kbn < kb_total) ? __ldcs(&sa_ptr[kbn]) : 0u;
        unsigned a0, a1, a2, a3;
        q4exp_nibbles8_to_i8(ra.x, a0, a1);
        q4exp_nibbles8_to_i8(ra.y, a2, a3);
        float fa = q4exp_decode_ue4m3(sca);
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            int4 xv = __ldg(&x4_base[m * kb_total + kb]);
            int d0 = __dp4a((int)a0, xv.x, 0);
            d0 = __dp4a((int)a1, xv.y, d0);
            d0 = __dp4a((int)a2, xv.z, d0);
            d0 = __dp4a((int)a3, xv.w, d0);
            acc0[m] += fa * (float)d0;
        }
        kb = kbn;
        ra = na; sca = nsa;
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        acc0[m] = warp_sum(acc0[m]);
        if (lane == 0) {
            float xs = __ldg(&x_scales[m]);
            y0_b[(size_t)m * y0_s + y0_c] = __float2bfloat16(acc0[m] * (xs * al0));
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(128) q4exp_nvfp4_dual_i8_mt_hw16_kernel(
    const uint8_t* __restrict__ wq_a, const uint8_t* __restrict__ sc_a, float alpha_a,
    const uint8_t* __restrict__ wq_b, const uint8_t* __restrict__ sc_b, float alpha_b,
    const int* __restrict__ x_q8, const float* __restrict__ x_scales,
    __nv_bfloat16* __restrict__ y_a, __nv_bfloat16* __restrict__ y_b,
    int Na, int Nb, int K
) {
    int tid = threadIdx.x;
    int sub = tid & 15;
    int hw = tid >> 4;
    unsigned hmask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
    int kb_total = K >> 4;
    int kp_total = K >> 5;
    int total = Na + Nb;
    int n0 = blockIdx.x * 8 + hw;
    if (n0 >= total) return;
    const uint8_t* w_row;
    const uint8_t* sa_ptr;
    float al0;
    __nv_bfloat16* y0_b;
    int y0_c, y0_s;
    if (n0 < Na) {
        w_row = wq_a + (size_t)n0 * (K >> 1);
        sa_ptr = sc_a + (size_t)n0 * kb_total; al0 = alpha_a; y0_b = y_a; y0_c = n0; y0_s = Na;
    } else {
        int r = n0 - Na;
        w_row = wq_b + (size_t)r * (K >> 1);
        sa_ptr = sc_b + (size_t)r * kb_total; al0 = alpha_b; y0_b = y_b; y0_c = r; y0_s = Nb;
    }
    const int4* x4_base = reinterpret_cast<const int4*>(x_q8);
    const uint4* wa = reinterpret_cast<const uint4*>(w_row);
    const uint16_t* sa = reinterpret_cast<const uint16_t*>(sa_ptr);
    uint4 ra = (sub < kp_total) ? __ldcs(&wa[sub]) : make_uint4(0u, 0u, 0u, 0u);
    uint16_t sp = (sub < kp_total) ? __ldcs(&sa[sub]) : 0u;
    NVK_PDL_PROLOG();
    float acc0[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) acc0[m] = 0.f;
    int kp = sub;
    while (kp < kp_total) {
        int kpn = kp + 16;
        uint4 na = (kpn < kp_total) ? __ldcs(&wa[kpn]) : make_uint4(0u, 0u, 0u, 0u);
        uint16_t nsp = (kpn < kp_total) ? __ldcs(&sa[kpn]) : 0u;
        unsigned a0, a1, a2, a3, b0, b1, b2, b3;
        q4exp_nibbles8_to_i8(ra.x, a0, a1);
        q4exp_nibbles8_to_i8(ra.y, a2, a3);
        q4exp_nibbles8_to_i8(ra.z, b0, b1);
        q4exp_nibbles8_to_i8(ra.w, b2, b3);
        float fa = q4exp_decode_ue4m3((uint8_t)(sp & 0xFFu));
        float fb = q4exp_decode_ue4m3((uint8_t)(sp >> 8));
        int kb0 = kp * 2;
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            int4 xv0 = __ldg(&x4_base[m * kb_total + kb0]);
            int4 xv1 = __ldg(&x4_base[m * kb_total + kb0 + 1]);
            int d0 = __dp4a((int)a0, xv0.x, 0);
            d0 = __dp4a((int)a1, xv0.y, d0);
            d0 = __dp4a((int)a2, xv0.z, d0);
            d0 = __dp4a((int)a3, xv0.w, d0);
            int d1 = __dp4a((int)b0, xv1.x, 0);
            d1 = __dp4a((int)b1, xv1.y, d1);
            d1 = __dp4a((int)b2, xv1.z, d1);
            d1 = __dp4a((int)b3, xv1.w, d1);
            acc0[m] += fa * (float)d0 + fb * (float)d1;
        }
        kp = kpn;
        ra = na; sp = nsp;
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1) acc0[m] += __shfl_xor_sync(hmask, acc0[m], o, 16);
        if (sub == 0) {
            float xs = __ldg(&x_scales[m]);
            y0_b[(size_t)m * y0_s + y0_c] = __float2bfloat16(acc0[m] * (xs * al0));
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(128) q4exp_nvfp4_qkv_idx_i8_mt_hw16_kernel(
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ scq, float aq,
    const uint8_t* __restrict__ wk, const uint8_t* __restrict__ sck, float ak,
    const uint8_t* __restrict__ wv, const uint8_t* __restrict__ scv, float av,
    const uint8_t* __restrict__ wi, const uint8_t* __restrict__ sci, float ai,
    const int* __restrict__ x_q8, const float* __restrict__ x_scales,
    __nv_bfloat16* __restrict__ yq, __nv_bfloat16* __restrict__ yk,
    __nv_bfloat16* __restrict__ yv, __nv_bfloat16* __restrict__ yi,
    int nq, int nk, int nv, int ni, int K
) {
    int tid = threadIdx.x;
    int sub = tid & 15;
    int hw = tid >> 4;
    unsigned hmask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
    int kb_total = K >> 4;
    int kp_total = K >> 5;
    int total = nq + nk + nv + ni;
    int n0 = blockIdx.x * 8 + hw;
    if (n0 >= total) return;
    const uint8_t* w_row;
    const uint8_t* sa_ptr;
    float al0;
    __nv_bfloat16* y0_b;
    int y0_c, y0_s;
    if (n0 < nq) {
        w_row = wq + (size_t)n0 * (K >> 1);
        sa_ptr = scq + (size_t)n0 * kb_total; al0 = aq; y0_b = yq; y0_c = n0; y0_s = nq;
    } else if (n0 < nq + nk) {
        int r = n0 - nq;
        w_row = wk + (size_t)r * (K >> 1);
        sa_ptr = sck + (size_t)r * kb_total; al0 = ak; y0_b = yk; y0_c = r; y0_s = nk;
    } else if (n0 < nq + nk + nv) {
        int r = n0 - nq - nk;
        w_row = wv + (size_t)r * (K >> 1);
        sa_ptr = scv + (size_t)r * kb_total; al0 = av; y0_b = yv; y0_c = r; y0_s = nv;
    } else {
        int r = n0 - nq - nk - nv;
        w_row = wi + (size_t)r * (K >> 1);
        sa_ptr = sci + (size_t)r * kb_total; al0 = ai; y0_b = yi; y0_c = r; y0_s = ni;
    }
    const int4* x4_base = reinterpret_cast<const int4*>(x_q8);
    const uint4* wa = reinterpret_cast<const uint4*>(w_row);
    const uint16_t* sa = reinterpret_cast<const uint16_t*>(sa_ptr);
    uint4 ra = (sub < kp_total) ? __ldcs(&wa[sub]) : make_uint4(0u, 0u, 0u, 0u);
    uint16_t sp = (sub < kp_total) ? __ldcs(&sa[sub]) : 0u;
    NVK_PDL_PROLOG();
    float acc0[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) acc0[m] = 0.f;
    int kp = sub;
    while (kp < kp_total) {
        int kpn = kp + 16;
        uint4 na = (kpn < kp_total) ? __ldcs(&wa[kpn]) : make_uint4(0u, 0u, 0u, 0u);
        uint16_t nsp = (kpn < kp_total) ? __ldcs(&sa[kpn]) : 0u;
        unsigned a0, a1, a2, a3, b0, b1, b2, b3;
        q4exp_nibbles8_to_i8(ra.x, a0, a1);
        q4exp_nibbles8_to_i8(ra.y, a2, a3);
        q4exp_nibbles8_to_i8(ra.z, b0, b1);
        q4exp_nibbles8_to_i8(ra.w, b2, b3);
        float fa = q4exp_decode_ue4m3((uint8_t)(sp & 0xFFu));
        float fb = q4exp_decode_ue4m3((uint8_t)(sp >> 8));
        int kb0 = kp * 2;
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            int4 xv0 = __ldg(&x4_base[m * kb_total + kb0]);
            int4 xv1 = __ldg(&x4_base[m * kb_total + kb0 + 1]);
            int d0 = __dp4a((int)a0, xv0.x, 0);
            d0 = __dp4a((int)a1, xv0.y, d0);
            d0 = __dp4a((int)a2, xv0.z, d0);
            d0 = __dp4a((int)a3, xv0.w, d0);
            int d1 = __dp4a((int)b0, xv1.x, 0);
            d1 = __dp4a((int)b1, xv1.y, d1);
            d1 = __dp4a((int)b2, xv1.z, d1);
            d1 = __dp4a((int)b3, xv1.w, d1);
            acc0[m] += fa * (float)d0 + fb * (float)d1;
        }
        kp = kpn;
        ra = na; sp = nsp;
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1) acc0[m] += __shfl_xor_sync(hmask, acc0[m], o, 16);
        if (sub == 0) {
            float xs = __ldg(&x_scales[m]);
            y0_b[(size_t)m * y0_s + y0_c] = __float2bfloat16(acc0[m] * (xs * al0));
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(320) q4exp_hc_down_nvfp4_g4i8_kernel(
    const uint16_t* __restrict__ xn, const int* __restrict__ xn_q8, const float* __restrict__ xn_gsc,
    const uint8_t* __restrict__ w_down_q, const uint8_t* __restrict__ w_down_sc,
    float alpha_over_2g, const uint16_t* __restrict__ w_inject, float* __restrict__ lora_out, float* __restrict__ inject_out,
    int hh, int nb, int rank
) {
    __shared__ float red0[TM][10];
    __shared__ float red1[TM][10];
    int n_inj = (w_inject != nullptr) ? nb : 0;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    if ((int)blockIdx.x < n_inj) {
        int ib = (int)blockIdx.x;
        float acc[TM];
        #pragma unroll
        for (int m = 0; m < TM; ++m) acc[m] = 0.f;
        const uint4* wr = reinterpret_cast<const uint4*>(w_inject + (size_t)ib * hh);
        int Kv = hh >> 3;
        uint4 wcur = (tid < Kv) ? __ldcs(&wr[tid]) : make_uint4(0u, 0u, 0u, 0u);
        NVK_PDL_PROLOG();
        for (int v = tid; v < Kv; v += 320) {
            int vn = v + 320;
            uint4 wnxt = (vn < Kv) ? __ldcs(&wr[vn]) : make_uint4(0u, 0u, 0u, 0u);
            float wf[8];
            unpack8_bf16(wcur, wf);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                uint4 xv = *reinterpret_cast<const uint4*>(xn + (size_t)m * hh + (size_t)v * 8);
                float xf[8];
                unpack8_bf16(xv, xf);
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[m] += wf[j] * xf[j];
            }
            wcur = wnxt;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            float v = warp_sum(acc[m]);
            if (lane == 0) red0[m][warp] = v;
        }
        __syncthreads();
        if (tid < TM) {
            float tot = red0[tid][0] + red0[tid][1] + red0[tid][2] + red0[tid][3] + red0[tid][4]
                      + red0[tid][5] + red0[tid][6] + red0[tid][7] + red0[tid][8] + red0[tid][9];
            float v = bf16_round(tot) / (float)nb;
            inject_out[(size_t)tid * nb + ib] = 2.f * sigmoidf_(v);
        }
        NVK_PDL_EPILOG();
        return;
    }
    int row0 = ((int)blockIdx.x - n_inj) * 2;
    int row1 = row0 + 1;
    bool has1 = row1 < rank;
    int kb_total = hh >> 4;
    const uint4* wr0 = reinterpret_cast<const uint4*>(w_down_q + (size_t)row0 * (hh >> 1));
    const uint16_t* sr0 = reinterpret_cast<const uint16_t*>(w_down_sc + (size_t)row0 * kb_total);
    const uint4* wr1 = has1 ? reinterpret_cast<const uint4*>(w_down_q + (size_t)row1 * (hh >> 1)) : nullptr;
    const uint16_t* sr1 = has1 ? reinterpret_cast<const uint16_t*>(w_down_sc + (size_t)row1 * kb_total) : nullptr;
    const int4* x4_base = reinterpret_cast<const int4*>(xn_q8);
    uint4 wcur0 = __ldcs(&wr0[tid]);
    uint16_t scur0 = __ldcs(&sr0[tid]);
    uint4 wcur1 = has1 ? __ldcs(&wr1[tid]) : make_uint4(0u, 0u, 0u, 0u);
    uint16_t scur1 = has1 ? __ldcs(&sr1[tid]) : 0u;
    NVK_PDL_PROLOG();
    unsigned a0, a1, a2, a3, b0, b1, b2, b3;
    q4exp_nibbles8_to_i8(wcur0.x, a0, a1);
    q4exp_nibbles8_to_i8(wcur0.y, a2, a3);
    q4exp_nibbles8_to_i8(wcur0.z, b0, b1);
    q4exp_nibbles8_to_i8(wcur0.w, b2, b3);
    float sf0a = q4exp_decode_ue4m3((uint8_t)(scur0 & 0xFFu));
    float sf0b = q4exp_decode_ue4m3((uint8_t)(scur0 >> 8));
    unsigned c0 = 0u, c1 = 0u, c2 = 0u, c3 = 0u, e0 = 0u, e1 = 0u, e2 = 0u, e3 = 0u;
    float sf1a = 0.f, sf1b = 0.f;
    if (has1) {
        q4exp_nibbles8_to_i8(wcur1.x, c0, c1);
        q4exp_nibbles8_to_i8(wcur1.y, c2, c3);
        q4exp_nibbles8_to_i8(wcur1.z, e0, e1);
        q4exp_nibbles8_to_i8(wcur1.w, e2, e3);
        sf1a = q4exp_decode_ue4m3((uint8_t)(scur1 & 0xFFu));
        sf1b = q4exp_decode_ue4m3((uint8_t)(scur1 >> 8));
    }
    int g = tid / 80;
    int kb0 = tid * 2;
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        float gsc = __ldg(&xn_gsc[m * 4 + g]);
        int4 xv0 = __ldg(&x4_base[m * kb_total + kb0]);
        int4 xv1 = __ldg(&x4_base[m * kb_total + kb0 + 1]);
        int d0 = __dp4a((int)a0, xv0.x, 0);
        d0 = __dp4a((int)a1, xv0.y, d0);
        d0 = __dp4a((int)a2, xv0.z, d0);
        d0 = __dp4a((int)a3, xv0.w, d0);
        int d1 = __dp4a((int)b0, xv1.x, 0);
        d1 = __dp4a((int)b1, xv1.y, d1);
        d1 = __dp4a((int)b2, xv1.z, d1);
        d1 = __dp4a((int)b3, xv1.w, d1);
        float v0 = warp_sum(gsc * (sf0a * (float)d0 + sf0b * (float)d1));
        float v1 = 0.f;
        if (has1) {
            int f0 = __dp4a((int)c0, xv0.x, 0);
            f0 = __dp4a((int)c1, xv0.y, f0);
            f0 = __dp4a((int)c2, xv0.z, f0);
            f0 = __dp4a((int)c3, xv0.w, f0);
            int f1 = __dp4a((int)e0, xv1.x, 0);
            f1 = __dp4a((int)e1, xv1.y, f1);
            f1 = __dp4a((int)e2, xv1.z, f1);
            f1 = __dp4a((int)e3, xv1.w, f1);
            v1 = warp_sum(gsc * (sf1a * (float)f0 + sf1b * (float)f1));
        }
        if (lane == 0) {
            red0[m][warp] = v0;
            red1[m][warp] = v1;
        }
    }
    __syncthreads();
    if (tid < TM) {
        float tot0 = red0[tid][0] + red0[tid][1] + red0[tid][2] + red0[tid][3] + red0[tid][4]
                   + red0[tid][5] + red0[tid][6] + red0[tid][7] + red0[tid][8] + red0[tid][9];
        float v0 = bf16_round(tot0 * alpha_over_2g) / (float)nb;
        lora_out[(size_t)tid * rank + row0] = bf16_round(v0 * sigmoidf_(v0));
    } else if (has1 && tid < 2 * TM) {
        int m = tid - TM;
        float tot1 = red1[m][0] + red1[m][1] + red1[m][2] + red1[m][3] + red1[m][4]
                   + red1[m][5] + red1[m][6] + red1[m][7] + red1[m][8] + red1[m][9];
        float v1 = bf16_round(tot1 * alpha_over_2g) / (float)nb;
        lora_out[(size_t)m * rank + row1] = bf16_round(v1 * sigmoidf_(v1));
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(128) q4exp_nvfp4_down_tail_sub8_kernel(
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ scales, float alpha_over_2g,
    const __nv_bfloat16* __restrict__ act, const __nv_bfloat16* __restrict__ y_down,
    long long slot_stride, const float* __restrict__ topk_w, int k,
    const __nv_bfloat16* __restrict__ sg, __nv_bfloat16* __restrict__ out, int N, int K
) {
    extern __shared__ int act_q8[];
    __shared__ float s_amax[TM][4];
    __shared__ float a_sc[TM];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int sub = tid & 7;
    int sgrp = tid >> 3;
    int n = blockIdx.x * 16 + sgrp;
    bool active = n < N;
    int kb_total = K >> 4;
    int row_words = K >> 2;
    bool use_u4 = (K & 31) == 0;
    int kp_total = K >> 5;
    const uint4* wa4 = (active && use_u4) ? reinterpret_cast<const uint4*>(wq + (size_t)n * (K >> 1)) : nullptr;
    const uint16_t* sa2 = (active && use_u4) ? reinterpret_cast<const uint16_t*>(scales + (size_t)n * kb_total) : nullptr;
    const uint2* wa = (active && !use_u4) ? reinterpret_cast<const uint2*>(wq + (size_t)n * (K >> 1)) : nullptr;
    const uint8_t* sa = (active && !use_u4) ? (scales + (size_t)n * kb_total) : nullptr;
    uint4 ra4 = (active && use_u4 && sub < kp_total) ? __ldcs(&wa4[sub]) : make_uint4(0u, 0u, 0u, 0u);
    uint16_t sca2 = (active && use_u4 && sub < kp_total) ? __ldcs(&sa2[sub]) : 0u;
    uint2 ra = (active && !use_u4 && sub < kb_total) ? __ldcs(&wa[sub]) : make_uint2(0u, 0u);
    uint8_t sca = (active && !use_u4 && sub < kb_total) ? __ldcs(&sa[sub]) : 0u;
    NVK_PDL_PROLOG();
    int n8 = K >> 3;
    if (TM <= 4 && n8 <= 128) {
        if (warp < TM) {
            const uint4* a4 = reinterpret_cast<const uint4*>(act + (size_t)warp * K);
            float fv[4][8];
            float mx = 0.f;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                int i = lane + p * 32;
                if (i < n8) {
                    unpack8_bf16(__ldg(&a4[i]), fv[p]);
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fabsf(fv[p][j]));
                }
            }
            mx = warp_max(mx);
            float inv = (mx > 0.f) ? (127.f / mx) : 0.f;
            if (lane == 0) a_sc[warp] = mx * (1.f / 127.f);
            int* dst = act_q8 + warp * row_words;
            #pragma unroll
            for (int p = 0; p < 4; ++p) {
                int i = lane + p * 32;
                if (i < n8) {
                    int q[8];
                    #pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        q[j] = __float2int_rn(fminf(fmaxf(fv[p][j] * inv, -127.f), 127.f)) & 0xFF;
                    }
                    dst[2 * i] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
                    dst[2 * i + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
                }
            }
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            const uint4* a4 = reinterpret_cast<const uint4*>(act + (size_t)m * K);
            float mx = 0.f;
            for (int i = tid; i < n8; i += 128) {
                float f[8];
                unpack8_bf16(__ldg(&a4[i]), f);
                #pragma unroll
                for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fabsf(f[j]));
            }
            mx = warp_max(mx);
            if (lane == 0) s_amax[m][warp] = mx;
        }
        __syncthreads();
        if (warp < TM) {
            float mx = (lane < 4) ? s_amax[warp][lane] : 0.f;
            mx = warp_max(mx);
            if (lane == 0) {
                s_amax[warp][0] = (mx > 0.f) ? (127.f / mx) : 0.f;
                a_sc[warp] = mx * (1.f / 127.f);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            const uint4* a4 = reinterpret_cast<const uint4*>(act + (size_t)m * K);
            float inv = s_amax[m][0];
            int* dst = act_q8 + m * row_words;
            for (int i = tid; i < n8; i += 128) {
                float f[8];
                unpack8_bf16(__ldg(&a4[i]), f);
                int q[8];
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    q[j] = __float2int_rn(fminf(fmaxf(f[j] * inv, -127.f), 127.f)) & 0xFF;
                }
                dst[2 * i] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
                dst[2 * i + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
            }
        }
        __syncthreads();
    }
    if (!active) {
        NVK_PDL_EPILOG();
        return;
    }
    float acc[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) acc[m] = 0.f;
    if (use_u4) {
        const int4* act4 = reinterpret_cast<const int4*>(act_q8);
        int kp = sub;
        while (kp < kp_total) {
            int kpn = kp + 8;
            uint4 na4 = (kpn < kp_total) ? __ldcs(&wa4[kpn]) : make_uint4(0u, 0u, 0u, 0u);
            uint16_t nsa2 = (kpn < kp_total) ? __ldcs(&sa2[kpn]) : 0u;
            unsigned a0, a1, a2, a3, b0, b1, b2, b3;
            q4exp_nibbles8_to_i8(ra4.x, a0, a1);
            q4exp_nibbles8_to_i8(ra4.y, a2, a3);
            q4exp_nibbles8_to_i8(ra4.z, b0, b1);
            q4exp_nibbles8_to_i8(ra4.w, b2, b3);
            float fa = q4exp_decode_ue4m3((uint8_t)(sca2 & 0xFFu));
            float fb = q4exp_decode_ue4m3((uint8_t)(sca2 >> 8));
            int kb0 = kp * 2;
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                int4 xv0 = act4[m * kb_total + kb0];
                int4 xv1 = act4[m * kb_total + kb0 + 1];
                int d0 = __dp4a((int)a0, xv0.x, 0);
                d0 = __dp4a((int)a1, xv0.y, d0);
                d0 = __dp4a((int)a2, xv0.z, d0);
                d0 = __dp4a((int)a3, xv0.w, d0);
                int d1 = __dp4a((int)b0, xv1.x, 0);
                d1 = __dp4a((int)b1, xv1.y, d1);
                d1 = __dp4a((int)b2, xv1.z, d1);
                d1 = __dp4a((int)b3, xv1.w, d1);
                acc[m] += fa * (float)d0 + fb * (float)d1;
            }
            kp = kpn;
            ra4 = na4;
            sca2 = nsa2;
        }
    } else {
        int kb = sub;
        while (kb < kb_total) {
            int kbn = kb + 8;
            uint2 na = (kbn < kb_total) ? __ldcs(&wa[kbn]) : make_uint2(0u, 0u);
            uint8_t nsa = (kbn < kb_total) ? __ldcs(&sa[kbn]) : 0u;
            unsigned a0, a1, a2, a3;
            q4exp_nibbles8_to_i8(ra.x, a0, a1);
            q4exp_nibbles8_to_i8(ra.y, a2, a3);
            float fa = q4exp_decode_ue4m3(sca);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                const int* xw = act_q8 + m * row_words + kb * 4;
                int d = __dp4a((int)a0, xw[0], 0);
                d = __dp4a((int)a1, xw[1], d);
                d = __dp4a((int)a2, xw[2], d);
                d = __dp4a((int)a3, xw[3], d);
                acc[m] += fa * (float)d;
            }
            kb = kbn;
            ra = na;
            sca = nsa;
        }
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], o, 8);
    }
    if (sub < TM) {
        int j = sub;
        float s = __bfloat162float(__float2bfloat16(acc[j] * (a_sc[j] * alpha_over_2g)));
        float gl = __bfloat162float(__ldg(&sg[j]));
        float gate = 1.f / (1.f + expf(-gl));
        float routed = 0.f;
        #pragma unroll
        for (int q = 0; q < kQ4ExpTailMaxSlots; ++q) {
            if (q < k) {
                int slot = j * k + q;
                float wt = __ldg(&topk_w[slot]);
                float val = __bfloat162float(__ldg(&y_down[(size_t)slot * slot_stride + n]));
                routed += wt * val;
            }
        }
        out[(size_t)j * N + n] = __float2bfloat16(__fadd_rn(routed, __fmul_rn(s, gate)));
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(256) q4exp_hc_down_nvfp4_i8_kernel(
    const uint16_t* __restrict__ xn, const int* __restrict__ xn_q8, const float* __restrict__ xn_sc,
    const uint8_t* __restrict__ w_down_q, const uint8_t* __restrict__ w_down_sc,
    float alpha_over_2g, const uint16_t* __restrict__ w_inject, float* __restrict__ lora_out, float* __restrict__ inject_out,
    int hh, int nb, int rank
) {
    __shared__ float red[TM][8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float acc[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) acc[m] = 0.f;
    if (row < rank) {
        int kb_total = hh >> 4;
        const uint2* wr = reinterpret_cast<const uint2*>(w_down_q + (size_t)row * (hh >> 1));
        const uint8_t* sr = w_down_sc + (size_t)row * kb_total;
        const int4* x4_base = reinterpret_cast<const int4*>(xn_q8);
        uint2 wcur = (tid < kb_total) ? __ldcs(&wr[tid]) : make_uint2(0u, 0u);
        uint8_t scur = (tid < kb_total) ? __ldcs(&sr[tid]) : 0u;
        NVK_PDL_PROLOG();
        for (int kb = tid; kb < kb_total; kb += 256) {
            int kbn = kb + 256;
            uint2 wnxt = (kbn < kb_total) ? __ldcs(&wr[kbn]) : make_uint2(0u, 0u);
            uint8_t snxt = (kbn < kb_total) ? __ldcs(&sr[kbn]) : 0u;
            unsigned a0, a1, a2, a3;
            q4exp_nibbles8_to_i8(wcur.x, a0, a1);
            q4exp_nibbles8_to_i8(wcur.y, a2, a3);
            float sf = q4exp_decode_ue4m3(scur);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                int4 xv = __ldg(&x4_base[m * kb_total + kb]);
                int d = __dp4a((int)a0, xv.x, 0);
                d = __dp4a((int)a1, xv.y, d);
                d = __dp4a((int)a2, xv.z, d);
                d = __dp4a((int)a3, xv.w, d);
                acc[m] += sf * (float)d;
            }
            wcur = wnxt;
            scur = snxt;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            float v = warp_sum(acc[m]);
            if (lane == 0) red[m][warp] = v;
        }
        __syncthreads();
        if (tid < TM) {
            float tot = 0.f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[tid][w];
            float v = bf16_round(tot * (__ldg(&xn_sc[tid]) * alpha_over_2g)) / (float)nb;
            lora_out[(size_t)tid * rank + row] = bf16_round(v * sigmoidf_(v));
        }
    } else {
        const uint4* wr = reinterpret_cast<const uint4*>(w_inject + (size_t)(row - rank) * hh);
        int Kv = hh >> 3;
        uint4 wcur = (tid < Kv) ? __ldcs(&wr[tid]) : make_uint4(0u, 0u, 0u, 0u);
        NVK_PDL_PROLOG();
        for (int v = tid; v < Kv; v += 256) {
            int vn = v + 256;
            uint4 wnxt = (vn < Kv) ? __ldcs(&wr[vn]) : make_uint4(0u, 0u, 0u, 0u);
            float wf[8];
            unpack8_bf16(wcur, wf);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                uint4 xv = *reinterpret_cast<const uint4*>(xn + (size_t)m * hh + (size_t)v * 8);
                float xf[8];
                unpack8_bf16(xv, xf);
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[m] += wf[j] * xf[j];
            }
            wcur = wnxt;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            float v = warp_sum(acc[m]);
            if (lane == 0) red[m][warp] = v;
        }
        __syncthreads();
        if (tid < TM) {
            float tot = 0.f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[tid][w];
            float v = bf16_round(tot) / (float)nb;
            inject_out[(size_t)tid * nb + (row - rank)] = 2.f * sigmoidf_(v);
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(256) q4exp_nvfp4_gemv_mt_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ scales,
    const uint16_t* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    float alpha_over_2g,
    int N,
    int K
) {
    extern __shared__ int xs_q8_words[];
    __shared__ float s_amax[TM][8];
    __shared__ float x_scales[TM];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int kb_total = K >> 4;
    int row_words = K >> 2;
    int n0 = blockIdx.x * 16 + warp;
    int n1 = n0 + 8;
    bool has0 = n0 < N;
    bool has1 = n1 < N;
    const uint2* wa = has0 ? reinterpret_cast<const uint2*>(wq + (size_t)n0 * (K >> 1)) : nullptr;
    const uint2* wb = has1 ? reinterpret_cast<const uint2*>(wq + (size_t)n1 * (K >> 1)) : nullptr;
    const uint8_t* sa_ptr = has0 ? (scales + (size_t)n0 * kb_total) : nullptr;
    const uint8_t* sb_ptr = has1 ? (scales + (size_t)n1 * kb_total) : nullptr;
    uint2 ra = (has0 && lane < kb_total) ? __ldcs(&wa[lane]) : make_uint2(0u, 0u);
    uint2 rb = (has1 && lane < kb_total) ? __ldcs(&wb[lane]) : make_uint2(0u, 0u);
    uint8_t sca = (has0 && lane < kb_total) ? __ldcs(&sa_ptr[lane]) : 0u;
    uint8_t scb = (has1 && lane < kb_total) ? __ldcs(&sb_ptr[lane]) : 0u;
    NVK_PDL_PROLOG();
    q4exp_stage_x_i8_smem<TM>(x, xs_q8_words, x_scales, s_amax, K);
    if (!has0) {
        NVK_PDL_EPILOG();
        return;
    }
    float acc0[TM], acc1[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) { acc0[m] = 0.f; acc1[m] = 0.f; }
    int kb = lane;
    while (kb < kb_total) {
        int kb_next = kb + 32;
        uint2 na = (kb_next < kb_total) ? __ldcs(&wa[kb_next]) : make_uint2(0u, 0u);
        uint2 nb = (has1 && kb_next < kb_total) ? __ldcs(&wb[kb_next]) : make_uint2(0u, 0u);
        uint8_t nsa = (kb_next < kb_total) ? __ldcs(&sa_ptr[kb_next]) : 0u;
        uint8_t nsb = (has1 && kb_next < kb_total) ? __ldcs(&sb_ptr[kb_next]) : 0u;
        unsigned a0, a1, a2, a3;
        q4exp_nibbles8_to_i8(ra.x, a0, a1);
        q4exp_nibbles8_to_i8(ra.y, a2, a3);
        float fa = q4exp_decode_ue4m3(sca);
        unsigned b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
        float fb = 0.f;
        if (has1) {
            q4exp_nibbles8_to_i8(rb.x, b0, b1);
            q4exp_nibbles8_to_i8(rb.y, b2, b3);
            fb = q4exp_decode_ue4m3(scb);
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            const int* xw = xs_q8_words + m * row_words + kb * 4;
            int x0 = xw[0], x1 = xw[1], x2 = xw[2], x3 = xw[3];
            int d0 = __dp4a((int)a0, x0, 0);
            d0 = __dp4a((int)a1, x1, d0);
            d0 = __dp4a((int)a2, x2, d0);
            d0 = __dp4a((int)a3, x3, d0);
            acc0[m] += fa * (float)d0;
            if (has1) {
                int d1 = __dp4a((int)b0, x0, 0);
                d1 = __dp4a((int)b1, x1, d1);
                d1 = __dp4a((int)b2, x2, d1);
                d1 = __dp4a((int)b3, x3, d1);
                acc1[m] += fb * (float)d1;
            }
        }
        kb = kb_next;
        ra = na; rb = nb; sca = nsa; scb = nsb;
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        acc0[m] = warp_sum(acc0[m]);
        if (has1) acc1[m] = warp_sum(acc1[m]);
        if (lane == 0) {
            float sc = x_scales[m] * alpha_over_2g;
            y[(size_t)m * N + n0] = __float2bfloat16(acc0[m] * sc);
            if (has1) y[(size_t)m * N + n1] = __float2bfloat16(acc1[m] * sc);
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(256) q4exp_nvfp4_dual_gemv_mt_kernel(
    const uint8_t* __restrict__ wq_a, const uint8_t* __restrict__ sc_a, float alpha_a,
    const uint8_t* __restrict__ wq_b, const uint8_t* __restrict__ sc_b, float alpha_b,
    const uint16_t* __restrict__ x,
    __nv_bfloat16* __restrict__ y_a, __nv_bfloat16* __restrict__ y_b,
    int Na, int Nb, int K
) {
    extern __shared__ int xs_q8_words[];
    __shared__ float s_amax[TM][8];
    __shared__ float x_scales[TM];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int kb_total = K >> 4;
    int row_words = K >> 2;
    int total = Na + Nb;
    auto resolve = [&](int g, const uint2*& wp, const uint8_t*& sp, float& al, __nv_bfloat16*& yb, int& yc, int& ys) {
        if (g < Na) {
            wp = reinterpret_cast<const uint2*>(wq_a + (size_t)g * (K >> 1));
            sp = sc_a + (size_t)g * kb_total;
            al = alpha_a; yb = y_a; yc = g; ys = Na;
        } else {
            int r = g - Na;
            wp = reinterpret_cast<const uint2*>(wq_b + (size_t)r * (K >> 1));
            sp = sc_b + (size_t)r * kb_total;
            al = alpha_b; yb = y_b; yc = r; ys = Nb;
        }
    };
    NVK_PDL_PROLOG();
    q4exp_stage_x_i8_smem<TM>(x, xs_q8_words, x_scales, s_amax, K);
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        int n0 = blockIdx.x * 32 + pair * 16 + warp;
        int n1 = n0 + 8;
        bool has0 = n0 < total;
        if (!has0) continue;
        bool has1 = n1 < total;
        const uint2* wa = nullptr; const uint8_t* sa_ptr = nullptr; float al0 = 0.f; __nv_bfloat16* y0_b = nullptr; int y0_c = 0, y0_s = 1;
        const uint2* wb = nullptr; const uint8_t* sb_ptr = nullptr; float al1 = 0.f; __nv_bfloat16* y1_b = nullptr; int y1_c = 0, y1_s = 1;
        resolve(n0, wa, sa_ptr, al0, y0_b, y0_c, y0_s);
        if (has1) resolve(n1, wb, sb_ptr, al1, y1_b, y1_c, y1_s);
        uint2 ra = (lane < kb_total) ? __ldcs(&wa[lane]) : make_uint2(0u, 0u);
        uint2 rb = (has1 && lane < kb_total) ? __ldcs(&wb[lane]) : make_uint2(0u, 0u);
        uint8_t sca = (lane < kb_total) ? __ldcs(&sa_ptr[lane]) : 0u;
        uint8_t scb = (has1 && lane < kb_total) ? __ldcs(&sb_ptr[lane]) : 0u;
        float acc0[TM], acc1[TM];
        #pragma unroll
        for (int m = 0; m < TM; ++m) { acc0[m] = 0.f; acc1[m] = 0.f; }
        int kb = lane;
        while (kb < kb_total) {
            int kb_next = kb + 32;
            uint2 na = (kb_next < kb_total) ? __ldcs(&wa[kb_next]) : make_uint2(0u, 0u);
            uint2 nb = (has1 && kb_next < kb_total) ? __ldcs(&wb[kb_next]) : make_uint2(0u, 0u);
            uint8_t nsa = (kb_next < kb_total) ? __ldcs(&sa_ptr[kb_next]) : 0u;
            uint8_t nsb = (has1 && kb_next < kb_total) ? __ldcs(&sb_ptr[kb_next]) : 0u;
            unsigned a0, a1, a2, a3;
            q4exp_nibbles8_to_i8(ra.x, a0, a1);
            q4exp_nibbles8_to_i8(ra.y, a2, a3);
            float fa = q4exp_decode_ue4m3(sca);
            unsigned b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
            float fb = 0.f;
            if (has1) {
                q4exp_nibbles8_to_i8(rb.x, b0, b1);
                q4exp_nibbles8_to_i8(rb.y, b2, b3);
                fb = q4exp_decode_ue4m3(scb);
            }
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                const int* xw = xs_q8_words + m * row_words + kb * 4;
                int x0 = xw[0], x1 = xw[1], x2 = xw[2], x3 = xw[3];
                int d0 = __dp4a((int)a0, x0, 0);
                d0 = __dp4a((int)a1, x1, d0);
                d0 = __dp4a((int)a2, x2, d0);
                d0 = __dp4a((int)a3, x3, d0);
                acc0[m] += fa * (float)d0;
                if (has1) {
                    int d1 = __dp4a((int)b0, x0, 0);
                    d1 = __dp4a((int)b1, x1, d1);
                    d1 = __dp4a((int)b2, x2, d1);
                    d1 = __dp4a((int)b3, x3, d1);
                    acc1[m] += fb * (float)d1;
                }
            }
            kb = kb_next;
            ra = na; rb = nb; sca = nsa; scb = nsb;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            acc0[m] = warp_sum(acc0[m]);
            if (has1) acc1[m] = warp_sum(acc1[m]);
            if (lane == 0) {
                y0_b[(size_t)m * y0_s + y0_c] = __float2bfloat16(acc0[m] * (x_scales[m] * al0));
                if (has1) y1_b[(size_t)m * y1_s + y1_c] = __float2bfloat16(acc1[m] * (x_scales[m] * al1));
            }
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(256) q4exp_nvfp4_qkv_idx_gemv_mt_kernel(
    const uint8_t* __restrict__ wq, const uint8_t* __restrict__ scq, float aq,
    const uint8_t* __restrict__ wk, const uint8_t* __restrict__ sck, float ak,
    const uint8_t* __restrict__ wv, const uint8_t* __restrict__ scv, float av,
    const uint8_t* __restrict__ wi, const uint8_t* __restrict__ sci, float ai,
    const uint16_t* __restrict__ x,
    __nv_bfloat16* __restrict__ yq, __nv_bfloat16* __restrict__ yk,
    __nv_bfloat16* __restrict__ yv, __nv_bfloat16* __restrict__ yi,
    int nq, int nk, int nv, int ni, int K
) {
    extern __shared__ int xs_q8_words[];
    __shared__ float s_amax[TM][8];
    __shared__ float x_scales[TM];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int kb_total = K >> 4;
    int row_words = K >> 2;
    int total = nq + nk + nv + ni;
    int n0 = blockIdx.x * 16 + warp;
    int n1 = n0 + 8;
    bool has0 = n0 < total;
    bool has1 = n1 < total;
    auto resolve = [&](int g, const uint2*& wp, const uint8_t*& sp, float& al, __nv_bfloat16*& yb, int& yc, int& ys) {
        if (g < nq) {
            wp = reinterpret_cast<const uint2*>(wq + (size_t)g * (K >> 1));
            sp = scq + (size_t)g * kb_total; al = aq; yb = yq; yc = g; ys = nq;
        } else if (g < nq + nk) {
            int r = g - nq;
            wp = reinterpret_cast<const uint2*>(wk + (size_t)r * (K >> 1));
            sp = sck + (size_t)r * kb_total; al = ak; yb = yk; yc = r; ys = nk;
        } else if (g < nq + nk + nv) {
            int r = g - nq - nk;
            wp = reinterpret_cast<const uint2*>(wv + (size_t)r * (K >> 1));
            sp = scv + (size_t)r * kb_total; al = av; yb = yv; yc = r; ys = nv;
        } else {
            int r = g - nq - nk - nv;
            wp = reinterpret_cast<const uint2*>(wi + (size_t)r * (K >> 1));
            sp = sci + (size_t)r * kb_total; al = ai; yb = yi; yc = r; ys = ni;
        }
    };
    const uint2* wa = nullptr; const uint8_t* sa_ptr = nullptr; float al0 = 0.f; __nv_bfloat16* y0_b = nullptr; int y0_c = 0, y0_s = 1;
    const uint2* wb = nullptr; const uint8_t* sb_ptr = nullptr; float al1 = 0.f; __nv_bfloat16* y1_b = nullptr; int y1_c = 0, y1_s = 1;
    if (has0) resolve(n0, wa, sa_ptr, al0, y0_b, y0_c, y0_s);
    if (has1) resolve(n1, wb, sb_ptr, al1, y1_b, y1_c, y1_s);
    uint2 ra = (has0 && lane < kb_total) ? __ldcs(&wa[lane]) : make_uint2(0u, 0u);
    uint2 rb = (has1 && lane < kb_total) ? __ldcs(&wb[lane]) : make_uint2(0u, 0u);
    uint8_t sca = (has0 && lane < kb_total) ? __ldcs(&sa_ptr[lane]) : 0u;
    uint8_t scb = (has1 && lane < kb_total) ? __ldcs(&sb_ptr[lane]) : 0u;
    NVK_PDL_PROLOG();
    q4exp_stage_x_i8_smem<TM>(x, xs_q8_words, x_scales, s_amax, K);
    if (!has0) {
        NVK_PDL_EPILOG();
        return;
    }
    float acc0[TM], acc1[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) { acc0[m] = 0.f; acc1[m] = 0.f; }
    int kb = lane;
    while (kb < kb_total) {
        int kb_next = kb + 32;
        uint2 na = (kb_next < kb_total) ? __ldcs(&wa[kb_next]) : make_uint2(0u, 0u);
        uint2 nb = (has1 && kb_next < kb_total) ? __ldcs(&wb[kb_next]) : make_uint2(0u, 0u);
        uint8_t nsa = (kb_next < kb_total) ? __ldcs(&sa_ptr[kb_next]) : 0u;
        uint8_t nsb = (has1 && kb_next < kb_total) ? __ldcs(&sb_ptr[kb_next]) : 0u;
        unsigned a0, a1, a2, a3;
        q4exp_nibbles8_to_i8(ra.x, a0, a1);
        q4exp_nibbles8_to_i8(ra.y, a2, a3);
        float fa = q4exp_decode_ue4m3(sca);
        unsigned b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u;
        float fb = 0.f;
        if (has1) {
            q4exp_nibbles8_to_i8(rb.x, b0, b1);
            q4exp_nibbles8_to_i8(rb.y, b2, b3);
            fb = q4exp_decode_ue4m3(scb);
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            const int* xw = xs_q8_words + m * row_words + kb * 4;
            int x0 = xw[0], x1 = xw[1], x2 = xw[2], x3 = xw[3];
            int d0 = __dp4a((int)a0, x0, 0);
            d0 = __dp4a((int)a1, x1, d0);
            d0 = __dp4a((int)a2, x2, d0);
            d0 = __dp4a((int)a3, x3, d0);
            acc0[m] += fa * (float)d0;
            if (has1) {
                int d1 = __dp4a((int)b0, x0, 0);
                d1 = __dp4a((int)b1, x1, d1);
                d1 = __dp4a((int)b2, x2, d1);
                d1 = __dp4a((int)b3, x3, d1);
                acc1[m] += fb * (float)d1;
            }
        }
        kb = kb_next;
        ra = na; rb = nb; sca = nsa; scb = nsb;
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        acc0[m] = warp_sum(acc0[m]);
        if (has1) acc1[m] = warp_sum(acc1[m]);
        if (lane == 0) {
            y0_b[(size_t)m * y0_s + y0_c] = __float2bfloat16(acc0[m] * (x_scales[m] * al0));
            if (has1) y1_b[(size_t)m * y1_s + y1_c] = __float2bfloat16(acc1[m] * (x_scales[m] * al1));
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(256) q4exp_hc_down_nvfp4_kernel(
    const uint16_t* __restrict__ xn, const uint8_t* __restrict__ w_down_q, const uint8_t* __restrict__ w_down_sc,
    float alpha_over_2g, const uint16_t* __restrict__ w_inject, float* __restrict__ lora_out, float* __restrict__ inject_out,
    int hh, int nb, int rank
) {
    __shared__ float red[TM][8];
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    float acc[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) acc[m] = 0.f;
    if (row < rank) {
        int kb_total = hh >> 4;
        const uint2* wr = reinterpret_cast<const uint2*>(w_down_q + (size_t)row * (hh >> 1));
        const uint8_t* sr = w_down_sc + (size_t)row * kb_total;
        uint2 wcur = (tid < kb_total) ? __ldcs(&wr[tid]) : make_uint2(0u, 0u);
        uint8_t scur = (tid < kb_total) ? __ldcs(&sr[tid]) : 0u;
        NVK_PDL_PROLOG();
        for (int kb = tid; kb < kb_total; kb += 256) {
            int kbn = kb + 256;
            uint2 wnxt = (kbn < kb_total) ? __ldcs(&wr[kbn]) : make_uint2(0u, 0u);
            uint8_t snxt = (kbn < kb_total) ? __ldcs(&sr[kbn]) : 0u;
            unsigned wi[4];
            q4exp_nibbles8_to_i8(wcur.x, wi[0], wi[1]);
            q4exp_nibbles8_to_i8(wcur.y, wi[2], wi[3]);
            float wf[16];
            #pragma unroll
            for (int b = 0; b < 4; ++b) {
                int w = (int)wi[b];
                wf[4 * b + 0] = (float)(int8_t)(w & 0xFF);
                wf[4 * b + 1] = (float)(int8_t)((w >> 8) & 0xFF);
                wf[4 * b + 2] = (float)(int8_t)((w >> 16) & 0xFF);
                wf[4 * b + 3] = (float)(int8_t)((w >> 24) & 0xFF);
            }
            float sf = q4exp_decode_ue4m3(scur);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                const uint4* xr = reinterpret_cast<const uint4*>(xn + (size_t)m * hh + (size_t)kb * 16);
                float xf0[8], xf1[8];
                unpack8_bf16(__ldg(&xr[0]), xf0);
                unpack8_bf16(__ldg(&xr[1]), xf1);
                float dot = 0.f;
                #pragma unroll
                for (int j = 0; j < 8; ++j) dot += wf[j] * xf0[j] + wf[8 + j] * xf1[j];
                acc[m] += sf * dot;
            }
            wcur = wnxt;
            scur = snxt;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            float v = warp_sum(acc[m]);
            if (lane == 0) red[m][warp] = v;
        }
        __syncthreads();
        if (tid < TM) {
            float tot = 0.f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[tid][w];
            float v = bf16_round(tot * alpha_over_2g) / (float)nb;
            lora_out[(size_t)tid * rank + row] = bf16_round(v * sigmoidf_(v));
        }
    } else {
        const uint4* wr = reinterpret_cast<const uint4*>(w_inject + (size_t)(row - rank) * hh);
        int Kv = hh >> 3;
        uint4 wcur = (tid < Kv) ? __ldcs(&wr[tid]) : make_uint4(0u, 0u, 0u, 0u);
        NVK_PDL_PROLOG();
        for (int v = tid; v < Kv; v += 256) {
            int vn = v + 256;
            uint4 wnxt = (vn < Kv) ? __ldcs(&wr[vn]) : make_uint4(0u, 0u, 0u, 0u);
            float wf[8];
            unpack8_bf16(wcur, wf);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                uint4 xv = *reinterpret_cast<const uint4*>(xn + (size_t)m * hh + (size_t)v * 8);
                float xf[8];
                unpack8_bf16(xv, xf);
                #pragma unroll
                for (int j = 0; j < 8; ++j) acc[m] += wf[j] * xf[j];
            }
            wcur = wnxt;
        }
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            float v = warp_sum(acc[m]);
            if (lane == 0) red[m][warp] = v;
        }
        __syncthreads();
        if (tid < TM) {
            float tot = 0.f;
            #pragma unroll
            for (int w = 0; w < 8; ++w) tot += red[tid][w];
            float v = bf16_round(tot) / (float)nb;
            inject_out[(size_t)tid * nb + (row - rank)] = 2.f * sigmoidf_(v);
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(256) q4exp_hc_up_mix_nvfp4_r320_nb4_kernel(
    const float* __restrict__ lora, const uint8_t* __restrict__ w_up_q, const uint8_t* __restrict__ w_up_sc,
    float alpha_over_2g, const uint16_t* __restrict__ xn, uint16_t* __restrict__ mixed, int h
) {
    __shared__ alignas(16) int ls_q8[TM * 80];
    __shared__ float x_scales[TM];
    __shared__ float s_part[TM][8][4];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int sub = tid & 7;
    int sg = tid >> 3;
    int j_loc = sg & 7;
    int b = sg >> 3;
    int j = blockIdx.x * 8 + j_loc;
    bool active = j < h;
    bool has_hi = active && (sub < 2);
    int row_idx = b * h + j;
    const uint4* wr4 = reinterpret_cast<const uint4*>(w_up_q + (size_t)(active ? row_idx : 0) * 160);
    const uint16_t* sr2 = reinterpret_cast<const uint16_t*>(w_up_sc + (size_t)(active ? row_idx : 0) * 20);
    uint4 w0 = active ? __ldcs(&wr4[sub]) : make_uint4(0u, 0u, 0u, 0u);
    uint16_t s0 = active ? __ldcs(&sr2[sub]) : 0u;
    uint4 w1 = has_hi ? __ldcs(&wr4[8 + sub]) : make_uint4(0u, 0u, 0u, 0u);
    uint16_t s1 = has_hi ? __ldcs(&sr2[8 + sub]) : 0u;
    NVK_PDL_PROLOG();
    if (warp < TM) {
        const float4* l4 = reinterpret_cast<const float4*>(lora + warp * 320);
        float4 v0 = __ldg(&l4[lane]);
        float4 v1 = __ldg(&l4[lane + 32]);
        float4 v2 = (lane < 16) ? __ldg(&l4[lane + 64]) : make_float4(0.f, 0.f, 0.f, 0.f);
        float mx = fmaxf(fmaxf(fabsf(v0.x), fabsf(v0.y)), fmaxf(fabsf(v0.z), fabsf(v0.w)));
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v1.x), fabsf(v1.y)), fmaxf(fabsf(v1.z), fabsf(v1.w))));
        if (lane < 16) {
            mx = fmaxf(mx, fmaxf(fmaxf(fabsf(v2.x), fabsf(v2.y)), fmaxf(fabsf(v2.z), fabsf(v2.w))));
        }
        mx = warp_max(mx);
        float inv = (mx > 0.f) ? (127.f / mx) : 0.f;
        if (lane == 0) x_scales[warp] = mx * (1.f / 127.f);
        auto pack4 = [inv](float4 v) -> int {
            int q0 = __float2int_rn(fminf(fmaxf(v.x * inv, -127.f), 127.f)) & 0xFF;
            int q1 = __float2int_rn(fminf(fmaxf(v.y * inv, -127.f), 127.f)) & 0xFF;
            int q2 = __float2int_rn(fminf(fmaxf(v.z * inv, -127.f), 127.f)) & 0xFF;
            int q3 = __float2int_rn(fminf(fmaxf(v.w * inv, -127.f), 127.f)) & 0xFF;
            return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
        };
        int* dst = ls_q8 + warp * 80;
        dst[lane] = pack4(v0);
        dst[lane + 32] = pack4(v1);
        if (lane < 16) dst[lane + 64] = pack4(v2);
    }
    __syncthreads();
    unsigned a0, a1, a2, a3, a4, a5, a6, a7;
    q4exp_nibbles8_to_i8(w0.x, a0, a1);
    q4exp_nibbles8_to_i8(w0.y, a2, a3);
    q4exp_nibbles8_to_i8(w0.z, a4, a5);
    q4exp_nibbles8_to_i8(w0.w, a6, a7);
    float f00 = q4exp_decode_ue4m3((uint8_t)(s0 & 0xFFu));
    float f01 = q4exp_decode_ue4m3((uint8_t)(s0 >> 8));
    unsigned b0 = 0u, b1 = 0u, b2 = 0u, b3 = 0u, b4 = 0u, b5 = 0u, b6 = 0u, b7 = 0u;
    float f10 = 0.f, f11 = 0.f;
    if (has_hi) {
        q4exp_nibbles8_to_i8(w1.x, b0, b1);
        q4exp_nibbles8_to_i8(w1.y, b2, b3);
        q4exp_nibbles8_to_i8(w1.z, b4, b5);
        q4exp_nibbles8_to_i8(w1.w, b6, b7);
        f10 = q4exp_decode_ue4m3((uint8_t)(s1 & 0xFFu));
        f11 = q4exp_decode_ue4m3((uint8_t)(s1 >> 8));
    }
    int hh = 4 * h;
    const int4* ls4 = reinterpret_cast<const int4*>(ls_q8);
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        int4 xw0 = ls4[m * 20 + 2 * sub];
        int4 xw1 = ls4[m * 20 + 2 * sub + 1];
        int d0 = __dp4a((int)a0, xw0.x, 0);
        d0 = __dp4a((int)a1, xw0.y, d0);
        d0 = __dp4a((int)a2, xw0.z, d0);
        d0 = __dp4a((int)a3, xw0.w, d0);
        int d1 = __dp4a((int)a4, xw1.x, 0);
        d1 = __dp4a((int)a5, xw1.y, d1);
        d1 = __dp4a((int)a6, xw1.z, d1);
        d1 = __dp4a((int)a7, xw1.w, d1);
        float acc = f00 * (float)d0 + f01 * (float)d1;
        if (has_hi) {
            int4 xh0 = ls4[m * 20 + 16 + 2 * sub];
            int4 xh1 = ls4[m * 20 + 16 + 2 * sub + 1];
            int e0 = __dp4a((int)b0, xh0.x, 0);
            e0 = __dp4a((int)b1, xh0.y, e0);
            e0 = __dp4a((int)b2, xh0.z, e0);
            e0 = __dp4a((int)b3, xh0.w, e0);
            int e1 = __dp4a((int)b4, xh1.x, 0);
            e1 = __dp4a((int)b5, xh1.y, e1);
            e1 = __dp4a((int)b6, xh1.z, e1);
            e1 = __dp4a((int)b7, xh1.w, e1);
            acc += f10 * (float)e0 + f11 * (float)e1;
        }
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o, 8);
        if (active && sub == 0) {
            float g = sigmoidf_(bf16_round(acc * (x_scales[m] * alpha_over_2g)));
            s_part[m][j_loc][b] = g * ld_bf16(xn + (size_t)m * hh + row_idx);
        }
    }
    __syncthreads();
    if (tid < TM * 8) {
        int m = tid >> 3;
        int jl = tid & 7;
        int jj = blockIdx.x * 8 + jl;
        if (jj < h) {
            float sum = s_part[m][jl][0] + s_part[m][jl][1] + s_part[m][jl][2] + s_part[m][jl][3];
            st_bf16(mixed + (size_t)m * h + jj, sum * 0.25f);
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
static int q4exp_launch_nvfp4_gemv_mt(
    cudaStream_t s, const uint8_t* wq, const uint8_t* scales, const uint16_t* x, uint16_t* y,
    float alpha_over_2g, int N, int K
) {
    if (g_q4exp_i8_buf != nullptr && K <= kQ4ExpI8MaxK) {
        dim3 g_rq((unsigned)TM);
        dim3 g_mv((unsigned)((N + 3) / 4));
        if (nvk_pdl_enabled()) {
            if (x != nullptr) {
                NVK_PDL_ATTR(c1, g_rq, dim3(256), 0, s);
                cudaLaunchKernelEx(&c1, q4exp_rowquant_i8_kernel, x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
            }
            NVK_PDL_ATTR(c2, g_mv, dim3(128), 0, s);
            cudaLaunchKernelEx(&c2, q4exp_nvfp4_gemv_i8_mt_kernel<TM>,
                wq, scales, g_q4exp_i8_buf, g_q4exp_i8_sc, reinterpret_cast<__nv_bfloat16*>(y), alpha_over_2g, N, K);
        } else {
            if (x != nullptr) {
                q4exp_rowquant_i8_kernel<<<g_rq, 256, 0, s>>>(x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
            }
            q4exp_nvfp4_gemv_i8_mt_kernel<TM><<<g_mv, 128, 0, s>>>(
                wq, scales, g_q4exp_i8_buf, g_q4exp_i8_sc, reinterpret_cast<__nv_bfloat16*>(y), alpha_over_2g, N, K);
        }
        return (int)cudaGetLastError();
    }
    if (x == nullptr) return -3;
    size_t smem = (size_t)TM * (size_t)K;
    if (smem > 48 * 1024) return -3;
    dim3 grid((unsigned)((N + 15) / 16));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_nvfp4_gemv_mt_kernel<TM>,
            wq, scales, x, reinterpret_cast<__nv_bfloat16*>(y), alpha_over_2g, N, K);
    } else {
        q4exp_nvfp4_gemv_mt_kernel<TM><<<grid, 256, smem, s>>>(
            wq, scales, x, reinterpret_cast<__nv_bfloat16*>(y), alpha_over_2g, N, K);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_nvfp4_dual_gemv_mt(
    cudaStream_t s, const uint8_t* wq_a, const uint8_t* sc_a, float alpha_a,
    const uint8_t* wq_b, const uint8_t* sc_b, float alpha_b,
    const uint16_t* x, uint16_t* y_a, uint16_t* y_b, int Na, int Nb, int K
) {
    if (g_q4exp_i8_buf != nullptr && K <= kQ4ExpI8MaxK) {
        dim3 g_rq((unsigned)TM);
        if ((K & 511) == 0 && (K & 1023) != 0) {
            dim3 g_mv((unsigned)((Na + Nb + 7) / 8));
            if (nvk_pdl_enabled()) {
                NVK_PDL_ATTR(c1, g_rq, dim3(256), 0, s);
                cudaLaunchKernelEx(&c1, q4exp_rowquant_i8_kernel, x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
                NVK_PDL_ATTR(c2, g_mv, dim3(128), 0, s);
                cudaLaunchKernelEx(&c2, q4exp_nvfp4_dual_i8_mt_hw16_kernel<TM>,
                    wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, g_q4exp_i8_buf, g_q4exp_i8_sc,
                    reinterpret_cast<__nv_bfloat16*>(y_a), reinterpret_cast<__nv_bfloat16*>(y_b), Na, Nb, K);
            } else {
                q4exp_rowquant_i8_kernel<<<g_rq, 256, 0, s>>>(x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
                q4exp_nvfp4_dual_i8_mt_hw16_kernel<TM><<<g_mv, 128, 0, s>>>(
                    wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, g_q4exp_i8_buf, g_q4exp_i8_sc,
                    reinterpret_cast<__nv_bfloat16*>(y_a), reinterpret_cast<__nv_bfloat16*>(y_b), Na, Nb, K);
            }
            return (int)cudaGetLastError();
        }
        dim3 g_mv((unsigned)((Na + Nb + 3) / 4));
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(c1, g_rq, dim3(256), 0, s);
            cudaLaunchKernelEx(&c1, q4exp_rowquant_i8_kernel, x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
            NVK_PDL_ATTR(c2, g_mv, dim3(128), 0, s);
            cudaLaunchKernelEx(&c2, q4exp_nvfp4_dual_i8_mt_kernel<TM>,
                wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, g_q4exp_i8_buf, g_q4exp_i8_sc,
                reinterpret_cast<__nv_bfloat16*>(y_a), reinterpret_cast<__nv_bfloat16*>(y_b), Na, Nb, K);
        } else {
            q4exp_rowquant_i8_kernel<<<g_rq, 256, 0, s>>>(x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
            q4exp_nvfp4_dual_i8_mt_kernel<TM><<<g_mv, 128, 0, s>>>(
                wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, g_q4exp_i8_buf, g_q4exp_i8_sc,
                reinterpret_cast<__nv_bfloat16*>(y_a), reinterpret_cast<__nv_bfloat16*>(y_b), Na, Nb, K);
        }
        return (int)cudaGetLastError();
    }
    size_t smem = (size_t)TM * (size_t)K;
    if (smem > 48 * 1024) return -3;
    dim3 grid((unsigned)((Na + Nb + 31) / 32));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_nvfp4_dual_gemv_mt_kernel<TM>,
            wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x,
            reinterpret_cast<__nv_bfloat16*>(y_a), reinterpret_cast<__nv_bfloat16*>(y_b), Na, Nb, K);
    } else {
        q4exp_nvfp4_dual_gemv_mt_kernel<TM><<<grid, 256, smem, s>>>(
            wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x,
            reinterpret_cast<__nv_bfloat16*>(y_a), reinterpret_cast<__nv_bfloat16*>(y_b), Na, Nb, K);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_nvfp4_qkv_idx_gemv_mt(
    cudaStream_t s,
    const uint8_t* wq, const uint8_t* scq, float aq,
    const uint8_t* wk, const uint8_t* sck, float ak,
    const uint8_t* wv, const uint8_t* scv, float av,
    const uint8_t* wi, const uint8_t* sci, float ai,
    const uint16_t* x, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yi,
    int nq, int nk, int nv, int ni, int K
) {
    if (g_q4exp_i8_buf != nullptr && K <= kQ4ExpI8MaxK) {
        dim3 g_rq((unsigned)TM);
        if ((K & 511) == 0 && (K & 1023) != 0) {
            dim3 g_mv((unsigned)((nq + nk + nv + ni + 7) / 8));
            if (nvk_pdl_enabled()) {
                NVK_PDL_ATTR(c1, g_rq, dim3(256), 0, s);
                cudaLaunchKernelEx(&c1, q4exp_rowquant_i8_kernel, x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
                NVK_PDL_ATTR(c2, g_mv, dim3(128), 0, s);
                cudaLaunchKernelEx(&c2, q4exp_nvfp4_qkv_idx_i8_mt_hw16_kernel<TM>,
                    wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, g_q4exp_i8_buf, g_q4exp_i8_sc,
                    reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk),
                    reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yi),
                    nq, nk, nv, ni, K);
            } else {
                q4exp_rowquant_i8_kernel<<<g_rq, 256, 0, s>>>(x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
                q4exp_nvfp4_qkv_idx_i8_mt_hw16_kernel<TM><<<g_mv, 128, 0, s>>>(
                    wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, g_q4exp_i8_buf, g_q4exp_i8_sc,
                    reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk),
                    reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yi),
                    nq, nk, nv, ni, K);
            }
            return (int)cudaGetLastError();
        }
        dim3 g_mv((unsigned)((nq + nk + nv + ni + 3) / 4));
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(c1, g_rq, dim3(256), 0, s);
            cudaLaunchKernelEx(&c1, q4exp_rowquant_i8_kernel, x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
            NVK_PDL_ATTR(c2, g_mv, dim3(128), 0, s);
            cudaLaunchKernelEx(&c2, q4exp_nvfp4_qkv_idx_i8_mt_kernel<TM>,
                wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, g_q4exp_i8_buf, g_q4exp_i8_sc,
                reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk),
                reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yi),
                nq, nk, nv, ni, K);
        } else {
            q4exp_rowquant_i8_kernel<<<g_rq, 256, 0, s>>>(x, g_q4exp_i8_buf, g_q4exp_i8_sc, K);
            q4exp_nvfp4_qkv_idx_i8_mt_kernel<TM><<<g_mv, 128, 0, s>>>(
                wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, g_q4exp_i8_buf, g_q4exp_i8_sc,
                reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk),
                reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yi),
                nq, nk, nv, ni, K);
        }
        return (int)cudaGetLastError();
    }
    size_t smem = (size_t)TM * (size_t)K;
    if (smem > 48 * 1024) return -3;
    dim3 grid((unsigned)((nq + nk + nv + ni + 15) / 16));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_nvfp4_qkv_idx_gemv_mt_kernel<TM>,
            wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x,
            reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk),
            reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yi),
            nq, nk, nv, ni, K);
    } else {
        q4exp_nvfp4_qkv_idx_gemv_mt_kernel<TM><<<grid, 256, smem, s>>>(
            wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x,
            reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk),
            reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yi),
            nq, nk, nv, ni, K);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_hc_down_nvfp4(
    cudaStream_t s, const uint16_t* xn, const uint8_t* wq, const uint8_t* sc, float alpha,
    const uint16_t* w_inject, float* lora_out, float* inject_out, int hh, int nb, int rank
) {
    int rows = rank + (w_inject ? nb : 0);
    dim3 grid((unsigned)rows, 1);
    if (g_q4exp_hc_xn_q8 != nullptr && hh == 10240 && nb == 4 && TM <= 8) {
        int n_inj = w_inject ? nb : 0;
        dim3 g_1w((unsigned)(n_inj + (rank + 1) / 2), 1);
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(c2, g_1w, dim3(320), 0, s);
            cudaLaunchKernelEx(&c2, q4exp_hc_down_nvfp4_g4i8_kernel<TM>,
                xn, g_q4exp_hc_xn_q8, g_q4exp_hc_xn_gsc, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        } else {
            q4exp_hc_down_nvfp4_g4i8_kernel<TM><<<g_1w, 320, 0, s>>>(
                xn, g_q4exp_hc_xn_q8, g_q4exp_hc_xn_gsc, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        }
        return (int)cudaGetLastError();
    }
    if (g_q4exp_i8_buf != nullptr && hh <= kQ4ExpI8MaxK) {
        dim3 g_rq((unsigned)TM);
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(c1, g_rq, dim3(256), 0, s);
            cudaLaunchKernelEx(&c1, q4exp_rowquant_i8_kernel, xn, g_q4exp_i8_buf, g_q4exp_i8_sc, hh);
            NVK_PDL_ATTR(c2, grid, dim3(256), 0, s);
            cudaLaunchKernelEx(&c2, q4exp_hc_down_nvfp4_i8_kernel<TM>,
                xn, g_q4exp_i8_buf, g_q4exp_i8_sc, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        } else {
            q4exp_rowquant_i8_kernel<<<g_rq, 256, 0, s>>>(xn, g_q4exp_i8_buf, g_q4exp_i8_sc, hh);
            q4exp_hc_down_nvfp4_i8_kernel<TM><<<grid, 256, 0, s>>>(
                xn, g_q4exp_i8_buf, g_q4exp_i8_sc, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        }
        return (int)cudaGetLastError();
    }
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_hc_down_nvfp4_kernel<TM>,
            xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
    } else {
        q4exp_hc_down_nvfp4_kernel<TM><<<grid, 256, 0, s>>>(
            xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_hc_up_mix_nvfp4(
    cudaStream_t s, const float* lora, const uint8_t* wq, const uint8_t* sc, float alpha,
    const uint16_t* xn, uint16_t* mixed, int h
) {
    dim3 grid((unsigned)((h + 7) / 8), 1);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), 0, s);
        cudaLaunchKernelEx(&cfg, q4exp_hc_up_mix_nvfp4_r320_nb4_kernel<TM>,
            lora, wq, sc, alpha, xn, mixed, h);
    } else {
        q4exp_hc_up_mix_nvfp4_r320_nb4_kernel<TM><<<grid, 256, 0, s>>>(
            lora, wq, sc, alpha, xn, mixed, h);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_nvfp4_down_tail(
    cudaStream_t s, const uint8_t* wq, const uint8_t* sc, float alpha,
    const uint16_t* act, const uint16_t* y_down, long long slot_stride,
    const float* topk_w, int topk, const uint16_t* sg, uint16_t* out, int N, int K
) {
    size_t smem = (size_t)TM * (size_t)K;
    if (smem > 48 * 1024) return -3;
    dim3 grid((unsigned)((N + 15) / 16));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(128), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_nvfp4_down_tail_sub8_kernel<TM>,
            wq, sc, alpha, reinterpret_cast<const __nv_bfloat16*>(act),
            reinterpret_cast<const __nv_bfloat16*>(y_down), slot_stride, topk_w, topk,
            reinterpret_cast<const __nv_bfloat16*>(sg), reinterpret_cast<__nv_bfloat16*>(out), N, K);
    } else {
        q4exp_nvfp4_down_tail_sub8_kernel<TM><<<grid, 128, smem, s>>>(
            wq, sc, alpha, reinterpret_cast<const __nv_bfloat16*>(act),
            reinterpret_cast<const __nv_bfloat16*>(y_down), slot_stride, topk_w, topk,
            reinterpret_cast<const __nv_bfloat16*>(sg), reinterpret_cast<__nv_bfloat16*>(out), N, K);
    }
    return (int)cudaGetLastError();
}

template <int TM>
__global__ void __launch_bounds__(128) q4exp_nvfp4_gateup_silu_kernel(
    const uint8_t* __restrict__ wq_g, const uint8_t* __restrict__ sc_g, float alpha_g,
    const uint8_t* __restrict__ wq_u, const uint8_t* __restrict__ sc_u, float alpha_u,
    const uint16_t* __restrict__ x, __nv_bfloat16* __restrict__ act, int N, int K
) {
    extern __shared__ int xs_q8[];
    __shared__ float s_amax[TM][4];
    __shared__ float x_sc[TM];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int n = blockIdx.x * 4 + warp;
    bool active = n < N;
    int kb_total = K >> 4;
    int row_words = K >> 2;
    const uint2* wg = active ? reinterpret_cast<const uint2*>(wq_g + (size_t)n * (K >> 1)) : nullptr;
    const uint8_t* sg = active ? (sc_g + (size_t)n * kb_total) : nullptr;
    const uint2* wu = active ? reinterpret_cast<const uint2*>(wq_u + (size_t)n * (K >> 1)) : nullptr;
    const uint8_t* su = active ? (sc_u + (size_t)n * kb_total) : nullptr;
    uint2 rg = (active && lane < kb_total) ? __ldcs(&wg[lane]) : make_uint2(0u, 0u);
    uint8_t sgc = (active && lane < kb_total) ? __ldcs(&sg[lane]) : 0u;
    uint2 ru = (active && lane < kb_total) ? __ldcs(&wu[lane]) : make_uint2(0u, 0u);
    uint8_t suc = (active && lane < kb_total) ? __ldcs(&su[lane]) : 0u;
    NVK_PDL_PROLOG();
    int n8 = K >> 3;
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)m * K);
        float mx = 0.f;
        for (int i = tid; i < n8; i += 128) {
            float f[8];
            unpack8_bf16(__ldg(&x4[i]), f);
            #pragma unroll
            for (int j = 0; j < 8; ++j) mx = fmaxf(mx, fabsf(f[j]));
        }
        mx = warp_max(mx);
        if (lane == 0) s_amax[m][warp] = mx;
    }
    __syncthreads();
    if (warp < TM) {
        float mx = (lane < 4) ? s_amax[warp][lane] : 0.f;
        mx = warp_max(mx);
        if (lane == 0) {
            s_amax[warp][0] = (mx > 0.f) ? (127.f / mx) : 0.f;
            x_sc[warp] = mx * (1.f / 127.f);
        }
    }
    __syncthreads();
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)m * K);
        float inv = s_amax[m][0];
        int* dst = xs_q8 + m * row_words;
        for (int i = tid; i < n8; i += 128) {
            float f[8];
            unpack8_bf16(__ldg(&x4[i]), f);
            int q[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                q[j] = __float2int_rn(fminf(fmaxf(f[j] * inv, -127.f), 127.f)) & 0xFF;
            }
            dst[2 * i] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
            dst[2 * i + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
        }
    }
    __syncthreads();
    if (!active) {
        NVK_PDL_EPILOG();
        return;
    }
    const int4* xs_i4 = reinterpret_cast<const int4*>(xs_q8);
    float ag[TM], au[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) { ag[m] = 0.f; au[m] = 0.f; }
    int kb = lane;
    while (kb < kb_total) {
        int kbn = kb + 32;
        uint2 ng = (kbn < kb_total) ? __ldcs(&wg[kbn]) : make_uint2(0u, 0u);
        uint8_t nsg = (kbn < kb_total) ? __ldcs(&sg[kbn]) : 0u;
        uint2 nu = (kbn < kb_total) ? __ldcs(&wu[kbn]) : make_uint2(0u, 0u);
        uint8_t nsu = (kbn < kb_total) ? __ldcs(&su[kbn]) : 0u;
        unsigned g0, g1, g2, g3, u0, u1, u2, u3;
        q4exp_nibbles8_to_i8(rg.x, g0, g1);
        q4exp_nibbles8_to_i8(rg.y, g2, g3);
        q4exp_nibbles8_to_i8(ru.x, u0, u1);
        q4exp_nibbles8_to_i8(ru.y, u2, u3);
        float fg = q4exp_decode_ue4m3(sgc);
        float fu = q4exp_decode_ue4m3(suc);
        #pragma unroll
        for (int m = 0; m < TM; ++m) {
            int4 xv = xs_i4[m * kb_total + kb];
            int dg = __dp4a((int)g0, xv.x, 0);
            dg = __dp4a((int)g1, xv.y, dg);
            dg = __dp4a((int)g2, xv.z, dg);
            dg = __dp4a((int)g3, xv.w, dg);
            int du = __dp4a((int)u0, xv.x, 0);
            du = __dp4a((int)u1, xv.y, du);
            du = __dp4a((int)u2, xv.z, du);
            du = __dp4a((int)u3, xv.w, du);
            ag[m] += fg * (float)dg;
            au[m] += fu * (float)du;
        }
        kb = kbn;
        rg = ng; sgc = nsg; ru = nu; suc = nsu;
    }
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        ag[m] = warp_sum(ag[m]);
        au[m] = warp_sum(au[m]);
        if (lane == 0) {
            float gv = __bfloat162float(__float2bfloat16(ag[m] * (x_sc[m] * alpha_g)));
            float uv = __bfloat162float(__float2bfloat16(au[m] * (x_sc[m] * alpha_u)));
            float sv = gv / (1.f + expf(-gv));
            act[(size_t)m * N + n] = __float2bfloat16(sv * uv);
        }
    }
    NVK_PDL_EPILOG();
}

template <int TM>
static int q4exp_launch_nvfp4_gateup_silu(
    cudaStream_t s, const uint8_t* wq_g, const uint8_t* sc_g, float alpha_g,
    const uint8_t* wq_u, const uint8_t* sc_u, float alpha_u,
    const uint16_t* x, uint16_t* act, int N, int K
) {
    size_t smem = (size_t)TM * (size_t)K;
    if (smem > 48 * 1024) return -3;
    dim3 grid((unsigned)((N + 3) / 4));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(128), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_nvfp4_gateup_silu_kernel<TM>,
            wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, reinterpret_cast<__nv_bfloat16*>(act), N, K);
    } else {
        q4exp_nvfp4_gateup_silu_kernel<TM><<<grid, 128, smem, s>>>(
            wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, reinterpret_cast<__nv_bfloat16*>(act), N, K);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_nvfp4_gateup_silu(
    void* stream, const uint8_t* wq_g, const uint8_t* sc_g, float alpha_g,
    const uint8_t* wq_u, const uint8_t* sc_u, float alpha_u,
    const uint16_t* x, uint16_t* act, int m, int n, int k
) {
    if (m <= 0 || n <= 0 || k <= 0 || (k & 15) != 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: return q4exp_launch_nvfp4_gateup_silu<1>(s, wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, act, n, k);
        case 2: return q4exp_launch_nvfp4_gateup_silu<2>(s, wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, act, n, k);
        case 3: return q4exp_launch_nvfp4_gateup_silu<3>(s, wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, act, n, k);
        case 4: return q4exp_launch_nvfp4_gateup_silu<4>(s, wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, act, n, k);
        case 5: return q4exp_launch_nvfp4_gateup_silu<5>(s, wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, act, n, k);
        case 6: return q4exp_launch_nvfp4_gateup_silu<6>(s, wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, act, n, k);
        case 7: return q4exp_launch_nvfp4_gateup_silu<7>(s, wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, act, n, k);
        case 8: return q4exp_launch_nvfp4_gateup_silu<8>(s, wq_g, sc_g, alpha_g, wq_u, sc_u, alpha_u, x, act, n, k);
        default: return -1;
    }
}

extern "C" int nv_kernels_q4exp_nvfp4_down_tail(
    void* stream, const uint8_t* wq, const uint8_t* sc, float alpha,
    const uint16_t* act, const uint16_t* y_down, long long slot_stride,
    const float* topk_w, int topk, const uint16_t* sg, uint16_t* out,
    int m, int n, int k
) {
    if (m <= 0 || n <= 0 || k <= 0 || (k & 15) != 0 || topk <= 0 || topk > kQ4ExpTailMaxSlots) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: return q4exp_launch_nvfp4_down_tail<1>(s, wq, sc, alpha, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 2: return q4exp_launch_nvfp4_down_tail<2>(s, wq, sc, alpha, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 3: return q4exp_launch_nvfp4_down_tail<3>(s, wq, sc, alpha, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 4: return q4exp_launch_nvfp4_down_tail<4>(s, wq, sc, alpha, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 5: return q4exp_launch_nvfp4_down_tail<5>(s, wq, sc, alpha, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 6: return q4exp_launch_nvfp4_down_tail<6>(s, wq, sc, alpha, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 7: return q4exp_launch_nvfp4_down_tail<7>(s, wq, sc, alpha, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        case 8: return q4exp_launch_nvfp4_down_tail<8>(s, wq, sc, alpha, act, y_down, slot_stride, topk_w, topk, sg, out, n, k);
        default: return -1;
    }
}

extern "C" void nvk_q4exp_get_i8_scratch(int** buf, float** sc) {
    q4exp_ensure_i8_scratch();
    if (buf) *buf = g_q4exp_i8_buf;
    if (sc) *sc = g_q4exp_i8_sc;
}

extern "C" int nv_kernels_q4exp_requant_fp8_to_nvfp4(
    void* stream, const uint8_t* fp8_w, const float* fp8_rs,
    uint8_t* wq_out, uint8_t* sc_out, int n, int k, float stored_global
) {
    q4exp_ensure_i8_scratch();
    if (n <= 0 || k <= 0 || (k & 15) != 0 || stored_global <= 0.f) return -1;
    if (!fp8_w || !fp8_rs || !wq_out || !sc_out) return -2;
    size_t total = (size_t)n * (size_t)(k >> 4);
    size_t blocks = (total + 255) / 256;
    if (blocks > 65535) blocks = 65535;
    q4exp_requant_fp8_to_nvfp4_kernel<<<(unsigned)blocks, 256, 0, (cudaStream_t)stream>>>(
        fp8_w, fp8_rs, wq_out, sc_out, n, k, stored_global);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_nvfp4_gemv_mt(
    void* stream, const uint8_t* wq, const uint8_t* scales, const uint16_t* x, uint16_t* y,
    float alpha_over_2g, int m, int n, int k
) {
    if (m <= 0 || n <= 0 || k <= 0 || (k & 15) != 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: return q4exp_launch_nvfp4_gemv_mt<1>(s, wq, scales, x, y, alpha_over_2g, n, k);
        case 2: return q4exp_launch_nvfp4_gemv_mt<2>(s, wq, scales, x, y, alpha_over_2g, n, k);
        case 3: return q4exp_launch_nvfp4_gemv_mt<3>(s, wq, scales, x, y, alpha_over_2g, n, k);
        case 4: return q4exp_launch_nvfp4_gemv_mt<4>(s, wq, scales, x, y, alpha_over_2g, n, k);
        case 5: return q4exp_launch_nvfp4_gemv_mt<5>(s, wq, scales, x, y, alpha_over_2g, n, k);
        case 6: return q4exp_launch_nvfp4_gemv_mt<6>(s, wq, scales, x, y, alpha_over_2g, n, k);
        case 7: return q4exp_launch_nvfp4_gemv_mt<7>(s, wq, scales, x, y, alpha_over_2g, n, k);
        case 8: return q4exp_launch_nvfp4_gemv_mt<8>(s, wq, scales, x, y, alpha_over_2g, n, k);
        default: return -1;
    }
}

extern "C" int nv_kernels_q4exp_nvfp4_dual_gemv_mt(
    void* stream,
    const uint8_t* wq_a, const uint8_t* sc_a, float alpha_a,
    const uint8_t* wq_b, const uint8_t* sc_b, float alpha_b,
    const uint16_t* x, uint16_t* y_a, uint16_t* y_b,
    int na, int nb, int k, int m
) {
    if (m <= 0 || na + nb <= 0 || k <= 0 || (k & 15) != 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: return q4exp_launch_nvfp4_dual_gemv_mt<1>(s, wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x, y_a, y_b, na, nb, k);
        case 2: return q4exp_launch_nvfp4_dual_gemv_mt<2>(s, wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x, y_a, y_b, na, nb, k);
        case 3: return q4exp_launch_nvfp4_dual_gemv_mt<3>(s, wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x, y_a, y_b, na, nb, k);
        case 4: return q4exp_launch_nvfp4_dual_gemv_mt<4>(s, wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x, y_a, y_b, na, nb, k);
        case 5: return q4exp_launch_nvfp4_dual_gemv_mt<5>(s, wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x, y_a, y_b, na, nb, k);
        case 6: return q4exp_launch_nvfp4_dual_gemv_mt<6>(s, wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x, y_a, y_b, na, nb, k);
        case 7: return q4exp_launch_nvfp4_dual_gemv_mt<7>(s, wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x, y_a, y_b, na, nb, k);
        case 8: return q4exp_launch_nvfp4_dual_gemv_mt<8>(s, wq_a, sc_a, alpha_a, wq_b, sc_b, alpha_b, x, y_a, y_b, na, nb, k);
        default: return -1;
    }
}

extern "C" int nv_kernels_q4exp_nvfp4_qkv_idx_gemv_mt(
    void* stream,
    const uint8_t* wq, const uint8_t* scq, float aq,
    const uint8_t* wk, const uint8_t* sck, float ak,
    const uint8_t* wv, const uint8_t* scv, float av,
    const uint8_t* wi, const uint8_t* sci, float ai,
    const uint16_t* x, uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yi,
    int nq, int nk, int nv, int ni, int k, int m
) {
    if (m <= 0 || nq + nk + nv + ni <= 0 || k <= 0 || (k & 15) != 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: return q4exp_launch_nvfp4_qkv_idx_gemv_mt<1>(s, wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 2: return q4exp_launch_nvfp4_qkv_idx_gemv_mt<2>(s, wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 3: return q4exp_launch_nvfp4_qkv_idx_gemv_mt<3>(s, wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 4: return q4exp_launch_nvfp4_qkv_idx_gemv_mt<4>(s, wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 5: return q4exp_launch_nvfp4_qkv_idx_gemv_mt<5>(s, wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 6: return q4exp_launch_nvfp4_qkv_idx_gemv_mt<6>(s, wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 7: return q4exp_launch_nvfp4_qkv_idx_gemv_mt<7>(s, wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        case 8: return q4exp_launch_nvfp4_qkv_idx_gemv_mt<8>(s, wq, scq, aq, wk, sck, ak, wv, scv, av, wi, sci, ai, x, yq, yk, yv, yi, nq, nk, nv, ni, k);
        default: return -1;
    }
}

extern "C" int nv_kernels_q4exp_hc_down_nvfp4(
    void* stream, const uint16_t* xn, const uint8_t* wq, const uint8_t* sc, float alpha,
    const uint16_t* w_inject, float* lora_out, float* inject_out, int m, int hh, int nb, int rank
) {
    if (m <= 0) return 0;
    if ((hh & 15) != 0 || hh % nb != 0 || nb > 8 || rank <= 0) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: return q4exp_launch_hc_down_nvfp4<1>(s, xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        case 2: return q4exp_launch_hc_down_nvfp4<2>(s, xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        case 3: return q4exp_launch_hc_down_nvfp4<3>(s, xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        case 4: return q4exp_launch_hc_down_nvfp4<4>(s, xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        case 5: return q4exp_launch_hc_down_nvfp4<5>(s, xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        case 6: return q4exp_launch_hc_down_nvfp4<6>(s, xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        case 7: return q4exp_launch_hc_down_nvfp4<7>(s, xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        case 8: return q4exp_launch_hc_down_nvfp4<8>(s, xn, wq, sc, alpha, w_inject, lora_out, inject_out, hh, nb, rank);
        default: return -1;
    }
}

extern "C" int nv_kernels_q4exp_hc_up_mix_nvfp4(
    void* stream, const float* lora, const uint8_t* wq, const uint8_t* sc, float alpha,
    const uint16_t* xn, uint16_t* mixed, int m, int h, int nb, int rank
) {
    if (m <= 0) return 0;
    if (rank != 320 || nb != 4 || h <= 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: return q4exp_launch_hc_up_mix_nvfp4<1>(s, lora, wq, sc, alpha, xn, mixed, h);
        case 2: return q4exp_launch_hc_up_mix_nvfp4<2>(s, lora, wq, sc, alpha, xn, mixed, h);
        case 3: return q4exp_launch_hc_up_mix_nvfp4<3>(s, lora, wq, sc, alpha, xn, mixed, h);
        case 4: return q4exp_launch_hc_up_mix_nvfp4<4>(s, lora, wq, sc, alpha, xn, mixed, h);
        case 5: return q4exp_launch_hc_up_mix_nvfp4<5>(s, lora, wq, sc, alpha, xn, mixed, h);
        case 6: return q4exp_launch_hc_up_mix_nvfp4<6>(s, lora, wq, sc, alpha, xn, mixed, h);
        case 7: return q4exp_launch_hc_up_mix_nvfp4<7>(s, lora, wq, sc, alpha, xn, mixed, h);
        case 8: return q4exp_launch_hc_up_mix_nvfp4<8>(s, lora, wq, sc, alpha, xn, mixed, h);
        default: return -1;
    }
}

__device__ __forceinline__ float q4exp_sign_word_lut_dot(
    uint32_t w, const __nv_bfloat16* lut_m, int r0, int col
) {
    float s = __bfloat162float(lut_m[((r0 + 0) * 20 + col) * 16 + (w & 0xFu)]);
    s += __bfloat162float(lut_m[((r0 + 1) * 20 + col) * 16 + ((w >> 4) & 0xFu)]);
    s += __bfloat162float(lut_m[((r0 + 2) * 20 + col) * 16 + ((w >> 8) & 0xFu)]);
    s += __bfloat162float(lut_m[((r0 + 3) * 20 + col) * 16 + ((w >> 12) & 0xFu)]);
    s += __bfloat162float(lut_m[((r0 + 4) * 20 + col) * 16 + ((w >> 16) & 0xFu)]);
    s += __bfloat162float(lut_m[((r0 + 5) * 20 + col) * 16 + ((w >> 20) & 0xFu)]);
    s += __bfloat162float(lut_m[((r0 + 6) * 20 + col) * 16 + ((w >> 24) & 0xFu)]);
    s += __bfloat162float(lut_m[((r0 + 7) * 20 + col) * 16 + ((w >> 28) & 0xFu)]);
    return s;
}

struct alignas(8) Q4ExpBf16x4 {
    __nv_bfloat162 lo;
    __nv_bfloat162 hi;
};

template <int TM>
__device__ __forceinline__ void q4exp_sign_word_lut_u2_dot(
    uint32_t w, const Q4ExpBf16x4* lut, int g0, float* acc
) {
    Q4ExpBf16x4 a0 = lut[(g0 + 0) * 16 + (w & 0xFu)];
    Q4ExpBf16x4 a1 = lut[(g0 + 1) * 16 + ((w >> 4) & 0xFu)];
    Q4ExpBf16x4 a2 = lut[(g0 + 2) * 16 + ((w >> 8) & 0xFu)];
    Q4ExpBf16x4 a3 = lut[(g0 + 3) * 16 + ((w >> 12) & 0xFu)];
    Q4ExpBf16x4 a4 = lut[(g0 + 4) * 16 + ((w >> 16) & 0xFu)];
    Q4ExpBf16x4 a5 = lut[(g0 + 5) * 16 + ((w >> 20) & 0xFu)];
    Q4ExpBf16x4 a6 = lut[(g0 + 6) * 16 + ((w >> 24) & 0xFu)];
    Q4ExpBf16x4 a7 = lut[(g0 + 7) * 16 + ((w >> 28) & 0xFu)];
    __nv_bfloat162 s01 = __hadd2(
        __hadd2(__hadd2(a0.lo, a1.lo), __hadd2(a2.lo, a3.lo)),
        __hadd2(__hadd2(a4.lo, a5.lo), __hadd2(a6.lo, a7.lo)));
    __nv_bfloat162 s23 = __hadd2(
        __hadd2(__hadd2(a0.hi, a1.hi), __hadd2(a2.hi, a3.hi)),
        __hadd2(__hadd2(a4.hi, a5.hi), __hadd2(a6.hi, a7.hi)));
    float2 f01 = __bfloat1622float2(s01);
    float2 f23 = __bfloat1622float2(s23);
    acc[0] += f01.x;
    acc[1] += f01.y;
    acc[2] += f23.x;
    if constexpr (TM == 4) acc[3] += f23.y;
}

template <int TM>
__global__ void __launch_bounds__(512) q4exp_lm_head_sign_screen_m34_k2560_kernel(
    const uint32_t* __restrict__ w_sign,
    const float* __restrict__ row_scale,
    const uint16_t* __restrict__ x,
    uint32_t* __restrict__ cand_ids,
    int N
) {
    extern __shared__ Q4ExpBf16x4 lut_u2[];
    __shared__ unsigned long long grp_best[TM][64];
    int tid = threadIdx.x;
    NVK_PDL_PROLOG();
    for (int g = tid; g < 640; g += 512) {
        float fx[4][4];
        #pragma unroll
        for (int m = 0; m < 4; ++m) {
            if (m < TM) {
                uint2 x2 = __ldg(reinterpret_cast<const uint2*>(x + (size_t)m * 2560 + g * 4));
                uint16_t b0 = (uint16_t)(x2.x & 0xFFFFu);
                uint16_t b1 = (uint16_t)(x2.x >> 16);
                uint16_t b2 = (uint16_t)(x2.y & 0xFFFFu);
                uint16_t b3 = (uint16_t)(x2.y >> 16);
                fx[m][0] = ld_bf16(&b0);
                fx[m][1] = ld_bf16(&b1);
                fx[m][2] = ld_bf16(&b2);
                fx[m][3] = ld_bf16(&b3);
            } else {
                fx[m][0] = 0.f; fx[m][1] = 0.f; fx[m][2] = 0.f; fx[m][3] = 0.f;
            }
        }
        Q4ExpBf16x4* dst = lut_u2 + g * 16;
        #pragma unroll
        for (int mask = 0; mask < 16; ++mask) {
            float s0 = ((mask & 1) ? fx[0][0] : -fx[0][0]) + ((mask & 2) ? fx[0][1] : -fx[0][1])
                     + ((mask & 4) ? fx[0][2] : -fx[0][2]) + ((mask & 8) ? fx[0][3] : -fx[0][3]);
            float s1 = ((mask & 1) ? fx[1][0] : -fx[1][0]) + ((mask & 2) ? fx[1][1] : -fx[1][1])
                     + ((mask & 4) ? fx[1][2] : -fx[1][2]) + ((mask & 8) ? fx[1][3] : -fx[1][3]);
            float s2 = ((mask & 1) ? fx[2][0] : -fx[2][0]) + ((mask & 2) ? fx[2][1] : -fx[2][1])
                     + ((mask & 4) ? fx[2][2] : -fx[2][2]) + ((mask & 8) ? fx[2][3] : -fx[2][3]);
            float s3 = 0.f;
            if constexpr (TM == 4) {
                s3 = ((mask & 1) ? fx[3][0] : -fx[3][0]) + ((mask & 2) ? fx[3][1] : -fx[3][1])
                   + ((mask & 4) ? fx[3][2] : -fx[3][2]) + ((mask & 8) ? fx[3][3] : -fx[3][3]);
            }
            dst[mask].lo = __halves2bfloat162(__float2bfloat16(s0), __float2bfloat16(s1));
            dst[mask].hi = __halves2bfloat162(__float2bfloat16(s2), __float2bfloat16(s3));
        }
    }
    __syncthreads();
    int sgrp = tid >> 3;
    int sub = tid & 7;
    int num_tiles = (N + 63) >> 6;
    unsigned long long my_best[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) my_best[m] = 0ull;
    int tile = (int)blockIdx.x;
    int n_cur = tile * 64 + sgrp;
    uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
    uint4 v1 = make_uint4(0u, 0u, 0u, 0u);
    uint2 v2 = make_uint2(0u, 0u);
    float rs_cur = 0.f;
    if (tile < num_tiles && n_cur < N) {
        const uint32_t* row = w_sign + (size_t)n_cur * 80;
        const uint4* w4 = reinterpret_cast<const uint4*>(row);
        const uint2* w2 = reinterpret_cast<const uint2*>(row);
        v0 = __ldcs(&w4[sub]);
        v1 = __ldcs(&w4[8 + sub]);
        v2 = __ldcs(&w2[32 + sub]);
        if (sub == 0) rs_cur = __ldg(&row_scale[n_cur]);
    }
    int g0 = sub * 32;
    int g1 = 256 + sub * 32;
    int g2 = 512 + sub * 16;
    while (tile < num_tiles) {
        int tile_n = tile + (int)gridDim.x;
        int n_nxt = tile_n * 64 + sgrp;
        uint4 nv0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 nv1 = make_uint4(0u, 0u, 0u, 0u);
        uint2 nv2 = make_uint2(0u, 0u);
        float rs_nxt = 0.f;
        if (tile_n < num_tiles && n_nxt < N) {
            const uint32_t* row = w_sign + (size_t)n_nxt * 80;
            const uint4* w4 = reinterpret_cast<const uint4*>(row);
            const uint2* w2 = reinterpret_cast<const uint2*>(row);
            nv0 = __ldcs(&w4[sub]);
            nv1 = __ldcs(&w4[8 + sub]);
            nv2 = __ldcs(&w2[32 + sub]);
            if (sub == 0) rs_nxt = __ldg(&row_scale[n_nxt]);
        }
        if (n_cur < N) {
            float acc[TM];
            #pragma unroll
            for (int m = 0; m < TM; ++m) acc[m] = 0.f;
            q4exp_sign_word_lut_u2_dot<TM>(v0.x, lut_u2, g0, acc);
            q4exp_sign_word_lut_u2_dot<TM>(v0.y, lut_u2, g0 + 8, acc);
            q4exp_sign_word_lut_u2_dot<TM>(v0.z, lut_u2, g0 + 16, acc);
            q4exp_sign_word_lut_u2_dot<TM>(v0.w, lut_u2, g0 + 24, acc);
            q4exp_sign_word_lut_u2_dot<TM>(v1.x, lut_u2, g1, acc);
            q4exp_sign_word_lut_u2_dot<TM>(v1.y, lut_u2, g1 + 8, acc);
            q4exp_sign_word_lut_u2_dot<TM>(v1.z, lut_u2, g1 + 16, acc);
            q4exp_sign_word_lut_u2_dot<TM>(v1.w, lut_u2, g1 + 24, acc);
            q4exp_sign_word_lut_u2_dot<TM>(v2.x, lut_u2, g2, acc);
            q4exp_sign_word_lut_u2_dot<TM>(v2.y, lut_u2, g2 + 8, acc);
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                float a = acc[m];
                #pragma unroll
                for (int o = 4; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o, 8);
                if (sub == 0) {
                    unsigned long long p = q4exp_pack_argmax_key(a * rs_cur, (unsigned)n_cur);
                    if (p > my_best[m]) my_best[m] = p;
                }
            }
        }
        tile = tile_n;
        n_cur = n_nxt;
        v0 = nv0; v1 = nv1; v2 = nv2; rs_cur = rs_nxt;
    }
    if (sub == 0) {
        #pragma unroll
        for (int m = 0; m < TM; ++m) grp_best[m][sgrp] = my_best[m];
    }
    __syncthreads();
    int num_cands = (int)gridDim.x * 4;
    if (tid < TM * 4) {
        int m = tid >> 2;
        int c = tid & 3;
        unsigned long long b = 0ull;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            unsigned long long v = grp_best[m][c * 16 + i];
            if (v > b) b = v;
        }
        uint32_t idx = (b == 0ull) ? 0xFFFFFFFFu : (((unsigned)(b & 0xFFFFFFFFull)) ^ 0xFFFFFFFFu);
        cand_ids[(size_t)m * num_cands + (size_t)blockIdx.x * 4 + c] = idx;
    }
    NVK_PDL_EPILOG();
}

template <int TM>
__global__ void __launch_bounds__(256) q4exp_lm_head_sign_screen_mt_k2560_kernel(
    const uint32_t* __restrict__ w_sign,
    const float* __restrict__ row_scale,
    const uint16_t* __restrict__ x,
    uint32_t* __restrict__ cand_ids,
    int N
) {
    extern __shared__ __nv_bfloat16 lut_all[];
    __shared__ unsigned long long grp_best[TM][32];
    int tid = threadIdx.x;
    NVK_PDL_PROLOG();
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        __nv_bfloat16* lut_m = lut_all + m * 640 * 16;
        const uint16_t* xm = x + (size_t)m * 2560;
        for (int g = tid; g < 640; g += 256) {
            uint2 x2 = __ldg(reinterpret_cast<const uint2*>(xm + g * 4));
            uint16_t b0 = (uint16_t)(x2.x & 0xFFFFu);
            uint16_t b1 = (uint16_t)(x2.x >> 16);
            uint16_t b2 = (uint16_t)(x2.y & 0xFFFFu);
            uint16_t b3 = (uint16_t)(x2.y >> 16);
            float f0 = ld_bf16(&b0);
            float f1 = ld_bf16(&b1);
            float f2 = ld_bf16(&b2);
            float f3 = ld_bf16(&b3);
            int g_smem = (g & 31) * 20 + (g >> 5);
            __nv_bfloat16* dst = lut_m + g_smem * 16;
            #pragma unroll
            for (int mask = 0; mask < 16; ++mask) {
                float s = ((mask & 1) ? f0 : -f0) + ((mask & 2) ? f1 : -f1)
                        + ((mask & 4) ? f2 : -f2) + ((mask & 8) ? f3 : -f3);
                dst[mask] = __float2bfloat16(s);
            }
        }
    }
    __syncthreads();
    int sgrp = tid >> 3;
    int sub = tid & 7;
    int num_tiles = (N + 31) >> 5;
    unsigned long long my_best[TM];
    #pragma unroll
    for (int m = 0; m < TM; ++m) my_best[m] = 0ull;
    int tile = (int)blockIdx.x;
    int n_cur = tile * 32 + sgrp;
    uint4 v0 = make_uint4(0u, 0u, 0u, 0u);
    uint4 v1 = make_uint4(0u, 0u, 0u, 0u);
    uint2 v2 = make_uint2(0u, 0u);
    float rs_cur = 0.f;
    if (tile < num_tiles && n_cur < N) {
        const uint32_t* row = w_sign + (size_t)n_cur * 80;
        const uint4* w4 = reinterpret_cast<const uint4*>(row);
        const uint2* w2 = reinterpret_cast<const uint2*>(row);
        v0 = __ldcs(&w4[sub]);
        v1 = __ldcs(&w4[8 + sub]);
        v2 = __ldcs(&w2[32 + sub]);
        if (sub == 0) rs_cur = __ldg(&row_scale[n_cur]);
    }
    int c0 = sub;
    int c1 = 8 + sub;
    int c2 = 16 + (sub >> 1);
    int r2 = (sub & 1) * 16;
    while (tile < num_tiles) {
        int tile_n = tile + (int)gridDim.x;
        int n_nxt = tile_n * 32 + sgrp;
        uint4 nv0 = make_uint4(0u, 0u, 0u, 0u);
        uint4 nv1 = make_uint4(0u, 0u, 0u, 0u);
        uint2 nv2 = make_uint2(0u, 0u);
        float rs_nxt = 0.f;
        if (tile_n < num_tiles && n_nxt < N) {
            const uint32_t* row = w_sign + (size_t)n_nxt * 80;
            const uint4* w4 = reinterpret_cast<const uint4*>(row);
            const uint2* w2 = reinterpret_cast<const uint2*>(row);
            nv0 = __ldcs(&w4[sub]);
            nv1 = __ldcs(&w4[8 + sub]);
            nv2 = __ldcs(&w2[32 + sub]);
            if (sub == 0) rs_nxt = __ldg(&row_scale[n_nxt]);
        }
        if (n_cur < N) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) {
                const __nv_bfloat16* lut_m = lut_all + m * 640 * 16;
                float acc = q4exp_sign_word_lut_dot(v0.x, lut_m, 0, c0)
                          + q4exp_sign_word_lut_dot(v0.y, lut_m, 8, c0)
                          + q4exp_sign_word_lut_dot(v0.z, lut_m, 16, c0)
                          + q4exp_sign_word_lut_dot(v0.w, lut_m, 24, c0)
                          + q4exp_sign_word_lut_dot(v1.x, lut_m, 0, c1)
                          + q4exp_sign_word_lut_dot(v1.y, lut_m, 8, c1)
                          + q4exp_sign_word_lut_dot(v1.z, lut_m, 16, c1)
                          + q4exp_sign_word_lut_dot(v1.w, lut_m, 24, c1)
                          + q4exp_sign_word_lut_dot(v2.x, lut_m, r2, c2)
                          + q4exp_sign_word_lut_dot(v2.y, lut_m, r2 + 8, c2);
                #pragma unroll
                for (int o = 4; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o, 8);
                if (sub == 0) {
                    unsigned long long p = q4exp_pack_argmax_key(acc * rs_cur, (unsigned)n_cur);
                    if (p > my_best[m]) my_best[m] = p;
                }
            }
        }
        tile = tile_n;
        n_cur = n_nxt;
        v0 = nv0; v1 = nv1; v2 = nv2; rs_cur = rs_nxt;
    }
    if (sub == 0) {
        #pragma unroll
        for (int m = 0; m < TM; ++m) grp_best[m][sgrp] = my_best[m];
    }
    __syncthreads();
    constexpr int kCandsPerCta = 4;
    constexpr int kGrpStride = 32 / kCandsPerCta;
    int num_cands = (int)gridDim.x * kCandsPerCta;
    if (tid < TM * kCandsPerCta) {
        int m = tid / kCandsPerCta;
        int c = tid - m * kCandsPerCta;
        unsigned long long b = 0ull;
        #pragma unroll
        for (int i = 0; i < kGrpStride; ++i) {
            unsigned long long v = grp_best[m][c * kGrpStride + i];
            if (v > b) b = v;
        }
        uint32_t idx = (b == 0ull) ? 0xFFFFFFFFu : (((unsigned)(b & 0xFFFFFFFFull)) ^ 0xFFFFFFFFu);
        cand_ids[(size_t)m * num_cands + (size_t)blockIdx.x * kCandsPerCta + c] = idx;
    }
    NVK_PDL_EPILOG();
}

__global__ void __launch_bounds__(256) q4exp_lm_head_fp8_rescore_argmax_mt_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ rs,
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ cand_ids,
    unsigned long long* __restrict__ packed_best,
    int num_cands,
    int N,
    int K
) {
    __shared__ float xs_f[2560];
    __shared__ unsigned long long warp_best[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int row = (int)blockIdx.y;
    NVK_PDL_PROLOG();
    const uint4* x4 = reinterpret_cast<const uint4*>(x + (size_t)row * K);
    int n8 = K >> 3;
    for (int i = tid; i < n8; i += 256) {
        float f[8];
        unpack8_bf16(__ldg(&x4[i]), f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) xs_f[i * 8 + j] = f[j];
    }
    __syncthreads();
    int ci = (int)blockIdx.x * 8 + warp;
    unsigned long long my_best = 0ull;
    if (ci < num_cands) {
        uint32_t n0 = __ldg(&cand_ids[(size_t)row * num_cands + ci]);
        if (n0 < (uint32_t)N) {
            const int4* wa = reinterpret_cast<const int4*>(wq + (size_t)n0 * K);
            int Kv = K >> 4;
            int v = lane;
            int4 ra = (v < Kv) ? __ldcs(&wa[v]) : make_int4(0, 0, 0, 0);
            float acc = 0.f;
            while (v < Kv) {
                int vn = v + 32;
                int4 na = (vn < Kv) ? __ldcs(&wa[vn]) : make_int4(0, 0, 0, 0);
                float wf[16];
                q4exp_deq_e4m3x16(ra, wf);
                const float* xp = xs_f + v * 16;
                #pragma unroll
                for (int j = 0; j < 16; ++j) acc += wf[j] * xp[j];
                v = vn;
                ra = na;
            }
            acc = warp_sum(acc);
            if (lane == 0) my_best = q4exp_pack_argmax_key(acc * __ldg(&rs[n0]), n0);
        }
    }
    if (lane == 0) warp_best[warp] = my_best;
    __syncthreads();
    if (tid == 0) {
        unsigned long long blk = 0ull;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (warp_best[w] > blk) blk = warp_best[w];
        }
        if (blk > packed_best[row]) atomicMax(&packed_best[row], blk);
    }
    NVK_PDL_EPILOG();
}

__global__ void __launch_bounds__(256) q4exp_lm_head_nvfp4_rescore_argmax_m1_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ scales,
    const uint16_t* __restrict__ x,
    const uint32_t* __restrict__ cand_ids,
    unsigned long long* __restrict__ packed_best,
    int num_cands,
    int N,
    int K
) {
    __shared__ alignas(16) int xs_q8_words[640];
    __shared__ float s_amax[8];
    __shared__ unsigned long long warp_best[8];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int kb_total = K >> 4;
    NVK_PDL_PROLOG();
    int n8 = K >> 3;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    float local_max = 0.f;
    for (int i = tid; i < n8; i += 256) {
        float f[8];
        unpack8_bf16(__ldg(&x4[i]), f);
        #pragma unroll
        for (int j = 0; j < 8; ++j) local_max = fmaxf(local_max, fabsf(f[j]));
    }
    local_max = warp_max(local_max);
    if (lane == 0) s_amax[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float m = (lane < 8) ? s_amax[lane] : 0.f;
        m = warp_max(m);
        if (lane == 0) s_amax[0] = (m > 0.f) ? (127.f / m) : 0.f;
    }
    __syncthreads();
    float inv = s_amax[0];
    for (int i = tid; i < n8; i += 256) {
        float f[8];
        unpack8_bf16(__ldg(&x4[i]), f);
        int q[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            int qi = __float2int_rn(fminf(fmaxf(f[j] * inv, -127.f), 127.f));
            q[j] = qi & 0xFF;
        }
        xs_q8_words[2 * i] = q[0] | (q[1] << 8) | (q[2] << 16) | (q[3] << 24);
        xs_q8_words[2 * i + 1] = q[4] | (q[5] << 8) | (q[6] << 16) | (q[7] << 24);
    }
    __syncthreads();
    int ci = (int)blockIdx.x * 8 + warp;
    unsigned long long my_best = 0ull;
    if (ci < num_cands) {
        uint32_t n0 = __ldg(&cand_ids[ci]);
        if (n0 < (uint32_t)N) {
            const int4* xs_i4 = reinterpret_cast<const int4*>(xs_q8_words);
            const uint2* wa = reinterpret_cast<const uint2*>(wq + (size_t)n0 * (K >> 1));
            const uint8_t* sa_ptr = scales + (size_t)n0 * kb_total;
            float acc0 = 0.f;
            for (int kb = lane; kb < kb_total; kb += 32) {
                uint2 ra = __ldcs(&wa[kb]);
                uint8_t sca = __ldcs(&sa_ptr[kb]);
                int4 xv = xs_i4[kb];
                unsigned a0, a1, a2, a3;
                q4exp_nibbles8_to_i8(ra.x, a0, a1);
                q4exp_nibbles8_to_i8(ra.y, a2, a3);
                int d0 = __dp4a((int)a0, xv.x, 0);
                d0 = __dp4a((int)a1, xv.y, d0);
                d0 = __dp4a((int)a2, xv.z, d0);
                d0 = __dp4a((int)a3, xv.w, d0);
                acc0 += q4exp_decode_ue4m3(sca) * (float)d0;
            }
            acc0 = warp_sum(acc0);
            if (lane == 0) my_best = q4exp_pack_argmax_key(acc0, n0);
        }
    }
    if (lane == 0) warp_best[warp] = my_best;
    __syncthreads();
    if (tid == 0) {
        unsigned long long blk = 0ull;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            if (warp_best[w] > blk) blk = warp_best[w];
        }
        if (blk > *packed_best) atomicMax(packed_best, blk);
    }
    NVK_PDL_EPILOG();
}

extern "C" int nv_kernels_q4exp_lm_head_nvfp4_argmax_m1(
    void* stream,
    const uint8_t* wq,
    const uint8_t* scales,
    const uint16_t* x,
    uint64_t* packed_best,
    uint32_t* out_id,
    int n,
    int k
) {
    if (n <= 0 || k <= 0 || (k & 15) != 0) return -1;
    if (wq == nullptr || scales == nullptr || x == nullptr || packed_best == nullptr || out_id == nullptr) return -2;
    size_t smem = (size_t)k;
    if (smem > 48 * 1024) return -3;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaError_t me = cudaMemsetAsync(packed_best, 0, sizeof(uint64_t), s);
    if (me != cudaSuccess) return (int)me;
    int num_tiles = (n + 15) / 16;
    dim3 grid((unsigned)(num_tiles < 376 ? num_tiles : 376));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
        cudaLaunchKernelEx(&cfg, q4exp_lm_head_nvfp4_argmax_m1_kernel,
            wq, scales, x, reinterpret_cast<unsigned long long*>(packed_best), n, k);
        NVK_PDL_ATTR(cfg2, dim3(1), dim3(1), 0, s);
        cudaLaunchKernelEx(&cfg2, q4exp_unpack_packed_argmax_kernel,
            reinterpret_cast<const unsigned long long*>(packed_best), out_id);
    } else {
        q4exp_lm_head_nvfp4_argmax_m1_kernel<<<grid, 256, smem, s>>>(
            wq, scales, x, reinterpret_cast<unsigned long long*>(packed_best), n, k);
        q4exp_unpack_packed_argmax_kernel<<<1, 1, 0, s>>>(
            reinterpret_cast<const unsigned long long*>(packed_best), out_id);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_lm_head_sign_screen_argmax_m1(
    void* stream,
    const uint32_t* w_sign,
    const float* row_scale,
    const uint8_t* wq,
    const uint8_t* scales,
    const uint16_t* x,
    uint32_t* cand_ids,
    uint64_t* packed_best,
    uint32_t* out_id,
    int n,
    int k
) {
    if (n <= 0 || k != 2560) return -1;
    if (!w_sign || !row_scale || !wq || !scales || !x || !cand_ids || !packed_best || !out_id) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    cudaError_t me = cudaMemsetAsync(packed_best, 0, sizeof(uint64_t), s);
    if (me != cudaSuccess) return (int)me;
    constexpr int kScreenCtas = 376;
    constexpr int kNumCands = kScreenCtas * 4;
    constexpr size_t kLutSmem = 640 * 16 * sizeof(__nv_bfloat16);
    dim3 g_scr(kScreenCtas);
    dim3 g_res((kNumCands + 7) / 8);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(c1, g_scr, dim3(256), kLutSmem, s);
        cudaLaunchKernelEx(&c1, q4exp_lm_head_sign_screen_mt_k2560_kernel<1>, w_sign, row_scale, x, cand_ids, n);
        NVK_PDL_ATTR(c2, g_res, dim3(256), 0, s);
        cudaLaunchKernelEx(&c2, q4exp_lm_head_nvfp4_rescore_argmax_m1_kernel,
            wq, scales, x, cand_ids, reinterpret_cast<unsigned long long*>(packed_best), kNumCands, n, k);
        NVK_PDL_ATTR(c3, dim3(1), dim3(1), 0, s);
        cudaLaunchKernelEx(&c3, q4exp_unpack_packed_argmax_kernel,
            reinterpret_cast<const unsigned long long*>(packed_best), out_id);
    } else {
        q4exp_lm_head_sign_screen_mt_k2560_kernel<1><<<g_scr, 256, kLutSmem, s>>>(w_sign, row_scale, x, cand_ids, n);
        q4exp_lm_head_nvfp4_rescore_argmax_m1_kernel<<<g_res, 256, 0, s>>>(
            wq, scales, x, cand_ids, reinterpret_cast<unsigned long long*>(packed_best), kNumCands, n, k);
        q4exp_unpack_packed_argmax_kernel<<<1, 1, 0, s>>>(
            reinterpret_cast<const unsigned long long*>(packed_best), out_id);
    }
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_lm_head_sign_screen_fp8_mt(
    cudaStream_t s, const uint32_t* w_sign, const float* row_scale,
    const uint8_t* wq, const float* rs, const uint16_t* x,
    uint32_t* cand_ids, uint64_t* packed_best, uint32_t* out_ids, int n, int k
) {
    cudaError_t me = cudaMemsetAsync(packed_best, 0, (size_t)TM * sizeof(uint64_t), s);
    if (me != cudaSuccess) return (int)me;
    constexpr int kNumCands = 376 * 4;
    dim3 g_res((unsigned)((kNumCands + 7) / 8), (unsigned)TM);
    if constexpr (TM >= 3) {
        constexpr int kNumCands34 = 188 * 4;
        dim3 g_res34((unsigned)((kNumCands34 + 7) / 8), (unsigned)TM);
        constexpr size_t kLutSmem = 640 * 16 * sizeof(Q4ExpBf16x4);
        if (kLutSmem + (size_t)TM * 64 * sizeof(unsigned long long) > (size_t)nvk_max_dynamic_smem_optin()) return -3;
        static DynamicSmemOptin s_optin;
        int orc = raise_dynamic_smem_optin_never_lowering_it(s_optin, (const void*)q4exp_lm_head_sign_screen_m34_k2560_kernel<TM>, kLutSmem);
        if (orc != 0) return orc;
        dim3 g_scr(188);
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(c1, g_scr, dim3(512), kLutSmem, s);
            cudaLaunchKernelEx(&c1, q4exp_lm_head_sign_screen_m34_k2560_kernel<TM>, w_sign, row_scale, x, cand_ids, n);
            NVK_PDL_ATTR(c2, g_res34, dim3(256), 0, s);
            cudaLaunchKernelEx(&c2, q4exp_lm_head_fp8_rescore_argmax_mt_kernel,
                wq, rs, x, cand_ids, reinterpret_cast<unsigned long long*>(packed_best), kNumCands34, n, k);
            NVK_PDL_ATTR(c3, dim3(1), dim3((unsigned)TM), 0, s);
            cudaLaunchKernelEx(&c3, q4exp_unpack_packed_argmax_mt_kernel,
                reinterpret_cast<const unsigned long long*>(packed_best), out_ids, TM);
        } else {
            q4exp_lm_head_sign_screen_m34_k2560_kernel<TM><<<g_scr, 512, kLutSmem, s>>>(w_sign, row_scale, x, cand_ids, n);
            q4exp_lm_head_fp8_rescore_argmax_mt_kernel<<<g_res34, 256, 0, s>>>(
                wq, rs, x, cand_ids, reinterpret_cast<unsigned long long*>(packed_best), kNumCands34, n, k);
            q4exp_unpack_packed_argmax_mt_kernel<<<1, TM, 0, s>>>(
                reinterpret_cast<const unsigned long long*>(packed_best), out_ids, TM);
        }
        return (int)cudaGetLastError();
    } else {
        constexpr int kNumCands12 = 376 * 4;
        dim3 g_res12((unsigned)((kNumCands12 + 7) / 8), (unsigned)TM);
        constexpr size_t kLutSmem = (size_t)TM * 640 * 16 * sizeof(__nv_bfloat16);
        if (kLutSmem > (size_t)nvk_max_dynamic_smem_optin()) return -3;
        static DynamicSmemOptin s_optin;
        int orc = raise_dynamic_smem_optin_never_lowering_it(s_optin, (const void*)q4exp_lm_head_sign_screen_mt_k2560_kernel<TM>, kLutSmem);
        if (orc != 0) return orc;
        dim3 g_scr(376);
        if (nvk_pdl_enabled()) {
            NVK_PDL_ATTR(c1, g_scr, dim3(256), kLutSmem, s);
            cudaLaunchKernelEx(&c1, q4exp_lm_head_sign_screen_mt_k2560_kernel<TM>, w_sign, row_scale, x, cand_ids, n);
            NVK_PDL_ATTR(c2, g_res12, dim3(256), 0, s);
            cudaLaunchKernelEx(&c2, q4exp_lm_head_fp8_rescore_argmax_mt_kernel,
                wq, rs, x, cand_ids, reinterpret_cast<unsigned long long*>(packed_best), kNumCands12, n, k);
            NVK_PDL_ATTR(c3, dim3(1), dim3((unsigned)TM), 0, s);
            cudaLaunchKernelEx(&c3, q4exp_unpack_packed_argmax_mt_kernel,
                reinterpret_cast<const unsigned long long*>(packed_best), out_ids, TM);
        } else {
            q4exp_lm_head_sign_screen_mt_k2560_kernel<TM><<<g_scr, 256, kLutSmem, s>>>(w_sign, row_scale, x, cand_ids, n);
            q4exp_lm_head_fp8_rescore_argmax_mt_kernel<<<g_res12, 256, 0, s>>>(
                wq, rs, x, cand_ids, reinterpret_cast<unsigned long long*>(packed_best), kNumCands12, n, k);
            q4exp_unpack_packed_argmax_mt_kernel<<<1, TM, 0, s>>>(
                reinterpret_cast<const unsigned long long*>(packed_best), out_ids, TM);
        }
        return (int)cudaGetLastError();
    }
}

extern "C" int nv_kernels_q4exp_lm_head_sign_screen_fp8_argmax_mt(
    void* stream,
    const uint32_t* w_sign,
    const float* row_scale,
    const uint8_t* wq,
    const float* rs,
    const uint16_t* x,
    uint32_t* cand_ids,
    uint64_t* packed_best,
    uint32_t* out_ids,
    int m,
    int n,
    int k
) {
    if (m <= 0 || m > 4 || n <= 0 || k != 2560) return -3;
    if (!w_sign || !row_scale || !wq || !rs || !x || !cand_ids || !packed_best || !out_ids) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    switch (m) {
        case 1: return q4exp_launch_lm_head_sign_screen_fp8_mt<1>(s, w_sign, row_scale, wq, rs, x, cand_ids, packed_best, out_ids, n, k);
        case 2: return q4exp_launch_lm_head_sign_screen_fp8_mt<2>(s, w_sign, row_scale, wq, rs, x, cand_ids, packed_best, out_ids, n, k);
        case 3: return q4exp_launch_lm_head_sign_screen_fp8_mt<3>(s, w_sign, row_scale, wq, rs, x, cand_ids, packed_best, out_ids, n, k);
        case 4: return q4exp_launch_lm_head_sign_screen_fp8_mt<4>(s, w_sign, row_scale, wq, rs, x, cand_ids, packed_best, out_ids, n, k);
        default: return -3;
    }
}
