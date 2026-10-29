#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>
#include "nv_kernels.h"
#include "nvk_smem_optin.cuh"

namespace {

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
    int t = blockIdx.x;
    int h = threadIdx.x;
    int n_heads = ctx * heads_per_ngram;
    if (t >= T || h >= n_heads) return;
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
}

__global__ void q4exp_ple_gather_kernel(
    const uint8_t* __restrict__ table, const int64_t* __restrict__ ids, float scale,
    int heads, int hd, uint16_t* __restrict__ out
) {
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
    const uint16_t* __restrict__ raw_k, int b0, const float* __restrict__ k_norm_w,
    const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    uint16_t* __restrict__ out, int ratio, int hd, int rot, float eps
) {
    extern __shared__ float smem[];
    float* v = smem;
    __shared__ float red[8];
    int b = b0 + blockIdx.x;
    int d = threadIdx.x;
    float acc = 0.f;
    for (int r = 0; r < ratio; ++r) acc += ld_bf16(raw_k + ((size_t)b * ratio + r) * hd + d);
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

__global__ void q4exp_qsa_scores_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ cmp_k,
    const int32_t* __restrict__ nb_per_row, int nh, int hd, int nb_stride, float scale,
    float* __restrict__ scores
) {
    extern __shared__ float smem[];
    float* qs = smem;
    int m = blockIdx.y;
    int nb = nb_per_row[m];
    for (int i = threadIdx.x; i < nh * hd; i += blockDim.x) qs[i] = ld_bf16(q + (size_t)m * nh * hd + i);
    __syncthreads();
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nb_stride) return;
    if (b >= nb) {
        scores[(size_t)m * nb_stride + b] = -INFINITY;
        return;
    }
    const uint16_t* kr = cmp_k + (size_t)b * hd;
    float acc[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) acc[h] = 0.f;
    for (int i = 0; i < hd; i += 8) {
        uint4 kv = *reinterpret_cast<const uint4*>(kr + i);
        float kf[8];
        unpack8_bf16(kv, kf);
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            if (h < nh) {
                const float* qh = qs + h * hd + i;
#pragma unroll
                for (int j = 0; j < 8; ++j) acc[h] += qh[j] * kf[j];
            }
        }
    }
    float s = 0.f;
#pragma unroll
    for (int h = 0; h < 4; ++h) if (h < nh) s += fmaxf(acc[h], 0.f);
    scores[(size_t)m * nb_stride + b] = s * scale;
}

template <int R, int KB>
__global__ void q4exp_qsa_scores_rows_kernel(const uint16_t* __restrict__ q, const uint16_t* __restrict__ cmp_k,
    const int32_t* __restrict__ nb_per_row, int m_rows, int nh, int hd, int nb_stride, float scale, float* __restrict__ scores) {
    extern __shared__ float4 q4exp_scores_smem4[];
    float* smem = reinterpret_cast<float*>(q4exp_scores_smem4);
    int m0 = blockIdx.y * R;
    int rows = min(R, m_rows - m0);
    for (int i = threadIdx.x; i < rows * nh * hd; i += blockDim.x) smem[i] = ld_bf16(q + (size_t)m0 * nh * hd + i);
    __syncthreads();
    int b0 = (blockIdx.x * blockDim.x + threadIdx.x) * KB;
    if (b0 >= nb_stride) return;
    int nbr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) nbr[r] = (r < rows) ? nb_per_row[m0 + r] : 0;
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
            uint4 kv = (b < nb_stride) ? *reinterpret_cast<const uint4*>(cmp_k + (size_t)b * hd + i) : make_uint4(0, 0, 0, 0);
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
        if (b >= nb_stride) break;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (r >= rows) break;
            float* out = scores + (size_t)(m0 + r) * nb_stride + b;
            if (b >= nbr[r]) { *out = -INFINITY; continue; }
            float sv = 0.f;
#pragma unroll
            for (int h = 0; h < 4; ++h) if (h < nh) sv += fmaxf(acc[kb][r][h], 0.f);
            *out = sv * scale;
        }
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

__global__ void Q4EXP_CLUSTER_DIMS_8 __launch_bounds__(QSA_TOPK_CLUSTER_THREADS) q4exp_qsa_topk_cluster_kernel(
    const float* __restrict__ scores, const int32_t* __restrict__ nb_per_row, int k, int nb_stride,
    int32_t* __restrict__ out_blocks, int32_t* __restrict__ out_count) {
#if Q4EXP_HAS_CLUSTERS
    __shared__ int lhist[256];
    __shared__ int suf[256];
    __shared__ int warp_sums[32];
    __shared__ uint32_t s_prefix;
    __shared__ int s_need;
    __shared__ int s_tot[2];
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
        return;
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
    int span = (nb + QSA_TOPK_CLUSTER_CTAS_PER_ROW - 1) / QSA_TOPK_CLUSTER_CTAS_PER_ROW;
    int base = (int)rank * span;
    int end = min(nb, base + span);
    int per = (span + QSA_TOPK_CLUSTER_THREADS - 1) / QSA_TOPK_CLUSTER_THREADS;
    int lo = base + tid * per;
    int cnt = max(0, min(per, end - lo));
    uint32_t key[QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS];
#pragma unroll
    for (int j = 0; j < QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS; ++j) key[j] = (j < cnt) ? ordered_key(row[lo + j]) : 0u;
    asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");
    uint32_t prefix = 0;
    int need = k;
    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - 8 * pass;
        uint32_t mask = (pass == 0) ? 0u : (0xffffffffu << (32 - 8 * pass));
        lhist[tid] = 0;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS; ++j)
            if (j < cnt && (key[j] & mask) == prefix) atomicAdd(&lhist[(key[j] >> shift) & 0xffu], 1);
        q4exp_topk_cluster_sync();
        int bin = 255 - tid;
        int c = 0;
#pragma unroll
        for (int r = 0; r < QSA_TOPK_CLUSTER_CTAS_PER_ROW; ++r) c += q4exp_topk_remote_ld(&lhist[bin], r);
        int lane = tid & 31, w = tid >> 5;
        int x = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) { int y = __shfl_up_sync(0xffffffffu, x, o); if (lane >= o) x += y; }
        if (lane == 31) warp_sums[w] = x;
        __syncthreads();
        int add = 0;
        for (int q = 0; q < w; ++q) add += warp_sums[q];
        int incl = x + add;
        suf[bin] = incl;
        if (tid == 0) { s_prefix = prefix; s_need = need - (warp_sums[0] + warp_sums[1] + warp_sums[2] + warp_sums[3] + warp_sums[4] + warp_sums[5] + warp_sums[6] + warp_sums[7]); }
        __syncthreads();
        int excl = incl - c;
        if (excl < need && incl >= need) { s_prefix = prefix | ((uint32_t)bin << shift); s_need = need - excl; }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        q4exp_topk_cluster_sync();
    }
    uint32_t threshold = prefix;
    int need_eq = need;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS; ++j) if (j < cnt) { gt += key[j] > threshold; eq += key[j] == threshold; }
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
            if (key[j] > threshold) outp[gt_off++] = lo + j;
            else if (key[j] == threshold) { if (eq_off < need_eq) outp[(k - need_eq) + eq_off] = lo + j; ++eq_off; }
        }
    }
    if (rank == 0 && tid == 0) out_count[m] = k;
    q4exp_topk_cluster_sync();
#else
    __trap();
#endif
}

__global__ void q4exp_qsa_expand_kernel(
    const int32_t* __restrict__ blocks, const int32_t* __restrict__ block_count,
    const int32_t* __restrict__ positions, int ratio, int k, int width,
    int32_t* __restrict__ out_idx, int32_t* __restrict__ out_count
) {
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
}

constexpr int ATTN_HD = 256;
constexpr int ATTN_HEADS_PER_WARP = 3;
constexpr int ATTN_WARPS = 4;
constexpr int ATTN_TOKENS_LOADED_AHEAD_OF_THE_ONLINE_SOFTMAX_CHAIN = 8;

__global__ void __launch_bounds__(128) q4exp_attn_rows_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const int32_t* __restrict__ idx,
    const int32_t* __restrict__ count, int nq, int nkv, int splits, int idx_stride, float scale,
    float* __restrict__ partial
) {
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
    float qf[ATTN_HEADS_PER_WARP][8];
    float mx[ATTN_HEADS_PER_WARP];
    float l[ATTN_HEADS_PER_WARP];
    float acc[ATTN_HEADS_PER_WARP][8];
#pragma unroll
    for (int hh = 0; hh < ATTN_HEADS_PER_WARP; ++hh) {
        int head = g * group + warp * ATTN_HEADS_PER_WARP + hh;
        uint4 qv = *reinterpret_cast<const uint4*>(q + ((size_t)m * nq + head) * ATTN_HD + lane * 8);
        unpack8_bf16(qv, qf[hh]);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            qf[hh][j] *= scale;
            acc[hh][j] = 0.f;
        }
        mx[hh] = -INFINITY;
        l[hh] = 0.f;
    }
    for (int t0 = start; t0 < end; t0 += ATTN_TOKENS_LOADED_AHEAD_OF_THE_ONLINE_SOFTMAX_CHAIN) {
        constexpr int B = ATTN_TOKENS_LOADED_AHEAD_OF_THE_ONLINE_SOFTMAX_CHAIN;
        int tok[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            int t = t0 + b;
            tok[b] = (t < end) ? (row_idx ? row_idx[t] : t) : -1;
        }
        uint4 kv[B];
        uint4 vv[B];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            if (tok[b] >= 0) {
                kv[b] = *reinterpret_cast<const uint4*>(k_cache + ((size_t)tok[b] * nkv + g) * ATTN_HD + lane * 8);
                vv[b] = *reinterpret_cast<const uint4*>(v_cache + ((size_t)tok[b] * nkv + g) * ATTN_HD + lane * 8);
            }
        }
        float sc[B][ATTN_HEADS_PER_WARP];
#pragma unroll
        for (int b = 0; b < B; ++b) {
            if (tok[b] >= 0) {
                float kf[8];
                unpack8_bf16(kv[b], kf);
#pragma unroll
                for (int hh = 0; hh < ATTN_HEADS_PER_WARP; ++hh) {
                    float s = 0.f;
#pragma unroll
                    for (int j = 0; j < 8; ++j) s += qf[hh][j] * kf[j];
                    sc[b][hh] = warp_sum(s);
                }
            }
        }
#pragma unroll
        for (int b = 0; b < B; ++b) {
            if (tok[b] >= 0) {
                float vf[8];
                unpack8_bf16(vv[b], vf);
#pragma unroll
                for (int hh = 0; hh < ATTN_HEADS_PER_WARP; ++hh) {
                    float s = sc[b][hh];
                    float m_new = fmaxf(mx[hh], s);
                    float corr = expf(mx[hh] - m_new);
                    float p = expf(s - m_new);
                    l[hh] = l[hh] * corr + p;
#pragma unroll
                    for (int j = 0; j < 8; ++j) acc[hh][j] = acc[hh][j] * corr + p * vf[j];
                    mx[hh] = m_new;
                }
            }
        }
    }
#pragma unroll
    for (int hh = 0; hh < ATTN_HEADS_PER_WARP; ++hh) {
        int head = g * group + warp * ATTN_HEADS_PER_WARP + hh;
        float* dst = partial + (((size_t)m * nq + head) * splits + split) * (ATTN_HD + 2);
#pragma unroll
        for (int j = 0; j < 8; ++j) dst[lane * 8 + j] = acc[hh][j];
        if (lane == 0) {
            dst[ATTN_HD] = mx[hh];
            dst[ATTN_HD + 1] = l[hh];
        }
    }
}

__global__ void q4exp_attn_combine_kernel(
    const float* __restrict__ partial, const uint16_t* __restrict__ gate,
    int nq, int splits, uint16_t* __restrict__ out
) {
    int head = blockIdx.x;
    int m = blockIdx.y;
    int d = threadIdx.x;
    const float* base = partial + ((size_t)m * nq + head) * splits * (ATTN_HD + 2);
    float gm = -INFINITY;
    for (int s = 0; s < splits; ++s) gm = fmaxf(gm, base[(size_t)s * (ATTN_HD + 2) + ATTN_HD]);
    float num = 0.f;
    float den = 0.f;
    for (int s = 0; s < splits; ++s) {
        const float* p = base + (size_t)s * (ATTN_HD + 2);
        float w = expf(p[ATTN_HD] - gm);
        num += w * p[d];
        den += w * p[ATTN_HD + 1];
    }
    float o = (den > 0.f) ? (num / den) : 0.f;
    o = bf16_round(o);
    if (gate) {
        float gv = ld_bf16(gate + ((size_t)m * nq + head) * ATTN_HD + d);
        o = o * sigmoidf_(gv);
    }
    st_bf16(out + ((size_t)m * nq + head) * ATTN_HD + d, o);
}

__global__ void q4exp_store_rows_at_kernel(
    const uint16_t* __restrict__ src, uint16_t* __restrict__ dst, const int32_t* __restrict__ pos,
    int t, int row_elems
) {
    int row = blockIdx.y;
    int p = pos[0] + row;
    const uint4* s4 = reinterpret_cast<const uint4*>(src + (size_t)row * row_elems);
    uint4* d4 = reinterpret_cast<uint4*>(dst + (size_t)p * row_elems);
    int n4 = row_elems / 8;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += gridDim.x * blockDim.x) d4[i] = s4[i];
}

__global__ void q4exp_decode_meta_kernel(
    const int32_t* __restrict__ pos, int t, int ratio, int32_t* __restrict__ positions,
    int32_t* __restrict__ nb_per_row
) {
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= t) return;
    int p = pos[0] + m;
    positions[m] = p;
    nb_per_row[m] = (p + 1) / ratio;
}

__global__ void q4exp_history_push_kernel(
    uint32_t* __restrict__ history, const uint32_t* __restrict__ tokens, int ctx, int t
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t tmp[8];
    for (int i = 0; i < ctx; ++i) {
        int src = t - ctx + i;
        tmp[i] = (src >= 0) ? tokens[src] : history[ctx + src];
    }
    for (int i = 0; i < ctx; ++i) history[i] = tmp[i];
}

__global__ void q4exp_qsa_compress_devpos_kernel(
    const uint16_t* __restrict__ raw_k, const int32_t* __restrict__ pos, int t,
    const float* __restrict__ k_norm_w, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    uint16_t* __restrict__ out, int ratio, int hd, int rot, float eps
) {
    extern __shared__ float smem[];
    float* v = smem;
    __shared__ float red[8];
    int p0 = pos[0];
    int b0 = p0 / ratio;
    int b_end = (p0 + t) / ratio;
    int b = b0 + blockIdx.x;
    if (b >= b_end) return;
    int d = threadIdx.x;
    float acc = 0.f;
    for (int r = 0; r < ratio; ++r) acc += ld_bf16(raw_k + ((size_t)b * ratio + r) * hd + d);
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

__global__ void q4exp_embed_hc_kernel(
    const uint16_t* __restrict__ embed, const uint32_t* __restrict__ tokens, int h, int nb,
    uint16_t* __restrict__ out
) {
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
}

__device__ __forceinline__ void q4exp_rope_head(float* v, int rot, int pos, const float* cos_t, const float* sin_t) {
    int half = rot / 2;
    int d = threadIdx.x;
    float a = v[d];
    __syncthreads();
    float o = a;
    if (d < half) {
        float c = cos_t[(size_t)pos * half + d];
        float s = sin_t[(size_t)pos * half + d];
        o = v[d] * c - v[d + half] * s;
    } else if (d < rot) {
        float c = cos_t[(size_t)pos * half + (d - half)];
        float s = sin_t[(size_t)pos * half + (d - half)];
        o = v[d] * c + v[d - half] * s;
    }
    __syncthreads();
    v[d] = o;
    __syncthreads();
}

__device__ __forceinline__ float q4exp_block_rstd(float x, int n, float eps, float* red) {
    float ss = warp_sum(x * x);
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = ss;
    __syncthreads();
    float tot = 0.f;
    int nw = (blockDim.x + 31) >> 5;
    for (int w = 0; w < nw; ++w) tot += red[w];
    __syncthreads();
    return rsqrtf(tot / (float)n + eps);
}

__global__ void q4exp_attn_prep_kernel(
    const uint16_t* __restrict__ q_raw, const uint16_t* __restrict__ k_raw,
    const uint16_t* __restrict__ v_raw, const uint16_t* __restrict__ idx_raw,
    const float* __restrict__ q_norm_w, const float* __restrict__ k_norm_w,
    const float* __restrict__ iq_norm_w, const float* __restrict__ cos_t, const float* __restrict__ sin_t,
    const int32_t* __restrict__ pos_dev, int nq, int nkv, int hd, int inh, int ih, int rot, float eps,
    uint16_t* __restrict__ q_out, uint16_t* __restrict__ gate_out, uint16_t* __restrict__ k_cache,
    uint16_t* __restrict__ v_cache, uint16_t* __restrict__ raw_idx_cache, uint16_t* __restrict__ iq_out
) {
    extern __shared__ float smem[];
    float* v = smem;
    __shared__ float red[32];
    int m = blockIdx.y;
    int task = blockIdx.x;
    int pos = pos_dev[0] + m;
    int d = threadIdx.x;
    if (task < nq) {
        int h = task;
        const uint16_t* src = q_raw + ((size_t)m * nq + h) * 2 * hd;
        float x = ld_bf16(src + d);
        float rstd = q4exp_block_rstd(x, hd, eps, red);
        v[d] = bf16_round(x * rstd * q_norm_w[d]);
        __syncthreads();
        q4exp_rope_head(v, rot, pos, cos_t, sin_t);
        st_bf16(q_out + ((size_t)m * nq + h) * hd + d, v[d]);
        gate_out[((size_t)m * nq + h) * hd + d] = src[hd + d];
        return;
    }
    task -= nq;
    if (task < nkv) {
        int h = task;
        float x = ld_bf16(k_raw + ((size_t)m * nkv + h) * hd + d);
        float rstd = q4exp_block_rstd(x, hd, eps, red);
        v[d] = bf16_round(x * rstd * k_norm_w[d]);
        __syncthreads();
        q4exp_rope_head(v, rot, pos, cos_t, sin_t);
        st_bf16(k_cache + ((size_t)pos * nkv + h) * hd + d, v[d]);
        v_cache[((size_t)pos * nkv + h) * hd + d] = v_raw[((size_t)m * nkv + h) * hd + d];
        return;
    }
    task -= nkv;
    if (task < inh) {
        int h = task;
        if (d < ih) {
            float x = ld_bf16(idx_raw + (size_t)m * (inh + 1) * ih + h * ih + d);
            float rstd = q4exp_block_rstd(x, ih, eps, red);
            v[d] = bf16_round(x * rstd * iq_norm_w[d]);
        } else {
            q4exp_block_rstd(0.f, ih, eps, red);
        }
        __syncthreads();
        if (d < ih) {
            int half = rot / 2;
            float o = v[d];
            if (d < half) {
                float c = cos_t[(size_t)pos * half + d];
                float s = sin_t[(size_t)pos * half + d];
                o = v[d] * c - v[d + half] * s;
            } else if (d < rot) {
                float c = cos_t[(size_t)pos * half + (d - half)];
                float s = sin_t[(size_t)pos * half + (d - half)];
                o = v[d] * c + v[d - half] * s;
            }
            st_bf16(iq_out + ((size_t)m * inh + h) * ih + d, o);
        }
        return;
    }
    if (d < ih) {
        raw_idx_cache[(size_t)pos * ih + d] = idx_raw[(size_t)m * (inh + 1) * ih + inh * ih + d];
    }
}

__global__ void q4exp_ple_gate_kernel(
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ value, const uint16_t* __restrict__ hc,
    const float* __restrict__ norm_key_w, const float* __restrict__ norm_query_w,
    const float* __restrict__ norm_conv_w, int h, int nb, float eps,
    uint16_t* __restrict__ gv_out, float* __restrict__ gvn_out
) {
    __shared__ float red[32];
    __shared__ float s_gate;
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
        float kn = bf16_round(ld_bf16(key + base + i) * rk * norm_key_w[b * h + i]);
        float qn = bf16_round(ld_bf16(hc + base + i) * rq * norm_query_w[b * h + i]);
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
        gvn_out[base + i] = bf16_round(gvv * rg * norm_conv_w[b * h + i]);
    }
}

__global__ void q4exp_ple_conv_kernel(
    const float* __restrict__ gvn, const uint16_t* __restrict__ gv, float* __restrict__ window,
    const float* __restrict__ conv_w, int t, int hh, int k, int dil, int win, int update_window,
    uint16_t* __restrict__ out
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hh) return;
    float full[64];
    for (int i = 0; i < win; ++i) full[i] = window[(size_t)i * hh + c];
    for (int r = 0; r < t; ++r) full[win + r] = gvn[(size_t)r * hh + c];
    for (int r = 0; r < t; ++r) {
        float acc = 0.f;
        for (int tap = 0; tap < k; ++tap) acc += conv_w[(size_t)c * k + tap] * full[r + tap * dil];
        float o = acc * sigmoidf_(acc);
        st_bf16(out + (size_t)r * hh + c, ld_bf16(gv + (size_t)r * hh + c) + o);
    }
    if (update_window) {
        for (int i = 0; i < win; ++i) window[(size_t)i * hh + c] = full[t + i];
    }
}

__global__ void q4exp_ple_window_advance_kernel(
    const float* __restrict__ gvn, float* __restrict__ window, int t, int hh, int win
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hh) return;
    float full[64];
    for (int i = 0; i < win; ++i) full[i] = window[(size_t)i * hh + c];
    for (int r = 0; r < t; ++r) full[win + r] = gvn[(size_t)r * hh + c];
    for (int i = 0; i < win; ++i) window[(size_t)i * hh + c] = full[t + i];
}

__global__ void q4exp_rmsnorm_rows_kernel(
    const uint16_t* __restrict__ x, const float* __restrict__ w, uint16_t* __restrict__ out, int n,
    float eps
) {
    __shared__ float red[32];
    int m = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const uint16_t* xr = x + (size_t)m * n;
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
    for (int i = tid; i < n; i += blockDim.x) st_bf16(out + (size_t)m * n + i, ld_bf16(xr + i) * rstd * w[i]);
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
    q4exp_ngram_ids_kernel<<<t, n_heads, 0, (cudaStream_t)stream>>>(
        tokens, history, multipliers, sizes, offsets, eos, t, ctx, heads_per_ngram, out);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_ple_gather_bf16(
    void* stream, const uint8_t* table, const int64_t* ids, float scale, int t, int heads, int hd,
    uint16_t* out
) {
    if (t <= 0) return 0;
    if (hd % 4 != 0) return -2;
    dim3 grid(t, heads);
    q4exp_ple_gather_kernel<<<grid, 64, 0, (cudaStream_t)stream>>>(table, ids, scale, heads, hd, out);
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
    void* stream, const uint16_t* raw_k, int b0, int nblocks, const float* k_norm_w,
    const float* cos_t, const float* sin_t, uint16_t* out, int ratio, int hd, int rot, float eps
) {
    if (nblocks <= 0) return 0;
    if (hd % 32 != 0 || hd > 1024 || rot % 2 != 0 || rot > hd) return -2;
    size_t smem = (size_t)hd * sizeof(float);
    q4exp_qsa_compress_kernel<<<nblocks, hd, smem, (cudaStream_t)stream>>>(
        raw_k, b0, k_norm_w, cos_t, sin_t, out, ratio, hd, rot, eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_qsa_scores(
    void* stream, const uint16_t* q, const uint16_t* cmp_k, const int32_t* nb_per_row, int m, int nh,
    int hd, int nb_stride, float scale, float* scores
) {
    if (m <= 0 || nb_stride <= 0) return 0;
    if (nh > 4 || hd % 8 != 0) return -2;
    if (m >= QSA_SCORE_ROWS_PER_TILE_16_EACH_KEY_BLOCK_IS_READ_ONCE_PER_TILE) {
        constexpr int R = QSA_SCORE_ROWS_PER_TILE_16_EACH_KEY_BLOCK_IS_READ_ONCE_PER_TILE;
        dim3 grid((nb_stride + 511) / 512, (m + R - 1) / R);
        size_t smem = (size_t)R * nh * hd * sizeof(float);
        q4exp_qsa_scores_rows_kernel<R, 2><<<grid, 256, smem, (cudaStream_t)stream>>>(
            q, cmp_k, nb_per_row, m, nh, hd, nb_stride, scale, scores);
        return (int)cudaGetLastError();
    }
    dim3 grid((nb_stride + 255) / 256, m);
    size_t smem = (size_t)nh * hd * sizeof(float);
    q4exp_qsa_scores_kernel<<<grid, 256, smem, (cudaStream_t)stream>>>(
        q, cmp_k, nb_per_row, nh, hd, nb_stride, scale, scores);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_qsa_topk(
    void* stream, const float* scores, const int32_t* nb_per_row, int m, int k, int nb_stride,
    int32_t* out_blocks, int32_t* out_count
) {
    if (m <= 0) return 0;
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major >= 9 && m < QSA_TOPK_CLUSTER_BELOW_16_ROWS_ONE_ROW_PER_BLOCK_STARVES_THE_GPU
        && nb_stride <= QSA_TOPK_CLUSTER_CTAS_PER_ROW * QSA_TOPK_CLUSTER_THREADS * QSA_TOPK_CLUSTER_KEYS_PER_THREAD_COVER_65536_BLOCKS) {
        q4exp_qsa_topk_cluster_kernel<<<dim3(QSA_TOPK_CLUSTER_CTAS_PER_ROW, m), QSA_TOPK_CLUSTER_THREADS, 0, (cudaStream_t)stream>>>(
            scores, nb_per_row, k, nb_stride, out_blocks, out_count);
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
    q4exp_qsa_expand_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        blocks, block_count, positions, ratio, k, width, out_idx, out_count);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_attn_rows(
    void* stream, const uint16_t* q, const uint16_t* k_cache, const uint16_t* v_cache,
    const int32_t* idx, const int32_t* count, int m, int nq, int nkv, int hd, int splits,
    int idx_stride, float scale, float* partial
) {
    if (m <= 0) return 0;
    if (hd != ATTN_HD || nq != nkv * ATTN_HEADS_PER_WARP * ATTN_WARPS || splits <= 0) return -2;
    dim3 grid(splits, nkv, m);
    q4exp_attn_rows_kernel<<<grid, ATTN_WARPS * 32, 0, (cudaStream_t)stream>>>(
        q, k_cache, v_cache, idx, count, nq, nkv, splits, idx_stride, scale, partial);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_attn_combine(
    void* stream, const float* partial, const uint16_t* gate, int m, int nq, int hd, int splits,
    uint16_t* out
) {
    if (m <= 0) return 0;
    if (hd != ATTN_HD) return -2;
    dim3 grid(nq, m);
    q4exp_attn_combine_kernel<<<grid, ATTN_HD, 0, (cudaStream_t)stream>>>(partial, gate, nq, splits, out);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_store_rows_at(
    void* stream, const uint16_t* src, uint16_t* dst, const int32_t* pos, int t, int row_elems
) {
    if (t <= 0) return 0;
    if (row_elems % 8 != 0) return -2;
    dim3 grid((row_elems / 8 + 255) / 256, t);
    q4exp_store_rows_at_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(src, dst, pos, t, row_elems);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_decode_meta(
    void* stream, const int32_t* pos, int t, int ratio, int32_t* positions, int32_t* nb_per_row
) {
    if (t <= 0) return 0;
    q4exp_decode_meta_kernel<<<(t + 127) / 128, 128, 0, (cudaStream_t)stream>>>(pos, t, ratio, positions, nb_per_row);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_history_push(
    void* stream, uint32_t* history, const uint32_t* tokens, int ctx, int t
) {
    if (ctx <= 0 || ctx > 8 || t <= 0) return -2;
    q4exp_history_push_kernel<<<1, 32, 0, (cudaStream_t)stream>>>(history, tokens, ctx, t);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_qsa_compress_devpos(
    void* stream, const uint16_t* raw_k, const int32_t* pos, int t, const float* k_norm_w,
    const float* cos_t, const float* sin_t, uint16_t* out, int ratio, int hd, int rot, float eps
) {
    if (t <= 0) return 0;
    if (hd % 32 != 0 || hd > 1024 || rot % 2 != 0 || rot > hd) return -2;
    int max_blocks = t / ratio + 1;
    size_t smem = (size_t)hd * sizeof(float);
    q4exp_qsa_compress_devpos_kernel<<<max_blocks, hd, smem, (cudaStream_t)stream>>>(
        raw_k, pos, t, k_norm_w, cos_t, sin_t, out, ratio, hd, rot, eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_embed_hc(
    void* stream, const uint16_t* embed, const uint32_t* tokens, int t, int h, int nb, uint16_t* out
) {
    if (t <= 0) return 0;
    if (h % 8 != 0) return -2;
    q4exp_embed_hc_kernel<<<t, 256, 0, (cudaStream_t)stream>>>(embed, tokens, h, nb, out);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_attn_prep(
    void* stream, const uint16_t* q_raw, const uint16_t* k_raw, const uint16_t* v_raw,
    const uint16_t* idx_raw, const float* q_norm_w, const float* k_norm_w, const float* iq_norm_w,
    const float* cos_t, const float* sin_t, const int32_t* pos_dev, int t, int nq, int nkv, int hd,
    int inh, int ih, int rot, float eps, uint16_t* q_out, uint16_t* gate_out, uint16_t* k_cache,
    uint16_t* v_cache, uint16_t* raw_idx_cache, uint16_t* iq_out
) {
    if (t <= 0) return 0;
    if (hd % 32 != 0 || hd > 1024 || ih > hd || ih % 32 != 0 || rot % 2 != 0 || rot > ih) return -2;
    dim3 grid(nq + nkv + inh + 1, t);
    size_t smem = (size_t)hd * sizeof(float);
    q4exp_attn_prep_kernel<<<grid, hd, smem, (cudaStream_t)stream>>>(
        q_raw, k_raw, v_raw, idx_raw, q_norm_w, k_norm_w, iq_norm_w, cos_t, sin_t, pos_dev, nq, nkv, hd,
        inh, ih, rot, eps, q_out, gate_out, k_cache, v_cache, raw_idx_cache, iq_out);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_ple_gate(
    void* stream, const uint16_t* key, const uint16_t* value, const uint16_t* hc, const float* norm_key_w,
    const float* norm_query_w, const float* norm_conv_w, int t, int h, int nb, float eps,
    uint16_t* gv_out, float* gvn_out
) {
    if (t <= 0) return 0;
    dim3 grid(nb, t);
    q4exp_ple_gate_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        key, value, hc, norm_key_w, norm_query_w, norm_conv_w, h, nb, eps, gv_out, gvn_out);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_ple_conv(
    void* stream, const float* gvn, const uint16_t* gv, float* window, const float* conv_w, int t,
    int hh, int k, int dil, int win, int update_window, uint16_t* out
) {
    if (t <= 0) return 0;
    if (win + t > 64 || (k - 1) * dil != win) return -2;
    q4exp_ple_conv_kernel<<<(hh + 255) / 256, 256, 0, (cudaStream_t)stream>>>(
        gvn, gv, window, conv_w, t, hh, k, dil, win, update_window, out);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_ple_window_advance(
    void* stream, const float* gvn, float* window, int t, int hh, int win
) {
    if (t <= 0) return 0;
    if (win + t > 64) return -2;
    q4exp_ple_window_advance_kernel<<<(hh + 255) / 256, 256, 0, (cudaStream_t)stream>>>(gvn, window, t, hh, win);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_q4exp_rmsnorm_rows(
    void* stream, const uint16_t* x, const float* w, uint16_t* out, int rows, int n, float eps
) {
    if (rows <= 0) return 0;
    q4exp_rmsnorm_rows_kernel<<<rows, 256, 0, (cudaStream_t)stream>>>(x, w, out, n, eps);
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
    q4exp_stage_rows_f32<TM>(x, xshf, K);
    __syncthreads();
    int lane = threadIdx.x & (kQ4ExpFp8Warp - 1);
    int warp = threadIdx.x / kQ4ExpFp8Warp;
    int n = blockIdx.x * kQ4ExpFp8Rows + warp;
    if (n >= N) return;
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
}

template <int TM>
__global__ void q4exp_fp8_down_tail_kernel(
    const uint8_t* __restrict__ wq, const float* __restrict__ rs, const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ y_down, long long slot_stride, const float* __restrict__ topk_w, int k,
    const __nv_bfloat16* __restrict__ sg, __nv_bfloat16* __restrict__ out, int N, int K
) {
    extern __shared__ float2 xshf[];
    q4exp_stage_rows_f32<TM>(act, xshf, K);
    __syncthreads();
    int lane = threadIdx.x & (kQ4ExpFp8Warp - 1);
    int warp = threadIdx.x / kQ4ExpFp8Warp;
    int n0 = blockIdx.x * (2 * kQ4ExpFp8Rows) + warp;
    int n1 = n0 + kQ4ExpFp8Rows;
    if (n0 >= N) return;
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
    int dev = 0, major = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    if (major < 9) {
        q4exp_hc_combine_norm_kernel<kQ4ExpCombineNormChunksPerThread><<<m, 256, 0, (cudaStream_t)stream>>>(
            hc_in, y, inject, norm_w, hc_out, xn_out, hh, h, nb, eps);
        return (int)cudaGetLastError();
    }
    int iters = (hh / 8 + 255) / 256;
    q4exp_hc_combine_norm_cluster_kernel<<<dim3(8, m), 32 * iters, 0, (cudaStream_t)stream>>>(
        hc_in, y, inject, norm_w, hc_out, xn_out, hh, h, nb, eps);
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
    q4exp_fp8_gateup_silu_kernel<TM><<<grid, kQ4ExpFp8Block, smem, s>>>(
        wg, rsg, wu, rsu, reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(act), N, K);
    return (int)cudaGetLastError();
}

template <int TM>
static int q4exp_launch_down_tail(cudaStream_t s, const uint8_t* wq, const float* rs, const uint16_t* act,
                                  const uint16_t* y_down, long long slot_stride, const float* topk_w, int k,
                                  const uint16_t* sg, uint16_t* out, int N, int K) {
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(float2);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(optin, (const void*)q4exp_fp8_down_tail_kernel<TM>, smem);
    if (orc != 0) return orc;
    dim3 grid((unsigned)((N + 2 * kQ4ExpFp8Rows - 1) / (2 * kQ4ExpFp8Rows)));
    q4exp_fp8_down_tail_kernel<TM><<<grid, kQ4ExpFp8Block, smem, s>>>(
        wq, rs, reinterpret_cast<const __nv_bfloat16*>(act), reinterpret_cast<const __nv_bfloat16*>(y_down), slot_stride,
        topk_w, k, reinterpret_cast<const __nv_bfloat16*>(sg), reinterpret_cast<__nv_bfloat16*>(out), N, K);
    return (int)cudaGetLastError();
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
