
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

namespace {

__global__ void laguna_rope_scale_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int* __restrict__ pos_base,
    int n_q,
    int n_kv,
    int head_dim,
    int rotary_dim,
    float rot_scale
) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int half = rotary_dim / 2;

    int pos = pos_base[0] + token_idx;
    const float* cos_row = cos_tbl + (size_t)pos * half;
    const float* sin_row = sin_tbl + (size_t)pos * half;

    const __nv_bfloat16* src;
    __nv_bfloat16* dst;
    if (head_idx < n_q) {
        src = q_in + ((size_t)token_idx * n_q + head_idx) * head_dim;
        dst = q_out + ((size_t)token_idx * n_q + head_idx) * head_dim;
    } else {
        int kv_head = head_idx - n_q;
        if (kv_head >= n_kv) return;
        src = k_in + ((size_t)token_idx * n_kv + kv_head) * head_dim;
        dst = k_out + ((size_t)token_idx * n_kv + kv_head) * head_dim;
    }

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float c = cos_row[i];
        float s = sin_row[i];
        float a = __bfloat162float(src[i]);
        float b = __bfloat162float(src[i + half]);
        float lo = a * c - b * s;
        float hi = a * s + b * c;
        lo = lo * rot_scale;
        hi = hi * rot_scale;
        dst[i] = __float2bfloat16(lo);
        dst[i + half] = __float2bfloat16(hi);
    }
    for (int d = rotary_dim + threadIdx.x; d < head_dim; d += blockDim.x) {
        dst[d] = src[d];
    }
}

constexpr int kNormBlock = 256;

__device__ inline float ls_block_sum256(float v) {
    constexpr int kWarp = 32;
    constexpr int kWarps = kNormBlock / kWarp;
    __shared__ float warp_sums[kWarps];
    __shared__ float total;
    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

__global__ void laguna_rstd256_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ rstd_out,
    int dim,
    float eps
) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += kNormBlock) {
        float v = __bfloat162float(x[i]);
        local += v * v;
    }
    float sum = ls_block_sum256(local);
    if (threadIdx.x == 0) rstd_out[0] = rsqrtf(sum / (float)dim + eps);
}

__global__ void laguna_qk_normrope_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int* __restrict__ pos_base,
    int n_q,
    int head_dim,
    int rotary_dim,
    float rot_scale,
    float eps_q,
    float eps_k
) {
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const bool is_q = h < n_q;
    const __nv_bfloat16* src = is_q ? q_in + (size_t)h * head_dim
                                    : k_in + (size_t)(h - n_q) * head_dim;
    __nv_bfloat16* dst = is_q ? q_out + (size_t)h * head_dim
                              : k_out + (size_t)(h - n_q) * head_dim;
    const __nv_bfloat16* w = is_q ? qw : kw;
    const float eps = is_q ? eps_q : eps_k;

    float x = (d < head_dim) ? __bfloat162float(src[d]) : 0.0f;
    float sum = ls_block_sum256(x * x);
    float rms = rsqrtf(sum / (float)head_dim + eps);

    __shared__ __nv_bfloat16 normed[512];
    if (d < head_dim) {
        normed[d] = __float2bfloat16(x * rms * __bfloat162float(w[d]));
    }
    __syncthreads();

    const int half = rotary_dim / 2;
    const int pos = pos_base[0];
    if (d < half) {
        float c = cos_tbl[(size_t)pos * half + d];
        float s = sin_tbl[(size_t)pos * half + d];
        float a = __bfloat162float(normed[d]);
        float b = __bfloat162float(normed[d + half]);
        float lo = (a * c - b * s) * rot_scale;
        float hi = (a * s + b * c) * rot_scale;
        dst[d] = __float2bfloat16(lo);
        dst[d + half] = __float2bfloat16(hi);
    } else if (d >= rotary_dim && d < head_dim) {
        dst[d] = normed[d];
    }
}

constexpr int kGqaWarp = 32;
constexpr int kGqaWarps = 8;
constexpr int kGqaThreads = kGqaWarp * kGqaWarps;
constexpr int kGqaHD = 128;
constexpr int kGqaVc = kGqaHD / kGqaWarp;

__inline__ __device__ float gqa_warp_sum(float x) {
    #pragma unroll
    for (int o = kGqaWarp / 2; o > 0; o >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

template <int GROUP, int SPLITS>
__global__ void laguna_flash_decode_gqa_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ total_ptr,
    int delta,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    int NKV,
    int WINDOW,
    float scale
) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int HD = kGqaHD;

    const int total = total_ptr[0] + delta;
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;

    const int lane = threadIdx.x & (kGqaWarp - 1);
    const int warp = threadIdx.x >> 5;

    __shared__ float qsh[GROUP][kGqaHD];
    for (int i = threadIdx.x; i < GROUP * HD; i += kGqaThreads) {
        int g = i / HD;
        int d = i - g * HD;
        qsh[g][d] = __bfloat162float(q[(size_t)(kvh * GROUP + g) * HD + d]) * scale;
    }
    __syncthreads();

    float acc[GROUP][kGqaVc];
    float m[GROUP], l[GROUP];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        #pragma unroll
        for (int j = 0; j < kGqaVc; ++j) acc[g][j] = 0.0f;
    }

    const int stride = SPLITS * kGqaWarps;
    int p = start + split * kGqaWarps + warp;
    uint2 kraw, vraw;
    if (p < total) {
        kraw = __ldg(reinterpret_cast<const uint2*>(k + ((size_t)p * NKV + kvh) * HD) + lane);
        vraw = __ldg(reinterpret_cast<const uint2*>(v + ((size_t)p * NKV + kvh) * HD) + lane);
    }
    for (; p < total; p += stride) {
        const int pn = p + stride;
        uint2 kn, vn;
        if (pn < total) {
            kn = __ldg(reinterpret_cast<const uint2*>(k + ((size_t)pn * NKV + kvh) * HD) + lane);
            vn = __ldg(reinterpret_cast<const uint2*>(v + ((size_t)pn * NKV + kvh) * HD) + lane);
        }
        const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&kraw);
        float2 k01 = __bfloat1622float2(kb[0]);
        float2 k23 = __bfloat1622float2(kb[1]);
        const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&vraw);
        float2 v01 = __bfloat1622float2(vb[0]);
        float2 v23 = __bfloat1622float2(vb[1]);
        #pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            const float* qp = qsh[g] + lane * kGqaVc;
            float partial = k01.x * qp[0] + k01.y * qp[1] + k23.x * qp[2] + k23.y * qp[3];
            float score = gqa_warp_sum(partial);
            float m_new = fmaxf(m[g], score);
            float corr = __expf(m[g] - m_new);
            float w = __expf(score - m_new);
            l[g] = l[g] * corr + w;
            acc[g][0] = acc[g][0] * corr + w * v01.x;
            acc[g][1] = acc[g][1] * corr + w * v01.y;
            acc[g][2] = acc[g][2] * corr + w * v23.x;
            acc[g][3] = acc[g][3] * corr + w * v23.y;
            m[g] = m_new;
        }
        if (pn < total) {
            kraw = kn;
            vraw = vn;
        }
    }

    __shared__ float sm[kGqaWarps][GROUP];
    __shared__ float sl[kGqaWarps][GROUP];
    __shared__ float sacc[kGqaWarps][GROUP][kGqaHD];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
        if (lane == 0) {
            sm[warp][g] = m[g];
            sl[warp][g] = l[g];
        }
        #pragma unroll
        for (int j = 0; j < kGqaVc; ++j) sacc[warp][g][lane * kGqaVc + j] = acc[g][j];
    }
    __syncthreads();

    float* part = scratch + ((size_t)kvh * SPLITS + split) * GROUP * (kGqaHD + 2);
    if (warp == 0 && lane < GROUP) {
        const int g = lane;
        float m_blk = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kGqaWarps; ++w) m_blk = fmaxf(m_blk, sm[w][g]);
        float l_blk = 0.0f;
        #pragma unroll
        for (int w = 0; w < kGqaWarps; ++w)
            l_blk += (sm[w][g] > -INFINITY) ? sl[w][g] * __expf(sm[w][g] - m_blk) : 0.0f;
        part[(size_t)g * (kGqaHD + 2)] = m_blk;
        part[(size_t)g * (kGqaHD + 2) + 1] = l_blk;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < GROUP * kGqaHD; i += kGqaThreads) {
        int g = i / kGqaHD;
        int d = i - g * kGqaHD;
        float m_blk = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kGqaWarps; ++w) m_blk = fmaxf(m_blk, sm[w][g]);
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < kGqaWarps; ++w)
            a += (sm[w][g] > -INFINITY) ? sacc[w][g][d] * __expf(sm[w][g] - m_blk) : 0.0f;
        part[(size_t)g * (kGqaHD + 2) + 2 + d] = a;
    }

    __threadfence();
    __syncthreads();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[kvh], 1u);
    __syncthreads();
    if (ticket != SPLITS - 1) return;
    __threadfence();

    const float* base = scratch + (size_t)kvh * SPLITS * GROUP * (kGqaHD + 2);
    __shared__ float ssc[GROUP][SPLITS];
    __shared__ float sinv_l[GROUP];
    if (warp == 0 && lane < GROUP) {
        const int g = lane;
        float m_glob = -INFINITY;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s)
            m_glob = fmaxf(m_glob, base[((size_t)s * GROUP + g) * (kGqaHD + 2)]);
        float l_glob = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s) {
            const float* pp = base + ((size_t)s * GROUP + g) * (kGqaHD + 2);
            float sc = (pp[0] > -INFINITY) ? __expf(pp[0] - m_glob) : 0.0f;
            ssc[g][s] = sc;
            l_glob += pp[1] * sc;
        }
        sinv_l[g] = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < GROUP * kGqaHD; i += kGqaThreads) {
        int g = i / kGqaHD;
        int d = i - g * kGqaHD;
        float a = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s)
            a += base[((size_t)s * GROUP + g) * (kGqaHD + 2) + 2 + d] * ssc[g][s];
        outp[(size_t)(kvh * GROUP + g) * kGqaHD + d] = __float2bfloat16(a * sinv_l[g]);
    }
    if (threadIdx.x == 0) fan_in[kvh] = 0u;
}

inline int laguna_m1_splits_env() {
    static int v = [] {
        const char* e = getenv("NV_LAGUNA_M1_SPLITS");
        if (e == nullptr) return 16;
        int x = atoi(e);
        return (x == 8 || x == 16 || x == 32) ? x : 16;
    }();
    return v;
}

__global__ void laguna_seqlens_prep_kernel(
    const int* __restrict__ meta,
    int* __restrict__ cu_full,
    int* __restrict__ cu_slide,
    int t
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        cu_full[0] = 0;
        cu_full[1] = meta[0] + t;
        cu_slide[0] = 0;
        cu_slide[1] = meta[3];
    }
}

__global__ void softplus_gate_exact_bf16_kernel(
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int groups,
    int hd
) {
    int g = blockIdx.x;
    if (g >= groups) return;
    float x = __bfloat162float(gate[g]);
    float sp = fmaxf(x, 0.f) + logf(1.0f + expf(-fabsf(x)));
    float gv = __bfloat162float(__float2bfloat16(sp));
    const __nv_bfloat16* a = attn + (size_t)g * hd;
    __nv_bfloat16* y = out + (size_t)g * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        y[d] = __float2bfloat16(__bfloat162float(a[d]) * gv);
    }
}

}

extern "C" int nv_kernels_laguna_rope_scale_bf16(
    void* stream,
    const uint16_t* q_in,
    const uint16_t* k_in,
    uint16_t* q_out,
    uint16_t* k_out,
    const float* cos_tbl,
    const float* sin_tbl,
    const int* pos_base,
    int t,
    int n_q,
    int n_kv,
    int head_dim,
    int rotary_dim,
    float rot_scale
) {
    if (t <= 0 || n_q <= 0 || n_kv <= 0) return -1;
    if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)t, (unsigned)(n_q + n_kv));
    int threads = rotary_dim / 2;
    if (threads < head_dim - rotary_dim) threads = head_dim - rotary_dim;
    if (threads > 256) threads = 256;
    if (threads < 32) threads = 32;
    laguna_rope_scale_bf16_kernel<<<grid, threads, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_in),
        reinterpret_cast<const __nv_bfloat16*>(k_in),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        cos_tbl,
        sin_tbl,
        pos_base,
        n_q,
        n_kv,
        head_dim,
        rotary_dim,
        rot_scale
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_laguna_rstd256_bf16(
    void* stream,
    const uint16_t* x,
    float* rstd_out,
    int dim,
    float eps
) {
    if (dim <= 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    laguna_rstd256_bf16_kernel<<<1, kNormBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), rstd_out, dim, eps);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_laguna_qk_normrope_bf16(
    void* stream,
    const uint16_t* q_in,
    const uint16_t* k_in,
    uint16_t* q_out,
    uint16_t* k_out,
    const uint16_t* qw,
    const uint16_t* kw,
    const float* cos_tbl,
    const float* sin_tbl,
    const int* pos_base,
    int n_q,
    int n_kv,
    int head_dim,
    int rotary_dim,
    float rot_scale,
    float eps_q,
    float eps_k
) {
    if (n_q <= 0 || n_kv <= 0) return -1;
    if (head_dim < 32 || head_dim > 256 || (head_dim & 31)) return -2;
    if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    laguna_qk_normrope_bf16_kernel<<<(unsigned)(n_q + n_kv), kNormBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_in),
        reinterpret_cast<const __nv_bfloat16*>(k_in),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<const __nv_bfloat16*>(qw),
        reinterpret_cast<const __nv_bfloat16*>(kw),
        cos_tbl,
        sin_tbl,
        pos_base,
        n_q,
        head_dim,
        rotary_dim,
        rot_scale,
        eps_q,
        eps_k
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_laguna_flash_decode_gqa_scratch_elems(int n_kv) {
    return n_kv * 32 * 8 * (128 + 2);
}

extern "C" int nv_kernels_laguna_flash_decode_gqa(
    void* stream,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* total_ptr,
    int delta,
    float* scratch,
    unsigned int* fan_in,
    int n_q,
    int n_kv,
    int head_dim,
    int window,
    float scale
) {
    if (n_q <= 0 || n_kv <= 0) return 0;
    if (head_dim != kGqaHD || (n_q % n_kv) != 0) return -1;
    const int group = n_q / n_kv;
    if (group != 1 && group != 6 && group != 8) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int sp = laguna_m1_splits_env();
    dim3 grid((unsigned)n_kv, (unsigned)sp);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(v);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    switch (group * 100 + sp) {
        case 108:
            laguna_flash_decode_gqa_kernel<1, 8><<<grid, kGqaThreads, 0, s>>>(
                qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale);
            break;
        case 116:
            laguna_flash_decode_gqa_kernel<1, 16><<<grid, kGqaThreads, 0, s>>>(
                qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale);
            break;
        case 132:
            laguna_flash_decode_gqa_kernel<1, 32><<<grid, kGqaThreads, 0, s>>>(
                qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale);
            break;
        case 608:
            laguna_flash_decode_gqa_kernel<6, 8><<<grid, kGqaThreads, 0, s>>>(
                qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale);
            break;
        case 616:
            laguna_flash_decode_gqa_kernel<6, 16><<<grid, kGqaThreads, 0, s>>>(
                qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale);
            break;
        case 632:
            laguna_flash_decode_gqa_kernel<6, 32><<<grid, kGqaThreads, 0, s>>>(
                qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale);
            break;
        case 808:
            laguna_flash_decode_gqa_kernel<8, 8><<<grid, kGqaThreads, 0, s>>>(
                qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale);
            break;
        case 816:
            laguna_flash_decode_gqa_kernel<8, 16><<<grid, kGqaThreads, 0, s>>>(
                qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale);
            break;
        case 832:
            laguna_flash_decode_gqa_kernel<8, 32><<<grid, kGqaThreads, 0, s>>>(
                qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale);
            break;
        default:
            return -1;
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_laguna_seqlens_prep(
    void* stream,
    const int* meta,
    int* cu_full,
    int* cu_slide,
    int t
) {
    if (t <= 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    laguna_seqlens_prep_kernel<<<1, 1, 0, s>>>(meta, cu_full, cu_slide, t);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

__global__ void prof_timestamp_kernel(unsigned long long* out) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *out = t;
}

extern "C" int nv_kernels_prof_timestamp(
    void* stream,
    unsigned long long* out
) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    prof_timestamp_kernel<<<1, 1, 0, s>>>(out);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_softplus_gate_exact_bf16(
    void* stream,
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int groups,
    int hd
) {
    if (groups <= 0 || hd <= 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    softplus_gate_exact_bf16_kernel<<<groups, 128, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(attn),
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<__nv_bfloat16*>(out),
        groups,
        hd
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}
