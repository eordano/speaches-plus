
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "laguna_flash_body.cuh"

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

__device__ __forceinline__ float2 laguna_fp8x2_to_float2(unsigned short packed) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed, __NV_E4M3);
    return __half22float2(*reinterpret_cast<__half2*>(&hr));
}

template<bool BASE_POSITION>
__global__ void laguna_qk_normrope_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    const void* __restrict__ qw,
    const void* __restrict__ kw,
    const float* __restrict__ qs,
    const float* __restrict__ ks,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int* __restrict__ pos_base,
    int n_q,
    int n_kv,
    int head_dim,
    int rotary_dim,
    float rot_scale,
    float eps_q,
    float eps_k
) {
    q_in += (size_t)blockIdx.y * n_q * head_dim;
    q_out += (size_t)blockIdx.y * n_q * head_dim;
    k_in += (size_t)blockIdx.y * n_kv * head_dim;
    k_out += (size_t)blockIdx.y * n_kv * head_dim;
    const int h = blockIdx.x;
    const int d = threadIdx.x;
    const bool is_q = h < n_q;
    const __nv_bfloat16* src = is_q ? q_in + (size_t)h * head_dim
                                    : k_in + (size_t)(h - n_q) * head_dim;
    __nv_bfloat16* dst = is_q ? q_out + (size_t)h * head_dim
                              : k_out + (size_t)(h - n_q) * head_dim;
    const void* w = is_q ? qw : kw;
    const float* ws = is_q ? qs : ks;
    const float eps = is_q ? eps_q : eps_k;

    float x = (d < head_dim) ? __bfloat162float(src[d]) : 0.0f;
    float sum = ls_block_sum256(x * x);
    float rms = rsqrtf(sum / (float)head_dim + eps);

    __shared__ __nv_bfloat16 normed[512];
    if (d < head_dim) {
        float weight;
        if (ws != nullptr) {
            const float raw = laguna_fp8x2_to_float2(reinterpret_cast<const uint8_t*>(w)[d]).x;
            weight = __bfloat162float(__float2bfloat16(raw * __ldg(ws)));
        } else {
            weight = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(w)[d]);
        }
        normed[d] = __float2bfloat16(x * rms * weight);
    }
    __syncthreads();

    const int half = rotary_dim / 2;
    const int pos = BASE_POSITION ? pos_base[0] + (int)blockIdx.y : pos_base[blockIdx.y];
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
    float scale,
    int mid_band_splits,
    int mid_band_hi_total
) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int HD = kGqaHD;
    const int row = blockIdx.z;
    q += (size_t)row * NKV * GROUP * HD;
    outp += (size_t)row * NKV * GROUP * HD;
    scratch += (size_t)row * NKV * SPLITS * GROUP * (kGqaHD + 2);
    fan_in += (size_t)row * NKV;
    delta += row;

    const int total = total_ptr[0] + delta;
    const int scanned = (WINDOW > 0 && total > WINDOW) ? WINDOW : total;
    const int s_eff =
        (mid_band_splits > 0 && mid_band_splits <= SPLITS && scanned <= mid_band_hi_total)
            ? mid_band_splits
            : SPLITS;
    if (split >= s_eff) return;
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

    const int stride = s_eff * kGqaWarps;
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
    if (ticket != s_eff - 1) return;
    __threadfence();

    const float* base = scratch + (size_t)kvh * SPLITS * GROUP * (kGqaHD + 2);
    __shared__ float ssc[GROUP][SPLITS];
    __shared__ float sinv_l[GROUP];
    if (warp == 0 && lane < GROUP) {
        const int g = lane;
        float m_glob = -INFINITY;
        for (int s = 0; s < s_eff; ++s)
            m_glob = fmaxf(m_glob, base[((size_t)s * GROUP + g) * (kGqaHD + 2)]);
        float l_glob = 0.0f;
        for (int s = 0; s < s_eff; ++s) {
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
        for (int s = 0; s < s_eff; ++s)
            a += base[((size_t)s * GROUP + g) * (kGqaHD + 2) + 2 + d] * ssc[g][s];
        outp[(size_t)(kvh * GROUP + g) * kGqaHD + d] = __float2bfloat16(a * sinv_l[g]);
    }
    if (threadIdx.x == 0) fan_in[kvh] = 0u;
}

constexpr int kLagFlashRegdietMinCtasPerSm3Needs85RegsVsShipped94RegsAt2CtasSmemAlreadyFits3 = 3;

template <int GROUP, int SPLITS, int KVQ>
__global__ void laguna_flash_decode_verify_kernel(
    const __nv_bfloat16* q,
    const uint8_t* k,
    const uint8_t* v,
    const float* k_scales,
    const float* v_scales,
    __nv_bfloat16* out,
    const int* committed,
    float* scratch,
    unsigned int* fan_in,
    int n_kv,
    int rows,
    int capacity,
    float scale
) {
    const int row = blockIdx.z;
    const size_t row_elems = (size_t)n_kv * GROUP * kGqaHD;
    if (committed[0] < 0 || committed[0] > capacity - rows) {
        if (blockIdx.y == 0) {
            for (int i = threadIdx.x; i < GROUP * kGqaHD; i += blockDim.x)
                out[row * row_elems + (size_t)blockIdx.x * GROUP * kGqaHD + i] =
                    __float2bfloat16(NAN);
        }
        return;
    }
    laguna_flash_decode_gqa_body<GROUP, SPLITS, 0, KVQ, true>(
        q + row * row_elems,
        reinterpret_cast<const __nv_bfloat16*>(k),
        reinterpret_cast<const __nv_bfloat16*>(v),
        out + row * row_elems,
        committed, row + 1,
        scratch + (size_t)row * n_kv * SPLITS * GROUP * (kGqaHD + 2),
        fan_in + (size_t)row * n_kv,
        n_kv, 0, scale, 0, 0, k_scales, v_scales);
}

constexpr int kLagFlashPipe2Stages3Is2KvRowsInFlightPerWarpVsShipped1DeepRegisterPrefetch = 3;

constexpr int kLagFlashVariantsCoverOnlyGroup6TheDeepFullAttnKernelGroup8IsSmemCappedAt2Ctas = 6;

template <int GROUP, int SPLITS>
__global__ void __launch_bounds__(
    kGqaThreads, kLagFlashRegdietMinCtasPerSm3Needs85RegsVsShipped94RegsAt2CtasSmemAlreadyFits3)
laguna_flash_decode_gqa_regdiet3_kernel(
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
    float scale,
    int mid_band_splits,
    int mid_band_hi_total
) {
    laguna_flash_decode_gqa_body<GROUP, SPLITS, 0>(
        q, k, v, outp, total_ptr, delta, scratch, fan_in, NKV, WINDOW, scale,
        mid_band_splits, mid_band_hi_total);
}

template <int GROUP, int SPLITS>
__global__ void laguna_flash_decode_gqa_pipe2_kernel(
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
    float scale,
    int mid_band_splits,
    int mid_band_hi_total
) {
    laguna_flash_decode_gqa_body<
        GROUP, SPLITS, kLagFlashPipe2Stages3Is2KvRowsInFlightPerWarpVsShipped1DeepRegisterPrefetch>(
        q, k, v, outp, total_ptr, delta, scratch, fan_in, NKV, WINDOW, scale,
        mid_band_splits, mid_band_hi_total);
}

constexpr bool
    kLagFlashRegdietPlusPipe2LaunchesPipe2Its72RegsAlreadyClearThe85RegCapAndATwinBoundsCopyOfOneBodyTemplateTripsNvlinkSmemChecksOnForeignOversizedKernels =
        true;

inline bool laguna_flash_regdiet_env_read_live_each_launch_so_the_bitwise_arm_gate_can_flip_in_process() {
    const char* e = getenv("NV_LAG_FLASH_REGDIET");
    return e != nullptr && e[0] == '1';
}

inline bool laguna_flash_pipe2_env_read_live_each_launch_so_the_bitwise_arm_gate_can_flip_in_process() {
    const char* e = getenv("NV_LAG_FLASH_PIPE2");
    return e != nullptr && e[0] == '1';
}

template <int GROUP, int SPLITS>
void laguna_flash_gqa_launch(
    dim3 grid,
    cudaStream_t s,
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* outp,
    const int* total_ptr,
    int delta,
    float* scratch,
    unsigned int* fan_in,
    int n_kv,
    int window,
    float scale,
    int mid_band_splits,
    int mid_band_hi_total
) {
    if constexpr (GROUP == kLagFlashVariantsCoverOnlyGroup6TheDeepFullAttnKernelGroup8IsSmemCappedAt2Ctas) {
        const bool rd = laguna_flash_regdiet_env_read_live_each_launch_so_the_bitwise_arm_gate_can_flip_in_process();
        const bool p2 = laguna_flash_pipe2_env_read_live_each_launch_so_the_bitwise_arm_gate_can_flip_in_process();
        if (p2 &&
            kLagFlashRegdietPlusPipe2LaunchesPipe2Its72RegsAlreadyClearThe85RegCapAndATwinBoundsCopyOfOneBodyTemplateTripsNvlinkSmemChecksOnForeignOversizedKernels) {
            laguna_flash_decode_gqa_pipe2_kernel<GROUP, SPLITS><<<grid, kGqaThreads, 0, s>>>(
                q, k, v, outp, total_ptr, delta, scratch, fan_in, n_kv, window, scale,
                mid_band_splits, mid_band_hi_total);
            return;
        }
        if (rd) {
            laguna_flash_decode_gqa_regdiet3_kernel<GROUP, SPLITS><<<grid, kGqaThreads, 0, s>>>(
                q, k, v, outp, total_ptr, delta, scratch, fan_in, n_kv, window, scale,
                mid_band_splits, mid_band_hi_total);
            return;
        }
    }
    laguna_flash_decode_gqa_kernel<GROUP, SPLITS><<<grid, kGqaThreads, 0, s>>>(
        q, k, v, outp, total_ptr, delta, scratch, fan_in, n_kv, window, scale,
        mid_band_splits, mid_band_hi_total);
}

constexpr int kLagunaM1SplitsDefault16IsShallowTunedAndIs128WorkgroupsAt8KvHeads = 16;

constexpr int kLagunaM1DepthgateDeepSplits47Is376WorkgroupsTwoPerSmAt8KvHeadsOn188Sms = 47;

constexpr int kLagunaM1DepthgateShallow16WinsThroughScanned16384And47WinsFrom32768Up = 16384;

inline bool laguna_m1_split_is_instantiated(int x) {
    return x == 8 || x == 16 || x == 24 || x == 32 || x == 47 || x == 64;
}

inline int laguna_m1_splits_env() {
    static int v = [] {
        const char* e = getenv("NV_LAGUNA_M1_SPLITS");
        if (e == nullptr)
            return kLagunaM1SplitsDefault16IsShallowTunedAndIs128WorkgroupsAt8KvHeads;
        int x = atoi(e);
        return laguna_m1_split_is_instantiated(x)
                   ? x
                   : kLagunaM1SplitsDefault16IsShallowTunedAndIs128WorkgroupsAt8KvHeads;
    }();
    return v;
}

inline int laguna_m1_depthgate_deep_splits_env() {
    static int v = [] {
        const char* e = getenv("NV_LAGUNA_M1_DEPTHGATE_SPLITS");
        if (e == nullptr) return 0;
        int x = atoi(e);
        return laguna_m1_split_is_instantiated(x) ? x : 0;
    }();
    return v;
}

inline int laguna_m1_depthgate_knee_env() {
    static int v = [] {
        const char* e = getenv("NV_LAGUNA_M1_DEPTHGATE_KNEE");
        if (e == nullptr) return 0;
        int x = atoi(e);
        return (x > 0) ? x : 0;
    }();
    return v;
}

inline int laguna_m1_launch_splits() {
    int deep = laguna_m1_depthgate_deep_splits_env();
    if (deep == 0)
        deep = kLagunaM1DepthgateDeepSplits47Is376WorkgroupsTwoPerSmAt8KvHeadsOn188Sms;
    int shallow = laguna_m1_splits_env();
    return (deep > shallow) ? deep : shallow;
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

static int laguna_launch_qk_normrope_rows(
    void* stream,
    const uint16_t* q_in,
    const uint16_t* k_in,
    uint16_t* q_out,
    uint16_t* k_out,
    const void* qw,
    const void* kw,
    const float* qs,
    const float* ks,
    const float* cos_tbl,
    const float* sin_tbl,
    const int* pos_base,
    int n_q,
    int n_kv,
    int head_dim,
    int rotary_dim,
    float rot_scale,
    float eps_q,
    float eps_k,
    int rows
) {
    if (n_q <= 0 || n_kv <= 0 || rows < 1 || rows > 8) return -1;
    if (head_dim < 32 || head_dim > 256 || (head_dim & 31)) return -2;
    if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    laguna_qk_normrope_bf16_kernel<false><<<dim3((unsigned)(n_q + n_kv), (unsigned)rows), kNormBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_in),
        reinterpret_cast<const __nv_bfloat16*>(k_in),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        qw, kw, qs, ks,
        cos_tbl,
        sin_tbl,
        pos_base,
        n_q,
        n_kv,
        head_dim,
        rotary_dim,
        rot_scale,
        eps_q,
        eps_k
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_laguna_qk_normrope_bf16_rows(
    void* stream,
    const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out, uint16_t* k_out,
    const uint16_t* qw, const uint16_t* kw, const float* cos_tbl, const float* sin_tbl,
    const int* positions, int n_q, int n_kv, int head_dim, int rotary_dim,
    float rot_scale, float eps_q, float eps_k, int rows
) {
    return laguna_launch_qk_normrope_rows(
        stream, q_in, k_in, q_out, k_out, qw, kw, nullptr, nullptr, cos_tbl, sin_tbl,
        positions, n_q, n_kv, head_dim, rotary_dim, rot_scale, eps_q, eps_k, rows);
}

extern "C" int nv_kernels_laguna_qk_normrope_fp8_weight_rows(
    void* stream,
    const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out, uint16_t* k_out,
    const uint8_t* qw, const float* qs, const uint8_t* kw, const float* ks,
    const float* cos_tbl, const float* sin_tbl, const int* positions,
    int n_q, int n_kv, int head_dim, int rotary_dim,
    float rot_scale, float eps_q, float eps_k, int rows
) {
    if (!q_in || !k_in || !q_out || !k_out || !qw || !kw || !qs || !ks || !cos_tbl || !sin_tbl || !positions) return -1;
    return laguna_launch_qk_normrope_rows(
        stream, q_in, k_in, q_out, k_out, qw, kw, qs, ks, cos_tbl, sin_tbl,
        positions, n_q, n_kv, head_dim, rotary_dim, rot_scale, eps_q, eps_k, rows);
}

extern "C" int nv_kernels_laguna_qk_normrope_bf16(
    void* stream,
    const uint16_t* q_in, const uint16_t* k_in, uint16_t* q_out, uint16_t* k_out,
    const uint16_t* qw, const uint16_t* kw, const float* cos_tbl, const float* sin_tbl,
    const int* pos_base, int n_q, int n_kv, int head_dim, int rotary_dim,
    float rot_scale, float eps_q, float eps_k
) {
    return nv_kernels_laguna_qk_normrope_bf16_rows(
        stream, q_in, k_in, q_out, k_out, qw, kw, cos_tbl, sin_tbl, pos_base,
        n_q, n_kv, head_dim, rotary_dim, rot_scale, eps_q, eps_k, 1);
}

extern "C" int nv_kernels_laguna_flash_decode_gqa_scratch_elems(int n_kv) {
    int sp = laguna_m1_launch_splits();
    if (sp < 32) sp = 32;
    return n_kv * sp * 8 * (128 + 2);
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
    const int sp = laguna_m1_launch_splits();
    const int mid_band_splits = laguna_m1_splits_env();
    const int mid_band_hi_total =
        (laguna_m1_depthgate_knee_env() > 0)
            ? laguna_m1_depthgate_knee_env()
            : kLagunaM1DepthgateShallow16WinsThroughScanned16384And47WinsFrom32768Up;
    dim3 grid((unsigned)n_kv, (unsigned)sp);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(v);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
#define NV_LAG_FLASH_CASE(G, S)                                                          \
    case (G) * 100 + (S):                                                                \
        laguna_flash_gqa_launch<G, S>(grid, s, qb, kb, vb, ob, total_ptr, delta,         \
                                      scratch, fan_in, n_kv, window, scale,              \
                                      mid_band_splits, mid_band_hi_total);               \
        break;
    switch (group * 100 + sp) {
        NV_LAG_FLASH_CASE(1, 8)
        NV_LAG_FLASH_CASE(1, 16)
        NV_LAG_FLASH_CASE(1, 24)
        NV_LAG_FLASH_CASE(1, 32)
        NV_LAG_FLASH_CASE(1, 47)
        NV_LAG_FLASH_CASE(1, 64)
        NV_LAG_FLASH_CASE(6, 8)
        NV_LAG_FLASH_CASE(6, 16)
        NV_LAG_FLASH_CASE(6, 24)
        NV_LAG_FLASH_CASE(6, 32)
        NV_LAG_FLASH_CASE(6, 47)
        NV_LAG_FLASH_CASE(6, 64)
        NV_LAG_FLASH_CASE(8, 8)
        NV_LAG_FLASH_CASE(8, 16)
        NV_LAG_FLASH_CASE(8, 24)
        NV_LAG_FLASH_CASE(8, 32)
        NV_LAG_FLASH_CASE(8, 47)
        NV_LAG_FLASH_CASE(8, 64)
        default:
            return -1;
    }
#undef NV_LAG_FLASH_CASE
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_laguna_flash_decode_gqa_rows(
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
    int rows,
    int window,
    float scale
) {
    if (n_q <= 0 || n_kv <= 0 || head_dim != kGqaHD || n_q % n_kv != 0 || rows < 1 || rows > 8
        || !isfinite(scale) || q == nullptr || k == nullptr || v == nullptr || out == nullptr
        || total_ptr == nullptr || scratch == nullptr || fan_in == nullptr)
        return -1;
    const int group = n_q / n_kv;
    if (group != 8) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int sp = laguna_m1_launch_splits();
    const int mid_band_splits = laguna_m1_splits_env();
    const int mid_band_hi_total =
        (laguna_m1_depthgate_knee_env() > 0)
            ? laguna_m1_depthgate_knee_env()
            : kLagunaM1DepthgateShallow16WinsThroughScanned16384And47WinsFrom32768Up;
    dim3 grid((unsigned)n_kv, (unsigned)sp, (unsigned)rows);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(v);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
#define NV_LAG_FLASH_ROWS_CASE(S) \
    case S: \
        laguna_flash_decode_gqa_kernel<8, S><<<grid, kGqaThreads, 0, s>>>( \
            qb, kb, vb, ob, total_ptr, delta, scratch, fan_in, n_kv, window, scale, \
            mid_band_splits, mid_band_hi_total); \
        break;
    switch (sp) {
        NV_LAG_FLASH_ROWS_CASE(8)
        NV_LAG_FLASH_ROWS_CASE(16)
        NV_LAG_FLASH_ROWS_CASE(24)
        NV_LAG_FLASH_ROWS_CASE(32)
        NV_LAG_FLASH_ROWS_CASE(47)
        NV_LAG_FLASH_ROWS_CASE(64)
        default: return -1;
    }
#undef NV_LAG_FLASH_ROWS_CASE
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_laguna_flash_decode_verify(
    void* stream,
    const uint16_t* q,
    const uint8_t* k,
    const uint8_t* v,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* committed,
    float* scratch,
    unsigned int* fan_in,
    int n_q,
    int n_kv,
    int head_dim,
    int rows,
    int capacity,
    int splits,
    float scale,
    int kv_is_fp8
) {
    if (n_q <= 0 || n_kv <= 0 || head_dim != kGqaHD || n_q % n_kv != 0
        || rows < 1 || rows > 32 || capacity < rows || !isfinite(scale)
        || (kv_is_fp8 != kLagKvBf16 && kv_is_fp8 != kLagKvE4m3PerTokenHeadScale && kv_is_fp8 != kLagKvInt8PerTokenHeadScale)
        || q == nullptr || k == nullptr || v == nullptr
        || (kv_is_fp8 && (k_scales == nullptr || v_scales == nullptr)) || out == nullptr || committed == nullptr
        || scratch == nullptr || fan_in == nullptr)
        return -1;
    const int group = n_q / n_kv;
    if ((group != 6 && group != 8) || (splits != 32 && splits != 64)) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)n_kv, (unsigned)splits, (unsigned)rows);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
#define NV_LAG_FLASH_VERIFY_CASE(G, S) \
    case (G) * 100 + (S): \
        if (kv_is_fp8 == kLagKvInt8PerTokenHeadScale) { \
            laguna_flash_decode_verify_kernel<G, S, kLagKvInt8PerTokenHeadScale><<<grid, kGqaThreads, 0, s>>>( \
                qb, k, v, k_scales, v_scales, ob, committed, scratch, fan_in, \
                n_kv, rows, capacity, scale); \
        } else if (kv_is_fp8 == kLagKvE4m3PerTokenHeadScale) { \
            laguna_flash_decode_verify_kernel<G, S, kLagKvE4m3PerTokenHeadScale><<<grid, kGqaThreads, 0, s>>>( \
                qb, k, v, k_scales, v_scales, ob, committed, scratch, fan_in, \
                n_kv, rows, capacity, scale); \
        } else { \
            laguna_flash_decode_verify_kernel<G, S, kLagKvBf16><<<grid, kGqaThreads, 0, s>>>( \
                qb, k, v, nullptr, nullptr, ob, committed, scratch, fan_in, \
                n_kv, rows, capacity, scale); \
        } \
        break;
    switch (group * 100 + splits) {
        NV_LAG_FLASH_VERIFY_CASE(6, 32)
        NV_LAG_FLASH_VERIFY_CASE(6, 64)
        NV_LAG_FLASH_VERIFY_CASE(8, 32)
        NV_LAG_FLASH_VERIFY_CASE(8, 64)
        default: return -1;
    }
#undef NV_LAG_FLASH_VERIFY_CASE
    return (int)cudaGetLastError();
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

constexpr int kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp = 8;
constexpr int kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock = 8;
constexpr int kLagRowsGemvThreads = kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock * 32;
constexpr int kLagRowsGemvMaxDynamicSmemBytes = kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp * 6144 * 2;

__device__ __forceinline__ void laguna_rows_gemv_warp_dot(
    const __nv_bfloat16* __restrict__ w_row,
    const __nv_bfloat16* __restrict__ xs_rows,
    int K,
    int rows,
    int lane,
    float* acc
) {
    const int Kv = K / 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(w_row);
    for (int v = lane; v < Kv; v += 32) {
        uint4 pw = __ldg(&w4[v]);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
        const int kb = v * 8;
        for (int r = 0; r < rows; ++r) {
            const __nv_bfloat16* xr = xs_rows + (size_t)r * K + kb;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 wf = __bfloat1622float2(wp[j]);
                acc[r] += wf.x * __bfloat162float(xr[2 * j]) + wf.y * __bfloat162float(xr[2 * j + 1]);
            }
        }
    }
    for (int r = 0; r < rows; ++r) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
    }
}

__global__ void __launch_bounds__(kLagRowsGemvThreads)
laguna_gemv_bf16_qkvg_normed_rows_kernel(
    const __nv_bfloat16* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ Wk,
    const __nv_bfloat16* __restrict__ Wv,
    const __nv_bfloat16* __restrict__ Wg,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ yq,
    __nv_bfloat16* __restrict__ yk,
    __nv_bfloat16* __restrict__ yv,
    __nv_bfloat16* __restrict__ yg,
    int Nq,
    int Nk,
    int Nv,
    int Ng,
    int K,
    int rows
) {
    extern __shared__ __align__(16) __nv_bfloat16 lag_rows_gemv_xs[];
    for (int i = threadIdx.x; i < rows * K; i += blockDim.x) {
        const int r = i / K;
        const int k = i - r * K;
        lag_rows_gemv_xs[i] = __float2bfloat16(__bfloat162float(x[(size_t)r * K + k]) * rstd[r] * __bfloat162float(wn[k]));
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock + warp;
    const __nv_bfloat16* W;
    __nv_bfloat16* y;
    int stride;
    int local = n;
    if (n < Nq) { W = Wq; y = yq; stride = Nq; }
    else if ((local -= Nq) < Nk) { W = Wk; y = yk; stride = Nk; }
    else if ((local -= Nk) < Nv) { W = Wv; y = yv; stride = Nv; }
    else if ((local -= Nv) < Ng) { W = Wg; y = yg; stride = Ng; }
    else return;
    float acc[kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    laguna_rows_gemv_warp_dot(W + (size_t)local * K, lag_rows_gemv_xs, K, rows, lane, acc);
    if (lane == 0) {
        for (int r = 0; r < rows; ++r) y[(size_t)r * stride + local] = __float2bfloat16(acc[r]);
    }
}

__global__ void __launch_bounds__(kLagRowsGemvThreads)
laguna_gemv_bf16_rows_kernel(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    int rows
) {
    extern __shared__ __align__(16) __nv_bfloat16 lag_rows_gemv_xs[];
    for (int i = threadIdx.x; i < rows * K; i += blockDim.x) lag_rows_gemv_xs[i] = x[i];
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = blockIdx.x * kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock + warp;
    if (n >= N) return;
    float acc[kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    laguna_rows_gemv_warp_dot(W + (size_t)n * K, lag_rows_gemv_xs, K, rows, lane, acc);
    if (lane == 0) {
        for (int r = 0; r < rows; ++r) y[(size_t)r * N + n] = __float2bfloat16(acc[r]);
    }
}

static bool laguna_rows_gemv_smem_attr_set(const void* fn, bool* done) {
    if (!*done) *done = cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, kLagRowsGemvMaxDynamicSmemBytes) == cudaSuccess;
    return *done;
}

extern "C" int nv_kernels_laguna_gemv_bf16_qkvg_normed_rows(
    void* stream, const uint16_t* Wq, const uint16_t* Wk, const uint16_t* Wv, const uint16_t* Wg,
    const uint16_t* x, const uint16_t* wn, const float* rstd,
    uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg,
    int Nq, int Nk, int Nv, int Ng, int K, int rows
) {
    if (Nq < 0 || Nk < 0 || Nv < 0 || Ng < 0 || Nq + Nk + Nv + Ng <= 0 || K <= 0 || (K & 7) != 0 || K > 4096) return -1;
    if ((Nq > 0 && Wq == nullptr) || (Nk > 0 && Wk == nullptr) || (Nv > 0 && Wv == nullptr) || (Ng > 0 && Wg == nullptr)) return -1;
    if (rows < 1 || rows > kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp) return -1;
    static bool attr = false;
    if (!laguna_rows_gemv_smem_attr_set((const void*)laguna_gemv_bf16_qkvg_normed_rows_kernel, &attr)) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    const int total = Nq + Nk + Nv + Ng;
    dim3 grid((unsigned)((total + kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock - 1) / kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock));
    laguna_gemv_bf16_qkvg_normed_rows_kernel<<<grid, kLagRowsGemvThreads, (size_t)rows * K * 2, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(Wq), reinterpret_cast<const __nv_bfloat16*>(Wk),
        reinterpret_cast<const __nv_bfloat16*>(Wv), reinterpret_cast<const __nv_bfloat16*>(Wg),
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(wn), rstd,
        reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk),
        reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yg), Nq, Nk, Nv, Ng, K, rows);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_laguna_gemv_bf16_rows(void* stream, const uint16_t* W, const uint16_t* x, uint16_t* y, int N, int K, int rows) {
    if (N <= 0 || K <= 0 || (K & 7) != 0 || K > 6144) return -1;
    if (rows < 1 || rows > kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp) return -1;
    static bool attr = false;
    if (!laguna_rows_gemv_smem_attr_set((const void*)laguna_gemv_bf16_rows_kernel, &attr)) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((N + kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock - 1) / kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock));
    laguna_gemv_bf16_rows_kernel<<<grid, kLagRowsGemvThreads, (size_t)rows * K * 2, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(W), reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y), N, K, rows);
    return (int)cudaGetLastError();
}

__device__ __forceinline__ int laguna_rows_q8_shared_pair(int p) { return p ^ ((p >> 5) & 7); }
constexpr int kLagRowsQ8LoadsInFlightPerLane4CoversK2048InOneGroup = 4;
constexpr int kLagRowsQ8MaxDynamicSmemBytes98304FitsTheSm120PerBlockOptin = 98304;
constexpr int kLagRowsQ8MaxK8192CoversTheSlidingLayersOProj = 8192;
constexpr int kLagFlashVerifyRowsPerBlockMax3KeepsRowsTimesGroupAccumulatorsInRegisters = 3;


template <bool kFp8Rows>
__device__ __forceinline__ void laguna_rows_q8_dequant16(const int4& raw, float* wf) {
    if (kFp8Rows) {
        const unsigned short* p2 = reinterpret_cast<const unsigned short*>(&raw);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            const float2 f = laguna_fp8x2_to_float2(p2[t]);
            wf[2 * t] = f.x;
            wf[2 * t + 1] = f.y;
        }
    } else {
        const unsigned* wu = reinterpret_cast<const unsigned*>(&raw);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned u = wu[t] ^ 0x80808080u;
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                wf[4 * t + i] = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650u + i)) - 8388736.0f;
        }
    }
}

constexpr int kLagRowsQ8XStageF32PairsCostOneLds64PerPairInsteadOfLds32PlusTwoUnpacks = 1;

__device__ __forceinline__ int laguna_rows_q8_f32_pair_slot_p_major_keeps_consecutive_lanes_on_consecutive_pairs(int p_global, int Kv) {
    return (p_global & 7) * Kv + (p_global >> 3);
}

constexpr int kLagRowsQ8OutputsPerWarp2HalvesTheBlockWavesAtTheSameLoadsInFlightPerLane = 2;

template <bool kFp8Rows, int ROWS, bool kXF32, int NPW>
__device__ __forceinline__ void laguna_rows_q8_warp_dot_loads_in_flight_same_order_as_gemv_e4m3_mk_h(
    const uint8_t* __restrict__ w_row0, const uint8_t* __restrict__ w_row1, const void* __restrict__ xs_any, int row_pairs, int Kv, int lane, float* acc
) {
    constexpr int PF = kLagRowsQ8LoadsInFlightPerLane4CoversK2048InOneGroup / NPW;
    static_assert(NPW == 1 || NPW == 2, "one or two output rows per warp");
    const int4* w16[NPW];
    w16[0] = reinterpret_cast<const int4*>(w_row0);
    if (NPW == 2) w16[NPW - 1] = reinterpret_cast<const int4*>(w_row1);
    const __nv_bfloat162* xs16 = reinterpret_cast<const __nv_bfloat162*>(xs_any);
    const float2* xs32 = reinterpret_cast<const float2*>(xs_any);
    for (int v0 = lane; v0 < Kv; v0 += 32 * PF) {
        int4 raw[NPW][PF];
        #pragma unroll
        for (int i = 0; i < PF; ++i) {
            const int v = v0 + i * 32;
            #pragma unroll
            for (int o = 0; o < NPW; ++o) raw[o][i] = (v < Kv) ? __ldcs(&w16[o][v]) : make_int4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int i = 0; i < PF; ++i) {
            const int v = v0 + i * 32;
            if (v < Kv) {
                float wf[NPW][16];
                #pragma unroll
                for (int o = 0; o < NPW; ++o) laguna_rows_q8_dequant16<kFp8Rows>(raw[o][i], wf[o]);
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    if (kXF32) {
                        const float2* xp = xs32 + r * row_pairs + v;
                        #pragma unroll
                        for (int p = 0; p < 8; ++p) {
                            const float2 xv = xp[p * Kv];
                            #pragma unroll
                            for (int o = 0; o < NPW; ++o) acc[o * ROWS + r] += wf[o][2 * p] * xv.x + wf[o][2 * p + 1] * xv.y;
                        }
                    } else {
                        const __nv_bfloat162* xp = xs16 + r * row_pairs + v * 8;
                        const int bank_xor = (v >> 2) & 7;
                        #pragma unroll
                        for (int p = 0; p < 8; ++p) {
                            const float2 xv = __bfloat1622float2(xp[p ^ bank_xor]);
                            #pragma unroll
                            for (int o = 0; o < NPW; ++o) acc[o * ROWS + r] += wf[o][2 * p] * xv.x + wf[o][2 * p + 1] * xv.y;
                        }
                    }
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < NPW * ROWS; ++i) {
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) acc[i] += __shfl_xor_sync(0xffffffffu, acc[i], offset);
    }
}

template <bool kXF32>
__device__ __forceinline__ void laguna_rows_q8_stage_pair(void* xs_any, int r, int row_pairs, int p_global, int Kv, __nv_bfloat162 val) {
    if (kXF32) {
        reinterpret_cast<float2*>(xs_any)[r * row_pairs + laguna_rows_q8_f32_pair_slot_p_major_keeps_consecutive_lanes_on_consecutive_pairs(p_global, Kv)] = __bfloat1622float2(val);
    } else {
        reinterpret_cast<__nv_bfloat162*>(xs_any)[r * row_pairs + laguna_rows_q8_shared_pair(p_global)] = val;
    }
}

template <bool kXF32>
__device__ __forceinline__ float2 laguna_rows_q8_load_pair(const void* xs_any, int r, int row_pairs, int p_global, int Kv) {
    if (kXF32) {
        return reinterpret_cast<const float2*>(xs_any)[r * row_pairs + laguna_rows_q8_f32_pair_slot_p_major_keeps_consecutive_lanes_on_consecutive_pairs(p_global, Kv)];
    }
    return __bfloat1622float2(reinterpret_cast<const __nv_bfloat162*>(xs_any)[r * row_pairs + laguna_rows_q8_shared_pair(p_global)]);
}

__device__ __forceinline__ float laguna_rows_q8_block_sum256_same_tree_as_ls_block_sum256(float v, float* warp_sums, float* total) {
    constexpr int kWarp = 32;
    constexpr int kWarps = kNormBlock / kWarp;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) *total = s;
    }
    __syncthreads();
    const float out = *total;
    __syncthreads();
    return out;
}

template <bool kFp8Rows, int ROWS, bool kXF32, int NPW>
__global__ void __launch_bounds__(kLagRowsGemvThreads)
laguna_gemv_q8_qkvg_normed_rows_kernel(
    const uint8_t* __restrict__ Wq, const float* __restrict__ Sq,
    const uint8_t* __restrict__ Wk, const float* __restrict__ Sk,
    const uint8_t* __restrict__ Wv, const float* __restrict__ Sv,
    const void* __restrict__ Wg, const float* __restrict__ Sg,
    const __nv_bfloat16* __restrict__ x,
    const void* __restrict__ wn,
    const float* __restrict__ Sn,
    const float* __restrict__ rstd,
    float eps_when_rstd_is_null_and_the_block_computes_it_like_laguna_rstd256_bf16,
    __nv_bfloat16* __restrict__ yq, __nv_bfloat16* __restrict__ yk, __nv_bfloat16* __restrict__ yv, __nv_bfloat16* __restrict__ yg,
    int Nq, int Nk, int Nv, int Ng, int K
) {
    static_assert(kLagRowsGemvThreads == kNormBlock, "THE_IN_KERNEL_RSTD_REPEATS_laguna_rstd256_bf16_THREAD_FOR_THREAD");
    extern __shared__ __align__(16) unsigned char lag_rows_q8_xsg_bytes[];
    void* lag_rows_q8_xsg = lag_rows_q8_xsg_bytes;
    __shared__ float lag_rows_q8_rstd_warp_sums[kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock];
    __shared__ float lag_rows_q8_rstd_total;
    const int row_pairs = (K >> 1);
    const int chunks8 = K >> 3;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    const uint4* wn4 = reinterpret_cast<const uint4*>(wn);
    float rr[ROWS];
    if (rstd == nullptr) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const __nv_bfloat16* xr = x + (size_t)r * K;
            float local = 0.0f;
            for (int i = threadIdx.x; i < K; i += kNormBlock) {
                float v = __bfloat162float(xr[i]);
                local += v * v;
            }
            const float sum = laguna_rows_q8_block_sum256_same_tree_as_ls_block_sum256(local, lag_rows_q8_rstd_warp_sums, &lag_rows_q8_rstd_total);
            rr[r] = rsqrtf(sum / (float)K + eps_when_rstd_is_null_and_the_block_computes_it_like_laguna_rstd256_bf16);
        }
    } else {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) rr[r] = rstd[r];
    }
    for (int c = threadIdx.x; c < chunks8; c += blockDim.x) {
        float2 norm_weight[4];
        if (Sn != nullptr) {
            const uint2 raw = __ldg(reinterpret_cast<const uint2*>(wn) + c);
            const unsigned short* packed = reinterpret_cast<const unsigned short*>(&raw);
            const float scale = __ldg(Sn);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float2 w = laguna_fp8x2_to_float2(packed[j]);
                norm_weight[j] = __bfloat1622float2(__floats2bfloat162_rn(w.x * scale, w.y * scale));
            }
        } else {
            const uint4 raw = __ldg(&wn4[c]);
            const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
            #pragma unroll
            for (int j = 0; j < 4; ++j) norm_weight[j] = __bfloat1622float2(w2[j]);
        }
        uint4 xraw[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) xraw[r] = __ldg(&x4[(size_t)r * chunks8 + c]);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const __nv_bfloat162* xx = reinterpret_cast<const __nv_bfloat162*>(&xraw[r]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int p = c * 4 + j;
                const float a = __bfloat162float(xx[j].x) * rr[r] * norm_weight[j].x;
                const float b = __bfloat162float(xx[j].y) * rr[r] * norm_weight[j].y;
                laguna_rows_q8_stage_pair<kXF32>(lag_rows_q8_xsg, r, row_pairs, p, K >> 4, __floats2bfloat162_rn(a, b));
            }
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = (blockIdx.x * kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock + warp) * NPW;
    const uint8_t* Wb = nullptr;
    const float* S = nullptr;
    __nv_bfloat16* y = nullptr;
    int stride = 0;
    int local = n;
    float acc[NPW * ROWS];
    #pragma unroll
    for (int i = 0; i < NPW * ROWS; ++i) acc[i] = 0.0f;
    if (n < Nq) { Wb = Wq; S = Sq; y = yq; stride = Nq; }
    else if ((local -= Nq) < Nk) { Wb = Wk; S = Sk; y = yk; stride = Nk; }
    else if ((local -= Nk) < Nv) { Wb = Wv; S = Sv; y = yv; stride = Nv; }
    else if ((local -= Nv) >= Ng) { return; }
    else if (Sg != nullptr) { Wb = reinterpret_cast<const uint8_t*>(Wg); S = Sg; y = yg; stride = Ng; }
    else {
        #pragma unroll
        for (int o = 0; o < NPW; ++o) {
            const int lo = local + o;
            if (lo >= Ng) break;
            const uint4* w4 = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(Wg) + (size_t)lo * K);
            const int Kg = K / 8;
            float* ao = acc + o * ROWS;
            for (int v = lane; v < Kg; v += 32) {
                const uint4 pw = __ldg(&w4[v]);
                const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
                const int pb = v * 4;
                #pragma unroll
                for (int r = 0; r < ROWS; ++r) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float2 wf = __bfloat1622float2(wp[j]);
                        const int p = pb + j;
                        const float2 xv = laguna_rows_q8_load_pair<kXF32>(lag_rows_q8_xsg, r, row_pairs, p, K >> 4);
                        ao[r] += wf.x * xv.x + wf.y * xv.y;
                    }
                }
            }
            #pragma unroll
            for (int r = 0; r < ROWS; ++r) {
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) ao[r] += __shfl_xor_sync(0xffffffffu, ao[r], offset);
                if (lane == 0) yg[(size_t)r * Ng + lo] = __float2bfloat16(ao[r]);
            }
        }
        return;
    }
    const int last = (NPW == 2 && local + 1 < stride) ? 1 : 0;
    laguna_rows_q8_warp_dot_loads_in_flight_same_order_as_gemv_e4m3_mk_h<kFp8Rows, ROWS, kXF32, NPW>(
        Wb + (size_t)local * K, Wb + (size_t)(local + last) * K, lag_rows_q8_xsg, row_pairs, K / 16, lane, acc);
    #pragma unroll
    for (int o = 0; o < NPW; ++o) {
        if (o > last) break;
        const float rs = __ldg(&S[local + o]);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (lane == 0) y[(size_t)r * stride + local + o] = __float2bfloat16(acc[o * ROWS + r] * rs);
        }
    }
}

constexpr int kLagRowsQ8GateHeadsMax64CoversTheSlidingLayers = 64;

__device__ __forceinline__ float laguna_softplus_gate_value_same_as_softplus_gate_exact_bf16(float x) {
    const float sp = fmaxf(x, 0.f) + logf(1.0f + expf(-fabsf(x)));
    return __bfloat162float(__float2bfloat16(sp));
}

template <bool kFp8Rows, int ROWS, bool kGate, bool kXF32, int NPW>
__global__ void __launch_bounds__(kLagRowsGemvThreads)
laguna_gemv_q8_rows_kernel(
    const uint8_t* __restrict__ W, const float* __restrict__ S,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    int gate_heads, int gate_hd,
    __nv_bfloat16* __restrict__ y,
    int N, int K
) {
    extern __shared__ __align__(16) unsigned char lag_rows_q8_xs_bytes[];
    void* lag_rows_q8_xs = lag_rows_q8_xs_bytes;
    __shared__ float gv[ROWS][kLagRowsQ8GateHeadsMax64CoversTheSlidingLayers];
    const int row_pairs = (K >> 1);
    const int chunks8 = K >> 3;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    if (kGate) {
        for (int i = threadIdx.x; i < ROWS * gate_heads; i += blockDim.x) {
            const int r = i / gate_heads;
            const int h = i - r * gate_heads;
            gv[r][h] = laguna_softplus_gate_value_same_as_softplus_gate_exact_bf16(__bfloat162float(gate[(size_t)r * gate_heads + h]));
        }
        __syncthreads();
    }
    #pragma unroll 4
    for (int c = threadIdx.x; c < chunks8; c += blockDim.x) {
        uint4 xraw[ROWS];
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) xraw[r] = __ldg(&x4[(size_t)r * chunks8 + c]);
        const int head = kGate ? (c * 8) / gate_hd : 0;
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            const __nv_bfloat162* xx = reinterpret_cast<const __nv_bfloat162*>(&xraw[r]);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int p = c * 4 + j;
                __nv_bfloat162 val = xx[j];
                if (kGate) {
                    const float g = gv[r][head];
                    val = __floats2bfloat162_rn(__bfloat162float(xx[j].x) * g, __bfloat162float(xx[j].y) * g);
                }
                laguna_rows_q8_stage_pair<kXF32>(lag_rows_q8_xs, r, row_pairs, p, K >> 4, val);
            }
        }
    }
    __syncthreads();
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int n = (blockIdx.x * kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock + warp) * NPW;
    if (n >= N) return;
    const int last = (NPW == 2 && n + 1 < N) ? 1 : 0;
    float acc[NPW * ROWS];
    #pragma unroll
    for (int i = 0; i < NPW * ROWS; ++i) acc[i] = 0.0f;
    laguna_rows_q8_warp_dot_loads_in_flight_same_order_as_gemv_e4m3_mk_h<kFp8Rows, ROWS, kXF32, NPW>(
        W + (size_t)n * K, W + (size_t)(n + last) * K, lag_rows_q8_xs, row_pairs, K / 16, lane, acc);
    #pragma unroll
    for (int o = 0; o < NPW; ++o) {
        if (o > last) break;
        const float rs = __ldg(&S[n + o]);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            if (lane == 0) y[(size_t)r * N + n + o] = __float2bfloat16(acc[o * ROWS + r] * rs);
        }
    }
}

static int laguna_rows_q8_dynamic_smem_limit_bytes(const void* fn) {
    int dev = 0;
    int optin = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess) return 0;
    cudaFuncAttributes fa;
    if (cudaFuncGetAttributes(&fa, fn) != cudaSuccess) return 0;
    const int limit = optin - (int)fa.sharedSizeBytes;
    return limit < kLagRowsQ8MaxDynamicSmemBytes98304FitsTheSm120PerBlockOptin ? limit : kLagRowsQ8MaxDynamicSmemBytes98304FitsTheSm120PerBlockOptin;
}

static bool laguna_rows_q8_smem_attr_set(const void* fn, bool* done, int* limit_out) {
    if (!*done) {
        const int limit = laguna_rows_q8_dynamic_smem_limit_bytes(fn);
        const cudaError_t e = (limit > 0) ? cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, limit) : cudaErrorInvalidValue;
        *done = e == cudaSuccess;
        if (*done) *limit_out = limit;
        else { fprintf(stderr, "[laguna_rows_q8] cudaFuncSetAttribute(max dynamic smem %d) failed: %s\n", limit, cudaGetErrorString(e)); (void)cudaGetLastError(); }
    }
    return *done;
}

template <bool kFp8Rows, int ROWS, bool kXF32, int NPW>
static int laguna_launch_gemv_q8_qkvg_normed_rows_staged(
    cudaStream_t s, const void* Wq, const float* Sq, const void* Wk, const float* Sk, const void* Wv, const float* Sv,
    const void* Wg, const float* Sg, const uint16_t* x, const void* wn, const float* Sn, const float* rstd, float eps,
    uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg, int Nq, int Nk, int Nv, int Ng, int K, size_t smem
) {
    static bool attr = false;
    static int limit = 0;
    if (!laguna_rows_q8_smem_attr_set((const void*)laguna_gemv_q8_qkvg_normed_rows_kernel<kFp8Rows, ROWS, kXF32, NPW>, &attr, &limit)) return -1;
    if (smem > (size_t)limit) return -2;
    const int total = Nq + Nk + Nv + Ng;
    const int per_block = kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock * NPW;
    dim3 grid((unsigned)((total + per_block - 1) / per_block));
    laguna_gemv_q8_qkvg_normed_rows_kernel<kFp8Rows, ROWS, kXF32, NPW><<<grid, kLagRowsGemvThreads, smem, s>>>(
        reinterpret_cast<const uint8_t*>(Wq), Sq, reinterpret_cast<const uint8_t*>(Wk), Sk, reinterpret_cast<const uint8_t*>(Wv), Sv,
        Wg, Sg, reinterpret_cast<const __nv_bfloat16*>(x), wn, Sn, rstd, eps,
        reinterpret_cast<__nv_bfloat16*>(yq), reinterpret_cast<__nv_bfloat16*>(yk), reinterpret_cast<__nv_bfloat16*>(yv), reinterpret_cast<__nv_bfloat16*>(yg), Nq, Nk, Nv, Ng, K);
    return (int)cudaGetLastError();
}

template <bool kFp8Rows, int ROWS>
static int laguna_launch_gemv_q8_qkvg_normed_rows(
    cudaStream_t s, const void* Wq, const float* Sq, const void* Wk, const float* Sk, const void* Wv, const float* Sv,
    const void* Wg, const float* Sg, const uint16_t* x, const void* wn, const float* Sn, const float* rstd, float eps,
    uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg, int Nq, int Nk, int Nv, int Ng, int K
) {
    const size_t smem32 = (size_t)ROWS * (K >> 1) * sizeof(float2);
    const size_t smem16 = (size_t)ROWS * (K >> 1) * sizeof(__nv_bfloat162);
    const bool two = (Nq % 16 == 0) && (Nk % 16 == 0) && (Nv % 16 == 0) && (Ng % 16 == 0);
    if (kLagRowsQ8XStageF32PairsCostOneLds64PerPairInsteadOfLds32PlusTwoUnpacks && smem32 <= (size_t)kLagRowsQ8MaxDynamicSmemBytes98304FitsTheSm120PerBlockOptin) {
        const int rc = two
            ? laguna_launch_gemv_q8_qkvg_normed_rows_staged<kFp8Rows, ROWS, true, 2>(s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, Sg, x, wn, Sn, rstd, eps, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K, smem32)
            : laguna_launch_gemv_q8_qkvg_normed_rows_staged<kFp8Rows, ROWS, true, 1>(s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, Sg, x, wn, Sn, rstd, eps, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K, smem32);
        if (rc != -2) return rc;
    }
    return two
        ? laguna_launch_gemv_q8_qkvg_normed_rows_staged<kFp8Rows, ROWS, false, 2>(s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, Sg, x, wn, Sn, rstd, eps, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K, smem16)
        : laguna_launch_gemv_q8_qkvg_normed_rows_staged<kFp8Rows, ROWS, false, 1>(s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, Sg, x, wn, Sn, rstd, eps, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K, smem16);
}

template <bool kFp8Rows, int ROWS, bool kGate, bool kXF32, int NPW>
static int laguna_launch_gemv_q8_rows_staged(cudaStream_t s, const void* W, const float* S, const uint16_t* x, const uint16_t* gate, int gate_heads, int gate_hd, uint16_t* y, int N, int K, size_t smem) {
    static bool attr = false;
    static int limit = 0;
    if (!laguna_rows_q8_smem_attr_set((const void*)laguna_gemv_q8_rows_kernel<kFp8Rows, ROWS, kGate, kXF32, NPW>, &attr, &limit)) return -1;
    if (smem > (size_t)limit) return -2;
    const int per_block = kLagRowsGemvWarpsPerBlock8MatchesGemvBf16RowsPerBlock * NPW;
    dim3 grid((unsigned)((N + per_block - 1) / per_block));
    laguna_gemv_q8_rows_kernel<kFp8Rows, ROWS, kGate, kXF32, NPW><<<grid, kLagRowsGemvThreads, smem, s>>>(
        reinterpret_cast<const uint8_t*>(W), S, reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(gate), gate_heads, gate_hd,
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return (int)cudaGetLastError();
}

template <bool kFp8Rows, int ROWS, bool kGate>
static int laguna_launch_gemv_q8_rows(cudaStream_t s, const void* W, const float* S, const uint16_t* x, const uint16_t* gate, int gate_heads, int gate_hd, uint16_t* y, int N, int K) {
    const size_t smem32 = (size_t)ROWS * (K >> 1) * sizeof(float2);
    const size_t smem16 = (size_t)ROWS * (K >> 1) * sizeof(__nv_bfloat162);
    if (kLagRowsQ8XStageF32PairsCostOneLds64PerPairInsteadOfLds32PlusTwoUnpacks && smem32 <= (size_t)kLagRowsQ8MaxDynamicSmemBytes98304FitsTheSm120PerBlockOptin) {
        const int rc = laguna_launch_gemv_q8_rows_staged<kFp8Rows, ROWS, kGate, true, kLagRowsQ8OutputsPerWarp2HalvesTheBlockWavesAtTheSameLoadsInFlightPerLane>(s, W, S, x, gate, gate_heads, gate_hd, y, N, K, smem32);
        if (rc != -2) return rc;
    }
    return laguna_launch_gemv_q8_rows_staged<kFp8Rows, ROWS, kGate, false, kLagRowsQ8OutputsPerWarp2HalvesTheBlockWavesAtTheSameLoadsInFlightPerLane>(s, W, S, x, gate, gate_heads, gate_hd, y, N, K, smem16);
}

#define NV_LAG_Q8ROWS_DISPATCH(FN, FP8V, ...) \
    switch (rows) { \
        case 1: return FN<FP8V, 1>(__VA_ARGS__); \
        case 2: return FN<FP8V, 2>(__VA_ARGS__); \
        case 3: return FN<FP8V, 3>(__VA_ARGS__); \
        case 4: return FN<FP8V, 4>(__VA_ARGS__); \
        case 5: return FN<FP8V, 5>(__VA_ARGS__); \
        case 6: return FN<FP8V, 6>(__VA_ARGS__); \
        case 7: return FN<FP8V, 7>(__VA_ARGS__); \
        case 8: return FN<FP8V, 8>(__VA_ARGS__); \
        default: return -1; \
    }

extern "C" int nv_kernels_laguna_gemv_q8_qkvg_normed_rows(
    void* stream, int fp8,
    const void* Wq, const float* Sq, const void* Wk, const float* Sk, const void* Wv, const float* Sv,
    const uint16_t* Wg, const uint16_t* x, const uint16_t* wn, const float* rstd,
    uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg,
    int Nq, int Nk, int Nv, int Ng, int K, int rows
) {
    if (Nq <= 0 || Nk <= 0 || Nv <= 0 || Ng < 0 || K <= 0 || (K & 15) != 0 || K > kLagRowsQ8MaxK8192CoversTheSlidingLayersOProj) return -1;
    if (Wq == nullptr || Wk == nullptr || Wv == nullptr || (Ng > 0 && Wg == nullptr)) return -1;
    if (rows < 1 || rows > kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp) return -1;
    if (rstd == nullptr) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    if (fp8) { NV_LAG_Q8ROWS_DISPATCH(laguna_launch_gemv_q8_qkvg_normed_rows, true, s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, nullptr, x, wn, nullptr, rstd, 0.0f, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K) }
    NV_LAG_Q8ROWS_DISPATCH(laguna_launch_gemv_q8_qkvg_normed_rows, false, s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, nullptr, x, wn, nullptr, rstd, 0.0f, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K)
}

extern "C" int nv_kernels_laguna_gemv_q8_qkvg_normed_rows_rstd_fused(
    void* stream, int fp8,
    const void* Wq, const float* Sq, const void* Wk, const float* Sk, const void* Wv, const float* Sv,
    const uint16_t* Wg, const uint16_t* x, const uint16_t* wn, float eps,
    uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg,
    int Nq, int Nk, int Nv, int Ng, int K, int rows
) {
    if (Nq <= 0 || Nk <= 0 || Nv <= 0 || Ng < 0 || K <= 0 || (K & 15) != 0 || K > kLagRowsQ8MaxK8192CoversTheSlidingLayersOProj) return -1;
    if (Wq == nullptr || Wk == nullptr || Wv == nullptr || (Ng > 0 && Wg == nullptr) || !(eps >= 0.0f)) return -1;
    if (rows < 1 || rows > kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    const float* no_rstd = nullptr;
    if (fp8) { NV_LAG_Q8ROWS_DISPATCH(laguna_launch_gemv_q8_qkvg_normed_rows, true, s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, nullptr, x, wn, nullptr, no_rstd, eps, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K) }
    NV_LAG_Q8ROWS_DISPATCH(laguna_launch_gemv_q8_qkvg_normed_rows, false, s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, nullptr, x, wn, nullptr, no_rstd, eps, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K)
}

extern "C" int nv_kernels_laguna_gemv_fp8_qkvg_normed_rows_rstd_fused(
    void* stream,
    const uint8_t* Wq, const float* Sq, const uint8_t* Wk, const float* Sk, const uint8_t* Wv, const float* Sv,
    const uint8_t* Wg, const float* Sg, const uint16_t* x, const uint16_t* wn, float eps,
    uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg,
    int Nq, int Nk, int Nv, int Ng, int K, int rows
) {
    if (Nq <= 0 || Nk <= 0 || Nv <= 0 || Ng <= 0 || K <= 0 || (K & 15) != 0 || K > kLagRowsQ8MaxK8192CoversTheSlidingLayersOProj) return -1;
    if (!Wq || !Sq || !Wk || !Sk || !Wv || !Sv || !Wg || !Sg || !x || !wn || !yq || !yk || !yv || !yg || !isfinite(eps) || eps < 0.0f) return -1;
    if (rows < 1 || rows > kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    NV_LAG_Q8ROWS_DISPATCH(laguna_launch_gemv_q8_qkvg_normed_rows, true, s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, Sg, x, wn, nullptr, nullptr, eps, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K)
}

extern "C" int nv_kernels_laguna_gemv_fp8_qkvg_fp8_norm_rows(
    void* stream,
    const uint8_t* Wq, const float* Sq, const uint8_t* Wk, const float* Sk,
    const uint8_t* Wv, const float* Sv, const uint8_t* Wg, const float* Sg,
    const uint16_t* x, const uint8_t* wn, const float* Sn, float eps,
    uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg,
    int Nq, int Nk, int Nv, int Ng, int K, int rows
) {
    if (Nq <= 0 || Nk <= 0 || Nv <= 0 || Ng <= 0 || K <= 0 || (K & 15) != 0 || K > kLagRowsQ8MaxK8192CoversTheSlidingLayersOProj) return -1;
    if (!Wq || !Sq || !Wk || !Sk || !Wv || !Sv || !Wg || !Sg || !x || !wn || !Sn || !yq || !yk || !yv || !yg || !isfinite(eps) || eps < 0.0f) return -1;
    if (rows < 1 || rows > kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    NV_LAG_Q8ROWS_DISPATCH(laguna_launch_gemv_q8_qkvg_normed_rows, true, s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, Sg, x, wn, Sn, nullptr, eps, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K)
}

extern "C" int nv_kernels_laguna_gemv_fp8_qkvg_rows_rstd(
    void* stream,
    const uint8_t* Wq, const float* Sq, const uint8_t* Wk, const float* Sk,
    const uint8_t* Wv, const float* Sv, const uint8_t* Wg, const float* Sg,
    const uint16_t* x, const void* wn, const float* Sn, const float* rstd,
    uint16_t* yq, uint16_t* yk, uint16_t* yv, uint16_t* yg,
    int Nq, int Nk, int Nv, int Ng, int K, int rows
) {
    if (Nq <= 0 || Nk <= 0 || Nv <= 0 || Ng <= 0 || K <= 0 || (K & 15) != 0 || K > kLagRowsQ8MaxK8192CoversTheSlidingLayersOProj) return -1;
    if (!Wq || !Sq || !Wk || !Sk || !Wv || !Sv || !Wg || !Sg || !x || !wn || !rstd || !yq || !yk || !yv || !yg) return -1;
    if (rows < 1 || rows > kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    NV_LAG_Q8ROWS_DISPATCH(laguna_launch_gemv_q8_qkvg_normed_rows, true, s, Wq, Sq, Wk, Sk, Wv, Sv, Wg, Sg, x, wn, Sn, rstd, 0.0f, yq, yk, yv, yg, Nq, Nk, Nv, Ng, K)
}

#define NV_LAG_Q8ROWS_DISPATCH_GATE(FN, FP8V, GATEV, ...) \
    switch (rows) { \
        case 1: return FN<FP8V, 1, GATEV>(__VA_ARGS__); \
        case 2: return FN<FP8V, 2, GATEV>(__VA_ARGS__); \
        case 3: return FN<FP8V, 3, GATEV>(__VA_ARGS__); \
        case 4: return FN<FP8V, 4, GATEV>(__VA_ARGS__); \
        case 5: return FN<FP8V, 5, GATEV>(__VA_ARGS__); \
        case 6: return FN<FP8V, 6, GATEV>(__VA_ARGS__); \
        case 7: return FN<FP8V, 7, GATEV>(__VA_ARGS__); \
        case 8: return FN<FP8V, 8, GATEV>(__VA_ARGS__); \
        default: return -1; \
    }

extern "C" int nv_kernels_laguna_gemv_q8_rows(void* stream, int fp8, const void* W, const float* S, const uint16_t* x, uint16_t* y, int N, int K, int rows) {
    if (N <= 0 || K <= 0 || (K & 15) != 0 || K > kLagRowsQ8MaxK8192CoversTheSlidingLayersOProj || W == nullptr || S == nullptr) return -1;
    if (rows < 1 || rows > kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    if (fp8) { NV_LAG_Q8ROWS_DISPATCH_GATE(laguna_launch_gemv_q8_rows, true, false, s, W, S, x, nullptr, 0, 0, y, N, K) }
    NV_LAG_Q8ROWS_DISPATCH_GATE(laguna_launch_gemv_q8_rows, false, false, s, W, S, x, nullptr, 0, 0, y, N, K)
}

extern "C" int nv_kernels_laguna_gemv_q8_rows_gated(void* stream, int fp8, const void* W, const float* S, const uint16_t* attn, const uint16_t* gate, int gate_heads, int gate_hd, uint16_t* y, int N, int K, int rows) {
    if (N <= 0 || K <= 0 || (K & 15) != 0 || K > kLagRowsQ8MaxK8192CoversTheSlidingLayersOProj || W == nullptr || S == nullptr || gate == nullptr) return -1;
    if (rows < 1 || rows > kLagRowsGemvMaxRows8CoversVerifyBlocksAndCatchUp) return -1;
    if (gate_heads < 1 || gate_heads > kLagRowsQ8GateHeadsMax64CoversTheSlidingLayers || gate_hd < 8 || (gate_hd & 7) != 0 || gate_heads * gate_hd != K) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    if (fp8) { NV_LAG_Q8ROWS_DISPATCH_GATE(laguna_launch_gemv_q8_rows, true, true, s, W, S, attn, gate, gate_heads, gate_hd, y, N, K) }
    NV_LAG_Q8ROWS_DISPATCH_GATE(laguna_launch_gemv_q8_rows, false, true, s, W, S, attn, gate, gate_heads, gate_hd, y, N, K)
}
#undef NV_LAG_Q8ROWS_DISPATCH_GATE
#undef NV_LAG_Q8ROWS_DISPATCH

__global__ void laguna_rstd256_bf16_rows_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ rstd_out, int dim, float eps) {
    const __nv_bfloat16* xr = x + (size_t)blockIdx.x * dim;
    float local = 0.0f;
    for (int i = threadIdx.x; i < dim; i += kNormBlock) {
        float v = __bfloat162float(xr[i]);
        local += v * v;
    }
    float sum = ls_block_sum256(local);
    if (threadIdx.x == 0) rstd_out[blockIdx.x] = rsqrtf(sum / (float)dim + eps);
}

extern "C" int nv_kernels_laguna_rstd256_bf16_rows(void* stream, const uint16_t* x, float* rstd_out, int dim, float eps, int rows) {
    if (dim <= 0 || rows < 1 || x == nullptr || rstd_out == nullptr) return -1;
    laguna_rstd256_bf16_rows_kernel<<<(unsigned)rows, kNormBlock, 0, (cudaStream_t)stream>>>(reinterpret_cast<const __nv_bfloat16*>(x), rstd_out, dim, eps);
    return (int)cudaGetLastError();
}

constexpr int kLagSlidingVerifySplits8Covers512KeysAt8WarpsPerSplit = 8;

template <int GROUP, int RB>
__global__ void __launch_bounds__(kGqaThreads)
laguna_flash_decode_verify_window_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* out,
    const int* committed_ring_len_after_append,
    float* scratch,
    unsigned int* fan_in,
    int n_kv,
    int rows,
    int capacity,
    int window,
    float scale
) {
    const int row0 = blockIdx.z * RB;
    const size_t row_elems = (size_t)n_kv * GROUP * kGqaHD;
    const int committed_before_the_block = committed_ring_len_after_append[0] - rows;
    if (committed_before_the_block < 0 || committed_before_the_block > capacity - rows) {
        if (blockIdx.y == 0) {
            for (int r = row0; r < rows && r < row0 + RB; ++r)
                for (int i = threadIdx.x; i < GROUP * kGqaHD; i += blockDim.x)
                    out[r * row_elems + (size_t)blockIdx.x * GROUP * kGqaHD + i] =
                        __float2bfloat16(NAN);
        }
        return;
    }
    laguna_flash_decode_gqa_rows_share_kv_body<GROUP, kLagSlidingVerifySplits8Covers512KeysAt8WarpsPerSplit, kLagKvBf16, RB>(
        q, k, v, out, committed_ring_len_after_append, scratch, fan_in, n_kv, rows, row0, scale, nullptr, nullptr, window, -rows);
}

extern "C" int nv_kernels_laguna_flash_decode_verify_window(
    void* stream,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* committed,
    float* scratch,
    unsigned int* fan_in,
    int n_q,
    int n_kv,
    int head_dim,
    int rows,
    int capacity,
    int window,
    int rows_per_block,
    float scale
) {
    if (n_q <= 0 || n_kv <= 0 || head_dim != kGqaHD || n_q % n_kv != 0 || rows < 1 || rows > 8 || capacity < rows
        || window <= 0 || rows_per_block < 1 || rows_per_block > kLagFlashVerifyRowsPerBlockMax3KeepsRowsTimesGroupAccumulatorsInRegisters
        || !isfinite(scale) || q == nullptr || k == nullptr || v == nullptr || out == nullptr || committed == nullptr
        || scratch == nullptr || fan_in == nullptr)
        return -1;
    const int group = n_q / n_kv;
    if (group != 6 && group != 8) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int rb = rows_per_block;
    dim3 grid((unsigned)n_kv, (unsigned)kLagSlidingVerifySplits8Covers512KeysAt8WarpsPerSplit, (unsigned)((rows + rb - 1) / rb));
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(v);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
#define NV_LAG_WIN(G, RBV) \
    laguna_flash_decode_verify_window_kernel<G, RBV><<<grid, kGqaThreads, 0, s>>>(qb, kb, vb, ob, committed, scratch, fan_in, n_kv, rows, capacity, window, scale)
    switch (group * 10 + rb) {
        case 61: NV_LAG_WIN(6, 1); break;
        case 62: NV_LAG_WIN(6, 2); break;
        case 63: NV_LAG_WIN(6, 3); break;
        case 81: NV_LAG_WIN(8, 1); break;
        case 82: NV_LAG_WIN(8, 2); break;
        case 83: NV_LAG_WIN(8, 3); break;
        default: return -1;
    }
#undef NV_LAG_WIN
    return (int)cudaGetLastError();
}


extern "C" int nv_kernels_laguna_qk_normrope_fp8_weight_base_rows(
    void* stream,
    const uint16_t* q_in,
    const uint16_t* k_in,
    uint16_t* q_out,
    uint16_t* k_out,
    const uint8_t* qw,
    const float* qs,
    const uint8_t* kw,
    const float* ks,
    const float* cos_tbl,
    const float* sin_tbl,
    const int* pos_base,
    int n_q,
    int n_kv,
    int head_dim,
    int rotary_dim,
    float rot_scale,
    float eps_q,
    float eps_k,
    int rows
) {
    if (n_q <= 0 || n_kv <= 0 || rows < 1 || rows > 32) return -1;
    if (!qw || !qs || !kw || !ks) return -1;
    if (head_dim < 32 || head_dim > 256 || (head_dim & 31)) return -2;
    if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    laguna_qk_normrope_bf16_kernel<true><<<dim3((unsigned)(n_q + n_kv), (unsigned)rows), kNormBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_in),
        reinterpret_cast<const __nv_bfloat16*>(k_in),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<const void*>(qw),
        reinterpret_cast<const void*>(kw),
        qs,
        ks,
        cos_tbl, sin_tbl, pos_base, n_q, n_kv, head_dim, rotary_dim, rot_scale, eps_q, eps_k);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_laguna_qk_normrope_bf16_base_rows(
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
    float eps_k,
    int rows
) {
    if (n_q <= 0 || n_kv <= 0 || rows < 1 || rows > 32) return -1;
    if (!q_in || !k_in || !q_out || !k_out || !qw || !kw || !cos_tbl || !sin_tbl || !pos_base) return -1;
    if (head_dim < 32 || head_dim > 256 || (head_dim & 31)) return -2;
    if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    laguna_qk_normrope_bf16_kernel<true><<<dim3((unsigned)(n_q + n_kv), (unsigned)rows), kNormBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_in),
        reinterpret_cast<const __nv_bfloat16*>(k_in),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<const void*>(qw),
        reinterpret_cast<const void*>(kw),
        nullptr,
        nullptr,
        cos_tbl, sin_tbl, pos_base, n_q, n_kv, head_dim, rotary_dim, rot_scale, eps_q, eps_k);
    return (int)cudaGetLastError();
}

template<bool FP8_WEIGHT>
__global__ void __launch_bounds__(kNormBlock) laguna_prefill_qk_normrope_kernel(
    const __nv_bfloat16* __restrict__ q_in,
    const __nv_bfloat16* __restrict__ k_in,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    const void* __restrict__ qw,
    const void* __restrict__ kw,
    const float* __restrict__ qs,
    const float* __restrict__ ks,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int* __restrict__ positions,
    int n_q,
    int n_kv,
    int head_dim,
    int rotary_dim,
    float rot_scale,
    float eps_q,
    float eps_k
) {
    const int heads = n_q + n_kv;
    const int row = blockIdx.x / heads;
    const int h = blockIdx.x - row * heads;
    const int d = threadIdx.x;
    const bool is_q = h < n_q;
    const __nv_bfloat16* src = is_q ? q_in + ((size_t)row * n_q + h) * head_dim
                                    : k_in + ((size_t)row * n_kv + (h - n_q)) * head_dim;
    __nv_bfloat16* dst = is_q ? q_out + ((size_t)row * n_q + h) * head_dim
                              : k_out + ((size_t)row * n_kv + (h - n_q)) * head_dim;
    const void* w = is_q ? qw : kw;
    const float eps = is_q ? eps_q : eps_k;

    float local = 0.f;
    float x = 0.f;
    if (d < head_dim) {
        x = __bfloat162float(src[d]);
        local += x * x;
    }
    float sum = ls_block_sum256(local);
    float rms = rsqrtf(sum / (float)head_dim + eps);

    __shared__ __nv_bfloat16 normed[kNormBlock];
    if (d < head_dim) {
        float weight;
        if constexpr (FP8_WEIGHT) {
            const float raw = laguna_fp8x2_to_float2(reinterpret_cast<const uint8_t*>(w)[d]).x;
            weight = __bfloat162float(__float2bfloat16(raw * (is_q ? qs[0] : ks[0])));
        } else {
            weight = __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(w)[d]);
        }
        float v = __bfloat162float(src[d]) * rms * weight;
        normed[d] = __float2bfloat16(v);
    }
    __syncthreads();

    const int half = rotary_dim / 2;
    const int pos = positions[row];
    if (d < half) {
        const float* cos_row = cos_tbl + (size_t)pos * half;
        const float* sin_row = sin_tbl + (size_t)pos * half;
        float c = cos_row[d];
        float s = sin_row[d];
        float a = __bfloat162float(normed[d]);
        float b = __bfloat162float(normed[d + half]);
        float lo = a * c - b * s;
        float hi = a * s + b * c;
        if (rot_scale != 1.0f) {
            lo = lo * rot_scale + 0.0f;
            hi = hi * rot_scale + 0.0f;
        }
        dst[d] = __float2bfloat16(lo);
        dst[d + half] = __float2bfloat16(hi);
    } else if (d >= rotary_dim && d < head_dim) {
        dst[d] = normed[d];
    }
}

extern "C" int nv_kernels_laguna_prefill_qk_normrope_bf16(
    void* stream,
    const uint16_t* q_in,
    const uint16_t* k_in,
    uint16_t* q_out,
    uint16_t* k_out,
    const uint16_t* qw,
    const uint16_t* kw,
    const float* cos_tbl,
    const float* sin_tbl,
    const int* positions,
    int n_q,
    int n_kv,
    int head_dim,
    int rotary_dim,
    float rot_scale,
    float eps_q,
    float eps_k,
    int rows
) {
    if (n_q <= 0 || n_kv <= 0 || rows < 1 || rows > 262144) return -1;
    if (head_dim < 32 || head_dim > kNormBlock || (head_dim & 31)) return -2;
    if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    laguna_prefill_qk_normrope_kernel<false><<<(unsigned)((size_t)rows * (n_q + n_kv)), kNormBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_in),
        reinterpret_cast<const __nv_bfloat16*>(k_in),
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<const __nv_bfloat16*>(qw),
        reinterpret_cast<const __nv_bfloat16*>(kw),
        nullptr, nullptr,
        cos_tbl, sin_tbl, positions, n_q, n_kv, head_dim, rotary_dim, rot_scale, eps_q, eps_k);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_laguna_prefill_qk_normrope_fp8_weight(
    void* stream, const uint16_t* q_in, const uint16_t* k_in,
    uint16_t* q_out, uint16_t* k_out,
    const uint8_t* qw, const uint8_t* kw, const float* qs, const float* ks,
    const float* cos_tbl, const float* sin_tbl, const int* positions,
    int n_q, int n_kv, int head_dim, int rotary_dim,
    float rot_scale, float eps_q, float eps_k, int rows
) {
    if (n_q <= 0 || n_kv <= 0 || rows < 1 || rows > 262144) return -1;
    if (head_dim < 32 || head_dim > kNormBlock || (head_dim & 31)) return -2;
    if (rotary_dim < 2 || rotary_dim > head_dim || (rotary_dim & 1)) return -2;
    if (!qw || !kw || !qs || !ks) return -3;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    laguna_prefill_qk_normrope_kernel<true><<<(unsigned)((size_t)rows * (n_q + n_kv)), kNormBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_in), reinterpret_cast<const __nv_bfloat16*>(k_in),
        reinterpret_cast<__nv_bfloat16*>(q_out), reinterpret_cast<__nv_bfloat16*>(k_out),
        qw, kw, qs, ks, cos_tbl, sin_tbl, positions,
        n_q, n_kv, head_dim, rotary_dim, rot_scale, eps_q, eps_k);
    return (int)cudaGetLastError();
}

__global__ void laguna_prefill_gate_bf16_kernel(
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int n_q,
    int head_dim
) {
    const int row = blockIdx.x / n_q;
    const int h = blockIdx.x - row * n_q;
    const int d = threadIdx.x;
    if (d >= head_dim) return;
    float g = __bfloat162float(gate[(size_t)row * n_q + h]);
    float relu = fmaxf(g, 0.0f);
    float neg_abs = -fabsf(g);
    float e = expf(neg_abs);
    float aff = e * 1.0f + 1.0f;
    float lg = logf(aff);
    float sp = relu + lg;
    __nv_bfloat16 spb = __float2bfloat16(sp);
    const size_t idx = ((size_t)row * n_q + h) * head_dim + d;
    out[idx] = __hmul(attn[idx], spb);
}

extern "C" int nv_kernels_laguna_prefill_gate_bf16(
    void* stream,
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int rows,
    int n_q,
    int head_dim
) {
    if (n_q <= 0 || rows < 1 || rows > 262144) return -1;
    if (head_dim < 32 || head_dim > 1024) return -2;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    laguna_prefill_gate_bf16_kernel<<<(unsigned)((size_t)rows * n_q), head_dim, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(attn),
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<__nv_bfloat16*>(out), n_q, head_dim);
    return (int)cudaGetLastError();
}
