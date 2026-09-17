#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>

namespace {

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

constexpr int kLagKvBf16 = 0;
constexpr int kLagKvE4m3PerTokenHeadScale = 1;
constexpr int kLagKvInt8PerTokenHeadScale = 2;

template <int KVQ>
__device__ __forceinline__ float laguna_kv_byte_to_float(uint8_t byte) {
    if constexpr (KVQ == kLagKvInt8PerTokenHeadScale) {
        return __uint_as_float(0x4B000000u | (static_cast<unsigned>(byte) ^ 0x80u)) - 8388736.0f;
    } else {
        __nv_fp8_e4m3 value;
        value.__x = byte;
        return static_cast<float>(value);
    }
}

template <int KVQ>
__device__ __forceinline__ uint2 laguna_load_kv4(
    const __nv_bfloat16* base, const float* scales, size_t row, int lane
) {
    if constexpr (KVQ == kLagKvBf16) {
        return __ldg(reinterpret_cast<const uint2*>(base + row * kGqaHD) + lane);
    } else {
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(base) + row * kGqaHD;
        const uint32_t packed = __ldg(reinterpret_cast<const uint32_t*>(bytes) + lane);
        const float scale = __ldg(scales + row);
        uint2 result;
        __nv_bfloat16* values = reinterpret_cast<__nv_bfloat16*>(&result);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            values[i] = __float2bfloat16(laguna_kv_byte_to_float<KVQ>(static_cast<uint8_t>(packed >> (i * 8))) * scale);
        }
        return result;
    }
}

template <int GROUP, int SPLITS, int STAGES, int KVQ = kLagKvBf16,
          bool VERIFY_ROWS_OWN_INSTANTIATION_SO_NO_TWO_KERNELS_SHARE_ONE_BODY_AND_TRIP_NVLINK_SMEM = false>
__device__ __forceinline__ void laguna_flash_decode_gqa_body(
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
    int mid_band_hi_total,
    const float* k_scales = nullptr,
    const float* v_scales = nullptr
) {
    static_assert(KVQ == kLagKvBf16 || STAGES == 0, "QUANTIZED_KV_LOADER_REQUIRES_REGISTER_PREFETCH_STAGE_ZERO");
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int HD = kGqaHD;

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

    __shared__ float sm[kGqaWarps][GROUP];
    __shared__ float sl[kGqaWarps][GROUP];
    __shared__ __align__(16) float sacc[kGqaWarps][GROUP][kGqaHD];

    const int stride = s_eff * kGqaWarps;
    int p = start + split * kGqaWarps + warp;
    if constexpr (STAGES == 0) {
        uint2 kraw, vraw;
        if (p < total) {
            kraw = laguna_load_kv4<KVQ>(k, k_scales, (size_t)p * NKV + kvh, lane);
            vraw = laguna_load_kv4<KVQ>(v, v_scales, (size_t)p * NKV + kvh, lane);
        }
        for (; p < total; p += stride) {
            const int pn = p + stride;
            uint2 kn, vn;
            if (pn < total) {
                kn = laguna_load_kv4<KVQ>(k, k_scales, (size_t)pn * NKV + kvh, lane);
                vn = laguna_load_kv4<KVQ>(v, v_scales, (size_t)pn * NKV + kvh, lane);
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
    } else {
        static_assert(STAGES >= 3,
            "STAGES_BELOW_3_IS_NOT_DEEPER_THAN_THE_SHIPPED_1_DEEP_REGISTER_PREFETCH");
        static_assert((size_t)2 * STAGES * kGqaWarps * kGqaWarp * sizeof(uint2) <= sizeof(sacc),
            "KV_STAGING_ALIASES_SACC_SO_PIPE_VARIANTS_ADD_ZERO_SMEM_AND_MUST_FIT_INSIDE_IT");
        uint2* kst = reinterpret_cast<uint2*>(&sacc[0][0][0]);
        uint2* vst = kst + STAGES * kGqaWarps * kGqaWarp;
        const int slot = warp * kGqaWarp + lane;
        #pragma unroll
        for (int i = 0; i < STAGES - 1; ++i) {
            const int pi = p + i * stride;
            if (pi < total) {
                __pipeline_memcpy_async(
                    kst + i * kGqaWarps * kGqaWarp + slot,
                    reinterpret_cast<const uint2*>(k + ((size_t)pi * NKV + kvh) * HD) + lane,
                    sizeof(uint2));
                __pipeline_memcpy_async(
                    vst + i * kGqaWarps * kGqaWarp + slot,
                    reinterpret_cast<const uint2*>(v + ((size_t)pi * NKV + kvh) * HD) + lane,
                    sizeof(uint2));
            }
            __pipeline_commit();
        }
        int stage = 0;
        for (; p < total; p += stride) {
            const int pn = p + (STAGES - 1) * stride;
            __pipeline_wait_prior(STAGES - 2);
            const uint2 kraw = kst[stage * kGqaWarps * kGqaWarp + slot];
            const uint2 vraw = vst[stage * kGqaWarps * kGqaWarp + slot];
            const int fill_stage_row_pn_lands_where_iter_plus_stages_minus_1_reads =
                (stage == 0) ? (STAGES - 1) : (stage - 1);
            if (pn < total) {
                __pipeline_memcpy_async(
                    kst + fill_stage_row_pn_lands_where_iter_plus_stages_minus_1_reads * kGqaWarps * kGqaWarp + slot,
                    reinterpret_cast<const uint2*>(k + ((size_t)pn * NKV + kvh) * HD) + lane,
                    sizeof(uint2));
                __pipeline_memcpy_async(
                    vst + fill_stage_row_pn_lands_where_iter_plus_stages_minus_1_reads * kGqaWarps * kGqaWarp + slot,
                    reinterpret_cast<const uint2*>(v + ((size_t)pn * NKV + kvh) * HD) + lane,
                    sizeof(uint2));
            }
            __pipeline_commit();
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
            stage = (stage + 1 == STAGES) ? 0 : stage + 1;
        }
        __syncthreads();
    }

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

template <int GROUP, int SPLITS, int KVQ, int RB>
__device__ __forceinline__ void laguna_flash_decode_gqa_rows_share_kv_body(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ committed_ptr,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    int NKV,
    int rows,
    int row0,
    float scale,
    const float* k_scales,
    const float* v_scales,
    int WINDOW = 0,
    int committed_delta = 0
) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int HD = kGqaHD;
    const size_t row_elems = (size_t)NKV * GROUP * HD;
    const size_t row_scratch = (size_t)NKV * SPLITS * GROUP * (HD + 2);
    const int committed = committed_ptr[0] + committed_delta;
    const int nr = min(RB, rows - row0);
    const int total_max = committed + row0 + nr;
    const int first_visible = (WINDOW > 0) ? max(0, committed + row0 + 1 - WINDOW) : 0;
    const int lane = threadIdx.x & (kGqaWarp - 1);
    const int warp = threadIdx.x >> 5;

    __shared__ float qsh[RB][GROUP][kGqaHD];
    for (int i = threadIdx.x; i < RB * GROUP * HD; i += kGqaThreads) {
        const int r = i / (GROUP * HD);
        const int rem = i - r * GROUP * HD;
        const int g = rem / HD;
        const int d = rem - g * HD;
        qsh[r][g][d] = (r < nr)
            ? __bfloat162float(q[(size_t)(row0 + r) * row_elems + (size_t)(kvh * GROUP + g) * HD + d]) * scale
            : 0.0f;
    }
    __syncthreads();

    float acc[RB][GROUP][kGqaVc];
    float m[RB][GROUP], l[RB][GROUP];
    #pragma unroll
    for (int r = 0; r < RB; ++r) {
        #pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            m[r][g] = -INFINITY;
            l[r][g] = 0.0f;
            #pragma unroll
            for (int j = 0; j < kGqaVc; ++j) acc[r][g][j] = 0.0f;
        }
    }

    const int stride = SPLITS * kGqaWarps;
    int p = first_visible + split * kGqaWarps + warp;
    uint2 kraw, vraw;
    if (p < total_max) {
        kraw = laguna_load_kv4<KVQ>(k, k_scales, (size_t)p * NKV + kvh, lane);
        vraw = laguna_load_kv4<KVQ>(v, v_scales, (size_t)p * NKV + kvh, lane);
    }
    for (; p < total_max; p += stride) {
        const int pn = p + stride;
        uint2 kn, vn;
        if (pn < total_max) {
            kn = laguna_load_kv4<KVQ>(k, k_scales, (size_t)pn * NKV + kvh, lane);
            vn = laguna_load_kv4<KVQ>(v, v_scales, (size_t)pn * NKV + kvh, lane);
        }
        const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&kraw);
        float2 k01 = __bfloat1622float2(kb[0]);
        float2 k23 = __bfloat1622float2(kb[1]);
        const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&vraw);
        float2 v01 = __bfloat1622float2(vb[0]);
        float2 v23 = __bfloat1622float2(vb[1]);
        #pragma unroll
        for (int r = 0; r < RB; ++r) {
            const bool visible = p < committed + row0 + r + 1 && (WINDOW <= 0 || p >= committed + row0 + r + 1 - WINDOW);
            #pragma unroll
            for (int g = 0; g < GROUP; ++g) {
                const float* qp = qsh[r][g] + lane * kGqaVc;
                float partial = k01.x * qp[0] + k01.y * qp[1] + k23.x * qp[2] + k23.y * qp[3];
                float score = gqa_warp_sum(partial);
                if (visible) {
                    float m_new = fmaxf(m[r][g], score);
                    float corr = __expf(m[r][g] - m_new);
                    float w = __expf(score - m_new);
                    l[r][g] = l[r][g] * corr + w;
                    acc[r][g][0] = acc[r][g][0] * corr + w * v01.x;
                    acc[r][g][1] = acc[r][g][1] * corr + w * v01.y;
                    acc[r][g][2] = acc[r][g][2] * corr + w * v23.x;
                    acc[r][g][3] = acc[r][g][3] * corr + w * v23.y;
                    m[r][g] = m_new;
                }
            }
        }
        if (pn < total_max) {
            kraw = kn;
            vraw = vn;
        }
    }

    __shared__ float sm[kGqaWarps][GROUP];
    __shared__ float sl[kGqaWarps][GROUP];
    __shared__ __align__(16) float sacc[kGqaWarps][GROUP][kGqaHD];
    __shared__ float ssc[GROUP][SPLITS];
    __shared__ float sinv_l[GROUP];
    __shared__ unsigned int ticket;
    #pragma unroll
    for (int r = 0; r < RB; ++r) {
        if (r >= nr) break;
        const int row = row0 + r;
        #pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            if (lane == 0) {
                sm[warp][g] = m[r][g];
                sl[warp][g] = l[r][g];
            }
            #pragma unroll
            for (int j = 0; j < kGqaVc; ++j) sacc[warp][g][lane * kGqaVc + j] = acc[r][g][j];
        }
        __syncthreads();

        float* part = scratch + (size_t)row * row_scratch + ((size_t)kvh * SPLITS + split) * GROUP * (kGqaHD + 2);
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
        if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[(size_t)row * NKV + kvh], 1u);
        __syncthreads();
        if (ticket == SPLITS - 1) {
            __threadfence();
            const float* base = scratch + (size_t)row * row_scratch + (size_t)kvh * SPLITS * GROUP * (kGqaHD + 2);
            if (warp == 0 && lane < GROUP) {
                const int g = lane;
                float m_glob = -INFINITY;
                for (int s = 0; s < SPLITS; ++s)
                    m_glob = fmaxf(m_glob, base[((size_t)s * GROUP + g) * (kGqaHD + 2)]);
                float l_glob = 0.0f;
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
                for (int s = 0; s < SPLITS; ++s)
                    a += base[((size_t)s * GROUP + g) * (kGqaHD + 2) + 2 + d] * ssc[g][s];
                outp[(size_t)row * row_elems + (size_t)(kvh * GROUP + g) * kGqaHD + d] =
                    __float2bfloat16(a * sinv_l[g]);
            }
            if (threadIdx.x == 0) fan_in[(size_t)row * NKV + kvh] = 0u;
        }
        __syncthreads();
    }
}

}
