#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "nvk_pdl.cuh"

namespace q38md {

constexpr int kWarp = 32;
constexpr int kFlashWarps = 8;
constexpr int kFlashThreads = kWarp * kFlashWarps;
constexpr int kHD = 256;
constexpr int kAccPerLane = kHD / kWarp;
constexpr int kIlp = 4;

__device__ __forceinline__ int kv_phys_slot(
    const int* __restrict__ block_table, int block_size, int RING, int p
) {
    if (block_table != nullptr) {
        int blk = p / block_size;
        return block_table[blk] * block_size + (p - blk * block_size);
    }
    return (RING > 0) ? (p % RING) : p;
}

__inline__ __device__ float warp_sum(float x) {
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__inline__ __device__ float2 fp8x2_to_float2(unsigned short packed) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed, __NV_E4M3);
    __half2 h2 = *reinterpret_cast<__half2*>(&hr);
    return __half22float2(h2);
}

template <int SPLITS>
__global__ void flash_splitk_fused_fp8_ilp4_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int NH,
    int NKV,
    int WINDOW,
    int RING,
    float scaling,
    const int* __restrict__ block_table,
    int block_size
) {
    constexpr int HD = kHD;
    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    NVK_PDL_PROLOG();

    const int total = n_total_dev[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

    __shared__ float qsh[HD];
    for (int d = threadIdx.x; d < HD; d += kFlashThreads)
        qsh[d] = __bfloat162float(q[(size_t)h * HD + d]);
    __syncthreads();

    float acc[kAccPerLane];
    #pragma unroll
    for (int i = 0; i < kAccPerLane; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    constexpr int nck = HD >> 7;
    const int iter_stride = SPLITS * kFlashWarps * kIlp;

    for (int p0 = start + (split * kFlashWarps + warp) * kIlp; p0 < total;
         p0 += iter_stride) {
        const int live = min(kIlp, total - p0);
        int sp[kIlp];
        float ks[kIlp];
        float vs[kIlp];
        uchar4 kraw[kIlp][nck];
        uint2 vraw[kIlp];
        #pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            const int pj = p0 + min(j, live - 1);
            sp[j] = kv_phys_slot(block_table, block_size, RING, pj);
        }
        #pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            const size_t row = (size_t)sp[j] * NKV + kvh;
            ks[j] = k_scales[row];
            vs[j] = v_scales[row];
        }
        #pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            const uchar4* k4 =
                reinterpret_cast<const uchar4*>(k_fp8 + ((size_t)sp[j] * NKV + kvh) * HD);
            #pragma unroll
            for (int c = 0; c < nck; ++c) kraw[j][c] = __ldg(&k4[lane + c * kWarp]);
        }
        #pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            const uint2* v8 = reinterpret_cast<const uint2*>(
                v_fp8 + ((size_t)sp[j] * NKV + kvh) * HD + (size_t)lane * kAccPerLane);
            vraw[j] = __ldg(v8);
        }
        float score[kIlp];
        #pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            float partial = 0.0f;
            #pragma unroll
            for (int c = 0; c < nck; ++c) {
                uchar4 raw = kraw[j][c];
                const float* qp = qsh + (lane + c * kWarp) * 4;
                float2 f01 = fp8x2_to_float2(
                    (unsigned short)(raw.x | ((unsigned short)raw.y << 8)));
                float2 f23 = fp8x2_to_float2(
                    (unsigned short)(raw.z | ((unsigned short)raw.w << 8)));
                partial += qp[0] * f01.x + qp[1] * f01.y + qp[2] * f23.x + qp[3] * f23.y;
            }
            score[j] = warp_sum(partial) * ks[j] * scaling;
        }
        #pragma unroll
        for (int j = 0; j < kIlp; ++j) {
            if (j >= live) break;
            float m_new = fmaxf(m, score[j]);
            float corr = __expf(m - m_new);
            float w = __expf(score[j] - m_new);
            l = l * corr + w;
            const float w_v = w * vs[j];
            uint2 raw = vraw[j];
            float2 f0 = fp8x2_to_float2((unsigned short)(raw.x & 0xffffu));
            float2 f1 = fp8x2_to_float2((unsigned short)(raw.x >> 16));
            float2 f2 = fp8x2_to_float2((unsigned short)(raw.y & 0xffffu));
            float2 f3 = fp8x2_to_float2((unsigned short)(raw.y >> 16));
            acc[0] = __fmaf_rn(w_v, f0.x, __fmul_rn(acc[0], corr));
            acc[1] = __fmaf_rn(w_v, f0.y, __fmul_rn(acc[1], corr));
            acc[2] = __fmaf_rn(w_v, f1.x, __fmul_rn(acc[2], corr));
            acc[3] = __fmaf_rn(w_v, f1.y, __fmul_rn(acc[3], corr));
            acc[4] = __fmaf_rn(w_v, f2.x, __fmul_rn(acc[4], corr));
            acc[5] = __fmaf_rn(w_v, f2.y, __fmul_rn(acc[5], corr));
            acc[6] = __fmaf_rn(w_v, f3.x, __fmul_rn(acc[6], corr));
            acc[7] = __fmaf_rn(w_v, f3.y, __fmul_rn(acc[7], corr));
            m = m_new;
        }
    }

    __shared__ float sm[kFlashWarps];
    __shared__ float sl[kFlashWarps];
    __shared__ float sacc[kFlashWarps][kHD];
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
    #pragma unroll
    for (int i = 0; i < kAccPerLane; ++i)
        sacc[warp][lane * kAccPerLane + i] = acc[i];
    __syncthreads();

    float* out = scratch + ((size_t)h * SPLITS + split) * (HD + 2);
    float m_blk = -INFINITY;
    #pragma unroll
    for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
    if (threadIdx.x == 0) {
        float l_blk = 0.0f;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            l_blk += (sm[w] > -INFINITY) ? sl[w] * __expf(sm[w] - m_blk) : 0.0f;
        out[0] = m_blk;
        out[1] = l_blk;
    }
    for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            a += (sm[w] > -INFINITY) ? sacc[w][d] * __expf(sm[w] - m_blk) : 0.0f;
        out[2 + d] = a;
    }

    __syncthreads();
    __threadfence();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[h], 1u);
    __syncthreads();
    if (ticket != SPLITS - 1) return;
    __threadfence();

    const float* base = scratch + (size_t)h * SPLITS * (HD + 2);
    __shared__ float ssc[32];
    __shared__ float sinv_l;
    if (threadIdx.x == 0) {
        float m_glob = -INFINITY;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s)
            m_glob = fmaxf(m_glob, base[(size_t)s * (HD + 2)]);
        float l_glob = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s) {
            const float* part = base + (size_t)s * (HD + 2);
            float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
            ssc[s] = sc;
            l_glob += part[1] * sc;
        }
        sinv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
    }
    __syncthreads();
    float inv_l = sinv_l;

    for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
        float a = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s)
            a += base[(size_t)s * (HD + 2) + 2 + d] * ssc[s];
        outp[(size_t)h * HD + d] = __float2bfloat16(a * inv_l);
    }
    if (threadIdx.x == 0) fan_in[h] = 0u;

    NVK_PDL_EPILOG();
}

constexpr int kGshareSplits = 64;
constexpr int kGsharePosPerWarpTarget16_SIZES_S_EFF_SO_SHALLOW_PAYS_A_SMALL_REDUCE_AND_DEEP_GETS_ALL_SPLITS = 16;

template <int GROUP>
__global__ void __launch_bounds__(kFlashThreads, 2)
flash_splitk_fused_fp8_gshare_adaptive_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int NKV,
    int WINDOW,
    int RING,
    float scaling,
    const int* __restrict__ block_table,
    int block_size
) {
    constexpr int HD = kHD;
    constexpr int vc = HD / kWarp;
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    if (kvh >= NKV) return;

    NVK_PDL_PROLOG();

    const int total = n_total_dev[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int span = total - start;
    const int per_split = kFlashWarps
        * kGsharePosPerWarpTarget16_SIZES_S_EFF_SO_SHALLOW_PAYS_A_SMALL_REDUCE_AND_DEEP_GETS_ALL_SPLITS;
    int s_eff = (span + per_split - 1) / per_split;
    if (s_eff < 1) s_eff = 1;
    if (s_eff > kGshareSplits) s_eff = kGshareSplits;

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

    __shared__ float qsh[GROUP][HD];
    if (split < s_eff) {
        for (int i = threadIdx.x; i < GROUP * HD; i += kFlashThreads) {
            const int g = i / HD;
            const int d = i - g * HD;
            qsh[g][d] = __bfloat162float(q[((size_t)kvh * GROUP + g) * HD + d]);
        }
        __syncthreads();
        float m[GROUP];
        float l[GROUP];
        float acc[GROUP][vc];
        #pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            m[g] = -INFINITY;
            l[g] = 0.0f;
            #pragma unroll
            for (int i = 0; i < vc; ++i) acc[g][i] = 0.0f;
        }
        const int stride = s_eff * kFlashWarps;
        for (int p = start + split * kFlashWarps + warp; p < total; p += 2 * stride) {
            const int pb = p + stride;
            const bool live_b = pb < total;
            const int sp_a = kv_phys_slot(block_table, block_size, RING, p);
            const int sp_b =
                live_b ? kv_phys_slot(block_table, block_size, RING, pb) : sp_a;
            const size_t row_a = (size_t)sp_a * NKV + kvh;
            const size_t row_b = (size_t)sp_b * NKV + kvh;
            const float ks_a = k_scales[row_a];
            const float vs_a = v_scales[row_a];
            const float ks_b = k_scales[row_b];
            const float vs_b = v_scales[row_b];
            const uchar4* k4a = reinterpret_cast<const uchar4*>(k_fp8 + row_a * HD);
            const uchar4* k4b = reinterpret_cast<const uchar4*>(k_fp8 + row_b * HD);
            uchar4 kraw[2][2];
            kraw[0][0] = __ldg(&k4a[lane]);
            kraw[0][1] = __ldg(&k4a[lane + kWarp]);
            kraw[1][0] = __ldg(&k4b[lane]);
            kraw[1][1] = __ldg(&k4b[lane + kWarp]);
            uint2 vraw[2];
            vraw[0] = __ldg(reinterpret_cast<const uint2*>(
                v_fp8 + row_a * HD + (size_t)lane * vc));
            vraw[1] = __ldg(reinterpret_cast<const uint2*>(
                v_fp8 + row_b * HD + (size_t)lane * vc));
            float kf[2][8];
            float vf[2][vc];
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                float2 f01 = fp8x2_to_float2(
                    (unsigned short)(kraw[j][0].x | ((unsigned short)kraw[j][0].y << 8)));
                float2 f23 = fp8x2_to_float2(
                    (unsigned short)(kraw[j][0].z | ((unsigned short)kraw[j][0].w << 8)));
                kf[j][0] = f01.x; kf[j][1] = f01.y; kf[j][2] = f23.x; kf[j][3] = f23.y;
                float2 g01 = fp8x2_to_float2(
                    (unsigned short)(kraw[j][1].x | ((unsigned short)kraw[j][1].y << 8)));
                float2 g23 = fp8x2_to_float2(
                    (unsigned short)(kraw[j][1].z | ((unsigned short)kraw[j][1].w << 8)));
                kf[j][4] = g01.x; kf[j][5] = g01.y; kf[j][6] = g23.x; kf[j][7] = g23.y;
                float2 v0 = fp8x2_to_float2((unsigned short)(vraw[j].x & 0xffffu));
                float2 v1 = fp8x2_to_float2((unsigned short)(vraw[j].x >> 16));
                float2 v2 = fp8x2_to_float2((unsigned short)(vraw[j].y & 0xffffu));
                float2 v3 = fp8x2_to_float2((unsigned short)(vraw[j].y >> 16));
                vf[j][0] = v0.x; vf[j][1] = v0.y; vf[j][2] = v1.x; vf[j][3] = v1.y;
                vf[j][4] = v2.x; vf[j][5] = v2.y; vf[j][6] = v3.x; vf[j][7] = v3.y;
            }
            #pragma unroll
            for (int g = 0; g < GROUP; ++g) {
                const float* q0 = &qsh[g][lane * 4];
                const float* q1 = &qsh[g][(lane + kWarp) * 4];
                float pa = q0[0] * kf[0][0] + q0[1] * kf[0][1] + q0[2] * kf[0][2]
                         + q0[3] * kf[0][3] + q1[0] * kf[0][4] + q1[1] * kf[0][5]
                         + q1[2] * kf[0][6] + q1[3] * kf[0][7];
                float pb2 = q0[0] * kf[1][0] + q0[1] * kf[1][1] + q0[2] * kf[1][2]
                          + q0[3] * kf[1][3] + q1[0] * kf[1][4] + q1[1] * kf[1][5]
                          + q1[2] * kf[1][6] + q1[3] * kf[1][7];
                const float score_a = warp_sum(pa) * ks_a * scaling;
                const float score_b = warp_sum(pb2) * ks_b * scaling;
                {
                    const float m_new = fmaxf(m[g], score_a);
                    const float corr = __expf(m[g] - m_new);
                    const float w = __expf(score_a - m_new);
                    l[g] = l[g] * corr + w;
                    const float w_v = w * vs_a;
                    #pragma unroll
                    for (int i = 0; i < vc; ++i)
                        acc[g][i] = __fmaf_rn(w_v, vf[0][i], __fmul_rn(acc[g][i], corr));
                    m[g] = m_new;
                }
                if (live_b) {
                    const float m_new = fmaxf(m[g], score_b);
                    const float corr = __expf(m[g] - m_new);
                    const float w = __expf(score_b - m_new);
                    l[g] = l[g] * corr + w;
                    const float w_v = w * vs_b;
                    #pragma unroll
                    for (int i = 0; i < vc; ++i)
                        acc[g][i] = __fmaf_rn(w_v, vf[1][i], __fmul_rn(acc[g][i], corr));
                    m[g] = m_new;
                }
            }
        }

        __shared__ float smw[kFlashWarps];
        __shared__ float slw[kFlashWarps];
        __shared__ float sacc[kFlashWarps][HD];
        #pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            if (lane == 0) {
                smw[warp] = m[g];
                slw[warp] = l[g];
            }
            #pragma unroll
            for (int i = 0; i < vc; ++i) sacc[warp][lane * vc + i] = acc[g][i];
            __syncthreads();
            float* out =
                scratch + (((size_t)kvh * GROUP + g) * kGshareSplits + split) * (HD + 2);
            float m_blk = -INFINITY;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, smw[w]);
            if (threadIdx.x == 0) {
                float l_blk = 0.0f;
                #pragma unroll
                for (int w = 0; w < kFlashWarps; ++w)
                    l_blk += (smw[w] > -INFINITY) ? slw[w] * __expf(smw[w] - m_blk) : 0.0f;
                out[0] = m_blk;
                out[1] = l_blk;
            }
            for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
                float a = 0.0f;
                #pragma unroll
                for (int w = 0; w < kFlashWarps; ++w)
                    a += (smw[w] > -INFINITY) ? sacc[w][d] * __expf(smw[w] - m_blk) : 0.0f;
                out[2 + d] = a;
            }
            __syncthreads();
        }
    }

    __threadfence();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[kvh], 1u);
    __syncthreads();
    if (ticket != (unsigned)(kGshareSplits - 1)) return;
    __threadfence();

    __shared__ float ssc[kGshareSplits];
    __shared__ float sinv_l;
    for (int g = 0; g < GROUP; ++g) {
        const float* base =
            scratch + ((size_t)kvh * GROUP + g) * kGshareSplits * (HD + 2);
        if (threadIdx.x == 0) {
            float m_glob = -INFINITY;
            for (int s = 0; s < s_eff; ++s)
                m_glob = fmaxf(m_glob, base[(size_t)s * (HD + 2)]);
            float l_glob = 0.0f;
            for (int s = 0; s < s_eff; ++s) {
                const float* part = base + (size_t)s * (HD + 2);
                float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
                ssc[s] = sc;
                l_glob += part[1] * sc;
            }
            sinv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
        }
        __syncthreads();
        const float inv_l = sinv_l;
        for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
            float a = 0.0f;
            for (int s = 0; s < s_eff; ++s)
                a += base[(size_t)s * (HD + 2) + 2 + d] * ssc[s];
            outp[((size_t)kvh * GROUP + g) * HD + d] = __float2bfloat16(a * inv_l);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) fan_in[kvh] = 0u;

    NVK_PDL_EPILOG();
}

inline int splits_env_shared_with_the_fused_kernel_so_scratch_sized_by_flash_splitk_scratch_elems_always_fits() {
    static int forced = [] {
        const char* e = getenv("NV_E4B_FLASH_SPLITS");
        if (e == nullptr) return 0;
        int x = atoi(e);
        return (x == 8 || x == 16 || x == 32) ? x : 0;
    }();
    return forced;
}

inline int sm_count() {
    static int sm = [] {
        int dev = 0, n = 148;
        if (cudaGetDevice(&dev) == cudaSuccess)
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 148;
    }();
    return sm;
}

inline int splits_pick_same_formula_as_flash_splits_pick(int NH) {
    int forced = splits_env_shared_with_the_fused_kernel_so_scratch_sized_by_flash_splitk_scratch_elems_always_fits();
    if (forced) return forced;
    if (NH <= 0) return 16;
    const long target = 2L * sm_count();
    if ((long)NH * 8 >= target) return 8;
    if ((long)NH * 16 >= target) return 16;
    return 32;
}

}

extern "C" int nv_kernels_flash_decode_fused_fp8kv_ilp4_paged(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling,
    const int* block_table,
    int block_size
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD != q38md::kHD) return -21;
    if ((NH % NKV) != 0) return -1;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    int sp = q38md::splits_pick_same_formula_as_flash_splits_pick(NH);
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(q38md::kFlashThreads), 0, cs);
        switch (sp) {
            case 8:
                cudaLaunchKernelEx(
                    &cfg, q38md::flash_splitk_fused_fp8_ilp4_kernel<8>,
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling,
                    block_table, block_size);
                break;
            case 32:
                cudaLaunchKernelEx(
                    &cfg, q38md::flash_splitk_fused_fp8_ilp4_kernel<32>,
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling,
                    block_table, block_size);
                break;
            default:
                cudaLaunchKernelEx(
                    &cfg, q38md::flash_splitk_fused_fp8_ilp4_kernel<16>,
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling,
                    block_table, block_size);
                break;
        }
        return (int)cudaGetLastError();
    }
    switch (sp) {
        case 8:
            q38md::flash_splitk_fused_fp8_ilp4_kernel<8>
                <<<grid, q38md::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling,
                    block_table, block_size);
            break;
        case 32:
            q38md::flash_splitk_fused_fp8_ilp4_kernel<32>
                <<<grid, q38md::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling,
                    block_table, block_size);
            break;
        default:
            q38md::flash_splitk_fused_fp8_ilp4_kernel<16>
                <<<grid, q38md::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling,
                    block_table, block_size);
            break;
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_flash_gshare_scratch_elems(int NH, int HD) {
    return NH * q38md::kGshareSplits * (HD + 2);
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv_gshare_paged(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling,
    const int* block_table,
    int block_size
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD != q38md::kHD) return -21;
    if ((NH % NKV) != 0) return -22;
    const int group = NH / NKV;
    if (group != 6 && group != 8) return -23;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    dim3 grid((unsigned)NKV, (unsigned)q38md::kGshareSplits);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(q38md::kFlashThreads), 0, cs);
        if (group == 6) {
            cudaLaunchKernelEx(
                &cfg, q38md::flash_splitk_fused_fp8_gshare_adaptive_kernel<6>,
                qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                n_total_dev, NKV, WINDOW, RING, scaling, block_table, block_size);
        } else {
            cudaLaunchKernelEx(
                &cfg, q38md::flash_splitk_fused_fp8_gshare_adaptive_kernel<8>,
                qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                n_total_dev, NKV, WINDOW, RING, scaling, block_table, block_size);
        }
        return (int)cudaGetLastError();
    }
    if (group == 6) {
        q38md::flash_splitk_fused_fp8_gshare_adaptive_kernel<6>
            <<<grid, q38md::kFlashThreads, 0, cs>>>(
                qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                n_total_dev, NKV, WINDOW, RING, scaling, block_table, block_size);
    } else {
        q38md::flash_splitk_fused_fp8_gshare_adaptive_kernel<8>
            <<<grid, q38md::kFlashThreads, 0, cs>>>(
                qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                n_total_dev, NKV, WINDOW, RING, scaling, block_table, block_size);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv_gshare(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling
) {
    return nv_kernels_flash_decode_fused_fp8kv_gshare_paged(
        stream, q, k_fp8, v_fp8, k_scales, v_scales, out, n_total_dev,
        scratch, fan_in, NH, NKV, HD, WINDOW, RING, scaling, nullptr, 0);
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv_ilp4(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling
) {
    return nv_kernels_flash_decode_fused_fp8kv_ilp4_paged(
        stream, q, k_fp8, v_fp8, k_scales, v_scales, out, n_total_dev,
        scratch, fan_in, NH, NKV, HD, WINDOW, RING, scaling, nullptr, 0);
}
