#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

#include "fused_tc_common.cuh"

namespace lagfusedtc {

using fusedtc::half2_bits;
using fusedtc::ldmatrix_x4_trans;
using fusedtc::mma_m16n8k16_bf16;
using fusedtc::mma_m16n8k16_f16;
using fusedtc::warp_max_over_gid;
using fusedtc::warp_sum_over_gid;

constexpr int kWarp = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarp * kWarps;
constexpr int kGroup6Is48QOver8KvTheLagunaFullAttentionShape = 6;
constexpr int kHD = 128;
constexpr int kMaxRows5IsTheServedK4DraftWidthAndKeeps2NchWarpsUnder8 = 5;
constexpr int kMaxSplits64KeepsRowsTimesSplitsUnderTheCombineScaleFloats = 64;
constexpr int kCombineScaleFloats2048HoldRowsTimesSplitsSoEveryRowCombinesInParallel = 2048;
constexpr int kTileP = 32;
constexpr int kKSteps = kHD / 16;
constexpr int kKRowHalvesPlus8KeepsTheAFragmentWordLoadsOffBankRepeat = kHD + 8;
constexpr int kVRowHalvesPlus8KeepsLdmatrixTransRowsOffBankRepeat = kHD + 8;
constexpr int kPRowHalvesPlus8KeepsTheBFragmentLoadsOffBankRepeat = kTileP + 8;
constexpr int kAIs1BecauseEightWarpsCoverHd128InSixteenWideSlices = kHD / 16 / kWarps;

__device__ __forceinline__ void load_q_fragments_bf16_exact(
    unsigned (&qb0)[kKSteps],
    unsigned (&qb1)[kKSteps],
    const __nv_bfloat16* __restrict__ q,
    int qr,
    int rows,
    int NH,
    int kvh,
    int tg
) {
    const int G = kGroup6Is48QOver8KvTheLagunaFullAttentionShape;
    const __nv_bfloat16* qrow = (qr < rows)
        ? q + ((size_t)(qr / G) * NH + (size_t)kvh * G + (qr % G)) * kHD
        : nullptr;
    #pragma unroll
    for (int kc = 0; kc < kKSteps; ++kc) {
        const int col = kc * 16 + tg * 2;
        if (qrow != nullptr) {
            qb0[kc] = *reinterpret_cast<const unsigned*>(qrow + col);
            qb1[kc] = *reinterpret_cast<const unsigned*>(qrow + col + 8);
        } else {
            qb0[kc] = 0u;
            qb1[kc] = 0u;
        }
    }
}

template <int NCH>
__device__ __forceinline__ void store_and_combine(
    const float (&o)[NCH][4],
    const float (&m_r)[2],
    const float (&l_r)[2],
    bool qk_row_owner,
    int r0,
    int rows,
    int warp,
    int gid,
    int tg,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    int kvh,
    int split,
    int SPLITS,
    int NH
) {
    const int G = kGroup6Is48QOver8KvTheLagunaFullAttentionShape;
    if (qk_row_owner) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = r0 + i;
            if (r < rows) {
                float* out = scratch + (((size_t)kvh * rows + r) * SPLITS + split) * (kHD + 2);
                out[0] = m_r[i];
                out[1] = l_r[i];
            }
        }
    }
    const int hd0 = warp * 16 + gid;
    #pragma unroll
    for (int n = 0; n < NCH; ++n) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int r = n * 8 + tg * 2 + i;
            if (r < rows) {
                float* out = scratch + (((size_t)kvh * rows + r) * SPLITS + split) * (kHD + 2);
                out[2 + hd0] = o[n][i];
                out[2 + hd0 + 8] = o[n][2 + i];
            }
        }
    }

    __threadfence();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[kvh], 1u);
    __syncthreads();
    if (ticket == SPLITS - 1) {
        __threadfence();
        __shared__ float ssc[kCombineScaleFloats2048HoldRowsTimesSplitsSoEveryRowCombinesInParallel];
        __shared__ float sinv_l[kMaxRows5IsTheServedK4DraftWidthAndKeeps2NchWarpsUnder8 * kGroup6Is48QOver8KvTheLagunaFullAttentionShape];
        for (int r = threadIdx.x; r < rows; r += kThreads) {
            const float* base = scratch + ((size_t)kvh * rows + r) * SPLITS * (kHD + 2);
            float m_glob = -INFINITY;
            for (int s = 0; s < SPLITS; ++s)
                m_glob = fmaxf(m_glob, base[(size_t)s * (kHD + 2)]);
            float l_glob = 0.0f;
            for (int s = 0; s < SPLITS; ++s) {
                const float* part = base + (size_t)s * (kHD + 2);
                float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
                ssc[r * SPLITS + s] = sc;
                l_glob += part[1] * sc;
            }
            sinv_l[r] = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
        }
        __syncthreads();
        for (int idx = threadIdx.x; idx < rows * kHD; idx += kThreads) {
            const int r = idx / kHD;
            const int d = idx - r * kHD;
            const int rq = r / G;
            const int g = r - rq * G;
            const float* base = scratch + ((size_t)kvh * rows + r) * SPLITS * (kHD + 2);
            const float* sc = ssc + r * SPLITS;
            float acc = 0.0f;
            for (int s = 0; s < SPLITS; ++s)
                acc += base[(size_t)s * (kHD + 2) + 2 + d] * sc[s];
            outp[((size_t)rq * NH + (size_t)kvh * G + g) * kHD + d] =
                __float2bfloat16(acc * sinv_l[r]);
        }
        __syncthreads();
        if (threadIdx.x == 0) fan_in[kvh] = 0u;
    }
}

template <int NCH, int KVQ>
__global__ void __launch_bounds__(kThreads, 2) laguna_flash_verify_fused_tc_kernel(
    const __nv_bfloat16* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ committed,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    int NKV,
    int SPLITS,
    int M,
    int capacity,
    float scaling
) {
    static_assert(NCH >= 1 && 2 * NCH <= kWarps, "every (position half, row chunk) qk task owns a warp");
    constexpr int G = kGroup6Is48QOver8KvTheLagunaFullAttentionShape;
    constexpr int kRows = NCH * 8;
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    if (kvh >= NKV || split >= SPLITS) return;
    const int NH = NKV * G;
    const int rows = M * G;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int gid = lane >> 2;
    const int tg = lane & 3;
    const int before = committed[0];
    if (before < 0 || before > capacity - M) {
        if (split == 0) {
            for (int i = threadIdx.x; i < M * G * kHD; i += kThreads) {
                const int rq = i / (G * kHD);
                const int rem = i - rq * G * kHD;
                outp[((size_t)rq * NH + (size_t)kvh * G) * kHD + rem] = __float2bfloat16(NAN);
            }
        }
        return;
    }
    const int total = before + M;

    __shared__ __align__(16) __nv_bfloat16 kh[kTileP][kKRowHalvesPlus8KeepsTheAFragmentWordLoadsOffBankRepeat];
    __shared__ __align__(16) __half vh[kTileP][kVRowHalvesPlus8KeepsLdmatrixTransRowsOffBankRepeat];
    __shared__ __align__(16) __half ph[kRows][kPRowHalvesPlus8KeepsTheBFragmentLoadsOffBankRepeat];
    __shared__ float cmax_sh[2][kRows];
    __shared__ float csum_sh[2][kRows];
    __shared__ float corr_sh[kRows];

    const bool qk = warp < 2 * NCH;
    const int mt = qk ? warp / NCH : 0;
    const int nq = qk ? warp - mt * NCH : 0;
    const int r0 = nq * 8 + tg * 2;
    const int r1 = r0 + 1;
    int tot[2];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = r0 + i;
        tot[i] = (r < rows) ? (before + 1 + r / G) : 0;
    }

    unsigned qb0[kKSteps];
    unsigned qb1[kKSteps];
    if (qk) load_q_fragments_bf16_exact(qb0, qb1, q, nq * 8 + gid, rows, NH, kvh, tg);

    float m_r[2] = {-INFINITY, -INFINITY};
    float l_r[2] = {0.0f, 0.0f};
    float o[NCH][4];
    #pragma unroll
    for (int n = 0; n < NCH; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) o[n][i] = 0.0f;
    }

    const int spos = threadIdx.x >> 3;
    const int schunk = threadIdx.x & 7;
    uint4 kpre[2];
    uint4 vpre[2];
    float kscale = 1.0f;
    float vscale = 1.0f;
    auto prefetch = [&](int base) {
        const int p = base + spos;
        if (p < total) {
            const size_t row = (size_t)p * NKV + kvh;
            if constexpr (KVQ != 0) {
                const uint4* kr = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(k) + row * kHD);
                const uint4* vr = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(v) + row * kHD);
                kpre[0] = __ldg(kr + schunk);
                vpre[0] = __ldg(vr + schunk);
                kscale = __ldg(k_scales + row);
                vscale = __ldg(v_scales + row);
            } else {
                const uint4* kr = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(k) + row * kHD);
                const uint4* vr = reinterpret_cast<const uint4*>(reinterpret_cast<const __nv_bfloat16*>(v) + row * kHD);
                kpre[0] = __ldg(kr + schunk * 2);
                kpre[1] = __ldg(kr + schunk * 2 + 1);
                vpre[0] = __ldg(vr + schunk * 2);
                vpre[1] = __ldg(vr + schunk * 2 + 1);
            }
        } else {
            kpre[0] = kpre[1] = vpre[0] = vpre[1] = make_uint4(0u, 0u, 0u, 0u);
            kscale = vscale = 0.0f;
        }
    };
    auto stage = [&]() {
        if constexpr (KVQ != 0) {
            const unsigned kw[4] = {kpre[0].x, kpre[0].y, kpre[0].z, kpre[0].w};
            const unsigned vw[4] = {vpre[0].x, vpre[0].y, vpre[0].z, vpre[0].w};
            unsigned kb[8];
            unsigned hv[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                float2 kf;
                float2 vf;
                if constexpr (KVQ == 2) {
                    const unsigned kwj = (kw[j >> 1] >> ((j & 1) * 16)) ^ 0x8080u;
                    const unsigned vwj = (vw[j >> 1] >> ((j & 1) * 16)) ^ 0x8080u;
                    kf = make_float2(__uint_as_float(__byte_perm(kwj, 0x4B000000u, 0x7650u)) - 8388736.0f,
                                     __uint_as_float(__byte_perm(kwj, 0x4B000000u, 0x7651u)) - 8388736.0f);
                    vf = make_float2(__uint_as_float(__byte_perm(vwj, 0x4B000000u, 0x7650u)) - 8388736.0f,
                                     __uint_as_float(__byte_perm(vwj, 0x4B000000u, 0x7651u)) - 8388736.0f);
                } else {
                    const __nv_fp8x2_storage_t kp = static_cast<__nv_fp8x2_storage_t>(kw[j >> 1] >> ((j & 1) * 16));
                    const __nv_fp8x2_storage_t vp = static_cast<__nv_fp8x2_storage_t>(vw[j >> 1] >> ((j & 1) * 16));
                    const __half2_raw kh2 = __nv_cvt_fp8x2_to_halfraw2(kp, __NV_E4M3);
                    const __half2_raw vh2 = __nv_cvt_fp8x2_to_halfraw2(vp, __NV_E4M3);
                    kf = __half22float2(*reinterpret_cast<const __half2*>(&kh2));
                    vf = __half22float2(*reinterpret_cast<const __half2*>(&vh2));
                }
                const __nv_bfloat16 kb0 = __float2bfloat16(kf.x * kscale);
                const __nv_bfloat16 kb1 = __float2bfloat16(kf.y * kscale);
                const __nv_bfloat16 vb0 = __float2bfloat16(vf.x * vscale);
                const __nv_bfloat16 vb1 = __float2bfloat16(vf.y * vscale);
                kb[j] = (unsigned)__bfloat16_as_ushort(kb0) | ((unsigned)__bfloat16_as_ushort(kb1) << 16);
                hv[j] = half2_bits(__bfloat162float(vb0), __bfloat162float(vb1));
            }
            uint4* kdst = reinterpret_cast<uint4*>(&kh[spos][schunk * 16]);
            kdst[0] = make_uint4(kb[0], kb[1], kb[2], kb[3]);
            kdst[1] = make_uint4(kb[4], kb[5], kb[6], kb[7]);
            uint4* vdst = reinterpret_cast<uint4*>(&vh[spos][schunk * 16]);
            vdst[0] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
            vdst[1] = make_uint4(hv[4], hv[5], hv[6], hv[7]);
        } else {
            uint4* kdst = reinterpret_cast<uint4*>(&kh[spos][schunk * 16]);
            kdst[0] = kpre[0];
            kdst[1] = kpre[1];
            const unsigned words[8] = {vpre[0].x, vpre[0].y, vpre[0].z, vpre[0].w,
                                       vpre[1].x, vpre[1].y, vpre[1].z, vpre[1].w};
            unsigned hv[8];
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&words[j]));
                hv[j] = half2_bits(f.x, f.y);
            }
            uint4* vdst = reinterpret_cast<uint4*>(&vh[spos][schunk * 16]);
            vdst[0] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
            vdst[1] = make_uint4(hv[4], hv[5], hv[6], hv[7]);
        }
    };

    const int tile_stride = SPLITS * kTileP;
    const int tile0 = split * kTileP;
    prefetch(tile0);
    for (int base = tile0; base < total; base += tile_stride) {
        __syncthreads();
        stage();
        prefetch(base + tile_stride);
        __syncthreads();

        float s[4];
        bool lv[4];
        if (qk) {
            float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            const __nv_bfloat16* ka = &kh[mt * 16 + gid][tg * 2];
            const __nv_bfloat16* kb = &kh[mt * 16 + gid + 8][tg * 2];
            #pragma unroll
            for (int kc = 0; kc < kKSteps; ++kc) {
                mma_m16n8k16_bf16(
                    c,
                    *reinterpret_cast<const unsigned*>(ka + kc * 16),
                    *reinterpret_cast<const unsigned*>(kb + kc * 16),
                    *reinterpret_cast<const unsigned*>(ka + kc * 16 + 8),
                    *reinterpret_cast<const unsigned*>(kb + kc * 16 + 8),
                    qb0[kc],
                    qb1[kc]);
            }
            const int p0 = base + mt * 16 + gid;
            const int p1 = p0 + 8;
            lv[0] = p0 < tot[0];
            lv[1] = p0 < tot[1];
            lv[2] = p1 < tot[0];
            lv[3] = p1 < tot[1];
            s[0] = lv[0] ? c[0] * scaling : -INFINITY;
            s[1] = lv[1] ? c[1] * scaling : -INFINITY;
            s[2] = lv[2] ? c[2] * scaling : -INFINITY;
            s[3] = lv[3] ? c[3] * scaling : -INFINITY;
            const float mx0 = warp_max_over_gid(fmaxf(s[0], s[2]));
            const float mx1 = warp_max_over_gid(fmaxf(s[1], s[3]));
            if (gid == 0) {
                cmax_sh[mt][r0] = mx0;
                cmax_sh[mt][r1] = mx1;
            }
        }
        __syncthreads();

        float cr[2] = {0.0f, 0.0f};
        if (qk) {
            float mn[2];
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = r0 + i;
                mn[i] = fmaxf(m_r[i], fmaxf(cmax_sh[0][r], cmax_sh[1][r]));
                cr[i] = (m_r[i] == -INFINITY) ? 0.0f : __expf(m_r[i] - mn[i]);
            }
            float p[4];
            p[0] = lv[0] ? __expf(s[0] - mn[0]) : 0.0f;
            p[1] = lv[1] ? __expf(s[1] - mn[1]) : 0.0f;
            p[2] = lv[2] ? __expf(s[2] - mn[0]) : 0.0f;
            p[3] = lv[3] ? __expf(s[3] - mn[1]) : 0.0f;
            const int pc = mt * 16 + gid;
            ph[r0][pc] = __float2half_rn(p[0]);
            ph[r1][pc] = __float2half_rn(p[1]);
            ph[r0][pc + 8] = __float2half_rn(p[2]);
            ph[r1][pc + 8] = __float2half_rn(p[3]);
            const float sm0 = warp_sum_over_gid(p[0] + p[2]);
            const float sm1 = warp_sum_over_gid(p[1] + p[3]);
            if (gid == 0) {
                csum_sh[mt][r0] = sm0;
                csum_sh[mt][r1] = sm1;
                if (mt == 0) {
                    corr_sh[r0] = cr[0];
                    corr_sh[r1] = cr[1];
                }
            }
            m_r[0] = mn[0];
            m_r[1] = mn[1];
        }
        __syncthreads();

        if (qk) {
            #pragma unroll
            for (int i = 0; i < 2; ++i)
                l_r[i] = l_r[i] * cr[i] + (csum_sh[0][r0 + i] + csum_sh[1][r0 + i]);
        }

        #pragma unroll
        for (int n = 0; n < NCH; ++n) {
            const float c0 = corr_sh[n * 8 + tg * 2];
            const float c1 = corr_sh[n * 8 + tg * 2 + 1];
            o[n][0] *= c0;
            o[n][1] *= c1;
            o[n][2] *= c0;
            o[n][3] *= c1;
        }
        const int lm = lane >> 3;
        const int lr = lane & 7;
        #pragma unroll
        for (int ks = 0; ks < kTileP / 16; ++ks) {
            unsigned pb0[NCH];
            unsigned pb1[NCH];
            #pragma unroll
            for (int n = 0; n < NCH; ++n) {
                const __half* prow = &ph[n * 8 + gid][ks * 16 + tg * 2];
                pb0[n] = *reinterpret_cast<const unsigned*>(prow);
                pb1[n] = *reinterpret_cast<const unsigned*>(prow + 8);
            }
            const int hd0 = warp * 16;
            const int prow = ks * 16 + lr + ((lm & 2) ? 8 : 0);
            const int hcol = hd0 + ((lm & 1) ? 8 : 0);
            unsigned va[4];
            ldmatrix_x4_trans(va, &vh[prow][hcol]);
            #pragma unroll
            for (int n = 0; n < NCH; ++n)
                mma_m16n8k16_f16(o[n], va[0], va[1], va[2], va[3], pb0[n], pb1[n]);
        }
    }

    store_and_combine<NCH>(
        o, m_r, l_r, qk && mt == 0 && gid == 0, r0, rows, warp, gid, tg, scratch, fan_in,
        outp, kvh, split, SPLITS, NH);
}

}

template <int KVQ>
static int laguna_flash_verify_fused_tc_launch(
    void* stream, const uint16_t* q, const void* k, const void* v, const float* k_scales, const float* v_scales,
    uint16_t* out, const int* committed, float* scratch, unsigned int* fan_in,
    int n_q, int n_kv, int head_dim, int rows, int capacity, int splits, float scale
) {
    using namespace lagfusedtc;
    if (n_q <= 0 || n_kv <= 0 || head_dim != kHD || n_q % n_kv != 0
        || n_q / n_kv != kGroup6Is48QOver8KvTheLagunaFullAttentionShape
        || rows < 1 || rows > kMaxRows5IsTheServedK4DraftWidthAndKeeps2NchWarpsUnder8
        || capacity < rows || splits < 1 || splits > kMaxSplits64KeepsRowsTimesSplitsUnderTheCombineScaleFloats
        || !isfinite(scale) || q == nullptr || k == nullptr || v == nullptr || out == nullptr
        || committed == nullptr || scratch == nullptr || fan_in == nullptr
        || (KVQ != 0 && (k_scales == nullptr || v_scales == nullptr)))
        return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid((unsigned)n_kv, (unsigned)splits);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    const int nch = (rows * kGroup6Is48QOver8KvTheLagunaFullAttentionShape + 7) / 8;
#define NV_LAG_FTC(N) \
    case N: laguna_flash_verify_fused_tc_kernel<N, KVQ><<<grid, kThreads, 0, s>>>(qb, k, v, k_scales, v_scales, ob, committed, scratch, fan_in, n_kv, splits, rows, capacity, scale); break;
    switch (nch) {
        NV_LAG_FTC(1)
        NV_LAG_FTC(2)
        NV_LAG_FTC(3)
        NV_LAG_FTC(4)
        default: return -1;
    }
#undef NV_LAG_FTC
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_laguna_flash_decode_verify_fused_tc(
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
    int splits,
    float scale
) {
    return laguna_flash_verify_fused_tc_launch<0>(stream, q, k, v, nullptr, nullptr, out, committed, scratch, fan_in, n_q, n_kv, head_dim, rows, capacity, splits, scale);
}

extern "C" int nv_kernels_laguna_flash_decode_verify_fused_tc_fp8(
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
    int kvq
) {
    if (kvq == 2) return laguna_flash_verify_fused_tc_launch<2>(stream, q, k, v, k_scales, v_scales, out, committed, scratch, fan_in, n_q, n_kv, head_dim, rows, capacity, splits, scale);
    if (kvq == 1) return laguna_flash_verify_fused_tc_launch<1>(stream, q, k, v, k_scales, v_scales, out, committed, scratch, fan_in, n_q, n_kv, head_dim, rows, capacity, splits, scale);
    return -1;
}
