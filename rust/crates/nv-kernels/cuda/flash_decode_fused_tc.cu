#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

#include "nvk_pdl.cuh"
#include "fused_tc_common.cuh"

extern "C" int nv_kernels_flash_decode_kvshare_splits_pick(int NKV);

namespace fusedtc {

constexpr int kWarp = 32;
constexpr int kWarps = 8;
constexpr int kGroup = 6;
constexpr int kHD = 256;
constexpr int kMaxSplits = 128;
constexpr int kMaxRows = 8;
constexpr int kCombineScaleFloats2048HoldRowsTimesSplitsSoEveryRowCombinesInParallel = 2048;
template <int NCH>
struct Geo {
    static constexpr int kWarpsT = (2 * NCH > kWarps) ? 2 * kWarps : kWarps;
    static constexpr int kThreadsT = kWarp * kWarpsT;
    static constexpr int kA = kHD / 16 / kWarpsT;
    static constexpr int kBlocksPerSm = (kWarpsT == kWarps) ? 2 : 1;
};
constexpr int kTileP = 32;
constexpr int kKSteps = kHD / 16;
constexpr int kKRowBytesPlus16KeepsTheFourWordAFragmentLoadsOffBankRepeat = kHD + 16;
constexpr int kVRowHalvesPlus8KeepsLdmatrixTransRowsOffBankRepeat = kHD + 8;
constexpr int kPRowHalvesPlus8KeepsTheBFragmentLoadsOffBankRepeat = kTileP + 8;
constexpr int kFusedTcArmIsShapedForTheQwen38Group6ShapeRcMinus29 = -29;

__device__ __forceinline__ int phys_slot(
    const int* __restrict__ block_table, int block_size, int RING, int p
) {
    if (block_table != nullptr) {
        int blk = p / block_size;
        return block_table[blk] * block_size + (p - blk * block_size);
    }
    return (RING > 0) ? (p % RING) : p;
}

struct RowMask {
    int tot[2];
    int st[2];
    bool holes[2];
    __device__ __forceinline__ bool live(int p, int i, int hole_lo, int hole_hi) const {
        return p < tot[i] && p >= st[i] && !(holes[i] && p >= hole_lo && p < hole_hi);
    }
};

__device__ __forceinline__ RowMask row_mask_for(
    int r0, int rows, int total, int M, int WINDOW, int hole_from_row
) {
    RowMask rm;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int r = r0 + i;
        const int rq = (r < rows) ? r / kGroup : 0;
        rm.tot[i] = (r < rows) ? (total - (M - 1) + rq) : 0;
        rm.st[i] = (WINDOW > 0 && rm.tot[i] > WINDOW) ? (rm.tot[i] - WINDOW) : 0;
        rm.holes[i] = rq >= hole_from_row;
    }
    return rm;
}

__device__ __forceinline__ void load_q_fragments(
    unsigned (&qb0)[kKSteps],
    unsigned (&qb1)[kKSteps],
    const __nv_bfloat16* __restrict__ q,
    int qr,
    int rows,
    int NH,
    int kvh,
    int tg
) {
    const __nv_bfloat16* qrow = (qr < rows)
        ? q + ((size_t)(qr / kGroup) * NH + (size_t)kvh * kGroup + (qr % kGroup)) * kHD
        : nullptr;
    #pragma unroll
    for (int kc = 0; kc < kKSteps; ++kc) {
        const int col = kc * 16 + tg * 2;
        if (qrow != nullptr) {
            const float2 lo = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(qrow + col));
            const float2 hi = __bfloat1622float2(
                *reinterpret_cast<const __nv_bfloat162*>(qrow + col + 8));
            qb0[kc] = half2_bits(lo.x, lo.y);
            qb1[kc] = half2_bits(hi.x, hi.y);
        } else {
            qb0[kc] = 0u;
            qb1[kc] = 0u;
        }
    }
}

template <int NCH, int KA, int KT>
__device__ __forceinline__ void fused_tc_store_and_combine(
    const float (&o)[KA][NCH][4],
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
    #pragma unroll
    for (int a = 0; a < KA; ++a) {
        const int hd0 = (warp * KA + a) * 16 + gid;
        #pragma unroll
        for (int n = 0; n < NCH; ++n) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int r = n * 8 + tg * 2 + i;
                if (r < rows) {
                    float* out = scratch + (((size_t)kvh * rows + r) * SPLITS + split) * (kHD + 2);
                    out[2 + hd0] = o[a][n][i];
                    out[2 + hd0 + 8] = o[a][n][2 + i];
                }
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
        __shared__ float sinv_l[kMaxRows * kGroup];
        for (int r = threadIdx.x; r < rows; r += KT) {
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
        for (int idx = threadIdx.x; idx < rows * kHD; idx += KT) {
            const int r = idx / kHD;
            const int d = idx - r * kHD;
            const int rq = r / kGroup;
            const int g = r - rq * kGroup;
            const float* base = scratch + ((size_t)kvh * rows + r) * SPLITS * (kHD + 2);
            const float* sc = ssc + r * SPLITS;
            float acc = 0.0f;
            for (int s = 0; s < SPLITS; ++s)
                acc += base[(size_t)s * (kHD + 2) + 2 + d] * sc[s];
            outp[((size_t)rq * NH + (size_t)kvh * kGroup + g) * kHD + d] =
                __float2bfloat16(acc * sinv_l[r]);
        }
        __syncthreads();
        if (threadIdx.x == 0) fan_in[kvh] = 0u;
    }
}

template <int NCH>
__global__ void __launch_bounds__(Geo<NCH>::kThreadsT, Geo<NCH>::kBlocksPerSm) flash_decode_fused_tc_kernel(
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
    int SPLITS,
    int WINDOW,
    int RING,
    float scaling,
    const int* __restrict__ block_table,
    int block_size,
    int delta,
    int M,
    int hole_from_row,
    int hole_lo_delta,
    int hole_hi_delta
) {
    static_assert(NCH >= 1 && 2 * NCH <= Geo<NCH>::kWarpsT, "every (position half, row chunk) qk task owns a warp");
    constexpr int kRows = NCH * 8;
    constexpr int kA = Geo<NCH>::kA;
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    if (kvh >= NKV || split >= SPLITS) return;

    NVK_PDL_PROLOG();

    const int total = n_total_dev[0] - delta;
    const int hole_lo = n_total_dev[0] - hole_lo_delta;
    const int hole_hi = n_total_dev[0] - hole_hi_delta;
    const int NH = NKV * kGroup;
    const int rows = M * kGroup;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int gid = lane >> 2;
    const int tg = lane & 3;
    const int total_q0 = total - (M - 1);
    const int start = (WINDOW > 0 && total_q0 > WINDOW) ? (total_q0 - WINDOW) : 0;

    __shared__ __align__(16) uint8_t kh[kTileP][kKRowBytesPlus16KeepsTheFourWordAFragmentLoadsOffBankRepeat];
    __shared__ __align__(16) __half vh[kTileP][kVRowHalvesPlus8KeepsLdmatrixTransRowsOffBankRepeat];
    __shared__ __align__(16) __half ph[kRows][kPRowHalvesPlus8KeepsTheBFragmentLoadsOffBankRepeat];
    __shared__ float ks_sh[kTileP];
    __shared__ float cmax_sh[2][kRows];
    __shared__ float csum_sh[2][kRows];
    __shared__ float corr_sh[kRows];

    const bool qk = warp < 2 * NCH;
    const int mt = qk ? warp / NCH : 0;
    const int nq = qk ? warp - mt * NCH : 0;
    const int r0 = nq * 8 + tg * 2;
    const int r1 = r0 + 1;
    const RowMask rm = row_mask_for(r0, rows, total, M, WINDOW, hole_from_row);

    unsigned qb0[kKSteps];
    unsigned qb1[kKSteps];
    if (qk) load_q_fragments(qb0, qb1, q, nq * 8 + gid, rows, NH, kvh, tg);

    float m_r[2] = {-INFINITY, -INFINITY};
    float l_r[2] = {0.0f, 0.0f};
    float o[kA][NCH][4];
    #pragma unroll
    for (int a = 0; a < kA; ++a) {
        #pragma unroll
        for (int n = 0; n < NCH; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[a][n][i] = 0.0f;
        }
    }

    const int spos = threadIdx.x >> 3;
    const int schunk = threadIdx.x & 7;
    const bool stager = spos < kTileP;
    uint4 kpre[2];
    uint4 vpre[2];
    float kspre = 0.0f;
    float vspre = 0.0f;
    auto prefetch = [&](int base) {
        const int p = base + spos;
        if (stager && base < total && p < total) {
            const int sp = phys_slot(block_table, block_size, RING, p);
            const size_t row = (size_t)sp * NKV + kvh;
            const uint4* kr = reinterpret_cast<const uint4*>(k_fp8 + row * kHD);
            const uint4* vr = reinterpret_cast<const uint4*>(v_fp8 + row * kHD);
            kpre[0] = __ldg(kr + schunk);
            kpre[1] = __ldg(kr + schunk + 8);
            vpre[0] = __ldg(vr + schunk);
            vpre[1] = __ldg(vr + schunk + 8);
            kspre = k_scales[row];
            vspre = v_scales[row];
        } else {
            kpre[0] = kpre[1] = vpre[0] = vpre[1] = make_uint4(0u, 0u, 0u, 0u);
            kspre = 0.0f;
            vspre = 0.0f;
        }
    };
    auto stage = [&]() {
        if (!stager) return;
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int c = schunk + 8 * i;
            const uint4 w = kpre[i];
            uint4 d;
            d.x = __byte_perm(w.x, w.z, 0x5410);
            d.y = __byte_perm(w.x, w.z, 0x7632);
            d.z = __byte_perm(w.y, w.w, 0x5410);
            d.w = __byte_perm(w.y, w.w, 0x7632);
            *reinterpret_cast<uint4*>(&kh[spos][c * 16]) = d;
            const unsigned words[4] = {vpre[i].x, vpre[i].y, vpre[i].z, vpre[i].w};
            unsigned hv[8];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const __half2_raw lo_raw =
                    __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[j] & 0xffffu), __NV_E4M3);
                const __half2_raw hi_raw =
                    __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(words[j] >> 16), __NV_E4M3);
                const float2 lo = __half22float2(*reinterpret_cast<const __half2*>(&lo_raw));
                const float2 hi = __half22float2(*reinterpret_cast<const __half2*>(&hi_raw));
                hv[2 * j] = half2_bits(lo.x * vspre, lo.y * vspre);
                hv[2 * j + 1] = half2_bits(hi.x * vspre, hi.y * vspre);
            }
            uint4* vdst = reinterpret_cast<uint4*>(&vh[spos][c * 16]);
            vdst[0] = make_uint4(hv[0], hv[1], hv[2], hv[3]);
            vdst[1] = make_uint4(hv[4], hv[5], hv[6], hv[7]);
        }
        if (schunk == 0) ks_sh[spos] = kspre;
    };

    const int tile_stride = SPLITS * kTileP;
    const int tile0 = start + split * kTileP;
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
            const uint8_t* ka = &kh[mt * 16 + gid][tg * 4];
            const uint8_t* kb = &kh[mt * 16 + gid + 8][tg * 4];
            #pragma unroll
            for (int kc = 0; kc < kKSteps; ++kc) {
                const unsigned wa = *reinterpret_cast<const unsigned*>(ka + kc * 16);
                const unsigned wb = *reinterpret_cast<const unsigned*>(kb + kc * 16);
                mma_m16n8k16_f16(
                    c,
                    fp8x2_to_half2_bits(wa & 0xffffu),
                    fp8x2_to_half2_bits(wb & 0xffffu),
                    fp8x2_to_half2_bits(wa >> 16),
                    fp8x2_to_half2_bits(wb >> 16),
                    qb0[kc],
                    qb1[kc]);
            }
            const int p0 = base + mt * 16 + gid;
            const int p1 = p0 + 8;
            const float ks0 = ks_sh[mt * 16 + gid];
            const float ks1 = ks_sh[mt * 16 + gid + 8];
            lv[0] = rm.live(p0, 0, hole_lo, hole_hi);
            lv[1] = rm.live(p0, 1, hole_lo, hole_hi);
            lv[2] = rm.live(p1, 0, hole_lo, hole_hi);
            lv[3] = rm.live(p1, 1, hole_lo, hole_hi);
            s[0] = lv[0] ? c[0] * ks0 * scaling : -INFINITY;
            s[1] = lv[1] ? c[1] * ks0 * scaling : -INFINITY;
            s[2] = lv[2] ? c[2] * ks1 * scaling : -INFINITY;
            s[3] = lv[3] ? c[3] * ks1 * scaling : -INFINITY;
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

        float crn[NCH][2];
        #pragma unroll
        for (int n = 0; n < NCH; ++n) {
            crn[n][0] = corr_sh[n * 8 + tg * 2];
            crn[n][1] = corr_sh[n * 8 + tg * 2 + 1];
        }
        #pragma unroll
        for (int a = 0; a < kA; ++a) {
            #pragma unroll
            for (int n = 0; n < NCH; ++n) {
                o[a][n][0] *= crn[n][0];
                o[a][n][1] *= crn[n][1];
                o[a][n][2] *= crn[n][0];
                o[a][n][3] *= crn[n][1];
            }
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
            #pragma unroll
            for (int a = 0; a < kA; ++a) {
                const int hd0 = (warp * kA + a) * 16;
                const int prow = ks * 16 + lr + ((lm & 2) ? 8 : 0);
                const int hcol = hd0 + ((lm & 1) ? 8 : 0);
                unsigned va[4];
                ldmatrix_x4_trans(va, &vh[prow][hcol]);
                #pragma unroll
                for (int n = 0; n < NCH; ++n)
                    mma_m16n8k16_f16(o[a][n], va[0], va[1], va[2], va[3], pb0[n], pb1[n]);
            }
        }
    }

    fused_tc_store_and_combine<NCH, kA, Geo<NCH>::kThreadsT>(
        o, m_r, l_r, qk && mt == 0 && gid == 0, r0, rows, warp, gid, tg, scratch, fan_in,
        outp, kvh, split, SPLITS, NH);

    NVK_PDL_EPILOG();
}

template <int NCH>
int launch(
    dim3 grid,
    cudaStream_t cs,
    const __nv_bfloat16* qb,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    float* scratch,
    unsigned int* fan_in,
    __nv_bfloat16* ob,
    const int* n_total_dev,
    int NKV,
    int sp,
    int WINDOW,
    int RING,
    float scaling,
    const int* block_table,
    int block_size,
    int delta,
    int M,
    int hole_from_row,
    int hole_lo_delta,
    int hole_hi_delta
) {
    auto kern = flash_decode_fused_tc_kernel<NCH>;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(Geo<NCH>::kThreadsT), 0, cs);
        cudaLaunchKernelEx(
            &cfg, kern, qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
            n_total_dev, NKV, sp, WINDOW, RING, scaling, block_table, block_size,
            delta, M, hole_from_row, hole_lo_delta, hole_hi_delta);
    } else {
        kern<<<grid, Geo<NCH>::kThreadsT, 0, cs>>>(
            qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob, n_total_dev,
            NKV, sp, WINDOW, RING, scaling, block_table, block_size, delta, M,
            hole_from_row, hole_lo_delta, hole_hi_delta);
    }
    return (int)cudaGetLastError();
}

int dispatch(
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
    int splits,
    int splits_cap_scratch_rows_over_m,
    float scaling,
    const int* block_table,
    int block_size,
    int delta,
    int M,
    int hole_from_row,
    int hole_lo_delta,
    int hole_hi_delta
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD != kHD) return -21;
    if ((NH % NKV) != 0 || NH / NKV != kGroup)
        return kFusedTcArmIsShapedForTheQwen38Group6ShapeRcMinus29;
    if (M < 1 || M > kMaxRows) return -26;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    if (hole_lo_delta < hole_hi_delta || hole_hi_delta < 0) return -25;
    int sp = (splits > 0) ? splits : nv_kernels_flash_decode_kvshare_splits_pick(NKV);
    if (splits_cap_scratch_rows_over_m > 0 && sp > splits_cap_scratch_rows_over_m)
        sp = splits_cap_scratch_rows_over_m;
    if (sp < 1 || sp > kMaxSplits) return -23;
    {
        const int sp_cap = kCombineScaleFloats2048HoldRowsTimesSplitsSoEveryRowCombinesInParallel / (M * kGroup);
        if (sp > sp_cap) sp = sp_cap;
    }
    dim3 grid((unsigned)NKV, (unsigned)sp);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
    const int nch = (M * kGroup + 7) / 8;
#define NVK_FUSED_TC_LAUNCH(N)                                                          \
    case N:                                                                             \
        return launch<N>(                                                               \
            grid, cs, qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,  \
            n_total_dev, NKV, sp, WINDOW, RING, scaling, block_table, block_size,       \
            delta, M, hole_from_row, hole_lo_delta, hole_hi_delta);
    switch (nch) {
        NVK_FUSED_TC_LAUNCH(1)
        NVK_FUSED_TC_LAUNCH(2)
        NVK_FUSED_TC_LAUNCH(3)
        NVK_FUSED_TC_LAUNCH(4)
        NVK_FUSED_TC_LAUNCH(5)
        NVK_FUSED_TC_LAUNCH(6)
        default: return -26;
    }
#undef NVK_FUSED_TC_LAUNCH
}

}

extern "C" int nv_kernels_flash_decode_fused_tc_fp8kv_mk_paged(
    void* stream, const uint16_t* q, const uint8_t* k_fp8, const uint8_t* v_fp8,
    const float* k_scales, const float* v_scales, uint16_t* out, const int* n_total_dev,
    float* scratch, unsigned int* fan_in, int NH, int NKV, int HD, int WINDOW, int RING,
    int splits, int splits_cap_scratch_rows_over_m, float scaling, const int* block_table,
    int block_size, int delta, int M, int hole_from_row, int hole_lo_delta, int hole_hi_delta
) {
    return fusedtc::dispatch(
        stream, q, k_fp8, v_fp8, k_scales, v_scales, out, n_total_dev, scratch, fan_in, NH,
        NKV, HD, WINDOW, RING, splits, splits_cap_scratch_rows_over_m, scaling, block_table,
        block_size, delta, M, hole_from_row, hole_lo_delta, hole_hi_delta);
}
