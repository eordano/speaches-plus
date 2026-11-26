
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdlib.h>
#include "nvk_grid.cuh"
#include "nvk_pdl.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerCta = 8;
constexpr int kBlockDim = kWarpSize * kRowsPerCta;

__device__ __forceinline__ void nibbles8_to_fp8_bytes_via_prmt_because_sm120_fp4_cvt_is_emulated(
    unsigned w,
    unsigned& r_lo,
    unsigned& r_hi
) {
    constexpr unsigned kE2m1MagAsE4m3Bytes0to3 = 0x3C383000u;
    constexpr unsigned kE2m1MagAsE4m3Bytes4to7 = 0x4C484440u;
    constexpr unsigned kSignByteTable = 0x00008000u;
    unsigned mag_sel = w & 0x77777777u;
    unsigned sgn_sel = (w >> 3) & 0x11111111u;
    r_lo = __byte_perm(kE2m1MagAsE4m3Bytes0to3, kE2m1MagAsE4m3Bytes4to7, mag_sel)
         | __byte_perm(kSignByteTable, 0u, sgn_sel);
    r_hi = __byte_perm(kE2m1MagAsE4m3Bytes0to3, kE2m1MagAsE4m3Bytes4to7, mag_sel >> 16)
         | __byte_perm(kSignByteTable, 0u, sgn_sel >> 16);
}

#define NV_UE4M3_SUBNORMAL_STEP  0.001953125f

__device__ __forceinline__ float decode_ue4m3_dev(uint8_t b) {
    unsigned biased = (unsigned)(b >> 3) & 0x0Fu;
    unsigned mant = (unsigned)(b & 0x07u);
    float norm = __uint_as_float(((biased + 120u) << 23) | (mant << 20));
    return biased ? norm : (float)mant * NV_UE4M3_SUBNORMAL_STEP;
}

__device__ __forceinline__ int swizzled_scale_dst(int m, int kb, int k_blocks) {
    int k_tiles = (k_blocks + 3) / 4;
    int m_tile = m / 128;
    int d2 = (m / 32) & 3;
    int d3 = m & 31;
    int k_tile = kb / 4;
    int d5 = kb & 3;
    return ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
}

bool fp4_dp4a_enabled() {
    const char* value = getenv("NV_MOE_FP4_DP4A");
    return value && value[0] == '1' && value[1] == '\0';
}

__device__ __forceinline__ void nibbles8_to_doubled_e2m1_int8_lanes(
    unsigned w, unsigned& lo, unsigned& hi
) {
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

template <bool DP4A>
__device__ __forceinline__ float dot16_e2m1(unsigned long long wq,
                                            unsigned long long aq) {
    if constexpr (DP4A) {
        unsigned wi[4], ai[4];
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)wq, wi[0], wi[1]);
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(wq >> 32), wi[2], wi[3]);
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)aq, ai[0], ai[1]);
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(aq >> 32), ai[2], ai[3]);
        int d = __dp4a((int)wi[0], (int)ai[0], 0);
        d = __dp4a((int)wi[1], (int)ai[1], d);
        d = __dp4a((int)wi[2], (int)ai[2], d);
        d = __dp4a((int)wi[3], (int)ai[3], d);
        return (float)d * 0.25f;
    }
    unsigned we[4];
    unsigned ae[4];
    nibbles8_to_fp8_bytes_via_prmt_because_sm120_fp4_cvt_is_emulated(
        (unsigned)(wq & 0xFFFFFFFFull), we[0], we[1]);
    nibbles8_to_fp8_bytes_via_prmt_because_sm120_fp4_cvt_is_emulated(
        (unsigned)(wq >> 32), we[2], we[3]);
    nibbles8_to_fp8_bytes_via_prmt_because_sm120_fp4_cvt_is_emulated(
        (unsigned)(aq & 0xFFFFFFFFull), ae[0], ae[1]);
    nibbles8_to_fp8_bytes_via_prmt_because_sm120_fp4_cvt_is_emulated(
        (unsigned)(aq >> 32), ae[2], ae[3]);
    const __nv_fp8x2_storage_t* wp =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(we);
    const __nv_fp8x2_storage_t* ap =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(ae);
    float d = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        __half2_raw wh = __nv_cvt_fp8x2_to_halfraw2(wp[i], __NV_E4M3);
        __half2_raw ah = __nv_cvt_fp8x2_to_halfraw2(ap[i], __NV_E4M3);
        float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&wh));
        float2 af = __half22float2(*reinterpret_cast<const __half2*>(&ah));
        d += wf.x * af.x + wf.y * af.y;
    }
    return d;
}

__device__ __forceinline__ uint8_t gvq_encode_e2m1(float x) {
    uint8_t sign = signbit(x) ? 0x08u : 0u;
    float a = fabsf(x);
    uint8_t c = (uint8_t)(a > 0.25f) + (uint8_t)(a > 0.75f) + (uint8_t)(a > 1.25f)
              + (uint8_t)(a > 1.75f) + (uint8_t)(a > 2.5f)  + (uint8_t)(a > 3.5f)
              + (uint8_t)(a > 5.0f);
    return sign | c;
}

#define NV_UE4M3_MIN_NORMAL 0.015625f

__device__ __forceinline__ uint8_t gvq_encode_ue4m3(float scale) {
    if (!isfinite(scale) || scale <= 0.f) return 0;
    float clamped = fminf(scale, 448.f);
    uint8_t b = (uint8_t)__nv_cvt_float_to_fp8(clamped, __NV_SATFINITE, __NV_E4M3);
    return (b == 0x7Fu) ? 0x7Eu : (b & 0x7Fu);
}

constexpr int kM1RowsPerWarpSharingTheActivationLoads = 4;
constexpr int kMaxDedupTokens = 4;
constexpr int kMaxDedupGroups = 80;

__device__ __forceinline__ int moe_collect_dedup_peers(
    const int32_t* __restrict__ ids,
    int num_groups,
    int g,
    int e,
    int* peer_g
) {
    int earlier = 0;
    for (int j = 0; j < g; ++j) {
        if (__ldg(&ids[j]) == e) ++earlier;
    }
    if ((earlier & (kMaxDedupTokens - 1)) != 0) return 0;
    int cnt = 0;
    peer_g[cnt++] = g;
    for (int j = g + 1; j < num_groups && cnt < kMaxDedupTokens; ++j) {
        if (__ldg(&ids[j]) == e) peer_g[cnt++] = j;
    }
    return cnt;
}

__global__ void moe_grouped_fp4_gemv_m1_bf16_subwarp8_dedup_dp4a_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ b_packed,
    const uint8_t* __restrict__ b_scales,
    const float* __restrict__ alphas,
    __nv_bfloat16* __restrict__ d,
    const int32_t* __restrict__ ids,
    int num_groups,
    int num_experts_total,
    int n_dim,
    int k_dim,
    int a_tile_stride_rows,
    long long d_group_stride_elems
) {
    extern __shared__ uint8_t smem_raw[];
    __shared__ int s_peers[kMaxDedupTokens];
    __shared__ int s_cnt;

    NVK_PDL_PROLOG();

    int tid = threadIdx.x;
    int g = blockIdx.y;
    int e = __ldg(&ids[g]);
    int n0 = (blockIdx.x * 32 + (tid >> 3)) * 2;

    if (e < 0 || e >= num_experts_total) {
        int sub = tid & 7;
        if (sub < 2 && n0 + sub < n_dim) {
            d[(long long)g * d_group_stride_elems + n0 + sub] = __float2bfloat16(0.f);
        }
        NVK_PDL_EPILOG();
        return;
    }

    if (tid == 0) {
        s_cnt = moe_collect_dedup_peers(ids, num_groups, g, e, s_peers);
    }
    __syncthreads();
    int cnt = s_cnt;
    if (cnt == 0) {
        NVK_PDL_EPILOG();
        return;
    }

    int half_k = k_dim >> 1;
    int group_k = k_dim >> 4;
    int k_tiles = (group_k + 3) >> 2;
    int group_k_pad = k_tiles * 4;
    long long n_pad = (((long long)n_dim + 127) / 128) * 128;

    int* a_i8 = reinterpret_cast<int*>(smem_raw);
    float* a_sc = reinterpret_cast<float*>(a_i8 + kMaxDedupTokens * group_k * 4);
    int total_kb = cnt * group_k;
    for (int idx = tid; idx < total_kb; idx += blockDim.x) {
        int c = idx / group_k;
        int kb = idx - c * group_k;
        int pg = s_peers[c];
        const unsigned long long* a_row = reinterpret_cast<const unsigned long long*>(
            a_packed + (size_t)pg * (size_t)a_tile_stride_rows * (size_t)half_k);
        const uint8_t* a_sf = a_scales +
            (size_t)pg * (size_t)(a_tile_stride_rows / 128) * (size_t)group_k_pad * 128;
        unsigned long long aq = __ldg(&a_row[kb]);
        unsigned ai0, ai1, ai2, ai3;
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)aq, ai0, ai1);
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(aq >> 32), ai2, ai3);
        int* dst = a_i8 + idx * 4;
        dst[0] = (int)ai0;
        dst[1] = (int)ai1;
        dst[2] = (int)ai2;
        dst[3] = (int)ai3;
        a_sc[idx] = decode_ue4m3_dev(__ldg(&a_sf[(kb >> 2) * 512 + (kb & 3)])) * 0.25f;
    }
    __syncthreads();

    if (n0 >= n_dim) {
        NVK_PDL_EPILOG();
        return;
    }
    int sub = tid & 7;
    bool has1 = (n0 + 1) < n_dim;
    const unsigned long long* w0_row = reinterpret_cast<const unsigned long long*>(
        b_packed + ((size_t)e * (size_t)n_dim + (size_t)n0) * (size_t)half_k);
    const unsigned long long* w1_row = has1 ? reinterpret_cast<const unsigned long long*>(
        b_packed + ((size_t)e * (size_t)n_dim + (size_t)(n0 + 1)) * (size_t)half_k) : w0_row;
    const uint8_t* w_sf = b_scales + (size_t)e * (size_t)n_pad * (size_t)group_k_pad;

    if (cnt == 1) {
        float acc0 = 0.f;
        float acc1 = 0.f;
        int kb = sub;
        unsigned long long rw0 = (kb < group_k) ? __ldcs(&w0_row[kb]) : 0ull;
        unsigned long long rw1 = (has1 && kb < group_k) ? __ldcs(&w1_row[kb]) : 0ull;
        uint8_t rs0 = (kb < group_k) ? __ldcs(&w_sf[swizzled_scale_dst(n0, kb, group_k)]) : 0u;
        uint8_t rs1 = (has1 && kb < group_k) ? __ldcs(&w_sf[swizzled_scale_dst(n0 + 1, kb, group_k)]) : 0u;
        while (kb < group_k) {
            int kbn = kb + 8;
            unsigned long long nw0 = (kbn < group_k) ? __ldcs(&w0_row[kbn]) : 0ull;
            unsigned long long nw1 = (has1 && kbn < group_k) ? __ldcs(&w1_row[kbn]) : 0ull;
            uint8_t ns0 = (kbn < group_k) ? __ldcs(&w_sf[swizzled_scale_dst(n0, kbn, group_k)]) : 0u;
            uint8_t ns1 = (has1 && kbn < group_k) ? __ldcs(&w_sf[swizzled_scale_dst(n0 + 1, kbn, group_k)]) : 0u;
            const int* ap = a_i8 + kb * 4;
            int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
            float asc = a_sc[kb];
            unsigned wi0, wi1, wi2, wi3;
            nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rw0, wi0, wi1);
            nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rw0 >> 32), wi2, wi3);
            int d0 = __dp4a((int)wi0, a0, 0);
            d0 = __dp4a((int)wi1, a1, d0);
            d0 = __dp4a((int)wi2, a2, d0);
            d0 = __dp4a((int)wi3, a3, d0);
            acc0 += (decode_ue4m3_dev(rs0) * asc) * (float)d0;
            if (has1) {
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rw1, wi0, wi1);
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rw1 >> 32), wi2, wi3);
                int d1 = __dp4a((int)wi0, a0, 0);
                d1 = __dp4a((int)wi1, a1, d1);
                d1 = __dp4a((int)wi2, a2, d1);
                d1 = __dp4a((int)wi3, a3, d1);
                acc1 += (decode_ue4m3_dev(rs1) * asc) * (float)d1;
            }
            kb = kbn;
            rw0 = nw0; rw1 = nw1; rs0 = ns0; rs1 = ns1;
        }
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1) {
            acc0 += __shfl_xor_sync(0xffffffffu, acc0, o, 8);
            acc1 += __shfl_xor_sync(0xffffffffu, acc1, o, 8);
        }
        if (sub == 0) {
            float al = __ldg(&alphas[e]);
            long long db = (long long)g * d_group_stride_elems;
            d[db + n0] = __float2bfloat16(acc0 * al);
            if (has1) d[db + n0 + 1] = __float2bfloat16(acc1 * al);
        }
    } else {
        float acc0[kMaxDedupTokens];
        float acc1[kMaxDedupTokens];
        #pragma unroll
        for (int c = 0; c < kMaxDedupTokens; ++c) {
            acc0[c] = 0.f;
            acc1[c] = 0.f;
        }
        int kb = sub;
        unsigned long long rw0 = (kb < group_k) ? __ldcs(&w0_row[kb]) : 0ull;
        unsigned long long rw1 = (has1 && kb < group_k) ? __ldcs(&w1_row[kb]) : 0ull;
        uint8_t rs0 = (kb < group_k) ? __ldcs(&w_sf[swizzled_scale_dst(n0, kb, group_k)]) : 0u;
        uint8_t rs1 = (has1 && kb < group_k) ? __ldcs(&w_sf[swizzled_scale_dst(n0 + 1, kb, group_k)]) : 0u;
        while (kb < group_k) {
            int kbn = kb + 8;
            unsigned long long nw0 = (kbn < group_k) ? __ldcs(&w0_row[kbn]) : 0ull;
            unsigned long long nw1 = (has1 && kbn < group_k) ? __ldcs(&w1_row[kbn]) : 0ull;
            uint8_t ns0 = (kbn < group_k) ? __ldcs(&w_sf[swizzled_scale_dst(n0, kbn, group_k)]) : 0u;
            uint8_t ns1 = (has1 && kbn < group_k) ? __ldcs(&w_sf[swizzled_scale_dst(n0 + 1, kbn, group_k)]) : 0u;
            unsigned w0i[4], w1i[4];
            nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rw0, w0i[0], w0i[1]);
            nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rw0 >> 32), w0i[2], w0i[3]);
            float ws0 = decode_ue4m3_dev(rs0);
            float ws1 = 0.f;
            if (has1) {
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rw1, w1i[0], w1i[1]);
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rw1 >> 32), w1i[2], w1i[3]);
                ws1 = decode_ue4m3_dev(rs1);
            }
            for (int c = 0; c < cnt; ++c) {
                int idx = c * group_k + kb;
                const int* ap = a_i8 + idx * 4;
                int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
                float asc = a_sc[idx];
                int d0 = __dp4a((int)w0i[0], a0, 0);
                d0 = __dp4a((int)w0i[1], a1, d0);
                d0 = __dp4a((int)w0i[2], a2, d0);
                d0 = __dp4a((int)w0i[3], a3, d0);
                acc0[c] += (ws0 * asc) * (float)d0;
                if (has1) {
                    int d1 = __dp4a((int)w1i[0], a0, 0);
                    d1 = __dp4a((int)w1i[1], a1, d1);
                    d1 = __dp4a((int)w1i[2], a2, d1);
                    d1 = __dp4a((int)w1i[3], a3, d1);
                    acc1[c] += (ws1 * asc) * (float)d1;
                }
            }
            kb = kbn;
            rw0 = nw0; rw1 = nw1; rs0 = ns0; rs1 = ns1;
        }
        float al = __ldg(&alphas[e]);
        for (int c = 0; c < cnt; ++c) {
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) {
                acc0[c] += __shfl_xor_sync(0xffffffffu, acc0[c], o, 8);
                acc1[c] += __shfl_xor_sync(0xffffffffu, acc1[c], o, 8);
            }
            if (sub == 0) {
                long long db = (long long)s_peers[c] * d_group_stride_elems;
                d[db + n0] = __float2bfloat16(acc0[c] * al);
                if (has1) d[db + n0 + 1] = __float2bfloat16(acc1[c] * al);
            }
        }
    }
    NVK_PDL_EPILOG();
}

__global__ void moe_grouped_fp4_gemv_gate_up_silu_quant_dedup_dp4a_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ gate_w,
    const uint8_t* __restrict__ gate_sf,
    const float* __restrict__ gate_alphas,
    const uint8_t* __restrict__ up_w,
    const uint8_t* __restrict__ up_sf,
    const float* __restrict__ up_alphas,
    const float* __restrict__ dn_mini,
    const int32_t* __restrict__ ids,
    uint8_t* __restrict__ y_act_fp4,
    uint8_t* __restrict__ y_act_sf,
    int num_groups,
    int num_experts_total,
    int n_dim,
    int k_dim,
    int a_tile_stride_rows
) {
    extern __shared__ uint8_t smem_raw[];
    __shared__ __nv_bfloat16 s_gate[kMaxDedupTokens][16];
    __shared__ __nv_bfloat16 s_up[kMaxDedupTokens][16];
    __shared__ int s_peers[kMaxDedupTokens];
    __shared__ int s_cnt;

    NVK_PDL_PROLOG();

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int g = blockIdx.y;
    int e = __ldg(&ids[g]);
    int n = blockIdx.x * 16 + warp;

    if (e < 0 || e >= num_experts_total) {
        if (lane == 0 && warp < 16) {
            s_gate[0][warp] = __float2bfloat16(0.f);
            s_up[0][warp] = __float2bfloat16(0.f);
        }
        __syncthreads();
        if (tid < 16) {
            int row = g * a_tile_stride_rows;
            int kb = blockIdx.x;
            if ((tid & 1) == 0) {
                uint8_t* pblock = y_act_fp4 + (size_t)row * (size_t)(n_dim / 2) + (size_t)kb * 8;
                pblock[tid >> 1] = 0u;
            }
            if (tid == 0) {
                y_act_sf[swizzled_scale_dst(row, kb, n_dim / 16)] = gvq_encode_ue4m3(1.f);
            }
        }
        NVK_PDL_EPILOG();
        return;
    }

    if (tid == 0) {
        s_cnt = moe_collect_dedup_peers(ids, num_groups, g, e, s_peers);
    }
    __syncthreads();
    int cnt = s_cnt;
    if (cnt == 0) {
        NVK_PDL_EPILOG();
        return;
    }

    int half_k = k_dim >> 1;
    int group_k = k_dim >> 4;
    int k_tiles = (group_k + 3) >> 2;
    int group_k_pad = k_tiles * 4;
    long long n_pad = (((long long)n_dim + 127) / 128) * 128;

    int* a_i8 = reinterpret_cast<int*>(smem_raw);
    float* a_sc = reinterpret_cast<float*>(a_i8 + kMaxDedupTokens * group_k * 4);
    int total_kb = cnt * group_k;
    for (int idx = tid; idx < total_kb; idx += blockDim.x) {
        int c = idx / group_k;
        int kb = idx - c * group_k;
        int pg = s_peers[c];
        const unsigned long long* a_row = reinterpret_cast<const unsigned long long*>(
            a_packed + (size_t)pg * (size_t)a_tile_stride_rows * (size_t)half_k);
        const uint8_t* a_sf = a_scales +
            (size_t)pg * (size_t)(a_tile_stride_rows / 128) * (size_t)group_k_pad * 128;
        unsigned long long aq = __ldg(&a_row[kb]);
        unsigned ai0, ai1, ai2, ai3;
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)aq, ai0, ai1);
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(aq >> 32), ai2, ai3);
        int* dst = a_i8 + idx * 4;
        dst[0] = (int)ai0;
        dst[1] = (int)ai1;
        dst[2] = (int)ai2;
        dst[3] = (int)ai3;
        a_sc[idx] = decode_ue4m3_dev(__ldg(&a_sf[(kb >> 2) * 512 + (kb & 3)])) * 0.25f;
    }
    __syncthreads();

    if (n < n_dim) {
        const unsigned long long* wg_row = reinterpret_cast<const unsigned long long*>(
            gate_w + ((size_t)e * (size_t)n_dim + (size_t)n) * (size_t)half_k);
        const unsigned long long* wu_row = reinterpret_cast<const unsigned long long*>(
            up_w + ((size_t)e * (size_t)n_dim + (size_t)n) * (size_t)half_k);
        const uint8_t* wg_sf = gate_sf + (size_t)e * (size_t)n_pad * (size_t)group_k_pad;
        const uint8_t* wu_sf = up_sf + (size_t)e * (size_t)n_pad * (size_t)group_k_pad;

        if (cnt == 1) {
            float acc_g = 0.f;
            float acc_u = 0.f;
            int kb = lane;
            unsigned long long rg = (kb < group_k) ? __ldcs(&wg_row[kb]) : 0ull;
            unsigned long long ru = (kb < group_k) ? __ldcs(&wu_row[kb]) : 0ull;
            uint8_t sg = (kb < group_k) ? __ldcs(&wg_sf[swizzled_scale_dst(n, kb, group_k)]) : 0u;
            uint8_t su = (kb < group_k) ? __ldcs(&wu_sf[swizzled_scale_dst(n, kb, group_k)]) : 0u;
            while (kb < group_k) {
                int kbn = kb + 32;
                unsigned long long ng = (kbn < group_k) ? __ldcs(&wg_row[kbn]) : 0ull;
                unsigned long long nu = (kbn < group_k) ? __ldcs(&wu_row[kbn]) : 0ull;
                uint8_t nsg = (kbn < group_k) ? __ldcs(&wg_sf[swizzled_scale_dst(n, kbn, group_k)]) : 0u;
                uint8_t nsu = (kbn < group_k) ? __ldcs(&wu_sf[swizzled_scale_dst(n, kbn, group_k)]) : 0u;
                const int* ap = a_i8 + kb * 4;
                int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
                float asc = a_sc[kb];
                unsigned wi0, wi1, wi2, wi3;
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rg, wi0, wi1);
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rg >> 32), wi2, wi3);
                int dg = __dp4a((int)wi0, a0, 0);
                dg = __dp4a((int)wi1, a1, dg);
                dg = __dp4a((int)wi2, a2, dg);
                dg = __dp4a((int)wi3, a3, dg);
                acc_g += (decode_ue4m3_dev(sg) * asc) * (float)dg;

                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)ru, wi0, wi1);
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(ru >> 32), wi2, wi3);
                int du = __dp4a((int)wi0, a0, 0);
                du = __dp4a((int)wi1, a1, du);
                du = __dp4a((int)wi2, a2, du);
                du = __dp4a((int)wi3, a3, du);
                acc_u += (decode_ue4m3_dev(su) * asc) * (float)du;
                kb = kbn;
                rg = ng; ru = nu; sg = nsg; su = nsu;
            }
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                acc_g += __shfl_xor_sync(0xffffffffu, acc_g, offset);
                acc_u += __shfl_xor_sync(0xffffffffu, acc_u, offset);
            }
            if (lane == 0) {
                s_gate[0][warp] = __float2bfloat16(acc_g * __ldg(&gate_alphas[e]));
                s_up[0][warp] = __float2bfloat16(acc_u * __ldg(&up_alphas[e]));
            }
        } else {
            float acc_g[kMaxDedupTokens];
            float acc_u[kMaxDedupTokens];
            #pragma unroll
            for (int c = 0; c < kMaxDedupTokens; ++c) {
                acc_g[c] = 0.f;
                acc_u[c] = 0.f;
            }
            int kb = lane;
            unsigned long long rg = (kb < group_k) ? __ldcs(&wg_row[kb]) : 0ull;
            unsigned long long ru = (kb < group_k) ? __ldcs(&wu_row[kb]) : 0ull;
            uint8_t sg = (kb < group_k) ? __ldcs(&wg_sf[swizzled_scale_dst(n, kb, group_k)]) : 0u;
            uint8_t su = (kb < group_k) ? __ldcs(&wu_sf[swizzled_scale_dst(n, kb, group_k)]) : 0u;
            while (kb < group_k) {
                int kbn = kb + 32;
                unsigned long long ng = (kbn < group_k) ? __ldcs(&wg_row[kbn]) : 0ull;
                unsigned long long nu = (kbn < group_k) ? __ldcs(&wu_row[kbn]) : 0ull;
                uint8_t nsg = (kbn < group_k) ? __ldcs(&wg_sf[swizzled_scale_dst(n, kbn, group_k)]) : 0u;
                uint8_t nsu = (kbn < group_k) ? __ldcs(&wu_sf[swizzled_scale_dst(n, kbn, group_k)]) : 0u;
                unsigned wgi[4], wui[4];
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rg, wgi[0], wgi[1]);
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rg >> 32), wgi[2], wgi[3]);
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)ru, wui[0], wui[1]);
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(ru >> 32), wui[2], wui[3]);
                float wgs = decode_ue4m3_dev(sg);
                float wus = decode_ue4m3_dev(su);
                for (int c = 0; c < cnt; ++c) {
                    int idx = c * group_k + kb;
                    const int* ap = a_i8 + idx * 4;
                    int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
                    float asc = a_sc[idx];
                    int dg = __dp4a((int)wgi[0], a0, 0);
                    dg = __dp4a((int)wgi[1], a1, dg);
                    dg = __dp4a((int)wgi[2], a2, dg);
                    dg = __dp4a((int)wgi[3], a3, dg);
                    acc_g[c] += (wgs * asc) * (float)dg;

                    int du = __dp4a((int)wui[0], a0, 0);
                    du = __dp4a((int)wui[1], a1, du);
                    du = __dp4a((int)wui[2], a2, du);
                    du = __dp4a((int)wui[3], a3, du);
                    acc_u[c] += (wus * asc) * (float)du;
                }
                kb = kbn;
                rg = ng; ru = nu; sg = nsg; su = nsu;
            }
            float ga = __ldg(&gate_alphas[e]);
            float ua = __ldg(&up_alphas[e]);
            for (int c = 0; c < cnt; ++c) {
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    acc_g[c] += __shfl_xor_sync(0xffffffffu, acc_g[c], offset);
                    acc_u[c] += __shfl_xor_sync(0xffffffffu, acc_u[c], offset);
                }
                if (lane == 0) {
                    s_gate[c][warp] = __float2bfloat16(acc_g[c] * ga);
                    s_up[c][warp] = __float2bfloat16(acc_u[c] * ua);
                }
            }
        }
    } else if (lane == 0 && warp < 16) {
        for (int c = 0; c < cnt; ++c) {
            s_gate[c][warp] = __float2bfloat16(0.f);
            s_up[c][warp] = __float2bfloat16(0.f);
        }
    }
    __syncthreads();

    if (warp < cnt && lane < 16) {
        int c = warp;
        int i = lane;
        int pg = s_peers[c];
        float gv = __bfloat162float(s_gate[c][i]);
        float uv = __bfloat162float(s_up[c][i]);
        float silu_g = gv / (1.f + expf(-gv));
        float v = silu_g * uv;
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0x0000ffffu, amax, o, 16));
        }
        float stored = __ldg(&dn_mini[pg]);
        if (stored == 0.f || !isfinite(stored)) stored = 1.f;
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        uint8_t scale_byte = gvq_encode_ue4m3(stored * local_scale);
        float scale_decoded = decode_ue4m3_dev(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);
        uint8_t nib = gvq_encode_e2m1(fminf(fmaxf(v * inv, -6.f), 6.f));
        unsigned hi = __shfl_down_sync(0x0000ffffu, (unsigned)nib, 1, 16);
        int row = pg * a_tile_stride_rows;
        int kb = blockIdx.x;
        if ((i & 1) == 0) {
            uint8_t* pblock = y_act_fp4 + (size_t)row * (size_t)(n_dim / 2) + (size_t)kb * 8;
            pblock[i >> 1] = (uint8_t)((hi << 4) | (nib & 0x0Fu));
        }
        if (i == 0) {
            int blocks_per_row = n_dim / 16;
            y_act_sf[swizzled_scale_dst(row, kb, blocks_per_row)] = scale_byte;
        }
    }
    NVK_PDL_EPILOG();
}

static bool moe_dedup_disabled() {
    const char* v = getenv("NV_MOE_FP4_NO_DEDUP");
    return v && v[0] == '1' && v[1] == '\0';
}

__global__ void moe_unswizzle_scales_kernel(
    const uint8_t* __restrict__ swizzled,
    uint8_t* __restrict__ linear,
    int num_experts,
    int n_dim,
    int group_k
) {
    int k_tiles = (group_k + 3) >> 2;
    int group_k_pad = k_tiles * 4;
    size_t n_pad = (size_t)((n_dim + 127) / 128) * 128;
    size_t swz_stride = n_pad * (size_t)group_k_pad;
    size_t lin_per_e = (size_t)n_dim * (size_t)group_k;
    size_t total = (size_t)num_experts * lin_per_e;
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t step = (size_t)gridDim.x * blockDim.x;
    for (; idx < total; idx += step) {
        int e = (int)(idx / lin_per_e);
        size_t rem = idx - (size_t)e * lin_per_e;
        int n = (int)(rem / (size_t)group_k);
        int kb = (int)(rem - (size_t)n * (size_t)group_k);
        linear[idx] = __ldg(&swizzled[(size_t)e * swz_stride + (size_t)swizzled_scale_dst(n, kb, group_k)]);
    }
}

__global__ void __launch_bounds__(256) moe_grouped_fp4_gemv_m1_bf16_subwarp8_dedup_dp4a_lin_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ b_packed,
    const uint8_t* __restrict__ b_scales_lin,
    const float* __restrict__ alphas,
    __nv_bfloat16* __restrict__ d,
    const int32_t* __restrict__ ids,
    int num_groups,
    int num_experts_total,
    int n_dim,
    int k_dim,
    int a_tile_stride_rows,
    long long d_group_stride_elems
) {
    extern __shared__ uint8_t smem_raw[];
    __shared__ int s_peers[kMaxDedupTokens];
    __shared__ int s_cnt;

    NVK_PDL_PROLOG();

    int tid = threadIdx.x;
    int g = blockIdx.y;
    int e = __ldg(&ids[g]);
    int n_base = blockIdx.x * 128 + (tid >> 3) * 2;

    if (e < 0 || e >= num_experts_total) {
        int sub = tid & 7;
        if (sub < 2) {
            if (n_base + sub < n_dim) d[(long long)g * d_group_stride_elems + n_base + sub] = __float2bfloat16(0.f);
            if (n_base + 64 + sub < n_dim) d[(long long)g * d_group_stride_elems + n_base + 64 + sub] = __float2bfloat16(0.f);
        }
        NVK_PDL_EPILOG();
        return;
    }

    if (tid == 0) {
        s_cnt = moe_collect_dedup_peers(ids, num_groups, g, e, s_peers);
    }
    __syncthreads();
    int cnt = s_cnt;
    if (cnt == 0) {
        NVK_PDL_EPILOG();
        return;
    }

    int half_k = k_dim >> 1;
    int group_k = k_dim >> 4;
    int k_tiles = (group_k + 3) >> 2;
    int group_k_pad = k_tiles * 4;

    int* a_i8 = reinterpret_cast<int*>(smem_raw);
    float* a_sc = reinterpret_cast<float*>(a_i8 + kMaxDedupTokens * group_k * 4);
    int total_kb = cnt * group_k;
    for (int idx = tid; idx < total_kb; idx += blockDim.x) {
        int c = idx / group_k;
        int kb = idx - c * group_k;
        int pg = s_peers[c];
        const unsigned long long* a_row = reinterpret_cast<const unsigned long long*>(
            a_packed + (size_t)pg * (size_t)a_tile_stride_rows * (size_t)half_k);
        const uint8_t* a_sf = a_scales +
            (size_t)pg * (size_t)(a_tile_stride_rows / 128) * (size_t)group_k_pad * 128;
        unsigned long long aq = __ldg(&a_row[kb]);
        unsigned ai0, ai1, ai2, ai3;
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)aq, ai0, ai1);
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(aq >> 32), ai2, ai3);
        int* dst = a_i8 + idx * 4;
        dst[0] = (int)ai0;
        dst[1] = (int)ai1;
        dst[2] = (int)ai2;
        dst[3] = (int)ai3;
        a_sc[idx] = decode_ue4m3_dev(__ldg(&a_sf[(kb >> 2) * 512 + (kb & 3)])) * 0.25f;
    }
    __syncthreads();

    int sub = tid & 7;
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
        int n0 = n_base + pair * 64;
        if (n0 >= n_dim) continue;
        bool has1 = (n0 + 1) < n_dim;
        size_t row0 = (size_t)e * (size_t)n_dim + (size_t)n0;
        size_t row1 = has1 ? (row0 + 1) : row0;
        const unsigned long long* w0_row = reinterpret_cast<const unsigned long long*>(b_packed + row0 * (size_t)half_k);
        const unsigned long long* w1_row = reinterpret_cast<const unsigned long long*>(b_packed + row1 * (size_t)half_k);
        const uint8_t* s0_row = b_scales_lin + row0 * (size_t)group_k;
        const uint8_t* s1_row = b_scales_lin + row1 * (size_t)group_k;

        if (cnt == 1) {
            float acc0 = 0.f;
            float acc1 = 0.f;
            int kb = sub;
            unsigned long long rw0 = (kb < group_k) ? __ldcs(&w0_row[kb]) : 0ull;
            unsigned long long rw1 = (has1 && kb < group_k) ? __ldcs(&w1_row[kb]) : 0ull;
            uint8_t rs0 = (kb < group_k) ? __ldcs(&s0_row[kb]) : 0u;
            uint8_t rs1 = (has1 && kb < group_k) ? __ldcs(&s1_row[kb]) : 0u;
            while (kb < group_k) {
                int kbn = kb + 8;
                unsigned long long nw0 = (kbn < group_k) ? __ldcs(&w0_row[kbn]) : 0ull;
                unsigned long long nw1 = (has1 && kbn < group_k) ? __ldcs(&w1_row[kbn]) : 0ull;
                uint8_t ns0 = (kbn < group_k) ? __ldcs(&s0_row[kbn]) : 0u;
                uint8_t ns1 = (has1 && kbn < group_k) ? __ldcs(&s1_row[kbn]) : 0u;
                const int* ap = a_i8 + kb * 4;
                int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
                float asc = a_sc[kb];
                unsigned wi0, wi1, wi2, wi3;
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rw0, wi0, wi1);
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rw0 >> 32), wi2, wi3);
                int d0 = __dp4a((int)wi0, a0, 0);
                d0 = __dp4a((int)wi1, a1, d0);
                d0 = __dp4a((int)wi2, a2, d0);
                d0 = __dp4a((int)wi3, a3, d0);
                acc0 += (decode_ue4m3_dev(rs0) * asc) * (float)d0;
                if (has1) {
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rw1, wi0, wi1);
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rw1 >> 32), wi2, wi3);
                    int d1 = __dp4a((int)wi0, a0, 0);
                    d1 = __dp4a((int)wi1, a1, d1);
                    d1 = __dp4a((int)wi2, a2, d1);
                    d1 = __dp4a((int)wi3, a3, d1);
                    acc1 += (decode_ue4m3_dev(rs1) * asc) * (float)d1;
                }
                kb = kbn;
                rw0 = nw0; rw1 = nw1; rs0 = ns0; rs1 = ns1;
            }
            #pragma unroll
            for (int o = 4; o > 0; o >>= 1) {
                acc0 += __shfl_xor_sync(0xffffffffu, acc0, o, 8);
                acc1 += __shfl_xor_sync(0xffffffffu, acc1, o, 8);
            }
            if (sub == 0) {
                float al = __ldg(&alphas[e]);
                long long db = (long long)g * d_group_stride_elems;
                d[db + n0] = __float2bfloat16(acc0 * al);
                if (has1) d[db + n0 + 1] = __float2bfloat16(acc1 * al);
            }
        } else {
            float acc0[kMaxDedupTokens];
            float acc1[kMaxDedupTokens];
            #pragma unroll
            for (int c = 0; c < kMaxDedupTokens; ++c) {
                acc0[c] = 0.f;
                acc1[c] = 0.f;
            }
            int kb = sub;
            unsigned long long rw0 = (kb < group_k) ? __ldcs(&w0_row[kb]) : 0ull;
            unsigned long long rw1 = (has1 && kb < group_k) ? __ldcs(&w1_row[kb]) : 0ull;
            uint8_t rs0 = (kb < group_k) ? __ldcs(&s0_row[kb]) : 0u;
            uint8_t rs1 = (has1 && kb < group_k) ? __ldcs(&s1_row[kb]) : 0u;
            while (kb < group_k) {
                int kbn = kb + 8;
                unsigned long long nw0 = (kbn < group_k) ? __ldcs(&w0_row[kbn]) : 0ull;
                unsigned long long nw1 = (has1 && kbn < group_k) ? __ldcs(&w1_row[kbn]) : 0ull;
                uint8_t ns0 = (kbn < group_k) ? __ldcs(&s0_row[kbn]) : 0u;
                uint8_t ns1 = (has1 && kbn < group_k) ? __ldcs(&s1_row[kbn]) : 0u;
                unsigned w0i[4], w1i[4];
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rw0, w0i[0], w0i[1]);
                nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rw0 >> 32), w0i[2], w0i[3]);
                float ws0 = decode_ue4m3_dev(rs0);
                float ws1 = 0.f;
                if (has1) {
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rw1, w1i[0], w1i[1]);
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rw1 >> 32), w1i[2], w1i[3]);
                    ws1 = decode_ue4m3_dev(rs1);
                }
                for (int c = 0; c < cnt; ++c) {
                    int idx = c * group_k + kb;
                    const int* ap = a_i8 + idx * 4;
                    int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
                    float asc = a_sc[idx];
                    int d0 = __dp4a((int)w0i[0], a0, 0);
                    d0 = __dp4a((int)w0i[1], a1, d0);
                    d0 = __dp4a((int)w0i[2], a2, d0);
                    d0 = __dp4a((int)w0i[3], a3, d0);
                    acc0[c] += (ws0 * asc) * (float)d0;
                    if (has1) {
                        int d1 = __dp4a((int)w1i[0], a0, 0);
                        d1 = __dp4a((int)w1i[1], a1, d1);
                        d1 = __dp4a((int)w1i[2], a2, d1);
                        d1 = __dp4a((int)w1i[3], a3, d1);
                        acc1[c] += (ws1 * asc) * (float)d1;
                    }
                }
                kb = kbn;
                rw0 = nw0; rw1 = nw1; rs0 = ns0; rs1 = ns1;
            }
            float al = __ldg(&alphas[e]);
            for (int c = 0; c < cnt; ++c) {
                #pragma unroll
                for (int o = 4; o > 0; o >>= 1) {
                    acc0[c] += __shfl_xor_sync(0xffffffffu, acc0[c], o, 8);
                    acc1[c] += __shfl_xor_sync(0xffffffffu, acc1[c], o, 8);
                }
                if (sub == 0) {
                    long long db = (long long)s_peers[c] * d_group_stride_elems;
                    d[db + n0] = __float2bfloat16(acc0[c] * al);
                    if (has1) d[db + n0 + 1] = __float2bfloat16(acc1[c] * al);
                }
            }
        }
    }
    NVK_PDL_EPILOG();
}

__global__ void __launch_bounds__(256) moe_grouped_fp4_gemv_gate_up_silu_quant_dedup_dp4a_lin_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ gate_w,
    const uint8_t* __restrict__ gate_sf_lin,
    const float* __restrict__ gate_alphas,
    const uint8_t* __restrict__ up_w,
    const uint8_t* __restrict__ up_sf_lin,
    const float* __restrict__ up_alphas,
    const float* __restrict__ dn_mini,
    const int32_t* __restrict__ ids,
    uint8_t* __restrict__ y_act_fp4,
    uint8_t* __restrict__ y_act_sf,
    int num_groups,
    int num_experts_total,
    int n_dim,
    int k_dim,
    int a_tile_stride_rows
) {
    extern __shared__ uint8_t smem_raw[];
    __shared__ __nv_bfloat16 s_gate[kMaxDedupTokens][16];
    __shared__ __nv_bfloat16 s_up[kMaxDedupTokens][16];
    __shared__ int s_peers[kMaxDedupTokens];
    __shared__ int s_cnt;

    NVK_PDL_PROLOG();

    int tid = threadIdx.x;
    int hw = tid >> 4;
    int sub = tid & 15;
    unsigned hmask = (tid & 16) ? 0xffff0000u : 0x0000ffffu;
    int g = blockIdx.y;
    int e = __ldg(&ids[g]);
    int n = blockIdx.x * 16 + hw;

    if (e < 0 || e >= num_experts_total) {
        if (tid < 16) {
            int row = g * a_tile_stride_rows;
            int kb = blockIdx.x;
            if ((tid & 1) == 0) {
                uint8_t* pblock = y_act_fp4 + (size_t)row * (size_t)(n_dim / 2) + (size_t)kb * 8;
                pblock[tid >> 1] = 0u;
            }
            if (tid == 0) {
                y_act_sf[swizzled_scale_dst(row, kb, n_dim / 16)] = gvq_encode_ue4m3(1.f);
            }
        }
        NVK_PDL_EPILOG();
        return;
    }

    if (tid == 0) {
        s_cnt = moe_collect_dedup_peers(ids, num_groups, g, e, s_peers);
    }
    __syncthreads();
    int cnt = s_cnt;
    if (cnt == 0) {
        NVK_PDL_EPILOG();
        return;
    }

    int half_k = k_dim >> 1;
    int group_k = k_dim >> 4;
    int k_tiles = (group_k + 3) >> 2;
    int group_k_pad = k_tiles * 4;

    int* a_i8 = reinterpret_cast<int*>(smem_raw);
    float* a_sc = reinterpret_cast<float*>(a_i8 + kMaxDedupTokens * group_k * 4);
    int total_kb = cnt * group_k;
    for (int idx = tid; idx < total_kb; idx += blockDim.x) {
        int c = idx / group_k;
        int kb = idx - c * group_k;
        int pg = s_peers[c];
        const unsigned long long* a_row = reinterpret_cast<const unsigned long long*>(
            a_packed + (size_t)pg * (size_t)a_tile_stride_rows * (size_t)half_k);
        const uint8_t* a_sf = a_scales +
            (size_t)pg * (size_t)(a_tile_stride_rows / 128) * (size_t)group_k_pad * 128;
        unsigned long long aq = __ldg(&a_row[kb]);
        unsigned ai0, ai1, ai2, ai3;
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)aq, ai0, ai1);
        nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(aq >> 32), ai2, ai3);
        int* dst = a_i8 + idx * 4;
        dst[0] = (int)ai0;
        dst[1] = (int)ai1;
        dst[2] = (int)ai2;
        dst[3] = (int)ai3;
        a_sc[idx] = decode_ue4m3_dev(__ldg(&a_sf[(kb >> 2) * 512 + (kb & 3)])) * 0.25f;
    }
    __syncthreads();

    if (n < n_dim) {
        size_t row = (size_t)e * (size_t)n_dim + (size_t)n;
        const unsigned long long* wg_row = reinterpret_cast<const unsigned long long*>(gate_w + row * (size_t)half_k);
        const unsigned long long* wu_row = reinterpret_cast<const unsigned long long*>(up_w + row * (size_t)half_k);
        const uint8_t* sg_row = gate_sf_lin + row * (size_t)group_k;
        const uint8_t* su_row = up_sf_lin + row * (size_t)group_k;

        if (cnt == 1) {
            float acc_g = 0.f;
            float acc_u = 0.f;
            if ((group_k & 1) == 0) {
                int kp_total = group_k >> 1;
                const uint4* wg_4 = reinterpret_cast<const uint4*>(wg_row);
                const uint4* wu_4 = reinterpret_cast<const uint4*>(wu_row);
                const uint16_t* sg_2 = reinterpret_cast<const uint16_t*>(sg_row);
                const uint16_t* su_2 = reinterpret_cast<const uint16_t*>(su_row);
                const int4* a4 = reinterpret_cast<const int4*>(a_i8);
                const float2* asc2 = reinterpret_cast<const float2*>(a_sc);
                int kp = sub;
                uint4 rg = (kp < kp_total) ? __ldcs(&wg_4[kp]) : make_uint4(0u, 0u, 0u, 0u);
                uint4 ru = (kp < kp_total) ? __ldcs(&wu_4[kp]) : make_uint4(0u, 0u, 0u, 0u);
                uint16_t sg = (kp < kp_total) ? __ldcs(&sg_2[kp]) : 0u;
                uint16_t su = (kp < kp_total) ? __ldcs(&su_2[kp]) : 0u;
                while (kp < kp_total) {
                    int kpn = kp + 16;
                    uint4 ng = (kpn < kp_total) ? __ldcs(&wg_4[kpn]) : make_uint4(0u, 0u, 0u, 0u);
                    uint4 nu = (kpn < kp_total) ? __ldcs(&wu_4[kpn]) : make_uint4(0u, 0u, 0u, 0u);
                    uint16_t nsg = (kpn < kp_total) ? __ldcs(&sg_2[kpn]) : 0u;
                    uint16_t nsu = (kpn < kp_total) ? __ldcs(&su_2[kpn]) : 0u;
                    int kb0 = kp * 2;
                    int4 av0 = a4[kb0];
                    int4 av1 = a4[kb0 + 1];
                    float2 asv = asc2[kp];
                    unsigned wi0, wi1, wi2, wi3;
                    nibbles8_to_doubled_e2m1_int8_lanes(rg.x, wi0, wi1);
                    nibbles8_to_doubled_e2m1_int8_lanes(rg.y, wi2, wi3);
                    int dga = __dp4a((int)wi0, av0.x, 0);
                    dga = __dp4a((int)wi1, av0.y, dga);
                    dga = __dp4a((int)wi2, av0.z, dga);
                    dga = __dp4a((int)wi3, av0.w, dga);
                    nibbles8_to_doubled_e2m1_int8_lanes(rg.z, wi0, wi1);
                    nibbles8_to_doubled_e2m1_int8_lanes(rg.w, wi2, wi3);
                    int dgb = __dp4a((int)wi0, av1.x, 0);
                    dgb = __dp4a((int)wi1, av1.y, dgb);
                    dgb = __dp4a((int)wi2, av1.z, dgb);
                    dgb = __dp4a((int)wi3, av1.w, dgb);
                    acc_g += (decode_ue4m3_dev((uint8_t)(sg & 0xFFu)) * asv.x) * (float)dga
                           + (decode_ue4m3_dev((uint8_t)(sg >> 8)) * asv.y) * (float)dgb;

                    nibbles8_to_doubled_e2m1_int8_lanes(ru.x, wi0, wi1);
                    nibbles8_to_doubled_e2m1_int8_lanes(ru.y, wi2, wi3);
                    int dua = __dp4a((int)wi0, av0.x, 0);
                    dua = __dp4a((int)wi1, av0.y, dua);
                    dua = __dp4a((int)wi2, av0.z, dua);
                    dua = __dp4a((int)wi3, av0.w, dua);
                    nibbles8_to_doubled_e2m1_int8_lanes(ru.z, wi0, wi1);
                    nibbles8_to_doubled_e2m1_int8_lanes(ru.w, wi2, wi3);
                    int dub = __dp4a((int)wi0, av1.x, 0);
                    dub = __dp4a((int)wi1, av1.y, dub);
                    dub = __dp4a((int)wi2, av1.z, dub);
                    dub = __dp4a((int)wi3, av1.w, dub);
                    acc_u += (decode_ue4m3_dev((uint8_t)(su & 0xFFu)) * asv.x) * (float)dua
                           + (decode_ue4m3_dev((uint8_t)(su >> 8)) * asv.y) * (float)dub;
                    kp = kpn;
                    rg = ng; ru = nu; sg = nsg; su = nsu;
                }
            } else {
                int kb = sub;
                unsigned long long rg = (kb < group_k) ? __ldcs(&wg_row[kb]) : 0ull;
                unsigned long long ru = (kb < group_k) ? __ldcs(&wu_row[kb]) : 0ull;
                uint8_t sg = (kb < group_k) ? __ldcs(&sg_row[kb]) : 0u;
                uint8_t su = (kb < group_k) ? __ldcs(&su_row[kb]) : 0u;
                while (kb < group_k) {
                    int kbn = kb + 16;
                    unsigned long long ng = (kbn < group_k) ? __ldcs(&wg_row[kbn]) : 0ull;
                    unsigned long long nu = (kbn < group_k) ? __ldcs(&wu_row[kbn]) : 0ull;
                    uint8_t nsg = (kbn < group_k) ? __ldcs(&sg_row[kbn]) : 0u;
                    uint8_t nsu = (kbn < group_k) ? __ldcs(&su_row[kbn]) : 0u;
                    const int* ap = a_i8 + kb * 4;
                    int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
                    float asc = a_sc[kb];
                    unsigned wi0, wi1, wi2, wi3;
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rg, wi0, wi1);
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rg >> 32), wi2, wi3);
                    int dg = __dp4a((int)wi0, a0, 0);
                    dg = __dp4a((int)wi1, a1, dg);
                    dg = __dp4a((int)wi2, a2, dg);
                    dg = __dp4a((int)wi3, a3, dg);
                    acc_g += (decode_ue4m3_dev(sg) * asc) * (float)dg;

                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)ru, wi0, wi1);
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(ru >> 32), wi2, wi3);
                    int du = __dp4a((int)wi0, a0, 0);
                    du = __dp4a((int)wi1, a1, du);
                    du = __dp4a((int)wi2, a2, du);
                    du = __dp4a((int)wi3, a3, du);
                    acc_u += (decode_ue4m3_dev(su) * asc) * (float)du;
                    kb = kbn;
                    rg = ng; ru = nu; sg = nsg; su = nsu;
                }
            }
            #pragma unroll
            for (int o = 8; o > 0; o >>= 1) {
                acc_g += __shfl_xor_sync(hmask, acc_g, o, 16);
                acc_u += __shfl_xor_sync(hmask, acc_u, o, 16);
            }
            if (sub == 0) {
                s_gate[0][hw] = __float2bfloat16(acc_g * __ldg(&gate_alphas[e]));
                s_up[0][hw] = __float2bfloat16(acc_u * __ldg(&up_alphas[e]));
            }
        } else {
            float acc_g[kMaxDedupTokens];
            float acc_u[kMaxDedupTokens];
            #pragma unroll
            for (int c = 0; c < kMaxDedupTokens; ++c) {
                acc_g[c] = 0.f;
                acc_u[c] = 0.f;
            }
            if ((group_k & 1) == 0) {
                int kp_total = group_k >> 1;
                const uint4* wg_4 = reinterpret_cast<const uint4*>(wg_row);
                const uint4* wu_4 = reinterpret_cast<const uint4*>(wu_row);
                const uint16_t* sg_2 = reinterpret_cast<const uint16_t*>(sg_row);
                const uint16_t* su_2 = reinterpret_cast<const uint16_t*>(su_row);
                const int4* a4 = reinterpret_cast<const int4*>(a_i8);
                const float2* asc2 = reinterpret_cast<const float2*>(a_sc);
                int kp = sub;
                uint4 rg = (kp < kp_total) ? __ldcs(&wg_4[kp]) : make_uint4(0u, 0u, 0u, 0u);
                uint4 ru = (kp < kp_total) ? __ldcs(&wu_4[kp]) : make_uint4(0u, 0u, 0u, 0u);
                uint16_t sg = (kp < kp_total) ? __ldcs(&sg_2[kp]) : 0u;
                uint16_t su = (kp < kp_total) ? __ldcs(&su_2[kp]) : 0u;
                while (kp < kp_total) {
                    int kpn = kp + 16;
                    uint4 ng = (kpn < kp_total) ? __ldcs(&wg_4[kpn]) : make_uint4(0u, 0u, 0u, 0u);
                    uint4 nu = (kpn < kp_total) ? __ldcs(&wu_4[kpn]) : make_uint4(0u, 0u, 0u, 0u);
                    uint16_t nsg = (kpn < kp_total) ? __ldcs(&sg_2[kpn]) : 0u;
                    uint16_t nsu = (kpn < kp_total) ? __ldcs(&su_2[kpn]) : 0u;
                    unsigned wga[4], wgb[4], wua[4], wub[4];
                    nibbles8_to_doubled_e2m1_int8_lanes(rg.x, wga[0], wga[1]);
                    nibbles8_to_doubled_e2m1_int8_lanes(rg.y, wga[2], wga[3]);
                    nibbles8_to_doubled_e2m1_int8_lanes(rg.z, wgb[0], wgb[1]);
                    nibbles8_to_doubled_e2m1_int8_lanes(rg.w, wgb[2], wgb[3]);
                    nibbles8_to_doubled_e2m1_int8_lanes(ru.x, wua[0], wua[1]);
                    nibbles8_to_doubled_e2m1_int8_lanes(ru.y, wua[2], wua[3]);
                    nibbles8_to_doubled_e2m1_int8_lanes(ru.z, wub[0], wub[1]);
                    nibbles8_to_doubled_e2m1_int8_lanes(ru.w, wub[2], wub[3]);
                    float wgsa = decode_ue4m3_dev((uint8_t)(sg & 0xFFu));
                    float wgsb = decode_ue4m3_dev((uint8_t)(sg >> 8));
                    float wusa = decode_ue4m3_dev((uint8_t)(su & 0xFFu));
                    float wusb = decode_ue4m3_dev((uint8_t)(su >> 8));
                    int kb0 = kp * 2;
                    for (int c = 0; c < cnt; ++c) {
                        int idx0 = c * group_k + kb0;
                        int4 av0 = a4[idx0];
                        int4 av1 = a4[idx0 + 1];
                        float2 asv = asc2[c * kp_total + kp];
                        int dga = __dp4a((int)wga[0], av0.x, 0);
                        dga = __dp4a((int)wga[1], av0.y, dga);
                        dga = __dp4a((int)wga[2], av0.z, dga);
                        dga = __dp4a((int)wga[3], av0.w, dga);
                        int dgb = __dp4a((int)wgb[0], av1.x, 0);
                        dgb = __dp4a((int)wgb[1], av1.y, dgb);
                        dgb = __dp4a((int)wgb[2], av1.z, dgb);
                        dgb = __dp4a((int)wgb[3], av1.w, dgb);
                        acc_g[c] += (wgsa * asv.x) * (float)dga + (wgsb * asv.y) * (float)dgb;

                        int dua = __dp4a((int)wua[0], av0.x, 0);
                        dua = __dp4a((int)wua[1], av0.y, dua);
                        dua = __dp4a((int)wua[2], av0.z, dua);
                        dua = __dp4a((int)wua[3], av0.w, dua);
                        int dub = __dp4a((int)wub[0], av1.x, 0);
                        dub = __dp4a((int)wub[1], av1.y, dub);
                        dub = __dp4a((int)wub[2], av1.z, dub);
                        dub = __dp4a((int)wub[3], av1.w, dub);
                        acc_u[c] += (wusa * asv.x) * (float)dua + (wusb * asv.y) * (float)dub;
                    }
                    kp = kpn;
                    rg = ng; ru = nu; sg = nsg; su = nsu;
                }
            } else {
                int kb = sub;
                unsigned long long rg = (kb < group_k) ? __ldcs(&wg_row[kb]) : 0ull;
                unsigned long long ru = (kb < group_k) ? __ldcs(&wu_row[kb]) : 0ull;
                uint8_t sg = (kb < group_k) ? __ldcs(&sg_row[kb]) : 0u;
                uint8_t su = (kb < group_k) ? __ldcs(&su_row[kb]) : 0u;
                while (kb < group_k) {
                    int kbn = kb + 16;
                    unsigned long long ng = (kbn < group_k) ? __ldcs(&wg_row[kbn]) : 0ull;
                    unsigned long long nu = (kbn < group_k) ? __ldcs(&wu_row[kbn]) : 0ull;
                    uint8_t nsg = (kbn < group_k) ? __ldcs(&sg_row[kbn]) : 0u;
                    uint8_t nsu = (kbn < group_k) ? __ldcs(&su_row[kbn]) : 0u;
                    unsigned wgi[4], wui[4];
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)rg, wgi[0], wgi[1]);
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(rg >> 32), wgi[2], wgi[3]);
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)ru, wui[0], wui[1]);
                    nibbles8_to_doubled_e2m1_int8_lanes((unsigned)(ru >> 32), wui[2], wui[3]);
                    float wgs = decode_ue4m3_dev(sg);
                    float wus = decode_ue4m3_dev(su);
                    for (int c = 0; c < cnt; ++c) {
                        int idx = c * group_k + kb;
                        const int* ap = a_i8 + idx * 4;
                        int a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
                        float asc = a_sc[idx];
                        int dg = __dp4a((int)wgi[0], a0, 0);
                        dg = __dp4a((int)wgi[1], a1, dg);
                        dg = __dp4a((int)wgi[2], a2, dg);
                        dg = __dp4a((int)wgi[3], a3, dg);
                        acc_g[c] += (wgs * asc) * (float)dg;

                        int du = __dp4a((int)wui[0], a0, 0);
                        du = __dp4a((int)wui[1], a1, du);
                        du = __dp4a((int)wui[2], a2, du);
                        du = __dp4a((int)wui[3], a3, du);
                        acc_u[c] += (wus * asc) * (float)du;
                    }
                    kb = kbn;
                    rg = ng; ru = nu; sg = nsg; su = nsu;
                }
            }
            float ga = __ldg(&gate_alphas[e]);
            float ua = __ldg(&up_alphas[e]);
            for (int c = 0; c < cnt; ++c) {
                #pragma unroll
                for (int o = 8; o > 0; o >>= 1) {
                    acc_g[c] += __shfl_xor_sync(hmask, acc_g[c], o, 16);
                    acc_u[c] += __shfl_xor_sync(hmask, acc_u[c], o, 16);
                }
                if (sub == 0) {
                    s_gate[c][hw] = __float2bfloat16(acc_g[c] * ga);
                    s_up[c][hw] = __float2bfloat16(acc_u[c] * ua);
                }
            }
        }
    } else if (sub == 0 && hw < 16) {
        for (int c = 0; c < cnt; ++c) {
            s_gate[c][hw] = __float2bfloat16(0.f);
            s_up[c][hw] = __float2bfloat16(0.f);
        }
    }
    __syncthreads();

    if (hw < cnt) {
        int c = hw;
        int i = sub;
        int pg = s_peers[c];
        float gv = __bfloat162float(s_gate[c][i]);
        float uv = __bfloat162float(s_up[c][i]);
        float silu_g = gv / (1.f + expf(-gv));
        float v = silu_g * uv;
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(hmask, amax, o, 16));
        }
        float stored = __ldg(&dn_mini[pg]);
        if (stored == 0.f || !isfinite(stored)) stored = 1.f;
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        uint8_t scale_byte = gvq_encode_ue4m3(stored * local_scale);
        float scale_decoded = decode_ue4m3_dev(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);
        uint8_t nib = gvq_encode_e2m1(fminf(fmaxf(v * inv, -6.f), 6.f));
        unsigned hi = __shfl_down_sync(hmask, (unsigned)nib, 1, 16);
        int row = pg * a_tile_stride_rows;
        int kb = blockIdx.x;
        if ((i & 1) == 0) {
            uint8_t* pblock = y_act_fp4 + (size_t)row * (size_t)(n_dim / 2) + (size_t)kb * 8;
            pblock[i >> 1] = (uint8_t)((hi << 4) | (nib & 0x0Fu));
        }
        if (i == 0) {
            int blocks_per_row = n_dim / 16;
            y_act_sf[swizzled_scale_dst(row, kb, blocks_per_row)] = scale_byte;
        }
    }
    NVK_PDL_EPILOG();
}


template <bool DP4A>
__global__ void moe_grouped_fp4_gemv_m1_bf16_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ b_packed,
    const uint8_t* __restrict__ b_scales,
    const float* __restrict__ alphas,
    __nv_bfloat16* __restrict__ d,
    const int32_t* __restrict__ ids,
    int num_experts_total,
    int n_dim,
    int k_dim,
    int a_tile_stride_rows,
    long long d_group_stride_elems
) {
    constexpr int R = kM1RowsPerWarpSharingTheActivationLoads;
    NVK_PDL_PROLOG();
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n0 = (blockIdx.x * kRowsPerCta + warp) * R;
    int g = blockIdx.y;
    if (n0 >= n_dim) return;

    long long d_base = (long long)g * d_group_stride_elems;
    int e = ids[g];
    if (e < 0 || e >= num_experts_total) {
        if (lane < R && n0 + lane < n_dim) d[d_base + n0 + lane] = __float2bfloat16(0.f);
        return;
    }

    int half_k = k_dim / 2;
    int group_k = k_dim / 16;
    int k_tiles = (group_k + 3) / 4;
    int group_k_pad = k_tiles * 4;
    long long n_pad = (((long long)n_dim + 127) / 128) * 128;

    const unsigned long long* a_row =
        reinterpret_cast<const unsigned long long*>(
            a_packed + (size_t)g * (size_t)a_tile_stride_rows * (size_t)half_k);
    const uint8_t* w_sf = b_scales + (size_t)e * (size_t)n_pad * (size_t)group_k_pad;
    const uint8_t* a_sf = a_scales +
        (size_t)g * (size_t)(a_tile_stride_rows / 128) * (size_t)group_k_pad * 128;

    float acc[R];
    #pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0.f;
    for (int kb = lane; kb < group_k; kb += kWarpSize) {
        float a_scale = decode_ue4m3_dev(__ldg(&a_sf[(kb / 4) * 512 + (kb & 3)]));
        unsigned long long aq = __ldg(&a_row[kb]);
        unsigned long long wq[R];
        uint8_t ws[R];
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            int n = n0 + r;
            if (n < n_dim) {
                const unsigned long long* w_row = reinterpret_cast<const unsigned long long*>(
                    b_packed + ((size_t)e * (size_t)n_dim + (size_t)n) * (size_t)half_k);
                wq[r] = __ldg(&w_row[kb]);
                ws[r] = __ldg(&w_sf[swizzled_scale_dst(n, kb, group_k)]);
            }
        }
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (n0 + r < n_dim) {
                float w_scale = decode_ue4m3_dev(ws[r]);
                acc[r] += (w_scale * a_scale) * dot16_e2m1<DP4A>(wq[r], aq);
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], offset);
        }
    }
    if (lane == 0) {
        float al = __ldg(&alphas[e]);
        #pragma unroll
        for (int r = 0; r < R; ++r) {
            if (n0 + r < n_dim) d[d_base + n0 + r] = __float2bfloat16(acc[r] * al);
        }
    }
    NVK_PDL_EPILOG();
}

constexpr int kGusqBlockN = 16;

template <bool DP4A>
__global__ void moe_grouped_fp4_gemv_gate_up_silu_quant_small_m_kernel(
    const uint8_t* __restrict__ a_packed,
    const uint8_t* __restrict__ a_scales,
    const uint8_t* __restrict__ gate_w,
    const uint8_t* __restrict__ gate_sf,
    const float* __restrict__ gate_alphas,
    const uint8_t* __restrict__ up_w,
    const uint8_t* __restrict__ up_sf,
    const float* __restrict__ up_alphas,
    const float* __restrict__ dn_mini,
    const int32_t* __restrict__ ids,
    uint8_t* __restrict__ y_act_fp4,
    uint8_t* __restrict__ y_act_sf,
    int num_experts_total,
    int n_dim,
    int k_dim,
    int a_tile_stride_rows
) {
    NVK_PDL_PROLOG();
    __shared__ __nv_bfloat16 s_gate[kGusqBlockN];
    __shared__ __nv_bfloat16 s_up[kGusqBlockN];

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kGusqBlockN + warp;
    int g = blockIdx.y;
    int e = ids[g];
    bool valid = (e >= 0 && e < num_experts_total) && n < n_dim;

    if (valid) {
        int half_k = k_dim / 2;
        int group_k = k_dim / 16;
        int k_tiles = (group_k + 3) / 4;
        int group_k_pad = k_tiles * 4;
        long long n_pad = (((long long)n_dim + 127) / 128) * 128;

        const unsigned long long* wg_row =
            reinterpret_cast<const unsigned long long*>(
                gate_w + ((size_t)e * (size_t)n_dim + (size_t)n) * (size_t)half_k);
        const unsigned long long* wu_row =
            reinterpret_cast<const unsigned long long*>(
                up_w + ((size_t)e * (size_t)n_dim + (size_t)n) * (size_t)half_k);
        const unsigned long long* a_row =
            reinterpret_cast<const unsigned long long*>(
                a_packed + (size_t)g * (size_t)a_tile_stride_rows * (size_t)half_k);
        const uint8_t* wg_sf = gate_sf + (size_t)e * (size_t)n_pad * (size_t)group_k_pad;
        const uint8_t* wu_sf = up_sf + (size_t)e * (size_t)n_pad * (size_t)group_k_pad;
        const uint8_t* a_sf = a_scales +
            (size_t)g * (size_t)(a_tile_stride_rows / 128) * (size_t)group_k_pad * 128;

        float acc_g = 0.f;
        float acc_u = 0.f;
        for (int kb = lane; kb < group_k; kb += kWarpSize) {
            float wg_scale =
                decode_ue4m3_dev(__ldg(&wg_sf[swizzled_scale_dst(n, kb, group_k)]));
            float wu_scale =
                decode_ue4m3_dev(__ldg(&wu_sf[swizzled_scale_dst(n, kb, group_k)]));
            float a_scale = decode_ue4m3_dev(__ldg(&a_sf[(kb / 4) * 512 + (kb & 3)]));
            unsigned long long wqg = __ldg(&wg_row[kb]);
            unsigned long long wqu = __ldg(&wu_row[kb]);
            unsigned long long aq = __ldg(&a_row[kb]);
            acc_g += (wg_scale * a_scale) * dot16_e2m1<DP4A>(wqg, aq);
            acc_u += (wu_scale * a_scale) * dot16_e2m1<DP4A>(wqu, aq);
        }

        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            acc_g += __shfl_xor_sync(0xffffffffu, acc_g, offset);
            acc_u += __shfl_xor_sync(0xffffffffu, acc_u, offset);
        }
        if (lane == 0) {
            s_gate[warp] = __float2bfloat16(acc_g * __ldg(&gate_alphas[e]));
            s_up[warp] = __float2bfloat16(acc_u * __ldg(&up_alphas[e]));
        }
    } else if (lane == 0 && warp < kGusqBlockN) {
        s_gate[warp] = __float2bfloat16(0.f);
        s_up[warp] = __float2bfloat16(0.f);
    }
    __syncthreads();

    if (threadIdx.x < 16) {
        int i = threadIdx.x;
        float gv = __bfloat162float(s_gate[i]);
        float uv = __bfloat162float(s_up[i]);
        float silu_g = gv / (1.f + expf(-gv));
        float v = silu_g * uv;
        float amax = fabsf(v);
        #pragma unroll
        for (int o = 8; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0x0000ffffu, amax, o, 16));
        }
        float stored = __ldg(&dn_mini[g]);
        if (stored == 0.f || !isfinite(stored)) stored = 1.f;
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        uint8_t scale_byte = gvq_encode_ue4m3(stored * local_scale);
        float scale_decoded = decode_ue4m3_dev(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);
        uint8_t nib = gvq_encode_e2m1(fminf(fmaxf(v * inv, -6.f), 6.f));
        unsigned hi = __shfl_down_sync(0x0000ffffu, (unsigned)nib, 1, 16);
        int row = g * a_tile_stride_rows;
        int kb = blockIdx.x;
        if ((i & 1) == 0) {
            uint8_t* pblock = y_act_fp4 + (size_t)row * (size_t)(n_dim / 2) + (size_t)kb * 8;
            pblock[i >> 1] = (uint8_t)((hi << 4) | (nib & 0x0Fu));
        }
        if (i == 0) {
            int blocks_per_row = n_dim / 16;
            y_act_sf[swizzled_scale_dst(row, kb, blocks_per_row)] = scale_byte;
        }
    }
    NVK_PDL_EPILOG();
}


}

template <bool DP4A>
int launch_moe_grouped_fp4_gemv_m1_bf16(
    void* stream,
    const uint8_t* a_packed,
    const uint8_t* a_scales,
    const uint8_t* b_packed,
    const uint8_t* b_scales,
    const float* alphas,
    uint16_t* d_bf16,
    const int32_t* group_expert_ids,
    int num_groups,
    int num_experts_total,
    int n,
    int k,
    int a_tile_stride_rows,
    long long d_group_stride_elems
) {
    if (num_groups <= 0 || num_experts_total <= 0 || n <= 0 || k <= 0) return -1;
    if ((k & 15) != 0) return -1;
    if (a_tile_stride_rows <= 0 || (a_tile_stride_rows & 127) != 0) return -1;
    if (d_group_stride_elems < (long long)n) return -1;
    if (((uintptr_t)a_packed & 7) != 0 || ((uintptr_t)b_packed & 7) != 0) return -2;
    if (num_groups > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = (cudaStream_t)stream;
    if constexpr (DP4A) {
        int group_k = k / 16;
        size_t smem = (size_t)kMaxDedupTokens * (size_t)group_k * (4 * sizeof(int) + sizeof(float));
        if (!moe_dedup_disabled() && num_groups <= kMaxDedupGroups && group_k <= 256 && smem <= 48 * 1024) {
            dim3 grid((unsigned)((n + 63) / 64), (unsigned)num_groups);
            if (nvk_pdl_enabled()) {
                NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
                cudaLaunchKernelEx(&cfg, moe_grouped_fp4_gemv_m1_bf16_subwarp8_dedup_dp4a_kernel,
                    a_packed, a_scales, b_packed, b_scales, alphas,
                    reinterpret_cast<__nv_bfloat16*>(d_bf16),
                    group_expert_ids, num_groups, num_experts_total,
                    n, k, a_tile_stride_rows, d_group_stride_elems);
            } else {
                moe_grouped_fp4_gemv_m1_bf16_subwarp8_dedup_dp4a_kernel<<<grid, dim3(256), smem, s>>>(
                    a_packed, a_scales, b_packed, b_scales, alphas,
                    reinterpret_cast<__nv_bfloat16*>(d_bf16),
                    group_expert_ids, num_groups, num_experts_total,
                    n, k, a_tile_stride_rows, d_group_stride_elems);
            }
            return (int)cudaGetLastError();
        }
    }
    dim3 grid((unsigned)((n + kRowsPerCta * kM1RowsPerWarpSharingTheActivationLoads - 1)
                         / (kRowsPerCta * kM1RowsPerWarpSharingTheActivationLoads)),
              (unsigned)num_groups);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(kBlockDim), 0, s);
        cudaLaunchKernelEx(&cfg, moe_grouped_fp4_gemv_m1_bf16_kernel<DP4A>,
            a_packed,
            a_scales,
            b_packed,
            b_scales,
            alphas,
            reinterpret_cast<__nv_bfloat16*>(d_bf16),
            group_expert_ids,
            num_experts_total,
            n,
            k,
            a_tile_stride_rows,
            d_group_stride_elems);
    } else {
        moe_grouped_fp4_gemv_m1_bf16_kernel<DP4A><<<grid, dim3(kBlockDim), 0, s>>>(
            a_packed,
            a_scales,
            b_packed,
            b_scales,
            alphas,
            reinterpret_cast<__nv_bfloat16*>(d_bf16),
            group_expert_ids,
            num_experts_total,
            n,
            k,
            a_tile_stride_rows,
            d_group_stride_elems);
    }
    return (int)cudaGetLastError();
}

template <bool DP4A>
int launch_moe_grouped_fp4_gemv_gate_up_silu_quant_small_m(
    void* stream,
    const uint8_t* a_packed,
    const uint8_t* a_scales,
    const uint8_t* gate_w,
    const uint8_t* gate_sf,
    const float* gate_alphas,
    const uint8_t* up_w,
    const uint8_t* up_sf,
    const float* up_alphas,
    const float* dn_mini,
    const int32_t* group_expert_ids,
    uint8_t* y_act_fp4,
    uint8_t* y_act_sf,
    int num_groups,
    int num_experts_total,
    int n,
    int k,
    int a_tile_stride_rows
) {
    if (num_groups <= 0 || num_experts_total <= 0 || n <= 0 || k <= 0) return -1;
    if ((k & 15) != 0 || (n & 15) != 0) return -1;
    if (a_tile_stride_rows <= 0 || (a_tile_stride_rows & 127) != 0) return -1;
    if (((uintptr_t)a_packed & 7) != 0 || ((uintptr_t)gate_w & 7) != 0
        || ((uintptr_t)up_w & 7) != 0) return -2;
    if (num_groups > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = (cudaStream_t)stream;
    if constexpr (DP4A) {
        int group_k = k / 16;
        size_t smem = (size_t)kMaxDedupTokens * (size_t)group_k * (4 * sizeof(int) + sizeof(float));
        if (!moe_dedup_disabled() && num_groups <= kMaxDedupGroups && smem <= 48 * 1024) {
            dim3 grid((unsigned)(n / kGusqBlockN), (unsigned)num_groups);
            if (nvk_pdl_enabled()) {
                NVK_PDL_ATTR(cfg, grid, dim3(kGusqBlockN * kWarpSize), smem, s);
                cudaLaunchKernelEx(&cfg, moe_grouped_fp4_gemv_gate_up_silu_quant_dedup_dp4a_kernel,
                    a_packed, a_scales, gate_w, gate_sf, gate_alphas,
                    up_w, up_sf, up_alphas, dn_mini, group_expert_ids,
                    y_act_fp4, y_act_sf, num_groups, num_experts_total,
                    n, k, a_tile_stride_rows);
            } else {
                moe_grouped_fp4_gemv_gate_up_silu_quant_dedup_dp4a_kernel<<<grid, dim3(kGusqBlockN * kWarpSize), smem, s>>>(
                    a_packed, a_scales, gate_w, gate_sf, gate_alphas,
                    up_w, up_sf, up_alphas, dn_mini, group_expert_ids,
                    y_act_fp4, y_act_sf, num_groups, num_experts_total,
                    n, k, a_tile_stride_rows);
            }
            return (int)cudaGetLastError();
        }
    }
    dim3 grid((unsigned)(n / kGusqBlockN), (unsigned)num_groups);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(kGusqBlockN * kWarpSize), 0, s);
        cudaLaunchKernelEx(&cfg, moe_grouped_fp4_gemv_gate_up_silu_quant_small_m_kernel<DP4A>,
            a_packed,
            a_scales,
            gate_w,
            gate_sf,
            gate_alphas,
            up_w,
            up_sf,
            up_alphas,
            dn_mini,
            group_expert_ids,
            y_act_fp4,
            y_act_sf,
            num_experts_total,
            n,
            k,
            a_tile_stride_rows);
    } else {
        moe_grouped_fp4_gemv_gate_up_silu_quant_small_m_kernel<DP4A><<<grid,
                                                                dim3(kGusqBlockN * kWarpSize),
                                                                0, s>>>(
            a_packed,
            a_scales,
            gate_w,
            gate_sf,
            gate_alphas,
            up_w,
            up_sf,
            up_alphas,
            dn_mini,
            group_expert_ids,
            y_act_fp4,
            y_act_sf,
            num_experts_total,
            n,
            k,
            a_tile_stride_rows);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_moe_grouped_fp4_gemv_m1_bf16(
    void* stream, const uint8_t* a_packed, const uint8_t* a_scales,
    const uint8_t* b_packed, const uint8_t* b_scales, const float* alphas,
    uint16_t* d_bf16, const int32_t* group_expert_ids,
    int num_groups, int num_experts_total, int n, int k,
    int a_tile_stride_rows, long long d_group_stride_elems
) {
    if (fp4_dp4a_enabled()) {
        return launch_moe_grouped_fp4_gemv_m1_bf16<true>(
            stream, a_packed, a_scales, b_packed, b_scales, alphas, d_bf16,
            group_expert_ids, num_groups, num_experts_total, n, k,
            a_tile_stride_rows, d_group_stride_elems);
    }
    return launch_moe_grouped_fp4_gemv_m1_bf16<false>(
        stream, a_packed, a_scales, b_packed, b_scales, alphas, d_bf16,
        group_expert_ids, num_groups, num_experts_total, n, k,
        a_tile_stride_rows, d_group_stride_elems);
}

extern "C" int nv_kernels_moe_grouped_fp4_gemv_gate_up_silu_quant_small_m(
    void* stream, const uint8_t* a_packed, const uint8_t* a_scales,
    const uint8_t* gate_w, const uint8_t* gate_sf, const float* gate_alphas,
    const uint8_t* up_w, const uint8_t* up_sf, const float* up_alphas,
    const float* dn_mini, const int32_t* group_expert_ids,
    uint8_t* y_act_fp4, uint8_t* y_act_sf,
    int num_groups, int num_experts_total, int n, int k,
    int a_tile_stride_rows
) {
    if (fp4_dp4a_enabled()) {
        return launch_moe_grouped_fp4_gemv_gate_up_silu_quant_small_m<true>(
            stream, a_packed, a_scales, gate_w, gate_sf, gate_alphas,
            up_w, up_sf, up_alphas, dn_mini, group_expert_ids,
            y_act_fp4, y_act_sf, num_groups, num_experts_total, n, k,
            a_tile_stride_rows);
    }
    return launch_moe_grouped_fp4_gemv_gate_up_silu_quant_small_m<false>(
        stream, a_packed, a_scales, gate_w, gate_sf, gate_alphas,
        up_w, up_sf, up_alphas, dn_mini, group_expert_ids,
        y_act_fp4, y_act_sf, num_groups, num_experts_total, n, k,
        a_tile_stride_rows);
}

extern "C" int nv_kernels_moe_unswizzle_scales(
    void* stream,
    const uint8_t* swizzled,
    uint8_t* linear,
    int num_experts,
    int n_dim,
    int k_dim
) {
    if (num_experts <= 0 || n_dim <= 0 || k_dim <= 0 || (k_dim & 15) != 0) return -1;
    if (swizzled == nullptr || linear == nullptr) return -2;
    int group_k = k_dim >> 4;
    size_t total = (size_t)num_experts * (size_t)n_dim * (size_t)group_k;
    int threads = 256;
    size_t blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    moe_unswizzle_scales_kernel<<<(unsigned)blocks, threads, 0, (cudaStream_t)stream>>>(
        swizzled, linear, num_experts, n_dim, group_k);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_moe_grouped_fp4_gemv_m1_bf16_lin(
    void* stream, const uint8_t* a_packed, const uint8_t* a_scales,
    const uint8_t* b_packed, const uint8_t* b_scales_lin, const float* alphas,
    uint16_t* d_bf16, const int32_t* group_expert_ids,
    int num_groups, int num_experts_total, int n, int k,
    int a_tile_stride_rows, long long d_group_stride_elems
) {
    if (num_groups <= 0 || num_experts_total <= 0 || n <= 0 || k <= 0) return -1;
    if ((k & 31) != 0) return -1;
    if (a_tile_stride_rows <= 0 || (a_tile_stride_rows & 127) != 0) return -1;
    if (d_group_stride_elems < (long long)n) return -1;
    if (((uintptr_t)a_packed & 7) != 0 || ((uintptr_t)b_packed & 15) != 0 || ((uintptr_t)b_scales_lin & 1) != 0) return -2;
    if (num_groups > kMaxDedupGroups) return -1;
    int group_k = k >> 4;
    size_t smem = (size_t)kMaxDedupTokens * (size_t)group_k * (4 * sizeof(int) + sizeof(float));
    if (group_k > 256 || smem > 48 * 1024) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((n + 127) / 128), (unsigned)num_groups);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
        cudaLaunchKernelEx(&cfg, moe_grouped_fp4_gemv_m1_bf16_subwarp8_dedup_dp4a_lin_kernel,
            a_packed, a_scales, b_packed, b_scales_lin, alphas,
            reinterpret_cast<__nv_bfloat16*>(d_bf16),
            group_expert_ids, num_groups, num_experts_total,
            n, k, a_tile_stride_rows, d_group_stride_elems);
    } else {
        moe_grouped_fp4_gemv_m1_bf16_subwarp8_dedup_dp4a_lin_kernel<<<grid, dim3(256), smem, s>>>(
            a_packed, a_scales, b_packed, b_scales_lin, alphas,
            reinterpret_cast<__nv_bfloat16*>(d_bf16),
            group_expert_ids, num_groups, num_experts_total,
            n, k, a_tile_stride_rows, d_group_stride_elems);
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_moe_grouped_fp4_gemv_gate_up_silu_quant_small_m_lin(
    void* stream, const uint8_t* a_packed, const uint8_t* a_scales,
    const uint8_t* gate_w, const uint8_t* gate_sf_lin, const float* gate_alphas,
    const uint8_t* up_w, const uint8_t* up_sf_lin, const float* up_alphas,
    const float* dn_mini, const int32_t* group_expert_ids,
    uint8_t* y_act_fp4, uint8_t* y_act_sf,
    int num_groups, int num_experts_total, int n, int k,
    int a_tile_stride_rows
) {
    if (num_groups <= 0 || num_experts_total <= 0 || n <= 0 || k <= 0) return -1;
    if ((k & 31) != 0 || (n & 15) != 0) return -1;
    if (a_tile_stride_rows <= 0 || (a_tile_stride_rows & 127) != 0) return -1;
    if (((uintptr_t)a_packed & 7) != 0 || ((uintptr_t)gate_w & 15) != 0 || ((uintptr_t)up_w & 15) != 0
        || ((uintptr_t)gate_sf_lin & 1) != 0 || ((uintptr_t)up_sf_lin & 1) != 0) return -2;
    if (num_groups > kMaxDedupGroups) return -1;
    int group_k = k >> 4;
    size_t smem = (size_t)kMaxDedupTokens * (size_t)group_k * (4 * sizeof(int) + sizeof(float));
    if (smem > 48 * 1024) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)(n / kGusqBlockN), (unsigned)num_groups);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(256), smem, s);
        cudaLaunchKernelEx(&cfg, moe_grouped_fp4_gemv_gate_up_silu_quant_dedup_dp4a_lin_kernel,
            a_packed, a_scales, gate_w, gate_sf_lin, gate_alphas,
            up_w, up_sf_lin, up_alphas, dn_mini, group_expert_ids,
            y_act_fp4, y_act_sf, num_groups, num_experts_total,
            n, k, a_tile_stride_rows);
    } else {
        moe_grouped_fp4_gemv_gate_up_silu_quant_dedup_dp4a_lin_kernel<<<grid, dim3(256), smem, s>>>(
            a_packed, a_scales, gate_w, gate_sf_lin, gate_alphas,
            up_w, up_sf_lin, up_alphas, dn_mini, group_expert_ids,
            y_act_fp4, y_act_sf, num_groups, num_experts_total,
            n, k, a_tile_stride_rows);
    }
    return (int)cudaGetLastError();
}
