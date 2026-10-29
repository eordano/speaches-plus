
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
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
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

constexpr int kM1RowsPerWarpSharingTheActivationLoads = 4;

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

__device__ __forceinline__ uint8_t gvq_encode_e2m1(float x) {
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

#define NV_UE4M3_MIN_NORMAL 0.015625f

__device__ __forceinline__ uint8_t gvq_encode_ue4m3(float scale) {
    if (!isfinite(scale) || scale <= 0.f) return 0;
    float clamped = fminf(scale, 448.f);
    if (clamped < NV_UE4M3_MIN_NORMAL) {
        int sub = (int)roundf(clamped / NV_UE4M3_SUBNORMAL_STEP);
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

    if (threadIdx.x == 0) {
        float stored = __ldg(&dn_mini[g]);
        if (stored == 0.f || !isfinite(stored)) stored = 1.f;
        int row = g * a_tile_stride_rows;
        int blocks_per_row = n_dim / 16;
        int kb = blockIdx.x;

        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float gv = __bfloat162float(s_gate[i]);
            float uv = __bfloat162float(s_up[i]);
            float silu_g = gv / (1.f + expf(-gv));
            float v = silu_g * uv;
            vals[i] = v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        float stored_scale = stored * local_scale;
        uint8_t scale_byte = gvq_encode_ue4m3(stored_scale);
        float scale_decoded = decode_ue4m3_dev(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);

        uint8_t* pblock = y_act_fp4 + (size_t)row * (size_t)(n_dim / 2) + (size_t)kb * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v_lo = fminf(fmaxf(vals[2 * i] * inv, -6.f), 6.f);
            float v_hi = fminf(fmaxf(vals[2 * i + 1] * inv, -6.f), 6.f);
            uint8_t lo = gvq_encode_e2m1(v_lo);
            uint8_t hi = gvq_encode_e2m1(v_hi);
            pblock[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
        }
        y_act_sf[swizzled_scale_dst(row, kb, blocks_per_row)] = scale_byte;
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
