
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "nvk_grid.cuh"

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

__device__ __forceinline__ float dot16_e2m1(unsigned long long wq,
                                            unsigned long long aq) {
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
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerCta + warp;
    int g = blockIdx.y;
    if (n >= n_dim) return;

    long long d_idx = (long long)g * d_group_stride_elems + n;
    int e = ids[g];
    if (e < 0 || e >= num_experts_total) {
        if (lane == 0) d[d_idx] = __float2bfloat16(0.f);
        return;
    }

    int half_k = k_dim / 2;
    int group_k = k_dim / 16;
    int k_tiles = (group_k + 3) / 4;
    int group_k_pad = k_tiles * 4;
    long long n_pad = (((long long)n_dim + 127) / 128) * 128;

    const unsigned long long* w_row =
        reinterpret_cast<const unsigned long long*>(
            b_packed + ((size_t)e * (size_t)n_dim + (size_t)n) * (size_t)half_k);
    const unsigned long long* a_row =
        reinterpret_cast<const unsigned long long*>(
            a_packed + (size_t)g * (size_t)a_tile_stride_rows * (size_t)half_k);
    const uint8_t* w_sf = b_scales + (size_t)e * (size_t)n_pad * (size_t)group_k_pad;
    const uint8_t* a_sf = a_scales +
        (size_t)g * (size_t)(a_tile_stride_rows / 128) * (size_t)group_k_pad * 128;

    float acc = 0.f;
    for (int kb = lane; kb < group_k; kb += kWarpSize) {
        float w_scale =
            decode_ue4m3_dev(__ldg(&w_sf[swizzled_scale_dst(n, kb, group_k)]));
        float a_scale = decode_ue4m3_dev(__ldg(&a_sf[(kb / 4) * 512 + (kb & 3)]));
        unsigned long long wq = __ldg(&w_row[kb]);
        unsigned long long aq = __ldg(&a_row[kb]);
        acc += (w_scale * a_scale) * dot16_e2m1(wq, aq);
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        d[d_idx] = __float2bfloat16(acc * __ldg(&alphas[e]));
    }
}

}

extern "C" int nv_kernels_moe_grouped_fp4_gemv_m1_bf16(
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
    dim3 grid((unsigned)((n + kRowsPerCta - 1) / kRowsPerCta),
              (unsigned)num_groups);
    moe_grouped_fp4_gemv_m1_bf16_kernel<<<grid, dim3(kBlockDim), 0, s>>>(
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
    return (int)cudaGetLastError();
}
