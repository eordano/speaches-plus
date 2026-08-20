#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kNvfp4Block = 16;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlockSharesOneStagedX = 16;
#define NV_UE4M3_SUBNORMAL_STEP 0.001953125f

struct SmemOptinHighWater {
    size_t raised = 0;
};

static int raise_dynamic_smem_optin_above_48k(
    SmemOptinHighWater& o,
    const void* func,
    size_t smem
) {
    if (smem <= 48 * 1024 || smem <= o.raised) return 0;
    cudaError_t e = cudaFuncSetAttribute(
        func, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return (int)e;
    o.raised = smem;
    return 0;
}

__device__ __constant__ float kE2m1Lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ void fill_pair_lut_smem_because_divergent_constant_indexing_serializes(
    float2* pair_lut
) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        pair_lut[i] = make_float2(kE2m1Lut[i & 0xF], kE2m1Lut[(i >> 4) & 0xF]);
    }
}

__device__ __forceinline__ float decode_ue4m3_scale(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
}

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

__device__ __forceinline__ float warp_row_dot_nvfp4_swizzled_scales(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ scales_sw,
    const float* __restrict__ xs,
    const float2* __restrict__ pair_lut,
    int r,
    int K,
    int lane
) {
    int kb_total = K / kNvfp4Block;
    int k_tiles = (kb_total + 3) >> 2;
    int m_tile = r >> 7;
    int d2 = (r >> 5) & 3;
    int d3 = r & 31;
    int sc_row_base = (m_tile * k_tiles) * 512 + d3 * 16 + d2 * 4;
    const uint2* w8 = reinterpret_cast<const uint2*>(wq + (size_t)r * (K >> 1));
    const float2* xs2 = reinterpret_cast<const float2*>(xs);
    (void)pair_lut;
    float acc = 0.0f;
    for (int kb = lane; kb < kb_total; kb += kWarp) {
        uint2 raw = __ldg(&w8[kb]);
        uint8_t sc = __ldg(&scales_sw[sc_row_base + (kb >> 2) * 512 + (kb & 3)]);
        float sf = decode_ue4m3_scale(sc);
        const float2* xb = xs2 + kb * (kNvfp4Block >> 1);
        float part = 0.0f;
        unsigned expanded[4];
        nibbles8_to_fp8_bytes_via_prmt_because_sm120_fp4_cvt_is_emulated(
            raw.x, expanded[0], expanded[1]);
        nibbles8_to_fp8_bytes_via_prmt_because_sm120_fp4_cvt_is_emulated(
            raw.y, expanded[2], expanded[3]);
        const __nv_fp8x2_storage_t* p2 =
            reinterpret_cast<const __nv_fp8x2_storage_t*>(expanded);
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p2[i], __NV_E4M3);
            float2 wf = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            float2 xv = xb[i];
            part += wf.x * xv.x + wf.y * xv.y;
        }
        acc += sf * part;
    }
    return acc;
}

__global__ void gemv_nvfp4_w4a16_dual_m1_kernel(
    const uint8_t* __restrict__ wq_a,
    const uint8_t* __restrict__ sc_a,
    const uint8_t* __restrict__ wq_b,
    const uint8_t* __restrict__ sc_b,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y_a,
    __nv_bfloat16* __restrict__ y_b,
    float alpha_a,
    float alpha_b,
    int N,
    int K
) {
    extern __shared__ float xs[];
    __shared__ float2 pair_lut[256];
    fill_pair_lut_smem_because_divergent_constant_indexing_serializes(pair_lut);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        xs[i] = __bfloat162float(x[i]);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    int row = blockIdx.x * kWarpsPerBlockSharesOneStagedX + warp;
    int total = (wq_b != nullptr) ? 2 * N : N;
    if (row >= total) return;
    bool second = row >= N;
    int r = second ? row - N : row;
    const uint8_t* wq = second ? wq_b : wq_a;
    const uint8_t* sc = second ? sc_b : sc_a;

    float acc = warp_row_dot_nvfp4_swizzled_scales(wq, sc, xs, pair_lut, r, K, lane);
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        __nv_bfloat16 out = __float2bfloat16(acc * (second ? alpha_b : alpha_a));
        if (second) y_b[r] = out;
        else y_a[r] = out;
    }
}

__global__ void gemv_nvfp4_w4a16_silu_gate_up_in_m1_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ sc,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ y,
    float alpha,
    int N,
    int K
) {
    extern __shared__ float xs[];
    __shared__ float2 pair_lut[256];
    fill_pair_lut_smem_because_divergent_constant_indexing_serializes(pair_lut);
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float act = (g / (1.0f + expf(-g))) * u;
        xs[i] = __bfloat162float(__float2bfloat16(act));
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    int r = blockIdx.x * kWarpsPerBlockSharesOneStagedX + warp;
    if (r >= N) return;

    float acc = warp_row_dot_nvfp4_swizzled_scales(wq, sc, xs, pair_lut, r, K, lane);
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) y[r] = __float2bfloat16(acc * alpha);
}

}

extern "C" int nv_kernels_gemv_nvfp4_w4a16_dual_m1(
    void* stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const uint16_t* x,
    uint16_t* y_a,
    uint16_t* y_b,
    float alpha_a,
    float alpha_b,
    int N,
    int K
) {
    if (N <= 0 || K <= 0 || (K & 31) != 0) return -1;
    if ((wq_b == nullptr) != (y_b == nullptr)) return -1;
    if ((wq_b == nullptr) != (sc_b == nullptr)) return -1;
    size_t smem = (size_t)K * sizeof(float);
    if (smem > 96 * 1024) return -1;
    static SmemOptinHighWater optin_dual;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin_dual, (const void*)gemv_nvfp4_w4a16_dual_m1_kernel, smem);
    if (orc != 0) return orc;
    int total = (wq_b != nullptr) ? 2 * N : N;
    unsigned grid =
        (unsigned)((total + kWarpsPerBlockSharesOneStagedX - 1) / kWarpsPerBlockSharesOneStagedX);
    gemv_nvfp4_w4a16_dual_m1_kernel<<<grid, dim3(kWarpsPerBlockSharesOneStagedX * kWarp), smem,
                                      (cudaStream_t)stream>>>(
        wq_a, sc_a, wq_b, sc_b,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y_a),
        reinterpret_cast<__nv_bfloat16*>(y_b),
        alpha_a, alpha_b, N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_nvfp4_w4a16_silu_gate_up_in_m1(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* y,
    float alpha,
    int N,
    int K
) {
    if (N <= 0 || K <= 0 || (K & 31) != 0) return -1;
    size_t smem = (size_t)K * sizeof(float);
    if (smem > 96 * 1024) return -1;
    static SmemOptinHighWater optin_silu;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin_silu, (const void*)gemv_nvfp4_w4a16_silu_gate_up_in_m1_kernel, smem);
    if (orc != 0) return orc;
    unsigned grid =
        (unsigned)((N + kWarpsPerBlockSharesOneStagedX - 1) / kWarpsPerBlockSharesOneStagedX);
    gemv_nvfp4_w4a16_silu_gate_up_in_m1_kernel<<<grid,
                                                 dim3(kWarpsPerBlockSharesOneStagedX * kWarp),
                                                 smem, (cudaStream_t)stream>>>(
        wq, sc,
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(up),
        reinterpret_cast<__nv_bfloat16*>(y),
        alpha, N, K);
    return (int)cudaGetLastError();
}
