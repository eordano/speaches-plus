#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kNvfp4Block = 16;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlockSharesOneStagedX = 16;
#define NV_UE4M3_SUBNORMAL_STEP 0.001953125f
constexpr float kHalfUndoesTheDoubledE2m1IntTables = 0.5f;
constexpr unsigned long long
    kPackedSentinelZeroIsBelowEveryRealKeyBecauseNoFiniteOrdKeyPairsWithIdxAllOnes = 0ull;

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

__device__ __forceinline__ float decode_ue4m3_scale(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
}

__device__ __forceinline__ void nibbles8_to_doubled_e2m1_int8_lanes_because_dp4a_needs_integers(
    unsigned w,
    unsigned& lo,
    unsigned& hi
) {
    constexpr unsigned kTwiceE2m1MagBytes0to3 = 0x03020100u;
    constexpr unsigned kTwiceE2m1MagBytes4to7 = 0x0C080604u;
    constexpr unsigned kSignFillByteTable = 0x0000FF00u;
    unsigned mag_sel = w & 0x77777777u;
    unsigned sgn_sel = (w >> 3) & 0x11111111u;
    unsigned mag_lo = __byte_perm(kTwiceE2m1MagBytes0to3, kTwiceE2m1MagBytes4to7, mag_sel);
    unsigned sgn_lo = __byte_perm(kSignFillByteTable, 0u, sgn_sel);
    lo = __vsub4(mag_lo ^ sgn_lo, sgn_lo);
    unsigned mag_hi = __byte_perm(kTwiceE2m1MagBytes0to3, kTwiceE2m1MagBytes4to7, mag_sel >> 16);
    unsigned sgn_hi = __byte_perm(kSignFillByteTable, 0u, sgn_sel >> 16);
    hi = __vsub4(mag_hi ^ sgn_hi, sgn_hi);
}

__device__ __forceinline__ float warp_row_dot_nvfp4_q8_dp4a_swizzled_scales(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ scales_sw,
    const int* __restrict__ xs_q8_words,
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
    float acc = 0.0f;
    int kb = lane;
    uint2 raw = (kb < kb_total) ? __ldcs(&w8[kb]) : make_uint2(0, 0);
    while (kb < kb_total) {
        int kb_next = kb + kWarp;
        uint2 nxt = (kb_next < kb_total) ? __ldcs(&w8[kb_next]) : make_uint2(0, 0);
        uint8_t sc = __ldg(&scales_sw[sc_row_base + (kb >> 2) * 512 + (kb & 3)]);
        float sf = decode_ue4m3_scale(sc);
        unsigned v0, v1, v2, v3;
        nibbles8_to_doubled_e2m1_int8_lanes_because_dp4a_needs_integers(raw.x, v0, v1);
        nibbles8_to_doubled_e2m1_int8_lanes_because_dp4a_needs_integers(raw.y, v2, v3);
        const int* xw = xs_q8_words + kb * 4;
        int idot = __dp4a((int)v0, xw[0], 0);
        idot = __dp4a((int)v1, xw[1], idot);
        idot = __dp4a((int)v2, xw[2], idot);
        idot = __dp4a((int)v3, xw[3], idot);
        acc += sf * (float)idot;
        kb = kb_next;
        raw = nxt;
    }
    return acc;
}

__device__ __forceinline__ void stage_q8_row_to_smem_int4(
    const int8_t* __restrict__ x_q8,
    int* xs_q8_words,
    int K
) {
    const int4* xg = reinterpret_cast<const int4*>(x_q8);
    int4* xs4 = reinterpret_cast<int4*>(xs_q8_words);
    for (int i = threadIdx.x; i < (K >> 4); i += blockDim.x) {
        xs4[i] = __ldg(&xg[i]);
    }
}

__device__ __forceinline__ unsigned ordered_u32_key_from_f32_where_neg_zero_ties_pos_zero(
    float v
) {
    unsigned b = __float_as_uint(v);
    return (v >= 0.0f) ? (b | 0x80000000u) : ~b;
}

__device__ __forceinline__ float f32_from_ordered_u32_key_where_both_zeros_read_back_pos(
    unsigned o
) {
    unsigned b = (o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o;
    return __uint_as_float(b);
}

__device__ __forceinline__ unsigned long long
packed_key_orders_ties_to_the_lowest_index_under_atomic_max(unsigned ord, unsigned idx) {
    return ((unsigned long long)ord << 32) | (unsigned long long)(idx ^ 0xFFFFFFFFu);
}

__global__ void gemv_nvfp4_w4a8_dual_argmax_m1_kernel(
    const uint8_t* __restrict__ wq_a,
    const uint8_t* __restrict__ sc_a,
    const uint8_t* __restrict__ wq_b,
    const uint8_t* __restrict__ sc_b,
    const int8_t* __restrict__ x_q8,
    const float* __restrict__ x_dequant_scale,
    unsigned long long* __restrict__ packed_best,
    float alpha_a,
    float alpha_b,
    int N,
    int N_bot,
    int bot_id_base,
    int K
) {
    extern __shared__ int xs_q8_words[];
    stage_q8_row_to_smem_int4(x_q8, xs_q8_words, K);
    __syncthreads();

    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    int row = blockIdx.x * kWarpsPerBlockSharesOneStagedX + warp;
    bool active = row < N + N_bot;
    bool second = row >= N;
    int r = second ? row - N : row;

    float acc = 0.0f;
    if (active) {
        acc = warp_row_dot_nvfp4_q8_dp4a_swizzled_scales(
            second ? wq_b : wq_a, second ? sc_b : sc_a, xs_q8_words, r, K, lane);
        #pragma unroll
        for (int o = kWarp / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    }
    __shared__ unsigned long long warp_best[kWarpsPerBlockSharesOneStagedX];
    if (lane == 0) {
        unsigned long long packed =
            kPackedSentinelZeroIsBelowEveryRealKeyBecauseNoFiniteOrdKeyPairsWithIdxAllOnes;
        if (active) {
            float s = (second ? alpha_b : alpha_a)
                * kHalfUndoesTheDoubledE2m1IntTables * __ldg(x_dequant_scale);
            float logit_as_the_unfused_path_would_have_materialized_it =
                __bfloat162float(__float2bfloat16(acc * s));
            unsigned ord = ordered_u32_key_from_f32_where_neg_zero_ties_pos_zero(
                logit_as_the_unfused_path_would_have_materialized_it);
            unsigned token_id = second ? (unsigned)(bot_id_base + r) : (unsigned)row;
            packed = packed_key_orders_ties_to_the_lowest_index_under_atomic_max(
                ord, token_id);
        }
        warp_best[warp] = packed;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long blk =
            kPackedSentinelZeroIsBelowEveryRealKeyBecauseNoFiniteOrdKeyPairsWithIdxAllOnes;
        for (int w = 0; w < kWarpsPerBlockSharesOneStagedX; ++w) {
            if (warp_best[w] > blk) blk = warp_best[w];
        }
        atomicMax(packed_best, blk);
    }
}

__global__ void unpack_argmax_id_and_logit_from_packed_best_kernel(
    const unsigned long long* __restrict__ packed_best,
    uint32_t* __restrict__ out_id,
    float* __restrict__ out_logit
) {
    unsigned long long p = *packed_best;
    out_id[0] = ((unsigned)(p & 0xFFFFFFFFull)) ^ 0xFFFFFFFFu;
    if (out_logit != nullptr) {
        out_logit[0] = f32_from_ordered_u32_key_where_both_zeros_read_back_pos(
            (unsigned)(p >> 32));
    }
}

}

extern "C" int nv_kernels_gemv_nvfp4_w4a8_dual_argmax_m1(
    void* stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const int8_t* x_q8,
    const float* x_dequant_scale,
    unsigned long long* packed_best_scratch,
    uint32_t* out_id,
    float* out_logit,
    float alpha_a,
    float alpha_b,
    int N,
    int N_bot,
    int bot_id_base,
    int K
) {
    if (N <= 0 || N_bot <= 0 || bot_id_base < 0 || K <= 0 || (K & 31) != 0) return -1;
    if (packed_best_scratch == nullptr || out_id == nullptr) return -1;
    size_t smem = (size_t)K;
    if (smem > 96 * 1024) return -1;
    static SmemOptinHighWater optin_dual_argmax;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin_dual_argmax, (const void*)gemv_nvfp4_w4a8_dual_argmax_m1_kernel, smem);
    if (orc != 0) return orc;
    cudaStream_t s = (cudaStream_t)stream;
    cudaError_t me = cudaMemsetAsync(
        packed_best_scratch, 0, sizeof(unsigned long long), s);
    if (me != cudaSuccess) return (int)me;
    unsigned grid = (unsigned)((N + N_bot + kWarpsPerBlockSharesOneStagedX - 1)
        / kWarpsPerBlockSharesOneStagedX);
    gemv_nvfp4_w4a8_dual_argmax_m1_kernel<<<grid,
                                            dim3(kWarpsPerBlockSharesOneStagedX * kWarp),
                                            smem, s>>>(
        wq_a, sc_a, wq_b, sc_b, x_q8, x_dequant_scale, packed_best_scratch,
        alpha_a, alpha_b, N, N_bot, bot_id_base, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    unpack_argmax_id_and_logit_from_packed_best_kernel<<<1, 1, 0, s>>>(
        packed_best_scratch, out_id, out_logit);
    return (int)cudaGetLastError();
}
