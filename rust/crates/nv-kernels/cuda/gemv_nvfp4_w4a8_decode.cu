#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

namespace {

constexpr int kNvfp4Block = 16;
constexpr int kWarp = 32;
constexpr int kWarpsPerBlockSharesOneStagedX = 16;
#define NV_UE4M3_SUBNORMAL_STEP 0.001953125f
constexpr float kHalfUndoesTheDoubledE2m1IntTables = 0.5f;
constexpr int kSiluQuantBlockThreads = 1024;

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

__global__ void gemv_nvfp4_w4a8_dual_m1_kernel(
    const uint8_t* __restrict__ wq_a,
    const uint8_t* __restrict__ sc_a,
    const uint8_t* __restrict__ wq_b,
    const uint8_t* __restrict__ sc_b,
    const int8_t* __restrict__ x_q8,
    const float* __restrict__ x_dequant_scale,
    __nv_bfloat16* __restrict__ y_a,
    __nv_bfloat16* __restrict__ y_b,
    float alpha_a,
    float alpha_b,
    int N,
    int K
) {
    extern __shared__ int xs_q8_words[];
    stage_q8_row_to_smem_int4(x_q8, xs_q8_words, K);
    __syncthreads();

    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    int row = blockIdx.x * kWarpsPerBlockSharesOneStagedX + warp;
    if (row >= 2 * N) return;
    bool second = row >= N;
    int r = second ? row - N : row;

    float acc = warp_row_dot_nvfp4_q8_dp4a_swizzled_scales(
        second ? wq_b : wq_a, second ? sc_b : sc_a, xs_q8_words, r, K, lane);
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        float s = (second ? alpha_b : alpha_a)
            * kHalfUndoesTheDoubledE2m1IntTables * __ldg(x_dequant_scale);
        __nv_bfloat16 out = __float2bfloat16(acc * s);
        if (second) y_b[r] = out;
        else y_a[r] = out;
    }
}

__global__ void silu_mul_rowquant_i8_m1_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    int8_t* __restrict__ act_q8,
    float* __restrict__ act_dequant_scale,
    int K
) {
    extern __shared__ __nv_bfloat16 act_staged_bf16_so_quant_reads_the_same_rounded_value[];
    float amax = 0.0f;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float g = __bfloat162float(gate[i]);
        float u = __bfloat162float(up[i]);
        float a = (g / (1.0f + expf(-g))) * u;
        __nv_bfloat16 ab = __float2bfloat16(a);
        act_staged_bf16_so_quant_reads_the_same_rounded_value[i] = ab;
        amax = fmaxf(amax, fabsf(__bfloat162float(ab)));
    }
    __shared__ float red[kSiluQuantBlockThreads];
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float scale = red[0] / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    if (threadIdx.x == 0) act_dequant_scale[0] = scale;
    for (int i = threadIdx.x; i < K; i += blockDim.x) {
        float v = __bfloat162float(act_staged_bf16_so_quant_reads_the_same_rounded_value[i]) * inv;
        int q = __float2int_rn(v);
        q = max(-127, min(127, q));
        act_q8[i] = (int8_t)q;
    }
}

constexpr int kSiluQuantSplitThreads = 256;
constexpr int kSiluQuantSplitMaxBlocksKeepsPass2PartialFanInOneSmemRound = 128;

__device__ __forceinline__ int silu_quant_split_blocks_x(int K) {
    int b = (K + kSiluQuantSplitThreads - 1) / kSiluQuantSplitThreads;
    return b < kSiluQuantSplitMaxBlocksKeepsPass2PartialFanInOneSmemRound
        ? b
        : kSiluQuantSplitMaxBlocksKeepsPass2PartialFanInOneSmemRound;
}

__global__ void silu_mul_stage_bf16_partial_absmax_mk_pass1_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ act_staged_bf16,
    float* __restrict__ partial_absmax,
    int K
) {
    int row = blockIdx.y;
    const __nv_bfloat16* g_row = gate + (size_t)row * K;
    const __nv_bfloat16* u_row = up + (size_t)row * K;
    __nv_bfloat16* a_row = act_staged_bf16 + (size_t)row * K;
    float amax = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < K; i += gridDim.x * blockDim.x) {
        float g = __bfloat162float(g_row[i]);
        float u = __bfloat162float(u_row[i]);
        float a = (g / (1.0f + expf(-g))) * u;
        __nv_bfloat16 ab = __float2bfloat16(a);
        a_row[i] = ab;
        amax = fmaxf(amax, fabsf(__bfloat162float(ab)));
    }
    __shared__ float red[kSiluQuantSplitThreads];
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_absmax[(size_t)row * gridDim.x + blockIdx.x] = red[0];
    }
}

__global__ void rowquant_i8_from_staged_bf16_partials_mk_pass2_kernel(
    const __nv_bfloat16* __restrict__ act_staged_bf16,
    const float* __restrict__ partial_absmax,
    int num_partials,
    int8_t* __restrict__ act_q8,
    float* __restrict__ act_dequant_scales,
    int K
) {
    int row = blockIdx.y;
    __shared__ float red[kSiluQuantSplitThreads];
    float a = 0.0f;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        a = fmaxf(a, partial_absmax[(size_t)row * num_partials + i]);
    }
    red[threadIdx.x] = a;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float scale = red[0] / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        act_dequant_scales[row] = scale;
    }
    const __nv_bfloat16* a_row = act_staged_bf16 + (size_t)row * K;
    int8_t* q_row = act_q8 + (size_t)row * K;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < K; i += gridDim.x * blockDim.x) {
        float v = __bfloat162float(a_row[i]) * inv;
        int q = __float2int_rn(v);
        q = max(-127, min(127, q));
        q_row[i] = (int8_t)q;
    }
}

__global__ void gemv_nvfp4_w4a8_down_residual_quant_prologue_m1_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ sc,
    const __nv_bfloat16* __restrict__ act_staged_bf16,
    const float* __restrict__ partial_absmax,
    int num_partials,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ y,
    float alpha,
    int N,
    int K
) {
    extern __shared__ int xs_q8_words[];
    __shared__ float red[kWarpsPerBlockSharesOneStagedX * kWarp];
    float a = 0.0f;
    for (int i = threadIdx.x; i < num_partials; i += blockDim.x) {
        a = fmaxf(a, __ldg(&partial_absmax[i]));
    }
    red[threadIdx.x] = a;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float scale = red[0] / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;

    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(act_staged_bf16);
    uint16_t* xs_pairs = reinterpret_cast<uint16_t*>(xs_q8_words);
    for (int p = threadIdx.x; p < (K >> 1); p += blockDim.x) {
        __nv_bfloat162 v = __ldg(&a2[p]);
        int q0 = __float2int_rn(__bfloat162float(v.x) * inv);
        int q1 = __float2int_rn(__bfloat162float(v.y) * inv);
        q0 = max(-127, min(127, q0));
        q1 = max(-127, min(127, q1));
        xs_pairs[p] = (uint16_t)((q0 & 0xff) | ((q1 & 0xff) << 8));
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    int r = blockIdx.x * kWarpsPerBlockSharesOneStagedX + warp;
    if (r >= N) return;

    float acc = warp_row_dot_nvfp4_q8_dp4a_swizzled_scales(wq, sc, xs_q8_words, r, K, lane);
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    if (lane == 0) {
        float s = alpha * kHalfUndoesTheDoubledE2m1IntTables * scale;
        float base = (residual != nullptr) ? __bfloat162float(residual[r]) : 0.0f;
        y[r] = __float2bfloat16(fmaf(acc, s, base));
    }
}

constexpr int kNormQuantFoldBlockThreads = 1024;
constexpr int kNormQuantFoldMaxHiddenBoundsTheTwoU16SmemRows = 24576;

__global__ void rmsnorm_residual_writeout_rowquant_i8_m1_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ res_in,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ res_out,
    int8_t* __restrict__ out_q8,
    float* __restrict__ out_dequant_scale,
    int hidden,
    float eps
) {
    extern __shared__ __nv_bfloat16
        summed_then_normed_staged_bf16_so_quant_reads_the_same_rounded_values[];
    __nv_bfloat16* s_staged = summed_then_normed_staged_bf16_so_quant_reads_the_same_rounded_values;
    __nv_bfloat16* v_staged = s_staged + hidden;
    __shared__ float red[kNormQuantFoldBlockThreads];
    __shared__ float row_stat;

    float sumsq = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float xv = __bfloat162float(x[i]);
        float rv = __bfloat162float(res_in[i]);
        float s = xv + rv;
        __nv_bfloat16 sb = __float2bfloat16(s);
        res_out[i] = sb;
        s_staged[i] = sb;
        sumsq += s * s;
    }
    red[threadIdx.x] = sumsq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) row_stat = rsqrtf(red[0] / (float)hidden + eps);
    __syncthreads();
    float rms = row_stat;

    float amax = 0.0f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float sv = __bfloat162float(s_staged[i]);
        float wv = __bfloat162float(weight[i]);
        __nv_bfloat16 vb = __float2bfloat16(sv * rms * wv);
        v_staged[i] = vb;
        amax = fmaxf(amax, fabsf(__bfloat162float(vb)));
    }
    __syncthreads();
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < (unsigned)s) {
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        }
        __syncthreads();
    }
    float scale = red[0] / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    if (threadIdx.x == 0) out_dequant_scale[0] = scale;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(v_staged[i]) * inv;
        int q = __float2int_rn(v);
        q = max(-127, min(127, q));
        out_q8[i] = (int8_t)q;
    }
}

template <bool kEmitNextPreNormRstdSoTheSeparateRmsnormKernelDies>
__global__ void gemv_nvfp4_w4a8_down_residual_m1_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ sc,
    const int8_t* __restrict__ x_q8,
    const float* __restrict__ x_dequant_scale,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ y,
    float alpha,
    float* __restrict__ rstd_ssq_count_pack,
    float rstd_eps,
    int N,
    int K
) {
    extern __shared__ int xs_q8_words[];
    stage_q8_row_to_smem_int4(x_q8, xs_q8_words, K);
    __syncthreads();

    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    int r = blockIdx.x * kWarpsPerBlockSharesOneStagedX + warp;
    bool active = r < N;

    float summed_bf16_rounded = 0.0f;
    if (active) {
        float acc = warp_row_dot_nvfp4_q8_dp4a_swizzled_scales(wq, sc, xs_q8_words, r, K, lane);
        #pragma unroll
        for (int o = kWarp / 2; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            float s = alpha * kHalfUndoesTheDoubledE2m1IntTables * __ldg(x_dequant_scale);
            float base = (residual != nullptr) ? __bfloat162float(residual[r]) : 0.0f;
            __nv_bfloat16 out = __float2bfloat16(fmaf(acc, s, base));
            y[r] = out;
            if (kEmitNextPreNormRstdSoTheSeparateRmsnormKernelDies) {
                summed_bf16_rounded = __bfloat162float(out);
            }
        }
    }
    if (kEmitNextPreNormRstdSoTheSeparateRmsnormKernelDies) {
        __shared__ float warp_lane0_sq[kWarpsPerBlockSharesOneStagedX];
        if (lane == 0) warp_lane0_sq[warp] = summed_bf16_rounded * summed_bf16_rounded;
        __syncthreads();
        if (threadIdx.x == 0) {
            float blk = 0.0f;
            for (int w = 0; w < kWarpsPerBlockSharesOneStagedX; ++w) blk += warp_lane0_sq[w];
            atomicAdd(&rstd_ssq_count_pack[1], blk);
            __threadfence();
            unsigned* count = reinterpret_cast<unsigned*>(rstd_ssq_count_pack) + 2;
            unsigned prev = atomicInc(count, gridDim.x - 1);
            if (prev == gridDim.x - 1) {
                float ssq = *((volatile float*)&rstd_ssq_count_pack[1]);
                rstd_ssq_count_pack[0] = rsqrtf(ssq / (float)N + rstd_eps);
                rstd_ssq_count_pack[1] = 0.0f;
            }
        }
    }
}

template <int TM>
__global__ void gemv_nvfp4_w4a8_dual_mk_kernel(
    const uint8_t* __restrict__ wq_a,
    const uint8_t* __restrict__ sc_a,
    const uint8_t* __restrict__ wq_b,
    const uint8_t* __restrict__ sc_b,
    const int8_t* __restrict__ x_q8,
    const float* __restrict__ x_dequant_scales,
    __nv_bfloat16* __restrict__ y_a,
    __nv_bfloat16* __restrict__ y_b,
    float alpha_a,
    float alpha_b,
    int N,
    int K
) {
    extern __shared__ int xs_q8_words[];
    {
        const int4* xg = reinterpret_cast<const int4*>(x_q8);
        int4* xs4 = reinterpret_cast<int4*>(xs_q8_words);
        for (int i = threadIdx.x; i < TM * (K >> 4); i += blockDim.x) {
            xs4[i] = __ldg(&xg[i]);
        }
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    int row = blockIdx.x * kWarpsPerBlockSharesOneStagedX + warp;
    if (row >= 2 * N) return;
    bool second = row >= N;
    int r = second ? row - N : row;
    const uint8_t* wq = second ? wq_b : wq_a;
    const uint8_t* scales_sw = second ? sc_b : sc_a;

    int kb_total = K / kNvfp4Block;
    int k_tiles = (kb_total + 3) >> 2;
    int m_tile = r >> 7;
    int d2 = (r >> 5) & 3;
    int d3 = r & 31;
    int sc_row_base = (m_tile * k_tiles) * 512 + d3 * 16 + d2 * 4;
    const uint2* w8 = reinterpret_cast<const uint2*>(wq + (size_t)r * (K >> 1));
    float acc[TM];
    #pragma unroll
    for (int j = 0; j < TM; ++j) acc[j] = 0.0f;
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
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int* xw = xs_q8_words + j * (K >> 2) + kb * 4;
            int idot = __dp4a((int)v0, xw[0], 0);
            idot = __dp4a((int)v1, xw[1], idot);
            idot = __dp4a((int)v2, xw[2], idot);
            idot = __dp4a((int)v3, xw[3], idot);
            acc[j] += sf * (float)idot;
        }
        kb = kb_next;
        raw = nxt;
    }
    float alpha = second ? alpha_b : alpha_a;
    __nv_bfloat16* y = second ? y_b : y_a;
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        float a = acc[j];
        #pragma unroll
        for (int o = kWarp / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) {
            float s = alpha * kHalfUndoesTheDoubledE2m1IntTables * __ldg(&x_dequant_scales[j]);
            y[(size_t)j * N + r] = __float2bfloat16(a * s);
        }
    }
}

template <int TM>
__global__ void gemv_nvfp4_w4a8_down_residual_mk_kernel(
    const uint8_t* __restrict__ wq,
    const uint8_t* __restrict__ sc,
    const int8_t* __restrict__ x_q8,
    const float* __restrict__ x_dequant_scales,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ y,
    float alpha,
    int N,
    int K
) {
    extern __shared__ int xs_q8_words[];
    {
        const int4* xg = reinterpret_cast<const int4*>(x_q8);
        int4* xs4 = reinterpret_cast<int4*>(xs_q8_words);
        for (int i = threadIdx.x; i < TM * (K >> 4); i += blockDim.x) {
            xs4[i] = __ldg(&xg[i]);
        }
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    int r = blockIdx.x * kWarpsPerBlockSharesOneStagedX + warp;
    if (r >= N) return;

    int kb_total = K / kNvfp4Block;
    int k_tiles = (kb_total + 3) >> 2;
    int m_tile = r >> 7;
    int d2 = (r >> 5) & 3;
    int d3 = r & 31;
    int sc_row_base = (m_tile * k_tiles) * 512 + d3 * 16 + d2 * 4;
    const uint2* w8 = reinterpret_cast<const uint2*>(wq + (size_t)r * (K >> 1));
    float acc[TM];
    #pragma unroll
    for (int j = 0; j < TM; ++j) acc[j] = 0.0f;
    int kb = lane;
    uint2 raw = (kb < kb_total) ? __ldcs(&w8[kb]) : make_uint2(0, 0);
    while (kb < kb_total) {
        int kb_next = kb + kWarp;
        uint2 nxt = (kb_next < kb_total) ? __ldcs(&w8[kb_next]) : make_uint2(0, 0);
        uint8_t sc_b = __ldg(&sc[sc_row_base + (kb >> 2) * 512 + (kb & 3)]);
        float sf = decode_ue4m3_scale(sc_b);
        unsigned v0, v1, v2, v3;
        nibbles8_to_doubled_e2m1_int8_lanes_because_dp4a_needs_integers(raw.x, v0, v1);
        nibbles8_to_doubled_e2m1_int8_lanes_because_dp4a_needs_integers(raw.y, v2, v3);
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            const int* xw = xs_q8_words + j * (K >> 2) + kb * 4;
            int idot = __dp4a((int)v0, xw[0], 0);
            idot = __dp4a((int)v1, xw[1], idot);
            idot = __dp4a((int)v2, xw[2], idot);
            idot = __dp4a((int)v3, xw[3], idot);
            acc[j] += sf * (float)idot;
        }
        kb = kb_next;
        raw = nxt;
    }
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        float a = acc[j];
        #pragma unroll
        for (int o = kWarp / 2; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        if (lane == 0) {
            float s = alpha * kHalfUndoesTheDoubledE2m1IntTables * __ldg(&x_dequant_scales[j]);
            float base = (residual != nullptr)
                ? __bfloat162float(residual[(size_t)j * N + r])
                : 0.0f;
            y[(size_t)j * N + r] = __float2bfloat16(fmaf(a, s, base));
        }
    }
}

template <int TM>
static int launch_dual_mk(
    cudaStream_t s,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const int8_t* x_q8,
    const float* x_scales,
    __nv_bfloat16* y_a,
    __nv_bfloat16* y_b,
    float alpha_a,
    float alpha_b,
    int N,
    int K
) {
    size_t smem = (size_t)TM * K;
    static SmemOptinHighWater optin;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin, (const void*)gemv_nvfp4_w4a8_dual_mk_kernel<TM>, smem);
    if (orc != 0) return orc;
    unsigned grid = (unsigned)((2 * N + kWarpsPerBlockSharesOneStagedX - 1)
        / kWarpsPerBlockSharesOneStagedX);
    gemv_nvfp4_w4a8_dual_mk_kernel<TM><<<grid, dim3(kWarpsPerBlockSharesOneStagedX * kWarp),
                                         smem, s>>>(
        wq_a, sc_a, wq_b, sc_b, x_q8, x_scales, y_a, y_b, alpha_a, alpha_b, N, K);
    return (int)cudaGetLastError();
}

template <int TM>
static int launch_down_mk(
    cudaStream_t s,
    const uint8_t* wq,
    const uint8_t* sc,
    const int8_t* x_q8,
    const float* x_scales,
    const __nv_bfloat16* residual,
    __nv_bfloat16* y,
    float alpha,
    int N,
    int K
) {
    size_t smem = (size_t)TM * K;
    static SmemOptinHighWater optin;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin, (const void*)gemv_nvfp4_w4a8_down_residual_mk_kernel<TM>, smem);
    if (orc != 0) return orc;
    unsigned grid = (unsigned)((N + kWarpsPerBlockSharesOneStagedX - 1)
        / kWarpsPerBlockSharesOneStagedX);
    gemv_nvfp4_w4a8_down_residual_mk_kernel<TM><<<grid,
                                                  dim3(kWarpsPerBlockSharesOneStagedX * kWarp),
                                                  smem, s>>>(
        wq, sc, x_q8, x_scales, residual, y, alpha, N, K);
    return (int)cudaGetLastError();
}

constexpr int kW4a8MkMaxTokensPerLaunch = 8;

int w4a8_mk_chunk_tokens_bounded_by_96k_smem(int K) {
    int by_smem = (int)((96 * 1024) / (size_t)K);
    return by_smem < kW4a8MkMaxTokensPerLaunch ? by_smem : kW4a8MkMaxTokensPerLaunch;
}

}

extern "C" int nv_kernels_gemv_nvfp4_w4a8_dual_m1(
    void* stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const int8_t* x_q8,
    const float* x_dequant_scale,
    uint16_t* y_a,
    uint16_t* y_b,
    float alpha_a,
    float alpha_b,
    int N,
    int K
) {
    if (N <= 0 || K <= 0 || (K & 31) != 0) return -1;
    size_t smem = (size_t)K;
    if (smem > 96 * 1024) return -1;
    static SmemOptinHighWater optin_dual;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin_dual, (const void*)gemv_nvfp4_w4a8_dual_m1_kernel, smem);
    if (orc != 0) return orc;
    unsigned grid = (unsigned)((2 * N + kWarpsPerBlockSharesOneStagedX - 1)
        / kWarpsPerBlockSharesOneStagedX);
    gemv_nvfp4_w4a8_dual_m1_kernel<<<grid, dim3(kWarpsPerBlockSharesOneStagedX * kWarp), smem,
                                     (cudaStream_t)stream>>>(
        wq_a, sc_a, wq_b, sc_b, x_q8, x_dequant_scale,
        reinterpret_cast<__nv_bfloat16*>(y_a),
        reinterpret_cast<__nv_bfloat16*>(y_b),
        alpha_a, alpha_b, N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_silu_mul_rowquant_i8_m1(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    int8_t* act_q8,
    float* act_dequant_scale,
    int K
) {
    if (K <= 0 || (K & 3) != 0) return -1;
    size_t smem = (size_t)K * sizeof(__nv_bfloat16);
    if (smem > 96 * 1024) return -1;
    static SmemOptinHighWater optin_silu_quant;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin_silu_quant, (const void*)silu_mul_rowquant_i8_m1_kernel, smem);
    if (orc != 0) return orc;
    silu_mul_rowquant_i8_m1_kernel<<<1, kSiluQuantBlockThreads, smem, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(up),
        act_q8, act_dequant_scale, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_silu_mul_rowquant_i8_mk_partials_len(int M, int K) {
    if (M <= 0 || K <= 0) return -1;
    int b = (K + kSiluQuantSplitThreads - 1) / kSiluQuantSplitThreads;
    if (b > kSiluQuantSplitMaxBlocksKeepsPass2PartialFanInOneSmemRound) {
        b = kSiluQuantSplitMaxBlocksKeepsPass2PartialFanInOneSmemRound;
    }
    return M * b;
}

extern "C" int nv_kernels_silu_mul_rowquant_i8_mk(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* act_staged_bf16,
    float* partial_absmax,
    int8_t* act_q8,
    float* act_dequant_scales,
    int M,
    int K
) {
    if (M <= 0 || K <= 0 || (K & 15) != 0) return -1;
    int bx = (K + kSiluQuantSplitThreads - 1) / kSiluQuantSplitThreads;
    if (bx > kSiluQuantSplitMaxBlocksKeepsPass2PartialFanInOneSmemRound) {
        bx = kSiluQuantSplitMaxBlocksKeepsPass2PartialFanInOneSmemRound;
    }
    dim3 grid((unsigned)bx, (unsigned)M);
    silu_mul_stage_bf16_partial_absmax_mk_pass1_kernel<<<grid, kSiluQuantSplitThreads, 0,
                                                         (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(up),
        reinterpret_cast<__nv_bfloat16*>(act_staged_bf16),
        partial_absmax, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    rowquant_i8_from_staged_bf16_partials_mk_pass2_kernel<<<grid, kSiluQuantSplitThreads, 0,
                                                            (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(act_staged_bf16),
        partial_absmax, bx, act_q8, act_dequant_scales, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_silu_mul_stage_partial_absmax_m1(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* act_staged_bf16,
    float* partial_absmax,
    int K
) {
    if (K <= 0 || (K & 15) != 0) return -1;
    int bx = (K + kSiluQuantSplitThreads - 1) / kSiluQuantSplitThreads;
    if (bx > kSiluQuantSplitMaxBlocksKeepsPass2PartialFanInOneSmemRound) {
        bx = kSiluQuantSplitMaxBlocksKeepsPass2PartialFanInOneSmemRound;
    }
    silu_mul_stage_bf16_partial_absmax_mk_pass1_kernel<<<dim3((unsigned)bx, 1),
                                                         kSiluQuantSplitThreads, 0,
                                                         (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(up),
        reinterpret_cast<__nv_bfloat16*>(act_staged_bf16),
        partial_absmax, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_nvfp4_w4a8_down_residual_quant_prologue_m1(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const uint16_t* act_staged_bf16,
    const float* partial_absmax,
    int num_partials,
    const uint16_t* residual,
    uint16_t* y,
    float alpha,
    int N,
    int K
) {
    if (N <= 0 || K <= 0 || (K & 31) != 0 || num_partials <= 0) return -1;
    size_t smem = (size_t)K;
    if (smem > 96 * 1024) return -1;
    static SmemOptinHighWater optin_down_qfold;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin_down_qfold,
        (const void*)gemv_nvfp4_w4a8_down_residual_quant_prologue_m1_kernel, smem);
    if (orc != 0) return orc;
    unsigned grid = (unsigned)((N + kWarpsPerBlockSharesOneStagedX - 1)
        / kWarpsPerBlockSharesOneStagedX);
    gemv_nvfp4_w4a8_down_residual_quant_prologue_m1_kernel<<<grid,
        dim3(kWarpsPerBlockSharesOneStagedX * kWarp), smem, (cudaStream_t)stream>>>(
        wq, sc,
        reinterpret_cast<const __nv_bfloat16*>(act_staged_bf16),
        partial_absmax, num_partials,
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<__nv_bfloat16*>(y),
        alpha, N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_rmsnorm_residual_writeout_rowquant_i8_m1(
    void* stream,
    const uint16_t* x,
    const uint16_t* res_in,
    const uint16_t* weight,
    uint16_t* res_out,
    int8_t* out_q8,
    float* out_dequant_scale,
    int hidden,
    float eps
) {
    if (hidden <= 0 || hidden > kNormQuantFoldMaxHiddenBoundsTheTwoU16SmemRows) return -1;
    size_t smem = (size_t)hidden * 2 * sizeof(__nv_bfloat16);
    static SmemOptinHighWater optin_norm_quant;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin_norm_quant, (const void*)rmsnorm_residual_writeout_rowquant_i8_m1_kernel, smem);
    if (orc != 0) return orc;
    rmsnorm_residual_writeout_rowquant_i8_m1_kernel<<<1, kNormQuantFoldBlockThreads, smem,
                                                      (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(res_in),
        reinterpret_cast<const __nv_bfloat16*>(weight),
        reinterpret_cast<__nv_bfloat16*>(res_out),
        out_q8, out_dequant_scale, hidden, eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_nvfp4_w4a8_dual_mk(
    void* stream,
    const uint8_t* wq_a,
    const uint8_t* sc_a,
    const uint8_t* wq_b,
    const uint8_t* sc_b,
    const int8_t* x_q8,
    const float* x_dequant_scales,
    uint16_t* y_a,
    uint16_t* y_b,
    float alpha_a,
    float alpha_b,
    int M,
    int N,
    int K
) {
    if (M <= 0 || N <= 0 || K <= 0 || (K & 31) != 0) return -1;
    int chunk_max = w4a8_mk_chunk_tokens_bounded_by_96k_smem(K);
    if (chunk_max < 1) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    for (int j0 = 0; j0 < M; j0 += chunk_max) {
        int tm = M - j0 < chunk_max ? M - j0 : chunk_max;
        const int8_t* xj = x_q8 + (size_t)j0 * K;
        const float* sj = x_dequant_scales + j0;
        __nv_bfloat16* ya = reinterpret_cast<__nv_bfloat16*>(y_a) + (size_t)j0 * N;
        __nv_bfloat16* yb = reinterpret_cast<__nv_bfloat16*>(y_b) + (size_t)j0 * N;
        int rc;
        switch (tm) {
            case 1: rc = launch_dual_mk<1>(s, wq_a, sc_a, wq_b, sc_b, xj, sj, ya, yb, alpha_a, alpha_b, N, K); break;
            case 2: rc = launch_dual_mk<2>(s, wq_a, sc_a, wq_b, sc_b, xj, sj, ya, yb, alpha_a, alpha_b, N, K); break;
            case 3: rc = launch_dual_mk<3>(s, wq_a, sc_a, wq_b, sc_b, xj, sj, ya, yb, alpha_a, alpha_b, N, K); break;
            case 4: rc = launch_dual_mk<4>(s, wq_a, sc_a, wq_b, sc_b, xj, sj, ya, yb, alpha_a, alpha_b, N, K); break;
            case 5: rc = launch_dual_mk<5>(s, wq_a, sc_a, wq_b, sc_b, xj, sj, ya, yb, alpha_a, alpha_b, N, K); break;
            case 6: rc = launch_dual_mk<6>(s, wq_a, sc_a, wq_b, sc_b, xj, sj, ya, yb, alpha_a, alpha_b, N, K); break;
            case 7: rc = launch_dual_mk<7>(s, wq_a, sc_a, wq_b, sc_b, xj, sj, ya, yb, alpha_a, alpha_b, N, K); break;
            case 8: rc = launch_dual_mk<8>(s, wq_a, sc_a, wq_b, sc_b, xj, sj, ya, yb, alpha_a, alpha_b, N, K); break;
            default: rc = -1;
        }
        if (rc != 0) return rc;
    }
    return 0;
}

extern "C" int nv_kernels_gemv_nvfp4_w4a8_down_residual_mk(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const int8_t* x_q8,
    const float* x_dequant_scales,
    const uint16_t* residual,
    uint16_t* y,
    float alpha,
    int M,
    int N,
    int K
) {
    if (M <= 0 || N <= 0 || K <= 0 || (K & 31) != 0) return -1;
    int chunk_max = w4a8_mk_chunk_tokens_bounded_by_96k_smem(K);
    if (chunk_max < 1) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    for (int j0 = 0; j0 < M; j0 += chunk_max) {
        int tm = M - j0 < chunk_max ? M - j0 : chunk_max;
        const int8_t* xj = x_q8 + (size_t)j0 * K;
        const float* sj = x_dequant_scales + j0;
        const __nv_bfloat16* rj = residual == nullptr
            ? nullptr
            : reinterpret_cast<const __nv_bfloat16*>(residual) + (size_t)j0 * N;
        __nv_bfloat16* yj = reinterpret_cast<__nv_bfloat16*>(y) + (size_t)j0 * N;
        int rc;
        switch (tm) {
            case 1: rc = launch_down_mk<1>(s, wq, sc, xj, sj, rj, yj, alpha, N, K); break;
            case 2: rc = launch_down_mk<2>(s, wq, sc, xj, sj, rj, yj, alpha, N, K); break;
            case 3: rc = launch_down_mk<3>(s, wq, sc, xj, sj, rj, yj, alpha, N, K); break;
            case 4: rc = launch_down_mk<4>(s, wq, sc, xj, sj, rj, yj, alpha, N, K); break;
            case 5: rc = launch_down_mk<5>(s, wq, sc, xj, sj, rj, yj, alpha, N, K); break;
            case 6: rc = launch_down_mk<6>(s, wq, sc, xj, sj, rj, yj, alpha, N, K); break;
            case 7: rc = launch_down_mk<7>(s, wq, sc, xj, sj, rj, yj, alpha, N, K); break;
            case 8: rc = launch_down_mk<8>(s, wq, sc, xj, sj, rj, yj, alpha, N, K); break;
            default: rc = -1;
        }
        if (rc != 0) return rc;
    }
    return 0;
}

extern "C" int nv_kernels_gemv_nvfp4_w4a8_down_residual_m1(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const int8_t* x_q8,
    const float* x_dequant_scale,
    const uint16_t* residual,
    uint16_t* y,
    float alpha,
    int N,
    int K
) {
    if (N <= 0 || K <= 0 || (K & 31) != 0) return -1;
    size_t smem = (size_t)K;
    if (smem > 96 * 1024) return -1;
    static SmemOptinHighWater optin_down;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin_down, (const void*)gemv_nvfp4_w4a8_down_residual_m1_kernel<false>, smem);
    if (orc != 0) return orc;
    unsigned grid = (unsigned)((N + kWarpsPerBlockSharesOneStagedX - 1)
        / kWarpsPerBlockSharesOneStagedX);
    gemv_nvfp4_w4a8_down_residual_m1_kernel<false><<<grid,
                                              dim3(kWarpsPerBlockSharesOneStagedX * kWarp),
                                              smem, (cudaStream_t)stream>>>(
        wq, sc, x_q8, x_dequant_scale,
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<__nv_bfloat16*>(y),
        alpha, nullptr, 0.0f, N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_nvfp4_w4a8_down_residual_m1_rstd_emit(
    void* stream,
    const uint8_t* wq,
    const uint8_t* sc,
    const int8_t* x_q8,
    const float* x_dequant_scale,
    const uint16_t* residual,
    uint16_t* y,
    float alpha,
    float* rstd_ssq_count_pack,
    float rstd_eps,
    int N,
    int K
) {
    if (N <= 0 || K <= 0 || (K & 31) != 0) return -1;
    if (rstd_ssq_count_pack == nullptr) return -1;
    size_t smem = (size_t)K;
    if (smem > 96 * 1024) return -1;
    static SmemOptinHighWater optin_down_rstd;
    int orc = raise_dynamic_smem_optin_above_48k(
        optin_down_rstd, (const void*)gemv_nvfp4_w4a8_down_residual_m1_kernel<true>, smem);
    if (orc != 0) return orc;
    unsigned grid = (unsigned)((N + kWarpsPerBlockSharesOneStagedX - 1)
        / kWarpsPerBlockSharesOneStagedX);
    gemv_nvfp4_w4a8_down_residual_m1_kernel<true><<<grid,
                                              dim3(kWarpsPerBlockSharesOneStagedX * kWarp),
                                              smem, (cudaStream_t)stream>>>(
        wq, sc, x_q8, x_dequant_scale,
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<__nv_bfloat16*>(y),
        alpha, rstd_ssq_count_pack, rstd_eps, N, K);
    return (int)cudaGetLastError();
}
