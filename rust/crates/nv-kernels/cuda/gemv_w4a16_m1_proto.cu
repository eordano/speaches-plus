
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "rmsnorm_add_scale_row.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kEpiNone = 0;
constexpr int kEpiGegluPairsGateRowNWithUpRowNPlusInterInOneBlock = 1;
constexpr int kEpiPostNormTheLastBlockToFinishRunsTheRowNormOnTheWholeOutput = 2;
constexpr int kNormBlock128MatchesRmsnormAddScaleBf16 = 128;

struct ProtoEpilogue {
    const __nv_bfloat16* res;
    const __nv_bfloat16* w;
    __nv_bfloat16* h_out;
    float* rstd_out;
    const __nv_bfloat16* next_w;
    __nv_bfloat16* normed_out;
    float eps;
    float scale;
    float eps_next;
    unsigned* counter;
};

__device__ __forceinline__ float gelu_tanh_mul_scalar(float gate, float up) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCubicCoeff = 0.044715f;
    float g3 = gate * gate * gate;
    float inner = kSqrt2OverPi * (gate + kCubicCoeff * g3);
    float t = tanhf(inner);
    float gelu = 0.5f * gate * (1.0f + t);
    return gelu * up;
}

__device__ __forceinline__ float dot32(const uint4& pw, const float* xp) {
    float a = 0.0f;
    const uint32_t w[4] = {pw.x, pw.y, pw.z, pw.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t lo = w[j] & 0x0F0F0F0Fu;
        uint32_t hi = (w[j] >> 4) & 0x0F0F0F0Fu;
        const float* x8 = xp + 8 * j;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float fe = __uint_as_float(__byte_perm(lo, 0x4B000000u, 0x7650u + i))
                       - 8388616.0f;
            float fo = __uint_as_float(__byte_perm(hi, 0x4B000000u, 0x7650u + i))
                       - 8388616.0f;
            a = fmaf(fe, x8[2 * i], a);
            a = fmaf(fo, x8[2 * i + 1], a);
        }
    }
    return a;
}

template <int kWarps, int kSplit, bool kStream, int kMaxV, int kEpi>
__global__ void gemv_w4a16_m1_proto_kernel(
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    ProtoEpilogue epi
) {
    constexpr int kBlockDim = kWarps * kWarpSize;
    constexpr int kRows = kWarps / kSplit;
    constexpr bool kGeglu = kEpi == kEpiGegluPairsGateRowNWithUpRowNPlusInterInOneBlock;
    constexpr int kHalf = kGeglu ? kRows / 2 : kRows;
    static_assert(kEpi != kEpiPostNormTheLastBlockToFinishRunsTheRowNormOnTheWholeOutput ||
                      kBlockDim >= kNormBlock128MatchesRmsnormAddScaleBf16,
                  "the post-norm tail needs the norm block inside the launch block");
    extern __shared__ float xs[];

    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    int K8 = K >> 3;
    for (int t = threadIdx.x; t < K8; t += kBlockDim) {
        uint4 raw = __ldg(&x4[t]);
        const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
        int k0 = t << 3;
        float* dst = xs + (k0 >> 5) * 33 + (k0 & 31);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 f = __bfloat1622float2(p2[i]);
            dst[2 * i] = f.x;
            dst[2 * i + 1] = f.y;
        }
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x >> 5;
    int rw = warp / kSplit;
    int part = warp % kSplit;
    const int inter = kGeglu ? (N >> 1) : N;
    const int pair = blockIdx.x * kHalf + (kGeglu ? (rw % kHalf) : rw);
    const int n = kGeglu ? (rw < kHalf ? pair : inter + pair) : pair;
    const bool row_live = kGeglu ? (pair < inter) : (n < N);

    __shared__ float partials[(kSplit > 1 || kGeglu) ? kWarps : 1];

    if (row_live) {
        int Kv = K >> 5;
        const uint4* w_row =
            reinterpret_cast<const uint4*>(packed + (size_t)n * (K >> 3));
        const __nv_bfloat16* s_row = scale + (size_t)n * Kv;

        constexpr int kStride = kWarpSize * kSplit;
        int v0 = part * kWarpSize + lane;

        uint4 pw[kMaxV];
        float sc[kMaxV];
        #pragma unroll
        for (int j = 0; j < kMaxV; ++j) {
            int v = v0 + j * kStride;
            if (v < Kv) {
                pw[j] = kStream ? __ldcs(&w_row[v]) : __ldg(&w_row[v]);
                sc[j] = __bfloat162float(__ldg(&s_row[v]));
            }
        }

        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < kMaxV; ++j) {
            int v = v0 + j * kStride;
            if (v < Kv) {
                acc = fmaf(sc[j], dot32(pw[j], xs + v * 33), acc);
            }
        }

        #pragma unroll
        for (int o = kWarpSize / 2; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(0xffffffffu, acc, o);
        }
        if (kGeglu) {
            if (lane == 0) partials[warp] = acc;
        } else if (kSplit == 1) {
            if (lane == 0) y[n] = __float2bfloat16(acc);
        } else if (lane == 0) {
            partials[warp] = acc;
        }
    }

    if (kGeglu) {
        __syncthreads();
        if (row_live && rw < kHalf && lane == 0) {
            float g = __bfloat162float(__float2bfloat16(partials[rw]));
            float u = __bfloat162float(__float2bfloat16(partials[rw + kHalf]));
            y[pair] = __float2bfloat16(gelu_tanh_mul_scalar(g, u));
        }
    } else if (kSplit > 1) {
        __syncthreads();
        if (row_live && part == 0 && lane == 0) {
            float sum = 0.0f;
            #pragma unroll
            for (int s = 0; s < kSplit; ++s) {
                sum += partials[rw * kSplit + s];
            }
            y[n] = __float2bfloat16(sum);
        }
    }

    if constexpr (kEpi == kEpiPostNormTheLastBlockToFinishRunsTheRowNormOnTheWholeOutput) {
        __shared__ unsigned s_ticket;
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) s_ticket = atomicAdd(epi.counter, 1u);
        __syncthreads();
        if (s_ticket == gridDim.x - 1) {
            __threadfence();
            nvk_norm_row::rmsnorm_add_scale_row<kNormBlock128MatchesRmsnormAddScaleBf16, kBlockDim>(
                y, epi.w, epi.res, epi.h_out, epi.rstd_out, epi.next_w, epi.normed_out, xs,
                xs + nvk_norm_row::kRmsnormRowReduceScratchFloats, N, epi.eps, epi.scale,
                epi.eps_next);
            __syncthreads();
            if (threadIdx.x == 0) *epi.counter = 0u;
        }
    }
}

template <int kWarps, int kSplit, bool kStream, int kEpi>
int launch_proto(
    cudaStream_t s,
    const uint32_t* packed,
    const __nv_bfloat16* scale,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int N,
    int K,
    const ProtoEpilogue& epi
) {
    constexpr int kRows = kWarps / kSplit;
    constexpr bool kGeglu = kEpi == kEpiGegluPairsGateRowNWithUpRowNPlusInterInOneBlock;
    constexpr int kHalf = kGeglu ? kRows / 2 : kRows;
    if constexpr (kGeglu && (kSplit != 1 || kRows % 2 != 0)) {
        return -1;
    } else {
    int Kv = K >> 5;
    int needed = (Kv + kWarpSize * kSplit - 1) / (kWarpSize * kSplit);
    int rows_out = kGeglu ? (N >> 1) : N;
    dim3 grid((unsigned)((rows_out + kHalf - 1) / kHalf));
    dim3 block(kWarps * kWarpSize);
    size_t smem = (size_t)Kv * 33 * sizeof(float);
    if (kEpi == kEpiPostNormTheLastBlockToFinishRunsTheRowNormOnTheWholeOutput) {
        size_t tail_floats = nvk_norm_row::kRmsnormRowReduceScratchFloats +
                             (epi.normed_out != nullptr ? (size_t)N : 0);
        if (smem < tail_floats * sizeof(float)) smem = tail_floats * sizeof(float);
    }
    #define NV_PROTO_CASE(MV)                                                     \
        if (needed <= MV) {                                                       \
            static bool the_runtime_charges_these_kernels_8kb_of_static_smem_beyond_the_elf_so_the_dynamic_limit_is_raised_once_##MV = false; \
            if (!the_runtime_charges_these_kernels_8kb_of_static_smem_beyond_the_elf_so_the_dynamic_limit_is_raised_once_##MV) { \
                cudaFuncAttributes fa0;                                           \
                cudaError_t ga = cudaFuncGetAttributes(                           \
                    &fa0, gemv_w4a16_m1_proto_kernel<kWarps, kSplit, kStream, MV, kEpi>); \
                if (ga == cudaSuccess && smem + fa0.sharedSizeBytes > 48 * 1024) { \
                    cudaFuncSetAttribute(                                         \
                        gemv_w4a16_m1_proto_kernel<kWarps, kSplit, kStream, MV, kEpi>, \
                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);  \
                }                                                                 \
                the_runtime_charges_these_kernels_8kb_of_static_smem_beyond_the_elf_so_the_dynamic_limit_is_raised_once_##MV = true; \
            }                                                                     \
            gemv_w4a16_m1_proto_kernel<kWarps, kSplit, kStream, MV, kEpi>         \
                <<<grid, block, smem, s>>>(packed, scale, x, y, N, K, epi);       \
            return (int)cudaGetLastError();                                       \
        }
    NV_PROTO_CASE(1)
    NV_PROTO_CASE(2)
    NV_PROTO_CASE(3)
    NV_PROTO_CASE(5)
    NV_PROTO_CASE(10)
    #undef NV_PROTO_CASE
    return -2;
    }
}

template <int kEpi>
int dispatch_variant(
    cudaStream_t s,
    const uint32_t* packed,
    const __nv_bfloat16* sc,
    const __nv_bfloat16* xb,
    __nv_bfloat16* yb,
    int N,
    int K,
    int variant,
    const ProtoEpilogue& epi
) {
    switch (variant) {
        case 0: return launch_proto<8, 1, true, kEpi>(s, packed, sc, xb, yb, N, K, epi);
        case 1: return launch_proto<8, 1, false, kEpi>(s, packed, sc, xb, yb, N, K, epi);
        case 2: return launch_proto<4, 1, true, kEpi>(s, packed, sc, xb, yb, N, K, epi);
        case 3: return launch_proto<16, 1, true, kEpi>(s, packed, sc, xb, yb, N, K, epi);
        case 4: return launch_proto<8, 2, true, kEpi>(s, packed, sc, xb, yb, N, K, epi);
        case 5: return launch_proto<16, 2, true, kEpi>(s, packed, sc, xb, yb, N, K, epi);
        case 6: return launch_proto<8, 4, true, kEpi>(s, packed, sc, xb, yb, N, K, epi);
        case 7: return launch_proto<16, 4, true, kEpi>(s, packed, sc, xb, yb, N, K, epi);
        default: return -1;
    }
}

}

extern "C" int nv_kernels_gemv_w4a16_m1_proto(
    void* stream,
    const uint32_t* packed,
    const uint16_t* scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int GS,
    int variant
) {
    if (N <= 0 || K <= 0) return 0;
    if (GS != 32 || (K & 31) != 0) return -1;
    ProtoEpilogue epi = {};
    return dispatch_variant<kEpiNone>(
        (cudaStream_t)stream, packed, reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y), N, K,
        variant, epi);
}

extern "C" int nv_kernels_gemv_w4a16_m1_proto_geglu(
    void* stream,
    const uint32_t* packed,
    const uint16_t* scale,
    const uint16_t* x,
    uint16_t* y,
    int N_gate_rows_then_up_rows,
    int K,
    int GS,
    int variant
) {
    if (N_gate_rows_then_up_rows <= 0 || K <= 0) return 0;
    if (GS != 32 || (K & 31) != 0 || (N_gate_rows_then_up_rows & 1) != 0) return -1;
    if (variant != 0 && variant != 1 && variant != 2 && variant != 3) return -1;
    ProtoEpilogue epi = {};
    return dispatch_variant<kEpiGegluPairsGateRowNWithUpRowNPlusInterInOneBlock>(
        (cudaStream_t)stream, packed, reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y),
        N_gate_rows_then_up_rows, K, variant, epi);
}

extern "C" int nv_kernels_gemv_w4a16_m1_proto_post_norm(
    void* stream,
    const uint32_t* packed,
    const uint16_t* scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int GS,
    int variant,
    const uint16_t* res,
    const uint16_t* w,
    uint16_t* h_out,
    float* rstd_out,
    const uint16_t* next_w,
    uint16_t* normed_out,
    float eps,
    float res_scale,
    float eps_next,
    unsigned* counter
) {
    if (N <= 0 || K <= 0) return 0;
    if (GS != 32 || (K & 31) != 0) return -1;
    if (res == nullptr || w == nullptr || h_out == nullptr || counter == nullptr) return -1;
    if (normed_out != nullptr && next_w == nullptr) return -1;
    ProtoEpilogue epi = {
        reinterpret_cast<const __nv_bfloat16*>(res),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<__nv_bfloat16*>(h_out),
        rstd_out,
        reinterpret_cast<const __nv_bfloat16*>(next_w),
        reinterpret_cast<__nv_bfloat16*>(normed_out),
        eps,
        res_scale,
        eps_next,
        counter,
    };
    return dispatch_variant<kEpiPostNormTheLastBlockToFinishRunsTheRowNormOnTheWholeOutput>(
        (cudaStream_t)stream, packed, reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(x), reinterpret_cast<__nv_bfloat16*>(y), N, K,
        variant, epi);
}
