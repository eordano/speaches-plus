
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kWarpSize = 32;

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

template <int kWarps, int kSplit, bool kStream, int kMaxV>
__global__ void gemv_w4a16_m1_proto_kernel(
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    constexpr int kBlockDim = kWarps * kWarpSize;
    constexpr int kRows = kWarps / kSplit;
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
    int n = blockIdx.x * kRows + rw;

    __shared__ float partials[kSplit > 1 ? kWarps : 1];

    if (n < N) {
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
        if (kSplit == 1) {
            if (lane == 0) y[n] = __float2bfloat16(acc);
        } else if (lane == 0) {
            partials[warp] = acc;
        }
    }

    if (kSplit > 1) {
        __syncthreads();
        if (n < N && part == 0 && lane == 0) {
            float sum = 0.0f;
            #pragma unroll
            for (int s = 0; s < kSplit; ++s) {
                sum += partials[rw * kSplit + s];
            }
            y[n] = __float2bfloat16(sum);
        }
    }
}

template <int kWarps, int kSplit, bool kStream>
int launch_proto(
    cudaStream_t s,
    const uint32_t* packed,
    const __nv_bfloat16* scale,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int N,
    int K
) {
    constexpr int kRows = kWarps / kSplit;
    int Kv = K >> 5;
    int needed = (Kv + kWarpSize * kSplit - 1) / (kWarpSize * kSplit);
    dim3 grid((unsigned)((N + kRows - 1) / kRows));
    dim3 block(kWarps * kWarpSize);
    size_t smem = (size_t)Kv * 33 * sizeof(float);

    #define NV_PROTO_CASE(MV)                                              \
        if (needed <= MV) {                                                \
            gemv_w4a16_m1_proto_kernel<kWarps, kSplit, kStream, MV>        \
                <<<grid, block, smem, s>>>(packed, scale, x, y, N, K);     \
            return (int)cudaGetLastError();                                \
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
    cudaStream_t s = (cudaStream_t)stream;
    const __nv_bfloat16* sc = reinterpret_cast<const __nv_bfloat16*>(scale);
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(x);
    __nv_bfloat16* yb = reinterpret_cast<__nv_bfloat16*>(y);
    switch (variant) {
        case 0: return launch_proto<8, 1, true>(s, packed, sc, xb, yb, N, K);
        case 1: return launch_proto<8, 1, false>(s, packed, sc, xb, yb, N, K);
        case 2: return launch_proto<4, 1, true>(s, packed, sc, xb, yb, N, K);
        case 3: return launch_proto<16, 1, true>(s, packed, sc, xb, yb, N, K);
        case 4: return launch_proto<8, 2, true>(s, packed, sc, xb, yb, N, K);
        case 5: return launch_proto<16, 2, true>(s, packed, sc, xb, yb, N, K);
        case 6: return launch_proto<8, 4, true>(s, packed, sc, xb, yb, N, K);
        case 7: return launch_proto<16, 4, true>(s, packed, sc, xb, yb, N, K);
        default: return -1;
    }
}
