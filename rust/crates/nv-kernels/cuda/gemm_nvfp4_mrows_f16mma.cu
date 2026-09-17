#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace nvfp4mrows {

constexpr int kWarp = 32;
constexpr int kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams = 4;
constexpr int kThreads = kWarp * kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams;
constexpr int kTileRows = 16;
constexpr int kTokensMax8IsOneN8Tile = 8;
constexpr int kNvfp4Block = 16;
constexpr int kLaneBytes32Is64ElementsSoFourLanesReadOne128ByteLinePerRowPerIteration = 32;
constexpr int kLaneElems = 2 * kLaneBytes32Is64ElementsSoFourLanesReadOne128ByteLinePerRowPerIteration;
constexpr int kWq = kLaneBytes32Is64ElementsSoFourLanesReadOne128ByteLinePerRowPerIteration / 16;
constexpr int kXq = kLaneElems * 2 / 16;
constexpr int kIterElems = kLaneElems * 4;
constexpr int kStepsPerIter = kLaneElems / 4;
constexpr int kKMultipleOf1024SoEveryWarpQuarterIsWholeIterations = kIterElems * kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams;
constexpr unsigned kE2m1MagToHalfHighByte0to3 = 0x3E3C3800u;
constexpr unsigned kE2m1MagToHalfHighByte4to7 = 0x46444240u;
#define NV_UE4M3_SUBNORMAL_STEP  0.001953125f

__device__ __forceinline__ float decode_ue4m3_dev(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
}

__device__ __forceinline__ unsigned bf16x2_to_half2_bits(unsigned packed) {
    __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&packed);
    float2 f = __bfloat1622float2(b);
    __half2 h = __floats2half2_rn(f.x, f.y);
    return *reinterpret_cast<unsigned*>(&h);
}

__device__ __forceinline__ unsigned scale_byte_to_half2_bits(unsigned b) {
    const float v = decode_ue4m3_dev((uint8_t)b);
    __half2 h = __floats2half2_rn(v, v);
    return *reinterpret_cast<unsigned*>(&h);
}

__device__ __forceinline__ void nibbles4_to_two_half2_bits(unsigned c, unsigned& lo, unsigned& hi) {
    const unsigned v = __byte_perm(kE2m1MagToHalfHighByte0to3, kE2m1MagToHalfHighByte4to7, c & 0x7777u);
    lo = __byte_perm(v, 0u, 0x1404u) | ((c & 0x0008u) << 12) | ((c & 0x0080u) << 24);
    hi = __byte_perm(v, 0u, 0x3424u) | ((c & 0x0800u) << 4) | ((c & 0x8000u) << 16);
}

__device__ __forceinline__ unsigned hmul2_bits(unsigned a, unsigned b) {
    __half2 ha = *reinterpret_cast<__half2*>(&a);
    __half2 hb = *reinterpret_cast<__half2*>(&b);
    __half2 r = __hmul2(ha, hb);
    return *reinterpret_cast<unsigned*>(&r);
}

__device__ __forceinline__ void mma_m16n8k16_f16(
    float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ int scale_row_base(int r, int k_tiles) {
    return ((r >> 7) * k_tiles) * 512 + (r & 31) * 16 + ((r >> 5) & 3) * 4;
}

__global__ void __launch_bounds__(kThreads) gemm_nvfp4_mrows_f16mma_kernel(
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
    int N_b,
    int K,
    int M
) {
    __shared__ float part[kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams][kTileRows][kTokensMax8IsOneN8Tile];
    const int tiles_a = (N + kTileRows - 1) / kTileRows;
    const bool second = (int)blockIdx.x >= tiles_a;
    const uint8_t* wq = second ? wq_b : wq_a;
    const uint8_t* sc = second ? sc_b : sc_a;
    const int n_rows = second ? N_b : N;
    const int tile = second ? (int)blockIdx.x - tiles_a : (int)blockIdx.x;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int gid = lane >> 2;
    const int tg = lane & 3;
    const int r0 = tile * kTileRows + gid;
    const int r1 = r0 + 8;
    const bool live0 = r0 < n_rows;
    const bool live1 = r1 < n_rows;
    const bool tok = gid < M;
    const int kq = K / kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams;
    const int k0 = warp * kq + tg * kLaneElems;
    const int k_tiles = (K / kNvfp4Block + 3) >> 2;
    const uint4* w0 = reinterpret_cast<const uint4*>(wq + (size_t)(live0 ? r0 : 0) * (K >> 1) + (k0 >> 1));
    const uint4* w1 = reinterpret_cast<const uint4*>(wq + (size_t)(live1 ? r1 : 0) * (K >> 1) + (k0 >> 1));
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)(tok ? gid : 0) * K + k0);
    const uint8_t* s0 = sc + scale_row_base(live0 ? r0 : 0, k_tiles);
    const uint8_t* s1 = sc + scale_row_base(live1 ? r1 : 0, k_tiles);

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint4 wa[kWq];
    uint4 wb[kWq];
    unsigned sa = 0u;
    unsigned sb = 0u;
    auto load = [&](int p, uint4 (&a)[kWq], uint4 (&b)[kWq], unsigned& xa, unsigned& xb) {
        const int pw = p / 32;
        #pragma unroll
        for (int i = 0; i < kWq; ++i) {
            a[i] = live0 ? __ldcs(w0 + pw + i) : make_uint4(0u, 0u, 0u, 0u);
            b[i] = live1 ? __ldcs(w1 + pw + i) : make_uint4(0u, 0u, 0u, 0u);
        }
        const int so = (((k0 + p) / kNvfp4Block) >> 2) * 512;
        xa = live0 ? __ldg(reinterpret_cast<const unsigned*>(s0 + so)) : 0u;
        xb = live1 ? __ldg(reinterpret_cast<const unsigned*>(s1 + so)) : 0u;
    };
    load(0, wa, wb, sa, sb);
    for (int p = 0; p < kq; p += kIterElems) {
        uint4 na[kWq];
        uint4 nb[kWq];
        unsigned nsa;
        unsigned nsb;
        if (p + kIterElems < kq) {
            load(p + kIterElems, na, nb, nsa, nsb);
        } else {
            #pragma unroll
            for (int i = 0; i < kWq; ++i) { na[i] = make_uint4(0u, 0u, 0u, 0u); nb[i] = na[i]; }
            nsa = 0u;
            nsb = 0u;
        }
        uint4 xv[kXq];
        #pragma unroll
        for (int i = 0; i < kXq; ++i) xv[i] = tok ? __ldg(xr + p / 8 + i) : make_uint4(0u, 0u, 0u, 0u);
        const unsigned* wa32 = reinterpret_cast<const unsigned*>(wa);
        const unsigned* wb32 = reinterpret_cast<const unsigned*>(wb);
        const unsigned* xv32 = reinterpret_cast<const unsigned*>(xv);
        #pragma unroll
        for (int st = 0; st < kStepsPerIter; ++st) {
            const unsigned ca = (wa32[st >> 1] >> (16 * (st & 1))) & 0xffffu;
            const unsigned cb = (wb32[st >> 1] >> (16 * (st & 1))) & 0xffffu;
            const unsigned fa = scale_byte_to_half2_bits((sa >> (8 * (st >> 2))) & 0xffu);
            const unsigned fb = scale_byte_to_half2_bits((sb >> (8 * (st >> 2))) & 0xffu);
            unsigned a0, a2, a1, a3;
            nibbles4_to_two_half2_bits(ca, a0, a2);
            nibbles4_to_two_half2_bits(cb, a1, a3);
            mma_m16n8k16_f16(
                c,
                hmul2_bits(a0, fa),
                hmul2_bits(a1, fb),
                hmul2_bits(a2, fa),
                hmul2_bits(a3, fb),
                bf16x2_to_half2_bits(xv32[2 * st]),
                bf16x2_to_half2_bits(xv32[2 * st + 1]));
        }
        #pragma unroll
        for (int i = 0; i < kWq; ++i) { wa[i] = na[i]; wb[i] = nb[i]; }
        sa = nsa;
        sb = nsb;
    }
    part[warp][gid][tg * 2] = c[0];
    part[warp][gid][tg * 2 + 1] = c[1];
    part[warp][gid + 8][tg * 2] = c[2];
    part[warp][gid + 8][tg * 2 + 1] = c[3];
    __syncthreads();
    if (warp == 0) {
        const float alpha = second ? alpha_b : alpha_a;
        __nv_bfloat16* y = second ? y_b : y_a;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = lane + i * kWarp;
            const int row = idx >> 3;
            const int n = idx & 7;
            const int r = tile * kTileRows + row;
            if (r < n_rows && n < M) {
                float a = 0.0f;
                #pragma unroll
                for (int w = 0; w < kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams; ++w)
                    a += part[w][row][n];
                y[(size_t)n * n_rows + r] = __float2bfloat16(a * alpha);
            }
        }
    }
}

int launch(
    void* stream, const uint8_t* wq_a, const uint8_t* sc_a, const uint8_t* wq_b, const uint8_t* sc_b,
    const uint16_t* x, uint16_t* y_a, uint16_t* y_b, float alpha_a, float alpha_b, int M, int N, int N_b, int K
) {
    if (M <= 0 || N <= 0 || N_b < 0 || K <= 0) return -1;
    if (M > kTokensMax8IsOneN8Tile) return -2;
    if ((K % kKMultipleOf1024SoEveryWarpQuarterIsWholeIterations) != 0) return -3;
    if (N_b > 0 && (wq_b == nullptr || sc_b == nullptr || y_b == nullptr)) return -4;
    const unsigned grid = (unsigned)((N + kTileRows - 1) / kTileRows + (N_b + kTileRows - 1) / kTileRows);
    gemm_nvfp4_mrows_f16mma_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
        wq_a, sc_a, wq_b, sc_b, reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y_a), reinterpret_cast<__nv_bfloat16*>(y_b),
        alpha_a, alpha_b, N, N_b, K, M);
    return (int)cudaGetLastError();
}

}

extern "C" int nv_kernels_gemm_nvfp4_mrows_f16mma(
    void* stream, const uint8_t* wq, const uint8_t* sc, const uint16_t* x, uint16_t* y, float alpha, int M, int N, int K
) {
    return nvfp4mrows::launch(stream, wq, sc, nullptr, nullptr, x, y, nullptr, alpha, 0.0f, M, N, 0, K);
}

extern "C" int nv_kernels_gemm_nvfp4_mrows_f16mma_dual(
    void* stream, const uint8_t* wq_a, const uint8_t* sc_a, const uint8_t* wq_b, const uint8_t* sc_b,
    const uint16_t* x, uint16_t* y_a, uint16_t* y_b, float alpha_a, float alpha_b, int M, int N, int N_b, int K
) {
    return nvfp4mrows::launch(stream, wq_a, sc_a, wq_b, sc_b, x, y_a, y_b, alpha_a, alpha_b, M, N, N_b, K);
}
