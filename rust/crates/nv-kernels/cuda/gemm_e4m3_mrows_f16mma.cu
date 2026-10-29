#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace e4m3mrows {

constexpr int kWarp = 32;
constexpr int kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams = 4;
constexpr int kThreads = kWarp * kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams;
constexpr int kTileRows = 16;
constexpr int kTokensMax8IsOneN8Tile = 8;
constexpr int kStepK32 = 32;
constexpr int kKMultipleOf128SoEveryWarpQuarterIsWholeK32Steps = kStepK32 * kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams;

__device__ __forceinline__ unsigned fp8x2_to_half2_bits(unsigned packed16) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed16, __NV_E4M3);
    return *reinterpret_cast<unsigned*>(&hr);
}

__device__ __forceinline__ unsigned bf16x2_to_half2_bits(unsigned packed) {
    __nv_bfloat162 b = *reinterpret_cast<__nv_bfloat162*>(&packed);
    float2 f = __bfloat1622float2(b);
    __half2 h = __floats2half2_rn(f.x, f.y);
    return *reinterpret_cast<unsigned*>(&h);
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

template <int S>
__global__ void __launch_bounds__(kThreads) gemm_e4m3_mrows_f16mma_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    int M
) {
    constexpr int kIter = kStepK32 * S;
    constexpr int kLaneBytes = 8 * S;
    constexpr int kWq = kLaneBytes / 16;
    constexpr int kXq = kLaneBytes * 2 / 16;
    __shared__ float part[kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams][kTileRows][kTokensMax8IsOneN8Tile];
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int gid = lane >> 2;
    const int tg = lane & 3;
    const int r0 = blockIdx.x * kTileRows + gid;
    const int r1 = r0 + 8;
    const bool live0 = r0 < N;
    const bool live1 = r1 < N;
    const bool tok = gid < M;
    const int kq = K / kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams;
    const int k0 = warp * kq;
    const uint4* w0 = reinterpret_cast<const uint4*>(wq + (size_t)(live0 ? r0 : 0) * K + k0 + tg * kLaneBytes);
    const uint4* w1 = reinterpret_cast<const uint4*>(wq + (size_t)(live1 ? r1 : 0) * K + k0 + tg * kLaneBytes);
    const uint4* xr = reinterpret_cast<const uint4*>(x + (size_t)(tok ? gid : 0) * K + k0 + tg * kLaneBytes);

    float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint4 wa[kWq];
    uint4 wb[kWq];
    uint4 xv[kXq];
    auto load = [&](int p, uint4 (&a)[kWq], uint4 (&b)[kWq], uint4 (&xx)[kXq]) {
        const int pw = p / 16;
        const int px = p / 8;
        #pragma unroll
        for (int i = 0; i < kWq; ++i) {
            a[i] = live0 ? __ldcs(w0 + pw + i) : make_uint4(0u, 0u, 0u, 0u);
            b[i] = live1 ? __ldcs(w1 + pw + i) : make_uint4(0u, 0u, 0u, 0u);
        }
        #pragma unroll
        for (int i = 0; i < kXq; ++i) xx[i] = tok ? __ldg(xr + px + i) : make_uint4(0u, 0u, 0u, 0u);
    };
    load(0, wa, wb, xv);
    for (int p = 0; p < kq; p += kIter) {
        uint4 na[kWq];
        uint4 nb[kWq];
        uint4 nx[kXq];
        if (p + kIter < kq) {
            load(p + kIter, na, nb, nx);
        } else {
            #pragma unroll
            for (int i = 0; i < kWq; ++i) { na[i] = make_uint4(0u, 0u, 0u, 0u); nb[i] = na[i]; }
            #pragma unroll
            for (int i = 0; i < kXq; ++i) nx[i] = make_uint4(0u, 0u, 0u, 0u);
        }
        const unsigned* wa32 = reinterpret_cast<const unsigned*>(wa);
        const unsigned* wb32 = reinterpret_cast<const unsigned*>(wb);
        const unsigned* xv32 = reinterpret_cast<const unsigned*>(xv);
        #pragma unroll
        for (int st = 0; st < 2 * S; ++st) {
            mma_m16n8k16_f16(
                c,
                fp8x2_to_half2_bits(wa32[st] & 0xffffu),
                fp8x2_to_half2_bits(wb32[st] & 0xffffu),
                fp8x2_to_half2_bits(wa32[st] >> 16),
                fp8x2_to_half2_bits(wb32[st] >> 16),
                bf16x2_to_half2_bits(xv32[2 * st]),
                bf16x2_to_half2_bits(xv32[2 * st + 1]));
        }
        #pragma unroll
        for (int i = 0; i < kWq; ++i) { wa[i] = na[i]; wb[i] = nb[i]; }
        #pragma unroll
        for (int i = 0; i < kXq; ++i) xv[i] = nx[i];
    }
    part[warp][gid][tg * 2] = c[0];
    part[warp][gid][tg * 2 + 1] = c[1];
    part[warp][gid + 8][tg * 2] = c[2];
    part[warp][gid + 8][tg * 2 + 1] = c[3];
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = lane + i * kWarp;
            const int row = idx >> 3;
            const int n = idx & 7;
            const int r = blockIdx.x * kTileRows + row;
            if (r < N && n < M) {
                float a = 0.0f;
                #pragma unroll
                for (int w = 0; w < kWarpsSplitK4SoOneTileReadsItsRowsAsFourContiguousQuarterStreams; ++w)
                    a += part[w][row][n];
                y[(size_t)n * N + r] = __float2bfloat16(a * __ldg(&row_scale[r]));
            }
        }
    }
}

constexpr int kStepsPerIter8Gives256ContiguousBytesPerRowPerWarpSoDramSeesFullBursts = 8;
constexpr int kKMultipleTheLauncherAccepts = kKMultipleOf128SoEveryWarpQuarterIsWholeK32Steps * kStepsPerIter8Gives256ContiguousBytesPerRowPerWarpSoDramSeesFullBursts;

}

extern "C" int nv_kernels_gemm_e4m3_mrows_f16mma_k_multiple() {
    return e4m3mrows::kKMultipleTheLauncherAccepts;
}

extern "C" int nv_kernels_gemm_e4m3_mrows_f16mma(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int M
) {
    using namespace e4m3mrows;
    if (N <= 0 || K <= 0 || M <= 0) return -1;
    if (M > kTokensMax8IsOneN8Tile) return -2;
    constexpr int S = kStepsPerIter8Gives256ContiguousBytesPerRowPerWarpSoDramSeesFullBursts;
    if ((K % kKMultipleTheLauncherAccepts) != 0) return -3;
    const unsigned grid = (unsigned)((N + kTileRows - 1) / kTileRows);
    gemm_e4m3_mrows_f16mma_kernel<S><<<grid, kThreads, 0, (cudaStream_t)stream>>>(
        wq, row_scale, reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), N, K, M);
    return (int)cudaGetLastError();
}
