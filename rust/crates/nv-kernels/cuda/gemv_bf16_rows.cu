#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "nvk_pdl.cuh"

namespace bf16rows {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock8MatchesGemvBf16SoEveryOutputRowKeepsItsWarp = 8;
constexpr int kBlockDim = kWarpSize * kRowsPerBlock8MatchesGemvBf16SoEveryOutputRowKeepsItsWarp;
constexpr int kMaxRows8CoversTheVerifyBlock = 8;
constexpr int kPastMaxRowsRcMinus2 = -2;

template <int ROWS>
__global__ void __launch_bounds__(kBlockDim) gemv_bf16_rows_kernel(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    NVK_PDL_PROLOG();
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x / kWarpSize;
    const int n = blockIdx.x * kRowsPerBlock8MatchesGemvBf16SoEveryOutputRowKeepsItsWarp + warp;
    if (n >= N) return;
    const __nv_bfloat16* w_row = W + (size_t)n * K;
    const int Kv = K / 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(w_row);
    float acc[ROWS];
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) acc[r] = 0.0f;
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    #pragma unroll 4
    for (int v = lane; v < Kv; v += kWarpSize) {
        uint4 pw = __ldg(&w4[v]);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) {
            uint4 px = __ldg(&x4[(size_t)r * Kv + v]);
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&px);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 wf = __bfloat1622float2(wp[j]);
                float2 xf = __bfloat1622float2(xp[j]);
                float xa = xf.x;
                float xb = xf.y;
                acc[r] += wf.x * xa + wf.y * xb;
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < ROWS; ++r) {
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            acc[r] += __shfl_xor_sync(0xffffffff, acc[r], offset);
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < ROWS; ++r) y[(size_t)r * N + n] = __float2bfloat16(acc[r]);
    }
    NVK_PDL_EPILOG();
}

template <int ROWS>
static int launch(dim3 grid, cudaStream_t s, const __nv_bfloat16* W, const __nv_bfloat16* x, __nv_bfloat16* y, int N, int K) {
    auto kern = gemv_bf16_rows_kernel<ROWS>;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(kBlockDim), 0, s);
        cudaLaunchKernelEx(&cfg, kern, W, x, y, N, K);
    } else {
        kern<<<grid, kBlockDim, 0, s>>>(W, x, y, N, K);
    }
    return (int)cudaGetLastError();
}

}

extern "C" int nv_kernels_gemv_bf16_rows(
    void* stream,
    const uint16_t* W,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int rows
) {
    using namespace bf16rows;
    if (N <= 0 || K <= 0 || (K & 7) != 0) return -1;
    if (rows < 1 || rows > kMaxRows8CoversTheVerifyBlock) return kPastMaxRowsRcMinus2;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock8MatchesGemvBf16SoEveryOutputRowKeepsItsWarp - 1) / kRowsPerBlock8MatchesGemvBf16SoEveryOutputRowKeepsItsWarp));
    const __nv_bfloat16* Wb = reinterpret_cast<const __nv_bfloat16*>(W);
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(x);
    __nv_bfloat16* yb = reinterpret_cast<__nv_bfloat16*>(y);
    switch (rows) {
        case 1: return launch<1>(grid, s, Wb, xb, yb, N, K);
        case 2: return launch<2>(grid, s, Wb, xb, yb, N, K);
        case 3: return launch<3>(grid, s, Wb, xb, yb, N, K);
        case 4: return launch<4>(grid, s, Wb, xb, yb, N, K);
        case 5: return launch<5>(grid, s, Wb, xb, yb, N, K);
        case 6: return launch<6>(grid, s, Wb, xb, yb, N, K);
        case 7: return launch<7>(grid, s, Wb, xb, yb, N, K);
        default: return launch<8>(grid, s, Wb, xb, yb, N, K);
    }
}
