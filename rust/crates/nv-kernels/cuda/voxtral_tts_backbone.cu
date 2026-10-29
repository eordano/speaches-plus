#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 8;
constexpr int kBlockDim = kWarpSize * kRowsPerBlock;
constexpr int kMaxK = 9216;
constexpr int kNormVpt = 24;
constexpr int kPrefetch = 4;

enum Mode { kPlain = 0, kNorm = 1, kResidualNorm = 2, kSiluResidual = 3 };

__device__ inline float block_sum_as_rmsnorm_cu(float v) {
    constexpr int kWarps = kBlockDim / kWarpSize;
    __shared__ float warp_sums[kWarps];
    __shared__ float total;
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    #pragma unroll
    for (int o = kWarpSize / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

__device__ inline float tree_sum_as_rmsnorm_residual_cu(float v) {
    __shared__ float scratch[kBlockDim];
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int stride = kBlockDim / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

template <int MODE>
__device__ inline void stage_x(
    float* xs, const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ aux,
    const __nv_bfloat16* __restrict__ wn, float eps, __nv_bfloat16* side, int K
) {
    if (MODE == kPlain) {
        for (int k = threadIdx.x; k < K; k += kBlockDim) xs[k] = __bfloat162float(x[k]);
    } else if (MODE == kSiluResidual) {
        for (int k = threadIdx.x; k < K; k += kBlockDim) {
            __nv_bfloat16 v = x[k];
            __nv_bfloat16 s = v / (static_cast<__nv_bfloat16>(1) + hexp(-v));
            xs[k] = __bfloat162float(s * x[K + k]);
        }
    } else {
        float xv[kNormVpt];
        float local = 0.f;
        #pragma unroll
        for (int u = 0; u < kNormVpt; ++u) {
            int i = threadIdx.x + u * kBlockDim;
            xv[u] = 0.f;
            if (i < K) {
                if (MODE == kNorm) {
                    xv[u] = __bfloat162float(x[i]);
                } else {
                    float s = __bfloat162float(x[i]) + __bfloat162float(aux[i]);
                    __nv_bfloat16 sb = __float2bfloat16(s);
                    if (blockIdx.x == 0) side[i] = sb;
                    xv[u] = __bfloat162float(sb);
                    local += s * s;
                }
            }
        }
        if (MODE == kNorm) {
            #pragma unroll
            for (int u = 0; u < kNormVpt; ++u) {
                int i = threadIdx.x + u * kBlockDim;
                if (i < K) local += xv[u] * xv[u];
            }
        }
        float sum = (MODE == kNorm) ? block_sum_as_rmsnorm_cu(local) : tree_sum_as_rmsnorm_residual_cu(local);
        float rms = rsqrtf(sum / (float)K + eps);
        #pragma unroll
        for (int u = 0; u < kNormVpt; ++u) {
            int i = threadIdx.x + u * kBlockDim;
            if (i < K) xs[i] = __bfloat162float(__float2bfloat16(xv[u] * rms * __bfloat162float(wn[i])));
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void dequant_e4m3x16(const int4& raw, float* wf) {
    const __nv_fp8x2_storage_t* p2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p2[t], __NV_E4M3);
        float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
        wf[2 * t] = f.x;
        wf[2 * t + 1] = f.y;
    }
}

template <int MODE, bool FP8>
__global__ void __launch_bounds__(kBlockDim) backbone_gemv_kernel(
    const void* __restrict__ W, const float* __restrict__ row_scale, const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ aux, const __nv_bfloat16* __restrict__ wn, float eps,
    __nv_bfloat16* side, __nv_bfloat16* __restrict__ y, int N, int K
) {
    extern __shared__ float xs[];
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    bool live = n < N;
    constexpr int kPerLoad = FP8 ? 16 : 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(
        reinterpret_cast<const uint8_t*>(W) + (size_t)(live ? n : 0) * K * (FP8 ? 1 : 2));
    int Kv = live ? K / kPerLoad : 0;
    uint4 pw[kPrefetch];
    #pragma unroll
    for (int u = 0; u < kPrefetch; ++u) {
        int v = lane + u * kWarpSize;
        pw[u] = (v < Kv) ? __ldg(&w4[v]) : make_uint4(0, 0, 0, 0);
    }
    stage_x<MODE>(xs, x, aux, wn, eps, side, K);
    if (!live) return;
    float acc = 0.0f;
    for (int v0 = lane; v0 < Kv; v0 += kWarpSize * kPrefetch) {
        uint4 nxt[kPrefetch];
        #pragma unroll
        for (int u = 0; u < kPrefetch; ++u) {
            int v = v0 + (u + kPrefetch) * kWarpSize;
            nxt[u] = (v < Kv) ? __ldg(&w4[v]) : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int u = 0; u < kPrefetch; ++u) {
            int v = v0 + u * kWarpSize;
            if (v < Kv) {
                int kb = v * kPerLoad;
                if (FP8) {
                    float wf[16];
                    dequant_e4m3x16(*reinterpret_cast<const int4*>(&pw[u]), wf);
                    #pragma unroll
                    for (int t = 0; t < 16; t += 2) {
                        acc += wf[t] * xs[kb + t] + wf[t + 1] * xs[kb + t + 1];
                    }
                } else {
                    const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw[u]);
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float2 wf = __bfloat1622float2(wp[j]);
                        acc += wf.x * xs[kb + 2 * j] + wf.y * xs[kb + 2 * j + 1];
                    }
                }
            }
        }
        #pragma unroll
        for (int u = 0; u < kPrefetch; ++u) pw[u] = nxt[u];
    }
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        __nv_bfloat16 r = __float2bfloat16(FP8 ? acc * __ldg(&row_scale[n]) : acc);
        y[n] = (MODE == kSiluResidual) ? (aux[n] + r) : r;
    }
}

}

template <bool FP8>
static int launch_backbone_gemv(
    cudaStream_t s, int mode, const void* w, const float* rs, const __nv_bfloat16* xx,
    const __nv_bfloat16* a, const __nv_bfloat16* nw, float eps, __nv_bfloat16* sd, __nv_bfloat16* yy,
    int N, int K
) {
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    size_t smem = (size_t)K * sizeof(float);
    void (*kernel)(const void*, const float*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, float, __nv_bfloat16*, __nv_bfloat16*, int, int);
    switch (mode) {
        case kPlain: kernel = backbone_gemv_kernel<kPlain, FP8>; break;
        case kNorm: kernel = backbone_gemv_kernel<kNorm, FP8>; break;
        case kResidualNorm: kernel = backbone_gemv_kernel<kResidualNorm, FP8>; break;
        case kSiluResidual: kernel = backbone_gemv_kernel<kSiluResidual, FP8>; break;
        default: return -2;
    }
    kernel<<<grid, kBlockDim, smem, s>>>(w, rs, xx, a, nw, eps, sd, yy, N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_backbone_gemv(
    void* stream, int mode, const void* W, const float* row_scale, const uint16_t* x, const uint16_t* aux,
    const uint16_t* wn, float eps, uint16_t* side, uint16_t* y, int N, int K
) {
    if (N < 1 || K < 16 || (K & 15) != 0 || K > kMaxK) return -1;
    if ((mode == kNorm || mode == kResidualNorm) && K > kNormVpt * kBlockDim) return -1;
    const __nv_bfloat16* xx = reinterpret_cast<const __nv_bfloat16*>(x);
    const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(aux);
    const __nv_bfloat16* nw = reinterpret_cast<const __nv_bfloat16*>(wn);
    __nv_bfloat16* sd = reinterpret_cast<__nv_bfloat16*>(side);
    __nv_bfloat16* yy = reinterpret_cast<__nv_bfloat16*>(y);
    cudaStream_t s = (cudaStream_t)stream;
    return row_scale ? launch_backbone_gemv<true>(s, mode, W, row_scale, xx, a, nw, eps, sd, yy, N, K)
                     : launch_backbone_gemv<false>(s, mode, W, row_scale, xx, a, nw, eps, sd, yy, N, K);
}
