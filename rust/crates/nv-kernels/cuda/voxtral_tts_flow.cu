#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 8;
constexpr int kBlockDim = kWarpSize * kRowsPerBlock;
constexpr int kRows = 6;
constexpr int kPrefetch = 2;
constexpr int kMinBlocksPerSm = 3;
constexpr int kMaxAcoustic = 64;

enum Epilogue { kStore = 0, kSiluPairs = 1, kAddResidual = 2 };

__device__ inline float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline __nv_bfloat16 rn(float v) { return __float2bfloat16(v); }

template <bool kFp8>
__device__ inline void dot_rows2(
    const uint4& raw0, const uint4& raw1, const uint4* __restrict__ x4, int Kv, int v, float* acc0, float* acc1
) {
    if (kFp8) {
        float wf0[16], wf1[16];
        const __nv_fp8x2_storage_t* p0 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw0);
        const __nv_fp8x2_storage_t* p1 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw1);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            __half2_raw h0 = __nv_cvt_fp8x2_to_halfraw2(p0[t], __NV_E4M3);
            __half2_raw h1 = __nv_cvt_fp8x2_to_halfraw2(p1[t], __NV_E4M3);
            float2 f0 = __half22float2(*reinterpret_cast<const __half2*>(&h0));
            float2 f1 = __half22float2(*reinterpret_cast<const __half2*>(&h1));
            wf0[2 * t] = f0.x;
            wf0[2 * t + 1] = f0.y;
            wf1[2 * t] = f1.x;
            wf1[2 * t + 1] = f1.y;
        }
        #pragma unroll
        for (int m = 0; m < kRows; ++m) {
            uint4 px0 = __ldg(&x4[(size_t)m * 2 * Kv + 2 * v]);
            uint4 px1 = __ldg(&x4[(size_t)m * 2 * Kv + 2 * v + 1]);
            const __nv_bfloat162* xa = reinterpret_cast<const __nv_bfloat162*>(&px0);
            const __nv_bfloat162* xb = reinterpret_cast<const __nv_bfloat162*>(&px1);
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 xv = __bfloat1622float2(p < 4 ? xa[p] : xb[p - 4]);
                acc0[m] += wf0[2 * p] * xv.x + wf0[2 * p + 1] * xv.y;
                acc1[m] += wf1[2 * p] * xv.x + wf1[2 * p + 1] * xv.y;
            }
        }
    } else {
        const __nv_bfloat162* w0 = reinterpret_cast<const __nv_bfloat162*>(&raw0);
        const __nv_bfloat162* w1 = reinterpret_cast<const __nv_bfloat162*>(&raw1);
        #pragma unroll
        for (int m = 0; m < kRows; ++m) {
            uint4 px = __ldg(&x4[(size_t)m * Kv + v]);
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&px);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 xf = __bfloat1622float2(xp[j]);
                float2 a0 = __bfloat1622float2(w0[j]);
                float2 a1 = __bfloat1622float2(w1[j]);
                acc0[m] += a0.x * xf.x + a0.y * xf.y;
                acc1[m] += a1.x * xf.x + a1.y * xf.y;
            }
        }
    }
}

template <int EPI, bool kFp8>
__global__ void __launch_bounds__(kBlockDim, kMinBlocksPerSm) tts_flow_gemm_kernel(
    const void* __restrict__ Wv, const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ epi_aux,
    __nv_bfloat16* __restrict__ y, int N, int K
) {
    constexpr bool kPairs = (EPI == kSiluPairs);
    constexpr int kPerChunk = kFp8 ? 16 : 8;
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int half = N / 2;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= half) return;
    const uint4* w4[2];
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        int row = kPairs ? n + s * half : 2 * n + s;
        w4[s] = reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(Wv) + (size_t)row * K * (kFp8 ? 1 : 2));
    }
    const uint4* x4 = reinterpret_cast<const uint4*>(x);
    int Kv = K / kPerChunk;
    float acc[2][kRows];
    #pragma unroll
    for (int s = 0; s < 2; ++s) {
        #pragma unroll
        for (int m = 0; m < kRows; ++m) acc[s][m] = 0.0f;
    }
    for (int v0 = lane; v0 < Kv; v0 += kWarpSize * kPrefetch) {
        uint4 pw[2][kPrefetch];
        #pragma unroll
        for (int u = 0; u < kPrefetch; ++u) {
            int v = v0 + u * kWarpSize;
            #pragma unroll
            for (int s = 0; s < 2; ++s) pw[s][u] = (v < Kv) ? __ldg(&w4[s][v]) : make_uint4(0, 0, 0, 0);
        }
        #pragma unroll
        for (int u = 0; u < kPrefetch; ++u) {
            int v = v0 + u * kWarpSize;
            if (v >= Kv) continue;
            dot_rows2<kFp8>(pw[0][u], pw[1][u], x4, Kv, v, acc[0], acc[1]);
        }
    }
    float rs[2];
    #pragma unroll
    for (int s = 0; s < 2; ++s) rs[s] = kFp8 ? __ldg(&row_scale[kPairs ? n + s * half : 2 * n + s]) : 1.0f;
    #pragma unroll
    for (int m = 0; m < kRows; ++m) {
        __nv_bfloat16 r[2];
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            float a = acc[s][m];
            #pragma unroll
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                a += __shfl_xor_sync(0xffffffff, a, offset);
            }
            r[s] = kFp8 ? rn(a * rs[s]) : rn(a);
        }
        if (lane == 0) {
            if (kPairs) {
                __nv_bfloat16 g = r[0];
                __nv_bfloat16 sg = g / (static_cast<__nv_bfloat16>(1) + hexp(-g));
                y[(size_t)m * half + n] = sg * r[1];
            } else {
                #pragma unroll
                for (int s = 0; s < 2; ++s) {
                    size_t o = (size_t)m * N + 2 * n + s;
                    y[o] = (EPI == kAddResidual) ? (epi_aux[o] + r[s]) : r[s];
                }
            }
        }
    }
}

__global__ void tts_flow_step_kernel(
    const __nv_bfloat16* __restrict__ x_in, const __nv_bfloat16* __restrict__ v, int uncond_off,
    float alpha, float one_minus_alpha, float dt, __nv_bfloat16* x_out,
    const __nv_bfloat16* __restrict__ w_in, const __nv_bfloat16* __restrict__ tp,
    const __nv_bfloat16* __restrict__ cond, __nv_bfloat16* tokens, int n, int d
) {
    __shared__ float xs[kMaxAcoustic];
    int tid = threadIdx.x;
    if (tid < n) {
        float xv = bf(x_in[tid]);
        if (v != nullptr) {
            float a = bf(rn(__fmul_rn(alpha, bf(v[tid]))));
            float b = bf(rn(__fmul_rn(one_minus_alpha, bf(v[uncond_off + tid]))));
            float vt = bf(rn(__fadd_rn(a, b)));
            float step = bf(rn(__fmul_rn(vt, dt)));
            xv = bf(rn(__fadd_rn(xv, step)));
        }
        xs[tid] = xv;
        if (x_out != nullptr && blockIdx.x == 0) x_out[tid] = rn(xv);
    }
    __syncthreads();
    if (tokens == nullptr) return;
    int lane = tid & (kWarpSize - 1);
    int r = blockIdx.x * kRowsPerBlock + tid / kWarpSize;
    if (r >= d) return;
    const __nv_bfloat16* w_row = w_in + (size_t)r * n;
    float acc = 0.0f;
    for (int k = lane; k < n; k += kWarpSize) acc += bf(w_row[k]) * xs[k];
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, offset);
    if (lane == 0) {
        __nv_bfloat16 a = rn(acc), t = tp[r];
        tokens[r] = a;
        tokens[d + r] = t;
        tokens[2 * d + r] = cond[r];
        tokens[3 * d + r] = a;
        tokens[4 * d + r] = t;
        tokens[5 * d + r] = rn(0.0f);
    }
}

template <int EPI, bool kFp8>
int launch_flow_gemm(
    cudaStream_t s, const void* W, const float* row_scale, const __nv_bfloat16* x,
    const __nv_bfloat16* epi_aux, __nv_bfloat16* y, int N, int K
) {
    dim3 grid((unsigned)((N / 2 + kRowsPerBlock - 1) / kRowsPerBlock));
    tts_flow_gemm_kernel<EPI, kFp8><<<grid, kBlockDim, 0, s>>>(W, row_scale, x, epi_aux, y, N, K);
    return (int)cudaGetLastError();
}

template <bool kFp8>
int launch_flow_gemm_epilogue(
    cudaStream_t s, int epi, const void* W, const float* row_scale, const __nv_bfloat16* x,
    const __nv_bfloat16* epi_aux, __nv_bfloat16* y, int N, int K
) {
    switch (epi) {
        case kStore: return launch_flow_gemm<kStore, kFp8>(s, W, row_scale, x, epi_aux, y, N, K);
        case kSiluPairs: return launch_flow_gemm<kSiluPairs, kFp8>(s, W, row_scale, x, epi_aux, y, N, K);
        case kAddResidual: return launch_flow_gemm<kAddResidual, kFp8>(s, W, row_scale, x, epi_aux, y, N, K);
        default: return -2;
    }
}

}

extern "C" int nv_kernels_voxtral_tts_flow_gemm(
    void* stream, int epi, int fp8, const void* W, const float* row_scale, const uint16_t* x,
    const uint16_t* epi_aux, uint16_t* y, int N, int K
) {
    if (N < 1 || K < 16 || (K & 15) != 0) return -11;
    if (fp8 && row_scale == nullptr) return -14;
    if (N & 1) return -15;
    if (epi == kAddResidual && epi_aux == nullptr) return -16;
    const __nv_bfloat16* xx = reinterpret_cast<const __nv_bfloat16*>(x);
    const __nv_bfloat16* ea = reinterpret_cast<const __nv_bfloat16*>(epi_aux);
    __nv_bfloat16* yy = reinterpret_cast<__nv_bfloat16*>(y);
    cudaStream_t s = (cudaStream_t)stream;
    return fp8 ? launch_flow_gemm_epilogue<true>(s, epi, W, row_scale, xx, ea, yy, N, K)
               : launch_flow_gemm_epilogue<false>(s, epi, W, row_scale, xx, ea, yy, N, K);
}

extern "C" int nv_kernels_voxtral_tts_flow_step(
    void* stream, const uint16_t* x_in, const uint16_t* v, int uncond_off, float alpha,
    float one_minus_alpha, float dt, uint16_t* x_out, const uint16_t* w_in, const uint16_t* tp,
    const uint16_t* cond, uint16_t* tokens, int n, int d
) {
    if (n < 1 || n > kMaxAcoustic || d < 1 || (v != nullptr && uncond_off < n)) return -1;
    if (tokens != nullptr && (w_in == nullptr || tp == nullptr || cond == nullptr)) return -1;
    unsigned grid = tokens ? (unsigned)((d + kRowsPerBlock - 1) / kRowsPerBlock) : 1u;
    tts_flow_step_kernel<<<grid, kBlockDim, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_in), reinterpret_cast<const __nv_bfloat16*>(v),
        uncond_off, alpha, one_minus_alpha, dt, reinterpret_cast<__nv_bfloat16*>(x_out),
        reinterpret_cast<const __nv_bfloat16*>(w_in), reinterpret_cast<const __nv_bfloat16*>(tp),
        reinterpret_cast<const __nv_bfloat16*>(cond), reinterpret_cast<__nv_bfloat16*>(tokens), n, d);
    return (int)cudaGetLastError();
}
