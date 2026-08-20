
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 8;
constexpr int kBlockDim = kWarpSize * kRowsPerBlock;
constexpr int kMaxSharedK = 3072;

__device__ __forceinline__ float dot32_one_scale(
    uint4 pw,
    const float* xs,
    int kbase
) {
    float acc = 0.0f;
    const uint32_t w[4] = {pw.x, pw.y, pw.z, pw.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        uint32_t pv = w[j];
        const float* xp = xs + kbase + j * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int q = (int)((pv >> (4 * i)) & 0xF) - 8;
            acc += (float)q * xp[i];
        }
    }
    return acc;
}

template <bool kUseShared>
__global__ void gemv_w4a16_kernel(
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    int GS
) {
    __shared__ float xs[kUseShared ? kMaxSharedK : 1];
    if (kUseShared) {
        for (int k = threadIdx.x; k < K; k += kBlockDim) {
            xs[k] = __bfloat162float(x[k]);
        }
        __syncthreads();
    }

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    int Kw = K / 8;
    int Kv = Kw / 4;
    int groups = K / GS;
    const uint4* w_row = reinterpret_cast<const uint4*>(packed + (size_t)n * Kw);
    const __nv_bfloat16* s_row = scale + (size_t)n * groups;

    float acc = 0.0f;
    if (kUseShared && GS >= 32 && Kv <= 4 * kWarpSize) {
        uint4 pw[4];
        float sc[4];
        int cnt = 0;
        for (int v = lane; v < Kv; v += kWarpSize) {
            pw[cnt] = __ldg(&w_row[v]);
            sc[cnt] = __bfloat162float(__ldg(&s_row[(v * 32) / GS]));
            ++cnt;
        }
        int i = 0;
        for (int v = lane; v < Kv; v += kWarpSize) {
            acc += sc[i] * dot32_one_scale(pw[i], xs, v * 32);
            ++i;
        }
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) y[n] = __float2bfloat16(acc);
        return;
    }
    for (int v = lane; v < Kv; v += kWarpSize) {
        uint4 pw = __ldg(&w_row[v]);
        int kbase = v * 32;
        if (GS >= 32) {
            float sc = __bfloat162float(s_row[kbase / GS]);
            if (kUseShared) {
                acc += sc * dot32_one_scale(pw, xs, kbase);
            } else {
                float a = 0.0f;
                const uint32_t w[4] = {pw.x, pw.y, pw.z, pw.w};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    uint32_t pv = w[j];
                    int kb = kbase + j * 8;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        int q = (int)((pv >> (4 * i)) & 0xF) - 8;
                        a += (float)q * __bfloat162float(x[kb + i]);
                    }
                }
                acc += sc * a;
            }
        } else {
            const uint32_t w[4] = {pw.x, pw.y, pw.z, pw.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int kb = kbase + j * 8;
                float sc = __bfloat162float(s_row[kb / GS]);
                uint32_t pv = w[j];
                float a = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    int q = (int)((pv >> (4 * i)) & 0xF) - 8;
                    float xv = kUseShared ? xs[kb + i] : __bfloat162float(x[kb + i]);
                    a += (float)q * xv;
                }
                acc += a * sc;
            }
        }
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

__global__ void gemv_w4a16_row_kernel(
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    int GS
) {
    constexpr int kThreads = 256;
    constexpr int kWarps = kThreads / kWarpSize;
    int n = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;

    int Kw = K / 8;
    int Kv = Kw / 4;
    int groups = K / GS;
    const uint4* w_row = reinterpret_cast<const uint4*>(packed + (size_t)n * Kw);
    const __nv_bfloat16* s_row = scale + (size_t)n * groups;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);

    float acc = 0.0f;
    for (int v = tid; v < Kv; v += kThreads) {
        uint4 pw = __ldg(&w_row[v]);
        int kbase = v * 32;
        float sc = (GS >= 32) ? __bfloat162float(s_row[kbase / GS]) : 0.0f;
        const uint32_t w[4] = {pw.x, pw.y, pw.z, pw.w};
        float block_acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int kb = kbase + j * 8;
            float scj = (GS >= 32) ? sc : __bfloat162float(s_row[kb / GS]);
            uint32_t pv = w[j];
            float a = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 xf = __bfloat1622float2(__ldg(&x2[(kb >> 1) + i]));
                int q0 = (int)((pv >> (8 * i)) & 0xF) - 8;
                int q1 = (int)((pv >> (8 * i + 4)) & 0xF) - 8;
                a += (float)q0 * xf.x + (float)q1 * xf.y;
            }
            if (GS >= 32) {
                block_acc += a;
            } else {
                block_acc += a * scj;
            }
        }
        acc += (GS >= 32) ? sc * block_acc : block_acc;
    }

    int lane = tid & (kWarpSize - 1);
    int warp = tid / kWarpSize;
    #pragma unroll
    for (int o = kWarpSize / 2; o > 0; o >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, o);
    __shared__ float warp_sums[kWarps];
    if (lane == 0) warp_sums[warp] = acc;
    __syncthreads();
    if (warp == 0) {
        float sum = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) y[n] = __float2bfloat16(sum);
    }
}

__global__ void gemv_w4a16_gelu_pli_kernel(
    const uint32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ pli,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    int GS
) {
    __shared__ float xs[kMaxSharedK];
    for (int k = threadIdx.x; k < K; k += kBlockDim) {
        xs[k] = __bfloat162float(x[k]);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    int Kw = K / 8;
    int Kv = Kw / 4;
    int groups = K / GS;
    const uint4* w_row = reinterpret_cast<const uint4*>(packed + (size_t)n * Kw);
    const __nv_bfloat16* s_row = scale + (size_t)n * groups;

    float acc = 0.0f;
    uint4 pw[4];
    float sc[4];
    int cnt = 0;
    for (int v = lane; v < Kv && cnt < 4; v += kWarpSize) {
        pw[cnt] = __ldg(&w_row[v]);
        sc[cnt] = __bfloat162float(__ldg(&s_row[(v * 32) / GS]));
        ++cnt;
    }
    int i = 0;
    for (int v = lane; v < Kv; v += kWarpSize) {
        if (i < 4) {
            acc += sc[i] * dot32_one_scale(pw[i], xs, v * 32);
        } else {
            uint4 w = __ldg(&w_row[v]);
            float s2 = __bfloat162float(__ldg(&s_row[(v * 32) / GS]));
            acc += s2 * dot32_one_scale(w, xs, v * 32);
        }
        ++i;
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        const float c = 0.7978845608028654f;
        float t = tanhf(c * (acc + 0.044715f * acc * acc * acc));
        float gelu = 0.5f * acc * (1.0f + t);
        y[n] = __float2bfloat16(gelu * pli[n]);
    }
}

}

extern "C" int nv_kernels_gemv_w4a16_gelu_pli(
    void* stream,
    const uint32_t* packed,
    const uint16_t* scale,
    const uint16_t* x,
    const float* pli,
    uint16_t* y,
    int N,
    int K,
    int GS
) {
    if (N <= 0 || K <= 0 || GS < 32) return -1;
    if ((GS & 31) != 0) return -1;
    if ((K & 31) != 0 || (K % GS) != 0 || K > kMaxSharedK) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_w4a16_gelu_pli_kernel<<<grid, dim3(kBlockDim), 0, s>>>(
        packed,
        reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(x),
        pli,
        reinterpret_cast<__nv_bfloat16*>(y),
        N, K, GS
    );
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_w4a16(
    void* stream,
    const uint32_t* packed,
    const uint16_t* scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int GS
) {
    if (N <= 0 || K <= 0) return 0;
    if (GS <= 0) return -1;
    if ((K & 31) != 0 || (K % GS) != 0) return -1;
    if (GS >= 32 ? ((GS & 31) != 0) : ((GS & 7) != 0)) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    dim3 block(kBlockDim);
    if (K <= kMaxSharedK) {
        gemv_w4a16_kernel<true><<<grid, block, 0, s>>>(
            packed,
            reinterpret_cast<const __nv_bfloat16*>(scale),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(y),
            N, K, GS
        );
    } else {
        gemv_w4a16_row_kernel<<<dim3((unsigned)N), dim3(256), 0, s>>>(
            packed,
            reinterpret_cast<const __nv_bfloat16*>(scale),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(y),
            N, K, GS
        );
    }
    return (int)cudaGetLastError();
}
