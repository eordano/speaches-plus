
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <atomic>
#include <cstdlib>
#include <mutex>

#include "nvk_pdl.cuh"
#include "nvk_grid.cuh"
#include "nvk_smem_optin.cuh"
#include "nvk_gdn_conv.cuh"

namespace {

constexpr int kWarpSize = 32;
constexpr int kRowsPerBlock = 8;
constexpr int kBlockDim = kWarpSize * kRowsPerBlock;
constexpr int kMaxSharedK = 4096;

template <bool kUseShared>
__global__ void gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    NVK_PDL_PROLOG();

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

    const __nv_bfloat16* w_row = W + (size_t)n * K;
    int Kv = K / 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(w_row);

    float acc = 0.0f;
    for (int v = lane; v < Kv; v += kWarpSize) {
        uint4 pw = __ldg(&w4[v]);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
        int kb = v * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wf = __bfloat1622float2(wp[j]);
            float xa, xb;
            if (kUseShared) {
                xa = xs[kb + 2 * j];
                xb = xs[kb + 2 * j + 1];
            } else {
                xa = __bfloat162float(x[kb + 2 * j]);
                xb = __bfloat162float(x[kb + 2 * j + 1]);
            }
            acc += wf.x * xa + wf.y * xb;
        }
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[n] = __float2bfloat16(acc);

    NVK_PDL_EPILOG();
}

__global__ void gemv_bf16_scalar_kernel(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;
    const __nv_bfloat16* w_row = W + (size_t)n * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += kWarpSize) {
        acc += __bfloat162float(w_row[k]) * __bfloat162float(x[k]);
    }
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

__global__ void gemv_bf16_normed_kernel(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    __shared__ float xs[kMaxSharedK];
    float r = rstd[0];
    for (int k = threadIdx.x; k < K; k += kBlockDim) {
        xs[k] = __bfloat162float(x[k]) * r * __bfloat162float(wn[k]);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const __nv_bfloat16* w_row = W + (size_t)n * K;
    int Kv = K / 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(w_row);

    float acc = 0.0f;
    for (int v = lane; v < Kv; v += kWarpSize) {
        uint4 pw = __ldg(&w4[v]);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
        int kb = v * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wf = __bfloat1622float2(wp[j]);
            acc += wf.x * xs[kb + 2 * j] + wf.y * xs[kb + 2 * j + 1];
        }
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

__global__ void gemv_bf16_normed_dynsmem_kernel_because_gdn_hidden_5120_exceeds_static_k(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    extern __shared__ float xs_dyn[];
    float r = rstd[0];
    for (int k = threadIdx.x; k < K; k += kBlockDim) {
        float v = __bfloat162float(x[k]) * r * __bfloat162float(wn[k]);
        xs_dyn[k] = __bfloat162float(__float2bfloat16(v));
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const __nv_bfloat16* w_row = W + (size_t)n * K;
    int Kv = K / 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(w_row);

    float acc = 0.0f;
    for (int v = lane; v < Kv; v += kWarpSize) {
        uint4 pw = __ldg(&w4[v]);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
        int kb = v * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wf = __bfloat1622float2(wp[j]);
            acc += wf.x * xs_dyn[kb + 2 * j] + wf.y * xs_dyn[kb + 2 * j + 1];
        }
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

__global__ void gemv_bf16_qkvg_kernel(
    const __nv_bfloat16* __restrict__ Wq,
    const __nv_bfloat16* __restrict__ Wk,
    const __nv_bfloat16* __restrict__ Wv,
    const __nv_bfloat16* __restrict__ Wg,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ yq,
    __nv_bfloat16* __restrict__ yk,
    __nv_bfloat16* __restrict__ yv,
    __nv_bfloat16* __restrict__ yg,
    int Nq,
    int Nk,
    int Nv,
    int Ng,
    int K
) {
    __shared__ float xs[kMaxSharedK];
    float r = rstd[0];
    for (int k = threadIdx.x; k < K; k += kBlockDim) {
        xs[k] = __bfloat162float(__float2bfloat16(
            __bfloat162float(x[k]) * r * __bfloat162float(wn[k])));
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;

    const __nv_bfloat16* W;
    __nv_bfloat16* y;
    int local = n;
    if (n < Nq) {
        W = Wq;
        y = yq;
    } else if ((local -= Nq) < Nk) {
        W = Wk;
        y = yk;
    } else if ((local -= Nk) < Nv) {
        W = Wv;
        y = yv;
    } else if ((local -= Nv) < Ng) {
        W = Wg;
        y = yg;
    } else {
        return;
    }

    const __nv_bfloat16* w_row = W + (size_t)local * K;
    int Kv = K / 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(w_row);

    float acc = 0.0f;
    for (int v = lane; v < Kv; v += kWarpSize) {
        uint4 pw = __ldg(&w4[v]);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
        int kb = v * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wf = __bfloat1622float2(wp[j]);
            acc += wf.x * xs[kb + 2 * j] + wf.y * xs[kb + 2 * j + 1];
        }
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[local] = __float2bfloat16(acc);
}

__global__ void rowquant_i8_kernel(
    const __nv_bfloat16* __restrict__ w,
    int8_t* __restrict__ wq,
    float* __restrict__ row_scale,
    int N,
    int K
) {
    int n = blockIdx.x;
    if (n >= N) return;
    const __nv_bfloat16* row = w + (size_t)n * K;
    float amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        amax = fmaxf(amax, fabsf(__bfloat162float(row[k])));
    }
    __shared__ float red[256];
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float scale = red[0] / 127.0f;
    float inv = (scale > 0.0f) ? (1.0f / scale) : 0.0f;
    if (threadIdx.x == 0) row_scale[n] = scale;
    int8_t* out = wq + (size_t)n * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __bfloat162float(row[k]) * inv;
        int q = __float2int_rn(v);
        q = max(-127, min(127, q));
        out[k] = (int8_t)q;
    }
}

__global__ void gemv_i8_normed_kernel(
    const int8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {

    __shared__ float xs[kMaxSharedK + kMaxSharedK / 16];
    float r = rstd[0];
    for (int k = threadIdx.x; k < K; k += kBlockDim) {
        xs[(k >> 4) * 17 + (k & 15)] = __bfloat162float(x[k]) * r * __bfloat162float(wn[k]);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const int8_t* w_row = wq + (size_t)n * K;
    int Kv = K / 16;
    const int4* w16 = reinterpret_cast<const int4*>(w_row);

    float acc = 0.0f;
    for (int v = lane; v < Kv; v += kWarpSize) {
        int4 raw = __ldg(&w16[v]);
        const unsigned* wu = reinterpret_cast<const unsigned*>(&raw);
        const float* xp = xs + v * 17;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned u = wu[t] ^ 0x80808080u;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float f = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650u + i)) - 8388736.0f;
                acc += f * xp[4 * t + i];
            }
        }
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[n] = __float2bfloat16(acc * __ldg(&row_scale[n]));
}

}

extern "C" int nv_kernels_rowquant_i8(
    void* stream,
    const uint16_t* w,
    int8_t* wq,
    float* row_scale,
    int N,
    int K
) {
    if (N <= 0 || K <= 0) return -1;
    rowquant_i8_kernel<<<(unsigned)N, 256, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(w), wq, row_scale, N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_i8_normed(
    void* stream,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K
) {
    if (N <= 0 || K <= 0) return 0;
    if ((K & 15) != 0 || K > kMaxSharedK) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_i8_normed_kernel<<<grid, dim3(kBlockDim), 0, s>>>(
        wq, row_scale,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(wn),
        rstd,
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_bf16_normed(
    void* stream,
    const uint16_t* W,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K
) {
    if (N <= 0 || K <= 0) return 0;
    if ((K & 7) != 0) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    dim3 block(kBlockDim);
    if (K > kMaxSharedK) {
        size_t smem = (size_t)K * sizeof(float);
        if (smem > (size_t)nvk_max_dynamic_smem_optin()) return -1;
        static DynamicSmemOptin optin_normed_dyn;
        int orc = raise_dynamic_smem_optin_never_lowering_it(
            optin_normed_dyn,
            (const void*)gemv_bf16_normed_dynsmem_kernel_because_gdn_hidden_5120_exceeds_static_k,
            smem);
        if (orc != 0) return orc;
        gemv_bf16_normed_dynsmem_kernel_because_gdn_hidden_5120_exceeds_static_k<<<grid, block,
                                                                                  smem, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(W),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(wn),
            rstd,
            reinterpret_cast<__nv_bfloat16*>(y), N, K);
        return (int)cudaGetLastError();
    }
    gemv_bf16_normed_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(W),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(wn),
        rstd,
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_bf16_qkvg_normed(
    void* stream,
    const uint16_t* Wq,
    const uint16_t* Wk,
    const uint16_t* Wv,
    const uint16_t* Wg,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* yq,
    uint16_t* yk,
    uint16_t* yv,
    uint16_t* yg,
    int Nq,
    int Nk,
    int Nv,
    int Ng,
    int K
) {
    if (Nq <= 0 || Nk <= 0 || Nv <= 0 || Ng < 0 || K <= 0) return -1;
    if ((K & 7) != 0 || K > kMaxSharedK) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    int total = Nq + Nk + Nv + Ng;
    dim3 grid((unsigned)((total + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_bf16_qkvg_kernel<<<grid, dim3(kBlockDim), 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(Wq),
        reinterpret_cast<const __nv_bfloat16*>(Wk),
        reinterpret_cast<const __nv_bfloat16*>(Wv),
        reinterpret_cast<const __nv_bfloat16*>(Wg),
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(wn),
        rstd,
        reinterpret_cast<__nv_bfloat16*>(yq),
        reinterpret_cast<__nv_bfloat16*>(yk),
        reinterpret_cast<__nv_bfloat16*>(yv),
        reinterpret_cast<__nv_bfloat16*>(yg),
        Nq, Nk, Nv, Ng, K);
    return (int)cudaGetLastError();
}

template <bool kUseShared>
static int nvk_launch_gemv_bf16(
    dim3 grid,
    dim3 block,
    cudaStream_t s,
    const __nv_bfloat16* W,
    const __nv_bfloat16* x,
    __nv_bfloat16* y,
    int N,
    int K
) {
    if (grid.y > 65535 || grid.z > 65535) return NVK_ERR_GRID_AXIS;
    if (nvk_pdl_enabled()) {
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = 0;
        cfg.stream = s;
        cfg.attrs = &attr;
        cfg.numAttrs = 1;
        cudaLaunchKernelEx(&cfg, gemv_bf16_kernel<kUseShared>, W, x, y, N, K);
    } else {
        gemv_bf16_kernel<kUseShared><<<grid, block, 0, s>>>(W, x, y, N, K);
    }
    return 0;
}

extern "C" int nv_kernels_gemv_bf16(
    void* stream,
    const uint16_t* W,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K
) {
    if (N <= 0 || K <= 0) return 0;
    if ((K & 1) != 0) {
        return -1;
    }
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    dim3 block(kBlockDim);
    if ((K & 7) == 0 && K <= kMaxSharedK) {
        int rc = nvk_launch_gemv_bf16<true>(
            grid, block, s,
            reinterpret_cast<const __nv_bfloat16*>(W),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(y), N, K);
        if (rc != 0) return rc;
    } else if ((K & 7) == 0) {
        int rc = nvk_launch_gemv_bf16<false>(
            grid, block, s,
            reinterpret_cast<const __nv_bfloat16*>(W),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(y), N, K);
        if (rc != 0) return rc;
    } else {
        gemv_bf16_scalar_kernel<<<grid, block, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(W),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<__nv_bfloat16*>(y), N, K);
    }
    return (int)cudaGetLastError();
}

__global__ void gemv_i8_normed_mk_kernel(
    const int8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    int M
) {
    extern __shared__ float xsd[];
    const int rowpad = (K >> 4) * 17;
    for (int idx = threadIdx.x; idx < M * K; idx += kBlockDim) {
        int j = idx / K;
        int k = idx - j * K;
        float v = __bfloat162float(x[(size_t)j * K + k]) * rstd[j] * __bfloat162float(wn[k]);
        xsd[j * rowpad + (k >> 4) * 17 + (k & 15)] = v;
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const int8_t* w_row = wq + (size_t)n * K;
    int Kv = K / 16;
    const int4* w16 = reinterpret_cast<const int4*>(w_row);

    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;

    for (int v = lane; v < Kv; v += kWarpSize) {
        int4 raw = __ldg(&w16[v]);
        const unsigned* wu = reinterpret_cast<const unsigned*>(&raw);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned u = wu[t] ^ 0x80808080u;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float f = __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650u + i)) - 8388736.0f;
                int koff = v * 17 + 4 * t + i;
                for (int j = 0; j < M; ++j) {
                    acc[j] += f * xsd[j * rowpad + koff];
                }
            }
        }
    }

    float rs = __ldg(&row_scale[n]);
    for (int j = 0; j < M; ++j) {
        float a = acc[j];
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, offset);
        }
        if (lane == 0) y[(size_t)j * N + n] = __float2bfloat16(a * rs);
    }
}

template <int TM>
__global__ void gemv_i8_normed_mk_h_kernel(
    const int8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    extern __shared__ __nv_bfloat162 xsh[];
    const int rowpad = (K >> 4) * 9;
    const int kp = K >> 1;
    for (int idx = threadIdx.x; idx < TM * kp; idx += kBlockDim) {
        int j = idx / kp;
        int p = idx - j * kp;
        int k0 = 2 * p;
        float r = rstd[j];
        float a = __bfloat162float(x[(size_t)j * K + k0]) * r * __bfloat162float(wn[k0]);
        float b = __bfloat162float(x[(size_t)j * K + k0 + 1]) * r * __bfloat162float(wn[k0 + 1]);
        xsh[j * rowpad + (p >> 3) * 9 + (p & 7)] = __floats2bfloat162_rn(a, b);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const int4* w16 = reinterpret_cast<const int4*>(wq + (size_t)n * K);
    int Kv = K / 16;
    float acc[TM];
    #pragma unroll
    for (int j = 0; j < TM; ++j) acc[j] = 0.0f;

    for (int v = lane; v < Kv; v += kWarpSize) {
        int4 raw = __ldg(&w16[v]);
        const unsigned* wu = reinterpret_cast<const unsigned*>(&raw);
        float wf[16];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned u = wu[t] ^ 0x80808080u;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                wf[4 * t + i] =
                    __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650u + i)) - 8388736.0f;
            }
        }
        const __nv_bfloat162* base = xsh + v * 9;
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            const __nv_bfloat162* xp = base + j * rowpad;
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 xv = __bfloat1622float2(xp[p]);
                acc[j] += wf[2 * p] * xv.x + wf[2 * p + 1] * xv.y;
            }
        }
    }

    float rs = __ldg(&row_scale[n]);
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        float a = acc[j];
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, offset);
        }
        if (lane == 0) y[(size_t)j * N + n] = __float2bfloat16(a * rs);
    }
}

static int mk_smem_limit() { return nvk_max_dynamic_smem_optin(); }

template <int TM>
static int launch_gemv_i8_mk_h(
    cudaStream_t s,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K
) {
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(__nv_bfloat162);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(
        optin, (const void*)gemv_i8_normed_mk_h_kernel<TM>, smem);
    if (orc != 0) return orc;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_i8_normed_mk_h_kernel<TM><<<grid, dim3(kBlockDim), smem, s>>>(
        wq, row_scale,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(wn),
        rstd,
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_i8_normed_mk_max_m(int K) {
    if (K <= 0 || (K & 15) != 0) return 0;
    size_t limit = (size_t)mk_smem_limit();
    if (limit == 0) return 8;
    size_t f32_row = (size_t)(K >> 4) * 17 * sizeof(float);
    size_t h_row = (size_t)(K >> 4) * 9 * sizeof(__nv_bfloat162);
    int mf = (int)(limit / f32_row);
    if (mf > 8) mf = 8;
    int mh = (int)(limit / h_row);
    if (mh > 16) mh = 16;
    return mf > mh ? mf : mh;
}

static int gemv_i8_mk_h_switch(
    cudaStream_t s,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K,
    int M
) {
    switch (M) {
        case 1: return launch_gemv_i8_mk_h<1>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 2: return launch_gemv_i8_mk_h<2>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 3: return launch_gemv_i8_mk_h<3>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 4: return launch_gemv_i8_mk_h<4>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 5: return launch_gemv_i8_mk_h<5>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 6: return launch_gemv_i8_mk_h<6>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 7: return launch_gemv_i8_mk_h<7>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 8: return launch_gemv_i8_mk_h<8>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 9: return launch_gemv_i8_mk_h<9>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 10: return launch_gemv_i8_mk_h<10>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 11: return launch_gemv_i8_mk_h<11>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 12: return launch_gemv_i8_mk_h<12>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 13: return launch_gemv_i8_mk_h<13>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 14: return launch_gemv_i8_mk_h<14>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 15: return launch_gemv_i8_mk_h<15>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 16: return launch_gemv_i8_mk_h<16>(s, wq, row_scale, x, wn, rstd, y, N, K);
        default: return -1;
    }
}

extern "C" int nv_kernels_gemv_i8_normed_mk(
    void* stream,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K,
    int M
) {
    if (N <= 0 || K <= 0 || M <= 0) return 0;
    if ((K & 15) != 0 || M > 16) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    size_t limit = (size_t)mk_smem_limit();
    size_t smem = (size_t)M * (K >> 4) * 17 * sizeof(float);
    if (M <= 8 && (limit == 0 || smem <= limit)) {
        static DynamicSmemOptin optin;
        int orc = raise_dynamic_smem_optin_never_lowering_it(
            optin, (const void*)gemv_i8_normed_mk_kernel, smem);
        if (orc != 0) return orc;
        dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
        gemv_i8_normed_mk_kernel<<<grid, dim3(kBlockDim), smem, s>>>(
            wq, row_scale,
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(wn),
            rstd,
            reinterpret_cast<__nv_bfloat16*>(y), N, K, M);
        return (int)cudaGetLastError();
    }
    size_t smem_h = (size_t)M * (K >> 4) * 9 * sizeof(__nv_bfloat162);
    if (limit == 0 || smem_h > limit) return -1;
    return gemv_i8_mk_h_switch(s, wq, row_scale, x, wn, rstd, y, N, K, M);
}

extern "C" int nv_kernels_gemv_i8_mk_h(
    void* stream,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K,
    int M
) {
    if (N <= 0 || K <= 0 || M <= 0) return 0;
    if ((K & 15) != 0 || M > 16) return -1;
    size_t limit = (size_t)mk_smem_limit();
    size_t smem_h = (size_t)M * (K >> 4) * 9 * sizeof(__nv_bfloat162);
    if (limit == 0 || smem_h > limit) return -1;
    return gemv_i8_mk_h_switch(
        (cudaStream_t)stream, wq, row_scale, x, wn, rstd, y, N, K, M);
}

__global__ void normx_mk_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ xn,
    int K,
    int M
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * K) return;
    int j = idx / K;
    int k = idx - j * K;
    xn[idx] = __float2bfloat16(
        __bfloat162float(x[idx]) * rstd[j] * __bfloat162float(wn[k]));
}

template <int TM>
__global__ void gemv_i8_prenormed_mk_kernel(
    const int8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ xn,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    extern __shared__ __nv_bfloat162 xsh[];
    const int rowpad = (K >> 4) * 9;
    const int kp = K >> 1;
    const int rows_per_block = blockDim.x / kWarpSize;
    const __nv_bfloat162* xn2 = reinterpret_cast<const __nv_bfloat162*>(xn);
    for (int idx = threadIdx.x; idx < TM * kp; idx += blockDim.x) {
        int j = idx / kp;
        int p = idx - j * kp;
        xsh[j * rowpad + (p >> 3) * 9 + (p & 7)] = xn2[(size_t)j * kp + p];
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int Kv = K / 16;
    int n_tiles = (N + rows_per_block - 1) / rows_per_block;

    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        int n = tile * rows_per_block + warp;
        if (n >= N) continue;

        const int4* w16 = reinterpret_cast<const int4*>(wq + (size_t)n * K);
        float acc[TM];
        #pragma unroll
        for (int j = 0; j < TM; ++j) acc[j] = 0.0f;

        for (int v = lane; v < Kv; v += kWarpSize) {
            int4 raw = __ldg(&w16[v]);
            const unsigned* wu = reinterpret_cast<const unsigned*>(&raw);
            float wf[16];
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                unsigned u = wu[t] ^ 0x80808080u;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    wf[4 * t + i] =
                        __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650u + i)) - 8388736.0f;
                }
            }
            const __nv_bfloat162* base = xsh + v * 9;
            #pragma unroll
            for (int j = 0; j < TM; ++j) {
                const __nv_bfloat162* xp = base + j * rowpad;
                #pragma unroll
                for (int p = 0; p < 8; ++p) {
                    float2 xv = __bfloat1622float2(xp[p]);
                    acc[j] += wf[2 * p] * xv.x + wf[2 * p + 1] * xv.y;
                }
            }
        }

        float rs = __ldg(&row_scale[n]);
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            float a = acc[j];
            #pragma unroll
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
                a += __shfl_xor_sync(0xffffffff, a, offset);
            }
            if (lane == 0) y[(size_t)j * N + n] = __float2bfloat16(a * rs);
        }
    }
}

template <int TM>
static int launch_gemv_i8_prenormed_mk(
    cudaStream_t s,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* xn,
    uint16_t* y,
    int N,
    int K
) {
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(__nv_bfloat162);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(
        optin, (const void*)gemv_i8_prenormed_mk_kernel<TM>, smem);
    if (orc != 0) return orc;
    int rows = TM <= 5 ? 32 : kRowsPerBlock;
    int n_tiles = (N + rows - 1) / rows;
    static int n_sms = 0;
    if (n_sms == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&n_sms, cudaDevAttrMultiProcessorCount, dev) !=
            cudaSuccess || n_sms <= 0) {
            n_sms = 128;
        }
    }
    int blocks = 4 * n_sms;
    if (blocks > n_tiles) blocks = n_tiles;
    dim3 grid((unsigned)blocks);
    gemv_i8_prenormed_mk_kernel<TM><<<grid, dim3(rows * kWarpSize), smem, s>>>(
        wq, row_scale,
        reinterpret_cast<const __nv_bfloat16*>(xn),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_normx_mk(
    void* stream,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* xn,
    int K,
    int M
) {
    if (K <= 0 || M <= 0) return 0;
    int total = M * K;
    int bs = 256;
    normx_mk_kernel<<<dim3((unsigned)((total + bs - 1) / bs)), dim3(bs), 0,
                      (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(wn), rstd,
        reinterpret_cast<__nv_bfloat16*>(xn), K, M);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_i8_prenormed_mk(
    void* stream,
    const int8_t* wq,
    const float* row_scale,
    const uint16_t* xn,
    uint16_t* y,
    int N,
    int K,
    int M
) {
    if (N <= 0 || K <= 0 || M <= 0) return 0;
    if ((K & 15) != 0 || M > 8) return -1;
    size_t limit = (size_t)mk_smem_limit();
    size_t smem = (size_t)M * (K >> 4) * 9 * sizeof(__nv_bfloat162);
    if (limit == 0 || smem > limit) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    switch (M) {
        case 1: return launch_gemv_i8_prenormed_mk<1>(s, wq, row_scale, xn, y, N, K);
        case 2: return launch_gemv_i8_prenormed_mk<2>(s, wq, row_scale, xn, y, N, K);
        case 3: return launch_gemv_i8_prenormed_mk<3>(s, wq, row_scale, xn, y, N, K);
        case 4: return launch_gemv_i8_prenormed_mk<4>(s, wq, row_scale, xn, y, N, K);
        case 5: return launch_gemv_i8_prenormed_mk<5>(s, wq, row_scale, xn, y, N, K);
        case 6: return launch_gemv_i8_prenormed_mk<6>(s, wq, row_scale, xn, y, N, K);
        case 7: return launch_gemv_i8_prenormed_mk<7>(s, wq, row_scale, xn, y, N, K);
        case 8: return launch_gemv_i8_prenormed_mk<8>(s, wq, row_scale, xn, y, N, K);
        default: return -1;
    }
}

__global__ void rowquant_e4m3_kernel(
    const __nv_bfloat16* __restrict__ w,
    uint8_t* __restrict__ wq,
    float* __restrict__ row_scale,
    int N,
    int K
) {
    int n = blockIdx.x;
    if (n >= N) return;
    const __nv_bfloat16* row = w + (size_t)n * K;
    float amax = 0.0f;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float a = fabsf(__bfloat162float(row[k]));
        if (isfinite(a)) amax = fmaxf(amax, a);
    }
    __shared__ float red[256];
    red[threadIdx.x] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    float rmax = red[0];
    float scale = (rmax > 0.0f) ? (rmax / 448.0f) : 0.0f;
    float inv = (rmax > 0.0f) ? (448.0f / rmax) : 0.0f;
    if (threadIdx.x == 0) row_scale[n] = scale;
    uint8_t* out = wq + (size_t)n * K;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        float v = __bfloat162float(row[k]) * inv;
        out[k] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
}

template <int TM>
__global__ void gemv_e4m3_mk_h_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    extern __shared__ __nv_bfloat162 xsh8[];
    const int rowpad = (K >> 4) * 9;
    const int kp = K >> 1;
    for (int idx = threadIdx.x; idx < TM * kp; idx += kBlockDim) {
        int j = idx / kp;
        int p = idx - j * kp;
        int k0 = 2 * p;
        float r = rstd[j];
        float a = __bfloat162float(x[(size_t)j * K + k0]) * r * __bfloat162float(wn[k0]);
        float b = __bfloat162float(x[(size_t)j * K + k0 + 1]) * r * __bfloat162float(wn[k0 + 1]);
        xsh8[j * rowpad + (p >> 3) * 9 + (p & 7)] = __floats2bfloat162_rn(a, b);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const int4* w16 = reinterpret_cast<const int4*>(wq + (size_t)n * K);
    int Kv = K / 16;
    float acc[TM];
    #pragma unroll
    for (int j = 0; j < TM; ++j) acc[j] = 0.0f;

    for (int v = lane; v < Kv; v += kWarpSize) {
        int4 raw = __ldg(&w16[v]);
        const __nv_fp8x2_storage_t* p2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
        float wf[16];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p2[t], __NV_E4M3);
            float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            wf[2 * t] = f.x;
            wf[2 * t + 1] = f.y;
        }
        const __nv_bfloat162* base = xsh8 + v * 9;
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            const __nv_bfloat162* xp = base + j * rowpad;
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 xv = __bfloat1622float2(xp[p]);
                acc[j] += wf[2 * p] * xv.x + wf[2 * p + 1] * xv.y;
            }
        }
    }

    float rs = __ldg(&row_scale[n]);
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        float a = acc[j];
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, offset);
        }
        if (lane == 0) y[(size_t)j * N + n] = __float2bfloat16(a * rs);
    }
}

template <int TM>
static int launch_gemv_e4m3_mk_h(
    cudaStream_t s,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K
) {
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(__nv_bfloat162);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(
        optin, (const void*)gemv_e4m3_mk_h_kernel<TM>, smem);
    if (orc != 0) return orc;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_e4m3_mk_h_kernel<TM><<<grid, dim3(kBlockDim), smem, s>>>(
        wq, row_scale,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(wn),
        rstd,
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_rowquant_e4m3(
    void* stream,
    const uint16_t* w,
    uint8_t* wq,
    float* row_scale,
    int N,
    int K
) {
    if (N <= 0 || K <= 0) return -1;
    rowquant_e4m3_kernel<<<(unsigned)N, 256, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(w), wq, row_scale, N, K);
    return (int)cudaGetLastError();
}

template <bool kPreNormFoldRoundsToBf16MatchingTheDeadRmsnormKernel>
__global__ void gemv_e4m3_m1_rows2_per_warp_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
);

extern "C" int nv_kernels_gemv_e4m3_mk_h(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* y,
    int N,
    int K,
    int M
) {
    if (N <= 0 || K <= 0 || M <= 0) return 0;
    if ((K & 15) != 0 || M > 16) return -1;
    size_t limit = (size_t)mk_smem_limit();
    size_t smem_h = (size_t)M * (K >> 4) * 9 * sizeof(__nv_bfloat162);
    if (limit == 0 || smem_h > limit) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    static const bool rows2_off_env_nv_q38_e4m3_r2_off_h =
        getenv("NV_Q38_E4M3_R2_OFF") != nullptr;
    if (M == 1 && !rows2_off_env_nv_q38_e4m3_r2_off_h) {
        size_t smem1 = (size_t)(K >> 4) * 9 * sizeof(float2);
        if (smem1 <= limit) {
            dim3 grid((unsigned)((N + 2 * kRowsPerBlock - 1) / (2 * kRowsPerBlock)));
            static DynamicSmemOptin optin_r2h;
            int orc = raise_dynamic_smem_optin_never_lowering_it(
                optin_r2h, (const void*)gemv_e4m3_m1_rows2_per_warp_kernel<true>, smem1);
            if (orc != 0) return orc;
            gemv_e4m3_m1_rows2_per_warp_kernel<true><<<grid, dim3(kBlockDim), smem1, s>>>(
                wq, row_scale,
                reinterpret_cast<const __nv_bfloat16*>(x),
                reinterpret_cast<const __nv_bfloat16*>(wn),
                rstd,
                reinterpret_cast<__nv_bfloat16*>(y), N, K);
            return (int)cudaGetLastError();
        }
    }
    switch (M) {
        case 1: return launch_gemv_e4m3_mk_h<1>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 2: return launch_gemv_e4m3_mk_h<2>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 3: return launch_gemv_e4m3_mk_h<3>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 4: return launch_gemv_e4m3_mk_h<4>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 5: return launch_gemv_e4m3_mk_h<5>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 6: return launch_gemv_e4m3_mk_h<6>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 7: return launch_gemv_e4m3_mk_h<7>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 8: return launch_gemv_e4m3_mk_h<8>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 9: return launch_gemv_e4m3_mk_h<9>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 10: return launch_gemv_e4m3_mk_h<10>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 11: return launch_gemv_e4m3_mk_h<11>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 12: return launch_gemv_e4m3_mk_h<12>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 13: return launch_gemv_e4m3_mk_h<13>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 14: return launch_gemv_e4m3_mk_h<14>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 15: return launch_gemv_e4m3_mk_h<15>(s, wq, row_scale, x, wn, rstd, y, N, K);
        case 16: return launch_gemv_e4m3_mk_h<16>(s, wq, row_scale, x, wn, rstd, y, N, K);
        default: return -1;
    }
}

template <bool kFp8>
__device__ __forceinline__ void nvk_deq_bytes16(const int4& raw, float* wf) {
    if (kFp8) {
        const __nv_fp8x2_storage_t* p2 = reinterpret_cast<const __nv_fp8x2_storage_t*>(&raw);
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2(p2[t], __NV_E4M3);
            float2 f = __half22float2(*reinterpret_cast<const __half2*>(&hr));
            wf[2 * t] = f.x;
            wf[2 * t + 1] = f.y;
        }
    } else {
        const unsigned* wu = reinterpret_cast<const unsigned*>(&raw);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned u = wu[t] ^ 0x80808080u;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                wf[4 * t + i] =
                    __uint_as_float(__byte_perm(u, 0x4B000000u, 0x7650u + i)) - 8388736.0f;
            }
        }
    }
}

template <int TM>
__global__ void gemv_e4m3_mk_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    extern __shared__ __nv_bfloat162 xshr[];
    const int rowpad = (K >> 4) * 9;
    const int kp = K >> 1;
    for (int idx = threadIdx.x; idx < TM * kp; idx += kBlockDim) {
        int j = idx / kp;
        int p = idx - j * kp;
        int k0 = 2 * p;
        xshr[j * rowpad + (p >> 3) * 9 + (p & 7)] = __floats2bfloat162_rn(
            __bfloat162float(x[(size_t)j * K + k0]),
            __bfloat162float(x[(size_t)j * K + k0 + 1]));
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const int4* w16 = reinterpret_cast<const int4*>(wq + (size_t)n * K);
    int Kv = K / 16;
    float acc[TM];
    #pragma unroll
    for (int j = 0; j < TM; ++j) acc[j] = 0.0f;

    int v = lane;
    int4 raw = (v < Kv) ? __ldcs(&w16[v]) : make_int4(0, 0, 0, 0);
    while (v < Kv) {
        int vn = v + kWarpSize;
        int4 nxt = (vn < Kv) ? __ldcs(&w16[vn]) : make_int4(0, 0, 0, 0);
        float wf[16];
        nvk_deq_bytes16<true>(raw, wf);
        const __nv_bfloat162* base = xshr + v * 9;
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            const __nv_bfloat162* xp = base + j * rowpad;
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 xv = __bfloat1622float2(xp[p]);
                acc[j] += wf[2 * p] * xv.x + wf[2 * p + 1] * xv.y;
            }
        }
        v = vn;
        raw = nxt;
    }

    float rs = __ldg(&row_scale[n]);
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        float a = acc[j];
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, offset);
        }
        if (lane == 0) y[(size_t)j * N + n] = __float2bfloat16(a * rs);
    }
}

template <bool kPreNormFoldRoundsToBf16MatchingTheDeadRmsnormKernel>
__global__ void gemv_e4m3_m1_rows2_per_warp_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    extern __shared__ float2 xshf[];
    const int kp = K >> 1;
    float r = kPreNormFoldRoundsToBf16MatchingTheDeadRmsnormKernel ? rstd[0] : 0.0f;
    for (int idx = threadIdx.x; idx < kp; idx += kBlockDim) {
        int k0 = 2 * idx;
        float a = __bfloat162float(x[k0]);
        float b = __bfloat162float(x[k0 + 1]);
        if (kPreNormFoldRoundsToBf16MatchingTheDeadRmsnormKernel) {
            a = a * r * __bfloat162float(wn[k0]);
            b = b * r * __bfloat162float(wn[k0 + 1]);
            float2 f = __bfloat1622float2(__floats2bfloat162_rn(a, b));
            a = f.x;
            b = f.y;
        }
        xshf[(idx >> 3) * 9 + (idx & 7)] = make_float2(a, b);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n0 = blockIdx.x * (2 * kRowsPerBlock) + warp;
    int n1 = n0 + kRowsPerBlock;
    if (n0 >= N) return;
    bool has1 = n1 < N;

    const int4* wa = reinterpret_cast<const int4*>(wq + (size_t)n0 * K);
    const int4* wb = reinterpret_cast<const int4*>(wq + (size_t)n1 * K);
    int Kv = K / 16;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    int v = lane;
    int4 ra = (v < Kv) ? __ldcs(&wa[v]) : make_int4(0, 0, 0, 0);
    int4 rb = (has1 && v < Kv) ? __ldcs(&wb[v]) : make_int4(0, 0, 0, 0);
    while (v < Kv) {
        int vn = v + kWarpSize;
        int4 na = (vn < Kv) ? __ldcs(&wa[vn]) : make_int4(0, 0, 0, 0);
        int4 nb = (has1 && vn < Kv) ? __ldcs(&wb[vn]) : make_int4(0, 0, 0, 0);
        const float2* xp = xshf + v * 9;
        float xv[16];
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            float2 f = xp[p];
            xv[2 * p] = f.x;
            xv[2 * p + 1] = f.y;
        }
        float wf[16];
        nvk_deq_bytes16<true>(ra, wf);
        #pragma unroll
        for (int t = 0; t < 16; t += 2) {
            acc0 += wf[t] * xv[t] + wf[t + 1] * xv[t + 1];
        }
        nvk_deq_bytes16<true>(rb, wf);
        #pragma unroll
        for (int t = 0; t < 16; t += 2) {
            acc1 += wf[t] * xv[t] + wf[t + 1] * xv[t + 1];
        }
        v = vn;
        ra = na;
        rb = nb;
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffff, acc0, offset);
        acc1 += __shfl_xor_sync(0xffffffff, acc1, offset);
    }
    if (lane == 0) {
        y[n0] = __float2bfloat16(acc0 * __ldg(&row_scale[n0]));
        if (has1) y[n1] = __float2bfloat16(acc1 * __ldg(&row_scale[n1]));
    }
}

template <int TM>
__global__ void gemv_e4m3_mk_xf32_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    extern __shared__ float2 xshf[];
    const int rowpad = (K >> 4) * 9;
    const int kp = K >> 1;
    for (int idx = threadIdx.x; idx < TM * kp; idx += kBlockDim) {
        int j = idx / kp;
        int p = idx - j * kp;
        int k0 = 2 * p;
        xshf[j * rowpad + (p >> 3) * 9 + (p & 7)] = make_float2(
            __bfloat162float(x[(size_t)j * K + k0]),
            __bfloat162float(x[(size_t)j * K + k0 + 1]));
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const int4* w16 = reinterpret_cast<const int4*>(wq + (size_t)n * K);
    int Kv = K / 16;
    float acc[TM];
    #pragma unroll
    for (int j = 0; j < TM; ++j) acc[j] = 0.0f;

    for (int v = lane; v < Kv; v += kWarpSize) {
        int4 raw = __ldg(&w16[v]);
        float wf[16];
        nvk_deq_bytes16<true>(raw, wf);
        const float2* base = xshf + v * 9;
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            const float2* xp = base + j * rowpad;
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 xv = xp[p];
                acc[j] += wf[2 * p] * xv.x + wf[2 * p + 1] * xv.y;
            }
        }
    }

    float rs = __ldg(&row_scale[n]);
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        float a = acc[j];
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, offset);
        }
        if (lane == 0) y[(size_t)j * N + n] = __float2bfloat16(a * rs);
    }
}

template <int TM>
__global__ void gemv_e4m3_mk_xf32_rows2_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    extern __shared__ float2 xshf[];
    const int rowpad = (K >> 4) * 9;
    const int kp = K >> 1;
    for (int idx = threadIdx.x; idx < TM * kp; idx += kBlockDim) {
        int j = idx / kp;
        int p = idx - j * kp;
        int k0 = 2 * p;
        xshf[j * rowpad + (p >> 3) * 9 + (p & 7)] = make_float2(
            __bfloat162float(x[(size_t)j * K + k0]),
            __bfloat162float(x[(size_t)j * K + k0 + 1]));
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n0 = blockIdx.x * (2 * kRowsPerBlock) + warp;
    int n1 = n0 + kRowsPerBlock;
    if (n0 >= N) return;
    bool has1 = n1 < N;

    const int4* wa = reinterpret_cast<const int4*>(wq + (size_t)n0 * K);
    const int4* wb = reinterpret_cast<const int4*>(wq + (size_t)n1 * K);
    int Kv = K / 16;
    float acc0[TM];
    float acc1[TM];
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        acc0[j] = 0.0f;
        acc1[j] = 0.0f;
    }

    int v = lane;
    int4 ra = (v < Kv) ? __ldcs(&wa[v]) : make_int4(0, 0, 0, 0);
    int4 rb = (has1 && v < Kv) ? __ldcs(&wb[v]) : make_int4(0, 0, 0, 0);
    while (v < Kv) {
        int vn = v + kWarpSize;
        int4 na = (vn < Kv) ? __ldcs(&wa[vn]) : make_int4(0, 0, 0, 0);
        int4 nb = (has1 && vn < Kv) ? __ldcs(&wb[vn]) : make_int4(0, 0, 0, 0);
        float wfa[16];
        float wfb[16];
        nvk_deq_bytes16<true>(ra, wfa);
        nvk_deq_bytes16<true>(rb, wfb);
        const float2* base = xshf + v * 9;
        #pragma unroll
        for (int j = 0; j < TM; ++j) {
            const float2* xp = base + j * rowpad;
            #pragma unroll
            for (int p = 0; p < 8; ++p) {
                float2 xv = xp[p];
                acc0[j] += wfa[2 * p] * xv.x + wfa[2 * p + 1] * xv.y;
                acc1[j] += wfb[2 * p] * xv.x + wfb[2 * p + 1] * xv.y;
            }
        }
        v = vn;
        ra = na;
        rb = nb;
    }

    float rs0 = __ldg(&row_scale[n0]);
    float rs1 = has1 ? __ldg(&row_scale[n1]) : 0.0f;
    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        float a0 = acc0[j];
        float a1 = acc1[j];
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            a0 += __shfl_xor_sync(0xffffffff, a0, offset);
            a1 += __shfl_xor_sync(0xffffffff, a1, offset);
        }
        if (lane == 0) {
            y[(size_t)j * N + n0] = __float2bfloat16(a0 * rs0);
            if (has1) y[(size_t)j * N + n1] = __float2bfloat16(a1 * rs1);
        }
    }
}

constexpr int kMkRows2MaxTmBeforeAccPlusDualDeqRegsSpill = 6;

template <int TM>
static int launch_gemv_e4m3_mk_xf32_rows2(
    cudaStream_t s,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K
) {
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(float2);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(
        optin, (const void*)gemv_e4m3_mk_xf32_rows2_kernel<TM>, smem);
    if (orc != 0) return orc;
    dim3 grid((unsigned)((N + 2 * kRowsPerBlock - 1) / (2 * kRowsPerBlock)));
    gemv_e4m3_mk_xf32_rows2_kernel<TM><<<grid, dim3(kBlockDim), smem, s>>>(
        wq, row_scale,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return (int)cudaGetLastError();
}

template <int TM>
static int launch_gemv_e4m3_mk_xf32_when_m_ge_2_pays_conversion_once_not_per_row(
    cudaStream_t s,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K
) {
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(float2);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(
        optin, (const void*)gemv_e4m3_mk_xf32_kernel<TM>, smem);
    if (orc != 0) return orc;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_e4m3_mk_xf32_kernel<TM><<<grid, dim3(kBlockDim), smem, s>>>(
        wq, row_scale,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return (int)cudaGetLastError();
}

template <int TM>
static int launch_gemv_e4m3_mk(
    cudaStream_t s,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K
) {
    if (TM >= 2) {
        size_t smem_f32 = (size_t)TM * (K >> 4) * 9 * sizeof(float2);
        if (smem_f32 <= (size_t)mk_smem_limit()) {
            if constexpr (TM <= kMkRows2MaxTmBeforeAccPlusDualDeqRegsSpill) {
                static const bool mk_rows2_off_env_nv_q38_e4m3_mk_r2_off =
                    getenv("NV_Q38_E4M3_MK_R2_OFF") != nullptr;
                if (!mk_rows2_off_env_nv_q38_e4m3_mk_r2_off) {
                    return launch_gemv_e4m3_mk_xf32_rows2<TM>(s, wq, row_scale, x, y, N, K);
                }
            }
            return launch_gemv_e4m3_mk_xf32_when_m_ge_2_pays_conversion_once_not_per_row<TM>(
                s, wq, row_scale, x, y, N, K);
        }
    }
    size_t smem = (size_t)TM * (K >> 4) * 9 * sizeof(__nv_bfloat162);
    static DynamicSmemOptin optin;
    int orc = raise_dynamic_smem_optin_never_lowering_it(
        optin, (const void*)gemv_e4m3_mk_kernel<TM>, smem);
    if (orc != 0) return orc;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_e4m3_mk_kernel<TM><<<grid, dim3(kBlockDim), smem, s>>>(
        wq, row_scale,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y), N, K);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gemv_e4m3_mk(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int M
) {
    if (N <= 0 || K <= 0 || M <= 0) return 0;
    if ((K & 15) != 0 || M > 16) return -1;
    size_t limit = (size_t)mk_smem_limit();
    size_t smem = (size_t)M * (K >> 4) * 9 * sizeof(__nv_bfloat162);
    if (limit == 0 || smem > limit) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    static const bool rows2_off_env_nv_q38_e4m3_r2_off =
        getenv("NV_Q38_E4M3_R2_OFF") != nullptr;
    if (M == 1 && !rows2_off_env_nv_q38_e4m3_r2_off) {
        size_t smem1 = (size_t)(K >> 4) * 9 * sizeof(float2);
        if (smem1 > (size_t)mk_smem_limit()) {
            size_t smem_bf = (size_t)(K >> 4) * 9 * sizeof(__nv_bfloat162);
            if (smem_bf <= (size_t)mk_smem_limit()) {
                return launch_gemv_e4m3_mk<1>(s, wq, row_scale, x, y, N, K);
            }
            return -1;
        }
        dim3 grid((unsigned)((N + 2 * kRowsPerBlock - 1) / (2 * kRowsPerBlock)));
        static DynamicSmemOptin optin_r2;
        int orc = raise_dynamic_smem_optin_never_lowering_it(
            optin_r2, (const void*)gemv_e4m3_m1_rows2_per_warp_kernel<false>, smem1);
        if (orc != 0) return orc;
        gemv_e4m3_m1_rows2_per_warp_kernel<false><<<grid, dim3(kBlockDim), smem1, s>>>(
            wq, row_scale,
            reinterpret_cast<const __nv_bfloat16*>(x),
            nullptr, nullptr,
            reinterpret_cast<__nv_bfloat16*>(y), N, K);
        return (int)cudaGetLastError();
    }
    switch (M) {
        case 1: return launch_gemv_e4m3_mk<1>(s, wq, row_scale, x, y, N, K);
        case 2: return launch_gemv_e4m3_mk<2>(s, wq, row_scale, x, y, N, K);
        case 3: return launch_gemv_e4m3_mk<3>(s, wq, row_scale, x, y, N, K);
        case 4: return launch_gemv_e4m3_mk<4>(s, wq, row_scale, x, y, N, K);
        case 5: return launch_gemv_e4m3_mk<5>(s, wq, row_scale, x, y, N, K);
        case 6: return launch_gemv_e4m3_mk<6>(s, wq, row_scale, x, y, N, K);
        case 7: return launch_gemv_e4m3_mk<7>(s, wq, row_scale, x, y, N, K);
        case 8: return launch_gemv_e4m3_mk<8>(s, wq, row_scale, x, y, N, K);
        case 9: return launch_gemv_e4m3_mk<9>(s, wq, row_scale, x, y, N, K);
        case 10: return launch_gemv_e4m3_mk<10>(s, wq, row_scale, x, y, N, K);
        case 11: return launch_gemv_e4m3_mk<11>(s, wq, row_scale, x, y, N, K);
        case 12: return launch_gemv_e4m3_mk<12>(s, wq, row_scale, x, y, N, K);
        case 13: return launch_gemv_e4m3_mk<13>(s, wq, row_scale, x, y, N, K);
        case 14: return launch_gemv_e4m3_mk<14>(s, wq, row_scale, x, y, N, K);
        case 15: return launch_gemv_e4m3_mk<15>(s, wq, row_scale, x, y, N, K);
        case 16: return launch_gemv_e4m3_mk<16>(s, wq, row_scale, x, y, N, K);
        default: return -1;
    }
}

__global__ void gemv_e4m3_m1_rows2_qkv_one_launch_kernel_row_math_matches_the_three_separate_rows2_launches_bitwise(
    const uint8_t* __restrict__ wq_q,
    const float* __restrict__ rs_q,
    const uint8_t* __restrict__ wq_k,
    const float* __restrict__ rs_k,
    const uint8_t* __restrict__ wq_v,
    const float* __restrict__ rs_v,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y_q,
    __nv_bfloat16* __restrict__ y_k,
    __nv_bfloat16* __restrict__ y_v,
    int n_q,
    int n_k,
    int n_v,
    int K
) {
    extern __shared__ float2 xshf[];
    const int kp = K >> 1;
    for (int idx = threadIdx.x; idx < kp; idx += kBlockDim) {
        int k0 = 2 * idx;
        float a = __bfloat162float(x[k0]);
        float b = __bfloat162float(x[k0 + 1]);
        xshf[(idx >> 3) * 9 + (idx & 7)] = make_float2(a, b);
    }
    __syncthreads();

    int blocks_q = n_q / (2 * kRowsPerBlock);
    int blocks_k = n_k / (2 * kRowsPerBlock);
    int seg_block = (int)blockIdx.x;
    const uint8_t* wq;
    const float* row_scale;
    __nv_bfloat16* y;
    int N;
    if (seg_block < blocks_q) {
        wq = wq_q;
        row_scale = rs_q;
        y = y_q;
        N = n_q;
    } else if (seg_block < blocks_q + blocks_k) {
        wq = wq_k;
        row_scale = rs_k;
        y = y_k;
        N = n_k;
        seg_block -= blocks_q;
    } else {
        wq = wq_v;
        row_scale = rs_v;
        y = y_v;
        N = n_v;
        seg_block -= blocks_q + blocks_k;
    }

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n0 = seg_block * (2 * kRowsPerBlock) + warp;
    int n1 = n0 + kRowsPerBlock;
    if (n0 >= N) return;
    bool has1 = n1 < N;

    const int4* wa = reinterpret_cast<const int4*>(wq + (size_t)n0 * K);
    const int4* wb = reinterpret_cast<const int4*>(wq + (size_t)n1 * K);
    int Kv = K / 16;
    float acc0 = 0.0f;
    float acc1 = 0.0f;

    int v = lane;
    int4 ra = (v < Kv) ? __ldcs(&wa[v]) : make_int4(0, 0, 0, 0);
    int4 rb = (has1 && v < Kv) ? __ldcs(&wb[v]) : make_int4(0, 0, 0, 0);
    while (v < Kv) {
        int vn = v + kWarpSize;
        int4 na = (vn < Kv) ? __ldcs(&wa[vn]) : make_int4(0, 0, 0, 0);
        int4 nb = (has1 && vn < Kv) ? __ldcs(&wb[vn]) : make_int4(0, 0, 0, 0);
        const float2* xp = xshf + v * 9;
        float xv[16];
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            float2 f = xp[p];
            xv[2 * p] = f.x;
            xv[2 * p + 1] = f.y;
        }
        float wf[16];
        nvk_deq_bytes16<true>(ra, wf);
        #pragma unroll
        for (int t = 0; t < 16; t += 2) {
            acc0 += wf[t] * xv[t] + wf[t + 1] * xv[t + 1];
        }
        nvk_deq_bytes16<true>(rb, wf);
        #pragma unroll
        for (int t = 0; t < 16; t += 2) {
            acc1 += wf[t] * xv[t] + wf[t + 1] * xv[t + 1];
        }
        v = vn;
        ra = na;
        rb = nb;
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_xor_sync(0xffffffff, acc0, offset);
        acc1 += __shfl_xor_sync(0xffffffff, acc1, offset);
    }
    if (lane == 0) {
        y[n0] = __float2bfloat16(acc0 * __ldg(&row_scale[n0]));
        if (has1) y[n1] = __float2bfloat16(acc1 * __ldg(&row_scale[n1]));
    }
}

extern "C" int nv_kernels_gemv_e4m3_qkv_one_m1(
    void* stream,
    const uint8_t* wq_q,
    const float* rs_q,
    const uint8_t* wq_k,
    const float* rs_k,
    const uint8_t* wq_v,
    const float* rs_v,
    const uint16_t* x,
    uint16_t* y_q,
    uint16_t* y_k,
    uint16_t* y_v,
    int n_q,
    int n_k,
    int n_v,
    int K
) {
    const int rows_per_launch_block = 2 * kRowsPerBlock;
    if (n_q <= 0 || n_k <= 0 || n_v <= 0 || K <= 0) return -1;
    if ((n_q % rows_per_launch_block) != 0 || (n_k % rows_per_launch_block) != 0 ||
        (n_v % rows_per_launch_block) != 0 || (K & 15) != 0) {
        return -1;
    }
    size_t smem1 = (size_t)(K >> 4) * 9 * sizeof(float2);
    if (smem1 > (size_t)mk_smem_limit()) return -1;
    static DynamicSmemOptin optin_qkv_one;
    int orc = raise_dynamic_smem_optin_never_lowering_it(
        optin_qkv_one,
        (const void*)gemv_e4m3_m1_rows2_qkv_one_launch_kernel_row_math_matches_the_three_separate_rows2_launches_bitwise,
        smem1);
    if (orc != 0) return orc;
    dim3 grid((unsigned)((n_q + n_k + n_v) / rows_per_launch_block));
    gemv_e4m3_m1_rows2_qkv_one_launch_kernel_row_math_matches_the_three_separate_rows2_launches_bitwise<<<
        grid, dim3(kBlockDim), smem1, (cudaStream_t)stream>>>(
        wq_q, rs_q, wq_k, rs_k, wq_v, rs_v,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<__nv_bfloat16*>(y_q),
        reinterpret_cast<__nv_bfloat16*>(y_k),
        reinterpret_cast<__nv_bfloat16*>(y_v),
        n_q, n_k, n_v, K);
    return (int)cudaGetLastError();
}

__global__ void gemv_e4m3_qkvz_conv_m1_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ conv_w,
    __nv_bfloat16* __restrict__ conv_state,
    __nv_bfloat16* __restrict__ mixed_out,
    __nv_bfloat16* __restrict__ z_out,
    int N,
    int K,
    int conv_dim,
    int K_c
) {
    extern __shared__ __nv_bfloat162 xshr[];
    const int kp = K >> 1;
    for (int p = threadIdx.x; p < kp; p += kBlockDim) {
        int k0 = 2 * p;
        xshr[(p >> 3) * 9 + (p & 7)] = __floats2bfloat162_rn(
            __bfloat162float(x[k0]),
            __bfloat162float(x[k0 + 1]));
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const int4* w16 = reinterpret_cast<const int4*>(wq + (size_t)n * K);
    int Kv = K / 16;
    float acc = 0.0f;

    for (int v = lane; v < Kv; v += kWarpSize) {
        int4 raw = __ldg(&w16[v]);
        float wf[16];
        nvk_deq_bytes16<true>(raw, wf);
        const __nv_bfloat162* xp = xshr + v * 9;
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            float2 xv = __bfloat1622float2(xp[p]);
            acc += wf[2 * p] * xv.x + wf[2 * p + 1] * xv.y;
        }
    }

    float rs = __ldg(&row_scale[n]);
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) {
        __nv_bfloat16 r = __float2bfloat16(acc * rs);
        if (n < conv_dim) {
            const __nv_bfloat16* w_row = conv_w + (size_t)n * K_c;
            __nv_bfloat16* s_row = conv_state + (size_t)n * (K_c - 1);
            mixed_out[n] = nvk_gdn_conv_step_silu(s_row, r, w_row, K_c);
            for (int i = 0; i < K_c - 2; ++i) {
                s_row[i] = s_row[i + 1];
            }
            s_row[K_c - 2] = r;
        } else {
            z_out[n - conv_dim] = r;
        }
    }
}

extern "C" int nv_kernels_gemv_e4m3_qkvz_conv_m1(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    const uint16_t* x,
    const uint16_t* conv_w,
    uint16_t* conv_state,
    uint16_t* mixed_out,
    uint16_t* z_out,
    int N,
    int K,
    int conv_dim,
    int K_c
) {
    if (N <= 0 || K <= 0) return -1;
    if ((K & 15) != 0) return -1;
    if (conv_dim <= 0 || conv_dim > N) return -1;
    if (K_c < 2 || K_c > NVK_GDN_CONV_MAX_K) return -1;
    size_t limit = (size_t)mk_smem_limit();
    size_t smem = (size_t)(K >> 4) * 9 * sizeof(__nv_bfloat162);
    if (limit == 0 || smem > limit) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    static DynamicSmemOptin optin_gemv_e4m3_qkvz_conv_m1_kernel;
    int orc_gemv_e4m3_qkvz_conv_m1_kernel = raise_dynamic_smem_optin_never_lowering_it(
        optin_gemv_e4m3_qkvz_conv_m1_kernel, (const void*)gemv_e4m3_qkvz_conv_m1_kernel, smem);
    if (orc_gemv_e4m3_qkvz_conv_m1_kernel != 0) return orc_gemv_e4m3_qkvz_conv_m1_kernel;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_e4m3_qkvz_conv_m1_kernel<<<grid, dim3(kBlockDim), smem, s>>>(
        wq, row_scale,
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(conv_w),
        reinterpret_cast<__nv_bfloat16*>(conv_state),
        reinterpret_cast<__nv_bfloat16*>(mixed_out),
        reinterpret_cast<__nv_bfloat16*>(z_out),
        N, K, conv_dim, K_c);
    return (int)cudaGetLastError();
}

__global__ void scale_rowcol_bf16_inplace_kernel(
    __nv_bfloat16* __restrict__ d,
    const float* __restrict__ row_scale_m,
    const float* __restrict__ col_scale_n,
    long long total,
    int n
) {
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        int i = (int)(idx / n);
        int j = (int)(idx - (long long)i * n);
        float v = __bfloat162float(d[idx]);
        d[idx] = __float2bfloat16(v * __ldg(&row_scale_m[i]) * __ldg(&col_scale_n[j]));
    }
}

extern "C" int nv_kernels_scale_rowcol_bf16(
    void* stream,
    uint16_t* d,
    const float* row_scale_m,
    const float* col_scale_n,
    int M,
    int N
) {
    if (M <= 0 || N <= 0) return -1;
    long long total = (long long)M * N;
    long long want = (total + 255) / 256;
    unsigned grid = (unsigned)(want < 65535LL ? want : 65535LL);
    scale_rowcol_bf16_inplace_kernel<<<grid, 256, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<__nv_bfloat16*>(d), row_scale_m, col_scale_n, total, N);
    return (int)cudaGetLastError();
}

template <bool kFp8>
__global__ void gemv_q8_qkvg_normed_kernel(
    const uint8_t* __restrict__ Wq,
    const float* __restrict__ Sq,
    const uint8_t* __restrict__ Wk,
    const float* __restrict__ Sk,
    const uint8_t* __restrict__ Wv,
    const float* __restrict__ Sv,
    const __nv_bfloat16* __restrict__ Wg,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __nv_bfloat16* __restrict__ yq,
    __nv_bfloat16* __restrict__ yk,
    __nv_bfloat16* __restrict__ yv,
    __nv_bfloat16* __restrict__ yg,
    int Nq,
    int Nk,
    int Nv,
    int Ng,
    int K
) {
    extern __shared__ __nv_bfloat162 xsg[];
    const int kp = K >> 1;
    float r = rstd[0];
    for (int p = threadIdx.x; p < kp; p += kBlockDim) {
        int k0 = 2 * p;
        float a = __bfloat162float(x[k0]) * r * __bfloat162float(wn[k0]);
        float b = __bfloat162float(x[k0 + 1]) * r * __bfloat162float(wn[k0 + 1]);
        xsg[(p >> 3) * 9 + (p & 7)] = __floats2bfloat162_rn(a, b);
    }
    __syncthreads();

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;

    const uint8_t* Wb = nullptr;
    const float* S = nullptr;
    __nv_bfloat16* y = nullptr;
    int local = n;
    if (n < Nq) {
        Wb = Wq;
        S = Sq;
        y = yq;
    } else if ((local -= Nq) < Nk) {
        Wb = Wk;
        S = Sk;
        y = yk;
    } else if ((local -= Nk) < Nv) {
        Wb = Wv;
        S = Sv;
        y = yv;
    } else if ((local -= Nv) < Ng) {
        const __nv_bfloat16* w_row = Wg + (size_t)local * K;
        const uint4* w4 = reinterpret_cast<const uint4*>(w_row);
        int Kg = K / 8;
        float gacc = 0.0f;
        for (int v = lane; v < Kg; v += kWarpSize) {
            uint4 pw = __ldg(&w4[v]);
            const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
            int pb = v * 4;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 wf = __bfloat1622float2(wp[j]);
                int p = pb + j;
                float2 xv = __bfloat1622float2(xsg[(p >> 3) * 9 + (p & 7)]);
                gacc += wf.x * xv.x + wf.y * xv.y;
            }
        }
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            gacc += __shfl_xor_sync(0xffffffff, gacc, offset);
        }
        if (lane == 0) yg[local] = __float2bfloat16(gacc);
        return;
    } else {
        return;
    }

    const int4* w16 = reinterpret_cast<const int4*>(Wb + (size_t)local * K);
    int Kv = K / 16;
    float acc = 0.0f;
    for (int v = lane; v < Kv; v += kWarpSize) {
        int4 raw = __ldg(&w16[v]);
        float wf[16];
        nvk_deq_bytes16<kFp8>(raw, wf);
        const __nv_bfloat162* xp = xsg + v * 9;
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            float2 xv = __bfloat1622float2(xp[p]);
            acc += wf[2 * p] * xv.x + wf[2 * p + 1] * xv.y;
        }
    }

    float rs = __ldg(&S[local]);
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[local] = __float2bfloat16(acc * rs);
}

extern "C" int nv_kernels_gemv_q8_qkvg_normed(
    void* stream,
    int fp8,
    const void* Wq,
    const float* Sq,
    const void* Wk,
    const float* Sk,
    const void* Wv,
    const float* Sv,
    const uint16_t* Wg,
    const uint16_t* x,
    const uint16_t* wn,
    const float* rstd,
    uint16_t* yq,
    uint16_t* yk,
    uint16_t* yv,
    uint16_t* yg,
    int Nq,
    int Nk,
    int Nv,
    int Ng,
    int K
) {
    if (Nq <= 0 || Nk <= 0 || Nv <= 0 || Ng < 0 || K <= 0) return -1;
    if ((K & 15) != 0 || K > kMaxSharedK) return -1;
    if (Wq == nullptr || Wk == nullptr || Wv == nullptr) return -1;
    if (Ng > 0 && Wg == nullptr) return -1;
    size_t smem = (size_t)(K >> 4) * 9 * sizeof(__nv_bfloat162);
    size_t limit = (size_t)mk_smem_limit();
    if (limit != 0 && smem > limit) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    int total = Nq + Nk + Nv + Ng;
    dim3 grid((unsigned)((total + kRowsPerBlock - 1) / kRowsPerBlock));
    if (fp8) {
        gemv_q8_qkvg_normed_kernel<true><<<grid, dim3(kBlockDim), smem, s>>>(
            reinterpret_cast<const uint8_t*>(Wq), Sq,
            reinterpret_cast<const uint8_t*>(Wk), Sk,
            reinterpret_cast<const uint8_t*>(Wv), Sv,
            reinterpret_cast<const __nv_bfloat16*>(Wg),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(wn),
            rstd,
            reinterpret_cast<__nv_bfloat16*>(yq),
            reinterpret_cast<__nv_bfloat16*>(yk),
            reinterpret_cast<__nv_bfloat16*>(yv),
            reinterpret_cast<__nv_bfloat16*>(yg),
            Nq, Nk, Nv, Ng, K);
    } else {
        gemv_q8_qkvg_normed_kernel<false><<<grid, dim3(kBlockDim), smem, s>>>(
            reinterpret_cast<const uint8_t*>(Wq), Sq,
            reinterpret_cast<const uint8_t*>(Wk), Sk,
            reinterpret_cast<const uint8_t*>(Wv), Sv,
            reinterpret_cast<const __nv_bfloat16*>(Wg),
            reinterpret_cast<const __nv_bfloat16*>(x),
            reinterpret_cast<const __nv_bfloat16*>(wn),
            rstd,
            reinterpret_cast<__nv_bfloat16*>(yq),
            reinterpret_cast<__nv_bfloat16*>(yk),
            reinterpret_cast<__nv_bfloat16*>(yv),
            reinterpret_cast<__nv_bfloat16*>(yg),
            Nq, Nk, Nv, Ng, K);
    }
    return (int)cudaGetLastError();
}

constexpr int kMaxGemmM = 16;

template <int M>
__global__ void gemm_bf16_mk_kernel(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const __nv_bfloat16* w_row = W + (size_t)n * K;
    int Kv = K / 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(w_row);
    const uint4* x4 = reinterpret_cast<const uint4*>(x);

    float acc[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) acc[m] = 0.0f;

    for (int v = lane; v < Kv; v += kWarpSize) {
        uint4 pw = __ldg(&w4[v]);
        const __nv_bfloat162* wp = reinterpret_cast<const __nv_bfloat162*>(&pw);
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            uint4 px = __ldg(&x4[(size_t)m * Kv + v]);
            const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(&px);
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float2 wf = __bfloat1622float2(wp[j]);
                float2 xf = __bfloat1622float2(xp[j]);
                acc[m] += wf.x * xf.x + wf.y * xf.y;
            }
        }
    }

    #pragma unroll
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, offset);
        }
        if (lane == 0) y[(size_t)m * N + n] = __float2bfloat16(a);
    }
}

__global__ void gemm_bf16_mk_scalar_kernel(
    const __nv_bfloat16* __restrict__ W,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    int N,
    int K,
    int M
) {
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;
    const __nv_bfloat16* w_row = W + (size_t)n * K;
    float acc[kMaxGemmM];
    for (int m = 0; m < kMaxGemmM; ++m) acc[m] = 0.0f;
    for (int k = lane; k < K; k += kWarpSize) {
        float wv = __bfloat162float(w_row[k]);
        for (int m = 0; m < M; ++m) {
            acc[m] += wv * __bfloat162float(x[(size_t)m * K + k]);
        }
    }
    for (int m = 0; m < M; ++m) {
        float a = acc[m];
        #pragma unroll
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            a += __shfl_xor_sync(0xffffffff, a, offset);
        }
        if (lane == 0) y[(size_t)m * N + n] = __float2bfloat16(a);
    }
}

extern "C" int nv_kernels_gemm_bf16_mk(
    void* stream,
    const uint16_t* W,
    const uint16_t* x,
    uint16_t* y,
    int N,
    int K,
    int M
) {
    if (N <= 0 || K <= 0 || M <= 0) return 0;
    if (M > kMaxGemmM) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    dim3 block(kBlockDim);
    const __nv_bfloat16* Wb = reinterpret_cast<const __nv_bfloat16*>(W);
    const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(x);
    __nv_bfloat16* yb = reinterpret_cast<__nv_bfloat16*>(y);
    if ((K & 7) != 0) {
        gemm_bf16_mk_scalar_kernel<<<grid, block, 0, s>>>(Wb, xb, yb, N, K, M);
        return (int)cudaGetLastError();
    }
    switch (M) {
        case 1: gemm_bf16_mk_kernel<1><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 2: gemm_bf16_mk_kernel<2><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 3: gemm_bf16_mk_kernel<3><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 4: gemm_bf16_mk_kernel<4><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 5: gemm_bf16_mk_kernel<5><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 6: gemm_bf16_mk_kernel<6><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 7: gemm_bf16_mk_kernel<7><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 8: gemm_bf16_mk_kernel<8><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 9: gemm_bf16_mk_kernel<9><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 10: gemm_bf16_mk_kernel<10><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 11: gemm_bf16_mk_kernel<11><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 12: gemm_bf16_mk_kernel<12><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 13: gemm_bf16_mk_kernel<13><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 14: gemm_bf16_mk_kernel<14><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        case 15: gemm_bf16_mk_kernel<15><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
        default: gemm_bf16_mk_kernel<16><<<grid, block, 0, s>>>(Wb, xb, yb, N, K); break;
    }
    return (int)cudaGetLastError();
}
