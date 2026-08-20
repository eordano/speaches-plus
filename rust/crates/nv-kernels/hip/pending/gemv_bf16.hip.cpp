#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#include "nv_hip_wave.h"

namespace {

constexpr int kRowLanes = 32;
constexpr int kRowsPerBlock = 8;
constexpr int kBlockDim = kRowLanes * kRowsPerBlock;
constexpr int kMaxSharedK = 4096;

__device__ __forceinline__ float row_group_sum(float v) {
    #pragma unroll
    for (int offset = kRowLanes / 2; offset > 0; offset >>= 1) {
        v += __shfl_xor(v, offset, kRowLanes);
    }
    return v;
}

__device__ __forceinline__ float ubyte_lane_f32(unsigned u, int i) {
    return (float)((u >> (8 * i)) & 0xFFu);
}

int max_lds_bytes() {
    static int cached = -1;
    if (cached < 0) {
        int dev = 0;
        cached = (hipGetDevice(&dev) == hipSuccess) ? nv_hip::host_max_lds_bytes(dev) : 0;
    }
    return cached;
}

template <bool kUseShared>
__global__ void gemv_bf16_kernel(
    const __hip_bfloat16* __restrict__ W,
    const __hip_bfloat16* __restrict__ x,
    __hip_bfloat16* __restrict__ y,
    int N,
    int K
) {
    __shared__ float xs[kUseShared ? kMaxSharedK : 1];
    if (kUseShared) {
        for (int k = threadIdx.x; k < K; k += kBlockDim) {
            xs[k] = __bfloat162float(x[k]);
        }
        __syncthreads();
    }

    int lane = threadIdx.x & (kRowLanes - 1);
    int warp = threadIdx.x / kRowLanes;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const __hip_bfloat16* w_row = W + (size_t)n * K;
    int Kv = K / 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(w_row);

    float acc = 0.0f;
    for (int v = lane; v < Kv; v += kRowLanes) {
        uint4 pw = w4[v];
        const __hip_bfloat162* wp = reinterpret_cast<const __hip_bfloat162*>(&pw);
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

    acc = row_group_sum(acc);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

__global__ void gemv_bf16_scalar_kernel(
    const __hip_bfloat16* __restrict__ W,
    const __hip_bfloat16* __restrict__ x,
    __hip_bfloat16* __restrict__ y,
    int N,
    int K
) {
    int lane = threadIdx.x & (kRowLanes - 1);
    int warp = threadIdx.x / kRowLanes;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;
    const __hip_bfloat16* w_row = W + (size_t)n * K;
    float acc = 0.0f;
    for (int k = lane; k < K; k += kRowLanes) {
        acc += __bfloat162float(w_row[k]) * __bfloat162float(x[k]);
    }
    acc = row_group_sum(acc);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

__global__ void gemv_bf16_normed_kernel(
    const __hip_bfloat16* __restrict__ W,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __hip_bfloat16* __restrict__ y,
    int N,
    int K
) {
    __shared__ float xs[kMaxSharedK];
    float r = rstd[0];
    for (int k = threadIdx.x; k < K; k += kBlockDim) {
        xs[k] = __bfloat162float(x[k]) * r * __bfloat162float(wn[k]);
    }
    __syncthreads();

    int lane = threadIdx.x & (kRowLanes - 1);
    int warp = threadIdx.x / kRowLanes;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const __hip_bfloat16* w_row = W + (size_t)n * K;
    int Kv = K / 8;
    const uint4* w4 = reinterpret_cast<const uint4*>(w_row);

    float acc = 0.0f;
    for (int v = lane; v < Kv; v += kRowLanes) {
        uint4 pw = w4[v];
        const __hip_bfloat162* wp = reinterpret_cast<const __hip_bfloat162*>(&pw);
        int kb = v * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float2 wf = __bfloat1622float2(wp[j]);
            acc += wf.x * xs[kb + 2 * j] + wf.y * xs[kb + 2 * j + 1];
        }
    }

    acc = row_group_sum(acc);
    if (lane == 0) y[n] = __float2bfloat16(acc);
}

__global__ void rowquant_i8_kernel(
    const __hip_bfloat16* __restrict__ w,
    int8_t* __restrict__ wq,
    float* __restrict__ row_scale,
    int N,
    int K
) {
    int n = blockIdx.x;
    if (n >= N) return;
    const __hip_bfloat16* row = w + (size_t)n * K;
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
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __hip_bfloat16* __restrict__ y,
    int N,
    int K
) {

    __shared__ float xs[kMaxSharedK + kMaxSharedK / 16];
    float r = rstd[0];
    for (int k = threadIdx.x; k < K; k += kBlockDim) {
        xs[(k >> 4) * 17 + (k & 15)] = __bfloat162float(x[k]) * r * __bfloat162float(wn[k]);
    }
    __syncthreads();

    int lane = threadIdx.x & (kRowLanes - 1);
    int warp = threadIdx.x / kRowLanes;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const int8_t* w_row = wq + (size_t)n * K;
    int Kv = K / 16;
    const int4* w16 = reinterpret_cast<const int4*>(w_row);

    float acc = 0.0f;
    for (int v = lane; v < Kv; v += kRowLanes) {
        int4 raw = w16[v];
        const unsigned* wu = reinterpret_cast<const unsigned*>(&raw);
        const float* xp = xs + v * 17;
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned u = wu[t] ^ 0x80808080u;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float f = ubyte_lane_f32(u, i) - 128.0f;
                acc += f * xp[4 * t + i];
            }
        }
    }

    acc = row_group_sum(acc);
    if (lane == 0) y[n] = __float2bfloat16(acc * row_scale[n]);
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
    rowquant_i8_kernel<<<(unsigned)N, 256, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(w), wq, row_scale, N, K);
    return (int)hipGetLastError();
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
    hipStream_t s = (hipStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_i8_normed_kernel<<<grid, dim3(kBlockDim), 0, s>>>(
        wq, row_scale,
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<const __hip_bfloat16*>(wn),
        rstd,
        reinterpret_cast<__hip_bfloat16*>(y), N, K);
    return (int)hipGetLastError();
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
    if ((K & 7) != 0 || K > kMaxSharedK) return -1;
    hipStream_t s = (hipStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    dim3 block(kBlockDim);
    gemv_bf16_normed_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(W),
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<const __hip_bfloat16*>(wn),
        rstd,
        reinterpret_cast<__hip_bfloat16*>(y), N, K);
    return (int)hipGetLastError();
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
    hipStream_t s = (hipStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    dim3 block(kBlockDim);
    if ((K & 7) == 0 && K <= kMaxSharedK) {
        gemv_bf16_kernel<true><<<grid, block, 0, s>>>(
            reinterpret_cast<const __hip_bfloat16*>(W),
            reinterpret_cast<const __hip_bfloat16*>(x),
            reinterpret_cast<__hip_bfloat16*>(y), N, K);
    } else if ((K & 7) == 0) {
        gemv_bf16_kernel<false><<<grid, block, 0, s>>>(
            reinterpret_cast<const __hip_bfloat16*>(W),
            reinterpret_cast<const __hip_bfloat16*>(x),
            reinterpret_cast<__hip_bfloat16*>(y), N, K);
    } else {
        gemv_bf16_scalar_kernel<<<grid, block, 0, s>>>(
            reinterpret_cast<const __hip_bfloat16*>(W),
            reinterpret_cast<const __hip_bfloat16*>(x),
            reinterpret_cast<__hip_bfloat16*>(y), N, K);
    }
    return (int)hipGetLastError();
}

__global__ void gemv_i8_normed_mk_kernel(
    const int8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    const __hip_bfloat16* __restrict__ x,
    const __hip_bfloat16* __restrict__ wn,
    const float* __restrict__ rstd,
    __hip_bfloat16* __restrict__ y,
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

    int lane = threadIdx.x & (kRowLanes - 1);
    int warp = threadIdx.x / kRowLanes;
    int n = blockIdx.x * kRowsPerBlock + warp;
    if (n >= N) return;

    const int8_t* w_row = wq + (size_t)n * K;
    int Kv = K / 16;
    const int4* w16 = reinterpret_cast<const int4*>(w_row);

    float acc[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.0f;

    for (int v = lane; v < Kv; v += kRowLanes) {
        int4 raw = w16[v];
        const unsigned* wu = reinterpret_cast<const unsigned*>(&raw);
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned u = wu[t] ^ 0x80808080u;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float f = ubyte_lane_f32(u, i) - 128.0f;
                int koff = v * 17 + 4 * t + i;
                for (int j = 0; j < M; ++j) {
                    acc[j] += f * xsd[j * rowpad + koff];
                }
            }
        }
    }

    float rs = row_scale[n];
    for (int j = 0; j < M; ++j) {
        float a = row_group_sum(acc[j]);
        if (lane == 0) y[(size_t)j * N + n] = __float2bfloat16(a * rs);
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
    if ((K & 15) != 0 || M > 8) return -1;
    size_t smem = (size_t)M * (K >> 4) * 17 * sizeof(float);
    int lds_cap = max_lds_bytes();
    if (lds_cap <= 0 || smem > (size_t)lds_cap) return -3;
    hipStream_t s = (hipStream_t)stream;
    dim3 grid((unsigned)((N + kRowsPerBlock - 1) / kRowsPerBlock));
    gemv_i8_normed_mk_kernel<<<grid, dim3(kBlockDim), smem, s>>>(
        wq, row_scale,
        reinterpret_cast<const __hip_bfloat16*>(x),
        reinterpret_cast<const __hip_bfloat16*>(wn),
        rstd,
        reinterpret_cast<__hip_bfloat16*>(y), N, K, M);
    return (int)hipGetLastError();
}
