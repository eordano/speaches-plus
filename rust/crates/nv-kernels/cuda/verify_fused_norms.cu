#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "nvk_grid.cuh"

namespace verifyfused {

constexpr int kBlock = 256;
constexpr int kWarp = 32;
constexpr int kMaxHD = 512;
constexpr float kFp8Max = 448.0f;

__inline__ __device__ float warp_max(float x) {
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}

template <int BLOCK>
__device__ inline float block_sum(float v) {
    constexpr int kWarps = BLOCK / kWarp;
    __shared__ float warp_sums[kWarps];
    __shared__ float total;
    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
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

__device__ inline float row_rms_bf16(const __nv_bfloat16* row, int hd, float eps) {
    float local = 0.0f;
    for (int i = threadIdx.x; i < hd; i += kBlock) {
        float v = __bfloat162float(row[i]);
        local += v * v;
    }
    float sum = block_sum<kBlock>(local);
    return rsqrtf(sum / (float)hd + eps);
}

__global__ void verify_qkv_prep_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    long long qkv_stride,
    long long q_off,
    long long k_off,
    long long v_off,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const __nv_bfloat16* __restrict__ vw,
    float eps,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int32_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ q_out,
    uint8_t* __restrict__ kc,
    uint8_t* __restrict__ vc,
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
    const int32_t* __restrict__ n_committed,
    int K, int NQ, int NKV, int HD, int ring
) {
    __shared__ __nv_bfloat16 s_a[kMaxHD];
    __shared__ __nv_bfloat16 s_b[kMaxHD];
    __shared__ float wmK[kBlock / kWarp];
    __shared__ float wmV[kBlock / kWarp];

    int h = blockIdx.x;
    int token = blockIdx.y;
    if (token >= K) return;
    int tid = threadIdx.x;
    int half = HD / 2;
    int32_t pos = positions[token];
    const float* cos_row = cos_tbl + (size_t)pos * half;
    const float* sin_row = sin_tbl + (size_t)pos * half;

    if (h < NQ) {
        const __nv_bfloat16* row = qkv + (size_t)token * qkv_stride + q_off + (size_t)h * HD;
        float rms = row_rms_bf16(row, HD, eps);
        for (int i = tid; i < HD; i += kBlock) {
            float v = __bfloat162float(row[i]) * rms * __bfloat162float(qw[i]);
            s_a[i] = __float2bfloat16(v);
        }
        __syncthreads();
        __nv_bfloat16* orow = q_out + ((size_t)token * NQ + h) * HD;
        if (tid < half) {
            float a = __bfloat162float(s_a[tid]);
            float b = __bfloat162float(s_a[tid + half]);
            float c = cos_row[tid];
            float s = sin_row[tid];
            orow[tid] = __float2bfloat16(a * c - b * s);
            orow[tid + half] = __float2bfloat16(__fmaf_rn(a, s, __fmul_rn(b, c)));
        }
        return;
    }

    int kvh = h - NQ;
    if (kvh >= NKV) return;
    if (ring > 0 && token + ring < K) return;
    const __nv_bfloat16* krow = qkv + (size_t)token * qkv_stride + k_off + (size_t)kvh * HD;
    const __nv_bfloat16* vrow = qkv + (size_t)token * qkv_stride + v_off + (size_t)kvh * HD;

    float rms_k = row_rms_bf16(krow, HD, eps);
    for (int i = tid; i < HD; i += kBlock) {
        float v = __bfloat162float(krow[i]) * rms_k * __bfloat162float(kw[i]);
        s_a[i] = __float2bfloat16(v);
    }
    float rms_v = row_rms_bf16(vrow, HD, eps);
    for (int i = tid; i < HD; i += kBlock) {
        float v = __bfloat162float(vrow[i]) * rms_v * __bfloat162float(vw[i]);
        s_b[i] = __float2bfloat16(v);
    }
    __syncthreads();

    if (tid < half) {
        float a = __bfloat162float(s_a[tid]);
        float b = __bfloat162float(s_a[tid + half]);
        float c = cos_row[tid];
        float s = sin_row[tid];
        s_a[tid] = __float2bfloat16(a * c - b * s);
        s_a[tid + half] = __float2bfloat16(__fmaf_rn(a, s, __fmul_rn(b, c)));
    }
    __syncthreads();

    float lmK = 0.0f, lmV = 0.0f;
    for (int d = tid; d < HD; d += kBlock) {
        lmK = fmaxf(lmK, fabsf(__bfloat162float(s_a[d])));
        lmV = fmaxf(lmV, fabsf(__bfloat162float(s_b[d])));
    }
    lmK = warp_max(lmK);
    lmV = warp_max(lmV);
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { wmK[warp] = lmK; wmV[warp] = lmV; }
    __syncthreads();
    if (warp == 0) {
        constexpr int nw = kBlock / kWarp;
        float vK = (lane < nw) ? wmK[lane] : 0.0f;
        float vV = (lane < nw) ? wmV[lane] : 0.0f;
        vK = warp_max(vK); vV = warp_max(vV);
        if (lane == 0) { wmK[0] = vK; wmV[0] = vV; }
    }
    __syncthreads();
    float amaxK = wmK[0], amaxV = wmV[0];
    float invK = (amaxK > 0.0f) ? (kFp8Max / amaxK) : 1.0f;
    float invV = (amaxV > 0.0f) ? (kFp8Max / amaxV) : 1.0f;

    int slot = n_committed[0] + token;
    if (ring > 0) slot = slot % ring;
    if (tid == 0) {
        k_scale[(size_t)slot * NKV + kvh] = (amaxK > 0.0f) ? (amaxK / kFp8Max) : 1.0f;
        v_scale[(size_t)slot * NKV + kvh] = (amaxV > 0.0f) ? (amaxV / kFp8Max) : 1.0f;
    }
    size_t base_dst = ((size_t)slot * NKV + kvh) * HD;
    for (int d = tid; d < HD; d += kBlock) {
        __nv_fp8_e4m3 ek = static_cast<__nv_fp8_e4m3>(__bfloat162float(s_a[d]) * invK);
        __nv_fp8_e4m3 ev = static_cast<__nv_fp8_e4m3>(__bfloat162float(s_b[d]) * invV);
        kc[base_dst + d] = ek.__x;
        vc[base_dst + d] = ev.__x;
    }
}

__global__ void rmsnorm2_residual_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ w1,
    const __nv_bfloat16* __restrict__ w2,
    __nv_bfloat16* __restrict__ sum_out,
    __nv_bfloat16* __restrict__ normed_out,
    size_t hidden,
    float eps
) {
    __shared__ float scratch[kBlock];
    __shared__ float row_rms;

    size_t row = blockIdx.x;
    const __nv_bfloat16* row_x = x + row * hidden;
    const __nv_bfloat16* row_res = residual + row * hidden;
    __nv_bfloat16* row_sum = sum_out + row * hidden;
    __nv_bfloat16* row_out = normed_out + row * hidden;

    float local1 = 0.0f;
    for (size_t i = threadIdx.x; i < hidden; i += kBlock) {
        float v = __bfloat162float(row_x[i]);
        local1 += v * v;
    }
    float sum1 = block_sum<kBlock>(local1);
    float rms1 = rsqrtf(sum1 / (float)hidden + eps);

    float local2 = 0.0f;
    for (size_t i = threadIdx.x; i < hidden; i += kBlock) {
        float t = __bfloat162float(row_x[i]) * rms1 * __bfloat162float(w1[i]);
        __nv_bfloat16 tb = __float2bfloat16(t);
        float s = __bfloat162float(tb) + __bfloat162float(row_res[i]);
        row_sum[i] = __float2bfloat16(s);
        local2 += s * s;
    }
    scratch[threadIdx.x] = local2;
    __syncthreads();
    for (int stride = kBlock / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        row_rms = rsqrtf(scratch[0] / (float)hidden + eps);
    }
    __syncthreads();

    float rms2 = row_rms;
    for (size_t i = threadIdx.x; i < hidden; i += kBlock) {
        float s = __bfloat162float(row_sum[i]);
        float w = __bfloat162float(w2[i]);
        row_out[i] = __float2bfloat16(s * rms2 * w);
    }
}

__global__ void rmsnorm_residual_scale_kernel_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ out,
    size_t hidden,
    float eps,
    float scale
) {
    size_t row = blockIdx.x;
    const __nv_bfloat16* row_x = x + row * hidden;
    const __nv_bfloat16* row_res = residual + row * hidden;
    __nv_bfloat16* row_out = out + row * hidden;

    float local = 0.0f;
    for (size_t i = threadIdx.x; i < hidden; i += kBlock) {
        float v = __bfloat162float(row_x[i]);
        local += v * v;
    }
    float sum = block_sum<kBlock>(local);
    float rms = rsqrtf(sum / (float)hidden + eps);

    for (size_t i = threadIdx.x; i < hidden; i += kBlock) {
        float v = __bfloat162float(row_x[i]) * rms * __bfloat162float(w[i]);
        __nv_bfloat16 nb = __float2bfloat16(v);
        float av = __bfloat162float(row_res[i]);
        float bv = __bfloat162float(nb);
        row_out[i] = __float2bfloat16((av + bv) * scale);
    }
}

}

extern "C" int nv_kernels_verify_qkv_prep(
    void* stream,
    const uint16_t* qkv,
    long long qkv_stride,
    long long q_off,
    long long k_off,
    long long v_off,
    const uint16_t* qw,
    const uint16_t* kw,
    const uint16_t* vw,
    float eps,
    const float* cos_tbl,
    const float* sin_tbl,
    const int32_t* positions,
    uint16_t* q_out,
    uint8_t* kc,
    uint8_t* vc,
    float* k_scale,
    float* v_scale,
    const int32_t* n_committed,
    int K, int NQ, int NKV, int HD, int ring
) {
    if (K <= 0 || NQ <= 0 || NKV <= 0 || HD <= 0) return 0;
    if (HD > verifyfused::kMaxHD || (HD & 1) != 0 || HD / 2 > verifyfused::kBlock) return -1;
    if (ring < 0) return -3;
    if (K > 65535) return NVK_ERR_GRID_AXIS;
    dim3 grid((unsigned)(NQ + NKV), (unsigned)K);
    verifyfused::verify_qkv_prep_kernel<<<grid, verifyfused::kBlock, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv),
        qkv_stride, q_off, k_off, v_off,
        reinterpret_cast<const __nv_bfloat16*>(qw),
        reinterpret_cast<const __nv_bfloat16*>(kw),
        reinterpret_cast<const __nv_bfloat16*>(vw),
        eps, cos_tbl, sin_tbl, positions,
        reinterpret_cast<__nv_bfloat16*>(q_out),
        kc, vc, k_scale, v_scale, n_committed,
        K, NQ, NKV, HD, ring);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_rmsnorm2_residual_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* residual,
    const uint16_t* w1,
    const uint16_t* w2,
    uint16_t* sum_out,
    uint16_t* normed_out,
    size_t batch,
    size_t hidden,
    float eps
) {
    if (batch == 0 || hidden == 0) return 0;
    verifyfused::rmsnorm2_residual_kernel_bf16<<<(unsigned)batch, verifyfused::kBlock, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(w1),
        reinterpret_cast<const __nv_bfloat16*>(w2),
        reinterpret_cast<__nv_bfloat16*>(sum_out),
        reinterpret_cast<__nv_bfloat16*>(normed_out),
        hidden, eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_rmsnorm_residual_scale_bf16(
    void* stream,
    const uint16_t* x,
    const uint16_t* residual,
    const uint16_t* w,
    uint16_t* out,
    size_t batch,
    size_t hidden,
    float eps,
    float scale
) {
    if (batch == 0 || hidden == 0) return 0;
    verifyfused::rmsnorm_residual_scale_kernel_bf16<<<(unsigned)batch, verifyfused::kBlock, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        reinterpret_cast<const __nv_bfloat16*>(residual),
        reinterpret_cast<const __nv_bfloat16*>(w),
        reinterpret_cast<__nv_bfloat16*>(out),
        hidden, eps, scale);
    return (int)cudaGetLastError();
}
