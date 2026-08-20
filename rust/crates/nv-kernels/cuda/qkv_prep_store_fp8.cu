#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr float kFp8E4m3MaxMatchesQuantizeKvFp8 = 448.0f;
constexpr int kMaxHeadDimOneThreadPerElement = 1024;

__device__ __forceinline__ float qkvps_block_sum(float v, float* red) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int n_warps = ((int)blockDim.x + 31) >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < n_warps) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) red[0] = s;
    }
    __syncthreads();
    return red[0];
}

__device__ __forceinline__ float qkvps_block_max(float v, float* red) {
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int n_warps = ((int)blockDim.x + 31) >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < n_warps) ? red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) s = fmaxf(s, __shfl_xor_sync(0xffffffffu, s, o));
        if (lane == 0) red[0] = s;
    }
    __syncthreads();
    return red[0];
}

__global__ void qkv_norm_rope_kvstore_fp8_decode_kernel(
    const __nv_bfloat16* __restrict__ q_raw,
    const __nv_bfloat16* __restrict__ k_raw,
    const __nv_bfloat16* __restrict__ v_raw,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    const int* __restrict__ pos_dev,
    uint8_t* __restrict__ k_fp8_base,
    uint8_t* __restrict__ v_fp8_base,
    float* __restrict__ k_scales_base,
    float* __restrict__ v_scales_base,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ q_sig_out,
    int n_q,
    int n_kv,
    int hd,
    int q_row_stride,
    int rotary_dim,
    float eps
) {
    extern __shared__ float vals[];
    __shared__ float red[32];

    int h = blockIdx.x;
    int t = threadIdx.x;
    int pos = *pos_dev;
    int half = rotary_dim >> 1;
    bool is_q = h < n_q;
    bool is_k = !is_q && h < n_q + n_kv;
    bool is_v = !is_q && !is_k;
    int hk = is_q ? h : (is_k ? h - n_q : h - n_q - n_kv);

    float x;
    if (is_q) {
        x = __bfloat162float(q_raw[(size_t)h * q_row_stride + t]);
    } else if (is_k) {
        x = __bfloat162float(k_raw[(size_t)hk * hd + t]);
    } else {
        x = __bfloat162float(v_raw[(size_t)hk * hd + t]);
    }

    float y = x;
    if (!is_v) {
        float ss = qkvps_block_sum(x * x, red);
        float rstd = rsqrtf(ss / (float)hd + eps);
        const __nv_bfloat16* w = is_q ? q_norm_w : k_norm_w;
        y = __bfloat162float(__float2bfloat16(x * rstd * __bfloat162float(w[t])));
        vals[t] = y;
        __syncthreads();
        if (t < rotary_dim) {
            int i = (t < half) ? t : (t - half);
            float c = cos_tab[(size_t)pos * half + i];
            float s = sin_tab[(size_t)pos * half + i];
            float lo = vals[i];
            float hi = vals[i + half];
            y = (t < half) ? (lo * c - hi * s) : (lo * s + hi * c);
        }
        y = __bfloat162float(__float2bfloat16(y));
        __syncthreads();
    }

    if (is_q) {
        q_out[(size_t)h * hd + t] = __float2bfloat16(y);
        if (q_sig_out != nullptr) {
            float g = __bfloat162float(q_raw[(size_t)h * q_row_stride + hd + t]);
            q_sig_out[(size_t)h * hd + t] = __float2bfloat16(1.0f / (1.0f + expf(-g)));
        }
    } else {
        float amax = qkvps_block_max(fabsf(y), red);
        float scale = (amax > 0.0f) ? (amax / kFp8E4m3MaxMatchesQuantizeKvFp8) : 1.0f;
        float inv_scale = (amax > 0.0f) ? (kFp8E4m3MaxMatchesQuantizeKvFp8 / amax) : 1.0f;
        uint8_t* fp8_base = is_k ? k_fp8_base : v_fp8_base;
        float* scales_base = is_k ? k_scales_base : v_scales_base;
        size_t slot_head = (size_t)pos * n_kv + hk;
        if (t == 0) scales_base[slot_head] = scale;
        __nv_fp8_e4m3 enc = static_cast<__nv_fp8_e4m3>(y * inv_scale);
        fp8_base[slot_head * hd + t] = enc.__x;
    }
}

}

extern "C" int nv_kernels_qkv_norm_rope_kvstore_fp8_decode(
    void* stream,
    const uint16_t* q_raw,
    const uint16_t* k_raw,
    const uint16_t* v_raw,
    const uint16_t* q_norm_w,
    const uint16_t* k_norm_w,
    const float* cos_tab,
    const float* sin_tab,
    const int* pos_dev,
    uint8_t* k_fp8_base,
    uint8_t* v_fp8_base,
    float* k_scales_base,
    float* v_scales_base,
    uint16_t* q_out,
    uint16_t* q_sig_out,
    int n_q,
    int n_kv,
    int hd,
    int q_row_stride,
    int rotary_dim,
    float eps
) {
    if (n_q <= 0 || n_kv <= 0) return -1;
    if (hd < 32 || hd > kMaxHeadDimOneThreadPerElement || (hd & 31) != 0) return -1;
    if (rotary_dim < 2 || rotary_dim > hd || (rotary_dim & 1) != 0) return -1;
    if (q_row_stride != hd && q_row_stride != 2 * hd) return -1;
    if (q_sig_out != nullptr && q_row_stride != 2 * hd) return -1;
    unsigned grid = (unsigned)(n_q + 2 * n_kv);
    size_t smem = (size_t)hd * sizeof(float);
    qkv_norm_rope_kvstore_fp8_decode_kernel<<<grid, dim3((unsigned)hd), smem,
                                              (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_raw),
        reinterpret_cast<const __nv_bfloat16*>(k_raw),
        reinterpret_cast<const __nv_bfloat16*>(v_raw),
        reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
        reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
        cos_tab, sin_tab, pos_dev,
        k_fp8_base, v_fp8_base, k_scales_base, v_scales_base,
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(q_sig_out),
        n_q, n_kv, hd, q_row_stride, rotary_dim, eps);
    return (int)cudaGetLastError();
}
