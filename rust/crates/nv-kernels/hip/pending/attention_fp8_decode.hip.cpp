#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

#include "nv_kernels.h"
#include "nv_hip_wave.h"

namespace {

__device__ __forceinline__ float e4m3_ocp_to_float(uint8_t b) {
    uint32_t s = (uint32_t)(b & 0x80u) << 24;
    uint32_t e = (uint32_t)(b >> 3) & 0xFu;
    uint32_t m = (uint32_t)b & 0x7u;
    uint32_t out;
    if (e == 0u) {
        if (m == 0u) {
            out = s;
        } else {
            int k = 31 - __builtin_clz(m);
            out = s | ((uint32_t)(127 + k - 9) << 23) | ((m - (1u << k)) << (23 - k));
        }
    } else if (e == 0xFu && m == 0x7u) {
        out = s | 0x7FC00000u;
    } else {
        out = s | ((e + 120u) << 23) | (m << 20);
    }
    return __uint_as_float(out);
}

inline int wave_size_now() {
    static int cache[16] = {0};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) dev = 0;
    if (dev < 0 || dev >= 16) dev = 0;
    int w = cache[dev];
    if (w == 0) {
        w = nv_hip::host_wave_size(dev);
        if (w != 32 && w != 64) w = 64;
        cache[dev] = w;
    }
    return w;
}

template <int WAVE>
__device__ __forceinline__ float block_reduce_sum(
    float val, float* reduce_buf, int block_dim
) {
    int lane = threadIdx.x & (WAVE - 1);
    int warp = threadIdx.x / WAVE;
    int n_warps = block_dim / WAVE;

    __syncthreads();
    val = nv_hip::wave_sum<WAVE>(val);
    if (lane == 0) reduce_buf[warp] = val;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < n_warps) ? reduce_buf[lane] : 0.f;
        v = nv_hip::wave_sum<WAVE>(v);
        if (lane == 0) reduce_buf[0] = v;
    }
    __syncthreads();
    return reduce_buf[0];
}

template <int WAVE>
__device__ __forceinline__ float block_reduce_max(
    float val, float* reduce_buf, int block_dim
) {
    int lane = threadIdx.x & (WAVE - 1);
    int warp = threadIdx.x / WAVE;
    int n_warps = block_dim / WAVE;

    __syncthreads();
    val = nv_hip::wave_max<WAVE>(val);
    if (lane == 0) reduce_buf[warp] = val;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < n_warps) ? reduce_buf[lane] : -INFINITY;
        v = nv_hip::wave_max<WAVE>(v);
        if (lane == 0) reduce_buf[0] = v;
    }
    __syncthreads();
    return reduce_buf[0];
}

template <int HEAD_DIM, int WAVE>
__global__ void attention_fp8_decode_kernel(
    const __hip_bfloat16* __restrict__ q,
    const uint8_t*        __restrict__ k_fp8,
    const uint8_t*        __restrict__ v_fp8,
    const float*          __restrict__ k_scales,
    const float*          __restrict__ v_scales,
    __hip_bfloat16*       __restrict__ out,
    int n_kv,
    const int* __restrict__ n_total_dev,
    int max_total,
    int sliding_window,
    int gqa_group,
    float scaling
) {
    const int q_head = blockIdx.x;
    const int kv_head = q_head / gqa_group;
    const int tid = threadIdx.x;
    const int block_dim = HEAD_DIM;
    const int n_total = *n_total_dev;

    extern __shared__ float smem[];
    float* scores_smem = smem;
    float* q_smem      = smem + max_total;
    float* reduce_buf  = smem + max_total + HEAD_DIM;

    q_smem[tid] = __bfloat162float(q[q_head * HEAD_DIM + tid]);
    __syncthreads();

    const int sw = sliding_window;
    for (int i = 0; i < n_total; ++i) {
        bool masked = (sw > 0) && ((n_total - 1 - i) >= sw);

        float partial = 0.f;
        if (!masked) {
            uint8_t k_byte = k_fp8[((size_t)i * n_kv + kv_head) * HEAD_DIM + tid];
            float k_dec = e4m3_ocp_to_float(k_byte);
            float k_scale = k_scales[i * n_kv + kv_head];
            partial = q_smem[tid] * k_dec * k_scale;
        }
        float sum = block_reduce_sum<WAVE>(partial, reduce_buf, block_dim);
        if (tid == 0) {
            scores_smem[i] = masked ? -INFINITY : (sum * scaling);
        }
        __syncthreads();
    }

    float thread_max = -INFINITY;
    for (int i = tid; i < n_total; i += block_dim) {
        thread_max = fmaxf(thread_max, scores_smem[i]);
    }
    float max_score = block_reduce_max<WAVE>(thread_max, reduce_buf, block_dim);

    float thread_sum = 0.f;
    for (int i = tid; i < n_total; i += block_dim) {
        float e = expf(scores_smem[i] - max_score);
        scores_smem[i] = e;
        thread_sum += e;
    }
    float total = block_reduce_sum<WAVE>(thread_sum, reduce_buf, block_dim);

    float inv_total = (total > 0.f) ? (1.f / total) : 0.f;
    for (int i = tid; i < n_total; i += block_dim) {
        scores_smem[i] *= inv_total;
    }
    __syncthreads();

    float acc = 0.f;
    for (int i = 0; i < n_total; ++i) {
        float s = scores_smem[i];
        if (s == 0.f) continue;
        float v_scale = v_scales[i * n_kv + kv_head];
        uint8_t v_byte = v_fp8[((size_t)i * n_kv + kv_head) * HEAD_DIM + tid];
        float v_dec = e4m3_ocp_to_float(v_byte);
        acc += s * v_dec * v_scale;
    }
    out[q_head * HEAD_DIM + tid] = __float2bfloat16(acc);
}

}

extern "C" int nv_kernels_attention_fp8_decode(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    int n_q,
    int n_kv,
    int head_dim,
    const int* n_total_dev,
    int max_total,
    int sliding_window,
    float scaling
) {
    if (n_q <= 0 || n_kv <= 0 || head_dim <= 0 || max_total <= 0) return 0;
    if (n_q % n_kv != 0) return -1;
    if ((head_dim & (head_dim - 1)) != 0) return -3;
    const int wave = wave_size_now();
    if ((head_dim % wave) != 0) return -4;
    const int gqa_group = n_q / n_kv;
    hipStream_t s = static_cast<hipStream_t>(stream);

    const int n_warps = head_dim / wave;
    const size_t smem_bytes =
        ((size_t)max_total + (size_t)head_dim + (size_t)n_warps) * sizeof(float);

    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) dev = 0;
    int lds_cap = nv_hip::host_max_lds_bytes(dev);
    if (lds_cap <= 0) lds_cap = 65536;
    if (smem_bytes > (size_t)lds_cap) return -5;

    dim3 grid(static_cast<unsigned>(n_q));
    dim3 block(static_cast<unsigned>(head_dim));

    const __hip_bfloat16* qb = reinterpret_cast<const __hip_bfloat16*>(q);
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(out);

#define NV_AFD_LAUNCH(HD, W)                                                   \
    attention_fp8_decode_kernel<HD, W><<<grid, block, smem_bytes, s>>>(        \
        qb, k_fp8, v_fp8, k_scales, v_scales, ob,                              \
        n_kv, n_total_dev, max_total, sliding_window, gqa_group, scaling)

    if (wave == 64) {
        if (head_dim == 256) { NV_AFD_LAUNCH(256, 64); }
        else if (head_dim == 512) { NV_AFD_LAUNCH(512, 64); }
        else if (head_dim == 128) { NV_AFD_LAUNCH(128, 64); }
        else if (head_dim == 64) { NV_AFD_LAUNCH(64, 64); }
        else { return -2; }
    } else {
        if (head_dim == 256) { NV_AFD_LAUNCH(256, 32); }
        else if (head_dim == 512) { NV_AFD_LAUNCH(512, 32); }
        else if (head_dim == 128) { NV_AFD_LAUNCH(128, 32); }
        else if (head_dim == 64) { NV_AFD_LAUNCH(64, 32); }
        else { return -2; }
    }
#undef NV_AFD_LAUNCH

    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : static_cast<int>(e);
}
