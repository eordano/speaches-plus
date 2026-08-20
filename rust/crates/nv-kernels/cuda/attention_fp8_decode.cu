
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t b) {
    __half_raw hr = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(hr.x));
    return f;
}

__device__ __forceinline__ float block_reduce_sum(
    float val, float* reduce_buf, int block_dim
) {
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x >> 5;
    int n_warps = block_dim >> 5;

    #pragma unroll
    for (int off = kWarpSize >> 1; off > 0; off >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, off);
    }
    if (lane == 0) reduce_buf[warp] = val;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < n_warps) ? reduce_buf[lane] : 0.f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, off);
        }
        if (lane == 0) reduce_buf[0] = v;
    }
    __syncthreads();
    return reduce_buf[0];
}

__device__ __forceinline__ float block_reduce_max(
    float val, float* reduce_buf, int block_dim
) {
    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x >> 5;
    int n_warps = block_dim >> 5;

    #pragma unroll
    for (int off = kWarpSize >> 1; off > 0; off >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, off));
    }
    if (lane == 0) reduce_buf[warp] = val;
    __syncthreads();

    if (warp == 0) {
        float v = (lane < n_warps) ? reduce_buf[lane] : -INFINITY;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off));
        }
        if (lane == 0) reduce_buf[0] = v;
    }
    __syncthreads();
    return reduce_buf[0];
}

__device__ __forceinline__ int kv_slot(
    const int* __restrict__ block_table, int block_size, int logical
) {
    if (block_table == nullptr) return logical;
    int blk = logical / block_size;
    return block_table[blk] * block_size + (logical - blk * block_size);
}

template <int HEAD_DIM>
__global__ void attention_fp8_decode_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t*        __restrict__ k_fp8,
    const uint8_t*        __restrict__ v_fp8,
    const float*          __restrict__ k_scales,
    const float*          __restrict__ v_scales,
    __nv_bfloat16*        __restrict__ out,
    int n_kv,
    const int* __restrict__ n_total_dev,
    int max_total,
    int sliding_window,
    int gqa_group,
    float scaling,
    const int* __restrict__ block_table,
    int block_size
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
            int slot = kv_slot(block_table, block_size, i);
            uint8_t k_byte = k_fp8[((size_t)slot * n_kv + kv_head) * HEAD_DIM + tid];
            float k_dec = fp8_e4m3_to_float(k_byte);
            float k_scale = k_scales[slot * n_kv + kv_head];
            partial = q_smem[tid] * k_dec * k_scale;
        }
        float sum = block_reduce_sum(partial, reduce_buf, block_dim);
        if (tid == 0) {
            scores_smem[i] = masked ? -INFINITY : (sum * scaling);
        }
        __syncthreads();
    }

    float thread_max = -INFINITY;
    for (int i = tid; i < n_total; i += block_dim) {
        thread_max = fmaxf(thread_max, scores_smem[i]);
    }
    float max_score = block_reduce_max(thread_max, reduce_buf, block_dim);

    float thread_sum = 0.f;
    for (int i = tid; i < n_total; i += block_dim) {
        float e = expf(scores_smem[i] - max_score);
        scores_smem[i] = e;
        thread_sum += e;
    }
    __syncthreads();
    float total = block_reduce_sum(thread_sum, reduce_buf, block_dim);

    float inv_total = (total > 0.f) ? (1.f / total) : 0.f;
    for (int i = tid; i < n_total; i += block_dim) {
        scores_smem[i] *= inv_total;
    }
    __syncthreads();

    float acc = 0.f;
    for (int i = 0; i < n_total; ++i) {
        float s = scores_smem[i];
        if (s == 0.f) continue;
        int vslot = kv_slot(block_table, block_size, i);
        float v_scale = v_scales[vslot * n_kv + kv_head];
        uint8_t v_byte = v_fp8[((size_t)vslot * n_kv + kv_head) * HEAD_DIM + tid];
        float v_dec = fp8_e4m3_to_float(v_byte);
        acc += s * v_dec * v_scale;
    }
    out[q_head * HEAD_DIM + tid] = __float2bfloat16(acc);
}

template <int HEAD_DIM>
__global__ void attention_fp8_decode_gscores_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t*        __restrict__ k_fp8,
    const uint8_t*        __restrict__ v_fp8,
    const float*          __restrict__ k_scales,
    const float*          __restrict__ v_scales,
    __nv_bfloat16*        __restrict__ out,
    int n_kv,
    const int* __restrict__ n_total_dev,
    int max_total,
    int sliding_window,
    int gqa_group,
    float scaling,
    float* __restrict__ scores_gmem,
    const int* __restrict__ block_table,
    int block_size
) {
    const int q_head = blockIdx.x;
    const int kv_head = q_head / gqa_group;
    const int tid = threadIdx.x;
    const int block_dim = HEAD_DIM;
    const int n_total = *n_total_dev;

    extern __shared__ float smem[];
    float* scores_smem = scores_gmem + (size_t)q_head * (size_t)max_total;
    float* q_smem      = smem;
    float* reduce_buf  = smem + HEAD_DIM;

    q_smem[tid] = __bfloat162float(q[q_head * HEAD_DIM + tid]);
    __syncthreads();

    const int sw = sliding_window;
    for (int i = 0; i < n_total; ++i) {
        bool masked = (sw > 0) && ((n_total - 1 - i) >= sw);

        float partial = 0.f;
        if (!masked) {
            int slot = kv_slot(block_table, block_size, i);
            uint8_t k_byte = k_fp8[((size_t)slot * n_kv + kv_head) * HEAD_DIM + tid];
            float k_dec = fp8_e4m3_to_float(k_byte);
            float k_scale = k_scales[slot * n_kv + kv_head];
            partial = q_smem[tid] * k_dec * k_scale;
        }
        float sum = block_reduce_sum(partial, reduce_buf, block_dim);
        if (tid == 0) {
            scores_smem[i] = masked ? -INFINITY : (sum * scaling);
        }
        __syncthreads();
    }

    float thread_max = -INFINITY;
    for (int i = tid; i < n_total; i += block_dim) {
        thread_max = fmaxf(thread_max, scores_smem[i]);
    }
    float max_score = block_reduce_max(thread_max, reduce_buf, block_dim);

    float thread_sum = 0.f;
    for (int i = tid; i < n_total; i += block_dim) {
        float e = expf(scores_smem[i] - max_score);
        scores_smem[i] = e;
        thread_sum += e;
    }
    __syncthreads();
    float total = block_reduce_sum(thread_sum, reduce_buf, block_dim);

    float inv_total = (total > 0.f) ? (1.f / total) : 0.f;
    for (int i = tid; i < n_total; i += block_dim) {
        scores_smem[i] *= inv_total;
    }
    __syncthreads();

    float acc = 0.f;
    for (int i = 0; i < n_total; ++i) {
        float s = scores_smem[i];
        if (s == 0.f) continue;
        int vslot = kv_slot(block_table, block_size, i);
        float v_scale = v_scales[vslot * n_kv + kv_head];
        uint8_t v_byte = v_fp8[((size_t)vslot * n_kv + kv_head) * HEAD_DIM + tid];
        float v_dec = fp8_e4m3_to_float(v_byte);
        acc += s * v_dec * v_scale;
    }
    out[q_head * HEAD_DIM + tid] = __float2bfloat16(acc);
}

}

extern "C" int nv_kernels_attention_fp8_decode_paged(
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
    float scaling,
    const int* block_table,
    int block_size
) {
    if (n_q <= 0 || n_kv <= 0 || head_dim <= 0 || max_total <= 0) return 0;
    if (n_q % n_kv != 0) return -1;
    if ((head_dim & (head_dim - 1)) != 0) return -3;
    if ((head_dim % 32) != 0) return -4;
    const int gqa_group = n_q / n_kv;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    const int n_warps = head_dim / 32;
    const size_t smem_bytes =
        ((size_t)max_total + (size_t)head_dim + (size_t)n_warps) * sizeof(float);

    dim3 grid(static_cast<unsigned>(n_q));
    dim3 block(static_cast<unsigned>(head_dim));

    if (head_dim == 256) {
        attention_fp8_decode_kernel<256><<<grid, block, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            k_fp8, v_fp8, k_scales, v_scales,
            reinterpret_cast<__nv_bfloat16*>(out),
            n_kv, n_total_dev, max_total, sliding_window, gqa_group, scaling,
            block_table, block_size);
    } else if (head_dim == 512) {
        attention_fp8_decode_kernel<512><<<grid, block, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            k_fp8, v_fp8, k_scales, v_scales,
            reinterpret_cast<__nv_bfloat16*>(out),
            n_kv, n_total_dev, max_total, sliding_window, gqa_group, scaling,
            block_table, block_size);
    } else if (head_dim == 128) {
        attention_fp8_decode_kernel<128><<<grid, block, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            k_fp8, v_fp8, k_scales, v_scales,
            reinterpret_cast<__nv_bfloat16*>(out),
            n_kv, n_total_dev, max_total, sliding_window, gqa_group, scaling,
            block_table, block_size);
    } else if (head_dim == 64) {
        attention_fp8_decode_kernel<64><<<grid, block, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            k_fp8, v_fp8, k_scales, v_scales,
            reinterpret_cast<__nv_bfloat16*>(out),
            n_kv, n_total_dev, max_total, sliding_window, gqa_group, scaling,
            block_table, block_size);
    } else {
        return -2;
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
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
    return nv_kernels_attention_fp8_decode_paged(
        stream, q, k_fp8, v_fp8, k_scales, v_scales, out, n_q, n_kv, head_dim,
        n_total_dev, max_total, sliding_window, scaling, nullptr, 0);
}

extern "C" int nv_kernels_attention_fp8_decode_gscores(
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
    float scaling,
    float* scores_gmem
) {
    if (n_q <= 0 || n_kv <= 0 || head_dim <= 0 || max_total <= 0) return 0;
    if (n_q % n_kv != 0) return -1;
    if ((head_dim & (head_dim - 1)) != 0) return -3;
    if ((head_dim % 32) != 0) return -4;
    if (scores_gmem == nullptr) return -5;
    const int gqa_group = n_q / n_kv;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    const int n_warps = head_dim / 32;
    const size_t smem_bytes = ((size_t)head_dim + (size_t)n_warps) * sizeof(float);

    dim3 grid(static_cast<unsigned>(n_q));
    dim3 block(static_cast<unsigned>(head_dim));

    if (head_dim == 256) {
        attention_fp8_decode_gscores_kernel<256><<<grid, block, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            k_fp8, v_fp8, k_scales, v_scales,
            reinterpret_cast<__nv_bfloat16*>(out),
            n_kv, n_total_dev, max_total, sliding_window, gqa_group, scaling,
            scores_gmem, nullptr, 0);
    } else if (head_dim == 512) {
        attention_fp8_decode_gscores_kernel<512><<<grid, block, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            k_fp8, v_fp8, k_scales, v_scales,
            reinterpret_cast<__nv_bfloat16*>(out),
            n_kv, n_total_dev, max_total, sliding_window, gqa_group, scaling,
            scores_gmem, nullptr, 0);
    } else if (head_dim == 128) {
        attention_fp8_decode_gscores_kernel<128><<<grid, block, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            k_fp8, v_fp8, k_scales, v_scales,
            reinterpret_cast<__nv_bfloat16*>(out),
            n_kv, n_total_dev, max_total, sliding_window, gqa_group, scaling,
            scores_gmem, nullptr, 0);
    } else if (head_dim == 64) {
        attention_fp8_decode_gscores_kernel<64><<<grid, block, smem_bytes, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(q),
            k_fp8, v_fp8, k_scales, v_scales,
            reinterpret_cast<__nv_bfloat16*>(out),
            n_kv, n_total_dev, max_total, sliding_window, gqa_group, scaling,
            scores_gmem, nullptr, 0);
    } else {
        return -2;
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}
