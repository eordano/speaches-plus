
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
#include "nvk_grid.cuh"

namespace {

constexpr float kFp8E4m3Max = 448.0f;

constexpr int kAngleTable = 0;
constexpr int kAngleF32 = 1;
constexpr int kAngleF64 = 2;

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

__device__ __forceinline__ int paged_slot(
    const int* __restrict__ block_table, int block_size, int logical
) {
    int blk = logical / block_size;
    int off = logical - blk * block_size;
    return block_table[blk] * block_size + off;
}

__global__ void quantize_kv_fp8_paged_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    uint8_t* __restrict__ x_fp8_base,
    float* __restrict__ scales_base,
    const int* __restrict__ start_dev,
    const int* __restrict__ block_table,
    int block_size,
    int n_tokens,
    int n_kv,
    int head_dim
) {
    int token = blockIdx.x;
    int kv_head = blockIdx.y;
    if (token >= n_tokens || kv_head >= n_kv) return;

    int start = *start_dev;
    int logical = start + token;
    int slot = paged_slot(block_table, block_size, logical);
    int base_src = (token * n_kv + kv_head) * head_dim;
    int base_dst = (slot * n_kv + kv_head) * head_dim;
    int tid = threadIdx.x;

    float local_max = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = __bfloat162float(x_bf16[base_src + d]);
        local_max = fmaxf(local_max, fabsf(v));
    }
    local_max = warp_reduce_max(local_max);
    __shared__ float warp_max[32];
    int warp_id = tid >> 5;
    int lane = tid & 31;
    if (lane == 0) warp_max[warp_id] = local_max;
    __syncthreads();
    if (warp_id == 0) {
        int n_warps = (blockDim.x + 31) >> 5;
        float v = (lane < n_warps) ? warp_max[lane] : 0.0f;
        v = warp_reduce_max(v);
        if (lane == 0) warp_max[0] = v;
    }
    __syncthreads();
    float amax = warp_max[0];

    float scale = (amax > 0.0f) ? (amax / kFp8E4m3Max) : 1.0f;
    float inv_scale = (amax > 0.0f) ? (kFp8E4m3Max / amax) : 1.0f;
    if (tid == 0) {
        scales_base[slot * n_kv + kv_head] = scale;
    }
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = __bfloat162float(x_bf16[base_src + d]);
        __nv_fp8_e4m3 enc = static_cast<__nv_fp8_e4m3>(v * inv_scale);
        x_fp8_base[base_dst + d] = enc.__x;
    }
}

__global__ void dequantize_kv_fp8_paged_kernel(
    const uint8_t* __restrict__ x_fp8_base,
    const float* __restrict__ scales_base,
    __nv_bfloat16* __restrict__ x_bf16_out,
    const int* __restrict__ block_table,
    int block_size,
    int len,
    int n_kv,
    int head_dim
) {
    int token = blockIdx.x;
    int kv_head = blockIdx.y;
    if (token >= len || kv_head >= n_kv) return;

    int slot = paged_slot(block_table, block_size, token);
    int base_src = (slot * n_kv + kv_head) * head_dim;
    int base_dst = (token * n_kv + kv_head) * head_dim;
    float scale = scales_base[slot * n_kv + kv_head];
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        __nv_fp8_e4m3 packed;
        packed.__x = x_fp8_base[base_src + d];
        float v = static_cast<float>(packed) * scale;
        x_bf16_out[base_dst + d] = __float2bfloat16(v);
    }
}

__global__ void derive_v_from_k_fp8_paged_kernel(
    const uint8_t* __restrict__ k_fp8_base,
    const float* __restrict__ k_scales_base,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    const float* __restrict__ inv_freq,
    __nv_bfloat16* __restrict__ v_out,
    const int* __restrict__ block_table,
    int block_size,
    int len,
    int n_kv,
    int head_dim,
    int rope_angles,
    int angle_mode,
    int pos_base,
    float w_inv
) {
    int token = blockIdx.x;
    int kv_head = blockIdx.y;
    if (token >= len || kv_head >= n_kv) return;
    int d = threadIdx.x;
    if (d >= head_dim) return;

    int half = head_dim >> 1;
    int slot = paged_slot(block_table, block_size, token);
    const uint8_t* kp = k_fp8_base + ((size_t)slot * n_kv + kv_head) * head_dim;
    float ks = k_scales_base[slot * n_kv + kv_head];

    int j = (d < half) ? d : (d - half);
    float c = 1.0f;
    float s = 0.0f;
    int pos = pos_base + token;
    if (j < rope_angles) {
        if (angle_mode == kAngleTable) {
            c = cos_tab[(size_t)token * half + j];
            s = sin_tab[(size_t)token * half + j];
        } else if (angle_mode == kAngleF32) {
            __sincosf((float)pos * inv_freq[j], &s, &c);
        } else {
            double th = (double)pos * (double)inv_freq[j];
            sincosf((float)(th - 6.283185307179586 * floor(th / 6.283185307179586)), &s, &c);
        }
    }

    __nv_fp8_e4m3 elo, ehi;
    elo.__x = kp[j];
    ehi.__x = kp[j + half];
    float klo = static_cast<float>(elo) * ks;
    float khi = static_cast<float>(ehi) * ks;

    float v = (d < half) ? (klo * c + khi * s) : (khi * c - klo * s);
    v_out[((size_t)token * n_kv + kv_head) * head_dim + d] =
        __float2bfloat16(v * w_inv);
}

__global__ void copy_kv_block_fp8_kernel(
    const uint8_t* __restrict__ fp8_base,
    const float* __restrict__ scales_base,
    uint8_t* __restrict__ fp8_dst_base,
    float* __restrict__ scales_dst_base,
    int src_block,
    int dst_block,
    int block_size,
    int n_kv,
    int head_dim
) {
    int slot_in_block = blockIdx.x;
    int kv_head = blockIdx.y;
    if (slot_in_block >= block_size || kv_head >= n_kv) return;

    int src_slot = src_block * block_size + slot_in_block;
    int dst_slot = dst_block * block_size + slot_in_block;
    int src_base = (src_slot * n_kv + kv_head) * head_dim;
    int dst_base = (dst_slot * n_kv + kv_head) * head_dim;
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        fp8_dst_base[dst_base + d] = fp8_base[src_base + d];
    }
    if (tid == 0) {
        scales_dst_base[dst_slot * n_kv + kv_head] =
            scales_base[src_slot * n_kv + kv_head];
    }
}

}

extern "C" int nv_kernels_quantize_kv_fp8_paged(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* x_fp8_base,
    float* scales_base,
    const int* start_dev,
    const int* block_table,
    int block_size,
    int n_tokens,
    int n_kv,
    int head_dim
) {
    if (n_tokens <= 0 || n_kv <= 0 || head_dim <= 0 || block_size <= 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = head_dim;
    if (block > 512) block = 512;
    if (block < 32) block = 32;
    dim3 grid(static_cast<unsigned>(n_tokens), static_cast<unsigned>(n_kv));
    quantize_kv_fp8_paged_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        x_fp8_base, scales_base, start_dev, block_table,
        block_size, n_tokens, n_kv, head_dim
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_dequantize_kv_fp8_paged(
    void* stream,
    const uint8_t* x_fp8_base,
    const float* scales_base,
    uint16_t* x_bf16_out,
    const int* block_table,
    int block_size,
    int len,
    int n_kv,
    int head_dim
) {
    if (len <= 0 || n_kv <= 0 || head_dim <= 0 || block_size <= 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = head_dim;
    if (block > 512) block = 512;
    if (block < 32) block = 32;
    dim3 grid(static_cast<unsigned>(len), static_cast<unsigned>(n_kv));
    dequantize_kv_fp8_paged_kernel<<<grid, block, 0, s>>>(
        x_fp8_base, scales_base,
        reinterpret_cast<__nv_bfloat16*>(x_bf16_out),
        block_table, block_size, len, n_kv, head_dim
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_derive_v_from_k_fp8_paged(
    void* stream,
    const uint8_t* k_fp8_base,
    const float* k_scales_base,
    const float* cos_tab,
    const float* sin_tab,
    const float* inv_freq,
    uint16_t* v_bf16_out,
    const int* block_table,
    int block_size,
    int len,
    int n_kv,
    int head_dim,
    int rope_angles,
    int angle_mode,
    int pos_base,
    float w_inv
) {
    if (len <= 0 || n_kv <= 0 || head_dim <= 0 || block_size <= 0) return 0;
    if ((head_dim & 1) != 0) return -1;
    if (head_dim > 1024) return -2;
    if (rope_angles < 0 || rope_angles > (head_dim >> 1)) return -3;
    if (angle_mode < kAngleTable || angle_mode > kAngleF64) return -4;
    if (angle_mode == kAngleTable && (cos_tab == nullptr || sin_tab == nullptr)) return -5;
    if (angle_mode != kAngleTable && inv_freq == nullptr) return -6;
    if (pos_base < 0) return -7;
    if (n_kv > 65535) return NVK_ERR_GRID_AXIS;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(static_cast<unsigned>(len), static_cast<unsigned>(n_kv));
    derive_v_from_k_fp8_paged_kernel<<<grid, head_dim, 0, s>>>(
        k_fp8_base, k_scales_base, cos_tab, sin_tab, inv_freq,
        reinterpret_cast<__nv_bfloat16*>(v_bf16_out),
        block_table, block_size, len, n_kv, head_dim, rope_angles,
        angle_mode, pos_base, w_inv
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_copy_kv_block_fp8(
    void* stream,
    const uint8_t* fp8_base,
    const float* scales_base,
    uint8_t* fp8_dst_base,
    float* scales_dst_base,
    int src_block,
    int dst_block,
    int block_size,
    int n_kv,
    int head_dim
) {
    if (block_size <= 0 || n_kv <= 0 || head_dim <= 0) return 0;
    if (src_block == dst_block) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = head_dim;
    if (block > 512) block = 512;
    if (block < 32) block = 32;
    dim3 grid(static_cast<unsigned>(block_size), static_cast<unsigned>(n_kv));
    copy_kv_block_fp8_kernel<<<grid, block, 0, s>>>(
        fp8_base, scales_base, fp8_dst_base, scales_dst_base,
        src_block, dst_block, block_size, n_kv, head_dim
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}
