
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>

#include "nvk_pdl.cuh"

namespace {

constexpr float kFp8E4m3Max = 448.0f;

__device__ __forceinline__ float warp_reduce_max(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other = __shfl_xor_sync(0xFFFFFFFFu, v, offset);
        v = fmaxf(v, other);
    }
    return v;
}

__global__ void quantize_kv_fp8_kernel(
    const __nv_bfloat16* __restrict__ x_bf16,
    uint8_t* __restrict__ x_fp8_base,
    float* __restrict__ scales_base,
    const int* __restrict__ start_dev,
    int n_tokens,
    int n_kv,
    int head_dim,
    int ring
) {
    NVK_PDL_PROLOG();

    int token = blockIdx.x;
    int kv_head = blockIdx.y;
    if (token >= n_tokens || kv_head >= n_kv) return;

    int start = *start_dev;
    int slot = start + token;
    if (ring > 0) slot = slot % ring;
    int base_src = (token * n_kv + kv_head) * head_dim;
    int base_dst = (slot * n_kv + kv_head) * head_dim;
    int tid = threadIdx.x;

    float local_max = 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float v = __bfloat162float(x_bf16[base_src + d]);
        float a = fabsf(v);
        if (a > local_max) local_max = a;
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

    NVK_PDL_EPILOG();
}

__global__ void dequantize_kv_fp8_kernel(
    const uint8_t* __restrict__ x_fp8,
    const float* __restrict__ scales,
    __nv_bfloat16* __restrict__ x_bf16,
    int start,
    int n_tokens,
    int n_kv,
    int head_dim,
    int ring
) {
    int token = blockIdx.x;
    int kv_head = blockIdx.y;
    if (token >= n_tokens || kv_head >= n_kv) return;

    int slot = start + token;
    if (ring > 0) slot = slot % ring;
    int base = (slot * n_kv + kv_head) * head_dim;
    int obase = (token * n_kv + kv_head) * head_dim;
    float scale = scales[slot * n_kv + kv_head];
    int tid = threadIdx.x;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        __nv_fp8_e4m3 packed;
        packed.__x = x_fp8[base + d];
        float v = static_cast<float>(packed) * scale;
        x_bf16[obase + d] = __float2bfloat16(v);
    }
}

}

extern "C" int nv_kernels_quantize_kv_fp8(
    void* stream,
    const uint16_t* x_bf16,
    uint8_t* x_fp8_base,
    float* scales_base,
    const int* start_dev,
    int n_tokens,
    int n_kv,
    int head_dim,
    int ring
) {
    if (n_tokens <= 0 || n_kv <= 0 || head_dim <= 0) return 0;
    if (ring < 0) return -3;
    cudaStream_t s = static_cast<cudaStream_t>(stream);

    int block = head_dim;
    if (block > 512) block = 512;
    if (block < 32) block = 32;
    dim3 grid(static_cast<unsigned>(n_tokens), static_cast<unsigned>(n_kv));
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(static_cast<unsigned>(block)), 0, s);
        cudaLaunchKernelEx(
            &cfg, quantize_kv_fp8_kernel,
            reinterpret_cast<const __nv_bfloat16*>(x_bf16),
            x_fp8_base, scales_base, start_dev,
            n_tokens, n_kv, head_dim, ring
        );
    } else {
        quantize_kv_fp8_kernel<<<grid, block, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_bf16),
            x_fp8_base, scales_base, start_dev,
            n_tokens, n_kv, head_dim, ring
        );
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_dequantize_kv_fp8(
    void* stream,
    const uint8_t* x_fp8,
    const float* scales,
    uint16_t* x_bf16,
    int start,
    int n_tokens,
    int n_kv,
    int head_dim,
    int ring
) {
    if (n_tokens <= 0 || n_kv <= 0 || head_dim <= 0) return 0;
    if (ring < 0 || start < 0) return -3;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    int block = head_dim;
    if (block > 512) block = 512;
    if (block < 32) block = 32;
    dim3 grid(static_cast<unsigned>(n_tokens), static_cast<unsigned>(n_kv));
    dequantize_kv_fp8_kernel<<<grid, block, 0, s>>>(
        x_fp8, scales,
        reinterpret_cast<__nv_bfloat16*>(x_bf16),
        start, n_tokens, n_kv, head_dim, ring
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}
