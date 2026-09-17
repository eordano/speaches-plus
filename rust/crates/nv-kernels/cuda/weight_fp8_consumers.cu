#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "nv_kernels.h"

namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float decode_e4m3_bf16_value(uint8_t b, float scale) {
    __half_raw raw = __nv_cvt_fp8_to_halfraw(b, __NV_E4M3);
    float v;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(v) : "h"(raw.x));
    return __bfloat162float(__float2bfloat16(v * scale));
}

template <int BLOCK>
__device__ __forceinline__ float block_sum(float v) {
    constexpr int kWarp = 32;
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
        float s = (lane < kWarps) ? warp_sums[lane] : 0.f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

__global__ void gather_rows_e4m3_bf16_kernel(
    const uint8_t* __restrict__ weight,
    const float* __restrict__ row_scale,
    const int32_t* __restrict__ token_ids,
    __nv_bfloat16* __restrict__ out,
    int tokens,
    int hidden,
    int vocab
) {
    int row = blockIdx.x;
    int token = token_ids[row];
    __nv_bfloat16* dst = out + (size_t)row * hidden;
    if (token < 0 || token >= vocab) {
        for (int h = threadIdx.x; h < hidden; h += blockDim.x) dst[h] = __float2bfloat16(0.f);
        return;
    }
    const uint8_t* src = weight + (size_t)token * hidden;
    float scale = row_scale[token];
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        dst[h] = __float2bfloat16(decode_e4m3_bf16_value(src[h], scale));
    }
}

__global__ void rmsnorm_e4m3_weight_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    __nv_bfloat16* __restrict__ out,
    size_t hidden,
    float eps
) {
    size_t row = blockIdx.x;
    const __nv_bfloat16* row_x = x + row * hidden;
    __nv_bfloat16* row_out = out + row * hidden;
    float local = 0.f;
    for (size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float v = __bfloat162float(row_x[i]);
        local += v * v;
    }
    float rms = rsqrtf(block_sum<kBlock>(local) / (float)hidden + eps);
    float scale = weight_scale[0];
    for (size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float w = decode_e4m3_bf16_value(weight[i], scale);
        row_out[i] = __float2bfloat16(__bfloat162float(row_x[i]) * rms * w);
    }
}

__global__ void rmsnorm_residual_e4m3_weight_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ residual,
    const uint8_t* __restrict__ weight,
    const float* __restrict__ weight_scale,
    __nv_bfloat16* __restrict__ out,
    size_t hidden,
    float eps
) {
    size_t row = blockIdx.x;
    const __nv_bfloat16* row_x = x + row * hidden;
    __nv_bfloat16* row_residual = residual + row * hidden;
    __nv_bfloat16* row_out = out + row * hidden;
    float local = 0.f;
    for (size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float sum = __bfloat162float(row_x[i]) + __bfloat162float(row_residual[i]);
        __nv_bfloat16 stored = __float2bfloat16(sum);
        row_residual[i] = stored;
        local += sum * sum;
    }
    float rms = rsqrtf(block_sum<kBlock>(local) / (float)hidden + eps);
    float scale = weight_scale[0];
    for (size_t i = threadIdx.x; i < hidden; i += blockDim.x) {
        float w = decode_e4m3_bf16_value(weight[i], scale);
        row_out[i] = __float2bfloat16(__bfloat162float(row_residual[i]) * rms * w);
    }
}

}

extern "C" int nv_kernels_gather_rows_e4m3_bf16(
    void* stream,
    const uint8_t* weight,
    const float* row_scale,
    const int32_t* token_ids,
    uint16_t* out_bf16,
    int tokens,
    int hidden,
    int vocab
) {
    if (tokens < 0 || hidden < 0 || vocab < 0) return -3;
    if (tokens == 0 || hidden == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    gather_rows_e4m3_bf16_kernel<<<tokens, kBlock, 0, s>>>(
        weight, row_scale, token_ids, reinterpret_cast<__nv_bfloat16*>(out_bf16),
        tokens, hidden, vocab);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_rmsnorm_e4m3_weight_bf16(
    void* stream,
    const uint16_t* x_bf16,
    const uint8_t* weight,
    const float* weight_scale,
    uint16_t* out_bf16,
    size_t batch,
    size_t hidden,
    float eps
) {
    if (batch == 0 || hidden == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rmsnorm_e4m3_weight_bf16_kernel<<<batch, kBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16), weight, weight_scale,
        reinterpret_cast<__nv_bfloat16*>(out_bf16), hidden, eps);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_rmsnorm_residual_e4m3_weight_bf16(
    void* stream,
    const uint16_t* x_bf16,
    uint16_t* residual_bf16,
    const uint8_t* weight,
    const float* weight_scale,
    uint16_t* out_bf16,
    size_t batch,
    size_t hidden,
    float eps
) {
    if (batch == 0 || hidden == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rmsnorm_residual_e4m3_weight_bf16_kernel<<<batch, kBlock, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<__nv_bfloat16*>(residual_bf16), weight, weight_scale,
        reinterpret_cast<__nv_bfloat16*>(out_bf16), hidden, eps);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
