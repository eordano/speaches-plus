
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "nv_kernels.h"

namespace {

constexpr int kDequantBlockDim = 256;
constexpr long long kDequantMaxGridBlocksGridStrideCoversTheRest = 1 << 20;
constexpr int kRcRefusesNegativeDims = -3;
constexpr int kRcNvfp4RefusesKNotMultipleOf16BecauseThePackedLayoutHasNoTailBlockPath = -4;

#define NV_UE4M3_SUBNORMAL_STEP 0.001953125f

__device__ __forceinline__ float decode_e2m1_dev(uint8_t nib) {
    static const float kE2M1[16] = {
         0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
    };
    return kE2M1[nib & 0xF];
}

__device__ __forceinline__ float decode_ue4m3_ldexp_exact_dev(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_UE4M3_SUBNORMAL_STEP;
    return ldexpf(1.f + mant * 0.125f, biased - 7);
}

__device__ __forceinline__ int swizzled_scale_src(int m, int kb, int k_blocks) {
    int k_tiles = (k_blocks + 3) / 4;
    int m_tile = m / 128;
    int d2 = (m / 32) & 3;
    int d3 = m & 31;
    int k_tile = kb / 4;
    int d5 = kb & 3;
    return ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
}

__device__ __forceinline__ float decode_e4m3_signed_canonical_nan_dev(uint8_t b) {
    int mag = b & 0x7F;
    float v;
    if (mag == 0x7F) {
        v = __int_as_float(0x7FC00000);
    } else {
        int e = mag >> 3;
        float m = (float)(mag & 0x07);
        v = (e == 0) ? ldexpf(m, -9) : ldexpf(1.f + m * 0.125f, e - 7);
    }
    return (b & 0x80) ? -v : v;
}

__global__ void dequant_weight_nvfp4_swizzled_bf16_kernel(
    const uint8_t* __restrict__ w_packed,
    const uint8_t* __restrict__ w_scales_swizzled,
    __nv_bfloat16* __restrict__ out,
    float alpha,
    int n,
    int k
) {
    int bytes_per_row = k / 2;
    int k_blocks = k / 16;
    long long total = (long long)n * (long long)bytes_per_row;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        int r = (int)(idx / bytes_per_row);
        int j = (int)(idx % bytes_per_row);
        int kb = j / 8;
        uint8_t sb = w_scales_swizzled[swizzled_scale_src(r, kb, k_blocks)];
        float scale = decode_ue4m3_ldexp_exact_dev(sb) * alpha;
        uint8_t b = w_packed[idx];
        long long o = (long long)r * k + (long long)j * 2;
        out[o] = __float2bfloat16(decode_e2m1_dev(b & 0xF) * scale);
        out[o + 1] = __float2bfloat16(decode_e2m1_dev((b >> 4) & 0xF) * scale);
    }
}

__global__ void dequant_weight_e4m3_rowscale_bf16_kernel(
    const uint8_t* __restrict__ wq,
    const float* __restrict__ row_scale,
    __nv_bfloat16* __restrict__ out,
    int n,
    int k
) {
    long long total = (long long)n * (long long)k;
    long long stride = (long long)gridDim.x * blockDim.x;
    for (long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += stride) {
        int r = (int)(idx / k);
        float v = decode_e4m3_signed_canonical_nan_dev(wq[idx]);
        out[idx] = __float2bfloat16(v * row_scale[r]);
    }
}

int dequant_grid_blocks(long long total) {
    long long blocks = (total + kDequantBlockDim - 1) / kDequantBlockDim;
    if (blocks > kDequantMaxGridBlocksGridStrideCoversTheRest) {
        blocks = kDequantMaxGridBlocksGridStrideCoversTheRest;
    }
    return (int)blocks;
}

}

extern "C" int nv_kernels_dequant_weight_nvfp4_swizzled_to_bf16_rowmajor(
    void* stream,
    const uint8_t* w_packed,
    const uint8_t* w_scales_swizzled,
    uint16_t* out_bf16,
    float alpha,
    int n,
    int k
) {
    if (n < 0 || k < 0) return kRcRefusesNegativeDims;
    if (k % 16 != 0) {
        return kRcNvfp4RefusesKNotMultipleOf16BecauseThePackedLayoutHasNoTailBlockPath;
    }
    if (n == 0 || k == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    long long total = (long long)n * (long long)(k / 2);
    dequant_weight_nvfp4_swizzled_bf16_kernel<<<dequant_grid_blocks(total), kDequantBlockDim, 0, s>>>(
        w_packed,
        w_scales_swizzled,
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        alpha, n, k
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}

extern "C" int nv_kernels_dequant_weight_e4m3_rowscale_to_bf16_rowmajor(
    void* stream,
    const uint8_t* wq,
    const float* row_scale,
    uint16_t* out_bf16,
    int n,
    int k
) {
    if (n < 0 || k < 0) return kRcRefusesNegativeDims;
    if (n == 0 || k == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    long long total = (long long)n * (long long)k;
    dequant_weight_e4m3_rowscale_bf16_kernel<<<dequant_grid_blocks(total), kDequantBlockDim, 0, s>>>(
        wq,
        row_scale,
        reinterpret_cast<__nv_bfloat16*>(out_bf16),
        n, k
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
