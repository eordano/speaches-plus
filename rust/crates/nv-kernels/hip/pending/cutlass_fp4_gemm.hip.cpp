#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <stddef.h>

#include "nv_kernels.h"

namespace {

constexpr int kRocmNotImplemented = -1001;
constexpr int kRocmUnsupportedTileVariant = -3;

constexpr int kNvfp4BlockSize = 16;

__device__ __forceinline__ float nvfp4_decode_e2m1(uint8_t nib) {
    const float lut[16] = {
         0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
    };
    return lut[nib & 0xF];
}

__device__ __forceinline__ float nvfp4_decode_ue4m3(uint8_t b) {
    int exp_v = ((int)(b >> 3) & 0x0F);
    float mant = (float)(b & 0x07);
    if (exp_v == 0) return mant * 0.001953125f;
    return (1.f + mant / 8.f) * exp2f((float)(exp_v - 7));
}

}

namespace nvk_fp4_rocm {

__global__ void nvfp4_decode_to_bf16_prologue(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales,
    const float* __restrict__ global_sf,
    __hip_bfloat16* __restrict__ out,
    int64_t n_elems
) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_elems) return;
    uint8_t byte = packed[i >> 1];
    uint8_t nib = (i & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xF);
    float blk = nvfp4_decode_ue4m3(scales[i / kNvfp4BlockSize]);
    float g = global_sf ? global_sf[0] : 1.f;
    out[i] = __float2bfloat16(nvfp4_decode_e2m1(nib) * blk * g);
}

}

extern "C" int nv_kernels_cutlass_fp4_gemm_sm120_bf16(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k,
    void* workspace,
    size_t workspace_bytes,
    size_t* required_workspace
) {
    (void)stream; (void)a_fp4; (void)a_sf; (void)b_fp4; (void)b_sf;
    (void)global_sf; (void)d_bf16; (void)m; (void)n; (void)k;
    (void)workspace; (void)workspace_bytes;
    if (required_workspace) *required_workspace = 0;
    return kRocmNotImplemented;
}

extern "C" int nv_kernels_cutlass_fp4_gemm_sm120_bf16_streamk(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k,
    void* workspace,
    size_t workspace_bytes,
    size_t* required_workspace
) {
    (void)stream; (void)a_fp4; (void)a_sf; (void)b_fp4; (void)b_sf;
    (void)global_sf; (void)d_bf16; (void)m; (void)n; (void)k;
    (void)workspace; (void)workspace_bytes;
    if (required_workspace) *required_workspace = 0;
    return kRocmNotImplemented;
}

extern "C" int nv_kernels_cutlass_fp4_gemm_sm120_bf16_tiled(
    void* stream,
    const void* a_fp4,
    const void* a_sf,
    const void* b_fp4,
    const void* b_sf,
    const float* global_sf,
    void* d_bf16,
    int m, int n, int k,
    int tile,
    int stream_k,
    void* workspace,
    size_t workspace_bytes,
    size_t* required_workspace
) {
    (void)stream; (void)a_fp4; (void)a_sf; (void)b_fp4; (void)b_sf;
    (void)global_sf; (void)d_bf16; (void)m; (void)n; (void)k;
    (void)workspace; (void)workspace_bytes;
    if (required_workspace) *required_workspace = 0;
    int variant = tile * 2 + (stream_k ? 1 : 0);
    if (variant < 0 || variant > 5) return kRocmUnsupportedTileVariant;
    return kRocmNotImplemented;
}
