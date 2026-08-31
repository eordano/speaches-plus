#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include "nv_kernels.h"
#include "nvk_pdl.cuh"

__device__ inline float softplus_safe(float x) {
    if (x > 20.0f) return x;
    if (x < -20.0f) return expf(x);
    return log1pf(expf(x));
}

__device__ inline float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void gdn_gating_kernel_bf16(const __nv_bfloat16* __restrict__ a,
                                       const __nv_bfloat16* __restrict__ b,
                                       const __nv_bfloat16* __restrict__ A_log,
                                       const __nv_bfloat16* __restrict__ dt_bias,
                                       float* __restrict__ g_out,
                                       __nv_bfloat16* __restrict__ beta_out,
                                       size_t tokens,
                                       size_t num_heads) {
    NVK_PDL_PROLOG();
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = tokens * num_heads;
    if (idx >= total) return;
    size_t h = idx % num_heads;

    float a_v   = __bfloat162float(a[idx]);
    float b_v   = __bfloat162float(b[idx]);
    float bias  = __bfloat162float(dt_bias[h]);
    float a_log = __bfloat162float(A_log[h]);

    float sp = softplus_safe(a_v + bias);
    float g  = -__expf(a_log) * sp;
    float beta = sigmoidf(b_v);

    g_out[idx]    = g;
    beta_out[idx] = __float2bfloat16(beta);
    NVK_PDL_EPILOG();
}

__global__ void gdn_gating_kernel_f32(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      const float* __restrict__ A_log,
                                      const float* __restrict__ dt_bias,
                                      float* __restrict__ g_out,
                                      float* __restrict__ beta_out,
                                      size_t tokens,
                                      size_t num_heads) {
    NVK_PDL_PROLOG();
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = tokens * num_heads;
    if (idx >= total) return;
    size_t h = idx % num_heads;

    float sp   = softplus_safe(a[idx] + dt_bias[h]);
    float g    = -__expf(A_log[h]) * sp;
    float beta = sigmoidf(b[idx]);
    g_out[idx]    = g;
    beta_out[idx] = beta;
}

extern "C" int nv_kernels_gdn_gating_bf16(
    void* stream,
    const uint16_t* a,
    const uint16_t* b,
    const uint16_t* A_log,
    const uint16_t* dt_bias,
    float* g_out,
    uint16_t* beta_out,
    size_t tokens,
    size_t num_heads
) {
    cudaStream_t s = (cudaStream_t)stream;
    size_t total = tokens * num_heads;
    if (total == 0) return 0;
    constexpr int BLOCK = 256;
    int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)grid), dim3(BLOCK), 0, s);
        cudaLaunchKernelEx(&cfg, gdn_gating_kernel_bf16,
            reinterpret_cast<const __nv_bfloat16*>(a),
            reinterpret_cast<const __nv_bfloat16*>(b),
            reinterpret_cast<const __nv_bfloat16*>(A_log),
            reinterpret_cast<const __nv_bfloat16*>(dt_bias),
            g_out,
            reinterpret_cast<__nv_bfloat16*>(beta_out),
            tokens, num_heads);
        return (int)cudaGetLastError();
    }
    gdn_gating_kernel_bf16<<<grid, BLOCK, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(a),
        reinterpret_cast<const __nv_bfloat16*>(b),
        reinterpret_cast<const __nv_bfloat16*>(A_log),
        reinterpret_cast<const __nv_bfloat16*>(dt_bias),
        g_out,
        reinterpret_cast<__nv_bfloat16*>(beta_out),
        tokens, num_heads);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gdn_gating_f32(
    void* stream,
    const float* a,
    const float* b,
    const float* A_log,
    const float* dt_bias,
    float* g_out,
    float* beta_out,
    size_t tokens,
    size_t num_heads
) {
    cudaStream_t s = (cudaStream_t)stream;
    size_t total = tokens * num_heads;
    if (total == 0) return 0;
    constexpr int BLOCK = 256;
    int grid = (int)((total + BLOCK - 1) / BLOCK);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)grid), dim3(BLOCK), 0, s);
        cudaLaunchKernelEx(&cfg, gdn_gating_kernel_f32,
            a, b, A_log, dt_bias, g_out, beta_out, tokens, num_heads);
        return (int)cudaGetLastError();
    }
    gdn_gating_kernel_f32<<<grid, BLOCK, 0, s>>>(
        a, b, A_log, dt_bias, g_out, beta_out, tokens, num_heads);
    return (int)cudaGetLastError();
}
