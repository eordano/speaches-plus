
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

#include "nvk_pdl.cuh"

namespace {

__global__ void residual_add_scale_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16*        __restrict__ y,
    float scale,
    size_t n
) {
    NVK_PDL_PROLOG();

    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float av = __bfloat162float(a[idx]);
    float bv = __bfloat162float(b[idx]);
    y[idx] = __float2bfloat16((av + bv) * scale);

    NVK_PDL_EPILOG();
}

__global__ void scale_bf16_kernel(
    __nv_bfloat16* __restrict__ y,
    float scale,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(y[idx]);
    y[idx] = __float2bfloat16(v * scale);
}

__global__ void scale_out_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16*       __restrict__ y,
    float scale,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(x[idx]);
    y[idx] = __float2bfloat16(v * scale);
}

}

extern "C" int nv_kernels_residual_add_scale_bf16(
    void* stream,
    const uint16_t* a_bf16,
    const uint16_t* b_bf16,
    uint16_t* y_bf16,
    float scale,
    size_t n
) {
    if (n == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int block = 256;
    const size_t grid = (n + block - 1) / block;
    if (grid > (size_t)0x7FFFFFFF) return -1;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3((unsigned)grid), dim3(block), 0, s);
        cudaLaunchKernelEx(
            &cfg, residual_add_scale_bf16_kernel,
            reinterpret_cast<const __nv_bfloat16*>(a_bf16),
            reinterpret_cast<const __nv_bfloat16*>(b_bf16),
            reinterpret_cast<__nv_bfloat16*>(y_bf16),
            scale,
            n
        );
    } else {
        residual_add_scale_bf16_kernel<<<(unsigned)grid, block, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(a_bf16),
            reinterpret_cast<const __nv_bfloat16*>(b_bf16),
            reinterpret_cast<__nv_bfloat16*>(y_bf16),
            scale,
            n
        );
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_scale_inplace_bf16(
    void* stream,
    uint16_t* y_bf16,
    float scale,
    size_t n
) {
    if (n == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int block = 256;
    const size_t grid = (n + block - 1) / block;
    if (grid > (size_t)0x7FFFFFFF) return -1;
    scale_bf16_kernel<<<(unsigned)grid, block, 0, s>>>(
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        scale,
        n
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

extern "C" int nv_kernels_scale_out_bf16(
    void* stream,
    const uint16_t* x_bf16,
    uint16_t* y_bf16,
    float scale,
    size_t n
) {
    if (n == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int block = 256;
    const size_t grid = (n + block - 1) / block;
    if (grid > (size_t)0x7FFFFFFF) return -1;
    scale_out_bf16_kernel<<<(unsigned)grid, block, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<__nv_bfloat16*>(y_bf16),
        scale,
        n
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

__global__ void tanh_softcap_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ x,
    float*               __restrict__ y,
    float cap,
    float inv_cap,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float v = __bfloat162float(x[idx]);
    y[idx] = tanhf(v * inv_cap) * cap;
}

__global__ void cast_bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ x,
    float*               __restrict__ y,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    y[idx] = __bfloat162float(x[idx]);
}

extern "C" int nv_kernels_tanh_softcap_bf16_to_f32(
    void* stream,
    const uint16_t* x_bf16,
    float* y_f32,
    float cap,
    size_t n
) {
    if (n == 0) return 0;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int block = 256;
    const size_t grid = (n + block - 1) / block;
    if (grid > (size_t)0x7FFFFFFF) return -1;
    if (cap > 0.f && isfinite(cap)) {
        const float inv_cap = 1.f / cap;
        tanh_softcap_bf16_to_f32_kernel<<<(unsigned)grid, block, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_bf16),
            y_f32, cap, inv_cap, n
        );
    } else {
        cast_bf16_to_f32_kernel<<<(unsigned)grid, block, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(x_bf16),
            y_f32, n
        );
    }
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}

__global__ void attn_gate_sigmoid_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ q_raw_value_gate_interleaved_per_head,
    __nv_bfloat16* __restrict__ out,
    size_t rows,
    size_t n_heads,
    size_t head_dim
) {
    NVK_PDL_PROLOG();
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = rows * n_heads * head_dim;
    if (idx < total) {
        size_t d = idx % head_dim;
        size_t h = (idx / head_dim) % n_heads;
        size_t row = idx / (head_dim * n_heads);
        size_t gate_at = (row * n_heads + h) * (2 * head_dim) + head_dim + d;
        float g = __bfloat162float(q_raw_value_gate_interleaved_per_head[gate_at]);
        __nv_bfloat16 sig = __float2bfloat16(1.0f / (1.0f + expf(-g)));
        out[idx] = __float2bfloat16(
            __bfloat162float(attn[idx]) * __bfloat162float(sig));
    }
    NVK_PDL_EPILOG();
}

extern "C" int nv_kernels_attn_gate_sigmoid_mul_bf16(
    void* stream,
    const uint16_t* attn,
    const uint16_t* q_raw,
    uint16_t* out,
    size_t rows,
    size_t n_heads,
    size_t head_dim
) {
    cudaStream_t s = (cudaStream_t)stream;
    size_t total = rows * n_heads * head_dim;
    if (total == 0) return -2;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, dim3(grid), dim3(block), 0, s);
        cudaLaunchKernelEx(
            &cfg, attn_gate_sigmoid_mul_bf16_kernel,
            reinterpret_cast<const __nv_bfloat16*>(attn),
            reinterpret_cast<const __nv_bfloat16*>(q_raw),
            reinterpret_cast<__nv_bfloat16*>(out),
            rows, n_heads, head_dim);
    } else {
        attn_gate_sigmoid_mul_bf16_kernel<<<grid, block, 0, s>>>(
            reinterpret_cast<const __nv_bfloat16*>(attn),
            reinterpret_cast<const __nv_bfloat16*>(q_raw),
            reinterpret_cast<__nv_bfloat16*>(out),
            rows, n_heads, head_dim);
    }
    return (int)cudaGetLastError();
}
