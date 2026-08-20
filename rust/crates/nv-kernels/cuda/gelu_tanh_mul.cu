
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

__device__ __forceinline__ float gelu_tanh_mul_scalar(float gate, float up) {
    constexpr float kSqrt2OverPi = 0.7978845608028654f;
    constexpr float kCubicCoeff = 0.044715f;
    float g3 = gate * gate * gate;
    float inner = kSqrt2OverPi * (gate + kCubicCoeff * g3);
    float t = tanhf(inner);
    float gelu = 0.5f * gate * (1.0f + t);
    return gelu * up;
}

__global__ void gelu_tanh_mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ y,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float g = __bfloat162float(gate[idx]);
    float u = __bfloat162float(up[idx]);
    y[idx] = __float2bfloat16(gelu_tanh_mul_scalar(g, u));
}

__global__ void gelu_tanh_mul_fused_bf16_kernel(
    const __nv_bfloat16* __restrict__ fused,
    __nv_bfloat16*       __restrict__ y,
    int inter,
    size_t tot_pairs
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tot_pairs) return;
    int i = (int)(idx % (size_t)inter);
    size_t bs = idx / (size_t)inter;
    size_t off = bs * 2 * (size_t)inter;
    float g = __bfloat162float(fused[off + i]);
    float u = __bfloat162float(fused[off + inter + i]);
    y[idx] = __float2bfloat16(gelu_tanh_mul_scalar(g, u));
}

}

extern "C" int nv_kernels_gelu_tanh_mul_bf16(
    void* stream,
    const uint16_t* gate,
    const uint16_t* up,
    uint16_t* y,
    size_t n
) {
    cudaStream_t s = (cudaStream_t)stream;
    constexpr int BLOCK = 256;
    int grid = (int)((n + BLOCK - 1) / BLOCK);
    if (grid <= 0) return 0;
    gelu_tanh_mul_bf16_kernel<<<grid, BLOCK, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<const __nv_bfloat16*>(up),
        reinterpret_cast<__nv_bfloat16*>(y),
        n);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gelu_tanh_mul_fused_bf16(
    void* stream,
    const uint16_t* fused,
    uint16_t* y,
    int inter,
    size_t tot_pairs
) {
    if (tot_pairs == 0 || inter <= 0) return 0;
    cudaStream_t s = (cudaStream_t)stream;
    constexpr int BLOCK = 256;
    size_t grid = (tot_pairs + BLOCK - 1) / BLOCK;
    if (grid > (size_t)0x7FFFFFFF) return -1;
    gelu_tanh_mul_fused_bf16_kernel<<<(unsigned)grid, BLOCK, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(fused),
        reinterpret_cast<__nv_bfloat16*>(y),
        inter,
        tot_pairs
    );
    return (int)cudaGetLastError();
}
