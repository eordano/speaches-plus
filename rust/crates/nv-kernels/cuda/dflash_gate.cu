
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

__device__ inline float softplus_stable(float x) {
    return fmaxf(x, 0.f) + log1pf(expf(-fabsf(x)));
}

__global__ void softplus_gate_bf16_kernel(
    const __nv_bfloat16* __restrict__ attn,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int groups,
    int hd
) {
    int g = blockIdx.x;
    if (g >= groups) return;
    float gv = __bfloat162float(
        __float2bfloat16(softplus_stable(__bfloat162float(gate[g]))));
    const __nv_bfloat16* a = attn + (size_t)g * hd;
    __nv_bfloat16* y = out + (size_t)g * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        y[d] = __float2bfloat16(__bfloat162float(a[d]) * gv);
    }
}

}

extern "C" int nv_kernels_softplus_gate_bf16(
    void* stream,
    const uint16_t* attn,
    const uint16_t* gate,
    uint16_t* out,
    int groups,
    int hd
) {
    if (groups <= 0 || hd <= 0) return -1;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    softplus_gate_bf16_kernel<<<groups, 128, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(attn),
        reinterpret_cast<const __nv_bfloat16*>(gate),
        reinterpret_cast<__nv_bfloat16*>(out),
        groups,
        hd
    );
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : static_cast<int>(e);
}
