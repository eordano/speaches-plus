#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include "nv_kernels.h"
#include "nv_hip_wave.h"

__global__ void gather_rows_bf16_kernel(
    const __hip_bfloat16* __restrict__ x,
    const int32_t* __restrict__ src_idx,
    __hip_bfloat16* __restrict__ out,
    int m_total_padded,
    int hidden,
    int n_tokens
) {
    int r = blockIdx.x;
    if (r >= m_total_padded) return;
    int s = src_idx[r];
    const __hip_bfloat16* src = nullptr;
    if (s >= 0 && s < n_tokens) {
        src = x + (size_t)s * hidden;
    }
    __hip_bfloat16* dst = out + (size_t)r * hidden;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        dst[h] = src ? src[h] : __float2bfloat16(0.f);
    }
}

extern "C" int nv_kernels_gather_rows_bf16(
    void* stream,
    const uint16_t* x_bf16,
    const int32_t* src_idx,
    uint16_t* out_bf16,
    int m_total_padded,
    int hidden,
    int n_tokens
) {
    if (m_total_padded <= 0 || hidden <= 0) return 0;
    hipStream_t s = static_cast<hipStream_t>(stream);
    int block = nv_hip::wave_aligned_block(hidden < 256 ? hidden : 256, 256);
    gather_rows_bf16_kernel<<<m_total_padded, block, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(x_bf16),
        src_idx,
        reinterpret_cast<__hip_bfloat16*>(out_bf16),
        m_total_padded, hidden, n_tokens
    );
    hipError_t e = hipGetLastError();
    return (e == hipSuccess) ? 0 : (int)e;
}
