#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#define NVK_GDN_CONV_MAX_K 8

__device__ __forceinline__ float nvk_gdn_conv_weight(const __nv_bfloat16* w, int i, float) {
    return __bfloat162float(w[i]);
}

__device__ __forceinline__ float nvk_gdn_conv_weight(const uint8_t* w, int i, float scale) {
    __half_raw raw = __nv_cvt_fp8_to_halfraw(w[i], __NV_E4M3);
    float value;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(value) : "h"(raw.x));
    return __bfloat162float(__float2bfloat16(value * scale));
}

template<typename Weight>
__device__ __forceinline__ __nv_bfloat16 nvk_gdn_conv_step_silu(
    const __nv_bfloat16* win,
    __nv_bfloat16 x_new,
    const Weight* w_row,
    int K,
    float scale = 1.f
) {
    float xn = __bfloat162float(x_new);
    float acc = 0.f;
    for (int i = 0; i < K - 1; ++i) {
        acc += __bfloat162float(win[i]) * nvk_gdn_conv_weight(w_row, i, scale);
    }
    acc += xn * nvk_gdn_conv_weight(w_row, K - 1, scale);
    float sig = 1.f / (1.f + expf(-acc));
    return __float2bfloat16(acc * sig);
}
