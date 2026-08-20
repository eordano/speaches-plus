#pragma once
#include <cuda_bf16.h>

#define NVK_GDN_CONV_MAX_K 8

__device__ __forceinline__ __nv_bfloat16 nvk_gdn_conv_step_silu(
    const __nv_bfloat16* win,
    __nv_bfloat16 x_new,
    const __nv_bfloat16* w_row,
    int K
) {
    float xn = __bfloat162float(x_new);
    float acc = 0.f;
    for (int i = 0; i < K - 1; ++i) {
        acc += __bfloat162float(win[i]) * __bfloat162float(w_row[i]);
    }
    acc += xn * __bfloat162float(w_row[K - 1]);
    float sig = 1.f / (1.f + expf(-acc));
    return __float2bfloat16(acc * sig);
}
