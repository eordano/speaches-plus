#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <stdint.h>

namespace fusedtc {

__device__ __forceinline__ unsigned fp8x2_to_half2_bits(unsigned packed16) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed16, __NV_E4M3);
    return *reinterpret_cast<unsigned*>(&hr);
}

__device__ __forceinline__ unsigned half2_bits(float x, float y) {
    __half2 h = __floats2half2_rn(x, y);
    return *reinterpret_cast<unsigned*>(&h);
}

__device__ __forceinline__ void mma_m16n8k16_f16(
    float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void ldmatrix_x4_trans(unsigned (&r)[4], const void* smem_row) {
    const unsigned addr = (unsigned)__cvta_generic_to_shared(smem_row);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
        : "r"(addr));
}

__device__ __forceinline__ float warp_max_over_gid(float x) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 4));
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 8));
    x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, 16));
    return x;
}

__device__ __forceinline__ float warp_sum_over_gid(float x) {
    x += __shfl_xor_sync(0xffffffffu, x, 4);
    x += __shfl_xor_sync(0xffffffffu, x, 8);
    x += __shfl_xor_sync(0xffffffffu, x, 16);
    return x;
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1
) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

}
