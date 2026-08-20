
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kBlockDim = 256;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float decode_e2m1_dev(uint8_t nib) {
    static const float kE2M1[16] = {
         0.f,  0.5f,  1.f,  1.5f,  2.f,  3.f,  4.f,  6.f,
        -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
    };
    return kE2M1[nib & 0xF];
}

__device__ __forceinline__ uint8_t encode_e2m1_dev(float x) {
    static const float kE2M1[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    uint8_t sign = signbit(x) ? 0b1000 : 0;
    float a = fabsf(x);
    uint8_t best = 0;
    float best_err = INFINITY;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float err = fabsf(a - kE2M1[i]);
        if (err < best_err) {
            best_err = err;
            best = (uint8_t)i;
        }
    }
    return (uint8_t)(sign | best);
}

#define NV_UE4M3_MIN_NORMAL      0.015625f
#define NV_UE4M3_SUBNORMAL_STEP  0.001953125f

__device__ __forceinline__ uint8_t encode_ue4m3_dev(float scale) {
    if (!isfinite(scale) || scale <= 0.f) return 0;
    float clamped = fminf(scale, 448.f);
    if (clamped < NV_UE4M3_MIN_NORMAL) {
        int sub = (int)roundf(clamped / NV_UE4M3_SUBNORMAL_STEP);
        if (sub <= 0) return 0;
        if (sub <= 7) return (uint8_t)sub;
        return 0x08;
    }
    int e2;
    frexpf(clamped, &e2);
    int exp_v = e2 - 1;
    float mant_f = ldexpf(clamped, -exp_v) - 1.f;
    int mant = (int)roundf(mant_f * 8.f);
    if (mant < 0) mant = 0;
    if (mant > 7) { mant = 0; exp_v += 1; }
    int biased = exp_v + 7;
    if (biased < 1) biased = 1;
    if (biased > 15) biased = 15;
    uint8_t byte = ((uint8_t)biased << 3) | (uint8_t)(mant & 0x07);
    return (byte == 0x7F) ? 0x7E : byte;
}

__device__ __forceinline__ float decode_ue4m3_dev(uint8_t b) {
    int biased = (int)(b >> 3) & 0x0F;
    float mant = (float)(b & 0x07);
    if (biased == 0) return mant * NV_UE4M3_SUBNORMAL_STEP;
    return (1.f + mant / 8.f) * exp2f((float)(biased - 7));
}

__device__ __forceinline__ int swizzled_scale_dst(int m, int kb, int k_blocks) {
    int k_tiles = (k_blocks + 3) / 4;
    int m_tile = m / 128;
    int d2 = (m / 32) & 3;
    int d3 = m & 31;
    int k_tile = kb / 4;
    int d5 = kb & 3;
    return ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
}

__global__ void nvfp4_quantize_row_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ packed_out,
    uint8_t* __restrict__ scales_out,
    float stored_global,
    int K
) {
    int tid = threadIdx.x;
    int n_blocks = K / 16;
    float stored = (stored_global == 0.f || !isfinite(stored_global)) ? 1.f : stored_global;

    for (int kb = tid; kb < n_blocks; kb += kBlockDim) {
        float vals[16];
        float amax = 0.f;
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            float v = __bfloat162float(x[kb * 16 + i]);
            vals[i] = v;
            float av = fabsf(v);
            if (av > amax) amax = av;
        }
        float local_scale = (amax == 0.f) ? 1.f : (amax / 6.f);
        uint8_t scale_byte = encode_ue4m3_dev(stored * local_scale);
        float scale_decoded = decode_ue4m3_dev(scale_byte);
        float inv = (scale_decoded == 0.f) ? 1.f : (stored / scale_decoded);

        uint8_t* dst = packed_out + kb * 8;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            float v_lo = fminf(fmaxf(vals[2*i]     * inv, -6.f), 6.f);
            float v_hi = fminf(fmaxf(vals[2*i + 1] * inv, -6.f), 6.f);
            uint8_t lo = encode_e2m1_dev(v_lo);
            uint8_t hi = encode_e2m1_dev(v_hi);
            dst[i] = (uint8_t)((hi << 4) | (lo & 0x0F));
        }

        scales_out[kb] = scale_byte;
    }
}

__global__ void nvfp4_gemv_bf16_kernel(
    const uint8_t* __restrict__ W_packed,
    const uint8_t* __restrict__ W_scales,
    const uint8_t* __restrict__ x_packed,
    const uint8_t* __restrict__ x_scales,
    __nv_bfloat16* __restrict__ y,
    float alpha,
    int N,
    int K
) {
    int n = blockIdx.x;
    if (n >= N) return;
    int tid = threadIdx.x;
    int n_blocks = K / 16;
    const uint8_t* w_row = W_packed + (size_t)n * (K / 2);

    float acc = 0.f;
    for (int kb = tid; kb < n_blocks; kb += kBlockDim) {
        int w_scale_idx = swizzled_scale_dst(n, kb, n_blocks);
        float w_scale = decode_ue4m3_dev(W_scales[w_scale_idx]);
        float x_scale = decode_ue4m3_dev(x_scales[kb]);
        float block_scale = w_scale * x_scale;

        const uint8_t* w_block = w_row + kb * 8;
        const uint8_t* x_block = x_packed + kb * 8;
        float block_dot = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t wb = w_block[i];
            uint8_t xb = x_block[i];
            float w_lo = decode_e2m1_dev(wb & 0xF);
            float w_hi = decode_e2m1_dev((wb >> 4) & 0xF);
            float x_lo = decode_e2m1_dev(xb & 0xF);
            float x_hi = decode_e2m1_dev((xb >> 4) & 0xF);
            block_dot += w_lo * x_lo + w_hi * x_hi;
        }
        acc += block_scale * block_dot;
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    __shared__ float smem[kBlockDim / kWarpSize];
    int lane = tid & (kWarpSize - 1);
    int warp = tid / kWarpSize;
    if (lane == 0) smem[warp] = acc;
    __syncthreads();

    if (warp == 0) {
        int n_warps = kBlockDim / kWarpSize;
        acc = (lane < n_warps) ? smem[lane] : 0.f;
        #pragma unroll
        for (int offset = n_warps / 2; offset > 0; offset >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, offset);
        }
        if (lane == 0) y[n] = __float2bfloat16(acc * alpha);
    }
}

}

extern "C" int nv_kernels_nvfp4_quantize_row_bf16(
    void* stream,
    const uint16_t* x,
    uint8_t* packed_out,
    uint8_t* scales_out,
    float stored_global,
    int K
) {
    if ((K & 15) != 0) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    nvfp4_quantize_row_bf16_kernel<<<1, kBlockDim, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x),
        packed_out,
        scales_out,
        stored_global,
        K
    );
    return (int)cudaGetLastError();
}

__device__ __forceinline__ float decode_ue4m3_branchless_dev(uint8_t b) {
    unsigned biased = (unsigned)(b >> 3) & 0x0Fu;
    unsigned mant = (unsigned)(b & 0x07u);
    float norm = __uint_as_float(((biased + 120u) << 23) | (mant << 20));
    return biased ? norm : (float)mant * NV_UE4M3_SUBNORMAL_STEP;
}

__device__ __forceinline__ float e2m1_lo_times_2e14(unsigned byte) {
    unsigned h = ((byte & 0x08u) << 12) | ((byte & 0x07u) << 9);
    return __half2float(__ushort_as_half((unsigned short)h));
}

__device__ __forceinline__ float e2m1_hi_times_2e14(unsigned byte) {
    unsigned h = ((byte & 0x80u) << 8) | ((byte & 0x70u) << 5);
    return __half2float(__ushort_as_half((unsigned short)h));
}

__device__ __forceinline__ float bf16_bits_to_f32(unsigned bits16) {
    return __uint_as_float(bits16 << 16);
}

constexpr int kGemvActWarps = 8;

__global__ void nvfp4_gemv_bf16act_kernel(
    const uint8_t* __restrict__ W_packed,
    const uint8_t* __restrict__ W_scales,
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ y,
    float alpha,
    int N,
    int K
) {
    int warp = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int n = blockIdx.x * kGemvActWarps + warp;
    if (n >= N) return;
    int n_blocks = K / 16;
    const uint8_t* w_row = W_packed + (size_t)n * (K / 2);
    const uint16_t* xb = reinterpret_cast<const uint16_t*>(x);

    float acc = 0.f;
    for (int kb = lane; kb < n_blocks; kb += 32) {
        float scale =
            decode_ue4m3_branchless_dev(W_scales[swizzled_scale_dst(n, kb, n_blocks)]) * 16384.f;
        uint2 wq = *reinterpret_cast<const uint2*>(w_row + (size_t)kb * 8);
        uint4 x0 = *reinterpret_cast<const uint4*>(xb + (size_t)kb * 16);
        uint4 x1 = *reinterpret_cast<const uint4*>(xb + (size_t)kb * 16 + 8);

        float bd = 0.f;
        unsigned w0 = wq.x;
        unsigned w1 = wq.y;
        unsigned xw;
        xw = x0.x;
        bd = fmaf(e2m1_lo_times_2e14(w0), bf16_bits_to_f32(xw & 0xffffu), bd);
        bd = fmaf(e2m1_hi_times_2e14(w0), bf16_bits_to_f32(xw >> 16), bd);
        xw = x0.y;
        bd = fmaf(e2m1_lo_times_2e14(w0 >> 8), bf16_bits_to_f32(xw & 0xffffu), bd);
        bd = fmaf(e2m1_hi_times_2e14(w0 >> 8), bf16_bits_to_f32(xw >> 16), bd);
        xw = x0.z;
        bd = fmaf(e2m1_lo_times_2e14(w0 >> 16), bf16_bits_to_f32(xw & 0xffffu), bd);
        bd = fmaf(e2m1_hi_times_2e14(w0 >> 16), bf16_bits_to_f32(xw >> 16), bd);
        xw = x0.w;
        bd = fmaf(e2m1_lo_times_2e14(w0 >> 24), bf16_bits_to_f32(xw & 0xffffu), bd);
        bd = fmaf(e2m1_hi_times_2e14(w0 >> 24), bf16_bits_to_f32(xw >> 16), bd);
        xw = x1.x;
        bd = fmaf(e2m1_lo_times_2e14(w1), bf16_bits_to_f32(xw & 0xffffu), bd);
        bd = fmaf(e2m1_hi_times_2e14(w1), bf16_bits_to_f32(xw >> 16), bd);
        xw = x1.y;
        bd = fmaf(e2m1_lo_times_2e14(w1 >> 8), bf16_bits_to_f32(xw & 0xffffu), bd);
        bd = fmaf(e2m1_hi_times_2e14(w1 >> 8), bf16_bits_to_f32(xw >> 16), bd);
        xw = x1.z;
        bd = fmaf(e2m1_lo_times_2e14(w1 >> 16), bf16_bits_to_f32(xw & 0xffffu), bd);
        bd = fmaf(e2m1_hi_times_2e14(w1 >> 16), bf16_bits_to_f32(xw >> 16), bd);
        xw = x1.w;
        bd = fmaf(e2m1_lo_times_2e14(w1 >> 24), bf16_bits_to_f32(xw & 0xffffu), bd);
        bd = fmaf(e2m1_hi_times_2e14(w1 >> 24), bf16_bits_to_f32(xw >> 16), bd);

        acc = fmaf(scale, bd, acc);
    }

    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_xor_sync(0xffffffff, acc, offset);
    }
    if (lane == 0) y[n] = __float2bfloat16(acc * alpha);
}

extern "C" int nv_kernels_nvfp4_gemv_bf16act(
    void* stream,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const uint16_t* x_bf16,
    uint16_t* y,
    float alpha,
    int N,
    int K
) {
    if (N <= 0 || K <= 0) return 0;
    if ((K & 15) != 0) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)((N + kGemvActWarps - 1) / kGemvActWarps));
    dim3 block(kGemvActWarps * kWarpSize);
    nvfp4_gemv_bf16act_kernel<<<grid, block, 0, s>>>(
        W_packed,
        W_scales,
        reinterpret_cast<const __nv_bfloat16*>(x_bf16),
        reinterpret_cast<__nv_bfloat16*>(y),
        alpha,
        N, K
    );
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_nvfp4_gemv_bf16(
    void* stream,
    const uint8_t* W_packed,
    const uint8_t* W_scales,
    const uint8_t* x_packed,
    const uint8_t* x_scales,
    uint16_t* y,
    float alpha,
    int N,
    int K
) {
    if (N <= 0 || K <= 0) return 0;
    if ((K & 15) != 0) return -1;
    cudaStream_t s = (cudaStream_t)stream;
    dim3 grid((unsigned)N);
    dim3 block(kBlockDim);
    nvfp4_gemv_bf16_kernel<<<grid, block, 0, s>>>(
        W_packed,
        W_scales,
        x_packed,
        x_scales,
        reinterpret_cast<__nv_bfloat16*>(y),
        alpha,
        N, K
    );
    return (int)cudaGetLastError();
}
