#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {

constexpr int kBlock256MatchesRmsnormBf16LauncherSoTheReductionOrderIsBitIdentical = 256;

template <int BLOCK>
__device__ inline float g4qr_block_sum_same_shape_as_rmsnorm_cu(float v) {
    constexpr int kWarp = 32;
    constexpr int kWarps = BLOCK / kWarp;
    __shared__ float warp_sums[kWarps];
    __shared__ float total;
    int lane = threadIdx.x & (kWarp - 1);
    int warp = threadIdx.x / kWarp;
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = (lane < kWarps) ? warp_sums[lane] : 0.0f;
        #pragma unroll
        for (int o = kWarps / 2; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

template <int BLOCK>
__global__ void g4_qknorm_rope_vnorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ q_raw,
    const __nv_bfloat16* __restrict__ k_raw,
    const __nv_bfloat16* __restrict__ v_raw,
    const __nv_bfloat16* __restrict__ q_norm_w,
    const __nv_bfloat16* __restrict__ k_norm_w,
    const __nv_bfloat16* __restrict__ v_norm_w,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    const int32_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_out,
    __nv_bfloat16* __restrict__ v_out,
    int n_q,
    int n_kv,
    int hd,
    float eps
) {
    extern __shared__ float normed_bf16_roundtripped[];

    int head = blockIdx.x;
    int token = blockIdx.y;
    bool is_q = head < n_q;
    bool is_k = !is_q && head < n_q + n_kv;
    int hk = is_q ? head : (is_k ? head - n_q : head - n_q - n_kv);

    const __nv_bfloat16* row_in;
    __nv_bfloat16* row_out;
    const __nv_bfloat16* w;
    if (is_q) {
        row_in = q_raw + ((size_t)token * n_q + hk) * hd;
        row_out = q_out + ((size_t)token * n_q + hk) * hd;
        w = q_norm_w;
    } else if (is_k) {
        row_in = k_raw + ((size_t)token * n_kv + hk) * hd;
        row_out = k_out + ((size_t)token * n_kv + hk) * hd;
        w = k_norm_w;
    } else {
        row_in = v_raw + ((size_t)token * n_kv + hk) * hd;
        row_out = v_out + ((size_t)token * n_kv + hk) * hd;
        w = v_norm_w;
    }

    float local = 0.f;
    for (int i = threadIdx.x; i < hd; i += BLOCK) {
        float v = __bfloat162float(row_in[i]);
        local += v * v;
    }
    float sum = g4qr_block_sum_same_shape_as_rmsnorm_cu<BLOCK>(local);
    float rms = rsqrtf(sum / (float)hd + eps);

    if (!is_q && !is_k) {
        for (int i = threadIdx.x; i < hd; i += BLOCK) {
            float v = __bfloat162float(row_in[i]) * rms * __bfloat162float(w[i]);
            row_out[i] = __float2bfloat16(v);
        }
        return;
    }

    for (int i = threadIdx.x; i < hd; i += BLOCK) {
        float v = __bfloat162float(row_in[i]) * rms * __bfloat162float(w[i]);
        normed_bf16_roundtripped[i] = __bfloat162float(__float2bfloat16(v));
    }
    __syncthreads();

    int half = hd / 2;
    int pos = positions[token];
    const float* cos_row = cos_tab + (size_t)pos * half;
    const float* sin_row = sin_tab + (size_t)pos * half;
    for (int i = threadIdx.x; i < half; i += BLOCK) {
        float c = cos_row[i];
        float s = sin_row[i];
        float a = normed_bf16_roundtripped[i];
        float b = normed_bf16_roundtripped[i + half];
        row_out[i] = __float2bfloat16(a * c - b * s);
        row_out[i + half] = __float2bfloat16(a * s + b * c);
    }
}

}

extern "C" int nv_kernels_g4_qknorm_rope_vnorm_bf16(
    void* stream,
    const uint16_t* q_raw,
    const uint16_t* k_raw,
    const uint16_t* v_raw,
    const uint16_t* q_norm_w,
    const uint16_t* k_norm_w,
    const uint16_t* v_norm_w,
    const float* cos_tab,
    const float* sin_tab,
    const int32_t* positions,
    uint16_t* q_out,
    uint16_t* k_out,
    uint16_t* v_out,
    int batch,
    int n_q,
    int n_kv,
    int hd,
    float eps
) {
    constexpr int BLOCK = kBlock256MatchesRmsnormBf16LauncherSoTheReductionOrderIsBitIdentical;
    if (batch <= 0 || n_q <= 0 || n_kv <= 0) return -1;
    if (hd < 2 || (hd & 1) != 0 || hd > 4096) return -1;
    dim3 grid((unsigned)(n_q + 2 * n_kv), (unsigned)batch);
    size_t smem = (size_t)hd * sizeof(float);
    g4_qknorm_rope_vnorm_bf16_kernel<BLOCK><<<grid, dim3(BLOCK), smem,
                                              (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q_raw),
        reinterpret_cast<const __nv_bfloat16*>(k_raw),
        reinterpret_cast<const __nv_bfloat16*>(v_raw),
        reinterpret_cast<const __nv_bfloat16*>(q_norm_w),
        reinterpret_cast<const __nv_bfloat16*>(k_norm_w),
        reinterpret_cast<const __nv_bfloat16*>(v_norm_w),
        cos_tab, sin_tab, positions,
        reinterpret_cast<__nv_bfloat16*>(q_out),
        reinterpret_cast<__nv_bfloat16*>(k_out),
        reinterpret_cast<__nv_bfloat16*>(v_out),
        n_q, n_kv, hd, eps);
    return (int)cudaGetLastError();
}
