#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kThreads = 256;
constexpr int kMaxHD512 = 512;

__device__ __forceinline__ float block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31;
    if (l == 0) red[w] = v;
    __syncthreads();
    v = (threadIdx.x < (kThreads / 32)) ? red[threadIdx.x] : 0.f;
    if (w == 0) {
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    if (threadIdx.x == 0) red[0] = v;
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

__global__ void __launch_bounds__(kThreads) e4b_prefill_qkv_prep_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    int qkv_row_stride,
    const __nv_bfloat16* __restrict__ qw,
    const __nv_bfloat16* __restrict__ kw,
    const float* __restrict__ cos_tbl,
    const float* __restrict__ sin_tbl,
    const int* __restrict__ pos,
    float* __restrict__ q_out,
    float* __restrict__ kcache,
    float* __restrict__ vcache,
    __nv_bfloat16* __restrict__ ktwin,
    __nv_bfloat16* __restrict__ vtwin,
    int write_off,
    int cap,
    int NH,
    int NKV,
    int HD,
    float eps) {
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const bool has_kv = kcache != nullptr;
    const int kind = head < NH ? 0 : (head < NH + NKV ? 1 : 2);
    const __nv_bfloat16* x = qkv + (size_t)row * qkv_row_stride + (size_t)head * HD;
    __shared__ float ns[kMaxHD512];
    __shared__ float red[kThreads / 32];
    float partial = 0.f;
    for (int d = threadIdx.x; d < HD; d += kThreads) {
        const float v = __bfloat162float(x[d]);
        partial += v * v;
    }
    const float sum = block_sum(partial, red);
    if (kind == 2) {
        const float denom = sqrtf(sum / (float)HD + eps);
        const int kvh = head - NH - NKV;
        const size_t dst = ((size_t)(write_off + row) * NKV + kvh) * (size_t)HD;
        for (int d = threadIdx.x; d < HD; d += kThreads) {
            const float v = __bfloat162float(x[d]) / denom;
            vcache[dst + d] = v;
            if (vtwin != nullptr) vtwin[((size_t)kvh * cap + write_off + row) * HD + d] = __float2bfloat16(v);
        }
        return;
    }
    const float rms = rsqrtf(sum / (float)HD + eps);
    const __nv_bfloat16* w = (kind == 0) ? qw : kw;
    for (int d = threadIdx.x; d < HD; d += kThreads) {
        ns[d] = __bfloat162float(__float2bfloat16(__bfloat162float(x[d]) * rms * __bfloat162float(w[d])));
    }
    __syncthreads();
    const int half = HD >> 1;
    const int p = pos[row];
    const float* cr = cos_tbl + (size_t)p * half;
    const float* sr = sin_tbl + (size_t)p * half;
    if (kind == 0) {
        float* out = q_out + ((size_t)row * NH + head) * HD;
        for (int d = threadIdx.x; d < HD; d += kThreads) {
            const int i = (d < half) ? d : d - half;
            const float a = ns[i];
            const float b = ns[i + half];
            out[d] = (d < half) ? (a * cr[i] - b * sr[i]) : (a * sr[i] + b * cr[i]);
        }
        return;
    }
    if (!has_kv) return;
    const int kvh = head - NH;
    const size_t dst = ((size_t)(write_off + row) * NKV + kvh) * (size_t)HD;
    for (int d = threadIdx.x; d < HD; d += kThreads) {
        const int i = (d < half) ? d : d - half;
        const float a = ns[i];
        const float b = ns[i + half];
        const float r = (d < half) ? (a * cr[i] - b * sr[i]) : (a * sr[i] + b * cr[i]);
        kcache[dst + d] = r;
        if (ktwin != nullptr) ktwin[((size_t)kvh * cap + write_off + row) * HD + d] = __float2bfloat16(r);
    }
}

}

extern "C" int nv_kernels_e4b_prefill_qkv_prep(
    void* stream,
    const uint16_t* qkv,
    int qkv_row_stride,
    const uint16_t* qw,
    const uint16_t* kw,
    const float* cos_tbl,
    const float* sin_tbl,
    const int* pos,
    float* q_out,
    float* kcache,
    float* vcache,
    uint16_t* ktwin,
    uint16_t* vtwin,
    int n,
    int write_off,
    int cap,
    int NH,
    int NKV,
    int HD,
    float eps) {
    if (n <= 0 || NH <= 0 || HD <= 0 || HD > kMaxHD512 || (HD & 1)) return -1;
    const bool has_kv = kcache != nullptr && vcache != nullptr && kw != nullptr && NKV > 0;
    const int heads = has_kv ? NH + 2 * NKV : NH;
    dim3 grid((unsigned)n, (unsigned)heads);
    e4b_prefill_qkv_prep_kernel<<<grid, kThreads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv), qkv_row_stride, reinterpret_cast<const __nv_bfloat16*>(qw),
        reinterpret_cast<const __nv_bfloat16*>(kw), cos_tbl, sin_tbl, pos, q_out, has_kv ? kcache : nullptr,
        has_kv ? vcache : nullptr, reinterpret_cast<__nv_bfloat16*>(ktwin), reinterpret_cast<__nv_bfloat16*>(vtwin),
        write_off, cap, NH, NKV, HD, eps);
    return (int)cudaGetLastError();
}
