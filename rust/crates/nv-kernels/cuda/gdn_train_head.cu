#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kHead = 128;
constexpr int kWarps = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kConvK = 4;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ float silu_f(float y) {
    return y / (1.f + __expf(-y));
}

__device__ __forceinline__ float silu_grad_f(float y) {
    float s = 1.f / (1.f + __expf(-y));
    return s * (1.f + y * (1.f - s));
}

struct HeadDims {
    int b;
    int t;
    int conv_dim;
    int key_dim;
    int n_v;
    int v_per_k;
    float q_scale;
};

__device__ __forceinline__ void conv_row_4(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    int bi,
    int ti,
    int c,
    const HeadDims& d,
    float* y4
) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) y4[i] = 0.f;
    #pragma unroll
    for (int j = 0; j < kConvK; ++j) {
        int ts = ti - (kConvK - 1) + j;
        if (ts < 0) continue;
        const __nv_bfloat16* xr = x + ((size_t)bi * d.t + ts) * d.conv_dim + c;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            y4[i] += w[(size_t)(c + i) * kConvK + j] * __bfloat162float(xr[i]);
        }
    }
}

__global__ void __launch_bounds__(kThreads) gdn_train_head_fwd_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ packed,
    HeadDims d
) {
    const int tok = blockIdx.x;
    const int bi = tok / d.t;
    const int ti = tok % d.t;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int passes = d.conv_dim / (kWarps * kHead);
    const int out_row = 2 * kHead + kHead;
    for (int p = 0; p < passes; ++p) {
        int c = (p * kWarps + warp) * kHead + lane * 4;
        float a[4];
        conv_row_4(x, w, bi, ti, c, d, a);
        #pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = silu_f(a[i]);
        int head_ch = c / kHead;
        int dd = lane * 4;
        if (c < 2 * d.key_dim) {
            float ss = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) ss += a[i] * a[i];
            ss = warp_sum(ss);
            float inv = 1.f / sqrtf(ss + 1e-6f);
            bool is_q = c < d.key_dim;
            int kh = is_q ? head_ch : head_ch - d.key_dim / kHead;
            float sc = is_q ? inv * d.q_scale : inv;
            int col0 = is_q ? 0 : kHead;
            for (int r = 0; r < d.v_per_k; ++r) {
                int vh = kh * d.v_per_k + r;
                float* o = packed + ((size_t)tok * d.n_v + vh) * out_row + col0 + dd;
                #pragma unroll
                for (int i = 0; i < 4; ++i) o[i] = a[i] * sc;
            }
        } else {
            int vh = (c - 2 * d.key_dim) / kHead;
            float* o = packed + ((size_t)tok * d.n_v + vh) * out_row + 2 * kHead + dd;
            #pragma unroll
            for (int i = 0; i < 4; ++i) o[i] = a[i];
        }
    }
}

__global__ void __launch_bounds__(kThreads) gdn_train_head_bwd_dy_kernel(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ w,
    const float* __restrict__ g_packed,
    float* __restrict__ dy,
    HeadDims d
) {
    const int tok = blockIdx.x;
    const int bi = tok / d.t;
    const int ti = tok % d.t;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int passes = d.conv_dim / (kWarps * kHead);
    const int out_row = 3 * kHead;
    for (int p = 0; p < passes; ++p) {
        int c = (p * kWarps + warp) * kHead + lane * 4;
        float y[4];
        conv_row_4(x, w, bi, ti, c, d, y);
        float a[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) a[i] = silu_f(y[i]);
        int head_ch = c / kHead;
        int dd = lane * 4;
        float da[4];
        if (c < 2 * d.key_dim) {
            bool is_q = c < d.key_dim;
            int kh = is_q ? head_ch : head_ch - d.key_dim / kHead;
            int col0 = is_q ? 0 : kHead;
            float gsum[4] = {0.f, 0.f, 0.f, 0.f};
            for (int r = 0; r < d.v_per_k; ++r) {
                int vh = kh * d.v_per_k + r;
                const float* g = g_packed + ((size_t)tok * d.n_v + vh) * out_row + col0 + dd;
                #pragma unroll
                for (int i = 0; i < 4; ++i) gsum[i] += g[i];
            }
            float sc = is_q ? d.q_scale : 1.f;
            float ss = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) ss += a[i] * a[i];
            ss = warp_sum(ss);
            float inv = 1.f / sqrtf(ss + 1e-6f);
            float dot = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) dot += a[i] * inv * gsum[i] * sc;
            dot = warp_sum(dot);
            #pragma unroll
            for (int i = 0; i < 4; ++i) da[i] = (gsum[i] * sc - a[i] * inv * dot) * inv;
        } else {
            int vh = (c - 2 * d.key_dim) / kHead;
            const float* g = g_packed + ((size_t)tok * d.n_v + vh) * out_row + 2 * kHead + dd;
            #pragma unroll
            for (int i = 0; i < 4; ++i) da[i] = g[i];
        }
        float* o = dy + (size_t)tok * d.conv_dim + c;
        #pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = da[i] * silu_grad_f(y[i]);
    }
}

__global__ void gdn_train_head_bwd_dx_kernel(
    const float* __restrict__ dy,
    const float* __restrict__ w,
    __nv_bfloat16* __restrict__ dx,
    HeadDims d
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)d.b * d.t * d.conv_dim;
    if (idx >= total) return;
    int c = (int)(idx % d.conv_dim);
    size_t tok = idx / d.conv_dim;
    int bi = (int)(tok / d.t);
    int ti = (int)(tok % d.t);
    float acc = 0.f;
    #pragma unroll
    for (int j = 0; j < kConvK; ++j) {
        int tq = ti + (kConvK - 1) - j;
        if (tq >= d.t) continue;
        acc += w[(size_t)c * kConvK + j] * dy[((size_t)bi * d.t + tq) * d.conv_dim + c];
    }
    dx[idx] = __float2bfloat16(acc);
}

}

extern "C" int nv_kernels_gdn_train_head_fwd(
    void* stream, const uint16_t* x, const float* w, float* packed,
    int b, int t, int conv_dim, int key_dim, int n_v, int v_per_k, int conv_k, float q_scale) {
    if (b * t <= 0) return 0;
    if (conv_k != kConvK || conv_dim % (kWarps * kHead) != 0 || key_dim % kHead != 0) return -2;
    if (n_v * kHead != conv_dim - 2 * key_dim || (key_dim / kHead) * v_per_k != n_v) return -2;
    HeadDims d = {b, t, conv_dim, key_dim, n_v, v_per_k, q_scale};
    gdn_train_head_fwd_kernel<<<b * t, kThreads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), w, packed, d);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gdn_train_head_bwd(
    void* stream, const uint16_t* x, const float* w, const float* g_packed, float* dy_scratch,
    uint16_t* dx, int b, int t, int conv_dim, int key_dim, int n_v, int v_per_k, int conv_k,
    float q_scale) {
    if (b * t <= 0) return 0;
    if (conv_k != kConvK || conv_dim % (kWarps * kHead) != 0 || key_dim % kHead != 0) return -2;
    if (n_v * kHead != conv_dim - 2 * key_dim || (key_dim / kHead) * v_per_k != n_v) return -2;
    HeadDims d = {b, t, conv_dim, key_dim, n_v, v_per_k, q_scale};
    cudaStream_t s = (cudaStream_t)stream;
    gdn_train_head_bwd_dy_kernel<<<b * t, kThreads, 0, s>>>(
        reinterpret_cast<const __nv_bfloat16*>(x), w, g_packed, dy_scratch, d);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return (int)e;
    size_t total = (size_t)b * t * conv_dim;
    unsigned blocks = (unsigned)((total + 255) / 256);
    gdn_train_head_bwd_dx_kernel<<<blocks, 256, 0, s>>>(
        dy_scratch, w, reinterpret_cast<__nv_bfloat16*>(dx), d);
    return (int)cudaGetLastError();
}
