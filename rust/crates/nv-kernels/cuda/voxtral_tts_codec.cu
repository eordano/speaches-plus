#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace {

constexpr int kDim = 1024;
constexpr int kHeads = 8;
constexpr int kHeadDim = 128;
constexpr int kQkvStride = 3 * kDim;
constexpr int kVOffset = 2 * kDim;
constexpr int kMaxKeys = 17;
constexpr int kRowThreads = 256;
constexpr int kPerThread = kDim / kRowThreads;
constexpr int kPadZero = 0;
constexpr int kPadReplicate = 1;
constexpr int kPadReflect = 2;

__device__ inline float bf(__nv_bfloat16 v) { return __bfloat162float(v); }
__device__ inline __nv_bfloat16 rn(float v) { return __float2bfloat16_rn(v); }
__device__ inline float round_bf(float v) { return bf(rn(v)); }

__device__ inline float block_sum_256(float v) {
    __shared__ float warp_sums[kRowThreads / 32];
    __shared__ float total;
    int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = lane < kRowThreads / 32 ? warp_sums[lane] : 0.f;
        for (int o = 4; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        if (lane == 0) total = s;
    }
    __syncthreads();
    return total;
}

__device__ inline void norm_row_store(const float* v, const __nv_bfloat16* w, __nv_bfloat16* h, float eps) {
    float ss = 0.f;
    for (int j = 0; j < kPerThread; ++j) ss += v[j] * v[j];
    float inv = 1.0f / sqrtf(block_sum_256(ss) / (float)kDim + eps);
    for (int j = 0; j < kPerThread; ++j) {
        int c = threadIdx.x + j * kRowThreads;
        h[c] = rn(v[j] * inv * bf(w[c]));
    }
}

__global__ void cx_im2col_kernel(
    const __nv_bfloat16* __restrict__ src, const __nv_bfloat16* __restrict__ hist, const int* ctl,
    float* __restrict__ dst, int rows, int c, int taps, int mode
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)rows * taps * c) return;
    int t = idx / (taps * c);
    int rem = idx % (taps * c);
    int tap = rem / c, ch = rem % c;
    int src_t = t + tap + 1 - taps;
    float v = 0.f;
    if (src_t >= 0) v = bf(src[(size_t)src_t * c + ch]);
    else if (ctl[0] != 0) v = bf(hist[(size_t)(taps - 1 + src_t) * c + ch]);
    else if (mode == kPadReplicate) v = bf(src[ch]);
    else if (mode == kPadReflect) v = bf(src[(size_t)(-src_t) * c + ch]);
    dst[idx] = v;
}

__global__ void cx_carry_kernel(const float* __restrict__ cols, __nv_bfloat16* __restrict__ hist, int rows, int c, int taps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (taps - 1) * c) return;
    hist[idx] = rn(cols[(size_t)(rows - 1) * taps * c + c + idx]);
}

__global__ void cx_cast_norm_kernel(
    const float* __restrict__ src, __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
    __nv_bfloat16* __restrict__ h, float eps
) {
    size_t row = (size_t)blockIdx.x * kDim;
    float v[kPerThread];
    for (int j = 0; j < kPerThread; ++j) {
        int c = threadIdx.x + j * kRowThreads;
        __nv_bfloat16 b = rn(src[row + c]);
        x[row + c] = b;
        v[j] = bf(b);
    }
    norm_row_store(v, w, h + row, eps);
}

__global__ void cx_qk_norm_kernel(
    const __nv_bfloat16* __restrict__ qkv, const __nv_bfloat16* __restrict__ qw, const __nv_bfloat16* __restrict__ kw,
    __nv_bfloat16* __restrict__ qn, __nv_bfloat16* __restrict__ kn, float eps
) {
    int which = blockIdx.y;
    const __nv_bfloat16* src = qkv + (size_t)blockIdx.x * kQkvStride + which * kDim;
    float v[kPerThread];
    for (int j = 0; j < kPerThread; ++j) v[j] = bf(src[threadIdx.x + j * kRowThreads]);
    norm_row_store(v, which ? kw : qw, (which ? kn : qn) + (size_t)blockIdx.x * kDim, eps);
}

__global__ void cx_residual_norm_kernel(
    __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ t, const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ h, float eps
) {
    size_t row = (size_t)blockIdx.x * kDim;
    float v[kPerThread];
    for (int j = 0; j < kPerThread; ++j) {
        int c = threadIdx.x + j * kRowThreads;
        __nv_bfloat16 b = rn(bf(x[row + c]) + round_bf(bf(t[row + c]) * bf(scale[c])));
        x[row + c] = b;
        v[j] = bf(b);
    }
    if (w != nullptr) norm_row_store(v, w, h + row, eps);
}

__global__ void cx_attn_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ kring, const __nv_bfloat16* __restrict__ vring, const int* ctl,
    __nv_bfloat16* __restrict__ out, int window, int rows_per_frame, float scale
) {
    __shared__ float sc[kMaxKeys];
    __shared__ int kv_row[kMaxKeys];
    int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    int lane = d & 31, warp = d >> 5;
    int f = ctl[0] * rows_per_frame;
    int p = f + i;
    int first = p > window ? p - window : 0;
    int nk = p - first + 1;
    float slope = exp2f(-(float)h);
    for (int kk = warp; kk < nk; kk += kHeadDim / 32) {
        int pos = first + kk;
        const __nv_bfloat16* kp = pos < f ? kring + (size_t)(pos & (window - 1)) * kDim + h * kHeadDim
                                          : k + (size_t)(pos - f) * kDim + h * kHeadDim;
        const __nv_bfloat16* qp = q + (size_t)i * kDim + h * kHeadDim;
        float acc = 0.f;
        for (int e = lane; e < kHeadDim; e += 32) acc += bf(qp[e]) * bf(kp[e]);
        for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
        if (lane == 0) {
            sc[kk] = acc * scale - slope * (float)(p - pos);
            kv_row[kk] = pos < f ? -1 - (pos & (window - 1)) : pos - f;
        }
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int kk = 0; kk < nk; ++kk) mx = fmaxf(mx, sc[kk]);
    float den = 0.f, o = 0.f;
    for (int kk = 0; kk < nk; ++kk) {
        float e = expf(sc[kk] - mx);
        int r = kv_row[kk];
        float v = r < 0 ? bf(vring[(size_t)(-1 - r) * kDim + h * kHeadDim + d])
                        : bf(qkv[(size_t)r * kQkvStride + kVOffset + h * kHeadDim + d]);
        den += e;
        o += e * v;
    }
    out[(size_t)i * kDim + h * kHeadDim + d] = rn(o / den);
}

__global__ void cx_kv_commit_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ qkv, __nv_bfloat16* __restrict__ kring,
    __nv_bfloat16* __restrict__ vring, const int* ctl, int rows, int window, int rows_per_frame
) {
    int cnt = rows < window ? rows : window;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cnt * kDim) return;
    int r = idx / kDim, c = idx % kDim;
    int j = rows - cnt + r;
    int slot = (ctl[0] * rows_per_frame + j) & (window - 1);
    kring[(size_t)slot * kDim + c] = k[(size_t)j * kDim + c];
    vring[(size_t)slot * kDim + c] = qkv[(size_t)j * kQkvStride + kVOffset + c];
}

__global__ void cx_silu_mul_kernel(const __nv_bfloat16* __restrict__ gu, __nv_bfloat16* __restrict__ act, int rows, int inter) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (size_t)rows * inter) return;
    size_t r = idx / inter, c = idx % inter;
    float g = bf(gu[r * 2 * inter + c]);
    float u = bf(gu[r * 2 * inter + inter + c]);
    act[idx] = rn(g / (1.0f + expf(-g)) * u);
}

__global__ void cx_advance_kernel(int* ctl, int frames) { ctl[0] += frames; }

inline unsigned blocks(size_t n, int per) { return (unsigned)((n + per - 1) / per); }

}

extern "C" int nv_kernels_voxtral_tts_codec_im2col(
    void* stream, const uint16_t* src, const uint16_t* hist, const int* ctl, float* dst, int rows, int channels, int taps, int mode
) {
    if (rows < 1 || channels < 1 || taps < 2 || mode < kPadZero || mode > kPadReflect) return -1;
    if (mode == kPadReflect && rows < taps) return -2;
    size_t n = (size_t)rows * taps * channels;
    cx_im2col_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(src), reinterpret_cast<const __nv_bfloat16*>(hist), ctl, dst, rows, channels, taps, mode);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_codec_carry(void* stream, const float* cols, uint16_t* hist, int rows, int channels, int taps) {
    if (rows < 1 || channels < 1 || taps < 2) return -1;
    int n = (taps - 1) * channels;
    cx_carry_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t)stream>>>(cols, reinterpret_cast<__nv_bfloat16*>(hist), rows, channels, taps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_codec_cast_norm(void* stream, const float* src, uint16_t* x, const uint16_t* w, uint16_t* h, int rows, float eps) {
    if (rows < 1) return -1;
    cx_cast_norm_kernel<<<rows, kRowThreads, 0, (cudaStream_t)stream>>>(
        src, reinterpret_cast<__nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(w), reinterpret_cast<__nv_bfloat16*>(h), eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_codec_qk_norm(
    void* stream, const uint16_t* qkv, const uint16_t* qw, const uint16_t* kw, uint16_t* qn, uint16_t* kn, int rows, float eps
) {
    if (rows < 1) return -1;
    cx_qk_norm_kernel<<<dim3(rows, 2), kRowThreads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(qkv), reinterpret_cast<const __nv_bfloat16*>(qw), reinterpret_cast<const __nv_bfloat16*>(kw),
        reinterpret_cast<__nv_bfloat16*>(qn), reinterpret_cast<__nv_bfloat16*>(kn), eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_codec_residual_norm(
    void* stream, uint16_t* x, const uint16_t* t, const uint16_t* scale, const uint16_t* w_next, uint16_t* h, int rows, float eps
) {
    if (rows < 1) return -1;
    cx_residual_norm_kernel<<<rows, kRowThreads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<__nv_bfloat16*>(x), reinterpret_cast<const __nv_bfloat16*>(t), reinterpret_cast<const __nv_bfloat16*>(scale),
        reinterpret_cast<const __nv_bfloat16*>(w_next), reinterpret_cast<__nv_bfloat16*>(h), eps);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_codec_attn(
    void* stream, const uint16_t* q, const uint16_t* k, const uint16_t* qkv, const uint16_t* kring, const uint16_t* vring,
    const int* ctl, uint16_t* out, int rows, int window, int rows_per_frame, float scale
) {
    if (rows < 1 || window < 1 || window > kMaxKeys - 1 || (window & (window - 1)) || rows_per_frame < 1) return -1;
    cx_attn_kernel<<<dim3(rows, kHeads), kHeadDim, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), reinterpret_cast<const __nv_bfloat16*>(k), reinterpret_cast<const __nv_bfloat16*>(qkv),
        reinterpret_cast<const __nv_bfloat16*>(kring), reinterpret_cast<const __nv_bfloat16*>(vring), ctl,
        reinterpret_cast<__nv_bfloat16*>(out), window, rows_per_frame, scale);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_codec_kv_commit(
    void* stream, const uint16_t* k, const uint16_t* qkv, uint16_t* kring, uint16_t* vring, const int* ctl, int rows, int window, int rows_per_frame
) {
    if (rows < 1 || window < 1 || (window & (window - 1)) || rows_per_frame < 1) return -1;
    int cnt = rows < window ? rows : window;
    cx_kv_commit_kernel<<<blocks((size_t)cnt * kDim, 256), 256, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k), reinterpret_cast<const __nv_bfloat16*>(qkv), reinterpret_cast<__nv_bfloat16*>(kring),
        reinterpret_cast<__nv_bfloat16*>(vring), ctl, rows, window, rows_per_frame);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_codec_silu_mul(void* stream, const uint16_t* gu, uint16_t* act, int rows, int inter) {
    if (rows < 1 || inter < 1) return -1;
    cx_silu_mul_kernel<<<blocks((size_t)rows * inter, 256), 256, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(gu), reinterpret_cast<__nv_bfloat16*>(act), rows, inter);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_voxtral_tts_codec_advance(void* stream, int* ctl, int frames) {
    cx_advance_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(ctl, frames);
    return (int)cudaGetLastError();
}
