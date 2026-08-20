#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

#include "nv_kernels.h"
#include "nv_hip_wave.h"

namespace treeverify {

constexpr int kThreads = 256;
constexpr int kMaxHD = 512;

inline int wave_size_now() {
    static int cache[16] = {0};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess) dev = 0;
    if (dev < 0 || dev >= 16) dev = 0;
    int w = cache[dev];
    if (w == 0) {
        w = nv_hip::host_wave_size(dev);
        if (w != 32 && w != 64) w = 64;
        cache[dev] = w;
    }
    return w;
}

template <int WAVE>
__global__ void tree_verify_attn_kernel(
    const __hip_bfloat16* __restrict__ q,
    const __hip_bfloat16* __restrict__ kc,
    const __hip_bfloat16* __restrict__ vc,
    const int* __restrict__ n_committed,
    const unsigned char* __restrict__ mask,
    const int* __restrict__ positions,
    __hip_bfloat16* __restrict__ out,
    int NH,
    int NKV,
    int HD,
    int K,
    int window
) {
    constexpr int kWarps = kThreads / WAVE;
    constexpr int kMaxAcc = kMaxHD / WAVE;

    const int h = blockIdx.x;
    const int qi = blockIdx.y;
    if (h >= NH || qi >= K) return;

    const int nc = n_committed[0];
    const int group = NH / NKV;
    const int kvh = h / group;
    const int lane = threadIdx.x & (WAVE - 1);
    const int warp = threadIdx.x / WAVE;

    const int qpos = (window > 0) ? positions[qi] : 0;
    int win_start = 0;
    if (window > 0) {
        int s = qpos - (window - 1);
        if (s > 0) win_start = s;
    }

    __shared__ float qsh[kMaxHD];
    for (int d = threadIdx.x; d < HD; d += kThreads)
        qsh[d] = __bfloat162float(q[((size_t)qi * NH + h) * HD + d]);
    __syncthreads();

    float acc[kMaxAcc];
    #pragma unroll
    for (int i = 0; i < kMaxAcc; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int p = win_start + warp; p < nc; p += kWarps) {
        const __hip_bfloat16* kp = kc + ((size_t)p * NKV + kvh) * HD;
        float partial = 0.0f;
        for (int d = lane; d < HD; d += WAVE)
            partial += qsh[d] * __bfloat162float(kp[d]);
        float score = nv_hip::wave_sum<WAVE>(partial);
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;
        const __hip_bfloat16* vp = vc + ((size_t)p * NKV + kvh) * HD;
        #pragma unroll
        for (int i = 0; i < kMaxAcc; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) acc[i] = acc[i] * corr + w * __bfloat162float(vp[d]);
        }
        m = m_new;
    }

    for (int j = warp; j < K; j += kWarps) {
        if (mask[(size_t)qi * K + j] == 0) continue;
        if (window > 0 && qpos - positions[j] >= window) continue;
        int p = nc + j;
        const __hip_bfloat16* kp = kc + ((size_t)p * NKV + kvh) * HD;
        float partial = 0.0f;
        for (int d = lane; d < HD; d += WAVE)
            partial += qsh[d] * __bfloat162float(kp[d]);
        float score = nv_hip::wave_sum<WAVE>(partial);
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;
        const __hip_bfloat16* vp = vc + ((size_t)p * NKV + kvh) * HD;
        #pragma unroll
        for (int i = 0; i < kMaxAcc; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) acc[i] = acc[i] * corr + w * __bfloat162float(vp[d]);
        }
        m = m_new;
    }

    __shared__ float sm[kWarps];
    __shared__ float sl[kWarps];
    __shared__ float sacc[kWarps][kMaxHD];
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
    #pragma unroll
    for (int i = 0; i < kMaxAcc; ++i) {
        int d = lane + i * WAVE;
        if (d < HD) sacc[warp][d] = acc[i];
    }
    __syncthreads();

    if (warp == 0) {
        float m_glob = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) m_glob = fmaxf(m_glob, sm[w]);
        float l_glob = 0.0f;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w)
            l_glob += (sm[w] > -INFINITY) ? sl[w] * __expf(sm[w] - m_glob) : 0.0f;
        float inv = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
        for (int d = lane; d < HD; d += WAVE) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w)
                a += (sm[w] > -INFINITY) ? sacc[w][d] * __expf(sm[w] - m_glob) : 0.0f;
            out[((size_t)qi * NH + h) * HD + d] = __float2bfloat16(a * inv);
        }
    }
}

}

extern "C" int nv_kernels_tree_verify_attn_bf16(
    void* stream,
    const uint16_t* q,
    const uint16_t* kc,
    const uint16_t* vc,
    const int* n_committed,
    const unsigned char* mask,
    const int* positions,
    uint16_t* out,
    int NH,
    int NKV,
    int HD,
    int K,
    int window
) {
    if (NH <= 0 || NKV <= 0 || K <= 0) return 0;
    if (HD > treeverify::kMaxHD || (NH % NKV) != 0) return -1;
    if (window > 0 && positions == nullptr) return -2;
    dim3 grid((unsigned)NH, (unsigned)K);
    hipStream_t s = (hipStream_t)stream;
    const __hip_bfloat16* qb = reinterpret_cast<const __hip_bfloat16*>(q);
    const __hip_bfloat16* kb = reinterpret_cast<const __hip_bfloat16*>(kc);
    const __hip_bfloat16* vb = reinterpret_cast<const __hip_bfloat16*>(vc);
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(out);
    if (treeverify::wave_size_now() == 64) {
        treeverify::tree_verify_attn_kernel<64><<<grid, treeverify::kThreads, 0, s>>>(
            qb, kb, vb, n_committed, mask, positions, ob, NH, NKV, HD, K, window);
    } else {
        treeverify::tree_verify_attn_kernel<32><<<grid, treeverify::kThreads, 0, s>>>(
            qb, kb, vb, n_committed, mask, positions, ob, NH, NKV, HD, K, window);
    }
    return (int)hipGetLastError();
}

__global__ void kv_append_bf16_kernel(
    const __hip_bfloat16* __restrict__ k_new,
    const __hip_bfloat16* __restrict__ v_new,
    __hip_bfloat16* __restrict__ kc,
    __hip_bfloat16* __restrict__ vc,
    const int* __restrict__ n_committed,
    int K,
    int NKV,
    int HD
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)K * NKV * HD;
    if (idx >= total) return;
    size_t off = (size_t)n_committed[0] * NKV * HD;
    kc[off + idx] = k_new[idx];
    vc[off + idx] = v_new[idx];
}

extern "C" int nv_kernels_kv_append_bf16(
    void* stream,
    const uint16_t* k_new,
    const uint16_t* v_new,
    uint16_t* kc,
    uint16_t* vc,
    const int* n_committed,
    int K,
    int NKV,
    int HD
) {
    if (K <= 0 || NKV <= 0 || HD <= 0) return 0;
    size_t total = (size_t)K * NKV * HD;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    kv_append_bf16_kernel<<<grid, block, 0, (hipStream_t)stream>>>(
        reinterpret_cast<const __hip_bfloat16*>(k_new),
        reinterpret_cast<const __hip_bfloat16*>(v_new),
        reinterpret_cast<__hip_bfloat16*>(kc),
        reinterpret_cast<__hip_bfloat16*>(vc),
        n_committed, K, NKV, HD);
    return (int)hipGetLastError();
}

__global__ void kv_gather_bf16_kernel(
    const __hip_bfloat16* __restrict__ kc,
    const __hip_bfloat16* __restrict__ vc,
    __hip_bfloat16* __restrict__ sk,
    __hip_bfloat16* __restrict__ sv,
    const int* __restrict__ path,
    int base,
    int A,
    int stride
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)A * stride;
    if (idx >= total) return;
    int i = (int)(idx / stride);
    int e = (int)(idx - (size_t)i * stride);
    size_t src = (size_t)(base + path[i]) * stride + e;
    sk[idx] = kc[src];
    sv[idx] = vc[src];
}

__global__ void kv_scatter_bf16_kernel(
    __hip_bfloat16* __restrict__ kc,
    __hip_bfloat16* __restrict__ vc,
    const __hip_bfloat16* __restrict__ sk,
    const __hip_bfloat16* __restrict__ sv,
    int base,
    int A,
    int stride
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)A * stride;
    if (idx >= total) return;
    int i = (int)(idx / stride);
    int e = (int)(idx - (size_t)i * stride);
    size_t dst = (size_t)(base + i) * stride + e;
    kc[dst] = sk[idx];
    vc[dst] = sv[idx];
}

extern "C" int nv_kernels_kv_compact_bf16(
    void* stream,
    uint16_t* kc,
    uint16_t* vc,
    uint16_t* sk,
    uint16_t* sv,
    const int* path,
    int base,
    int A,
    int stride
) {
    if (A <= 0 || stride <= 0) return 0;
    size_t total = (size_t)A * stride;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    hipStream_t s = (hipStream_t)stream;
    kv_gather_bf16_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<const __hip_bfloat16*>(kc),
        reinterpret_cast<const __hip_bfloat16*>(vc),
        reinterpret_cast<__hip_bfloat16*>(sk),
        reinterpret_cast<__hip_bfloat16*>(sv),
        path, base, A, stride);
    kv_scatter_bf16_kernel<<<grid, block, 0, s>>>(
        reinterpret_cast<__hip_bfloat16*>(kc),
        reinterpret_cast<__hip_bfloat16*>(vc),
        reinterpret_cast<const __hip_bfloat16*>(sk),
        reinterpret_cast<const __hip_bfloat16*>(sv),
        base, A, stride);
    return (int)hipGetLastError();
}
