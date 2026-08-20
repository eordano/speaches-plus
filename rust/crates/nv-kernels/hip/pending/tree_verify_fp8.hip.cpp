#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <math.h>

#include "nv_kernels.h"
#include "nv_hip_wave.h"

namespace treeverifyfp8 {

constexpr int kThreads = 256;
constexpr int kMaxHD = 512;
constexpr int kMaxAppendBlock = 512;
constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float e4m3_ocp_to_float(uint8_t b) {
    uint32_t s = (uint32_t)(b & 0x80u) << 24;
    uint32_t e = (uint32_t)(b >> 3) & 0xFu;
    uint32_t m = (uint32_t)b & 0x7u;
    uint32_t out;
    if (e == 0u) {
        if (m == 0u) {
            out = s;
        } else {
            int k = 31 - __builtin_clz(m);
            out = s | ((uint32_t)(127 + k - 9) << 23) | ((m - (1u << k)) << (23 - k));
        }
    } else if (e == 0xFu && m == 0x7u) {
        out = s | 0x7FC00000u;
    } else {
        out = s | ((e + 120u) << 23) | (m << 20);
    }
    return __uint_as_float(out);
}

__device__ __forceinline__ uint8_t float_to_e4m3_ocp(float f) {
    uint32_t u = __float_as_uint(f);
    uint32_t s = (u >> 24) & 0x80u;
    uint32_t a = u & 0x7FFFFFFFu;
    if (a >= 0x7F800000u) return (uint8_t)(s | (a > 0x7F800000u ? 0x7Fu : 0x7Eu));
    if (a == 0u) return (uint8_t)s;
    int ne = (int)((a >> 23) & 0xFFu) - 120;
    uint32_t mant = a & 0x7FFFFFu;
    uint32_t r;
    if (ne <= 0) {
        int sh = 1 - ne;
        if (sh > 24) return (uint8_t)s;
        uint32_t mm = mant | 0x800000u;
        int rs = 20 + sh;
        uint32_t q = mm >> rs;
        uint32_t rem = mm & ((1u << rs) - 1u);
        uint32_t half = 1u << (rs - 1);
        if (rem > half || (rem == half && (q & 1u))) ++q;
        r = q;
    } else {
        uint32_t q = mant >> 20;
        uint32_t rem = mant & 0xFFFFFu;
        if (rem > 0x80000u || (rem == 0x80000u && (q & 1u))) ++q;
        if (q == 8u) { q = 0u; ++ne; }
        r = (ne > 15 || (ne == 15 && q > 6u)) ? 0x7Eu : (((uint32_t)ne << 3) | q);
    }
    return (uint8_t)(s | r);
}

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
__global__ void tree_verify_attn_fp8_kernel(
    const __hip_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kc,
    const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int* __restrict__ n_committed,
    const unsigned char* __restrict__ mask,
    const int* __restrict__ positions,
    __hip_bfloat16* __restrict__ out,
    int NH, int NKV, int HD, int K, int window, int ring
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

    const bool vec4 = (HD & 3) == 0;
    for (int p = win_start + warp; p < nc; p += kWarps) {
        const int sp = (ring > 0) ? (p % ring) : p;
        const uint8_t* kp = kc + ((size_t)sp * NKV + kvh) * HD;
        float ks = k_scale[(size_t)sp * NKV + kvh];
        float partial = 0.0f;
        if (vec4) {
            const uchar4* k4 = reinterpret_cast<const uchar4*>(kp);
            const int n4 = HD >> 2;
            for (int j = lane; j < n4; j += WAVE) {
                uchar4 raw = k4[j];
                const float* qp = qsh + j * 4;
                partial += qp[0] * e4m3_ocp_to_float(raw.x)
                         + qp[1] * e4m3_ocp_to_float(raw.y)
                         + qp[2] * e4m3_ocp_to_float(raw.z)
                         + qp[3] * e4m3_ocp_to_float(raw.w);
            }
        } else {
            for (int d = lane; d < HD; d += WAVE)
                partial += qsh[d] * e4m3_ocp_to_float(kp[d]);
        }
        float score = nv_hip::wave_sum<WAVE>(partial) * ks;
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;
        const uint8_t* vp = vc + ((size_t)sp * NKV + kvh) * HD;
        float vs = v_scale[(size_t)sp * NKV + kvh];
        const float w_v = w * vs;
        #pragma unroll
        for (int i = 0; i < kMaxAcc; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) acc[i] = acc[i] * corr + w_v * e4m3_ocp_to_float(vp[d]);
        }
        m = m_new;
    }

    for (int j = warp; j < K; j += kWarps) {
        if (mask[(size_t)qi * K + j] == 0) continue;
        if (window > 0 && qpos - positions[j] >= window) continue;
        int p = nc + j;
        const int sp = (ring > 0) ? (p % ring) : p;
        const uint8_t* kp = kc + ((size_t)sp * NKV + kvh) * HD;
        float ks = k_scale[(size_t)sp * NKV + kvh];
        float partial = 0.0f;
        for (int d = lane; d < HD; d += WAVE)
            partial += qsh[d] * e4m3_ocp_to_float(kp[d]);
        float score = nv_hip::wave_sum<WAVE>(partial) * ks;
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;
        const uint8_t* vp = vc + ((size_t)sp * NKV + kvh) * HD;
        float vs = v_scale[(size_t)sp * NKV + kvh];
        #pragma unroll
        for (int i = 0; i < kMaxAcc; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) acc[i] = acc[i] * corr + w * (e4m3_ocp_to_float(vp[d]) * vs);
        }
        m = m_new;
    }

    __shared__ float sm[kWarps];
    __shared__ float sl[kWarps];
    __shared__ float sacc[kWarps][kMaxHD];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
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

template <int WAVE>
__global__ void kv_append_fp8_kernel(
    const __hip_bfloat16* __restrict__ k_new,
    const __hip_bfloat16* __restrict__ v_new,
    uint8_t* __restrict__ kc,
    uint8_t* __restrict__ vc,
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
    const int* __restrict__ n_committed,
    int K, int NKV, int HD, int ring
) {
    constexpr int kMaxAppendWaves = kMaxAppendBlock / WAVE;

    int kvh = blockIdx.x;
    int token = blockIdx.y;
    if (token >= K || kvh >= NKV) return;
    int slot = *n_committed + token;
    if (ring > 0) slot = slot % ring;
    size_t base_src = ((size_t)token * NKV + kvh) * HD;
    size_t base_dst = ((size_t)slot * NKV + kvh) * HD;
    int tid = threadIdx.x;

    float lmK = 0.0f, lmV = 0.0f;
    for (int d = tid; d < HD; d += blockDim.x) {
        lmK = fmaxf(lmK, fabsf(__bfloat162float(k_new[base_src + d])));
        lmV = fmaxf(lmV, fabsf(__bfloat162float(v_new[base_src + d])));
    }
    lmK = nv_hip::wave_max<WAVE>(lmK);
    lmV = nv_hip::wave_max<WAVE>(lmV);
    __shared__ float wmK[kMaxAppendWaves];
    __shared__ float wmV[kMaxAppendWaves];
    int warp = tid / WAVE, lane = tid & (WAVE - 1);
    if (lane == 0) { wmK[warp] = lmK; wmV[warp] = lmV; }
    __syncthreads();
    if (warp == 0) {
        int nw = ((int)blockDim.x + WAVE - 1) / WAVE;
        float vK = (lane < nw) ? wmK[lane] : 0.0f;
        float vV = (lane < nw) ? wmV[lane] : 0.0f;
        vK = nv_hip::wave_max<WAVE>(vK);
        vV = nv_hip::wave_max<WAVE>(vV);
        if (lane == 0) { wmK[0] = vK; wmV[0] = vV; }
    }
    __syncthreads();
    float amaxK = wmK[0], amaxV = wmV[0];
    float invK = (amaxK > 0.0f) ? (kFp8Max / amaxK) : 1.0f;
    float invV = (amaxV > 0.0f) ? (kFp8Max / amaxV) : 1.0f;
    if (tid == 0) {
        k_scale[(size_t)slot * NKV + kvh] = (amaxK > 0.0f) ? (amaxK / kFp8Max) : 1.0f;
        v_scale[(size_t)slot * NKV + kvh] = (amaxV > 0.0f) ? (amaxV / kFp8Max) : 1.0f;
    }
    for (int d = tid; d < HD; d += blockDim.x) {
        kc[base_dst + d] = float_to_e4m3_ocp(__bfloat162float(k_new[base_src + d]) * invK);
        vc[base_dst + d] = float_to_e4m3_ocp(__bfloat162float(v_new[base_src + d]) * invV);
    }
}

__global__ void kv_gather_fp8_kernel(
    const uint8_t* __restrict__ kc, const uint8_t* __restrict__ vc,
    const float* __restrict__ ksc, const float* __restrict__ vsc,
    uint8_t* __restrict__ sk, uint8_t* __restrict__ sv,
    float* __restrict__ ssk, float* __restrict__ ssv,
    const int* __restrict__ path, int base, int A, int NKV, int HD, int ring
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)NKV * HD;
    size_t total = (size_t)A * stride;
    if (idx >= total) return;
    int i = (int)(idx / stride);
    size_t e = idx - (size_t)i * stride;
    int srow = base + path[i];
    if (ring > 0) srow = srow % ring;
    size_t src = (size_t)srow * stride + e;
    sk[idx] = kc[src];
    sv[idx] = vc[src];
    if (e < (size_t)NKV) {
        size_t ssrc = (size_t)srow * NKV + e;
        ssk[(size_t)i * NKV + e] = ksc[ssrc];
        ssv[(size_t)i * NKV + e] = vsc[ssrc];
    }
}

__global__ void kv_scatter_fp8_kernel(
    uint8_t* __restrict__ kc, uint8_t* __restrict__ vc,
    float* __restrict__ ksc, float* __restrict__ vsc,
    const uint8_t* __restrict__ sk, const uint8_t* __restrict__ sv,
    const float* __restrict__ ssk, const float* __restrict__ ssv,
    int base, int A, int NKV, int HD, int ring
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)NKV * HD;
    size_t total = (size_t)A * stride;
    if (idx >= total) return;
    int i = (int)(idx / stride);
    size_t e = idx - (size_t)i * stride;
    int drow = base + i;
    if (ring > 0) drow = drow % ring;
    size_t dst = (size_t)drow * stride + e;
    kc[dst] = sk[idx];
    vc[dst] = sv[idx];
    if (e < (size_t)NKV) {
        size_t sdst = (size_t)drow * NKV + e;
        ksc[sdst] = ssk[(size_t)i * NKV + e];
        vsc[sdst] = ssv[(size_t)i * NKV + e];
    }
}

}

extern "C" int nv_kernels_tree_verify_attn_fp8(
    void* stream,
    const uint16_t* q,
    const uint8_t* kc,
    const uint8_t* vc,
    const float* k_scale,
    const float* v_scale,
    const int* n_committed,
    const unsigned char* mask,
    const int* positions,
    uint16_t* out,
    int NH, int NKV, int HD, int K, int window, int ring
) {
    if (NH <= 0 || NKV <= 0 || K <= 0) return 0;
    if (HD > treeverifyfp8::kMaxHD || (NH % NKV) != 0) return -1;
    if (window > 0 && positions == nullptr) return -2;
    if (ring < 0 || (ring > 0 && window <= 0)) return -3;
    dim3 grid((unsigned)NH, (unsigned)K);
    hipStream_t s = (hipStream_t)stream;
    const __hip_bfloat16* qb = reinterpret_cast<const __hip_bfloat16*>(q);
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(out);
    if (treeverifyfp8::wave_size_now() == 64) {
        treeverifyfp8::tree_verify_attn_fp8_kernel<64>
            <<<grid, treeverifyfp8::kThreads, 0, s>>>(
                qb, kc, vc, k_scale, v_scale, n_committed, mask, positions, ob,
                NH, NKV, HD, K, window, ring);
    } else {
        treeverifyfp8::tree_verify_attn_fp8_kernel<32>
            <<<grid, treeverifyfp8::kThreads, 0, s>>>(
                qb, kc, vc, k_scale, v_scale, n_committed, mask, positions, ob,
                NH, NKV, HD, K, window, ring);
    }
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_kv_append_fp8(
    void* stream,
    const uint16_t* k_new,
    const uint16_t* v_new,
    uint8_t* kc,
    uint8_t* vc,
    float* k_scale,
    float* v_scale,
    const int* n_committed,
    int K, int NKV, int HD, int ring
) {
    if (K <= 0 || NKV <= 0 || HD <= 0) return 0;
    if (ring < 0) return -3;
    const int wave = treeverifyfp8::wave_size_now();
    int block = HD;
    if (block > treeverifyfp8::kMaxAppendBlock) block = treeverifyfp8::kMaxAppendBlock;
    block = ((block + wave - 1) / wave) * wave;
    if (block > treeverifyfp8::kMaxAppendBlock) block = treeverifyfp8::kMaxAppendBlock;
    if (block < wave) block = wave;
    dim3 grid((unsigned)NKV, (unsigned)K);
    hipStream_t s = (hipStream_t)stream;
    const __hip_bfloat16* kb = reinterpret_cast<const __hip_bfloat16*>(k_new);
    const __hip_bfloat16* vb = reinterpret_cast<const __hip_bfloat16*>(v_new);
    if (wave == 64) {
        treeverifyfp8::kv_append_fp8_kernel<64><<<grid, block, 0, s>>>(
            kb, vb, kc, vc, k_scale, v_scale, n_committed, K, NKV, HD, ring);
    } else {
        treeverifyfp8::kv_append_fp8_kernel<32><<<grid, block, 0, s>>>(
            kb, vb, kc, vc, k_scale, v_scale, n_committed, K, NKV, HD, ring);
    }
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_kv_compact_fp8(
    void* stream,
    uint8_t* kc,
    uint8_t* vc,
    float* k_scale,
    float* v_scale,
    uint8_t* sk,
    uint8_t* sv,
    float* ssk,
    float* ssv,
    const int* path,
    int base, int A, int NKV, int HD, int ring
) {
    if (A <= 0 || NKV <= 0 || HD <= 0) return 0;
    if (ring < 0) return -3;
    size_t total = (size_t)A * NKV * HD;
    int block = 256;
    int grid = (int)((total + block - 1) / block);
    hipStream_t s = (hipStream_t)stream;
    treeverifyfp8::kv_gather_fp8_kernel<<<grid, block, 0, s>>>(
        kc, vc, k_scale, v_scale, sk, sv, ssk, ssv, path, base, A, NKV, HD, ring);
    treeverifyfp8::kv_scatter_fp8_kernel<<<grid, block, 0, s>>>(
        kc, vc, k_scale, v_scale, sk, sv, ssk, ssv, base, A, NKV, HD, ring);
    return (int)hipGetLastError();
}
