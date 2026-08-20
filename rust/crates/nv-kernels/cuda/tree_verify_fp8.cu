
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <math.h>
#include "nvk_grid.cuh"

namespace treeverifyfp8 {

constexpr int kWarp = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarp * kWarps;
constexpr int kMaxHD = 512;
constexpr int kMaxAcc = kMaxHD / kWarp;
constexpr float kFp8Max = 448.0f;

__inline__ __device__ float warp_sum(float x) {
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__inline__ __device__ float warp_max(float x) {
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}

__inline__ __device__ float2 tv_fp8x2_to_float2(unsigned short packed) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed, __NV_E4M3);
    __half2 h2 = *reinterpret_cast<__half2*>(&hr);
    return __half22float2(h2);
}

__global__ void tree_verify_attn_fp8_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ kc,
    const uint8_t* __restrict__ vc,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int* __restrict__ n_committed,
    const unsigned char* __restrict__ mask,
    const int* __restrict__ positions,
    __nv_bfloat16* __restrict__ out,
    int NH, int NKV, int HD, int K, int window, int ring
) {
    const int h = blockIdx.x;
    const int qi = blockIdx.y;
    if (h >= NH || qi >= K) return;

    const int nc = n_committed[0];
    const int group = NH / NKV;
    const int kvh = h / group;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

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
    const int vcn = HD / kWarp;
    const bool vecv = (HD % (kWarp * 8)) == 0 && vcn <= kMaxAcc;
    const int nck = HD >> 7;
    const int nv2 = vcn >> 3;

    if (vec4 && vecv && (HD & 127) == 0) {
        uchar4 kcur[kMaxAcc / 4];
        uint2 vcur[kMaxAcc / 8];
        float kscur = 0.0f, vscur = 0.0f;
        uchar4 knx[kMaxAcc / 4];
        uint2 vnx[kMaxAcc / 8];
        float ksnx = 0.0f, vsnx = 0.0f;
        int p = win_start + warp;
        if (p < nc) {
            const int sp = (ring > 0) ? (p % ring) : p;
            const uchar4* k4 = reinterpret_cast<const uchar4*>(
                kc + ((size_t)sp * NKV + kvh) * HD);
            #pragma unroll
            for (int c = 0; c < kMaxAcc / 4; ++c)
                if (c < nck) kcur[c] = __ldg(&k4[lane + c * kWarp]);
            const uint2* v8 = reinterpret_cast<const uint2*>(
                vc + ((size_t)sp * NKV + kvh) * HD + lane * vcn);
            #pragma unroll
            for (int t = 0; t < kMaxAcc / 8; ++t)
                if (t < nv2) vcur[t] = __ldg(&v8[t]);
            kscur = k_scale[(size_t)sp * NKV + kvh];
            vscur = v_scale[(size_t)sp * NKV + kvh];
        }
        if (p + kWarps < nc) {
            const int spn = (ring > 0) ? ((p + kWarps) % ring) : (p + kWarps);
            const uchar4* k4 = reinterpret_cast<const uchar4*>(
                kc + ((size_t)spn * NKV + kvh) * HD);
            #pragma unroll
            for (int c = 0; c < kMaxAcc / 4; ++c)
                if (c < nck) knx[c] = __ldg(&k4[lane + c * kWarp]);
            const uint2* v8 = reinterpret_cast<const uint2*>(
                vc + ((size_t)spn * NKV + kvh) * HD + lane * vcn);
            #pragma unroll
            for (int t = 0; t < kMaxAcc / 8; ++t)
                if (t < nv2) vnx[t] = __ldg(&v8[t]);
            ksnx = k_scale[(size_t)spn * NKV + kvh];
            vsnx = v_scale[(size_t)spn * NKV + kvh];
        }
        for (; p < nc; p += kWarps) {
            const int pn = p + 2 * kWarps;
            uchar4 knn[kMaxAcc / 4];
            uint2 vnn[kMaxAcc / 8];
            float ksnn = 0.0f, vsnn = 0.0f;
            if (pn < nc) {
                const int spn = (ring > 0) ? (pn % ring) : pn;
                const uchar4* k4 = reinterpret_cast<const uchar4*>(
                    kc + ((size_t)spn * NKV + kvh) * HD);
                #pragma unroll
                for (int c = 0; c < kMaxAcc / 4; ++c)
                    if (c < nck) knn[c] = __ldg(&k4[lane + c * kWarp]);
                const uint2* v8 = reinterpret_cast<const uint2*>(
                    vc + ((size_t)spn * NKV + kvh) * HD + lane * vcn);
                #pragma unroll
                for (int t = 0; t < kMaxAcc / 8; ++t)
                    if (t < nv2) vnn[t] = __ldg(&v8[t]);
                ksnn = k_scale[(size_t)spn * NKV + kvh];
                vsnn = v_scale[(size_t)spn * NKV + kvh];
            }
            float partial = 0.0f;
            #pragma unroll
            for (int c = 0; c < kMaxAcc / 4; ++c) {
                if (c >= nck) break;
                uchar4 raw = kcur[c];
                const float* qp = qsh + (lane + c * kWarp) * 4;
                float2 f01 = tv_fp8x2_to_float2(
                    (unsigned short)(raw.x | ((unsigned short)raw.y << 8)));
                float2 f23 = tv_fp8x2_to_float2(
                    (unsigned short)(raw.z | ((unsigned short)raw.w << 8)));
                partial += qp[0] * f01.x
                         + qp[1] * f01.y
                         + qp[2] * f23.x
                         + qp[3] * f23.y;
            }
            float score = warp_sum(partial) * kscur;
            float m_new = fmaxf(m, score);
            float corr = __expf(m - m_new);
            float w = __expf(score - m_new);
            l = l * corr + w;
            const float w_v = w * vscur;
            #pragma unroll
            for (int t = 0; t < kMaxAcc / 8; ++t) {
                if (t >= nv2) break;
                uint2 raw = vcur[t];
                float2 f0 = tv_fp8x2_to_float2((unsigned short)(raw.x & 0xffffu));
                float2 f1 = tv_fp8x2_to_float2((unsigned short)(raw.x >> 16));
                float2 f2 = tv_fp8x2_to_float2((unsigned short)(raw.y & 0xffffu));
                float2 f3 = tv_fp8x2_to_float2((unsigned short)(raw.y >> 16));
                acc[t * 8 + 0] = acc[t * 8 + 0] * corr + w_v * f0.x;
                acc[t * 8 + 1] = acc[t * 8 + 1] * corr + w_v * f0.y;
                acc[t * 8 + 2] = acc[t * 8 + 2] * corr + w_v * f1.x;
                acc[t * 8 + 3] = acc[t * 8 + 3] * corr + w_v * f1.y;
                acc[t * 8 + 4] = acc[t * 8 + 4] * corr + w_v * f2.x;
                acc[t * 8 + 5] = acc[t * 8 + 5] * corr + w_v * f2.y;
                acc[t * 8 + 6] = acc[t * 8 + 6] * corr + w_v * f3.x;
                acc[t * 8 + 7] = acc[t * 8 + 7] * corr + w_v * f3.y;
            }
            m = m_new;
            if (p + kWarps < nc) {
                #pragma unroll
                for (int c = 0; c < kMaxAcc / 4; ++c) kcur[c] = knx[c];
                #pragma unroll
                for (int t = 0; t < kMaxAcc / 8; ++t) vcur[t] = vnx[t];
                kscur = ksnx;
                vscur = vsnx;
            }
            if (pn < nc) {
                #pragma unroll
                for (int c = 0; c < kMaxAcc / 4; ++c) knx[c] = knn[c];
                #pragma unroll
                for (int t = 0; t < kMaxAcc / 8; ++t) vnx[t] = vnn[t];
                ksnx = ksnn;
                vsnx = vsnn;
            }
        }
    } else
    for (int p = win_start + warp; p < nc; p += kWarps) {
        const int sp = (ring > 0) ? (p % ring) : p;
        const uint8_t* kp = kc + ((size_t)sp * NKV + kvh) * HD;
        float ks = k_scale[(size_t)sp * NKV + kvh];
        float partial = 0.0f;
        if (vec4) {
            const uchar4* k4 = reinterpret_cast<const uchar4*>(kp);
            const int n4 = HD >> 2;
            for (int j = lane; j < n4; j += kWarp) {
                uchar4 raw = __ldg(&k4[j]);
                const float* qp = qsh + j * 4;
                float2 f01 = tv_fp8x2_to_float2(
                    (unsigned short)(raw.x | ((unsigned short)raw.y << 8)));
                float2 f23 = tv_fp8x2_to_float2(
                    (unsigned short)(raw.z | ((unsigned short)raw.w << 8)));
                partial += qp[0] * f01.x
                         + qp[1] * f01.y
                         + qp[2] * f23.x
                         + qp[3] * f23.y;
            }
        } else {
            for (int d = lane; d < HD; d += kWarp) {
                partial += qsh[d] * tv_fp8x2_to_float2((unsigned short)kp[d]).x;
            }
        }
        float score = warp_sum(partial) * ks;
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;
        const uint8_t* vp = vc + ((size_t)sp * NKV + kvh) * HD;
        float vs = v_scale[(size_t)sp * NKV + kvh];
        const float w_v = w * vs;
        if (vecv) {
            const uint2* v8 = reinterpret_cast<const uint2*>(vp + lane * vcn);
            #pragma unroll
            for (int t = 0; t < kMaxAcc / 8; ++t) {
                if (t >= vcn / 8) break;
                uint2 raw = __ldg(&v8[t]);
                float2 f0 = tv_fp8x2_to_float2((unsigned short)(raw.x & 0xffffu));
                float2 f1 = tv_fp8x2_to_float2((unsigned short)(raw.x >> 16));
                float2 f2 = tv_fp8x2_to_float2((unsigned short)(raw.y & 0xffffu));
                float2 f3 = tv_fp8x2_to_float2((unsigned short)(raw.y >> 16));
                acc[t * 8 + 0] = acc[t * 8 + 0] * corr + w_v * f0.x;
                acc[t * 8 + 1] = acc[t * 8 + 1] * corr + w_v * f0.y;
                acc[t * 8 + 2] = acc[t * 8 + 2] * corr + w_v * f1.x;
                acc[t * 8 + 3] = acc[t * 8 + 3] * corr + w_v * f1.y;
                acc[t * 8 + 4] = acc[t * 8 + 4] * corr + w_v * f2.x;
                acc[t * 8 + 5] = acc[t * 8 + 5] * corr + w_v * f2.y;
                acc[t * 8 + 6] = acc[t * 8 + 6] * corr + w_v * f3.x;
                acc[t * 8 + 7] = acc[t * 8 + 7] * corr + w_v * f3.y;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kMaxAcc; ++i) {
                int d = lane + i * kWarp;
                if (d < HD) {
                    acc[i] = acc[i] * corr
                        + w_v * tv_fp8x2_to_float2((unsigned short)__ldg(&vp[d])).x;
                }
            }
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
        for (int d = lane; d < HD; d += kWarp) {
            partial += qsh[d] * tv_fp8x2_to_float2((unsigned short)kp[d]).x;
        }
        float score = warp_sum(partial) * ks;
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;
        const uint8_t* vp = vc + ((size_t)sp * NKV + kvh) * HD;
        float vs = v_scale[(size_t)sp * NKV + kvh];
        const float w_v2 = w * vs;
        if (vecv) {
            const uint2* v8 = reinterpret_cast<const uint2*>(vp + lane * vcn);
            #pragma unroll
            for (int t = 0; t < kMaxAcc / 8; ++t) {
                if (t >= vcn / 8) break;
                uint2 raw = __ldg(&v8[t]);
                float2 f0 = tv_fp8x2_to_float2((unsigned short)(raw.x & 0xffffu));
                float2 f1 = tv_fp8x2_to_float2((unsigned short)(raw.x >> 16));
                float2 f2 = tv_fp8x2_to_float2((unsigned short)(raw.y & 0xffffu));
                float2 f3 = tv_fp8x2_to_float2((unsigned short)(raw.y >> 16));
                acc[t * 8 + 0] = acc[t * 8 + 0] * corr + w_v2 * f0.x;
                acc[t * 8 + 1] = acc[t * 8 + 1] * corr + w_v2 * f0.y;
                acc[t * 8 + 2] = acc[t * 8 + 2] * corr + w_v2 * f1.x;
                acc[t * 8 + 3] = acc[t * 8 + 3] * corr + w_v2 * f1.y;
                acc[t * 8 + 4] = acc[t * 8 + 4] * corr + w_v2 * f2.x;
                acc[t * 8 + 5] = acc[t * 8 + 5] * corr + w_v2 * f2.y;
                acc[t * 8 + 6] = acc[t * 8 + 6] * corr + w_v2 * f3.x;
                acc[t * 8 + 7] = acc[t * 8 + 7] * corr + w_v2 * f3.y;
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kMaxAcc; ++i) {
                int d = lane + i * kWarp;
                if (d < HD) {
                    acc[i] = acc[i] * corr
                        + w_v2 * tv_fp8x2_to_float2((unsigned short)vp[d]).x;
                }
            }
        }
        m = m_new;
    }

    __shared__ float sm[kWarps];
    __shared__ float sl[kWarps];
    __shared__ float sacc[kWarps][kMaxHD];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    if (vecv) {
        #pragma unroll
        for (int i = 0; i < kMaxAcc; ++i) {
            if (i >= vcn) break;
            sacc[warp][lane * vcn + i] = acc[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < kMaxAcc; ++i) {
            int d = lane + i * kWarp;
            if (d < HD) sacc[warp][d] = acc[i];
        }
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
        for (int d = lane; d < HD; d += kWarp) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w)
                a += (sm[w] > -INFINITY) ? sacc[w][d] * __expf(sm[w] - m_glob) : 0.0f;
            out[((size_t)qi * NH + h) * HD + d] = __float2bfloat16(a * inv);
        }
    }
}

__global__ void kv_append_fp8_kernel(
    const __nv_bfloat16* __restrict__ k_new,
    const __nv_bfloat16* __restrict__ v_new,
    uint8_t* __restrict__ kc,
    uint8_t* __restrict__ vc,
    float* __restrict__ k_scale,
    float* __restrict__ v_scale,
    const int* __restrict__ n_committed,
    int K, int NKV, int HD, int ring
) {
    int kvh = blockIdx.x;
    int token = blockIdx.y;
    if (token >= K || kvh >= NKV) return;
    int slot = *n_committed + token;
    if (ring > 0) {
        if (token + ring < K) return;
        slot = slot % ring;
    }
    size_t base_src = ((size_t)token * NKV + kvh) * HD;
    size_t base_dst = ((size_t)slot * NKV + kvh) * HD;
    int tid = threadIdx.x;

    float lmK = 0.0f, lmV = 0.0f;
    for (int d = tid; d < HD; d += blockDim.x) {
        lmK = fmaxf(lmK, fabsf(__bfloat162float(k_new[base_src + d])));
        lmV = fmaxf(lmV, fabsf(__bfloat162float(v_new[base_src + d])));
    }
    lmK = warp_max(lmK);
    lmV = warp_max(lmV);
    __shared__ float wmK[kWarp];
    __shared__ float wmV[kWarp];
    int warp = tid >> 5, lane = tid & 31;
    if (lane == 0) { wmK[warp] = lmK; wmV[warp] = lmV; }
    __syncthreads();
    if (warp == 0) {
        int nw = (blockDim.x + 31) >> 5;
        float vK = (lane < nw) ? wmK[lane] : 0.0f;
        float vV = (lane < nw) ? wmV[lane] : 0.0f;
        vK = warp_max(vK); vV = warp_max(vV);
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
        __nv_fp8_e4m3 ek = static_cast<__nv_fp8_e4m3>(__bfloat162float(k_new[base_src + d]) * invK);
        __nv_fp8_e4m3 ev = static_cast<__nv_fp8_e4m3>(__bfloat162float(v_new[base_src + d]) * invV);
        kc[base_dst + d] = ek.__x;
        vc[base_dst + d] = ev.__x;
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
    if (K > 65535) return NVK_ERR_GRID_AXIS;
    dim3 grid((unsigned)NH, (unsigned)K);
    treeverifyfp8::tree_verify_attn_fp8_kernel<<<grid, treeverifyfp8::kThreads, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), kc, vc, k_scale, v_scale,
        n_committed, mask, positions, reinterpret_cast<__nv_bfloat16*>(out), NH, NKV, HD, K, window, ring);
    return (int)cudaGetLastError();
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
    if (K > 65535) return NVK_ERR_GRID_AXIS;
    int block = HD; if (block > 512) block = 512; if (block < 32) block = 32;
    dim3 grid((unsigned)NKV, (unsigned)K);
    treeverifyfp8::kv_append_fp8_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k_new),
        reinterpret_cast<const __nv_bfloat16*>(v_new),
        kc, vc, k_scale, v_scale, n_committed, K, NKV, HD, ring);
    return (int)cudaGetLastError();
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
    cudaStream_t s = (cudaStream_t)stream;
    treeverifyfp8::kv_gather_fp8_kernel<<<grid, block, 0, s>>>(
        kc, vc, k_scale, v_scale, sk, sv, ssk, ssv, path, base, A, NKV, HD, ring);
    treeverifyfp8::kv_scatter_fp8_kernel<<<grid, block, 0, s>>>(
        kc, vc, k_scale, v_scale, sk, sv, ssk, ssv, base, A, NKV, HD, ring);
    return (int)cudaGetLastError();
}
