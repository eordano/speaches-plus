#include "hip/hip_runtime.h"

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "nv_kernels.h"
#include "nv_hip_wave.h"

namespace {

constexpr int kFlashThreads = 256;
constexpr int kMaxHD = 512;

template <typename T>
__device__ __forceinline__ T nv_ldg(const T* p) {
    return *p;
}

template <typename T>
__device__ __forceinline__ T from_f32(float x);

template <>
__device__ __forceinline__ float from_f32<float>(float x) { return x; }

template <>
__device__ __forceinline__ __hip_bfloat16 from_f32<__hip_bfloat16>(float x) {
    return __float2bfloat16(x);
}

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

template <typename OutT, int WAVE>
__global__ void flash_decode_kernel(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    OutT* __restrict__ out,
    const int* __restrict__ pos,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    constexpr int kFlashWarps = kFlashThreads / WAVE;
    constexpr int kMaxAccPerLane = kMaxHD / WAVE;

    const int h = blockIdx.x;
    if (h >= NH) return;

    const int total = pos[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (WAVE - 1);
    const int warp = threadIdx.x / WAVE;

    __shared__ float qsh[kMaxHD];
    for (int d = threadIdx.x; d < HD; d += kFlashThreads)
        qsh[d] = q[(size_t)h * HD + d];
    __syncthreads();

    float acc[kMaxAccPerLane];
    #pragma unroll
    for (int i = 0; i < kMaxAccPerLane; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    const bool vec4 = (HD & 3) == 0;

    for (int p = start + warp; p < total; p += kFlashWarps) {
        const float* kp = k + ((size_t)p * NKV + kvh) * HD;

        float partial = 0.0f;
        if (vec4) {
            const float4* q4 = reinterpret_cast<const float4*>(qsh);
            const float4* k4 = reinterpret_cast<const float4*>(kp);
            const int n4 = HD >> 2;
            for (int j = lane; j < n4; j += WAVE) {
                float4 a = q4[j];
                float4 b = k4[j];
                partial += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        } else {
            for (int d = lane; d < HD; d += WAVE)
                partial += qsh[d] * kp[d];
        }
        float score = nv_hip::wave_sum<WAVE>(partial);

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;

        const float* vp = v + ((size_t)p * NKV + kvh) * HD;
        #pragma unroll
        for (int i = 0; i < kMaxAccPerLane; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) acc[i] = acc[i] * corr + w * vp[d];
        }
        m = m_new;
    }

    __shared__ float sm[kFlashWarps];
    __shared__ float sl[kFlashWarps];

    __shared__ float sacc[kFlashWarps][kMaxHD];

    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
    #pragma unroll
    for (int i = 0; i < kMaxAccPerLane; ++i) {
        int d = lane + i * WAVE;
        if (d < HD) sacc[warp][d] = acc[i];
    }
    __syncthreads();

    if (warp == 0) {

        float m_glob = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w) m_glob = fmaxf(m_glob, sm[w]);

        float l_glob = 0.0f;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            l_glob += (sm[w] > -INFINITY) ? sl[w] * __expf(sm[w] - m_glob) : 0.0f;
        float inv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;

        float scale[kFlashWarps];
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            scale[w] = (sm[w] > -INFINITY) ? __expf(sm[w] - m_glob) : 0.0f;

        for (int d = lane; d < HD; d += WAVE) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w) a += sacc[w][d] * scale[w];
            out[(size_t)h * HD + d] = from_f32<OutT>(a * inv_l);
        }
    }
}

}

extern "C" int nv_kernels_flash_decode_dev_f32(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    float* out,
    const int* pos,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD > kMaxHD || (NH % NKV) != 0) return -1;
    hipStream_t s = (hipStream_t)stream;
    if (wave_size_now() == 64) {
        flash_decode_kernel<float, 64><<<(unsigned)NH, kFlashThreads, 0, s>>>(
            q, k, v, out, pos, NH, NKV, HD, WINDOW);
    } else {
        flash_decode_kernel<float, 32><<<(unsigned)NH, kFlashThreads, 0, s>>>(
            q, k, v, out, pos, NH, NKV, HD, WINDOW);
    }
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_flash_decode_dev_f32_bf16out(
    void* stream,
    const float* q,
    const float* k,
    const float* v,
    uint16_t* out,
    const int* pos,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD > kMaxHD || (NH % NKV) != 0) return -1;
    hipStream_t s = (hipStream_t)stream;
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(out);
    if (wave_size_now() == 64) {
        flash_decode_kernel<__hip_bfloat16, 64><<<(unsigned)NH, kFlashThreads, 0, s>>>(
            q, k, v, ob, pos, NH, NKV, HD, WINDOW);
    } else {
        flash_decode_kernel<__hip_bfloat16, 32><<<(unsigned)NH, kFlashThreads, 0, s>>>(
            q, k, v, ob, pos, NH, NKV, HD, WINDOW);
    }
    return (int)hipGetLastError();
}

namespace splitk {

constexpr int kSplits = 16;

template <int SPLITS, int WAVE>
__global__ void flash_splitk_stage1_kernel(
    const float* __restrict__ q,
    const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v,
    float* __restrict__ scratch,
    const int* __restrict__ pos,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    constexpr int kFlashWarps = kFlashThreads / WAVE;
    constexpr int kMaxAccPerLane = kMaxHD / WAVE;

    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    const int total = pos[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (WAVE - 1);
    const int warp = threadIdx.x / WAVE;

    __shared__ float qsh[kMaxHD];
    for (int d = threadIdx.x; d < HD; d += kFlashThreads)
        qsh[d] = q[(size_t)h * HD + d];
    __syncthreads();

    float acc[kMaxAccPerLane];
    #pragma unroll
    for (int i = 0; i < kMaxAccPerLane; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    const bool vec8 = (HD & 7) == 0;
    const int lane_stride = SPLITS * kFlashWarps;

    for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
        const __hip_bfloat16* kp = k + ((size_t)p * NKV + kvh) * HD;
        float partial = 0.0f;
        if (vec8) {
            const uint4* k8 = reinterpret_cast<const uint4*>(kp);
            const int n8 = HD >> 3;
            for (int j = lane; j < n8; j += WAVE) {
                uint4 raw = nv_ldg(&k8[j]);
                const __hip_bfloat162* kb = reinterpret_cast<const __hip_bfloat162*>(&raw);
                const float* qp = qsh + j * 8;
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    float2 kf = __bfloat1622float2(kb[t]);
                    partial += kf.x * qp[2 * t] + kf.y * qp[2 * t + 1];
                }
            }
        } else {
            for (int d = lane; d < HD; d += WAVE)
                partial += qsh[d] * __bfloat162float(kp[d]);
        }
        float score = nv_hip::wave_sum<WAVE>(partial);

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;

        const __hip_bfloat16* vp = v + ((size_t)p * NKV + kvh) * HD;
        #pragma unroll
        for (int i = 0; i < kMaxAccPerLane; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) acc[i] = acc[i] * corr + w * __bfloat162float(nv_ldg(&vp[d]));
        }
        m = m_new;
    }

    __shared__ float sm[kFlashWarps];
    __shared__ float sl[kFlashWarps];
    __shared__ float sacc[kFlashWarps][kMaxHD];
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
    #pragma unroll
    for (int i = 0; i < kMaxAccPerLane; ++i) {
        int d = lane + i * WAVE;
        if (d < HD) sacc[warp][d] = acc[i];
    }
    __syncthreads();

    float* out = scratch + ((size_t)h * SPLITS + split) * (HD + 2);
    if (warp == 0) {
        float m_blk = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
        float l_blk = 0.0f;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            l_blk += (sm[w] > -INFINITY) ? sl[w] * __expf(sm[w] - m_blk) : 0.0f;
        if (lane == 0) {
            out[0] = m_blk;
            out[1] = l_blk;
        }
    }
    __syncthreads();

    float m_blk = -INFINITY;
    #pragma unroll
    for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
    for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            a += (sm[w] > -INFINITY) ? sacc[w][d] * __expf(sm[w] - m_blk) : 0.0f;
        out[2 + d] = a;
    }
}

template <int SPLITS>
__global__ void flash_splitk_stage2_kernel(
    const float* __restrict__ scratch,
    __hip_bfloat16* __restrict__ out,
    int NH,
    int HD
) {
    const int h = blockIdx.x;
    if (h >= NH) return;
    const float* base = scratch + (size_t)h * SPLITS * (HD + 2);

    float m_glob = -INFINITY;
    #pragma unroll
    for (int s = 0; s < SPLITS; ++s)
        m_glob = fmaxf(m_glob, base[(size_t)s * (HD + 2)]);
    float l_glob = 0.0f;
    float scale[SPLITS];
    #pragma unroll
    for (int s = 0; s < SPLITS; ++s) {
        const float* part = base + (size_t)s * (HD + 2);
        float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
        scale[s] = sc;
        l_glob += part[1] * sc;
    }
    float inv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;

    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        float a = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s)
            a += base[(size_t)s * (HD + 2) + 2 + d] * scale[s];
        out[(size_t)h * HD + d] = __float2bfloat16(a * inv_l);
    }
}

template <int SPLITS, int WAVE>
__global__ void flash_splitk_fused_kernel(
    const float* __restrict__ q,
    const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __hip_bfloat16* __restrict__ outp,
    const int* __restrict__ pos,
    int delta,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    constexpr int kFlashWarps = kFlashThreads / WAVE;
    constexpr int kMaxAccPerLane = kMaxHD / WAVE;

    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    const int total = pos[0] - delta;
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (WAVE - 1);
    const int warp = threadIdx.x / WAVE;

    __shared__ float qsh[kMaxHD];
    for (int d = threadIdx.x; d < HD; d += kFlashThreads)
        qsh[d] = q[(size_t)h * HD + d];
    __syncthreads();

    float acc[kMaxAccPerLane];
    #pragma unroll
    for (int i = 0; i < kMaxAccPerLane; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    const bool vec8 = (HD & 7) == 0;

    const int vc = HD / WAVE;
    const bool vecv = (HD % (WAVE * 8)) == 0 && vc <= kMaxAccPerLane;
    const int lane_stride = SPLITS * kFlashWarps;

    if (vec8 && (HD >> 3) == WAVE && vecv) {

        int p = start + split * kFlashWarps + warp;
        uint4 kraw, vraw;
        if (p < total) {
            kraw = nv_ldg(reinterpret_cast<const uint4*>(k + ((size_t)p * NKV + kvh) * HD) + lane);
            vraw = nv_ldg(reinterpret_cast<const uint4*>(v + ((size_t)p * NKV + kvh) * HD) + lane);
        }
        for (; p < total; p += lane_stride) {
            const int pn = p + lane_stride;
            uint4 kn, vn;
            if (pn < total) {
                kn = nv_ldg(reinterpret_cast<const uint4*>(k + ((size_t)pn * NKV + kvh) * HD) + lane);
                vn = nv_ldg(reinterpret_cast<const uint4*>(v + ((size_t)pn * NKV + kvh) * HD) + lane);
            }
            const __hip_bfloat162* kb = reinterpret_cast<const __hip_bfloat162*>(&kraw);
            const float* qp = qsh + lane * 8;
            float partial = 0.0f;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                float2 kf = __bfloat1622float2(kb[t]);
                partial += kf.x * qp[2 * t] + kf.y * qp[2 * t + 1];
            }
            float score = nv_hip::wave_sum<WAVE>(partial);
            float m_new = fmaxf(m, score);
            float corr = __expf(m - m_new);
            float w = __expf(score - m_new);
            l = l * corr + w;
            const __hip_bfloat162* vb = reinterpret_cast<const __hip_bfloat162*>(&vraw);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float2 vf = __bfloat1622float2(vb[u]);
                acc[2 * u] = acc[2 * u] * corr + w * vf.x;
                acc[2 * u + 1] = acc[2 * u + 1] * corr + w * vf.y;
            }
            m = m_new;
            if (pn < total) {
                kraw = kn;
                vraw = vn;
            }
        }
    } else
    for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
        const __hip_bfloat16* kp = k + ((size_t)p * NKV + kvh) * HD;
        float partial = 0.0f;
        if (vec8) {
            const uint4* k8 = reinterpret_cast<const uint4*>(kp);
            const int n8 = HD >> 3;
            for (int j = lane; j < n8; j += WAVE) {
                uint4 raw = nv_ldg(&k8[j]);
                const __hip_bfloat162* kb = reinterpret_cast<const __hip_bfloat162*>(&raw);
                const float* qp = qsh + j * 8;
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    float2 kf = __bfloat1622float2(kb[t]);
                    partial += kf.x * qp[2 * t] + kf.y * qp[2 * t + 1];
                }
            }
        } else {
            for (int d = lane; d < HD; d += WAVE)
                partial += qsh[d] * __bfloat162float(kp[d]);
        }
        float score = nv_hip::wave_sum<WAVE>(partial);

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;

        const __hip_bfloat16* vp = v + ((size_t)p * NKV + kvh) * HD;
        if (vecv) {
            const uint4* v8 = reinterpret_cast<const uint4*>(vp + lane * vc);
            #pragma unroll
            for (int t = 0; t < kMaxAccPerLane / 8; ++t) {
                if (t >= vc / 8) break;
                uint4 raw = nv_ldg(&v8[t]);
                const __hip_bfloat162* vb = reinterpret_cast<const __hip_bfloat162*>(&raw);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    float2 vf = __bfloat1622float2(vb[u]);
                    acc[t * 8 + 2 * u] = acc[t * 8 + 2 * u] * corr + w * vf.x;
                    acc[t * 8 + 2 * u + 1] = acc[t * 8 + 2 * u + 1] * corr + w * vf.y;
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kMaxAccPerLane; ++i) {
                int d = lane + i * WAVE;
                if (d < HD) acc[i] = acc[i] * corr + w * __bfloat162float(nv_ldg(&vp[d]));
            }
        }
        m = m_new;
    }

    __shared__ float sm[kFlashWarps];
    __shared__ float sl[kFlashWarps];
    __shared__ float sacc[kFlashWarps][kMaxHD];
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
    if (vecv) {
        #pragma unroll
        for (int i = 0; i < kMaxAccPerLane; ++i) {
            if (i >= vc) break;
            sacc[warp][lane * vc + i] = acc[i];
        }
    } else {
        #pragma unroll
        for (int i = 0; i < kMaxAccPerLane; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) sacc[warp][d] = acc[i];
        }
    }
    __syncthreads();

    float* out = scratch + ((size_t)h * SPLITS + split) * (HD + 2);
    if (warp == 0) {
        float m_blk = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
        float l_blk = 0.0f;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            l_blk += (sm[w] > -INFINITY) ? sl[w] * __expf(sm[w] - m_blk) : 0.0f;
        if (lane == 0) {
            out[0] = m_blk;
            out[1] = l_blk;
        }
    }
    __syncthreads();
    float m_blk = -INFINITY;
    #pragma unroll
    for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
    for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            a += (sm[w] > -INFINITY) ? sacc[w][d] * __expf(sm[w] - m_blk) : 0.0f;
        out[2 + d] = a;
    }

    __syncthreads();
    __threadfence();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) {
        ticket = __hip_atomic_fetch_add(&fan_in[h], 1u, __ATOMIC_ACQ_REL,
                                        __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (ticket != SPLITS - 1) return;
    __threadfence();

    const float* base = scratch + (size_t)h * SPLITS * (HD + 2);
    __shared__ float ssc[32];
    __shared__ float sinv_l;
    if (threadIdx.x == 0) {
        float m_glob = -INFINITY;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s)
            m_glob = fmaxf(m_glob, base[(size_t)s * (HD + 2)]);
        float l_glob = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s) {
            const float* part = base + (size_t)s * (HD + 2);
            float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
            ssc[s] = sc;
            l_glob += part[1] * sc;
        }
        sinv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
    }
    __syncthreads();
    float inv_l = sinv_l;

    for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
        float a = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s)
            a += base[(size_t)s * (HD + 2) + 2 + d] * ssc[s];
        outp[(size_t)h * HD + d] = __float2bfloat16(a * inv_l);
    }
    if (threadIdx.x == 0) {
        __hip_atomic_store(&fan_in[h], 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

template <int SPLITS, int WAVE>
__global__ void flash_splitk_fused_fp8_kernel(
    const __hip_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __hip_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling
) {
    constexpr int kFlashWarps = kFlashThreads / WAVE;
    constexpr int kMaxAccPerLane = kMaxHD / WAVE;

    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    const int total = n_total_dev[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (WAVE - 1);
    const int warp = threadIdx.x / WAVE;

    __shared__ float qsh[kMaxHD];
    for (int d = threadIdx.x; d < HD; d += kFlashThreads)
        qsh[d] = __bfloat162float(q[(size_t)h * HD + d]);
    __syncthreads();

    float acc[kMaxAccPerLane];
    #pragma unroll
    for (int i = 0; i < kMaxAccPerLane; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    const bool vec4 = (HD & 3) == 0;
    const int lane_stride = SPLITS * kFlashWarps;

    for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
        const int sp = (RING > 0) ? (p % RING) : p;
        const uint8_t* kp = k_fp8 + ((size_t)sp * NKV + kvh) * HD;
        const float ks = k_scales[(size_t)sp * NKV + kvh];
        float partial = 0.0f;
        if (vec4) {
            const uchar4* k4 = reinterpret_cast<const uchar4*>(kp);
            const int n4 = HD >> 2;
            for (int j = lane; j < n4; j += WAVE) {
                uchar4 raw = nv_ldg(&k4[j]);
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
        float score = nv_hip::wave_sum<WAVE>(partial) * ks * scaling;

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;

        const uint8_t* vp = v_fp8 + ((size_t)sp * NKV + kvh) * HD;
        const float w_v = w * v_scales[(size_t)sp * NKV + kvh];
        #pragma unroll
        for (int i = 0; i < kMaxAccPerLane; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) {
                acc[i] = acc[i] * corr + w_v * e4m3_ocp_to_float(nv_ldg(&vp[d]));
            }
        }
        m = m_new;
    }

    __shared__ float sm[kFlashWarps];
    __shared__ float sl[kFlashWarps];
    __shared__ float sacc[kFlashWarps][kMaxHD];
    if (lane == 0) {
        sm[warp] = m;
        sl[warp] = l;
    }
    #pragma unroll
    for (int i = 0; i < kMaxAccPerLane; ++i) {
        int d = lane + i * WAVE;
        if (d < HD) sacc[warp][d] = acc[i];
    }
    __syncthreads();

    float* out = scratch + ((size_t)h * SPLITS + split) * (HD + 2);
    if (warp == 0) {
        float m_blk = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
        float l_blk = 0.0f;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            l_blk += (sm[w] > -INFINITY) ? sl[w] * __expf(sm[w] - m_blk) : 0.0f;
        if (lane == 0) {
            out[0] = m_blk;
            out[1] = l_blk;
        }
    }
    __syncthreads();
    float m_blk = -INFINITY;
    #pragma unroll
    for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
    for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
        float a = 0.0f;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w)
            a += (sm[w] > -INFINITY) ? sacc[w][d] * __expf(sm[w] - m_blk) : 0.0f;
        out[2 + d] = a;
    }

    __syncthreads();
    __threadfence();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) {
        ticket = __hip_atomic_fetch_add(&fan_in[h], 1u, __ATOMIC_ACQ_REL,
                                        __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (ticket != SPLITS - 1) return;
    __threadfence();

    const float* base = scratch + (size_t)h * SPLITS * (HD + 2);
    __shared__ float ssc[32];
    __shared__ float sinv_l;
    if (threadIdx.x == 0) {
        float m_glob = -INFINITY;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s)
            m_glob = fmaxf(m_glob, base[(size_t)s * (HD + 2)]);
        float l_glob = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s) {
            const float* part = base + (size_t)s * (HD + 2);
            float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
            ssc[s] = sc;
            l_glob += part[1] * sc;
        }
        sinv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
    }
    __syncthreads();
    float inv_l = sinv_l;

    for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
        float a = 0.0f;
        #pragma unroll
        for (int s = 0; s < SPLITS; ++s)
            a += base[(size_t)s * (HD + 2) + 2 + d] * ssc[s];
        outp[(size_t)h * HD + d] = __float2bfloat16(a * inv_l);
    }
    if (threadIdx.x == 0) {
        __hip_atomic_store(&fan_in[h], 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

inline int flash_splits_env() {
    static int v = [] {
        const char* e = getenv("NV_E4B_FLASH_SPLITS");
        if (e == nullptr) return (int)kSplits;
        int x = atoi(e);
        return (x == 8 || x == 16 || x == 32) ? x : (int)kSplits;
    }();
    return v;
}

}

extern "C" int nv_kernels_flash_splitk_scratch_elems(int NH, int HD) {
    return NH * splitk::flash_splits_env() * (HD + 2);
}

extern "C" int nv_kernels_flash_decode_splitk_bf16kv(
    void* stream,
    const float* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* pos,
    float* scratch,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD > kMaxHD || (NH % NKV) != 0) return -1;
    int sp = splitk::flash_splits_env();
    dim3 grid1((unsigned)NH, (unsigned)sp);
    const __hip_bfloat16* kb = reinterpret_cast<const __hip_bfloat16*>(k);
    const __hip_bfloat16* vb = reinterpret_cast<const __hip_bfloat16*>(v);
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(out);
    hipStream_t cs = (hipStream_t)stream;
    const int wave = wave_size_now();

#define NV_SK_STAGE(SP, W)                                                       \
    do {                                                                         \
        splitk::flash_splitk_stage1_kernel<SP, W>                                \
            <<<grid1, kFlashThreads, 0, cs>>>(q, kb, vb, scratch, pos,           \
                                              NH, NKV, HD, WINDOW);              \
        splitk::flash_splitk_stage2_kernel<SP>                                   \
            <<<(unsigned)NH, 256, 0, cs>>>(scratch, ob, NH, HD);                 \
    } while (0)

    if (wave == 64) {
        switch (sp) {
            case 8: NV_SK_STAGE(8, 64); break;
            case 32: NV_SK_STAGE(32, 64); break;
            default: NV_SK_STAGE(16, 64); break;
        }
    } else {
        switch (sp) {
            case 8: NV_SK_STAGE(8, 32); break;
            case 32: NV_SK_STAGE(32, 32); break;
            default: NV_SK_STAGE(16, 32); break;
        }
    }
#undef NV_SK_STAGE
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_flash_decode_fused_bf16kv(
    void* stream,
    const float* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* pos,
    int delta,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD > kMaxHD || (NH % NKV) != 0) return -1;
    int sp = splitk::flash_splits_env();
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __hip_bfloat16* kb = reinterpret_cast<const __hip_bfloat16*>(k);
    const __hip_bfloat16* vb = reinterpret_cast<const __hip_bfloat16*>(v);
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(out);
    hipStream_t cs = (hipStream_t)stream;
    const int wave = wave_size_now();

#define NV_SK_FUSED(SP, W)                                                       \
    splitk::flash_splitk_fused_kernel<SP, W>                                     \
        <<<grid, kFlashThreads, 0, cs>>>(q, kb, vb, scratch, fan_in, ob, pos,    \
                                         delta, NH, NKV, HD, WINDOW)

    if (wave == 64) {
        switch (sp) {
            case 8: NV_SK_FUSED(8, 64); break;
            case 32: NV_SK_FUSED(32, 64); break;
            default: NV_SK_FUSED(16, 64); break;
        }
    } else {
        switch (sp) {
            case 8: NV_SK_FUSED(8, 32); break;
            case 32: NV_SK_FUSED(32, 32); break;
            default: NV_SK_FUSED(16, 32); break;
        }
    }
#undef NV_SK_FUSED
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD > kMaxHD || (NH % NKV) != 0) return -1;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    int sp = splitk::flash_splits_env();
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __hip_bfloat16* qb = reinterpret_cast<const __hip_bfloat16*>(q);
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(out);
    hipStream_t cs = (hipStream_t)stream;
    const int wave = wave_size_now();

#define NV_SK_FUSED_FP8(SP, W)                                                   \
    splitk::flash_splitk_fused_fp8_kernel<SP, W>                                 \
        <<<grid, kFlashThreads, 0, cs>>>(qb, k_fp8, v_fp8, k_scales, v_scales,   \
                                         scratch, fan_in, ob, n_total_dev,       \
                                         NH, NKV, HD, WINDOW, RING, scaling)

    if (wave == 64) {
        switch (sp) {
            case 8: NV_SK_FUSED_FP8(8, 64); break;
            case 32: NV_SK_FUSED_FP8(32, 64); break;
            default: NV_SK_FUSED_FP8(16, 64); break;
        }
    } else {
        switch (sp) {
            case 8: NV_SK_FUSED_FP8(8, 32); break;
            case 32: NV_SK_FUSED_FP8(32, 32); break;
            default: NV_SK_FUSED_FP8(16, 32); break;
        }
    }
#undef NV_SK_FUSED_FP8
    return (int)hipGetLastError();
}

namespace splitk_mk {

constexpr int kMaxHDmk = 256;
constexpr int kMaxM = 8;

template <int SPLITS, int M, int WAVE>
__global__ void flash_splitk_fused_mk_kernel(
    const float* __restrict__ q,
    const __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __hip_bfloat16* __restrict__ outp,
    const int* __restrict__ pos,
    int delta,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    constexpr int kFlashWarps = kFlashThreads / WAVE;
    constexpr int kMaxAccMk = kMaxHDmk / WAVE;
    constexpr bool kHasFast = kMaxAccMk >= 8;

    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    const int total = pos[0] - delta;
    int total_q[M];
    int start_q[M];
    #pragma unroll
    for (int qi = 0; qi < M; ++qi) {
        total_q[qi] = total - (M - 1) + qi;
        start_q[qi] = (WINDOW > 0 && total_q[qi] > WINDOW) ? (total_q[qi] - WINDOW) : 0;
    }
    const int start = start_q[0];
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (WAVE - 1);
    const int warp = threadIdx.x / WAVE;

    __shared__ float qsh[M * kMaxHDmk];
    for (int t = threadIdx.x; t < M * HD; t += kFlashThreads) {
        int qi = t / HD;
        int d = t - qi * HD;
        qsh[t] = q[((size_t)qi * NH + h) * HD + d];
    }
    __syncthreads();

    float acc[M][kMaxAccMk];
    float m[M];
    float l[M];
    #pragma unroll
    for (int qi = 0; qi < M; ++qi) {
        #pragma unroll
        for (int i = 0; i < kMaxAccMk; ++i) acc[qi][i] = 0.0f;
        m[qi] = -INFINITY;
        l[qi] = 0.0f;
    }

    const bool vec8 = (HD & 7) == 0;
    const int vc = HD / WAVE;
    const bool vecv = (HD % (WAVE * 8)) == 0 && vc <= kMaxAccMk;
    const int lane_stride = SPLITS * kFlashWarps;

    if (kHasFast && vec8 && (HD >> 3) == WAVE && vecv) {
        if constexpr (kHasFast) {
            int p = start + split * kFlashWarps + warp;
            uint4 kraw, vraw;
            if (p < total) {
                kraw = nv_ldg(reinterpret_cast<const uint4*>(k + ((size_t)p * NKV + kvh) * HD) + lane);
                vraw = nv_ldg(reinterpret_cast<const uint4*>(v + ((size_t)p * NKV + kvh) * HD) + lane);
            }
            for (; p < total; p += lane_stride) {
                const int pn = p + lane_stride;
                uint4 kn, vn;
                if (pn < total) {
                    kn = nv_ldg(reinterpret_cast<const uint4*>(k + ((size_t)pn * NKV + kvh) * HD) + lane);
                    vn = nv_ldg(reinterpret_cast<const uint4*>(v + ((size_t)pn * NKV + kvh) * HD) + lane);
                }
                const __hip_bfloat162* kb = reinterpret_cast<const __hip_bfloat162*>(&kraw);
                const __hip_bfloat162* vb = reinterpret_cast<const __hip_bfloat162*>(&vraw);
                float kf[8], vf[8];
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    float2 kf2 = __bfloat1622float2(kb[t]);
                    kf[2 * t] = kf2.x;
                    kf[2 * t + 1] = kf2.y;
                    float2 vf2 = __bfloat1622float2(vb[t]);
                    vf[2 * t] = vf2.x;
                    vf[2 * t + 1] = vf2.y;
                }
                #pragma unroll
                for (int qi = 0; qi < M; ++qi) {
                    if (p < start_q[qi] || p >= total_q[qi]) continue;
                    const float* qp = qsh + qi * HD + lane * 8;
                    float partial = 0.0f;
                    #pragma unroll
                    for (int t = 0; t < 4; ++t)
                        partial += kf[2 * t] * qp[2 * t] + kf[2 * t + 1] * qp[2 * t + 1];
                    float score = nv_hip::wave_sum<WAVE>(partial);
                    float m_new = fmaxf(m[qi], score);
                    float corr = __expf(m[qi] - m_new);
                    float w = __expf(score - m_new);
                    l[qi] = l[qi] * corr + w;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        acc[qi][2 * u] = __fmaf_rn(w, vf[2 * u], corr * acc[qi][2 * u]);
                        acc[qi][2 * u + 1] = __fmaf_rn(w, vf[2 * u + 1], corr * acc[qi][2 * u + 1]);
                    }
                    m[qi] = m_new;
                }
                if (pn < total) {
                    kraw = kn;
                    vraw = vn;
                }
            }
        }
    } else
    for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
        const __hip_bfloat16* kp = k + ((size_t)p * NKV + kvh) * HD;
        const __hip_bfloat16* vp = v + ((size_t)p * NKV + kvh) * HD;
        const int n8 = HD >> 3;
        float kf[8];
        float vf[kMaxAccMk];
        if (vec8) {
            if (lane < n8) {
                uint4 raw = nv_ldg(reinterpret_cast<const uint4*>(kp) + lane);
                const __hip_bfloat162* kb = reinterpret_cast<const __hip_bfloat162*>(&raw);
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    float2 f = __bfloat1622float2(kb[t]);
                    kf[2 * t] = f.x;
                    kf[2 * t + 1] = f.y;
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kMaxAccMk; ++i) {
                int d = lane + i * WAVE;
                if (d < HD) kf[i] = __bfloat162float(kp[d]);
            }
        }
        #pragma unroll
        for (int i = 0; i < kMaxAccMk; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) vf[i] = __bfloat162float(nv_ldg(&vp[d]));
        }
        #pragma unroll
        for (int qi = 0; qi < M; ++qi) {
            if (p < start_q[qi] || p >= total_q[qi]) continue;
            float partial = 0.0f;
            if (vec8) {
                if (lane < n8) {
                    const float* qp = qsh + qi * HD + lane * 8;
                    #pragma unroll
                    for (int t = 0; t < 4; ++t)
                        partial += kf[2 * t] * qp[2 * t] + kf[2 * t + 1] * qp[2 * t + 1];
                }
            } else {
                #pragma unroll
                for (int i = 0; i < kMaxAccMk; ++i) {
                    int d = lane + i * WAVE;
                    if (d < HD) partial += qsh[qi * HD + d] * kf[i];
                }
            }
            float score = nv_hip::wave_sum<WAVE>(partial);
            float m_new = fmaxf(m[qi], score);
            float corr = __expf(m[qi] - m_new);
            float w = __expf(score - m_new);
            l[qi] = l[qi] * corr + w;
            #pragma unroll
            for (int i = 0; i < kMaxAccMk; ++i) {
                int d = lane + i * WAVE;
                if (d < HD) acc[qi][i] = __fmaf_rn(w, vf[i], corr * acc[qi][i]);
            }
            m[qi] = m_new;
        }
    }

    __shared__ float sm[kFlashWarps];
    __shared__ float sl[kFlashWarps];
    __shared__ float sacc[kFlashWarps][kMaxHDmk];
    for (int qi = 0; qi < M; ++qi) {
        __syncthreads();
        if (lane == 0) {
            sm[warp] = m[qi];
            sl[warp] = l[qi];
        }
        if (vecv) {
            #pragma unroll
            for (int i = 0; i < kMaxAccMk; ++i) {
                if (i >= vc) break;
                sacc[warp][lane * vc + i] = acc[qi][i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kMaxAccMk; ++i) {
                int d = lane + i * WAVE;
                if (d < HD) sacc[warp][d] = acc[qi][i];
            }
        }
        __syncthreads();

        float* out = scratch + (((size_t)h * M + qi) * SPLITS + split) * (HD + 2);
        if (warp == 0) {
            float m_blk = -INFINITY;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
            float l_blk = 0.0f;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w)
                l_blk += (sm[w] > -INFINITY) ? sl[w] * __expf(sm[w] - m_blk) : 0.0f;
            if (lane == 0) {
                out[0] = m_blk;
                out[1] = l_blk;
            }
        }
        __syncthreads();
        float m_blk = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
        for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w)
                a += (sm[w] > -INFINITY) ? sacc[w][d] * __expf(sm[w] - m_blk) : 0.0f;
            out[2 + d] = a;
        }
    }

    __syncthreads();
    __threadfence();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) {
        ticket = __hip_atomic_fetch_add(&fan_in[h], 1u, __ATOMIC_ACQ_REL,
                                        __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (ticket != SPLITS - 1) return;
    __threadfence();

    __shared__ float ssc[32];
    __shared__ float sinv_l;
    for (int qi = 0; qi < M; ++qi) {
        const float* base = scratch + ((size_t)h * M + qi) * SPLITS * (HD + 2);
        if (threadIdx.x == 0) {
            float m_glob = -INFINITY;
            #pragma unroll
            for (int s = 0; s < SPLITS; ++s)
                m_glob = fmaxf(m_glob, base[(size_t)s * (HD + 2)]);
            float l_glob = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLITS; ++s) {
                const float* part = base + (size_t)s * (HD + 2);
                float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
                ssc[s] = sc;
                l_glob += part[1] * sc;
            }
            sinv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
        }
        __syncthreads();
        float inv_l = sinv_l;
        for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
            float a = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLITS; ++s)
                a += base[(size_t)s * (HD + 2) + 2 + d] * ssc[s];
            outp[((size_t)qi * NH + h) * HD + d] = __float2bfloat16(a * inv_l);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        __hip_atomic_store(&fan_in[h], 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

template <int SPLITS, int M, int WAVE>
__global__ void flash_splitk_fused_fp8_mk_kernel(
    const __hip_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __hip_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int delta,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling
) {
    constexpr int kFlashWarps = kFlashThreads / WAVE;
    constexpr int kMaxAccMk = kMaxHDmk / WAVE;
    constexpr int kChunks4 = kMaxHDmk / (4 * WAVE);

    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    const int total = n_total_dev[0] - delta;
    int total_q[M];
    int start_q[M];
    #pragma unroll
    for (int qi = 0; qi < M; ++qi) {
        total_q[qi] = total - (M - 1) + qi;
        start_q[qi] = (WINDOW > 0 && total_q[qi] > WINDOW) ? (total_q[qi] - WINDOW) : 0;
    }
    const int start = start_q[0];
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (WAVE - 1);
    const int warp = threadIdx.x / WAVE;

    __shared__ float qsh[M * kMaxHDmk];
    for (int t = threadIdx.x; t < M * HD; t += kFlashThreads) {
        int qi = t / HD;
        int d = t - qi * HD;
        qsh[t] = __bfloat162float(q[((size_t)qi * NH + h) * HD + d]);
    }
    __syncthreads();

    float acc[M][kMaxAccMk];
    float m[M];
    float l[M];
    #pragma unroll
    for (int qi = 0; qi < M; ++qi) {
        #pragma unroll
        for (int i = 0; i < kMaxAccMk; ++i) acc[qi][i] = 0.0f;
        m[qi] = -INFINITY;
        l[qi] = 0.0f;
    }

    const bool vec4 = (HD & 3) == 0;
    const int n4 = HD >> 2;
    const int lane_stride = SPLITS * kFlashWarps;

    for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
        const int sp = (RING > 0) ? (p % RING) : p;
        const uint8_t* kp = k_fp8 + ((size_t)sp * NKV + kvh) * HD;
        const float ks = k_scales[(size_t)sp * NKV + kvh];
        const uint8_t* vp = v_fp8 + ((size_t)sp * NKV + kvh) * HD;
        const float vs = v_scales[(size_t)sp * NKV + kvh];

        float kf[4 * kChunks4];
        if (vec4) {
            const uchar4* k4 = reinterpret_cast<const uchar4*>(kp);
            #pragma unroll
            for (int c = 0; c < kChunks4; ++c) {
                int j = lane + c * WAVE;
                if (j < n4) {
                    uchar4 raw = nv_ldg(&k4[j]);
                    kf[4 * c] = e4m3_ocp_to_float(raw.x);
                    kf[4 * c + 1] = e4m3_ocp_to_float(raw.y);
                    kf[4 * c + 2] = e4m3_ocp_to_float(raw.z);
                    kf[4 * c + 3] = e4m3_ocp_to_float(raw.w);
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kMaxAccMk; ++i) {
                int d = lane + i * WAVE;
                if (d < HD) kf[i] = e4m3_ocp_to_float(kp[d]);
            }
        }
        float vf[kMaxAccMk];
        #pragma unroll
        for (int i = 0; i < kMaxAccMk; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) vf[i] = e4m3_ocp_to_float(nv_ldg(&vp[d]));
        }

        #pragma unroll
        for (int qi = 0; qi < M; ++qi) {
            if (p < start_q[qi] || p >= total_q[qi]) continue;
            float partial = 0.0f;
            if (vec4) {
                #pragma unroll
                for (int c = 0; c < kChunks4; ++c) {
                    int j = lane + c * WAVE;
                    if (j < n4) {
                        const float* qp = qsh + qi * HD + j * 4;
                        partial += qp[0] * kf[4 * c]
                                 + qp[1] * kf[4 * c + 1]
                                 + qp[2] * kf[4 * c + 2]
                                 + qp[3] * kf[4 * c + 3];
                    }
                }
            } else {
                #pragma unroll
                for (int i = 0; i < kMaxAccMk; ++i) {
                    int d = lane + i * WAVE;
                    if (d < HD) partial += qsh[qi * HD + d] * kf[i];
                }
            }
            float score = nv_hip::wave_sum<WAVE>(partial) * ks * scaling;
            float m_new = fmaxf(m[qi], score);
            float corr = __expf(m[qi] - m_new);
            float w = __expf(score - m_new);
            l[qi] = l[qi] * corr + w;
            const float w_v = w * vs;
            #pragma unroll
            for (int i = 0; i < kMaxAccMk; ++i) {
                int d = lane + i * WAVE;
                if (d < HD) acc[qi][i] = acc[qi][i] * corr + w_v * vf[i];
            }
            m[qi] = m_new;
        }
    }

    __shared__ float sm[kFlashWarps];
    __shared__ float sl[kFlashWarps];
    __shared__ float sacc[kFlashWarps][kMaxHDmk];
    for (int qi = 0; qi < M; ++qi) {
        __syncthreads();
        if (lane == 0) {
            sm[warp] = m[qi];
            sl[warp] = l[qi];
        }
        #pragma unroll
        for (int i = 0; i < kMaxAccMk; ++i) {
            int d = lane + i * WAVE;
            if (d < HD) sacc[warp][d] = acc[qi][i];
        }
        __syncthreads();

        float* out = scratch + (((size_t)h * M + qi) * SPLITS + split) * (HD + 2);
        if (warp == 0) {
            float m_blk = -INFINITY;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
            float l_blk = 0.0f;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w)
                l_blk += (sm[w] > -INFINITY) ? sl[w] * __expf(sm[w] - m_blk) : 0.0f;
            if (lane == 0) {
                out[0] = m_blk;
                out[1] = l_blk;
            }
        }
        __syncthreads();
        float m_blk = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
        for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w)
                a += (sm[w] > -INFINITY) ? sacc[w][d] * __expf(sm[w] - m_blk) : 0.0f;
            out[2 + d] = a;
        }
    }

    __syncthreads();
    __threadfence();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) {
        ticket = __hip_atomic_fetch_add(&fan_in[h], 1u, __ATOMIC_ACQ_REL,
                                        __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    if (ticket != SPLITS - 1) return;
    __threadfence();

    __shared__ float ssc[32];
    __shared__ float sinv_l;
    for (int qi = 0; qi < M; ++qi) {
        const float* base = scratch + ((size_t)h * M + qi) * SPLITS * (HD + 2);
        if (threadIdx.x == 0) {
            float m_glob = -INFINITY;
            #pragma unroll
            for (int s = 0; s < SPLITS; ++s)
                m_glob = fmaxf(m_glob, base[(size_t)s * (HD + 2)]);
            float l_glob = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLITS; ++s) {
                const float* part = base + (size_t)s * (HD + 2);
                float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
                ssc[s] = sc;
                l_glob += part[1] * sc;
            }
            sinv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
        }
        __syncthreads();
        float inv_l = sinv_l;
        for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
            float a = 0.0f;
            #pragma unroll
            for (int s = 0; s < SPLITS; ++s)
                a += base[(size_t)s * (HD + 2) + 2 + d] * ssc[s];
            outp[((size_t)qi * NH + h) * HD + d] = __float2bfloat16(a * inv_l);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        __hip_atomic_store(&fan_in[h], 0u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

}

extern "C" int nv_kernels_flash_splitk_scratch_elems_mk(int NH, int HD, int M) {
    return NH * M * splitk::flash_splits_env() * (HD + 2);
}

extern "C" int nv_kernels_flash_decode_fused_bf16kv_mk(
    void* stream,
    const float* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* pos,
    int delta,
    int M,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (M < 1 || M > splitk_mk::kMaxM) return -1;
    if (HD > splitk_mk::kMaxHDmk || (NH % NKV) != 0) return -1;
    int sp = splitk::flash_splits_env();
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __hip_bfloat16* kb = reinterpret_cast<const __hip_bfloat16*>(k);
    const __hip_bfloat16* vb = reinterpret_cast<const __hip_bfloat16*>(v);
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(out);
    hipStream_t cs = (hipStream_t)stream;
    const int wave = wave_size_now();
#define NV_MK_BF16_CASE(SP, MM, W) \
    case MM: \
        splitk_mk::flash_splitk_fused_mk_kernel<SP, MM, W> \
            <<<grid, kFlashThreads, 0, cs>>>( \
                q, kb, vb, scratch, fan_in, ob, pos, delta, NH, NKV, HD, WINDOW); \
        break;
#define NV_MK_BF16_SWITCH(SP, W) \
    switch (M) { \
        NV_MK_BF16_CASE(SP, 1, W) \
        NV_MK_BF16_CASE(SP, 2, W) \
        NV_MK_BF16_CASE(SP, 3, W) \
        NV_MK_BF16_CASE(SP, 4, W) \
        NV_MK_BF16_CASE(SP, 5, W) \
        NV_MK_BF16_CASE(SP, 6, W) \
        NV_MK_BF16_CASE(SP, 7, W) \
        NV_MK_BF16_CASE(SP, 8, W) \
    } \
    break;
    if (wave == 64) {
        switch (sp) {
            case 8: NV_MK_BF16_SWITCH(8, 64)
            case 32: NV_MK_BF16_SWITCH(32, 64)
            default: NV_MK_BF16_SWITCH(16, 64)
        }
    } else {
        switch (sp) {
            case 8: NV_MK_BF16_SWITCH(8, 32)
            case 32: NV_MK_BF16_SWITCH(32, 32)
            default: NV_MK_BF16_SWITCH(16, 32)
        }
    }
#undef NV_MK_BF16_SWITCH
#undef NV_MK_BF16_CASE
    return (int)hipGetLastError();
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv_mk(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    uint16_t* out,
    const int* n_total_dev,
    int delta,
    int M,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (M < 1 || M > splitk_mk::kMaxM) return -1;
    if (HD > splitk_mk::kMaxHDmk || (NH % NKV) != 0) return -1;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    int sp = splitk::flash_splits_env();
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __hip_bfloat16* qb = reinterpret_cast<const __hip_bfloat16*>(q);
    __hip_bfloat16* ob = reinterpret_cast<__hip_bfloat16*>(out);
    hipStream_t cs = (hipStream_t)stream;
    const int wave = wave_size_now();
#define NV_MK_FP8_CASE(SP, MM, W) \
    case MM: \
        splitk_mk::flash_splitk_fused_fp8_mk_kernel<SP, MM, W> \
            <<<grid, kFlashThreads, 0, cs>>>( \
                qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob, \
                n_total_dev, delta, NH, NKV, HD, WINDOW, RING, scaling); \
        break;
#define NV_MK_FP8_SWITCH(SP, W) \
    switch (M) { \
        NV_MK_FP8_CASE(SP, 1, W) \
        NV_MK_FP8_CASE(SP, 2, W) \
        NV_MK_FP8_CASE(SP, 3, W) \
        NV_MK_FP8_CASE(SP, 4, W) \
        NV_MK_FP8_CASE(SP, 5, W) \
        NV_MK_FP8_CASE(SP, 6, W) \
        NV_MK_FP8_CASE(SP, 7, W) \
        NV_MK_FP8_CASE(SP, 8, W) \
    } \
    break;
    if (wave == 64) {
        switch (sp) {
            case 8: NV_MK_FP8_SWITCH(8, 64)
            case 32: NV_MK_FP8_SWITCH(32, 64)
            default: NV_MK_FP8_SWITCH(16, 64)
        }
    } else {
        switch (sp) {
            case 8: NV_MK_FP8_SWITCH(8, 32)
            case 32: NV_MK_FP8_SWITCH(32, 32)
            default: NV_MK_FP8_SWITCH(16, 32)
        }
    }
#undef NV_MK_FP8_SWITCH
#undef NV_MK_FP8_CASE
    return (int)hipGetLastError();
}

__global__ void write_kv_bf16_kernel(
    const float* __restrict__ src_k,
    const float* __restrict__ src_v,
    __hip_bfloat16* __restrict__ cache_k,
    __hip_bfloat16* __restrict__ cache_v,
    const int* __restrict__ pos,
    int NKV,
    int HD
) {
    int kvh = blockIdx.x;
    if (kvh >= NKV) return;
    int slot = pos[0] - 1;
    if (slot < 0) return;
    size_t dst = ((size_t)slot * NKV + kvh) * HD;
    size_t src = (size_t)kvh * HD;
    for (int d = threadIdx.x; d < HD; d += blockDim.x) {
        cache_k[dst + d] = __float2bfloat16(src_k[src + d]);
        cache_v[dst + d] = __float2bfloat16(src_v[src + d]);
    }
}

extern "C" int nv_kernels_write_kv_bf16(
    void* stream,
    const float* src_k,
    const float* src_v,
    uint16_t* cache_k,
    uint16_t* cache_v,
    const int* pos,
    int NKV,
    int HD
) {
    if (NKV <= 0 || HD <= 0) return 0;
    write_kv_bf16_kernel<<<(unsigned)NKV, 128, 0, (hipStream_t)stream>>>(
        src_k, src_v,
        reinterpret_cast<__hip_bfloat16*>(cache_k),
        reinterpret_cast<__hip_bfloat16*>(cache_v),
        pos, NKV, HD);
    return (int)hipGetLastError();
}
