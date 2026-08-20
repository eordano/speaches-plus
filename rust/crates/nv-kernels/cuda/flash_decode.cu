
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "nvk_pdl.cuh"
#include <cuda_pipeline_primitives.h>

__device__ __forceinline__ int kv_phys_slot(
    const int* __restrict__ block_table, int block_size, int RING, int p
) {
    if (block_table != nullptr) {
        int blk = p / block_size;
        return block_table[blk] * block_size + (p - blk * block_size);
    }
    return (RING > 0) ? (p % RING) : p;
}

namespace {

constexpr int kWarp = 32;
constexpr int kFlashWarps = 8;
constexpr int kFlashThreads = kWarp * kFlashWarps;
constexpr int kMaxHD = 512;
constexpr int kMaxAccPerLane = kMaxHD / kWarp;

__inline__ __device__ float warp_sum(float x) {
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

template <typename OutT>
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
    const int h = blockIdx.x;
    if (h >= NH) return;

    const int total = pos[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

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
            for (int j = lane; j < n4; j += kWarp) {
                float4 a = q4[j];
                float4 b = k4[j];
                partial += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
        } else {
            for (int d = lane; d < HD; d += kWarp)
                partial += qsh[d] * kp[d];
        }
        float score = warp_sum(partial);

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;

        const float* vp = v + ((size_t)p * NKV + kvh) * HD;
        #pragma unroll
        for (int i = 0; i < kMaxAccPerLane; ++i) {
            int d = lane + i * kWarp;
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
        int d = lane + i * kWarp;
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
            l_glob += sl[w] * __expf(sm[w] - m_glob);
        float inv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;

        float scale[kFlashWarps];
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w) scale[w] = __expf(sm[w] - m_glob);

        for (int d = lane; d < HD; d += kWarp) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w) a += sacc[w][d] * scale[w];
            out[(size_t)h * HD + d] = static_cast<OutT>(a * inv_l);
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
    flash_decode_kernel<float><<<(unsigned)NH, kFlashThreads, 0, (cudaStream_t)stream>>>(
        q, k, v, out, pos, NH, NKV, HD, WINDOW
    );
    return (int)cudaGetLastError();
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
    flash_decode_kernel<__nv_bfloat16><<<(unsigned)NH, kFlashThreads, 0, (cudaStream_t)stream>>>(
        q, k, v, reinterpret_cast<__nv_bfloat16*>(out), pos, NH, NKV, HD, WINDOW
    );
    return (int)cudaGetLastError();
}

__inline__ __device__ float2 nv_fp8x2_to_float2(unsigned short packed) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed, __NV_E4M3);
    __half2 h2 = *reinterpret_cast<__half2*>(&hr);
    return __half22float2(h2);
}

namespace splitk {

constexpr int kWarp = 32;
constexpr int kFlashWarps = 8;
constexpr int kFlashThreads = kWarp * kFlashWarps;
constexpr int kMaxHD = 512;
constexpr int kMaxAccPerLane = kMaxHD / kWarp;
constexpr int kSplits = 16;

__inline__ __device__ float warp_sum(float x) {
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

template <int SPLITS>
__global__ void flash_splitk_stage1_kernel(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ scratch,
    const int* __restrict__ pos,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    const int total = pos[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

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
        const __nv_bfloat16* kp = k + ((size_t)p * NKV + kvh) * HD;
        float partial = 0.0f;
        if (vec8) {
            const uint4* k8 = reinterpret_cast<const uint4*>(kp);
            const int n8 = HD >> 3;
            for (int j = lane; j < n8; j += kWarp) {
                uint4 raw = __ldg(&k8[j]);
                const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&raw);
                const float* qp = qsh + j * 8;
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    float2 kf = __bfloat1622float2(kb[t]);
                    partial += kf.x * qp[2 * t] + kf.y * qp[2 * t + 1];
                }
            }
        } else {
            for (int d = lane; d < HD; d += kWarp)
                partial += qsh[d] * __bfloat162float(kp[d]);
        }
        float score = warp_sum(partial);

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;

        const __nv_bfloat16* vp = v + ((size_t)p * NKV + kvh) * HD;
        #pragma unroll
        for (int i = 0; i < kMaxAccPerLane; ++i) {
            int d = lane + i * kWarp;
            if (d < HD) acc[i] = acc[i] * corr + w * __bfloat162float(__ldg(&vp[d]));
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
        int d = lane + i * kWarp;
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
    __nv_bfloat16* __restrict__ out,
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

template <int SPLITS>
__global__ void flash_splitk_fused_kernel(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ pos,
    int delta,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    const int total = pos[0] - delta;
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

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
    const int vc = HD / kWarp;
    const bool vecv = (HD % (kWarp * 8)) == 0 && vc <= kMaxAccPerLane;
    const int lane_stride = SPLITS * kFlashWarps;

    if (vec8 && (HD >> 3) == kWarp && vecv) {
        int p = start + split * kFlashWarps + warp;
        uint4 kraw, vraw;
        if (p < total) {
            kraw = __ldg(reinterpret_cast<const uint4*>(k + ((size_t)p * NKV + kvh) * HD) + lane);
            vraw = __ldg(reinterpret_cast<const uint4*>(v + ((size_t)p * NKV + kvh) * HD) + lane);
        }
        for (; p < total; p += lane_stride) {
            const int pn = p + lane_stride;
            uint4 kn, vn;
            if (pn < total) {
                kn = __ldg(reinterpret_cast<const uint4*>(k + ((size_t)pn * NKV + kvh) * HD) + lane);
                vn = __ldg(reinterpret_cast<const uint4*>(v + ((size_t)pn * NKV + kvh) * HD) + lane);
            }
            const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&kraw);
            const float* qp = qsh + lane * 8;
            float partial = 0.0f;
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                float2 kf = __bfloat1622float2(kb[t]);
                partial += kf.x * qp[2 * t] + kf.y * qp[2 * t + 1];
            }
            float score = warp_sum(partial);
            float m_new = fmaxf(m, score);
            float corr = __expf(m - m_new);
            float w = __expf(score - m_new);
            l = l * corr + w;
            const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&vraw);
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
        const __nv_bfloat16* kp = k + ((size_t)p * NKV + kvh) * HD;
        float partial = 0.0f;
        if (vec8) {
            const uint4* k8 = reinterpret_cast<const uint4*>(kp);
            const int n8 = HD >> 3;
            for (int j = lane; j < n8; j += kWarp) {
                uint4 raw = __ldg(&k8[j]);
                const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&raw);
                const float* qp = qsh + j * 8;
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    float2 kf = __bfloat1622float2(kb[t]);
                    partial += kf.x * qp[2 * t] + kf.y * qp[2 * t + 1];
                }
            }
        } else {
            for (int d = lane; d < HD; d += kWarp)
                partial += qsh[d] * __bfloat162float(kp[d]);
        }
        float score = warp_sum(partial);

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;

        const __nv_bfloat16* vp = v + ((size_t)p * NKV + kvh) * HD;
        if (vecv) {
            const uint4* v8 = reinterpret_cast<const uint4*>(vp + lane * vc);
            #pragma unroll
            for (int t = 0; t < kMaxAccPerLane / 8; ++t) {
                if (t >= vc / 8) break;
                uint4 raw = __ldg(&v8[t]);
                const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&raw);
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
                int d = lane + i * kWarp;
                if (d < HD) acc[i] = acc[i] * corr + w * __bfloat162float(__ldg(&vp[d]));
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
            int d = lane + i * kWarp;
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
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[h], 1u);
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
    if (threadIdx.x == 0) fan_in[h] = 0u;
}

template <int SPLITS, int HD>
__global__ void flash_splitk_fused_fp8_gqa_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int NH,
    int NKV,
    int WINDOW,
    int RING,
    float scaling,
    const int* __restrict__ block_table,
    int block_size
) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    if (kvh >= NKV) return;

    const int group = NH / NKV;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int h = kvh * group + warp;

    const int total = n_total_dev[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;

    __shared__ float qsh[kFlashWarps][kMaxHD];
    for (int d = lane; d < HD; d += kWarp)
        qsh[warp][d] = __bfloat162float(q[(size_t)h * HD + d]);

    __shared__ uint8_t ksh[kMaxHD];
    __shared__ uint8_t vsh[kMaxHD];
    __shared__ float kvs[2];
    __syncthreads();

    constexpr int kPerLane = HD / kWarp;
    float acc[kPerLane];
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    for (int p = start + split; p < total; p += SPLITS) {
        const int sp = kv_phys_slot(block_table, block_size, RING, p);
        const uint8_t* kp = k_fp8 + ((size_t)sp * NKV + kvh) * HD;
        const uint8_t* vp = v_fp8 + ((size_t)sp * NKV + kvh) * HD;
        for (int i = threadIdx.x; i < HD; i += kFlashThreads) {
            ksh[i] = __ldg(&kp[i]);
            vsh[i] = __ldg(&vp[i]);
        }
        if (threadIdx.x == 0) {
            kvs[0] = k_scales[(size_t)sp * NKV + kvh];
            kvs[1] = v_scales[(size_t)sp * NKV + kvh];
        }
        __syncthreads();

        float partial = 0.0f;
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int d = lane + i * kWarp;
            __nv_fp8_e4m3 e;
            e.__x = ksh[d];
            partial += qsh[warp][d] * static_cast<float>(e);
        }
        const float score = warp_sum(partial) * kvs[0] * scaling;
        const float m_new = fmaxf(m, score);
        const float corr = __expf(m - m_new);
        const float w = __expf(score - m_new);
        l = l * corr + w;
        const float w_v = w * kvs[1];
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) {
            const int d = lane + i * kWarp;
            __nv_fp8_e4m3 e;
            e.__x = vsh[d];
            acc[i] = __fmaf_rn(w_v, static_cast<float>(e), __fmul_rn(acc[i], corr));
        }
        m = m_new;
        __syncthreads();
    }

    float* out = scratch + ((size_t)h * SPLITS + split) * (HD + 2);
    if (lane == 0) {
        out[0] = m;
        out[1] = l;
    }
    #pragma unroll
    for (int i = 0; i < kPerLane; ++i) out[2 + lane + i * kWarp] = acc[i];

    __threadfence();
    __syncthreads();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[kvh], 1u);
    __syncthreads();
    if (ticket != SPLITS - 1) return;
    __threadfence();

    for (int g = 0; g < group; ++g) {
        const int hh = kvh * group + g;
        const float* base = scratch + (size_t)hh * SPLITS * (HD + 2);
        __shared__ float ssc[128];
        __shared__ float sinv_l;
        if (threadIdx.x == 0) {
            float m_glob = -INFINITY;
            for (int s = 0; s < SPLITS; ++s)
                m_glob = fmaxf(m_glob, base[(size_t)s * (HD + 2)]);
            float l_glob = 0.0f;
            for (int s = 0; s < SPLITS; ++s) {
                const float* part = base + (size_t)s * (HD + 2);
                float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
                ssc[s] = sc;
                l_glob += part[1] * sc;
            }
            sinv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
        }
        __syncthreads();
        const float inv_l = sinv_l;
        for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
            float a = 0.0f;
            for (int s = 0; s < SPLITS; ++s)
                a += base[(size_t)s * (HD + 2) + 2 + d] * ssc[s];
            outp[(size_t)hh * HD + d] = __float2bfloat16(a * inv_l);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) fan_in[kvh] = 0u;
}

template <int SPLITS, int HD>
__global__ void flash_splitk_fused_fp8_derivev_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ inv_freq,
    const float* __restrict__ cos_pk,
    const float* __restrict__ sin_pk,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int NH,
    int NKV,
    int WINDOW,
    int RING,
    int rope_angles,
    float w_inv,
    float scaling,
    const int* __restrict__ block_table,
    int block_size
) {
    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    NVK_PDL_PROLOG();

    const int total = n_total_dev[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

    __shared__ float qsh[kMaxHD];
    for (int d = threadIdx.x; d < HD; d += kFlashThreads)
        qsh[d] = __bfloat162float(q[(size_t)h * HD + d]);
    __syncthreads();

    float acc[kMaxAccPerLane];
    #pragma unroll
    for (int i = 0; i < kMaxAccPerLane; ++i) acc[i] = 0.0f;
    float m = -INFINITY;
    float l = 0.0f;

    constexpr int nck = HD >> 7;
    constexpr int nlo = nck >> 1;
    const int lane_stride = SPLITS * kFlashWarps;

    for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
        const int sp = kv_phys_slot(block_table, block_size, RING, p);
        const float ks = k_scales[(size_t)sp * NKV + kvh];
        uchar4 kcur[nck];
        const uchar4* k4 = reinterpret_cast<const uchar4*>(
            k_fp8 + ((size_t)sp * NKV + kvh) * HD);
        #pragma unroll
        for (int c = 0; c < nck; ++c) kcur[c] = __ldg(&k4[lane + c * kWarp]);

        float kf[nck * 4];
        #pragma unroll
        for (int c = 0; c < nck; ++c) {
            uchar4 raw = kcur[c];
            float2 f01 = nv_fp8x2_to_float2(
                (unsigned short)(raw.x | ((unsigned short)raw.y << 8)));
            float2 f23 = nv_fp8x2_to_float2(
                (unsigned short)(raw.z | ((unsigned short)raw.w << 8)));
            kf[c * 4 + 0] = f01.x;
            kf[c * 4 + 1] = f01.y;
            kf[c * 4 + 2] = f23.x;
            kf[c * 4 + 3] = f23.y;
        }

        float partial = 0.0f;
        #pragma unroll
        for (int c = 0; c < nck; ++c) {
            const float* qp = qsh + (lane + c * kWarp) * 4;
            partial += qp[0] * kf[c * 4 + 0]
                     + qp[1] * kf[c * 4 + 1]
                     + qp[2] * kf[c * 4 + 2]
                     + qp[3] * kf[c * 4 + 3];
        }
        float score = warp_sum(partial) * ks * scaling;
        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;
        const float wq = w * ks * w_inv;

        float ca0[4], sa0[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            ca0[i] = 1.0f;
            sa0[i] = 0.0f;
        }
        if (lane * 4 < rope_angles) {
            if (cos_pk != nullptr) {
                const float4 c4 = __ldg(
                    reinterpret_cast<const float4*>(cos_pk + (size_t)p * rope_angles) + lane);
                const float4 s4 = __ldg(
                    reinterpret_cast<const float4*>(sin_pk + (size_t)p * rope_angles) + lane);
                ca0[0] = c4.x; ca0[1] = c4.y; ca0[2] = c4.z; ca0[3] = c4.w;
                sa0[0] = s4.x; sa0[1] = s4.y; sa0[2] = s4.z; sa0[3] = s4.w;
            } else {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int j = lane * 4 + i;
                    if (j < rope_angles) {
                        double th = (double)p * (double)inv_freq[j];
                        th -= 6.283185307179586 * floor(th * 0.15915494309189535);
                        __sincosf((float)th, &sa0[i], &ca0[i]);
                    }
                }
            }
        }

        #pragma unroll
        for (int c = 0; c < nlo; ++c) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float ca = (c == 0) ? ca0[i] : 1.0f;
                const float sa = (c == 0) ? sa0[i] : 0.0f;
                const float klo = kf[c * 4 + i];
                const float khi = kf[(c + nlo) * 4 + i];
                const int alo = c * 4 + i;
                const int ahi = (c + nlo) * 4 + i;
                acc[alo] = __fmaf_rn(wq, klo * ca + khi * sa, __fmul_rn(acc[alo], corr));
                acc[ahi] = __fmaf_rn(wq, khi * ca - klo * sa, __fmul_rn(acc[ahi], corr));
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
    for (int c = 0; c < nck; ++c) {
        #pragma unroll
        for (int i = 0; i < 4; ++i)
            sacc[warp][c * 128 + lane * 4 + i] = acc[c * 4 + i];
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
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[h], 1u);
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
    if (threadIdx.x == 0) fan_in[h] = 0u;
}

template <int SPLITS>
__global__ void flash_splitk_fused_fp8_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling,
    const int* __restrict__ block_table,
    int block_size
) {
    const int h = blockIdx.x;
    const int split = blockIdx.y;
    if (h >= NH) return;

    NVK_PDL_PROLOG();

    const int total = n_total_dev[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int group = NH / NKV;
    const int kvh = h / group;

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

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
    const int vc = HD / kWarp;
    const bool vecv = (HD % (kWarp * 8)) == 0 && vc <= kMaxAccPerLane;
    const int lane_stride = SPLITS * kFlashWarps;
    const int nck = HD >> 7;
    const int nv2 = vc >> 3;

    if (vec4 && vecv && (HD & 127) == 0) {
        for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
            const int sp = kv_phys_slot(block_table, block_size, RING, p);
            const float ks = k_scales[(size_t)sp * NKV + kvh];
            const float vs = v_scales[(size_t)sp * NKV + kvh];
            uchar4 kcur[kMaxAccPerLane / 4];
            uint2 vcur[kMaxAccPerLane / 8];
            const uchar4* k4 = reinterpret_cast<const uchar4*>(
                k_fp8 + ((size_t)sp * NKV + kvh) * HD);
            #pragma unroll
            for (int c = 0; c < kMaxAccPerLane / 4; ++c)
                if (c < nck) kcur[c] = __ldg(&k4[lane + c * kWarp]);
            const uint2* v8 = reinterpret_cast<const uint2*>(
                v_fp8 + ((size_t)sp * NKV + kvh) * HD + lane * vc);
            #pragma unroll
            for (int t = 0; t < kMaxAccPerLane / 8; ++t)
                if (t < nv2) vcur[t] = __ldg(&v8[t]);
            float partial = 0.0f;
            #pragma unroll
            for (int c = 0; c < kMaxAccPerLane / 4; ++c) {
                if (c >= nck) break;
                uchar4 raw = kcur[c];
                const float* qp = qsh + (lane + c * kWarp) * 4;
                float2 f01 = nv_fp8x2_to_float2(
                    (unsigned short)(raw.x | ((unsigned short)raw.y << 8)));
                float2 f23 = nv_fp8x2_to_float2(
                    (unsigned short)(raw.z | ((unsigned short)raw.w << 8)));
                partial += qp[0] * f01.x
                         + qp[1] * f01.y
                         + qp[2] * f23.x
                         + qp[3] * f23.y;
            }
            float score = warp_sum(partial) * ks * scaling;
            float m_new = fmaxf(m, score);
            float corr = __expf(m - m_new);
            float w = __expf(score - m_new);
            l = l * corr + w;
            const float w_v = w * vs;
            #pragma unroll
            for (int t = 0; t < kMaxAccPerLane / 8; ++t) {
                if (t >= nv2) break;
                uint2 raw = vcur[t];
                float2 f0 = nv_fp8x2_to_float2((unsigned short)(raw.x & 0xffffu));
                float2 f1 = nv_fp8x2_to_float2((unsigned short)(raw.x >> 16));
                float2 f2 = nv_fp8x2_to_float2((unsigned short)(raw.y & 0xffffu));
                float2 f3 = nv_fp8x2_to_float2((unsigned short)(raw.y >> 16));
                acc[t * 8 + 0] = __fmaf_rn(w_v, f0.x, __fmul_rn(acc[t * 8 + 0], corr));
                acc[t * 8 + 1] = __fmaf_rn(w_v, f0.y, __fmul_rn(acc[t * 8 + 1], corr));
                acc[t * 8 + 2] = __fmaf_rn(w_v, f1.x, __fmul_rn(acc[t * 8 + 2], corr));
                acc[t * 8 + 3] = __fmaf_rn(w_v, f1.y, __fmul_rn(acc[t * 8 + 3], corr));
                acc[t * 8 + 4] = __fmaf_rn(w_v, f2.x, __fmul_rn(acc[t * 8 + 4], corr));
                acc[t * 8 + 5] = __fmaf_rn(w_v, f2.y, __fmul_rn(acc[t * 8 + 5], corr));
                acc[t * 8 + 6] = __fmaf_rn(w_v, f3.x, __fmul_rn(acc[t * 8 + 6], corr));
                acc[t * 8 + 7] = __fmaf_rn(w_v, f3.y, __fmul_rn(acc[t * 8 + 7], corr));
            }
            m = m_new;
        }
    } else
    for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
        const int sp = kv_phys_slot(block_table, block_size, RING, p);
        const uint8_t* kp = k_fp8 + ((size_t)sp * NKV + kvh) * HD;
        const float ks = k_scales[(size_t)sp * NKV + kvh];
        float partial = 0.0f;
        if (vec4) {
            const uchar4* k4 = reinterpret_cast<const uchar4*>(kp);
            const int n4 = HD >> 2;
            for (int j = lane; j < n4; j += kWarp) {
                uchar4 raw = __ldg(&k4[j]);
                const float* qp = qsh + j * 4;
                float2 f01 = nv_fp8x2_to_float2(
                    (unsigned short)(raw.x | ((unsigned short)raw.y << 8)));
                float2 f23 = nv_fp8x2_to_float2(
                    (unsigned short)(raw.z | ((unsigned short)raw.w << 8)));
                partial += qp[0] * f01.x
                         + qp[1] * f01.y
                         + qp[2] * f23.x
                         + qp[3] * f23.y;
            }
        } else {
            for (int d = lane; d < HD; d += kWarp) {
                __nv_fp8_e4m3 e; e.__x = kp[d];
                partial += qsh[d] * static_cast<float>(e);
            }
        }
        float score = warp_sum(partial) * ks * scaling;

        float m_new = fmaxf(m, score);
        float corr = __expf(m - m_new);
        float w = __expf(score - m_new);
        l = l * corr + w;

        const uint8_t* vp = v_fp8 + ((size_t)sp * NKV + kvh) * HD;
        const float w_v = w * v_scales[(size_t)sp * NKV + kvh];
        if (vecv) {
            const uint2* v8 = reinterpret_cast<const uint2*>(vp + lane * vc);
            #pragma unroll
            for (int t = 0; t < kMaxAccPerLane / 8; ++t) {
                if (t >= vc / 8) break;
                uint2 raw = __ldg(&v8[t]);
                float2 f0 = nv_fp8x2_to_float2((unsigned short)(raw.x & 0xffffu));
                float2 f1 = nv_fp8x2_to_float2((unsigned short)(raw.x >> 16));
                float2 f2 = nv_fp8x2_to_float2((unsigned short)(raw.y & 0xffffu));
                float2 f3 = nv_fp8x2_to_float2((unsigned short)(raw.y >> 16));
                acc[t * 8 + 0] = __fmaf_rn(w_v, f0.x, __fmul_rn(acc[t * 8 + 0], corr));
                acc[t * 8 + 1] = __fmaf_rn(w_v, f0.y, __fmul_rn(acc[t * 8 + 1], corr));
                acc[t * 8 + 2] = __fmaf_rn(w_v, f1.x, __fmul_rn(acc[t * 8 + 2], corr));
                acc[t * 8 + 3] = __fmaf_rn(w_v, f1.y, __fmul_rn(acc[t * 8 + 3], corr));
                acc[t * 8 + 4] = __fmaf_rn(w_v, f2.x, __fmul_rn(acc[t * 8 + 4], corr));
                acc[t * 8 + 5] = __fmaf_rn(w_v, f2.y, __fmul_rn(acc[t * 8 + 5], corr));
                acc[t * 8 + 6] = __fmaf_rn(w_v, f3.x, __fmul_rn(acc[t * 8 + 6], corr));
                acc[t * 8 + 7] = __fmaf_rn(w_v, f3.y, __fmul_rn(acc[t * 8 + 7], corr));
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kMaxAccPerLane; ++i) {
                int d = lane + i * kWarp;
                if (d < HD) {
                    acc[i] = __fmaf_rn(
                        nv_fp8x2_to_float2((unsigned short)__ldg(&vp[d])).x, w_v,
                        __fmul_rn(acc[i], corr));
                }
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
            int d = lane + i * kWarp;
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
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[h], 1u);
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
    if (threadIdx.x == 0) fan_in[h] = 0u;

    NVK_PDL_EPILOG();
}

inline int flash_splits_env() {
    static int forced = [] {
        const char* e = getenv("NV_E4B_FLASH_SPLITS");
        if (e == nullptr) return 0;
        int x = atoi(e);
        return (x == 8 || x == 16 || x == 32) ? x : 0;
    }();
    return forced;
}

inline int flash_sm_count() {
    static int sm = [] {
        int dev = 0, n = 148;
        if (cudaGetDevice(&dev) == cudaSuccess)
            cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n > 0 ? n : 148;
    }();
    return sm;
}

inline int flash_splits_pick(int NH) {
    int forced = flash_splits_env();
    if (forced) return forced;
    if (NH <= 0) return (int)kSplits;
    const long target = 2L * flash_sm_count();
    if ((long)NH * 8 >= target) return 8;
    if ((long)NH * 16 >= target) return 16;
    return 32;
}

inline bool q38_attn_gqa_stage_env() {
    static int v = -1;
    if (v < 0) {
        const char* e = getenv("NV_Q38_ATTN_GQA_STAGE");
        v = (e && e[0] == '1') ? 1 : 0;
    }
    return v == 1;
}

constexpr int kKvShareGroup = 6;
constexpr int kKvShareHD = 256;
constexpr int kKvShareMaxSplits = 128;

inline int kvshare_splits_pick(int NKV) {
    int forced = flash_splits_env();
    if (forced) return forced;
    if (NKV <= 0) return (int)kSplits;
    int one_full_wave_no_tail = (2 * flash_sm_count()) / NKV;
    if (one_full_wave_no_tail < 8) one_full_wave_no_tail = 8;
    if (one_full_wave_no_tail > kKvShareMaxSplits)
        one_full_wave_no_tail = kKvShareMaxSplits;
    return one_full_wave_no_tail;
}

inline int kvshare_env_int_reread_each_launch_so_one_build_sweeps(
    const char* name, int dflt, int lo, int hi
) {
    const char* e = getenv(name);
    if (e == nullptr) return dflt;
    int x = atoi(e);
    return (x >= lo && x <= hi) ? x : dflt;
}

inline int kvshare_direct_loads_env() {
    return kvshare_env_int_reread_each_launch_so_one_build_sweeps(
        "NV_KVSHARE_DIRECT", 1, 0, 1);
}

inline bool kvshare_debug_env() {
    const char* e = getenv("NV_KVSHARE_DEBUG");
    return e != nullptr && e[0] == '1';
}

template <int GROUP, int HD>
__device__ __forceinline__ void kvshare_split_store_and_final_reduce(
    float (&acc)[GROUP][HD / kWarp],
    float (&m)[GROUP],
    float (&l)[GROUP],
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    int kvh,
    int split,
    int SPLITS,
    int lane,
    int warp
) {
    constexpr int vc = HD / kWarp;
    __shared__ float smw[kFlashWarps];
    __shared__ float slw[kFlashWarps];
    __shared__ float sacc[kFlashWarps][HD];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
        if (lane == 0) {
            smw[warp] = m[g];
            slw[warp] = l[g];
        }
        #pragma unroll
        for (int i = 0; i < vc; ++i) sacc[warp][lane * vc + i] = acc[g][i];
        __syncthreads();

        float* out = scratch + (((size_t)kvh * GROUP + g) * SPLITS + split) * (HD + 2);
        if (warp == 0) {
            float m_blk = -INFINITY;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, smw[w]);
            float l_blk = 0.0f;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w)
                l_blk += (smw[w] > -INFINITY) ? slw[w] * __expf(smw[w] - m_blk) : 0.0f;
            if (lane == 0) {
                out[0] = m_blk;
                out[1] = l_blk;
            }
        }
        __syncthreads();
        float m_blk = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kFlashWarps; ++w) m_blk = fmaxf(m_blk, smw[w]);
        for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < kFlashWarps; ++w)
                a += (smw[w] > -INFINITY) ? sacc[w][d] * __expf(smw[w] - m_blk) : 0.0f;
            out[2 + d] = a;
        }
        __syncthreads();
    }

    __threadfence();
    __shared__ unsigned int ticket;
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[kvh], 1u);
    __syncthreads();
    if (ticket != SPLITS - 1) return;
    __threadfence();

    __shared__ float ssc[kKvShareMaxSplits];
    __shared__ float sinv_l;
    for (int g = 0; g < GROUP; ++g) {
        const size_t hh = (size_t)kvh * GROUP + g;
        const float* base = scratch + hh * SPLITS * (HD + 2);
        if (threadIdx.x == 0) {
            float m_glob = -INFINITY;
            for (int s = 0; s < SPLITS; ++s)
                m_glob = fmaxf(m_glob, base[(size_t)s * (HD + 2)]);
            float l_glob = 0.0f;
            for (int s = 0; s < SPLITS; ++s) {
                const float* part = base + (size_t)s * (HD + 2);
                float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
                ssc[s] = sc;
                l_glob += part[1] * sc;
            }
            sinv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
        }
        __syncthreads();
        const float inv_l = sinv_l;
        for (int d = threadIdx.x; d < HD; d += kFlashThreads) {
            float a = 0.0f;
            for (int s = 0; s < SPLITS; ++s)
                a += base[(size_t)s * (HD + 2) + 2 + d] * ssc[s];
            outp[hh * HD + d] = __float2bfloat16(a * inv_l);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) fan_in[kvh] = 0u;
}

__device__ __forceinline__ int kv_phys_slot_pow2hint(
    const int* __restrict__ block_table,
    int block_size,
    int RING,
    int p,
    bool bs_pow2,
    int bs_shift
) {
    if (block_table != nullptr) {
        int blk = bs_pow2 ? (p >> bs_shift) : (p / block_size);
        return block_table[blk] * block_size + (p - blk * block_size);
    }
    return (RING > 0) ? (p % RING) : p;
}

template <int GROUP, int HD>
__device__ __forceinline__ void kvshare_position_softmax_update(
    const float (&qsh)[GROUP][HD],
    const uchar4 (&kcur)[HD >> 7],
    const uint2 (&vcur)[HD / kWarp / 8],
    float ks,
    float vs,
    float scaling,
    int lane,
    bool live,
    float (&m)[GROUP],
    float (&l)[GROUP],
    float (&acc)[GROUP][HD / kWarp]
) {
    constexpr int vc = HD / kWarp;
    constexpr int nck = HD >> 7;
    constexpr int nv2 = vc / 8;

    float kf[nck * 4];
    #pragma unroll
    for (int c = 0; c < nck; ++c) {
        uchar4 raw = kcur[c];
        float2 f01 = nv_fp8x2_to_float2(
            (unsigned short)(raw.x | ((unsigned short)raw.y << 8)));
        float2 f23 = nv_fp8x2_to_float2(
            (unsigned short)(raw.z | ((unsigned short)raw.w << 8)));
        kf[c * 4 + 0] = f01.x;
        kf[c * 4 + 1] = f01.y;
        kf[c * 4 + 2] = f23.x;
        kf[c * 4 + 3] = f23.y;
    }
    float vf[vc];
    #pragma unroll
    for (int t = 0; t < nv2; ++t) {
        uint2 raw = vcur[t];
        float2 f0 = nv_fp8x2_to_float2((unsigned short)(raw.x & 0xffffu));
        float2 f1 = nv_fp8x2_to_float2((unsigned short)(raw.x >> 16));
        float2 f2 = nv_fp8x2_to_float2((unsigned short)(raw.y & 0xffffu));
        float2 f3 = nv_fp8x2_to_float2((unsigned short)(raw.y >> 16));
        vf[t * 8 + 0] = f0.x;
        vf[t * 8 + 1] = f0.y;
        vf[t * 8 + 2] = f1.x;
        vf[t * 8 + 3] = f1.y;
        vf[t * 8 + 4] = f2.x;
        vf[t * 8 + 5] = f2.y;
        vf[t * 8 + 6] = f3.x;
        vf[t * 8 + 7] = f3.y;
    }

    float partial[GROUP];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
        float acc_p = 0.0f;
        #pragma unroll
        for (int c = 0; c < nck; ++c) {
            const float* qp = qsh[g] + (lane + c * kWarp) * 4;
            acc_p += qp[0] * kf[c * 4 + 0]
                   + qp[1] * kf[c * 4 + 1]
                   + qp[2] * kf[c * 4 + 2]
                   + qp[3] * kf[c * 4 + 3];
        }
        partial[g] = acc_p;
    }
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) {
        #pragma unroll
        for (int g = 0; g < GROUP; ++g)
            partial[g] += __shfl_xor_sync(0xffffffffu, partial[g], o);
    }
    if (live) {
        float corr[GROUP];
        float w_v[GROUP];
        #pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            float score = partial[g] * ks * scaling;
            float m_new = fmaxf(m[g], score);
            corr[g] = __expf(m[g] - m_new);
            float w = __expf(score - m_new);
            l[g] = l[g] * corr[g] + w;
            w_v[g] = w * vs;
            m[g] = m_new;
        }
        #pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            #pragma unroll
            for (int i = 0; i < vc; ++i)
                acc[g][i] = __fmaf_rn(w_v[g], vf[i], __fmul_rn(acc[g][i], corr[g]));
        }
    }
}

template <int GROUP, int HD>
__global__ void __launch_bounds__(kFlashThreads, 2) flash_splitk_fused_fp8_kvshare_direct_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int NKV,
    int SPLITS,
    int WINDOW,
    int RING,
    float scaling,
    const int* __restrict__ block_table,
    int block_size
) {
    static_assert((HD & 127) == 0 && HD <= kMaxHD, "uchar4 K chunks assume 128-dim multiples");
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    if (kvh >= NKV) return;

    NVK_PDL_PROLOG();

    const int total = n_total_dev[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

    constexpr int vc = HD / kWarp;
    constexpr int nck = HD >> 7;
    constexpr int nv2 = vc / 8;

    __shared__ float qsh[GROUP][HD];
    for (int i = threadIdx.x; i < GROUP * HD; i += kFlashThreads)
        qsh[i / HD][i % HD] = __bfloat162float(q[(size_t)kvh * GROUP * HD + i]);
    __syncthreads();

    float acc[GROUP][vc];
    float m[GROUP];
    float l[GROUP];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < vc; ++i) acc[g][i] = 0.0f;
    }

    const bool bs_pow2 = (block_size & (block_size - 1)) == 0;
    const int bs_shift = __ffs((unsigned)block_size) - 1;
    const int ls = SPLITS * kFlashWarps;
    const int tile0 = start + split * kFlashWarps;

    auto load_pos = [&](int p, bool& live, float& kso, float& vso,
                        uchar4 (&kc)[nck], uint2 (&vv)[nv2]) {
        live = p < total;
        kso = 0.0f;
        vso = 0.0f;
        if (live) {
            const int sp = kv_phys_slot_pow2hint(
                block_table, block_size, RING, p, bs_pow2, bs_shift);
            const size_t row = (size_t)sp * NKV + kvh;
            kso = k_scales[row];
            vso = v_scales[row];
            const uchar4* krow = reinterpret_cast<const uchar4*>(k_fp8 + row * HD);
            #pragma unroll
            for (int c = 0; c < nck; ++c) kc[c] = krow[lane + c * kWarp];
            const uint2* vrow = reinterpret_cast<const uint2*>(v_fp8 + row * HD);
            #pragma unroll
            for (int t = 0; t < nv2; ++t) vv[t] = vrow[lane * nv2 + t];
        }
    };

    for (int base = tile0; base < total; base += ls) {
        bool live;
        float ksc;
        float vsc;
        uchar4 kc[nck];
        uint2 vv[nv2];
        load_pos(base + warp, live, ksc, vsc, kc, vv);
        kvshare_position_softmax_update<GROUP, HD>(
            qsh, kc, vv, ksc, vsc, scaling, lane, live, m, l, acc);
    }

    kvshare_split_store_and_final_reduce<GROUP, HD>(
        acc, m, l, scratch, fan_in, outp, kvh, split, SPLITS, lane, warp);

    NVK_PDL_EPILOG();
}

template <int GROUP, int HD>
__global__ void __launch_bounds__(kFlashThreads, 2) flash_splitk_fused_fp8_kvshare_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int NKV,
    int SPLITS,
    int WINDOW,
    int RING,
    float scaling,
    const int* __restrict__ block_table,
    int block_size
) {
    static_assert((HD & 127) == 0 && HD <= kMaxHD, "uchar4 K chunks assume 128-dim multiples");
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    if (kvh >= NKV) return;

    NVK_PDL_PROLOG();

    const int total = n_total_dev[0];
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

    constexpr int vc = HD / kWarp;
    constexpr int nck = HD >> 7;
    constexpr int nv2 = vc / 8;

    __shared__ float qsh[GROUP][HD];
    for (int i = threadIdx.x; i < GROUP * HD; i += kFlashThreads)
        qsh[i / HD][i % HD] = __bfloat162float(q[(size_t)kvh * GROUP * HD + i]);
    __syncthreads();

    float acc[GROUP][vc];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
        #pragma unroll
        for (int i = 0; i < vc; ++i) acc[g][i] = 0.0f;
    }
    float m[GROUP];
    float l[GROUP];
    #pragma unroll
    for (int g = 0; g < GROUP; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }

    const int lane_stride = SPLITS * kFlashWarps;

    __shared__ uint4 ktile[2][kFlashWarps][HD / 16];
    __shared__ uint4 vtile[2][kFlashWarps][HD / 16];
    constexpr int kChunks16 = HD / 16;
    const int stage_row = (threadIdx.x & 127) / kChunks16;
    const int stage_chunk = (threadIdx.x & 127) % kChunks16;
    const bool stage_v = threadIdx.x >= 128;

    const int tile0 = start + split * kFlashWarps;
    auto stage_tile = [&](int buf, int base) {
        const int pos = base + stage_row;
        if (pos < total) {
            const int sp = kv_phys_slot(block_table, block_size, RING, pos);
            const uint8_t* src = (stage_v ? v_fp8 : k_fp8)
                + ((size_t)sp * NKV + kvh) * HD + stage_chunk * 16;
            uint4* dst = stage_v ? &vtile[buf][stage_row][stage_chunk]
                                 : &ktile[buf][stage_row][stage_chunk];
            __pipeline_memcpy_async(dst, src, 16);
        }
    };

    stage_tile(0, tile0);
    __pipeline_commit();
    float ks = 0.0f;
    float vs = 0.0f;
    if (tile0 + warp < total) {
        const int sp0 = kv_phys_slot(block_table, block_size, RING, tile0 + warp);
        ks = k_scales[(size_t)sp0 * NKV + kvh];
        vs = v_scales[(size_t)sp0 * NKV + kvh];
    }

    int buf = 0;
    for (int base = tile0; base < total; base += lane_stride, buf ^= 1) {
        const int nbase = base + lane_stride;
        if (nbase < total) stage_tile(buf ^ 1, nbase);
        __pipeline_commit();
        float ksn = 0.0f;
        float vsn = 0.0f;
        if (nbase + warp < total) {
            const int spn = kv_phys_slot(block_table, block_size, RING, nbase + warp);
            ksn = k_scales[(size_t)spn * NKV + kvh];
            vsn = v_scales[(size_t)spn * NKV + kvh];
        }
        __pipeline_wait_prior(1);
        __syncthreads();

        const bool live = base + warp < total;
        uchar4 kcur[nck];
        uint2 vcur[nv2];
        if (live) {
            const uchar4* krow = reinterpret_cast<const uchar4*>(ktile[buf][warp]);
            #pragma unroll
            for (int c = 0; c < nck; ++c) kcur[c] = krow[lane + c * kWarp];
            const uint2* vrow = reinterpret_cast<const uint2*>(vtile[buf][warp]);
            #pragma unroll
            for (int t = 0; t < nv2; ++t) vcur[t] = vrow[lane * nv2 + t];
        }
        kvshare_position_softmax_update<GROUP, HD>(
            qsh, kcur, vcur, ks, vs, scaling, lane, live, m, l, acc);
        ks = ksn;
        vs = vsn;
        __syncthreads();
    }

    kvshare_split_store_and_final_reduce<GROUP, HD>(
        acc, m, l, scratch, fan_in, outp, kvh, split, SPLITS, lane, warp);

    NVK_PDL_EPILOG();
}

constexpr int kProbeModeStagedPattern = 0;
constexpr int kProbeModeLinearStream = 1;
constexpr int kProbeModePlainPattern = 2;
constexpr int kProbeModePlusSlotsAndScales = 3;
constexpr int kProbeModePlusDotAndShuffle = 4;
constexpr int kProbeModeFullArithmetic = 5;

__global__ void __launch_bounds__(kFlashThreads, 2) kvshare_bw_probe_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    const int* __restrict__ block_table,
    int block_size,
    float* __restrict__ sink,
    int total,
    int NKV,
    int SPLITS,
    int mode
) {
    constexpr int HD = 256;
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    uint4 s = make_uint4(0u, 0u, 0u, 0u);

    if (mode == kProbeModeLinearStream) {
        const size_t n16 = ((size_t)total * NKV * HD) / 16;
        const size_t nthreads = (size_t)gridDim.x * gridDim.y * kFlashThreads;
        const size_t t0 =
            ((size_t)blockIdx.y * gridDim.x + blockIdx.x) * kFlashThreads + threadIdx.x;
        const uint4* k4 = reinterpret_cast<const uint4*>(k_fp8);
        const uint4* v4 = reinterpret_cast<const uint4*>(v_fp8);
        for (size_t i = t0; i < n16; i += nthreads) {
            uint4 a = k4[i];
            uint4 b = v4[i];
            s.x ^= a.x ^ b.x;
            s.y ^= a.y ^ b.y;
            s.z ^= a.z ^ b.z;
            s.w ^= a.w ^ b.w;
        }
    } else if (mode == kProbeModePlainPattern) {
        const int ls = SPLITS * kFlashWarps;
        for (int base = split * kFlashWarps; base < total; base += ls) {
            const int p = base + warp;
            if (p < total) {
                const uint8_t* src =
                    (lane < 16 ? k_fp8 : v_fp8) + ((size_t)p * NKV + kvh) * HD;
                uint4 a = reinterpret_cast<const uint4*>(src)[lane & 15];
                s.x ^= a.x;
                s.y ^= a.y;
                s.z ^= a.z;
                s.w ^= a.w;
            }
        }
    } else if (mode >= kProbeModePlusSlotsAndScales) {
        constexpr int GROUP = kKvShareGroup;
        constexpr int vc = HD / kWarp;
        constexpr int nck = HD >> 7;
        constexpr int nv2 = vc / 8;
        __shared__ float qsh[GROUP][HD];
        for (int i = threadIdx.x; i < GROUP * HD; i += kFlashThreads)
            qsh[i / HD][i % HD] = __bfloat162float(q[(size_t)kvh * GROUP * HD + i]);
        __syncthreads();
        float acc[GROUP][vc];
        float m[GROUP];
        float l[GROUP];
        #pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            m[g] = -INFINITY;
            l[g] = 0.0f;
            #pragma unroll
            for (int i = 0; i < vc; ++i) acc[g][i] = 0.0f;
        }
        float fs = 0.0f;
        const int ls = SPLITS * kFlashWarps;
        for (int base = split * kFlashWarps; base < total; base += ls) {
            const int p = base + warp;
            if (p < total) {
                const int sp = kv_phys_slot(block_table, block_size, 0, p);
                const float ks = k_scales[(size_t)sp * NKV + kvh];
                const float vs = v_scales[(size_t)sp * NKV + kvh];
                const uchar4* krow = reinterpret_cast<const uchar4*>(
                    k_fp8 + ((size_t)sp * NKV + kvh) * HD);
                uchar4 kcur[nck];
                #pragma unroll
                for (int c = 0; c < nck; ++c) kcur[c] = krow[lane + c * kWarp];
                const uint2* vrow = reinterpret_cast<const uint2*>(
                    v_fp8 + ((size_t)sp * NKV + kvh) * HD);
                uint2 vcur[nv2];
                #pragma unroll
                for (int t = 0; t < nv2; ++t) vcur[t] = vrow[lane * nv2 + t];
                if (mode == kProbeModePlusSlotsAndScales) {
                    fs += ks + vs;
                    #pragma unroll
                    for (int c = 0; c < nck; ++c)
                        s.x ^= kcur[c].x ^ ((unsigned)kcur[c].z << 8);
                    #pragma unroll
                    for (int t = 0; t < nv2; ++t) s.y ^= vcur[t].x ^ vcur[t].y;
                    continue;
                }
                float kf[nck * 4];
                #pragma unroll
                for (int c = 0; c < nck; ++c) {
                    uchar4 raw = kcur[c];
                    float2 f01 = nv_fp8x2_to_float2(
                        (unsigned short)(raw.x | ((unsigned short)raw.y << 8)));
                    float2 f23 = nv_fp8x2_to_float2(
                        (unsigned short)(raw.z | ((unsigned short)raw.w << 8)));
                    kf[c * 4 + 0] = f01.x;
                    kf[c * 4 + 1] = f01.y;
                    kf[c * 4 + 2] = f23.x;
                    kf[c * 4 + 3] = f23.y;
                }
                float vf[vc];
                #pragma unroll
                for (int t = 0; t < nv2; ++t) {
                    uint2 raw = vcur[t];
                    float2 f0 = nv_fp8x2_to_float2((unsigned short)(raw.x & 0xffffu));
                    float2 f1 = nv_fp8x2_to_float2((unsigned short)(raw.x >> 16));
                    float2 f2 = nv_fp8x2_to_float2((unsigned short)(raw.y & 0xffffu));
                    float2 f3 = nv_fp8x2_to_float2((unsigned short)(raw.y >> 16));
                    vf[t * 8 + 0] = f0.x;
                    vf[t * 8 + 1] = f0.y;
                    vf[t * 8 + 2] = f1.x;
                    vf[t * 8 + 3] = f1.y;
                    vf[t * 8 + 4] = f2.x;
                    vf[t * 8 + 5] = f2.y;
                    vf[t * 8 + 6] = f3.x;
                    vf[t * 8 + 7] = f3.y;
                }
                float partial[GROUP];
                #pragma unroll
                for (int g = 0; g < GROUP; ++g) {
                    float acc_p = 0.0f;
                    #pragma unroll
                    for (int c = 0; c < nck; ++c) {
                        const float* qp = qsh[g] + (lane + c * kWarp) * 4;
                        acc_p += qp[0] * kf[c * 4 + 0]
                               + qp[1] * kf[c * 4 + 1]
                               + qp[2] * kf[c * 4 + 2]
                               + qp[3] * kf[c * 4 + 3];
                    }
                    partial[g] = acc_p;
                }
                #pragma unroll
                for (int o = kWarp / 2; o > 0; o >>= 1) {
                    #pragma unroll
                    for (int g = 0; g < GROUP; ++g)
                        partial[g] += __shfl_xor_sync(0xffffffffu, partial[g], o);
                }
                if (mode == kProbeModePlusDotAndShuffle) {
                    #pragma unroll
                    for (int g = 0; g < GROUP; ++g) fs += partial[g] * ks;
                    #pragma unroll
                    for (int i = 0; i < vc; ++i) fs += vf[i] * vs;
                    continue;
                }
                float corr[GROUP];
                float w_v[GROUP];
                #pragma unroll
                for (int g = 0; g < GROUP; ++g) {
                    float score = partial[g] * ks * 0.0625f;
                    float m_new = fmaxf(m[g], score);
                    corr[g] = __expf(m[g] - m_new);
                    float w = __expf(score - m_new);
                    l[g] = l[g] * corr[g] + w;
                    w_v[g] = w * vs;
                    m[g] = m_new;
                }
                #pragma unroll
                for (int g = 0; g < GROUP; ++g) {
                    #pragma unroll
                    for (int i = 0; i < vc; ++i)
                        acc[g][i] = __fmaf_rn(w_v[g], vf[i], __fmul_rn(acc[g][i], corr[g]));
                }
            }
        }
        #pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            fs += m[g] + l[g];
            #pragma unroll
            for (int i = 0; i < vc; ++i) fs += acc[g][i];
        }
        s.z ^= __float_as_uint(fs);
    } else {
        constexpr int kChunks16 = HD / 16;
        __shared__ uint4 ktile[2][kFlashWarps][kChunks16];
        __shared__ uint4 vtile[2][kFlashWarps][kChunks16];
        const int stage_row = (threadIdx.x & 127) / kChunks16;
        const int stage_chunk = (threadIdx.x & 127) % kChunks16;
        const bool stage_v = threadIdx.x >= 128;
        const int ls = SPLITS * kFlashWarps;
        const int tile0 = split * kFlashWarps;
        auto stage_tile = [&](int buf, int base) {
            const int pos = base + stage_row;
            if (pos < total) {
                const uint8_t* src = (stage_v ? v_fp8 : k_fp8)
                    + ((size_t)pos * NKV + kvh) * HD + stage_chunk * 16;
                uint4* dst = stage_v ? &vtile[buf][stage_row][stage_chunk]
                                     : &ktile[buf][stage_row][stage_chunk];
                __pipeline_memcpy_async(dst, src, 16);
            }
        };
        stage_tile(0, tile0);
        __pipeline_commit();
        int buf = 0;
        for (int base = tile0; base < total; base += ls, buf ^= 1) {
            const int nbase = base + ls;
            if (nbase < total) stage_tile(buf ^ 1, nbase);
            __pipeline_commit();
            __pipeline_wait_prior(1);
            __syncthreads();
            if (base + warp < total) {
                const uint4* row = lane < 16 ? &ktile[buf][warp][0] : &vtile[buf][warp][0];
                uint4 a = row[lane & 15];
                s.x ^= a.x;
                s.y ^= a.y;
                s.z ^= a.z;
                s.w ^= a.w;
            }
            __syncthreads();
        }
    }

    unsigned r = s.x ^ s.y ^ s.z ^ s.w;
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) r ^= __shfl_xor_sync(0xffffffffu, r, o);
    if (lane == 0 && r == 0x13572468u) sink[kvh] = 1.0f;
}

}

extern "C" int nv_kernels_flash_splitk_scratch_elems(int NH, int HD) {
    int sp = splitk::flash_splits_pick(NH);
    if (splitk::q38_attn_gqa_stage_env() && sp < splitk::kKvShareMaxSplits)
        sp = splitk::kKvShareMaxSplits;
    return NH * sp * (HD + 2);
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
    if (HD > splitk::kMaxHD || (NH % NKV) != 0) return -1;
    int sp = splitk::flash_splits_pick(NH);
    dim3 grid1((unsigned)NH, (unsigned)sp);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(v);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
    switch (sp) {
        case 8:
            splitk::flash_splitk_stage1_kernel<8><<<grid1, splitk::kFlashThreads, 0, cs>>>(
                q, kb, vb, scratch, pos, NH, NKV, HD, WINDOW);
            splitk::flash_splitk_stage2_kernel<8><<<(unsigned)NH, 256, 0, cs>>>(scratch, ob, NH, HD);
            break;
        case 32:
            splitk::flash_splitk_stage1_kernel<32><<<grid1, splitk::kFlashThreads, 0, cs>>>(
                q, kb, vb, scratch, pos, NH, NKV, HD, WINDOW);
            splitk::flash_splitk_stage2_kernel<32><<<(unsigned)NH, 256, 0, cs>>>(scratch, ob, NH, HD);
            break;
        default:
            splitk::flash_splitk_stage1_kernel<16><<<grid1, splitk::kFlashThreads, 0, cs>>>(
                q, kb, vb, scratch, pos, NH, NKV, HD, WINDOW);
            splitk::flash_splitk_stage2_kernel<16><<<(unsigned)NH, 256, 0, cs>>>(scratch, ob, NH, HD);
            break;
    }
    return (int)cudaGetLastError();
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
    if (HD > splitk::kMaxHD || (NH % NKV) != 0) return -1;
    int sp = splitk::flash_splits_pick(NH);
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(v);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
    switch (sp) {
        case 8:
            splitk::flash_splitk_fused_kernel<8><<<grid, splitk::kFlashThreads, 0, cs>>>(
                q, kb, vb, scratch, fan_in, ob, pos, delta, NH, NKV, HD, WINDOW);
            break;
        case 32:
            splitk::flash_splitk_fused_kernel<32><<<grid, splitk::kFlashThreads, 0, cs>>>(
                q, kb, vb, scratch, fan_in, ob, pos, delta, NH, NKV, HD, WINDOW);
            break;
        default:
            splitk::flash_splitk_fused_kernel<16><<<grid, splitk::kFlashThreads, 0, cs>>>(
                q, kb, vb, scratch, fan_in, ob, pos, delta, NH, NKV, HD, WINDOW);
            break;
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv_mk_paged(
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
    float scaling,
    const int* block_table,
    int block_size
);

extern "C" int nv_kernels_flash_decode_gqa_fp8kv_paged(
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
    int splits,
    float scaling,
    const int* block_table,
    int block_size
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD != 512) return -1;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    if ((NH % NKV) != 0 || (NH / NKV) != splitk::kFlashWarps) return -12;
    dim3 grid((unsigned)NKV, (unsigned)splits);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
    switch (splits) {
        case 16:
            splitk::flash_splitk_fused_fp8_gqa_kernel<16, 512>
                <<<grid, splitk::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling, block_table,
                    block_size);
            break;
        case 32:
            splitk::flash_splitk_fused_fp8_gqa_kernel<32, 512>
                <<<grid, splitk::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling, block_table,
                    block_size);
            break;
        case 64:
            splitk::flash_splitk_fused_fp8_gqa_kernel<64, 512>
                <<<grid, splitk::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling, block_table,
                    block_size);
            break;
        case 128:
            splitk::flash_splitk_fused_fp8_gqa_kernel<128, 512>
                <<<grid, splitk::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, scaling, block_table,
                    block_size);
            break;
        default:
            return -13;
    }
    return (int)cudaGetLastError();
}

namespace {

int kvshare_launch(
    dim3 grid,
    cudaStream_t cs,
    const __nv_bfloat16* qb,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    float* scratch,
    unsigned int* fan_in,
    __nv_bfloat16* ob,
    const int* n_total_dev,
    int NKV,
    int sp,
    int WINDOW,
    int RING,
    float scaling,
    const int* block_table,
    int block_size
) {
    const int direct = splitk::kvshare_direct_loads_env();
    auto kern = splitk::flash_splitk_fused_fp8_kvshare_direct_kernel<
        splitk::kKvShareGroup, splitk::kKvShareHD>;
    if (direct == 0) {
        kern = splitk::flash_splitk_fused_fp8_kvshare_kernel<
            splitk::kKvShareGroup, splitk::kKvShareHD>;
    }
    if (splitk::kvshare_debug_env()) {
        cudaFuncAttributes fa{};
        cudaFuncGetAttributes(&fa, kern);
        int occ = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, kern, splitk::kFlashThreads, 0);
        fprintf(stderr,
                "[kvshare-debug] direct=%d regs=%d smem=%zu "
                "local=%zu occ_blocks_per_sm=%d sms=%d\n",
                direct, fa.numRegs, fa.sharedSizeBytes,
                fa.localSizeBytes, occ, splitk::flash_sm_count());
    }
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(splitk::kFlashThreads), 0, cs);
        cudaLaunchKernelEx(
            &cfg, kern, qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in,
            ob, n_total_dev, NKV, sp, WINDOW, RING, scaling, block_table,
            block_size);
    } else {
        kern<<<grid, splitk::kFlashThreads, 0, cs>>>(
            qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
            n_total_dev, NKV, sp, WINDOW, RING, scaling, block_table, block_size);
    }
    return (int)cudaGetLastError();
}

}

extern "C" int nv_kernels_flash_decode_kvshare_fp8kv_paged(
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
    int splits,
    float scaling,
    const int* block_table,
    int block_size
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD != splitk::kKvShareHD) return -21;
    if ((NH % NKV) != 0 || (NH / NKV) != splitk::kKvShareGroup) return -22;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    int sp = (splits > 0) ? splits : splitk::kvshare_splits_pick(NKV);
    if (sp < 1 || sp > splitk::kKvShareMaxSplits) return -23;
    dim3 grid((unsigned)NKV, (unsigned)sp);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
    return kvshare_launch(
        grid, cs, qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
        n_total_dev, NKV, sp, WINDOW, RING, scaling, block_table, block_size);
}

extern "C" int nv_kernels_kvshare_bw_probe(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scales,
    const float* v_scales,
    const int* block_table,
    int block_size,
    float* sink,
    int total,
    int NKV,
    int SPLITS,
    int mode
) {
    dim3 grid((unsigned)NKV, (unsigned)SPLITS);
    cudaStream_t cs = (cudaStream_t)stream;
    splitk::kvshare_bw_probe_kernel<<<grid, splitk::kFlashThreads, 0, cs>>>(
        reinterpret_cast<const __nv_bfloat16*>(q), k_fp8, v_fp8, k_scales,
        v_scales, block_table, block_size, sink, total, NKV, SPLITS, mode);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_flash_decode_derivev_fp8kv_paged(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const float* k_scales,
    const float* inv_freq,
    const float* cos_pk,
    const float* sin_pk,
    uint16_t* out,
    const int* n_total_dev,
    float* scratch,
    unsigned int* fan_in,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    int rope_angles,
    float w_inv,
    float scaling,
    const int* block_table,
    int block_size
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD > splitk::kMaxHD || (NH % NKV) != 0) return -1;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    if (HD != 512) return -8;
    if (rope_angles < 0 || rope_angles > 128) return -9;
    if (inv_freq == nullptr && cos_pk == nullptr) return -10;
    if (cos_pk != nullptr && ((rope_angles & 3) != 0 || sin_pk == nullptr)) return -11;
    int sp = splitk::flash_splits_pick(NH);
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
    switch (sp) {
        case 8:
            splitk::flash_splitk_fused_fp8_derivev_kernel<8, 512>
                <<<grid, splitk::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, k_scales, inv_freq, cos_pk, sin_pk, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, rope_angles, w_inv,
                    scaling, block_table, block_size);
            break;
        case 32:
            splitk::flash_splitk_fused_fp8_derivev_kernel<32, 512>
                <<<grid, splitk::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, k_scales, inv_freq, cos_pk, sin_pk, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, rope_angles, w_inv,
                    scaling, block_table, block_size);
            break;
        default:
            splitk::flash_splitk_fused_fp8_derivev_kernel<16, 512>
                <<<grid, splitk::kFlashThreads, 0, cs>>>(
                    qb, k_fp8, k_scales, inv_freq, cos_pk, sin_pk, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, WINDOW, RING, rope_angles, w_inv,
                    scaling, block_table, block_size);
            break;
    }
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv_paged(
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
    float scaling,
    const int* block_table,
    int block_size
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (HD > splitk::kMaxHD || (NH % NKV) != 0) return -1;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    if (splitk::q38_attn_gqa_stage_env() && HD == splitk::kKvShareHD
        && (NH / NKV) == splitk::kKvShareGroup) {
        return nv_kernels_flash_decode_kvshare_fp8kv_paged(
            stream, q, k_fp8, v_fp8, k_scales, v_scales, out, n_total_dev,
            scratch, fan_in, NH, NKV, HD, WINDOW, RING, 0, scaling,
            block_table, block_size);
    }
    int sp = splitk::flash_splits_pick(NH);
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
    if (nvk_pdl_enabled()) {
        NVK_PDL_ATTR(cfg, grid, dim3(splitk::kFlashThreads), 0, cs);
        switch (sp) {
            case 8:
                cudaLaunchKernelEx(
                    &cfg, splitk::flash_splitk_fused_fp8_kernel<8>,
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, HD, WINDOW, RING, scaling,
                    block_table, block_size);
                break;
            case 32:
                cudaLaunchKernelEx(
                    &cfg, splitk::flash_splitk_fused_fp8_kernel<32>,
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, HD, WINDOW, RING, scaling,
                    block_table, block_size);
                break;
            default:
                cudaLaunchKernelEx(
                    &cfg, splitk::flash_splitk_fused_fp8_kernel<16>,
                    qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                    n_total_dev, NH, NKV, HD, WINDOW, RING, scaling,
                    block_table, block_size);
                break;
        }
        return (int)cudaGetLastError();
    }
    switch (sp) {
        case 8:
            splitk::flash_splitk_fused_fp8_kernel<8><<<grid, splitk::kFlashThreads, 0, cs>>>(
                qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                n_total_dev, NH, NKV, HD, WINDOW, RING, scaling,
                block_table, block_size);
            break;
        case 32:
            splitk::flash_splitk_fused_fp8_kernel<32><<<grid, splitk::kFlashThreads, 0, cs>>>(
                qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                n_total_dev, NH, NKV, HD, WINDOW, RING, scaling,
                block_table, block_size);
            break;
        default:
            splitk::flash_splitk_fused_fp8_kernel<16><<<grid, splitk::kFlashThreads, 0, cs>>>(
                qb, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, ob,
                n_total_dev, NH, NKV, HD, WINDOW, RING, scaling,
                block_table, block_size);
            break;
    }
    return (int)cudaGetLastError();
}

namespace splitk_mk {

constexpr int kWarp = 32;
constexpr int kFlashWarps = 8;
constexpr int kFlashThreads = kWarp * kFlashWarps;
constexpr int kMaxHDmk = 256;
constexpr int kMaxAccMk = kMaxHDmk / kWarp;
constexpr int kMaxM = 8;
constexpr int kMaxHDmkFp8 = 512;
static_assert(
    kMaxHDmkFp8 > kMaxHDmk,
    "the fp8 mk launcher instantiates MH=512 kernels and the bf16 one does not, so the "
    "fp8 head-dim bound is deliberately larger than kMaxHDmk; collapsing it back makes the "
    "HD > kMaxHDmk branch in nv_kernels_flash_decode_fused_fp8kv_mk_paged unreachable and "
    "every hd512 call returns -1 having launched nothing");
constexpr int kChunks4 = kMaxHDmk / (4 * kWarp);

__inline__ __device__ float2 fp8x2_to_float2(unsigned short packed) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed, __NV_E4M3);
    __half2 h2 = *reinterpret_cast<__half2*>(&hr);
    return __half22float2(h2);
}

__inline__ __device__ float fp8_to_float(unsigned char b) {
    return fp8x2_to_float2((unsigned short)b).x;
}

template <int SPLITS, int M>
__global__ void flash_splitk_fused_mk_kernel(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ pos,
    int delta,
    int NH,
    int NKV,
    int HD,
    int WINDOW
) {
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

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

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
    const int vc = HD / kWarp;
    const bool vecv = (HD % (kWarp * 8)) == 0 && vc <= kMaxAccMk;
    const int lane_stride = SPLITS * kFlashWarps;

    if (vec8 && (HD >> 3) == kWarp && vecv) {
        int p = start + split * kFlashWarps + warp;
        uint4 kraw, vraw;
        if (p < total) {
            kraw = __ldg(reinterpret_cast<const uint4*>(k + ((size_t)p * NKV + kvh) * HD) + lane);
            vraw = __ldg(reinterpret_cast<const uint4*>(v + ((size_t)p * NKV + kvh) * HD) + lane);
        }
        for (; p < total; p += lane_stride) {
            const int pn = p + lane_stride;
            uint4 kn, vn;
            if (pn < total) {
                kn = __ldg(reinterpret_cast<const uint4*>(k + ((size_t)pn * NKV + kvh) * HD) + lane);
                vn = __ldg(reinterpret_cast<const uint4*>(v + ((size_t)pn * NKV + kvh) * HD) + lane);
            }
            const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&kraw);
            const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(&vraw);
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
                float score = splitk::warp_sum(partial);
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
    } else
    for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
        const __nv_bfloat16* kp = k + ((size_t)p * NKV + kvh) * HD;
        const __nv_bfloat16* vp = v + ((size_t)p * NKV + kvh) * HD;
        const int n8 = HD >> 3;
        float kf[8];
        float vf[kMaxAccMk];
        if (vec8) {
            if (lane < n8) {
                uint4 raw = __ldg(reinterpret_cast<const uint4*>(kp) + lane);
                const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(&raw);
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
                int d = lane + i * kWarp;
                if (d < HD) kf[i] = __bfloat162float(kp[d]);
            }
        }
        #pragma unroll
        for (int i = 0; i < kMaxAccMk; ++i) {
            int d = lane + i * kWarp;
            if (d < HD) vf[i] = __bfloat162float(__ldg(&vp[d]));
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
                    int d = lane + i * kWarp;
                    if (d < HD) partial += qsh[qi * HD + d] * kf[i];
                }
            }
            float score = splitk::warp_sum(partial);
            float m_new = fmaxf(m[qi], score);
            float corr = __expf(m[qi] - m_new);
            float w = __expf(score - m_new);
            l[qi] = l[qi] * corr + w;
            #pragma unroll
            for (int i = 0; i < kMaxAccMk; ++i) {
                int d = lane + i * kWarp;
                if (d < HD) acc[qi][i] = __fmaf_rn(acc[qi][i], corr, w * vf[i]);
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
                int d = lane + i * kWarp;
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
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[h], 1u);
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
    if (threadIdx.x == 0) fan_in[h] = 0u;
}

template <int SPLITS, int M, int MAXHD>
__global__ void flash_splitk_fused_fp8_mk_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ n_total_dev,
    int delta,
    int NH,
    int NKV,
    int HD,
    int WINDOW,
    int RING,
    float scaling,
    const int* __restrict__ block_table,
    int block_size
) {
    constexpr int kAccT = MAXHD / kWarp;
    constexpr int kCh4T = MAXHD / (4 * kWarp);
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

    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

    __shared__ float qsh[M * MAXHD];
    for (int t = threadIdx.x; t < M * HD; t += kFlashThreads) {
        int qi = t / HD;
        int d = t - qi * HD;
        qsh[t] = __bfloat162float(q[((size_t)qi * NH + h) * HD + d]);
    }
    __syncthreads();

    float acc[M][kAccT];
    float m[M];
    float l[M];
    #pragma unroll
    for (int qi = 0; qi < M; ++qi) {
        #pragma unroll
        for (int i = 0; i < kAccT; ++i) acc[qi][i] = 0.0f;
        m[qi] = -INFINITY;
        l[qi] = 0.0f;
    }

    const bool vec4 = (HD & 3) == 0;
    const int n4 = HD >> 2;
    const int vc = HD / kWarp;
    const bool vecv = (HD % (kWarp * 8)) == 0 && vc == 8 && M <= 4;
    const int lane_stride = SPLITS * kFlashWarps;

    for (int p = start + split * kFlashWarps + warp; p < total; p += lane_stride) {
        const int sp = kv_phys_slot(block_table, block_size, RING, p);
        const uint8_t* kp = k_fp8 + ((size_t)sp * NKV + kvh) * HD;
        const float ks = k_scales[(size_t)sp * NKV + kvh];
        const uint8_t* vp = v_fp8 + ((size_t)sp * NKV + kvh) * HD;
        const float vs = v_scales[(size_t)sp * NKV + kvh];

        float kf[4 * kCh4T];
        if (vec4) {
            const uchar4* k4 = reinterpret_cast<const uchar4*>(kp);
            #pragma unroll
            for (int c = 0; c < kCh4T; ++c) {
                int j = lane + c * kWarp;
                if (j < n4) {
                    uchar4 raw = __ldg(&k4[j]);
                    float2 f01 = fp8x2_to_float2(
                        (unsigned short)(raw.x | ((unsigned short)raw.y << 8)));
                    float2 f23 = fp8x2_to_float2(
                        (unsigned short)(raw.z | ((unsigned short)raw.w << 8)));
                    kf[4 * c] = f01.x;
                    kf[4 * c + 1] = f01.y;
                    kf[4 * c + 2] = f23.x;
                    kf[4 * c + 3] = f23.y;
                }
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kAccT; ++i) {
                int d = lane + i * kWarp;
                if (d < HD) kf[i] = fp8_to_float(kp[d]);
            }
        }
        float vf[kAccT];
        if (vecv) {
            uint2 raw = __ldg(reinterpret_cast<const uint2*>(vp + lane * vc));
            float2 f0 = fp8x2_to_float2((unsigned short)(raw.x & 0xffffu));
            float2 f1 = fp8x2_to_float2((unsigned short)(raw.x >> 16));
            float2 f2 = fp8x2_to_float2((unsigned short)(raw.y & 0xffffu));
            float2 f3 = fp8x2_to_float2((unsigned short)(raw.y >> 16));
            vf[0] = f0.x;
            vf[1] = f0.y;
            vf[2] = f1.x;
            vf[3] = f1.y;
            vf[4] = f2.x;
            vf[5] = f2.y;
            vf[6] = f3.x;
            vf[7] = f3.y;
        } else {
            #pragma unroll
            for (int i = 0; i < kAccT; ++i) {
                int d = lane + i * kWarp;
                if (d < HD) vf[i] = fp8_to_float(__ldg(&vp[d]));
            }
        }

        #pragma unroll
        for (int qi = 0; qi < M; ++qi) {
            if (p < start_q[qi] || p >= total_q[qi]) continue;
            float partial = 0.0f;
            if (vec4) {
                #pragma unroll
                for (int c = 0; c < kCh4T; ++c) {
                    int j = lane + c * kWarp;
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
                for (int i = 0; i < kAccT; ++i) {
                    int d = lane + i * kWarp;
                    if (d < HD) partial += qsh[qi * HD + d] * kf[i];
                }
            }
            float score = splitk::warp_sum(partial) * ks * scaling;
            float m_new = fmaxf(m[qi], score);
            float corr = __expf(m[qi] - m_new);
            float w = __expf(score - m_new);
            l[qi] = l[qi] * corr + w;
            const float w_v = w * vs;
            #pragma unroll
            for (int i = 0; i < kAccT; ++i) {
                int d = lane + i * kWarp;
                if (d < HD) acc[qi][i] = __fmaf_rn(w_v, vf[i], __fmul_rn(acc[qi][i], corr));
            }
            m[qi] = m_new;
        }
    }

    __shared__ float sm[kFlashWarps];
    __shared__ float sl[kFlashWarps];
    __shared__ float sacc[kFlashWarps][MAXHD];
    for (int qi = 0; qi < M; ++qi) {
        __syncthreads();
        if (lane == 0) {
            sm[warp] = m[qi];
            sl[warp] = l[qi];
        }
        if (vecv) {
            #pragma unroll
            for (int i = 0; i < kAccT; ++i) {
                if (i >= vc) break;
                sacc[warp][lane * vc + i] = acc[qi][i];
            }
        } else {
            #pragma unroll
            for (int i = 0; i < kAccT; ++i) {
                int d = lane + i * kWarp;
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
    if (threadIdx.x == 0) ticket = atomicAdd(&fan_in[h], 1u);
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
    if (threadIdx.x == 0) fan_in[h] = 0u;
}

}

extern "C" int nv_kernels_flash_splitk_scratch_elems_mk(int NH, int HD, int M) {
    return NH * M * splitk::flash_splits_pick(NH) * (HD + 2);
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
    int sp = splitk::flash_splits_pick(NH);
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(v);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
#define NV_MK_BF16_CASE(SP, MM) \
    case MM: \
        splitk_mk::flash_splitk_fused_mk_kernel<SP, MM> \
            <<<grid, splitk_mk::kFlashThreads, 0, cs>>>( \
                q, kb, vb, scratch, fan_in, ob, pos, delta, NH, NKV, HD, WINDOW); \
        break;
#define NV_MK_BF16_SWITCH(SP) \
    switch (M) { \
        NV_MK_BF16_CASE(SP, 1) \
        NV_MK_BF16_CASE(SP, 2) \
        NV_MK_BF16_CASE(SP, 3) \
        NV_MK_BF16_CASE(SP, 4) \
        NV_MK_BF16_CASE(SP, 5) \
        NV_MK_BF16_CASE(SP, 6) \
        NV_MK_BF16_CASE(SP, 7) \
        NV_MK_BF16_CASE(SP, 8) \
    } \
    break;
    switch (sp) {
        case 8: NV_MK_BF16_SWITCH(8)
        case 32: NV_MK_BF16_SWITCH(32)
        default: NV_MK_BF16_SWITCH(16)
    }
#undef NV_MK_BF16_SWITCH
#undef NV_MK_BF16_CASE
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv_mk_paged(
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
    float scaling,
    const int* block_table,
    int block_size
) {
    if (NH <= 0 || NKV <= 0) return 0;
    if (M < 1 || M > splitk_mk::kMaxM) return -1;
    if (HD > splitk_mk::kMaxHDmkFp8 || (NH % NKV) != 0) return -1;
    if (RING < 0 || (RING > 0 && WINDOW <= 0)) return -3;
    int sp = splitk::flash_splits_pick(NH);
    dim3 grid((unsigned)NH, (unsigned)sp);
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
    cudaStream_t cs = (cudaStream_t)stream;
#define NV_MK_FP8_CASE(SP, MM, MH) \
    case MM: \
        splitk_mk::flash_splitk_fused_fp8_mk_kernel<SP, MM, MH> \
            <<<grid, splitk_mk::kFlashThreads, 0, cs>>>( \
                qc, k_fp8, v_fp8, k_scales, v_scales, scratch, fan_in, oc, \
                n_total_dev, dc, NH, NKV, HD, WINDOW, RING, scaling, \
                block_table, block_size); \
        break;
    if (HD > splitk_mk::kMaxHDmk) {
        static int chunk = [] {
            const char* v = getenv("NV_MK512_CHUNK");
            int c = v ? atoi(v) : 4;
            return (c >= 1 && c <= 8) ? c : 8;
        }();
        for (int q0 = 0; q0 < M; q0 += chunk) {
            const int cm = (M - q0 < chunk) ? (M - q0) : chunk;
            const __nv_bfloat16* qc = qb + (size_t)q0 * NH * HD;
            __nv_bfloat16* oc = ob + (size_t)q0 * NH * HD;
            const int dc = delta + (M - q0 - cm);
#define NV_MK_FP8_SWITCH512(SP) \
    switch (cm) { \
        NV_MK_FP8_CASE(SP, 1, 512) \
        NV_MK_FP8_CASE(SP, 2, 512) \
        NV_MK_FP8_CASE(SP, 3, 512) \
        NV_MK_FP8_CASE(SP, 4, 512) \
        NV_MK_FP8_CASE(SP, 5, 512) \
        NV_MK_FP8_CASE(SP, 6, 512) \
        NV_MK_FP8_CASE(SP, 7, 512) \
        NV_MK_FP8_CASE(SP, 8, 512) \
    } \
    break;
            switch (sp) {
                case 8: NV_MK_FP8_SWITCH512(8)
                case 32: NV_MK_FP8_SWITCH512(32)
                default: NV_MK_FP8_SWITCH512(16)
            }
#undef NV_MK_FP8_SWITCH512
        }
        return (int)cudaGetLastError();
    }
    {
        const __nv_bfloat16* qc = qb;
        __nv_bfloat16* oc = ob;
        const int dc = delta;
#define NV_MK_FP8_SWITCH(SP) \
    switch (M) { \
        NV_MK_FP8_CASE(SP, 1, 256) \
        NV_MK_FP8_CASE(SP, 2, 256) \
        NV_MK_FP8_CASE(SP, 3, 256) \
        NV_MK_FP8_CASE(SP, 4, 256) \
        NV_MK_FP8_CASE(SP, 5, 256) \
        NV_MK_FP8_CASE(SP, 6, 256) \
        NV_MK_FP8_CASE(SP, 7, 256) \
        NV_MK_FP8_CASE(SP, 8, 256) \
    } \
    break;
        switch (sp) {
            case 8: NV_MK_FP8_SWITCH(8)
            case 32: NV_MK_FP8_SWITCH(32)
            default: NV_MK_FP8_SWITCH(16)
        }
#undef NV_MK_FP8_SWITCH
    }
#undef NV_MK_FP8_CASE
    return (int)cudaGetLastError();
}

__global__ void write_kv_bf16_kernel(
    const float* __restrict__ src_k,
    const float* __restrict__ src_v,
    __nv_bfloat16* __restrict__ cache_k,
    __nv_bfloat16* __restrict__ cache_v,
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
    write_kv_bf16_kernel<<<(unsigned)NKV, 128, 0, (cudaStream_t)stream>>>(
        src_k, src_v,
        reinterpret_cast<__nv_bfloat16*>(cache_k),
        reinterpret_cast<__nv_bfloat16*>(cache_v),
        pos, NKV, HD);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv(
    void* stream, const uint16_t* q, const uint8_t* k_fp8, const uint8_t* v_fp8,
    const float* k_scales, const float* v_scales, uint16_t* out,
    const int* n_total_dev, float* scratch, unsigned int* fan_in,
    int NH, int NKV, int HD, int WINDOW, int RING, float scaling
) {
    return nv_kernels_flash_decode_fused_fp8kv_paged(
        stream, q, k_fp8, v_fp8, k_scales, v_scales, out, n_total_dev, scratch,
        fan_in, NH, NKV, HD, WINDOW, RING, scaling, nullptr, 0);
}

extern "C" int nv_kernels_flash_decode_fused_fp8kv_mk(
    void* stream, const uint16_t* q, const uint8_t* k_fp8, const uint8_t* v_fp8,
    const float* k_scales, const float* v_scales, uint16_t* out,
    const int* n_total_dev, int delta, int M, float* scratch,
    unsigned int* fan_in, int NH, int NKV, int HD, int WINDOW, int RING,
    float scaling
) {
    return nv_kernels_flash_decode_fused_fp8kv_mk_paged(
        stream, q, k_fp8, v_fp8, k_scales, v_scales, out, n_total_dev, delta, M,
        scratch, fan_in, NH, NKV, HD, WINDOW, RING, scaling, nullptr, 0);
}
