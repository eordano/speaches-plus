#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <stdint.h>
#include <math.h>

namespace gqa512 {

constexpr int kWarp = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarp * kWarps;
constexpr int kHD = 512;
constexpr int kRow4 = kHD / 8;
constexpr int kTile = 8;
constexpr int kRow4F = kHD / 16;
constexpr int kTileF = 16;
constexpr int kMaxSplits = 128;
constexpr int kQSmemMinM = 5;
constexpr int kQRow4 = kHD / 8;

__host__ __device__ constexpr size_t q_smem_bytes(int m) {
    return m >= kQSmemMinM ? (size_t)m * kWarps * kHD * 2 : 0;
}

__inline__ __device__ float2 fp8x2_to_float2(unsigned short packed) {
    __half2_raw hr = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)packed, __NV_E4M3);
    __half2 h2 = *reinterpret_cast<__half2*>(&hr);
    return __half22float2(h2);
}

__inline__ __device__ void unpack16_fp8(const uint4& raw, float* dst) {
    const unsigned int words[4] = {raw.x, raw.y, raw.z, raw.w};
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 lo = fp8x2_to_float2((unsigned short)(words[j] & 0xffffu));
        float2 hi = fp8x2_to_float2((unsigned short)(words[j] >> 16));
        dst[4 * j] = lo.x;
        dst[4 * j + 1] = lo.y;
        dst[4 * j + 2] = hi.x;
        dst[4 * j + 3] = hi.y;
    }
}

__inline__ __device__ float warp_sum(float x) {
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__inline__ __device__ void unpack8(const uint4& raw, float* dst) {
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float2 f = __bfloat1622float2(b[j]);
        dst[2 * j] = f.x;
        dst[2 * j + 1] = f.y;
    }
}

template <int M>
__global__ void __launch_bounds__(kThreads, 1) gqa512_stage1_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ scratch,
    const int* __restrict__ pos,
    int delta,
    int NH,
    int NKV
) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int splits = gridDim.y;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int h = kvh * kWarps + warp;

    const int total = pos[0] - delta;
    int total_q[M];
    #pragma unroll
    for (int qi = 0; qi < M; ++qi) total_q[qi] = total - (M - 1) + qi;

    const int chunk = (total + splits - 1) / splits;
    const int p0 = split * chunk;
    int p1 = p0 + chunk;
    if (p1 > total) p1 = total;

    constexpr bool kQS = M >= kQSmemMinM;
    constexpr int kQR = kQS ? 1 : M;
    extern __shared__ uint4 qsm4[];
    uint4 qreg[kQR][2];
    if constexpr (kQS) {
        const int nq4 = M * kWarps * kQRow4;
        for (int u = threadIdx.x; u < nq4; u += kThreads) {
            const int qi = u / (kWarps * kQRow4);
            const int rem = u - qi * (kWarps * kQRow4);
            const int hh = rem / kQRow4;
            const int c = rem - hh * kQRow4;
            qsm4[u] = reinterpret_cast<const uint4*>(
                q + ((size_t)qi * NH + kvh * kWarps + hh) * kHD)[c];
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int qi = 0; qi < kQR; ++qi) {
            const uint4* qp = reinterpret_cast<const uint4*>(q + ((size_t)qi * NH + h) * kHD);
            qreg[qi][0] = qp[lane];
            qreg[qi][1] = qp[lane + kWarp];
        }
    }

    float acc[M][16];
    float m[M];
    float l[M];
    #pragma unroll
    for (int qi = 0; qi < M; ++qi) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc[qi][i] = 0.0f;
        m[qi] = -INFINITY;
        l[qi] = 0.0f;
    }

    __shared__ uint4 ksm[2][kTile][kRow4];
    __shared__ uint4 vsm[2][kTile][kRow4];

    auto load_tile = [&](int tb, int buf) {
        for (int u = threadIdx.x; u < kTile * kRow4; u += kThreads) {
            const int r = u >> 6;
            const int c = u & (kRow4 - 1);
            const int p = tb + r;
            if (p < p1) {
                const uint4* ks =
                    reinterpret_cast<const uint4*>(k + ((size_t)p * NKV + kvh) * kHD) + c;
                const uint4* vs =
                    reinterpret_cast<const uint4*>(v + ((size_t)p * NKV + kvh) * kHD) + c;
                __pipeline_memcpy_async(&ksm[buf][r][c], ks, 16);
                __pipeline_memcpy_async(&vsm[buf][r][c], vs, 16);
            }
        }
        __pipeline_commit();
    };

    if (p0 < p1) {
        load_tile(p0, 0);
        int buf = 0;
        for (int tb = p0; tb < p1; tb += kTile, buf ^= 1) {
            if (tb + kTile < p1) {
                load_tile(tb + kTile, buf ^ 1);
            } else {
                __pipeline_commit();
            }
            __pipeline_wait_prior(1);
            __syncthreads();
            const int tcnt = min(kTile, p1 - tb);
            for (int t = 0; t < tcnt; ++t) {
                const int p = tb + t;
                float kf[16];
                float vf[16];
                unpack8(ksm[buf][t][lane], kf);
                unpack8(ksm[buf][t][lane + kWarp], kf + 8);
                unpack8(vsm[buf][t][lane], vf);
                unpack8(vsm[buf][t][lane + kWarp], vf + 8);
                #pragma unroll
                for (int qi = 0; qi < M; ++qi) {
                    if (p >= total_q[qi]) continue;
                    uint4 q0, q1;
                    if constexpr (kQS) {
                        const uint4* qrow = qsm4 + ((size_t)qi * kWarps + warp) * kQRow4;
                        q0 = qrow[lane];
                        q1 = qrow[lane + kWarp];
                    } else {
                        q0 = qreg[qi][0];
                        q1 = qreg[qi][1];
                    }
                    const __nv_bfloat162* qb0 = reinterpret_cast<const __nv_bfloat162*>(&q0);
                    const __nv_bfloat162* qb1 = reinterpret_cast<const __nv_bfloat162*>(&q1);
                    float partial = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float2 f0 = __bfloat1622float2(qb0[j]);
                        float2 f1 = __bfloat1622float2(qb1[j]);
                        partial += f0.x * kf[2 * j] + f0.y * kf[2 * j + 1]
                                 + f1.x * kf[8 + 2 * j] + f1.y * kf[8 + 2 * j + 1];
                    }
                    float score = warp_sum(partial);
                    float m_new = fmaxf(m[qi], score);
                    float corr = __expf(m[qi] - m_new);
                    float w = __expf(score - m_new);
                    l[qi] = l[qi] * corr + w;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i)
                        acc[qi][i] = __fmaf_rn(w, vf[i], __fmul_rn(acc[qi][i], corr));
                    m[qi] = m_new;
                }
            }
            __syncthreads();
        }
        __pipeline_wait_prior(0);
    }

    #pragma unroll
    for (int qi = 0; qi < M; ++qi) {
        float* outp = scratch + (((size_t)h * M + qi) * splits + split) * (kHD + 2);
        if (lane == 0) {
            outp[0] = m[qi];
            outp[1] = l[qi];
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) outp[2 + 8 * lane + i] = acc[qi][i];
        #pragma unroll
        for (int i = 0; i < 8; ++i) outp[2 + 256 + 8 * lane + i] = acc[qi][8 + i];
    }
}

template <int M>
__global__ void __launch_bounds__(kThreads, 1) gqa512_stage1_fp8_kernel(
    const __nv_bfloat16* __restrict__ q,
    const uint8_t* __restrict__ k_fp8,
    const uint8_t* __restrict__ v_fp8,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    float* __restrict__ scratch,
    const int* __restrict__ pos,
    int delta,
    int NH,
    int NKV,
    float scaling
) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int splits = gridDim.y;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    const int h = kvh * kWarps + warp;

    const int total = pos[0] - delta;
    int total_q[M];
    #pragma unroll
    for (int qi = 0; qi < M; ++qi) total_q[qi] = total - (M - 1) + qi;

    const int chunk = (total + splits - 1) / splits;
    const int p0 = split * chunk;
    int p1 = p0 + chunk;
    if (p1 > total) p1 = total;

    constexpr bool kQS = M >= kQSmemMinM;
    constexpr int kQR = kQS ? 1 : M;
    extern __shared__ uint4 qsm4[];
    uint4 qreg[kQR][2];
    if constexpr (kQS) {
        const int nq4 = M * kWarps * kQRow4;
        for (int u = threadIdx.x; u < nq4; u += kThreads) {
            const int qi = u / (kWarps * kQRow4);
            const int rem = u - qi * (kWarps * kQRow4);
            const int hh = rem / kQRow4;
            const int c = rem - hh * kQRow4;
            qsm4[u] = reinterpret_cast<const uint4*>(
                q + ((size_t)qi * NH + kvh * kWarps + hh) * kHD)[c];
        }
        __syncthreads();
    } else {
        #pragma unroll
        for (int qi = 0; qi < kQR; ++qi) {
            const uint4* qp = reinterpret_cast<const uint4*>(q + ((size_t)qi * NH + h) * kHD);
            qreg[qi][0] = qp[2 * lane];
            qreg[qi][1] = qp[2 * lane + 1];
        }
    }

    float acc[M][16];
    float m[M];
    float l[M];
    #pragma unroll
    for (int qi = 0; qi < M; ++qi) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) acc[qi][i] = 0.0f;
        m[qi] = -INFINITY;
        l[qi] = 0.0f;
    }

    __shared__ uint4 ksm[2][kTileF][kRow4F];
    __shared__ uint4 vsm[2][kTileF][kRow4F];
    __shared__ float ssc[2][2][kTileF];

    auto load_tile = [&](int tb, int buf) {
        for (int u = threadIdx.x; u < kTileF * kRow4F; u += kThreads) {
            const int r = u >> 5;
            const int c = u & (kRow4F - 1);
            const int p = tb + r;
            if (p < p1) {
                const uint4* ks =
                    reinterpret_cast<const uint4*>(k_fp8 + ((size_t)p * NKV + kvh) * kHD) + c;
                const uint4* vs =
                    reinterpret_cast<const uint4*>(v_fp8 + ((size_t)p * NKV + kvh) * kHD) + c;
                __pipeline_memcpy_async(&ksm[buf][r][c], ks, 16);
                __pipeline_memcpy_async(&vsm[buf][r][c], vs, 16);
            }
        }
        for (int r = threadIdx.x; r < kTileF; r += kThreads) {
            const int p = tb + r;
            if (p < p1) {
                __pipeline_memcpy_async(&ssc[buf][0][r], &k_scale[(size_t)p * NKV + kvh], 4);
                __pipeline_memcpy_async(&ssc[buf][1][r], &v_scale[(size_t)p * NKV + kvh], 4);
            }
        }
        __pipeline_commit();
    };

    if (p0 < p1) {
        load_tile(p0, 0);
        int buf = 0;
        for (int tb = p0; tb < p1; tb += kTileF, buf ^= 1) {
            if (tb + kTileF < p1) {
                load_tile(tb + kTileF, buf ^ 1);
            } else {
                __pipeline_commit();
            }
            __pipeline_wait_prior(1);
            __syncthreads();
            const int tcnt = min(kTileF, p1 - tb);
            for (int t = 0; t < tcnt; ++t) {
                const int p = tb + t;
                const float ks = ssc[buf][0][t];
                const float vs = ssc[buf][1][t];
                float kf[16];
                float vf[16];
                unpack16_fp8(ksm[buf][t][lane], kf);
                unpack16_fp8(vsm[buf][t][lane], vf);
                #pragma unroll
                for (int qi = 0; qi < M; ++qi) {
                    if (p >= total_q[qi]) continue;
                    uint4 q0, q1;
                    if constexpr (kQS) {
                        const uint4* qrow = qsm4 + ((size_t)qi * kWarps + warp) * kQRow4;
                        q0 = qrow[2 * lane];
                        q1 = qrow[2 * lane + 1];
                    } else {
                        q0 = qreg[qi][0];
                        q1 = qreg[qi][1];
                    }
                    const __nv_bfloat162* qb0 = reinterpret_cast<const __nv_bfloat162*>(&q0);
                    const __nv_bfloat162* qb1 = reinterpret_cast<const __nv_bfloat162*>(&q1);
                    float partial = 0.0f;
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        float2 f0 = __bfloat1622float2(qb0[j]);
                        float2 f1 = __bfloat1622float2(qb1[j]);
                        partial += f0.x * kf[2 * j] + f0.y * kf[2 * j + 1]
                                 + f1.x * kf[8 + 2 * j] + f1.y * kf[8 + 2 * j + 1];
                    }
                    float score = warp_sum(partial) * ks * scaling;
                    float m_new = fmaxf(m[qi], score);
                    float corr = __expf(m[qi] - m_new);
                    float w = __expf(score - m_new);
                    l[qi] = l[qi] * corr + w;
                    const float w_v = w * vs;
                    #pragma unroll
                    for (int i = 0; i < 16; ++i)
                        acc[qi][i] = __fmaf_rn(w_v, vf[i], __fmul_rn(acc[qi][i], corr));
                    m[qi] = m_new;
                }
            }
            __syncthreads();
        }
        __pipeline_wait_prior(0);
    }

    #pragma unroll
    for (int qi = 0; qi < M; ++qi) {
        float* outp = scratch + (((size_t)h * M + qi) * splits + split) * (kHD + 2);
        if (lane == 0) {
            outp[0] = m[qi];
            outp[1] = l[qi];
        }
        #pragma unroll
        for (int i = 0; i < 16; ++i) outp[2 + 16 * lane + i] = acc[qi][i];
    }
}

__global__ void gqa512_stage2_kernel(
    const float* __restrict__ scratch,
    __nv_bfloat16* __restrict__ out,
    int NH,
    int M,
    int splits
) {
    const int row = blockIdx.x;
    const int h = row / M;
    const int qi = row - h * M;
    const float* base = scratch + (size_t)row * splits * (kHD + 2);
    __shared__ float ssc[kMaxSplits];
    __shared__ float sinv;
    if (threadIdx.x == 0) {
        float m_glob = -INFINITY;
        for (int s = 0; s < splits; ++s)
            m_glob = fmaxf(m_glob, base[(size_t)s * (kHD + 2)]);
        float l_glob = 0.0f;
        for (int s = 0; s < splits; ++s) {
            const float* part = base + (size_t)s * (kHD + 2);
            float sc = (part[0] > -INFINITY) ? __expf(part[0] - m_glob) : 0.0f;
            ssc[s] = sc;
            l_glob += part[1] * sc;
        }
        sinv = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < kHD; d += blockDim.x) {
        float a = 0.0f;
        for (int s = 0; s < splits; ++s)
            a += base[(size_t)s * (kHD + 2) + 2 + d] * ssc[s];
        out[((size_t)qi * NH + h) * kHD + d] = __float2bfloat16(a * sinv);
    }
}

}

extern "C" int nv_kernels_gqa512_scratch_elems(int NH, int M, int splits) {
    if (splits < 1) splits = 64;
    if (splits > gqa512::kMaxSplits) splits = gqa512::kMaxSplits;
    return NH * M * splits * (gqa512::kHD + 2);
}

extern "C" int nv_kernels_gqa512_verify_bf16(
    void* stream,
    const uint16_t* q,
    const uint16_t* k,
    const uint16_t* v,
    uint16_t* out,
    const int* pos,
    int delta,
    int M,
    float* scratch,
    int NH,
    int NKV,
    int HD,
    int splits
) {
    if (NH <= 0 || NKV <= 0 || M <= 0) return 0;
    if (HD != gqa512::kHD) return -1;
    if (NKV * gqa512::kWarps != NH) return -1;
    if (M > 8) return -1;
    if (splits < 1) splits = 64;
    if (splits > gqa512::kMaxSplits) splits = gqa512::kMaxSplits;
    dim3 grid((unsigned)NKV, (unsigned)splits);
    cudaStream_t cs = (cudaStream_t)stream;
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    const __nv_bfloat16* kb = reinterpret_cast<const __nv_bfloat16*>(k);
    const __nv_bfloat16* vb = reinterpret_cast<const __nv_bfloat16*>(v);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
#define NV_GQA512_CASE(MM) \
    case MM: { \
        constexpr size_t qsm = gqa512::q_smem_bytes(MM); \
        if (qsm > 0) { \
            static const cudaError_t attr_rc = cudaFuncSetAttribute( \
                gqa512::gqa512_stage1_kernel<MM>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)qsm); \
            if (attr_rc != cudaSuccess) return (int)attr_rc; \
        } \
        gqa512::gqa512_stage1_kernel<MM><<<grid, gqa512::kThreads, qsm, cs>>>( \
            qb, kb, vb, scratch, pos, delta, NH, NKV); \
    } break;
    switch (M) {
        NV_GQA512_CASE(1)
        NV_GQA512_CASE(2)
        NV_GQA512_CASE(3)
        NV_GQA512_CASE(4)
        NV_GQA512_CASE(5)
        NV_GQA512_CASE(6)
        NV_GQA512_CASE(7)
        NV_GQA512_CASE(8)
    }
#undef NV_GQA512_CASE
    gqa512::gqa512_stage2_kernel<<<(unsigned)(NH * M), 256, 0, cs>>>(
        scratch, ob, NH, M, splits);
    return (int)cudaGetLastError();
}

extern "C" int nv_kernels_gqa512_verify_fp8(
    void* stream,
    const uint16_t* q,
    const uint8_t* k_fp8,
    const uint8_t* v_fp8,
    const float* k_scale,
    const float* v_scale,
    uint16_t* out,
    const int* pos,
    int delta,
    int M,
    float* scratch,
    int NH,
    int NKV,
    int HD,
    int splits,
    float scaling
) {
    if (NH <= 0 || NKV <= 0 || M <= 0) return 0;
    if (HD != gqa512::kHD) return -1;
    if (NKV * gqa512::kWarps != NH) return -1;
    if (M > 8) return -1;
    if (splits < 1) splits = 64;
    if (splits > gqa512::kMaxSplits) splits = gqa512::kMaxSplits;
    dim3 grid((unsigned)NKV, (unsigned)splits);
    cudaStream_t cs = (cudaStream_t)stream;
    const __nv_bfloat16* qb = reinterpret_cast<const __nv_bfloat16*>(q);
    __nv_bfloat16* ob = reinterpret_cast<__nv_bfloat16*>(out);
#define NV_GQA512_FP8_CASE(MM) \
    case MM: { \
        constexpr size_t qsm = gqa512::q_smem_bytes(MM); \
        if (qsm > 0) { \
            static const cudaError_t attr_rc = cudaFuncSetAttribute( \
                gqa512::gqa512_stage1_fp8_kernel<MM>, \
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)qsm); \
            if (attr_rc != cudaSuccess) return (int)attr_rc; \
        } \
        gqa512::gqa512_stage1_fp8_kernel<MM><<<grid, gqa512::kThreads, qsm, cs>>>( \
            qb, k_fp8, v_fp8, k_scale, v_scale, scratch, pos, delta, NH, NKV, scaling); \
    } break;
    switch (M) {
        NV_GQA512_FP8_CASE(1)
        NV_GQA512_FP8_CASE(2)
        NV_GQA512_FP8_CASE(3)
        NV_GQA512_FP8_CASE(4)
        NV_GQA512_FP8_CASE(5)
        NV_GQA512_FP8_CASE(6)
        NV_GQA512_FP8_CASE(7)
        NV_GQA512_FP8_CASE(8)
    }
#undef NV_GQA512_FP8_CASE
    gqa512::gqa512_stage2_kernel<<<(unsigned)(NH * M), 256, 0, cs>>>(
        scratch, ob, NH, M, splits);
    return (int)cudaGetLastError();
}
