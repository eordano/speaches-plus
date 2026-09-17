#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <math.h>

namespace gqa4 {

constexpr int kWarp = 32;
constexpr int kWarps = 8;
constexpr int kThreads = kWarp * kWarps;
constexpr int kHD = 512;
constexpr int kGroup = 4;
constexpr int kPerLane = kHD / kWarp;
constexpr int kMaxSplits256KeepsTheCombineTableInSharedMemory = 256;
constexpr int kBlocksPerSmTarget2MatchesTheStockSplitPicker = 2;

__inline__ __device__ float warp_sum(float x) {
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

__device__ __forceinline__ void unpack8(const uint4& raw, float* f) {
    const __nv_bfloat162* b = reinterpret_cast<const __nv_bfloat162*>(&raw);
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        float2 x = __bfloat1622float2(b[t]);
        f[2 * t] = x.x;
        f[2 * t + 1] = x.y;
    }
}

__global__ void __launch_bounds__(kThreads) flash_decode_fused_gqa4_hd512_kernel(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    float* __restrict__ scratch,
    unsigned int* __restrict__ fan_in,
    __nv_bfloat16* __restrict__ outp,
    const int* __restrict__ pos,
    int delta,
    int NKV,
    int WINDOW) {
    const int kvh = blockIdx.x;
    const int split = blockIdx.y;
    const int splits = gridDim.y;
    const int total = pos[0] - delta;
    const int start = (WINDOW > 0 && total > WINDOW) ? (total - WINDOW) : 0;
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;

    __shared__ float qsh[kGroup][kHD];
    for (int i = threadIdx.x; i < kGroup * kHD; i += kThreads)
        qsh[i / kHD][i % kHD] = q[((size_t)kvh * kGroup) * kHD + i];
    __syncthreads();

    float qreg[kGroup][kPerLane];
    #pragma unroll
    for (int g = 0; g < kGroup; ++g) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            qreg[g][e] = qsh[g][lane * 8 + e];
            qreg[g][8 + e] = qsh[g][kHD / 2 + lane * 8 + e];
        }
    }

    float acc[kGroup][kPerLane];
    float m[kGroup], l[kGroup];
    #pragma unroll
    for (int g = 0; g < kGroup; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.0f;
        #pragma unroll
        for (int i = 0; i < kPerLane; ++i) acc[g][i] = 0.0f;
    }

    const int stride = splits * kWarps;
    int p = start + split * kWarps + warp;
    uint4 k0, k1, v0, v1;
    if (p < total) {
        const uint4* kr = reinterpret_cast<const uint4*>(k + ((size_t)p * NKV + kvh) * kHD);
        const uint4* vr = reinterpret_cast<const uint4*>(v + ((size_t)p * NKV + kvh) * kHD);
        k0 = __ldg(kr + lane);
        k1 = __ldg(kr + kWarp + lane);
        v0 = __ldg(vr + lane);
        v1 = __ldg(vr + kWarp + lane);
    }
    for (; p < total; p += stride) {
        const int pn = p + stride;
        uint4 kn0, kn1, vn0, vn1;
        if (pn < total) {
            const uint4* kr = reinterpret_cast<const uint4*>(k + ((size_t)pn * NKV + kvh) * kHD);
            const uint4* vr = reinterpret_cast<const uint4*>(v + ((size_t)pn * NKV + kvh) * kHD);
            kn0 = __ldg(kr + lane);
            kn1 = __ldg(kr + kWarp + lane);
            vn0 = __ldg(vr + lane);
            vn1 = __ldg(vr + kWarp + lane);
        }
        float kf[kPerLane], vf[kPerLane];
        unpack8(k0, kf);
        unpack8(k1, kf + 8);
        unpack8(v0, vf);
        unpack8(v1, vf + 8);
        float partial[kGroup];
        #pragma unroll
        for (int g = 0; g < kGroup; ++g) {
            float s = 0.0f;
            #pragma unroll
            for (int i = 0; i < kPerLane; ++i) s += qreg[g][i] * kf[i];
            partial[g] = s;
        }
        #pragma unroll
        for (int g = 0; g < kGroup; ++g) {
            const float score = warp_sum(partial[g]);
            const float m_new = fmaxf(m[g], score);
            const float corr = __expf(m[g] - m_new);
            const float w = __expf(score - m_new);
            l[g] = l[g] * corr + w;
            #pragma unroll
            for (int i = 0; i < kPerLane; ++i) acc[g][i] = acc[g][i] * corr + w * vf[i];
            m[g] = m_new;
        }
        if (pn < total) {
            k0 = kn0;
            k1 = kn1;
            v0 = vn0;
            v1 = vn1;
        }
    }

    __shared__ float sm[kWarps];
    __shared__ float sl[kWarps];
    __shared__ float sacc[kWarps][kHD];
    for (int g = 0; g < kGroup; ++g) {
        if (lane == 0) {
            sm[warp] = m[g];
            sl[warp] = l[g];
        }
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            sacc[warp][lane * 8 + e] = acc[g][e];
            sacc[warp][kHD / 2 + lane * 8 + e] = acc[g][8 + e];
        }
        __syncthreads();
        const int h = kvh * kGroup + g;
        float* out = scratch + ((size_t)h * splits + split) * (kHD + 2);
        float m_blk = -INFINITY;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) m_blk = fmaxf(m_blk, sm[w]);
        if (threadIdx.x == 0) {
            float l_blk = 0.0f;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) l_blk += (sm[w] > -INFINITY) ? sl[w] * __expf(sm[w] - m_blk) : 0.0f;
            out[0] = m_blk;
            out[1] = l_blk;
        }
        for (int d = threadIdx.x; d < kHD; d += kThreads) {
            float a = 0.0f;
            #pragma unroll
            for (int w = 0; w < kWarps; ++w) a += (sm[w] > -INFINITY) ? sacc[w][d] * __expf(sm[w] - m_blk) : 0.0f;
            out[2 + d] = a;
        }
        __syncthreads();
    }

}

__global__ void __launch_bounds__(kThreads) gqa4_hd512_combine_kernel(
    const float* __restrict__ scratch,
    __nv_bfloat16* __restrict__ outp,
    int splits) {
    const int h = blockIdx.x;
    const float* base = scratch + (size_t)h * splits * (kHD + 2);
    __shared__ float ssc[kMaxSplits256KeepsTheCombineTableInSharedMemory];
    __shared__ float red[kWarps];
    const int lane = threadIdx.x & (kWarp - 1);
    const int warp = threadIdx.x >> 5;
    float pm = (threadIdx.x < splits) ? base[(size_t)threadIdx.x * (kHD + 2)] : -INFINITY;
    float mx = pm;
    #pragma unroll
    for (int o = kWarp / 2; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    float m_glob = -INFINITY;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) m_glob = fmaxf(m_glob, red[w]);
    __syncthreads();
    float pl = 0.0f;
    if (threadIdx.x < splits) {
        const float sc = (pm > -INFINITY) ? __expf(pm - m_glob) : 0.0f;
        ssc[threadIdx.x] = sc;
        pl = base[(size_t)threadIdx.x * (kHD + 2) + 1] * sc;
    }
    pl = warp_sum(pl);
    if (lane == 0) red[warp] = pl;
    __syncthreads();
    float l_glob = 0.0f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) l_glob += red[w];
    const float inv_l = (l_glob > 0.0f) ? (1.0f / l_glob) : 0.0f;
    for (int d = threadIdx.x; d < kHD; d += kThreads) {
        float a = 0.0f;
        for (int s = 0; s < splits; ++s) a += base[(size_t)s * (kHD + 2) + 2 + d] * ssc[s];
        outp[(size_t)h * kHD + d] = __float2bfloat16(a * inv_l);
    }
}

inline int sm_count() {
    static int cached = 0;
    if (cached == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&cached, cudaDevAttrMultiProcessorCount, dev);
        if (cached <= 0) cached = 1;
    }
    return cached;
}

constexpr int kKeysPerBlockFloor256CapsTheSplitCountForShallowCapacitiesSoTheFixedPerBlockEpilogueDoesNotDominate2K = 256;

inline int splits_for(int NKV, int capacity) {
    int s = (kBlocksPerSmTarget2MatchesTheStockSplitPicker * sm_count() + NKV - 1) / NKV;
    if (capacity > 0) {
        const int by_capacity =
            (capacity + kKeysPerBlockFloor256CapsTheSplitCountForShallowCapacitiesSoTheFixedPerBlockEpilogueDoesNotDominate2K - 1) /
            kKeysPerBlockFloor256CapsTheSplitCountForShallowCapacitiesSoTheFixedPerBlockEpilogueDoesNotDominate2K;
        if (by_capacity < s) s = by_capacity;
    }
    if (s < 1) s = 1;
    if (s > kMaxSplits256KeepsTheCombineTableInSharedMemory) s = kMaxSplits256KeepsTheCombineTableInSharedMemory;
    return s;
}

}

extern "C" int nv_kernels_flash_decode_gqa4_hd512_scratch_elems(int NH, int capacity) {
    if (NH <= 0 || (NH % gqa4::kGroup) != 0) return 0;
    return NH * gqa4::splits_for(NH / gqa4::kGroup, capacity) * (gqa4::kHD + 2);
}

extern "C" int nv_kernels_flash_decode_fused_gqa4_hd512(
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
    int WINDOW,
    int capacity) {
    if (NH <= 0 || NKV <= 0) return -1;
    if (HD != gqa4::kHD || NH != NKV * gqa4::kGroup) return -2;
    const int sp = gqa4::splits_for(NKV, capacity);
    dim3 grid((unsigned)NKV, (unsigned)sp);
    gqa4::flash_decode_fused_gqa4_hd512_kernel<<<grid, gqa4::kThreads, 0, (cudaStream_t)stream>>>(
        q, reinterpret_cast<const __nv_bfloat16*>(k), reinterpret_cast<const __nv_bfloat16*>(v), scratch, fan_in,
        reinterpret_cast<__nv_bfloat16*>(out), pos, delta, NKV, WINDOW);
    gqa4::gqa4_hd512_combine_kernel<<<(unsigned)NH, gqa4::kThreads, 0, (cudaStream_t)stream>>>(
        scratch, reinterpret_cast<__nv_bfloat16*>(out), sp);
    return (int)cudaGetLastError();
}
