#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include "nv_kernels.h"
#include "nvk_smem_optin.cuh"

namespace q4exp_qsa_tc {

constexpr int HD = 256;
constexpr int GQA_ROWS_PADDED_TO_ONE_M16_TILE = 16;
constexpr int OUT_ROW_HALVES_264_SPREADS_THE_STAGED_OUTPUT_ROWS_OVER_ALL_BANKS = HD + 8;
constexpr int OUT_ROW_HALVES = OUT_ROW_HALVES_264_SPREADS_THE_STAGED_OUTPUT_ROWS_OVER_ALL_BANKS;
constexpr float LOG2E = 1.4426950408889634f;

__device__ __forceinline__ void mma_bf16(float (&c)[4], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ unsigned pack_bf16(float lo, float hi, float& rlo, float& rhi) {
    __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    float2 back = __bfloat1622float2(h);
    rlo = back.x;
    rhi = back.y;
    return *reinterpret_cast<unsigned*>(&h);
}

__device__ __forceinline__ float sigmoid_ref(float x) { return 1.f / (1.f + expf(-x)); }

__device__ __forceinline__ unsigned u4_at(const uint4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

template <int T, int W>
struct Layout {
    static constexpr int NT = T / 8;
    static constexpr size_t sp_bytes = (size_t)2 * W * NT * 32 * 16;
    static constexpr size_t out_bytes = (size_t)GQA_ROWS_PADDED_TO_ONE_M16_TILE * OUT_ROW_HALVES * 2;
    static constexpr size_t fixed = sp_bytes > out_bytes ? sp_bytes : out_bytes;
    static size_t bytes(int idx_stride) { return fixed + (size_t)((idx_stride + 3) & ~3) * 4; }
};

template <int T, int W, int MINB, bool P_SPLIT_HI_LO>
__global__ void __launch_bounds__(W * 32, MINB) rows_kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ gate, const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache, const int32_t* __restrict__ idx, const int32_t* __restrict__ count,
    int nq, int nkv, int idx_stride, float scale_log2, uint16_t* __restrict__ out
) {
    constexpr int NT = T / 8;
    constexpr int KT = T / 16;
    constexpr int DS = HD / W;
    constexpr int KCH = DS / 32;
    constexpr int KK = DS / 16;
    constexpr int VCH = DS / 64;
    constexpr int NJ = DS / 8;
    constexpr int NTHR = W * 32;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    float4* sp = reinterpret_cast<float4*>(smem_raw);
    int* sidx = reinterpret_cast<int*>(smem_raw + Layout<T, W>::fixed);
    const int m = blockIdx.x;
    const int g = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int group = nq / nkv;
    const int n = count[m];
    const int32_t* rowidx = idx + (size_t)m * idx_stride;
    for (int i = tid; i < n; i += NTHR) sidx[i] = rowidx[i];
    const int gr = lane >> 2;
    const int q4 = lane & 3;
    const int dw = warp * DS;
    unsigned qa[KK][4];
    {
        const uint16_t* qb = q + ((size_t)m * nq + g * group) * HD;
        const bool r0 = gr < group;
        const bool r1 = gr + 8 < group;
#pragma unroll
        for (int kk = 0; kk < KK; ++kk) {
            const int d = dw + (kk >> 1) * 32 + 8 * q4 + (kk & 1) * 4;
            qa[kk][0] = r0 ? *reinterpret_cast<const unsigned*>(qb + (size_t)gr * HD + d) : 0u;
            qa[kk][1] = r1 ? *reinterpret_cast<const unsigned*>(qb + (size_t)(gr + 8) * HD + d) : 0u;
            qa[kk][2] = r0 ? *reinterpret_cast<const unsigned*>(qb + (size_t)gr * HD + d + 2) : 0u;
            qa[kk][3] = r1 ? *reinterpret_cast<const unsigned*>(qb + (size_t)(gr + 8) * HD + d + 2) : 0u;
        }
    }
    __syncthreads();
    const int ntiles = (n + T - 1) / T;
    uint4 kb[NT][KCH];
    uint4 vb[KT][4][VCH];
    auto load_k = [&](int t0) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int t = min(t0 + 8 * j + gr, n - 1);
            const uint16_t* base = k_cache + ((size_t)sidx[t] * nkv + g) * HD + dw + 8 * q4;
#pragma unroll
            for (int c = 0; c < KCH; ++c) kb[j][c] = __ldg(reinterpret_cast<const uint4*>(base + 32 * c));
        }
    };
    auto load_v = [&](int t0) {
#pragma unroll
        for (int kk = 0; kk < KT; ++kk)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int t = min(t0 + 16 * kk + 2 * q4 + (e & 1) + (e >> 1) * 8, n - 1);
                const uint16_t* base = v_cache + ((size_t)sidx[t] * nkv + g) * HD + dw + (DS / 8) * gr;
#pragma unroll
                for (int c = 0; c < VCH; ++c) vb[kk][e][c] = __ldg(reinterpret_cast<const uint4*>(base + 8 * c));
            }
    };
    float o[NJ][4];
#pragma unroll
    for (int j = 0; j < NJ; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[j][e] = 0.f;
    float mrow[2] = {-INFINITY, -INFINITY};
    float lrow[2] = {0.f, 0.f};
    if (ntiles > 0) {
        load_k(0);
        load_v(0);
    }
    for (int tile = 0; tile < ntiles; ++tile) {
        float4* spb = sp + (size_t)(tile & 1) * W * NT * 32;
        float s[NT][4];
#pragma unroll
        for (int j = 0; j < NT; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) s[j][e] = 0.f;
#pragma unroll
            for (int kk = 0; kk < KK; ++kk)
                mma_bf16(s[j], qa[kk], u4_at(kb[j][kk >> 1], (kk & 1) * 2), u4_at(kb[j][kk >> 1], (kk & 1) * 2 + 1));
        }
        if (tile + 1 < ntiles) load_k((tile + 1) * T);
#pragma unroll
        for (int j = 0; j < NT; ++j) spb[(warp * NT + j) * 32 + lane] = make_float4(s[j][0], s[j][1], s[j][2], s[j][3]);
        __syncthreads();
        const int t0 = tile * T;
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
            for (int w = 0; w < W; ++w) {
                const float4 x = spb[(w * NT + j) * 32 + lane];
                a.x += x.x; a.y += x.y; a.z += x.z; a.w += x.w;
            }
            const int t = t0 + j * 8 + 2 * q4;
            const bool v0 = t < n;
            const bool v1 = t + 1 < n;
            s[j][0] = v0 ? a.x * scale_log2 : -INFINITY;
            s[j][1] = v1 ? a.y * scale_log2 : -INFINITY;
            s[j][2] = v0 ? a.z * scale_log2 : -INFINITY;
            s[j][3] = v1 ? a.w * scale_log2 : -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]));
            mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(mrow[0], mx0);
        const float mn1 = fmaxf(mrow[1], mx1);
        if (__any_sync(0xffffffffu, mn0 > mrow[0] || mn1 > mrow[1])) {
            const float c0 = exp2f(mrow[0] - mn0);
            const float c1 = exp2f(mrow[1] - mn1);
            lrow[0] *= c0;
            lrow[1] *= c1;
#pragma unroll
            for (int j = 0; j < NJ; ++j) {
                o[j][0] *= c0; o[j][1] *= c0; o[j][2] *= c1; o[j][3] *= c1;
            }
            mrow[0] = mn0;
            mrow[1] = mn1;
        }
        unsigned pa[KT][4];
        unsigned pl[KT][4];
#pragma unroll
        for (int kk = 0; kk < KT; ++kk)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int j = 2 * kk + h;
                const float e0 = exp2f(s[j][0] - mn0), e1 = exp2f(s[j][1] - mn0);
                const float e2 = exp2f(s[j][2] - mn1), e3 = exp2f(s[j][3] - mn1);
                float r0, r1, r2, r3;
                pa[kk][2 * h] = pack_bf16(e0, e1, r0, r1);
                pa[kk][2 * h + 1] = pack_bf16(e2, e3, r2, r3);
                if (P_SPLIT_HI_LO) {
                    float d0, d1, d2, d3;
                    pl[kk][2 * h] = pack_bf16(e0 - r0, e1 - r1, d0, d1);
                    pl[kk][2 * h + 1] = pack_bf16(e2 - r2, e3 - r3, d2, d3);
                    lrow[0] += e0 + e1;
                    lrow[1] += e2 + e3;
                } else {
                    lrow[0] += r0 + r1;
                    lrow[1] += r2 + r3;
                }
            }
#pragma unroll
        for (int kk = 0; kk < KT; ++kk)
#pragma unroll
            for (int nj = 0; nj < NJ; ++nj) {
                const int c = nj >> 3;
                const int r = (nj & 7) >> 1;
                const unsigned sel = (nj & 1) ? 0x7632u : 0x5410u;
                const unsigned b0 = __byte_perm(u4_at(vb[kk][0][c], r), u4_at(vb[kk][1][c], r), sel);
                const unsigned b1 = __byte_perm(u4_at(vb[kk][2][c], r), u4_at(vb[kk][3][c], r), sel);
                mma_bf16(o[nj], pa[kk], b0, b1);
                if (P_SPLIT_HI_LO) mma_bf16(o[nj], pl[kk], b0, b1);
            }
        if (tile + 1 < ntiles) load_v((tile + 1) * T);
    }
    lrow[0] += __shfl_xor_sync(0xffffffffu, lrow[0], 1);
    lrow[0] += __shfl_xor_sync(0xffffffffu, lrow[0], 2);
    lrow[1] += __shfl_xor_sync(0xffffffffu, lrow[1], 1);
    lrow[1] += __shfl_xor_sync(0xffffffffu, lrow[1], 2);
    __syncthreads();
    uint16_t* so = reinterpret_cast<uint16_t*>(smem_raw);
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float inv = lrow[h] > 0.f ? 1.f / lrow[h] : 0.f;
        const int row = gr + 8 * h;
#pragma unroll
        for (int nj = 0; nj < NJ; ++nj)
#pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int d = dw + (DS / 8) * (2 * q4 + e) + nj;
                *reinterpret_cast<__nv_bfloat16*>(&so[row * OUT_ROW_HALVES + d]) = __float2bfloat16(o[nj][2 * h + e] * inv);
            }
    }
    __syncthreads();
    for (int i = tid; i < group * (HD / 2); i += NTHR) {
        const int row = i / (HD / 2);
        const int d = (i % (HD / 2)) * 2;
        float2 x = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&so[row * OUT_ROW_HALVES + d]));
        const size_t base = ((size_t)m * nq + g * group + row) * HD + d;
        if (gate) {
            float2 gv = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(gate + base));
            x.x *= sigmoid_ref(gv.x);
            x.y *= sigmoid_ref(gv.y);
        }
        *reinterpret_cast<__nv_bfloat162*>(out + base) = __floats2bfloat162_rn(x.x, x.y);
    }
}

template <int T, int W, int MINB, bool P_SPLIT_HI_LO>
int launch_rows(cudaStream_t stream, const uint16_t* q, const uint16_t* gate, const uint16_t* k_cache,
                const uint16_t* v_cache, const int32_t* idx, const int32_t* count, int m, int nq, int nkv,
                int idx_stride, float scale, uint16_t* out) {
    static DynamicSmemOptin optin;
    const size_t smem = Layout<T, W>::bytes(idx_stride);
    if ((int)smem > nvk_max_dynamic_smem_optin()) return -2;
    if (raise_dynamic_smem_optin_never_lowering_it(optin, (const void*)rows_kernel<T, W, MINB, P_SPLIT_HI_LO>, smem) != 0)
        return -3;
    dim3 grid(m, nkv);
    rows_kernel<T, W, MINB, P_SPLIT_HI_LO><<<grid, W * 32, smem, stream>>>(q, gate, k_cache, v_cache, idx, count, nq,
                                                                         nkv, idx_stride, scale * LOG2E, out);
    return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

}

extern "C" int nv_kernels_q4exp_qsa_tc_rows(void* stream, const uint16_t* q, const uint16_t* gate,
                                             const uint16_t* k_cache, const uint16_t* v_cache, const int32_t* idx,
                                             const int32_t* count, int m, int nq, int nkv, int hd, int idx_stride,
                                             float scale, uint16_t* out) {
    if (m <= 0) return 0;
    if (hd != q4exp_qsa_tc::HD || nkv <= 0 || nq % nkv != 0 || nq / nkv > q4exp_qsa_tc::GQA_ROWS_PADDED_TO_ONE_M16_TILE || !idx)
        return -1;
    return q4exp_qsa_tc::launch_rows<16, 2, 4, true>((cudaStream_t)stream, q, gate, k_cache, v_cache, idx, count, m,
                                                       nq, nkv, idx_stride, scale, out);
}
