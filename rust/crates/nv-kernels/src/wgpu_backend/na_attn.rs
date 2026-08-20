use std::sync::{Arc, Mutex, OnceLock};

use super::device::WgpuContext;
use super::dispatch::profile;
use super::na;
use super::na::{storage_entry, uniform_entry};
use super::{Result, WgpuError};

pub const TILE_M: u32 = 16;
pub const TILE_N: u32 = 32;
pub const HEAD_DIM: u32 = 256;
pub const HEAD_DIM_G: u32 = 512;
pub const WG_THREADS: u32 = 128;

pub const ENTRY: &str = "na_attn_prefill";
pub const ENTRY_G: &str = "na_attn_prefill_g";

pub const MSL: &str = r#"
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;

struct NaFdParams {
    uint n_heads;
    uint n_kv;
    uint head_dim;
    uint total;
    uint start;
    uint splits;
    uint ring;
    uint out_bf16;
    float scaling;
    uint pad0;
    uint fused;
    uint pad2;
    uint m_rows;
    uint window;
    uint pad3;
    uint pad4;
};

constexpr constant int ATM = 16;
constexpr constant int ATN = 32;
constexpr constant int ATHD = 256;
constexpr constant int ATHQ = 64;
constexpr constant int ATHW = ATHD / 2;
constexpr constant float AT_LOG2E = 1.4426950408889634f;

static inline uint na_attn_bf16_encode(float x) {
    uint b = as_type<uint>(x);
    uint r = 0x7fffu + ((b >> 16u) & 1u);
    return (x != x) ? 0x7fc0u : ((b + r) >> 16u);
}

static inline float na_attn_e4m3(uint b) {
    if ((b & 127u) == 127u) {
        return as_type<float>(0x7fc00000u);
    }
    uint e = (b >> 3u) & 15u;
    uint m = b & 7u;
    float mag = (e == 0u)
        ? float(m) * 0.001953125f
        : as_type<float>(((e + 120u) << 23u) | (m << 20u));
    return ((b & 128u) != 0u) ? -mag : mag;
}

static inline float na_attn_neg() {
    return as_type<float>(0xff800000u);
}

kernel void na_attn_prefill(
    device float* q [[buffer(0)]],
    device uint* kw [[buffer(1)]],
    device uint* vw [[buffer(2)]],
    device float* ksc [[buffer(3)]],
    device float* vsc [[buffer(4)]],
    device uint* yout [[buffer(5)]],
    constant NaFdParams& fd [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup bfloat qt[ATM * ATHD];
    threadgroup float shf[ATM * ATHD];
    threadgroup float sct[ATM * ATN];
    threadgroup float pt[ATM * ATN];
    threadgroup float kst[ATN];
    threadgroup float vst[ATN];
    threadgroup float row_m[ATM];
    threadgroup float row_l[ATM];
    threadgroup float row_c[ATM];
    threadgroup float row_inv[ATM];

    uint h = tgid.x;
    if (h >= fd.n_heads || fd.head_dim != uint(ATHD)) {
        return;
    }
    uint mr = fd.m_rows;
    uint qt0 = tgid.y * uint(ATM);
    if (qt0 >= mr) {
        return;
    }
    uint rows = min(uint(ATM), mr - qt0);
    uint nkv = fd.n_kv;
    uint kvh = h / (fd.n_heads / nkv);

    for (uint i = lid; i < uint(ATM * ATHD); i += 128u) {
        uint r = i >> 8u;
        uint d = i & 255u;
        qt[i] = (r < rows)
            ? bfloat(q[((qt0 + r) * fd.n_heads + h) * uint(ATHD) + d])
            : bfloat(0.0f);
    }
    if (lid < uint(ATM)) {
        row_m[lid] = na_attn_neg();
        row_l[lid] = 0.0f;
        row_c[lid] = 1.0f;
    }

    uint total_hi = fd.total - (mr - 1u - (qt0 + rows - 1u));
    uint total_lo = fd.total - (mr - 1u - qt0);
    uint st_lo = (fd.window > 0u && total_lo > fd.window) ? total_lo - fd.window : 0u;
    uint c0 = (st_lo / uint(ATN)) * uint(ATN);

    auto tQ = tensor(qt, dextents<int32_t, 2>(ATHD, ATM), array<int, 2>{1, ATHD});
    threadgroup bfloat* kt = reinterpret_cast<threadgroup bfloat*>(shf);
    auto tK = tensor(kt, dextents<int32_t, 2>(ATHD, ATN), array<int, 2>{1, ATHD});
    threadgroup half* vt = reinterpret_cast<threadgroup half*>(shf);
    auto tV = tensor(vt, dextents<int32_t, 2>(ATHD, ATN), array<int, 2>{1, ATHD});
    auto tP = tensor(pt, dextents<int32_t, 2>(ATN, ATM), array<int, 2>{1, ATN});

    constexpr auto dqk = matmul2d_descriptor(ATM, ATN, ATHQ, false, true, false,
                                             matmul2d_descriptor::mode::multiply);
    matmul2d<dqk, execution_simdgroup> opqk;
    constexpr auto dpv = matmul2d_descriptor(ATM, ATHQ, ATN, false, false, false,
                                             matmul2d_descriptor::mode::multiply);
    matmul2d<dpv, execution_simdgroup> oppv;

    auto sQ0 = tQ.slice(int(sgid) * ATHQ, 0);
    auto sK0 = tK.slice(int(sgid) * ATHQ, 0);
    auto sP0 = tP.slice(0, 0);
    auto sV0 = tV.slice(int(sgid) * ATHQ, 0);

    auto cqk = opqk.get_destination_cooperative_tensor<decltype(sQ0), decltype(sK0), float>();
    auto cpv = oppv.get_destination_cooperative_tensor<decltype(sP0), decltype(sV0), float>();

    constexpr int CAPQK = (ATM * ATN) / 32;
    constexpr int CAPPV = (ATM * ATHQ) / 32;
    uint16_t nq[CAPQK];
    uint16_t mq[CAPQK];
    #pragma unroll
    for (uint16_t i = 0; i < cqk.get_capacity(); ++i) {
        auto idx = cqk.get_multidimensional_index(i);
        nq[i] = cqk.is_valid_element(i) ? uint16_t(idx[0]) : uint16_t(0);
        mq[i] = cqk.is_valid_element(i) ? uint16_t(idx[1]) : uint16_t(0);
    }
    float acc[CAPPV];
    uint16_t np[CAPPV];
    uint16_t mp[CAPPV];
    bool vp[CAPPV];
    #pragma unroll
    for (uint16_t i = 0; i < cpv.get_capacity(); ++i) {
        acc[i] = 0.0f;
        auto idx = cpv.get_multidimensional_index(i);
        vp[i] = cpv.is_valid_element(i);
        np[i] = vp[i] ? uint16_t(idx[0]) : uint16_t(0);
        mp[i] = vp[i] ? uint16_t(idx[1]) : uint16_t(0);
    }

    for (uint cb = c0; cb < total_hi; cb += uint(ATN)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < uint(ATN * ATHD); i += 128u) {
            uint n = i >> 8u;
            uint d = i & 255u;
            uint p = cb + n;
            float v = 0.0f;
            if (p < fd.total) {
                uint sp = (fd.ring > 0u) ? (p % fd.ring) : p;
                uint idx = (sp * nkv + kvh) * uint(ATHD) + d;
                v = na_attn_e4m3((kw[idx >> 2u] >> (8u * (idx & 3u))) & 255u);
            }
            kt[i] = bfloat(v);
        }
        if (lid < uint(ATN)) {
            uint p = cb + lid;
            float ks = 0.0f;
            float vs = 0.0f;
            if (p < fd.total) {
                uint sp = (fd.ring > 0u) ? (p % fd.ring) : p;
                ks = ksc[sp * nkv + kvh];
                vs = vsc[sp * nkv + kvh];
            }
            kst[lid] = ks;
            vst[lid] = vs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sQ = tQ.slice(int(sgid) * ATHQ, 0);
        auto sK = tK.slice(int(sgid) * ATHQ, 0);
        opqk.run(sQ, sK, cqk);

        for (uint s = 0u; s < 4u; s++) {
            if (sgid == s) {
                #pragma unroll
                for (uint16_t i = 0; i < uint16_t(CAPQK); ++i) {
                    if (cqk.is_valid_element(i)) {
                        if (s == 0u) {
                            sct[uint(mq[i]) * uint(ATN) + uint(nq[i])] = cqk[i];
                        } else {
                            sct[uint(mq[i]) * uint(ATN) + uint(nq[i])] += cqk[i];
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid < uint(ATM)) {
            uint r = lid;
            if (r < rows) {
                uint qi = qt0 + r;
                uint tr = fd.total - (mr - 1u - qi);
                uint st = (fd.window > 0u && tr > fd.window) ? tr - fd.window : 0u;
                float neg = na_attn_neg();
                float mx = row_m[r];
                float cmx = neg;
                float sv[ATN];
                bool lv[ATN];
                for (uint n = 0u; n < uint(ATN); n++) {
                    uint p = cb + n;
                    bool live = p >= st && p < tr;
                    lv[n] = live;
                    float v = live ? (sct[r * uint(ATN) + n] * kst[n]) * fd.scaling : 0.0f;
                    sv[n] = v;
                    if (live && v > cmx) {
                        cmx = v;
                    }
                }
                float mnew = max(mx, cmx);
                float corr = 1.0f;
                float l = row_l[r];
                if (mnew > neg) {
                    if (mx > neg) {
                        corr = exp2((mx - mnew) * AT_LOG2E);
                    }
                    l = l * corr;
                    for (uint n = 0u; n < uint(ATN); n++) {
                        float w = lv[n] ? exp2((sv[n] - mnew) * AT_LOG2E) : 0.0f;
                        l += w;
                        pt[r * uint(ATN) + n] = w * vst[n];
                    }
                    row_m[r] = mnew;
                    row_l[r] = l;
                } else {
                    for (uint n = 0u; n < uint(ATN); n++) {
                        pt[r * uint(ATN) + n] = 0.0f;
                    }
                }
                row_c[r] = corr;
            } else {
                for (uint n = 0u; n < uint(ATN); n++) {
                    pt[r * uint(ATN) + n] = 0.0f;
                }
                row_c[r] = 1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lid; i < uint(ATN * ATHD); i += 128u) {
            uint n = i >> 8u;
            uint d = i & 255u;
            uint p = cb + n;
            float v = 0.0f;
            if (p < fd.total) {
                uint sp = (fd.ring > 0u) ? (p % fd.ring) : p;
                uint idx = (sp * nkv + kvh) * uint(ATHD) + d;
                v = na_attn_e4m3((vw[idx >> 2u] >> (8u * (idx & 3u))) & 255u);
            }
            vt[i] = half(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sP = tP.slice(0, 0);
        auto sV = tV.slice(int(sgid) * ATHQ, 0);
        oppv.run(sP, sV, cpv);
        #pragma unroll
        for (uint16_t i = 0; i < uint16_t(CAPPV); ++i) {
            acc[i] = acc[i] * row_c[mp[i]] + cpv[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < rows) {
        float l = row_l[lid];
        float inv = 0.0f;
        if (l > 0.0f) {
            float rr = 1.0f / l;
            inv = fma(fma(-l, rr, 1.0f), rr, rr);
        }
        row_inv[lid] = inv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll
    for (uint16_t i = 0; i < uint16_t(CAPPV); ++i) {
        if (vp[i]) {
            shf[uint(mp[i]) * uint(ATHD) + sgid * uint(ATHQ) + uint(np[i])] =
                acc[i] * row_inv[mp[i]];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < rows * uint(ATHW); i += 128u) {
        uint r = i / uint(ATHW);
        uint w = i % uint(ATHW);
        uint lo = na_attn_bf16_encode(shf[r * uint(ATHD) + 2u * w]) & 0xffffu;
        uint hi = na_attn_bf16_encode(shf[r * uint(ATHD) + 2u * w + 1u]) & 0xffffu;
        yout[((qt0 + r) * fd.n_heads + h) * uint(ATHW) + w] = lo | (hi << 16u);
    }
}

constexpr constant int GTM = 16;
constexpr constant int GTN = 16;
constexpr constant int GHD = 512;
constexpr constant int GHQ = 128;

kernel void na_attn_prefill_g(
    device float* q [[buffer(0)]],
    device uint* kw [[buffer(1)]],
    device uint* vw [[buffer(2)]],
    device float* ksc [[buffer(3)]],
    device float* vsc [[buffer(4)]],
    device uint* yout [[buffer(5)]],
    constant NaFdParams& fd [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint sgid [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint lid [[thread_index_in_threadgroup]])
{
    threadgroup float shf[GTN * GHD / 2];
    threadgroup float sct[GTM * GTN];
    threadgroup float pt[GTM * GTN];
    threadgroup float kst[GTN];
    threadgroup float vst[GTN];
    threadgroup float row_m[GTM];
    threadgroup float row_l[GTM];
    threadgroup float row_c[GTM];
    threadgroup float row_inv[GTM];

    uint h = tgid.x;
    if (h >= fd.n_heads || fd.head_dim != uint(GHD)) {
        return;
    }
    uint mr = fd.m_rows;
    uint qt0 = tgid.y * uint(GTM);
    if (qt0 >= mr) {
        return;
    }
    uint rows = min(uint(GTM), mr - qt0);
    uint nkv = fd.n_kv;
    uint kvh = h / (fd.n_heads / nkv);

    if (lid < uint(GTM)) {
        row_m[lid] = na_attn_neg();
        row_l[lid] = 0.0f;
        row_c[lid] = 1.0f;
    }

    uint total_hi = fd.total - (mr - 1u - (qt0 + rows - 1u));
    uint total_lo = fd.total - (mr - 1u - qt0);
    uint st_lo = (fd.window > 0u && total_lo > fd.window) ? total_lo - fd.window : 0u;
    uint c0 = (st_lo / uint(GTN)) * uint(GTN);

    device float* qh = q + (qt0 * fd.n_heads + h) * uint(GHD);
    auto tQ = tensor(qh, dextents<int32_t, 2>(GHD, int(rows)),
                     array<int, 2>{1, int(fd.n_heads) * GHD});
    threadgroup bfloat* kt = reinterpret_cast<threadgroup bfloat*>(shf);
    auto tK = tensor(kt, dextents<int32_t, 2>(GHD, GTN), array<int, 2>{1, GHD});
    threadgroup half* vt = reinterpret_cast<threadgroup half*>(shf);
    auto tV = tensor(vt, dextents<int32_t, 2>(GHD, GTN), array<int, 2>{1, GHD});
    auto tP = tensor(pt, dextents<int32_t, 2>(GTN, GTM), array<int, 2>{1, GTN});

    constexpr auto gqk = matmul2d_descriptor(GTM, GTN, GHQ, false, true, false,
                                             matmul2d_descriptor::mode::multiply);
    matmul2d<gqk, execution_simdgroup> opqk;
    constexpr auto gpv = matmul2d_descriptor(GTM, GHQ, GTN, false, false, false,
                                             matmul2d_descriptor::mode::multiply);
    matmul2d<gpv, execution_simdgroup> oppv;

    auto sQ0 = tQ.slice(int(sgid) * GHQ, 0);
    auto sK0 = tK.slice(int(sgid) * GHQ, 0);
    auto sP0 = tP.slice(0, 0);
    auto sV0 = tV.slice(int(sgid) * GHQ, 0);

    auto cqk = opqk.get_destination_cooperative_tensor<decltype(sQ0), decltype(sK0), float>();
    auto cpv = oppv.get_destination_cooperative_tensor<decltype(sP0), decltype(sV0), float>();

    constexpr int GCAPQK = (GTM * GTN) / 32;
    constexpr int GCAPPV = (GTM * GHQ) / 32;
    uint16_t nq[GCAPQK];
    uint16_t mq[GCAPQK];
    #pragma unroll
    for (uint16_t i = 0; i < cqk.get_capacity(); ++i) {
        auto idx = cqk.get_multidimensional_index(i);
        nq[i] = cqk.is_valid_element(i) ? uint16_t(idx[0]) : uint16_t(0);
        mq[i] = cqk.is_valid_element(i) ? uint16_t(idx[1]) : uint16_t(0);
    }
    float acc[GCAPPV];
    uint16_t np[GCAPPV];
    uint16_t mp[GCAPPV];
    bool vp[GCAPPV];
    #pragma unroll
    for (uint16_t i = 0; i < cpv.get_capacity(); ++i) {
        acc[i] = 0.0f;
        auto idx = cpv.get_multidimensional_index(i);
        vp[i] = cpv.is_valid_element(i);
        np[i] = vp[i] ? uint16_t(idx[0]) : uint16_t(0);
        mp[i] = vp[i] ? uint16_t(idx[1]) : uint16_t(0);
    }

    for (uint cb = c0; cb < total_hi; cb += uint(GTN)) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < uint(GTN * GHD); i += 128u) {
            uint n = i >> 9u;
            uint d = i & 511u;
            uint p = cb + n;
            float v = 0.0f;
            if (p < fd.total) {
                uint sp = (fd.ring > 0u) ? (p % fd.ring) : p;
                uint idx = (sp * nkv + kvh) * uint(GHD) + d;
                v = na_attn_e4m3((kw[idx >> 2u] >> (8u * (idx & 3u))) & 255u);
            }
            kt[i] = bfloat(v);
        }
        if (lid < uint(GTN)) {
            uint p = cb + lid;
            float ks = 0.0f;
            float vs = 0.0f;
            if (p < fd.total) {
                uint sp = (fd.ring > 0u) ? (p % fd.ring) : p;
                ks = ksc[sp * nkv + kvh];
                vs = vsc[sp * nkv + kvh];
            }
            kst[lid] = ks;
            vst[lid] = vs;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sQ = tQ.slice(int(sgid) * GHQ, 0);
        auto sK = tK.slice(int(sgid) * GHQ, 0);
        opqk.run(sQ, sK, cqk);

        for (uint s = 0u; s < 4u; s++) {
            if (sgid == s) {
                #pragma unroll
                for (uint16_t i = 0; i < uint16_t(GCAPQK); ++i) {
                    if (cqk.is_valid_element(i)) {
                        if (s == 0u) {
                            sct[uint(mq[i]) * uint(GTN) + uint(nq[i])] = cqk[i];
                        } else {
                            sct[uint(mq[i]) * uint(GTN) + uint(nq[i])] += cqk[i];
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (lid < uint(GTM)) {
            uint r = lid;
            if (r < rows) {
                uint qi = qt0 + r;
                uint tr = fd.total - (mr - 1u - qi);
                uint st = (fd.window > 0u && tr > fd.window) ? tr - fd.window : 0u;
                float neg = na_attn_neg();
                float mx = row_m[r];
                float cmx = neg;
                float sv[GTN];
                bool lv[GTN];
                for (uint n = 0u; n < uint(GTN); n++) {
                    uint p = cb + n;
                    bool live = p >= st && p < tr;
                    lv[n] = live;
                    float v = live ? (sct[r * uint(GTN) + n] * kst[n]) * fd.scaling : 0.0f;
                    sv[n] = v;
                    if (live && v > cmx) {
                        cmx = v;
                    }
                }
                float mnew = max(mx, cmx);
                float corr = 1.0f;
                float l = row_l[r];
                if (mnew > neg) {
                    if (mx > neg) {
                        corr = exp2((mx - mnew) * AT_LOG2E);
                    }
                    l = l * corr;
                    for (uint n = 0u; n < uint(GTN); n++) {
                        float w = lv[n] ? exp2((sv[n] - mnew) * AT_LOG2E) : 0.0f;
                        l += w;
                        pt[r * uint(GTN) + n] = w * vst[n];
                    }
                    row_m[r] = mnew;
                    row_l[r] = l;
                } else {
                    for (uint n = 0u; n < uint(GTN); n++) {
                        pt[r * uint(GTN) + n] = 0.0f;
                    }
                }
                row_c[r] = corr;
            } else {
                for (uint n = 0u; n < uint(GTN); n++) {
                    pt[r * uint(GTN) + n] = 0.0f;
                }
                row_c[r] = 1.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = lid; i < uint(GTN * GHD); i += 128u) {
            uint n = i >> 9u;
            uint d = i & 511u;
            uint p = cb + n;
            float v = 0.0f;
            if (p < fd.total) {
                uint sp = (fd.ring > 0u) ? (p % fd.ring) : p;
                uint idx = (sp * nkv + kvh) * uint(GHD) + d;
                v = na_attn_e4m3((vw[idx >> 2u] >> (8u * (idx & 3u))) & 255u);
            }
            vt[i] = half(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto sP = tP.slice(0, 0);
        auto sV = tV.slice(int(sgid) * GHQ, 0);
        oppv.run(sP, sV, cpv);
        #pragma unroll
        for (uint16_t i = 0; i < uint16_t(GCAPPV); ++i) {
            acc[i] = acc[i] * row_c[mp[i]] + cpv[i];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid < rows) {
        float l = row_l[lid];
        float inv = 0.0f;
        if (l > 0.0f) {
            float rr = 1.0f / l;
            inv = fma(fma(-l, rr, 1.0f), rr, rr);
        }
        row_inv[lid] = inv;
    }
    for (uint g = 0u; g < 4u; g++) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgid == g) {
            #pragma unroll
            for (uint16_t i = 0; i < uint16_t(GCAPPV); ++i) {
                if (vp[i]) {
                    shf[uint(mp[i]) * uint(GHQ) + uint(np[i])] = acc[i] * row_inv[mp[i]];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = lid; i < rows * uint(GHQ / 2); i += 128u) {
            uint r = i / uint(GHQ / 2);
            uint w = i % uint(GHQ / 2);
            uint lo = na_attn_bf16_encode(shf[r * uint(GHQ) + 2u * w]) & 0xffffu;
            uint hi = na_attn_bf16_encode(shf[r * uint(GHQ) + 2u * w + 1u]) & 0xffffu;
            yout[((qt0 + r) * fd.n_heads + h) * uint(GHD / 2) + g * uint(GHQ / 2) + w] =
                lo | (hi << 16u);
        }
    }
}
"#;

struct NaAttnPipelines {
    hd256: Arc<wgpu::ComputePipeline>,
    hd512: Arc<wgpu::ComputePipeline>,
}

fn build_pipelines(ctx: &WgpuContext) -> Result<Arc<NaAttnPipelines>> {
    let module = na::msl_module(
        ctx,
        "nv-na-attn",
        &[(ENTRY, (WG_THREADS, 1, 1)), (ENTRY_G, (WG_THREADS, 1, 1))],
        MSL,
    )?;
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let mut entries: Vec<wgpu::BindGroupLayoutEntry> =
        (0..5).map(|b| storage_entry(b, true)).collect();
    entries.push(storage_entry(5, false));
    entries.push(uniform_entry(6));
    let bgl = ctx
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("nv-na-attn"),
            entries: &entries,
        });
    let layout = ctx
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("nv-na-attn"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
    let mk = |entry: &str| {
        Arc::new(
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("nv-na-attn"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        )
    };
    let hd256 = mk(ENTRY);
    let hd512 = mk(ENTRY_G);
    if let Some(err) = pollster::block_on(scope.pop()) {
        return Err(WgpuError::ShaderCompile(format!(
            "na_attn passthrough: {err}"
        )));
    }
    profile::name_pipeline(&hd256, "na-attn:na_attn_prefill");
    profile::name_pipeline(&hd512, "na-attn:na_attn_prefill_g");
    Ok(Arc::new(NaAttnPipelines { hd256, hd512 }))
}

type CacheEntry = (
    wgpu::Device,
    std::result::Result<Arc<NaAttnPipelines>, WgpuError>,
);

fn cache() -> &'static Mutex<Vec<CacheEntry>> {
    static CACHE: OnceLock<Mutex<Vec<CacheEntry>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(Vec::new()))
}

fn pipelines(ctx: &WgpuContext) -> Result<Arc<NaAttnPipelines>> {
    let mut guard = cache().lock().unwrap();
    for (dev, entry) in guard.iter() {
        if *dev == ctx.device {
            return entry.clone();
        }
    }
    let built = if na::supported(ctx) {
        build_pipelines(ctx)
    } else {
        Err(WgpuError::Unsupported(
            "na tensor-ops need the Metal backend with PASSTHROUGH_SHADERS".into(),
        ))
    };
    guard.push((ctx.device.clone(), built.clone()));
    built
}

pub fn pipeline(ctx: &WgpuContext) -> Result<Arc<wgpu::ComputePipeline>> {
    pipelines(ctx).map(|p| p.hd256.clone())
}

pub fn pipeline_g(ctx: &WgpuContext) -> Result<Arc<wgpu::ComputePipeline>> {
    pipelines(ctx).map(|p| p.hd512.clone())
}

pub fn available(ctx: &WgpuContext) -> bool {
    pipelines(ctx).is_ok()
}

pub fn pipeline_label(p: &wgpu::ComputePipeline) -> Option<&'static str> {
    let guard = cache().lock().ok()?;
    for (_, entry) in guard.iter() {
        if let Ok(pls) = entry {
            if pls.hd256.as_ref() == p {
                return Some("na_attn");
            }
            if pls.hd512.as_ref() == p {
                return Some("na_attn_g");
            }
        }
    }
    None
}

pub fn grid(n_heads: u32, m_rows: u32) -> (u32, u32, u32) {
    (n_heads, m_rows.div_ceil(TILE_M), 1)
}
