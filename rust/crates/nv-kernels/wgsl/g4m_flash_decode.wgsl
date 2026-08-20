
struct GfParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    splits: u32,
    total: u32,
    start: u32,
    ring: u32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read> gf_q: array<u32>;
@group(0) @binding(1) var<storage, read> gf_kc: array<u32>;
@group(0) @binding(2) var<storage, read> gf_vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> gf_scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> gf_out: array<u32>;
@group(0) @binding(6) var<uniform> gf_p: GfParams;
@group(0) @binding(7) var<storage, read> gf_ksc: array<f32>;
@group(0) @binding(8) var<storage, read> gf_vsc: array<f32>;

const GF_BLOCK: u32 = 256u;
const GF_WARPS: u32 = 8u;
const GF_LANES: u32 = 32u;
const GF_MAX_HD: u32 = 512u;
const GF_LOG2E: f32 = 1.4426950408889634;

var<workgroup> gf_qsh: array<f32, 512>;
var<workgroup> gf_red: array<f32, 256>;
var<workgroup> gf_sm: array<f32, 8>;
var<workgroup> gf_sl: array<f32, 8>;
var<workgroup> gf_sacc: array<f32, 4096>;

fn gf_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn gf_exp(x: f32) -> f32 {
    return exp2(x * GF_LOG2E);
}

fn gf_recip(x: f32) -> f32 {
    let r = 1.0 / x;
    return fma(fma(-x, r, 1.0), r, r);
}

fn gf_warp_sum(lid: u32, x: f32) -> f32 {
    gf_red[lid] = x;
    workgroupBarrier();
    for (var o = 16u; o > 0u; o = o >> 1u) {
        let other = gf_red[lid ^ o];
        workgroupBarrier();
        gf_red[lid] = gf_red[lid] + other;
        workgroupBarrier();
    }
    return gf_red[lid];
}

fn gf_k_bf16(idx: u32) -> f32 {
    return bf16_decode(u16_at(gf_kc[idx >> 1u], idx));
}

fn gf_v_bf16(idx: u32) -> f32 {
    return bf16_decode(u16_at(gf_vc[idx >> 1u], idx));
}

fn gf_stage1_epilogue(lid: u32, lane: u32, warp: u32, hd: u32, slot: u32, m: f32, l: f32) {
    if (lane == 0u) {
        gf_sm[warp] = m;
        gf_sl[warp] = l;
    }
    workgroupBarrier();
    if (warp == 0u) {
        var m_blk = gf_neg_inf();
        for (var w = 0u; w < GF_WARPS; w = w + 1u) {
            m_blk = max(m_blk, gf_sm[w]);
        }
        var l_blk = 0.0;
        for (var w = 0u; w < GF_WARPS; w = w + 1u) {
            if (gf_sm[w] > gf_neg_inf()) {
                l_blk = l_blk + gf_sl[w] * gf_exp(gf_sm[w] - m_blk);
            }
        }
        if (lane == 0u) {
            gf_scratch[slot] = m_blk;
            gf_scratch[slot + 1u] = l_blk;
        }
    }
    var m_blk = gf_neg_inf();
    for (var w = 0u; w < GF_WARPS; w = w + 1u) {
        m_blk = max(m_blk, gf_sm[w]);
    }
    for (var d = lid; d < hd; d = d + GF_BLOCK) {
        var a = 0.0;
        for (var w = 0u; w < GF_WARPS; w = w + 1u) {
            if (gf_sm[w] > gf_neg_inf()) {
                a = a + gf_sacc[w * GF_MAX_HD + d] * gf_exp(gf_sm[w] - m_blk);
            }
        }
        gf_scratch[slot + 2u + d] = a;
    }
}

@compute @workgroup_size(256)
fn g4m_flash_stage1_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= gf_p.n_heads) {
        return;
    }
    let hd = gf_p.head_dim;
    let nkv = gf_p.n_kv;
    let group = gf_p.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var d = lid; d < hd; d = d + GF_BLOCK) {
        let qi = h * hd + d;
        gf_qsh[d] = bf16_decode(u16_at(gf_q[qi >> 1u], qi));
    }
    workgroupBarrier();

    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    var acc4 = 0.0;
    var acc5 = 0.0;
    var acc6 = 0.0;
    var acc7 = 0.0;
    var acc8 = 0.0;
    var acc9 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc12 = 0.0;
    var acc13 = 0.0;
    var acc14 = 0.0;
    var acc15 = 0.0;
    var m = gf_neg_inf();
    var l = 0.0;

    let total = gf_p.total;
    let base = gf_p.start + split * GF_WARPS;
    let stride = gf_p.splits * GF_WARPS;
    var rounds = 0u;
    if (total > base) {
        rounds = (total - base + stride - 1u) / stride;
    }
    let use_vec8 = (hd & 7u) == 0u;

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = base + warp + r * stride;
        let live = p < total;
        var sp = p;
        if (gf_p.ring > 0u) {
            sp = p % gf_p.ring;
        }
        var partial = 0.0;
        if (live) {
            let kbase = (sp * nkv + kvh) * hd;
            if (use_vec8) {
                let n8 = hd >> 3u;
                for (var j = lane; j < n8; j = j + GF_LANES) {
                    let qb = j * 8u;
                    let kb = kbase + qb;
                    for (var t = 0u; t < 4u; t = t + 1u) {
                        let kx = gf_k_bf16(kb + 2u * t);
                        let ky = gf_k_bf16(kb + 2u * t + 1u);
                        let pair = fma(kx, gf_qsh[qb + 2u * t], ky * gf_qsh[qb + 2u * t + 1u]);
                        partial = partial + pair;
                    }
                }
            } else {
                for (var d = lane; d < hd; d = d + GF_LANES) {
                    partial = fma(gf_qsh[d], gf_k_bf16(kbase + d), partial);
                }
            }
        }
        let score = gf_warp_sum(lid, partial) * gf_p.scale;
        if (live) {
            let m_new = max(m, score);
            let corr = gf_exp(m - m_new);
            let w = gf_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (sp * nkv + kvh) * hd;
            {
                let d = lane + 0u * GF_LANES;
                if (d < hd) {
                    acc0 = fma(acc0, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 1u * GF_LANES;
                if (d < hd) {
                    acc1 = fma(acc1, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 2u * GF_LANES;
                if (d < hd) {
                    acc2 = fma(acc2, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 3u * GF_LANES;
                if (d < hd) {
                    acc3 = fma(acc3, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 4u * GF_LANES;
                if (d < hd) {
                    acc4 = fma(acc4, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 5u * GF_LANES;
                if (d < hd) {
                    acc5 = fma(acc5, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 6u * GF_LANES;
                if (d < hd) {
                    acc6 = fma(acc6, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 7u * GF_LANES;
                if (d < hd) {
                    acc7 = fma(acc7, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 8u * GF_LANES;
                if (d < hd) {
                    acc8 = fma(acc8, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 9u * GF_LANES;
                if (d < hd) {
                    acc9 = fma(acc9, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 10u * GF_LANES;
                if (d < hd) {
                    acc10 = fma(acc10, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 11u * GF_LANES;
                if (d < hd) {
                    acc11 = fma(acc11, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 12u * GF_LANES;
                if (d < hd) {
                    acc12 = fma(acc12, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 13u * GF_LANES;
                if (d < hd) {
                    acc13 = fma(acc13, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 14u * GF_LANES;
                if (d < hd) {
                    acc14 = fma(acc14, corr, w * gf_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 15u * GF_LANES;
                if (d < hd) {
                    acc15 = fma(acc15, corr, w * gf_v_bf16(vbase + d));
                }
            }
            m = m_new;
        }
    }

    {
        let d = lane + 0u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc0;
        }
    }
    {
        let d = lane + 1u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc1;
        }
    }
    {
        let d = lane + 2u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc2;
        }
    }
    {
        let d = lane + 3u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc3;
        }
    }
    {
        let d = lane + 4u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc4;
        }
    }
    {
        let d = lane + 5u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc5;
        }
    }
    {
        let d = lane + 6u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc6;
        }
    }
    {
        let d = lane + 7u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc7;
        }
    }
    {
        let d = lane + 8u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc8;
        }
    }
    {
        let d = lane + 9u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc9;
        }
    }
    {
        let d = lane + 10u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc10;
        }
    }
    {
        let d = lane + 11u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc11;
        }
    }
    {
        let d = lane + 12u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc12;
        }
    }
    {
        let d = lane + 13u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc13;
        }
    }
    {
        let d = lane + 14u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc14;
        }
    }
    {
        let d = lane + 15u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc15;
        }
    }
    let slot = (h * gf_p.splits + split) * (hd + 2u);
    gf_stage1_epilogue(lid, lane, warp, hd, slot, m, l);
}

const GF_E4M3_SHIFT_CARRY_2POW120_RIDES_Q_AND_V_SCALE: f32 = 1329227995784915872903807060280344576.0;

fn gf_v_fp8(idx: u32) -> f32 {
    return e4m3_shift_decode_scale_must_carry_2pow120(byte_at(gf_vc[idx >> 2u], idx));
}

@compute @workgroup_size(256)
fn g4m_flash_stage1_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= gf_p.n_heads) {
        return;
    }
    let hd = gf_p.head_dim;
    let nkv = gf_p.n_kv;
    let group = gf_p.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var d = lid; d < hd; d = d + GF_BLOCK) {
        let qi = h * hd + d;
        gf_qsh[d] = bf16_decode(u16_at(gf_q[qi >> 1u], qi))
            * GF_E4M3_SHIFT_CARRY_2POW120_RIDES_Q_AND_V_SCALE;
    }
    workgroupBarrier();

    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    var acc4 = 0.0;
    var acc5 = 0.0;
    var acc6 = 0.0;
    var acc7 = 0.0;
    var acc8 = 0.0;
    var acc9 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc12 = 0.0;
    var acc13 = 0.0;
    var acc14 = 0.0;
    var acc15 = 0.0;
    var m = gf_neg_inf();
    var l = 0.0;

    let total = gf_p.total;
    let base = gf_p.start + split * GF_WARPS;
    let stride = gf_p.splits * GF_WARPS;
    var rounds = 0u;
    if (total > base) {
        rounds = (total - base + stride - 1u) / stride;
    }

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = base + warp + r * stride;
        let live = p < total;
        var sp = p;
        if (gf_p.ring > 0u) {
            sp = p % gf_p.ring;
        }
        var partial = 0.0;
        var ks = 0.0;
        var vs = 0.0;
        if (live) {
            let krow = sp * nkv + kvh;
            ks = gf_ksc[krow];
            vs = gf_vsc[krow] * GF_E4M3_SHIFT_CARRY_2POW120_RIDES_Q_AND_V_SCALE;
            let kw = (krow * hd) >> 2u;
            let n4 = hd >> 2u;
            for (var j = lane; j < n4; j = j + GF_LANES) {
                let w = gf_kc[kw + j];
                let d0 = j * 4u;
                partial = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(w, 0u)), gf_qsh[d0], partial);
                partial = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(w, 1u)), gf_qsh[d0 + 1u], partial);
                partial = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(w, 2u)), gf_qsh[d0 + 2u], partial);
                partial = fma(e4m3_shift_decode_scale_must_carry_2pow120(byte_at(w, 3u)), gf_qsh[d0 + 3u], partial);
            }
        }
        let score = gf_warp_sum(lid, partial) * gf_p.scale * ks;
        if (live) {
            let m_new = max(m, score);
            let corr = gf_exp(m - m_new);
            let w = gf_exp(score - m_new);
            l = fma(l, corr, w);
            let wv = w * vs;
            let vbase = (sp * nkv + kvh) * hd;
            {
                let d = lane + 0u * GF_LANES;
                if (d < hd) {
                    acc0 = fma(acc0, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 1u * GF_LANES;
                if (d < hd) {
                    acc1 = fma(acc1, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 2u * GF_LANES;
                if (d < hd) {
                    acc2 = fma(acc2, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 3u * GF_LANES;
                if (d < hd) {
                    acc3 = fma(acc3, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 4u * GF_LANES;
                if (d < hd) {
                    acc4 = fma(acc4, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 5u * GF_LANES;
                if (d < hd) {
                    acc5 = fma(acc5, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 6u * GF_LANES;
                if (d < hd) {
                    acc6 = fma(acc6, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 7u * GF_LANES;
                if (d < hd) {
                    acc7 = fma(acc7, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 8u * GF_LANES;
                if (d < hd) {
                    acc8 = fma(acc8, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 9u * GF_LANES;
                if (d < hd) {
                    acc9 = fma(acc9, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 10u * GF_LANES;
                if (d < hd) {
                    acc10 = fma(acc10, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 11u * GF_LANES;
                if (d < hd) {
                    acc11 = fma(acc11, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 12u * GF_LANES;
                if (d < hd) {
                    acc12 = fma(acc12, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 13u * GF_LANES;
                if (d < hd) {
                    acc13 = fma(acc13, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 14u * GF_LANES;
                if (d < hd) {
                    acc14 = fma(acc14, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            {
                let d = lane + 15u * GF_LANES;
                if (d < hd) {
                    acc15 = fma(acc15, corr, wv * gf_v_fp8(vbase + d));
                }
            }
            m = m_new;
        }
    }

    {
        let d = lane + 0u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc0;
        }
    }
    {
        let d = lane + 1u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc1;
        }
    }
    {
        let d = lane + 2u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc2;
        }
    }
    {
        let d = lane + 3u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc3;
        }
    }
    {
        let d = lane + 4u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc4;
        }
    }
    {
        let d = lane + 5u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc5;
        }
    }
    {
        let d = lane + 6u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc6;
        }
    }
    {
        let d = lane + 7u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc7;
        }
    }
    {
        let d = lane + 8u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc8;
        }
    }
    {
        let d = lane + 9u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc9;
        }
    }
    {
        let d = lane + 10u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc10;
        }
    }
    {
        let d = lane + 11u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc11;
        }
    }
    {
        let d = lane + 12u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc12;
        }
    }
    {
        let d = lane + 13u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc13;
        }
    }
    {
        let d = lane + 14u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc14;
        }
    }
    {
        let d = lane + 15u * GF_LANES;
        if (d < hd) {
            gf_sacc[warp * GF_MAX_HD + d] = acc15;
        }
    }
    let slot = (h * gf_p.splits + split) * (hd + 2u);
    gf_stage1_epilogue(lid, lane, warp, hd, slot, m, l);
}

fn gf_split_scale(base: u32, stride: u32, s: u32, m_glob: f32) -> f32 {
    let p0 = gf_scratch[base + s * stride];
    var sc = 0.0;
    if (p0 > gf_neg_inf()) {
        sc = gf_exp(p0 - m_glob);
    }
    return sc;
}

@compute @workgroup_size(256)
fn g4m_flash_stage2_pk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= gf_p.n_heads) {
        return;
    }
    let hd = gf_p.head_dim;
    let splits = gf_p.splits;
    let stride = hd + 2u;
    let base = h * splits * stride;

    var m_glob = gf_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) {
        m_glob = max(m_glob, gf_scratch[base + s * stride]);
    }
    var l_glob = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        let sc = gf_split_scale(base, stride, s, m_glob);
        l_glob = fma(gf_scratch[base + s * stride + 1u], sc, l_glob);
    }
    var inv_l = 0.0;
    if (l_glob > 0.0) {
        inv_l = gf_recip(l_glob);
    }
    let hw = hd >> 1u;
    for (var w2 = tid.x; w2 < hw; w2 = w2 + GF_BLOCK) {
        let d0 = w2 * 2u;
        var a0 = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            let sc = gf_split_scale(base, stride, s, m_glob);
            a0 = fma(gf_scratch[base + s * stride + 2u + d0], sc, a0);
        }
        var a1 = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            let sc = gf_split_scale(base, stride, s, m_glob);
            a1 = fma(gf_scratch[base + s * stride + 2u + d0 + 1u], sc, a1);
        }
        gf_out[h * hw + w2] = bf16_pack(a0 * inv_l, a1 * inv_l);
    }
}
