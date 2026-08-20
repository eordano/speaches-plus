
struct GowFdParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    splits: u32,
    total: u32,
    start: u32,
    pad0: u32,
    scale: f32,
};

@group(0) @binding(0) var<storage, read> gfd_q: array<u32>;
@group(0) @binding(1) var<storage, read> gfd_kc: array<u32>;
@group(0) @binding(2) var<storage, read> gfd_vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> gfd_scratch: array<f32>;
@group(0) @binding(4) var<storage, read_write> gfd_out: array<u32>;
@group(0) @binding(5) var<storage, read> gfd_sinks: array<f32>;
@group(0) @binding(6) var<uniform> gfd_p: GowFdParams;

const GFD_BLOCK: u32 = 256u;
const GFD_WARPS: u32 = 8u;
const GFD_LANES: u32 = 32u;
const GFD_MAX_HD: u32 = 256u;
const GFD_LOG2E: f32 = 1.4426950408889634;

var<workgroup> gfd_qsh: array<f32, 256>;
var<workgroup> gfd_red: array<f32, 256>;
var<workgroup> gfd_sm: array<f32, 8>;
var<workgroup> gfd_sl: array<f32, 8>;
var<workgroup> gfd_sacc: array<f32, 2048>;

fn gfd_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn gfd_exp(x: f32) -> f32 {
    return exp2(x * GFD_LOG2E);
}

fn gfd_recip(x: f32) -> f32 {
    let r = 1.0 / x;
    return fma(fma(-x, r, 1.0), r, r);
}

fn gfd_warp_sum(lid: u32, x: f32) -> f32 {
    gfd_red[lid] = x;
    workgroupBarrier();
    for (var o = 16u; o > 0u; o = o >> 1u) {
        let other = gfd_red[lid ^ o];
        workgroupBarrier();
        gfd_red[lid] = gfd_red[lid] + other;
        workgroupBarrier();
    }
    return gfd_red[lid];
}

fn gfd_k_bf16(idx: u32) -> f32 {
    return bf16_decode(u16_at(gfd_kc[idx >> 1u], idx));
}

fn gfd_v_bf16(idx: u32) -> f32 {
    return bf16_decode(u16_at(gfd_vc[idx >> 1u], idx));
}

@compute @workgroup_size(256)
fn gow_flash_stage1(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= gfd_p.n_heads) {
        return;
    }
    let hd = gfd_p.head_dim;
    let nkv = gfd_p.n_kv;
    let group = gfd_p.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var d = lid; d < hd; d = d + GFD_BLOCK) {
        let qi = h * hd + d;
        gfd_qsh[d] = bf16_decode(u16_at(gfd_q[qi >> 1u], qi));
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
    var m = gfd_neg_inf();
    var l = 0.0;

    let total = gfd_p.total;
    let base = gfd_p.start + split * GFD_WARPS;
    let stride = gfd_p.splits * GFD_WARPS;
    var rounds = 0u;
    if (total > base) {
        rounds = (total - base + stride - 1u) / stride;
    }
    let use_vec8 = (hd & 7u) == 0u;

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = base + warp + r * stride;
        let live = p < total;
        var partial = 0.0;
        if (live) {
            let kbase = (p * nkv + kvh) * hd;
            if (use_vec8) {
                let n8 = hd >> 3u;
                for (var j = lane; j < n8; j = j + GFD_LANES) {
                    let qb = j * 8u;
                    let kb = kbase + qb;
                    for (var t = 0u; t < 4u; t = t + 1u) {
                        let kx = gfd_k_bf16(kb + 2u * t);
                        let ky = gfd_k_bf16(kb + 2u * t + 1u);
                        let pair = fma(kx, gfd_qsh[qb + 2u * t], ky * gfd_qsh[qb + 2u * t + 1u]);
                        partial = partial + pair;
                    }
                }
            } else {
                for (var d = lane; d < hd; d = d + GFD_LANES) {
                    partial = fma(gfd_qsh[d], gfd_k_bf16(kbase + d), partial);
                }
            }
        }
        let score = gfd_warp_sum(lid, partial) * gfd_p.scale;
        if (live) {
            let m_new = max(m, score);
            let corr = gfd_exp(m - m_new);
            let w = gfd_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (p * nkv + kvh) * hd;
            {
                let d = lane + 0u * GFD_LANES;
                if (d < hd) {
                    acc0 = fma(acc0, corr, w * gfd_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 1u * GFD_LANES;
                if (d < hd) {
                    acc1 = fma(acc1, corr, w * gfd_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 2u * GFD_LANES;
                if (d < hd) {
                    acc2 = fma(acc2, corr, w * gfd_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 3u * GFD_LANES;
                if (d < hd) {
                    acc3 = fma(acc3, corr, w * gfd_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 4u * GFD_LANES;
                if (d < hd) {
                    acc4 = fma(acc4, corr, w * gfd_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 5u * GFD_LANES;
                if (d < hd) {
                    acc5 = fma(acc5, corr, w * gfd_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 6u * GFD_LANES;
                if (d < hd) {
                    acc6 = fma(acc6, corr, w * gfd_v_bf16(vbase + d));
                }
            }
            {
                let d = lane + 7u * GFD_LANES;
                if (d < hd) {
                    acc7 = fma(acc7, corr, w * gfd_v_bf16(vbase + d));
                }
            }
            m = m_new;
        }
    }

    {
        let d = lane + 0u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc0;
        }
    }
    {
        let d = lane + 1u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc1;
        }
    }
    {
        let d = lane + 2u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc2;
        }
    }
    {
        let d = lane + 3u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc3;
        }
    }
    {
        let d = lane + 4u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc4;
        }
    }
    {
        let d = lane + 5u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc5;
        }
    }
    {
        let d = lane + 6u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc6;
        }
    }
    {
        let d = lane + 7u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc7;
        }
    }
    let slot = (h * gfd_p.splits + split) * (hd + 2u);
    if (lane == 0u) {
        gfd_sm[warp] = m;
        gfd_sl[warp] = l;
    }
    workgroupBarrier();
    var m_blk = gfd_neg_inf();
    for (var w = 0u; w < GFD_WARPS; w = w + 1u) {
        m_blk = max(m_blk, gfd_sm[w]);
    }
    if (lid == 0u) {
        var l_blk = 0.0;
        for (var w = 0u; w < GFD_WARPS; w = w + 1u) {
            if (gfd_sm[w] > gfd_neg_inf()) {
                l_blk = l_blk + gfd_sl[w] * gfd_exp(gfd_sm[w] - m_blk);
            }
        }
        gfd_scratch[slot] = m_blk;
        gfd_scratch[slot + 1u] = l_blk;
    }
    for (var d = lid; d < hd; d = d + GFD_BLOCK) {
        var a = 0.0;
        for (var w = 0u; w < GFD_WARPS; w = w + 1u) {
            if (gfd_sm[w] > gfd_neg_inf()) {
                a = a + gfd_sacc[w * GFD_MAX_HD + d] * gfd_exp(gfd_sm[w] - m_blk);
            }
        }
        gfd_scratch[slot + 2u + d] = a;
    }
}

@compute @workgroup_size(256)
fn gow_flash_stage1_sg(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) sglane: u32
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= gfd_p.n_heads) {
        return;
    }
    let hd = gfd_p.head_dim;
    let nkv = gfd_p.n_kv;
    let group = gfd_p.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;
    let lane = sglane;
    let warp = sgid;

    for (var d = lid; d < hd; d = d + GFD_BLOCK) {
        let qi = h * hd + d;
        gfd_qsh[d] = bf16_decode(u16_at(gfd_q[qi >> 1u], qi));
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
    var m = gfd_neg_inf();
    var l = 0.0;

    let total = gfd_p.total;
    let base = gfd_p.start + split * GFD_WARPS;
    let stride = gfd_p.splits * GFD_WARPS;
    let use_vec8 = (hd & 7u) == 0u;

    for (var p = base + warp; p < total; p = p + stride) {
        var partial = 0.0;
        let kbase = (p * nkv + kvh) * hd;
        if (use_vec8) {
            let n8 = hd >> 3u;
            for (var j = lane; j < n8; j = j + GFD_LANES) {
                let qb = j * 8u;
                let kb = kbase + qb;
                for (var t = 0u; t < 4u; t = t + 1u) {
                    let kx = gfd_k_bf16(kb + 2u * t);
                    let ky = gfd_k_bf16(kb + 2u * t + 1u);
                    let pair = fma(kx, gfd_qsh[qb + 2u * t], ky * gfd_qsh[qb + 2u * t + 1u]);
                    partial = partial + pair;
                }
            }
        } else {
            for (var d = lane; d < hd; d = d + GFD_LANES) {
                partial = fma(gfd_qsh[d], gfd_k_bf16(kbase + d), partial);
            }
        }
        let score = subgroupAdd(partial) * gfd_p.scale;
        let m_new = max(m, score);
        let corr = gfd_exp(m - m_new);
        let w = gfd_exp(score - m_new);
        l = fma(l, corr, w);
        let vbase = (p * nkv + kvh) * hd;
        {
            let d = lane + 0u * GFD_LANES;
            if (d < hd) {
                acc0 = fma(acc0, corr, w * gfd_v_bf16(vbase + d));
            }
        }
        {
            let d = lane + 1u * GFD_LANES;
            if (d < hd) {
                acc1 = fma(acc1, corr, w * gfd_v_bf16(vbase + d));
            }
        }
        {
            let d = lane + 2u * GFD_LANES;
            if (d < hd) {
                acc2 = fma(acc2, corr, w * gfd_v_bf16(vbase + d));
            }
        }
        {
            let d = lane + 3u * GFD_LANES;
            if (d < hd) {
                acc3 = fma(acc3, corr, w * gfd_v_bf16(vbase + d));
            }
        }
        {
            let d = lane + 4u * GFD_LANES;
            if (d < hd) {
                acc4 = fma(acc4, corr, w * gfd_v_bf16(vbase + d));
            }
        }
        {
            let d = lane + 5u * GFD_LANES;
            if (d < hd) {
                acc5 = fma(acc5, corr, w * gfd_v_bf16(vbase + d));
            }
        }
        {
            let d = lane + 6u * GFD_LANES;
            if (d < hd) {
                acc6 = fma(acc6, corr, w * gfd_v_bf16(vbase + d));
            }
        }
        {
            let d = lane + 7u * GFD_LANES;
            if (d < hd) {
                acc7 = fma(acc7, corr, w * gfd_v_bf16(vbase + d));
            }
        }
        m = m_new;
    }

    {
        let d = lane + 0u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc0;
        }
    }
    {
        let d = lane + 1u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc1;
        }
    }
    {
        let d = lane + 2u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc2;
        }
    }
    {
        let d = lane + 3u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc3;
        }
    }
    {
        let d = lane + 4u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc4;
        }
    }
    {
        let d = lane + 5u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc5;
        }
    }
    {
        let d = lane + 6u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc6;
        }
    }
    {
        let d = lane + 7u * GFD_LANES;
        if (d < hd) {
            gfd_sacc[warp * GFD_MAX_HD + d] = acc7;
        }
    }
    let slot = (h * gfd_p.splits + split) * (hd + 2u);
    if (lane == 0u) {
        gfd_sm[warp] = m;
        gfd_sl[warp] = l;
    }
    workgroupBarrier();
    var m_blk = gfd_neg_inf();
    for (var w = 0u; w < GFD_WARPS; w = w + 1u) {
        m_blk = max(m_blk, gfd_sm[w]);
    }
    if (lid == 0u) {
        var l_blk = 0.0;
        for (var w = 0u; w < GFD_WARPS; w = w + 1u) {
            if (gfd_sm[w] > gfd_neg_inf()) {
                l_blk = l_blk + gfd_sl[w] * gfd_exp(gfd_sm[w] - m_blk);
            }
        }
        gfd_scratch[slot] = m_blk;
        gfd_scratch[slot + 1u] = l_blk;
    }
    for (var d = lid; d < hd; d = d + GFD_BLOCK) {
        var a = 0.0;
        for (var w = 0u; w < GFD_WARPS; w = w + 1u) {
            if (gfd_sm[w] > gfd_neg_inf()) {
                a = a + gfd_sacc[w * GFD_MAX_HD + d] * gfd_exp(gfd_sm[w] - m_blk);
            }
        }
        gfd_scratch[slot + 2u + d] = a;
    }
}

const GK_HD: u32 = 64u;
const GK_GROUP: u32 = 8u;

var<workgroup> gk_q: array<f32, 512>;
var<workgroup> gk_sm: array<f32, 64>;
var<workgroup> gk_sl: array<f32, 64>;
var<workgroup> gk_sacc: array<f32, 4096>;

@compute @workgroup_size(256)
fn gow_flash_stage1_kvshare_sg_group8_hd64(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) sglane: u32
) {
    let kvh = wg.x;
    let split = wg.y;
    let nkv = gfd_p.n_kv;
    if (kvh >= nkv) {
        return;
    }
    let lid = tid.x;
    let lane = sglane;
    let warp = sgid;

    for (var i = lid; i < GK_GROUP * GK_HD; i = i + GFD_BLOCK) {
        let qi = kvh * GK_GROUP * GK_HD + i;
        gk_q[i] = bf16_decode(u16_at(gfd_q[qi >> 1u], qi));
    }
    workgroupBarrier();

    var m: array<f32, 8>;
    var l: array<f32, 8>;
    var acc0: array<f32, 8>;
    var acc1: array<f32, 8>;
    for (var g = 0u; g < GK_GROUP; g = g + 1u) {
        m[g] = gfd_neg_inf();
        l[g] = 0.0;
        acc0[g] = 0.0;
        acc1[g] = 0.0;
    }

    let total = gfd_p.total;
    let base = gfd_p.start + split * GFD_WARPS;
    let stride = gfd_p.splits * GFD_WARPS;

    for (var p = base + warp; p < total; p = p + stride) {
        let kbase = (p * nkv + kvh) * GK_HD;
        let kd0 = gfd_k_bf16(kbase + lane);
        let kd1 = gfd_k_bf16(kbase + lane + GFD_LANES);
        let vd0 = gfd_v_bf16(kbase + lane);
        let vd1 = gfd_v_bf16(kbase + lane + GFD_LANES);
        for (var g = 0u; g < GK_GROUP; g = g + 1u) {
            let partial = fma(kd0, gk_q[g * GK_HD + lane], kd1 * gk_q[g * GK_HD + lane + GFD_LANES]);
            let score = subgroupAdd(partial) * gfd_p.scale;
            let m_new = max(m[g], score);
            let corr = gfd_exp(m[g] - m_new);
            let w = gfd_exp(score - m_new);
            l[g] = fma(l[g], corr, w);
            acc0[g] = fma(acc0[g], corr, w * vd0);
            acc1[g] = fma(acc1[g], corr, w * vd1);
            m[g] = m_new;
        }
    }

    for (var g = 0u; g < GK_GROUP; g = g + 1u) {
        gk_sacc[(warp * GK_GROUP + g) * GK_HD + lane] = acc0[g];
        gk_sacc[(warp * GK_GROUP + g) * GK_HD + lane + GFD_LANES] = acc1[g];
        if (lane == 0u) {
            gk_sm[warp * GK_GROUP + g] = m[g];
            gk_sl[warp * GK_GROUP + g] = l[g];
        }
    }
    workgroupBarrier();

    for (var i = lid; i < GK_GROUP * GK_HD; i = i + GFD_BLOCK) {
        let g = i / GK_HD;
        let d = i % GK_HD;
        var m_blk = gfd_neg_inf();
        for (var w = 0u; w < GFD_WARPS; w = w + 1u) {
            m_blk = max(m_blk, gk_sm[w * GK_GROUP + g]);
        }
        var a = 0.0;
        for (var w = 0u; w < GFD_WARPS; w = w + 1u) {
            if (gk_sm[w * GK_GROUP + g] > gfd_neg_inf()) {
                a = a + gk_sacc[(w * GK_GROUP + g) * GK_HD + d] * gfd_exp(gk_sm[w * GK_GROUP + g] - m_blk);
            }
        }
        let slot = ((kvh * GK_GROUP + g) * gfd_p.splits + split) * (GK_HD + 2u);
        gfd_scratch[slot + 2u + d] = a;
        if (d == 0u) {
            var l_blk = 0.0;
            for (var w = 0u; w < GFD_WARPS; w = w + 1u) {
                if (gk_sm[w * GK_GROUP + g] > gfd_neg_inf()) {
                    l_blk = l_blk + gk_sl[w * GK_GROUP + g] * gfd_exp(gk_sm[w * GK_GROUP + g] - m_blk);
                }
            }
            gfd_scratch[slot] = m_blk;
            gfd_scratch[slot + 1u] = l_blk;
        }
    }
}

fn gfd_split_scale(base: u32, stride: u32, s: u32, m_glob: f32) -> f32 {
    let p0 = gfd_scratch[base + s * stride];
    var sc = 0.0;
    if (p0 > gfd_neg_inf()) {
        sc = gfd_exp(p0 - m_glob);
    }
    return sc;
}

@compute @workgroup_size(256)
fn gow_flash_stage2_sink_pk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= gfd_p.n_heads) {
        return;
    }
    let hd = gfd_p.head_dim;
    let splits = gfd_p.splits;
    let stride = hd + 2u;
    let base = h * splits * stride;
    let sink = gfd_sinks[h];

    var m_glob = sink;
    for (var s = 0u; s < splits; s = s + 1u) {
        m_glob = max(m_glob, gfd_scratch[base + s * stride]);
    }
    var l_glob = gfd_exp(sink - m_glob);
    for (var s = 0u; s < splits; s = s + 1u) {
        let sc = gfd_split_scale(base, stride, s, m_glob);
        l_glob = fma(gfd_scratch[base + s * stride + 1u], sc, l_glob);
    }
    let inv_l = gfd_recip(l_glob);
    let hw = hd >> 1u;
    for (var w2 = tid.x; w2 < hw; w2 = w2 + GFD_BLOCK) {
        let d0 = w2 * 2u;
        var a0 = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            let sc = gfd_split_scale(base, stride, s, m_glob);
            a0 = fma(gfd_scratch[base + s * stride + 2u + d0], sc, a0);
        }
        var a1 = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            let sc = gfd_split_scale(base, stride, s, m_glob);
            a1 = fma(gfd_scratch[base + s * stride + 2u + d0 + 1u], sc, a1);
        }
        gfd_out[h * hw + w2] = bf16_pack(a0 * inv_l, a1 * inv_l);
    }
}
