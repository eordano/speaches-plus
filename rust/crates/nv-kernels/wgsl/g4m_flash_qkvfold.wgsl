
struct GfqParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    splits: u32,
    total: u32,
    start: u32,
    ring: u32,
    scale: f32,
};

struct GfqNrParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    has_rope: u32,
    n_q: u32,
    n_kv: u32,
    eps: f32,
};

struct GfqKvParams {
    words: u32,
    ring: u32,
    k_off_words: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> gfq_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> gfq_kc: array<u32>;
@group(0) @binding(2) var<storage, read_write> gfq_vc: array<u32>;
@group(0) @binding(3) var<storage, read_write> gfq_scratch: array<f32>;
@group(0) @binding(4) var<storage, read> gfq_w: array<u32>;
@group(0) @binding(5) var<storage, read> gfq_cos: array<f32>;
@group(0) @binding(6) var<uniform> gfq_p: GfqParams;
@group(0) @binding(7) var<storage, read> gfq_sin: array<f32>;
@group(0) @binding(8) var<uniform> gfq_nr: GfqNrParams;
@group(0) @binding(9) var<uniform> gfq_kv: GfqKvParams;

const GFQ_BLOCK: u32 = 256u;
const GFQ_WARPS: u32 = 8u;
const GFQ_LANES: u32 = 32u;
const GFQ_MAX_HD: u32 = 512u;
const GFQ_LOG2E: f32 = 1.4426950408889634;

var<workgroup> gfq_qsh: array<f32, 512>;
var<workgroup> gfq_red: array<f32, 256>;
var<workgroup> gfq_sm: array<f32, 8>;
var<workgroup> gfq_sl: array<f32, 8>;
var<workgroup> gfq_sacc: array<f32, 4096>;
var<workgroup> gfq_buf: array<f32, 512>;
var<workgroup> gfq_kcur: array<u32, 256>;
var<workgroup> gfq_vcur: array<u32, 256>;

fn gfq_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn gfq_exp(x: f32) -> f32 {
    return exp2(x * GFQ_LOG2E);
}

fn gfq_warp_sum(lid: u32, x: f32) -> f32 {
    gfq_red[lid] = x;
    workgroupBarrier();
    for (var o = 16u; o > 0u; o = o >> 1u) {
        let other = gfq_red[lid ^ o];
        workgroupBarrier();
        gfq_red[lid] = gfq_red[lid] + other;
        workgroupBarrier();
    }
    return gfq_red[lid];
}

fn gfq_row_norm_rope(lid: u32, src_row: u32, w_off: u32, rope: bool, p: u32) -> u32 {
    let hd = gfq_nr.head_dim;
    let e0 = 2u * lid;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < hd) {
        let base = src_row * gfq_nr.src_stride + e0;
        v0 = bf16_decode(u16_at(gfq_src[base >> 1u], base));
        v1 = bf16_decode(u16_at(gfq_src[(base + 1u) >> 1u], base + 1u));
    }
    gfq_red[lid] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            gfq_red[lid] = gfq_red[lid] + gfq_red[lid + s];
        }
        workgroupBarrier();
    }
    let rms = inverseSqrt(gfq_red[0] / f32(hd) + gfq_nr.eps);
    if (e0 < hd) {
        let we0 = w_off + e0;
        let w0 = bf16_decode(u16_at(gfq_w[we0 >> 1u], we0));
        let w1 = bf16_decode(u16_at(gfq_w[(we0 + 1u) >> 1u], we0 + 1u));
        gfq_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        gfq_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    var o0 = 0.0;
    var o1 = 0.0;
    if (e0 < hd) {
        o0 = gfq_buf[e0];
        o1 = gfq_buf[e0 + 1u];
        if (rope) {
            let rh = gfq_nr.rot_half;
            if (e0 < rh) {
                let c = gfq_cos[p * rh + e0];
                let s = gfq_sin[p * rh + e0];
                o0 = gfq_buf[e0] * c - gfq_buf[e0 + rh] * s;
            } else {
                let i0 = e0 - rh;
                let c = gfq_cos[p * rh + i0];
                let s = gfq_sin[p * rh + i0];
                o0 = gfq_buf[i0] * s + gfq_buf[e0] * c;
            }
            let e1 = e0 + 1u;
            if (e1 < rh) {
                let c = gfq_cos[p * rh + e1];
                let s = gfq_sin[p * rh + e1];
                o1 = gfq_buf[e1] * c - gfq_buf[e1 + rh] * s;
            } else {
                let i1 = e1 - rh;
                let c = gfq_cos[p * rh + i1];
                let s = gfq_sin[p * rh + i1];
                o1 = gfq_buf[i1] * s + gfq_buf[e1] * c;
            }
        }
    }
    return bf16_pack(o0, o1);
}

fn gfq_k_cache(idx: u32) -> f32 {
    return bf16_decode(u16_at(gfq_kc[idx >> 1u], idx));
}

fn gfq_v_at(vbase: u32, d: u32, cur: bool) -> f32 {
    if (cur) {
        return bf16_decode(u16_at(gfq_vcur[d >> 1u], d));
    }
    let idx = vbase + d;
    return bf16_decode(u16_at(gfq_vc[idx >> 1u], idx));
}

fn gfq_stage1_epilogue(lid: u32, lane: u32, warp: u32, hd: u32, slot: u32, m: f32, l: f32) {
    if (lane == 0u) {
        gfq_sm[warp] = m;
        gfq_sl[warp] = l;
    }
    workgroupBarrier();
    if (warp == 0u) {
        var m_blk = gfq_neg_inf();
        for (var w = 0u; w < GFQ_WARPS; w = w + 1u) {
            m_blk = max(m_blk, gfq_sm[w]);
        }
        var l_blk = 0.0;
        for (var w = 0u; w < GFQ_WARPS; w = w + 1u) {
            if (gfq_sm[w] > gfq_neg_inf()) {
                l_blk = l_blk + gfq_sl[w] * gfq_exp(gfq_sm[w] - m_blk);
            }
        }
        if (lane == 0u) {
            gfq_scratch[slot] = m_blk;
            gfq_scratch[slot + 1u] = l_blk;
        }
    }
    var m_blk = gfq_neg_inf();
    for (var w = 0u; w < GFQ_WARPS; w = w + 1u) {
        m_blk = max(m_blk, gfq_sm[w]);
    }
    for (var d = lid; d < hd; d = d + GFQ_BLOCK) {
        var a = 0.0;
        for (var w = 0u; w < GFQ_WARPS; w = w + 1u) {
            if (gfq_sm[w] > gfq_neg_inf()) {
                a = a + gfq_sacc[w * GFQ_MAX_HD + d] * gfq_exp(gfq_sm[w] - m_blk);
            }
        }
        gfq_scratch[slot + 2u + d] = a;
    }
}

@compute @workgroup_size(256)
fn g4m_flash_stage1_bf16_qkvfold(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= gfq_p.n_heads) {
        return;
    }
    let hd = gfq_p.head_dim;
    let nkv = gfq_p.n_kv;
    let group = gfq_p.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let pos = gfq_p.total - 1u;
    let e0 = 2u * lid;

    let qw = gfq_row_norm_rope(lid, h, 0u, true, pos);
    if (e0 < hd) {
        gfq_qsh[e0] = bf16_decode(u16_at(qw, e0));
        gfq_qsh[e0 + 1u] = bf16_decode(u16_at(qw, e0 + 1u));
    }
    let kw = gfq_row_norm_rope(lid, gfq_nr.n_q + kvh, hd, true, pos);
    if (e0 < hd) {
        gfq_kcur[e0 >> 1u] = kw;
    }
    var vsrc = gfq_nr.n_q + kvh;
    if (gfq_nr.has_rope == 1u) {
        vsrc = gfq_nr.n_q + gfq_nr.n_kv + kvh;
    }
    let vw = gfq_row_norm_rope(lid, vsrc, 2u * hd, false, pos);
    if (e0 < hd) {
        gfq_vcur[e0 >> 1u] = vw;
    }
    if (split == 0u && h == kvh * group && e0 < hd) {
        var wslot = pos;
        if (gfq_kv.ring > 0u) {
            wslot = pos % gfq_kv.ring;
        }
        let cbase = wslot * gfq_kv.words;
        gfq_kc[cbase + ((kvh * hd + e0) >> 1u)] = kw;
        gfq_vc[cbase + ((kvh * hd + e0) >> 1u)] = vw;
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
    var m = gfq_neg_inf();
    var l = 0.0;

    let total = gfq_p.total;
    let pbase = gfq_p.start + split * GFQ_WARPS;
    let stride = gfq_p.splits * GFQ_WARPS;
    var rounds = 0u;
    if (total > pbase) {
        rounds = (total - pbase + stride - 1u) / stride;
    }
    let use_vec8 = (hd & 7u) == 0u;

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = pbase + warp + r * stride;
        let live = p < total;
        let cur = live && (p + 1u == total);
        var sp = p;
        if (gfq_p.ring > 0u) {
            sp = p % gfq_p.ring;
        }
        var partial = 0.0;
        if (live) {
            let kbase = (sp * nkv + kvh) * hd;
            if (cur) {
                if (use_vec8) {
                    let n8 = hd >> 3u;
                    for (var j = lane; j < n8; j = j + GFQ_LANES) {
                        let qb = j * 8u;
                        for (var t = 0u; t < 4u; t = t + 1u) {
                            let d0 = qb + 2u * t;
                            let kx = bf16_decode(u16_at(gfq_kcur[d0 >> 1u], d0));
                            let ky = bf16_decode(u16_at(gfq_kcur[(d0 + 1u) >> 1u], d0 + 1u));
                            let pair = fma(kx, gfq_qsh[d0], ky * gfq_qsh[d0 + 1u]);
                            partial = partial + pair;
                        }
                    }
                } else {
                    for (var d = lane; d < hd; d = d + GFQ_LANES) {
                        partial = fma(
                            gfq_qsh[d],
                            bf16_decode(u16_at(gfq_kcur[d >> 1u], d)),
                            partial
                        );
                    }
                }
            } else {
                if (use_vec8) {
                    let n8 = hd >> 3u;
                    for (var j = lane; j < n8; j = j + GFQ_LANES) {
                        let qb = j * 8u;
                        let kb = kbase + qb;
                        for (var t = 0u; t < 4u; t = t + 1u) {
                            let kx = gfq_k_cache(kb + 2u * t);
                            let ky = gfq_k_cache(kb + 2u * t + 1u);
                            let pair = fma(kx, gfq_qsh[qb + 2u * t], ky * gfq_qsh[qb + 2u * t + 1u]);
                            partial = partial + pair;
                        }
                    }
                } else {
                    for (var d = lane; d < hd; d = d + GFQ_LANES) {
                        partial = fma(gfq_qsh[d], gfq_k_cache(kbase + d), partial);
                    }
                }
            }
        }
        let score = gfq_warp_sum(lid, partial) * gfq_p.scale;
        if (live) {
            let m_new = max(m, score);
            let corr = gfq_exp(m - m_new);
            let w = gfq_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (sp * nkv + kvh) * hd;
            {
                let d = lane + 0u * GFQ_LANES;
                if (d < hd) {
                    acc0 = fma(acc0, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 1u * GFQ_LANES;
                if (d < hd) {
                    acc1 = fma(acc1, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 2u * GFQ_LANES;
                if (d < hd) {
                    acc2 = fma(acc2, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 3u * GFQ_LANES;
                if (d < hd) {
                    acc3 = fma(acc3, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 4u * GFQ_LANES;
                if (d < hd) {
                    acc4 = fma(acc4, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 5u * GFQ_LANES;
                if (d < hd) {
                    acc5 = fma(acc5, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 6u * GFQ_LANES;
                if (d < hd) {
                    acc6 = fma(acc6, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 7u * GFQ_LANES;
                if (d < hd) {
                    acc7 = fma(acc7, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 8u * GFQ_LANES;
                if (d < hd) {
                    acc8 = fma(acc8, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 9u * GFQ_LANES;
                if (d < hd) {
                    acc9 = fma(acc9, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 10u * GFQ_LANES;
                if (d < hd) {
                    acc10 = fma(acc10, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 11u * GFQ_LANES;
                if (d < hd) {
                    acc11 = fma(acc11, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 12u * GFQ_LANES;
                if (d < hd) {
                    acc12 = fma(acc12, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 13u * GFQ_LANES;
                if (d < hd) {
                    acc13 = fma(acc13, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 14u * GFQ_LANES;
                if (d < hd) {
                    acc14 = fma(acc14, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            {
                let d = lane + 15u * GFQ_LANES;
                if (d < hd) {
                    acc15 = fma(acc15, corr, w * gfq_v_at(vbase, d, cur));
                }
            }
            m = m_new;
        }
    }

    {
        let d = lane + 0u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc0;
        }
    }
    {
        let d = lane + 1u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc1;
        }
    }
    {
        let d = lane + 2u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc2;
        }
    }
    {
        let d = lane + 3u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc3;
        }
    }
    {
        let d = lane + 4u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc4;
        }
    }
    {
        let d = lane + 5u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc5;
        }
    }
    {
        let d = lane + 6u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc6;
        }
    }
    {
        let d = lane + 7u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc7;
        }
    }
    {
        let d = lane + 8u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc8;
        }
    }
    {
        let d = lane + 9u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc9;
        }
    }
    {
        let d = lane + 10u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc10;
        }
    }
    {
        let d = lane + 11u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc11;
        }
    }
    {
        let d = lane + 12u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc12;
        }
    }
    {
        let d = lane + 13u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc13;
        }
    }
    {
        let d = lane + 14u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc14;
        }
    }
    {
        let d = lane + 15u * GFQ_LANES;
        if (d < hd) {
            gfq_sacc[warp * GFQ_MAX_HD + d] = acc15;
        }
    }
    let slot = (h * gfq_p.splits + split) * (hd + 2u);
    gfq_stage1_epilogue(lid, lane, warp, hd, slot, m, l);
}
