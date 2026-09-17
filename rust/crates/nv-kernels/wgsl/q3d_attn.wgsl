
struct Q3arParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> ar_src: array<u32>;
@group(0) @binding(1) var<storage, read> ar_w: array<u32>;
@group(0) @binding(2) var<storage, read> ar_cos: array<f32>;
@group(0) @binding(3) var<storage, read> ar_sin: array<f32>;
@group(0) @binding(4) var<storage, read> ar_pos: array<i32>;
@group(0) @binding(5) var<storage, read_write> ar_out: array<u32>;
@group(0) @binding(6) var<uniform> ar_p: Q3arParams;

var<workgroup> ar_red: array<f32, 128>;
var<workgroup> ar_buf: array<f32, 256>;

fn ar_rope_at_rh(d: u32, p: u32, rh: u32) -> f32 {
    if (d < rh) {
        let c = ar_cos[p * rh + d];
        let s = ar_sin[p * rh + d];
        return ar_buf[d] * c - ar_buf[d + rh] * s;
    }
    if (d < 2u * rh) {
        let i = d - rh;
        let c = ar_cos[p * rh + i];
        let s = ar_sin[p * rh + i];
        return ar_buf[i] * s + ar_buf[d] * c;
    }
    return ar_buf[d];
}

@compute @workgroup_size(128)
fn q3w_attn_norm_rope(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.x;
    let lane = lid.x;
    let hd = ar_p.head_dim;
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < hd) {
        let base = r * ar_p.src_stride + e0;
        v0 = bf16_decode(u16_at(ar_src[base >> 1u], base));
        v1 = bf16_decode(u16_at(ar_src[(base + 1u) >> 1u], base + 1u));
    }
    ar_red[lane] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            ar_red[lane] = ar_red[lane] + ar_red[lane + s];
        }
        workgroupBarrier();
    }
    let rms = inverseSqrt(ar_red[0] / f32(hd) + ar_p.eps);
    if (e0 < hd) {
        let w0 = bf16_decode(u16_at(ar_w[e0 >> 1u], e0));
        let w1 = bf16_decode(u16_at(ar_w[(e0 + 1u) >> 1u], e0 + 1u));
        ar_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        ar_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    if (e0 >= hd) {
        return;
    }
    var p = 0u;
    if (ar_pos[0] > 0) {
        p = u32(ar_pos[0]);
    }
    let o0 = ar_rope_at_rh(e0, p, ar_p.rot_half);
    let o1 = ar_rope_at_rh(e0 + 1u, p, ar_p.rot_half);
    ar_out[(r * hd + e0) >> 1u] = bf16_pack(o0, o1);
}

struct Q3afParams {
    n_q_rows: u32,
    n_k_rows: u32,
    head_dim: u32,
    q_src_stride: u32,
    k_src_stride: u32,
    rot_half: u32,
    pad0: u32,
    eps: f32,
};

@group(0) @binding(40) var<storage, read> af_ksrc: array<u32>;
@group(0) @binding(41) var<storage, read> af_kw: array<u32>;
@group(0) @binding(42) var<storage, read_write> af_kout: array<u32>;
@group(0) @binding(43) var<storage, read_write> af_qf32: array<f32>;
@group(0) @binding(44) var<uniform> af_p: Q3afParams;

fn af_src_word(is_q: bool, i: u32) -> u32 {
    if (is_q) {
        return ar_src[i];
    }
    return af_ksrc[i];
}

fn af_w_word(is_q: bool, i: u32) -> u32 {
    if (is_q) {
        return ar_w[i];
    }
    return af_kw[i];
}

@compute @workgroup_size(128)
fn q3w_attn_qk_norm_rope_qcast(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let is_q = wid.x < af_p.n_q_rows;
    let r = select(wid.x - af_p.n_q_rows, wid.x, is_q);
    let lane = lid.x;
    let hd = af_p.head_dim;
    let stride = select(af_p.k_src_stride, af_p.q_src_stride, is_q);
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < hd) {
        let base = r * stride + e0;
        v0 = bf16_decode(u16_at(af_src_word(is_q, base >> 1u), base));
        v1 = bf16_decode(u16_at(af_src_word(is_q, (base + 1u) >> 1u), base + 1u));
    }
    ar_red[lane] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            ar_red[lane] = ar_red[lane] + ar_red[lane + s];
        }
        workgroupBarrier();
    }
    let rms = inverseSqrt(ar_red[0] / f32(hd) + af_p.eps);
    if (e0 < hd) {
        let w0 = bf16_decode(u16_at(af_w_word(is_q, e0 >> 1u), e0));
        let w1 = bf16_decode(u16_at(af_w_word(is_q, (e0 + 1u) >> 1u), e0 + 1u));
        ar_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        ar_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    if (e0 >= hd) {
        return;
    }
    var p = 0u;
    if (ar_pos[0] > 0) {
        p = u32(ar_pos[0]);
    }
    let o0 = ar_rope_at_rh(e0, p, af_p.rot_half);
    let o1 = ar_rope_at_rh(e0 + 1u, p, af_p.rot_half);
    let packed = bf16_pack(o0, o1);
    let yi = r * hd + e0;
    if (is_q) {
        ar_out[yi >> 1u] = packed;
        af_qf32[yi] = bf16_lo(packed);
        af_qf32[yi + 1u] = bf16_hi(packed);
    } else {
        af_kout[yi >> 1u] = packed;
    }
}

struct Q3abParams {
    n_q_rows: u32,
    n_k_rows: u32,
    n_v_rows: u32,
    head_dim: u32,
    rot_half: u32,
    write_kv: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(45) var<storage, read> ab_bias: array<u32>;
@group(0) @binding(46) var<storage, read> ab_vsrc: array<u32>;
@group(0) @binding(47) var<storage, read_write> ab_vout: array<u32>;
@group(0) @binding(48) var<uniform> ab_p: Q3abParams;
@group(0) @binding(56) var<storage, read_write> ab_kc: array<u32>;
@group(0) @binding(57) var<storage, read_write> ab_vc: array<u32>;

fn ab_src_word(sel: u32, i: u32) -> u32 {
    if (sel == 0u) {
        return ar_src[i];
    }
    if (sel == 1u) {
        return af_ksrc[i];
    }
    return ab_vsrc[i];
}

@compute @workgroup_size(128)
fn q3w_attn_bias_rope_qcast(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let nq = ab_p.n_q_rows;
    let nk = ab_p.n_k_rows;
    var sel = 0u;
    var r = wid.x;
    if (wid.x >= nq + nk) {
        sel = 2u;
        r = wid.x - nq - nk;
    } else if (wid.x >= nq) {
        sel = 1u;
        r = wid.x - nq;
    }
    let lane = lid.x;
    let hd = ab_p.head_dim;
    let e0 = 2u * lane;
    if (e0 < hd) {
        let base = r * hd + e0;
        let v0 = bf16_decode(u16_at(ab_src_word(sel, base >> 1u), base));
        let v1 = bf16_decode(u16_at(ab_src_word(sel, (base + 1u) >> 1u), base + 1u));
        let bb = wid.x * hd + e0;
        let b0 = bf16_decode(u16_at(ab_bias[bb >> 1u], bb));
        let b1 = bf16_decode(u16_at(ab_bias[(bb + 1u) >> 1u], bb + 1u));
        ar_buf[e0] = bf16_decode(bf16_encode(v0 + b0));
        ar_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 + b1));
    }
    workgroupBarrier();
    if (e0 >= hd) {
        return;
    }
    var p = 0u;
    if (ar_pos[0] > 0) {
        p = u32(ar_pos[0]);
    }
    var rh = ab_p.rot_half;
    if (sel == 2u) {
        rh = 0u;
    }
    let o0 = ar_rope_at_rh(e0, p, rh);
    let o1 = ar_rope_at_rh(e0 + 1u, p, rh);
    let packed = bf16_pack(o0, o1);
    let yi = r * hd + e0;
    if (sel == 0u) {
        ar_out[yi >> 1u] = packed;
        af_qf32[yi] = bf16_lo(packed);
        af_qf32[yi + 1u] = bf16_hi(packed);
    } else if (sel == 1u) {
        af_kout[yi >> 1u] = packed;
        if (ab_p.write_kv == 1u) {
            ab_kc[(p * ab_p.n_k_rows * hd + yi) >> 1u] = packed;
        }
    } else {
        ab_vout[yi >> 1u] = packed;
        if (ab_p.write_kv == 1u) {
            ab_vc[(p * ab_p.n_v_rows * hd + yi) >> 1u] = packed;
        }
    }
}

struct Q3kvParams {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> kw_k: array<u32>;
@group(0) @binding(11) var<storage, read> kw_v: array<u32>;
@group(0) @binding(12) var<storage, read_write> kw_kc: array<u32>;
@group(0) @binding(13) var<storage, read_write> kw_vc: array<u32>;
@group(0) @binding(14) var<storage, read> kw_pos: array<i32>;
@group(0) @binding(15) var<uniform> kw_p: Q3kvParams;

@compute @workgroup_size(64)
fn q3w_kv_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= kw_p.words) {
        return;
    }
    var p = 0u;
    if (kw_pos[0] > 0) {
        p = u32(kw_pos[0]);
    }
    let base = p * kw_p.words;
    kw_kc[base + i] = kw_k[i];
    kw_vc[base + i] = kw_v[i];
}

struct Q3adParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
};

@group(0) @binding(20) var<storage, read> ad_q: array<u32>;
@group(0) @binding(21) var<storage, read> ad_kc: array<u32>;
@group(0) @binding(22) var<storage, read> ad_vc: array<u32>;
@group(0) @binding(23) var<storage, read_write> ad_scores: array<f32>;
@group(0) @binding(24) var<storage, read_write> ad_out: array<f32>;
@group(0) @binding(25) var<storage, read> ad_pos: array<i32>;
@group(0) @binding(26) var<uniform> ad_p: Q3adParams;

var<workgroup> ad_qs: array<f32, 256>;
var<workgroup> ad_red: array<f32, 256>;
var<workgroup> ad_m: f32;
var<workgroup> ad_z: f32;

@compute @workgroup_size(256)
fn q3w_attn_decode(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tid = lid.x;
    let hd = ad_p.head_dim;
    var p = 0u;
    if (ad_pos[0] > 0) {
        p = u32(ad_pos[0]);
    }
    let total = min(p + 1u, ad_p.max_seq);
    let kv = h / ad_p.group;
    let srow = h * ad_p.max_seq;

    for (var d = tid; d < hd; d = d + 256u) {
        let idx = h * hd + d;
        ad_qs[d] = bf16_decode(u16_at(ad_q[idx >> 1u], idx));
    }
    workgroupBarrier();

    var lmax = -3.4028235e38;
    for (var t = tid; t < total; t = t + 256u) {
        let kbase = (t * ad_p.n_kv + kv) * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let idx = kbase + d;
            dot = fma(bf16_decode(u16_at(ad_kc[idx >> 1u], idx)), ad_qs[d], dot);
        }
        let s = dot * ad_p.scale;
        ad_scores[srow + t] = s;
        lmax = max(lmax, s);
    }
    ad_red[tid] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            ad_red[tid] = max(ad_red[tid], ad_red[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        ad_m = ad_red[0];
    }
    workgroupBarrier();
    let m = ad_m;

    var lsum = 0.0;
    for (var t = tid; t < total; t = t + 256u) {
        let e = exp(ad_scores[srow + t] - m);
        ad_scores[srow + t] = e;
        lsum = lsum + e;
    }
    workgroupBarrier();
    ad_red[tid] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            ad_red[tid] = ad_red[tid] + ad_red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        ad_z = ad_red[0];
    }
    workgroupBarrier();
    let z = ad_z;

    if (tid < hd) {
        var acc = 0.0;
        for (var t = 0u; t < total; t = t + 1u) {
            let idx = (t * ad_p.n_kv + kv) * hd + tid;
            acc = fma(ad_scores[srow + t], bf16_decode(u16_at(ad_vc[idx >> 1u], idx)), acc);
        }
        ad_out[h * hd + tid] = acc / z;
    }
}

struct Q3asParams {
    n_kv: u32,
    head_dim: u32,
    group: u32,
    rot_half: u32,
    scale: f32,
    splits: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(50) var<storage, read> as_q: array<f32>;
@group(0) @binding(53) var<storage, read> as_pos: array<i32>;
@group(0) @binding(54) var<storage, read_write> as_out: array<u32>;
@group(0) @binding(55) var<uniform> as_p: Q3asParams;

var<workgroup> as_acc: array<f32, 1024>;
var<workgroup> as_mz: array<f32, 16>;
var<workgroup> am_stage: array<f32, 256>;
var<workgroup> am_qs: array<f32, 128>;
var<workgroup> am_ks: array<f32, 128>;
var<workgroup> am_vs: array<f32, 128>;

fn am_rope_stage(base: u32, d: u32, pp: u32, rh: u32) -> f32 {
    if (d < rh) {
        let c = ar_cos[pp * rh + d];
        let s = ar_sin[pp * rh + d];
        return am_stage[base + d] * c - am_stage[base + d + rh] * s;
    }
    if (d < 2u * rh) {
        let i = d - rh;
        let c = ar_cos[pp * rh + i];
        let s = ar_sin[pp * rh + i];
        return am_stage[base + i] * s + am_stage[base + d] * c;
    }
    return am_stage[base + d];
}

@compute @workgroup_size(256)
fn q3w_attn_bias_rope_decode_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let hd = as_p.head_dim;
    let rh = as_p.rot_half;
    let n_kv = as_p.n_kv;
    let n_h = n_kv * as_p.group;
    var p = 0u;
    if (ar_pos[0] > 0) {
        p = u32(ar_pos[0]);
    }
    let total = p + 1u;
    let kvh = h / as_p.group;

    if (tid < hd) {
        let qi = h * hd + tid;
        let qb = qi;
        am_stage[tid] = bf16_decode(bf16_encode(
            bf16_decode(u16_at(ar_src[qi >> 1u], qi))
                + bf16_decode(u16_at(ab_bias[qb >> 1u], qb))
        ));
    } else if (tid < 2u * hd) {
        let d = tid - hd;
        let ki = kvh * hd + d;
        let kb = (n_h + kvh) * hd + d;
        am_stage[tid] = bf16_decode(bf16_encode(
            bf16_decode(u16_at(af_ksrc[ki >> 1u], ki))
                + bf16_decode(u16_at(ab_bias[kb >> 1u], kb))
        ));
        let vb = (n_h + n_kv + kvh) * hd + d;
        am_vs[d] = bf16_decode(bf16_encode(
            bf16_decode(u16_at(ab_vsrc[ki >> 1u], ki))
                + bf16_decode(u16_at(ab_bias[vb >> 1u], vb))
        ));
    }
    workgroupBarrier();
    if (tid < hd) {
        am_qs[tid] = bf16_decode(bf16_encode(am_rope_stage(0u, tid, p, rh)));
    } else if (tid < 2u * hd) {
        let d = tid - hd;
        am_ks[d] = bf16_decode(bf16_encode(am_rope_stage(hd, d, p, rh)));
    }
    workgroupBarrier();

    if (h == kvh * as_p.group && tid < hd / 2u) {
        let d = tid * 2u;
        let ci = (p * n_kv + kvh) * hd + d;
        ab_kc[ci >> 1u] = bf16_pack(am_ks[d], am_ks[d + 1u]);
        ab_vc[ci >> 1u] = bf16_pack(am_vs[d], am_vs[d + 1u]);
    }

    let epl = hd >> 5u;
    let d0 = lane * epl;
    var q0 = 0.0;
    var q1 = 0.0;
    var q2 = 0.0;
    var q3 = 0.0;
    if (epl == 4u) {
        q0 = am_qs[d0];
        q1 = am_qs[d0 + 1u];
        q2 = am_qs[d0 + 2u];
        q3 = am_qs[d0 + 3u];
    }

    var m = -3.4028235e38;
    var z = 0.0;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    for (var t = warp; t < total; t = t + 8u) {
        var k0 = 0.0;
        var k1 = 0.0;
        var k2 = 0.0;
        var k3 = 0.0;
        var v0 = 0.0;
        var v1 = 0.0;
        var v2 = 0.0;
        var v3 = 0.0;
        if (t == p) {
            k0 = am_ks[d0];
            k1 = am_ks[d0 + 1u];
            k2 = am_ks[d0 + 2u];
            k3 = am_ks[d0 + 3u];
            v0 = am_vs[d0];
            v1 = am_vs[d0 + 1u];
            v2 = am_vs[d0 + 2u];
            v3 = am_vs[d0 + 3u];
        } else {
            let base = (t * n_kv + kvh) * hd + d0;
            let kw0 = ab_kc[base >> 1u];
            let kw1 = ab_kc[(base >> 1u) + 1u];
            k0 = bf16_lo(kw0);
            k1 = bf16_hi(kw0);
            k2 = bf16_lo(kw1);
            k3 = bf16_hi(kw1);
            let vw0 = ab_vc[base >> 1u];
            let vw1 = ab_vc[(base >> 1u) + 1u];
            v0 = bf16_lo(vw0);
            v1 = bf16_hi(vw0);
            v2 = bf16_lo(vw1);
            v3 = bf16_hi(vw1);
        }
        var d = fma(k0, q0, 0.0);
        d = fma(k1, q1, d);
        d = fma(k2, q2, d);
        d = fma(k3, q3, d);
        d = d + subgroupShuffleXor(d, 16u);
        d = d + subgroupShuffleXor(d, 8u);
        d = d + subgroupShuffleXor(d, 4u);
        d = d + subgroupShuffleXor(d, 2u);
        d = d + subgroupShuffleXor(d, 1u);
        let s = d * as_p.scale;
        if (s > m) {
            let r = exp(m - s);
            z = fma(z, r, 1.0);
            a0 = fma(a0, r, v0);
            a1 = fma(a1, r, v1);
            a2 = fma(a2, r, v2);
            a3 = fma(a3, r, v3);
            m = s;
        } else {
            let e = exp(s - m);
            z = z + e;
            a0 = fma(e, v0, a0);
            a1 = fma(e, v1, a1);
            a2 = fma(e, v2, a2);
            a3 = fma(e, v3, a3);
        }
    }

    let wb = warp * 128u + d0;
    as_acc[wb] = a0;
    as_acc[wb + 1u] = a1;
    as_acc[wb + 2u] = a2;
    as_acc[wb + 3u] = a3;
    if (lane == 0u) {
        as_mz[warp] = m;
        as_mz[warp + 8u] = z;
    }
    workgroupBarrier();

    if (warp == 0u && epl == 4u) {
        var gm = -3.4028235e38;
        for (var w = 0u; w < 8u; w = w + 1u) {
            gm = max(gm, as_mz[w]);
        }
        var gz = 0.0;
        var o0 = 0.0;
        var o1 = 0.0;
        var o2 = 0.0;
        var o3 = 0.0;
        for (var w = 0u; w < 8u; w = w + 1u) {
            let mw = as_mz[w];
            if (mw > -3.0e38) {
                let r = exp(mw - gm);
                gz = fma(as_mz[w + 8u], r, gz);
                let b = w * 128u + d0;
                o0 = fma(as_acc[b], r, o0);
                o1 = fma(as_acc[b + 1u], r, o1);
                o2 = fma(as_acc[b + 2u], r, o2);
                o3 = fma(as_acc[b + 3u], r, o3);
            }
        }
        let inv = 1.0 / gz;
        let yb = (h * hd + d0) >> 1u;
        as_out[yb] = bf16_pack(o0 * inv, o1 * inv);
        as_out[yb + 1u] = bf16_pack(o2 * inv, o3 * inv);
    }
}


@compute @workgroup_size(256)
fn q3w_attn_decode_shallow_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let hd = as_p.head_dim;
    let epl = hd >> 5u;
    var p = 0u;
    if (as_pos[0] > 0) {
        p = u32(as_pos[0]);
    }
    let total = p + 1u;
    let kvh = h / as_p.group;
    let d0 = lane * epl;

    var q0 = 0.0;
    var q1 = 0.0;
    var q2 = 0.0;
    var q3 = 0.0;
    if (epl == 4u) {
        let qb = h * hd + d0;
        q0 = as_q[qb];
        q1 = as_q[qb + 1u];
        q2 = as_q[qb + 2u];
        q3 = as_q[qb + 3u];
    }

    var m = -3.4028235e38;
    var z = 0.0;
    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    for (var t = warp; t < total; t = t + 8u) {
        let base = (t * as_p.n_kv + kvh) * hd + d0;
        let kw0 = ab_kc[base >> 1u];
        let kw1 = ab_kc[(base >> 1u) + 1u];
        var d = fma(bf16_lo(kw0), q0, 0.0);
        d = fma(bf16_hi(kw0), q1, d);
        d = fma(bf16_lo(kw1), q2, d);
        d = fma(bf16_hi(kw1), q3, d);
        d = d + subgroupShuffleXor(d, 16u);
        d = d + subgroupShuffleXor(d, 8u);
        d = d + subgroupShuffleXor(d, 4u);
        d = d + subgroupShuffleXor(d, 2u);
        d = d + subgroupShuffleXor(d, 1u);
        let s = d * as_p.scale;
        let vw0 = ab_vc[base >> 1u];
        let vw1 = ab_vc[(base >> 1u) + 1u];
        if (s > m) {
            let r = exp(m - s);
            z = fma(z, r, 1.0);
            a0 = fma(a0, r, bf16_lo(vw0));
            a1 = fma(a1, r, bf16_hi(vw0));
            a2 = fma(a2, r, bf16_lo(vw1));
            a3 = fma(a3, r, bf16_hi(vw1));
            m = s;
        } else {
            let e = exp(s - m);
            z = z + e;
            a0 = fma(e, bf16_lo(vw0), a0);
            a1 = fma(e, bf16_hi(vw0), a1);
            a2 = fma(e, bf16_lo(vw1), a2);
            a3 = fma(e, bf16_hi(vw1), a3);
        }
    }

    let wb = warp * 128u + d0;
    as_acc[wb] = a0;
    as_acc[wb + 1u] = a1;
    as_acc[wb + 2u] = a2;
    as_acc[wb + 3u] = a3;
    if (lane == 0u) {
        as_mz[warp] = m;
        as_mz[warp + 8u] = z;
    }
    workgroupBarrier();

    if (warp == 0u && epl == 4u) {
        var gm = -3.4028235e38;
        for (var w = 0u; w < 8u; w = w + 1u) {
            gm = max(gm, as_mz[w]);
        }
        var gz = 0.0;
        var o0 = 0.0;
        var o1 = 0.0;
        var o2 = 0.0;
        var o3 = 0.0;
        for (var w = 0u; w < 8u; w = w + 1u) {
            let mw = as_mz[w];
            if (mw > -3.0e38) {
                let r = exp(mw - gm);
                gz = fma(as_mz[w + 8u], r, gz);
                let b = w * 128u + d0;
                o0 = fma(as_acc[b], r, o0);
                o1 = fma(as_acc[b + 1u], r, o1);
                o2 = fma(as_acc[b + 2u], r, o2);
                o3 = fma(as_acc[b + 3u], r, o3);
            }
        }
        let inv = 1.0 / gz;
        let yb = (h * hd + d0) >> 1u;
        as_out[yb] = bf16_pack(o0 * inv, o1 * inv);
        as_out[yb + 1u] = bf16_pack(o2 * inv, o3 * inv);
    }
}

@group(0) @binding(58) var<storage, read_write> amf_scratch: array<f32>;

var<workgroup> amf_stage: array<f32, 1024>;
var<workgroup> amf_qs: array<f32, 768>;
var<workgroup> amf_ks: array<f32, 128>;
var<workgroup> amf_vs: array<f32, 128>;
var<workgroup> amf_acc: array<f32, 768>;
var<workgroup> amf_mz: array<f32, 12>;

fn amf_rope_stage(base: u32, d: u32, pp: u32, rh: u32) -> f32 {
    if (d < rh) {
        let c = ar_cos[pp * rh + d];
        let s = ar_sin[pp * rh + d];
        return amf_stage[base + d] * c - amf_stage[base + d + rh] * s;
    }
    if (d < 2u * rh) {
        let i = d - rh;
        let c = ar_cos[pp * rh + i];
        let s = ar_sin[pp * rh + i];
        return amf_stage[base + i] * s + amf_stage[base + d] * c;
    }
    return amf_stage[base + d];
}

@compute @workgroup_size(256)
fn q3w_attn_fold_split1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let kvh = wid.x;
    let sp = wid.y;
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let hd = as_p.head_dim;
    let grp = as_p.group;
    let rh = as_p.rot_half;
    let n_kv = as_p.n_kv;
    let n_h = n_kv * grp;
    var p = 0u;
    if (ar_pos[0] > 0) {
        p = u32(ar_pos[0]);
    }
    let total = p + 1u;

    for (var e = tid; e < 8u * hd; e = e + 256u) {
        let r = e / hd;
        let d = e - r * hd;
        if (r < grp) {
            let h = kvh * grp + r;
            let qi = h * hd + d;
            amf_stage[e] = bf16_decode(bf16_encode(
                bf16_decode(u16_at(ar_src[qi >> 1u], qi))
                    + bf16_decode(u16_at(ab_bias[qi >> 1u], qi))
            ));
        } else if (r == grp) {
            let ki = kvh * hd + d;
            let kb = (n_h + kvh) * hd + d;
            amf_stage[e] = bf16_decode(bf16_encode(
                bf16_decode(u16_at(af_ksrc[ki >> 1u], ki))
                    + bf16_decode(u16_at(ab_bias[kb >> 1u], kb))
            ));
        } else if (r == grp + 1u) {
            let vi = kvh * hd + d;
            let vb = (n_h + n_kv + kvh) * hd + d;
            amf_vs[d] = bf16_decode(bf16_encode(
                bf16_decode(u16_at(ab_vsrc[vi >> 1u], vi))
                    + bf16_decode(u16_at(ab_bias[vb >> 1u], vb))
            ));
        }
    }
    workgroupBarrier();
    for (var e = tid; e < 7u * hd; e = e + 256u) {
        let r = e / hd;
        let d = e - r * hd;
        if (r < grp) {
            amf_qs[e] = bf16_decode(bf16_encode(amf_rope_stage(r * hd, d, p, rh))) * as_p.scale;
        } else if (r == grp) {
            amf_ks[d] = bf16_decode(bf16_encode(amf_rope_stage(grp * hd, d, p, rh)));
        }
    }
    workgroupBarrier();

    if (sp == 0u && tid < hd / 2u) {
        let d = tid * 2u;
        let ci = (p * n_kv + kvh) * hd + d;
        ab_kc[ci >> 1u] = bf16_pack(amf_ks[d], amf_ks[d + 1u]);
        ab_vc[ci >> 1u] = bf16_pack(amf_vs[d], amf_vs[d + 1u]);
    }

    let splits = max(as_p.splits, 1u);
    let slice = (total + splits - 1u) / splits;
    let start = sp * slice;
    let end = min(total, start + slice);

    let d0 = lane * 4u;
    var mv = vec4<f32>(-3.4028235e38);
    var m2 = vec2<f32>(-3.4028235e38);
    var zv = vec4<f32>(0.0);
    var z2 = vec2<f32>(0.0);
    var a0 = vec4<f32>(0.0);
    var a1 = vec4<f32>(0.0);
    var a2v = vec4<f32>(0.0);
    var a3 = vec4<f32>(0.0);
    var b0 = vec2<f32>(0.0);
    var b1 = vec2<f32>(0.0);
    var b2 = vec2<f32>(0.0);
    var b3 = vec2<f32>(0.0);

    for (var t = start + warp; t < end; t = t + 8u) {
        var k0 = 0.0;
        var k1 = 0.0;
        var k2 = 0.0;
        var k3 = 0.0;
        var v0 = 0.0;
        var v1 = 0.0;
        var v2 = 0.0;
        var v3 = 0.0;
        if (t == p) {
            k0 = amf_ks[d0];
            k1 = amf_ks[d0 + 1u];
            k2 = amf_ks[d0 + 2u];
            k3 = amf_ks[d0 + 3u];
            v0 = amf_vs[d0];
            v1 = amf_vs[d0 + 1u];
            v2 = amf_vs[d0 + 2u];
            v3 = amf_vs[d0 + 3u];
        } else {
            let base = (t * n_kv + kvh) * hd + d0;
            let kw0 = ab_kc[base >> 1u];
            let kw1 = ab_kc[(base >> 1u) + 1u];
            k0 = bf16_lo(kw0);
            k1 = bf16_hi(kw0);
            k2 = bf16_lo(kw1);
            k3 = bf16_hi(kw1);
            let vw0 = ab_vc[base >> 1u];
            let vw1 = ab_vc[(base >> 1u) + 1u];
            v0 = bf16_lo(vw0);
            v1 = bf16_hi(vw0);
            v2 = bf16_lo(vw1);
            v3 = bf16_hi(vw1);
        }
        var s4 = vec4<f32>(0.0);
        var s2 = vec2<f32>(0.0);
        for (var g = 0u; g < 4u; g = g + 1u) {
            let qb = g * hd + d0;
            s4[g] = fma(k0, amf_qs[qb], fma(k1, amf_qs[qb + 1u], fma(k2, amf_qs[qb + 2u], k3 * amf_qs[qb + 3u])));
        }
        for (var g = 0u; g < 2u; g = g + 1u) {
            let qb = (g + 4u) * hd + d0;
            s2[g] = fma(k0, amf_qs[qb], fma(k1, amf_qs[qb + 1u], fma(k2, amf_qs[qb + 2u], k3 * amf_qs[qb + 3u])));
        }
        s4 = s4 + subgroupShuffleXor(s4, 16u);
        s4 = s4 + subgroupShuffleXor(s4, 8u);
        s4 = s4 + subgroupShuffleXor(s4, 4u);
        s4 = s4 + subgroupShuffleXor(s4, 2u);
        s4 = s4 + subgroupShuffleXor(s4, 1u);
        s2 = s2 + subgroupShuffleXor(s2, 16u);
        s2 = s2 + subgroupShuffleXor(s2, 8u);
        s2 = s2 + subgroupShuffleXor(s2, 4u);
        s2 = s2 + subgroupShuffleXor(s2, 2u);
        s2 = s2 + subgroupShuffleXor(s2, 1u);
        let nm = max(mv, s4);
        let rr = exp(mv - nm);
        let ee = exp(s4 - nm);
        zv = fma(zv, rr, ee);
        a0 = fma(a0, rr, ee * v0);
        a1 = fma(a1, rr, ee * v1);
        a2v = fma(a2v, rr, ee * v2);
        a3 = fma(a3, rr, ee * v3);
        mv = nm;
        let nm2 = max(m2, s2);
        let rr2 = exp(m2 - nm2);
        let ee2 = exp(s2 - nm2);
        z2 = fma(z2, rr2, ee2);
        b0 = fma(b0, rr2, ee2 * v0);
        b1 = fma(b1, rr2, ee2 * v1);
        b2 = fma(b2, rr2, ee2 * v2);
        b3 = fma(b3, rr2, ee2 * v3);
        m2 = nm2;
    }

    for (var e = tid; e < 6u * hd; e = e + 256u) {
        amf_acc[e] = 0.0;
    }
    if (tid < 12u) {
        amf_mz[tid] = select(0.0, -3.4028235e38, tid < 6u);
    }
    workgroupBarrier();

    for (var w = 0u; w < 8u; w = w + 1u) {
        if (warp == w) {
            for (var g = 0u; g < 6u; g = g + 1u) {
                let mw = select(m2[g - 4u], mv[g], g < 4u);
                let zw = select(z2[g - 4u], zv[g], g < 4u);
                if (mw > -3.0e38) {
                    let gm = amf_mz[g];
                    let nm = max(gm, mw);
                    let rg = exp(gm - nm);
                    let rw = exp(mw - nm);
                    if (lane == 0u) {
                        amf_mz[g] = nm;
                        amf_mz[g + 6u] = amf_mz[g + 6u] * rg + zw * rw;
                    }
                    var av = vec4<f32>(0.0);
                    if (g < 4u) {
                        av = vec4<f32>(a0[g], a1[g], a2v[g], a3[g]);
                    } else {
                        av = vec4<f32>(b0[g - 4u], b1[g - 4u], b2[g - 4u], b3[g - 4u]);
                    }
                    let ab = g * hd + d0;
                    amf_acc[ab] = amf_acc[ab] * rg + av.x * rw;
                    amf_acc[ab + 1u] = amf_acc[ab + 1u] * rg + av.y * rw;
                    amf_acc[ab + 2u] = amf_acc[ab + 2u] * rg + av.z * rw;
                    amf_acc[ab + 3u] = amf_acc[ab + 3u] * rg + av.w * rw;
                }
            }
        }
        workgroupBarrier();
    }

    for (var e = tid; e < 6u * hd; e = e + 256u) {
        let g = e / hd;
        let d = e - g * hd;
        let h = kvh * grp + g;
        let sb = (h * as_p.splits + sp) * (hd + 2u);
        amf_scratch[sb + d] = amf_acc[e];
        if (d == 0u) {
            amf_scratch[sb + hd] = amf_mz[g];
            amf_scratch[sb + hd + 1u] = amf_mz[g + 6u];
        }
    }
}

@compute @workgroup_size(128)
fn q3w_attn_fold_merge_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tid = lid.x;
    let hd = as_p.head_dim;
    let splits = max(as_p.splits, 1u);
    var gm = -3.4028235e38;
    for (var sp = 0u; sp < splits; sp = sp + 1u) {
        gm = max(gm, amf_scratch[(h * splits + sp) * (hd + 2u) + hd]);
    }
    var gz = 0.0;
    var o = 0.0;
    for (var sp = 0u; sp < splits; sp = sp + 1u) {
        let sb = (h * splits + sp) * (hd + 2u);
        let ms = amf_scratch[sb + hd];
        if (ms > -3.0e38) {
            let r = exp(ms - gm);
            gz = gz + amf_scratch[sb + hd + 1u] * r;
            o = o + amf_scratch[sb + tid] * r;
        }
    }
    amf_stage[tid] = o / max(gz, 1e-30);
    workgroupBarrier();
    if (tid < hd / 2u) {
        let d = tid * 2u;
        as_out[(h * hd + d) >> 1u] = bf16_pack(amf_stage[d], amf_stage[d + 1u]);
    }
}

struct Q3agParams {
    n_words: u32,
    head_dim: u32,
    src_stride: u32,
    gate_off: u32,
    has_gate: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(30) var<storage, read> ag_attn: array<f32>;
@group(0) @binding(31) var<storage, read> ag_qraw: array<u32>;
@group(0) @binding(32) var<storage, read_write> ag_out: array<u32>;
@group(0) @binding(33) var<uniform> ag_p: Q3agParams;

@compute @workgroup_size(64)
fn q3w_attn_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= ag_p.n_words) {
        return;
    }
    let e0 = w * 2u;
    let a0 = bf16_decode(bf16_encode(ag_attn[e0]));
    let a1 = bf16_decode(bf16_encode(ag_attn[e0 + 1u]));
    if (ag_p.has_gate == 0u) {
        ag_out[w] = bf16_pack(a0, a1);
        return;
    }
    let h = e0 / ag_p.head_dim;
    let d = e0 % ag_p.head_dim;
    let gb = h * ag_p.src_stride + ag_p.gate_off + d;
    let g0 = bf16_decode(u16_at(ag_qraw[gb >> 1u], gb));
    let g1 = bf16_decode(u16_at(ag_qraw[(gb + 1u) >> 1u], gb + 1u));
    let s0 = bf16_decode(bf16_encode(1.0 / (1.0 + exp(-g0))));
    let s1 = bf16_decode(bf16_encode(1.0 / (1.0 + exp(-g1))));
    ag_out[w] = bf16_pack(a0 * s0, a1 * s1);
}
