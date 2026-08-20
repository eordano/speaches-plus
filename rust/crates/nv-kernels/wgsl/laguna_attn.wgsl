
const LGA_NR_WG: u32 = 128u;
const LGA_NR_MAXW: u32 = 128u;
const LGA_AD_WG: u32 = 256u;
const LGA_AD_MAXHD: u32 = 256u;
const LGA_STEP_POS: u32 = 1u;

struct LgaNormRopeParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    eps: f32,
    out_scale: f32,
    rope_table_half: u32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> lnr_src: array<u32>;
@group(0) @binding(1) var<storage, read> lnr_w: array<u32>;
@group(0) @binding(2) var<storage, read> lnr_cos: array<f32>;
@group(0) @binding(3) var<storage, read> lnr_sin: array<f32>;
@group(0) @binding(4) var<storage, read> lnr_step: array<u32>;
@group(0) @binding(5) var<uniform> lnr_p: LgaNormRopeParams;
@group(0) @binding(6) var<storage, read_write> lnr_dst: array<u32>;

var<workgroup> lnr_red: array<f32, LGA_NR_WG>;
var<workgroup> lnr_inv: f32;
var<workgroup> lnr_a: array<u32, LGA_NR_MAXW>;

fn lnr_at(elem: u32) -> f32 {
    let word = lnr_a[elem >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (elem & 1u) == 1u);
}

fn lnr_rot(row_base: u32, elem: u32, rh: u32, sc: f32) -> f32 {
    if (elem >= rh + rh) {
        return lnr_at(elem);
    }
    if (elem < rh) {
        let c = lnr_cos[row_base + elem];
        let s = lnr_sin[row_base + elem];
        let a = lnr_at(elem);
        let b = lnr_at(elem + rh);
        return (a * c - b * s) * sc;
    }
    let pair = elem - rh;
    let c = lnr_cos[row_base + pair];
    let s = lnr_sin[row_base + pair];
    let a = lnr_at(pair);
    let b = lnr_at(elem);
    return (a * s + b * c) * sc;
}

@compute @workgroup_size(128)
fn lgw_norm_rope(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= lnr_p.n_rows) {
        return;
    }
    let lid = tid.x;
    let hd = lnr_p.head_dim;
    let words = hd >> 1u;
    let base = row * lnr_p.src_stride;

    var local = 0.0;
    for (var i = lid; i < hd; i = i + LGA_NR_WG) {
        let word = lnr_src[base + (i >> 1u)];
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local = fma(v, v, local);
    }
    lnr_red[lid] = local;
    workgroupBarrier();
    for (var s = LGA_NR_WG >> 1u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            lnr_red[lid] = lnr_red[lid] + lnr_red[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        lnr_inv = 1.0 / sqrt(lnr_red[0] / f32(hd) + lnr_p.eps);
    }
    workgroupBarrier();
    let inv = lnr_inv;
    for (var i = lid; i < words; i = i + LGA_NR_WG) {
        let xw = lnr_src[base + i];
        let ww = lnr_w[i];
        lnr_a[i] = bf16_pack(bf16_lo(xw) * inv * bf16_lo(ww), bf16_hi(xw) * inv * bf16_hi(ww));
    }
    workgroupBarrier();

    let row_base = lnr_step[LGA_STEP_POS] * lnr_p.rope_table_half;
    let rh = lnr_p.rot_half;
    let sc = lnr_p.out_scale;
    for (var i = lid; i < words; i = i + LGA_NR_WG) {
        let e = i * 2u;
        lnr_dst[base + i] = bf16_pack(lnr_rot(row_base, e, rh, sc), lnr_rot(row_base, e + 1u, rh, sc));
    }
}

struct LgaKvWriteParams {
    kv_words: u32,
    kv_capacity: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(10) var<storage, read> lkv_src: array<u32>;
@group(0) @binding(11) var<storage, read> lkv_step: array<u32>;
@group(0) @binding(12) var<uniform> lkv_p: LgaKvWriteParams;
@group(0) @binding(13) var<storage, read_write> lkv_dst: array<u32>;

@compute @workgroup_size(64)
fn lgw_kv_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= lkv_p.kv_words) {
        return;
    }
    let slot = lkv_step[LGA_STEP_POS];
    if (slot >= lkv_p.kv_capacity) {
        return;
    }
    lkv_dst[slot * lkv_p.kv_words + i] = lkv_src[i];
}

struct LgaDecodeParams {
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    gqa_group: u32,
    kv_capacity: u32,
    is_sliding: u32,
    scale: f32,
    pad0: u32,
};

struct LgaStepU {
    tok: u32,
    pos: u32,
    total: u32,
    sstart: u32,
};

@group(0) @binding(20) var<storage, read> lad_q: array<u32>;
@group(0) @binding(21) var<storage, read> lad_k: array<u32>;
@group(0) @binding(22) var<storage, read> lad_v: array<u32>;
@group(0) @binding(23) var<uniform> lad_step: LgaStepU;
@group(0) @binding(24) var<uniform> lad_par: LgaDecodeParams;
@group(0) @binding(25) var<storage, read_write> lad_out: array<u32>;

var<workgroup> lad_qs: array<f32, LGA_AD_MAXHD>;
var<workgroup> lad_acc: array<f32, LGA_AD_MAXHD>;
var<workgroup> lad_pr: array<f32, LGA_AD_WG>;
var<workgroup> lad_red: array<f32, LGA_AD_WG>;

fn lad_kval(idx: u32) -> f32 {
    let word = lad_k[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn lad_vval(idx: u32) -> f32 {
    let word = lad_v[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn lad_score(kbase: u32, hd: u32) -> f32 {
    var a = 0.0;
    for (var d = 0u; d < hd; d = d + 1u) {
        a = fma(lad_qs[d], lad_kval(kbase + d), a);
    }
    return a;
}

@compute @workgroup_size(256)
fn lgw_attn_decode(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x + wg.y * nwg.x;
    if (h >= lad_par.n_q_heads) {
        return;
    }
    let lid = tid.x;
    let hd = lad_par.head_dim;
    let words = hd >> 1u;
    let nkv = lad_par.n_kv_heads;
    let kvh = h / lad_par.gqa_group;
    let cap = lad_par.kv_capacity;
    let total = min(lad_step.total, cap);
    var start = 0u;
    if (lad_par.is_sliding == 1u) {
        start = min(lad_step.sstart, total);
    }

    for (var i = lid; i < hd; i = i + LGA_AD_WG) {
        let word = lad_q[h * words + (i >> 1u)];
        lad_qs[i] = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        lad_acc[i] = 0.0;
    }
    workgroupBarrier();

    var lmax = -3.0e38;
    for (var j = start + lid; j < total; j = j + LGA_AD_WG) {
        let s = lad_score((j * nkv + kvh) * hd, hd) * lad_par.scale;
        lmax = max(lmax, s);
    }
    lad_red[lid] = lmax;
    workgroupBarrier();
    for (var s = LGA_AD_WG >> 1u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            lad_red[lid] = max(lad_red[lid], lad_red[lid + s]);
        }
        workgroupBarrier();
    }
    let m = lad_red[0];

    var lsum = 0.0;
    let tile0 = (start / LGA_AD_WG) * LGA_AD_WG;
    for (var base = tile0; base < total; base = base + LGA_AD_WG) {
        let j = base + lid;
        var pv = 0.0;
        if (j >= start && j < total) {
            pv = exp(lad_score((j * nkv + kvh) * hd, hd) * lad_par.scale - m);
        }
        lsum = lsum + pv;
        lad_pr[lid] = pv;
        workgroupBarrier();
        let tmax = min(LGA_AD_WG, total - base);
        for (var d = lid; d < hd; d = d + LGA_AD_WG) {
            var a = 0.0;
            for (var t = 0u; t < tmax; t = t + 1u) {
                a = fma(lad_pr[t], lad_vval(((base + t) * nkv + kvh) * hd + d), a);
            }
            lad_acc[d] = lad_acc[d] + a;
        }
        workgroupBarrier();
    }

    lad_red[lid] = lsum;
    workgroupBarrier();
    for (var s = LGA_AD_WG >> 1u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            lad_red[lid] = lad_red[lid] + lad_red[lid + s];
        }
        workgroupBarrier();
    }
    var inv = 0.0;
    if (lad_red[0] > 0.0) {
        inv = 1.0 / lad_red[0];
    }
    for (var i = lid; i < words; i = i + LGA_AD_WG) {
        let e = i * 2u;
        lad_out[h * words + i] = bf16_pack(lad_acc[e] * inv, lad_acc[e + 1u] * inv);
    }
}

struct LgaGateParams {
    n_words: u32,
    head_dim: u32,
    gate_kind: u32,
    n_q_heads: u32,
};

@group(0) @binding(30) var<storage, read> lag_x: array<u32>;
@group(0) @binding(31) var<storage, read> lag_g: array<u32>;
@group(0) @binding(32) var<uniform> lag_p: LgaGateParams;
@group(0) @binding(33) var<storage, read_write> lag_out: array<u32>;

fn lag_gval(elem: u32) -> f32 {
    var idx = elem;
    if (lag_p.gate_kind == 1u) {
        idx = elem / lag_p.head_dim;
    }
    let word = lag_g[idx >> 1u];
    let g = select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
    let sp = max(g, 0.0) + log(1.0 + exp(-abs(g)));
    return bf16_decode(bf16_encode(sp));
}

@compute @workgroup_size(64)
fn lgw_attn_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= lag_p.n_words) {
        return;
    }
    let w = lag_x[i];
    let e = i * 2u;
    lag_out[i] = bf16_pack(bf16_lo(w) * lag_gval(e), bf16_hi(w) * lag_gval(e + 1u));
}
