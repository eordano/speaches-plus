
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
    let total = p + 1u;
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
