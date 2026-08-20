
struct G4aParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    has_rope: u32,
    n_q: u32,
    n_kv: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> ga_src: array<u32>;
@group(0) @binding(1) var<storage, read> ga_w: array<u32>;
@group(0) @binding(2) var<storage, read> ga_cos: array<f32>;
@group(0) @binding(3) var<storage, read> ga_sin: array<f32>;
@group(0) @binding(4) var<storage, read> ga_pos: array<i32>;
@group(0) @binding(5) var<storage, read_write> ga_out: array<u32>;
@group(0) @binding(6) var<uniform> ga_p: G4aParams;

var<workgroup> ga_red: array<f32, 256>;
var<workgroup> ga_buf: array<f32, 512>;

fn ga_rope_at(d: u32, p: u32) -> f32 {
    if (ga_p.has_rope == 0u) {
        return ga_buf[d];
    }
    let rh = ga_p.rot_half;
    if (d < rh) {
        let c = ga_cos[p * rh + d];
        let s = ga_sin[p * rh + d];
        return ga_buf[d] * c - ga_buf[d + rh] * s;
    }
    let i = d - rh;
    let c = ga_cos[p * rh + i];
    let s = ga_sin[p * rh + i];
    return ga_buf[i] * s + ga_buf[d] * c;
}

@compute @workgroup_size(256)
fn g4m_attn_norm_rope(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.x;
    let lane = lid.x;
    let hd = ga_p.head_dim;
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < hd) {
        let base = r * ga_p.src_stride + e0;
        v0 = bf16_decode(u16_at(ga_src[base >> 1u], base));
        v1 = bf16_decode(u16_at(ga_src[(base + 1u) >> 1u], base + 1u));
    }
    ga_red[lane] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            ga_red[lane] = ga_red[lane] + ga_red[lane + s];
        }
        workgroupBarrier();
    }
    let rms = inverseSqrt(ga_red[0] / f32(hd) + ga_p.eps);
    if (e0 < hd) {
        let w0 = bf16_decode(u16_at(ga_w[e0 >> 1u], e0));
        let w1 = bf16_decode(u16_at(ga_w[(e0 + 1u) >> 1u], e0 + 1u));
        ga_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        ga_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    if (e0 >= hd) {
        return;
    }
    var p = 0u;
    if (ga_pos[0] > 0) {
        p = u32(ga_pos[0]);
    }
    let o0 = ga_rope_at(e0, p);
    let o1 = ga_rope_at(e0 + 1u, p);
    ga_out[(r * hd + e0) >> 1u] = bf16_pack(o0, o1);
}

struct G4kvParams {
    words: u32,
    ring: u32,
    k_off_words: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> gk_k: array<u32>;
@group(0) @binding(11) var<storage, read> gk_v: array<u32>;
@group(0) @binding(12) var<storage, read_write> gk_kc: array<u32>;
@group(0) @binding(13) var<storage, read_write> gk_vc: array<u32>;
@group(0) @binding(14) var<storage, read> gk_pos: array<i32>;
@group(0) @binding(15) var<uniform> gk_p: G4kvParams;

@compute @workgroup_size(64)
fn g4m_kv_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= gk_p.words) {
        return;
    }
    var p = 0u;
    if (gk_pos[0] > 0) {
        p = u32(gk_pos[0]);
    }
    var slot = p;
    if (gk_p.ring > 0u) {
        slot = p % gk_p.ring;
    }
    let base = slot * gk_p.words;
    gk_kc[base + i] = gk_k[i];
    gk_vc[base + i] = gk_v[i];
}

struct G4dParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    window: u32,
    ring: u32,
    scale: f32,
};

@group(0) @binding(20) var<storage, read> gd_q: array<u32>;
@group(0) @binding(21) var<storage, read> gd_kc: array<u32>;
@group(0) @binding(22) var<storage, read> gd_vc: array<u32>;
@group(0) @binding(23) var<storage, read_write> gd_scores: array<f32>;
@group(0) @binding(24) var<storage, read_write> gd_out: array<f32>;
@group(0) @binding(25) var<storage, read> gd_pos: array<i32>;
@group(0) @binding(26) var<uniform> gd_p: G4dParams;

@group(0) @binding(27) var<storage, read> gd_ksc: array<f32>;
@group(0) @binding(28) var<storage, read> gd_vsc: array<f32>;

var<workgroup> gd_qs: array<f32, 512>;
var<workgroup> gd_red: array<f32, 256>;
var<workgroup> gd_m: f32;
var<workgroup> gd_z: f32;

@compute @workgroup_size(256)
fn g4m_attn_decode(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tid = lid.x;
    let hd = gd_p.head_dim;
    var p = 0u;
    if (gd_pos[0] > 0) {
        p = u32(gd_pos[0]);
    }
    let total = p + 1u;
    var start = 0u;
    if (gd_p.window > 0u && total > gd_p.window) {
        start = total - gd_p.window;
    }
    let kv = h / gd_p.group;
    let srow = h * gd_p.max_seq;

    for (var d = tid; d < hd; d = d + 256u) {
        let idx = h * hd + d;
        gd_qs[d] = bf16_decode(u16_at(gd_q[idx >> 1u], idx));
    }
    workgroupBarrier();

    var lmax = -3.4028235e38;
    for (var t = start + tid; t < total; t = t + 256u) {
        var kslot = t;
        if (gd_p.ring > 0u) {
            kslot = t % gd_p.ring;
        }
        let kbase = (kslot * gd_p.n_kv + kv) * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let idx = kbase + d;
            dot = fma(bf16_decode(u16_at(gd_kc[idx >> 1u], idx)), gd_qs[d], dot);
        }
        let s = dot * gd_p.scale;
        gd_scores[srow + t] = s;
        lmax = max(lmax, s);
    }
    gd_red[tid] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gd_red[tid] = max(gd_red[tid], gd_red[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        gd_m = gd_red[0];
    }
    workgroupBarrier();
    let m = gd_m;

    var lsum = 0.0;
    for (var t = start + tid; t < total; t = t + 256u) {
        let e = exp(gd_scores[srow + t] - m);
        gd_scores[srow + t] = e;
        lsum = lsum + e;
    }
    workgroupBarrier();
    gd_red[tid] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gd_red[tid] = gd_red[tid] + gd_red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        gd_z = gd_red[0];
    }
    workgroupBarrier();
    let z = gd_z;

    for (var d = tid; d < hd; d = d + 256u) {
        var acc = 0.0;
        for (var t = start; t < total; t = t + 1u) {
            var vslot = t;
            if (gd_p.ring > 0u) {
                vslot = t % gd_p.ring;
            }
            let idx = (vslot * gd_p.n_kv + kv) * hd + d;
            acc = fma(gd_scores[srow + t], bf16_decode(u16_at(gd_vc[idx >> 1u], idx)), acc);
        }
        gd_out[h * hd + d] = acc / z;
    }
}

const GD_E4M3_SHIFT_CARRY_2POW120_RIDES_Q_AND_V_SCALE: f32 = 1329227995784915872903807060280344576.0;

@compute @workgroup_size(256)
fn g4m_attn_decode_fp8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tid = lid.x;
    let hd = gd_p.head_dim;
    var p = 0u;
    if (gd_pos[0] > 0) {
        p = u32(gd_pos[0]);
    }
    let total = p + 1u;
    var start = 0u;
    if (gd_p.window > 0u && total > gd_p.window) {
        start = total - gd_p.window;
    }
    let kv = h / gd_p.group;
    let srow = h * gd_p.max_seq;

    for (var d = tid; d < hd; d = d + 256u) {
        let idx = h * hd + d;
        gd_qs[d] = bf16_decode(u16_at(gd_q[idx >> 1u], idx))
            * GD_E4M3_SHIFT_CARRY_2POW120_RIDES_Q_AND_V_SCALE;
    }
    workgroupBarrier();

    var lmax = -3.4028235e38;
    for (var t = start + tid; t < total; t = t + 256u) {
        var kslot = t;
        if (gd_p.ring > 0u) {
            kslot = t % gd_p.ring;
        }
        let krow = kslot * gd_p.n_kv + kv;
        let kbase = krow * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let idx = kbase + d;
            dot = fma(
                e4m3_shift_decode_scale_must_carry_2pow120(byte_at(gd_kc[idx >> 2u], idx)),
                gd_qs[d],
                dot
            );
        }
        let s = dot * gd_ksc[krow] * gd_p.scale;
        gd_scores[srow + t] = s;
        lmax = max(lmax, s);
    }
    gd_red[tid] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gd_red[tid] = max(gd_red[tid], gd_red[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        gd_m = gd_red[0];
    }
    workgroupBarrier();
    let m = gd_m;

    var lsum = 0.0;
    for (var t = start + tid; t < total; t = t + 256u) {
        let e = exp(gd_scores[srow + t] - m);
        gd_scores[srow + t] = e;
        lsum = lsum + e;
    }
    workgroupBarrier();
    gd_red[tid] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gd_red[tid] = gd_red[tid] + gd_red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        gd_z = gd_red[0];
    }
    workgroupBarrier();
    let z = gd_z;

    for (var d = tid; d < hd; d = d + 256u) {
        var acc = 0.0;
        for (var t = start; t < total; t = t + 1u) {
            var vslot = t;
            if (gd_p.ring > 0u) {
                vslot = t % gd_p.ring;
            }
            let vrow = vslot * gd_p.n_kv + kv;
            let idx = vrow * hd + d;
            acc = fma(
                gd_scores[srow + t] * (gd_vsc[vrow] * GD_E4M3_SHIFT_CARRY_2POW120_RIDES_Q_AND_V_SCALE),
                e4m3_shift_decode_scale_must_carry_2pow120(byte_at(gd_vc[idx >> 2u], idx)),
                acc
            );
        }
        gd_out[h * hd + d] = acc / z;
    }
}

@compute @workgroup_size(256)
fn g4m_attn_norm_rope_qkv(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.x;
    let lane = lid.x;
    let hd = ga_p.head_dim;
    let n_q = ga_p.n_q;
    let n_kv = ga_p.n_kv;
    let has_v = ga_p.has_rope;
    var src_row = r;
    var w_off = 0u;
    var rope = true;
    var out_row = r;
    if (r >= n_q + n_kv) {
        let vr = r - n_q - n_kv;
        if (has_v == 1u) {
            src_row = n_q + n_kv + vr;
        } else {
            src_row = n_q + vr;
        }
        w_off = 2u * hd;
        rope = false;
        out_row = r;
    } else if (r >= n_q) {
        w_off = hd;
    }
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < hd) {
        let base = src_row * ga_p.src_stride + e0;
        v0 = bf16_decode(u16_at(ga_src[base >> 1u], base));
        v1 = bf16_decode(u16_at(ga_src[(base + 1u) >> 1u], base + 1u));
    }
    ga_red[lane] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            ga_red[lane] = ga_red[lane] + ga_red[lane + s];
        }
        workgroupBarrier();
    }
    let rms = inverseSqrt(ga_red[0] / f32(hd) + ga_p.eps);
    if (e0 < hd) {
        let we0 = w_off + e0;
        let w0 = bf16_decode(u16_at(ga_w[we0 >> 1u], we0));
        let w1 = bf16_decode(u16_at(ga_w[(we0 + 1u) >> 1u], we0 + 1u));
        ga_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        ga_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    if (e0 >= hd) {
        return;
    }
    var p = 0u;
    if (ga_pos[0] > 0) {
        p = u32(ga_pos[0]);
    }
    var o0 = ga_buf[e0];
    var o1 = ga_buf[e0 + 1u];
    if (rope) {
        let rh = ga_p.rot_half;
        if (e0 < rh) {
            let c = ga_cos[p * rh + e0];
            let s = ga_sin[p * rh + e0];
            o0 = ga_buf[e0] * c - ga_buf[e0 + rh] * s;
        } else {
            let i0 = e0 - rh;
            let c = ga_cos[p * rh + i0];
            let s = ga_sin[p * rh + i0];
            o0 = ga_buf[i0] * s + ga_buf[e0] * c;
        }
        let e1 = e0 + 1u;
        if (e1 < rh) {
            let c = ga_cos[p * rh + e1];
            let s = ga_sin[p * rh + e1];
            o1 = ga_buf[e1] * c - ga_buf[e1 + rh] * s;
        } else {
            let i1 = e1 - rh;
            let c = ga_cos[p * rh + i1];
            let s = ga_sin[p * rh + i1];
            o1 = ga_buf[i1] * s + ga_buf[e1] * c;
        }
    }
    ga_out[(out_row * hd + e0) >> 1u] = bf16_pack(o0, o1);
}

@compute @workgroup_size(64)
fn g4m_kv_write_stacked(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= gk_p.words) {
        return;
    }
    var p = 0u;
    if (gk_pos[0] > 0) {
        p = u32(gk_pos[0]);
    }
    var slot = p;
    if (gk_p.ring > 0u) {
        slot = p % gk_p.ring;
    }
    let base = slot * gk_p.words;
    gk_kc[base + i] = gk_k[gk_p.k_off_words + i];
    gk_vc[base + i] = gk_k[gk_p.k_off_words + gk_p.words + i];
}
