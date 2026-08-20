
struct Q3arParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    src_off: u32,
    w_off: u32,
    n_q: u32,
    k_src_off: u32,
    k_src_stride: u32,
    k_w_off: u32,
    sin_off: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> ar_src: array<u32>;
@group(0) @binding(1) var<storage, read> ar_w: array<u32>;
@group(0) @binding(2) var<storage, read> ar_cs: array<f32>;
@group(0) @binding(3) var<storage, read> ar_pos: array<i32>;
@group(0) @binding(4) var<storage, read> ar_ksrc: array<u32>;
@group(0) @binding(5) var<storage, read_write> ar_out: array<u32>;
@group(0) @binding(6) var<uniform> ar_p: Q3arParams;
@group(0) @binding(7) var<storage, read_write> ar_outf: array<f32>;
@group(0) @binding(8) var<storage, read_write> ar_kout: array<u32>;

var<workgroup> ar_red: array<f32, 128>;
var<workgroup> ar_buf: array<f32, 256>;

fn ar_rope_at(d: u32, p: u32) -> f32 {
    let rh = ar_p.rot_half;
    let so = ar_p.sin_off;
    if (d < rh) {
        let c = ar_cs[p * rh + d];
        let s = ar_cs[so + p * rh + d];
        return ar_buf[d] * c - ar_buf[d + rh] * s;
    }
    if (d < 2u * rh) {
        let i = d - rh;
        let c = ar_cs[p * rh + i];
        let s = ar_cs[so + p * rh + i];
        return ar_buf[i] * s + ar_buf[d] * c;
    }
    return ar_buf[d];
}

fn ar_norm_rope(v0: f32, v1: f32, w_off: u32, lane: u32) -> vec2<f32> {
    let hd = ar_p.head_dim;
    let e0 = 2u * lane;
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
        let wi = w_off + e0;
        let w0 = bf16_decode(u16_at(ar_w[wi >> 1u], wi));
        let w1 = bf16_decode(u16_at(ar_w[(wi + 1u) >> 1u], wi + 1u));
        ar_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        ar_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    var o = vec2<f32>(0.0, 0.0);
    if (e0 < hd) {
        var p = 0u;
        if (ar_pos[0] > 0) {
            p = u32(ar_pos[0]);
        }
        let o0 = ar_rope_at(e0, p);
        let o1 = ar_rope_at(e0 + 1u, p);
        o = vec2<f32>(o0, o1);
    }
    return o;
}

fn ar_load(base: u32) -> vec2<f32> {
    return vec2<f32>(
        bf16_decode(u16_at(ar_src[base >> 1u], base)),
        bf16_decode(u16_at(ar_src[(base + 1u) >> 1u], base + 1u))
    );
}

fn ar_kload(base: u32) -> vec2<f32> {
    return vec2<f32>(
        bf16_decode(u16_at(ar_ksrc[base >> 1u], base)),
        bf16_decode(u16_at(ar_ksrc[(base + 1u) >> 1u], base + 1u))
    );
}

@compute @workgroup_size(128)
fn q3w_attn_norm_rope(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let hd = ar_p.head_dim;
    let e0 = 2u * lid.x;
    var v = vec2<f32>(0.0, 0.0);
    if (e0 < hd) {
        v = ar_load(ar_p.src_off + wid.x * ar_p.src_stride + e0);
    }
    let o = ar_norm_rope(v.x, v.y, ar_p.w_off, lid.x);
    if (e0 < hd) {
        ar_out[(wid.x * hd + e0) >> 1u] = bf16_pack(o.x, o.y);
    }
}

@compute @workgroup_size(128)
fn q3w_attn_norm_rope_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let hd = ar_p.head_dim;
    let e0 = 2u * lid.x;
    var v = vec2<f32>(0.0, 0.0);
    if (e0 < hd) {
        v = ar_load(ar_p.src_off + wid.x * ar_p.src_stride + e0);
    }
    let o = ar_norm_rope(v.x, v.y, ar_p.w_off, lid.x);
    if (e0 < hd) {
        let i = wid.x * hd + e0;
        ar_out[i >> 1u] = bf16_pack(o.x, o.y);
        ar_outf[i] = bf16_decode(bf16_encode(o.x));
        ar_outf[i + 1u] = bf16_decode(bf16_encode(o.y));
    }
}

@compute @workgroup_size(128)
fn q3w_attn_norm_rope_qk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let hd = ar_p.head_dim;
    let e0 = 2u * lid.x;
    let is_k = wid.x >= ar_p.n_q;
    var row = wid.x;
    var w_off = ar_p.w_off;
    if (is_k) {
        row = wid.x - ar_p.n_q;
        w_off = ar_p.k_w_off;
    }
    var v = vec2<f32>(0.0, 0.0);
    if (e0 < hd) {
        if (is_k) {
            v = ar_kload(ar_p.k_src_off + row * ar_p.k_src_stride + e0);
        } else {
            v = ar_load(ar_p.src_off + row * ar_p.src_stride + e0);
        }
    }
    let o = ar_norm_rope(v.x, v.y, w_off, lid.x);
    if (e0 < hd) {
        let i = row * hd + e0;
        if (is_k) {
            ar_kout[i >> 1u] = bf16_pack(o.x, o.y);
        } else {
            ar_out[i >> 1u] = bf16_pack(o.x, o.y);
            ar_outf[i] = bf16_decode(bf16_encode(o.x));
            ar_outf[i + 1u] = bf16_decode(bf16_encode(o.y));
        }
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
