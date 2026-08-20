
struct GowRopeParams {
    n_rows: u32,
    head_dim: u32,
    rot_half: u32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> gr_src: array<u32>;
@group(0) @binding(1) var<storage, read> gr_cos: array<f32>;
@group(0) @binding(2) var<storage, read> gr_sin: array<f32>;
@group(0) @binding(3) var<storage, read> gr_pos: array<i32>;
@group(0) @binding(4) var<storage, read_write> gr_out: array<u32>;
@group(0) @binding(5) var<uniform> gr_p: GowRopeParams;

fn gr_at(r: u32, e: u32) -> f32 {
    let idx = r * gr_p.head_dim + e;
    return bf16_decode(u16_at(gr_src[idx >> 1u], idx));
}

fn gr_rot(r: u32, e: u32, p: u32) -> f32 {
    let rh = gr_p.rot_half;
    if (e < rh) {
        let c = gr_cos[p * rh + e];
        let s = gr_sin[p * rh + e];
        return gr_at(r, e) * c - gr_at(r, e + rh) * s;
    }
    let i = e - rh;
    let c = gr_cos[p * rh + i];
    let s = gr_sin[p * rh + i];
    return gr_at(r, e) * c + gr_at(r, i) * s;
}

@compute @workgroup_size(32)
fn gow_rope(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.x;
    let w = lid.x;
    let e0 = 2u * w;
    if (e0 >= gr_p.head_dim) {
        return;
    }
    var p = 0u;
    if (gr_pos[0] > 0) {
        p = u32(gr_pos[0]);
    }
    let o0 = gr_rot(r, e0, p);
    let o1 = gr_rot(r, e0 + 1u, p);
    gr_out[(r * gr_p.head_dim + e0) >> 1u] = bf16_pack(o0, o1);
}

struct GowKvParams {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> gkv_k: array<u32>;
@group(0) @binding(11) var<storage, read> gkv_v: array<u32>;
@group(0) @binding(12) var<storage, read_write> gkv_kc: array<u32>;
@group(0) @binding(13) var<storage, read_write> gkv_vc: array<u32>;
@group(0) @binding(14) var<storage, read> gkv_pos: array<i32>;
@group(0) @binding(15) var<uniform> gkv_p: GowKvParams;

@compute @workgroup_size(64)
fn gow_kv_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= gkv_p.words) {
        return;
    }
    var p = 0u;
    if (gkv_pos[0] > 0) {
        p = u32(gkv_pos[0]);
    }
    let base = p * gkv_p.words;
    gkv_kc[base + i] = gkv_k[i];
    gkv_vc[base + i] = gkv_v[i];
}

struct GowAdParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    window: u32,
    pad0: u32,
    scale: f32,
};

@group(0) @binding(20) var<storage, read> gad_q: array<u32>;
@group(0) @binding(21) var<storage, read> gad_kc: array<u32>;
@group(0) @binding(22) var<storage, read> gad_vc: array<u32>;
@group(0) @binding(23) var<storage, read_write> gad_scores: array<f32>;
@group(0) @binding(24) var<storage, read_write> gad_out: array<f32>;
@group(0) @binding(25) var<storage, read> gad_pos: array<i32>;
@group(0) @binding(26) var<uniform> gad_p: GowAdParams;
@group(0) @binding(27) var<storage, read> gad_sinks: array<f32>;

var<workgroup> gad_qs: array<f32, 256>;
var<workgroup> gad_red: array<f32, 256>;
var<workgroup> gad_m: f32;
var<workgroup> gad_z: f32;

@compute @workgroup_size(256)
fn gow_attn_decode(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tid = lid.x;
    let hd = gad_p.head_dim;
    var p = 0u;
    if (gad_pos[0] > 0) {
        p = u32(gad_pos[0]);
    }
    let total = p + 1u;
    var start = 0u;
    if (gad_p.window > 0u && total > gad_p.window) {
        start = total - gad_p.window;
    }
    let kv = h / gad_p.group;
    let srow = h * gad_p.max_seq;
    let sink = gad_sinks[h];

    for (var d = tid; d < hd; d = d + 256u) {
        let idx = h * hd + d;
        gad_qs[d] = bf16_decode(u16_at(gad_q[idx >> 1u], idx));
    }
    workgroupBarrier();

    var lmax = -3.4028235e38;
    for (var t = start + tid; t < total; t = t + 256u) {
        let kbase = (t * gad_p.n_kv + kv) * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let idx = kbase + d;
            dot = fma(bf16_decode(u16_at(gad_kc[idx >> 1u], idx)), gad_qs[d], dot);
        }
        let s = dot * gad_p.scale;
        gad_scores[srow + t] = s;
        lmax = max(lmax, s);
    }
    gad_red[tid] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gad_red[tid] = max(gad_red[tid], gad_red[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        gad_m = max(gad_red[0], sink);
    }
    workgroupBarrier();
    let m = gad_m;

    var lsum = 0.0;
    for (var t = start + tid; t < total; t = t + 256u) {
        let e = exp(gad_scores[srow + t] - m);
        gad_scores[srow + t] = e;
        lsum = lsum + e;
    }
    workgroupBarrier();
    gad_red[tid] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gad_red[tid] = gad_red[tid] + gad_red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        gad_z = gad_red[0] + exp(sink - m);
    }
    workgroupBarrier();
    let z = gad_z;

    if (tid < hd) {
        var acc = 0.0;
        for (var t = start; t < total; t = t + 1u) {
            let idx = (t * gad_p.n_kv + kv) * hd + tid;
            acc = fma(gad_scores[srow + t], bf16_decode(u16_at(gad_vc[idx >> 1u], idx)), acc);
        }
        gad_out[h * hd + tid] = acc / z;
    }
}

struct GowPkParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(30) var<storage, read> gpk_src: array<f32>;
@group(0) @binding(31) var<storage, read_write> gpk_dst: array<u32>;
@group(0) @binding(32) var<uniform> gpk_p: GowPkParams;

@compute @workgroup_size(64)
fn gow_pack_bf16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= gpk_p.n_words) {
        return;
    }
    gpk_dst[w] = bf16_pack(gpk_src[2u * w], gpk_src[2u * w + 1u]);
}
