
struct PfGatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
    scale: f32,
    m: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> pg_emb: array<u32>;
@group(0) @binding(1) var<storage, read> pg_tok: array<i32>;
@group(0) @binding(2) var<storage, read_write> pg_out: array<u32>;
@group(0) @binding(3) var<uniform> pg_p: PfGatherParams;

@compute @workgroup_size(256)
fn pm_gather_embed(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.y;
    if (r >= pg_p.m) {
        return;
    }
    var t = 0u;
    if (pg_tok[r] > 0) {
        t = u32(pg_tok[r]);
    }
    if (t >= pg_p.vocab) {
        t = 0u;
    }
    if (t < pg_p.row_off) {
        return;
    }
    if (t >= pg_p.row_off + pg_p.n_rows) {
        return;
    }
    let base = (t - pg_p.row_off) * pg_p.hidden_words;
    let w = wid.x * 256u + lid.x;
    if (w >= pg_p.hidden_words) {
        return;
    }
    let word = pg_emb[base + w];
    pg_out[r * pg_p.hidden_words + w] =
        bf16_pack(bf16_lo(word) * pg_p.scale, bf16_hi(word) * pg_p.scale);
}

struct PfNormParams {
    hidden: u32,
    words: u32,
    eps: f32,
    m: u32,
};

@group(0) @binding(10) var<storage, read> pn_x: array<u32>;
@group(0) @binding(11) var<storage, read> pn_w: array<u32>;
@group(0) @binding(12) var<storage, read_write> pn_y: array<u32>;
@group(0) @binding(13) var<uniform> pn_p: PfNormParams;
@group(0) @binding(14) var<storage, read_write> pn_res: array<u32>;

var<workgroup> pn_red: array<f32, 256>;
var<workgroup> pn_s: f32;

fn pn_reduce(lid: u32, local: f32) -> f32 {
    pn_red[lid] = local;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            pn_red[lid] = pn_red[lid] + pn_red[lid + s];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        pn_s = inverseSqrt(pn_red[0] / f32(pn_p.hidden) + pn_p.eps);
    }
    workgroupBarrier();
    return pn_s;
}

@compute @workgroup_size(256)
fn pm_norm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>
) {
    let lid = lid3.x;
    let off = wid.x * pn_p.words;
    var local = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[off + w];
        let x0 = bf16_lo(xw);
        let x1 = bf16_hi(xw);
        local = local + x0 * x0 + x1 * x1;
    }
    let s = pn_reduce(lid, local);
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[off + w];
        let ww = pn_w[w];
        pn_y[off + w] =
            bf16_pack(bf16_lo(xw) * s * bf16_lo(ww), bf16_hi(xw) * s * bf16_hi(ww));
    }
}

@compute @workgroup_size(256)
fn pm_norm_residual(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>
) {
    let lid = lid3.x;
    let off = wid.x * pn_p.words;
    var local = 0.0;
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let xw = pn_x[off + w];
        let rw = pn_res[off + w];
        let s0 = bf16_lo(xw) + bf16_lo(rw);
        let s1 = bf16_hi(xw) + bf16_hi(rw);
        pn_res[off + w] = bf16_pack(s0, s1);
        let sr = pn_res[off + w];
        let r0 = bf16_lo(sr);
        let r1 = bf16_hi(sr);
        local = local + r0 * r0 + r1 * r1;
    }
    let s = pn_reduce(lid, local);
    for (var w = lid; w < pn_p.words; w = w + 256u) {
        let sr = pn_res[off + w];
        let ww = pn_w[w];
        pn_y[off + w] =
            bf16_pack(bf16_lo(sr) * s * bf16_lo(ww), bf16_hi(sr) * s * bf16_hi(ww));
    }
}

struct PfGemmParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    y_off_words: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(20) var<storage, read> pb_w: array<u32>;
@group(0) @binding(21) var<storage, read> pb_x: array<u32>;
@group(0) @binding(22) var<uniform> pb_p: PfGemmParams;
@group(0) @binding(23) var<storage, read_write> pb_y: array<u32>;

var<workgroup> pb_red: array<f32, 256>;

struct PfRopeParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    has_rope: u32,
    tok_src_stride: u32,
    tok_dst_stride: u32,
    pad0: u32,
    eps: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(30) var<storage, read> pr_src: array<u32>;
@group(0) @binding(31) var<storage, read> pr_w: array<u32>;
@group(0) @binding(32) var<storage, read> pr_cos: array<f32>;
@group(0) @binding(33) var<storage, read> pr_sin: array<f32>;
@group(0) @binding(34) var<storage, read> pr_pos: array<i32>;
@group(0) @binding(35) var<storage, read_write> pr_out: array<u32>;
@group(0) @binding(36) var<uniform> pr_p: PfRopeParams;

var<workgroup> pr_red: array<f32, 256>;
var<workgroup> pr_buf: array<f32, 512>;

fn pr_rope_at(d: u32, p: u32) -> f32 {
    if (pr_p.has_rope == 0u) {
        return pr_buf[d];
    }
    let rh = pr_p.rot_half;
    if (d < rh) {
        let c = pr_cos[p * rh + d];
        let s = pr_sin[p * rh + d];
        return pr_buf[d] * c - pr_buf[d + rh] * s;
    }
    let i = d - rh;
    let c = pr_cos[p * rh + i];
    let s = pr_sin[p * rh + i];
    return pr_buf[i] * s + pr_buf[d] * c;
}

@compute @workgroup_size(256)
fn pm_attn_norm_rope(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.x;
    let t = wid.y;
    let lane = lid.x;
    let hd = pr_p.head_dim;
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < hd) {
        let base = t * pr_p.tok_src_stride + r * pr_p.src_stride + e0;
        v0 = bf16_decode(u16_at(pr_src[base >> 1u], base));
        v1 = bf16_decode(u16_at(pr_src[(base + 1u) >> 1u], base + 1u));
    }
    pr_red[lane] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            pr_red[lane] = pr_red[lane] + pr_red[lane + s];
        }
        workgroupBarrier();
    }
    let rms = inverseSqrt(pr_red[0] / f32(hd) + pr_p.eps);
    if (e0 < hd) {
        let w0 = bf16_decode(u16_at(pr_w[e0 >> 1u], e0));
        let w1 = bf16_decode(u16_at(pr_w[(e0 + 1u) >> 1u], e0 + 1u));
        pr_buf[e0] = bf16_decode(bf16_encode(v0 * rms * w0));
        pr_buf[e0 + 1u] = bf16_decode(bf16_encode(v1 * rms * w1));
    }
    workgroupBarrier();
    if (e0 >= hd) {
        return;
    }
    var p = 0u;
    if (pr_pos[t] > 0) {
        p = u32(pr_pos[t]);
    }
    let o0 = pr_rope_at(e0, p);
    let o1 = pr_rope_at(e0 + 1u, p);
    pr_out[(t * pr_p.tok_dst_stride + r * hd + e0) >> 1u] = bf16_pack(o0, o1);
}

struct PfKvParams {
    words: u32,
    m: u32,
    ring: u32,
    pad1: u32,
};

@group(0) @binding(40) var<storage, read> pk_k: array<u32>;
@group(0) @binding(41) var<storage, read> pk_v: array<u32>;
@group(0) @binding(42) var<storage, read_write> pk_kc: array<u32>;
@group(0) @binding(43) var<storage, read_write> pk_vc: array<u32>;
@group(0) @binding(44) var<storage, read> pk_pos: array<i32>;
@group(0) @binding(45) var<uniform> pk_p: PfKvParams;

@compute @workgroup_size(64)
fn pm_kv_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    if (i >= pk_p.words || t >= pk_p.m) {
        return;
    }
    var p = 0u;
    if (pk_pos[t] > 0) {
        p = u32(pk_pos[t]);
    }
    var slot = p;
    if (pk_p.ring > 0u) {
        slot = p % pk_p.ring;
    }
    let base = slot * pk_p.words;
    let src = t * pk_p.words;
    pk_kc[base + i] = pk_k[src + i];
    pk_vc[base + i] = pk_v[src + i];
}

struct PfAttnParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    window: u32,
    m: u32,
    ring: u32,
    scale: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(50) var<storage, read> pa_q: array<u32>;
@group(0) @binding(51) var<storage, read> pa_kc: array<u32>;
@group(0) @binding(52) var<storage, read> pa_vc: array<u32>;
@group(0) @binding(53) var<storage, read_write> pa_scores: array<f32>;
@group(0) @binding(54) var<storage, read_write> pa_out: array<f32>;
@group(0) @binding(55) var<storage, read> pa_pos: array<i32>;
@group(0) @binding(56) var<uniform> pa_p: PfAttnParams;

var<workgroup> pa_qs: array<f32, 512>;
var<workgroup> pa_red: array<f32, 256>;
var<workgroup> pa_m: f32;
var<workgroup> pa_z: f32;

@compute @workgroup_size(256)
fn pm_attn(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tk = wid.y;
    let tid = lid.x;
    let hd = pa_p.head_dim;
    var p = 0u;
    if (pa_pos[tk] > 0) {
        p = u32(pa_pos[tk]);
    }
    let total = p + 1u;
    var start = 0u;
    if (pa_p.window > 0u && total > pa_p.window) {
        start = total - pa_p.window;
    }
    let kv = h / pa_p.group;
    let srow = (tk * pa_p.n_heads + h) * pa_p.max_seq;
    let qrow = tk * pa_p.n_heads * hd;

    for (var d = tid; d < hd; d = d + 256u) {
        let idx = qrow + h * hd + d;
        pa_qs[d] = bf16_decode(u16_at(pa_q[idx >> 1u], idx));
    }
    workgroupBarrier();

    var lmax = -3.4028235e38;
    for (var t = start + tid; t < total; t = t + 256u) {
        var kslot = t;
        if (pa_p.ring > 0u) {
            kslot = t % pa_p.ring;
        }
        let kbase = (kslot * pa_p.n_kv + kv) * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let idx = kbase + d;
            dot = fma(bf16_decode(u16_at(pa_kc[idx >> 1u], idx)), pa_qs[d], dot);
        }
        let s = dot * pa_p.scale;
        pa_scores[srow + t] = s;
        lmax = max(lmax, s);
    }
    pa_red[tid] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            pa_red[tid] = max(pa_red[tid], pa_red[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        pa_m = pa_red[0];
    }
    workgroupBarrier();
    let m = pa_m;

    var lsum = 0.0;
    for (var t = start + tid; t < total; t = t + 256u) {
        let e = exp(pa_scores[srow + t] - m);
        pa_scores[srow + t] = e;
        lsum = lsum + e;
    }
    workgroupBarrier();
    pa_red[tid] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            pa_red[tid] = pa_red[tid] + pa_red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        pa_z = pa_red[0];
    }
    workgroupBarrier();
    let z = pa_z;

    for (var d = tid; d < hd; d = d + 256u) {
        var acc = 0.0;
        for (var t = start; t < total; t = t + 1u) {
            var vslot = t;
            if (pa_p.ring > 0u) {
                vslot = t % pa_p.ring;
            }
            let idx = (vslot * pa_p.n_kv + kv) * hd + d;
            acc = fma(pa_scores[srow + t], bf16_decode(u16_at(pa_vc[idx >> 1u], idx)), acc);
        }
        pa_out[qrow + h * hd + d] = acc / z;
    }
}

struct PfRouterParams {
    n_experts: u32,
    k: u32,
    m: u32,
    pad0: u32,
};

@group(0) @binding(60) var<storage, read> pt_logits: array<u32>;
@group(0) @binding(61) var<storage, read_write> pt_ids: array<u32>;
@group(0) @binding(62) var<storage, read_write> pt_w: array<f32>;
@group(0) @binding(63) var<uniform> pt_p: PfRouterParams;
@group(0) @binding(64) var<storage, read> pt_pes: array<f32>;

@compute @workgroup_size(1)
fn pm_router_topk(@builtin(workgroup_id) wid: vec3<u32>) {
    let r = wid.x;
    if (r >= pt_p.m) {
        return;
    }
    let lbase = r * pt_p.n_experts;
    let obase = r * pt_p.k;
    var taken: array<u32, 256>;
    var chosen: array<f32, 16>;
    for (var i = 0u; i < pt_p.n_experts; i = i + 1u) {
        taken[i] = 0u;
    }
    for (var j = 0u; j < pt_p.k; j = j + 1u) {
        var best = -3.4028235e38;
        var bi = 0u;
        var found = false;
        for (var i = 0u; i < pt_p.n_experts; i = i + 1u) {
            if (taken[i] == 1u) {
                continue;
            }
            let v = bitcast<f32>(pt_logits[lbase + i]);
            if (!found || v > best) {
                best = v;
                bi = i;
                found = true;
            }
        }
        taken[bi] = 1u;
        pt_ids[obase + j] = bi;
        chosen[j] = best;
    }
    var mx = chosen[0];
    for (var j = 1u; j < pt_p.k; j = j + 1u) {
        mx = max(mx, chosen[j]);
    }
    var ssum = 0.0;
    for (var j = 0u; j < pt_p.k; j = j + 1u) {
        let e = exp(chosen[j] - mx);
        chosen[j] = e;
        ssum = ssum + e;
    }
    for (var j = 0u; j < pt_p.k; j = j + 1u) {
        pt_w[obase + j] = chosen[j] / ssum * pt_pes[pt_ids[obase + j]];
    }
}

struct PfW4Params {
    n_rows: u32,
    groups: u32,
    groups_x: u32,
    w_e_stride_words: u32,
    s_e_stride_elems: u32,
    x_slot_stride_words: u32,
    y_slot_stride_words: u32,
    x_tok_stride_words: u32,
    k_top: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(70) var<storage, read> pw_w: array<u32>;
@group(0) @binding(71) var<storage, read> pw_ws: array<u32>;
@group(0) @binding(72) var<storage, read> pw_x: array<u32>;
@group(0) @binding(73) var<uniform> pw_p: PfW4Params;
@group(0) @binding(74) var<storage, read_write> pw_y: array<u32>;
@group(0) @binding(75) var<storage, read> pw_sel: array<u32>;

var<workgroup> pw_red: array<f32, 256>;

@compute @workgroup_size(256)
fn pm_gemv_w4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let slot = wid.z;
    let e = pw_sel[slot];
    let pair = wid.x + wid.y * pw_p.groups_x;
    let row = pair * 2u + half;
    let live = row < pw_p.n_rows;
    let row_words = pw_p.groups * 4u;
    let wbase = select(0u, e * pw_p.w_e_stride_words + row * row_words, live);
    let sbase = e * pw_p.s_e_stride_elems + row * pw_p.groups;
    let xbase = (slot / pw_p.k_top) * pw_p.x_tok_stride_words
        + slot * pw_p.x_slot_stride_words;
    let groups = select(0u, pw_p.groups, live);

    var acc = 0.0;
    for (var g = lane; g < groups; g = g + 128u) {
        let si = sbase + g;
        let scale = bf16_decode(u16_at(pw_ws[si >> 1u], si));
        var gdot = 0.0;
        let wg = wbase + g * 4u;
        let xg = xbase + g * 16u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let pv = pw_w[wg + j];
            for (var i = 0u; i < 4u; i = i + 1u) {
                let xw = pw_x[xg + j * 4u + i];
                let q0 = f32((pv >> (8u * i)) & 15u) - 8.0;
                let q1 = f32((pv >> (8u * i + 4u)) & 15u) - 8.0;
                gdot = fma(q0, bf16_lo(xw), gdot);
                gdot = fma(q1, bf16_hi(xw), gdot);
            }
        }
        acc = fma(scale, gdot, acc);
    }
    pw_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            pw_red[tid] = pw_red[tid] + pw_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let lo = pw_red[0];
        var hi = 0.0;
        if (row + 1u < pw_p.n_rows) {
            hi = pw_red[128];
        }
        pw_y[slot * pw_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

struct PfMulParams {
    row_words: u32,
    m: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(90) var<storage, read> pu_x: array<u32>;
@group(0) @binding(91) var<storage, read> pu_s: array<u32>;
@group(0) @binding(92) var<storage, read_write> pu_y: array<u32>;
@group(0) @binding(93) var<uniform> pu_p: PfMulParams;

@compute @workgroup_size(64)
fn pm_mul_rowscale(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let t = gid.y;
    if (w >= pu_p.row_words || t >= pu_p.m) {
        return;
    }
    let i = t * pu_p.row_words + w;
    let xw = pu_x[i];
    let sw = pu_s[w];
    pu_y[i] = bf16_pack(bf16_lo(xw) * bf16_lo(sw), bf16_hi(xw) * bf16_hi(sw));
}

struct PfCombineParams {
    hidden_words: u32,
    k: u32,
    slot_stride_words: u32,
    m: u32,
};

@group(0) @binding(80) var<storage, read> pc_y: array<u32>;
@group(0) @binding(81) var<storage, read> pc_w: array<f32>;
@group(0) @binding(82) var<storage, read_write> pc_out: array<u32>;
@group(0) @binding(83) var<uniform> pc_p: PfCombineParams;

@compute @workgroup_size(64)
fn pm_moe_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let t = gid.y;
    if (w >= pc_p.hidden_words || t >= pc_p.m) {
        return;
    }
    let sbase = t * pc_p.k;
    var a0 = 0.0;
    var a1 = 0.0;
    for (var j = 0u; j < pc_p.k; j = j + 1u) {
        let word = pc_y[(sbase + j) * pc_p.slot_stride_words + w];
        let wt = pc_w[sbase + j];
        a0 = fma(bf16_lo(word), wt, a0);
        a1 = fma(bf16_hi(word), wt, a1);
    }
    pc_out[t * pc_p.hidden_words + w] = bf16_pack(a0, a1);
}

@group(0) @binding(57) var<storage, read> pa_ksc: array<f32>;
@group(0) @binding(58) var<storage, read> pa_vsc: array<f32>;

const PA_E4M3_SHIFT_CARRY_2POW120_RIDES_Q_AND_V_SCALE: f32 = 1329227995784915872903807060280344576.0;

@compute @workgroup_size(256)
fn pm_attn_fp8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tk = wid.y;
    let tid = lid.x;
    let hd = pa_p.head_dim;
    var p = 0u;
    if (pa_pos[tk] > 0) {
        p = u32(pa_pos[tk]);
    }
    let total = p + 1u;
    var start = 0u;
    if (pa_p.window > 0u && total > pa_p.window) {
        start = total - pa_p.window;
    }
    let kv = h / pa_p.group;
    let srow = (tk * pa_p.n_heads + h) * pa_p.max_seq;
    let qrow = tk * pa_p.n_heads * hd;

    for (var d = tid; d < hd; d = d + 256u) {
        let idx = qrow + h * hd + d;
        pa_qs[d] = bf16_decode(u16_at(pa_q[idx >> 1u], idx))
            * PA_E4M3_SHIFT_CARRY_2POW120_RIDES_Q_AND_V_SCALE;
    }
    workgroupBarrier();

    var lmax = -3.4028235e38;
    for (var t = start + tid; t < total; t = t + 256u) {
        var kslot = t;
        if (pa_p.ring > 0u) {
            kslot = t % pa_p.ring;
        }
        let krow = kslot * pa_p.n_kv + kv;
        let kbase = krow * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let idx = kbase + d;
            dot = fma(
                e4m3_shift_decode_scale_must_carry_2pow120(byte_at(pa_kc[idx >> 2u], idx)),
                pa_qs[d],
                dot
            );
        }
        let s = dot * pa_ksc[krow] * pa_p.scale;
        pa_scores[srow + t] = s;
        lmax = max(lmax, s);
    }
    pa_red[tid] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            pa_red[tid] = max(pa_red[tid], pa_red[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        pa_m = pa_red[0];
    }
    workgroupBarrier();
    let m = pa_m;

    var lsum = 0.0;
    for (var t = start + tid; t < total; t = t + 256u) {
        let e = exp(pa_scores[srow + t] - m);
        pa_scores[srow + t] = e;
        lsum = lsum + e;
    }
    workgroupBarrier();
    pa_red[tid] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            pa_red[tid] = pa_red[tid] + pa_red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        pa_z = pa_red[0];
    }
    workgroupBarrier();
    let z = pa_z;

    for (var d = tid; d < hd; d = d + 256u) {
        var acc = 0.0;
        for (var t = start; t < total; t = t + 1u) {
            var vslot = t;
            if (pa_p.ring > 0u) {
                vslot = t % pa_p.ring;
            }
            let vrow = vslot * pa_p.n_kv + kv;
            let idx = vrow * hd + d;
            acc = fma(
                pa_scores[srow + t] * (pa_vsc[vrow] * PA_E4M3_SHIFT_CARRY_2POW120_RIDES_Q_AND_V_SCALE),
                e4m3_shift_decode_scale_must_carry_2pow120(byte_at(pa_vc[idx >> 2u], idx)),
                acc
            );
        }
        pa_out[qrow + h * hd + d] = acc / z;
    }
}
