struct GowCkParams {
    m_live: u32,
    base: u32,
    pad0: u32,
    pad1: u32,
};

struct GowPfGeParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
    m: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> pge_emb: array<u32>;
@group(0) @binding(1) var<storage, read> pge_tok: array<i32>;
@group(0) @binding(2) var<storage, read_write> pge_out: array<u32>;
@group(0) @binding(3) var<uniform> pge_p: GowPfGeParams;

@compute @workgroup_size(256)
fn gow_pf_gather_embed(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    if (t >= pge_p.m) {
        return;
    }
    var s = 0u;
    if (pge_tok[t] > 0) {
        s = u32(pge_tok[t]);
    }
    if (s >= pge_p.vocab) {
        s = 0u;
    }
    if (s < pge_p.row_off) {
        return;
    }
    if (s >= pge_p.row_off + pge_p.n_rows) {
        return;
    }
    let base = (s - pge_p.row_off) * pge_p.hidden_words;
    let w = wid.x * 256u + lid.x;
    if (w >= pge_p.hidden_words) {
        return;
    }
    pge_out[t * pge_p.hidden_words + w] = pge_emb[base + w];
}

struct GowPfGbParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_row_words: u32,
    y_row_words: u32,
    has_bias: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> pgb_w: array<u32>;
@group(0) @binding(11) var<storage, read> pgb_x: array<u32>;
@group(0) @binding(12) var<uniform> pgb_p: GowPfGbParams;
@group(0) @binding(13) var<storage, read_write> pgb_y: array<u32>;
@group(0) @binding(14) var<storage, read> pgb_b: array<u32>;

var<workgroup> pgb_red: array<f32, 256>;

fn pgb_bias(row: u32) -> f32 {
    if (pgb_p.has_bias == 0u) {
        return 0.0;
    }
    return bf16_decode(u16_at(pgb_b[row >> 1u], row));
}

@compute @workgroup_size(256)
fn gow_pf_gemv_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let t = wid.z;
    let pair = wid.x + wid.y * pgb_p.groups_x;
    let row = pair * 2u + half;
    let live = row < pgb_p.n_rows;
    let wbase = select(0u, row * pgb_p.w_row_words, live);
    let kw = select(0u, pgb_p.k_words, live);
    let xbase = t * pgb_p.x_row_words;

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = pgb_w[wbase + i];
        let xw = pgb_x[xbase + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    pgb_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            pgb_red[tid] = pgb_red[tid] + pgb_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (pgb_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            let v = pgb_red[tid] * pgb_p.alpha + pgb_bias(row);
            pgb_y[t * pgb_p.y_row_words + row] = bitcast<u32>(v);
        }
    } else if (tid == 0u) {
        let lo = pgb_red[0] * pgb_p.alpha + pgb_bias(row);
        var hi = 0.0;
        if (row + 1u < pgb_p.n_rows) {
            hi = pgb_red[128] * pgb_p.alpha + pgb_bias(row + 1u);
        }
        pgb_y[t * pgb_p.y_row_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

struct GowPfRopeParams {
    n_rows: u32,
    head_dim: u32,
    rot_half: u32,
    pad0: u32,
};

@group(0) @binding(20) var<storage, read> pro_src: array<u32>;
@group(0) @binding(21) var<storage, read> pro_cos: array<f32>;
@group(0) @binding(22) var<storage, read> pro_sin: array<f32>;
@group(0) @binding(23) var<storage, read_write> pro_out: array<u32>;
@group(0) @binding(24) var<uniform> pro_p: GowPfRopeParams;
@group(0) @binding(25) var<uniform> pro_ck: GowCkParams;

fn pro_at(t: u32, r: u32, e: u32) -> f32 {
    let idx = (t * pro_p.n_rows + r) * pro_p.head_dim + e;
    return bf16_decode(u16_at(pro_src[idx >> 1u], idx));
}

fn pro_rot(t: u32, r: u32, e: u32, p: u32) -> f32 {
    let rh = pro_p.rot_half;
    if (e < rh) {
        let c = pro_cos[p * rh + e];
        let s = pro_sin[p * rh + e];
        return pro_at(t, r, e) * c - pro_at(t, r, e + rh) * s;
    }
    let i = e - rh;
    let c = pro_cos[p * rh + i];
    let s = pro_sin[p * rh + i];
    return pro_at(t, r, e) * c + pro_at(t, r, i) * s;
}

@compute @workgroup_size(32)
fn gow_pf_rope(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.x;
    let t = wid.y;
    if (t >= pro_ck.m_live) {
        return;
    }
    let w = lid.x;
    let e0 = 2u * w;
    if (e0 >= pro_p.head_dim) {
        return;
    }
    let p = pro_ck.base + t;
    let o0 = pro_rot(t, r, e0, p);
    let o1 = pro_rot(t, r, e0 + 1u, p);
    pro_out[((t * pro_p.n_rows + r) * pro_p.head_dim + e0) >> 1u] = bf16_pack(o0, o1);
}

struct GowPfKvParams {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(30) var<storage, read> pkv_k: array<u32>;
@group(0) @binding(31) var<storage, read> pkv_v: array<u32>;
@group(0) @binding(32) var<storage, read_write> pkv_kc: array<u32>;
@group(0) @binding(33) var<storage, read_write> pkv_vc: array<u32>;
@group(0) @binding(34) var<uniform> pkv_p: GowPfKvParams;
@group(0) @binding(35) var<uniform> pkv_ck: GowCkParams;

@compute @workgroup_size(64)
fn gow_pf_kv_write(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let t = gid.y;
    if (i >= pkv_p.words || t >= pkv_ck.m_live) {
        return;
    }
    let src = t * pkv_p.words + i;
    let dst = (pkv_ck.base + t) * pkv_p.words + i;
    pkv_kc[dst] = pkv_k[src];
    pkv_vc[dst] = pkv_v[src];
}

struct GowPfAdParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    window: u32,
    pad0: u32,
    scale: f32,
};

@group(0) @binding(40) var<storage, read> pad_q: array<u32>;
@group(0) @binding(41) var<storage, read> pad_kc: array<u32>;
@group(0) @binding(42) var<storage, read> pad_vc: array<u32>;
@group(0) @binding(43) var<storage, read_write> pad_scores: array<f32>;
@group(0) @binding(44) var<storage, read_write> pad_out: array<f32>;
@group(0) @binding(45) var<uniform> pad_p: GowPfAdParams;
@group(0) @binding(46) var<storage, read> pad_sinks: array<f32>;
@group(0) @binding(47) var<uniform> pad_ck: GowCkParams;

var<workgroup> pad_qs: array<f32, 256>;
var<workgroup> pad_red: array<f32, 256>;
var<workgroup> pad_max: f32;
var<workgroup> pad_z: f32;

@compute @workgroup_size(256)
fn gow_pf_attn(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tk = wid.y;
    if (tk >= pad_ck.m_live) {
        return;
    }
    let tid = lid.x;
    let hd = pad_p.head_dim;
    let p = pad_ck.base + tk;
    let total = p + 1u;
    var start = 0u;
    if (pad_p.window > 0u && total > pad_p.window) {
        start = total - pad_p.window;
    }
    let kv = h / pad_p.group;
    let srow = (tk * pad_p.n_heads + h) * pad_p.max_seq;
    let qrow = (tk * pad_p.n_heads + h) * hd;
    let sink = pad_sinks[h];

    for (var d = tid; d < hd; d = d + 256u) {
        let idx = qrow + d;
        pad_qs[d] = bf16_decode(u16_at(pad_q[idx >> 1u], idx));
    }
    workgroupBarrier();

    var lmax = -3.4028235e38;
    for (var t = start + tid; t < total; t = t + 256u) {
        let kbase = (t * pad_p.n_kv + kv) * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let idx = kbase + d;
            dot = fma(bf16_decode(u16_at(pad_kc[idx >> 1u], idx)), pad_qs[d], dot);
        }
        let s = dot * pad_p.scale;
        pad_scores[srow + t] = s;
        lmax = max(lmax, s);
    }
    pad_red[tid] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            pad_red[tid] = max(pad_red[tid], pad_red[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        pad_max = max(pad_red[0], sink);
    }
    workgroupBarrier();
    let m = pad_max;

    var lsum = 0.0;
    for (var t = start + tid; t < total; t = t + 256u) {
        let e = exp(pad_scores[srow + t] - m);
        pad_scores[srow + t] = e;
        lsum = lsum + e;
    }
    workgroupBarrier();
    pad_red[tid] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            pad_red[tid] = pad_red[tid] + pad_red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        pad_z = pad_red[0] + exp(sink - m);
    }
    workgroupBarrier();
    let z = pad_z;

    if (tid < hd) {
        var acc = 0.0;
        for (var t = start; t < total; t = t + 1u) {
            let idx = (t * pad_p.n_kv + kv) * hd + tid;
            acc = fma(pad_scores[srow + t], bf16_decode(u16_at(pad_vc[idx >> 1u], idx)), acc);
        }
        pad_out[qrow + tid] = acc / z;
    }
}

struct GowPfRtParams {
    n_experts: u32,
    k: u32,
    m: u32,
    pad0: u32,
};

@group(0) @binding(50) var<storage, read> prt_logits: array<u32>;
@group(0) @binding(51) var<storage, read_write> prt_ids: array<u32>;
@group(0) @binding(52) var<storage, read_write> prt_w: array<f32>;
@group(0) @binding(53) var<uniform> prt_p: GowPfRtParams;

@compute @workgroup_size(1)
fn gow_pf_router_topk(@builtin(workgroup_id) wid: vec3<u32>) {
    let r = wid.x;
    if (r >= prt_p.m) {
        return;
    }
    let lbase = r * prt_p.n_experts;
    let obase = r * prt_p.k;
    var taken: array<u32, 256>;
    var chosen: array<f32, 16>;
    for (var i = 0u; i < prt_p.n_experts; i = i + 1u) {
        taken[i] = 0u;
    }
    for (var j = 0u; j < prt_p.k; j = j + 1u) {
        var best = -3.4028235e38;
        var bi = 0u;
        var found = false;
        for (var i = 0u; i < prt_p.n_experts; i = i + 1u) {
            if (taken[i] == 1u) {
                continue;
            }
            let v = bitcast<f32>(prt_logits[lbase + i]);
            if (!found || v > best) {
                best = v;
                bi = i;
                found = true;
            }
        }
        taken[bi] = 1u;
        prt_ids[obase + j] = bi;
        chosen[j] = best;
    }
    var m = chosen[0];
    for (var j = 1u; j < prt_p.k; j = j + 1u) {
        m = max(m, chosen[j]);
    }
    var s = 0.0;
    for (var j = 0u; j < prt_p.k; j = j + 1u) {
        let e = exp(chosen[j] - m);
        chosen[j] = e;
        s = s + e;
    }
    for (var j = 0u; j < prt_p.k; j = j + 1u) {
        prt_w[obase + j] = chosen[j] / s;
    }
}

struct GowPfMxParams {
    n_rows: u32,
    k_blocks: u32,
    groups_x: u32,
    has_bias: u32,
    w_e_stride_v4: u32,
    sf_e_stride_bytes: u32,
    bias_e_stride: u32,
    x_slot_stride_words: u32,
    x_tok_stride_words: u32,
    y_slot_stride: u32,
    k_top: u32,
    pad0: u32,
};

@group(0) @binding(60) var<storage, read> pmx_w: array<vec4<u32>>;
@group(0) @binding(61) var<storage, read> pmx_sf: array<u32>;
@group(0) @binding(62) var<storage, read> pmx_x: array<u32>;
@group(0) @binding(63) var<uniform> pmx_p: GowPfMxParams;
@group(0) @binding(64) var<storage, read_write> pmx_y: array<f32>;
@group(0) @binding(65) var<storage, read> pmx_sel: array<u32>;
@group(0) @binding(66) var<storage, read> pmx_b: array<u32>;

var<workgroup> pmx_red: array<f32, 256>;

fn pmx_e8m0(byte: u32) -> f32 {
    return bitcast<f32>((byte & 255u) << 23u);
}

fn pmx_e2m1(word: u32, j: u32) -> f32 {
    let n = nvfp4_nibble(word, j);
    let c = n & 7u;
    let normal = (((c >> 1u) + 126u) << 23u) | ((c & 1u) << 22u);
    let mag = select(normal, c * 0x3F000000u, c < 2u);
    return bitcast<f32>(mag | ((n & 8u) << 28u));
}

fn pmx_dot_word(word: u32, xbase: u32) -> f32 {
    var a = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
        let xw = pmx_x[xbase + (j >> 1u)];
        let xv = select(bf16_lo(xw), bf16_hi(xw), (j & 1u) == 1u);
        a = fma(pmx_e2m1(word, j), xv, a);
    }
    return a;
}

@compute @workgroup_size(256)
fn gow_pf_gemv_mx(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let slot = wid.z;
    let e = pmx_sel[slot];
    let pair = wid.x + wid.y * pmx_p.groups_x;
    let row = pair * 2u + half;
    let live = row < pmx_p.n_rows;
    let blocks = select(0u, pmx_p.k_blocks, live);
    let wbase = e * pmx_p.w_e_stride_v4 + row * pmx_p.k_blocks;
    let sfbase = e * pmx_p.sf_e_stride_bytes + row * pmx_p.k_blocks;
    let xbase = (slot / pmx_p.k_top) * pmx_p.x_tok_stride_words
        + slot * pmx_p.x_slot_stride_words;

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + 128u) {
        let wv = pmx_w[wbase + kb];
        let xb = xbase + kb * 16u;
        var dot = pmx_dot_word(wv.x, xb);
        dot = dot + pmx_dot_word(wv.y, xb + 4u);
        dot = dot + pmx_dot_word(wv.z, xb + 8u);
        dot = dot + pmx_dot_word(wv.w, xb + 12u);
        let si = sfbase + kb;
        acc = fma(pmx_e8m0(byte_at(pmx_sf[si >> 2u], si)), dot, acc);
    }
    pmx_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            pmx_red[tid] = pmx_red[tid] + pmx_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u && live) {
        var v = pmx_red[tid];
        if (pmx_p.has_bias == 1u) {
            let bi = e * pmx_p.bias_e_stride + row;
            v = v + bf16_decode(u16_at(pmx_b[bi >> 1u], bi));
        }
        pmx_y[slot * pmx_p.y_slot_stride + row] = v;
    }
}

struct GowPfCbParams {
    hidden_words: u32,
    k: u32,
    slot_stride: u32,
    m: u32,
};

@group(0) @binding(70) var<storage, read> pcb_y: array<f32>;
@group(0) @binding(71) var<storage, read> pcb_w: array<f32>;
@group(0) @binding(72) var<storage, read_write> pcb_out: array<u32>;
@group(0) @binding(73) var<uniform> pcb_p: GowPfCbParams;

@compute @workgroup_size(64)
fn gow_pf_moe_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let t = gid.y;
    if (w >= pcb_p.hidden_words || t >= pcb_p.m) {
        return;
    }
    let sbase = t * pcb_p.k;
    var a0 = 0.0;
    var a1 = 0.0;
    for (var j = 0u; j < pcb_p.k; j = j + 1u) {
        let base = (sbase + j) * pcb_p.slot_stride + 2u * w;
        let wt = pcb_w[sbase + j];
        a0 = fma(pcb_y[base], wt, a0);
        a1 = fma(pcb_y[base + 1u], wt, a1);
    }
    pcb_out[t * pcb_p.hidden_words + w] = bf16_pack(a0, a1);
}
