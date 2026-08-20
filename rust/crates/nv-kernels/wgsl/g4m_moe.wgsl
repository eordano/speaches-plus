
struct G4tParams {
    n_experts: u32,
    k: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> gt_logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> gt_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> gt_w: array<f32>;
@group(0) @binding(3) var<uniform> gt_p: G4tParams;
@group(0) @binding(4) var<storage, read> gt_pes: array<f32>;

@compute @workgroup_size(1)
fn g4m_router_topk() {
    var taken: array<u32, 256>;
    var chosen: array<f32, 16>;
    for (var i = 0u; i < gt_p.n_experts; i = i + 1u) {
        taken[i] = 0u;
    }
    for (var j = 0u; j < gt_p.k; j = j + 1u) {
        var best = -3.4028235e38;
        var bi = 0u;
        var found = false;
        for (var i = 0u; i < gt_p.n_experts; i = i + 1u) {
            if (taken[i] == 1u) {
                continue;
            }
            let v = bitcast<f32>(gt_logits[i]);
            if (!found || v > best) {
                best = v;
                bi = i;
                found = true;
            }
        }
        taken[bi] = 1u;
        gt_ids[j] = bi;
        chosen[j] = best;
    }
    var m = chosen[0];
    for (var j = 1u; j < gt_p.k; j = j + 1u) {
        m = max(m, chosen[j]);
    }
    var ssum = 0.0;
    for (var j = 0u; j < gt_p.k; j = j + 1u) {
        let e = exp(chosen[j] - m);
        chosen[j] = e;
        ssum = ssum + e;
    }
    for (var j = 0u; j < gt_p.k; j = j + 1u) {
        gt_w[j] = chosen[j] / ssum * gt_pes[gt_ids[j]];
    }
}

var<workgroup> tk_v: array<f32, 256>;
var<workgroup> tk_ch: array<f32, 16>;
var<workgroup> tk_id: array<u32, 16>;

@compute @workgroup_size(256)
fn g4m_router_topk_par(@builtin(local_invocation_id) lid3: vec3<u32>) {
    let tid = lid3.x;
    let ne = gt_p.n_experts;
    if (tid < ne) {
        tk_v[tid] = bitcast<f32>(gt_logits[tid]);
    }
    workgroupBarrier();
    if (tid < ne) {
        let v = tk_v[tid];
        var r = 0u;
        for (var j = 0u; j < ne; j = j + 1u) {
            let w = tk_v[j];
            if (w > v || (w == v && j < tid)) {
                r = r + 1u;
            }
        }
        if (r < gt_p.k) {
            tk_id[r] = tid;
            tk_ch[r] = v;
            gt_ids[r] = tid;
        }
    }
    workgroupBarrier();
    if (tid == 0u) {
        var m = tk_ch[0];
        for (var j = 1u; j < gt_p.k; j = j + 1u) {
            m = max(m, tk_ch[j]);
        }
        var ssum = 0.0;
        for (var j = 0u; j < gt_p.k; j = j + 1u) {
            let e = exp(tk_ch[j] - m);
            tk_ch[j] = e;
            ssum = ssum + e;
        }
        for (var j = 0u; j < gt_p.k; j = j + 1u) {
            gt_w[j] = tk_ch[j] / ssum * gt_pes[tk_id[j]];
        }
    }
}

struct G4eParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> ge_g: array<u32>;
@group(0) @binding(11) var<storage, read> ge_u: array<u32>;
@group(0) @binding(12) var<storage, read_write> ge_y: array<u32>;
@group(0) @binding(13) var<uniform> ge_p: G4eParams;

fn g4m_gelu(x: f32) -> f32 {
    let c = 0.7978845608028654;
    let t = nv_tanhf(c * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1.0 + t);
}

@compute @workgroup_size(64)
fn g4m_gelu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= ge_p.n_words) {
        return;
    }
    let gw = ge_g[w];
    let uw = ge_u[w];
    let a0 = bf16_decode(bf16_encode(g4m_gelu(bf16_lo(gw)))) * bf16_lo(uw);
    let a1 = bf16_decode(bf16_encode(g4m_gelu(bf16_hi(gw)))) * bf16_hi(uw);
    ge_y[w] = bf16_pack(a0, a1);
}

struct G4cParams {
    hidden_words: u32,
    k: u32,
    slot_stride_words: u32,
    pad0: u32,
};

@group(0) @binding(20) var<storage, read> gc_y: array<u32>;
@group(0) @binding(21) var<storage, read> gc_w: array<f32>;
@group(0) @binding(22) var<storage, read_write> gc_out: array<u32>;
@group(0) @binding(23) var<uniform> gc_p: G4cParams;

@compute @workgroup_size(64)
fn g4m_moe_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= gc_p.hidden_words) {
        return;
    }
    var a0 = 0.0;
    var a1 = 0.0;
    for (var j = 0u; j < gc_p.k; j = j + 1u) {
        let word = gc_y[j * gc_p.slot_stride_words + w];
        let wt = gc_w[j];
        a0 = fma(bf16_lo(word), wt, a0);
        a1 = fma(bf16_hi(word), wt, a1);
    }
    gc_out[w] = bf16_pack(a0, a1);
}

struct G4gParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(30) var<storage, read> gg_emb: array<u32>;
@group(0) @binding(31) var<storage, read> gg_tok: array<i32>;
@group(0) @binding(32) var<storage, read_write> gg_out: array<u32>;
@group(0) @binding(33) var<uniform> gg_p: G4gParams;

@compute @workgroup_size(256)
fn g4m_gather_embed(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    var t = 0u;
    if (gg_tok[0] > 0) {
        t = u32(gg_tok[0]);
    }
    if (t >= gg_p.vocab) {
        t = 0u;
    }
    if (t < gg_p.row_off) {
        return;
    }
    if (t >= gg_p.row_off + gg_p.n_rows) {
        return;
    }
    let base = (t - gg_p.row_off) * gg_p.hidden_words;
    let w = wid.x * 256u + lid.x;
    if (w >= gg_p.hidden_words) {
        return;
    }
    let word = gg_emb[base + w];
    gg_out[w] = bf16_pack(bf16_lo(word) * gg_p.scale, bf16_hi(word) * gg_p.scale);
}

struct G4xParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(40) var<storage, read> gx_x: array<u32>;
@group(0) @binding(41) var<storage, read> gx_s: array<u32>;
@group(0) @binding(42) var<storage, read_write> gx_y: array<u32>;
@group(0) @binding(43) var<uniform> gx_p: G4xParams;

@compute @workgroup_size(64)
fn g4m_mul_bf16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= gx_p.n_words) {
        return;
    }
    let xw = gx_x[w];
    let sw = gx_s[w];
    gx_y[w] = bf16_pack(bf16_lo(xw) * bf16_lo(sw), bf16_hi(xw) * bf16_hi(sw));
}

struct G4pParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(50) var<storage, read> gp_x: array<f32>;
@group(0) @binding(51) var<storage, read_write> gp_y: array<u32>;
@group(0) @binding(52) var<uniform> gp_p: G4pParams;

@compute @workgroup_size(64)
fn g4m_pack_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= gp_p.n_words) {
        return;
    }
    gp_y[w] = bf16_pack(gp_x[w * 2u], gp_x[w * 2u + 1u]);
}

struct G4amParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(60) var<storage, read> am_x: array<u32>;
@group(0) @binding(61) var<storage, read_write> am_pv: array<f32>;
@group(0) @binding(62) var<storage, read_write> am_pi: array<u32>;
@group(0) @binding(63) var<storage, read_write> am_out: array<u32>;
@group(0) @binding(64) var<uniform> am_p: G4amParams;

var<workgroup> am_v: array<f32, 256>;
var<workgroup> am_i: array<u32, 256>;

fn am_reduce(tid: u32) {
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let o = tid + s;
            if (am_v[o] > am_v[tid] || (am_v[o] == am_v[tid] && am_i[o] < am_i[tid])) {
                am_v[tid] = am_v[o];
                am_i[tid] = am_i[o];
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn g4m_argmax_bf16_stage1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let g = wid.x;
    let tid = lid.x;
    let n_words = (am_p.n + 1u) / 2u;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    for (var w = g * 256u + tid; w < n_words; w = w + am_p.groups * 256u) {
        let word = am_x[w];
        let i0 = w * 2u;
        let v0 = bf16_lo(word);
        if (i0 < am_p.n && (v0 > bv || (v0 == bv && i0 < bi))) {
            bv = v0;
            bi = i0;
        }
        let v1 = bf16_hi(word);
        if (i0 + 1u < am_p.n && (v1 > bv || (v1 == bv && i0 + 1u < bi))) {
            bv = v1;
            bi = i0 + 1u;
        }
    }
    am_v[tid] = bv;
    am_i[tid] = bi;
    workgroupBarrier();
    am_reduce(tid);
    if (tid == 0u) {
        am_pv[g] = am_v[0];
        am_pi[g] = am_i[0];
    }
}

@compute @workgroup_size(256)
fn g4m_argmax_stage2(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    if (tid < am_p.groups) {
        bv = am_pv[tid];
        bi = am_pi[tid];
    }
    am_v[tid] = bv;
    am_i[tid] = bi;
    workgroupBarrier();
    am_reduce(tid);
    if (tid == 0u) {
        am_out[0] = am_i[0];
    }
}
