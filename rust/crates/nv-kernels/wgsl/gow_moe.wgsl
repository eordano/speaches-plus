
struct GowRtParams {
    n_experts: u32,
    k: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> grt_logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> grt_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> grt_w: array<f32>;
@group(0) @binding(3) var<uniform> grt_p: GowRtParams;

@compute @workgroup_size(1)
fn gow_router_topk() {
    var taken: array<u32, 256>;
    var chosen: array<f32, 16>;
    for (var i = 0u; i < grt_p.n_experts; i = i + 1u) {
        taken[i] = 0u;
    }
    for (var j = 0u; j < grt_p.k; j = j + 1u) {
        var best = -3.4028235e38;
        var bi = 0u;
        var found = false;
        for (var i = 0u; i < grt_p.n_experts; i = i + 1u) {
            if (taken[i] == 1u) {
                continue;
            }
            let v = bitcast<f32>(grt_logits[i]);
            if (!found || v > best) {
                best = v;
                bi = i;
                found = true;
            }
        }
        taken[bi] = 1u;
        grt_ids[j] = bi;
        chosen[j] = best;
    }
    var m = chosen[0];
    for (var j = 1u; j < grt_p.k; j = j + 1u) {
        m = max(m, chosen[j]);
    }
    var s = 0.0;
    for (var j = 0u; j < grt_p.k; j = j + 1u) {
        let e = exp(chosen[j] - m);
        chosen[j] = e;
        s = s + e;
    }
    for (var j = 0u; j < grt_p.k; j = j + 1u) {
        grt_w[j] = chosen[j] / s;
    }
}

struct GowSwParams {
    n_words: u32,
    inter_words: u32,
    two_inter: u32,
    pad0: u32,
    limit: f32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> gsw_gu: array<f32>;
@group(0) @binding(11) var<storage, read_write> gsw_act: array<u32>;
@group(0) @binding(12) var<uniform> gsw_p: GowSwParams;

fn gow_swiglu_one(gate_raw: f32, up_raw: f32) -> f32 {
    let g = min(gate_raw, gsw_p.limit);
    let u = clamp(up_raw, -gsw_p.limit, gsw_p.limit);
    let glu = g / (1.0 + exp(-gsw_p.alpha * g));
    return glu * (u + 1.0);
}

@compute @workgroup_size(64)
fn gow_swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= gsw_p.n_words) {
        return;
    }
    let slot = w / gsw_p.inter_words;
    let lw = w % gsw_p.inter_words;
    let base = slot * gsw_p.two_inter;
    let j0 = 2u * lw;
    let a0 = gow_swiglu_one(gsw_gu[base + 2u * j0], gsw_gu[base + 2u * j0 + 1u]);
    let a1 = gow_swiglu_one(gsw_gu[base + 2u * j0 + 2u], gsw_gu[base + 2u * j0 + 3u]);
    gsw_act[w] = bf16_pack(a0, a1);
}

struct GowCbParams {
    hidden_words: u32,
    k: u32,
    slot_stride: u32,
    pad0: u32,
};

@group(0) @binding(20) var<storage, read> gcb_y: array<f32>;
@group(0) @binding(21) var<storage, read> gcb_w: array<f32>;
@group(0) @binding(22) var<storage, read_write> gcb_out: array<u32>;
@group(0) @binding(23) var<uniform> gcb_p: GowCbParams;

@compute @workgroup_size(64)
fn gow_moe_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= gcb_p.hidden_words) {
        return;
    }
    var a0 = 0.0;
    var a1 = 0.0;
    for (var j = 0u; j < gcb_p.k; j = j + 1u) {
        let base = j * gcb_p.slot_stride + 2u * w;
        let wt = gcb_w[j];
        a0 = fma(gcb_y[base], wt, a0);
        a1 = fma(gcb_y[base + 1u], wt, a1);
    }
    gcb_out[w] = bf16_pack(a0, a1);
}

struct GowGeParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
};

@group(0) @binding(30) var<storage, read> gge_emb: array<u32>;
@group(0) @binding(31) var<storage, read> gge_tok: array<i32>;
@group(0) @binding(32) var<storage, read_write> gge_out: array<u32>;
@group(0) @binding(33) var<uniform> gge_p: GowGeParams;

@compute @workgroup_size(256)
fn gow_gather_embed(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    var s = 0u;
    if (gge_tok[0] > 0) {
        s = u32(gge_tok[0]);
    }
    if (s >= gge_p.vocab) {
        s = 0u;
    }
    if (s < gge_p.row_off) {
        return;
    }
    if (s >= gge_p.row_off + gge_p.n_rows) {
        return;
    }
    let base = (s - gge_p.row_off) * gge_p.hidden_words;
    let w = wid.x * 256u + lid.x;
    if (w >= gge_p.hidden_words) {
        return;
    }
    gge_out[w] = gge_emb[base + w];
}

struct GowAmParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(40) var<storage, read> gam_x: array<u32>;
@group(0) @binding(41) var<storage, read_write> gam_pv: array<f32>;
@group(0) @binding(42) var<storage, read_write> gam_pi: array<u32>;
@group(0) @binding(43) var<storage, read_write> gam_out: array<u32>;
@group(0) @binding(44) var<uniform> gam_p: GowAmParams;

var<workgroup> gam_v: array<f32, 256>;
var<workgroup> gam_i: array<u32, 256>;

fn gam_reduce(tid: u32) {
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let o = tid + s;
            if (gam_v[o] > gam_v[tid] || (gam_v[o] == gam_v[tid] && gam_i[o] < gam_i[tid])) {
                gam_v[tid] = gam_v[o];
                gam_i[tid] = gam_i[o];
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn gow_argmax_stage1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let g = wid.x;
    let tid = lid.x;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    for (var i = g * 256u + tid; i < gam_p.n; i = i + gam_p.groups * 256u) {
        let v = bitcast<f32>(gam_x[i]);
        if (v > bv || (v == bv && i < bi)) {
            bv = v;
            bi = i;
        }
    }
    gam_v[tid] = bv;
    gam_i[tid] = bi;
    workgroupBarrier();
    gam_reduce(tid);
    if (tid == 0u) {
        gam_pv[g] = gam_v[0];
        gam_pi[g] = gam_i[0];
    }
}

@compute @workgroup_size(256)
fn gow_argmax_stage2(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    if (tid < gam_p.groups) {
        bv = gam_pv[tid];
        bi = gam_pi[tid];
    }
    gam_v[tid] = bv;
    gam_i[tid] = bi;
    workgroupBarrier();
    gam_reduce(tid);
    if (tid == 0u) {
        gam_out[0] = gam_i[0];
    }
}
