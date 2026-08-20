
struct Q3rtParams {
    n_experts: u32,
    k: u32,
    shared_slot: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> rt_logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> rt_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> rt_w: array<f32>;
@group(0) @binding(3) var<uniform> rt_p: Q3rtParams;

@compute @workgroup_size(1)
fn q3w_router_topk() {
    var taken: array<u32, 256>;
    var chosen: array<f32, 16>;
    for (var i = 0u; i < rt_p.n_experts; i = i + 1u) {
        taken[i] = 0u;
    }
    for (var j = 0u; j < rt_p.k; j = j + 1u) {
        var best = -3.4028235e38;
        var bi = 0u;
        var found = false;
        for (var i = 0u; i < rt_p.n_experts; i = i + 1u) {
            if (taken[i] == 1u) {
                continue;
            }
            let v = bitcast<f32>(rt_logits[i]);
            if (!found || v > best) {
                best = v;
                bi = i;
                found = true;
            }
        }
        taken[bi] = 1u;
        rt_ids[j] = bi;
        chosen[j] = best;
    }
    var m = chosen[0];
    for (var j = 1u; j < rt_p.k; j = j + 1u) {
        m = max(m, chosen[j]);
    }
    var s = 0.0;
    for (var j = 0u; j < rt_p.k; j = j + 1u) {
        let e = exp(chosen[j] - m);
        chosen[j] = e;
        s = s + e;
    }
    for (var j = 0u; j < rt_p.k; j = j + 1u) {
        rt_w[j] = chosen[j] / s;
    }
    if (rt_p.shared_slot == 1u) {
        rt_ids[rt_p.k] = rt_p.n_experts;
    }
}

var<workgroup> rtp_v: array<f32, 256>;
var<workgroup> rtp_chosen: array<f32, 16>;

var<workgroup> rtt_v: array<f32, 256>;
var<workgroup> rtt_i: array<u32, 256>;
var<workgroup> rtt_taken: array<u32, 256>;
var<workgroup> rtt_chosen: array<f32, 16>;

@compute @workgroup_size(256)
fn q3w_router_topk_tree(@builtin(local_invocation_id) lid: vec3<u32>) {
    let t = lid.x;
    if (t < rt_p.n_experts) {
        rtt_taken[t] = 0u;
    }
    workgroupBarrier();

    for (var j = 0u; j < rt_p.k; j = j + 1u) {
        var v = -3.4028235e38;
        var idx = 0xffffffffu;
        if (t < rt_p.n_experts && rtt_taken[t] == 0u) {
            v = bitcast<f32>(rt_logits[t]);
            idx = t;
        }
        rtt_v[t] = v;
        rtt_i[t] = idx;
        workgroupBarrier();

        for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
            if (t < stride) {
                let av = rtt_v[t];
                let ai = rtt_i[t];
                let bv = rtt_v[t + stride];
                let bi = rtt_i[t + stride];
                let a_dead = ai == 0xffffffffu;
                let b_live = bi != 0xffffffffu;
                let take_b = (a_dead && b_live)
                    || (b_live && !a_dead && (bv > av || (bv == av && bi < ai)));
                if (take_b) {
                    rtt_v[t] = bv;
                    rtt_i[t] = bi;
                }
            }
            workgroupBarrier();
        }

        if (t == 0u) {
            let bi = rtt_i[0];
            rtt_taken[bi] = 1u;
            rt_ids[j] = bi;
            rtt_chosen[j] = rtt_v[0];
        }
        workgroupBarrier();
    }

    if (t == 0u) {
        var m = rtt_chosen[0];
        for (var j = 1u; j < rt_p.k; j = j + 1u) {
            m = max(m, rtt_chosen[j]);
        }
        var s = 0.0;
        for (var j = 0u; j < rt_p.k; j = j + 1u) {
            let e = exp(rtt_chosen[j] - m);
            rtt_chosen[j] = e;
            s = s + e;
        }
        for (var j = 0u; j < rt_p.k; j = j + 1u) {
            rt_w[j] = rtt_chosen[j] / s;
        }
        if (rt_p.shared_slot == 1u) {
            rt_ids[rt_p.k] = rt_p.n_experts;
        }
    }
}

struct Q3smParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> sm_g: array<u32>;
@group(0) @binding(11) var<storage, read> sm_u: array<u32>;
@group(0) @binding(12) var<storage, read_write> sm_y: array<u32>;
@group(0) @binding(13) var<uniform> sm_p: Q3smParams;

@compute @workgroup_size(64)
fn q3w_silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= sm_p.n_words) {
        return;
    }
    let gw = sm_g[w];
    let uw = sm_u[w];
    let g0 = bf16_lo(gw);
    let g1 = bf16_hi(gw);
    let a0 = bf16_decode(bf16_encode(g0 / (1.0 + exp(-g0)))) * bf16_lo(uw);
    let a1 = bf16_decode(bf16_encode(g1 / (1.0 + exp(-g1)))) * bf16_hi(uw);
    sm_y[w] = bf16_pack(a0, a1);
}

struct Q3mcParams {
    hidden_words: u32,
    k: u32,
    slot_stride_words: u32,
    shared_off_words: u32,
    slogit_off: u32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(20) var<storage, read> mc_y: array<u32>;
@group(0) @binding(21) var<storage, read> mc_w: array<f32>;
@group(0) @binding(22) var<storage, read> mc_shared: array<u32>;
@group(0) @binding(23) var<storage, read> mc_slogit: array<u32>;
@group(0) @binding(24) var<storage, read_write> mc_out: array<u32>;
@group(0) @binding(25) var<uniform> mc_p: Q3mcParams;

@compute @workgroup_size(64)
fn q3w_moe_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= mc_p.hidden_words) {
        return;
    }
    var a0 = 0.0;
    var a1 = 0.0;
    for (var j = 0u; j < mc_p.k; j = j + 1u) {
        let word = mc_y[j * mc_p.slot_stride_words + w];
        let wt = mc_w[j];
        a0 = fma(bf16_lo(word), wt, a0);
        a1 = fma(bf16_hi(word), wt, a1);
    }
    let sg = 1.0 / (1.0 + exp(-bitcast<f32>(mc_slogit[mc_p.slogit_off])));
    let sw = mc_shared[mc_p.shared_off_words + w];
    a0 = a0 + sg * bf16_lo(sw);
    a1 = a1 + sg * bf16_hi(sw);
    mc_out[w] = bf16_pack(a0, a1);
}

struct Q3geParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
};

@group(0) @binding(30) var<storage, read> ge_emb: array<u32>;
@group(0) @binding(31) var<storage, read> ge_tok: array<i32>;
@group(0) @binding(32) var<storage, read_write> ge_out: array<u32>;
@group(0) @binding(33) var<uniform> ge_p: Q3geParams;

@compute @workgroup_size(256)
fn q3w_gather_embed(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    var s = 0u;
    if (ge_tok[0] > 0) {
        s = u32(ge_tok[0]);
    }
    if (s >= ge_p.vocab) {
        s = 0u;
    }
    if (s < ge_p.row_off) {
        return;
    }
    if (s >= ge_p.row_off + ge_p.n_rows) {
        return;
    }
    let base = (s - ge_p.row_off) * ge_p.hidden_words;
    let w = wid.x * 256u + lid.x;
    if (w >= ge_p.hidden_words) {
        return;
    }
    ge_out[w] = ge_emb[base + w];
}

