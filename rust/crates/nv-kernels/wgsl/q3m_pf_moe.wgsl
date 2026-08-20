
struct Q3ptkParams {
    tokens: u32,
    rl_stride_words: u32,
    sel_stride_words: u32,
    pad0: u32,
};

@group(0) @binding(4) var<uniform> ptk_p: Q3ptkParams;

@compute @workgroup_size(256)
fn q3w_pf_router_topk_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tk = wid.x;
    if (tk >= ptk_p.tokens) {
        return;
    }
    let t = lid.x;
    let n = rt_p.n_experts;
    let lb = tk * ptk_p.rl_stride_words;
    let sb = tk * ptk_p.sel_stride_words;
    var vt = 0.0;
    if (t < n) {
        vt = bitcast<f32>(rt_logits[lb + t]);
        rtp_v[t] = vt;
    }
    workgroupBarrier();

    if (t < n) {
        var rank = 0u;
        for (var l = 0u; l < n; l = l + 1u) {
            let vl = rtp_v[l];
            if (vl > vt || (vl == vt && l < t)) {
                rank = rank + 1u;
            }
        }
        if (rank < rt_p.k) {
            rt_ids[sb + rank] = t;
            rtp_chosen[rank] = vt;
        }
    }
    workgroupBarrier();

    if (t == 0u) {
        var m = rtp_chosen[0];
        for (var j = 1u; j < rt_p.k; j = j + 1u) {
            m = max(m, rtp_chosen[j]);
        }
        var s = 0.0;
        for (var j = 0u; j < rt_p.k; j = j + 1u) {
            let e = exp(rtp_chosen[j] - m);
            rtp_chosen[j] = e;
            s = s + e;
        }
        for (var j = 0u; j < rt_p.k; j = j + 1u) {
            rt_w[sb + j] = rtp_chosen[j] / s;
        }
        if (rt_p.shared_slot == 1u) {
            rt_ids[sb + rt_p.k] = rt_p.n_experts;
        }
    }
}

struct Q3psParams {
    tokens: u32,
    slots_per_token: u32,
    ids_stride_words: u32,
    bins_cover_n_experts_plus_the_shared_slot: u32,
};

@group(0) @binding(5) var<uniform> ps_p: Q3psParams;
@group(0) @binding(6) var<storage, read> ps_ids: array<u32>;
@group(0) @binding(7) var<storage, read_write> ps_sorted_sel: array<u32>;
@group(0) @binding(8) var<storage, read_write> ps_perm: array<u32>;

var<workgroup> ps_bins: array<atomic<u32>, 257>;

@compute @workgroup_size(256)
fn q3w_pf_group_slots_by_expert_for_weight_load_reuse(
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = lid.x;
    let bins = ps_p.bins_cover_n_experts_plus_the_shared_slot;
    for (var bi = t; bi < bins; bi = bi + 256u) {
        atomicStore(&ps_bins[bi], 0u);
    }
    workgroupBarrier();
    let zn = ps_p.tokens * ps_p.slots_per_token;
    for (var f = t; f < zn; f = f + 256u) {
        let tok = f / ps_p.slots_per_token;
        let j = f - tok * ps_p.slots_per_token;
        let e = ps_ids[tok * ps_p.ids_stride_words + j];
        atomicAdd(&ps_bins[e], 1u);
    }
    workgroupBarrier();
    if (t == 0u) {
        var acc = 0u;
        for (var bi = 0u; bi < bins; bi = bi + 1u) {
            let c = atomicLoad(&ps_bins[bi]);
            atomicStore(&ps_bins[bi], acc);
            acc = acc + c;
        }
    }
    workgroupBarrier();
    for (var f = t; f < zn; f = f + 256u) {
        let tok = f / ps_p.slots_per_token;
        let j = f - tok * ps_p.slots_per_token;
        let e = ps_ids[tok * ps_p.ids_stride_words + j];
        let z = atomicAdd(&ps_bins[e], 1u);
        ps_sorted_sel[z] = e;
        ps_perm[z] = f;
    }
}

struct Q3pmcParams {
    tokens: u32,
    y_stride_words: u32,
    wts_stride_words: u32,
    slogit_stride_words: u32,
    out_stride_words: u32,
    sel_slots_per_token: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(26) var<uniform> pmc_p: Q3pmcParams;

@compute @workgroup_size(64)
fn q3w_pf_moe_combine_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    let w = wid.x * 64u + lid.x;
    if (w >= mc_p.hidden_words) {
        return;
    }
    let yb = t * pmc_p.y_stride_words;
    let wb = t * pmc_p.wts_stride_words;
    var a0 = 0.0;
    var a1 = 0.0;
    for (var j = 0u; j < mc_p.k; j = j + 1u) {
        let word = mc_y[yb + j * mc_p.slot_stride_words + w];
        let wt = mc_w[wb + j];
        a0 = fma(bf16_lo(word), wt, a0);
        a1 = fma(bf16_hi(word), wt, a1);
    }
    let si = t * pmc_p.slogit_stride_words + mc_p.slogit_off;
    let sg = 1.0 / (1.0 + exp(-bitcast<f32>(mc_slogit[si])));
    let sw = mc_shared[yb + mc_p.shared_off_words + w];
    a0 = a0 + sg * bf16_lo(sw);
    a1 = a1 + sg * bf16_hi(sw);
    mc_out[t * pmc_p.out_stride_words + w] = bf16_pack(a0, a1);
}
