struct TvParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    k: u32,
    window: u32,
    scaling: f32,
    ring: u32,
    n_committed: u32,
    base: u32,
    n_accept: u32,
    stride_words: u32,
    total: u32,
};

@group(0) @binding(0) var<storage, read> tva_q: array<u32>;
@group(0) @binding(1) var<storage, read> tva_kc: array<u32>;
@group(0) @binding(2) var<storage, read> tva_vc: array<u32>;
@group(0) @binding(3) var<storage, read> tva_mask: array<u32>;
@group(0) @binding(4) var<storage, read> tva_pos: array<i32>;
@group(0) @binding(5) var<storage, read_write> tva_out: array<u32>;
@group(0) @binding(6) var<uniform> tva_p: TvParams;

@group(0) @binding(7) var<storage, read> tvap_ksrc: array<u32>;
@group(0) @binding(8) var<storage, read> tvap_vsrc: array<u32>;
@group(0) @binding(9) var<storage, read_write> tvap_kc: array<u32>;
@group(0) @binding(10) var<storage, read_write> tvap_vc: array<u32>;
@group(0) @binding(11) var<uniform> tvap_p: TvParams;

@group(0) @binding(12) var<storage, read_write> tvac_kc: array<u32>;
@group(0) @binding(13) var<storage, read_write> tvac_vc: array<u32>;
@group(0) @binding(14) var<storage, read_write> tvac_sk: array<u32>;
@group(0) @binding(15) var<storage, read_write> tvac_sv: array<u32>;
@group(0) @binding(16) var<storage, read> tvac_path: array<i32>;
@group(0) @binding(17) var<uniform> tvac_p: TvParams;

const TV_WARP: u32 = 32u;
const TV_WARPS: u32 = 8u;
const TV_BLOCK: u32 = 256u;
const TV_MAX_HD: u32 = 512u;
const TV_MAX_ACC: u32 = 16u;
const TV_LOG2_E: f32 = 1.4426950408889634;

var<workgroup> tva_qsh: array<f32, 512>;
var<workgroup> tva_red: array<f32, 256>;
var<workgroup> tva_sm: array<f32, 8>;
var<workgroup> tva_sl: array<f32, 8>;
var<workgroup> tva_sacc: array<f32, 4096>;

fn tv_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn tv_exp(x: f32) -> f32 {
    return exp2(x * TV_LOG2_E);
}

fn tv_recip(x: f32) -> f32 {
    let r = 1.0 / x;
    return fma(fma(-x, r, 1.0), r, r);
}

fn tva_warp_sum(tid: u32, x: f32) -> f32 {
    var v = x;
    tva_red[tid] = v;
    workgroupBarrier();
    for (var o = 16u; o > 0u; o = o >> 1u) {
        let other = tva_red[tid ^ o];
        workgroupBarrier();
        v = v + other;
        tva_red[tid] = v;
        workgroupBarrier();
    }
    return v;
}

fn tva_q_at(i: u32) -> f32 {
    let w = tva_q[i >> 1u];
    return select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
}

fn tva_kc_at(i: u32) -> f32 {
    let w = tva_kc[i >> 1u];
    return select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
}

fn tva_vc_at(i: u32) -> f32 {
    let w = tva_vc[i >> 1u];
    return select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
}

@compute @workgroup_size(256)
fn tree_verify_attn_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg.x;
    let qi = wg.y;
    if (h >= tva_p.n_heads || qi >= tva_p.k) {
        return;
    }
    let hd = tva_p.head_dim;
    let nkv = tva_p.n_kv_heads;
    let nc = tva_p.n_committed;
    let group = tva_p.n_heads / nkv;
    let kvh = h / group;
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;

    var qpos = 0i;
    var win_start = 0u;
    if (tva_p.window > 0u) {
        qpos = tva_pos[qi];
        let s = qpos - i32(tva_p.window) + 1;
        if (s > 0) {
            win_start = u32(s);
        }
    }

    let qbase = (qi * tva_p.n_heads + h) * hd;
    for (var d = tid; d < hd; d = d + TV_BLOCK) {
        tva_qsh[d] = tva_q_at(qbase + d);
    }
    workgroupBarrier();

    var acc: array<f32, 16>;
    for (var i = 0u; i < TV_MAX_ACC; i = i + 1u) {
        acc[i] = 0.0;
    }
    var m = tv_neg_inf();
    var l = 0.0;

    let n_iter = (nc + TV_WARPS - 1u) / TV_WARPS;
    for (var it = 0u; it < n_iter; it = it + 1u) {
        let p = win_start + warp + it * TV_WARPS;
        let live = p < nc;
        var partial = 0.0;
        if (live) {
            let kbase = (p * nkv + kvh) * hd;
            for (var d = lane; d < hd; d = d + TV_WARP) {
                partial = fma(tva_qsh[d], tva_kc_at(kbase + d), partial);
            }
        }
        let reduced = tva_warp_sum(tid, partial);
        if (live) {
            let score = reduced * tva_p.scaling;
            let m_new = max(m, score);
            let corr = tv_exp(m - m_new);
            let w = tv_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (p * nkv + kvh) * hd;
            for (var i = 0u; i < TV_MAX_ACC; i = i + 1u) {
                let d = lane + i * TV_WARP;
                if (d < hd) {
                    acc[i] = fma(acc[i], corr, w * tva_vc_at(vbase + d));
                }
            }
            m = m_new;
        }
    }

    let n_tree = (tva_p.k + TV_WARPS - 1u) / TV_WARPS;
    for (var it = 0u; it < n_tree; it = it + 1u) {
        let j = warp + it * TV_WARPS;
        var live = j < tva_p.k;
        if (live) {
            let mi = qi * tva_p.k + j;
            if (byte_at(tva_mask[mi >> 2u], mi) == 0u) {
                live = false;
            }
        }
        if (live && tva_p.window > 0u) {
            if (qpos - tva_pos[j] >= i32(tva_p.window)) {
                live = false;
            }
        }
        var partial = 0.0;
        var p = 0u;
        if (live) {
            p = nc + j;
            let kbase = (p * nkv + kvh) * hd;
            for (var d = lane; d < hd; d = d + TV_WARP) {
                partial = fma(tva_qsh[d], tva_kc_at(kbase + d), partial);
            }
        }
        let reduced = tva_warp_sum(tid, partial);
        if (live) {
            let score = reduced * tva_p.scaling;
            let m_new = max(m, score);
            let corr = tv_exp(m - m_new);
            let w = tv_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (p * nkv + kvh) * hd;
            for (var i = 0u; i < TV_MAX_ACC; i = i + 1u) {
                let d = lane + i * TV_WARP;
                if (d < hd) {
                    acc[i] = fma(acc[i], corr, w * tva_vc_at(vbase + d));
                }
            }
            m = m_new;
        }
    }

    if (lane == 0u) {
        tva_sm[warp] = m;
        tva_sl[warp] = l;
    }
    for (var i = 0u; i < TV_MAX_ACC; i = i + 1u) {
        let d = lane + i * TV_WARP;
        if (d < hd) {
            tva_sacc[warp * TV_MAX_HD + d] = acc[i];
        }
    }
    workgroupBarrier();

    if (warp == 0u) {
        var m_glob = tv_neg_inf();
        for (var w = 0u; w < TV_WARPS; w = w + 1u) {
            m_glob = max(m_glob, tva_sm[w]);
        }
        var l_glob = 0.0;
        for (var w = 0u; w < TV_WARPS; w = w + 1u) {
            var t = 0.0;
            if (tva_sm[w] > tv_neg_inf()) {
                t = fma(tva_sl[w], tv_exp(tva_sm[w] - m_glob), -0.0);
            }
            l_glob = l_glob + t;
        }
        var inv = 0.0;
        if (l_glob > 0.0) {
            inv = tv_recip(l_glob);
        }
        let obase = (qi * tva_p.n_heads + h) * hd;
        let words = hd >> 1u;
        for (var ww = lane; ww < words; ww = ww + TV_WARP) {
            let d0 = ww * 2u;
            var a0 = 0.0;
            var a1 = 0.0;
            for (var w = 0u; w < TV_WARPS; w = w + 1u) {
                if (tva_sm[w] > tv_neg_inf()) {
                    let e = tv_exp(tva_sm[w] - m_glob);
                    a0 = a0 + fma(tva_sacc[w * TV_MAX_HD + d0], e, -0.0);
                    a1 = a1 + fma(tva_sacc[w * TV_MAX_HD + d0 + 1u], e, -0.0);
                }
            }
            tva_out[(obase >> 1u) + ww] = bf16_pack(a0 * inv, a1 * inv);
        }
    }
}

@compute @workgroup_size(256)
fn kv_append_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = (wg.y * ng.x + wg.x) * TV_BLOCK + lid.x;
    if (idx >= tvap_p.total) {
        return;
    }
    let row = tvap_p.stride_words;
    let token = idx / row;
    let e = idx - token * row;
    var slot = tvap_p.n_committed + token;
    if (tvap_p.ring > 0u) {
        if (token + tvap_p.ring < tvap_p.k) {
            return;
        }
        slot = slot % tvap_p.ring;
    }
    let dst = slot * row + e;
    tvap_kc[dst] = tvap_ksrc[idx];
    tvap_vc[dst] = tvap_vsrc[idx];
}

@compute @workgroup_size(256)
fn kv_gather_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = (wg.y * ng.x + wg.x) * TV_BLOCK + lid.x;
    if (idx >= tvac_p.total) {
        return;
    }
    let row = tvac_p.stride_words;
    let i = idx / row;
    let e = idx - i * row;
    var srow = tvac_p.base + u32(max(tvac_path[i], 0));
    if (tvac_p.ring > 0u) {
        srow = srow % tvac_p.ring;
    }
    let src = srow * row + e;
    tvac_sk[idx] = tvac_kc[src];
    tvac_sv[idx] = tvac_vc[src];
}

@compute @workgroup_size(256)
fn kv_scatter_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = (wg.y * ng.x + wg.x) * TV_BLOCK + lid.x;
    if (idx >= tvac_p.total) {
        return;
    }
    let row = tvac_p.stride_words;
    let i = idx / row;
    let e = idx - i * row;
    var drow = tvac_p.base + i;
    if (tvac_p.ring > 0u) {
        if (i + tvac_p.ring < tvac_p.n_accept) {
            return;
        }
        drow = drow % tvac_p.ring;
    }
    let dst = drow * row + e;
    tvac_kc[dst] = tvac_sk[idx];
    tvac_vc[dst] = tvac_sv[idx];
}
