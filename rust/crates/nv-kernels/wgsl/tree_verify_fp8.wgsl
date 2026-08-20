struct TvfParams {
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

@group(0) @binding(20) var<storage, read> tvf_q: array<u32>;
@group(0) @binding(21) var<storage, read> tvf_kc: array<u32>;
@group(0) @binding(22) var<storage, read> tvf_vc: array<u32>;
@group(0) @binding(23) var<storage, read> tvf_ks: array<f32>;
@group(0) @binding(24) var<storage, read> tvf_vs: array<f32>;
@group(0) @binding(25) var<storage, read> tvf_mask: array<u32>;
@group(0) @binding(26) var<storage, read> tvf_pos: array<i32>;
@group(0) @binding(27) var<storage, read_write> tvf_out: array<u32>;
@group(0) @binding(28) var<uniform> tvf_p: TvfParams;

@group(0) @binding(29) var<storage, read> af_ksrc: array<u32>;
@group(0) @binding(30) var<storage, read> af_vsrc: array<u32>;
@group(0) @binding(31) var<storage, read_write> af_kc: array<u32>;
@group(0) @binding(32) var<storage, read_write> af_vc: array<u32>;
@group(0) @binding(33) var<storage, read_write> af_ksc: array<f32>;
@group(0) @binding(34) var<storage, read_write> af_vsc: array<f32>;
@group(0) @binding(35) var<uniform> af_p: TvfParams;

@group(0) @binding(36) var<storage, read_write> cf_kc: array<u32>;
@group(0) @binding(37) var<storage, read_write> cf_vc: array<u32>;
@group(0) @binding(38) var<storage, read_write> cf_sk: array<u32>;
@group(0) @binding(39) var<storage, read_write> cf_sv: array<u32>;
@group(0) @binding(40) var<storage, read> cf_path: array<i32>;
@group(0) @binding(41) var<uniform> cf_p: TvfParams;
@group(0) @binding(42) var<storage, read_write> cf_ksc: array<f32>;
@group(0) @binding(43) var<storage, read_write> cf_vsc: array<f32>;
@group(0) @binding(44) var<storage, read_write> cf_ssk: array<f32>;
@group(0) @binding(45) var<storage, read_write> cf_ssv: array<f32>;

const TVF_WARP: u32 = 32u;
const TVF_WARPS: u32 = 8u;
const TVF_BLOCK: u32 = 256u;
const TVF_MAX_HD: u32 = 512u;
const TVF_MAX_ACC: u32 = 16u;
const TVF_LOG2_E: f32 = 1.4426950408889634;
const TVF_FP8_MAX: f32 = 448.0;

var<workgroup> tvf_qsh: array<f32, 512>;
var<workgroup> tvf_red: array<f32, 256>;
var<workgroup> tvf_sm: array<f32, 8>;
var<workgroup> tvf_sl: array<f32, 8>;
var<workgroup> tvf_sacc: array<f32, 4096>;
var<workgroup> tvf_mk: array<f32, 256>;
var<workgroup> tvf_mv: array<f32, 256>;

fn tvf_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn tvf_exp(x: f32) -> f32 {
    return exp2(x * TVF_LOG2_E);
}

fn tvf_recip(x: f32) -> f32 {
    let r = 1.0 / x;
    return fma(fma(-x, r, 1.0), r, r);
}

fn tvf_warp_sum(tid: u32, x: f32) -> f32 {
    var v = x;
    tvf_red[tid] = v;
    workgroupBarrier();
    for (var o = 16u; o > 0u; o = o >> 1u) {
        let other = tvf_red[tid ^ o];
        workgroupBarrier();
        v = v + other;
        tvf_red[tid] = v;
        workgroupBarrier();
    }
    return v;
}

fn tvf_q_at(i: u32) -> f32 {
    let w = tvf_q[i >> 1u];
    return select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
}

fn tvf_kc_at(i: u32) -> f32 {
    return e4m3_decode(byte_at(tvf_kc[i >> 2u], i));
}

fn tvf_vc_at(i: u32) -> f32 {
    return e4m3_decode(byte_at(tvf_vc[i >> 2u], i));
}

@compute @workgroup_size(256)
fn tree_verify_attn_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let h = wg.x;
    let qi = wg.y;
    if (h >= tvf_p.n_heads || qi >= tvf_p.k) {
        return;
    }
    let hd = tvf_p.head_dim;
    let nkv = tvf_p.n_kv_heads;
    let nc = tvf_p.n_committed;
    let ring = tvf_p.ring;
    let group = tvf_p.n_heads / nkv;
    let kvh = h / group;
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;

    let vec4ok = (hd & 3u) == 0u;
    let vcn = hd / TVF_WARP;
    let vecv = (hd % 256u) == 0u && vcn <= TVF_MAX_ACC;
    let n4 = hd >> 2u;

    var qpos = 0i;
    var win_start = 0u;
    if (tvf_p.window > 0u) {
        qpos = tvf_pos[qi];
        let s = qpos - i32(tvf_p.window) + 1;
        if (s > 0) {
            win_start = u32(s);
        }
    }

    let qbase = (qi * tvf_p.n_heads + h) * hd;
    for (var d = tid; d < hd; d = d + TVF_BLOCK) {
        tvf_qsh[d] = tvf_q_at(qbase + d);
    }
    workgroupBarrier();

    var acc: array<f32, 16>;
    for (var i = 0u; i < TVF_MAX_ACC; i = i + 1u) {
        acc[i] = 0.0;
    }
    var m = tvf_neg_inf();
    var l = 0.0;

    let n_iter = (nc + TVF_WARPS - 1u) / TVF_WARPS;
    for (var it = 0u; it < n_iter; it = it + 1u) {
        let p = win_start + warp + it * TVF_WARPS;
        let live = p < nc;
        var sp = 0u;
        var partial = 0.0;
        if (live) {
            sp = p;
            if (ring > 0u) {
                sp = p % ring;
            }
            let kbase = (sp * nkv + kvh) * hd;
            if (vec4ok) {
                for (var j4 = lane; j4 < n4; j4 = j4 + TVF_WARP) {
                    let d0 = j4 * 4u;
                    let word = tvf_kc[(kbase + d0) >> 2u];
                    let f0 = e4m3_decode(word);
                    let f1 = e4m3_decode(word >> 8u);
                    let f2 = e4m3_decode(word >> 16u);
                    let f3 = e4m3_decode(word >> 24u);
                    let q0 = tvf_qsh[d0];
                    let q1 = tvf_qsh[d0 + 1u];
                    let q2 = tvf_qsh[d0 + 2u];
                    let q3 = tvf_qsh[d0 + 3u];
                    partial = partial + fma(q3, f3, fma(q2, f2, fma(q0, f0, q1 * f1)));
                }
            } else {
                for (var d = lane; d < hd; d = d + TVF_WARP) {
                    partial = fma(tvf_qsh[d], tvf_kc_at(kbase + d), partial);
                }
            }
        }
        let reduced = tvf_warp_sum(tid, partial);
        if (live) {
            let score = reduced * tvf_ks[sp * nkv + kvh];
            let m_new = max(m, score);
            let corr = tvf_exp(m - m_new);
            let w = tvf_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (sp * nkv + kvh) * hd;
            let w_v = w * tvf_vs[sp * nkv + kvh];
            if (vecv) {
                for (var i = 0u; i < vcn; i = i + 1u) {
                    let d = lane * vcn + i;
                    acc[i] = fma(w_v, tvf_vc_at(vbase + d), acc[i] * corr);
                }
            } else {
                for (var i = 0u; i < TVF_MAX_ACC; i = i + 1u) {
                    let d = lane + i * TVF_WARP;
                    if (d < hd) {
                        acc[i] = fma(acc[i], corr, w_v * tvf_vc_at(vbase + d));
                    }
                }
            }
            m = m_new;
        }
    }

    let n_tree = (tvf_p.k + TVF_WARPS - 1u) / TVF_WARPS;
    for (var it = 0u; it < n_tree; it = it + 1u) {
        let j = warp + it * TVF_WARPS;
        var live = j < tvf_p.k;
        if (live) {
            let mi = qi * tvf_p.k + j;
            if (byte_at(tvf_mask[mi >> 2u], mi) == 0u) {
                live = false;
            }
        }
        if (live && tvf_p.window > 0u) {
            if (qpos - tvf_pos[j] >= i32(tvf_p.window)) {
                live = false;
            }
        }
        var sp = 0u;
        var partial = 0.0;
        if (live) {
            let p = nc + j;
            sp = p;
            if (ring > 0u) {
                sp = p % ring;
            }
            let kbase = (sp * nkv + kvh) * hd;
            for (var d = lane; d < hd; d = d + TVF_WARP) {
                partial = fma(tvf_qsh[d], tvf_kc_at(kbase + d), partial);
            }
        }
        let reduced = tvf_warp_sum(tid, partial);
        if (live) {
            let score = reduced * tvf_ks[sp * nkv + kvh];
            let m_new = max(m, score);
            let corr = tvf_exp(m - m_new);
            let w = tvf_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (sp * nkv + kvh) * hd;
            let w_v = w * tvf_vs[sp * nkv + kvh];
            if (vecv) {
                for (var i = 0u; i < vcn; i = i + 1u) {
                    let d = lane * vcn + i;
                    acc[i] = fma(w_v, tvf_vc_at(vbase + d), acc[i] * corr);
                }
            } else {
                for (var i = 0u; i < TVF_MAX_ACC; i = i + 1u) {
                    let d = lane + i * TVF_WARP;
                    if (d < hd) {
                        acc[i] = fma(acc[i], corr, w_v * tvf_vc_at(vbase + d));
                    }
                }
            }
            m = m_new;
        }
    }

    if (lane == 0u) {
        tvf_sm[warp] = m;
        tvf_sl[warp] = l;
    }
    if (vecv) {
        for (var i = 0u; i < vcn; i = i + 1u) {
            tvf_sacc[warp * TVF_MAX_HD + lane * vcn + i] = acc[i];
        }
    } else {
        for (var i = 0u; i < TVF_MAX_ACC; i = i + 1u) {
            let d = lane + i * TVF_WARP;
            if (d < hd) {
                tvf_sacc[warp * TVF_MAX_HD + d] = acc[i];
            }
        }
    }
    workgroupBarrier();

    if (warp == 0u) {
        var m_glob = tvf_neg_inf();
        for (var w = 0u; w < TVF_WARPS; w = w + 1u) {
            m_glob = max(m_glob, tvf_sm[w]);
        }
        var l_glob = 0.0;
        for (var w = 0u; w < TVF_WARPS; w = w + 1u) {
            var t = 0.0;
            if (tvf_sm[w] > tvf_neg_inf()) {
                t = fma(tvf_sl[w], tvf_exp(tvf_sm[w] - m_glob), -0.0);
            }
            l_glob = l_glob + t;
        }
        var inv = 0.0;
        if (l_glob > 0.0) {
            inv = tvf_recip(l_glob);
        }
        let obase = (qi * tvf_p.n_heads + h) * hd;
        let words = hd >> 1u;
        for (var ww = lane; ww < words; ww = ww + TVF_WARP) {
            let d0 = ww * 2u;
            var a0 = 0.0;
            var a1 = 0.0;
            for (var w = 0u; w < TVF_WARPS; w = w + 1u) {
                if (tvf_sm[w] > tvf_neg_inf()) {
                    let e = tvf_exp(tvf_sm[w] - m_glob);
                    a0 = a0 + fma(tvf_sacc[w * TVF_MAX_HD + d0], e, -0.0);
                    a1 = a1 + fma(tvf_sacc[w * TVF_MAX_HD + d0 + 1u], e, -0.0);
                }
            }
            tvf_out[(obase >> 1u) + ww] = bf16_pack(a0 * inv, a1 * inv);
        }
    }
}

fn af_src_at(base: u32, d: u32, is_v: bool) -> f32 {
    let i = base + d;
    var w = 0u;
    if (is_v) {
        w = af_vsrc[i >> 1u];
    } else {
        w = af_ksrc[i >> 1u];
    }
    return select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
}

@compute @workgroup_size(256)
fn kv_append_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let kvh = wg.x;
    let token = wg.y;
    if (kvh >= af_p.n_kv_heads || token >= af_p.k) {
        return;
    }
    let nkv = af_p.n_kv_heads;
    let hd = af_p.head_dim;
    var slot = af_p.n_committed + token;
    if (af_p.ring > 0u) {
        if (token + af_p.ring < af_p.k) {
            return;
        }
        slot = slot % af_p.ring;
    }
    let base_src = (token * nkv + kvh) * hd;
    let base_dst = (slot * nkv + kvh) * hd;
    let tid = lid.x;

    var lk = 0.0;
    var lv = 0.0;
    for (var d = tid; d < hd; d = d + TVF_BLOCK) {
        lk = max(lk, abs(af_src_at(base_src, d, false)));
        lv = max(lv, abs(af_src_at(base_src, d, true)));
    }
    tvf_mk[tid] = lk;
    tvf_mv[tid] = lv;
    workgroupBarrier();
    for (var s = TVF_BLOCK / 2u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            tvf_mk[tid] = max(tvf_mk[tid], tvf_mk[tid + s]);
            tvf_mv[tid] = max(tvf_mv[tid], tvf_mv[tid + s]);
        }
        workgroupBarrier();
    }
    let amax_k = tvf_mk[0];
    let amax_v = tvf_mv[0];
    let inv_k = select(1.0, kv_div_rn(TVF_FP8_MAX, amax_k), amax_k > 0.0);
    let inv_v = select(1.0, kv_div_rn(TVF_FP8_MAX, amax_v), amax_v > 0.0);
    if (tid == 0u) {
        af_ksc[slot * nkv + kvh] = select(1.0, kv_div_rn(amax_k, TVF_FP8_MAX), amax_k > 0.0);
        af_vsc[slot * nkv + kvh] = select(1.0, kv_div_rn(amax_v, TVF_FP8_MAX), amax_v > 0.0);
    }

    let out_words = hd >> 2u;
    for (var w = tid; w < out_words; w = w + TVF_BLOCK) {
        let d0 = w * 4u;
        var pk = 0u;
        var pv = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            pk = pk | (kv_encode_e4m3(af_src_at(base_src, d0 + j, false) * inv_k) << (8u * j));
            pv = pv | (kv_encode_e4m3(af_src_at(base_src, d0 + j, true) * inv_v) << (8u * j));
        }
        af_kc[(base_dst >> 2u) + w] = pk;
        af_vc[(base_dst >> 2u) + w] = pv;
    }
}

@compute @workgroup_size(256)
fn kv_gather_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = (wg.y * ng.x + wg.x) * TVF_BLOCK + lid.x;
    if (idx >= cf_p.total) {
        return;
    }
    let row = cf_p.stride_words;
    let i = idx / row;
    let e = idx - i * row;
    var srow = cf_p.base + u32(max(cf_path[i], 0));
    if (cf_p.ring > 0u) {
        srow = srow % cf_p.ring;
    }
    let src = srow * row + e;
    cf_sk[idx] = cf_kc[src];
    cf_sv[idx] = cf_vc[src];
}

@compute @workgroup_size(256)
fn kv_scatter_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = (wg.y * ng.x + wg.x) * TVF_BLOCK + lid.x;
    if (idx >= cf_p.total) {
        return;
    }
    let row = cf_p.stride_words;
    let i = idx / row;
    let e = idx - i * row;
    var drow = cf_p.base + i;
    if (cf_p.ring > 0u) {
        if (i + cf_p.ring < cf_p.n_accept) {
            return;
        }
        drow = drow % cf_p.ring;
    }
    let dst = drow * row + e;
    cf_kc[dst] = cf_sk[idx];
    cf_vc[dst] = cf_sv[idx];
}

@compute @workgroup_size(256)
fn kv_gather_scales_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = (wg.y * ng.x + wg.x) * TVF_BLOCK + lid.x;
    if (idx >= cf_p.n_accept * cf_p.n_kv_heads) {
        return;
    }
    let nkv = cf_p.n_kv_heads;
    let i = idx / nkv;
    let e = idx - i * nkv;
    var srow = cf_p.base + u32(max(cf_path[i], 0));
    if (cf_p.ring > 0u) {
        srow = srow % cf_p.ring;
    }
    cf_ssk[idx] = cf_ksc[srow * nkv + e];
    cf_ssv[idx] = cf_vsc[srow * nkv + e];
}

@compute @workgroup_size(256)
fn kv_scatter_scales_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) ng: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let idx = (wg.y * ng.x + wg.x) * TVF_BLOCK + lid.x;
    if (idx >= cf_p.n_accept * cf_p.n_kv_heads) {
        return;
    }
    let nkv = cf_p.n_kv_heads;
    let i = idx / nkv;
    let e = idx - i * nkv;
    var drow = cf_p.base + i;
    if (cf_p.ring > 0u) {
        if (i + cf_p.ring < cf_p.n_accept) {
            return;
        }
        drow = drow % cf_p.ring;
    }
    cf_ksc[drow * nkv + e] = cf_ssk[idx];
    cf_vsc[drow * nkv + e] = cf_ssv[idx];
}
