struct SmParams {
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    total: u32,
    m_rows: u32,
    window: u32,
    scaling: f32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> sm_q: array<f32>;
@group(0) @binding(1) var<storage, read> sm_k_f32: array<f32>;
@group(0) @binding(2) var<storage, read> sm_v_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> sm_out: array<f32>;
@group(0) @binding(4) var<uniform> sm_params: SmParams;
@group(0) @binding(5) var<storage, read> sm_k_words: array<u32>;
@group(0) @binding(6) var<storage, read> sm_v_words: array<u32>;
@group(0) @binding(7) var<storage, read> smf_q: array<u32>;
@group(0) @binding(8) var<storage, read> smf_k: array<u32>;
@group(0) @binding(9) var<storage, read> smf_v: array<u32>;
@group(0) @binding(10) var<storage, read> smf_kscale: array<f32>;
@group(0) @binding(11) var<storage, read> smf_vscale: array<f32>;
@group(0) @binding(12) var<storage, read_write> smf_out: array<u32>;
@group(0) @binding(13) var<storage, read_write> smf_scores: array<f32>;

const SM_BLOCK: u32 = 128u;
const SM_MAX_PER_THREAD: u32 = 4u;
const SM_MAX_HD: u32 = 512u;
const SM_MAX_M: u32 = 9u;
const SM_LOG2_E: f32 = 1.4426950408889634;

var<workgroup> sm_qsh: array<f32, 4608>;
var<workgroup> sm_red: array<f32, 128>;
var<workgroup> smf_red: array<f32, 512>;
var<workgroup> smf_warp: array<f32, 32>;

fn sm_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn sm_fast_exp(x: f32) -> f32 {
    return exp2(x * SM_LOG2_E);
}

fn sm_recip(x: f32) -> f32 {
    let r = 1.0 / x;
    return fma(fma(-x, r, 1.0), r, r);
}

fn sm_k_bf16(idx: u32) -> f32 {
    let word = sm_k_words[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn sm_v_bf16(idx: u32) -> f32 {
    let word = sm_v_words[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn sm_row_bounds(total: i32, mr: u32, win: i32, qi: u32) -> vec2<i32> {
    let tq = total - i32(mr - 1u - qi);
    var sq = 0;
    if (win > 0 && tq > win) {
        sq = tq - win;
    }
    return vec2<i32>(sq, tq);
}

fn sm_sweep_start(total: i32, mr: u32, win: i32) -> i32 {
    let b = sm_row_bounds(total, mr, win, 0u);
    return b.x;
}

@compute @workgroup_size(128)
fn attn_decode_small_m_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x + wg.y * nwg.x;
    if (h >= sm_params.n_heads) {
        return;
    }
    let hd = sm_params.head_dim;
    let nkv = sm_params.n_kv_heads;
    let group = sm_params.n_heads / nkv;
    let kvh = h / group;
    let mr = sm_params.m_rows;
    let lid = tid.x;

    for (var t = lid; t < mr * hd; t = t + SM_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        sm_qsh[t] = sm_q[(qi * sm_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    var acc: array<array<f32, SM_MAX_PER_THREAD>, SM_MAX_M>;
    var mreg: array<f32, SM_MAX_M>;
    var lreg: array<f32, SM_MAX_M>;
    for (var qi = 0u; qi < SM_MAX_M; qi = qi + 1u) {
        acc[qi] = array<f32, SM_MAX_PER_THREAD>(0.0, 0.0, 0.0, 0.0);
        mreg[qi] = sm_neg_inf();
        lreg[qi] = 0.0;
    }

    let total = i32(sm_params.total);
    let win = i32(sm_params.window);
    let p_start = sm_sweep_start(total, mr, win);

    for (var p = p_start; p < total; p = p + 1) {
        let kbase = (u32(p) * nkv + kvh) * hd;
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let bounds = sm_row_bounds(total, mr, win, qi);
            if (p >= bounds.x && p < bounds.y) {
                let qoff = qi * hd;
                var partial = 0.0;
                for (var d = lid; d < hd; d = d + SM_BLOCK) {
                    partial = fma(sm_qsh[qoff + d], sm_k_f32[kbase + d], partial);
                }
                sm_red[lid] = partial;
                workgroupBarrier();
                for (var s = SM_BLOCK / 2u; s > 0u; s = s >> 1u) {
                    if (lid < s) {
                        sm_red[lid] = sm_red[lid] + sm_red[lid + s];
                    }
                    workgroupBarrier();
                }
                let score = sm_red[0] * sm_params.scaling;
                workgroupBarrier();

                let m_new = max(mreg[qi], score);
                let corr = sm_fast_exp(mreg[qi] - m_new);
                let w = sm_fast_exp(score - m_new);
                lreg[qi] = fma(lreg[qi], corr, w);
                for (var i = 0u; i < SM_MAX_PER_THREAD; i = i + 1u) {
                    let d = lid + i * SM_BLOCK;
                    if (d < hd) {
                        acc[qi][i] = fma(acc[qi][i], corr, w * sm_v_f32[kbase + d]);
                    }
                }
                mreg[qi] = m_new;
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        var inv_l = 0.0;
        if (lreg[qi] > 0.0) {
            inv_l = sm_recip(lreg[qi]);
        }
        for (var i = 0u; i < SM_MAX_PER_THREAD; i = i + 1u) {
            let d = lid + i * SM_BLOCK;
            if (d < hd) {
                sm_out[(qi * sm_params.n_heads + h) * hd + d] = acc[qi][i] * inv_l;
            }
        }
    }
}

@compute @workgroup_size(128)
fn attn_decode_small_m_bf16kv(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x + wg.y * nwg.x;
    if (h >= sm_params.n_heads) {
        return;
    }
    let hd = sm_params.head_dim;
    let nkv = sm_params.n_kv_heads;
    let group = sm_params.n_heads / nkv;
    let kvh = h / group;
    let mr = sm_params.m_rows;
    let lid = tid.x;

    for (var t = lid; t < mr * hd; t = t + SM_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        sm_qsh[t] = sm_q[(qi * sm_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    var acc: array<array<f32, SM_MAX_PER_THREAD>, SM_MAX_M>;
    var mreg: array<f32, SM_MAX_M>;
    var lreg: array<f32, SM_MAX_M>;
    for (var qi = 0u; qi < SM_MAX_M; qi = qi + 1u) {
        acc[qi] = array<f32, SM_MAX_PER_THREAD>(0.0, 0.0, 0.0, 0.0);
        mreg[qi] = sm_neg_inf();
        lreg[qi] = 0.0;
    }

    let total = i32(sm_params.total);
    let win = i32(sm_params.window);
    let p_start = sm_sweep_start(total, mr, win);

    for (var p = p_start; p < total; p = p + 1) {
        let kbase = (u32(p) * nkv + kvh) * hd;
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let bounds = sm_row_bounds(total, mr, win, qi);
            if (p >= bounds.x && p < bounds.y) {
                let qoff = qi * hd;
                var partial = 0.0;
                for (var d = lid; d < hd; d = d + SM_BLOCK) {
                    partial = fma(sm_qsh[qoff + d], sm_k_bf16(kbase + d), partial);
                }
                sm_red[lid] = partial;
                workgroupBarrier();
                for (var s = SM_BLOCK / 2u; s > 0u; s = s >> 1u) {
                    if (lid < s) {
                        sm_red[lid] = sm_red[lid] + sm_red[lid + s];
                    }
                    workgroupBarrier();
                }
                let score = sm_red[0] * sm_params.scaling;
                workgroupBarrier();

                let m_new = max(mreg[qi], score);
                let corr = sm_fast_exp(mreg[qi] - m_new);
                let w = sm_fast_exp(score - m_new);
                lreg[qi] = fma(lreg[qi], corr, w);
                for (var i = 0u; i < SM_MAX_PER_THREAD; i = i + 1u) {
                    let d = lid + i * SM_BLOCK;
                    if (d < hd) {
                        acc[qi][i] = fma(acc[qi][i], corr, w * sm_v_bf16(kbase + d));
                    }
                }
                mreg[qi] = m_new;
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        var inv_l = 0.0;
        if (lreg[qi] > 0.0) {
            inv_l = sm_recip(lreg[qi]);
        }
        for (var i = 0u; i < SM_MAX_PER_THREAD; i = i + 1u) {
            let d = lid + i * SM_BLOCK;
            if (d < hd) {
                sm_out[(qi * sm_params.n_heads + h) * hd + d] = acc[qi][i] * inv_l;
            }
        }
    }
}

fn smf_expf(x: f32) -> f32 {
    let c = bitcast<f32>(0x3bbb989du);
    let hi = bitcast<f32>(0x3fb8aa3bu);
    let lo = bitcast<f32>(0x32a57060u);
    let t = clamp(fma(x, c, 0.5), 0.0, 1.0);
    let p = t * 252.0;
    let e = fma(t, 252.0, -p);
    var f = floor(p);
    if (e < 0.0 && p == f) {
        f = f - 1.0;
    }
    let z = f - 126.0;
    let a = fma(x, lo, fma(x, hi, -z));
    let scale = bitcast<f32>((u32(i32(f)) + 1u) << 23u);
    return exp2(a) * scale;
}

fn smf_fp8_k(idx: u32) -> f32 {
    return e4m3_decode(byte_at(smf_k[idx >> 2u], idx));
}

fn smf_fp8_v(idx: u32) -> f32 {
    return e4m3_decode(byte_at(smf_v[idx >> 2u], idx));
}

fn smf_reduce_sum(tid: u32, val: f32) -> f32 {
    let lane = tid & 31u;
    smf_red[tid] = val;
    if (tid < 32u) {
        smf_warp[tid] = 0.0;
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            smf_red[tid] = smf_red[tid] + smf_red[tid + off];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        smf_warp[tid >> 5u] = smf_red[tid];
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (tid < off) {
            smf_warp[tid] = smf_warp[tid] + smf_warp[tid + off];
        }
        workgroupBarrier();
    }
    let total = smf_warp[0];
    workgroupBarrier();
    return total;
}

fn smf_reduce_max(tid: u32, val: f32) -> f32 {
    let lane = tid & 31u;
    smf_red[tid] = val;
    if (tid < 32u) {
        smf_warp[tid] = sm_neg_inf();
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            smf_red[tid] = max(smf_red[tid], smf_red[tid + off]);
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        smf_warp[tid >> 5u] = smf_red[tid];
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (tid < off) {
            smf_warp[tid] = max(smf_warp[tid], smf_warp[tid + off]);
        }
        workgroupBarrier();
    }
    let total = smf_warp[0];
    workgroupBarrier();
    return total;
}

fn smf_row_tq(qi: u32) -> u32 {
    return sm_params.total - (sm_params.m_rows - 1u - qi);
}

fn smf_row_sq(tq: u32) -> u32 {
    let win = sm_params.window;
    var sq = 0u;
    if (win > 0u && tq > win) {
        sq = tq - win;
    }
    return sq;
}

fn smf_body(tid: u32, head: u32) {
    let hd = sm_params.head_dim;
    let nkv = sm_params.n_kv_heads;
    let group = sm_params.n_heads / nkv;
    let kvh = head / group;
    let mr = sm_params.m_rows;
    let total = sm_params.total;

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        sm_qsh[qi * hd + tid] = bf16_decode(smf_q[(qi * sm_params.n_heads + head) * hd + tid]);
    }
    workgroupBarrier();

    let p_start = smf_row_sq(smf_row_tq(0u));

    for (var p = p_start; p < total; p = p + 1u) {
        let slot = p * nkv + kvh;
        let kd = smf_fp8_k(slot * hd + tid);
        let ks = smf_kscale[slot];
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let tq = smf_row_tq(qi);
            let sq = smf_row_sq(tq);
            if (p >= sq && p < tq) {
                let partial = (sm_qsh[qi * hd + tid] * kd) * ks;
                let sum = smf_reduce_sum(tid, partial);
                if (tid == 0u) {
                    smf_scores[(qi * sm_params.n_heads + head) * total + p] = sum * sm_params.scaling;
                }
                storageBarrier();
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        let tq = smf_row_tq(qi);
        let sq = smf_row_sq(tq);
        let sbase = (qi * sm_params.n_heads + head) * total;

        var thread_max = sm_neg_inf();
        for (var i = tid; i < tq; i = i + hd) {
            if (i >= sq) {
                thread_max = max(thread_max, smf_scores[sbase + i]);
            }
        }
        let max_score = smf_reduce_max(tid, thread_max);

        var thread_sum = 0.0;
        for (var i = tid; i < tq; i = i + hd) {
            if (i >= sq) {
                let e = smf_expf(smf_scores[sbase + i] - max_score);
                smf_scores[sbase + i] = e;
                thread_sum = thread_sum + e;
            }
        }
        storageBarrier();
        let l = smf_reduce_sum(tid, thread_sum);

        var inv_l = 0.0;
        if (l > 0.0) {
            inv_l = sm_recip(l);
        }
        for (var i = tid; i < tq; i = i + hd) {
            if (i >= sq) {
                smf_scores[sbase + i] = inv_l * smf_scores[sbase + i];
            }
        }
        storageBarrier();
    }

    var acc: array<f32, SM_MAX_M>;
    for (var qi = 0u; qi < SM_MAX_M; qi = qi + 1u) {
        acc[qi] = 0.0;
    }
    for (var p = p_start; p < total; p = p + 1u) {
        let slot = p * nkv + kvh;
        let vs = smf_vscale[slot];
        let vd = smf_fp8_v(slot * hd + tid);
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let tq = smf_row_tq(qi);
            let sq = smf_row_sq(tq);
            if (p >= sq && p < tq) {
                let s = smf_scores[(qi * sm_params.n_heads + head) * total + p];
                if (s != 0.0) {
                    acc[qi] = fma(vs, s * vd, acc[qi]);
                }
            }
        }
    }
    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        smf_out[(qi * sm_params.n_heads + head) * hd + tid] = bf16_encode(acc[qi]);
    }
}

@compute @workgroup_size(64)
fn attn_decode_small_m_fp8_hd64(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < sm_params.n_heads) {
        smf_body(tid.x, head);
    }
}

@compute @workgroup_size(128)
fn attn_decode_small_m_fp8_hd128(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < sm_params.n_heads) {
        smf_body(tid.x, head);
    }
}

@compute @workgroup_size(256)
fn attn_decode_small_m_fp8_hd256(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < sm_params.n_heads) {
        smf_body(tid.x, head);
    }
}

@compute @workgroup_size(512)
fn attn_decode_small_m_fp8_hd512(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < sm_params.n_heads) {
        smf_body(tid.x, head);
    }
}
