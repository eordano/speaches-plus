struct Sm8Params {
    n_q: u32,
    n_kv: u32,
    head_dim: u32,
    total: u32,
    m_rows: u32,
    window: u32,
    score_stride: u32,
    scaling: f32,
};

@group(0) @binding(0) var<storage, read> sm8_q: array<u32>;
@group(0) @binding(1) var<storage, read> sm8_k: array<u32>;
@group(0) @binding(2) var<storage, read> sm8_v: array<u32>;
@group(0) @binding(3) var<storage, read> sm8_kscale: array<f32>;
@group(0) @binding(4) var<storage, read> sm8_vscale: array<f32>;
@group(0) @binding(5) var<storage, read_write> sm8_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> sm8_scores: array<f32>;
@group(0) @binding(7) var<uniform> sm8_params: Sm8Params;

const SM8_MAX_M: u32 = 10u;

var<workgroup> sm8_qsh: array<f32, 5120>;
var<workgroup> sm8_red: array<f32, 512>;
var<workgroup> sm8_warp: array<f32, 32>;

fn sm8_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn sm8_expf(x: f32) -> f32 {
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

fn sm8_recip(x: f32) -> f32 {
    let r = 1.0 / x;
    return fma(fma(-x, r, 1.0), r, r);
}

fn sm8_fp8_k(idx: u32) -> f32 {
    return e4m3_decode(byte_at(sm8_k[idx >> 2u], idx));
}

fn sm8_fp8_v(idx: u32) -> f32 {
    return e4m3_decode(byte_at(sm8_v[idx >> 2u], idx));
}

fn sm8_reduce_sum(tid: u32, val: f32) -> f32 {
    let lane = tid & 31u;
    sm8_red[tid] = val;
    if (tid < 32u) {
        sm8_warp[tid] = 0.0;
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            sm8_red[tid] = sm8_red[tid] + sm8_red[tid + off];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        sm8_warp[tid >> 5u] = sm8_red[tid];
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (tid < off) {
            sm8_warp[tid] = sm8_warp[tid] + sm8_warp[tid + off];
        }
        workgroupBarrier();
    }
    let total = sm8_warp[0];
    workgroupBarrier();
    return total;
}

fn sm8_reduce_max(tid: u32, val: f32) -> f32 {
    let lane = tid & 31u;
    sm8_red[tid] = val;
    if (tid < 32u) {
        sm8_warp[tid] = sm8_neg_inf();
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            sm8_red[tid] = max(sm8_red[tid], sm8_red[tid + off]);
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        sm8_warp[tid >> 5u] = sm8_red[tid];
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (tid < off) {
            sm8_warp[tid] = max(sm8_warp[tid], sm8_warp[tid + off]);
        }
        workgroupBarrier();
    }
    let total = sm8_warp[0];
    workgroupBarrier();
    return total;
}

fn sm8_body(tid: u32, head: u32) {
    let hd = sm8_params.head_dim;
    let nq = sm8_params.n_q;
    let nkv = sm8_params.n_kv;
    let group = nq / nkv;
    let kvh = head / group;
    let total = sm8_params.total;
    let mr = sm8_params.m_rows;
    let sw = sm8_params.window;
    let stride = sm8_params.score_stride;

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        sm8_qsh[qi * hd + tid] = bf16_decode(sm8_q[(qi * nq + head) * hd + tid]);
    }
    workgroupBarrier();

    for (var i = 0u; i < total; i = i + 1u) {
        let slot = i * nkv + kvh;
        let kd = sm8_fp8_k(slot * hd + tid);
        let ks = sm8_kscale[slot];
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let row_total = total - (mr - 1u - qi);
            if (i < row_total) {
                let masked = sw > 0u && (row_total - 1u - i) >= sw;
                var partial = 0.0;
                if (!masked) {
                    partial = (sm8_qsh[qi * hd + tid] * kd) * ks;
                }
                let sum = sm8_reduce_sum(tid, partial);
                if (tid == 0u) {
                    sm8_scores[(head * mr + qi) * stride + i] =
                        select(sum * sm8_params.scaling, sm8_neg_inf(), masked);
                }
                storageBarrier();
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        let row_total = total - (mr - 1u - qi);
        let sbase = (head * mr + qi) * stride;

        var thread_max = sm8_neg_inf();
        for (var i = tid; i < row_total; i = i + hd) {
            thread_max = max(thread_max, sm8_scores[sbase + i]);
        }
        let max_score = sm8_reduce_max(tid, thread_max);

        var thread_sum = 0.0;
        for (var i = tid; i < row_total; i = i + hd) {
            let e = sm8_expf(sm8_scores[sbase + i] - max_score);
            sm8_scores[sbase + i] = e;
            thread_sum = thread_sum + e;
        }
        storageBarrier();
        let row_sum = sm8_reduce_sum(tid, thread_sum);

        var inv_total = 0.0;
        if (row_sum > 0.0) {
            inv_total = sm8_recip(row_sum);
        }
        for (var i = tid; i < row_total; i = i + hd) {
            sm8_scores[sbase + i] = inv_total * sm8_scores[sbase + i];
        }
        storageBarrier();
    }

    var acc: array<f32, SM8_MAX_M>;
    for (var qi = 0u; qi < SM8_MAX_M; qi = qi + 1u) {
        acc[qi] = 0.0;
    }
    for (var i = 0u; i < total; i = i + 1u) {
        let slot = i * nkv + kvh;
        let vs = sm8_vscale[slot];
        let vd = sm8_fp8_v(slot * hd + tid);
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let row_total = total - (mr - 1u - qi);
            if (i < row_total) {
                let s = sm8_scores[(head * mr + qi) * stride + i];
                if (s != 0.0) {
                    acc[qi] = fma(vs, s * vd, acc[qi]);
                }
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        sm8_out[(qi * nq + head) * hd + tid] = bf16_encode(acc[qi]);
    }
}

@compute @workgroup_size(64)
fn attn_decode_small_m_fp8_hd64(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < sm8_params.n_q) {
        sm8_body(tid.x, head);
    }
}

@compute @workgroup_size(128)
fn attn_decode_small_m_fp8_hd128(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < sm8_params.n_q) {
        sm8_body(tid.x, head);
    }
}

@compute @workgroup_size(256)
fn attn_decode_small_m_fp8_hd256(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < sm8_params.n_q) {
        sm8_body(tid.x, head);
    }
}

@compute @workgroup_size(512)
fn attn_decode_small_m_fp8_hd512(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < sm8_params.n_q) {
        sm8_body(tid.x, head);
    }
}
