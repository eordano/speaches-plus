struct AttnFp8Params {
    n_q: u32,
    n_kv: u32,
    head_dim: u32,
    n_total: u32,
    sliding_window: u32,
    score_stride: u32,
    scaling: f32,
    reserved: u32,
};

@group(0) @binding(0) var<storage, read> afd_q: array<u32>;
@group(0) @binding(1) var<storage, read> afd_k: array<u32>;
@group(0) @binding(2) var<storage, read> afd_v: array<u32>;
@group(0) @binding(3) var<storage, read> afd_kscale: array<f32>;
@group(0) @binding(4) var<storage, read> afd_vscale: array<f32>;
@group(0) @binding(5) var<storage, read_write> afd_out: array<u32>;
@group(0) @binding(6) var<storage, read_write> afd_scores: array<f32>;
@group(0) @binding(7) var<uniform> afd_params: AttnFp8Params;

var<workgroup> afd_qsh: array<f32, 512>;
var<workgroup> afd_red: array<f32, 512>;
var<workgroup> afd_warp: array<f32, 32>;

fn afd_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn afd_expf(x: f32) -> f32 {
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

fn afd_recip(x: f32) -> f32 {
    let r = 1.0 / x;
    return fma(fma(-x, r, 1.0), r, r);
}

fn afd_fp8_k(idx: u32) -> f32 {
    return e4m3_decode(byte_at(afd_k[idx >> 2u], idx));
}

fn afd_fp8_v(idx: u32) -> f32 {
    return e4m3_decode(byte_at(afd_v[idx >> 2u], idx));
}

fn afd_reduce_sum(tid: u32, val: f32) -> f32 {
    let lane = tid & 31u;
    afd_red[tid] = val;
    if (tid < 32u) {
        afd_warp[tid] = 0.0;
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            afd_red[tid] = afd_red[tid] + afd_red[tid + off];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        afd_warp[tid >> 5u] = afd_red[tid];
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (tid < off) {
            afd_warp[tid] = afd_warp[tid] + afd_warp[tid + off];
        }
        workgroupBarrier();
    }
    let total = afd_warp[0];
    workgroupBarrier();
    return total;
}

fn afd_reduce_max(tid: u32, val: f32) -> f32 {
    let lane = tid & 31u;
    afd_red[tid] = val;
    if (tid < 32u) {
        afd_warp[tid] = afd_neg_inf();
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (lane < off) {
            afd_red[tid] = max(afd_red[tid], afd_red[tid + off]);
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        afd_warp[tid >> 5u] = afd_red[tid];
    }
    workgroupBarrier();
    for (var off = 16u; off > 0u; off = off >> 1u) {
        if (tid < off) {
            afd_warp[tid] = max(afd_warp[tid], afd_warp[tid + off]);
        }
        workgroupBarrier();
    }
    let total = afd_warp[0];
    workgroupBarrier();
    return total;
}

fn afd_body(tid: u32, head: u32) {
    let hd = afd_params.head_dim;
    let nkv = afd_params.n_kv;
    let group = afd_params.n_q / nkv;
    let kvh = head / group;
    let n_total = afd_params.n_total;
    let sw = afd_params.sliding_window;
    let sbase = head * afd_params.score_stride;

    afd_qsh[tid] = bf16_decode(afd_q[head * hd + tid]);
    workgroupBarrier();

    for (var i = 0u; i < n_total; i = i + 1u) {
        let masked = sw > 0u && (n_total - 1u - i) >= sw;
        var partial = 0.0;
        if (!masked) {
            let slot = i * nkv + kvh;
            let kd = afd_fp8_k(slot * hd + tid);
            let ks = afd_kscale[slot];
            partial = (afd_qsh[tid] * kd) * ks;
        }
        let sum = afd_reduce_sum(tid, partial);
        if (tid == 0u) {
            afd_scores[sbase + i] = select(sum * afd_params.scaling, afd_neg_inf(), masked);
        }
        storageBarrier();
    }

    var thread_max = afd_neg_inf();
    for (var i = tid; i < n_total; i = i + hd) {
        thread_max = max(thread_max, afd_scores[sbase + i]);
    }
    let max_score = afd_reduce_max(tid, thread_max);

    var thread_sum = 0.0;
    for (var i = tid; i < n_total; i = i + hd) {
        let e = afd_expf(afd_scores[sbase + i] - max_score);
        afd_scores[sbase + i] = e;
        thread_sum = thread_sum + e;
    }
    storageBarrier();
    let total = afd_reduce_sum(tid, thread_sum);

    var inv_total = 0.0;
    if (total > 0.0) {
        inv_total = afd_recip(total);
    }
    for (var i = tid; i < n_total; i = i + hd) {
        afd_scores[sbase + i] = inv_total * afd_scores[sbase + i];
    }
    storageBarrier();

    var acc = 0.0;
    for (var i = 0u; i < n_total; i = i + 1u) {
        let s = afd_scores[sbase + i];
        if (s == 0.0) {
            continue;
        }
        let slot = i * nkv + kvh;
        let vs = afd_vscale[slot];
        let vd = afd_fp8_v(slot * hd + tid);
        acc = fma(vs, s * vd, acc);
    }
    afd_out[head * hd + tid] = bf16_encode(acc);
}

@compute @workgroup_size(64)
fn attention_fp8_decode_hd64(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < afd_params.n_q) {
        afd_body(tid.x, head);
    }
}

@compute @workgroup_size(128)
fn attention_fp8_decode_hd128(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < afd_params.n_q) {
        afd_body(tid.x, head);
    }
}

@compute @workgroup_size(256)
fn attention_fp8_decode_hd256(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < afd_params.n_q) {
        afd_body(tid.x, head);
    }
}

@compute @workgroup_size(512)
fn attention_fp8_decode_hd512(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let head = wg.x + wg.y * nwg.x;
    if (head < afd_params.n_q) {
        afd_body(tid.x, head);
    }
}
