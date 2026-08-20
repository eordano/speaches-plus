
struct Q3cvParams {
    conv_dim: u32,
    kernel: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> cv_x: array<u32>;
@group(0) @binding(1) var<storage, read> cv_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> cv_state: array<f32>;
@group(0) @binding(3) var<storage, read_write> cv_out: array<f32>;
@group(0) @binding(4) var<uniform> cv_p: Q3cvParams;

@compute @workgroup_size(64)
fn q3w_delta_conv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= cv_p.conv_dim) {
        return;
    }
    let ks = cv_p.kernel;
    let hist = ks - 1u;
    let xv = bf16_decode(u16_at(cv_x[c >> 1u], c));
    var acc = 0.0;
    for (var j = 0u; j < hist; j = j + 1u) {
        acc = fma(cv_w[c * ks + j], cv_state[c * hist + j], acc);
    }
    acc = fma(cv_w[c * ks + hist], xv, acc);
    for (var j = 0u; j + 1u < hist; j = j + 1u) {
        cv_state[c * hist + j] = cv_state[c * hist + j + 1u];
    }
    if (hist > 0u) {
        cv_state[c * hist + hist - 1u] = xv;
    }
    let silu = acc / (1.0 + exp(-acc));
    cv_out[c] = bf16_decode(bf16_encode(silu));
}

struct Q3dqParams {
    n_v: u32,
    d_k: u32,
    d_v: u32,
    key_dim: u32,
    v_per_k: u32,
    ab_off: u32,
    pad1: u32,
    scale: f32,
};

@group(0) @binding(10) var<storage, read> dq_mixed: array<f32>;
@group(0) @binding(11) var<storage, read_write> dq_q: array<f32>;
@group(0) @binding(12) var<storage, read_write> dq_k: array<f32>;
@group(0) @binding(13) var<storage, read_write> dq_v: array<f32>;
@group(0) @binding(14) var<uniform> dq_p: Q3dqParams;

var<workgroup> dq_rq: array<f32, 128>;
var<workgroup> dq_rk: array<f32, 128>;

fn q3w_split_lane(h: u32, d: u32) {
    let kh = h / dq_p.v_per_k;
    var qv = 0.0;
    var kv = 0.0;
    if (d < dq_p.d_k) {
        qv = dq_mixed[kh * dq_p.d_k + d];
        kv = dq_mixed[dq_p.key_dim + kh * dq_p.d_k + d];
    }
    dq_rq[d] = qv * qv;
    dq_rk[d] = kv * kv;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (d < s) {
            dq_rq[d] = dq_rq[d] + dq_rq[d + s];
            dq_rk[d] = dq_rk[d] + dq_rk[d + s];
        }
        workgroupBarrier();
    }
    let nq = sqrt(dq_rq[0] + 1e-6);
    let nk = sqrt(dq_rk[0] + 1e-6);
    if (d < dq_p.d_k) {
        dq_q[h * dq_p.d_k + d] = (qv / nq) * dq_p.scale;
        dq_k[h * dq_p.d_k + d] = kv / nk;
    }
    if (d < dq_p.d_v) {
        dq_v[h * dq_p.d_v + d] = dq_mixed[2u * dq_p.key_dim + h * dq_p.d_v + d];
    }
}

@compute @workgroup_size(128)
fn q3w_delta_qkv(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    q3w_split_lane(wid.x, lid.x);
}

@compute @workgroup_size(128)
fn q3w_delta_qkv_gated(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    q3w_split_lane(wid.x, lid.x);
    if (lid.x == 0u) {
        q3w_gate_lane(wid.x, dq_p.n_v, dq_p.ab_off);
    }
}

struct Q3gParams {
    n_v: u32,
    ab_off: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(20) var<storage, read> dg_ab: array<u32>;
@group(0) @binding(21) var<storage, read> dg_alogdt: array<f32>;
@group(0) @binding(23) var<storage, read_write> dg_g: array<f32>;
@group(0) @binding(24) var<storage, read_write> dg_beta: array<f32>;
@group(0) @binding(25) var<uniform> dg_p: Q3gParams;

fn q3w_gate_lane(i: u32, n_v: u32, ab_off: u32) {
    let ii = ab_off + i;
    let a = bf16_decode(u16_at(dg_ab[ii >> 1u], ii));
    let j = ab_off + n_v + i;
    let b = bf16_decode(u16_at(dg_ab[j >> 1u], j));
    dg_beta[i] = 1.0 / (1.0 + exp(-b));
    let t = a + dg_alogdt[n_v + i];
    let sp = max(t, 0.0) + log(1.0 + exp(-abs(t)));
    dg_g[i] = exp(sp * (-exp(dg_alogdt[i])));
}

@compute @workgroup_size(64)
fn q3w_delta_gating(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dg_p.n_v) {
        return;
    }
    q3w_gate_lane(i, dg_p.n_v, dg_p.ab_off);
}

struct Q3rParams {
    heads: u32,
    d_k: u32,
    d_v: u32,
    pad0: u32,
};

@group(0) @binding(30) var<storage, read> dr_q: array<f32>;
@group(0) @binding(31) var<storage, read> dr_k: array<f32>;
@group(0) @binding(32) var<storage, read> dr_v: array<f32>;
@group(0) @binding(33) var<storage, read> dr_g: array<f32>;
@group(0) @binding(34) var<storage, read> dr_beta: array<f32>;
@group(0) @binding(35) var<storage, read_write> dr_out: array<f32>;
@group(0) @binding(36) var<storage, read_write> dr_state: array<f32>;
@group(0) @binding(37) var<uniform> dr_p: Q3rParams;

var<workgroup> dr_kb: array<f32, 128>;
var<workgroup> dr_qb: array<f32, 128>;

@compute @workgroup_size(128)
fn q3w_delta_recurrent(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let lane = lid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    if (lane < dk) {
        dr_kb[lane] = dr_k[h * dk + lane];
        dr_qb[lane] = dr_q[h * dk + lane];
    }
    workgroupBarrier();
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;

    var kv_mem = 0.0;
    for (var i = 0u; i < dk; i = i + 1u) {
        let s = dr_state[sbase + i * dv] * ge;
        dr_state[sbase + i * dv] = s;
        kv_mem = fma(s, dr_kb[i], kv_mem);
    }
    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;
    var outv = 0.0;
    for (var i = 0u; i < dk; i = i + 1u) {
        let s = fma(dr_kb[i], delta, dr_state[sbase + i * dv]);
        dr_state[sbase + i * dv] = s;
        outv = fma(s, dr_qb[i], outv);
    }
    dr_out[h * dv + lane] = outv;
}

@compute @workgroup_size(128)
fn q3w_delta_recurrent_u4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let lane = lid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    if (lane < dk) {
        dr_kb[lane] = dr_k[h * dk + lane];
        dr_qb[lane] = dr_q[h * dk + lane];
    }
    workgroupBarrier();
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;
    let dv2 = dv + dv;
    let dv3 = dv2 + dv;

    var kv_mem = 0.0;
    var i = 0u;
    loop {
        if (i + 4u > dk) {
            break;
        }
        let p = sbase + i * dv;
        let s0 = dr_state[p] * ge;
        let s1 = dr_state[p + dv] * ge;
        let s2 = dr_state[p + dv2] * ge;
        let s3 = dr_state[p + dv3] * ge;
        kv_mem = fma(s0, dr_kb[i], kv_mem);
        kv_mem = fma(s1, dr_kb[i + 1u], kv_mem);
        kv_mem = fma(s2, dr_kb[i + 2u], kv_mem);
        kv_mem = fma(s3, dr_kb[i + 3u], kv_mem);
        i = i + 4u;
    }
    loop {
        if (i >= dk) {
            break;
        }
        kv_mem = fma(dr_state[sbase + i * dv] * ge, dr_kb[i], kv_mem);
        i = i + 1u;
    }

    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;

    var outv = 0.0;
    var j = 0u;
    loop {
        if (j + 4u > dk) {
            break;
        }
        let p = sbase + j * dv;
        let t0 = dr_state[p] * ge;
        let t1 = dr_state[p + dv] * ge;
        let t2 = dr_state[p + dv2] * ge;
        let t3 = dr_state[p + dv3] * ge;
        let u0 = fma(dr_kb[j], delta, t0);
        let u1 = fma(dr_kb[j + 1u], delta, t1);
        let u2 = fma(dr_kb[j + 2u], delta, t2);
        let u3 = fma(dr_kb[j + 3u], delta, t3);
        dr_state[p] = u0;
        dr_state[p + dv] = u1;
        dr_state[p + dv2] = u2;
        dr_state[p + dv3] = u3;
        outv = fma(u0, dr_qb[j], outv);
        outv = fma(u1, dr_qb[j + 1u], outv);
        outv = fma(u2, dr_qb[j + 2u], outv);
        outv = fma(u3, dr_qb[j + 3u], outv);
        j = j + 4u;
    }
    loop {
        if (j >= dk) {
            break;
        }
        let p = sbase + j * dv;
        let u = fma(dr_kb[j], delta, dr_state[p] * ge);
        dr_state[p] = u;
        outv = fma(u, dr_qb[j], outv);
        j = j + 1u;
    }
    dr_out[h * dv + lane] = outv;
}

@compute @workgroup_size(32)
fn q3w_delta_recurrent_l32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    let lane = wid.y * 32u + lid.x;
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;
    let kbase = h * dk;

    var kv_mem = 0.0;
    for (var i = 0u; i < dk; i = i + 1u) {
        let s = dr_state[sbase + i * dv] * ge;
        dr_state[sbase + i * dv] = s;
        kv_mem = fma(s, dr_k[kbase + i], kv_mem);
    }
    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;
    var outv = 0.0;
    for (var i = 0u; i < dk; i = i + 1u) {
        let s = fma(dr_k[kbase + i], delta, dr_state[sbase + i * dv]);
        dr_state[sbase + i * dv] = s;
        outv = fma(s, dr_q[kbase + i], outv);
    }
    dr_out[h * dv + lane] = outv;
}

@compute @workgroup_size(32)
fn q3w_delta_recurrent_u4l32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    let lane = wid.y * 32u + lid.x;
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;
    let kbase = h * dk;
    let dv2 = dv + dv;
    let dv3 = dv2 + dv;

    var kv_mem = 0.0;
    var i = 0u;
    loop {
        if (i + 4u > dk) {
            break;
        }
        let p = sbase + i * dv;
        let s0 = dr_state[p] * ge;
        let s1 = dr_state[p + dv] * ge;
        let s2 = dr_state[p + dv2] * ge;
        let s3 = dr_state[p + dv3] * ge;
        kv_mem = fma(s0, dr_k[kbase + i], kv_mem);
        kv_mem = fma(s1, dr_k[kbase + i + 1u], kv_mem);
        kv_mem = fma(s2, dr_k[kbase + i + 2u], kv_mem);
        kv_mem = fma(s3, dr_k[kbase + i + 3u], kv_mem);
        i = i + 4u;
    }
    loop {
        if (i >= dk) {
            break;
        }
        kv_mem = fma(dr_state[sbase + i * dv] * ge, dr_k[kbase + i], kv_mem);
        i = i + 1u;
    }

    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;

    var outv = 0.0;
    var j = 0u;
    loop {
        if (j + 4u > dk) {
            break;
        }
        let p = sbase + j * dv;
        let t0 = dr_state[p] * ge;
        let t1 = dr_state[p + dv] * ge;
        let t2 = dr_state[p + dv2] * ge;
        let t3 = dr_state[p + dv3] * ge;
        let u0 = fma(dr_k[kbase + j], delta, t0);
        let u1 = fma(dr_k[kbase + j + 1u], delta, t1);
        let u2 = fma(dr_k[kbase + j + 2u], delta, t2);
        let u3 = fma(dr_k[kbase + j + 3u], delta, t3);
        dr_state[p] = u0;
        dr_state[p + dv] = u1;
        dr_state[p + dv2] = u2;
        dr_state[p + dv3] = u3;
        outv = fma(u0, dr_q[kbase + j], outv);
        outv = fma(u1, dr_q[kbase + j + 1u], outv);
        outv = fma(u2, dr_q[kbase + j + 2u], outv);
        outv = fma(u3, dr_q[kbase + j + 3u], outv);
        j = j + 4u;
    }
    loop {
        if (j >= dk) {
            break;
        }
        let p = sbase + j * dv;
        let u = fma(dr_k[kbase + j], delta, dr_state[p] * ge);
        dr_state[p] = u;
        outv = fma(u, dr_q[kbase + j], outv);
        j = j + 1u;
    }
    dr_out[h * dv + lane] = outv;
}

@compute @workgroup_size(128)
fn q3w_delta_recurrent_u8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    let lane = wid.y * 128u + lid.x;
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;
    let kbase = h * dk;
    let dv2 = dv * 2u;
    let dv3 = dv * 3u;
    let dv4 = dv * 4u;
    let dv5 = dv * 5u;
    let dv6 = dv * 6u;
    let dv7 = dv * 7u;

    var kv_mem = 0.0;
    var i = 0u;
    loop {
        if (i + 8u > dk) {
            break;
        }
        let p = sbase + i * dv;
        let s0 = dr_state[p] * ge;
        let s1 = dr_state[p + dv] * ge;
        let s2 = dr_state[p + dv2] * ge;
        let s3 = dr_state[p + dv3] * ge;
        let s4 = dr_state[p + dv4] * ge;
        let s5 = dr_state[p + dv5] * ge;
        let s6 = dr_state[p + dv6] * ge;
        let s7 = dr_state[p + dv7] * ge;
        kv_mem = fma(s0, dr_k[kbase + i], kv_mem);
        kv_mem = fma(s1, dr_k[kbase + i + 1u], kv_mem);
        kv_mem = fma(s2, dr_k[kbase + i + 2u], kv_mem);
        kv_mem = fma(s3, dr_k[kbase + i + 3u], kv_mem);
        kv_mem = fma(s4, dr_k[kbase + i + 4u], kv_mem);
        kv_mem = fma(s5, dr_k[kbase + i + 5u], kv_mem);
        kv_mem = fma(s6, dr_k[kbase + i + 6u], kv_mem);
        kv_mem = fma(s7, dr_k[kbase + i + 7u], kv_mem);
        i = i + 8u;
    }
    loop {
        if (i >= dk) {
            break;
        }
        kv_mem = fma(dr_state[sbase + i * dv] * ge, dr_k[kbase + i], kv_mem);
        i = i + 1u;
    }

    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;

    var outv = 0.0;
    var j = 0u;
    loop {
        if (j + 8u > dk) {
            break;
        }
        let p = sbase + j * dv;
        let t0 = dr_state[p] * ge;
        let t1 = dr_state[p + dv] * ge;
        let t2 = dr_state[p + dv2] * ge;
        let t3 = dr_state[p + dv3] * ge;
        let t4 = dr_state[p + dv4] * ge;
        let t5 = dr_state[p + dv5] * ge;
        let t6 = dr_state[p + dv6] * ge;
        let t7 = dr_state[p + dv7] * ge;
        let u0 = fma(dr_k[kbase + j], delta, t0);
        let u1 = fma(dr_k[kbase + j + 1u], delta, t1);
        let u2 = fma(dr_k[kbase + j + 2u], delta, t2);
        let u3 = fma(dr_k[kbase + j + 3u], delta, t3);
        let u4 = fma(dr_k[kbase + j + 4u], delta, t4);
        let u5 = fma(dr_k[kbase + j + 5u], delta, t5);
        let u6 = fma(dr_k[kbase + j + 6u], delta, t6);
        let u7 = fma(dr_k[kbase + j + 7u], delta, t7);
        dr_state[p] = u0;
        dr_state[p + dv] = u1;
        dr_state[p + dv2] = u2;
        dr_state[p + dv3] = u3;
        dr_state[p + dv4] = u4;
        dr_state[p + dv5] = u5;
        dr_state[p + dv6] = u6;
        dr_state[p + dv7] = u7;
        outv = fma(u0, dr_q[kbase + j], outv);
        outv = fma(u1, dr_q[kbase + j + 1u], outv);
        outv = fma(u2, dr_q[kbase + j + 2u], outv);
        outv = fma(u3, dr_q[kbase + j + 3u], outv);
        outv = fma(u4, dr_q[kbase + j + 4u], outv);
        outv = fma(u5, dr_q[kbase + j + 5u], outv);
        outv = fma(u6, dr_q[kbase + j + 6u], outv);
        outv = fma(u7, dr_q[kbase + j + 7u], outv);
        j = j + 8u;
    }
    loop {
        if (j >= dk) {
            break;
        }
        let p = sbase + j * dv;
        let u = fma(dr_k[kbase + j], delta, dr_state[p] * ge);
        dr_state[p] = u;
        outv = fma(u, dr_q[kbase + j], outv);
        j = j + 1u;
    }
    dr_out[h * dv + lane] = outv;
}

@compute @workgroup_size(32)
fn q3w_delta_recurrent_u8l32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    let lane = wid.y * 32u + lid.x;
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;
    let kbase = h * dk;
    let dv2 = dv * 2u;
    let dv3 = dv * 3u;
    let dv4 = dv * 4u;
    let dv5 = dv * 5u;
    let dv6 = dv * 6u;
    let dv7 = dv * 7u;

    var kv_mem = 0.0;
    var i = 0u;
    loop {
        if (i + 8u > dk) {
            break;
        }
        let p = sbase + i * dv;
        let s0 = dr_state[p] * ge;
        let s1 = dr_state[p + dv] * ge;
        let s2 = dr_state[p + dv2] * ge;
        let s3 = dr_state[p + dv3] * ge;
        let s4 = dr_state[p + dv4] * ge;
        let s5 = dr_state[p + dv5] * ge;
        let s6 = dr_state[p + dv6] * ge;
        let s7 = dr_state[p + dv7] * ge;
        kv_mem = fma(s0, dr_k[kbase + i], kv_mem);
        kv_mem = fma(s1, dr_k[kbase + i + 1u], kv_mem);
        kv_mem = fma(s2, dr_k[kbase + i + 2u], kv_mem);
        kv_mem = fma(s3, dr_k[kbase + i + 3u], kv_mem);
        kv_mem = fma(s4, dr_k[kbase + i + 4u], kv_mem);
        kv_mem = fma(s5, dr_k[kbase + i + 5u], kv_mem);
        kv_mem = fma(s6, dr_k[kbase + i + 6u], kv_mem);
        kv_mem = fma(s7, dr_k[kbase + i + 7u], kv_mem);
        i = i + 8u;
    }
    loop {
        if (i >= dk) {
            break;
        }
        kv_mem = fma(dr_state[sbase + i * dv] * ge, dr_k[kbase + i], kv_mem);
        i = i + 1u;
    }

    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;

    var outv = 0.0;
    var j = 0u;
    loop {
        if (j + 8u > dk) {
            break;
        }
        let p = sbase + j * dv;
        let t0 = dr_state[p] * ge;
        let t1 = dr_state[p + dv] * ge;
        let t2 = dr_state[p + dv2] * ge;
        let t3 = dr_state[p + dv3] * ge;
        let t4 = dr_state[p + dv4] * ge;
        let t5 = dr_state[p + dv5] * ge;
        let t6 = dr_state[p + dv6] * ge;
        let t7 = dr_state[p + dv7] * ge;
        let u0 = fma(dr_k[kbase + j], delta, t0);
        let u1 = fma(dr_k[kbase + j + 1u], delta, t1);
        let u2 = fma(dr_k[kbase + j + 2u], delta, t2);
        let u3 = fma(dr_k[kbase + j + 3u], delta, t3);
        let u4 = fma(dr_k[kbase + j + 4u], delta, t4);
        let u5 = fma(dr_k[kbase + j + 5u], delta, t5);
        let u6 = fma(dr_k[kbase + j + 6u], delta, t6);
        let u7 = fma(dr_k[kbase + j + 7u], delta, t7);
        dr_state[p] = u0;
        dr_state[p + dv] = u1;
        dr_state[p + dv2] = u2;
        dr_state[p + dv3] = u3;
        dr_state[p + dv4] = u4;
        dr_state[p + dv5] = u5;
        dr_state[p + dv6] = u6;
        dr_state[p + dv7] = u7;
        outv = fma(u0, dr_q[kbase + j], outv);
        outv = fma(u1, dr_q[kbase + j + 1u], outv);
        outv = fma(u2, dr_q[kbase + j + 2u], outv);
        outv = fma(u3, dr_q[kbase + j + 3u], outv);
        outv = fma(u4, dr_q[kbase + j + 4u], outv);
        outv = fma(u5, dr_q[kbase + j + 5u], outv);
        outv = fma(u6, dr_q[kbase + j + 6u], outv);
        outv = fma(u7, dr_q[kbase + j + 7u], outv);
        j = j + 8u;
    }
    loop {
        if (j >= dk) {
            break;
        }
        let p = sbase + j * dv;
        let u = fma(dr_k[kbase + j], delta, dr_state[p] * ge);
        dr_state[p] = u;
        outv = fma(u, dr_q[kbase + j], outv);
        j = j + 1u;
    }
    dr_out[h * dv + lane] = outv;
}

@compute @workgroup_size(128)
fn q3w_delta_recurrent_u16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    let lane = wid.y * 128u + lid.x;
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;
    let kbase = h * dk;
    let dv2 = dv * 2u;
    let dv3 = dv * 3u;
    let dv4 = dv * 4u;
    let dv5 = dv * 5u;
    let dv6 = dv * 6u;
    let dv7 = dv * 7u;
    let dv8 = dv * 8u;
    let dv9 = dv * 9u;
    let dv10 = dv * 10u;
    let dv11 = dv * 11u;
    let dv12 = dv * 12u;
    let dv13 = dv * 13u;
    let dv14 = dv * 14u;
    let dv15 = dv * 15u;

    var kv_mem = 0.0;
    var i = 0u;
    loop {
        if (i + 16u > dk) {
            break;
        }
        let p = sbase + i * dv;
        let s0 = dr_state[p] * ge;
        let s1 = dr_state[p + dv] * ge;
        let s2 = dr_state[p + dv2] * ge;
        let s3 = dr_state[p + dv3] * ge;
        let s4 = dr_state[p + dv4] * ge;
        let s5 = dr_state[p + dv5] * ge;
        let s6 = dr_state[p + dv6] * ge;
        let s7 = dr_state[p + dv7] * ge;
        let s8 = dr_state[p + dv8] * ge;
        let s9 = dr_state[p + dv9] * ge;
        let s10 = dr_state[p + dv10] * ge;
        let s11 = dr_state[p + dv11] * ge;
        let s12 = dr_state[p + dv12] * ge;
        let s13 = dr_state[p + dv13] * ge;
        let s14 = dr_state[p + dv14] * ge;
        let s15 = dr_state[p + dv15] * ge;
        kv_mem = fma(s0, dr_k[kbase + i], kv_mem);
        kv_mem = fma(s1, dr_k[kbase + i + 1u], kv_mem);
        kv_mem = fma(s2, dr_k[kbase + i + 2u], kv_mem);
        kv_mem = fma(s3, dr_k[kbase + i + 3u], kv_mem);
        kv_mem = fma(s4, dr_k[kbase + i + 4u], kv_mem);
        kv_mem = fma(s5, dr_k[kbase + i + 5u], kv_mem);
        kv_mem = fma(s6, dr_k[kbase + i + 6u], kv_mem);
        kv_mem = fma(s7, dr_k[kbase + i + 7u], kv_mem);
        kv_mem = fma(s8, dr_k[kbase + i + 8u], kv_mem);
        kv_mem = fma(s9, dr_k[kbase + i + 9u], kv_mem);
        kv_mem = fma(s10, dr_k[kbase + i + 10u], kv_mem);
        kv_mem = fma(s11, dr_k[kbase + i + 11u], kv_mem);
        kv_mem = fma(s12, dr_k[kbase + i + 12u], kv_mem);
        kv_mem = fma(s13, dr_k[kbase + i + 13u], kv_mem);
        kv_mem = fma(s14, dr_k[kbase + i + 14u], kv_mem);
        kv_mem = fma(s15, dr_k[kbase + i + 15u], kv_mem);
        i = i + 16u;
    }
    loop {
        if (i >= dk) {
            break;
        }
        kv_mem = fma(dr_state[sbase + i * dv] * ge, dr_k[kbase + i], kv_mem);
        i = i + 1u;
    }

    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;

    var outv = 0.0;
    var j = 0u;
    loop {
        if (j + 16u > dk) {
            break;
        }
        let p = sbase + j * dv;
        let t0 = dr_state[p] * ge;
        let t1 = dr_state[p + dv] * ge;
        let t2 = dr_state[p + dv2] * ge;
        let t3 = dr_state[p + dv3] * ge;
        let t4 = dr_state[p + dv4] * ge;
        let t5 = dr_state[p + dv5] * ge;
        let t6 = dr_state[p + dv6] * ge;
        let t7 = dr_state[p + dv7] * ge;
        let t8 = dr_state[p + dv8] * ge;
        let t9 = dr_state[p + dv9] * ge;
        let t10 = dr_state[p + dv10] * ge;
        let t11 = dr_state[p + dv11] * ge;
        let t12 = dr_state[p + dv12] * ge;
        let t13 = dr_state[p + dv13] * ge;
        let t14 = dr_state[p + dv14] * ge;
        let t15 = dr_state[p + dv15] * ge;
        let u0 = fma(dr_k[kbase + j], delta, t0);
        let u1 = fma(dr_k[kbase + j + 1u], delta, t1);
        let u2 = fma(dr_k[kbase + j + 2u], delta, t2);
        let u3 = fma(dr_k[kbase + j + 3u], delta, t3);
        let u4 = fma(dr_k[kbase + j + 4u], delta, t4);
        let u5 = fma(dr_k[kbase + j + 5u], delta, t5);
        let u6 = fma(dr_k[kbase + j + 6u], delta, t6);
        let u7 = fma(dr_k[kbase + j + 7u], delta, t7);
        let u8 = fma(dr_k[kbase + j + 8u], delta, t8);
        let u9 = fma(dr_k[kbase + j + 9u], delta, t9);
        let u10 = fma(dr_k[kbase + j + 10u], delta, t10);
        let u11 = fma(dr_k[kbase + j + 11u], delta, t11);
        let u12 = fma(dr_k[kbase + j + 12u], delta, t12);
        let u13 = fma(dr_k[kbase + j + 13u], delta, t13);
        let u14 = fma(dr_k[kbase + j + 14u], delta, t14);
        let u15 = fma(dr_k[kbase + j + 15u], delta, t15);
        dr_state[p] = u0;
        dr_state[p + dv] = u1;
        dr_state[p + dv2] = u2;
        dr_state[p + dv3] = u3;
        dr_state[p + dv4] = u4;
        dr_state[p + dv5] = u5;
        dr_state[p + dv6] = u6;
        dr_state[p + dv7] = u7;
        dr_state[p + dv8] = u8;
        dr_state[p + dv9] = u9;
        dr_state[p + dv10] = u10;
        dr_state[p + dv11] = u11;
        dr_state[p + dv12] = u12;
        dr_state[p + dv13] = u13;
        dr_state[p + dv14] = u14;
        dr_state[p + dv15] = u15;
        outv = fma(u0, dr_q[kbase + j], outv);
        outv = fma(u1, dr_q[kbase + j + 1u], outv);
        outv = fma(u2, dr_q[kbase + j + 2u], outv);
        outv = fma(u3, dr_q[kbase + j + 3u], outv);
        outv = fma(u4, dr_q[kbase + j + 4u], outv);
        outv = fma(u5, dr_q[kbase + j + 5u], outv);
        outv = fma(u6, dr_q[kbase + j + 6u], outv);
        outv = fma(u7, dr_q[kbase + j + 7u], outv);
        outv = fma(u8, dr_q[kbase + j + 8u], outv);
        outv = fma(u9, dr_q[kbase + j + 9u], outv);
        outv = fma(u10, dr_q[kbase + j + 10u], outv);
        outv = fma(u11, dr_q[kbase + j + 11u], outv);
        outv = fma(u12, dr_q[kbase + j + 12u], outv);
        outv = fma(u13, dr_q[kbase + j + 13u], outv);
        outv = fma(u14, dr_q[kbase + j + 14u], outv);
        outv = fma(u15, dr_q[kbase + j + 15u], outv);
        j = j + 16u;
    }
    loop {
        if (j >= dk) {
            break;
        }
        let p = sbase + j * dv;
        let u = fma(dr_k[kbase + j], delta, dr_state[p] * ge);
        dr_state[p] = u;
        outv = fma(u, dr_q[kbase + j], outv);
        j = j + 1u;
    }
    dr_out[h * dv + lane] = outv;
}

@compute @workgroup_size(32)
fn q3w_delta_recurrent_u16l32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    let lane = wid.y * 32u + lid.x;
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;
    let kbase = h * dk;
    let dv2 = dv * 2u;
    let dv3 = dv * 3u;
    let dv4 = dv * 4u;
    let dv5 = dv * 5u;
    let dv6 = dv * 6u;
    let dv7 = dv * 7u;
    let dv8 = dv * 8u;
    let dv9 = dv * 9u;
    let dv10 = dv * 10u;
    let dv11 = dv * 11u;
    let dv12 = dv * 12u;
    let dv13 = dv * 13u;
    let dv14 = dv * 14u;
    let dv15 = dv * 15u;

    var kv_mem = 0.0;
    var i = 0u;
    loop {
        if (i + 16u > dk) {
            break;
        }
        let p = sbase + i * dv;
        let s0 = dr_state[p] * ge;
        let s1 = dr_state[p + dv] * ge;
        let s2 = dr_state[p + dv2] * ge;
        let s3 = dr_state[p + dv3] * ge;
        let s4 = dr_state[p + dv4] * ge;
        let s5 = dr_state[p + dv5] * ge;
        let s6 = dr_state[p + dv6] * ge;
        let s7 = dr_state[p + dv7] * ge;
        let s8 = dr_state[p + dv8] * ge;
        let s9 = dr_state[p + dv9] * ge;
        let s10 = dr_state[p + dv10] * ge;
        let s11 = dr_state[p + dv11] * ge;
        let s12 = dr_state[p + dv12] * ge;
        let s13 = dr_state[p + dv13] * ge;
        let s14 = dr_state[p + dv14] * ge;
        let s15 = dr_state[p + dv15] * ge;
        kv_mem = fma(s0, dr_k[kbase + i], kv_mem);
        kv_mem = fma(s1, dr_k[kbase + i + 1u], kv_mem);
        kv_mem = fma(s2, dr_k[kbase + i + 2u], kv_mem);
        kv_mem = fma(s3, dr_k[kbase + i + 3u], kv_mem);
        kv_mem = fma(s4, dr_k[kbase + i + 4u], kv_mem);
        kv_mem = fma(s5, dr_k[kbase + i + 5u], kv_mem);
        kv_mem = fma(s6, dr_k[kbase + i + 6u], kv_mem);
        kv_mem = fma(s7, dr_k[kbase + i + 7u], kv_mem);
        kv_mem = fma(s8, dr_k[kbase + i + 8u], kv_mem);
        kv_mem = fma(s9, dr_k[kbase + i + 9u], kv_mem);
        kv_mem = fma(s10, dr_k[kbase + i + 10u], kv_mem);
        kv_mem = fma(s11, dr_k[kbase + i + 11u], kv_mem);
        kv_mem = fma(s12, dr_k[kbase + i + 12u], kv_mem);
        kv_mem = fma(s13, dr_k[kbase + i + 13u], kv_mem);
        kv_mem = fma(s14, dr_k[kbase + i + 14u], kv_mem);
        kv_mem = fma(s15, dr_k[kbase + i + 15u], kv_mem);
        i = i + 16u;
    }
    loop {
        if (i >= dk) {
            break;
        }
        kv_mem = fma(dr_state[sbase + i * dv] * ge, dr_k[kbase + i], kv_mem);
        i = i + 1u;
    }

    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;

    var outv = 0.0;
    var j = 0u;
    loop {
        if (j + 16u > dk) {
            break;
        }
        let p = sbase + j * dv;
        let t0 = dr_state[p] * ge;
        let t1 = dr_state[p + dv] * ge;
        let t2 = dr_state[p + dv2] * ge;
        let t3 = dr_state[p + dv3] * ge;
        let t4 = dr_state[p + dv4] * ge;
        let t5 = dr_state[p + dv5] * ge;
        let t6 = dr_state[p + dv6] * ge;
        let t7 = dr_state[p + dv7] * ge;
        let t8 = dr_state[p + dv8] * ge;
        let t9 = dr_state[p + dv9] * ge;
        let t10 = dr_state[p + dv10] * ge;
        let t11 = dr_state[p + dv11] * ge;
        let t12 = dr_state[p + dv12] * ge;
        let t13 = dr_state[p + dv13] * ge;
        let t14 = dr_state[p + dv14] * ge;
        let t15 = dr_state[p + dv15] * ge;
        let u0 = fma(dr_k[kbase + j], delta, t0);
        let u1 = fma(dr_k[kbase + j + 1u], delta, t1);
        let u2 = fma(dr_k[kbase + j + 2u], delta, t2);
        let u3 = fma(dr_k[kbase + j + 3u], delta, t3);
        let u4 = fma(dr_k[kbase + j + 4u], delta, t4);
        let u5 = fma(dr_k[kbase + j + 5u], delta, t5);
        let u6 = fma(dr_k[kbase + j + 6u], delta, t6);
        let u7 = fma(dr_k[kbase + j + 7u], delta, t7);
        let u8 = fma(dr_k[kbase + j + 8u], delta, t8);
        let u9 = fma(dr_k[kbase + j + 9u], delta, t9);
        let u10 = fma(dr_k[kbase + j + 10u], delta, t10);
        let u11 = fma(dr_k[kbase + j + 11u], delta, t11);
        let u12 = fma(dr_k[kbase + j + 12u], delta, t12);
        let u13 = fma(dr_k[kbase + j + 13u], delta, t13);
        let u14 = fma(dr_k[kbase + j + 14u], delta, t14);
        let u15 = fma(dr_k[kbase + j + 15u], delta, t15);
        dr_state[p] = u0;
        dr_state[p + dv] = u1;
        dr_state[p + dv2] = u2;
        dr_state[p + dv3] = u3;
        dr_state[p + dv4] = u4;
        dr_state[p + dv5] = u5;
        dr_state[p + dv6] = u6;
        dr_state[p + dv7] = u7;
        dr_state[p + dv8] = u8;
        dr_state[p + dv9] = u9;
        dr_state[p + dv10] = u10;
        dr_state[p + dv11] = u11;
        dr_state[p + dv12] = u12;
        dr_state[p + dv13] = u13;
        dr_state[p + dv14] = u14;
        dr_state[p + dv15] = u15;
        outv = fma(u0, dr_q[kbase + j], outv);
        outv = fma(u1, dr_q[kbase + j + 1u], outv);
        outv = fma(u2, dr_q[kbase + j + 2u], outv);
        outv = fma(u3, dr_q[kbase + j + 3u], outv);
        outv = fma(u4, dr_q[kbase + j + 4u], outv);
        outv = fma(u5, dr_q[kbase + j + 5u], outv);
        outv = fma(u6, dr_q[kbase + j + 6u], outv);
        outv = fma(u7, dr_q[kbase + j + 7u], outv);
        outv = fma(u8, dr_q[kbase + j + 8u], outv);
        outv = fma(u9, dr_q[kbase + j + 9u], outv);
        outv = fma(u10, dr_q[kbase + j + 10u], outv);
        outv = fma(u11, dr_q[kbase + j + 11u], outv);
        outv = fma(u12, dr_q[kbase + j + 12u], outv);
        outv = fma(u13, dr_q[kbase + j + 13u], outv);
        outv = fma(u14, dr_q[kbase + j + 14u], outv);
        outv = fma(u15, dr_q[kbase + j + 15u], outv);
        j = j + 16u;
    }
    loop {
        if (j >= dk) {
            break;
        }
        let p = sbase + j * dv;
        let u = fma(dr_k[kbase + j], delta, dr_state[p] * ge);
        dr_state[p] = u;
        outv = fma(u, dr_q[kbase + j], outv);
        j = j + 1u;
    }
    dr_out[h * dv + lane] = outv;
}

@compute @workgroup_size(128)
fn q3w_delta_recurrent_u32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    let lane = wid.y * 128u + lid.x;
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;
    let kbase = h * dk;
    let dv2 = dv * 2u;
    let dv3 = dv * 3u;
    let dv4 = dv * 4u;
    let dv5 = dv * 5u;
    let dv6 = dv * 6u;
    let dv7 = dv * 7u;
    let dv8 = dv * 8u;
    let dv9 = dv * 9u;
    let dv10 = dv * 10u;
    let dv11 = dv * 11u;
    let dv12 = dv * 12u;
    let dv13 = dv * 13u;
    let dv14 = dv * 14u;
    let dv15 = dv * 15u;
    let dv16 = dv * 16u;
    let dv17 = dv * 17u;
    let dv18 = dv * 18u;
    let dv19 = dv * 19u;
    let dv20 = dv * 20u;
    let dv21 = dv * 21u;
    let dv22 = dv * 22u;
    let dv23 = dv * 23u;
    let dv24 = dv * 24u;
    let dv25 = dv * 25u;
    let dv26 = dv * 26u;
    let dv27 = dv * 27u;
    let dv28 = dv * 28u;
    let dv29 = dv * 29u;
    let dv30 = dv * 30u;
    let dv31 = dv * 31u;

    var kv_mem = 0.0;
    var i = 0u;
    loop {
        if (i + 32u > dk) {
            break;
        }
        let p = sbase + i * dv;
        let s0 = dr_state[p] * ge;
        let s1 = dr_state[p + dv] * ge;
        let s2 = dr_state[p + dv2] * ge;
        let s3 = dr_state[p + dv3] * ge;
        let s4 = dr_state[p + dv4] * ge;
        let s5 = dr_state[p + dv5] * ge;
        let s6 = dr_state[p + dv6] * ge;
        let s7 = dr_state[p + dv7] * ge;
        let s8 = dr_state[p + dv8] * ge;
        let s9 = dr_state[p + dv9] * ge;
        let s10 = dr_state[p + dv10] * ge;
        let s11 = dr_state[p + dv11] * ge;
        let s12 = dr_state[p + dv12] * ge;
        let s13 = dr_state[p + dv13] * ge;
        let s14 = dr_state[p + dv14] * ge;
        let s15 = dr_state[p + dv15] * ge;
        let s16 = dr_state[p + dv16] * ge;
        let s17 = dr_state[p + dv17] * ge;
        let s18 = dr_state[p + dv18] * ge;
        let s19 = dr_state[p + dv19] * ge;
        let s20 = dr_state[p + dv20] * ge;
        let s21 = dr_state[p + dv21] * ge;
        let s22 = dr_state[p + dv22] * ge;
        let s23 = dr_state[p + dv23] * ge;
        let s24 = dr_state[p + dv24] * ge;
        let s25 = dr_state[p + dv25] * ge;
        let s26 = dr_state[p + dv26] * ge;
        let s27 = dr_state[p + dv27] * ge;
        let s28 = dr_state[p + dv28] * ge;
        let s29 = dr_state[p + dv29] * ge;
        let s30 = dr_state[p + dv30] * ge;
        let s31 = dr_state[p + dv31] * ge;
        kv_mem = fma(s0, dr_k[kbase + i], kv_mem);
        kv_mem = fma(s1, dr_k[kbase + i + 1u], kv_mem);
        kv_mem = fma(s2, dr_k[kbase + i + 2u], kv_mem);
        kv_mem = fma(s3, dr_k[kbase + i + 3u], kv_mem);
        kv_mem = fma(s4, dr_k[kbase + i + 4u], kv_mem);
        kv_mem = fma(s5, dr_k[kbase + i + 5u], kv_mem);
        kv_mem = fma(s6, dr_k[kbase + i + 6u], kv_mem);
        kv_mem = fma(s7, dr_k[kbase + i + 7u], kv_mem);
        kv_mem = fma(s8, dr_k[kbase + i + 8u], kv_mem);
        kv_mem = fma(s9, dr_k[kbase + i + 9u], kv_mem);
        kv_mem = fma(s10, dr_k[kbase + i + 10u], kv_mem);
        kv_mem = fma(s11, dr_k[kbase + i + 11u], kv_mem);
        kv_mem = fma(s12, dr_k[kbase + i + 12u], kv_mem);
        kv_mem = fma(s13, dr_k[kbase + i + 13u], kv_mem);
        kv_mem = fma(s14, dr_k[kbase + i + 14u], kv_mem);
        kv_mem = fma(s15, dr_k[kbase + i + 15u], kv_mem);
        kv_mem = fma(s16, dr_k[kbase + i + 16u], kv_mem);
        kv_mem = fma(s17, dr_k[kbase + i + 17u], kv_mem);
        kv_mem = fma(s18, dr_k[kbase + i + 18u], kv_mem);
        kv_mem = fma(s19, dr_k[kbase + i + 19u], kv_mem);
        kv_mem = fma(s20, dr_k[kbase + i + 20u], kv_mem);
        kv_mem = fma(s21, dr_k[kbase + i + 21u], kv_mem);
        kv_mem = fma(s22, dr_k[kbase + i + 22u], kv_mem);
        kv_mem = fma(s23, dr_k[kbase + i + 23u], kv_mem);
        kv_mem = fma(s24, dr_k[kbase + i + 24u], kv_mem);
        kv_mem = fma(s25, dr_k[kbase + i + 25u], kv_mem);
        kv_mem = fma(s26, dr_k[kbase + i + 26u], kv_mem);
        kv_mem = fma(s27, dr_k[kbase + i + 27u], kv_mem);
        kv_mem = fma(s28, dr_k[kbase + i + 28u], kv_mem);
        kv_mem = fma(s29, dr_k[kbase + i + 29u], kv_mem);
        kv_mem = fma(s30, dr_k[kbase + i + 30u], kv_mem);
        kv_mem = fma(s31, dr_k[kbase + i + 31u], kv_mem);
        i = i + 32u;
    }
    loop {
        if (i >= dk) {
            break;
        }
        kv_mem = fma(dr_state[sbase + i * dv] * ge, dr_k[kbase + i], kv_mem);
        i = i + 1u;
    }

    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;

    var outv = 0.0;
    var j = 0u;
    loop {
        if (j + 32u > dk) {
            break;
        }
        let p = sbase + j * dv;
        let t0 = dr_state[p] * ge;
        let t1 = dr_state[p + dv] * ge;
        let t2 = dr_state[p + dv2] * ge;
        let t3 = dr_state[p + dv3] * ge;
        let t4 = dr_state[p + dv4] * ge;
        let t5 = dr_state[p + dv5] * ge;
        let t6 = dr_state[p + dv6] * ge;
        let t7 = dr_state[p + dv7] * ge;
        let t8 = dr_state[p + dv8] * ge;
        let t9 = dr_state[p + dv9] * ge;
        let t10 = dr_state[p + dv10] * ge;
        let t11 = dr_state[p + dv11] * ge;
        let t12 = dr_state[p + dv12] * ge;
        let t13 = dr_state[p + dv13] * ge;
        let t14 = dr_state[p + dv14] * ge;
        let t15 = dr_state[p + dv15] * ge;
        let t16 = dr_state[p + dv16] * ge;
        let t17 = dr_state[p + dv17] * ge;
        let t18 = dr_state[p + dv18] * ge;
        let t19 = dr_state[p + dv19] * ge;
        let t20 = dr_state[p + dv20] * ge;
        let t21 = dr_state[p + dv21] * ge;
        let t22 = dr_state[p + dv22] * ge;
        let t23 = dr_state[p + dv23] * ge;
        let t24 = dr_state[p + dv24] * ge;
        let t25 = dr_state[p + dv25] * ge;
        let t26 = dr_state[p + dv26] * ge;
        let t27 = dr_state[p + dv27] * ge;
        let t28 = dr_state[p + dv28] * ge;
        let t29 = dr_state[p + dv29] * ge;
        let t30 = dr_state[p + dv30] * ge;
        let t31 = dr_state[p + dv31] * ge;
        let u0 = fma(dr_k[kbase + j], delta, t0);
        let u1 = fma(dr_k[kbase + j + 1u], delta, t1);
        let u2 = fma(dr_k[kbase + j + 2u], delta, t2);
        let u3 = fma(dr_k[kbase + j + 3u], delta, t3);
        let u4 = fma(dr_k[kbase + j + 4u], delta, t4);
        let u5 = fma(dr_k[kbase + j + 5u], delta, t5);
        let u6 = fma(dr_k[kbase + j + 6u], delta, t6);
        let u7 = fma(dr_k[kbase + j + 7u], delta, t7);
        let u8 = fma(dr_k[kbase + j + 8u], delta, t8);
        let u9 = fma(dr_k[kbase + j + 9u], delta, t9);
        let u10 = fma(dr_k[kbase + j + 10u], delta, t10);
        let u11 = fma(dr_k[kbase + j + 11u], delta, t11);
        let u12 = fma(dr_k[kbase + j + 12u], delta, t12);
        let u13 = fma(dr_k[kbase + j + 13u], delta, t13);
        let u14 = fma(dr_k[kbase + j + 14u], delta, t14);
        let u15 = fma(dr_k[kbase + j + 15u], delta, t15);
        let u16 = fma(dr_k[kbase + j + 16u], delta, t16);
        let u17 = fma(dr_k[kbase + j + 17u], delta, t17);
        let u18 = fma(dr_k[kbase + j + 18u], delta, t18);
        let u19 = fma(dr_k[kbase + j + 19u], delta, t19);
        let u20 = fma(dr_k[kbase + j + 20u], delta, t20);
        let u21 = fma(dr_k[kbase + j + 21u], delta, t21);
        let u22 = fma(dr_k[kbase + j + 22u], delta, t22);
        let u23 = fma(dr_k[kbase + j + 23u], delta, t23);
        let u24 = fma(dr_k[kbase + j + 24u], delta, t24);
        let u25 = fma(dr_k[kbase + j + 25u], delta, t25);
        let u26 = fma(dr_k[kbase + j + 26u], delta, t26);
        let u27 = fma(dr_k[kbase + j + 27u], delta, t27);
        let u28 = fma(dr_k[kbase + j + 28u], delta, t28);
        let u29 = fma(dr_k[kbase + j + 29u], delta, t29);
        let u30 = fma(dr_k[kbase + j + 30u], delta, t30);
        let u31 = fma(dr_k[kbase + j + 31u], delta, t31);
        dr_state[p] = u0;
        dr_state[p + dv] = u1;
        dr_state[p + dv2] = u2;
        dr_state[p + dv3] = u3;
        dr_state[p + dv4] = u4;
        dr_state[p + dv5] = u5;
        dr_state[p + dv6] = u6;
        dr_state[p + dv7] = u7;
        dr_state[p + dv8] = u8;
        dr_state[p + dv9] = u9;
        dr_state[p + dv10] = u10;
        dr_state[p + dv11] = u11;
        dr_state[p + dv12] = u12;
        dr_state[p + dv13] = u13;
        dr_state[p + dv14] = u14;
        dr_state[p + dv15] = u15;
        dr_state[p + dv16] = u16;
        dr_state[p + dv17] = u17;
        dr_state[p + dv18] = u18;
        dr_state[p + dv19] = u19;
        dr_state[p + dv20] = u20;
        dr_state[p + dv21] = u21;
        dr_state[p + dv22] = u22;
        dr_state[p + dv23] = u23;
        dr_state[p + dv24] = u24;
        dr_state[p + dv25] = u25;
        dr_state[p + dv26] = u26;
        dr_state[p + dv27] = u27;
        dr_state[p + dv28] = u28;
        dr_state[p + dv29] = u29;
        dr_state[p + dv30] = u30;
        dr_state[p + dv31] = u31;
        outv = fma(u0, dr_q[kbase + j], outv);
        outv = fma(u1, dr_q[kbase + j + 1u], outv);
        outv = fma(u2, dr_q[kbase + j + 2u], outv);
        outv = fma(u3, dr_q[kbase + j + 3u], outv);
        outv = fma(u4, dr_q[kbase + j + 4u], outv);
        outv = fma(u5, dr_q[kbase + j + 5u], outv);
        outv = fma(u6, dr_q[kbase + j + 6u], outv);
        outv = fma(u7, dr_q[kbase + j + 7u], outv);
        outv = fma(u8, dr_q[kbase + j + 8u], outv);
        outv = fma(u9, dr_q[kbase + j + 9u], outv);
        outv = fma(u10, dr_q[kbase + j + 10u], outv);
        outv = fma(u11, dr_q[kbase + j + 11u], outv);
        outv = fma(u12, dr_q[kbase + j + 12u], outv);
        outv = fma(u13, dr_q[kbase + j + 13u], outv);
        outv = fma(u14, dr_q[kbase + j + 14u], outv);
        outv = fma(u15, dr_q[kbase + j + 15u], outv);
        outv = fma(u16, dr_q[kbase + j + 16u], outv);
        outv = fma(u17, dr_q[kbase + j + 17u], outv);
        outv = fma(u18, dr_q[kbase + j + 18u], outv);
        outv = fma(u19, dr_q[kbase + j + 19u], outv);
        outv = fma(u20, dr_q[kbase + j + 20u], outv);
        outv = fma(u21, dr_q[kbase + j + 21u], outv);
        outv = fma(u22, dr_q[kbase + j + 22u], outv);
        outv = fma(u23, dr_q[kbase + j + 23u], outv);
        outv = fma(u24, dr_q[kbase + j + 24u], outv);
        outv = fma(u25, dr_q[kbase + j + 25u], outv);
        outv = fma(u26, dr_q[kbase + j + 26u], outv);
        outv = fma(u27, dr_q[kbase + j + 27u], outv);
        outv = fma(u28, dr_q[kbase + j + 28u], outv);
        outv = fma(u29, dr_q[kbase + j + 29u], outv);
        outv = fma(u30, dr_q[kbase + j + 30u], outv);
        outv = fma(u31, dr_q[kbase + j + 31u], outv);
        j = j + 32u;
    }
    loop {
        if (j >= dk) {
            break;
        }
        let p = sbase + j * dv;
        let u = fma(dr_k[kbase + j], delta, dr_state[p] * ge);
        dr_state[p] = u;
        outv = fma(u, dr_q[kbase + j], outv);
        j = j + 1u;
    }
    dr_out[h * dv + lane] = outv;
}

@compute @workgroup_size(32)
fn q3w_delta_recurrent_u32l32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let dk = dr_p.d_k;
    let dv = dr_p.d_v;
    let lane = wid.y * 32u + lid.x;
    if (lane >= dv) {
        return;
    }
    let ge = dr_g[h];
    let bt = dr_beta[h];
    let sbase = h * dk * dv + lane;
    let kbase = h * dk;
    let dv2 = dv * 2u;
    let dv3 = dv * 3u;
    let dv4 = dv * 4u;
    let dv5 = dv * 5u;
    let dv6 = dv * 6u;
    let dv7 = dv * 7u;
    let dv8 = dv * 8u;
    let dv9 = dv * 9u;
    let dv10 = dv * 10u;
    let dv11 = dv * 11u;
    let dv12 = dv * 12u;
    let dv13 = dv * 13u;
    let dv14 = dv * 14u;
    let dv15 = dv * 15u;
    let dv16 = dv * 16u;
    let dv17 = dv * 17u;
    let dv18 = dv * 18u;
    let dv19 = dv * 19u;
    let dv20 = dv * 20u;
    let dv21 = dv * 21u;
    let dv22 = dv * 22u;
    let dv23 = dv * 23u;
    let dv24 = dv * 24u;
    let dv25 = dv * 25u;
    let dv26 = dv * 26u;
    let dv27 = dv * 27u;
    let dv28 = dv * 28u;
    let dv29 = dv * 29u;
    let dv30 = dv * 30u;
    let dv31 = dv * 31u;

    var kv_mem = 0.0;
    var i = 0u;
    loop {
        if (i + 32u > dk) {
            break;
        }
        let p = sbase + i * dv;
        let s0 = dr_state[p] * ge;
        let s1 = dr_state[p + dv] * ge;
        let s2 = dr_state[p + dv2] * ge;
        let s3 = dr_state[p + dv3] * ge;
        let s4 = dr_state[p + dv4] * ge;
        let s5 = dr_state[p + dv5] * ge;
        let s6 = dr_state[p + dv6] * ge;
        let s7 = dr_state[p + dv7] * ge;
        let s8 = dr_state[p + dv8] * ge;
        let s9 = dr_state[p + dv9] * ge;
        let s10 = dr_state[p + dv10] * ge;
        let s11 = dr_state[p + dv11] * ge;
        let s12 = dr_state[p + dv12] * ge;
        let s13 = dr_state[p + dv13] * ge;
        let s14 = dr_state[p + dv14] * ge;
        let s15 = dr_state[p + dv15] * ge;
        let s16 = dr_state[p + dv16] * ge;
        let s17 = dr_state[p + dv17] * ge;
        let s18 = dr_state[p + dv18] * ge;
        let s19 = dr_state[p + dv19] * ge;
        let s20 = dr_state[p + dv20] * ge;
        let s21 = dr_state[p + dv21] * ge;
        let s22 = dr_state[p + dv22] * ge;
        let s23 = dr_state[p + dv23] * ge;
        let s24 = dr_state[p + dv24] * ge;
        let s25 = dr_state[p + dv25] * ge;
        let s26 = dr_state[p + dv26] * ge;
        let s27 = dr_state[p + dv27] * ge;
        let s28 = dr_state[p + dv28] * ge;
        let s29 = dr_state[p + dv29] * ge;
        let s30 = dr_state[p + dv30] * ge;
        let s31 = dr_state[p + dv31] * ge;
        kv_mem = fma(s0, dr_k[kbase + i], kv_mem);
        kv_mem = fma(s1, dr_k[kbase + i + 1u], kv_mem);
        kv_mem = fma(s2, dr_k[kbase + i + 2u], kv_mem);
        kv_mem = fma(s3, dr_k[kbase + i + 3u], kv_mem);
        kv_mem = fma(s4, dr_k[kbase + i + 4u], kv_mem);
        kv_mem = fma(s5, dr_k[kbase + i + 5u], kv_mem);
        kv_mem = fma(s6, dr_k[kbase + i + 6u], kv_mem);
        kv_mem = fma(s7, dr_k[kbase + i + 7u], kv_mem);
        kv_mem = fma(s8, dr_k[kbase + i + 8u], kv_mem);
        kv_mem = fma(s9, dr_k[kbase + i + 9u], kv_mem);
        kv_mem = fma(s10, dr_k[kbase + i + 10u], kv_mem);
        kv_mem = fma(s11, dr_k[kbase + i + 11u], kv_mem);
        kv_mem = fma(s12, dr_k[kbase + i + 12u], kv_mem);
        kv_mem = fma(s13, dr_k[kbase + i + 13u], kv_mem);
        kv_mem = fma(s14, dr_k[kbase + i + 14u], kv_mem);
        kv_mem = fma(s15, dr_k[kbase + i + 15u], kv_mem);
        kv_mem = fma(s16, dr_k[kbase + i + 16u], kv_mem);
        kv_mem = fma(s17, dr_k[kbase + i + 17u], kv_mem);
        kv_mem = fma(s18, dr_k[kbase + i + 18u], kv_mem);
        kv_mem = fma(s19, dr_k[kbase + i + 19u], kv_mem);
        kv_mem = fma(s20, dr_k[kbase + i + 20u], kv_mem);
        kv_mem = fma(s21, dr_k[kbase + i + 21u], kv_mem);
        kv_mem = fma(s22, dr_k[kbase + i + 22u], kv_mem);
        kv_mem = fma(s23, dr_k[kbase + i + 23u], kv_mem);
        kv_mem = fma(s24, dr_k[kbase + i + 24u], kv_mem);
        kv_mem = fma(s25, dr_k[kbase + i + 25u], kv_mem);
        kv_mem = fma(s26, dr_k[kbase + i + 26u], kv_mem);
        kv_mem = fma(s27, dr_k[kbase + i + 27u], kv_mem);
        kv_mem = fma(s28, dr_k[kbase + i + 28u], kv_mem);
        kv_mem = fma(s29, dr_k[kbase + i + 29u], kv_mem);
        kv_mem = fma(s30, dr_k[kbase + i + 30u], kv_mem);
        kv_mem = fma(s31, dr_k[kbase + i + 31u], kv_mem);
        i = i + 32u;
    }
    loop {
        if (i >= dk) {
            break;
        }
        kv_mem = fma(dr_state[sbase + i * dv] * ge, dr_k[kbase + i], kv_mem);
        i = i + 1u;
    }

    let delta = (dr_v[h * dv + lane] - kv_mem) * bt;

    var outv = 0.0;
    var j = 0u;
    loop {
        if (j + 32u > dk) {
            break;
        }
        let p = sbase + j * dv;
        let t0 = dr_state[p] * ge;
        let t1 = dr_state[p + dv] * ge;
        let t2 = dr_state[p + dv2] * ge;
        let t3 = dr_state[p + dv3] * ge;
        let t4 = dr_state[p + dv4] * ge;
        let t5 = dr_state[p + dv5] * ge;
        let t6 = dr_state[p + dv6] * ge;
        let t7 = dr_state[p + dv7] * ge;
        let t8 = dr_state[p + dv8] * ge;
        let t9 = dr_state[p + dv9] * ge;
        let t10 = dr_state[p + dv10] * ge;
        let t11 = dr_state[p + dv11] * ge;
        let t12 = dr_state[p + dv12] * ge;
        let t13 = dr_state[p + dv13] * ge;
        let t14 = dr_state[p + dv14] * ge;
        let t15 = dr_state[p + dv15] * ge;
        let t16 = dr_state[p + dv16] * ge;
        let t17 = dr_state[p + dv17] * ge;
        let t18 = dr_state[p + dv18] * ge;
        let t19 = dr_state[p + dv19] * ge;
        let t20 = dr_state[p + dv20] * ge;
        let t21 = dr_state[p + dv21] * ge;
        let t22 = dr_state[p + dv22] * ge;
        let t23 = dr_state[p + dv23] * ge;
        let t24 = dr_state[p + dv24] * ge;
        let t25 = dr_state[p + dv25] * ge;
        let t26 = dr_state[p + dv26] * ge;
        let t27 = dr_state[p + dv27] * ge;
        let t28 = dr_state[p + dv28] * ge;
        let t29 = dr_state[p + dv29] * ge;
        let t30 = dr_state[p + dv30] * ge;
        let t31 = dr_state[p + dv31] * ge;
        let u0 = fma(dr_k[kbase + j], delta, t0);
        let u1 = fma(dr_k[kbase + j + 1u], delta, t1);
        let u2 = fma(dr_k[kbase + j + 2u], delta, t2);
        let u3 = fma(dr_k[kbase + j + 3u], delta, t3);
        let u4 = fma(dr_k[kbase + j + 4u], delta, t4);
        let u5 = fma(dr_k[kbase + j + 5u], delta, t5);
        let u6 = fma(dr_k[kbase + j + 6u], delta, t6);
        let u7 = fma(dr_k[kbase + j + 7u], delta, t7);
        let u8 = fma(dr_k[kbase + j + 8u], delta, t8);
        let u9 = fma(dr_k[kbase + j + 9u], delta, t9);
        let u10 = fma(dr_k[kbase + j + 10u], delta, t10);
        let u11 = fma(dr_k[kbase + j + 11u], delta, t11);
        let u12 = fma(dr_k[kbase + j + 12u], delta, t12);
        let u13 = fma(dr_k[kbase + j + 13u], delta, t13);
        let u14 = fma(dr_k[kbase + j + 14u], delta, t14);
        let u15 = fma(dr_k[kbase + j + 15u], delta, t15);
        let u16 = fma(dr_k[kbase + j + 16u], delta, t16);
        let u17 = fma(dr_k[kbase + j + 17u], delta, t17);
        let u18 = fma(dr_k[kbase + j + 18u], delta, t18);
        let u19 = fma(dr_k[kbase + j + 19u], delta, t19);
        let u20 = fma(dr_k[kbase + j + 20u], delta, t20);
        let u21 = fma(dr_k[kbase + j + 21u], delta, t21);
        let u22 = fma(dr_k[kbase + j + 22u], delta, t22);
        let u23 = fma(dr_k[kbase + j + 23u], delta, t23);
        let u24 = fma(dr_k[kbase + j + 24u], delta, t24);
        let u25 = fma(dr_k[kbase + j + 25u], delta, t25);
        let u26 = fma(dr_k[kbase + j + 26u], delta, t26);
        let u27 = fma(dr_k[kbase + j + 27u], delta, t27);
        let u28 = fma(dr_k[kbase + j + 28u], delta, t28);
        let u29 = fma(dr_k[kbase + j + 29u], delta, t29);
        let u30 = fma(dr_k[kbase + j + 30u], delta, t30);
        let u31 = fma(dr_k[kbase + j + 31u], delta, t31);
        dr_state[p] = u0;
        dr_state[p + dv] = u1;
        dr_state[p + dv2] = u2;
        dr_state[p + dv3] = u3;
        dr_state[p + dv4] = u4;
        dr_state[p + dv5] = u5;
        dr_state[p + dv6] = u6;
        dr_state[p + dv7] = u7;
        dr_state[p + dv8] = u8;
        dr_state[p + dv9] = u9;
        dr_state[p + dv10] = u10;
        dr_state[p + dv11] = u11;
        dr_state[p + dv12] = u12;
        dr_state[p + dv13] = u13;
        dr_state[p + dv14] = u14;
        dr_state[p + dv15] = u15;
        dr_state[p + dv16] = u16;
        dr_state[p + dv17] = u17;
        dr_state[p + dv18] = u18;
        dr_state[p + dv19] = u19;
        dr_state[p + dv20] = u20;
        dr_state[p + dv21] = u21;
        dr_state[p + dv22] = u22;
        dr_state[p + dv23] = u23;
        dr_state[p + dv24] = u24;
        dr_state[p + dv25] = u25;
        dr_state[p + dv26] = u26;
        dr_state[p + dv27] = u27;
        dr_state[p + dv28] = u28;
        dr_state[p + dv29] = u29;
        dr_state[p + dv30] = u30;
        dr_state[p + dv31] = u31;
        outv = fma(u0, dr_q[kbase + j], outv);
        outv = fma(u1, dr_q[kbase + j + 1u], outv);
        outv = fma(u2, dr_q[kbase + j + 2u], outv);
        outv = fma(u3, dr_q[kbase + j + 3u], outv);
        outv = fma(u4, dr_q[kbase + j + 4u], outv);
        outv = fma(u5, dr_q[kbase + j + 5u], outv);
        outv = fma(u6, dr_q[kbase + j + 6u], outv);
        outv = fma(u7, dr_q[kbase + j + 7u], outv);
        outv = fma(u8, dr_q[kbase + j + 8u], outv);
        outv = fma(u9, dr_q[kbase + j + 9u], outv);
        outv = fma(u10, dr_q[kbase + j + 10u], outv);
        outv = fma(u11, dr_q[kbase + j + 11u], outv);
        outv = fma(u12, dr_q[kbase + j + 12u], outv);
        outv = fma(u13, dr_q[kbase + j + 13u], outv);
        outv = fma(u14, dr_q[kbase + j + 14u], outv);
        outv = fma(u15, dr_q[kbase + j + 15u], outv);
        outv = fma(u16, dr_q[kbase + j + 16u], outv);
        outv = fma(u17, dr_q[kbase + j + 17u], outv);
        outv = fma(u18, dr_q[kbase + j + 18u], outv);
        outv = fma(u19, dr_q[kbase + j + 19u], outv);
        outv = fma(u20, dr_q[kbase + j + 20u], outv);
        outv = fma(u21, dr_q[kbase + j + 21u], outv);
        outv = fma(u22, dr_q[kbase + j + 22u], outv);
        outv = fma(u23, dr_q[kbase + j + 23u], outv);
        outv = fma(u24, dr_q[kbase + j + 24u], outv);
        outv = fma(u25, dr_q[kbase + j + 25u], outv);
        outv = fma(u26, dr_q[kbase + j + 26u], outv);
        outv = fma(u27, dr_q[kbase + j + 27u], outv);
        outv = fma(u28, dr_q[kbase + j + 28u], outv);
        outv = fma(u29, dr_q[kbase + j + 29u], outv);
        outv = fma(u30, dr_q[kbase + j + 30u], outv);
        outv = fma(u31, dr_q[kbase + j + 31u], outv);
        j = j + 32u;
    }
    loop {
        if (j >= dk) {
            break;
        }
        let p = sbase + j * dv;
        let u = fma(dr_k[kbase + j], delta, dr_state[p] * ge);
        dr_state[p] = u;
        outv = fma(u, dr_q[kbase + j], outv);
        j = j + 1u;
    }
    dr_out[h * dv + lane] = outv;
}

struct Q3doParams {
    n_v: u32,
    d_v: u32,
    z_off: u32,
    eps: f32,
};

@group(0) @binding(40) var<storage, read> do_core: array<f32>;
@group(0) @binding(41) var<storage, read> do_w: array<u32>;
@group(0) @binding(42) var<storage, read> do_z: array<u32>;
@group(0) @binding(43) var<storage, read_write> do_out: array<u32>;
@group(0) @binding(44) var<uniform> do_p: Q3doParams;

var<workgroup> do_red: array<f32, 128>;

@compute @workgroup_size(128)
fn q3w_delta_out(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let lane = lid.x;
    let dv = do_p.d_v;
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < dv) {
        v0 = bf16_decode(bf16_encode(do_core[h * dv + e0]));
        v1 = bf16_decode(bf16_encode(do_core[h * dv + e0 + 1u]));
    }
    do_red[lane] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            do_red[lane] = do_red[lane] + do_red[lane + s];
        }
        workgroupBarrier();
    }
    if (e0 >= dv) {
        return;
    }
    let rms = inverseSqrt(do_red[0] / f32(dv) + do_p.eps);
    let w0 = bf16_decode(u16_at(do_w[e0 >> 1u], e0));
    let w1 = bf16_decode(u16_at(do_w[(e0 + 1u) >> 1u], e0 + 1u));
    let zi = h * dv + e0;
    let zs = do_p.z_off + zi;
    let z0 = bf16_decode(u16_at(do_z[zs >> 1u], zs));
    let z1 = bf16_decode(u16_at(do_z[(zs + 1u) >> 1u], zs + 1u));
    let g0 = bf16_decode(bf16_encode(z0 / (1.0 + exp(-z0))));
    let g1 = bf16_decode(bf16_encode(z1 / (1.0 + exp(-z1))));
    let n0 = bf16_decode(bf16_encode(v0 * rms * w0));
    let n1 = bf16_decode(bf16_encode(v1 * rms * w1));
    do_out[zi >> 1u] = bf16_pack(n0 * g0, n1 * g1);
}
