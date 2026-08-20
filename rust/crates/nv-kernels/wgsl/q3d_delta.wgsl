
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
    pad0: u32,
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
        q3w_gate_lane(wid.x);
    }
}

struct Q3gParams {
    n_v: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(20) var<storage, read> dg_ab: array<u32>;
@group(0) @binding(21) var<storage, read> dg_alog: array<f32>;
@group(0) @binding(22) var<storage, read> dg_dt: array<f32>;
@group(0) @binding(23) var<storage, read_write> dg_g: array<f32>;
@group(0) @binding(24) var<storage, read_write> dg_beta: array<f32>;
@group(0) @binding(25) var<uniform> dg_p: Q3gParams;

fn q3w_gate_values(i: u32) -> vec2<f32> {
    let a = bf16_decode(u16_at(dg_ab[i >> 1u], i));
    let j = dg_p.n_v + i;
    let b = bf16_decode(u16_at(dg_ab[j >> 1u], j));
    let beta = 1.0 / (1.0 + exp(-b));
    let t = a + dg_dt[i];
    let sp = max(t, 0.0) + log(1.0 + exp(-abs(t)));
    return vec2<f32>(exp(sp * (-exp(dg_alog[i]))), beta);
}

fn q3w_gate_lane(i: u32) {
    let gb = q3w_gate_values(i);
    dg_beta[i] = gb.y;
    dg_g[i] = gb.x;
}

@compute @workgroup_size(64)
fn q3w_delta_gating(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= dg_p.n_v) {
        return;
    }
    q3w_gate_lane(i);
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

struct Q3doParams {
    n_v: u32,
    d_v: u32,
    pad0: u32,
    eps: f32,
};

@group(0) @binding(40) var<storage, read> do_core: array<f32>;
@group(0) @binding(41) var<storage, read> do_w: array<u32>;
@group(0) @binding(42) var<storage, read> do_z: array<u32>;
@group(0) @binding(43) var<storage, read_write> do_out: array<u32>;
@group(0) @binding(44) var<uniform> do_p: Q3doParams;

var<workgroup> do_red: array<f32, 128>;

fn q3w_out_tail(h: u32, e0: u32, v0: f32, v1: f32, red0: f32) {
    let dv = do_p.d_v;
    let rms = inverseSqrt(red0 / f32(dv) + do_p.eps);
    let w0 = bf16_decode(u16_at(do_w[e0 >> 1u], e0));
    let w1 = bf16_decode(u16_at(do_w[(e0 + 1u) >> 1u], e0 + 1u));
    let zi = h * dv + e0;
    let z0 = bf16_decode(u16_at(do_z[zi >> 1u], zi));
    let z1 = bf16_decode(u16_at(do_z[(zi + 1u) >> 1u], zi + 1u));
    let g0 = bf16_decode(bf16_encode(z0 / (1.0 + exp(-z0))));
    let g1 = bf16_decode(bf16_encode(z1 / (1.0 + exp(-z1))));
    let n0 = bf16_decode(bf16_encode(v0 * rms * w0));
    let n1 = bf16_decode(bf16_encode(v1 * rms * w1));
    do_out[zi >> 1u] = bf16_pack(n0 * g0, n1 * g1);
}

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
    q3w_out_tail(h, e0, v0, v1, do_red[0]);
}

var<workgroup> fh_q: array<f32, 128>;
var<workgroup> fh_k: array<f32, 128>;
var<workgroup> fh_core: array<f32, 128>;
var<workgroup> fh_g: f32;
var<workgroup> fh_beta: f32;

@compute @workgroup_size(128)
fn q3w_delta_head_fused(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let d = lid.x;
    let kh = h / dq_p.v_per_k;
    var qv = 0.0;
    var kv = 0.0;
    if (d < dq_p.d_k) {
        qv = dq_mixed[kh * dq_p.d_k + d];
        kv = dq_mixed[dq_p.key_dim + kh * dq_p.d_k + d];
    }
    dq_rq[d] = qv * qv;
    dq_rk[d] = kv * kv;
    if (d == 0u) {
        let gb = q3w_gate_values(h);
        fh_g = gb.x;
        fh_beta = gb.y;
    }
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
        fh_q[d] = (qv / nq) * dq_p.scale;
        fh_k[d] = kv / nk;
    }
    workgroupBarrier();
    let dk = dq_p.d_k;
    let dv = dq_p.d_v;
    var outv = 0.0;
    if (d < dv) {
        let vv = dq_mixed[2u * dq_p.key_dim + h * dv + d];
        let ge = fh_g;
        let bt = fh_beta;
        let sbase = h * dk * dv + d;
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
            kv_mem = fma(s0, fh_k[i], kv_mem);
            kv_mem = fma(s1, fh_k[i + 1u], kv_mem);
            kv_mem = fma(s2, fh_k[i + 2u], kv_mem);
            kv_mem = fma(s3, fh_k[i + 3u], kv_mem);
            kv_mem = fma(s4, fh_k[i + 4u], kv_mem);
            kv_mem = fma(s5, fh_k[i + 5u], kv_mem);
            kv_mem = fma(s6, fh_k[i + 6u], kv_mem);
            kv_mem = fma(s7, fh_k[i + 7u], kv_mem);
            i = i + 8u;
        }
        loop {
            if (i >= dk) {
                break;
            }
            kv_mem = fma(dr_state[sbase + i * dv] * ge, fh_k[i], kv_mem);
            i = i + 1u;
        }
        let delta = (vv - kv_mem) * bt;
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
            let u0 = fma(fh_k[j], delta, t0);
            let u1 = fma(fh_k[j + 1u], delta, t1);
            let u2 = fma(fh_k[j + 2u], delta, t2);
            let u3 = fma(fh_k[j + 3u], delta, t3);
            let u4 = fma(fh_k[j + 4u], delta, t4);
            let u5 = fma(fh_k[j + 5u], delta, t5);
            let u6 = fma(fh_k[j + 6u], delta, t6);
            let u7 = fma(fh_k[j + 7u], delta, t7);
            dr_state[p] = u0;
            dr_state[p + dv] = u1;
            dr_state[p + dv2] = u2;
            dr_state[p + dv3] = u3;
            dr_state[p + dv4] = u4;
            dr_state[p + dv5] = u5;
            dr_state[p + dv6] = u6;
            dr_state[p + dv7] = u7;
            outv = fma(u0, fh_q[j], outv);
            outv = fma(u1, fh_q[j + 1u], outv);
            outv = fma(u2, fh_q[j + 2u], outv);
            outv = fma(u3, fh_q[j + 3u], outv);
            outv = fma(u4, fh_q[j + 4u], outv);
            outv = fma(u5, fh_q[j + 5u], outv);
            outv = fma(u6, fh_q[j + 6u], outv);
            outv = fma(u7, fh_q[j + 7u], outv);
            j = j + 8u;
        }
        loop {
            if (j >= dk) {
                break;
            }
            let p = sbase + j * dv;
            let u = fma(fh_k[j], delta, dr_state[p] * ge);
            dr_state[p] = u;
            outv = fma(u, fh_q[j], outv);
            j = j + 1u;
        }
    }
    fh_core[d] = outv;
    workgroupBarrier();
    let e0 = 2u * d;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < dv) {
        v0 = bf16_decode(bf16_encode(fh_core[e0]));
        v1 = bf16_decode(bf16_encode(fh_core[e0 + 1u]));
    }
    do_red[d] = v0 * v0 + v1 * v1;
    workgroupBarrier();
    for (var s = 64u; s > 0u; s = s >> 1u) {
        if (d < s) {
            do_red[d] = do_red[d] + do_red[d + s];
        }
        workgroupBarrier();
    }
    if (e0 >= dv) {
        return;
    }
    q3w_out_tail(h, e0, v0, v1, do_red[0]);
}
