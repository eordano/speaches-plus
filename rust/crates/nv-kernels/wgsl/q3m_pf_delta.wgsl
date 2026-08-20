
struct Q3pdParams {
    tokens: u32,
    in_stride_elems: u32,
    mixed_stride_elems: u32,
    qk_stride_elems: u32,
    v_stride_elems: u32,
    gb_stride_elems: u32,
    core_stride_elems: u32,
    gated_stride_elems: u32,
};

@group(0) @binding(50) var<uniform> pd_p: Q3pdParams;

const Q3PD_CONV_WIN_MAX: u32 = 8u;

@compute @workgroup_size(64)
fn q3w_pf_delta_conv_chunk(@builtin(global_invocation_id) gid: vec3<u32>) {
    let c = gid.x;
    if (c >= cv_p.conv_dim) {
        return;
    }
    let ks = cv_p.kernel;
    let hist = ks - 1u;
    var win: array<f32, Q3PD_CONV_WIN_MAX>;
    for (var j = 0u; j < hist; j = j + 1u) {
        win[j] = cv_state[c * hist + j];
    }
    for (var t = 0u; t < pd_p.tokens; t = t + 1u) {
        let xi = t * pd_p.in_stride_elems + c;
        let xv = bf16_decode(u16_at(cv_x[xi >> 1u], xi));
        var acc = 0.0;
        for (var j = 0u; j < hist; j = j + 1u) {
            acc = fma(cv_w[c * ks + j], win[j], acc);
        }
        acc = fma(cv_w[c * ks + hist], xv, acc);
        for (var j = 0u; j + 1u < hist; j = j + 1u) {
            win[j] = win[j + 1u];
        }
        if (hist > 0u) {
            win[hist - 1u] = xv;
        }
        let silu = acc / (1.0 + exp(-acc));
        cv_out[t * pd_p.mixed_stride_elems + c] = bf16_decode(bf16_encode(silu));
    }
    for (var j = 0u; j < hist; j = j + 1u) {
        cv_state[c * hist + j] = win[j];
    }
}

@compute @workgroup_size(128)
fn q3w_pf_delta_split_gated_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    let h = wid.x;
    let d = lid.x;
    let mb = t * pd_p.mixed_stride_elems;
    let kh = h / dq_p.v_per_k;
    var qv = 0.0;
    var kv = 0.0;
    if (d < dq_p.d_k) {
        qv = dq_mixed[mb + kh * dq_p.d_k + d];
        kv = dq_mixed[mb + dq_p.key_dim + kh * dq_p.d_k + d];
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
        dq_q[t * pd_p.qk_stride_elems + h * dq_p.d_k + d] = (qv / nq) * dq_p.scale;
        dq_k[t * pd_p.qk_stride_elems + h * dq_p.d_k + d] = kv / nk;
    }
    if (d < dq_p.d_v) {
        dq_v[t * pd_p.v_stride_elems + h * dq_p.d_v + d] =
            dq_mixed[mb + 2u * dq_p.key_dim + h * dq_p.d_v + d];
    }
    if (d == 0u) {
        let ib = t * pd_p.in_stride_elems + dq_p.ab_off;
        let ii = ib + h;
        let a = bf16_decode(u16_at(dg_ab[ii >> 1u], ii));
        let j = ib + dq_p.n_v + h;
        let b = bf16_decode(u16_at(dg_ab[j >> 1u], j));
        dg_beta[t * pd_p.gb_stride_elems + h] = 1.0 / (1.0 + exp(-b));
        let tt = a + dg_alogdt[dq_p.n_v + h];
        let sp = max(tt, 0.0) + log(1.0 + exp(-abs(tt)));
        dg_g[t * pd_p.gb_stride_elems + h] = exp(sp * (-exp(dg_alogdt[h])));
    }
}

@compute @workgroup_size(32)
fn q3w_pf_delta_recurrent_chunk(
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
    let sbase = h * dk * dv + lane;
    var st: array<f32, Q3PD_DK>;
    for (var i = 0u; i < dk; i = i + 1u) {
        st[i] = dr_state[sbase + i * dv];
    }
    for (var t = 0u; t < pd_p.tokens; t = t + 1u) {
        let ge = dr_g[t * pd_p.gb_stride_elems + h];
        let bt = dr_beta[t * pd_p.gb_stride_elems + h];
        let kbase = t * pd_p.qk_stride_elems + h * dk;
        var kv_mem = 0.0;
        var i = 0u;
        loop {
            if (i + 16u > dk) {
                break;
            }
            let s0 = st[i] * ge;
            let s1 = st[i + 1u] * ge;
            let s2 = st[i + 2u] * ge;
            let s3 = st[i + 3u] * ge;
            let s4 = st[i + 4u] * ge;
            let s5 = st[i + 5u] * ge;
            let s6 = st[i + 6u] * ge;
            let s7 = st[i + 7u] * ge;
            let s8 = st[i + 8u] * ge;
            let s9 = st[i + 9u] * ge;
            let s10 = st[i + 10u] * ge;
            let s11 = st[i + 11u] * ge;
            let s12 = st[i + 12u] * ge;
            let s13 = st[i + 13u] * ge;
            let s14 = st[i + 14u] * ge;
            let s15 = st[i + 15u] * ge;
            st[i] = s0;
            st[i + 1u] = s1;
            st[i + 2u] = s2;
            st[i + 3u] = s3;
            st[i + 4u] = s4;
            st[i + 5u] = s5;
            st[i + 6u] = s6;
            st[i + 7u] = s7;
            st[i + 8u] = s8;
            st[i + 9u] = s9;
            st[i + 10u] = s10;
            st[i + 11u] = s11;
            st[i + 12u] = s12;
            st[i + 13u] = s13;
            st[i + 14u] = s14;
            st[i + 15u] = s15;
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
            let s = st[i] * ge;
            st[i] = s;
            kv_mem = fma(s, dr_k[kbase + i], kv_mem);
            i = i + 1u;
        }
        let delta = (dr_v[t * pd_p.v_stride_elems + h * dv + lane] - kv_mem) * bt;
        var outv = 0.0;
        var j = 0u;
        loop {
            if (j + 16u > dk) {
                break;
            }
            let u0 = fma(dr_k[kbase + j], delta, st[j]);
            let u1 = fma(dr_k[kbase + j + 1u], delta, st[j + 1u]);
            let u2 = fma(dr_k[kbase + j + 2u], delta, st[j + 2u]);
            let u3 = fma(dr_k[kbase + j + 3u], delta, st[j + 3u]);
            let u4 = fma(dr_k[kbase + j + 4u], delta, st[j + 4u]);
            let u5 = fma(dr_k[kbase + j + 5u], delta, st[j + 5u]);
            let u6 = fma(dr_k[kbase + j + 6u], delta, st[j + 6u]);
            let u7 = fma(dr_k[kbase + j + 7u], delta, st[j + 7u]);
            let u8 = fma(dr_k[kbase + j + 8u], delta, st[j + 8u]);
            let u9 = fma(dr_k[kbase + j + 9u], delta, st[j + 9u]);
            let u10 = fma(dr_k[kbase + j + 10u], delta, st[j + 10u]);
            let u11 = fma(dr_k[kbase + j + 11u], delta, st[j + 11u]);
            let u12 = fma(dr_k[kbase + j + 12u], delta, st[j + 12u]);
            let u13 = fma(dr_k[kbase + j + 13u], delta, st[j + 13u]);
            let u14 = fma(dr_k[kbase + j + 14u], delta, st[j + 14u]);
            let u15 = fma(dr_k[kbase + j + 15u], delta, st[j + 15u]);
            st[j] = u0;
            st[j + 1u] = u1;
            st[j + 2u] = u2;
            st[j + 3u] = u3;
            st[j + 4u] = u4;
            st[j + 5u] = u5;
            st[j + 6u] = u6;
            st[j + 7u] = u7;
            st[j + 8u] = u8;
            st[j + 9u] = u9;
            st[j + 10u] = u10;
            st[j + 11u] = u11;
            st[j + 12u] = u12;
            st[j + 13u] = u13;
            st[j + 14u] = u14;
            st[j + 15u] = u15;
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
            let s = fma(dr_k[kbase + j], delta, st[j]);
            st[j] = s;
            outv = fma(s, dr_q[kbase + j], outv);
            j = j + 1u;
        }
        dr_out[t * pd_p.core_stride_elems + h * dv + lane] = outv;
    }
    for (var i = 0u; i < dk; i = i + 1u) {
        dr_state[sbase + i * dv] = st[i];
    }
}

@compute @workgroup_size(128)
fn q3w_pf_delta_out_m(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    let h = wid.x;
    let lane = lid.x;
    let dv = do_p.d_v;
    let cb = t * pd_p.core_stride_elems;
    let e0 = 2u * lane;
    var v0 = 0.0;
    var v1 = 0.0;
    if (e0 < dv) {
        v0 = bf16_decode(bf16_encode(do_core[cb + h * dv + e0]));
        v1 = bf16_decode(bf16_encode(do_core[cb + h * dv + e0 + 1u]));
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
    let zs = t * pd_p.in_stride_elems + do_p.z_off + zi;
    let z0 = bf16_decode(u16_at(do_z[zs >> 1u], zs));
    let z1 = bf16_decode(u16_at(do_z[(zs + 1u) >> 1u], zs + 1u));
    let g0 = bf16_decode(bf16_encode(z0 / (1.0 + exp(-z0))));
    let g1 = bf16_decode(bf16_encode(z1 / (1.0 + exp(-z1))));
    let n0 = bf16_decode(bf16_encode(v0 * rms * w0));
    let n1 = bf16_decode(bf16_encode(v1 * rms * w1));
    let ow = (t * pd_p.gated_stride_elems + zi) >> 1u;
    do_out[ow] = bf16_pack(n0 * g0, n1 * g1);
}
