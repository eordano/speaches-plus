
fn pfl_warp_sum_same_butterfly_order_as_fd_warp_sum(x: f32) -> f32 {
    var a = x;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

@compute @workgroup_size(256)
fn q3w_pf_flash1_fp8kv_mk_sg(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let nkv = fd_params.n_kv;
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let mr = fd_params.m_rows;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var t = lid; t < mr * hd; t = t + FD_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        fd_qsh_mk[t] = fd_q[(qi * fd_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    var acc: array<f32, 64>;
    var mreg: array<f32, 8>;
    var lreg: array<f32, 8>;
    for (var qi = 0u; qi < 8u; qi = qi + 1u) {
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[qi * FD_MK_MAX_ACC + i] = 0.0;
        }
        mreg[qi] = fd_neg_inf();
        lreg[qi] = 0.0;
    }

    let total = i32(fd_params.total);
    let start0 = fd_mk_start_of(total - i32(mr - 1u));
    let base = start0 + i32(split * FD_WARPS);
    let stride = i32(fd_params.splits * FD_WARPS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }
    let use_vec4 = (hd & 3u) == 0u;

    for (var r = 0; r < rounds; r = r + 1) {
        let p = base + i32(warp) + r * stride;
        let live = p < total;
        var sp = 0u;
        if (live) {
            sp = u32(p);
            if (fd_params.ring > 0u) {
                sp = sp % fd_params.ring;
            }
        }
        let kbase = (sp * nkv + kvh) * hd;
        var ks = 0.0;
        var vs = 0.0;
        if (live) {
            ks = fd_k_scales[sp * nkv + kvh];
            vs = fd_v_scales[sp * nkv + kvh];
        }
        var kd: array<f32, 8>;
        var vd: array<f32, 8>;
        if (live) {
            if (use_vec4) {
                let n4 = hd >> 2u;
                var slot4 = 0u;
                for (var j = lane; j < n4; j = j + FD_LANES) {
                    let kb = kbase + j * 4u;
                    kd[slot4 * 4u] = fd_k_fp8(kb);
                    kd[slot4 * 4u + 1u] = fd_k_fp8(kb + 1u);
                    kd[slot4 * 4u + 2u] = fd_k_fp8(kb + 2u);
                    kd[slot4 * 4u + 3u] = fd_k_fp8(kb + 3u);
                    slot4 = slot4 + 1u;
                }
            }
            for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                let d = lane + i * FD_LANES;
                if (d < hd) {
                    vd[i] = fd_v_fp8(kbase + d);
                }
            }
        }
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let tq = total - i32(mr - 1u - qi);
            let sq = fd_mk_start_of(tq);
            let act = live && p >= sq && p < tq;
            let qoff = qi * hd;
            var partial = 0.0;
            if (act) {
                if (use_vec4) {
                    let n4 = hd >> 2u;
                    var slot4 = 0u;
                    for (var j = lane; j < n4; j = j + FD_LANES) {
                        let qb = qoff + j * 4u;
                        var t = fd_qsh_mk[qb + 1u] * kd[slot4 * 4u + 1u];
                        t = fma(fd_qsh_mk[qb], kd[slot4 * 4u], t);
                        t = fma(fd_qsh_mk[qb + 2u], kd[slot4 * 4u + 2u], t);
                        t = fma(fd_qsh_mk[qb + 3u], kd[slot4 * 4u + 3u], t);
                        partial = partial + t;
                        slot4 = slot4 + 1u;
                    }
                } else {
                    for (var d = lane; d < hd; d = d + FD_LANES) {
                        partial = fma(fd_qsh_mk[qoff + d], fd_k_fp8(kbase + d), partial);
                    }
                }
            }
            let score =
                (pfl_warp_sum_same_butterfly_order_as_fd_warp_sum(partial) * ks) * fd_params.scaling;
            if (act) {
                let m_new = max(mreg[qi], score);
                let corr = fd_exp(mreg[qi] - m_new);
                let w = fd_exp(score - m_new);
                lreg[qi] = fma(lreg[qi], corr, w);
                let w_v = w * vs;
                for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                    let d = lane + i * FD_LANES;
                    if (d < hd) {
                        acc[qi * FD_MK_MAX_ACC + i] =
                            fma(w_v, vd[i], acc[qi * FD_MK_MAX_ACC + i] * corr);
                    }
                }
                mreg[qi] = m_new;
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        workgroupBarrier();
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            let d = lane + i * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc[qi * FD_MK_MAX_ACC + i];
            }
        }
        let slot = ((h * mr + qi) * fd_params.splits + split) * (hd + 2u);
        fd_stage1_epilogue(lid, lane, warp, hd, slot, mreg[qi], lreg[qi]);
    }
}
