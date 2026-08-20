
fn g4w_win_start(tq: u32) -> u32 {
    let win = fd_params.window;
    if (win > 0u && tq > win) {
        return tq - win;
    }
    return 0u;
}

@compute @workgroup_size(256)
fn g4w_flash_rows_stage1_fp8(
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
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;
    let mr = fd_params.m_rows;
    let use_vec4 = (hd & 3u) == 0u;

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        workgroupBarrier();
        for (var d = lid; d < hd; d = d + FD_BLOCK) {
            fd_qsh[d] = fd_q[(qi * fd_params.n_heads + h) * hd + d];
        }
        workgroupBarrier();

        var acc0 = 0.0;
        var acc1 = 0.0;
        var acc2 = 0.0;
        var acc3 = 0.0;
        var acc4 = 0.0;
        var acc5 = 0.0;
        var acc6 = 0.0;
        var acc7 = 0.0;
        var acc8 = 0.0;
        var acc9 = 0.0;
        var acc10 = 0.0;
        var acc11 = 0.0;
        var acc12 = 0.0;
        var acc13 = 0.0;
        var acc14 = 0.0;
        var acc15 = 0.0;
        var m = fd_neg_inf();
        var l = 0.0;

        let total = fd_params.total - (mr - 1u - qi);
        let base = g4w_win_start(total) + split * FD_WARPS;
        let stride = fd_params.splits * FD_WARPS;
        var rounds = 0u;
        if (total > base) {
            rounds = (total - base + stride - 1u) / stride;
        }

        for (var r = 0u; r < rounds; r = r + 1u) {
            let p = base + warp + r * stride;
            let live = p < total;
            var sp = p;
            if (fd_params.ring > 0u) {
                sp = p % fd_params.ring;
            }
            var partial = 0.0;
            var ks = 0.0;
            if (live) {
                let kbase = (sp * nkv + kvh) * hd;
                ks = fd_k_scales[sp * nkv + kvh];
                if (use_vec4) {
                    let n4 = hd >> 2u;
                    for (var j = lane; j < n4; j = j + FD_LANES) {
                        let qb = j * 4u;
                        let kb = kbase + qb;
                        let f0 = fd_k_fp8(kb);
                        let f1 = fd_k_fp8(kb + 1u);
                        let f2 = fd_k_fp8(kb + 2u);
                        let f3 = fd_k_fp8(kb + 3u);
                        var t = fd_qsh[qb + 1u] * f1;
                        t = fma(fd_qsh[qb], f0, t);
                        t = fma(fd_qsh[qb + 2u], f2, t);
                        t = fma(fd_qsh[qb + 3u], f3, t);
                        partial = partial + t;
                    }
                } else {
                    for (var d = lane; d < hd; d = d + FD_LANES) {
                        partial = fma(fd_qsh[d], fd_k_fp8(kbase + d), partial);
                    }
                }
            }
            let score = (fd_warp_sum(lid, partial) * ks) * fd_params.scaling;
            if (live) {
                let m_new = max(m, score);
                let corr = fd_exp(m - m_new);
                let w = fd_exp(score - m_new);
                l = fma(l, corr, w);
                let vbase = (sp * nkv + kvh) * hd;
                let w_v = w * fd_v_scales[sp * nkv + kvh];
                {
                    let d = lane + 0u * FD_LANES;
                    if (d < hd) {
                        acc0 = fma(w_v, fd_v_fp8(vbase + d), acc0 * corr);
                    }
                }
                {
                    let d = lane + 1u * FD_LANES;
                    if (d < hd) {
                        acc1 = fma(w_v, fd_v_fp8(vbase + d), acc1 * corr);
                    }
                }
                {
                    let d = lane + 2u * FD_LANES;
                    if (d < hd) {
                        acc2 = fma(w_v, fd_v_fp8(vbase + d), acc2 * corr);
                    }
                }
                {
                    let d = lane + 3u * FD_LANES;
                    if (d < hd) {
                        acc3 = fma(w_v, fd_v_fp8(vbase + d), acc3 * corr);
                    }
                }
                {
                    let d = lane + 4u * FD_LANES;
                    if (d < hd) {
                        acc4 = fma(w_v, fd_v_fp8(vbase + d), acc4 * corr);
                    }
                }
                {
                    let d = lane + 5u * FD_LANES;
                    if (d < hd) {
                        acc5 = fma(w_v, fd_v_fp8(vbase + d), acc5 * corr);
                    }
                }
                {
                    let d = lane + 6u * FD_LANES;
                    if (d < hd) {
                        acc6 = fma(w_v, fd_v_fp8(vbase + d), acc6 * corr);
                    }
                }
                {
                    let d = lane + 7u * FD_LANES;
                    if (d < hd) {
                        acc7 = fma(w_v, fd_v_fp8(vbase + d), acc7 * corr);
                    }
                }
                {
                    let d = lane + 8u * FD_LANES;
                    if (d < hd) {
                        acc8 = fma(w_v, fd_v_fp8(vbase + d), acc8 * corr);
                    }
                }
                {
                    let d = lane + 9u * FD_LANES;
                    if (d < hd) {
                        acc9 = fma(w_v, fd_v_fp8(vbase + d), acc9 * corr);
                    }
                }
                {
                    let d = lane + 10u * FD_LANES;
                    if (d < hd) {
                        acc10 = fma(w_v, fd_v_fp8(vbase + d), acc10 * corr);
                    }
                }
                {
                    let d = lane + 11u * FD_LANES;
                    if (d < hd) {
                        acc11 = fma(w_v, fd_v_fp8(vbase + d), acc11 * corr);
                    }
                }
                {
                    let d = lane + 12u * FD_LANES;
                    if (d < hd) {
                        acc12 = fma(w_v, fd_v_fp8(vbase + d), acc12 * corr);
                    }
                }
                {
                    let d = lane + 13u * FD_LANES;
                    if (d < hd) {
                        acc13 = fma(w_v, fd_v_fp8(vbase + d), acc13 * corr);
                    }
                }
                {
                    let d = lane + 14u * FD_LANES;
                    if (d < hd) {
                        acc14 = fma(w_v, fd_v_fp8(vbase + d), acc14 * corr);
                    }
                }
                {
                    let d = lane + 15u * FD_LANES;
                    if (d < hd) {
                        acc15 = fma(w_v, fd_v_fp8(vbase + d), acc15 * corr);
                    }
                }
                m = m_new;
            }
        }

        {
            let d = lane + 0u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc0;
            }
        }
        {
            let d = lane + 1u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc1;
            }
        }
        {
            let d = lane + 2u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc2;
            }
        }
        {
            let d = lane + 3u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc3;
            }
        }
        {
            let d = lane + 4u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc4;
            }
        }
        {
            let d = lane + 5u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc5;
            }
        }
        {
            let d = lane + 6u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc6;
            }
        }
        {
            let d = lane + 7u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc7;
            }
        }
        {
            let d = lane + 8u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc8;
            }
        }
        {
            let d = lane + 9u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc9;
            }
        }
        {
            let d = lane + 10u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc10;
            }
        }
        {
            let d = lane + 11u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc11;
            }
        }
        {
            let d = lane + 12u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc12;
            }
        }
        {
            let d = lane + 13u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc13;
            }
        }
        {
            let d = lane + 14u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc14;
            }
        }
        {
            let d = lane + 15u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc15;
            }
        }
        let slot = ((h * mr + qi) * fd_params.splits + split) * (hd + 2u);
        fd_stage1_epilogue(lid, lane, warp, hd, slot, m, l);
    }
}

@compute @workgroup_size(256)
fn g4w_flash_rows_stage2_pk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let qi = wg.y;
    if (h >= fd_params.n_heads || qi >= fd_params.m_rows) {
        return;
    }
    let hd = fd_params.head_dim;
    let splits = 16u;
    let stride = hd + 2u;
    let base = (h * fd_params.m_rows + qi) * splits * stride;

    var m_glob = fd_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) {
        m_glob = max(m_glob, fd_scratch[base + s * stride]);
    }
    var ssc0 = 0.0;
    var ssc1 = 0.0;
    var ssc2 = 0.0;
    var ssc3 = 0.0;
    var ssc4 = 0.0;
    var ssc5 = 0.0;
    var ssc6 = 0.0;
    var ssc7 = 0.0;
    var ssc8 = 0.0;
    var ssc9 = 0.0;
    var ssc10 = 0.0;
    var ssc11 = 0.0;
    var ssc12 = 0.0;
    var ssc13 = 0.0;
    var ssc14 = 0.0;
    var ssc15 = 0.0;
    var l_glob = 0.0;
    {
        let p0 = fd_scratch[base + 0u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc0 = sc;
        l_glob = fma(fd_scratch[base + 0u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 1u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc1 = sc;
        l_glob = fma(fd_scratch[base + 1u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 2u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc2 = sc;
        l_glob = fma(fd_scratch[base + 2u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 3u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc3 = sc;
        l_glob = fma(fd_scratch[base + 3u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 4u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc4 = sc;
        l_glob = fma(fd_scratch[base + 4u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 5u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc5 = sc;
        l_glob = fma(fd_scratch[base + 5u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 6u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc6 = sc;
        l_glob = fma(fd_scratch[base + 6u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 7u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc7 = sc;
        l_glob = fma(fd_scratch[base + 7u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 8u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc8 = sc;
        l_glob = fma(fd_scratch[base + 8u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 9u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc9 = sc;
        l_glob = fma(fd_scratch[base + 9u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 10u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc10 = sc;
        l_glob = fma(fd_scratch[base + 10u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 11u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc11 = sc;
        l_glob = fma(fd_scratch[base + 11u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 12u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc12 = sc;
        l_glob = fma(fd_scratch[base + 12u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 13u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc13 = sc;
        l_glob = fma(fd_scratch[base + 13u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 14u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc14 = sc;
        l_glob = fma(fd_scratch[base + 14u * stride + 1u], sc, l_glob);
    }
    {
        let p0 = fd_scratch[base + 15u * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc15 = sc;
        l_glob = fma(fd_scratch[base + 15u * stride + 1u], sc, l_glob);
    }
    var inv_l = 0.0;
    if (l_glob > 0.0) {
        inv_l = fd_recip(l_glob);
    }
    let hw = hd >> 1u;
    for (var w = tid.x; w < hw; w = w + FD_BLOCK) {
        let d0 = w * 2u;
        var a0 = 0.0;
        var a1 = 0.0;
        a0 = fma(fd_scratch[base + 0u * stride + 2u + d0], ssc0, a0);
        a0 = fma(fd_scratch[base + 1u * stride + 2u + d0], ssc1, a0);
        a0 = fma(fd_scratch[base + 2u * stride + 2u + d0], ssc2, a0);
        a0 = fma(fd_scratch[base + 3u * stride + 2u + d0], ssc3, a0);
        a0 = fma(fd_scratch[base + 4u * stride + 2u + d0], ssc4, a0);
        a0 = fma(fd_scratch[base + 5u * stride + 2u + d0], ssc5, a0);
        a0 = fma(fd_scratch[base + 6u * stride + 2u + d0], ssc6, a0);
        a0 = fma(fd_scratch[base + 7u * stride + 2u + d0], ssc7, a0);
        a0 = fma(fd_scratch[base + 8u * stride + 2u + d0], ssc8, a0);
        a0 = fma(fd_scratch[base + 9u * stride + 2u + d0], ssc9, a0);
        a0 = fma(fd_scratch[base + 10u * stride + 2u + d0], ssc10, a0);
        a0 = fma(fd_scratch[base + 11u * stride + 2u + d0], ssc11, a0);
        a0 = fma(fd_scratch[base + 12u * stride + 2u + d0], ssc12, a0);
        a0 = fma(fd_scratch[base + 13u * stride + 2u + d0], ssc13, a0);
        a0 = fma(fd_scratch[base + 14u * stride + 2u + d0], ssc14, a0);
        a0 = fma(fd_scratch[base + 15u * stride + 2u + d0], ssc15, a0);
        a1 = fma(fd_scratch[base + 0u * stride + 2u + d0 + 1u], ssc0, a1);
        a1 = fma(fd_scratch[base + 1u * stride + 2u + d0 + 1u], ssc1, a1);
        a1 = fma(fd_scratch[base + 2u * stride + 2u + d0 + 1u], ssc2, a1);
        a1 = fma(fd_scratch[base + 3u * stride + 2u + d0 + 1u], ssc3, a1);
        a1 = fma(fd_scratch[base + 4u * stride + 2u + d0 + 1u], ssc4, a1);
        a1 = fma(fd_scratch[base + 5u * stride + 2u + d0 + 1u], ssc5, a1);
        a1 = fma(fd_scratch[base + 6u * stride + 2u + d0 + 1u], ssc6, a1);
        a1 = fma(fd_scratch[base + 7u * stride + 2u + d0 + 1u], ssc7, a1);
        a1 = fma(fd_scratch[base + 8u * stride + 2u + d0 + 1u], ssc8, a1);
        a1 = fma(fd_scratch[base + 9u * stride + 2u + d0 + 1u], ssc9, a1);
        a1 = fma(fd_scratch[base + 10u * stride + 2u + d0 + 1u], ssc10, a1);
        a1 = fma(fd_scratch[base + 11u * stride + 2u + d0 + 1u], ssc11, a1);
        a1 = fma(fd_scratch[base + 12u * stride + 2u + d0 + 1u], ssc12, a1);
        a1 = fma(fd_scratch[base + 13u * stride + 2u + d0 + 1u], ssc13, a1);
        a1 = fma(fd_scratch[base + 14u * stride + 2u + d0 + 1u], ssc14, a1);
        a1 = fma(fd_scratch[base + 15u * stride + 2u + d0 + 1u], ssc15, a1);
        fd_out[(qi * fd_params.n_heads + h) * hw + w] = (bf16_encode(a0 * inv_l) & 0xffffu)
            | ((bf16_encode(a1 * inv_l) & 0xffffu) << 16u);
    }
}
