
@compute @workgroup_size(256)
fn g4w_flash_splitk_stage2_pk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let splits = 16u;
    let stride = hd + 2u;
    let base = h * splits * stride;

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
        fd_out[h * hw + w] = (bf16_encode(a0 * inv_l) & 0xffffu)
            | ((bf16_encode(a1 * inv_l) & 0xffffu) << 16u);
    }
}
