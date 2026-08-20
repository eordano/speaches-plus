
@compute @workgroup_size(256)
fn g4w_flash_splitk_stage2_pk_mk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let qi = wg.y;
    if (h >= fd_params.n_heads || qi >= fd_params.m_rows) {
        return;
    }
    let hd = fd_params.head_dim;
    let splits = fd_params.splits;
    let stride = hd + 2u;
    let base = (h * fd_params.m_rows + qi) * splits * stride;

    var m_glob = fd_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) {
        m_glob = max(m_glob, fd_scratch[base + s * stride]);
    }
    var ssc: array<f32, 32>;
    var l_glob = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        let p0 = fd_scratch[base + s * stride];
        var sc = 0.0;
        if (p0 > fd_neg_inf()) {
            sc = fd_exp(p0 - m_glob);
        }
        ssc[s] = sc;
        l_glob = fma(fd_scratch[base + s * stride + 1u], sc, l_glob);
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
        for (var s = 0u; s < splits; s = s + 1u) {
            a0 = fma(fd_scratch[base + s * stride + 2u + d0], ssc[s], a0);
            a1 = fma(fd_scratch[base + s * stride + 2u + d0 + 1u], ssc[s], a1);
        }
        fd_out[(qi * fd_params.n_heads + h) * hw + w] = (bf16_encode(a0 * inv_l) & 0xffffu)
            | ((bf16_encode(a1 * inv_l) & 0xffffu) << 16u);
    }
}
