
fn {P}_epilogue(lid: u32, lane: u32, warp: u32, hd: u32, slot: u32, m: f32, l: f32) {
    if (lane == 0u) {
        {P}_sm[warp] = m;
        {P}_sl[warp] = l;
    }
    workgroupBarrier();
    if (warp == 0u) {
        var m_blk = fd_neg_inf();
        for (var w = 0u; w < FD_WARPS; w = w + 1u) {
            m_blk = max(m_blk, {P}_sm[w]);
        }
        var l_blk = 0.0;
        for (var w = 0u; w < FD_WARPS; w = w + 1u) {
            if ({P}_sm[w] > fd_neg_inf()) {
                l_blk = l_blk + fd_round({P}_sl[w] * fd_exp({P}_sm[w] - m_blk));
            }
        }
        if (lane == 0u) {
            fd_scratch[slot] = m_blk;
            fd_scratch[slot + 1u] = l_blk;
        }
    }
    var m_blk = fd_neg_inf();
    for (var w = 0u; w < FD_WARPS; w = w + 1u) {
        m_blk = max(m_blk, {P}_sm[w]);
    }
    for (var d = lid; d < hd; d = d + FD_BLOCK) {
        var a = 0.0;
        for (var w = 0u; w < FD_WARPS; w = w + 1u) {
            if ({P}_sm[w] > fd_neg_inf()) {
                a = a + fd_round({P}_sacc[w * {HD}u + d] * fd_exp({P}_sm[w] - m_blk));
            }
        }
        fd_scratch[slot + 2u + d] = a;
    }
}
