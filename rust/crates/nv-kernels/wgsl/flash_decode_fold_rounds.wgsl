
    let total = fd_params.total;
    let base = fd_params.start + split * FD_WARPS;
    let stride = fd_params.splits * FD_WARPS;
    var rounds = 0u;
    if (total > base) {
        rounds = (total - base + stride - 1u) / stride;
    }
    let use_vec4 = (hd & 3u) == 0u;

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = base + warp + r * stride;
        let live = p < total;
        var sp = p;
        if (fd_params.ring > 0u) {
            sp = p % fd_params.ring;
        }
