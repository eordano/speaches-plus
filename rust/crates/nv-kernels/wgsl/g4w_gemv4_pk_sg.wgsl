
@group(0) @binding(6) var<storage, read_write> g4w_y4_pk: array<u32>;

var<workgroup> g4w_sg_rowbits: array<u32, 4>;

@compute @workgroup_size(128)
fn g4w_gemv_nvfp4_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * GEMV_SG_ROWS + sgid;
    let row_live = row < gemv_params.n_rows;
    let blocks = gemv_params.k_blocks;
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);
    let scale_row = select(0u, row, row_live);

    var warp_sums: array<f32, 8>;
    for (var w = 0u; w < GEMV_SG_WARPS; w = w + 1u) {
        var acc = 0.0;
        for (var kb = w * GEMV_SG_LANES + lane; kb < blocks; kb = kb + GEMV_WORKGROUP) {
            let ws_idx = nvfp4_scale_byte_index(scale_row, kb, gemv_params.k_tiles);
            let ws = byte_at(gemv_w_scales[ws_idx >> 2u], ws_idx);
            let xs = byte_at(gemv_x_scales[kb >> 2u], kb);
            let block_scale = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
            let wv = gemv_w_packed[w_vec_base + kb];
            let xv = gemv_x_packed[kb];
            var dot = 0.0;
            dot = gemv_dot8(wv.x, xv.x, dot);
            dot = gemv_dot8(wv.y, xv.y, dot);
            acc = fma(block_scale, dot, acc);
        }
        warp_sums[w] = gemv_sg_butterfly(acc);
    }

    let total = ((warp_sums[0] + warp_sums[4]) + (warp_sums[2] + warp_sums[6]))
        + ((warp_sums[1] + warp_sums[5]) + (warp_sums[3] + warp_sums[7]));
    let bits = bf16_encode(total * gemv_params.alpha) & 0xffffu;
    if (lane == 0u) {
        g4w_sg_rowbits[sgid] = bits;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && row_live) {
        var word = g4w_sg_rowbits[sgid];
        if (row + 1u < gemv_params.n_rows) {
            word = word | (g4w_sg_rowbits[sgid + 1u] << 16u);
        }
        g4w_y4_pk[row >> 1u] = word;
    }
}
