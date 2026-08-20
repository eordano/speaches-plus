
@group(0) @binding(6) var<storage, read_write> g4w_y4_pk: array<atomic<u32>>;

@compute @workgroup_size(256)
fn g4w_gemv_nvfp4_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.x + wid.y * gemv_params.groups_x;
    let tid = lid.x;
    let row_live = row < gemv_params.n_rows;
    let blocks = select(0u, gemv_params.k_blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);

    var acc = 0.0;
    for (var kb = tid; kb < blocks; kb = kb + GEMV_WORKGROUP) {
        let ws_idx = nvfp4_scale_byte_index(row, kb, gemv_params.k_tiles);
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

    gemv_partial[tid] = acc;
    workgroupBarrier();
    for (var step = 0u; step < 8u; step = step + 1u) {
        let stride = GEMV_SHUFFLE_ORDER[step];
        let taking = (step < 5u) || ((tid & 31u) == 0u);
        if (taking && (tid & stride) == 0u) {
            gemv_partial[tid] = gemv_partial[tid] + gemv_partial[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u && row_live) {
        let bits = bf16_encode(gemv_partial[0] * gemv_params.alpha) & 0xffffu;
        let sh = (row & 1u) << 4u;
        atomicAnd(&g4w_y4_pk[row >> 1u], ~(0xffffu << sh));
        atomicOr(&g4w_y4_pk[row >> 1u], bits << sh);
    }
}
