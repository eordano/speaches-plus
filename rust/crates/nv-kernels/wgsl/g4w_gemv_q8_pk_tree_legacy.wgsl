
fn qg_legacy_word(tid: u32, row: u32, total: f32) -> u32 {
    let lo = bf16_encode(total * qg_row_scale[row]) & 0xffffu;
    var hi = 0u;
    if (row + 1u < qg_params.n_rows) {
        hi = bf16_encode(qg_partial[tid + QG_LANES] * qg_row_scale[row + 1u]) & 0xffffu;
    }
    return lo | (hi << 16u);
}

@compute @workgroup_size(256)
fn g4w_gemv_legacy_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, qg_row_acc_e4m3(row, live, lane));
    if (lane == 0u && live && (warp & 1u) == 0u) {
        qg_y[qg_pk_params.dst_word_off + (row >> 1u)] = qg_legacy_word(tid, row, total);
    }
}

@compute @workgroup_size(256)
fn g4w_gemv_legacy_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, qg_row_acc_e4m3(row, live, lane));
    if (lane == 0u && live && (warp & 1u) == 0u) {
        qg_scatter(row, qg_legacy_word(tid, row, total));
    }
}
