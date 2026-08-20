
@compute @workgroup_size(256)
fn g4w_gemv_TAG_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, ACC(row, live, lane));
    if (lane == 0u && live && (warp & 1u) == 0u) {
        qg_y[qg_pk_params.dst_word_off + (row >> 1u)] = qg_pk_word(tid, row, total);
    }
}

@compute @workgroup_size(256)
fn g4w_gemv_TAG_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (QG_LANES - 1u);
    let warp = tid / QG_LANES;
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;
    let live = row < qg_params.n_rows;
    let total = qg_reduce(tid, lane, ACC(row, live, lane));
    if (lane == 0u && live && (warp & 1u) == 0u) {
        qg_scatter(row, qg_pk_word(tid, row, total));
    }
}
