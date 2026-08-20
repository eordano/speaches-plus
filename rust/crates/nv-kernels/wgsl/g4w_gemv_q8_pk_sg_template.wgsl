
fn qg_sg_bits_TAG(row: u32, live: bool, lane: u32) -> u32 {
    return bf16_encode(qg_butterfly(ACC(row, live, lane))) & 0xffffu;
}

@compute @workgroup_size(128)
fn g4w_gemv_TAG_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let bits = qg_sg_bits_TAG(row, live, lane);
    if (lane == 0u) {
        qg_pk_rowbits[sgid] = bits;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        qg_y[qg_pk_params.dst_word_off + (row >> 1u)] = qg_sg_word(sgid, row);
    }
}

@compute @workgroup_size(128)
fn g4w_gemv_TAG_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let bits = qg_sg_bits_TAG(row, live, lane);
    if (lane == 0u) {
        qg_pk_rowbits[sgid] = bits;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        qg_scatter(row, qg_sg_word(sgid, row));
    }
}
