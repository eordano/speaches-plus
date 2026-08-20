
fn qg_chunked_unscaled(row: u32, live: bool, lane: u32) -> f32 {
    let kv = select(0u, qg_params.k_elems >> 4u, live);
    let wbase = select(0u, row * (qg_params.k_elems >> 4u), live);
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + QG_LANES) {
        acc = acc + qg_dot16_e4m3(qg_w4[wbase + v], qg_x4[2u * v], qg_x4[2u * v + 1u]);
    }
    return acc;
}

fn qg_legacy_bits(row: u32, live: bool, lane: u32) -> u32 {
    var raw = 0.0;
    if (qg_params.pad1 == 2u) {
        raw = qg_chunked_unscaled(row, live, lane);
    } else {
        raw = qg_row_acc_e4m3(row, live, lane);
    }
    let total = qg_butterfly(raw);
    let sc = qg_row_scale[select(0u, row, live)];
    return bf16_encode(total * sc) & 0xffffu;
}

@compute @workgroup_size(128)
fn g4w_gemv_legacy_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let bits = qg_legacy_bits(row, live, lane);
    if (lane == 0u) {
        qg_pk_rowbits[sgid] = bits;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        qg_y[qg_pk_params.dst_word_off + (row >> 1u)] = qg_sg_word(sgid, row);
    }
}

@compute @workgroup_size(128)
fn g4w_gemv_legacy_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * qg_params.groups_x) * QG_SG_ROWS + sgid;
    let live = row < qg_params.n_rows;
    let bits = qg_legacy_bits(row, live, lane);
    if (lane == 0u) {
        qg_pk_rowbits[sgid] = bits;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        qg_scatter(row, qg_sg_word(sgid, row));
    }
}
