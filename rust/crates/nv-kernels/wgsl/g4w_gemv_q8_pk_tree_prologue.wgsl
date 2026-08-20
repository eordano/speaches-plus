
fn qg_pk_word(tid: u32, row: u32, total: f32) -> u32 {
    let lo = bf16_encode(total) & 0xffffu;
    var hi = 0u;
    if (row + 1u < qg_params.n_rows) {
        hi = bf16_encode(qg_partial[tid + QG_LANES]) & 0xffffu;
    }
    return lo | (hi << 16u);
}
