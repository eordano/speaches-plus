
var<workgroup> qg_pk_rowbits: array<u32, 4>;

fn qg_sg_word(sgid: u32, row: u32) -> u32 {
    var word = qg_pk_rowbits[sgid];
    if (row + 1u < qg_params.n_rows) {
        word = word | (qg_pk_rowbits[sgid + 1u] << 16u);
    }
    return word;
}
