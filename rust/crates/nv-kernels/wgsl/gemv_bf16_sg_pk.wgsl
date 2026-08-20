
struct GemvSgPkOff {
    dst_word_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(30) var<uniform> sg_pk_off: GemvSgPkOff;

var<workgroup> sg_pk_tot: array<f32, 16>;

fn sg_pk_body(wid: vec3<u32>, sgid: u32, lane: u32, rows: u32) {
    let row = sg_row(wid, rows, sgid);
    let live = row < sg_params.n_rows;
    let kv = sg_params.k_elems >> 3u;
    let w_base = select(0u, row * (sg_params.w_row_words >> 2u), live);
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + 32u) {
        let ww = sg_w4[w_base + v];
        let xw = sg_x4[v];
        for (var j = 0u; j < 4u; j = j + 1u) {
            acc = acc + (bf16_lo(ww[j]) * bf16_lo(xw[j]) + bf16_hi(ww[j]) * bf16_hi(xw[j]));
        }
    }
    let total = sg_butterfly(acc);
    if (lane == 0u) {
        sg_pk_tot[sgid] = total;
    }
    workgroupBarrier();
    if (lane == 0u && live && (sgid & 1u) == 0u) {
        let lo = bf16_encode(total) & 0xffffu;
        let hi = bf16_encode(sg_pk_tot[sgid + 1u]) & 0xffffu;
        let hi_live = row + 1u < sg_params.n_rows;
        sg_y[sg_pk_off.dst_word_off + (row >> 1u)] = lo | (select(0u, hi, hi_live) << 16u);
    }
}

@compute @workgroup_size(128)
fn gemv_bf16_sg_v4_pk_wg128(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    sg_pk_body(wid, sgid, lane, 4u);
}

@compute @workgroup_size(256)
fn gemv_bf16_sg_v4_pk_wg256(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    sg_pk_body(wid, sgid, lane, 8u);
}
