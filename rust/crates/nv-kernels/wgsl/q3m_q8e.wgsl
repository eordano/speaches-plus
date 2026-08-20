
struct Q8eParams {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    x_slot_stride_elems: u32,
    w_e_stride_words: u32,
    y_slot_stride_words: u32,
    use_sel: u32,
    groups_per_row: u32,
    group_shift: u32,
    y_off_words: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> q8e_w: array<u32>;
@group(0) @binding(1) var<storage, read> q8e_s: array<f32>;
@group(0) @binding(2) var<storage, read> q8e_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> q8e_y: array<u32>;
@group(0) @binding(4) var<uniform> q8e_p: Q8eParams;
@group(0) @binding(5) var<storage, read> q8e_sel: array<u32>;

var<workgroup> q8e_pk_bits: array<u32, 8>;

@compute @workgroup_size(256)
fn q3w_gemv_i8g_experts(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    var e = slot;
    if (q8e_p.use_sel == 1u) {
        e = q8e_sel[slot];
    }
    let row = (wid.x + wid.y * q8e_p.groups_x) * 8u + sgid;
    let live = row < q8e_p.n_rows;
    let words = select(0u, q8e_p.k_elems >> 2u, live);
    let wbase = select(0u, e * q8e_p.w_e_stride_words + row * (q8e_p.k_elems >> 2u), live);
    let sbase = select(0u, (e * q8e_p.n_rows + row) * q8e_p.groups_per_row, live);
    let gshift = q8e_p.group_shift;
    let xw_base = (slot * q8e_p.x_slot_stride_elems) >> 1u;
    var acc = 0.0;
    for (var i = lane; i < words; i = i + 32u) {
        let w = q8e_w[wbase + i];
        let x0 = q8e_x[xw_base + 2u * i];
        let x1 = q8e_x[xw_base + 2u * i + 1u];
        var d = 0.0;
        d = fma(int8_decode(w, 0u), bf16_lo(x0), d);
        d = fma(int8_decode(w, 1u), bf16_hi(x0), d);
        d = fma(int8_decode(w, 2u), bf16_lo(x1), d);
        d = fma(int8_decode(w, 3u), bf16_hi(x1), d);
        acc = fma(q8e_s[sbase + (i >> gshift)], d, acc);
    }
    acc = acc + subgroupShuffleXor(acc, 16u);
    acc = acc + subgroupShuffleXor(acc, 8u);
    acc = acc + subgroupShuffleXor(acc, 4u);
    acc = acc + subgroupShuffleXor(acc, 2u);
    acc = acc + subgroupShuffleXor(acc, 1u);
    if (lane == 0u) {
        q8e_pk_bits[sgid] = bf16_encode(acc) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = q8e_pk_bits[sgid];
        if (row + 1u < q8e_p.n_rows) {
            word = word | (q8e_pk_bits[sgid + 1u] << 16u);
        }
        q8e_y[slot * q8e_p.y_slot_stride_words + (row >> 1u)] = word;
    }
}

@compute @workgroup_size(256)
fn q3w_gemv_i8_experts(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    var e = slot;
    if (q8e_p.use_sel == 1u) {
        e = q8e_sel[slot];
    }
    let row = (wid.x + wid.y * q8e_p.groups_x) * 8u + sgid;
    let live = row < q8e_p.n_rows;
    let words = select(0u, q8e_p.k_elems >> 2u, live);
    let wbase = select(0u, e * q8e_p.w_e_stride_words + row * (q8e_p.k_elems >> 2u), live);
    let xw_base = (slot * q8e_p.x_slot_stride_elems) >> 1u;
    var acc = 0.0;
    for (var i = lane; i < words; i = i + 32u) {
        let w = q8e_w[wbase + i];
        let x0 = q8e_x[xw_base + 2u * i];
        let x1 = q8e_x[xw_base + 2u * i + 1u];
        acc = fma(int8_decode(w, 0u), bf16_lo(x0), acc);
        acc = fma(int8_decode(w, 1u), bf16_hi(x0), acc);
        acc = fma(int8_decode(w, 2u), bf16_lo(x1), acc);
        acc = fma(int8_decode(w, 3u), bf16_hi(x1), acc);
    }
    acc = acc + subgroupShuffleXor(acc, 16u);
    acc = acc + subgroupShuffleXor(acc, 8u);
    acc = acc + subgroupShuffleXor(acc, 4u);
    acc = acc + subgroupShuffleXor(acc, 2u);
    acc = acc + subgroupShuffleXor(acc, 1u);
    if (lane == 0u) {
        let sr = q8e_s[e * q8e_p.n_rows + select(0u, row, live)];
        q8e_pk_bits[sgid] = bf16_encode(acc * sr) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = q8e_pk_bits[sgid];
        if (row + 1u < q8e_p.n_rows) {
            word = word | (q8e_pk_bits[sgid + 1u] << 16u);
        }
        q8e_y[slot * q8e_p.y_slot_stride_words + (row >> 1u)] = word;
    }
}

@compute @workgroup_size(256)
fn q3w_gemv_i8g_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * q8e_p.groups_x) * 8u + sgid;
    let live = row < q8e_p.n_rows;
    let words = select(0u, q8e_p.k_elems >> 2u, live);
    let wbase = select(0u, row * (q8e_p.k_elems >> 2u), live);
    let sbase = select(0u, row * q8e_p.groups_per_row, live);
    let gshift = q8e_p.group_shift;
    var acc = 0.0;
    for (var i = lane; i < words; i = i + 32u) {
        let w = q8e_w[wbase + i];
        let x0 = q8e_x[2u * i];
        let x1 = q8e_x[2u * i + 1u];
        var d = 0.0;
        d = fma(int8_decode(w, 0u), bf16_lo(x0), d);
        d = fma(int8_decode(w, 1u), bf16_hi(x0), d);
        d = fma(int8_decode(w, 2u), bf16_lo(x1), d);
        d = fma(int8_decode(w, 3u), bf16_hi(x1), d);
        acc = fma(q8e_s[sbase + (i >> gshift)], d, acc);
    }
    acc = acc + subgroupShuffleXor(acc, 16u);
    acc = acc + subgroupShuffleXor(acc, 8u);
    acc = acc + subgroupShuffleXor(acc, 4u);
    acc = acc + subgroupShuffleXor(acc, 2u);
    acc = acc + subgroupShuffleXor(acc, 1u);
    if (lane == 0u && live) {
        q8e_y[q8e_p.y_off_words + row] = bitcast<u32>(acc);
    }
}

