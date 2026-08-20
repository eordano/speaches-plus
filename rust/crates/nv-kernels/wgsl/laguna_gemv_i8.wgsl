
struct Lg8Params {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    x_slot_stride_elems: u32,
    w_e_stride_words: u32,
    y_slot_stride_words: u32,
    use_sel: u32,
    groups_per_row: u32,
    group_shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> lg8_w: array<u32>;
@group(0) @binding(1) var<storage, read> lg8_s: array<f32>;
@group(0) @binding(2) var<storage, read> lg8_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> lg8_y: array<u32>;
@group(0) @binding(4) var<uniform> lg8_p: Lg8Params;
@group(0) @binding(5) var<storage, read> lg8_sel: array<u32>;

var<workgroup> lg8_pk_bits: array<u32, 8>;

@compute @workgroup_size(256)
fn lgw_gemv_i8g_experts(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    var e = slot;
    if (lg8_p.use_sel == 1u) {
        e = lg8_sel[slot];
    }
    let row = (wid.x + wid.y * lg8_p.groups_x) * 8u + sgid;
    let live = row < lg8_p.n_rows;
    let words = select(0u, lg8_p.k_elems >> 2u, live);
    let wbase = select(0u, e * lg8_p.w_e_stride_words + row * (lg8_p.k_elems >> 2u), live);
    let sbase = select(0u, (e * lg8_p.n_rows + row) * lg8_p.groups_per_row, live);
    let gshift = lg8_p.group_shift;
    let xw_base = (slot * lg8_p.x_slot_stride_elems) >> 1u;
    var acc = 0.0;
    for (var i = lane; i < words; i = i + 32u) {
        let w = lg8_w[wbase + i];
        let x0 = lg8_x[xw_base + 2u * i];
        let x1 = lg8_x[xw_base + 2u * i + 1u];
        var d = 0.0;
        d = fma(int8_decode(w, 0u), bf16_lo(x0), d);
        d = fma(int8_decode(w, 1u), bf16_hi(x0), d);
        d = fma(int8_decode(w, 2u), bf16_lo(x1), d);
        d = fma(int8_decode(w, 3u), bf16_hi(x1), d);
        acc = fma(lg8_s[sbase + (i >> gshift)], d, acc);
    }
    acc = acc + subgroupShuffleXor(acc, 16u);
    acc = acc + subgroupShuffleXor(acc, 8u);
    acc = acc + subgroupShuffleXor(acc, 4u);
    acc = acc + subgroupShuffleXor(acc, 2u);
    acc = acc + subgroupShuffleXor(acc, 1u);
    if (lane == 0u) {
        lg8_pk_bits[sgid] = bf16_encode(acc) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = lg8_pk_bits[sgid];
        if (row + 1u < lg8_p.n_rows) {
            word = word | (lg8_pk_bits[sgid + 1u] << 16u);
        }
        lg8_y[slot * lg8_p.y_slot_stride_words + (row >> 1u)] = word;
    }
}

@compute @workgroup_size(256)
fn lgw_gemv_i8_experts(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    var e = slot;
    if (lg8_p.use_sel == 1u) {
        e = lg8_sel[slot];
    }
    let row = (wid.x + wid.y * lg8_p.groups_x) * 8u + sgid;
    let live = row < lg8_p.n_rows;
    let words = select(0u, lg8_p.k_elems >> 2u, live);
    let wbase = select(0u, e * lg8_p.w_e_stride_words + row * (lg8_p.k_elems >> 2u), live);
    let xw_base = (slot * lg8_p.x_slot_stride_elems) >> 1u;
    var acc = 0.0;
    for (var i = lane; i < words; i = i + 32u) {
        let w = lg8_w[wbase + i];
        let x0 = lg8_x[xw_base + 2u * i];
        let x1 = lg8_x[xw_base + 2u * i + 1u];
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
        let sr = lg8_s[e * lg8_p.n_rows + select(0u, row, live)];
        lg8_pk_bits[sgid] = bf16_encode(acc * sr) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = lg8_pk_bits[sgid];
        if (row + 1u < lg8_p.n_rows) {
            word = word | (lg8_pk_bits[sgid + 1u] << 16u);
        }
        lg8_y[slot * lg8_p.y_slot_stride_words + (row >> 1u)] = word;
    }
}
