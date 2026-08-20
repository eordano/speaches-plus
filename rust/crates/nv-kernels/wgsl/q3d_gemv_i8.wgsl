
struct Q3q8Params {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    groups_per_row: u32,
    group_shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> q3q8_w: array<u32>;
@group(0) @binding(1) var<storage, read> q3q8_s: array<f32>;
@group(0) @binding(2) var<storage, read> q3q8_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> q3q8_y: array<u32>;
@group(0) @binding(4) var<uniform> q3q8_p: Q3q8Params;

var<workgroup> q3q8_pk_bits: array<u32, 8>;

@compute @workgroup_size(256)
fn q3d_gemv_i8g(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * q3q8_p.groups_x) * 8u + sgid;
    let live = row < q3q8_p.n_rows;
    let words = select(0u, q3q8_p.k_elems >> 2u, live);
    let wbase = select(0u, row * (q3q8_p.k_elems >> 2u), live);
    let sbase = select(0u, row * q3q8_p.groups_per_row, live);
    let gshift = q3q8_p.group_shift;
    var acc = 0.0;
    for (var i = lane; i < words; i = i + 32u) {
        let w = q3q8_w[wbase + i];
        let x0 = q3q8_x[2u * i];
        let x1 = q3q8_x[2u * i + 1u];
        var d = 0.0;
        d = fma(int8_decode(w, 0u), bf16_lo(x0), d);
        d = fma(int8_decode(w, 1u), bf16_hi(x0), d);
        d = fma(int8_decode(w, 2u), bf16_lo(x1), d);
        d = fma(int8_decode(w, 3u), bf16_hi(x1), d);
        acc = fma(q3q8_s[sbase + (i >> gshift)], d, acc);
    }
    acc = acc + subgroupShuffleXor(acc, 16u);
    acc = acc + subgroupShuffleXor(acc, 8u);
    acc = acc + subgroupShuffleXor(acc, 4u);
    acc = acc + subgroupShuffleXor(acc, 2u);
    acc = acc + subgroupShuffleXor(acc, 1u);
    if (lane == 0u) {
        q3q8_pk_bits[sgid] = bf16_encode(acc) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = q3q8_pk_bits[sgid];
        if (row + 1u < q3q8_p.n_rows) {
            word = word | (q3q8_pk_bits[sgid + 1u] << 16u);
        }
        q3q8_y[row >> 1u] = word;
    }
}

@compute @workgroup_size(256)
fn q3d_gemv_i8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * q3q8_p.groups_x) * 8u + sgid;
    let live = row < q3q8_p.n_rows;
    let words = select(0u, q3q8_p.k_elems >> 2u, live);
    let wbase = select(0u, row * (q3q8_p.k_elems >> 2u), live);
    var acc = 0.0;
    for (var i = lane; i < words; i = i + 32u) {
        let w = q3q8_w[wbase + i];
        let x0 = q3q8_x[2u * i];
        let x1 = q3q8_x[2u * i + 1u];
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
        q3q8_pk_bits[sgid] = bf16_encode(acc * q3q8_s[select(0u, row, live)]) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = q3q8_pk_bits[sgid];
        if (row + 1u < q3q8_p.n_rows) {
            word = word | (q3q8_pk_bits[sgid + 1u] << 16u);
        }
        q3q8_y[row >> 1u] = word;
    }
}
