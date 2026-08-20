
struct GemvSgParams {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
};

@group(0) @binding(0) var<storage, read> sg_w4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> sg_x4: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> sg_y: array<u32>;
@group(0) @binding(3) var<uniform> sg_params: GemvSgParams;
@group(0) @binding(4) var<storage, read> sg_w: array<u32>;
@group(0) @binding(5) var<storage, read> sg_x: array<u32>;

fn sg_butterfly(acc: f32) -> f32 {
    var a = acc;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn sg_row(wid: vec3<u32>, rows: u32, sgid: u32) -> u32 {
    return (wid.x + wid.y * sg_params.groups_x) * rows + sgid;
}

fn sg_v4_body(wid: vec3<u32>, sgid: u32, lane: u32, rows: u32) {
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
    if (lane == 0u && live) {
        sg_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(64)
fn gemv_bf16_sg_v4_wg64(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    sg_v4_body(wid, sgid, lane, 2u);
}

@compute @workgroup_size(128)
fn gemv_bf16_sg_v4_wg128(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    sg_v4_body(wid, sgid, lane, 4u);
}

@compute @workgroup_size(256)
fn gemv_bf16_sg_v4_wg256(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    sg_v4_body(wid, sgid, lane, 8u);
}

@compute @workgroup_size(512)
fn gemv_bf16_sg_v4_wg512(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    sg_v4_body(wid, sgid, lane, 16u);
}

@compute @workgroup_size(256)
fn gemv_bf16_sg_u32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = sg_row(wid, 8u, sgid);
    let live = row < sg_params.n_rows;
    let kv = sg_params.k_elems >> 3u;
    let w_base = select(0u, row * sg_params.w_row_words, live);
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + 32u) {
        let wo = w_base + (v << 2u);
        let xo = v << 2u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let ww = sg_w[wo + j];
            let xw = sg_x[xo + j];
            acc = acc + (bf16_lo(ww) * bf16_lo(xw) + bf16_hi(ww) * bf16_hi(xw));
        }
    }
    let total = sg_butterfly(acc);
    if (lane == 0u && live) {
        sg_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(256)
fn gemv_bf16_sg_scalar(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = sg_row(wid, 8u, sgid);
    let live = row < sg_params.n_rows;
    let ke = select(0u, sg_params.k_elems, live);
    let w_base = select(0u, row * sg_params.w_row_words, live);
    let last = (max(sg_params.k_elems, 1u) - 1u) >> 1u;
    let tpairs = (sg_params.k_elems + 63u) >> 6u;
    let odd = (lane & 1u) == 1u;
    var acc = 0.0;
    for (var t = 0u; t < tpairs; t = t + 1u) {
        let myw = (lane >> 1u) + 32u * t + select(0u, 16u, odd);
        let wi = min(myw, last);
        let ww = sg_w[w_base + wi];
        let xw = sg_x[wi];
        let oww = subgroupShuffleXor(ww, 1u);
        let oxw = subgroupShuffleXor(xw, 1u);
        let w1 = select(ww, oww, odd);
        let x1 = select(xw, oxw, odd);
        let w2 = select(oww, ww, odd);
        let x2 = select(oxw, xw, odd);
        let k1 = lane + 64u * t;
        let k2 = k1 + 32u;
        let p1 = bf16_decode(u16_at(w1, k1)) * bf16_decode(u16_at(x1, k1));
        acc = select(acc, acc + p1, k1 < ke);
        let p2 = bf16_decode(u16_at(w2, k2)) * bf16_decode(u16_at(x2, k2));
        acc = select(acc, acc + p2, k2 < ke);
    }
    let total = sg_butterfly(acc);
    if (lane == 0u && live) {
        sg_y[row] = bf16_encode(total);
    }
}
