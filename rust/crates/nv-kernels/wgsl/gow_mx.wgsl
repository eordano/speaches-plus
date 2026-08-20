
struct GowMxParams {
    n_rows: u32,
    k_blocks: u32,
    groups_x: u32,
    has_bias: u32,
    w_e_stride_v4: u32,
    sf_e_stride_bytes: u32,
    bias_e_stride: u32,
    x_slot_stride_words: u32,
    y_slot_stride: u32,
    use_sel: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(10) var<storage, read> gow_mx_w: array<vec4<u32>>;
@group(0) @binding(11) var<storage, read> gow_mx_sf: array<u32>;
@group(0) @binding(12) var<storage, read> gow_mx_x: array<u32>;
@group(0) @binding(13) var<uniform> gow_mx_p: GowMxParams;
@group(0) @binding(14) var<storage, read_write> gow_mx_y: array<f32>;
@group(0) @binding(15) var<storage, read> gow_mx_sel: array<u32>;
@group(0) @binding(16) var<storage, read> gow_mx_b: array<u32>;

var<workgroup> gow_mx_red: array<f32, 256>;

fn gow_e8m0(byte: u32) -> f32 {
    return bitcast<f32>((byte & 255u) << 23u);
}

fn gow_mx_e2m1(word: u32, j: u32) -> f32 {
    let n = nvfp4_nibble(word, j);
    let c = n & 7u;
    let normal = (((c >> 1u) + 126u) << 23u) | ((c & 1u) << 22u);
    let mag = select(normal, c * 0x3F000000u, c < 2u);
    return bitcast<f32>(mag | ((n & 8u) << 28u));
}

@compute @workgroup_size(16)
fn gow_mx_e2m1_probe(@builtin(local_invocation_id) lid: vec3<u32>) {
    let i = lid.x;
    let word = gow_mx_w[0][i >> 3u];
    let elem = i & 7u;
    gow_mx_y[i] = gow_mx_e2m1(word, elem);
    gow_mx_y[16u + i] =
        e2m1_shift_decode_scale_must_carry_2pow126(nvfp4_nibble(word, elem)) * 0x1p126;
}

fn gow_mx_dot_word(word: u32, xbase: u32) -> f32 {
    var a = 0.0;
    for (var j = 0u; j < 8u; j = j + 1u) {
        let xw = gow_mx_x[xbase + (j >> 1u)];
        let xv = select(bf16_lo(xw), bf16_hi(xw), (j & 1u) == 1u);
        a = fma(gow_mx_e2m1(word, j), xv, a);
    }
    return a;
}

@compute @workgroup_size(256)
fn gow_gemv_mx(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let slot = wid.z;
    var e = slot;
    if (gow_mx_p.use_sel == 1u) {
        e = gow_mx_sel[slot];
    }
    let pair = wid.x + wid.y * gow_mx_p.groups_x;
    let row = pair * 2u + half;
    let live = row < gow_mx_p.n_rows;
    let blocks = select(0u, gow_mx_p.k_blocks, live);
    let wbase = e * gow_mx_p.w_e_stride_v4 + row * gow_mx_p.k_blocks;
    let sfbase = e * gow_mx_p.sf_e_stride_bytes + row * gow_mx_p.k_blocks;
    let xbase = slot * gow_mx_p.x_slot_stride_words;

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + 128u) {
        let wv = gow_mx_w[wbase + kb];
        let xb = xbase + kb * 16u;
        var dot = gow_mx_dot_word(wv.x, xb);
        dot = dot + gow_mx_dot_word(wv.y, xb + 4u);
        dot = dot + gow_mx_dot_word(wv.z, xb + 8u);
        dot = dot + gow_mx_dot_word(wv.w, xb + 12u);
        let si = sfbase + kb;
        acc = fma(gow_e8m0(byte_at(gow_mx_sf[si >> 2u], si)), dot, acc);
    }
    gow_mx_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            gow_mx_red[tid] = gow_mx_red[tid] + gow_mx_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u && live) {
        var v = gow_mx_red[tid];
        if (gow_mx_p.has_bias == 1u) {
            let bi = e * gow_mx_p.bias_e_stride + row;
            v = v + bf16_decode(u16_at(gow_mx_b[bi >> 1u], bi));
        }
        gow_mx_y[slot * gow_mx_p.y_slot_stride + row] = v;
    }
}

const GOW_MX_SECTION_SG: u32 = 1u;

const GOW_MX_SG_LANES: u32 = 32u;
const GOW_MX_SG_ROWS: u32 = 8u;

fn gow_mx_sg_butterfly32(v: f32) -> f32 {
    var a = v;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn gow_mx_dot8(word: u32, x0: u32, x1: u32, x2: u32, x3: u32) -> f32 {
    var a = fma(gow_mx_e2m1(word, 0u), bf16_lo(x0), 0.0);
    a = fma(gow_mx_e2m1(word, 1u), bf16_hi(x0), a);
    a = fma(gow_mx_e2m1(word, 2u), bf16_lo(x1), a);
    a = fma(gow_mx_e2m1(word, 3u), bf16_hi(x1), a);
    a = fma(gow_mx_e2m1(word, 4u), bf16_lo(x2), a);
    a = fma(gow_mx_e2m1(word, 5u), bf16_hi(x2), a);
    a = fma(gow_mx_e2m1(word, 6u), bf16_lo(x3), a);
    a = fma(gow_mx_e2m1(word, 7u), bf16_hi(x3), a);
    return a;
}

fn gow_mx_dot_block(wv: vec4<u32>, xb: u32) -> f32 {
    var d = gow_mx_dot8(wv.x, gow_mx_x[xb], gow_mx_x[xb + 1u], gow_mx_x[xb + 2u], gow_mx_x[xb + 3u]);
    d = d + gow_mx_dot8(wv.y, gow_mx_x[xb + 4u], gow_mx_x[xb + 5u], gow_mx_x[xb + 6u], gow_mx_x[xb + 7u]);
    d = d + gow_mx_dot8(wv.z, gow_mx_x[xb + 8u], gow_mx_x[xb + 9u], gow_mx_x[xb + 10u], gow_mx_x[xb + 11u]);
    d = d + gow_mx_dot8(wv.w, gow_mx_x[xb + 12u], gow_mx_x[xb + 13u], gow_mx_x[xb + 14u], gow_mx_x[xb + 15u]);
    return d;
}

@compute @workgroup_size(256)
fn gow_gemv_mx_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    var e = slot;
    if (gow_mx_p.use_sel == 1u) {
        e = gow_mx_sel[slot];
    }
    let row = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_ROWS + sgid;
    let live = row < gow_mx_p.n_rows;
    let blocks = select(0u, gow_mx_p.k_blocks, live);
    let wbase = e * gow_mx_p.w_e_stride_v4 + select(0u, row * gow_mx_p.k_blocks, live);
    let sfbase = e * gow_mx_p.sf_e_stride_bytes + select(0u, row * gow_mx_p.k_blocks, live);
    let xbase = slot * gow_mx_p.x_slot_stride_words;

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + GOW_MX_SG_LANES) {
        let si = sfbase + kb;
        let dot = gow_mx_dot_block(gow_mx_w[wbase + kb], xbase + kb * 16u);
        acc = fma(gow_e8m0(byte_at(gow_mx_sf[si >> 2u], si)), dot, acc);
    }
    let total = gow_mx_sg_butterfly32(acc);

    if (lane == 0u && live) {
        var v = total;
        if (gow_mx_p.has_bias == 1u) {
            let bi = e * gow_mx_p.bias_e_stride + row;
            v = v + bf16_decode(u16_at(gow_mx_b[bi >> 1u], bi));
        }
        gow_mx_y[slot * gow_mx_p.y_slot_stride + row] = v;
    }
}
