
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
