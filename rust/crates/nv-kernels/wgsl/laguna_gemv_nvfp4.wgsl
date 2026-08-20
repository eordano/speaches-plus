
struct Lg4Params {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    groups_x: u32,
    w_e_stride_vec2: u32,
    sf_e_stride_bytes: u32,
    x_slot_stride_vec2: u32,
    xsf_slot_stride_bytes: u32,
    y_slot_stride_words: u32,
    per_expert_alpha: u32,
    pad0: u32,
};

@group(0) @binding(10) var<storage, read> lg4_w: array<vec2<u32>>;
@group(0) @binding(11) var<storage, read> lg4_ws: array<u32>;
@group(0) @binding(12) var<storage, read> lg4_x: array<vec2<u32>>;
@group(0) @binding(13) var<storage, read> lg4_xs: array<u32>;
@group(0) @binding(14) var<uniform> lg4_p: Lg4Params;
@group(0) @binding(15) var<storage, read_write> lg4_y: array<u32>;
@group(0) @binding(16) var<storage, read> lg4_sel: array<u32>;
@group(0) @binding(17) var<storage, read> lg4_alphas: array<f32>;

var<workgroup> lg4_red: array<f32, 256>;

const LG4_UE4M3_EXP_REBIAS_120_PLUS_24: u32 = 0x48000000u;
const LG4_UE4M3_SUBNORMAL_STEP_TIMES_2POW24: f32 = 32768.0;
const LG4_QUARTER_TIMES_2POW96_RESTORES_BS_CARRY: f32 = 1.9807040628566084e28;

fn lg4_ws_shift_decode_true_times_2powm120(bits: u32) -> f32 {
    return bitcast<f32>((bits & 127u) << 20u);
}

fn lg4_xs_decode_true_times_2pow24_lands_bs_at_2powm96(bits: u32) -> f32 {
    let b = bits & 127u;
    return select(
        bitcast<f32>((b << 20u) + LG4_UE4M3_EXP_REBIAS_120_PLUS_24),
        f32(b) * LG4_UE4M3_SUBNORMAL_STEP_TIMES_2POW24,
        b < 8u
    );
}

fn lg4_dot8_true_times_2pow96(ww: u32, xw: u32, dot_in: f32) -> f32 {
    let d = dot4I8Packed(gemv_i8map(ww), gemv_i8map(xw))
        + dot4I8Packed(gemv_i8map(ww >> 4u), gemv_i8map(xw >> 4u));
    return dot_in + f32(d) * LG4_QUARTER_TIMES_2POW96_RESTORES_BS_CARRY;
}

@compute @workgroup_size(256)
fn lgw_gemv_nvfp4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let slot = wid.z;
    let e = lg4_sel[slot];
    let pair = wid.x + wid.y * lg4_p.groups_x;
    let row = pair * 2u + half;
    let live = row < lg4_p.n_rows;
    let wbase = select(0u, e * lg4_p.w_e_stride_vec2 + row * lg4_p.k_blocks, live);
    let sfbase = e * lg4_p.sf_e_stride_bytes;
    let xbase = slot * lg4_p.x_slot_stride_vec2;
    let xsfbase = slot * lg4_p.xsf_slot_stride_bytes;
    let blocks = select(0u, lg4_p.k_blocks, live);

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + 128u) {
        let wsi = sfbase + nvfp4_scale_byte_index(row, kb, lg4_p.k_tiles);
        let ws = byte_at(lg4_ws[wsi >> 2u], wsi);
        let xsi = xsfbase + kb;
        let xs = byte_at(lg4_xs[xsi >> 2u], xsi);
        let bs = lg4_ws_shift_decode_true_times_2powm120(ws)
            * lg4_xs_decode_true_times_2pow24_lands_bs_at_2powm96(xs);
        let wv = lg4_w[wbase + kb];
        let xv = lg4_x[xbase + kb];
        var dot = 0.0;
        dot = lg4_dot8_true_times_2pow96(wv.x, xv.x, dot);
        dot = lg4_dot8_true_times_2pow96(wv.y, xv.y, dot);
        acc = fma(bs, dot, acc);
    }
    lg4_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            lg4_red[tid] = lg4_red[tid] + lg4_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        var alpha = lg4_p.alpha;
        if (lg4_p.per_expert_alpha == 1u) {
            alpha = lg4_alphas[e];
        }
        let lo = lg4_red[0] * alpha;
        var hi = 0.0;
        if (row + 1u < lg4_p.n_rows) {
            hi = lg4_red[128] * alpha;
        }
        lg4_y[slot * lg4_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}
