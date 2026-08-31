
struct G4g16Params {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    s_row_elems: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<storage, read> g4g16_w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> g4g16_x: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> g4g16_p: G4g16Params;
@group(0) @binding(3) var<storage, read_write> g4g16_y: array<u32>;
@group(0) @binding(4) var<storage, read> g4g16_s: array<u32>;

var<workgroup> g4g16_red: array<f32, 256>;

fn g4g16_dot4(word: u32, xw0: u32, xw1: u32, acc_in: f32) -> f32 {
    var acc = acc_in;
    acc = fma(int8_decode(word, 0u), bf16_lo(xw0), acc);
    acc = fma(int8_decode(word, 1u), bf16_hi(xw0), acc);
    acc = fma(int8_decode(word, 2u), bf16_lo(xw1), acc);
    acc = fma(int8_decode(word, 3u), bf16_hi(xw1), acc);
    return acc;
}

@compute @workgroup_size(256)
fn g4m_gemv_i8_v4_g16_one_scale_per_vec4_word_halves_the_group_of_the_shared_i8_entry(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * g4g16_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4g16_p.n_rows;
    let wbase = select(0u, row * g4g16_p.w_row_words, live);
    let sbase = select(0u, row * g4g16_p.s_row_elems, live);
    let kv = select(0u, g4g16_p.k_words, live);

    var acc = 0.0;
    for (var v = lane; v < kv; v = v + 128u) {
        let wv = g4g16_w[wbase + v];
        let xa = g4g16_x[g4g16_p.x_off_words + 2u * v];
        let xb = g4g16_x[g4g16_p.x_off_words + 2u * v + 1u];
        let si = sbase + v;
        let scale = bf16_decode(u16_at(g4g16_s[si >> 1u], si));
        var d = 0.0;
        d = g4g16_dot4(wv.x, xa.x, xa.y, d);
        d = g4g16_dot4(wv.y, xa.z, xa.w, d);
        d = g4g16_dot4(wv.z, xb.x, xb.y, d);
        d = g4g16_dot4(wv.w, xb.z, xb.w, d);
        acc = fma(scale, d, acc);
    }
    g4g16_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4g16_red[tid] = g4g16_red[tid] + g4g16_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (g4g16_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            g4g16_y[g4g16_p.y_off_words + row] = bitcast<u32>(g4g16_red[tid] * g4g16_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = g4g16_red[0] * g4g16_p.alpha;
        var hi = 0.0;
        if (row + 1u < g4g16_p.n_rows) {
            hi = g4g16_red[128] * g4g16_p.alpha;
        }
        g4g16_y[g4g16_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}
