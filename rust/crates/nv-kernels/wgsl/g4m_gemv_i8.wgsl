
struct G4iParams {
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

@group(0) @binding(0) var<storage, read> g4i_w: array<u32>;
@group(0) @binding(1) var<storage, read> g4i_x: array<u32>;
@group(0) @binding(2) var<uniform> g4i_p: G4iParams;
@group(0) @binding(3) var<storage, read_write> g4i_y: array<u32>;
@group(0) @binding(4) var<storage, read> g4i_s: array<u32>;

var<workgroup> g4i_red: array<f32, 256>;

@compute @workgroup_size(256)
fn g4m_gemv_i8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * g4i_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4i_p.n_rows;
    let wbase = select(0u, row * g4i_p.w_row_words, live);
    let sbase = select(0u, row * g4i_p.s_row_elems, live);
    let kw = select(0u, g4i_p.k_words, live);

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = g4i_w[wbase + i];
        let x0 = g4i_x[g4i_p.x_off_words + 2u * i];
        let x1 = g4i_x[g4i_p.x_off_words + 2u * i + 1u];
        let si = sbase + (i >> 3u);
        let scale = bf16_decode(u16_at(g4i_s[si >> 1u], si));
        var d = 0.0;
        d = fma(int8_decode(ww, 0u), bf16_lo(x0), d);
        d = fma(int8_decode(ww, 1u), bf16_hi(x0), d);
        d = fma(int8_decode(ww, 2u), bf16_lo(x1), d);
        d = fma(int8_decode(ww, 3u), bf16_hi(x1), d);
        acc = fma(scale, d, acc);
    }
    g4i_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4i_red[tid] = g4i_red[tid] + g4i_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (g4i_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            g4i_y[g4i_p.y_off_words + row] = bitcast<u32>(g4i_red[tid] * g4i_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = g4i_red[0] * g4i_p.alpha;
        var hi = 0.0;
        if (row + 1u < g4i_p.n_rows) {
            hi = g4i_red[128] * g4i_p.alpha;
        }
        g4i_y[g4i_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}
