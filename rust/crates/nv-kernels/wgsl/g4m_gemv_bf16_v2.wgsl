
struct G4bParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    wide: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<storage, read> g4b_w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> g4b_x: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> g4b_p: G4bParams;
@group(0) @binding(3) var<storage, read_write> g4b_y: array<u32>;

var<workgroup> g4b_v2_row_roots: array<f32, 8>;

fn g4b_v2_dot8(ww: vec4<u32>, xw: vec4<u32>, acc_in: f32) -> f32 {
    var acc = acc_in;
    acc = fma(bf16_lo(ww.x), bf16_lo(xw.x), acc);
    acc = fma(bf16_hi(ww.x), bf16_hi(xw.x), acc);
    acc = fma(bf16_lo(ww.y), bf16_lo(xw.y), acc);
    acc = fma(bf16_hi(ww.y), bf16_hi(xw.y), acc);
    acc = fma(bf16_lo(ww.z), bf16_lo(xw.z), acc);
    acc = fma(bf16_hi(ww.z), bf16_hi(xw.z), acc);
    acc = fma(bf16_lo(ww.w), bf16_lo(xw.w), acc);
    acc = fma(bf16_hi(ww.w), bf16_hi(xw.w), acc);
    return acc;
}

@compute @workgroup_size(256)
fn g4m_gemv_bf16_v2_row_per_sg32_dual_vec4_zero_barrier_reduce(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let sub = tid >> 5u;
    let lane = tid & 31u;
    let block = wid.x + wid.y * g4b_p.groups_x;
    let row = block * 8u + sub;
    let live = row < g4b_p.n_rows;
    let wbase = select(0u, row * g4b_p.w_row_words, live);
    let kw = select(0u, g4b_p.k_words, live);
    let wv = wbase >> 2u;
    let xv = g4b_p.x_off_words >> 2u;
    let kv = kw >> 2u;

    var a0 = 0.0;
    var a1 = 0.0;
    for (var v = lane; v < kv; v = v + 64u) {
        a0 = g4b_v2_dot8(g4b_w[wv + v], g4b_x[xv + v], a0);
        let v1 = v + 32u;
        if (v1 < kv) {
            a1 = g4b_v2_dot8(g4b_w[wv + v1], g4b_x[xv + v1], a1);
        }
    }
    var racc = a0 + a1;
    racc = racc + subgroupShuffleXor(racc, 16u);
    racc = racc + subgroupShuffleXor(racc, 8u);
    racc = racc + subgroupShuffleXor(racc, 4u);
    racc = racc + subgroupShuffleXor(racc, 2u);
    racc = racc + subgroupShuffleXor(racc, 1u);

    if (g4b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            g4b_y[g4b_p.y_off_words + row] = bitcast<u32>(racc * g4b_p.alpha);
        }
    } else {
        if (lane == 0u) {
            g4b_v2_row_roots[sub] = racc;
        }
        workgroupBarrier();
        if (lane == 0u && (sub & 1u) == 0u && live) {
            let lo = racc * g4b_p.alpha;
            var hi = 0.0;
            if (row + 1u < g4b_p.n_rows) {
                hi = g4b_v2_row_roots[sub + 1u] * g4b_p.alpha;
            }
            g4b_y[g4b_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
        }
    }
}
