
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
@group(0) @binding(5) var<storage, read> g4b_rsw: array<u32>;

var<workgroup> g4b_red: array<f32, 256>;
var<workgroup> g4b_xn: array<u32, 2048>;
var<workgroup> g4b_ns: f32;

fn rpl_norm_reduce_two_barriers_lane_combines_strides_128_64_32_then_shuffles_16_to_1_matching_the_scratch_tree_order(
    tid: u32,
    local: f32
) -> f32 {
    g4b_red[tid] = local;
    workgroupBarrier();
    var acc = 0.0;
    if (tid < 32u) {
        acc = ((g4b_red[tid] + g4b_red[tid + 128u]) + (g4b_red[tid + 64u] + g4b_red[tid + 192u]))
            + ((g4b_red[tid + 32u] + g4b_red[tid + 160u]) + (g4b_red[tid + 96u] + g4b_red[tid + 224u]));
    }
    acc = acc + subgroupShuffleXor(acc, 16u);
    acc = acc + subgroupShuffleXor(acc, 8u);
    acc = acc + subgroupShuffleXor(acc, 4u);
    acc = acc + subgroupShuffleXor(acc, 2u);
    acc = acc + subgroupShuffleXor(acc, 1u);
    if (tid == 0u) {
        g4b_ns = inverseSqrt(acc / f32(g4b_p.pad1) + bitcast<f32>(g4b_p.pad2));
    }
    workgroupBarrier();
    return g4b_ns;
}

@compute @workgroup_size(256)
fn g4m_gemv_bf16_rpnorm_sglat(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let words = g4b_p.k_words;
    var local = 0.0;
    for (var w = tid; w < words; w = w + 256u) {
        let xw = g4b_x[w >> 2u][w & 3u];
        let x0 = bf16_lo(xw);
        let x1 = bf16_hi(xw);
        local = local + x0 * x0 + x1 * x1;
    }
    let s = rpl_norm_reduce_two_barriers_lane_combines_strides_128_64_32_then_shuffles_16_to_1_matching_the_scratch_tree_order(
        tid,
        local
    );
    for (var w = tid; w < words; w = w + 256u) {
        let xw = g4b_x[w >> 2u][w & 3u];
        let ww = g4b_rsw[w];
        let t = bf16_pack(bf16_lo(xw) * s, bf16_hi(xw) * s);
        g4b_xn[w] = bf16_pack(bf16_lo(t) * bf16_lo(ww), bf16_hi(t) * bf16_hi(ww));
    }
    workgroupBarrier();

    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * g4b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4b_p.n_rows;
    let wbase = select(0u, row * g4b_p.w_row_words, live);
    let kw = select(0u, g4b_p.k_words, live);
    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let wi = wbase + i;
        let ww = g4b_w[wi >> 2u][wi & 3u];
        let xw = g4b_xn[i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    g4b_red[tid] = acc;
    workgroupBarrier();
    let hb = half << 7u;
    var racc = 0.0;
    if (lane < 32u) {
        racc = (g4b_red[hb + lane] + g4b_red[hb + lane + 64u])
            + (g4b_red[hb + lane + 32u] + g4b_red[hb + lane + 96u]);
    }
    racc = racc + subgroupShuffleXor(racc, 16u);
    racc = racc + subgroupShuffleXor(racc, 8u);
    racc = racc + subgroupShuffleXor(racc, 4u);
    racc = racc + subgroupShuffleXor(racc, 2u);
    racc = racc + subgroupShuffleXor(racc, 1u);
    if (lane == 0u && live) {
        g4b_y[g4b_p.y_off_words + row] = bitcast<u32>(racc * g4b_p.alpha);
    }
}
