
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
@group(0) @binding(4) var<storage, read> g4b_w2: array<vec4<u32>>;

var<workgroup> g4b_red: array<f32, 256>;
var<workgroup> g4b_gu: array<f32, 2>;
var<workgroup> g4b_uhi: f32;

fn g4b_gelu(x: f32) -> f32 {
    let c = 0.7978845608028654;
    let t = nv_tanhf(c * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1.0 + t);
}

fn g4b_dot_row(row: u32, live: bool, lane: u32, from_w2: bool) -> f32 {
    let wbase = select(0u, row * g4b_p.w_row_words, live);
    let kw = select(0u, g4b_p.k_words, live);
    var acc = 0.0;
    if (g4b_p.wide == 1u) {
        let wv = wbase >> 2u;
        let xv = g4b_p.x_off_words >> 2u;
        let kv = kw >> 2u;
        for (var v = lane; v < kv; v = v + 128u) {
            var ww: vec4<u32>;
            if (from_w2) {
                ww = g4b_w2[wv + v];
            } else {
                ww = g4b_w[wv + v];
            }
            let xw = g4b_x[xv + v];
            acc = fma(bf16_lo(ww.x), bf16_lo(xw.x), acc);
            acc = fma(bf16_hi(ww.x), bf16_hi(xw.x), acc);
            acc = fma(bf16_lo(ww.y), bf16_lo(xw.y), acc);
            acc = fma(bf16_hi(ww.y), bf16_hi(xw.y), acc);
            acc = fma(bf16_lo(ww.z), bf16_lo(xw.z), acc);
            acc = fma(bf16_hi(ww.z), bf16_hi(xw.z), acc);
            acc = fma(bf16_lo(ww.w), bf16_lo(xw.w), acc);
            acc = fma(bf16_hi(ww.w), bf16_hi(xw.w), acc);
        }
    } else {
        for (var i = lane; i < kw; i = i + 128u) {
            let wi = wbase + i;
            let xi = g4b_p.x_off_words + i;
            var ww: u32;
            if (from_w2) {
                ww = g4b_w2[wi >> 2u][wi & 3u];
            } else {
                ww = g4b_w[wi >> 2u][wi & 3u];
            }
            let xw = g4b_x[xi >> 2u][xi & 3u];
            acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
            acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
        }
    }
    return acc;
}

fn g4b_half_reduce_one_barrier_lane_combines_strides_64_32_then_shuffles_16_to_1_matching_the_scratch_tree_order(
    tid: u32,
    acc: f32
) -> f32 {
    g4b_red[tid] = acc;
    workgroupBarrier();
    let hb = tid & 128u;
    let lane = tid & 127u;
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
    return racc;
}

@compute @workgroup_size(256)
fn g4m_gemv_bf16_sglat(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & 127u;
    let half = tid >> 7u;
    let pair = wid.x + wid.y * g4b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4b_p.n_rows;
    let wbase = select(0u, row * g4b_p.w_row_words, live);
    let kw = select(0u, g4b_p.k_words, live);

    var acc = 0.0;
    if (g4b_p.wide == 1u) {
        let wv = wbase >> 2u;
        let xv = g4b_p.x_off_words >> 2u;
        let kv = kw >> 2u;
        for (var v = lane; v < kv; v = v + 128u) {
            let ww = g4b_w[wv + v];
            let xw = g4b_x[xv + v];
            acc = fma(bf16_lo(ww.x), bf16_lo(xw.x), acc);
            acc = fma(bf16_hi(ww.x), bf16_hi(xw.x), acc);
            acc = fma(bf16_lo(ww.y), bf16_lo(xw.y), acc);
            acc = fma(bf16_hi(ww.y), bf16_hi(xw.y), acc);
            acc = fma(bf16_lo(ww.z), bf16_lo(xw.z), acc);
            acc = fma(bf16_hi(ww.z), bf16_hi(xw.z), acc);
            acc = fma(bf16_lo(ww.w), bf16_lo(xw.w), acc);
            acc = fma(bf16_hi(ww.w), bf16_hi(xw.w), acc);
        }
    } else {
        for (var i = lane; i < kw; i = i + 128u) {
            let wi = wbase + i;
            let xi = g4b_p.x_off_words + i;
            let ww = g4b_w[wi >> 2u][wi & 3u];
            let xw = g4b_x[xi >> 2u][xi & 3u];
            acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
            acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
        }
    }
    let racc = g4b_half_reduce_one_barrier_lane_combines_strides_64_32_then_shuffles_16_to_1_matching_the_scratch_tree_order(
        tid,
        acc
    );

    if (g4b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            g4b_y[g4b_p.y_off_words + row] = bitcast<u32>(racc * g4b_p.alpha);
        }
    } else {
        if (tid == 128u) {
            g4b_gu[1] = racc;
        }
        workgroupBarrier();
        if (tid == 0u) {
            let lo = racc * g4b_p.alpha;
            var hi = 0.0;
            if (row + 1u < g4b_p.n_rows) {
                hi = g4b_gu[1] * g4b_p.alpha;
            }
            g4b_y[g4b_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
        }
    }
}

@compute @workgroup_size(256)
fn g4m_gemv_bf16_gu_gelu_sglat(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * g4b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4b_p.n_rows;

    let g_root = g4b_half_reduce_one_barrier_lane_combines_strides_64_32_then_shuffles_16_to_1_matching_the_scratch_tree_order(
        tid,
        g4b_dot_row(row, live, lane, false)
    );
    if (lane == 0u) {
        g4b_gu[half] = g_root * g4b_p.alpha;
    }
    workgroupBarrier();

    let u_root = g4b_half_reduce_one_barrier_lane_combines_strides_64_32_then_shuffles_16_to_1_matching_the_scratch_tree_order(
        tid,
        g4b_dot_row(row, live, lane, true)
    );
    if (tid == 128u) {
        g4b_uhi = u_root;
    }
    workgroupBarrier();
    if (tid == 0u) {
        let gw = bf16_pack(g4b_gu[0], g4b_gu[1]);
        var hi_u = 0.0;
        if (row + 1u < g4b_p.n_rows) {
            hi_u = g4b_uhi * g4b_p.alpha;
        }
        let uw = bf16_pack(u_root * g4b_p.alpha, hi_u);
        let a0 = bf16_decode(bf16_encode(g4b_gelu(bf16_lo(gw)))) * bf16_lo(uw);
        let a1 = bf16_decode(bf16_encode(g4b_gelu(bf16_hi(gw)))) * bf16_hi(uw);
        g4b_y[g4b_p.y_off_words + (row >> 1u)] = bf16_pack(a0, a1);
    }
}
