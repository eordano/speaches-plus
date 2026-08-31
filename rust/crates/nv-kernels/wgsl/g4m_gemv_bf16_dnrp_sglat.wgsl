
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

@group(0) @binding(0) var<storage, read> dn_w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> dn_x: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> dn_p: G4bParams;
@group(0) @binding(3) var<storage, read_write> dn_y: array<u32>;
@group(0) @binding(4) var<storage, read> rp_w: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> rp_rsw: array<u32>;
@group(0) @binding(6) var<storage, read> rp_x: array<vec4<u32>>;
@group(0) @binding(7) var<uniform> rp_p: G4bParams;
@group(0) @binding(8) var<storage, read_write> rp_y: array<u32>;

var<workgroup> g4b_red: array<f32, 256>;
var<workgroup> g4b_gu: array<f32, 2>;
var<workgroup> rp_xn: array<u32, 2048>;
var<workgroup> rp_ns: f32;

fn dnrp_shuffle_combine_strides_64_32_then_16_to_1_matching_the_scratch_tree_order(
    hb: u32,
    lane: u32
) -> f32 {
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
fn g4m_gemv_bf16_down_rpnorm_sglat(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & 127u;
    let half = tid >> 7u;
    let is_rp = wid.x >= dn_p.groups_x;

    var acc = 0.0;
    var dn_live = false;
    var dn_row = 0u;
    var rp_row = 0u;
    var rp_live = false;
    if (is_rp) {
        let words = rp_p.k_words;
        var local = 0.0;
        for (var w = tid; w < words; w = w + 256u) {
            let xw = rp_x[w >> 2u][w & 3u];
            let x0 = bf16_lo(xw);
            let x1 = bf16_hi(xw);
            local = local + x0 * x0 + x1 * x1;
        }
        g4b_red[tid] = local;
    } else {
        let pair = wid.x + wid.y * dn_p.groups_x;
        dn_row = pair * 2u + half;
        dn_live = dn_row < dn_p.n_rows;
        let wbase = select(0u, dn_row * dn_p.w_row_words, dn_live);
        let kw = select(0u, dn_p.k_words, dn_live);
        if (dn_p.wide == 1u) {
            let wv = wbase >> 2u;
            let xv = dn_p.x_off_words >> 2u;
            let kv = kw >> 2u;
            for (var v = lane; v < kv; v = v + 128u) {
                let ww = dn_w[wv + v];
                let xw = dn_x[xv + v];
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
                let xi = dn_p.x_off_words + i;
                let ww = dn_w[wi >> 2u][wi & 3u];
                let xw = dn_x[xi >> 2u][xi & 3u];
                acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
                acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
            }
        }
        g4b_red[tid] = acc;
    }
    workgroupBarrier();

    var dn_racc = 0.0;
    if (is_rp) {
        var nacc = 0.0;
        if (tid < 32u) {
            nacc = ((g4b_red[tid] + g4b_red[tid + 128u]) + (g4b_red[tid + 64u] + g4b_red[tid + 192u]))
                + ((g4b_red[tid + 32u] + g4b_red[tid + 160u]) + (g4b_red[tid + 96u] + g4b_red[tid + 224u]));
        }
        nacc = nacc + subgroupShuffleXor(nacc, 16u);
        nacc = nacc + subgroupShuffleXor(nacc, 8u);
        nacc = nacc + subgroupShuffleXor(nacc, 4u);
        nacc = nacc + subgroupShuffleXor(nacc, 2u);
        nacc = nacc + subgroupShuffleXor(nacc, 1u);
        if (tid == 0u) {
            rp_ns = inverseSqrt(nacc / f32(rp_p.pad1) + bitcast<f32>(rp_p.pad2));
        }
    } else {
        dn_racc = dnrp_shuffle_combine_strides_64_32_then_16_to_1_matching_the_scratch_tree_order(
            tid & 128u,
            lane
        );
        if (dn_p.out_f32 == 1u) {
            if (lane == 0u && dn_live) {
                dn_y[dn_p.y_off_words + dn_row] = bitcast<u32>(dn_racc * dn_p.alpha);
            }
        } else {
            if (tid == 128u) {
                g4b_gu[1] = dn_racc;
            }
        }
    }
    workgroupBarrier();

    if (is_rp) {
        let s = rp_ns;
        let words = rp_p.k_words;
        for (var w = tid; w < words; w = w + 256u) {
            let xw = rp_x[w >> 2u][w & 3u];
            let ww = rp_rsw[w];
            let t = bf16_pack(bf16_lo(xw) * s, bf16_hi(xw) * s);
            rp_xn[w] = bf16_pack(bf16_lo(t) * bf16_lo(ww), bf16_hi(t) * bf16_hi(ww));
        }
    } else {
        if (dn_p.out_f32 != 1u && tid == 0u) {
            let lo = dn_racc * dn_p.alpha;
            var hi = 0.0;
            if (dn_row + 1u < dn_p.n_rows) {
                hi = g4b_gu[1] * dn_p.alpha;
            }
            dn_y[dn_p.y_off_words + (dn_row >> 1u)] = bf16_pack(lo, hi);
        }
    }
    workgroupBarrier();

    var acc2 = 0.0;
    if (is_rp) {
        let pair = (wid.x - dn_p.groups_x) + wid.y * rp_p.groups_x;
        rp_row = pair * 2u + half;
        rp_live = rp_row < rp_p.n_rows;
        let wbase = select(0u, rp_row * rp_p.w_row_words, rp_live);
        let kw = select(0u, rp_p.k_words, rp_live);
        for (var i = lane; i < kw; i = i + 128u) {
            let wi = wbase + i;
            let ww = rp_w[wi >> 2u][wi & 3u];
            let xw = rp_xn[i];
            acc2 = fma(bf16_lo(ww), bf16_lo(xw), acc2);
            acc2 = fma(bf16_hi(ww), bf16_hi(xw), acc2);
        }
        g4b_red[tid] = acc2;
    }
    workgroupBarrier();

    if (is_rp) {
        let racc = dnrp_shuffle_combine_strides_64_32_then_16_to_1_matching_the_scratch_tree_order(
            half << 7u,
            lane
        );
        if (lane == 0u && rp_live) {
            rp_y[rp_p.y_off_words + rp_row] = bitcast<u32>(racc * rp_p.alpha);
        }
    }
}
