struct G4i8dnParams {
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

struct G4i8rpParams {
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

@group(0) @binding(0) var<storage, read> i8dn_w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> i8dn_x: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> i8dn_p: G4i8dnParams;
@group(0) @binding(3) var<storage, read_write> i8dn_y: array<u32>;
@group(0) @binding(4) var<storage, read> i8rp_w: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read> i8rp_rsw: array<u32>;
@group(0) @binding(6) var<storage, read> i8rp_x: array<vec4<u32>>;
@group(0) @binding(7) var<uniform> i8rp_p: G4i8rpParams;
@group(0) @binding(8) var<storage, read_write> i8rp_y: array<u32>;
@group(0) @binding(9) var<storage, read> i8dn_s: array<u32>;

var<workgroup> i8dn_red: array<f32, 256>;
var<workgroup> i8dn_hi: array<f32, 2>;
var<workgroup> i8rp_xn: array<u32, 2048>;
var<workgroup> i8rp_ns: f32;

fn i8dn_dot4(word: u32, xw0: u32, xw1: u32, acc_in: f32) -> f32 {
    var acc = acc_in;
    acc = fma(int8_decode(word, 0u), bf16_lo(xw0), acc);
    acc = fma(int8_decode(word, 1u), bf16_hi(xw0), acc);
    acc = fma(int8_decode(word, 2u), bf16_lo(xw1), acc);
    acc = fma(int8_decode(word, 3u), bf16_hi(xw1), acc);
    return acc;
}

fn i8dnrp_shuffle_combine_strides_64_32_then_16_to_1_matching_the_scratch_tree_order(
    hb: u32,
    lane: u32
) -> f32 {
    var racc = 0.0;
    if (lane < 32u) {
        racc = (i8dn_red[hb + lane] + i8dn_red[hb + lane + 64u])
            + (i8dn_red[hb + lane + 32u] + i8dn_red[hb + lane + 96u]);
    }
    racc = racc + subgroupShuffleXor(racc, 16u);
    racc = racc + subgroupShuffleXor(racc, 8u);
    racc = racc + subgroupShuffleXor(racc, 4u);
    racc = racc + subgroupShuffleXor(racc, 2u);
    racc = racc + subgroupShuffleXor(racc, 1u);
    return racc;
}

@compute @workgroup_size(256)
fn g4m_gemv_i8_down_rpnorm_sglat(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & 127u;
    let half = tid >> 7u;
    let is_rp = wid.x >= i8dn_p.groups_x;

    var dn_live = false;
    var dn_row = 0u;
    var rp_row = 0u;
    var rp_live = false;
    if (is_rp) {
        let words = i8rp_p.k_words;
        var local = 0.0;
        for (var w = tid; w < words; w = w + 256u) {
            let xw = i8rp_x[w >> 2u][w & 3u];
            let x0 = bf16_lo(xw);
            let x1 = bf16_hi(xw);
            local = local + x0 * x0 + x1 * x1;
        }
        i8dn_red[tid] = local;
    } else {
        let pair = wid.x + wid.y * i8dn_p.groups_x;
        dn_row = pair * 2u + half;
        dn_live = dn_row < i8dn_p.n_rows;
        let wbase = select(0u, dn_row * i8dn_p.w_row_words, dn_live);
        let sbase = select(0u, dn_row * i8dn_p.s_row_elems, dn_live);
        let kv = select(0u, i8dn_p.k_words, dn_live);
        var acc = 0.0;
        for (var v = lane; v < kv; v = v + 128u) {
            let wv = i8dn_w[wbase + v];
            let xa = i8dn_x[i8dn_p.x_off_words + 2u * v];
            let xb = i8dn_x[i8dn_p.x_off_words + 2u * v + 1u];
            let si = sbase + (v >> 1u);
            let scale = bf16_decode(u16_at(i8dn_s[si >> 1u], si));
            var d = 0.0;
            d = i8dn_dot4(wv.x, xa.x, xa.y, d);
            d = i8dn_dot4(wv.y, xa.z, xa.w, d);
            d = i8dn_dot4(wv.z, xb.x, xb.y, d);
            d = i8dn_dot4(wv.w, xb.z, xb.w, d);
            acc = fma(scale, d, acc);
        }
        i8dn_red[tid] = acc;
    }
    workgroupBarrier();

    var dn_racc = 0.0;
    if (is_rp) {
        var nacc = 0.0;
        if (tid < 32u) {
            nacc = ((i8dn_red[tid] + i8dn_red[tid + 128u]) + (i8dn_red[tid + 64u] + i8dn_red[tid + 192u]))
                + ((i8dn_red[tid + 32u] + i8dn_red[tid + 160u]) + (i8dn_red[tid + 96u] + i8dn_red[tid + 224u]));
        }
        nacc = nacc + subgroupShuffleXor(nacc, 16u);
        nacc = nacc + subgroupShuffleXor(nacc, 8u);
        nacc = nacc + subgroupShuffleXor(nacc, 4u);
        nacc = nacc + subgroupShuffleXor(nacc, 2u);
        nacc = nacc + subgroupShuffleXor(nacc, 1u);
        if (tid == 0u) {
            i8rp_ns = inverseSqrt(nacc / f32(i8rp_p.pad1) + bitcast<f32>(i8rp_p.pad2));
        }
    } else {
        dn_racc = i8dnrp_shuffle_combine_strides_64_32_then_16_to_1_matching_the_scratch_tree_order(
            tid & 128u,
            lane
        );
        if (i8dn_p.out_f32 == 1u) {
            if (lane == 0u && dn_live) {
                i8dn_y[i8dn_p.y_off_words + dn_row] = bitcast<u32>(dn_racc * i8dn_p.alpha);
            }
        } else {
            if (tid == 128u) {
                i8dn_hi[1] = dn_racc;
            }
        }
    }
    workgroupBarrier();

    if (is_rp) {
        let s = i8rp_ns;
        let words = i8rp_p.k_words;
        for (var w = tid; w < words; w = w + 256u) {
            let xw = i8rp_x[w >> 2u][w & 3u];
            let ww = i8rp_rsw[w];
            let t = bf16_pack(bf16_lo(xw) * s, bf16_hi(xw) * s);
            i8rp_xn[w] = bf16_pack(bf16_lo(t) * bf16_lo(ww), bf16_hi(t) * bf16_hi(ww));
        }
    } else {
        if (i8dn_p.out_f32 != 1u && tid == 0u) {
            let lo = dn_racc * i8dn_p.alpha;
            var hi = 0.0;
            if (dn_row + 1u < i8dn_p.n_rows) {
                hi = i8dn_hi[1] * i8dn_p.alpha;
            }
            i8dn_y[i8dn_p.y_off_words + (dn_row >> 1u)] = bf16_pack(lo, hi);
        }
    }
    workgroupBarrier();

    var acc2 = 0.0;
    if (is_rp) {
        let pair = (wid.x - i8dn_p.groups_x) + wid.y * i8rp_p.groups_x;
        rp_row = pair * 2u + half;
        rp_live = rp_row < i8rp_p.n_rows;
        let wbase = select(0u, rp_row * i8rp_p.w_row_words, rp_live);
        let kw = select(0u, i8rp_p.k_words, rp_live);
        for (var i = lane; i < kw; i = i + 128u) {
            let wi = wbase + i;
            let ww = i8rp_w[wi >> 2u][wi & 3u];
            let xw = i8rp_xn[i];
            acc2 = fma(bf16_lo(ww), bf16_lo(xw), acc2);
            acc2 = fma(bf16_hi(ww), bf16_hi(xw), acc2);
        }
        i8dn_red[tid] = acc2;
    }
    workgroupBarrier();

    if (is_rp) {
        let racc = i8dnrp_shuffle_combine_strides_64_32_then_16_to_1_matching_the_scratch_tree_order(
            half << 7u,
            lane
        );
        if (lane == 0u && rp_live) {
            i8rp_y[i8rp_p.y_off_words + rp_row] = bitcast<u32>(racc * i8rp_p.alpha);
        }
    }
}
