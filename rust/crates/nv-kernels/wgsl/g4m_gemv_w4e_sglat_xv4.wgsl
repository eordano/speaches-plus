
struct G4wqParams {
    n_rows: u32,
    groups: u32,
    groups_x: u32,
    w_e_stride_words: u32,
    s_e_stride_elems: u32,
    x_slot_stride_words: u32,
    y_slot_stride_words: u32,
    k_top: u32,
    n_experts: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> g4w_w: array<vec4<u32>>;
@group(0) @binding(11) var<storage, read> g4w_ws: array<u32>;
@group(0) @binding(12) var<storage, read> g4w_x: array<vec4<u32>>;
@group(0) @binding(14) var<uniform> g4w_p: G4wqParams;
@group(0) @binding(15) var<storage, read_write> g4w_y: array<u32>;
@group(0) @binding(16) var<storage, read> g4w_sel: array<u32>;
@group(0) @binding(17) var<storage, read> g4w_w2: array<vec4<u32>>;
@group(0) @binding(18) var<storage, read> g4w_ws2: array<u32>;
@group(0) @binding(19) var<storage, read> g4w_cw: array<f32>;

var<workgroup> g4w_red: array<f32, 256>;
var<workgroup> g4w_red2: array<f32, 256>;
var<workgroup> g4w_gu: array<f32, 8>;
var<workgroup> g4w_dots: array<f32, 16>;
var<workgroup> g4w_hi2: vec2<f32>;

fn g4w_gelu(x: f32) -> f32 {
    let c = 0.7978845608028654;
    let t = nv_tanhf(c * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1.0 + t);
}

fn g4w_word_dot4_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
    pv: u32,
    xv: vec4<u32>,
    acc_in: f32
) -> f32 {
    var acc = acc_in;
    acc = fma(f32(pv & 15u) - 8.0, bf16_lo(xv.x), acc);
    acc = fma(f32((pv >> 4u) & 15u) - 8.0, bf16_hi(xv.x), acc);
    acc = fma(f32((pv >> 8u) & 15u) - 8.0, bf16_lo(xv.y), acc);
    acc = fma(f32((pv >> 12u) & 15u) - 8.0, bf16_hi(xv.y), acc);
    acc = fma(f32((pv >> 16u) & 15u) - 8.0, bf16_lo(xv.z), acc);
    acc = fma(f32((pv >> 20u) & 15u) - 8.0, bf16_hi(xv.z), acc);
    acc = fma(f32((pv >> 24u) & 15u) - 8.0, bf16_lo(xv.w), acc);
    acc = fma(f32((pv >> 28u) & 15u) - 8.0, bf16_hi(xv.w), acc);
    return acc;
}

fn g4w_word_dot4x2_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
    pg: u32,
    pu: u32,
    xv: vec4<u32>,
    gd_in: f32,
    ud_in: f32
) -> vec2<f32> {
    var gd = gd_in;
    var ud = ud_in;
    let xl0 = bf16_lo(xv.x);
    let xh0 = bf16_hi(xv.x);
    gd = fma(f32(pg & 15u) - 8.0, xl0, gd);
    gd = fma(f32((pg >> 4u) & 15u) - 8.0, xh0, gd);
    ud = fma(f32(pu & 15u) - 8.0, xl0, ud);
    ud = fma(f32((pu >> 4u) & 15u) - 8.0, xh0, ud);
    let xl1 = bf16_lo(xv.y);
    let xh1 = bf16_hi(xv.y);
    gd = fma(f32((pg >> 8u) & 15u) - 8.0, xl1, gd);
    gd = fma(f32((pg >> 12u) & 15u) - 8.0, xh1, gd);
    ud = fma(f32((pu >> 8u) & 15u) - 8.0, xl1, ud);
    ud = fma(f32((pu >> 12u) & 15u) - 8.0, xh1, ud);
    let xl2 = bf16_lo(xv.z);
    let xh2 = bf16_hi(xv.z);
    gd = fma(f32((pg >> 16u) & 15u) - 8.0, xl2, gd);
    gd = fma(f32((pg >> 20u) & 15u) - 8.0, xh2, gd);
    ud = fma(f32((pu >> 16u) & 15u) - 8.0, xl2, ud);
    ud = fma(f32((pu >> 20u) & 15u) - 8.0, xh2, ud);
    let xl3 = bf16_lo(xv.w);
    let xh3 = bf16_hi(xv.w);
    gd = fma(f32((pg >> 24u) & 15u) - 8.0, xl3, gd);
    gd = fma(f32((pg >> 28u) & 15u) - 8.0, xh3, gd);
    ud = fma(f32((pu >> 24u) & 15u) - 8.0, xl3, ud);
    ud = fma(f32((pu >> 28u) & 15u) - 8.0, xh3, ud);
    return vec2<f32>(gd, ud);
}

fn g4w_group_dot_xv4_one_vec4_per_mat_group_same_fma_order(
    wg_words: u32,
    sbase_g: u32,
    xg_words: u32,
    from_w2: bool
) -> f32 {
    var si_word: u32;
    if (from_w2) {
        si_word = g4w_ws2[sbase_g >> 1u];
    } else {
        si_word = g4w_ws[sbase_g >> 1u];
    }
    let scale = bf16_decode(u16_at(si_word, sbase_g));
    var pv4: vec4<u32>;
    if (from_w2) {
        pv4 = g4w_w2[wg_words >> 2u];
    } else {
        pv4 = g4w_w[wg_words >> 2u];
    }
    let xb = xg_words >> 2u;
    var gdot = 0.0;
    gdot = g4w_word_dot4_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
        pv4.x, g4w_x[xb], gdot);
    gdot = g4w_word_dot4_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
        pv4.y, g4w_x[xb + 1u], gdot);
    gdot = g4w_word_dot4_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
        pv4.z, g4w_x[xb + 2u], gdot);
    gdot = g4w_word_dot4_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
        pv4.w, g4w_x[xb + 3u], gdot);
    return scale * gdot;
}

fn g4w_row_acc_xv4(
    e: u32,
    row: u32,
    live: bool,
    lane: u32,
    lane_stride: u32,
    xbase: u32,
    from_w2: bool
) -> f32 {
    let row_words = g4w_p.groups * 4u;
    let wbase = select(0u, e * g4w_p.w_e_stride_words + row * row_words, live);
    let sbase = e * g4w_p.s_e_stride_elems + row * g4w_p.groups;
    let groups = select(0u, g4w_p.groups, live);
    var acc = 0.0;
    for (var g = lane; g < groups; g = g + lane_stride) {
        acc = acc + g4w_group_dot_xv4_one_vec4_per_mat_group_same_fma_order(
            wbase + g * 4u, sbase + g, xbase + g * 16u, from_w2
        );
    }
    return acc;
}

fn g4w_group_dot2_xv4_single_x_pass_bitwise_per_mat(
    wg_words: u32,
    sbase_g: u32,
    xg_words: u32
) -> vec2<f32> {
    let sg = bf16_decode(u16_at(g4w_ws[sbase_g >> 1u], sbase_g));
    let su = bf16_decode(u16_at(g4w_ws2[sbase_g >> 1u], sbase_g));
    let pgv = g4w_w[wg_words >> 2u];
    let puv = g4w_w2[wg_words >> 2u];
    let xb = xg_words >> 2u;
    var d = vec2<f32>(0.0, 0.0);
    d = g4w_word_dot4x2_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
        pgv.x, puv.x, g4w_x[xb], d.x, d.y);
    d = g4w_word_dot4x2_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
        pgv.y, puv.y, g4w_x[xb + 1u], d.x, d.y);
    d = g4w_word_dot4x2_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
        pgv.z, puv.z, g4w_x[xb + 2u], d.x, d.y);
    d = g4w_word_dot4x2_straightline_naga30_injects_no_loop_bound_counter_same_fma_order(
        pgv.w, puv.w, g4w_x[xb + 3u], d.x, d.y);
    return vec2<f32>(sg * d.x, su * d.y);
}

@compute @workgroup_size(256)
fn g4m_gemv_w4_gu_gelu_fusedx_sglat_xv4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let slot = wid.z;
    let e = g4w_sel[slot];
    let pair = wid.x + wid.y * g4w_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4w_p.n_rows;
    let xbase = slot * g4w_p.x_slot_stride_words;

    let row_words = g4w_p.groups * 4u;
    let wbase = select(0u, e * g4w_p.w_e_stride_words + row * row_words, live);
    let sbase = e * g4w_p.s_e_stride_elems + row * g4w_p.groups;
    let groups = select(0u, g4w_p.groups, live);
    var gacc = 0.0;
    var uacc = 0.0;
    for (var g = lane; g < groups; g = g + 128u) {
        let d2 = g4w_group_dot2_xv4_single_x_pass_bitwise_per_mat(
            wbase + g * 4u, sbase + g, xbase + g * 16u
        );
        gacc = gacc + d2.x;
        uacc = uacc + d2.y;
    }
    g4w_red[tid] = gacc;
    g4w_red2[tid] = uacc;
    workgroupBarrier();
    let hb = tid & 128u;
    var rg = 0.0;
    var ru = 0.0;
    if (lane < 32u) {
        rg = (g4w_red[hb + lane] + g4w_red[hb + lane + 64u])
            + (g4w_red[hb + lane + 32u] + g4w_red[hb + lane + 96u]);
        ru = (g4w_red2[hb + lane] + g4w_red2[hb + lane + 64u])
            + (g4w_red2[hb + lane + 32u] + g4w_red2[hb + lane + 96u]);
    }
    rg = rg + subgroupShuffleXor(rg, 16u);
    rg = rg + subgroupShuffleXor(rg, 8u);
    rg = rg + subgroupShuffleXor(rg, 4u);
    rg = rg + subgroupShuffleXor(rg, 2u);
    rg = rg + subgroupShuffleXor(rg, 1u);
    ru = ru + subgroupShuffleXor(ru, 16u);
    ru = ru + subgroupShuffleXor(ru, 8u);
    ru = ru + subgroupShuffleXor(ru, 4u);
    ru = ru + subgroupShuffleXor(ru, 2u);
    ru = ru + subgroupShuffleXor(ru, 1u);
    if (tid == 128u) {
        g4w_hi2 = vec2<f32>(rg, ru);
    }
    workgroupBarrier();
    if (tid == 0u) {
        var g_hi = 0.0;
        var u_hi = 0.0;
        if (row + 1u < g4w_p.n_rows) {
            g_hi = g4w_hi2.x;
            u_hi = g4w_hi2.y;
        }
        let gw = bf16_pack(rg, g_hi);
        let uw = bf16_pack(ru, u_hi);
        let a0 = bf16_decode(bf16_encode(g4w_gelu(bf16_lo(gw)))) * bf16_lo(uw);
        let a1 = bf16_decode(bf16_encode(g4w_gelu(bf16_hi(gw)))) * bf16_hi(uw);
        g4w_y[slot * g4w_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(a0, a1);
    }
}

@compute @workgroup_size(256)
fn g4m_gemv_w4_down_combine_expar_sglat_xv4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let rhalf = tid >> 7u;
    let es = (tid >> 5u) & 3u;
    let lane = tid & 31u;
    let block = wid.x + wid.y * g4w_p.groups_x;
    let row = block * 2u + rhalf;
    let live = row < g4w_p.n_rows;

    for (var jj = 0u; jj < 2u; jj = jj + 1u) {
        let j = es + jj * 4u;
        var acc = 0.0;
        if (j < g4w_p.k_top) {
            let e = g4w_sel[j];
            let xbase = j * g4w_p.x_slot_stride_words;
            acc = g4w_row_acc_xv4(e, row, live, lane, 32u, xbase, false);
        }
        acc = acc + subgroupShuffleXor(acc, 16u);
        acc = acc + subgroupShuffleXor(acc, 8u);
        acc = acc + subgroupShuffleXor(acc, 4u);
        acc = acc + subgroupShuffleXor(acc, 2u);
        acc = acc + subgroupShuffleXor(acc, 1u);
        if (lane == 0u && j < g4w_p.k_top) {
            g4w_dots[rhalf * 8u + j] = acc;
        }
    }
    workgroupBarrier();
    if (tid == (rhalf << 7u)) {
        var comb = 0.0;
        for (var j = 0u; j < g4w_p.k_top; j = j + 1u) {
            let dot_b = bf16_decode(bf16_encode(g4w_dots[rhalf * 8u + j]));
            comb = fma(dot_b, g4w_cw[j], comb);
        }
        g4w_gu[rhalf] = comb;
    }
    workgroupBarrier();
    if (tid == 0u && live) {
        var hi = 0.0;
        if (row + 1u < g4w_p.n_rows) {
            hi = g4w_gu[1];
        }
        g4w_y[row >> 1u] = bf16_pack(g4w_gu[0], hi);
    }
}
