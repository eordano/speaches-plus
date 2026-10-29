
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

@group(0) @binding(10) var<storage, read> g4w_w: array<u32>;
@group(0) @binding(11) var<storage, read> g4w_ws: array<u32>;
@group(0) @binding(12) var<storage, read> g4w_x: array<u32>;
@group(0) @binding(14) var<uniform> g4w_p: G4wqParams;
@group(0) @binding(15) var<storage, read_write> g4w_y: array<u32>;
@group(0) @binding(16) var<storage, read> g4w_sel: array<u32>;

var<workgroup> g4w_red: array<f32, 256>;

@compute @workgroup_size(256)
fn g4m_gemv_w4(
    #def b3
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
    #end
    let row_words = g4w_p.groups * 4u;
    let wbase = select(0u, e * g4w_p.w_e_stride_words + row * row_words, live);
    let sbase = e * g4w_p.s_e_stride_elems + row * g4w_p.groups;
    let xbase = slot * g4w_p.x_slot_stride_words;
    let groups = select(0u, g4w_p.groups, live);

    var acc = 0.0;
    for (var g = lane; g < groups; g = g + 128u) {
        #def b4
        let si = sbase + g;
        let scale = bf16_decode(u16_at(g4w_ws[si >> 1u], si));
        var gdot = 0.0;
        let wg = wbase + g * 4u;
        let xg = xbase + g * 16u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let pv = g4w_w[wg + j];
            for (var i = 0u; i < 4u; i = i + 1u) {
                let xw = g4w_x[xg + j * 4u + i];
                let q0 = f32((pv >> (8u * i)) & 15u) - 8.0;
                let q1 = f32((pv >> (8u * i + 4u)) & 15u) - 8.0;
                gdot = fma(q0, bf16_lo(xw), gdot);
                gdot = fma(q1, bf16_hi(xw), gdot);
            }
        }
        acc = fma(scale, gdot, acc);
    }
    g4w_red[tid] = acc;
    workgroupBarrier();
        #end
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        let lo = g4w_red[0];
        var hi = 0.0;
        if (row + 1u < g4w_p.n_rows) {
            hi = g4w_red[128];
        }
        g4w_y[slot * g4w_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

@compute @workgroup_size(256)
fn g4m_gemv_w4_r8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let sub = tid >> 5u;
    let lane = tid & 31u;
    let slot = wid.z;
    let e = g4w_sel[slot];
    let block = wid.x + wid.y * g4w_p.groups_x;
    let row = block * 8u + sub;
    let live = row < g4w_p.n_rows;
    let row_words = g4w_p.groups * 4u;
    let wbase = select(0u, e * g4w_p.w_e_stride_words + row * row_words, live);
    let sbase = e * g4w_p.s_e_stride_elems + row * g4w_p.groups;
    let xbase = slot * g4w_p.x_slot_stride_words;
    let groups = select(0u, g4w_p.groups, live);

    var acc = 0.0;
    for (var g = lane; g < groups; g = g + 32u) {
        #use b4
    for (var stride = 16u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u && (sub & 1u) == 0u && live) {
        let lo = g4w_red[tid];
        var hi = 0.0;
        if (row + 1u < g4w_p.n_rows) {
            hi = g4w_red[tid + 32u];
        }
        g4w_y[slot * g4w_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

@group(0) @binding(17) var<storage, read> g4w_w2: array<u32>;
@group(0) @binding(18) var<storage, read> g4w_ws2: array<u32>;
@group(0) @binding(19) var<storage, read> g4w_cw: array<f32>;

var<workgroup> g4w_gu: array<f32, 8>;

fn g4w_gelu(x: f32) -> f32 {
    let c = 0.7978845608028654;
    let t = nv_tanhf(c * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1.0 + t);
}

fn g4w_group_dot(wbase_g: u32, sbase_g: u32, xg: u32, from_w2: bool) -> f32 {
    var si_word: u32;
    if (from_w2) {
        si_word = g4w_ws2[sbase_g >> 1u];
    } else {
        si_word = g4w_ws[sbase_g >> 1u];
    }
    let scale = bf16_decode(u16_at(si_word, sbase_g));
    var gdot = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        var pv: u32;
        if (from_w2) {
            pv = g4w_w2[wbase_g + j];
        } else {
            pv = g4w_w[wbase_g + j];
        }
        for (var i = 0u; i < 4u; i = i + 1u) {
            let xw = g4w_x[xg + j * 4u + i];
            let q0 = f32((pv >> (8u * i)) & 15u) - 8.0;
            let q1 = f32((pv >> (8u * i + 4u)) & 15u) - 8.0;
            gdot = fma(q0, bf16_lo(xw), gdot);
            gdot = fma(q1, bf16_hi(xw), gdot);
        }
    }
    return scale * gdot;
}

fn g4w_row_acc(
    #def b9
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
    #end
    let groups = select(0u, g4w_p.groups, live);
    var acc = 0.0;
    for (var g = lane; g < groups; g = g + lane_stride) {
        acc = acc + g4w_group_dot(wbase + g * 4u, sbase + g, xbase + g * 16u, from_w2);
    }
    return acc;
}

@compute @workgroup_size(256)
fn g4m_gemv_w4_gu_gelu(
    #use b3
    let xbase = slot * g4w_p.x_slot_stride_words;

    g4w_red[tid] = g4w_row_acc(e, row, live, lane, 128u, xbase, false);
    #def b6
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        g4w_gu[0] = g4w_red[0];
        g4w_gu[1] = g4w_red[128];
    }
    workgroupBarrier();
    #end

    g4w_red[tid] = g4w_row_acc(e, row, live, lane, 128u, xbase, true);
    #def b1
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        var g_hi = 0.0;
        var u_hi = 0.0;
        if (row + 1u < g4w_p.n_rows) {
            g_hi = g4w_gu[1];
            u_hi = g4w_red[128];
        }
        let gw = bf16_pack(g4w_gu[0], g_hi);
        let uw = bf16_pack(g4w_red[0], u_hi);
        let a0 = bf16_decode(bf16_encode(g4w_gelu(bf16_lo(gw)))) * bf16_lo(uw);
        let a1 = bf16_decode(bf16_encode(g4w_gelu(bf16_hi(gw)))) * bf16_hi(uw);
        g4w_y[slot * g4w_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(a0, a1);
    }
}
    #end

@compute @workgroup_size(256)
fn g4m_gemv_w4_r8_gu_gelu(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let sub = tid >> 5u;
    let lane = tid & 31u;
    let slot = wid.z;
    let e = g4w_sel[slot];
    #def b2
    let block = wid.x + wid.y * g4w_p.groups_x;
    let row = block * 8u + sub;
    let live = row < g4w_p.n_rows;
    let xbase = slot * g4w_p.x_slot_stride_words;

    g4w_red[tid] = g4w_row_acc(e, row, live, lane, 32u, xbase, false);
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        g4w_gu[sub] = g4w_red[tid];
    }
    workgroupBarrier();

    g4w_red[tid] = g4w_row_acc(e, row, live, lane, 32u, xbase, true);
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (lane == 0u && (sub & 1u) == 0u && live) {
        var g_hi = 0.0;
        var u_hi = 0.0;
        if (row + 1u < g4w_p.n_rows) {
            g_hi = g4w_gu[sub + 1u];
            u_hi = g4w_red[tid + 32u];
        }
        let gw = bf16_pack(g4w_gu[sub], g_hi);
        let uw = bf16_pack(g4w_red[tid], u_hi);
        let a0 = bf16_decode(bf16_encode(g4w_gelu(bf16_lo(gw)))) * bf16_lo(uw);
        let a1 = bf16_decode(bf16_encode(g4w_gelu(bf16_hi(gw)))) * bf16_hi(uw);
        g4w_y[slot * g4w_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(a0, a1);
    }
}
    #end

@compute @workgroup_size(256)
fn g4m_gemv_w4_r8_down_combine(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let sub = tid >> 5u;
    let lane = tid & 31u;
    let block = wid.x + wid.y * g4w_p.groups_x;
    let row = block * 8u + sub;
    let live = row < g4w_p.n_rows;

    var comb = 0.0;
    for (var j = 0u; j < g4w_p.k_top; j = j + 1u) {
        let e = g4w_sel[j];
        let xbase = j * g4w_p.x_slot_stride_words;
        g4w_red[tid] = g4w_row_acc(e, row, live, lane, 32u, xbase, false);
        #def b5
        #def b8
        workgroupBarrier();
        for (var stride = 16u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
            }
            workgroupBarrier();
        }
        let dot_b = bf16_decode(bf16_encode(g4w_red[tid & 0xe0u]));
        #end
        comb = fma(dot_b, g4w_cw[j], comb);
        #def b7
        workgroupBarrier();
    }
    if (lane == 0u) {
        g4w_gu[sub] = comb;
    }
    workgroupBarrier();
    if (lane == 0u && (sub & 1u) == 0u && live) {
        var hi = 0.0;
        if (row + 1u < g4w_p.n_rows) {
            hi = g4w_gu[sub + 1u];
        }
        g4w_y[row >> 1u] = bf16_pack(g4w_gu[sub], hi);
    }
}
        #end
        #end

var<workgroup> g4w_xbuf: array<u32, 3072>;

fn g4w_row_acc_coal_consecutive_lanes_read_consecutive_packed_words(
    #use b9
    let words = select(0u, row_words, live);
    var acc = 0.0;
    for (var idx = lane; idx < words; idx = idx + lane_stride) {
        let g = idx >> 2u;
        let j = idx & 3u;
        var si_word: u32;
        if (from_w2) {
            si_word = g4w_ws2[(sbase + g) >> 1u];
        } else {
            si_word = g4w_ws[(sbase + g) >> 1u];
        }
        let scale = bf16_decode(u16_at(si_word, sbase + g));
        var pv: u32;
        if (from_w2) {
            pv = g4w_w2[wbase + idx];
        } else {
            pv = g4w_w[wbase + idx];
        }
        var d = 0.0;
        let xg = xbase + g * 16u + j * 4u;
        for (var i = 0u; i < 4u; i = i + 1u) {
            let xw = g4w_xbuf[xg + i];
            let q0 = f32((pv >> (8u * i)) & 15u) - 8.0;
            let q1 = f32((pv >> (8u * i + 4u)) & 15u) - 8.0;
            d = fma(q0, bf16_lo(xw), d);
            d = fma(q1, bf16_hi(xw), d);
        }
        acc = fma(scale, d, acc);
    }
    return acc;
}

fn g4w_preload_x_words(tid: u32, n_words: u32) {
    for (var w = tid; w < n_words; w = w + 256u) {
        g4w_xbuf[w] = g4w_x[w];
    }
    workgroupBarrier();
}

@compute @workgroup_size(256)
fn g4m_gemv_w4_gu_gelu_coal(
    #use b3
    let xbase = slot * g4w_p.x_slot_stride_words;
    g4w_preload_x_words(tid, xbase + g4w_p.groups * 16u);

    g4w_red[tid] = g4w_row_acc_coal_consecutive_lanes_read_consecutive_packed_words(
        e, row, live, lane, 128u, xbase, false
    );
    #use b6

    g4w_red[tid] = g4w_row_acc_coal_consecutive_lanes_read_consecutive_packed_words(
        e, row, live, lane, 128u, xbase, true
    );
    #use b1

@compute @workgroup_size(256)
fn g4m_gemv_w4_r8_down_combine_coal(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let sub = tid >> 5u;
    let lane = tid & 31u;
    let block = wid.x + wid.y * g4w_p.groups_x;
    let row = block * 8u + sub;
    let live = row < g4w_p.n_rows;
    g4w_preload_x_words(tid, g4w_p.k_top * g4w_p.x_slot_stride_words);

    var comb = 0.0;
    for (var j = 0u; j < g4w_p.k_top; j = j + 1u) {
        let e = g4w_sel[j];
        let xbase = j * g4w_p.x_slot_stride_words;
        g4w_red[tid] = g4w_row_acc_coal_consecutive_lanes_read_consecutive_packed_words(
            e, row, live, lane, 32u, xbase, false
        );
        #use b5

@group(0) @binding(20) var<storage, read> g4w_logits: array<f32>;
@group(0) @binding(21) var<storage, read> g4w_pes: array<f32>;

var<workgroup> g4w_rank_id: array<u32, 256>;
var<workgroup> g4w_rank_w: array<f32, 16>;

fn g4w_rank_experts_mirroring_router_topk_par(tid: u32) {
    let ne = g4w_p.n_experts;
    if (tid < ne) {
        let v = g4w_logits[tid];
        var r = 0u;
        for (var j = 0u; j < ne; j = j + 1u) {
            let w = g4w_logits[j];
            if (w > v || (w == v && j < tid)) {
                r = r + 1u;
            }
        }
        if (r < g4w_p.k_top) {
            g4w_rank_id[r] = tid;
        }
    }
    workgroupBarrier();
}

fn g4w_topk_weights_mirroring_router_topk_par(tid: u32) {
    if (tid == 0u) {
        var m = g4w_logits[g4w_rank_id[0]];
        for (var j = 1u; j < g4w_p.k_top; j = j + 1u) {
            m = max(m, g4w_logits[g4w_rank_id[j]]);
        }
        var ssum = 0.0;
        for (var j = 0u; j < g4w_p.k_top; j = j + 1u) {
            let e = exp(g4w_logits[g4w_rank_id[j]] - m);
            g4w_rank_w[j] = e;
            ssum = ssum + e;
        }
        for (var j = 0u; j < g4w_p.k_top; j = j + 1u) {
            g4w_rank_w[j] = g4w_rank_w[j] / ssum * g4w_pes[g4w_rank_id[j]];
        }
    }
    workgroupBarrier();
}

@compute @workgroup_size(256)
fn g4m_gemv_w4_gu_gelu_rank(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    g4w_rank_experts_mirroring_router_topk_par(tid);
    let half = tid >> 7u;
    let lane = tid & 127u;
    let slot = wid.z;
    let e = g4w_rank_id[slot];
    let pair = wid.x + wid.y * g4w_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4w_p.n_rows;
    let xbase = slot * g4w_p.x_slot_stride_words;

    g4w_red[tid] = g4w_row_acc(e, row, live, lane, 128u, xbase, false);
    #use b6

    g4w_red[tid] = g4w_row_acc(e, row, live, lane, 128u, xbase, true);
    #use b1

@compute @workgroup_size(256)
fn g4m_gemv_w4_r8_gu_gelu_rank(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    g4w_rank_experts_mirroring_router_topk_par(tid);
    let sub = tid >> 5u;
    let lane = tid & 31u;
    let slot = wid.z;
    let e = g4w_rank_id[slot];
    #use b2

@compute @workgroup_size(256)
fn g4m_gemv_w4_r8_down_combine_rank(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    g4w_rank_experts_mirroring_router_topk_par(tid);
    g4w_topk_weights_mirroring_router_topk_par(tid);
    let sub = tid >> 5u;
    let lane = tid & 31u;
    let block = wid.x + wid.y * g4w_p.groups_x;
    let row = block * 8u + sub;
    let live = row < g4w_p.n_rows;

    var comb = 0.0;
    for (var j = 0u; j < g4w_p.k_top; j = j + 1u) {
        let e = g4w_rank_id[j];
        let xbase = j * g4w_p.x_slot_stride_words;
        g4w_red[tid] = g4w_row_acc(e, row, live, lane, 32u, xbase, false);
        #use b8
        comb = fma(dot_b, g4w_rank_w[j], comb);
        #use b7

var<workgroup> g4w_red2: array<f32, 256>;

fn g4w_group_dot2_single_x_pass_bitwise_per_mat(
    wbase_g: u32,
    sbase_g: u32,
    xg: u32
) -> vec2<f32> {
    let sg = bf16_decode(u16_at(g4w_ws[sbase_g >> 1u], sbase_g));
    let su = bf16_decode(u16_at(g4w_ws2[sbase_g >> 1u], sbase_g));
    var gd = 0.0;
    var ud = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let pg = g4w_w[wbase_g + j];
        let pu = g4w_w2[wbase_g + j];
        for (var i = 0u; i < 4u; i = i + 1u) {
            let xw = g4w_x[xg + j * 4u + i];
            let xl = bf16_lo(xw);
            let xh = bf16_hi(xw);
            gd = fma(f32((pg >> (8u * i)) & 15u) - 8.0, xl, gd);
            gd = fma(f32((pg >> (8u * i + 4u)) & 15u) - 8.0, xh, gd);
            ud = fma(f32((pu >> (8u * i)) & 15u) - 8.0, xl, ud);
            ud = fma(f32((pu >> (8u * i + 4u)) & 15u) - 8.0, xh, ud);
        }
    }
    return vec2<f32>(sg * gd, su * ud);
}

@compute @workgroup_size(256)
fn g4m_gemv_w4_gu_gelu_fusedx(
    #use b3
    let xbase = slot * g4w_p.x_slot_stride_words;

    let row_words = g4w_p.groups * 4u;
    let wbase = select(0u, e * g4w_p.w_e_stride_words + row * row_words, live);
    let sbase = e * g4w_p.s_e_stride_elems + row * g4w_p.groups;
    let groups = select(0u, g4w_p.groups, live);
    var gacc = 0.0;
    var uacc = 0.0;
    for (var g = lane; g < groups; g = g + 128u) {
        let d2 = g4w_group_dot2_single_x_pass_bitwise_per_mat(
            wbase + g * 4u, sbase + g, xbase + g * 16u
        );
        gacc = gacc + d2.x;
        uacc = uacc + d2.y;
    }
    g4w_red[tid] = gacc;
    g4w_red2[tid] = uacc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
            g4w_red2[tid] = g4w_red2[tid] + g4w_red2[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        var g_hi = 0.0;
        var u_hi = 0.0;
        if (row + 1u < g4w_p.n_rows) {
            g_hi = g4w_red[128];
            u_hi = g4w_red2[128];
        }
        let gw = bf16_pack(g4w_red[0], g_hi);
        let uw = bf16_pack(g4w_red2[0], u_hi);
        let a0 = bf16_decode(bf16_encode(g4w_gelu(bf16_lo(gw)))) * bf16_lo(uw);
        let a1 = bf16_decode(bf16_encode(g4w_gelu(bf16_hi(gw)))) * bf16_hi(uw);
        g4w_y[slot * g4w_p.y_slot_stride_words + (row >> 1u)] = bf16_pack(a0, a1);
    }
}

var<workgroup> g4w_dots: array<f32, 16>;

@compute @workgroup_size(256)
fn g4m_gemv_w4_down_combine_expar(
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
            acc = g4w_row_acc(e, row, live, lane, 32u, xbase, false);
        }
        g4w_red[tid] = acc;
        workgroupBarrier();
        for (var stride = 16u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
            }
            workgroupBarrier();
        }
        if (lane == 0u && j < g4w_p.k_top) {
            g4w_dots[rhalf * 8u + j] = g4w_red[tid];
        }
        workgroupBarrier();
    }
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
