
struct G4wqParams {
    n_rows: u32,
    groups: u32,
    groups_x: u32,
    w_e_stride_words: u32,
    s_e_stride_elems: u32,
    x_slot_stride_words: u32,
    y_slot_stride_words: u32,
    k_top: u32,
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
    let row_words = g4w_p.groups * 4u;
    let wbase = select(0u, e * g4w_p.w_e_stride_words + row * row_words, live);
    let sbase = e * g4w_p.s_e_stride_elems + row * g4w_p.groups;
    let xbase = slot * g4w_p.x_slot_stride_words;
    let groups = select(0u, g4w_p.groups, live);

    var acc = 0.0;
    for (var g = lane; g < groups; g = g + 128u) {
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
        acc = acc + g4w_group_dot(wbase + g * 4u, sbase + g, xbase + g * 16u, from_w2);
    }
    return acc;
}

@compute @workgroup_size(256)
fn g4m_gemv_w4_gu_gelu(
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

    g4w_red[tid] = g4w_row_acc(e, row, live, lane, 128u, xbase, false);
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

    g4w_red[tid] = g4w_row_acc(e, row, live, lane, 128u, xbase, true);
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
        workgroupBarrier();
        for (var stride = 16u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                g4w_red[tid] = g4w_red[tid] + g4w_red[tid + stride];
            }
            workgroupBarrier();
        }
        let dot_b = bf16_decode(bf16_encode(g4w_red[tid & 0xe0u]));
        comb = fma(dot_b, g4w_cw[j], comb);
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
