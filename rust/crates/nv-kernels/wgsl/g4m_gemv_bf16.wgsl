
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

var<workgroup> g4b_red: array<f32, 256>;

@compute @workgroup_size(256)
fn g4m_gemv_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
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
    g4b_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4b_red[tid] = g4b_red[tid] + g4b_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (g4b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            g4b_y[g4b_p.y_off_words + row] = bitcast<u32>(g4b_red[tid] * g4b_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = g4b_red[0] * g4b_p.alpha;
        var hi = 0.0;
        if (row + 1u < g4b_p.n_rows) {
            hi = g4b_red[128] * g4b_p.alpha;
        }
        g4b_y[g4b_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

@group(0) @binding(4) var<storage, read> g4b_w2: array<vec4<u32>>;

var<workgroup> g4b_gu: array<f32, 2>;

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

@compute @workgroup_size(256)
fn g4m_gemv_bf16_gu_gelu(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * g4b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4b_p.n_rows;

    g4b_red[tid] = g4b_dot_row(row, live, lane, false);
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4b_red[tid] = g4b_red[tid] + g4b_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        g4b_gu[0] = g4b_red[0] * g4b_p.alpha;
        g4b_gu[1] = g4b_red[128] * g4b_p.alpha;
    }
    workgroupBarrier();

    g4b_red[tid] = g4b_dot_row(row, live, lane, true);
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4b_red[tid] = g4b_red[tid] + g4b_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let gw = bf16_pack(g4b_gu[0], g4b_gu[1]);
        var hi_u = 0.0;
        if (row + 1u < g4b_p.n_rows) {
            hi_u = g4b_red[128] * g4b_p.alpha;
        }
        let uw = bf16_pack(g4b_red[0] * g4b_p.alpha, hi_u);
        let a0 = bf16_decode(bf16_encode(g4b_gelu(bf16_lo(gw)))) * bf16_lo(uw);
        let a1 = bf16_decode(bf16_encode(g4b_gelu(bf16_hi(gw)))) * bf16_hi(uw);
        g4b_y[g4b_p.y_off_words + (row >> 1u)] = bf16_pack(a0, a1);
    }
}

@compute @workgroup_size(256)
fn g4m_gemv_bf16_r8(
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

    var acc = 0.0;
    let wv = wbase >> 2u;
    let xv = g4b_p.x_off_words >> 2u;
    let kv = kw >> 2u;
    for (var v = lane; v < kv; v = v + 32u) {
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
    g4b_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 16u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4b_red[tid] = g4b_red[tid] + g4b_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (g4b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            g4b_y[g4b_p.y_off_words + row] = bitcast<u32>(g4b_red[tid] * g4b_p.alpha);
        }
    } else if (lane == 0u && (sub & 1u) == 0u && live) {
        let lo = g4b_red[tid] * g4b_p.alpha;
        var hi = 0.0;
        if (row + 1u < g4b_p.n_rows) {
            hi = g4b_red[tid + 32u] * g4b_p.alpha;
        }
        g4b_y[g4b_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}

@group(0) @binding(5) var<storage, read> g4b_rsw: array<u32>;

var<workgroup> g4b_xn: array<u32, 2048>;
var<workgroup> g4b_ns: f32;

@compute @workgroup_size(256)
fn g4m_gemv_bf16_rpnorm(
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
    g4b_red[tid] = local;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (tid < st) {
            g4b_red[tid] = g4b_red[tid] + g4b_red[tid + st];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        g4b_ns = inverseSqrt(g4b_red[0] / f32(g4b_p.pad1) + bitcast<f32>(g4b_p.pad2));
    }
    workgroupBarrier();
    let s = g4b_ns;
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
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4b_red[tid] = g4b_red[tid] + g4b_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (lane == 0u && live) {
        g4b_y[g4b_p.y_off_words + row] = bitcast<u32>(g4b_red[tid] * g4b_p.alpha);
    }
}

var<workgroup> g4b_red2: array<f32, 256>;

fn g4b_dot_row2_single_x_pass_bitwise_per_mat(row: u32, live: bool, lane: u32) -> vec2<f32> {
    let wbase = select(0u, row * g4b_p.w_row_words, live);
    let kw = select(0u, g4b_p.k_words, live);
    var gacc = 0.0;
    var uacc = 0.0;
    if (g4b_p.wide == 1u) {
        let wv = wbase >> 2u;
        let xv = g4b_p.x_off_words >> 2u;
        let kv = kw >> 2u;
        for (var v = lane; v < kv; v = v + 128u) {
            let wg = g4b_w[wv + v];
            let wu = g4b_w2[wv + v];
            let xw = g4b_x[xv + v];
            gacc = fma(bf16_lo(wg.x), bf16_lo(xw.x), gacc);
            gacc = fma(bf16_hi(wg.x), bf16_hi(xw.x), gacc);
            gacc = fma(bf16_lo(wg.y), bf16_lo(xw.y), gacc);
            gacc = fma(bf16_hi(wg.y), bf16_hi(xw.y), gacc);
            gacc = fma(bf16_lo(wg.z), bf16_lo(xw.z), gacc);
            gacc = fma(bf16_hi(wg.z), bf16_hi(xw.z), gacc);
            gacc = fma(bf16_lo(wg.w), bf16_lo(xw.w), gacc);
            gacc = fma(bf16_hi(wg.w), bf16_hi(xw.w), gacc);
            uacc = fma(bf16_lo(wu.x), bf16_lo(xw.x), uacc);
            uacc = fma(bf16_hi(wu.x), bf16_hi(xw.x), uacc);
            uacc = fma(bf16_lo(wu.y), bf16_lo(xw.y), uacc);
            uacc = fma(bf16_hi(wu.y), bf16_hi(xw.y), uacc);
            uacc = fma(bf16_lo(wu.z), bf16_lo(xw.z), uacc);
            uacc = fma(bf16_hi(wu.z), bf16_hi(xw.z), uacc);
            uacc = fma(bf16_lo(wu.w), bf16_lo(xw.w), uacc);
            uacc = fma(bf16_hi(wu.w), bf16_hi(xw.w), uacc);
        }
    } else {
        for (var i = lane; i < kw; i = i + 128u) {
            let wi = wbase + i;
            let xi = g4b_p.x_off_words + i;
            let wg = g4b_w[wi >> 2u][wi & 3u];
            let wu = g4b_w2[wi >> 2u][wi & 3u];
            let xw = g4b_x[xi >> 2u][xi & 3u];
            gacc = fma(bf16_lo(wg), bf16_lo(xw), gacc);
            gacc = fma(bf16_hi(wg), bf16_hi(xw), gacc);
            uacc = fma(bf16_lo(wu), bf16_lo(xw), uacc);
            uacc = fma(bf16_hi(wu), bf16_hi(xw), uacc);
        }
    }
    return vec2<f32>(gacc, uacc);
}

@compute @workgroup_size(256)
fn g4m_gemv_bf16_gu_gelu_fusedx(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * g4b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4b_p.n_rows;

    let d2 = g4b_dot_row2_single_x_pass_bitwise_per_mat(row, live, lane);
    g4b_red[tid] = d2.x;
    g4b_red2[tid] = d2.y;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4b_red[tid] = g4b_red[tid] + g4b_red[tid + stride];
            g4b_red2[tid] = g4b_red2[tid] + g4b_red2[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let gw = bf16_pack(g4b_red[0] * g4b_p.alpha, g4b_red[128] * g4b_p.alpha);
        var hi_u = 0.0;
        if (row + 1u < g4b_p.n_rows) {
            hi_u = g4b_red2[128] * g4b_p.alpha;
        }
        let uw = bf16_pack(g4b_red2[0] * g4b_p.alpha, hi_u);
        let a0 = bf16_decode(bf16_encode(g4b_gelu(bf16_lo(gw)))) * bf16_lo(uw);
        let a1 = bf16_decode(bf16_encode(g4b_gelu(bf16_hi(gw)))) * bf16_hi(uw);
        g4b_y[g4b_p.y_off_words + (row >> 1u)] = bf16_pack(a0, a1);
    }
}

@compute @workgroup_size(256)
fn g4m_gemv_bf16_acc4_ilp_reorders_the_wide_dot_sum(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * g4b_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4b_p.n_rows;
    let wbase = select(0u, row * g4b_p.w_row_words, live);
    let kw = select(0u, g4b_p.k_words, live);

    var a0 = 0.0;
    var a1 = 0.0;
    var a2 = 0.0;
    var a3 = 0.0;
    let wv = wbase >> 2u;
    let xv = g4b_p.x_off_words >> 2u;
    let kv = kw >> 2u;
    for (var v = lane; v < kv; v = v + 128u) {
        let ww = g4b_w[wv + v];
        let xw = g4b_x[xv + v];
        a0 = fma(bf16_lo(ww.x), bf16_lo(xw.x), a0);
        a0 = fma(bf16_hi(ww.x), bf16_hi(xw.x), a0);
        a1 = fma(bf16_lo(ww.y), bf16_lo(xw.y), a1);
        a1 = fma(bf16_hi(ww.y), bf16_hi(xw.y), a1);
        a2 = fma(bf16_lo(ww.z), bf16_lo(xw.z), a2);
        a2 = fma(bf16_hi(ww.z), bf16_hi(xw.z), a2);
        a3 = fma(bf16_lo(ww.w), bf16_lo(xw.w), a3);
        a3 = fma(bf16_hi(ww.w), bf16_hi(xw.w), a3);
    }
    g4b_red[tid] = (a0 + a1) + (a2 + a3);
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4b_red[tid] = g4b_red[tid] + g4b_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (g4b_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            g4b_y[g4b_p.y_off_words + row] = bitcast<u32>(g4b_red[tid] * g4b_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = g4b_red[0] * g4b_p.alpha;
        var hi = 0.0;
        if (row + 1u < g4b_p.n_rows) {
            hi = g4b_red[128] * g4b_p.alpha;
        }
        g4b_y[g4b_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}
