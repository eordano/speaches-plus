struct G4i8guParams {
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

@group(0) @binding(0) var<storage, read> g8gu_wg: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> g8gu_x: array<vec4<u32>>;
@group(0) @binding(2) var<uniform> g8gu_p: G4i8guParams;
@group(0) @binding(3) var<storage, read_write> g8gu_y: array<u32>;
@group(0) @binding(4) var<storage, read> g8gu_sg: array<u32>;
@group(0) @binding(5) var<storage, read> g8gu_wu: array<vec4<u32>>;
@group(0) @binding(6) var<storage, read> g8gu_su: array<u32>;

var<workgroup> g8gu_red: array<f32, 256>;
var<workgroup> g8gu_gpair: array<f32, 2>;

fn g8gu_gelu(x: f32) -> f32 {
    let c = 0.7978845608028654;
    let t = nv_tanhf(c * (x + 0.044715 * x * x * x));
    return 0.5 * x * (1.0 + t);
}

fn g8gu_dot4(word: u32, xw0: u32, xw1: u32, acc_in: f32) -> f32 {
    var acc = acc_in;
    acc = fma(int8_decode(word, 0u), bf16_lo(xw0), acc);
    acc = fma(int8_decode(word, 1u), bf16_hi(xw0), acc);
    acc = fma(int8_decode(word, 2u), bf16_lo(xw1), acc);
    acc = fma(int8_decode(word, 3u), bf16_hi(xw1), acc);
    return acc;
}

fn g8gu_dot_row(row: u32, live: bool, lane: u32, from_up: bool) -> f32 {
    let wbase = select(0u, row * g8gu_p.w_row_words, live);
    let sbase = select(0u, row * g8gu_p.s_row_elems, live);
    let kv = select(0u, g8gu_p.k_words, live);
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + 128u) {
        var wv: vec4<u32>;
        if (from_up) {
            wv = g8gu_wu[wbase + v];
        } else {
            wv = g8gu_wg[wbase + v];
        }
        let xa = g8gu_x[g8gu_p.x_off_words + 2u * v];
        let xb = g8gu_x[g8gu_p.x_off_words + 2u * v + 1u];
        let si = sbase + (v >> 1u);
        var sw: u32;
        if (from_up) {
            sw = g8gu_su[si >> 1u];
        } else {
            sw = g8gu_sg[si >> 1u];
        }
        let scale = bf16_decode(u16_at(sw, si));
        var d = 0.0;
        d = g8gu_dot4(wv.x, xa.x, xa.y, d);
        d = g8gu_dot4(wv.y, xa.z, xa.w, d);
        d = g8gu_dot4(wv.z, xb.x, xb.y, d);
        d = g8gu_dot4(wv.w, xb.z, xb.w, d);
        acc = fma(scale, d, acc);
    }
    return acc;
}

@compute @workgroup_size(256)
fn g4m_gemv_i8_v4_gu_gelu(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * g8gu_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g8gu_p.n_rows;

    g8gu_red[tid] = g8gu_dot_row(row, live, lane, false);
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g8gu_red[tid] = g8gu_red[tid] + g8gu_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        g8gu_gpair[0] = g8gu_red[0] * g8gu_p.alpha;
        g8gu_gpair[1] = g8gu_red[128] * g8gu_p.alpha;
    }
    workgroupBarrier();

    g8gu_red[tid] = g8gu_dot_row(row, live, lane, true);
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g8gu_red[tid] = g8gu_red[tid] + g8gu_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let gw = bf16_pack(g8gu_gpair[0], g8gu_gpair[1]);
        var hi_u = 0.0;
        if (row + 1u < g8gu_p.n_rows) {
            hi_u = g8gu_red[128] * g8gu_p.alpha;
        }
        let uw = bf16_pack(g8gu_red[0] * g8gu_p.alpha, hi_u);
        let a0 = bf16_decode(bf16_encode(g8gu_gelu(bf16_lo(gw)))) * bf16_lo(uw);
        let a1 = bf16_decode(bf16_encode(g8gu_gelu(bf16_hi(gw)))) * bf16_hi(uw);
        g8gu_y[g8gu_p.y_off_words + (row >> 1u)] = bf16_pack(a0, a1);
    }
}
