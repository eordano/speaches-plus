
struct G4bLegacyParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    pad0: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
};

@group(0) @binding(0) var<storage, read> g4bl_w: array<u32>;
@group(0) @binding(1) var<storage, read> g4bl_x: array<u32>;
@group(0) @binding(2) var<uniform> g4bl_p: G4bLegacyParams;
@group(0) @binding(3) var<storage, read_write> g4bl_y: array<u32>;

var<workgroup> g4bl_red: array<f32, 256>;

@compute @workgroup_size(256)
fn g4m_gemv_bf16_legacy(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * g4bl_p.groups_x;
    let row = pair * 2u + half;
    let live = row < g4bl_p.n_rows;
    let wbase = select(0u, row * g4bl_p.w_row_words, live);
    let kw = select(0u, g4bl_p.k_words, live);

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = g4bl_w[wbase + i];
        let xw = g4bl_x[g4bl_p.x_off_words + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    g4bl_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            g4bl_red[tid] = g4bl_red[tid] + g4bl_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (g4bl_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            g4bl_y[g4bl_p.y_off_words + row] = bitcast<u32>(g4bl_red[tid] * g4bl_p.alpha);
        }
    } else if (tid == 0u) {
        let lo = g4bl_red[0] * g4bl_p.alpha;
        var hi = 0.0;
        if (row + 1u < g4bl_p.n_rows) {
            hi = g4bl_red[128] * g4bl_p.alpha;
        }
        g4bl_y[g4bl_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}
