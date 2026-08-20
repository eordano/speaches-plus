
struct GowGbParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    has_bias: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> gow_gb_w: array<u32>;
@group(0) @binding(1) var<storage, read> gow_gb_x: array<u32>;
@group(0) @binding(2) var<uniform> gow_gb_p: GowGbParams;
@group(0) @binding(3) var<storage, read_write> gow_gb_y: array<u32>;
@group(0) @binding(4) var<storage, read> gow_gb_b: array<u32>;

var<workgroup> gow_gb_red: array<f32, 256>;

fn gow_gb_bias(row: u32) -> f32 {
    if (gow_gb_p.has_bias == 0u) {
        return 0.0;
    }
    return bf16_decode(u16_at(gow_gb_b[row >> 1u], row));
}

@compute @workgroup_size(256)
fn gow_gemv_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * gow_gb_p.groups_x;
    let row = pair * 2u + half;
    let live = row < gow_gb_p.n_rows;
    let wbase = select(0u, row * gow_gb_p.w_row_words, live);
    let kw = select(0u, gow_gb_p.k_words, live);

    var acc = 0.0;
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = gow_gb_w[wbase + i];
        let xw = gow_gb_x[gow_gb_p.x_off_words + i];
        acc = fma(bf16_lo(ww), bf16_lo(xw), acc);
        acc = fma(bf16_hi(ww), bf16_hi(xw), acc);
    }
    gow_gb_red[tid] = acc;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            gow_gb_red[tid] = gow_gb_red[tid] + gow_gb_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (gow_gb_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            let v = gow_gb_red[tid] * gow_gb_p.alpha + gow_gb_bias(row);
            gow_gb_y[gow_gb_p.y_off_words + row] = bitcast<u32>(v);
        }
    } else if (tid == 0u) {
        let lo = gow_gb_red[0] * gow_gb_p.alpha + gow_gb_bias(row);
        var hi = 0.0;
        if (row + 1u < gow_gb_p.n_rows) {
            hi = gow_gb_red[128] * gow_gb_p.alpha + gow_gb_bias(row + 1u);
        }
        gow_gb_y[gow_gb_p.y_off_words + (row >> 1u)] = bf16_pack(lo, hi);
    }
}
