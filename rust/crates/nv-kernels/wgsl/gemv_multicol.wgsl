
const MC_NCOLS: u32 = 2u;

struct McParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    x_col_stride_words: u32,
    y_col_stride_words: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> mc_w: array<u32>;
@group(0) @binding(1) var<storage, read> mc_x: array<u32>;
@group(0) @binding(2) var<uniform> mc_p: McParams;
@group(0) @binding(3) var<storage, read_write> mc_y: array<u32>;

var<workgroup> mc_red: array<f32, 256u * MC_NCOLS>;

@compute @workgroup_size(256)
fn gemv_bf16_mc(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * mc_p.groups_x;
    let row = pair * 2u + half;
    let live = row < mc_p.n_rows;
    let wbase = select(0u, row * mc_p.w_row_words, live);
    let kw = select(0u, mc_p.k_words, live);

    var acc: array<f32, MC_NCOLS>;
    for (var c = 0u; c < MC_NCOLS; c = c + 1u) {
        acc[c] = 0.0;
    }
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = mc_w[wbase + i];
        let wlo = bf16_lo(ww);
        let whi = bf16_hi(ww);
        for (var c = 0u; c < MC_NCOLS; c = c + 1u) {
            let xw = mc_x[mc_p.x_off_words + c * mc_p.x_col_stride_words + i];
            acc[c] = fma(wlo, bf16_lo(xw), acc[c]);
            acc[c] = fma(whi, bf16_hi(xw), acc[c]);
        }
    }
    for (var c = 0u; c < MC_NCOLS; c = c + 1u) {
        mc_red[c * 256u + tid] = acc[c];
    }
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            for (var c = 0u; c < MC_NCOLS; c = c + 1u) {
                mc_red[c * 256u + tid] = mc_red[c * 256u + tid] + mc_red[c * 256u + tid + stride];
            }
        }
        workgroupBarrier();
    }

    if (mc_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            for (var c = 0u; c < MC_NCOLS; c = c + 1u) {
                let v = mc_red[c * 256u + tid] * mc_p.alpha;
                mc_y[mc_p.y_off_words + c * mc_p.y_col_stride_words + row] = bitcast<u32>(v);
            }
        }
    } else if (tid == 0u) {
        for (var c = 0u; c < MC_NCOLS; c = c + 1u) {
            let lo = mc_red[c * 256u] * mc_p.alpha;
            var hi = 0.0;
            if (row + 1u < mc_p.n_rows) {
                hi = mc_red[c * 256u + 128u] * mc_p.alpha;
            }
            mc_y[mc_p.y_off_words + c * mc_p.y_col_stride_words + (row >> 1u)] = bf16_pack(lo, hi);
        }
    }
}
