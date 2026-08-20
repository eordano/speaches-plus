
struct G4wMkParams {
    m: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    dst_word_off: u32,
};

@group(0) @binding(35) var<uniform> g4w_mk_params: G4wMkParams;

@compute @workgroup_size(256)
fn g4w_gemm_bf16_mk_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = wid.x * GEMV_BF16_ROWS + warp;
    let live = row < gemv_bf16_params.n_rows;
    let kv = select(0u, gemv_bf16_params.k_elems >> 3u, live);
    let w_base = select(0u, row * gemv_bf16_params.w_row_words, live);
    let mm = g4w_mk_params.m;
    let xs = g4w_mk_params.x_stride_words;
    var acc: array<f32, 8>;
    for (var t = 0u; t < 8u; t = t + 1u) {
        acc[t] = 0.0;
    }
    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {
        let wo = w_base + (v << 2u);
        let xo = v << 2u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let ww = gemv_bf16_w[wo + j];
            let wl = bf16_lo(ww);
            let wh = bf16_hi(ww);
            for (var t = 0u; t < 8u; t = t + 1u) {
                if (t < mm) {
                    let xw = gemv_bf16_x[t * xs + xo + j];
                    acc[t] = acc[t] + (wl * bf16_lo(xw) + wh * bf16_hi(xw));
                }
            }
        }
    }
    for (var t = 0u; t < 8u; t = t + 1u) {
        if (t >= mm) {
            break;
        }
        let total = gemv_bf16_reduce(tid, lane, acc[t]);
        if (lane == 0u && live && (warp & 1u) == 0u) {
            let word = g4w_pair_word(tid, total, row + 1u < gemv_bf16_params.n_rows);
            gemv_bf16_y[g4w_mk_params.dst_word_off + t * g4w_mk_params.y_stride_words + (row >> 1u)] = word;
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn g4w_gemm_bf16_mk_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = wid.x * GEMV_BF16_ROWS + warp;
    let live = row < gemv_bf16_params.n_rows;
    let kv = select(0u, gemv_bf16_params.k_elems >> 3u, live);
    let w_base = select(0u, row * gemv_bf16_params.w_row_words, live);
    let mm = g4w_mk_params.m;
    let xs = g4w_mk_params.x_stride_words;
    var acc: array<f32, 8>;
    for (var t = 0u; t < 8u; t = t + 1u) {
        acc[t] = 0.0;
    }
    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {
        let wo = w_base + (v << 2u);
        let xo = v << 2u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let ww = gemv_bf16_w[wo + j];
            let wl = bf16_lo(ww);
            let wh = bf16_hi(ww);
            for (var t = 0u; t < 8u; t = t + 1u) {
                if (t < mm) {
                    let xw = gemv_bf16_x[t * xs + xo + j];
                    acc[t] = acc[t] + (wl * bf16_lo(xw) + wh * bf16_hi(xw));
                }
            }
        }
    }
    for (var t = 0u; t < 8u; t = t + 1u) {
        if (t >= mm) {
            break;
        }
        let total = gemv_bf16_reduce(tid, lane, acc[t]);
        if (lane == 0u && live && (warp & 1u) == 0u) {
            let word = g4w_pair_word(tid, total, row + 1u < gemv_bf16_params.n_rows);
            if (row < g4w_split_params.q_rows) {
                g4w_y_q[t * (g4w_split_params.q_rows >> 1u) + (row >> 1u)] = word;
            } else {
                let kr = row - g4w_split_params.q_rows;
                if (kr < g4w_split_params.kv_rows) {
                    g4w_y_k[t * (g4w_split_params.kv_rows >> 1u) + (kr >> 1u)] = word;
                }
                if (row >= g4w_split_params.v_off) {
                    let vr = row - g4w_split_params.v_off;
                    if (vr < g4w_split_params.kv_rows) {
                        g4w_y_v[t * (g4w_split_params.kv_rows >> 1u) + (vr >> 1u)] = word;
                    }
                }
            }
        }
        workgroupBarrier();
    }
}
