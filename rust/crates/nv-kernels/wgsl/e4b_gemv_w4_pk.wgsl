
struct G4wPkParams {
    dst_word_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct G4wSplitParams {
    q_rows: u32,
    kv_rows: u32,
    v_off: u32,
    pad0: u32,
};

@group(0) @binding(30) var<uniform> g4w_pk_params: G4wPkParams;
@group(0) @binding(31) var<storage, read_write> g4w_y_q: array<u32>;
@group(0) @binding(32) var<storage, read_write> g4w_y_k: array<u32>;
@group(0) @binding(33) var<storage, read_write> g4w_y_v: array<u32>;
@group(0) @binding(34) var<uniform> g4w_split_params: G4wSplitParams;

fn g4w_w4_pair_word(tid: u32, total: f32, hi_live: bool) -> u32 {
    let lo = bf16_encode(total) & 0xffffu;
    let hi = bf16_encode(w4a16_partial[tid + W4A16_LANES]) & 0xffffu;
    return lo | (select(0u, hi, hi_live) << 16u);
}

@compute @workgroup_size(256)
fn g4w_gemv_w4a16_block_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4A16_LANES - 1u);
    let warp = tid / W4A16_LANES;
    let row = (wid.x + wid.y * w4a16_params.groups_x) * W4A16_ROWS + warp;
    let live = row < w4a16_params.n_rows;
    let kv = select(0u, w4a16_params.k_elems >> 5u, live);
    let wbase = select(0u, row * w4a16_params.w_row_words, live);
    let sbase = select(0u, row * w4a16_params.scale_row_stride, live);
    let acc = w4a16_row_acc_block(wbase, sbase, kv, lane, w4a16_params.gs);
    let total = w4a16_lane_reduce(tid, lane, acc);
    if (lane == 0u && live && (warp & 1u) == 0u) {
        let word = g4w_w4_pair_word(tid, total, row + 1u < w4a16_params.n_rows);
        w4a16_y[g4w_pk_params.dst_word_off + (row >> 1u)] = word;
    }
}

@compute @workgroup_size(256)
fn g4w_gemv_w4a16_block_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4A16_LANES - 1u);
    let warp = tid / W4A16_LANES;
    let row = (wid.x + wid.y * w4a16_params.groups_x) * W4A16_ROWS + warp;
    let live = row < w4a16_params.n_rows;
    let kv = select(0u, w4a16_params.k_elems >> 5u, live);
    let wbase = select(0u, row * w4a16_params.w_row_words, live);
    let sbase = select(0u, row * w4a16_params.scale_row_stride, live);
    let acc = w4a16_row_acc_block(wbase, sbase, kv, lane, w4a16_params.gs);
    let total = w4a16_lane_reduce(tid, lane, acc);
    if (lane == 0u && live && (warp & 1u) == 0u) {
        let word = g4w_w4_pair_word(tid, total, row + 1u < w4a16_params.n_rows);
        if (row < g4w_split_params.q_rows) {
            g4w_y_q[row >> 1u] = word;
        } else {
            let kr = row - g4w_split_params.q_rows;
            if (kr < g4w_split_params.kv_rows) {
                g4w_y_k[kr >> 1u] = word;
            }
            if (row >= g4w_split_params.v_off) {
                let vr = row - g4w_split_params.v_off;
                if (vr < g4w_split_params.kv_rows) {
                    g4w_y_v[vr >> 1u] = word;
                }
            }
        }
    }
}

@compute @workgroup_size(256)
fn g4w_gemv_w4a16_v4_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4A16_LANES - 1u);
    let warp = tid / W4A16_LANES;
    let row = (wid.x + wid.y * w4a16_params.groups_x) * W4A16_ROWS + warp;
    let live = row < w4a16_params.n_rows;
    let kv = select(0u, w4a16_params.k_elems >> 5u, live);
    let wbase4 = select(0u, row * (w4a16_params.w_row_words >> 2u), live);
    let sbase = select(0u, row * w4a16_params.scale_row_stride, live);
    let acc = w4a16_row_acc_v4(wbase4, sbase, kv, lane, w4a16_params.gs);
    let total = w4a16_lane_reduce(tid, lane, acc);
    if (lane == 0u && live && (warp & 1u) == 0u) {
        let word = g4w_w4_pair_word(tid, total, row + 1u < w4a16_params.n_rows);
        w4a16_y[g4w_pk_params.dst_word_off + (row >> 1u)] = word;
    }
}

@compute @workgroup_size(256)
fn g4w_gemv_w4a16_v4_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4A16_LANES - 1u);
    let warp = tid / W4A16_LANES;
    let row = (wid.x + wid.y * w4a16_params.groups_x) * W4A16_ROWS + warp;
    let live = row < w4a16_params.n_rows;
    let kv = select(0u, w4a16_params.k_elems >> 5u, live);
    let wbase4 = select(0u, row * (w4a16_params.w_row_words >> 2u), live);
    let sbase = select(0u, row * w4a16_params.scale_row_stride, live);
    let acc = w4a16_row_acc_v4(wbase4, sbase, kv, lane, w4a16_params.gs);
    let total = w4a16_lane_reduce(tid, lane, acc);
    if (lane == 0u && live && (warp & 1u) == 0u) {
        let word = g4w_w4_pair_word(tid, total, row + 1u < w4a16_params.n_rows);
        if (row < g4w_split_params.q_rows) {
            g4w_y_q[row >> 1u] = word;
        } else {
            let kr = row - g4w_split_params.q_rows;
            if (kr < g4w_split_params.kv_rows) {
                g4w_y_k[kr >> 1u] = word;
            }
            if (row >= g4w_split_params.v_off) {
                let vr = row - g4w_split_params.v_off;
                if (vr < g4w_split_params.kv_rows) {
                    g4w_y_v[vr >> 1u] = word;
                }
            }
        }
    }
}

struct G4wMkParams {
    m: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    dst_word_off: u32,
};

@group(0) @binding(35) var<uniform> g4w_mk_params: G4wMkParams;

fn w4a16_dot8_xoff(pv: u32, kb: u32, xoff_words: u32, acc_in: f32) -> f32 {
    var acc = acc_in;
    let xb = (kb >> 1u) + xoff_words;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let xp = w4a16_x_pair(xb + i);
        acc = fma(w4a16_q(pv, 2u * i), xp.x, acc);
        acc = fma(w4a16_q(pv, 2u * i + 1u), xp.y, acc);
    }
    return acc;
}

fn g4w_w4_mk_rows(tid: u32, lane: u32, warp: u32, row: u32, live: bool,
                  accs: ptr<function, array<f32, 8>>) {
    let kv = select(0u, w4a16_params.k_elems >> 5u, live);
    let wbase = select(0u, row * w4a16_params.w_row_words, live);
    let sbase = select(0u, row * w4a16_params.scale_row_stride, live);
    let gs = w4a16_params.gs;
    let mm = g4w_mk_params.m;
    let xs = g4w_mk_params.x_stride_words;
    if (gs >= 32u) {
        for (var v = lane; v < kv; v = v + W4A16_LANES) {
            let kbase = v * 32u;
            let sc = w4a16_scale_at(sbase, kbase / gs);
            var blk: array<f32, 8>;
            for (var t = 0u; t < 8u; t = t + 1u) {
                blk[t] = 0.0;
            }
            for (var j = 0u; j < 4u; j = j + 1u) {
                let pv = w4a16_packed[wbase + v * 4u + j];
                let kb = kbase + j * 8u;
                for (var t = 0u; t < 8u; t = t + 1u) {
                    if (t < mm) {
                        blk[t] = w4a16_dot8_xoff(pv, kb, t * xs, blk[t]);
                    }
                }
            }
            for (var t = 0u; t < 8u; t = t + 1u) {
                if (t < mm) {
                    (*accs)[t] = fma(sc, blk[t], (*accs)[t]);
                }
            }
        }
    } else {
        for (var v = lane; v < kv; v = v + W4A16_LANES) {
            let kbase = v * 32u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let pv = w4a16_packed[wbase + v * 4u + j];
                let kb = kbase + j * 8u;
                let sc = w4a16_scale_at(sbase, kb / gs);
                for (var t = 0u; t < 8u; t = t + 1u) {
                    if (t < mm) {
                        let a = w4a16_dot8_xoff(pv, kb, t * xs, 0.0);
                        (*accs)[t] = fma(a, sc, (*accs)[t]);
                    }
                }
            }
        }
    }
}

fn g4w_w4_v4_mk_rows(tid: u32, lane: u32, warp: u32, row: u32, live: bool,
                     accs: ptr<function, array<f32, 8>>) {
    let kv = select(0u, w4a16_params.k_elems >> 5u, live);
    let wbase4 = select(0u, row * (w4a16_params.w_row_words >> 2u), live);
    let sbase = select(0u, row * w4a16_params.scale_row_stride, live);
    let gs = w4a16_params.gs;
    let mm = g4w_mk_params.m;
    let xs4 = g4w_mk_params.x_stride_words >> 2u;
    for (var v = lane; v < kv; v = v + W4A16_LANES) {
        let sc = w4a16_scale_at(sbase, (v << 5u) / gs);
        let wv = w4a16_packed4[wbase4 + v];
        for (var t = 0u; t < 8u; t = t + 1u) {
            if (t < mm) {
                (*accs)[t] = fma(sc, w4a16_dot32_v4(wv, t * xs4 + (v << 2u)), (*accs)[t]);
            }
        }
    }
}

fn g4w_w4_mk_store_pk(tid: u32, lane: u32, warp: u32, row: u32, live: bool,
                      accs: ptr<function, array<f32, 8>>) {
    let mm = g4w_mk_params.m;
    for (var t = 0u; t < 8u; t = t + 1u) {
        if (t >= mm) {
            break;
        }
        let total = w4a16_lane_reduce(tid, lane, (*accs)[t]);
        if (lane == 0u && live && (warp & 1u) == 0u) {
            let word = g4w_w4_pair_word(tid, total, row + 1u < w4a16_params.n_rows);
            w4a16_y[g4w_mk_params.dst_word_off + t * g4w_mk_params.y_stride_words + (row >> 1u)] = word;
        }
        workgroupBarrier();
    }
}

fn g4w_w4_mk_store_pk3(tid: u32, lane: u32, warp: u32, row: u32, live: bool,
                       accs: ptr<function, array<f32, 8>>) {
    let mm = g4w_mk_params.m;
    for (var t = 0u; t < 8u; t = t + 1u) {
        if (t >= mm) {
            break;
        }
        let total = w4a16_lane_reduce(tid, lane, (*accs)[t]);
        if (lane == 0u && live && (warp & 1u) == 0u) {
            let word = g4w_w4_pair_word(tid, total, row + 1u < w4a16_params.n_rows);
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

@compute @workgroup_size(256)
fn g4w_gemm_w4a16_block_mk_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4A16_LANES - 1u);
    let warp = tid / W4A16_LANES;
    let row = wid.x * W4A16_ROWS + warp;
    let live = row < w4a16_params.n_rows;
    var accs: array<f32, 8>;
    for (var t = 0u; t < 8u; t = t + 1u) {
        accs[t] = 0.0;
    }
    g4w_w4_mk_rows(tid, lane, warp, row, live, &accs);
    g4w_w4_mk_store_pk(tid, lane, warp, row, live, &accs);
}

@compute @workgroup_size(256)
fn g4w_gemm_w4a16_block_mk_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4A16_LANES - 1u);
    let warp = tid / W4A16_LANES;
    let row = wid.x * W4A16_ROWS + warp;
    let live = row < w4a16_params.n_rows;
    var accs: array<f32, 8>;
    for (var t = 0u; t < 8u; t = t + 1u) {
        accs[t] = 0.0;
    }
    g4w_w4_mk_rows(tid, lane, warp, row, live, &accs);
    g4w_w4_mk_store_pk3(tid, lane, warp, row, live, &accs);
}

@compute @workgroup_size(256)
fn g4w_gemm_w4a16_v4_mk_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4A16_LANES - 1u);
    let warp = tid / W4A16_LANES;
    let row = wid.x * W4A16_ROWS + warp;
    let live = row < w4a16_params.n_rows;
    var accs: array<f32, 8>;
    for (var t = 0u; t < 8u; t = t + 1u) {
        accs[t] = 0.0;
    }
    g4w_w4_v4_mk_rows(tid, lane, warp, row, live, &accs);
    g4w_w4_mk_store_pk(tid, lane, warp, row, live, &accs);
}

@compute @workgroup_size(256)
fn g4w_gemm_w4a16_v4_mk_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4A16_LANES - 1u);
    let warp = tid / W4A16_LANES;
    let row = wid.x * W4A16_ROWS + warp;
    let live = row < w4a16_params.n_rows;
    var accs: array<f32, 8>;
    for (var t = 0u; t < 8u; t = t + 1u) {
        accs[t] = 0.0;
    }
    g4w_w4_v4_mk_rows(tid, lane, warp, row, live, &accs);
    g4w_w4_mk_store_pk3(tid, lane, warp, row, live, &accs);
}
