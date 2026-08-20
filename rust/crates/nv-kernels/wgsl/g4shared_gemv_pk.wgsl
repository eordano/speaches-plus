
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

fn g4w_vec8_acc(row: u32, live: bool, lane: u32) -> f32 {
    let kv = select(0u, gemv_bf16_params.k_elems >> 3u, live);
    let w_base = select(0u, row * gemv_bf16_params.w_row_words, live);
    var acc = 0.0;
    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {
        let wo = w_base + (v << 2u);
        let xo = v << 2u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let ww = gemv_bf16_w[wo + j];
            let xw = gemv_bf16_x[xo + j];
            acc = acc + (bf16_lo(ww) * bf16_lo(xw) + bf16_hi(ww) * bf16_hi(xw));
        }
    }
    return acc;
}

fn g4w_pair_word(tid: u32, total: f32, hi_live: bool) -> u32 {
    let lo = bf16_encode(total) & 0xffffu;
    let hi = bf16_encode(gemv_bf16_partial[tid + GEMV_BF16_LANES]) & 0xffffu;
    return lo | (select(0u, hi, hi_live) << 16u);
}

@compute @workgroup_size(256)
fn g4w_gemv_bf16_vec8_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = gemv_bf16_row(wid, warp);
    let live = row < gemv_bf16_params.n_rows;
    let acc = g4w_vec8_acc(row, live, lane);
    let total = gemv_bf16_reduce(tid, lane, acc);
    if (lane == 0u && live && (warp & 1u) == 0u) {
        let word = g4w_pair_word(tid, total, row + 1u < gemv_bf16_params.n_rows);
        gemv_bf16_y[g4w_pk_params.dst_word_off + (row >> 1u)] = word;
    }
}

@compute @workgroup_size(256)
fn g4w_gemv_bf16_vec8_pk3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = gemv_bf16_row(wid, warp);
    let live = row < gemv_bf16_params.n_rows;
    let acc = g4w_vec8_acc(row, live, lane);
    let total = gemv_bf16_reduce(tid, lane, acc);
    if (lane == 0u && live && (warp & 1u) == 0u) {
        let word = g4w_pair_word(tid, total, row + 1u < gemv_bf16_params.n_rows);
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
