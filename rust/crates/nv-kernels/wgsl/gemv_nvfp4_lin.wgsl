struct GemvParams {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    w_row_words: u32,
    groups_x: u32,
    ws_row_stride: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> gemv_w_packed: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> gemv_w_scales: array<u32>;
@group(0) @binding(2) var<storage, read> gemv_x_packed: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> gemv_x_scales: array<u32>;
@group(0) @binding(4) var<uniform> gemv_params: GemvParams;
@group(0) @binding(5) var<storage, read_write> gemv_y: array<u32>;
@group(0) @binding(6) var<storage, read> gemv_w_scales_lin: array<u32>;
@group(0) @binding(7) var<storage, read> gemv_w4: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> gemv_x_i8: array<vec4<u32>>;

const LIN_STRIDE: u32 = 256u;
const V3_ROWS_PER_WG: u32 = 4u;
const LIN_ROWS_PER_WG: u32 = 4u;

fn lin_butterfly(acc: f32) -> f32 {
    var a = acc;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn lin_acc(w_vec_base: u32, ws_base: u32, blocks: u32, kb0: u32) -> f32 {
    var acc = 0.0;
    for (var kb = kb0; kb < blocks; kb = kb + LIN_STRIDE) {
        let si = ws_base + kb;
        let ws = byte_at(gemv_w_scales_lin[si >> 2u], si);
        let xs = byte_at(gemv_x_scales[kb >> 2u], kb);
        let block_scale = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
        let wv = gemv_w_packed[w_vec_base + kb];
        let xv = gemv_x_packed[kb];
        var dot = 0.0;
        dot = gemv_dot8(wv.x, xv.x, dot);
        dot = gemv_dot8(wv.y, xv.y, dot);
        acc = fma(block_scale, dot, acc);
    }
    return acc;
}

fn swz_acc(w_vec_base: u32, scale_row: u32, blocks: u32, kb0: u32) -> f32 {
    var acc = 0.0;
    for (var kb = kb0; kb < blocks; kb = kb + LIN_STRIDE) {
        let ws_idx = nvfp4_scale_byte_index(scale_row, kb, gemv_params.k_tiles);
        let ws = byte_at(gemv_w_scales[ws_idx >> 2u], ws_idx);
        let xs = byte_at(gemv_x_scales[kb >> 2u], kb);
        let block_scale = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
        let wv = gemv_w_packed[w_vec_base + kb];
        let xv = gemv_x_packed[kb];
        var dot = 0.0;
        dot = gemv_dot8(wv.x, xv.x, dot);
        dot = gemv_dot8(wv.y, xv.y, dot);
        acc = fma(block_scale, dot, acc);
    }
    return acc;
}

fn noscale_acc(w_vec_base: u32, blocks: u32, kb0: u32) -> f32 {
    var acc = 0.0;
    for (var kb = kb0; kb < blocks; kb = kb + LIN_STRIDE) {
        let xs = byte_at(gemv_x_scales[kb >> 2u], kb);
        let block_scale = gemv_ue4m3_decode(0x30u) * gemv_ue4m3_decode(xs);
        let wv = gemv_w_packed[w_vec_base + kb];
        let xv = gemv_x_packed[kb];
        var dot = 0.0;
        dot = gemv_dot8(wv.x, xv.x, dot);
        dot = gemv_dot8(wv.y, xv.y, dot);
        acc = fma(block_scale, dot, acc);
    }
    return acc;
}

fn nodec_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    let d = dot4I8Packed(ww, xw) + dot4I8Packed(ww >> 4u, xw >> 4u);
    return dot_in + f32(d) * 0.25;
}

fn xpre_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    let d = dot4I8Packed(gemv_i8map(ww), xw) + dot4I8Packed(gemv_i8map(ww >> 4u), xw >> 4u);
    return dot_in + f32(d) * 0.25;
}

fn nodec_acc(w_vec_base: u32, ws_base: u32, blocks: u32, kb0: u32) -> f32 {
    var acc = 0.0;
    for (var kb = kb0; kb < blocks; kb = kb + LIN_STRIDE) {
        let si = ws_base + kb;
        let ws = byte_at(gemv_w_scales_lin[si >> 2u], si);
        let xs = byte_at(gemv_x_scales[kb >> 2u], kb);
        let block_scale = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
        let wv = gemv_w_packed[w_vec_base + kb];
        let xv = gemv_x_packed[kb];
        var dot = 0.0;
        dot = nodec_dot8(wv.x, xv.x, dot);
        dot = nodec_dot8(wv.y, xv.y, dot);
        acc = fma(block_scale, dot, acc);
    }
    return acc;
}

fn xpre_acc(w_vec_base: u32, ws_base: u32, blocks: u32, kb0: u32) -> f32 {
    var acc = 0.0;
    for (var kb = kb0; kb < blocks; kb = kb + LIN_STRIDE) {
        let si = ws_base + kb;
        let ws = byte_at(gemv_w_scales_lin[si >> 2u], si);
        let xs = byte_at(gemv_x_scales[kb >> 2u], kb);
        let block_scale = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
        let wv = gemv_w_packed[w_vec_base + kb];
        let xv = gemv_x_packed[kb];
        var dot = 0.0;
        dot = xpre_dot8(wv.x, xv.x, dot);
        dot = xpre_dot8(wv.y, xv.y, dot);
        acc = fma(block_scale, dot, acc);
    }
    return acc;
}

fn v3_dot8(ww: u32, x_lo: u32, x_hi: u32, dot_in: f32) -> f32 {
    let d = dot4I8Packed(gemv_i8map(ww), x_lo) + dot4I8Packed(gemv_i8map(ww >> 4u), x_hi);
    return dot_in + f32(d) * 0.25;
}

@compute @workgroup_size(128)
fn gemv_nvfp4_v3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * V3_ROWS_PER_WG + sgid;
    let row_live = row < gemv_params.n_rows;
    let pairs = select(0u, gemv_params.k_blocks >> 1u, row_live);
    let w4_base = select(0u, row * (gemv_params.w_row_words >> 2u), row_live);
    let ws_base = select(0u, row * gemv_params.ws_row_stride, row_live);

    var acc = 0.0;
    for (var v = lane; v < pairs; v = v + 32u) {
        let kb = v << 1u;
        let wv = gemv_w4[w4_base + v];
        let xa = gemv_x_i8[v << 1u];
        let xb = gemv_x_i8[(v << 1u) + 1u];
        let si = ws_base + kb;
        let wsw = gemv_w_scales_lin[si >> 2u];
        let xsw = gemv_x_scales[kb >> 2u];
        let s0 = gemv_ue4m3_decode(byte_at(wsw, si)) * gemv_ue4m3_decode(byte_at(xsw, kb));
        let s1 = gemv_ue4m3_decode(byte_at(wsw, si + 1u)) * gemv_ue4m3_decode(byte_at(xsw, kb + 1u));
        var d0 = 0.0;
        d0 = v3_dot8(wv.x, xa.x, xa.y, d0);
        d0 = v3_dot8(wv.y, xa.z, xa.w, d0);
        acc = fma(s0, d0, acc);
        var d1 = 0.0;
        d1 = v3_dot8(wv.z, xb.x, xb.y, d1);
        d1 = v3_dot8(wv.w, xb.z, xb.w, d1);
        acc = fma(s1, d1, acc);
    }

    let total = lin_butterfly(acc);
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}

fn v3_dot8_nodec(ww: u32, x_lo: u32, x_hi: u32, dot_in: f32) -> f32 {
    let d = dot4I8Packed(ww, x_lo) + dot4I8Packed(ww >> 4u, x_hi);
    return dot_in + f32(d) * 0.25;
}

@compute @workgroup_size(128)
fn gemv_nvfp4_v3_nodec(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * V3_ROWS_PER_WG + sgid;
    let row_live = row < gemv_params.n_rows;
    let pairs = select(0u, gemv_params.k_blocks >> 1u, row_live);
    let w4_base = select(0u, row * (gemv_params.w_row_words >> 2u), row_live);
    let ws_base = select(0u, row * gemv_params.ws_row_stride, row_live);

    var acc = 0.0;
    for (var v = lane; v < pairs; v = v + 32u) {
        let kb = v << 1u;
        let wv = gemv_w4[w4_base + v];
        let xa = gemv_x_i8[v << 1u];
        let xb = gemv_x_i8[(v << 1u) + 1u];
        let si = ws_base + kb;
        let wsw = gemv_w_scales_lin[si >> 2u];
        let xsw = gemv_x_scales[kb >> 2u];
        let s0 = gemv_ue4m3_decode(byte_at(wsw, si)) * gemv_ue4m3_decode(byte_at(xsw, kb));
        let s1 = gemv_ue4m3_decode(byte_at(wsw, si + 1u)) * gemv_ue4m3_decode(byte_at(xsw, kb + 1u));
        var d0 = 0.0;
        d0 = v3_dot8_nodec(wv.x, xa.x, xa.y, d0);
        d0 = v3_dot8_nodec(wv.y, xa.z, xa.w, d0);
        acc = fma(s0, d0, acc);
        var d1 = 0.0;
        d1 = v3_dot8_nodec(wv.z, xb.x, xb.y, d1);
        d1 = v3_dot8_nodec(wv.w, xb.z, xb.w, d1);
        acc = fma(s1, d1, acc);
    }

    let total = lin_butterfly(acc);
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}

@compute @workgroup_size(128)
fn gemv_nvfp4_v3_stream(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * V3_ROWS_PER_WG + sgid;
    let row_live = row < gemv_params.n_rows;
    let pairs = select(0u, gemv_params.k_blocks >> 1u, row_live);
    let w4_base = select(0u, row * (gemv_params.w_row_words >> 2u), row_live);
    let ws_base = select(0u, row * gemv_params.ws_row_stride, row_live);

    var bits = 0u;
    for (var v = lane; v < pairs; v = v + 32u) {
        let kb = v << 1u;
        let wv = gemv_w4[w4_base + v];
        let xa = gemv_x_i8[v << 1u];
        let xb = gemv_x_i8[(v << 1u) + 1u];
        let si = ws_base + kb;
        bits = bits ^ wv.x ^ wv.y ^ wv.z ^ wv.w;
        bits = bits ^ xa.x ^ xa.y ^ xa.z ^ xa.w ^ xb.x ^ xb.y ^ xb.z ^ xb.w;
        bits = bits ^ gemv_w_scales_lin[si >> 2u] ^ gemv_x_scales[kb >> 2u];
    }

    let total = lin_butterfly(f32(bits & 1023u));
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}

@compute @workgroup_size(128)
fn gemv_nvfp4_nodec_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * LIN_ROWS_PER_WG + sgid;
    let row_live = row < gemv_params.n_rows;
    let blocks = select(0u, gemv_params.k_blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);
    let ws_base = select(0u, row * gemv_params.ws_row_stride, row_live);

    let s0 = lin_butterfly(nodec_acc(w_vec_base, ws_base, blocks, 0u * 32u + lane));
    let s1 = lin_butterfly(nodec_acc(w_vec_base, ws_base, blocks, 1u * 32u + lane));
    let s2 = lin_butterfly(nodec_acc(w_vec_base, ws_base, blocks, 2u * 32u + lane));
    let s3 = lin_butterfly(nodec_acc(w_vec_base, ws_base, blocks, 3u * 32u + lane));
    let s4 = lin_butterfly(nodec_acc(w_vec_base, ws_base, blocks, 4u * 32u + lane));
    let s5 = lin_butterfly(nodec_acc(w_vec_base, ws_base, blocks, 5u * 32u + lane));
    let s6 = lin_butterfly(nodec_acc(w_vec_base, ws_base, blocks, 6u * 32u + lane));
    let s7 = lin_butterfly(nodec_acc(w_vec_base, ws_base, blocks, 7u * 32u + lane));

    let total = ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7));
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}

@compute @workgroup_size(128)
fn gemv_nvfp4_xpre_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * LIN_ROWS_PER_WG + sgid;
    let row_live = row < gemv_params.n_rows;
    let blocks = select(0u, gemv_params.k_blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);
    let ws_base = select(0u, row * gemv_params.ws_row_stride, row_live);

    let s0 = lin_butterfly(xpre_acc(w_vec_base, ws_base, blocks, 0u * 32u + lane));
    let s1 = lin_butterfly(xpre_acc(w_vec_base, ws_base, blocks, 1u * 32u + lane));
    let s2 = lin_butterfly(xpre_acc(w_vec_base, ws_base, blocks, 2u * 32u + lane));
    let s3 = lin_butterfly(xpre_acc(w_vec_base, ws_base, blocks, 3u * 32u + lane));
    let s4 = lin_butterfly(xpre_acc(w_vec_base, ws_base, blocks, 4u * 32u + lane));
    let s5 = lin_butterfly(xpre_acc(w_vec_base, ws_base, blocks, 5u * 32u + lane));
    let s6 = lin_butterfly(xpre_acc(w_vec_base, ws_base, blocks, 6u * 32u + lane));
    let s7 = lin_butterfly(xpre_acc(w_vec_base, ws_base, blocks, 7u * 32u + lane));

    let total = ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7));
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}

@compute @workgroup_size(128)
fn gemv_nvfp4_lin_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * LIN_ROWS_PER_WG + sgid;
    let row_live = row < gemv_params.n_rows;
    let blocks = select(0u, gemv_params.k_blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);
    let ws_base = select(0u, row * gemv_params.ws_row_stride, row_live);

    let s0 = lin_butterfly(lin_acc(w_vec_base, ws_base, blocks, 0u * 32u + lane));
    let s1 = lin_butterfly(lin_acc(w_vec_base, ws_base, blocks, 1u * 32u + lane));
    let s2 = lin_butterfly(lin_acc(w_vec_base, ws_base, blocks, 2u * 32u + lane));
    let s3 = lin_butterfly(lin_acc(w_vec_base, ws_base, blocks, 3u * 32u + lane));
    let s4 = lin_butterfly(lin_acc(w_vec_base, ws_base, blocks, 4u * 32u + lane));
    let s5 = lin_butterfly(lin_acc(w_vec_base, ws_base, blocks, 5u * 32u + lane));
    let s6 = lin_butterfly(lin_acc(w_vec_base, ws_base, blocks, 6u * 32u + lane));
    let s7 = lin_butterfly(lin_acc(w_vec_base, ws_base, blocks, 7u * 32u + lane));

    let total = ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7));
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}

@compute @workgroup_size(128)
fn gemv_nvfp4_swz_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * LIN_ROWS_PER_WG + sgid;
    let row_live = row < gemv_params.n_rows;
    let blocks = select(0u, gemv_params.k_blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);
    let scale_row = select(0u, row, row_live);

    let s0 = lin_butterfly(swz_acc(w_vec_base, scale_row, blocks, 0u * 32u + lane));
    let s1 = lin_butterfly(swz_acc(w_vec_base, scale_row, blocks, 1u * 32u + lane));
    let s2 = lin_butterfly(swz_acc(w_vec_base, scale_row, blocks, 2u * 32u + lane));
    let s3 = lin_butterfly(swz_acc(w_vec_base, scale_row, blocks, 3u * 32u + lane));
    let s4 = lin_butterfly(swz_acc(w_vec_base, scale_row, blocks, 4u * 32u + lane));
    let s5 = lin_butterfly(swz_acc(w_vec_base, scale_row, blocks, 5u * 32u + lane));
    let s6 = lin_butterfly(swz_acc(w_vec_base, scale_row, blocks, 6u * 32u + lane));
    let s7 = lin_butterfly(swz_acc(w_vec_base, scale_row, blocks, 7u * 32u + lane));

    let total = ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7));
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}

@compute @workgroup_size(128)
fn gemv_nvfp4_noscale_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * LIN_ROWS_PER_WG + sgid;
    let row_live = row < gemv_params.n_rows;
    let blocks = select(0u, gemv_params.k_blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);

    let s0 = lin_butterfly(noscale_acc(w_vec_base, blocks, 0u * 32u + lane));
    let s1 = lin_butterfly(noscale_acc(w_vec_base, blocks, 1u * 32u + lane));
    let s2 = lin_butterfly(noscale_acc(w_vec_base, blocks, 2u * 32u + lane));
    let s3 = lin_butterfly(noscale_acc(w_vec_base, blocks, 3u * 32u + lane));
    let s4 = lin_butterfly(noscale_acc(w_vec_base, blocks, 4u * 32u + lane));
    let s5 = lin_butterfly(noscale_acc(w_vec_base, blocks, 5u * 32u + lane));
    let s6 = lin_butterfly(noscale_acc(w_vec_base, blocks, 6u * 32u + lane));
    let s7 = lin_butterfly(noscale_acc(w_vec_base, blocks, 7u * 32u + lane));

    let total = ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7));
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}
