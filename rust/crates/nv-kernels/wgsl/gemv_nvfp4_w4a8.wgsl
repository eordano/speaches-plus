struct W48Params {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    w_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> w48_w4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> w48_ws: array<u32>;
@group(0) @binding(2) var<storage, read> w48_xq4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read> w48_xs: array<f32>;
@group(0) @binding(4) var<uniform> w48_p: W48Params;
@group(0) @binding(5) var<storage, read_write> w48_y: array<u32>;
@group(0) @binding(6) var<storage, read> w48_xin4: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read_write> w48_xq_out: array<u32>;
@group(0) @binding(8) var<storage, read_write> w48_xs_out: array<f32>;

const W48_WG: u32 = 128u;
const W48_SGS: u32 = W48_WG / 32u;
const W48_LANES: u32 = 32u;
const W48_QWG: u32 = 256u;
const W48_EPILOGUE_HALF_UNDOES_THE_I8MAP_DOUBLING_OF_E2M1: f32 = 0.5;

fn w48_i8map(s: u32) -> u32 {
    let k = s & 0x07070707u;
    let hm = ((k >> 2u) & 0x01010101u) * 255u;
    let e7 = (k & (k >> 1u) & (k >> 2u)) & 0x01010101u;
    let m = k + ((k & 0x03030303u) & hm) + (e7 << 1u);
    let sb = (s & ((k + 0x07070707u) & 0x08080808u)) >> 3u;
    return (m ^ (sb * 255u)) + sb;
}

fn w48_bfly(acc: f32) -> f32 {
    var a = acc;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

var<workgroup> w48q_amax: array<f32, W48_QWG>;

fn w48_amax4(w: vec4<u32>) -> f32 {
    let a = max(abs(bf16_lo(w.x)), abs(bf16_hi(w.x)));
    let b = max(abs(bf16_lo(w.y)), abs(bf16_hi(w.y)));
    let c = max(abs(bf16_lo(w.z)), abs(bf16_hi(w.z)));
    let e = max(abs(bf16_lo(w.w)), abs(bf16_hi(w.w)));
    return max(max(a, b), max(c, e));
}

@compute @workgroup_size(W48_QWG)
fn w48_quantize_x_evenodd(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wgc: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let nvec = w48_p.k_blocks * 2u;
    var m = 0.0;
    for (var i = tid; i < nvec; i = i + W48_QWG) {
        m = max(m, w48_amax4(w48_xin4[i]));
    }
    w48q_amax[tid] = m;
    workgroupBarrier();
    var s = W48_QWG >> 1u;
    while (s > 0u) {
        if (tid < s) {
            w48q_amax[tid] = max(w48q_amax[tid], w48q_amax[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    let d = w48q_amax[0] / 127.0;
    let dinv = select(0.0, 1.0 / d, d != 0.0);
    if (wid.x == 0u && tid == 0u) {
        w48_xs_out[0] = d;
    }
    let stride = wgc.x * W48_QWG;
    for (var g = wid.x * W48_QWG + tid; g < nvec; g = g + stride) {
        let w = w48_xin4[g];
        var ev = 0u;
        var od = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let ww = w[j];
            let qe = i32(round(clamp(bf16_lo(ww) * dinv, -127.0, 127.0)));
            let qo = i32(round(clamp(bf16_hi(ww) * dinv, -127.0, 127.0)));
            ev = ev | ((u32(qe) & 0xffu) << (8u * j));
            od = od | ((u32(qo) & 0xffu) << (8u * j));
        }
        w48_xq_out[2u * g] = ev;
        w48_xq_out[2u * g + 1u] = od;
    }
}

struct W48LoraFoldParams {
    rank: u32,
    n_tiles: u32,
    part_base: u32,
    b_base: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(9) var<storage, read> w48_lora_b: array<u32>;
@group(0) @binding(10) var<storage, read> w48_lora_partials: array<f32>;
@group(0) @binding(11) var<uniform> w48_lp: W48LoraFoldParams;

const W48_LORA_FOLD_MAX_RANK_64_MATCHES_FUSED_MAX_RANK: u32 = 64u;

var<workgroup> w48_lora_h: array<f32, W48_LORA_FOLD_MAX_RANK_64_MATCHES_FUSED_MAX_RANK>;

fn w48_lora_row_delta(row: u32) -> f32 {
    var d = 0.0;
    let e0 = w48_lp.b_base + row * w48_lp.rank;
    for (var r = 0u; r < w48_lp.rank; r = r + 1u) {
        let e = e0 + r;
        let w = w48_lora_b[e >> 1u];
        d = fma(w48_lora_h[r], select(bf16_lo(w), bf16_hi(w), (e & 1u) != 0u), d);
    }
    return d;
}

@compute @workgroup_size(W48_WG)
fn gemv_nvfp4_w4a8_mrow2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row0 = (wid.x + wid.y * w48_p.groups_x) * (W48_SGS * 2u) + sgid * 2u;
    let row1 = row0 + 1u;
    let pairs = w48_p.k_blocks >> 1u;
    let stride = w48_p.k_blocks >> 1u;
    let live0 = row0 < w48_p.n_rows;
    let live1 = row1 < w48_p.n_rows;
    let wb0 = select(0u, row0 * stride, live0);
    let wb1 = select(0u, row1 * stride, live1);
    let sr0 = select(0u, row0, live0);
    let sr1 = select(0u, row1, live1);

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var p = lane; p < pairs; p = p + W48_LANES) {
        let xa = w48_xq4[2u * p];
        let xc = w48_xq4[2u * p + 1u];
        let b0 = p << 1u;

        let wv0 = w48_w4[wb0 + p];
        let wsi0 = nvfp4_scale_byte_index(sr0, b0, w48_p.k_tiles);
        let wsw0 = w48_ws[wsi0 >> 2u];
        let a0 = dot4I8Packed(w48_i8map(wv0.x), xa.x)
            + dot4I8Packed(w48_i8map(wv0.x >> 4u), xa.y)
            + dot4I8Packed(w48_i8map(wv0.y), xa.z)
            + dot4I8Packed(w48_i8map(wv0.y >> 4u), xa.w);
        let a1 = dot4I8Packed(w48_i8map(wv0.z), xc.x)
            + dot4I8Packed(w48_i8map(wv0.z >> 4u), xc.y)
            + dot4I8Packed(w48_i8map(wv0.w), xc.z)
            + dot4I8Packed(w48_i8map(wv0.w >> 4u), xc.w);
        acc0 = fma(ue4m3_decode(byte_at(wsw0, wsi0)), f32(a0), acc0);
        acc0 = fma(ue4m3_decode(byte_at(wsw0, wsi0 + 1u)), f32(a1), acc0);

        let wv1 = w48_w4[wb1 + p];
        let wsi1 = nvfp4_scale_byte_index(sr1, b0, w48_p.k_tiles);
        let wsw1 = w48_ws[wsi1 >> 2u];
        let b0d = dot4I8Packed(w48_i8map(wv1.x), xa.x)
            + dot4I8Packed(w48_i8map(wv1.x >> 4u), xa.y)
            + dot4I8Packed(w48_i8map(wv1.y), xa.z)
            + dot4I8Packed(w48_i8map(wv1.y >> 4u), xa.w);
        let b1d = dot4I8Packed(w48_i8map(wv1.z), xc.x)
            + dot4I8Packed(w48_i8map(wv1.z >> 4u), xc.y)
            + dot4I8Packed(w48_i8map(wv1.w), xc.z)
            + dot4I8Packed(w48_i8map(wv1.w >> 4u), xc.w);
        acc1 = fma(ue4m3_decode(byte_at(wsw1, wsi1)), f32(b0d), acc1);
        acc1 = fma(ue4m3_decode(byte_at(wsw1, wsi1 + 1u)), f32(b1d), acc1);
    }

    let t0 = w48_bfly(acc0);
    let t1 = w48_bfly(acc1);
    let ep = (w48_p.alpha * w48_xs[0]) * W48_EPILOGUE_HALF_UNDOES_THE_I8MAP_DOUBLING_OF_E2M1;
    if (lane == 0u && live0) {
        var word = bf16_encode(t0 * ep) & 0xffffu;
        if (live1) {
            word = word | (bf16_encode(t1 * ep) << 16u);
        }
        w48_y[row0 >> 1u] = word;
    }
}

@compute @workgroup_size(W48_WG)
fn gemv_nvfp4_w4a8_mrow2_lora_fold(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    if (lid.x < w48_lp.rank) {
        var s = 0.0;
        for (var t = 0u; t < w48_lp.n_tiles; t = t + 1u) {
            s = s + w48_lora_partials[w48_lp.part_base + t * w48_lp.rank + lid.x];
        }
        w48_lora_h[lid.x] = s * w48_lp.scale;
    }
    workgroupBarrier();

    let row0 = (wid.x + wid.y * w48_p.groups_x) * (W48_SGS * 2u) + sgid * 2u;
    let row1 = row0 + 1u;
    let pairs = w48_p.k_blocks >> 1u;
    let stride = w48_p.k_blocks >> 1u;
    let live0 = row0 < w48_p.n_rows;
    let live1 = row1 < w48_p.n_rows;
    let wb0 = select(0u, row0 * stride, live0);
    let wb1 = select(0u, row1 * stride, live1);
    let sr0 = select(0u, row0, live0);
    let sr1 = select(0u, row1, live1);

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var p = lane; p < pairs; p = p + W48_LANES) {
        let xa = w48_xq4[2u * p];
        let xc = w48_xq4[2u * p + 1u];
        let b0 = p << 1u;

        let wv0 = w48_w4[wb0 + p];
        let wsi0 = nvfp4_scale_byte_index(sr0, b0, w48_p.k_tiles);
        let wsw0 = w48_ws[wsi0 >> 2u];
        let a0 = dot4I8Packed(w48_i8map(wv0.x), xa.x)
            + dot4I8Packed(w48_i8map(wv0.x >> 4u), xa.y)
            + dot4I8Packed(w48_i8map(wv0.y), xa.z)
            + dot4I8Packed(w48_i8map(wv0.y >> 4u), xa.w);
        let a1 = dot4I8Packed(w48_i8map(wv0.z), xc.x)
            + dot4I8Packed(w48_i8map(wv0.z >> 4u), xc.y)
            + dot4I8Packed(w48_i8map(wv0.w), xc.z)
            + dot4I8Packed(w48_i8map(wv0.w >> 4u), xc.w);
        acc0 = fma(ue4m3_decode(byte_at(wsw0, wsi0)), f32(a0), acc0);
        acc0 = fma(ue4m3_decode(byte_at(wsw0, wsi0 + 1u)), f32(a1), acc0);

        let wv1 = w48_w4[wb1 + p];
        let wsi1 = nvfp4_scale_byte_index(sr1, b0, w48_p.k_tiles);
        let wsw1 = w48_ws[wsi1 >> 2u];
        let b0d = dot4I8Packed(w48_i8map(wv1.x), xa.x)
            + dot4I8Packed(w48_i8map(wv1.x >> 4u), xa.y)
            + dot4I8Packed(w48_i8map(wv1.y), xa.z)
            + dot4I8Packed(w48_i8map(wv1.y >> 4u), xa.w);
        let b1d = dot4I8Packed(w48_i8map(wv1.z), xc.x)
            + dot4I8Packed(w48_i8map(wv1.z >> 4u), xc.y)
            + dot4I8Packed(w48_i8map(wv1.w), xc.z)
            + dot4I8Packed(w48_i8map(wv1.w >> 4u), xc.w);
        acc1 = fma(ue4m3_decode(byte_at(wsw1, wsi1)), f32(b0d), acc1);
        acc1 = fma(ue4m3_decode(byte_at(wsw1, wsi1 + 1u)), f32(b1d), acc1);
    }

    let t0 = w48_bfly(acc0);
    let t1 = w48_bfly(acc1);
    let ep = (w48_p.alpha * w48_xs[0]) * W48_EPILOGUE_HALF_UNDOES_THE_I8MAP_DOUBLING_OF_E2M1;
    if (lane == 0u && live0) {
        let y0 = t0 * ep;
        let d0 = w48_lora_row_delta(row0);
        var word = bf16_encode(select(y0, y0 + d0, d0 != 0.0)) & 0xffffu;
        if (live1) {
            let y1 = t1 * ep;
            let d1 = w48_lora_row_delta(row1);
            word = word | (bf16_encode(select(y1, y1 + d1, d1 != 0.0)) << 16u);
        }
        w48_y[row0 >> 1u] = word;
    }
}
