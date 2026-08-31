struct GemmW4A16AffineParams {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    m_live: u32,
    x_stride_words: u32,
    y_stride_words: u32,
};

@group(0) @binding(0) var<storage, read> w4affm_packed4: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> w4affm_scale: array<u32>;
@group(0) @binding(2) var<storage, read> w4affm_x4: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> w4affm_y: array<u32>;
@group(0) @binding(4) var<uniform> w4affm_params: GemmW4A16AffineParams;
@group(0) @binding(6) var<storage, read> w4affm_bias: array<u32>;

const W4AFFM_LANES: u32 = 32u;
const W4AFFM_ROWS: u32 = 8u;
const W4AFFM_MAX_M: u32 = 16u;
const W4AFFM_WORKGROUP_BYTES_TOTAL_VS_METAL_LIMIT_32768: u32 = 8192u;

var<workgroup> w4affm_xf_tile: array<vec4<f32>, 512>;
const W4AFFM_ONES: vec4<f32> = vec4<f32>(1.0, 1.0, 1.0, 1.0);

fn w4affm_lane_reduce(acc: f32) -> f32 {
    var a = acc;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn w4affm_row_acc(wbase4: u32, sbase: u32, kv: u32, ml: u32, lane: u32) -> array<f32, 16> {
    let xs4 = w4affm_params.x_stride_words >> 2u;
    var acc: array<f32, 16>;
    for (var t = 0u; t < W4AFFM_MAX_M; t = t + 1u) {
        acc[t] = 0.0;
    }

    for (var v = lane; v < kv; v = v + W4AFFM_LANES) {
        var wv = w4affm_packed4[wbase4 + v];
        let gi = sbase + ((v << 5u) / w4affm_params.gs);
        let sc = bf16_decode(w4affm_scale[gi]);
        let bi = bf16_decode(w4affm_bias[gi]);
        var qe: array<vec4<f32>, 4>;
        var qo: array<vec4<f32>, 4>;
        for (var c = 0u; c < 4u; c = c + 1u) {
            qe[c] = vec4<f32>(unpack4xU8(wv[c] & 0x0f0f0f0fu));
            qo[c] = vec4<f32>(unpack4xU8((wv[c] >> 4u) & 0x0f0f0f0fu));
        }
        let xb = v << 2u;
        for (var t = 0u; t < W4AFFM_MAX_M; t = t + 1u) {
            var qdot = 0.0;
            var xsum = 0.0;
            let xbase = min(t, ml - 1u) * xs4 + xb;
            for (var c = 0u; c < 4u; c = c + 1u) {
                let xw = w4affm_x4[xbase + c];
                let xe = bitcast<vec4<f32>>(xw << vec4<u32>(16u));
                let xo = bitcast<vec4<f32>>(xw & vec4<u32>(0xffff0000u));
                qdot = qdot + dot(qe[c], xe) + dot(qo[c], xo);
                xsum = xsum + dot(xe + xo, W4AFFM_ONES);
            }
            acc[t] = fma(sc, qdot, fma(bi, xsum, acc[t]));
        }
    }

    return acc;
}

@compute @workgroup_size(256)
fn gemm_w4a16_affine_mrows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4AFFM_LANES - 1u);
    let warp = tid / W4AFFM_LANES;
    let row = (wid.x + wid.y * w4affm_params.groups_x) * W4AFFM_ROWS + warp;
    let live = row < w4affm_params.n_rows;
    let kv = select(0u, w4affm_params.k_elems >> 5u, live);
    let wbase4 = select(0u, row * (w4affm_params.w_row_words >> 2u), live);
    let sbase = select(0u, row * w4affm_params.scale_row_stride, live);
    let ml = min(w4affm_params.m_live, W4AFFM_MAX_M);
    var acc = w4affm_row_acc(wbase4, sbase, kv, ml, lane);
    for (var t = 0u; t < W4AFFM_MAX_M; t = t + 1u) {
        if (t >= ml) {
            break;
        }
        let total = w4affm_lane_reduce(acc[t]);
        if (lane == 0u && live) {
            w4affm_y[t * w4affm_params.y_stride_words + row] = bf16_encode(total);
        }
    }
}

fn w4affm_xf_half(word2: vec2<u32>) -> vec4<f32> {
    let lo = bitcast<vec2<f32>>(word2 << vec2<u32>(16u));
    let hi = bitcast<vec2<f32>>(word2 & vec2<u32>(0xffff0000u));
    return vec4<f32>(lo.x, hi.x, lo.y, hi.y);
}

fn w4affm_v2_acc(
    tid: u32,
    row: u32,
    quad: u32,
    kv: u32,
    ml: u32,
    rw4: u32,
    sbase: u32
) -> vec4<f32> {
    var acc = vec4<f32>(0.0);
    let xs4 = w4affm_params.x_stride_words >> 2u;
    for (var v0 = 0u; v0 < kv; v0 = v0 + 2u) {
        let hb = ((v0 >> 1u) & 1u) * 256u;
        if (tid < 128u) {
            let t = tid >> 3u;
            let c2 = tid & 7u;
            let xw = w4affm_x4[min(t, ml - 1u) * xs4 + v0 * 4u + c2];
            let sl = hb + (c2 >> 2u) * 128u + (t << 3u) + (c2 & 3u) * 2u;
            w4affm_xf_tile[sl] = bitcast<vec4<f32>>(xw << vec4<u32>(16u));
            w4affm_xf_tile[sl + 1u] = bitcast<vec4<f32>>(xw & vec4<u32>(0xffff0000u));
        }
        workgroupBarrier();
        for (var blk = 0u; blk < 2u; blk = blk + 1u) {
        let v = v0 + blk;
        if (v >= kv) {
            break;
        }
        let wv = w4affm_packed4[row * rw4 + v];
        let gi = sbase + ((v << 5u) / w4affm_params.gs);
        let sc = bf16_decode(w4affm_scale[gi]);
        let bi = bf16_decode(w4affm_bias[gi]);
        let qe0 = vec4<f32>(unpack4xU8(wv.x & 0x0f0f0f0fu));
        let qo0 = vec4<f32>(unpack4xU8((wv.x >> 4u) & 0x0f0f0f0fu));
        let qe1 = vec4<f32>(unpack4xU8(wv.y & 0x0f0f0f0fu));
        let qo1 = vec4<f32>(unpack4xU8((wv.y >> 4u) & 0x0f0f0f0fu));
        let qe2 = vec4<f32>(unpack4xU8(wv.z & 0x0f0f0f0fu));
        let qo2 = vec4<f32>(unpack4xU8((wv.z >> 4u) & 0x0f0f0f0fu));
        let qe3 = vec4<f32>(unpack4xU8(wv.w & 0x0f0f0f0fu));
        let qo3 = vec4<f32>(unpack4xU8((wv.w >> 4u) & 0x0f0f0f0fu));
        for (var j = 0u; j < 4u; j = j + 1u) {
            let t = quad * 4u + j;
            let xb = hb + blk * 128u + (t << 3u);
            let x0 = w4affm_xf_tile[xb];
            let x1 = w4affm_xf_tile[xb + 1u];
            let x2 = w4affm_xf_tile[xb + 2u];
            let x3 = w4affm_xf_tile[xb + 3u];
            let x4v = w4affm_xf_tile[xb + 4u];
            let x5 = w4affm_xf_tile[xb + 5u];
            let x6 = w4affm_xf_tile[xb + 6u];
            let x7 = w4affm_xf_tile[xb + 7u];
            var qdot = dot(qe0, x0) + dot(qo0, x1);
            qdot = qdot + dot(qe1, x2) + dot(qo1, x3);
            qdot = qdot + dot(qe2, x4v) + dot(qo2, x5);
            qdot = qdot + dot(qe3, x6) + dot(qo3, x7);
            let xsv = ((x0 + x1) + (x2 + x3)) + ((x4v + x5) + (x6 + x7));
            acc[j] = fma(sc, qdot, fma(bi, dot(xsv, W4AFFM_ONES), acc[j]));
        }
        }
    }
    return acc;
}

@compute @workgroup_size(256)
fn gemm_w4a16_affine_mrows_v2_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let quad = tid & 3u;
    let rl = tid >> 2u;
    let row0 = (wid.x + wid.y * w4affm_params.groups_x) * 64u;
    let row = min(row0 + rl, w4affm_params.n_rows - 1u);
    let live = row0 + rl < w4affm_params.n_rows;
    let kv = w4affm_params.k_elems >> 5u;
    let rw4 = w4affm_params.w_row_words >> 2u;
    let sbase = row * w4affm_params.scale_row_stride;
    let ml = min(w4affm_params.m_live, W4AFFM_MAX_M);
    let acc = w4affm_v2_acc(tid, row, quad, kv, ml, rw4, sbase);
    let other = subgroupShuffleDown(acc, 4u);
    if ((rl & 1u) == 0u && live) {
        for (var j = 0u; j < 4u; j = j + 1u) {
            let t = quad * 4u + j;
            if (t >= ml) {
                break;
            }
            let lo = bf16_encode(acc[j]) & 0xffffu;
            var hi = 0u;
            if (row0 + rl + 1u < w4affm_params.n_rows) {
                hi = bf16_encode(other[j]) & 0xffffu;
            }
            w4affm_y[t * w4affm_params.y_stride_words + ((row0 + rl) >> 1u)] = lo | (hi << 16u);
        }
    }
}
