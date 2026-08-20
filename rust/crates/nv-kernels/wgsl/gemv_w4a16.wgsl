struct GemvW4A16Params {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
};

@group(0) @binding(0) var<storage, read> w4a16_packed: array<u32>;
@group(0) @binding(1) var<storage, read> w4a16_scale: array<u32>;
@group(0) @binding(2) var<storage, read> w4a16_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> w4a16_y: array<u32>;
@group(0) @binding(4) var<uniform> w4a16_params: GemvW4A16Params;
@group(0) @binding(5) var<storage, read> w4a16_pli: array<f32>;

const W4A16_LANES: u32 = 32u;
const W4A16_ROWS: u32 = 8u;
const W4A16_BLOCK: u32 = 256u;
const W4A16_WARPS: u32 = 8u;

var<workgroup> w4a16_partial: array<f32, 256>;
var<workgroup> w4a16_warp_sums: array<f32, 8>;

fn w4a16_x_pair(idx: u32) -> vec2<f32> {
    let word = w4a16_x[idx];
    return vec2<f32>(bf16_lo(word), bf16_hi(word));
}

fn w4a16_scale_at(base: u32, g: u32) -> f32 {
    return bf16_decode(w4a16_scale[base + g]);
}

fn w4a16_q(pv: u32, elem: u32) -> f32 {
    return f32(u4_unpack(pv, elem)) - 8.0;
}

fn w4a16_dot8(pv: u32, kb: u32, acc_in: f32) -> f32 {
    var acc = acc_in;
    let xb = kb >> 1u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let xp = w4a16_x_pair(xb + i);
        acc = fma(w4a16_q(pv, 2u * i), xp.x, acc);
        acc = fma(w4a16_q(pv, 2u * i + 1u), xp.y, acc);
    }
    return acc;
}

fn w4a16_dot8_pairwise(pv: u32, kb: u32) -> f32 {
    var a = 0.0;
    let xb = kb >> 1u;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let xp = w4a16_x_pair(xb + i);
        a = a + (w4a16_q(pv, 2u * i) * xp.x + w4a16_q(pv, 2u * i + 1u) * xp.y);
    }
    return a;
}

fn w4a16_dot32(wbase: u32, kbase: u32) -> f32 {
    var a = 0.0;
    for (var j = 0u; j < 4u; j = j + 1u) {
        a = w4a16_dot8(w4a16_packed[wbase + j], kbase + j * 8u, a);
    }
    return a;
}

fn w4a16_lane_reduce(tid: u32, lane: u32, acc: f32) -> f32 {
    w4a16_partial[tid] = acc;
    workgroupBarrier();
    for (var stride = W4A16_LANES >> 1u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            w4a16_partial[tid] = w4a16_partial[tid] + w4a16_partial[tid + stride];
        }
        workgroupBarrier();
    }
    return w4a16_partial[tid - lane];
}

fn w4a16_block_reduce(tid: u32, acc: f32) -> f32 {
    let lane = tid & (W4A16_LANES - 1u);
    let warp = tid / W4A16_LANES;
    let warp_total = w4a16_lane_reduce(tid, lane, acc);
    if (lane == 0u) {
        w4a16_warp_sums[warp] = warp_total;
    }
    workgroupBarrier();
    for (var stride = W4A16_WARPS >> 1u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            w4a16_warp_sums[tid] = w4a16_warp_sums[tid] + w4a16_warp_sums[tid + stride];
        }
        workgroupBarrier();
    }
    return w4a16_warp_sums[0];
}

fn w4a16_row_acc_block(wbase: u32, sbase: u32, kv: u32, lane: u32, gs: u32) -> f32 {
    var acc = 0.0;
    if ((gs & 31u) == 0u) {
        for (var v = lane; v < kv; v = v + W4A16_LANES) {
            let kbase = v * 32u;
            let sc = w4a16_scale_at(sbase, kbase / gs);
            acc = fma(sc, w4a16_dot32(wbase + v * 4u, kbase), acc);
        }
    } else {
        for (var v = lane; v < kv; v = v + W4A16_LANES) {
            let kbase = v * 32u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let kb = kbase + j * 8u;
                let sc = w4a16_scale_at(sbase, kb / gs);
                let a = w4a16_dot8(w4a16_packed[wbase + v * 4u + j], kb, 0.0);
                acc = fma(a, sc, acc);
            }
        }
    }
    return acc;
}

@compute @workgroup_size(256)
fn gemv_w4a16_block(
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
    if (lane == 0u && live) {
        w4a16_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(256)
fn gemv_w4a16_gelu_pli(
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
    if (lane == 0u && live) {
        let c = 0.7978845608028654;
        let t = nv_tanhf(c * (total + 0.044715 * total * total * total));
        let gelu = 0.5 * total * (1.0 + t);
        w4a16_y[row] = bf16_encode(gelu * w4a16_pli[row]);
    }
}

@group(0) @binding(6) var<storage, read> w4a16_packed4: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> w4a16_x4: array<vec4<u32>>;

fn w4a16_dot8_v4(pv: u32, xw: vec4<u32>, acc_in: f32) -> f32 {
    let qe = vec4<f32>(unpack4xU8(pv & 0x0f0f0f0fu)) - vec4<f32>(8.0);
    let qo = vec4<f32>(unpack4xU8((pv >> 4u) & 0x0f0f0f0fu)) - vec4<f32>(8.0);
    let xe = bitcast<vec4<f32>>(xw << vec4<u32>(16u));
    let xo = bitcast<vec4<f32>>(xw & vec4<u32>(0xffff0000u));
    var s = acc_in;
    s = fma(qe.x, xe.x, s);
    s = fma(qo.x, xo.x, s);
    s = fma(qe.y, xe.y, s);
    s = fma(qo.y, xo.y, s);
    s = fma(qe.z, xe.z, s);
    s = fma(qo.z, xo.z, s);
    s = fma(qe.w, xe.w, s);
    s = fma(qo.w, xo.w, s);
    return s;
}

fn w4a16_dot32_v4(wv: vec4<u32>, xb: u32) -> f32 {
    var a = 0.0;
    a = w4a16_dot8_v4(wv.x, w4a16_x4[xb], a);
    a = w4a16_dot8_v4(wv.y, w4a16_x4[xb + 1u], a);
    a = w4a16_dot8_v4(wv.z, w4a16_x4[xb + 2u], a);
    a = w4a16_dot8_v4(wv.w, w4a16_x4[xb + 3u], a);
    return a;
}

fn w4a16_row_acc_v4(wbase4: u32, sbase: u32, kv: u32, lane: u32, gs: u32) -> f32 {
    var acc = 0.0;
    if ((gs & 31u) == 0u) {
        for (var v = lane; v < kv; v = v + W4A16_LANES) {
            let sc = w4a16_scale_at(sbase, (v << 5u) / gs);
            acc = fma(sc, w4a16_dot32_v4(w4a16_packed4[wbase4 + v], v << 2u), acc);
        }
    } else {
        for (var v = lane; v < kv; v = v + W4A16_LANES) {
            let wv = w4a16_packed4[wbase4 + v];
            let kbase = v << 5u;
            for (var j = 0u; j < 4u; j = j + 1u) {
                let kb = kbase + j * 8u;
                let sc = w4a16_scale_at(sbase, kb / gs);
                let a = w4a16_dot8_v4(wv[j], w4a16_x4[(v << 2u) + j], 0.0);
                acc = fma(a, sc, acc);
            }
        }
    }
    return acc;
}

@compute @workgroup_size(256)
fn gemv_w4a16_v4(
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
    if (lane == 0u && live) {
        w4a16_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(256)
fn gemv_w4a16_row(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let row = wid.x + wid.y * w4a16_params.groups_x;
    let live = row < w4a16_params.n_rows;
    let kv = select(0u, w4a16_params.k_elems >> 5u, live);
    let wbase = select(0u, row * w4a16_params.w_row_words, live);
    let sbase = select(0u, row * w4a16_params.scale_row_stride, live);
    let gs = w4a16_params.gs;
    let wide = (gs & 31u) == 0u;

    var acc = 0.0;
    for (var v = tid; v < kv; v = v + W4A16_BLOCK) {
        let kbase = v * 32u;
        let sc = select(0.0, w4a16_scale_at(sbase, kbase / gs), wide);
        var block_acc = 0.0;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let kb = kbase + j * 8u;
            let scj = select(w4a16_scale_at(sbase, kb / gs), sc, wide);
            let a = w4a16_dot8_pairwise(w4a16_packed[wbase + v * 4u + j], kb);
            if (wide) {
                block_acc = block_acc + a;
            } else {
                block_acc = fma(a, scj, block_acc);
            }
        }
        if (wide) {
            acc = fma(sc, block_acc, acc);
        } else {
            acc = acc + block_acc;
        }
    }

    let total = w4a16_block_reduce(tid, acc);
    if (tid == 0u && live) {
        w4a16_y[row] = bf16_encode(total);
    }
}
