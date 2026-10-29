struct GemvW4A16AffineParams {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    seg_nq: u32,
    seg_nk: u32,
};

@group(0) @binding(0) var<storage, read> w4aff_packed: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> w4aff_scale: array<u32>;
@group(0) @binding(2) var<storage, read> w4aff_x: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> w4aff_y: array<u32>;
@group(0) @binding(4) var<uniform> w4aff_params: GemvW4A16AffineParams;
@group(0) @binding(6) var<storage, read> w4aff_bias: array<u32>;
@group(0) @binding(5) var<storage, read_write> w4aff_y_k: array<u32>;
@group(0) @binding(7) var<storage, read_write> w4aff_y_v: array<u32>;

const W4AFF_LANES: u32 = 32u;
const W4AFF_ROWS: u32 = 8u;

var<workgroup> w4aff_warp_sums: array<f32, 8>;

fn w4aff_x_word(word: u32) -> vec2<f32> {
    return vec2<f32>(bf16_lo(word), bf16_hi(word));
}

fn w4aff_scale_at(base: u32, g: u32) -> f32 {
    return bf16_decode(w4aff_scale[base + g]);
}

fn w4aff_bias_at(base: u32, g: u32) -> f32 {
    return bf16_decode(w4aff_bias[base + g]);
}

fn w4aff_q(pv: u32, elem: u32) -> f32 {
    return f32(u4_unpack(pv, elem));
}

fn w4aff_dot8(pv: u32, xv: vec4<u32>, acc_in: vec2<f32>) -> vec2<f32> {
    var qdot = acc_in.x;
    var xsum = acc_in.y;
    for (var i = 0u; i < 4u; i = i + 1u) {
        let xp = w4aff_x_word(xv[i]);
        qdot = fma(w4aff_q(pv, 2u * i), xp.x, qdot);
        qdot = fma(w4aff_q(pv, 2u * i + 1u), xp.y, qdot);
        xsum = xsum + xp.x + xp.y;
    }
    return vec2<f32>(qdot, xsum);
}

fn w4aff_dot32(wv: vec4<u32>, xi: u32) -> vec2<f32> {
    var a = vec2<f32>(0.0, 0.0);
    a = w4aff_dot8(wv.x, w4aff_x[xi], a);
    a = w4aff_dot8(wv.y, w4aff_x[xi + 1u], a);
    a = w4aff_dot8(wv.z, w4aff_x[xi + 2u], a);
    a = w4aff_dot8(wv.w, w4aff_x[xi + 3u], a);
    return a;
}

fn w4aff_lane_reduce(tid: u32, lane: u32, acc: f32) -> f32 {
    var a = acc;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn w4aff_row_acc_block(wbase4: u32, sbase: u32, kv: u32, lane: u32, gs: u32) -> f32 {
    var acc = 0.0;
    if ((gs & 31u) == 0u) {
        for (var v = lane; v < kv; v = v + W4AFF_LANES) {
            let g = (v * 32u) / gs;
            let d = w4aff_dot32(w4aff_packed[wbase4 + v], v * 4u);
            acc = fma(w4aff_scale_at(sbase, g), d.x, fma(w4aff_bias_at(sbase, g), d.y, acc));
        }
    } else {
        for (var v = lane; v < kv; v = v + W4AFF_LANES) {
            let wv = w4aff_packed[wbase4 + v];
            for (var j = 0u; j < 4u; j = j + 1u) {
                let g = (v * 32u + j * 8u) / gs;
                let d = w4aff_dot8(wv[j], w4aff_x[v * 4u + j], vec2<f32>(0.0, 0.0));
                acc = fma(w4aff_scale_at(sbase, g), d.x, fma(w4aff_bias_at(sbase, g), d.y, acc));
            }
        }
    }
    return acc;
}

@compute @workgroup_size(256)
fn gemv_w4a16_affine_block(
    #def b3
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (W4AFF_LANES - 1u);
    let warp = tid / W4AFF_LANES;
    let row = (wid.x + wid.y * w4aff_params.groups_x) * W4AFF_ROWS + warp;
    let live = row < w4aff_params.n_rows;
    let kv = select(0u, w4aff_params.k_elems >> 5u, live);
    let wbase = select(0u, row * (w4aff_params.w_row_words >> 2u), live);
    let sbase = select(0u, row * w4aff_params.scale_row_stride, live);

    let acc = w4aff_row_acc_block(wbase, sbase, kv, lane, w4aff_params.gs);
    let total = w4aff_lane_reduce(tid, lane, acc);
    #end
    if (lane == 0u && live) {
        w4aff_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(256)
fn gemv_w4a16_affine_block_pk(
    #use b3
    if (lane == 0u) {
        w4aff_warp_sums[warp] = total;
    }
    workgroupBarrier();
    if (lane == 0u && (warp & 1u) == 0u && live) {
        let lo = bf16_encode(w4aff_warp_sums[warp]) & 0xffffu;
        var hi = 0u;
        if (row + 1u < w4aff_params.n_rows) {
            hi = bf16_encode(w4aff_warp_sums[warp + 1u]) & 0xffffu;
        }
        w4aff_y[row >> 1u] = lo | (hi << 16u);
    }
}

fn w4aff_qdot8v(pv: u32, lo: vec4<f32>, hi: vec4<f32>) -> f32 {
    let ql = vec4<f32>(
        f32(pv & 0xfu),
        f32((pv >> 4u) & 0xfu),
        f32((pv >> 8u) & 0xfu),
        f32((pv >> 12u) & 0xfu)
    );
    let qh = vec4<f32>(
        f32((pv >> 16u) & 0xfu),
        f32((pv >> 20u) & 0xfu),
        f32((pv >> 24u) & 0xfu),
        f32(pv >> 28u)
    );
    return dot(ql, lo) + dot(qh, hi);
}

fn w4aff_xf_pair(w: vec2<u32>) -> vec4<f32> {
    return vec4<f32>(bf16_lo(w.x), bf16_hi(w.x), bf16_lo(w.y), bf16_hi(w.y));
}

fn w4aff_lane_reduce4(a_in: vec4<f32>) -> vec4<f32> {
    var a = a_in;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn w4aff_mrow4_total(row0: u32, rlive: u32, kv: u32, lane: u32) -> vec4<f32> {
    let rw4 = w4aff_params.w_row_words >> 2u;
    let ss = w4aff_params.scale_row_stride;
    let gs = w4aff_params.gs;
    var acc = vec4<f32>(0.0);
    #def b1
    for (var v = lane; v < kv; v = v + 32u) {
        let xa = w4aff_x[v * 4u];
        let xb = w4aff_x[v * 4u + 1u];
        let xc = w4aff_x[v * 4u + 2u];
        let xd = w4aff_x[v * 4u + 3u];
        let xf0 = w4aff_xf_pair(xa.xy);
        let xf1 = w4aff_xf_pair(xa.zw);
        let xf2 = w4aff_xf_pair(xb.xy);
        let xf3 = w4aff_xf_pair(xb.zw);
        let xf4 = w4aff_xf_pair(xc.xy);
        let xf5 = w4aff_xf_pair(xc.zw);
        let xf6 = w4aff_xf_pair(xd.xy);
        let xf7 = w4aff_xf_pair(xd.zw);
        let xsum = dot(
            ((xf0 + xf1) + (xf2 + xf3)) + ((xf4 + xf5) + (xf6 + xf7)),
            vec4<f32>(1.0)
        );
        let g = (v * 32u) / gs;
        for (var r = 0u; r < rlive; r = r + 1u) {
            let row = row0 + r;
    #end
            let wv = w4aff_packed[row * rw4 + v];
            var qd = w4aff_qdot8v(wv.x, xf0, xf1);
            qd = qd + w4aff_qdot8v(wv.y, xf2, xf3);
            qd = qd + w4aff_qdot8v(wv.z, xf4, xf5);
            qd = qd + w4aff_qdot8v(wv.w, xf6, xf7);
            let sb = row * ss + g;
            acc[r] = fma(
                bf16_decode(w4aff_scale[sb]),
                qd,
                fma(bf16_decode(w4aff_bias[sb]), xsum, acc[r])
            );
        }
    }
    return w4aff_lane_reduce4(acc);
}

fn w4aff_mrow8_total(row0: u32, rlive: u32, kv: u32, lane: u32) -> array<vec4<f32>, 2> {
    let rw4 = w4aff_params.w_row_words >> 2u;
    let ss = w4aff_params.scale_row_stride;
    let gs = w4aff_params.gs;
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    #use b1
            let wv = w4aff_packed[row * rw4 + v];
            var qd = w4aff_qdot8v(wv.x, xf0, xf1);
            qd = qd + w4aff_qdot8v(wv.y, xf2, xf3);
            qd = qd + w4aff_qdot8v(wv.z, xf4, xf5);
            qd = qd + w4aff_qdot8v(wv.w, xf6, xf7);
            let sb = row * ss + g;
            let contrib = fma(
                bf16_decode(w4aff_scale[sb]),
                qd,
                bf16_decode(w4aff_bias[sb]) * xsum
            );
            if (r < 4u) {
                acc0[r] = acc0[r] + contrib;
            } else {
                acc1[r - 4u] = acc1[r - 4u] + contrib;
            }
        }
    }
    return array<vec4<f32>, 2>(w4aff_lane_reduce4(acc0), w4aff_lane_reduce4(acc1));
}

@compute @workgroup_size(256)
fn gemv_w4a16_affine_mrow8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let row0 = ((wid.x + wid.y * w4aff_params.groups_x) * 8u + warp) * 8u;
    let live = row0 < w4aff_params.n_rows;
    let rlive = select(0u, min(w4aff_params.n_rows - row0, 8u), live);
    let kv = select(0u, w4aff_params.k_elems >> 5u, live);
    let total = w4aff_mrow8_total(row0, rlive, kv, lane);
    if (lane == 0u) {
        for (var r = 0u; r < rlive; r = r + 1u) {
            w4aff_y[row0 + r] = bf16_encode(total[r >> 2u][r & 3u]);
        }
    }
}

@compute @workgroup_size(128)
fn gemv_w4a16_affine_mrow4h(
    #def b4
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let row0 = ((wid.x + wid.y * w4aff_params.groups_x) * 4u + warp) * 4u;
    let live = row0 < w4aff_params.n_rows;
    let rlive = select(0u, min(w4aff_params.n_rows - row0, 4u), live);
    let kv = select(0u, w4aff_params.k_elems >> 5u, live);
    let total = w4aff_mrow4_total(row0, rlive, kv, lane);
    if (lane == 0u) {
    #end
        for (var r = 0u; r < rlive; r = r + 1u) {
            w4aff_y[row0 + r] = bf16_encode(total[r]);
        }
    }
}

@group(0) @binding(20) var<storage, read> w4aff_packed_b: array<vec4<u32>>;
@group(0) @binding(21) var<storage, read> w4aff_scale_b: array<u32>;
@group(0) @binding(22) var<storage, read> w4aff_bias_b: array<u32>;

fn w4aff_mrow4_total_b(row0: u32, rlive: u32, kv: u32, lane: u32) -> vec4<f32> {
    let rw4 = w4aff_params.w_row_words >> 2u;
    let ss = w4aff_params.scale_row_stride;
    let gs = w4aff_params.gs;
    var acc = vec4<f32>(0.0);
    #use b1
            let wv = w4aff_packed_b[row * rw4 + v];
            var qd = w4aff_qdot8v(wv.x, xf0, xf1);
            qd = qd + w4aff_qdot8v(wv.y, xf2, xf3);
            qd = qd + w4aff_qdot8v(wv.z, xf4, xf5);
            qd = qd + w4aff_qdot8v(wv.w, xf6, xf7);
            let sb = row * ss + g;
            acc[r] = fma(
                bf16_decode(w4aff_scale_b[sb]),
                qd,
                fma(bf16_decode(w4aff_bias_b[sb]), xsum, acc[r])
            );
        }
    }
    return w4aff_lane_reduce4(acc);
}

@compute @workgroup_size(128)
fn gemv_w4a16_affine_mrow4h_gu_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let seg = w4aff_params.seg_nq;
    let row0g = ((wid.x + wid.y * w4aff_params.groups_x) * 4u + warp) * 4u;
    let live = row0g < w4aff_params.n_rows;
    let is_a = row0g < seg;
    let row0 = select(row0g - seg, row0g, is_a);
    let seg_rows = select(w4aff_params.n_rows - seg, seg, is_a);
    let rlive = select(0u, min(seg_rows - min(row0, seg_rows), 4u), live);
    let kv = select(0u, w4aff_params.k_elems >> 5u, live);
    var total = vec4<f32>(0.0);
    if (is_a) {
        total = w4aff_mrow4_total(row0, rlive, kv, lane);
    } else {
        total = w4aff_mrow4_total_b(row0, rlive, kv, lane);
    }
    if (lane == 0u) {
        for (var rp = 0u; rp * 2u < rlive; rp = rp + 1u) {
            let lo = bf16_encode(total[rp * 2u]) & 0xffffu;
            var hi = 0u;
            if (rp * 2u + 1u < rlive) {
                hi = bf16_encode(total[rp * 2u + 1u]) & 0xffffu;
            }
            if (is_a) {
                w4aff_y[(row0 >> 1u) + rp] = lo | (hi << 16u);
            } else {
                w4aff_y_k[(row0 >> 1u) + rp] = lo | (hi << 16u);
            }
        }
    }
}

@compute @workgroup_size(128)
fn gemv_w4a16_affine_mrow4h_pk(
    #use b4
        #def b5
        for (var rp = 0u; rp * 2u < rlive; rp = rp + 1u) {
            let lo = bf16_encode(total[rp * 2u]) & 0xffffu;
            var hi = 0u;
            if (rp * 2u + 1u < rlive) {
                hi = bf16_encode(total[rp * 2u + 1u]) & 0xffffu;
            }
            w4aff_y[(row0 >> 1u) + rp] = lo | (hi << 16u);
        }
    }
}
        #end

@compute @workgroup_size(256)
fn gemv_w4a16_affine_mrow4(
    #def b2
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & 31u;
    let warp = tid >> 5u;
    let row0 = ((wid.x + wid.y * w4aff_params.groups_x) * 8u + warp) * 4u;
    let live = row0 < w4aff_params.n_rows;
    let rlive = select(0u, min(w4aff_params.n_rows - row0, 4u), live);
    let kv = select(0u, w4aff_params.k_elems >> 5u, live);
    let total = w4aff_mrow4_total(row0, rlive, kv, lane);
    if (lane == 0u) {
    #end
        for (var r = 0u; r < rlive; r = r + 1u) {
            w4aff_y[row0 + r] = bf16_encode(total[r]);
        }
    }
}

@compute @workgroup_size(256)
fn gemv_w4a16_affine_mrow4_pk(
    #use b2
        #use b5

@compute @workgroup_size(256)
fn gemv_w4a16_affine_mrow4_qkv_pk(
    #use b2
        for (var rp = 0u; rp * 2u < rlive; rp = rp + 1u) {
            let row = row0 + rp * 2u;
            let lo = bf16_encode(total[rp * 2u]) & 0xffffu;
            var hi = 0u;
            if (rp * 2u + 1u < rlive) {
                hi = bf16_encode(total[rp * 2u + 1u]) & 0xffffu;
            }
            let word = lo | (hi << 16u);
            if (row < w4aff_params.seg_nq) {
                w4aff_y[row >> 1u] = word;
            } else if (row < w4aff_params.seg_nq + w4aff_params.seg_nk) {
                w4aff_y_k[(row - w4aff_params.seg_nq) >> 1u] = word;
            } else {
                w4aff_y_v[(row - w4aff_params.seg_nq - w4aff_params.seg_nk) >> 1u] = word;
            }
        }
    }
}
