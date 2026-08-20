struct GemvBf16Params {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
};

@group(0) @binding(0) var<storage, read> gemv_bf16_w: array<u32>;
@group(0) @binding(1) var<storage, read> gemv_bf16_x: array<u32>;
@group(0) @binding(2) var<storage, read_write> gemv_bf16_y: array<u32>;
@group(0) @binding(3) var<uniform> gemv_bf16_params: GemvBf16Params;

const GEMV_BF16_LANES: u32 = 32u;
const GEMV_BF16_ROWS: u32 = 8u;
const GEMV_BF16_BLOCK: u32 = 256u;

var<workgroup> gemv_bf16_partial: array<f32, 256>;

fn gemv_bf16_row(wid: vec3<u32>, warp: u32) -> u32 {
    return (wid.x + wid.y * gemv_bf16_params.groups_x) * GEMV_BF16_ROWS + warp;
}

fn gemv_bf16_reduce(tid: u32, lane: u32, acc: f32) -> f32 {
    gemv_bf16_partial[tid] = acc;
    workgroupBarrier();
    for (var stride = GEMV_BF16_LANES >> 1u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            gemv_bf16_partial[tid] = gemv_bf16_partial[tid] + gemv_bf16_partial[tid + stride];
        }
        workgroupBarrier();
    }
    return gemv_bf16_partial[tid - lane];
}

@compute @workgroup_size(256)
fn gemv_bf16_vec8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = gemv_bf16_row(wid, warp);
    let live = row < gemv_bf16_params.n_rows;
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

    let total = gemv_bf16_reduce(tid, lane, acc);
    if (lane == 0u && live) {
        gemv_bf16_y[row] = bf16_encode(total);
    }
}

@compute @workgroup_size(256)
fn gemv_bf16_scalar(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = gemv_bf16_row(wid, warp);
    let live = row < gemv_bf16_params.n_rows;
    let k_elems = select(0u, gemv_bf16_params.k_elems, live);
    let w_base = select(0u, row * gemv_bf16_params.w_row_words, live);

    var acc = 0.0;
    for (var k = lane; k < k_elems; k = k + GEMV_BF16_LANES) {
        let wv = bf16_decode(u16_at(gemv_bf16_w[w_base + (k >> 1u)], k));
        let xv = bf16_decode(u16_at(gemv_bf16_x[k >> 1u], k));
        acc = acc + wv * xv;
    }

    let total = gemv_bf16_reduce(tid, lane, acc);
    if (lane == 0u && live) {
        gemv_bf16_y[row] = bf16_encode(total);
    }
}

@group(0) @binding(20) var<storage, read> gemv_bf16_w4: array<vec4<u32>>;
@group(0) @binding(21) var<storage, read> gemv_bf16_x4: array<vec4<u32>>;

@compute @workgroup_size(256)
fn gemv_bf16_vec8_v4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = gemv_bf16_row(wid, warp);
    let live = row < gemv_bf16_params.n_rows;
    let kv = select(0u, gemv_bf16_params.k_elems >> 3u, live);
    let w_base = select(0u, row * (gemv_bf16_params.w_row_words >> 2u), live);

    var acc = 0.0;
    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {
        let ww = gemv_bf16_w4[w_base + v];
        let xw = gemv_bf16_x4[v];
        for (var j = 0u; j < 4u; j = j + 1u) {
            acc = acc + (bf16_lo(ww[j]) * bf16_lo(xw[j]) + bf16_hi(ww[j]) * bf16_hi(xw[j]));
        }
    }

    let total = gemv_bf16_reduce(tid, lane, acc);
    if (lane == 0u && live) {
        gemv_bf16_y[row] = bf16_encode(total);
    }
}

struct GemvNormedParams {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
    rstd: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct RowQuantParams {
    n_rows: u32,
    k_elems: u32,
    src_row_words: u32,
    dst_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct GemvI8Params {
    n_rows: u32,
    k_elems: u32,
    wq_row_words: u32,
    groups_x: u32,
    m_rows: u32,
    x_row_words: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(4) var<storage, read> gnb_w: array<u32>;
@group(0) @binding(5) var<storage, read> gnb_x: array<u32>;
@group(0) @binding(6) var<storage, read> gnb_wn: array<u32>;
@group(0) @binding(7) var<storage, read_write> gnb_y: array<u32>;
@group(0) @binding(8) var<uniform> gnb_params: GemvNormedParams;

@group(0) @binding(9) var<storage, read> rq_w: array<u32>;
@group(0) @binding(10) var<storage, read_write> rq_q: array<u32>;
@group(0) @binding(11) var<storage, read_write> rq_scale: array<f32>;
@group(0) @binding(12) var<uniform> rq_params: RowQuantParams;

@group(0) @binding(13) var<storage, read> gi8_wq: array<u32>;
@group(0) @binding(14) var<storage, read> gi8_row_scale: array<f32>;
@group(0) @binding(15) var<storage, read> gi8_x: array<u32>;
@group(0) @binding(16) var<storage, read> gi8_wn: array<u32>;
@group(0) @binding(17) var<storage, read> gi8_rstd: array<f32>;
@group(0) @binding(18) var<storage, read_write> gi8_y: array<u32>;
@group(0) @binding(19) var<uniform> gi8_params: GemvI8Params;

const GEMV_RQ_INV127: u32 = 0x3c010204u;

fn gemv_bf16_reduce_seq(tid: u32, lane: u32, acc: f32) -> f32 {
    workgroupBarrier();
    gemv_bf16_partial[tid] = acc;
    workgroupBarrier();
    for (var stride = GEMV_BF16_LANES >> 1u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            gemv_bf16_partial[tid] = gemv_bf16_partial[tid] + gemv_bf16_partial[tid + stride];
        }
        workgroupBarrier();
    }
    return gemv_bf16_partial[tid - lane];
}

fn gnb_xval(k: u32) -> f32 {
    let xv = bf16_decode(u16_at(gnb_x[k >> 1u], k));
    let nv = bf16_decode(u16_at(gnb_wn[k >> 1u], k));
    return xv * gnb_params.rstd * nv;
}

@compute @workgroup_size(256)
fn gemv_bf16_normed(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = (wid.x + wid.y * gnb_params.groups_x) * GEMV_BF16_ROWS + warp;
    let live = row < gnb_params.n_rows;
    let kv = select(0u, gnb_params.k_elems >> 3u, live);
    let w_base = select(0u, row * gnb_params.w_row_words, live);

    var acc = 0.0;
    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {
        let wo = w_base + (v << 2u);
        let kb = v << 3u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let ww = gnb_w[wo + j];
            let xa = gnb_xval(kb + (j << 1u));
            let xb = gnb_xval(kb + (j << 1u) + 1u);
            acc = acc + fma(bf16_lo(ww), xa, bf16_hi(ww) * xb);
        }
    }

    let total = gemv_bf16_reduce_seq(tid, lane, acc);
    if (lane == 0u && live) {
        gnb_y[row] = bf16_encode(total);
    }
}

fn rq_div127(a: f32) -> f32 {
    let y = bitcast<f32>(GEMV_RQ_INV127);
    let q = a * y;
    let r = fma(-127.0, q, a);
    return fma(r, y, q);
}

fn rq_recip_normal(b: f32) -> f32 {
    var y = 1.0 / b;
    y = fma(fma(-b, y, 1.0), y, y);
    let r = fma(-b, y, 1.0);
    let yb = bitcast<u32>(y);
    let ob = select(yb - 1u, yb + 1u, r >= 0.0);
    let yo = bitcast<f32>(ob);
    let ro = fma(-b, yo, 1.0);
    let take = (abs(ro) < abs(r)) || (abs(ro) == abs(r) && (ob & 1u) == 0u);
    return select(y, yo, take);
}

fn rq_round_even(v: f32) -> f32 {
    let frac = abs(v - trunc(v));
    return select(round(v), 2.0 * round(v * 0.5), frac == 0.5);
}

struct RqNorm {
    m: f32,
    e: i32,
};

fn rq_bf16_norm(bits: u32) -> RqNorm {
    let e = (bits >> 7u) & 255u;
    let m = bits & 127u;
    if (e > 0u) {
        return RqNorm(bitcast<f32>((127u << 23u) | (m << 16u)), i32(e) - 127);
    }
    let b = 31u - countLeadingZeros(max(m, 1u));
    return RqNorm(bitcast<f32>((127u << 23u) | ((m << (23u - b)) & 0x007fffffu)), i32(b) - 133);
}

fn rq_f32_norm(x: f32) -> RqNorm {
    let bits = bitcast<u32>(x);
    let e = (bits >> 23u) & 255u;
    if (e > 0u) {
        return RqNorm(bitcast<f32>((127u << 23u) | (bits & 0x007fffffu)), i32(e) - 127);
    }
    let m = bits & 0x007fffffu;
    let t = 31u - countLeadingZeros(max(m, 1u));
    return RqNorm(bitcast<f32>((127u << 23u) | ((m << (23u - t)) & 0x007fffffu)), i32(t) - 149);
}

fn rq_scale_from_amax(bits: u32) -> f32 {
    let mag = bits & 0x7fffu;
    if (mag == 0u) {
        return 0.0;
    }
    if (mag == 0x7f80u) {
        return bitcast<f32>(0x7f800000u);
    }
    let n = rq_bf16_norm(mag);
    let qn = rq_div127(n.m);
    let rr = fma(-127.0, qn, n.m);
    let qb = bitcast<u32>(qn);
    let eq = i32((qb >> 23u) & 255u) - 127;
    let mq = (qb & 0x007fffffu) | 0x00800000u;
    let ep = eq + n.e;
    if (ep >= -126) {
        return bitcast<f32>((u32(ep + 127) << 23u) | (qb & 0x007fffffu));
    }
    let shift = u32(-126 - ep);
    if (shift > 24u) {
        return 0.0;
    }
    let hi = mq >> shift;
    let rem = mq & ((1u << shift) - 1u);
    let half = 1u << (shift - 1u);
    let up = (rem > half) || (rem == half && (rr > 0.0 || (rr == 0.0 && (hi & 1u) == 1u)));
    return bitcast<f32>(hi + select(0u, 1u, up));
}

fn rq_quant_one(w_bits: u32, invn: f32, ps: i32, inv_over: bool, inv_zero: bool) -> u32 {
    let mag = w_bits & 0x7fffu;
    if (mag == 0u || mag > 0x7f80u || inv_zero) {
        return 0u;
    }
    let n = rq_bf16_norm(mag);
    let e = n.e - ps;
    var q = 0;
    if (inv_over || e >= 8) {
        q = 127;
    } else if (e > -126) {
        let t = n.m * invn;
        let tb = bitcast<u32>(t);
        let ne = u32(i32((tb >> 23u) & 255u) + e);
        let vf = bitcast<f32>((ne << 23u) | (tb & 0x007fffffu));
        q = min(i32(rq_round_even(vf)), 127);
    }
    return bitcast<u32>(select(q, -q, (w_bits & 0x8000u) != 0u)) & 255u;
}

@compute @workgroup_size(256)
fn rowquant_i8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let row = wid.x + wid.y * rq_params.groups_x;
    let live = row < rq_params.n_rows;
    let k = select(0u, rq_params.k_elems, live);
    let src_base = select(0u, row * rq_params.src_row_words, live);

    var amax = 0u;
    for (var i = tid; i < k; i = i + GEMV_BF16_BLOCK) {
        let mg = u16_at(rq_w[src_base + (i >> 1u)], i) & 0x7fffu;
        amax = max(amax, select(mg, 0u, mg > 0x7f80u));
    }
    gemv_bf16_partial[tid] = bitcast<f32>(amax);
    workgroupBarrier();
    for (var s = GEMV_BF16_BLOCK >> 1u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let a = bitcast<u32>(gemv_bf16_partial[tid]);
            let b = bitcast<u32>(gemv_bf16_partial[tid + s]);
            gemv_bf16_partial[tid] = bitcast<f32>(max(a, b));
        }
        workgroupBarrier();
    }

    let peak = bitcast<u32>(gemv_bf16_partial[0]);
    let scale = rq_scale_from_amax(peak);
    let sn = rq_f32_norm(scale);
    let invn = rq_recip_normal(sn.m);
    let inv_exp = (i32((bitcast<u32>(invn) >> 23u) & 255u) - 127) - sn.e;
    let inv_over = inv_exp > 127;
    let inv_zero = (peak == 0u) || (peak == 0x7f80u);
    if (tid == 0u && live) {
        rq_scale[row] = scale;
    }

    let dst_base = select(0u, row * rq_params.dst_row_words, live);
    let dst_words = select(0u, rq_params.dst_row_words, live);
    for (var wi = tid; wi < dst_words; wi = wi + GEMV_BF16_BLOCK) {
        var packed = 0u;
        for (var b = 0u; b < 4u; b = b + 1u) {
            let idx = (wi << 2u) + b;
            if (idx < k) {
                let wb = u16_at(rq_w[src_base + (idx >> 1u)], idx);
                packed = packed | (rq_quant_one(wb, invn, sn.e, inv_over, inv_zero) << (b << 3u));
            }
        }
        rq_q[dst_base + wi] = packed;
    }
}

fn gi8_scale_mul(total: f32, rs: f32) -> f32 {
    let rb = bitcast<u32>(rs);
    if (((rb >> 23u) & 255u) != 0u || (rb & 0x7fffffffu) == 0u) {
        return total * rs;
    }
    let n = rq_f32_norm(rs);
    let t = total * n.m;
    let err = fma(total, n.m, -t);
    let tb = bitcast<u32>(t);
    let te = i32((tb >> 23u) & 255u);
    let ne = te + n.e;
    let sign = (tb ^ rb) & 0x80000000u;
    if (te == 255) {
        return bitcast<f32>((tb & 0x7fffffffu) | sign);
    }
    if (te == 0) {
        return bitcast<f32>(sign);
    }
    if (ne >= 1) {
        return bitcast<f32>(u32(ne) << 23u | (tb & 0x007fffffu) | sign);
    }
    let shift = u32(1 - ne);
    if (shift > 25u) {
        return bitcast<f32>(sign);
    }
    let m24 = (tb & 0x007fffffu) | 0x00800000u;
    let hi = m24 >> shift;
    let rem = m24 & ((1u << shift) - 1u);
    let half = 1u << (shift - 1u);
    let grew = (err != 0.0) && (((bitcast<u32>(err) ^ tb) & 0x80000000u) == 0u);
    let up = (rem > half) || (rem == half && (grew || (err == 0.0 && (hi & 1u) == 1u)));
    return bitcast<f32>((hi + select(0u, 1u, up)) | sign);
}

fn gi8_xval(j: u32, k: u32) -> f32 {
    let xv = bf16_decode(u16_at(gi8_x[j * gi8_params.x_row_words + (k >> 1u)], k));
    let nv = bf16_decode(u16_at(gi8_wn[k >> 1u], k));
    return xv * gi8_rstd[j] * nv;
}

@compute @workgroup_size(256)
fn gemv_i8_normed(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = (wid.x + wid.y * gi8_params.groups_x) * GEMV_BF16_ROWS + warp;
    let live = row < gi8_params.n_rows;
    let kv = select(0u, gi8_params.k_elems >> 4u, live);
    let w_base = select(0u, row * gi8_params.wq_row_words, live);

    var acc = 0.0;
    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {
        let wo = w_base + (v << 2u);
        let kb = v << 4u;
        for (var t = 0u; t < 4u; t = t + 1u) {
            let word = gi8_wq[wo + t];
            for (var i = 0u; i < 4u; i = i + 1u) {
                acc = fma(int8_decode(word, i), gi8_xval(0u, kb + (t << 2u) + i), acc);
            }
        }
    }

    let total = gemv_bf16_reduce_seq(tid, lane, acc);
    if (lane == 0u && live) {
        gi8_y[row] = bf16_encode(gi8_scale_mul(total, gi8_row_scale[select(0u, row, live)]));
    }
}

@compute @workgroup_size(256)
fn gemv_i8_normed_mk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let lane = tid & (GEMV_BF16_LANES - 1u);
    let warp = tid / GEMV_BF16_LANES;
    let row = (wid.x + wid.y * gi8_params.groups_x) * GEMV_BF16_ROWS + warp;
    let live = row < gi8_params.n_rows;
    let kv = select(0u, gi8_params.k_elems >> 4u, live);
    let w_base = select(0u, row * gi8_params.wq_row_words, live);
    let m = gi8_params.m_rows;

    var acc = array<f32, 8>(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {
        let wo = w_base + (v << 2u);
        let kb = v << 4u;
        for (var t = 0u; t < 4u; t = t + 1u) {
            let word = gi8_wq[wo + t];
            for (var i = 0u; i < 4u; i = i + 1u) {
                let f = int8_decode(word, i);
                let kk = kb + (t << 2u) + i;
                for (var j = 0u; j < m; j = j + 1u) {
                    acc[j] = fma(f, gi8_xval(j, kk), acc[j]);
                }
            }
        }
    }

    let rs = gi8_row_scale[select(0u, row, live)];
    for (var j = 0u; j < m; j = j + 1u) {
        let a = gemv_bf16_reduce_seq(tid, lane, acc[j]);
        if (lane == 0u && live) {
            gi8_y[j * gi8_params.n_rows + row] = bf16_encode(gi8_scale_mul(a, rs));
        }
    }
}
