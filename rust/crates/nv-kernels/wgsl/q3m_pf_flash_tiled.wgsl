
const FDT_ROWS: u32 = 32u;
const FDT_WARP_ROWS: u32 = 4u;
const FDT_POS: u32 = 8u;

var<workgroup> fdt_k_stage_decoded_once_per_workgroup: array<f32, 2048>;
var<workgroup> fdt_v_stage_decoded_once_per_workgroup: array<f32, 2048>;
var<workgroup> fdt_k_scale_stage: array<f32, 8>;
var<workgroup> fdt_v_scale_stage: array<f32, 8>;

fn fdt_ring_slot(p: i32) -> u32 {
    var sp = u32(p);
    if (fd_params.ring > 0u) {
        sp = sp % fd_params.ring;
    }
    return sp;
}

fn fdt_stage_scales(lid: u32, p0: i32, total: i32, nkv: u32, kvh: u32) {
    if (lid < FDT_POS) {
        let p = p0 + i32(lid);
        var ks = 0.0;
        var vs = 0.0;
        if (p < total) {
            let sp = fdt_ring_slot(p);
            ks = fd_k_scales[sp * nkv + kvh];
            vs = fd_v_scales[sp * nkv + kvh];
        }
        fdt_k_scale_stage[lid] = ks;
        fdt_v_scale_stage[lid] = vs;
    }
}

fn fdt_stage_k_and_v_in_one_pass_so_each_round_needs_two_barriers_not_four(
    lid: u32,
    p0: i32,
    total: i32,
    hd: u32,
    nkv: u32,
    kvh: u32
) {
    let n4 = (FDT_POS * hd) >> 2u;
    for (var t = lid; t < n4; t = t + FD_BLOCK) {
        let e = t << 2u;
        let j = e / hd;
        let d = e - j * hd;
        let p = p0 + i32(j);
        var wk = 0u;
        var wv = 0u;
        if (p < total) {
            let g = (fdt_ring_slot(p) * nkv + kvh) * hd + d;
            wk = fd_k_words[g >> 2u];
            wv = fd_v_words[g >> 2u];
        }
        fdt_k_stage_decoded_once_per_workgroup[e] = e4m3_decode(byte_at(wk, 0u));
        fdt_k_stage_decoded_once_per_workgroup[e + 1u] = e4m3_decode(byte_at(wk, 1u));
        fdt_k_stage_decoded_once_per_workgroup[e + 2u] = e4m3_decode(byte_at(wk, 2u));
        fdt_k_stage_decoded_once_per_workgroup[e + 3u] = e4m3_decode(byte_at(wk, 3u));
        fdt_v_stage_decoded_once_per_workgroup[e] = e4m3_decode(byte_at(wv, 0u));
        fdt_v_stage_decoded_once_per_workgroup[e + 1u] = e4m3_decode(byte_at(wv, 1u));
        fdt_v_stage_decoded_once_per_workgroup[e + 2u] = e4m3_decode(byte_at(wv, 2u));
        fdt_v_stage_decoded_once_per_workgroup[e + 3u] = e4m3_decode(byte_at(wv, 3u));
    }
}

const FDT_VEC4_GROUPS_MATCH_THE_MK_SG_LANE_TO_DIM_MAP_FOR_BIT_EXACT_SCORES: u32 = 2u;

fn fdt_load_qreg_vec4(
    q: ptr<function, array<f32, 32>>,
    h: u32,
    row0: u32,
    rows: u32,
    hd: u32,
    lane: u32,
    warp: u32
) {
    for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
        let qi = warp + rr * FD_WARPS;
        for (var g = 0u; g < FDT_VEC4_GROUPS_MATCH_THE_MK_SG_LANE_TO_DIM_MAP_FOR_BIT_EXACT_SCORES; g = g + 1u) {
            let d4 = (lane + g * FD_LANES) << 2u;
            for (var c = 0u; c < 4u; c = c + 1u) {
                var v = 0.0;
                if (qi < rows && d4 + c < hd) {
                    v = fd_q[((row0 + qi) * fd_params.n_heads + h) * hd + d4 + c];
                }
                (*q)[(rr * FDT_VEC4_GROUPS_MATCH_THE_MK_SG_LANE_TO_DIM_MAP_FOR_BIT_EXACT_SCORES + g) * 4u + c] = v;
            }
        }
    }
}

fn fdt_score_partial_same_fma_chain_as_the_mk_sg_vec4_path(
    q: ptr<function, array<f32, 32>>,
    k: ptr<function, array<f32, 8>>,
    rr: u32,
    hd: u32,
    lane: u32
) -> f32 {
    var partial = 0.0;
    for (var g = 0u; g < FDT_VEC4_GROUPS_MATCH_THE_MK_SG_LANE_TO_DIM_MAP_FOR_BIT_EXACT_SCORES; g = g + 1u) {
        if (((lane + g * FD_LANES) << 2u) < hd) {
            let qb = (rr * FDT_VEC4_GROUPS_MATCH_THE_MK_SG_LANE_TO_DIM_MAP_FOR_BIT_EXACT_SCORES + g) * 4u;
            let kb = g * 4u;
            var t = (*q)[qb + 1u] * (*k)[kb + 1u];
            t = fma((*q)[qb], (*k)[kb], t);
            t = fma((*q)[qb + 2u], (*k)[kb + 2u], t);
            t = fma((*q)[qb + 3u], (*k)[kb + 3u], t);
            partial = partial + t;
        }
    }
    return partial;
}

fn fdt_load_kreg_vec4(k: ptr<function, array<f32, 8>>, j: u32, hd: u32, lane: u32) {
    for (var g = 0u; g < FDT_VEC4_GROUPS_MATCH_THE_MK_SG_LANE_TO_DIM_MAP_FOR_BIT_EXACT_SCORES; g = g + 1u) {
        let d4 = (lane + g * FD_LANES) << 2u;
        for (var c = 0u; c < 4u; c = c + 1u) {
            (*k)[g * 4u + c] = 0.0;
            if (d4 + c < hd) {
                (*k)[g * 4u + c] = fdt_k_stage_decoded_once_per_workgroup[j * hd + d4 + c];
            }
        }
    }
}

fn fdt_row_causal_bounds(
    tqs: ptr<function, array<i32, 4>>,
    sqs: ptr<function, array<i32, 4>>,
    total: i32,
    rows: u32,
    warp: u32
) {
    for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
        let qi = warp + rr * FD_WARPS;
        if (qi < rows) {
            let tq = total - i32(rows - 1u - qi);
            (*tqs)[rr] = tq;
            (*sqs)[rr] = fd_mk_start_of(tq);
        } else {
            (*tqs)[rr] = 0;
            (*sqs)[rr] = 0;
        }
    }
}

fn fdt_block_softmax_update(
    sc: ptr<function, array<f32, 32>>,
    rr: u32,
    m_prev: f32,
    l_prev: f32
) -> vec3<f32> {
    var mblk = fd_neg_inf();
    for (var j = 0u; j < FDT_POS; j = j + 1u) {
        mblk = max(mblk, (*sc)[rr * FDT_POS + j]);
    }
    if (mblk > fd_neg_inf()) {
        let m_new = max(m_prev, mblk);
        let corr = fd_exp(m_prev - m_new);
        var wsum = 0.0;
        for (var j = 0u; j < FDT_POS; j = j + 1u) {
            let w = fd_exp((*sc)[rr * FDT_POS + j] - m_new);
            (*sc)[rr * FDT_POS + j] = w * fdt_v_scale_stage[j];
            wsum = wsum + w;
        }
        return vec3<f32>(m_new, fma(l_prev, corr, wsum), corr);
    }
    for (var j = 0u; j < FDT_POS; j = j + 1u) {
        (*sc)[rr * FDT_POS + j] = 0.0;
    }
    return vec3<f32>(m_prev, l_prev, 1.0);
}

fn fdt_accumulate(
    acc: ptr<function, array<f32, 32>>,
    sc: ptr<function, array<f32, 32>>,
    corr: ptr<function, array<f32, 4>>,
    hd: u32,
    lane: u32
) {
    for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            (*acc)[rr * FD_MK_MAX_ACC + i] = (*acc)[rr * FD_MK_MAX_ACC + i] * (*corr)[rr];
        }
    }
    for (var j = 0u; j < FDT_POS; j = j + 1u) {
        var vreg: array<f32, 8>;
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            let d = lane + i * FD_LANES;
            vreg[i] = 0.0;
            if (d < hd) {
                vreg[i] = fdt_v_stage_decoded_once_per_workgroup[j * hd + d];
            }
        }
        for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
            let w = (*sc)[rr * FDT_POS + j];
            for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                (*acc)[rr * FD_MK_MAX_ACC + i] = fma(w, vreg[i], (*acc)[rr * FD_MK_MAX_ACC + i]);
            }
        }
    }
}

fn fdt_write_split_result(
    acc: ptr<function, array<f32, 32>>,
    mreg: ptr<function, array<f32, 4>>,
    lreg: ptr<function, array<f32, 4>>,
    h: u32,
    row0: u32,
    rows: u32,
    split: u32,
    hd: u32,
    lane: u32,
    warp: u32
) {
    for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
        let qi = warp + rr * FD_WARPS;
        if (qi < rows) {
            let slot =
                ((h * fd_params.m_rows + row0 + qi) * fd_params.splits + split) * (hd + 2u);
            let live = (*mreg)[rr] > fd_neg_inf();
            if (lane == 0u) {
                fd_scratch[slot] = (*mreg)[rr];
                fd_scratch[slot + 1u] = select(0.0, fd_round((*lreg)[rr]), live);
            }
            for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                let d = lane + i * FD_LANES;
                if (d < hd) {
                    fd_scratch[slot + 2u + d] =
                        select(0.0, fd_round((*acc)[rr * FD_MK_MAX_ACC + i]), live);
                }
            }
        }
    }
}

@compute @workgroup_size(256)
fn q3w_pf_flash1_fp8kv_tiled_wg(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let nkv = fd_params.n_kv;
    let kvh = h / (fd_params.n_heads / nkv);
    let mrf = fd_params.m_rows;
    let row0 = wg.z * FDT_ROWS;
    let rows = min(FDT_ROWS, mrf - row0);
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    var qreg: array<f32, 32>;
    fdt_load_qreg_vec4(&qreg, h, row0, rows, hd, lane, warp);

    var acc: array<f32, 32>;
    var mreg: array<f32, 4>;
    var lreg: array<f32, 4>;
    for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
        mreg[rr] = fd_neg_inf();
        lreg[rr] = 0.0;
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[rr * FD_MK_MAX_ACC + i] = 0.0;
        }
    }

    let total = i32(fd_params.total) - i32(mrf - row0 - rows);
    var tqs: array<i32, 4>;
    var sqs: array<i32, 4>;
    fdt_row_causal_bounds(&tqs, &sqs, total, rows, warp);
    let start0 = fd_mk_start_of(total - i32(rows - 1u));
    let base = start0 + i32(split * FDT_POS);
    let stride = i32(fd_params.splits * FDT_POS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }

    for (var r = 0; r < rounds; r = r + 1) {
        let p0 = base + r * stride;
        workgroupBarrier();
        fdt_stage_scales(lid, p0, total, nkv, kvh);
        fdt_stage_k_and_v_in_one_pass_so_each_round_needs_two_barriers_not_four(
            lid, p0, total, hd, nkv, kvh
        );
        workgroupBarrier();

        var sc: array<f32, 32>;
        for (var j = 0u; j < FDT_POS; j = j + 1u) {
            let p = p0 + i32(j);
            var kreg: array<f32, 8>;
            fdt_load_kreg_vec4(&kreg, j, hd, lane);
            for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
                let act = p >= sqs[rr] && p < tqs[rr];
                var partial = 0.0;
                if (act) {
                    partial = fdt_score_partial_same_fma_chain_as_the_mk_sg_vec4_path(
                        &qreg, &kreg, rr, hd, lane
                    );
                }
                let s = (fd_warp_sum(lid, partial) * fdt_k_scale_stage[j]) * fd_params.scaling;
                sc[rr * FDT_POS + j] = select(fd_neg_inf(), s, act);
            }
        }

        var corr: array<f32, 4>;
        for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
            let u = fdt_block_softmax_update(&sc, rr, mreg[rr], lreg[rr]);
            mreg[rr] = u.x;
            lreg[rr] = u.y;
            corr[rr] = u.z;
        }

        fdt_accumulate(&acc, &sc, &corr, hd, lane);
    }

    fdt_write_split_result(&acc, &mreg, &lreg, h, row0, rows, split, hd, lane, warp);
}

fn fdt_slotml_update_keeps_one_ml_chain_per_j_slot_like_the_grouped_kernels_8_warps(
    sc: ptr<function, array<f32, 32>>,
    mj: ptr<function, array<f32, 32>>,
    lj: ptr<function, array<f32, 32>>,
    rr: u32,
    macc_prev: f32
) -> vec2<f32> {
    var mblk = fd_neg_inf();
    for (var j = 0u; j < FDT_POS; j = j + 1u) {
        mblk = max(mblk, (*sc)[rr * FDT_POS + j]);
    }
    if (mblk > fd_neg_inf()) {
        let m_new = max(macc_prev, mblk);
        let corr = fd_exp(macc_prev - m_new);
        for (var j = 0u; j < FDT_POS; j = j + 1u) {
            let s = (*sc)[rr * FDT_POS + j];
            var w = 0.0;
            if (s > fd_neg_inf()) {
                let mj_new = max((*mj)[rr * FDT_POS + j], s);
                let cj = fd_exp((*mj)[rr * FDT_POS + j] - mj_new);
                let wj = fd_exp(s - mj_new);
                (*lj)[rr * FDT_POS + j] = fma((*lj)[rr * FDT_POS + j], cj, wj);
                (*mj)[rr * FDT_POS + j] = mj_new;
                w = fd_exp(s - m_new) * fdt_v_scale_stage[j];
            }
            (*sc)[rr * FDT_POS + j] = w;
        }
        return vec2<f32>(m_new, corr);
    }
    for (var j = 0u; j < FDT_POS; j = j + 1u) {
        (*sc)[rr * FDT_POS + j] = 0.0;
    }
    return vec2<f32>(macc_prev, 1.0);
}

fn fdt_slotml_merge_streams_once_at_end_same_guard_and_round_as_fd_stage1_epilogue(
    mj: ptr<function, array<f32, 32>>,
    lj: ptr<function, array<f32, 32>>,
    macc: ptr<function, array<f32, 4>>,
    mreg: ptr<function, array<f32, 4>>,
    lreg: ptr<function, array<f32, 4>>
) {
    for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
        let m_blk = (*macc)[rr];
        var l_blk = 0.0;
        for (var j = 0u; j < FDT_POS; j = j + 1u) {
            if ((*mj)[rr * FDT_POS + j] > fd_neg_inf()) {
                l_blk = l_blk
                    + fd_round((*lj)[rr * FDT_POS + j]
                        * fd_exp((*mj)[rr * FDT_POS + j] - m_blk));
            }
        }
        (*mreg)[rr] = m_blk;
        (*lreg)[rr] = l_blk;
    }
}

@compute @workgroup_size(256)
fn q3w_pf_flash1_fp8kv_tiled_slotml_wg(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let nkv = fd_params.n_kv;
    let kvh = h / (fd_params.n_heads / nkv);
    let mrf = fd_params.m_rows;
    let row0 = wg.z * FDT_ROWS;
    let rows = min(FDT_ROWS, mrf - row0);
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    var qreg: array<f32, 32>;
    fdt_load_qreg_vec4(&qreg, h, row0, rows, hd, lane, warp);

    var acc: array<f32, 32>;
    var mj: array<f32, 32>;
    var lj: array<f32, 32>;
    var macc: array<f32, 4>;
    for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
        macc[rr] = fd_neg_inf();
        for (var j = 0u; j < FDT_POS; j = j + 1u) {
            mj[rr * FDT_POS + j] = fd_neg_inf();
            lj[rr * FDT_POS + j] = 0.0;
        }
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[rr * FD_MK_MAX_ACC + i] = 0.0;
        }
    }

    let total = i32(fd_params.total) - i32(mrf - row0 - rows);
    var tqs: array<i32, 4>;
    var sqs: array<i32, 4>;
    fdt_row_causal_bounds(&tqs, &sqs, total, rows, warp);
    let start0 = fd_mk_start_of(total - i32(rows - 1u));
    let base = start0 + i32(split * FDT_POS);
    let stride = i32(fd_params.splits * FDT_POS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }

    for (var r = 0; r < rounds; r = r + 1) {
        let p0 = base + r * stride;
        workgroupBarrier();
        fdt_stage_scales(lid, p0, total, nkv, kvh);
        fdt_stage_k_and_v_in_one_pass_so_each_round_needs_two_barriers_not_four(
            lid, p0, total, hd, nkv, kvh
        );
        workgroupBarrier();

        var sc: array<f32, 32>;
        for (var j = 0u; j < FDT_POS; j = j + 1u) {
            let p = p0 + i32(j);
            var kreg: array<f32, 8>;
            fdt_load_kreg_vec4(&kreg, j, hd, lane);
            for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
                let act = p >= sqs[rr] && p < tqs[rr];
                var partial = 0.0;
                if (act) {
                    partial = fdt_score_partial_same_fma_chain_as_the_mk_sg_vec4_path(
                        &qreg, &kreg, rr, hd, lane
                    );
                }
                let s = (fd_warp_sum(lid, partial) * fdt_k_scale_stage[j]) * fd_params.scaling;
                sc[rr * FDT_POS + j] = select(fd_neg_inf(), s, act);
            }
        }

        var corr: array<f32, 4>;
        for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
            let u = fdt_slotml_update_keeps_one_ml_chain_per_j_slot_like_the_grouped_kernels_8_warps(
                &sc, &mj, &lj, rr, macc[rr]
            );
            macc[rr] = u.x;
            corr[rr] = u.y;
        }

        fdt_accumulate(&acc, &sc, &corr, hd, lane);
    }

    var mreg: array<f32, 4>;
    var lreg: array<f32, 4>;
    fdt_slotml_merge_streams_once_at_end_same_guard_and_round_as_fd_stage1_epilogue(
        &mj, &lj, &macc, &mreg, &lreg
    );
    fdt_write_split_result(&acc, &mreg, &lreg, h, row0, rows, split, hd, lane, warp);
}

fn fdt_warp_sum_butterfly_same_tree_as_pfl_warp_sum(x: f32) -> f32 {
    var a = x;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

@compute @workgroup_size(256)
fn q3w_pf_flash1_fp8kv_tiled_sg(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let nkv = fd_params.n_kv;
    let kvh = h / (fd_params.n_heads / nkv);
    let mrf = fd_params.m_rows;
    let row0 = wg.z * FDT_ROWS;
    let rows = min(FDT_ROWS, mrf - row0);
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    var qreg: array<f32, 32>;
    fdt_load_qreg_vec4(&qreg, h, row0, rows, hd, lane, warp);

    var acc: array<f32, 32>;
    var mreg: array<f32, 4>;
    var lreg: array<f32, 4>;
    for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
        mreg[rr] = fd_neg_inf();
        lreg[rr] = 0.0;
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[rr * FD_MK_MAX_ACC + i] = 0.0;
        }
    }

    let total = i32(fd_params.total) - i32(mrf - row0 - rows);
    var tqs: array<i32, 4>;
    var sqs: array<i32, 4>;
    fdt_row_causal_bounds(&tqs, &sqs, total, rows, warp);
    let start0 = fd_mk_start_of(total - i32(rows - 1u));
    let base = start0 + i32(split * FDT_POS);
    let stride = i32(fd_params.splits * FDT_POS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }

    for (var r = 0; r < rounds; r = r + 1) {
        let p0 = base + r * stride;
        workgroupBarrier();
        fdt_stage_scales(lid, p0, total, nkv, kvh);
        fdt_stage_k_and_v_in_one_pass_so_each_round_needs_two_barriers_not_four(
            lid, p0, total, hd, nkv, kvh
        );
        workgroupBarrier();

        var sc: array<f32, 32>;
        for (var j = 0u; j < FDT_POS; j = j + 1u) {
            let p = p0 + i32(j);
            var kreg: array<f32, 8>;
            fdt_load_kreg_vec4(&kreg, j, hd, lane);
            for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
                let act = p >= sqs[rr] && p < tqs[rr];
                var partial = 0.0;
                if (act) {
                    partial = fdt_score_partial_same_fma_chain_as_the_mk_sg_vec4_path(
                        &qreg, &kreg, rr, hd, lane
                    );
                }
                let s = (fdt_warp_sum_butterfly_same_tree_as_pfl_warp_sum(partial)
                    * fdt_k_scale_stage[j])
                    * fd_params.scaling;
                sc[rr * FDT_POS + j] = select(fd_neg_inf(), s, act);
            }
        }

        var corr: array<f32, 4>;
        for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
            let u = fdt_block_softmax_update(&sc, rr, mreg[rr], lreg[rr]);
            mreg[rr] = u.x;
            lreg[rr] = u.y;
            corr[rr] = u.z;
        }

        fdt_accumulate(&acc, &sc, &corr, hd, lane);
    }

    fdt_write_split_result(&acc, &mreg, &lreg, h, row0, rows, split, hd, lane, warp);
}

@compute @workgroup_size(256)
fn q3w_pf_flash1_fp8kv_tiled_slotml_sg(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let nkv = fd_params.n_kv;
    let kvh = h / (fd_params.n_heads / nkv);
    let mrf = fd_params.m_rows;
    let row0 = wg.z * FDT_ROWS;
    let rows = min(FDT_ROWS, mrf - row0);
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    var qreg: array<f32, 32>;
    fdt_load_qreg_vec4(&qreg, h, row0, rows, hd, lane, warp);

    var acc: array<f32, 32>;
    var mj: array<f32, 32>;
    var lj: array<f32, 32>;
    var macc: array<f32, 4>;
    for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
        macc[rr] = fd_neg_inf();
        for (var j = 0u; j < FDT_POS; j = j + 1u) {
            mj[rr * FDT_POS + j] = fd_neg_inf();
            lj[rr * FDT_POS + j] = 0.0;
        }
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[rr * FD_MK_MAX_ACC + i] = 0.0;
        }
    }

    let total = i32(fd_params.total) - i32(mrf - row0 - rows);
    var tqs: array<i32, 4>;
    var sqs: array<i32, 4>;
    fdt_row_causal_bounds(&tqs, &sqs, total, rows, warp);
    let start0 = fd_mk_start_of(total - i32(rows - 1u));
    let base = start0 + i32(split * FDT_POS);
    let stride = i32(fd_params.splits * FDT_POS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }

    for (var r = 0; r < rounds; r = r + 1) {
        let p0 = base + r * stride;
        workgroupBarrier();
        fdt_stage_scales(lid, p0, total, nkv, kvh);
        fdt_stage_k_and_v_in_one_pass_so_each_round_needs_two_barriers_not_four(
            lid, p0, total, hd, nkv, kvh
        );
        workgroupBarrier();

        var sc: array<f32, 32>;
        for (var j = 0u; j < FDT_POS; j = j + 1u) {
            let p = p0 + i32(j);
            var kreg: array<f32, 8>;
            fdt_load_kreg_vec4(&kreg, j, hd, lane);
            for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
                let act = p >= sqs[rr] && p < tqs[rr];
                var partial = 0.0;
                if (act) {
                    partial = fdt_score_partial_same_fma_chain_as_the_mk_sg_vec4_path(
                        &qreg, &kreg, rr, hd, lane
                    );
                }
                let s = (fdt_warp_sum_butterfly_same_tree_as_pfl_warp_sum(partial)
                    * fdt_k_scale_stage[j])
                    * fd_params.scaling;
                sc[rr * FDT_POS + j] = select(fd_neg_inf(), s, act);
            }
        }

        var corr: array<f32, 4>;
        for (var rr = 0u; rr < FDT_WARP_ROWS; rr = rr + 1u) {
            let u = fdt_slotml_update_keeps_one_ml_chain_per_j_slot_like_the_grouped_kernels_8_warps(
                &sc, &mj, &lj, rr, macc[rr]
            );
            macc[rr] = u.x;
            corr[rr] = u.y;
        }

        fdt_accumulate(&acc, &sc, &corr, hd, lane);
    }

    var mreg: array<f32, 4>;
    var lreg: array<f32, 4>;
    fdt_slotml_merge_streams_once_at_end_same_guard_and_round_as_fd_stage1_epilogue(
        &mj, &lj, &macc, &mreg, &lreg
    );
    fdt_write_split_result(&acc, &mreg, &lreg, h, row0, rows, split, hd, lane, warp);
}
