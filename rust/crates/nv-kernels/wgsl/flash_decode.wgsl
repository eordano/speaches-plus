struct FdParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    total: u32,
    start: u32,
    splits: u32,
    ring: u32,
    out_bf16: u32,
    scaling: f32,
    pad0: u32,
    fused: u32,
    pad2: u32,
    m_rows: u32,
    window: u32,
    pad3: u32,
    pad4: u32,
};

@group(0) @binding(0) var<storage, read> fd_q: array<f32>;
@group(0) @binding(1) var<storage, read> fd_k_f32: array<f32>;
@group(0) @binding(2) var<storage, read> fd_v_f32: array<f32>;
@group(0) @binding(3) var<storage, read_write> fd_out: array<u32>;
@group(0) @binding(4) var<uniform> fd_params: FdParams;
@group(0) @binding(5) var<storage, read> fd_k_words: array<u32>;
@group(0) @binding(6) var<storage, read> fd_v_words: array<u32>;
@group(0) @binding(7) var<storage, read_write> fd_scratch: array<f32>;
@group(0) @binding(8) var<storage, read> fd_k_scales: array<f32>;
@group(0) @binding(9) var<storage, read> fd_v_scales: array<f32>;
@group(0) @binding(10) var<storage, read> fd_src_k: array<f32>;
@group(0) @binding(11) var<storage, read> fd_src_v: array<f32>;
@group(0) @binding(12) var<storage, read_write> fd_cache_k: array<u32>;
@group(0) @binding(13) var<storage, read_write> fd_cache_v: array<u32>;

const FD_BLOCK: u32 = 256u;
const FD_WARPS: u32 = 8u;
const FD_LANES: u32 = 32u;
const FD_MAX_HD: u32 = 512u;
const FD_MAX_ACC: u32 = 16u;
const FD_LOG2E: f32 = 1.4426950408889634;

var<workgroup> fd_qsh: array<f32, 512>;
var<workgroup> fd_red: array<f32, 256>;
var<workgroup> fd_sm: array<f32, 8>;
var<workgroup> fd_sl: array<f32, 8>;
var<workgroup> fd_sacc: array<f32, 4096>;
var<workgroup> fd_s2w: array<f32, 256>;

fn fd_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn fd_exp(x: f32) -> f32 {
    return exp2(x * FD_LOG2E);
}

fn fd_round(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ fd_params.pad0);
}

fn fd_recip(x: f32) -> f32 {
    let r = 1.0 / x;
    return fma(fma(-x, r, 1.0), r, r);
}

fn fd_warp_sum(lid: u32, x: f32) -> f32 {
    fd_red[lid] = x;
    workgroupBarrier();
    for (var o = 16u; o > 0u; o = o >> 1u) {
        let other = fd_red[lid ^ o];
        workgroupBarrier();
        fd_red[lid] = fd_red[lid] + other;
        workgroupBarrier();
    }
    return fd_red[lid];
}

fn fd_k_bf16(idx: u32) -> f32 {
    let word = fd_k_words[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn fd_v_bf16(idx: u32) -> f32 {
    let word = fd_v_words[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

fn fd_k_fp8(idx: u32) -> f32 {
    return e4m3_decode(byte_at(fd_k_words[idx >> 2u], idx));
}

fn fd_v_fp8(idx: u32) -> f32 {
    return e4m3_decode(byte_at(fd_v_words[idx >> 2u], idx));
}

fn fd_store_out(idx: u32, y: f32) {
    if (fd_params.out_bf16 == 1u) {
        fd_out[idx] = bf16_encode(y);
    } else {
        fd_out[idx] = bitcast<u32>(y);
    }
}

@compute @workgroup_size(256)
fn flash_decode_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let nkv = fd_params.n_kv;
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var d = lid; d < hd; d = d + FD_BLOCK) {
        fd_qsh[d] = fd_q[h * hd + d];
    }
    workgroupBarrier();

    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    var acc4 = 0.0;
    var acc5 = 0.0;
    var acc6 = 0.0;
    var acc7 = 0.0;
    var acc8 = 0.0;
    var acc9 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc12 = 0.0;
    var acc13 = 0.0;
    var acc14 = 0.0;
    var acc15 = 0.0;
    var m = fd_neg_inf();
    var l = 0.0;

    let total = fd_params.total;
    let start = fd_params.start;
    var rounds = 0u;
    if (total > start) {
        rounds = (total - start + FD_WARPS - 1u) / FD_WARPS;
    }
    let use_vec4 = (hd & 3u) == 0u;

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = start + warp + r * FD_WARPS;
        let live = p < total;
        var partial = 0.0;
        if (live) {
            let kbase = (p * nkv + kvh) * hd;
            if (use_vec4) {
                let n4 = hd >> 2u;
                for (var j = lane; j < n4; j = j + FD_LANES) {
                    let qb = j * 4u;
                    let kb = kbase + qb;
                    var t = fd_qsh[qb + 1u] * fd_k_f32[kb + 1u];
                    t = fma(fd_qsh[qb], fd_k_f32[kb], t);
                    t = fma(fd_qsh[qb + 2u], fd_k_f32[kb + 2u], t);
                    t = fma(fd_qsh[qb + 3u], fd_k_f32[kb + 3u], t);
                    partial = partial + t;
                }
            } else {
                for (var d = lane; d < hd; d = d + FD_LANES) {
                    partial = fma(fd_qsh[d], fd_k_f32[kbase + d], partial);
                }
            }
        }
        let score = fd_warp_sum(lid, partial) * fd_params.scaling;
        if (live) {
            let m_new = max(m, score);
            let corr = fd_exp(m - m_new);
            let w = fd_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (p * nkv + kvh) * hd;
            {
                let d = lane + 0u * FD_LANES;
                if (d < hd) {
                    acc0 = fma(acc0, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 1u * FD_LANES;
                if (d < hd) {
                    acc1 = fma(acc1, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 2u * FD_LANES;
                if (d < hd) {
                    acc2 = fma(acc2, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 3u * FD_LANES;
                if (d < hd) {
                    acc3 = fma(acc3, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 4u * FD_LANES;
                if (d < hd) {
                    acc4 = fma(acc4, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 5u * FD_LANES;
                if (d < hd) {
                    acc5 = fma(acc5, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 6u * FD_LANES;
                if (d < hd) {
                    acc6 = fma(acc6, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 7u * FD_LANES;
                if (d < hd) {
                    acc7 = fma(acc7, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 8u * FD_LANES;
                if (d < hd) {
                    acc8 = fma(acc8, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 9u * FD_LANES;
                if (d < hd) {
                    acc9 = fma(acc9, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 10u * FD_LANES;
                if (d < hd) {
                    acc10 = fma(acc10, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 11u * FD_LANES;
                if (d < hd) {
                    acc11 = fma(acc11, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 12u * FD_LANES;
                if (d < hd) {
                    acc12 = fma(acc12, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 13u * FD_LANES;
                if (d < hd) {
                    acc13 = fma(acc13, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 14u * FD_LANES;
                if (d < hd) {
                    acc14 = fma(acc14, corr, w * fd_v_f32[vbase + d]);
                }
            }
            {
                let d = lane + 15u * FD_LANES;
                if (d < hd) {
                    acc15 = fma(acc15, corr, w * fd_v_f32[vbase + d]);
                }
            }
            m = m_new;
        }
    }

    if (lane == 0u) {
        fd_sm[warp] = m;
        fd_sl[warp] = l;
    }
    {
        let d = lane + 0u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc0;
        }
    }
    {
        let d = lane + 1u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc1;
        }
    }
    {
        let d = lane + 2u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc2;
        }
    }
    {
        let d = lane + 3u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc3;
        }
    }
    {
        let d = lane + 4u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc4;
        }
    }
    {
        let d = lane + 5u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc5;
        }
    }
    {
        let d = lane + 6u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc6;
        }
    }
    {
        let d = lane + 7u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc7;
        }
    }
    {
        let d = lane + 8u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc8;
        }
    }
    {
        let d = lane + 9u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc9;
        }
    }
    {
        let d = lane + 10u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc10;
        }
    }
    {
        let d = lane + 11u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc11;
        }
    }
    {
        let d = lane + 12u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc12;
        }
    }
    {
        let d = lane + 13u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc13;
        }
    }
    {
        let d = lane + 14u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc14;
        }
    }
    {
        let d = lane + 15u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc15;
        }
    }
    workgroupBarrier();

    if (warp == 0u) {
        var m_glob = fd_neg_inf();
        for (var w = 0u; w < FD_WARPS; w = w + 1u) {
            m_glob = max(m_glob, fd_sm[w]);
        }
        var l_glob = 0.0;
        for (var w = 0u; w < FD_WARPS; w = w + 1u) {
            l_glob = fma(fd_sl[w], fd_exp(fd_sm[w] - m_glob), l_glob);
        }
        var inv_l = 0.0;
        if (l_glob > 0.0) {
            inv_l = fd_recip(l_glob);
        }
        var scale: array<f32, 8>;
        for (var w = 0u; w < FD_WARPS; w = w + 1u) {
            scale[w] = fd_exp(fd_sm[w] - m_glob);
        }
        for (var d = lane; d < hd; d = d + FD_LANES) {
            var a = 0.0;
            for (var w = 0u; w < FD_WARPS; w = w + 1u) {
                a = fma(fd_sacc[w * FD_MAX_HD + d], scale[w], a);
            }
            fd_store_out(h * hd + d, a * inv_l);
        }
    }
}

fn fd_stage1_epilogue(lid: u32, lane: u32, warp: u32, hd: u32, slot: u32, m: f32, l: f32) {
    if (lane == 0u) {
        fd_sm[warp] = m;
        fd_sl[warp] = l;
    }
    workgroupBarrier();
    if (warp == 0u) {
        var m_blk = fd_neg_inf();
        for (var w = 0u; w < FD_WARPS; w = w + 1u) {
            m_blk = max(m_blk, fd_sm[w]);
        }
        var l_blk = 0.0;
        for (var w = 0u; w < FD_WARPS; w = w + 1u) {
            if (fd_sm[w] > fd_neg_inf()) {
                l_blk = l_blk + fd_round(fd_sl[w] * fd_exp(fd_sm[w] - m_blk));
            }
        }
        if (lane == 0u) {
            fd_scratch[slot] = m_blk;
            fd_scratch[slot + 1u] = l_blk;
        }
    }
    var m_blk = fd_neg_inf();
    for (var w = 0u; w < FD_WARPS; w = w + 1u) {
        m_blk = max(m_blk, fd_sm[w]);
    }
    for (var d = lid; d < hd; d = d + FD_BLOCK) {
        var a = 0.0;
        for (var w = 0u; w < FD_WARPS; w = w + 1u) {
            if (fd_sm[w] > fd_neg_inf()) {
                a = a + fd_round(fd_sacc[w * FD_MAX_HD + d] * fd_exp(fd_sm[w] - m_blk));
            }
        }
        fd_scratch[slot + 2u + d] = a;
    }
}

@compute @workgroup_size(256)
fn flash_splitk_stage1_bf16kv(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var d = lid; d < hd; d = d + FD_BLOCK) {
        fd_qsh[d] = fd_q[h * hd + d];
    }
    workgroupBarrier();

    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    var acc4 = 0.0;
    var acc5 = 0.0;
    var acc6 = 0.0;
    var acc7 = 0.0;
    var acc8 = 0.0;
    var acc9 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc12 = 0.0;
    var acc13 = 0.0;
    var acc14 = 0.0;
    var acc15 = 0.0;
    var m = fd_neg_inf();
    var l = 0.0;

    let total = fd_params.total;
    let base = fd_params.start + split * FD_WARPS;
    let stride = fd_params.splits * FD_WARPS;
    var rounds = 0u;
    if (total > base) {
        rounds = (total - base + stride - 1u) / stride;
    }
    let use_vec8 = (hd & 7u) == 0u;
    let fused_vecv = fd_params.fused == 1u && (hd % 256u) == 0u && (hd / FD_LANES) <= FD_MAX_ACC;

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = base + warp + r * stride;
        let live = p < total;
        var sp = p;
        if (fd_params.ring > 0u) {
            sp = p % fd_params.ring;
        }
        var partial = 0.0;
        if (live) {
            let kbase = (sp * nkv + kvh) * hd;
            if (use_vec8) {
                let n8 = hd >> 3u;
                for (var j = lane; j < n8; j = j + FD_LANES) {
                    let qb = j * 8u;
                    let kb = kbase + qb;
                    for (var t = 0u; t < 4u; t = t + 1u) {
                        let kx = fd_k_bf16(kb + 2u * t);
                        let ky = fd_k_bf16(kb + 2u * t + 1u);
                        let pair = fma(kx, fd_qsh[qb + 2u * t], ky * fd_qsh[qb + 2u * t + 1u]);
                        partial = partial + pair;
                    }
                }
            } else {
                for (var d = lane; d < hd; d = d + FD_LANES) {
                    partial = fma(fd_qsh[d], fd_k_bf16(kbase + d), partial);
                }
            }
        }
        let score = fd_warp_sum(lid, partial) * fd_params.scaling;
        if (live) {
            let m_new = max(m, score);
            let corr = fd_exp(m - m_new);
            let w = fd_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (sp * nkv + kvh) * hd;
            if (fused_vecv) {
                {
                    let d = lane + 0u * FD_LANES;
                    if (d < hd) {
                        acc0 = fma(w, fd_v_bf16(vbase + d), acc0 * corr);
                    }
                }
                {
                    let d = lane + 1u * FD_LANES;
                    if (d < hd) {
                        acc1 = fma(w, fd_v_bf16(vbase + d), acc1 * corr);
                    }
                }
                {
                    let d = lane + 2u * FD_LANES;
                    if (d < hd) {
                        acc2 = fma(w, fd_v_bf16(vbase + d), acc2 * corr);
                    }
                }
                {
                    let d = lane + 3u * FD_LANES;
                    if (d < hd) {
                        acc3 = fma(w, fd_v_bf16(vbase + d), acc3 * corr);
                    }
                }
                {
                    let d = lane + 4u * FD_LANES;
                    if (d < hd) {
                        acc4 = fma(w, fd_v_bf16(vbase + d), acc4 * corr);
                    }
                }
                {
                    let d = lane + 5u * FD_LANES;
                    if (d < hd) {
                        acc5 = fma(w, fd_v_bf16(vbase + d), acc5 * corr);
                    }
                }
                {
                    let d = lane + 6u * FD_LANES;
                    if (d < hd) {
                        acc6 = fma(w, fd_v_bf16(vbase + d), acc6 * corr);
                    }
                }
                {
                    let d = lane + 7u * FD_LANES;
                    if (d < hd) {
                        acc7 = fma(w, fd_v_bf16(vbase + d), acc7 * corr);
                    }
                }
                {
                    let d = lane + 8u * FD_LANES;
                    if (d < hd) {
                        acc8 = fma(w, fd_v_bf16(vbase + d), acc8 * corr);
                    }
                }
                {
                    let d = lane + 9u * FD_LANES;
                    if (d < hd) {
                        acc9 = fma(w, fd_v_bf16(vbase + d), acc9 * corr);
                    }
                }
                {
                    let d = lane + 10u * FD_LANES;
                    if (d < hd) {
                        acc10 = fma(w, fd_v_bf16(vbase + d), acc10 * corr);
                    }
                }
                {
                    let d = lane + 11u * FD_LANES;
                    if (d < hd) {
                        acc11 = fma(w, fd_v_bf16(vbase + d), acc11 * corr);
                    }
                }
                {
                    let d = lane + 12u * FD_LANES;
                    if (d < hd) {
                        acc12 = fma(w, fd_v_bf16(vbase + d), acc12 * corr);
                    }
                }
                {
                    let d = lane + 13u * FD_LANES;
                    if (d < hd) {
                        acc13 = fma(w, fd_v_bf16(vbase + d), acc13 * corr);
                    }
                }
                {
                    let d = lane + 14u * FD_LANES;
                    if (d < hd) {
                        acc14 = fma(w, fd_v_bf16(vbase + d), acc14 * corr);
                    }
                }
                {
                    let d = lane + 15u * FD_LANES;
                    if (d < hd) {
                        acc15 = fma(w, fd_v_bf16(vbase + d), acc15 * corr);
                    }
                }
            } else {
                {
                    let d = lane + 0u * FD_LANES;
                    if (d < hd) {
                        acc0 = fma(acc0, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 1u * FD_LANES;
                    if (d < hd) {
                        acc1 = fma(acc1, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 2u * FD_LANES;
                    if (d < hd) {
                        acc2 = fma(acc2, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 3u * FD_LANES;
                    if (d < hd) {
                        acc3 = fma(acc3, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 4u * FD_LANES;
                    if (d < hd) {
                        acc4 = fma(acc4, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 5u * FD_LANES;
                    if (d < hd) {
                        acc5 = fma(acc5, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 6u * FD_LANES;
                    if (d < hd) {
                        acc6 = fma(acc6, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 7u * FD_LANES;
                    if (d < hd) {
                        acc7 = fma(acc7, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 8u * FD_LANES;
                    if (d < hd) {
                        acc8 = fma(acc8, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 9u * FD_LANES;
                    if (d < hd) {
                        acc9 = fma(acc9, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 10u * FD_LANES;
                    if (d < hd) {
                        acc10 = fma(acc10, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 11u * FD_LANES;
                    if (d < hd) {
                        acc11 = fma(acc11, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 12u * FD_LANES;
                    if (d < hd) {
                        acc12 = fma(acc12, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 13u * FD_LANES;
                    if (d < hd) {
                        acc13 = fma(acc13, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 14u * FD_LANES;
                    if (d < hd) {
                        acc14 = fma(acc14, corr, w * fd_v_bf16(vbase + d));
                    }
                }
                {
                    let d = lane + 15u * FD_LANES;
                    if (d < hd) {
                        acc15 = fma(acc15, corr, w * fd_v_bf16(vbase + d));
                    }
                }
            }
            m = m_new;
        }
    }

    {
        let d = lane + 0u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc0;
        }
    }
    {
        let d = lane + 1u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc1;
        }
    }
    {
        let d = lane + 2u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc2;
        }
    }
    {
        let d = lane + 3u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc3;
        }
    }
    {
        let d = lane + 4u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc4;
        }
    }
    {
        let d = lane + 5u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc5;
        }
    }
    {
        let d = lane + 6u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc6;
        }
    }
    {
        let d = lane + 7u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc7;
        }
    }
    {
        let d = lane + 8u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc8;
        }
    }
    {
        let d = lane + 9u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc9;
        }
    }
    {
        let d = lane + 10u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc10;
        }
    }
    {
        let d = lane + 11u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc11;
        }
    }
    {
        let d = lane + 12u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc12;
        }
    }
    {
        let d = lane + 13u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc13;
        }
    }
    {
        let d = lane + 14u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc14;
        }
    }
    {
        let d = lane + 15u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc15;
        }
    }
    let slot = (h * fd_params.splits + split) * (hd + 2u);
    fd_stage1_epilogue(lid, lane, warp, hd, slot, m, l);
}

@compute @workgroup_size(256)
fn flash_splitk_stage1_fp8kv(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var d = lid; d < hd; d = d + FD_BLOCK) {
        fd_qsh[d] = fd_q[h * hd + d];
    }
    workgroupBarrier();

    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    var acc4 = 0.0;
    var acc5 = 0.0;
    var acc6 = 0.0;
    var acc7 = 0.0;
    var acc8 = 0.0;
    var acc9 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc12 = 0.0;
    var acc13 = 0.0;
    var acc14 = 0.0;
    var acc15 = 0.0;
    var m = fd_neg_inf();
    var l = 0.0;

    let total = fd_params.total;
    let base = fd_params.start + split * FD_WARPS;
    let stride = fd_params.splits * FD_WARPS;
    var rounds = 0u;
    if (total > base) {
        rounds = (total - base + stride - 1u) / stride;
    }
    let use_vec4 = (hd & 3u) == 0u;

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = base + warp + r * stride;
        let live = p < total;
        var sp = p;
        if (fd_params.ring > 0u) {
            sp = p % fd_params.ring;
        }
        var partial = 0.0;
        var ks = 0.0;
        if (live) {
            let kbase = (sp * nkv + kvh) * hd;
            ks = fd_k_scales[sp * nkv + kvh];
            if (use_vec4) {
                let n4 = hd >> 2u;
                for (var j = lane; j < n4; j = j + FD_LANES) {
                    let qb = j * 4u;
                    let kb = kbase + qb;
                    let f0 = fd_k_fp8(kb);
                    let f1 = fd_k_fp8(kb + 1u);
                    let f2 = fd_k_fp8(kb + 2u);
                    let f3 = fd_k_fp8(kb + 3u);
                    var t = fd_qsh[qb + 1u] * f1;
                    t = fma(fd_qsh[qb], f0, t);
                    t = fma(fd_qsh[qb + 2u], f2, t);
                    t = fma(fd_qsh[qb + 3u], f3, t);
                    partial = partial + t;
                }
            } else {
                for (var d = lane; d < hd; d = d + FD_LANES) {
                    partial = fma(fd_qsh[d], fd_k_fp8(kbase + d), partial);
                }
            }
        }
        let score = (fd_warp_sum(lid, partial) * ks) * fd_params.scaling;
        if (live) {
            let m_new = max(m, score);
            let corr = fd_exp(m - m_new);
            let w = fd_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (sp * nkv + kvh) * hd;
            let w_v = w * fd_v_scales[sp * nkv + kvh];
            {
                let d = lane + 0u * FD_LANES;
                if (d < hd) {
                    acc0 = fma(w_v, fd_v_fp8(vbase + d), acc0 * corr);
                }
            }
            {
                let d = lane + 1u * FD_LANES;
                if (d < hd) {
                    acc1 = fma(w_v, fd_v_fp8(vbase + d), acc1 * corr);
                }
            }
            {
                let d = lane + 2u * FD_LANES;
                if (d < hd) {
                    acc2 = fma(w_v, fd_v_fp8(vbase + d), acc2 * corr);
                }
            }
            {
                let d = lane + 3u * FD_LANES;
                if (d < hd) {
                    acc3 = fma(w_v, fd_v_fp8(vbase + d), acc3 * corr);
                }
            }
            {
                let d = lane + 4u * FD_LANES;
                if (d < hd) {
                    acc4 = fma(w_v, fd_v_fp8(vbase + d), acc4 * corr);
                }
            }
            {
                let d = lane + 5u * FD_LANES;
                if (d < hd) {
                    acc5 = fma(w_v, fd_v_fp8(vbase + d), acc5 * corr);
                }
            }
            {
                let d = lane + 6u * FD_LANES;
                if (d < hd) {
                    acc6 = fma(w_v, fd_v_fp8(vbase + d), acc6 * corr);
                }
            }
            {
                let d = lane + 7u * FD_LANES;
                if (d < hd) {
                    acc7 = fma(w_v, fd_v_fp8(vbase + d), acc7 * corr);
                }
            }
            {
                let d = lane + 8u * FD_LANES;
                if (d < hd) {
                    acc8 = fma(w_v, fd_v_fp8(vbase + d), acc8 * corr);
                }
            }
            {
                let d = lane + 9u * FD_LANES;
                if (d < hd) {
                    acc9 = fma(w_v, fd_v_fp8(vbase + d), acc9 * corr);
                }
            }
            {
                let d = lane + 10u * FD_LANES;
                if (d < hd) {
                    acc10 = fma(w_v, fd_v_fp8(vbase + d), acc10 * corr);
                }
            }
            {
                let d = lane + 11u * FD_LANES;
                if (d < hd) {
                    acc11 = fma(w_v, fd_v_fp8(vbase + d), acc11 * corr);
                }
            }
            {
                let d = lane + 12u * FD_LANES;
                if (d < hd) {
                    acc12 = fma(w_v, fd_v_fp8(vbase + d), acc12 * corr);
                }
            }
            {
                let d = lane + 13u * FD_LANES;
                if (d < hd) {
                    acc13 = fma(w_v, fd_v_fp8(vbase + d), acc13 * corr);
                }
            }
            {
                let d = lane + 14u * FD_LANES;
                if (d < hd) {
                    acc14 = fma(w_v, fd_v_fp8(vbase + d), acc14 * corr);
                }
            }
            {
                let d = lane + 15u * FD_LANES;
                if (d < hd) {
                    acc15 = fma(w_v, fd_v_fp8(vbase + d), acc15 * corr);
                }
            }
            m = m_new;
        }
    }

    {
        let d = lane + 0u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc0;
        }
    }
    {
        let d = lane + 1u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc1;
        }
    }
    {
        let d = lane + 2u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc2;
        }
    }
    {
        let d = lane + 3u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc3;
        }
    }
    {
        let d = lane + 4u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc4;
        }
    }
    {
        let d = lane + 5u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc5;
        }
    }
    {
        let d = lane + 6u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc6;
        }
    }
    {
        let d = lane + 7u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc7;
        }
    }
    {
        let d = lane + 8u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc8;
        }
    }
    {
        let d = lane + 9u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc9;
        }
    }
    {
        let d = lane + 10u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc10;
        }
    }
    {
        let d = lane + 11u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc11;
        }
    }
    {
        let d = lane + 12u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc12;
        }
    }
    {
        let d = lane + 13u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc13;
        }
    }
    {
        let d = lane + 14u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc14;
        }
    }
    {
        let d = lane + 15u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc15;
        }
    }
    let slot = (h * fd_params.splits + split) * (hd + 2u);
    fd_stage1_epilogue(lid, lane, warp, hd, slot, m, l);
}

fn fd_k_fp8_sd(idx: u32) -> f32 {
    return e4m3_shift_decode_scale_must_carry_2pow120(byte_at(fd_k_words[idx >> 2u], idx));
}

fn fd_v_fp8_sd(idx: u32) -> f32 {
    return e4m3_shift_decode_scale_must_carry_2pow120(byte_at(fd_v_words[idx >> 2u], idx));
}

@compute @workgroup_size(256)
fn flash_splitk_stage1_fp8kv_sd(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var d = lid; d < hd; d = d + FD_BLOCK) {
        fd_qsh[d] = fd_q[h * hd + d];
    }
    workgroupBarrier();

    var acc0 = 0.0;
    var acc1 = 0.0;
    var acc2 = 0.0;
    var acc3 = 0.0;
    var acc4 = 0.0;
    var acc5 = 0.0;
    var acc6 = 0.0;
    var acc7 = 0.0;
    var acc8 = 0.0;
    var acc9 = 0.0;
    var acc10 = 0.0;
    var acc11 = 0.0;
    var acc12 = 0.0;
    var acc13 = 0.0;
    var acc14 = 0.0;
    var acc15 = 0.0;
    var m = fd_neg_inf();
    var l = 0.0;

    let total = fd_params.total;
    let base = fd_params.start + split * FD_WARPS;
    let stride = fd_params.splits * FD_WARPS;
    var rounds = 0u;
    if (total > base) {
        rounds = (total - base + stride - 1u) / stride;
    }
    let use_vec4 = (hd & 3u) == 0u;

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = base + warp + r * stride;
        let live = p < total;
        var sp = p;
        if (fd_params.ring > 0u) {
            sp = p % fd_params.ring;
        }
        var partial = 0.0;
        var ks = 0.0;
        if (live) {
            let kbase = (sp * nkv + kvh) * hd;
            ks = fd_k_scales[sp * nkv + kvh] * bitcast<f32>(0x7B800000u);
            if (use_vec4) {
                let n4 = hd >> 2u;
                for (var j = lane; j < n4; j = j + FD_LANES) {
                    let qb = j * 4u;
                    let kb = kbase + qb;
                    let f0 = fd_k_fp8_sd(kb);
                    let f1 = fd_k_fp8_sd(kb + 1u);
                    let f2 = fd_k_fp8_sd(kb + 2u);
                    let f3 = fd_k_fp8_sd(kb + 3u);
                    var t = fd_qsh[qb + 1u] * f1;
                    t = fma(fd_qsh[qb], f0, t);
                    t = fma(fd_qsh[qb + 2u], f2, t);
                    t = fma(fd_qsh[qb + 3u], f3, t);
                    partial = partial + t;
                }
            } else {
                for (var d = lane; d < hd; d = d + FD_LANES) {
                    partial = fma(fd_qsh[d], fd_k_fp8_sd(kbase + d), partial);
                }
            }
        }
        let score = (fd_warp_sum(lid, partial) * ks) * fd_params.scaling;
        if (live) {
            let m_new = max(m, score);
            let corr = fd_exp(m - m_new);
            let w = fd_exp(score - m_new);
            l = fma(l, corr, w);
            let vbase = (sp * nkv + kvh) * hd;
            let w_v = w * fd_v_scales[sp * nkv + kvh] * bitcast<f32>(0x7B800000u);
            {
                let d = lane + 0u * FD_LANES;
                if (d < hd) {
                    acc0 = fma(w_v, fd_v_fp8_sd(vbase + d), acc0 * corr);
                }
            }
            {
                let d = lane + 1u * FD_LANES;
                if (d < hd) {
                    acc1 = fma(w_v, fd_v_fp8_sd(vbase + d), acc1 * corr);
                }
            }
            {
                let d = lane + 2u * FD_LANES;
                if (d < hd) {
                    acc2 = fma(w_v, fd_v_fp8_sd(vbase + d), acc2 * corr);
                }
            }
            {
                let d = lane + 3u * FD_LANES;
                if (d < hd) {
                    acc3 = fma(w_v, fd_v_fp8_sd(vbase + d), acc3 * corr);
                }
            }
            {
                let d = lane + 4u * FD_LANES;
                if (d < hd) {
                    acc4 = fma(w_v, fd_v_fp8_sd(vbase + d), acc4 * corr);
                }
            }
            {
                let d = lane + 5u * FD_LANES;
                if (d < hd) {
                    acc5 = fma(w_v, fd_v_fp8_sd(vbase + d), acc5 * corr);
                }
            }
            {
                let d = lane + 6u * FD_LANES;
                if (d < hd) {
                    acc6 = fma(w_v, fd_v_fp8_sd(vbase + d), acc6 * corr);
                }
            }
            {
                let d = lane + 7u * FD_LANES;
                if (d < hd) {
                    acc7 = fma(w_v, fd_v_fp8_sd(vbase + d), acc7 * corr);
                }
            }
            {
                let d = lane + 8u * FD_LANES;
                if (d < hd) {
                    acc8 = fma(w_v, fd_v_fp8_sd(vbase + d), acc8 * corr);
                }
            }
            {
                let d = lane + 9u * FD_LANES;
                if (d < hd) {
                    acc9 = fma(w_v, fd_v_fp8_sd(vbase + d), acc9 * corr);
                }
            }
            {
                let d = lane + 10u * FD_LANES;
                if (d < hd) {
                    acc10 = fma(w_v, fd_v_fp8_sd(vbase + d), acc10 * corr);
                }
            }
            {
                let d = lane + 11u * FD_LANES;
                if (d < hd) {
                    acc11 = fma(w_v, fd_v_fp8_sd(vbase + d), acc11 * corr);
                }
            }
            {
                let d = lane + 12u * FD_LANES;
                if (d < hd) {
                    acc12 = fma(w_v, fd_v_fp8_sd(vbase + d), acc12 * corr);
                }
            }
            {
                let d = lane + 13u * FD_LANES;
                if (d < hd) {
                    acc13 = fma(w_v, fd_v_fp8_sd(vbase + d), acc13 * corr);
                }
            }
            {
                let d = lane + 14u * FD_LANES;
                if (d < hd) {
                    acc14 = fma(w_v, fd_v_fp8_sd(vbase + d), acc14 * corr);
                }
            }
            {
                let d = lane + 15u * FD_LANES;
                if (d < hd) {
                    acc15 = fma(w_v, fd_v_fp8_sd(vbase + d), acc15 * corr);
                }
            }
            m = m_new;
        }
    }

    {
        let d = lane + 0u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc0;
        }
    }
    {
        let d = lane + 1u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc1;
        }
    }
    {
        let d = lane + 2u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc2;
        }
    }
    {
        let d = lane + 3u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc3;
        }
    }
    {
        let d = lane + 4u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc4;
        }
    }
    {
        let d = lane + 5u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc5;
        }
    }
    {
        let d = lane + 6u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc6;
        }
    }
    {
        let d = lane + 7u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc7;
        }
    }
    {
        let d = lane + 8u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc8;
        }
    }
    {
        let d = lane + 9u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc9;
        }
    }
    {
        let d = lane + 10u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc10;
        }
    }
    {
        let d = lane + 11u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc11;
        }
    }
    {
        let d = lane + 12u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc12;
        }
    }
    {
        let d = lane + 13u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc13;
        }
    }
    {
        let d = lane + 14u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc14;
        }
    }
    {
        let d = lane + 15u * FD_LANES;
        if (d < hd) {
            fd_sacc[warp * FD_MAX_HD + d] = acc15;
        }
    }
    let slot = (h * fd_params.splits + split) * (hd + 2u);
    fd_stage1_epilogue(lid, lane, warp, hd, slot, m, l);
}

@compute @workgroup_size(256)
fn flash_splitk_stage2(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let splits = fd_params.splits;
    let stride = hd + 2u;
    let base = h * splits * stride;

    var m_glob = fd_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) {
        m_glob = max(m_glob, fd_scratch[base + s * stride]);
    }
    for (var s = tid.x; s < splits; s = s + FD_BLOCK) {
        fd_s2w[s] = fd_split_scale(base, stride, s, m_glob);
    }
    workgroupBarrier();
    var l_glob = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        l_glob = fma(fd_scratch[base + s * stride + 1u], fd_s2w[s], l_glob);
    }
    var inv_l = 0.0;
    if (l_glob > 0.0) {
        inv_l = fd_recip(l_glob);
    }
    for (var d = tid.x; d < hd; d = d + FD_BLOCK) {
        var a = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            a = fma(fd_scratch[base + s * stride + 2u + d], fd_s2w[s], a);
        }
        fd_store_out(h * hd + d, a * inv_l);
    }
}

fn fd_split_scale(base: u32, stride: u32, s: u32, m_glob: f32) -> f32 {
    let p0 = fd_scratch[base + s * stride];
    var sc = 0.0;
    if (p0 > fd_neg_inf()) {
        sc = fd_exp(p0 - m_glob);
    }
    return sc;
}

@compute @workgroup_size(256)
fn flash_splitk_stage2_u(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let splits = fd_params.splits;
    let stride = hd + 2u;
    let base = h * splits * stride;

    var m_glob = fd_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) {
        m_glob = max(m_glob, fd_scratch[base + s * stride]);
    }
    var l_glob = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        let sc = fd_split_scale(base, stride, s, m_glob);
        l_glob = fma(fd_scratch[base + s * stride + 1u], sc, l_glob);
    }
    var inv_l = 0.0;
    if (l_glob > 0.0) {
        inv_l = fd_recip(l_glob);
    }
    for (var d = tid.x; d < hd; d = d + FD_BLOCK) {
        var a = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            let sc = fd_split_scale(base, stride, s, m_glob);
            a = fma(fd_scratch[base + s * stride + 2u + d], sc, a);
        }
        fd_store_out(h * hd + d, a * inv_l);
    }
}

@compute @workgroup_size(256)
fn flash_splitk_stage2_pk_rt(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= fd_params.n_heads) {
        return;
    }
    let hd = fd_params.head_dim;
    let splits = fd_params.splits;
    let stride = hd + 2u;
    let base = h * splits * stride;

    var m_glob = fd_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) {
        m_glob = max(m_glob, fd_scratch[base + s * stride]);
    }
    var l_glob = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        let sc = fd_split_scale(base, stride, s, m_glob);
        l_glob = fma(fd_scratch[base + s * stride + 1u], sc, l_glob);
    }
    var inv_l = 0.0;
    if (l_glob > 0.0) {
        inv_l = fd_recip(l_glob);
    }
    let hw = hd >> 1u;
    for (var w = tid.x; w < hw; w = w + FD_BLOCK) {
        let d0 = w * 2u;
        var a0 = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            let sc = fd_split_scale(base, stride, s, m_glob);
            a0 = fma(fd_scratch[base + s * stride + 2u + d0], sc, a0);
        }
        var a1 = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            let sc = fd_split_scale(base, stride, s, m_glob);
            a1 = fma(fd_scratch[base + s * stride + 2u + d0 + 1u], sc, a1);
        }
        fd_out[h * hw + w] = (bf16_encode(a0 * inv_l) & 0xffffu)
            | ((bf16_encode(a1 * inv_l) & 0xffffu) << 16u);
    }
}

var<workgroup> fd_qsh_mk: array<f32, 2048>;

const FD_MK_MAX_HD: u32 = 256u;
const FD_MK_MAX_ACC: u32 = 8u;

fn fd_mk_start_of(tq: i32) -> i32 {
    let win = i32(fd_params.window);
    if (win > 0 && tq > win) {
        return tq - win;
    }
    return 0;
}

@compute @workgroup_size(256)
fn flash_splitk_stage1_bf16kv_mk(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let mr = fd_params.m_rows;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var t = lid; t < mr * hd; t = t + FD_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        fd_qsh_mk[t] = fd_q[(qi * fd_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    var acc: array<f32, 64>;
    var mreg: array<f32, 8>;
    var lreg: array<f32, 8>;
    for (var qi = 0u; qi < 8u; qi = qi + 1u) {
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[qi * FD_MK_MAX_ACC + i] = 0.0;
        }
        mreg[qi] = fd_neg_inf();
        lreg[qi] = 0.0;
    }

    let total = i32(fd_params.total);
    let start0 = fd_mk_start_of(total - i32(mr - 1u));
    let base = start0 + i32(split * FD_WARPS);
    let stride = i32(fd_params.splits * FD_WARPS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }
    let use_vec8 = (hd & 7u) == 0u;
    let hot = hd == 256u;

    for (var r = 0; r < rounds; r = r + 1) {
        let p = base + i32(warp) + r * stride;
        let live = p < total;
        var kbase = 0u;
        if (live) {
            kbase = (u32(p) * nkv + kvh) * hd;
        }
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let tq = total - i32(mr - 1u - qi);
            let sq = fd_mk_start_of(tq);
            let act = live && p >= sq && p < tq;
            let qoff = qi * hd;
            var partial = 0.0;
            if (act) {
                if (use_vec8) {
                    let n8 = hd >> 3u;
                    for (var j = lane; j < n8; j = j + FD_LANES) {
                        let qb = qoff + j * 8u;
                        let kb = kbase + j * 8u;
                        for (var t = 0u; t < 4u; t = t + 1u) {
                            let kx = fd_k_bf16(kb + 2u * t);
                            let ky = fd_k_bf16(kb + 2u * t + 1u);
                            let pair = fma(kx, fd_qsh_mk[qb + 2u * t], ky * fd_qsh_mk[qb + 2u * t + 1u]);
                            partial = partial + pair;
                        }
                    }
                } else {
                    for (var d = lane; d < hd; d = d + FD_LANES) {
                        partial = fma(fd_qsh_mk[qoff + d], fd_k_bf16(kbase + d), partial);
                    }
                }
            }
            let score = fd_warp_sum(lid, partial);
            if (act) {
                let m_new = max(mreg[qi], score);
                let corr = fd_exp(mreg[qi] - m_new);
                let w = fd_exp(score - m_new);
                lreg[qi] = fma(lreg[qi], corr, w);
                if (hot) {
                    for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                        let d = lane + i * FD_LANES;
                        if (d < hd) {
                            acc[qi * FD_MK_MAX_ACC + i] =
                                fma(w, fd_v_bf16(kbase + d), acc[qi * FD_MK_MAX_ACC + i] * corr);
                        }
                    }
                } else {
                    for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                        let d = lane + i * FD_LANES;
                        if (d < hd) {
                            acc[qi * FD_MK_MAX_ACC + i] =
                                fma(acc[qi * FD_MK_MAX_ACC + i], corr, w * fd_v_bf16(kbase + d));
                        }
                    }
                }
                mreg[qi] = m_new;
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        workgroupBarrier();
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            let d = lane + i * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc[qi * FD_MK_MAX_ACC + i];
            }
        }
        let slot = ((h * mr + qi) * fd_params.splits + split) * (hd + 2u);
        fd_stage1_epilogue(lid, lane, warp, hd, slot, mreg[qi], lreg[qi]);
    }
}

@compute @workgroup_size(256)
fn flash_splitk_stage1_fp8kv_mk(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let mr = fd_params.m_rows;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var t = lid; t < mr * hd; t = t + FD_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        fd_qsh_mk[t] = fd_q[(qi * fd_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    var acc: array<f32, 64>;
    var mreg: array<f32, 8>;
    var lreg: array<f32, 8>;
    for (var qi = 0u; qi < 8u; qi = qi + 1u) {
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[qi * FD_MK_MAX_ACC + i] = 0.0;
        }
        mreg[qi] = fd_neg_inf();
        lreg[qi] = 0.0;
    }

    let total = i32(fd_params.total);
    let start0 = fd_mk_start_of(total - i32(mr - 1u));
    let base = start0 + i32(split * FD_WARPS);
    let stride = i32(fd_params.splits * FD_WARPS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }
    let use_vec4 = (hd & 3u) == 0u;

    for (var r = 0; r < rounds; r = r + 1) {
        let p = base + i32(warp) + r * stride;
        let live = p < total;
        var sp = 0u;
        if (live) {
            sp = u32(p);
            if (fd_params.ring > 0u) {
                sp = sp % fd_params.ring;
            }
        }
        let kbase = (sp * nkv + kvh) * hd;
        var ks = 0.0;
        var vs = 0.0;
        if (live) {
            ks = fd_k_scales[sp * nkv + kvh];
            vs = fd_v_scales[sp * nkv + kvh];
        }
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let tq = total - i32(mr - 1u - qi);
            let sq = fd_mk_start_of(tq);
            let act = live && p >= sq && p < tq;
            let qoff = qi * hd;
            var partial = 0.0;
            if (act) {
                if (use_vec4) {
                    let n4 = hd >> 2u;
                    for (var j = lane; j < n4; j = j + FD_LANES) {
                        let qb = qoff + j * 4u;
                        let kb = kbase + j * 4u;
                        let f0 = fd_k_fp8(kb);
                        let f1 = fd_k_fp8(kb + 1u);
                        let f2 = fd_k_fp8(kb + 2u);
                        let f3 = fd_k_fp8(kb + 3u);
                        var t = fd_qsh_mk[qb + 1u] * f1;
                        t = fma(fd_qsh_mk[qb], f0, t);
                        t = fma(fd_qsh_mk[qb + 2u], f2, t);
                        t = fma(fd_qsh_mk[qb + 3u], f3, t);
                        partial = partial + t;
                    }
                } else {
                    for (var d = lane; d < hd; d = d + FD_LANES) {
                        partial = fma(fd_qsh_mk[qoff + d], fd_k_fp8(kbase + d), partial);
                    }
                }
            }
            let score = (fd_warp_sum(lid, partial) * ks) * fd_params.scaling;
            if (act) {
                let m_new = max(mreg[qi], score);
                let corr = fd_exp(mreg[qi] - m_new);
                let w = fd_exp(score - m_new);
                lreg[qi] = fma(lreg[qi], corr, w);
                let w_v = w * vs;
                for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                    let d = lane + i * FD_LANES;
                    if (d < hd) {
                        acc[qi * FD_MK_MAX_ACC + i] =
                            fma(w_v, fd_v_fp8(kbase + d), acc[qi * FD_MK_MAX_ACC + i] * corr);
                    }
                }
                mreg[qi] = m_new;
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        workgroupBarrier();
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            let d = lane + i * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc[qi * FD_MK_MAX_ACC + i];
            }
        }
        let slot = ((h * mr + qi) * fd_params.splits + split) * (hd + 2u);
        fd_stage1_epilogue(lid, lane, warp, hd, slot, mreg[qi], lreg[qi]);
    }
}

@compute @workgroup_size(256)
fn flash_splitk_stage2_mk(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let qi = wg.y;
    if (h >= fd_params.n_heads || qi >= fd_params.m_rows) {
        return;
    }
    let hd = fd_params.head_dim;
    let splits = fd_params.splits;
    let stride = hd + 2u;
    let base = (h * fd_params.m_rows + qi) * splits * stride;

    var m_glob = fd_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) {
        m_glob = max(m_glob, fd_scratch[base + s * stride]);
    }
    for (var s = tid.x; s < splits; s = s + FD_BLOCK) {
        fd_s2w[s] = fd_split_scale(base, stride, s, m_glob);
    }
    workgroupBarrier();
    var l_glob = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        l_glob = fma(fd_scratch[base + s * stride + 1u], fd_s2w[s], l_glob);
    }
    var inv_l = 0.0;
    if (l_glob > 0.0) {
        inv_l = fd_recip(l_glob);
    }
    for (var d = tid.x; d < hd; d = d + FD_BLOCK) {
        var a = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            a = fma(fd_scratch[base + s * stride + 2u + d], fd_s2w[s], a);
        }
        fd_store_out((qi * fd_params.n_heads + h) * hd + d, a * inv_l);
    }
}

@compute @workgroup_size(256)
fn flash_splitk_stage2_mk_u(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let qi = wg.y;
    if (h >= fd_params.n_heads || qi >= fd_params.m_rows) {
        return;
    }
    let hd = fd_params.head_dim;
    let splits = fd_params.splits;
    let stride = hd + 2u;
    let base = (h * fd_params.m_rows + qi) * splits * stride;

    var m_glob = fd_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) {
        m_glob = max(m_glob, fd_scratch[base + s * stride]);
    }
    var l_glob = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {
        let sc = fd_split_scale(base, stride, s, m_glob);
        l_glob = fma(fd_scratch[base + s * stride + 1u], sc, l_glob);
    }
    var inv_l = 0.0;
    if (l_glob > 0.0) {
        inv_l = fd_recip(l_glob);
    }
    for (var d = tid.x; d < hd; d = d + FD_BLOCK) {
        var a = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {
            let sc = fd_split_scale(base, stride, s, m_glob);
            a = fma(fd_scratch[base + s * stride + 2u + d], sc, a);
        }
        fd_store_out((qi * fd_params.n_heads + h) * hd + d, a * inv_l);
    }
}

fn fd_smv2_tq(qi: u32) -> u32 {
    return fd_params.total - (fd_params.m_rows - 1u - qi);
}

fn fd_smv2_sq(tq: u32) -> u32 {
    let win = fd_params.window;
    if (win > 0u && tq > win) {
        return tq - win;
    }
    return 0u;
}

@compute @workgroup_size(256)
fn flash_smv2_stage1_bf16kv(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let mr = fd_params.m_rows;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var t = lid; t < mr * hd; t = t + FD_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        fd_qsh_mk[t] = fd_q[(qi * fd_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    var acc: array<f32, 64>;
    var mreg: array<f32, 8>;
    var lreg: array<f32, 8>;
    for (var qi = 0u; qi < 8u; qi = qi + 1u) {
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[qi * FD_MK_MAX_ACC + i] = 0.0;
        }
        mreg[qi] = fd_neg_inf();
        lreg[qi] = 0.0;
    }

    var tqr: array<u32, 8>;
    var baser: array<u32, 8>;
    let stride = fd_params.splits * FD_WARPS;
    var rounds = 0u;
    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        let tq = fd_smv2_tq(qi);
        tqr[qi] = tq;
        baser[qi] = fd_smv2_sq(tq) + split * FD_WARPS;
        if (tq > baser[qi]) {
            rounds = max(rounds, (tq - baser[qi] + stride - 1u) / stride);
        }
    }
    let use_vec8 = (hd & 7u) == 0u;
    let fused_vecv = fd_params.fused == 1u && (hd % 256u) == 0u && (hd / FD_LANES) <= FD_MAX_ACC;

    for (var r = 0u; r < rounds; r = r + 1u) {
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let p = baser[qi] + warp + r * stride;
            let live = p < tqr[qi];
            var sp = p;
            if (fd_params.ring > 0u) {
                sp = p % fd_params.ring;
            }
            let qoff = qi * hd;
            var partial = 0.0;
            if (live) {
                let kbase = (sp * nkv + kvh) * hd;
                if (use_vec8) {
                    let n8 = hd >> 3u;
                    for (var j = lane; j < n8; j = j + FD_LANES) {
                        let qb = qoff + j * 8u;
                        let kb = kbase + j * 8u;
                        for (var t = 0u; t < 4u; t = t + 1u) {
                            let kx = fd_k_bf16(kb + 2u * t);
                            let ky = fd_k_bf16(kb + 2u * t + 1u);
                            let pair = fma(kx, fd_qsh_mk[qb + 2u * t], ky * fd_qsh_mk[qb + 2u * t + 1u]);
                            partial = partial + pair;
                        }
                    }
                } else {
                    for (var d = lane; d < hd; d = d + FD_LANES) {
                        partial = fma(fd_qsh_mk[qoff + d], fd_k_bf16(kbase + d), partial);
                    }
                }
            }
            let score = fd_warp_sum(lid, partial) * fd_params.scaling;
            if (live) {
                let m_new = max(mreg[qi], score);
                let corr = fd_exp(mreg[qi] - m_new);
                let w = fd_exp(score - m_new);
                lreg[qi] = fma(lreg[qi], corr, w);
                let vbase = (sp * nkv + kvh) * hd;
                if (fused_vecv) {
                    for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                        let d = lane + i * FD_LANES;
                        if (d < hd) {
                            acc[qi * FD_MK_MAX_ACC + i] =
                                fma(w, fd_v_bf16(vbase + d), acc[qi * FD_MK_MAX_ACC + i] * corr);
                        }
                    }
                } else {
                    for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                        let d = lane + i * FD_LANES;
                        if (d < hd) {
                            acc[qi * FD_MK_MAX_ACC + i] =
                                fma(acc[qi * FD_MK_MAX_ACC + i], corr, w * fd_v_bf16(vbase + d));
                        }
                    }
                }
                mreg[qi] = m_new;
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        workgroupBarrier();
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            let d = lane + i * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc[qi * FD_MK_MAX_ACC + i];
            }
        }
        let slot = ((h * mr + qi) * fd_params.splits + split) * (hd + 2u);
        fd_stage1_epilogue(lid, lane, warp, hd, slot, mreg[qi], lreg[qi]);
    }
}

@compute @workgroup_size(256)
fn flash_smv2_stage1_f32(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let mr = fd_params.m_rows;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var t = lid; t < mr * hd; t = t + FD_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        fd_qsh_mk[t] = fd_q[(qi * fd_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    var acc: array<f32, 64>;
    var mreg: array<f32, 8>;
    var lreg: array<f32, 8>;
    for (var qi = 0u; qi < 8u; qi = qi + 1u) {
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[qi * FD_MK_MAX_ACC + i] = 0.0;
        }
        mreg[qi] = fd_neg_inf();
        lreg[qi] = 0.0;
    }

    var tqr: array<u32, 8>;
    var baser: array<u32, 8>;
    let stride = fd_params.splits * FD_WARPS;
    var rounds = 0u;
    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        let tq = fd_smv2_tq(qi);
        tqr[qi] = tq;
        baser[qi] = fd_smv2_sq(tq) + split * FD_WARPS;
        if (tq > baser[qi]) {
            rounds = max(rounds, (tq - baser[qi] + stride - 1u) / stride);
        }
    }
    let use_vec8 = (hd & 7u) == 0u;
    let fused_vecv = fd_params.fused == 1u && (hd % 256u) == 0u && (hd / FD_LANES) <= FD_MAX_ACC;

    for (var r = 0u; r < rounds; r = r + 1u) {
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let p = baser[qi] + warp + r * stride;
            let live = p < tqr[qi];
            var sp = p;
            if (fd_params.ring > 0u) {
                sp = p % fd_params.ring;
            }
            let qoff = qi * hd;
            var partial = 0.0;
            if (live) {
                let kbase = (sp * nkv + kvh) * hd;
                if (use_vec8) {
                    let n8 = hd >> 3u;
                    for (var j = lane; j < n8; j = j + FD_LANES) {
                        let qb = qoff + j * 8u;
                        let kb = kbase + j * 8u;
                        for (var t = 0u; t < 4u; t = t + 1u) {
                            let kx = fd_k_f32[kb + 2u * t];
                            let ky = fd_k_f32[kb + 2u * t + 1u];
                            let pair = fma(kx, fd_qsh_mk[qb + 2u * t], ky * fd_qsh_mk[qb + 2u * t + 1u]);
                            partial = partial + pair;
                        }
                    }
                } else {
                    for (var d = lane; d < hd; d = d + FD_LANES) {
                        partial = fma(fd_qsh_mk[qoff + d], fd_k_f32[kbase + d], partial);
                    }
                }
            }
            let score = fd_warp_sum(lid, partial) * fd_params.scaling;
            if (live) {
                let m_new = max(mreg[qi], score);
                let corr = fd_exp(mreg[qi] - m_new);
                let w = fd_exp(score - m_new);
                lreg[qi] = fma(lreg[qi], corr, w);
                let vbase = (sp * nkv + kvh) * hd;
                if (fused_vecv) {
                    for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                        let d = lane + i * FD_LANES;
                        if (d < hd) {
                            acc[qi * FD_MK_MAX_ACC + i] =
                                fma(w, fd_v_f32[vbase + d], acc[qi * FD_MK_MAX_ACC + i] * corr);
                        }
                    }
                } else {
                    for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                        let d = lane + i * FD_LANES;
                        if (d < hd) {
                            acc[qi * FD_MK_MAX_ACC + i] =
                                fma(acc[qi * FD_MK_MAX_ACC + i], corr, w * fd_v_f32[vbase + d]);
                        }
                    }
                }
                mreg[qi] = m_new;
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        workgroupBarrier();
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            let d = lane + i * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc[qi * FD_MK_MAX_ACC + i];
            }
        }
        let slot = ((h * mr + qi) * fd_params.splits + split) * (hd + 2u);
        fd_stage1_epilogue(lid, lane, warp, hd, slot, mreg[qi], lreg[qi]);
    }
}

@compute @workgroup_size(256)
fn flash_smv2_stage1_fp8kv(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let mr = fd_params.m_rows;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var t = lid; t < mr * hd; t = t + FD_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        fd_qsh_mk[t] = fd_q[(qi * fd_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    var acc: array<f32, 64>;
    var mreg: array<f32, 8>;
    var lreg: array<f32, 8>;
    for (var qi = 0u; qi < 8u; qi = qi + 1u) {
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            acc[qi * FD_MK_MAX_ACC + i] = 0.0;
        }
        mreg[qi] = fd_neg_inf();
        lreg[qi] = 0.0;
    }

    var tqr: array<u32, 8>;
    var baser: array<u32, 8>;
    let stride = fd_params.splits * FD_WARPS;
    var rounds = 0u;
    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        let tq = fd_smv2_tq(qi);
        tqr[qi] = tq;
        baser[qi] = fd_smv2_sq(tq) + split * FD_WARPS;
        if (tq > baser[qi]) {
            rounds = max(rounds, (tq - baser[qi] + stride - 1u) / stride);
        }
    }
    let use_vec4 = (hd & 3u) == 0u;

    for (var r = 0u; r < rounds; r = r + 1u) {
        for (var qi = 0u; qi < mr; qi = qi + 1u) {
            let p = baser[qi] + warp + r * stride;
            let live = p < tqr[qi];
            var sp = p;
            if (fd_params.ring > 0u) {
                sp = p % fd_params.ring;
            }
            let qoff = qi * hd;
            var partial = 0.0;
            var ks = 0.0;
            if (live) {
                let kbase = (sp * nkv + kvh) * hd;
                ks = fd_k_scales[sp * nkv + kvh];
                if (use_vec4) {
                    let n4 = hd >> 2u;
                    for (var j = lane; j < n4; j = j + FD_LANES) {
                        let qb = qoff + j * 4u;
                        let kb = kbase + j * 4u;
                        let f0 = fd_k_fp8(kb);
                        let f1 = fd_k_fp8(kb + 1u);
                        let f2 = fd_k_fp8(kb + 2u);
                        let f3 = fd_k_fp8(kb + 3u);
                        var t = fd_qsh_mk[qb + 1u] * f1;
                        t = fma(fd_qsh_mk[qb], f0, t);
                        t = fma(fd_qsh_mk[qb + 2u], f2, t);
                        t = fma(fd_qsh_mk[qb + 3u], f3, t);
                        partial = partial + t;
                    }
                } else {
                    for (var d = lane; d < hd; d = d + FD_LANES) {
                        partial = fma(fd_qsh_mk[qoff + d], fd_k_fp8(kbase + d), partial);
                    }
                }
            }
            let score = (fd_warp_sum(lid, partial) * ks) * fd_params.scaling;
            if (live) {
                let m_new = max(mreg[qi], score);
                let corr = fd_exp(mreg[qi] - m_new);
                let w = fd_exp(score - m_new);
                lreg[qi] = fma(lreg[qi], corr, w);
                let vbase = (sp * nkv + kvh) * hd;
                let w_v = w * fd_v_scales[sp * nkv + kvh];
                for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
                    let d = lane + i * FD_LANES;
                    if (d < hd) {
                        acc[qi * FD_MK_MAX_ACC + i] =
                            fma(w_v, fd_v_fp8(vbase + d), acc[qi * FD_MK_MAX_ACC + i] * corr);
                    }
                }
                mreg[qi] = m_new;
            }
        }
    }

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        workgroupBarrier();
        for (var i = 0u; i < FD_MK_MAX_ACC; i = i + 1u) {
            let d = lane + i * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc[qi * FD_MK_MAX_ACC + i];
            }
        }
        let slot = ((h * mr + qi) * fd_params.splits + split) * (hd + 2u);
        fd_stage1_epilogue(lid, lane, warp, hd, slot, mreg[qi], lreg[qi]);
    }
}

fn fd_slot_base() -> u32 {
    var slot = fd_params.total - 1u;
    if (fd_params.ring > 0u) {
        slot = slot % fd_params.ring;
    }
    return slot * fd_params.n_kv * fd_params.head_dim;
}

@compute @workgroup_size(128)
fn write_kv_from_f32(@builtin(local_invocation_id) tid: vec3<u32>) {
    let region = fd_params.n_kv * fd_params.head_dim;
    if (fd_params.total == 0u || region == 0u) {
        return;
    }
    let base = fd_slot_base();
    let w_lo = base >> 1u;
    let w_hi = (base + region - 1u) >> 1u;
    for (var w = w_lo + tid.x; w <= w_hi; w = w + 128u) {
        let e0 = w * 2u;
        let e1 = e0 + 1u;
        let old_k = fd_cache_k[w];
        let old_v = fd_cache_v[w];
        var k_lo = old_k & 0xffffu;
        var k_hi = (old_k >> 16u) & 0xffffu;
        var v_lo = old_v & 0xffffu;
        var v_hi = (old_v >> 16u) & 0xffffu;
        if (e0 >= base && e0 < base + region) {
            let s = e0 - base;
            k_lo = bf16_encode(fd_src_k[s]);
            v_lo = bf16_encode(fd_src_v[s]);
        }
        if (e1 >= base && e1 < base + region) {
            let s = e1 - base;
            k_hi = bf16_encode(fd_src_k[s]);
            v_hi = bf16_encode(fd_src_v[s]);
        }
        fd_cache_k[w] = k_lo | (k_hi << 16u);
        fd_cache_v[w] = v_lo | (v_hi << 16u);
    }
}

@compute @workgroup_size(128)
fn write_kv_from_bf16(@builtin(local_invocation_id) tid: vec3<u32>) {
    let region = fd_params.n_kv * fd_params.head_dim;
    if (fd_params.total == 0u || region == 0u) {
        return;
    }
    let base = fd_slot_base();
    let w_lo = base >> 1u;
    let w_hi = (base + region - 1u) >> 1u;
    for (var w = w_lo + tid.x; w <= w_hi; w = w + 128u) {
        let e0 = w * 2u;
        let e1 = e0 + 1u;
        let old_k = fd_cache_k[w];
        let old_v = fd_cache_v[w];
        var k_lo = old_k & 0xffffu;
        var k_hi = (old_k >> 16u) & 0xffffu;
        var v_lo = old_v & 0xffffu;
        var v_hi = (old_v >> 16u) & 0xffffu;
        if (e0 >= base && e0 < base + region) {
            let s = e0 - base;
            k_lo = u16_at(fd_k_words[s >> 1u], s);
            v_lo = u16_at(fd_v_words[s >> 1u], s);
        }
        if (e1 >= base && e1 < base + region) {
            let s = e1 - base;
            k_hi = u16_at(fd_k_words[s >> 1u], s);
            v_hi = u16_at(fd_v_words[s >> 1u], s);
        }
        fd_cache_k[w] = k_lo | (k_hi << 16u);
        fd_cache_v[w] = v_lo | (v_hi << 16u);
    }
}

@compute @workgroup_size(256)
fn flash_splitk_stage1_bf16kv_mk_u(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let mr = fd_params.m_rows;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var t = lid; t < mr * hd; t = t + FD_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        fd_qsh_mk[t] = fd_q[(qi * fd_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    let total = i32(fd_params.total);
    let start0 = fd_mk_start_of(total - i32(mr - 1u));
    let base = start0 + i32(split * FD_WARPS);
    let stride = i32(fd_params.splits * FD_WARPS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }
    let use_vec8 = (hd & 7u) == 0u;
    let hot = hd == 256u;

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        var acc0 = 0.0;
        var acc1 = 0.0;
        var acc2 = 0.0;
        var acc3 = 0.0;
        var acc4 = 0.0;
        var acc5 = 0.0;
        var acc6 = 0.0;
        var acc7 = 0.0;
        var mreg = fd_neg_inf();
        var lreg = 0.0;
        let tq = total - i32(mr - 1u - qi);
        let sq = fd_mk_start_of(tq);
        let qoff = qi * hd;

        for (var r = 0; r < rounds; r = r + 1) {
            let p = base + i32(warp) + r * stride;
            let live = p < total;
            var kbase = 0u;
            if (live) {
                kbase = (u32(p) * nkv + kvh) * hd;
            }
            let act = live && p >= sq && p < tq;
            var partial = 0.0;
            if (act) {
                if (use_vec8) {
                    let n8 = hd >> 3u;
                    for (var j = lane; j < n8; j = j + FD_LANES) {
                        let qb = qoff + j * 8u;
                        let kb = kbase + j * 8u;
                        for (var t = 0u; t < 4u; t = t + 1u) {
                            let kx = fd_k_bf16(kb + 2u * t);
                            let ky = fd_k_bf16(kb + 2u * t + 1u);
                            let pair = fma(kx, fd_qsh_mk[qb + 2u * t], ky * fd_qsh_mk[qb + 2u * t + 1u]);
                            partial = partial + pair;
                        }
                    }
                } else {
                    for (var d = lane; d < hd; d = d + FD_LANES) {
                        partial = fma(fd_qsh_mk[qoff + d], fd_k_bf16(kbase + d), partial);
                    }
                }
            }
            let score = fd_warp_sum(lid, partial);
            if (act) {
                let m_new = max(mreg, score);
                let corr = fd_exp(mreg - m_new);
                let w = fd_exp(score - m_new);
                lreg = fma(lreg, corr, w);
                if (hot) {
                    {
                        let d = lane + 0u * FD_LANES;
                        if (d < hd) {
                            acc0 = fma(w, fd_v_bf16(kbase + d), acc0 * corr);
                        }
                    }
                    {
                        let d = lane + 1u * FD_LANES;
                        if (d < hd) {
                            acc1 = fma(w, fd_v_bf16(kbase + d), acc1 * corr);
                        }
                    }
                    {
                        let d = lane + 2u * FD_LANES;
                        if (d < hd) {
                            acc2 = fma(w, fd_v_bf16(kbase + d), acc2 * corr);
                        }
                    }
                    {
                        let d = lane + 3u * FD_LANES;
                        if (d < hd) {
                            acc3 = fma(w, fd_v_bf16(kbase + d), acc3 * corr);
                        }
                    }
                    {
                        let d = lane + 4u * FD_LANES;
                        if (d < hd) {
                            acc4 = fma(w, fd_v_bf16(kbase + d), acc4 * corr);
                        }
                    }
                    {
                        let d = lane + 5u * FD_LANES;
                        if (d < hd) {
                            acc5 = fma(w, fd_v_bf16(kbase + d), acc5 * corr);
                        }
                    }
                    {
                        let d = lane + 6u * FD_LANES;
                        if (d < hd) {
                            acc6 = fma(w, fd_v_bf16(kbase + d), acc6 * corr);
                        }
                    }
                    {
                        let d = lane + 7u * FD_LANES;
                        if (d < hd) {
                            acc7 = fma(w, fd_v_bf16(kbase + d), acc7 * corr);
                        }
                    }
                } else {
                    {
                        let d = lane + 0u * FD_LANES;
                        if (d < hd) {
                            acc0 = fma(acc0, corr, w * fd_v_bf16(kbase + d));
                        }
                    }
                    {
                        let d = lane + 1u * FD_LANES;
                        if (d < hd) {
                            acc1 = fma(acc1, corr, w * fd_v_bf16(kbase + d));
                        }
                    }
                    {
                        let d = lane + 2u * FD_LANES;
                        if (d < hd) {
                            acc2 = fma(acc2, corr, w * fd_v_bf16(kbase + d));
                        }
                    }
                    {
                        let d = lane + 3u * FD_LANES;
                        if (d < hd) {
                            acc3 = fma(acc3, corr, w * fd_v_bf16(kbase + d));
                        }
                    }
                    {
                        let d = lane + 4u * FD_LANES;
                        if (d < hd) {
                            acc4 = fma(acc4, corr, w * fd_v_bf16(kbase + d));
                        }
                    }
                    {
                        let d = lane + 5u * FD_LANES;
                        if (d < hd) {
                            acc5 = fma(acc5, corr, w * fd_v_bf16(kbase + d));
                        }
                    }
                    {
                        let d = lane + 6u * FD_LANES;
                        if (d < hd) {
                            acc6 = fma(acc6, corr, w * fd_v_bf16(kbase + d));
                        }
                    }
                    {
                        let d = lane + 7u * FD_LANES;
                        if (d < hd) {
                            acc7 = fma(acc7, corr, w * fd_v_bf16(kbase + d));
                        }
                    }
                }
                mreg = m_new;
            }
        }

        workgroupBarrier();
        {
            let d = lane + 0u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc0;
            }
        }
        {
            let d = lane + 1u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc1;
            }
        }
        {
            let d = lane + 2u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc2;
            }
        }
        {
            let d = lane + 3u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc3;
            }
        }
        {
            let d = lane + 4u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc4;
            }
        }
        {
            let d = lane + 5u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc5;
            }
        }
        {
            let d = lane + 6u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc6;
            }
        }
        {
            let d = lane + 7u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc7;
            }
        }
        let slot = ((h * mr + qi) * fd_params.splits + split) * (hd + 2u);
        fd_stage1_epilogue(lid, lane, warp, hd, slot, mreg, lreg);
    }
}

@compute @workgroup_size(256)
fn flash_splitk_stage1_fp8kv_mk_u(
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
    let group = fd_params.n_heads / nkv;
    let kvh = h / group;
    let mr = fd_params.m_rows;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

    for (var t = lid; t < mr * hd; t = t + FD_BLOCK) {
        let qi = t / hd;
        let d = t - qi * hd;
        fd_qsh_mk[t] = fd_q[(qi * fd_params.n_heads + h) * hd + d];
    }
    workgroupBarrier();

    let total = i32(fd_params.total);
    let start0 = fd_mk_start_of(total - i32(mr - 1u));
    let base = start0 + i32(split * FD_WARPS);
    let stride = i32(fd_params.splits * FD_WARPS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }
    let use_vec4 = (hd & 3u) == 0u;

    for (var qi = 0u; qi < mr; qi = qi + 1u) {
        var acc0 = 0.0;
        var acc1 = 0.0;
        var acc2 = 0.0;
        var acc3 = 0.0;
        var acc4 = 0.0;
        var acc5 = 0.0;
        var acc6 = 0.0;
        var acc7 = 0.0;
        var mreg = fd_neg_inf();
        var lreg = 0.0;
        let tq = total - i32(mr - 1u - qi);
        let sq = fd_mk_start_of(tq);
        let qoff = qi * hd;

        for (var r = 0; r < rounds; r = r + 1) {
            let p = base + i32(warp) + r * stride;
            let live = p < total;
            var sp = 0u;
            if (live) {
                sp = u32(p);
                if (fd_params.ring > 0u) {
                    sp = sp % fd_params.ring;
                }
            }
            let kbase = (sp * nkv + kvh) * hd;
            var ks = 0.0;
            var vs = 0.0;
            if (live) {
                ks = fd_k_scales[sp * nkv + kvh];
                vs = fd_v_scales[sp * nkv + kvh];
            }
            let act = live && p >= sq && p < tq;
            var partial = 0.0;
            if (act) {
                if (use_vec4) {
                    let n4 = hd >> 2u;
                    for (var j = lane; j < n4; j = j + FD_LANES) {
                        let qb = qoff + j * 4u;
                        let kb = kbase + j * 4u;
                        let f0 = fd_k_fp8(kb);
                        let f1 = fd_k_fp8(kb + 1u);
                        let f2 = fd_k_fp8(kb + 2u);
                        let f3 = fd_k_fp8(kb + 3u);
                        var t = fd_qsh_mk[qb + 1u] * f1;
                        t = fma(fd_qsh_mk[qb], f0, t);
                        t = fma(fd_qsh_mk[qb + 2u], f2, t);
                        t = fma(fd_qsh_mk[qb + 3u], f3, t);
                        partial = partial + t;
                    }
                } else {
                    for (var d = lane; d < hd; d = d + FD_LANES) {
                        partial = fma(fd_qsh_mk[qoff + d], fd_k_fp8(kbase + d), partial);
                    }
                }
            }
            let score = (fd_warp_sum(lid, partial) * ks) * fd_params.scaling;
            if (act) {
                let m_new = max(mreg, score);
                let corr = fd_exp(mreg - m_new);
                let w = fd_exp(score - m_new);
                lreg = fma(lreg, corr, w);
                let w_v = w * vs;
                {
                    let d = lane + 0u * FD_LANES;
                    if (d < hd) {
                        acc0 = fma(w_v, fd_v_fp8(kbase + d), acc0 * corr);
                    }
                }
                {
                    let d = lane + 1u * FD_LANES;
                    if (d < hd) {
                        acc1 = fma(w_v, fd_v_fp8(kbase + d), acc1 * corr);
                    }
                }
                {
                    let d = lane + 2u * FD_LANES;
                    if (d < hd) {
                        acc2 = fma(w_v, fd_v_fp8(kbase + d), acc2 * corr);
                    }
                }
                {
                    let d = lane + 3u * FD_LANES;
                    if (d < hd) {
                        acc3 = fma(w_v, fd_v_fp8(kbase + d), acc3 * corr);
                    }
                }
                {
                    let d = lane + 4u * FD_LANES;
                    if (d < hd) {
                        acc4 = fma(w_v, fd_v_fp8(kbase + d), acc4 * corr);
                    }
                }
                {
                    let d = lane + 5u * FD_LANES;
                    if (d < hd) {
                        acc5 = fma(w_v, fd_v_fp8(kbase + d), acc5 * corr);
                    }
                }
                {
                    let d = lane + 6u * FD_LANES;
                    if (d < hd) {
                        acc6 = fma(w_v, fd_v_fp8(kbase + d), acc6 * corr);
                    }
                }
                {
                    let d = lane + 7u * FD_LANES;
                    if (d < hd) {
                        acc7 = fma(w_v, fd_v_fp8(kbase + d), acc7 * corr);
                    }
                }
                mreg = m_new;
            }
        }

        workgroupBarrier();
        {
            let d = lane + 0u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc0;
            }
        }
        {
            let d = lane + 1u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc1;
            }
        }
        {
            let d = lane + 2u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc2;
            }
        }
        {
            let d = lane + 3u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc3;
            }
        }
        {
            let d = lane + 4u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc4;
            }
        }
        {
            let d = lane + 5u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc5;
            }
        }
        {
            let d = lane + 6u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc6;
            }
        }
        {
            let d = lane + 7u * FD_LANES;
            if (d < hd) {
                fd_sacc[warp * FD_MAX_HD + d] = acc7;
            }
        }
        let slot = ((h * mr + qi) * fd_params.splits + split) * (hd + 2u);
        fd_stage1_epilogue(lid, lane, warp, hd, slot, mreg, lreg);
    }
}
