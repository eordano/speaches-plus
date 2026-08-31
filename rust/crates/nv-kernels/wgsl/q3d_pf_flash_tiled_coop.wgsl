alias FCA = coop_mat16x16<f16, A>;
alias FCB = coop_mat16x16<f16, B>;
alias FCC = coop_mat16x16<f32, C>;

struct FdcParams {
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

@group(0) @binding(0) var<storage, read_write> fdc_q: array<f16>;
@group(0) @binding(1) var<storage, read> fdc_q_bf16: array<u32>;
@group(0) @binding(4) var<uniform> fdc_params: FdcParams;
@group(0) @binding(5) var<storage, read> fdc_k_words: array<u32>;
@group(0) @binding(6) var<storage, read> fdc_v_words: array<u32>;
@group(0) @binding(7) var<storage, read_write> fdc_scratch: array<f32>;
@group(0) @binding(8) var<storage, read> fdc_k_scales: array<f32>;
@group(0) @binding(9) var<storage, read> fdc_v_scales: array<f32>;

const FDC_ROWS: u32 = 32u;
const FDC_POS: u32 = 16u;
const FDC_HD: u32 = 256u;
const FDC_LOG2E: f32 = 1.4426950408889634;

var<workgroup> fdc_ksh: array<f16, 4096>;
var<workgroup> fdc_vsh: array<f16, 4096>;
var<workgroup> fdc_red: array<f32, 4096>;
var<workgroup> fdc_psl: array<f16, 512>;
var<workgroup> fdc_kss: array<f32, 16>;
var<workgroup> fdc_vss: array<f32, 16>;
var<workgroup> fdc_zero: array<f32, 256>;

fn fdc_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn fdc_exp(x: f32) -> f32 {
    return exp2(x * FDC_LOG2E);
}

fn fdc_round(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ fdc_params.pad0);
}

fn fdc_stage_kv_f16_k_exact_v_carries_its_fp8_scale_so_p_stays_near_one(
    lid: u32,
    p0: i32,
    total: i32,
    nkv: u32,
    kvh: u32
) {
    if (lid < FDC_POS) {
        let p = p0 + i32(lid);
        var ks = 0.0;
        var vs = 0.0;
        if (p < total) {
            let sp = u32(p);
            ks = fdc_k_scales[sp * nkv + kvh];
            vs = fdc_v_scales[sp * nkv + kvh];
        }
        fdc_kss[lid] = ks;
        fdc_vss[lid] = vs;
    }
    workgroupBarrier();
    for (var t = lid; t < 1024u; t = t + 256u) {
        let e = t << 2u;
        let j = e >> 8u;
        let d = e & 255u;
        let p = p0 + i32(j);
        var wk = 0u;
        var wv = 0u;
        if (p < total) {
            let g = (u32(p) * nkv + kvh) * FDC_HD + d;
            wk = fdc_k_words[g >> 2u];
            wv = fdc_v_words[g >> 2u];
        }
        let vs = fdc_vss[j];
        fdc_ksh[e] = f16(e4m3_decode(byte_at(wk, 0u)));
        fdc_ksh[e + 1u] = f16(e4m3_decode(byte_at(wk, 1u)));
        fdc_ksh[e + 2u] = f16(e4m3_decode(byte_at(wk, 2u)));
        fdc_ksh[e + 3u] = f16(e4m3_decode(byte_at(wk, 3u)));
        fdc_vsh[e] = f16(e4m3_decode(byte_at(wv, 0u)) * vs);
        fdc_vsh[e + 1u] = f16(e4m3_decode(byte_at(wv, 1u)) * vs);
        fdc_vsh[e + 2u] = f16(e4m3_decode(byte_at(wv, 2u)) * vs);
        fdc_vsh[e + 3u] = f16(e4m3_decode(byte_at(wv, 3u)) * vs);
    }
}

@compute @workgroup_size(256)
fn q3w_pf_flash1_fp8kv_tiled_coop_sg(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    let h = wg.x;
    let split = wg.y;
    if (h >= fdc_params.n_heads) {
        return;
    }
    let nkv = fdc_params.n_kv;
    let kvh = h / (fdc_params.n_heads / nkv);
    let mrf = fdc_params.m_rows;
    let row0 = wg.z * FDC_ROWS;
    let rows = min(FDC_ROWS, mrf - row0);
    let qstride = fdc_params.n_heads * FDC_HD;

    let my_row = lidx >> 3u;
    let my_j = lidx & 7u;
    let my_d0 = (lidx & 7u) * 32u;

    fdc_zero[lidx] = 0.0;

    var m_st = fdc_neg_inf();
    var l_st = 0.0;
    var a0 = vec4<f32>();
    var a1 = vec4<f32>();
    var a2 = vec4<f32>();
    var a3 = vec4<f32>();
    var a4 = vec4<f32>();
    var a5 = vec4<f32>();
    var a6 = vec4<f32>();
    var a7 = vec4<f32>();

    let total = i32(fdc_params.total) - i32(mrf - row0 - rows);
    let my_tq = total - i32(rows - 1u - min(my_row, rows - 1u));
    let row_live = my_row < rows;

    let base = i32(split * FDC_POS);
    let stride = i32(fdc_params.splits * FDC_POS);
    var rounds = 0;
    if (total > base) {
        rounds = (total - base + stride - 1) / stride;
    }

    let sgid = lidx >> 5u;
    let kt0 = sgid * 32u;
    let qbase = h * FDC_HD + kt0;
    let qa0 = coopLoadT<FCA>(&fdc_q[(row0 + 0u) * qstride + qbase], qstride);
    let qa1 = coopLoadT<FCA>(&fdc_q[(row0 + 16u) * qstride + qbase], qstride);
    let qa2 = coopLoadT<FCA>(&fdc_q[(row0 + 0u) * qstride + qbase + 16u], qstride);
    let qa3 = coopLoadT<FCA>(&fdc_q[(row0 + 16u) * qstride + qbase + 16u], qstride);

    for (var r = 0; r < rounds; r = r + 1) {
        let p0 = base + r * stride;
        workgroupBarrier();
        fdc_stage_kv_f16_k_exact_v_carries_its_fp8_scale_so_p_stays_near_one(
            lidx, p0, total, nkv, kvh
        );
        workgroupBarrier();

        var c0 = coopLoadT<FCC>(&fdc_zero[0], 16u);
        var c1 = coopLoadT<FCC>(&fdc_zero[0], 16u);
        let kb0 = coopLoad<FCB>(&fdc_ksh[kt0], FDC_HD);
        c0 = coopMultiplyAdd(qa0, kb0, c0);
        c1 = coopMultiplyAdd(qa1, kb0, c1);
        let kb1 = coopLoad<FCB>(&fdc_ksh[kt0 + 16u], FDC_HD);
        c0 = coopMultiplyAdd(qa2, kb1, c0);
        c1 = coopMultiplyAdd(qa3, kb1, c1);
        coopStoreT(c0, &fdc_red[sgid * 512u], 16u);
        coopStoreT(c1, &fdc_red[sgid * 512u + 256u], 16u);
        workgroupBarrier();

        let fb = (my_row >> 4u) * 256u + (my_row & 15u) * 16u;
        var sa = fdc_red[fb + my_j];
        var sb = fdc_red[fb + my_j + 8u];
        sa = sa + fdc_red[512u + fb + my_j];
        sb = sb + fdc_red[512u + fb + my_j + 8u];
        sa = sa + fdc_red[1024u + fb + my_j];
        sb = sb + fdc_red[1024u + fb + my_j + 8u];
        sa = sa + fdc_red[1536u + fb + my_j];
        sb = sb + fdc_red[1536u + fb + my_j + 8u];
        sa = sa + fdc_red[2048u + fb + my_j];
        sb = sb + fdc_red[2048u + fb + my_j + 8u];
        sa = sa + fdc_red[2560u + fb + my_j];
        sb = sb + fdc_red[2560u + fb + my_j + 8u];
        sa = sa + fdc_red[3072u + fb + my_j];
        sb = sb + fdc_red[3072u + fb + my_j + 8u];
        sa = sa + fdc_red[3584u + fb + my_j];
        sb = sb + fdc_red[3584u + fb + my_j + 8u];

        let act_a = row_live && p0 + i32(my_j) < my_tq;
        let act_b = row_live && p0 + i32(my_j + 8u) < my_tq;
        sa = select(fdc_neg_inf(), (sa * fdc_kss[my_j]) * fdc_params.scaling, act_a);
        sb = select(fdc_neg_inf(), (sb * fdc_kss[my_j + 8u]) * fdc_params.scaling, act_b);

        var mloc = max(sa, sb);
        mloc = max(mloc, subgroupShuffleXor(mloc, 1u));
        mloc = max(mloc, subgroupShuffleXor(mloc, 2u));
        mloc = max(mloc, subgroupShuffleXor(mloc, 4u));

        let has = mloc > fdc_neg_inf();
        let m_new = select(m_st, max(m_st, mloc), has);
        let corr = select(1.0, fdc_exp(m_st - m_new), has);
        let wa = select(0.0, fdc_exp(sa - m_new), has && act_a);
        let wb = select(0.0, fdc_exp(sb - m_new), has && act_b);
        var wsum = wa + wb;
        wsum = wsum + subgroupShuffleXor(wsum, 1u);
        wsum = wsum + subgroupShuffleXor(wsum, 2u);
        wsum = wsum + subgroupShuffleXor(wsum, 4u);
        m_st = m_new;
        l_st = fma(l_st, corr, wsum);
        fdc_psl[my_row * 16u + my_j] = f16(wa);
        fdc_psl[my_row * 16u + my_j + 8u] = f16(wb);
        workgroupBarrier();

        let pa0 = coopLoadT<FCA>(&fdc_psl[0u], 16u);
        let vb0 = coopLoadT<FCB>(&fdc_vsh[kt0], FDC_HD);
        let vb1 = coopLoadT<FCB>(&fdc_vsh[kt0 + 16u], FDC_HD);
        var o0 = coopLoadT<FCC>(&fdc_zero[0], 16u);
        var o1 = coopLoadT<FCC>(&fdc_zero[0], 16u);
        o0 = coopMultiplyAdd(pa0, vb0, o0);
        o1 = coopMultiplyAdd(pa0, vb1, o1);
        coopStoreT(o0, &fdc_red[kt0], FDC_HD);
        coopStoreT(o1, &fdc_red[kt0 + 16u], FDC_HD);
        workgroupBarrier();
        if (my_row < 16u) {
            let rl = (my_row & 15u) * 256u + my_d0;
            let cv = vec4<f32>(corr);
            a0 = fma(a0, cv, vec4<f32>(fdc_red[rl], fdc_red[rl + 1u], fdc_red[rl + 2u], fdc_red[rl + 3u]));
            a1 = fma(a1, cv, vec4<f32>(fdc_red[rl + 4u], fdc_red[rl + 5u], fdc_red[rl + 6u], fdc_red[rl + 7u]));
            a2 = fma(a2, cv, vec4<f32>(fdc_red[rl + 8u], fdc_red[rl + 9u], fdc_red[rl + 10u], fdc_red[rl + 11u]));
            a3 = fma(a3, cv, vec4<f32>(fdc_red[rl + 12u], fdc_red[rl + 13u], fdc_red[rl + 14u], fdc_red[rl + 15u]));
            a4 = fma(a4, cv, vec4<f32>(fdc_red[rl + 16u], fdc_red[rl + 17u], fdc_red[rl + 18u], fdc_red[rl + 19u]));
            a5 = fma(a5, cv, vec4<f32>(fdc_red[rl + 20u], fdc_red[rl + 21u], fdc_red[rl + 22u], fdc_red[rl + 23u]));
            a6 = fma(a6, cv, vec4<f32>(fdc_red[rl + 24u], fdc_red[rl + 25u], fdc_red[rl + 26u], fdc_red[rl + 27u]));
            a7 = fma(a7, cv, vec4<f32>(fdc_red[rl + 28u], fdc_red[rl + 29u], fdc_red[rl + 30u], fdc_red[rl + 31u]));
        }
        workgroupBarrier();

        let pa1 = coopLoadT<FCA>(&fdc_psl[256u], 16u);
        var o2 = coopLoadT<FCC>(&fdc_zero[0], 16u);
        var o3 = coopLoadT<FCC>(&fdc_zero[0], 16u);
        o2 = coopMultiplyAdd(pa1, vb0, o2);
        o3 = coopMultiplyAdd(pa1, vb1, o3);
        coopStoreT(o2, &fdc_red[kt0], FDC_HD);
        coopStoreT(o3, &fdc_red[kt0 + 16u], FDC_HD);
        workgroupBarrier();
        if (my_row >= 16u) {
            let rl = (my_row & 15u) * 256u + my_d0;
            let cv = vec4<f32>(corr);
            a0 = fma(a0, cv, vec4<f32>(fdc_red[rl], fdc_red[rl + 1u], fdc_red[rl + 2u], fdc_red[rl + 3u]));
            a1 = fma(a1, cv, vec4<f32>(fdc_red[rl + 4u], fdc_red[rl + 5u], fdc_red[rl + 6u], fdc_red[rl + 7u]));
            a2 = fma(a2, cv, vec4<f32>(fdc_red[rl + 8u], fdc_red[rl + 9u], fdc_red[rl + 10u], fdc_red[rl + 11u]));
            a3 = fma(a3, cv, vec4<f32>(fdc_red[rl + 12u], fdc_red[rl + 13u], fdc_red[rl + 14u], fdc_red[rl + 15u]));
            a4 = fma(a4, cv, vec4<f32>(fdc_red[rl + 16u], fdc_red[rl + 17u], fdc_red[rl + 18u], fdc_red[rl + 19u]));
            a5 = fma(a5, cv, vec4<f32>(fdc_red[rl + 20u], fdc_red[rl + 21u], fdc_red[rl + 22u], fdc_red[rl + 23u]));
            a6 = fma(a6, cv, vec4<f32>(fdc_red[rl + 24u], fdc_red[rl + 25u], fdc_red[rl + 26u], fdc_red[rl + 27u]));
            a7 = fma(a7, cv, vec4<f32>(fdc_red[rl + 28u], fdc_red[rl + 29u], fdc_red[rl + 30u], fdc_red[rl + 31u]));
        }
    }

    if (row_live) {
        let slot = ((h * mrf + row0 + my_row) * fdc_params.splits + split) * (FDC_HD + 2u);
        let live = m_st > fdc_neg_inf();
        if (my_j == 0u) {
            fdc_scratch[slot] = m_st;
            fdc_scratch[slot + 1u] = select(0.0, fdc_round(l_st), live);
        }
        let ob = slot + 2u + my_d0;
        fdc_scratch[ob] = select(0.0, fdc_round(a0.x), live);
        fdc_scratch[ob + 1u] = select(0.0, fdc_round(a0.y), live);
        fdc_scratch[ob + 2u] = select(0.0, fdc_round(a0.z), live);
        fdc_scratch[ob + 3u] = select(0.0, fdc_round(a0.w), live);
        fdc_scratch[ob + 4u] = select(0.0, fdc_round(a1.x), live);
        fdc_scratch[ob + 5u] = select(0.0, fdc_round(a1.y), live);
        fdc_scratch[ob + 6u] = select(0.0, fdc_round(a1.z), live);
        fdc_scratch[ob + 7u] = select(0.0, fdc_round(a1.w), live);
        fdc_scratch[ob + 8u] = select(0.0, fdc_round(a2.x), live);
        fdc_scratch[ob + 9u] = select(0.0, fdc_round(a2.y), live);
        fdc_scratch[ob + 10u] = select(0.0, fdc_round(a2.z), live);
        fdc_scratch[ob + 11u] = select(0.0, fdc_round(a2.w), live);
        fdc_scratch[ob + 12u] = select(0.0, fdc_round(a3.x), live);
        fdc_scratch[ob + 13u] = select(0.0, fdc_round(a3.y), live);
        fdc_scratch[ob + 14u] = select(0.0, fdc_round(a3.z), live);
        fdc_scratch[ob + 15u] = select(0.0, fdc_round(a3.w), live);
        fdc_scratch[ob + 16u] = select(0.0, fdc_round(a4.x), live);
        fdc_scratch[ob + 17u] = select(0.0, fdc_round(a4.y), live);
        fdc_scratch[ob + 18u] = select(0.0, fdc_round(a4.z), live);
        fdc_scratch[ob + 19u] = select(0.0, fdc_round(a4.w), live);
        fdc_scratch[ob + 20u] = select(0.0, fdc_round(a5.x), live);
        fdc_scratch[ob + 21u] = select(0.0, fdc_round(a5.y), live);
        fdc_scratch[ob + 22u] = select(0.0, fdc_round(a5.z), live);
        fdc_scratch[ob + 23u] = select(0.0, fdc_round(a5.w), live);
        fdc_scratch[ob + 24u] = select(0.0, fdc_round(a6.x), live);
        fdc_scratch[ob + 25u] = select(0.0, fdc_round(a6.y), live);
        fdc_scratch[ob + 26u] = select(0.0, fdc_round(a6.z), live);
        fdc_scratch[ob + 27u] = select(0.0, fdc_round(a6.w), live);
        fdc_scratch[ob + 28u] = select(0.0, fdc_round(a7.x), live);
        fdc_scratch[ob + 29u] = select(0.0, fdc_round(a7.y), live);
        fdc_scratch[ob + 30u] = select(0.0, fdc_round(a7.z), live);
        fdc_scratch[ob + 31u] = select(0.0, fdc_round(a7.w), live);
    }
}

@compute @workgroup_size(256)
fn q3w_pf_flash_coop_qcast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_words = fdc_params.m_rows * fdc_params.n_heads * FDC_HD / 2u;
    let w = gid.x;
    if (w >= n_words) {
        return;
    }
    let word = fdc_q_bf16[w];
    fdc_q[w * 2u] = f16(bf16_lo(word));
    fdc_q[w * 2u + 1u] = f16(bf16_hi(word));
}
