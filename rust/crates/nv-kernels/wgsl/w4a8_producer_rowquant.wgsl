struct W48pParams {
    k_blocks: u32,
    words_per_row: u32,
    hidden: u32,
    eps: f32,
};

@group(0) @binding(0) var<storage, read> w48p_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> w48p_res: array<u32>;
@group(0) @binding(2) var<storage, read> w48p_w: array<u32>;
@group(0) @binding(3) var<storage, read_write> w48p_out: array<u32>;
@group(0) @binding(4) var<uniform> w48p_p: W48pParams;
@group(0) @binding(5) var<storage, read_write> w48p_xq: array<u32>;
@group(0) @binding(6) var<storage, read_write> w48p_xs: array<f32>;
@group(0) @binding(7) var<storage, read> w48p_gate: array<u32>;
@group(0) @binding(8) var<storage, read> w48p_up: array<u32>;

const W48P_WG: u32 = 256u;

var<workgroup> w48p_red: array<f32, W48P_WG>;
var<workgroup> w48p_rms: f32;

fn w48p_word_amax(w: u32) -> f32 {
    return max(abs(bf16_lo(w)), abs(bf16_hi(w)));
}

fn w48p_scale_from_amax(tid: u32, m: f32) -> f32 {
    w48p_red[tid] = m;
    workgroupBarrier();
    var s = W48P_WG >> 1u;
    while (s > 0u) {
        if (tid < s) {
            w48p_red[tid] = max(w48p_red[tid], w48p_red[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    return w48p_red[0] / 127.0;
}

fn w48p_quant_word(w: u32, dinv: f32) -> vec2<u32> {
    let qe = i32(round(clamp(bf16_lo(w) * dinv, -127.0, 127.0)));
    let qo = i32(round(clamp(bf16_hi(w) * dinv, -127.0, 127.0)));
    return vec2<u32>(u32(qe) & 0xffu, u32(qo) & 0xffu);
}

fn w48p_emit_group(g: u32, w0: u32, w1: u32, w2: u32, w3: u32, dinv: f32) {
    let q0 = w48p_quant_word(w0, dinv);
    let q1 = w48p_quant_word(w1, dinv);
    let q2 = w48p_quant_word(w2, dinv);
    let q3 = w48p_quant_word(w3, dinv);
    w48p_xq[2u * g] = q0.x | (q1.x << 8u) | (q2.x << 16u) | (q3.x << 24u);
    w48p_xq[2u * g + 1u] = q0.y | (q1.y << 8u) | (q2.y << 16u) | (q3.y << 24u);
}

@compute @workgroup_size(W48P_WG)
fn rmsnorm_residual_bf16_rowquant_evenodd(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let words = w48p_p.words_per_row;
    var local = 0.0;
    for (var i = tid; i < words; i = i + W48P_WG) {
        let xw = w48p_x[i];
        let rw = w48p_res[i];
        let lo = bf16_lo(xw) + bf16_lo(rw);
        let hi = bf16_hi(xw) + bf16_hi(rw);
        w48p_res[i] = bf16_pack(lo, hi);
        local = local + lo * lo + hi * hi;
    }
    w48p_red[tid] = local;
    workgroupBarrier();
    for (var stride = W48P_WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            w48p_red[tid] = w48p_red[tid] + w48p_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        w48p_rms = inverseSqrt(w48p_red[0] / f32(w48p_p.hidden) + w48p_p.eps);
    }
    workgroupBarrier();
    let rms = w48p_rms;
    var m = 0.0;
    for (var i = tid; i < words; i = i + W48P_WG) {
        let sw = w48p_res[i];
        let ww = w48p_w[i];
        let ow = bf16_pack(bf16_lo(sw) * rms * bf16_lo(ww), bf16_hi(sw) * rms * bf16_hi(ww));
        w48p_out[i] = ow;
        m = max(m, w48p_word_amax(ow));
    }
    let d = w48p_scale_from_amax(tid, m);
    let dinv = select(0.0, 1.0 / d, d != 0.0);
    if (tid == 0u) {
        w48p_xs[0] = d;
    }
    storageBarrier();
    let nvec = words >> 2u;
    for (var g = tid; g < nvec; g = g + W48P_WG) {
        w48p_emit_group(
            g,
            w48p_out[4u * g],
            w48p_out[4u * g + 1u],
            w48p_out[4u * g + 2u],
            w48p_out[4u * g + 3u],
            dinv
        );
    }
}

fn w48p_silu_mul_word(gw: u32, uw: u32) -> u32 {
    let g0 = bf16_lo(gw);
    let g1 = bf16_hi(gw);
    let a0 = bf16_decode(bf16_encode(g0 / (1.0 + exp(-g0)))) * bf16_lo(uw);
    let a1 = bf16_decode(bf16_encode(g1 / (1.0 + exp(-g1)))) * bf16_hi(uw);
    return bf16_pack(a0, a1);
}

const W48P_SILU_SMEM_WORDS_40KB_CAPS_K_AT_20480_AND_BEAT_THE_REDUNDANT_SCAN_12US_TO_5US: u32 =
    10240u;

const W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT: u32 = 1024u;

var<workgroup> w48p_act: array<u32, W48P_SILU_SMEM_WORDS_40KB_CAPS_K_AT_20480_AND_BEAT_THE_REDUNDANT_SCAN_12US_TO_5US>;
var<workgroup> w48p_silu_red: array<f32, W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT>;

fn w48p_silu_scale_from_amax(tid: u32, m: f32) -> f32 {
    w48p_silu_red[tid] = m;
    workgroupBarrier();
    var s = W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT >> 1u;
    while (s > 0u) {
        if (tid < s) {
            w48p_silu_red[tid] = max(w48p_silu_red[tid], w48p_silu_red[tid + s]);
        }
        workgroupBarrier();
        s = s >> 1u;
    }
    return w48p_silu_red[0] / 127.0;
}

@compute @workgroup_size(W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT)
fn silu_mul_rowquant_evenodd(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let wg = W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT;
    let nwords = w48p_p.k_blocks * 8u;
    var m = 0.0;
    for (var i = tid; i < nwords; i = i + wg) {
        let vw = w48p_silu_mul_word(w48p_gate[i], w48p_up[i]);
        w48p_act[i] = vw;
        #def b2
        m = max(m, w48p_word_amax(vw));
    }
    let d = w48p_silu_scale_from_amax(tid, m);
    let dinv = select(0.0, 1.0 / d, d != 0.0);
    if (tid == 0u) {
        w48p_xs[0] = d;
    }
    let nvec = w48p_p.k_blocks * 2u;
    for (var g = tid; g < nvec; g = g + wg) {
        w48p_emit_group(
            g,
            w48p_act[4u * g],
            w48p_act[4u * g + 1u],
            w48p_act[4u * g + 2u],
            w48p_act[4u * g + 3u],
            dinv
        );
    }
}
        #end

@compute @workgroup_size(W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT)
fn silu_mul_rowquant_evenodd_nosmem(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let wg = W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT;
    let nwords = w48p_p.k_blocks * 8u;
    var m = 0.0;
    for (var i = tid; i < nwords; i = i + wg) {
        m = max(m, w48p_word_amax(w48p_silu_mul_word(w48p_gate[i], w48p_up[i])));
    #def b1
    }
    let d = w48p_silu_scale_from_amax(tid, m);
    let dinv = select(0.0, 1.0 / d, d != 0.0);
    if (tid == 0u) {
        w48p_xs[0] = d;
    }
    let nvec = w48p_p.k_blocks * 2u;
    for (var g = tid; g < nvec; g = g + wg) {
        w48p_emit_group(
            g,
            w48p_silu_mul_word(w48p_gate[4u * g], w48p_up[4u * g]),
            w48p_silu_mul_word(w48p_gate[4u * g + 1u], w48p_up[4u * g + 1u]),
            w48p_silu_mul_word(w48p_gate[4u * g + 2u], w48p_up[4u * g + 2u]),
            w48p_silu_mul_word(w48p_gate[4u * g + 3u], w48p_up[4u * g + 3u]),
            dinv
        );
    }
}
    #end

@compute @workgroup_size(W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT)
fn silu_mul_rowquant_evenodd_act_bf16_nosmem(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let wg = W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT;
    let nwords = w48p_p.k_blocks * 8u;
    var m = 0.0;
    for (var i = tid; i < nwords; i = i + wg) {
        let vw = w48p_silu_mul_word(w48p_gate[i], w48p_up[i]);
        w48p_out[i] = vw;
        m = max(m, w48p_word_amax(vw));
    #use b1

@compute @workgroup_size(W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT)
fn silu_mul_rowquant_evenodd_act_bf16(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let wg = W48P_SILU_WG_1024_AMAX_IS_ORDER_FREE_SO_THE_WIDE_REDUCE_STAYS_BIT_EXACT;
    let nwords = w48p_p.k_blocks * 8u;
    var m = 0.0;
    for (var i = tid; i < nwords; i = i + wg) {
        let vw = w48p_silu_mul_word(w48p_gate[i], w48p_up[i]);
        w48p_act[i] = vw;
        w48p_out[i] = vw;
        #use b2
