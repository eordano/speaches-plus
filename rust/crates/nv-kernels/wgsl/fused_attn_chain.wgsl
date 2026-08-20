struct FacParams {
    n_heads: u32,
    head_dim: u32,
    half_dim: u32,
    eps: f32,
    rows: u32,
    ring: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> fac_src: array<u32>;
@group(0) @binding(1) var<storage, read> fac_w: array<u32>;
@group(0) @binding(2) var<storage, read> fac_cos: array<f32>;
@group(0) @binding(3) var<storage, read> fac_sin: array<f32>;
@group(0) @binding(4) var<storage, read> fac_pos: array<i32>;
@group(0) @binding(5) var<storage, read> fac_start: array<i32>;
@group(0) @binding(6) var<storage, read_write> fac_qf32: array<f32>;
@group(0) @binding(7) var<storage, read_write> fac_fp8: array<u32>;
@group(0) @binding(8) var<storage, read_write> fac_scales: array<f32>;
@group(0) @binding(9) var<uniform> fac_params: FacParams;

const FAC_BLOCK: u32 = 256u;
const FAC_WARP: u32 = 32u;
const FAC_FP8_E4M3_MAX: f32 = 448.0;

var<workgroup> fac_scratch: array<f32, 256>;
var<workgroup> fac_shared: f32;
var<workgroup> fac_stage: array<u32, 256>;
var<workgroup> fac_stage2: array<u32, 256>;

fn fac_rms_div_rn(a: f32, b: f32) -> f32 {
    let r0 = 1.0 / b;
    let r = fma(fma(-b, r0, 1.0), r0, r0);
    let q0 = a * r;
    let e = fma(-b, q0, a);
    return fma(e, r, q0);
}

fn fac_rms_reduce(lid: u32, local: f32) -> f32 {
    fac_scratch[lid] = local;
    workgroupBarrier();

    for (var stride = FAC_WARP / 2u; stride > 0u; stride = stride >> 1u) {
        if ((lid & (FAC_WARP - 1u)) < stride) {
            fac_scratch[lid] = fac_scratch[lid] + fac_scratch[lid + stride];
        }
        workgroupBarrier();
    }

    if (lid == 0u) {
        let a = (fac_scratch[0u] + fac_scratch[128u]) + (fac_scratch[64u] + fac_scratch[192u]);
        let b = (fac_scratch[32u] + fac_scratch[160u]) + (fac_scratch[96u] + fac_scratch[224u]);
        let sum = a + b;
        let mean = fac_rms_div_rn(sum, f32(fac_params.head_dim));
        fac_shared = inverseSqrt(fac_params.eps + mean);
    }
    workgroupBarrier();
    return fac_shared;
}

fn fac_kv_div_rn(a: f32, b: f32) -> f32 {
    let ua = bitcast<u32>(a);
    let ub = bitcast<u32>(b);
    let sign = (ua ^ ub) & 0x80000000u;
    let amag = ua & 0x7fffffffu;
    let bmag = ub & 0x7fffffffu;
    if (amag > 0x7f800000u || bmag > 0x7f800000u
        || (amag == 0u && bmag == 0u)
        || (amag == 0x7f800000u && bmag == 0x7f800000u)) {
        return bitcast<f32>(0x7fc00000u);
    }
    if (amag == 0u || bmag == 0x7f800000u) {
        return bitcast<f32>(sign);
    }
    if (amag == 0x7f800000u || bmag == 0u) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    var ea = i32(amag >> 23u);
    var ma = amag & 0x7fffffu;
    if (ea == 0) {
        let sh = countLeadingZeros(ma) - 8u;
        ma = ma << sh;
        ea = 1 - i32(sh);
    } else {
        ma = ma | 0x800000u;
    }
    var eb = i32(bmag >> 23u);
    var mb = bmag & 0x7fffffu;
    if (eb == 0) {
        let sh = countLeadingZeros(mb) - 8u;
        mb = mb << sh;
        eb = 1 - i32(sh);
    } else {
        mb = mb | 0x800000u;
    }
    var q = 0u;
    var rem = ma;
    for (var i = 0u; i < 26u; i = i + 1u) {
        q = q << 1u;
        if (rem >= mb) {
            rem = rem - mb;
            q = q | 1u;
        }
        rem = rem << 1u;
    }
    var mant: u32;
    var round: u32;
    var sticky = u32(rem != 0u);
    var be: i32;
    if (q >= 0x2000000u) {
        mant = q >> 2u;
        round = (q >> 1u) & 1u;
        sticky = sticky | (q & 1u);
        be = ea - eb + 127;
    } else {
        mant = q >> 1u;
        round = q & 1u;
        be = ea - eb + 126;
    }
    if (be >= 255) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    if (be <= 0) {
        let s = u32(1 - be);
        if (s > 24u) {
            return bitcast<f32>(sign);
        }
        sticky = sticky | round | u32((mant & ((1u << (s - 1u)) - 1u)) != 0u);
        round = (mant >> (s - 1u)) & 1u;
        mant = mant >> s;
        be = 0;
    }
    if (round == 1u && (sticky != 0u || (mant & 1u) == 1u)) {
        mant = mant + 1u;
    }
    var bits: u32;
    if (be == 0) {
        bits = sign | mant;
    } else {
        bits = sign | ((u32(be) << 23u) + (mant - 0x800000u));
    }
    return bitcast<f32>(bits);
}

fn fac_encode_e4m3(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let sign = (b >> 31u) << 7u;
    let mag = b & 0x7fffffffu;
    if (mag > 0x7f800000u) {
        return sign | 0x7fu;
    }
    let e = i32(mag >> 23u) - 127;
    if (e >= -6) {
        let lsb = (mag >> 20u) & 1u;
        let r = mag + 0x7ffffu + lsb;
        let e2 = i32(r >> 23u) - 127;
        let m2 = (r >> 20u) & 7u;
        if (e2 > 8 || (e2 == 8 && m2 == 7u)) {
            return sign | 0x7eu;
        }
        return sign | (u32(e2 + 7) << 3u) | m2;
    }
    let s = u32(14 - e);
    if (s >= 32u) {
        return sign;
    }
    let full = 0x800000u | (mag & 0x7fffffu);
    let q = full >> s;
    let round_bit = (full >> (s - 1u)) & 1u;
    let rest = full & ((1u << (s - 1u)) - 1u);
    var n = q;
    if (round_bit == 1u && (rest != 0u || (q & 1u) == 1u)) {
        n = n + 1u;
    }
    return sign | n;
}

fn fac_kv_reduce_max(lid: u32, local: f32) -> f32 {
    fac_scratch[lid] = local;
    workgroupBarrier();
    for (var stride = FAC_BLOCK / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let other = fac_scratch[lid + stride];
            if (other > fac_scratch[lid]) {
                fac_scratch[lid] = other;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        fac_shared = fac_scratch[0];
    }
    workgroupBarrier();
    return fac_shared;
}

fn fac_row_index(wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wg.x + wg.y * nwg.x;
}

fn fac_norm_stage(lid: u32, base: u32) {
    let hd = fac_params.head_dim;
    let words = hd >> 1u;

    var local = 0.0;
    for (var i = lid; i < hd; i = i + FAC_BLOCK) {
        let word = fac_src[base + (i >> 1u)];
        let v = select(bf16_lo(word), bf16_hi(word), (i & 1u) == 1u);
        local = fma(v, v, local);
    }
    let rms = fac_rms_reduce(lid, local);

    for (var i = lid; i < words; i = i + FAC_BLOCK) {
        let xw = fac_src[base + i];
        let ww = fac_w[i];
        let lo = bf16_lo(xw) * rms * bf16_lo(ww);
        let hi = bf16_hi(xw) * rms * bf16_hi(ww);
        fac_stage[i] = bf16_pack(lo, hi);
    }
    workgroupBarrier();
}

fn fac_stage_load(elem: u32) -> f32 {
    let word = fac_stage[elem >> 1u];
    if ((elem & 1u) == 0u) {
        return bf16_lo(word);
    }
    return bf16_hi(word);
}

fn fac_rotate(row_base: u32, elem: u32) -> f32 {
    let half = fac_params.half_dim;
    if (elem < half) {
        let c = fac_cos[row_base + elem];
        let s = fac_sin[row_base + elem];
        let a = fac_stage_load(elem);
        let b = fac_stage_load(elem + half);
        return fma(a, c, -(b * s));
    }
    let pair = elem - half;
    let c = fac_cos[row_base + pair];
    let s = fac_sin[row_base + pair];
    let a = fac_stage_load(pair);
    let b = fac_stage_load(elem);
    return fma(a, s, b * c);
}

fn fac_stage2_at(d: u32) -> f32 {
    let word = fac_stage2[d >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (d & 1u) == 1u);
}

fn fac_quant_stage2(lid: u32, row: u32) {
    let n_kv = fac_params.n_heads;
    let hd = fac_params.head_dim;
    let token = row / n_kv;
    let kv_head = row % n_kv;

    var slot = u32(max(fac_start[0], 0)) + token;
    if (fac_params.ring > 0u) {
        slot = slot % fac_params.ring;
    }
    let base_dst = (slot * n_kv + kv_head) * hd;

    var local = 0.0;
    for (var d = lid; d < hd; d = d + FAC_BLOCK) {
        let a = abs(fac_stage2_at(d));
        if (a > local) {
            local = a;
        }
    }
    let amax = fac_kv_reduce_max(lid, local);
    let positive = amax > 0.0;
    let scale = select(1.0, fac_kv_div_rn(amax, FAC_FP8_E4M3_MAX), positive);
    let inv_scale = select(1.0, fac_kv_div_rn(FAC_FP8_E4M3_MAX, amax), positive);

    if (lid == 0u) {
        fac_scales[slot * n_kv + kv_head] = scale;
    }

    let out_words = hd >> 2u;
    for (var w = lid; w < out_words; w = w + FAC_BLOCK) {
        let d0 = w * 4u;
        var packed = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let v = fac_stage2_at(d0 + j);
            packed = packed | (fac_encode_e4m3(v * inv_scale) << (8u * j));
        }
        fac_fp8[(base_dst >> 2u) + w] = packed;
    }
}

@compute @workgroup_size(256)
fn e4b_attn_q_rms_rope_f32(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = fac_row_index(wg, nwg);
    if (row >= fac_params.rows) {
        return;
    }
    let lid = tid.x;
    let half = fac_params.half_dim;
    let base = row * half;
    fac_norm_stage(lid, base);

    let token = row / fac_params.n_heads;
    let pos = u32(fac_pos[token]);
    let row_base = pos * half;
    for (var w = lid; w < half; w = w + FAC_BLOCK) {
        let elem = w * 2u;
        let lo = fac_rotate(row_base, elem);
        let hi = fac_rotate(row_base, elem + 1u);
        let word = bf16_pack(lo, hi);
        let idx = base + w;
        fac_qf32[idx * 2u] = bf16_lo(word);
        fac_qf32[idx * 2u + 1u] = bf16_hi(word);
    }
}

@compute @workgroup_size(256)
fn e4b_attn_k_rms_rope_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = fac_row_index(wg, nwg);
    if (row >= fac_params.rows) {
        return;
    }
    let lid = tid.x;
    let half = fac_params.half_dim;
    let base = row * half;
    fac_norm_stage(lid, base);

    let token = row / fac_params.n_heads;
    let pos = u32(fac_pos[token]);
    let row_base = pos * half;
    for (var w = lid; w < half; w = w + FAC_BLOCK) {
        let elem = w * 2u;
        let lo = fac_rotate(row_base, elem);
        let hi = fac_rotate(row_base, elem + 1u);
        fac_stage2[w] = bf16_pack(lo, hi);
    }
    workgroupBarrier();

    fac_quant_stage2(lid, row);
}

@compute @workgroup_size(256)
fn e4b_attn_v_rms_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = fac_row_index(wg, nwg);
    if (row >= fac_params.rows) {
        return;
    }
    let lid = tid.x;
    let half = fac_params.half_dim;
    let base = row * half;
    fac_norm_stage(lid, base);

    for (var w = lid; w < half; w = w + FAC_BLOCK) {
        fac_stage2[w] = fac_stage[w];
    }
    workgroupBarrier();

    fac_quant_stage2(lid, row);
}
