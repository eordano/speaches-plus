struct KvFp8Params {
    n_tokens: u32,
    n_kv: u32,
    head_dim: u32,
    ring: u32,
    pairs: u32,
    start: u32,
    slots: u32,
    reserved: u32,
};

@group(0) @binding(0) var<storage, read> kvq_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> kvq_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> kvq_scales: array<f32>;
@group(0) @binding(3) var<storage, read> kvq_start: array<i32>;
@group(0) @binding(4) var<uniform> kvq_params: KvFp8Params;

@group(0) @binding(5) var<storage, read> kvd_src: array<u32>;
@group(0) @binding(6) var<storage, read> kvd_scales: array<f32>;
@group(0) @binding(7) var<storage, read_write> kvd_out: array<u32>;
@group(0) @binding(8) var<uniform> kvd_params: KvFp8Params;

const KV_WG: u32 = 256u;
const KV_FP8_E4M3_MAX: f32 = 448.0;

var<workgroup> kv_scratch: array<f32, 256>;
var<workgroup> kv_amax: f32;

fn kv_div_rn(a: f32, b: f32) -> f32 {
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

fn kv_encode_e4m3(x: f32) -> u32 {
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

fn kv_reduce_max(lid: u32, local: f32) -> f32 {
    kv_scratch[lid] = local;
    workgroupBarrier();
    for (var stride = KV_WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let other = kv_scratch[lid + stride];
            if (other > kv_scratch[lid]) {
                kv_scratch[lid] = other;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        kv_amax = kv_scratch[0];
    }
    workgroupBarrier();
    return kv_amax;
}

fn kv_bf16_at(base: u32, d: u32) -> f32 {
    let idx = base + d;
    let word = kvq_x[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

@compute @workgroup_size(256)
fn quantize_kv_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let flat = wg.x + wg.y * nwg.x;
    if (flat >= kvq_params.pairs) {
        return;
    }
    let n_kv = kvq_params.n_kv;
    let head_dim = kvq_params.head_dim;
    let token = flat / n_kv;
    let kv_head = flat % n_kv;

    var slot = u32(max(kvq_start[0], 0)) + token;
    if (kvq_params.ring > 0u) {
        slot = slot % kvq_params.ring;
    }
    let base_src = (token * n_kv + kv_head) * head_dim;
    let base_dst = (slot * n_kv + kv_head) * head_dim;
    let lid = tid.x;

    var local = 0.0;
    for (var d = lid; d < head_dim; d = d + KV_WG) {
        let a = abs(kv_bf16_at(base_src, d));
        if (a > local) {
            local = a;
        }
    }
    let amax = kv_reduce_max(lid, local);
    let positive = amax > 0.0;
    let scale = select(1.0, kv_div_rn(amax, KV_FP8_E4M3_MAX), positive);
    let inv_scale = select(1.0, kv_div_rn(KV_FP8_E4M3_MAX, amax), positive);

    if (lid == 0u) {
        kvq_scales[slot * n_kv + kv_head] = scale;
    }

    let out_words = head_dim >> 2u;
    for (var w = lid; w < out_words; w = w + KV_WG) {
        let d0 = w * 4u;
        var packed = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let v = kv_bf16_at(base_src, d0 + j);
            packed = packed | (kv_encode_e4m3(v * inv_scale) << (8u * j));
        }
        kvq_out[(base_dst >> 2u) + w] = packed;
    }
}

@compute @workgroup_size(256)
fn quantize_kv_fp8_kt(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let flat = wg.x + wg.y * nwg.x;
    if (flat >= kvq_params.pairs) {
        return;
    }
    let n_kv = kvq_params.n_kv;
    let head_dim = kvq_params.head_dim;
    let token = flat / n_kv;
    let kv_head = flat % n_kv;

    var slot = u32(max(kvq_start[0], 0)) + token;
    if (kvq_params.ring > 0u) {
        slot = slot % kvq_params.ring;
    }
    let base_src = (token * n_kv + kv_head) * head_dim;
    let lid = tid.x;

    var local = 0.0;
    for (var d = lid; d < head_dim; d = d + KV_WG) {
        let a = abs(kv_bf16_at(base_src, d));
        if (a > local) {
            local = a;
        }
    }
    let amax = kv_reduce_max(lid, local);
    let positive = amax > 0.0;
    let scale = select(1.0, kv_div_rn(amax, KV_FP8_E4M3_MAX), positive);
    let inv_scale = select(1.0, kv_div_rn(KV_FP8_E4M3_MAX, amax), positive);

    if (lid == 0u) {
        kvq_scales[slot * n_kv + kv_head] = scale;
    }

    let out_words = head_dim >> 2u;
    let plane0 = kv_head * out_words;
    for (var w = lid; w < out_words; w = w + KV_WG) {
        let d0 = w * 4u;
        var packed = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let v = kv_bf16_at(base_src, d0 + j);
            packed = packed | (kv_encode_e4m3(v * inv_scale) << (8u * j));
        }
        kvq_out[(plane0 + w) * kvq_params.slots + slot] = packed;
    }
}

@compute @workgroup_size(256)
fn dequantize_kv_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let flat = wg.x + wg.y * nwg.x;
    if (flat >= kvd_params.pairs) {
        return;
    }
    let n_kv = kvd_params.n_kv;
    let head_dim = kvd_params.head_dim;
    let token = flat / n_kv;
    let kv_head = flat % n_kv;

    var slot = kvd_params.start + token;
    if (kvd_params.ring > 0u) {
        slot = slot % kvd_params.ring;
    }
    let base = (slot * n_kv + kv_head) * head_dim;
    let obase = (token * n_kv + kv_head) * head_dim;
    let scale = kvd_scales[slot * n_kv + kv_head];

    let words = head_dim >> 1u;
    for (var w = tid.x; w < words; w = w + KV_WG) {
        let i0 = base + w * 2u;
        let i1 = i0 + 1u;
        let lo = e4m3_decode(byte_at(kvd_src[i0 >> 2u], i0)) * scale;
        let hi = e4m3_decode(byte_at(kvd_src[i1 >> 2u], i1)) * scale;
        kvd_out[(obase >> 1u) + w] = bf16_pack(lo, hi);
    }
}

@group(0) @binding(9) var<storage, read> kvw_v: array<u32>;
@group(0) @binding(10) var<storage, read_write> kvw_vout: array<u32>;
@group(0) @binding(11) var<storage, read_write> kvw_vscales: array<f32>;
@group(0) @binding(12) var<storage, read_write> kvw_kc: array<u32>;
@group(0) @binding(13) var<storage, read_write> kvw_vc: array<u32>;

fn kvw_bf16_at(use_v: u32, base: u32, d: u32) -> f32 {
    let idx = base + d;
    let word = select(kvq_x[idx >> 1u], kvw_v[idx >> 1u], use_v == 1u);
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

@compute @workgroup_size(256)
fn quantize_kv_fp8_kv_write_bf16(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let flat = wg.x;
    if (flat >= kvq_params.pairs) {
        return;
    }
    let use_v = wg.y;
    let n_kv = kvq_params.n_kv;
    let head_dim = kvq_params.head_dim;
    let token = flat / n_kv;
    let kv_head = flat % n_kv;

    var slot = u32(max(kvq_start[0], 0)) + token;
    if (kvq_params.ring > 0u) {
        slot = slot % kvq_params.ring;
    }
    let base_src = (token * n_kv + kv_head) * head_dim;
    let base_dst = (slot * n_kv + kv_head) * head_dim;
    let lid = tid.x;

    var local = 0.0;
    for (var d = lid; d < head_dim; d = d + KV_WG) {
        let a = abs(kvw_bf16_at(use_v, base_src, d));
        if (a > local) {
            local = a;
        }
    }
    let amax = kv_reduce_max(lid, local);
    let positive = amax > 0.0;
    let scale = select(1.0, kv_div_rn(amax, KV_FP8_E4M3_MAX), positive);
    let inv_scale = select(1.0, kv_div_rn(KV_FP8_E4M3_MAX, amax), positive);

    if (lid == 0u) {
        if (use_v == 1u) {
            kvw_vscales[slot * n_kv + kv_head] = scale;
        } else {
            kvq_scales[slot * n_kv + kv_head] = scale;
        }
    }

    let out_words = head_dim >> 2u;
    for (var w = lid; w < out_words; w = w + KV_WG) {
        let d0 = w * 4u;
        var packed = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let v = kvw_bf16_at(use_v, base_src, d0 + j);
            packed = packed | (kv_encode_e4m3(v * inv_scale) << (8u * j));
        }
        if (use_v == 1u) {
            kvw_vout[(base_dst >> 2u) + w] = packed;
        } else {
            kvq_out[(base_dst >> 2u) + w] = packed;
        }
    }

    let row_words = head_dim >> 1u;
    for (var w = lid; w < row_words; w = w + KV_WG) {
        let word = select(kvq_x[(base_src >> 1u) + w], kvw_v[(base_src >> 1u) + w], use_v == 1u);
        if (use_v == 1u) {
            kvw_vc[(base_dst >> 1u) + w] = word;
        } else {
            kvw_kc[(base_dst >> 1u) + w] = word;
        }
    }
}
