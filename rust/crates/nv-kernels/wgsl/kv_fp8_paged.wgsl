struct KvPagedParams {
    n_tokens: u32,
    n_kv: u32,
    head_dim: u32,
    block_size: u32,
    pairs: u32,
    start: u32,
    slots: u32,
    reserved: u32,
};

struct KvCopyParams {
    src_block: u32,
    dst_block: u32,
    block_size: u32,
    n_kv: u32,
    head_dim: u32,
    pairs: u32,
    reserved0: u32,
    reserved1: u32,
};

@group(0) @binding(0) var<storage, read> kvpq_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> kvpq_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> kvpq_scales: array<f32>;
@group(0) @binding(3) var<storage, read> kvpq_start: array<i32>;
@group(0) @binding(4) var<storage, read> kvpq_table: array<i32>;
@group(0) @binding(5) var<uniform> kvpq_params: KvPagedParams;

@group(0) @binding(6) var<storage, read> kvpd_src: array<u32>;
@group(0) @binding(7) var<storage, read> kvpd_scales: array<f32>;
@group(0) @binding(8) var<storage, read_write> kvpd_out: array<u32>;
@group(0) @binding(9) var<storage, read> kvpd_table: array<i32>;
@group(0) @binding(10) var<uniform> kvpd_params: KvPagedParams;

@group(0) @binding(11) var<storage, read_write> kvpc_fp8: array<u32>;
@group(0) @binding(12) var<storage, read_write> kvpc_scales: array<f32>;
@group(0) @binding(13) var<uniform> kvpc_params: KvCopyParams;

@group(0) @binding(14) var<storage, read> kvpx_src_fp8: array<u32>;
@group(0) @binding(15) var<storage, read> kvpx_src_scales: array<f32>;
@group(0) @binding(16) var<storage, read_write> kvpx_dst_fp8: array<u32>;
@group(0) @binding(17) var<storage, read_write> kvpx_dst_scales: array<f32>;
@group(0) @binding(18) var<uniform> kvpx_params: KvCopyParams;

const KVP_WG: u32 = 256u;
const KVP_FP8_E4M3_MAX: f32 = 448.0;

var<workgroup> kvp_scratch: array<f32, 256>;
var<workgroup> kvp_amax: f32;

fn kvp_div_core(a: f32, b: f32) -> f32 {
    let r = 1.0 / b;
    let q = a * r;
    return fma(fma(-q, b, a), r, q);
}

fn kvp_div_rn(a: f32, b: f32) -> f32 {
    let ba = bitcast<u32>(a);
    let bb = bitcast<u32>(b);
    let ea = (ba >> 23u) & 0xffu;
    let eb = (bb >> 23u) & 0xffu;
    if (ea == 0u || ea == 0xffu || eb == 0u || eb == 0xffu) {
        return kvp_div_core(a, b);
    }
    let sign = (ba ^ bb) & 0x80000000u;
    let an = bitcast<f32>(0x3f800000u | (ba & 0x7fffffu));
    let bn = bitcast<f32>(0x3f800000u | (bb & 0x7fffffu));
    let q0 = kvp_div_core(an, bn);
    let rr = fma(-q0, bn, an);
    let qb = bitcast<u32>(q0);
    let e0 = i32((qb >> 23u) & 0xffu);
    let m0 = 0x800000u | (qb & 0x7fffffu);
    let et = e0 + i32(ea) - i32(eb);
    if (et >= 255) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    if (et >= 1) {
        return bitcast<f32>(sign | (u32(et) << 23u) | (qb & 0x7fffffu));
    }
    let s = u32(1 - et);
    if (s >= 26u) {
        return bitcast<f32>(sign);
    }
    let half = 1u << (s - 1u);
    let low = m0 & ((1u << s) - 1u);
    var n = m0 >> s;
    var up = low > half;
    if (low == half) {
        up = rr > 0.0 || (rr == 0.0 && (n & 1u) == 1u);
    }
    if (up) {
        n = n + 1u;
    }
    return bitcast<f32>(sign | n);
}

struct KvpSplit {
    mant: f32,
    exp: i32,
    ok: bool,
};

fn kvp_split(x: f32) -> KvpSplit {
    var out: KvpSplit;
    out.mant = 1.0;
    out.exp = 0;
    out.ok = false;
    let b = bitcast<u32>(x);
    let e = (b >> 23u) & 0xffu;
    let m = b & 0x7fffffu;
    if (e == 0xffu) {
        return out;
    }
    if (e == 0u) {
        if (m == 0u) {
            return out;
        }
        let sh = countLeadingZeros(m) - 8u;
        out.mant = bitcast<f32>(0x3f800000u | ((m << sh) & 0x7fffffu));
        out.exp = -126 - i32(sh);
        out.ok = true;
        return out;
    }
    out.mant = bitcast<f32>(0x3f800000u | m);
    out.exp = i32(e) - 127;
    out.ok = true;
    return out;
}

fn kvp_mul_rn(a: f32, b: f32) -> f32 {
    let sa = kvp_split(a);
    let sb = kvp_split(b);
    if (!sa.ok || !sb.ok) {
        return a * b;
    }
    let sign = (bitcast<u32>(a) ^ bitcast<u32>(b)) & 0x80000000u;
    let p = sa.mant * sb.mant;
    let err = fma(sa.mant, sb.mant, -p);
    let pb = bitcast<u32>(p);
    let m0 = 0x800000u | (pb & 0x7fffffu);
    let et = i32((pb >> 23u) & 0xffu) + sa.exp + sb.exp;
    if (et >= 255) {
        return bitcast<f32>(sign | 0x7f800000u);
    }
    if (et >= 1) {
        return bitcast<f32>(sign | (u32(et) << 23u) | (pb & 0x7fffffu));
    }
    let s = u32(1 - et);
    if (s >= 26u) {
        return bitcast<f32>(sign);
    }
    let half = 1u << (s - 1u);
    let low = m0 & ((1u << s) - 1u);
    var n = m0 >> s;
    var up = low > half;
    if (low == half) {
        up = err > 0.0 || (err == 0.0 && (n & 1u) == 1u);
    }
    if (up) {
        n = n + 1u;
    }
    return bitcast<f32>(sign | n);
}

fn kvp_encode_e4m3(x: f32) -> u32 {
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

fn kvp_reduce_max(lid: u32, local: f32) -> f32 {
    kvp_scratch[lid] = local;
    workgroupBarrier();
    for (var stride = KVP_WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            kvp_scratch[lid] = max(kvp_scratch[lid], kvp_scratch[lid + stride]);
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        kvp_amax = kvp_scratch[0];
    }
    workgroupBarrier();
    return kvp_amax;
}

fn kvp_slot(table_entry: i32, block_size: u32, logical: u32) -> u32 {
    let blk = logical / block_size;
    let off = logical - blk * block_size;
    return u32(max(table_entry, 0)) * block_size + off;
}

fn kvp_bf16_at(base: u32, d: u32) -> f32 {
    let idx = base + d;
    let word = kvpq_x[idx >> 1u];
    return select(bf16_lo(word), bf16_hi(word), (idx & 1u) == 1u);
}

@compute @workgroup_size(256)
fn quantize_kv_fp8_paged(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let flat = wg.x + wg.y * nwg.x;
    if (flat >= kvpq_params.pairs) {
        return;
    }
    let n_kv = kvpq_params.n_kv;
    let head_dim = kvpq_params.head_dim;
    let block_size = kvpq_params.block_size;
    let token = flat / n_kv;
    let kv_head = flat % n_kv;

    let logical = u32(max(kvpq_start[0], 0)) + token;
    let blk = logical / block_size;
    let slot = kvp_slot(kvpq_table[blk], block_size, logical);
    let base_src = (token * n_kv + kv_head) * head_dim;
    let base_dst = (slot * n_kv + kv_head) * head_dim;
    let lid = tid.x;

    var local = 0.0;
    for (var d = lid; d < head_dim; d = d + KVP_WG) {
        local = max(local, abs(kvp_bf16_at(base_src, d)));
    }
    let amax = kvp_reduce_max(lid, local);
    let positive = amax > 0.0;
    let scale = select(1.0, kvp_div_rn(amax, KVP_FP8_E4M3_MAX), positive);
    let inv_scale = select(1.0, kvp_div_rn(KVP_FP8_E4M3_MAX, amax), positive);

    if (lid == 0u) {
        kvpq_scales[slot * n_kv + kv_head] = scale;
    }

    let out_words = head_dim >> 2u;
    for (var w = lid; w < out_words; w = w + KVP_WG) {
        let d0 = w * 4u;
        var packed = 0u;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let v = kvp_bf16_at(base_src, d0 + j);
            packed = packed | (kvp_encode_e4m3(v * inv_scale) << (8u * j));
        }
        kvpq_out[(base_dst >> 2u) + w] = packed;
    }
}

@compute @workgroup_size(256)
fn dequantize_kv_fp8_paged(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let flat = wg.x + wg.y * nwg.x;
    if (flat >= kvpd_params.pairs) {
        return;
    }
    let n_kv = kvpd_params.n_kv;
    let head_dim = kvpd_params.head_dim;
    let block_size = kvpd_params.block_size;
    let token = flat / n_kv;
    let kv_head = flat % n_kv;

    let blk = token / block_size;
    let slot = kvp_slot(kvpd_table[blk], block_size, token);
    let base = (slot * n_kv + kv_head) * head_dim;
    let obase = (token * n_kv + kv_head) * head_dim;
    let scale = kvpd_scales[slot * n_kv + kv_head];

    let words = head_dim >> 1u;
    for (var w = tid.x; w < words; w = w + KVP_WG) {
        let i0 = base + w * 2u;
        let i1 = i0 + 1u;
        let lo = kvp_mul_rn(e4m3_decode(byte_at(kvpd_src[i0 >> 2u], i0)), scale);
        let hi = kvp_mul_rn(e4m3_decode(byte_at(kvpd_src[i1 >> 2u], i1)), scale);
        kvpd_out[(obase >> 1u) + w] = bf16_pack(lo, hi);
    }
}

@compute @workgroup_size(256)
fn copy_kv_block_fp8(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let flat = wg.x + wg.y * nwg.x;
    if (flat >= kvpc_params.pairs) {
        return;
    }
    let n_kv = kvpc_params.n_kv;
    let head_dim = kvpc_params.head_dim;
    let block_size = kvpc_params.block_size;
    let slot_in_block = flat / n_kv;
    let kv_head = flat % n_kv;

    let src_slot = kvpc_params.src_block * block_size + slot_in_block;
    let dst_slot = kvpc_params.dst_block * block_size + slot_in_block;
    let src_base = (src_slot * n_kv + kv_head) * head_dim;
    let dst_base = (dst_slot * n_kv + kv_head) * head_dim;

    let words = head_dim >> 2u;
    for (var w = tid.x; w < words; w = w + KVP_WG) {
        kvpc_fp8[(dst_base >> 2u) + w] = kvpc_fp8[(src_base >> 2u) + w];
    }
    if (tid.x == 0u) {
        kvpc_scales[dst_slot * n_kv + kv_head] = kvpc_scales[src_slot * n_kv + kv_head];
    }
}

@compute @workgroup_size(256)
fn copy_kv_block_fp8_x(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let flat = wg.x + wg.y * nwg.x;
    if (flat >= kvpx_params.pairs) {
        return;
    }
    let n_kv = kvpx_params.n_kv;
    let head_dim = kvpx_params.head_dim;
    let block_size = kvpx_params.block_size;
    let slot_in_block = flat / n_kv;
    let kv_head = flat % n_kv;

    let src_slot = kvpx_params.src_block * block_size + slot_in_block;
    let dst_slot = kvpx_params.dst_block * block_size + slot_in_block;
    let src_base = (src_slot * n_kv + kv_head) * head_dim;
    let dst_base = (dst_slot * n_kv + kv_head) * head_dim;

    let words = head_dim >> 2u;
    for (var w = tid.x; w < words; w = w + KVP_WG) {
        kvpx_dst_fp8[(dst_base >> 2u) + w] = kvpx_src_fp8[(src_base >> 2u) + w];
    }
    if (tid.x == 0u) {
        kvpx_dst_scales[dst_slot * n_kv + kv_head] = kvpx_src_scales[src_slot * n_kv + kv_head];
    }
}
