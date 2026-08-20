struct Kv4Params {
    n_kv: u32,
    head_dim: u32,
    tokens: u32,
    slots: u32,
};

@group(0) @binding(0) var<storage, read> kv4_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> kv4_out: array<u32>;
@group(0) @binding(2) var<storage, read_write> kv4_scales: array<f32>;
@group(0) @binding(3) var<storage, read> kv4_start: array<i32>;
@group(0) @binding(4) var<uniform> kv4_p: Kv4Params;

const KV4_WG: u32 = 256u;
const KV4_K_BLOCK_TOKENS: u32 = 32u;
const KV4_MAX_HEAD_DIM: u32 = 512u;
const KV4_DIV_BIG: f32 = 1.0e37;
const KV4_DIV_DOWN: f32 = 0.00390625;

fn kv4_div_rn(a: f32, b: f32) -> f32 {
    var bb = b;
    var post = 1.0;
    if (abs(b) > KV4_DIV_BIG) {
        bb = b * KV4_DIV_DOWN;
        post = KV4_DIV_DOWN;
    }
    let r = 1.0 / bb;
    let q = a * r;
    return fma(fma(-q, bb, a), r, q) * post;
}

fn kv4_bf16_cache_at(idx: u32) -> f32 {
    let w = kv4_src[idx >> 1u];
    return select(bf16_lo(w), bf16_hi(w), (idx & 1u) == 1u);
}

var<workgroup> kv4_red: array<f32, 256>;
var<workgroup> kv4_amax: f32;

fn kv4_reduce_max(lid: u32, local: f32) -> f32 {
    kv4_red[lid] = local;
    workgroupBarrier();
    for (var stride = KV4_WG / 2u; stride > 0u; stride = stride >> 1u) {
        if (lid < stride) {
            let other = kv4_red[lid + stride];
            if (other > kv4_red[lid]) {
                kv4_red[lid] = other;
            }
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        kv4_amax = kv4_red[0];
    }
    workgroupBarrier();
    return kv4_amax;
}

@compute @workgroup_size(256)
fn quantize_kv_nvfp4_v_rows(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let kvh = wg.x;
    let token = wg.y;
    if (kvh >= kv4_p.n_kv || token >= kv4_p.tokens) {
        return;
    }
    let hd = kv4_p.head_dim;
    let slot = u32(max(kv4_start[0], 0)) + token;
    if (slot >= kv4_p.slots) {
        return;
    }
    let base = (slot * kv4_p.n_kv + kvh) * hd;
    let lid = tid.x;
    var local = 0.0;
    for (var d = lid; d < hd; d = d + KV4_WG) {
        local = max(local, abs(kv4_bf16_cache_at(base + d)));
    }
    let amax = kv4_reduce_max(lid, local);
    let positive = amax > 0.0;
    let scale = select(1.0, kv4_div_rn(amax, E2M1_MAX), positive);
    let inv = select(1.0, kv4_div_rn(E2M1_MAX, amax), positive);
    if (lid == 0u) {
        kv4_scales[slot * kv4_p.n_kv + kvh] = scale;
    }
    let words = hd >> 3u;
    for (var w = lid; w < words; w = w + KV4_WG) {
        var packed = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let v = kv4_bf16_cache_at(base + w * 8u + j);
            packed = packed | (nvfp4_encode_e2m1(v * inv) << (4u * j));
        }
        kv4_out[(base >> 3u) + w] = packed;
    }
}

var<workgroup> kv4_ch_inv: array<f32, 512>;

@compute @workgroup_size(256)
fn quantize_kv_nvfp4_k_channel_blocks(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
) {
    let kvh = wg.x;
    if (kvh >= kv4_p.n_kv) {
        return;
    }
    let start = u32(max(kv4_start[0], 0));
    let end = min(start + kv4_p.tokens, kv4_p.slots);
    let block = start / KV4_K_BLOCK_TOKENS + wg.y;
    let b0 = block * KV4_K_BLOCK_TOKENS;
    if (b0 >= end) {
        return;
    }
    let b1 = min(b0 + KV4_K_BLOCK_TOKENS, end);
    let hd = kv4_p.head_dim;
    let n_kv = kv4_p.n_kv;
    let lid = tid.x;
    for (var d = lid; d < hd; d = d + KV4_WG) {
        var amax = 0.0;
        for (var t = b0; t < b1; t = t + 1u) {
            amax = max(amax, abs(kv4_bf16_cache_at((t * n_kv + kvh) * hd + d)));
        }
        let positive = amax > 0.0;
        kv4_scales[(block * n_kv + kvh) * hd + d] =
            select(1.0, kv4_div_rn(amax, E2M1_MAX), positive);
        kv4_ch_inv[d] = select(1.0, kv4_div_rn(E2M1_MAX, amax), positive);
    }
    workgroupBarrier();
    let words = hd >> 3u;
    let total = (b1 - b0) * words;
    for (var i = lid; i < total; i = i + KV4_WG) {
        let t = b0 + i / words;
        let w = i % words;
        let base = (t * n_kv + kvh) * hd;
        var packed = 0u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            let d = w * 8u + j;
            let v = kv4_bf16_cache_at(base + d);
            packed = packed | (nvfp4_encode_e2m1(v * kv4_ch_inv[d]) << (4u * j));
        }
        kv4_out[(base >> 3u) + w] = packed;
    }
}
