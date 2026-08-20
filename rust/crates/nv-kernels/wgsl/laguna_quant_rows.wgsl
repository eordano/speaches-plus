
struct LgqParams {
    k_blocks: u32,
    n_slots: u32,
    use_sel: u32,
    x_slot_stride_elems: u32,
};

@group(0) @binding(10) var<storage, read> lgq_x: array<u32>;
@group(0) @binding(11) var<uniform> lgq_p: LgqParams;
@group(0) @binding(12) var<storage, read_write> lgq_packed: array<u32>;
@group(0) @binding(13) var<storage, read_write> lgq_scales: array<u32>;
@group(0) @binding(14) var<storage, read> lgq_sel: array<u32>;
@group(0) @binding(15) var<storage, read> lgq_glob: array<f32>;

var<workgroup> lgq_sbytes: array<u32, 256>;

fn lgw_qz_block(x_base: u32, out_base: u32, kb: u32, stored: f32) -> u32 {
    var vbits: array<u32, 16>;
    var amax_bits = 0u;
    let base = x_base + kb * NVFP4_BLOCK_SIZE;
    for (var i = 0u; i < NVFP4_BLOCK_SIZE; i = i + 1u) {
        let j = base + i;
        let w = u16_at(lgq_x[j >> 1u], j);
        vbits[i] = w;
        let mag = w & 0x7fffu;
        if (mag <= 0x7f80u && mag > amax_bits) {
            amax_bits = mag;
        }
    }
    var local_scale = 1.0;
    if (amax_bits != 0u) {
        local_scale = q_div_small(bf16_decode(amax_bits), 3u, 1);
    }
    let scale_byte = q_encode_scale(stored, local_scale);
    let parts = q_scale_parts(scale_byte);
    let inv = q_div_small(stored, u32(parts.x), parts.y);
    let u_up = q_subnormal_shift(inv);
    let inv_up = q_scale_up_pow2(inv, u_up);

    var w0 = 0u;
    var w1 = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let lo = nvfp4_encode_e2m1(q_scaled_product(vbits[2u * i], inv_up, u_up));
        let hi = nvfp4_encode_e2m1(q_scaled_product(vbits[2u * i + 1u], inv_up, u_up));
        let packed = ((hi & 15u) << 4u) | (lo & 15u);
        if (i < 4u) {
            w0 = w0 | (packed << (8u * i));
        } else {
            w1 = w1 | (packed << (8u * (i - 4u)));
        }
    }
    lgq_packed[out_base + kb * 2u] = w0;
    lgq_packed[out_base + kb * 2u + 1u] = w1;
    return scale_byte & 255u;
}

@compute @workgroup_size(256)
fn lgw_quant_rows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let slot = wid.y;
    var gi = slot;
    if (lgq_p.use_sel == 1u) {
        gi = lgq_sel[slot];
    }
    let g = lgq_glob[gi];
    let g_mag = bitcast<u32>(g) & 0x7fffffffu;
    let bad = (g_mag == 0u) || (g_mag >= F32_INF);
    var stored = g;
    if (bad) {
        stored = 1.0;
    }
    let x_base = slot * lgq_p.x_slot_stride_elems;
    let out_base = slot * (lgq_p.k_blocks * 2u);
    let sf_base_words = slot * (lgq_p.k_blocks >> 2u);

    let kb = wid.x * 256u + lid.x;
    var sb = 0u;
    if (kb < lgq_p.k_blocks) {
        sb = lgw_qz_block(x_base, out_base, kb, stored);
    }
    lgq_sbytes[lid.x] = sb;
    workgroupBarrier();
    if ((lid.x & 3u) == 0u && kb < lgq_p.k_blocks) {
        lgq_scales[sf_base_words + (kb >> 2u)] = lgq_sbytes[lid.x]
            | (lgq_sbytes[lid.x + 1u] << 8u)
            | (lgq_sbytes[lid.x + 2u] << 16u)
            | (lgq_sbytes[lid.x + 3u] << 24u);
    }
}
