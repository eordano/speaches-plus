
struct Q3qParams {
    k_blocks: u32,
    n_slots: u32,
    use_sel: u32,
    x_slot_stride_elems: u32,
};

struct Q3sqParams {
    u_off_elems: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> q3q_x: array<u32>;
@group(0) @binding(11) var<uniform> q3q_p: Q3qParams;
@group(0) @binding(12) var<storage, read_write> q3q_packed: array<u32>;
@group(0) @binding(13) var<storage, read_write> q3q_scales: array<u32>;
@group(0) @binding(14) var<storage, read> q3q_sel: array<u32>;
@group(0) @binding(15) var<storage, read> q3q_glob: array<f32>;
@group(0) @binding(16) var<storage, read> q3sq_g: array<u32>;
@group(0) @binding(17) var<storage, read> q3sq_u: array<u32>;
@group(0) @binding(18) var<uniform> q3sq_p: Q3sqParams;

var<workgroup> q3q_sbytes: array<u32, 256>;

fn q3w_qz_core(vbits: ptr<function, array<u32, 16>>, out_base: u32, kb: u32, stored: f32) -> u32 {
    var amax_bits = 0u;
    for (var i = 0u; i < NVFP4_BLOCK_SIZE; i = i + 1u) {
        let mag = (*vbits)[i] & 0x7fffu;
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
        let lo = nvfp4_encode_e2m1(q_scaled_product((*vbits)[2u * i], inv_up, u_up));
        let hi = nvfp4_encode_e2m1(q_scaled_product((*vbits)[2u * i + 1u], inv_up, u_up));
        let packed = ((hi & 15u) << 4u) | (lo & 15u);
        if (i < 4u) {
            w0 = w0 | (packed << (8u * i));
        } else {
            w1 = w1 | (packed << (8u * (i - 4u)));
        }
    }
    q3q_packed[out_base + kb * 2u] = w0;
    q3q_packed[out_base + kb * 2u + 1u] = w1;
    return scale_byte & 255u;
}

fn q3w_qz_block(x_base: u32, out_base: u32, kb: u32, stored: f32) -> u32 {
    var vbits: array<u32, 16>;
    let base = x_base + kb * NVFP4_BLOCK_SIZE;
    for (var i = 0u; i < NVFP4_BLOCK_SIZE; i = i + 1u) {
        let j = base + i;
        vbits[i] = u16_at(q3q_x[j >> 1u], j);
    }
    return q3w_qz_core(&vbits, out_base, kb, stored);
}

@compute @workgroup_size(256)
fn q3w_quant_rows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let slot = wid.y;
    var gi = slot;
    if (q3q_p.use_sel == 1u) {
        gi = q3q_sel[slot];
    }
    let g = q3q_glob[gi];
    let g_mag = bitcast<u32>(g) & 0x7fffffffu;
    let bad = (g_mag == 0u) || (g_mag >= F32_INF);
    var stored = g;
    if (bad) {
        stored = 1.0;
    }
    let x_base = slot * q3q_p.x_slot_stride_elems;
    let out_base = slot * (q3q_p.k_blocks * 2u);
    let sf_base_words = slot * (q3q_p.k_blocks >> 2u);

    let kb = wid.x * 256u + lid.x;
    var sb = 0u;
    if (kb < q3q_p.k_blocks) {
        sb = q3w_qz_block(x_base, out_base, kb, stored);
    }
    q3q_sbytes[lid.x] = sb;
    workgroupBarrier();
    if ((lid.x & 3u) == 0u && kb < q3q_p.k_blocks) {
        q3q_scales[sf_base_words + (kb >> 2u)] = q3q_sbytes[lid.x]
            | (q3q_sbytes[lid.x + 1u] << 8u)
            | (q3q_sbytes[lid.x + 2u] << 16u)
            | (q3q_sbytes[lid.x + 3u] << 24u);
    }
}

@compute @workgroup_size(256)
fn q3w_silu_mul_quant(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let slot = wid.y;
    var gi = slot;
    if (q3q_p.use_sel == 1u) {
        gi = q3q_sel[slot];
    }
    let g = q3q_glob[gi];
    let g_mag = bitcast<u32>(g) & 0x7fffffffu;
    let bad = (g_mag == 0u) || (g_mag >= F32_INF);
    var stored = g;
    if (bad) {
        stored = 1.0;
    }
    let out_base = slot * (q3q_p.k_blocks * 2u);
    let sf_base_words = slot * (q3q_p.k_blocks >> 2u);

    let kb = wid.x * 256u + lid.x;
    var sb = 0u;
    if (kb < q3q_p.k_blocks) {
        let ebase = slot * q3q_p.x_slot_stride_elems + kb * NVFP4_BLOCK_SIZE;
        var vbits: array<u32, 16>;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let gw = q3sq_g[(ebase >> 1u) + i];
            let uw = q3sq_u[((ebase + q3sq_p.u_off_elems) >> 1u) + i];
            let g0 = bf16_lo(gw);
            let g1 = bf16_hi(gw);
            let a0 = bf16_decode(bf16_encode(g0 / (1.0 + exp(-g0)))) * bf16_lo(uw);
            let a1 = bf16_decode(bf16_encode(g1 / (1.0 + exp(-g1)))) * bf16_hi(uw);
            vbits[2u * i] = bf16_encode(a0) & 0xffffu;
            vbits[2u * i + 1u] = bf16_encode(a1) & 0xffffu;
        }
        sb = q3w_qz_core(&vbits, out_base, kb, stored);
    }
    q3q_sbytes[lid.x] = sb;
    workgroupBarrier();
    if ((lid.x & 3u) == 0u && kb < q3q_p.k_blocks) {
        q3q_scales[sf_base_words + (kb >> 2u)] = q3q_sbytes[lid.x]
            | (q3q_sbytes[lid.x + 1u] << 8u)
            | (q3q_sbytes[lid.x + 2u] << 16u)
            | (q3q_sbytes[lid.x + 3u] << 24u);
    }
}
