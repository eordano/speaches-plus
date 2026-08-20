
fn qzg_stored() -> f32 {
    let g = quant_params.global_scale;
    let g_mag = bitcast<u32>(g) & 0x7fffffffu;
    let bad = (g_mag == 0u) || (g_mag >= F32_INF);
    return select(g, 1.0, bad);
}

fn qzg_block(kb: u32, stored: f32) {
    var vbits: array<u32, 16>;
    var amax_bits = 0u;
    let base = kb * NVFP4_BLOCK_SIZE;
    for (var i = 0u; i < NVFP4_BLOCK_SIZE; i = i + 1u) {
        let j = base + i;
        let w = u16_at(quant_x[j >> 1u], j);
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
    quant_packed[kb * 2u] = w0;
    quant_packed[kb * 2u + 1u] = w1;
    quant_scales[kb] = scale_byte;
}

@compute @workgroup_size(256)
fn quantize_row_nvfp4_bf16_grid(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wgc: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let stored = qzg_stored();
    let stride = wgc.x * 256u;
    for (var kb = wid.x * 256u + lid.x; kb < quant_params.k_blocks; kb = kb + stride) {
        qzg_block(kb, stored);
    }
}

@compute @workgroup_size(64)
fn quantize_row_nvfp4_bf16_grid_wg64(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(num_workgroups) wgc: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let stored = qzg_stored();
    let stride = wgc.x * 64u;
    for (var kb = wid.x * 64u + lid.x; kb < quant_params.k_blocks; kb = kb + stride) {
        qzg_block(kb, stored);
    }
}
