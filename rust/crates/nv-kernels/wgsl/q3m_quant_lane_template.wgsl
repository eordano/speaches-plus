
@compute @workgroup_size(QL_WORKGROUP_THREADS)
fn QL_ENTRY_POINT(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
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

    let el = lane & 7u;
    let kb = wid.x * QL_BLOCKS_PER_WORKGROUPu + sgid * 4u + (lane >> 3u);
    let live = kb < q3q_p.k_blocks;
QL_LOAD_HEAD
    var m0 = v0 & 0x7fffu;
    if (m0 > 0x7f80u) { m0 = 0u; }
    var m1 = v1 & 0x7fffu;
    if (m1 > 0x7f80u) { m1 = 0u; }
    var amax_bits = max(m0, m1);
    amax_bits = max(amax_bits, subgroupShuffleXor(amax_bits, 1u));
    amax_bits = max(amax_bits, subgroupShuffleXor(amax_bits, 2u));
    amax_bits = max(amax_bits, subgroupShuffleXor(amax_bits, 4u));

    var local_scale = 1.0;
    if (amax_bits != 0u) {
        local_scale = q_div_small(bf16_decode(amax_bits), 3u, 1);
    }
    let scale_byte = q_encode_scale(stored, local_scale);
    let parts = q_scale_parts(scale_byte);
    let inv = q_div_small(stored, u32(parts.x), parts.y);
    let u_up = q_subnormal_shift(inv);
    let inv_up = q_scale_up_pow2(inv, u_up);

    let lo = nvfp4_encode_e2m1(q_scaled_product(v0, inv_up, u_up));
    let hi = nvfp4_encode_e2m1(q_scaled_product(v1, inv_up, u_up));
    var word = (((hi & 15u) << 4u) | (lo & 15u)) << (8u * (el & 3u));
    word = word | subgroupShuffleXor(word, 1u);
    word = word | subgroupShuffleXor(word, 2u);

    var sw = 0u;
    if (live) {
        sw = (scale_byte & 255u) << (8u * (kb & 3u));
    }
    sw = sw | subgroupShuffleXor(sw, 1u);
    sw = sw | subgroupShuffleXor(sw, 2u);
    sw = sw | subgroupShuffleXor(sw, 4u);
    sw = sw | subgroupShuffleXor(sw, 8u);
    sw = sw | subgroupShuffleXor(sw, 16u);

    if (live) {
        if (el == 0u) {
            q3q_packed[out_base + kb * 2u] = word;
        }
        if (el == 4u) {
            q3q_packed[out_base + kb * 2u + 1u] = word;
        }
        if ((lane & 31u) == 0u) {
            q3q_scales[sf_base_words + (kb >> 2u)] = sw;
        }
    }
}
