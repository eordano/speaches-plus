struct QuantizeNvfp4Params {
    rows: u32,
    m_data_rows: u32,
    m_read_rows: u32,
    k: u32,
    k_tiles: u32,
    blocks_per_row: u32,
    rows_per_expert: u32,
    mode: u32,
};

@group(0) @binding(0) var<storage, read> qz_x: array<u32>;
@group(0) @binding(1) var<storage, read> qz_y: array<u32>;
@group(0) @binding(2) var<storage, read> qz_globals: array<f32>;
@group(0) @binding(3) var<storage, read_write> qz_packed: array<u32>;
@group(0) @binding(4) var<storage, read_write> qz_scales: array<u32>;
@group(0) @binding(5) var<uniform> qz_params: QuantizeNvfp4Params;

const QZ_WG: u32 = 256u;
const QZ_MODE_SILU_MUL: u32 = 1u;

fn qz_is_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7f800000u) != 0x7f800000u;
}

fn qz_silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}

fn qz_encode_e2m1(x: f32) -> u32 {
    let sign = (bitcast<u32>(x) >> 31u) << 3u;
    let a = abs(x);
    var mag = 0u;
    if (a > 0.25) { mag = 1u; }
    if (a > 0.75) { mag = 2u; }
    if (a > 1.25) { mag = 3u; }
    if (a > 1.75) { mag = 4u; }
    if (a > 2.5) { mag = 5u; }
    if (a > 3.5) { mag = 6u; }
    if (a > 5.0) { mag = 7u; }
    return sign | mag;
}

fn qz_quantize_block(row: u32, kb: u32, stored: f32) -> u32 {
    var vals: array<f32, 16>;
    let in_base = row * (qz_params.k >> 1u) + kb * 8u;
    var amax = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let w = qz_x[in_base + i];
        var a = bf16_lo(w);
        var b = bf16_hi(w);
        if (qz_params.mode == QZ_MODE_SILU_MUL) {
            let wy = qz_y[in_base + i];
            a = qz_silu(a) * bf16_lo(wy);
            b = qz_silu(b) * bf16_hi(wy);
        }
        vals[2u * i] = a;
        vals[2u * i + 1u] = b;
        amax = max(amax, max(abs(a), abs(b)));
    }
    let local_scale = select(q_div_small(amax, 3u, 1), 1.0, amax == 0.0);
    let scale_byte = q_encode_scale_ref(stored, local_scale);
    let parts = q_scale_parts_ref(scale_byte);
    let inv = select(q_div_small(stored, u32(parts.x), parts.y), 1.0, scale_byte == 0u);

    var w0 = 0u;
    var w1 = 0u;
    for (var i = 0u; i < 8u; i = i + 1u) {
        let lo = qz_encode_e2m1(clamp(vals[2u * i] * inv, -E2M1_MAX, E2M1_MAX));
        let hi = qz_encode_e2m1(clamp(vals[2u * i + 1u] * inv, -E2M1_MAX, E2M1_MAX));
        let packed_byte = ((hi << 4u) | (lo & 15u)) & 255u;
        if (i < 4u) {
            w0 = w0 | (packed_byte << (8u * i));
        } else {
            w1 = w1 | (packed_byte << (8u * (i - 4u)));
        }
    }
    let out_base = row * (qz_params.k >> 3u) + kb * 2u;
    qz_packed[out_base] = w0;
    qz_packed[out_base + 1u] = w1;
    return scale_byte;
}

@compute @workgroup_size(256)
fn quantize_nvfp4_bf16(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(num_workgroups) wg_count: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let flat = (wg_id.x + wg_id.y * wg_count.x) * QZ_WG + lid.x;
    if (flat >= qz_params.rows * qz_params.k_tiles) {
        return;
    }
    let row = flat / qz_params.k_tiles;
    let t = flat % qz_params.k_tiles;

    let row_in_range = row < qz_params.m_read_rows;
    let data_row = row < qz_params.m_data_rows;

    let expert = min(
        row / max(qz_params.rows_per_expert, 1u),
        arrayLength(&qz_globals) - 1u
    );
    let raw = qz_globals[expert];
    let stored = select(raw, 1.0, raw == 0.0 || !qz_is_finite(raw));

    var scale_word = 0u;
    for (var j = 0u; j < 4u; j = j + 1u) {
        let kb = t * 4u + j;
        let live = kb < qz_params.blocks_per_row;
        var scale_byte = 0u;
        if (row_in_range && live) {
            scale_byte = qz_quantize_block(row, kb, stored);
        } else if (data_row && live) {
            let out_base = row * (qz_params.k >> 3u) + kb * 2u;
            qz_packed[out_base] = 0u;
            qz_packed[out_base + 1u] = 0u;
        }
        scale_word = scale_word | (scale_byte << (8u * j));
    }
    qz_scales[nvfp4_scale_byte_index(row, t * 4u, qz_params.k_tiles) >> 2u] = scale_word;
}
