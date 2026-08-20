struct GemvParams {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    w_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> gemv_w_packed: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> gemv_w_scales: array<u32>;
@group(0) @binding(2) var<storage, read> gemv_x_packed: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> gemv_x_scales: array<u32>;
@group(0) @binding(4) var<uniform> gemv_params: GemvParams;
@group(0) @binding(5) var<storage, read_write> gemv_y: array<u32>;

const GEMV_WORKGROUP: u32 = 256u;

var<workgroup> gemv_partial: array<f32, 256>;

var<private> GEMV_SHUFFLE_ORDER: array<u32, 8> = array<u32, 8>(
    16u, 8u, 4u, 2u, 1u, 128u, 64u, 32u
);

const NVFP4_GEMV_DECODE_BEGIN: u32 = 0u;

fn gemv_ue4m3_decode(bits: u32) -> f32 {
    let b = bits & 127u;
    return select(
        bitcast<f32>((b << 20u) + 0x3c000000u),
        f32(b) * UE4M3_SUBNORMAL_STEP,
        b < 8u
    );
}

fn gemv_e2m1_decode(nibble: u32) -> f32 {
    let n = nibble & 15u;
    let k = n & 7u;
    let mag = select((k + 252u) << 22u, (k & 1u) * 0x3f000000u, k < 2u);
    return bitcast<f32>(((n & 8u) << 28u) | mag);
}

fn gemv_i8map(s: u32) -> u32 {
    let k = s & 0x07070707u;
    let hm = ((k >> 2u) & 0x01010101u) * 255u;
    let e7 = (k & (k >> 1u) & (k >> 2u)) & 0x01010101u;
    let m = k + ((k & 0x03030303u) & hm) + (e7 << 1u);
    let sb = (s & ((k + 0x07070707u) & 0x08080808u)) >> 3u;
    return (m ^ (sb * 255u)) + sb;
}

fn gemv_dot8(ww: u32, xw: u32, dot_in: f32) -> f32 {
    let d = dot4I8Packed(gemv_i8map(ww), gemv_i8map(xw))
        + dot4I8Packed(gemv_i8map(ww >> 4u), gemv_i8map(xw >> 4u));
    return dot_in + f32(d) * 0.25;
}

const NVFP4_GEMV_DECODE_END: u32 = 0u;

@compute @workgroup_size(256)
fn gemv_nvfp4_bf16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = wid.x + wid.y * gemv_params.groups_x;
    let tid = lid.x;
    let row_live = row < gemv_params.n_rows;
    let blocks = select(0u, gemv_params.k_blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);

    var acc = 0.0;
    for (var kb = tid; kb < blocks; kb = kb + GEMV_WORKGROUP) {
        let ws_idx = nvfp4_scale_byte_index(row, kb, gemv_params.k_tiles);
        let ws = byte_at(gemv_w_scales[ws_idx >> 2u], ws_idx);
        let xs = byte_at(gemv_x_scales[kb >> 2u], kb);
        let block_scale = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
        let wv = gemv_w_packed[w_vec_base + kb];
        let xv = gemv_x_packed[kb];
        var dot = 0.0;
        dot = gemv_dot8(wv.x, xv.x, dot);
        dot = gemv_dot8(wv.y, xv.y, dot);
        acc = fma(block_scale, dot, acc);
    }

    gemv_partial[tid] = acc;
    workgroupBarrier();
    for (var step = 0u; step < 8u; step = step + 1u) {
        let stride = GEMV_SHUFFLE_ORDER[step];
        let taking = (step < 5u) || ((tid & 31u) == 0u);
        if (taking && (tid & stride) == 0u) {
            gemv_partial[tid] = gemv_partial[tid] + gemv_partial[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u && row_live) {
        gemv_y[row] = bf16_encode(gemv_partial[0] * gemv_params.alpha);
    }
}

const NVFP4_GEMV_SECTION_SPLIT: u32 = 1u;

struct QuantParams {
    global_scale: f32,
    k_blocks: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> quant_x: array<u32>;
@group(0) @binding(1) var<uniform> quant_params: QuantParams;
@group(0) @binding(2) var<storage, read_write> quant_packed: array<u32>;
@group(0) @binding(3) var<storage, read_write> quant_scales: array<u32>;

const QUANT_WORKGROUP: u32 = 256u;
@compute @workgroup_size(256)
fn quantize_row_nvfp4_bf16(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let g = quant_params.global_scale;
    let g_mag = bitcast<u32>(g) & 0x7fffffffu;
    let bad = (g_mag == 0u) || (g_mag >= F32_INF);
    let stored = select(g, 1.0, bad);

    for (var kb = tid; kb < quant_params.k_blocks; kb = kb + QUANT_WORKGROUP) {
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
}

const NVFP4_GEMV_SECTION_SG: u32 = 2u;

const GEMV_SG_LANES: u32 = 32u;
const GEMV_SG_ROWS: u32 = 4u;
const GEMV_SG_WARPS: u32 = 8u;

fn gemv_sg_butterfly(acc: f32) -> f32 {
    var a = acc;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

@compute @workgroup_size(128)
fn gemv_nvfp4_bf16_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * GEMV_SG_ROWS + sgid;
    let row_live = row < gemv_params.n_rows;
    let blocks = select(0u, gemv_params.k_blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);
    let scale_row = select(0u, row, row_live);

    var warp_sums: array<f32, 8>;
    for (var w = 0u; w < GEMV_SG_WARPS; w = w + 1u) {
        var acc = 0.0;
        for (var kb = w * GEMV_SG_LANES + lane; kb < blocks; kb = kb + GEMV_WORKGROUP) {
            let ws_idx = nvfp4_scale_byte_index(scale_row, kb, gemv_params.k_tiles);
            let ws = byte_at(gemv_w_scales[ws_idx >> 2u], ws_idx);
            let xs = byte_at(gemv_x_scales[kb >> 2u], kb);
            let block_scale = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
            let wv = gemv_w_packed[w_vec_base + kb];
            let xv = gemv_x_packed[kb];
            var dot = 0.0;
            dot = gemv_dot8(wv.x, xv.x, dot);
            dot = gemv_dot8(wv.y, xv.y, dot);
            acc = fma(block_scale, dot, acc);
        }
        warp_sums[w] = gemv_sg_butterfly(acc);
    }

    let total = ((warp_sums[0] + warp_sums[4]) + (warp_sums[2] + warp_sums[6]))
        + ((warp_sums[1] + warp_sums[5]) + (warp_sums[3] + warp_sums[7]));
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}

fn gemv_sg_lane_acc(w_vec_base: u32, scale_row: u32, blocks: u32, kb0: u32) -> f32 {
    var acc = 0.0;
    for (var kb = kb0; kb < blocks; kb = kb + GEMV_WORKGROUP) {
        let ws_idx = nvfp4_scale_byte_index(scale_row, kb, gemv_params.k_tiles);
        let ws = byte_at(gemv_w_scales[ws_idx >> 2u], ws_idx);
        let xs = byte_at(gemv_x_scales[kb >> 2u], kb);
        let block_scale = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
        let wv = gemv_w_packed[w_vec_base + kb];
        let xv = gemv_x_packed[kb];
        var dot = 0.0;
        dot = gemv_dot8(wv.x, xv.x, dot);
        dot = gemv_dot8(wv.y, xv.y, dot);
        acc = fma(block_scale, dot, acc);
    }
    return acc;
}

@compute @workgroup_size(128)
fn gemv_nvfp4_bf16_sgu(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gemv_params.groups_x) * GEMV_SG_ROWS + sgid;
    let row_live = row < gemv_params.n_rows;
    let blocks = select(0u, gemv_params.k_blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);
    let scale_row = select(0u, row, row_live);

    let s0 = gemv_sg_butterfly(gemv_sg_lane_acc(w_vec_base, scale_row, blocks, 0u * GEMV_SG_LANES + lane));
    let s1 = gemv_sg_butterfly(gemv_sg_lane_acc(w_vec_base, scale_row, blocks, 1u * GEMV_SG_LANES + lane));
    let s2 = gemv_sg_butterfly(gemv_sg_lane_acc(w_vec_base, scale_row, blocks, 2u * GEMV_SG_LANES + lane));
    let s3 = gemv_sg_butterfly(gemv_sg_lane_acc(w_vec_base, scale_row, blocks, 3u * GEMV_SG_LANES + lane));
    let s4 = gemv_sg_butterfly(gemv_sg_lane_acc(w_vec_base, scale_row, blocks, 4u * GEMV_SG_LANES + lane));
    let s5 = gemv_sg_butterfly(gemv_sg_lane_acc(w_vec_base, scale_row, blocks, 5u * GEMV_SG_LANES + lane));
    let s6 = gemv_sg_butterfly(gemv_sg_lane_acc(w_vec_base, scale_row, blocks, 6u * GEMV_SG_LANES + lane));
    let s7 = gemv_sg_butterfly(gemv_sg_lane_acc(w_vec_base, scale_row, blocks, 7u * GEMV_SG_LANES + lane));

    let total = ((s0 + s4) + (s2 + s6)) + ((s1 + s5) + (s3 + s7));
    if (lane == 0u && row_live) {
        gemv_y[row] = bf16_encode(total * gemv_params.alpha);
    }
}

const NVFP4_GEMV_SECTION_SGW: u32 = 3u;

const SGW_WG: u32 = 256u;
const SGW_ROWS: u32 = 1u;
const SGW_SGPR: u32 = 8u;
const SGW_VW: u32 = 1u;
const SGW_TILED: u32 = 0u;
const SGW_STAGE: u32 = 0u;
const SGW_STAGE_LEN: u32 = 1u;
const SGW_PART_LEN: u32 = 8u;

var<workgroup> sgw_part: array<f32, SGW_PART_LEN>;
var<workgroup> sgw_xp: array<vec2<u32>, SGW_STAGE_LEN>;
var<workgroup> sgw_xs: array<f32, SGW_STAGE_LEN>;

fn sgw_block(w_vec_base: u32, scale_row: u32, kb: u32, tile_base: u32, acc_in: f32) -> f32 {
    let ws_idx = nvfp4_scale_byte_index(scale_row, kb, gemv_params.k_tiles);
    let ws = byte_at(gemv_w_scales[ws_idx >> 2u], ws_idx);
    var xsf: f32;
    var xv: vec2<u32>;
    if (SGW_STAGE == 1u) {
        let j = kb - tile_base;
        xsf = sgw_xs[j];
        xv = sgw_xp[j];
    } else {
        xsf = gemv_ue4m3_decode(byte_at(gemv_x_scales[kb >> 2u], kb));
        xv = gemv_x_packed[kb];
    }
    let block_scale = gemv_ue4m3_decode(ws) * xsf;
    let wv = gemv_w_packed[w_vec_base + kb];
    var dot = 0.0;
    dot = gemv_dot8(wv.x, xv.x, dot);
    dot = gemv_dot8(wv.y, xv.y, dot);
    return fma(block_scale, dot, acc_in);
}

@compute @workgroup_size(SGW_WG)
fn gemv_nvfp4_bf16_sgw(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row_in_wg = sgid / SGW_SGPR;
    let slot = sgid % SGW_SGPR;
    let row = (wid.x + wid.y * gemv_params.groups_x) * SGW_ROWS + row_in_wg;
    let row_live = row < gemv_params.n_rows;
    let blocks = gemv_params.k_blocks;
    let live_blocks = select(0u, blocks, row_live);
    let w_vec_base = select(0u, row * (gemv_params.w_row_words >> 1u), row_live);
    let scale_row = select(0u, row, row_live);

    var a: array<f32, SGW_VW>;
    if (SGW_TILED == 0u) {
        for (var m = 0u; m < SGW_VW; m = m + 1u) {
            var acc = 0.0;
            for (var kb = (slot + m * SGW_SGPR) * 32u + lane; kb < live_blocks; kb = kb + GEMV_WORKGROUP) {
                acc = sgw_block(w_vec_base, scale_row, kb, 0u, acc);
            }
            a[m] = acc;
        }
    } else {
        for (var m = 0u; m < SGW_VW; m = m + 1u) {
            a[m] = 0.0;
        }
        let tiles = (blocks + GEMV_WORKGROUP - 1u) / GEMV_WORKGROUP;
        for (var t = 0u; t < tiles; t = t + 1u) {
            let base = t * GEMV_WORKGROUP;
            if (SGW_STAGE == 1u) {
                workgroupBarrier();
                for (var j = lidx; j < GEMV_WORKGROUP; j = j + SGW_WG) {
                    let kb = base + j;
                    if (kb < blocks) {
                        sgw_xp[j] = gemv_x_packed[kb];
                        sgw_xs[j] = gemv_ue4m3_decode(byte_at(gemv_x_scales[kb >> 2u], kb));
                    }
                }
                workgroupBarrier();
            }
            for (var m = 0u; m < SGW_VW; m = m + 1u) {
                let kb = base + (slot + m * SGW_SGPR) * 32u + lane;
                if (kb < live_blocks) {
                    a[m] = sgw_block(w_vec_base, scale_row, kb, base, a[m]);
                }
            }
        }
    }

    for (var m = 0u; m < SGW_VW; m = m + 1u) {
        a[m] = gemv_sg_butterfly(a[m]);
    }
    for (var stride = SGW_VW >> 1u; stride > 0u; stride = stride >> 1u) {
        for (var m = 0u; m < stride; m = m + 1u) {
            a[m] = a[m] + a[m + stride];
        }
    }

    if (SGW_SGPR == 1u) {
        if (lane == 0u && row_live) {
            gemv_y[row] = bf16_encode(a[0] * gemv_params.alpha);
        }
    } else {
        if (lane == 0u) {
            sgw_part[sgid] = a[0];
        }
        workgroupBarrier();
        for (var stride = SGW_SGPR >> 1u; stride > 0u; stride = stride >> 1u) {
            if (lane == 0u && slot < stride) {
                sgw_part[sgid] = sgw_part[sgid] + sgw_part[sgid + stride];
            }
            workgroupBarrier();
        }
        if (lane == 0u && slot == 0u && row_live) {
            gemv_y[row] = bf16_encode(sgw_part[sgid] * gemv_params.alpha);
        }
    }
}

@group(0) @binding(6) var<storage, read> gemv_w_packed4: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> gemv_x_packed4: array<vec4<u32>>;

fn sgq_pair(w0: u32, w1: u32, x0: u32, x1: u32, wsb: u32, xsb: u32, acc_in: f32) -> f32 {
    let block_scale = gemv_ue4m3_decode(wsb) * gemv_ue4m3_decode(xsb);
    var dot = 0.0;
    dot = gemv_dot8(w0, x0, dot);
    dot = gemv_dot8(w1, x1, dot);
    return fma(block_scale, dot, acc_in);
}

fn sgq_quad(w4_base: u32, scale_row: u32, q: u32, acc_in: f32) -> f32 {
    let wsi = nvfp4_scale_byte_index(scale_row, q << 2u, gemv_params.k_tiles);
    let wsw = gemv_w_scales[wsi >> 2u];
    let xsw = gemv_x_scales[q];
    let wa = gemv_w_packed4[w4_base + 2u * q];
    let wb = gemv_w_packed4[w4_base + 2u * q + 1u];
    let xa = gemv_x_packed4[2u * q];
    let xb = gemv_x_packed4[2u * q + 1u];
    var acc = acc_in;
    acc = sgq_pair(wa.x, wa.y, xa.x, xa.y, byte_at(wsw, 0u), byte_at(xsw, 0u), acc);
    acc = sgq_pair(wa.z, wa.w, xa.z, xa.w, byte_at(wsw, 1u), byte_at(xsw, 1u), acc);
    acc = sgq_pair(wb.x, wb.y, xb.x, xb.y, byte_at(wsw, 2u), byte_at(xsw, 2u), acc);
    acc = sgq_pair(wb.z, wb.w, xb.z, xb.w, byte_at(wsw, 3u), byte_at(xsw, 3u), acc);
    return acc;
}

@compute @workgroup_size(SGW_WG)
fn gemv_nvfp4_bf16_sgq(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row_in_wg = sgid / SGW_SGPR;
    let slot = sgid % SGW_SGPR;
    let row = (wid.x + wid.y * gemv_params.groups_x) * SGW_ROWS + row_in_wg;
    let row_live = row < gemv_params.n_rows;
    let quads = select(0u, gemv_params.k_blocks >> 2u, row_live);
    let w4_base = select(0u, row * (gemv_params.w_row_words >> 2u), row_live);
    let scale_row = select(0u, row, row_live);

    var a: array<f32, SGW_VW>;
    for (var m = 0u; m < SGW_VW; m = m + 1u) {
        var acc = 0.0;
        for (var q = (slot + m * SGW_SGPR) * 32u + lane; q < quads; q = q + GEMV_WORKGROUP) {
            acc = sgq_quad(w4_base, scale_row, q, acc);
        }
        a[m] = gemv_sg_butterfly(acc);
    }
    for (var stride = SGW_VW >> 1u; stride > 0u; stride = stride >> 1u) {
        for (var m = 0u; m < stride; m = m + 1u) {
            a[m] = a[m] + a[m + stride];
        }
    }

    if (SGW_SGPR == 1u) {
        if (lane == 0u && row_live) {
            gemv_y[row] = bf16_encode(a[0] * gemv_params.alpha);
        }
    } else {
        if (lane == 0u) {
            sgw_part[sgid] = a[0];
        }
        workgroupBarrier();
        for (var stride = SGW_SGPR >> 1u; stride > 0u; stride = stride >> 1u) {
            if (lane == 0u && slot < stride) {
                sgw_part[sgid] = sgw_part[sgid] + sgw_part[sgid + stride];
            }
            workgroupBarrier();
        }
        if (lane == 0u && slot == 0u && row_live) {
            gemv_y[row] = bf16_encode(sgw_part[sgid] * gemv_params.alpha);
        }
    }
}
