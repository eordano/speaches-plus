
struct GowMxParams {
    n_rows: u32,
    k_blocks: u32,
    groups_x: u32,
    has_bias: u32,
    w_e_stride_v4: u32,
    sf_e_stride_bytes: u32,
    bias_e_stride: u32,
    x_slot_stride_words: u32,
    y_slot_stride: u32,
    use_sel: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(10) var<storage, read> gow_mx_w: array<vec4<u32>>;
@group(0) @binding(11) var<storage, read> gow_mx_sf: array<u32>;
@group(0) @binding(12) var<storage, read> gow_mx_xv4: array<vec4<u32>>;
@group(0) @binding(13) var<uniform> gow_mx_p: GowMxParams;
@group(0) @binding(14) var<storage, read_write> gow_mx_y: array<f32>;
@group(0) @binding(15) var<storage, read> gow_mx_sel: array<u32>;
@group(0) @binding(16) var<storage, read> gow_mx_b: array<u32>;

fn gow_e8m0(byte: u32) -> f32 {
    return bitcast<f32>((byte & 255u) << 23u);
}

fn gow_mx_e2m1(word: u32, j: u32) -> f32 {
    let n = nvfp4_nibble(word, j);
    let c = n & 7u;
    let normal = (((c >> 1u) + 126u) << 23u) | ((c & 1u) << 22u);
    let mag = select(normal, c * 0x3F000000u, c < 2u);
    return bitcast<f32>(mag | ((n & 8u) << 28u));
}

const GOW_MX_SG_LANES: u32 = 32u;
const GOW_MX_SG_ROWS: u32 = 8u;

fn gow_mx_sg_butterfly32(v: f32) -> f32 {
    var a = v;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn gow_mx_dot8(word: u32, x0: u32, x1: u32, x2: u32, x3: u32) -> f32 {
    var a = fma(gow_mx_e2m1(word, 0u), bf16_lo(x0), 0.0);
    a = fma(gow_mx_e2m1(word, 1u), bf16_hi(x0), a);
    a = fma(gow_mx_e2m1(word, 2u), bf16_lo(x1), a);
    a = fma(gow_mx_e2m1(word, 3u), bf16_hi(x1), a);
    a = fma(gow_mx_e2m1(word, 4u), bf16_lo(x2), a);
    a = fma(gow_mx_e2m1(word, 5u), bf16_hi(x2), a);
    a = fma(gow_mx_e2m1(word, 6u), bf16_lo(x3), a);
    a = fma(gow_mx_e2m1(word, 7u), bf16_hi(x3), a);
    return a;
}

fn gow_mx_dot_block_xv4(wv: vec4<u32>, xq: u32) -> f32 {
    let xa = gow_mx_xv4[xq];
    let xb = gow_mx_xv4[xq + 1u];
    let xc = gow_mx_xv4[xq + 2u];
    let xd = gow_mx_xv4[xq + 3u];
    var d = gow_mx_dot8(wv.x, xa.x, xa.y, xa.z, xa.w);
    d = d + gow_mx_dot8(wv.y, xb.x, xb.y, xb.z, xb.w);
    d = d + gow_mx_dot8(wv.z, xc.x, xc.y, xc.z, xc.w);
    d = d + gow_mx_dot8(wv.w, xd.x, xd.y, xd.z, xd.w);
    return d;
}

@compute @workgroup_size(256)
fn gow_gemv_mx_sg_xv4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    #def b5
    var e = slot;
    if (gow_mx_p.use_sel == 1u) {
        e = gow_mx_sel[slot];
    }
    let row = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_ROWS + sgid;
    let live = row < gow_mx_p.n_rows;
    let blocks = select(0u, gow_mx_p.k_blocks, live);
    let wbase = e * gow_mx_p.w_e_stride_v4 + select(0u, row * gow_mx_p.k_blocks, live);
    let sfbase = e * gow_mx_p.sf_e_stride_bytes + select(0u, row * gow_mx_p.k_blocks, live);
    #end
    let xqbase = slot * (gow_mx_p.x_slot_stride_words >> 2u);

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + GOW_MX_SG_LANES) {
        let si = sfbase + kb;
        let dot = gow_mx_dot_block_xv4(gow_mx_w[wbase + kb], xqbase + kb * 4u);
        acc = fma(gow_e8m0(byte_at(gow_mx_sf[si >> 2u], si)), dot, acc);
    }
    let total = gow_mx_sg_butterfly32(acc);

    if (lane == 0u && live) {
        var v = total;
        if (gow_mx_p.has_bias == 1u) {
            let bi = e * gow_mx_p.bias_e_stride + row;
            v = v + bf16_decode(u16_at(gow_mx_b[bi >> 1u], bi));
        }
        gow_mx_y[slot * gow_mx_p.y_slot_stride + row] = v;
    }
}

const GOW_MX_SG_R2_ROWS_PER_WG: u32 = 16u;

var<workgroup> gow_mx_xv4_byte_lut_every_e2m1_value_is_exact_in_f16: array<u32, 256>;

fn gow_mx_lut_pair(b: u32) -> vec2<f32> {
    return unpack2x16float(gow_mx_xv4_byte_lut_every_e2m1_value_is_exact_in_f16[b]);
}

fn gow_mx_lut_dot8(word: u32, x0: u32, x1: u32, x2: u32, x3: u32) -> f32 {
    let p0 = gow_mx_lut_pair(extractBits(word, 0u, 8u));
    let p1 = gow_mx_lut_pair(extractBits(word, 8u, 8u));
    let p2 = gow_mx_lut_pair(extractBits(word, 16u, 8u));
    let p3 = gow_mx_lut_pair(extractBits(word, 24u, 8u));
    var a = fma(p0.x, bf16_lo(x0), 0.0);
    a = fma(p0.y, bf16_hi(x0), a);
    a = fma(p1.x, bf16_lo(x1), a);
    a = fma(p1.y, bf16_hi(x1), a);
    a = fma(p2.x, bf16_lo(x2), a);
    a = fma(p2.y, bf16_hi(x2), a);
    a = fma(p3.x, bf16_lo(x3), a);
    a = fma(p3.y, bf16_hi(x3), a);
    return a;
}

const GOW_MX_XS_VEC4_CAP_COVERS_K_UP_TO_4096: u32 = 512u;

var<workgroup> gow_mx_xsh: array<vec4<u32>, 512>;

const GOW_MX_SG_R4SW_ROWS_PER_WG: u32 = 32u;

fn gow_mx_r4sw_swiglu_one_matches_gow_swiglu_one_bitwise(gate_raw: f32, up_raw: f32) -> f32 {
    let limit = bitcast<f32>(gow_mx_p.pad0);
    let alpha = bitcast<f32>(gow_mx_p.pad1);
    let g = min(gate_raw, limit);
    let u = clamp(up_raw, -limit, limit);
    let glu = g / (1.0 + exp(-alpha * g));
    return glu * (u + 1.0);
}

fn gow_mx_r4sw_row_dot(wbase: u32, sfbase: u32, blocks: u32, lane: u32) -> f32 {
    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + GOW_MX_SG_LANES) {
        let xq = kb * 4u;
        let xa = gow_mx_xsh[xq];
        let xb = gow_mx_xsh[xq + 1u];
        let xc = gow_mx_xsh[xq + 2u];
        let xd = gow_mx_xsh[xq + 3u];
        #def b6
        let wv = gow_mx_w[wbase + kb];
        var d = gow_mx_lut_dot8(wv.x, xa.x, xa.y, xa.z, xa.w);
        d = d + gow_mx_lut_dot8(wv.y, xb.x, xb.y, xb.z, xb.w);
        d = d + gow_mx_lut_dot8(wv.z, xc.x, xc.y, xc.z, xc.w);
        d = d + gow_mx_lut_dot8(wv.w, xd.x, xd.y, xd.z, xd.w);
        let si = sfbase + kb;
        acc = fma(gow_e8m0(byte_at(gow_mx_sf[si >> 2u], si)), d, acc);
    }
        #end
    return acc;
}

fn gow_mx_biased_row(e: u32, row: u32, total: f32) -> f32 {
    var v = total;
    if (gow_mx_p.has_bias == 1u) {
        let bi = e * gow_mx_p.bias_e_stride + row;
        v = v + bf16_decode(u16_at(gow_mx_b[bi >> 1u], bi));
    }
    return v;
}

@compute @workgroup_size(256)
fn gow_gemv_mx_sg_xv4_r4_lut_xs_swiglu(
    #def b4
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    gow_mx_xv4_byte_lut_every_e2m1_value_is_exact_in_f16[lid.x] =
        pack2x16float(vec2<f32>(gow_mx_e2m1(lid.x, 0u), gow_mx_e2m1(lid.x, 1u)));
    let slot = wid.z;
    let xqbase = slot * (gow_mx_p.x_slot_stride_words >> 2u);
    let xwords4 = gow_mx_p.k_blocks * 4u;
    for (var i = lid.x; i < xwords4; i = i + 256u) {
        gow_mx_xsh[i] = gow_mx_xv4[xqbase + i];
    }
    workgroupBarrier();
    #end

    var e = slot;
    if (gow_mx_p.use_sel == 1u) {
        e = gow_mx_sel[slot];
    }
    let r0 = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_R4SW_ROWS_PER_WG + sgid * 4u;
    let live = r0 + 3u < gow_mx_p.n_rows;
    let blocks = select(0u, gow_mx_p.k_blocks, live);
    let ebase_w = e * gow_mx_p.w_e_stride_v4;
    let ebase_sf = e * gow_mx_p.sf_e_stride_bytes;
    let wrow = select(0u, r0 * gow_mx_p.k_blocks, live);
    let t0 = gow_mx_sg_butterfly32(gow_mx_r4sw_row_dot(ebase_w + wrow, ebase_sf + wrow, blocks, lane));
    let t1 = gow_mx_sg_butterfly32(gow_mx_r4sw_row_dot(ebase_w + wrow + blocks, ebase_sf + wrow + blocks, blocks, lane));
    let t2 = gow_mx_sg_butterfly32(gow_mx_r4sw_row_dot(ebase_w + wrow + 2u * blocks, ebase_sf + wrow + 2u * blocks, blocks, lane));
    let t3 = gow_mx_sg_butterfly32(gow_mx_r4sw_row_dot(ebase_w + wrow + 3u * blocks, ebase_sf + wrow + 3u * blocks, blocks, lane));
    if (lane == 0u && live) {
        let a0 = gow_mx_r4sw_swiglu_one_matches_gow_swiglu_one_bitwise(
            gow_mx_biased_row(e, r0, t0),
            gow_mx_biased_row(e, r0 + 1u, t1)
        );
        let a1 = gow_mx_r4sw_swiglu_one_matches_gow_swiglu_one_bitwise(
            gow_mx_biased_row(e, r0 + 2u, t2),
            gow_mx_biased_row(e, r0 + 3u, t3)
        );
        gow_mx_y[slot * gow_mx_p.y_slot_stride + (r0 >> 2u)] = bitcast<f32>(bf16_pack(a0, a1));
    }
}

@compute @workgroup_size(256)
fn gow_gemv_mx_sg_xv4_r2_lut_xs(
    #use b4

    #def b3
    var e = slot;
    if (gow_mx_p.use_sel == 1u) {
        e = gow_mx_sel[slot];
    }
    let row0 = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_R2_ROWS_PER_WG + sgid * 2u;
    let row1 = row0 + 1u;
    let live0 = row0 < gow_mx_p.n_rows;
    let live1 = row1 < gow_mx_p.n_rows;
    let blocks = select(0u, gow_mx_p.k_blocks, live0);
    let wbase0 = e * gow_mx_p.w_e_stride_v4 + select(0u, row0 * gow_mx_p.k_blocks, live0);
    let wbase1 = e * gow_mx_p.w_e_stride_v4 + select(0u, row1 * gow_mx_p.k_blocks, live1);
    let sfbase0 = e * gow_mx_p.sf_e_stride_bytes + select(0u, row0 * gow_mx_p.k_blocks, live0);
    let sfbase1 = e * gow_mx_p.sf_e_stride_bytes + select(0u, row1 * gow_mx_p.k_blocks, live1);
    #end

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + GOW_MX_SG_LANES) {
        let xq = kb * 4u;
        let xa = gow_mx_xsh[xq];
        let xb = gow_mx_xsh[xq + 1u];
        let xc = gow_mx_xsh[xq + 2u];
        let xd = gow_mx_xsh[xq + 3u];
        #def b2
        let wv0 = gow_mx_w[wbase0 + kb];
        var d0 = gow_mx_lut_dot8(wv0.x, xa.x, xa.y, xa.z, xa.w);
        d0 = d0 + gow_mx_lut_dot8(wv0.y, xb.x, xb.y, xb.z, xb.w);
        d0 = d0 + gow_mx_lut_dot8(wv0.z, xc.x, xc.y, xc.z, xc.w);
        d0 = d0 + gow_mx_lut_dot8(wv0.w, xd.x, xd.y, xd.z, xd.w);
        let si0 = sfbase0 + kb;
        acc0 = fma(gow_e8m0(byte_at(gow_mx_sf[si0 >> 2u], si0)), d0, acc0);
        if (live1) {
            let wv1 = gow_mx_w[wbase1 + kb];
            var d1 = gow_mx_lut_dot8(wv1.x, xa.x, xa.y, xa.z, xa.w);
            d1 = d1 + gow_mx_lut_dot8(wv1.y, xb.x, xb.y, xb.z, xb.w);
            d1 = d1 + gow_mx_lut_dot8(wv1.z, xc.x, xc.y, xc.z, xc.w);
            d1 = d1 + gow_mx_lut_dot8(wv1.w, xd.x, xd.y, xd.z, xd.w);
            let si1 = sfbase1 + kb;
            acc1 = fma(gow_e8m0(byte_at(gow_mx_sf[si1 >> 2u], si1)), d1, acc1);
        }
    }
    let total0 = gow_mx_sg_butterfly32(acc0);
    let total1 = gow_mx_sg_butterfly32(acc1);
        #end
    if (lane == 0u && live0) {
        gow_mx_out_row(slot, e, row0, total0);
    }
    if (lane == 0u && live1) {
        gow_mx_out_row(slot, e, row1, total1);
    }
}

@group(0) @binding(17) var<storage, read> gow_mx_dncomb_wts: array<f32>;
@group(0) @binding(18) var<storage, read_write> gow_mx_dncomb_out: array<u32>;

fn gow_mx_dncomb_slots_ride_pad0_because_the_down_gemv_carries_no_swiglu_limit() -> u32 {
    return gow_mx_p.pad0;
}

const GOW_MX_SG_DNCOMB_SGSLOT_ROWS_PER_WG: u32 = 4u;

var<workgroup> gow_mx_dncomb_fold_two_row_pairs_by_four_expert_slots_by_two_rows: array<f32, 16>;

@compute @workgroup_size(256)
fn gow_gemv_mx_sg_xv4_r2_lut_dncomb_sgslot(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    gow_mx_xv4_byte_lut_every_e2m1_value_is_exact_in_f16[lid.x] =
        pack2x16float(vec2<f32>(gow_mx_e2m1(lid.x, 0u), gow_mx_e2m1(lid.x, 1u)));
    let slot = sgid >> 1u;
    let pair = sgid & 1u;
    let row0 = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_DNCOMB_SGSLOT_ROWS_PER_WG
        + pair * 2u;
    let row1 = row0 + 1u;
    let live0 = row0 < gow_mx_p.n_rows;
    let live1 = row1 < gow_mx_p.n_rows;
    let blocks = select(0u, gow_mx_p.k_blocks, live0);
    var e = slot;
    if (gow_mx_p.use_sel == 1u) {
        e = gow_mx_sel[slot];
    }
    let wbase0 = e * gow_mx_p.w_e_stride_v4 + select(0u, row0 * gow_mx_p.k_blocks, live0);
    let wbase1 = e * gow_mx_p.w_e_stride_v4 + select(0u, row1 * gow_mx_p.k_blocks, live1);
    let sfbase0 = e * gow_mx_p.sf_e_stride_bytes + select(0u, row0 * gow_mx_p.k_blocks, live0);
    let sfbase1 = e * gow_mx_p.sf_e_stride_bytes + select(0u, row1 * gow_mx_p.k_blocks, live1);
    let xqbase = slot * (gow_mx_p.x_slot_stride_words >> 2u);
    workgroupBarrier();

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + GOW_MX_SG_LANES) {
        let xq = xqbase + kb * 4u;
        let xa = gow_mx_xv4[xq];
        let xb = gow_mx_xv4[xq + 1u];
        let xc = gow_mx_xv4[xq + 2u];
        let xd = gow_mx_xv4[xq + 3u];
        #use b2
    if (lane == 0u) {
        let fb = (pair * 4u + slot) * 2u;
        gow_mx_dncomb_fold_two_row_pairs_by_four_expert_slots_by_two_rows[fb] =
            gow_mx_biased_row(e, row0, total0);
        gow_mx_dncomb_fold_two_row_pairs_by_four_expert_slots_by_two_rows[fb + 1u] =
            gow_mx_biased_row(e, row1, total1);
    }
    workgroupBarrier();

    if (lid.x < 2u) {
        let rp = lid.x;
        let r0 = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_DNCOMB_SGSLOT_ROWS_PER_WG
            + rp * 2u;
        if (r0 < gow_mx_p.n_rows) {
            var a0 = 0.0;
            var a1 = 0.0;
            let slots = gow_mx_dncomb_slots_ride_pad0_because_the_down_gemv_carries_no_swiglu_limit();
            for (var j = 0u; j < slots; j = j + 1u) {
                let fb = (rp * 4u + j) * 2u;
                let wt = gow_mx_dncomb_wts[j];
                a0 = fma(gow_mx_dncomb_fold_two_row_pairs_by_four_expert_slots_by_two_rows[fb],
                         wt, a0);
                a1 = fma(gow_mx_dncomb_fold_two_row_pairs_by_four_expert_slots_by_two_rows[fb + 1u],
                         wt, a1);
            }
            gow_mx_dncomb_out[r0 >> 1u] = bf16_pack(a0, a1);
        }
    }
}

const GOW_MX_SG_DNCOMB_WG64_ROWS_PER_WG: u32 = 4u;

const GOW_MX_SG_DNCOMB_WG64_THREADS_KEEP_ONE_WORKGROUP_PER_EXPERT_SLOT_OF_RESIDENCY: u32 = 64u;

@compute @workgroup_size(64)
fn gow_gemv_mx_sg_xv4_r2_lut_xs_dncomb_wg64(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    for (var t = lid.x; t < 256u;
         t = t + GOW_MX_SG_DNCOMB_WG64_THREADS_KEEP_ONE_WORKGROUP_PER_EXPERT_SLOT_OF_RESIDENCY) {
        gow_mx_xv4_byte_lut_every_e2m1_value_is_exact_in_f16[t] =
            pack2x16float(vec2<f32>(gow_mx_e2m1(t, 0u), gow_mx_e2m1(t, 1u)));
    }
    let xwords4 = gow_mx_p.k_blocks * 4u;
    let row0 = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_DNCOMB_WG64_ROWS_PER_WG
        + sgid * 2u;
    #def b7
    let row1 = row0 + 1u;
    let live0 = row0 < gow_mx_p.n_rows;
    let live1 = row1 < gow_mx_p.n_rows;
    let blocks = select(0u, gow_mx_p.k_blocks, live0);

    var a0 = 0.0;
    var a1 = 0.0;
    let slots = gow_mx_dncomb_slots_ride_pad0_because_the_down_gemv_carries_no_swiglu_limit();
    for (var slot = 0u; slot < slots; slot = slot + 1u) {
        workgroupBarrier();
        let xqbase = slot * (gow_mx_p.x_slot_stride_words >> 2u);
    #end
        for (var i = lid.x; i < xwords4;
             i = i + GOW_MX_SG_DNCOMB_WG64_THREADS_KEEP_ONE_WORKGROUP_PER_EXPERT_SLOT_OF_RESIDENCY) {
            #def b1
            gow_mx_xsh[i] = gow_mx_xv4[xqbase + i];
        }
        workgroupBarrier();

        var e = slot;
        if (gow_mx_p.use_sel == 1u) {
            e = gow_mx_sel[slot];
        }
        let wbase0 = e * gow_mx_p.w_e_stride_v4 + select(0u, row0 * gow_mx_p.k_blocks, live0);
        let wbase1 = e * gow_mx_p.w_e_stride_v4 + select(0u, row1 * gow_mx_p.k_blocks, live1);
        let sfbase0 = e * gow_mx_p.sf_e_stride_bytes + select(0u, row0 * gow_mx_p.k_blocks, live0);
        let sfbase1 = e * gow_mx_p.sf_e_stride_bytes + select(0u, row1 * gow_mx_p.k_blocks, live1);

        var acc0 = 0.0;
        var acc1 = 0.0;
        for (var kb = lane; kb < blocks; kb = kb + GOW_MX_SG_LANES) {
            let xq = kb * 4u;
            let xa = gow_mx_xsh[xq];
            let xb = gow_mx_xsh[xq + 1u];
            let xc = gow_mx_xsh[xq + 2u];
            let xd = gow_mx_xsh[xq + 3u];
            let wv0 = gow_mx_w[wbase0 + kb];
            var d0 = gow_mx_lut_dot8(wv0.x, xa.x, xa.y, xa.z, xa.w);
            d0 = d0 + gow_mx_lut_dot8(wv0.y, xb.x, xb.y, xb.z, xb.w);
            d0 = d0 + gow_mx_lut_dot8(wv0.z, xc.x, xc.y, xc.z, xc.w);
            d0 = d0 + gow_mx_lut_dot8(wv0.w, xd.x, xd.y, xd.z, xd.w);
            let si0 = sfbase0 + kb;
            acc0 = fma(gow_e8m0(byte_at(gow_mx_sf[si0 >> 2u], si0)), d0, acc0);
            if (live1) {
                let wv1 = gow_mx_w[wbase1 + kb];
                var d1 = gow_mx_lut_dot8(wv1.x, xa.x, xa.y, xa.z, xa.w);
                d1 = d1 + gow_mx_lut_dot8(wv1.y, xb.x, xb.y, xb.z, xb.w);
                d1 = d1 + gow_mx_lut_dot8(wv1.z, xc.x, xc.y, xc.z, xc.w);
                d1 = d1 + gow_mx_lut_dot8(wv1.w, xd.x, xd.y, xd.z, xd.w);
                let si1 = sfbase1 + kb;
                acc1 = fma(gow_e8m0(byte_at(gow_mx_sf[si1 >> 2u], si1)), d1, acc1);
            }
        }
        let total0 = gow_mx_sg_butterfly32(acc0);
        let total1 = gow_mx_sg_butterfly32(acc1);
        let wt = gow_mx_dncomb_wts[slot];
        a0 = fma(gow_mx_biased_row(e, row0, total0), wt, a0);
        a1 = fma(gow_mx_biased_row(e, row1, total1), wt, a1);
    }
    if (lane == 0u && live0) {
        gow_mx_dncomb_out[row0 >> 1u] = bf16_pack(a0, a1);
    }
}
            #end

@compute @workgroup_size(256)
fn gow_gemv_mx_sg_xv4_r2_lut_xs_dncomb(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    gow_mx_xv4_byte_lut_every_e2m1_value_is_exact_in_f16[lid.x] =
        pack2x16float(vec2<f32>(gow_mx_e2m1(lid.x, 0u), gow_mx_e2m1(lid.x, 1u)));
    let xwords4 = gow_mx_p.k_blocks * 4u;
    let row0 = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_R2_ROWS_PER_WG + sgid * 2u;
    #use b7
        for (var i = lid.x; i < xwords4; i = i + 256u) {
            #use b1

@compute @workgroup_size(256)
fn gow_gemv_mx_sg_xv4_r2_lut(
    #def b9
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    gow_mx_xv4_byte_lut_every_e2m1_value_is_exact_in_f16[lid.x] =
        pack2x16float(vec2<f32>(gow_mx_e2m1(lid.x, 0u), gow_mx_e2m1(lid.x, 1u)));
    workgroupBarrier();
    #end

    #def b8
    let slot = wid.z;
    #use b3
    let xqbase = slot * (gow_mx_p.x_slot_stride_words >> 2u);

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + GOW_MX_SG_LANES) {
        let xq = xqbase + kb * 4u;
        let xa = gow_mx_xv4[xq];
        let xb = gow_mx_xv4[xq + 1u];
        let xc = gow_mx_xv4[xq + 2u];
        let xd = gow_mx_xv4[xq + 3u];
    #end
        #use b2
    if (lane == 0u && live0) {
        gow_mx_out_row(slot, e, row0, total0);
    }
    if (lane == 0u && live1) {
        gow_mx_out_row(slot, e, row1, total1);
    }
}

fn gow_mx_out_row(slot: u32, e: u32, row: u32, total: f32) {
    var v = total;
    if (gow_mx_p.has_bias == 1u) {
        let bi = e * gow_mx_p.bias_e_stride + row;
        v = v + bf16_decode(u16_at(gow_mx_b[bi >> 1u], bi));
    }
    gow_mx_y[slot * gow_mx_p.y_slot_stride + row] = v;
}
