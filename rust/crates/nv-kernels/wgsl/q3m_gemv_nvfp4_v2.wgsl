
struct Q34Params {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    groups_x: u32,
    w_e_stride_vec2: u32,
    sf_e_stride_bytes: u32,
    x_slot_stride_vec2: u32,
    xsf_slot_stride_bytes: u32,
    y_slot_stride_words: u32,
    per_expert_alpha: u32,
    m_slots_sharing_expert_zero: u32,
};

@group(0) @binding(10) var<storage, read> q34_w2: array<vec2<u32>>;
@group(0) @binding(11) var<storage, read> q34_ws: array<u32>;
@group(0) @binding(12) var<storage, read> q34_x2: array<vec2<u32>>;
@group(0) @binding(13) var<storage, read> q34_xs: array<u32>;
@group(0) @binding(14) var<uniform> q34_p: Q34Params;
@group(0) @binding(15) var<storage, read_write> q34_y: array<u32>;
@group(0) @binding(16) var<storage, read> q34_sel: array<u32>;
@group(0) @binding(17) var<storage, read> q34_alphas: array<f32>;
@group(0) @binding(18) var<storage, read> q34_w4: array<vec4<u32>>;
@group(0) @binding(19) var<storage, read> q34_x4: array<vec4<u32>>;

var<workgroup> q34_pk_bits: array<u32, NV2_SGS>;

fn q34_alpha(e: u32) -> f32 {
    if (q34_p.per_expert_alpha == 1u) {
        return q34_alphas[e];
    }
    return q34_p.alpha;
}

fn q34_pk_word(lo_acc: f32, hi_acc: f32, row: u32, a: f32) -> u32 {
    let lo = bf16_encode(lo_acc * a) & 0xffffu;
    var hi = 0u;
    if (row + 1u < q34_p.n_rows) {
        hi = bf16_encode(hi_acc * a) & 0xffffu;
    }
    return lo | (hi << 16u);
}

@compute @workgroup_size(NV2_WG)
fn q3w_gemv_nvfp4_fmlut(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    let e = q34_sel[slot];
    let row0 = (wid.x + wid.y * q34_p.groups_x) * (NV2_SGS * NV2_MR) + sgid * NV2_MR;
    let pairs = q34_p.k_blocks >> 1u;
    let stride = q34_p.k_blocks >> 1u;
    let webase = e * (q34_p.w_e_stride_vec2 >> 1u);
    let sfbase = e * q34_p.sf_e_stride_bytes;
    let x4base = slot * (q34_p.x_slot_stride_vec2 >> 1u);
    let xsfbase = slot * q34_p.xsf_slot_stride_bytes;

    var live: array<u32, NV2_MR>;
    var wbase: array<u32, NV2_MR>;
    var srow: array<u32, NV2_MR>;
    var acc: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let r = row0 + m;
        let ok = r < q34_p.n_rows;
        live[m] = select(0u, 1u, ok);
        wbase[m] = select(0u, webase + r * stride, ok);
        srow[m] = select(0u, r, ok);
        acc[m] = 0.0;
    }

    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = q34_x4[x4base + p];
        let b0 = p << 1u;
        let xsi = xsfbase + b0;
        let xsw = q34_xs[xsi >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, xsi)) * 0.25;
        let xs1 = nv2_ue4m3(byte_at(xsw, xsi + 1u)) * 0.25;
        let xe0 = nv2_meven(xv.x) * xs0;
        let xo0 = nv2_modd(xv.x) * xs0;
        let xe1 = nv2_meven(xv.y) * xs0;
        let xo1 = nv2_modd(xv.y) * xs0;
        let xe2 = nv2_meven(xv.z) * xs1;
        let xo2 = nv2_modd(xv.z) * xs1;
        let xe3 = nv2_meven(xv.w) * xs1;
        let xo3 = nv2_modd(xv.w) * xs1;
        for (var m = 0u; m < NV2_MR; m = m + 1u) {
            let wv = q34_w4[wbase[m] + p];
            let wsi = sfbase + nvfp4_scale_byte_index(srow[m], b0, q34_p.k_tiles);
            let wsw = q34_ws[wsi >> 2u];
            let d0 = nv2_mdot8(wv.y, xe1, xo1, nv2_mdot8(wv.x, xe0, xo0, 0.0));
            let d1 = nv2_mdot8(wv.w, xe3, xo3, nv2_mdot8(wv.z, xe2, xo2, 0.0));
            acc[m] = fma(nv2_ue4m3(byte_at(wsw, wsi)), d0, acc[m]);
            acc[m] = fma(nv2_ue4m3(byte_at(wsw, wsi + 1u)), d1, acc[m]);
        }
    }

    var tot: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        tot[m] = nv2_bfly(acc[m]);
    }
    if (lane == 0u) {
        let a = q34_alpha(e);
        let ybase = slot * q34_p.y_slot_stride_words;
        for (var m = 0u; m < NV2_MR; m = m + 2u) {
            if (live[m] == 1u) {
                q34_y[ybase + ((row0 + m) >> 1u)] =
                    q34_pk_word(tot[m], tot[m + 1u], row0 + m, a);
            }
        }
    }
}

@compute @workgroup_size(NV2_WG)
fn q3w_gemv_nvfp4_mrow(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    let e = q34_sel[slot];
    let row0 = (wid.x + wid.y * q34_p.groups_x) * (NV2_SGS * NV2_MR) + sgid * NV2_MR;
    let pairs = q34_p.k_blocks >> 1u;
    let stride = q34_p.k_blocks >> 1u;
    let webase = e * (q34_p.w_e_stride_vec2 >> 1u);
    let sfbase = e * q34_p.sf_e_stride_bytes;
    let x4base = slot * (q34_p.x_slot_stride_vec2 >> 1u);
    let xsfbase = slot * q34_p.xsf_slot_stride_bytes;

    var live: array<u32, NV2_MR>;
    var wbase: array<u32, NV2_MR>;
    var srow: array<u32, NV2_MR>;
    var acc: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let r = row0 + m;
        let ok = r < q34_p.n_rows;
        live[m] = select(0u, 1u, ok);
        wbase[m] = select(0u, webase + r * stride, ok);
        srow[m] = select(0u, r, ok);
        acc[m] = 0.0;
    }

    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = q34_x4[x4base + p];
        let xm0 = nv2_i8map(xv.x);
        let xm1 = nv2_i8map(xv.x >> 4u);
        let xm2 = nv2_i8map(xv.y);
        let xm3 = nv2_i8map(xv.y >> 4u);
        let xm4 = nv2_i8map(xv.z);
        let xm5 = nv2_i8map(xv.z >> 4u);
        let xm6 = nv2_i8map(xv.w);
        let xm7 = nv2_i8map(xv.w >> 4u);
        let b0 = p << 1u;
        let xsi = xsfbase + b0;
        let xsw = q34_xs[xsi >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, xsi));
        let xs1 = nv2_ue4m3(byte_at(xsw, xsi + 1u));
        for (var m = 0u; m < NV2_MR; m = m + 1u) {
            let wv = q34_w4[wbase[m] + p];
            let wsi = sfbase + nvfp4_scale_byte_index(srow[m], b0, q34_p.k_tiles);
            let wsw = q34_ws[wsi >> 2u];
            let d0 = dot4I8Packed(nv2_i8map(wv.x), xm0)
                + dot4I8Packed(nv2_i8map(wv.x >> 4u), xm1)
                + dot4I8Packed(nv2_i8map(wv.y), xm2)
                + dot4I8Packed(nv2_i8map(wv.y >> 4u), xm3);
            let d1 = dot4I8Packed(nv2_i8map(wv.z), xm4)
                + dot4I8Packed(nv2_i8map(wv.z >> 4u), xm5)
                + dot4I8Packed(nv2_i8map(wv.w), xm6)
                + dot4I8Packed(nv2_i8map(wv.w >> 4u), xm7);
            acc[m] = fma(nv2_ue4m3(byte_at(wsw, wsi)) * xs0, f32(d0) * 0.25, acc[m]);
            acc[m] = fma(nv2_ue4m3(byte_at(wsw, wsi + 1u)) * xs1, f32(d1) * 0.25, acc[m]);
        }
    }

    var tot: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        tot[m] = nv2_bfly(acc[m]);
    }
    if (lane == 0u) {
        let a = q34_alpha(e);
        let ybase = slot * q34_p.y_slot_stride_words;
        for (var m = 0u; m < NV2_MR; m = m + 2u) {
            if (live[m] == 1u) {
                q34_y[ybase + ((row0 + m) >> 1u)] =
                    q34_pk_word(tot[m], tot[m + 1u], row0 + m, a);
            }
        }
    }
}

@compute @workgroup_size(NV2_WG)
fn q3w_gemv_nvfp4_mrow2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    let e = q34_sel[slot];
    let row0 = (wid.x + wid.y * q34_p.groups_x) * (NV2_SGS * 2u) + sgid * 2u;
    let row1 = row0 + 1u;
    let pairs = q34_p.k_blocks >> 1u;
    let stride = q34_p.k_blocks >> 1u;
    let webase = e * (q34_p.w_e_stride_vec2 >> 1u);
    let sfbase = e * q34_p.sf_e_stride_bytes;
    let x4base = slot * (q34_p.x_slot_stride_vec2 >> 1u);
    let xsfbase = slot * q34_p.xsf_slot_stride_bytes;
    let live0 = row0 < q34_p.n_rows;
    let live1 = row1 < q34_p.n_rows;
    let wb0 = select(0u, webase + row0 * stride, live0);
    let wb1 = select(0u, webase + row1 * stride, live1);
    let sr0 = select(0u, row0, live0);
    let sr1 = select(0u, row1, live1);

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = q34_x4[x4base + p];
        let xm0 = nv2_i8map(xv.x);
        let xm1 = nv2_i8map(xv.x >> 4u);
        let xm2 = nv2_i8map(xv.y);
        let xm3 = nv2_i8map(xv.y >> 4u);
        let xm4 = nv2_i8map(xv.z);
        let xm5 = nv2_i8map(xv.z >> 4u);
        let xm6 = nv2_i8map(xv.w);
        let xm7 = nv2_i8map(xv.w >> 4u);
        let b0 = p << 1u;
        let xsi = xsfbase + b0;
        let xsw = q34_xs[xsi >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, xsi));
        let xs1 = nv2_ue4m3(byte_at(xsw, xsi + 1u));

        let wv0 = q34_w4[wb0 + p];
        let wsi0 = sfbase + nvfp4_scale_byte_index(sr0, b0, q34_p.k_tiles);
        let wsw0 = q34_ws[wsi0 >> 2u];
        let a0 = dot4I8Packed(nv2_i8map(wv0.x), xm0)
            + dot4I8Packed(nv2_i8map(wv0.x >> 4u), xm1)
            + dot4I8Packed(nv2_i8map(wv0.y), xm2)
            + dot4I8Packed(nv2_i8map(wv0.y >> 4u), xm3);
        let a1 = dot4I8Packed(nv2_i8map(wv0.z), xm4)
            + dot4I8Packed(nv2_i8map(wv0.z >> 4u), xm5)
            + dot4I8Packed(nv2_i8map(wv0.w), xm6)
            + dot4I8Packed(nv2_i8map(wv0.w >> 4u), xm7);
        acc0 = fma(nv2_ue4m3(byte_at(wsw0, wsi0)) * xs0, f32(a0) * 0.25, acc0);
        acc0 = fma(nv2_ue4m3(byte_at(wsw0, wsi0 + 1u)) * xs1, f32(a1) * 0.25, acc0);

        let wv1 = q34_w4[wb1 + p];
        let wsi1 = sfbase + nvfp4_scale_byte_index(sr1, b0, q34_p.k_tiles);
        let wsw1 = q34_ws[wsi1 >> 2u];
        let b0d = dot4I8Packed(nv2_i8map(wv1.x), xm0)
            + dot4I8Packed(nv2_i8map(wv1.x >> 4u), xm1)
            + dot4I8Packed(nv2_i8map(wv1.y), xm2)
            + dot4I8Packed(nv2_i8map(wv1.y >> 4u), xm3);
        let b1d = dot4I8Packed(nv2_i8map(wv1.z), xm4)
            + dot4I8Packed(nv2_i8map(wv1.z >> 4u), xm5)
            + dot4I8Packed(nv2_i8map(wv1.w), xm6)
            + dot4I8Packed(nv2_i8map(wv1.w >> 4u), xm7);
        acc1 = fma(nv2_ue4m3(byte_at(wsw1, wsi1)) * xs0, f32(b0d) * 0.25, acc1);
        acc1 = fma(nv2_ue4m3(byte_at(wsw1, wsi1 + 1u)) * xs1, f32(b1d) * 0.25, acc1);
    }

    let t0 = nv2_bfly(acc0);
    let t1 = nv2_bfly(acc1);
    if (lane == 0u && live0) {
        q34_y[slot * q34_p.y_slot_stride_words + (row0 >> 1u)] =
            q34_pk_word(t0, t1, row0, q34_alpha(e));
    }
}

@group(0) @binding(21) var<storage, read> q34_wb4: array<vec4<u32>>;
@group(0) @binding(22) var<storage, read> q34_wbs: array<u32>;
@group(0) @binding(23) var<storage, read_write> q34_yb: array<u32>;

@compute @workgroup_size(NV2_WG)
fn q3w_gemv_nvfp4_mrow2_2w(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    let e = q34_sel[slot];
    let second = wid.y == 1u;
    let row0 = wid.x * (NV2_SGS * 2u) + sgid * 2u;
    let row1 = row0 + 1u;
    let pairs = q34_p.k_blocks >> 1u;
    let stride = q34_p.k_blocks >> 1u;
    let webase = e * (q34_p.w_e_stride_vec2 >> 1u);
    let sfbase = e * q34_p.sf_e_stride_bytes;
    let x4base = slot * (q34_p.x_slot_stride_vec2 >> 1u);
    let xsfbase = slot * q34_p.xsf_slot_stride_bytes;
    let live0 = row0 < q34_p.n_rows;
    let live1 = row1 < q34_p.n_rows;
    let wb0 = select(0u, webase + row0 * stride, live0);
    let wb1 = select(0u, webase + row1 * stride, live1);
    let sr0 = select(0u, row0, live0);
    let sr1 = select(0u, row1, live1);

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = q34_x4[x4base + p];
        let xm0 = nv2_i8map(xv.x);
        let xm1 = nv2_i8map(xv.x >> 4u);
        let xm2 = nv2_i8map(xv.y);
        let xm3 = nv2_i8map(xv.y >> 4u);
        let xm4 = nv2_i8map(xv.z);
        let xm5 = nv2_i8map(xv.z >> 4u);
        let xm6 = nv2_i8map(xv.w);
        let xm7 = nv2_i8map(xv.w >> 4u);
        let b0 = p << 1u;
        let xsi = xsfbase + b0;
        let xsw = q34_xs[xsi >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, xsi));
        let xs1 = nv2_ue4m3(byte_at(xsw, xsi + 1u));

        var wv0: vec4<u32>;
        var wsw0: u32;
        var wv1: vec4<u32>;
        var wsw1: u32;
        let wsi0 = sfbase + nvfp4_scale_byte_index(sr0, b0, q34_p.k_tiles);
        let wsi1 = sfbase + nvfp4_scale_byte_index(sr1, b0, q34_p.k_tiles);
        if (second) {
            wv0 = q34_wb4[wb0 + p];
            wsw0 = q34_wbs[wsi0 >> 2u];
            wv1 = q34_wb4[wb1 + p];
            wsw1 = q34_wbs[wsi1 >> 2u];
        } else {
            wv0 = q34_w4[wb0 + p];
            wsw0 = q34_ws[wsi0 >> 2u];
            wv1 = q34_w4[wb1 + p];
            wsw1 = q34_ws[wsi1 >> 2u];
        }
        let a0 = dot4I8Packed(nv2_i8map(wv0.x), xm0)
            + dot4I8Packed(nv2_i8map(wv0.x >> 4u), xm1)
            + dot4I8Packed(nv2_i8map(wv0.y), xm2)
            + dot4I8Packed(nv2_i8map(wv0.y >> 4u), xm3);
        let a1 = dot4I8Packed(nv2_i8map(wv0.z), xm4)
            + dot4I8Packed(nv2_i8map(wv0.z >> 4u), xm5)
            + dot4I8Packed(nv2_i8map(wv0.w), xm6)
            + dot4I8Packed(nv2_i8map(wv0.w >> 4u), xm7);
        acc0 = fma(nv2_ue4m3(byte_at(wsw0, wsi0)) * xs0, f32(a0) * 0.25, acc0);
        acc0 = fma(nv2_ue4m3(byte_at(wsw0, wsi0 + 1u)) * xs1, f32(a1) * 0.25, acc0);

        let b0d = dot4I8Packed(nv2_i8map(wv1.x), xm0)
            + dot4I8Packed(nv2_i8map(wv1.x >> 4u), xm1)
            + dot4I8Packed(nv2_i8map(wv1.y), xm2)
            + dot4I8Packed(nv2_i8map(wv1.y >> 4u), xm3);
        let b1d = dot4I8Packed(nv2_i8map(wv1.z), xm4)
            + dot4I8Packed(nv2_i8map(wv1.z >> 4u), xm5)
            + dot4I8Packed(nv2_i8map(wv1.w), xm6)
            + dot4I8Packed(nv2_i8map(wv1.w >> 4u), xm7);
        acc1 = fma(nv2_ue4m3(byte_at(wsw1, wsi1)) * xs0, f32(b0d) * 0.25, acc1);
        acc1 = fma(nv2_ue4m3(byte_at(wsw1, wsi1 + 1u)) * xs1, f32(b1d) * 0.25, acc1);
    }

    let t0 = nv2_bfly(acc0);
    let t1 = nv2_bfly(acc1);
    if (lane == 0u && live0) {
        let yi = slot * q34_p.y_slot_stride_words + (row0 >> 1u);
        let pk = q34_pk_word(t0, t1, row0, q34_alphas[u32(second)]);
        if (second) {
            q34_yb[yi] = pk;
        } else {
            q34_y[yi] = pk;
        }
    }
}

@compute @workgroup_size(NV2_WG)
fn q3w_gemv_nvfp4_fdec(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    let e = q34_sel[slot];
    let row = (wid.x + wid.y * q34_p.groups_x) * NV2_SGS + sgid;
    let live = row < q34_p.n_rows;
    let quads = select(0u, q34_p.k_blocks >> 2u, live);
    let w4b = select(0u, e * (q34_p.w_e_stride_vec2 >> 1u) + row * (q34_p.k_blocks >> 1u), live);
    let sfbase = e * q34_p.sf_e_stride_bytes;
    let srow = select(0u, row, live);
    let x4base = slot * (q34_p.x_slot_stride_vec2 >> 1u);
    let xsfw = (slot * q34_p.xsf_slot_stride_bytes) >> 2u;

    var acc = 0.0;
    for (var q = lane; q < quads; q = q + NV2_LANES) {
        let wsi = sfbase + nvfp4_scale_byte_index(srow, q << 2u, q34_p.k_tiles);
        let wsw = q34_ws[wsi >> 2u];
        let xsw = q34_xs[xsfw + q];
        let wa = q34_w4[w4b + 2u * q];
        let wc = q34_w4[w4b + 2u * q + 1u];
        let xa = q34_x4[x4base + 2u * q];
        let xc = q34_x4[x4base + 2u * q + 1u];
        acc = fma(nv2_ue4m3(byte_at(wsw, wsi)) * nv2_ue4m3(byte_at(xsw, 0u)),
            nv2_fblock(wa.x, wa.y, xa.x, xa.y), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, wsi + 1u)) * nv2_ue4m3(byte_at(xsw, 1u)),
            nv2_fblock(wa.z, wa.w, xa.z, xa.w), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, wsi + 2u)) * nv2_ue4m3(byte_at(xsw, 2u)),
            nv2_fblock(wc.x, wc.y, xc.x, xc.y), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, wsi + 3u)) * nv2_ue4m3(byte_at(xsw, 3u)),
            nv2_fblock(wc.z, wc.w, xc.z, xc.w), acc);
    }

    let total = nv2_bfly(acc);
    if (lane == 0u) {
        q34_pk_bits[sgid] = bf16_encode(total * q34_alpha(e)) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = q34_pk_bits[sgid];
        if (row + 1u < q34_p.n_rows) {
            word = word | (q34_pk_bits[sgid + 1u] << 16u);
        }
        q34_y[slot * q34_p.y_slot_stride_words + (row >> 1u)] = word;
    }
}

const Q34_SLOTSHARED_MAX_SLOTS: u32 = 16u;

@group(0) @binding(20) var<storage, read_write> q34_xm: array<vec4<u32>>;

@compute @workgroup_size(256)
fn q3w_i8map_x_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let kb = gid.x;
    let s = gid.y;
    if (kb >= q34_p.k_blocks || s >= q34_p.m_slots_sharing_expert_zero) {
        return;
    }
    let xv = q34_x2[s * q34_p.x_slot_stride_vec2 + kb];
    q34_xm[s * q34_p.k_blocks + kb] = vec4<u32>(
        nv2_i8map(xv.x),
        nv2_i8map(xv.x >> 4u),
        nv2_i8map(xv.y),
        nv2_i8map(xv.y >> 4u)
    );
}

@compute @workgroup_size(NV2_WG)
fn q3w_gemv_nvfp4_slotshared(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let e = q34_sel[0u];
    let row = (wid.x + wid.y * q34_p.groups_x) * NV2_SGS + sgid;
    let live = row < q34_p.n_rows;
    let blocks = select(0u, q34_p.k_blocks, live);
    let wb = select(0u, e * q34_p.w_e_stride_vec2 + row * q34_p.k_blocks, live);
    let sfbase = e * q34_p.sf_e_stride_bytes;
    let srow = select(0u, row, live);
    let n_slots = min(q34_p.m_slots_sharing_expert_zero, Q34_SLOTSHARED_MAX_SLOTS);

    var acc: array<f32, Q34_SLOTSHARED_MAX_SLOTS>;
    for (var s = 0u; s < Q34_SLOTSHARED_MAX_SLOTS; s = s + 1u) {
        acc[s] = 0.0;
    }
    for (var kb = lane; kb < blocks; kb = kb + NV2_LANES) {
        let wsi = sfbase + nvfp4_scale_byte_index(srow, kb, q34_p.k_tiles);
        let wsv = nv2_ue4m3(byte_at(q34_ws[wsi >> 2u], wsi));
        let wv = q34_w2[wb + kb];
        let wm0 = nv2_i8map(wv.x);
        let wm1 = nv2_i8map(wv.x >> 4u);
        let wm2 = nv2_i8map(wv.y);
        let wm3 = nv2_i8map(wv.y >> 4u);
        for (var s = 0u; s < n_slots; s = s + 1u) {
            let xsi = s * q34_p.xsf_slot_stride_bytes + kb;
            let bs = wsv * nv2_ue4m3(byte_at(q34_xs[xsi >> 2u], xsi));
            let xm = q34_xm[s * q34_p.k_blocks + kb];
            let d = dot4I8Packed(wm0, xm.x)
                + dot4I8Packed(wm1, xm.y)
                + dot4I8Packed(wm2, xm.z)
                + dot4I8Packed(wm3, xm.w);
            acc[s] = fma(bs, f32(d) * 0.25, acc[s]);
        }
    }

    let a = q34_alpha(e);
    for (var s = 0u; s < n_slots; s = s + 1u) {
        let total = nv2_bfly(acc[s]);
        if (lane == 0u) {
            q34_pk_bits[sgid] = bf16_encode(total * a) & 0xffffu;
        }
        workgroupBarrier();
        if ((sgid & 1u) == 0u && lane == 0u && live) {
            var word = q34_pk_bits[sgid];
            if (row + 1u < q34_p.n_rows) {
                word = word | (q34_pk_bits[sgid + 1u] << 16u);
            }
            q34_y[s * q34_p.y_slot_stride_words + (row >> 1u)] = word;
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(NV2_WG)
fn q3w_gemv_nvfp4_warp(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let slot = wid.z;
    let e = q34_sel[slot];
    let row = (wid.x + wid.y * q34_p.groups_x) * NV2_SGS + sgid;
    let live = row < q34_p.n_rows;
    let blocks = select(0u, q34_p.k_blocks, live);
    let wb = select(0u, e * q34_p.w_e_stride_vec2 + row * q34_p.k_blocks, live);
    let sfbase = e * q34_p.sf_e_stride_bytes;
    let srow = select(0u, row, live);
    let x2base = slot * q34_p.x_slot_stride_vec2;
    let xsfbase = slot * q34_p.xsf_slot_stride_bytes;

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + NV2_LANES) {
        let wsi = sfbase + nvfp4_scale_byte_index(srow, kb, q34_p.k_tiles);
        let xsi = xsfbase + kb;
        let bs = nv2_ue4m3(byte_at(q34_ws[wsi >> 2u], wsi))
            * nv2_ue4m3(byte_at(q34_xs[xsi >> 2u], xsi));
        let wv = q34_w2[wb + kb];
        let xv = q34_x2[x2base + kb];
        acc = fma(bs, nv2_iblock(wv.x, wv.y, xv.x, xv.y), acc);
    }

    let total = nv2_bfly(acc);
    if (lane == 0u) {
        q34_pk_bits[sgid] = bf16_encode(total * q34_alpha(e)) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = q34_pk_bits[sgid];
        if (row + 1u < q34_p.n_rows) {
            word = word | (q34_pk_bits[sgid + 1u] << 16u);
        }
        q34_y[slot * q34_p.y_slot_stride_words + (row >> 1u)] = word;
    }
}
