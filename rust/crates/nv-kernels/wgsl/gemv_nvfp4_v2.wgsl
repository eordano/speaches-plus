struct Nv2Params {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    w_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> nv2_w2: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> nv2_ws: array<u32>;
@group(0) @binding(2) var<storage, read> nv2_x2: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> nv2_xs: array<u32>;
@group(0) @binding(4) var<uniform> nv2_p: Nv2Params;
@group(0) @binding(5) var<storage, read_write> nv2_y: array<u32>;
@group(0) @binding(6) var<storage, read> nv2_w4: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> nv2_x4: array<vec4<u32>>;

const NV2_WG: u32 = 256u;
const NV2_MR: u32 = 4u;
const NV2_SGS: u32 = NV2_WG / 32u;
const NV2_LANES: u32 = 32u;

const NV2_UE4M3_SUBNORMAL_STEP: f32 = 0.001953125;

fn nv2_ue4m3(bits: u32) -> f32 {
    let b = bits & 127u;
    return select(
        bitcast<f32>((b << 20u) + 0x3c000000u),
        f32(b) * NV2_UE4M3_SUBNORMAL_STEP,
        b < 8u
    );
}

const NV2_DECODE_BEGIN: u32 = 0u;

fn nv2_i8map(s: u32) -> u32 {
    let k = s & 0x07070707u;
    let hm = ((k >> 2u) & 0x01010101u) * 255u;
    let e7 = (k & (k >> 1u) & (k >> 2u)) & 0x01010101u;
    let m = k + ((k & 0x03030303u) & hm) + (e7 << 1u);
    let sb = (s & ((k + 0x07070707u) & 0x08080808u)) >> 3u;
    return (m ^ (sb * 255u)) + sb;
}

fn nv2_dec4(n: vec4<u32>) -> vec4<f32> {
    let k = n & vec4<u32>(7u);
    let sgn = (n & vec4<u32>(8u)) << vec4<u32>(28u);
    let big = (k + vec4<u32>(252u)) << vec4<u32>(22u);
    let sml = (k & vec4<u32>(1u)) * vec4<u32>(0x3f000000u);
    return bitcast<vec4<f32>>(sgn | select(big, sml, k < vec4<u32>(2u)));
}

fn nv2_mdec4(n: vec4<u32>) -> vec4<f32> {
    let sh = (n & vec4<u32>(7u)) << vec4<u32>(2u);
    let m = (vec4<u32>(0xc8643210u) >> sh) & vec4<u32>(15u);
    let sgn = (n & vec4<u32>(8u)) << vec4<u32>(28u);
    return bitcast<vec4<f32>>(bitcast<vec4<u32>>(vec4<f32>(m)) | sgn);
}

const NV2_DECODE_END: u32 = 0u;

fn nv2_idot(w: u32, x: u32) -> i32 {
    return dot4I8Packed(nv2_i8map(w), nv2_i8map(x))
        + dot4I8Packed(nv2_i8map(w >> 4u), nv2_i8map(x >> 4u));
}

fn nv2_iblock(w0: u32, w1: u32, x0: u32, x1: u32) -> f32 {
    return f32(nv2_idot(w0, x0) + nv2_idot(w1, x1)) * 0.25;
}

fn nv2_even(w: u32) -> vec4<f32> {
    return nv2_dec4(unpack4xU8(w & 0x0f0f0f0fu));
}

fn nv2_odd(w: u32) -> vec4<f32> {
    return nv2_dec4(unpack4xU8((w >> 4u) & 0x0f0f0f0fu));
}

fn nv2_fdot8(w: u32, x: u32, acc_in: f32) -> f32 {
    let we = nv2_even(w);
    let wo = nv2_odd(w);
    let xe = nv2_even(x);
    let xo = nv2_odd(x);
    var s = acc_in;
    s = fma(we.x, xe.x, s);
    s = fma(wo.x, xo.x, s);
    s = fma(we.y, xe.y, s);
    s = fma(wo.y, xo.y, s);
    s = fma(we.z, xe.z, s);
    s = fma(wo.z, xo.z, s);
    s = fma(we.w, xe.w, s);
    s = fma(wo.w, xo.w, s);
    return s;
}

fn nv2_fblock(w0: u32, w1: u32, x0: u32, x1: u32) -> f32 {
    return nv2_fdot8(w1, x1, nv2_fdot8(w0, x0, 0.0));
}

fn nv2_meven(w: u32) -> vec4<f32> {
    return nv2_mdec4(unpack4xU8(w & 0x0f0f0f0fu));
}

fn nv2_modd(w: u32) -> vec4<f32> {
    return nv2_mdec4(unpack4xU8((w >> 4u) & 0x0f0f0f0fu));
}

fn nv2_mdot8(w: u32, xe: vec4<f32>, xo: vec4<f32>, acc_in: f32) -> f32 {
    let we = nv2_meven(w);
    let wo = nv2_modd(w);
    var s = acc_in;
    s = fma(we.x, xe.x, s);
    s = fma(wo.x, xo.x, s);
    s = fma(we.y, xe.y, s);
    s = fma(wo.y, xo.y, s);
    s = fma(we.z, xe.z, s);
    s = fma(wo.z, xo.z, s);
    s = fma(we.w, xe.w, s);
    s = fma(wo.w, xo.w, s);
    return s;
}

fn nv2_pdot8(w: u32, xe: vec4<f32>, xo: vec4<f32>, acc_in: f32) -> f32 {
    let we = nv2_even(w);
    let wo = nv2_odd(w);
    var s = acc_in;
    s = fma(we.x, xe.x, s);
    s = fma(wo.x, xo.x, s);
    s = fma(we.y, xe.y, s);
    s = fma(wo.y, xo.y, s);
    s = fma(we.z, xe.z, s);
    s = fma(wo.z, xo.z, s);
    s = fma(we.w, xe.w, s);
    s = fma(wo.w, xo.w, s);
    return s;
}

fn nv2_bfly(acc: f32) -> f32 {
    var a = acc;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

const NV2_SECTION_WARP: u32 = 1u;

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_warp(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * nv2_p.groups_x) * NV2_SGS + sgid;
    let live = row < nv2_p.n_rows;
    let blocks = select(0u, nv2_p.k_blocks, live);
    let wb = select(0u, row * nv2_p.k_blocks, live);
    let srow = select(0u, row, live);

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + NV2_LANES) {
        let wsi = nvfp4_scale_byte_index(srow, kb, nv2_p.k_tiles);
        let bs = nv2_ue4m3(byte_at(nv2_ws[wsi >> 2u], wsi))
            * nv2_ue4m3(byte_at(nv2_xs[kb >> 2u], kb));
        let wv = nv2_w2[wb + kb];
        let xv = nv2_x2[kb];
        acc = fma(bs, nv2_iblock(wv.x, wv.y, xv.x, xv.y), acc);
    }

    let total = nv2_bfly(acc);
    if (lane == 0u && live) {
        nv2_y[row] = bf16_encode(total * nv2_p.alpha);
    }
}

const NV2_SECTION_WARPQ: u32 = 2u;

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_warpq(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * nv2_p.groups_x) * NV2_SGS + sgid;
    let live = row < nv2_p.n_rows;
    let quads = select(0u, nv2_p.k_blocks >> 2u, live);
    let w4b = select(0u, row * (nv2_p.k_blocks >> 1u), live);
    let srow = select(0u, row, live);

    var acc = 0.0;
    for (var q = lane; q < quads; q = q + NV2_LANES) {
        let wsi = nvfp4_scale_byte_index(srow, q << 2u, nv2_p.k_tiles);
        let wsw = nv2_ws[wsi >> 2u];
        let xsw = nv2_xs[q];
        let wa = nv2_w4[w4b + 2u * q];
        let wc = nv2_w4[w4b + 2u * q + 1u];
        let xa = nv2_x4[2u * q];
        let xc = nv2_x4[2u * q + 1u];
        acc = fma(nv2_ue4m3(byte_at(wsw, 0u)) * nv2_ue4m3(byte_at(xsw, 0u)),
            nv2_iblock(wa.x, wa.y, xa.x, xa.y), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, 1u)) * nv2_ue4m3(byte_at(xsw, 1u)),
            nv2_iblock(wa.z, wa.w, xa.z, xa.w), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, 2u)) * nv2_ue4m3(byte_at(xsw, 2u)),
            nv2_iblock(wc.x, wc.y, xc.x, xc.y), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, 3u)) * nv2_ue4m3(byte_at(xsw, 3u)),
            nv2_iblock(wc.z, wc.w, xc.z, xc.w), acc);
    }

    let total = nv2_bfly(acc);
    if (lane == 0u && live) {
        nv2_y[row] = bf16_encode(total * nv2_p.alpha);
    }
}

const NV2_SECTION_FDEC: u32 = 3u;

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_fdec(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * nv2_p.groups_x) * NV2_SGS + sgid;
    let live = row < nv2_p.n_rows;
    let quads = select(0u, nv2_p.k_blocks >> 2u, live);
    let w4b = select(0u, row * (nv2_p.k_blocks >> 1u), live);
    let srow = select(0u, row, live);

    var acc = 0.0;
    for (var q = lane; q < quads; q = q + NV2_LANES) {
        let wsi = nvfp4_scale_byte_index(srow, q << 2u, nv2_p.k_tiles);
        let wsw = nv2_ws[wsi >> 2u];
        let xsw = nv2_xs[q];
        let wa = nv2_w4[w4b + 2u * q];
        let wc = nv2_w4[w4b + 2u * q + 1u];
        let xa = nv2_x4[2u * q];
        let xc = nv2_x4[2u * q + 1u];
        acc = fma(nv2_ue4m3(byte_at(wsw, 0u)) * nv2_ue4m3(byte_at(xsw, 0u)),
            nv2_fblock(wa.x, wa.y, xa.x, xa.y), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, 1u)) * nv2_ue4m3(byte_at(xsw, 1u)),
            nv2_fblock(wa.z, wa.w, xa.z, xa.w), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, 2u)) * nv2_ue4m3(byte_at(xsw, 2u)),
            nv2_fblock(wc.x, wc.y, xc.x, xc.y), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, 3u)) * nv2_ue4m3(byte_at(xsw, 3u)),
            nv2_fblock(wc.z, wc.w, xc.z, xc.w), acc);
    }

    let total = nv2_bfly(acc);
    if (lane == 0u && live) {
        nv2_y[row] = bf16_encode(total * nv2_p.alpha);
    }
}

const NV2_SECTION_MROW: u32 = 4u;

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_mrow(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row0 = (wid.x + wid.y * nv2_p.groups_x) * (NV2_SGS * NV2_MR) + sgid * NV2_MR;
    let pairs = nv2_p.k_blocks >> 1u;
    let stride = nv2_p.k_blocks >> 1u;

    var live: array<u32, NV2_MR>;
    var wbase: array<u32, NV2_MR>;
    var srow: array<u32, NV2_MR>;
    var acc: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let r = row0 + m;
        let ok = r < nv2_p.n_rows;
        live[m] = select(0u, 1u, ok);
        wbase[m] = select(0u, r * stride, ok);
        srow[m] = select(0u, r, ok);
        acc[m] = 0.0;
    }

    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = nv2_x4[p];
        let xm0 = nv2_i8map(xv.x);
        let xm1 = nv2_i8map(xv.x >> 4u);
        let xm2 = nv2_i8map(xv.y);
        let xm3 = nv2_i8map(xv.y >> 4u);
        let xm4 = nv2_i8map(xv.z);
        let xm5 = nv2_i8map(xv.z >> 4u);
        let xm6 = nv2_i8map(xv.w);
        let xm7 = nv2_i8map(xv.w >> 4u);
        let b0 = p << 1u;
        let xsw = nv2_xs[b0 >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, b0));
        let xs1 = nv2_ue4m3(byte_at(xsw, b0 + 1u));
        for (var m = 0u; m < NV2_MR; m = m + 1u) {
            let wv = nv2_w4[wbase[m] + p];
            let wsi = nvfp4_scale_byte_index(srow[m], b0, nv2_p.k_tiles);
            let wsw = nv2_ws[wsi >> 2u];
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

    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let total = nv2_bfly(acc[m]);
        if (lane == 0u && live[m] == 1u) {
            nv2_y[row0 + m] = bf16_encode(total * nv2_p.alpha);
        }
    }
}

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_mrow_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row0 = (wid.x + wid.y * nv2_p.groups_x) * (NV2_SGS * NV2_MR) + sgid * NV2_MR;
    let pairs = nv2_p.k_blocks >> 1u;
    let stride = nv2_p.k_blocks >> 1u;

    var live: array<u32, NV2_MR>;
    var wbase: array<u32, NV2_MR>;
    var srow: array<u32, NV2_MR>;
    var acc: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let r = row0 + m;
        let ok = r < nv2_p.n_rows;
        live[m] = select(0u, 1u, ok);
        wbase[m] = select(0u, r * stride, ok);
        srow[m] = select(0u, r, ok);
        acc[m] = 0.0;
    }

    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = nv2_x4[p];
        let xm0 = nv2_i8map(xv.x);
        let xm1 = nv2_i8map(xv.x >> 4u);
        let xm2 = nv2_i8map(xv.y);
        let xm3 = nv2_i8map(xv.y >> 4u);
        let xm4 = nv2_i8map(xv.z);
        let xm5 = nv2_i8map(xv.z >> 4u);
        let xm6 = nv2_i8map(xv.w);
        let xm7 = nv2_i8map(xv.w >> 4u);
        let b0 = p << 1u;
        let xsw = nv2_xs[b0 >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, b0));
        let xs1 = nv2_ue4m3(byte_at(xsw, b0 + 1u));
        for (var m = 0u; m < NV2_MR; m = m + 1u) {
            let wv = nv2_w4[wbase[m] + p];
            let wsi = nvfp4_scale_byte_index(srow[m], b0, nv2_p.k_tiles);
            let wsw = nv2_ws[wsi >> 2u];
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
        for (var m = 0u; m < NV2_MR; m = m + 2u) {
            if (live[m] == 1u) {
                nv2_y[(row0 + m) >> 1u] = nv2_pk_word(tot[m], tot[m + 1u], row0 + m);
            }
        }
    }
}

const NV2_SECTION_FMROW: u32 = 5u;

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_fmrow(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row0 = (wid.x + wid.y * nv2_p.groups_x) * (NV2_SGS * NV2_MR) + sgid * NV2_MR;
    let pairs = nv2_p.k_blocks >> 1u;
    let stride = nv2_p.k_blocks >> 1u;

    var live: array<u32, NV2_MR>;
    var wbase: array<u32, NV2_MR>;
    var srow: array<u32, NV2_MR>;
    var acc: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let r = row0 + m;
        let ok = r < nv2_p.n_rows;
        live[m] = select(0u, 1u, ok);
        wbase[m] = select(0u, r * stride, ok);
        srow[m] = select(0u, r, ok);
        acc[m] = 0.0;
    }

    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = nv2_x4[p];
        let b0 = p << 1u;
        let xsw = nv2_xs[b0 >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, b0));
        let xs1 = nv2_ue4m3(byte_at(xsw, b0 + 1u));
        let xe0 = nv2_even(xv.x) * xs0;
        let xo0 = nv2_odd(xv.x) * xs0;
        let xe1 = nv2_even(xv.y) * xs0;
        let xo1 = nv2_odd(xv.y) * xs0;
        let xe2 = nv2_even(xv.z) * xs1;
        let xo2 = nv2_odd(xv.z) * xs1;
        let xe3 = nv2_even(xv.w) * xs1;
        let xo3 = nv2_odd(xv.w) * xs1;
        for (var m = 0u; m < NV2_MR; m = m + 1u) {
            let wv = nv2_w4[wbase[m] + p];
            let wsi = nvfp4_scale_byte_index(srow[m], b0, nv2_p.k_tiles);
            let wsw = nv2_ws[wsi >> 2u];
            let d0 = nv2_pdot8(wv.y, xe1, xo1, nv2_pdot8(wv.x, xe0, xo0, 0.0));
            let d1 = nv2_pdot8(wv.w, xe3, xo3, nv2_pdot8(wv.z, xe2, xo2, 0.0));
            acc[m] = fma(nv2_ue4m3(byte_at(wsw, wsi)), d0, acc[m]);
            acc[m] = fma(nv2_ue4m3(byte_at(wsw, wsi + 1u)), d1, acc[m]);
        }
    }

    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let total = nv2_bfly(acc[m]);
        if (lane == 0u && live[m] == 1u) {
            nv2_y[row0 + m] = bf16_encode(total * nv2_p.alpha);
        }
    }
}

const NV2_SECTION_FMLUT: u32 = 6u;

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_fmlut(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row0 = (wid.x + wid.y * nv2_p.groups_x) * (NV2_SGS * NV2_MR) + sgid * NV2_MR;
    let pairs = nv2_p.k_blocks >> 1u;
    let stride = nv2_p.k_blocks >> 1u;

    var live: array<u32, NV2_MR>;
    var wbase: array<u32, NV2_MR>;
    var srow: array<u32, NV2_MR>;
    var acc: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let r = row0 + m;
        let ok = r < nv2_p.n_rows;
        live[m] = select(0u, 1u, ok);
        wbase[m] = select(0u, r * stride, ok);
        srow[m] = select(0u, r, ok);
        acc[m] = 0.0;
    }

    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = nv2_x4[p];
        let b0 = p << 1u;
        let xsw = nv2_xs[b0 >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, b0)) * 0.25;
        let xs1 = nv2_ue4m3(byte_at(xsw, b0 + 1u)) * 0.25;
        let xe0 = nv2_meven(xv.x) * xs0;
        let xo0 = nv2_modd(xv.x) * xs0;
        let xe1 = nv2_meven(xv.y) * xs0;
        let xo1 = nv2_modd(xv.y) * xs0;
        let xe2 = nv2_meven(xv.z) * xs1;
        let xo2 = nv2_modd(xv.z) * xs1;
        let xe3 = nv2_meven(xv.w) * xs1;
        let xo3 = nv2_modd(xv.w) * xs1;
        for (var m = 0u; m < NV2_MR; m = m + 1u) {
            let wv = nv2_w4[wbase[m] + p];
            let wsi = nvfp4_scale_byte_index(srow[m], b0, nv2_p.k_tiles);
            let wsw = nv2_ws[wsi >> 2u];
            let d0 = nv2_mdot8(wv.y, xe1, xo1, nv2_mdot8(wv.x, xe0, xo0, 0.0));
            let d1 = nv2_mdot8(wv.w, xe3, xo3, nv2_mdot8(wv.z, xe2, xo2, 0.0));
            acc[m] = fma(nv2_ue4m3(byte_at(wsw, wsi)), d0, acc[m]);
            acc[m] = fma(nv2_ue4m3(byte_at(wsw, wsi + 1u)), d1, acc[m]);
        }
    }

    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let total = nv2_bfly(acc[m]);
        if (lane == 0u && live[m] == 1u) {
            nv2_y[row0 + m] = bf16_encode(total * nv2_p.alpha);
        }
    }
}

const NV2_SECTION_PK: u32 = 7u;

var<workgroup> nv2_pk_bits: array<u32, NV2_SGS>;

fn nv2_pk_word(lo_acc: f32, hi_acc: f32, row: u32) -> u32 {
    let lo = bf16_encode(lo_acc * nv2_p.alpha) & 0xffffu;
    var hi = 0u;
    if (row + 1u < nv2_p.n_rows) {
        hi = bf16_encode(hi_acc * nv2_p.alpha) & 0xffffu;
    }
    return lo | (hi << 16u);
}

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_fmlut_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row0 = (wid.x + wid.y * nv2_p.groups_x) * (NV2_SGS * NV2_MR) + sgid * NV2_MR;
    let pairs = nv2_p.k_blocks >> 1u;
    let stride = nv2_p.k_blocks >> 1u;

    var live: array<u32, NV2_MR>;
    var wbase: array<u32, NV2_MR>;
    var srow: array<u32, NV2_MR>;
    var acc: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let r = row0 + m;
        let ok = r < nv2_p.n_rows;
        live[m] = select(0u, 1u, ok);
        wbase[m] = select(0u, r * stride, ok);
        srow[m] = select(0u, r, ok);
        acc[m] = 0.0;
    }

    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = nv2_x4[p];
        let b0 = p << 1u;
        let xsw = nv2_xs[b0 >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, b0)) * 0.25;
        let xs1 = nv2_ue4m3(byte_at(xsw, b0 + 1u)) * 0.25;
        let xe0 = nv2_meven(xv.x) * xs0;
        let xo0 = nv2_modd(xv.x) * xs0;
        let xe1 = nv2_meven(xv.y) * xs0;
        let xo1 = nv2_modd(xv.y) * xs0;
        let xe2 = nv2_meven(xv.z) * xs1;
        let xo2 = nv2_modd(xv.z) * xs1;
        let xe3 = nv2_meven(xv.w) * xs1;
        let xo3 = nv2_modd(xv.w) * xs1;
        for (var m = 0u; m < NV2_MR; m = m + 1u) {
            let wv = nv2_w4[wbase[m] + p];
            let wsi = nvfp4_scale_byte_index(srow[m], b0, nv2_p.k_tiles);
            let wsw = nv2_ws[wsi >> 2u];
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
        for (var m = 0u; m < NV2_MR; m = m + 2u) {
            if (live[m] == 1u) {
                nv2_y[(row0 + m) >> 1u] = nv2_pk_word(tot[m], tot[m + 1u], row0 + m);
            }
        }
    }
}

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_fdec_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * nv2_p.groups_x) * NV2_SGS + sgid;
    let live = row < nv2_p.n_rows;
    let quads = select(0u, nv2_p.k_blocks >> 2u, live);
    let w4b = select(0u, row * (nv2_p.k_blocks >> 1u), live);
    let srow = select(0u, row, live);

    var acc = 0.0;
    for (var q = lane; q < quads; q = q + NV2_LANES) {
        let wsi = nvfp4_scale_byte_index(srow, q << 2u, nv2_p.k_tiles);
        let wsw = nv2_ws[wsi >> 2u];
        let xsw = nv2_xs[q];
        let wa = nv2_w4[w4b + 2u * q];
        let wc = nv2_w4[w4b + 2u * q + 1u];
        let xa = nv2_x4[2u * q];
        let xc = nv2_x4[2u * q + 1u];
        acc = fma(nv2_ue4m3(byte_at(wsw, 0u)) * nv2_ue4m3(byte_at(xsw, 0u)),
            nv2_fblock(wa.x, wa.y, xa.x, xa.y), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, 1u)) * nv2_ue4m3(byte_at(xsw, 1u)),
            nv2_fblock(wa.z, wa.w, xa.z, xa.w), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, 2u)) * nv2_ue4m3(byte_at(xsw, 2u)),
            nv2_fblock(wc.x, wc.y, xc.x, xc.y), acc);
        acc = fma(nv2_ue4m3(byte_at(wsw, 3u)) * nv2_ue4m3(byte_at(xsw, 3u)),
            nv2_fblock(wc.z, wc.w, xc.z, xc.w), acc);
    }

    let total = nv2_bfly(acc);
    if (lane == 0u) {
        nv2_pk_bits[sgid] = bf16_encode(total * nv2_p.alpha) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = nv2_pk_bits[sgid];
        if (row + 1u < nv2_p.n_rows) {
            word = word | (nv2_pk_bits[sgid + 1u] << 16u);
        }
        nv2_y[row >> 1u] = word;
    }
}

const NV2_SECTION_MROW2: u32 = 8u;

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_mrow2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row0 = (wid.x + wid.y * nv2_p.groups_x) * (NV2_SGS * 2u) + sgid * 2u;
    let row1 = row0 + 1u;
    let pairs = nv2_p.k_blocks >> 1u;
    let stride = nv2_p.k_blocks >> 1u;
    let live0 = row0 < nv2_p.n_rows;
    let live1 = row1 < nv2_p.n_rows;
    let wb0 = select(0u, row0 * stride, live0);
    let wb1 = select(0u, row1 * stride, live1);
    let sr0 = select(0u, row0, live0);
    let sr1 = select(0u, row1, live1);

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = nv2_x4[p];
        let xm0 = nv2_i8map(xv.x);
        let xm1 = nv2_i8map(xv.x >> 4u);
        let xm2 = nv2_i8map(xv.y);
        let xm3 = nv2_i8map(xv.y >> 4u);
        let xm4 = nv2_i8map(xv.z);
        let xm5 = nv2_i8map(xv.z >> 4u);
        let xm6 = nv2_i8map(xv.w);
        let xm7 = nv2_i8map(xv.w >> 4u);
        let b0 = p << 1u;
        let xsw = nv2_xs[b0 >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, b0));
        let xs1 = nv2_ue4m3(byte_at(xsw, b0 + 1u));

        let wv0 = nv2_w4[wb0 + p];
        let wsi0 = nvfp4_scale_byte_index(sr0, b0, nv2_p.k_tiles);
        let wsw0 = nv2_ws[wsi0 >> 2u];
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

        let wv1 = nv2_w4[wb1 + p];
        let wsi1 = nvfp4_scale_byte_index(sr1, b0, nv2_p.k_tiles);
        let wsw1 = nv2_ws[wsi1 >> 2u];
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
        nv2_y[row0] = bf16_encode(t0 * nv2_p.alpha);
    }
    if (lane == 0u && live1) {
        nv2_y[row1] = bf16_encode(t1 * nv2_p.alpha);
    }
}

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_mrow2_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row0 = (wid.x + wid.y * nv2_p.groups_x) * (NV2_SGS * 2u) + sgid * 2u;
    let row1 = row0 + 1u;
    let pairs = nv2_p.k_blocks >> 1u;
    let stride = nv2_p.k_blocks >> 1u;
    let live0 = row0 < nv2_p.n_rows;
    let live1 = row1 < nv2_p.n_rows;
    let wb0 = select(0u, row0 * stride, live0);
    let wb1 = select(0u, row1 * stride, live1);
    let sr0 = select(0u, row0, live0);
    let sr1 = select(0u, row1, live1);

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let xv = nv2_x4[p];
        let xm0 = nv2_i8map(xv.x);
        let xm1 = nv2_i8map(xv.x >> 4u);
        let xm2 = nv2_i8map(xv.y);
        let xm3 = nv2_i8map(xv.y >> 4u);
        let xm4 = nv2_i8map(xv.z);
        let xm5 = nv2_i8map(xv.z >> 4u);
        let xm6 = nv2_i8map(xv.w);
        let xm7 = nv2_i8map(xv.w >> 4u);
        let b0 = p << 1u;
        let xsw = nv2_xs[b0 >> 2u];
        let xs0 = nv2_ue4m3(byte_at(xsw, b0));
        let xs1 = nv2_ue4m3(byte_at(xsw, b0 + 1u));

        let wv0 = nv2_w4[wb0 + p];
        let wsi0 = nvfp4_scale_byte_index(sr0, b0, nv2_p.k_tiles);
        let wsw0 = nv2_ws[wsi0 >> 2u];
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

        let wv1 = nv2_w4[wb1 + p];
        let wsi1 = nvfp4_scale_byte_index(sr1, b0, nv2_p.k_tiles);
        let wsw1 = nv2_ws[wsi1 >> 2u];
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
        nv2_y[row0 >> 1u] = nv2_pk_word(t0, t1, row0);
    }
}

const NV2_SECTION_MROWQ: u32 = 9u;

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_mrowq(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row0 = (wid.x + wid.y * nv2_p.groups_x) * (NV2_SGS * 2u) + sgid * 2u;
    let row1 = row0 + 1u;
    let quads = nv2_p.k_blocks >> 2u;
    let stride = nv2_p.k_blocks >> 1u;
    let live0 = row0 < nv2_p.n_rows;
    let live1 = row1 < nv2_p.n_rows;
    let wb0 = select(0u, row0 * stride, live0);
    let wb1 = select(0u, row1 * stride, live1);
    let sr0 = select(0u, row0, live0);
    let sr1 = select(0u, row1, live1);

    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var q = lane; q < quads; q = q + NV2_LANES) {
        let xa = nv2_x4[2u * q];
        let xc = nv2_x4[2u * q + 1u];
        let xm0 = nv2_i8map(xa.x);
        let xm1 = nv2_i8map(xa.x >> 4u);
        let xm2 = nv2_i8map(xa.y);
        let xm3 = nv2_i8map(xa.y >> 4u);
        let xm4 = nv2_i8map(xa.z);
        let xm5 = nv2_i8map(xa.z >> 4u);
        let xm6 = nv2_i8map(xa.w);
        let xm7 = nv2_i8map(xa.w >> 4u);
        let xm8 = nv2_i8map(xc.x);
        let xm9 = nv2_i8map(xc.x >> 4u);
        let xm10 = nv2_i8map(xc.y);
        let xm11 = nv2_i8map(xc.y >> 4u);
        let xm12 = nv2_i8map(xc.z);
        let xm13 = nv2_i8map(xc.z >> 4u);
        let xm14 = nv2_i8map(xc.w);
        let xm15 = nv2_i8map(xc.w >> 4u);
        let xsw = nv2_xs[q];
        let xs0 = nv2_ue4m3(byte_at(xsw, 0u));
        let xs1 = nv2_ue4m3(byte_at(xsw, 1u));
        let xs2 = nv2_ue4m3(byte_at(xsw, 2u));
        let xs3 = nv2_ue4m3(byte_at(xsw, 3u));

        let wa0 = nv2_w4[wb0 + 2u * q];
        let wc0 = nv2_w4[wb0 + 2u * q + 1u];
        let ws0 = nv2_ws[nvfp4_scale_byte_index(sr0, q << 2u, nv2_p.k_tiles) >> 2u];
        let a0 = dot4I8Packed(nv2_i8map(wa0.x), xm0)
            + dot4I8Packed(nv2_i8map(wa0.x >> 4u), xm1)
            + dot4I8Packed(nv2_i8map(wa0.y), xm2)
            + dot4I8Packed(nv2_i8map(wa0.y >> 4u), xm3);
        let a1 = dot4I8Packed(nv2_i8map(wa0.z), xm4)
            + dot4I8Packed(nv2_i8map(wa0.z >> 4u), xm5)
            + dot4I8Packed(nv2_i8map(wa0.w), xm6)
            + dot4I8Packed(nv2_i8map(wa0.w >> 4u), xm7);
        let a2 = dot4I8Packed(nv2_i8map(wc0.x), xm8)
            + dot4I8Packed(nv2_i8map(wc0.x >> 4u), xm9)
            + dot4I8Packed(nv2_i8map(wc0.y), xm10)
            + dot4I8Packed(nv2_i8map(wc0.y >> 4u), xm11);
        let a3 = dot4I8Packed(nv2_i8map(wc0.z), xm12)
            + dot4I8Packed(nv2_i8map(wc0.z >> 4u), xm13)
            + dot4I8Packed(nv2_i8map(wc0.w), xm14)
            + dot4I8Packed(nv2_i8map(wc0.w >> 4u), xm15);
        acc0 = fma(nv2_ue4m3(byte_at(ws0, 0u)) * xs0, f32(a0) * 0.25, acc0);
        acc0 = fma(nv2_ue4m3(byte_at(ws0, 1u)) * xs1, f32(a1) * 0.25, acc0);
        acc0 = fma(nv2_ue4m3(byte_at(ws0, 2u)) * xs2, f32(a2) * 0.25, acc0);
        acc0 = fma(nv2_ue4m3(byte_at(ws0, 3u)) * xs3, f32(a3) * 0.25, acc0);

        let wa1 = nv2_w4[wb1 + 2u * q];
        let wc1 = nv2_w4[wb1 + 2u * q + 1u];
        let ws1 = nv2_ws[nvfp4_scale_byte_index(sr1, q << 2u, nv2_p.k_tiles) >> 2u];
        let b0d = dot4I8Packed(nv2_i8map(wa1.x), xm0)
            + dot4I8Packed(nv2_i8map(wa1.x >> 4u), xm1)
            + dot4I8Packed(nv2_i8map(wa1.y), xm2)
            + dot4I8Packed(nv2_i8map(wa1.y >> 4u), xm3);
        let b1d = dot4I8Packed(nv2_i8map(wa1.z), xm4)
            + dot4I8Packed(nv2_i8map(wa1.z >> 4u), xm5)
            + dot4I8Packed(nv2_i8map(wa1.w), xm6)
            + dot4I8Packed(nv2_i8map(wa1.w >> 4u), xm7);
        let b2d = dot4I8Packed(nv2_i8map(wc1.x), xm8)
            + dot4I8Packed(nv2_i8map(wc1.x >> 4u), xm9)
            + dot4I8Packed(nv2_i8map(wc1.y), xm10)
            + dot4I8Packed(nv2_i8map(wc1.y >> 4u), xm11);
        let b3d = dot4I8Packed(nv2_i8map(wc1.z), xm12)
            + dot4I8Packed(nv2_i8map(wc1.z >> 4u), xm13)
            + dot4I8Packed(nv2_i8map(wc1.w), xm14)
            + dot4I8Packed(nv2_i8map(wc1.w >> 4u), xm15);
        acc1 = fma(nv2_ue4m3(byte_at(ws1, 0u)) * xs0, f32(b0d) * 0.25, acc1);
        acc1 = fma(nv2_ue4m3(byte_at(ws1, 1u)) * xs1, f32(b1d) * 0.25, acc1);
        acc1 = fma(nv2_ue4m3(byte_at(ws1, 2u)) * xs2, f32(b2d) * 0.25, acc1);
        acc1 = fma(nv2_ue4m3(byte_at(ws1, 3u)) * xs3, f32(b3d) * 0.25, acc1);
    }

    let t0 = nv2_bfly(acc0);
    let t1 = nv2_bfly(acc1);
    if (lane == 0u && live0) {
        nv2_y[row0] = bf16_encode(t0 * nv2_p.alpha);
    }
    if (lane == 0u && live1) {
        nv2_y[row1] = bf16_encode(t1 * nv2_p.alpha);
    }
}

@compute @workgroup_size(NV2_WG)
fn gemv_nvfp4_warp_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * nv2_p.groups_x) * NV2_SGS + sgid;
    let live = row < nv2_p.n_rows;
    let blocks = select(0u, nv2_p.k_blocks, live);
    let wb = select(0u, row * nv2_p.k_blocks, live);
    let srow = select(0u, row, live);

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + NV2_LANES) {
        let wsi = nvfp4_scale_byte_index(srow, kb, nv2_p.k_tiles);
        let bs = nv2_ue4m3(byte_at(nv2_ws[wsi >> 2u], wsi))
            * nv2_ue4m3(byte_at(nv2_xs[kb >> 2u], kb));
        let wv = nv2_w2[wb + kb];
        let xv = nv2_x2[kb];
        acc = fma(bs, nv2_iblock(wv.x, wv.y, xv.x, xv.y), acc);
    }

    let total = nv2_bfly(acc);
    if (lane == 0u) {
        nv2_pk_bits[sgid] = bf16_encode(total * nv2_p.alpha) & 0xffffu;
    }
    workgroupBarrier();
    if ((sgid & 1u) == 0u && lane == 0u && live) {
        var word = nv2_pk_bits[sgid];
        if (row + 1u < nv2_p.n_rows) {
            word = word | (nv2_pk_bits[sgid + 1u] << 16u);
        }
        nv2_y[row >> 1u] = word;
    }
}
