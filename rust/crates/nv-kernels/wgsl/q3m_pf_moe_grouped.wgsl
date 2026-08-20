
struct Q3ggParams {
    zn: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(20) var<storage, read> gg_sel_sorted: array<u32>;
@group(0) @binding(21) var<storage, read> gg_perm: array<u32>;
@group(0) @binding(22) var<uniform> gg_p: Q3ggParams;

const Q3GG_WARP_BATCH: u32 = 8u;

@compute @workgroup_size(NV2_WG)
fn q3w_pf_gemv_nvfp4_fmlut_pair_grouped(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let z0 = wid.z * 2u;
    let z1 = z0 + 1u;
    let has1 = z1 < gg_p.zn;
    let e0 = gg_sel_sorted[z0];
    var e1 = e0;
    let slot0 = gg_perm[z0];
    var slot1 = slot0;
    if (has1) {
        e1 = gg_sel_sorted[z1];
        slot1 = gg_perm[z1];
    }
    let shares_weight = has1 && e1 == e0;
    let row0 = (wid.x + wid.y * q34_p.groups_x) * (NV2_SGS * NV2_MR) + sgid * NV2_MR;
    let pairs = q34_p.k_blocks >> 1u;
    let stride = q34_p.k_blocks >> 1u;
    let webase0 = e0 * (q34_p.w_e_stride_vec2 >> 1u);
    let webase1 = e1 * (q34_p.w_e_stride_vec2 >> 1u);
    let sfbase0 = e0 * q34_p.sf_e_stride_bytes;
    let sfbase1 = e1 * q34_p.sf_e_stride_bytes;
    let x4base0 = slot0 * (q34_p.x_slot_stride_vec2 >> 1u);
    let x4base1 = slot1 * (q34_p.x_slot_stride_vec2 >> 1u);
    let xsfbase0 = slot0 * q34_p.xsf_slot_stride_bytes;
    let xsfbase1 = slot1 * q34_p.xsf_slot_stride_bytes;

    var live: array<u32, NV2_MR>;
    var wbase0: array<u32, NV2_MR>;
    var wbase1: array<u32, NV2_MR>;
    var srow: array<u32, NV2_MR>;
    var acca: array<f32, NV2_MR>;
    var accb: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        let r = row0 + m;
        let ok = r < q34_p.n_rows;
        live[m] = select(0u, 1u, ok);
        wbase0[m] = select(0u, webase0 + r * stride, ok);
        wbase1[m] = select(0u, webase1 + r * stride, ok);
        srow[m] = select(0u, r, ok);
        acca[m] = 0.0;
        accb[m] = 0.0;
    }

    for (var p = lane; p < pairs; p = p + NV2_LANES) {
        let b0 = p << 1u;
        let xva = q34_x4[x4base0 + p];
        let xsia = xsfbase0 + b0;
        let xswa = q34_xs[xsia >> 2u];
        let xsa0 = nv2_ue4m3(byte_at(xswa, xsia)) * 0.25;
        let xsa1 = nv2_ue4m3(byte_at(xswa, xsia + 1u)) * 0.25;
        let xea0 = nv2_meven(xva.x) * xsa0;
        let xoa0 = nv2_modd(xva.x) * xsa0;
        let xea1 = nv2_meven(xva.y) * xsa0;
        let xoa1 = nv2_modd(xva.y) * xsa0;
        let xea2 = nv2_meven(xva.z) * xsa1;
        let xoa2 = nv2_modd(xva.z) * xsa1;
        let xea3 = nv2_meven(xva.w) * xsa1;
        let xoa3 = nv2_modd(xva.w) * xsa1;
        var xeb0 = vec4<f32>(0.0);
        var xob0 = vec4<f32>(0.0);
        var xeb1 = vec4<f32>(0.0);
        var xob1 = vec4<f32>(0.0);
        var xeb2 = vec4<f32>(0.0);
        var xob2 = vec4<f32>(0.0);
        var xeb3 = vec4<f32>(0.0);
        var xob3 = vec4<f32>(0.0);
        if (has1) {
            let xvb = q34_x4[x4base1 + p];
            let xsib = xsfbase1 + b0;
            let xswb = q34_xs[xsib >> 2u];
            let xsb0 = nv2_ue4m3(byte_at(xswb, xsib)) * 0.25;
            let xsb1 = nv2_ue4m3(byte_at(xswb, xsib + 1u)) * 0.25;
            xeb0 = nv2_meven(xvb.x) * xsb0;
            xob0 = nv2_modd(xvb.x) * xsb0;
            xeb1 = nv2_meven(xvb.y) * xsb0;
            xob1 = nv2_modd(xvb.y) * xsb0;
            xeb2 = nv2_meven(xvb.z) * xsb1;
            xob2 = nv2_modd(xvb.z) * xsb1;
            xeb3 = nv2_meven(xvb.w) * xsb1;
            xob3 = nv2_modd(xvb.w) * xsb1;
        }
        for (var m = 0u; m < NV2_MR; m = m + 1u) {
            let wva = q34_w4[wbase0[m] + p];
            let wsia = sfbase0 + nvfp4_scale_byte_index(srow[m], b0, q34_p.k_tiles);
            let wswa = q34_ws[wsia >> 2u];
            let ws0a = nv2_ue4m3(byte_at(wswa, wsia));
            let ws1a = nv2_ue4m3(byte_at(wswa, wsia + 1u));
            let d0a = nv2_mdot8(wva.y, xea1, xoa1, nv2_mdot8(wva.x, xea0, xoa0, 0.0));
            let d1a = nv2_mdot8(wva.w, xea3, xoa3, nv2_mdot8(wva.z, xea2, xoa2, 0.0));
            acca[m] = fma(ws0a, d0a, acca[m]);
            acca[m] = fma(ws1a, d1a, acca[m]);
            if (has1) {
                var wvb = wva;
                var ws0b = ws0a;
                var ws1b = ws1a;
                if (!shares_weight) {
                    wvb = q34_w4[wbase1[m] + p];
                    let wsib = sfbase1 + nvfp4_scale_byte_index(srow[m], b0, q34_p.k_tiles);
                    let wswb = q34_ws[wsib >> 2u];
                    ws0b = nv2_ue4m3(byte_at(wswb, wsib));
                    ws1b = nv2_ue4m3(byte_at(wswb, wsib + 1u));
                }
                let d0b = nv2_mdot8(wvb.y, xeb1, xob1, nv2_mdot8(wvb.x, xeb0, xob0, 0.0));
                let d1b = nv2_mdot8(wvb.w, xeb3, xob3, nv2_mdot8(wvb.z, xeb2, xob2, 0.0));
                accb[m] = fma(ws0b, d0b, accb[m]);
                accb[m] = fma(ws1b, d1b, accb[m]);
            }
        }
    }

    var tota: array<f32, NV2_MR>;
    var totb: array<f32, NV2_MR>;
    for (var m = 0u; m < NV2_MR; m = m + 1u) {
        tota[m] = nv2_bfly(acca[m]);
        totb[m] = nv2_bfly(accb[m]);
    }
    if (lane == 0u) {
        let a0 = q34_alpha(e0);
        let yb0 = slot0 * q34_p.y_slot_stride_words;
        for (var m = 0u; m < NV2_MR; m = m + 2u) {
            if (live[m] == 1u) {
                q34_y[yb0 + ((row0 + m) >> 1u)] =
                    q34_pk_word(tota[m], tota[m + 1u], row0 + m, a0);
            }
        }
        if (has1) {
            let a1 = q34_alpha(e1);
            let yb1 = slot1 * q34_p.y_slot_stride_words;
            for (var m = 0u; m < NV2_MR; m = m + 2u) {
                if (live[m] == 1u) {
                    q34_y[yb1 + ((row0 + m) >> 1u)] =
                        q34_pk_word(totb[m], totb[m + 1u], row0 + m, a1);
                }
            }
        }
    }
}

@compute @workgroup_size(NV2_WG)
fn q3w_pf_gemv_nvfp4_warp_grouped(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let zb = wid.z * Q3GG_WARP_BATCH;
    let row = (wid.x + wid.y * q34_p.groups_x) * NV2_SGS + sgid;
    let live = row < q34_p.n_rows;
    let blocks = select(0u, q34_p.k_blocks, live);
    let sfrow = select(0u, row, live);
    let wrow = select(0u, row * q34_p.k_blocks, live);

    var ee: array<u32, Q3GG_WARP_BATCH>;
    var xb: array<u32, Q3GG_WARP_BATCH>;
    var xsb: array<u32, Q3GG_WARP_BATCH>;
    var yslot: array<u32, Q3GG_WARP_BATCH>;
    let nz = min(Q3GG_WARP_BATCH, gg_p.zn - zb);
    for (var r = 0u; r < Q3GG_WARP_BATCH; r = r + 1u) {
        let z = zb + min(r, nz - 1u);
        ee[r] = gg_sel_sorted[z];
        let slot = gg_perm[z];
        xb[r] = slot * q34_p.x_slot_stride_vec2;
        xsb[r] = slot * q34_p.xsf_slot_stride_bytes;
        yslot[r] = slot;
    }

    var acc: array<f32, Q3GG_WARP_BATCH>;
    for (var r = 0u; r < Q3GG_WARP_BATCH; r = r + 1u) {
        acc[r] = 0.0;
    }
    for (var kb = lane; kb < blocks; kb = kb + NV2_LANES) {
        let sfoff = nvfp4_scale_byte_index(sfrow, kb, q34_p.k_tiles);
        var eprev = ee[0];
        var wv = q34_w2[eprev * q34_p.w_e_stride_vec2 + wrow + kb];
        let wsi0 = eprev * q34_p.sf_e_stride_bytes + sfoff;
        var ws = nv2_ue4m3(byte_at(q34_ws[wsi0 >> 2u], wsi0));
        for (var r = 0u; r < Q3GG_WARP_BATCH; r = r + 1u) {
            if (r >= nz) {
                break;
            }
            if (ee[r] != eprev) {
                eprev = ee[r];
                wv = q34_w2[eprev * q34_p.w_e_stride_vec2 + wrow + kb];
                let wsi = eprev * q34_p.sf_e_stride_bytes + sfoff;
                ws = nv2_ue4m3(byte_at(q34_ws[wsi >> 2u], wsi));
            }
            let xsi = xsb[r] + kb;
            let bs = ws * nv2_ue4m3(byte_at(q34_xs[xsi >> 2u], xsi));
            let xv = q34_x2[xb[r] + kb];
            acc[r] = fma(bs, nv2_iblock(wv.x, wv.y, xv.x, xv.y), acc[r]);
        }
    }

    for (var r = 0u; r < Q3GG_WARP_BATCH; r = r + 1u) {
        if (r >= nz) {
            break;
        }
        let total = nv2_bfly(acc[r]);
        if (lane == 0u) {
            q34_pk_bits[sgid] = bf16_encode(total * q34_alpha(ee[r])) & 0xffffu;
        }
        workgroupBarrier();
        if ((sgid & 1u) == 0u && lane == 0u && live) {
            var word = q34_pk_bits[sgid];
            if (row + 1u < q34_p.n_rows) {
                word = word | (q34_pk_bits[sgid + 1u] << 16u);
            }
            q34_y[yslot[r] * q34_p.y_slot_stride_words + (row >> 1u)] = word;
        }
        workgroupBarrier();
    }
}
