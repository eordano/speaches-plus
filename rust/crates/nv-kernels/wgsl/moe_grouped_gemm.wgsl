struct MoeGemmParams {
    n: u32,
    k: u32,
    row_words: u32,
    k_tiles: u32,
    b_sf_stride_bytes: u32,
    b_words_per_expert: u32,
    total_m: u32,
    groups_x: u32,
    total_tiles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> moe_a: array<u32>;
@group(0) @binding(1) var<storage, read> moe_a_sf: array<u32>;
@group(0) @binding(2) var<storage, read> moe_b: array<u32>;
@group(0) @binding(3) var<storage, read> moe_b_sf: array<u32>;
@group(0) @binding(4) var<uniform> moe_params: MoeGemmParams;
@group(0) @binding(5) var<storage, read_write> moe_d: array<u32>;
@group(0) @binding(6) var<storage, read> moe_group_meta: array<u32>;
@group(0) @binding(7) var<storage, read> moe_map: array<u32>;

fn moe_a_elem(row: u32, kk: u32) -> f32 {
    let w = moe_a[row * moe_params.row_words + (kk >> 3u)];
    let si = nvfp4_scale_byte_index(row, kk / NVFP4_BLOCK_SIZE, moe_params.k_tiles);
    let sb = byte_at(moe_a_sf[si >> 2u], si);
    return nvfp4_decode(nvfp4_nibble(w, kk)) * ue4m3_decode(sb);
}

fn moe_b_elem(expert: u32, col: u32, kk: u32) -> f32 {
    let w = moe_b[expert * moe_params.b_words_per_expert + col * moe_params.row_words + (kk >> 3u)];
    let si = expert * moe_params.b_sf_stride_bytes
        + nvfp4_scale_byte_index(col, kk / NVFP4_BLOCK_SIZE, moe_params.k_tiles);
    let sb = byte_at(moe_b_sf[si >> 2u], si);
    return nvfp4_decode(nvfp4_nibble(w, kk)) * ue4m3_decode(sb);
}

const MOE_SCALAR_WG: u32 = 64u;
const MOE_SHARED_A_K_MAX: u32 = 4096u;

@group(0) @binding(8) var<storage, read> moe_a4: array<vec4<u32>>;
@group(0) @binding(9) var<storage, read> moe_b4: array<vec4<u32>>;

var<workgroup> moe_a_stage: array<f32, 4096>;

fn moe_a_sf_at(row: u32, blk: u32) -> f32 {
    let si = nvfp4_scale_byte_index(row, blk, moe_params.k_tiles);
    return ue4m3_decode(byte_at(moe_a_sf[si >> 2u], si));
}

fn moe_b_sf_at(expert: u32, col: u32, blk: u32) -> f32 {
    let si = expert * moe_params.b_sf_stride_bytes
        + nvfp4_scale_byte_index(col, blk, moe_params.k_tiles);
    return ue4m3_decode(byte_at(moe_b_sf[si >> 2u], si));
}

@compute @workgroup_size(64)
fn moe_grouped_gemm_scalar(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = (wid.x + wid.y * moe_params.groups_x) * MOE_SCALAR_WG + lid.x;
    if (idx >= moe_params.total_m * moe_params.n) {
        return;
    }
    let row = idx / moe_params.n;
    let col = idx % moe_params.n;
    let g = moe_map[row];
    let expert = moe_group_meta[g * 2u];
    let alpha = bitcast<f32>(moe_group_meta[g * 2u + 1u]);
    let blocks = moe_params.k / NVFP4_BLOCK_SIZE;
    var acc = 0.0;
    for (var b = 0u; b < blocks; b = b + 1u) {
        let k0 = b * NVFP4_BLOCK_SIZE;
        var block_dot = 0.0;
        for (var t = 0u; t < NVFP4_BLOCK_SIZE; t = t + 1u) {
            block_dot = block_dot + moe_a_elem(row, k0 + t) * moe_b_elem(expert, col, k0 + t);
        }
        acc = acc + block_dot;
    }
    moe_d[idx] = bf16_encode(acc * alpha);
}

@compute @workgroup_size(64)
fn moe_grouped_gemm_scalar_hoist(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = (wid.x + wid.y * moe_params.groups_x) * MOE_SCALAR_WG + lid.x;
    if (idx >= moe_params.total_m * moe_params.n) {
        return;
    }
    let row = idx / moe_params.n;
    let col = idx % moe_params.n;
    let g = moe_map[row];
    let expert = moe_group_meta[g * 2u];
    let alpha = bitcast<f32>(moe_group_meta[g * 2u + 1u]);
    let blocks = moe_params.k / NVFP4_BLOCK_SIZE;
    let a_base = row * moe_params.row_words;
    let b_base = expert * moe_params.b_words_per_expert + col * moe_params.row_words;
    var acc = 0.0;
    for (var b = 0u; b < blocks; b = b + 1u) {
        let sa = moe_a_sf_at(row, b);
        let sb = moe_b_sf_at(expert, col, b);
        let aw0 = moe_a[a_base + 2u * b];
        let aw1 = moe_a[a_base + 2u * b + 1u];
        let bw0 = moe_b[b_base + 2u * b];
        let bw1 = moe_b[b_base + 2u * b + 1u];
        var block_dot = 0.0;
        for (var t = 0u; t < 8u; t = t + 1u) {
            block_dot = block_dot + (nvfp4_decode(nvfp4_nibble(aw0, t)) * sa) * (nvfp4_decode(nvfp4_nibble(bw0, t)) * sb);
        }
        for (var t = 0u; t < 8u; t = t + 1u) {
            block_dot = block_dot + (nvfp4_decode(nvfp4_nibble(aw1, t)) * sa) * (nvfp4_decode(nvfp4_nibble(bw1, t)) * sb);
        }
        acc = acc + block_dot;
    }
    moe_d[idx] = bf16_encode(acc * alpha);
}

@compute @workgroup_size(64)
fn moe_grouped_gemm_scalar_hoist_v4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = (wid.x + wid.y * moe_params.groups_x) * MOE_SCALAR_WG + lid.x;
    if (idx >= moe_params.total_m * moe_params.n) {
        return;
    }
    let row = idx / moe_params.n;
    let col = idx % moe_params.n;
    let g = moe_map[row];
    let expert = moe_group_meta[g * 2u];
    let alpha = bitcast<f32>(moe_group_meta[g * 2u + 1u]);
    let pairs = moe_params.k / (2u * NVFP4_BLOCK_SIZE);
    let a4_base = (row * moe_params.row_words) >> 2u;
    let b4_base = (expert * moe_params.b_words_per_expert + col * moe_params.row_words) >> 2u;
    var acc = 0.0;
    for (var p = 0u; p < pairs; p = p + 1u) {
        let aw = moe_a4[a4_base + p];
        let bw = moe_b4[b4_base + p];
        let sa0 = moe_a_sf_at(row, 2u * p);
        let sb0 = moe_b_sf_at(expert, col, 2u * p);
        var block_dot = 0.0;
        for (var t = 0u; t < 8u; t = t + 1u) {
            block_dot = block_dot + (nvfp4_decode(nvfp4_nibble(aw.x, t)) * sa0) * (nvfp4_decode(nvfp4_nibble(bw.x, t)) * sb0);
        }
        for (var t = 0u; t < 8u; t = t + 1u) {
            block_dot = block_dot + (nvfp4_decode(nvfp4_nibble(aw.y, t)) * sa0) * (nvfp4_decode(nvfp4_nibble(bw.y, t)) * sb0);
        }
        acc = acc + block_dot;
        let sa1 = moe_a_sf_at(row, 2u * p + 1u);
        let sb1 = moe_b_sf_at(expert, col, 2u * p + 1u);
        var block_dot1 = 0.0;
        for (var t = 0u; t < 8u; t = t + 1u) {
            block_dot1 = block_dot1 + (nvfp4_decode(nvfp4_nibble(aw.z, t)) * sa1) * (nvfp4_decode(nvfp4_nibble(bw.z, t)) * sb1);
        }
        for (var t = 0u; t < 8u; t = t + 1u) {
            block_dot1 = block_dot1 + (nvfp4_decode(nvfp4_nibble(aw.w, t)) * sa1) * (nvfp4_decode(nvfp4_nibble(bw.w, t)) * sb1);
        }
        acc = acc + block_dot1;
    }
    moe_d[idx] = bf16_encode(acc * alpha);
}

fn nvfp4_decode_bits(nib: u32) -> f32 {
    let n = nib & 7u;
    let mag = select(((126u + (n >> 1u)) << 23u) | ((n & 1u) << 22u), n * 0x3F000000u, n < 2u);
    return bitcast<f32>(mag | ((nib & 8u) << 28u));
}

fn moe_quad_body(idx: u32, table: bool) {
    let row = idx / moe_params.n;
    let col = idx % moe_params.n;
    let g = moe_map[row];
    let expert = moe_group_meta[g * 2u];
    let alpha = bitcast<f32>(moe_group_meta[g * 2u + 1u]);
    let quads = moe_params.k / (4u * NVFP4_BLOCK_SIZE);
    let a4_base = (row * moe_params.row_words) >> 2u;
    let b4_base = (expert * moe_params.b_words_per_expert + col * moe_params.row_words) >> 2u;
    var acc = 0.0;
    for (var q = 0u; q < quads; q = q + 1u) {
        let aw0 = moe_a4[a4_base + 2u * q];
        let aw1 = moe_a4[a4_base + 2u * q + 1u];
        let bw0 = moe_b4[b4_base + 2u * q];
        let bw1 = moe_b4[b4_base + 2u * q + 1u];
        let asi = nvfp4_scale_byte_index(row, 4u * q, moe_params.k_tiles);
        let asw = moe_a_sf[asi >> 2u];
        let bsi = expert * moe_params.b_sf_stride_bytes
            + nvfp4_scale_byte_index(col, 4u * q, moe_params.k_tiles);
        let bsw = moe_b_sf[bsi >> 2u];
        for (var j = 0u; j < 4u; j = j + 1u) {
            let sa = ue4m3_decode(byte_at(asw, j));
            let sb = ue4m3_decode(byte_at(bsw, j));
            let jj = (j & 1u) * 2u;
            let av = select(aw1, aw0, j < 2u);
            let bv = select(bw1, bw0, j < 2u);
            let awlo = av[jj];
            let awhi = av[jj + 1u];
            let bwlo = bv[jj];
            let bwhi = bv[jj + 1u];
            var block_dot = 0.0;
            if (table) {
                for (var t = 0u; t < 8u; t = t + 1u) {
                    block_dot = block_dot + (nvfp4_decode(nvfp4_nibble(awlo, t)) * sa) * (nvfp4_decode(nvfp4_nibble(bwlo, t)) * sb);
                }
                for (var t = 0u; t < 8u; t = t + 1u) {
                    block_dot = block_dot + (nvfp4_decode(nvfp4_nibble(awhi, t)) * sa) * (nvfp4_decode(nvfp4_nibble(bwhi, t)) * sb);
                }
            } else {
                for (var t = 0u; t < 8u; t = t + 1u) {
                    block_dot = block_dot + (nvfp4_decode_bits(nvfp4_nibble(awlo, t)) * sa) * (nvfp4_decode_bits(nvfp4_nibble(bwlo, t)) * sb);
                }
                for (var t = 0u; t < 8u; t = t + 1u) {
                    block_dot = block_dot + (nvfp4_decode_bits(nvfp4_nibble(awhi, t)) * sa) * (nvfp4_decode_bits(nvfp4_nibble(bwhi, t)) * sb);
                }
            }
            acc = acc + block_dot;
        }
    }
    moe_d[idx] = bf16_encode(acc * alpha);
}

@compute @workgroup_size(64)
fn moe_grouped_gemm_scalar_quad(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = (wid.x + wid.y * moe_params.groups_x) * MOE_SCALAR_WG + lid.x;
    if (idx >= moe_params.total_m * moe_params.n) {
        return;
    }
    moe_quad_body(idx, true);
}

@compute @workgroup_size(64)
fn moe_grouped_gemm_scalar_quad_bits(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = (wid.x + wid.y * moe_params.groups_x) * MOE_SCALAR_WG + lid.x;
    if (idx >= moe_params.total_m * moe_params.n) {
        return;
    }
    moe_quad_body(idx, false);
}

@compute @workgroup_size(64)
fn moe_grouped_gemm_scalar_shared_a(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = (wid.x + wid.y * moe_params.groups_x) * MOE_SCALAR_WG + lid.x;
    let live = idx < moe_params.total_m * moe_params.n;
    let row = min(idx / moe_params.n, moe_params.total_m - 1u);
    for (var e = lid.x; e < moe_params.k; e = e + MOE_SCALAR_WG) {
        moe_a_stage[e] = moe_a_elem(row, e);
    }
    workgroupBarrier();
    let col = idx % moe_params.n;
    let g = moe_map[row];
    let expert = moe_group_meta[g * 2u];
    let alpha = bitcast<f32>(moe_group_meta[g * 2u + 1u]);
    let quads = moe_params.k / (4u * NVFP4_BLOCK_SIZE);
    let b4_base = (expert * moe_params.b_words_per_expert + col * moe_params.row_words) >> 2u;
    var acc = 0.0;
    for (var q = 0u; q < quads; q = q + 1u) {
        let bw0 = moe_b4[b4_base + 2u * q];
        let bw1 = moe_b4[b4_base + 2u * q + 1u];
        let bsi = expert * moe_params.b_sf_stride_bytes
            + nvfp4_scale_byte_index(col, 4u * q, moe_params.k_tiles);
        let bsw = moe_b_sf[bsi >> 2u];
        for (var j = 0u; j < 4u; j = j + 1u) {
            let sb = ue4m3_decode(byte_at(bsw, j));
            let jj = (j & 1u) * 2u;
            let bv = select(bw1, bw0, j < 2u);
            let bwlo = bv[jj];
            let bwhi = bv[jj + 1u];
            let k0 = (4u * q + j) * NVFP4_BLOCK_SIZE;
            var block_dot = 0.0;
            for (var t = 0u; t < 8u; t = t + 1u) {
                block_dot = block_dot + moe_a_stage[k0 + t] * (nvfp4_decode_bits(nvfp4_nibble(bwlo, t)) * sb);
            }
            for (var t = 0u; t < 8u; t = t + 1u) {
                block_dot = block_dot + moe_a_stage[k0 + 8u + t] * (nvfp4_decode_bits(nvfp4_nibble(bwhi, t)) * sb);
            }
            acc = acc + block_dot;
        }
    }
    if (live) {
        moe_d[idx] = bf16_encode(acc * alpha);
    }
}

const MOE_GEMM_SECTION_SPLIT: u32 = 1u;

struct MoeGemmParams {
    n: u32,
    k: u32,
    row_words: u32,
    k_tiles: u32,
    b_sf_stride_bytes: u32,
    b_words_per_expert: u32,
    total_m: u32,
    groups_x: u32,
    total_tiles: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> moe_a: array<u32>;
@group(0) @binding(1) var<storage, read> moe_a_sf: array<u32>;
@group(0) @binding(2) var<storage, read> moe_b: array<u32>;
@group(0) @binding(3) var<storage, read> moe_b_sf: array<u32>;
@group(0) @binding(4) var<uniform> moe_params: MoeGemmParams;
@group(0) @binding(5) var<storage, read_write> moe_d: array<u32>;
@group(0) @binding(6) var<storage, read> moe_group_meta: array<u32>;
@group(0) @binding(7) var<storage, read> moe_map: array<u32>;

fn moe_a_code(row: u32, kk: u32) -> f32 {
    let w = moe_a[row * moe_params.row_words + (kk >> 3u)];
    return nvfp4_decode(nvfp4_nibble(w, kk));
}

fn moe_b_code(expert: u32, col: u32, kk: u32) -> f32 {
    let w = moe_b[expert * moe_params.b_words_per_expert + col * moe_params.row_words + (kk >> 3u)];
    return nvfp4_decode(nvfp4_nibble(w, kk));
}

fn moe_a_scale(row: u32, blk: u32) -> f32 {
    let si = nvfp4_scale_byte_index(row, blk, moe_params.k_tiles);
    return ue4m3_decode(byte_at(moe_a_sf[si >> 2u], si));
}

fn moe_b_scale(expert: u32, col: u32, blk: u32) -> f32 {
    let si = expert * moe_params.b_sf_stride_bytes
        + nvfp4_scale_byte_index(col, blk, moe_params.k_tiles);
    return ue4m3_decode(byte_at(moe_b_sf[si >> 2u], si));
}

alias CoopA = coop_mat16x16<f16, A>;
alias CoopB = coop_mat16x16<f16, B>;
alias CoopC = coop_mat16x16<f32, C>;

const COOP_WG: u32 = 128u;
const COOP_TILE: u32 = 16u;
const COOP_TILE_ELEMS: u32 = 256u;
const COOP_BM: u32 = 2u;
const COOP_BN: u32 = 2u;
const COOP_FRAG_PER_LANE: u32 = 8u;
const COOP_CODE_SHIFT: f32 = 8.0;

var<workgroup> coop_a: array<f16, 512>;
var<workgroup> coop_b: array<f16, 512>;
var<workgroup> coop_c: array<f32, 1024>;
var<workgroup> coop_zero: array<f32, 256>;
var<workgroup> coop_sa: array<f32, 32>;
var<workgroup> coop_sb: array<f32, 32>;
var<workgroup> coop_rsa: array<f32, 32>;
var<workgroup> coop_rsb: array<f32, 32>;

fn moe_coop_stage_a(m0: u32, m_end: u32, k0: u32, lidx: u32) {
    for (var e = lidx; e < COOP_BM * COOP_TILE_ELEMS; e = e + COOP_WG) {
        let t = e / COOP_TILE_ELEMS;
        let r = e - t * COOP_TILE_ELEMS;
        let i = r / COOP_TILE;
        let j = r - i * COOP_TILE;
        let gm = m0 + t * COOP_TILE + i;
        var av = COOP_CODE_SHIFT;
        if (gm < m_end) {
            av = moe_a_code(gm, k0 + j) + COOP_CODE_SHIFT;
        }
        coop_a[e] = f16(av);
    }
}

fn moe_coop_stage_b(expert: u32, n0: u32, k0: u32, lidx: u32) {
    for (var e = lidx; e < COOP_BN * COOP_TILE_ELEMS; e = e + COOP_WG) {
        let t = e / COOP_TILE_ELEMS;
        let r = e - t * COOP_TILE_ELEMS;
        let i = r / COOP_TILE;
        let j = r - i * COOP_TILE;
        let gn = n0 + t * COOP_TILE + i;
        var bv = COOP_CODE_SHIFT;
        if (gn < moe_params.n) {
            bv = moe_b_code(expert, gn, k0 + j) + COOP_CODE_SHIFT;
        }
        coop_b[t * COOP_TILE_ELEMS + j * COOP_TILE + i] = f16(bv);
    }
}

fn moe_coop_stage_scales(expert: u32, m0: u32, m_end: u32, n0: u32, k0: u32, lidx: u32) {
    let blk = k0 / NVFP4_BLOCK_SIZE;
    if (lidx < 32u) {
        let gm = m0 + lidx;
        var sa = 0.0;
        var rs = 0.0;
        if (gm < m_end) {
            sa = moe_a_scale(gm, blk);
            for (var t = 0u; t < NVFP4_BLOCK_SIZE; t = t + 1u) {
                rs = rs + moe_a_code(gm, k0 + t);
            }
        }
        coop_sa[lidx] = sa;
        coop_rsa[lidx] = rs;
    } else if (lidx < 64u) {
        let gn = n0 + lidx - 32u;
        var sb = 0.0;
        var rs = 0.0;
        if (gn < moe_params.n) {
            sb = moe_b_scale(expert, gn, blk);
            for (var t = 0u; t < NVFP4_BLOCK_SIZE; t = t + 1u) {
                rs = rs + moe_b_code(expert, gn, k0 + t);
            }
        }
        coop_sb[lidx - 32u] = sb;
        coop_rsb[lidx - 32u] = rs;
    }
}

@compute @workgroup_size(128)
fn moe_grouped_gemm_coop(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    let block = wid.x + wid.y * moe_params.groups_x;
    if (block >= moe_params.total_tiles) {
        return;
    }
    let g = moe_map[block * 4u];
    let m0 = moe_map[block * 4u + 1u];
    let m_end = moe_map[block * 4u + 2u];
    let n0 = moe_map[block * 4u + 3u];
    let expert = moe_group_meta[g * 2u];
    let alpha = bitcast<f32>(moe_group_meta[g * 2u + 1u]);

    let sg = lidx / 32u;
    let sm = sg / COOP_BN;
    let sn = sg % COOP_BN;

    var acc: array<f32, 8>;
    for (var s = 0u; s < COOP_FRAG_PER_LANE; s = s + 1u) {
        acc[s] = 0.0;
    }
    for (var e = lidx; e < COOP_TILE_ELEMS; e = e + COOP_WG) {
        coop_zero[e] = 0.0;
    }
    workgroupBarrier();
    let zero = coopLoadT<CoopC>(&coop_zero[0]);

    let kt_count = moe_params.k / COOP_TILE;
    for (var kt = 0u; kt < kt_count; kt = kt + 1u) {
        let k0 = kt * COOP_TILE;
        workgroupBarrier();
        moe_coop_stage_a(m0, m_end, k0, lidx);
        moe_coop_stage_b(expert, n0, k0, lidx);
        moe_coop_stage_scales(expert, m0, m_end, n0, k0, lidx);
        workgroupBarrier();
        let ta = coopLoadT<CoopA>(&coop_a[sm * COOP_TILE_ELEMS]);
        let tb = coopLoadT<CoopB>(&coop_b[sn * COOP_TILE_ELEMS]);
        let dot = coopMultiplyAdd(ta, tb, zero);
        coopStoreT(dot, &coop_c[sg * COOP_TILE_ELEMS]);
        workgroupBarrier();
        for (var s = 0u; s < COOP_FRAG_PER_LANE; s = s + 1u) {
            let e = lidx + s * COOP_WG;
            let t = e / COOP_TILE_ELEMS;
            let r = e - t * COOP_TILE_ELEMS;
            let i = r / COOP_TILE;
            let j = r - i * COOP_TILE;
            let ri = (t / COOP_BN) * COOP_TILE + i;
            let cj = (t % COOP_BN) * COOP_TILE + j;
            let shift_sum = COOP_CODE_SHIFT * (coop_rsa[ri] + coop_rsb[cj])
                + COOP_CODE_SHIFT * COOP_CODE_SHIFT * f32(NVFP4_BLOCK_SIZE);
            acc[s] = acc[s] + (coop_c[e] - shift_sum) * (coop_sa[ri] * coop_sb[cj]);
        }
    }

    for (var s = 0u; s < COOP_FRAG_PER_LANE; s = s + 1u) {
        let e = lidx + s * COOP_WG;
        let t = e / COOP_TILE_ELEMS;
        let r = e - t * COOP_TILE_ELEMS;
        let i = r / COOP_TILE;
        let j = r - i * COOP_TILE;
        let gm = m0 + (t / COOP_BN) * COOP_TILE + i;
        let gn = n0 + (t % COOP_BN) * COOP_TILE + j;
        if (gm < m_end && gn < moe_params.n) {
            moe_d[gm * moe_params.n + gn] = bf16_encode(acc[s] * alpha);
        }
    }
}
