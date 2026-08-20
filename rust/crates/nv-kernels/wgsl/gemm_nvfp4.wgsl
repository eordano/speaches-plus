struct GemmParams {
    alpha: f32,
    m: u32,
    n: u32,
    k: u32,
    row_words: u32,
    k_tiles: u32,
    tiles_n: u32,
    groups_x: u32,
};

@group(0) @binding(0) var<storage, read> gemm_a: array<u32>;
@group(0) @binding(1) var<storage, read> gemm_a_sf: array<u32>;
@group(0) @binding(2) var<storage, read> gemm_b: array<u32>;
@group(0) @binding(3) var<storage, read> gemm_b_sf: array<u32>;
@group(0) @binding(4) var<uniform> gemm_params: GemmParams;
@group(0) @binding(5) var<storage, read_write> gemm_d: array<u32>;

fn gemm_a_elem(row: u32, kk: u32) -> f32 {
    let w = gemm_a[row * gemm_params.row_words + (kk >> 3u)];
    let si = nvfp4_scale_byte_index(row, kk / NVFP4_BLOCK_SIZE, gemm_params.k_tiles);
    let sb = byte_at(gemm_a_sf[si >> 2u], si);
    return nvfp4_decode(nvfp4_nibble(w, kk)) * ue4m3_decode(sb);
}

fn gemm_b_elem(row: u32, kk: u32) -> f32 {
    let w = gemm_b[row * gemm_params.row_words + (kk >> 3u)];
    let si = nvfp4_scale_byte_index(row, kk / NVFP4_BLOCK_SIZE, gemm_params.k_tiles);
    let sb = byte_at(gemm_b_sf[si >> 2u], si);
    return nvfp4_decode(nvfp4_nibble(w, kk)) * ue4m3_decode(sb);
}

const GEMM_SCALAR_WG: u32 = 64u;

@compute @workgroup_size(64)
fn gemm_nvfp4_scalar(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let idx = (wid.x + wid.y * gemm_params.groups_x) * GEMM_SCALAR_WG + lid.x;
    if (idx >= gemm_params.m * gemm_params.n) {
        return;
    }
    let row = idx / gemm_params.n;
    let col = idx % gemm_params.n;
    let blocks = gemm_params.k / NVFP4_BLOCK_SIZE;
    var acc = 0.0;
    for (var b = 0u; b < blocks; b = b + 1u) {
        let k0 = b * NVFP4_BLOCK_SIZE;
        var block_dot = 0.0;
        for (var t = 0u; t < NVFP4_BLOCK_SIZE; t = t + 1u) {
            block_dot = block_dot + gemm_a_elem(row, k0 + t) * gemm_b_elem(col, k0 + t);
        }
        acc = acc + block_dot;
    }
    gemm_d[idx] = bf16_encode(acc * gemm_params.alpha);
}

const NVFP4_GEMM_SECTION_SPLIT: u32 = 1u;

struct GemmParams {
    alpha: f32,
    m: u32,
    n: u32,
    k: u32,
    row_words: u32,
    k_tiles: u32,
    tiles_n: u32,
    groups_x: u32,
};

@group(0) @binding(0) var<storage, read> gemm_a: array<u32>;
@group(0) @binding(1) var<storage, read> gemm_a_sf: array<u32>;
@group(0) @binding(2) var<storage, read> gemm_b: array<u32>;
@group(0) @binding(3) var<storage, read> gemm_b_sf: array<u32>;
@group(0) @binding(4) var<uniform> gemm_params: GemmParams;
@group(0) @binding(5) var<storage, read_write> gemm_d: array<u32>;

fn gemm_a_elem(row: u32, kk: u32) -> f32 {
    let w = gemm_a[row * gemm_params.row_words + (kk >> 3u)];
    let si = nvfp4_scale_byte_index(row, kk / NVFP4_BLOCK_SIZE, gemm_params.k_tiles);
    let sb = byte_at(gemm_a_sf[si >> 2u], si);
    return nvfp4_decode(nvfp4_nibble(w, kk)) * ue4m3_decode(sb);
}

fn gemm_b_elem(row: u32, kk: u32) -> f32 {
    let w = gemm_b[row * gemm_params.row_words + (kk >> 3u)];
    let si = nvfp4_scale_byte_index(row, kk / NVFP4_BLOCK_SIZE, gemm_params.k_tiles);
    let sb = byte_at(gemm_b_sf[si >> 2u], si);
    return nvfp4_decode(nvfp4_nibble(w, kk)) * ue4m3_decode(sb);
}

alias CoopA = coop_mat16x16<f16, A>;
alias CoopB = coop_mat16x16<f16, B>;
alias CoopC = coop_mat16x16<f32, C>;

const COOP_WG: u32 = 128u;
const COOP_TILE: u32 = 16u;
const COOP_TILE_ELEMS: u32 = 256u;
const COOP_BM: u32 = 2u;
const COOP_BN: u32 = 2u;
const COOP_TILES: u32 = 4u;
const COOP_BLOCK_M: u32 = 32u;
const COOP_BLOCK_N: u32 = 32u;
const COOP_FRAG_ELEMS: u32 = 1024u;
const COOP_FRAG_PER_LANE: u32 = 8u;

var<workgroup> coop_a: array<f16, 512>;
var<workgroup> coop_b: array<f16, 512>;
var<workgroup> coop_c: array<f32, 1024>;
var<workgroup> coop_zero: array<f32, 256>;

fn coop_stage_a(m0: u32, k0: u32, lidx: u32) {
    for (var e = lidx; e < COOP_BM * COOP_TILE_ELEMS; e = e + COOP_WG) {
        let t = e / COOP_TILE_ELEMS;
        let r = e - t * COOP_TILE_ELEMS;
        let i = r / COOP_TILE;
        let j = r - i * COOP_TILE;
        let gm = m0 + t * COOP_TILE + i;
        var av = 0.0;
        if (gm < gemm_params.m) {
            av = gemm_a_elem(gm, k0 + j);
        }
        coop_a[e] = f16(av);
    }
}

fn coop_stage_b(n0: u32, k0: u32, lidx: u32) {
    for (var e = lidx; e < COOP_BN * COOP_TILE_ELEMS; e = e + COOP_WG) {
        let t = e / COOP_TILE_ELEMS;
        let r = e - t * COOP_TILE_ELEMS;
        let i = r / COOP_TILE;
        let j = r - i * COOP_TILE;
        let gn = n0 + t * COOP_TILE + i;
        var bv = 0.0;
        if (gn < gemm_params.n) {
            bv = gemm_b_elem(gn, k0 + j);
        }
        coop_b[t * COOP_TILE_ELEMS + j * COOP_TILE + i] = f16(bv);
    }
}

@compute @workgroup_size(128)
fn gemm_nvfp4_coop(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32
) {
    let block = wid.x + wid.y * gemm_params.groups_x;
    let blocks_m = (gemm_params.m + COOP_BLOCK_M - 1u) / COOP_BLOCK_M;
    let bm = block / gemm_params.tiles_n;
    if (bm >= blocks_m) {
        return;
    }
    let bn = block % gemm_params.tiles_n;
    let m0 = bm * COOP_BLOCK_M;
    let n0 = bn * COOP_BLOCK_N;

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

    let kt_count = gemm_params.k / COOP_TILE;
    for (var kt = 0u; kt < kt_count; kt = kt + 1u) {
        let k0 = kt * COOP_TILE;
        workgroupBarrier();
        coop_stage_a(m0, k0, lidx);
        coop_stage_b(n0, k0, lidx);
        workgroupBarrier();
        let ta = coopLoadT<CoopA>(&coop_a[sm * COOP_TILE_ELEMS]);
        let tb = coopLoadT<CoopB>(&coop_b[sn * COOP_TILE_ELEMS]);
        let dot = coopMultiplyAdd(ta, tb, zero);
        coopStoreT(dot, &coop_c[sg * COOP_TILE_ELEMS]);
        workgroupBarrier();
        for (var s = 0u; s < COOP_FRAG_PER_LANE; s = s + 1u) {
            acc[s] = acc[s] + coop_c[lidx + s * COOP_WG];
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
        if (gm < gemm_params.m && gn < gemm_params.n) {
            gemm_d[gm * gemm_params.n + gn] = bf16_encode(acc[s] * gemm_params.alpha);
        }
    }
}
