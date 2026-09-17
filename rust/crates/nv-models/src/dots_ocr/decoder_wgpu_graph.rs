use anyhow::{Context, Result};

use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};

use crate::gemma4_wgpu_shared::{pack_pairs, rope_tables_from_inv_freq};
use crate::nvfp4_host::{quantize_nvfp4_host, upload_nvfp4, Nvfp4Gpu, NVFP4_BLOCK};
use crate::wgpu_ledger::{VramLedger, VramReport};

use super::decode_gemv_msl;
use super::decoder::{argmax_banned, DotsDecoderConfig, GenerateOptions, GenerateOutcome};
use crate::deepseek_ocr::decoder::{banned_tokens_windowed_ngram, detect_loop};

const LAGUNA_COMMON_WGSL: &str = include_str!("../../../nv-kernels/wgsl/laguna_common.wgsl");
const LAGUNA_GEMV_BF16_WGSL: &str = include_str!("../../../nv-kernels/wgsl/laguna_gemv_bf16.wgsl");
const LAGUNA_QUANT_ROWS_WGSL: &str =
    include_str!("../../../nv-kernels/wgsl/laguna_quant_rows.wgsl");

pub const NVFP4_GEMV_WGSL: &str = r#"
struct DgGemv4Params {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> dg4_w: array<vec2<u32>>;
@group(0) @binding(11) var<storage, read> dg4_ws: array<u32>;
@group(0) @binding(12) var<storage, read> dg4_x: array<vec2<u32>>;
@group(0) @binding(13) var<storage, read> dg4_xs: array<u32>;
@group(0) @binding(14) var<uniform> dg4_p: DgGemv4Params;
@group(0) @binding(15) var<storage, read_write> dg4_y: array<u32>;

var<workgroup> dg4_red: array<f32, 256>;

const DG4_LANES_PER_ROW: u32 = 32u;
const DG4_ROWS_PER_GROUP: u32 = 8u;

@compute @workgroup_size(256)
fn dg_gemv_nvfp4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let slot = tid / DG4_LANES_PER_ROW;
    let lane = tid % DG4_LANES_PER_ROW;
    let group = wid.x + wid.y * dg4_p.groups_x;
    let row = group * DG4_ROWS_PER_GROUP + slot;
    let live = row < dg4_p.n_rows;
    let wbase = select(0u, row * dg4_p.k_blocks, live);
    let blocks = select(0u, dg4_p.k_blocks, live);

    var acc = 0.0;
    for (var kb = lane; kb < blocks; kb = kb + DG4_LANES_PER_ROW) {
        let wsi = nvfp4_scale_byte_index(row, kb, dg4_p.k_tiles);
        let ws = byte_at(dg4_ws[wsi >> 2u], wsi);
        let xs = byte_at(dg4_xs[kb >> 2u], kb);
        let block_scale = gemv_ue4m3_decode(ws) * gemv_ue4m3_decode(xs);
        let wv = dg4_w[wbase + kb];
        let xv = dg4_x[kb];
        var dot = 0.0;
        dot = gemv_dot8(wv.x, xv.x, dot);
        dot = gemv_dot8(wv.y, xv.y, dot);
        acc = fma(block_scale, dot, acc);
    }
    dg4_red[tid] = acc;
    workgroupBarrier();
    for (var stride = DG4_LANES_PER_ROW >> 1u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            dg4_red[tid] = dg4_red[tid] + dg4_red[tid + stride];
        }
        workgroupBarrier();
    }

    if (lane == 0u && (slot & 1u) == 0u && row < dg4_p.n_rows) {
        let lo = dg4_red[tid] * dg4_p.alpha;
        var hi = 0.0;
        if (row + 1u < dg4_p.n_rows) {
            hi = dg4_red[tid + DG4_LANES_PER_ROW] * dg4_p.alpha;
        }
        dg4_y[row >> 1u] = bf16_pack(lo, hi);
    }
}

struct DgnParams {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(90) var<storage, read> dgn_w: array<vec2<u32>>;
@group(0) @binding(91) var<storage, read> dgn_ws: array<u32>;
@group(0) @binding(92) var<storage, read> dgn_x: array<u32>;
@group(0) @binding(93) var<uniform> dgn_p: DgnParams;
@group(0) @binding(94) var<storage, read_write> dgn_y: array<u32>;

const DGN_LANES: u32 = 32u;
const DGN_ROWS: u32 = 4u;
const DGN_ROWS_PER_GROUP: u32 = 4u;

var<workgroup> dgn_red: array<vec4<f32>, 32>;

fn dgn_e2m1_vec4(pw: u32, base: u32) -> vec4<f32> {
    let sh = vec4<u32>(base, base + 4u, base + 8u, base + 12u);
    let n = (vec4<u32>(pw, pw, pw, pw) >> sh) & vec4<u32>(15u, 15u, 15u, 15u);
    let k = n & vec4<u32>(7u, 7u, 7u, 7u);
    let norm = (k + vec4<u32>(252u, 252u, 252u, 252u)) << vec4<u32>(22u, 22u, 22u, 22u);
    let sub = (k & vec4<u32>(1u, 1u, 1u, 1u)) * vec4<u32>(0x3f000000u);
    let mag = select(norm, sub, k < vec4<u32>(2u, 2u, 2u, 2u));
    let sign = (n & vec4<u32>(8u, 8u, 8u, 8u)) << vec4<u32>(28u, 28u, 28u, 28u);
    return bitcast<vec4<f32>>(sign | mag);
}

fn dgn_xf_pair(a: u32, b: u32) -> vec4<f32> {
    return vec4<f32>(bf16_lo(a), bf16_hi(a), bf16_lo(b), bf16_hi(b));
}

fn dgn_dot8(pw: u32, a: vec4<f32>, b: vec4<f32>) -> f32 {
    return dot(dgn_e2m1_vec4(pw, 0u), a) + dot(dgn_e2m1_vec4(pw, 16u), b);
}

@compute @workgroup_size(32)
fn dg_gemv_nvfp4_mrow4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let sg = tid / DGN_LANES;
    let lane = tid % DGN_LANES;
    let group = wid.x + wid.y * dgn_p.groups_x;
    let row0 = group * DGN_ROWS_PER_GROUP + sg * DGN_ROWS;
    let blocks = dgn_p.k_blocks;

    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var kb = lane; kb < blocks; kb = kb + DGN_LANES) {
        let xb = kb * 8u;
        let xa = dgn_xf_pair(dgn_x[xb], dgn_x[xb + 1u]);
        let xb1 = dgn_xf_pair(dgn_x[xb + 2u], dgn_x[xb + 3u]);
        let xc = dgn_xf_pair(dgn_x[xb + 4u], dgn_x[xb + 5u]);
        let xd = dgn_xf_pair(dgn_x[xb + 6u], dgn_x[xb + 7u]);
        for (var r = 0u; r < DGN_ROWS; r = r + 1u) {
            let row = row0 + r;
            if (row >= dgn_p.n_rows) {
                continue;
            }
            let wv = dgn_w[row * blocks + kb];
            let wsi = nvfp4_scale_byte_index(row, kb, dgn_p.k_tiles);
            let bs = gemv_ue4m3_decode(byte_at(dgn_ws[wsi >> 2u], wsi));
            var d = dgn_dot8(wv.x, xa, xb1);
            d = d + dgn_dot8(wv.y, xc, xd);
            acc[r] = fma(bs, d, acc[r]);
        }
    }

    dgn_red[tid] = acc;
    workgroupBarrier();
    for (var s = DGN_LANES >> 1u; s > 0u; s = s >> 1u) {
        if (lane < s) {
            dgn_red[tid] = dgn_red[tid] + dgn_red[tid + s];
        }
        workgroupBarrier();
    }

    if (lane == 0u) {
        let t = dgn_red[tid] * dgn_p.alpha;
        if (row0 + 1u < dgn_p.n_rows) {
            dgn_y[row0 >> 1u] = bf16_pack(t.x, t.y);
        } else if (row0 < dgn_p.n_rows) {
            dgn_y[row0 >> 1u] = bf16_pack(t.x, 0.0);
        }
        if (row0 + 3u < dgn_p.n_rows) {
            dgn_y[(row0 + 2u) >> 1u] = bf16_pack(t.z, t.w);
        } else if (row0 + 2u < dgn_p.n_rows) {
            dgn_y[(row0 + 2u) >> 1u] = bf16_pack(t.z, 0.0);
        }
    }
}
"#;

pub const NVFP4_GEMV_CALLS_ONE_DOT4I8PACKED_PER_STATEMENT_BECAUSE_THE_MSL_POLYFILL_COLLIDES: &str =
    "naga's MSL backend polyfills dot4I8Packed with local declarations that collide when two \
     calls are inlined into one expression, so this entry reaches for nv-kernels' gemv_dot8, \
     which is written one call per statement for exactly that reason. The otherwise identical \
     laguna_gemv_nvfp4 entry inlines both calls and silently returns wrong products on Metal.";

pub const QWEN2_BIAS_ROPE_WGSL: &str = r#"
struct DgRopeParams {
    n_rows: u32,
    head_dim: u32,
    half: u32,
    rope_rows: u32,
    has_bias: u32,
    src_f32: u32,
    src_slot_words: u32,
    dst_slot_words: u32,
};

@group(0) @binding(0) var<storage, read> dgr_src: array<u32>;
@group(0) @binding(1) var<storage, read> dgr_bias: array<u32>;
@group(0) @binding(2) var<storage, read> dgr_cos: array<f32>;
@group(0) @binding(3) var<storage, read> dgr_sin: array<f32>;
@group(0) @binding(4) var<storage, read> dgr_step: array<u32>;
@group(0) @binding(5) var<uniform> dgr_p: DgRopeParams;
@group(0) @binding(6) var<storage, read_write> dgr_dst: array<u32>;

const DG_ROPE_WG: u32 = 64u;
const DG_ROPE_MAX_HEAD_DIM: u32 = 256u;
const DG_STEP_POS: u32 = 1u;

var<workgroup> dgr_a: array<f32, DG_ROPE_MAX_HEAD_DIM>;

fn dgr_rot(e: u32, row_base: u32, rope: bool) -> f32 {
    if (!rope) {
        return dgr_a[e];
    }
    let h = dgr_p.half;
    if (e < h) {
        return dgr_a[e] * dgr_cos[row_base + e] - dgr_a[e + h] * dgr_sin[row_base + e];
    }
    let i = e - h;
    return dgr_a[i] * dgr_sin[row_base + i] + dgr_a[e] * dgr_cos[row_base + i];
}

@compute @workgroup_size(64)
fn dg_bias_rope(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x;
    let slot = wg.y;
    if (row >= dgr_p.n_rows) {
        return;
    }
    let lid = tid.x;
    let hd = dgr_p.head_dim;
    let words = hd >> 1u;
    let src_base = slot * dgr_p.src_slot_words;
    let dst_base = slot * dgr_p.dst_slot_words + row * words;
    for (var e = lid; e < hd; e = e + DG_ROPE_WG) {
        var v = 0.0;
        if (dgr_p.src_f32 == 1u) {
            v = bitcast<f32>(dgr_src[src_base * 2u + row * hd + e]);
        } else {
            let xw = dgr_src[src_base + row * words + (e >> 1u)];
            v = select(bf16_lo(xw), bf16_hi(xw), (e & 1u) == 1u);
        }
        if (dgr_p.has_bias == 1u) {
            let bw = dgr_bias[row * words + (e >> 1u)];
            v = v + select(bf16_lo(bw), bf16_hi(bw), (e & 1u) == 1u);
        }
        dgr_a[e] = v;
    }
    workgroupBarrier();
    let row_base = (dgr_step[DG_STEP_POS] + slot) * dgr_p.half;
    let rope = row < dgr_p.rope_rows;
    for (var i = lid; i < words; i = i + DG_ROPE_WG) {
        let e = i * 2u;
        dgr_dst[dst_base + i] = bf16_pack(dgr_rot(e, row_base, rope), dgr_rot(e + 1u, row_base, rope));
    }
}

@compute @workgroup_size(64)
fn dg_bias_rope_kvw(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x;
    let slot = wg.y;
    if (row >= dgr_p.n_rows) {
        return;
    }
    let lid = tid.x;
    let hd = dgr_p.head_dim;
    let words = hd >> 1u;
    let src_base = slot * dgr_p.src_slot_words;
    let dst_base = slot * dgr_p.dst_slot_words + row * words;
    for (var e = lid; e < hd; e = e + DG_ROPE_WG) {
        var v = 0.0;
        if (dgr_p.src_f32 == 1u) {
            v = bitcast<f32>(dgr_src[src_base * 2u + row * hd + e]);
        } else {
            let xw = dgr_src[src_base + row * words + (e >> 1u)];
            v = select(bf16_lo(xw), bf16_hi(xw), (e & 1u) == 1u);
        }
        if (dgr_p.has_bias == 1u) {
            let bw = dgr_bias[row * words + (e >> 1u)];
            v = v + select(bf16_lo(bw), bf16_hi(bw), (e & 1u) == 1u);
        }
        dgr_a[e] = v;
    }
    workgroupBarrier();
    let row_base = (dgr_step[DG_STEP_POS] + slot) * dgr_p.half;
    let rope = row < dgr_p.rope_rows;
    let n_kv = dgr_p.n_rows - dgr_p.rope_rows;
    let k_first = dgr_p.rope_rows - n_kv;
    let cache = dgr_step[DG_STEP_POS] + slot;
    for (var i = lid; i < words; i = i + DG_ROPE_WG) {
        let e = i * 2u;
        let packed = bf16_pack(dgr_rot(e, row_base, rope), dgr_rot(e + 1u, row_base, rope));
        dgr_dst[dst_base + i] = packed;
        if (row >= k_first && cache < dgk_p.kv_capacity) {
            if (row < dgr_p.rope_rows) {
                let kat = dg_kword(cache, row - k_first, i, dgk_p.kv_words, words, dgk_p.kv_capacity);
                dgk_kc[kat] = dgkv_store(packed);
            } else {
                dgk_vc[cache * dgk_p.kv_words + (row - dgr_p.rope_rows) * words + i] = dgkv_store(packed);
            }
        }
    }
}

struct DgKvParams {
    kv_words: u32,
    kv_capacity: u32,
    k_off_words: u32,
    v_off_words: u32,
    src_slot_words: u32,
    n_kv_heads: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(10) var<storage, read> dgk_src: array<u32>;
@group(0) @binding(11) var<storage, read> dgk_step: array<u32>;
@group(0) @binding(12) var<uniform> dgk_p: DgKvParams;
@group(0) @binding(13) var<storage, read_write> dgk_kc: array<u32>;
@group(0) @binding(14) var<storage, read_write> dgk_vc: array<u32>;

const DG_KT: u32 = 0u;

const DGKV16: u32 = 0u;

fn dgkv_store(w: u32) -> u32 {
    if (DGKV16 == 1u) {
        return pack2x16float(vec2<f32>(bf16_lo(w), bf16_hi(w)));
    }
    return w;
}

fn dgkv_lo(w: u32) -> f32 {
    if (DGKV16 == 1u) {
        return unpack2x16float(w).x;
    }
    return bf16_lo(w);
}

fn dgkv_hi(w: u32) -> f32 {
    if (DGKV16 == 1u) {
        return unpack2x16float(w).y;
    }
    return bf16_hi(w);
}

fn dg_kword(j: u32, kvh: u32, w: u32, kv_words: u32, words: u32, cap: u32) -> u32 {
    if (DG_KT == 1u) {
        return (kvh * words + w) * cap + j;
    }
    return j * kv_words + kvh * words + w;
}

@compute @workgroup_size(64)
fn dg_kv_write(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let i = wg.x * 64u + tid.x;
    if (i >= dgk_p.kv_words) {
        return;
    }
    let slot = wg.y;
    let cache = dgk_step[DG_STEP_POS] + slot;
    if (cache >= dgk_p.kv_capacity) {
        return;
    }
    let src = slot * dgk_p.src_slot_words;
    let at = cache * dgk_p.kv_words + i;
    let words = dgk_p.kv_words / dgk_p.n_kv_heads;
    let kat = dg_kword(cache, i / words, i % words, dgk_p.kv_words, words, dgk_p.kv_capacity);
    dgk_kc[kat] = dgkv_store(dgk_src[src + dgk_p.k_off_words + i]);
    dgk_vc[at] = dgkv_store(dgk_src[src + dgk_p.v_off_words + i]);
}

struct DgSiluParams {
    n_words: u32,
    gate_off_words: u32,
    up_off_words: u32,
    src_slot_words: u32,
    dst_slot_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

const DGACT16: u32 = 0u;

fn dg_act_pack(lo: f32, hi: f32) -> u32 {
    if (DGACT16 == 1u) {
        return pack2x16float(vec2<f32>(lo, hi));
    }
    return bf16_pack(lo, hi);
}

@group(0) @binding(20) var<storage, read> dgs_gu: array<u32>;
@group(0) @binding(21) var<uniform> dgs_p: DgSiluParams;
@group(0) @binding(22) var<storage, read_write> dgs_out: array<u32>;

@compute @workgroup_size(64)
fn dg_silu_mul(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let i = wg.x * 64u + tid.x;
    if (i >= dgs_p.n_words) {
        return;
    }
    let slot = wg.y;
    let src = slot * dgs_p.src_slot_words;
    let g = dgs_gu[src + dgs_p.gate_off_words + i];
    let u = dgs_gu[src + dgs_p.up_off_words + i];
    let lo = bf16_lo(g);
    let hi = bf16_hi(g);
    dgs_out[slot * dgs_p.dst_slot_words + i] = dg_act_pack(
        (lo / (1.0 + exp(-lo))) * bf16_lo(u),
        (hi / (1.0 + exp(-hi))) * bf16_hi(u)
    );
}

struct DgAttnParams {
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    gqa_group: u32,
    kv_capacity: u32,
    q_slot_words: u32,
    out_slot_words: u32,
    scale: f32,
};

@group(0) @binding(30) var<storage, read> dga_q: array<u32>;
@group(0) @binding(31) var<storage, read> dga_k: array<u32>;
@group(0) @binding(32) var<storage, read> dga_v: array<u32>;
@group(0) @binding(33) var<storage, read> dga_step: array<u32>;
@group(0) @binding(34) var<uniform> dga_p: DgAttnParams;
@group(0) @binding(35) var<storage, read_write> dga_out: array<u32>;

const DG_ATTN_WG: u32 = 256u;
const DG_ATTN_MAX_HEAD_DIM: u32 = 256u;

var<workgroup> dga_qs: array<f32, DG_ATTN_MAX_HEAD_DIM>;
var<workgroup> dga_acc: array<f32, DG_ATTN_MAX_HEAD_DIM>;
var<workgroup> dga_pr: array<f32, DG_ATTN_WG>;
var<workgroup> dga_red: array<f32, DG_ATTN_WG>;

const DGA_ABL: u32 = 0u;

fn dga_kld(idx: u32) -> u32 {
    if (DGA_ABL == 2u || DGA_ABL == 3u) {
        return 0x3f803f80u;
    }
    return dga_k[idx];
}

fn dga_vld(idx: u32) -> u32 {
    if (DGA_ABL == 1u || DGA_ABL == 3u) {
        return 0x3f803f80u;
    }
    return dga_v[idx];
}

fn dga_kval(idx: u32) -> f32 {
    let w = dga_k[idx >> 1u];
    return select(dgkv_lo(w), dgkv_hi(w), (idx & 1u) == 1u);
}

fn dga_vval(idx: u32) -> f32 {
    let w = dga_vld(idx >> 1u);
    return select(dgkv_lo(w), dgkv_hi(w), (idx & 1u) == 1u);
}

fn dga_score(j: u32, kvh: u32, hd: u32) -> f32 {
    let words = hd >> 1u;
    var a = 0.0;
    for (var w = 0u; w < words; w = w + 1u) {
        let kw = dga_kld(dg_kword(j, kvh, w, dga_p.n_kv_heads * words, words, dga_p.kv_capacity));
        a = fma(dga_qs[w * 2u], dgkv_lo(kw), a);
        a = fma(dga_qs[w * 2u + 1u], dgkv_hi(kw), a);
    }
    return a;
}

const DGA_V2: u32 = 1u;
const DGA_SG: u32 = 1u;
const DGA_SC_CAP: u32 = 1024u;

var<workgroup> dga_sc: array<f32, DGA_SC_CAP>;
var<workgroup> dga_sgm: array<f32, 8>;

fn dga_score_la(j: u32, kvh: u32, hd: u32) -> f32 {
    let words = hd >> 1u;
    let nw = dga_p.n_kv_heads * words;
    var a = 0.0;
    var w = 0u;
    for (; w + 4u <= words; w = w + 4u) {
        let k0 = dga_kld(dg_kword(j, kvh, w, nw, words, dga_p.kv_capacity));
        let k1 = dga_kld(dg_kword(j, kvh, w + 1u, nw, words, dga_p.kv_capacity));
        let k2 = dga_kld(dg_kword(j, kvh, w + 2u, nw, words, dga_p.kv_capacity));
        let k3 = dga_kld(dg_kword(j, kvh, w + 3u, nw, words, dga_p.kv_capacity));
        a = fma(dga_qs[w * 2u], dgkv_lo(k0), a);
        a = fma(dga_qs[w * 2u + 1u], dgkv_hi(k0), a);
        a = fma(dga_qs[w * 2u + 2u], dgkv_lo(k1), a);
        a = fma(dga_qs[w * 2u + 3u], dgkv_hi(k1), a);
        a = fma(dga_qs[w * 2u + 4u], dgkv_lo(k2), a);
        a = fma(dga_qs[w * 2u + 5u], dgkv_hi(k2), a);
        a = fma(dga_qs[w * 2u + 6u], dgkv_lo(k3), a);
        a = fma(dga_qs[w * 2u + 7u], dgkv_hi(k3), a);
    }
    for (; w < words; w = w + 1u) {
        let kw = dga_kld(dg_kword(j, kvh, w, nw, words, dga_p.kv_capacity));
        a = fma(dga_qs[w * 2u], dgkv_lo(kw), a);
        a = fma(dga_qs[w * 2u + 1u], dgkv_hi(kw), a);
    }
    return a;
}

fn dga_range_attend_v2(slot: u32, h: u32, lid: u32, start: u32, total: u32) {
    let hd = dga_p.head_dim;
    let words = hd >> 1u;
    let nkv = dga_p.n_kv_heads;
    let kvh = h / dga_p.gqa_group;
    let qbase = slot * dga_p.q_slot_words + h * words;

    for (var i = lid; i < hd; i = i + DG_ATTN_WG) {
        let w = dga_q[qbase + (i >> 1u)];
        dga_qs[i] = select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
        dga_acc[i] = 0.0;
    }
    workgroupBarrier();

    var lmax = -3.0e38;
    for (var j = start + lid; j < total; j = j + DG_ATTN_WG) {
        let raw = dga_score_la(j, kvh, hd);
        let rel = j - start;
        if (rel < DGA_SC_CAP) {
            dga_sc[rel] = raw;
        }
        lmax = max(lmax, raw * dga_p.scale);
    }
    var m = 0.0;
    if (DGA_SG == 1u) {
        let sgmax = subgroupMax(lmax);
        if ((lid & 31u) == 0u) {
            dga_sgm[lid >> 5u] = sgmax;
        }
        workgroupBarrier();
        m = max(
            max(max(dga_sgm[0], dga_sgm[1]), max(dga_sgm[2], dga_sgm[3])),
            max(max(dga_sgm[4], dga_sgm[5]), max(dga_sgm[6], dga_sgm[7])));
    } else {
        dga_red[lid] = lmax;
        workgroupBarrier();
        for (var s = DG_ATTN_WG >> 1u; s > 0u; s = s >> 1u) {
            if (lid < s) {
                dga_red[lid] = max(dga_red[lid], dga_red[lid + s]);
            }
            workgroupBarrier();
        }
        m = dga_red[0];
    }

    var lsum = 0.0;
    for (var base = start; base < total; base = base + DG_ATTN_WG) {
        let j = base + lid;
        var pv = 0.0;
        if (j < total) {
            let rel = j - start;
            var raw = 0.0;
            if (rel < DGA_SC_CAP) {
                raw = dga_sc[rel];
            } else {
                raw = dga_score_la(j, kvh, hd);
            }
            pv = exp(raw * dga_p.scale - m);
        }
        lsum = lsum + pv;
        dga_pr[lid] = pv;
        workgroupBarrier();
        let tmax = min(DG_ATTN_WG, total - base);
        for (var d = lid; d < hd; d = d + DG_ATTN_WG) {
            var a = 0.0;
            for (var t = 0u; t < tmax; t = t + 1u) {
                a = fma(dga_pr[t], dga_vval(((base + t) * nkv + kvh) * hd + d), a);
            }
            dga_acc[d] = dga_acc[d] + a;
        }
        workgroupBarrier();
    }

    dga_red[lid] = lsum;
    workgroupBarrier();
    for (var s = DG_ATTN_WG >> 1u; s > 31u; s = s >> 1u) {
        if (lid < s) {
            dga_red[lid] = dga_red[lid] + dga_red[lid + s];
        }
        workgroupBarrier();
    }
    var tot = 0.0;
    if (DGA_SG == 1u) {
        var sg = dga_red[lid];
        sg = sg + subgroupShuffleDown(sg, 16u);
        sg = sg + subgroupShuffleDown(sg, 8u);
        sg = sg + subgroupShuffleDown(sg, 4u);
        sg = sg + subgroupShuffleDown(sg, 2u);
        sg = sg + subgroupShuffleDown(sg, 1u);
        if (lid == 0u) {
            dga_sgm[0] = sg;
        }
        workgroupBarrier();
        tot = dga_sgm[0];
    } else {
        for (var s = 16u; s > 0u; s = s >> 1u) {
            if (lid < s) {
                dga_red[lid] = dga_red[lid] + dga_red[lid + s];
            }
            workgroupBarrier();
        }
        tot = dga_red[0];
    }
    var inv = 0.0;
    if (tot > 0.0) {
        inv = 1.0 / tot;
    }
    let obase = slot * dga_p.out_slot_words + h * words;
    for (var i = lid; i < words; i = i + DG_ATTN_WG) {
        let e = i * 2u;
        dga_out[obase + i] = dg_act_pack(dga_acc[e] * inv, dga_acc[e + 1u] * inv);
    }
}

@compute @workgroup_size(256)
fn dg_attn(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let slot = wg.y;
    if (h >= dga_p.n_q_heads) {
        return;
    }
    let lid = tid.x;
    let hd = dga_p.head_dim;
    let words = hd >> 1u;
    let nkv = dga_p.n_kv_heads;
    let kvh = h / dga_p.gqa_group;
    let total = min(dga_step[DG_STEP_POS] + slot + 1u, dga_p.kv_capacity);
    let qbase = slot * dga_p.q_slot_words + h * words;

    for (var i = lid; i < hd; i = i + DG_ATTN_WG) {
        let w = dga_q[qbase + (i >> 1u)];
        dga_qs[i] = select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
        dga_acc[i] = 0.0;
    }
    workgroupBarrier();

    var lmax = -3.0e38;
    for (var j = lid; j < total; j = j + DG_ATTN_WG) {
        lmax = max(lmax, dga_score(j, kvh, hd) * dga_p.scale);
    }
    dga_red[lid] = lmax;
    workgroupBarrier();
    for (var s = DG_ATTN_WG >> 1u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            dga_red[lid] = max(dga_red[lid], dga_red[lid + s]);
        }
        workgroupBarrier();
    }
    let m = dga_red[0];

    var lsum = 0.0;
    for (var base = 0u; base < total; base = base + DG_ATTN_WG) {
        let j = base + lid;
        var pv = 0.0;
        if (j < total) {
            pv = exp(dga_score(j, kvh, hd) * dga_p.scale - m);
        }
        lsum = lsum + pv;
        dga_pr[lid] = pv;
        workgroupBarrier();
        let tmax = min(DG_ATTN_WG, total - base);
        for (var d = lid; d < hd; d = d + DG_ATTN_WG) {
            var a = 0.0;
            for (var t = 0u; t < tmax; t = t + 1u) {
                a = fma(dga_pr[t], dga_vval(((base + t) * nkv + kvh) * hd + d), a);
            }
            dga_acc[d] = dga_acc[d] + a;
        }
        workgroupBarrier();
    }

    dga_red[lid] = lsum;
    workgroupBarrier();
    for (var s = DG_ATTN_WG >> 1u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            dga_red[lid] = dga_red[lid] + dga_red[lid + s];
        }
        workgroupBarrier();
    }
    var inv = 0.0;
    if (dga_red[0] > 0.0) {
        inv = 1.0 / dga_red[0];
    }
    let obase = slot * dga_p.out_slot_words + h * words;
    for (var i = lid; i < words; i = i + DG_ATTN_WG) {
        let e = i * 2u;
        dga_out[obase + i] = dg_act_pack(dga_acc[e] * inv, dga_acc[e + 1u] * inv);
    }
}

struct DgmParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    w_row_words: u32,
    x_slot_words: u32,
    y_slot_words: u32,
    out_f32: u32,
    slot_base: u32,
};

@group(0) @binding(40) var<storage, read> dgm_w: array<u32>;
@group(0) @binding(41) var<storage, read> dgm_x: array<u32>;
@group(0) @binding(42) var<uniform> dgm_p: DgmParams;
@group(0) @binding(43) var<storage, read_write> dgm_y: array<u32>;

var<workgroup> dgm_r0: array<vec4<f32>, 256>;
var<workgroup> dgm_r1: array<vec4<f32>, 256>;
var<workgroup> dgm_r2: array<vec4<f32>, 256>;
var<workgroup> dgm_r3: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn dg_gemm_m8(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * dgm_p.groups_x;
    let row = pair * 2u + half;
    let live = row < dgm_p.n_rows;
    let wbase = select(0u, row * dgm_p.w_row_words, live);
    let kw = select(0u, dgm_p.k_words, live);
    let xs = dgm_p.x_slot_words;
    let xb = dgm_p.slot_base * xs;

    var a0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var a1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = dgm_w[wbase + i];
        let wl = bf16_lo(ww);
        let wh = bf16_hi(ww);
        let x0 = dgm_x[xb + 0u * xs + i];
        let x1 = dgm_x[xb + 1u * xs + i];
        let x2 = dgm_x[xb + 2u * xs + i];
        let x3 = dgm_x[xb + 3u * xs + i];
        let x4 = dgm_x[xb + 4u * xs + i];
        let x5 = dgm_x[xb + 5u * xs + i];
        let x6 = dgm_x[xb + 6u * xs + i];
        let x7 = dgm_x[xb + 7u * xs + i];
        a0 = fma(vec4<f32>(wl, wl, wl, wl),
                 vec4<f32>(bf16_lo(x0), bf16_lo(x1), bf16_lo(x2), bf16_lo(x3)), a0);
        a0 = fma(vec4<f32>(wh, wh, wh, wh),
                 vec4<f32>(bf16_hi(x0), bf16_hi(x1), bf16_hi(x2), bf16_hi(x3)), a0);
        a1 = fma(vec4<f32>(wl, wl, wl, wl),
                 vec4<f32>(bf16_lo(x4), bf16_lo(x5), bf16_lo(x6), bf16_lo(x7)), a1);
        a1 = fma(vec4<f32>(wh, wh, wh, wh),
                 vec4<f32>(bf16_hi(x4), bf16_hi(x5), bf16_hi(x6), bf16_hi(x7)), a1);
    }
    dgm_r0[tid] = a0;
    dgm_r1[tid] = a1;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            dgm_r0[tid] = dgm_r0[tid] + dgm_r0[tid + stride];
            dgm_r1[tid] = dgm_r1[tid] + dgm_r1[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let ys = dgm_p.y_slot_words;
        let yb = dgm_p.slot_base * ys;
        let o = yb + (row >> 1u);
        let of32 = yb + row;
        let lo0 = dgm_r0[0];
        let lo1 = dgm_r1[0];
        var hi0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var hi1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (row + 1u < dgm_p.n_rows) {
            hi0 = dgm_r0[128];
            hi1 = dgm_r1[128];
        }
        if (dgm_p.out_f32 == 1u) {
            dgm_y[of32] = bitcast<u32>(lo0.x);
            dgm_y[ys + of32] = bitcast<u32>(lo0.y);
            dgm_y[2u * ys + of32] = bitcast<u32>(lo0.z);
            dgm_y[3u * ys + of32] = bitcast<u32>(lo0.w);
            dgm_y[4u * ys + of32] = bitcast<u32>(lo1.x);
            dgm_y[5u * ys + of32] = bitcast<u32>(lo1.y);
            dgm_y[6u * ys + of32] = bitcast<u32>(lo1.z);
            dgm_y[7u * ys + of32] = bitcast<u32>(lo1.w);
            if (row + 1u < dgm_p.n_rows) {
                dgm_y[of32 + 1u] = bitcast<u32>(hi0.x);
                dgm_y[ys + of32 + 1u] = bitcast<u32>(hi0.y);
                dgm_y[2u * ys + of32 + 1u] = bitcast<u32>(hi0.z);
                dgm_y[3u * ys + of32 + 1u] = bitcast<u32>(hi0.w);
                dgm_y[4u * ys + of32 + 1u] = bitcast<u32>(hi1.x);
                dgm_y[5u * ys + of32 + 1u] = bitcast<u32>(hi1.y);
                dgm_y[6u * ys + of32 + 1u] = bitcast<u32>(hi1.z);
                dgm_y[7u * ys + of32 + 1u] = bitcast<u32>(hi1.w);
            }
            return;
        }
        dgm_y[o] = bf16_pack(lo0.x, hi0.x);
        dgm_y[ys + o] = bf16_pack(lo0.y, hi0.y);
        dgm_y[2u * ys + o] = bf16_pack(lo0.z, hi0.z);
        dgm_y[3u * ys + o] = bf16_pack(lo0.w, hi0.w);
        dgm_y[4u * ys + o] = bf16_pack(lo1.x, hi1.x);
        dgm_y[5u * ys + o] = bf16_pack(lo1.y, hi1.y);
        dgm_y[6u * ys + o] = bf16_pack(lo1.z, hi1.z);
        dgm_y[7u * ys + o] = bf16_pack(lo1.w, hi1.w);
    }
}

@compute @workgroup_size(256)
fn dg_gemm_m16(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * dgm_p.groups_x;
    let row = pair * 2u + half;
    let live = row < dgm_p.n_rows;
    let wbase = select(0u, row * dgm_p.w_row_words, live);
    let kw = select(0u, dgm_p.k_words, live);
    let xs = dgm_p.x_slot_words;
    let xb = dgm_p.slot_base * xs;

    var a0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var a1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var a2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var a3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    for (var i = lane; i < kw; i = i + 128u) {
        let ww = dgm_w[wbase + i];
        let wl = bf16_lo(ww);
        let wh = bf16_hi(ww);
        let x0 = dgm_x[xb + 0u * xs + i];
        let x1 = dgm_x[xb + 1u * xs + i];
        let x2 = dgm_x[xb + 2u * xs + i];
        let x3 = dgm_x[xb + 3u * xs + i];
        let x4 = dgm_x[xb + 4u * xs + i];
        let x5 = dgm_x[xb + 5u * xs + i];
        let x6 = dgm_x[xb + 6u * xs + i];
        let x7 = dgm_x[xb + 7u * xs + i];
        let x8 = dgm_x[xb + 8u * xs + i];
        let x9 = dgm_x[xb + 9u * xs + i];
        let x10 = dgm_x[xb + 10u * xs + i];
        let x11 = dgm_x[xb + 11u * xs + i];
        let x12 = dgm_x[xb + 12u * xs + i];
        let x13 = dgm_x[xb + 13u * xs + i];
        let x14 = dgm_x[xb + 14u * xs + i];
        let x15 = dgm_x[xb + 15u * xs + i];
        a0 = fma(vec4<f32>(wl, wl, wl, wl),
                 vec4<f32>(bf16_lo(x0), bf16_lo(x1), bf16_lo(x2), bf16_lo(x3)), a0);
        a0 = fma(vec4<f32>(wh, wh, wh, wh),
                 vec4<f32>(bf16_hi(x0), bf16_hi(x1), bf16_hi(x2), bf16_hi(x3)), a0);
        a1 = fma(vec4<f32>(wl, wl, wl, wl),
                 vec4<f32>(bf16_lo(x4), bf16_lo(x5), bf16_lo(x6), bf16_lo(x7)), a1);
        a1 = fma(vec4<f32>(wh, wh, wh, wh),
                 vec4<f32>(bf16_hi(x4), bf16_hi(x5), bf16_hi(x6), bf16_hi(x7)), a1);
        a2 = fma(vec4<f32>(wl, wl, wl, wl),
                 vec4<f32>(bf16_lo(x8), bf16_lo(x9), bf16_lo(x10), bf16_lo(x11)), a2);
        a2 = fma(vec4<f32>(wh, wh, wh, wh),
                 vec4<f32>(bf16_hi(x8), bf16_hi(x9), bf16_hi(x10), bf16_hi(x11)), a2);
        a3 = fma(vec4<f32>(wl, wl, wl, wl),
                 vec4<f32>(bf16_lo(x12), bf16_lo(x13), bf16_lo(x14), bf16_lo(x15)), a3);
        a3 = fma(vec4<f32>(wh, wh, wh, wh),
                 vec4<f32>(bf16_hi(x12), bf16_hi(x13), bf16_hi(x14), bf16_hi(x15)), a3);
    }
    dgm_r0[tid] = a0;
    dgm_r1[tid] = a1;
    dgm_r2[tid] = a2;
    dgm_r3[tid] = a3;
    workgroupBarrier();
    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            dgm_r0[tid] = dgm_r0[tid] + dgm_r0[tid + stride];
            dgm_r1[tid] = dgm_r1[tid] + dgm_r1[tid + stride];
            dgm_r2[tid] = dgm_r2[tid] + dgm_r2[tid + stride];
            dgm_r3[tid] = dgm_r3[tid] + dgm_r3[tid + stride];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        let ys = dgm_p.y_slot_words;
        let yb = dgm_p.slot_base * ys;
        let o = yb + (row >> 1u);
        let of32 = yb + row;
        let lo0 = dgm_r0[0];
        let lo1 = dgm_r1[0];
        let lo2 = dgm_r2[0];
        let lo3 = dgm_r3[0];
        var hi0 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var hi1 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var hi2 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        var hi3 = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        if (row + 1u < dgm_p.n_rows) {
            hi0 = dgm_r0[128];
            hi1 = dgm_r1[128];
            hi2 = dgm_r2[128];
            hi3 = dgm_r3[128];
        }
        if (dgm_p.out_f32 == 1u) {
            dgm_y[of32] = bitcast<u32>(lo0.x);
            dgm_y[ys + of32] = bitcast<u32>(lo0.y);
            dgm_y[2u * ys + of32] = bitcast<u32>(lo0.z);
            dgm_y[3u * ys + of32] = bitcast<u32>(lo0.w);
            dgm_y[4u * ys + of32] = bitcast<u32>(lo1.x);
            dgm_y[5u * ys + of32] = bitcast<u32>(lo1.y);
            dgm_y[6u * ys + of32] = bitcast<u32>(lo1.z);
            dgm_y[7u * ys + of32] = bitcast<u32>(lo1.w);
            dgm_y[8u * ys + of32] = bitcast<u32>(lo2.x);
            dgm_y[9u * ys + of32] = bitcast<u32>(lo2.y);
            dgm_y[10u * ys + of32] = bitcast<u32>(lo2.z);
            dgm_y[11u * ys + of32] = bitcast<u32>(lo2.w);
            dgm_y[12u * ys + of32] = bitcast<u32>(lo3.x);
            dgm_y[13u * ys + of32] = bitcast<u32>(lo3.y);
            dgm_y[14u * ys + of32] = bitcast<u32>(lo3.z);
            dgm_y[15u * ys + of32] = bitcast<u32>(lo3.w);
            if (row + 1u < dgm_p.n_rows) {
                dgm_y[of32 + 1u] = bitcast<u32>(hi0.x);
                dgm_y[ys + of32 + 1u] = bitcast<u32>(hi0.y);
                dgm_y[2u * ys + of32 + 1u] = bitcast<u32>(hi0.z);
                dgm_y[3u * ys + of32 + 1u] = bitcast<u32>(hi0.w);
                dgm_y[4u * ys + of32 + 1u] = bitcast<u32>(hi1.x);
                dgm_y[5u * ys + of32 + 1u] = bitcast<u32>(hi1.y);
                dgm_y[6u * ys + of32 + 1u] = bitcast<u32>(hi1.z);
                dgm_y[7u * ys + of32 + 1u] = bitcast<u32>(hi1.w);
                dgm_y[8u * ys + of32 + 1u] = bitcast<u32>(hi2.x);
                dgm_y[9u * ys + of32 + 1u] = bitcast<u32>(hi2.y);
                dgm_y[10u * ys + of32 + 1u] = bitcast<u32>(hi2.z);
                dgm_y[11u * ys + of32 + 1u] = bitcast<u32>(hi2.w);
                dgm_y[12u * ys + of32 + 1u] = bitcast<u32>(hi3.x);
                dgm_y[13u * ys + of32 + 1u] = bitcast<u32>(hi3.y);
                dgm_y[14u * ys + of32 + 1u] = bitcast<u32>(hi3.z);
                dgm_y[15u * ys + of32 + 1u] = bitcast<u32>(hi3.w);
            }
            return;
        }
        dgm_y[o] = bf16_pack(lo0.x, hi0.x);
        dgm_y[ys + o] = bf16_pack(lo0.y, hi0.y);
        dgm_y[2u * ys + o] = bf16_pack(lo0.z, hi0.z);
        dgm_y[3u * ys + o] = bf16_pack(lo0.w, hi0.w);
        dgm_y[4u * ys + o] = bf16_pack(lo1.x, hi1.x);
        dgm_y[5u * ys + o] = bf16_pack(lo1.y, hi1.y);
        dgm_y[6u * ys + o] = bf16_pack(lo1.z, hi1.z);
        dgm_y[7u * ys + o] = bf16_pack(lo1.w, hi1.w);
        dgm_y[8u * ys + o] = bf16_pack(lo2.x, hi2.x);
        dgm_y[9u * ys + o] = bf16_pack(lo2.y, hi2.y);
        dgm_y[10u * ys + o] = bf16_pack(lo2.z, hi2.z);
        dgm_y[11u * ys + o] = bf16_pack(lo2.w, hi2.w);
        dgm_y[12u * ys + o] = bf16_pack(lo3.x, hi3.x);
        dgm_y[13u * ys + o] = bf16_pack(lo3.y, hi3.y);
        dgm_y[14u * ys + o] = bf16_pack(lo3.z, hi3.z);
        dgm_y[15u * ys + o] = bf16_pack(lo3.w, hi3.w);
    }
}

struct DgGatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    pad0: u32,
};

@group(0) @binding(50) var<storage, read> dgg_src: array<u32>;
@group(0) @binding(51) var<storage, read> dgg_tok: array<u32>;
@group(0) @binding(52) var<storage, read> dgg_mask: array<u32>;
@group(0) @binding(53) var<uniform> dgg_p: DgGatherParams;
@group(0) @binding(54) var<storage, read_write> dgg_out: array<u32>;

@compute @workgroup_size(64)
fn dg_gather_m(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let i = wg.x * 64u + tid.x;
    if (i >= dgg_p.hidden_words) {
        return;
    }
    let slot = wg.y;
    if (dgg_mask[slot] == 1u) {
        return;
    }
    let tok = dgg_tok[slot];
    if (tok < dgg_p.row_off || tok >= dgg_p.row_off + dgg_p.n_rows) {
        return;
    }
    dgg_out[slot * dgg_p.hidden_words + i] =
        dgg_src[(tok - dgg_p.row_off) * dgg_p.hidden_words + i];
}

@compute @workgroup_size(64)
fn dg_splice_m(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let i = wg.x * 64u + tid.x;
    if (i >= dgg_p.hidden_words) {
        return;
    }
    let slot = wg.y;
    if (dgg_mask[slot] == 0u) {
        return;
    }
    dgg_out[slot * dgg_p.hidden_words + i] = dgg_src[slot * dgg_p.hidden_words + i];
}

struct DgCvtParams {
    n_elems: u32,
    src_slot: u32,
    dst_slot: u32,
    to_f32: u32,
};

@group(0) @binding(100) var<storage, read> dgc_src: array<u32>;
@group(0) @binding(101) var<uniform> dgc_p: DgCvtParams;
@group(0) @binding(102) var<storage, read_write> dgc_dst: array<u32>;

@compute @workgroup_size(64)
fn dg_cvt(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let i = wg.x * 64u + tid.x;
    let slot = wg.y;
    if (dgc_p.to_f32 == 1u) {
        if (i >= dgc_p.n_elems) {
            return;
        }
        let w = dgc_src[slot * dgc_p.src_slot + (i >> 1u)];
        let v = select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
        dgc_dst[slot * dgc_p.dst_slot + i] = bitcast<u32>(v);
        return;
    }
    if (i * 2u >= dgc_p.n_elems) {
        return;
    }
    let a = bitcast<f32>(dgc_src[slot * dgc_p.src_slot + i * 2u]);
    let b = bitcast<f32>(dgc_src[slot * dgc_p.src_slot + i * 2u + 1u]);
    dgc_dst[slot * dgc_p.dst_slot + i] = bf16_pack(a, b);
}

struct DgSplitParams {
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    gqa_group: u32,
    kv_capacity: u32,
    n_splits: u32,
    part_stride: u32,
    scale: f32,
};

@group(0) @binding(60) var<storage, read> dgx_q: array<u32>;
@group(0) @binding(61) var<storage, read> dgx_k: array<u32>;
@group(0) @binding(62) var<storage, read> dgx_v: array<u32>;
@group(0) @binding(63) var<storage, read> dgx_step: array<u32>;
@group(0) @binding(64) var<uniform> dgx_p: DgSplitParams;
@group(0) @binding(65) var<storage, read_write> dgx_part: array<f32>;
@group(0) @binding(66) var<storage, read_write> dgx_out: array<u32>;

const DG_SPLIT_LIVE_SLICE_CAP_28_IS_THE_SPAN_3460_SHAPE_THE_CANON_DECODE_NUMBERS_RIDE: u32 = 28u;

const DGX_SG: u32 = 0u;

const DGX_ABL: u32 = 0u;

const DGX_PIPE: u32 = 0u;

fn dgx_kld(idx: u32) -> u32 {
    if (DGX_ABL == 2u || DGX_ABL == 3u) {
        return 0x3f803f80u;
    }
    return dgx_k[idx];
}

fn dgx_vld(idx: u32) -> u32 {
    if (DGX_ABL == 1u || DGX_ABL == 3u) {
        return 0x3f803f80u;
    }
    return dgx_v[idx];
}

fn dgx_per_slice(total: u32) -> u32 {
    let live = min(dgx_p.n_splits, DG_SPLIT_LIVE_SLICE_CAP_28_IS_THE_SPAN_3460_SHAPE_THE_CANON_DECODE_NUMBERS_RIDE);
    return (total + live - 1u) / live;
}

var<workgroup> dgx_qs: array<f32, DG_ATTN_MAX_HEAD_DIM>;
var<workgroup> dgx_acc: array<f32, DG_ATTN_MAX_HEAD_DIM>;
var<workgroup> dgx_pr: array<f32, DG_ATTN_WG>;
var<workgroup> dgx_red: array<f32, DG_ATTN_WG>;

fn dgx_kval(idx: u32) -> f32 {
    let w = dgx_k[idx >> 1u];
    return select(dgkv_lo(w), dgkv_hi(w), (idx & 1u) == 1u);
}

fn dgx_vval(idx: u32) -> f32 {
    let w = dgx_v[idx >> 1u];
    return select(dgkv_lo(w), dgkv_hi(w), (idx & 1u) == 1u);
}

fn dgx_score(j: u32, kvh: u32, hd: u32) -> f32 {
    let words = hd >> 1u;
    var a = 0.0;
    for (var w = 0u; w < words; w = w + 1u) {
        let kw = dgx_k[dg_kword(j, kvh, w, dgx_p.n_kv_heads * words, words, dgx_p.kv_capacity)];
        a = fma(dgx_qs[w * 2u], dgkv_lo(kw), a);
        a = fma(dgx_qs[w * 2u + 1u], dgkv_hi(kw), a);
    }
    return a;
}

var<workgroup> dgx_pvp: array<f32, 512>;

var<workgroup> dgx_sgm: array<f32, 8>;

@compute @workgroup_size(256)
fn dg_attn_split(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let sp = wg.y;
    if (h >= dgx_p.n_q_heads) {
        return;
    }
    let lid = tid.x;
    let hd = dgx_p.head_dim;
    let words = hd >> 1u;
    let nkv = dgx_p.n_kv_heads;
    let kvh = h / dgx_p.gqa_group;
    let total = min(dgx_step[DG_STEP_POS] + 1u, dgx_p.kv_capacity);
    let per = dgx_per_slice(total);
    let lo = sp * per;
    let hi = min(lo + per, total);
    let ml = h * dgx_p.n_splits + sp;
    let lbase = dgx_p.n_q_heads * dgx_p.n_splits;
    let abase = 2u * lbase;

    if (lo >= hi || DGX_ABL == 4u) {
        if (lid == 0u) {
            dgx_part[ml] = -3.0e38;
            dgx_part[lbase + ml] = 0.0;
        }
        return;
    }

    for (var i = lid; i < hd; i = i + DG_ATTN_WG) {
        let w = dgx_q[h * words + (i >> 1u)];
        dgx_qs[i] = select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
        dgx_acc[i] = 0.0;
    }
    workgroupBarrier();

    var m = -3.0e38;
    var lsum = 0.0;
    for (var base = lo; base < hi; base = base + DG_ATTN_WG) {
        let j = base + lid;
        var sv = -3.0e38;
        if (j < hi) {
            var a0 = 0.0;
            var a1 = 0.0;
            var a2 = 0.0;
            var a3 = 0.0;
            var w = 0u;
            if (DGX_PIPE == 1u) {
                for (; w + 8u <= words; w = w + 8u) {
                    let k0 = dgx_kld(dg_kword(j, kvh, w, nkv * words, words, dgx_p.kv_capacity));
                    let k1 = dgx_kld(dg_kword(j, kvh, w + 1u, nkv * words, words, dgx_p.kv_capacity));
                    let k2 = dgx_kld(dg_kword(j, kvh, w + 2u, nkv * words, words, dgx_p.kv_capacity));
                    let k3 = dgx_kld(dg_kword(j, kvh, w + 3u, nkv * words, words, dgx_p.kv_capacity));
                    let k4 = dgx_kld(dg_kword(j, kvh, w + 4u, nkv * words, words, dgx_p.kv_capacity));
                    let k5 = dgx_kld(dg_kword(j, kvh, w + 5u, nkv * words, words, dgx_p.kv_capacity));
                    let k6 = dgx_kld(dg_kword(j, kvh, w + 6u, nkv * words, words, dgx_p.kv_capacity));
                    let k7 = dgx_kld(dg_kword(j, kvh, w + 7u, nkv * words, words, dgx_p.kv_capacity));
                    a0 = fma(dgx_qs[w * 2u], dgkv_lo(k0), fma(dgx_qs[w * 2u + 1u], dgkv_hi(k0), a0));
                    a1 = fma(dgx_qs[w * 2u + 2u], dgkv_lo(k1), fma(dgx_qs[w * 2u + 3u], dgkv_hi(k1), a1));
                    a2 = fma(dgx_qs[w * 2u + 4u], dgkv_lo(k2), fma(dgx_qs[w * 2u + 5u], dgkv_hi(k2), a2));
                    a3 = fma(dgx_qs[w * 2u + 6u], dgkv_lo(k3), fma(dgx_qs[w * 2u + 7u], dgkv_hi(k3), a3));
                    a0 = fma(dgx_qs[w * 2u + 8u], dgkv_lo(k4), fma(dgx_qs[w * 2u + 9u], dgkv_hi(k4), a0));
                    a1 = fma(dgx_qs[w * 2u + 10u], dgkv_lo(k5), fma(dgx_qs[w * 2u + 11u], dgkv_hi(k5), a1));
                    a2 = fma(dgx_qs[w * 2u + 12u], dgkv_lo(k6), fma(dgx_qs[w * 2u + 13u], dgkv_hi(k6), a2));
                    a3 = fma(dgx_qs[w * 2u + 14u], dgkv_lo(k7), fma(dgx_qs[w * 2u + 15u], dgkv_hi(k7), a3));
                }
            }
            for (; w < words; w = w + 4u) {
                let k0 = dgx_kld(dg_kword(j, kvh, w, nkv * words, words, dgx_p.kv_capacity));
                a0 = fma(dgx_qs[w * 2u], dgkv_lo(k0), fma(dgx_qs[w * 2u + 1u], dgkv_hi(k0), a0));
                let k1 = dgx_kld(dg_kword(j, kvh, w + 1u, nkv * words, words, dgx_p.kv_capacity));
                a1 = fma(dgx_qs[w * 2u + 2u], dgkv_lo(k1), fma(dgx_qs[w * 2u + 3u], dgkv_hi(k1), a1));
                let k2 = dgx_kld(dg_kword(j, kvh, w + 2u, nkv * words, words, dgx_p.kv_capacity));
                a2 = fma(dgx_qs[w * 2u + 4u], dgkv_lo(k2), fma(dgx_qs[w * 2u + 5u], dgkv_hi(k2), a2));
                let k3 = dgx_kld(dg_kword(j, kvh, w + 3u, nkv * words, words, dgx_p.kv_capacity));
                a3 = fma(dgx_qs[w * 2u + 6u], dgkv_lo(k3), fma(dgx_qs[w * 2u + 7u], dgkv_hi(k3), a3));
            }
            sv = ((a0 + a1) + (a2 + a3)) * dgx_p.scale;
        }
        var mnew = m;
        if (DGX_SG == 1u) {
            let sm = subgroupMax(sv);
            if ((lid & 31u) == 0u) {
                dgx_sgm[lid >> 5u] = sm;
            }
            workgroupBarrier();
            mnew = max(m, max(
                max(max(dgx_sgm[0], dgx_sgm[1]), max(dgx_sgm[2], dgx_sgm[3])),
                max(max(dgx_sgm[4], dgx_sgm[5]), max(dgx_sgm[6], dgx_sgm[7]))));
        } else {
            dgx_red[lid] = sv;
            workgroupBarrier();
            for (var s = DG_ATTN_WG >> 1u; s > 0u; s = s >> 1u) {
                if (lid < s) {
                    dgx_red[lid] = max(dgx_red[lid], dgx_red[lid + s]);
                }
                workgroupBarrier();
            }
            mnew = max(m, dgx_red[0]);
            workgroupBarrier();
        }
        var corr = 1.0;
        if (m > -2.0e38) {
            corr = exp(m - mnew);
        }
        var pv = 0.0;
        if (j < hi) {
            pv = exp(sv - mnew);
        }
        dgx_pr[lid] = pv;
        dgx_red[lid] = pv;
        workgroupBarrier();
        if (DGX_SG == 1u) {
            for (var s = DG_ATTN_WG >> 1u; s > 31u; s = s >> 1u) {
                if (lid < s) {
                    dgx_red[lid] = dgx_red[lid] + dgx_red[lid + s];
                }
                workgroupBarrier();
            }
            var v = dgx_red[lid];
            v = v + subgroupShuffleDown(v, 16u);
            v = v + subgroupShuffleDown(v, 8u);
            v = v + subgroupShuffleDown(v, 4u);
            v = v + subgroupShuffleDown(v, 2u);
            v = v + subgroupShuffleDown(v, 1u);
            var tsum = 0.0;
            if (lid == 0u) {
                tsum = v;
            }
            lsum = lsum * corr + tsum;
        } else {
            for (var s = DG_ATTN_WG >> 1u; s > 0u; s = s >> 1u) {
                if (lid < s) {
                    dgx_red[lid] = dgx_red[lid] + dgx_red[lid + s];
                }
                workgroupBarrier();
            }
            lsum = lsum * corr + dgx_red[0];
        }
        let tmax = min(DG_ATTN_WG, hi - base);
        let tq = lid >> 6u;
        let wd = lid & 63u;
        var e0 = 0.0;
        var e1 = 0.0;
        var o0 = 0.0;
        var o1 = 0.0;
        if (wd < words && DGX_ABL != 5u) {
            let tend = min(tmax, (tq + 1u) * 64u);
            var t = tq * 64u;
            if (DGX_PIPE == 1u) {
                for (; t + 8u <= tend; t = t + 8u) {
                    let v0 = dgx_vld(((base + t) * nkv + kvh) * words + wd);
                    let v1 = dgx_vld(((base + t + 1u) * nkv + kvh) * words + wd);
                    let v2 = dgx_vld(((base + t + 2u) * nkv + kvh) * words + wd);
                    let v3 = dgx_vld(((base + t + 3u) * nkv + kvh) * words + wd);
                    let v4 = dgx_vld(((base + t + 4u) * nkv + kvh) * words + wd);
                    let v5 = dgx_vld(((base + t + 5u) * nkv + kvh) * words + wd);
                    let v6 = dgx_vld(((base + t + 6u) * nkv + kvh) * words + wd);
                    let v7 = dgx_vld(((base + t + 7u) * nkv + kvh) * words + wd);
                    e0 = fma(dgx_pr[t], dgkv_lo(v0), e0);
                    e1 = fma(dgx_pr[t], dgkv_hi(v0), e1);
                    o0 = fma(dgx_pr[t + 1u], dgkv_lo(v1), o0);
                    o1 = fma(dgx_pr[t + 1u], dgkv_hi(v1), o1);
                    e0 = fma(dgx_pr[t + 2u], dgkv_lo(v2), e0);
                    e1 = fma(dgx_pr[t + 2u], dgkv_hi(v2), e1);
                    o0 = fma(dgx_pr[t + 3u], dgkv_lo(v3), o0);
                    o1 = fma(dgx_pr[t + 3u], dgkv_hi(v3), o1);
                    e0 = fma(dgx_pr[t + 4u], dgkv_lo(v4), e0);
                    e1 = fma(dgx_pr[t + 4u], dgkv_hi(v4), e1);
                    o0 = fma(dgx_pr[t + 5u], dgkv_lo(v5), o0);
                    o1 = fma(dgx_pr[t + 5u], dgkv_hi(v5), o1);
                    e0 = fma(dgx_pr[t + 6u], dgkv_lo(v6), e0);
                    e1 = fma(dgx_pr[t + 6u], dgkv_hi(v6), e1);
                    o0 = fma(dgx_pr[t + 7u], dgkv_lo(v7), o0);
                    o1 = fma(dgx_pr[t + 7u], dgkv_hi(v7), o1);
                }
            }
            for (; t + 2u <= tend; t = t + 2u) {
                let v0 = dgx_vld(((base + t) * nkv + kvh) * words + wd);
                e0 = fma(dgx_pr[t], dgkv_lo(v0), e0);
                e1 = fma(dgx_pr[t], dgkv_hi(v0), e1);
                let v1 = dgx_vld(((base + t + 1u) * nkv + kvh) * words + wd);
                o0 = fma(dgx_pr[t + 1u], dgkv_lo(v1), o0);
                o1 = fma(dgx_pr[t + 1u], dgkv_hi(v1), o1);
            }
            if (t < tend) {
                let v0 = dgx_vld(((base + t) * nkv + kvh) * words + wd);
                e0 = fma(dgx_pr[t], dgkv_lo(v0), e0);
                e1 = fma(dgx_pr[t], dgkv_hi(v0), e1);
            }
        }
        dgx_pvp[tq * 128u + wd * 2u] = e0 + o0;
        dgx_pvp[tq * 128u + wd * 2u + 1u] = e1 + o1;
        workgroupBarrier();
        if (lid < 2u * words) {
            let sum = ((dgx_pvp[lid] + dgx_pvp[128u + lid])
                + (dgx_pvp[256u + lid] + dgx_pvp[384u + lid]));
            dgx_acc[lid] = dgx_acc[lid] * corr + sum;
        }
        m = mnew;
        workgroupBarrier();
    }

    if (lid == 0u) {
        dgx_part[ml] = m;
        dgx_part[lbase + ml] = lsum;
    }
    for (var i = lid; i < hd; i = i + DG_ATTN_WG) {
        dgx_part[abase + (h * hd + i) * dgx_p.n_splits + sp] = dgx_acc[i];
    }
}

const DG_FOLD_F: u32 = 1u;

var<workgroup> dgf_qs: array<f32, 1024>;
var<workgroup> dgf_acc: array<f32, 1024>;
var<workgroup> dgf_sc: array<f32, 2048>;

@compute @workgroup_size(256)
fn dg_attn_fold(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h0 = wg.x * DG_FOLD_F;
    let sp = wg.y;
    if (h0 >= dgx_p.n_q_heads) {
        return;
    }
    let lid = tid.x;
    let hd = dgx_p.head_dim;
    let words = hd >> 1u;
    let nkv = dgx_p.n_kv_heads;
    let kvh = h0 / dgx_p.gqa_group;
    let total = min(dgx_step[DG_STEP_POS] + 1u, dgx_p.kv_capacity);
    let per = dgx_per_slice(total);
    let lo = sp * per;
    let hi = min(lo + per, total);

    if (lo >= hi) {
        if (lid < DG_FOLD_F) {
            let pb = (h0 + lid) * dgx_p.n_splits + sp;
            dgx_part[pb] = -3.0e38;
            dgx_part[dgx_p.n_q_heads * dgx_p.n_splits + pb] = 0.0;
        }
        return;
    }

    let span = DG_FOLD_F * hd;
    for (var i = lid; i < span; i = i + DG_ATTN_WG) {
        let g = i / hd;
        let d = i - g * hd;
        let w = dgx_q[(h0 + g) * words + (d >> 1u)];
        dgf_qs[i] = select(bf16_lo(w), bf16_hi(w), (d & 1u) == 1u);
        dgf_acc[i] = 0.0;
    }
    workgroupBarrier();

    var mx: array<f32, 8>;
    for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
        mx[g] = -3.0e38;
    }
    for (var j = lo + lid; j < hi; j = j + DG_ATTN_WG) {
        var sc: array<f32, 8>;
        for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
            sc[g] = 0.0;
        }
        for (var w = 0u; w < words; w = w + 1u) {
            let kw = dgx_k[dg_kword(j, kvh, w, nkv * words, words, dgx_p.kv_capacity)];
            let klo = dgkv_lo(kw);
            let khi = dgkv_hi(kw);
            let d0 = w * 2u;
            for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                sc[g] = fma(dgf_qs[g * hd + d0], klo, sc[g]);
                sc[g] = fma(dgf_qs[g * hd + d0 + 1u], khi, sc[g]);
            }
        }
        for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
            mx[g] = max(mx[g], sc[g] * dgx_p.scale);
        }
    }
    for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
        dgf_sc[g * DG_ATTN_WG + lid] = mx[g];
    }
    workgroupBarrier();
    for (var st = DG_ATTN_WG >> 1u; st > 0u; st = st >> 1u) {
        if (lid < st) {
            for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                let b0 = g * DG_ATTN_WG + lid;
                dgf_sc[b0] = max(dgf_sc[b0], dgf_sc[b0 + st]);
            }
        }
        workgroupBarrier();
    }
    var mm: array<f32, 8>;
    for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
        mm[g] = dgf_sc[g * DG_ATTN_WG];
    }
    workgroupBarrier();

    var lsum: array<f32, 8>;
    for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
        lsum[g] = 0.0;
    }
    for (var base = lo; base < hi; base = base + DG_ATTN_WG) {
        let j = base + lid;
        if (j < hi) {
            var sc: array<f32, 8>;
            for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                sc[g] = 0.0;
            }
            for (var w = 0u; w < words; w = w + 1u) {
                let kw = dgx_k[dg_kword(j, kvh, w, nkv * words, words, dgx_p.kv_capacity)];
                let klo = dgkv_lo(kw);
                let khi = dgkv_hi(kw);
                let d0 = w * 2u;
                for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                    sc[g] = fma(dgf_qs[g * hd + d0], klo, sc[g]);
                    sc[g] = fma(dgf_qs[g * hd + d0 + 1u], khi, sc[g]);
                }
            }
            for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                let pv = exp(sc[g] * dgx_p.scale - mm[g]);
                lsum[g] = lsum[g] + pv;
                dgf_sc[g * DG_ATTN_WG + lid] = pv;
            }
        } else {
            for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                dgf_sc[g * DG_ATTN_WG + lid] = 0.0;
            }
        }
        workgroupBarrier();
        let tmax = min(DG_ATTN_WG, hi - base);
        for (var d = lid; d < hd; d = d + DG_ATTN_WG) {
            var a: array<f32, 8>;
            for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                a[g] = 0.0;
            }
            for (var t = 0u; t < tmax; t = t + 1u) {
                let vv = dgx_vval(((base + t) * nkv + kvh) * hd + d);
                for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                    a[g] = fma(dgf_sc[g * DG_ATTN_WG + t], vv, a[g]);
                }
            }
            for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                dgf_acc[g * hd + d] = dgf_acc[g * hd + d] + a[g];
            }
        }
        workgroupBarrier();
    }

    for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
        dgf_sc[g * DG_ATTN_WG + lid] = lsum[g];
    }
    workgroupBarrier();
    for (var st = DG_ATTN_WG >> 1u; st > 0u; st = st >> 1u) {
        if (lid < st) {
            for (var g = 0u; g < DG_FOLD_F; g = g + 1u) {
                let b0 = g * DG_ATTN_WG + lid;
                dgf_sc[b0] = dgf_sc[b0] + dgf_sc[b0 + st];
            }
        }
        workgroupBarrier();
    }

    if (lid < DG_FOLD_F) {
        let pb = (h0 + lid) * dgx_p.n_splits + sp;
        dgx_part[pb] = mm[lid];
        dgx_part[dgx_p.n_q_heads * dgx_p.n_splits + pb] = dgf_sc[lid * DG_ATTN_WG];
    }
    for (var i = lid; i < span; i = i + DG_ATTN_WG) {
        let g = i / hd;
        let d = i - g * hd;
        dgx_part[2u * dgx_p.n_q_heads * dgx_p.n_splits
            + ((h0 + g) * dgx_p.head_dim + d) * dgx_p.n_splits + sp] = dgf_acc[i];
    }
}

var<workgroup> dgxc_m: array<f32, 64>;
var<workgroup> dgxc_l: array<f32, 64>;
var<workgroup> dgxc_w: array<f32, 64>;

@compute @workgroup_size(256)
fn dg_attn_combine(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= dgx_p.n_q_heads || DGX_ABL == 6u) {
        return;
    }
    let lid = tid.x;
    let hd = dgx_p.head_dim;
    let words = hd >> 1u;
    let ns = dgx_p.n_splits;
    let total = min(dgx_step[DG_STEP_POS] + 1u, dgx_p.kv_capacity);
    let per = dgx_per_slice(total);
    let live = min(ns, (total + per - 1u) / per);
    let lbase = dgx_p.n_q_heads * ns;
    let abase = 2u * lbase;

    if (lid < live) {
        dgxc_m[lid] = dgx_part[h * ns + lid];
        dgxc_l[lid] = dgx_part[lbase + h * ns + lid];
    }
    workgroupBarrier();
    var big = -3.0e38;
    for (var s = 0u; s < live; s = s + 1u) {
        big = max(big, dgxc_m[s]);
    }
    if (lid < live) {
        dgxc_w[lid] = exp(dgxc_m[lid] - big);
    }
    workgroupBarrier();
    var den = 0.0;
    for (var s = 0u; s < live; s = s + 1u) {
        den = den + dgxc_w[s] * dgxc_l[s];
    }
    var inv = 0.0;
    if (den > 0.0) {
        inv = 1.0 / den;
    }

    for (var i = lid; i < words; i = i + DG_ATTN_WG) {
        let e = i * 2u;
        let r0 = abase + (h * hd + e) * ns;
        let r1 = r0 + ns;
        var a0 = 0.0;
        var a1 = 0.0;
        for (var s = 0u; s < live; s = s + 1u) {
            let w = dgxc_w[s];
            a0 = fma(w, dgx_part[r0 + s], a0);
            a1 = fma(w, dgx_part[r1 + s], a1);
        }
        dgx_out[h * words + i] = dg_act_pack(a0 * inv, a1 * inv);
    }
}

struct DgG4Params {
    n_rows: u32,
    k_vecs: u32,
    w_row_vecs: u32,
    groups_x: u32,
    out_f32: u32,
    y_off_words: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(120) var<storage, read> dg4_w: array<vec4<u32>>;
@group(0) @binding(121) var<storage, read> dg4_x: array<vec4<u32>>;
@group(0) @binding(122) var<uniform> dg4_p: DgG4Params;
@group(0) @binding(123) var<storage, read_write> dg4_y: array<u32>;

var<workgroup> dg4_red: array<f32, 256>;

const DG4_ROWS: u32 = 2u;
const DG4_LANES: u32 = 256u / DG4_ROWS;

@compute @workgroup_size(256)
fn dg_gemv_bf16_v4(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let rh = tid / DG4_LANES;
    let lane = tid % DG4_LANES;
    let pair = wid.x + wid.y * dg4_p.groups_x;
    let row = pair * DG4_ROWS + rh;
    let live = row < dg4_p.n_rows;
    let wbase = select(0u, row * dg4_p.w_row_vecs, live);
    let kv = select(0u, dg4_p.k_vecs, live);

    var acc = 0.0;
    for (var i = lane; i < kv; i = i + DG4_LANES) {
        let wv = dg4_w[wbase + i];
        let xv = dg4_x[i];
        acc = fma(bf16_lo(wv.x), bf16_lo(xv.x), acc);
        acc = fma(bf16_hi(wv.x), bf16_hi(xv.x), acc);
        acc = fma(bf16_lo(wv.y), bf16_lo(xv.y), acc);
        acc = fma(bf16_hi(wv.y), bf16_hi(xv.y), acc);
        acc = fma(bf16_lo(wv.z), bf16_lo(xv.z), acc);
        acc = fma(bf16_hi(wv.z), bf16_hi(xv.z), acc);
        acc = fma(bf16_lo(wv.w), bf16_lo(xv.w), acc);
        acc = fma(bf16_hi(wv.w), bf16_hi(xv.w), acc);
    }
    dg4_red[tid] = acc;
    workgroupBarrier();
    for (var stride = DG4_LANES >> 1u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            dg4_red[tid] = dg4_red[tid] + dg4_red[tid + stride];
        }
        workgroupBarrier();
    }
    if (dg4_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            dg4_y[dg4_p.y_off_words + row] = bitcast<u32>(dg4_red[tid]);
        }
    } else if (tid == 0u) {
        for (var rp = 0u; rp < DG4_ROWS; rp = rp + 2u) {
            let r0 = pair * DG4_ROWS + rp;
            if (r0 >= dg4_p.n_rows) {
                break;
            }
            let lo = dg4_red[rp * DG4_LANES];
            var hi = 0.0;
            if (r0 + 1u < dg4_p.n_rows) {
                hi = dg4_red[(rp + 1u) * DG4_LANES];
            }
            dg4_y[dg4_p.y_off_words + (r0 >> 1u)] = bf16_pack(lo, hi);
        }
    }
}

struct DgTileParams {
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    gqa_group: u32,
    kv_capacity: u32,
    chunk: u32,
    q_slot_words: u32,
    out_slot_words: u32,
    kv_words: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(70) var<storage, read> dgt_qsrc: array<u32>;
@group(0) @binding(71) var<storage, read> dgt_ksrc: array<u32>;
@group(0) @binding(72) var<storage, read> dgt_vsrc: array<u32>;
@group(0) @binding(73) var<storage, read> dgt_step: array<u32>;
@group(0) @binding(74) var<uniform> dgt_p: DgTileParams;
@group(0) @binding(75) var<storage, read_write> dgt_out: array<u32>;

const DGT_QT: u32 = 8u;
const DGT_KT: u32 = 32u;
const DGT_HD: u32 = 128u;
const DGT_WG: u32 = 256u;

var<workgroup> dgt_k: array<u32, 2048>;
var<workgroup> dgt_v: array<u32, 2048>;
var<workgroup> dgt_q: array<f32, 1024>;
var<workgroup> dgt_acc: array<f32, 1024>;
var<workgroup> dgt_s: array<f32, 256>;
var<workgroup> dgt_m: array<f32, 8>;
var<workgroup> dgt_l: array<f32, 8>;
var<workgroup> dgt_corr: array<f32, 8>;

fn dgt_tileval(tile_word: u32, elem: u32) -> f32 {
    let w = dgt_k[tile_word + (elem >> 1u)];
    return select(dgkv_lo(w), dgkv_hi(w), (elem & 1u) == 1u);
}

fn dgt_vtileval(tile_word: u32, elem: u32) -> f32 {
    let w = dgt_v[tile_word + (elem >> 1u)];
    return select(dgkv_lo(w), dgkv_hi(w), (elem & 1u) == 1u);
}

@compute @workgroup_size(256)
fn dg_attn_tiled(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= dgt_p.n_q_heads) {
        return;
    }
    let lid = tid.x;
    let hd = dgt_p.head_dim;
    let words = hd >> 1u;
    let kvh = h / dgt_p.gqa_group;
    let base = dgt_step[DG_STEP_POS];
    let s0 = wg.y * DGT_QT;

    for (var i = lid; i < DGT_QT * hd; i = i + DGT_WG) {
        let q = i / hd;
        let d = i - q * hd;
        let slot = s0 + q;
        var val = 0.0;
        if (slot < dgt_p.chunk) {
            let w = dgt_qsrc[slot * dgt_p.q_slot_words + h * words + (d >> 1u)];
            val = select(bf16_lo(w), bf16_hi(w), (d & 1u) == 1u);
        }
        dgt_q[q * hd + d] = val;
        dgt_acc[q * hd + d] = 0.0;
    }
    if (lid < DGT_QT) {
        dgt_m[lid] = -3.0e38;
        dgt_l[lid] = 0.0;
    }
    workgroupBarrier();

    let last = min(base + s0 + DGT_QT, dgt_p.kv_capacity);
    for (var kb = 0u; kb < last; kb = kb + DGT_KT) {
        let n = min(DGT_KT, last - kb);
        for (var i = lid; i < n * words; i = i + DGT_WG) {
            let t = i / words;
            let w = i - t * words;
            let src = (kb + t) * dgt_p.kv_words + kvh * words + w;
            dgt_k[t * words + w] =
                dgt_ksrc[dg_kword(kb + t, kvh, w, dgt_p.kv_words, words, dgt_p.kv_capacity)];
            dgt_v[t * words + w] = dgt_vsrc[src];
        }
        workgroupBarrier();

        for (var i = lid; i < DGT_QT * n; i = i + DGT_WG) {
            let q = i / n;
            let j = i - q * n;
            let slot = s0 + q;
            var sc = -3.0e38;
            if (slot < dgt_p.chunk && kb + j <= base + slot) {
                var a = 0.0;
                for (var d = 0u; d < hd; d = d + 1u) {
                    a = fma(dgt_q[q * hd + d], dgt_tileval(j * words, d), a);
                }
                sc = a * dgt_p.scale;
            }
            dgt_s[q * DGT_KT + j] = sc;
        }
        workgroupBarrier();

        if (lid < DGT_QT) {
            var rowmax = -3.0e38;
            for (var j = 0u; j < n; j = j + 1u) {
                rowmax = max(rowmax, dgt_s[lid * DGT_KT + j]);
            }
            let mnew = max(dgt_m[lid], rowmax);
            var corr = 0.0;
            if (dgt_m[lid] > -2.0e38) {
                corr = exp(dgt_m[lid] - mnew);
            }
            var rowsum = 0.0;
            for (var j = 0u; j < n; j = j + 1u) {
                let p = exp(dgt_s[lid * DGT_KT + j] - mnew);
                dgt_s[lid * DGT_KT + j] = p;
                rowsum = rowsum + p;
            }
            dgt_m[lid] = mnew;
            dgt_l[lid] = dgt_l[lid] * corr + rowsum;
            dgt_corr[lid] = corr;
        }
        workgroupBarrier();

        for (var i = lid; i < DGT_QT * hd; i = i + DGT_WG) {
            let q = i / hd;
            let d = i - q * hd;
            var a = 0.0;
            for (var j = 0u; j < n; j = j + 1u) {
                a = fma(dgt_s[q * DGT_KT + j], dgt_vtileval(j * words, d), a);
            }
            dgt_acc[i] = dgt_acc[i] * dgt_corr[q] + a;
        }
        workgroupBarrier();
    }

    for (var i = lid; i < DGT_QT * words; i = i + DGT_WG) {
        let q = i / words;
        let w = i - q * words;
        let slot = s0 + q;
        if (slot >= dgt_p.chunk) {
            continue;
        }
        var inv = 0.0;
        if (dgt_l[q] > 0.0) {
            inv = 1.0 / dgt_l[q];
        }
        let e = w * 2u;
        dgt_out[slot * dgt_p.out_slot_words + h * words + w] =
            bf16_pack(dgt_acc[q * hd + e] * inv, dgt_acc[q * hd + e + 1u] * inv);
    }
}

@group(0) @binding(80) var<storage, read> dgw_qsrc: array<u32>;
@group(0) @binding(81) var<storage, read> dgw_ksrc: array<u32>;
@group(0) @binding(82) var<storage, read> dgw_vsrc: array<u32>;
@group(0) @binding(83) var<storage, read> dgw_step: array<u32>;
@group(0) @binding(84) var<uniform> dgw_p: DgTileParams;
@group(0) @binding(85) var<storage, read_write> dgw_out: array<u32>;

const DGW_QT: u32 = 8u;
const DGW_KT: u32 = 256u;
const DGW_WG: u32 = 256u;
const DGW_ABLATE: u32 = 0u;

var<workgroup> dgw_q: array<f32, 1024>;
var<workgroup> dgw_p_tile: array<f32, 2048>;
var<workgroup> dgw_acc: array<f32, 1024>;
var<workgroup> dgw_m: array<f32, 8>;
var<workgroup> dgw_l: array<f32, 8>;
var<workgroup> dgw_corr: array<f32, 8>;

@compute @workgroup_size(256)
fn dg_attn_reg(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    if (h >= dgw_p.n_q_heads) {
        return;
    }
    let lid = tid.x;
    let hd = dgw_p.head_dim;
    let words = hd >> 1u;
    let kvh = h / dgw_p.gqa_group;
    let base = dgw_step[DG_STEP_POS];
    let s0 = wg.y * DGW_QT;

    for (var i = lid; i < DGW_QT * hd; i = i + DGW_WG) {
        let q = i / hd;
        let d = i - q * hd;
        let slot = s0 + q;
        var val = 0.0;
        if (slot < dgw_p.chunk) {
            let w = dgw_qsrc[slot * dgw_p.q_slot_words + h * words + (d >> 1u)];
            val = select(bf16_lo(w), bf16_hi(w), (d & 1u) == 1u);
        }
        dgw_q[q * hd + d] = val;
        dgw_acc[q * hd + d] = 0.0;
    }
    if (lid < DGW_QT) {
        dgw_m[lid] = -3.0e38;
        dgw_l[lid] = 0.0;
        dgw_corr[lid] = 1.0;
    }
    workgroupBarrier();

    let last = min(base + s0 + DGW_QT, dgw_p.kv_capacity);
    for (var kb = 0u; kb < last; kb = kb + DGW_KT) {
        let n = min(DGW_KT, last - kb);
        let j = kb + lid;
        var s0v = 0.0;
        var s1v = 0.0;
        var s2v = 0.0;
        var s3v = 0.0;
        var s4v = 0.0;
        var s5v = 0.0;
        var s6v = 0.0;
        var s7v = 0.0;
        if (DGW_ABLATE != 1u && lid < n) {
            for (var w = 0u; w < words; w = w + 1u) {
                let kw =
                    dgw_ksrc[dg_kword(j, kvh, w, dgw_p.kv_words, words, dgw_p.kv_capacity)];
                let klo = dgkv_lo(kw);
                let khi = dgkv_hi(kw);
                let d0 = w * 2u;
                s0v = fma(dgw_q[d0], klo, fma(dgw_q[d0 + 1u], khi, s0v));
                s1v = fma(dgw_q[hd + d0], klo, fma(dgw_q[hd + d0 + 1u], khi, s1v));
                s2v = fma(dgw_q[2u * hd + d0], klo, fma(dgw_q[2u * hd + d0 + 1u], khi, s2v));
                s3v = fma(dgw_q[3u * hd + d0], klo, fma(dgw_q[3u * hd + d0 + 1u], khi, s3v));
                s4v = fma(dgw_q[4u * hd + d0], klo, fma(dgw_q[4u * hd + d0 + 1u], khi, s4v));
                s5v = fma(dgw_q[5u * hd + d0], klo, fma(dgw_q[5u * hd + d0 + 1u], khi, s5v));
                s6v = fma(dgw_q[6u * hd + d0], klo, fma(dgw_q[6u * hd + d0 + 1u], khi, s6v));
                s7v = fma(dgw_q[7u * hd + d0], klo, fma(dgw_q[7u * hd + d0 + 1u], khi, s7v));
            }
        }
        let sc = dgw_p.scale;
        let lim = base + s0;
        let msk = select(-3.0e38, 0.0, DGW_ABLATE == 2u);
        dgw_p_tile[lid] = select(msk, s0v * sc, lid < n && j <= lim);
        dgw_p_tile[DGW_KT + lid] = select(msk, s1v * sc, lid < n && j <= lim + 1u);
        dgw_p_tile[2u * DGW_KT + lid] = select(msk, s2v * sc, lid < n && j <= lim + 2u);
        dgw_p_tile[3u * DGW_KT + lid] = select(msk, s3v * sc, lid < n && j <= lim + 3u);
        dgw_p_tile[4u * DGW_KT + lid] = select(msk, s4v * sc, lid < n && j <= lim + 4u);
        dgw_p_tile[5u * DGW_KT + lid] = select(msk, s5v * sc, lid < n && j <= lim + 5u);
        dgw_p_tile[6u * DGW_KT + lid] = select(msk, s6v * sc, lid < n && j <= lim + 6u);
        dgw_p_tile[7u * DGW_KT + lid] = select(msk, s7v * sc, lid < n && j <= lim + 7u);
        workgroupBarrier();

        if (DGW_ABLATE != 2u && lid < DGW_QT) {
            let rowbase = lid * DGW_KT;
            var rowmax = -3.0e38;
            for (var t = 0u; t < n; t = t + 1u) {
                rowmax = max(rowmax, dgw_p_tile[rowbase + t]);
            }
            let mnew = max(dgw_m[lid], rowmax);
            var corr = 0.0;
            if (dgw_m[lid] > -2.0e38) {
                corr = exp(dgw_m[lid] - mnew);
            }
            var rowsum = 0.0;
            for (var t = 0u; t < n; t = t + 1u) {
                var pv = 0.0;
                if (dgw_p_tile[rowbase + t] > -2.0e38) {
                    pv = exp(dgw_p_tile[rowbase + t] - mnew);
                }
                dgw_p_tile[rowbase + t] = pv;
                rowsum = rowsum + pv;
            }
            dgw_m[lid] = mnew;
            dgw_l[lid] = dgw_l[lid] * corr + rowsum;
            dgw_corr[lid] = corr;
        }
        workgroupBarrier();

        if (DGW_ABLATE != 3u && lid < hd) {
            let d = lid;
            var a0 = 0.0;
            var a1 = 0.0;
            var a2 = 0.0;
            var a3 = 0.0;
            var a4 = 0.0;
            var a5 = 0.0;
            var a6 = 0.0;
            var a7 = 0.0;
            for (var t = 0u; t < n; t = t + 1u) {
                let vw = dgw_vsrc[(kb + t) * dgw_p.kv_words + kvh * words + (d >> 1u)];
                let vv = select(dgkv_lo(vw), dgkv_hi(vw), (d & 1u) == 1u);
                a0 = fma(dgw_p_tile[t], vv, a0);
                a1 = fma(dgw_p_tile[DGW_KT + t], vv, a1);
                a2 = fma(dgw_p_tile[2u * DGW_KT + t], vv, a2);
                a3 = fma(dgw_p_tile[3u * DGW_KT + t], vv, a3);
                a4 = fma(dgw_p_tile[4u * DGW_KT + t], vv, a4);
                a5 = fma(dgw_p_tile[5u * DGW_KT + t], vv, a5);
                a6 = fma(dgw_p_tile[6u * DGW_KT + t], vv, a6);
                a7 = fma(dgw_p_tile[7u * DGW_KT + t], vv, a7);
            }
            dgw_acc[d] = dgw_acc[d] * dgw_corr[0] + a0;
            dgw_acc[hd + d] = dgw_acc[hd + d] * dgw_corr[1] + a1;
            dgw_acc[2u * hd + d] = dgw_acc[2u * hd + d] * dgw_corr[2] + a2;
            dgw_acc[3u * hd + d] = dgw_acc[3u * hd + d] * dgw_corr[3] + a3;
            dgw_acc[4u * hd + d] = dgw_acc[4u * hd + d] * dgw_corr[4] + a4;
            dgw_acc[5u * hd + d] = dgw_acc[5u * hd + d] * dgw_corr[5] + a5;
            dgw_acc[6u * hd + d] = dgw_acc[6u * hd + d] * dgw_corr[6] + a6;
            dgw_acc[7u * hd + d] = dgw_acc[7u * hd + d] * dgw_corr[7] + a7;
        }
        workgroupBarrier();
    }

    for (var i = lid; i < DGW_QT * words; i = i + DGW_WG) {
        let q = i / words;
        let w = i - q * words;
        let slot = s0 + q;
        if (slot >= dgw_p.chunk) {
            continue;
        }
        var inv = 0.0;
        if (dgw_l[q] > 0.0) {
            inv = 1.0 / dgw_l[q];
        }
        let e = w * 2u;
        dgw_out[slot * dgw_p.out_slot_words + h * words + w] =
            bf16_pack(dgw_acc[q * hd + e] * inv, dgw_acc[q * hd + e + 1u] * inv);
    }
}

const DG_STEP_KVBASE: u32 = 3u;

@compute @workgroup_size(64)
fn dg_kv_write_r(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let i = wg.x * 64u + tid.x;
    if (i >= dgk_p.kv_words) {
        return;
    }
    let slot = wg.y;
    let cache = dgk_step[DG_STEP_KVBASE] + dgk_step[DG_STEP_POS] + slot;
    if (cache >= dgk_p.kv_capacity) {
        return;
    }
    let src = slot * dgk_p.src_slot_words;
    let at = cache * dgk_p.kv_words + i;
    let words = dgk_p.kv_words / dgk_p.n_kv_heads;
    let kat = dg_kword(cache, i / words, i % words, dgk_p.kv_words, words, dgk_p.kv_capacity);
    dgk_kc[kat] = dgkv_store(dgk_src[src + dgk_p.k_off_words + i]);
    dgk_vc[at] = dgkv_store(dgk_src[src + dgk_p.v_off_words + i]);
}

@group(0) @binding(15) var<storage, read> dgk_bslot: array<u32>;

@compute @workgroup_size(64)
fn dg_kv_write_b(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let i = wg.x * 64u + tid.x;
    if (i >= dgk_p.kv_words) {
        return;
    }
    let slot = wg.y;
    let cache = dgk_bslot[slot];
    if (cache >= dgk_p.kv_capacity) {
        return;
    }
    let src = slot * dgk_p.src_slot_words;
    let at = cache * dgk_p.kv_words + i;
    let words = dgk_p.kv_words / dgk_p.n_kv_heads;
    let kat = dg_kword(cache, i / words, i % words, dgk_p.kv_words, words, dgk_p.kv_capacity);
    dgk_kc[kat] = dgkv_store(dgk_src[src + dgk_p.k_off_words + i]);
    dgk_vc[at] = dgkv_store(dgk_src[src + dgk_p.v_off_words + i]);
}

@group(0) @binding(7) var<storage, read> dgr_bpos: array<u32>;

@compute @workgroup_size(64)
fn dg_bias_rope_b(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x;
    let slot = wg.y;
    if (row >= dgr_p.n_rows) {
        return;
    }
    let lid = tid.x;
    let hd = dgr_p.head_dim;
    let words = hd >> 1u;
    let src_base = slot * dgr_p.src_slot_words;
    let dst_base = slot * dgr_p.dst_slot_words + row * words;
    for (var e = lid; e < hd; e = e + DG_ROPE_WG) {
        var v = 0.0;
        if (dgr_p.src_f32 == 1u) {
            v = bitcast<f32>(dgr_src[src_base * 2u + row * hd + e]);
        } else {
            let xw = dgr_src[src_base + row * words + (e >> 1u)];
            v = select(bf16_lo(xw), bf16_hi(xw), (e & 1u) == 1u);
        }
        if (dgr_p.has_bias == 1u) {
            let bw = dgr_bias[row * words + (e >> 1u)];
            v = v + select(bf16_lo(bw), bf16_hi(bw), (e & 1u) == 1u);
        }
        dgr_a[e] = v;
    }
    workgroupBarrier();
    let row_base = dgr_bpos[slot] * dgr_p.half;
    let rope = row < dgr_p.rope_rows;
    for (var i = lid; i < words; i = i + DG_ROPE_WG) {
        let e = i * 2u;
        dgr_dst[dst_base + i] = bf16_pack(dgr_rot(e, row_base, rope), dgr_rot(e + 1u, row_base, rope));
    }
}

@compute @workgroup_size(64)
fn dg_bias_rope_kvw_b(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x;
    let slot = wg.y;
    if (row >= dgr_p.n_rows) {
        return;
    }
    let lid = tid.x;
    let hd = dgr_p.head_dim;
    let words = hd >> 1u;
    let src_base = slot * dgr_p.src_slot_words;
    let dst_base = slot * dgr_p.dst_slot_words + row * words;
    for (var e = lid; e < hd; e = e + DG_ROPE_WG) {
        var v = 0.0;
        if (dgr_p.src_f32 == 1u) {
            v = bitcast<f32>(dgr_src[src_base * 2u + row * hd + e]);
        } else {
            let xw = dgr_src[src_base + row * words + (e >> 1u)];
            v = select(bf16_lo(xw), bf16_hi(xw), (e & 1u) == 1u);
        }
        if (dgr_p.has_bias == 1u) {
            let bw = dgr_bias[row * words + (e >> 1u)];
            v = v + select(bf16_lo(bw), bf16_hi(bw), (e & 1u) == 1u);
        }
        dgr_a[e] = v;
    }
    workgroupBarrier();
    let row_base = dgr_bpos[slot] * dgr_p.half;
    let rope = row < dgr_p.rope_rows;
    let n_kv = dgr_p.n_rows - dgr_p.rope_rows;
    let k_first = dgr_p.rope_rows - n_kv;
    let cache = dgk_bslot[slot];
    for (var i = lid; i < words; i = i + DG_ROPE_WG) {
        let e = i * 2u;
        let packed = bf16_pack(dgr_rot(e, row_base, rope), dgr_rot(e + 1u, row_base, rope));
        dgr_dst[dst_base + i] = packed;
        if (row >= k_first && cache < dgk_p.kv_capacity) {
            if (row < dgr_p.rope_rows) {
                let kat = dg_kword(cache, row - k_first, i, dgk_p.kv_words, words, dgk_p.kv_capacity);
                dgk_kc[kat] = dgkv_store(packed);
            } else {
                dgk_vc[cache * dgk_p.kv_words + (row - dgr_p.rope_rows) * words + i] = dgkv_store(packed);
            }
        }
    }
}

fn dga_range_attend(slot: u32, h: u32, lid: u32, start: u32, total: u32) {
    if (DGA_V2 == 1u) {
        dga_range_attend_v2(slot, h, lid, start, total);
        return;
    }
    let hd = dga_p.head_dim;
    let words = hd >> 1u;
    let nkv = dga_p.n_kv_heads;
    let kvh = h / dga_p.gqa_group;
    let qbase = slot * dga_p.q_slot_words + h * words;

    for (var i = lid; i < hd; i = i + DG_ATTN_WG) {
        let w = dga_q[qbase + (i >> 1u)];
        dga_qs[i] = select(bf16_lo(w), bf16_hi(w), (i & 1u) == 1u);
        dga_acc[i] = 0.0;
    }
    workgroupBarrier();

    var lmax = -3.0e38;
    for (var j = start + lid; j < total; j = j + DG_ATTN_WG) {
        lmax = max(lmax, dga_score(j, kvh, hd) * dga_p.scale);
    }
    dga_red[lid] = lmax;
    workgroupBarrier();
    for (var s = DG_ATTN_WG >> 1u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            dga_red[lid] = max(dga_red[lid], dga_red[lid + s]);
        }
        workgroupBarrier();
    }
    let m = dga_red[0];

    var lsum = 0.0;
    for (var base = start; base < total; base = base + DG_ATTN_WG) {
        let j = base + lid;
        var pv = 0.0;
        if (j < total) {
            pv = exp(dga_score(j, kvh, hd) * dga_p.scale - m);
        }
        lsum = lsum + pv;
        dga_pr[lid] = pv;
        workgroupBarrier();
        let tmax = min(DG_ATTN_WG, total - base);
        for (var d = lid; d < hd; d = d + DG_ATTN_WG) {
            var a = 0.0;
            for (var t = 0u; t < tmax; t = t + 1u) {
                a = fma(dga_pr[t], dga_vval(((base + t) * nkv + kvh) * hd + d), a);
            }
            dga_acc[d] = dga_acc[d] + a;
        }
        workgroupBarrier();
    }

    dga_red[lid] = lsum;
    workgroupBarrier();
    for (var s = DG_ATTN_WG >> 1u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            dga_red[lid] = dga_red[lid] + dga_red[lid + s];
        }
        workgroupBarrier();
    }
    var inv = 0.0;
    if (dga_red[0] > 0.0) {
        inv = 1.0 / dga_red[0];
    }
    let obase = slot * dga_p.out_slot_words + h * words;
    for (var i = lid; i < words; i = i + DG_ATTN_WG) {
        let e = i * 2u;
        dga_out[obase + i] = dg_act_pack(dga_acc[e] * inv, dga_acc[e + 1u] * inv);
    }
}

@compute @workgroup_size(256)
fn dg_attn_r(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let slot = wg.y;
    if (h >= dga_p.n_q_heads) {
        return;
    }
    let start = dga_step[DG_STEP_KVBASE];
    let total = min(start + dga_step[DG_STEP_POS] + slot + 1u, dga_p.kv_capacity);
    dga_range_attend(slot, h, tid.x, start, total);
}

@group(0) @binding(36) var<storage, read> dga_brange: array<u32>;

@compute @workgroup_size(256)
fn dg_attn_b(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let h = wg.x;
    let slot = wg.y;
    if (h >= dga_p.n_q_heads) {
        return;
    }
    let start = dga_brange[2u * slot];
    let total = min(start + dga_brange[2u * slot + 1u], dga_p.kv_capacity);
    dga_range_attend(slot, h, tid.x, start, total);
}

struct DgRowGatherParams {
    words: u32,
    src_slot_words: u32,
    row: u32,
    pad0: u32,
};

@group(0) @binding(90) var<storage, read> dgrg_src: array<u32>;
@group(0) @binding(91) var<uniform> dgrg_p: DgRowGatherParams;
@group(0) @binding(92) var<storage, read_write> dgrg_dst: array<u32>;

@compute @workgroup_size(64)
fn dg_row_gather_b(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let i = wg.x * 64u + tid.x;
    if (i >= dgrg_p.words) {
        return;
    }
    let w = dgrg_src[dgrg_p.row * dgrg_p.src_slot_words + i];
    dgrg_dst[i] = dg_act_pack(bf16_lo(w), bf16_hi(w));
}

@group(0) @binding(93) var<storage, read> dgrg_tok: array<u32>;
@group(0) @binding(94) var<storage, read_write> dgrg_btoks: array<u32>;

@compute @workgroup_size(1)
fn dg_tok_stash_b(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let _i = wg.x + tid.x;
    dgrg_btoks[dgrg_p.row] = dgrg_tok[0];
}

"#;

pub const DG_ATTN_FUSED_WGSL: &str = r#"
struct DgFusedParams {
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    gqa_group: u32,
    kv_capacity: u32,
    chunk: u32,
    q_slot_words: u32,
    out_slot_words: u32,
    kv_words: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(90) var<storage, read> dgf_qsrc: array<u32>;
@group(0) @binding(91) var<storage, read> dgf_ksrc: array<u32>;
@group(0) @binding(92) var<storage, read> dgf_vsrc: array<u32>;
@group(0) @binding(93) var<storage, read> dgf_step: array<u32>;
@group(0) @binding(94) var<uniform> dgf_p: DgFusedParams;
@group(0) @binding(95) var<storage, read_write> dgf_out: array<u32>;
@group(0) @binding(80) var<storage, read_write> dgf_qa: array<f32>;
@group(0) @binding(81) var<storage, read_write> dgf_qb: array<f32>;
@group(0) @binding(82) var<storage, read_write> dgf_kf0: array<f32>;
@group(0) @binding(83) var<storage, read_write> dgf_kf1: array<f32>;
@group(0) @binding(84) var<storage, read_write> dgf_vt0: array<f32>;
@group(0) @binding(85) var<storage, read_write> dgf_vt1: array<f32>;
@group(0) @binding(86) var<storage, read_write> dgf_s: array<f32>;
@group(0) @binding(88) var<storage, read_write> dgf_l: array<f32>;
@group(0) @binding(96) var<storage, read> dgf_oa: array<f32>;
@group(0) @binding(98) var<storage, read> dgf_ob: array<f32>;
@group(0) @binding(87) var<storage, read> dgf_la: array<f32>;
@group(0) @binding(89) var<storage, read> dgf_lb: array<f32>;
const DGF_STEP_POS: u32 = 1u;
const DGF_KT: u32 = 0u;
const DGF_CAUSAL: u32 = 1u;
const DGF_LOG2E: f32 = 1.4426950408889634;

const DGF_KV16: u32 = 0u;

fn dgf_kv_lo(w: u32) -> f32 {
    if (DGF_KV16 == 1u) {
        return unpack2x16float(w).x;
    }
    return bf16_lo(w);
}

fn dgf_kv_hi(w: u32) -> f32 {
    if (DGF_KV16 == 1u) {
        return unpack2x16float(w).y;
    }
    return bf16_hi(w);
}

fn dgf_kword(j: u32, kvh: u32, w: u32) -> u32 {
    if (DGF_KT == 1u) {
        return (kvh * (dgf_p.head_dim >> 1u) + w) * dgf_p.kv_capacity + j;
    }
    return j * dgf_p.kv_words + kvh * (dgf_p.head_dim >> 1u) + w;
}

fn dgf_cap16() -> u32 {
    return ((dgf_p.kv_capacity + 15u) / 16u) * 16u;
}

@compute @workgroup_size(64)
fn dg_qpack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let words = dgf_p.head_dim >> 1u;
    let n = dgf_p.chunk * dgf_p.n_q_heads * words;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let slot = i / (dgf_p.n_q_heads * words);
    let r = i - slot * dgf_p.n_q_heads * words;
    let h = r / words;
    let w = r - h * words;
    let word = dgf_qsrc[slot * dgf_p.q_slot_words + h * words + w];
    let hg = h % dgf_p.gqa_group;
    let dst = (hg * dgf_p.chunk + slot) * dgf_p.head_dim + w * 2u;
    if (h < dgf_p.gqa_group) {
        dgf_qa[dst] = bf16_lo(word);
        dgf_qa[dst + 1u] = bf16_hi(word);
    } else {
        dgf_qb[dst] = bf16_lo(word);
        dgf_qb[dst + 1u] = bf16_hi(word);
    }
}

@compute @workgroup_size(64)
fn dg_kv_mirror(@builtin(global_invocation_id) gid: vec3<u32>) {
    let words = dgf_p.head_dim >> 1u;
    let n = dgf_p.chunk * dgf_p.n_kv_heads * words;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let slot = i / (dgf_p.n_kv_heads * words);
    let r = i - slot * dgf_p.n_kv_heads * words;
    let kvh = r / words;
    let w = r - kvh * words;
    let pos = dgf_step[DGF_STEP_POS] + slot;
    if (pos >= dgf_p.kv_capacity) {
        return;
    }
    let kw = dgf_ksrc[dgf_kword(pos, kvh, w)];
    let vw = dgf_vsrc[pos * dgf_p.kv_words + kvh * words + w];
    let d = w * 2u;
    let kdst = pos * dgf_p.head_dim + d;
    let c16 = dgf_cap16();
    if (kvh == 0u) {
        dgf_kf0[kdst] = dgf_kv_lo(kw);
        dgf_kf0[kdst + 1u] = dgf_kv_hi(kw);
        dgf_vt0[d * c16 + pos] = dgf_kv_lo(vw);
        dgf_vt0[(d + 1u) * c16 + pos] = dgf_kv_hi(vw);
    } else {
        dgf_kf1[kdst] = dgf_kv_lo(kw);
        dgf_kf1[kdst + 1u] = dgf_kv_hi(kw);
        dgf_vt1[d * c16 + pos] = dgf_kv_lo(vw);
        dgf_vt1[(d + 1u) * c16 + pos] = dgf_kv_hi(vw);
    }
}

var<workgroup> dgf_red: array<f32, 256>;

@compute @workgroup_size(256)
fn dg_asoftmax(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32
) {
    let row = wg.x;
    let slot = row % dgf_p.chunk;
    let base = dgf_step[DGF_STEP_POS];
    let kvlen = min(base + dgf_p.chunk, dgf_p.kv_capacity);
    let kdyn = ((kvlen + 15u) / 16u) * 16u;
    var qlim = kvlen;
    if (DGF_CAUSAL == 1u) {
        qlim = min(kvlen, base + slot + 1u);
    }
    let stride = dgf_cap16();
    let rowbase = row * stride;
    let sc2 = dgf_p.scale * DGF_LOG2E;

    var lmax = -3.0e38;
    for (var j = lid; j < qlim; j = j + 256u) {
        lmax = max(lmax, dgf_s[rowbase + j] * sc2);
    }
    dgf_red[lid] = lmax;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (lid < st) {
            dgf_red[lid] = max(dgf_red[lid], dgf_red[lid + st]);
        }
        workgroupBarrier();
    }
    let rmax = dgf_red[0];
    workgroupBarrier();

    var lsum = 0.0;
    for (var j = lid; j < kdyn; j = j + 256u) {
        var e = 0.0;
        if (j < qlim && rmax > -2.0e38) {
            e = exp2(dgf_s[rowbase + j] * sc2 - rmax);
        }
        dgf_s[rowbase + j] = e;
        lsum = lsum + e;
    }
    dgf_red[lid] = lsum;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (lid < st) {
            dgf_red[lid] = dgf_red[lid] + dgf_red[lid + st];
        }
        workgroupBarrier();
    }
    if (lid == 0u) {
        dgf_l[row] = dgf_red[0];
    }
}

@compute @workgroup_size(64)
fn dg_opack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let words = dgf_p.head_dim >> 1u;
    let n = dgf_p.chunk * dgf_p.n_q_heads * words;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let slot = i / (dgf_p.n_q_heads * words);
    let r = i - slot * dgf_p.n_q_heads * words;
    let h = r / words;
    let w = r - h * words;
    let hg = h % dgf_p.gqa_group;
    let orow = hg * dgf_p.chunk + slot;
    let ob = orow * dgf_p.head_dim + w * 2u;
    var lo = 0.0;
    var hi = 0.0;
    var lsum = 0.0;
    if (h < dgf_p.gqa_group) {
        lo = dgf_oa[ob];
        hi = dgf_oa[ob + 1u];
        lsum = dgf_la[orow];
    } else {
        lo = dgf_ob[ob];
        hi = dgf_ob[ob + 1u];
        lsum = dgf_lb[orow];
    }
    var inv = 0.0;
    if (lsum > 0.0) {
        inv = 1.0 / lsum;
    }
    dgf_out[slot * dgf_p.out_slot_words + h * words + w] = bf16_pack(lo * inv, hi * inv);
}
"#;

pub const DG_STEEL_CVT_WGSL: &str = r#"
struct DgScvtParams {
    m: u32,
    qkv_words: u32,
    q_words: u32,
    kv_words: u32,
    capacity: u32,
    heads_words: u32,
    row_off: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> dsc_step: array<u32>;
@group(0) @binding(1) var<storage, read> dsc_qsrc: array<u32>;
@group(0) @binding(2) var<storage, read> dsc_ksrc: array<u32>;
@group(0) @binding(3) var<storage, read> dsc_vsrc: array<u32>;
@group(0) @binding(4) var<uniform> dsc_p: DgScvtParams;
@group(0) @binding(5) var<storage, read_write> dsc_qh: array<u32>;
@group(0) @binding(6) var<storage, read_write> dsc_kh: array<u32>;
@group(0) @binding(7) var<storage, read_write> dsc_vh: array<u32>;
@group(0) @binding(8) var<storage, read> dsc_oh: array<u32>;
@group(0) @binding(9) var<storage, read_write> dsc_out: array<u32>;
@group(0) @binding(10) var<storage, read_write> dsc_sp: array<u32>;
const DSC_KT: u32 = 0u;
const DSC_KV16: u32 = 0u;
const DSC_STEP_POS: u32 = 1u;

fn dsc_h16(w: u32) -> u32 {
    return pack2x16float(vec2<f32>(bf16_lo(w), bf16_hi(w)));
}

fn dsc_kv_h16(w: u32) -> u32 {
    if (DSC_KV16 == 1u) {
        return w;
    }
    return dsc_h16(w);
}

@compute @workgroup_size(64)
fn dg_scvt_q(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= dsc_p.q_words) {
        return;
    }
    dsc_qh[gid.y * dsc_p.q_words + w] =
        dsc_h16(dsc_qsrc[(gid.y + dsc_p.row_off) * dsc_p.qkv_words + w]);
}

@compute @workgroup_size(64)
fn dg_scvt_kv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = dsc_p.m * dsc_p.kv_words;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let slot = i / dsc_p.kv_words;
    let w = i - slot * dsc_p.kv_words;
    let pos = dsc_step[DSC_STEP_POS] + slot;
    if (pos >= dsc_p.capacity) {
        return;
    }
    var kidx = pos * dsc_p.kv_words + w;
    if (DSC_KT == 1u) {
        kidx = w * dsc_p.capacity + pos;
    }
    let o = pos * dsc_p.kv_words + w;
    dsc_kh[o] = dsc_kv_h16(dsc_ksrc[kidx]);
    dsc_vh[o] = dsc_kv_h16(dsc_vsrc[pos * dsc_p.kv_words + w]);
}

@compute @workgroup_size(64)
fn dg_scvt_o(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = dsc_p.m * dsc_p.q_words;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let slot = i / dsc_p.q_words;
    let w = i - slot * dsc_p.q_words;
    let v = unpack2x16float(dsc_oh[i]);
    dsc_out[(slot + dsc_p.row_off) * dsc_p.heads_words + w] = bf16_pack(v.x, v.y);
}

@compute @workgroup_size(1)
fn dg_steel_pp() {
    let base = dsc_step[DSC_STEP_POS];
    let kl = min(base + dsc_p.m, dsc_p.capacity);
    dsc_sp[4u] = kl;
    dsc_sp[8u] = (kl + 15u) / 16u;
    dsc_sp[10u] = kl / 16u;
    dsc_sp[12u] = kl - (kl / 16u) * 16u;
    dsc_sp[13u] = base;
}

const DSC_STEP_KVBASE: u32 = 3u;

@compute @workgroup_size(64)
fn dg_scvt_kv_r(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = dsc_p.m * dsc_p.kv_words;
    let i = gid.x;
    if (i >= n) {
        return;
    }
    let slot = i / dsc_p.kv_words;
    let w = i - slot * dsc_p.kv_words;
    let pos = dsc_step[DSC_STEP_POS] + slot;
    let src = dsc_step[DSC_STEP_KVBASE] + pos;
    if (src >= dsc_p.capacity) {
        return;
    }
    var kidx = src * dsc_p.kv_words + w;
    if (DSC_KT == 1u) {
        kidx = w * dsc_p.capacity + src;
    }
    let o = pos * dsc_p.kv_words + w;
    dsc_kh[o] = dsc_kv_h16(dsc_ksrc[kidx]);
    dsc_vh[o] = dsc_kv_h16(dsc_vsrc[src * dsc_p.kv_words + w]);
}
"#;

pub const NVFP4_GEMV_ROWS_PER_WORKGROUP: usize = 8;

pub const NVFP4_GEMV_SPENDS_32_LANES_ON_A_ROW_SO_A_1536_WIDE_K_STILL_AMORTIZES_THE_REDUCTION: &str =
    "A decode GEMV row of k=1536 is 96 nvfp4 blocks. Spread over 128 lanes that is under \
            one block each, and the seven-step tree reduction then costs more than the arithmetic \
            it reduces; 32 lanes per row give three blocks each and a five-step tree, and the \
            workgroup carries eight rows instead of two so a projection launches a quarter of the \
            workgroups.";

pub const PREFILL_GEMM_SLOTS_PER_DISPATCH_ARE_THE_TWO_VEC4_ACCUMULATORS_OF_DG_GEMM_M8: usize = 8;

pub const PREFILL_M_ENV: &str = "NV_DOTS_PREFILL_M";

pub const PREFILL_M_DEFAULT_32_IS_THE_LADDER_KNEE_19_6_S_PAGE_64_FLAT_128_COSTS_SMALL_PAGES: usize =
    32;

pub const PREFILL_M_LADDER_IS_CAPPED_BY_THE_VISION_TOWER_NOT_BY_THE_CHUNK: &str =
    "Widening the chunk stops paying at 32 rows because prefill is no longer what the page \
     spends its time on: the candle vision tower is about 40 s of the invoice page's 64 s, so \
     the whole ladder moves the remaining quarter. Above 64 rows the chunk starts costing small \
     pages, whose prompts are shorter than one chunk and fall back to per-token stepping for the \
     remainder.";

pub const PREFILL_M_MAX: usize = 4096;

pub const PREFILL_M_WIDE: usize = 128;

pub const COOP_PLAN_TM4_TN4_SG4_KU2_WON_EVERY_SHAPE_OF_THE_WIDTH_LADDER: (u32, u32, u32, u32) =
    (4, 4, 4, 2);

pub const COOP_ENV: &str = "NV_DOTS_COOP_PREFILL";

pub const COOP_PREFILL_PAYS_ONLY_AT_THE_WIDE_CHUNK: &str =
    "The tile-8 f32 cooperative matrix loses at m=32 and wins from m=128, so it rides the wide \
     list of the progressive chunker and the narrow list keeps dg_gemm_m8. It is wired on the \
     gate/up projection, which is 59% of a layer's weights and the shape the width ladder put at \
     5.44x. f32 staging rather than f16 keeps the operands exact, and the logits stay bit- \
     identical to the hand kernel through it. The cost is a second copy of that projection in \
     f32, about 3.1 GiB.";

pub fn coop_prefill_enabled() -> bool {
    let _ = COOP_PREFILL_PAYS_ONLY_AT_THE_WIDE_CHUNK;
    !matches!(
        std::env::var(COOP_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const DEC_W4_ENV: &str = "NV_DOTS_DEC_W4";

pub const DEC_ABLATE_ENV: &str = "NV_DOTS_DEC_ABLATE";

pub const DEC_GRID1_ENV: &str = "NV_DOTS_DEC_GRID1";

pub const DEC_ABLATE_IS_THE_DECODE_DELETED_WORK_BRACKET_BECAUSE_TIMESTAMP_PROFILES_INVERTED_THRICE:
    &str = "Measurement instruments only, default off, outputs are garbage while set. \
     NV_DOTS_DEC_ABLATE drops one class of the decode step's passes so the wall difference \
     prices that class execution-plus-issue: gemv / gemv_qkv / gemv_o / gemv_gu / gemv_down / \
     attn / elem / head / all. The per-projection classes ride the fixed qkv,o,gu,down cycle a \
     serving-default layer records, so they are only meaningful on that default (lora and the \
     nvfp4 quantize passes carry their own entries and do not shift the cycle). \
     NV_DOTS_DEC_GRID1=1 keeps every one of the step's passes and collapses each grid to one \
     workgroup, which prices the serial-dispatch issue floor with the execution deleted -- the \
     285-pass structure intact. Deleted work rather than per-pass timestamps because summed \
     timestamps inverted the prefill profile once and the decode orchestration theory twice.";

fn dec_ablate() -> Option<String> {
    let _ =
        DEC_ABLATE_IS_THE_DECODE_DELETED_WORK_BRACKET_BECAUSE_TIMESTAMP_PROFILES_INVERTED_THRICE;
    std::env::var(DEC_ABLATE_ENV).ok().filter(|v| !v.is_empty())
}

fn dec_grid1() -> bool {
    matches!(std::env::var(DEC_GRID1_ENV).ok().as_deref(), Some("1"))
}

pub const DEC_SUBMIT_CHUNK_ENV: &str = "NV_DOTS_DEC_SUBMIT_CHUNK";

pub const DEC_PASS_BOUNDARY_IS_METAL_SERIAL_DISPATCH_DRAIN_NOT_CPU_ISSUE: &str =
    "The decode step's ~2 ms pass-boundary class is the GPU waiting between serially \
     dependent dispatches under Metal's serial dispatch mode (the vendored wgpu-hal default): \
     kernel N retires and flushes before N+1 launches, priced per preceding kernel \
     (~3.5us after a plain elementwise, ~7us after a smem+subgroup rmsnorm, between for the \
     msl w4 gemv, larger after attn). It is NOT CPU issue (re-encoding all 285 dispatches \
     costs 0.15 ms/step, 0.5us each, submit 0.013 ms), NOT pipeline-state switching \
     (alternating two pipelines costs exactly the average of their same-pipeline ladders), \
     NOT launch overhead (the same dispatches with no hazards under concurrent dispatch run \
     0.7us each), and NOT wgpu hazard tracking (serial mode discards wgpu-core's barriers; \
     re-adding them under concurrent dispatch is a wash-to-loss on this dependency chain). \
     The only live cut is overlapping CPU-side command-buffer prep and submit latency with \
     execution via chunked submits; the drain itself needs fewer or wider passes, both \
     measured dead for dots decode.";

const DEC_SUBMIT_CHUNK_DEFAULT_64_OVERLAPS_ENCODE_WITH_EXEC_MINUS_A_QUARTER_MS_BITWISE: u32 = 64;

fn dec_submit_chunk() -> usize {
    let _ = DEC_PASS_BOUNDARY_IS_METAL_SERIAL_DISPATCH_DRAIN_NOT_CPU_ISSUE;
    std::env::var(DEC_SUBMIT_CHUNK_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(
            DEC_SUBMIT_CHUNK_DEFAULT_64_OVERLAPS_ENCODE_WITH_EXEC_MINUS_A_QUARTER_MS_BITWISE
                as usize,
        )
}

pub const DEC_RUNAHEAD_ENV: &str = "NV_DOTS_DEC_RUNAHEAD";

pub const DEC_TOKEN_TRACE_ENV: &str = "NV_DOTS_TOKEN_TRACE";

pub const DEC_RUNAHEAD_IS_EXACT_BECAUSE_THE_BAN_LIST_IS_KNOWN_BEFORE_THE_TOKEN: &str =
    "the windowed-ngram ban list for selecting token N depends only on already-generated \
     tokens, so when it is empty the shipped loop defines token N as the GPU argmax the head \
     already wrote to token_out; copying that 4-byte buffer into the embed gather's token \
     slot at the front of step N+1 on the serial Metal queue reproduces the selection \
     bit-identically before the CPU has seen it, deleting the per-step readback-then-encode \
     gap. A step whose ban list is non-empty, and the step consuming its corrected token, \
     serialize exactly as the shipped loop does, so no selected token is ever speculative. \
     The only run-ahead waste is one discarded trailing step when eos, loop-detect or the kv \
     bound ends generation: it emits nothing, its kv row dies with the session's reset, and \
     pos ends one higher than the serial loop's.";

const DEC_RUNAHEAD_DEFAULT_ON_HIDES_THE_READBACK_AND_ENCODE_GAP_MINUS_44MS_CORPUS_MEAN: &str =
    "=0 restores the serial loop that waits for each token before encoding the next step";

fn dec_runahead() -> bool {
    let _ = DEC_RUNAHEAD_DEFAULT_ON_HIDES_THE_READBACK_AND_ENCODE_GAP_MINUS_44MS_CORPUS_MEAN;
    !matches!(std::env::var(DEC_RUNAHEAD_ENV).ok().as_deref(), Some("0"))
}

pub struct BoundaryRep {
    pub total_ms: f64,
    pub encode_ms: f64,
    pub submit_ms: f64,
    pub wait_ms: f64,
    pub passes: usize,
}

pub fn dec_pass_class(entry: &str, is_head: bool, gemv_ord: usize) -> &'static str {
    if is_head {
        return "head";
    }
    if entry.contains("gemv") {
        return match gemv_ord % 4 {
            0 => "gemv_qkv",
            1 => "gemv_o",
            2 => "gemv_gu",
            _ => "gemv_down",
        };
    }
    if entry.starts_with("dg_attn") {
        return "attn";
    }
    "elem"
}

fn dec_ablate_drops(class: &str, want: &str) -> bool {
    want == "all" || want == class || (want == "gemv" && class.starts_with("gemv"))
}

pub const MSL_GEMV_ENV: &str = "NV_DOTS_MSL_GEMV";

pub fn msl_gemv_enabled() -> bool {
    let _ = decode_gemv_msl::MSL_GEMV_EXISTS_BECAUSE_THE_FIVE_ISSUE_SLOT_BOUND_WAS_NAGAS_NOT_THE_MACHINES;
    !matches!(
        std::env::var(MSL_GEMV_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const DEC_W4_PAYS_ONLY_AFTER_THE_BF16_GEMV_REACHED_ITS_EQUILIBRIUM_AND_ONLY_CONSISTENTLY: &str =
    "The first w4 verdict -- this adapter never pays for the bytes it saves -- was earned \
     against a bf16 gemv that was issue-bound, and it flipped the day that gemv reached its \
     measured equilibrium (vec4 loads, eight rows per workgroup): with the same tuning applied \
     to the nibble kernel (a weight vec4 is 32 codes, warp per row, subgroupAdd), decode reads \
     7.26 ms against bf16's 7.89 at ctx 3412 and 6.94 at 512, still arithmetic-exact against \
     the host oracle. Twice now the SLC-warmed micro-probe inverted the real verdict for \
     bandwidth-bound kernels (0.2x cached, 1.09x real) -- per-shape cached times are not \
     evidence in either direction. The speed lands on the CONSISTENT arm only: \
     NV_DOTS_DEC_PRECISION=w4 (dequantised prefill, packed decode) holds all twelve stable \
     pages at exactly 0.0000; NV_DOTS_DEC_W4=1 on a bf16 graph mixes a bf16-weight kv cache \
     with quantised decode weights and moved a stable page (report, 0.0201), so the mixed arm \
     stays a measurement tool, never a serving default.";

pub fn dec_w4_enabled() -> bool {
    let _ = DEC_W4_PAYS_ONLY_AFTER_THE_BF16_GEMV_REACHED_ITS_EQUILIBRIUM_AND_ONLY_CONSISTENTLY;
    matches!(
        std::env::var(DEC_W4_ENV).ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

pub const COOP_PROJ_ENV: &str = "NV_DOTS_COOP_PROJ";

pub const COOP_PROJ_COVERS_THE_OTHER_41_PERCENT_BECAUSE_THE_ABLATION_NAMED_PROJECTIONS_THE_WALL:
    &str = "The deleted-work ablation read projections at 62.1 percent of the 3412-token \
     prefill wall against attention's 34, inverting the per-pass-timestamp profile that had \
     capped coop coverage at the gate/up pair. This arm extends the same tile-8 f32 \
     cooperative GEMM to qkv, o_proj and down on the wide chunk list, with the same f32 \
     weight copy the gate/up arm carries and the narrow list keeping dg_gemm_m8; the qkv \
     output was already f32 so that leg needs no convert out.";

pub fn coop_proj_enabled() -> bool {
    let _ = COOP_PROJ_COVERS_THE_OTHER_41_PERCENT_BECAUSE_THE_ABLATION_NAMED_PROJECTIONS_THE_WALL;
    !matches!(
        std::env::var(COOP_PROJ_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const PREFILL_WIDTH_KNEE_MOVED_WHEN_ATTENTION_STARTED_SHARING_K_ACROSS_QUERY_ROWS: &str =
    "The first width ladder put the knee at 32 and showed 128 costing small pages, and that was \
     a property of the attention it was measured against: one workgroup per query row re-read \
     the whole KV cache, so more rows per chunk bought nothing. Once the register tile spends one \
     loaded K value on eight query rows, a wider chunk is exactly what feeds it, and prefill of a \
     3412-token prompt reads 9.97 s at 32 rows against 7.23 at 128. Short prompts still prefer \
     32, but not because wide chunks are slow -- because a prompt shorter than one chunk falls \
     back to stepping tokens one at a time. So the graph carries both widths and spends the \
     widest that fits before falling back, which is faster than either width alone.";

pub const PREFILL_BIG_ENV: &str = "NV_DOTS_PREFILL_BIG";

pub const PREFILL_BIG_3456_COVERS_THE_STANDARD_PAGE_IN_ONE_CHUNK: usize = 3456;

fn prefill_big() -> usize {
    let raw = std::env::var(PREFILL_BIG_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(PREFILL_BIG_3456_COVERS_THE_STANDARD_PAGE_IN_ONE_CHUNK);
    (raw / 8) * 8
}

pub const PREFILL_BIG_MID_1024_IS_THE_CHUNKED_LADDER_KNEE: usize = 1024;

pub const PREFILL_BIG_HEAD_FIRES_AT_FIVE_EIGHTHS_BECAUSE_ONE_PADDED_CHUNK_READS_KV_ONCE: &str =
    "Chunked prefill re-reads the whole KV prefix every chunk, so a page that fits one padded \
     chunk pays its attention once: the 3411-row invoice reads 944.8 ms as a single 3456-row \
     chunk against 1143.5 via 1024-row chunks and 1701.7 via 128s. Below five eighths of the \
     big width the fixed projection cost of the padded rows overtakes that saving and the \
     width loops take over. The stale verdict this replaces -- widening past 128 is a loss -- \
     was measured on dg_attn_reg with coop on gate/up only; both pillars are gone.";

pub fn prefill_widths() -> Vec<usize> {
    let _ = PREFILL_WIDTH_KNEE_MOVED_WHEN_ATTENTION_STARTED_SHARING_K_ACROSS_QUERY_ROWS;
    let m = prefill_m();
    if m == 0 {
        return Vec::new();
    }
    let slots = PREFILL_GEMM_SLOTS_PER_DISPATCH_ARE_THE_TWO_VEC4_ACCUMULATORS_OF_DG_GEMM_M8;
    if PREFILL_M_WIDE > m
        && PREFILL_M_WIDE.is_multiple_of(slots)
        && PREFILL_M_WIDE.is_multiple_of(ATTN_TILE_QUERY_ROWS)
    {
        let big = prefill_big();
        if big > PREFILL_M_WIDE {
            return vec![
                big,
                PREFILL_BIG_MID_1024_IS_THE_CHUNKED_LADDER_KNEE,
                PREFILL_M_WIDE,
                m,
            ];
        }
        vec![PREFILL_M_WIDE, m]
    } else {
        vec![m]
    }
}

pub fn prefill_m() -> usize {
    let raw = std::env::var(PREFILL_M_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(
            PREFILL_M_DEFAULT_32_IS_THE_LADDER_KNEE_19_6_S_PAGE_64_FLAT_128_COSTS_SMALL_PAGES,
        );
    if raw == 0 {
        return 0;
    }
    let _ = PREFILL_M_LADDER_IS_CAPPED_BY_THE_VISION_TOWER_NOT_BY_THE_CHUNK;
    let slots = PREFILL_GEMM_SLOTS_PER_DISPATCH_ARE_THE_TWO_VEC4_ACCUMULATORS_OF_DG_GEMM_M8;
    (raw.clamp(slots, PREFILL_M_MAX) / slots) * slots
}

pub const PREFILL_IS_A_SECOND_PASS_LIST_OVER_THE_SAME_WEIGHTS: &str =
    "A page's prompt is thousands of vision rows, so prompt throughput is page throughput. One \
     submit per token spends the whole weight file per token and launches n_q workgroups of \
     attention against the whole KV cache; the chunk list runs eight rows per submit through one \
     GEMM that reads each weight word once for all eight, and one attention dispatch that carries \
     eight positions, so the same graph serves both shapes off one upload.";

pub const FOLDING_THE_GQA_GROUP_LOSES_MORE_PARALLELISM_THAN_THE_REUSE_IS_WORTH: &str =
    "Six query heads share each KV head, so dg_attn_split reads every K and V word of a slice six \
     times, and dg_attn_fold removes exactly that redundancy: one workgroup takes F query heads \
     of one KV head and spends each loaded word on all F. It is bitwise identical and it is \
     slower at every factor. Kernel ms of a 28-call bf16 step at 3412, F = 1/2/3/6: 4.221, 5.198, \
     5.881, 7.964. The reason is arithmetic the ladder makes plain -- the grid is (n_q, n_splits) \
     so folding divides the workgroup count by F, taking 324 workgroups to 54 at F=6, and this \
     adapter has never once paid for traffic it did not have to move while paying every time for \
     work it could run at once. The K transpose does not rescue it either: at F=2 with the \
     transpose on the kernel is 5.247, slightly worse than the 5.198 without, because a folded \
     grid is short of workgroups rather than short of coalescing. This is the fifth traffic-\
     reduction attempt on this file to measure a loss, and it is the cleanest of them, since the \
     fold changes nothing but the workgroup count and the number of times a word is read.";

pub const TRANSPOSING_K_TO_HEAD_MAJOR_IS_THE_ONE_LAYOUT_CHANGE_THAT_PAID: &str =
    "Decode attention reads K with 256 threads whose addresses sit 512 B apart, because the cache \
     is [token][kv_head][head_dim] and a thread owns a token. Writing K as \
     [kv_head][dim_pair][token], with dims 2w and 2w+1 of one token in a word so the write stays \
     race-free, makes adjacent threads read adjacent words. Kernel ms of a 28-call step at 3412: \
     dg_attn_split 4.221 -> 2.978, and the whole decode step 12.43 -> 11.20 in bf16 and 12.64 -> \
     11.40 in nvfp4, all bitwise identical. V is deliberately left token-major: its accumulation \
     loop has thread d reading V[t][kvh][d] at a fixed t, which already coalesces, and moving it \
     would break what works. This is a layout change rather than a traffic change, which is why \
     it pays where the fold above does not -- it moves the same bytes in the same number of \
     workgroups and only changes which thread asks for which word.";

pub const ATTN_SPLIT_ENV: &str = "NV_DOTS_ATTN_SPLIT";

pub const ATTN_SPLIT_MAX: usize = 64;

pub const ATTN_SPLIT_TOKENS_PER_SLICE: usize = 128;

pub const ATTN_SPLIT_MIN_CONTEXT_IS_WHERE_ONE_WORKGROUP_PER_HEAD_STOPS_FILLING_THE_CORES: usize =
    256;

pub const ATTN_SPLIT_LIVE_SLICE_CAP_28_MUST_EQUAL_THE_WGSL_DG_SPLIT_LIVE_SLICE_CAP: u32 = 28;

pub const ATTN_SPLIT_TURNS_TWELVE_WORKGROUPS_INTO_TWELVE_TIMES_THE_SLICES: &str =
    "Decode attention launches one workgroup per query head, which is twelve on this checkpoint \
     against a GPU with dozens of cores, and each of them walks the whole KV cache -- so the step \
     cost grows with context and the machine idles while it does. Splitting the KV dimension \
     gives every head n_splits workgroups over disjoint slices, each returning a running max, a \
     running sum and a partial accumulator, and a second small pass folds them with the standard \
     log-sum-exp rescale. Short contexts keep the single-pass kernel, which is why the graph \
     carries both and picks by the position it is decoding at.";

pub fn attn_split_enabled() -> bool {
    !matches!(
        std::env::var(ATTN_SPLIT_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const ATTN_SG_ENV: &str = "NV_DOTS_ATTN_SG";

pub const ATTN_ABL_ENV: &str = "NV_DOTS_ATTN_ABL";

pub const BATTN_ENV: &str = "NV_DOTS_BATTN";

pub const BATTN_V2_REGROUPS_FETCHES_WITHOUT_TOUCHING_ANY_PER_ELEMENT_FMA_CHAIN: &str =
    "The batched range walk (dga_range_attend behind dg_attn_r and dg_attn_b; dg_attn \
     keeps its own inline body) priced as issue-bound on its load chains, not on unique \
     bytes: deleting both chains at plen 200-260 recovers 2-3 ms of the 12.6-14.4 ms \
     bstep while the issued-byte model implies apparent bandwidth at or above the DRAM \
     roof, so the GQA-redundant re-reads are cache-served and a fold that dedupes them \
     would buy bytes the cache already absorbs at the price of a 384->64 workgroup grid \
     (the recorded fold loss). The v2 walk deletes issued work bitwise: pass 2 reuses \
     pass 1's RAW dot from a workgroup cache rather than re-walking K (recompute past \
     DGA_SC_CAP keeps any span exact), K loads hoist 4-wide with the fma order \
     untouched, and the two one-shot reductions take the ATTN_SG exact-pairing forms -- \
     11.1/11.8/12.8/14.6 -> 10.2/10.9/11.8/12.7 ms/bstep at plen 64/120/200/260 x 32 \
     rows, two agreeing flock stamps per arm. Two banked findings: the cache MUST hold \
     the pre-scale dot, because caching score*scale rounds the product that Metal's \
     fp-contraction folds into exp's (raw*scale - m) as one fma, and that single-ulp \
     class moved every real-weight logit bit while the tiny gate stayed clean (the \
     RMS_FUSE compiled-value-vs-recompute hazard in its fp-contract form); and the \
     lane-owned V-word regroup is a banked NEGATIVE here -- halving the V walk's active \
     lanes halves its outstanding loads and lost 0.3 ms at plen200, so the tpw pattern \
     does not transfer to a walk with spare lanes and the per-element 128-lane V loop \
     stays. NV_DOTS_BATTN=0 restores the v1 walk.";

pub fn battn_v2_enabled() -> bool {
    let _ = BATTN_V2_REGROUPS_FETCHES_WITHOUT_TOUCHING_ANY_PER_ELEMENT_FMA_CHAIN;
    !matches!(
        std::env::var(BATTN_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const BATTN_ABL_ENV: &str = "NV_DOTS_BATTN_ABL";

pub const BATTN_ABL_DELETES_THE_MONOLITHIC_WALKS_LOAD_CHAINS_TO_SPLIT_BYTES_FROM_LATENCY: &str =
    "Instrument only, outputs garbage while set: 1 deletes the batched range walk's V loads, \
     2 deletes its K score loads, 3 deletes both, keeping every loop, reduction and barrier so \
     the wall delta prices the memory chains of the dga_range_attend family (dg_attn, \
     dg_attn_r, dg_attn_b) alone, the same split the DGX_ABL arms gave the serial split pass.";

pub const ATTN_PIPE_ENV: &str = "NV_DOTS_ATTN_PIPE";

pub const ATTN_PIPE_ISSUES_EIGHT_LOADS_AHEAD_BECAUSE_THE_SPLIT_PASS_IS_A_LATENCY_CHAIN: &str =
    "Banked NEGATIVE, ships default-off as a measurement arm: eight loads ahead with the fma \
     order untouched (bit-identical, 0/151936 at every rung) measured a LOSS -- 5.53 vs the \
     tree arm's 5.29 ms/step at ctx 3412 and losing at every rung -- because the V walk's \
     apparent 13us prices out at ~800 GB/s, the roof, not a stallable latency chain, and eight \
     live loads only spend registers. The split pass's remaining body cost is bandwidth at the \
     GQA-redundant read (the fold that would dedupe it lost at every factor on grid fill, \
     recorded at the KV-transpose constant).";

pub fn attn_pipe_enabled() -> bool {
    let _ = ATTN_PIPE_ISSUES_EIGHT_LOADS_AHEAD_BECAUSE_THE_SPLIT_PASS_IS_A_LATENCY_CHAIN;
    matches!(
        std::env::var(ATTN_PIPE_ENV).ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

pub const ROPE_KV_FUSE_ENV: &str = "NV_DOTS_ROPE_KV_FUSE";

pub const ROPE_KV_FUSE_DELETES_ONE_PASS_PER_LAYER_BECAUSE_THE_BRACKET_PRICED_EVERY_PASS_BOUNDARY:
    &str = "The decode-step bracket priced the serial-encoder pass boundary at ~7us on trivial \
     kernels (the elem class deletes 142 passes for 1.0 ms with near-zero kernel work), so a \
     pass exists to be deleted, not just to be made faster. dg_bias_rope_kvw writes the k/v \
     rows' rotated words to the caches at dg_kv_write's exact addresses while it writes \
     qkv_rot, and the kv-write pass is not recorded -- byte-identical stores, one boundary \
     fewer per layer. The batched-region and prefill paths keep their own kv writers. \
     NV_DOTS_ROPE_KV_FUSE=0 restores the separate pass.";

pub fn rope_kv_fuse_enabled() -> bool {
    !matches!(
        std::env::var(ROPE_KV_FUSE_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const RMS_GEMV_FUSE_ENV: &str = "NV_DOTS_RMS_GEMV_FUSE";

pub const RMS_GEMV_FUSE_IS_A_BANKED_NEGATIVE_TOLERANCE_CLASS_AND_A_WALL_LOSS: &str =
    "Measurement arm, ships default-off. The decode bracket priced 142 elementwise passes as \
     ~7us of pure pass boundary each and the 56 rms passes are the largest deletable block \
     (every norm feeds exactly one w4 MSL gemv), so this arm recomputes the row norm in the \
     consumer gemv's prologue (qkv, gate-up) and folds the residual add into the producer's \
     epilogue (o, down), deleting 2*layers passes. Both verdicts are against it. WALL: at ctx \
     64 the gu side reads +0.7 ms/step (17920 workgroups re-reduce the row and pay ~3x inner \
     loop issue for the in-register normalize on a bandwidth-roof kernel) and the qkv side \
     washes (+0.02) -- the redundant norm costs more than the ~0.4 ms of boundaries it \
     deletes, the exact redundancy the fusion charter told this round to price. EXACTNESS: \
     tolerance-class only, mechanism in \
     RMS_FUSE_IS_TOLERANCE_CLASS_BECAUSE_SIMD_SUM_COMPILES_TO_A_PER_KERNEL_ORDER -- the norm \
     recompute is measured bit-exact but the dot's simd_sum order is a per-kernel compiler \
     choice, so recompiled accs sit one ulp off the shipped gemv's on most rows and surface \
     as rare bf16-boundary flips (full logit divergence at data-dependent context cells, \
     first seen at ctx 250 layer 18). The last layer's rmsnext and the lm head stay unfused \
     because the batched-region step reuses the head passes on rows final_x already holds \
     pre-normed, and lora disables the fusion because its slices read the standalone normed \
     buffers. NV_DOTS_RMS_GEMV_FUSE=1 fuses both sides, =qkv or =gu one side; unset stays \
     the shipped standalone-norm default.";

pub fn rms_gemv_fuse_mode() -> (bool, bool) {
    let _ = RMS_GEMV_FUSE_IS_A_BANKED_NEGATIVE_TOLERANCE_CLASS_AND_A_WALL_LOSS;
    match std::env::var(RMS_GEMV_FUSE_ENV).ok().as_deref() {
        Some("qkv") => (true, false),
        Some("gu") => (false, true),
        Some("1") | Some("all") | Some("on") | Some("true") => (true, true),
        _ => (false, false),
    }
}

pub const DEC_GSILU_ENV: &str = "NV_DOTS_DEC_GSILU";

const DEC_GSILU_DEFAULT_ON_DELETES_THE_SILU_PASS_BOUNDARY_PER_LAYER_BITWISE: &str =
    "=0 restores the standalone dg_silu_mul pass after the gate-up gemv; the rms-fuse \
     measurement arms and lora keep the standalone pass so every banked row keeps its \
     pre-flip meaning (see \
     GSILU_FUSE_KEEPS_EVERY_REDUCTION_IN_ITS_KERNEL_AND_REPLAYS_THE_SILU_PASS_VERBATIM)";

pub fn dec_gsilu_enabled() -> bool {
    let _ = DEC_GSILU_DEFAULT_ON_DELETES_THE_SILU_PASS_BOUNDARY_PER_LAYER_BITWISE;
    !matches!(
        std::env::var(DEC_GSILU_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const ATTN_ABL_DELETES_THE_SPLIT_KERNELS_LOAD_CHAINS_TO_PRICE_LATENCY_NOT_BANDWIDTH: &str =
    "Instrument only, outputs garbage while set: 1 deletes the V-walk loads, 2 deletes the \
     K-score loads, 3 deletes both, keeping every loop, reduction and barrier so the wall \
     difference prices the memory chain alone. Built because the step bracket read the split \
     pass as latency-bound (one workgroup costs what the whole grid costs) and the subgroup \
     barrier cut recovered only ~1.5us of the ~54us pass -- the serial cost had to be named \
     before any restructure.";

pub const ATTN_SG_CUTS_THE_SPLIT_KERNELS_BARRIER_CHAIN_KEEPING_THE_EXACT_REDUCTION_PAIRING: &str =
    "The decode-step bracket read the attention class at 31 us per pass with the grid collapsed \
     to ONE workgroup costing the same 34 -- the kernel is latency-bound on its in-workgroup \
     serial chain of ~21 workgroupBarrier sweeps per chunk, not on bandwidth or occupancy. The \
     subgroup form keeps every arithmetic pairing of the shipped tree: max reduces by \
     subgroupMax (max is pairing-free bit-for-bit), the sum keeps the cross-simd tree levels \
     s=128/64/32 in shared memory and finishes s=16..1 with subgroupShuffleDown steps that \
     reproduce red[l] += red[l+s] exactly, and lsum lives on lane 0 alone so no broadcast \
     barrier is needed -- 21 barriers become 7 with bit-identical logits. NV_DOTS_ATTN_SG=0 \
     restores the tree form.";

pub fn attn_sg_enabled() -> bool {
    let _ = ATTN_SG_CUTS_THE_SPLIT_KERNELS_BARRIER_CHAIN_KEEPING_THE_EXACT_REDUCTION_PAIRING;
    !matches!(
        std::env::var(ATTN_SG_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const SG_TREE_ENV: &str = "NV_DOTS_SG_TREE";

pub const SG_TREE_TWINS_ARE_THE_ONLY_ARM_WHERE_SUBGROUP_WIDTH_IS_NOT_PROVABLY_32: &str =
    "Every subgroup reduction in the decode graph assumes width exactly 32: the (lid & 31u) \
     staging into 8- and 32-slot warp arrays double-counts lanes on a 64-wide device and \
     drops them on 16-wide, and a browser primes the runtime width unprobed. Pipeline build \
     therefore selects workgroup-memory tree twins for the batched attention walk \
     (dga_range_attend_v2), the rmsnorm pair (dg_rms/dg_rmsres) and the w4 gemv \
     (dg_gemv_w4_pk) whenever the device cannot prove width 32, and the subgroup forms stay \
     the default everywhere it can. The attention twin keeps the exact operand pairing of \
     the subgroup form (max is pairing-free; the sum tree's s=16..1 tail pairs identically \
     to the shuffleDown chain) and gates bitwise, so necessity-selection is bit-preserving \
     there. The rms and w4 twins replace subgroupAdd, whose lane-combination order is \
     implementation-defined, so they are reassociation-class: never default on a width-32 \
     device, engaged only by necessity or by NV_DOTS_SG_TREE (1/all/on/true, or a comma \
     list of attn,rms,w4), with necessity-selection allowed under the flip law because no \
     subgroup arm exists on such a device to preserve. Selected tree modules also swap the \
     const-disabled subgroup calls for scalar stubs so naga and browser tint validate the \
     text without the subgroups capability.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SgTreeSites {
    pub attn: bool,
    pub rms: bool,
    pub w4: bool,
}

pub fn sg_tree_sites_from(env: Option<&str>, subgroup_width: Option<u32>) -> SgTreeSites {
    let _ = SG_TREE_TWINS_ARE_THE_ONLY_ARM_WHERE_SUBGROUP_WIDTH_IS_NOT_PROVABLY_32;
    let need = subgroup_width != Some(32);
    let v = env.unwrap_or("");
    let all = matches!(v, "1" | "all" | "on" | "true");
    let has = |site: &str| all || v.split(',').any(|t| t.trim() == site);
    SgTreeSites {
        attn: need || has("attn"),
        rms: need || has("rms"),
        w4: need || has("w4"),
    }
}

pub fn sg_tree_sites(subgroup_width: Option<u32>) -> SgTreeSites {
    sg_tree_sites_from(std::env::var(SG_TREE_ENV).ok().as_deref(), subgroup_width)
}

const DG_SGLESS_STUBS_WGSL: &str = r#"
fn dg_sgless_max(x: f32) -> f32 { return x; }
fn dg_sgless_add(x: f32) -> f32 { return x; }
fn dg_sgless_shfl(x: f32, d: u32) -> f32 { return x; }
"#;

fn swap_dead_subgroup_calls_for_stubs(src: &str) -> String {
    let out = src
        .replace("subgroupMax(", "dg_sgless_max(")
        .replace("subgroupAdd(", "dg_sgless_add(")
        .replace("subgroupShuffleDown(", "dg_sgless_shfl(");
    assert!(
        !out.contains("subgroup"),
        "a subgroup builtin survived the stub swap, so this module cannot validate on a \
         device without the subgroups capability"
    );
    format!("{DG_SGLESS_STUBS_WGSL}\n{out}")
}

pub const DECODE_HAS_NO_DISPATCH_RESIDUAL_AND_ITS_ONLY_HOST_COST_IS_THE_LOGIT_READBACK: &str =
    "Decode was suspected of spending a fifth of its time outside the kernels, on the theory that \
     285 dispatches a step cost something each. Wall clock around a whole step, readback included, \
     at 3412 on an uncontended adapter: the full step is 11.486 ms, the same step reading back only \
     the 4-byte argmax token is 11.016, the 151936-float logit readback is 0.470 and the host \
     argmax over it is 0.181, so a generate-shaped step is 11.667 ms and 85.7 tok/s. Summed pass \
     timestamps for the same step read 11.20, which is within the cost of the timestamp writes \
     themselves, so there is no per-dispatch residual to divide by 285 and the orchestration \
     theory is dead. Context growth is not hiding it either: 384 steps walking 3412 to 3796 \
     median 11.107 over the first 64 and 11.277 over the last, a 1.5 percent drift across a whole \
     page of output. What is left is 0.65 ms of host cost, 5.6 percent, and all of it is the \
     logit vector: the graph already computes its own argmax into a 4-byte buffer, and generate \
     copies 0.6 MB back every step only so the host can redo that argmax against an n-gram ban \
     list that is empty on almost every step.";

pub const KV_TRANSPOSE_ENV: &str = "NV_DOTS_KT";

pub const KV_TRANSPOSE_IS_THE_ACCESS_PATTERN_FIX_THE_DECODE_ATTENTION_ABLATION_ASKED_FOR: &str =
    "Decode attention was never starved of bandwidth: at 3392 context, of the 3.67 ms the K score \
     read costs, giving it a coalesced index recovers 2.56 and giving it a constant address only \
     1.11 more, so two thirds of the cost is the ACCESS PATTERN and one third is volume. Laying \
     the K cache out as [kv_head][dim_pair][token] puts consecutive lanes on consecutive \
     addresses and collects exactly that: dg_attn 4.221 -> 2.978 ms at 3412 context, 29 percent \
     of the kernel and 9.9 percent of the whole decode step, 12.43 -> 11.20 ms bf16. It is \
     BITWISE IDENTICAL -- a transpose moves which thread reads which word and changes no \
     arithmetic and no reduction order -- so it is gated on bits rather than on tokens, and \
     NV_DOTS_KT=0 restores the untransposed layout. Contrast the GQA fold, which attacked the \
     same reads from the volume side and lost at every factor because the grid is (n_q, \
     n_splits): folding six query heads onto one kv head divides 324 workgroups by six, and no \
     amount of deleted redundant reading pays for a grid that no longer fills the machine.";

pub fn kv_transpose_enabled() -> bool {
    let _ = KV_TRANSPOSE_IS_THE_ACCESS_PATTERN_FIX_THE_DECODE_ATTENTION_ABLATION_ASKED_FOR;
    !matches!(
        std::env::var(KV_TRANSPOSE_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const KV_F16_ENV: &str = "NV_DOTS_KV_F16";

pub const KV_F16_REENCODES_THE_STORED_WORD_SO_BYTES_CANNOT_CHANGE_AND_ONLY_RANGE_EDGES_CAN_DIVERGE:
    &str = "Measurement arm, ships default-off, BANKED WASH and tolerance-class BY NATURE \
     (bf16 to f16 storage changes stored bits). Under NV_DOTS_KV_F16=1 the decode K and V \
     caches hold pack2x16float pairs instead of bf16 pairs, and every writer -- the fused \
     rope writer, the prefill chunk writer, and the batched-region B=32 writers, which share \
     the same buffers and therefore the same arm -- stores the f16 re-encoding of exactly \
     the bf16 word the shipped cache stores, one rounding rule everywhere. bf16 and f16 are \
     the same width, so the attention walks move IDENTICAL bytes (the decode-round sizing \
     line 'halves attn bytes' was wrong and its -0.3 ms ceiling is falsified with it); the \
     only buyable mechanism was unpack cost, one native unpack2x16float against two \
     shift-plus-bitcast sequences per cache word, and it measured a WASH at every rung: \
     base/kv16 med ms 4.725/4.686, 4.653/4.664, 4.950/4.951, 5.007/4.972 at ctx \
     64/512/2048/3412 (second stamp agrees inside the rep drift band) and corpus mean \
     2599.2 on vs 2602.6 off. A bf16 value re-encodes to f16 exactly when \
     2^-16 <= |x| < 65536, so an in-range cache holds the same real values and every f32 \
     fma chain is unchanged -- every synthetic rung measured 0/151936 logit bits moved -- \
     but the real-page census reads max |K| 322 and max |V| 25.75 with zero overflow and \
     zero nonfinite stored halves and ~1-2 per 100k values below 2^-16 that round in f16 \
     subnormals, so real pages are tolerance-class, not bitwise (twelve zeros EXACT and \
     mean CER 0.0700 EXACT held on the stamped corpus regardless). The steel prefill \
     mirrors are bitwise unchanged because dg_scvt_kv already stores exactly this f16 \
     word; under the arm that conversion is the identity. Unset serves the shipped bf16 \
     cache.";

pub fn kv_f16_enabled() -> bool {
    let _ =
        KV_F16_REENCODES_THE_STORED_WORD_SO_BYTES_CANNOT_CHANGE_AND_ONLY_RANGE_EDGES_CAN_DIVERGE;
    matches!(
        std::env::var(KV_F16_ENV).ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

pub const ATTN_FOLD_ENV: &str = "NV_DOTS_ATTN_FOLD";

pub const ATTN_FOLD_MAX: usize = 8;

pub const ATTN_FOLD_MAX_HEAD_DIM: usize = 128;

pub fn attn_fold_factor(cfg: &DotsDecoderConfig) -> usize {
    let group = cfg.num_attention_heads / cfg.num_key_value_heads;
    let want = std::env::var(ATTN_FOLD_ENV)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1)
        .clamp(1, ATTN_FOLD_MAX);
    if want > 1 && (!group.is_multiple_of(want) || cfg.head_dim() > ATTN_FOLD_MAX_HEAD_DIM) {
        return 1;
    }
    want
}

pub fn attn_split_count(max_seq: usize) -> usize {
    max_seq
        .div_ceil(ATTN_SPLIT_TOKENS_PER_SLICE)
        .clamp(1, ATTN_SPLIT_MAX)
}

pub const ATTN_TILED_ENV: &str = "NV_DOTS_ATTN_TILED";

pub const ATTN_TILE_QUERY_ROWS: usize = 8;

pub const ATTN_TILE_KV_TOKENS: usize = 32;

pub const ATTN_TILE_MAX_HEAD_DIM: usize = 128;

pub const ATTN_TILE_STAGES_25664_OF_THE_32768_BYTE_WORKGROUP_BUDGET: &str =
    "The tile is 8192 B of K plus 8192 of V plus 4096 of Q plus 4096 of accumulators plus 1024 of \
     scores plus 64 of running max and sum, which is 25664 B against this adapter's 32768; a head \
     dimension above 128 would not fit and keeps the untiled kernel rather than failing to load. \
     It is opt-in because it did not pay: staging cut the KV a 32-row chunk moves from 18.67 GB \
     to 2.33 GB and bought 2 percent, because prefill attention was never bandwidth-bound. At \
     2.33 GB in 159 ms it is moving 15 GB/s and computing 117 GFLOP/s, while dg_gemm_m8 in the \
     same file and the same bf16 unpacking reaches 1.88 TFLOP/s. The 16x is arithmetic per \
     unpacked value: the GEMM spends one weight word on 16 fused multiplies because eight query \
     slots share it, and attention spends one K word on two. Register-tiling the query rows so \
     each loaded K and V value feeds eight rows is the fix, and it is a different rewrite from \
     this one.";

pub const GEMM_SLOTS_ENV: &str = "NV_DOTS_GEMM_SLOTS";

pub const GEMM_M8_ACTIVATION_LOADS_ARE_ONLY_14_PERCENT_SO_A_VECTOR_DECODE_HAS_NO_ROOM: &str =
    "Ablated by giving all eight activation slots the same loaded word, which removes seven of \
     the eight loads and the decodes with them: 50.89 ms to 43.64 per chunk, 14%. That is the \
     ceiling on any change to this kernel's activation path, vector decode included, so the \
     rewrite that raised the sibling affine GEMV was not worth porting here. Removing roughly \
     half the kernel's operations for 14% also says it is not purely issue-bound, unlike the \
     nvfp4 GEMV beside it.";

pub const GEMM_SLOTS_16_HALVED_THE_TRAFFIC_AND_COST_15_PERCENT_SO_8_STAYS: &str =
    "The prefill GEMM spends one loaded weight word on two multiplies per activation slot, so a \
     chunk re-reads the whole weight file once per m/slots dispatch: at m=32 and 8 slots that is \
     four reads of 3.3 GB where 3.3 GB is unique. Sixteen slots halves the reads and the dispatch \
     count and measured WORSE -- 50.89 ms to 58.37 ms per chunk, 448 dispatches to 224 -- because \
     four more vec4 accumulators per thread crosses this adapter's register budget, the same \
     boundary the sibling tower's tm8_tn8 plan fell off. It also settles what the GEMM is bound \
     by: 13.2 GB in 50.89 ms is 259 GB/s and 6.6 GB in 58.37 ms is 113 GB/s, both far under this \
     machine's bandwidth, so neither arm was ever bandwidth-bound and halving traffic could not \
     help. Eight stays the default and sixteen is reachable for a different register file.";

pub fn gemm_slots() -> usize {
    let _ = GEMM_SLOTS_16_HALVED_THE_TRAFFIC_AND_COST_15_PERCENT_SO_8_STAYS;
    let _ = GEMM_M8_ACTIVATION_LOADS_ARE_ONLY_14_PERCENT_SO_A_VECTOR_DECODE_HAS_NO_ROOM;
    match std::env::var(GEMM_SLOTS_ENV).ok().as_deref() {
        Some("16") => 16,
        _ => PREFILL_GEMM_SLOTS_PER_DISPATCH_ARE_THE_TWO_VEC4_ACCUMULATORS_OF_DG_GEMM_M8,
    }
}

pub const NVFP4_MROW4_ENV: &str = "NV_DOTS_NVFP4_MROW4";

pub const DECODE_ATTENTION_IS_UNCOALESCED_ON_K_NOT_STARVED_OF_BANDWIDTH: &str =
    "Decode attention reads 97.8 MB of KV per step at 3412 context and takes 7.44 ms, which reads \
     as 13 GB/s and invites the conclusion that it is starved. Three ablations at 3392 context, \
     dg_attn_split of 6.51 ms: giving the K score read a constant address drops it to 2.84, so K \
     reads cost 3.67; giving the K read the same volume but an index that puts consecutive lanes \
     on consecutive addresses drops it to 3.95, so 2.56 of that 3.67 is the ACCESS PATTERN and \
     only 1.11 is volume; giving V a constant address drops it to 5.75, so V costs 0.76 and is \
     already coalesced. The remaining 2.08 ms is softmax, reductions and barriers. The cache is \
     [token][kv_head][head_dim], so a thread walking one K vector reads 128 contiguous elements \
     while its neighbours sit 512 B away -- every lane on its own cache line. V escapes because \
     its loop puts consecutive lanes on consecutive head dims. Transposing the K cache alone to \
     [kv_head][dim_pair][token], leaving V's layout untouched, is what the measurement points \
     at; it would recover about 2.56 ms and cost a strided write of one token against 3412 \
     coalesced reads. It does NOT make the kernel bandwidth-bound: a third of it is arithmetic \
     and a third of the K cost is irreducible, so the ceiling is about 4.9 ms, not 1.";

pub const PREFILL_IS_TWO_THIRDS_ATTENTION_AND_ITS_GEMMS_ARE_ALREADY_FAST: &str =
    "Prefill has been priced as a GEMM problem and it is not one. Kernel ms of one 128-row chunk \
     at 3412, which is 1/27th of the prompt and scales to it: dg_attn_reg 90.547 ms at 63.3 \
     percent, dg_gemm_m8 over every projection 51.043 at 35.7, and everything else together under \
     1. Converting those to rates settles which half is slow. A chunk does 0.335 TFLOP of \
     projection against 0.0376 TFLOP of attention, so the projections run at 6.57 TFLOP/s and the \
     attention at 0.415 -- a factor of 16 apart, with the fast one already carrying nine tenths \
     of the arithmetic. The aggregate 2.6 TFLOP/s that prefill reports is therefore not a GEMM \
     that needs a better kernel; it is a fast GEMM averaged with an attention kernel an order of \
     magnitude behind it. Extending the coop path to the projections it does not yet cover is \
     bounded by that 35.7 percent no matter how fast it gets: making every projection free leaves \
     prefill at 2460 ms against 3887, and cannot approach a 749 ms target on its own. Bringing \
     attention to the rate the projections already reach is worth more than the whole GEMM budget \
     and is the one prefill kernel nobody has attacked. It is also two matmuls, QK-transpose and \
     PV, which is the shape the coop path is good at.";

pub const COOP_WAS_REJECTED_AT_A_CHUNK_WIDTH_CHOSEN_FOR_A_DIFFERENT_KERNEL: &str =
    "The tile-8 f32 cooperative matrix was measured at m=32 and rejected as a thin GEMM. m=32 \
     was our parameter, picked by a ladder against dg_gemm_m8, whose time is flat in m; it is \
     not a property of the workload. Re-measured across widths, best plan of five per point, \
     coop TFLOP/s against dg_gemm_m8 on the same data: at m=32 qkv 0.40x, gateup 3.30x, down \
     0.42x; at m=128 2.55x, 5.44x, 1.48x; at m=256 4.73x, 7.15x, 2.47x; at m=512 6.02x, 7.08x, \
     3.23x. Coop reaches 11.11 TFLOP/s at m=256 gateup where the hand kernel reaches 1.55. The \
     thin-GEMM diagnosis was right about m=32 and wrong as a verdict on coop.";

pub const WHAT_MOVES_A_KERNEL_ON_THIS_ADAPTER_IN_THE_ORDER_WORTH_TRYING: &str =
    "Eleven controlled experiments in this file, M2 Ultra via wgpu-Metal, in the order a next \
     kernel should try them. FIRST, ablate rather than reason: replace one suspected cost with a \
     constant and measure -- one probe run settles in minutes what four rounds of argument did \
     not, and reasoning about arithmetic was wrong four times running. WHAT PAID: adding \
     parallelism that did not exist (split-KV decode attention, 12 workgroups to 12*n_splits, \
     5.7x); raising arithmetic per loaded value (register-tiled prefill attention, one K word \
     spent on eight query rows instead of one, 1.92x); batching rows per submit (8-row prefill \
     chunk, 2.8-4.0x); and raising the launch count (nvfp4 GEMV workgroup 32 rows to 4, 9%). \
     WHAT DID NOT: spending fewer bytes, four times -- staging K/V cut traffic 8x for 2%, a \
     16-slot GEMM halved traffic and cost 15%, nvfp4 quartered weight bytes and lost 32%, and a \
     cooperative-matrix route measured 0.29-1.36x because M=32 is too thin to fill its \
     fragments; and cutting instructions per byte -- a vec4 nibble decode measured null twice, \
     and removing seven of eight activation loads from the prefill GEMM bought 14%. THE RULE: \
     this hardware punishes low occupancy and rewards reuse per load; bytes are the currency in \
     exactly one kernel here, the bf16 decode GEMV, which already runs at 75% of roofline. \
     Measure occupancy and reuse first, traffic last.";

pub const NVFP4_GEMV_FOUR_HYPOTHESES_ONLY_OCCUPANCY_PAID: &str =
    "Four structural changes were measured on this GEMV at 3412 context, kernel ms of a 112-call \
     step: sharing one activation decode across four rows 5.49 -> 5.71 (null), decoding four \
     nibbles into a vec4 and spending them in dot() instead of eight scalar decode+fma pairs \
     5.71 -> 5.71 (null), giving the block scale a constant address to remove its 4 KB stride \
     5.71 -> 5.60 (2%), and cutting the workgroup from 32 rows to 4 so a projection launches \
     eight times the workgroups 5.71 -> 5.18 (9%). Only the last moved, which is the same answer \
     the rest of this file gave: the kernel is issue and occupancy bound, and neither the bytes \
     it reads nor the instructions it spends decoding them are what it is waiting on. What \
     remains unexplained is the 155 GB/s it reaches against the bf16 GEMV's 598 on the same \
     step, and no hypothesis tried here accounts for it.";

pub const THE_Q4_GEMV_IS_ALU_BOUND_SO_ITS_GB_PER_SECOND_WAS_THE_WRONG_SCORE: &str =
    "The 155 GB/s this GEMV reached against the bf16 GEMV's 598 was read for a whole round as a \
     bandwidth failure. Two ablations at 3412 context, kernel ms of a 112-call step, say it was \
     the wrong ceiling to score against: pointing every row at one weight block so the entire \
     weight stream becomes cache resident takes 5.200 -> 4.431, and removing the e2m1 unpack \
     while leaving both loads in place takes 5.200 -> 3.735. Deleting all of the traffic buys 15 \
     percent and deleting the dequant arithmetic buys 28, so this kernel is ALU bound on the \
     unpack and a memory roofline never applied to it. That also retires the search for a faster \
     unpack: hoisting the activation bf16 decode out of the row loop, which the source repeats \
     four times, measured 5.200 -> 5.186 because the compiler had already hoisted it, and the \
     magnitude decode is five vector ops in every closed form tried. A nibble format costs the \
     28 percent to read. The lever for q4 decode is fewer or wider GEMV calls, not a better \
     kernel, and any future GB/s quoted for it should be quoted against issue rate instead.";

pub const NVFP4_MROW4_SHARES_ONE_ACTIVATION_DECODE_ACROSS_FOUR_ROWS: &str =
    "The block-scale decode, the nibble unpack and the activation read are per row in a one-row-\
     per-lane-group GEMV, which left this arm at 146 GB/s where the bf16 GEMV beside it reaches \
     598. This entry gives each 32-lane simdgroup four output rows and decodes the activation \
     block once for all four, the structure the sibling affine-w4 GEMV used to go from 250 to 547 \
     GB/s. Reading the activation as bf16 rather than as pre-quantized codes also deletes the \
     separate quantize pass the arm used to run before every projection, which was 4.06 ms of a \
     19.39 ms step and which the bf16 arm never paid.";

pub fn nvfp4_mrow4_enabled() -> bool {
    let _ = NVFP4_MROW4_SHARES_ONE_ACTIVATION_DECODE_ACROSS_FOUR_ROWS;
    let _ = NVFP4_GEMV_FOUR_HYPOTHESES_ONLY_OCCUPANCY_PAID;
    let _ = FOLDING_THE_GQA_GROUP_LOSES_MORE_PARALLELISM_THAN_THE_REUSE_IS_WORTH;
    let _ = TRANSPOSING_K_TO_HEAD_MAJOR_IS_THE_ONE_LAYOUT_CHANGE_THAT_PAID;
    let _ = DECODE_HAS_NO_DISPATCH_RESIDUAL_AND_ITS_ONLY_HOST_COST_IS_THE_LOGIT_READBACK;
    let _ = THE_Q4_GEMV_IS_ALU_BOUND_SO_ITS_GB_PER_SECOND_WAS_THE_WRONG_SCORE;
    let _ = WHAT_MOVES_A_KERNEL_ON_THIS_ADAPTER_IN_THE_ORDER_WORTH_TRYING;
    let _ = COOP_WAS_REJECTED_AT_A_CHUNK_WIDTH_CHOSEN_FOR_A_DIFFERENT_KERNEL;
    let _ = PREFILL_IS_TWO_THIRDS_ATTENTION_AND_ITS_GEMMS_ARE_ALREADY_FAST;
    let _ = DECODE_ATTENTION_IS_UNCOALESCED_ON_K_NOT_STARVED_OF_BANDWIDTH;
    let _ = THE_PAGE_HAS_NO_UNACCOUNTED_TIME_AND_ITS_CONVERSIONS_ARE_NOT_THE_LEVER;
    let _ = THE_TWO_DOTS_DECODERS_SPLIT_PREFILL_FROM_DECODE;
    !matches!(
        std::env::var(NVFP4_MROW4_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const ATTN_REG_ENV: &str = "NV_DOTS_ATTN_REG";

pub const ATTN_FUSED_ENV: &str = "NV_DOTS_ATTN_FUSED";

pub const ATTN_FUSED_IS_TWO_COOP_GEMMS_BECAUSE_A_128_ROW_CHUNK_MAKES_THE_SCORE_SLAB_CHEAP: &str =
    "Flash fusion exists to avoid materialising the score matrix; at a 128-row chunk that \
     matrix is 768 rows by the kv length, ten megabytes, and materialising it lets both \
     matmuls run as the proven 8x8-f32 cooperative GEMM instead of a bespoke kernel. S = Q K^T \
     stacks the six heads of one GQA group into m=768 against an f32 K mirror; a row-parallel \
     exp2 softmax masks and normalises in place; O = P V rides a V mirror stored dim-major. \
     Three hand-fused flash forms lost to dg_attn_reg first (4024.8, 5419.9, 5172.4 ms against \
     its 3808 on the 3412-token wall) because this WGSL coop extension exposes no per-element \
     fragment access, so online-softmax rescale forces either a per-block spill or a second \
     QK pass, and at 8x8 tiles those overheads eat the matrix unit. The GEMM form measures \
     2813.3 ms, minus 26 percent, attention by ablation arithmetic about 305 ms from 1295. The \
     static graph replays one pass list at every chunk position, so the GEMM shapes are sized \
     for capacity and a two-line patch on the generated source reads the true kv length from \
     the step buffer at run time: dead column blocks and dead k iterations fold to a zero loop \
     bound, never a return, which is what naga's coop uniformity analysis accepts. DGF_CAUSAL \
     folds away in dg_asoftmax so a non-causal caller is a source replace, not a fork.";

pub const PAD_ROWS_COST_NOTHING_BECAUSE_CAUSALITY_SHIELDS_THEM_AND_DECODE_OVERWRITES_THEM: &str =
    "A prompt tail shorter than a chunk used to run one 11 ms decode step per token; padding it \
     into a single chunk is bitwise-free because every real row attends only up to its own \
     position, which is strictly before every pad row, and the pad rows' garbage kv entries \
     sit at positions each later decode step rewrites before its own attention reads them -- \
     the position counter advances by the real row count, not the chunk width. The tail picks \
     the narrowest chunk list that fits and falls back to stepping only when padding would \
     overrun the kv span. The chunk-vs-steps bitwise suite covers the padded path because its \
     prompt length is deliberately not a multiple of any chunk width.";

pub const ATTN_FUSED_HEAD_DIM: usize = 128;

pub const ATTN_FUSED_QUERY_ROWS: usize = 8;

pub fn attn_fused_enabled() -> bool {
    let _ = ATTN_FUSED_IS_TWO_COOP_GEMMS_BECAUSE_A_128_ROW_CHUNK_MAKES_THE_SCORE_SLAB_CHEAP;
    !matches!(
        std::env::var(ATTN_FUSED_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const PF_STEEL_ENV: &str = "NV_DOTS_PF_STEEL";

pub const PF_STEEL_FEEDS_HALF_MIRRORS_BECAUSE_THE_KERNEL_IS_TYPED_HALF_AND_THE_MATH_STAYS_F32:
    &str = "The causal steel arm replaces the GEMM-form ten-pass attention with three word-wide \
    conversion passes and one fused dispatch. Q, K and V convert from the graph's packed bf16 \
    into half buffers -- steel is instantiated at half, loads convert on the way into \
    threadgroup memory, and every accumulator is float, so half is a storage format here, not \
    an arithmetic one. The K and V mirrors are per-layer and incremental like the fused arm's \
    f32 mirrors at a quarter the bytes, and the rectangle the GEMM arm paid for in full is cut \
    to the causal triangle by the kernel's own block skip. O converts back to packed bf16 in \
    the heads buffer so every downstream consumer is untouched. NV_DOTS_PF_STEEL=0 restores \
    the GEMM-form arm.";

pub fn pf_steel_enabled() -> bool {
    let _ = PF_STEEL_FEEDS_HALF_MIRRORS_BECAUSE_THE_KERNEL_IS_TYPED_HALF_AND_THE_MATH_STAYS_F32;
    !matches!(
        std::env::var(PF_STEEL_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const PF_STEEL_PROJ_ENV: &str = "NV_DOTS_PF_STEEL_PROJ";

pub const PF_STEEL_PROJ_CONFIG_IS_THE_VISION_LADDERS_CROWN: (u32, u32, u32, u32, u32) =
    (64, 64, 32, 2, 2);

pub const PF_STEEL_PROJ_DELETES_THE_F32_WEIGHT_COPIES_BECAUSE_BF16_TO_F16_IS_EXACT_IN_WEIGHT_RANGE:
    &str = "The four prefill projections run the vendored steel gemm the vision ladder crowned \
    (bm64 bn64 bk32 wm2 wn2, 1.25-1.40x over the coop incumbent on every k-divisible shape), \
    dispatched raw-MSL through the shared per-device pipeline cache. The nt pattern is the \
    projection layout as-is, so the only new pass is a word-wide bf16-to-half cvt on the \
    activations; every bf16 weight whose exponent sits in half's normal range converts \
    exactly, so the f16 weight copies carry the same values the f32 mirrors did at half the \
    bytes, and the f32 mirrors are not uploaded at all when this arm is on. U=float lands \
    the output in the same f32 buffers the coop arm wrote, so rope, the out-cvts and every \
    other consumer are untouched, and the trunk k values 1536 and 8960 are both whole \
    numbers of k-blocks. NV_DOTS_PF_STEEL_PROJ=0 restores the coop-f32 staged projections \
    and their f32 weight copies unchanged.";

pub fn pf_steel_proj_enabled() -> bool {
    let _ =
        PF_STEEL_PROJ_DELETES_THE_F32_WEIGHT_COPIES_BECAUSE_BF16_TO_F16_IS_EXACT_IN_WEIGHT_RANGE;
    !matches!(
        std::env::var(PF_STEEL_PROJ_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const PF_STEEL_CFG_ENV: &str = "NV_DOTS_PF_STEEL_CFG";

pub const PF_STEEL_CFG_IS_BITWISE_CLASS_BECAUSE_EVERY_TILE_WALKS_K_AS_THE_SAME_ASCENDING_MMA_CHAIN:
    &str = "A steel gemm tile config changes which threadgroup computes which output block and \
    how many k-blocks sit between barriers, never the arithmetic: each output fragment is the \
    same ascending-k chain of 8-deep simdgroup MMAs with f32 accumulators under every \
    bm/bn/bk/wm/wn, so config arms gate on bit equality like the steel-vs-coop flip did. \
    NV_DOTS_PF_STEEL_CFG=bm,bn,bk,wm,wn selects a ladder arm; unset rides the shipped config.";

pub fn pf_steel_proj_cfg() -> (u32, u32, u32, u32, u32) {
    let _ =
        PF_STEEL_CFG_IS_BITWISE_CLASS_BECAUSE_EVERY_TILE_WALKS_K_AS_THE_SAME_ASCENDING_MMA_CHAIN;
    std::env::var(PF_STEEL_CFG_ENV)
        .ok()
        .and_then(|v| {
            let p: Vec<u32> = v.split(',').filter_map(|t| t.trim().parse().ok()).collect();
            <[u32; 5]>::try_from(p).ok()
        })
        .map(|[bm, bn, bk, wm, wn]| (bm, bn, bk, wm, wn))
        .unwrap_or(PF_STEEL_PROJ_CONFIG_IS_THE_VISION_LADDERS_CROWN)
}

pub const PF_STEEL_NARROW_TILE_IS_32_32_16_BECAUSE_SUB1024_TRUNKS_ARE_GRID_STARVED_AT_THE_CROWN_TILE:
    &str = "At chunk widths 32 and 128 the crown tile leaves the steel gemm grid at 24-280 \
    threadgroups and the trunk runs about 7x over its f16 weight-read floor, so the smallest \
    instantiated tile wins by grid parallelism, not arithmetic: serving-span width curve m=32 \
    40.6 -> 23.4 ms, m=128 35.4 -> 28.9, batched region step at gazette state 43.4 -> 27.0 \
    ms/step, while m=1024 (149.2 -> 154.2) and the whole-page wide pass (503 -> 532 at ctx \
    3412) keep the crown because their projections already run at the ~24 TF/s steel rate. \
    Widths through 128 ride (32,32,16,2,2); wider trunks keep the vision ladder's crown. Every \
    tile walks k as the same ascending MMA chain, so all selections are bit-identical and \
    NV_DOTS_PF_STEEL_CFG=64,64,32,2,2 pins every width to the exact pre-flip config.";

pub const PF_STEEL_NARROW_MAX_WIDTH: usize = 128;

pub const PF_STEEL_NARROW_CONFIG: (u32, u32, u32, u32, u32) = (32, 32, 16, 2, 2);

pub fn pf_steel_proj_cfg_for(m: usize) -> (u32, u32, u32, u32, u32) {
    let _ =
        PF_STEEL_NARROW_TILE_IS_32_32_16_BECAUSE_SUB1024_TRUNKS_ARE_GRID_STARVED_AT_THE_CROWN_TILE;
    if std::env::var(PF_STEEL_CFG_ENV).is_ok() {
        return pf_steel_proj_cfg();
    }
    if m <= PF_STEEL_NARROW_MAX_WIDTH {
        PF_STEEL_NARROW_CONFIG
    } else {
        PF_STEEL_PROJ_CONFIG_IS_THE_VISION_LADDERS_CROWN
    }
}

pub const TRUNK_SPLITK_ENV: &str = "NV_DOTS_TRUNK_SPLITK";

pub const TRUNK_SPLITK_IS_TOLERANCE_CLASS_DEFAULT_ON_BECAUSE_ITS_ONLY_TEXT_MOVES_ARE_WHITESPACE:
    &str = "The default serves each m<=32 trunk projection as ONE grid-z steel dispatch of S \
    k-slices landing f32 partials plus one ascending-order reduce, against the single starved \
    dispatch NV_DOTS_TRUNK_SPLITK=0 restores byte-identically (=1 keeps its pre-flip opt-in \
    spelling). The m=32 trunk ran 4.6x over its byte floor because 24-64-workgroup grids \
    leave the k-walk latency-bound and slice-per-dispatch forms die on the Metal serial \
    drain, so concurrency comes from grid z inside one dispatch. The partial-sum add \
    reassociates the ascending MMA k-chain, so the arm can move logit bits at the ulp scale \
    -- it shipped opt-in until the word-diff adjudication over the full corpus and gazette \
    read content_moves=0 (the arm's one content move was a seam-crop artifact the scout ink \
    bridge dissolved; what remains is a single line-break->space class), and the coordinator \
    flipped it on that showing: gazette 7.05-7.09 vs 7.85-7.95 single-dispatch at ncer \
    0.0000 both arms, bstep 21.7 -> 13.8 ms class.";

pub const TRUNK_SPLITK_MAX_WIDTH: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProjFamily {
    Qkv,
    OProj,
    GateUp,
    Down,
}

pub const TRUNK_SPLITK_CROWNS_ARE_QKV8_O8_DOWN16_AND_GATEUP_STAYS_SINGLE: &str =
    "Bracket crowns at the m=32 shapes on the 28-layer distinct-weight chain: qkv 0.062 -> \
     0.035 ms at S=8, o 0.061 -> 0.032 at S=8, down 0.304 -> 0.075 at S=16 (its 48-workgroup \
     560-iteration k-walk was the single largest trunk term), while gateup's 560-workgroup \
     grid is already fed and every split lost to its 573k-element reduce, so it keeps the \
     single dispatch. Composed chain 14.71 -> 7.11 ms against a 3.6 ms byte floor.";

fn trunk_splitk_target(f: ProjFamily) -> usize {
    let _ = TRUNK_SPLITK_CROWNS_ARE_QKV8_O8_DOWN16_AND_GATEUP_STAYS_SINGLE;
    match f {
        ProjFamily::Qkv | ProjFamily::OProj => 8,
        ProjFamily::GateUp => 1,
        ProjFamily::Down => 16,
    }
}

pub fn trunk_splitk_enabled() -> bool {
    let _ = TRUNK_SPLITK_IS_TOLERANCE_CLASS_DEFAULT_ON_BECAUSE_ITS_ONLY_TEXT_MOVES_ARE_WHITESPACE;
    std::env::var(TRUNK_SPLITK_ENV).as_deref() != Ok("0")
}

pub fn trunk_splitk_s(f: ProjFamily, m: usize, k: usize, bk: u32) -> usize {
    if !trunk_splitk_enabled() || m > TRUNK_SPLITK_MAX_WIDTH {
        return 1;
    }
    let mut s = trunk_splitk_target(f);
    while s > 1 && !(k.is_multiple_of(s) && (k / s).is_multiple_of(bk as usize)) {
        s -= 1;
    }
    s
}

pub const TRUNK_BF16OUT_ENV: &str = "NV_DOTS_TRUNK_BF16OUT";

pub const TRUNK_ABF16_ENV: &str = "NV_DOTS_TRUNK_ABF16";

pub const TRUNK_CVT_FUSIONS_DELETE_196_MIRROR_PASSES_PER_BSTEP_AND_STAY_IN_THE_BITWISE_CLASS: &str =
    "The batched-step crumb census prices dg_scvt_q at 112 passes/0.73 ms and dg_cvt at \
    84 passes/0.65 ms per m=32 step -- both essentially pure Metal serial-drain boundary \
    around the trunk steel GEMMs. NV_DOTS_TRUNK_ABF16 folds the bf16->f16 activation mirror \
    into a replicated A-side block loader that stages the identical half values (bf16 bits \
    shift to f32 exactly, one f32->f16 RNE, dg_scvt_q's own chain), and NV_DOTS_TRUNK_BF16OUT \
    folds the f32->bf16e output conversion into the store walker on the same live accumulator \
    dg_cvt re-read (qkv keeps its f32 sink -- rope consumes it directly). No reduction moves \
    between kernels: the MMA k-chain is untouched and both conversions are single-op \
    elementwise, the gsilu/resadd event class. Both default on; =0 restores the standalone \
    mirror passes byte-identically. Splitk families keep f32 partials and ride the sk_abf16 \
    loader variant when ABF16 is on; their output fold lives in the reduce store (dg_sk_red_bf16) \
    per TRUNK_BF16OUT_FOLDS_THE_SPLITK_REDUCE_STORE_BECAUSE_THE_F32_PARTIAL_SUM_NEVER_MOVES.";

pub fn trunk_bf16out_enabled() -> bool {
    let _ = TRUNK_CVT_FUSIONS_DELETE_196_MIRROR_PASSES_PER_BSTEP_AND_STAY_IN_THE_BITWISE_CLASS;
    std::env::var(TRUNK_BF16OUT_ENV).as_deref() != Ok("0")
}

pub fn trunk_abf16_enabled() -> bool {
    std::env::var(TRUNK_ABF16_ENV).as_deref() != Ok("0")
}

pub const ROPE_KV_FUSE_B_ENV: &str = "NV_DOTS_ROPE_KV_FUSE_B";

pub const ROPE_KV_FUSE_B_PORTS_THE_SERIAL_FUSE_TO_THE_BATCHED_STEP: &str =
    "dg_kv_write_b is 28 passes/0.23 ms of pure copy per batched step: it re-reads the k/v \
    words dg_bias_rope_b just packed and lands them at each row's absolute cache slot. \
    dg_bias_rope_kvw_b writes the cache in the rope pass's own store loop from the identical \
    packed word, exactly as the serial decode's dg_bias_rope_kvw does from the step position; \
    the only difference is the slot source (dgk_bslot per row vs step pos), so the batched \
    fuse inherits the serial fuse's bitwise argument wholesale. Default on; =0 restores the \
    separate dg_kv_write_b pass byte-identically. The rpf and pack arms keep their own kv \
    passes untouched.";

pub fn rope_kv_fuse_b_enabled() -> bool {
    let _ = ROPE_KV_FUSE_B_PORTS_THE_SERIAL_FUSE_TO_THE_BATCHED_STEP;
    std::env::var(ROPE_KV_FUSE_B_ENV).as_deref() != Ok("0")
}

pub const SPEC_PROBE_ENV: &str = "NV_DOTS_SPEC_PROBE";

pub const SPEC_PROBE_MEASURES_FIRST_N_AGREEMENT_BECAUSE_ACCEPTANCE_IS_THE_ONE_UNKNOWN: &str =
    "Layer-skip self-speculation is lossless by construction and every cost but one is already \
     priced: drafting with the first N layers plus final norm plus lm_head, then verifying k \
     drafts in one whole-page-chunk-shaped batched step, models at 2.5-3.8 ms/token against \
     the 5.14 serving step. The unknown is acceptance, so this probe measures it and changes \
     nothing else: NV_DOTS_SPEC_PROBE=6,8,10 taps the residual stream after each listed layer \
     count with the model's own final norm, runs the same serving lm_head arm and the same \
     two-stage argmax into per-N buffers, and generate() reports per-call agreement between \
     each first-N argmax and the token it actually accepted. Probe logits live in their own \
     buffer so the ngram-ban path still reads the full model's logits; the probe argmax is \
     unbanned, which is what a draft head would propose. Unset, the graph gains no pass and \
     generate reads nothing. The probe rides the affine-w4 lm_head arm only, which is the \
     serving default the build-or-die curve must price. THE CURVE SAID DIE: over the sixteen \
     fixture corpus at serving defaults the long fixtures read N6 0.008-0.045, N8 0.000-0.056, \
     N10 0.015-0.053, N14 0.013-0.067, N20 0.160-0.386, N27 0.873-0.971 -- twenty-fold under \
     the 0.6 bar at every priced depth, with the N27 anchor near one proving the instrument \
     and the smooth monotonic shape ruling out plumbing. Even N20, at 71 percent of the depth \
     and nearly a full step of draft cost, accepts under a third. No k or N cell rescues 0.03 \
     acceptance; the build round was never started and the campaign's enumerated frontier \
     closed on this measurement.";

fn spec_probe_layers(layers: usize) -> Vec<usize> {
    let _ = SPEC_PROBE_MEASURES_FIRST_N_AGREEMENT_BECAUSE_ACCEPTANCE_IS_THE_ONE_UNKNOWN;
    std::env::var(SPEC_PROBE_ENV)
        .ok()
        .map(|v| {
            v.split(',')
                .filter_map(|t| t.trim().parse::<usize>().ok())
                .filter(|&n| n >= 1 && n < layers)
                .collect()
        })
        .unwrap_or_default()
}

pub const PF_ABLATE_ENV: &str = "NV_DOTS_PF_ABLATE";

pub const PF_ABLATE_DELETES_WORK_BECAUSE_SUMMED_PASS_TIMESTAMPS_INVERTED_THE_PREFILL_PROFILE: &str =
    "Per-pass timestamps read prefill as 63 percent dg_attn_reg and 36 percent \
     projections; deleting the work reads the opposite: projections 62.1 percent of the 3412 \
     token wall, attention 34.0 (qk 5.0, softmax 10.9, pv 18.0, the three summing to 99 percent \
     of the whole-dispatch delta), residual 3.9. Timestamps reward latency and idle-thread \
     reduction, so a share taken from them is an upper bound, not a measure. These arms are the \
     honest instrument: each deletes work and the wall is read around a logits readback. qk, \
     softmax and pv fold a shader constant so the compiler removes one phase of dg_attn_reg; \
     attn and proj skip whole prefill dispatches, leaving zeroed inputs whose downstream cost \
     is value-independent.";

pub const G4_ROWS_ENV: &str = "NV_DOTS_G4_ROWS";

pub const G4_ROWS_SMALL_ENV: &str = "NV_DOTS_G4_ROWS_SMALL";

pub const G4_ROWS_LM_ENV: &str = "NV_DOTS_G4_ROWS_LM";

fn g4_env(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .filter(|r| [2, 4, 8, 16, 32].contains(r))
        .unwrap_or(default)
}

fn g4_rows_for(n: usize) -> u32 {
    if n >= 65536 {
        g4_env(G4_ROWS_LM_ENV, 8)
    } else if n >= 8192 {
        g4_env(G4_ROWS_ENV, 8)
    } else {
        g4_env(G4_ROWS_SMALL_ENV, 8)
    }
}

fn pf_ablated(part: &str) -> bool {
    let _ = PF_ABLATE_DELETES_WORK_BECAUSE_SUMMED_PASS_TIMESTAMPS_INVERTED_THE_PREFILL_PROFILE;
    std::env::var(PF_ABLATE_ENV).ok().as_deref() == Some(part)
}

fn pf_ablate_shader_phase() -> u32 {
    match std::env::var(PF_ABLATE_ENV).ok().as_deref() {
        Some("qk") => 1,
        Some("softmax") => 2,
        Some("pv") => 3,
        _ => 0,
    }
}

pub const ATTN_REG_HOLDS_EIGHT_QUERY_ROWS_PER_LOADED_K_VALUE: &str =
    "dg_gemm_m8 reaches 1.88 TFLOP/s because one loaded weight word feeds sixteen fused \
     multiplies; the row-per-workgroup attention reached 117 GFLOP/s because one loaded K word \
     fed two. This entry gives every thread one key token and all eight query rows of the tile, \
     so a K word loaded once is spent on sixteen multiplies against eight accumulators held in \
     registers, and the V loop does the same with the probability row. head_dim is the reduction \
     axis, so it cannot also be the axis a register tile parallelises: holding a score tile would \
     need dimgroups * 8 * KT floats of partials, which does not fit 32768 B beside staged K and \
     V. That is why this stages only Q, the probabilities and the accumulators -- 16480 B -- and \
     streams K and V from global.";

pub fn attn_reg_enabled() -> bool {
    let _ = ATTN_REG_HOLDS_EIGHT_QUERY_ROWS_PER_LOADED_K_VALUE;
    !matches!(
        std::env::var(ATTN_REG_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub fn attn_tiled_enabled() -> bool {
    let _ = ATTN_TILE_STAGES_25664_OF_THE_32768_BYTE_WORKGROUP_BUDGET;
    matches!(
        std::env::var(ATTN_TILED_ENV).ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

pub const ARGMAX_GROUPS: usize = 256;

pub const ROPE_WORKGROUP_STAGING_ARRAY_CAPS_HEAD_DIM_AT_256: usize = 256;

pub const STAGING_FLUSH_BYTES: u64 = 256 << 20;

pub const DOTS_GRAPH_VRAM_ENV: &str = "NV_DOTS_GRAPH_VRAM";

pub const THE_TWO_DOTS_DECODERS_SPLIT_PREFILL_FROM_DECODE: &str =
    "Two dots.ocr decoders exist in this tree: this graph, and the q3d bias arm in \
     dots_ocr::decoder_wgpu on the mlx 4-bit checkpoint. Measured against each other at matched \
     context, readback-terminated, decode tok/s theirs/mine: 109.8/107.2 at 64, 105.9/105.4 at \
     512, 95.4/83.2 at 2048, 90.2/67.9 at 3412; prompt tok/s theirs/mine: 118.3/202.3, \
     120.7/698.8, 115.2/896.6, 110.5/890.6. They split. Their decode scales better with context \
     and is 1.33x at page length; this graph's chunked prefill is 8.1x at the same length \
     because theirs prefills a token at a time. Composed over an invoice page of 3412 prompt \
     and 414 generated tokens that is 9.93 s here against 35.47 s there, so this graph serves \
     the page and their decode attention is the technique worth taking, not their decoder.";

pub const PAGE_PROFILE_ENV: &str = "NV_DOTS_PAGE_PROFILE";

pub const THE_PAGE_HAS_NO_UNACCOUNTED_TIME_AND_ITS_CONVERSIONS_ARE_NOT_THE_LEVER: &str =
    "Instrumenting recognize() end to end on the invoice page, wgpu tower and graph decoder: \
     wall 18.43 s = prepare 0.010 + vision 7.766 + prompt 0.000 + prefill 3.919 + decode 6.672 + \
     post 0.000, residual 0.000. Nothing is missing; earlier ledgers were short because their \
     prefill and decode components predated the coop and split-kv work. The format conversions \
     on that path price out the same way: staging the tower's host f32 into a candle bf16 tensor \
     costs 0.002 s, pulling it back to host f32 for the decoder costs 0.064, and repacking it to \
     bf16 per chunk costs 0.007 -- 0.073 s of an 18.43 s page, so the whole round trip through \
     candle is removable waste worth four tenths of one percent. The coop path's f32 staging and \
     its two converts per layer are net positive, since the arm they enable measures 1.74x on \
     prefill with them included, and the f32 q/k/v intermediate is load bearing: it is what fixed \
     the bias cancellation, 0.133 to 0.026 relative on the tiny fixture.";

pub fn page_profile_enabled() -> bool {
    matches!(
        std::env::var(PAGE_PROFILE_ENV).ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

pub const NVFP4_ACTIVATION_GLOBAL_2POW7_LANDS_DECODE_BLOCK_SCALES_INSIDE_THE_E4M3_WINDOW: f32 =
    128.0;

pub const NVFP4_ACTIVATION_GLOBAL_IS_A_POWER_OF_TWO_SO_IT_ADDS_NO_ROUNDING: &str =
    "The nvfp4 block scale is a ue4m3 byte, so a block whose amax lands below 6 * 2^-9 quantizes \
     to the zero scale and the whole block is lost. A bf16 checkpoint carries no calibrated \
     input_global_scale, so the activation global is a fixed power of two: it shifts every block \
     scale into the e4m3 window without adding a rounding step of its own, and the weight alpha \
     divides it back out.";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DotsGraphQuant {
    #[default]
    Bf16,
    Nvfp4,
    AffineW4,
}

impl DotsGraphQuant {
    pub fn name(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Nvfp4 => "nvfp4",
            Self::AffineW4 => "affine-w4",
        }
    }

    pub fn bf16_like(self) -> bool {
        matches!(self, Self::Bf16 | Self::AffineW4)
    }
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct DgW4Params {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct DgG4Params {
    n_rows: u32,
    k_vecs: u32,
    w_row_vecs: u32,
    groups_x: u32,
    out_f32: u32,
    y_off_words: u32,
    pad0: u32,
    pad1: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GemvBf16Params {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_off_words: u32,
    y_off_words: u32,
    pad0: u32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GemvNvfp4Params {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct QuantRowsParams {
    k_blocks: u32,
    n_slots: u32,
    use_sel: u32,
    x_slot_stride_elems: u32,
    x_slot_div_zero_means_one_padding_the_shared_lgq_layout: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct SiluMulParams {
    n_words: u32,
    gate_off_words: u32,
    up_off_words: u32,
    src_slot_words: u32,
    dst_slot_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct ArgmaxParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct RowGatherParams {
    words: u32,
    src_slot_words: u32,
    row: u32,
    pad0: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct BiasRopeParams {
    n_rows: u32,
    head_dim: u32,
    half: u32,
    rope_rows: u32,
    has_bias: u32,
    src_f32: u32,
    src_slot_words: u32,
    dst_slot_words: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct KvWriteParams {
    kv_words: u32,
    kv_capacity: u32,
    k_off_words: u32,
    v_off_words: u32,
    src_slot_words: u32,
    n_kv_heads: u32,
    pad1: u32,
    pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct AttnParams {
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    gqa_group: u32,
    kv_capacity: u32,
    q_slot_words: u32,
    out_slot_words: u32,
    scale: f32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct SplitAttnParams {
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    gqa_group: u32,
    kv_capacity: u32,
    n_splits: u32,
    part_stride: u32,
    scale: f32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GemvNvfp4Mrow4Params {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CvtParams {
    n_elems: u32,
    src_slot: u32,
    dst_slot: u32,
    to_f32: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct CoopGemmParams {
    n_rows: u32,
    k_elems: u32,
    m_rows: u32,
    blocks_n: u32,
    y_stride: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct TileAttnParams {
    n_q_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    gqa_group: u32,
    kv_capacity: u32,
    chunk: u32,
    q_slot_words: u32,
    out_slot_words: u32,
    kv_words: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GemmMParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    w_row_words: u32,
    x_slot_words: u32,
    y_slot_words: u32,
    out_f32: u32,
    slot_base: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct GatherMParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    pad0: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct StepUniform {
    tok: u32,
    pos: u32,
    total: u32,
    sliding_start: u32,
}

pub const DOTS_GRAPH_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_GRAPH_ARM: &str =
    "NV_DOTS_GRAPH_LORA_DIR";

pub const SINK16_IS_BF16_ON_EVERY_ARM_BECAUSE_THE_MSL_GEMV_WRITES_BF16E: &str =
    "the act_f16 arm reads activations as device half but its gemv packs outputs with bf16e \
     and every 16-bit sink consumer (dg_rmsres, dg_silu_mul, rope) decodes bf16, so the lora \
     expand's read-modify-write on attn_out / y_gu / ffn_out must speak bf16 regardless of \
     the activation format; DgSink::F16 exists only for the synthetic kernel probe";

pub const GRAPH_LORA_OFF_IS_BITWISE_OFF: &str =
    "with no adapter dir the builder pushes zero lora passes and allocates zero lora buffers, \
     so the lora-off graph is pass-for-pass and bit-for-bit the pre-lora graph; the \
     chunk-vs-steps suite and the pass counts printed at load are the instruments";

const DG_LORA_SHRINK_WGSL: &str = r#"
struct DglParams {
  m: u32, k: u32, rank: u32, n: u32,
  x_fmt: u32, y_fmt: u32, y_off: u32, y_row: u32,
}
@group(0) @binding(0) var<uniform> dgl_p: DglParams;
@group(0) @binding(1) var<storage, read> dgl_a: array<u32>;
@group(0) @binding(2) var<storage, read> dgl_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> dgl_part: array<f32>;
fn dgl_bf1(w: u32) -> f32 { return bitcast<f32>((w & 0xffffu) << 16u); }
fn dgl_bf2(w: u32) -> f32 { return bitcast<f32>(w & 0xffff0000u); }
fn dgl_x1(w: u32) -> f32 {
  if dgl_p.x_fmt == 1u { return unpack2x16float(w).x; }
  return dgl_bf1(w);
}
fn dgl_x2(w: u32) -> f32 {
  if dgl_p.x_fmt == 1u { return unpack2x16float(w).y; }
  return dgl_bf2(w);
}
var<workgroup> dgl_red: array<f32, 64>;
@compute @workgroup_size(64)
fn dg_lora_shrink(
  @builtin(workgroup_id) wg: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let mi = wg.y;
  let r = wg.x;
  let t = mi * dgl_p.rank + r;
  let kw = dgl_p.k / 2u;
  let abase = r * kw;
  let xbase = mi * kw;
  var acc = 0.0;
  if t < dgl_p.m * dgl_p.rank {
    for (var w = lid.x; w < kw; w = w + 64u) {
      let aw = dgl_a[abase + w];
      let xw = dgl_x[xbase + w];
      acc = acc + dgl_bf1(aw) * dgl_x1(xw) + dgl_bf2(aw) * dgl_x2(xw);
    }
  }
  dgl_red[lid.x] = acc;
  workgroupBarrier();
  var span = 32u;
  while span > 0u {
    if lid.x < span {
      dgl_red[lid.x] = dgl_red[lid.x] + dgl_red[lid.x + span];
    }
    workgroupBarrier();
    span = span / 2u;
  }
  if lid.x == 0u && t < dgl_p.m * dgl_p.rank {
    dgl_part[mi * dgl_p.rank + r] = dgl_red[0u];
  }
}
"#;

const DG_LORA_EXPAND_WGSL: &str = r#"
struct DglParams {
  m: u32, k: u32, rank: u32, n: u32,
  x_fmt: u32, y_fmt: u32, y_off: u32, y_row: u32,
}
@group(0) @binding(0) var<uniform> dgl_p: DglParams;
@group(0) @binding(1) var<storage, read> dgl_b: array<u32>;
@group(0) @binding(2) var<storage, read> dgl_part: array<f32>;
@group(0) @binding(3) var<storage, read_write> dgl_y: array<u32>;
fn dgl_bf1(w: u32) -> f32 { return bitcast<f32>((w & 0xffffu) << 16u); }
fn dgl_bf2(w: u32) -> f32 { return bitcast<f32>(w & 0xffff0000u); }
fn dgl_rne_bf16(v: f32) -> u32 {
  let b = bitcast<u32>(v);
  return (b + 0x7fffu + ((b >> 16u) & 1u)) >> 16u;
}
@compute @workgroup_size(64)
fn dg_lora_expand(@builtin(global_invocation_id) gid: vec3<u32>) {
  let nw = dgl_p.n / 2u;
  let mi = gid.y;
  let wi = gid.x;
  if mi >= dgl_p.m || wi >= nw { return; }
  let n0 = wi * 2u;
  var acc0 = 0.0;
  var acc1 = 0.0;
  let rw = dgl_p.rank / 2u;
  let pbase = mi * dgl_p.rank;
  for (var r2 = 0u; r2 < rw; r2 = r2 + 1u) {
    let p0 = dgl_part[pbase + 2u * r2];
    let p1 = dgl_part[pbase + 2u * r2 + 1u];
    let b0 = dgl_b[n0 * rw + r2];
    let b1 = dgl_b[(n0 + 1u) * rw + r2];
    acc0 = acc0 + dgl_bf1(b0) * p0 + dgl_bf2(b0) * p1;
    acc1 = acc1 + dgl_bf1(b1) * p0 + dgl_bf2(b1) * p1;
  }
  if dgl_p.y_fmt == 0u {
    let ybase = mi * dgl_p.y_row + dgl_p.y_off + n0;
    dgl_y[ybase] = bitcast<u32>(bitcast<f32>(dgl_y[ybase]) + acc0);
    dgl_y[ybase + 1u] = bitcast<u32>(bitcast<f32>(dgl_y[ybase + 1u]) + acc1);
  } else {
    let yw = (mi * dgl_p.y_row + dgl_p.y_off) / 2u + wi;
    let old = dgl_y[yw];
    if dgl_p.y_fmt == 2u {
      let v = unpack2x16float(old);
      dgl_y[yw] = pack2x16float(vec2f(v.x + acc0, v.y + acc1));
    } else {
      let v0 = dgl_bf1(old) + acc0;
      let v1 = dgl_bf2(old) + acc1;
      dgl_y[yw] = (dgl_rne_bf16(v1) << 16u) | dgl_rne_bf16(v0);
    }
  }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DgLoraParams {
    m: u32,
    k: u32,
    rank: u32,
    n: u32,
    x_fmt: u32,
    y_fmt: u32,
    y_off: u32,
    y_row: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DgAct16 {
    Bf16,
    F16,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DgSink {
    F32,
    Bf16,
    F16,
}

struct DgLoraLeaf {
    a_words: Vec<u32>,
    b_words: Vec<u32>,
    rank: usize,
    n: usize,
    k: usize,
}

pub struct DgLora {
    layers: Vec<std::collections::HashMap<&'static str, DgLoraLeaf>>,
    pub rank: usize,
    pub matched: usize,
}

const DG_LORA_LEAVES: [&str; 7] = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

fn dg_lora_leaf_path(leaf: &str) -> String {
    match leaf {
        "q_proj" | "k_proj" | "v_proj" | "o_proj" => format!("self_attn.{leaf}"),
        _ => format!("mlp.{leaf}"),
    }
}

fn dg_pack_bf16_rows(t: &candle_core::Tensor, n: usize, k: usize) -> Result<Vec<u32>> {
    anyhow::ensure!(
        t.dims() == [n, k],
        "lora factor {:?} != [{n}, {k}]",
        t.dims()
    );
    let v: Vec<f32> = t
        .to_dtype(candle_core::DType::F32)?
        .flatten_all()?
        .to_vec1()?;
    let mut out = Vec::with_capacity(n * k / 2);
    for pair in v.chunks_exact(2) {
        let lo = half::bf16::from_f32(pair[0]).to_bits() as u32;
        let hi = half::bf16::from_f32(pair[1]).to_bits() as u32;
        out.push(lo | (hi << 16));
    }
    Ok(out)
}

impl DgLora {
    pub fn from_peft_dir(dir: &std::path::Path, cfg: &DotsDecoderConfig) -> Result<Self> {
        let adapter = nv_weights::lora_adapter::LoraAdapter::load(dir, &candle_core::Device::Cpu)?;
        Self::from_adapter(&adapter, cfg, &dir.display().to_string())
    }

    pub fn from_adapter(
        adapter: &nv_weights::lora_adapter::LoraAdapter,
        cfg: &DotsDecoderConfig,
        origin: &str,
    ) -> Result<Self> {
        let hidden = cfg.hidden_size;
        let q_rows = cfg.num_attention_heads * cfg.head_dim();
        let kv_rows = cfg.num_key_value_heads * cfg.head_dim();
        let inter = cfg.intermediate_size;
        let dims = |leaf: &str| -> (usize, usize) {
            match leaf {
                "q_proj" => (q_rows, hidden),
                "k_proj" | "v_proj" => (kv_rows, hidden),
                "o_proj" => (hidden, q_rows),
                "gate_proj" | "up_proj" => (inter, hidden),
                "down_proj" => (hidden, inter),
                _ => unreachable!("DG_LORA_LEAVES is the leaf universe"),
            }
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let mut rank = 0usize;
        let mut matched = 0usize;
        for li in 0..cfg.num_hidden_layers {
            let mut map = std::collections::HashMap::new();
            for leaf in DG_LORA_LEAVES {
                let module = format!(
                    "model.language_model.layers.{li}.{}",
                    dg_lora_leaf_path(leaf)
                );
                let Some(w) = adapter.loras.get(&module) else {
                    continue;
                };
                anyhow::ensure!(
                    !w.is_embedding,
                    "{module}: embedding lora is not servable on the graph arm"
                );
                let (n, k) = dims(leaf);
                let r = w.lora_a.dims()[0];
                let b_scaled = if w.scaling == 1.0 {
                    w.lora_b.clone()
                } else {
                    w.lora_b
                        .affine(w.scaling, 0.0)
                        .map_err(|e| anyhow::anyhow!("{module}: fold scaling: {e}"))?
                };
                anyhow::ensure!(
                    r % 2 == 0,
                    "{module}: rank {r} must be even (the expand kernel walks bf16 pairs)"
                );
                anyhow::ensure!(
                    rank == 0 || rank == r,
                    "{module}: rank {r} != adapter rank {rank}; mixed ranks are not servable"
                );
                rank = r;
                map.insert(
                    leaf,
                    DgLoraLeaf {
                        a_words: dg_pack_bf16_rows(&w.lora_a, r, k)?,
                        b_words: dg_pack_bf16_rows(&b_scaled, n, r)?,
                        rank: r,
                        n,
                        k,
                    },
                );
                matched += 1;
            }
            layers.push(map);
        }
        anyhow::ensure!(
            matched > 0,
            "lora adapter {origin} matched no dots graph projection; expected modules like \
             model.language_model.layers.0.self_attn.q_proj after normalization"
        );
        Ok(Self {
            layers,
            rank,
            matched,
        })
    }

    pub fn from_env(cfg: &DotsDecoderConfig) -> Result<Option<Self>> {
        match std::env::var(DOTS_GRAPH_LORA_DIR_ENV_HOT_LOADS_A_PEFT_ADAPTER_ON_THE_GRAPH_ARM) {
            Ok(d) if !d.is_empty() => {
                let lora = Self::from_peft_dir(std::path::Path::new(&d), cfg)?;
                eprintln!(
                    "[dots] graph lora adapter: {} modules matched at rank {} from {d}",
                    lora.matched, lora.rank
                );
                Ok(Some(lora))
            }
            _ => Ok(None),
        }
    }

    pub fn matched(&self) -> usize {
        self.matched
    }

    pub fn rank(&self) -> usize {
        self.rank
    }
}

struct DgLoraLeafGpu {
    a: wgpu::Buffer,
    b: wgpu::Buffer,
    part: wgpu::Buffer,
    rank: usize,
    n: usize,
    k: usize,
}

struct DgLoraLayerGpu {
    leaves: std::collections::HashMap<&'static str, DgLoraLeafGpu>,
}

fn upload_dg_lora_layer(
    b: &mut Builder,
    li: usize,
    layer: &std::collections::HashMap<&'static str, DgLoraLeaf>,
    m_max: usize,
) -> DgLoraLayerGpu {
    let mut leaves = std::collections::HashMap::new();
    for (leaf, l) in layer {
        let a = b.upload_u32(&format!("dg-l{li}-lora-{leaf}-a"), &l.a_words);
        let bw = b.upload_u32(&format!("dg-l{li}-lora-{leaf}-b"), &l.b_words);
        let part = b.zeros(
            &format!("dg-l{li}-lora-{leaf}-part"),
            (m_max * l.rank * 4) as u64,
        );
        leaves.insert(
            *leaf,
            DgLoraLeafGpu {
                a,
                b: bw,
                part,
                rank: l.rank,
                n: l.n,
                k: l.k,
            },
        );
    }
    DgLoraLayerGpu { leaves }
}

#[allow(clippy::too_many_arguments)]
fn push_dg_lora_slices(
    b: &mut Builder,
    label: &str,
    layer: &DgLoraLayerGpu,
    slices: &[(&'static str, usize)],
    x: &wgpu::Buffer,
    x_fmt: DgAct16,
    y: &wgpu::Buffer,
    y_fmt: DgSink,
    y_row: usize,
    m: usize,
) -> Result<()> {
    for (leaf, y_off) in slices {
        let Some(g) = layer.leaves.get(leaf) else {
            continue;
        };
        anyhow::ensure!(
            y_off % 2 == 0 && y_row.is_multiple_of(2) && g.n % 2 == 0 && g.k % 2 == 0,
            "{label}/{leaf}: lora slice geometry must be pair-aligned \
             (y_off {y_off}, y_row {y_row}, n {}, k {})",
            g.n,
            g.k
        );
        let p = DgLoraParams {
            m: m as u32,
            k: g.k as u32,
            rank: g.rank as u32,
            n: g.n as u32,
            x_fmt: match x_fmt {
                DgAct16::Bf16 => 0,
                DgAct16::F16 => 1,
            },
            y_fmt: match y_fmt {
                DgSink::F32 => 0,
                DgSink::Bf16 => 1,
                DgSink::F16 => 2,
            },
            y_off: *y_off as u32,
            y_row: y_row as u32,
        };
        anyhow::ensure!(
            g.rank <= 65535 && m <= 65535,
            "{label}/{leaf}: lora grid (rank {}, m {m}) exceeds the 65535 dispatch limit",
            g.rank
        );
        let pu = b.uni(&format!("{label}-{leaf}-p"), p);
        let shrink_grid = (g.rank as u32, m as u32, 1);
        b.push(
            &format!("{label}-{leaf}-shrink"),
            DG_LORA_SHRINK_WGSL,
            "dg_lora_shrink",
            &[(0, &pu), (1, &g.a), (2, x), (3, &g.part)],
            shrink_grid,
        )?;
        let expand_grid = (((g.n / 2) as u32).div_ceil(64), m as u32, 1);
        b.push(
            &format!("{label}-{leaf}-expand"),
            DG_LORA_EXPAND_WGSL,
            "dg_lora_expand",
            &[(0, &pu), (1, &g.b), (2, &g.part), (3, y)],
            expand_grid,
        )?;
    }
    Ok(())
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn dg_lora_probe(
    x_bf16: &[u16],
    a_bf16: &[u16],
    b_bf16: &[u16],
    y_init_words: &[u32],
    m: usize,
    k: usize,
    n: usize,
    rank: usize,
    x_f16: bool,
    y_fmt_code: u32,
    y_off: usize,
    y_row: usize,
) -> Result<Vec<u32>> {
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .map_err(|e| anyhow::anyhow!("no wgpu adapter: {e}"))?;
    let mut b = Builder::new(ctx);
    let xw = b.upload_u32("dgl-probe-x", &pack_pairs(x_bf16));
    let aw = b.upload_u32("dgl-probe-a", &pack_pairs(a_bf16));
    let bw = b.upload_u32("dgl-probe-b", &pack_pairs(b_bf16));
    let y = b.upload_u32("dgl-probe-y", y_init_words);
    let part = b.zeros("dgl-probe-part", (m * rank * 4) as u64);
    let leaf = DgLoraLeafGpu {
        a: aw,
        b: bw,
        part,
        rank,
        n,
        k,
    };
    let mut layer = DgLoraLayerGpu {
        leaves: std::collections::HashMap::new(),
    };
    layer.leaves.insert("q_proj", leaf);
    let y_fmt = match y_fmt_code {
        0 => DgSink::F32,
        1 => DgSink::Bf16,
        _ => DgSink::F16,
    };
    let x_fmt = if x_f16 { DgAct16::F16 } else { DgAct16::Bf16 };
    push_dg_lora_slices(
        &mut b,
        "dgl-probe",
        &layer,
        &[("q_proj", y_off)],
        &xw.clone(),
        x_fmt,
        &y.clone(),
        y_fmt,
        y_row,
        m,
    )?;
    b.flush_staging();
    let cb = dispatch::encode_pass_list_labeled(
        ctx,
        b.passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
        &[],
    );
    ctx.queue.submit([cb]);
    let want_words = if y_fmt_code == 0 {
        m * y_row
    } else {
        m * y_row / 2
    };
    dispatch::read_back(ctx, &y, want_words)
        .map_err(|e| anyhow::anyhow!("dg lora probe readback: {e}"))
}

struct Pass {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    bind: wgpu::BindGroup,
    grid: (u32, u32, u32),
    entry: String,
}

struct Sources {
    gemv_bf16: String,
    gemv_nvfp4: String,
    quant: String,
    common: String,
    rms: String,
    _rmsres: String,
    rope: String,
    rope_act16: String,
    fused: String,
    scvt: String,
}

impl Sources {
    fn new(fold: usize, kt: bool, attn_tree: bool) -> Self {
        let rope_raw = QWEN2_BIAS_ROPE_WGSL
            .replace(
                "const DG_FOLD_F: u32 = 1u;",
                &format!("const DG_FOLD_F: u32 = {fold}u;"),
            )
            .replace(
                "const DG_KT: u32 = 0u;",
                &format!("const DG_KT: u32 = {}u;", u32::from(kt)),
            )
            .replace(
                "const DGW_ABLATE: u32 = 0u;",
                &format!("const DGW_ABLATE: u32 = {}u;", pf_ablate_shader_phase()),
            )
            .replace(
                "const DGX_SG: u32 = 0u;",
                &format!(
                    "const DGX_SG: u32 = {}u;",
                    u32::from(attn_sg_enabled() && !attn_tree)
                ),
            )
            .replace(
                "const DGA_SG: u32 = 1u;",
                &format!("const DGA_SG: u32 = {}u;", u32::from(!attn_tree)),
            )
            .replace(
                "const DGX_ABL: u32 = 0u;",
                &format!(
                    "const DGX_ABL: u32 = {}u;",
                    std::env::var(ATTN_ABL_ENV)
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0)
                ),
            )
            .replace(
                "const DGX_PIPE: u32 = 0u;",
                &format!("const DGX_PIPE: u32 = {}u;", u32::from(attn_pipe_enabled())),
            )
            .replace(
                "const DGA_V2: u32 = 1u;",
                &format!("const DGA_V2: u32 = {}u;", u32::from(battn_v2_enabled())),
            )
            .replace(
                "const DGA_ABL: u32 = 0u;",
                &format!("const DGA_ABL: u32 = {}u;", {
                    let _ = BATTN_ABL_DELETES_THE_MONOLITHIC_WALKS_LOAD_CHAINS_TO_SPLIT_BYTES_FROM_LATENCY;
                    std::env::var(BATTN_ABL_ENV)
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok())
                        .unwrap_or(0)
                }),
            )
            .replace(
                "const DGKV16: u32 = 0u;",
                &format!("const DGKV16: u32 = {}u;", u32::from(kv_f16_enabled())),
            );
        let rope_raw = if attn_tree {
            swap_dead_subgroup_calls_for_stubs(&rope_raw)
        } else {
            rope_raw
        };
        Self {
            gemv_bf16: compose(LAGUNA_GEMV_BF16_WGSL),
            gemv_nvfp4: format!("{}\n{}", wk::gemv_nvfp4::gemv_source(), NVFP4_GEMV_WGSL),
            quant: format!(
                "{}\n{}",
                wk::gemv_nvfp4::quantize_source(),
                LAGUNA_QUANT_ROWS_WGSL
            ),
            common: compose(LAGUNA_COMMON_WGSL),
            rms: compose(wk::rmsnorm::WGSL),
            _rmsres: compose(wk::rmsnorm_residual::WGSL),
            rope: compose(&rope_raw),
            rope_act16: compose(
                &rope_raw.replace("const DGACT16: u32 = 0u;", "const DGACT16: u32 = 1u;"),
            ),
            fused: compose(DG_ATTN_FUSED_WGSL)
                .replace(
                    "const DGF_KT: u32 = 0u;",
                    &format!("const DGF_KT: u32 = {}u;", u32::from(kt)),
                )
                .replace(
                    "const DGF_KV16: u32 = 0u;",
                    &format!("const DGF_KV16: u32 = {}u;", u32::from(kv_f16_enabled())),
                ),
            scvt: compose(DG_STEEL_CVT_WGSL)
                .replace(
                    "const DSC_KT: u32 = 0u;",
                    &format!("const DSC_KT: u32 = {}u;", u32::from(kt)),
                )
                .replace(
                    "const DSC_KV16: u32 = 0u;",
                    &format!("const DSC_KV16: u32 = {}u;", u32::from(kv_f16_enabled())),
                ),
        }
    }
}

fn staging_flush_enabled() -> bool {
    !matches!(
        std::env::var("NV_DOTS_GRAPH_STAGING_FLUSH").ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

struct Builder {
    core: VramLedger,
    passes: Vec<Pass>,
    pf_lists: Vec<Vec<Pass>>,
    split: Vec<Pass>,
    pf_target: Option<usize>,
    split_mode: bool,
    state: Vec<wgpu::Buffer>,
    sk_families: u64,
    bf16out_families: u64,
    abf16_families: u64,
    brope_kvw_layers: u64,
}

impl std::ops::Deref for Builder {
    type Target = VramLedger;
    fn deref(&self) -> &VramLedger {
        &self.core
    }
}

impl std::ops::DerefMut for Builder {
    fn deref_mut(&mut self) -> &mut VramLedger {
        &mut self.core
    }
}

impl Builder {
    fn new(ctx: &'static WgpuContext) -> Self {
        Self {
            core: VramLedger::new(ctx, "dg-", staging_flush_enabled, STAGING_FLUSH_BYTES),
            passes: Vec::new(),
            pf_lists: Vec::new(),
            split: Vec::new(),
            pf_target: None,
            split_mode: false,
            state: Vec::new(),
            sk_families: 0,
            bf16out_families: 0,
            abf16_families: 0,
            brope_kvw_layers: 0,
        }
    }

    fn state_zeros(&mut self, label: &str, bytes: u64) -> wgpu::Buffer {
        let b = self.zeros(label, bytes);
        self.state.push(b.clone());
        b
    }

    fn push(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, label, source, entry)
            .map_err(|e| anyhow::anyhow!("pipeline {label}::{entry}: {e}"))?;
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        let pass = Pass {
            pipeline,
            bind,
            grid,
            entry: entry.to_string(),
        };
        if self.split_mode {
            self.split.push(pass);
        } else if let Some(t) = self.pf_target {
            self.pf_lists[t].push(pass);
        } else {
            self.passes.push(pass);
        }
        Ok(())
    }

    fn make_pass(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<Pass> {
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, label, source, entry)
            .map_err(|e| anyhow::anyhow!("pipeline {label}::{entry}: {e}"))?;
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        Ok(Pass {
            pipeline,
            bind,
            grid,
            entry: entry.to_string(),
        })
    }

    fn push_prebuilt(
        &mut self,
        entry: &str,
        pipeline: std::sync::Arc<wgpu::ComputePipeline>,
        bind: wgpu::BindGroup,
        grid: (u32, u32, u32),
    ) {
        let pass = Pass {
            pipeline,
            bind,
            grid,
            entry: entry.to_string(),
        };
        if self.split_mode {
            self.split.push(pass);
        } else if let Some(t) = self.pf_target {
            self.pf_lists[t].push(pass);
        } else {
            self.passes.push(pass);
        }
    }

    fn row_chunk(&self, hidden: usize) -> usize {
        let limit = self
            .ctx
            .caps
            .max_storage_buffer_binding_size
            .clamp(1 << 20, 1u64 << 30);
        let per_row = (hidden * 2) as u64;
        ((limit / per_row) as usize).max(2) & !1usize
    }
}

struct StepBuffers {
    tok: wgpu::Buffer,
    step: wgpu::Buffer,
    uni: wgpu::Buffer,
}

impl StepBuffers {
    fn alloc(b: &mut Builder) -> Self {
        Self {
            tok: b.upload_u32("dg-tok", &[0u32]),
            step: b.upload_u32("dg-step", &[0u32; 4]),
            uni: b.uni("dg-stepu", StepUniform::default()),
        }
    }

    fn write(&self, ctx: &WgpuContext, token: u32, pos: u32, total: u32) {
        self.write_base(ctx, token, pos, total, 0);
    }

    fn write_base(&self, ctx: &WgpuContext, token: u32, pos: u32, total: u32, kv_base: u32) {
        ctx.queue
            .write_buffer(&self.tok, 0, bytemuck::bytes_of(&token));
        let slots: [u32; 4] = [token, pos, total, kv_base];
        ctx.queue
            .write_buffer(&self.step, 0, bytemuck::cast_slice(&slots));
        ctx.queue.write_buffer(
            &self.uni,
            0,
            bytemuck::bytes_of(&StepUniform {
                tok: token,
                pos,
                total,
                sliding_start: 0,
            }),
        );
    }
}

enum Lin {
    Bf16 { w: wgpu::Buffer, n: usize, k: usize },
    Nvfp4(Nvfp4Gpu),
}

struct Act {
    x: wgpu::Buffer,
    k: usize,
    xq: wgpu::Buffer,
    xs: wgpu::Buffer,
}

struct PfBuffers {
    m: usize,
    tok: wgpu::Buffer,
    mask: wgpu::Buffer,
    splice: wgpu::Buffer,
    res: wgpu::Buffer,
    normed: wgpu::Buffer,
    normed_post: wgpu::Buffer,
    qkv: wgpu::Buffer,
    qkv_rot: wgpu::Buffer,
    heads: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    ffn_out: wgpu::Buffer,
    y_gu: wgpu::Buffer,
    act: wgpu::Buffer,
    x_f32: wgpu::Buffer,
    y_f32: wgpu::Buffer,
    zero: wgpu::Buffer,
}

struct SingleSlot {
    sel0: wgpu::Buffer,
    global1: wgpu::Buffer,
}

fn wbytes(elems: usize) -> u64 {
    (elems.div_ceil(2).max(1) * 4) as u64
}

fn load_bf16(w: &dyn nv_weights::TensorSource, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
    let t = w
        .get(name, candle_core::DType::BF16)
        .with_context(|| format!("load {name}"))?;
    anyhow::ensure!(
        t.dims() == shape,
        "{name}: shape {:?} != {shape:?}",
        t.dims()
    );
    let v: Vec<half::bf16> = t.flatten_all()?.to_vec1()?;
    Ok(v.into_iter().map(|x| x.to_bits()).collect())
}

fn layer_tensor(idx: usize, suffix: &str) -> String {
    format!("model.layers.{idx}.{suffix}")
}

pub const W4_GROUP: u32 = 64;

pub const DG_GEMV_W4_RIDES_PLACE_VALUE_X_SO_A_NIBBLE_COSTS_ONE_AND_AND_ONE_FMA: &str =
    "mlx's qmv_fast never shifts a nibble out of its place: masking a 16-bit half with \
     0xF, 0xF0, 0xF00 or 0xF000 leaves the code multiplied by 16^j, and x arrives \
     pre-multiplied by 16^-j, so the inner loop is one AND, one int-to-float and one fma per \
     weight. Two simdgroups of a 64-thread workgroup each own four output rows; a thread owns \
     sixteen k elements held in registers -- no threadgroup staging, no barriers -- computes \
     its raw-x sum beside the scaled copies, applies the gs-64 epilogue s times qdot plus b \
     times xsum per block from its own thread-local group, and one subgroupAdd per row \
     finishes the dot. Output is packed bf16 row pairs, the graph's native layout; the f32 \
     consumers (qkv, logits) get a dg_cvt expand pass.";

pub const DG_RMSRES_WGSL: &str = r#"
struct DgRmsResParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
};

@group(0) @binding(0) var<storage, read> drr_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> drr_res: array<u32>;
@group(0) @binding(2) var<storage, read> drr_w: array<u32>;
@group(0) @binding(3) var<storage, read_write> drr_out: array<u32>;
@group(0) @binding(4) var<uniform> drr_p: DgRmsResParams;

const DRR_F16OUT: u32 = 0u;

const DRR_SG: u32 = 1u;

fn drr_pack(lo: f32, hi: f32) -> u32 {
    if (DRR_F16OUT == 1u) {
        return pack2x16float(vec2<f32>(lo, hi));
    }
    return bf16_pack(lo, hi);
}

var<workgroup> drr_warp: array<f32, 8>;
var<workgroup> drr_red: array<f32, 256>;

fn drr_row_sum(local: f32, lid: u32) -> f32 {
    if (DRR_SG == 1u) {
        let wsum = subgroupAdd(local);
        if ((lid & 31u) == 0u) {
            drr_warp[lid >> 5u] = wsum;
        }
        workgroupBarrier();
        return ((drr_warp[0] + drr_warp[1]) + (drr_warp[2] + drr_warp[3]))
            + ((drr_warp[4] + drr_warp[5]) + (drr_warp[6] + drr_warp[7]));
    }
    drr_red[lid] = local;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid < s) {
            drr_red[lid] = drr_red[lid] + drr_red[lid + s];
        }
        workgroupBarrier();
    }
    return drr_red[0];
}

@compute @workgroup_size(256)
fn dg_rms(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= drr_p.batch) {
        return;
    }
    let lid = tid.x;
    let words = drr_p.words_per_row;
    let base = row * words;

    var sw: array<u32, 4>;
    var local = 0.0;
    var n = 0u;
    for (var i = lid; i < words; i = i + 256u) {
        let xw = drr_x[base + i];
        let lo = bf16_lo(xw);
        let hi = bf16_hi(xw);
        sw[n] = xw;
        n = n + 1u;
        local = local + lo * lo + hi * hi;
    }
    let tot = drr_row_sum(local, lid);
    let rms = inverseSqrt(tot / f32(drr_p.hidden) + drr_p.eps);
    n = 0u;
    for (var i = lid; i < words; i = i + 256u) {
        let s = sw[n];
        n = n + 1u;
        let ww = drr_w[i];
        drr_out[base + i] = drr_pack(
            bf16_lo(s) * rms * bf16_lo(ww),
            bf16_hi(s) * rms * bf16_hi(ww)
        );
    }
}

@compute @workgroup_size(256)
fn dg_rmsres(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= drr_p.batch) {
        return;
    }
    let lid = tid.x;
    let words = drr_p.words_per_row;
    let base = row * words;

    var sw: array<u32, 4>;
    var local = 0.0;
    var n = 0u;
    for (var i = lid; i < words; i = i + 256u) {
        let xw = drr_x[base + i];
        let rw = drr_res[base + i];
        let lo = bf16_lo(xw) + bf16_lo(rw);
        let hi = bf16_hi(xw) + bf16_hi(rw);
        let packed = bf16_pack(lo, hi);
        drr_res[base + i] = packed;
        sw[n] = packed;
        n = n + 1u;
        local = local + lo * lo + hi * hi;
    }

    let tot = drr_row_sum(local, lid);
    let rms = inverseSqrt(tot / f32(drr_p.hidden) + drr_p.eps);

    n = 0u;
    for (var i = lid; i < words; i = i + 256u) {
        let s = sw[n];
        n = n + 1u;
        let ww = drr_w[i];
        drr_out[base + i] = drr_pack(
            bf16_lo(s) * rms * bf16_lo(ww),
            bf16_hi(s) * rms * bf16_hi(ww)
        );
    }
}
"#;

pub const DG_GEMV_W4_WGSL: &str = r#"
struct DgW4Params {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<storage, read> w4g_w: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> w4g_s: array<u32>;
@group(0) @binding(2) var<storage, read> w4g_b: array<u32>;
@group(0) @binding(3) var<storage, read> w4g_x: array<vec4<u32>>;
@group(0) @binding(4) var<uniform> w4g_p: DgW4Params;
@group(0) @binding(5) var<storage, read_write> w4g_y: array<u32>;

fn w4g_wordx(ww: u32, xv: vec4<u32>, qdot: ptr<function, f32>, xsum: ptr<function, f32>) {
    for (var j = 0u; j < 4u; j = j + 1u) {
        let xw = xv[j];
        let lo = bf16_lo(xw);
        let hi = bf16_hi(xw);
        let hb = (ww >> (8u * j)) & 0xFFu;
        *xsum = *xsum + lo + hi;
        *qdot = fma(f32(hb & 0xFu), lo, *qdot);
        *qdot = fma(f32(hb & 0xF0u), hi * 0.0625, *qdot);
    }
}

const W4G_ROWS: u32 = 8u;
const W4G_LANES: u32 = 256u / W4G_ROWS;
const W4G_SG: u32 = 1u;

var<workgroup> w4g_rows: array<f32, 32>;
var<workgroup> w4g_acc: array<f32, 256>;

@compute @workgroup_size(256)
fn dg_gemv_w4_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let rh = tid / W4G_LANES;
    let lane = tid % W4G_LANES;
    let wg8 = wid.x + wid.y * w4g_p.groups_x;
    let row0 = wg8 * W4G_ROWS;
    let row = row0 + rh;
    let live = row < w4g_p.n_rows;
    let kw4 = select(0u, w4g_p.w_row_words >> 2u, live);
    let wbase = row * (w4g_p.w_row_words >> 2u);
    let sbase = row * w4g_p.scale_row_stride;

    var acc = 0.0;
    for (var i = lane; i < kw4; i = i + W4G_LANES) {
        let wv = w4g_w[wbase + i];
        let xb = i * 4u;
        var qdot = 0.0;
        var xsum = 0.0;
        w4g_wordx(wv.x, w4g_x[xb], &qdot, &xsum);
        w4g_wordx(wv.y, w4g_x[xb + 1u], &qdot, &xsum);
        w4g_wordx(wv.z, w4g_x[xb + 2u], &qdot, &xsum);
        w4g_wordx(wv.w, w4g_x[xb + 3u], &qdot, &xsum);
        let g = i >> 1u;
        acc = fma(
            bf16_decode(w4g_s[sbase + g]),
            qdot,
            fma(bf16_decode(w4g_b[sbase + g]), xsum, acc)
        );
    }
    if (W4G_SG == 1u) {
        let rsum = subgroupAdd(acc);
        if ((tid & 31u) == 0u) {
            w4g_rows[tid >> 5u] = rsum;
        }
    } else {
        w4g_acc[tid] = acc;
        workgroupBarrier();
        for (var s = 16u; s > 0u; s = s >> 1u) {
            if ((tid & 31u) < s) {
                w4g_acc[tid] = w4g_acc[tid] + w4g_acc[tid + s];
            }
            workgroupBarrier();
        }
        if ((tid & 31u) == 0u) {
            w4g_rows[tid >> 5u] = w4g_acc[tid];
        }
    }
    workgroupBarrier();
    let wpr = W4G_LANES / 32u;
    if (w4g_p.pad0 == 1u) {
        if (tid < W4G_ROWS && row0 + tid < w4g_p.n_rows) {
            var t = 0.0;
            for (var w = 0u; w < wpr; w = w + 1u) {
                t = t + w4g_rows[tid * wpr + w];
            }
            w4g_y[w4g_p.pad1 + row0 + tid] = bitcast<u32>(t);
        }
    } else if (tid < W4G_ROWS / 2u && row0 + tid * 2u < w4g_p.n_rows) {
        var lo = 0.0;
        var hi = 0.0;
        for (var w = 0u; w < wpr; w = w + 1u) {
            lo = lo + w4g_rows[(tid * 2u) * wpr + w];
            if (row0 + tid * 2u + 1u < w4g_p.n_rows) {
                hi = hi + w4g_rows[(tid * 2u + 1u) * wpr + w];
            }
        }
        w4g_y[((row0 >> 1u) + tid)] = bf16_pack(lo, hi);
    }
}
"#;

fn drr_source(f16_out: bool, tree: bool) -> String {
    let mut src = DG_RMSRES_WGSL.replace(
        "const DRR_F16OUT: u32 = 0u;",
        &format!("const DRR_F16OUT: u32 = {}u;", u32::from(f16_out)),
    );
    if tree {
        src = src.replace("const DRR_SG: u32 = 1u;", "const DRR_SG: u32 = 0u;");
        src = swap_dead_subgroup_calls_for_stubs(&src);
    }
    compose(&src)
}

fn w4g_source(rows: usize, tree: bool) -> String {
    let mut src = DG_GEMV_W4_WGSL.replace(
        "const W4G_ROWS: u32 = 8u;",
        &format!("const W4G_ROWS: u32 = {rows}u;"),
    );
    if tree {
        src = src.replace("const W4G_SG: u32 = 1u;", "const W4G_SG: u32 = 0u;");
        src = swap_dead_subgroup_calls_for_stubs(&src);
    }
    compose(&src)
}

struct DgAffine {
    w: wgpu::Buffer,
    scales: wgpu::Buffer,
    biases: wgpu::Buffer,
    n: usize,
    k: usize,
}

type W4Parts = (Vec<u32>, Vec<f32>, Vec<f32>);

fn prep_lin_w4(
    host: Vec<u16>,
    n: usize,
    k: usize,
    quant: DotsGraphQuant,
    w4_dec: bool,
) -> (Vec<u16>, Option<W4Parts>) {
    if !w4_dec {
        return (host, None);
    }
    let f: Vec<f32> = host
        .iter()
        .map(|v| f32::from_bits((*v as u32) << 16))
        .collect();
    let parts = nv_quant::mlx_affine::quantize_mlx_affine_linear(&f, n, k, W4_GROUP as usize);
    let host2 = if quant == DotsGraphQuant::AffineW4 {
        nv_quant::mlx_affine::dequantize_mlx_affine_linear(
            &parts.0,
            &parts.1,
            &parts.2,
            n,
            k,
            W4_GROUP as usize,
        )
        .iter()
        .map(|v| half::bf16::from_f32(*v).to_bits())
        .collect()
    } else {
        host
    };
    (host2, Some(parts))
}

pub const MLX_SNAPSHOT_IS_THE_LOAD_TIME_QUANTIZER_ALREADY_RUN: &str =
    "mlx-community/dots.ocr-4bit stores exactly what quantize_mlx_affine_linear emits from the \
     rednote bf16 master -- u32 words with LSB-first nibbles, gs-64 bf16 scales and biases, \
     verified bitwise on embed_tokens, lm_head and trunk projections -- so a graph constructed \
     from the mlx snapshot is the rednote AffineW4 graph without the 5.7 GB bf16 intermediate \
     that cannot transit wasm32's 4 GiB linear memory; the fused qkv and gate-up uploads are \
     row concatenations, which the per-row group layout makes a plain buffer concat, and the \
     bf16 prefill twins come from the same dequant expression prep_lin_w4 applies";

pub struct DgMlxWeights<'a> {
    loader: &'a nv_weights::WeightLoader,
}

impl<'a> DgMlxWeights<'a> {
    pub fn new(loader: &'a nv_weights::WeightLoader) -> Result<Self> {
        let _ = MLX_SNAPSHOT_IS_THE_LOAD_TIME_QUANTIZER_ALREADY_RUN;
        anyhow::ensure!(
            loader.has("language_model.model.embed_tokens.scales"),
            "not an mlx affine-u4 dots.ocr snapshot: \
             language_model.model.embed_tokens.scales is missing"
        );
        Ok(Self { loader })
    }

    fn map(name: &str) -> String {
        if name.starts_with("model.") || name.starts_with("lm_head.") {
            format!(
                "language_model.model.{}",
                name.strip_prefix("model.").unwrap_or(name)
            )
        } else {
            name.to_string()
        }
    }

    fn scales_name(mapped: &str) -> Option<String> {
        mapped
            .strip_suffix(".weight")
            .map(|stem| format!("{stem}.scales"))
    }

    fn quantized(&self, mapped: &str) -> bool {
        Self::scales_name(mapped)
            .map(|s| self.loader.has(&s))
            .unwrap_or(false)
    }

    fn raw_bf16_f32(&self, name: &str, expect: usize) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.loader.st_dtype_of(name) == Some(nv_weights::StDtype::BF16),
            "{name}: expected BF16 storage, got {:?}",
            self.loader.st_dtype_of(name)
        );
        let bytes = self.loader.raw_bytes(name)?;
        anyhow::ensure!(
            bytes.len() == 2 * expect,
            "{name}: {} bytes != {expect} bf16 values",
            bytes.len()
        );
        Ok(bytes
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect())
    }

    pub fn w4(&self, name: &str, n: usize, k: usize) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
        let gs = W4_GROUP as usize;
        let mapped = Self::map(name);
        let scales_name = Self::scales_name(&mapped)
            .ok_or_else(|| anyhow::anyhow!("{name}: mlx w4 reads .weight tensors"))?;
        let shape = self
            .loader
            .shape_of(&mapped)
            .ok_or_else(|| anyhow::anyhow!("{mapped}: missing from the mlx snapshot"))?;
        anyhow::ensure!(
            shape == [n, k / 8],
            "{mapped}: packed shape {shape:?} != [{n}, {}]",
            k / 8
        );
        anyhow::ensure!(
            self.loader.st_dtype_of(&mapped) == Some(nv_weights::StDtype::U32),
            "{mapped}: expected U32 packed nibbles, got {:?}",
            self.loader.st_dtype_of(&mapped)
        );
        let wbytes = self.loader.raw_bytes(&mapped)?;
        let words: Vec<u32> = wbytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let scales = self.raw_bf16_f32(&scales_name, n * (k / gs))?;
        let biases = self.raw_bf16_f32(&scales_name.replace(".scales", ".biases"), n * (k / gs))?;
        Ok((words, scales, biases))
    }

    pub fn dequant_bf16_host(&self, name: &str, n: usize, k: usize) -> Result<Vec<u16>> {
        let (words, scales, biases) = self.w4(name, n, k)?;
        Ok(dequant_mlx_bf16_host(&words, &scales, &biases, n, k))
    }
}

impl nv_weights::TensorSource for DgMlxWeights<'_> {
    fn get(&self, name: &str, dtype: candle_core::DType) -> Result<candle_core::Tensor> {
        let mapped = Self::map(name);
        if self.quantized(&mapped) {
            let shape = self
                .loader
                .shape_of(&mapped)
                .ok_or_else(|| anyhow::anyhow!("{mapped}: missing from the mlx snapshot"))?;
            anyhow::ensure!(
                shape.len() == 2,
                "{mapped}: packed shape {shape:?} is not 2-d"
            );
            let (n, k) = (shape[0], shape[1] * 8);
            let host: Vec<half::bf16> = self
                .dequant_bf16_host(name, n, k)?
                .into_iter()
                .map(half::bf16::from_bits)
                .collect();
            let t = candle_core::Tensor::from_vec(host, (n, k), &candle_core::Device::Cpu)?;
            return Ok(if dtype == candle_core::DType::BF16 {
                t
            } else {
                t.to_dtype(dtype)?
            });
        }
        self.loader.get(&mapped, dtype)
    }

    fn has(&self, name: &str) -> bool {
        self.loader.has(&Self::map(name))
    }
}

fn dequant_mlx_bf16_host(
    words: &[u32],
    scales: &[f32],
    biases: &[f32],
    n: usize,
    k: usize,
) -> Vec<u16> {
    let gs = W4_GROUP as usize;
    let groups = k / gs;
    let mut out = vec![0u16; n * k];
    for row in 0..n {
        for kw in 0..k / 8 {
            let word = words[row * (k / 8) + kw];
            for e in 0..8 {
                let k_idx = kw * 8 + e;
                let g = k_idx / gs;
                let q = ((word >> (4 * e)) & 0xF) as f32;
                let v = q * scales[row * groups + g] + biases[row * groups + g];
                out[row * k + k_idx] = half::bf16::from_f32(v).to_bits();
            }
        }
    }
    out
}

fn lin_prep(
    weights: &dyn nv_weights::TensorSource,
    mlx: Option<&DgMlxWeights>,
    parts_of: &[(String, usize)],
    k: usize,
    quant: DotsGraphQuant,
    w4_dec: bool,
) -> Result<(Vec<u16>, Option<W4Parts>)> {
    let n: usize = parts_of.iter().map(|(_, rows)| rows).sum();
    match mlx {
        Some(m) => {
            let gs = W4_GROUP as usize;
            let mut words = Vec::with_capacity(n * k / 8);
            let mut scales = Vec::with_capacity(n * (k / gs));
            let mut biases = Vec::with_capacity(n * (k / gs));
            for (name, rows) in parts_of {
                let (w, s, b) = m.w4(name, *rows, k)?;
                words.extend(w);
                scales.extend(s);
                biases.extend(b);
            }
            let host = dequant_mlx_bf16_host(&words, &scales, &biases, n, k);
            Ok((host, w4_dec.then_some((words, scales, biases))))
        }
        None => {
            let mut host = Vec::with_capacity(n * k);
            for (name, rows) in parts_of {
                host.extend_from_slice(&load_bf16(weights, name, &[*rows, k])?);
            }
            Ok(prep_lin_w4(host, n, k, quant, w4_dec))
        }
    }
}

fn upload_affine_w4(b: &mut Builder, label: &str, parts: &W4Parts, n: usize, k: usize) -> DgAffine {
    let pack = |v: &[f32]| -> Vec<u32> {
        v.iter()
            .map(|s| (half::bf16::from_f32(*s).to_bits()) as u32)
            .collect()
    };
    DgAffine {
        w: b.upload_u32(&format!("{label}-w4"), &parts.0),
        scales: b.upload_u32(&format!("{label}-w4s"), &pack(&parts.1)),
        biases: b.upload_u32(&format!("{label}-w4b"), &pack(&parts.2)),
        n,
        k,
    }
}

pub const W4_GEMV_BOUND_IS_FIVE_ISSUE_SLOTS_PER_THIRTYTWO_WEIGHTS: &str =
    "The w4 gemv ladder is flat: rows per workgroup bracketed at 8 (4: 7.35, 8: 7.29, 16: 7.48 \
     ms/step), and staging x in a threadgroup slab to cut its eightfold re-read was \
     catastrophic (575 ms/step -- the 17.9 KB slab collapses occupancy, and x was \
     cache-resident anyway). The bound worth naming: about 4 ms of step time moves 0.96 GB of \
     weights at ~240 GB/s effective because every 32 weights cost five load-issue slots (one \
     weight vec4, four x vec4s) plus 32 mask-convert-fma chains, and the decode k of 1536 \
     leaves 1.5 iterations per lane -- no loop to unroll, no density left at this geometry. \
     Descending further needs fewer x issue slots per weight, which WGSL scalar ALU does not \
     offer; decode closes here unless the operand model changes. The authorized \
     follow-up -- an f16 activation model for the decode chain, vec2<f16> fma in the nibble \
     inner loop with the five gemv-feeding buffers stored f16 at their producers -- was built \
     end to end, held all twelve stable zeros on the corpus, and measured SLOWER: 7.95/8.11 \
     ms/step against the f32-operand arm's 7.26, because naga's f16 path pays unpack-convert \
     chains without a packed-ALU dividend (and u32-to-vec2<f16> bitcast does not lower at \
     all). Reverted whole; decode closes at 7.26 with this constant as its bound.";

fn push_gemv_w4(
    b: &mut Builder,
    label: &str,
    w: &DgAffine,
    x: &wgpu::Buffer,
    y_pairs: &wgpu::Buffer,
    out_f32: bool,
    msl_x16: bool,
) -> Result<()> {
    if msl_x16 {
        let pls = decode_gemv_msl::msl_gemv_pipelines(b.ctx)?;
        let grid = b.grid1(w.n as u64, decode_gemv_msl::MSL_GEMV_ROWS);
        let p = b.uni(
            &format!("{label}-p"),
            decode_gemv_msl::DgMslW4Params {
                n_rows: w.n as u32,
                w_vec4: (w.k / 32) as u32,
                scale_stride: (w.k / W4_GROUP as usize) as u32,
                groups_x: grid.0,
                out_f32: u32::from(out_f32),
                y_off: 0,
                pad0: 0,
                pad1: 0,
            },
        );
        let bind = b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout: &pls.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: w.w.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w.scales.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: w.biases.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: x.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: y_pairs.as_entire_binding(),
                },
            ],
        });
        b.push_prebuilt(
            decode_gemv_msl::MSL_GEMV_ENTRY,
            pls.pipeline.clone(),
            bind,
            (grid.0, grid.1, 1),
        );
        return Ok(());
    }
    anyhow::ensure!(
        w.n.is_multiple_of(2) && w.k.is_multiple_of(2 * W4_GROUP as usize),
        "{label}: dg_gemv_w4_pk reads whole weight vec4s inside gs-64 groups, got n {} k {}",
        w.n,
        w.k
    );
    let _ = W4_GEMV_BOUND_IS_FIVE_ISSUE_SLOTS_PER_THIRTYTWO_WEIGHTS;
    let rows = g4_env("NV_DOTS_W4_ROWS", 8) as usize;
    let grid = b.grid1(w.n as u64, rows as u32);
    let p = b.uni(
        &format!("{label}-p"),
        DgW4Params {
            n_rows: w.n as u32,
            k_elems: w.k as u32,
            w_row_words: (w.k / 8) as u32,
            scale_row_stride: (w.k / W4_GROUP as usize) as u32,
            groups_x: grid.0,
            pad0: u32::from(out_f32),
            pad1: 0,
            pad2: 0,
        },
    );
    b.push(
        label,
        &w4g_source(rows, sg_tree_sites(b.ctx.subgroup_width()).w4),
        "dg_gemv_w4_pk",
        &[
            (0, &w.w),
            (1, &w.scales),
            (2, &w.biases),
            (3, x),
            (4, &p),
            (5, y_pairs),
        ],
        (grid.0, grid.1, 1),
    )
}

fn push_gemv_w4_gsilu(
    b: &mut Builder,
    label: &str,
    w: &DgAffine,
    x: &wgpu::Buffer,
    act_pairs: &wgpu::Buffer,
) -> Result<()> {
    let pls = decode_gemv_msl::msl_gemv_pipelines(b.ctx)?;
    anyhow::ensure!(
        w.n.is_multiple_of(8),
        "{label}: dg_msl_w4_gemv_gsilu pairs four gate rows with their four up rows per \
         workgroup, got n {}",
        w.n
    );
    let grid = b.grid1(w.n as u64, decode_gemv_msl::MSL_GEMV_ROWS);
    let p = b.uni(
        &format!("{label}-p"),
        decode_gemv_msl::DgMslW4Params {
            n_rows: w.n as u32,
            w_vec4: (w.k / 32) as u32,
            scale_stride: (w.k / W4_GROUP as usize) as u32,
            groups_x: grid.0,
            out_f32: 0,
            y_off: 0,
            pad0: 0,
            pad1: 0,
        },
    );
    let bind = b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pls.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: w.w.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: w.scales.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: w.biases.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: p.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: act_pairs.as_entire_binding(),
            },
        ],
    });
    b.push_prebuilt(
        decode_gemv_msl::MSL_GEMV_GSILU_ENTRY,
        pls.gsilu_pipeline.clone(),
        bind,
        (grid.0, grid.1, 1),
    );
    Ok(())
}

struct RmsFuseSite<'a> {
    rms_mode: u32,
    fold_res: bool,
    nw: &'a wgpu::Buffer,
    res: &'a wgpu::Buffer,
    res_f32: &'a wgpu::Buffer,
    eps: f32,
}

fn push_gemv_w4_rms(
    b: &mut Builder,
    label: &str,
    w: &DgAffine,
    x: &wgpu::Buffer,
    y_pairs: &wgpu::Buffer,
    out_f32: bool,
    site: RmsFuseSite<'_>,
) -> Result<()> {
    let pls = decode_gemv_msl::msl_gemv_pipelines(b.ctx)?;
    let grid = b.grid1(w.n as u64, decode_gemv_msl::MSL_GEMV_ROWS);
    anyhow::ensure!(
        site.rms_mode == 0 || !site.fold_res,
        "{label}: a fold kernel must not also norm -- its in-place res write would race the \
         prologue reads"
    );
    let p = b.uni(
        &format!("{label}-p"),
        decode_gemv_msl::DgMslW4RmsParams {
            n_rows: w.n as u32,
            w_vec4: (w.k / 32) as u32,
            scale_stride: (w.k / W4_GROUP as usize) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            y_off: 0,
            rms_mode: site.rms_mode,
            fold_res: u32::from(site.fold_res),
            eps: site.eps,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        },
    );
    let bind = b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &pls.rms_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: w.w.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: w.scales.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: w.biases.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: x.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: p.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: y_pairs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: site.nw.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: site.res.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: site.res_f32.as_entire_binding(),
            },
        ],
    });
    b.push_prebuilt(
        decode_gemv_msl::MSL_GEMV_RMS_ENTRY,
        pls.rms_pipeline.clone(),
        bind,
        (grid.0, grid.1, 1),
    );
    Ok(())
}

fn upload_f32_of_bf16(b: &mut Builder, label: &str, host: &[u16]) -> wgpu::Buffer {
    let f: Vec<f32> = host
        .iter()
        .map(|v| f32::from_bits((*v as u32) << 16))
        .collect();
    b.upload_f32(label, &f)
}

fn upload_lin(
    b: &mut Builder,
    label: &str,
    host: &[u16],
    n: usize,
    k: usize,
    quant: DotsGraphQuant,
) -> Result<Lin> {
    anyhow::ensure!(
        host.len() == n * k,
        "{label}: {} values, want [{n}, {k}]",
        host.len()
    );
    match quant {
        DotsGraphQuant::Bf16 | DotsGraphQuant::AffineW4 => Ok(Lin::Bf16 {
            w: b.upload_u32(label, &pack_pairs(host)),
            n,
            k,
        }),
        DotsGraphQuant::Nvfp4 => {
            anyhow::ensure!(
                k.is_multiple_of(4 * NVFP4_BLOCK),
                "{label}: nvfp4 needs k divisible by {} for the 4-block scale word, got {k}",
                4 * NVFP4_BLOCK
            );
            let h = quantize_nvfp4_host(host, n, k);
            anyhow::ensure!(
                h.input_global == 1.0,
                "{label}: host nvfp4 quantization must leave the activation global at 1.0 so one \
                 quantize pass can feed every projection reading the same activation"
            );
            Ok(Lin::Nvfp4(upload_nvfp4(b, label, &h)))
        }
    }
}

fn alloc_act(b: &mut Builder, label: &str, k: usize, quant: DotsGraphQuant) -> Act {
    let x = b.zeros(label, wbytes(k));
    act_over(b, label, x, k, quant)
}

fn act_over(b: &mut Builder, label: &str, x: wgpu::Buffer, k: usize, quant: DotsGraphQuant) -> Act {
    let (qb, sb) = match quant {
        DotsGraphQuant::Bf16 | DotsGraphQuant::AffineW4 => (4u64, 4u64),
        DotsGraphQuant::Nvfp4 => ((k / 2) as u64, (k / NVFP4_BLOCK).div_ceil(4) as u64 * 4),
    };
    Act {
        x,
        k,
        xq: b.zeros(&format!("{label}-xq"), qb),
        xs: b.zeros(&format!("{label}-xs"), sb),
    }
}

fn push_quant(b: &mut Builder, s: &Sources, label: &str, a: &Act, sl: &SingleSlot) -> Result<()> {
    let k_blocks = a.k / NVFP4_BLOCK;
    anyhow::ensure!(
        k_blocks.is_multiple_of(4),
        "{label}: quantize rows needs k/{NVFP4_BLOCK} divisible by 4, got {k_blocks}"
    );
    let p = b.uni(
        &format!("{label}-p"),
        QuantRowsParams {
            k_blocks: k_blocks as u32,
            n_slots: 1,
            use_sel: 0,
            x_slot_stride_elems: 0,
            ..Default::default()
        },
    );
    let gx = (k_blocks as u32).div_ceil(256).max(1);
    b.push(
        label,
        &s.quant,
        "lgw_quant_rows",
        &[
            (10, &a.x),
            (11, &p),
            (12, &a.xq),
            (13, &a.xs),
            (14, &sl.sel0),
            (15, &sl.global1),
        ],
        (gx, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemv(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Lin,
    x: &Act,
    y: &wgpu::Buffer,
    out_f32: bool,
    y_off_words: usize,
) -> Result<()> {
    match w {
        Lin::Bf16 { w, n, k } => {
            anyhow::ensure!(*k == x.k, "{label}: gemv k {k} != activation k {}", x.k);
            let rows = g4_rows_for(*n);
            let grid = b.grid1(n.div_ceil(rows as usize) as u64, 1);
            if k.is_multiple_of(8) {
                let p = b.uni(
                    &format!("{label}-p"),
                    DgG4Params {
                        n_rows: *n as u32,
                        k_vecs: (k / 8) as u32,
                        w_row_vecs: (k / 8) as u32,
                        groups_x: grid.0,
                        out_f32: u32::from(out_f32),
                        y_off_words: y_off_words as u32,
                        pad0: 0,
                        pad1: 0,
                    },
                );
                let src4 = s.rope.replace(
                    "const DG4_ROWS: u32 = 2u;",
                    &format!("const DG4_ROWS: u32 = {rows}u;"),
                );
                return b.push(
                    label,
                    &src4,
                    "dg_gemv_bf16_v4",
                    &[(120, w), (121, &x.x), (122, &p), (123, y)],
                    (grid.0, grid.1, 1),
                );
            }
            let grid = b.grid1(n.div_ceil(2) as u64, 1);
            let p = b.uni(
                &format!("{label}-p"),
                GemvBf16Params {
                    n_rows: *n as u32,
                    k_words: (k / 2) as u32,
                    groups_x: grid.0,
                    out_f32: u32::from(out_f32),
                    w_row_words: (k / 2) as u32,
                    y_off_words: y_off_words as u32,
                    alpha: 1.0,
                    ..Default::default()
                },
            );
            b.push(
                label,
                &s.gemv_bf16,
                "lgw_gemv_bf16",
                &[(0, w), (1, &x.x), (2, &p), (3, y)],
                grid,
            )
        }
        Lin::Nvfp4(w) if nvfp4_mrow4_enabled() => {
            anyhow::ensure!(
                w.k == x.k,
                "{label}: gemv k {} != activation k {}",
                w.k,
                x.k
            );
            anyhow::ensure!(
                !out_f32,
                "{label}: the nvfp4 gemv writes bf16 pairs, so an f32 sink needs the bf16 arm"
            );
            let k_blocks = w.k / NVFP4_BLOCK;
            let grid = b.grid1(w.n.div_ceil(4) as u64, 1);
            let p = b.uni(
                &format!("{label}-p"),
                GemvNvfp4Mrow4Params {
                    alpha: w.alpha,
                    n_rows: w.n as u32,
                    k_blocks: k_blocks as u32,
                    k_tiles: k_blocks.div_ceil(4) as u32,
                    groups_x: grid.0,
                    ..Default::default()
                },
            );
            b.push(
                label,
                &s.gemv_nvfp4,
                "dg_gemv_nvfp4_mrow4",
                &[(90, &w.w), (91, &w.scales), (92, &x.x), (93, &p), (94, y)],
                grid,
            )
        }
        Lin::Nvfp4(w) => {
            anyhow::ensure!(
                w.k == x.k,
                "{label}: gemv k {} != activation k {}",
                w.k,
                x.k
            );
            anyhow::ensure!(
                !out_f32,
                "{label}: the nvfp4 gemv writes bf16 pairs, so an f32 sink needs the bf16 arm"
            );
            let k_blocks = w.k / NVFP4_BLOCK;
            let grid = b.grid1(w.n.div_ceil(NVFP4_GEMV_ROWS_PER_WORKGROUP) as u64, 1);
            let _ = NVFP4_ACTIVATION_GLOBAL_IS_A_POWER_OF_TWO_SO_IT_ADDS_NO_ROUNDING;
            let _ =
                NVFP4_GEMV_SPENDS_32_LANES_ON_A_ROW_SO_A_1536_WIDE_K_STILL_AMORTIZES_THE_REDUCTION;
            let _ =
                NVFP4_GEMV_CALLS_ONE_DOT4I8PACKED_PER_STATEMENT_BECAUSE_THE_MSL_POLYFILL_COLLIDES;
            let p = b.uni(
                &format!("{label}-p"),
                GemvNvfp4Params {
                    alpha: w.alpha
                        / NVFP4_ACTIVATION_GLOBAL_2POW7_LANDS_DECODE_BLOCK_SCALES_INSIDE_THE_E4M3_WINDOW,
                    n_rows: w.n as u32,
                    k_blocks: k_blocks as u32,
                    k_tiles: k_blocks.div_ceil(4) as u32,
                    groups_x: grid.0,
                    ..Default::default()
                },
            );
            b.push(
                label,
                &s.gemv_nvfp4,
                "dg_gemv_nvfp4",
                &[
                    (10, &w.w),
                    (11, &w.scales),
                    (12, &x.xq),
                    (13, &x.xs),
                    (14, &p),
                    (15, y),
                ],
                grid,
            )
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn push_rmsnorm(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
    batch: usize,
) -> Result<()> {
    push_rmsnorm_fmt(b, s, label, x, w, y, hidden, eps, batch, false)
}

#[allow(clippy::too_many_arguments)]
fn push_rmsnorm_fmt(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
    batch: usize,
    f16_out: bool,
) -> Result<()> {
    let p = b.uni(
        &format!("{label}-p"),
        RmsParams {
            hidden: hidden as u32,
            batch: batch as u32,
            eps,
            words_per_row: (hidden / 2) as u32,
        },
    );
    let grid = b.grid1(batch as u64, 1);
    if f16_out {
        return b.push(
            label,
            &drr_source(true, sg_tree_sites(b.ctx.subgroup_width()).rms),
            "dg_rms",
            &[(0, x), (2, w), (3, y), (4, &p)],
            grid,
        );
    }
    b.push(
        label,
        &s.rms,
        "rmsnorm_bf16",
        &[(0, x), (1, w), (2, y), (3, &p)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_rmsnorm_residual(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    delta: &wgpu::Buffer,
    res: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
    batch: usize,
) -> Result<()> {
    push_rmsnorm_residual_fmt(b, s, label, delta, res, w, y, hidden, eps, batch, false)
}

#[allow(clippy::too_many_arguments)]
fn push_rmsnorm_residual_fmt(
    b: &mut Builder,
    _s: &Sources,
    label: &str,
    delta: &wgpu::Buffer,
    res: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
    batch: usize,
    f16_out: bool,
) -> Result<()> {
    let p = b.uni(
        &format!("{label}-p"),
        RmsParams {
            hidden: hidden as u32,
            batch: batch as u32,
            eps,
            words_per_row: (hidden / 2) as u32,
        },
    );
    let grid = b.grid1(batch as u64, 1);
    anyhow::ensure!(
        hidden / 2 <= 4 * 256,
        "{label}: dg_rmsres stashes each lane's row words in four registers, hidden {hidden}"
    );
    b.push(
        label,
        &drr_source(f16_out, sg_tree_sites(b.ctx.subgroup_width()).rms),
        "dg_rmsres",
        &[(0, delta), (1, res), (2, w), (3, y), (4, &p)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
#[derive(Clone, Copy)]
struct RopeSite<'a> {
    src: &'a wgpu::Buffer,
    bias: &'a wgpu::Buffer,
    cos: &'a wgpu::Buffer,
    sin: &'a wgpu::Buffer,
    dst: &'a wgpu::Buffer,
    n_rows: usize,
    rope_rows: usize,
    head_dim: usize,
    has_bias: bool,
    src_f32: bool,
    slots: usize,
    src_slot_words: usize,
    dst_slot_words: usize,
}

fn push_bias_rope(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    site: RopeSite<'_>,
    step: &StepBuffers,
) -> Result<()> {
    let p = b.uni(
        &format!("{label}-p"),
        BiasRopeParams {
            n_rows: site.n_rows as u32,
            head_dim: site.head_dim as u32,
            half: (site.head_dim / 2) as u32,
            rope_rows: site.rope_rows as u32,
            has_bias: u32::from(site.has_bias),
            src_f32: u32::from(site.src_f32),
            src_slot_words: site.src_slot_words as u32,
            dst_slot_words: site.dst_slot_words as u32,
        },
    );
    b.push(
        label,
        &s.rope,
        "dg_bias_rope",
        &[
            (0, site.src),
            (1, site.bias),
            (2, site.cos),
            (3, site.sin),
            (4, &step.step),
            (5, &p),
            (6, site.dst),
        ],
        (site.n_rows as u32, site.slots as u32, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_kv_write(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    src: &wgpu::Buffer,
    step: &StepBuffers,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    kv_rows: usize,
    k_off_rows: usize,
    v_off_rows: usize,
    capacity: usize,
    slots: usize,
    src_slot_words: usize,
    n_kv_heads: usize,
) -> Result<()> {
    let kv_words = kv_rows / 2;
    let p = b.uni(
        &format!("{label}-p"),
        KvWriteParams {
            kv_words: kv_words as u32,
            kv_capacity: capacity as u32,
            k_off_words: (k_off_rows / 2) as u32,
            v_off_words: (v_off_rows / 2) as u32,
            src_slot_words: src_slot_words as u32,
            n_kv_heads: n_kv_heads as u32,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.rope,
        "dg_kv_write",
        &[(10, src), (11, &step.step), (12, &p), (13, kc), (14, vc)],
        ((kv_words as u32).div_ceil(64).max(1), slots as u32, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_bias_rope_kvw(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    site: RopeSite<'_>,
    step: &StepBuffers,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    kv_rows: usize,
    capacity: usize,
) -> Result<()> {
    let _ = ROPE_KV_FUSE_DELETES_ONE_PASS_PER_LAYER_BECAUSE_THE_BRACKET_PRICED_EVERY_PASS_BOUNDARY;
    let p = b.uni(
        &format!("{label}-p"),
        BiasRopeParams {
            n_rows: site.n_rows as u32,
            head_dim: site.head_dim as u32,
            half: (site.head_dim / 2) as u32,
            rope_rows: site.rope_rows as u32,
            has_bias: u32::from(site.has_bias),
            src_f32: u32::from(site.src_f32),
            src_slot_words: site.src_slot_words as u32,
            dst_slot_words: site.dst_slot_words as u32,
        },
    );
    let kp = b.uni(
        &format!("{label}-kp"),
        KvWriteParams {
            kv_words: (kv_rows / 2) as u32,
            kv_capacity: capacity as u32,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.rope,
        "dg_bias_rope_kvw",
        &[
            (0, site.src),
            (1, site.bias),
            (2, site.cos),
            (3, site.sin),
            (4, &step.step),
            (5, &p),
            (6, site.dst),
            (12, &kp),
            (13, kc),
            (14, vc),
        ],
        (site.n_rows as u32, site.slots as u32, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_silu_mul(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    gate_up: &wgpu::Buffer,
    out: &wgpu::Buffer,
    inter: usize,
    slots: usize,
    src_slot_words: usize,
    dst_slot_words: usize,
) -> Result<()> {
    push_silu_mul_fmt(
        b,
        s,
        label,
        gate_up,
        out,
        inter,
        slots,
        src_slot_words,
        dst_slot_words,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_silu_mul_fmt(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    gate_up: &wgpu::Buffer,
    out: &wgpu::Buffer,
    inter: usize,
    slots: usize,
    src_slot_words: usize,
    dst_slot_words: usize,
    f16_out: bool,
) -> Result<()> {
    let n_words = inter / 2;
    let p = b.uni(
        &format!("{label}-p"),
        SiluMulParams {
            n_words: n_words as u32,
            gate_off_words: 0,
            up_off_words: n_words as u32,
            src_slot_words: src_slot_words as u32,
            dst_slot_words: dst_slot_words as u32,
            ..Default::default()
        },
    );
    b.push(
        label,
        if f16_out { &s.rope_act16 } else { &s.rope },
        "dg_silu_mul",
        &[(20, gate_up), (21, &p), (22, out)],
        ((n_words as u32).div_ceil(64).max(1), slots as u32, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_attn(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    q: &wgpu::Buffer,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    step: &StepBuffers,
    out: &wgpu::Buffer,
    cfg: &DotsDecoderConfig,
    capacity: usize,
    slots: usize,
    q_slot_words: usize,
    out_slot_words: usize,
    f16_out: bool,
) -> Result<()> {
    let n_q = cfg.num_attention_heads;
    let p = b.uni(
        &format!("{label}-p"),
        AttnParams {
            n_q_heads: n_q as u32,
            n_kv_heads: cfg.num_key_value_heads as u32,
            head_dim: cfg.head_dim() as u32,
            gqa_group: (n_q / cfg.num_key_value_heads) as u32,
            kv_capacity: capacity as u32,
            q_slot_words: q_slot_words as u32,
            out_slot_words: out_slot_words as u32,
            scale: (cfg.head_dim() as f32).powf(-0.5),
        },
    );
    b.push(
        label,
        if f16_out { &s.rope_act16 } else { &s.rope },
        "dg_attn",
        &[
            (30, q),
            (31, kc),
            (32, vc),
            (33, &step.step),
            (34, &p),
            (35, out),
        ],
        (n_q as u32, slots as u32, 1),
    )
}

struct RopeKvwSite<'a> {
    kc: &'a wgpu::Buffer,
    vc: &'a wgpu::Buffer,
    bslot: &'a wgpu::Buffer,
    kv_rows: usize,
    k_off_rows: usize,
    v_off_rows: usize,
    capacity: usize,
    n_kv_heads: usize,
}

#[allow(clippy::too_many_arguments)]
fn region_rope_pass(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    site: RopeSite<'_>,
    step: &StepBuffers,
    bpos: &wgpu::Buffer,
    kvw: Option<RopeKvwSite<'_>>,
) -> Result<Pass> {
    let p = b.uni(
        &format!("{label}-p"),
        BiasRopeParams {
            n_rows: site.n_rows as u32,
            head_dim: site.head_dim as u32,
            half: (site.head_dim / 2) as u32,
            rope_rows: site.rope_rows as u32,
            has_bias: u32::from(site.has_bias),
            src_f32: u32::from(site.src_f32),
            src_slot_words: site.src_slot_words as u32,
            dst_slot_words: site.dst_slot_words as u32,
        },
    );
    let _ = step;
    let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
        (0, site.src),
        (1, site.bias),
        (2, site.cos),
        (3, site.sin),
        (5, &p),
        (6, site.dst),
        (7, bpos),
    ];
    let entry = if kvw.is_some() {
        "dg_bias_rope_kvw_b"
    } else {
        "dg_bias_rope_b"
    };
    let kp;
    if let Some(kv) = &kvw {
        kp = b.uni(
            &format!("{label}-kp"),
            KvWriteParams {
                kv_words: (kv.kv_rows / 2) as u32,
                kv_capacity: kv.capacity as u32,
                k_off_words: (kv.k_off_rows / 2) as u32,
                v_off_words: (kv.v_off_rows / 2) as u32,
                src_slot_words: site.src_slot_words as u32,
                n_kv_heads: kv.n_kv_heads as u32,
                ..Default::default()
            },
        );
        binds.push((12, &kp));
        binds.push((13, kv.kc));
        binds.push((14, kv.vc));
        binds.push((15, kv.bslot));
    }
    b.make_pass(
        label,
        &s.rope,
        entry,
        &binds,
        (site.n_rows as u32, site.slots as u32, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn region_kv_pass(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    entry: &str,
    src: &wgpu::Buffer,
    step: &StepBuffers,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    kv_rows: usize,
    k_off_rows: usize,
    v_off_rows: usize,
    capacity: usize,
    slots: usize,
    src_slot_words: usize,
    n_kv_heads: usize,
    bslot: Option<&wgpu::Buffer>,
) -> Result<Pass> {
    let kv_words = kv_rows / 2;
    let p = b.uni(
        &format!("{label}-p"),
        KvWriteParams {
            kv_words: kv_words as u32,
            kv_capacity: capacity as u32,
            k_off_words: (k_off_rows / 2) as u32,
            v_off_words: (v_off_rows / 2) as u32,
            src_slot_words: src_slot_words as u32,
            n_kv_heads: n_kv_heads as u32,
            ..Default::default()
        },
    );
    let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![(10, src), (12, &p), (13, kc), (14, vc)];
    if let Some(bs) = bslot {
        binds.push((15, bs));
    } else {
        binds.push((11, &step.step));
    }
    b.make_pass(
        label,
        &s.rope,
        entry,
        &binds,
        ((kv_words as u32).div_ceil(64).max(1), slots as u32, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn region_attn_pass(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    entry: &str,
    q: &wgpu::Buffer,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    step: &StepBuffers,
    out: &wgpu::Buffer,
    cfg: &DotsDecoderConfig,
    capacity: usize,
    slots: usize,
    q_slot_words: usize,
    out_slot_words: usize,
    brange: Option<&wgpu::Buffer>,
) -> Result<Pass> {
    let n_q = cfg.num_attention_heads;
    let p = b.uni(
        &format!("{label}-p"),
        AttnParams {
            n_q_heads: n_q as u32,
            n_kv_heads: cfg.num_key_value_heads as u32,
            head_dim: cfg.head_dim() as u32,
            gqa_group: (n_q / cfg.num_key_value_heads) as u32,
            kv_capacity: capacity as u32,
            q_slot_words: q_slot_words as u32,
            out_slot_words: out_slot_words as u32,
            scale: (cfg.head_dim() as f32).powf(-0.5),
        },
    );
    let mut binds: Vec<(u32, &wgpu::Buffer)> =
        vec![(30, q), (31, kc), (32, vc), (34, &p), (35, out)];
    if let Some(br) = brange {
        binds.push((36, br));
    } else {
        binds.push((33, &step.step));
    }
    b.make_pass(label, &s.rope, entry, &binds, (n_q as u32, slots as u32, 1))
}

#[allow(clippy::too_many_arguments)]
fn push_attn_split(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    q: &wgpu::Buffer,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    step: &StepBuffers,
    part: &wgpu::Buffer,
    out: &wgpu::Buffer,
    cfg: &DotsDecoderConfig,
    capacity: usize,
    n_splits: usize,
    fold: usize,
    f16_out: bool,
) -> Result<()> {
    let n_q = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let p = b.uni(
        &format!("{label}-p"),
        SplitAttnParams {
            n_q_heads: n_q as u32,
            n_kv_heads: cfg.num_key_value_heads as u32,
            head_dim: hd as u32,
            gqa_group: (n_q / cfg.num_key_value_heads) as u32,
            kv_capacity: capacity as u32,
            n_splits: n_splits as u32,
            part_stride: (hd + 2) as u32,
            scale: (hd as f32).powf(-0.5),
        },
    );
    b.push(
        &format!("{label}-slice"),
        &s.rope,
        if fold > 1 {
            "dg_attn_fold"
        } else {
            "dg_attn_split"
        },
        &[
            (60, q),
            (61, kc),
            (62, vc),
            (63, &step.step),
            (64, &p),
            (65, part),
        ],
        ((n_q / fold) as u32, n_splits as u32, 1),
    )?;
    b.push(
        &format!("{label}-fold"),
        if f16_out { &s.rope_act16 } else { &s.rope },
        "dg_attn_combine",
        &[(63, &step.step), (64, &p), (65, part), (66, out)],
        (n_q as u32, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_attn_tiled(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    entry: &str,
    q: &wgpu::Buffer,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    step: &StepBuffers,
    out: &wgpu::Buffer,
    cfg: &DotsDecoderConfig,
    capacity: usize,
    chunk: usize,
    q_slot_words: usize,
    out_slot_words: usize,
) -> Result<()> {
    let n_q = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim();
    anyhow::ensure!(
        hd <= ATTN_TILE_MAX_HEAD_DIM,
        "{label}: the tiled attention stages {ATTN_TILE_MAX_HEAD_DIM} dims per row to fit the \
         workgroup budget, head_dim is {hd}"
    );
    anyhow::ensure!(
        chunk.is_multiple_of(ATTN_TILE_QUERY_ROWS),
        "{label}: a {chunk}-row chunk is not a whole number of {ATTN_TILE_QUERY_ROWS}-row query \
         tiles"
    );
    let p = b.uni(
        &format!("{label}-p"),
        TileAttnParams {
            n_q_heads: n_q as u32,
            n_kv_heads: n_kv as u32,
            head_dim: hd as u32,
            gqa_group: (n_q / n_kv) as u32,
            kv_capacity: capacity as u32,
            chunk: chunk as u32,
            q_slot_words: q_slot_words as u32,
            out_slot_words: out_slot_words as u32,
            kv_words: (n_kv * hd / 2) as u32,
            scale: (hd as f32).powf(-0.5),
            pad0: 0,
            pad1: 0,
        },
    );
    b.push(
        label,
        &s.rope,
        entry,
        &if entry == "dg_attn_reg" {
            [
                (80, q),
                (81, kc),
                (82, vc),
                (83, &step.step),
                (84, &p),
                (85, out),
            ]
        } else {
            [
                (70, q),
                (71, kc),
                (72, vc),
                (73, &step.step),
                (74, &p),
                (75, out),
            ]
        },
        (n_q as u32, (chunk / ATTN_TILE_QUERY_ROWS) as u32, 1),
    )
}

struct FusedSlabs {
    qa: wgpu::Buffer,
    qb: wgpu::Buffer,
    sa: wgpu::Buffer,
    sb: wgpu::Buffer,
    la: wgpu::Buffer,
    lb: wgpu::Buffer,
    oa: wgpu::Buffer,
    ob: wgpu::Buffer,
}

struct FusedMirrors {
    kf0: wgpu::Buffer,
    kf1: wgpu::Buffer,
    vt0: wgpu::Buffer,
    vt1: wgpu::Buffer,
}

pub const ATTN_FUSED_QK_PLAN: (u32, u32, u32, u32) = (4, 4, 4, 2);

pub const ATTN_FUSED_PV_PLAN_TM2_TN1_SG2_IS_THE_M768_N128_LADDER_KNEE: (u32, u32, u32, u32) =
    (2, 1, 2, 2);

fn attn_coop_source(tm: u32, tn: u32, sg: u32, ku: u32) -> String {
    let plan = coop_plan();
    let src = plan.source(tm, tn, sg, ku);
    let cols = plan.cols_per_workgroup(tn, sg);
    let step = 8 * ku;
    let with_step = src.replace(
        "@group(0) @binding(4) var<storage, read> cg_zero: array<f32>;",
        "@group(0) @binding(4) var<storage, read> cg_zero: array<f32>;\n@group(0) @binding(5) \
         var<storage, read> cgd_step: array<u32>;",
    );
    let with_dyn = with_step.replace(
        "    let kk = cg_p.k_elems;\n",
        &format!(
            "    let kk = cg_p.k_elems;\n    var cg_kdyn = kk;\n    if (cg_p.pad0 != 0u) {{\n        \
             cg_kdyn = min(kk, ((cgd_step[1u] + cg_p.pad0 + 15u) / 16u) * 16u);\n    }}\n    \
             if (cg_p.pad1 != 0u && bn * {cols}u >= ((cgd_step[1u] + cg_p.pad1 + 7u) / 8u) * 8u) \
             {{\n        cg_kdyn = 0u;\n    }}\n"
        ),
    );
    let looped = with_dyn.replace(
        &format!("for (var kt = 0u; kt < kk; kt = kt + {step}u) {{"),
        &format!("for (var kt = 0u; kt < cg_kdyn; kt = kt + {step}u) {{"),
    );
    assert!(
        with_step != src && with_dyn != with_step && looped != with_dyn,
        "the coop GEMM source anchors this runtime-shape patch relies on have moved"
    );
    looped
}

fn attn_cap16(capacity: usize) -> usize {
    capacity.div_ceil(16) * 16
}

#[allow(clippy::too_many_arguments)]
fn push_attn_fused(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    q: &wgpu::Buffer,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    step: &StepBuffers,
    out: &wgpu::Buffer,
    zero: &wgpu::Buffer,
    fb: &FusedSlabs,
    mir: &FusedMirrors,
    cfg: &DotsDecoderConfig,
    capacity: usize,
    chunk: usize,
    q_slot_words: usize,
    out_slot_words: usize,
) -> Result<()> {
    let n_q = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim();
    let gqa = n_q / n_kv;
    anyhow::ensure!(
        hd == ATTN_FUSED_HEAD_DIM && n_kv == 2,
        "{label}: the GEMM attention arm bakes head_dim {ATTN_FUSED_HEAD_DIM} and two kv heads, \
         got {hd}/{n_kv}"
    );
    anyhow::ensure!(
        chunk.is_multiple_of(ATTN_FUSED_QUERY_ROWS),
        "{label}: a {chunk}-row chunk is not a whole number of {ATTN_FUSED_QUERY_ROWS}-row rows"
    );
    let cap16 = attn_cap16(capacity);
    let rows = gqa * chunk;
    let words = hd / 2;
    let p = b.uni(
        &format!("{label}-p"),
        TileAttnParams {
            n_q_heads: n_q as u32,
            n_kv_heads: n_kv as u32,
            head_dim: hd as u32,
            gqa_group: gqa as u32,
            kv_capacity: capacity as u32,
            chunk: chunk as u32,
            q_slot_words: q_slot_words as u32,
            out_slot_words: out_slot_words as u32,
            kv_words: (n_kv * hd / 2) as u32,
            scale: (hd as f32).powf(-0.5),
            pad0: 0,
            pad1: 0,
        },
    );
    let lanes = (chunk * n_q * words) as u32;
    b.push(
        &format!("{label}-qpack"),
        &s.fused,
        "dg_qpack",
        &[(90, q), (94, &p), (80, &fb.qa), (81, &fb.qb)],
        (lanes.div_ceil(64), 1, 1),
    )?;
    let mlanes = (chunk * n_kv * words) as u32;
    b.push(
        &format!("{label}-mirror"),
        &s.fused,
        "dg_kv_mirror",
        &[
            (91, kc),
            (92, vc),
            (93, &step.step),
            (94, &p),
            (82, &mir.kf0),
            (83, &mir.kf1),
            (84, &mir.vt0),
            (85, &mir.vt1),
        ],
        (mlanes.div_ceil(64), 1, 1),
    )?;
    let plan = coop_plan();
    let (qtm, qtn, qsg, qku) = ATTN_FUSED_QK_PLAN;
    plan.check_shape(rows as u32, cap16 as u32, hd as u32)
        .map_err(|e| anyhow::anyhow!("{label}: qk shape: {e}"))?;
    let qk_src = attn_coop_source(qtm, qtn, qsg, qku);
    let qk_entry = plan.entry(qtm, qtn, qsg, qku);
    let (qbm, qbn) = plan.grid(rows as u32, cap16 as u32, qtm, qtn, qsg);
    let qk_grid = b.grid1((qbm * qbn) as u64, 1);
    let qk_p = b.uni(
        &format!("{label}-qkp"),
        CoopGemmParams {
            n_rows: cap16 as u32,
            k_elems: hd as u32,
            m_rows: rows as u32,
            blocks_n: qbn,
            y_stride: cap16 as u32,
            groups_x: qk_grid.0,
            pad0: 0,
            pad1: chunk as u32,
        },
    );
    for (g, (kf, qx, sy)) in [(&mir.kf0, &fb.qa, &fb.sa), (&mir.kf1, &fb.qb, &fb.sb)]
        .iter()
        .enumerate()
    {
        b.push(
            &format!("{label}-qk{g}"),
            &qk_src,
            &qk_entry,
            &[
                (0, *kf),
                (1, *qx),
                (2, *sy),
                (3, &qk_p),
                (4, zero),
                (5, &step.step),
            ],
            qk_grid,
        )?;
    }
    for (g, (sx, ly)) in [(&fb.sa, &fb.la), (&fb.sb, &fb.lb)].iter().enumerate() {
        b.push(
            &format!("{label}-sm{g}"),
            &s.fused,
            "dg_asoftmax",
            &[(86, *sx), (88, *ly), (93, &step.step), (94, &p)],
            (rows as u32, 1, 1),
        )?;
    }
    let (ptm, ptn, psg, pku) = ATTN_FUSED_PV_PLAN_TM2_TN1_SG2_IS_THE_M768_N128_LADDER_KNEE;
    plan.check_shape(rows as u32, hd as u32, cap16 as u32)
        .map_err(|e| anyhow::anyhow!("{label}: pv shape: {e}"))?;
    let pv_src = attn_coop_source(ptm, ptn, psg, pku);
    let pv_entry = plan.entry(ptm, ptn, psg, pku);
    let (pbm, pbn) = plan.grid(rows as u32, hd as u32, ptm, ptn, psg);
    let pv_grid = b.grid1((pbm * pbn) as u64, 1);
    let pv_p = b.uni(
        &format!("{label}-pvp"),
        CoopGemmParams {
            n_rows: hd as u32,
            k_elems: cap16 as u32,
            m_rows: rows as u32,
            blocks_n: pbn,
            y_stride: hd as u32,
            groups_x: pv_grid.0,
            pad0: chunk as u32,
            pad1: 0,
        },
    );
    for (g, (vt, sx, oy)) in [(&mir.vt0, &fb.sa, &fb.oa), (&mir.vt1, &fb.sb, &fb.ob)]
        .iter()
        .enumerate()
    {
        b.push(
            &format!("{label}-pv{g}"),
            &pv_src,
            &pv_entry,
            &[
                (0, *vt),
                (1, *sx),
                (2, *oy),
                (3, &pv_p),
                (4, zero),
                (5, &step.step),
            ],
            pv_grid,
        )?;
    }
    b.push(
        &format!("{label}-opack"),
        &s.fused,
        "dg_opack",
        &[
            (95, out),
            (94, &p),
            (96, &fb.oa),
            (98, &fb.ob),
            (87, &fb.la),
            (89, &fb.lb),
        ],
        (lanes.div_ceil(64), 1, 1),
    )
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct DgScvtParams {
    m: u32,
    qkv_words: u32,
    q_words: u32,
    kv_words: u32,
    capacity: u32,
    heads_words: u32,
    row_off: u32,
    pad1: u32,
}

struct SteelBufs {
    qh: wgpu::Buffer,
    oh: wgpu::Buffer,
    sp: Vec<wgpu::Buffer>,
    pu: Vec<wgpu::Buffer>,
    mp: wgpu::Buffer,
    dummy: wgpu::Buffer,
    pls: std::sync::Arc<super::prefill_attn_msl::CausalSteelPipelines>,
}

fn steel_params_buffer(
    b: &mut Builder,
    label: &str,
    p: &super::vision_attn_msl::SteelAttnParams,
) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    let buf = b
        .ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::bytes_of(p),
            usage: wgpu::BufferUsages::UNIFORM
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST,
        });
    b.core.store(label, buf)
}

#[allow(clippy::too_many_arguments)]
fn push_attn_steel(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    q: &wgpu::Buffer,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    step: &StepBuffers,
    out: &wgpu::Buffer,
    sb: &SteelBufs,
    pfi: usize,
    kh: &wgpu::Buffer,
    vh: &wgpu::Buffer,
    cfg: &DotsDecoderConfig,
    chunk: usize,
) -> Result<()> {
    push_attn_steel_kv(
        b,
        s,
        label,
        q,
        kc,
        vc,
        step,
        out,
        sb,
        pfi,
        kh,
        vh,
        cfg,
        chunk,
        "dg_scvt_kv",
    )
}

#[allow(clippy::too_many_arguments)]
fn push_attn_steel_kv(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    q: &wgpu::Buffer,
    kc: &wgpu::Buffer,
    vc: &wgpu::Buffer,
    step: &StepBuffers,
    out: &wgpu::Buffer,
    sb: &SteelBufs,
    pfi: usize,
    kh: &wgpu::Buffer,
    vh: &wgpu::Buffer,
    cfg: &DotsDecoderConfig,
    chunk: usize,
    kv_entry: &str,
) -> Result<()> {
    let n_q = cfg.num_attention_heads;
    let hd = cfg.head_dim();
    let q_words = n_q * hd / 2;
    let kv_words = cfg.num_key_value_heads * hd / 2;
    let bq = super::vision_attn_msl::STEEL_BQ as usize;
    anyhow::ensure!(
        hd == ATTN_FUSED_HEAD_DIM && chunk.is_multiple_of(bq),
        "{label}: the causal steel arm bakes head_dim {ATTN_FUSED_HEAD_DIM} and {bq}-row query \
         blocks, got {hd}/{chunk}"
    );
    let p = &sb.pu[pfi];
    b.push(
        &format!("{label}-qcvt"),
        &s.scvt,
        "dg_scvt_q",
        &[(1, q), (4, p), (5, &sb.qh)],
        ((q_words as u32).div_ceil(64), chunk as u32, 1),
    )?;
    b.push(
        &format!("{label}-kvcvt"),
        &s.scvt,
        kv_entry,
        &[(0, &step.step), (2, kc), (3, vc), (4, p), (6, kh), (7, vh)],
        (((chunk * kv_words) as u32).div_ceil(64), 1, 1),
    )?;
    b.push(
        &format!("{label}-pp"),
        &s.scvt,
        "dg_steel_pp",
        &[(0, &step.step), (4, p), (10, &sb.sp[pfi])],
        (1, 1, 1),
    )?;
    let bind = b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &sb.pls.layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: sb.qh.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: kh.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: vh.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: sb.oh.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: sb.sp[pfi].as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: sb.mp.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: sb.dummy.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: sb.dummy.as_entire_binding(),
            },
        ],
    });
    b.push_prebuilt(
        super::prefill_attn_msl::CAUSAL_ENTRY,
        sb.pls.pipeline.clone(),
        bind,
        ((chunk / bq) as u32, n_q as u32, 1),
    );
    b.push(
        &format!("{label}-ocvt"),
        &s.scvt,
        "dg_scvt_o",
        &[(4, p), (8, &sb.oh), (9, out)],
        (((chunk * q_words) as u32).div_ceil(64), 1, 1),
    )
}

struct SteelProjBufs {
    xh: wgpu::Buffer,
    skpart: Option<wgpu::Buffer>,
    pls: std::sync::Arc<super::vision_attn_msl::SteelGemmPipelines>,
    cvt: Option<std::sync::Arc<super::vision_attn_msl::SteelGemmCvtPipelines>>,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct DgSkRedParams {
    n_elems: u32,
    slices: u32,
    pad0: u32,
    pad1: u32,
}

const DG_SK_RED_WGSL: &str = "
struct P { n_elems: u32, slices: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn dg_sk_red(@builtin(global_invocation_id) g: vec3<u32>) {
  if (g.x < p.n_elems) {
    var acc = 0.0;
    for (var s = 0u; s < p.slices; s++) { acc += part[s * p.n_elems + g.x]; }
    y[g.x] = acc;
  }
}
";

pub const TRUNK_BF16OUT_FOLDS_THE_SPLITK_REDUCE_STORE_BECAUSE_THE_F32_PARTIAL_SUM_NEVER_MOVES:
    &str = "The splitk families kept f32 partials plus a standalone dg_cvt after the reduce; \
    dg_sk_red_bf16 keeps the identical ascending-slice f32 sum and only rounds the final store \
    through dg_cvt's exact bf16_encode chain (RNE incl the NaN map), so the fold is bitwise -- \
    rounding partials to bf16 BEFORE the reduce would change the sum and is the tolerance form \
    this fold refuses. It rides NV_DOTS_TRUNK_BF16OUT (the output-fold env): =0 restores \
    dg_sk_red plus the standalone dg_cvt byte-identically. qkv keeps the f32 reduce (rope \
    consumes the f32 sink directly); gateup never splits.";

const DG_SK_RED_BF16_WGSL: &str = "
struct P { n_elems: u32, slices: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> part: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<u32>;
@group(0) @binding(2) var<uniform> p: P;
fn skr_bf16e(x: f32) -> u32 {
  let b = bitcast<u32>(x);
  let r = 0x7fffu + ((b >> 16u) & 1u);
  return select((b + r) >> 16u, 0x7fc0u, x != x);
}
@compute @workgroup_size(256)
fn dg_sk_red_bf16(@builtin(global_invocation_id) g: vec3<u32>) {
  if (2u * g.x < p.n_elems) {
    var lo = 0.0;
    var hi = 0.0;
    for (var s = 0u; s < p.slices; s++) {
      lo += part[s * p.n_elems + 2u * g.x];
      hi += part[s * p.n_elems + 2u * g.x + 1u];
    }
    y[g.x] = skr_bf16e(lo) | (skr_bf16e(hi) << 16u);
  }
}
";

fn upload_f16_of_bf16(b: &mut Builder, label: &str, host: &[u16]) -> wgpu::Buffer {
    let h16 = |bits: u16| half::f16::from_f32(half::bf16::from_bits(bits).to_f32()).to_bits();
    let mut words = Vec::with_capacity(host.len().div_ceil(2));
    for pair in host.chunks(2) {
        let lo = h16(pair[0]) as u32;
        let hi = pair.get(1).map(|&x| h16(x)).unwrap_or(0) as u32;
        words.push(lo | (hi << 16));
    }
    b.upload_u32(label, &words)
}

#[allow(clippy::too_many_arguments)]
fn push_proj_steel(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    fam: ProjFamily,
    w_h: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sink16: Option<&wgpu::Buffer>,
    sb: &SteelProjBufs,
    m: usize,
    n: usize,
    k: usize,
) -> Result<bool> {
    let cfg = pf_steel_proj_cfg_for(m);
    anyhow::ensure!(
        k.is_multiple_of(2 * cfg.2 as usize),
        "{label}: k {k} is not a whole even number of {}-deep steel k-blocks",
        cfg.2
    );
    anyhow::ensure!(
        n.is_multiple_of(2),
        "{label}: n {n} must be even so bf16 word stores never split a pair"
    );
    let kw = k / 2;
    let abf16 = trunk_abf16_enabled() && sb.cvt.is_some();
    if !abf16 {
        let p = b.uni(
            &format!("{label}-cp"),
            DgScvtParams {
                m: m as u32,
                qkv_words: kw as u32,
                q_words: kw as u32,
                kv_words: 0,
                capacity: 0,
                heads_words: 0,
                row_off: 0,
                pad1: 0,
            },
        );
        b.push(
            &format!("{label}-xcvt"),
            &s.scvt,
            "dg_scvt_q",
            &[(1, x), (4, &p), (5, &sb.xh)],
            ((kw as u32).div_ceil(64), m as u32, 1),
        )?;
    }
    let a_buf = if abf16 { x } else { &sb.xh };
    let sk = trunk_splitk_s(fam, m, k, cfg.2);
    let sk_pipeline = if sk > 1 {
        if abf16 {
            anyhow::ensure!(
                cfg == super::vision_attn_msl::STEEL_GEMM_CVT_SK_CFG_IS_THE_ONLY_TILE_SPLITK_ENGAGES,
                "{label}: splitk engaged an uninstantiated abf16 tile {cfg:?}"
            );
            sb.cvt.as_ref().and_then(|c| c.pick("sk_abf16", cfg))
        } else {
            sb.pls.pick_splitk(cfg)
        }
    } else {
        None
    };
    if let (Some(pipeline), Some(part)) = (sk_pipeline, sb.skpart.as_ref()) {
        let gp = b.uni(
            &format!("{label}-skgp"),
            super::vision_attn_msl::steel_gemm_splitk_params(m, n, k, sk, cfg),
        );
        let layout = if abf16 {
            &sb.cvt.as_ref().unwrap().layout
        } else {
            &sb.pls.layout
        };
        let bind = b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(label),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: w_h.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: part.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: gp.as_entire_binding(),
                },
            ],
        });
        let entry = if abf16 {
            super::vision_attn_msl::steel_gemm_cvt_entry("sk_abf16", cfg)
        } else {
            super::vision_attn_msl::steel_gemm_splitk_entry(cfg.0, cfg.1, cfg.2, cfg.3, cfg.4)
        };
        b.push_prebuilt(
            &entry,
            pipeline,
            bind,
            super::vision_attn_msl::steel_gemm_splitk_grid(m, n, sk, cfg),
        );
        let rp = b.uni(
            &format!("{label}-skrp"),
            DgSkRedParams {
                n_elems: (m * n) as u32,
                slices: sk as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        let red16 = sink16.filter(|_| trunk_bf16out_enabled());
        match red16 {
            Some(sink) => {
                let _ = TRUNK_BF16OUT_FOLDS_THE_SPLITK_REDUCE_STORE_BECAUSE_THE_F32_PARTIAL_SUM_NEVER_MOVES;
                b.push(
                    &format!("{label}-skred16"),
                    DG_SK_RED_BF16_WGSL,
                    "dg_sk_red_bf16",
                    &[(0, part), (1, sink), (2, &rp)],
                    (((m * n / 2) as u32).div_ceil(256), 1, 1),
                )?;
                b.bf16out_families += 1;
            }
            None => {
                b.push(
                    &format!("{label}-skred"),
                    DG_SK_RED_WGSL,
                    "dg_sk_red",
                    &[(0, part), (1, y), (2, &rp)],
                    (((m * n) as u32).div_ceil(256), 1, 1),
                )?;
            }
        }
        b.sk_families += 1;
        if abf16 {
            b.abf16_families += 1;
        }
        return Ok(red16.is_some());
    }
    let engage_out = sink16.is_some() && trunk_bf16out_enabled() && sb.cvt.is_some();
    let gp = b.uni(
        &format!("{label}-gp"),
        super::vision_attn_msl::steel_gemm_params(m, n, k, n, cfg),
    );
    let kind = match (abf16, engage_out) {
        (false, false) => None,
        (true, false) => Some("abf16"),
        (false, true) => Some("bf16o"),
        (true, true) => Some("abf16o"),
    };
    let d_buf = if engage_out { sink16.unwrap() } else { y };
    let (entry, pipeline, layout) = match kind {
        None => (
            super::vision_attn_msl::steel_gemm_entry(cfg.0, cfg.1, cfg.2, cfg.3, cfg.4),
            sb.pls
                .pick(cfg)
                .ok_or_else(|| anyhow::anyhow!("{label}: steel gemm config {cfg:?} missing"))?,
            &sb.pls.layout,
        ),
        Some(kind) => {
            let cvt = sb.cvt.as_ref().unwrap();
            (
                super::vision_attn_msl::steel_gemm_cvt_entry(kind, cfg),
                cvt.pick(kind, cfg).ok_or_else(|| {
                    anyhow::anyhow!("{label}: steel cvt kernel {kind} at {cfg:?} missing")
                })?,
                &cvt.layout,
            )
        }
    };
    let bind = b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: a_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: w_h.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: d_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: gp.as_entire_binding(),
            },
        ],
    });
    b.push_prebuilt(
        &entry,
        pipeline,
        bind,
        super::vision_attn_msl::steel_gemm_grid(m, n, cfg),
    );
    if abf16 {
        b.abf16_families += 1;
    }
    if engage_out {
        b.bf16out_families += 1;
    }
    Ok(engage_out)
}

#[allow(clippy::too_many_arguments)]
fn push_cvt(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    src: &wgpu::Buffer,
    dst: &wgpu::Buffer,
    n_elems: usize,
    src_slot: usize,
    dst_slot: usize,
    slots: usize,
    to_f32: bool,
) -> Result<()> {
    let p = b.uni(
        &format!("{label}-p"),
        CvtParams {
            n_elems: n_elems as u32,
            src_slot: src_slot as u32,
            dst_slot: dst_slot as u32,
            to_f32: u32::from(to_f32),
        },
    );
    let lanes = if to_f32 { n_elems } else { n_elems / 2 };
    b.push(
        label,
        &s.rope,
        "dg_cvt",
        &[(100, src), (101, &p), (102, dst)],
        ((lanes as u32).div_ceil(64).max(1), slots as u32, 1),
    )
}

pub const COOP_F16_OPERANDS_MEASURED_A_LOSS_AT_DECODER_SHAPES_SO_F32_AND_ITS_BITWISE_GATE_STAY:
    &str = "The vision tower's f16-operand coop GEMM reads 15 to 18 TFLOP/s at m=13600, so an \
     f16 arm for the four projection GEMMs was built and measured: bf16-to-f16 weight copies, \
     pack2x16float activation staging, same plans. The 3412-token prefill wall read 1741 ms \
     against the f32 arm's 1702, minus 2.3 percent, both reps agreeing -- at m=128 the operand \
     bytes are not what the projection GEMMs wait on, and the tower's f16 advantage does not \
     transfer down-shape. The arm is deleted rather than kept as a knob; f32 operands also \
     keep the chunk-vs-steps suite bitwise. The steel read this constant once pointed at is \
     shipped: NV_DOTS_PF_STEEL_PROJ routes the four prefill projections through the vendored \
     steel gemm by default, 1.59x over this coop plan on the projections at the 3412-row page \
     chunk, and the NV_DOTS_PF_STEEL_CFG ladder holds the vision crown tile as the decoder-shape \
     crown too -- bigger k-blocks lose here, so the coop plan below is the fallback arm, not the \
     frontier.";

pub fn coop_prefill_supported(caps: &nv_kernels::wgpu_backend::qualify::Capabilities) -> bool {
    caps.cooperative_matrix
        && caps.shader_f16
        && caps.subgroup_width_known() == Some(32)
        && caps.coop_advertises(&coop_plan().request())
}

fn coop_plan() -> nv_kernels::wgpu_backend::kernels::gemm_coop_f16::CoopGemm {
    let _ = COOP_F16_OPERANDS_MEASURED_A_LOSS_AT_DECODER_SHAPES_SO_F32_AND_ITS_BITWISE_GATE_STAY;
    use nv_kernels::wgpu_backend::kernels::gemm_coop_f16 as coop;
    coop::CoopGemm::new(8, coop::Operand::F32)
}

#[allow(clippy::too_many_arguments)]
fn push_coop_gemm(
    b: &mut Builder,
    label: &str,
    w_f32: &wgpu::Buffer,
    x_f32: &wgpu::Buffer,
    y_f32: &wgpu::Buffer,
    zero: &wgpu::Buffer,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let plan = coop_plan();
    plan.check_shape(m as u32, n as u32, k as u32)
        .map_err(|e| anyhow::anyhow!("{label}: coop shape: {e}"))?;
    let (tm, tn, sg, ku) = COOP_PLAN_TM4_TN4_SG4_KU2_WON_EVERY_SHAPE_OF_THE_WIDTH_LADDER;
    let (blocks_m, blocks_n) = plan.grid(m as u32, n as u32, tm, tn, sg);
    let grid = b.grid1((blocks_m * blocks_n) as u64, 1);
    let p = b.uni(
        &format!("{label}-p"),
        CoopGemmParams {
            n_rows: n as u32,
            k_elems: k as u32,
            m_rows: m as u32,
            blocks_n,
            y_stride: n as u32,
            groups_x: grid.0,
            pad0: 0,
            pad1: 0,
        },
    );
    let src = plan.source(tm, tn, sg, ku);
    let entry = plan.entry(tm, tn, sg, ku);
    b.push(
        label,
        &src,
        &entry,
        &[(0, w_f32), (1, x_f32), (2, y_f32), (3, &p), (4, zero)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemm_m8(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Lin,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    x_slot_words: usize,
    y_slot_words: usize,
    out_f32: bool,
    m: usize,
) -> Result<()> {
    let Lin::Bf16 { w, n, k } = w else {
        anyhow::bail!("{label}: the m-row prefill GEMM is a bf16 kernel");
    };
    let want = gemm_slots();
    let slots = if m.is_multiple_of(want) {
        want
    } else {
        PREFILL_GEMM_SLOTS_PER_DISPATCH_ARE_THE_TWO_VEC4_ACCUMULATORS_OF_DG_GEMM_M8
    };
    anyhow::ensure!(
        m.is_multiple_of(slots),
        "{label}: a {m}-row chunk is not a whole number of {slots}-slot GEMM dispatches"
    );
    let entry = if slots == 16 {
        "dg_gemm_m16"
    } else {
        "dg_gemm_m8"
    };
    let grid = b.grid1(n.div_ceil(2) as u64, 1);
    for base in (0..m).step_by(slots) {
        let p = b.uni(
            &format!("{label}-p{base}"),
            GemmMParams {
                n_rows: *n as u32,
                k_words: (k / 2) as u32,
                groups_x: grid.0,
                w_row_words: (k / 2) as u32,
                x_slot_words: x_slot_words as u32,
                y_slot_words: y_slot_words as u32,
                out_f32: u32::from(out_f32),
                slot_base: base as u32,
            },
        );
        b.push(
            label,
            &s.rope,
            entry,
            &[(40, w), (41, x), (42, &p), (43, y)],
            grid,
        )?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_gather_m(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    entry: &str,
    src: &wgpu::Buffer,
    tok: &wgpu::Buffer,
    mask: &wgpu::Buffer,
    out: &wgpu::Buffer,
    row_off: usize,
    n_rows: usize,
    hidden_words: usize,
    slots: usize,
) -> Result<()> {
    let p = b.uni(
        &format!("{label}-p"),
        GatherMParams {
            row_off: row_off as u32,
            n_rows: n_rows as u32,
            hidden_words: hidden_words as u32,
            pad0: 0,
        },
    );
    let binds: Vec<(u32, &wgpu::Buffer)> = if entry == "dg_splice_m" {
        vec![(50, src), (52, mask), (53, &p), (54, out)]
    } else {
        vec![(50, src), (51, tok), (52, mask), (53, &p), (54, out)]
    };
    b.push(
        label,
        &s.rope,
        entry,
        &binds,
        ((hidden_words as u32).div_ceil(64).max(1), slots as u32, 1),
    )
}

pub struct CoopProbe {
    pub coop_ms: f64,
    pub m8_ms: f64,
    pub max_abs_diff: f32,
    pub coop_tflops: f64,
    pub m8_tflops: f64,
}

#[allow(clippy::too_many_arguments)]
pub fn coop_gemm_probe(
    m: usize,
    n: usize,
    k: usize,
    tm: u32,
    tn: u32,
    sg: u32,
    ku: u32,
    reps: usize,
) -> Result<CoopProbe> {
    use nv_kernels::wgpu_backend::kernels::gemm_coop_f16 as coop;

    let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
    let plan = coop::CoopGemm::new(8, coop::Operand::F32);
    plan.check_shape(m as u32, n as u32, k as u32)
        .map_err(|e| anyhow::anyhow!("coop shape: {e}"))?;
    anyhow::ensure!(
        plan.acc_fits_a_register_file(tm, tn),
        "tm{tm} tn{tn} needs {} accumulator dwords per lane, over the budget",
        plan.acc_lane_dwords(tm, tn)
    );
    anyhow::ensure!(
        ctx.caps.coop_advertises(&plan.request()),
        "adapter does not advertise the tile-8 f32 cooperative matrix"
    );

    let s = Sources::new(1, false, sg_tree_sites(ctx.subgroup_width()).attn);
    let mut b = Builder::new(ctx);
    let wbits: Vec<u16> = (0..n * k)
        .map(|i| half::bf16::from_f32(((i % 97) as f32 - 48.0) / 512.0).to_bits())
        .collect();
    let xbits: Vec<u16> = (0..m * k)
        .map(|i| half::bf16::from_f32(((i % 89) as f32 - 44.0) / 256.0).to_bits())
        .collect();
    let f32_of = |v: &[u16]| -> Vec<f32> {
        v.iter()
            .map(|b| f32::from_bits((*b as u32) << 16))
            .collect()
    };
    let wf = b.upload_f32("dg-coop-w", &f32_of(&wbits));
    let xf = b.upload_f32("dg-coop-x", &f32_of(&xbits));
    let yf = b.zeros("dg-coop-y", (m * n * 4) as u64);
    let zero = b.upload_f32("dg-coop-zero", &vec![0f32; plan.zero_elems()]);

    let (blocks_m, blocks_n) = plan.grid(m as u32, n as u32, tm, tn, sg);
    let grid = b.grid1((blocks_m * blocks_n) as u64, 1);
    let cp = b.uni(
        "dg-coop-p",
        CoopGemmParams {
            n_rows: n as u32,
            k_elems: k as u32,
            m_rows: m as u32,
            blocks_n,
            y_stride: n as u32,
            groups_x: grid.0,
            pad0: 0,
            pad1: 0,
        },
    );
    let src = plan.source(tm, tn, sg, ku);
    let entry = plan.entry(tm, tn, sg, ku);
    b.push(
        "dg-coop",
        &src,
        &entry,
        &[(0, &wf), (1, &xf), (2, &yf), (3, &cp), (4, &zero)],
        grid,
    )?;
    let coop_pass = b.passes.pop().expect("coop pass");

    let wlin = Lin::Bf16 {
        w: b.upload_u32("dg-coop-m8w", &pack_pairs(&wbits)),
        n,
        k,
    };
    let x8 = b.upload_u32("dg-coop-m8x", &pack_pairs(&xbits));
    let y8 = b.zeros("dg-coop-m8y", (m * n * 4) as u64);
    push_gemm_m8(&mut b, &s, "dg-coop-m8", &wlin, &x8, &y8, k / 2, n, true, m)?;
    let m8_passes: Vec<Pass> = b.passes.drain(..).collect();
    b.flush_staging();

    let run = |passes: Vec<&Pass>| -> Result<f64> {
        let list: Vec<&Pass> = (0..reps).flat_map(|_| passes.clone()).collect();
        let t = std::time::Instant::now();
        let cb = dispatch::encode_pass_list_labeled(
            ctx,
            list.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
            &[],
        );
        ctx.queue.submit([cb]);
        let _: Vec<f32> = dispatch::read_back(ctx, &yf, 4)
            .map_err(|e| anyhow::anyhow!("coop probe readback: {e}"))?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / reps as f64)
    };
    let coop_ms = run(vec![&coop_pass])?;
    let m8_ms = run(m8_passes.iter().collect())?;

    let a: Vec<f32> = dispatch::read_back(ctx, &yf, m * n)
        .map_err(|e| anyhow::anyhow!("coop y readback: {e}"))?;
    let c: Vec<f32> =
        dispatch::read_back(ctx, &y8, m * n).map_err(|e| anyhow::anyhow!("m8 y readback: {e}"))?;
    let mut max_abs_diff = 0f32;
    for slot in 0..m {
        for row in 0..n {
            let d = (a[slot * n + row] - c[slot * n + row]).abs();
            if d > max_abs_diff {
                max_abs_diff = d;
            }
        }
    }
    let flop = 2.0 * m as f64 * n as f64 * k as f64;
    Ok(CoopProbe {
        coop_ms,
        m8_ms,
        max_abs_diff,
        coop_tflops: flop / (coop_ms / 1e3) / 1e12,
        m8_tflops: flop / (m8_ms / 1e3) / 1e12,
    })
}

pub fn gemv_w4_probe(n: usize, k: usize, reps: usize) -> Result<(f64, f64, f32)> {
    let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
    let s = Sources::new(1, false, sg_tree_sites(ctx.subgroup_width()).attn);
    let mut b = Builder::new(ctx);
    let wf: Vec<f32> = (0..n * k)
        .map(|i| ((i % 97) as f32 - 48.0) / 512.0)
        .collect();
    let xb: Vec<u16> = (0..k)
        .map(|i| half::bf16::from_f32(((i % 89) as f32 - 44.0) / 256.0).to_bits())
        .collect();
    let xf: Vec<f32> = xb
        .iter()
        .map(|v| f32::from_bits((*v as u32) << 16))
        .collect();
    let gs = W4_GROUP as usize;
    let parts = nv_quant::mlx_affine::quantize_mlx_affine_linear(&wf, n, k, gs);
    let aff = upload_affine_w4(&mut b, "dg-w4probe", &parts, n, k);
    let x = b.upload_u32("dg-w4probe-x", &pack_pairs(&xb));
    let y = b.zeros("dg-w4probe-y", wbytes(n));
    push_gemv_w4(&mut b, "dg-w4probe-gemv", &aff, &x, &y, false, false)?;
    let w4_pass = b.passes.pop().expect("w4 pass");

    let dq =
        nv_quant::mlx_affine::dequantize_mlx_affine_linear(&parts.0, &parts.1, &parts.2, n, k, gs);
    let dqb: Vec<u16> = dq
        .iter()
        .map(|v| half::bf16::from_f32(*v).to_bits())
        .collect();
    let wlin = upload_lin(&mut b, "dg-w4probe-bw", &dqb, n, k, DotsGraphQuant::Bf16)?;
    let xa = act_over(&mut b, "dg-w4probe-xa", x.clone(), k, DotsGraphQuant::Bf16);
    let yb = b.zeros("dg-w4probe-yb", wbytes(n));
    push_gemv(&mut b, &s, "dg-w4probe-bf16", &wlin, &xa, &yb, false, 0)?;
    let bf_pass = b.passes.pop().expect("bf16 pass");
    b.flush_staging();

    let run = |pass: &Pass| -> Result<f64> {
        let list: Vec<&Pass> = (0..reps).map(|_| pass).collect();
        let t = std::time::Instant::now();
        let cb = dispatch::encode_pass_list_labeled(
            ctx,
            list.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
            &[],
        );
        ctx.queue.submit([cb]);
        let _: Vec<f32> = dispatch::read_back(ctx, &y, 1)
            .map_err(|e| anyhow::anyhow!("w4 probe readback: {e}"))?;
        Ok(t.elapsed().as_secs_f64() * 1e3 / reps as f64)
    };
    let w4_ms = run(&w4_pass)?;
    let bf_ms = run(&bf_pass)?;

    let got_words: Vec<u32> =
        dispatch::read_back(ctx, &y, n / 2).map_err(|e| anyhow::anyhow!("w4 y readback: {e}"))?;
    let groups = k / gs;
    let mut max_rel = 0f32;
    for row in 0..n {
        let mut want = 0f64;
        for g in 0..groups {
            let mut qdot = 0f64;
            let mut xsum = 0f64;
            for e in 0..gs {
                let ki = g * gs + e;
                let code = (parts.0[row * (k / 8) + ki / 8] >> (4 * (ki % 8))) & 0xF;
                qdot += code as f64 * xf[ki] as f64;
                xsum += xf[ki] as f64;
            }
            want +=
                parts.1[row * groups + g] as f64 * qdot + parts.2[row * groups + g] as f64 * xsum;
        }
        let word = got_words[row / 2];
        let bits = if row % 2 == 0 {
            word & 0xFFFF
        } else {
            word >> 16
        };
        let got = f32::from_bits(bits << 16);
        let want_b = half::bf16::from_f32(want as f32).to_f32();
        let ulp = (want_b.abs().max(1e-3)) / 256.0;
        max_rel = max_rel.max((got - want_b).abs() / ulp);
    }
    Ok((w4_ms, bf_ms, max_rel))
}

pub fn nvfp4_gemv_once(w_bf16: &[u16], n: usize, k: usize, x_bf16: &[u16]) -> Result<Vec<f32>> {
    anyhow::ensure!(
        x_bf16.len() == k,
        "nvfp4 gemv probe: activation has {} values, want {k}",
        x_bf16.len()
    );
    let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
    let s = Sources::new(1, false, sg_tree_sites(ctx.subgroup_width()).attn);
    let mut b = Builder::new(ctx);
    let sl = SingleSlot {
        sel0: b.upload_u32("dg-probe-sel", &[0u32]),
        global1: b.upload_f32(
            "dg-probe-global",
            &[NVFP4_ACTIVATION_GLOBAL_2POW7_LANDS_DECODE_BLOCK_SCALES_INSIDE_THE_E4M3_WINDOW],
        ),
    };
    let lin = upload_lin(&mut b, "dg-probe-w", w_bf16, n, k, DotsGraphQuant::Nvfp4)?;
    let xb = b.upload_u32("dg-probe-x", &pack_pairs(x_bf16));
    let act = act_over(&mut b, "dg-probe-act", xb, k, DotsGraphQuant::Nvfp4);
    let y = b.zeros("dg-probe-y", wbytes(n));
    push_quant(&mut b, &s, "dg-probe-quant", &act, &sl)?;
    push_gemv(&mut b, &s, "dg-probe-gemv", &lin, &act, &y, false, 0)?;
    b.flush_staging();
    let cb = dispatch::encode_pass_list_labeled(
        ctx,
        b.passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
        &[],
    );
    ctx.queue.submit([cb]);
    let words: Vec<u32> = dispatch::read_back(ctx, &y, n.div_ceil(2))
        .map_err(|e| anyhow::anyhow!("nvfp4 gemv probe readback: {e}"))?;
    Ok((0..n)
        .map(|i| {
            let w = words[i / 2];
            let bits = if i.is_multiple_of(2) {
                w & 0xffff
            } else {
                w >> 16
            };
            f32::from_bits(bits << 16)
        })
        .collect())
}

pub const REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES: usize = 32;

pub const BATCHED_REGION_DECODE_SERVES_B_CROPS_PER_TRUNK_PASS: &str =
    "the m=32 chunk trunk prices at 1.15 ms/row against 4.26 ms for a serial step, so B \
     concurrent region generations ride the recorded m=32 prefill trunk with three swapped \
     kernels: rope takes each row's LOCAL position from bpos (region computation stays a \
     bitwise fresh session; only KV slots carry the stride base, and RoPE phase differences \
     cancel), kv-write takes each row's absolute cache slot from bslot, and attention reads \
     each row's own [base, base+len) range from brange. Region prefill reuses the same pf \
     chunk lists with the ranged kv/attn variants swapped in at recorded sites, exactly the \
     split_passes precedent. The head replays the existing decode head per row behind a \
     row-gather into final_x and a token stash into btoks, so every lm-head arm serves \
     unchanged. Ngram bans need per-row logits the batch does not persist; a banned row is \
     finished and truncated by the host instead, which plain-OCR crops already do at loop \
     onset.";

pub const RPF_WIDE_ENV: &str = "NV_DOTS_RPF_WIDE";

pub const RPF_CHUNK_COST_IS_NEAR_FLAT_IN_WIDTH_SO_FEWER_WIDER_CHUNKS_WIN: &str =
    "one trunk chunk prices at 40.5 ms for m=32, 35.5 for m=128 and 149 for m=1024 on the \
     serving span (weight-read-bound, not row-bound), so a region prompt wants the fewest \
     chunks that cover it: a single padded chunk when a width at least half-filled covers \
     the whole prompt (the m/2 guard sits at the measured ~4.2-chunk break-even against \
     128-chains), and 128-class padded tails otherwise -- the old plan let 32-wide chunks \
     eat every sub-128 remainder at 40.5 ms each, pricing a 250-token crop at 197 ms where \
     two 128 chunks cost 71. Regrouping rows into wider chunks is addressing-only: each \
     row's gemm k-loop, rmsnorm and causal steel key-block walk are chunk-width-independent, \
     so both arms produce byte-identical region kv rows and token streams, and the padded \
     rows write only inside the region's own stride (the stride's last row is the dead-row \
     park slot, rewritten by every parked bstep before any read). NV_DOTS_RPF_WIDE=0 \
     restores the narrow plan.";

fn rpf_wide_enabled() -> bool {
    let _ = RPF_CHUNK_COST_IS_NEAR_FLAT_IN_WIDTH_SO_FEWER_WIDER_CHUNKS_WIN;
    !matches!(std::env::var(RPF_WIDE_ENV).ok().as_deref(), Some("0"))
}

pub const RPF_PACK_ENV: &str = "NV_DOTS_RPF_PACK";

pub const RPF_DEFER_ENV: &str = "NV_DOTS_RPF_DEFER";

pub const RPF_PACK_LANE_ROWS_ARE_THE_128_CHUNK_SO_EVERY_LANE_REPLAYS_TODAYS_STEEL_ATTENTION: usize =
    PREFILL_M_WIDE;

pub const RPF_PACK_LANES_8_FILL_THE_1024_TRUNK: usize = 8;

pub const RPF_PACK_MIN_LANES_5_IS_THE_1024_OVER_128_WEIGHT_READ_BREAK_EVEN: usize = 5;

pub const RPF_PACK_ATTN_PASSES_PER_LANE: usize = 5;

pub const RPF_PACK_SHARES_ONE_TRUNK_READ_ACROSS_REGIONS_BECAUSE_ONLY_ATTENTION_HAS_AN_ORIGIN: &str =
    "a region prefill chunk is weight-read-bound (~34 ms at any width), and its trunk work -- \
     gemms, rmsnorms, elementwise -- is row-independent; only rope positions, kv slots and the \
     attention prefix carry a region origin. A pack rides the 1024 trunk once for up to 8 \
     regions laid out in 128-row lanes: rope reads each row's LOCAL position from a per-row \
     buffer (dg_bias_rope_b, same values the per-region uniform produced), kv-write takes each \
     row's absolute cache slot from a per-row buffer (dg_kv_write_b, same slots dg_kv_write_r \
     computed; rows past a lane's region park at capacity and are skipped), and attention runs \
     per-lane: the SAME dg_scvt_kv_r + dg_steel_pp + steel causal dispatch a 128 chunk runs \
     today, fed a per-lane step buffer holding exactly the (pos, kv_base) the unpacked chunk's \
     uniform held, with the lane's rows addressed by row_off in DgScvtParams. Every kernel sees \
     bit-identical inputs at bit-identical uniform values, so packing is addressing-only: \
     region kv rows and token streams match the per-region plan byte for byte. A region's \
     lanes are consecutive and in order (the kh/vh mirror accumulates local rows across a \
     region's chunks and tolerates clobbering only BETWEEN regions), and a region never splits \
     across packs. Packs engage only on the batched (>=24 regions) route, only when the \
     region stride is at least 1024 so lane-local pad rows stay inside their region's stride, \
     and only for groups of at least 2 regions worth at least 5 lanes (the measured 149 ms \
     1024-chunk vs 34 ms 128-chunk break-even at ~4.4). NV_DOTS_RPF_PACK=0 restores the \
     per-region plan. NV_DOTS_RPF_DEFER=K holds mid-loop slot refills until K slots are free \
     so retirements arrive in packable bursts, trading idle lanes for pack grouping -- the \
     gazette ladder read 12.27/12.34 at K=0, 12.09/12.11 at 4, 11.71/11.74 at 8 and 12.19 at \
     12 against 12.99/13.05 unpacked, so K defaults to 8 while packing is on (=0 restores \
     immediate refill).";

pub const RPF_DEFER_DEFAULT_8_MATCHES_THE_PACK_LANES_AND_WON_ITS_GAZETTE_LADDER: usize = 8;

fn rpf_pack_enabled() -> bool {
    let _ = RPF_PACK_SHARES_ONE_TRUNK_READ_ACROSS_REGIONS_BECAUSE_ONLY_ATTENTION_HAS_AN_ORIGIN;
    !matches!(std::env::var(RPF_PACK_ENV).ok().as_deref(), Some("0"))
}

pub fn rpf_defer_rows() -> usize {
    std::env::var(RPF_DEFER_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(if rpf_pack_enabled() {
            RPF_DEFER_DEFAULT_8_MATCHES_THE_PACK_LANES_AND_WON_ITS_GAZETTE_LADDER
        } else {
            0
        })
}

pub const RANGED_STEEL_PREFILL_READS_THE_REGION_INTO_LOCAL_MIRROR_ROWS: &str =
    "the steel causal kernel needs no ranged variant: dg_scvt_kv_r converts the region's \
     cache rows [kv_base + pos, kv_base + pos + m) into LOCAL mirror rows [pos, pos + m), \
     and dg_steel_pp already computes kl and qL_off from the local step position, so the \
     kernel sees an ordinary fresh session of the region's length. The naive dg_attn_r \
     chunk attention this replaces cost the 4K screen's batched arm 10.9 s against 6.5 \
     serial and capped batching at 256-vision-token crops.";

struct SwapSite {
    start: usize,
    len: usize,
    rpf: Option<Vec<Pass>>,
    bat: Option<Vec<Pass>>,
    pack: Option<Vec<Pass>>,
    pack_lane: usize,
}

struct PackMachinery {
    wi: usize,
    bpos: wgpu::Buffer,
    bslot: wgpu::Buffer,
    brange: wgpu::Buffer,
    lane_step: Vec<wgpu::Buffer>,
    validated: bool,
}

struct RegionMachinery {
    wi: usize,
    stride: usize,
    sites: Vec<Vec<SwapSite>>,
    head_range: (usize, usize),
    gathers: Vec<Pass>,
    stashes: Vec<Pass>,
    bpos: wgpu::Buffer,
    bslot: wgpu::Buffer,
    brange: wgpu::Buffer,
    btoks: wgpu::Buffer,
    bhead: Option<BheadMachinery>,
    validated: bool,
    bat_validated: bool,
}

pub const BHEAD_ENV: &str = "NV_DOTS_BHEAD";

pub const BHEAD_X_SLAB_ROW_HALFS: usize = 1536;

pub const BHEAD_READS_THE_W4_HEAD_ONCE_PER_BSTEP_INSTEAD_OF_ONCE_PER_LIVE_ROW: &str =
    "the bstep's per-row head replay is 32 x (gather + w4 lm_head gemv + argmax pair + stash) \
     = 160 dispatches re-reading the 116 MB head from DRAM per live row (~8.3 ms of a 27 ms \
     step; the head fits no SLC, so replay order cannot delete the traffic). The batched head \
     is 4 dispatches: dg_bh_gather converts every live normed row to the act-f16 words the \
     per-row gather produced (same pack2x16float on the same bf16 values), dg_msl_w4_gemv_b \
     walks the batch rows around the shipped dot text so each weight word is read from DRAM \
     once, and dg_bh_am1/dg_bh_am2 replay the shipped argmax comparisons per row into btoks \
     directly. Every row's k-walk, simd_sum and argmax selection is the shipped text or a \
     deterministic selection over identical bits, so per-row logits and tokens are gated \
     bit-identical to the replay. NV_DOTS_BHEAD=0 restores the per-row replay.";

fn bhead_enabled() -> bool {
    let _ = BHEAD_READS_THE_W4_HEAD_ONCE_PER_BSTEP_INSTEAD_OF_ONCE_PER_LIVE_ROW;
    !matches!(std::env::var(BHEAD_ENV).ok().as_deref(), Some("0"))
}

pub const BHEAD_MMA_ENV: &str = "NV_DOTS_BHEAD_MMA";

pub const BHEAD_MMA_CFG_IS_32_32_16_BECAUSE_THE_HEAD_AT_M32_IS_B_READ_BOUND_AND_THE_BRACKET_CROWNED_IT:
    (u32, u32, u32, u32, u32) = (32, 32, 16, 2, 2);

pub const BHEAD_MMA_IS_THE_DEFAULT_TOLERANCE_CLASS_WITH_ITS_CENSUS_AND_ZERO_RESTORES_THE_SCALAR_HEAD_BITWISE:
    &str =
    "The batched-region lm head serves as one steel f16 GEMM (bm32bn32bk16) over a resident \
     f16 twin of the dequantized w4 head (the exact trunk-twin construction: dequant -> bf16 \
     -> f16, +467 MB resident) instead of the scalar dg_msl_w4_gemv_b row loop; \
     NV_DOTS_BHEAD_MMA=0 restores the scalar head byte-identically and frees the twin. The \
     bracket priced the swap at the real shape (m=32 x 1536 x 151936): scalar chain 7.15 ms \
     at 2.1 T-op/s issue vs steel 0.98 ms at 478 GB/s of f16-B read -- the MMA head is \
     B-read-bound, not issue-bound, and the m=32 row tile starves nothing (4748 \
     workgroups). TOLERANCE CLASS, censused, arbitrated default-on: each weight is rounded \
     once to f16 (the gemv factors scale*qdot+bias*xsum in f32) and the MMA k-chain is a \
     different reduction order than the lane-strided simd_sum, so logits move at the ulp \
     scale (max_abs 2.9e-2, rel_rms ~2e-3, 0/41 argmax moves censused) and a near-tie \
     argmax CAN flip -- batched-vs-serial byte identity is an empirical fact under this \
     default, not a construction; gates that mean the scalar head pin =0. The corpus never \
     routes a bstep (trace md5 stamped identical both arms). At m=1 the f16-B read alone \
     (0.86 ms) exceeds the whole scalar head (0.55 ms), so the serial decode head keeps \
     the gemv on every arm. Pre-flip numbers: gazette 8.94-9.03 s, bstep 27.86 ms, \
     isolated 32-row head 8.20 ms.";

fn bhead_mma_enabled() -> bool {
    let _ = BHEAD_MMA_IS_THE_DEFAULT_TOLERANCE_CLASS_WITH_ITS_CENSUS_AND_ZERO_RESTORES_THE_SCALAR_HEAD_BITWISE;
    !matches!(std::env::var(BHEAD_MMA_ENV).ok().as_deref(), Some("0"))
}

struct BheadMachinery {
    gather: Pass,
    gemv: Pass,
    am1: Pass,
    am2: Pass,
    gather_p: wgpu::Buffer,
    gemv_p: wgpu::Buffer,
    am_p: wgpu::Buffer,
    gather_params: BhGatherParams,
    gemv_params: decode_gemv_msl::DgMslW4BParams,
    am_params: BhAmParams,
    blogits: wgpu::Buffer,
    mma: Option<(Pass, wgpu::Buffer, super::vision_attn_msl::SteelGemmParams)>,
}

impl BheadMachinery {
    fn head_pass(&self) -> (&Pass, bool) {
        match self.mma.as_ref().filter(|_| bhead_mma_enabled()) {
            Some((p, _, _)) => (p, true),
            None => (&self.gemv, false),
        }
    }
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct BhGatherParams {
    words: u32,
    src_slot_words: u32,
    b_rows: u32,
    pad0: u32,
}

#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct BhAmParams {
    n: u32,
    groups: u32,
    b_rows: u32,
    pad0: u32,
}

const DG_BHEAD_WGSL: &str = r#"
struct BhGatherParams {
    words: u32,
    src_slot_words: u32,
    b_rows: u32,
    pad0: u32,
};

@group(0) @binding(90) var<storage, read> bhg_src: array<u32>;
@group(0) @binding(91) var<uniform> bhg_p: BhGatherParams;
@group(0) @binding(92) var<storage, read_write> bhg_dst: array<u32>;

@compute @workgroup_size(64)
fn dg_bh_gather(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
    let row = wg.y;
    if (row >= bhg_p.b_rows) {
        return;
    }
    let i = wg.x * 64u + tid.x;
    if (i >= bhg_p.words) {
        return;
    }
    let w = bhg_src[row * bhg_p.src_slot_words + i];
    bhg_dst[row * bhg_p.words + i] = pack2x16float(vec2<f32>(bf16_lo(w), bf16_hi(w)));
}

struct BhAmParams {
    n: u32,
    groups: u32,
    b_rows: u32,
    pad0: u32,
};

@group(0) @binding(40) var<storage, read> bha_logits: array<f32>;
@group(0) @binding(41) var<storage, read_write> bha_pv: array<f32>;
@group(0) @binding(42) var<storage, read_write> bha_pi: array<u32>;
@group(0) @binding(43) var<storage, read_write> bha_out: array<u32>;
@group(0) @binding(44) var<uniform> bha_p: BhAmParams;

var<workgroup> bha_v: array<f32, 256>;
var<workgroup> bha_i: array<u32, 256>;

fn bha_reduce(tid: u32) {
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            let a = bha_v[tid];
            let bv = bha_v[tid + stride];
            if (bv > a || (bv == a && bha_i[tid + stride] < bha_i[tid])) {
                bha_v[tid] = bv;
                bha_i[tid] = bha_i[tid + stride];
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn dg_bh_am1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let row = wid.y;
    if (row >= bha_p.b_rows) {
        return;
    }
    let base = row * bha_p.n;
    let per = (bha_p.n + bha_p.groups - 1u) / bha_p.groups;
    let start = wid.x * per;
    let end = min(start + per, bha_p.n);
    var best = -3.4e38;
    var bi = 0u;
    var i = start + tid;
    loop {
        if (i >= end) { break; }
        let v = bha_logits[base + i];
        if (v > best || (v == best && i < bi)) {
            best = v;
            bi = i;
        }
        i = i + 256u;
    }
    bha_v[tid] = best;
    bha_i[tid] = bi;
    workgroupBarrier();
    bha_reduce(tid);
    if (tid == 0u) {
        bha_pv[row * bha_p.groups + wid.x] = bha_v[0];
        bha_pi[row * bha_p.groups + wid.x] = bha_i[0];
    }
}

@compute @workgroup_size(256)
fn dg_bh_am2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let row = wid.x;
    if (row >= bha_p.b_rows) {
        return;
    }
    var best = -3.4e38;
    var bi = 0u;
    var i = tid;
    loop {
        if (i >= bha_p.groups) { break; }
        let v = bha_pv[row * bha_p.groups + i];
        let idx = bha_pi[row * bha_p.groups + i];
        if (v > best || (v == best && idx < bi)) {
            best = v;
            bi = idx;
        }
        i = i + 256u;
    }
    bha_v[tid] = best;
    bha_i[tid] = bi;
    workgroupBarrier();
    bha_reduce(tid);
    if (tid == 0u) {
        bha_out[row] = bha_i[0];
    }
}
"#;

#[derive(Clone, Copy, Debug)]
pub struct RegionRow {
    pub token: u32,
    pub base: u32,
    pub len: u32,
}

pub const HANDOFF_SPLICE_WORDS_MATCH_BECAUSE_EVERY_HOST_HOP_ONLY_EVER_ROUNDED_ONCE: &str =
    "The shipped handoff rounds the merger's f32 output to bf16 exactly once with round-to-nearest-\
     even (the candle CPU cast; the Metal widen back to f32 is exact and the prefill repack's \
     second half::bf16 round of an exact bf16 value is the identity), so the pf splice words are \
     RNE(y) packed in pairs. dvh_pack_rows applies the same RNE construction to the same f32 bits \
     on the device, so the splice bytes match word for word; splice rows outside the vision span \
     are unreachable because dg_splice_m copies only mask==1 rows and dg_gather_m skips them.";

pub struct VisionRowsGpu {
    pub buf: wgpu::Buffer,
    pub rows: usize,
    pub hidden: usize,
}

fn unpack_bf16_pair_words(w: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(w.len() * 2);
    for word in w {
        out.push(f32::from_bits(word << 16));
        out.push(f32::from_bits(word & 0xffff_0000));
    }
    out
}

pub async fn vision_rows_host_pumped(ctx: &WgpuContext, rows: &VisionRowsGpu) -> Result<Vec<f32>> {
    anyhow::ensure!(
        rows.hidden.is_multiple_of(2),
        "bf16 pair packing needs an even hidden, got {}",
        rows.hidden
    );
    let words = rows.hidden / 2;
    let w: Vec<u32> = dispatch::read_back_at_arm(
        ctx,
        &rows.buf,
        0,
        rows.rows * words,
        dispatch::ReadbackArm::Pumped,
    )
    .await
    .map_err(|e| anyhow::anyhow!("resident vision rows readback: {e}"))?;
    Ok(unpack_bf16_pair_words(&w))
}

struct PfGpuVisionRows {
    buf: wgpu::Buffer,
    vs: usize,
    n_vis: usize,
}

enum RowsSrc<'a> {
    None,
    Host(&'a [f32]),
    Gpu(&'a VisionRowsGpu),
}

pub enum RegionRowsSrc<'a> {
    None,
    Host(&'a [f32]),
    Gpu(&'a VisionRowsGpu),
}

impl<'a> RegionRowsSrc<'a> {
    fn reborrow(&self) -> RegionRowsSrc<'a> {
        match self {
            Self::None => RegionRowsSrc::None,
            Self::Host(f) => RegionRowsSrc::Host(f),
            Self::Gpu(g) => RegionRowsSrc::Gpu(g),
        }
    }
}

pub struct RegionPackJob<'a> {
    pub slot: usize,
    pub prompt_tokens: &'a [u32],
    pub vision_start: usize,
    pub src: RegionRowsSrc<'a>,
}

struct RpfGpuCopy<'a> {
    buf: &'a wgpu::Buffer,
    src_row: usize,
    dst_row: usize,
    rows: usize,
}

pub const PACK_ROWS_BF16_WGSL: &str = "
struct DvhPackParams { n_words: u32, src_off: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> dvh_x: array<f32>;
@group(0) @binding(1) var<storage, read_write> dvh_y: array<u32>;
@group(0) @binding(2) var<uniform> dvh_p: DvhPackParams;

fn dvh_bf16e(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let r = 0x7fffu + ((b >> 16u) & 1u);
    return select((b + r) >> 16u, 0x7fc0u, x != x);
}

@compute @workgroup_size(256)
fn dvh_pack_rows(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) tid: vec3<u32>) {
    let i = wg.x * 256u + tid.x;
    if (i >= dvh_p.n_words) { return; }
    let s = dvh_p.src_off + i;
    dvh_y[i] = dvh_bf16e(dvh_x[2u * s]) | (dvh_bf16e(dvh_x[2u * s + 1u]) << 16u);
}
";

static PACK_ROWS_PIPELINE: std::sync::OnceLock<std::sync::Arc<wgpu::ComputePipeline>> =
    std::sync::OnceLock::new();

pub fn pack_f32_rows_to_bf16_words(
    ctx: &'static WgpuContext,
    src: &wgpu::Buffer,
    words: usize,
) -> Result<wgpu::Buffer> {
    pack_f32_rows_to_bf16_words_at(ctx, src, 0, words)
}

pub fn pack_f32_rows_to_bf16_words_at(
    ctx: &'static WgpuContext,
    src: &wgpu::Buffer,
    src_word_off: usize,
    words: usize,
) -> Result<wgpu::Buffer> {
    let _ = HANDOFF_SPLICE_WORDS_MATCH_BECAUSE_EVERY_HOST_HOP_ONLY_EVER_ROUNDED_ONCE;
    if PACK_ROWS_PIPELINE.get().is_none() {
        let p = dispatch::compute_pipeline_opts(
            ctx,
            "dg-handoff-pack",
            PACK_ROWS_BF16_WGSL,
            "dvh_pack_rows",
            false,
        )
        .map_err(|e| anyhow::anyhow!("handoff pack pipeline: {e}"))?;
        let _ = PACK_ROWS_PIPELINE.set(std::sync::Arc::new(p));
    }
    let pipeline = PACK_ROWS_PIPELINE
        .get()
        .expect("pack pipeline present")
        .clone();
    let groups = words.div_ceil(256);
    anyhow::ensure!(
        groups <= 65535,
        "handoff pack of {words} words needs {groups} workgroups, over the 1d grid limit"
    );
    anyhow::ensure!(
        src.size() >= ((src_word_off + words) * 8) as u64,
        "pack source holds {} bytes, want {} f32 pairs from pair offset {src_word_off}",
        src.size(),
        words
    );
    let out = dispatch::storage_zeroed(ctx, "dg-handoff-rows", (words * 4) as u64);
    let p = dispatch::uniform_from(
        ctx,
        "dg-handoff-pack-p",
        &[words as u32, src_word_off as u32, 0u32, 0u32],
    );
    dispatch::dispatch(
        ctx,
        &pipeline,
        &[(0, src), (1, &out), (2, &p)],
        (groups as u32, 1, 1),
    )
    .map_err(|e| anyhow::anyhow!("handoff pack dispatch: {e}"))?;
    Ok(out)
}

pub struct DotsDecoderWgpuGraph {
    ctx: &'static WgpuContext,
    cfg: DotsDecoderConfig,
    quant: DotsGraphQuant,
    max_seq: usize,
    pos: usize,
    repack_s: f64,
    rpf_repack_s: f64,
    bstep_cpu_s: f64,
    bstep_read_s: f64,
    bstep_count: u64,
    bstep_bhead: u64,
    bstep_bhead_mma: u64,
    trunk_splitk_families: u64,
    trunk_bf16out_families: u64,
    trunk_abf16_families: u64,
    brope_kvw_fused_layers: u64,
    battn_v2: bool,
    sg_tree: SgTreeSites,
    rpf_chunk_count: u64,
    step_submits: u32,
    ra_gpu_fed: u64,
    validated: bool,
    prefix_validated: bool,
    pf_validated: bool,
    passes: Vec<Pass>,
    pf_passes: Vec<Vec<Pass>>,
    split_passes: Vec<Pass>,
    attn_idx: Vec<usize>,
    pf: Vec<PfBuffers>,
    pf_widths: Vec<usize>,
    embed_end: usize,
    head_start: usize,
    res: wgpu::Buffer,
    _buffers: Vec<wgpu::Buffer>,
    step: StepBuffers,
    token_out: wgpu::Buffer,
    spec_tok: Vec<(usize, wgpu::Buffer)>,
    logits: wgpu::Buffer,
    state_buffers: Vec<wgpu::Buffer>,
    vocab: usize,
    vram: VramReport,
    region: Option<RegionMachinery>,
    pack: Option<PackMachinery>,
    rpf_packs: u64,
    rpf_pack_lanes: u64,
    pf_gpu: Option<PfGpuVisionRows>,
    handoff_copies: u64,
}

impl DotsDecoderWgpuGraph {
    pub fn from_loader(
        cfg: DotsDecoderConfig,
        weights: &dyn nv_weights::TensorSource,
        max_seq: usize,
        quant: DotsGraphQuant,
    ) -> Result<Self> {
        let lora = DgLora::from_env(&cfg)?;
        Self::from_loader_with_lora(cfg, weights, max_seq, quant, lora)
    }

    pub fn from_loader_with_lora(
        cfg: DotsDecoderConfig,
        weights: &dyn nv_weights::TensorSource,
        max_seq: usize,
        quant: DotsGraphQuant,
        lora: Option<DgLora>,
    ) -> Result<Self> {
        Self::build_from_source(cfg, weights, None, max_seq, quant, lora)
    }

    pub fn from_loader_mlx(
        cfg: DotsDecoderConfig,
        loader: &nv_weights::WeightLoader,
        max_seq: usize,
    ) -> Result<Self> {
        let lora = DgLora::from_env(&cfg)?;
        Self::from_loader_mlx_with_lora(cfg, loader, max_seq, lora)
    }

    pub fn from_loader_mlx_with_lora(
        cfg: DotsDecoderConfig,
        loader: &nv_weights::WeightLoader,
        max_seq: usize,
        lora: Option<DgLora>,
    ) -> Result<Self> {
        let mlx = DgMlxWeights::new(loader)?;
        Self::build_from_source(
            cfg,
            &mlx,
            Some(&mlx),
            max_seq,
            DotsGraphQuant::AffineW4,
            lora,
        )
    }

    fn build_from_source(
        cfg: DotsDecoderConfig,
        weights: &dyn nv_weights::TensorSource,
        mlx: Option<&DgMlxWeights>,
        max_seq: usize,
        quant: DotsGraphQuant,
        lora: Option<DgLora>,
    ) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let n_q = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let q_rows = n_q * head_dim;
        let kv_rows = n_kv * head_dim;
        let inter = cfg.intermediate_size;
        let vocab = cfg.vocab_size;
        let layers = cfg.num_hidden_layers;
        let eps = cfg.rms_norm_eps as f32;
        let hidden_words = hidden / 2;

        anyhow::ensure!(max_seq > 0, "max_seq must be positive");
        anyhow::ensure!(
            hidden.is_multiple_of(2) && inter.is_multiple_of(2),
            "hidden {hidden} and intermediate {inter} must be even for bf16 word packing"
        );
        anyhow::ensure!(
            head_dim.is_multiple_of(2)
                && head_dim <= ROPE_WORKGROUP_STAGING_ARRAY_CAPS_HEAD_DIM_AT_256,
            "head_dim {head_dim} must be even and at most {}",
            ROPE_WORKGROUP_STAGING_ARRAY_CAPS_HEAD_DIM_AT_256
        );
        anyhow::ensure!(
            n_kv > 0 && n_q.is_multiple_of(n_kv),
            "{n_q} query heads is not a multiple of {n_kv} key-value heads"
        );

        let fold = attn_fold_factor(&cfg);
        let sg_tree = sg_tree_sites(ctx.subgroup_width());
        let s = Sources::new(fold, kv_transpose_enabled(), sg_tree.attn);
        let mut b = Builder::new(ctx);
        let step = StepBuffers::alloc(&mut b);
        let sl = SingleSlot {
            sel0: b.upload_u32("dg-sel0", &[0u32]),
            global1: b.upload_f32(
                "dg-global1",
                &[NVFP4_ACTIVATION_GLOBAL_2POW7_LANDS_DECODE_BLOCK_SCALES_INSIDE_THE_E4M3_WINDOW],
            ),
        };

        let res = b.zeros("dg-res", wbytes(hidden));
        let normed = alloc_act(&mut b, "dg-normed", hidden, quant);
        let normed_post = alloc_act(&mut b, "dg-normed-post", hidden, quant);
        let heads = alloc_act(&mut b, "dg-heads", q_rows, quant);
        let act = alloc_act(&mut b, "dg-act", inter, quant);
        let pf_widths: Vec<usize> = if quant.bf16_like() {
            let mut w = prefill_widths();
            w.retain(|&m| m <= max_seq);
            w
        } else {
            Vec::new()
        };
        let pf: Vec<PfBuffers> = pf_widths
            .iter()
            .copied()
            .map(|m| {
                let qkv_rows = q_rows + 2 * kv_rows;
                PfBuffers {
                    m,
                    tok: b.upload_u32("dg-pf-tok", &vec![0u32; m]),
                    mask: b.upload_u32("dg-pf-mask", &vec![0u32; m]),
                    splice: b.zeros("dg-pf-splice", (m * hidden * 2) as u64),
                    res: b.zeros("dg-pf-res", (m * hidden * 2) as u64),
                    normed: b.zeros("dg-pf-normed", (m * hidden * 2) as u64),
                    normed_post: b.zeros("dg-pf-normed-post", (m * hidden * 2) as u64),
                    qkv: b.zeros("dg-pf-qkv", (m * qkv_rows * 4) as u64),
                    qkv_rot: b.zeros("dg-pf-qkvrot", (m * qkv_rows * 2) as u64),
                    heads: b.zeros("dg-pf-heads", (m * q_rows * 2) as u64),
                    attn_out: b.zeros("dg-pf-attn-out", (m * hidden * 2) as u64),
                    ffn_out: b.zeros("dg-pf-ffn-out", (m * hidden * 2) as u64),
                    y_gu: b.zeros("dg-pf-ygu", (m * 4 * inter) as u64),
                    act: b.zeros("dg-pf-act", (m * inter * 2) as u64),
                    x_f32: b.zeros("dg-pf-xf32", (m * hidden * 4) as u64),
                    y_f32: b.zeros("dg-pf-yf32", (m * 2 * inter * 4) as u64),
                    zero: b.upload_f32("dg-pf-zero", &vec![0f32; 64]),
                }
            })
            .collect();
        b.pf_lists = (0..pf.len()).map(|_| Vec::new()).collect();
        let region_wi = pf
            .iter()
            .position(|p| p.m == REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES)
            .filter(|_| {
                max_seq >= 64 * REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES
            });
        let rg_bufs = region_wi.map(|_| {
            let slots = REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES;
            (
                b.upload_u32("dg-rg-bpos", &vec![0u32; slots]),
                b.upload_u32("dg-rg-bslot", &vec![0u32; slots]),
                b.upload_u32("dg-rg-brange", &vec![0u32; 2 * slots]),
                b.zeros("dg-rg-btoks", (slots * 4) as u64),
            )
        });
        let mut rg_sites: Vec<Vec<SwapSite>> = pf.iter().map(|_| Vec::new()).collect();
        let n_splits = attn_split_count(max_seq);
        let split_on = attn_split_enabled()
            && max_seq
                > ATTN_SPLIT_MIN_CONTEXT_IS_WHERE_ONE_WORKGROUP_PER_HEAD_STOPS_FILLING_THE_CORES;
        let attn_part =
            split_on.then(|| b.zeros("dg-attn-part", (n_q * n_splits * (head_dim + 2) * 4) as u64));
        let attn_out = b.zeros("dg-attn-out", wbytes(hidden));
        let ffn_out = b.zeros("dg-ffn-out", wbytes(hidden));
        let final_x = b.zeros("dg-final-x", wbytes(hidden));

        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / cfg.rope_theta.powf((i as f32 * 2.0) / head_dim as f32))
            .collect();
        let (cos, sin) = rope_tables_from_inv_freq(&inv_freq, max_seq);
        let rope_cos = b.upload_f32("dg-rope-cos", &cos);
        let rope_sin = b.upload_f32("dg-rope-sin", &sin);

        let chunk_rows = b.row_chunk(hidden);
        let mut splits: Vec<(usize, usize)> = Vec::new();
        let mut off = 0usize;
        while off < vocab {
            let rows = chunk_rows.min(vocab - off);
            splits.push((off, rows));
            off += rows;
        }

        let embed = match mlx {
            Some(m) => m.dequant_bf16_host("model.embed_tokens.weight", vocab, hidden)?,
            None => load_bf16(weights, "model.embed_tokens.weight", &[vocab, hidden])?,
        };
        let mut embed_chunks: Vec<wgpu::Buffer> = Vec::new();
        for (off, rows) in splits.iter().copied() {
            let buf = b.upload_u32(
                "dg-embed",
                &pack_pairs(&embed[off * hidden..(off + rows) * hidden]),
            );
            let p = b.uni(
                "dg-embed-p",
                GatherParams {
                    row_off: off as u32,
                    n_rows: rows as u32,
                    hidden_words: hidden_words as u32,
                    vocab: vocab as u32,
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.push(
                "dg-gather",
                &s.common,
                "lgw_gather_embed",
                &[(30, &buf), (31, &step.tok), (32, &res), (33, &p)],
                grid,
            )?;
            embed_chunks.push(buf);
        }
        let embed_end = b.passes.len();

        let tied = cfg.tie_word_embeddings || !weights.has("lm_head.weight");
        drop(embed);
        let w4_dec = quant == DotsGraphQuant::AffineW4
            || (quant == DotsGraphQuant::Bf16 && dec_w4_enabled());
        let (head_host, lm_w4) = if tied {
            (Vec::new(), None)
        } else if w4_dec && splits.len() == 1 {
            lin_prep(
                weights,
                mlx,
                &[("lm_head.weight".to_string(), vocab)],
                hidden,
                quant,
                true,
            )?
        } else {
            (
                load_bf16(weights, "lm_head.weight", &[vocab, hidden])?,
                None,
            )
        };
        let act_f16 = quant == DotsGraphQuant::AffineW4
            && !tied
            && splits.len() == 1
            && msl_gemv_enabled()
            && decode_gemv_msl::msl_gemv_pipelines(ctx).is_ok();

        let dg_lora = lora;
        let dg_lora_act16 = if act_f16 { DgAct16::F16 } else { DgAct16::Bf16 };
        let _ = SINK16_IS_BF16_ON_EVERY_ARM_BECAUSE_THE_MSL_GEMV_WRITES_BF16E;
        let dg_lora_sink16 = DgSink::Bf16;
        let dg_lora_pf_m = pf.iter().map(|p| p.m).max().unwrap_or(0).max(1);

        let (fuse_qkv, fuse_gu) = if act_f16 && dg_lora.is_none() {
            rms_gemv_fuse_mode()
        } else {
            (false, false)
        };
        let gsilu_fuse =
            act_f16 && dg_lora.is_none() && !fuse_qkv && !fuse_gu && dec_gsilu_enabled();
        let res_f32 = b.zeros("dg-res-f32", (hidden * 4) as u64);

        let ln0 = load_bf16(
            weights,
            &layer_tensor(0, "input_layernorm.weight"),
            &[hidden],
        )?;
        let ln0w = b.upload_u32("dg-ln0", &pack_pairs(&ln0));
        if !fuse_qkv {
            push_rmsnorm_fmt(
                &mut b, &s, "dg-rms0", &res, &ln0w, &normed.x, hidden, eps, 1, act_f16,
            )?;
        }
        let mut cur_ln = ln0w.clone();

        for (pfi, pfb) in pf.iter().enumerate() {
            let pf = pfb;
            b.pf_target = Some(pfi);
            push_gather_m(
                &mut b,
                &s,
                "dg-pf-splice",
                "dg_splice_m",
                &pf.splice,
                &pf.tok,
                &pf.mask,
                &pf.res,
                0,
                0,
                hidden_words,
                pf.m,
            )?;
            for (i, (off, rows)) in splits.iter().copied().enumerate() {
                push_gather_m(
                    &mut b,
                    &s,
                    "dg-pf-gather",
                    "dg_gather_m",
                    &embed_chunks[i],
                    &pf.tok,
                    &pf.mask,
                    &pf.res,
                    off,
                    rows,
                    hidden_words,
                    pf.m,
                )?;
            }
            push_rmsnorm(
                &mut b,
                &s,
                "dg-pf-rms0",
                &pf.res,
                &ln0w,
                &pf.normed,
                hidden,
                eps,
                pf.m,
            )?;
            b.pf_target = None;
        }

        let tile_shaped = head_dim <= ATTN_TILE_MAX_HEAD_DIM
            && !pf.is_empty()
            && pf.iter().all(|p| p.m.is_multiple_of(ATTN_TILE_QUERY_ROWS));
        let prefill_attn_entry = if tile_shaped && attn_reg_enabled() {
            Some("dg_attn_reg")
        } else if tile_shaped && attn_tiled_enabled() {
            Some("dg_attn_tiled")
        } else {
            None
        };
        let steel_prefill = pf_steel_enabled()
            && head_dim == ATTN_FUSED_HEAD_DIM
            && !pf.is_empty()
            && pf.iter().all(|p| {
                p.m.is_multiple_of(super::vision_attn_msl::STEEL_BQ as usize)
            })
            && super::prefill_attn_msl::causal_pipelines(ctx).is_ok();
        let fused_prefill = attn_fused_enabled()
            && !steel_prefill
            && head_dim == ATTN_FUSED_HEAD_DIM
            && n_kv == 2
            && !pf.is_empty()
            && pf.iter().all(|p| p.m.is_multiple_of(ATTN_FUSED_QUERY_ROWS))
            && ctx.caps.coop_advertises(&coop_plan().request())
            && ctx.subgroup_width() == Some(32);
        let steel_bufs = if steel_prefill {
            let maxm = pf.iter().map(|p| p.m).max().unwrap_or(0);
            let q_words = q_rows / 2;
            let kv_words = kv_rows / 2;
            let qkv_words = (q_rows + 2 * kv_rows) / 2;
            let pls = super::prefill_attn_msl::causal_pipelines(ctx)?;
            let qh = b.zeros("dg-steel-qh", (maxm * q_words * 4) as u64);
            let oh = b.zeros("dg-steel-oh", (maxm * q_words * 4) as u64);
            let mp = b.uni("dg-steel-mp", [0i64; 3]);
            let dummy = b.zeros("dg-steel-dummy", 4);
            let mut sp = Vec::with_capacity(pf.len());
            let mut pu = Vec::with_capacity(pf.len());
            for (i, p) in pf.iter().enumerate() {
                let m = p.m;
                let bq = super::vision_attn_msl::STEEL_BQ as usize;
                let sap = super::vision_attn_msl::SteelAttnParams {
                    b: 1,
                    h: n_q as i32,
                    d: head_dim as i32,
                    q_l: m as i32,
                    k_l: 0,
                    gqa_factor: (n_q / n_kv) as i32,
                    scale: (head_dim as f32).powf(-0.5),
                    nq: (m / bq) as i32,
                    nk: 0,
                    nq_aligned: (m / bq) as i32,
                    nk_aligned: 0,
                    q_l_rem: 0,
                    k_l_rem: 0,
                    q_l_off: 0,
                    q_strides: [
                        (m * n_q * head_dim) as i64,
                        head_dim as i64,
                        (n_q * head_dim) as i64,
                    ],
                    k_strides: [
                        (max_seq * n_kv * head_dim) as i64,
                        head_dim as i64,
                        (n_kv * head_dim) as i64,
                    ],
                    v_strides: [
                        (max_seq * n_kv * head_dim) as i64,
                        head_dim as i64,
                        (n_kv * head_dim) as i64,
                    ],
                    o_strides: [
                        (m * n_q * head_dim) as i64,
                        head_dim as i64,
                        (n_q * head_dim) as i64,
                    ],
                };
                sp.push(steel_params_buffer(
                    &mut b,
                    &format!("dg-steel-sp{i}"),
                    &sap,
                ));
                pu.push(b.uni(
                    &format!("dg-steel-pu{i}"),
                    DgScvtParams {
                        m: m as u32,
                        qkv_words: qkv_words as u32,
                        q_words: q_words as u32,
                        kv_words: kv_words as u32,
                        capacity: max_seq as u32,
                        heads_words: q_words as u32,
                        row_off: 0,
                        pad1: 0,
                    },
                ));
            }
            Some(SteelBufs {
                qh,
                oh,
                sp,
                pu,
                mp,
                dummy,
                pls,
            })
        } else {
            None
        };
        let steel_proj_cfgs: Vec<(u32, u32, u32, u32, u32)> = {
            let mut v = vec![
                pf_steel_proj_cfg_for(PF_STEEL_NARROW_MAX_WIDTH),
                pf_steel_proj_cfg_for(usize::MAX),
            ];
            v.dedup();
            v
        };
        let steel_proj_bufs = if pf_steel_proj_enabled()
            && quant.bf16_like()
            && !pf.is_empty()
            && steel_proj_cfgs.iter().all(|c| {
                hidden.is_multiple_of(2 * c.2 as usize) && inter.is_multiple_of(2 * c.2 as usize)
            }) {
            match super::vision_attn_msl::steel_gemm_pipelines(ctx) {
                Ok(pls) if steel_proj_cfgs.iter().all(|c| pls.pick(*c).is_some()) => {
                    let maxm = pf.iter().map(|p| p.m).max().unwrap_or(0);
                    let kmax = hidden.max(inter);
                    let skm = pf
                        .iter()
                        .map(|p| p.m)
                        .filter(|&m| m <= TRUNK_SPLITK_MAX_WIDTH)
                        .max();
                    let skpart = match (trunk_splitk_enabled(), skm) {
                        (true, Some(skm)) => {
                            let narrow_bk = pf_steel_proj_cfg_for(skm).2;
                            let fams = [
                                (ProjFamily::Qkv, q_rows + 2 * kv_rows, hidden),
                                (ProjFamily::OProj, hidden, q_rows),
                                (ProjFamily::GateUp, 2 * inter, hidden),
                                (ProjFamily::Down, hidden, inter),
                            ];
                            let worst = fams
                                .iter()
                                .map(|(f, n, k)| {
                                    trunk_splitk_s(*f, skm, *k, narrow_bk) * skm * n * 4
                                })
                                .max()
                                .unwrap_or(0);
                            (worst > 0).then(|| b.zeros("dg-steelproj-skpart", worst as u64))
                        }
                        _ => None,
                    };
                    let cvt = if trunk_bf16out_enabled() || trunk_abf16_enabled() {
                        Some(super::vision_attn_msl::steel_gemm_cvt_pipelines(ctx)?)
                    } else {
                        None
                    };
                    Some(SteelProjBufs {
                        xh: b.zeros("dg-steelproj-xh", (maxm * kmax / 2 * 4) as u64),
                        skpart,
                        pls,
                        cvt,
                    })
                }
                _ => None,
            }
        } else {
            None
        };
        let steel_proj = steel_proj_bufs.is_some();
        let coop_prefill = coop_prefill_enabled() && coop_prefill_supported(&ctx.caps);
        let pack_wi = pf
            .iter()
            .position(|p| p.m == PREFILL_BIG_MID_1024_IS_THE_CHUNKED_LADDER_KNEE)
            .filter(|_| {
                region_wi.is_some()
                    && pf.iter().any(|p| p.m == PREFILL_M_WIDE)
                    && max_seq / REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES
                        >= PREFILL_BIG_MID_1024_IS_THE_CHUNKED_LADDER_KNEE
            });
        let pack_lane_wi = pf.iter().position(|p| p.m == PREFILL_M_WIDE);
        let pack_bufs = pack_wi.map(|_| {
            let _ =
                RPF_PACK_SHARES_ONE_TRUNK_READ_ACROSS_REGIONS_BECAUSE_ONLY_ATTENTION_HAS_AN_ORIGIN;
            let m = PREFILL_BIG_MID_1024_IS_THE_CHUNKED_LADDER_KNEE;
            let lanes = RPF_PACK_LANES_8_FILL_THE_1024_TRUNK;
            let q_words = q_rows / 2;
            let kv_words = kv_rows / 2;
            let qkv_words = (q_rows + 2 * kv_rows) / 2;
            let bpos = b.upload_u32("dg-pk-bpos", &vec![0u32; m]);
            let bslot = b.upload_u32("dg-pk-bslot", &vec![u32::MAX; m]);
            let brange = b.upload_u32("dg-pk-brange", &vec![0u32; 2 * m]);
            let steps: Vec<wgpu::Buffer> = (0..lanes)
                .map(|l| b.upload_u32(&format!("dg-pk-step{l}"), &[0u32; 4]))
                .collect();
            let pus: Vec<wgpu::Buffer> = (0..lanes)
                .map(|l| {
                    b.uni(
                        &format!("dg-pk-pu{l}"),
                        DgScvtParams {
                            m: PREFILL_M_WIDE as u32,
                            qkv_words: qkv_words as u32,
                            q_words: q_words as u32,
                            kv_words: kv_words as u32,
                            capacity: max_seq as u32,
                            heads_words: q_words as u32,
                            row_off: (l * PREFILL_M_WIDE) as u32,
                            pad1: 0,
                        },
                    )
                })
                .collect();
            (bpos, bslot, brange, steps, pus)
        });
        let fused_slabs = fused_prefill.then(|| {
            let maxm = pf.iter().map(|p| p.m).max().unwrap_or(0);
            let rows = (n_q / n_kv) * maxm;
            let cap16 = attn_cap16(max_seq);
            FusedSlabs {
                qa: b.zeros("dg-fat-qa", (rows * head_dim * 4) as u64),
                qb: b.zeros("dg-fat-qb", (rows * head_dim * 4) as u64),
                sa: b.zeros("dg-fat-sa", (rows * cap16 * 4) as u64),
                sb: b.zeros("dg-fat-sb", (rows * cap16 * 4) as u64),
                la: b.zeros("dg-fat-la", (rows * 4) as u64),
                lb: b.zeros("dg-fat-lb", (rows * 4) as u64),
                oa: b.zeros("dg-fat-oa", (rows * head_dim * 4) as u64),
                ob: b.zeros("dg-fat-ob", (rows * head_dim * 4) as u64),
            }
        });
        let spec_layers = spec_probe_layers(layers);
        let spec_normw = (!spec_layers.is_empty())
            .then(|| -> Result<wgpu::Buffer> {
                Ok(b.upload_u32(
                    "dg-spec-normw",
                    &pack_pairs(&load_bf16(weights, "model.norm.weight", &[hidden])?),
                ))
            })
            .transpose()?;
        let spec_x: Vec<(usize, wgpu::Buffer)> = spec_layers
            .iter()
            .map(|&n| (n, b.zeros(&format!("dg-spec-x{n}"), wbytes(hidden))))
            .collect();
        let mut attn_idx: Vec<usize> = Vec::with_capacity(layers);
        let zero_bias = b.upload_u32("dg-zero-bias", &[0u32]);
        for li in 0..layers {
            let last = li + 1 == layers;
            if quant == DotsGraphQuant::Nvfp4 && !nvfp4_mrow4_enabled() {
                push_quant(&mut b, &s, &format!("dg-l{li}-qn"), &normed, &sl)?;
            }
            let lora_layer = dg_lora
                .as_ref()
                .map(|l| upload_dg_lora_layer(&mut b, li, &l.layers[li], dg_lora_pf_m));

            let qkv_rows = q_rows + 2 * kv_rows;
            let qkv_names: Vec<(String, usize)> = [
                ("self_attn.q_proj.weight", q_rows),
                ("self_attn.k_proj.weight", kv_rows),
                ("self_attn.v_proj.weight", kv_rows),
            ]
            .iter()
            .map(|(name, rows)| (layer_tensor(li, name), *rows))
            .collect();
            let (qkv_host, qkv_w4) = lin_prep(weights, mlx, &qkv_names, hidden, quant, w4_dec)?;
            let qkv_aff = qkv_w4.map(|pw| {
                upload_affine_w4(&mut b, &format!("dg-l{li}-qkv"), &pw, qkv_rows, hidden)
            });
            let qkvw = upload_lin(
                &mut b,
                &format!("dg-l{li}-qkvw"),
                &qkv_host,
                qkv_rows,
                hidden,
                quant,
            )?;
            let proj_coop = coop_prefill && coop_proj_enabled() && quant.bf16_like();
            let qkvw_f32 = (proj_coop && !steel_proj)
                .then(|| upload_f32_of_bf16(&mut b, &format!("dg-l{li}-qkvw-f32"), &qkv_host));
            let qkvw_h = steel_proj
                .then(|| upload_f16_of_bf16(&mut b, &format!("dg-l{li}-qkvw-h"), &qkv_host));
            drop(qkv_host);
            let (o_host, o_w4) = lin_prep(
                weights,
                mlx,
                &[(layer_tensor(li, "self_attn.o_proj.weight"), hidden)],
                q_rows,
                quant,
                w4_dec,
            )?;
            let o_aff = o_w4
                .map(|pw| upload_affine_w4(&mut b, &format!("dg-l{li}-o"), &pw, hidden, q_rows));
            let ow = upload_lin(
                &mut b,
                &format!("dg-l{li}-ow"),
                &o_host,
                hidden,
                q_rows,
                quant,
            )?;
            let ow_f32 = (proj_coop && !steel_proj)
                .then(|| upload_f32_of_bf16(&mut b, &format!("dg-l{li}-ow-f32"), &o_host));
            let ow_h =
                steel_proj.then(|| upload_f16_of_bf16(&mut b, &format!("dg-l{li}-ow-h"), &o_host));
            drop(o_host);

            let qkv_f32 = quant.bf16_like();
            let qkv = b.zeros(
                &format!("dg-l{li}-qkv"),
                if qkv_f32 {
                    (qkv_rows * 4) as u64
                } else {
                    wbytes(qkv_rows)
                },
            );
            let qkv_rot = b.zeros(&format!("dg-l{li}-qkvrot"), wbytes(qkv_rows));

            match &qkv_aff {
                Some(aff) if fuse_qkv => push_gemv_w4_rms(
                    &mut b,
                    &format!("dg-l{li}-qkvw4"),
                    aff,
                    &normed.x,
                    &qkv,
                    true,
                    RmsFuseSite {
                        rms_mode: if li == 0 { 1 } else { 2 },
                        fold_res: false,
                        nw: &cur_ln,
                        res: &res,
                        res_f32: &res_f32,
                        eps,
                    },
                )?,
                Some(aff) => push_gemv_w4(
                    &mut b,
                    &format!("dg-l{li}-qkvw4"),
                    aff,
                    &normed.x,
                    &qkv,
                    true,
                    act_f16,
                )?,
                None => push_gemv(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-qkvproj"),
                    &qkvw,
                    &normed,
                    &qkv,
                    qkv_f32,
                    0,
                )?,
            }

            if let Some(ll) = &lora_layer {
                push_dg_lora_slices(
                    &mut b,
                    &format!("dg-l{li}-lora-qkv"),
                    ll,
                    &[
                        ("q_proj", 0),
                        ("k_proj", q_rows),
                        ("v_proj", q_rows + kv_rows),
                    ],
                    &normed.x,
                    dg_lora_act16,
                    &qkv,
                    if qkv_f32 { DgSink::F32 } else { dg_lora_sink16 },
                    qkv_rows,
                    1,
                )?;
            }

            let bias_names = [
                "self_attn.q_proj.bias",
                "self_attn.k_proj.bias",
                "self_attn.v_proj.bias",
            ];
            let has_bias =
                cfg.attention_bias && bias_names.iter().all(|n| weights.has(&layer_tensor(li, n)));
            let bias_buf = if has_bias {
                let mut host = Vec::with_capacity(qkv_rows);
                for (name, rows) in bias_names.iter().zip([q_rows, kv_rows, kv_rows]) {
                    host.extend_from_slice(&load_bf16(weights, &layer_tensor(li, name), &[rows])?);
                }
                b.upload_u32(&format!("dg-l{li}-qkvbias"), &pack_pairs(&host))
            } else {
                zero_bias.clone()
            };
            let kc = b.state_zeros(
                &format!("dg-l{li}-kc"),
                (max_seq * (kv_rows / 2) * 4) as u64,
            );
            let vc = b.state_zeros(
                &format!("dg-l{li}-vc"),
                (max_seq * (kv_rows / 2) * 4) as u64,
            );
            let rope_site = RopeSite {
                src: &qkv,
                bias: &bias_buf,
                cos: &rope_cos,
                sin: &rope_sin,
                dst: &qkv_rot,
                n_rows: n_q + 2 * n_kv,
                rope_rows: n_q + n_kv,
                head_dim,
                has_bias,
                src_f32: qkv_f32,
                slots: 1,
                src_slot_words: 0,
                dst_slot_words: 0,
            };
            let rope_kv_fused = rope_kv_fuse_enabled();
            if rope_kv_fused {
                push_bias_rope_kvw(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-qkvrope"),
                    rope_site,
                    &step,
                    &kc,
                    &vc,
                    kv_rows,
                    max_seq,
                )?;
            } else {
                push_bias_rope(&mut b, &s, &format!("dg-l{li}-qkvrope"), rope_site, &step)?;
            }

            let fused_mir = fused_prefill.then(|| {
                let cap16 = attn_cap16(max_seq);
                FusedMirrors {
                    kf0: b.zeros(&format!("dg-l{li}-kf0"), (cap16 * head_dim * 4) as u64),
                    kf1: b.zeros(&format!("dg-l{li}-kf1"), (cap16 * head_dim * 4) as u64),
                    vt0: b.zeros(&format!("dg-l{li}-vt0"), (head_dim * cap16 * 4) as u64),
                    vt1: b.zeros(&format!("dg-l{li}-vt1"), (head_dim * cap16 * 4) as u64),
                }
            });
            let steel_mir = steel_prefill.then(|| {
                (
                    b.zeros(&format!("dg-l{li}-skh"), (max_seq * kv_rows / 2 * 4) as u64),
                    b.zeros(&format!("dg-l{li}-svh"), (max_seq * kv_rows / 2 * 4) as u64),
                )
            });
            if !rope_kv_fused {
                push_kv_write(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-kvwrite"),
                    &qkv_rot,
                    &step,
                    &kc,
                    &vc,
                    kv_rows,
                    q_rows,
                    q_rows + kv_rows,
                    max_seq,
                    1,
                    0,
                    n_kv,
                )?;
            }

            attn_idx.push(b.passes.len());
            push_attn(
                &mut b,
                &s,
                &format!("dg-l{li}-attn"),
                &qkv_rot,
                &kc,
                &vc,
                &step,
                &heads.x,
                &cfg,
                max_seq,
                1,
                0,
                0,
                act_f16,
            )?;
            if let Some(part) = &attn_part {
                b.split_mode = true;
                push_attn_split(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-attnsplit"),
                    &qkv_rot,
                    &kc,
                    &vc,
                    &step,
                    part,
                    &heads.x,
                    &cfg,
                    max_seq,
                    n_splits,
                    fold,
                    act_f16,
                )?;
                b.split_mode = false;
            }

            if quant == DotsGraphQuant::Nvfp4 && !nvfp4_mrow4_enabled() {
                push_quant(&mut b, &s, &format!("dg-l{li}-qh"), &heads, &sl)?;
            }
            match &o_aff {
                Some(aff) if fuse_gu => push_gemv_w4_rms(
                    &mut b,
                    &format!("dg-l{li}-ow4"),
                    aff,
                    &heads.x,
                    &attn_out,
                    false,
                    RmsFuseSite {
                        rms_mode: 0,
                        fold_res: true,
                        nw: &zero_bias,
                        res: &res,
                        res_f32: &res_f32,
                        eps,
                    },
                )?,
                Some(aff) => push_gemv_w4(
                    &mut b,
                    &format!("dg-l{li}-ow4"),
                    aff,
                    &heads.x,
                    &attn_out,
                    false,
                    act_f16,
                )?,
                None => push_gemv(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-oproj"),
                    &ow,
                    &heads,
                    &attn_out,
                    false,
                    0,
                )?,
            }

            if let Some(ll) = &lora_layer {
                push_dg_lora_slices(
                    &mut b,
                    &format!("dg-l{li}-lora-o"),
                    ll,
                    &[("o_proj", 0)],
                    &heads.x,
                    dg_lora_act16,
                    &attn_out,
                    dg_lora_sink16,
                    hidden,
                    1,
                )?;
            }

            let post = load_bf16(
                weights,
                &layer_tensor(li, "post_attention_layernorm.weight"),
                &[hidden],
            )?;
            let postw = b.upload_u32(&format!("dg-l{li}-postln"), &pack_pairs(&post));
            if !fuse_gu {
                push_rmsnorm_residual_fmt(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-rmsres"),
                    &attn_out,
                    &res,
                    &postw,
                    &normed_post.x,
                    hidden,
                    eps,
                    1,
                    act_f16,
                )?;
            }

            if quant == DotsGraphQuant::Nvfp4 && !nvfp4_mrow4_enabled() {
                push_quant(&mut b, &s, &format!("dg-l{li}-qp"), &normed_post, &sl)?;
            }
            let gu_names: Vec<(String, usize)> = ["mlp.gate_proj.weight", "mlp.up_proj.weight"]
                .iter()
                .map(|name| (layer_tensor(li, name), inter))
                .collect();
            let (gu_host, gu_w4) = lin_prep(weights, mlx, &gu_names, hidden, quant, w4_dec)?;
            let gu_aff = gu_w4.map(|pw| {
                upload_affine_w4(&mut b, &format!("dg-l{li}-gu"), &pw, 2 * inter, hidden)
            });
            let gu = upload_lin(
                &mut b,
                &format!("dg-l{li}-gateup"),
                &gu_host,
                2 * inter,
                hidden,
                quant,
            )?;
            let gu_f32 = (coop_prefill && quant.bf16_like() && !steel_proj)
                .then(|| upload_f32_of_bf16(&mut b, &format!("dg-l{li}-gateup-f32"), &gu_host));
            let gu_h = steel_proj
                .then(|| upload_f16_of_bf16(&mut b, &format!("dg-l{li}-gateup-h"), &gu_host));
            drop(gu_host);
            let (down_host, down_w4) = lin_prep(
                weights,
                mlx,
                &[(layer_tensor(li, "mlp.down_proj.weight"), hidden)],
                inter,
                quant,
                w4_dec,
            )?;
            let down_aff = down_w4
                .map(|pw| upload_affine_w4(&mut b, &format!("dg-l{li}-down"), &pw, hidden, inter));
            let down = upload_lin(
                &mut b,
                &format!("dg-l{li}-down"),
                &down_host,
                hidden,
                inter,
                quant,
            )?;
            let downw_f32 = (proj_coop && !steel_proj)
                .then(|| upload_f32_of_bf16(&mut b, &format!("dg-l{li}-down-f32"), &down_host));
            let down_h = steel_proj
                .then(|| upload_f16_of_bf16(&mut b, &format!("dg-l{li}-down-h"), &down_host));
            drop(down_host);
            let y_gu = b.zeros(&format!("dg-l{li}-ygu"), wbytes(2 * inter));
            match &gu_aff {
                Some(aff) if fuse_gu => push_gemv_w4_rms(
                    &mut b,
                    &format!("dg-l{li}-guw4"),
                    aff,
                    &normed_post.x,
                    &y_gu,
                    false,
                    RmsFuseSite {
                        rms_mode: 2,
                        fold_res: false,
                        nw: &postw,
                        res: &res,
                        res_f32: &res_f32,
                        eps,
                    },
                )?,
                Some(aff) if gsilu_fuse => push_gemv_w4_gsilu(
                    &mut b,
                    &format!("dg-l{li}-guw4"),
                    aff,
                    &normed_post.x,
                    &act.x,
                )?,
                Some(aff) => push_gemv_w4(
                    &mut b,
                    &format!("dg-l{li}-guw4"),
                    aff,
                    &normed_post.x,
                    &y_gu,
                    false,
                    act_f16,
                )?,
                None => push_gemv(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-gateupproj"),
                    &gu,
                    &normed_post,
                    &y_gu,
                    false,
                    0,
                )?,
            }
            if let Some(ll) = &lora_layer {
                push_dg_lora_slices(
                    &mut b,
                    &format!("dg-l{li}-lora-gu"),
                    ll,
                    &[("gate_proj", 0), ("up_proj", inter)],
                    &normed_post.x,
                    dg_lora_act16,
                    &y_gu,
                    dg_lora_sink16,
                    2 * inter,
                    1,
                )?;
            }

            if !gsilu_fuse {
                push_silu_mul_fmt(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-silu"),
                    &y_gu,
                    &act.x,
                    inter,
                    1,
                    0,
                    0,
                    act_f16,
                )?;
            }
            if quant == DotsGraphQuant::Nvfp4 && !nvfp4_mrow4_enabled() {
                push_quant(&mut b, &s, &format!("dg-l{li}-qa"), &act, &sl)?;
            }
            match &down_aff {
                Some(aff) if fuse_qkv && !last => push_gemv_w4_rms(
                    &mut b,
                    &format!("dg-l{li}-dw4"),
                    aff,
                    &act.x,
                    &ffn_out,
                    false,
                    RmsFuseSite {
                        rms_mode: 0,
                        fold_res: true,
                        nw: &zero_bias,
                        res: &res,
                        res_f32: &res_f32,
                        eps,
                    },
                )?,
                Some(aff) => push_gemv_w4(
                    &mut b,
                    &format!("dg-l{li}-dw4"),
                    aff,
                    &act.x,
                    &ffn_out,
                    false,
                    act_f16,
                )?,
                None => push_gemv(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-downproj"),
                    &down,
                    &act,
                    &ffn_out,
                    false,
                    0,
                )?,
            }

            if let Some(ll) = &lora_layer {
                push_dg_lora_slices(
                    &mut b,
                    &format!("dg-l{li}-lora-down"),
                    ll,
                    &[("down_proj", 0)],
                    &act.x,
                    dg_lora_act16,
                    &ffn_out,
                    dg_lora_sink16,
                    hidden,
                    1,
                )?;
            }

            for (pfi, pfb) in pf.iter().enumerate() {
                let pf = pfb;
                let qkv_words = (q_rows + 2 * kv_rows) / 2;
                let q_words = q_rows / 2;
                b.pf_target = Some(pfi);
                let proj_wide = |n: usize, k: usize| {
                    pf.m >= PREFILL_M_WIDE
                        && coop_plan()
                            .check_shape(pf.m as u32, n as u32, k as u32)
                            .is_ok()
                };
                if pf_ablated("proj") {
                } else if let Some(sp) = &steel_proj_bufs {
                    push_proj_steel(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-qkvsteel"),
                        ProjFamily::Qkv,
                        qkvw_h.as_ref().unwrap(),
                        &pf.normed,
                        &pf.qkv,
                        None,
                        sp,
                        pf.m,
                        q_rows + 2 * kv_rows,
                        hidden,
                    )?;
                } else {
                    match qkvw_f32
                        .as_ref()
                        .filter(|_| proj_wide(q_rows + 2 * kv_rows, hidden))
                    {
                        Some(wf) => {
                            push_cvt(
                                &mut b,
                                &s,
                                &format!("dg-pf-l{li}-qkvcvtin"),
                                &pf.normed,
                                &pf.x_f32,
                                hidden,
                                hidden_words,
                                hidden,
                                pf.m,
                                true,
                            )?;
                            push_coop_gemm(
                                &mut b,
                                &format!("dg-pf-l{li}-qkvcoop"),
                                wf,
                                &pf.x_f32,
                                &pf.qkv,
                                &pf.zero,
                                pf.m,
                                q_rows + 2 * kv_rows,
                                hidden,
                            )?;
                        }
                        None => push_gemm_m8(
                            &mut b,
                            &s,
                            &format!("dg-pf-l{li}-qkv"),
                            &qkvw,
                            &pf.normed,
                            &pf.qkv,
                            hidden_words,
                            q_rows + 2 * kv_rows,
                            true,
                            pf.m,
                        )?,
                    }
                }
                if let Some(ll) = lora_layer.as_ref().filter(|_| !pf_ablated("proj")) {
                    push_dg_lora_slices(
                        &mut b,
                        &format!("dg-pf-l{li}-lora-qkv"),
                        ll,
                        &[
                            ("q_proj", 0),
                            ("k_proj", q_rows),
                            ("v_proj", q_rows + kv_rows),
                        ],
                        &pf.normed,
                        DgAct16::Bf16,
                        &pf.qkv,
                        DgSink::F32,
                        q_rows + 2 * kv_rows,
                        pf.m,
                    )?;
                }
                let rope_site = RopeSite {
                    src: &pf.qkv,
                    bias: &bias_buf,
                    cos: &rope_cos,
                    sin: &rope_sin,
                    dst: &pf.qkv_rot,
                    n_rows: n_q + 2 * n_kv,
                    rope_rows: n_q + n_kv,
                    head_dim,
                    has_bias,
                    src_f32: true,
                    slots: pf.m,
                    src_slot_words: qkv_words,
                    dst_slot_words: qkv_words,
                };
                let rg_rope_start = b.pf_lists[pfi].len();
                push_bias_rope(&mut b, &s, &format!("dg-pf-l{li}-rope"), rope_site, &step)?;
                if let Some((bpos, bslot, _, _)) =
                    rg_bufs.as_ref().filter(|_| Some(pfi) == region_wi)
                {
                    let kvw = rope_kv_fuse_b_enabled().then(|| RopeKvwSite {
                        kc: &kc,
                        vc: &vc,
                        bslot,
                        kv_rows,
                        k_off_rows: q_rows,
                        v_off_rows: q_rows + kv_rows,
                        capacity: max_seq,
                        n_kv_heads: n_kv,
                    });
                    if kvw.is_some() {
                        b.brope_kvw_layers += 1;
                    }
                    let bat = region_rope_pass(
                        &mut b,
                        &s,
                        &format!("dg-rg-l{li}-rope"),
                        rope_site,
                        &step,
                        bpos,
                        kvw,
                    )?;
                    rg_sites[pfi].push(SwapSite {
                        start: rg_rope_start,
                        len: b.pf_lists[pfi].len() - rg_rope_start,
                        rpf: None,
                        bat: Some(vec![bat]),
                        pack: None,
                        pack_lane: 0,
                    });
                }
                if let Some((pk_bpos, _, _, _, _)) =
                    pack_bufs.as_ref().filter(|_| Some(pfi) == pack_wi)
                {
                    let pkr = region_rope_pass(
                        &mut b,
                        &s,
                        &format!("dg-pk-l{li}-rope"),
                        rope_site,
                        &step,
                        pk_bpos,
                        None,
                    )?;
                    rg_sites[pfi].push(SwapSite {
                        start: rg_rope_start,
                        len: b.pf_lists[pfi].len() - rg_rope_start,
                        rpf: None,
                        bat: None,
                        pack: Some(vec![pkr]),
                        pack_lane: 0,
                    });
                }
                let rg_kv_start = b.pf_lists[pfi].len();
                push_kv_write(
                    &mut b,
                    &s,
                    &format!("dg-pf-l{li}-kvwrite"),
                    &pf.qkv_rot,
                    &step,
                    &kc,
                    &vc,
                    kv_rows,
                    q_rows,
                    q_rows + kv_rows,
                    max_seq,
                    pf.m,
                    qkv_words,
                    n_kv,
                )?;
                if let Some((_, bslot, _, _)) = &rg_bufs {
                    let rpf = region_kv_pass(
                        &mut b,
                        &s,
                        &format!("dg-rg-l{li}-kvr-w{}", pf.m),
                        "dg_kv_write_r",
                        &pf.qkv_rot,
                        &step,
                        &kc,
                        &vc,
                        kv_rows,
                        q_rows,
                        q_rows + kv_rows,
                        max_seq,
                        pf.m,
                        qkv_words,
                        n_kv,
                        None,
                    )?;
                    let bat: Option<Vec<Pass>> = if Some(pfi) == region_wi {
                        if rope_kv_fuse_b_enabled() {
                            Some(Vec::new())
                        } else {
                            Some(vec![region_kv_pass(
                                &mut b,
                                &s,
                                &format!("dg-rg-l{li}-kvb"),
                                "dg_kv_write_b",
                                &pf.qkv_rot,
                                &step,
                                &kc,
                                &vc,
                                kv_rows,
                                q_rows,
                                q_rows + kv_rows,
                                max_seq,
                                pf.m,
                                qkv_words,
                                n_kv,
                                Some(bslot),
                            )?])
                        }
                    } else {
                        None
                    };
                    let pack = if Some(pfi) == pack_wi {
                        let (_, pk_bslot, _, _, _) = pack_bufs.as_ref().unwrap();
                        Some(region_kv_pass(
                            &mut b,
                            &s,
                            &format!("dg-pk-l{li}-kvb"),
                            "dg_kv_write_b",
                            &pf.qkv_rot,
                            &step,
                            &kc,
                            &vc,
                            kv_rows,
                            q_rows,
                            q_rows + kv_rows,
                            max_seq,
                            pf.m,
                            qkv_words,
                            n_kv,
                            Some(pk_bslot),
                        )?)
                    } else {
                        None
                    };
                    rg_sites[pfi].push(SwapSite {
                        start: rg_kv_start,
                        len: b.pf_lists[pfi].len() - rg_kv_start,
                        rpf: Some(vec![rpf]),
                        bat,
                        pack: pack.map(|p| vec![p]),
                        pack_lane: 0,
                    });
                }
                let rg_attn_start = b.pf_lists[pfi].len();
                if pf_ablated("attn") {
                } else if let Some(sb) = &steel_bufs {
                    let (kh, vh) = steel_mir.as_ref().unwrap();
                    push_attn_steel(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-steel"),
                        &pf.qkv_rot,
                        &kc,
                        &vc,
                        &step,
                        &pf.heads,
                        sb,
                        pfi,
                        kh,
                        vh,
                        &cfg,
                        pf.m,
                    )?;
                } else if fused_prefill {
                    push_attn_fused(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-fattn"),
                        &pf.qkv_rot,
                        &kc,
                        &vc,
                        &step,
                        &pf.heads,
                        &pf.zero,
                        fused_slabs.as_ref().unwrap(),
                        fused_mir.as_ref().unwrap(),
                        &cfg,
                        max_seq,
                        pf.m,
                        qkv_words,
                        q_words,
                    )?;
                } else if let Some(entry) = prefill_attn_entry {
                    push_attn_tiled(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-{entry}"),
                        entry,
                        &pf.qkv_rot,
                        &kc,
                        &vc,
                        &step,
                        &pf.heads,
                        &cfg,
                        max_seq,
                        pf.m,
                        qkv_words,
                        q_words,
                    )?;
                } else {
                    push_attn(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-attn"),
                        &pf.qkv_rot,
                        &kc,
                        &vc,
                        &step,
                        &pf.heads,
                        &cfg,
                        max_seq,
                        pf.m,
                        qkv_words,
                        q_words,
                        false,
                    )?;
                }
                if let Some((_, _, brange, _)) = &rg_bufs {
                    let rpf = if let Some(sb) = &steel_bufs {
                        let _ = RANGED_STEEL_PREFILL_READS_THE_REGION_INTO_LOCAL_MIRROR_ROWS;
                        let (kh, vh) = steel_mir.as_ref().unwrap();
                        let side_start = b.pf_lists[pfi].len();
                        push_attn_steel_kv(
                            &mut b,
                            &s,
                            &format!("dg-rg-l{li}-steelr-w{}", pf.m),
                            &pf.qkv_rot,
                            &kc,
                            &vc,
                            &step,
                            &pf.heads,
                            sb,
                            pfi,
                            kh,
                            vh,
                            &cfg,
                            pf.m,
                            "dg_scvt_kv_r",
                        )?;
                        b.pf_lists[pfi].drain(side_start..).collect()
                    } else {
                        vec![region_attn_pass(
                            &mut b,
                            &s,
                            &format!("dg-rg-l{li}-attnr-w{}", pf.m),
                            "dg_attn_r",
                            &pf.qkv_rot,
                            &kc,
                            &vc,
                            &step,
                            &pf.heads,
                            &cfg,
                            max_seq,
                            pf.m,
                            qkv_words,
                            q_words,
                            None,
                        )?]
                    };
                    let bat = if Some(pfi) == region_wi {
                        Some(region_attn_pass(
                            &mut b,
                            &s,
                            &format!("dg-rg-l{li}-attnb"),
                            "dg_attn_b",
                            &pf.qkv_rot,
                            &kc,
                            &vc,
                            &step,
                            &pf.heads,
                            &cfg,
                            max_seq,
                            pf.m,
                            qkv_words,
                            q_words,
                            Some(brange),
                        )?)
                    } else {
                        None
                    };
                    let (pack, pack_lane) = match (
                        pack_bufs.as_ref().filter(|_| Some(pfi) == pack_wi),
                        steel_bufs.as_ref(),
                    ) {
                        (Some((_, _, _, pk_steps, pk_pus)), Some(sb)) => {
                            let lwi = pack_lane_wi.expect("pack machinery implies a 128 chunk");
                            let (kh, vh) = steel_mir.as_ref().unwrap();
                            let lane = RPF_PACK_LANE_ROWS_ARE_THE_128_CHUNK_SO_EVERY_LANE_REPLAYS_TODAYS_STEEL_ATTENTION;
                            let bq = super::vision_attn_msl::STEEL_BQ as usize;
                            let q_words = q_rows / 2;
                            let kvw = kv_rows / 2;
                            let mut v: Vec<Pass> = Vec::new();
                            for l in 0..RPF_PACK_LANES_8_FILL_THE_1024_TRUNK {
                                v.push(b.make_pass(
                                    &format!("dg-pk-l{li}-qcvt{l}"),
                                    &s.scvt,
                                    "dg_scvt_q",
                                    &[(1, &pf.qkv_rot), (4, &pk_pus[l]), (5, &sb.qh)],
                                    ((q_words as u32).div_ceil(64), lane as u32, 1),
                                )?);
                                v.push(b.make_pass(
                                    &format!("dg-pk-l{li}-kvcvt{l}"),
                                    &s.scvt,
                                    "dg_scvt_kv_r",
                                    &[
                                        (0, &pk_steps[l]),
                                        (2, &kc),
                                        (3, &vc),
                                        (4, &sb.pu[lwi]),
                                        (6, kh),
                                        (7, vh),
                                    ],
                                    (((lane * kvw) as u32).div_ceil(64), 1, 1),
                                )?);
                                v.push(b.make_pass(
                                    &format!("dg-pk-l{li}-pp{l}"),
                                    &s.scvt,
                                    "dg_steel_pp",
                                    &[(0, &pk_steps[l]), (4, &sb.pu[lwi]), (10, &sb.sp[lwi])],
                                    (1, 1, 1),
                                )?);
                                let bind =
                                    b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some(&format!("dg-pk-l{li}-steel{l}")),
                                        layout: &sb.pls.layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: sb.qh.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: kh.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 2,
                                                resource: vh.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 3,
                                                resource: sb.oh.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 4,
                                                resource: sb.sp[lwi].as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 5,
                                                resource: sb.mp.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 6,
                                                resource: sb.dummy.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 7,
                                                resource: sb.dummy.as_entire_binding(),
                                            },
                                        ],
                                    });
                                v.push(Pass {
                                    pipeline: sb.pls.pipeline.clone(),
                                    bind,
                                    grid: ((lane / bq) as u32, n_q as u32, 1),
                                    entry: super::prefill_attn_msl::CAUSAL_ENTRY.to_string(),
                                });
                                v.push(b.make_pass(
                                    &format!("dg-pk-l{li}-ocvt{l}"),
                                    &s.scvt,
                                    "dg_scvt_o",
                                    &[(4, &pk_pus[l]), (8, &sb.oh), (9, &pf.heads)],
                                    (((lane * q_words) as u32).div_ceil(64), 1, 1),
                                )?);
                            }
                            (Some(v), RPF_PACK_ATTN_PASSES_PER_LANE)
                        }
                        (Some((_, _, pk_brange, _, _)), None) => (
                            Some(vec![region_attn_pass(
                                &mut b,
                                &s,
                                &format!("dg-pk-l{li}-attnb"),
                                "dg_attn_b",
                                &pf.qkv_rot,
                                &kc,
                                &vc,
                                &step,
                                &pf.heads,
                                &cfg,
                                max_seq,
                                pf.m,
                                qkv_words,
                                q_words,
                                Some(pk_brange),
                            )?]),
                            0,
                        ),
                        _ => (None, 0),
                    };
                    rg_sites[pfi].push(SwapSite {
                        start: rg_attn_start,
                        len: b.pf_lists[pfi].len() - rg_attn_start,
                        rpf: Some(rpf),
                        bat: bat.map(|p| vec![p]),
                        pack,
                        pack_lane,
                    });
                }
                if pf_ablated("proj") {
                } else if let Some(sp) = &steel_proj_bufs {
                    let wrote16 = push_proj_steel(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-osteel"),
                        ProjFamily::OProj,
                        ow_h.as_ref().unwrap(),
                        &pf.heads,
                        &pf.y_f32,
                        Some(&pf.attn_out),
                        sp,
                        pf.m,
                        hidden,
                        q_rows,
                    )?;
                    if !wrote16 {
                        push_cvt(
                            &mut b,
                            &s,
                            &format!("dg-pf-l{li}-ocvtout"),
                            &pf.y_f32,
                            &pf.attn_out,
                            hidden,
                            hidden,
                            hidden_words,
                            pf.m,
                            false,
                        )?;
                    }
                } else {
                    match ow_f32.as_ref().filter(|_| proj_wide(hidden, q_rows)) {
                        Some(wf) => {
                            push_cvt(
                                &mut b,
                                &s,
                                &format!("dg-pf-l{li}-ocvtin"),
                                &pf.heads,
                                &pf.x_f32,
                                q_rows,
                                q_words,
                                q_rows,
                                pf.m,
                                true,
                            )?;
                            push_coop_gemm(
                                &mut b,
                                &format!("dg-pf-l{li}-ocoop"),
                                wf,
                                &pf.x_f32,
                                &pf.y_f32,
                                &pf.zero,
                                pf.m,
                                hidden,
                                q_rows,
                            )?;
                            push_cvt(
                                &mut b,
                                &s,
                                &format!("dg-pf-l{li}-ocvtout"),
                                &pf.y_f32,
                                &pf.attn_out,
                                hidden,
                                hidden,
                                hidden_words,
                                pf.m,
                                false,
                            )?;
                        }
                        None => push_gemm_m8(
                            &mut b,
                            &s,
                            &format!("dg-pf-l{li}-oproj"),
                            &ow,
                            &pf.heads,
                            &pf.attn_out,
                            q_words,
                            hidden_words,
                            false,
                            pf.m,
                        )?,
                    }
                }
                if let Some(ll) = lora_layer.as_ref().filter(|_| !pf_ablated("proj")) {
                    push_dg_lora_slices(
                        &mut b,
                        &format!("dg-pf-l{li}-lora-o"),
                        ll,
                        &[("o_proj", 0)],
                        &pf.heads,
                        DgAct16::Bf16,
                        &pf.attn_out,
                        DgSink::Bf16,
                        hidden,
                        pf.m,
                    )?;
                }
                push_rmsnorm_residual(
                    &mut b,
                    &s,
                    &format!("dg-pf-l{li}-rmsres"),
                    &pf.attn_out,
                    &pf.res,
                    &postw,
                    &pf.normed_post,
                    hidden,
                    eps,
                    pf.m,
                )?;
                let coop_here = gu_f32.as_ref().filter(|_| {
                    pf.m >= PREFILL_M_WIDE
                        && coop_plan()
                            .check_shape(pf.m as u32, (2 * inter) as u32, hidden as u32)
                            .is_ok()
                });
                if !pf_ablated("proj") {
                    if let Some(sp) = &steel_proj_bufs {
                        let wrote16 = push_proj_steel(
                            &mut b,
                            &s,
                            &format!("dg-pf-l{li}-gusteel"),
                            ProjFamily::GateUp,
                            gu_h.as_ref().unwrap(),
                            &pf.normed_post,
                            &pf.y_f32,
                            Some(&pf.y_gu),
                            sp,
                            pf.m,
                            2 * inter,
                            hidden,
                        )?;
                        if !wrote16 {
                            push_cvt(
                                &mut b,
                                &s,
                                &format!("dg-pf-l{li}-gucvtout"),
                                &pf.y_f32,
                                &pf.y_gu,
                                2 * inter,
                                2 * inter,
                                inter,
                                pf.m,
                                false,
                            )?;
                        }
                    }
                }
                match coop_here {
                    _ if pf_ablated("proj") || steel_proj => {}
                    Some(wf) => {
                        push_cvt(
                            &mut b,
                            &s,
                            &format!("dg-pf-l{li}-gucvtin"),
                            &pf.normed_post,
                            &pf.x_f32,
                            hidden,
                            hidden_words,
                            hidden,
                            pf.m,
                            true,
                        )?;
                        push_coop_gemm(
                            &mut b,
                            &format!("dg-pf-l{li}-gucoop"),
                            wf,
                            &pf.x_f32,
                            &pf.y_f32,
                            &pf.zero,
                            pf.m,
                            2 * inter,
                            hidden,
                        )?;
                        push_cvt(
                            &mut b,
                            &s,
                            &format!("dg-pf-l{li}-gucvtout"),
                            &pf.y_f32,
                            &pf.y_gu,
                            2 * inter,
                            2 * inter,
                            inter,
                            pf.m,
                            false,
                        )?;
                    }
                    None => push_gemm_m8(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-gateup"),
                        &gu,
                        &pf.normed_post,
                        &pf.y_gu,
                        hidden_words,
                        inter,
                        false,
                        pf.m,
                    )?,
                }
                if let Some(ll) = lora_layer.as_ref().filter(|_| !pf_ablated("proj")) {
                    push_dg_lora_slices(
                        &mut b,
                        &format!("dg-pf-l{li}-lora-gu"),
                        ll,
                        &[("gate_proj", 0), ("up_proj", inter)],
                        &pf.normed_post,
                        DgAct16::Bf16,
                        &pf.y_gu,
                        DgSink::Bf16,
                        2 * inter,
                        pf.m,
                    )?;
                }
                push_silu_mul(
                    &mut b,
                    &s,
                    &format!("dg-pf-l{li}-silu"),
                    &pf.y_gu,
                    &pf.act,
                    inter,
                    pf.m,
                    inter,
                    inter / 2,
                )?;
                if pf_ablated("proj") {
                } else if let Some(sp) = &steel_proj_bufs {
                    let wrote16 = push_proj_steel(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-dsteel"),
                        ProjFamily::Down,
                        down_h.as_ref().unwrap(),
                        &pf.act,
                        &pf.x_f32,
                        Some(&pf.ffn_out),
                        sp,
                        pf.m,
                        hidden,
                        inter,
                    )?;
                    if !wrote16 {
                        push_cvt(
                            &mut b,
                            &s,
                            &format!("dg-pf-l{li}-dcvtout"),
                            &pf.x_f32,
                            &pf.ffn_out,
                            hidden,
                            hidden,
                            hidden_words,
                            pf.m,
                            false,
                        )?;
                    }
                } else {
                    match downw_f32.as_ref().filter(|_| proj_wide(hidden, inter)) {
                        Some(wf) => {
                            push_cvt(
                                &mut b,
                                &s,
                                &format!("dg-pf-l{li}-dcvtin"),
                                &pf.act,
                                &pf.y_f32,
                                inter,
                                inter / 2,
                                inter,
                                pf.m,
                                true,
                            )?;
                            push_coop_gemm(
                                &mut b,
                                &format!("dg-pf-l{li}-dcoop"),
                                wf,
                                &pf.y_f32,
                                &pf.x_f32,
                                &pf.zero,
                                pf.m,
                                hidden,
                                inter,
                            )?;
                            push_cvt(
                                &mut b,
                                &s,
                                &format!("dg-pf-l{li}-dcvtout"),
                                &pf.x_f32,
                                &pf.ffn_out,
                                hidden,
                                hidden,
                                hidden_words,
                                pf.m,
                                false,
                            )?;
                        }
                        None => push_gemm_m8(
                            &mut b,
                            &s,
                            &format!("dg-pf-l{li}-down"),
                            &down,
                            &pf.act,
                            &pf.ffn_out,
                            inter / 2,
                            hidden_words,
                            false,
                            pf.m,
                        )?,
                    }
                }
                if let Some(ll) = lora_layer.as_ref().filter(|_| !pf_ablated("proj")) {
                    push_dg_lora_slices(
                        &mut b,
                        &format!("dg-pf-l{li}-lora-down"),
                        ll,
                        &[("down_proj", 0)],
                        &pf.act,
                        DgAct16::Bf16,
                        &pf.ffn_out,
                        DgSink::Bf16,
                        hidden,
                        pf.m,
                    )?;
                }
                b.pf_target = None;
            }

            let next_name = if last {
                "model.norm.weight".to_string()
            } else {
                layer_tensor(li + 1, "input_layernorm.weight")
            };
            let nw = b.upload_u32(
                &format!("dg-l{li}-nextln"),
                &pack_pairs(&load_bf16(weights, &next_name, &[hidden])?),
            );
            let sink = if last { &final_x } else { &normed.x };
            if !fuse_qkv || last {
                push_rmsnorm_residual_fmt(
                    &mut b,
                    &s,
                    &format!("dg-l{li}-rmsnext"),
                    &ffn_out,
                    &res,
                    &nw,
                    sink,
                    hidden,
                    eps,
                    1,
                    act_f16,
                )?;
            }
            cur_ln = nw.clone();
            if let Some(snw) = &spec_normw {
                for (n, sx) in &spec_x {
                    if *n == li + 1 {
                        push_rmsnorm_fmt(
                            &mut b,
                            &s,
                            &format!("dg-spec-rms{n}"),
                            &res,
                            snw,
                            sx,
                            hidden,
                            eps,
                            1,
                            act_f16,
                        )?;
                    }
                }
            }

            for (pfi, pfb) in pf.iter().enumerate() {
                let pf = pfb;
                b.pf_target = Some(pfi);
                push_rmsnorm_residual(
                    &mut b,
                    &s,
                    &format!("dg-pf-l{li}-rmsnext"),
                    &pf.ffn_out,
                    &pf.res,
                    &nw,
                    &pf.normed,
                    hidden,
                    eps,
                    pf.m,
                )?;
                b.pf_target = None;
            }
        }

        let head_start = b.passes.len();
        let logits = b.zeros("dg-logits", (vocab * 4) as u64);
        let head_x = Act {
            x: final_x.clone(),
            k: hidden,
            xq: sl.sel0.clone(),
            xs: sl.sel0.clone(),
        };
        let lm_aff = lm_w4.map(|pw| upload_affine_w4(&mut b, "dg-lmhead", &pw, vocab, hidden));
        let lm_mma_h = (bhead_mma_enabled() && lm_aff.is_some())
            .then(|| upload_f16_of_bf16(&mut b, "dg-lmhead-h", &head_host));
        match &lm_aff {
            Some(aff) => {
                push_gemv_w4(
                    &mut b,
                    "dg-lmhead-w4",
                    aff,
                    &final_x,
                    &logits,
                    true,
                    act_f16,
                )?;
            }
            None => {
                for (i, (off, rows)) in splits.iter().copied().enumerate() {
                    let lin = Lin::Bf16 {
                        w: if tied {
                            embed_chunks[i].clone()
                        } else {
                            b.upload_u32(
                                "dg-lmhead",
                                &pack_pairs(&head_host[off * hidden..(off + rows) * hidden]),
                            )
                        },
                        n: rows,
                        k: hidden,
                    };
                    push_gemv(&mut b, &s, "dg-lmhead", &lin, &head_x, &logits, true, off)?;
                }
            }
        }
        drop(head_host);
        drop(embed_chunks);

        let pv = b.zeros("dg-am-pv", (ARGMAX_GROUPS * 4) as u64);
        let pi = b.zeros("dg-am-pi", (ARGMAX_GROUPS * 4) as u64);
        let token_out = b.zeros("dg-token", 4);
        let ap = b.uni(
            "dg-am-p",
            ArgmaxParams {
                n: vocab as u32,
                groups: ARGMAX_GROUPS as u32,
                ..Default::default()
            },
        );
        b.push(
            "dg-am1",
            &s.common,
            "lgw_argmax_stage1",
            &[(40, &logits), (41, &pv), (42, &pi), (44, &ap)],
            (ARGMAX_GROUPS as u32, 1, 1),
        )?;
        b.push(
            "dg-am2",
            &s.common,
            "lgw_argmax_stage2",
            &[(41, &pv), (42, &pi), (43, &token_out), (44, &ap)],
            (1, 1, 1),
        )?;

        let head_end_for_regions = b.passes.len();
        let region = match (region_wi, rg_bufs) {
            (Some(wi), Some((bpos, bslot, brange, btoks))) => {
                let _ = BATCHED_REGION_DECODE_SERVES_B_CROPS_PER_TRUNK_PASS;
                let slots = REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES;
                let words = hidden / 2;
                let mut gathers = Vec::with_capacity(slots);
                let mut stashes = Vec::with_capacity(slots);
                for row in 0..slots {
                    let gp = b.uni(
                        &format!("dg-rg-gp{row}"),
                        RowGatherParams {
                            words: words as u32,
                            src_slot_words: words as u32,
                            row: row as u32,
                            pad0: 0,
                        },
                    );
                    gathers.push(b.make_pass(
                        &format!("dg-rg-gather{row}"),
                        if act_f16 { &s.rope_act16 } else { &s.rope },
                        "dg_row_gather_b",
                        &[(90, &pf[wi].normed), (91, &gp), (92, &final_x)],
                        ((words as u32).div_ceil(64).max(1), 1, 1),
                    )?);
                    stashes.push(b.make_pass(
                        &format!("dg-rg-stash{row}"),
                        &s.rope,
                        "dg_tok_stash_b",
                        &[(91, &gp), (93, &token_out), (94, &btoks)],
                        (1, 1, 1),
                    )?);
                }
                let bhead = match (&lm_aff, act_f16 && hidden <= BHEAD_X_SLAB_ROW_HALFS) {
                    (Some(aff), true) => {
                        let _ = BHEAD_READS_THE_W4_HEAD_ONCE_PER_BSTEP_INSTEAD_OF_ONCE_PER_LIVE_ROW;
                        let pls = decode_gemv_msl::msl_gemv_pipelines(ctx)?;
                        let bx = b.zeros("dg-bh-x", (slots * hidden * 2) as u64);
                        let blogits = b.zeros("dg-bh-logits", (slots * vocab * 4) as u64);
                        let bpv = b.zeros("dg-bh-pv", (slots * ARGMAX_GROUPS * 4) as u64);
                        let bpi = b.zeros("dg-bh-pi", (slots * ARGMAX_GROUPS * 4) as u64);
                        let gather_params = BhGatherParams {
                            words: words as u32,
                            src_slot_words: words as u32,
                            b_rows: slots as u32,
                            pad0: 0,
                        };
                        let gather_p = b.uni("dg-bh-gp", gather_params);
                        let grid = b.grid1(vocab as u64, decode_gemv_msl::MSL_GEMV_ROWS);
                        let gemv_params = decode_gemv_msl::DgMslW4BParams {
                            n_rows: vocab as u32,
                            w_vec4: (hidden / 32) as u32,
                            scale_stride: (hidden / W4_GROUP as usize) as u32,
                            groups_x: grid.0,
                            b_rows: slots as u32,
                            x_stride: hidden as u32,
                            y_stride: vocab as u32,
                            pad0: 0,
                        };
                        let gemv_p = b.uni("dg-bh-p", gemv_params);
                        let am_params = BhAmParams {
                            n: vocab as u32,
                            groups: ARGMAX_GROUPS as u32,
                            b_rows: slots as u32,
                            pad0: 0,
                        };
                        let am_p = b.uni("dg-bh-amp", am_params);
                        let bh_src = compose(DG_BHEAD_WGSL);
                        let gather = b.make_pass(
                            "dg-bh-gather",
                            &bh_src,
                            "dg_bh_gather",
                            &[(90, &pf[wi].normed), (91, &gather_p), (92, &bx)],
                            ((words as u32).div_ceil(64).max(1), slots as u32, 1),
                        )?;
                        let gemv_bind =
                            b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                label: Some("dg-bh-gemv"),
                                layout: &pls.layout,
                                entries: &[
                                    wgpu::BindGroupEntry {
                                        binding: 0,
                                        resource: aff.w.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 1,
                                        resource: aff.scales.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 2,
                                        resource: aff.biases.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 3,
                                        resource: bx.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 4,
                                        resource: gemv_p.as_entire_binding(),
                                    },
                                    wgpu::BindGroupEntry {
                                        binding: 5,
                                        resource: blogits.as_entire_binding(),
                                    },
                                ],
                            });
                        let gemv = Pass {
                            pipeline: pls.batch_pipeline.clone(),
                            bind: gemv_bind,
                            grid: (grid.0, grid.1, 1),
                            entry: decode_gemv_msl::MSL_GEMV_BATCH_ENTRY.to_string(),
                        };
                        let am1 = b.make_pass(
                            "dg-bh-am1",
                            &bh_src,
                            "dg_bh_am1",
                            &[(40, &blogits), (41, &bpv), (42, &bpi), (44, &am_p)],
                            (ARGMAX_GROUPS as u32, slots as u32, 1),
                        )?;
                        let am2 = b.make_pass(
                            "dg-bh-am2",
                            &bh_src,
                            "dg_bh_am2",
                            &[(41, &bpv), (42, &bpi), (43, &btoks), (44, &am_p)],
                            (slots as u32, 1, 1),
                        )?;
                        let mma = match &lm_mma_h {
                            Some(wh) => {
                                let cfg = BHEAD_MMA_CFG_IS_32_32_16_BECAUSE_THE_HEAD_AT_M32_IS_B_READ_BOUND_AND_THE_BRACKET_CROWNED_IT;
                                let spls = super::vision_attn_msl::steel_gemm_pipelines(ctx)?;
                                let pipeline = spls.pick(cfg).ok_or_else(|| {
                                    anyhow::anyhow!("bhead mma: steel cfg {cfg:?} missing")
                                })?;
                                let mma_params = super::vision_attn_msl::steel_gemm_params(
                                    slots, vocab, hidden, vocab, cfg,
                                );
                                let mma_p = b.uni("dg-bh-mmap", mma_params);
                                let bind =
                                    b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                                        label: Some("dg-bh-mma"),
                                        layout: &spls.layout,
                                        entries: &[
                                            wgpu::BindGroupEntry {
                                                binding: 0,
                                                resource: bx.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 1,
                                                resource: wh.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 2,
                                                resource: blogits.as_entire_binding(),
                                            },
                                            wgpu::BindGroupEntry {
                                                binding: 3,
                                                resource: mma_p.as_entire_binding(),
                                            },
                                        ],
                                    });
                                let pass = Pass {
                                    pipeline,
                                    bind,
                                    grid: super::vision_attn_msl::steel_gemm_grid(
                                        slots, vocab, cfg,
                                    ),
                                    entry: super::vision_attn_msl::steel_gemm_entry(
                                        cfg.0, cfg.1, cfg.2, cfg.3, cfg.4,
                                    ),
                                };
                                Some((pass, mma_p, mma_params))
                            }
                            None => None,
                        };
                        Some(BheadMachinery {
                            gather,
                            gemv,
                            am1,
                            am2,
                            gather_p,
                            gemv_p,
                            am_p,
                            gather_params,
                            gemv_params,
                            am_params,
                            blogits,
                            mma,
                        })
                    }
                    _ => None,
                };
                Some(RegionMachinery {
                    wi,
                    stride: max_seq / slots,
                    sites: std::mem::take(&mut rg_sites),
                    head_range: (head_start, head_end_for_regions),
                    gathers,
                    stashes,
                    bpos,
                    bslot,
                    brange,
                    btoks,
                    bhead,
                    validated: false,
                    bat_validated: false,
                })
            }
            _ => None,
        };
        let pack = match (pack_wi, pack_bufs) {
            (Some(wi), Some((bpos, bslot, brange, steps, _))) => Some(PackMachinery {
                wi,
                bpos,
                bslot,
                brange,
                lane_step: steps,
                validated: false,
            }),
            _ => None,
        };

        let mut spec_tok: Vec<(usize, wgpu::Buffer)> = Vec::new();
        if let Some(aff) = lm_aff.as_ref().filter(|_| !spec_x.is_empty()) {
            let spec_logits = b.zeros("dg-spec-logits", (vocab * 4) as u64);
            for (n, sx) in &spec_x {
                push_gemv_w4(
                    &mut b,
                    &format!("dg-spec-head{n}"),
                    aff,
                    sx,
                    &spec_logits,
                    true,
                    act_f16,
                )?;
                let tok = b.zeros(&format!("dg-spec-tok{n}"), 4);
                b.push(
                    &format!("dg-spec-am1-{n}"),
                    &s.common,
                    "lgw_argmax_stage1",
                    &[(40, &spec_logits), (41, &pv), (42, &pi), (44, &ap)],
                    (ARGMAX_GROUPS as u32, 1, 1),
                )?;
                b.push(
                    &format!("dg-spec-am2-{n}"),
                    &s.common,
                    "lgw_argmax_stage2",
                    &[(41, &pv), (42, &pi), (43, &tok), (44, &ap)],
                    (1, 1, 1),
                )?;
                spec_tok.push((*n, tok));
            }
        }

        b.flush_staging();
        let vram = b.report();
        if crate::wgpu_ledger::vram_report_var_enabled(DOTS_GRAPH_VRAM_ENV) {
            eprint!("[dots-graph] {}", vram.render());
        }

        let Builder {
            core,
            passes,
            pf_lists: pf_passes,
            split: split_passes,
            state,
            sk_families,
            bf16out_families,
            abf16_families,
            brope_kvw_layers,
            ..
        } = b;
        Ok(Self {
            ctx,
            cfg,
            quant,
            max_seq,
            pos: 0,
            repack_s: 0.0,
            rpf_repack_s: 0.0,
            bstep_cpu_s: 0.0,
            bstep_read_s: 0.0,
            bstep_count: 0,
            bstep_bhead: 0,
            bstep_bhead_mma: 0,
            trunk_splitk_families: sk_families,
            trunk_bf16out_families: bf16out_families,
            trunk_abf16_families: abf16_families,
            brope_kvw_fused_layers: brope_kvw_layers,
            battn_v2: battn_v2_enabled(),
            sg_tree,
            rpf_chunk_count: 0,
            step_submits: 0,
            ra_gpu_fed: 0,
            validated: false,
            prefix_validated: false,
            pf_validated: false,
            passes,
            pf_passes,
            split_passes,
            attn_idx,
            pf,
            pf_widths,
            embed_end,
            head_start,
            res,
            _buffers: core.buffers,
            step,
            token_out,
            spec_tok,
            logits,
            state_buffers: state,
            vocab,
            vram,
            region,
            pack,
            rpf_packs: 0,
            rpf_pack_lanes: 0,
            pf_gpu: None,
            handoff_copies: 0,
        })
    }

    pub fn config(&self) -> &DotsDecoderConfig {
        &self.cfg
    }

    pub fn quant(&self) -> DotsGraphQuant {
        self.quant
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn prefill_pass_count(&self) -> usize {
        self.pf_passes.len()
    }

    pub fn attn_split_passes(&self) -> usize {
        self.split_passes.len()
    }

    pub fn runahead_gpu_fed_steps(&self) -> u64 {
        self.ra_gpu_fed
    }

    pub fn bstep_bhead_steps(&self) -> u64 {
        self.bstep_bhead
    }

    pub fn bstep_bhead_mma_steps(&self) -> u64 {
        self.bstep_bhead_mma
    }

    pub fn pf_chunk_widths(&self) -> &[usize] {
        &self.pf_widths
    }

    pub fn trunk_bf16out_families(&self) -> u64 {
        self.trunk_bf16out_families
    }

    pub fn trunk_abf16_families(&self) -> u64 {
        self.trunk_abf16_families
    }

    pub fn brope_kvw_fused_layers(&self) -> u64 {
        self.brope_kvw_fused_layers
    }

    pub fn trunk_splitk_families(&self) -> u64 {
        self.trunk_splitk_families
    }

    pub fn battn_v2(&self) -> bool {
        self.battn_v2
    }

    pub fn sg_tree(&self) -> SgTreeSites {
        self.sg_tree
    }

    fn write_bhead_uniforms(&self, bh: &BheadMachinery, b_rows: u32) {
        self.ctx.queue.write_buffer(
            &bh.gather_p,
            0,
            bytemuck::bytes_of(&BhGatherParams {
                b_rows,
                ..bh.gather_params
            }),
        );
        self.ctx.queue.write_buffer(
            &bh.gemv_p,
            0,
            bytemuck::bytes_of(&decode_gemv_msl::DgMslW4BParams {
                b_rows,
                ..bh.gemv_params
            }),
        );
        self.ctx.queue.write_buffer(
            &bh.am_p,
            0,
            bytemuck::bytes_of(&BhAmParams {
                b_rows,
                ..bh.am_params
            }),
        );
        if let Some((_, mp, base)) = &bh.mma {
            self.ctx.queue.write_buffer(
                mp,
                0,
                bytemuck::bytes_of(&super::vision_attn_msl::SteelGemmParams {
                    m: b_rows as i32,
                    ..*base
                }),
            );
        }
    }

    pub fn bstep_head_probe(
        &mut self,
        rows: &[RegionRow],
        reps: usize,
    ) -> Result<Option<(f64, f64)>> {
        if self.region.as_ref().is_none_or(|rg| rg.bhead.is_none()) {
            return Ok(None);
        }
        let rg = self.region.as_ref().expect("region machinery");
        let (hs, he) = rg.head_range;
        let med = |run: &[&Pass]| -> Result<f64> {
            let mut ms: Vec<f64> = Vec::with_capacity(reps);
            for _ in 0..reps {
                let t = std::time::Instant::now();
                let cb = dispatch::encode_pass_list_labeled(
                    self.ctx,
                    run.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
                    &[],
                );
                self.ctx.queue.submit([cb]);
                let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
                ms.push(t.elapsed().as_secs_f64() * 1e3);
            }
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            Ok(ms[ms.len() / 2])
        };
        let mut replay_run: Vec<&Pass> = Vec::new();
        for r in 0..rows.len() {
            replay_run.push(&rg.gathers[r]);
            for p in &self.passes[hs..he] {
                replay_run.push(p);
            }
            replay_run.push(&rg.stashes[r]);
        }
        let replay_ms = med(&replay_run)?;
        let bh = rg.bhead.as_ref().expect("bhead machinery");
        self.write_bhead_uniforms(bh, rows.len() as u32);
        let batched_run = [&bh.gather, bh.head_pass().0, &bh.am1, &bh.am2];
        let batched_ms = med(&batched_run)?;
        Ok(Some((replay_ms, batched_ms)))
    }

    pub fn bstep_head_census(
        &mut self,
        rows: &[RegionRow],
    ) -> Result<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<u32>, Vec<u32>)>> {
        if self.region.as_ref().is_none_or(|rg| rg.bhead.is_none()) {
            return Ok(None);
        }
        let replay_toks = {
            let prev = std::env::var(BHEAD_ENV).ok();
            std::env::set_var(BHEAD_ENV, "0");
            let r = self.bstep_tokens(rows);
            match prev {
                Some(v) => std::env::set_var(BHEAD_ENV, v),
                None => std::env::remove_var(BHEAD_ENV),
            }
            r?
        };
        let vocab = self.vocab;
        let rg = self.region.as_ref().expect("region machinery");
        let (hs, he) = rg.head_range;
        let mut replay_logits: Vec<Vec<f32>> = Vec::with_capacity(rows.len());
        for r in 0..rows.len() {
            let mut run: Vec<&Pass> = vec![&rg.gathers[r]];
            for p in &self.passes[hs..he] {
                run.push(p);
            }
            let cb = dispatch::encode_pass_list_labeled(
                self.ctx,
                run.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
                &[],
            );
            self.ctx.queue.submit([cb]);
            let l: Vec<f32> = dispatch::read_back(self.ctx, &self.logits, vocab)
                .map_err(|e| anyhow::anyhow!("census replay logits: {e}"))?;
            replay_logits.push(l);
        }
        let bh = rg.bhead.as_ref().expect("bhead machinery");
        self.write_bhead_uniforms(bh, rows.len() as u32);
        let run = [&bh.gather, bh.head_pass().0, &bh.am1, &bh.am2];
        let cb = dispatch::encode_pass_list_labeled(
            self.ctx,
            run.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
            &[],
        );
        self.ctx.queue.submit([cb]);
        let flat: Vec<f32> = dispatch::read_back(self.ctx, &bh.blogits, rows.len() * vocab)
            .map_err(|e| anyhow::anyhow!("census batched logits: {e}"))?;
        let batched_logits: Vec<Vec<f32>> = flat.chunks(vocab).map(|c| c.to_vec()).collect();
        let m = self.pf_widths[rg.wi];
        let bt: Vec<u32> = dispatch::read_back(self.ctx, &rg.btoks, m)
            .map_err(|e| anyhow::anyhow!("census batched btoks: {e}"))?;
        Ok(Some((
            replay_logits,
            batched_logits,
            replay_toks,
            bt[..rows.len()].to_vec(),
        )))
    }

    pub fn handoff_gpu_copies(&self) -> u64 {
        self.handoff_copies
    }

    pub fn take_region_repack_s(&mut self) -> f64 {
        std::mem::take(&mut self.rpf_repack_s)
    }

    pub fn drain(&self) {
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    pub fn rpf_chunks(&self) -> u64 {
        self.rpf_chunk_count
    }

    pub fn pf_steel_gemm_entries(&self) -> Vec<(usize, Vec<String>)> {
        self.pf_widths
            .iter()
            .zip(&self.pf_passes)
            .map(|(m, ps)| {
                (
                    *m,
                    ps.iter()
                        .filter(|p| p.entry.starts_with("dv_steel_gemm_nt"))
                        .map(|p| p.entry.clone())
                        .collect(),
                )
            })
            .collect()
    }

    pub fn take_bstep_split(&mut self) -> (f64, f64, u64) {
        (
            std::mem::take(&mut self.bstep_cpu_s),
            std::mem::take(&mut self.bstep_read_s),
            std::mem::take(&mut self.bstep_count),
        )
    }

    pub fn step_submits(&self) -> u32 {
        self.step_submits
    }

    pub fn step_issue_count(&self) -> usize {
        self.step_pass_list(0, self.passes.len()).len()
    }

    pub fn decode_pass_census(&self) -> Vec<(&'static str, usize, String)> {
        let mut rows: Vec<(&'static str, usize, String)> = Vec::new();
        let mut gemv_ord = 0usize;
        let mut k = 0usize;
        let mut note =
            |class: &'static str, entry: &str| match rows.iter_mut().find(|r| r.0 == class) {
                Some(r) => {
                    r.1 += 1;
                    if !r.2.contains(entry) {
                        r.2.push('+');
                        r.2.push_str(entry);
                    }
                }
                None => rows.push((class, 1, entry.to_string())),
            };
        for (i, p) in self.passes.iter().enumerate() {
            let is_head = i >= self.head_start;
            let is_attn = k < self.attn_idx.len() && self.attn_idx[k] == i;
            if is_attn {
                k += 1;
                if !self.split_passes.is_empty() {
                    for sp in &self.split_passes[2 * (k - 1)..2 * k] {
                        note(dec_pass_class(&sp.entry, is_head, gemv_ord), &sp.entry);
                    }
                    continue;
                }
            }
            note(dec_pass_class(&p.entry, is_head, gemv_ord), &p.entry);
            if !is_head && p.entry.contains("gemv") {
                gemv_ord += 1;
            }
        }
        rows
    }

    pub fn prefill_chunk_len(&self) -> usize {
        self.pf_widths.last().copied().unwrap_or(0)
    }

    pub fn prefill_widths(&self) -> &[usize] {
        &self.pf_widths
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn max_seq_tokens(&self) -> usize {
        self.max_seq
    }

    pub fn vram_report(&self) -> &VramReport {
        &self.vram
    }

    pub fn kv_cache_raw(&self) -> Result<Vec<(Vec<u32>, Vec<u32>)>> {
        anyhow::ensure!(
            self.state_buffers.len() == 2 * self.cfg.num_hidden_layers,
            "the state buffers are exactly the per-layer kc/vc pairs; a new state buffer \
             class must teach this readback its layout before the kv census can trust it"
        );
        self.state_buffers
            .chunks(2)
            .map(|pair| {
                let k = dispatch::read_back(self.ctx, &pair[0], (pair[0].size() / 4) as usize)
                    .map_err(|e| anyhow::anyhow!("k cache readback: {e}"))?;
                let v = dispatch::read_back(self.ctx, &pair[1], (pair[1].size() / 4) as usize)
                    .map_err(|e| anyhow::anyhow!("v cache readback: {e}"))?;
                Ok((k, v))
            })
            .collect()
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        self.pf_gpu = None;
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for buf in &self.state_buffers {
            enc.clear_buffer(buf, 0, None);
        }
        self.ctx.queue.submit([enc.finish()]);
        Ok(())
    }

    fn step_pass_list(&self, start: usize, end: usize) -> Vec<(&Pass, (u32, u32, u32))> {
        let split = !self.split_passes.is_empty();
        let live_slices = self.split_passes.first().map(|slice| {
            let total = (self.pos + 1) as u32;
            let cap = slice
                .grid
                .1
                .min(ATTN_SPLIT_LIVE_SLICE_CAP_28_MUST_EQUAL_THE_WGSL_DG_SPLIT_LIVE_SLICE_CAP);
            total.div_ceil(total.div_ceil(cap))
        });
        let mut run: Vec<(&Pass, (u32, u32, u32))> =
            Vec::with_capacity(end - start + self.split_passes.len());
        let mut k = 0usize;
        for i in start..end {
            let is_attn = k < self.attn_idx.len() && self.attn_idx[k] == i;
            if is_attn {
                k += 1;
                if split {
                    let pair = &self.split_passes[2 * (k - 1)..2 * k];
                    let mut slice_grid = pair[0].grid;
                    slice_grid.1 = live_slices.unwrap_or(slice_grid.1);
                    run.push((&pair[0], slice_grid));
                    run.push((&pair[1], pair[1].grid));
                    continue;
                }
            }
            run.push((&self.passes[i], self.passes[i].grid));
        }
        if dec_grid1() {
            for r in run.iter_mut() {
                r.1 = (1, 1, 1);
            }
        }
        if let Some(want) = dec_ablate() {
            let head_tail = end.saturating_sub(self.head_start.min(end));
            let body = run.len() - head_tail;
            let mut gemv_ord = 0usize;
            let mut kept = Vec::with_capacity(run.len());
            for (j, r) in run.into_iter().enumerate() {
                let entry = r.0.entry.as_str();
                let is_head = j >= body;
                let class = dec_pass_class(entry, is_head, gemv_ord);
                if !is_head && entry.contains("gemv") {
                    gemv_ord += 1;
                }
                if !dec_ablate_drops(class, &want) {
                    kept.push(r);
                }
            }
            run = kept;
        }
        run
    }

    pub fn boundary_probe(&mut self, arm: &str, reps: usize) -> Result<Vec<BoundaryRep>> {
        let res_words = (self.res.size() / 4) as usize;
        let res_snap: Vec<u32> = dispatch::read_back(self.ctx, &self.res, res_words)
            .map_err(|e| anyhow::anyhow!("res snapshot: {e}"))?;
        self.step
            .write(self.ctx, 100, self.pos as u32, (self.pos + 1) as u32);
        let full = self.step_pass_list(0, self.passes.len());
        let n = full.len();
        let rms = self
            .passes
            .iter()
            .find(|p| p.entry == "dg_rms" || p.entry == "rmsnorm_bf16")
            .ok_or_else(|| anyhow::anyhow!("no rms pass anchors the same-pipeline arms"))?;
        let silu = self
            .passes
            .iter()
            .find(|p| p.entry == "dg_silu_mul")
            .ok_or_else(|| anyhow::anyhow!("no silu pass anchors the wide-grid arm"))?;
        let hidden = self.cfg.hidden_size;
        let disjoint: Vec<wgpu::BindGroup> = if arm == "disjoint" {
            let w = dispatch::storage_zeroed(self.ctx, "bp-disjoint-w", (hidden / 2) as u64 * 4);
            let p = dispatch::uniform_from(
                self.ctx,
                "bp-disjoint-p",
                &RmsParams {
                    hidden: hidden as u32,
                    batch: 1,
                    eps: self.cfg.rms_norm_eps as f32,
                    words_per_row: (hidden / 2) as u32,
                },
            );
            (0..n)
                .map(|i| {
                    let x = dispatch::storage_zeroed(
                        self.ctx,
                        &format!("bp-disjoint-x{i}"),
                        (hidden / 2) as u64 * 4,
                    );
                    let y = dispatch::storage_zeroed(
                        self.ctx,
                        &format!("bp-disjoint-y{i}"),
                        (hidden / 2) as u64 * 4,
                    );
                    let binds: Vec<(u32, &wgpu::Buffer)> = if rms.entry == "dg_rms" {
                        vec![(0, &x), (2, &w), (3, &y), (4, &p)]
                    } else {
                        vec![(0, &x), (1, &w), (2, &y), (3, &p)]
                    };
                    dispatch::bind_group(self.ctx, &rms.pipeline, &binds)
                })
                .collect()
        } else {
            Vec::new()
        };
        let mut out = Vec::with_capacity(reps);
        for _ in 0..reps {
            self.ctx
                .queue
                .write_buffer(&self.res, 0, bytemuck::cast_slice(&res_snap));
            let list: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))> = match arm {
                "samepipe" => (0..n)
                    .map(|_| (&*rms.pipeline, &rms.bind, rms.grid))
                    .collect(),
                "samepipe_silu" => (0..n)
                    .map(|_| (&*silu.pipeline, &silu.bind, silu.grid))
                    .collect(),
                "samepipe_silu1" => (0..n)
                    .map(|_| (&*silu.pipeline, &silu.bind, (1u32, 1u32, 1u32)))
                    .collect(),
                "samepipe_gemv1" => {
                    let g = self
                        .passes
                        .iter()
                        .find(|p| p.entry.contains("gemv"))
                        .ok_or_else(|| anyhow::anyhow!("no gemv pass for samepipe_gemv1"))?;
                    (0..n)
                        .map(|_| (&*g.pipeline, &g.bind, (1u32, 1u32, 1u32)))
                        .collect()
                }
                "samepipe_attn1" | "samepipe_comb1" => {
                    let i = usize::from(arm == "samepipe_comb1");
                    let a = self
                        .split_passes
                        .get(i)
                        .ok_or_else(|| anyhow::anyhow!("no split pass for {arm}"))?;
                    (0..n)
                        .map(|_| (&*a.pipeline, &a.bind, (1u32, 1u32, 1u32)))
                        .collect()
                }
                "altpipe" => (0..n)
                    .map(|i| {
                        let p = if i % 2 == 0 { rms } else { silu };
                        (&*p.pipeline, &p.bind, p.grid)
                    })
                    .collect(),
                "disjoint" => (0..n)
                    .map(|i| (&*rms.pipeline, &disjoint[i], rms.grid))
                    .collect(),
                "nullsubmit" => Vec::new(),
                _ => full
                    .iter()
                    .map(|(p, g)| (&*p.pipeline, &p.bind, *g))
                    .collect(),
            };
            let chunk = match arm {
                "persubmit" => 1usize,
                "chunk16" => 16,
                "chunk32" => 32,
                "chunk64" => 64,
                "chunk128" => 128,
                _ => list.len().max(1),
            };
            let per_submit = arm == "persubmit" || arm.starts_with("chunk");
            let one_pass_per_chunk = arm != "perpass";
            let t0 = std::time::Instant::now();
            let mut encode_s = 0.0f64;
            let mut submit_s = 0.0f64;
            let mut pending: Vec<wgpu::CommandBuffer> = Vec::new();
            for group in list.chunks(chunk.max(1)) {
                let te = std::time::Instant::now();
                let mut enc = self.ctx.device.create_command_encoder(&Default::default());
                if one_pass_per_chunk {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    for (pipeline, bind, wg) in group {
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, *bind, &[]);
                        pass.dispatch_workgroups(wg.0, wg.1, wg.2);
                    }
                } else {
                    for (pipeline, bind, wg) in group {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        pass.set_pipeline(pipeline);
                        pass.set_bind_group(0, *bind, &[]);
                        pass.dispatch_workgroups(wg.0, wg.1, wg.2);
                    }
                }
                let cb = enc.finish();
                encode_s += te.elapsed().as_secs_f64();
                if arm == "encode_drop" {
                    drop(cb);
                    continue;
                }
                if per_submit {
                    let ts = std::time::Instant::now();
                    self.ctx.queue.submit([cb]);
                    submit_s += ts.elapsed().as_secs_f64();
                } else {
                    pending.push(cb);
                }
            }
            if arm == "nullsubmit" {
                let te = std::time::Instant::now();
                let enc = self.ctx.device.create_command_encoder(&Default::default());
                let cb = enc.finish();
                encode_s += te.elapsed().as_secs_f64();
                pending.push(cb);
            }
            if !pending.is_empty() {
                let ts = std::time::Instant::now();
                self.ctx.queue.submit(pending);
                submit_s += ts.elapsed().as_secs_f64();
            }
            let tw = std::time::Instant::now();
            if arm != "encode_drop" {
                let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
            }
            let wait_s = tw.elapsed().as_secs_f64();
            out.push(BoundaryRep {
                total_ms: t0.elapsed().as_secs_f64() * 1e3,
                encode_ms: encode_s * 1e3,
                submit_ms: submit_s * 1e3,
                wait_ms: wait_s * 1e3,
                passes: if arm == "nullsubmit" { 0 } else { n },
            });
        }
        Ok(out)
    }

    fn run(&mut self, token: u32, row: Option<&[f32]>, full: bool) -> Result<()> {
        anyhow::ensure!(
            self.pos < self.max_seq,
            "kv cache full at {} (max_seq {})",
            self.pos,
            self.max_seq
        );
        let start = match row {
            None => {
                anyhow::ensure!((token as usize) < self.vocab, "token {token} out of vocab");
                0
            }
            Some(row) => {
                anyhow::ensure!(
                    row.len() == self.cfg.hidden_size,
                    "supplied hidden row has {} values, want {}",
                    row.len(),
                    self.cfg.hidden_size
                );
                let bits: Vec<u16> = row
                    .iter()
                    .map(|v| half::bf16::from_f32(*v).to_bits())
                    .collect();
                self.ctx
                    .queue
                    .write_buffer(&self.res, 0, bytemuck::cast_slice(&pack_pairs(&bits)));
                self.embed_end
            }
        };
        self.step
            .write(self.ctx, token, self.pos as u32, (self.pos + 1) as u32);

        let need_scope = if full {
            !self.validated
        } else {
            !self.validated && !self.prefix_validated
        };
        let scope = need_scope.then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        let end = if full {
            self.passes.len()
        } else {
            self.head_start
        };
        let run = self.step_pass_list(start, end);
        let chunk = dec_submit_chunk();
        let mut submits = 0u32;
        if chunk > 0 && chunk < run.len() && !dispatch::profile::enabled() {
            for group in run.chunks(chunk) {
                let cb = dispatch::encode_pass_list_labeled(
                    self.ctx,
                    group.iter().map(|(p, g)| (&*p.pipeline, &p.bind, *g)),
                    &[],
                );
                self.ctx.queue.submit([cb]);
                submits += 1;
            }
        } else {
            let labels: Vec<&str> = if dispatch::profile::enabled() {
                run.iter().map(|(p, _)| p.entry.as_str()).collect()
            } else {
                Vec::new()
            };
            let cb = dispatch::encode_pass_list_labeled(
                self.ctx,
                run.iter().map(|(p, g)| (&*p.pipeline, &p.bind, *g)),
                &labels,
            );
            self.ctx.queue.submit([cb]);
            submits = 1;
        }
        drop(run);
        self.step_submits = submits;
        if let Some(scope) = scope {
            if let Some(e) = dispatch::pop_scope(scope) {
                anyhow::bail!("dots decoder graph step validation: {e}");
            }
            if full {
                self.validated = true;
            }
            self.prefix_validated = true;
        }
        self.pos += 1;
        Ok(())
    }

    fn step_inner(&mut self, token: u32, full: bool) -> Result<()> {
        self.run(token, None, full)
    }

    fn step_ahead(
        &mut self,
        token: Option<u32>,
        stage: &wgpu::Buffer,
    ) -> Result<wgpu::SubmissionIndex> {
        let _ = DEC_RUNAHEAD_IS_EXACT_BECAUSE_THE_BAN_LIST_IS_KNOWN_BEFORE_THE_TOKEN;
        anyhow::ensure!(
            self.pos < self.max_seq,
            "kv cache full at {} (max_seq {})",
            self.pos,
            self.max_seq
        );
        anyhow::ensure!(
            self.validated,
            "run-ahead needs the validated full-step graph"
        );
        match token {
            Some(t) => {
                anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
                self.step
                    .write(self.ctx, t, self.pos as u32, (self.pos + 1) as u32);
            }
            None => {
                self.step
                    .write(self.ctx, 0, self.pos as u32, (self.pos + 1) as u32);
                let mut enc = self.ctx.device.create_command_encoder(&Default::default());
                enc.copy_buffer_to_buffer(&self.token_out, 0, &self.step.tok, 0, 4);
                self.ctx.queue.submit([enc.finish()]);
                self.ra_gpu_fed += 1;
            }
        }
        let run = self.step_pass_list(0, self.passes.len());
        let chunk = match dec_submit_chunk() {
            0 => run.len(),
            c => c,
        };
        for group in run.chunks(chunk) {
            let cb = dispatch::encode_pass_list_labeled(
                self.ctx,
                group.iter().map(|(p, g)| (&*p.pipeline, &p.bind, *g)),
                &[],
            );
            self.ctx.queue.submit([cb]);
        }
        drop(run);
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&self.token_out, 0, stage, 0, 4);
        let idx = self.ctx.queue.submit([enc.finish()]);
        self.pos += 1;
        Ok(idx)
    }

    fn await_token(&self, idx: wgpu::SubmissionIndex, stage: &wgpu::Buffer) -> Result<u32> {
        let slice = stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.ctx
            .device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: None,
            })
            .map_err(|e| anyhow::anyhow!("run-ahead token poll: {e}"))?;
        rx.recv()
            .map_err(|e| anyhow::anyhow!("run-ahead map callback: {e}"))?
            .map_err(|e| anyhow::anyhow!("run-ahead map: {e}"))?;
        let t = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("run-ahead mapped range: {e}"))?;
            u32::from_le_bytes(view[0..4].try_into().expect("4-byte token stage"))
        };
        stage.unmap();
        Ok(t)
    }

    pub fn read_logits(&self) -> Result<Vec<f32>> {
        dispatch::drive_blocking(self.read_logits_arm(dispatch::ReadbackArm::Blocking))
    }

    async fn read_logits_arm(&self, arm: dispatch::ReadbackArm) -> Result<Vec<f32>> {
        dispatch::read_back_arm(self.ctx, &self.logits, self.vocab, arm)
            .await
            .map_err(|e| anyhow::anyhow!("logit readback: {e}"))
    }

    fn read_token_and_logits(&self) -> Result<(u32, Vec<f32>)> {
        dispatch::drive_blocking(self.read_token_and_logits_arm(dispatch::ReadbackArm::Blocking))
    }

    async fn read_token_and_logits_arm(
        &self,
        arm: dispatch::ReadbackArm,
    ) -> Result<(u32, Vec<f32>)> {
        let t: Vec<u32> = dispatch::read_back_arm(self.ctx, &self.token_out, 1, arm)
            .await
            .map_err(|e| anyhow::anyhow!("token readback: {e}"))?;
        let l = self.read_logits_arm(arm).await?;
        Ok((t[0], l))
    }

    async fn read_token_arm(&self, arm: dispatch::ReadbackArm) -> Result<u32> {
        let t: Vec<u32> = dispatch::read_back_arm(self.ctx, &self.token_out, 1, arm)
            .await
            .map_err(|e| anyhow::anyhow!("token readback: {e}"))?;
        Ok(t[0])
    }

    async fn gpu_row_host_arm(
        &self,
        local_row: usize,
        arm: dispatch::ReadbackArm,
    ) -> Result<Vec<f32>> {
        let g = self
            .pf_gpu
            .as_ref()
            .expect("a gpu row fetch happens only under a gpu vision source");
        anyhow::ensure!(
            local_row < g.n_vis,
            "gpu row {local_row} out of {} rows",
            g.n_vis
        );
        let words = self.cfg.hidden_size / 2;
        let w: Vec<u32> =
            dispatch::read_back_at_arm(self.ctx, &g.buf, local_row * words, words, arm)
                .await
                .map_err(|e| anyhow::anyhow!("gpu vision row readback: {e}"))?;
        Ok(unpack_bf16_pair_words(&w))
    }

    fn prefill_chunk(&mut self, wi: usize, chunk: &[(u32, Option<&[f32]>)]) -> Result<()> {
        self.prefill_chunk_real(wi, chunk, chunk.len())
    }

    fn prefill_chunk_real(
        &mut self,
        wi: usize,
        chunk: &[(u32, Option<&[f32]>)],
        real: usize,
    ) -> Result<()> {
        let _ = PAD_ROWS_COST_NOTHING_BECAUSE_CAUSALITY_SHIELDS_THEM_AND_DECODE_OVERWRITES_THEM;
        anyhow::ensure!(
            real >= 1 && real <= chunk.len(),
            "padded chunk advances {real} of {} rows",
            chunk.len()
        );
        let m = *self
            .pf_widths
            .get(wi)
            .ok_or_else(|| anyhow::anyhow!("prefill width {wi} not built"))?;
        anyhow::ensure!(
            chunk.len() == m,
            "prefill chunk is {} rows, want the {m}-row chunk graph",
            chunk.len()
        );
        anyhow::ensure!(
            self.pos + m <= self.max_seq,
            "kv cache full at {} + {m} (max_seq {})",
            self.pos,
            self.max_seq
        );
        let hidden = self.cfg.hidden_size;
        let words = hidden / 2;
        let gpu_mode = self.pf_gpu.is_some();
        let gspan = self.pf_gpu.as_ref().and_then(|g| {
            let a = self.pos.max(g.vs);
            let b = (self.pos + real).min(g.vs + g.n_vis);
            (a < b).then_some((a, b))
        });
        let t_repack = std::time::Instant::now();
        let mut toks = vec![0u32; m];
        let mut mask = vec![0u32; m];
        let mut splice = (!gpu_mode).then(|| vec![0u32; m * words]);
        for (i, (tok, row)) in chunk.iter().enumerate() {
            if let Some((a, b)) = gspan {
                let j = self.pos + i;
                if j >= a && j < b {
                    anyhow::ensure!(
                        row.is_none(),
                        "gpu vision rows arrive by device copy, never as host slices"
                    );
                    mask[i] = 1;
                    continue;
                }
            }
            match row {
                Some(r) => {
                    anyhow::ensure!(
                        !gpu_mode,
                        "a host vision slice inside a gpu-handoff prefill would bypass the \
                         device copy this chunk rides"
                    );
                    anyhow::ensure!(
                        r.len() == hidden,
                        "supplied hidden row has {} values, want {hidden}",
                        r.len()
                    );
                    mask[i] = 1;
                    let bits: Vec<u16> = r
                        .iter()
                        .map(|v| half::bf16::from_f32(*v).to_bits())
                        .collect();
                    splice.as_mut().expect("host splice staging present")
                        [i * words..(i + 1) * words]
                        .copy_from_slice(&pack_pairs(&bits));
                }
                None => {
                    anyhow::ensure!((*tok as usize) < self.vocab, "token {tok} out of vocab");
                    toks[i] = *tok;
                }
            }
        }
        self.repack_s += t_repack.elapsed().as_secs_f64();
        {
            let pf = &self.pf[wi];
            self.ctx
                .queue
                .write_buffer(&pf.tok, 0, bytemuck::cast_slice(&toks));
            self.ctx
                .queue
                .write_buffer(&pf.mask, 0, bytemuck::cast_slice(&mask));
            if let Some(splice) = &splice {
                self.ctx
                    .queue
                    .write_buffer(&pf.splice, 0, bytemuck::cast_slice(splice));
            }
        }
        if let Some((a, b)) = gspan {
            let g = self.pf_gpu.as_ref().expect("gpu span implies gpu rows");
            let pf = &self.pf[wi];
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(
                &g.buf,
                ((a - g.vs) * words * 4) as u64,
                &pf.splice,
                ((a - self.pos) * words * 4) as u64,
                ((b - a) * words * 4) as u64,
            );
            self.ctx.queue.submit([enc.finish()]);
            self.handoff_copies += 1;
        }
        self.step
            .write(self.ctx, 0, self.pos as u32, (self.pos + m) as u32);
        let scope = (!self.pf_validated).then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        let cb = dispatch::encode_pass_list_labeled(
            self.ctx,
            self.pf_passes[wi]
                .iter()
                .map(|p| (&*p.pipeline, &p.bind, p.grid)),
            &[],
        );
        self.ctx.queue.submit([cb]);
        if let Some(scope) = scope {
            if let Some(e) = dispatch::pop_scope(scope) {
                anyhow::bail!("dots decoder prefill chunk validation: {e}");
            }
            self.pf_validated = true;
        }
        self.pos += real;
        Ok(())
    }

    fn swapped_run(&self, wi: usize, bat: bool) -> Vec<&Pass> {
        let rg = self.region.as_ref().expect("region machinery present");
        let list = &self.pf_passes[wi];
        let sites = &rg.sites[wi];
        let mut run: Vec<&Pass> = Vec::with_capacity(list.len());
        let mut i = 0usize;
        for site in sites {
            let repl = if bat {
                site.bat.as_ref()
            } else {
                site.rpf.as_ref()
            };
            let Some(repl) = repl else {
                continue;
            };
            while i < site.start {
                run.push(&list[i]);
                i += 1;
            }
            for r in repl {
                run.push(r);
            }
            i += site.len;
        }
        while i < list.len() {
            run.push(&list[i]);
            i += 1;
        }
        run
    }

    fn pack_run(&self, lanes_used: usize) -> Vec<&Pass> {
        let pk = self.pack.as_ref().expect("pack machinery present");
        let rg = self.region.as_ref().expect("region machinery present");
        let list = &self.pf_passes[pk.wi];
        let sites = &rg.sites[pk.wi];
        let mut run: Vec<&Pass> = Vec::with_capacity(list.len());
        let mut i = 0usize;
        for site in sites {
            let Some(repl) = site.pack.as_ref() else {
                continue;
            };
            while i < site.start {
                run.push(&list[i]);
                i += 1;
            }
            let take = if site.pack_lane > 0 {
                site.pack_lane * lanes_used
            } else {
                repl.len()
            };
            for r in &repl[..take] {
                run.push(r);
            }
            i += site.len;
        }
        while i < list.len() {
            run.push(&list[i]);
            i += 1;
        }
        run
    }

    pub fn rpf_packs(&self) -> u64 {
        self.rpf_packs
    }

    pub fn rpf_pack_lanes(&self) -> u64 {
        self.rpf_pack_lanes
    }

    pub fn region_capacity(&self) -> Option<(usize, usize)> {
        self.region.as_ref().map(|rg| {
            (
                REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES,
                rg.stride - 1,
            )
        })
    }

    fn rpf_chunk(
        &mut self,
        wi: usize,
        chunk: &[(u32, Option<&[f32]>)],
        base: usize,
        lpos: usize,
        gcopy: Option<RpfGpuCopy<'_>>,
    ) -> Result<()> {
        let m = self.pf_widths[wi];
        anyhow::ensure!(
            chunk.len() == m,
            "rpf chunk is {} rows, want {m}",
            chunk.len()
        );
        self.rpf_chunk_count += 1;
        let hidden = self.cfg.hidden_size;
        let words = hidden / 2;
        let gpu_mode = gcopy.is_some();
        let in_gspan = |i: usize| {
            gcopy
                .as_ref()
                .is_some_and(|gc| i >= gc.dst_row && i < gc.dst_row + gc.rows)
        };
        let t_repack = std::time::Instant::now();
        let mut toks = vec![0u32; m];
        let mut mask = vec![0u32; m];
        let mut splice = (!gpu_mode).then(|| vec![0u32; m * words]);
        for (i, (tok, row)) in chunk.iter().enumerate() {
            if in_gspan(i) {
                anyhow::ensure!(
                    row.is_none(),
                    "gpu vision rows arrive by device copy, never as host slices"
                );
                mask[i] = 1;
                continue;
            }
            match row {
                Some(r) => {
                    anyhow::ensure!(
                        !gpu_mode,
                        "a host vision slice inside a gpu-handoff region chunk would bypass \
                         the device copy this chunk rides"
                    );
                    anyhow::ensure!(
                        r.len() == hidden,
                        "supplied hidden row has {} values, want {hidden}",
                        r.len()
                    );
                    mask[i] = 1;
                    let bits: Vec<u16> = r
                        .iter()
                        .map(|v| half::bf16::from_f32(*v).to_bits())
                        .collect();
                    splice.as_mut().expect("host splice staging present")
                        [i * words..(i + 1) * words]
                        .copy_from_slice(&pack_pairs(&bits));
                }
                None => {
                    anyhow::ensure!((*tok as usize) < self.vocab, "token {tok} out of vocab");
                    toks[i] = *tok;
                }
            }
        }
        self.rpf_repack_s += t_repack.elapsed().as_secs_f64();
        {
            let pf = &self.pf[wi];
            self.ctx
                .queue
                .write_buffer(&pf.tok, 0, bytemuck::cast_slice(&toks));
            self.ctx
                .queue
                .write_buffer(&pf.mask, 0, bytemuck::cast_slice(&mask));
            if let Some(splice) = &splice {
                self.ctx
                    .queue
                    .write_buffer(&pf.splice, 0, bytemuck::cast_slice(splice));
            }
        }
        if let Some(gc) = &gcopy {
            let pf = &self.pf[wi];
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(
                gc.buf,
                (gc.src_row * words * 4) as u64,
                &pf.splice,
                (gc.dst_row * words * 4) as u64,
                (gc.rows * words * 4) as u64,
            );
            self.ctx.queue.submit([enc.finish()]);
            self.handoff_copies += 1;
        }
        self.step
            .write_base(self.ctx, 0, lpos as u32, (lpos + m) as u32, base as u32);
        let need_scope = !self.region.as_ref().expect("region machinery").validated;
        let scope = need_scope.then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        {
            let run = self.swapped_run(wi, false);
            let cb = dispatch::encode_pass_list_labeled(
                self.ctx,
                run.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
                &[],
            );
            self.ctx.queue.submit([cb]);
        }
        if let Some(scope) = scope {
            if let Some(e) = dispatch::pop_scope(scope) {
                anyhow::bail!("dots region prefill chunk validation: {e}");
            }
            self.region.as_mut().expect("region machinery").validated = true;
        }
        Ok(())
    }

    pub fn region_prefill(
        &mut self,
        slot: usize,
        prompt_tokens: &[u32],
        vision_start: usize,
        vision_features: Option<&[f32]>,
    ) -> Result<usize> {
        let src = match vision_features {
            Some(f) => RegionRowsSrc::Host(f),
            None => RegionRowsSrc::None,
        };
        self.region_prefill_src(slot, prompt_tokens, vision_start, src)
    }

    pub fn region_prefill_src(
        &mut self,
        slot: usize,
        prompt_tokens: &[u32],
        vision_start: usize,
        src: RegionRowsSrc<'_>,
    ) -> Result<usize> {
        let c0 = self.rpf_chunk_count;
        let r = self.region_prefill_src_inner(slot, prompt_tokens, vision_start, src);
        if page_profile_enabled() {
            if let Ok(gend) = &r {
                eprintln!(
                    "DOTS-RPF-PLAN slot={slot} gend={gend} chunks={}",
                    self.rpf_chunk_count - c0
                );
            }
        }
        r
    }

    pub fn region_prefill_pack(&mut self, jobs: &[RegionPackJob<'_>]) -> Result<Vec<usize>> {
        let lane =
            RPF_PACK_LANE_ROWS_ARE_THE_128_CHUNK_SO_EVERY_LANE_REPLAYS_TODAYS_STEEL_ATTENTION;
        let lanes_max = RPF_PACK_LANES_8_FILL_THE_1024_TRUNK;
        let serial = |g: &mut Self, j: &RegionPackJob<'_>| {
            g.region_prefill_src(j.slot, j.prompt_tokens, j.vision_start, j.src.reborrow())
        };
        if self.pack.is_none() || !rpf_pack_enabled() || jobs.len() < 2 {
            return jobs.iter().map(|j| serial(self, j)).collect();
        }
        let mut lens = vec![0usize; jobs.len()];
        let mut groups: Vec<(Vec<usize>, usize)> = Vec::new();
        let mut group: Vec<usize> = Vec::new();
        let mut group_lanes = 0usize;
        for (ji, j) in jobs.iter().enumerate() {
            let lj = j
                .prompt_tokens
                .len()
                .saturating_sub(1)
                .max(1)
                .div_ceil(lane);
            if group_lanes + lj > lanes_max {
                groups.push((std::mem::take(&mut group), group_lanes));
                group_lanes = 0;
            }
            group.push(ji);
            group_lanes += lj;
        }
        groups.push((group, group_lanes));
        for (g, g_lanes) in groups {
            if g.len() >= 2
                && g_lanes >= RPF_PACK_MIN_LANES_5_IS_THE_1024_OVER_128_WEIGHT_READ_BREAK_EVEN
            {
                self.rpf_pack_exec(jobs, &g, &mut lens)?;
            } else {
                for ji in g {
                    lens[ji] = serial(self, &jobs[ji])?;
                }
            }
        }
        Ok(lens)
    }

    fn rpf_pack_exec(
        &mut self,
        jobs: &[RegionPackJob<'_>],
        idxs: &[usize],
        lens: &mut [usize],
    ) -> Result<()> {
        let lane =
            RPF_PACK_LANE_ROWS_ARE_THE_128_CHUNK_SO_EVERY_LANE_REPLAYS_TODAYS_STEEL_ATTENTION;
        let pk_wi = self.pack.as_ref().expect("pack machinery present").wi;
        let m = self.pf_widths[pk_wi];
        let rg = self.region.as_ref().expect("region machinery present");
        let stride = rg.stride;
        let slots = REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES;
        let hidden = self.cfg.hidden_size;
        let words = hidden / 2;
        let mut toks = vec![0u32; m];
        let mut mask = vec![0u32; m];
        let mut bpos = vec![0u32; m];
        let mut bslot = vec![u32::MAX; m];
        let mut brange = vec![0u32; 2 * m];
        let mut splice: Option<Vec<u32>> = None;
        let mut copies: Vec<(usize, usize, usize)> = Vec::new();
        let mut lane_writes: Vec<(usize, [u32; 4])> = Vec::new();
        let mut lc = 0usize;
        let t_repack = std::time::Instant::now();
        for &ji in idxs {
            let j = &jobs[ji];
            anyhow::ensure!(j.slot < slots, "region slot {} out of {slots}", j.slot);
            anyhow::ensure!(
                j.prompt_tokens.len() > 1,
                "region prompt needs at least two tokens"
            );
            let gend = j.prompt_tokens.len() - 1;
            anyhow::ensure!(
                gend < stride,
                "region prompt of {gend} rows does not fit the {stride}-row stride minus its \
                 reserved pad row (lower NV_DOTS_CROP_MAX_PIXELS or raise NV_DOTS_MAX_SEQ)"
            );
            let lj = gend.div_ceil(lane).max(1);
            anyhow::ensure!(
                lc + lj <= RPF_PACK_LANES_8_FILL_THE_1024_TRUNK,
                "pack lane overflow: {lc} + {lj} lanes"
            );
            let base = j.slot * stride;
            let vs = j.vision_start;
            let (n_vis, host, gpu): (usize, Option<&[f32]>, Option<&VisionRowsGpu>) = match &j.src {
                RegionRowsSrc::None => (0, None, None),
                RegionRowsSrc::Host(f) => {
                    anyhow::ensure!(
                        f.len().is_multiple_of(hidden),
                        "vision features hold {} values, not a multiple of hidden {hidden}",
                        f.len()
                    );
                    let n = f.len() / hidden;
                    anyhow::ensure!(
                        vs + n <= j.prompt_tokens.len(),
                        "vision span {vs}+{n} exceeds prompt {}",
                        j.prompt_tokens.len()
                    );
                    (n, Some(f), None)
                }
                RegionRowsSrc::Gpu(g) => {
                    anyhow::ensure!(
                        g.hidden == hidden,
                        "gpu vision rows are {} wide, decoder hidden is {hidden}",
                        g.hidden
                    );
                    anyhow::ensure!(
                        g.buf.size() >= (g.rows * hidden * 2) as u64,
                        "gpu vision row buffer holds {} bytes for {} rows",
                        g.buf.size(),
                        g.rows
                    );
                    anyhow::ensure!(
                        vs + g.rows <= j.prompt_tokens.len(),
                        "vision span {vs}+{} exceeds prompt {}",
                        g.rows,
                        j.prompt_tokens.len()
                    );
                    (g.rows, None, Some(g))
                }
            };
            let row0 = lc * lane;
            for local in 0..lj * lane {
                let p = row0 + local;
                bpos[p] = local as u32;
                bslot[p] = (base + local) as u32;
                brange[2 * p] = base as u32;
                brange[2 * p + 1] = (local + 1) as u32;
                if local >= gend {
                    continue;
                }
                if local >= vs && local < vs + n_vis {
                    mask[p] = 1;
                    if let Some(f) = host {
                        let r = &f[(local - vs) * hidden..(local - vs + 1) * hidden];
                        let bits: Vec<u16> = r
                            .iter()
                            .map(|v| half::bf16::from_f32(*v).to_bits())
                            .collect();
                        splice.get_or_insert_with(|| vec![0u32; m * words])
                            [p * words..(p + 1) * words]
                            .copy_from_slice(&pack_pairs(&bits));
                    }
                } else {
                    let tok = j.prompt_tokens[local];
                    anyhow::ensure!((tok as usize) < self.vocab, "token {tok} out of vocab");
                    toks[p] = tok;
                }
            }
            if gpu.is_some() {
                let bnd = (vs + n_vis).min(gend);
                if vs < bnd {
                    copies.push((ji, row0 + vs, bnd - vs));
                }
            }
            for l2 in 0..lj {
                lane_writes.push((
                    lc + l2,
                    [0, (l2 * lane) as u32, ((l2 + 1) * lane) as u32, base as u32],
                ));
            }
            lc += lj;
            lens[ji] = gend;
        }
        self.rpf_repack_s += t_repack.elapsed().as_secs_f64();
        {
            let pf = &self.pf[pk_wi];
            let pk = self.pack.as_ref().expect("pack machinery present");
            self.ctx
                .queue
                .write_buffer(&pf.tok, 0, bytemuck::cast_slice(&toks));
            self.ctx
                .queue
                .write_buffer(&pf.mask, 0, bytemuck::cast_slice(&mask));
            if let Some(splice) = &splice {
                self.ctx
                    .queue
                    .write_buffer(&pf.splice, 0, bytemuck::cast_slice(splice));
            }
            self.ctx
                .queue
                .write_buffer(&pk.bpos, 0, bytemuck::cast_slice(&bpos));
            self.ctx
                .queue
                .write_buffer(&pk.bslot, 0, bytemuck::cast_slice(&bslot));
            self.ctx
                .queue
                .write_buffer(&pk.brange, 0, bytemuck::cast_slice(&brange));
            for (l, st) in &lane_writes {
                self.ctx
                    .queue
                    .write_buffer(&pk.lane_step[*l], 0, bytemuck::cast_slice(st));
            }
        }
        if !copies.is_empty() {
            let pf = &self.pf[pk_wi];
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            for (ji, dst_row, rows) in &copies {
                let RegionRowsSrc::Gpu(g) = &jobs[*ji].src else {
                    anyhow::bail!("pack copy source must be gpu rows");
                };
                enc.copy_buffer_to_buffer(
                    &g.buf,
                    0,
                    &pf.splice,
                    (dst_row * words * 4) as u64,
                    (rows * words * 4) as u64,
                );
            }
            self.ctx.queue.submit([enc.finish()]);
            self.handoff_copies += copies.len() as u64;
        }
        let need_scope = !self
            .pack
            .as_ref()
            .expect("pack machinery present")
            .validated;
        let scope = need_scope.then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        {
            let run = self.pack_run(lc);
            let cb = dispatch::encode_pass_list_labeled(
                self.ctx,
                run.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
                &[],
            );
            self.ctx.queue.submit([cb]);
        }
        if let Some(scope) = scope {
            if let Some(e) = dispatch::pop_scope(scope) {
                anyhow::bail!("dots region pack prefill validation: {e}");
            }
            self.pack
                .as_mut()
                .expect("pack machinery present")
                .validated = true;
        }
        self.rpf_packs += 1;
        self.rpf_pack_lanes += lc as u64;
        if page_profile_enabled() {
            eprintln!("DOTS-RPF-PACK lanes={lc} regions={}", idxs.len());
        }
        Ok(())
    }

    fn region_prefill_src_inner(
        &mut self,
        slot: usize,
        prompt_tokens: &[u32],
        vision_start: usize,
        src: RegionRowsSrc<'_>,
    ) -> Result<usize> {
        let Some(rg) = self.region.as_ref() else {
            anyhow::bail!(
                "region machinery needs the 32-row prefill width and a span of at least 64"
            );
        };
        let stride = rg.stride;
        let slots = REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES;
        anyhow::ensure!(slot < slots, "region slot {slot} out of {slots}");
        anyhow::ensure!(
            prompt_tokens.len() > 1,
            "region prompt needs at least two tokens"
        );
        let base = slot * stride;
        let hidden = self.cfg.hidden_size;
        let host_rows: Option<&[f32]> = match &src {
            RegionRowsSrc::Host(f) => Some(f),
            _ => None,
        };
        let gpu_rows: Option<&VisionRowsGpu> = match &src {
            RegionRowsSrc::Gpu(g) => Some(g),
            _ => None,
        };
        let n_vis = match &src {
            RegionRowsSrc::None => 0,
            RegionRowsSrc::Host(f) => {
                anyhow::ensure!(
                    f.len().is_multiple_of(hidden),
                    "vision features hold {} values, not a multiple of hidden {hidden}",
                    f.len()
                );
                let n = f.len() / hidden;
                anyhow::ensure!(
                    vision_start + n <= prompt_tokens.len(),
                    "vision span {vision_start}+{n} exceeds prompt {}",
                    prompt_tokens.len()
                );
                n
            }
            RegionRowsSrc::Gpu(g) => {
                anyhow::ensure!(
                    g.hidden == hidden,
                    "gpu vision rows are {} wide, decoder hidden is {hidden}",
                    g.hidden
                );
                anyhow::ensure!(
                    g.buf.size() >= (g.rows * hidden * 2) as u64,
                    "gpu vision row buffer holds {} bytes for {} rows",
                    g.buf.size(),
                    g.rows
                );
                anyhow::ensure!(
                    vision_start + g.rows <= prompt_tokens.len(),
                    "vision span {vision_start}+{} exceeds prompt {}",
                    g.rows,
                    prompt_tokens.len()
                );
                g.rows
            }
        };
        let gend = prompt_tokens.len() - 1;
        anyhow::ensure!(
            gend < stride,
            "region prompt of {gend} rows does not fit the {stride}-row stride minus its \
             reserved pad row (lower NV_DOTS_CROP_MAX_PIXELS or raise NV_DOTS_MAX_SEQ)"
        );
        let row_at = |i: usize| -> Option<&[f32]> {
            host_rows
                .filter(|_| i >= vision_start && i < vision_start + n_vis)
                .map(|f| &f[(i - vision_start) * hidden..(i - vision_start + 1) * hidden])
        };
        let in_span = |j: usize| j >= vision_start && j < vision_start + n_vis;
        let gpu_vis = gpu_rows.is_some();
        let entry = |j: usize| -> (u32, Option<&[f32]>) {
            if gpu_vis && in_span(j) {
                (0, None)
            } else {
                (prompt_tokens[j], row_at(j))
            }
        };
        let gcopy_for = |i: usize, m: usize| -> Option<RpfGpuCopy<'_>> {
            let g = gpu_rows?;
            let a = i.max(vision_start);
            let b = (i + m).min(gend).min(vision_start + n_vis);
            (a < b).then_some(RpfGpuCopy {
                buf: &g.buf,
                src_row: a - vision_start,
                dst_row: a - i,
                rows: b - a,
            })
        };
        let widths = self.pf_widths.clone();
        let wide = rpf_wide_enabled();
        if wide {
            let one = widths
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, m)| {
                    *m >= gend && *m <= stride && (*m <= PREFILL_M_WIDE || gend * 2 >= *m)
                })
                .min_by_key(|&(_, m)| m);
            if let Some((wi, m)) = one {
                let mut chunk: Vec<(u32, Option<&[f32]>)> = (0..gend).map(entry).collect();
                chunk.resize(m, (0, None));
                self.rpf_chunk(wi, &chunk, base, 0, gcopy_for(0, m))?;
                return Ok(gend);
            }
        }
        let usable: Vec<(usize, usize)> = widths
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, m)| *m < stride)
            .collect();
        anyhow::ensure!(
            !usable.is_empty(),
            "no prefill width fits the region stride"
        );
        let mut lpos = 0usize;
        let mut i = 0usize;
        for (wi, m) in usable.iter().copied() {
            if wide && m < PREFILL_M_WIDE {
                break;
            }
            while i + m <= gend && lpos + m < stride {
                let chunk: Vec<(u32, Option<&[f32]>)> = (i..i + m).map(entry).collect();
                self.rpf_chunk(wi, &chunk, base, lpos, gcopy_for(i, m))?;
                lpos += m;
                i += m;
            }
        }
        if wide && gend > i {
            let rem = gend - i;
            let wide_tail = usable
                .iter()
                .copied()
                .rev()
                .find(|(_, m)| *m >= rem && lpos + m < stride && *m >= PREFILL_M_WIDE);
            if let Some((wi, m)) = wide_tail {
                let mut chunk: Vec<(u32, Option<&[f32]>)> = (i..gend).map(entry).collect();
                chunk.resize(m, (0, None));
                self.rpf_chunk(wi, &chunk, base, lpos, gcopy_for(i, m))?;
                return Ok(gend);
            }
            for (wi, m) in usable.iter().copied().filter(|(_, m)| *m < PREFILL_M_WIDE) {
                while i + m <= gend && lpos + m < stride {
                    let chunk: Vec<(u32, Option<&[f32]>)> = (i..i + m).map(entry).collect();
                    self.rpf_chunk(wi, &chunk, base, lpos, gcopy_for(i, m))?;
                    lpos += m;
                    i += m;
                }
            }
        }
        let rem = gend - i;
        if rem > 0 {
            let tail = usable
                .iter()
                .copied()
                .rev()
                .find(|(_, m)| *m >= rem && lpos + m < stride);
            let Some((wi, m)) = tail else {
                anyhow::bail!(
                    "region prompt tail of {rem} rows does not fit the {stride}-row stride at \
                     local position {lpos}"
                );
            };
            let mut chunk: Vec<(u32, Option<&[f32]>)> = (i..gend).map(entry).collect();
            chunk.resize(m, (0, None));
            self.rpf_chunk(wi, &chunk, base, lpos, gcopy_for(i, m))?;
            lpos += rem;
        }
        Ok(lpos)
    }

    pub fn bstep_tokens(&mut self, rows: &[RegionRow]) -> Result<Vec<u32>> {
        let Some(rg) = self.region.as_ref() else {
            anyhow::bail!("region machinery absent");
        };
        let wi = rg.wi;
        let stride = rg.stride as u32;
        let m = self.pf_widths[wi];
        anyhow::ensure!(
            !rows.is_empty() && rows.len() <= m,
            "bstep takes 1..={m} rows, got {}",
            rows.len()
        );
        let mut toks = vec![0u32; m];
        let mask = vec![0u32; m];
        let mut bpos = vec![0u32; m];
        let mut bslot = vec![0u32; m];
        let mut brange = vec![0u32; 2 * m];
        for (r, row) in rows.iter().enumerate() {
            anyhow::ensure!(
                (row.token as usize) < self.vocab,
                "token {} out of vocab",
                row.token
            );
            anyhow::ensure!(
                row.len + 1 < stride && row.base.is_multiple_of(stride),
                "region row {r}: len {} base {} outside the {stride}-row stride contract",
                row.len,
                row.base
            );
            toks[r] = row.token;
            bpos[r] = row.len;
            bslot[r] = row.base + row.len;
            brange[2 * r] = row.base;
            brange[2 * r + 1] = row.len + 1;
        }
        for r in rows.len()..m {
            let park = (r as u32) * stride + stride - 1;
            bslot[r] = park;
            brange[2 * r] = park;
            brange[2 * r + 1] = 1;
        }
        let t_cpu = std::time::Instant::now();
        {
            let pf = &self.pf[wi];
            self.ctx
                .queue
                .write_buffer(&pf.tok, 0, bytemuck::cast_slice(&toks));
            self.ctx
                .queue
                .write_buffer(&pf.mask, 0, bytemuck::cast_slice(&mask));
            self.ctx
                .queue
                .write_buffer(&rg.bpos, 0, bytemuck::cast_slice(&bpos));
            self.ctx
                .queue
                .write_buffer(&rg.bslot, 0, bytemuck::cast_slice(&bslot));
            self.ctx
                .queue
                .write_buffer(&rg.brange, 0, bytemuck::cast_slice(&brange));
        }
        self.step.write_base(self.ctx, 0, 0, m as u32, 0);
        let use_bhead = rg.bhead.is_some() && bhead_enabled();
        let need_scope = !rg.bat_validated;
        let scope = need_scope.then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        {
            let rg = self.region.as_ref().expect("region machinery");
            let mut run = self.swapped_run(wi, true);
            if use_bhead {
                let bh = rg.bhead.as_ref().expect("bhead machinery");
                self.write_bhead_uniforms(bh, rows.len() as u32);
                let (head, _) = bh.head_pass();
                run.push(&bh.gather);
                run.push(head);
                run.push(&bh.am1);
                run.push(&bh.am2);
            } else {
                let (hs, he) = rg.head_range;
                for r in 0..rows.len() {
                    run.push(&rg.gathers[r]);
                    for p in &self.passes[hs..he] {
                        run.push(p);
                    }
                    run.push(&rg.stashes[r]);
                }
            }
            let cb = dispatch::encode_pass_list_labeled(
                self.ctx,
                run.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
                &[],
            );
            self.ctx.queue.submit([cb]);
        }
        if use_bhead {
            self.bstep_bhead += 1;
            let rg = self.region.as_ref().expect("region machinery");
            let bh = rg.bhead.as_ref().expect("bhead machinery");
            if bh.head_pass().1 {
                self.bstep_bhead_mma += 1;
            }
        }
        if let Some(scope) = scope {
            if let Some(e) = dispatch::pop_scope(scope) {
                anyhow::bail!("dots batched region step validation: {e}");
            }
            self.region
                .as_mut()
                .expect("region machinery")
                .bat_validated = true;
        }
        self.bstep_cpu_s += t_cpu.elapsed().as_secs_f64();
        let t_read = std::time::Instant::now();
        let rg = self.region.as_ref().expect("region machinery");
        let all: Vec<u32> = dispatch::read_back(self.ctx, &rg.btoks, m)
            .map_err(|e| anyhow::anyhow!("btoks readback: {e}"))?;
        self.bstep_read_s += t_read.elapsed().as_secs_f64();
        self.bstep_count += 1;
        Ok(all[..rows.len()].to_vec())
    }

    pub fn prefill_embed(&mut self, row: &[f32]) -> Result<()> {
        self.run(0, Some(row), false)
    }

    pub fn decode_embed_logits(&mut self, row: &[f32]) -> Result<(u32, Vec<f32>)> {
        self.run(0, Some(row), true)?;
        self.read_token_and_logits()
    }

    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        let widths = self.pf_widths.clone();
        let mut done = 0usize;
        for (wi, m) in widths.iter().copied().enumerate() {
            while done + m <= tokens.len() {
                let chunk: Vec<(u32, Option<&[f32]>)> =
                    tokens[done..done + m].iter().map(|t| (*t, None)).collect();
                self.prefill_chunk(wi, &chunk)?;
                done += m;
            }
        }
        Ok(done)
    }

    pub fn prefill_prompt(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        anyhow::ensure!(!tokens.is_empty(), "prefill_prompt needs a token");
        let widths = self.pf_widths.clone();
        let end = tokens.len() - 1;
        let mut i = 0usize;
        if let Some(&bw) = widths.first().filter(|&&bw| {
            bw > PREFILL_M_WIDE && end <= bw && end >= bw * 5 / 8 && self.pos + bw <= self.max_seq
        }) {
            let _ = PREFILL_BIG_HEAD_FIRES_AT_FIVE_EIGHTHS_BECAUSE_ONE_PADDED_CHUNK_READS_KV_ONCE;
            let mut chunk: Vec<(u32, Option<&[f32]>)> =
                tokens[..end].iter().map(|t| (*t, None)).collect();
            chunk.resize(bw, (0, None));
            self.prefill_chunk_real(0, &chunk, end)?;
            i = end;
        }
        for (wi, m) in widths.iter().copied().enumerate() {
            while i + m <= end {
                let chunk: Vec<(u32, Option<&[f32]>)> =
                    tokens[i..i + m].iter().map(|t| (*t, None)).collect();
                self.prefill_chunk(wi, &chunk)?;
                i += m;
            }
        }
        let rem = end - i;
        if rem > 0 {
            let tail = widths
                .iter()
                .copied()
                .enumerate()
                .rev()
                .find(|(_, m)| *m >= rem && self.pos + m <= self.max_seq);
            match tail {
                Some((wi, m)) => {
                    let mut chunk: Vec<(u32, Option<&[f32]>)> =
                        tokens[i..end].iter().map(|t| (*t, None)).collect();
                    chunk.resize(m, (0, None));
                    self.prefill_chunk_real(wi, &chunk, rem)?;
                }
                None => {
                    while i < end {
                        self.prefill_step(tokens[i])?;
                        i += 1;
                    }
                }
            }
        }
        Ok(self.decode_step_logits(tokens[end])?.1)
    }

    pub fn generate(
        &mut self,
        prompt_tokens: &[u32],
        vision_start: usize,
        vision_features: Option<&[f32]>,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        let src = match vision_features {
            Some(f) => RowsSrc::Host(f),
            None => RowsSrc::None,
        };
        self.generate_src(prompt_tokens, vision_start, src, opts)
    }

    pub fn generate_gpu_rows(
        &mut self,
        prompt_tokens: &[u32],
        vision_start: usize,
        rows: &VisionRowsGpu,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        self.generate_src(prompt_tokens, vision_start, RowsSrc::Gpu(rows), opts)
    }

    pub async fn generate_gpu_rows_pumped(
        &mut self,
        prompt_tokens: &[u32],
        vision_start: usize,
        rows: &VisionRowsGpu,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        self.generate_arm(
            prompt_tokens,
            vision_start,
            RowsSrc::Gpu(rows),
            opts,
            dispatch::ReadbackArm::Pumped,
        )
        .await
    }

    pub async fn generate_pumped(
        &mut self,
        prompt_tokens: &[u32],
        vision_start: usize,
        vision_features: Option<&[f32]>,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        let src = match vision_features {
            Some(f) => RowsSrc::Host(f),
            None => RowsSrc::None,
        };
        self.generate_arm(
            prompt_tokens,
            vision_start,
            src,
            opts,
            dispatch::ReadbackArm::Pumped,
        )
        .await
    }

    fn generate_src(
        &mut self,
        prompt_tokens: &[u32],
        vision_start: usize,
        src: RowsSrc<'_>,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        dispatch::drive_blocking(self.generate_arm(
            prompt_tokens,
            vision_start,
            src,
            opts,
            dispatch::ReadbackArm::Blocking,
        ))
    }

    async fn generate_arm(
        &mut self,
        prompt_tokens: &[u32],
        vision_start: usize,
        src: RowsSrc<'_>,
        opts: &GenerateOptions,
        arm: dispatch::ReadbackArm,
    ) -> Result<GenerateOutcome> {
        let hidden = self.cfg.hidden_size;
        let prompt_len = prompt_tokens.len();
        anyhow::ensure!(prompt_len > 0, "generate needs a prompt");
        anyhow::ensure!(
            prompt_len < self.max_seq,
            "prompt of {prompt_len} tokens exceeds the graph kv span {} (raise NV_DOTS_MAX_SEQ \
             or lower NV_DOTS_MAX_PIXELS)",
            self.max_seq
        );
        let n_vis = match &src {
            RowsSrc::None => 0,
            RowsSrc::Host(f) => {
                anyhow::ensure!(
                    f.len().is_multiple_of(hidden),
                    "vision features hold {} values, not a multiple of hidden {hidden}",
                    f.len()
                );
                let n = f.len() / hidden;
                anyhow::ensure!(
                    vision_start + n <= prompt_len,
                    "vision span {vision_start}+{n} exceeds prompt {prompt_len}"
                );
                n
            }
            RowsSrc::Gpu(g) => {
                anyhow::ensure!(
                    g.hidden == hidden,
                    "gpu vision rows are {} wide, decoder hidden is {hidden}",
                    g.hidden
                );
                anyhow::ensure!(
                    g.buf.size() >= (g.rows * hidden * 2) as u64,
                    "gpu vision row buffer holds {} bytes for {} rows",
                    g.buf.size(),
                    g.rows
                );
                anyhow::ensure!(
                    vision_start + g.rows <= prompt_len,
                    "vision span {vision_start}+{} exceeds prompt {prompt_len}",
                    g.rows
                );
                g.rows
            }
        };
        self.reset()?;
        if let RowsSrc::Gpu(g) = &src {
            self.pf_gpu = Some(PfGpuVisionRows {
                buf: g.buf.clone(),
                vs: vision_start,
                n_vis,
            });
        }
        self.handoff_copies = 0;

        let t_gen = std::time::Instant::now();
        self.repack_s = 0.0;
        let host_rows: Option<&[f32]> = match &src {
            RowsSrc::Host(f) => Some(f),
            _ => None,
        };
        let gpu_vis = matches!(&src, RowsSrc::Gpu(_));
        let in_span = |i: usize| i >= vision_start && i < vision_start + n_vis;
        let row_at = |i: usize| -> Option<&[f32]> {
            host_rows
                .filter(|_| in_span(i))
                .map(|f| &f[(i - vision_start) * hidden..(i - vision_start + 1) * hidden])
        };
        let entry = |j: usize| -> (u32, Option<&[f32]>) {
            if gpu_vis && in_span(j) {
                (0, None)
            } else {
                (prompt_tokens[j], row_at(j))
            }
        };
        let widths = self.pf_widths.clone();
        let mut i = 0usize;
        let gend = prompt_len - 1;
        if let Some(&bw) = widths.first().filter(|&&bw| {
            bw > PREFILL_M_WIDE && gend <= bw && gend >= bw * 5 / 8 && self.pos + bw <= self.max_seq
        }) {
            let mut chunk: Vec<(u32, Option<&[f32]>)> = (0..gend).map(entry).collect();
            chunk.resize(bw, (0, None));
            self.prefill_chunk_real(0, &chunk, gend)?;
            i = gend;
        }
        for (wi, m) in widths.iter().copied().enumerate() {
            while i + m < prompt_len {
                let chunk: Vec<(u32, Option<&[f32]>)> = (i..i + m).map(entry).collect();
                self.prefill_chunk(wi, &chunk)?;
                i += m;
            }
        }
        let rem = (prompt_len - 1) - i;
        if rem > 0 {
            let tail = widths
                .iter()
                .copied()
                .enumerate()
                .rev()
                .find(|(_, m)| *m >= rem && self.pos + m <= self.max_seq);
            if let Some((wi, m)) = tail {
                let mut chunk: Vec<(u32, Option<&[f32]>)> =
                    (i..prompt_len - 1).map(entry).collect();
                chunk.resize(m, (0, None));
                self.prefill_chunk_real(wi, &chunk, rem)?;
                i = prompt_len - 1;
            }
        }
        while i + 1 < prompt_len {
            if gpu_vis && in_span(i) {
                let r = self.gpu_row_host_arm(i - vision_start, arm).await?;
                self.prefill_embed(&r)?;
            } else {
                match row_at(i) {
                    Some(r) => self.prefill_embed(r)?,
                    None => self.prefill_step(prompt_tokens[i])?,
                }
            }
            i += 1;
        }
        let gpu_tok = {
            if gpu_vis && in_span(prompt_len - 1) {
                let r = self
                    .gpu_row_host_arm(prompt_len - 1 - vision_start, arm)
                    .await?;
                self.run(0, Some(&r), true)?;
            } else {
                match row_at(prompt_len - 1) {
                    Some(r) => self.run(0, Some(r), true)?,
                    None => self.step_inner(prompt_tokens[prompt_len - 1], true)?,
                }
            }
            self.read_token_and_logits_arm(arm).await?.0
        };
        self.pf_gpu = None;
        let prefill_s = t_gen.elapsed().as_secs_f64();
        let t_decode = std::time::Instant::now();

        let mut generated: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        let mut spec_agree: Vec<u64> = vec![0; self.spec_tok.len()];
        let mut spec_steps: u64 = 0;
        self.ra_gpu_fed = 0;
        let mk_stage = |label| {
            self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: 4,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let stages = (arm == dispatch::ReadbackArm::Blocking
            && dec_runahead()
            && self.spec_tok.is_empty()
            && !dispatch::profile::enabled())
        .then(|| [mk_stage("dg-ra-stage0"), mk_stage("dg-ra-stage1")]);
        let mut pending: Option<(wgpu::SubmissionIndex, usize)> = None;
        let mut cpu_tok = Some(gpu_tok);
        for _ in 0..opts.max_new_tokens {
            let banned = match opts.ngram_size {
                Some(n) if n > 0 => banned_tokens_windowed_ngram(
                    &generated,
                    n,
                    opts.ngram_window,
                    &opts.ngram_whitelist,
                ),
                _ => Vec::new(),
            };
            let next = if banned.is_empty() {
                match pending.take() {
                    Some((idx, s)) => {
                        let st = stages.as_ref().expect("pending implies run-ahead stages");
                        if self.pos + 1 < self.max_seq {
                            let ni = self.step_ahead(None, &st[1 - s])?;
                            pending = Some((ni, 1 - s));
                        }
                        self.await_token(idx, &st[s])?
                    }
                    None => cpu_tok.take().expect("the serial entry token is present"),
                }
            } else {
                pending = None;
                argmax_banned(&self.read_logits_arm(arm).await?, &banned)
            };
            if !self.spec_tok.is_empty() {
                for (si, (_, buf)) in self.spec_tok.iter().enumerate() {
                    let t: Vec<u32> = dispatch::read_back(self.ctx, buf, 1)
                        .map_err(|e| anyhow::anyhow!("spec probe readback: {e}"))?;
                    if t[0] == next {
                        spec_agree[si] += 1;
                    }
                }
                spec_steps += 1;
            }
            if opts.eos_token_ids.contains(&next) {
                hit_eos = true;
                break;
            }
            generated.push(next);
            if opts.stop_on_loop
                && generated.len().is_multiple_of(64)
                && detect_loop(&generated).is_some()
            {
                break;
            }
            if self.pos + usize::from(pending.is_none()) >= self.max_seq {
                break;
            }
            if pending.is_none() {
                match &stages {
                    Some(st) => pending = Some((self.step_ahead(Some(next), &st[0])?, 0)),
                    None => {
                        self.step_inner(next, true)?;
                        cpu_tok = Some(self.read_token_arm(arm).await?);
                    }
                }
            }
        }
        if page_profile_enabled() {
            let decode_s = t_decode.elapsed().as_secs_f64();
            eprintln!(
                "DOTS-PAGE-PROFILE   prefill_s={prefill_s:.3} (of which host repack {:.3}) \
                 ({prompt_len} prompt tokens, {:.0} tok/s)  decode_s={decode_s:.3} ({} generated, \
                 {:.1} tok/s, {:.2} ms/step)",
                self.repack_s,
                prompt_len as f64 / prefill_s.max(1e-9),
                generated.len(),
                generated.len() as f64 / decode_s.max(1e-9),
                decode_s * 1e3 / generated.len().max(1) as f64
            );
        }
        if spec_steps > 0 {
            let curve: Vec<String> = self
                .spec_tok
                .iter()
                .zip(&spec_agree)
                .map(|((n, _), a)| format!("N{n}={:.4}", *a as f64 / spec_steps as f64))
                .collect();
            eprintln!("[dots-spec] steps={spec_steps} {}", curve.join(" "));
        }
        if let Some(path) = std::env::var(DEC_TOKEN_TRACE_ENV)
            .ok()
            .filter(|p| !p.is_empty())
        {
            use std::io::Write;
            let ids: Vec<String> = generated.iter().map(|t| t.to_string()).collect();
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| anyhow::anyhow!("token trace open {path}: {e}"))?;
            writeln!(
                f,
                "n={} eos={} ids={}",
                generated.len(),
                hit_eos,
                ids.join(",")
            )
            .map_err(|e| anyhow::anyhow!("token trace write {path}: {e}"))?;
        }
        Ok(GenerateOutcome {
            tokens: generated.clone(),
            loop_detection: detect_loop(&generated),
            hit_eos,
        })
    }

    crate::wgpu_step_readback_api!();

    pub fn forward_tokens(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        let (last, rest) = tokens
            .split_last()
            .ok_or_else(|| anyhow::anyhow!("forward_tokens needs at least one token"))?;
        for t in rest {
            self.prefill_step(*t)?;
        }
        Ok(self.decode_step_logits(*last)?.1)
    }

    pub fn greedy_token(logits: &[f32]) -> u32 {
        let mut best = f32::NEG_INFINITY;
        let mut bi = 0u32;
        for (i, v) in logits.iter().enumerate() {
            if *v > best {
                best = *v;
                bi = i as u32;
            }
        }
        bi
    }
}

#[cfg(test)]
mod workgroup_limit_gate {
    use super::*;

    const METAL_EXECUTES_OVER_LIMIT_WORKGROUP_PIPELINES_ABOUT_500X_SLOW_LIMIT_BYTES: u32 = 32768;

    fn assert_every_entry_point_fits(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name}: wgsl parse failed: {e}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: naga validation failed: {e:?}"));
        for (i, ep) in module.entry_points.iter().enumerate() {
            let fi = info.get_entry_point(i);
            let mut bytes = 0u32;
            for (h, gv) in module.global_variables.iter() {
                if gv.space == naga::AddressSpace::WorkGroup && !fi[h].is_empty() {
                    bytes += module.types[gv.ty].inner.size(module.to_ctx());
                }
            }
            assert!(
                bytes <= METAL_EXECUTES_OVER_LIMIT_WORKGROUP_PIPELINES_ABOUT_500X_SLOW_LIMIT_BYTES,
                "{name}::{}: declares {bytes}B of workgroup memory; Metal executes pipelines \
                 over its 32768B limit roughly 500x slow instead of refusing them, so this gate \
                 fails at test time instead",
                ep.name
            );
        }
    }

    #[test]
    fn every_dots_decoder_entry_point_fits_the_metal_workgroup_memory_limit() {
        for (fold, kt) in [
            (1usize, false),
            (1, true),
            (ATTN_FOLD_MAX, false),
            (ATTN_FOLD_MAX, true),
        ] {
            let s = Sources::new(fold, kt, false);
            assert_every_entry_point_fits(
                &format!("rope_tree(fold={fold},kt={kt})"),
                &Sources::new(fold, kt, true).rope,
            );
            for (n, src) in [
                ("gemv_bf16", &s.gemv_bf16),
                ("gemv_nvfp4", &s.gemv_nvfp4),
                ("quant", &s.quant),
                ("common", &s.common),
                ("rms", &s.rms),
                ("rmsres", &s._rmsres),
                ("rope", &s.rope),
                ("rope_act16", &s.rope_act16),
                ("fused", &s.fused),
                ("scvt", &s.scvt),
            ] {
                assert_every_entry_point_fits(&format!("{n}(fold={fold},kt={kt})"), src);
            }
            assert_every_entry_point_fits(
                &format!("rope_sg(fold={fold},kt={kt})"),
                &s.rope
                    .replace("const DGX_SG: u32 = 0u;", "const DGX_SG: u32 = 1u;"),
            );
            assert_every_entry_point_fits(
                &format!("rope_kv16(fold={fold},kt={kt})"),
                &s.rope
                    .replace("const DGKV16: u32 = 0u;", "const DGKV16: u32 = 1u;"),
            );
            assert_every_entry_point_fits(
                &format!("fused_kv16(fold={fold},kt={kt})"),
                &s.fused
                    .replace("const DGF_KV16: u32 = 0u;", "const DGF_KV16: u32 = 1u;"),
            );
            assert_every_entry_point_fits(
                &format!("scvt_kv16(fold={fold},kt={kt})"),
                &s.scvt
                    .replace("const DSC_KV16: u32 = 0u;", "const DSC_KV16: u32 = 1u;"),
            );
        }
        for rows in [8usize, 32] {
            for tree in [false, true] {
                assert_every_entry_point_fits(
                    &format!("gemv_w4(rows={rows},tree={tree})"),
                    &w4g_source(rows, tree),
                );
            }
        }
        assert_every_entry_point_fits("lora_shrink", DG_LORA_SHRINK_WGSL);
        assert_every_entry_point_fits("lora_expand", DG_LORA_EXPAND_WGSL);
        assert_every_entry_point_fits("handoff_pack", PACK_ROWS_BF16_WGSL);
        for f16_out in [false, true] {
            for tree in [false, true] {
                assert_every_entry_point_fits(
                    &format!("rmsres(f16_out={f16_out},tree={tree})"),
                    &drr_source(f16_out, tree),
                );
            }
        }
        let plan = coop_plan();
        for (n, (tm, tn, sg, ku)) in [
            (
                "coop_width_ladder",
                COOP_PLAN_TM4_TN4_SG4_KU2_WON_EVERY_SHAPE_OF_THE_WIDTH_LADDER,
            ),
            ("coop_attn_qk", ATTN_FUSED_QK_PLAN),
            (
                "coop_attn_pv",
                ATTN_FUSED_PV_PLAN_TM2_TN1_SG2_IS_THE_M768_N128_LADDER_KNEE,
            ),
        ] {
            assert_every_entry_point_fits(n, &plan.source(tm, tn, sg, ku));
            assert_every_entry_point_fits(
                &format!("{n}_attn_wrapped"),
                &attn_coop_source(tm, tn, sg, ku),
            );
        }
    }

    fn assert_validates_without_the_subgroup_capability(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name}: wgsl parse failed: {e}"));
        let caps = naga::valid::Capabilities::all().difference(
            naga::valid::Capabilities::SUBGROUP | naga::valid::Capabilities::SUBGROUP_BARRIER,
        );
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), caps)
            .validate(&module)
            .unwrap_or_else(|e| {
                panic!(
                    "{name}: the tree twin must validate without the subgroup capability, or \
                     a browser that does not grant the subgroups feature cannot compile the \
                     W4 decode graph: {e:?}"
                )
            });
    }

    #[test]
    fn tree_twin_modules_validate_without_the_subgroup_capability() {
        for (fold, kt) in [(1usize, false), (ATTN_FOLD_MAX, true)] {
            let st = Sources::new(fold, kt, true);
            assert_validates_without_the_subgroup_capability(
                &format!("rope_tree(fold={fold},kt={kt})"),
                &st.rope,
            );
            assert_validates_without_the_subgroup_capability(
                &format!("rope_act16_tree(fold={fold},kt={kt})"),
                &st.rope_act16,
            );
        }
        for f16_out in [false, true] {
            assert_validates_without_the_subgroup_capability(
                &format!("rmsres_tree(f16_out={f16_out})"),
                &drr_source(f16_out, true),
            );
        }
        for rows in [8usize, 32] {
            assert_validates_without_the_subgroup_capability(
                &format!("gemv_w4_tree(rows={rows})"),
                &w4g_source(rows, true),
            );
        }
        for (name, src) in [
            ("rope_sg", Sources::new(1, false, false).rope),
            ("rmsres_sg", drr_source(false, false)),
            ("gemv_w4_sg", w4g_source(8, false)),
        ] {
            assert!(
                src.contains("subgroup"),
                "{name}: the subgroup arm lost its subgroup builtins, so the no-capability \
                 gate above is vacuous"
            );
        }
    }

    #[test]
    fn sg_tree_site_selection_follows_width_necessity_and_the_env_list() {
        let all = |b| SgTreeSites {
            attn: b,
            rms: b,
            w4: b,
        };
        assert_eq!(sg_tree_sites_from(None, Some(32)), all(false));
        assert_eq!(sg_tree_sites_from(None, None), all(true));
        assert_eq!(sg_tree_sites_from(None, Some(64)), all(true));
        assert_eq!(sg_tree_sites_from(Some("1"), Some(32)), all(true));
        assert_eq!(sg_tree_sites_from(Some("all"), Some(32)), all(true));
        assert_eq!(
            sg_tree_sites_from(Some("attn"), Some(32)),
            SgTreeSites {
                attn: true,
                rms: false,
                w4: false
            }
        );
        assert_eq!(
            sg_tree_sites_from(Some("rms,w4"), Some(32)),
            SgTreeSites {
                attn: false,
                rms: true,
                w4: true
            }
        );
        assert_eq!(sg_tree_sites_from(Some("rms"), None), all(true));
        assert_eq!(sg_tree_sites_from(Some("0"), Some(32)), all(false));
    }
}
