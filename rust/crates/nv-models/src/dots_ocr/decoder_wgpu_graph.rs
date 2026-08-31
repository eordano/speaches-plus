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
    dgk_kc[kat] = dgk_src[src + dgk_p.k_off_words + i];
    dgk_vc[at] = dgk_src[src + dgk_p.v_off_words + i];
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

fn dga_kval(idx: u32) -> f32 {
    let w = dga_k[idx >> 1u];
    return select(bf16_lo(w), bf16_hi(w), (idx & 1u) == 1u);
}

fn dga_vval(idx: u32) -> f32 {
    let w = dga_v[idx >> 1u];
    return select(bf16_lo(w), bf16_hi(w), (idx & 1u) == 1u);
}

fn dga_score(j: u32, kvh: u32, hd: u32) -> f32 {
    let words = hd >> 1u;
    var a = 0.0;
    for (var w = 0u; w < words; w = w + 1u) {
        let kw = dga_k[dg_kword(j, kvh, w, dga_p.n_kv_heads * words, words, dga_p.kv_capacity)];
        a = fma(dga_qs[w * 2u], bf16_lo(kw), a);
        a = fma(dga_qs[w * 2u + 1u], bf16_hi(kw), a);
    }
    return a;
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

var<workgroup> dgx_qs: array<f32, DG_ATTN_MAX_HEAD_DIM>;
var<workgroup> dgx_acc: array<f32, DG_ATTN_MAX_HEAD_DIM>;
var<workgroup> dgx_pr: array<f32, DG_ATTN_WG>;
var<workgroup> dgx_red: array<f32, DG_ATTN_WG>;

fn dgx_kval(idx: u32) -> f32 {
    let w = dgx_k[idx >> 1u];
    return select(bf16_lo(w), bf16_hi(w), (idx & 1u) == 1u);
}

fn dgx_vval(idx: u32) -> f32 {
    let w = dgx_v[idx >> 1u];
    return select(bf16_lo(w), bf16_hi(w), (idx & 1u) == 1u);
}

fn dgx_score(j: u32, kvh: u32, hd: u32) -> f32 {
    let words = hd >> 1u;
    var a = 0.0;
    for (var w = 0u; w < words; w = w + 1u) {
        let kw = dgx_k[dg_kword(j, kvh, w, dgx_p.n_kv_heads * words, words, dgx_p.kv_capacity)];
        a = fma(dgx_qs[w * 2u], bf16_lo(kw), a);
        a = fma(dgx_qs[w * 2u + 1u], bf16_hi(kw), a);
    }
    return a;
}

var<workgroup> dgx_pvp: array<f32, 512>;

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
    let per = (total + dgx_p.n_splits - 1u) / dgx_p.n_splits;
    let lo = sp * per;
    let hi = min(lo + per, total);
    let ml = h * dgx_p.n_splits + sp;
    let lbase = dgx_p.n_q_heads * dgx_p.n_splits;
    let abase = 2u * lbase;

    if (lo >= hi) {
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
            for (var w = 0u; w < words; w = w + 4u) {
                let k0 = dgx_k[dg_kword(j, kvh, w, nkv * words, words, dgx_p.kv_capacity)];
                a0 = fma(dgx_qs[w * 2u], bf16_lo(k0), fma(dgx_qs[w * 2u + 1u], bf16_hi(k0), a0));
                let k1 = dgx_k[dg_kword(j, kvh, w + 1u, nkv * words, words, dgx_p.kv_capacity)];
                a1 = fma(dgx_qs[w * 2u + 2u], bf16_lo(k1), fma(dgx_qs[w * 2u + 3u], bf16_hi(k1), a1));
                let k2 = dgx_k[dg_kword(j, kvh, w + 2u, nkv * words, words, dgx_p.kv_capacity)];
                a2 = fma(dgx_qs[w * 2u + 4u], bf16_lo(k2), fma(dgx_qs[w * 2u + 5u], bf16_hi(k2), a2));
                let k3 = dgx_k[dg_kword(j, kvh, w + 3u, nkv * words, words, dgx_p.kv_capacity)];
                a3 = fma(dgx_qs[w * 2u + 6u], bf16_lo(k3), fma(dgx_qs[w * 2u + 7u], bf16_hi(k3), a3));
            }
            sv = ((a0 + a1) + (a2 + a3)) * dgx_p.scale;
        }
        dgx_red[lid] = sv;
        workgroupBarrier();
        for (var s = DG_ATTN_WG >> 1u; s > 0u; s = s >> 1u) {
            if (lid < s) {
                dgx_red[lid] = max(dgx_red[lid], dgx_red[lid + s]);
            }
            workgroupBarrier();
        }
        let mnew = max(m, dgx_red[0]);
        workgroupBarrier();
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
        for (var s = DG_ATTN_WG >> 1u; s > 0u; s = s >> 1u) {
            if (lid < s) {
                dgx_red[lid] = dgx_red[lid] + dgx_red[lid + s];
            }
            workgroupBarrier();
        }
        lsum = lsum * corr + dgx_red[0];
        let tmax = min(DG_ATTN_WG, hi - base);
        let tq = lid >> 6u;
        let wd = lid & 63u;
        var e0 = 0.0;
        var e1 = 0.0;
        var o0 = 0.0;
        var o1 = 0.0;
        if (wd < words) {
            let tend = min(tmax, (tq + 1u) * 64u);
            var t = tq * 64u;
            for (; t + 2u <= tend; t = t + 2u) {
                let v0 = dgx_v[((base + t) * nkv + kvh) * words + wd];
                e0 = fma(dgx_pr[t], bf16_lo(v0), e0);
                e1 = fma(dgx_pr[t], bf16_hi(v0), e1);
                let v1 = dgx_v[((base + t + 1u) * nkv + kvh) * words + wd];
                o0 = fma(dgx_pr[t + 1u], bf16_lo(v1), o0);
                o1 = fma(dgx_pr[t + 1u], bf16_hi(v1), o1);
            }
            if (t < tend) {
                let v0 = dgx_v[((base + t) * nkv + kvh) * words + wd];
                e0 = fma(dgx_pr[t], bf16_lo(v0), e0);
                e1 = fma(dgx_pr[t], bf16_hi(v0), e1);
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
    let per = (total + dgx_p.n_splits - 1u) / dgx_p.n_splits;
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
            let klo = bf16_lo(kw);
            let khi = bf16_hi(kw);
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
                let klo = bf16_lo(kw);
                let khi = bf16_hi(kw);
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
    if (h >= dgx_p.n_q_heads) {
        return;
    }
    let lid = tid.x;
    let hd = dgx_p.head_dim;
    let words = hd >> 1u;
    let ns = dgx_p.n_splits;
    let lbase = dgx_p.n_q_heads * ns;
    let abase = 2u * lbase;

    if (lid < ns) {
        dgxc_m[lid] = dgx_part[h * ns + lid];
        dgxc_l[lid] = dgx_part[lbase + h * ns + lid];
    }
    workgroupBarrier();
    var big = -3.0e38;
    for (var s = 0u; s < ns; s = s + 1u) {
        big = max(big, dgxc_m[s]);
    }
    if (lid < ns) {
        dgxc_w[lid] = exp(dgxc_m[lid] - big);
    }
    workgroupBarrier();
    var den = 0.0;
    for (var s = 0u; s < ns; s = s + 1u) {
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
        for (var s = 0u; s < ns; s = s + 1u) {
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
    return select(bf16_lo(w), bf16_hi(w), (elem & 1u) == 1u);
}

fn dgt_vtileval(tile_word: u32, elem: u32) -> f32 {
    let w = dgt_v[tile_word + (elem >> 1u)];
    return select(bf16_lo(w), bf16_hi(w), (elem & 1u) == 1u);
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
                let klo = bf16_lo(kw);
                let khi = bf16_hi(kw);
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
                let vv = select(bf16_lo(vw), bf16_hi(vw), (d & 1u) == 1u);
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
    dgk_kc[kat] = dgk_src[src + dgk_p.k_off_words + i];
    dgk_vc[at] = dgk_src[src + dgk_p.v_off_words + i];
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
    dgk_kc[kat] = dgk_src[src + dgk_p.k_off_words + i];
    dgk_vc[at] = dgk_src[src + dgk_p.v_off_words + i];
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

fn dga_range_attend(slot: u32, h: u32, lid: u32, start: u32, total: u32) {
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
        dgf_kf0[kdst] = bf16_lo(kw);
        dgf_kf0[kdst + 1u] = bf16_hi(kw);
        dgf_vt0[d * c16 + pos] = bf16_lo(vw);
        dgf_vt0[(d + 1u) * c16 + pos] = bf16_hi(vw);
    } else {
        dgf_kf1[kdst] = bf16_lo(kw);
        dgf_kf1[kdst + 1u] = bf16_hi(kw);
        dgf_vt1[d * c16 + pos] = bf16_lo(vw);
        dgf_vt1[(d + 1u) * c16 + pos] = bf16_hi(vw);
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
    pad0: u32,
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
const DSC_STEP_POS: u32 = 1u;

fn dsc_h16(w: u32) -> u32 {
    return pack2x16float(vec2<f32>(bf16_lo(w), bf16_hi(w)));
}

@compute @workgroup_size(64)
fn dg_scvt_q(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= dsc_p.q_words) {
        return;
    }
    dsc_qh[gid.y * dsc_p.q_words + w] = dsc_h16(dsc_qsrc[gid.y * dsc_p.qkv_words + w]);
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
    dsc_kh[o] = dsc_h16(dsc_ksrc[kidx]);
    dsc_vh[o] = dsc_h16(dsc_vsrc[pos * dsc_p.kv_words + w]);
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
    dsc_out[slot * dsc_p.heads_words + w] = bf16_pack(v.x, v.y);
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
"#;

pub const NVFP4_GEMV_ROWS_PER_WORKGROUP: usize = 8;

pub const NVFP4_GEMV_SPENDS_32_LANES_ON_A_ROW_SO_A_1536_WIDE_K_STILL_AMORTIZES_THE_REDUCTION:
    &str = "A decode GEMV row of k=1536 is 96 nvfp4 blocks. Spread over 128 lanes that is under \
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

pub const MSL_GEMV_ENV: &str = "NV_DOTS_MSL_GEMV";

pub fn msl_gemv_enabled() -> bool {
    let _ = decode_gemv_msl::MSL_GEMV_EXISTS_BECAUSE_THE_FIVE_ISSUE_SLOT_BOUND_WAS_NAGAS_NOT_THE_MACHINES;
    !matches!(
        std::env::var(MSL_GEMV_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

pub const DEC_W4_PAYS_ONLY_AFTER_THE_BF16_GEMV_REACHED_ITS_EQUILIBRIUM_AND_ONLY_CONSISTENTLY:
    &str = "The first w4 verdict -- this adapter never pays for the bytes it saves -- was earned \
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
            return vec![big, PREFILL_BIG_MID_1024_IS_THE_CHUNKED_LADDER_KNEE, PREFILL_M_WIDE, m];
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
        .unwrap_or(PREFILL_M_DEFAULT_32_IS_THE_LADDER_KNEE_19_6_S_PAGE_64_FLAT_128_COSTS_SMALL_PAGES);
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
    if want > 1 && (group % want != 0 || cfg.head_dim() > ATTN_FOLD_MAX_HEAD_DIM) {
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
    let _ = PF_STEEL_PROJ_DELETES_THE_F32_WEIGHT_COPIES_BECAUSE_BF16_TO_F16_IS_EXACT_IN_WEIGHT_RANGE;
    !matches!(
        std::env::var(PF_STEEL_PROJ_ENV).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
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

pub const PF_ABLATE_DELETES_WORK_BECAUSE_SUMMED_PASS_TIMESTAMPS_INVERTED_THE_PREFILL_PROFILE:
    &str = "Per-pass timestamps read prefill as 63 percent dg_attn_reg and 36 percent \
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
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
];

fn dg_lora_leaf_path(leaf: &str) -> String {
    match leaf {
        "q_proj" | "k_proj" | "v_proj" | "o_proj" => format!("self_attn.{leaf}"),
        _ => format!("mlp.{leaf}"),
    }
}

fn dg_pack_bf16_rows(t: &candle_core::Tensor, n: usize, k: usize) -> Result<Vec<u32>> {
    anyhow::ensure!(t.dims() == [n, k], "lora factor {:?} != [{n}, {k}]", t.dims());
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
                let module =
                    format!("model.language_model.layers.{li}.{}", dg_lora_leaf_path(leaf));
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
            "lora adapter {} matched no dots graph projection; expected modules like \
             model.language_model.layers.0.self_attn.q_proj after normalization",
            dir.display()
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
            y_off % 2 == 0 && y_row % 2 == 0 && g.n % 2 == 0 && g.k % 2 == 0,
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
    rmsres: String,
    rope: String,
    rope_act16: String,
    fused: String,
    scvt: String,
}

impl Sources {
    fn new(fold: usize, kt: bool) -> Self {
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
            );
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
            rmsres: compose(wk::rmsnorm_residual::WGSL),
            rope: compose(&rope_raw),
            rope_act16: compose(&rope_raw.replace(
                "const DGACT16: u32 = 0u;",
                "const DGACT16: u32 = 1u;",
            )),
            fused: compose(DG_ATTN_FUSED_WGSL).replace(
                "const DGF_KT: u32 = 0u;",
                &format!("const DGF_KT: u32 = {}u;", u32::from(kt)),
            ),
            scvt: compose(DG_STEEL_CVT_WGSL).replace(
                "const DSC_KT: u32 = 0u;",
                &format!("const DSC_KT: u32 = {}u;", u32::from(kt)),
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
    Bf16 {
        w: wgpu::Buffer,
        n: usize,
        k: usize,
    },
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
    anyhow::ensure!(t.dims() == shape, "{name}: shape {:?} != {shape:?}", t.dims());
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

fn drr_pack(lo: f32, hi: f32) -> u32 {
    if (DRR_F16OUT == 1u) {
        return pack2x16float(vec2<f32>(lo, hi));
    }
    return bf16_pack(lo, hi);
}

var<workgroup> drr_warp: array<f32, 8>;

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
    let wsum = subgroupAdd(local);
    if ((lid & 31u) == 0u) {
        drr_warp[lid >> 5u] = wsum;
    }
    workgroupBarrier();
    let tot = ((drr_warp[0] + drr_warp[1]) + (drr_warp[2] + drr_warp[3]))
        + ((drr_warp[4] + drr_warp[5]) + (drr_warp[6] + drr_warp[7]));
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

    let wsum = subgroupAdd(local);
    if ((lid & 31u) == 0u) {
        drr_warp[lid >> 5u] = wsum;
    }
    workgroupBarrier();
    let tot = ((drr_warp[0] + drr_warp[1]) + (drr_warp[2] + drr_warp[3]))
        + ((drr_warp[4] + drr_warp[5]) + (drr_warp[6] + drr_warp[7]));
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

var<workgroup> w4g_rows: array<f32, 32>;

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
    let rsum = subgroupAdd(acc);
    if ((tid & 31u) == 0u) {
        w4g_rows[tid >> 5u] = rsum;
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
        nv_quant::mlx_affine::dequantize_mlx_affine_linear(&parts.0, &parts.1, &parts.2, n, k, W4_GROUP as usize)
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_bits())
            .collect()
    } else {
        host
    };
    (host2, Some(parts))
}

fn upload_affine_w4(
    b: &mut Builder,
    label: &str,
    parts: &W4Parts,
    n: usize,
    k: usize,
) -> DgAffine {
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
        &compose(DG_GEMV_W4_WGSL).replace(
            "const W4G_ROWS: u32 = 8u;",
            &format!("const W4G_ROWS: u32 = {rows}u;"),
        ),
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

fn act_over(
    b: &mut Builder,
    label: &str,
    x: wgpu::Buffer,
    k: usize,
    quant: DotsGraphQuant,
) -> Act {
    let (qb, sb) = match quant {
        DotsGraphQuant::Bf16 | DotsGraphQuant::AffineW4 => (4u64, 4u64),
        DotsGraphQuant::Nvfp4 => (
            (k / 2) as u64,
            (k / NVFP4_BLOCK).div_ceil(4) as u64 * 4,
        ),
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
                let src4 = s
                    .rope
                    .replace("const DG4_ROWS: u32 = 2u;", &format!("const DG4_ROWS: u32 = {rows}u;"));
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
            anyhow::ensure!(w.k == x.k, "{label}: gemv k {} != activation k {}", w.k, x.k);
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
            anyhow::ensure!(w.k == x.k, "{label}: gemv k {} != activation k {}", w.k, x.k);
            anyhow::ensure!(
                !out_f32,
                "{label}: the nvfp4 gemv writes bf16 pairs, so an f32 sink needs the bf16 arm"
            );
            let k_blocks = w.k / NVFP4_BLOCK;
            let grid = b.grid1(w.n.div_ceil(NVFP4_GEMV_ROWS_PER_WORKGROUP) as u64, 1);
            let _ = NVFP4_ACTIVATION_GLOBAL_IS_A_POWER_OF_TWO_SO_IT_ADDS_NO_ROUNDING;
            let _ = NVFP4_GEMV_SPENDS_32_LANES_ON_A_ROW_SO_A_1536_WIDE_K_STILL_AMORTIZES_THE_REDUCTION;
            let _ = NVFP4_GEMV_CALLS_ONE_DOT4I8PACKED_PER_STATEMENT_BECAUSE_THE_MSL_POLYFILL_COLLIDES;
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
            &compose(&DG_RMSRES_WGSL.replace(
                "const DRR_F16OUT: u32 = 0u;",
                "const DRR_F16OUT: u32 = 1u;",
            )),
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
    s: &Sources,
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
        &compose(&DG_RMSRES_WGSL.replace(
            "const DRR_F16OUT: u32 = 0u;",
            &format!("const DRR_F16OUT: u32 = {}u;", u32::from(f16_out)),
        )),
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
        &[
            (10, src),
            (11, &step.step),
            (12, &p),
            (13, kc),
            (14, vc),
        ],
        ((kv_words as u32).div_ceil(64).max(1), slots as u32, 1),
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
        b, s, label, gate_up, out, inter, slots, src_slot_words, dst_slot_words, false,
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

#[allow(clippy::too_many_arguments)]
fn region_rope_pass(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    site: RopeSite<'_>,
    step: &StepBuffers,
    bpos: &wgpu::Buffer,
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
    b.make_pass(
        label,
        &s.rope,
        "dg_bias_rope_b",
        &[
            (0, site.src),
            (1, site.bias),
            (2, site.cos),
            (3, site.sin),
            (5, &p),
            (6, site.dst),
            (7, bpos),
        ],
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
        if fold > 1 { "dg_attn_fold" } else { "dg_attn_split" },
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
        &[(64, &p), (65, part), (66, out)],
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
        (
            n_q as u32,
            (chunk / ATTN_TILE_QUERY_ROWS) as u32,
            1,
        ),
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

pub const ATTN_FUSED_PV_PLAN_TM2_TN1_SG2_IS_THE_M768_N128_LADDER_KNEE:
    (u32, u32, u32, u32) = (2, 1, 2, 2);

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
    for (g, (kf, qx, sy)) in [
        (&mir.kf0, &fb.qa, &fb.sa),
        (&mir.kf1, &fb.qb, &fb.sb),
    ]
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
    for (g, (vt, sx, oy)) in [
        (&mir.vt0, &fb.sa, &fb.oa),
        (&mir.vt1, &fb.sb, &fb.ob),
    ]
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
    pad0: u32,
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
        "dg_scvt_kv",
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
            wgpu::BindGroupEntry { binding: 0, resource: sb.qh.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: kh.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: vh.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: sb.oh.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: sb.sp[pfi].as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: sb.mp.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: sb.dummy.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 7, resource: sb.dummy.as_entire_binding() },
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
    pls: std::sync::Arc<super::vision_attn_msl::SteelGemmPipelines>,
}

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
    w_h: &wgpu::Buffer,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sb: &SteelProjBufs,
    m: usize,
    n: usize,
    k: usize,
) -> Result<()> {
    let cfg = PF_STEEL_PROJ_CONFIG_IS_THE_VISION_LADDERS_CROWN;
    anyhow::ensure!(
        k.is_multiple_of(2 * cfg.2 as usize),
        "{label}: k {k} is not a whole even number of {}-deep steel k-blocks",
        cfg.2
    );
    let kw = k / 2;
    let p = b.uni(
        &format!("{label}-cp"),
        DgScvtParams {
            m: m as u32,
            qkv_words: kw as u32,
            q_words: kw as u32,
            kv_words: 0,
            capacity: 0,
            heads_words: 0,
            pad0: 0,
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
    let gp = b.uni(
        &format!("{label}-gp"),
        super::vision_attn_msl::steel_gemm_params(m, n, k, n, cfg),
    );
    let pipeline = sb
        .pls
        .pick(cfg)
        .ok_or_else(|| anyhow::anyhow!("{label}: steel gemm config {cfg:?} missing"))?;
    let bind = b.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(label),
        layout: &sb.pls.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: sb.xh.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: w_h.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: y.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: gp.as_entire_binding() },
        ],
    });
    b.push_prebuilt(
        "dv_steel_gemm_nt",
        pipeline,
        bind,
        super::vision_attn_msl::steel_gemm_grid(m, n, cfg),
    );
    Ok(())
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
     keep the chunk-vs-steps suite bitwise. The next rate lever on record is the steel read: \
     mlx stages both operands through threadgroup memory with big k blocks, and workgroup-space \
     coop loads are proven live on this adapter by the flash round's kernels.";

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
        (
            (hidden_words as u32).div_ceil(64).max(1),
            slots as u32,
            1,
        ),
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

    let s = Sources::new(1, false);
    let mut b = Builder::new(ctx);
    let wbits: Vec<u16> = (0..n * k)
        .map(|i| half::bf16::from_f32(((i % 97) as f32 - 48.0) / 512.0).to_bits())
        .collect();
    let xbits: Vec<u16> = (0..m * k)
        .map(|i| half::bf16::from_f32(((i % 89) as f32 - 44.0) / 256.0).to_bits())
        .collect();
    let f32_of = |v: &[u16]| -> Vec<f32> {
        v.iter().map(|b| f32::from_bits((*b as u32) << 16)).collect()
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
    let c: Vec<f32> = dispatch::read_back(ctx, &y8, m * n)
        .map_err(|e| anyhow::anyhow!("m8 y readback: {e}"))?;
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
    let s = Sources::new(1, false);
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

    let dq = nv_quant::mlx_affine::dequantize_mlx_affine_linear(&parts.0, &parts.1, &parts.2, n, k, gs);
    let dqb: Vec<u16> = dq.iter().map(|v| half::bf16::from_f32(*v).to_bits()).collect();
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

    let got_words: Vec<u32> = dispatch::read_back(ctx, &y, n / 2)
        .map_err(|e| anyhow::anyhow!("w4 y readback: {e}"))?;
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
            want += parts.1[row * groups + g] as f64 * qdot
                + parts.2[row * groups + g] as f64 * xsum;
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
    let s = Sources::new(1, false);
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
            let bits = if i.is_multiple_of(2) { w & 0xffff } else { w >> 16 };
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

struct SwapSite {
    start: usize,
    len: usize,
    rpf: Option<Pass>,
    bat: Option<Pass>,
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
    validated: bool,
    bat_validated: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct RegionRow {
    pub token: u32,
    pub base: u32,
    pub len: u32,
}

pub struct DotsDecoderWgpuGraph {
    ctx: &'static WgpuContext,
    cfg: DotsDecoderConfig,
    quant: DotsGraphQuant,
    max_seq: usize,
    pos: usize,
    repack_s: f64,
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
        let s = Sources::new(fold, kv_transpose_enabled());
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
            && max_seq > ATTN_SPLIT_MIN_CONTEXT_IS_WHERE_ONE_WORKGROUP_PER_HEAD_STOPS_FILLING_THE_CORES;
        let attn_part = split_on.then(|| {
            b.zeros(
                "dg-attn-part",
                (n_q * n_splits * (head_dim + 2) * 4) as u64,
            )
        });
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

        let embed = load_bf16(weights, "model.embed_tokens.weight", &[vocab, hidden])?;
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
        let head_host = if tied {
            drop(embed);
            Vec::new()
        } else {
            drop(embed);
            load_bf16(weights, "lm_head.weight", &[vocab, hidden])?
        };

        let w4_dec = quant == DotsGraphQuant::AffineW4
            || (quant == DotsGraphQuant::Bf16 && dec_w4_enabled());
        let (head_host, lm_w4) = if w4_dec && !tied && splits.len() == 1 {
            prep_lin_w4(head_host, vocab, hidden, quant, true)
        } else {
            (head_host, None)
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

        let ln0 = load_bf16(weights, &layer_tensor(0, "input_layernorm.weight"), &[hidden])?;
        let ln0w = b.upload_u32("dg-ln0", &pack_pairs(&ln0));
        push_rmsnorm_fmt(
            &mut b, &s, "dg-rms0", &res, &ln0w, &normed.x, hidden, eps, 1, act_f16,
        )?;

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
                &mut b, &s, "dg-pf-rms0", &pf.res, &ln0w, &pf.normed, hidden, eps, pf.m,
            )?;
            b.pf_target = None;
        }

        let tile_shaped = head_dim <= ATTN_TILE_MAX_HEAD_DIM
            && !pf.is_empty()
            && pf
                .iter()
                .all(|p| p.m.is_multiple_of(ATTN_TILE_QUERY_ROWS));
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
            && pf
                .iter()
                .all(|p| p.m.is_multiple_of(super::vision_attn_msl::STEEL_BQ as usize))
            && super::prefill_attn_msl::causal_pipelines(ctx).is_ok();
        let fused_prefill = attn_fused_enabled()
            && !steel_prefill
            && head_dim == ATTN_FUSED_HEAD_DIM
            && n_kv == 2
            && !pf.is_empty()
            && pf
                .iter()
                .all(|p| p.m.is_multiple_of(ATTN_FUSED_QUERY_ROWS))
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
                sp.push(steel_params_buffer(&mut b, &format!("dg-steel-sp{i}"), &sap));
                pu.push(b.uni(
                    &format!("dg-steel-pu{i}"),
                    DgScvtParams {
                        m: m as u32,
                        qkv_words: qkv_words as u32,
                        q_words: q_words as u32,
                        kv_words: kv_words as u32,
                        capacity: max_seq as u32,
                        heads_words: q_words as u32,
                        pad0: 0,
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
        let steel_proj_bufs = if pf_steel_proj_enabled()
            && quant.bf16_like()
            && !pf.is_empty()
            && hidden.is_multiple_of(2 * PF_STEEL_PROJ_CONFIG_IS_THE_VISION_LADDERS_CROWN.2 as usize)
            && inter.is_multiple_of(2 * PF_STEEL_PROJ_CONFIG_IS_THE_VISION_LADDERS_CROWN.2 as usize)
        {
            match super::vision_attn_msl::steel_gemm_pipelines(ctx) {
                Ok(pls) if pls.pick(PF_STEEL_PROJ_CONFIG_IS_THE_VISION_LADDERS_CROWN).is_some() => {
                    let maxm = pf.iter().map(|p| p.m).max().unwrap_or(0);
                    let kmax = hidden.max(inter);
                    Some(SteelProjBufs {
                        xh: b.zeros("dg-steelproj-xh", (maxm * kmax / 2 * 4) as u64),
                        pls,
                    })
                }
                _ => None,
            }
        } else {
            None
        };
        let steel_proj = steel_proj_bufs.is_some();
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
            if quant == DotsGraphQuant::Nvfp4 && !nvfp4_mrow4_enabled() {
                push_quant(&mut b, &s, &format!("dg-l{li}-qn"), &normed, &sl)?;
            }
            let lora_layer = dg_lora
                .as_ref()
                .map(|l| upload_dg_lora_layer(&mut b, li, &l.layers[li], dg_lora_pf_m));

            let qkv_rows = q_rows + 2 * kv_rows;
            let mut qkv_host = Vec::with_capacity(qkv_rows * hidden);
            for (name, rows) in [
                ("self_attn.q_proj.weight", q_rows),
                ("self_attn.k_proj.weight", kv_rows),
                ("self_attn.v_proj.weight", kv_rows),
            ] {
                qkv_host.extend_from_slice(&load_bf16(
                    weights,
                    &layer_tensor(li, name),
                    &[rows, hidden],
                )?);
            }
            let (qkv_host, qkv_w4) = prep_lin_w4(qkv_host, qkv_rows, hidden, quant, w4_dec);
            let qkv_aff = qkv_w4
                .map(|pw| upload_affine_w4(&mut b, &format!("dg-l{li}-qkv"), &pw, qkv_rows, hidden));
            let qkvw = upload_lin(
                &mut b,
                &format!("dg-l{li}-qkvw"),
                &qkv_host,
                qkv_rows,
                hidden,
                quant,
            )?;
            let proj_coop =
                coop_prefill_enabled() && coop_proj_enabled() && quant.bf16_like();
            let qkvw_f32 = (proj_coop && !steel_proj)
                .then(|| upload_f32_of_bf16(&mut b, &format!("dg-l{li}-qkvw-f32"), &qkv_host));
            let qkvw_h = steel_proj
                .then(|| upload_f16_of_bf16(&mut b, &format!("dg-l{li}-qkvw-h"), &qkv_host));
            drop(qkv_host);
            let o_host = load_bf16(
                weights,
                &layer_tensor(li, "self_attn.o_proj.weight"),
                &[hidden, q_rows],
            )?;
            let (o_host, o_w4) = prep_lin_w4(o_host, hidden, q_rows, quant, w4_dec);
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
            let ow_h = steel_proj
                .then(|| upload_f16_of_bf16(&mut b, &format!("dg-l{li}-ow-h"), &o_host));
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
                Some(aff) => {
                    push_gemv_w4(
                    &mut b,
                    &format!("dg-l{li}-qkvw4"),
                    aff,
                    &normed.x,
                    &qkv,
                    true,
                    act_f16,
                )?
                }
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
                    &[("q_proj", 0), ("k_proj", q_rows), ("v_proj", q_rows + kv_rows)],
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
            let has_bias = cfg.attention_bias
                && bias_names
                    .iter()
                    .all(|n| weights.has(&layer_tensor(li, n)));
            let bias_buf = if has_bias {
                let mut host = Vec::with_capacity(qkv_rows);
                for (name, rows) in bias_names.iter().zip([q_rows, kv_rows, kv_rows]) {
                    host.extend_from_slice(&load_bf16(
                        weights,
                        &layer_tensor(li, name),
                        &[rows],
                    )?);
                }
                b.upload_u32(&format!("dg-l{li}-qkvbias"), &pack_pairs(&host))
            } else {
                zero_bias.clone()
            };
            push_bias_rope(
                &mut b,
                &s,
                &format!("dg-l{li}-qkvrope"),
                RopeSite {
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
                },
                &step,
            )?;

            let kc = b.state_zeros(
                &format!("dg-l{li}-kc"),
                (max_seq * (kv_rows / 2) * 4) as u64,
            );
            let vc = b.state_zeros(
                &format!("dg-l{li}-vc"),
                (max_seq * (kv_rows / 2) * 4) as u64,
            );
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

            if quant == DotsGraphQuant::Nvfp4 && !nvfp4_mrow4_enabled() {
                push_quant(&mut b, &s, &format!("dg-l{li}-qp"), &normed_post, &sl)?;
            }
            let mut gu_host = Vec::with_capacity(2 * inter * hidden);
            for name in ["mlp.gate_proj.weight", "mlp.up_proj.weight"] {
                gu_host.extend_from_slice(&load_bf16(
                    weights,
                    &layer_tensor(li, name),
                    &[inter, hidden],
                )?);
            }
            let (gu_host, gu_w4) = prep_lin_w4(gu_host, 2 * inter, hidden, quant, w4_dec);
            let gu_aff = gu_w4
                .map(|pw| upload_affine_w4(&mut b, &format!("dg-l{li}-gu"), &pw, 2 * inter, hidden));
            let gu = upload_lin(
                &mut b,
                &format!("dg-l{li}-gateup"),
                &gu_host,
                2 * inter,
                hidden,
                quant,
            )?;
            let gu_f32 = (coop_prefill_enabled() && quant.bf16_like() && !steel_proj)
                .then(|| upload_f32_of_bf16(&mut b, &format!("dg-l{li}-gateup-f32"), &gu_host));
            let gu_h = steel_proj
                .then(|| upload_f16_of_bf16(&mut b, &format!("dg-l{li}-gateup-h"), &gu_host));
            drop(gu_host);
            let down_host = load_bf16(
                weights,
                &layer_tensor(li, "mlp.down_proj.weight"),
                &[hidden, inter],
            )?;
            let (down_host, down_w4) = prep_lin_w4(down_host, hidden, inter, quant, w4_dec);
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
            if quant == DotsGraphQuant::Nvfp4 && !nvfp4_mrow4_enabled() {
                push_quant(&mut b, &s, &format!("dg-l{li}-qa"), &act, &sl)?;
            }
            match &down_aff {
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
                        qkvw_h.as_ref().unwrap(),
                        &pf.normed,
                        &pf.qkv,
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
                        &[("q_proj", 0), ("k_proj", q_rows), ("v_proj", q_rows + kv_rows)],
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
                push_bias_rope(
                    &mut b,
                    &s,
                    &format!("dg-pf-l{li}-rope"),
                    rope_site,
                    &step,
                )?;
                if let Some((bpos, _, _, _)) = rg_bufs.as_ref().filter(|_| Some(pfi) == region_wi)
                {
                    let bat = region_rope_pass(
                        &mut b,
                        &s,
                        &format!("dg-rg-l{li}-rope"),
                        rope_site,
                        &step,
                        bpos,
                    )?;
                    rg_sites[pfi].push(SwapSite {
                        start: rg_rope_start,
                        len: b.pf_lists[pfi].len() - rg_rope_start,
                        rpf: None,
                        bat: Some(bat),
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
                if rg_bufs.is_some() {
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
                    let bat = if Some(pfi) == region_wi {
                        let (_, bslot, _, _) = rg_bufs.as_ref().unwrap();
                        Some(region_kv_pass(
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
                        )?)
                    } else {
                        None
                    };
                    rg_sites[pfi].push(SwapSite {
                        start: rg_kv_start,
                        len: b.pf_lists[pfi].len() - rg_kv_start,
                        rpf: Some(rpf),
                        bat,
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
                if rg_bufs.is_some() {
                    let rpf = region_attn_pass(
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
                    )?;
                    let bat = if Some(pfi) == region_wi {
                        let (_, _, brange, _) = rg_bufs.as_ref().unwrap();
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
                    rg_sites[pfi].push(SwapSite {
                        start: rg_attn_start,
                        len: b.pf_lists[pfi].len() - rg_attn_start,
                        rpf: Some(rpf),
                        bat,
                    });
                }
                if pf_ablated("proj") {
                } else if let Some(sp) = &steel_proj_bufs {
                    push_proj_steel(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-osteel"),
                        ow_h.as_ref().unwrap(),
                        &pf.heads,
                        &pf.y_f32,
                        sp,
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
                    pf.m >= PREFILL_M_WIDE && coop_plan().check_shape(
                        pf.m as u32,
                        (2 * inter) as u32,
                        hidden as u32,
                    ).is_ok()
                });
                if !pf_ablated("proj") {
                    if let Some(sp) = &steel_proj_bufs {
                        push_proj_steel(
                            &mut b,
                            &s,
                            &format!("dg-pf-l{li}-gusteel"),
                            gu_h.as_ref().unwrap(),
                            &pf.normed_post,
                            &pf.y_f32,
                            sp,
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
                    push_proj_steel(
                        &mut b,
                        &s,
                        &format!("dg-pf-l{li}-dsteel"),
                        down_h.as_ref().unwrap(),
                        &pf.act,
                        &pf.x_f32,
                        sp,
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

            let last = li + 1 == layers;
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
        let lm_aff = lm_w4
            .map(|pw| upload_affine_w4(&mut b, "dg-lmhead", &pw, vocab, hidden));
        match &lm_aff {
            Some(aff) => {
                push_gemv_w4(&mut b, "dg-lmhead-w4", aff, &final_x, &logits, true, act_f16)?;
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
                    push_gemv(
                        &mut b,
                        &s,
                        "dg-lmhead",
                        &lin,
                        &head_x,
                        &logits,
                        true,
                        off,
                    )?;
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
                    validated: false,
                    bat_validated: false,
                })
            }
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
            ..
        } = b;
        Ok(Self {
            ctx,
            cfg,
            quant,
            max_seq,
            pos: 0,
            repack_s: 0.0,
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

    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for buf in &self.state_buffers {
            enc.clear_buffer(buf, 0, None);
        }
        self.ctx.queue.submit([enc.finish()]);
        Ok(())
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
                let bits: Vec<u16> = row.iter().map(|v| half::bf16::from_f32(*v).to_bits()).collect();
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
        let split = !self.split_passes.is_empty()
            && self.pos + 1
                >= ATTN_SPLIT_MIN_CONTEXT_IS_WHERE_ONE_WORKGROUP_PER_HEAD_STOPS_FILLING_THE_CORES;
        let mut run: Vec<&Pass> = Vec::with_capacity(end - start + self.split_passes.len());
        let mut k = 0usize;
        for i in start..end {
            let is_attn = k < self.attn_idx.len() && self.attn_idx[k] == i;
            if is_attn {
                k += 1;
                if split {
                    run.extend(self.split_passes[2 * (k - 1)..2 * k].iter());
                    continue;
                }
            }
            run.push(&self.passes[i]);
        }
        let labels: Vec<&str> = if dispatch::profile::enabled() {
            run.iter().map(|p| p.entry.as_str()).collect()
        } else {
            Vec::new()
        };
        let cb = dispatch::encode_pass_list_labeled(
            self.ctx,
            run.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
            &labels,
        );
        self.ctx.queue.submit([cb]);
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
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

    pub fn read_logits(&self) -> Result<Vec<f32>> {
        dispatch::read_back(self.ctx, &self.logits, self.vocab)
            .map_err(|e| anyhow::anyhow!("logit readback: {e}"))
    }

    fn read_token_and_logits(&self) -> Result<(u32, Vec<f32>)> {
        let t: Vec<u32> = dispatch::read_back(self.ctx, &self.token_out, 1)
            .map_err(|e| anyhow::anyhow!("token readback: {e}"))?;
        let l: Vec<f32> = dispatch::read_back(self.ctx, &self.logits, self.vocab)
            .map_err(|e| anyhow::anyhow!("logit readback: {e}"))?;
        Ok((t[0], l))
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
        let t_repack = std::time::Instant::now();
        let mut toks = vec![0u32; m];
        let mut mask = vec![0u32; m];
        let mut splice = vec![0u32; m * words];
        for (i, (tok, row)) in chunk.iter().enumerate() {
            match row {
                Some(r) => {
                    anyhow::ensure!(
                        r.len() == hidden,
                        "supplied hidden row has {} values, want {hidden}",
                        r.len()
                    );
                    mask[i] = 1;
                    let bits: Vec<u16> =
                        r.iter().map(|v| half::bf16::from_f32(*v).to_bits()).collect();
                    splice[i * words..(i + 1) * words].copy_from_slice(&pack_pairs(&bits));
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
            self.ctx
                .queue
                .write_buffer(&pf.splice, 0, bytemuck::cast_slice(&splice));
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
            if let Some(e) = pollster::block_on(scope.pop()) {
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
            let repl = if bat { site.bat.as_ref() } else { site.rpf.as_ref() };
            let Some(repl) = repl else {
                continue;
            };
            while i < site.start {
                run.push(&list[i]);
                i += 1;
            }
            run.push(repl);
            i += site.len;
        }
        while i < list.len() {
            run.push(&list[i]);
            i += 1;
        }
        run
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
    ) -> Result<()> {
        let m = self.pf_widths[wi];
        anyhow::ensure!(chunk.len() == m, "rpf chunk is {} rows, want {m}", chunk.len());
        let hidden = self.cfg.hidden_size;
        let words = hidden / 2;
        let mut toks = vec![0u32; m];
        let mut mask = vec![0u32; m];
        let mut splice = vec![0u32; m * words];
        for (i, (tok, row)) in chunk.iter().enumerate() {
            match row {
                Some(r) => {
                    anyhow::ensure!(
                        r.len() == hidden,
                        "supplied hidden row has {} values, want {hidden}",
                        r.len()
                    );
                    mask[i] = 1;
                    let bits: Vec<u16> =
                        r.iter().map(|v| half::bf16::from_f32(*v).to_bits()).collect();
                    splice[i * words..(i + 1) * words].copy_from_slice(&pack_pairs(&bits));
                }
                None => {
                    anyhow::ensure!((*tok as usize) < self.vocab, "token {tok} out of vocab");
                    toks[i] = *tok;
                }
            }
        }
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
                .write_buffer(&pf.splice, 0, bytemuck::cast_slice(&splice));
        }
        self.step.write_base(
            self.ctx,
            0,
            lpos as u32,
            (lpos + m) as u32,
            base as u32,
        );
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
            if let Some(e) = pollster::block_on(scope.pop()) {
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
        let Some(rg) = self.region.as_ref() else {
            anyhow::bail!(
                "region machinery needs the 32-row prefill width and a span of at least 64"
            );
        };
        let stride = rg.stride;
        let slots = REGION_SLOTS_32_IS_THE_PF_KNEE_WIDTH_THE_BATCHED_STEP_RIDES;
        anyhow::ensure!(slot < slots, "region slot {slot} out of {slots}");
        anyhow::ensure!(prompt_tokens.len() > 1, "region prompt needs at least two tokens");
        let base = slot * stride;
        let hidden = self.cfg.hidden_size;
        let n_vis = match vision_features {
            None => 0,
            Some(f) => {
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
        };
        let gend = prompt_tokens.len() - 1;
        anyhow::ensure!(
            gend < stride,
            "region prompt of {gend} rows does not fit the {stride}-row stride minus its \
             reserved pad row (lower NV_DOTS_CROP_MAX_PIXELS or raise NV_DOTS_MAX_SEQ)"
        );
        let row_at = |i: usize| -> Option<&[f32]> {
            vision_features
                .filter(|_| i >= vision_start && i < vision_start + n_vis)
                .map(|f| &f[(i - vision_start) * hidden..(i - vision_start + 1) * hidden])
        };
        let widths = self.pf_widths.clone();
        let usable: Vec<(usize, usize)> = widths
            .iter()
            .copied()
            .enumerate()
            .filter(|(_, m)| *m < stride)
            .collect();
        anyhow::ensure!(!usable.is_empty(), "no prefill width fits the region stride");
        let mut lpos = 0usize;
        let mut i = 0usize;
        for (wi, m) in usable.iter().copied() {
            while i + m <= gend && lpos + m < stride {
                let chunk: Vec<(u32, Option<&[f32]>)> = (i..i + m)
                    .map(|j| (prompt_tokens[j], row_at(j)))
                    .collect();
                self.rpf_chunk(wi, &chunk, base, lpos)?;
                lpos += m;
                i += m;
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
            let mut chunk: Vec<(u32, Option<&[f32]>)> = (i..gend)
                .map(|j| (prompt_tokens[j], row_at(j)))
                .collect();
            chunk.resize(m, (0, None));
            self.rpf_chunk(wi, &chunk, base, lpos)?;
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
            anyhow::ensure!((row.token as usize) < self.vocab, "token {} out of vocab", row.token);
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
        let need_scope = !rg.bat_validated;
        let scope = need_scope.then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        {
            let rg = self.region.as_ref().expect("region machinery");
            let mut run = self.swapped_run(wi, true);
            let (hs, he) = rg.head_range;
            for r in 0..rows.len() {
                run.push(&rg.gathers[r]);
                for p in &self.passes[hs..he] {
                    run.push(p);
                }
                run.push(&rg.stashes[r]);
            }
            let cb = dispatch::encode_pass_list_labeled(
                self.ctx,
                run.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
                &[],
            );
            self.ctx.queue.submit([cb]);
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("dots batched region step validation: {e}");
            }
            self.region.as_mut().expect("region machinery").bat_validated = true;
        }
        let rg = self.region.as_ref().expect("region machinery");
        let all: Vec<u32> = dispatch::read_back(self.ctx, &rg.btoks, m)
            .map_err(|e| anyhow::anyhow!("btoks readback: {e}"))?;
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
            bw > PREFILL_M_WIDE
                && end <= bw
                && end >= bw * 5 / 8
                && self.pos + bw <= self.max_seq
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
                    i = end;
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
        let hidden = self.cfg.hidden_size;
        let prompt_len = prompt_tokens.len();
        anyhow::ensure!(prompt_len > 0, "generate needs a prompt");
        anyhow::ensure!(
            prompt_len < self.max_seq,
            "prompt of {prompt_len} tokens exceeds the graph kv span {} (raise NV_DOTS_MAX_SEQ \
             or lower NV_DOTS_MAX_PIXELS)",
            self.max_seq
        );
        let n_vis = match vision_features {
            None => 0,
            Some(f) => {
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
        };
        self.reset()?;

        let t_gen = std::time::Instant::now();
        self.repack_s = 0.0;
        let row_at = |i: usize| -> Option<&[f32]> {
            vision_features
                .filter(|_| i >= vision_start && i < vision_start + n_vis)
                .map(|f| &f[(i - vision_start) * hidden..(i - vision_start + 1) * hidden])
        };
        let widths = self.pf_widths.clone();
        let mut i = 0usize;
        let gend = prompt_len - 1;
        if let Some(&bw) = widths.first().filter(|&&bw| {
            bw > PREFILL_M_WIDE
                && gend <= bw
                && gend >= bw * 5 / 8
                && self.pos + bw <= self.max_seq
        }) {
            let mut chunk: Vec<(u32, Option<&[f32]>)> =
                (0..gend).map(|j| (prompt_tokens[j], row_at(j))).collect();
            chunk.resize(bw, (0, None));
            self.prefill_chunk_real(0, &chunk, gend)?;
            i = gend;
        }
        for (wi, m) in widths.iter().copied().enumerate() {
            while i + m <= prompt_len - 1 {
                let chunk: Vec<(u32, Option<&[f32]>)> = (i..i + m)
                    .map(|j| (prompt_tokens[j], row_at(j)))
                    .collect();
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
            match tail {
                Some((wi, m)) => {
                    let mut chunk: Vec<(u32, Option<&[f32]>)> = (i..prompt_len - 1)
                        .map(|j| (prompt_tokens[j], row_at(j)))
                        .collect();
                    chunk.resize(m, (0, None));
                    self.prefill_chunk_real(wi, &chunk, rem)?;
                    i = prompt_len - 1;
                }
                None => {}
            }
        }
        while i + 1 < prompt_len {
            match row_at(i) {
                Some(r) => self.prefill_embed(r)?,
                None => self.prefill_step(prompt_tokens[i])?,
            }
            i += 1;
        }
        let mut gpu_tok = match row_at(prompt_len - 1) {
            Some(r) => self.decode_embed_logits(r)?.0,
            None => self.decode_step_logits(prompt_tokens[prompt_len - 1])?.0,
        };
        let prefill_s = t_gen.elapsed().as_secs_f64();
        let t_decode = std::time::Instant::now();

        let mut generated: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        let mut spec_agree: Vec<u64> = vec![0; self.spec_tok.len()];
        let mut spec_steps: u64 = 0;
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
                gpu_tok
            } else {
                argmax_banned(&self.read_logits()?, &banned)
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
            if self.pos + 1 >= self.max_seq {
                break;
            }
            gpu_tok = self.decode_step(next)?;
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
