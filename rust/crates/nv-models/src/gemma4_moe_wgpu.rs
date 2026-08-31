use anyhow::{Context, Result};

use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};

use crate::gemma4::LayerType;
use crate::gemma4_moe::Gemma4MoeConfig;
use crate::qwen3_5_moe_wgpu::{staging_flush_enabled, VramReport};

pub const W4_GS: usize = 32;

const ARGMAX_GROUPS: usize = 256;
const MAX_TOPK: usize = 16;
const MAX_HEAD_DIM: usize = 512;
const STAGING_FLUSH_BYTES: u64 = 256 << 20;

const GEMV_BF16_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_bf16.wgsl");

const GEMV_BF16_RPNORM_SGLAT_WGSL: &str =
    include_str!("../../nv-kernels/wgsl/g4m_gemv_bf16_rpnorm_sglat.wgsl");

const GEMV_BF16_SGLAT_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_bf16_sglat.wgsl");

const GEMV_BF16_V2_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_bf16_v2.wgsl");

const GEMV_BF16_DNRP_SGLAT_WGSL: &str =
    include_str!("../../nv-kernels/wgsl/g4m_gemv_bf16_dnrp_sglat.wgsl");

const GEMV_W4E_SGLAT_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_w4e_sglat.wgsl");

const GEMV_W4E_SGLAT_XV4_WGSL: &str =
    include_str!("../../nv-kernels/wgsl/g4m_gemv_w4e_sglat_xv4.wgsl");

const GEMV_BF16_LEGACY_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_bf16_legacy.wgsl");

const GEMV_BF16_TAILFOLD_WGSL: &str =
    include_str!("../../nv-kernels/wgsl/g4m_gemv_bf16_tailfold.wgsl");

const GEMV_I8_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_i8.wgsl");

const GEMV_I8_V4_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_i8_v4.wgsl");

const GEMV_I8_G16_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_i8_g16.wgsl");

const GEMV_W4E_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_w4e.wgsl");

const ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_attn.wgsl");

const ATTN_SGLAT_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_attn_sglat.wgsl");

const FLASH_DECODE_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_flash_decode.wgsl");

const FLASH_REDUCE_WB_ELEVEN_WORKGROUP_BARRIERS_AND_TEN_LDS_ROUND_TRIPS_PER_KV_ROW_BATCH: &str =
    include_str!("../../nv-kernels/wgsl/g4m_flash_reduce_wb.wgsl");

const FLASH_REDUCE_SG_SHUFFLE_XOR_CHAIN_16_8_4_2_1_IS_BIT_IDENTICAL_TO_THE_LDS_BUTTERFLY_SAME_OPERAND_ORDER: &str =
    include_str!("../../nv-kernels/wgsl/g4m_flash_reduce_sg.wgsl");

const FLASH_QKVFOLD_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_flash_qkvfold.wgsl");

const PROP_NORM_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_prop_norm.wgsl");

const PROP_NORM_SGLAT_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_prop_norm_sglat.wgsl");

const PROP_NORM_X2_TOPK_SGLAT_WGSL: &str =
    include_str!("../../nv-kernels/wgsl/g4m_prop_norm_x2_topk_sglat.wgsl");

const MOE_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_moe.wgsl");

const PREFILL_HEAD_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_prefill_head.wgsl");

const EMBED_ROW_SPLICE_WGSL: &str = r#"
struct PmSpParams { hidden_words: u32, m: u32, pad0: u32, pad1: u32 };
@group(0) @binding(100) var<storage, read> psp_rows: array<u32>;
@group(0) @binding(101) var<storage, read> psp_mask: array<u32>;
@group(0) @binding(102) var<storage, read_write> psp_out: array<u32>;
@group(0) @binding(103) var<uniform> psp_p: PmSpParams;

@compute @workgroup_size(256)
fn pm_splice_embed_rows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    if (t >= psp_p.m || psp_mask[t] == 0u) {
        return;
    }
    let w = wid.x * 256u + lid.x;
    if (w >= psp_p.hidden_words) {
        return;
    }
    psp_out[t * psp_p.hidden_words + w] = psp_rows[t * psp_p.hidden_words + w];
}
"#;

const VERIFY_ARGMAX_WGSL: &str = r#"
struct PmVamParams { n: u32, groups: u32, pad0: u32, pad1: u32 };
@group(0) @binding(110) var<storage, read> vam_x: array<u32>;
@group(0) @binding(111) var<storage, read_write> vam_pv: array<f32>;
@group(0) @binding(112) var<storage, read_write> vam_pi: array<u32>;
@group(0) @binding(113) var<storage, read_write> vam_out: array<u32>;
@group(0) @binding(114) var<uniform> vam_p: PmVamParams;

var<workgroup> vam_v: array<f32, 256>;
var<workgroup> vam_i: array<u32, 256>;

fn vam_reduce(tid: u32) {
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            let o = tid + s;
            if (vam_v[o] > vam_v[tid] || (vam_v[o] == vam_v[tid] && vam_i[o] < vam_i[tid])) {
                vam_v[tid] = vam_v[o];
                vam_i[tid] = vam_i[o];
            }
        }
        workgroupBarrier();
    }
}

@compute @workgroup_size(256)
fn pm_verify_argmax_bf16_stage1(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let g = wid.x;
    let r = wid.y;
    let tid = lid.x;
    let n_words = (vam_p.n + 1u) / 2u;
    let xbase = r * n_words;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    for (var w = g * 256u + tid; w < n_words; w = w + vam_p.groups * 256u) {
        let word = vam_x[xbase + w];
        let i0 = w * 2u;
        let v0 = bf16_lo(word);
        if (i0 < vam_p.n && (v0 > bv || (v0 == bv && i0 < bi))) {
            bv = v0;
            bi = i0;
        }
        let v1 = bf16_hi(word);
        if (i0 + 1u < vam_p.n && (v1 > bv || (v1 == bv && i0 + 1u < bi))) {
            bv = v1;
            bi = i0 + 1u;
        }
    }
    vam_v[tid] = bv;
    vam_i[tid] = bi;
    workgroupBarrier();
    vam_reduce(tid);
    if (tid == 0u) {
        vam_pv[r * vam_p.groups + g] = vam_v[0];
        vam_pi[r * vam_p.groups + g] = vam_i[0];
    }
}

@compute @workgroup_size(256)
fn pm_verify_argmax_stage2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.y;
    let tid = lid.x;
    var bv = -3.4028235e38;
    var bi = 0xffffffffu;
    if (tid < vam_p.groups) {
        bv = vam_pv[r * vam_p.groups + tid];
        bi = vam_pi[r * vam_p.groups + tid];
    }
    vam_v[tid] = bv;
    vam_i[tid] = bi;
    workgroupBarrier();
    vam_reduce(tid);
    if (tid == 0u) {
        vam_out[r] = vam_i[0];
    }
}
"#;

fn prefill_wgsl(m: usize) -> String {
    use std::fmt::Write as _;
    let mut b = String::from(PREFILL_HEAD_WGSL);
    b.push_str("\n@compute @workgroup_size(256)\nfn pm_gemm_bf16(\n");
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n");
    b.push_str("    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
    b.push_str("    let tid = lid.x;\n");
    b.push_str("    let half = tid >> 7u;\n");
    b.push_str("    let lane = tid & 127u;\n");
    b.push_str("    let pair = wid.x + wid.y * pb_p.groups_x;\n");
    b.push_str("    let row = pair * 2u + half;\n");
    b.push_str("    let live = row < pb_p.n_rows;\n");
    b.push_str("    let wbase = select(0u, row * pb_p.w_row_words, live);\n");
    b.push_str("    let kw = select(0u, pb_p.k_words, live);\n");
    for t in 0..m {
        writeln!(b, "    var acc{t} = 0.0;").unwrap();
    }
    b.push_str("    for (var i = lane; i < kw; i = i + 128u) {\n");
    b.push_str("        let ww = pb_w[wbase + i];\n");
    b.push_str("        let wl = bf16_lo(ww);\n");
    b.push_str("        let wh = bf16_hi(ww);\n");
    for t in 0..m {
        writeln!(
            b,
            "        let xw{t} = pb_x[{t}u * pb_p.x_stride_words + i];"
        )
        .unwrap();
        writeln!(b, "        acc{t} = fma(wl, bf16_lo(xw{t}), acc{t});").unwrap();
        writeln!(b, "        acc{t} = fma(wh, bf16_hi(xw{t}), acc{t});").unwrap();
    }
    b.push_str("    }\n");
    for t in 0..m {
        writeln!(b, "    pb_red[tid] = acc{t};").unwrap();
        b.push_str("    workgroupBarrier();\n");
        b.push_str("    for (var stride = 64u; stride > 0u; stride = stride >> 1u) {\n");
        b.push_str("        if (lane < stride) {\n");
        b.push_str("            pb_red[tid] = pb_red[tid] + pb_red[tid + stride];\n");
        b.push_str("        }\n        workgroupBarrier();\n    }\n");
        b.push_str("    if (pb_p.out_f32 == 1u) {\n");
        b.push_str("        if (lane == 0u && live) {\n");
        writeln!(
            b,
            "            pb_y[{t}u * pb_p.y_stride_words + pb_p.y_off_words + row] = bitcast<u32>(pb_red[tid] * pb_p.alpha);"
        )
        .unwrap();
        b.push_str("        }\n    } else if (tid == 0u) {\n");
        writeln!(b, "        let lo{t} = pb_red[0] * pb_p.alpha;").unwrap();
        writeln!(b, "        var hi{t} = 0.0;").unwrap();
        b.push_str("        if (row + 1u < pb_p.n_rows) {\n");
        writeln!(b, "            hi{t} = pb_red[128] * pb_p.alpha;").unwrap();
        b.push_str("        }\n");
        writeln!(
            b,
            "        pb_y[{t}u * pb_p.y_stride_words + pb_p.y_off_words + (row >> 1u)] = bf16_pack(lo{t}, hi{t});"
        )
        .unwrap();
        b.push_str("    }\n    workgroupBarrier();\n");
    }
    b.push_str("}\n");
    b
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfGatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
    scale: f32,
    m: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfSpliceParams {
    hidden_words: u32,
    m: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfVerifyArgmaxParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfNormParams {
    hidden: u32,
    words: u32,
    eps: f32,
    m: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfGemmParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    y_off_words: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfRopeParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    has_rope: u32,
    tok_src_stride: u32,
    tok_dst_stride: u32,
    pad0: u32,
    eps: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfKvParams {
    words: u32,
    m: u32,
    ring: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfAttnParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    window: u32,
    m: u32,
    ring: u32,
    scale: f32,
    pad1: u32,
    pad2: u32,
    pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfRouterParams {
    n_experts: u32,
    k: u32,
    m: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfW4Params {
    n_rows: u32,
    groups: u32,
    groups_x: u32,
    w_e_stride_words: u32,
    s_e_stride_elems: u32,
    x_slot_stride_words: u32,
    y_slot_stride_words: u32,
    x_tok_stride_words: u32,
    k_top: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfMulParams {
    row_words: u32,
    m: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfCombineParams {
    hidden_words: u32,
    k: u32,
    slot_stride_words: u32,
    m: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GemvBf16Params {
    pub n_rows: u32,
    pub k_words: u32,
    pub groups_x: u32,
    pub out_f32: u32,
    pub w_row_words: u32,
    pub x_off_words: u32,
    pub y_off_words: u32,

    pub wide: u32,
    pub alpha: f32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GemvI8Params {
    pub n_rows: u32,
    pub k_words: u32,
    pub groups_x: u32,
    pub out_f32: u32,
    pub w_row_words: u32,
    pub x_off_words: u32,
    pub y_off_words: u32,
    pub s_row_elems: u32,
    pub alpha: f32,
    pub pad1: u32,
    pub pad2: u32,
    pub pad3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NormParams {
    hidden: u32,
    words: u32,
    eps: f32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ResScaleParams {
    n: u32,
    n_words: u32,
    scale: f32,
    cap: f32,
    inv_cap: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NormRopeParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    has_rope: u32,
    n_q: u32,
    n_kv: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvWriteParams {
    words: u32,
    ring: u32,
    k_off_words: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvFp8ParamsMirroringNvKernelsKvFp8UniformLayout {
    n_tokens: u32,
    n_kv: u32,
    head_dim: u32,
    ring: u32,
    pairs: u32,
    start: u32,
    slots: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnDecodeParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    window: u32,
    ring: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FlashDecodeParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    splits: u32,
    total: u32,
    start: u32,
    ring: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RouterParams {
    n_experts: u32,
    k: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RouterChainParams {
    n_experts: u32,
    k: u32,
    words: u32,
    hidden: u32,
    eps: f32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct WordsParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvW4Params {
    n_rows: u32,
    groups: u32,
    groups_x: u32,
    w_e_stride_words: u32,
    s_e_stride_elems: u32,
    x_slot_stride_words: u32,
    y_slot_stride_words: u32,
    k_top: u32,
    n_experts: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CombineParams {
    hidden_words: u32,
    k: u32,
    slot_stride_words: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxParams {
    n: u32,
    groups: u32,
    pad0: u32,
    pad1: u32,
}

#[derive(Clone)]
pub struct HostLin {
    pub w: Vec<u16>,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone)]
pub struct HostLayer {
    pub kind: LayerType,
    pub input_ln: Vec<u16>,
    pub post_attn_ln: Vec<u16>,
    pub pre_ffw_ln: Vec<u16>,
    pub post_ffw_ln: Vec<u16>,
    pub post_ffw_ln_1: Vec<u16>,
    pub pre_ffw_ln_2: Vec<u16>,
    pub post_ffw_ln_2: Vec<u16>,
    pub layer_scalar: f32,
    pub q: HostLin,
    pub k: HostLin,
    pub v: Option<HostLin>,
    pub o: HostLin,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
    pub mlp_gate: HostLin,
    pub mlp_up: HostLin,
    pub mlp_down: HostLin,
    pub router: HostLin,
    pub router_scale: Vec<u16>,
    pub per_expert_scale: Vec<f32>,
    pub experts_gate: HostW4Stack,
    pub experts_up: HostW4Stack,
    pub experts_down: HostW4Stack,
}

#[derive(Clone)]
pub struct HostW4Mat {
    pub packed: Vec<u32>,
    pub scales: Vec<u16>,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone)]
pub struct HostW4Stack {
    pub packed: Vec<u32>,
    pub scales: Vec<u16>,
    pub e: usize,
    pub n: usize,
    pub k: usize,
}

pub fn quantize_w4_host(w: &[u16], n: usize, k: usize) -> HostW4Mat {
    assert!(k.is_multiple_of(W4_GS));
    let groups = k / W4_GS;
    let mut packed = vec![0u32; n * k / 8];
    let mut scales = vec![0u16; n * groups];
    for r in 0..n {
        for g in 0..groups {
            let base = r * k + g * W4_GS;
            let mut amax = 0f32;
            for j in 0..W4_GS {
                let v = f32::from_bits((w[base + j] as u32) << 16).abs();
                if v > amax {
                    amax = v;
                }
            }
            let scale = if amax > 0.0 && amax.is_finite() {
                amax / 7.0
            } else {
                1.0
            };
            let scale_b = half::bf16::from_f32(scale);
            scales[r * groups + g] = scale_b.to_bits();
            let inv = 1.0 / scale_b.to_f32();
            for j in 0..W4_GS {
                let v = f32::from_bits((w[base + j] as u32) << 16);
                let q = (v * inv).round().clamp(-8.0, 7.0) as i32;
                let u = (q + 8) as u32;
                let elem = g * W4_GS + j;
                packed[r * (k / 8) + elem / 8] |= u << (4 * (elem % 8));
            }
        }
    }
    HostW4Mat {
        packed,
        scales,
        n,
        k,
    }
}

pub const I8_GS: usize = 32;

#[derive(Clone)]
pub struct HostI8Mat {
    pub q: Vec<u32>,
    pub scales: Vec<u16>,
    pub n: usize,
    pub k: usize,
}

pub fn quantize_i8_host(w: &[u16], n: usize, k: usize) -> HostI8Mat {
    quantize_i8_host_group(w, n, k, I8_GS)
}

pub fn quantize_i8_host_group(w: &[u16], n: usize, k: usize, gs: usize) -> HostI8Mat {
    assert!(k.is_multiple_of(gs), "k={k} is not a multiple of {gs}");
    assert_eq!(w.len(), n * k, "quantize_i8_host: w is not n*k");
    let groups = k / gs;
    let mut q = vec![0u32; n * k / 4];
    let mut scales = vec![0u16; n * groups];
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(n.max(1));
    let rows_per = n.div_ceil(threads);
    std::thread::scope(|sc| {
        let mut q_rest = q.as_mut_slice();
        let mut s_rest = scales.as_mut_slice();
        let mut r0 = 0usize;
        while r0 < n {
            let rows = rows_per.min(n - r0);
            let (q_chunk, q_next) = q_rest.split_at_mut(rows * k / 4);
            let (s_chunk, s_next) = s_rest.split_at_mut(rows * groups);
            q_rest = q_next;
            s_rest = s_next;
            let w_chunk = &w[r0 * k..(r0 + rows) * k];
            sc.spawn(move || {
                for rr in 0..rows {
                    for g in 0..groups {
                        let base = rr * k + g * gs;
                        let mut amax = 0f32;
                        for j in 0..gs {
                            let v = f32::from_bits((w_chunk[base + j] as u32) << 16).abs();
                            if v > amax {
                                amax = v;
                            }
                        }
                        let scale = if amax > 0.0 && amax.is_finite() {
                            amax / 127.0
                        } else {
                            1.0
                        };
                        let scale_b = half::bf16::from_f32(scale);
                        s_chunk[rr * groups + g] = scale_b.to_bits();
                        let inv = 1.0 / scale_b.to_f32();
                        for j in 0..gs {
                            let v = f32::from_bits((w_chunk[base + j] as u32) << 16);
                            let qi = (v * inv).round().clamp(-127.0, 127.0) as i32;
                            let elem = g * gs + j;
                            q_chunk[rr * (k / 4) + elem / 4] |=
                                ((qi as u8) as u32) << (8 * (elem % 4));
                        }
                    }
                }
            });
            r0 += rows;
        }
    });
    HostI8Mat { q, scales, n, k }
}

pub fn dequantize_i8_row(m: &HostI8Mat, r: usize) -> Vec<f32> {
    let groups = m.k / I8_GS;
    let mut out = vec![0f32; m.k];
    for (elem, o) in out.iter_mut().enumerate() {
        let word = m.q[r * (m.k / 4) + elem / 4];
        let byte = ((word >> (8 * (elem % 4))) & 0xff) as u8 as i8;
        let scale = f32::from_bits((m.scales[r * groups + elem / I8_GS] as u32) << 16);
        *o = byte as f32 * scale;
    }
    out
}

#[doc(hidden)]
pub mod bench {
    pub const GEMV_BF16_ENTRY: &str = "g4m_gemv_bf16";
    pub const GEMV_I8_ENTRY: &str = "g4m_gemv_i8";

    pub const GEMV_BF16_V4_ENTRY: &str = "g4m_gemv_bf16";
    pub const GEMV_I8_V4_ENTRY: &str = "g4m_gemv_i8_v4";

    pub fn gemv_bf16_source() -> String {
        nv_kernels::wgpu_backend::compose(super::GEMV_BF16_WGSL)
    }

    pub fn gemv_i8_source() -> String {
        nv_kernels::wgpu_backend::compose(super::GEMV_I8_WGSL)
    }

    pub fn gemv_bf16_v4_source() -> String {
        nv_kernels::wgpu_backend::compose(super::GEMV_BF16_WGSL)
    }

    pub fn gemv_i8_v4_source() -> String {
        nv_kernels::wgpu_backend::compose(super::GEMV_I8_V4_WGSL)
    }

    pub const GEMV_BF16_LEGACY_ENTRY: &str = "g4m_gemv_bf16_legacy";

    pub fn gemv_bf16_legacy_source() -> String {
        nv_kernels::wgpu_backend::compose(super::GEMV_BF16_LEGACY_WGSL)
    }
}

pub fn stack_w4_host(mats: &[HostW4Mat]) -> HostW4Stack {
    let n = mats[0].n;
    let k = mats[0].k;
    let mut packed = Vec::with_capacity(mats.len() * n * k / 8);
    let mut scales = Vec::with_capacity(mats.len() * n * k / W4_GS);
    for m in mats {
        packed.extend_from_slice(&m.packed);
        scales.extend_from_slice(&m.scales);
    }
    HostW4Stack {
        packed,
        scales,
        e: mats.len(),
        n,
        k,
    }
}

pub fn dequantize_w4_expert(stack: &HostW4Stack, e: usize) -> Vec<f32> {
    let (n, k) = (stack.n, stack.k);
    let groups = k / W4_GS;
    let pw = n * k / 8;
    let sw = n * groups;
    let packed = &stack.packed[e * pw..(e + 1) * pw];
    let scales = &stack.scales[e * sw..(e + 1) * sw];
    let mut out = vec![0f32; n * k];
    for r in 0..n {
        for elem in 0..k {
            let word = packed[r * (k / 8) + elem / 8];
            let q = ((word >> (4 * (elem % 8))) & 15) as i32 - 8;
            let scale = f32::from_bits((scales[r * groups + elem / W4_GS] as u32) << 16);
            out[r * k + elem] = q as f32 * scale;
        }
    }
    out
}

fn bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

use crate::gemma4_wgpu_shared::{pack_pairs, rope_tables};

fn load_bf16<S: nv_weights::TensorSource>(w: &S, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
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

fn load_lin<S: nv_weights::TensorSource>(w: &S, name: &str, n: usize, k: usize) -> Result<HostLin> {
    Ok(HostLin {
        w: load_bf16(w, name, &[n, k])?,
        n,
        k,
    })
}

fn load_f32_vec<S: nv_weights::TensorSource>(w: &S, name: &str, dim: usize) -> Result<Vec<f32>> {
    let t = w
        .get(name, candle_core::DType::F32)
        .with_context(|| format!("load {name}"))?;
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    anyhow::ensure!(v.len() == dim, "{name}: len {} != {dim}", v.len());
    Ok(v)
}

pub trait CompressedTensorsRawBytes {
    fn raw_bytes_and_shape_because_tensor_source_get_has_no_f8e4m3_dtype(
        &self,
        name: &str,
    ) -> Option<(Vec<u8>, Vec<usize>)>;
}

impl CompressedTensorsRawBytes for nv_weights::WeightLoader {
    fn raw_bytes_and_shape_because_tensor_source_get_has_no_f8e4m3_dtype(
        &self,
        name: &str,
    ) -> Option<(Vec<u8>, Vec<usize>)> {
        let shape = self.shape_of(name)?;
        let bytes = self.raw_bytes(name).ok()?.to_vec();
        Some((bytes, shape))
    }
}

impl CompressedTensorsRawBytes for nv_weights::GgufLoader {
    fn raw_bytes_and_shape_because_tensor_source_get_has_no_f8e4m3_dtype(
        &self,
        _name: &str,
    ) -> Option<(Vec<u8>, Vec<usize>)> {
        None
    }
}

pub struct CtNvfp4Parts {
    pub packed: Vec<u8>,
    pub scales: Vec<u8>,
    pub inv_global: f32,
}

pub fn compressed_tensors_nvfp4_parts<S: CompressedTensorsRawBytes>(
    w: &S,
    module: &str,
    n: usize,
    k: usize,
) -> Result<CtNvfp4Parts> {
    let group = nv_quant::nvfp4::BLOCK_SIZE;
    anyhow::ensure!(
        k.is_multiple_of(group),
        "{module}: in_features {k} must be a multiple of the nvfp4 group size {group}"
    );
    let read = |suffix: &str| -> Result<(Vec<u8>, Vec<usize>)> {
        let name = format!("{module}.{suffix}");
        w.raw_bytes_and_shape_because_tensor_source_get_has_no_f8e4m3_dtype(&name)
            .ok_or_else(|| anyhow::anyhow!("tensor not found: {name}"))
    };
    let (packed, ps) = read("weight_packed")?;
    anyhow::ensure!(
        ps == [n, k / 2],
        "{module}.weight_packed: shape {ps:?} != [{n}, {}]",
        k / 2
    );
    anyhow::ensure!(
        packed.len() == n * k / 2,
        "{module}.weight_packed: {} bytes != {}",
        packed.len(),
        n * k / 2
    );
    let (scales, ss) = read("weight_scale")?;
    anyhow::ensure!(
        ss == [n, k / group],
        "{module}.weight_scale: shape {ss:?} != [{n}, {}]",
        k / group
    );
    anyhow::ensure!(
        scales.len() == n * k / group,
        "{module}.weight_scale: {} bytes != {}",
        scales.len(),
        n * k / group
    );
    let (gs, _) = read("weight_global_scale")?;
    anyhow::ensure!(
        gs.len() == 4,
        "{module}.weight_global_scale: {} bytes != 4 (one f32)",
        gs.len()
    );
    let g = f32::from_le_bytes([gs[0], gs[1], gs[2], gs[3]]);
    let inv_global = if g.is_finite() && g != 0.0 { 1.0 / g } else { 1.0 };
    Ok(CtNvfp4Parts {
        packed,
        scales,
        inv_global,
    })
}

pub fn ct_nvfp4_dequant_to_bf16_bits(p: &CtNvfp4Parts, n: usize, k: usize) -> Vec<u16> {
    nv_quant::nvfp4::dequantize_packed_linear(&p.packed, &p.scales, n, k, p.inv_global)
        .into_iter()
        .map(bf16_bits)
        .collect()
}

fn load_lin_bf16_else_compressed_tensors_nvfp4<
    S: nv_weights::TensorSource + CompressedTensorsRawBytes,
>(
    w: &S,
    module: &str,
    n: usize,
    k: usize,
) -> Result<HostLin> {
    let wname = format!("{module}.weight");
    if !w.has(&wname) && w.has(&format!("{module}.weight_packed")) {
        let parts = compressed_tensors_nvfp4_parts(w, module, n, k)?;
        return Ok(HostLin {
            w: ct_nvfp4_dequant_to_bf16_bits(&parts, n, k),
            n,
            k,
        });
    }
    load_lin(w, &wname, n, k)
}

fn par_experts<T, F>(n_e: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let jobs = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4)
        .clamp(1, n_e.max(1));
    let chunk = n_e.div_ceil(jobs);
    let mut out: Vec<Option<T>> = (0..n_e).map(|_| None).collect();
    std::thread::scope(|s| {
        for (ci, slots) in out.chunks_mut(chunk).enumerate() {
            let f = &f;
            s.spawn(move || {
                for (j, slot) in slots.iter_mut().enumerate() {
                    *slot = Some(f(ci * chunk + j));
                }
            });
        }
    });
    out.into_iter().map(|x| x.unwrap()).collect()
}

pub fn host_layer_from_loader<S: nv_weights::TensorSource + CompressedTensorsRawBytes>(
    cfg: &Gemma4MoeConfig,
    w: &S,
    idx: usize,
) -> Result<HostLayer> {
    let base = &cfg.base;
    let prefix = format!("model.language_model.layers.{idx}");
    let hidden = base.hidden_size;
    let kind = base.layer_kind(idx);
    let hd = base.head_dim_for(kind);
    let n_q = base.num_attention_heads;
    let n_kv = base.num_kv_heads_for(kind);
    let inter = base.intermediate_size;
    let n_e = cfg.num_experts;
    let mi = cfg.moe_intermediate_size;
    let has_v = !matches!(
        (kind, base.attention_k_eq_v),
        (LayerType::FullAttention, true)
    );

    let norm = |name: &str, dim: usize| load_bf16(w, &format!("{prefix}.{name}.weight"), &[dim]);

    let layer_scalar = load_f32_vec(w, &format!("{prefix}.layer_scalar"), 1)?[0];

    let fused_gate_up_name = format!("{prefix}.experts.gate_up_proj");
    let unfused_ct_nvfp4_probe = format!("{prefix}.experts.0.gate_proj.weight_packed");
    let (gate_mats, up_mats, down_mats) = if !w.has(&fused_gate_up_name)
        && w.has(&unfused_ct_nvfp4_probe)
    {
        let per_expert_w4_via_serial_fetch_then_parallel_requant =
            |proj: &str, rows: usize, cols: usize| -> Result<Vec<HostW4Mat>> {
                let raw: Vec<CtNvfp4Parts> = (0..n_e)
                    .map(|e| {
                        compressed_tensors_nvfp4_parts(
                            w,
                            &format!("{prefix}.experts.{e}.{proj}"),
                            rows,
                            cols,
                        )
                    })
                    .collect::<Result<_>>()?;
                Ok(par_experts(n_e, |e| {
                    quantize_w4_host(
                        &ct_nvfp4_dequant_to_bf16_bits(&raw[e], rows, cols),
                        rows,
                        cols,
                    )
                }))
            };
        (
            per_expert_w4_via_serial_fetch_then_parallel_requant("gate_proj", mi, hidden)?,
            per_expert_w4_via_serial_fetch_then_parallel_requant("up_proj", mi, hidden)?,
            per_expert_w4_via_serial_fetch_then_parallel_requant("down_proj", hidden, mi)?,
        )
    } else {
        let gu = load_bf16(w, &fused_gate_up_name, &[n_e, 2 * mi, hidden])?;
        let dn = load_bf16(
            w,
            &format!("{prefix}.experts.down_proj"),
            &[n_e, hidden, mi],
        )?;
        let gate_mats = par_experts(n_e, |e| {
            let b = e * 2 * mi * hidden;
            quantize_w4_host(&gu[b..b + mi * hidden], mi, hidden)
        });
        let up_mats = par_experts(n_e, |e| {
            let b = e * 2 * mi * hidden + mi * hidden;
            quantize_w4_host(&gu[b..b + mi * hidden], mi, hidden)
        });
        drop(gu);
        let down_mats = par_experts(n_e, |e| {
            quantize_w4_host(&dn[e * hidden * mi..(e + 1) * hidden * mi], hidden, mi)
        });
        (gate_mats, up_mats, down_mats)
    };

    Ok(HostLayer {
        kind,
        input_ln: norm("input_layernorm", hidden)?,
        post_attn_ln: norm("post_attention_layernorm", hidden)?,
        pre_ffw_ln: norm("pre_feedforward_layernorm", hidden)?,
        post_ffw_ln: norm("post_feedforward_layernorm", hidden)?,
        post_ffw_ln_1: norm("post_feedforward_layernorm_1", hidden)?,
        pre_ffw_ln_2: norm("pre_feedforward_layernorm_2", hidden)?,
        post_ffw_ln_2: norm("post_feedforward_layernorm_2", hidden)?,
        layer_scalar,
        q: load_lin_bf16_else_compressed_tensors_nvfp4(
            w,
            &format!("{prefix}.self_attn.q_proj"),
            n_q * hd,
            hidden,
        )?,
        k: load_lin_bf16_else_compressed_tensors_nvfp4(
            w,
            &format!("{prefix}.self_attn.k_proj"),
            n_kv * hd,
            hidden,
        )?,
        v: if has_v {
            Some(load_lin_bf16_else_compressed_tensors_nvfp4(
                w,
                &format!("{prefix}.self_attn.v_proj"),
                n_kv * hd,
                hidden,
            )?)
        } else {
            None
        },
        o: load_lin_bf16_else_compressed_tensors_nvfp4(
            w,
            &format!("{prefix}.self_attn.o_proj"),
            hidden,
            n_q * hd,
        )?,
        q_norm: norm("self_attn.q_norm", hd)?,
        k_norm: norm("self_attn.k_norm", hd)?,
        mlp_gate: load_lin_bf16_else_compressed_tensors_nvfp4(
            w,
            &format!("{prefix}.mlp.gate_proj"),
            inter,
            hidden,
        )?,
        mlp_up: load_lin_bf16_else_compressed_tensors_nvfp4(
            w,
            &format!("{prefix}.mlp.up_proj"),
            inter,
            hidden,
        )?,
        mlp_down: load_lin_bf16_else_compressed_tensors_nvfp4(
            w,
            &format!("{prefix}.mlp.down_proj"),
            hidden,
            inter,
        )?,
        router: load_lin(w, &format!("{prefix}.router.proj.weight"), n_e, hidden)?,
        router_scale: load_bf16(w, &format!("{prefix}.router.scale"), &[hidden])?,
        per_expert_scale: load_f32_vec(w, &format!("{prefix}.router.per_expert_scale"), n_e)?,
        experts_gate: stack_w4_host(&gate_mats),
        experts_up: stack_w4_host(&up_mats),
        experts_down: stack_w4_host(&down_mats),
    })
}

struct Pass {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    bind: wgpu::BindGroup,
    grid: (u32, u32, u32),
    label: String,
    bind_bytes: u64,
    storage_binds_are_what_wgpu_tracks_for_hazards_uniforms_never_transition: Vec<(u32, bool)>,
}

fn wgsl_binding_is_storage_read_write_a_read_only_binding_can_never_raise_a_hazard(
    source: &str,
    binding: u32,
) -> bool {
    let needle = format!("@binding({binding})");
    source
        .match_indices(&needle)
        .filter_map(|(i, _)| source[i..].split(';').next())
        .find(|decl| decl.contains("var<"))
        .is_none_or(|decl| decl.contains("read_write"))
}

pub fn barrier_cause_raw_waw_is_a_true_dependency_war_alone_can_be_an_artifact_of_a_wide_binding_access_mode(
    storage_bind_sets: &[Vec<(u32, bool)>],
) -> Vec<Option<bool>> {
    let mut read_since: std::collections::BTreeSet<u32> = Default::default();
    let mut written_since: std::collections::BTreeSet<u32> = Default::default();
    let mut cause_at: Vec<Option<bool>> = Vec::with_capacity(storage_bind_sets.len());
    for set in storage_bind_sets {
        let raw_or_waw = set.iter().any(|(id, _)| written_since.contains(id));
        let war = set
            .iter()
            .any(|(id, w)| *w && (read_since.contains(id) || written_since.contains(id)));
        let cause = if raw_or_waw {
            Some(true)
        } else if war {
            Some(false)
        } else {
            None
        };
        cause_at.push(cause);
        if cause.is_some() {
            read_since.clear();
            written_since.clear();
        }
        for (id, w) in set {
            if *w {
                written_since.insert(*id);
            } else {
                read_since.insert(*id);
            }
        }
    }
    cause_at
}

pub fn a_pass_is_a_wgpu_barrier_iff_it_reads_or_writes_a_buffer_written_since_the_last_barrier_or_writes_one_read_since(
    storage_bind_sets: &[Vec<(u32, bool)>],
) -> Vec<bool> {
    barrier_cause_raw_waw_is_a_true_dependency_war_alone_can_be_an_artifact_of_a_wide_binding_access_mode(
        storage_bind_sets,
    )
    .into_iter()
    .map(|c| c.is_some())
    .collect()
}

fn vram_class(label: &str) -> &str {
    label.strip_prefix("g4m-").unwrap_or(label)
}

fn per_token_weight_share(class: &str, expert_share: f64) -> f64 {
    match class {
        "moe-eg" | "moe-eu" | "moe-ed" | "moe-eg-sf" | "moe-eu-sf" | "moe-ed-sf" => expert_share,
        "embed" | "ln" | "final-ln" | "at-qw" | "at-kw" | "at-vw" | "at-ow" | "at-qn" | "at-kn"
        | "at-vn" | "mlp-gw" | "mlp-uw" | "mlp-dw" | "moe-router" | "moe-rscale" | "moe-pes"
        | "moe-ones" | "lmhead-i8" | "lmhead-i8-sf" | "at-qkv-i8" | "at-qkv-i8-sf" | "at-o-i8"
        | "at-o-i8-sf" => 1.0,
        _ => 0.0,
    }
}

pub fn moe_weight_classes() -> &'static [&'static str] {
    &[
        "moe-eg",
        "moe-eu",
        "moe-ed",
        "moe-eg-sf",
        "moe-eu-sf",
        "moe-ed-sf",
        "embed",
        "ln",
        "final-ln",
        "at-qw",
        "at-kw",
        "at-vw",
        "at-ow",
        "at-qn",
        "at-kn",
        "at-vn",
        "mlp-gw",
        "mlp-uw",
        "mlp-dw",
        "moe-router",
        "moe-rscale",
        "moe-pes",
        "moe-ones",
        "lmhead-i8",
        "lmhead-i8-sf",
        "at-qkv-i8",
        "at-qkv-i8-sf",
        "at-o-i8",
        "at-o-i8-sf",
    ]
}

struct Builder {
    ctx: &'static WgpuContext,
    passes: Vec<Pass>,
    prefill_passes: Vec<Pass>,
    to_prefill: bool,
    buffers: Vec<wgpu::Buffer>,
    alloc: std::collections::BTreeMap<String, (usize, u64)>,
    alloc_total: u64,
    since_flush: u64,
    expert_share: f64,
    weight_bpt: f64,
    wide_gemvs: usize,
    narrow_gemvs: usize,
    depthgate_flash1_pass_indices_full_attention_the_band_is_keyed_on_the_whole_context:
        Vec<usize>,
    depthgate_flash1_pass_indices_sliding_the_window_caps_the_scan_so_the_band_is_keyed_on_min_total_window:
        Vec<usize>,
    foldgate_alt_flash1_passes_one_deep_twin_per_full_attention_flash1_in_the_same_order: Vec<Pass>,
    storage_buffer_registry_indexes_every_hazard_tracked_buffer_the_graph_binds: Vec<wgpu::Buffer>,
    flash_scratch_bytes_per_full_attention_layer_sized_at_its_own_split_count: Vec<u64>,
    flash_scratch_bytes_per_sliding_layer_sized_at_its_own_split_count_not_the_max_of_the_two:
        Vec<u64>,
}

impl Builder {
    fn record(&mut self, label: &str, bytes: u64) {
        let class = vram_class(label);
        self.weight_bpt += bytes as f64 * per_token_weight_share(class, self.expert_share);
        let e = self.alloc.entry(class.to_string()).or_insert((0, 0));
        e.0 += 1;
        e.1 += bytes;
        self.alloc_total += bytes;
        self.since_flush += bytes;
    }

    fn report(&self) -> VramReport {
        let mut by_class: Vec<(String, usize, u64)> = self
            .alloc
            .iter()
            .map(|(k, (c, b))| (k.clone(), *c, *b))
            .collect();
        by_class.sort_by_key(|x| std::cmp::Reverse(x.2));
        VramReport {
            buffers: self.buffers.len(),
            total_bytes: self.alloc_total,
            by_class,
        }
    }

    fn flush_staging(&mut self) {
        if self.since_flush == 0 {
            return;
        }
        self.since_flush = 0;
        self.ctx.queue.submit(std::iter::empty());
        let _ = self.ctx.device.poll(wgpu::PollType::wait_indefinitely());
    }

    fn flush_staging_if_due(&mut self) {
        if staging_flush_enabled() && self.since_flush >= STAGING_FLUSH_BYTES {
            self.flush_staging();
        }
    }

    fn store(&mut self, label: &str, b: wgpu::Buffer) -> wgpu::Buffer {
        self.record(label, b.size());
        self.buffers.push(b.clone());
        b
    }

    fn zeros(&mut self, label: &str, bytes: u64) -> wgpu::Buffer {
        let b = dispatch::storage_zeroed(self.ctx, label, bytes.next_multiple_of(16));
        self.store(label, b)
    }

    fn upload_u32(&mut self, label: &str, data: &[u32]) -> wgpu::Buffer {
        let b = if data.len().is_multiple_of(4) {
            dispatch::storage_from_slice(self.ctx, label, data)
        } else {
            let mut padded = data.to_vec();
            padded.resize(data.len().next_multiple_of(4), 0);
            dispatch::storage_from_slice(self.ctx, label, &padded)
        };
        let b = self.store(label, b);
        self.flush_staging_if_due();
        b
    }

    fn upload_i32(&mut self, label: &str, data: &[i32]) -> wgpu::Buffer {
        let b = dispatch::storage_from_slice(self.ctx, label, data);
        self.store(label, b)
    }

    fn upload_f32(&mut self, label: &str, data: &[f32]) -> wgpu::Buffer {
        let b = dispatch::storage_from_slice(self.ctx, label, data);
        let b = self.store(label, b);
        self.flush_staging_if_due();
        b
    }

    fn uni<T: bytemuck::Pod>(&mut self, label: &str, v: T) -> wgpu::Buffer {
        let b = dispatch::uniform_from(self.ctx, label, &v);
        self.store(label, b)
    }

    fn built_pass(
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
        let mut storage_ids: Vec<(u32, bool)> = binds
            .iter()
            .filter(|(_, buf)| buf.usage().contains(wgpu::BufferUsages::STORAGE))
            .map(|(slot, buf)| {
                let w = wgsl_binding_is_storage_read_write_a_read_only_binding_can_never_raise_a_hazard(
                    source, *slot,
                );
                (self.intern_storage_buffer(buf), w)
            })
            .collect();
        storage_ids.sort_unstable();
        storage_ids.dedup();
        Ok(Pass {
            pipeline,
            bind,
            grid,
            label: format!("{label}:{entry}"),
            bind_bytes: binds.iter().map(|(_, b)| b.size()).sum(),
            storage_binds_are_what_wgpu_tracks_for_hazards_uniforms_never_transition: storage_ids,
        })
    }

    fn push(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let pass = self.built_pass(label, source, entry, binds, grid)?;
        let sink = if self.to_prefill {
            &mut self.prefill_passes
        } else {
            &mut self.passes
        };
        sink.push(pass);
        Ok(())
    }

    fn intern_storage_buffer(&mut self, buf: &wgpu::Buffer) -> u32 {
        let reg =
            &mut self.storage_buffer_registry_indexes_every_hazard_tracked_buffer_the_graph_binds;
        match reg.iter().position(|b| b == buf) {
            Some(i) => i as u32,
            None => {
                reg.push(buf.clone());
                (reg.len() - 1) as u32
            }
        }
    }

    fn grid1(&self, invocations: u64, wg: u32) -> (u32, u32, u32) {
        dispatch::workgroup_count_1d(self.ctx, invocations, wg)
    }

    fn wide_flag(
        &mut self,
        label: &str,
        load: GemvLoad,
        k: usize,
        x_off_words: usize,
        w: &wgpu::Buffer,
        x: &wgpu::Buffer,
    ) -> u32 {
        assert!(
            w.size().is_multiple_of(16) && x.size().is_multiple_of(16),
            "g4m_gemv_bf16 binds {} weight bytes and {} activation bytes; a vec4 view \
             truncates anything that is not a multiple of 16 and the tail of the row would \
             be silently dropped",
            w.size(),
            x.size()
        );
        if load != GemvLoad::Wide {
            return 0;
        }
        let ok = gemv_wide_enabled(label) && k.is_multiple_of(8) && x_off_words.is_multiple_of(4);
        if ok {
            self.wide_gemvs += 1;
        } else {
            self.narrow_gemvs += 1;
        }
        u32::from(ok)
    }
}

struct Sources {
    gemv_bf16: String,
    gemv_i8_v4: String,
    gemv_i8_g16: String,
    gemv_bf16_tailfold: String,
    gemv_bf16_rpnorm_sglat: String,
    gemv_bf16_dnrp_sglat: String,
    gemv_bf16_sglat: String,
    gemv_bf16_v2: String,
    prop_norm_x2_topk_sglat: String,
    gemv_w4: String,
    gemv_w4_sglat: String,
    gemv_w4_sglat_xv4: String,
    attn: String,
    attn_sglat: String,
    flash: String,
    flash_qkvfold: String,
    flash_fold_entry: Option<String>,
    flash_foldgate_deep_source_and_entry: Option<(String, String)>,
    moe: String,
    prop: String,
    prop_sglat: String,
    resscale: String,
    prefill: String,
    kv_fp8: String,
}

#[doc(hidden)]
pub fn nozi_audit_sources() -> Vec<(&'static str, String)> {
    vec![
        ("g4m:gemv_bf16", compose(GEMV_BF16_WGSL)),
        ("g4m:gemv_w4", compose(GEMV_W4E_WGSL)),
        ("g4m:attn", compose(ATTN_WGSL)),
        ("g4m:moe", compose(MOE_WGSL)),
        ("g4m:prop_norm", compose(PROP_NORM_WGSL)),
    ]
}

impl Sources {
    fn new(prefill_m: usize, flash_sg: bool, sacc_chunk: u32, fold: u32, foldgate: bool) -> Self {
        let flash_reduce = if flash_sg {
            FLASH_REDUCE_SG_SHUFFLE_XOR_CHAIN_16_8_4_2_1_IS_BIT_IDENTICAL_TO_THE_LDS_BUTTERFLY_SAME_OPERAND_ORDER
        } else {
            FLASH_REDUCE_WB_ELEVEN_WORKGROUP_BARRIERS_AND_TEN_LDS_ROUND_TRIPS_PER_KV_ROW_BATCH
        };
        let fold_src = (fold > 1).then(|| flash_stage1_fold_source(fold));
        Self {
            flash_fold_entry: fold_src
                .as_ref()
                .map(|_| flash_stage1_fold_entry(fold)),
            flash_foldgate_deep_source_and_entry: foldgate.then(|| {
                let fg_fold =
                    FOLDGATE_DEEP_FOLD_2_QUERY_HEADS_PER_WORKGROUP_HALVES_FULL_ATTENTION_KV_TRAFFIC;
                (
                    compose(&format!(
                        "{FLASH_DECODE_WGSL}\n{flash_reduce}\n{}\n{}",
                        flash_stage1_tail_source(
                            FOLDGATE_DEEP_SACC_128_KEEPS_THE_FOLDED_WORKGROUP_UNDER_A_THIRD_OF_48KB_SO_THREE_BLOCKS_PLACE_PER_SM_AND_IS_DIGIT_IDENTICAL_TO_SACC_512_ON_REAL_WEIGHTS
                        ),
                        flash_stage1_fold_source(fg_fold)
                    )),
                    flash_stage1_fold_entry(fg_fold),
                )
            }),
            gemv_bf16: compose(GEMV_BF16_WGSL),
            gemv_i8_v4: compose(GEMV_I8_V4_WGSL),
            gemv_i8_g16: compose(GEMV_I8_G16_WGSL),
            gemv_bf16_tailfold: compose(GEMV_BF16_TAILFOLD_WGSL),
            gemv_bf16_rpnorm_sglat: compose(GEMV_BF16_RPNORM_SGLAT_WGSL),
            gemv_bf16_dnrp_sglat: compose(GEMV_BF16_DNRP_SGLAT_WGSL),
            gemv_bf16_sglat: compose(GEMV_BF16_SGLAT_WGSL),
            gemv_bf16_v2: compose(GEMV_BF16_V2_WGSL),
            prop_norm_x2_topk_sglat: compose(PROP_NORM_X2_TOPK_SGLAT_WGSL),
            gemv_w4: compose(GEMV_W4E_WGSL),
            gemv_w4_sglat: compose(GEMV_W4E_SGLAT_WGSL),
            gemv_w4_sglat_xv4: compose(GEMV_W4E_SGLAT_XV4_WGSL),
            attn: compose(ATTN_WGSL),
            attn_sglat: compose(ATTN_SGLAT_WGSL),
            flash: compose(&format!(
                "{FLASH_DECODE_WGSL}\n{flash_reduce}\n{}\n{}",
                flash_stage1_tail_source(sacc_chunk),
                fold_src.clone().unwrap_or_default()
            )),
            flash_qkvfold: compose(FLASH_QKVFOLD_WGSL),
            moe: compose(MOE_WGSL),
            prop: compose(PROP_NORM_WGSL),
            prop_sglat: compose(PROP_NORM_SGLAT_WGSL),
            resscale: compose(wk::residual_scale::WGSL),
            prefill: if prefill_m > 0 {
                compose(&format!(
                    "{}\n{EMBED_ROW_SPLICE_WGSL}\n{VERIFY_ARGMAX_WGSL}",
                    prefill_wgsl(prefill_m)
                ))
            } else {
                String::new()
            },
            kv_fp8: compose(wk::kv_fp8::WGSL),
        }
    }
}

struct Bf16Gpu {
    w: wgpu::Buffer,
    n: usize,
    k: usize,
}

struct ExpertGpu {
    w: wgpu::Buffer,
    scales: wgpu::Buffer,
    n: usize,
    k: usize,
}

fn upload_bf16(b: &mut Builder, label: &str, l: &HostLin) -> Bf16Gpu {
    Bf16Gpu {
        w: b.upload_u32(label, &pack_pairs(&l.w)),
        n: l.n,
        k: l.k,
    }
}

fn upload_experts(b: &mut Builder, label: &str, st: &HostW4Stack) -> ExpertGpu {
    ExpertGpu {
        w: b.upload_u32(label, &st.packed),
        scales: b.upload_u32(&format!("{label}-sf"), &pack_pairs(&st.scales)),
        n: st.n,
        k: st.k,
    }
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_w4(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    e: &ExpertGpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    slots: usize,
    x_per_slot: bool,
) -> Result<()> {
    let groups = e.k / W4_GS;
    let entry = w4_gemv_entry(groups, e.n);
    let rows_per_wg = if entry == "g4m_gemv_w4_r8" { 8 } else { 2 };
    let grid = b.grid1(e.n.div_ceil(rows_per_wg) as u64, 1);
    let p = b.uni(
        "g4m-gemvw4-p",
        GemvW4Params {
            n_rows: e.n as u32,
            groups: groups as u32,
            groups_x: grid.0,
            w_e_stride_words: (e.n * e.k / 8) as u32,
            s_e_stride_elems: (e.n * groups) as u32,
            x_slot_stride_words: if x_per_slot { (e.k / 2) as u32 } else { 0 },
            y_slot_stride_words: (e.n / 2) as u32,
            k_top: 0,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.gemv_w4,
        entry,
        &[
            (10, &e.w),
            (11, &e.scales),
            (12, x),
            (14, &p),
            (15, y),
            (16, sel),
        ],
        (grid.0, grid.1, slots as u32),
    )
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GemvLoad {
    Wide,

    Scalar,
}

pub fn gemv_wide_enabled(label: &str) -> bool {
    match std::env::var("NV_G4MOE_GEMV_WIDE") {
        Err(_) => false,
        Ok(v) if v == "0" || v.is_empty() => false,
        Ok(v) if v == "1" => true,
        Ok(v) => v.split(',').any(|c| !c.is_empty() && label.contains(c)),
    }
}

pub fn lmhead_i8_entry_off_until_real_checkpoint_quality_gate() -> Option<&'static str> {
    match std::env::var("NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED") {
        Err(_) => None,
        Ok(v) => match v.as_str() {
            "" | "0" => None,
            "1" => Some("g4m_gemv_i8"),
            "v4" => Some("g4m_gemv_i8_v4"),
            other => panic!(
                "NV_G4MOE_LMHEAD_I8_QUALITY_UNGATED={other}: 0 keeps the bf16 head, \
                 1 routes g4m_gemv_i8, v4 routes g4m_gemv_i8_v4"
            ),
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_bf16_alpha(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    out_f32: bool,
    y_off_words: usize,
    alpha: f32,
    load: GemvLoad,
) -> Result<()> {
    let mut wide = b.wide_flag(label, load, w.k, 0, &w.w, x);
    let v2 = gemv_v2_active_implies_the_vec4_wide_layout(b.ctx, label, load, w.k, 0);
    if v2 {
        wide = 1;
    }
    let r8 = bf16_r8_enabled() && wide == 1 && w.n.is_multiple_of(8);
    let (entry, grid) = if v2 {
        (GEMV_BF16_V2_ENTRY, b.grid1(w.n.div_ceil(8) as u64, 1))
    } else if r8 {
        ("g4m_gemv_bf16_r8", b.grid1((w.n / 8) as u64, 1))
    } else if wide == 1 && bf16_acc4_enabled() {
        (
            "g4m_gemv_bf16_acc4_ilp_reorders_the_wide_dot_sum",
            b.grid1(w.n.div_ceil(2) as u64, 1),
        )
    } else if gemv_sglat_active(b.ctx, label) {
        ("g4m_gemv_bf16_sglat", b.grid1(w.n.div_ceil(2) as u64, 1))
    } else {
        ("g4m_gemv_bf16", b.grid1(w.n.div_ceil(2) as u64, 1))
    };
    let p = b.uni(
        "g4m-gemvb-p",
        GemvBf16Params {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: (w.k / 2) as u32,
            x_off_words: 0,
            y_off_words: y_off_words as u32,
            wide,
            alpha,
            ..Default::default()
        },
    );
    let src = if entry == "g4m_gemv_bf16_sglat" {
        &s.gemv_bf16_sglat
    } else if entry == GEMV_BF16_V2_ENTRY {
        &s.gemv_bf16_v2
    } else {
        &s.gemv_bf16
    };
    b.push(label, src, entry, &[(0, &w.w), (1, x), (2, &p), (3, y)], grid)
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_bf16(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    out_f32: bool,
    y_off_words: usize,
) -> Result<()> {
    push_gemv_bf16_alpha(
        b,
        s,
        label,
        w,
        x,
        y,
        out_f32,
        y_off_words,
        1.0,
        GemvLoad::Wide,
    )
}

struct AttnI8Gpu {
    q: wgpu::Buffer,
    scales: wgpu::Buffer,
    n: usize,
    k: usize,
    gs: usize,
}

fn upload_attn_i8(
    b: &mut Builder,
    label_q: &str,
    label_s: &str,
    w: &[u16],
    n: usize,
    k: usize,
) -> AttnI8Gpu {
    let gs = attn_i8_group_elems();
    let m = quantize_i8_host_group(w, n, k, gs);
    AttnI8Gpu {
        q: b.upload_u32(label_q, &m.q),
        scales: b.upload_u32(label_s, &pack_pairs(&m.scales)),
        n,
        k,
        gs,
    }
}

fn push_gemv_attn_i8(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &AttnI8Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
) -> Result<()> {
    anyhow::ensure!(
        w.k.is_multiple_of(16),
        "g4m_gemv_i8_v4 reads one vec4 of 16 int8 elems per k-word, so k={} must be a \
         multiple of 16",
        w.k
    );
    let grid = b.grid1(w.n.div_ceil(2) as u64, 1);
    let kw = w.k / 16;
    let p = b.uni(
        "g4m-gemvi8-p",
        GemvI8Params {
            n_rows: w.n as u32,
            k_words: kw as u32,
            groups_x: grid.0,
            out_f32: 0,
            w_row_words: kw as u32,
            x_off_words: 0,
            y_off_words: 0,
            s_row_elems: (w.k / w.gs) as u32,
            alpha: 1.0,
            ..Default::default()
        },
    );
    let (src, entry) = if w.gs == 16 {
        (
            &s.gemv_i8_g16,
            "g4m_gemv_i8_v4_g16_one_scale_per_vec4_word_halves_the_group_of_the_shared_i8_entry",
        )
    } else {
        (&s.gemv_i8_v4, "g4m_gemv_i8_v4")
    };
    b.push(
        label,
        src,
        entry,
        &[(0, &w.q), (1, x), (2, &p), (3, y), (4, &w.scales)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_bf16_gu_gelu(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    wg: &Bf16Gpu,
    wu: &Bf16Gpu,
    x: &wgpu::Buffer,
    act: &wgpu::Buffer,
    wide_label_of_the_unfused_gate_class: &str,
) -> Result<()> {
    anyhow::ensure!(
        wg.n == wu.n && wg.k == wu.k,
        "fused gate/up gemv needs twin shapes, got {}x{} vs {}x{}",
        wg.n,
        wg.k,
        wu.n,
        wu.k
    );
    let pairs = wg.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let wide = b.wide_flag(
        wide_label_of_the_unfused_gate_class,
        GemvLoad::Wide,
        wg.k,
        0,
        &wg.w,
        x,
    );
    let p = b.uni(
        "g4m-gemvb-p",
        GemvBf16Params {
            n_rows: wg.n as u32,
            k_words: (wg.k / 2) as u32,
            groups_x: grid.0,
            out_f32: 0,
            w_row_words: (wg.k / 2) as u32,
            x_off_words: 0,
            y_off_words: 0,
            wide,
            alpha: 1.0,
            ..Default::default()
        },
    );
    let (src, entry) = if mlpgu_fusedx_enabled() {
        (&s.gemv_bf16, "g4m_gemv_bf16_gu_gelu_fusedx")
    } else if gemv_sglat_active(b.ctx, label) {
        (&s.gemv_bf16_sglat, "g4m_gemv_bf16_gu_gelu_sglat")
    } else {
        (&s.gemv_bf16, "g4m_gemv_bf16_gu_gelu")
    };
    b.push(
        label,
        src,
        entry,
        &[(0, &wg.w), (1, x), (2, &p), (3, act), (4, &wu.w)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_w4_gu_gelu(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    eg: &ExpertGpu,
    eu: &ExpertGpu,
    x: &wgpu::Buffer,
    act: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    slots: usize,
    rank: Option<(&wgpu::Buffer, &wgpu::Buffer, usize)>,
) -> Result<()> {
    anyhow::ensure!(
        eg.n == eu.n && eg.k == eu.k,
        "fused w4 gate/up gemv needs twin shapes, got {}x{} vs {}x{}",
        eg.n,
        eg.k,
        eu.n,
        eu.k
    );
    let groups = eg.k / W4_GS;
    let base_entry = w4_gemv_entry(groups, eg.n);
    let coal = rank.is_none()
        && w4_coal_enabled()
        && base_entry != "g4m_gemv_w4_r8"
        && groups * 16 <= W4_COAL_XBUF_WORDS_3072_BOUNDS_THE_PRELOADED_ACTIVATION_SLOTS;
    let (entry, rows_per_wg) = if let Some(_) = rank {
        if base_entry == "g4m_gemv_w4_r8" {
            ("g4m_gemv_w4_r8_gu_gelu_rank", 8)
        } else {
            ("g4m_gemv_w4_gu_gelu_rank", 2)
        }
    } else if coal {
        ("g4m_gemv_w4_gu_gelu_coal", 2)
    } else if base_entry == "g4m_gemv_w4_r8" {
        ("g4m_gemv_w4_r8_gu_gelu", 8)
    } else if gu_fusedx_enabled() {
        if gemv_sglat_active(b.ctx, label) && w4_xv4_enabled() {
            ("g4m_gemv_w4_gu_gelu_fusedx_sglat_xv4", 2)
        } else if gemv_sglat_active(b.ctx, label) {
            ("g4m_gemv_w4_gu_gelu_fusedx_sglat", 2)
        } else {
            ("g4m_gemv_w4_gu_gelu_fusedx", 2)
        }
    } else {
        ("g4m_gemv_w4_gu_gelu", 2)
    };
    let grid = b.grid1(eg.n.div_ceil(rows_per_wg) as u64, 1);
    let p = b.uni(
        "g4m-gemvw4-p",
        GemvW4Params {
            n_rows: eg.n as u32,
            groups: groups as u32,
            groups_x: grid.0,
            w_e_stride_words: (eg.n * eg.k / 8) as u32,
            s_e_stride_elems: (eg.n * groups) as u32,
            x_slot_stride_words: 0,
            y_slot_stride_words: (eg.n / 2) as u32,
            k_top: if rank.is_some() { slots as u32 } else { 0 },
            n_experts: rank.map(|(_, _, ne)| ne as u32).unwrap_or(0),
            ..Default::default()
        },
    );
    let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
        (10, &eg.w),
        (11, &eg.scales),
        (12, x),
        (14, &p),
        (15, act),
        (17, &eu.w),
        (18, &eu.scales),
    ];
    match rank {
        Some((logits, _, _)) => binds.push((20, logits)),
        None => binds.push((16, sel)),
    }
    let src = if entry.ends_with("_sglat_xv4") {
        &s.gemv_w4_sglat_xv4
    } else if entry.ends_with("_sglat") {
        &s.gemv_w4_sglat
    } else {
        &s.gemv_w4
    };
    b.push(label, src, entry, &binds, (grid.0, grid.1, slots as u32))
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_w4_down_combine(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    ed: &ExpertGpu,
    act: &wgpu::Buffer,
    out: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    wts: &wgpu::Buffer,
    k_top: usize,
    rank: Option<(&wgpu::Buffer, &wgpu::Buffer, usize)>,
) -> Result<()> {
    let groups = ed.k / W4_GS;
    anyhow::ensure!(
        w4_gemv_entry(groups, ed.n) == "g4m_gemv_w4_r8",
        "g4m_gemv_w4_r8_down_combine mirrors the r8 row walk only; the caller must fall \
         back to the unfused down+combine pair for the pair-entry shape"
    );
    let coal = rank.is_none()
        && w4_coal_enabled()
        && k_top * (ed.k / 2) <= W4_COAL_XBUF_WORDS_3072_BOUNDS_THE_PRELOADED_ACTIVATION_SLOTS;
    let expar = rank.is_none() && !coal && down_expar_enabled();
    let (entry, rows_per_wg) = if rank.is_some() {
        ("g4m_gemv_w4_r8_down_combine_rank", 8)
    } else if coal {
        ("g4m_gemv_w4_r8_down_combine_coal", 8)
    } else if expar {
        if gemv_sglat_active(b.ctx, label) && w4_xv4_enabled() {
            ("g4m_gemv_w4_down_combine_expar_sglat_xv4", 2)
        } else if gemv_sglat_active(b.ctx, label) {
            ("g4m_gemv_w4_down_combine_expar_sglat", 2)
        } else {
            ("g4m_gemv_w4_down_combine_expar", 2)
        }
    } else {
        ("g4m_gemv_w4_r8_down_combine", 8)
    };
    let grid = b.grid1(ed.n.div_ceil(rows_per_wg) as u64, 1);
    let p = b.uni(
        "g4m-gemvw4-p",
        GemvW4Params {
            n_rows: ed.n as u32,
            groups: groups as u32,
            groups_x: grid.0,
            w_e_stride_words: (ed.n * ed.k / 8) as u32,
            s_e_stride_elems: (ed.n * groups) as u32,
            x_slot_stride_words: (ed.k / 2) as u32,
            y_slot_stride_words: (ed.n / 2) as u32,
            k_top: k_top as u32,
            n_experts: rank.map(|(_, _, ne)| ne as u32).unwrap_or(0),
            ..Default::default()
        },
    );
    let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
        (10, &ed.w),
        (11, &ed.scales),
        (12, act),
        (14, &p),
        (15, out),
        (17, &ed.w),
        (18, &ed.scales),
    ];
    match rank {
        Some((logits, pes, _)) => {
            binds.push((20, logits));
            binds.push((21, pes));
        }
        None => {
            binds.push((16, sel));
            binds.push((19, wts));
        }
    }
    let src = if entry.ends_with("_sglat_xv4") {
        &s.gemv_w4_sglat_xv4
    } else if entry.ends_with("_sglat") {
        &s.gemv_w4_sglat
    } else {
        &s.gemv_w4
    };
    b.push(label, src, entry, &binds, (grid.0, grid.1, 1))
}

#[allow(clippy::too_many_arguments)]
fn push_norm(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let p = b.uni(
        "g4m-pn-p",
        NormParams {
            hidden: hidden as u32,
            words: (hidden / 2) as u32,
            eps,
            scale: 0.0,
        },
    );
    b.push(
        label,
        &s.prop,
        "g4m_norm",
        &[(0, x), (1, w), (2, y), (3, &p)],
        (1, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_norm_residual(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    res: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
) -> Result<()> {
    let p = b.uni(
        "g4m-pnr-p",
        NormParams {
            hidden: hidden as u32,
            words: (hidden / 2) as u32,
            eps,
            scale: 0.0,
        },
    );
    b.push(
        label,
        &s.prop,
        "g4m_norm_residual",
        &[(0, x), (1, w), (2, y), (3, &p), (4, res)],
        (1, 1, 1),
    )
}

pub fn router_topk_entry(n_experts: usize, top_k: usize) -> &'static str {
    let forced_serial = matches!(
        std::env::var("NV_G4MOE_ROUTER_TOPK").ok().as_deref(),
        Some("serial") | Some("0")
    );
    if !forced_serial && n_experts <= 256 && top_k <= MAX_TOPK {
        "g4m_router_topk_par"
    } else {
        "g4m_router_topk"
    }
}

pub fn w4_gemv_entry(groups: usize, n_rows: usize) -> &'static str {
    let forced_wide = matches!(
        std::env::var("NV_G4MOE_W4_GEMV").ok().as_deref(),
        Some("wide") | Some("0")
    );
    if !forced_wide && groups <= 32 && n_rows.is_multiple_of(8) {
        "g4m_gemv_w4_r8"
    } else {
        "g4m_gemv_w4"
    }
}

pub const PREFILL_M_MAX: usize = 64;

pub fn prefill_m() -> usize {
    match std::env::var("NV_G4MOE_WGPU_PREFILL_M")
        .or_else(|_| std::env::var("NV_WGPU_PREFILL_M"))
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(0) | Some(1) => 0,
        Some(m) => m.clamp(2, PREFILL_M_MAX),
        None => 8,
    }
}

pub const G4MOE_SLIDING_KV_RING_DEFAULT_ON: bool = true;

pub fn sliding_kv_ring_enabled() -> bool {
    match std::env::var("NV_G4MOE_KV_RING") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_SLIDING_KV_RING_DEFAULT_ON,
    }
}

pub use crate::gemma4_wgpu::sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom;

pub const G4MOE_FLASH_DECODE_DEFAULT_ON: bool = true;

pub const G4MOE_FLASH_SPLITS_32_IS_THE_MEASURED_KNEE_AT_DEPTH: u32 = 32;

pub const G4MOE_FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH: u32 = 256;

pub fn flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build() -> u32 {
    match std::env::var("NV_G4MOE_FLASH_SPLITS") {
        Ok(v) => {
            let n: u32 = v.trim().parse().unwrap_or_else(|_| {
                panic!("NV_G4MOE_FLASH_SPLITS={v}: expected an integer split count")
            });
            assert!(
                n >= 1 && n <= G4MOE_FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH,
                "NV_G4MOE_FLASH_SPLITS={n} out of 1..={}: stage2 scans splits serially per head \
                 and the scratch buffer carries splits*(head_dim+2) f32 rows per head",
                G4MOE_FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH
            );
            n
        }
        Err(_) => G4MOE_FLASH_SPLITS_32_IS_THE_MEASURED_KNEE_AT_DEPTH,
    }
}

pub const G4MOE_FLASH_SLIDING_SPLITS_ENV: &str = "NV_G4MOE_FLASH_SPLITS_SLIDING";

pub const FLASH_SLIDING_SPLITS_UNSET_TRACKS_THE_FULL_ATTENTION_COUNT_SO_ADDING_THIS_KNOB_CHANGES_NO_EXISTING_ARM: &str =
    "NV_G4MOE_FLASH_SPLITS sizes the stage1 grid.y, the g4m-at-fscr scratch rows and the stage2 \
     serial scan for BOTH layer kinds, but the two kinds do not want the same number. Full \
     attention scans the whole context, so pairing NV_G4MOE_FLASH_FOLD=F with 32*F splits is what \
     holds its 512-workgroup grid while the fold halves its KV load requests. The sliding kind is \
     ring-capped at sliding_window+chunk+headroom rows (1160 on the 26B-A4B), which at 32 splits \
     is already ceil(1160/256)=5 rounds per workgroup: extra splits add no stage1 parallelism it \
     can use and are paid in full by g4m_flash_stage2_pk, whose scan over splits is serial and \
     lands on 25 sliding layers against 5 full-attention ones. Unset, this env returns the \
     full-attention count, so every arm measured before it existed reproduces byte for byte.";

pub fn flash_sliding_splits_the_ring_capped_scan_gains_no_stage1_rows_from_extra_splits_and_pays_every_one_in_stage2(
    full_attention_splits: u32,
) -> u32 {
    let Ok(v) = std::env::var(G4MOE_FLASH_SLIDING_SPLITS_ENV) else {
        return full_attention_splits;
    };
    let t = v.trim();
    if t.is_empty() {
        return full_attention_splits;
    }
    let n: u32 = t.parse().unwrap_or_else(|_| {
        panic!("{G4MOE_FLASH_SLIDING_SPLITS_ENV}={t}: expected an integer split count")
    });
    assert!(
        n >= 1 && n <= G4MOE_FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH,
        "{G4MOE_FLASH_SLIDING_SPLITS_ENV}={n} out of 1..={}. {}",
        G4MOE_FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH,
        FLASH_SLIDING_SPLITS_UNSET_TRACKS_THE_FULL_ATTENTION_COUNT_SO_ADDING_THIS_KNOB_CHANGES_NO_EXISTING_ARM
    );
    n
}

pub fn flash_splits_for_layer_kind_full_attention_holds_the_folded_grid_sliding_holds_the_stage2_scan(
    kind: LayerType,
    full_attention_splits: u32,
) -> u32 {
    match kind {
        LayerType::FullAttention => full_attention_splits,
        LayerType::SlidingAttention => {
            flash_sliding_splits_the_ring_capped_scan_gains_no_stage1_rows_from_extra_splits_and_pays_every_one_in_stage2(
                full_attention_splits,
            )
        }
    }
}

pub const G4MOE_FLASH_DEPTHGATE_ENV: &str = "NV_G4MOE_FLASH_DEPTHGATE";

pub const G4MOE_FLASH_DEPTHGATE_SPLITS_ENV: &str = "NV_G4MOE_FLASH_DEPTHGATE_SPLITS";

pub const FLASH_DEPTHGATE_KNEE_TOKENS_512_IS_WHERE_12_FIRST_BEATS_THE_SHALLOW_PIN_AND_BELOW_IT_8_WINS:
    u32 = 512;

pub const FLASH_DEPTHGATE_BAND1_SPLITS_12_IS_192_WORKGROUPS_ONE_PER_SM_AT_16_Q_HEADS_ON_188_SMS:
    u32 = 12;

pub const FLASH_DEPTHGATE_KNEE2_TOKENS_768_IS_WHERE_24_OVERTAKES_12_ON_THE_SAME_LADDER: u32 = 768;

pub const FLASH_DEPTHGATE_BAND2_SPLITS_24_IS_384_WORKGROUPS_TWO_PER_SM_AT_16_Q_HEADS_ON_188_SMS:
    u32 = 24;

pub const FLASH_DEPTHGATE_KNEE3_TOKENS_1024_IS_WHERE_32_OVERTAKES_24_AND_WINS_EVERY_DEPTH_BEYOND:
    u32 = 1024;

pub const FLASH_DEPTHGATE_BAND3_SPLITS_32_IS_512_WORKGROUPS_THE_MEASURED_RESIDENCY_CEILING_AND_47_AND_64_LOSE_ABOVE_IT:
    u32 = G4MOE_FLASH_SPLITS_32_IS_THE_MEASURED_KNEE_AT_DEPTH;

pub fn flash_depthgate_enabled_a_shallow_tuned_split_pin_starves_the_grid_at_depth() -> bool {
    matches!(std::env::var(G4MOE_FLASH_DEPTHGATE_ENV).as_deref(), Ok("1"))
}

fn flash_depthgate_forced_deep_splits_env_overrides_the_band_above_the_knee() -> Option<u32> {
    let v = std::env::var(G4MOE_FLASH_DEPTHGATE_SPLITS_ENV).ok()?;
    let n: u32 = v.trim().parse().unwrap_or_else(|_| {
        panic!("{G4MOE_FLASH_DEPTHGATE_SPLITS_ENV}={v}: expected an integer split count")
    });
    assert!(
        n >= 1 && n <= G4MOE_FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH,
        "{G4MOE_FLASH_DEPTHGATE_SPLITS_ENV}={n} out of 1..={}",
        G4MOE_FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH
    );
    Some(n)
}

pub fn flash_depthgate_splits_for_context_tokens(tokens: u32, shallow_splits: u32) -> u32 {
    if tokens
        < FLASH_DEPTHGATE_KNEE_TOKENS_512_IS_WHERE_12_FIRST_BEATS_THE_SHALLOW_PIN_AND_BELOW_IT_8_WINS
    {
        return shallow_splits;
    }
    if let Some(forced) = flash_depthgate_forced_deep_splits_env_overrides_the_band_above_the_knee()
    {
        return forced;
    }
    if tokens < FLASH_DEPTHGATE_KNEE2_TOKENS_768_IS_WHERE_24_OVERTAKES_12_ON_THE_SAME_LADDER {
        FLASH_DEPTHGATE_BAND1_SPLITS_12_IS_192_WORKGROUPS_ONE_PER_SM_AT_16_Q_HEADS_ON_188_SMS
    } else if tokens
        < FLASH_DEPTHGATE_KNEE3_TOKENS_1024_IS_WHERE_32_OVERTAKES_24_AND_WINS_EVERY_DEPTH_BEYOND
    {
        FLASH_DEPTHGATE_BAND2_SPLITS_24_IS_384_WORKGROUPS_TWO_PER_SM_AT_16_Q_HEADS_ON_188_SMS
    } else {
        FLASH_DEPTHGATE_BAND3_SPLITS_32_IS_512_WORKGROUPS_THE_MEASURED_RESIDENCY_CEILING_AND_47_AND_64_LOSE_ABOVE_IT
    }
}

pub fn flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(
    shallow_splits: u32,
) -> u32 {
    if !flash_depthgate_enabled_a_shallow_tuned_split_pin_starves_the_grid_at_depth() {
        return shallow_splits;
    }
    match flash_depthgate_forced_deep_splits_env_overrides_the_band_above_the_knee() {
        Some(forced) => shallow_splits.max(forced),
        None => shallow_splits
            .max(FLASH_DEPTHGATE_BAND1_SPLITS_12_IS_192_WORKGROUPS_ONE_PER_SM_AT_16_Q_HEADS_ON_188_SMS)
            .max(FLASH_DEPTHGATE_BAND2_SPLITS_24_IS_384_WORKGROUPS_TWO_PER_SM_AT_16_Q_HEADS_ON_188_SMS)
            .max(FLASH_DEPTHGATE_BAND3_SPLITS_32_IS_512_WORKGROUPS_THE_MEASURED_RESIDENCY_CEILING_AND_47_AND_64_LOSE_ABOVE_IT),
    }
}

pub const G4MOE_FLASH_FOLDGATE_ENV: &str = "NV_G4MOE_FLASH_FOLDGATE";

pub const FOLDGATE_KNEE_TOKENS_10240_IS_THE_MEASURED_SIGN_CROSS_EVERY_FOLD_ARM_LOSS_SITS_SHALLOWER_AND_BOTH_STANDING_TARGETS_16K_AND_196K_SIT_DEEPER:
    u32 = 10240;

pub const FOLDGATE_DEEP_FOLD_2_QUERY_HEADS_PER_WORKGROUP_HALVES_FULL_ATTENTION_KV_TRAFFIC: u32 = 2;

pub const FOLDGATE_DEEP_SPLITS_64_RESTORES_512_WORKGROUPS_UNDER_FOLD_2_AND_IS_A_MEASURED_RUNG_THE_LADDER_HAS_A_REPRODUCED_POTHOLE_AT_40_47_48_SO_NEVER_INTERPOLATE:
    u32 = 64;

pub const FOLDGATE_DEEP_SACC_128_KEEPS_THE_FOLDED_WORKGROUP_UNDER_A_THIRD_OF_48KB_SO_THREE_BLOCKS_PLACE_PER_SM_AND_IS_DIGIT_IDENTICAL_TO_SACC_512_ON_REAL_WEIGHTS:
    u32 = 128;

pub const FOLDGATE_SWAPS_ONLY_FULL_ATTENTION_BECAUSE_SPELLING_THE_FOLD_THROUGH_THE_SHARED_DEPTHGATE_SPLITS_ENV_TAXES_SLIDING: &str =
    "NV_G4MOE_FLASH_FOLDGATE=1 swaps the FULL-ATTENTION flash1 passes to the fold-2 / splits-64 \
     / sacc-128 arm once the whole-context depth reaches the knee, and leaves every sliding pass \
     on its own shallow pin or depthgate band; forcing the fold's 64 splits onto both kinds via \
     NV_G4MOE_FLASH_DEPTHGATE_SPLITS was measured +7.2% on the sliding kind at 1024, and at 256 \
     the sliding split override alone was two thirds of the fold arm's whole control-band loss";

pub fn flash_foldgate_enabled_full_attention_swaps_to_the_fold2_arm_at_depth_while_sliding_keeps_its_own_band(
) -> bool {
    matches!(std::env::var(G4MOE_FLASH_FOLDGATE_ENV).as_deref(), Ok("1"))
}

pub fn foldgate_deep_at_these_whole_context_tokens(tokens: u32) -> bool {
    tokens
        >= FOLDGATE_KNEE_TOKENS_10240_IS_THE_MEASURED_SIGN_CROSS_EVERY_FOLD_ARM_LOSS_SITS_SHALLOWER_AND_BOTH_STANDING_TARGETS_16K_AND_196K_SIT_DEEPER
}

pub fn foldgate_scratch_splits_full_attention_only_must_also_cover_the_deep_arm_rows(
    covered_splits: u32,
) -> u32 {
    if flash_foldgate_enabled_full_attention_swaps_to_the_fold2_arm_at_depth_while_sliding_keeps_its_own_band()
    {
        covered_splits.max(
            FOLDGATE_DEEP_SPLITS_64_RESTORES_512_WORKGROUPS_UNDER_FOLD_2_AND_IS_A_MEASURED_RUNG_THE_LADDER_HAS_A_REPRODUCED_POTHOLE_AT_40_47_48_SO_NEVER_INTERPOLATE,
        )
    } else {
        covered_splits
    }
}

const FLASH_WORKGROUP_BYTES_QSH512_RED256_SM8_SL8_SACC4096_F32: u32 =
    (512 + 256 + 8 + 8 + 4096) * 4;

const FLASH_QKVFOLD_WORKGROUP_BYTES_ADDS_BUF512_KCUR256_VCUR256_U32: u32 =
    FLASH_WORKGROUP_BYTES_QSH512_RED256_SM8_SL8_SACC4096_F32 + (512 + 256 + 256) * 4;

pub fn flash_decode_enabled() -> bool {
    match std::env::var("NV_G4MOE_FLASH_DECODE") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_FLASH_DECODE_DEFAULT_ON,
    }
}

pub const G4MOE_FLASH_SG_DEFAULT_OFF_UNTIL_A_ROUND_PUBLISHES_ITS_LADDER: bool = false;

pub fn flash_sg_reduce_enabled() -> bool {
    match std::env::var("NV_G4MOE_FLASH_SG") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_FLASH_SG_DEFAULT_OFF_UNTIL_A_ROUND_PUBLISHES_ITS_LADDER,
    }
}

pub const FLASH_SACC_CHUNK_ENV: &str = "NV_G4MOE_FLASH_SACC";

pub const FLASH_SACC_CHUNK_512_IS_THE_SHIPPED_WHOLE_HEAD_DIM_STAGING_AT_16KB_PER_WORKGROUP: u32 =
    512;

pub fn flash_sacc_chunk_head_dim_slice_staged_per_warp_read_once_at_build() -> u32 {
    let hd_max = MAX_HEAD_DIM as u32;
    match std::env::var(FLASH_SACC_CHUNK_ENV) {
        Ok(v) => {
            let t = v.trim();
            if t.is_empty() {
                return FLASH_SACC_CHUNK_512_IS_THE_SHIPPED_WHOLE_HEAD_DIM_STAGING_AT_16KB_PER_WORKGROUP;
            }
            let n: u32 = t.parse().unwrap_or_else(|_| {
                panic!("{FLASH_SACC_CHUNK_ENV}={t}: expected a head-dim slice width in elements")
            });
            assert!(
                n >= 32 && n <= hd_max && n.is_multiple_of(32) && hd_max.is_multiple_of(n),
                "{FLASH_SACC_CHUNK_ENV}={n}: the stage1 tail stages one 32-lane accumulator strip \
                 at a time, so the slice must be a multiple of 32 that divides {hd_max}"
            );
            n
        }
        Err(_) => FLASH_SACC_CHUNK_512_IS_THE_SHIPPED_WHOLE_HEAD_DIM_STAGING_AT_16KB_PER_WORKGROUP,
    }
}

pub fn flash_workgroup_bytes_qsh512_sm8_sl8_plus_red256_only_when_wb_plus_sacc_8_warps_by_chunk(
    flash_sg: bool,
    sacc_chunk: u32,
    fold: u32,
) -> u32 {
    (512 * fold + if flash_sg { 0 } else { 256 } + 8 + 8 + 8 * sacc_chunk) * 4
}

pub fn flash_stage1_tail_source(sacc_chunk: u32) -> String {
    use std::fmt::Write;
    let hd_max = MAX_HEAD_DIM as u32;
    assert!(
        sacc_chunk >= 32
            && sacc_chunk <= hd_max
            && sacc_chunk.is_multiple_of(32)
            && hd_max.is_multiple_of(sacc_chunk),
        "flash stage1 tail slice {sacc_chunk} must be a multiple of 32 dividing {hd_max}"
    );
    let strips_per_chunk = sacc_chunk / 32;
    let chunks = hd_max / sacc_chunk;
    let mut b = String::with_capacity(8192);
    writeln!(
        b,
        "var<workgroup> gf_sacc: array<f32, {}>;\n",
        8 * sacc_chunk
    )
    .unwrap();
    b.push_str("fn gf_stage1_tail(lid: u32, lane: u32, warp: u32, hd: u32, slot: u32, m: f32, l: f32");
    for i in 0..(hd_max / 32) {
        write!(b, ", a{i}: f32").unwrap();
    }
    b.push_str(
        ") {
    workgroupBarrier();
    if (lane == 0u) {
        gf_sm[warp] = m;
        gf_sl[warp] = l;
    }
    workgroupBarrier();
    if (warp == 0u) {
        var m_hd = gf_neg_inf();
        for (var w = 0u; w < GF_WARPS; w = w + 1u) {
            m_hd = max(m_hd, gf_sm[w]);
        }
        var l_hd = 0.0;
        for (var w = 0u; w < GF_WARPS; w = w + 1u) {
            if (gf_sm[w] > gf_neg_inf()) {
                l_hd = l_hd + gf_sl[w] * gf_exp(gf_sm[w] - m_hd);
            }
        }
        if (lane == 0u) {
            gf_scratch[slot] = m_hd;
            gf_scratch[slot + 1u] = l_hd;
        }
    }
    var m_blk = gf_neg_inf();
    for (var w = 0u; w < GF_WARPS; w = w + 1u) {
        m_blk = max(m_blk, gf_sm[w]);
    }
",
    );
    for c in 0..chunks {
        b.push_str("    workgroupBarrier();\n");
        for k in 0..strips_per_chunk {
            let i = c * strips_per_chunk + k;
            writeln!(
                b,
                "    {{
        let d = lane + {i}u * GF_LANES;
        if (d < hd) {{
            gf_sacc[warp * {sacc_chunk}u + lane + {k}u * GF_LANES] = a{i};
        }}
    }}"
            )
            .unwrap();
        }
        b.push_str("    workgroupBarrier();\n");
        writeln!(
            b,
            "    for (var dl = lid; dl < {sacc_chunk}u; dl = dl + GF_BLOCK) {{
        let d = dl + {}u;
        if (d < hd) {{
            var a = 0.0;
            for (var w = 0u; w < GF_WARPS; w = w + 1u) {{
                if (gf_sm[w] > gf_neg_inf()) {{
                    a = a + gf_sacc[w * {sacc_chunk}u + dl] * gf_exp(gf_sm[w] - m_blk);
                }}
            }}
            gf_scratch[slot + 2u + d] = a;
        }}
    }}",
            c * sacc_chunk
        )
        .unwrap();
    }
    b.push_str("}\n");
    b
}

pub const FLASH_GQA_FOLD_ENV: &str = "NV_G4MOE_FLASH_FOLD";

pub const FLASH_GQA_FOLD_1_IS_THE_SHIPPED_ONE_WORKGROUP_PER_QUERY_HEAD_DISPATCH: u32 = 1;

pub const FLASH_GQA_FOLD_MAX_8_IS_THE_FULL_ATTENTION_GROUP_AND_128_ACCUMULATORS_PER_THREAD: u32 = 8;

pub fn flash_gqa_fold_query_heads_per_workgroup_read_once_at_build() -> u32 {
    match std::env::var(FLASH_GQA_FOLD_ENV) {
        Ok(v) => {
            let t = v.trim();
            if t.is_empty() {
                return FLASH_GQA_FOLD_1_IS_THE_SHIPPED_ONE_WORKGROUP_PER_QUERY_HEAD_DISPATCH;
            }
            let n: u32 = t.parse().unwrap_or_else(|_| {
                panic!("{FLASH_GQA_FOLD_ENV}={t}: expected a query-head fold count")
            });
            assert!(
                n >= 1
                    && n <= FLASH_GQA_FOLD_MAX_8_IS_THE_FULL_ATTENTION_GROUP_AND_128_ACCUMULATORS_PER_THREAD
                    && n.is_power_of_two(),
                "{FLASH_GQA_FOLD_ENV}={n}: a folded workgroup carries head_dim/32 more live \
                 accumulators per thread for every extra query head, so the fold must be a power \
                 of two in 1..={}",
                FLASH_GQA_FOLD_MAX_8_IS_THE_FULL_ATTENTION_GROUP_AND_128_ACCUMULATORS_PER_THREAD
            );
            n
        }
        Err(_) => FLASH_GQA_FOLD_1_IS_THE_SHIPPED_ONE_WORKGROUP_PER_QUERY_HEAD_DISPATCH,
    }
}

pub const FLASH_GQA_FOLD_DIVIDES_THE_GRID_SO_IT_ONLY_PAYS_PAIRED_WITH_SPLITS_TIMES_FOLD_AND_A_SACC_SLICE: &str =
    "NV_G4MOE_FLASH_FOLD=F puts F query heads of one KV group in a workgroup, so each K and V \
     element is loaded once instead of F times -- and it divides the full-attention stage1 grid \
     by F. At the shipped NV_G4MOE_FLASH_SPLITS=32 that grid is 16x32=512 workgroups on 188 SMs, \
     and every fold rung alone loses: +10.9% at F=2, +15.5% at F=4, +50.3% at F=8 on \
     g4m-at-flash1-full at 196608. Pair the fold with NV_G4MOE_FLASH_SPLITS=32*F to hold the grid \
     at 512, and with NV_G4MOE_FLASH_SACC<=256 so the folded workgroup (which adds gf_qshf at \
     512*F floats) stays under 49152/3 bytes and three blocks place per SM: fold 2 / splits 64 / \
     slice 128 measures -25.8% on the pass and -16.1% on the token at 196608. It is a slope fix, \
     not an offset fix -- at 16384 the same arm is +4.1%, because the doubled split count is paid \
     back in g4m_flash_stage2_pk's serial scan over splits, which is depth-independent and lands \
     on all 30 layers while the fold only helps the 5 full-attention ones";

pub const G4MOE_FLASH_SLIDING_FOLD_ENV: &str = "NV_G4MOE_FLASH_FOLD_SLIDING";

pub const FLASH_SLIDING_FOLD_HALVES_A_GRID_THE_RING_IS_TOO_SHORT_TO_REFILL_AND_IS_WHAT_FORCES_THE_SLIDING_SPLIT_TAX: &str =
    "flash_gqa_fold_fits_this_layer accepts the sliding layers too -- their GQA group is even -- so \
     a global NV_G4MOE_FLASH_FOLD folds all 30 layers. On full attention that trade is the whole \
     win: the scan is the entire context, so halving the KV load requests per query head pays for \
     the halved grid many times over once NV_G4MOE_FLASH_SPLITS restores it. On sliding it is a \
     pure loss twice over. The ring caps the scan at sliding_window+chunk+headroom rows (1160 on \
     the 26B-A4B), so a 16x32 grid already runs only about five rounds per workgroup and there is \
     nothing for the reuse to amortise against: measured at EQUAL grid, folding sliding costs +8.3% \
     on g4m-at-flash1-sliding at 16384. And it is the fold that forces the doubled split count, \
     whose serial g4m_flash_stage2_pk scan lands on 25 sliding layers against 5 full-attention \
     ones. Unset, this env returns the full-attention fold, so every arm measured before it \
     existed reproduces exactly.";

pub fn flash_sliding_fold_the_ring_is_too_short_for_kv_reuse_to_pay_for_the_grid_it_deletes(
    full_attention_fold: u32,
) -> u32 {
    let Ok(v) = std::env::var(G4MOE_FLASH_SLIDING_FOLD_ENV) else {
        return full_attention_fold;
    };
    let t = v.trim();
    if t.is_empty() {
        return full_attention_fold;
    }
    let n: u32 = t.parse().unwrap_or_else(|_| {
        panic!("{G4MOE_FLASH_SLIDING_FOLD_ENV}={t}: expected a query-head fold count")
    });
    assert!(
        n == 1 || n == full_attention_fold,
        "{G4MOE_FLASH_SLIDING_FOLD_ENV}={n}: Sources::new generates exactly one folded stage1 \
         entry, at the full-attention fold {full_attention_fold}, so the sliding kind may only \
         take that entry or the unfolded one. {}",
        FLASH_SLIDING_FOLD_HALVES_A_GRID_THE_RING_IS_TOO_SHORT_TO_REFILL_AND_IS_WHAT_FORCES_THE_SLIDING_SPLIT_TAX
    );
    n
}

pub fn flash_gqa_fold_for_layer_kind_only_full_attention_scans_enough_rows_to_amortise_a_halved_grid(
    kind: LayerType,
    full_attention_fold: u32,
) -> u32 {
    match kind {
        LayerType::FullAttention => full_attention_fold,
        LayerType::SlidingAttention => {
            flash_sliding_fold_the_ring_is_too_short_for_kv_reuse_to_pay_for_the_grid_it_deletes(
                full_attention_fold,
            )
        }
    }
}

pub fn flash_stage1_fold_entry(fold: u32) -> String {
    format!("g4m_flash_stage1_bf16_fold{fold}")
}

pub fn flash_gqa_fold_fits_this_layer(fold: u32, n_q: usize, n_kv: usize) -> bool {
    fold > 1
        && n_kv > 0
        && n_q.is_multiple_of(n_kv)
        && n_q.is_multiple_of(fold as usize)
        && (n_q / n_kv).is_multiple_of(fold as usize)
}

pub fn flash_stage1_fold_source(fold: u32) -> String {
    use std::fmt::Write;
    let hd_max = MAX_HEAD_DIM as u32;
    let accs = hd_max / 32;
    let entry = flash_stage1_fold_entry(fold);
    let mut b = String::with_capacity(32768);
    writeln!(
        b,
        "var<workgroup> gf_qshf{fold}: array<f32, {}>;\n",
        hd_max * fold
    )
    .unwrap();
    writeln!(b, "@compute @workgroup_size(256)").unwrap();
    writeln!(b, "fn {entry}(").unwrap();
    b.push_str(
        "    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {
",
    );
    writeln!(b, "    let h0 = wg.x * {fold}u;").unwrap();
    b.push_str(
        "    let split = wg.y;
    if (h0 >= gf_p.n_heads) {
        return;
    }
    let hd = gf_p.head_dim;
    let nkv = gf_p.n_kv;
    let group = gf_p.n_heads / nkv;
    let kvh = h0 / group;
    let lid = tid.x;
    let lane = lid & 31u;
    let warp = lid >> 5u;

",
    );
    writeln!(
        b,
        "    for (var d = lid; d < hd * {fold}u; d = d + GF_BLOCK) {{
        let qi = h0 * hd + d;
        gf_qshf{fold}[d] = bf16_decode(u16_at(gf_q[qi >> 1u], qi));
    }}
    workgroupBarrier();
"
    )
    .unwrap();
    for j in 0..fold {
        writeln!(b, "    var m{j} = gf_neg_inf();").unwrap();
        writeln!(b, "    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "    var a{j}_{i} = 0.0;").unwrap();
        }
    }
    b.push_str(
        "
    let total = gf_p.total;
    let base = gf_p.start + split * GF_WARPS;
    let stride = gf_p.splits * GF_WARPS;
    var rounds = 0u;
    if (total > base) {
        rounds = (total - base + stride - 1u) / stride;
    }
    let use_vec8 = (hd & 7u) == 0u;

    for (var r = 0u; r < rounds; r = r + 1u) {
        let p = base + warp + r * stride;
        let live = p < total;
        var sp = p;
        if (gf_p.ring > 0u) {
            sp = p % gf_p.ring;
        }
",
    );
    for j in 0..fold {
        writeln!(b, "        var pt{j} = 0.0;").unwrap();
    }
    b.push_str(
        "        if (live) {
            let kbase = (sp * nkv + kvh) * hd;
            if (use_vec8) {
                let n8 = hd >> 3u;
                for (var jv = lane; jv < n8; jv = jv + GF_LANES) {
                    let qb = jv * 8u;
                    let kb = kbase + qb;
                    for (var t = 0u; t < 4u; t = t + 1u) {
                        let kx = gf_k_bf16(kb + 2u * t);
                        let ky = gf_k_bf16(kb + 2u * t + 1u);
",
    );
    for j in 0..fold {
        writeln!(
            b,
            "                        pt{j} = pt{j} + fma(kx, gf_qshf{fold}[{j}u * hd + qb + 2u * t], ky \
             * gf_qshf{fold}[{j}u * hd + qb + 2u * t + 1u]);"
        )
        .unwrap();
    }
    b.push_str(
        "                    }
                }
            } else {
                for (var d = lane; d < hd; d = d + GF_LANES) {
                    let kx = gf_k_bf16(kbase + d);
",
    );
    for j in 0..fold {
        writeln!(
            b,
            "                    pt{j} = fma(gf_qshf{fold}[{j}u * hd + d], kx, pt{j});"
        )
        .unwrap();
    }
    b.push_str(
        "                }
            }
        }
",
    );
    for j in 0..fold {
        writeln!(
            b,
            "        let sc{j} = gf_warp_sum(lid, pt{j}) * gf_p.scale;"
        )
        .unwrap();
    }
    b.push_str(
        "        if (live) {
            let vbase = (sp * nkv + kvh) * hd;
",
    );
    for j in 0..fold {
        writeln!(b, "            let mn{j} = max(m{j}, sc{j});").unwrap();
        writeln!(b, "            let cr{j} = gf_exp(m{j} - mn{j});").unwrap();
        writeln!(b, "            let wt{j} = gf_exp(sc{j} - mn{j});").unwrap();
        writeln!(b, "            l{j} = fma(l{j}, cr{j}, wt{j});").unwrap();
    }
    for i in 0..accs {
        writeln!(b, "            {{").unwrap();
        writeln!(b, "                let d = lane + {i}u * GF_LANES;").unwrap();
        writeln!(b, "                if (d < hd) {{").unwrap();
        writeln!(b, "                    let vx = gf_v_bf16(vbase + d);").unwrap();
        for j in 0..fold {
            writeln!(
                b,
                "                    a{j}_{i} = fma(a{j}_{i}, cr{j}, wt{j} * vx);"
            )
            .unwrap();
        }
        writeln!(b, "                }}").unwrap();
        writeln!(b, "            }}").unwrap();
    }
    for j in 0..fold {
        writeln!(b, "            m{j} = mn{j};").unwrap();
    }
    b.push_str(
        "        }
    }

",
    );
    for j in 0..fold {
        writeln!(
            b,
            "    gf_stage1_tail(lid, lane, warp, hd, ((h0 + {j}u) * gf_p.splits + split) * (hd + \
             2u), m{j}, l{j},"
        )
        .unwrap();
        let args: Vec<String> = (0..accs).map(|i| format!("a{j}_{i}")).collect();
        writeln!(b, "        {});", args.join(", ")).unwrap();
    }
    b.push_str("}\n");
    b
}

pub const G4MOE_FLASH_SLIDING_DEFAULT_ON: bool = true;

pub fn flash_sliding_enabled() -> bool {
    if !flash_decode_enabled() {
        return false;
    }
    match std::env::var("NV_G4MOE_FLASH_SLIDING") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_FLASH_SLIDING_DEFAULT_ON,
    }
}

pub const G4MOE_FUSION_CHAMPION_ROWS_G4MOE_FUSION_R1_R2_R2C_R3_111_1_VS_87_3_TOK_S_AT_256_TF_PPL_31_9_ACC_0_5254_BOTH_ARMS_DEFAULT_ON:
    bool = true;

fn env_flag(name: &str) -> Option<bool> {
    std::env::var(name).ok().map(|v| {
        let t = v.trim();
        !t.is_empty() && t != "0"
    })
}

fn champion_defaults_qualify_subgroup_width_32_otherwise_prior_defaults_silently(
    ctx: &WgpuContext,
) -> bool {
    G4MOE_FUSION_CHAMPION_ROWS_G4MOE_FUSION_R1_R2_R2C_R3_111_1_VS_87_3_TOK_S_AT_256_TF_PPL_31_9_ACC_0_5254_BOTH_ARMS_DEFAULT_ON
        && ctx.caps.subgroup
        && ctx.subgroup_width() == Some(32)
}

pub const G4MOE_GROUPED_GEMV_DEFAULT_ON: bool =
    G4MOE_FUSION_CHAMPION_ROWS_G4MOE_FUSION_R1_R2_R2C_R3_111_1_VS_87_3_TOK_S_AT_256_TF_PPL_31_9_ACC_0_5254_BOTH_ARMS_DEFAULT_ON;

pub const GROUPED_NORM_CHAIN_MAX_HIDDEN_WORDS_PN_BUF: usize = 2048;

pub fn grouped_gemv_enabled() -> bool {
    env_flag("NV_G4MOE_GROUPED_GEMV").unwrap_or(G4MOE_GROUPED_GEMV_DEFAULT_ON)
}

fn grouped_gemv_active(ctx: &WgpuContext) -> bool {
    env_flag("NV_G4MOE_GROUPED_GEMV")
        .unwrap_or_else(|| champion_defaults_qualify_subgroup_width_32_otherwise_prior_defaults_silently(ctx))
}

pub const G4MOE_TOPK_FOLD_DEFAULT_ON: bool = false;

pub fn topk_fold_enabled() -> bool {
    match std::env::var("NV_G4MOE_TOPK_FOLD") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_TOPK_FOLD_DEFAULT_ON,
    }
}

pub const G4MOE_HEAD_ONE_SHARD_LETS_THE_EMBED_TABLE_BIND_WHOLE_INSTEAD_OF_AT_A_1_GIB_CEILING_DEFAULT_ON:
    bool = false;

pub fn head_one_shard_enabled() -> bool {
    match std::env::var("NV_G4MOE_HEAD_ONE_SHARD") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => {
            G4MOE_HEAD_ONE_SHARD_LETS_THE_EMBED_TABLE_BIND_WHOLE_INSTEAD_OF_AT_A_1_GIB_CEILING_DEFAULT_ON
        }
    }
}

pub const G4MOE_LMHEAD_TIGHT_GRID_SEPARATES_THE_IDLE_WORKGROUP_LEVER_FROM_THE_BARRIER_LEVER_DEFAULT_ON:
    bool = false;

pub fn lmhead_tight_grid_enabled() -> bool {
    match std::env::var("NV_G4MOE_LMHEAD_TIGHT_GRID") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => {
            G4MOE_LMHEAD_TIGHT_GRID_SEPARATES_THE_IDLE_WORKGROUP_LEVER_FROM_THE_BARRIER_LEVER_DEFAULT_ON
        }
    }
}

pub const G4MOE_RPNORM_DEFAULT_ON: bool =
    G4MOE_FUSION_CHAMPION_ROWS_G4MOE_FUSION_R1_R2_R2C_R3_111_1_VS_87_3_TOK_S_AT_256_TF_PPL_31_9_ACC_0_5254_BOTH_ARMS_DEFAULT_ON;

pub fn rpnorm_enabled() -> bool {
    env_flag("NV_G4MOE_RPNORM").unwrap_or(G4MOE_RPNORM_DEFAULT_ON)
}

fn rpnorm_active(ctx: &WgpuContext) -> bool {
    env_flag("NV_G4MOE_RPNORM").unwrap_or_else(|| {
        champion_defaults_qualify_subgroup_width_32_otherwise_prior_defaults_silently(ctx)
            && grouped_gemv_active(ctx)
            && !router_chain_enabled()
    })
}

pub const G4MOE_RPFUSE_RIDES_THE_ROUTER_NORM_GEMV_ON_THE_MLP_DOWN_DISPATCH_DEFAULT_ON: bool =
    G4MOE_FUSION_CHAMPION_ROWS_G4MOE_FUSION_R1_R2_R2C_R3_111_1_VS_87_3_TOK_S_AT_256_TF_PPL_31_9_ACC_0_5254_BOTH_ARMS_DEFAULT_ON;

pub fn rpfuse_enabled() -> bool {
    env_flag("NV_G4MOE_RPFUSE")
        .unwrap_or(G4MOE_RPFUSE_RIDES_THE_ROUTER_NORM_GEMV_ON_THE_MLP_DOWN_DISPATCH_DEFAULT_ON)
}

fn rpfuse_active(ctx: &WgpuContext) -> bool {
    env_flag("NV_G4MOE_RPFUSE").unwrap_or_else(|| {
        champion_defaults_qualify_subgroup_width_32_otherwise_prior_defaults_silently(ctx)
            && grouped_gemv_active(ctx)
            && rpnorm_active(ctx)
            && !router_chain_enabled()
            && !topk_fold_enabled()
    })
}

pub const G4MOE_TOPK_NX2_RIDES_THE_ROUTER_TOPK_ON_THE_NORMX2_DISPATCH_DEFAULT_ON: bool =
    G4MOE_FUSION_CHAMPION_ROWS_G4MOE_FUSION_R1_R2_R2C_R3_111_1_VS_87_3_TOK_S_AT_256_TF_PPL_31_9_ACC_0_5254_BOTH_ARMS_DEFAULT_ON;

pub const TOPK_PAR_TIE_RULE_HIGHER_LOGIT_OUTRANKS_AND_EQUAL_LOGITS_RANK_BY_LOWER_EXPERT_INDEX:
    &str = "w > v || (w == v && j < lid)";

pub fn topk_nx2_enabled() -> bool {
    env_flag("NV_G4MOE_TOPK_NX2")
        .unwrap_or(G4MOE_TOPK_NX2_RIDES_THE_ROUTER_TOPK_ON_THE_NORMX2_DISPATCH_DEFAULT_ON)
}

fn topk_nx2_active(ctx: &WgpuContext, n_experts: usize, top_k: usize) -> bool {
    env_flag("NV_G4MOE_TOPK_NX2").unwrap_or_else(|| {
        rpfuse_active(ctx) && router_topk_entry(n_experts, top_k) == "g4m_router_topk_par"
    })
}

pub const G4MOE_BF16_R8_DEFAULT_ON: bool = false;

pub fn bf16_r8_enabled() -> bool {
    match std::env::var("NV_G4MOE_BF16_R8") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_BF16_R8_DEFAULT_ON,
    }
}

pub const G4MOE_BF16_ACC4_MEASURED_NO_WALL_CHANGE_DEFAULT_OFF: bool = false;

pub fn bf16_acc4_enabled() -> bool {
    match std::env::var("NV_G4MOE_BF16_ACC4") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_BF16_ACC4_MEASURED_NO_WALL_CHANGE_DEFAULT_OFF,
    }
}

pub const G4MOE_MLPGU_FUSEDX_MEASURED_NEGATIVE_BF16_TWO_MAT_STREAMS_DEFAULT_OFF: bool = false;

pub fn mlpgu_fusedx_enabled() -> bool {
    match std::env::var("NV_G4MOE_MLPGU_FUSEDX") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_MLPGU_FUSEDX_MEASURED_NEGATIVE_BF16_TWO_MAT_STREAMS_DEFAULT_OFF,
    }
}

pub const G4MOE_GU_FUSEDX_DEFAULT_ON: bool = false;

pub fn gu_fusedx_enabled() -> bool {
    match std::env::var("NV_G4MOE_GU_FUSEDX") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_GU_FUSEDX_DEFAULT_ON,
    }
}

pub const G4MOE_DOWN_EXPAR_DEFAULT_ON: bool = false;

pub fn down_expar_enabled() -> bool {
    match std::env::var("NV_G4MOE_DOWN_EXPAR") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_DOWN_EXPAR_DEFAULT_ON,
    }
}

pub const G4MOE_NORM_LAT_DEFAULT_ON: bool = false;

pub fn norm_lat_enabled() -> bool {
    match std::env::var("NV_G4MOE_NORM_LAT") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_NORM_LAT_DEFAULT_ON,
    }
}

fn norm_lat_active(ctx: &WgpuContext) -> bool {
    norm_lat_enabled() && ctx.caps.subgroup && ctx.subgroup_width() == Some(32)
}

pub const G4MOE_W4_XV4_STACKS_ON_GEMV_SGLAT_DEFAULT_ON: bool = false;

pub fn w4_xv4_enabled() -> bool {
    match std::env::var("NV_G4MOE_W4_XV4") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_W4_XV4_STACKS_ON_GEMV_SGLAT_DEFAULT_ON,
    }
}

pub const G4MOE_ATTN_I8_DEFAULT_ON: bool = false;

pub const ATTN_I8_GROUP_32_IS_THE_GOW_RECIPE_G16_HALVES_THE_QUANT_STEP_AT_DOUBLE_THE_SCALE_STREAM:
    usize = I8_GS;

pub const ATTN_I8_G16_TOKEN: &str = "g16";

pub fn attn_i8_group_elems() -> usize {
    match std::env::var("NV_G4MOE_ATTN_I8") {
        Ok(v) if v.split(',').any(|c| c == ATTN_I8_G16_TOKEN) => 16,
        _ => ATTN_I8_GROUP_32_IS_THE_GOW_RECIPE_G16_HALVES_THE_QUANT_STEP_AT_DOUBLE_THE_SCALE_STREAM,
    }
}

pub fn attn_i8_site_enabled(label: &str) -> bool {
    match std::env::var("NV_G4MOE_ATTN_I8") {
        Err(_) => G4MOE_ATTN_I8_DEFAULT_ON,
        Ok(v) if v.is_empty() || v == "0" => false,
        Ok(v) => {
            let mut sites = v
                .split(',')
                .filter(|c| !c.is_empty() && *c != ATTN_I8_G16_TOKEN)
                .peekable();
            match sites.peek() {
                None => true,
                Some(_) => sites.any(|c| c == "1" || label.contains(c)),
            }
        }
    }
}

pub fn attn_i8_enabled() -> bool {
    attn_i8_site_enabled("g4m-at-qkvproj") || attn_i8_site_enabled("g4m-at-oproj")
}

pub fn attn_i8_caps_verify_rows_to_zero_because_int8_decode_vs_bf16_prefill_breaks_the_argmax_contract(
) -> bool {
    attn_i8_enabled()
}

pub const G4MOE_GEMV_SGLAT_UNSET_ROUTES_AT_OPROJ_ONLY_ROW_R3_111_1_TOK_S_BROADER_LABEL_SETS_DILUTE:
    &str = "at-oproj";

pub fn gemv_sglat_enabled(label: &str) -> bool {
    match std::env::var("NV_G4MOE_GEMV_SGLAT") {
        Err(_) => {
            G4MOE_FUSION_CHAMPION_ROWS_G4MOE_FUSION_R1_R2_R2C_R3_111_1_VS_87_3_TOK_S_AT_256_TF_PPL_31_9_ACC_0_5254_BOTH_ARMS_DEFAULT_ON
                && label.contains(
                    G4MOE_GEMV_SGLAT_UNSET_ROUTES_AT_OPROJ_ONLY_ROW_R3_111_1_TOK_S_BROADER_LABEL_SETS_DILUTE,
                )
        }
        Ok(v) => {
            let t = v.trim();
            if t.is_empty() || t == "0" {
                false
            } else if t == "1" {
                true
            } else {
                t.split(',').any(|c| !c.is_empty() && label.contains(c))
            }
        }
    }
}

fn gemv_sglat_active(ctx: &WgpuContext, label: &str) -> bool {
    gemv_sglat_enabled(label) && ctx.caps.subgroup && ctx.subgroup_width() == Some(32)
}

pub const G4MOE_GEMV_V2_OPT_IN_ONLY_ROW_PER_SUBGROUP_REASSOCIATES_THE_DOT_SO_ADOPTION_NEEDS_A_TF_PPL_ROW:
    bool = false;

pub fn gemv_v2_enabled(label: &str) -> bool {
    match std::env::var("NV_G4MOE_GEMV_V2") {
        Err(_) => {
            G4MOE_GEMV_V2_OPT_IN_ONLY_ROW_PER_SUBGROUP_REASSOCIATES_THE_DOT_SO_ADOPTION_NEEDS_A_TF_PPL_ROW
        }
        Ok(v) => {
            let t = v.trim();
            if t.is_empty() || t == "0" {
                false
            } else if t == "1" {
                true
            } else {
                t.split(',').any(|c| !c.is_empty() && label.contains(c))
            }
        }
    }
}

fn gemv_v2_active_implies_the_vec4_wide_layout(
    ctx: &WgpuContext,
    label: &str,
    load: GemvLoad,
    k: usize,
    x_off_words: usize,
) -> bool {
    gemv_v2_enabled(label)
        && ctx.caps.subgroup
        && ctx.subgroup_width() == Some(32)
        && load == GemvLoad::Wide
        && k.is_multiple_of(8)
        && x_off_words.is_multiple_of(4)
}

const GEMV_BF16_V2_ENTRY: &str = "g4m_gemv_bf16_v2_row_per_sg32_dual_vec4_zero_barrier_reduce";

pub const G4MOE_PREENC_DEFAULT_ON: bool = false;

pub fn preenc_enabled() -> bool {
    match std::env::var("NV_G4MOE_PREENC") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_PREENC_DEFAULT_ON,
    }
}

pub const G4MOE_W4_COAL_DEFAULT_ON: bool = false;

pub const W4_COAL_XBUF_WORDS_3072_BOUNDS_THE_PRELOADED_ACTIVATION_SLOTS: usize = 3072;

pub fn w4_coal_enabled() -> bool {
    match std::env::var("NV_G4MOE_W4_COAL") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_W4_COAL_DEFAULT_ON,
    }
}

pub const G4MOE_FOLD_CHAIN_DEFAULT_ON: bool = false;

pub fn fold_chain_enabled() -> bool {
    match std::env::var("NV_G4MOE_FOLD_CHAIN") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_FOLD_CHAIN_DEFAULT_ON,
    }
}

pub const G4MOE_TAILFOLD_MEASURED_NEGATIVE_PROLOGUE_REDUNDANCY_SCALES_WITH_3072_QKVPROJ_WGS_DEFAULT_OFF:
    bool = false;

pub fn tailfold_enabled() -> bool {
    match std::env::var("NV_G4MOE_TAILFOLD") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_TAILFOLD_MEASURED_NEGATIVE_PROLOGUE_REDUNDANCY_SCALES_WITH_3072_QKVPROJ_WGS_DEFAULT_OFF,
    }
}

pub const G4MOE_QKVFOLD_MEASURED_NEGATIVE_SERIAL_TREE_PROLOGUE_DOUBLES_STAGE1_DEFAULT_OFF: bool =
    false;

pub fn qkvfold_enabled() -> bool {
    match std::env::var("NV_G4MOE_QKVFOLD") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_QKVFOLD_MEASURED_NEGATIVE_SERIAL_TREE_PROLOGUE_DOUBLES_STAGE1_DEFAULT_OFF,
    }
}

pub const G4MOE_ROUTER_CHAIN_DEFAULT_ON: bool = false;

pub fn router_chain_enabled() -> bool {
    match std::env::var("NV_G4MOE_ROUTER_CHAIN") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_ROUTER_CHAIN_DEFAULT_ON,
    }
}

pub const G4MOE_KV_FP8_DEFAULT_ON: bool = false;

pub fn kv_fp8_enabled() -> bool {
    match std::env::var("NV_G4MOE_KV_FP8") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_KV_FP8_DEFAULT_ON,
    }
}

pub const G4MOE_KIND_LABELS_DEFAULT_ON: bool = false;

pub fn layer_kind_decode_labels_enabled() -> bool {
    match std::env::var("NV_G4MOE_KIND_LABELS") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_KIND_LABELS_DEFAULT_ON,
    }
}

struct PfScratch {
    m: usize,
    tok: wgpu::Buffer,
    pos: wgpu::Buffer,
    splice_rows: wgpu::Buffer,
    splice_mask: wgpu::Buffer,
    res_a: wgpu::Buffer,
    res_b: wgpu::Buffer,
    normed: wgpu::Buffer,
    attn_out: wgpu::Buffer,
    attn_post: wgpu::Buffer,
    normed_mlp: wgpu::Buffer,
    dense_out: wgpu::Buffer,
    h1: wgpu::Buffer,
    moe_in: wgpu::Buffer,
    moe_out: wgpu::Buffer,
    h2: wgpu::Buffer,
    ffw_sum: wgpu::Buffer,
    combined: wgpu::Buffer,
    q_raw: wgpu::Buffer,
    k_raw: wgpu::Buffer,
    v_raw: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    v: wgpu::Buffer,
    scores: wgpu::Buffer,
    attn_f32: wgpu::Buffer,
    attn_bf16: wgpu::Buffer,
    mlp_g: wgpu::Buffer,
    mlp_u: wgpu::Buffer,
    mlp_act: wgpu::Buffer,
    rnormed: wgpu::Buffer,
    router_in: wgpu::Buffer,
    rlogits: wgpu::Buffer,
    ids: wgpu::Buffer,
    wts: wgpu::Buffer,
    y_gate: wgpu::Buffer,
    y_up: wgpu::Buffer,
    moe_act: wgpu::Buffer,
    y_down: wgpu::Buffer,
}

fn alloc_pf_scratch(b: &mut Builder, cfg: &Gemma4MoeConfig, max_seq: usize, m: usize) -> PfScratch {
    let base = &cfg.base;
    let hidden = base.hidden_size;
    let hidden_words = hidden / 2;
    let n_q = base.num_attention_heads;
    let hd_max = base
        .head_dim_for(LayerType::SlidingAttention)
        .max(base.head_dim_for(LayerType::FullAttention));
    let n_kv_max = base
        .num_kv_heads_for(LayerType::SlidingAttention)
        .max(base.num_kv_heads_for(LayerType::FullAttention));
    let inter = base.intermediate_size;
    let moe_inter = cfg.moe_intermediate_size;
    let k_top = cfg.top_k_experts;
    let row = (m * hidden_words * 4) as u64;
    PfScratch {
        m,
        tok: b.upload_i32("g4m-pf-tok", &vec![0i32; m]),
        pos: b.upload_i32("g4m-pf-pos", &vec![0i32; m]),
        splice_rows: b.zeros("g4m-pf-splice-rows", row),
        splice_mask: b.upload_u32("g4m-pf-splice-mask", &vec![0u32; m.max(4)]),
        res_a: b.zeros("g4m-pf-res-a", row),
        res_b: b.zeros("g4m-pf-res-b", row),
        normed: b.zeros("g4m-pf-normed", row),
        attn_out: b.zeros("g4m-pf-attn-out", row),
        attn_post: b.zeros("g4m-pf-attn-post", row),
        normed_mlp: b.zeros("g4m-pf-normed-mlp", row),
        dense_out: b.zeros("g4m-pf-dense-out", row),
        h1: b.zeros("g4m-pf-h1", row),
        moe_in: b.zeros("g4m-pf-moe-in", row),
        moe_out: b.zeros("g4m-pf-moe-out", row),
        h2: b.zeros("g4m-pf-h2", row),
        ffw_sum: b.zeros("g4m-pf-ffw-sum", row),
        combined: b.zeros("g4m-pf-combined", row),
        q_raw: b.zeros("g4m-pf-qraw", (m * n_q * hd_max * 2) as u64),
        k_raw: b.zeros("g4m-pf-kraw", (m * n_kv_max * hd_max * 2) as u64),
        v_raw: b.zeros("g4m-pf-vraw", (m * n_kv_max * hd_max * 2) as u64),
        q: b.zeros("g4m-pf-q", (m * n_q * hd_max * 2) as u64),
        k: b.zeros("g4m-pf-k", (m * n_kv_max * hd_max * 2) as u64),
        v: b.zeros("g4m-pf-v", (m * n_kv_max * hd_max * 2) as u64),
        scores: b.zeros("g4m-pf-scores", (m * n_q * max_seq * 4) as u64),
        attn_f32: b.zeros("g4m-pf-of32", (m * n_q * hd_max * 4) as u64),
        attn_bf16: b.zeros("g4m-pf-obf16", (m * n_q * hd_max * 2) as u64),
        mlp_g: b.zeros("g4m-pf-mlp-g", (m * inter * 2) as u64),
        mlp_u: b.zeros("g4m-pf-mlp-u", (m * inter * 2) as u64),
        mlp_act: b.zeros("g4m-pf-mlp-act", (m * inter * 2) as u64),
        rnormed: b.zeros("g4m-pf-rnorm", row),
        router_in: b.zeros("g4m-pf-rin", row),
        rlogits: b.zeros("g4m-pf-rlogits", (m * cfg.num_experts * 4) as u64),
        ids: b.zeros("g4m-pf-ids", (m * k_top * 4) as u64),
        wts: b.zeros("g4m-pf-wts", (m * k_top * 4) as u64),
        y_gate: b.zeros("g4m-pf-ygate", (m * k_top * moe_inter * 2) as u64),
        y_up: b.zeros("g4m-pf-yup", (m * k_top * moe_inter * 2) as u64),
        moe_act: b.zeros("g4m-pf-moe-act", (m * k_top * moe_inter * 2) as u64),
        y_down: b.zeros("g4m-pf-ydown", (m * k_top * hidden * 2) as u64),
    }
}

#[allow(clippy::too_many_arguments)]
fn pf_gemm_bf16(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    x_stride_words: usize,
    y: &wgpu::Buffer,
    y_stride_words: usize,
    out_f32: bool,
    alpha: f32,
) -> Result<()> {
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "g4m-pf-gemm-p",
        PfGemmParams {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: (w.k / 2) as u32,
            x_stride_words: x_stride_words as u32,
            y_stride_words: y_stride_words as u32,
            y_off_words: 0,
            alpha,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.prefill,
        "pm_gemm_bf16",
        &[(20, &w.w), (21, x), (22, &p), (23, y)],
        grid,
    )
}

struct LmHeadChunk<'a> {
    w: &'a wgpu::Buffer,
    row_off: usize,
    n_rows: usize,
    i8: Option<&'a (wgpu::Buffer, wgpu::Buffer)>,
    wide: u32,
}

#[allow(clippy::too_many_arguments)]
fn push_lmhead_row(
    b: &mut Builder,
    s: &Sources,
    gemv_i8_src: &str,
    lmhead_i8: Option<&'static str>,
    c: &LmHeadChunk,
    x: &wgpu::Buffer,
    x_off_words: usize,
    y: &wgpu::Buffer,
    y_off_words: usize,
    hidden: usize,
) -> Result<()> {
    let pairs = c.n_rows.div_ceil(2);
    let grid = if head_one_shard_enabled() || lmhead_tight_grid_enabled() {
        square_up_the_2d_tiling_a_65535_wide_row_leaves_65533_workgroups_idle_at_one_shard(
            b.ctx, pairs as u64,
        )
    } else {
        b.grid1(pairs as u64, 1)
    };
    let alpha = 1.0;
    if let (Some(entry), Some((qb, sb))) = (lmhead_i8, c.i8) {
        assert!(
            hidden.is_multiple_of(16),
            "hidden {hidden} cannot feed the vec4 int8 head load"
        );
        assert_eq!(
            x_off_words, 0,
            "the int8 lm_head is wired for the single-row decode activation only; \
             the verify epilogue stays off while it is selected"
        );
        let kw = if entry == "g4m_gemv_i8_v4" {
            hidden / 16
        } else {
            hidden / 4
        };
        let p = b.uni(
            "g4m-lmhead-i8-p",
            GemvI8Params {
                n_rows: c.n_rows as u32,
                k_words: kw as u32,
                groups_x: grid.0,
                out_f32: 0,
                w_row_words: kw as u32,
                x_off_words: 0,
                y_off_words: y_off_words as u32,
                s_row_elems: (hidden / I8_GS) as u32,
                alpha,
                ..Default::default()
            },
        );
        return b.push(
            "g4m-lmhead",
            gemv_i8_src,
            entry,
            &[(0, qb), (1, x), (2, &p), (3, y), (4, sb)],
            grid,
        );
    }
    let v2 = gemv_v2_active_implies_the_vec4_wide_layout(
        b.ctx,
        "g4m-lmhead",
        GemvLoad::Wide,
        hidden,
        x_off_words,
    );
    let (entry, src, grid) = if v2 {
        (
            GEMV_BF16_V2_ENTRY,
            &s.gemv_bf16_v2,
            b.grid1(c.n_rows.div_ceil(8) as u64, 1),
        )
    } else {
        ("g4m_gemv_bf16", &s.gemv_bf16, grid)
    };
    let p = b.uni(
        "g4m-lmhead-p",
        GemvBf16Params {
            n_rows: c.n_rows as u32,
            k_words: (hidden / 2) as u32,
            groups_x: grid.0,
            out_f32: 0,
            w_row_words: (hidden / 2) as u32,
            x_off_words: x_off_words as u32,
            y_off_words: y_off_words as u32,
            wide: if v2 { 1 } else { c.wide },
            alpha,
            ..Default::default()
        },
    );
    b.push(
        "g4m-lmhead",
        src,
        entry,
        &[(0, c.w), (1, x), (2, &p), (3, y)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn pf_norm(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
    m: usize,
) -> Result<()> {
    let p = b.uni(
        "g4m-pf-norm-p",
        PfNormParams {
            hidden: hidden as u32,
            words: (hidden / 2) as u32,
            eps,
            m: m as u32,
        },
    );
    b.push(
        label,
        &s.prefill,
        "pm_norm",
        &[(10, x), (11, w), (12, y), (13, &p)],
        (m as u32, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn pf_norm_residual(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    res: &wgpu::Buffer,
    w: &wgpu::Buffer,
    y: &wgpu::Buffer,
    hidden: usize,
    eps: f32,
    m: usize,
) -> Result<()> {
    let p = b.uni(
        "g4m-pf-normres-p",
        PfNormParams {
            hidden: hidden as u32,
            words: (hidden / 2) as u32,
            eps,
            m: m as u32,
        },
    );
    b.push(
        label,
        &s.prefill,
        "pm_norm_residual",
        &[(10, x), (11, w), (12, y), (13, &p), (14, res)],
        (m as u32, 1, 1),
    )
}

pub struct LoadReport {
    pub build_s: f64,
    pub quantize_s: f64,
    pub wired_bytes: u64,
}

struct PfState {
    m: usize,
    passes: Vec<Pass>,
    tok: wgpu::Buffer,
    pos: wgpu::Buffer,
    splice_rows: wgpu::Buffer,
    splice_mask: wgpu::Buffer,
    splice_mask_live: bool,
    validated: bool,
}

pub use crate::gemma4_wgpu::VERIFY_ROWS_MAX_IS_THE_LONGEST_CHAIN_THE_SPEC_LOOP_SUBMITS;

struct VerifyState {
    rows: usize,
    passes: Vec<Pass>,
    logits: wgpu::Buffer,
    tokens: wgpu::Buffer,
    validated: bool,
}

pub use crate::gemma4_wgpu::EmbedRowSplice;

struct ChunkRowSplice<'a> {
    rel_pos: usize,
    row_words: &'a [u32],
}

pub struct Gemma4MoeWgpu {
    ctx: &'static WgpuContext,
    config: Gemma4MoeConfig,
    max_seq: usize,
    pos: usize,
    validated: bool,
    prefix_validated: bool,
    passes: Vec<Pass>,
    prefill: Option<PfState>,
    verify: Option<VerifyState>,
    head_start: usize,
    preenc: bool,
    pending_cb: Option<(PendingCbBandKey, wgpu::CommandBuffer)>,
    preenc_replays: usize,
    preenc_band_discards: usize,
    _buffers: Vec<wgpu::Buffer>,
    tok_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    flash_fd: Option<(wgpu::Buffer, FlashDecodeParams)>,
    flash_sl: Option<(wgpu::Buffer, FlashDecodeParams, u32)>,
    depthgate_flash1_passes: Vec<usize>,
    depthgate_flash1_sliding_passes: Vec<usize>,
    depthgate_on: bool,
    depthgate_shallow_splits: u32,
    depthgate_shallow_splits_sliding: u32,
    foldgate_alt_flash1: Vec<Pass>,
    foldgate_deep_active: bool,
    flash_scratch_bytes_full: Vec<u64>,
    flash_scratch_bytes_sliding: Vec<u64>,
    token_out: wgpu::Buffer,
    logits: wgpu::Buffer,
    state_buffers: Vec<(wgpu::Buffer, u64)>,
    vocab: usize,
    vram: VramReport,
    load: LoadReport,
    weight_bytes: u64,
    dense_gemv_wide: (usize, usize),
    assistant_taps: AssistantTaps,
}

type PendingCbBandKey = (usize, u32, u32, bool);

#[derive(Clone, Copy, Debug)]
pub struct AssistantKvTapGeometry {
    pub n_kv: usize,
    pub head_dim: usize,
    pub cache_rows: usize,
    pub ring: bool,
}

struct AssistantKvTap {
    kc: wgpu::Buffer,
    vc: wgpu::Buffer,
    fp8: bool,
    geom: AssistantKvTapGeometry,
}

struct AssistantTaps {
    sliding_layer: Option<usize>,
    full_layer: Option<usize>,
    sliding: Option<AssistantKvTap>,
    full: Option<AssistantKvTap>,
    decode_prenorm_hidden: wgpu::Buffer,
    verify_prenorm_hidden: Option<wgpu::Buffer>,
}

fn unpack_bf16_pairs_to_f32(words: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(words.len() * 2);
    for &w in words {
        out.push(f32::from_bits((w & 0xffff) << 16));
        out.push(f32::from_bits(w & 0xffff_0000));
    }
    out
}

pub const ASSISTANT_RING_PAD_CLOBBER_SLOTS_ARE_THE_PREFILL_M_ROWS_VERIFY_PADS_PAST_THE_COMMITTED_POS: &str =
    "verify_chain runs the full m-row prefill graph, so its pad rows write ring slots belonging \
     to positions cache_rows-m back; a reader must stay at least prefill_m positions clear of \
     the ring tail";

pub fn synthetic_state_bytes_0x30_to_0x3e_finite_small_under_fp8_bf16_and_f32_views_so_no_nan_and_moe_routing_stays_nondegenerate(
    len: usize,
) -> Vec<u8> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0x30u8 + ((state >> 41) % 15) as u8
        })
        .collect()
}

pub fn vram_report_enabled() -> bool {
    crate::wgpu_ledger::vram_report_var_enabled("NV_G4MOE_WGPU_VRAM")
}

impl Gemma4MoeWgpu {
    pub fn config(&self) -> &Gemma4MoeConfig {
        &self.config
    }

    pub fn vram_report(&self) -> &VramReport {
        &self.vram
    }

    pub fn weight_bytes_per_token(&self) -> u64 {
        self.weight_bytes
    }

    pub fn load_report(&self) -> &LoadReport {
        &self.load
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn dense_gemv_wide_counts(&self) -> (usize, usize) {
        self.dense_gemv_wide
    }

    pub fn decode_pass_storage_binding_sets_a_shared_id_between_two_passes_is_a_wgpu_hazard(
        &self,
    ) -> Vec<Vec<(u32, bool)>> {
        self.passes
            .iter()
            .map(|p| {
                p.storage_binds_are_what_wgpu_tracks_for_hazards_uniforms_never_transition
                    .clone()
            })
            .collect()
    }

    pub fn pass_rows(&self) -> Vec<(String, String, (u32, u32, u32), u64)> {
        self.passes
            .iter()
            .map(|p| {
                let (label, entry) = p.label.split_once(':').unwrap_or((&p.label, ""));
                (label.to_string(), entry.to_string(), p.grid, p.bind_bytes)
            })
            .collect()
    }

    pub fn decode_step_replicated(
        &mut self,
        token: u32,
        class: Option<&str>,
        k: usize,
    ) -> Result<u32> {
        anyhow::ensure!((token as usize) < self.vocab, "token {token} out of vocab");
        anyhow::ensure!(self.pos < self.max_seq, "kv cache full at {}", self.pos);
        self.ctx
            .queue
            .write_buffer(&self.tok_buf, 0, bytemuck::bytes_of(&(token as i32)));
        self.ctx
            .queue
            .write_buffer(&self.pos_buf, 0, bytemuck::bytes_of(&(self.pos as i32)));
        self.write_flash_total_uniform_mirroring_gemma4_wgpu_write_pos_uniforms();

        let extra: Vec<&Pass> = match class {
            Some(c) => self
                .passes
                .iter()
                .filter(|p| p.label.split_once(':').map(|(l, _)| l) == Some(c))
                .collect(),
            None => Vec::new(),
        };
        anyhow::ensure!(
            class.is_none() || !extra.is_empty(),
            "no decode pass is labelled {class:?}"
        );

        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in &self.passes {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
            for _ in 0..k {
                for p in &extra {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        self.pos += 1;
        let t: Vec<u32> = dispatch::read_back(self.ctx, &self.token_out, 1)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(t[0])
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn read_logits(&self) -> Result<Vec<f32>> {
        dispatch::read_back(self.ctx, &self.logits, self.vocab).map_err(|e| anyhow::anyhow!("{e}"))
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        self.pending_cb = None;
        for (buf, bytes) in &self.state_buffers {
            let zeros = vec![0u8; *bytes as usize];
            self.ctx.queue.write_buffer(buf, 0, &zeros);
        }
        Ok(())
    }

    pub fn fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(
        &mut self,
        pos: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            pos <= self.max_seq,
            "synthetic fill pos {pos} past max_seq {}; ring wrap and window start are pure functions of pos, so pos alone is the cache-depth state",
            self.max_seq
        );
        let max_bytes = self
            .state_buffers
            .iter()
            .map(|(_, b)| *b as usize)
            .max()
            .unwrap_or(0);
        let pattern = synthetic_state_bytes_0x30_to_0x3e_finite_small_under_fp8_bf16_and_f32_views_so_no_nan_and_moe_routing_stays_nondegenerate(max_bytes);
        for (buf, bytes) in &self.state_buffers {
            self.ctx
                .queue
                .write_buffer(buf, 0, &pattern[..*bytes as usize]);
        }
        self.pos = pos;
        Ok(())
    }

    pub fn from_gguf(path: &std::path::Path, max_seq: usize) -> Result<Self> {
        let loader = nv_weights::GgufLoader::open(path, &candle_core::Device::Cpu)?;
        let config = crate::gemma4_gguf::gemma4_moe_config_from_gguf(&loader)?;
        Self::from_loader(config, &loader, max_seq)
    }

    pub fn from_loader<S: nv_weights::TensorSource + CompressedTensorsRawBytes>(
        config: Gemma4MoeConfig,
        weights: &S,
        max_seq: usize,
    ) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
        let mut pf_m = prefill_m();
        if pf_m > max_seq {
            pf_m = 0;
        }
        let flash_sg = flash_sg_reduce_enabled();
        anyhow::ensure!(
            !flash_sg || (ctx.caps.subgroup && ctx.subgroup_width() == Some(32)),
            "NV_G4MOE_FLASH_SG=1 replaces the flash stage1 LDS butterfly with a \
             subgroupShuffleXor chain over lanes 16/8/4/2/1, which is bit-identical only at \
             subgroup width 32; this adapter reports subgroup={} width={:?}",
            ctx.caps.subgroup,
            ctx.subgroup_width()
        );
        let sacc_chunk = flash_sacc_chunk_head_dim_slice_staged_per_warp_read_once_at_build();
        let flash_fold = flash_gqa_fold_query_heads_per_workgroup_read_once_at_build();
        anyhow::ensure!(
            flash_fold == 1 || (ctx.caps.subgroup && ctx.subgroup_width() == Some(32)),
            "NV_G4MOE_FLASH_FOLD={flash_fold} calls the stage1 row reduction once per folded \
             query head, which only pays with the subgroup chain; this adapter reports \
             subgroup={} width={:?}. {}",
            ctx.caps.subgroup,
            ctx.subgroup_width(),
            FLASH_GQA_FOLD_DIVIDES_THE_GRID_SO_IT_ONLY_PAYS_PAIRED_WITH_SPLITS_TIMES_FOLD_AND_A_SACC_SLICE
        );
        let foldgate =
            flash_foldgate_enabled_full_attention_swaps_to_the_fold2_arm_at_depth_while_sliding_keeps_its_own_band();
        anyhow::ensure!(
            !foldgate || (ctx.caps.subgroup && ctx.subgroup_width() == Some(32)),
            "NV_G4MOE_FLASH_FOLDGATE=1 swaps in a fold-2 stage1 whose row reduction runs once per \
             folded query head, which only pays with the subgroup chain; this adapter reports \
             subgroup={} width={:?}",
            ctx.caps.subgroup,
            ctx.subgroup_width()
        );
        anyhow::ensure!(
            !foldgate || !qkvfold_enabled(),
            "NV_G4MOE_FLASH_FOLDGATE=1 swaps the plain bf16 flash1 pass; the qkvfold entry fuses \
             the qkv norm into stage1 and has no fold-2 twin -- unset NV_G4MOE_QKVFOLD"
        );
        anyhow::ensure!(
            !foldgate || !kv_fp8_enabled(),
            "NV_G4MOE_FLASH_FOLDGATE=1 has no fp8 fold-2 stage1 entry -- unset NV_G4MOE_KV_FP8"
        );
        anyhow::ensure!(
            !foldgate
                || ctx.caps.workgroup_storage_fits(
                    flash_workgroup_bytes_qsh512_sm8_sl8_plus_red256_only_when_wb_plus_sacc_8_warps_by_chunk(
                        flash_sg,
                        FOLDGATE_DEEP_SACC_128_KEEPS_THE_FOLDED_WORKGROUP_UNDER_A_THIRD_OF_48KB_SO_THREE_BLOCKS_PLACE_PER_SM_AND_IS_DIGIT_IDENTICAL_TO_SACC_512_ON_REAL_WEIGHTS,
                        FOLDGATE_DEEP_FOLD_2_QUERY_HEADS_PER_WORKGROUP_HALVES_FULL_ATTENTION_KV_TRAFFIC,
                    )
                ),
            "NV_G4MOE_FLASH_FOLDGATE=1: the fold-2 sacc-128 deep workgroup does not fit this \
             device's workgroup storage of {} bytes",
            ctx.caps.max_compute_workgroup_storage_size
        );
        let full_splits = flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build();
        eprintln!(
            "[g4m-wgpu] flash stage1: reduce={} sacc slice {sacc_chunk} gqa fold full {flash_fold} \
             sliding {} splits full {full_splits} sliding {} -> {} workgroup bytes, foldgate={}",
            if flash_sg { "sg" } else { "wb" },
            flash_gqa_fold_for_layer_kind_only_full_attention_scans_enough_rows_to_amortise_a_halved_grid(
                LayerType::SlidingAttention,
                flash_fold,
            ),
            flash_splits_for_layer_kind_full_attention_holds_the_folded_grid_sliding_holds_the_stage2_scan(
                LayerType::SlidingAttention,
                full_splits,
            ),
            flash_workgroup_bytes_qsh512_sm8_sl8_plus_red256_only_when_wb_plus_sacc_8_warps_by_chunk(
                flash_sg, sacc_chunk, flash_fold
            ),
            if foldgate { "armed" } else { "off" }
        );
        let s = Sources::new(pf_m, flash_sg, sacc_chunk, flash_fold, foldgate);
        let base = &config.base;
        let sliding_ring_rows: Option<usize> = if sliding_kv_ring_enabled() {
            let rows = sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom(
                base.sliding_window,
                pf_m,
            );
            if max_seq > rows {
                eprintln!(
                    "[g4m-wgpu] sliding kv ring: {rows} rows/layer (window {} + chunk {} + headroom) instead of {max_seq}",
                    base.sliding_window,
                    pf_m.max(1)
                );
                Some(rows)
            } else {
                eprintln!(
                    "[g4m-wgpu] sliding kv ring off: max_seq {max_seq} <= ring {rows}, full-depth rows are already smaller"
                );
                None
            }
        } else {
            None
        };
        let t_build = std::time::Instant::now();
        let mut quantize_s = 0f64;

        anyhow::ensure!(max_seq > 0, "max_seq must be positive");
        anyhow::ensure!(
            base.hidden_size.is_multiple_of(2 * W4_GS),
            "hidden_size {} must be a multiple of {}",
            base.hidden_size,
            2 * W4_GS
        );
        anyhow::ensure!(
            config.moe_intermediate_size.is_multiple_of(2 * W4_GS),
            "moe_intermediate_size {} must be a multiple of {}",
            config.moe_intermediate_size,
            2 * W4_GS
        );
        anyhow::ensure!(
            config.num_experts <= 256,
            "router top-k kernel caps num_experts at 256, got {}",
            config.num_experts
        );
        anyhow::ensure!(
            config.top_k_experts <= MAX_TOPK,
            "top_k_experts {} exceeds {MAX_TOPK}",
            config.top_k_experts
        );
        for kind in [LayerType::SlidingAttention, LayerType::FullAttention] {
            let hd = base.head_dim_for(kind);
            anyhow::ensure!(
                hd.is_multiple_of(2) && hd <= MAX_HEAD_DIM,
                "head_dim {hd} must be even and <= {MAX_HEAD_DIM}"
            );
            let n_kv = base.num_kv_heads_for(kind);
            anyhow::ensure!(
                base.num_attention_heads.is_multiple_of(n_kv),
                "attention heads {} must be a multiple of kv heads {n_kv}",
                base.num_attention_heads
            );
        }
        anyhow::ensure!(
            base.hidden_size.is_multiple_of(2) && base.intermediate_size.is_multiple_of(2),
            "hidden/intermediate must be even"
        );
        anyhow::ensure!(
            base.tie_word_embeddings,
            "gemma4-moe wgpu expects tied embeddings"
        );

        let hidden = base.hidden_size;
        let hidden_words = hidden / 2;
        let eps = base.rms_norm_eps as f32;
        let vocab = base.vocab_size;

        let mut b = Builder {
            ctx,
            passes: Vec::new(),
            prefill_passes: Vec::new(),
            to_prefill: false,
            buffers: Vec::new(),
            alloc: std::collections::BTreeMap::new(),
            alloc_total: 0,
            since_flush: 0,
            expert_share: config.top_k_experts as f64 / config.num_experts as f64,
            weight_bpt: 0.0,
            wide_gemvs: 0,
            narrow_gemvs: 0,
            depthgate_flash1_pass_indices_full_attention_the_band_is_keyed_on_the_whole_context:
                Vec::new(),
            depthgate_flash1_pass_indices_sliding_the_window_caps_the_scan_so_the_band_is_keyed_on_min_total_window:
                Vec::new(),
            foldgate_alt_flash1_passes_one_deep_twin_per_full_attention_flash1_in_the_same_order:
                Vec::new(),
            storage_buffer_registry_indexes_every_hazard_tracked_buffer_the_graph_binds: Vec::new(),
            flash_scratch_bytes_per_full_attention_layer_sized_at_its_own_split_count: Vec::new(),
            flash_scratch_bytes_per_sliding_layer_sized_at_its_own_split_count_not_the_max_of_the_two:
                Vec::new(),
        };

        let tok_buf = b.upload_u32("g4m-tok", &[0u32]);
        let pos_buf = b.upload_u32("g4m-pos", &[0u32]);
        let flash_fd: Option<(wgpu::Buffer, FlashDecodeParams)> =
            flash_decode_enabled().then(|| {
                let p = FlashDecodeParams {
                    n_heads: base.num_attention_heads as u32,
                    n_kv: base.num_kv_heads_for(LayerType::FullAttention) as u32,
                    head_dim: base.head_dim_for(LayerType::FullAttention) as u32,
                    splits: flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
                    total: 0,
                    start: 0,
                    ring: 0,
                    scale: 1.0,
                };
                (b.uni("g4m-at-fl-p", p), p)
            });
        let flash_sl: Option<(wgpu::Buffer, FlashDecodeParams, u32)> = (flash_sliding_enabled()
            && (0..base.num_hidden_layers)
                .any(|i| base.layer_kind(i) == LayerType::SlidingAttention))
        .then(|| {
            let ring = sliding_ring_rows.map(|r| r as u32).unwrap_or(0);
            let p = FlashDecodeParams {
                n_heads: base.num_attention_heads as u32,
                n_kv: base.num_kv_heads_for(LayerType::SlidingAttention) as u32,
                head_dim: base.head_dim_for(LayerType::SlidingAttention) as u32,
                splits: flash_splits_for_layer_kind_full_attention_holds_the_folded_grid_sliding_holds_the_stage2_scan(
                    LayerType::SlidingAttention,
                    flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
                ),
                total: 0,
                start: 0,
                ring,
                scale: 1.0,
            };
            (b.uni("g4m-at-flsl-p", p), p, base.sliding_window as u32)
        });
        let pf = (pf_m > 0).then(|| alloc_pf_scratch(&mut b, &config, max_seq, pf_m));

        let res_a = b.zeros("g4m-res-a", (hidden_words * 4) as u64);
        let res_b = b.zeros("g4m-res-b", (hidden_words * 4) as u64);
        let normed = b.zeros("g4m-normed", (hidden_words * 4) as u64);

        let embed_name = "model.language_model.embed_tokens.weight";
        let embed = load_bf16(weights, embed_name, &[vocab, hidden])?;
        let embed_scale = (hidden as f32).sqrt();
        let chunk_rows = row_chunk(ctx, hidden);
        let lmhead_i8 = lmhead_i8_entry_off_until_real_checkpoint_quality_gate();
        let embed_label = if lmhead_i8.is_some() {
            "g4m-embed-rowgather"
        } else {
            "g4m-embed"
        };
        type I8Chunk = Option<(wgpu::Buffer, wgpu::Buffer)>;
        let mut embed_chunks: Vec<(wgpu::Buffer, usize, usize, I8Chunk)> = Vec::new();
        let mut off = 0usize;
        while off < vocab {
            let rows = (chunk_rows.min(vocab - off)) & !1usize;
            let rows = if rows == 0 { vocab - off } else { rows };
            let buf = b.upload_u32(
                embed_label,
                &pack_pairs(&embed[off * hidden..(off + rows) * hidden]),
            );
            let i8_chunk = lmhead_i8
                .map(|_| {
                    let t_q = std::time::Instant::now();
                    let q = quantize_i8_host(&embed[off * hidden..(off + rows) * hidden], rows, hidden);
                    quantize_s += t_q.elapsed().as_secs_f64();
                    (
                        b.upload_u32("g4m-lmhead-i8", &q.q),
                        b.upload_u32("g4m-lmhead-i8-sf", &pack_pairs(&q.scales)),
                    )
                });
            let p = b.uni(
                "g4m-embed-p",
                GatherParams {
                    row_off: off as u32,
                    n_rows: rows as u32,
                    hidden_words: hidden_words as u32,
                    vocab: vocab as u32,
                    scale: embed_scale,
                    ..Default::default()
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.push(
                "g4m-gather",
                &s.moe,
                "g4m_gather_embed",
                &[(30, &buf), (31, &tok_buf), (32, &res_a), (33, &p)],
                grid,
            )?;
            if let Some(sc) = &pf {
                let pp = b.uni(
                    "g4m-pf-embed-p",
                    PfGatherParams {
                        row_off: off as u32,
                        n_rows: rows as u32,
                        hidden_words: hidden_words as u32,
                        vocab: vocab as u32,
                        scale: embed_scale,
                        m: pf_m as u32,
                        ..Default::default()
                    },
                );
                b.to_prefill = true;
                b.push(
                    "g4m-pf-gather",
                    &s.prefill,
                    "pm_gather_embed",
                    &[(0, &buf), (1, &sc.tok), (2, &sc.res_a), (3, &pp)],
                    ((hidden_words as u32).div_ceil(256), pf_m as u32, 1),
                )?;
                b.to_prefill = false;
            }
            embed_chunks.push((buf.clone(), off, rows, i8_chunk));
            off += rows;
        }
        drop(embed);
        anyhow::ensure!(
            !head_one_shard_enabled() || embed_chunks.len() == 1,
            "NV_G4MOE_HEAD_ONE_SHARD=1 must leave the {} B embed/lm-head table in ONE shard so \
             g4m-gather and g4m-lmhead are one dispatch each; it split into {} because this \
             adapter binds at most {} B of storage in one binding (max_buffer_size {} B). The \
             arm removes two wgpu barriers or it removes none -- it must never half-apply",
            (vocab * hidden * 2),
            embed_chunks.len(),
            ctx.caps.max_storage_buffer_binding_size,
            ctx.caps.max_buffer_size
        );

        if let Some(sc) = &pf {
            let sp = b.uni(
                "g4m-pf-splice-p",
                PfSpliceParams {
                    hidden_words: hidden_words as u32,
                    m: pf_m as u32,
                    ..Default::default()
                },
            );
            b.to_prefill = true;
            b.push(
                "g4m-pf-splice",
                &s.prefill,
                "pm_splice_embed_rows",
                &[
                    (100, &sc.splice_rows),
                    (101, &sc.splice_mask),
                    (102, &sc.res_a),
                    (103, &sp),
                ],
                ((hidden_words as u32).div_ceil(256), pf_m as u32, 1),
            )?;
            b.to_prefill = false;
        }

        let mut state_buffers: Vec<(wgpu::Buffer, u64)> = Vec::new();
        let mut rope_cache: RopeTableCache = RopeTableCache::new();

        let grouped = grouped_gemv_active(ctx);
        anyhow::ensure!(
            !grouped || hidden_words <= GROUPED_NORM_CHAIN_MAX_HIDDEN_WORDS_PN_BUF,
            "NV_G4MOE_GROUPED_GEMV=1: the fused norm chain stages {hidden_words} hidden words \
             in a {GROUPED_NORM_CHAIN_MAX_HIDDEN_WORDS_PN_BUF}-word workgroup array"
        );
        let fold = fold_chain_enabled();
        anyhow::ensure!(
            !fold || grouped,
            "NV_G4MOE_FOLD_CHAIN=1 extends the grouped tailnorm entry; set NV_G4MOE_GROUPED_GEMV=1 too"
        );
        let rpfuse = rpfuse_active(ctx);
        let topk_nx2 = topk_nx2_active(ctx, config.num_experts, config.top_k_experts);
        anyhow::ensure!(
            !rpfuse
                || (grouped
                    && rpnorm_active(ctx)
                    && !router_chain_enabled()
                    && !topk_fold_enabled()
                    && ctx.caps.subgroup
                    && ctx.subgroup_width() == Some(32)),
            "NV_G4MOE_RPFUSE=1 rides the rpnorm router gemv on the mlp-down dispatch: it needs \
             NV_G4MOE_GROUPED_GEMV=1 + NV_G4MOE_RPNORM=1, no router-chain/topk-fold, and its \
             subgroupShuffleXor reductions need subgroup width 32"
        );
        anyhow::ensure!(
            !topk_nx2 || rpfuse,
            "NV_G4MOE_TOPK_NX2=1 needs the router logits written before the normx2 dispatch, \
             which only NV_G4MOE_RPFUSE=1 provides"
        );
        anyhow::ensure!(
            !topk_nx2
                || router_topk_entry(config.num_experts, config.top_k_experts)
                    == "g4m_router_topk_par",
            "NV_G4MOE_TOPK_NX2=1 inlines the parallel-rank topk (tie rule \
             {TOPK_PAR_TIE_RULE_HIGHER_LOGIT_OUTRANKS_AND_EQUAL_LOGITS_RANK_BY_LOWER_EXPERT_INDEX}) \
             into the normx2 entry; {} experts / top-{} must route g4m_router_topk_par and \
             NV_G4MOE_ROUTER_TOPK must not force serial",
            config.num_experts,
            config.top_k_experts
        );
        let mut pending_tail: Option<TailFoldDeferredUntilNextLayersInputNormWeightIsUploaded> =
            None;
        let push_tail_normnext =
            |b: &mut Builder,
             pt: TailFoldDeferredUntilNextLayersInputNormWeightIsUploaded,
             next_norm_w: &wgpu::Buffer,
             normed_out: &wgpu::Buffer|
             -> Result<()> {
                let p = b.uni(
                    "g4m-pn-p",
                    NormParams {
                        hidden: hidden as u32,
                        words: hidden_words as u32,
                        eps,
                        scale: pt.layer_scalar,
                    },
                );
                let (src, entry) = if norm_lat_active(b.ctx) {
                    (&s.prop_sglat, "g4m_norm_add_norm_resout_normnext_sglat")
                } else {
                    (&s.prop, "g4m_norm_add_norm_resout_normnext")
                };
                b.push(
                    "g4m-tailnorm-nn",
                    src,
                    entry,
                    &[
                        (0, &pt.moe_out),
                        (1, &pt.post_ffw2_w),
                        (2, &pt.r_out),
                        (3, &p),
                        (4, &pt.r_in),
                        (5, &pt.post_ffw_w),
                        (6, &pt.h1),
                        (7, normed_out),
                        (8, next_norm_w),
                    ],
                    (1, 1, 1),
                )
            };
        let tap_layer_of = |want: LayerType| -> Option<usize> {
            (0..base.num_hidden_layers)
                .filter(|&i| base.kv_source_layer(i).is_none() && base.layer_kind(i) == want)
                .next_back()
        };
        let tap_sliding_layer = tap_layer_of(LayerType::SlidingAttention);
        let tap_full_layer = tap_layer_of(LayerType::FullAttention);
        let mut tap_sliding_kv: Option<AssistantKvTap> = None;
        let mut tap_full_kv: Option<AssistantKvTap> = None;

        for li in 0..base.num_hidden_layers {
            let t_layer = std::time::Instant::now();
            let layer = host_layer_from_loader(&config, weights, li)?;
            quantize_s += t_layer.elapsed().as_secs_f64();
            let (r_in, r_out) = if li % 2 == 0 {
                (&res_a, &res_b)
            } else {
                (&res_b, &res_a)
            };

            let in_w = b.upload_u32("g4m-ln", &pack_pairs(&layer.input_ln));
            let tailfold = tailfold_enabled()
                && fold
                && !kv_fp8_enabled()
                && gemv_wide_enabled("g4m-at-qkvproj")
                && hidden.is_multiple_of(8);
            let deferred_tail = match pending_tail.take() {
                Some(pt) if tailfold => Some(pt),
                Some(pt) => {
                    push_tail_normnext(&mut b, pt, &in_w, &normed)?;
                    None
                }
                None => {
                    push_norm(&mut b, &s, "g4m-rms-in", r_in, &in_w, &normed, hidden, eps)?;
                    None
                }
            };

            let attn_out = b.zeros("g4m-attn-out", (hidden_words * 4) as u64);
            let ag = build_attn(
                &mut b,
                &s,
                &config,
                &layer,
                &normed,
                &attn_out,
                &pos_buf,
                max_seq,
                sliding_ring_rows,
                flash_fd.as_ref().map(|(buf, _)| buf),
                flash_sl.as_ref().map(|(buf, _, _)| buf),
                &mut rope_cache,
                &mut state_buffers,
                deferred_tail.as_ref().map(|pt| (pt, &in_w)),
            )?;
            for (want_li, slot) in [
                (tap_sliding_layer, &mut tap_sliding_kv),
                (tap_full_layer, &mut tap_full_kv),
            ] {
                if want_li == Some(li) {
                    *slot = Some(AssistantKvTap {
                        kc: ag.kc.clone(),
                        vc: ag.vc.clone(),
                        fp8: ag.kv_scales.is_some(),
                        geom: AssistantKvTapGeometry {
                            n_kv: ag.n_kv,
                            head_dim: ag.hd,
                            cache_rows: if ag.ring > 0 {
                                ag.ring as usize
                            } else {
                                max_seq
                            },
                            ring: ag.ring > 0,
                        },
                    });
                }
            }

            let attn_post = b.zeros("g4m-attn-post", (hidden_words * 4) as u64);
            let post_attn_w = b.upload_u32("g4m-ln", &pack_pairs(&layer.post_attn_ln));
            let pre_ffw_w = b.upload_u32("g4m-ln", &pack_pairs(&layer.pre_ffw_ln));
            let normed_mlp = b.zeros("g4m-normed-mlp", (hidden_words * 4) as u64);
            if grouped {
                let p = b.uni(
                    "g4m-pn-p",
                    NormParams {
                        hidden: hidden as u32,
                        words: hidden_words as u32,
                        eps,
                        scale: 0.0,
                    },
                );
                let (src, entry) = if norm_lat_active(b.ctx) {
                    (&s.prop_sglat, "g4m_norm_norm_residual_sglat")
                } else {
                    (&s.prop, "g4m_norm_norm_residual")
                };
                b.push(
                    "g4m-nnres",
                    src,
                    entry,
                    &[
                        (0, &attn_out),
                        (1, &post_attn_w),
                        (2, &normed_mlp),
                        (3, &p),
                        (4, r_in),
                        (5, &pre_ffw_w),
                    ],
                    (1, 1, 1),
                )?;
            } else {
                push_norm(
                    &mut b,
                    &s,
                    "g4m-rms-postattn",
                    &attn_out,
                    &post_attn_w,
                    &attn_post,
                    hidden,
                    eps,
                )?;
                push_norm_residual(
                    &mut b,
                    &s,
                    "g4m-pnres",
                    &attn_post,
                    r_in,
                    &pre_ffw_w,
                    &normed_mlp,
                    hidden,
                    eps,
                )?;
            }

            let dense_out = b.zeros("g4m-dense-out", (hidden_words * 4) as u64);
            let rp_pre = rpfuse.then(|| upload_router_pre(&mut b, &config, &layer));
            let mg = build_mlp(
                &mut b,
                &s,
                &config,
                &layer,
                &normed_mlp,
                &dense_out,
                rp_pre.as_ref().map(|p| (p, r_in as &wgpu::Buffer)),
            )?;
            let h1 = b.zeros("g4m-h1", (hidden_words * 4) as u64);
            let post_ffw1_w = b.upload_u32("g4m-ln", &pack_pairs(&layer.post_ffw_ln_1));
            let moe_in = b.zeros("g4m-moe-in", (hidden_words * 4) as u64);
            let pre_ffw2_w = b.upload_u32("g4m-ln", &pack_pairs(&layer.pre_ffw_ln_2));
            if grouped {
                let p = b.uni(
                    "g4m-pn-p",
                    NormParams {
                        hidden: hidden as u32,
                        words: hidden_words as u32,
                        eps,
                        scale: 0.0,
                    },
                );
                match (topk_nx2, &rp_pre) {
                    (true, Some(pre)) => {
                        let tp = b.uni(
                            "g4m-moe-topk-p",
                            RouterParams {
                                n_experts: config.num_experts as u32,
                                k: config.top_k_experts as u32,
                                ..Default::default()
                            },
                        );
                        b.push(
                            "g4m-normx2-topk",
                            &s.prop_norm_x2_topk_sglat,
                            "g4m_norm_x2_topk_sglat",
                            &[
                                (0, &dense_out),
                                (1, &post_ffw1_w),
                                (2, &h1),
                                (3, &p),
                                (4, r_in),
                                (5, &pre_ffw2_w),
                                (7, &moe_in),
                                (10, &pre.rlogits),
                                (11, &pre.ids),
                                (12, &pre.wts),
                                (13, &tp),
                                (14, &pre.pes),
                            ],
                            (3, 1, 1),
                        )?;
                    }
                    _ => {
                        let (src, entry) = if norm_lat_active(b.ctx) {
                            (&s.prop_sglat, "g4m_norm_x2_sglat")
                        } else {
                            (&s.prop, "g4m_norm_x2")
                        };
                        b.push(
                            "g4m-normx2",
                            src,
                            entry,
                            &[
                                (0, &dense_out),
                                (1, &post_ffw1_w),
                                (2, &h1),
                                (3, &p),
                                (4, r_in),
                                (5, &pre_ffw2_w),
                                (7, &moe_in),
                            ],
                            (2, 1, 1),
                        )?;
                    }
                }
            } else {
                push_norm(
                    &mut b,
                    &s,
                    "g4m-rms-postffw1",
                    &dense_out,
                    &post_ffw1_w,
                    &h1,
                    hidden,
                    eps,
                )?;
                push_norm(
                    &mut b,
                    &s,
                    "g4m-rms-preffw2",
                    r_in,
                    &pre_ffw2_w,
                    &moe_in,
                    hidden,
                    eps,
                )?;
            }

            let moe_out = b.zeros("g4m-moe-out", (hidden_words * 4) as u64);
            let moeg = build_moe(
                &mut b,
                &s,
                &config,
                &layer,
                r_in,
                &moe_in,
                &moe_out,
                rp_pre.as_ref(),
                topk_nx2,
            )?;
            let post_ffw2_w = b.upload_u32("g4m-ln", &pack_pairs(&layer.post_ffw_ln_2));
            let post_ffw_w = b.upload_u32("g4m-ln", &pack_pairs(&layer.post_ffw_ln));
            if fold {
                pending_tail = Some(TailFoldDeferredUntilNextLayersInputNormWeightIsUploaded {
                    moe_out: moe_out.clone(),
                    post_ffw2_w: post_ffw2_w.clone(),
                    r_out: r_out.clone(),
                    r_in: r_in.clone(),
                    post_ffw_w: post_ffw_w.clone(),
                    h1: h1.clone(),
                    layer_scalar: layer.layer_scalar,
                });
            } else if grouped {
                let p = b.uni(
                    "g4m-pn-p",
                    NormParams {
                        hidden: hidden as u32,
                        words: hidden_words as u32,
                        eps,
                        scale: layer.layer_scalar,
                    },
                );
                b.push(
                    "g4m-tailnorm",
                    &s.prop,
                    "g4m_norm_add_norm_resout",
                    &[
                        (0, &moe_out),
                        (1, &post_ffw2_w),
                        (2, r_out),
                        (3, &p),
                        (4, r_in),
                        (5, &post_ffw_w),
                        (6, &h1),
                    ],
                    (1, 1, 1),
                )?;
            } else {
                let h2 = b.zeros("g4m-h2", (hidden_words * 4) as u64);
                push_norm(
                    &mut b,
                    &s,
                    "g4m-rms-postffw2",
                    &moe_out,
                    &post_ffw2_w,
                    &h2,
                    hidden,
                    eps,
                )?;

                let ffw_sum = b.zeros("g4m-ffw-sum", (hidden_words * 4) as u64);
                let ap = b.uni(
                    "g4m-add-p",
                    ResScaleParams {
                        n: hidden as u32,
                        n_words: hidden_words as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let grid = b.grid1(hidden_words as u64, 256);
                b.push(
                    "g4m-ffw-add",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &h1), (1, &h2), (2, &ffw_sum), (3, &ap)],
                    grid,
                )?;

                let combined = b.zeros("g4m-combined", (hidden_words * 4) as u64);
                push_norm(
                    &mut b,
                    &s,
                    "g4m-rms-postffw",
                    &ffw_sum,
                    &post_ffw_w,
                    &combined,
                    hidden,
                    eps,
                )?;

                let rp2 = b.uni(
                    "g4m-resout-p",
                    ResScaleParams {
                        n: hidden as u32,
                        n_words: hidden_words as u32,
                        scale: layer.layer_scalar,
                        ..Default::default()
                    },
                );
                b.push(
                    "g4m-res-out",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &combined), (1, r_in), (2, r_out), (3, &rp2)],
                    grid,
                )?;
            }

            if let Some(sc) = &pf {
                let (p_in, p_out) = if li % 2 == 0 {
                    (&sc.res_a, &sc.res_b)
                } else {
                    (&sc.res_b, &sc.res_a)
                };
                let m = sc.m;
                let mw = m * hidden_words;
                b.to_prefill = true;
                pf_norm(
                    &mut b,
                    &s,
                    "g4m-pf-rms-in",
                    p_in,
                    &in_w,
                    &sc.normed,
                    hidden,
                    eps,
                    m,
                )?;
                build_attn_prefill(&mut b, &s, &ag, sc, &sc.normed, &sc.attn_out, max_seq)?;
                pf_norm(
                    &mut b,
                    &s,
                    "g4m-pf-rms-postattn",
                    &sc.attn_out,
                    &post_attn_w,
                    &sc.attn_post,
                    hidden,
                    eps,
                    m,
                )?;
                pf_norm_residual(
                    &mut b,
                    &s,
                    "g4m-pf-pnres",
                    &sc.attn_post,
                    p_in,
                    &pre_ffw_w,
                    &sc.normed_mlp,
                    hidden,
                    eps,
                    m,
                )?;
                build_mlp_prefill(&mut b, &s, &mg, sc, &sc.normed_mlp, &sc.dense_out)?;
                pf_norm(
                    &mut b,
                    &s,
                    "g4m-pf-rms-postffw1",
                    &sc.dense_out,
                    &post_ffw1_w,
                    &sc.h1,
                    hidden,
                    eps,
                    m,
                )?;
                pf_norm(
                    &mut b,
                    &s,
                    "g4m-pf-rms-preffw2",
                    p_in,
                    &pre_ffw2_w,
                    &sc.moe_in,
                    hidden,
                    eps,
                    m,
                )?;
                build_moe_prefill(
                    &mut b,
                    &s,
                    &config,
                    &moeg,
                    sc,
                    p_in,
                    &sc.moe_in,
                    &sc.moe_out,
                )?;
                pf_norm(
                    &mut b,
                    &s,
                    "g4m-pf-rms-postffw2",
                    &sc.moe_out,
                    &post_ffw2_w,
                    &sc.h2,
                    hidden,
                    eps,
                    m,
                )?;
                let ap = b.uni(
                    "g4m-pf-add-p",
                    ResScaleParams {
                        n: (m * hidden) as u32,
                        n_words: mw as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let mgrid = b.grid1(mw as u64, 256);
                b.push(
                    "g4m-pf-ffw-add",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &sc.h1), (1, &sc.h2), (2, &sc.ffw_sum), (3, &ap)],
                    mgrid,
                )?;
                pf_norm(
                    &mut b,
                    &s,
                    "g4m-pf-rms-postffw",
                    &sc.ffw_sum,
                    &post_ffw_w,
                    &sc.combined,
                    hidden,
                    eps,
                    m,
                )?;
                let rp = b.uni(
                    "g4m-pf-resout-p",
                    ResScaleParams {
                        n: (m * hidden) as u32,
                        n_words: mw as u32,
                        scale: layer.layer_scalar,
                        ..Default::default()
                    },
                );
                b.push(
                    "g4m-pf-res-out",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &sc.combined), (1, p_in), (2, p_out), (3, &rp)],
                    mgrid,
                )?;
                b.to_prefill = false;
            }

            b.flush_staging_if_due();
        }

        let r_last = if base.num_hidden_layers.is_multiple_of(2) {
            &res_a
        } else {
            &res_b
        };
        let assistant_taps = AssistantTaps {
            sliding_layer: tap_sliding_layer,
            full_layer: tap_full_layer,
            sliding: tap_sliding_kv,
            full: tap_full_kv,
            decode_prenorm_hidden: r_last.clone(),
            verify_prenorm_hidden: pf.as_ref().map(|sc| {
                if base.num_hidden_layers.is_multiple_of(2) {
                    sc.res_a.clone()
                } else {
                    sc.res_b.clone()
                }
            }),
        };
        let final_w = b.upload_u32(
            "g4m-final-ln",
            &pack_pairs(&load_bf16(
                weights,
                "model.language_model.norm.weight",
                &[hidden],
            )?),
        );
        let final_x = b.zeros("g4m-final-x", (hidden_words * 4) as u64);
        match pending_tail.take() {
            Some(pt) => push_tail_normnext(&mut b, pt, &final_w, &final_x)?,
            None => push_norm(
                &mut b,
                &s,
                "g4m-final-rms",
                r_last,
                &final_w,
                &final_x,
                hidden,
                eps,
            )?,
        }

        let head_start = b.passes.len();
        let logits_bf16 = b.zeros("g4m-logits-bf16", (vocab * 2) as u64);
        let gemv_i8_src = match lmhead_i8 {
            Some("g4m_gemv_i8_v4") => compose(GEMV_I8_V4_WGSL),
            Some(_) => compose(GEMV_I8_WGSL),
            None => String::new(),
        };
        let mut head_chunks: Vec<LmHeadChunk> = Vec::with_capacity(embed_chunks.len());
        for (buf, off, rows, i8_chunk) in &embed_chunks {
            let wide = if lmhead_i8.is_some() && i8_chunk.is_some() {
                0
            } else {
                b.wide_flag("g4m-lmhead", GemvLoad::Wide, hidden, 0, buf, &final_x)
            };
            head_chunks.push(LmHeadChunk {
                w: buf,
                row_off: *off,
                n_rows: *rows,
                i8: i8_chunk.as_ref(),
                wide,
            });
        }
        for c in &head_chunks {
            push_lmhead_row(
                &mut b,
                &s,
                &gemv_i8_src,
                lmhead_i8,
                c,
                &final_x,
                0,
                &logits_bf16,
                c.row_off / 2,
                hidden,
            )?;
        }

        let logits = b.zeros("g4m-logits", (vocab * 4) as u64);
        let cap = base.final_logit_softcapping;
        let cp = b.uni(
            "g4m-cap-p",
            ResScaleParams {
                n: vocab as u32,
                n_words: (vocab / 2) as u32,
                scale: 1.0,
                cap,
                inv_cap: if cap > 0.0 { 1.0 / cap } else { 0.0 },
                ..Default::default()
            },
        );
        let grid = b.grid1((vocab / 2) as u64, 256);
        let cap_entry = if cap > 0.0 {
            "tanh_softcap_bf16_to_f32"
        } else {
            "cast_bf16_to_f32"
        };
        b.push(
            "g4m-softcap",
            &s.resscale,
            cap_entry,
            &[(0, &logits_bf16), (3, &cp), (4, &logits)],
            grid,
        )?;

        let pv = b.zeros("g4m-am-pv", (ARGMAX_GROUPS * 4) as u64);
        let pi = b.zeros("g4m-am-pi", (ARGMAX_GROUPS * 4) as u64);
        let token_out = b.zeros("g4m-token", 4);
        let ap = b.uni(
            "g4m-am-p",
            ArgmaxParams {
                n: vocab as u32,
                groups: ARGMAX_GROUPS as u32,
                ..Default::default()
            },
        );
        b.push(
            "g4m-am1",
            &s.moe,
            "g4m_argmax_bf16_stage1",
            &[(60, &logits_bf16), (61, &pv), (62, &pi), (64, &ap)],
            (ARGMAX_GROUPS as u32, 1, 1),
        )?;
        b.push(
            "g4m-am2",
            &s.moe,
            "g4m_argmax_stage2",
            &[(61, &pv), (62, &pi), (63, &token_out), (64, &ap)],
            (1, 1, 1),
        )?;

        let verify = match &pf {
            Some(sc)
                if lmhead_i8.is_none()
                    && !attn_i8_caps_verify_rows_to_zero_because_int8_decode_vs_bf16_prefill_breaks_the_argmax_contract()
                    && hidden_words.is_multiple_of(4) =>
            {
                let vrows = sc.m.min(VERIFY_ROWS_MAX_IS_THE_LONGEST_CHAIN_THE_SPEC_LOOP_SUBMITS);
                let pf_last = if base.num_hidden_layers.is_multiple_of(2) {
                    &sc.res_a
                } else {
                    &sc.res_b
                };
                let v_normed = b.zeros("g4m-verify-normed", (vrows * hidden_words * 4) as u64);
                let v_logits_bf16 = b.zeros("g4m-verify-logits-bf16", (vrows * vocab * 2) as u64);
                let v_logits = b.zeros("g4m-verify-logits", (vrows * vocab * 4) as u64);
                let v_pv = b.zeros("g4m-verify-am-pv", (vrows * ARGMAX_GROUPS * 4) as u64);
                let v_pi = b.zeros("g4m-verify-am-pi", (vrows * ARGMAX_GROUPS * 4) as u64);
                let v_tokens = b.zeros("g4m-verify-tokens", (vrows * 4) as u64);
                let vstart = b.prefill_passes.len();
                b.to_prefill = true;
                pf_norm(
                    &mut b,
                    &s,
                    "g4m-verify-final-rms",
                    pf_last,
                    &final_w,
                    &v_normed,
                    hidden,
                    eps,
                    vrows,
                )?;
                for r in 0..vrows {
                    for c in &head_chunks {
                        push_lmhead_row(
                            &mut b,
                            &s,
                            &gemv_i8_src,
                            lmhead_i8,
                            c,
                            &v_normed,
                            r * hidden_words,
                            &v_logits_bf16,
                            r * (vocab / 2) + c.row_off / 2,
                            hidden,
                        )?;
                    }
                }
                let vcp = b.uni(
                    "g4m-verify-cap-p",
                    ResScaleParams {
                        n: (vrows * vocab) as u32,
                        n_words: (vrows * vocab / 2) as u32,
                        scale: 1.0,
                        cap,
                        inv_cap: if cap > 0.0 { 1.0 / cap } else { 0.0 },
                        ..Default::default()
                    },
                );
                let vgrid = b.grid1((vrows * vocab / 2) as u64, 256);
                b.push(
                    "g4m-verify-softcap",
                    &s.resscale,
                    cap_entry,
                    &[(0, &v_logits_bf16), (3, &vcp), (4, &v_logits)],
                    vgrid,
                )?;
                let vap = b.uni(
                    "g4m-verify-am-p",
                    PfVerifyArgmaxParams {
                        n: vocab as u32,
                        groups: ARGMAX_GROUPS as u32,
                        ..Default::default()
                    },
                );
                b.push(
                    "g4m-verify-am1",
                    &s.prefill,
                    "pm_verify_argmax_bf16_stage1",
                    &[(110, &v_logits_bf16), (111, &v_pv), (112, &v_pi), (114, &vap)],
                    (ARGMAX_GROUPS as u32, vrows as u32, 1),
                )?;
                b.push(
                    "g4m-verify-am2",
                    &s.prefill,
                    "pm_verify_argmax_stage2",
                    &[(111, &v_pv), (112, &v_pi), (113, &v_tokens), (114, &vap)],
                    (1, vrows as u32, 1),
                )?;
                b.to_prefill = false;
                let passes = b.prefill_passes.split_off(vstart);
                Some(VerifyState {
                    rows: vrows,
                    passes,
                    logits: v_logits,
                    tokens: v_tokens,
                    validated: false,
                })
            }
            _ => None,
        };
        drop(head_chunks);
        drop(embed_chunks);

        b.flush_staging();
        let vram = b.report();
        if vram_report_enabled() {
            eprint!("[g4m-wgpu] {}", vram.render());
        }
        let load = LoadReport {
            build_s: t_build.elapsed().as_secs_f64(),
            quantize_s,
            wired_bytes: vram.total_bytes,
        };

        let Builder {
            passes,
            prefill_passes,
            buffers,
            weight_bpt,
            wide_gemvs,
            narrow_gemvs,
            depthgate_flash1_pass_indices_full_attention_the_band_is_keyed_on_the_whole_context:
                depthgate_flash1_passes,
            depthgate_flash1_pass_indices_sliding_the_window_caps_the_scan_so_the_band_is_keyed_on_min_total_window:
                depthgate_flash1_sliding_passes,
            foldgate_alt_flash1_passes_one_deep_twin_per_full_attention_flash1_in_the_same_order:
                foldgate_alt_flash1,
            flash_scratch_bytes_per_full_attention_layer_sized_at_its_own_split_count:
                flash_scratch_bytes_full,
            flash_scratch_bytes_per_sliding_layer_sized_at_its_own_split_count_not_the_max_of_the_two:
                flash_scratch_bytes_sliding,
            ..
        } = b;
        let dense_gemv_wide = (wide_gemvs, narrow_gemvs);

        let prefill = match pf {
            Some(sc) if !prefill_passes.is_empty() => Some(PfState {
                m: sc.m,
                passes: prefill_passes,
                tok: sc.tok,
                pos: sc.pos,
                splice_rows: sc.splice_rows,
                splice_mask: sc.splice_mask,
                splice_mask_live: false,
                validated: false,
            }),
            _ => None,
        };
        let verify = match (&prefill, verify) {
            (Some(_), v) => v,
            (None, _) => None,
        };
        eprintln!(
            "[g4m-wgpu] chunked prefill: {} verify rows: {}",
            match &prefill {
                Some(p) => format!("m={} passes={}", p.m, p.passes.len()),
                None => "off".to_string(),
            },
            verify.as_ref().map(|v| v.rows).unwrap_or(0)
        );

        let mut model = Self {
            ctx,
            config,
            max_seq,
            pos: 0,
            validated: false,
            prefix_validated: false,
            passes,
            prefill,
            verify,
            head_start,
            preenc: preenc_enabled(),
            pending_cb: None,
            preenc_replays: 0,
            preenc_band_discards: 0,
            _buffers: buffers,
            tok_buf,
            pos_buf,
            flash_fd,
            flash_sl,
            depthgate_flash1_passes,
            depthgate_flash1_sliding_passes,
            depthgate_on:
                flash_depthgate_enabled_a_shallow_tuned_split_pin_starves_the_grid_at_depth(),
            depthgate_shallow_splits:
                flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
            depthgate_shallow_splits_sliding:
                flash_splits_for_layer_kind_full_attention_holds_the_folded_grid_sliding_holds_the_stage2_scan(
                    LayerType::SlidingAttention,
                    flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
                ),
            foldgate_alt_flash1,
            foldgate_deep_active: false,
            flash_scratch_bytes_full,
            flash_scratch_bytes_sliding,
            token_out,
            logits,
            state_buffers,
            vocab,
            vram,
            load,
            weight_bytes: weight_bpt.round() as u64,
            dense_gemv_wide,
            assistant_taps,
        };
        anyhow::ensure!(
            model.foldgate_alt_flash1.is_empty()
                || model.foldgate_alt_flash1.len() == model.depthgate_flash1_passes.len(),
            "the foldgate built {} deep twins for {} full-attention flash1 passes; a mismatched \
             slot order would swap the wrong pass at the knee",
            model.foldgate_alt_flash1.len(),
            model.depthgate_flash1_passes.len()
        );
        let t_warm = std::time::Instant::now();
        let warm_m = model.prefill_chunk_len();
        if warm_m > 0 && warm_m <= max_seq {
            model
                .prefill_chunk(&vec![0u32; warm_m])
                .context("warmup prefill chunk")?;
            model.reset()?;
        }
        model.decode_step(0).context("warmup decode step")?;
        model.reset()?;
        model.load.build_s += t_warm.elapsed().as_secs_f64();
        Ok(model)
    }

    fn depthgate_splits_for_this_step(&self) -> Option<u32> {
        (self.depthgate_on && !self.depthgate_flash1_passes.is_empty()).then(|| {
            flash_depthgate_splits_for_context_tokens(
                (self.pos + 1) as u32,
                self.depthgate_shallow_splits,
            )
        })
    }

    fn depthgate_sliding_splits_for_this_step(&self) -> Option<u32> {
        let window = self.flash_sl.as_ref().map(|(_, _, w)| *w)?;
        (self.depthgate_on && !self.depthgate_flash1_sliding_passes.is_empty()).then(|| {
            flash_depthgate_splits_for_context_tokens(
                ((self.pos + 1) as u32).min(window),
                self.depthgate_shallow_splits_sliding,
            )
        })
    }

    #[doc(hidden)]
    pub fn flash_scratch_bytes_by_layer_kind_full_then_sliding(&self) -> (Vec<u64>, Vec<u64>) {
        (
            self.flash_scratch_bytes_full.clone(),
            self.flash_scratch_bytes_sliding.clone(),
        )
    }

    #[doc(hidden)]
    pub fn depthgate_full_attention_flash1_grid_y_as_last_encoded(&self) -> Option<u32> {
        self.depthgate_flash1_passes
            .first()
            .map(|&i| self.passes[i].grid.1)
    }

    #[doc(hidden)]
    pub fn depthgate_sliding_flash1_grid_y_as_last_encoded(&self) -> Option<u32> {
        self.depthgate_flash1_sliding_passes
            .first()
            .map(|&i| self.passes[i].grid.1)
    }

    #[doc(hidden)]
    pub fn foldgate_full_attention_flash1_grid_xy_as_last_encoded(&self) -> Option<(u32, u32)> {
        self.depthgate_flash1_passes
            .first()
            .map(|&i| (self.passes[i].grid.0, self.passes[i].grid.1))
    }

    #[doc(hidden)]
    pub fn foldgate_deep_arm_swapped_in_as_of_last_step(&self) -> bool {
        self.foldgate_deep_active
    }

    #[doc(hidden)]
    pub fn preenc_replay_and_band_discard_counts(&self) -> (usize, usize) {
        (self.preenc_replays, self.preenc_band_discards)
    }

    fn pending_cb_band_key_a_cb_encoded_under_one_gate_band_must_never_be_submitted_under_another(
        &self,
        full: bool,
    ) -> PendingCbBandKey {
        (
            if full {
                self.passes.len()
            } else {
                self.head_start
            },
            self.depthgate_flash1_passes
                .first()
                .map(|&i| self.passes[i].grid.1)
                .unwrap_or(0),
            self.depthgate_flash1_sliding_passes
                .first()
                .map(|&i| self.passes[i].grid.1)
                .unwrap_or(0),
            self.foldgate_deep_active,
        )
    }

    #[doc(hidden)]
    pub fn full_attention_flash1_pass_labels(&self) -> Vec<String> {
        self.depthgate_flash1_passes
            .iter()
            .map(|&i| self.passes[i].label.clone())
            .collect()
    }

    fn foldgate_swap_full_attention_passes_at_the_knee_then_report_the_deep_split_override(
        &mut self,
    ) -> Option<u32> {
        if self.foldgate_alt_flash1.is_empty() {
            return None;
        }
        let want_deep = foldgate_deep_at_these_whole_context_tokens((self.pos + 1) as u32);
        if want_deep != self.foldgate_deep_active {
            for (slot, i) in self.depthgate_flash1_passes.iter().enumerate() {
                std::mem::swap(&mut self.passes[*i], &mut self.foldgate_alt_flash1[slot]);
            }
            self.foldgate_deep_active = want_deep;
        }
        self.foldgate_deep_active.then_some(
            FOLDGATE_DEEP_SPLITS_64_RESTORES_512_WORKGROUPS_UNDER_FOLD_2_AND_IS_A_MEASURED_RUNG_THE_LADDER_HAS_A_REPRODUCED_POTHOLE_AT_40_47_48_SO_NEVER_INTERPOLATE,
        )
    }

    fn write_flash_total_uniform_mirroring_gemma4_wgpu_write_pos_uniforms(&mut self) {
        let foldgate_deep_splits =
            self.foldgate_swap_full_attention_passes_at_the_knee_then_report_the_deep_split_override();
        let full_splits = foldgate_deep_splits.or_else(|| self.depthgate_splits_for_this_step());
        let depthgate_sliding_splits = self.depthgate_sliding_splits_for_this_step();
        if let Some(sp) = full_splits {
            for i in 0..self.depthgate_flash1_passes.len() {
                let pass = self.depthgate_flash1_passes[i];
                self.passes[pass].grid.1 = sp;
            }
        }
        if let Some(sp) = depthgate_sliding_splits {
            for i in 0..self.depthgate_flash1_sliding_passes.len() {
                let pass = self.depthgate_flash1_sliding_passes[i];
                self.passes[pass].grid.1 = sp;
            }
        }
        if let Some((buf, base)) = &self.flash_fd {
            let mut p = *base;
            p.total = (self.pos + 1) as u32;
            if let Some(sp) = full_splits {
                p.splits = sp;
            }
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }
        if let Some((buf, base, window)) = &self.flash_sl {
            let mut p = *base;
            p.total = (self.pos + 1) as u32;
            p.start = if p.total > *window { p.total - *window } else { 0 };
            if let Some(sp) = depthgate_sliding_splits {
                p.splits = sp;
            }
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }
    }

    fn step_inner(&mut self, token: u32, full: bool) -> Result<()> {
        anyhow::ensure!((token as usize) < self.vocab, "token {token} out of vocab");
        anyhow::ensure!(
            self.pos < self.max_seq,
            "kv cache full at {} (max_seq {})",
            self.pos,
            self.max_seq
        );
        self.ctx
            .queue
            .write_buffer(&self.tok_buf, 0, bytemuck::bytes_of(&(token as i32)));
        self.ctx
            .queue
            .write_buffer(&self.pos_buf, 0, bytemuck::bytes_of(&(self.pos as i32)));
        self.write_flash_total_uniform_mirroring_gemma4_wgpu_write_pos_uniforms();

        let need_scope = if full {
            !self.validated
        } else {
            !self.validated && !self.prefix_validated
        };
        let scope = if need_scope {
            Some(
                self.ctx
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation),
            )
        } else {
            None
        };
        let band_key = self
            .pending_cb_band_key_a_cb_encoded_under_one_gate_band_must_never_be_submitted_under_another(
                full,
            );
        let passes = if full {
            &self.passes[..]
        } else {
            &self.passes[..self.head_start]
        };
        if dispatch::profile::enabled() && self.ctx.caps.timestamp_query {
            let raw: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))> = passes
                .iter()
                .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
                .collect();
            let labels: Vec<String> = passes.iter().map(|p| p.label.clone()).collect();
            dispatch::submit_profiled_slices(self.ctx, &raw, &labels)
                .map_err(|e| anyhow::anyhow!("profiled submit: {e}"))?;
        } else {
            let ready = match self.pending_cb.take() {
                Some((k, cb)) if k == band_key => Some(cb),
                Some(_) => {
                    self.preenc_band_discards += 1;
                    None
                }
                None => None,
            };
            match ready {
                Some(cb) => {
                    self.preenc_replays += 1;
                    self.ctx.queue.submit([cb]);
                }
                None => dispatch::submit_pass_list(
                    self.ctx,
                    passes.iter().map(|p| (p.pipeline.as_ref(), &p.bind, p.grid)),
                ),
            }
            if self.preenc {
                self.pending_cb = Some((
                    band_key,
                    dispatch::encode_pass_list(
                        self.ctx,
                        passes.iter().map(|p| (p.pipeline.as_ref(), &p.bind, p.grid)),
                    ),
                ));
            }
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_moe_wgpu decode step validation: {e}");
            }
            if full {
                self.validated = true;
            }
            self.prefix_validated = true;
        }
        self.pos += 1;
        Ok(())
    }

    crate::wgpu_step_readback_api!();

    pub fn prefill_chunk_len(&self) -> usize {
        self.prefill.as_ref().map(|p| p.m).unwrap_or(0)
    }

    pub fn prefill_pass_count(&self) -> usize {
        self.prefill.as_ref().map(|p| p.passes.len()).unwrap_or(0)
    }

    fn write_prefill_inputs(
        &mut self,
        tokens: &[u32],
        live: usize,
        splices: &[ChunkRowSplice],
    ) -> Result<()> {
        let ctx = self.ctx;
        let vocab = self.vocab;
        let max_seq = self.max_seq;
        let pos0 = self.pos;
        let hidden_words = self.config.base.hidden_size / 2;
        let Some(pf) = self.prefill.as_mut() else {
            anyhow::bail!("chunked prefill is disabled on this graph");
        };
        let m = pf.m;
        anyhow::ensure!(
            tokens.len() == m,
            "prefill_chunk wants exactly {m} tokens, got {}",
            tokens.len()
        );
        anyhow::ensure!(
            (1..=m).contains(&live),
            "prefill live rows {live} out of 1..={m}"
        );
        for &t in tokens {
            anyhow::ensure!((t as usize) < vocab, "token {t} out of vocab");
        }
        anyhow::ensure!(pos0 + m <= max_seq, "kv cache full at {pos0} + {m}");

        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let poss: Vec<i32> = (0..m).map(|i| (pos0 + i) as i32).collect();
        ctx.queue
            .write_buffer(&pf.tok, 0, bytemuck::cast_slice(&ids));
        ctx.queue
            .write_buffer(&pf.pos, 0, bytemuck::cast_slice(&poss));
        let mut mask = vec![0u32; m.max(4)];
        for sp in splices {
            anyhow::ensure!(
                sp.rel_pos < live,
                "embed-row splice at {} is past the {live} live rows",
                sp.rel_pos
            );
            anyhow::ensure!(
                sp.row_words.len() == hidden_words,
                "embed-row splice has {} words, want {hidden_words}",
                sp.row_words.len()
            );
            mask[sp.rel_pos] = 1;
            ctx.queue.write_buffer(
                &pf.splice_rows,
                (sp.rel_pos * hidden_words * 4) as u64,
                bytemuck::cast_slice(sp.row_words),
            );
        }
        if !splices.is_empty() || pf.splice_mask_live {
            ctx.queue
                .write_buffer(&pf.splice_mask, 0, bytemuck::cast_slice(&mask));
        }
        pf.splice_mask_live = !splices.is_empty();
        Ok(())
    }

    fn prefill_chunk_masked(
        &mut self,
        tokens: &[u32],
        live: usize,
        splices: &[ChunkRowSplice],
    ) -> Result<()> {
        self.write_prefill_inputs(tokens, live, splices)?;
        let ctx = self.ctx;
        let pf = self
            .prefill
            .as_mut()
            .expect("write_prefill_inputs proved the prefill graph exists");
        let scope = if pf.validated {
            None
        } else {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        };
        run_pass_list(ctx, &pf.passes)?;
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_moe_wgpu prefill chunk validation: {e}");
            }
            pf.validated = true;
        }
        self.pos += live;
        Ok(())
    }

    pub fn prefill_chunk(&mut self, tokens: &[u32]) -> Result<()> {
        let m = self.prefill_chunk_len();
        anyhow::ensure!(m > 0, "chunked prefill is disabled on this graph");
        self.prefill_chunk_masked(tokens, m, &[])
    }

    pub fn verify_max_rows(&self) -> usize {
        self.verify.as_ref().map(|v| v.rows).unwrap_or(0)
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        anyhow::ensure!(
            self.pos + n <= self.max_seq,
            "advance {n} past max_seq {} at {}",
            self.max_seq,
            self.pos
        );
        self.pos += n;
        Ok(())
    }

    pub fn truncate_to(&mut self, pos: usize) -> Result<()> {
        anyhow::ensure!(pos <= self.pos, "truncate_to {pos} beyond pos {}", self.pos);
        self.pos = pos;
        Ok(())
    }

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        anyhow::ensure!(
            !flash_decode_enabled(),
            "verify_chain's m-row pm_attn rows argmax-mismatch the flash decode stream on real \
             weights at final_logit_softcapping near-ties, so the verify surface refuses the \
             flash default; set NV_G4MOE_FLASH_DECODE=0 for any spec/verify consumer until the \
             m-row graph grows a flash-consistent arm"
        );
        let rows = self.verify_max_rows();
        anyhow::ensure!(
            rows > 0,
            "verify_chain needs the m-row prefill graph and the bf16 lm_head epilogue"
        );
        let mb = batch.len();
        anyhow::ensure!(
            (1..=rows).contains(&mb),
            "verify_chain batch {mb} out of 1..={rows}"
        );
        let m = self.prefill_chunk_len();
        let mut padded = batch.to_vec();
        let pad = *padded.last().expect("batch is non-empty");
        padded.resize(m, pad);
        self.write_prefill_inputs(&padded, mb, &[])?;
        let ctx = self.ctx;
        {
            let pf = self
                .prefill
                .as_mut()
                .expect("write_prefill_inputs proved the prefill graph exists");
            let scope = if pf.validated {
                None
            } else {
                Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
            };
            run_pass_list(ctx, &pf.passes)?;
            if let Some(scope) = scope {
                if let Some(e) = pollster::block_on(scope.pop()) {
                    anyhow::bail!("gemma4_moe_wgpu verify chain prefill validation: {e}");
                }
                pf.validated = true;
            }
        }
        let vs = self
            .verify
            .as_mut()
            .expect("verify_max_rows > 0 proved the verify epilogue exists");
        let scope = if vs.validated {
            None
        } else {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        };
        run_pass_list(ctx, &vs.passes)?;
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_moe_wgpu verify chain epilogue validation: {e}");
            }
            vs.validated = true;
        }
        let toks: Vec<u32> = dispatch::read_back(ctx, &vs.tokens, rows)
            .map_err(|e| anyhow::anyhow!("verify tokens read back: {e}"))?;
        Ok(toks[..mb].to_vec())
    }

    pub fn verify_row_logits(&self, row: usize) -> Result<Vec<f32>> {
        let vs = self
            .verify
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("verify epilogue disabled"))?;
        anyhow::ensure!(row < vs.rows, "verify row {row} out of 0..{}", vs.rows);
        let all: Vec<f32> = dispatch::read_back(self.ctx, &vs.logits, vs.rows * self.vocab)
            .map_err(|e| anyhow::anyhow!("verify logits read back: {e}"))?;
        Ok(all[row * self.vocab..(row + 1) * self.vocab].to_vec())
    }

    pub fn assistant_shared_kv_layers_last_kv_writing_sliding_then_full(
        &self,
    ) -> (Option<usize>, Option<usize>) {
        (
            self.assistant_taps.sliding_layer,
            self.assistant_taps.full_layer,
        )
    }

    fn assistant_kv_tap(&self, kind: LayerType) -> Result<&AssistantKvTap> {
        let tap = match kind {
            LayerType::SlidingAttention => self.assistant_taps.sliding.as_ref(),
            LayerType::FullAttention => self.assistant_taps.full.as_ref(),
        };
        let tap = tap.context("assistant kv tap missing for this layer kind")?;
        anyhow::ensure!(
            !tap.fp8,
            "assistant kv tap reads the bf16 cache layout; NV_G4MOE_KV_FP8=1 stores e4m3 bytes \
             plus scales -- unset it for assistant drafting"
        );
        Ok(tap)
    }

    pub fn assistant_kv_geometry(&self, kind: LayerType) -> Result<AssistantKvTapGeometry> {
        Ok(self.assistant_kv_tap(kind)?.geom)
    }

    pub fn assistant_kv_rows_f32_layout_len_by_n_kv_by_head_dim(
        &self,
        kind: LayerType,
        start: usize,
        len: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let tap = self.assistant_kv_tap(kind)?;
        let g = tap.geom;
        anyhow::ensure!(len > 0, "empty assistant kv row read");
        anyhow::ensure!(
            start + len <= self.pos,
            "assistant kv rows {start}+{len} beyond committed {}",
            self.pos
        );
        if g.ring {
            let pad = self.prefill_chunk_len();
            anyhow::ensure!(
                start + g.cache_rows >= self.pos + pad,
                "{ASSISTANT_RING_PAD_CLOBBER_SLOTS_ARE_THE_PREFILL_M_ROWS_VERIFY_PADS_PAST_THE_COMMITTED_POS}: \
                 start {start} with ring {} pad {pad} cannot serve committed {}",
                g.cache_rows,
                self.pos
            );
        }
        let words_per_row = g.n_kv * g.head_dim / 2;
        let mut k_words: Vec<u32> = Vec::with_capacity(len * words_per_row);
        let mut v_words: Vec<u32> = Vec::with_capacity(len * words_per_row);
        let mut i = 0usize;
        while i < len {
            let slot = if g.ring {
                (start + i) % g.cache_rows
            } else {
                start + i
            };
            let span = if g.ring {
                (len - i).min(g.cache_rows - slot)
            } else {
                len - i
            };
            for (buf, out) in [(&tap.kc, &mut k_words), (&tap.vc, &mut v_words)] {
                out.extend(
                    dispatch::read_back_at::<u32>(
                        self.ctx,
                        buf,
                        slot * words_per_row,
                        span * words_per_row,
                    )
                    .map_err(|e| anyhow::anyhow!("assistant kv read back: {e}"))?,
                );
            }
            i += span;
        }
        Ok((
            unpack_bf16_pairs_to_f32(&k_words),
            unpack_bf16_pairs_to_f32(&v_words),
        ))
    }

    pub fn decode_prenorm_hidden_row_the_gemma4_assistant_drafter_consumes(
        &self,
    ) -> Result<Vec<f32>> {
        let words: Vec<u32> = dispatch::read_back(
            self.ctx,
            &self.assistant_taps.decode_prenorm_hidden,
            self.config.base.hidden_size / 2,
        )
        .map_err(|e| anyhow::anyhow!("decode prenorm hidden read back: {e}"))?;
        Ok(unpack_bf16_pairs_to_f32(&words))
    }

    pub fn verify_prenorm_hidden_row_the_gemma4_assistant_drafter_consumes(
        &self,
        row: usize,
    ) -> Result<Vec<f32>> {
        let rows = self.verify_max_rows().min(self.prefill_chunk_len());
        anyhow::ensure!(row < rows, "verify prenorm hidden row {row} out of 0..{rows}");
        let buf = self
            .assistant_taps
            .verify_prenorm_hidden
            .as_ref()
            .context("verify prenorm hidden needs the m-row prefill graph")?;
        let hidden_words = self.config.base.hidden_size / 2;
        let words: Vec<u32> =
            dispatch::read_back_at(self.ctx, buf, row * hidden_words, hidden_words)
                .map_err(|e| anyhow::anyhow!("verify prenorm hidden read back: {e}"))?;
        Ok(unpack_bf16_pairs_to_f32(&words))
    }

    pub fn prefill_tokens_with_embed_rows(
        &mut self,
        tokens: &[u32],
        splices: &[EmbedRowSplice],
    ) -> Result<usize> {
        let m = self.prefill_chunk_len();
        anyhow::ensure!(
            m >= 2,
            "embed-row splice prefill requires the m-row prefill graph (NV_G4MOE_WGPU_PREFILL_M >= 2)"
        );
        let hidden = self.config.base.hidden_size;
        let hidden_words = hidden / 2;
        let mut prev_end = 0usize;
        let mut packed: Vec<Vec<u32>> = Vec::with_capacity(splices.len());
        for sp in splices {
            anyhow::ensure!(
                !sp.rows_bf16.is_empty() && sp.rows_bf16.len().is_multiple_of(hidden),
                "embed-row splice rows_bf16 len {} is not a positive multiple of hidden {hidden}",
                sp.rows_bf16.len()
            );
            let n_slots = sp.rows_bf16.len() / hidden;
            anyhow::ensure!(
                sp.position >= prev_end,
                "embed-row splices must be sorted and non-overlapping"
            );
            anyhow::ensure!(
                sp.position + n_slots <= tokens.len(),
                "embed-row splice at {} with {n_slots} rows exceeds {} tokens",
                sp.position,
                tokens.len()
            );
            prev_end = sp.position + n_slots;
            packed.push(pack_pairs(&sp.rows_bf16));
        }
        let mut done = 0usize;
        while done < tokens.len() {
            let live = m.min(tokens.len() - done);
            anyhow::ensure!(
                self.pos + m <= self.max_seq,
                "kv cache full at {} + {m} (max_seq {})",
                self.pos,
                self.max_seq
            );
            let mut chunk: Vec<u32> = tokens[done..done + live].to_vec();
            let pad = *chunk.last().expect("non-empty chunk");
            chunk.resize(m, pad);
            let chunk_end = done + live;
            let mut rows: Vec<ChunkRowSplice> = Vec::new();
            for (si, sp) in splices.iter().enumerate() {
                let n_slots = sp.rows_bf16.len() / hidden;
                let lo = sp.position.max(done);
                let hi = (sp.position + n_slots).min(chunk_end);
                for abs in lo..hi {
                    let w0 = (abs - sp.position) * hidden_words;
                    rows.push(ChunkRowSplice {
                        rel_pos: abs - done,
                        row_words: &packed[si][w0..w0 + hidden_words],
                    });
                }
            }
            self.prefill_chunk_masked(&chunk, live, &rows)?;
            done += live;
        }
        Ok(done)
    }

    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        let m = self.prefill_chunk_len();
        if m == 0 {
            return Ok(0);
        }
        let mut done = 0usize;
        while tokens.len() - done >= m && self.pos + m <= self.max_seq {
            self.prefill_chunk(&tokens[done..done + m])?;
            done += m;
        }
        Ok(done)
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        anyhow::ensure!(!tokens.is_empty(), "prefill needs at least one token");
        let (last, rest) = tokens.split_last().expect("non-empty");
        for t in rest {
            self.prefill_step(*t)?;
        }
        self.decode_step(*last)
    }
}

fn run_pass_list(ctx: &'static WgpuContext, passes: &[Pass]) -> Result<()> {
    if dispatch::profile::enabled() && ctx.caps.timestamp_query {
        let raw: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))> = passes
            .iter()
            .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
            .collect();
        let labels: Vec<String> = passes.iter().map(|p| p.label.clone()).collect();
        dispatch::submit_profiled_slices(ctx, &raw, &labels)
            .map_err(|e| anyhow::anyhow!("profiled submit: {e}"))?;
        return Ok(());
    }
    dispatch::submit_pass_list(
        ctx,
        passes.iter().map(|p| (p.pipeline.as_ref(), &p.bind, p.grid)),
    );
    Ok(())
}

fn square_up_the_2d_tiling_a_65535_wide_row_leaves_65533_workgroups_idle_at_one_shard(
    ctx: &WgpuContext,
    groups: u64,
) -> (u32, u32, u32) {
    let limit = ctx.caps.max_compute_workgroups_per_dimension.max(1) as u64;
    if groups <= limit {
        return (groups as u32, 1, 1);
    }
    let y = groups.div_ceil(limit);
    let x = groups.div_ceil(y);
    assert!(
        x * y >= groups && x <= limit,
        "a {x} x {y} tiling must cover all {groups} workgroups and keep x within the device's \
         {limit} per-dimension limit, or the lm-head silently drops rows"
    );
    (x as u32, y as u32, 1)
}

const EMBED_SHARD_CEILING_BYTES_IS_A_HARD_1_GIB_NOT_A_DEVICE_LIMIT_AND_SPLITS_A_1_476_GB_TABLE_IN_TWO:
    u64 = 1u64 << 30;

const EMBED_SHARD_CEILING_BYTES: u64 =
    EMBED_SHARD_CEILING_BYTES_IS_A_HARD_1_GIB_NOT_A_DEVICE_LIMIT_AND_SPLITS_A_1_476_GB_TABLE_IN_TWO;

fn row_chunk(ctx: &WgpuContext, hidden: usize) -> usize {
    if let Some(rows) = std::env::var("NV_G4MOE_ROW_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 2)
    {
        return rows & !1usize;
    }
    let ceiling = if head_one_shard_enabled() {
        ctx.caps.max_buffer_size.max(EMBED_SHARD_CEILING_BYTES)
    } else {
        EMBED_SHARD_CEILING_BYTES
    };
    let limit = ctx
        .caps
        .max_storage_buffer_binding_size
        .clamp(1 << 20, ceiling);
    let per_row = (hidden * 2) as u64;
    ((limit / per_row) as usize).max(2) & !1usize
}

type RopeTableCache = std::collections::HashMap<(usize, u32, u32), (wgpu::Buffer, wgpu::Buffer)>;

struct TailFoldDeferredUntilNextLayersInputNormWeightIsUploaded {
    moe_out: wgpu::Buffer,
    post_ffw2_w: wgpu::Buffer,
    r_out: wgpu::Buffer,
    r_in: wgpu::Buffer,
    post_ffw_w: wgpu::Buffer,
    h1: wgpu::Buffer,
    layer_scalar: f32,
}

struct AttnGpu {
    wq: Bf16Gpu,
    wk: Bf16Gpu,
    wv: Option<Bf16Gpu>,
    wo: Bf16Gpu,
    cosb: wgpu::Buffer,
    sinb: wgpu::Buffer,
    qn: wgpu::Buffer,
    kn: wgpu::Buffer,
    vn: wgpu::Buffer,
    kc: wgpu::Buffer,
    vc: wgpu::Buffer,
    kv_scales: Option<(wgpu::Buffer, wgpu::Buffer)>,
    hd: usize,
    n_q: usize,
    n_kv: usize,
    window: usize,
    ring: u32,
    eps: f32,
}

struct MlpGpu {
    wg: Bf16Gpu,
    wu: Bf16Gpu,
    wd: Bf16Gpu,
}

struct MoeGpu {
    ones_w: wgpu::Buffer,
    rsw: wgpu::Buffer,
    router: Bf16Gpu,
    pes: wgpu::Buffer,
    eg: ExpertGpu,
    eu: ExpertGpu,
    ed: ExpertGpu,
}

struct RouterPreUploadedBeforeTheMlpSoTheDownDispatchCanCarryTheRouterGemv {
    rsw: wgpu::Buffer,
    router: Bf16Gpu,
    pes: wgpu::Buffer,
    rlogits: wgpu::Buffer,
    ids: wgpu::Buffer,
    wts: wgpu::Buffer,
}

fn upload_router_pre(
    b: &mut Builder,
    cfg: &Gemma4MoeConfig,
    layer: &HostLayer,
) -> RouterPreUploadedBeforeTheMlpSoTheDownDispatchCanCarryTheRouterGemv {
    RouterPreUploadedBeforeTheMlpSoTheDownDispatchCanCarryTheRouterGemv {
        rsw: b.upload_u32("g4m-moe-rscale", &pack_pairs(&layer.router_scale)),
        router: upload_bf16(b, "g4m-moe-router", &layer.router),
        pes: b.upload_f32("g4m-moe-pes", &layer.per_expert_scale),
        rlogits: b.zeros("g4m-moe-rlogits", (cfg.num_experts * 4) as u64),
        ids: b.zeros("g4m-moe-ids", (cfg.top_k_experts * 4) as u64),
        wts: b.zeros("g4m-moe-wts", (cfg.top_k_experts * 4) as u64),
    }
}

fn push_gemv_bf16_down_rpnorm(
    b: &mut Builder,
    s: &Sources,
    cfg: &Gemma4MoeConfig,
    wd: &Bf16Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    pre: &RouterPreUploadedBeforeTheMlpSoTheDownDispatchCanCarryTheRouterGemv,
    x_router: &wgpu::Buffer,
) -> Result<()> {
    let hidden = cfg.base.hidden_size;
    let hidden_words = hidden / 2;
    let eps = cfg.base.rms_norm_eps as f32;
    let n_e = cfg.num_experts;
    anyhow::ensure!(
        !bf16_r8_enabled() && !bf16_acc4_enabled(),
        "NV_G4MOE_RPFUSE=1 copies the sglat mlp-down body verbatim; the r8/acc4 down \
         entries have no fused twin"
    );
    let wide = b.wide_flag("g4m-mlp-down", GemvLoad::Wide, wd.k, 0, &wd.w, x);
    let dn_grid = b.grid1(wd.n.div_ceil(2) as u64, 1);
    anyhow::ensure!(
        dn_grid.1 == 1 && dn_grid.2 == 1,
        "NV_G4MOE_RPFUSE=1 appends the router workgroups on grid x, so the down grid must \
         stay one-dimensional; got {dn_grid:?}"
    );
    let rp_groups = n_e.div_ceil(2) as u32;
    let dn_p = b.uni(
        "g4m-gemvb-p",
        GemvBf16Params {
            n_rows: wd.n as u32,
            k_words: (wd.k / 2) as u32,
            groups_x: dn_grid.0,
            out_f32: 0,
            w_row_words: (wd.k / 2) as u32,
            x_off_words: 0,
            y_off_words: 0,
            wide,
            alpha: 1.0,
            ..Default::default()
        },
    );
    let rp_p = b.uni(
        "g4m-gemvb-p",
        GemvBf16Params {
            n_rows: n_e as u32,
            k_words: hidden_words as u32,
            groups_x: rp_groups,
            out_f32: 1,
            w_row_words: hidden_words as u32,
            x_off_words: 0,
            y_off_words: 0,
            wide: 0,
            alpha: 1.0 / (hidden as f32).sqrt(),
            pad1: hidden as u32,
            pad2: eps.to_bits(),
            ..Default::default()
        },
    );
    b.push(
        "g4m-mlp-down-rp",
        &s.gemv_bf16_dnrp_sglat,
        "g4m_gemv_bf16_down_rpnorm_sglat",
        &[
            (0, &wd.w),
            (1, x),
            (2, &dn_p),
            (3, y),
            (4, &pre.router.w),
            (5, &pre.rsw),
            (6, x_router),
            (7, &rp_p),
            (8, &pre.rlogits),
        ],
        (dn_grid.0 + rp_groups, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_attn(
    b: &mut Builder,
    s: &Sources,
    cfg: &Gemma4MoeConfig,
    layer: &HostLayer,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    pos_buf: &wgpu::Buffer,
    max_seq: usize,
    sliding_ring_rows: Option<usize>,
    flash_fd: Option<&wgpu::Buffer>,
    flash_sl: Option<&wgpu::Buffer>,
    rope_tables_shared_across_layers_because_per_layer_copies_cost_gib_at_196k: &mut RopeTableCache,
    states: &mut Vec<(wgpu::Buffer, u64)>,
    tail_fold: Option<(
        &TailFoldDeferredUntilNextLayersInputNormWeightIsUploaded,
        &wgpu::Buffer,
    )>,
) -> Result<AttnGpu> {
    let base = &cfg.base;
    let kind = layer.kind;
    let hd = base.head_dim_for(kind);
    let n_q = base.num_attention_heads;
    let n_kv = base.num_kv_heads_for(kind);
    let theta = base.rope_theta_for(kind);
    let partial = base.rope_partial_factor_for(kind);
    let window = match kind {
        LayerType::SlidingAttention => base.sliding_window,
        LayerType::FullAttention => 0,
    };
    let eps = base.rms_norm_eps as f32;

    let fp8 = kv_fp8_enabled();
    let grouped_attn = grouped_gemv_active(b.ctx) && !fp8;
    anyhow::ensure!(
        tail_fold.is_none() || grouped_attn,
        "NV_G4MOE_TAILFOLD=1 folds the deferred tailnorm into the grouped qkvproj gemv; \
         dropping the bundle on the non-grouped path would skip the tailnorm entirely"
    );
    let attn_i8_qkv = attn_i8_site_enabled("g4m-at-qkvproj");
    let attn_i8_o = attn_i8_site_enabled("g4m-at-oproj");
    let attn_i8 = attn_i8_qkv || attn_i8_o;
    anyhow::ensure!(
        !attn_i8 || grouped_attn,
        "NV_G4MOE_ATTN_I8=1 quantizes the stacked qkvproj and the oproj decode gemvs; the \
         non-grouped q/k/v split path has no int8 route -- set NV_G4MOE_GROUPED_GEMV=1"
    );
    anyhow::ensure!(
        tail_fold.is_none() || !attn_i8,
        "NV_G4MOE_ATTN_I8=1 has no int8 tailfold qkvproj entry and NV_G4MOE_TAILFOLD is a \
         banked negative -- unset one of the two"
    );
    let (ql, kl, vl) = if attn_i8_qkv {
        (
            "g4m-at-qw-prefill-only",
            "g4m-at-kw-prefill-only",
            "g4m-at-vw-prefill-only",
        )
    } else {
        ("g4m-at-qw", "g4m-at-kw", "g4m-at-vw")
    };
    let ol = if attn_i8_o {
        "g4m-at-ow-prefill-only"
    } else {
        "g4m-at-ow"
    };

    let wq = upload_bf16(b, ql, &layer.q);
    let wk_ = upload_bf16(b, kl, &layer.k);
    let wo = upload_bf16(b, ol, &layer.o);
    let wv = layer.v.as_ref().map(|vlin| upload_bf16(b, vl, vlin));

    let (q_raw, k_raw, v_raw) = if grouped_attn {
        let zero = b.zeros("g4m-at-qraw", 16);
        (zero.clone(), zero.clone(), zero)
    } else {
        let q_raw = b.zeros("g4m-at-qraw", (wq.n * 2) as u64);
        let k_raw = b.zeros("g4m-at-kraw", (wk_.n * 2) as u64);
        push_gemv_bf16(b, s, "g4m-at-qproj", &wq, x, &q_raw, false, 0)?;
        push_gemv_bf16(b, s, "g4m-at-kproj", &wk_, x, &k_raw, false, 0)?;
        let v_raw = if let Some(wv) = &wv {
            let v_raw = b.zeros("g4m-at-vraw", (wv.n * 2) as u64);
            push_gemv_bf16(b, s, "g4m-at-vproj", wv, x, &v_raw, false, 0)?;
            v_raw
        } else {
            k_raw.clone()
        };
        (q_raw, k_raw, v_raw)
    };

    let rope_key = (hd, theta.to_bits(), partial.to_bits());
    let (cosb, sinb) = match rope_tables_shared_across_layers_because_per_layer_copies_cost_gib_at_196k
        .get(&rope_key)
    {
        Some((c, sn)) => (c.clone(), sn.clone()),
        None => {
            let (cos, sin) = rope_tables(hd, theta, partial, max_seq);
            let c = b.upload_f32("g4m-at-cos", &cos);
            let sn = b.upload_f32("g4m-at-sin", &sin);
            rope_tables_shared_across_layers_because_per_layer_copies_cost_gib_at_196k
                .insert(rope_key, (c.clone(), sn.clone()));
            (c, sn)
        }
    };
    let qn = b.upload_u32("g4m-at-qn", &pack_pairs(&layer.q_norm));
    let kn = b.upload_u32("g4m-at-kn", &pack_pairs(&layer.k_norm));
    let ones: Vec<u16> = vec![bf16_bits(1.0); hd];
    let vn = b.upload_u32("g4m-at-vn", &pack_pairs(&ones));

    let ring_rows = match (kind, sliding_ring_rows) {
        (LayerType::SlidingAttention, Some(rows)) => Some(rows),
        _ => None,
    };
    let ring = ring_rows.map(|r| r as u32).unwrap_or(0);
    let cache_rows = ring_rows.unwrap_or(max_seq);
    let fold_kv = fold_chain_enabled() && grouped_attn;
    let use_flash = match kind {
        LayerType::FullAttention => flash_decode_enabled(),
        LayerType::SlidingAttention => flash_sliding_enabled(),
    };
    let qkvfold = qkvfold_enabled() && fold_kv && use_flash;
    let mut caches_written_by_the_fused_qkvnorm: Option<(wgpu::Buffer, wgpu::Buffer)> = None;
    let mut qkvnorm_deferred_into_flash_stage1: Option<(
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
        wgpu::Buffer,
    )> = None;

    let (q, k, v) = if grouped_attn {
        let has_v = wv.is_some();
        let n_stack = wq.n + wk_.n + wv.as_ref().map(|w| w.n).unwrap_or(0);
        let mut stacked: Vec<u16> =
            Vec::with_capacity(layer.q.w.len() + layer.k.w.len());
        stacked.extend_from_slice(&layer.q.w);
        stacked.extend_from_slice(&layer.k.w);
        if let Some(vlin) = &layer.v {
            stacked.extend_from_slice(&vlin.w);
        }
        let wqkv = Bf16Gpu {
            w: b.upload_u32("g4m-at-qkvw", &pack_pairs(&stacked)),
            n: n_stack,
            k: wq.k,
        };
        let qkv_i8 = attn_i8_qkv
            .then(|| upload_attn_i8(b, "g4m-at-qkv-i8", "g4m-at-qkv-i8-sf", &stacked, n_stack, wq.k));
        drop(stacked);
        let qkv_raw = b.zeros("g4m-at-qkvraw", (n_stack * 2) as u64);
        match tail_fold {
            Some((pt, next_w)) => {
                let wide = b.wide_flag("g4m-at-qkvproj", GemvLoad::Wide, wqkv.k, 0, &wqkv.w, x);
                anyhow::ensure!(
                    wide == 1,
                    "NV_G4MOE_TAILFOLD=1 stages the folded normnext vector for the vec4 wide \
                     dot, so the qkvproj gemv must qualify for wide loads"
                );
                let grid = b.grid1(n_stack.div_ceil(2) as u64, 1);
                let gp = b.uni(
                    "g4m-gemvb-p",
                    GemvBf16Params {
                        n_rows: n_stack as u32,
                        k_words: (wqkv.k / 2) as u32,
                        groups_x: grid.0,
                        out_f32: 0,
                        w_row_words: (wqkv.k / 2) as u32,
                        x_off_words: 0,
                        y_off_words: 0,
                        wide,
                        alpha: 1.0,
                        ..Default::default()
                    },
                );
                let np = b.uni(
                    "g4m-pn-p",
                    NormParams {
                        hidden: wqkv.k as u32,
                        words: (wqkv.k / 2) as u32,
                        eps,
                        scale: pt.layer_scalar,
                    },
                );
                b.push(
                    "g4m-at-qkvproj",
                    &s.gemv_bf16_tailfold,
                    "g4m_gemv_bf16_tailfold",
                    &[
                        (0, &wqkv.w),
                        (1, &gp),
                        (2, &qkv_raw),
                        (3, &np),
                        (4, &pt.moe_out),
                        (5, &pt.post_ffw2_w),
                        (6, &pt.r_out),
                        (7, &pt.r_in),
                        (8, &pt.post_ffw_w),
                        (9, &pt.h1),
                        (10, x),
                        (11, next_w),
                    ],
                    grid,
                )?;
            }
            None => match &qkv_i8 {
                Some(qi8) => push_gemv_attn_i8(b, s, "g4m-at-qkvproj", qi8, x, &qkv_raw)?,
                None => push_gemv_bf16(b, s, "g4m-at-qkvproj", &wqkv, x, &qkv_raw, false, 0)?,
            },
        }

        let mut wcat: Vec<u16> = Vec::with_capacity(3 * hd);
        wcat.extend_from_slice(&layer.q_norm);
        wcat.extend_from_slice(&layer.k_norm);
        wcat.extend_from_slice(&ones);
        let wcat_b = b.upload_u32("g4m-at-qkvn", &pack_pairs(&wcat));
        let qkv = b.zeros("g4m-at-qkv", ((n_q + 2 * n_kv) * hd * 2) as u64);
        let p = b.uni(
            "g4m-at-nr-p",
            NormRopeParams {
                n_rows: (n_q + 2 * n_kv) as u32,
                head_dim: hd as u32,
                src_stride: hd as u32,
                rot_half: (hd / 2) as u32,
                has_rope: u32::from(has_v),
                n_q: n_q as u32,
                n_kv: n_kv as u32,
                eps,
            },
        );
        if fold_kv {
            let kv_words = n_kv * hd / 2;
            let cache_bytes = (cache_rows * kv_words * 4) as u64;
            let kc = b.zeros("g4m-at-kc", cache_bytes);
            let vc = b.zeros("g4m-at-vc", cache_bytes);
            states.push((kc.clone(), cache_bytes));
            states.push((vc.clone(), cache_bytes));
            let kvp = b.uni(
                "g4m-at-kv-p",
                KvWriteParams {
                    words: kv_words as u32,
                    ring,
                    k_off_words: 0,
                    ..Default::default()
                },
            );
            if qkvfold {
                qkvnorm_deferred_into_flash_stage1 =
                    Some((qkv_raw.clone(), wcat_b.clone(), p.clone(), kvp.clone()));
            } else {
                let (src, entry) = if norm_lat_active(b.ctx) {
                    (&s.attn_sglat, "g4m_attn_norm_rope_qkv_kvwrite_sglat")
                } else {
                    (&s.attn, "g4m_attn_norm_rope_qkv_kvwrite")
                };
                b.push(
                    "g4m-at-qkvnorm",
                    src,
                    entry,
                    &[
                        (0, &qkv_raw),
                        (1, &wcat_b),
                        (2, &cosb),
                        (3, &sinb),
                        (4, pos_buf),
                        (5, &qkv),
                        (6, &p),
                        (7, &kc),
                        (8, &vc),
                        (9, &kvp),
                    ],
                    ((n_q + 2 * n_kv) as u32, 1, 1),
                )?;
            }
            caches_written_by_the_fused_qkvnorm = Some((kc, vc));
        } else {
            b.push(
                "g4m-at-qkvnorm",
                &s.attn,
                "g4m_attn_norm_rope_qkv",
                &[
                    (0, &qkv_raw),
                    (1, &wcat_b),
                    (2, &cosb),
                    (3, &sinb),
                    (4, pos_buf),
                    (5, &qkv),
                    (6, &p),
                ],
                ((n_q + 2 * n_kv) as u32, 1, 1),
            )?;
        }
        (qkv.clone(), qkv.clone(), qkv)
    } else {
        let q = b.zeros("g4m-at-q", (n_q * hd * 2) as u64);
        let k = b.zeros("g4m-at-k", (n_kv * hd * 2) as u64);
        let v = b.zeros("g4m-at-v", (n_kv * hd * 2) as u64);
        (q, k, v)
    };

    let norm_rope = |label: &str,
                     src: &wgpu::Buffer,
                     w: &wgpu::Buffer,
                     dst: &wgpu::Buffer,
                     rows: usize,
                     rope: bool,
                     b: &mut Builder|
     -> Result<()> {
        let p = b.uni(
            "g4m-at-nr-p",
            NormRopeParams {
                n_rows: rows as u32,
                head_dim: hd as u32,
                src_stride: hd as u32,
                rot_half: (hd / 2) as u32,
                has_rope: u32::from(rope),
                eps,
                ..Default::default()
            },
        );
        b.push(
            label,
            &s.attn,
            "g4m_attn_norm_rope",
            &[
                (0, src),
                (1, w),
                (2, &cosb),
                (3, &sinb),
                (4, pos_buf),
                (5, dst),
                (6, &p),
            ],
            (rows as u32, 1, 1),
        )
    };
    if !grouped_attn {
        norm_rope("g4m-at-qnorm", &q_raw, &qn, &q, n_q, true, b)?;
        norm_rope("g4m-at-knorm", &k_raw, &kn, &k, n_kv, true, b)?;
        norm_rope("g4m-at-vnorm", &v_raw, &vn, &v, n_kv, false, b)?;
    }

    let (kc, vc, kv_scales) = if let Some((kc, vc)) = caches_written_by_the_fused_qkvnorm.take() {
        (kc, vc, None)
    } else if fp8 {
        anyhow::ensure!(
            hd.is_multiple_of(4),
            "NV_G4MOE_KV_FP8=1: quantize_kv_fp8 packs 4 e4m3 bytes per u32 word, so head_dim \
             must be a multiple of 4; got {hd}"
        );
        let cache_bytes = (cache_rows * n_kv * hd) as u64;
        let scale_bytes = (cache_rows * n_kv * 4) as u64;
        let kc = b.zeros("g4m-at-kc", cache_bytes);
        let vc = b.zeros("g4m-at-vc", cache_bytes);
        let ksc = b.zeros("g4m-at-ksc", scale_bytes);
        let vsc = b.zeros("g4m-at-vsc", scale_bytes);
        states.push((kc.clone(), cache_bytes));
        states.push((vc.clone(), cache_bytes));
        states.push((ksc.clone(), scale_bytes));
        states.push((vsc.clone(), scale_bytes));
        let qp = b.uni(
            "g4m-at-kvq-p",
            KvFp8ParamsMirroringNvKernelsKvFp8UniformLayout {
                n_tokens: 1,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                ring,
                pairs: n_kv as u32,
                start: 0,
                slots: cache_rows as u32,
                reserved: 0,
            },
        );
        for (label, src_buf, cache, scales) in [
            ("g4m-at-kvq-k", &k, &kc, &ksc),
            ("g4m-at-kvq-v", &v, &vc, &vsc),
        ] {
            b.push(
                label,
                &s.kv_fp8,
                "quantize_kv_fp8",
                &[(0, src_buf), (1, cache), (2, scales), (3, pos_buf), (4, &qp)],
                (n_kv as u32, 1, 1),
            )?;
        }
        (kc, vc, Some((ksc, vsc)))
    } else {
        let kv_words = n_kv * hd / 2;
        let cache_bytes = (cache_rows * kv_words * 4) as u64;
        let kc = b.zeros("g4m-at-kc", cache_bytes);
        let vc = b.zeros("g4m-at-vc", cache_bytes);
        states.push((kc.clone(), cache_bytes));
        states.push((vc.clone(), cache_bytes));
        let kvp = b.uni(
            "g4m-at-kv-p",
            KvWriteParams {
                words: kv_words as u32,
                ring,
                k_off_words: if grouped_attn { (n_q * hd / 2) as u32 } else { 0 },
                ..Default::default()
            },
        );
        let grid = b.grid1(kv_words as u64, 64);
        if grouped_attn {
            b.push(
                "g4m-at-kvwrite",
                &s.attn,
                "g4m_kv_write_stacked",
                &[(10, &k), (12, &kc), (13, &vc), (14, pos_buf), (15, &kvp)],
                grid,
            )?;
        } else {
            b.push(
                "g4m-at-kvwrite",
                &s.attn,
                "g4m_kv_write",
                &[
                    (10, &k),
                    (11, &v),
                    (12, &kc),
                    (13, &vc),
                    (14, pos_buf),
                    (15, &kvp),
                ],
                grid,
            )?;
        }
        (kc, vc, None)
    };

    let kind_suffix = if layer_kind_decode_labels_enabled() {
        match kind {
            LayerType::SlidingAttention => "-sliding",
            LayerType::FullAttention => "-full",
        }
    } else {
        ""
    };
    let attn_bf16 = b.zeros("g4m-at-obf16", (n_q * hd * 2) as u64);
    if use_flash {
        let fp = match kind {
            LayerType::FullAttention => flash_fd.expect(
                "flash_decode_enabled() was true for this layer but from_loader allocated no \
                 shared flash uniform; the gate must be read once, before the layer loop",
            ),
            LayerType::SlidingAttention => flash_sl.expect(
                "flash_sliding_enabled() was true for this layer but from_loader allocated no \
                 sliding flash uniform; the gate must be read once, before the layer loop",
            ),
        };
        anyhow::ensure!(
            hd.is_multiple_of(2) && hd <= MAX_HEAD_DIM,
            "flash stage1 guards every 32-lane strip with d < head_dim and stage2 packs \
             bf16 pairs, so head_dim must be even and <= {MAX_HEAD_DIM}, got {hd}"
        );
        anyhow::ensure!(
            kind == LayerType::SlidingAttention || (window == 0 && ring == 0),
            "the shared flash uniform carries start=0 and ring=0, which is only correct for \
             full-attention layers (window {window}, ring {ring})"
        );
        let flash_wg_bytes =
            flash_workgroup_bytes_qsh512_sm8_sl8_plus_red256_only_when_wb_plus_sacc_8_warps_by_chunk(
                flash_sg_reduce_enabled(),
                flash_sacc_chunk_head_dim_slice_staged_per_warp_read_once_at_build(),
                flash_gqa_fold_query_heads_per_workgroup_read_once_at_build(),
            );
        anyhow::ensure!(
            b.ctx.caps.max_compute_invocations_per_workgroup >= 256
                && b.ctx.caps.max_compute_workgroup_size_x >= 256
                && b.ctx.caps.workgroup_storage_fits(flash_wg_bytes),
            "NV_G4MOE_FLASH_DECODE=1 needs a 256-invocation workgroup and {flash_wg_bytes} \
             workgroup bytes; device allows {} invocations / {} bytes -- unset the env to keep \
             the serial arm",
            b.ctx.caps.max_compute_invocations_per_workgroup,
            b.ctx.caps.max_compute_workgroup_storage_size
        );
        let splits =
            flash_splits_for_layer_kind_full_attention_holds_the_folded_grid_sliding_holds_the_stage2_scan(
                kind,
                flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
            );
        let scratch_splits =
            flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(splits);
        let scratch_splits = match kind {
            LayerType::FullAttention => {
                foldgate_scratch_splits_full_attention_only_must_also_cover_the_deep_arm_rows(
                    scratch_splits,
                )
            }
            LayerType::SlidingAttention => scratch_splits,
        };
        let scratch_bytes = (n_q * scratch_splits as usize * (hd + 2) * 4) as u64;
        let scratch = b.zeros("g4m-at-fscr", scratch_bytes);
        match kind {
            LayerType::FullAttention => b
                .flash_scratch_bytes_per_full_attention_layer_sized_at_its_own_split_count
                .push(scratch_bytes),
            LayerType::SlidingAttention => b
                .flash_scratch_bytes_per_sliding_layer_sized_at_its_own_split_count_not_the_max_of_the_two
                .push(scratch_bytes),
        }
        if !b.to_prefill {
            let flash1_pass_index = b.passes.len();
            match kind {
                LayerType::FullAttention => b
                    .depthgate_flash1_pass_indices_full_attention_the_band_is_keyed_on_the_whole_context
                    .push(flash1_pass_index),
                LayerType::SlidingAttention => b
                    .depthgate_flash1_pass_indices_sliding_the_window_caps_the_scan_so_the_band_is_keyed_on_min_total_window
                    .push(flash1_pass_index),
            }
        }
        if let Some((qkv_raw_b, wcat_b, nr_p, kvp)) = &qkvnorm_deferred_into_flash_stage1 {
            anyhow::ensure!(
                b.ctx
                    .caps
                    .workgroup_storage_fits(FLASH_QKVFOLD_WORKGROUP_BYTES_ADDS_BUF512_KCUR256_VCUR256_U32),
                "NV_G4MOE_QKVFOLD=1 stages the recomputed q/k/v head rows in {} workgroup \
                 bytes; device allows {} -- unset the env to keep the standalone qkvnorm",
                FLASH_QKVFOLD_WORKGROUP_BYTES_ADDS_BUF512_KCUR256_VCUR256_U32,
                b.ctx.caps.max_compute_workgroup_storage_size
            );
            b.push(
                &format!("g4m-at-flash1{kind_suffix}"),
                &s.flash_qkvfold,
                "g4m_flash_stage1_bf16_qkvfold",
                &[
                    (0, qkv_raw_b),
                    (1, &kc),
                    (2, &vc),
                    (3, &scratch),
                    (4, wcat_b),
                    (5, &cosb),
                    (6, fp),
                    (7, &sinb),
                    (8, nr_p),
                    (9, kvp),
                ],
                (n_q as u32, splits, 1),
            )?;
        } else {
            let mut flash1_binds: Vec<(u32, &wgpu::Buffer)> =
                vec![(0, &q), (1, &kc), (2, &vc), (3, &scratch), (6, fp)];
            let fold =
                flash_gqa_fold_for_layer_kind_only_full_attention_scans_enough_rows_to_amortise_a_halved_grid(
                    kind,
                    flash_gqa_fold_query_heads_per_workgroup_read_once_at_build(),
                );
            let folded = kv_scales.is_none()
                && s.flash_fold_entry.is_some()
                && flash_gqa_fold_fits_this_layer(fold, n_q, n_kv);
            let flash1_entry = match &kv_scales {
                Some((ksc, vsc)) => {
                    flash1_binds.push((7, ksc));
                    flash1_binds.push((8, vsc));
                    "g4m_flash_stage1_fp8"
                }
                None if folded => s
                    .flash_fold_entry
                    .as_deref()
                    .expect("folded implies a generated fold entry"),
                None => "g4m_flash_stage1_bf16",
            };
            let flash1_grid_x = if folded {
                (n_q / fold as usize) as u32
            } else {
                n_q as u32
            };
            b.push(
                &format!("g4m-at-flash1{kind_suffix}"),
                &s.flash,
                flash1_entry,
                &flash1_binds,
                (flash1_grid_x, splits, 1),
            )?;
            if kind == LayerType::FullAttention && !b.to_prefill {
                if let Some((fg_src, fg_entry)) = &s.flash_foldgate_deep_source_and_entry {
                    let fg_fold =
                        FOLDGATE_DEEP_FOLD_2_QUERY_HEADS_PER_WORKGROUP_HALVES_FULL_ATTENTION_KV_TRAFFIC;
                    anyhow::ensure!(
                        kv_scales.is_none(),
                        "NV_G4MOE_FLASH_FOLDGATE=1 has no fp8 fold-2 stage1 entry; unset it or \
                         NV_G4MOE_KV_FP8. {}",
                        FOLDGATE_SWAPS_ONLY_FULL_ATTENTION_BECAUSE_SPELLING_THE_FOLD_THROUGH_THE_SHARED_DEPTHGATE_SPLITS_ENV_TAXES_SLIDING
                    );
                    anyhow::ensure!(
                        flash_gqa_fold_fits_this_layer(fg_fold, n_q, n_kv),
                        "NV_G4MOE_FLASH_FOLDGATE=1 folds {fg_fold} query heads of one KV group \
                         per workgroup but this full-attention layer has n_q {n_q} / n_kv {n_kv}, \
                         which the fold does not divide -- unset the env"
                    );
                    let alt = b.built_pass(
                        &format!("g4m-at-flash1{kind_suffix}-deep"),
                        fg_src,
                        fg_entry,
                        &flash1_binds,
                        (
                            n_q as u32 / fg_fold,
                            FOLDGATE_DEEP_SPLITS_64_RESTORES_512_WORKGROUPS_UNDER_FOLD_2_AND_IS_A_MEASURED_RUNG_THE_LADDER_HAS_A_REPRODUCED_POTHOLE_AT_40_47_48_SO_NEVER_INTERPOLATE,
                            1,
                        ),
                    )?;
                    b.foldgate_alt_flash1_passes_one_deep_twin_per_full_attention_flash1_in_the_same_order
                        .push(alt);
                }
            }
        }
        b.push(
            &format!("g4m-at-flash2{kind_suffix}"),
            &s.flash,
            "g4m_flash_stage2_pk",
            &[(3, &scratch), (4, &attn_bf16), (6, fp)],
            (n_q as u32, 1, 1),
        )?;
    } else {
        let scores = b.zeros("g4m-at-scores", (n_q * max_seq * 4) as u64);
        let attn_f32 = b.zeros("g4m-at-of32", (n_q * hd * 4) as u64);
        let adp = b.uni(
            "g4m-at-dec-p",
            AttnDecodeParams {
                n_heads: n_q as u32,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                max_seq: max_seq as u32,
                group: (n_q / n_kv) as u32,
                window: window as u32,
                ring,
                scale: 1.0,
            },
        );
        let mut decode_binds: Vec<(u32, &wgpu::Buffer)> = vec![
            (20, &q),
            (21, &kc),
            (22, &vc),
            (23, &scores),
            (24, &attn_f32),
            (25, pos_buf),
            (26, &adp),
        ];
        let decode_entry = match &kv_scales {
            Some((ksc, vsc)) => {
                decode_binds.push((27, ksc));
                decode_binds.push((28, vsc));
                "g4m_attn_decode_fp8"
            }
            None => "g4m_attn_decode",
        };
        b.push(
            &format!("g4m-at-decode{kind_suffix}"),
            &s.attn,
            decode_entry,
            &decode_binds,
            (n_q as u32, 1, 1),
        )?;

        let pp = b.uni(
            "g4m-pack-p",
            WordsParams {
                n_words: (n_q * hd / 2) as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1((n_q * hd / 2) as u64, 64);
        b.push(
            "g4m-at-pack",
            &s.moe,
            "g4m_pack_f32",
            &[(50, &attn_f32), (51, &attn_bf16), (52, &pp)],
            grid,
        )?;
    }

    if attn_i8_o {
        let o_i8 = upload_attn_i8(b, "g4m-at-o-i8", "g4m-at-o-i8-sf", &layer.o.w, wo.n, wo.k);
        push_gemv_attn_i8(b, s, "g4m-at-oproj", &o_i8, &attn_bf16, out)?;
    } else {
        push_gemv_bf16(b, s, "g4m-at-oproj", &wo, &attn_bf16, out, false, 0)?;
    }

    Ok(AttnGpu {
        wq,
        wk: wk_,
        wv,
        wo,
        cosb,
        sinb,
        qn,
        kn,
        vn,
        kc,
        vc,
        kv_scales,
        hd,
        n_q,
        n_kv,
        window,
        ring,
        eps,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_attn_prefill(
    b: &mut Builder,
    s: &Sources,
    a: &AttnGpu,
    sc: &PfScratch,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    max_seq: usize,
) -> Result<()> {
    let m = sc.m;
    let hd = a.hd;
    let n_q = a.n_q;
    let n_kv = a.n_kv;
    let hidden_words = a.wq.k / 2;

    pf_gemm_bf16(
        b,
        s,
        "g4m-pf-qproj",
        &a.wq,
        x,
        hidden_words,
        &sc.q_raw,
        n_q * hd / 2,
        false,
        1.0,
    )?;
    pf_gemm_bf16(
        b,
        s,
        "g4m-pf-kproj",
        &a.wk,
        x,
        hidden_words,
        &sc.k_raw,
        n_kv * hd / 2,
        false,
        1.0,
    )?;
    let v_raw = match &a.wv {
        Some(wv) => {
            pf_gemm_bf16(
                b,
                s,
                "g4m-pf-vproj",
                wv,
                x,
                hidden_words,
                &sc.v_raw,
                n_kv * hd / 2,
                false,
                1.0,
            )?;
            sc.v_raw.clone()
        }
        None => sc.k_raw.clone(),
    };

    let norm_rope = |label: &str,
                     src: &wgpu::Buffer,
                     w: &wgpu::Buffer,
                     dst: &wgpu::Buffer,
                     rows: usize,
                     rope: bool,
                     b: &mut Builder|
     -> Result<()> {
        let p = b.uni(
            "g4m-pf-nr-p",
            PfRopeParams {
                n_rows: rows as u32,
                head_dim: hd as u32,
                src_stride: hd as u32,
                rot_half: (hd / 2) as u32,
                has_rope: u32::from(rope),
                tok_src_stride: (rows * hd) as u32,
                tok_dst_stride: (rows * hd) as u32,
                eps: a.eps,
                ..Default::default()
            },
        );
        b.push(
            label,
            &s.prefill,
            "pm_attn_norm_rope",
            &[
                (30, src),
                (31, w),
                (32, &a.cosb),
                (33, &a.sinb),
                (34, &sc.pos),
                (35, dst),
                (36, &p),
            ],
            (rows as u32, m as u32, 1),
        )
    };
    norm_rope("g4m-pf-qnorm", &sc.q_raw, &a.qn, &sc.q, n_q, true, b)?;
    norm_rope("g4m-pf-knorm", &sc.k_raw, &a.kn, &sc.k, n_kv, true, b)?;
    norm_rope("g4m-pf-vnorm", &v_raw, &a.vn, &sc.v, n_kv, false, b)?;

    if let Some((ksc, vsc)) = &a.kv_scales {
        let qp = b.uni(
            "g4m-pf-kvq-p",
            KvFp8ParamsMirroringNvKernelsKvFp8UniformLayout {
                n_tokens: m as u32,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                ring: a.ring,
                pairs: (m * n_kv) as u32,
                start: 0,
                slots: 0,
                reserved: 0,
            },
        );
        for (label, src_buf, cache, scales) in [
            ("g4m-pf-kvq-k", &sc.k, &a.kc, ksc),
            ("g4m-pf-kvq-v", &sc.v, &a.vc, vsc),
        ] {
            b.push(
                label,
                &s.kv_fp8,
                "quantize_kv_fp8",
                &[
                    (0, src_buf),
                    (1, cache),
                    (2, scales),
                    (3, &sc.pos),
                    (4, &qp),
                ],
                ((m * n_kv) as u32, 1, 1),
            )?;
        }
    } else {
        let kv_words = n_kv * hd / 2;
        let kvp = b.uni(
            "g4m-pf-kv-p",
            PfKvParams {
                words: kv_words as u32,
                m: m as u32,
                ring: a.ring,
                ..Default::default()
            },
        );
        let kv_x = (kv_words as u32).div_ceil(64);
        b.push(
            "g4m-pf-kvwrite",
            &s.prefill,
            "pm_kv_write",
            &[
                (40, &sc.k),
                (41, &sc.v),
                (42, &a.kc),
                (43, &a.vc),
                (44, &sc.pos),
                (45, &kvp),
            ],
            (kv_x, m as u32, 1),
        )?;
    }

    let adp = b.uni(
        "g4m-pf-attn-p",
        PfAttnParams {
            n_heads: n_q as u32,
            n_kv: n_kv as u32,
            head_dim: hd as u32,
            max_seq: max_seq as u32,
            group: (n_q / n_kv) as u32,
            window: a.window as u32,
            m: m as u32,
            ring: a.ring,
            scale: 1.0,
            ..Default::default()
        },
    );
    let mut pf_attn_binds: Vec<(u32, &wgpu::Buffer)> = vec![
        (50, &sc.q),
        (51, &a.kc),
        (52, &a.vc),
        (53, &sc.scores),
        (54, &sc.attn_f32),
        (55, &sc.pos),
        (56, &adp),
    ];
    let pf_attn_entry = match &a.kv_scales {
        Some((ksc, vsc)) => {
            pf_attn_binds.push((57, ksc));
            pf_attn_binds.push((58, vsc));
            "pm_attn_fp8"
        }
        None => "pm_attn",
    };
    b.push(
        "g4m-pf-attn",
        &s.prefill,
        pf_attn_entry,
        &pf_attn_binds,
        (n_q as u32, m as u32, 1),
    )?;

    let words = m * n_q * hd / 2;
    let pp = b.uni(
        "g4m-pack-p",
        WordsParams {
            n_words: words as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1(words as u64, 64);
    b.push(
        "g4m-pf-pack",
        &s.moe,
        "g4m_pack_f32",
        &[(50, &sc.attn_f32), (51, &sc.attn_bf16), (52, &pp)],
        grid,
    )?;

    pf_gemm_bf16(
        b,
        s,
        "g4m-pf-oproj",
        &a.wo,
        &sc.attn_bf16,
        n_q * hd / 2,
        out,
        hidden_words,
        false,
        1.0,
    )
}

fn build_mlp(
    b: &mut Builder,
    s: &Sources,
    cfg: &Gemma4MoeConfig,
    layer: &HostLayer,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    rp: Option<(
        &RouterPreUploadedBeforeTheMlpSoTheDownDispatchCanCarryTheRouterGemv,
        &wgpu::Buffer,
    )>,
) -> Result<MlpGpu> {
    let inter = cfg.base.intermediate_size;
    let wg = upload_bf16(b, "g4m-mlp-gw", &layer.mlp_gate);
    let wu = upload_bf16(b, "g4m-mlp-uw", &layer.mlp_up);
    let wd = upload_bf16(b, "g4m-mlp-dw", &layer.mlp_down);
    let act = b.zeros("g4m-mlp-act", (inter * 2) as u64);
    if grouped_gemv_active(b.ctx) {
        push_gemv_bf16_gu_gelu(b, s, "g4m-mlp-gu", &wg, &wu, x, &act, "g4m-mlp-gate")?;
    } else {
        let g = b.zeros("g4m-mlp-g", (inter * 2) as u64);
        let u = b.zeros("g4m-mlp-u", (inter * 2) as u64);
        push_gemv_bf16(b, s, "g4m-mlp-gate", &wg, x, &g, false, 0)?;
        push_gemv_bf16(b, s, "g4m-mlp-up", &wu, x, &u, false, 0)?;
        let gp = b.uni(
            "g4m-gelu-p",
            WordsParams {
                n_words: (inter / 2) as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1((inter / 2) as u64, 64);
        b.push(
            "g4m-mlp-gelu",
            &s.moe,
            "g4m_gelu_mul",
            &[(10, &g), (11, &u), (12, &act), (13, &gp)],
            grid,
        )?;
    }
    match rp {
        Some((pre, x_router)) => {
            push_gemv_bf16_down_rpnorm(b, s, cfg, &wd, &act, out, pre, x_router)?
        }
        None => push_gemv_bf16(b, s, "g4m-mlp-down", &wd, &act, out, false, 0)?,
    }
    Ok(MlpGpu { wg, wu, wd })
}

fn build_mlp_prefill(
    b: &mut Builder,
    s: &Sources,
    mlp: &MlpGpu,
    sc: &PfScratch,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
) -> Result<()> {
    let m = sc.m;
    let inter = mlp.wg.n;
    let hidden_words = mlp.wg.k / 2;
    pf_gemm_bf16(
        b,
        s,
        "g4m-pf-mlp-gate",
        &mlp.wg,
        x,
        hidden_words,
        &sc.mlp_g,
        inter / 2,
        false,
        1.0,
    )?;
    pf_gemm_bf16(
        b,
        s,
        "g4m-pf-mlp-up",
        &mlp.wu,
        x,
        hidden_words,
        &sc.mlp_u,
        inter / 2,
        false,
        1.0,
    )?;
    let words = m * inter / 2;
    let gp = b.uni(
        "g4m-gelu-p",
        WordsParams {
            n_words: words as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1(words as u64, 64);
    b.push(
        "g4m-pf-mlp-gelu",
        &s.moe,
        "g4m_gelu_mul",
        &[
            (10, &sc.mlp_g),
            (11, &sc.mlp_u),
            (12, &sc.mlp_act),
            (13, &gp),
        ],
        grid,
    )?;
    pf_gemm_bf16(
        b,
        s,
        "g4m-pf-mlp-down",
        &mlp.wd,
        &sc.mlp_act,
        inter / 2,
        out,
        hidden_words,
        false,
        1.0,
    )
}

#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn build_moe(
    b: &mut Builder,
    s: &Sources,
    cfg: &Gemma4MoeConfig,
    layer: &HostLayer,
    x_router: &wgpu::Buffer,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    pre: Option<&RouterPreUploadedBeforeTheMlpSoTheDownDispatchCanCarryTheRouterGemv>,
    topk_prefused: bool,
) -> Result<MoeGpu> {
    let hidden = cfg.base.hidden_size;
    let hidden_words = hidden / 2;
    let inter = cfg.moe_intermediate_size;
    let n_e = cfg.num_experts;
    let k_top = cfg.top_k_experts;
    let eps = cfg.base.rms_norm_eps as f32;

    anyhow::ensure!(
        hidden.is_multiple_of(2 * W4_GS) && inter.is_multiple_of(2 * W4_GS),
        "w4 expert kernel needs hidden ({hidden}) and moe inter ({inter}) multiples of {}",
        2 * W4_GS
    );

    let grouped = grouped_gemv_active(b.ctx);
    let chain = router_chain_enabled();
    let rpnorm = rpnorm_active(b.ctx);
    anyhow::ensure!(
        !rpnorm || (grouped && !chain && hidden_words <= 2048),
        "NV_G4MOE_RPNORM=1 replays the grouped norm-mul inside every rproj workgroup and \
         stages {hidden_words} hidden words in a 2048-word workgroup array"
    );
    anyhow::ensure!(
        pre.is_none() || (grouped && rpnorm && !chain),
        "the pre-uploaded router bundle exists only for the NV_G4MOE_RPFUSE shape \
         (grouped + rpnorm, no router-chain)"
    );
    let ones: Vec<u16> = vec![bf16_bits(1.0); hidden];
    let ones_w = b.upload_u32("g4m-moe-ones", &pack_pairs(&ones));
    let router_in = b.zeros("g4m-moe-rin", (hidden_words * 4) as u64);
    let rsw = match pre {
        Some(p) => p.rsw.clone(),
        None => b.upload_u32("g4m-moe-rscale", &pack_pairs(&layer.router_scale)),
    };
    if chain {
        anyhow::ensure!(
            2 * n_e <= 256 && k_top <= 16 && hidden_words <= 2048 && hidden_words.is_multiple_of(2),
            "NV_G4MOE_ROUTER_CHAIN=1: the single-workgroup chain dots 2 lanes per expert row \
             ({n_e} experts), stages {hidden_words} hidden words, and holds {k_top} choices"
        );
    } else if rpnorm {
    } else if grouped {
        let p = b.uni(
            "g4m-pn-p",
            NormParams {
                hidden: hidden as u32,
                words: hidden_words as u32,
                eps,
                scale: 0.0,
            },
        );
        b.push(
            "g4m-moe-rnormmul",
            &s.prop,
            "g4m_norm_mul",
            &[(0, x_router), (1, &rsw), (2, &router_in), (3, &p)],
            (1, 1, 1),
        )?;
    } else {
        let rnormed = b.zeros("g4m-moe-rnorm", (hidden_words * 4) as u64);
        push_norm(
            b,
            s,
            "g4m-moe-rnorm",
            x_router,
            &ones_w,
            &rnormed,
            hidden,
            eps,
        )?;
        let mp = b.uni(
            "g4m-mul-p",
            WordsParams {
                n_words: hidden_words as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1(hidden_words as u64, 64);
        b.push(
            "g4m-moe-rmul",
            &s.moe,
            "g4m_mul_bf16",
            &[(40, &rnormed), (41, &rsw), (42, &router_in), (43, &mp)],
            grid,
        )?;
    }

    let (router, ids, wts, pes, rlogits) = match pre {
        Some(p) => (
            Bf16Gpu {
                w: p.router.w.clone(),
                n: p.router.n,
                k: p.router.k,
            },
            p.ids.clone(),
            p.wts.clone(),
            p.pes.clone(),
            p.rlogits.clone(),
        ),
        None => (
            upload_bf16(b, "g4m-moe-router", &layer.router),
            b.zeros("g4m-moe-ids", (k_top * 4) as u64),
            b.zeros("g4m-moe-wts", (k_top * 4) as u64),
            b.upload_f32("g4m-moe-pes", &layer.per_expert_scale),
            b.zeros("g4m-moe-rlogits", (n_e * 4) as u64),
        ),
    };
    let topk_fold = topk_fold_enabled();
    anyhow::ensure!(
        !topk_fold || (grouped && !chain && n_e <= 256 && k_top <= 16),
        "NV_G4MOE_TOPK_FOLD=1 replays the parallel-rank topk inside every expert gemv          workgroup; it rides the grouped fused gu/down entries ({n_e} experts, top-{k_top})"
    );
    if chain {
        let cp = b.uni(
            "g4m-rc-p",
            RouterChainParams {
                n_experts: n_e as u32,
                k: k_top as u32,
                words: hidden_words as u32,
                hidden: hidden as u32,
                eps,
                alpha: 1.0 / (hidden as f32).sqrt(),
                ..Default::default()
            },
        );
        b.push(
            "g4m-moe-routerchain",
            &s.moe,
            "g4m_router_chain",
            &[
                (70, x_router),
                (71, &rsw),
                (72, &router.w),
                (73, &pes),
                (74, &ids),
                (75, &wts),
                (76, &cp),
            ],
            (1, 1, 1),
        )?;
    } else {
        if pre.is_some() {
        } else if rpnorm {
            let grid = b.grid1(n_e.div_ceil(2) as u64, 1);
            let p = b.uni(
                "g4m-gemvb-p",
                GemvBf16Params {
                    n_rows: n_e as u32,
                    k_words: hidden_words as u32,
                    groups_x: grid.0,
                    out_f32: 1,
                    w_row_words: hidden_words as u32,
                    x_off_words: 0,
                    y_off_words: 0,
                    wide: 0,
                    alpha: 1.0 / (hidden as f32).sqrt(),
                    pad1: hidden as u32,
                    pad2: eps.to_bits(),
                    ..Default::default()
                },
            );
            let (src, entry) = if norm_lat_active(b.ctx) {
                (&s.gemv_bf16_rpnorm_sglat, "g4m_gemv_bf16_rpnorm_sglat")
            } else {
                (&s.gemv_bf16, "g4m_gemv_bf16_rpnorm")
            };
            b.push(
                "g4m-moe-rpnorm",
                src,
                entry,
                &[
                    (0, &router.w),
                    (1, x_router),
                    (2, &p),
                    (3, &rlogits),
                    (5, &rsw),
                ],
                grid,
            )?;
        } else {
            push_gemv_bf16_alpha(
                b,
                s,
                "g4m-moe-rproj",
                &router,
                &router_in,
                &rlogits,
                true,
                0,
                1.0 / (hidden as f32).sqrt(),
                GemvLoad::Scalar,
            )?;
        }
        if !topk_fold && !(topk_prefused && pre.is_some()) {
            let rp = b.uni(
                "g4m-moe-topk-p",
                RouterParams {
                    n_experts: n_e as u32,
                    k: k_top as u32,
                    ..Default::default()
                },
            );
            b.push(
                "g4m-moe-topk",
                &s.moe,
                router_topk_entry(n_e, k_top),
                &[(0, &rlogits), (1, &ids), (2, &wts), (3, &rp), (4, &pes)],
                (1, 1, 1),
            )?;
        }
    }
    let rank_src = topk_fold.then_some((&rlogits, &pes, n_e));

    let eg = upload_experts(b, "g4m-moe-eg", &layer.experts_gate);
    let eu = upload_experts(b, "g4m-moe-eu", &layer.experts_up);
    let ed = upload_experts(b, "g4m-moe-ed", &layer.experts_down);

    let act = b.zeros("g4m-moe-act", (k_top * inter * 2) as u64);
    let down_groups = ed.k / W4_GS;
    let fused_down = grouped && w4_gemv_entry(down_groups, ed.n) == "g4m_gemv_w4_r8";
    anyhow::ensure!(
        !topk_fold || fused_down,
        "NV_G4MOE_TOPK_FOLD=1 has no unfused down+combine surface; the down shape must \
         route g4m_gemv_w4_r8"
    );
    if grouped {
        push_gemv_w4_gu_gelu(
            b,
            s,
            "g4m-moe-gu",
            &eg,
            &eu,
            x,
            &act,
            &ids,
            k_top,
            rank_src.map(|(l, p, ne)| (l as &wgpu::Buffer, p as &wgpu::Buffer, ne)),
        )?;
    } else {
        let y_gate = b.zeros("g4m-moe-ygate", (k_top * inter * 2) as u64);
        let y_up = b.zeros("g4m-moe-yup", (k_top * inter * 2) as u64);
        push_gemv_w4(b, s, "g4m-moe-gate", &eg, x, &y_gate, &ids, k_top, false)?;
        push_gemv_w4(b, s, "g4m-moe-up", &eu, x, &y_up, &ids, k_top, false)?;
        let gp = b.uni(
            "g4m-gelu-p",
            WordsParams {
                n_words: (k_top * inter / 2) as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1((k_top * inter / 2) as u64, 64);
        b.push(
            "g4m-moe-gelu",
            &s.moe,
            "g4m_gelu_mul",
            &[(10, &y_gate), (11, &y_up), (12, &act), (13, &gp)],
            grid,
        )?;
    }

    if fused_down {
        push_gemv_w4_down_combine(
            b,
            s,
            "g4m-moe-downcomb",
            &ed,
            &act,
            out,
            &ids,
            &wts,
            k_top,
            rank_src.map(|(l, p, ne)| (l as &wgpu::Buffer, p as &wgpu::Buffer, ne)),
        )?;
    } else {
        let y_down = b.zeros("g4m-moe-ydown", (k_top * hidden * 2) as u64);
        push_gemv_w4(b, s, "g4m-moe-down", &ed, &act, &y_down, &ids, k_top, true)?;
        let cp = b.uni(
            "g4m-moe-comb-p",
            CombineParams {
                hidden_words: hidden_words as u32,
                k: k_top as u32,
                slot_stride_words: hidden_words as u32,
                pad0: 0,
            },
        );
        let grid = b.grid1(hidden_words as u64, 64);
        b.push(
            "g4m-moe-combine",
            &s.moe,
            "g4m_moe_combine",
            &[(20, &y_down), (21, &wts), (22, out), (23, &cp)],
            grid,
        )?;
    }

    Ok(MoeGpu {
        ones_w,
        rsw,
        router,
        pes,
        eg,
        eu,
        ed,
    })
}

#[allow(clippy::too_many_arguments)]
fn pf_gemv_w4(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    e: &ExpertGpu,
    x: &wgpu::Buffer,
    x_tok_stride_words: usize,
    x_slot_stride_words: usize,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    slots: usize,
    k_top: usize,
) -> Result<()> {
    let groups = e.k / W4_GS;
    let pairs = e.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "g4m-pf-w4-p",
        PfW4Params {
            n_rows: e.n as u32,
            groups: groups as u32,
            groups_x: grid.0,
            w_e_stride_words: (e.n * e.k / 8) as u32,
            s_e_stride_elems: (e.n * groups) as u32,
            x_slot_stride_words: x_slot_stride_words as u32,
            y_slot_stride_words: (e.n / 2) as u32,
            x_tok_stride_words: x_tok_stride_words as u32,
            k_top: k_top as u32,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.prefill,
        "pm_gemv_w4",
        &[
            (70, &e.w),
            (71, &e.scales),
            (72, x),
            (73, &p),
            (74, y),
            (75, sel),
        ],
        (grid.0, grid.1, slots as u32),
    )
}

#[allow(clippy::too_many_arguments)]
fn build_moe_prefill(
    b: &mut Builder,
    s: &Sources,
    cfg: &Gemma4MoeConfig,
    moe: &MoeGpu,
    sc: &PfScratch,
    x_router: &wgpu::Buffer,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
) -> Result<()> {
    let m = sc.m;
    let hidden = cfg.base.hidden_size;
    let hidden_words = hidden / 2;
    let inter = cfg.moe_intermediate_size;
    let n_e = cfg.num_experts;
    let k_top = cfg.top_k_experts;
    let eps = cfg.base.rms_norm_eps as f32;

    pf_norm(
        b,
        s,
        "g4m-pf-moe-rnorm",
        x_router,
        &moe.ones_w,
        &sc.rnormed,
        hidden,
        eps,
        m,
    )?;

    let mp = b.uni(
        "g4m-pf-mul-p",
        PfMulParams {
            row_words: hidden_words as u32,
            m: m as u32,
            ..Default::default()
        },
    );
    b.push(
        "g4m-pf-moe-rmul",
        &s.prefill,
        "pm_mul_rowscale",
        &[
            (90, &sc.rnormed),
            (91, &moe.rsw),
            (92, &sc.router_in),
            (93, &mp),
        ],
        ((hidden_words as u32).div_ceil(64), m as u32, 1),
    )?;

    pf_gemm_bf16(
        b,
        s,
        "g4m-pf-moe-rproj",
        &moe.router,
        &sc.router_in,
        hidden_words,
        &sc.rlogits,
        n_e,
        true,
        1.0 / (hidden as f32).sqrt(),
    )?;

    let rp = b.uni(
        "g4m-pf-topk-p",
        PfRouterParams {
            n_experts: n_e as u32,
            k: k_top as u32,
            m: m as u32,
            pad0: 0,
        },
    );
    b.push(
        "g4m-pf-moe-topk",
        &s.prefill,
        "pm_router_topk",
        &[
            (60, &sc.rlogits),
            (61, &sc.ids),
            (62, &sc.wts),
            (63, &rp),
            (64, &moe.pes),
        ],
        (m as u32, 1, 1),
    )?;

    let slots = m * k_top;
    pf_gemv_w4(
        b,
        s,
        "g4m-pf-moe-gate",
        &moe.eg,
        x,
        hidden_words,
        0,
        &sc.y_gate,
        &sc.ids,
        slots,
        k_top,
    )?;
    pf_gemv_w4(
        b,
        s,
        "g4m-pf-moe-up",
        &moe.eu,
        x,
        hidden_words,
        0,
        &sc.y_up,
        &sc.ids,
        slots,
        k_top,
    )?;

    let act_words = slots * inter / 2;
    let gp = b.uni(
        "g4m-gelu-p",
        WordsParams {
            n_words: act_words as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1(act_words as u64, 64);
    b.push(
        "g4m-pf-moe-gelu",
        &s.moe,
        "g4m_gelu_mul",
        &[
            (10, &sc.y_gate),
            (11, &sc.y_up),
            (12, &sc.moe_act),
            (13, &gp),
        ],
        grid,
    )?;

    pf_gemv_w4(
        b,
        s,
        "g4m-pf-moe-down",
        &moe.ed,
        &sc.moe_act,
        0,
        inter / 2,
        &sc.y_down,
        &sc.ids,
        slots,
        k_top,
    )?;

    let cp = b.uni(
        "g4m-pf-comb-p",
        PfCombineParams {
            hidden_words: hidden_words as u32,
            k: k_top as u32,
            slot_stride_words: hidden_words as u32,
            m: m as u32,
        },
    );
    let cx = (hidden_words as u32).div_ceil(64);
    b.push(
        "g4m-pf-moe-combine",
        &s.prefill,
        "pm_moe_combine",
        &[(80, &sc.y_down), (81, &sc.wts), (82, out), (83, &cp)],
        (cx, m as u32, 1),
    )
}

crate::wgpu_state_snapshot::impl_wgpu_state_snapshot!(Gemma4MoeWgpu, max_seq);
