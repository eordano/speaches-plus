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

const GEMV_BF16_LEGACY_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_bf16_legacy.wgsl");

const GEMV_I8_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_i8.wgsl");

const GEMV_I8_V4_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_i8_v4.wgsl");

const GEMV_W4E_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_gemv_w4e.wgsl");

const ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_attn.wgsl");

const FLASH_DECODE_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_flash_decode.wgsl");

const PROP_NORM_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4m_prop_norm.wgsl");

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
    assert!(
        k.is_multiple_of(I8_GS),
        "k={k} is not a multiple of {I8_GS}"
    );
    assert_eq!(w.len(), n * k, "quantize_i8_host: w is not n*k");
    let groups = k / I8_GS;
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
                        let base = rr * k + g * I8_GS;
                        let mut amax = 0f32;
                        for j in 0..I8_GS {
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
                        for j in 0..I8_GS {
                            let v = f32::from_bits((w_chunk[base + j] as u32) << 16);
                            let qi = (v * inv).round().clamp(-127.0, 127.0) as i32;
                            let elem = g * I8_GS + j;
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
}

fn vram_class(label: &str) -> &str {
    label.strip_prefix("g4m-").unwrap_or(label)
}

fn per_token_weight_share(class: &str, expert_share: f64) -> f64 {
    match class {
        "moe-eg" | "moe-eu" | "moe-ed" | "moe-eg-sf" | "moe-eu-sf" | "moe-ed-sf" => expert_share,
        "embed" | "ln" | "final-ln" | "at-qw" | "at-kw" | "at-vw" | "at-ow" | "at-qn" | "at-kn"
        | "at-vn" | "mlp-gw" | "mlp-uw" | "mlp-dw" | "moe-router" | "moe-rscale" | "moe-pes"
        | "moe-ones" | "lmhead-i8" | "lmhead-i8-sf" => 1.0,
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
        let sink = if self.to_prefill {
            &mut self.prefill_passes
        } else {
            &mut self.passes
        };
        sink.push(Pass {
            pipeline,
            bind,
            grid,
            label: format!("{label}:{entry}"),
            bind_bytes: binds.iter().map(|(_, b)| b.size()).sum(),
        });
        Ok(())
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
    gemv_w4: String,
    attn: String,
    flash: String,
    moe: String,
    prop: String,
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
    fn new(prefill_m: usize) -> Self {
        Self {
            gemv_bf16: compose(GEMV_BF16_WGSL),
            gemv_w4: compose(GEMV_W4E_WGSL),
            attn: compose(ATTN_WGSL),
            flash: compose(FLASH_DECODE_WGSL),
            moe: compose(MOE_WGSL),
            prop: compose(PROP_NORM_WGSL),
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
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let wide = b.wide_flag(label, load, w.k, 0, &w.w, x);
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
    b.push(
        label,
        &s.gemv_bf16,
        "g4m_gemv_bf16",
        &[(0, &w.w), (1, x), (2, &p), (3, y)],
        grid,
    )
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
    b.push(
        label,
        &s.gemv_bf16,
        "g4m_gemv_bf16_gu_gelu",
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
    let (entry, rows_per_wg) = if base_entry == "g4m_gemv_w4_r8" {
        ("g4m_gemv_w4_r8_gu_gelu", 8)
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
            k_top: 0,
        },
    );
    b.push(
        label,
        &s.gemv_w4,
        entry,
        &[
            (10, &eg.w),
            (11, &eg.scales),
            (12, x),
            (14, &p),
            (15, act),
            (16, sel),
            (17, &eu.w),
            (18, &eu.scales),
        ],
        (grid.0, grid.1, slots as u32),
    )
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
) -> Result<()> {
    let groups = ed.k / W4_GS;
    anyhow::ensure!(
        w4_gemv_entry(groups, ed.n) == "g4m_gemv_w4_r8",
        "g4m_gemv_w4_r8_down_combine mirrors the r8 row walk only; the caller must fall \
         back to the unfused down+combine pair for the pair-entry shape"
    );
    let grid = b.grid1(ed.n.div_ceil(8) as u64, 1);
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
        },
    );
    b.push(
        label,
        &s.gemv_w4,
        "g4m_gemv_w4_r8_down_combine",
        &[
            (10, &ed.w),
            (11, &ed.scales),
            (12, act),
            (14, &p),
            (15, out),
            (16, sel),
            (17, &ed.w),
            (18, &ed.scales),
            (19, wts),
        ],
        (grid.0, grid.1, 1),
    )
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

pub const G4MOE_SLIDING_KV_RING_DEFAULT_ON: bool = false;

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

pub const G4MOE_FLASH_DECODE_DEFAULT_ON: bool = false;

pub const G4MOE_FLASH_SPLITS_16_MIRRORS_GEMMA4_WGPU_FLASH_SPLITS: u32 = 16;

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
        Err(_) => G4MOE_FLASH_SPLITS_16_MIRRORS_GEMMA4_WGPU_FLASH_SPLITS,
    }
}

const FLASH_WORKGROUP_BYTES_QSH512_RED256_SM8_SL8_SACC4096_F32: u32 =
    (512 + 256 + 8 + 8 + 4096) * 4;

pub fn flash_decode_enabled() -> bool {
    match std::env::var("NV_G4MOE_FLASH_DECODE") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_FLASH_DECODE_DEFAULT_ON,
    }
}

pub const G4MOE_FLASH_SLIDING_DEFAULT_ON: bool = false;

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

pub const G4MOE_GROUPED_GEMV_DEFAULT_ON: bool = false;

pub const GROUPED_NORM_CHAIN_MAX_HIDDEN_WORDS_PN_BUF: usize = 2048;

pub fn grouped_gemv_enabled() -> bool {
    match std::env::var("NV_G4MOE_GROUPED_GEMV") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => G4MOE_GROUPED_GEMV_DEFAULT_ON,
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
    let grid = b.grid1(pairs as u64, 1);
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
            wide: c.wide,
            alpha,
            ..Default::default()
        },
    );
    b.push(
        "g4m-lmhead",
        &s.gemv_bf16,
        "g4m_gemv_bf16",
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
    _buffers: Vec<wgpu::Buffer>,
    tok_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    flash_fd: Option<(wgpu::Buffer, FlashDecodeParams)>,
    flash_sl: Option<(wgpu::Buffer, FlashDecodeParams, u32)>,
    token_out: wgpu::Buffer,
    logits: wgpu::Buffer,
    state_buffers: Vec<(wgpu::Buffer, u64)>,
    vocab: usize,
    vram: VramReport,
    load: LoadReport,
    weight_bytes: u64,
    dense_gemv_wide: (usize, usize),
}

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
        let s = Sources::new(pf_m);
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
                splits: flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
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

        let grouped = grouped_gemv_enabled();
        anyhow::ensure!(
            !grouped || hidden_words <= GROUPED_NORM_CHAIN_MAX_HIDDEN_WORDS_PN_BUF,
            "NV_G4MOE_GROUPED_GEMV=1: the fused norm chain stages {hidden_words} hidden words \
             in a {GROUPED_NORM_CHAIN_MAX_HIDDEN_WORDS_PN_BUF}-word workgroup array"
        );
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
            push_norm(&mut b, &s, "g4m-rms-in", r_in, &in_w, &normed, hidden, eps)?;

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
            )?;

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
                b.push(
                    "g4m-nnres",
                    &s.prop,
                    "g4m_norm_norm_residual",
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
            let mg = build_mlp(&mut b, &s, &config, &layer, &normed_mlp, &dense_out)?;
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
                b.push(
                    "g4m-normx2",
                    &s.prop,
                    "g4m_norm_x2",
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
            let moeg = build_moe(&mut b, &s, &config, &layer, r_in, &moe_in, &moe_out)?;
            let post_ffw2_w = b.upload_u32("g4m-ln", &pack_pairs(&layer.post_ffw_ln_2));
            let post_ffw_w = b.upload_u32("g4m-ln", &pack_pairs(&layer.post_ffw_ln));
            if grouped {
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
        let final_w = b.upload_u32(
            "g4m-final-ln",
            &pack_pairs(&load_bf16(
                weights,
                "model.language_model.norm.weight",
                &[hidden],
            )?),
        );
        let final_x = b.zeros("g4m-final-x", (hidden_words * 4) as u64);
        push_norm(
            &mut b,
            &s,
            "g4m-final-rms",
            r_last,
            &final_w,
            &final_x,
            hidden,
            eps,
        )?;

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
            Some(sc) if lmhead_i8.is_none() && hidden_words.is_multiple_of(4) => {
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
            _buffers: buffers,
            tok_buf,
            pos_buf,
            flash_fd,
            flash_sl,
            token_out,
            logits,
            state_buffers,
            vocab,
            vram,
            load,
            weight_bytes: weight_bpt.round() as u64,
            dense_gemv_wide,
        };
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

    fn write_flash_total_uniform_mirroring_gemma4_wgpu_write_pos_uniforms(&self) {
        if let Some((buf, base)) = &self.flash_fd {
            let mut p = *base;
            p.total = (self.pos + 1) as u32;
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }
        if let Some((buf, base, window)) = &self.flash_sl {
            let mut p = *base;
            p.total = (self.pos + 1) as u32;
            p.start = if p.total > *window { p.total - *window } else { 0 };
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
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in passes {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
            self.ctx.queue.submit([enc.finish()]);
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
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for p in passes {
            pass.set_pipeline(&p.pipeline);
            pass.set_bind_group(0, &p.bind, &[]);
            pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
        }
    }
    ctx.queue.submit([enc.finish()]);
    Ok(())
}

fn row_chunk(ctx: &WgpuContext, hidden: usize) -> usize {
    if let Some(rows) = std::env::var("NV_G4MOE_ROW_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|v| *v >= 2)
    {
        return rows & !1usize;
    }
    let limit = ctx
        .caps
        .max_storage_buffer_binding_size
        .clamp(1 << 20, 1u64 << 30);
    let per_row = (hidden * 2) as u64;
    ((limit / per_row) as usize).max(2) & !1usize
}

type RopeTableCache = std::collections::HashMap<(usize, u32, u32), (wgpu::Buffer, wgpu::Buffer)>;

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
    let grouped_attn = grouped_gemv_enabled() && !fp8;

    let wq = upload_bf16(b, "g4m-at-qw", &layer.q);
    let wk_ = upload_bf16(b, "g4m-at-kw", &layer.k);
    let wo = upload_bf16(b, "g4m-at-ow", &layer.o);
    let wv = layer
        .v
        .as_ref()
        .map(|vlin| upload_bf16(b, "g4m-at-vw", vlin));

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
        drop(stacked);
        let qkv_raw = b.zeros("g4m-at-qkvraw", (n_stack * 2) as u64);
        push_gemv_bf16(b, s, "g4m-at-qkvproj", &wqkv, x, &qkv_raw, false, 0)?;

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

    let ring_rows = match (kind, sliding_ring_rows) {
        (LayerType::SlidingAttention, Some(rows)) => Some(rows),
        _ => None,
    };
    let ring = ring_rows.map(|r| r as u32).unwrap_or(0);
    let cache_rows = ring_rows.unwrap_or(max_seq);
    let (kc, vc, kv_scales) = if fp8 {
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
    let use_flash = match kind {
        LayerType::FullAttention => flash_decode_enabled(),
        LayerType::SlidingAttention => flash_sliding_enabled(),
    };
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
        anyhow::ensure!(
            b.ctx.caps.max_compute_invocations_per_workgroup >= 256
                && b.ctx.caps.max_compute_workgroup_size_x >= 256
                && b
                    .ctx
                    .caps
                    .workgroup_storage_fits(FLASH_WORKGROUP_BYTES_QSH512_RED256_SM8_SL8_SACC4096_F32),
            "NV_G4MOE_FLASH_DECODE=1 needs a 256-invocation workgroup and {} workgroup bytes; \
             device allows {} invocations / {} bytes -- unset the env to keep the serial arm",
            FLASH_WORKGROUP_BYTES_QSH512_RED256_SM8_SL8_SACC4096_F32,
            b.ctx.caps.max_compute_invocations_per_workgroup,
            b.ctx.caps.max_compute_workgroup_storage_size
        );
        let splits = flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build();
        let scratch = b.zeros(
            "g4m-at-fscr",
            (n_q * splits as usize * (hd + 2) * 4) as u64,
        );
        let mut flash1_binds: Vec<(u32, &wgpu::Buffer)> =
            vec![(0, &q), (1, &kc), (2, &vc), (3, &scratch), (6, fp)];
        let flash1_entry = match &kv_scales {
            Some((ksc, vsc)) => {
                flash1_binds.push((7, ksc));
                flash1_binds.push((8, vsc));
                "g4m_flash_stage1_fp8"
            }
            None => "g4m_flash_stage1_bf16",
        };
        b.push(
            &format!("g4m-at-flash1{kind_suffix}"),
            &s.flash,
            flash1_entry,
            &flash1_binds,
            (n_q as u32, splits, 1),
        )?;
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

    push_gemv_bf16(b, s, "g4m-at-oproj", &wo, &attn_bf16, out, false, 0)?;

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
) -> Result<MlpGpu> {
    let inter = cfg.base.intermediate_size;
    let wg = upload_bf16(b, "g4m-mlp-gw", &layer.mlp_gate);
    let wu = upload_bf16(b, "g4m-mlp-uw", &layer.mlp_up);
    let wd = upload_bf16(b, "g4m-mlp-dw", &layer.mlp_down);
    let act = b.zeros("g4m-mlp-act", (inter * 2) as u64);
    if grouped_gemv_enabled() {
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
    push_gemv_bf16(b, s, "g4m-mlp-down", &wd, &act, out, false, 0)?;
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
fn build_moe(
    b: &mut Builder,
    s: &Sources,
    cfg: &Gemma4MoeConfig,
    layer: &HostLayer,
    x_router: &wgpu::Buffer,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
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

    let grouped = grouped_gemv_enabled();
    let ones: Vec<u16> = vec![bf16_bits(1.0); hidden];
    let ones_w = b.upload_u32("g4m-moe-ones", &pack_pairs(&ones));
    let router_in = b.zeros("g4m-moe-rin", (hidden_words * 4) as u64);
    let rsw = b.upload_u32("g4m-moe-rscale", &pack_pairs(&layer.router_scale));
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

    let router = upload_bf16(b, "g4m-moe-router", &layer.router);
    let rlogits = b.zeros("g4m-moe-rlogits", (n_e * 4) as u64);
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

    let ids = b.zeros("g4m-moe-ids", (k_top * 4) as u64);
    let wts = b.zeros("g4m-moe-wts", (k_top * 4) as u64);
    let pes = b.upload_f32("g4m-moe-pes", &layer.per_expert_scale);
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

    let eg = upload_experts(b, "g4m-moe-eg", &layer.experts_gate);
    let eu = upload_experts(b, "g4m-moe-eu", &layer.experts_up);
    let ed = upload_experts(b, "g4m-moe-ed", &layer.experts_down);

    let act = b.zeros("g4m-moe-act", (k_top * inter * 2) as u64);
    let down_groups = ed.k / W4_GS;
    let fused_down = grouped && w4_gemv_entry(down_groups, ed.n) == "g4m_gemv_w4_r8";
    if grouped {
        push_gemv_w4_gu_gelu(b, s, "g4m-moe-gu", &eg, &eu, x, &act, &ids, k_top)?;
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
