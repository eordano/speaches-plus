use anyhow::{Context, Result};

use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, compose_enabled, WgpuContext};

use crate::qwen3_5_moe::LayerType;
pub use crate::qwen3_5_moe::Qwen3_5DenseConfig;
use crate::qwen3_5_moe_wgpu::{
    bytes_to_words, delta_recurrent_kernel, delta_recurrent_variants_source, load_nvfp4,
    nvfp4_gemv_source, nvfp4_quant_source, nvfp4_v2_route,
    nvfp4_v2_route_slotshared, quant_lane_entry, staging_flush_enabled, GemvNvfp4Params,
    HostBf16Lin, HostDeltaNet, HostNvfp4Lin, QuantRowsParams, SiluPairParams, VramReport,
    NVFP4_BLOCK,
};

const MAX_HEAD_DIM: usize = 256;
const MAX_LIN_HEAD_DIM: usize = 128;
const ARGMAX_GROUPS: usize = 256;

const STAGING_FLUSH_BYTES: u64 = 256 << 20;

const GEMV_BF16_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_gemv_bf16.wgsl");

const GEMV_I8_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_gemv_i8.wgsl");

const DELTA_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_delta.wgsl");

const ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_attn.wgsl");

const MISC_WGSL: &str = concat!(
    include_str!("../../nv-kernels/wgsl/q3d_misc.wgsl"),
    include_str!("../../nv-kernels/wgsl/q3w_argmax.wgsl")
);

const PREFILL_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_prefill.wgsl");

const SPLICE_WGSL: &str = r#"
struct Q3spParams { hidden_words: u32, m: u32, pad0: u32, pad1: u32 };
@group(0) @binding(4) var<storage, read> psp_rows: array<u32>;
@group(0) @binding(5) var<storage, read> psp_mask: array<u32>;
@group(0) @binding(6) var<uniform> psp_p: Q3spParams;
@compute @workgroup_size(256)
fn q3w_splice_image_rows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    if (psp_mask[t] == 0u) { return; }
    let w = wid.x * 256u + lid.x;
    if (w >= psp_p.hidden_words) { return; }
    pge_out[t * psp_p.hidden_words + w] = psp_rows[t * psp_p.hidden_words + w];
}
"#;

fn prefill_composed() -> String {
    compose(&format!("{PREFILL_WGSL}\n{SPLICE_WGSL}"))
}

fn gemm_mk_source(m: usize) -> String {
    use std::fmt::Write as _;
    assert!((2..=16).contains(&m), "gemm_mk m must be 2..=16, got {m}");
    let mut b = String::new();
    b.push_str(
        "struct Q3gmParams {\n    n_rows: u32,\n    k_words: u32,\n    groups_x: u32,\n    w_row_words: u32,\n    x_stride_words: u32,\n    y_stride_words: u32,\n    pad0: u32,\n    pad1: u32,\n};\n\n",
    );
    b.push_str("@group(0) @binding(0) var<storage, read> gm_w: array<u32>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> gm_x: array<u32>;\n");
    b.push_str("@group(0) @binding(2) var<uniform> gm_p: Q3gmParams;\n");
    b.push_str("@group(0) @binding(3) var<storage, read_write> gm_y: array<u32>;\n\n");
    b.push_str("var<workgroup> gm_red: array<f32, 256>;\n\n");
    writeln!(b, "@compute @workgroup_size(256)").unwrap();
    writeln!(b, "fn {}(", gemm_mk_entry(m)).unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
    b.push_str("    let tid = lid.x;\n");
    b.push_str("    let half = tid >> 7u;\n");
    b.push_str("    let lane = tid & 127u;\n");
    b.push_str("    let pair = wid.x + wid.y * gm_p.groups_x;\n");
    b.push_str("    let row = pair * 2u + half;\n");
    b.push_str("    let live = row < gm_p.n_rows;\n");
    b.push_str("    let wbase = select(0u, row * gm_p.w_row_words, live);\n");
    b.push_str("    let kw = select(0u, gm_p.k_words, live);\n");
    writeln!(b, "    var acc: array<f32, {m}>;").unwrap();
    writeln!(
        b,
        "    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{ acc[mi] = 0.0; }}"
    )
    .unwrap();
    b.push_str("    for (var i = lane; i < kw; i = i + 128u) {\n");
    b.push_str("        let ww = gm_w[wbase + i];\n");
    writeln!(b, "        for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
    b.push_str("            let xw = gm_x[mi * gm_p.x_stride_words + i];\n");
    b.push_str("            acc[mi] = fma(bf16_lo(ww), bf16_lo(xw), acc[mi]);\n");
    b.push_str("            acc[mi] = fma(bf16_hi(ww), bf16_hi(xw), acc[mi]);\n");
    b.push_str("        }\n    }\n");
    writeln!(b, "    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
    b.push_str("        gm_red[tid] = acc[mi];\n");
    b.push_str("        workgroupBarrier();\n");
    b.push_str("        for (var stride = 64u; stride > 0u; stride = stride >> 1u) {\n");
    b.push_str("            if (lane < stride) {\n");
    b.push_str("                gm_red[tid] = gm_red[tid] + gm_red[tid + stride];\n");
    b.push_str("            }\n            workgroupBarrier();\n        }\n");
    b.push_str("        if (tid == 0u) {\n");
    b.push_str("            let lo = gm_red[0];\n");
    b.push_str("            var hi = 0.0;\n");
    b.push_str("            if (row + 1u < gm_p.n_rows) {\n");
    b.push_str("                hi = gm_red[128];\n");
    b.push_str("            }\n");
    b.push_str("            gm_y[mi * gm_p.y_stride_words + (row >> 1u)] = bf16_pack(lo, hi);\n");
    b.push_str("        }\n");
    b.push_str("        workgroupBarrier();\n    }\n}\n");
    compose(&b)
}

fn gemm_mk_entry(m: usize) -> String {
    format!("q3w_gemm_bf16_m{m}")
}

fn gemm_i8_mk_source(m: usize) -> String {
    use std::fmt::Write as _;
    assert!(
        (2..=16).contains(&m),
        "gemm_i8_mk m must be 2..=16, got {m}"
    );
    let mut b = String::new();
    b.push_str(
        "struct Q3q8mParams {\n    n_rows: u32,\n    k_elems: u32,\n    groups_x: u32,\n    groups_per_row: u32,\n    group_shift: u32,\n    x_stride_words: u32,\n    y_stride_words: u32,\n    pad0: u32,\n};\n\n",
    );
    b.push_str("@group(0) @binding(0) var<storage, read> q8m_w: array<u32>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> q8m_s: array<f32>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read> q8m_x: array<u32>;\n");
    b.push_str("@group(0) @binding(3) var<storage, read_write> q8m_y: array<u32>;\n");
    b.push_str("@group(0) @binding(4) var<uniform> q8m_p: Q3q8mParams;\n\n");
    b.push_str("var<workgroup> q8m_pk_bits: array<u32, 8>;\n\n");
    for grouped in [true, false] {
        writeln!(b, "@compute @workgroup_size(256)").unwrap();
        writeln!(b, "fn {}(", gemm_i8_mk_entry(m, grouped)).unwrap();
        b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(subgroup_id) sgid: u32,\n    @builtin(subgroup_invocation_id) lane: u32\n) {\n");
        b.push_str("    let row = (wid.x + wid.y * q8m_p.groups_x) * 8u + sgid;\n");
        b.push_str("    let live = row < q8m_p.n_rows;\n");
        b.push_str("    let words = select(0u, q8m_p.k_elems >> 2u, live);\n");
        b.push_str("    let wbase = select(0u, row * (q8m_p.k_elems >> 2u), live);\n");
        if grouped {
            b.push_str("    let sbase = select(0u, row * q8m_p.groups_per_row, live);\n");
            b.push_str("    let gshift = q8m_p.group_shift;\n");
        }
        writeln!(b, "    var acc: array<f32, {m}>;").unwrap();
        writeln!(
            b,
            "    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{ acc[mi] = 0.0; }}"
        )
        .unwrap();
        b.push_str("    for (var i = lane; i < words; i = i + 32u) {\n");
        b.push_str("        let w = q8m_w[wbase + i];\n");
        if grouped {
            b.push_str("        let sc = q8m_s[sbase + (i >> gshift)];\n");
        }
        writeln!(b, "        for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
        b.push_str("            let x0 = q8m_x[mi * q8m_p.x_stride_words + 2u * i];\n");
        b.push_str("            let x1 = q8m_x[mi * q8m_p.x_stride_words + 2u * i + 1u];\n");
        if grouped {
            b.push_str("            var d = 0.0;\n");
            b.push_str("            d = fma(int8_decode(w, 0u), bf16_lo(x0), d);\n");
            b.push_str("            d = fma(int8_decode(w, 1u), bf16_hi(x0), d);\n");
            b.push_str("            d = fma(int8_decode(w, 2u), bf16_lo(x1), d);\n");
            b.push_str("            d = fma(int8_decode(w, 3u), bf16_hi(x1), d);\n");
            b.push_str("            acc[mi] = fma(sc, d, acc[mi]);\n");
        } else {
            b.push_str("            acc[mi] = fma(int8_decode(w, 0u), bf16_lo(x0), acc[mi]);\n");
            b.push_str("            acc[mi] = fma(int8_decode(w, 1u), bf16_hi(x0), acc[mi]);\n");
            b.push_str("            acc[mi] = fma(int8_decode(w, 2u), bf16_lo(x1), acc[mi]);\n");
            b.push_str("            acc[mi] = fma(int8_decode(w, 3u), bf16_hi(x1), acc[mi]);\n");
        }
        b.push_str("        }\n    }\n");
        writeln!(b, "    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
        b.push_str("        var a = acc[mi];\n");
        b.push_str("        a = a + subgroupShuffleXor(a, 16u);\n");
        b.push_str("        a = a + subgroupShuffleXor(a, 8u);\n");
        b.push_str("        a = a + subgroupShuffleXor(a, 4u);\n");
        b.push_str("        a = a + subgroupShuffleXor(a, 2u);\n");
        b.push_str("        a = a + subgroupShuffleXor(a, 1u);\n");
        if grouped {
            b.push_str("        if (lane == 0u) {\n");
            b.push_str("            q8m_pk_bits[sgid] = bf16_encode(a) & 0xffffu;\n");
            b.push_str("        }\n");
        } else {
            b.push_str("        if (lane == 0u) {\n");
            b.push_str(
                "            q8m_pk_bits[sgid] = bf16_encode(a * q8m_s[select(0u, row, live)]) & 0xffffu;\n",
            );
            b.push_str("        }\n");
        }
        b.push_str("        workgroupBarrier();\n");
        b.push_str("        if ((sgid & 1u) == 0u && lane == 0u && live) {\n");
        b.push_str("            var word = q8m_pk_bits[sgid];\n");
        b.push_str("            if (row + 1u < q8m_p.n_rows) {\n");
        b.push_str("                word = word | (q8m_pk_bits[sgid + 1u] << 16u);\n");
        b.push_str("            }\n");
        b.push_str("            q8m_y[mi * q8m_p.y_stride_words + (row >> 1u)] = word;\n");
        b.push_str("        }\n");
        b.push_str("        workgroupBarrier();\n    }\n}\n\n");
    }
    compose(&b)
}

fn gemm_i8_mk_entry(m: usize, grouped: bool) -> String {
    if grouped {
        format!("q3d_gemv_i8g_m{m}")
    } else {
        format!("q3d_gemv_i8_m{m}")
    }
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmI8MkParams {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    groups_per_row: u32,
    group_shift: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmMkParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    w_row_words: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CkParams {
    m_live: u32,
    base: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SpliceParams {
    hidden_words: u32,
    m: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ConvMParams {
    conv_dim: u32,
    kernel: u32,
    x_words: u32,
    mixed_stride: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DeltaQkvMParams {
    n_v: u32,
    d_k: u32,
    d_v: u32,
    key_dim: u32,
    v_per_k: u32,
    mixed_stride: u32,
    pad0: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NormRopeMParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    x_row_elems: u32,
    y_row_elems: u32,
    pad0: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnGateMParams {
    n_words: u32,
    head_dim: u32,
    src_stride: u32,
    gate_off: u32,
    has_gate: u32,
    x_row_elems: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvI8Params {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    groups_per_row: u32,
    group_shift: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
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

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvFp8Params {
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
struct FdParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    total: u32,
    start: u32,
    splits: u32,
    ring: u32,
    out_bf16: u32,
    scaling: f32,
    pad0: u32,
    fused: u32,
    pad2: u32,
    m_rows: u32,
    window: u32,
    pad3: u32,
    pad4: u32,
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
struct ConvParams {
    conv_dim: u32,
    kernel: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DeltaQkvParams {
    n_v: u32,
    d_k: u32,
    d_v: u32,
    key_dim: u32,
    v_per_k: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GatingParams {
    n_v: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RecurrentParams {
    heads: u32,
    d_k: u32,
    d_v: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct DeltaOutParams {
    n_v: u32,
    d_v: u32,
    pad0: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NormRopeParams {
    n_rows: u32,
    head_dim: u32,
    src_stride: u32,
    rot_half: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct KvWriteParams {
    words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnDecodeParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AttnGateParams {
    n_words: u32,
    head_dim: u32,
    src_stride: u32,
    gate_off: u32,
    has_gate: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SiluMulParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NormRopeFusedParams {
    n_q_rows: u32,
    n_k_rows: u32,
    head_dim: u32,
    q_src_stride: u32,
    k_src_stride: u32,
    rot_half: u32,
    pad0: u32,
    eps: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvDnMergedParams {
    qkv_pairs: u32,
    z_pairs: u32,
    ab_pairs: u32,
    qkv_rows: u32,
    z_rows: u32,
    ab_rows: u32,
    fp8_row_words: u32,
    bf16_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
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
pub enum HostDenseLin {
    Bf16(HostBf16Lin),
    Nvfp4(HostNvfp4Lin),
    Fp8 {
        fp8: HostFp8Lin,
        bf16: HostBf16Lin,
    },
}

impl HostDenseLin {
    pub fn n(&self) -> usize {
        match self {
            Self::Bf16(l) => l.n,
            Self::Nvfp4(l) => l.n,
            Self::Fp8 { fp8, .. } => fp8.n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            Self::Bf16(l) => l.k,
            Self::Nvfp4(l) => l.k,
            Self::Fp8 { fp8, .. } => fp8.k,
        }
    }

    pub fn is_nvfp4(&self) -> bool {
        matches!(self, Self::Nvfp4(_))
    }
}

impl From<HostBf16Lin> for HostDenseLin {
    fn from(l: HostBf16Lin) -> Self {
        Self::Bf16(l)
    }
}

impl From<HostNvfp4Lin> for HostDenseLin {
    fn from(l: HostNvfp4Lin) -> Self {
        Self::Nvfp4(l)
    }
}

#[derive(Clone)]
pub struct HostDenseAttention {
    pub q: HostDenseLin,
    pub k: HostDenseLin,
    pub v: HostDenseLin,
    pub o: HostDenseLin,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
}

#[derive(Clone)]
pub enum HostDenseMixer {
    Delta(Box<HostDeltaNet>),
    Attn(Box<HostDenseAttention>),
}

#[derive(Clone)]
pub struct HostDenseMlp {
    pub gate: HostDenseLin,
    pub up: HostDenseLin,
    pub down: HostDenseLin,
}

#[derive(Clone)]
pub struct HostFp8Lin {
    pub packed: Vec<u32>,
    pub scales: Vec<f32>,
    pub n: usize,
    pub k: usize,
}

#[derive(Clone, Default)]
pub struct DeltaFp8 {
    pub qkv: Option<HostFp8Lin>,
    pub z: Option<HostFp8Lin>,
    pub out: Option<HostFp8Lin>,
}

pub const FP8_PROJ_STREAMS_CHECKPOINT_BYTES_INSTEAD_OF_A_BF16_UPCAST: &str =
    "the qwen3.8 nvfp4 checkpoints ship the DeltaNet projections as F8_E4M3 with per-row \
     scales; loading them through load_bf16 upcasts 3-bit-mantissa data into an 8-bit-mantissa \
     container and doubles the decode weight stream for zero information. NV_Q3D_FP8_PROJ=0 \
     restores the upcast route; the prefill M-row gemm keeps the bf16 copy either way, so fp8 \
     residency is additive VRAM, not a swap";

pub const Q3D_FOLDS_THE_FULL_GQA_GROUP_UNLIKE_E4B: &str =
    "q3d decode folds all group=6 query heads per kv head by default: the 168k sweep reads \
     monotonically faster with fold depth on fp8-sd KV (perf/runs.jsonl -- the fold amortizes \
     both the KV stream and the e4m3 decode 6x), 8k improves 25.1 -> 24.8 and depth 256 is \
     neutral; ppl and accuracy identical to unfolded (runs.jsonl). e4b's fold-2 cap was measured on \
     ITS group-4 short-kv regime and does not transfer. NV_WGPU_GQA_FOLD overrides";

pub const KV_FP8_DEFAULT_ON_SET_0_FOR_THE_BF16_DECODE_ARM: &str =
    "fp8 KV decode is the default: it wins at every measured depth with quality held or
     better on the pinned corpus (rows in perf/runs.jsonl), and it pairs ONLY with the
     tiled prefill arm, whose chunk path quantizes into the same e4m3 cache the decode
     reads (a build ensure refuses the scores/mk escape arms with fp8 on, because they
     leave chunk rows unquantized). NV_Q3D_KV_FP8=0 restores the bf16 decode reads at
     the cost of the depth scaling the runs.jsonl ladder records; chunked-vs-replayed
     caches differ only by the coop arms' accepted reassociation drift, so the contract
     is the drift-tolerant pair (same-session ppl match + acceptance A/B), not byte
     identity";

pub const FLASH_TILE_STAYS_OFF_THE_COOPERATIVE_DOT_PAYS_THE_REDUCE_NOT_THE_RESCALE: &str =
    "NV_Q3D_FLASH_TILE=2..8 selects the tile-batched online-softmax fold stage1 (one \
     accumulator rescale per tile, llama.cpp fattn-vec shape). v1 lost to the fold ladder at \
     tile 2/4/8 on 168k vs 38.2 untiled (per-element K loads, 4 lanes per word); v2 restored \
     the vec4 lane-owns-word K path and still loses 85.5/69.2/59.0. rescale amortization \
     cannot pay here because the cost it removes was never dominant: this kernel computes \
     each position's score with a 32-lane cooperative dot plus a 5-shuffle butterfly PER \
     (position, folded head), which tiling does not touch, and the unrolled tile bodies \
     (fold 6 x tile 8) bloat the instruction stream. llama.cpp's fattn-vec avoids the \
     per-position reduce entirely by giving each THREAD its own position -- a KV-layout \
     redesign, not a tile knob. default 0 = untiled ships";

fn flash_tile_env() -> u32 {
    static T: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *T.get_or_init(|| {
        std::env::var("NV_Q3D_FLASH_TILE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|t| t.clamp(0, 8))
            .unwrap_or(0)
    })
}

fn kv_fp8_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_Q3D_KV_FP8").ok().as_deref() != Some("0"))
}

pub const KV_NVFP4_ENV_OFF_BY_DEFAULT_FP8_STAYS_THE_SHIPPED_KV_DECODE_FORMAT: &str =
    "NV_Q3D_KV_NVFP4";

pub const KV_NVFP4_ARMS_HALVE_THE_KV_STREAM_AGAIN_UNDER_A_PPL_GATE: &str =
    "NV_Q3D_KV_NVFP4=v4k8 keeps K on the fp8 exact-decode path and moves V to e2m1 nibbles with \
     per-row f32 scales; =k4v4 moves K too, with per-(32-token-block, kv-head, channel) scales \
     because post-RoPE K carries channel-wise outliers that a row amax would flatten. Both arms \
     keep the first slots at fp8 (attention sinks anchor softmax; \
     KV_NVFP4_SINK_SLOTS_STAY_FP8_TO_ANCHOR_SOFTMAX) and so require the fp8 cache and its \
     quantize dispatches to stay recorded, which also keeps chunked prefill coherent: the nvfp4 \
     quantizers read the bf16 KV cache after kv_write, so the same entries serve M=1 decode and \
     the M-row chunk with only the tokens uniform changing. All nvfp4 stage1 reads are exact \
     decodes: e2m1 decodes as int8x2 with the 0.5 fold carried once per slot in the V scale, \
     because e2m1 sd would land the 0.5 codes in flushed f32 denormals and sd-K measured worse \
     ppl for zero depth win (runs.jsonl kv-v4k8 rows). v4k8 is the measured arm that holds the \
     pinned-corpus ppl band and leads fp8 at 172k depth; k4v4 does not hold the ppl gate at \
     any measured sink-exemption size and remains only as the substrate a Hadamard-rotation \
     fallback would extend";

#[derive(Clone, Copy, PartialEq, Eq)]
enum KvNvfp4Arm {
    V4K8,
    K4V4,
}

fn kv_nvfp4_arm() -> Option<KvNvfp4Arm> {
    match std::env::var(KV_NVFP4_ENV_OFF_BY_DEFAULT_FP8_STAYS_THE_SHIPPED_KV_DECODE_FORMAT)
        .ok()
        .as_deref()
    {
        None | Some("") | Some("0") => None,
        Some("v4k8") => Some(KvNvfp4Arm::V4K8),
        Some("k4v4") => Some(KvNvfp4Arm::K4V4),
        Some(other) => panic!(
            "{KV_NVFP4_ENV_OFF_BY_DEFAULT_FP8_STAYS_THE_SHIPPED_KV_DECODE_FORMAT}={other} names \
             no arm; the measured arms are v4k8 and k4v4"
        ),
    }
}

fn fp8_proj_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_Q3D_FP8_PROJ").ok().as_deref() != Some("0"))
}

pub const PF_COOP_ENV_DEFAULT_ON_SET_0_FOR_THE_LEGACY_M16_GEMM_ARM: &str = "NV_Q3D_PF_COOP";

pub const PF_COOP_ROUTES_PF_PROJECTIONS_THROUGH_16X16_COOPMAT_AND_DROPS_THE_FP8_BF16_TWIN: &str =
    "The chunked-prefill projections route through the 16x16 f16 cooperative-matrix kernels by \
     default: w4a16 (WqFmt::Nvfp4Block16) for nvfp4 MLP weights, w8a16 \
     (WqFmt::Fp8RowscalePlain) for fp8 DN qkv/z/out and attn q/k/v/o projections, and the plain \
     f16 coop GEMM for the small bf16 DN ab projection. The coop route lifts the \
     NV_WGPU_PREFILL_M cap and default from 16 to 256 and skips the fp8 projections' bf16-twin \
     upload, whose only consumer was the m-row bf16 gemm the coop path replaces. Decode passes \
     are byte-identical on both arms; prefill numerics drift within the ppl gate because the \
     a16 side multiplies f16 activations instead of quantized ones. NV_Q3D_PF_COOP=0 restores \
     the m<=16 bf16 gemm prefill arm at the cost of the coop ladder's prefill throughput \
     (the prefill cost of the legacy arm is on the runs.jsonl ladder).";

fn pf_coop_requested() -> bool {
    std::env::var(PF_COOP_ENV_DEFAULT_ON_SET_0_FOR_THE_LEGACY_M16_GEMM_ARM)
        .ok()
        .as_deref()
        != Some("0")
}

pub const PF_ATTN_MK_ENV_DEFAULT_OFF_UNTIL_THE_LADDER_AND_PPL_GATE_ARE_ON_RECORD: &str =
    "NV_Q3D_PF_ATTN_MK";

pub const PF_ATTN_MK_STREAMS_KV_ONCE_PER_8_ROWS_INSTEAD_OF_MATERIALIZING_M_X_MAXSEQ_SCORES: &str =
    "NV_Q3D_PF_ATTN_MK=1 reroutes the M-row prefill attention through the in-tree flash pair \
     flash_splitk_stage1_bf16kv_mk + flash_splitk_stage2_mk in ceil(m/8) row groups: warp-parallel \
     online softmax over the same bf16 KV cache the scores path reads, one KV stream per 8 rows, \
     no a_scores slab write/re-read and no serial per-thread AV loop over the prefix. Group g \
     covers chunk rows [8g, 8g+mr) with fd total = base + 8g + mr so each row's causal end is its \
     own position; rows past m_live get m_rows clamped on the host exactly where the scores kernel \
     early-returns on ck.m_live. The bf16 mk stage1 applies no fd scaling, so the q cast into f32 \
     pre-multiplies 1/sqrt(head_dim). Default off: the tiled fp8-kv arm is the prefill-attention \
     default, and NV_Q3D_PF_ATTN_MK=1 selects this arm in its place until the mk arm's ladder \
     and ppl gates are on record.";

const PF_ATTN_MK_ROWS_PER_DISPATCH_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS: usize = 8;

fn pf_attn_mk_enabled() -> bool {
    std::env::var(PF_ATTN_MK_ENV_DEFAULT_OFF_UNTIL_THE_LADDER_AND_PPL_GATE_ARE_ON_RECORD)
        .ok()
        .as_deref()
        == Some("1")
}

const PF_ATTN_MK_QCAST_WGSL: &str = "
struct QcParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
};
@group(0) @binding(0) var<storage, read> qc_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> qc_y: array<f32>;
@group(0) @binding(2) var<uniform> qc_p: QcParams;
@compute @workgroup_size(256)
fn q3w_pf_attn_mk_qcast(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= qc_p.n_words) {
        return;
    }
    let x = qc_x[w];
    let i = w * 2u;
    qc_y[i] = bitcast<f32>((x & 0xffffu) << 16u) * qc_p.scale;
    qc_y[i + 1u] = bitcast<f32>((x >> 16u) << 16u) * qc_p.scale;
}
";

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfAttnMkQcastParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    scale: f32,
}

pub const PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM: &str =
    "NV_Q3D_PF_ATTN_TILED";

pub const PF_ATTN_TILED_REUSES_THE_MOE_TILED_FLASH_FAMILY_OVER_A_PF_FP8_KV_QUANTIZE: &str =
    "The M-row prefill attention routes through the qwen3_5_moe_wgpu tiled flash family by \
     default (q3w_pf_flash1_fp8kv_tiled_slotml_* + flash_splitk_stage2_mk): one fp8 KV \
     stream serves 32 rows at 4 rows per warp instead of the mk pair's one bf16 stream per 8 rows \
     with a workgroup collective per (row, position). The chunk's K/V rows are quantized into the \
     fp8 e4m3 cache by q3w_pf_quantize_kv_fp8_m before the flash dispatch, which also halves the \
     KV bytes streamed at depth versus bf16; the M=1 stepping path keeps the same fp8 cache in \
     sync via the kv_fp8 quantize twin so chunks may follow single steps, while M=1 decode keeps \
     reading the bf16 cache bit-identically unless NV_Q3D_KV_FP8 asks for the fp8 decode route. \
     The tiled fd uniform always views the whole baked chunk (m_rows = m, total = base + m) so \
     every row's causal end is base+row+1 and the split scratch never aliases across rows; rows \
     past m_live compute dead finite values into dead a_attn rows, exactly the class of stale \
     garbage the scores arm already leaves there. The full default stack's same-box ladder reads \
     faster prefill at every depth with same-session ppl held (perf/runs.jsonl ladder); \
     NV_Q3D_PF_ATTN_TILED=0 restores the \
     q3w_attn_decode_m scores-slab arm at that cost, and NV_Q3D_PF_ATTN_MK=1 selects the mk arm \
     instead of the tiled one.";

fn pf_attn_tiled_enabled() -> bool {
    std::env::var(PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM)
        .ok()
        .as_deref()
        != Some("0")
        && !pf_attn_mk_enabled()
}

fn host_pf_projections_ride_the_coop_route(h: &HostDenseWeights) -> bool {
    let fp8_16aligned = |f: &Option<HostFp8Lin>| {
        f.as_ref()
            .is_some_and(|f| f.n.is_multiple_of(16) && f.k.is_multiple_of(16))
    };
    h.layers.iter().all(|l| {
        let mixer_ok = match &l.mixer {
            HostDenseMixer::Delta(_) => {
                fp8_16aligned(&l.delta_fp8.qkv)
                    && fp8_16aligned(&l.delta_fp8.z)
                    && fp8_16aligned(&l.delta_fp8.out)
            }
            HostDenseMixer::Attn(a) => [&a.q, &a.k, &a.v, &a.o]
                .iter()
                .all(|w| !matches!(w, HostDenseLin::Bf16(_))),
        };
        mixer_ok
            && [&l.mlp.gate, &l.mlp.up, &l.mlp.down]
                .iter()
                .all(|w| !matches!(w, HostDenseLin::Bf16(_)))
    })
}

fn pf_attn_tiled_stage1_source_exact_decode_because_the_sd_twin_measured_a_1pct_null_here(
    subgroup: bool,
) -> String {
    crate::qwen3_5_moe_wgpu::pf_flash_tiled_source(subgroup)
}

const PF_ATTN_TILED_ENTRY_IS_THE_MOE_DEFAULT_SLOTML_ARM_SG_THEN_WG: [&str; 2] = [
    "q3w_pf_flash1_fp8kv_tiled_slotml_sg",
    "q3w_pf_flash1_fp8kv_tiled_slotml_wg",
];

const PF_ATTN_TILED_STAGE_ARRAYS_HOLD_FDT_POS_8_TIMES_HEAD_DIM_F32: usize = 2048;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfKvqMParams {
    tokens: u32,
    x_stride_elems: u32,
    pad0: u32,
    pad1: u32,
}

fn pf_coop_active(ctx: &WgpuContext) -> bool {
    if !pf_coop_requested() {
        return false;
    }
    static OK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *OK.get_or_init(
        || match wk::gemm_coop_f16::select(ctx, wk::gemm_coop_f16::Operand::F16) {
            Ok(g) if g.tile == 16 => true,
            Ok(g) => {
                eprintln!(
                    "[q3d-wgpu] the default coop prefill \
                     ({PF_COOP_ENV_DEFAULT_ON_SET_0_FOR_THE_LEGACY_M16_GEMM_ARM}) needs 16x16 f16 \
                     fragments for the wq16 staging unpack but the adapter serves only {t}x{t}; \
                     keeping the m<=16 gemm path",
                    t = g.tile
                );
                false
            }
            Err(why) => {
                eprintln!(
                    "[q3d-wgpu] the default coop prefill \
                     ({PF_COOP_ENV_DEFAULT_ON_SET_0_FOR_THE_LEGACY_M16_GEMM_ARM}) is unavailable: \
                     {why}; keeping the m<=16 gemm path"
                );
                false
            }
        },
    )
}

pub const PF_SCAN_WY_ENV_DEFAULT_ON_SET_0_FOR_THE_TOKEN_SERIAL_SCAN: &str = "NV_Q3D_PF_SCAN_WY";

pub const PF_SCAN_WY_SOLVES_32_TOKEN_SUBCHUNKS_AS_A_UT_TRANSFORM_INSTEAD_OF_TOKEN_SERIAL: &str =
    "The prefill DeltaNet scan routes through q3w_delta_scan_wy by default \
     (NV_Q3D_PF_SCAN_WY=0 restores the token-serial q3w_delta_scan), which \
     rewrites each 32-token sub-chunk of the sequential recurrence as (I + A) U = diag(beta) \
     (V - decayed K S_prev) solved by forward substitution, with outputs and the state update \
     as chunk-level rank-32 products: the 64 KB per-head f32 state is touched 2 reads + 1 \
     read-modify-write per chunk instead of 2 read-modify-writes per token, and the \
     2-barriers-per-token chain drops to ~7 barriers per chunk. Four workgroups per head each \
     own 32 v-columns (grid y), with all hot accumulators in explicitly-indexed vec4 registers \
     because dynamically-indexed private arrays spill to local memory (the array-accumulator \
     draft was an order of magnitude slower on the same prefill; runs.jsonl has the rate). \
     The kernel is dk=dv=128 only; any other geometry keeps q3w_delta_scan. The f64 algebra \
     gate and the 1e-4 reassociation tolerance model live in tests/q3d_delta_scan_wy.rs";

const PF_SCAN_WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS: u32 = 4;

fn pf_scan_wy_route(d_k: usize, d_v: usize) -> (&'static str, u32) {
    if d_k == 128
        && d_v == 128
        && std::env::var(PF_SCAN_WY_ENV_DEFAULT_ON_SET_0_FOR_THE_TOKEN_SERIAL_SCAN)
            .ok()
            .as_deref()
            != Some("0")
    {
        (
            "q3w_delta_scan_wy",
            PF_SCAN_WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS,
        )
    } else {
        ("q3w_delta_scan", 1)
    }
}

pub const PF_ATTN_SCORES_SLAB_IS_M_X_NHEADS_X_MAXSEQ_F32_SO_PF_M_CLAMPS_TO_THIS_BUDGET_BYTES:
    u64 = 1 << 31;

pub const SMALL_M_GRAPHS_ROUTE_TO_THE_LEGACY_ARMS_BECAUSE_COOP_TILED_AND_WY_ARE_SIZED_FOR_CHUNK_ROWS:
    &str = "a pf graph built with m <= 8 is a verify/MTP graph (MTP builds m = k+1, the serving \
     verify graph is k+1 rows), not a chunk-prefill graph; the coop projection GEMM, the tiled \
     fp8 flash prefill attention and the wy DeltaNet scan are sized for chunk row counts and \
     measured drastically slower on M-row rounds at these row counts (mtpseam ladder: \
     perf/runs.jsonl), and the legacy arms -- the m<=16 gemm, the scores-slab attention, the \
     token-serial scan -- were measured bit-identical to the default stack there. The build \
     therefore routes THIS graph onto the legacy arms while any m > 8 graph keeps the shipped \
     default stack untouched; an explicit NV_Q3D_PF_COOP=1 / NV_Q3D_PF_ATTN_TILED=1 / \
     NV_Q3D_PF_SCAN_WY=1 still selects the chunk arm for a small graph. Under fp8-KV decode the \
     non-tiled arms record their own q3w_pf_quantize_kv_fp8_m pair, so chunk rows never leave \
     the fp8 cache empty on any route.";

const SMALL_M_LEGACY_ROUTE_BOUND_VERIFY_GRAPHS_BUILD_M_LE_8: usize = 8;

fn pf_small_m_legacy(m: usize) -> bool {
    (2..=SMALL_M_LEGACY_ROUTE_BOUND_VERIFY_GRAPHS_BUILD_M_LE_8).contains(&m)
}

fn env_explicit_1(name: &str) -> bool {
    std::env::var(name).ok().as_deref() == Some("1")
}

#[derive(Clone)]
pub struct HostDenseLayer {
    pub input_ln: Vec<u16>,
    pub post_attn_ln: Vec<u16>,
    pub mixer: HostDenseMixer,
    pub mlp: HostDenseMlp,
    pub delta_fp8: DeltaFp8,
}

pub struct HostDenseWeights {
    pub embed: Vec<u16>,
    pub final_norm: Vec<u16>,
    pub lm_head: Vec<u16>,
    pub layers: Vec<HostDenseLayer>,
}

pub enum WeightSource<'a> {
    Host(&'a HostDenseWeights),
    Loader(&'a nv_weights::WeightLoader),
}

fn bf16_bits(x: f32) -> u16 {
    half::bf16::from_f32(x).to_bits()
}

fn bf16_val(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

fn pack_pairs(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len().div_ceil(2).max(1)];
    for (i, v) in src.iter().enumerate() {
        out[i / 2] |= (*v as u32) << (16 * (i % 2));
    }
    out
}

fn pack_f16_pairs_from_bf16(src: &[u16]) -> Vec<u32> {
    let mut out = vec![0u32; src.len().div_ceil(2).max(1)];
    for (i, v) in src.iter().enumerate() {
        let h = half::f16::from_f32(bf16_val(*v)).to_bits() as u32;
        out[i / 2] |= h << (16 * (i % 2));
    }
    out
}

pub fn rope_tables(rotary_dim: usize, theta: f32, rows: usize) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0f32 / theta.powf((i as f32 * 2.0) / rotary_dim as f32))
        .collect();
    crate::gemma4_wgpu_shared::rope_tables_from_inv_freq(&inv_freq, rows)
}

pub use crate::embed_row_splice::EmbedRowSplice as ImageRowSplice;

struct ChunkRowSplice<'a> {
    rel_pos: usize,
    row_words: &'a [u32],
}

struct Pass {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    bind: wgpu::BindGroup,
    grid: (u32, u32, u32),
    label: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum VerifyTail {
    ArgmaxRows,
    ArgmaxAndLogitRows,
    SkippedBecauseCommitOnlyCallersAlreadyKnowEveryToken,
}

pub const VERIFY_CHAIN_COMMITS_BY_SNAPSHOT_AND_M1_REPLAY_BECAUSE_DELTANET_STATE_IS_RECURRENT_NOT_POSITION_MASKED: &str = "gemma4_e4b_wgpu::advance moves pos only because every one of its mixers is attention-KV: rows written past the accepted prefix are masked out by the attention total. Qwen3.5/3.8 carry a DeltaNet recurrent state and a causal short conv whose buffers hold no position, so a partially accepted chain cannot be repaired by moving pos. verify_chain therefore copies every recurrent and conv state buffer to a rollback buffer before the M-row forward; advance(n) commits in place only when n covers every row the forward advanced, and otherwise restores the snapshot and replays the accepted prefix through the M=1 stepping path, which is the definition of the stream this decoder must stay bit-identical to.";

struct Verify {
    rows: usize,
    res2_rows: wgpu::Buffer,
    resadd: Pass,
    tok: wgpu::Buffer,
    row_logits: Option<wgpu::Buffer>,
    rollback: Vec<wgpu::Buffer>,
    validated: bool,
    pending: Option<Vec<u32>>,
}

pub fn prefill_m() -> usize {
    let cap = if pf_coop_requested() { 256 } else { 16 };
    match std::env::var("NV_WGPU_PREFILL_M")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(0) | Some(1) => 0,
        Some(m) => m.clamp(2, cap),
        None => cap,
    }
}

fn na_enabled() -> bool {
    matches!(
        std::env::var("NV_WGPU_NA").ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

struct PfCoop {
    tm: u32,
    tn: u32,
    sg: u32,
    m_alloc: u32,
    x_f16: wgpu::Buffer,
    y_f32: wgpu::Buffer,
    zero: wgpu::Buffer,
    helpers: String,
}

const PF_COOP_HELPERS_WGSL: &str = r#"
struct Q3cxParams {
    k_elems: u32,
    x_stride_words: u32,
    rows_in: u32,
    rows_out: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

struct Q3pkParams {
    n_cols: u32,
    y_src_stride: u32,
    y_dst_stride_words: u32,
    rows: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    alpha: f32,
};

@group(0) @binding(0) var<storage, read> pcx_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> pcx_out: array<f16>;
@group(0) @binding(2) var<uniform> pcx_p: Q3cxParams;

@compute @workgroup_size(256)
fn q3w_pf_coop_cast_x(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let i = (wid.x + wid.y * pcx_p.groups_x) * 256u + lid.x;
    let kw = pcx_p.k_elems >> 1u;
    if (i >= pcx_p.rows_out * kw) {
        return;
    }
    let row = i / kw;
    let c = i - row * kw;
    var lo = 0.0;
    var hi = 0.0;
    if (row < pcx_p.rows_in) {
        let v = pcx_x[row * pcx_p.x_stride_words + c];
        lo = bf16_lo(v);
        hi = bf16_hi(v);
    }
    pcx_out[2u * i] = f16(lo);
    pcx_out[2u * i + 1u] = f16(hi);
}

@group(0) @binding(4) var<storage, read> ppk_y: array<f32>;
@group(0) @binding(5) var<storage, read_write> ppk_out: array<u32>;
@group(0) @binding(6) var<uniform> ppk_p: Q3pkParams;

@compute @workgroup_size(256)
fn q3w_pf_coop_pack_y(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let i = (wid.x + wid.y * ppk_p.groups_x) * 256u + lid.x;
    let nw = ppk_p.n_cols >> 1u;
    if (i >= ppk_p.rows * nw) {
        return;
    }
    let row = i / nw;
    let c = i - row * nw;
    let a = ppk_y[row * ppk_p.y_src_stride + 2u * c] * ppk_p.alpha;
    let b = ppk_y[row * ppk_p.y_src_stride + 2u * c + 1u] * ppk_p.alpha;
    ppk_out[row * ppk_p.y_dst_stride_words + c] = bf16_pack(a, b);
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CoopCastParams {
    k_elems: u32,
    x_stride_words: u32,
    rows_in: u32,
    rows_out: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CoopPackParams {
    n_cols: u32,
    y_src_stride: u32,
    y_dst_stride_words: u32,
    rows: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    alpha: f32,
}

struct PfAttnMk {
    q_f32: wgpu::Buffer,
    scratch: wgpu::Buffer,
    fd: Vec<wgpu::Buffer>,
}

struct PfAttnTiled {
    q_f32: wgpu::Buffer,
    scratch: wgpu::Buffer,
    fd: wgpu::Buffer,
    start: wgpu::Buffer,
}

struct Pf {
    m: usize,
    attn_mk: Option<PfAttnMk>,
    attn_tiled: Option<PfAttnTiled>,
    attn_kvq_fd_start: Option<(wgpu::Buffer, wgpu::Buffer)>,
    na: Option<std::sync::Arc<wgpu::ComputePipeline>>,
    coop: Option<PfCoop>,
    gemm_src: String,
    gemm_entry: String,
    gemm_i8_src: String,
    ck: wgpu::Buffer,
    tok: wgpu::Buffer,
    splice: wgpu::Buffer,
    mask: wgpu::Buffer,
    res: wgpu::Buffer,
    normed: wgpu::Buffer,
    mix: wgpu::Buffer,
    mlp_out: wgpu::Buffer,
    normed_post: wgpu::Buffer,
    d_qkv: wgpu::Buffer,
    d_z: wgpu::Buffer,
    d_ab: wgpu::Buffer,
    d_mixed: wgpu::Buffer,
    d_q: wgpu::Buffer,
    d_k: wgpu::Buffer,
    d_v: wgpu::Buffer,
    d_g: wgpu::Buffer,
    d_beta: wgpu::Buffer,
    d_core: wgpu::Buffer,
    d_gated: wgpu::Buffer,
    a_qraw: wgpu::Buffer,
    a_kraw: wgpu::Buffer,
    a_vraw: wgpu::Buffer,
    a_q: wgpu::Buffer,
    a_k: wgpu::Buffer,
    a_scores: wgpu::Buffer,
    a_attn: wgpu::Buffer,
    a_gated: wgpu::Buffer,
    m_ygate: wgpu::Buffer,
    m_yup: wgpu::Buffer,
    m_act: wgpu::Buffer,
}

struct Builder {
    core: crate::wgpu_ledger::VramLedger,
    passes: Vec<Pass>,
    pf_passes: Vec<Pass>,
}

impl std::ops::Deref for Builder {
    type Target = crate::wgpu_ledger::VramLedger;
    fn deref(&self) -> &crate::wgpu_ledger::VramLedger {
        &self.core
    }
}

impl std::ops::DerefMut for Builder {
    fn deref_mut(&mut self) -> &mut crate::wgpu_ledger::VramLedger {
        &mut self.core
    }
}

impl Builder {
    fn make(
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
            label: if dispatch::profile::enabled() {
                format!("{label}:{entry}")
            } else {
                String::new()
            },
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
        let pass = self.make(label, source, entry, binds, grid)?;
        self.passes.push(pass);
        Ok(())
    }

    fn push_pf(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        if std::env::var("NV_Q3D_DUMP_PF_PASSES").as_deref() == Ok("1") {
            eprintln!(
                "[pf-pass {:>3}] {label} :: {entry} grid={grid:?}",
                self.pf_passes.len()
            );
        }
        let pass = self.make(label, source, entry, binds, grid)?;
        self.pf_passes.push(pass);
        Ok(())
    }

    fn push_pf_off(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &wgpu::Buffer, u64)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, label, source, entry)
            .map_err(|e| anyhow::anyhow!("pipeline {label}::{entry}: {e}"))?;
        let bind = dispatch::bind_group_offsets(self.ctx, &pipeline, binds);
        self.pf_passes.push(Pass {
            pipeline,
            bind,
            grid,
            label: if dispatch::profile::enabled() {
                format!("{label}:{entry}")
            } else {
                String::new()
            },
        });
        Ok(())
    }

    fn push_pf_pipeline(
        &mut self,
        label: &str,
        pipeline: std::sync::Arc<wgpu::ComputePipeline>,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) {
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        self.pf_passes.push(Pass {
            pipeline,
            bind,
            grid,
            label: if dispatch::profile::enabled() {
                label.to_string()
            } else {
                String::new()
            },
        });
    }

}

struct Sources {
    gemv_bf16: String,
    gemv_i8: String,
    gemv_nvfp4: String,
    quant: String,
    delta: String,
    attn: String,
    flash: String,
    kvq: String,
    kvq4: String,
    misc: String,
    rms: String,
    rmsres: String,
    resscale: String,
    prefill: String,
}

#[doc(hidden)]
pub fn nozi_audit_sources() -> Vec<(&'static str, String)> {
    vec![
        ("q3d:gemv_bf16", compose(GEMV_BF16_WGSL)),
        (
            "q3d:delta",
            compose(&format!(
                "{}\n{}",
                DELTA_WGSL,
                delta_recurrent_variants_source()
            )),
        ),
        ("q3d:attn", compose(ATTN_WGSL)),
        ("q3d:misc", compose(MISC_WGSL)),
    ]
}

#[doc(hidden)]
pub fn shipped_gemv_i8_source() -> String {
    compose(GEMV_I8_WGSL)
}

#[doc(hidden)]
pub fn shipped_delta_source() -> String {
    compose(DELTA_WGSL)
}

#[doc(hidden)]
pub fn shipped_prefill_source() -> String {
    prefill_composed()
}

#[doc(hidden)]
pub fn shipped_prefill_gemm(m: usize) -> (String, String) {
    (gemm_mk_source(m), gemm_mk_entry(m))
}

#[doc(hidden)]
pub fn shipped_prefill_gemm_i8(m: usize) -> (String, String, String) {
    (
        gemm_i8_mk_source(m),
        gemm_i8_mk_entry(m, true),
        gemm_i8_mk_entry(m, false),
    )
}

impl Sources {
    fn new() -> Self {
        Self {
            gemv_bf16: compose(GEMV_BF16_WGSL),
            gemv_i8: compose(GEMV_I8_WGSL),
            gemv_nvfp4: nvfp4_gemv_source(),
            quant: nvfp4_quant_source(),
            delta: compose(&format!(
                "{}\n{}",
                DELTA_WGSL,
                delta_recurrent_variants_source()
            )),
            attn: compose(ATTN_WGSL),
            flash: compose(wk::flash_decode::WGSL),
            kvq: compose(wk::kv_fp8::WGSL),
            kvq4: compose(wk::kv_nvfp4::WGSL),
            misc: compose(MISC_WGSL),
            rms: compose(wk::rmsnorm::WGSL),
            rmsres: compose(wk::rmsnorm_residual::WGSL),
            resscale: compose(wk::residual_scale::WGSL),
            prefill: prefill_composed(),
        }
    }
}

struct Bf16Gpu {
    w: wgpu::Buffer,
    n: usize,
    k: usize,
}

struct Nvfp4Gpu {
    w: wgpu::Buffer,
    scales: wgpu::Buffer,
    scales_linear: Option<wgpu::Buffer>,
    alpha: f32,
    input_global: f32,
    n: usize,
    k: usize,
}

struct I8Gpu {
    w: wgpu::Buffer,
    s: wgpu::Buffer,
    n: usize,
    k: usize,
    group: usize,
}

enum DenseLinGpu {
    Bf16(Bf16Gpu),
    Nvfp4(Nvfp4Gpu),
    Int8(I8Gpu),
    Fp8(Fp8Gpu, Option<Bf16Gpu>),
}

impl DenseLinGpu {
    fn n(&self) -> usize {
        match self {
            Self::Bf16(w) => w.n,
            Self::Nvfp4(w) => w.n,
            Self::Int8(w) => w.n,
            Self::Fp8(w, _) => w.n,
        }
    }

    fn k(&self) -> usize {
        match self {
            Self::Bf16(w) => w.k,
            Self::Nvfp4(w) => w.k,
            Self::Int8(w) => w.k,
            Self::Fp8(w, _) => w.k,
        }
    }

    fn input_global(&self) -> Option<f32> {
        match self {
            Self::Bf16(_) | Self::Int8(_) | Self::Fp8(..) => None,
            Self::Nvfp4(w) => Some(w.input_global),
        }
    }
}

struct QuantIn {
    xq: wgpu::Buffer,
    xs: wgpu::Buffer,
    sel: wgpu::Buffer,
    alpha_dummy: wgpu::Buffer,
    xm: Option<wgpu::Buffer>,
}

pub const MTP_HEAD_CONVENTIONS_ZERO_CENTERED_NORMS_EMB_FIRST_FC_AND_SHIFT_BY_ONE_KV_WITH_A_ZERO_HIDDEN_AT_POS_0:
    &str =
    "the qwen3.8 MTP head stores every RMSNorm weight zero-centered (loaded as w+1 through the \
     trunk norm_plus_one route), fuses concat(rmsnorm(embed[token]), rmsnorm(trunk_hidden)) with \
     the EMBEDDING half first through fc, and fills its drafter-owned KV shifted by one: row p is \
     built from (token_p, trunk_hidden_{p-1}) with a zero hidden standing in at p=0. The drafter \
     KV is plain positional GQA attention, so rewind is a host-side length, never a snapshot; the \
     trunk's DeltaNet snapshot rollback is untouched and every emitted token stays gated by the \
     trunk verify argmax, so an attached MTP head can only change speed, never the served stream.";

pub struct MtpHostWeights {
    pub pre_fc_norm_embedding: Vec<u16>,
    pub pre_fc_norm_hidden: Vec<u16>,
    pub fc: HostBf16Lin,
    pub input_ln: Vec<u16>,
    pub attn: HostDenseAttention,
    pub post_attn_ln: Vec<u16>,
    pub mlp: HostDenseMlp,
    pub final_norm: Vec<u16>,
}

struct MtpDraft {
    passes: Vec<Pass>,
    kv_end: usize,
    pos_buf: wgpu::Buffer,
    fd_buf: wgpu::Buffer,
    hid: wgpu::Buffer,
    len: usize,
    round_base: Option<usize>,
    validated_kv: bool,
    validated_full: bool,
    _buffers: Vec<wgpu::Buffer>,
}

#[derive(Clone, Copy)]
enum MtpHid {
    Keep,
    Zero,
    VerifyRow(usize),
}

#[derive(Clone, Copy)]
enum MtpAfter {
    Nothing,
    HidFromRes2,
    HidFromVerifyRow(usize),
}

pub struct Qwen3_5DenseWgpu {
    ctx: &'static WgpuContext,
    config: Qwen3_5DenseConfig,
    max_seq: usize,
    pos: usize,
    validated: bool,
    prefix_validated: bool,
    pf_validated: bool,
    passes: Vec<Pass>,
    pf_passes: Vec<Pass>,
    pf_m: usize,
    pf_tok: Option<wgpu::Buffer>,
    pf_ck: Option<wgpu::Buffer>,
    pf_splice: Option<wgpu::Buffer>,
    pf_mask: Option<wgpu::Buffer>,
    pf_attn_mk_fd: Vec<wgpu::Buffer>,
    pf_chunk_fd_and_start: Option<(wgpu::Buffer, wgpu::Buffer)>,
    pf_res: Option<wgpu::Buffer>,
    pf_embed_end: usize,
    head_start: usize,
    final_start: usize,
    verify: Option<Verify>,
    _buffers: Vec<wgpu::Buffer>,
    tok_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    fd_buf: wgpu::Buffer,
    fd_base: FdParams,
    res: wgpu::Buffer,
    res2: wgpu::Buffer,
    final_x: wgpu::Buffer,
    embed_gather_end: usize,
    mtp: Option<MtpDraft>,
    mtp_replay: bool,
    token_out: wgpu::Buffer,
    logits: wgpu::Buffer,
    state_buffers: Vec<(wgpu::Buffer, u64)>,
    recurrent_states: Vec<(wgpu::Buffer, u64)>,
    rope_row_buffers: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    rope_std_host_cos_sin: (Vec<f32>, Vec<f32>),
    mrope_rows_installed: bool,
    vocab: usize,
    vram: VramReport,
    preenc: bool,
    pending_cb: Option<wgpu::CommandBuffer>,
    staged_read: bool,
    tok_stage: wgpu::Buffer,
}

pub fn vram_report_enabled() -> bool {
    crate::wgpu_ledger::vram_report_var_enabled("NV_QWEN35_DENSE_WGPU_VRAM")
}

const PREENC_ENV_A_CB_ENCODED_LAST_STEP_READS_THIS_STEPS_UNIFORMS_BECAUSE_QUEUE_WRITE_BUFFER_ORDERS_BEFORE_EVERY_LATER_SUBMIT: &str =
    "NV_QWEN35_DENSE_WGPU_PREENC";

fn preenc_enabled() -> bool {
    std::env::var(PREENC_ENV_A_CB_ENCODED_LAST_STEP_READS_THIS_STEPS_UNIFORMS_BECAUSE_QUEUE_WRITE_BUFFER_ORDERS_BEFORE_EVERY_LATER_SUBMIT)
        .ok()
        .as_deref()
        != Some("0")
}

const STAGED_READ_ENV_PERSISTENT_MAP_READ_STAGE_SKIPS_PER_STEP_STAGING_ALLOC_AND_A_SECOND_SUBMIT: &str =
    "NV_QWEN35_DENSE_WGPU_STAGED_READ";

fn staged_read_enabled() -> bool {
    std::env::var(STAGED_READ_ENV_PERSISTENT_MAP_READ_STAGE_SKIPS_PER_STEP_STAGING_ALLOC_AND_A_SECOND_SUBMIT)
        .ok()
        .as_deref()
        != Some("0")
}

const QKV_GATED_ENV_FUSION_IS_BIT_IDENTICAL_BY_SHARED_LANE_FNS_BUT_UNMEASURED_SO_DEFAULT_OFF:
    &str = "NV_QWEN35_DENSE_WGPU_QKV_GATED";

fn qkv_gated_enabled() -> bool {
    matches!(
        std::env::var(QKV_GATED_ENV_FUSION_IS_BIT_IDENTICAL_BY_SHARED_LANE_FNS_BUT_UNMEASURED_SO_DEFAULT_OFF)
            .ok()
            .as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

pub const FUSE_DN_ENV_SPLIT_GATING_RECURRENT_OUT_ARE_ONE_DISPATCH_PER_DN_LAYER_BIT_IDENTICAL_DEFAULT_ON_SET_0_FOR_THE_CHAIN:
    &str = "NV_Q3D_FUSE_DN";

pub const FUSE_ATTN_ENV_QNORM_KNORM_QCAST_ARE_ONE_DISPATCH_PER_ATTN_LAYER_BIT_IDENTICAL_DEFAULT_ON_SET_0_FOR_THE_CHAIN:
    &str = "NV_Q3D_FUSE_ATTN";

pub const FUSE_DN_GEMV_ENV_QKV_Z_AB_PROJECTIONS_ARE_ONE_DISPATCH_PER_DN_LAYER_BIT_IDENTICAL_DEFAULT_ON_SET_0_FOR_THE_CHAIN:
    &str = "NV_Q3D_FUSE_DN_GEMV";

fn fuse_env_on(name: &str) -> bool {
    !matches!(
        std::env::var(name).ok().as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

fn fuse_dn_enabled() -> bool {
    fuse_env_on(FUSE_DN_ENV_SPLIT_GATING_RECURRENT_OUT_ARE_ONE_DISPATCH_PER_DN_LAYER_BIT_IDENTICAL_DEFAULT_ON_SET_0_FOR_THE_CHAIN)
}

fn fuse_attn_enabled() -> bool {
    fuse_env_on(FUSE_ATTN_ENV_QNORM_KNORM_QCAST_ARE_ONE_DISPATCH_PER_ATTN_LAYER_BIT_IDENTICAL_DEFAULT_ON_SET_0_FOR_THE_CHAIN)
}

fn fuse_dn_gemv_enabled() -> bool {
    fuse_env_on(FUSE_DN_GEMV_ENV_QKV_Z_AB_PROJECTIONS_ARE_ONE_DISPATCH_PER_DN_LAYER_BIT_IDENTICAL_DEFAULT_ON_SET_0_FOR_THE_CHAIN)
}

pub const FUSE_MLP_ENV_SILU_AND_DOWN_QUANT_ARE_ONE_DISPATCH_PER_LAYER_BIT_IDENTICAL_DEFAULT_ON_SET_0_FOR_THE_CHAIN:
    &str = "NV_Q3D_FUSE_MLP";

fn fuse_mlp_enabled() -> bool {
    fuse_env_on(FUSE_MLP_ENV_SILU_AND_DOWN_QUANT_ARE_ONE_DISPATCH_PER_LAYER_BIT_IDENTICAL_DEFAULT_ON_SET_0_FOR_THE_CHAIN)
}

pub const FUSE_MLP_GEMV_ENV_GATE_AND_UP_NVFP4_GEMVS_ARE_ONE_MROW2_2W_DISPATCH_PER_MLP_BIT_IDENTICAL_DEFAULT_OFF_SET_1_TO_ENGAGE:
    &str = "NV_Q3D_FUSE_MLP_GEMV";

fn fuse_env_opt_in(name: &str) -> bool {
    matches!(
        std::env::var(name).ok().as_deref(),
        Some("1") | Some("on") | Some("true")
    )
}

fn fuse_mlp_gemv_enabled() -> bool {
    fuse_env_opt_in(FUSE_MLP_GEMV_ENV_GATE_AND_UP_NVFP4_GEMVS_ARE_ONE_MROW2_2W_DISPATCH_PER_MLP_BIT_IDENTICAL_DEFAULT_OFF_SET_1_TO_ENGAGE)
}

pub const FUSE_KVW_ENV_KV_WRITE_AND_THE_FP8_KV_QUANT_PAIR_ARE_ONE_DISPATCH_PER_ATTN_LAYER_BIT_IDENTICAL_DEFAULT_OFF_SET_1_TO_ENGAGE:
    &str = "NV_Q3D_FUSE_KVW";

fn fuse_kvw_enabled() -> bool {
    fuse_env_opt_in(FUSE_KVW_ENV_KV_WRITE_AND_THE_FP8_KV_QUANT_PAIR_ARE_ONE_DISPATCH_PER_ATTN_LAYER_BIT_IDENTICAL_DEFAULT_OFF_SET_1_TO_ENGAGE)
}

const MLP_GEMV_2W_ROUTE_IS_EXACTLY_THE_MROW2_DECODE_ROUTE_ANY_OTHER_ENTRY_KEEPS_THE_PAIR: &str =
    "q3w_gemv_nvfp4_mrow2";

pub const MLP_GEMV_2W_GRID_FUSES_ON_WID_Y_BECAUSE_REGISTER_FUSING_BOTH_WEIGHTS_PER_SUBGROUP_MEASURED_SLOWER_THAN_THE_SPLIT_PAIR:
    &str =
    "q3w_gemv_nvfp4_mrow2_2w keeps mrow2's exact two-row working set per subgroup and selects \
     gate vs up by grid y: mrow2 sits near the stream ceiling, so a draft that walked both \
     weight streams in one subgroup serialized two dependency chains and measured slower than \
     the two independent dispatches it replaced, while this form measures level with them and \
     still drops one dispatch per nvfp4 MLP layer (fused2 rows: perf/runs.jsonl); back-to-back \
     independent gemv passes in one command buffer already overlap on this adapter, which is \
     why the dispatch-count drop alone is a null";

const MLP_GEMV_2W_BINDS_TEN_STORAGE_BUFFERS: u32 = 10;

const KVW_FUSED_BINDS_NINE_STORAGE_BUFFERS: u32 = 9;

const DN_MERGED_GEMV_BINDS_NINE_STORAGE_BUFFERS: u32 = 9;

const ATTN_FUSED_NORM_BINDS_TEN_STORAGE_BUFFERS: u32 = 10;

const DELTA_HEAD_FUSED_WG128_COVERS_ONE_LANE_PER_DV_AND_ONE_REDUCTION_SLOT_PER_DK: usize = 128;

impl Qwen3_5DenseWgpu {
    pub fn config(&self) -> &Qwen3_5DenseConfig {
        &self.config
    }

    pub fn vram_report(&self) -> &VramReport {
        &self.vram
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn reset(&mut self) -> Result<()> {
        self.pos = 0;
        if let Some(v) = self.verify.as_mut() {
            v.pending = None;
        }
        for (buf, bytes) in &self.state_buffers {
            let zeros = vec![0u8; *bytes as usize];
            self.ctx.queue.write_buffer(buf, 0, &zeros);
        }
        self.restore_text_rope_rows();
        if let Some(m) = self.mtp.as_mut() {
            m.len = 0;
            m.round_base = None;
        }
        Ok(())
    }

    pub fn rope_rot_half(&self) -> usize {
        self.config.rotary_dim().max(2) / 2
    }

    pub fn install_mrope_rows_for_prompt_and_shift_continuation_rows_by_delta(
        &mut self,
        pos: &crate::qwen3_mm_splice::Qwen3MropePositions,
        section: [usize; 3],
    ) -> Result<()> {
        let rh = self.rope_rot_half();
        anyhow::ensure!(
            section.iter().sum::<usize>() == rh,
            "mrope section {section:?} must tile the {rh} rotary half-frequencies of this graph"
        );
        let n = pos.len();
        anyhow::ensure!(
            n > 0 && n <= self.max_seq,
            "mrope prompt of {n} tokens does not fit max_seq {}",
            self.max_seq
        );
        let (std_cos, std_sin) = &self.rope_std_host_cos_sin;
        let mut cos = vec![0f32; self.max_seq * rh];
        let mut sin = vec![0f32; self.max_seq * rh];
        for i in 0..n {
            for j in 0..rh {
                let axis = crate::qwen3_mm_splice::interleaved_axis_of_half_freq_matching_hf_overwrite_semantics(j, section);
                let p = match axis {
                    1 => pos.h[i],
                    2 => pos.w[i],
                    _ => pos.t[i],
                } as usize;
                anyhow::ensure!(
                    p < self.max_seq,
                    "mrope axis-{axis} position {p} at token {i} exceeds the {}-row rope table",
                    self.max_seq
                );
                cos[i * rh + j] = std_cos[p * rh + j];
                sin[i * rh + j] = std_sin[p * rh + j];
            }
        }
        let delta = pos.delta_added_to_token_index_for_every_position_after_this_prefill;
        for i in n..self.max_seq {
            let p = i as i64 + delta;
            anyhow::ensure!(
                (0..self.max_seq as i64).contains(&p),
                "mrope continuation row {i} shifts to {p}, outside the {}-row rope table",
                self.max_seq
            );
            let p = p as usize;
            cos[i * rh..(i + 1) * rh].copy_from_slice(&std_cos[p * rh..(p + 1) * rh]);
            sin[i * rh..(i + 1) * rh].copy_from_slice(&std_sin[p * rh..(p + 1) * rh]);
        }
        for (cb, sb) in &self.rope_row_buffers {
            self.ctx.queue.write_buffer(cb, 0, bytemuck::cast_slice(&cos));
            self.ctx.queue.write_buffer(sb, 0, bytemuck::cast_slice(&sin));
        }
        self.mrope_rows_installed = true;
        Ok(())
    }

    pub fn restore_text_rope_rows(&mut self) {
        if !self.mrope_rows_installed {
            return;
        }
        let (std_cos, std_sin) = &self.rope_std_host_cos_sin;
        for (cb, sb) in &self.rope_row_buffers {
            self.ctx
                .queue
                .write_buffer(cb, 0, bytemuck::cast_slice(std_cos));
            self.ctx
                .queue
                .write_buffer(sb, 0, bytemuck::cast_slice(std_sin));
        }
        self.mrope_rows_installed = false;
    }

    pub fn mrope_rows_installed(&self) -> bool {
        self.mrope_rows_installed
    }

    pub fn read_rope_rows_for_test(&self, layer: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let (cb, sb) = self
            .rope_row_buffers
            .get(layer)
            .ok_or_else(|| anyhow::anyhow!("no attention layer {layer} with rope rows"))?;
        let rows = self.max_seq * self.rope_rot_half();
        Ok((
            dispatch::read_back::<f32>(self.ctx, cb, rows).map_err(|e| anyhow::anyhow!("{e}"))?,
            dispatch::read_back::<f32>(self.ctx, sb, rows).map_err(|e| anyhow::anyhow!("{e}"))?,
        ))
    }

    pub fn fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(
        &mut self,
        pos: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            pos <= self.max_seq,
            "synthetic fill pos {pos} past max_seq {}; the full-attention window and gdn state \
             selection are pure functions of pos, so pos alone is the cache-depth state",
            self.max_seq
        );
        let max_bytes = self
            .state_buffers
            .iter()
            .map(|(_, b)| *b as usize)
            .max()
            .unwrap_or(0);
        let pattern = crate::gemma4_moe_wgpu::synthetic_state_bytes_0x30_to_0x3e_finite_small_under_fp8_bf16_and_f32_views_so_no_nan_and_moe_routing_stays_nondegenerate(max_bytes);
        for (buf, bytes) in &self.state_buffers {
            self.ctx
                .queue
                .write_buffer(buf, 0, &pattern[..*bytes as usize]);
        }
        self.pos = pos;
        if let Some(m) = self.mtp.as_mut() {
            m.len = pos;
            m.round_base = None;
        }
        Ok(())
    }

    pub fn new(
        config: Qwen3_5DenseConfig,
        weights: &HostDenseWeights,
        max_seq: usize,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Host(weights), max_seq)
    }

    pub fn from_loader(
        config: Qwen3_5DenseConfig,
        weights: &nv_weights::WeightLoader,
        max_seq: usize,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Loader(weights), max_seq)
    }

    fn build(config: Qwen3_5DenseConfig, src: WeightSource<'_>, max_seq: usize) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
        let s = Sources::new();
        let cfg = &config;

        anyhow::ensure!(max_seq > 0, "max_seq must be positive");
        anyhow::ensure!(
            cfg.hidden_size.is_multiple_of(4),
            "hidden_size {} must be a multiple of 4",
            cfg.hidden_size
        );
        anyhow::ensure!(
            cfg.intermediate_size.is_multiple_of(2),
            "intermediate_size {} must be even",
            cfg.intermediate_size
        );
        anyhow::ensure!(
            cfg.head_dim <= MAX_HEAD_DIM && cfg.head_dim.is_multiple_of(2),
            "head_dim {} must be even and <= {MAX_HEAD_DIM}",
            cfg.head_dim
        );
        anyhow::ensure!(
            cfg.linear_key_head_dim <= MAX_LIN_HEAD_DIM
                && cfg.linear_value_head_dim <= MAX_LIN_HEAD_DIM
                && cfg.linear_value_head_dim.is_multiple_of(2),
            "linear head dims must be even and <= {MAX_LIN_HEAD_DIM}"
        );
        anyhow::ensure!(
            cfg.linear_num_value_heads
                .is_multiple_of(cfg.linear_num_key_heads),
            "linear value heads must be a multiple of key heads"
        );
        anyhow::ensure!(
            cfg.num_attention_heads
                .is_multiple_of(cfg.num_key_value_heads),
            "attention heads must be a multiple of kv heads"
        );
        anyhow::ensure!(
            cfg.layer_types.len() == cfg.num_hidden_layers,
            "layer_types has {} entries for {} layers",
            cfg.layer_types.len(),
            cfg.num_hidden_layers
        );

        let hidden = cfg.hidden_size;
        let hidden_words = hidden / 2;
        let eps = cfg.rms_norm_eps as f32;
        let vocab = cfg.vocab_size;

        let mut b = Builder {
            core: crate::wgpu_ledger::VramLedger::new(
                ctx,
                "q3d-",
                staging_flush_enabled,
                STAGING_FLUSH_BYTES,
            ),
            passes: Vec::new(),
            pf_passes: Vec::new(),
        };

        let tok_buf = b.upload_u32("q3d-tok", &[0u32]);
        let pos_buf = b.upload_u32("q3d-pos", &[0u32]);
        let fd_base = FdParams {
            n_heads: cfg.num_attention_heads as u32,
            n_kv: cfg.num_key_value_heads as u32,
            head_dim: cfg.head_dim as u32,
            splits: wk::flash_decode::splits_for(max_seq) as u32,
            out_bf16: 0,
            scaling: 1.0 / (cfg.head_dim as f32).sqrt(),
            m_rows: 1,
            ..Default::default()
        };
        let fd_buf = b.uni("q3d-fd", fd_base);

        anyhow::ensure!(
            cfg.linear_conv_kernel_dim >= 1 && cfg.linear_conv_kernel_dim <= 8,
            "linear_conv_kernel_dim {} out of 1..=8",
            cfg.linear_conv_kernel_dim
        );

        anyhow::ensure!(
            !(pf_attn_mk_enabled()
                && std::env::var(PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM)
                    .ok()
                    .as_deref()
                    == Some("1")),
            "{} and {} both request the pf attention explicitly; the arms are alternatives, not \
             a ladder, so set at most one",
            PF_ATTN_MK_ENV_DEFAULT_OFF_UNTIL_THE_LADDER_AND_PPL_GATE_ARE_ON_RECORD,
            PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM
        );

        let mut pf: Option<Pf> = {
            let mut m = prefill_m();
            if m > 16 && !pf_coop_active(ctx) {
                eprintln!(
                    "[q3d-wgpu] prefill m={m} needs the coop route \
                     ({PF_COOP_ENV_DEFAULT_ON_SET_0_FOR_THE_LEGACY_M16_GEMM_ARM} not 0) and a \
                     16x16 f16 coop adapter; clamping to 16"
                );
                m = 16;
            }
            if m > 16 {
                if let WeightSource::Host(h) = &src {
                    if !host_pf_projections_ride_the_coop_route(h) {
                        eprintln!(
                            "[q3d-wgpu] prefill m={m}: bf16-only projections cannot ride the \
                             coop route (the m-row bf16 gemm stops at 16); clamping to 16"
                        );
                        m = 16;
                    }
                }
            }
            let tiled_routed = |m: usize| {
                pf_attn_tiled_enabled()
                    && (!pf_small_m_legacy(m)
                        || env_explicit_1(PF_ATTN_TILED_ENV_DEFAULT_ON_SET_0_FOR_THE_SCORES_SLAB_ARM))
            };
            if m >= 2 && !pf_attn_mk_enabled() && !tiled_routed(m) {
                let scores_budget =
                    PF_ATTN_SCORES_SLAB_IS_M_X_NHEADS_X_MAXSEQ_F32_SO_PF_M_CLAMPS_TO_THIS_BUDGET_BYTES
                        .min(ctx.caps.max_storage_buffer_binding_size);
                let scores_row_bytes = (cfg.num_attention_heads * max_seq * 4) as u64;
                let scores_cap = ((scores_budget / scores_row_bytes.max(1)) as usize).max(2);
                if m > scores_cap {
                    eprintln!(
                        "[q3d-wgpu] pf m {m} -> {scores_cap}: \
                         PF_ATTN_SCORES_SLAB_IS_M_X_NHEADS_X_MAXSEQ_F32_SO_PF_M_CLAMPS_TO_THIS_BUDGET_BYTES"
                    );
                    m = scores_cap;
                }
            }
            if m >= 2 {
                let n_k = cfg.linear_num_key_heads;
                let n_v = cfg.linear_num_value_heads;
                let d_k = cfg.linear_key_head_dim;
                let d_v = cfg.linear_value_head_dim;
                let key_dim = n_k * d_k;
                let value_dim = n_v * d_v;
                let conv_dim = 2 * key_dim + value_dim;
                let n_h = cfg.num_attention_heads;
                let n_kv = cfg.num_key_value_heads;
                let hd = cfg.head_dim;
                let src_stride_q = if cfg.attn_output_gate { 2 * hd } else { hd };
                let inter = cfg.intermediate_size;
                let mu = m as u64;
                let na = if na_enabled() {
                    match nv_kernels::wgpu_backend::na_bf16::pipeline(ctx) {
                        Ok(p) => Some(p),
                        Err(e) => {
                            eprintln!("[q3d-wgpu] NV_WGPU_NA set but na bf16 unavailable: {e}");
                            None
                        }
                    }
                } else {
                    None
                };
                let small_m = pf_small_m_legacy(m);
                let use_coop = pf_coop_active(ctx)
                    && (!small_m
                        || env_explicit_1(PF_COOP_ENV_DEFAULT_ON_SET_0_FOR_THE_LEGACY_M16_GEMM_ARM));
                let use_tiled = tiled_routed(m);
                let coop = if use_coop {
                    let m_pad = (m as u32).div_ceil(16) * 16;
                    let tm = (m_pad / 16).clamp(1, 8);
                    let tn = (16 / tm).clamp(1, 4);
                    let sg = 2u32;
                    let m_alloc = m_pad.div_ceil(16 * tm) * (16 * tm);
                    let max_k = hidden.max(inter).max(value_dim).max(n_h * hd);
                    let max_n = hidden
                        .max(inter)
                        .max(conv_dim)
                        .max(value_dim)
                        .max(2 * n_v)
                        .max(n_h * src_stride_q)
                        .max(n_kv * hd);
                    Some(PfCoop {
                        tm,
                        tn,
                        sg,
                        m_alloc,
                        x_f16: b.zeros("q3d-pf-coop-x16", m_alloc as u64 * (max_k * 2) as u64),
                        y_f32: b.zeros("q3d-pf-coop-y32", m_alloc as u64 * (max_n * 4) as u64),
                        zero: b.upload_f32("q3d-pf-coop-zero", &[0f32; 256]),
                        helpers: compose_enabled(&["f16"], PF_COOP_HELPERS_WGSL),
                    })
                } else {
                    None
                };
                let attn_mk = if pf_attn_mk_enabled() {
                    let rows = PF_ATTN_MK_ROWS_PER_DISPATCH_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS;
                    let splits = wk::flash_decode::splits_for(max_seq);
                    Some(PfAttnMk {
                        q_f32: b.zeros("q3d-pf-at-mk-qf32", mu * (n_h * hd * 4) as u64),
                        scratch: b.zeros(
                            "q3d-pf-at-mk-scratch",
                            (n_h * rows * splits * (hd + 2) * 4) as u64,
                        ),
                        fd: (0..m.div_ceil(rows))
                            .map(|_| b.uni("q3d-pf-at-mk-fd", FdParams::default()))
                            .collect(),
                    })
                } else {
                    None
                };
                let attn_tiled = if use_tiled {
                    let splits = wk::flash_decode::splits_for(max_seq);
                    Some(PfAttnTiled {
                        q_f32: b.zeros("q3d-pf-at-td-qf32", mu * (n_h * hd * 4) as u64),
                        scratch: b.zeros(
                            "q3d-pf-at-td-scratch",
                            (n_h * m * splits * (hd + 2) * 4) as u64,
                        ),
                        fd: b.uni("q3d-pf-at-td-fd", FdParams::default()),
                        start: b.zeros("q3d-pf-at-td-start", 4),
                    })
                } else {
                    None
                };
                let attn_kvq_fd_start = if attn_tiled.is_none() && kv_fp8_enabled() {
                    Some((
                        b.uni("q3d-pf-kvq-fd", FdParams::default()),
                        b.zeros("q3d-pf-kvq-start", 4),
                    ))
                } else {
                    None
                };
                let scores_slab = attn_mk.is_none() && attn_tiled.is_none();
                Some(Pf {
                    m,
                    attn_mk,
                    attn_tiled,
                    attn_kvq_fd_start,
                    na: if m <= 16 { na } else { None },
                    coop,
                    gemm_src: if m <= 16 { gemm_mk_source(m) } else { String::new() },
                    gemm_entry: if m <= 16 { gemm_mk_entry(m) } else { String::new() },
                    gemm_i8_src: if m <= 16 {
                        gemm_i8_mk_source(m)
                    } else {
                        String::new()
                    },
                    ck: b.uni("q3d-pf-ck", CkParams::default()),
                    tok: b.upload_u32("q3d-pf-tok", &vec![0u32; m]),
                    splice: b.zeros("q3d-pf-splice", mu * (hidden_words * 4) as u64),
                    mask: b.upload_u32("q3d-pf-mask", &vec![0u32; m]),
                    res: b.zeros("q3d-pf-res", mu * (hidden_words * 4) as u64),
                    normed: b.zeros("q3d-pf-normed", mu * (hidden_words * 4) as u64),
                    mix: b.zeros("q3d-pf-mix", mu * (hidden_words * 4) as u64),
                    mlp_out: b.zeros("q3d-pf-mlpout", mu * (hidden_words * 4) as u64),
                    normed_post: b.zeros("q3d-pf-normed-post", mu * (hidden_words * 4) as u64),
                    d_qkv: b.zeros("q3d-pf-dn-qkv", mu * (conv_dim * 2) as u64),
                    d_z: b.zeros("q3d-pf-dn-z", mu * (value_dim * 2) as u64),
                    d_ab: b.zeros("q3d-pf-dn-ab", mu * (2 * n_v * 2).max(4) as u64),
                    d_mixed: b.zeros("q3d-pf-dn-mixed", mu * (conv_dim * 4) as u64),
                    d_q: b.zeros("q3d-pf-dn-q", mu * (n_v * d_k * 4) as u64),
                    d_k: b.zeros("q3d-pf-dn-k", mu * (n_v * d_k * 4) as u64),
                    d_v: b.zeros("q3d-pf-dn-v", mu * (n_v * d_v * 4) as u64),
                    d_g: b.zeros("q3d-pf-dn-g", mu * (n_v * 4) as u64),
                    d_beta: b.zeros("q3d-pf-dn-beta", mu * (n_v * 4) as u64),
                    d_core: b.zeros("q3d-pf-dn-core", mu * (n_v * d_v * 4) as u64),
                    d_gated: b.zeros("q3d-pf-dn-gated", mu * (value_dim * 2) as u64),
                    a_qraw: b.zeros("q3d-pf-at-qraw", mu * (n_h * src_stride_q * 2) as u64),
                    a_kraw: b.zeros("q3d-pf-at-kraw", mu * (n_kv * hd * 2) as u64),
                    a_vraw: b.zeros("q3d-pf-at-vraw", mu * (n_kv * hd * 2) as u64),
                    a_q: b.zeros("q3d-pf-at-q", mu * (n_h * hd * 2) as u64),
                    a_k: b.zeros("q3d-pf-at-k", mu * (n_kv * hd * 2) as u64),
                    a_scores: b.zeros(
                        "q3d-pf-at-scores",
                        if scores_slab {
                            mu * (n_h * max_seq * 4) as u64
                        } else {
                            4
                        },
                    ),
                    a_attn: b.zeros("q3d-pf-at-attn", mu * (n_h * hd * 4) as u64),
                    a_gated: b.zeros("q3d-pf-at-gated", mu * (n_h * hd * 2) as u64),
                    m_ygate: b.zeros("q3d-pf-mlp-ygate", mu * (inter * 2) as u64),
                    m_yup: b.zeros("q3d-pf-mlp-yup", mu * (inter * 2) as u64),
                    m_act: b.zeros("q3d-pf-mlp-act", mu * (inter * 2) as u64),
                })
            } else {
                None
            }
        };

        let res = b.zeros("q3d-res", (hidden_words * 4) as u64);
        let res2 = b.zeros("q3d-res2", (hidden_words * 4) as u64);
        let normed = b.zeros("q3d-normed", (hidden_words * 4) as u64);
        let mixed_out = b.zeros("q3d-mix", (hidden_words * 4) as u64);
        let mlp_out = b.zeros("q3d-mlpout", (hidden_words * 4) as u64);

        let embed = src.embed(cfg)?;
        anyhow::ensure!(
            embed.len() == vocab * hidden,
            "embed has {} values, want {}",
            embed.len(),
            vocab * hidden
        );
        let chunk_rows = row_chunk(ctx, hidden);
        let mut off = 0usize;
        while off < vocab {
            let rows = chunk_rows.min(vocab - off);
            let buf = b.upload_u32(
                "q3d-embed",
                &pack_pairs(&embed[off * hidden..(off + rows) * hidden]),
            );
            let p = b.uni(
                "q3d-embed-p",
                GatherParams {
                    row_off: off as u32,
                    n_rows: rows as u32,
                    hidden_words: hidden_words as u32,
                    vocab: vocab as u32,
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.push(
                "q3d-gather",
                &s.misc,
                "q3w_gather_embed",
                &[(30, &buf), (31, &tok_buf), (32, &res), (33, &p)],
                grid,
            )?;
            if let Some(pf) = &pf {
                let pp = b.uni(
                    "q3d-pf-embed-p",
                    GatherParams {
                        row_off: off as u32,
                        n_rows: rows as u32,
                        hidden_words: hidden_words as u32,
                        vocab: vocab as u32,
                    },
                );
                b.push_pf(
                    "q3d-pf-gather",
                    &s.prefill,
                    "q3w_gather_embed_m",
                    &[(0, &buf), (1, &pf.tok), (2, &pf.res), (3, &pp)],
                    (grid.0, pf.m as u32, 1),
                )?;
            }
            off += rows;
        }
        drop(embed);
        let embed_gather_end = b.passes.len();

        if let Some(pf) = &pf {
            let sp = b.uni(
                "q3d-pf-splice-p",
                SpliceParams {
                    hidden_words: hidden_words as u32,
                    m: pf.m as u32,
                    pad0: 0,
                    pad1: 0,
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.push_pf(
                "q3d-pf-splice",
                &s.prefill,
                "q3w_splice_image_rows",
                &[(2, &pf.res), (4, &pf.splice), (5, &pf.mask), (6, &sp)],
                (grid.0, pf.m as u32, 1),
            )?;
        }
        let pf_embed_end = b.pf_passes.len();

        let mut state_buffers: Vec<(wgpu::Buffer, u64)> = Vec::new();
        let mut recurrent_states: Vec<(wgpu::Buffer, u64)> = Vec::new();
        let mut rope_row_buffers: Vec<(wgpu::Buffer, wgpu::Buffer)> = Vec::new();

        for li in 0..cfg.num_hidden_layers {
            let layer = src.layer(cfg, li)?;
            let ln_w = b.upload_u32("q3d-ln", &pack_pairs(&layer.input_ln));
            if li == 0 {
                let p = b.uni(
                    "q3d-rms-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: 1,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                b.push(
                    "q3d-rms0",
                    &s.rms,
                    "rmsnorm_bf16",
                    &[(0, &res), (1, &ln_w), (2, &normed), (3, &p)],
                    (1, 1, 1),
                )?;
                if let Some(pf) = &pf {
                    let pp = b.uni(
                        "q3d-pf-rms-p",
                        RmsParams {
                            hidden: hidden as u32,
                            batch: pf.m as u32,
                            eps,
                            words_per_row: hidden_words as u32,
                        },
                    );
                    b.push_pf(
                        "q3d-pf-rms0",
                        &s.rms,
                        "rmsnorm_bf16",
                        &[(0, &pf.res), (1, &ln_w), (2, &pf.normed), (3, &pp)],
                        (pf.m as u32, 1, 1),
                    )?;
                }
            }

            match &layer.mixer {
                HostDenseMixer::Delta(d) => {
                    let recurrent_from = state_buffers.len();
                    build_delta(
                        &mut b,
                        &s,
                        cfg,
                        d,
                        &layer.delta_fp8,
                        &normed,
                        &mixed_out,
                        &mut state_buffers,
                        &mut pf,
                    )?;
                    recurrent_states.extend_from_slice(&state_buffers[recurrent_from..]);
                }
                HostDenseMixer::Attn(a) => {
                    let _kv_pass_end_used_only_by_the_mtp_builder = build_attn(
                        &mut b,
                        &s,
                        cfg,
                        a,
                        &normed,
                        &mixed_out,
                        &pos_buf,
                        &fd_buf,
                        max_seq,
                        &mut state_buffers,
                        &mut pf,
                        &mut rope_row_buffers,
                    )?;
                }
            }

            let post_w = b.upload_u32("q3d-post-ln", &pack_pairs(&layer.post_attn_ln));
            let normed_post = b.zeros("q3d-normed-post", (hidden_words * 4) as u64);
            let rp = b.uni(
                "q3d-rmsres-p",
                RmsParams {
                    hidden: hidden as u32,
                    batch: 1,
                    eps,
                    words_per_row: hidden_words as u32,
                },
            );
            b.push(
                "q3d-rmsres-post",
                &s.rmsres,
                "rmsnorm_residual_bf16",
                &[
                    (0, &mixed_out),
                    (1, &res),
                    (2, &post_w),
                    (3, &normed_post),
                    (4, &rp),
                ],
                (1, 1, 1),
            )?;
            if let Some(pf) = &pf {
                let pp = b.uni(
                    "q3d-pf-rmsres-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: pf.m as u32,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                b.push_pf(
                    "q3d-pf-rmsres-post",
                    &s.rmsres,
                    "rmsnorm_residual_bf16",
                    &[
                        (0, &pf.mix),
                        (1, &pf.res),
                        (2, &post_w),
                        (3, &pf.normed_post),
                        (4, &pp),
                    ],
                    (pf.m as u32, 1, 1),
                )?;
            }

            build_mlp(&mut b, &s, cfg, &layer.mlp, &normed_post, &mlp_out, &mut pf)?;

            if li + 1 < cfg.num_hidden_layers {
                let next = src.layer_input_ln(cfg, li + 1)?;
                let nw = b.upload_u32("q3d-next-ln", &pack_pairs(&next));
                let rp2 = b.uni(
                    "q3d-rmsres2-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: 1,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                b.push(
                    "q3d-rmsres-next",
                    &s.rmsres,
                    "rmsnorm_residual_bf16",
                    &[(0, &mlp_out), (1, &res), (2, &nw), (3, &normed), (4, &rp2)],
                    (1, 1, 1),
                )?;
                if let Some(pf) = &pf {
                    let pp = b.uni(
                        "q3d-pf-rmsres2-p",
                        RmsParams {
                            hidden: hidden as u32,
                            batch: pf.m as u32,
                            eps,
                            words_per_row: hidden_words as u32,
                        },
                    );
                    b.push_pf(
                        "q3d-pf-rmsres-next",
                        &s.rmsres,
                        "rmsnorm_residual_bf16",
                        &[
                            (0, &pf.mlp_out),
                            (1, &pf.res),
                            (2, &nw),
                            (3, &pf.normed),
                            (4, &pp),
                        ],
                        (pf.m as u32, 1, 1),
                    )?;
                }
            } else {
                let sp = b.uni(
                    "q3d-resadd-p",
                    ResScaleParams {
                        n: hidden as u32,
                        n_words: hidden_words as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let grid = b.grid1(hidden_words as u64, 256);
                b.push(
                    "q3d-resadd",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &mlp_out), (1, &res), (2, &res2), (3, &sp)],
                    grid,
                )?;
            }
        }

        let verify = match &pf {
            Some(p) => {
                let m = p.m;
                let res2_rows = b.zeros("q3d-vf-res2", (m * hidden_words * 4) as u64);
                let vp = b.uni(
                    "q3d-vf-resadd-p",
                    ResScaleParams {
                        n: (m * hidden) as u32,
                        n_words: (m * hidden_words) as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let grid = b.grid1((m * hidden_words) as u64, 256);
                let resadd = b.make(
                    "q3d-vf-resadd",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &p.mlp_out), (1, &p.res), (2, &res2_rows), (3, &vp)],
                    grid,
                )?;
                let tok = b.zeros("q3d-vf-tok", (m * 4) as u64);
                Some(Verify {
                    rows: m,
                    res2_rows,
                    resadd,
                    tok,
                    row_logits: None,
                    rollback: Vec::new(),
                    validated: false,
                    pending: None,
                })
            }
            None => None,
        };

        let final_start = b.passes.len();
        let final_w = b.upload_u32("q3d-final-ln", &pack_pairs(&src.final_norm(cfg)?));
        let final_x = b.zeros("q3d-final-x", (hidden_words * 4) as u64);
        let fp = b.uni(
            "q3d-final-p",
            RmsParams {
                hidden: hidden as u32,
                batch: 1,
                eps,
                words_per_row: hidden_words as u32,
            },
        );
        b.push(
            "q3d-final-rms",
            &s.rms,
            "rmsnorm_bf16",
            &[(0, &res2), (1, &final_w), (2, &final_x), (3, &fp)],
            (1, 1, 1),
        )?;

        let head_start = b.passes.len();
        let logits = b.zeros("q3d-logits", (vocab * 4) as u64);
        if let Some((packed, scales)) = src.lm_head_fp8_packed(cfg)? {
            let row_words = hidden / 4;
            anyhow::ensure!(
                packed.len() == vocab * row_words,
                "fp8 lm_head has {} words, want {}",
                packed.len(),
                vocab * row_words
            );
            let mut off = 0usize;
            while off < vocab {
                let rows = (chunk_rows.min(vocab - off)) & !1usize;
                let rows = if rows == 0 { vocab - off } else { rows };
                let wbuf = b.upload_u32(
                    "q3d-lmhead-fp8",
                    &packed[off * row_words..(off + rows) * row_words],
                );
                let sbuf = b.upload_f32(
                    "q3d-lmhead-fp8-s",
                    &nv_kernels::shift_decode_fold::fold_scales_for_e4m3_shift_decode(
                        &scales[off..off + rows],
                    ),
                );
                let pairs = rows.div_ceil(2);
                let grid = b.grid1(pairs as u64, 1);
                let p = b.uni(
                    "q3d-lmhead-p",
                    GemvBf16Params {
                        n_rows: rows as u32,
                        k_words: row_words as u32,
                        groups_x: grid.0,
                        out_f32: 1,
                        w_row_words: row_words as u32,
                        x_off_words: 0,
                        y_off_words: off as u32,
                        alpha: 1.0,
                        ..Default::default()
                    },
                );
                b.push(
                    "q3d-lmhead-fp8",
                    &s.gemv_bf16,
                    "q3w_gemv_fp8_rowscale",
                    &[(0, &wbuf), (1, &final_x), (2, &p), (3, &logits), (4, &sbuf)],
                    grid,
                )?;
                off += rows;
            }
        } else {
            let lm = src.lm_head(cfg)?;
            anyhow::ensure!(
                lm.len() == vocab * hidden,
                "lm_head has {} values, want {}",
                lm.len(),
                vocab * hidden
            );
            let mut off = 0usize;
            while off < vocab {
                let rows = (chunk_rows.min(vocab - off)) & !1usize;
                let rows = if rows == 0 { vocab - off } else { rows };
                let wbuf = b.upload_u32(
                    "q3d-lmhead",
                    &pack_pairs(&lm[off * hidden..(off + rows) * hidden]),
                );
                let pairs = rows.div_ceil(2);
                let grid = b.grid1(pairs as u64, 1);
                let p = b.uni(
                    "q3d-lmhead-p",
                    GemvBf16Params {
                        n_rows: rows as u32,
                        k_words: hidden_words as u32,
                        groups_x: grid.0,
                        out_f32: 1,
                        w_row_words: hidden_words as u32,
                        x_off_words: 0,
                        y_off_words: off as u32,
                        alpha: 1.0,
                        ..Default::default()
                    },
                );
                b.push(
                    "q3d-lmhead",
                    &s.gemv_bf16,
                    "q3w_gemv_bf16",
                    &[(0, &wbuf), (1, &final_x), (2, &p), (3, &logits)],
                    grid,
                )?;
                off += rows;
            }
        }

        let pv = b.zeros("q3d-am-pv", (ARGMAX_GROUPS * 4) as u64);
        let pi = b.zeros("q3d-am-pi", (ARGMAX_GROUPS * 4) as u64);
        let token_out = b.zeros("q3d-token", 4);
        let ap = b.uni(
            "q3d-am-p",
            ArgmaxParams {
                n: vocab as u32,
                groups: ARGMAX_GROUPS as u32,
                ..Default::default()
            },
        );
        b.push(
            "q3d-am1",
            &s.misc,
            "q3w_argmax_stage1",
            &[(40, &logits), (41, &pv), (42, &pi), (44, &ap)],
            (ARGMAX_GROUPS as u32, 1, 1),
        )?;
        b.push(
            "q3d-am2",
            &s.misc,
            "q3w_argmax_stage2",
            &[(41, &pv), (42, &pi), (43, &token_out), (44, &ap)],
            (1, 1, 1),
        )?;

        b.flush_staging();
        let vram = b.report();
        if vram_report_enabled() {
            eprint!("[q3d-wgpu] {}", vram.render());
        }

        let Builder {
            passes,
            pf_passes,
            core,
        } = b;
        let buffers = core.buffers;

        let (pf_passes, pf_m, pf_tok, pf_ck, pf_splice, pf_mask, pf_attn_mk_fd, pf_tiled, pf_res) =
            match pf {
                Some(p) => (
                    pf_passes,
                    p.m,
                    Some(p.tok.clone()),
                    Some(p.ck.clone()),
                    Some(p.splice.clone()),
                    Some(p.mask.clone()),
                    p.attn_mk.as_ref().map(|k| k.fd.clone()).unwrap_or_default(),
                    p.attn_tiled
                        .as_ref()
                        .map(|t| (t.fd.clone(), t.start.clone()))
                        .or_else(|| p.attn_kvq_fd_start.clone()),
                    Some(p.res.clone()),
                ),
                None => (Vec::new(), 0, None, None, None, None, Vec::new(), None, None),
            };

        let tok_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q3d-token-stage"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rope_std_host_cos_sin =
            rope_tables(config.rotary_dim().max(2), config.rope_theta, max_seq);

        Ok(Self {
            ctx,
            config,
            max_seq,
            pos: 0,
            validated: false,
            prefix_validated: false,
            pf_validated: false,
            passes,
            pf_passes,
            pf_m,
            pf_tok,
            pf_ck,
            pf_splice,
            pf_mask,
            pf_attn_mk_fd,
            pf_chunk_fd_and_start: pf_tiled,
            pf_res,
            pf_embed_end,
            head_start,
            final_start,
            verify,
            _buffers: buffers,
            tok_buf,
            pos_buf,
            fd_buf,
            fd_base,
            res,
            res2,
            final_x,
            embed_gather_end,
            mtp: None,
            mtp_replay: false,
            token_out,
            logits,
            state_buffers,
            recurrent_states,
            rope_row_buffers,
            rope_std_host_cos_sin,
            mrope_rows_installed: false,
            vocab,
            vram,
            preenc: preenc_enabled(),
            pending_cb: None,
            staged_read: staged_read_enabled(),
            tok_stage,
        })
    }

    fn profiled(&self) -> bool {
        dispatch::profile::enabled() && self.ctx.caps.timestamp_query
    }

    fn encode_step_cb(&self, full: bool) -> wgpu::CommandBuffer {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let passes = if full {
                &self.passes[..]
            } else {
                &self.passes[..self.head_start]
            };
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in passes {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        if full && self.staged_read {
            enc.copy_buffer_to_buffer(&self.token_out, 0, &self.tok_stage, 0, 4);
        }
        enc.finish()
    }

    fn read_token_stage(&self) -> Result<u32> {
        let slice = self.tok_stage.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.ctx
            .poll_blocking()
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        rx.recv()
            .map_err(|e| anyhow::anyhow!("token stage map callback: {e}"))?
            .map_err(|e| anyhow::anyhow!("token stage map: {e}"))?;
        let t = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("token stage mapped range: {e}"))?;
            bytemuck::cast_slice::<u8, u32>(&view)[0]
        };
        self.tok_stage.unmap();
        Ok(t)
    }

    fn read_token_out(&self) -> Result<u32> {
        if self.staged_read && !self.profiled() {
            return self.read_token_stage();
        }
        let t: Vec<u32> = dispatch::read_back(self.ctx, &self.token_out, 1)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(t[0])
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
        let mut fd = self.fd_base;
        fd.total = (self.pos + 1) as u32;
        self.ctx
            .queue
            .write_buffer(&self.fd_buf, 0, bytemuck::bytes_of(&fd));

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
        if self.profiled() {
            let passes = if full {
                &self.passes[..]
            } else {
                &self.passes[..self.head_start]
            };
            let raw: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))> = passes
                .iter()
                .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
                .collect();
            let labels: Vec<String> = passes.iter().map(|p| p.label.clone()).collect();
            dispatch::submit_profiled_slices(self.ctx, &raw, &labels)
                .map_err(|e| anyhow::anyhow!("profiled submit: {e}"))?;
        } else {
            let cb = if full {
                self.pending_cb
                    .take()
                    .unwrap_or_else(|| self.encode_step_cb(true))
            } else {
                self.encode_step_cb(false)
            };
            self.ctx.queue.submit([cb]);
            if full && self.preenc {
                self.pending_cb = Some(self.encode_step_cb(true));
            }
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("qwen3_5_dense_wgpu decode step validation: {e}");
            }
            if full {
                self.validated = true;
            }
            self.prefix_validated = true;
        }
        self.pos += 1;
        if self.mtp.is_some() && !self.mtp_replay {
            self.mtp_sync_committed_step(token)?;
        }
        Ok(())
    }

    pub fn decode_step(&mut self, token: u32) -> Result<u32> {
        self.step_inner(token, true)?;
        self.read_token_out()
    }

    pub fn decode_step_logits(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        self.step_inner(token, true)?;
        let t = self.read_token_out()?;
        let l: Vec<f32> = dispatch::read_back(self.ctx, &self.logits, self.vocab)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok((t, l))
    }

    pub fn prefill_step(&mut self, token: u32) -> Result<()> {
        self.step_inner(token, false)
    }

    pub fn prefill_chunk_len(&self) -> usize {
        if self.pf_passes.is_empty() {
            0
        } else {
            self.pf_m
        }
    }

    pub fn prefill_pass_count(&self) -> usize {
        self.pf_passes.len()
    }

    fn prefill_chunk(&mut self, chunk: &[u32], live: usize) -> Result<()> {
        self.prefill_chunk_masked(chunk, live, &[])
    }

    pub fn debug_pf_embed_splice_rows_for_test(
        &mut self,
        chunk: &[u32],
        splices: &[(usize, Vec<u16>)],
    ) -> Result<Vec<u16>> {
        let m = self.pf_m;
        anyhow::ensure!(m >= 2 && chunk.len() == m, "needs the pf graph and a full chunk");
        let hidden = self.config.hidden_size;
        let hidden_words = hidden / 2;
        let splice_buf = self.pf_splice.as_ref().expect("pf splice buffer");
        let mut mask = vec![0u32; m];
        for (rel, row) in splices {
            anyhow::ensure!(*rel < m && row.len() == hidden, "bad debug splice row");
            mask[*rel] = 1;
            self.ctx.queue.write_buffer(
                splice_buf,
                (*rel * hidden_words * 4) as u64,
                bytemuck::cast_slice(&pack_pairs(row)),
            );
        }
        self.ctx.queue.write_buffer(
            self.pf_mask.as_ref().expect("pf mask buffer"),
            0,
            bytemuck::cast_slice(&mask),
        );
        let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
        self.ctx.queue.write_buffer(
            self.pf_tok.as_ref().expect("pf tok buffer"),
            0,
            bytemuck::cast_slice(&ids),
        );
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in &self.pf_passes[..self.pf_embed_end] {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        let res = self.pf_res.as_ref().expect("pf res buffer");
        let words: Vec<u32> = dispatch::read_back::<u32>(self.ctx, res, m * hidden_words)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        let mut out = Vec::with_capacity(m * hidden);
        for w in words {
            out.push((w & 0xffff) as u16);
            out.push((w >> 16) as u16);
        }
        Ok(out)
    }

    pub fn debug_state_buffer_words_for_test(&self) -> Vec<(u64, Vec<u32>)> {
        self.state_buffers
            .iter()
            .map(|(buf, bytes)| {
                (
                    *bytes,
                    dispatch::read_back::<u32>(self.ctx, buf, (*bytes / 4) as usize)
                        .unwrap_or_default(),
                )
            })
            .collect()
    }

    fn write_pf_attn_mk_fd(&self, live: usize, base: usize) {
        let rows = PF_ATTN_MK_ROWS_PER_DISPATCH_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS;
        for (g, buf) in self.pf_attn_mk_fd.iter().enumerate() {
            let start = g * rows;
            let mr = live.saturating_sub(start).min(rows);
            let mut fd = self.fd_base;
            fd.m_rows = mr as u32;
            fd.total = if mr == 0 {
                0
            } else {
                (base + start + mr) as u32
            };
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&fd));
        }
    }

    fn write_pf_chunk_fd_whole_view_and_start_for_tiled_flash_and_fp8_chunk_quantize(&self, base: usize) {
        if let Some((fd_buf, start_buf)) = &self.pf_chunk_fd_and_start {
            let mut fd = self.fd_base;
            fd.m_rows = self.pf_m as u32;
            fd.total = (base + self.pf_m) as u32;
            self.ctx
                .queue
                .write_buffer(fd_buf, 0, bytemuck::bytes_of(&fd));
            self.ctx
                .queue
                .write_buffer(start_buf, 0, bytemuck::bytes_of(&(base as i32)));
        }
    }

    fn prefill_chunk_masked(
        &mut self,
        chunk: &[u32],
        live: usize,
        splices: &[ChunkRowSplice],
    ) -> Result<()> {
        let m = self.pf_m;
        anyhow::ensure!(chunk.len() == m && (1..=m).contains(&live), "bad chunk");
        anyhow::ensure!(
            splices.is_empty() || self.mtp.is_none(),
            "the mtp drafter has no image-splice row convention: its KV row embeds the token id, \
             not the spliced media row; serve mm without the mtp head attached"
        );
        anyhow::ensure!(
            self.pos + live <= self.max_seq,
            "kv cache full at {} + {live} (max_seq {})",
            self.pos,
            self.max_seq
        );
        for &t in chunk {
            anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        let hidden_words = self.config.hidden_size / 2;
        let mut mask = vec![0u32; m];
        for sp in splices {
            anyhow::ensure!(sp.rel_pos < live, "splice rel_pos {} >= live {live}", sp.rel_pos);
            anyhow::ensure!(
                sp.row_words.len() == hidden_words,
                "splice row has {} words, want {hidden_words}",
                sp.row_words.len()
            );
            mask[sp.rel_pos] = 1;
            let splice_buf = self
                .pf_splice
                .as_ref()
                .expect("splice prefill without pf list");
            self.ctx.queue.write_buffer(
                splice_buf,
                (sp.rel_pos * hidden_words * 4) as u64,
                bytemuck::cast_slice(sp.row_words),
            );
        }
        let mask_buf = self.pf_mask.as_ref().expect("prefill chunk without pf list");
        self.ctx
            .queue
            .write_buffer(mask_buf, 0, bytemuck::cast_slice(&mask));
        let tok = self.pf_tok.as_ref().expect("prefill chunk without pf list");
        let ck = self.pf_ck.as_ref().expect("prefill chunk without pf list");
        let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
        self.ctx
            .queue
            .write_buffer(tok, 0, bytemuck::cast_slice(&ids));
        self.ctx.queue.write_buffer(
            ck,
            0,
            bytemuck::bytes_of(&CkParams {
                m_live: live as u32,
                base: self.pos as u32,
                pad0: 0,
                pad1: 0,
            }),
        );
        self.write_pf_attn_mk_fd(live, self.pos);
        self.write_pf_chunk_fd_whole_view_and_start_for_tiled_flash_and_fp8_chunk_quantize(self.pos);
        let scope = if self.pf_validated {
            None
        } else {
            Some(
                self.ctx
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation),
            )
        };
        if dispatch::profile::enabled() && self.ctx.caps.timestamp_query {
            let raw: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))> = self
                .pf_passes
                .iter()
                .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
                .collect();
            let labels: Vec<String> = self.pf_passes.iter().map(|p| p.label.clone()).collect();
            dispatch::submit_profiled_slices(self.ctx, &raw, &labels)
                .map_err(|e| anyhow::anyhow!("profiled prefill submit: {e}"))?;
        } else {
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in &self.pf_passes {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
            self.ctx.queue.submit([enc.finish()]);
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("qwen3_5_dense_wgpu prefill chunk validation: {e}");
            }
            self.pf_validated = true;
        }
        self.pos += live;
        if self.mtp.is_some() && !self.mtp_replay {
            let live_tokens: Vec<u32> = chunk[..live].to_vec();
            self.mtp_sync_committed_chunk(&live_tokens)?;
        }
        Ok(())
    }

    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        let m = self.prefill_chunk_len();
        if m == 0 {
            return Ok(0);
        }
        let mut done = 0usize;
        while tokens.len() - done >= m && self.pos + m <= self.max_seq {
            let chunk: Vec<u32> = tokens[done..done + m].to_vec();
            self.prefill_chunk(&chunk, m)?;
            done += m;
        }
        let left = tokens.len() - done;
        if left >= 2 && self.pos + left <= self.max_seq {
            let mut padded: Vec<u32> = tokens[done..].to_vec();
            let pad = *padded.last().expect("non-empty tail");
            padded.resize(m, pad);
            self.prefill_chunk(&padded, left)?;
            done += left;
        }
        Ok(done)
    }

    pub fn prefill_tokens_with_image_rows(
        &mut self,
        tokens: &[u32],
        splices: &[ImageRowSplice],
    ) -> Result<usize> {
        let m = self.prefill_chunk_len();
        anyhow::ensure!(
            m >= 2,
            "image-row splice prefill requires the chunked prefill graph: NV_WGPU_PREFILL_M>=2 and \
             a live pf pass list"
        );
        let hidden = self.config.hidden_size;
        let hidden_words = hidden / 2;
        let mut prev_end = 0usize;
        let mut packed: Vec<Vec<u32>> = Vec::with_capacity(splices.len());
        for sp in splices {
            anyhow::ensure!(
                sp.rows_bf16.len() % hidden == 0,
                "splice rows_bf16 len {} not a multiple of hidden {hidden}",
                sp.rows_bf16.len()
            );
            let n_slots = sp.rows_bf16.len() / hidden;
            anyhow::ensure!(
                sp.position >= prev_end,
                "image splices must be sorted and non-overlapping"
            );
            anyhow::ensure!(
                sp.position + n_slots <= tokens.len(),
                "image splice at {} with {n_slots} slots exceeds {} tokens",
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
                self.pos + live <= self.max_seq,
                "kv cache full at {} + {live} (max_seq {})",
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
                let s0 = sp.position;
                let s1 = sp.position + n_slots;
                let lo = s0.max(done);
                let hi = s1.min(chunk_end);
                for abs in lo..hi {
                    let slot = abs - s0;
                    let w0 = slot * hidden_words;
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

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        anyhow::ensure!(!tokens.is_empty(), "prefill needs at least one token");
        let (last, rest) = tokens.split_last().expect("non-empty");
        let done = self.prefill_tokens(rest)?;
        for t in &rest[done..] {
            self.prefill_step(*t)?;
        }
        self.decode_step(*last)
    }

    pub fn verify_max_rows(&self) -> usize {
        self.verify.as_ref().map_or(0, |v| v.rows)
    }

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        Ok(self.verify_chain_inner(batch, VerifyTail::ArgmaxRows)?.0)
    }

    pub fn verify_chain_logits(&mut self, batch: &[u32]) -> Result<(Vec<u32>, Vec<f32>)> {
        self.verify_chain_inner(batch, VerifyTail::ArgmaxAndLogitRows)
    }

    pub fn verify_chain_commit_only(&mut self, batch: &[u32]) -> Result<()> {
        self.verify_chain_inner(
            batch,
            VerifyTail::SkippedBecauseCommitOnlyCallersAlreadyKnowEveryToken,
        )?;
        Ok(())
    }

    fn verify_chain_inner(
        &mut self,
        batch: &[u32],
        tail: VerifyTail,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let want_logits = tail == VerifyTail::ArgmaxAndLogitRows;
        let commit_only =
            tail == VerifyTail::SkippedBecauseCommitOnlyCallersAlreadyKnowEveryToken;
        let rows = self.verify_max_rows();
        anyhow::ensure!(
            rows >= 2,
            "verify_chain needs the M-row prefill graph: NV_WGPU_PREFILL_M >= 2"
        );
        let live = batch.len();
        anyhow::ensure!(
            (1..=rows).contains(&live),
            "verify_chain batch {live} out of 1..={rows}"
        );
        anyhow::ensure!(
            self.verify.as_ref().is_some_and(|v| v.pending.is_none()),
            "verify_chain twice without advance(): {VERIFY_CHAIN_COMMITS_BY_SNAPSHOT_AND_M1_REPLAY_BECAUSE_DELTANET_STATE_IS_RECURRENT_NOT_POSITION_MASKED}"
        );
        anyhow::ensure!(
            self.pos + live <= self.max_seq,
            "kv cache full at {} + {live} (max_seq {})",
            self.pos,
            self.max_seq
        );
        for &t in batch {
            anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        let hidden_words = self.config.hidden_size / 2;
        let row_bytes = (hidden_words * 4) as u64;
        let vocab_bytes = (self.vocab * 4) as u64;
        if self.verify.as_ref().is_some_and(|v| v.rollback.is_empty()) {
            let rollback: Vec<wgpu::Buffer> = self
                .recurrent_states
                .iter()
                .map(|(_, bytes)| dispatch::storage_zeroed(self.ctx, "q3d-vf-rollback", *bytes))
                .collect();
            self.verify.as_mut().expect("verify list").rollback = rollback;
        }
        if want_logits && self.verify.as_ref().is_some_and(|v| v.row_logits.is_none()) {
            let buf = dispatch::storage_zeroed(
                self.ctx,
                "q3d-vf-row-logits",
                rows as u64 * vocab_bytes,
            );
            self.verify.as_mut().expect("verify list").row_logits = Some(buf);
        }

        let mut padded: Vec<u32> = batch.to_vec();
        padded.resize(rows, *batch.last().expect("non-empty batch"));
        let ids: Vec<i32> = padded.iter().map(|&t| t as i32).collect();
        let tok = self.pf_tok.as_ref().expect("verify without pf list");
        let ck = self.pf_ck.as_ref().expect("verify without pf list");
        let mask = self.pf_mask.as_ref().expect("verify without pf list");
        self.ctx
            .queue
            .write_buffer(mask, 0, bytemuck::cast_slice(&vec![0u32; rows]));
        self.ctx
            .queue
            .write_buffer(tok, 0, bytemuck::cast_slice(&ids));
        self.ctx.queue.write_buffer(
            ck,
            0,
            bytemuck::bytes_of(&CkParams {
                m_live: live as u32,
                base: self.pos as u32,
                pad0: 0,
                pad1: 0,
            }),
        );
        self.write_pf_attn_mk_fd(live, self.pos);
        self.write_pf_chunk_fd_whole_view_and_start_for_tiled_flash_and_fp8_chunk_quantize(self.pos);

        let v = self.verify.as_ref().expect("verify list");
        let scope = (!v.validated).then(|| {
            self.ctx
                .device
                .push_error_scope(wgpu::ErrorFilter::Validation)
        });
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for ((src, bytes), dst) in self.recurrent_states.iter().zip(&v.rollback) {
            enc.copy_buffer_to_buffer(src, 0, dst, 0, *bytes);
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in self.pf_passes.iter().chain(std::iter::once(&v.resadd)) {
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        if !commit_only {
            for r in 0..live {
                enc.copy_buffer_to_buffer(
                    &v.res2_rows,
                    r as u64 * row_bytes,
                    &self.res2,
                    0,
                    row_bytes,
                );
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    for p in &self.passes[self.final_start..] {
                        pass.set_pipeline(&p.pipeline);
                        pass.set_bind_group(0, &p.bind, &[]);
                        pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                    }
                }
                enc.copy_buffer_to_buffer(&self.token_out, 0, &v.tok, r as u64 * 4, 4);
                if want_logits {
                    let rl = v.row_logits.as_ref().expect("row logits buffer");
                    enc.copy_buffer_to_buffer(
                        &self.logits,
                        0,
                        rl,
                        r as u64 * vocab_bytes,
                        vocab_bytes,
                    );
                }
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("qwen3_5_dense_wgpu verify chain validation: {e}");
            }
        }
        let toks: Vec<u32> = if commit_only {
            Vec::new()
        } else {
            dispatch::read_back(self.ctx, &v.tok, live).map_err(|e| anyhow::anyhow!("{e}"))?
        };
        let logits = match v.row_logits.as_ref().filter(|_| want_logits) {
            Some(rl) => dispatch::read_back(self.ctx, rl, live * self.vocab)
                .map_err(|e| anyhow::anyhow!("{e}"))?,
            None => Vec::new(),
        };
        let vm = self.verify.as_mut().expect("verify list");
        vm.validated = true;
        vm.pending = Some(batch.to_vec());
        Ok((toks, logits))
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        let rows = self
            .verify
            .as_ref()
            .and_then(|v| v.pending.as_ref().map(|b| b.len()))
            .ok_or_else(|| anyhow::anyhow!("advance() without a pending verify_chain"))?;
        anyhow::ensure!(
            n <= rows,
            "advance {n} beyond the {rows} rows verify_chain forwarded"
        );
        let batch = self
            .verify
            .as_mut()
            .and_then(|v| v.pending.take())
            .expect("pending chain");
        if n == batch.len() {
            self.pos += n;
            return Ok(());
        }
        self.verify_rollback()?;
        self.mtp_replay = true;
        let mut replay_err = None;
        for &t in &batch[..n] {
            if let Err(e) = self.prefill_step(t) {
                replay_err = Some(e);
                break;
            }
        }
        self.mtp_replay = false;
        match replay_err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    fn verify_rollback(&mut self) -> Result<()> {
        let v = self.verify.as_ref().expect("verify list");
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for ((dst, bytes), src) in self.recurrent_states.iter().zip(&v.rollback) {
            enc.copy_buffer_to_buffer(src, 0, dst, 0, *bytes);
        }
        self.ctx.queue.submit([enc.finish()]);
        Ok(())
    }

    pub fn mtp_active(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn mtp_len(&self) -> usize {
        self.mtp.as_ref().map_or(0, |m| m.len)
    }

    pub fn mtp_attach(&mut self, w: &nv_weights::WeightLoader) -> Result<()> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let hd = cfg.head_dim;
        let heads = cfg.num_attention_heads;
        let q_out = heads * hd * if cfg.attn_output_gate { 2 } else { 1 };
        let kv_out = cfg.num_key_value_heads * hd;
        let inter = cfg.intermediate_size;
        let host = MtpHostWeights {
            pre_fc_norm_embedding: load_norm_plus_one(
                w,
                &["mtp.pre_fc_norm_embedding.weight"],
                hidden,
            )?,
            pre_fc_norm_hidden: load_norm_plus_one(w, &["mtp.pre_fc_norm_hidden.weight"], hidden)?,
            fc: load_lin(w, "mtp.fc.weight", hidden, 2 * hidden)?,
            input_ln: load_norm_plus_one(w, &["mtp.layers.0.input_layernorm.weight"], hidden)?,
            attn: HostDenseAttention {
                q: load_lin(w, "mtp.layers.0.self_attn.q_proj.weight", q_out, hidden)?.into(),
                k: load_lin(w, "mtp.layers.0.self_attn.k_proj.weight", kv_out, hidden)?.into(),
                v: load_lin(w, "mtp.layers.0.self_attn.v_proj.weight", kv_out, hidden)?.into(),
                o: load_lin(w, "mtp.layers.0.self_attn.o_proj.weight", hidden, heads * hd)?.into(),
                q_norm: load_norm_plus_one(w, &["mtp.layers.0.self_attn.q_norm.weight"], hd)?,
                k_norm: load_norm_plus_one(w, &["mtp.layers.0.self_attn.k_norm.weight"], hd)?,
            },
            post_attn_ln: load_norm_plus_one(
                w,
                &["mtp.layers.0.post_attention_layernorm.weight"],
                hidden,
            )?,
            mlp: HostDenseMlp {
                gate: load_lin(w, "mtp.layers.0.mlp.gate_proj.weight", inter, hidden)?.into(),
                up: load_lin(w, "mtp.layers.0.mlp.up_proj.weight", inter, hidden)?.into(),
                down: load_lin(w, "mtp.layers.0.mlp.down_proj.weight", hidden, inter)?.into(),
            },
            final_norm: load_norm_plus_one(w, &["mtp.norm.weight"], hidden)?,
        };
        self.mtp_attach_host(&host)
    }

    pub fn mtp_attach_host(&mut self, w: &MtpHostWeights) -> Result<()> {
        anyhow::ensure!(self.mtp.is_none(), "mtp head already attached");
        anyhow::ensure!(
            self.verify.is_some(),
            "the mtp drafter needs the M-row prefill graph (NV_WGPU_PREFILL_M >= 2): drafts are \
             verified through verify_chain and the prompt/round KV catch-up reads its hidden rows"
        );
        let cfg = self.config.clone();
        let hidden = cfg.hidden_size;
        let hw = hidden / 2;
        let eps = cfg.rms_norm_eps as f32;
        let hd = cfg.head_dim;
        let heads = cfg.num_attention_heads;
        let q_out = heads * hd * if cfg.attn_output_gate { 2 } else { 1 };
        let kv_out = cfg.num_key_value_heads * hd;
        anyhow::ensure!(
            w.fc.n == hidden && w.fc.k == 2 * hidden,
            "mtp fc must be [{hidden}, {}], got [{}, {}]; \
             {MTP_HEAD_CONVENTIONS_ZERO_CENTERED_NORMS_EMB_FIRST_FC_AND_SHIFT_BY_ONE_KV_WITH_A_ZERO_HIDDEN_AT_POS_0}",
            2 * hidden,
            w.fc.n,
            w.fc.k
        );
        anyhow::ensure!(
            w.attn.q.n() == q_out
                && w.attn.q.k() == hidden
                && w.attn.k.n() == kv_out
                && w.attn.v.n() == kv_out
                && w.attn.o.n() == hidden
                && w.attn.o.k() == heads * hd
                && w.attn.q_norm.len() == hd
                && w.attn.k_norm.len() == hd,
            "mtp attention shapes disagree with the trunk full-attn geometry \
             (heads={heads}, kv={}, hd={hd}, gate={})",
            cfg.num_key_value_heads,
            cfg.attn_output_gate
        );
        anyhow::ensure!(
            w.pre_fc_norm_embedding.len() == hidden
                && w.pre_fc_norm_hidden.len() == hidden
                && w.input_ln.len() == hidden
                && w.post_attn_ln.len() == hidden
                && w.final_norm.len() == hidden,
            "mtp norm vectors must all be [{hidden}]"
        );

        let s = Sources::new();
        let mut b = Builder {
            core: crate::wgpu_ledger::VramLedger::new(
                self.ctx,
                "q3d-",
                staging_flush_enabled,
                STAGING_FLUSH_BYTES,
            ),
            passes: Vec::new(),
            pf_passes: Vec::new(),
        };
        let row_bytes = (hw * 4) as u64;
        let pos_buf = b.upload_u32("q3d-mtp-pos", &[0u32]);
        let fd_buf = b.uni("q3d-mtp-fd", self.fd_base);
        let hid = b.zeros("q3d-mtp-hid", row_bytes);
        let ne = b.zeros("q3d-mtp-ne", row_bytes);
        let nh = b.zeros("q3d-mtp-nh", row_bytes);
        let fe = b.zeros("q3d-mtp-fe", row_bytes);
        let fh = b.zeros("q3d-mtp-fh", row_bytes);
        let mres = b.zeros("q3d-mtp-res", row_bytes);
        let mnormed = b.zeros("q3d-mtp-normed", row_bytes);
        let mmix = b.zeros("q3d-mtp-mix", row_bytes);
        let mnormed_post = b.zeros("q3d-mtp-normed-post", row_bytes);
        let mmlp_out = b.zeros("q3d-mtp-mlpout", row_bytes);

        let rms1 = b.uni(
            "q3d-mtp-rms-p",
            RmsParams {
                hidden: hidden as u32,
                batch: 1,
                eps,
                words_per_row: hw as u32,
            },
        );
        let wne = b.upload_u32("q3d-mtp-ln-emb", &pack_pairs(&w.pre_fc_norm_embedding));
        b.push(
            "q3d-mtp-rms-emb",
            &s.rms,
            "rmsnorm_bf16",
            &[(0, &self.res), (1, &wne), (2, &ne), (3, &rms1)],
            (1, 1, 1),
        )?;
        let wnh = b.upload_u32("q3d-mtp-ln-hid", &pack_pairs(&w.pre_fc_norm_hidden));
        b.push(
            "q3d-mtp-rms-hid",
            &s.rms,
            "rmsnorm_bf16",
            &[(0, &hid), (1, &wnh), (2, &nh), (3, &rms1)],
            (1, 1, 1),
        )?;

        let mut fc_emb = vec![0u16; hidden * hidden];
        let mut fc_hid = vec![0u16; hidden * hidden];
        for r in 0..hidden {
            let row = &w.fc.w[r * 2 * hidden..(r + 1) * 2 * hidden];
            fc_emb[r * hidden..(r + 1) * hidden].copy_from_slice(&row[..hidden]);
            fc_hid[r * hidden..(r + 1) * hidden].copy_from_slice(&row[hidden..]);
        }
        let gfe = upload_bf16(
            &mut b,
            "q3d-mtp-fc-emb",
            &HostBf16Lin {
                w: fc_emb,
                n: hidden,
                k: hidden,
            },
        );
        let gfh = upload_bf16(
            &mut b,
            "q3d-mtp-fc-hid",
            &HostBf16Lin {
                w: fc_hid,
                n: hidden,
                k: hidden,
            },
        );
        push_gemv_bf16(&mut b, &s, "q3d-mtp-fc-e", &gfe, &ne, &fe, false)?;
        push_gemv_bf16(&mut b, &s, "q3d-mtp-fc-h", &gfh, &nh, &fh, false)?;
        let addp = b.uni(
            "q3d-mtp-add-p",
            ResScaleParams {
                n: hidden as u32,
                n_words: hw as u32,
                scale: 1.0,
                ..Default::default()
            },
        );
        let add_grid = b.grid1(hw as u64, 256);
        b.push(
            "q3d-mtp-fc-add",
            &s.resscale,
            "residual_add_scale_bf16",
            &[(0, &fe), (1, &fh), (2, &mres), (3, &addp)],
            add_grid,
        )?;

        let wln = b.upload_u32("q3d-mtp-ln-in", &pack_pairs(&w.input_ln));
        b.push(
            "q3d-mtp-rms-in",
            &s.rms,
            "rmsnorm_bf16",
            &[(0, &mres), (1, &wln), (2, &mnormed), (3, &rms1)],
            (1, 1, 1),
        )?;

        let mut kv_state: Vec<(wgpu::Buffer, u64)> = Vec::new();
        let mut no_pf: Option<Pf> = None;
        let mut mtp_rope_rows_never_mrope_swapped: Vec<(wgpu::Buffer, wgpu::Buffer)> = Vec::new();
        let kv_end = build_attn(
            &mut b,
            &s,
            &cfg,
            &w.attn,
            &mnormed,
            &mmix,
            &pos_buf,
            &fd_buf,
            self.max_seq,
            &mut kv_state,
            &mut no_pf,
            &mut mtp_rope_rows_never_mrope_swapped,
        )?;

        let wpost = b.upload_u32("q3d-mtp-ln-post", &pack_pairs(&w.post_attn_ln));
        b.push(
            "q3d-mtp-rmsres-post",
            &s.rmsres,
            "rmsnorm_residual_bf16",
            &[
                (0, &mmix),
                (1, &mres),
                (2, &wpost),
                (3, &mnormed_post),
                (4, &rms1),
            ],
            (1, 1, 1),
        )?;
        build_mlp(&mut b, &s, &cfg, &w.mlp, &mnormed_post, &mmlp_out, &mut no_pf)?;
        b.push(
            "q3d-mtp-resadd",
            &s.resscale,
            "residual_add_scale_bf16",
            &[(0, &mmlp_out), (1, &mres), (2, &hid), (3, &addp)],
            add_grid,
        )?;
        let wfin = b.upload_u32("q3d-mtp-ln-final", &pack_pairs(&w.final_norm));
        b.push(
            "q3d-mtp-rms-final",
            &s.rms,
            "rmsnorm_bf16",
            &[(0, &hid), (1, &wfin), (2, &self.final_x), (3, &rms1)],
            (1, 1, 1),
        )?;
        b.flush_staging();

        self.mtp = Some(MtpDraft {
            passes: std::mem::take(&mut b.passes),
            kv_end,
            pos_buf,
            fd_buf,
            hid,
            len: 0,
            round_base: None,
            validated_kv: false,
            validated_full: false,
            _buffers: std::mem::take(&mut b.buffers),
        });
        Ok(())
    }

    fn mtp_run(
        &mut self,
        tok: u32,
        pos: usize,
        full: bool,
        pre: MtpHid,
        after: MtpAfter,
    ) -> Result<Option<u32>> {
        anyhow::ensure!((tok as usize) < self.vocab, "mtp token {tok} out of vocab");
        anyhow::ensure!(
            pos < self.max_seq,
            "mtp kv full at {pos} (max_seq {})",
            self.max_seq
        );
        let row_bytes = (self.config.hidden_size / 2 * 4) as u64;
        self.ctx
            .queue
            .write_buffer(&self.tok_buf, 0, bytemuck::bytes_of(&(tok as i32)));
        let needs_scope;
        {
            let m = self.mtp.as_ref().expect("mtp_run without an attached head");
            self.ctx
                .queue
                .write_buffer(&m.pos_buf, 0, bytemuck::bytes_of(&(pos as i32)));
            if full {
                let mut fd = self.fd_base;
                fd.total = (pos + 1) as u32;
                self.ctx
                    .queue
                    .write_buffer(&m.fd_buf, 0, bytemuck::bytes_of(&fd));
            }
            needs_scope = if full {
                !m.validated_full
            } else {
                !m.validated_kv
            };
            let scope = needs_scope.then(|| {
                self.ctx
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation)
            });
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            match pre {
                MtpHid::Keep => {}
                MtpHid::Zero => enc.clear_buffer(&m.hid, 0, None),
                MtpHid::VerifyRow(r) => {
                    let v = self
                        .verify
                        .as_ref()
                        .expect("mtp catch-up without the verify graph");
                    enc.copy_buffer_to_buffer(
                        &v.res2_rows,
                        r as u64 * row_bytes,
                        &m.hid,
                        0,
                        row_bytes,
                    );
                }
            }
            {
                let gather = &self.passes[..self.embed_gather_end];
                let mtp_slice: &[Pass] = if full { &m.passes } else { &m.passes[..m.kv_end] };
                let head: &[Pass] = if full {
                    &self.passes[self.head_start..]
                } else {
                    &[]
                };
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in gather.iter().chain(mtp_slice).chain(head) {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
            match after {
                MtpAfter::Nothing => {}
                MtpAfter::HidFromRes2 => {
                    enc.copy_buffer_to_buffer(&self.res2, 0, &m.hid, 0, row_bytes)
                }
                MtpAfter::HidFromVerifyRow(r) => {
                    let v = self
                        .verify
                        .as_ref()
                        .expect("mtp reanchor without the verify graph");
                    enc.copy_buffer_to_buffer(
                        &v.res2_rows,
                        r as u64 * row_bytes,
                        &m.hid,
                        0,
                        row_bytes,
                    );
                }
            }
            self.ctx.queue.submit([enc.finish()]);
            if let Some(scope) = scope {
                if let Some(e) = pollster::block_on(scope.pop()) {
                    anyhow::bail!("qwen3_5_dense_wgpu mtp step validation: {e}");
                }
            }
        }
        if needs_scope {
            let m = self.mtp.as_mut().expect("mtp validated flag");
            if full {
                m.validated_full = true;
            } else {
                m.validated_kv = true;
            }
        }
        if !full {
            return Ok(None);
        }
        let t: Vec<u32> = dispatch::read_back(self.ctx, &self.token_out, 1)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok(Some(t[0]))
    }

    pub fn mtp_draft_round(&mut self, anchor: u32, k: usize) -> Result<Vec<u32>> {
        anyhow::ensure!(k >= 1, "mtp draft round needs k >= 1");
        anyhow::ensure!(
            self.verify.as_ref().is_some_and(|v| v.pending.is_none()),
            "mtp draft round during a pending verify_chain"
        );
        let l = self.mtp_len();
        anyhow::ensure!(
            self.mtp.is_some() && l == self.pos,
            "mtp kv desync: drafter holds {l} rows but the trunk committed {}; \
             {MTP_HEAD_CONVENTIONS_ZERO_CENTERED_NORMS_EMB_FIRST_FC_AND_SHIFT_BY_ONE_KV_WITH_A_ZERO_HIDDEN_AT_POS_0}",
            self.pos
        );
        anyhow::ensure!(
            l + k <= self.max_seq && self.pos + k + 1 <= self.max_seq,
            "mtp round does not fit: committed={l}, k={k}, max_seq={}",
            self.max_seq
        );
        self.mtp.as_mut().expect("mtp attached").round_base = Some(l);
        let mut drafts = Vec::with_capacity(k);
        let mut t = anchor;
        for j in 0..k {
            let pre = if l + j == 0 { MtpHid::Zero } else { MtpHid::Keep };
            let d = self
                .mtp_run(t, l + j, true, pre, MtpAfter::Nothing)?
                .expect("full mtp step returns a token");
            self.mtp.as_mut().expect("mtp attached").len += 1;
            drafts.push(d);
            t = d;
        }
        Ok(drafts)
    }

    pub fn mtp_post_verify(&mut self, accepted: &[u32]) -> Result<()> {
        let base = self
            .mtp
            .as_mut()
            .and_then(|m| m.round_base.take())
            .ok_or_else(|| anyhow::anyhow!("mtp_post_verify without a preceding mtp_draft_round"))?;
        let n = accepted.len();
        for (j, &t) in accepted.iter().enumerate() {
            let after = if j + 1 == n {
                MtpAfter::HidFromVerifyRow(n)
            } else {
                MtpAfter::Nothing
            };
            self.mtp_run(t, base + 1 + j, false, MtpHid::VerifyRow(j), after)?;
        }
        if n == 0 {
            let row_bytes = (self.config.hidden_size / 2 * 4) as u64;
            let m = self.mtp.as_ref().expect("mtp attached");
            let v = self
                .verify
                .as_ref()
                .expect("mtp reanchor without the verify graph");
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&v.res2_rows, 0, &m.hid, 0, row_bytes);
            self.ctx.queue.submit([enc.finish()]);
        }
        self.mtp.as_mut().expect("mtp attached").len = base + 1 + n;
        anyhow::ensure!(
            self.mtp_len() == self.pos,
            "mtp kv catch-up desync: drafter holds {} rows, trunk committed {}",
            self.mtp_len(),
            self.pos
        );
        Ok(())
    }

    fn mtp_sync_committed_step(&mut self, token: u32) -> Result<()> {
        let pos_before = self.pos - 1;
        anyhow::ensure!(
            self.mtp_len() == pos_before,
            "mtp kv desync at an M=1 step: drafter holds {} rows, trunk committed {pos_before}",
            self.mtp_len()
        );
        let pre = if pos_before == 0 {
            MtpHid::Zero
        } else {
            MtpHid::Keep
        };
        self.mtp_run(token, pos_before, false, pre, MtpAfter::HidFromRes2)?;
        self.mtp.as_mut().expect("mtp attached").len += 1;
        Ok(())
    }

    fn mtp_sync_committed_chunk(&mut self, live_tokens: &[u32]) -> Result<()> {
        let live = live_tokens.len();
        let base = self.pos - live;
        anyhow::ensure!(
            self.mtp_len() == base,
            "mtp kv desync at a prefill chunk: drafter holds {} rows, trunk committed {base}",
            self.mtp_len()
        );
        {
            let v = self
                .verify
                .as_ref()
                .expect("mtp chunk sync without the verify graph");
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(&v.resadd.pipeline);
                pass.set_bind_group(0, &v.resadd.bind, &[]);
                pass.dispatch_workgroups(v.resadd.grid.0, v.resadd.grid.1, v.resadd.grid.2);
            }
            self.ctx.queue.submit([enc.finish()]);
        }
        for (i, &t) in live_tokens.iter().enumerate() {
            let pre = if base + i == 0 {
                MtpHid::Zero
            } else if i == 0 {
                MtpHid::Keep
            } else {
                MtpHid::VerifyRow(i - 1)
            };
            let after = if i + 1 == live {
                MtpAfter::HidFromVerifyRow(live - 1)
            } else {
                MtpAfter::Nothing
            };
            self.mtp_run(t, base + i, false, pre, after)?;
            self.mtp.as_mut().expect("mtp attached").len += 1;
        }
        Ok(())
    }
}

fn row_chunk(ctx: &WgpuContext, hidden: usize) -> usize {
    let limit = ctx
        .caps
        .max_storage_buffer_binding_size
        .clamp(1 << 20, 1u64 << 30);
    let per_row = (hidden * 2) as u64;
    ((limit / per_row) as usize).max(2) & !1usize
}

fn push_gemv_bf16(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    out_f32: bool,
) -> Result<()> {
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "q3d-gemvb-p",
        GemvBf16Params {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: (w.k / 2) as u32,
            x_off_words: 0,
            y_off_words: 0,
            alpha: 1.0,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.gemv_bf16,
        "q3w_gemv_bf16",
        &[(0, &w.w), (1, x), (2, &p), (3, y)],
        grid,
    )
}

struct Fp8Gpu {
    w: wgpu::Buffer,
    s: wgpu::Buffer,
    s_plain: Option<wgpu::Buffer>,
    n: usize,
    k: usize,
}

pub const PF_COOP_UPLOADS_PLAIN_FP8_SCALES_BECAUSE_WQ16_E4M3_PLAIN_DECODES_WITHOUT_THE_2P120_FOLD:
    &str = "the decode gemv's scale buffer carries fold_scales_for_e4m3_shift_decode (2^120 \
     folded in for the shift decode); the coop w8a16 kernel decodes with wq16_e4m3_plain and \
     multiplies the unfolded per-row scale, so the coop arm uploads l.scales verbatim as an \
     additive 4*n-byte buffer per fp8 projection";

fn upload_fp8(b: &mut Builder, label: &str, l: &HostFp8Lin) -> Fp8Gpu {
    let s_plain = if pf_coop_active(b.ctx) && l.n.is_multiple_of(16) && l.k.is_multiple_of(16) {
        Some(b.upload_f32(&format!("{label}-sp"), &l.scales))
    } else {
        None
    };
    Fp8Gpu {
        w: b.upload_u32(label, &l.packed),
        s: b.upload_f32(
            &format!("{label}-s"),
            &nv_kernels::shift_decode_fold::fold_scales_for_e4m3_shift_decode(&l.scales),
        ),
        s_plain,
        n: l.n,
        k: l.k,
    }
}

fn push_gemv_fp8_rowscale(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Fp8Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    out_f32: bool,
) -> Result<()> {
    anyhow::ensure!(
        w.k.is_multiple_of(4),
        "{label}: fp8 rowscale gemv packs 4 e4m3 bytes per word; k {} % 4 != 0",
        w.k
    );
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let row_words = w.k / 4;
    let p = b.uni(
        "q3d-gemvf8-p",
        GemvBf16Params {
            n_rows: w.n as u32,
            k_words: row_words as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: row_words as u32,
            x_off_words: 0,
            y_off_words: 0,
            alpha: 1.0,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.gemv_bf16,
        "q3w_gemv_fp8_rowscale",
        &[(0, &w.w), (1, x), (2, &p), (3, y), (4, &w.s)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_dn_merged(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    qkv: &Fp8Gpu,
    z: &Fp8Gpu,
    ab: &Bf16Gpu,
    x: &wgpu::Buffer,
    y_qkv: &wgpu::Buffer,
    y_z: &wgpu::Buffer,
    y_ab: &wgpu::Buffer,
) -> Result<()> {
    anyhow::ensure!(
        qkv.k == z.k && qkv.k == ab.k,
        "{label}: merged DN projections read one normed x, so k must agree; got qkv {} z {} ab {}",
        qkv.k,
        z.k,
        ab.k
    );
    anyhow::ensure!(
        qkv.k.is_multiple_of(4),
        "{label}: fp8 rowscale gemv packs 4 e4m3 bytes per word; k {} % 4 != 0",
        qkv.k
    );
    let qkv_pairs = qkv.n.div_ceil(2);
    let z_pairs = z.n.div_ceil(2);
    let ab_pairs = ab.n.div_ceil(2);
    let grid = b.grid1((qkv_pairs + z_pairs + ab_pairs) as u64, 1);
    let p = b.uni(
        "q3d-dn-merged-p",
        GemvDnMergedParams {
            qkv_pairs: qkv_pairs as u32,
            z_pairs: z_pairs as u32,
            ab_pairs: ab_pairs as u32,
            qkv_rows: qkv.n as u32,
            z_rows: z.n as u32,
            ab_rows: ab.n as u32,
            fp8_row_words: (qkv.k / 4) as u32,
            bf16_row_words: (ab.k / 2) as u32,
            groups_x: grid.0,
            ..Default::default()
        },
    );
    b.push(
        label,
        &s.gemv_bf16,
        "q3w_gemv_dn_merged_fp8_qkv_fp8_z_bf16_ab",
        &[
            (0, &qkv.w),
            (1, x),
            (3, y_qkv),
            (4, &qkv.s),
            (5, &z.w),
            (6, &z.s),
            (7, y_z),
            (8, &ab.w),
            (9, y_ab),
            (10, &p),
        ],
        grid,
    )
}

pub const PF_FP8_REPLAY_ROWS_REPLAY_THE_DECODE_GEMV_PER_SLOT: &str =
    "NV_Q3D_PF_FP8_REPLAY=1 makes the M-row graph's DeltaNet projections replay \
     q3w_gemv_fp8_rowscale once per slot with slot offsets, so each row's projection \
     bits equal the decode step it stands in for at fp8 defaults (the verify_chain \
     bit gate's first divergence source, task #13). The bf16-twin gemm stays the \
     default because it streams the weights once for all M rows; the replay streams \
     them M times and only a small verify graph can afford that.";

fn pf_fp8_replay_enabled() -> bool {
    std::env::var("NV_Q3D_PF_FP8_REPLAY").ok().as_deref() == Some("1")
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_fp8_rowscale_pf(
    b: &mut Builder,
    s: &Sources,
    pf: &Pf,
    label: &str,
    w: &Fp8Gpu,
    x: &wgpu::Buffer,
    x_stride_words: usize,
    y: &wgpu::Buffer,
    y_stride_words: usize,
) -> Result<()> {
    anyhow::ensure!(
        pf.m <= 16,
        "{label}: NV_Q3D_PF_FP8_REPLAY streams the full weight once per slot, which only the \
         small verify graphs can afford; m={} is a coop-scale prefill and the replay is refused",
        pf.m
    );
    anyhow::ensure!(
        w.k.is_multiple_of(4),
        "{label}: fp8 rowscale gemv packs 4 e4m3 bytes per word; k {} % 4 != 0",
        w.k
    );
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let row_words = w.k / 4;
    for slot in 0..pf.m {
        let sl = format!("{label}-s{slot}");
        let p = b.uni(
            &sl,
            GemvBf16Params {
                n_rows: w.n as u32,
                k_words: row_words as u32,
                groups_x: grid.0,
                out_f32: 0,
                w_row_words: row_words as u32,
                x_off_words: (slot * x_stride_words) as u32,
                y_off_words: (slot * y_stride_words) as u32,
                alpha: 1.0,
                ..Default::default()
            },
        );
        b.push_pf(
            &sl,
            &s.gemv_bf16,
            "q3w_gemv_fp8_rowscale",
            &[(0, &w.w), (1, x), (2, &p), (3, y), (4, &w.s)],
            grid,
        )?;
    }
    Ok(())
}

fn upload_bf16(b: &mut Builder, label: &str, l: &HostBf16Lin) -> Bf16Gpu {
    Bf16Gpu {
        w: b.upload_u32(label, &pack_pairs(&l.w)),
        n: l.n,
        k: l.k,
    }
}

pub const PF_COOP_UPLOADS_LINEAR_NVFP4_SCALES_ADDITIVELY_BECAUSE_THE_DECODE_GEMV_READS_THEM_SWIZZLED:
    &str = "the coop w4a16 kernel fetches block scales at gr*sf_row + kb (row-linear bytes) \
     while every shipping nvfp4 gemv reads the 128x4-swizzled layout, so the coop arm carries a \
     second n*k/16-byte scale buffer per nvfp4 weight (~17 MB/layer on qwen3.8 MLP dims); a \
     swizzled-scale coop variant in the kernel lane would delete this upload";

fn unswizzle_nvfp4_scales_linear(l: &HostNvfp4Lin) -> Vec<u8> {
    let k_blocks = l.k / NVFP4_BLOCK;
    let k_tiles = k_blocks.div_ceil(4);
    let mut out = vec![0u8; l.n * k_blocks];
    for r in 0..l.n {
        for kb in 0..k_blocks {
            let si = (((r / 128) * k_tiles + kb / 4) * 32 + r % 32) * 16 + ((r / 32) % 4) * 4
                + kb % 4;
            out[r * k_blocks + kb] = l.scales_swizzled[si];
        }
    }
    out
}

fn upload_nvfp4(b: &mut Builder, label: &str, l: &HostNvfp4Lin) -> Nvfp4Gpu {
    let scales_linear =
        if pf_coop_active(b.ctx) && l.n.is_multiple_of(16) && l.k.is_multiple_of(16) {
            Some(b.upload_u32(
                &format!("{label}-sfl"),
                &bytes_to_words(&unswizzle_nvfp4_scales_linear(l)),
            ))
        } else {
            None
        };
    Nvfp4Gpu {
        w: b.upload_u32(label, &bytes_to_words(&l.packed)),
        scales: b.upload_u32(&format!("{label}-sf"), &bytes_to_words(&l.scales_swizzled)),
        scales_linear,
        alpha: l.alpha,
        input_global: l.input_global,
        n: l.n,
        k: l.k,
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum W8Scope {
    Attn,
    Ffn,
}

pub const FP8_LMHEAD_DEFAULT_ON_FOR_THE_SAME_REASON_AS_FP8_PROJ: &str =
    "lm_head.weight ships as F8_E4M3 with per-row scales in the qwen3.8 nvfp4 checkpoints; \
     the bf16 route upcasts it to 2x the bytes for zero information (see \
     FP8_PROJ_STREAMS_CHECKPOINT_BYTES_INSTEAD_OF_A_BF16_UPCAST). NV_Q3D_WGPU_FP8_LMHEAD=0 \
     restores the upcast route";

pub fn fp8_lm_head_enabled() -> bool {
    std::env::var("NV_Q3D_WGPU_FP8_LMHEAD").ok().as_deref() != Some("0")
}

pub fn w8_mode() -> (bool, bool) {
    let v = std::env::var("NV_Q3D_WGPU_W8").unwrap_or_default();
    match v.trim() {
        "ffn" => (false, true),
        "attn" => (true, false),
        "1" | "all" => (true, true),
        _ => (false, false),
    }
}

pub fn w8_group() -> usize {
    crate::nvfp4_host::w8_group_from_env("NV_Q3D_WGPU_W8_GROUP")
}

pub use crate::nvfp4_host::assert_w8_group_divides_k;

fn w8_enabled(ctx: &WgpuContext, scope: W8Scope) -> bool {
    let (attn, ffn) = w8_mode();
    let on = match scope {
        W8Scope::Attn => attn,
        W8Scope::Ffn => ffn,
    };
    on && wk::gemv_nvfp4_v2::subgroup32_ok(ctx)
}

fn quantize_nvfp4_i8(l: &HostNvfp4Lin, group: usize) -> (Vec<u32>, Vec<f32>) {
    let st = crate::nvfp4_host::HostNvfp4ExpertStack {
        packed: l.packed.clone(),
        scales_swizzled: l.scales_swizzled.clone(),
        alphas: vec![l.alpha],
        input_globals: vec![l.input_global],
        e: 1,
        n: l.n,
        k: l.k,
    };
    crate::nvfp4_host::quantize_nvfp4_stack_i8(&st, group)
}

fn upload_nvfp4_i8(b: &mut Builder, label: &str, l: &HostNvfp4Lin) -> I8Gpu {
    let group = w8_group();
    assert_w8_group_divides_k(label, group, l.k, "NV_Q3D_WGPU_W8_GROUP");
    let (packed, scales) = quantize_nvfp4_i8(l, group);
    I8Gpu {
        w: b.upload_u32(label, &packed),
        s: b.upload_f32(&format!("{label}-s"), &scales),
        n: l.n,
        k: l.k,
        group,
    }
}

fn upload_lin(
    b: &mut Builder,
    label: &str,
    l: &HostDenseLin,
    scope: W8Scope,
    pf_rides_coop: bool,
) -> DenseLinGpu {
    match l {
        HostDenseLin::Bf16(x) => DenseLinGpu::Bf16(upload_bf16(b, label, x)),
        HostDenseLin::Fp8 { fp8, bf16 } if fp8_proj_enabled() => {
            let g = upload_fp8(b, &format!("{label}-fp8"), fp8);
            let twin = if g.s_plain.is_some() && !pf_fp8_replay_enabled() && pf_rides_coop {
                None
            } else {
                Some(upload_bf16(b, label, bf16))
            };
            DenseLinGpu::Fp8(g, twin)
        }
        HostDenseLin::Fp8 { bf16, .. } => DenseLinGpu::Bf16(upload_bf16(b, label, bf16)),
        HostDenseLin::Nvfp4(x) if w8_enabled(b.ctx, scope) => {
            DenseLinGpu::Int8(upload_nvfp4_i8(b, label, x))
        }
        HostDenseLin::Nvfp4(x) => DenseLinGpu::Nvfp4(upload_nvfp4(b, label, x)),
    }
}

fn push_gemv_i8(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &I8Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
) -> Result<()> {
    anyhow::ensure!(
        w.k.is_multiple_of(4),
        "{label}: i8 gemv needs k % 4 == 0, got {}",
        w.k
    );
    let groups = w.n.div_ceil(8) as u32;
    let p = b.uni(
        label,
        GemvI8Params {
            n_rows: w.n as u32,
            k_elems: w.k as u32,
            groups_x: groups,
            groups_per_row: w.k.checked_div(w.group).unwrap_or(1) as u32,
            group_shift: if w.group > 0 {
                (w.group / 4).trailing_zeros()
            } else {
                0
            },
            ..Default::default()
        },
    );
    let entry = if w.group > 0 {
        "q3d_gemv_i8g"
    } else {
        "q3d_gemv_i8"
    };
    b.push(
        label,
        &s.gemv_i8,
        entry,
        &[(0, &w.w), (1, &w.s), (2, x), (3, y), (4, &p)],
        (groups, 1, 1),
    )
}

fn quant_for(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    x: &wgpu::Buffer,
    k: usize,
    consumers: &[&DenseLinGpu],
) -> Result<Option<QuantIn>> {
    let mut global: Option<f32> = None;
    for c in consumers {
        let Some(g) = c.input_global() else { continue };
        match global {
            None => global = Some(g),
            Some(prev) => anyhow::ensure!(
                (prev - g).abs() <= 1e-6,
                "{label}: consumers disagree on input_global_scale ({prev} vs {g})"
            ),
        }
    }
    let Some(global) = global else {
        return Ok(None);
    };
    let k_blocks = k / NVFP4_BLOCK;
    anyhow::ensure!(
        k.is_multiple_of(NVFP4_BLOCK) && k_blocks.is_multiple_of(4),
        "{label}: k {k} must be a multiple of {}",
        NVFP4_BLOCK * 4
    );
    let xq = b.zeros(label, (k / 2) as u64);
    let xs = b.zeros(label, (k_blocks.div_ceil(4) * 4) as u64);
    let sel = b.upload_u32(label, &[0u32]);
    let alpha_dummy = b.upload_f32(label, &[1.0f32]);
    let globals = b.upload_f32(label, &[global]);
    let p = b.uni(
        label,
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
        "q3w_quant_rows",
        &[
            (10, x),
            (11, &p),
            (12, &xq),
            (13, &xs),
            (14, &sel),
            (15, &globals),
        ],
        (gx, 1, 1),
    )?;
    Ok(Some(QuantIn {
        xq,
        xs,
        sel,
        alpha_dummy,
        xm: None,
    }))
}

fn push_silu_mul_quant_rows(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    y_gate: &wgpu::Buffer,
    y_up: &wgpu::Buffer,
    w: &Nvfp4Gpu,
    k: usize,
) -> Result<QuantIn> {
    let k_blocks = k / NVFP4_BLOCK;
    anyhow::ensure!(
        k.is_multiple_of(NVFP4_BLOCK) && k_blocks.is_multiple_of(4),
        "{label}: k {k} must be a multiple of {}",
        NVFP4_BLOCK * 4
    );
    let xq = b.zeros(label, (k / 2) as u64);
    let xs = b.zeros(label, (k_blocks.div_ceil(4) * 4) as u64);
    let sel = b.upload_u32(label, &[0u32]);
    let alpha_dummy = b.upload_f32(label, &[1.0f32]);
    let globals = b.upload_f32(label, &[w.input_global]);
    let p = b.uni(
        label,
        QuantRowsParams {
            k_blocks: k_blocks as u32,
            n_slots: 1,
            use_sel: 0,
            x_slot_stride_elems: 0,
        },
    );
    let pp = b.uni(
        label,
        SiluPairParams {
            u_off_elems: 0,
            ..Default::default()
        },
    );
    let (entry, wg_blocks) = match quant_lane_entry(b.ctx, "q3w_silu_mul_quant") {
        Some((e, wg)) => (e, wg / 8),
        None => ("q3w_silu_mul_quant", 256),
    };
    let gx = (k_blocks as u32).div_ceil(wg_blocks as u32).max(1);
    b.push(
        label,
        &s.quant,
        entry,
        &[
            (11, &p),
            (12, &xq),
            (13, &xs),
            (14, &sel),
            (15, &globals),
            (16, y_gate),
            (17, y_up),
            (18, &pp),
        ],
        (gx, 1, 1),
    )?;
    Ok(QuantIn {
        xq,
        xs,
        sel,
        alpha_dummy,
        xm: None,
    })
}

fn push_gemv_nvfp4(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Nvfp4Gpu,
    q: &QuantIn,
    y: &wgpu::Buffer,
) -> Result<()> {
    let k_blocks = w.k / NVFP4_BLOCK;
    let route = nvfp4_v2_route(b.ctx, w.n, w.k, k_blocks, 1);
    let grid = match &route {
        Some(r) => b.grid1(w.n as u64, r.rows_per_group),
        None => b.grid1(w.n.div_ceil(2) as u64, 1),
    };
    let p = b.uni(
        label,
        GemvNvfp4Params {
            alpha: w.alpha,
            n_rows: w.n as u32,
            k_blocks: k_blocks as u32,
            k_tiles: k_blocks.div_ceil(4) as u32,
            groups_x: grid.0,
            w_e_stride_vec2: 0,
            sf_e_stride_bytes: 0,
            x_slot_stride_vec2: 0,
            xsf_slot_stride_bytes: 0,
            y_slot_stride_words: (w.n / 2) as u32,
            per_expert_alpha: 0,
            m_slots_sharing_expert_zero: 0,
        },
    );
    if let Some(r) = route {
        let (w_slot, x_slot) = if r.vec4 { (18, 19) } else { (10, 12) };
        return b.push(
            label,
            &r.source,
            r.entry,
            &[
                (w_slot, &w.w),
                (11, &w.scales),
                (x_slot, &q.xq),
                (13, &q.xs),
                (14, &p),
                (15, y),
                (16, &q.sel),
                (17, &q.alpha_dummy),
            ],
            (grid.0, grid.1, 1),
        );
    }
    b.push(
        label,
        &s.gemv_nvfp4,
        "q3w_gemv_nvfp4",
        &[
            (10, &w.w),
            (11, &w.scales),
            (12, &q.xq),
            (13, &q.xs),
            (14, &p),
            (15, y),
            (16, &q.sel),
            (17, &q.alpha_dummy),
        ],
        (grid.0, grid.1, 1),
    )
}

fn push_gemv_nvfp4_2w(
    b: &mut Builder,
    label: &str,
    wa: &Nvfp4Gpu,
    wb: &Nvfp4Gpu,
    q: &QuantIn,
    ya: &wgpu::Buffer,
    yb: &wgpu::Buffer,
) -> Result<bool> {
    if wa.n != wb.n
        || wa.k != wb.k
        || b.ctx.caps.max_storage_buffers_per_shader_stage < MLP_GEMV_2W_BINDS_TEN_STORAGE_BUFFERS
    {
        return Ok(false);
    }
    let k_blocks = wa.k / NVFP4_BLOCK;
    let Some(r) = nvfp4_v2_route(b.ctx, wa.n, wa.k, k_blocks, 1) else {
        return Ok(false);
    };
    if r.entry != MLP_GEMV_2W_ROUTE_IS_EXACTLY_THE_MROW2_DECODE_ROUTE_ANY_OTHER_ENTRY_KEEPS_THE_PAIR {
        return Ok(false);
    }
    let grid = b.grid1(wa.n as u64, r.rows_per_group);
    if grid.1 != 1 {
        return Ok(false);
    }
    let alphas = b.upload_f32(label, &[wa.alpha, wb.alpha]);
    let p = b.uni(
        label,
        GemvNvfp4Params {
            alpha: wa.alpha,
            n_rows: wa.n as u32,
            k_blocks: k_blocks as u32,
            k_tiles: k_blocks.div_ceil(4) as u32,
            groups_x: grid.0,
            w_e_stride_vec2: 0,
            sf_e_stride_bytes: 0,
            x_slot_stride_vec2: 0,
            xsf_slot_stride_bytes: 0,
            y_slot_stride_words: (wa.n / 2) as u32,
            per_expert_alpha: 0,
            m_slots_sharing_expert_zero: 0,
        },
    );
    b.push(
        label,
        &r.source,
        "q3w_gemv_nvfp4_mrow2_2w",
        &[
            (18, &wa.w),
            (11, &wa.scales),
            (19, &q.xq),
            (13, &q.xs),
            (14, &p),
            (15, ya),
            (16, &q.sel),
            (17, &alphas),
            (21, &wb.w),
            (22, &wb.scales),
            (23, yb),
        ],
        (grid.0, 2, 1),
    )?;
    Ok(true)
}

fn push_gemv(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &DenseLinGpu,
    x: &wgpu::Buffer,
    q: Option<&QuantIn>,
    y: &wgpu::Buffer,
) -> Result<()> {
    match w {
        DenseLinGpu::Bf16(w) => push_gemv_bf16(b, s, label, w, x, y, false),
        DenseLinGpu::Fp8(w, _) => push_gemv_fp8_rowscale(b, s, label, w, x, y, false),
        DenseLinGpu::Int8(w) => push_gemv_i8(b, s, label, w, x, y),
        DenseLinGpu::Nvfp4(w) => {
            let q = q.ok_or_else(|| anyhow::anyhow!("{label}: nvfp4 gemv without quantized x"))?;
            push_gemv_nvfp4(b, s, label, w, q, y)
        }
    }
}

fn pf_coop_covers(pf: &Pf, w: &DenseLinGpu) -> bool {
    if pf.coop.is_none() {
        return false;
    }
    match w {
        DenseLinGpu::Fp8(f, _) => {
            !pf_fp8_replay_enabled()
                && f.s_plain.is_some()
                && f.n.is_multiple_of(16)
                && f.k.is_multiple_of(16)
        }
        DenseLinGpu::Nvfp4(w) => {
            w.scales_linear.is_some() && w.n.is_multiple_of(16) && w.k.is_multiple_of(16)
        }
        DenseLinGpu::Bf16(_) | DenseLinGpu::Int8(_) => false,
    }
}

fn nvfp4_input_global_folded_into_x_codes_by_q3w_qz_core_so_the_a16_route_multiplies_it(w: &Nvfp4Gpu) -> f32 {
    if w.input_global == 0.0 || !w.input_global.is_finite() {
        1.0
    } else {
        w.input_global
    }
}

fn pf_coop_ku(k: usize, tn: u32, sg: u32) -> u32 {
    let cols = 16 * tn * sg;
    for ku in [4u32, 2] {
        if k.is_multiple_of(16 * ku as usize)
            && cols * 16 * ku
                <= wk::gemm_coop_f16::W4A16_STAGE_BUDGET_F16_IS_HALF_THE_48K_WORKGROUP_LIMIT
        {
            return ku;
        }
    }
    1
}

enum PfCoopW<'a> {
    Wq(wk::gemm_coop_f16::WqFmt, &'a wgpu::Buffer, &'a wgpu::Buffer),
    F16(&'a wgpu::Buffer),
}

pub const NARROW_COLS_WIN_ON_TALL_K_SHAPES_ON_THE_DOWN_PROJ: &str =
    "the coop rate probe's best down-proj (5120x17408) config is tm8 tn1 sg1 ku4 at \
     1.63ms/dispatch vs 2.22 for the shared tm8 tn2 sg2 ku4 the plan carries; k>n \
     selects the narrow-column arm per projection. NV_Q3D_PF_COOP_NARROW=0 restores \
     the shared config.";

fn pf_coop_narrow_tall_enabled() -> bool {
    std::env::var("NV_Q3D_PF_COOP_NARROW").ok().as_deref() != Some("0")
}

fn pf_coop_cast_x(
    b: &mut Builder,
    pf: &Pf,
    label: &str,
    x: &wgpu::Buffer,
    x_stride_words: usize,
    k: usize,
) -> Result<()> {
    let cp = pf
        .coop
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("{label}: coop cast without a coop plan"))?;
    anyhow::ensure!(
        k.is_multiple_of(16),
        "{label}: the 16x16 coop fragments need k % 16 == 0, got {k}"
    );
    let kw = (k / 2) as u32;
    let grid = b.grid1(cp.m_alloc as u64 * kw as u64, 256);
    let p = b.uni(
        label,
        CoopCastParams {
            k_elems: k as u32,
            x_stride_words: x_stride_words as u32,
            rows_in: pf.m as u32,
            rows_out: cp.m_alloc,
            groups_x: grid.0,
            ..Default::default()
        },
    );
    b.push_pf(
        label,
        &cp.helpers,
        "q3w_pf_coop_cast_x",
        &[(0, x), (1, &cp.x_f16), (2, &p)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn pf_coop_gemm(
    b: &mut Builder,
    pf: &Pf,
    label: &str,
    w: PfCoopW<'_>,
    n: usize,
    k: usize,
    alpha: f32,
    y: &wgpu::Buffer,
    y_stride_words: usize,
) -> Result<()> {
    use wk::gemm_coop_f16 as cf;
    let cp = pf
        .coop
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("{label}: coop gemm without a coop plan"))?;
    anyhow::ensure!(
        n.is_multiple_of(16) && k.is_multiple_of(16),
        "{label}: the coop epilogue stores whole 16x16 fragments; n={n} k={k} must be 16-aligned"
    );
    anyhow::ensure!(
        y_stride_words * 2 == n,
        "{label}: pf consumers read bf16 rows of n/2 words; y_stride_words {y_stride_words} != {}",
        n / 2
    );
    let narrow = k > n && pf_coop_narrow_tall_enabled();
    let (tm, tn, sg) = if narrow {
        (cp.tm, 1, 1)
    } else {
        (cp.tm, cp.tn, cp.sg)
    };
    let ku = pf_coop_ku(k, tn, sg);
    let y16 = matches!(&w, PfCoopW::Wq(..)) && cp.m_alloc as usize == pf.m;
    let (src, entry) = match &w {
        PfCoopW::Wq(fmt, _, _) if y16 => (
            cf::source_wq16_act_y16(*fmt, cf::WqAct::F16, tm, tn, sg, ku),
            cf::entry_wq16_act_y16(*fmt, cf::WqAct::F16, tm, tn, sg, ku),
        ),
        PfCoopW::Wq(fmt, _, _) => (
            cf::source_wq16(*fmt, tm, tn, sg, ku),
            cf::entry_wq16(*fmt, tm, tn, sg, ku),
        ),
        PfCoopW::F16(_) => {
            let g = cf::CoopGemm::new(16, cf::Operand::F16);
            (g.source(tm, tn, sg, ku), g.entry(tm, tn, sg, ku))
        }
    };
    let cols = 16 * tn * sg;
    let bn = (n as u32).div_ceil(cols);
    let bm = cp.m_alloc / (16 * tm);
    let grid = b.grid1(bm as u64 * bn as u64, 1);
    let p = b.uni(
        label,
        cf::CoopGemmParams {
            n_rows: n as u32,
            k_elems: k as u32,
            m_rows: cp.m_alloc,
            blocks_n: bn,
            y_stride: n as u32,
            groups_x: grid.0,
            pad0: if y16 { alpha.to_bits() } else { 0 },
            pad1: 0,
        },
    );
    if y16 {
        if let PfCoopW::Wq(_, wq, sf) = &w {
            return b.push_pf(
                label,
                &src,
                &entry,
                &[
                    (0, *wq),
                    (1, &cp.x_f16),
                    (2, y),
                    (3, &p),
                    (4, &cp.zero),
                    (5, *sf),
                ],
                grid,
            );
        }
    }
    match &w {
        PfCoopW::Wq(_, wq, sf) => b.push_pf(
            label,
            &src,
            &entry,
            &[
                (0, *wq),
                (1, &cp.x_f16),
                (2, &cp.y_f32),
                (3, &p),
                (4, &cp.zero),
                (5, *sf),
            ],
            grid,
        )?,
        PfCoopW::F16(wf) => b.push_pf(
            label,
            &src,
            &entry,
            &[
                (0, *wf),
                (1, &cp.x_f16),
                (2, &cp.y_f32),
                (3, &p),
                (4, &cp.zero),
            ],
            grid,
        )?,
    }
    let pl = format!("{label}-pack");
    let pgrid = b.grid1(pf.m as u64 * (n / 2) as u64, 256);
    let pp = b.uni(
        &pl,
        CoopPackParams {
            n_cols: n as u32,
            y_src_stride: n as u32,
            y_dst_stride_words: y_stride_words as u32,
            rows: pf.m as u32,
            groups_x: pgrid.0,
            pad0: 0,
            pad1: 0,
            alpha,
        },
    );
    b.push_pf(
        &pl,
        &cp.helpers,
        "q3w_pf_coop_pack_y",
        &[(4, &cp.y_f32), (5, y), (6, &pp)],
        pgrid,
    )
}

fn quant_for_m_rows(
    b: &mut Builder,
    s: &Sources,
    pf: &Pf,
    label: &str,
    x: &wgpu::Buffer,
    k: usize,
    consumers: &[&DenseLinGpu],
) -> Result<Option<QuantIn>> {
    let mut global: Option<f32> = None;
    for c in consumers {
        let Some(g) = c.input_global() else { continue };
        match global {
            None => global = Some(g),
            Some(prev) => anyhow::ensure!(
                (prev - g).abs() <= 1e-6,
                "{label}: consumers disagree on input_global_scale ({prev} vs {g})"
            ),
        }
    }
    let Some(global) = global else {
        return Ok(None);
    };
    anyhow::ensure!(
        pf.m <= 16,
        "{label}: the m-row nvfp4 quant/gemv routes (slotshared included) carry per-slot state \
         sized for m 2..=16; m={} runs only through the NV_Q3D_PF_COOP coop route, which does \
         not quantize activations",
        pf.m
    );
    let m = pf.m;
    let k_blocks = k / NVFP4_BLOCK;
    anyhow::ensure!(
        k.is_multiple_of(NVFP4_BLOCK) && k_blocks.is_multiple_of(4),
        "{label}: k {k} must be a multiple of {}",
        NVFP4_BLOCK * 4
    );
    let xq = b.zeros(label, (m * k / 2) as u64);
    let xs = b.zeros(label, (m * k_blocks).max(4) as u64);
    let sel = b.upload_u32(label, &vec![0u32; m]);
    let alpha_dummy = b.upload_f32(label, &[1.0f32]);
    let globals = b.upload_f32(label, &vec![global; m]);
    let p = b.uni(
        label,
        QuantRowsParams {
            k_blocks: k_blocks as u32,
            n_slots: m as u32,
            use_sel: 0,
            x_slot_stride_elems: k as u32,
        },
    );
    let gx = (k_blocks as u32).div_ceil(256).max(1);
    b.push_pf(
        label,
        &s.quant,
        "q3w_quant_rows",
        &[
            (10, x),
            (11, &p),
            (12, &xq),
            (13, &xs),
            (14, &sel),
            (15, &globals),
        ],
        (gx, m as u32, 1),
    )?;
    let xm = if crate::qwen3_5_moe_wgpu::nvfp4_slotshared_enabled() {
        let xm = b.zeros(label, (m * k_blocks * 16) as u64);
        let mp = b.uni(
            label,
            GemvNvfp4Params {
                k_blocks: k_blocks as u32,
                x_slot_stride_vec2: k_blocks as u32,
                m_slots_sharing_expert_zero: m as u32,
                ..Default::default()
            },
        );
        b.push_pf(
            label,
            &crate::qwen3_5_moe_wgpu::nvfp4_slotshared_sources(),
            crate::qwen3_5_moe_wgpu::SLOTSHARED_MAP_ENTRY,
            &[(12, &xq), (14, &mp), (20, &xm)],
            (gx, m as u32, 1),
        )?;
        Some(xm)
    } else {
        None
    };
    Ok(Some(QuantIn {
        xq,
        xs,
        sel,
        alpha_dummy,
        xm,
    }))
}

fn push_gemm_pf_nvfp4(
    b: &mut Builder,
    s: &Sources,
    pf: &Pf,
    label: &str,
    w: &Nvfp4Gpu,
    q: &QuantIn,
    y: &wgpu::Buffer,
    y_stride_words: usize,
) -> Result<()> {
    anyhow::ensure!(
        pf.m <= 16,
        "{label}: the nvfp4 m-row gemv routes dispatch one z-slice per slot and were only ever \
         built for m 2..=16; m={} runs only through the NV_Q3D_PF_COOP coop route",
        pf.m
    );
    anyhow::ensure!(
        y_stride_words == w.n / 2,
        "{label}: the nvfp4 M-row GEMV writes one packed row of n/2 words per slot, \
         so the caller's y row stride {y_stride_words} must equal n/2 = {}",
        w.n / 2
    );
    let k_blocks = w.k / NVFP4_BLOCK;
    let slotshared = nvfp4_v2_route_slotshared(b.ctx, w.n, w.k, k_blocks, pf.m);
    let route = match slotshared {
        Some(r) => Some(r),
        None => nvfp4_v2_route(b.ctx, w.n, w.k, k_blocks, pf.m),
    };
    let one_weight_sweep = route
        .as_ref()
        .is_some_and(|r| r.entry == crate::qwen3_5_moe_wgpu::SLOTSHARED_ENTRY);
    let grid = match &route {
        Some(r) => b.grid1(w.n as u64, r.rows_per_group),
        None => b.grid1(w.n.div_ceil(2) as u64, 1),
    };
    let p = b.uni(
        label,
        GemvNvfp4Params {
            alpha: w.alpha,
            n_rows: w.n as u32,
            k_blocks: k_blocks as u32,
            k_tiles: k_blocks.div_ceil(4) as u32,
            groups_x: grid.0,
            w_e_stride_vec2: 0,
            sf_e_stride_bytes: 0,
            x_slot_stride_vec2: k_blocks as u32,
            xsf_slot_stride_bytes: k_blocks as u32,
            y_slot_stride_words: (w.n / 2) as u32,
            per_expert_alpha: 0,
            m_slots_sharing_expert_zero: if one_weight_sweep { pf.m as u32 } else { 0 },
        },
    );
    let m = if one_weight_sweep { 1 } else { pf.m as u32 };
    if let Some(r) = route {
        let (w_slot, x_slot) = if r.vec4 { (18, 19) } else { (10, 12) };
        let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
            (w_slot, &w.w),
            (11, &w.scales),
            (13, &q.xs),
            (14, &p),
            (15, y),
            (16, &q.sel),
            (17, &q.alpha_dummy),
        ];
        if one_weight_sweep {
            let xm = q.xm.as_ref().ok_or_else(|| {
                anyhow::anyhow!("{label}: slotshared route without the i8-mapped x buffer")
            })?;
            binds.push((20, xm));
        } else {
            binds.push((x_slot, &q.xq));
        }
        return b.push_pf(label, &r.source, r.entry, &binds, (grid.0, grid.1, m));
    }
    b.push_pf(
        label,
        &s.gemv_nvfp4,
        "q3w_gemv_nvfp4",
        &[
            (10, &w.w),
            (11, &w.scales),
            (12, &q.xq),
            (13, &q.xs),
            (14, &p),
            (15, y),
            (16, &q.sel),
            (17, &q.alpha_dummy),
        ],
        (grid.0, grid.1, m),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemm_pf_dense(
    b: &mut Builder,
    s: &Sources,
    pf: &Pf,
    label: &str,
    w: &DenseLinGpu,
    x: &wgpu::Buffer,
    x_stride_words: usize,
    q: Option<&QuantIn>,
    y: &wgpu::Buffer,
    y_stride_words: usize,
) -> Result<()> {
    if pf_coop_covers(pf, w) {
        return match w {
            DenseLinGpu::Fp8(f, _) => pf_coop_gemm(
                b,
                pf,
                label,
                PfCoopW::Wq(
                    wk::gemm_coop_f16::WqFmt::Fp8RowscalePlain,
                    &f.w,
                    f.s_plain
                        .as_ref()
                        .expect("pf_coop_covers checked s_plain"),
                ),
                f.n,
                f.k,
                1.0,
                y,
                y_stride_words,
            ),
            DenseLinGpu::Nvfp4(w) => pf_coop_gemm(
                b,
                pf,
                label,
                PfCoopW::Wq(
                    wk::gemm_coop_f16::WqFmt::Nvfp4Block16,
                    &w.w,
                    w.scales_linear
                        .as_ref()
                        .expect("pf_coop_covers checked scales_linear"),
                ),
                w.n,
                w.k,
                w.alpha * nvfp4_input_global_folded_into_x_codes_by_q3w_qz_core_so_the_a16_route_multiplies_it(w),
                y,
                y_stride_words,
            ),
            DenseLinGpu::Bf16(_) | DenseLinGpu::Int8(_) => {
                anyhow::bail!("{label}: pf_coop_covers never selects bf16/int8 weights")
            }
        };
    }
    match w {
        DenseLinGpu::Bf16(w) => {
            push_gemm_pf(b, pf, label, w, x, x_stride_words, y, y_stride_words)
        }
        DenseLinGpu::Fp8(f, w) => {
            if pf_fp8_replay_enabled() {
                push_gemv_fp8_rowscale_pf(b, s, pf, label, f, x, x_stride_words, y, y_stride_words)
            } else {
                let w = w.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "{label}: the fp8 bf16 twin was not uploaded because the coop route was \
                         selected at upload time, yet the coop route does not cover this dispatch"
                    )
                })?;
                push_gemm_pf(b, pf, label, w, x, x_stride_words, y, y_stride_words)
            }
        }
        DenseLinGpu::Nvfp4(w) => {
            let q = q
                .ok_or_else(|| anyhow::anyhow!("{label}: nvfp4 M-row gemm without quantized x"))?;
            push_gemm_pf_nvfp4(b, s, pf, label, w, q, y, y_stride_words)
        }
        DenseLinGpu::Int8(w) => push_gemm_pf_i8(b, pf, label, w, x, x_stride_words, y, y_stride_words),
    }
}

#[allow(clippy::too_many_arguments)]
fn push_gemm_pf_i8(
    b: &mut Builder,
    pf: &Pf,
    label: &str,
    w: &I8Gpu,
    x: &wgpu::Buffer,
    x_stride_words: usize,
    y: &wgpu::Buffer,
    y_stride_words: usize,
) -> Result<()> {
    anyhow::ensure!(
        pf.m <= 16,
        "{label}: q3d_gemv_i8*_m<m> keeps one accumulator register per row and is generated \
         only for m 2..=16; m={} needs the NV_Q3D_PF_COOP coop route, which does not carry \
         int8 weights",
        pf.m
    );
    anyhow::ensure!(
        wk::gemv_nvfp4_v2::subgroup32_ok(b.ctx),
        "{label}: the i8 M-row GEMV reduces with subgroupShuffleXor over exactly 32 lanes, the \
         same predicate that gates the per-token int8 entries; upload_lin only produces Int8 \
         weights where subgroup32_ok holds, so portable adapters take the nvfp4 arm and must \
         never reach this dispatch"
    );
    anyhow::ensure!(
        w.k.is_multiple_of(4) && x_stride_words * 2 == w.k,
        "{label}: i8 M-row gemm needs k % 4 == 0 and an x row stride of k/2 packed-bf16 words; \
         got k={} x_stride_words={x_stride_words}",
        w.k
    );
    anyhow::ensure!(
        y_stride_words == w.n.div_ceil(2),
        "{label}: the i8 M-row GEMV packs two bf16 rows per output word, so the caller's y row \
         stride {y_stride_words} must equal ceil(n/2) = {}",
        w.n.div_ceil(2)
    );
    let groups = w.n.div_ceil(8) as u32;
    let p = b.uni(
        label,
        GemmI8MkParams {
            n_rows: w.n as u32,
            k_elems: w.k as u32,
            groups_x: groups,
            groups_per_row: w.k.checked_div(w.group).unwrap_or(1) as u32,
            group_shift: if w.group > 0 {
                (w.group / 4).trailing_zeros()
            } else {
                0
            },
            x_stride_words: x_stride_words as u32,
            y_stride_words: y_stride_words as u32,
            pad0: 0,
        },
    );
    b.push_pf(
        label,
        &pf.gemm_i8_src,
        &gemm_i8_mk_entry(pf.m, w.group > 0),
        &[(0, &w.w), (1, &w.s), (2, x), (3, y), (4, &p)],
        (groups, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemm_pf(
    b: &mut Builder,
    pf: &Pf,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    x_stride_words: usize,
    y: &wgpu::Buffer,
    y_stride_words: usize,
) -> Result<()> {
    use nv_kernels::wgpu_backend::na_bf16;
    anyhow::ensure!(
        pf.m <= 16,
        "{label}: q3w_gemm_bf16_m<m> keeps one accumulator register per row and is generated \
         only for m 2..=16; m={} needs the NV_Q3D_PF_COOP coop route, which does not carry \
         this bf16 weight",
        pf.m
    );
    if let Some(na) = &pf.na {
        if na_bf16::shape_ok(w.n, w.k, pf.m) {
            let np = b.uni(
                "q3d-pf-na-p",
                na_bf16::NaBf16Params {
                    n_rows: w.n as u32,
                    k_elems: w.k as u32,
                    x_stride_words: x_stride_words as u32,
                    y_stride_words: y_stride_words as u32,
                    dst_word_off: 0,
                    ..Default::default()
                },
            );
            b.push_pf_pipeline(
                "q3d-pf-na:na_gemm_bf16",
                na.clone(),
                &[(0, &w.w), (1, x), (2, y), (3, &np), (4, &pf.ck)],
                (na_bf16::grid_x(w.n as u32), 1, 1),
            );
            return Ok(());
        }
    }
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "q3d-pf-gemm-p",
        GemmMkParams {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            w_row_words: (w.k / 2) as u32,
            x_stride_words: x_stride_words as u32,
            y_stride_words: y_stride_words as u32,
            ..Default::default()
        },
    );
    b.push_pf(
        label,
        &pf.gemm_src,
        &pf.gemm_entry,
        &[(0, &w.w), (1, x), (2, &p), (3, y)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_delta(
    b: &mut Builder,
    s: &Sources,
    cfg: &Qwen3_5DenseConfig,
    d: &HostDeltaNet,
    dfp8: &DeltaFp8,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    states: &mut Vec<(wgpu::Buffer, u64)>,
    pf: &mut Option<Pf>,
) -> Result<()> {
    let n_k = cfg.linear_num_key_heads;
    let n_v = cfg.linear_num_value_heads;
    let d_k = cfg.linear_key_head_dim;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;

    let hidden = cfg.hidden_size;
    let coop_on = pf.as_ref().is_some_and(|p| p.coop.is_some());
    let dn_coop = |g: &Option<Fp8Gpu>| {
        coop_on
            && !pf_fp8_replay_enabled()
            && g.as_ref().is_some_and(|f| {
                f.s_plain.is_some() && f.n.is_multiple_of(16) && f.k.is_multiple_of(16)
            })
    };

    let g_qkv = match (&dfp8.qkv, fp8_proj_enabled()) {
        (Some(f), true) => Some(upload_fp8(b, "q3d-dn-qkv-fp8", f)),
        _ => None,
    };
    let g_z = match (&dfp8.z, fp8_proj_enabled()) {
        (Some(f), true) => Some(upload_fp8(b, "q3d-dn-z-fp8", f)),
        _ => None,
    };
    let g_out = match (&dfp8.out, fp8_proj_enabled()) {
        (Some(f), true) => Some(upload_fp8(b, "q3d-dn-oproj-fp8", f)),
        _ => None,
    };
    let qkv_coop = dn_coop(&g_qkv);
    let z_coop = dn_coop(&g_z);
    let out_coop = dn_coop(&g_out);
    let w_qkv = if qkv_coop {
        None
    } else {
        Some(upload_bf16(b, "q3d-dn-qkv", &d.in_proj_qkv))
    };
    let w_z = if z_coop {
        None
    } else {
        Some(upload_bf16(b, "q3d-dn-z", &d.in_proj_z))
    };
    let w_out = if out_coop {
        None
    } else {
        Some(upload_bf16(b, "q3d-dn-out", &d.out_proj))
    };
    let w_ab = upload_bf16(b, "q3d-dn-ab", &d.in_proj_ab);
    let w_ab_f16 = if coop_on && (2 * n_v).is_multiple_of(16) && hidden.is_multiple_of(16) {
        Some(b.upload_u32(
            "q3d-dn-ab-f16",
            &pack_f16_pairs_from_bf16(&d.in_proj_ab.w),
        ))
    } else {
        None
    };

    let qkv = b.zeros("q3d-dn-qkvbuf", (conv_dim * 2) as u64);
    let z = b.zeros("q3d-dn-zbuf", (value_dim * 2) as u64);
    let ab = b.zeros("q3d-dn-abbuf", (2 * n_v * 2).max(4) as u64);
    let merged_gemv = fuse_dn_gemv_enabled()
        && g_qkv.is_some()
        && g_z.is_some()
        && b.ctx.caps.max_storage_buffers_per_shader_stage
            >= DN_MERGED_GEMV_BINDS_NINE_STORAGE_BUFFERS;
    if merged_gemv {
        push_gemv_dn_merged(
            b,
            s,
            "q3d-dn-qkv-z-ab",
            g_qkv.as_ref().expect("merged route checked g_qkv"),
            g_z.as_ref().expect("merged route checked g_z"),
            &w_ab,
            x,
            &qkv,
            &z,
            &ab,
        )?;
    } else {
        match (&g_qkv, &w_qkv) {
            (Some(g), _) => push_gemv_fp8_rowscale(b, s, "q3d-dn-qkv", g, x, &qkv, false)?,
            (None, Some(w)) => push_gemv_bf16(b, s, "q3d-dn-qkv", w, x, &qkv, false)?,
            (None, None) => anyhow::bail!("q3d-dn-qkv: neither fp8 nor bf16 weights uploaded"),
        }
        match (&g_z, &w_z) {
            (Some(g), _) => push_gemv_fp8_rowscale(b, s, "q3d-dn-z", g, x, &z, false)?,
            (None, Some(w)) => push_gemv_bf16(b, s, "q3d-dn-z", w, x, &z, false)?,
            (None, None) => anyhow::bail!("q3d-dn-z: neither fp8 nor bf16 weights uploaded"),
        }
        push_gemv_bf16(b, s, "q3d-dn-ab", &w_ab, x, &ab, false)?;
    }

    let conv_w = b.upload_f32("q3d-dn-convw", &d.conv1d);
    let conv_state_bytes = (conv_dim * (ks - 1) * 4) as u64;
    let conv_state = b.zeros("q3d-dn-convstate", conv_state_bytes.max(4));
    states.push((conv_state.clone(), conv_state_bytes.max(4)));
    let mixed = b.zeros("q3d-dn-mixed", (conv_dim * 4) as u64);
    let cp = b.uni(
        "q3d-dn-conv-p",
        ConvParams {
            conv_dim: conv_dim as u32,
            kernel: ks as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1(conv_dim as u64, 64);
    b.push(
        "q3d-dn-conv",
        &s.delta,
        "q3w_delta_conv",
        &[
            (0, &qkv),
            (1, &conv_w),
            (2, &conv_state),
            (3, &mixed),
            (4, &cp),
        ],
        grid,
    )?;

    let dqp = b.uni(
        "q3d-dn-qkv-p",
        DeltaQkvParams {
            n_v: n_v as u32,
            d_k: d_k as u32,
            d_v: d_v as u32,
            key_dim: key_dim as u32,
            v_per_k: (n_v / n_k) as u32,
            scale: 1.0 / (d_k as f32).sqrt(),
            ..Default::default()
        },
    );
    let alog = b.upload_f32("q3d-dn-alog", &d.a_log);
    let dtb = b.upload_f32("q3d-dn-dt", &d.dt_bias);
    let gp = b.uni(
        "q3d-dn-gate-p",
        GatingParams {
            n_v: n_v as u32,
            ..Default::default()
        },
    );
    let state_bytes = (n_v * d_k * d_v * 4) as u64;
    let state = b.zeros("q3d-dn-state", state_bytes);
    states.push((state.clone(), state_bytes));
    let norm_w = b.upload_u32("q3d-dn-normw", &pack_pairs(&d.norm_w));
    let gated = b.zeros("q3d-dn-gated", (value_dim * 2) as u64);
    let dop = b.uni(
        "q3d-dn-out-p",
        DeltaOutParams {
            n_v: n_v as u32,
            d_v: d_v as u32,
            pad0: 0,
            eps: cfg.rms_norm_eps as f32,
        },
    );
    let wg = DELTA_HEAD_FUSED_WG128_COVERS_ONE_LANE_PER_DV_AND_ONE_REDUCTION_SLOT_PER_DK;
    let fused_head = fuse_dn_enabled() && d_k <= wg && d_v <= wg && d_v.is_multiple_of(2);
    if fused_head {
        b.push(
            "q3d-dn-head-fused",
            &s.delta,
            "q3w_delta_head_fused",
            &[
                (10, &mixed),
                (14, &dqp),
                (20, &ab),
                (21, &alog),
                (22, &dtb),
                (25, &gp),
                (36, &state),
                (41, &norm_w),
                (42, &z),
                (43, &gated),
                (44, &dop),
            ],
            (n_v as u32, 1, 1),
        )?;
    } else {
        let qg = b.zeros("q3d-dn-q", (n_v * d_k * 4) as u64);
        let kg = b.zeros("q3d-dn-k", (n_v * d_k * 4) as u64);
        let vg = b.zeros("q3d-dn-v", (n_v * d_v * 4) as u64);
        let gexp = b.zeros("q3d-dn-g", (n_v * 4) as u64);
        let beta = b.zeros("q3d-dn-beta", (n_v * 4) as u64);
        if qkv_gated_enabled() {
            b.push(
                "q3d-dn-split-gated",
                &s.delta,
                "q3w_delta_qkv_gated",
                &[
                    (10, &mixed),
                    (11, &qg),
                    (12, &kg),
                    (13, &vg),
                    (14, &dqp),
                    (20, &ab),
                    (21, &alog),
                    (22, &dtb),
                    (23, &gexp),
                    (24, &beta),
                    (25, &gp),
                ],
                (n_v as u32, 1, 1),
            )?;
        } else {
            b.push(
                "q3d-dn-split",
                &s.delta,
                "q3w_delta_qkv",
                &[(10, &mixed), (11, &qg), (12, &kg), (13, &vg), (14, &dqp)],
                (n_v as u32, 1, 1),
            )?;
            let grid = b.grid1(n_v as u64, 64);
            b.push(
                "q3d-dn-gating",
                &s.delta,
                "q3w_delta_gating",
                &[
                    (20, &ab),
                    (21, &alog),
                    (22, &dtb),
                    (23, &gexp),
                    (24, &beta),
                    (25, &gp),
                ],
                grid,
            )?;
        }

        let core = b.zeros("q3d-dn-core", (n_v * d_v * 4) as u64);
        let rp = b.uni(
            "q3d-dn-rec-p",
            RecurrentParams {
                heads: n_v as u32,
                d_k: d_k as u32,
                d_v: d_v as u32,
                pad0: 0,
            },
        );
        let (rec_entry, rec_lanes) = delta_recurrent_kernel();
        anyhow::ensure!(
            rec_lanes > 0 || d_v <= 128,
            "{rec_entry} covers one lane per invocation of a 128-wide workgroup, got d_v {d_v}"
        );
        let rec_grid_y = if rec_lanes > 0 {
            (d_v as u32).div_ceil(rec_lanes)
        } else {
            1
        };
        b.push(
            "q3d-dn-recurrent",
            &s.delta,
            rec_entry,
            &[
                (30, &qg),
                (31, &kg),
                (32, &vg),
                (33, &gexp),
                (34, &beta),
                (35, &core),
                (36, &state),
                (37, &rp),
            ],
            (n_v as u32, rec_grid_y, 1),
        )?;

        b.push(
            "q3d-dn-out",
            &s.delta,
            "q3w_delta_out",
            &[
                (40, &core),
                (41, &norm_w),
                (42, &z),
                (43, &gated),
                (44, &dop),
            ],
            (n_v as u32, 1, 1),
        )?;
    }

    match (&g_out, &w_out) {
        (Some(g), _) => push_gemv_fp8_rowscale(b, s, "q3d-dn-oproj", g, &gated, out, false)?,
        (None, Some(w)) => push_gemv_bf16(b, s, "q3d-dn-oproj", w, &gated, out, false)?,
        (None, None) => anyhow::bail!("q3d-dn-oproj: neither fp8 nor bf16 weights uploaded"),
    }

    if let Some(pf) = pf.as_ref() {
        let hidden_words = cfg.hidden_size / 2;
        let conv_words = conv_dim / 2;
        let value_words = value_dim / 2;
        let m = pf.m as u32;
        if qkv_coop || z_coop || w_ab_f16.is_some() {
            pf_coop_cast_x(b, pf, "q3d-pf-dn-castx", &pf.normed, hidden_words, hidden)?;
        }
        match (&g_qkv, pf_fp8_replay_enabled()) {
            (Some(g), true) => push_gemv_fp8_rowscale_pf(
                b,
                s,
                pf,
                "q3d-pf-dn-qkv-fp8r",
                g,
                &pf.normed,
                hidden_words,
                &pf.d_qkv,
                conv_words,
            )?,
            (Some(g), false) if qkv_coop => pf_coop_gemm(
                b,
                pf,
                "q3d-pf-dn-qkv-coop",
                PfCoopW::Wq(
                    wk::gemm_coop_f16::WqFmt::Fp8RowscalePlain,
                    &g.w,
                    g.s_plain.as_ref().expect("qkv_coop checked s_plain"),
                ),
                conv_dim,
                hidden,
                1.0,
                &pf.d_qkv,
                conv_words,
            )?,
            _ => push_gemm_pf(
                b,
                pf,
                "q3d-pf-dn-qkv",
                w_qkv.as_ref().expect("no coop route, twin uploaded"),
                &pf.normed,
                hidden_words,
                &pf.d_qkv,
                conv_words,
            )?,
        }
        match (&g_z, pf_fp8_replay_enabled()) {
            (Some(g), true) => push_gemv_fp8_rowscale_pf(
                b,
                s,
                pf,
                "q3d-pf-dn-z-fp8r",
                g,
                &pf.normed,
                hidden_words,
                &pf.d_z,
                value_words,
            )?,
            (Some(g), false) if z_coop => pf_coop_gemm(
                b,
                pf,
                "q3d-pf-dn-z-coop",
                PfCoopW::Wq(
                    wk::gemm_coop_f16::WqFmt::Fp8RowscalePlain,
                    &g.w,
                    g.s_plain.as_ref().expect("z_coop checked s_plain"),
                ),
                value_dim,
                hidden,
                1.0,
                &pf.d_z,
                value_words,
            )?,
            _ => push_gemm_pf(
                b,
                pf,
                "q3d-pf-dn-z",
                w_z.as_ref().expect("no coop route, twin uploaded"),
                &pf.normed,
                hidden_words,
                &pf.d_z,
                value_words,
            )?,
        }
        match &w_ab_f16 {
            Some(wf) => pf_coop_gemm(
                b,
                pf,
                "q3d-pf-dn-ab-coop",
                PfCoopW::F16(wf),
                2 * n_v,
                hidden,
                1.0,
                &pf.d_ab,
                n_v,
            )?,
            None => push_gemm_pf(
                b,
                pf,
                "q3d-pf-dn-ab",
                &w_ab,
                &pf.normed,
                hidden_words,
                &pf.d_ab,
                n_v,
            )?,
        }
        let cpm = b.uni(
            "q3d-pf-dn-conv-p",
            ConvMParams {
                conv_dim: conv_dim as u32,
                kernel: ks as u32,
                x_words: conv_words as u32,
                mixed_stride: conv_dim as u32,
            },
        );
        let cgrid = b.grid1(conv_dim as u64, 64);
        b.push_pf(
            "q3d-pf-dn-conv",
            &s.prefill,
            "q3w_delta_conv_m",
            &[
                (10, &pf.d_qkv),
                (11, &conv_w),
                (12, &conv_state),
                (13, &pf.d_mixed),
                (14, &cpm),
            ],
            (cgrid.0, m, 1),
        )?;
        b.push_pf(
            "q3d-pf-dn-shift",
            &s.prefill,
            "q3w_delta_conv_shift",
            &[(10, &pf.d_qkv), (12, &conv_state), (14, &cpm), (15, &pf.ck)],
            cgrid,
        )?;
        let dqm = b.uni(
            "q3d-pf-dn-qkv-p",
            DeltaQkvMParams {
                n_v: n_v as u32,
                d_k: d_k as u32,
                d_v: d_v as u32,
                key_dim: key_dim as u32,
                v_per_k: (n_v / n_k) as u32,
                mixed_stride: conv_dim as u32,
                scale: 1.0 / (d_k as f32).sqrt(),
                ..Default::default()
            },
        );
        b.push_pf(
            "q3d-pf-dn-split",
            &s.prefill,
            "q3w_delta_qkv_m",
            &[
                (20, &pf.d_mixed),
                (21, &pf.d_q),
                (22, &pf.d_k),
                (23, &pf.d_v),
                (24, &dqm),
            ],
            (n_v as u32, m, 1),
        )?;
        let gpm = b.uni(
            "q3d-pf-dn-gate-p",
            GatingParams {
                n_v: n_v as u32,
                ..Default::default()
            },
        );
        let ggrid = b.grid1(n_v as u64, 64);
        b.push_pf(
            "q3d-pf-dn-gating",
            &s.prefill,
            "q3w_delta_gating_m",
            &[
                (30, &pf.d_ab),
                (31, &alog),
                (32, &dtb),
                (33, &pf.d_g),
                (34, &pf.d_beta),
                (35, &gpm),
            ],
            (ggrid.0, m, 1),
        )?;
        let rpm = b.uni(
            "q3d-pf-dn-rec-p",
            RecurrentParams {
                heads: n_v as u32,
                d_k: d_k as u32,
                d_v: d_v as u32,
                pad0: 0,
            },
        );
        let (scan_entry, scan_grid_y) = if pf_small_m_legacy(pf.m)
            && !env_explicit_1(PF_SCAN_WY_ENV_DEFAULT_ON_SET_0_FOR_THE_TOKEN_SERIAL_SCAN)
        {
            ("q3w_delta_scan", 1)
        } else {
            pf_scan_wy_route(d_k, d_v)
        };
        b.push_pf(
            "q3d-pf-dn-scan",
            &s.prefill,
            scan_entry,
            &[
                (40, &pf.d_q),
                (41, &pf.d_k),
                (42, &pf.d_v),
                (43, &pf.d_g),
                (44, &pf.d_beta),
                (45, &pf.d_core),
                (46, &state),
                (47, &rpm),
                (48, &pf.ck),
            ],
            (n_v as u32, scan_grid_y, 1),
        )?;
        let dom = b.uni(
            "q3d-pf-dn-out-p",
            DeltaOutParams {
                n_v: n_v as u32,
                d_v: d_v as u32,
                pad0: 0,
                eps: cfg.rms_norm_eps as f32,
            },
        );
        b.push_pf(
            "q3d-pf-dn-out",
            &s.prefill,
            "q3w_delta_out_m",
            &[
                (50, &pf.d_core),
                (51, &norm_w),
                (52, &pf.d_z),
                (53, &pf.d_gated),
                (54, &dom),
            ],
            (n_v as u32, m, 1),
        )?;
        match (&g_out, pf_fp8_replay_enabled()) {
            (Some(g), true) => push_gemv_fp8_rowscale_pf(
                b,
                s,
                pf,
                "q3d-pf-dn-oproj-fp8r",
                g,
                &pf.d_gated,
                value_words,
                &pf.mix,
                hidden_words,
            )?,
            (Some(g), false) if out_coop => {
                pf_coop_cast_x(
                    b,
                    pf,
                    "q3d-pf-dn-ocastx",
                    &pf.d_gated,
                    value_words,
                    value_dim,
                )?;
                pf_coop_gemm(
                    b,
                    pf,
                    "q3d-pf-dn-oproj-coop",
                    PfCoopW::Wq(
                        wk::gemm_coop_f16::WqFmt::Fp8RowscalePlain,
                        &g.w,
                        g.s_plain.as_ref().expect("out_coop checked s_plain"),
                    ),
                    cfg.hidden_size,
                    value_dim,
                    1.0,
                    &pf.mix,
                    hidden_words,
                )?;
            }
            _ => push_gemm_pf(
                b,
                pf,
                "q3d-pf-dn-oproj",
                w_out.as_ref().expect("no coop route, twin uploaded"),
                &pf.d_gated,
                value_words,
                &pf.mix,
                hidden_words,
            )?,
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_attn(
    b: &mut Builder,
    s: &Sources,
    cfg: &Qwen3_5DenseConfig,
    a: &HostDenseAttention,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    pos_buf: &wgpu::Buffer,
    fd_buf: &wgpu::Buffer,
    max_seq: usize,
    states: &mut Vec<(wgpu::Buffer, u64)>,
    pf: &mut Option<Pf>,
    rope_rows: &mut Vec<(wgpu::Buffer, wgpu::Buffer)>,
) -> Result<usize> {
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let rot = cfg.rotary_dim();
    anyhow::ensure!(
        rot.is_multiple_of(2) && rot <= hd,
        "rotary_dim {rot} invalid"
    );

    let pf_rides_coop = pf.as_ref().map_or(true, |p| p.coop.is_some());
    let wq = upload_lin(b, "q3d-at-wq", &a.q, W8Scope::Attn, pf_rides_coop);
    let wk_ = upload_lin(b, "q3d-at-wk", &a.k, W8Scope::Attn, pf_rides_coop);
    let wv = upload_lin(b, "q3d-at-wv", &a.v, W8Scope::Attn, pf_rides_coop);
    let wo = upload_lin(b, "q3d-at-wo", &a.o, W8Scope::Attn, pf_rides_coop);

    let qkv_q = quant_for(b, s, "q3d-at-quant", x, wq.k(), &[&wq, &wk_, &wv])?;
    let q_raw = b.zeros("q3d-at-qraw", (wq.n() * 2) as u64);
    let k_raw = b.zeros("q3d-at-kraw", (wk_.n() * 2) as u64);
    let v_raw = b.zeros("q3d-at-vraw", (wv.n() * 2) as u64);
    push_gemv(b, s, "q3d-at-qproj", &wq, x, qkv_q.as_ref(), &q_raw)?;
    push_gemv(b, s, "q3d-at-kproj", &wk_, x, qkv_q.as_ref(), &k_raw)?;
    push_gemv(b, s, "q3d-at-vproj", &wv, x, qkv_q.as_ref(), &v_raw)?;

    let (cos, sin) = rope_tables(rot.max(2), cfg.rope_theta, max_seq);
    let cosb = b.upload_f32("q3d-at-cos", &cos);
    let sinb = b.upload_f32("q3d-at-sin", &sin);
    rope_rows.push((cosb.clone(), sinb.clone()));
    let qn = b.upload_u32("q3d-at-qn", &pack_pairs(&a.q_norm));
    let kn = b.upload_u32("q3d-at-kn", &pack_pairs(&a.k_norm));
    let q = b.zeros("q3d-at-q", (n_h * hd * 2) as u64);
    let k = b.zeros("q3d-at-k", (n_kv * hd * 2) as u64);
    let src_stride_q = if cfg.attn_output_gate { 2 * hd } else { hd };
    let q_f32 = b.zeros("q3d-at-qf32", (n_h * hd * 4) as u64);
    let fused_norm = fuse_attn_enabled()
        && hd <= MAX_HEAD_DIM
        && hd.is_multiple_of(2)
        && b.ctx.caps.max_storage_buffers_per_shader_stage
            >= ATTN_FUSED_NORM_BINDS_TEN_STORAGE_BUFFERS;
    if fused_norm {
        let afp = b.uni(
            "q3d-at-qk-fused-p",
            NormRopeFusedParams {
                n_q_rows: n_h as u32,
                n_k_rows: n_kv as u32,
                head_dim: hd as u32,
                q_src_stride: src_stride_q as u32,
                k_src_stride: hd as u32,
                rot_half: (rot / 2) as u32,
                pad0: 0,
                eps: cfg.rms_norm_eps as f32,
            },
        );
        b.push(
            "q3d-at-qk-norm-fused",
            &s.attn,
            "q3w_attn_qk_norm_rope_qcast",
            &[
                (0, &q_raw),
                (1, &qn),
                (2, &cosb),
                (3, &sinb),
                (4, pos_buf),
                (5, &q),
                (40, &k_raw),
                (41, &kn),
                (42, &k),
                (43, &q_f32),
                (44, &afp),
            ],
            ((n_h + n_kv) as u32, 1, 1),
        )?;
    } else {
        let qp = b.uni(
            "q3d-at-qnorm-p",
            NormRopeParams {
                n_rows: n_h as u32,
                head_dim: hd as u32,
                src_stride: src_stride_q as u32,
                rot_half: (rot / 2) as u32,
                eps: cfg.rms_norm_eps as f32,
                ..Default::default()
            },
        );
        b.push(
            "q3d-at-qnorm",
            &s.attn,
            "q3w_attn_norm_rope",
            &[
                (0, &q_raw),
                (1, &qn),
                (2, &cosb),
                (3, &sinb),
                (4, pos_buf),
                (5, &q),
                (6, &qp),
            ],
            (n_h as u32, 1, 1),
        )?;
        let kp = b.uni(
            "q3d-at-knorm-p",
            NormRopeParams {
                n_rows: n_kv as u32,
                head_dim: hd as u32,
                src_stride: hd as u32,
                rot_half: (rot / 2) as u32,
                eps: cfg.rms_norm_eps as f32,
                ..Default::default()
            },
        );
        b.push(
            "q3d-at-knorm",
            &s.attn,
            "q3w_attn_norm_rope",
            &[
                (0, &k_raw),
                (1, &kn),
                (2, &cosb),
                (3, &sinb),
                (4, pos_buf),
                (5, &k),
                (6, &kp),
            ],
            (n_kv as u32, 1, 1),
        )?;
    }

    let kv_words = n_kv * hd / 2;
    let cache_bytes = (max_seq * kv_words * 4) as u64;
    let kc = b.zeros("q3d-at-kc", cache_bytes);
    let vc = b.zeros("q3d-at-vc", cache_bytes);
    states.push((kc.clone(), cache_bytes));
    states.push((vc.clone(), cache_bytes));
    let kvp = b.uni(
        "q3d-at-kv-p",
        KvWriteParams {
            words: kv_words as u32,
            ..Default::default()
        },
    );
    let kv8_decode = kv_fp8_enabled();
    let tiled_pf = pf.as_ref().is_some_and(|p| p.attn_tiled.is_some());
    let fused_kvw = fuse_kvw_enabled()
        && (kv8_decode || tiled_pf)
        && b.ctx.caps.max_storage_buffers_per_shader_stage
            >= KVW_FUSED_BINDS_NINE_STORAGE_BUFFERS;
    if !fused_kvw {
        let grid = b.grid1(kv_words as u64, 64);
        b.push(
            "q3d-at-kvwrite",
            &s.attn,
            "q3w_kv_write",
            &[
                (10, &k),
                (11, &v_raw),
                (12, &kc),
                (13, &vc),
                (14, pos_buf),
                (15, &kvp),
            ],
            grid,
        )?;
    }
    let kv8 = if kv8_decode || tiled_pf {
        let cache8_bytes = (max_seq * n_kv * hd) as u64;
        let scale_bytes = (max_seq * n_kv * 4) as u64;
        let kc8 = b.zeros("q3d-at-kc8", cache8_bytes);
        let vc8 = b.zeros("q3d-at-vc8", cache8_bytes);
        let ksc = b.zeros("q3d-at-ks8", scale_bytes);
        let vsc = b.zeros("q3d-at-vs8", scale_bytes);
        states.push((kc8.clone(), cache8_bytes));
        states.push((vc8.clone(), cache8_bytes));
        states.push((ksc.clone(), scale_bytes));
        states.push((vsc.clone(), scale_bytes));
        let kvq_p = b.uni(
            "q3d-at-kvq-p",
            KvFp8Params {
                n_tokens: 1,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                pairs: n_kv as u32,
                slots: max_seq as u32,
                ..Default::default()
            },
        );
        if fused_kvw {
            b.push(
                "q3d-at-kvw-fused",
                &s.kvq,
                wk::kv_fp8::QUANTIZE_PAIR_KV_WRITE_ENTRY_GRID_Y_IS_2_K_THEN_V_AND_ALSO_COPIES_THE_BF16_ROWS,
                &[
                    (0, &k),
                    (1, &kc8),
                    (2, &ksc),
                    (3, pos_buf),
                    (4, &kvq_p),
                    (9, &v_raw),
                    (10, &vc8),
                    (11, &vsc),
                    (12, &kc),
                    (13, &vc),
                ],
                (n_kv as u32, 2, 1),
            )?;
        } else {
            b.push(
                "q3d-at-kvq-k",
                &s.kvq,
                wk::kv_fp8::QUANTIZE_ENTRY,
                &[(0, &k), (1, &kc8), (2, &ksc), (3, pos_buf), (4, &kvq_p)],
                (n_kv as u32, 1, 1),
            )?;
            b.push(
                "q3d-at-kvq-v",
                &s.kvq,
                wk::kv_fp8::QUANTIZE_ENTRY,
                &[(0, &v_raw), (1, &vc8), (2, &vsc), (3, pos_buf), (4, &kvq_p)],
                (n_kv as u32, 1, 1),
            )?;
        }
        Some((kc8, vc8, ksc, vsc, kvq_p))
    } else {
        None
    };
    let kv8_dec = if kv8_decode { kv8.as_ref() } else { None };

    let kv4_arm = kv_nvfp4_arm();
    anyhow::ensure!(
        kv4_arm.is_none() || kv8_decode,
        "{} rides the fp8 cache for its first {} sink slots and for chunk coherence, so it \
         cannot pair with NV_Q3D_KV_FP8=0; unset one",
        KV_NVFP4_ENV_OFF_BY_DEFAULT_FP8_STAYS_THE_SHIPPED_KV_DECODE_FORMAT,
        wk::kv_nvfp4::KV_NVFP4_SINK_SLOTS_STAY_FP8_TO_ANCHOR_SOFTMAX
    );
    let kv4 = if let Some(arm) = kv4_arm {
        anyhow::ensure!(
            hd.is_multiple_of(8),
            "kv_nvfp4 packs 8 e2m1 nibbles per u32 word; head_dim {hd} % 8 != 0"
        );
        anyhow::ensure!(
            b.ctx.caps.max_storage_buffers_per_shader_stage
                >= wk::flash_decode::KV_NVFP4_STAGE1_BINDS_TEN_STORAGE_BUFFERS,
            "the k4v4 stage1 binds {} storage buffers (fp8 sink K/V + scales, nvfp4 K/V + \
             scales, q, scratch); this adapter caps at {}",
            wk::flash_decode::KV_NVFP4_STAGE1_BINDS_TEN_STORAGE_BUFFERS,
            b.ctx.caps.max_storage_buffers_per_shader_stage
        );
        let payload_bytes = (max_seq * n_kv * hd / 2) as u64;
        let v4w = b.zeros("q3d-at-v4w", payload_bytes);
        let v4s_bytes = (max_seq * n_kv * 4) as u64;
        let v4s = b.zeros("q3d-at-v4s", v4s_bytes);
        states.push((v4w.clone(), payload_bytes));
        states.push((v4s.clone(), v4s_bytes));
        let kv4_p = b.uni(
            "q3d-at-kv4-p",
            wk::kv_nvfp4::Kv4Params {
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                tokens: 1,
                slots: max_seq as u32,
            },
        );
        b.push(
            "q3d-at-kv4-v",
            &s.kvq4,
            wk::kv_nvfp4::QUANTIZE_V_ROWS_ENTRY,
            &[(0, &vc), (1, &v4w), (2, &v4s), (3, pos_buf), (4, &kv4_p)],
            (n_kv as u32, 1, 1),
        )?;
        let k4 = if arm == KvNvfp4Arm::K4V4 {
            let k4w = b.zeros("q3d-at-k4w", payload_bytes);
            let k4s_bytes = (wk::kv_nvfp4::k_scale_blocks(max_seq) * n_kv * hd * 4) as u64;
            let k4s = b.zeros("q3d-at-k4s", k4s_bytes);
            states.push((k4w.clone(), payload_bytes));
            states.push((k4s.clone(), k4s_bytes));
            b.push(
                "q3d-at-kv4-k",
                &s.kvq4,
                wk::kv_nvfp4::QUANTIZE_K_BLOCKS_ENTRY,
                &[(0, &kc), (1, &k4w), (2, &k4s), (3, pos_buf), (4, &kv4_p)],
                (n_kv as u32, 1, 1),
            )?;
            Some((k4w, k4s))
        } else {
            None
        };
        Some((k4, v4w, v4s))
    } else {
        None
    };
    let kv_written_pass_end = b.passes.len();

    let attn = b.zeros("q3d-at-out", (n_h * hd * 4) as u64);
    if !fused_norm {
        let cast_p = b.uni(
            "q3d-at-cast-p",
            ResScaleParams {
                n: (n_h * hd) as u32,
                n_words: (n_h * hd / 2) as u32,
                scale: 1.0,
                ..Default::default()
            },
        );
        let cast_grid = b.grid1((n_h * hd / 2) as u64, 256);
        b.push(
            "q3d-at-qcast",
            &s.resscale,
            "cast_bf16_to_f32",
            &[(0, &q), (3, &cast_p), (4, &q_f32)],
            cast_grid,
        )?;
    }
    let splits = wk::flash_decode::splits_for(max_seq) as u32;
    let scratch = b.zeros(
        "q3d-at-scratch",
        (n_h * splits as usize * (hd + 2) * 4) as u64,
    );
    let group = n_h / n_kv;
    let fold = if kv8_dec.is_some() {
        let f = if std::env::var(wk::flash_decode::GQA_FOLD_ENV).is_ok() {
            wk::flash_decode::gqa_fold_env(group)
        } else {
            group.min(wk::flash_decode::MAX_GQA_FOLD)
        };
        if f > 1 && group.is_multiple_of(f) && n_h.is_multiple_of(f) {
            f
        } else {
            1
        }
    } else {
        1
    };
    if let Some((k4, v4w, v4s)) = kv4.as_ref() {
        let (kc8, vc8, ksc, vsc, _) = kv8_dec.expect("the nvfp4 arms ride the fp8 sink cache");
        let sg = b.ctx.caps.subgroup && b.ctx.subgroup_width() == Some(32);
        let body =
            wk::flash_decode::fold_stage1_source_nvfp4(hd as u32, sg, fold as u32, k4.is_some());
        let entry =
            wk::flash_decode::fold_stage1_entry_nvfp4(hd as u32, sg, fold as u32, k4.is_some());
        let src = compose(&format!("{}\n{}", wk::flash_decode::WGSL, body));
        let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
            (0, &q_f32),
            (4, fd_buf),
            (5, kc8),
            (6, vc8),
            (7, &scratch),
            (8, ksc),
            (9, vsc),
            (15, v4w),
            (17, v4s),
        ];
        if let Some((k4w, k4s)) = k4.as_ref() {
            binds.push((14, k4w));
            binds.push((16, k4s));
        }
        b.push(
            "q3d-at-flash1",
            &src,
            &entry,
            &binds,
            ((n_h / fold) as u32, splits, 1),
        )?;
    } else if fold > 1 {
        let (kc8, vc8, ksc, vsc, _) = kv8_dec.expect("fold requires fp8 kv");
        let sg = b.ctx.caps.subgroup && b.ctx.subgroup_width() == Some(32);
        let tile = flash_tile_env();
        let (fold_body, fold_entry) = if tile >= 2 && sg {
            (
                wk::flash_decode::fold_stage1_source_sd_tiled(hd as u32, fold as u32, tile),
                wk::flash_decode::fold_stage1_entry_sd_tiled(hd as u32, fold as u32, tile),
            )
        } else {
            (
                wk::flash_decode::fold_stage1_source_sd(hd as u32, sg, fold as u32),
                wk::flash_decode::fold_stage1_entry_sd(hd as u32, sg, fold as u32),
            )
        };
        let fold_src = compose(&format!("{}\n{}", wk::flash_decode::WGSL, fold_body));
        b.push(
            "q3d-at-flash1",
            &fold_src,
            &fold_entry,
            &[
                (0, &q_f32),
                (4, fd_buf),
                (5, kc8),
                (6, vc8),
                (7, &scratch),
                (8, ksc),
                (9, vsc),
            ],
            ((n_h / fold) as u32, splits, 1),
        )?;
    } else {
        b.push(
            "q3d-at-flash1",
            &s.flash,
            match kv8_dec {
                Some(_) => wk::flash_decode::ENTRY_STAGE1_FP8_SD,
                None => wk::flash_decode::ENTRY_STAGE1_BF16,
            },
            &match kv8_dec {
                Some((kc8, vc8, ksc, vsc, _)) => vec![
                    (0, &q_f32),
                    (4, fd_buf),
                    (5, kc8),
                    (6, vc8),
                    (7, &scratch),
                    (8, ksc),
                    (9, vsc),
                ],
                None => vec![
                    (0, &q_f32),
                    (4, fd_buf),
                    (5, &kc),
                    (6, &vc),
                    (7, &scratch),
                ],
            },
            (n_h as u32, splits, 1),
        )?;
    }
    b.push(
        "q3d-at-flash2",
        &s.flash,
        wk::flash_decode::ENTRY_STAGE2,
        &[(3, &attn), (4, fd_buf), (7, &scratch)],
        (n_h as u32, 1, 1),
    )?;

    let gated = b.zeros("q3d-at-gated", (n_h * hd * 2) as u64);
    let agp = b.uni(
        "q3d-at-gate-p",
        AttnGateParams {
            n_words: (n_h * hd / 2) as u32,
            head_dim: hd as u32,
            src_stride: src_stride_q as u32,
            gate_off: hd as u32,
            has_gate: u32::from(cfg.attn_output_gate),
            ..Default::default()
        },
    );
    let grid = b.grid1((n_h * hd / 2) as u64, 64);
    b.push(
        "q3d-at-gate",
        &s.attn,
        "q3w_attn_gate",
        &[(30, &attn), (31, &q_raw), (32, &gated), (33, &agp)],
        grid,
    )?;

    let o_q = quant_for(b, s, "q3d-at-oquant", &gated, wo.k(), &[&wo])?;
    push_gemv(b, s, "q3d-at-oproj", &wo, &gated, o_q.as_ref(), out)?;

    if let Some(pf) = pf.as_ref() {
        let hidden_words = cfg.hidden_size / 2;
        let m = pf.m as u32;
        let q_row_elems = n_h * src_stride_q;
        let kv_row_elems = n_kv * hd;
        if [&wq, &wk_, &wv].iter().any(|w| pf_coop_covers(pf, w)) {
            pf_coop_cast_x(b, pf, "q3d-pf-at-castx", &pf.normed, hidden_words, wq.k())?;
        }
        let in_qm = if [&wq, &wk_, &wv]
            .iter()
            .any(|w| matches!(w, DenseLinGpu::Nvfp4(_)) && !pf_coop_covers(pf, w))
        {
            quant_for_m_rows(
                b,
                s,
                pf,
                "q3d-pf-at-quant",
                &pf.normed,
                wq.k(),
                &[&wq, &wk_, &wv],
            )?
        } else {
            None
        };
        push_gemm_pf_dense(
            b,
            s,
            pf,
            "q3d-pf-at-qproj",
            &wq,
            &pf.normed,
            hidden_words,
            in_qm.as_ref(),
            &pf.a_qraw,
            q_row_elems / 2,
        )?;
        push_gemm_pf_dense(
            b,
            s,
            pf,
            "q3d-pf-at-kproj",
            &wk_,
            &pf.normed,
            hidden_words,
            in_qm.as_ref(),
            &pf.a_kraw,
            kv_row_elems / 2,
        )?;
        push_gemm_pf_dense(
            b,
            s,
            pf,
            "q3d-pf-at-vproj",
            &wv,
            &pf.normed,
            hidden_words,
            in_qm.as_ref(),
            &pf.a_vraw,
            kv_row_elems / 2,
        )?;
        let qpm = b.uni(
            "q3d-pf-at-qnorm-p",
            NormRopeMParams {
                n_rows: n_h as u32,
                head_dim: hd as u32,
                src_stride: src_stride_q as u32,
                rot_half: (rot / 2) as u32,
                x_row_elems: q_row_elems as u32,
                y_row_elems: (n_h * hd) as u32,
                eps: cfg.rms_norm_eps as f32,
                ..Default::default()
            },
        );
        b.push_pf(
            "q3d-pf-at-qnorm",
            &s.prefill,
            "q3w_attn_norm_rope_m",
            &[
                (60, &pf.a_qraw),
                (61, &qn),
                (62, &cosb),
                (63, &sinb),
                (64, &pf.a_q),
                (65, &qpm),
                (66, &pf.ck),
            ],
            (n_h as u32, m, 1),
        )?;
        let kpm = b.uni(
            "q3d-pf-at-knorm-p",
            NormRopeMParams {
                n_rows: n_kv as u32,
                head_dim: hd as u32,
                src_stride: hd as u32,
                rot_half: (rot / 2) as u32,
                x_row_elems: kv_row_elems as u32,
                y_row_elems: kv_row_elems as u32,
                eps: cfg.rms_norm_eps as f32,
                ..Default::default()
            },
        );
        b.push_pf(
            "q3d-pf-at-knorm",
            &s.prefill,
            "q3w_attn_norm_rope_m",
            &[
                (60, &pf.a_kraw),
                (61, &kn),
                (62, &cosb),
                (63, &sinb),
                (64, &pf.a_k),
                (65, &kpm),
                (66, &pf.ck),
            ],
            (n_kv as u32, m, 1),
        )?;
        let kvm = b.uni(
            "q3d-pf-at-kv-p",
            KvWriteParams {
                words: kv_words as u32,
                ..Default::default()
            },
        );
        let kv_grid = b.grid1(kv_words as u64, 64);
        b.push_pf(
            "q3d-pf-at-kvwrite",
            &s.prefill,
            "q3w_kv_write_m",
            &[
                (70, &pf.a_k),
                (71, &pf.a_vraw),
                (72, &kc),
                (73, &vc),
                (74, &kvm),
                (75, &pf.ck),
            ],
            (kv_grid.0, m, 1),
        )?;
        let pf_kv_write_recorded = b.pf_passes.len();
        let chunk_kvq_start = pf
            .attn_tiled
            .as_ref()
            .map(|td| &td.start)
            .or_else(|| pf.attn_kvq_fd_start.as_ref().map(|(_, start)| start));
        anyhow::ensure!(
            !kv8_decode || chunk_kvq_start.is_some(),
            "NV_Q3D_KV_FP8 decode reads the fp8 cache for every row, so every chunk arm must \
             record the q3w_pf_quantize_kv_fp8_m pair for chunk rows; an arm that skips it \
             starves decode past the chunk boundary (measured before the pair was recorded on \
             the non-tiled arms: bit-green through 30 prefilled, silent 0.005 logit divergence \
             from 31)"
        );
        if let Some(start) = chunk_kvq_start {
            let (kc8, vc8, ksc, vsc, kvq_p) = kv8.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "a chunk-quantize start buffer exists (tiled arm or fp8-KV decode), so \
                     build_attn must have allocated the fp8 KV cache above"
                )
            })?;
            let pf_kvq_src = crate::qwen3_5_moe_wgpu::pf_kvq_source();
            let pkvm = b.uni(
                "q3d-pf-at-kvq-p",
                PfKvqMParams {
                    tokens: m,
                    x_stride_elems: kv_row_elems as u32,
                    ..Default::default()
                },
            );
            b.push_pf(
                "q3d-pf-at-kvq-k",
                &pf_kvq_src,
                "q3w_pf_quantize_kv_fp8_m",
                &[
                    (0, &pf.a_k),
                    (1, kc8),
                    (2, ksc),
                    (3, start),
                    (4, kvq_p),
                    (9, &pkvm),
                ],
                (n_kv as u32, m, 1),
            )?;
            b.push_pf(
                "q3d-pf-at-kvq-v",
                &pf_kvq_src,
                "q3w_pf_quantize_kv_fp8_m",
                &[
                    (0, &pf.a_vraw),
                    (1, vc8),
                    (2, vsc),
                    (3, start),
                    (4, kvq_p),
                    (9, &pkvm),
                ],
                (n_kv as u32, m, 1),
            )?;
            if let Some((k4, v4w, v4s)) = kv4.as_ref() {
                anyhow::ensure!(
                    pf_kv_write_recorded > 0,
                    "the nvfp4 chunk quantizers read the bf16 KV cache the chunk just wrote, so \
                     q3w_kv_write_m must stay recorded before them in the pf pass list"
                );
                let kv4_pm = b.uni(
                    "q3d-pf-at-kv4-p",
                    wk::kv_nvfp4::Kv4Params {
                        n_kv: n_kv as u32,
                        head_dim: hd as u32,
                        tokens: m,
                        slots: max_seq as u32,
                    },
                );
                b.push_pf(
                    "q3d-pf-at-kv4-v",
                    &s.kvq4,
                    wk::kv_nvfp4::QUANTIZE_V_ROWS_ENTRY,
                    &[(0, &vc), (1, v4w), (2, v4s), (3, start), (4, &kv4_pm)],
                    (n_kv as u32, m, 1),
                )?;
                if let Some((k4w, k4s)) = k4.as_ref() {
                    b.push_pf(
                        "q3d-pf-at-kv4-k",
                        &s.kvq4,
                        wk::kv_nvfp4::QUANTIZE_K_BLOCKS_ENTRY,
                        &[(0, &kc), (1, k4w), (2, k4s), (3, start), (4, &kv4_pm)],
                        (n_kv as u32, wk::kv_nvfp4::k_blocks_grid_y(m as usize), 1),
                    )?;
                }
            }
        }
        anyhow::ensure!(
            kv4.is_none() || chunk_kvq_start.is_some(),
            "{} decode reads the nvfp4 caches for every non-sink row, so every chunk arm must \
             record the nvfp4 chunk quantizers; an arm without a chunk-quantize start buffer \
             leaves chunk rows unquantized and starves decode past the chunk boundary",
            KV_NVFP4_ENV_OFF_BY_DEFAULT_FP8_STAYS_THE_SHIPPED_KV_DECODE_FORMAT
        );
        if let Some(td) = pf.attn_tiled.as_ref() {
            let (kc8, vc8, ksc, vsc, _) = kv8.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "the tiled pf attention streams the fp8 KV cache, so build_attn must have \
                     allocated it above whenever pf.attn_tiled is set"
                )
            })?;
            anyhow::ensure!(
                pf_kv_write_recorded > 0,
                "the M=1 decode path keeps reading the bf16 KV cache under the tiled pf arm, so \
                 q3w_kv_write_m must stay recorded in the pf pass list; dropping it would starve \
                 every decode step that follows a chunk"
            );
            let rows_t =
                crate::qwen3_5_moe_wgpu::PF_TILED_FLASH_ROWS_ONE_KV_STREAM_SERVES_32_ROWS_AT_4_PER_WARP;
            let subgroup = wk::gemv_nvfp4_v2::subgroup32_ok(b.ctx);
            let tiled_src =
                pf_attn_tiled_stage1_source_exact_decode_because_the_sd_twin_measured_a_1pct_null_here(
                    subgroup,
                );
            anyhow::ensure!(
                rows_t == 32 && tiled_src.contains("const FDT_ROWS: u32 = 32u"),
                "the tiled flash tile height is baked as 32 on both sides; \
                 PF_TILED_FLASH_ROWS_ONE_KV_STREAM_SERVES_32_ROWS_AT_4_PER_WARP and the kernel's \
                 FDT_ROWS must move together"
            );
            anyhow::ensure!(
                8 * hd <= PF_ATTN_TILED_STAGE_ARRAYS_HOLD_FDT_POS_8_TIMES_HEAD_DIM_F32,
                "head_dim {hd} overflows the tiled kernel's 2048-f32 workgroup K/V stages \
                 (FDT_POS 8 x head_dim)"
            );
            let qc_words = (pf.m * n_h * hd / 2) as u64;
            let qc_grid = qc_words.div_ceil(256);
            anyhow::ensure!(
                qc_grid <= 65535,
                "pf tiled qcast grid {qc_grid} exceeds the 1d dispatch limit"
            );
            let qcp = b.uni(
                "q3d-pf-at-td-qcast-p",
                PfAttnMkQcastParams {
                    n_words: qc_words as u32,
                    scale: 1.0,
                    ..Default::default()
                },
            );
            b.push_pf(
                "q3d-pf-at-td-qcast",
                PF_ATTN_MK_QCAST_WGSL,
                "q3w_pf_attn_mk_qcast",
                &[(0, &pf.a_q), (1, &td.q_f32), (2, &qcp)],
                (qc_grid as u32, 1, 1),
            )?;
            let splits = wk::flash_decode::splits_for(max_seq) as u32;
            let tiles = pf.m.div_ceil(rows_t) as u32;
            let entry = if subgroup {
                PF_ATTN_TILED_ENTRY_IS_THE_MOE_DEFAULT_SLOTML_ARM_SG_THEN_WG[0]
            } else {
                PF_ATTN_TILED_ENTRY_IS_THE_MOE_DEFAULT_SLOTML_ARM_SG_THEN_WG[1]
            };
            b.push_pf(
                "q3d-pf-at-td-flash1",
                &tiled_src,
                entry,
                &[
                    (0, &td.q_f32),
                    (4, &td.fd),
                    (5, kc8),
                    (6, vc8),
                    (7, &td.scratch),
                    (8, ksc),
                    (9, vsc),
                ],
                (n_h as u32, splits, tiles),
            )?;
            b.push_pf(
                "q3d-pf-at-td-flash2",
                &s.flash,
                wk::flash_decode::ENTRY_STAGE2_MK,
                &[(3, &pf.a_attn), (4, &td.fd), (7, &td.scratch)],
                (n_h as u32, m, 1),
            )?;
        } else if let Some(mk) = pf.attn_mk.as_ref() {
            anyhow::ensure!(
                pf_kv_write_recorded > 0,
                "the mk flash prefill attention reads chunk rows 0..base+m_live from the bf16 KV \
                 cache, so q3w_kv_write_m must be recorded in the same pf pass list before the mk \
                 dispatches; sequential dispatch order within one compute pass is what makes those \
                 rows visible"
            );
            let rows = PF_ATTN_MK_ROWS_PER_DISPATCH_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS;
            let group_bytes = (rows * n_h * hd * 4) as u64;
            anyhow::ensure!(
                group_bytes.is_multiple_of(256),
                "pf mk row-group bind offset stride {group_bytes} (8 rows x {n_h} heads x hd {hd} \
                 x 4B) must sit on the 256-byte storage-binding alignment"
            );
            anyhow::ensure!(
                pf.m.div_ceil(rows) == mk.fd.len(),
                "pf mk fd uniforms {} disagree with ceil(m {}/{rows}) row groups",
                mk.fd.len(),
                pf.m
            );
            let qc_words = (pf.m * n_h * hd / 2) as u64;
            let qc_grid = qc_words.div_ceil(256);
            anyhow::ensure!(
                qc_grid <= 65535,
                "pf mk qcast grid {qc_grid} exceeds the 1d dispatch limit"
            );
            let qcp = b.uni(
                "q3d-pf-at-mk-qcast-p",
                PfAttnMkQcastParams {
                    n_words: qc_words as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                    ..Default::default()
                },
            );
            b.push_pf(
                "q3d-pf-at-mk-qcast",
                PF_ATTN_MK_QCAST_WGSL,
                "q3w_pf_attn_mk_qcast",
                &[(0, &pf.a_q), (1, &mk.q_f32), (2, &qcp)],
                (qc_grid as u32, 1, 1),
            )?;
            let splits = wk::flash_decode::splits_for(max_seq) as u32;
            for (g, fd_g) in mk.fd.iter().enumerate() {
                let off = g as u64 * group_bytes;
                let mr_g = rows.min(pf.m - g * rows) as u32;
                b.push_pf_off(
                    "q3d-pf-at-mk-flash1",
                    &s.flash,
                    wk::flash_decode::ENTRY_STAGE1_BF16_MK,
                    &[
                        (0, &mk.q_f32, off),
                        (4, fd_g, 0),
                        (5, &kc, 0),
                        (6, &vc, 0),
                        (7, &mk.scratch, 0),
                    ],
                    (n_h as u32, splits, 1),
                )?;
                b.push_pf_off(
                    "q3d-pf-at-mk-flash2",
                    &s.flash,
                    wk::flash_decode::ENTRY_STAGE2_MK,
                    &[(3, &pf.a_attn, off), (4, fd_g, 0), (7, &mk.scratch, 0)],
                    (n_h as u32, mr_g, 1),
                )?;
            }
        } else {
            let adm = b.uni(
                "q3d-pf-at-dec-p",
                AttnDecodeParams {
                    n_heads: n_h as u32,
                    n_kv: n_kv as u32,
                    head_dim: hd as u32,
                    max_seq: max_seq as u32,
                    group: (n_h / n_kv) as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                    ..Default::default()
                },
            );
            b.push_pf(
                "q3d-pf-at-decode",
                &s.prefill,
                "q3w_attn_decode_m",
                &[
                    (80, &pf.a_q),
                    (81, &kc),
                    (82, &vc),
                    (83, &pf.a_scores),
                    (84, &pf.a_attn),
                    (85, &adm),
                    (86, &pf.ck),
                ],
                (n_h as u32, m, 1),
            )?;
        }
        let agm = b.uni(
            "q3d-pf-at-gate-p",
            AttnGateMParams {
                n_words: (n_h * hd / 2) as u32,
                head_dim: hd as u32,
                src_stride: src_stride_q as u32,
                gate_off: hd as u32,
                has_gate: u32::from(cfg.attn_output_gate),
                x_row_elems: q_row_elems as u32,
                ..Default::default()
            },
        );
        let g_grid = b.grid1((n_h * hd / 2) as u64, 64);
        b.push_pf(
            "q3d-pf-at-gate",
            &s.prefill,
            "q3w_attn_gate_m",
            &[
                (90, &pf.a_attn),
                (91, &pf.a_qraw),
                (92, &pf.a_gated),
                (93, &agm),
            ],
            (g_grid.0, m, 1),
        )?;
        if pf_coop_covers(pf, &wo) {
            pf_coop_cast_x(b, pf, "q3d-pf-at-ocastx", &pf.a_gated, n_h * hd / 2, wo.k())?;
        }
        let o_qm = if matches!(&wo, DenseLinGpu::Nvfp4(_)) && !pf_coop_covers(pf, &wo) {
            quant_for_m_rows(
                b,
                s,
                pf,
                "q3d-pf-at-oquant",
                &pf.a_gated,
                wo.k(),
                &[&wo],
            )?
        } else {
            None
        };
        push_gemm_pf_dense(
            b,
            s,
            pf,
            "q3d-pf-at-oproj",
            &wo,
            &pf.a_gated,
            n_h * hd / 2,
            o_qm.as_ref(),
            &pf.mix,
            hidden_words,
        )?;
    }
    Ok(kv_written_pass_end)
}

fn build_mlp(
    b: &mut Builder,
    s: &Sources,
    cfg: &Qwen3_5DenseConfig,
    m: &HostDenseMlp,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    pf: &mut Option<Pf>,
) -> Result<()> {
    let inter = cfg.intermediate_size;
    anyhow::ensure!(
        m.gate.n() == inter && m.up.n() == inter && m.down.k() == inter,
        "mlp shapes disagree with intermediate_size {inter}"
    );

    let pf_rides_coop = pf.as_ref().map_or(true, |p| p.coop.is_some());
    let wg = upload_lin(b, "q3d-mlp-gate", &m.gate, W8Scope::Ffn, pf_rides_coop);
    let wu = upload_lin(b, "q3d-mlp-up", &m.up, W8Scope::Ffn, pf_rides_coop);
    let wd = upload_lin(b, "q3d-mlp-down", &m.down, W8Scope::Ffn, pf_rides_coop);

    let in_q = quant_for(b, s, "q3d-mlp-quant", x, wg.k(), &[&wg, &wu])?;
    let y_gate = b.zeros("q3d-mlp-ygate", (inter * 2) as u64);
    let y_up = b.zeros("q3d-mlp-yup", (inter * 2) as u64);
    let merged_gu = match (&wg, &wu, in_q.as_ref()) {
        (DenseLinGpu::Nvfp4(a), DenseLinGpu::Nvfp4(u), Some(q)) if fuse_mlp_gemv_enabled() => {
            push_gemv_nvfp4_2w(b, "q3d-mlp-guproj", a, u, q, &y_gate, &y_up)?
        }
        _ => false,
    };
    if !merged_gu {
        push_gemv(b, s, "q3d-mlp-gproj", &wg, x, in_q.as_ref(), &y_gate)?;
        push_gemv(b, s, "q3d-mlp-uproj", &wu, x, in_q.as_ref(), &y_up)?;
    }

    let fused_mlp = if let DenseLinGpu::Nvfp4(w) = &wd {
        fuse_mlp_enabled() && w.k.is_multiple_of(NVFP4_BLOCK * 4)
    } else {
        false
    };
    if fused_mlp {
        let DenseLinGpu::Nvfp4(w) = &wd else {
            anyhow::bail!("q3d-mlp-siluq: fused_mlp checked the nvfp4 arm")
        };
        let down_q =
            push_silu_mul_quant_rows(b, s, "q3d-mlp-siluq", &y_gate, &y_up, w, wd.k())?;
        push_gemv_nvfp4(b, s, "q3d-mlp-dproj", w, &down_q, out)?;
    } else {
        let act = b.zeros("q3d-mlp-act", (inter * 2) as u64);
        let smp = b.uni(
            "q3d-mlp-silu-p",
            SiluMulParams {
                n_words: (inter / 2) as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1((inter / 2) as u64, 64);
        b.push(
            "q3d-mlp-silu",
            &s.misc,
            "q3w_silu_mul",
            &[(10, &y_gate), (11, &y_up), (12, &act), (13, &smp)],
            grid,
        )?;

        let down_q = quant_for(b, s, "q3d-mlp-dquant", &act, wd.k(), &[&wd])?;
        push_gemv(b, s, "q3d-mlp-dproj", &wd, &act, down_q.as_ref(), out)?;
    }

    if let Some(pf) = pf.as_ref() {
        let hidden_words = cfg.hidden_size / 2;
        let inter_words = inter / 2;
        if [&wg, &wu].iter().any(|w| pf_coop_covers(pf, w)) {
            pf_coop_cast_x(
                b,
                pf,
                "q3d-pf-mlp-castx",
                &pf.normed_post,
                hidden_words,
                wg.k(),
            )?;
        }
        let in_qm = if [&wg, &wu]
            .iter()
            .any(|w| matches!(w, DenseLinGpu::Nvfp4(_)) && !pf_coop_covers(pf, w))
        {
            quant_for_m_rows(
                b,
                s,
                pf,
                "q3d-pf-mlp-quant",
                &pf.normed_post,
                wg.k(),
                &[&wg, &wu],
            )?
        } else {
            None
        };
        push_gemm_pf_dense(
            b,
            s,
            pf,
            "q3d-pf-mlp-gproj",
            &wg,
            &pf.normed_post,
            hidden_words,
            in_qm.as_ref(),
            &pf.m_ygate,
            inter_words,
        )?;
        push_gemm_pf_dense(
            b,
            s,
            pf,
            "q3d-pf-mlp-uproj",
            &wu,
            &pf.normed_post,
            hidden_words,
            in_qm.as_ref(),
            &pf.m_yup,
            inter_words,
        )?;
        let smpm = b.uni(
            "q3d-pf-mlp-silu-p",
            SiluMulParams {
                n_words: (pf.m * inter_words) as u32,
                ..Default::default()
            },
        );
        let sgrid = b.grid1((pf.m * inter_words) as u64, 64);
        b.push_pf(
            "q3d-pf-mlp-silu",
            &s.misc,
            "q3w_silu_mul",
            &[
                (10, &pf.m_ygate),
                (11, &pf.m_yup),
                (12, &pf.m_act),
                (13, &smpm),
            ],
            sgrid,
        )?;
        if pf_coop_covers(pf, &wd) {
            pf_coop_cast_x(b, pf, "q3d-pf-mlp-dcastx", &pf.m_act, inter_words, wd.k())?;
        }
        let down_qm = if matches!(&wd, DenseLinGpu::Nvfp4(_)) && !pf_coop_covers(pf, &wd) {
            quant_for_m_rows(
                b,
                s,
                pf,
                "q3d-pf-mlp-dquant",
                &pf.m_act,
                wd.k(),
                &[&wd],
            )?
        } else {
            None
        };
        push_gemm_pf_dense(
            b,
            s,
            pf,
            "q3d-pf-mlp-dproj",
            &wd,
            &pf.m_act,
            inter_words,
            down_qm.as_ref(),
            &pf.mlp_out,
            hidden_words,
        )?;
    }
    Ok(())
}

impl WeightSource<'_> {
    fn embed(&self, cfg: &Qwen3_5DenseConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.embed.clone()),
            Self::Loader(w) => load_bf16(
                w,
                &[
                    "model.language_model.embed_tokens.weight",
                    "model.embed_tokens.weight",
                ],
                &[cfg.vocab_size, cfg.hidden_size],
            ),
        }
    }

    fn lm_head(&self, cfg: &Qwen3_5DenseConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.lm_head.clone()),
            Self::Loader(w) => {
                if cfg.tie_word_embeddings {
                    self.embed(cfg)
                } else {
                    load_bf16(w, &["lm_head.weight"], &[cfg.vocab_size, cfg.hidden_size])
                }
            }
        }
    }

    fn lm_head_fp8_packed(
        &self,
        cfg: &Qwen3_5DenseConfig,
    ) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        let Self::Loader(w) = self else {
            return Ok(None);
        };
        if cfg.tie_word_embeddings
            || !fp8_lm_head_enabled()
            || w.st_dtype_of("lm_head.weight") != Some(nv_weights::StDtype::F8_E4M3)
        {
            return Ok(None);
        }
        let (vocab, hidden) = (cfg.vocab_size, cfg.hidden_size);
        anyhow::ensure!(
            hidden % 4 == 0,
            "fp8 lm_head packs 4 e4m3 bytes per word; hidden {hidden} % 4 != 0"
        );
        let qw = w.load_quantized_weight("lm_head.weight", nv_weights::QuantScheme::Fp8E4m3)?;
        anyhow::ensure!(
            qw.shape == [vocab, hidden],
            "lm_head.weight: fp8 shape {:?} != [{vocab}, {hidden}]",
            qw.shape
        );
        let scales = qw
            .fp8_weight_scale_rows()?
            .unwrap_or_else(|| vec![1.0f32; vocab]);
        anyhow::ensure!(
            scales.len() == vocab,
            "lm_head.weight_scale rows {} != vocab {vocab}",
            scales.len()
        );
        let packed: Vec<u32> = qw
            .packed_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(Some((packed, scales)))
    }

    fn final_norm(&self, cfg: &Qwen3_5DenseConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.final_norm.clone()),
            Self::Loader(w) => load_norm_plus_one(
                w,
                &["model.language_model.norm.weight", "model.norm.weight"],
                cfg.hidden_size,
            ),
        }
    }

    fn layer_input_ln(&self, cfg: &Qwen3_5DenseConfig, idx: usize) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.layers[idx].input_ln.clone()),
            Self::Loader(w) => load_norm_plus_one(
                w,
                &[
                    &format!("model.language_model.layers.{idx}.input_layernorm.weight"),
                    &format!("model.layers.{idx}.input_layernorm.weight"),
                ],
                cfg.hidden_size,
            ),
        }
    }

    fn layer(&self, cfg: &Qwen3_5DenseConfig, idx: usize) -> Result<HostDenseLayer> {
        match self {
            Self::Host(h) => Ok(h.layers[idx].clone()),
            Self::Loader(w) => load_layer(cfg, w, idx),
        }
    }
}

fn load_bf16(w: &nv_weights::WeightLoader, names: &[&str], shape: &[usize]) -> Result<Vec<u16>> {
    for n in names {
        if w.has(n) {
            if w.st_dtype_of(n) == Some(nv_weights::StDtype::F8_E4M3) {
                return load_fp8_e4m3_as_bf16(w, n, shape);
            }
            let t = w
                .get(n, candle_core::DType::BF16)
                .with_context(|| format!("load {n}"))?;
            anyhow::ensure!(t.dims() == shape, "{n}: shape {:?} != {shape:?}", t.dims());
            let v: Vec<half::bf16> = t.flatten_all()?.to_vec1()?;
            return Ok(v.into_iter().map(|x| x.to_bits()).collect());
        }
    }
    anyhow::bail!("none of {names:?} found")
}

fn load_fp8_e4m3_as_bf16(
    w: &nv_weights::WeightLoader,
    name: &str,
    shape: &[usize],
) -> Result<Vec<u16>> {
    let rows = *shape.first().unwrap_or(&1);
    let cols: usize = shape.iter().skip(1).product::<usize>().max(1);
    let qw = w.load_quantized_weight(name, nv_weights::QuantScheme::Fp8E4m3)?;
    anyhow::ensure!(
        qw.shape == shape,
        "{name}: fp8 shape {:?} != {shape:?}",
        qw.shape
    );
    let scales = qw.fp8_weight_scale_rows()?.unwrap_or_else(|| vec![1.0f32; rows]);
    let dequant = nv_quant::fp8::dequantize_e4m3_per_row(&qw.packed_bytes, rows, cols, &scales)?;
    Ok(dequant.iter().map(|&x| bf16_bits(x)).collect())
}

fn load_norm_plus_one(
    w: &nv_weights::WeightLoader,
    names: &[&str],
    dim: usize,
) -> Result<Vec<u16>> {
    let raw = load_bf16(w, names, &[dim])?;
    if !crate::qwen3_5_moe_wgpu::norm_plus_one_enabled() {
        return Ok(raw);
    }
    Ok(raw
        .into_iter()
        .map(|b| bf16_bits(bf16_val(b) + 1.0))
        .collect())
}

fn load_f32_vec(w: &nv_weights::WeightLoader, names: &[&str], dim: usize) -> Result<Vec<f32>> {
    let raw = load_bf16(w, names, &[dim])?;
    Ok(raw.into_iter().map(bf16_val).collect())
}

fn load_lin(w: &nv_weights::WeightLoader, name: &str, n: usize, k: usize) -> Result<HostBf16Lin> {
    Ok(HostBf16Lin {
        w: load_bf16(w, &[name], &[n, k])?,
        n,
        k,
    })
}

fn load_fp8_lin(
    w: &nv_weights::WeightLoader,
    name: &str,
    n: usize,
    k: usize,
) -> Result<Option<HostFp8Lin>> {
    if w.st_dtype_of(name) != Some(nv_weights::StDtype::F8_E4M3) || !k.is_multiple_of(4) {
        return Ok(None);
    }
    let qw = w.load_quantized_weight(name, nv_weights::QuantScheme::Fp8E4m3)?;
    anyhow::ensure!(
        qw.shape == [n, k],
        "{name}: fp8 shape {:?} != [{n}, {k}]",
        qw.shape
    );
    let scales = qw
        .fp8_weight_scale_rows()?
        .unwrap_or_else(|| vec![1.0f32; n]);
    anyhow::ensure!(
        scales.len() == n,
        "{name}: fp8 weight_scale rows {} != {n}",
        scales.len()
    );
    let packed: Vec<u32> = qw
        .packed_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    Ok(Some(HostFp8Lin {
        packed,
        scales,
        n,
        k,
    }))
}

fn load_dense_lin(
    w: &nv_weights::WeightLoader,
    module: &str,
    n: usize,
    k: usize,
) -> Result<HostDenseLin> {
    let dense = format!("{module}.weight");
    if let Some(fp8) = load_fp8_lin(w, &dense, n, k)? {
        return Ok(HostDenseLin::Fp8 {
            fp8,
            bf16: load_lin(w, &dense, n, k)?,
        });
    }
    if w.has(&dense) {
        return Ok(HostDenseLin::Bf16(load_lin(w, &dense, n, k)?));
    }
    if w.has(&format!("{module}.weight_packed")) {
        return Ok(HostDenseLin::Nvfp4(load_nvfp4(w, module, n, k)?));
    }
    anyhow::bail!("{module}: neither .weight nor .weight_packed present")
}

fn load_layer(
    cfg: &Qwen3_5DenseConfig,
    w: &nv_weights::WeightLoader,
    idx: usize,
) -> Result<HostDenseLayer> {
    let prefix = format!("model.language_model.layers.{idx}");
    let alt = format!("model.layers.{idx}");
    let input_ln = load_norm_plus_one(
        w,
        &[
            &format!("{prefix}.input_layernorm.weight"),
            &format!("{alt}.input_layernorm.weight"),
        ],
        cfg.hidden_size,
    )?;
    let post_attn_ln = load_norm_plus_one(
        w,
        &[
            &format!("{prefix}.post_attention_layernorm.weight"),
            &format!("{alt}.post_attention_layernorm.weight"),
        ],
        cfg.hidden_size,
    )?;

    let base = if w.has(&format!("{prefix}.post_attention_layernorm.weight")) {
        prefix
    } else {
        alt
    };
    let mut delta_fp8 = DeltaFp8::default();
    let mixer = match cfg.layer_types[idx] {
        LayerType::LinearAttention => {
            let p = format!("{base}.linear_attn");
            let n_k = cfg.linear_num_key_heads;
            let n_v = cfg.linear_num_value_heads;
            let d_k = cfg.linear_key_head_dim;
            let d_v = cfg.linear_value_head_dim;
            let key_dim = n_k * d_k;
            let value_dim = n_v * d_v;
            let conv_dim = 2 * key_dim + value_dim;
            let ks = cfg.linear_conv_kernel_dim;
            let a_w = load_bf16(
                w,
                &[&format!("{p}.in_proj_a.weight")],
                &[n_v, cfg.hidden_size],
            )?;
            let b_w = load_bf16(
                w,
                &[&format!("{p}.in_proj_b.weight")],
                &[n_v, cfg.hidden_size],
            )?;
            let mut ab = a_w;
            ab.extend_from_slice(&b_w);
            let conv_raw = load_bf16(w, &[&format!("{p}.conv1d.weight")], &[conv_dim, 1, ks])?;
            delta_fp8 = DeltaFp8 {
                qkv: load_fp8_lin(
                    w,
                    &format!("{p}.in_proj_qkv.weight"),
                    conv_dim,
                    cfg.hidden_size,
                )?,
                z: load_fp8_lin(
                    w,
                    &format!("{p}.in_proj_z.weight"),
                    value_dim,
                    cfg.hidden_size,
                )?,
                out: load_fp8_lin(
                    w,
                    &format!("{p}.out_proj.weight"),
                    cfg.hidden_size,
                    value_dim,
                )?,
            };
            HostDenseMixer::Delta(Box::new(HostDeltaNet {
                in_proj_qkv: load_lin(
                    w,
                    &format!("{p}.in_proj_qkv.weight"),
                    conv_dim,
                    cfg.hidden_size,
                )?,
                in_proj_z: load_lin(
                    w,
                    &format!("{p}.in_proj_z.weight"),
                    value_dim,
                    cfg.hidden_size,
                )?,
                in_proj_ab: HostBf16Lin {
                    w: ab,
                    n: 2 * n_v,
                    k: cfg.hidden_size,
                },
                conv1d: conv_raw.into_iter().map(bf16_val).collect(),
                a_log: load_f32_vec(w, &[&format!("{p}.A_log")], n_v)?,
                dt_bias: load_f32_vec(w, &[&format!("{p}.dt_bias")], n_v)?,
                norm_w: load_bf16(w, &[&format!("{p}.norm.weight")], &[d_v])?,
                out_proj: load_lin(
                    w,
                    &format!("{p}.out_proj.weight"),
                    cfg.hidden_size,
                    value_dim,
                )?,
            }))
        }
        LayerType::FullAttention => {
            let p = format!("{base}.self_attn");
            let hd = cfg.head_dim;
            let q_out = if cfg.attn_output_gate {
                cfg.num_attention_heads * hd * 2
            } else {
                cfg.num_attention_heads * hd
            };
            let kv_out = cfg.num_key_value_heads * hd;
            HostDenseMixer::Attn(Box::new(HostDenseAttention {
                q: load_dense_lin(w, &format!("{p}.q_proj"), q_out, cfg.hidden_size)?,
                k: load_dense_lin(w, &format!("{p}.k_proj"), kv_out, cfg.hidden_size)?,
                v: load_dense_lin(w, &format!("{p}.v_proj"), kv_out, cfg.hidden_size)?,
                o: load_dense_lin(
                    w,
                    &format!("{p}.o_proj"),
                    cfg.hidden_size,
                    cfg.num_attention_heads * hd,
                )?,
                q_norm: load_norm_plus_one(w, &[&format!("{p}.q_norm.weight")], hd)?,
                k_norm: load_norm_plus_one(w, &[&format!("{p}.k_norm.weight")], hd)?,
            }))
        }
    };

    let mp = format!("{base}.mlp");
    let inter = cfg.intermediate_size;
    Ok(HostDenseLayer {
        input_ln,
        post_attn_ln,
        mixer,
        mlp: HostDenseMlp {
            gate: load_dense_lin(w, &format!("{mp}.gate_proj"), inter, cfg.hidden_size)?,
            up: load_dense_lin(w, &format!("{mp}.up_proj"), inter, cfg.hidden_size)?,
            down: load_dense_lin(w, &format!("{mp}.down_proj"), cfg.hidden_size, inter)?,
        },
        delta_fp8,
    })
}

fn rbf(x: f32) -> f32 {
    bf16_val(bf16_bits(x))
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn ref_gemv_bf16(w: &HostBf16Lin, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0f32; w.n];
    for (r, yr) in y.iter_mut().enumerate() {
        let mut acc = 0f32;
        for (c, xc) in x.iter().enumerate().take(w.k) {
            acc += bf16_val(w.w[r * w.k + c]) * xc;
        }
        *yr = acc;
    }
    y
}

fn ref_gemv_dense(w: &HostDenseLin, x: &[f32]) -> Vec<f32> {
    match w {
        HostDenseLin::Bf16(l) => ref_gemv_bf16(l, x),
        HostDenseLin::Fp8 { bf16, .. } => ref_gemv_bf16(bf16, x),
        HostDenseLin::Nvfp4(l) => {
            let dq = crate::qwen3_5_moe_wgpu::dequantize_nvfp4_host(l);
            let mut y = vec![0f32; l.n];
            for (r, yr) in y.iter_mut().enumerate() {
                let mut acc = 0f32;
                for (c, xc) in x.iter().enumerate().take(l.k) {
                    acc += dq[r * l.k + c] * xc;
                }
                *yr = acc;
            }
            y
        }
    }
}

fn ref_rmsnorm(x: &[f32], w: &[u16], eps: f32) -> Vec<f32> {
    let n = x.len();
    let mut ss = 0f32;
    for v in x {
        ss += v * v;
    }
    let inv = 1.0 / (ss / n as f32 + eps).sqrt();
    (0..n).map(|i| rbf(x[i] * inv * bf16_val(w[i]))).collect()
}

pub struct RefState {
    conv: Vec<Vec<f32>>,
    rec: Vec<Vec<f32>>,
    kc: Vec<Vec<f32>>,
    vc: Vec<Vec<f32>>,
    pos: usize,
}

impl RefState {
    pub fn new(cfg: &Qwen3_5DenseConfig) -> Self {
        let l = cfg.num_hidden_layers;
        Self {
            conv: vec![Vec::new(); l],
            rec: vec![Vec::new(); l],
            kc: vec![Vec::new(); l],
            vc: vec![Vec::new(); l],
            pos: 0,
        }
    }
}

pub fn reference_step(
    cfg: &Qwen3_5DenseConfig,
    hw: &HostDenseWeights,
    st: &mut RefState,
    token: u32,
) -> Result<Vec<f32>> {
    let hidden = cfg.hidden_size;
    let eps = cfg.rms_norm_eps as f32;
    let pos = st.pos;

    let mut res: Vec<f32> = (0..hidden)
        .map(|i| bf16_val(hw.embed[token as usize * hidden + i]))
        .collect();

    for li in 0..cfg.num_hidden_layers {
        let layer = &hw.layers[li];
        let normed = ref_rmsnorm(&res, &layer.input_ln, eps);
        let mixed = match &layer.mixer {
            HostDenseMixer::Delta(d) => ref_delta(cfg, d, &normed, st, li),
            HostDenseMixer::Attn(a) => ref_attn(cfg, a, &normed, st, li, pos),
        };
        for i in 0..hidden {
            res[i] = rbf(res[i] + mixed[i]);
        }
        let normed_post = ref_rmsnorm(&res, &layer.post_attn_ln, eps);
        let mlp_out = ref_mlp(cfg, &layer.mlp, &normed_post);
        for i in 0..hidden {
            res[i] = rbf(res[i] + mlp_out[i]);
        }
    }

    let fx = ref_rmsnorm(&res, &hw.final_norm, eps);
    let lm = HostBf16Lin {
        w: hw.lm_head.clone(),
        n: cfg.vocab_size,
        k: hidden,
    };
    st.pos += 1;
    Ok(ref_gemv_bf16(&lm, &fx))
}

fn ref_delta(
    cfg: &Qwen3_5DenseConfig,
    d: &HostDeltaNet,
    x: &[f32],
    st: &mut RefState,
    li: usize,
) -> Vec<f32> {
    let n_k = cfg.linear_num_key_heads;
    let n_v = cfg.linear_num_value_heads;
    let d_k = cfg.linear_key_head_dim;
    let d_v = cfg.linear_value_head_dim;
    let key_dim = n_k * d_k;
    let value_dim = n_v * d_v;
    let conv_dim = 2 * key_dim + value_dim;
    let ks = cfg.linear_conv_kernel_dim;
    let hist = ks - 1;

    let qkv: Vec<f32> = ref_gemv_bf16(&d.in_proj_qkv, x)
        .into_iter()
        .map(rbf)
        .collect();
    let z: Vec<f32> = ref_gemv_bf16(&d.in_proj_z, x)
        .into_iter()
        .map(rbf)
        .collect();
    let ab: Vec<f32> = ref_gemv_bf16(&d.in_proj_ab, x)
        .into_iter()
        .map(rbf)
        .collect();

    if st.conv[li].is_empty() {
        st.conv[li] = vec![0f32; conv_dim * hist];
    }
    let mut mixed = vec![0f32; conv_dim];
    for c in 0..conv_dim {
        let mut acc = 0f32;
        for j in 0..hist {
            acc += d.conv1d[c * ks + j] * st.conv[li][c * hist + j];
        }
        acc += d.conv1d[c * ks + hist] * qkv[c];
        for j in 0..hist.saturating_sub(1) {
            st.conv[li][c * hist + j] = st.conv[li][c * hist + j + 1];
        }
        if hist > 0 {
            st.conv[li][c * hist + hist - 1] = qkv[c];
        }
        mixed[c] = rbf(silu(acc));
    }

    let v_per_k = n_v / n_k;
    let scale = 1.0 / (d_k as f32).sqrt();
    let mut qg = vec![0f32; n_v * d_k];
    let mut kg = vec![0f32; n_v * d_k];
    let mut vg = vec![0f32; n_v * d_v];
    for h in 0..n_v {
        let kh = h / v_per_k;
        let mut sq = 0f32;
        let mut sk = 0f32;
        for i in 0..d_k {
            let qv = mixed[kh * d_k + i];
            let kv = mixed[key_dim + kh * d_k + i];
            sq += qv * qv;
            sk += kv * kv;
        }
        let nq = (sq + 1e-6).sqrt();
        let nk = (sk + 1e-6).sqrt();
        for i in 0..d_k {
            qg[h * d_k + i] = (mixed[kh * d_k + i] / nq) * scale;
            kg[h * d_k + i] = mixed[key_dim + kh * d_k + i] / nk;
        }
        for i in 0..d_v {
            vg[h * d_v + i] = mixed[2 * key_dim + h * d_v + i];
        }
    }

    let mut gexp = vec![0f32; n_v];
    let mut beta = vec![0f32; n_v];
    for i in 0..n_v {
        let a = ab[i];
        let bb = ab[n_v + i];
        beta[i] = sigmoid(bb);
        let t = a + d.dt_bias[i];
        let sp = t.max(0.0) + (1.0 + (-t.abs()).exp()).ln();
        gexp[i] = (sp * -(d.a_log[i].exp())).exp();
    }

    if st.rec[li].is_empty() {
        st.rec[li] = vec![0f32; n_v * d_k * d_v];
    }
    let mut core = vec![0f32; n_v * d_v];
    for h in 0..n_v {
        let base = h * d_k * d_v;
        for dv in 0..d_v {
            let mut kv_mem = 0f32;
            for dk in 0..d_k {
                let s = st.rec[li][base + dk * d_v + dv] * gexp[h];
                st.rec[li][base + dk * d_v + dv] = s;
                kv_mem += s * kg[h * d_k + dk];
            }
            let delta = (vg[h * d_v + dv] - kv_mem) * beta[h];
            let mut outv = 0f32;
            for dk in 0..d_k {
                let s = st.rec[li][base + dk * d_v + dv] + kg[h * d_k + dk] * delta;
                st.rec[li][base + dk * d_v + dv] = s;
                outv += s * qg[h * d_k + dk];
            }
            core[h * d_v + dv] = outv;
        }
    }

    let mut gated = vec![0f32; value_dim];
    for h in 0..n_v {
        let mut ss = 0f32;
        for i in 0..d_v {
            let c = rbf(core[h * d_v + i]);
            ss += c * c;
        }
        let inv = 1.0 / (ss / d_v as f32 + cfg.rms_norm_eps as f32).sqrt();
        for i in 0..d_v {
            let c = rbf(core[h * d_v + i]);
            let n = rbf(c * inv * bf16_val(d.norm_w[i]));
            let g = rbf(silu(z[h * d_v + i]));
            gated[h * d_v + i] = rbf(n * g);
        }
    }
    ref_gemv_bf16(&d.out_proj, &gated)
        .into_iter()
        .map(rbf)
        .collect()
}

fn ref_attn(
    cfg: &Qwen3_5DenseConfig,
    a: &HostDenseAttention,
    x: &[f32],
    st: &mut RefState,
    li: usize,
    pos: usize,
) -> Vec<f32> {
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let rot = cfg.rotary_dim();
    let rh = rot / 2;
    let eps = cfg.rms_norm_eps as f32;

    let q_raw: Vec<f32> = ref_gemv_dense(&a.q, x).into_iter().map(rbf).collect();
    let k_raw: Vec<f32> = ref_gemv_dense(&a.k, x).into_iter().map(rbf).collect();
    let v_raw: Vec<f32> = ref_gemv_dense(&a.v, x).into_iter().map(rbf).collect();

    let stride_q = if cfg.attn_output_gate { 2 * hd } else { hd };
    let rope = |vals: &mut [f32], p: usize| {
        let src: Vec<f32> = vals.to_vec();
        for i in 0..rh {
            let inv = 1.0f32 / cfg.rope_theta.powf((i as f32 * 2.0) / rot as f32);
            let th = (p as f32) * inv;
            let (c, s) = (th.cos(), th.sin());
            vals[i] = src[i] * c - src[i + rh] * s;
            vals[i + rh] = src[i] * s + src[i + rh] * c;
        }
    };

    let mut q = vec![0f32; n_h * hd];
    for h in 0..n_h {
        let row: Vec<f32> = (0..hd).map(|i| q_raw[h * stride_q + i]).collect();
        let mut n = ref_rmsnorm(&row, &a.q_norm, eps);
        rope(&mut n, pos);
        for i in 0..hd {
            q[h * hd + i] = rbf(n[i]);
        }
    }
    let mut kk = vec![0f32; n_kv * hd];
    for h in 0..n_kv {
        let row: Vec<f32> = (0..hd).map(|i| k_raw[h * hd + i]).collect();
        let mut n = ref_rmsnorm(&row, &a.k_norm, eps);
        rope(&mut n, pos);
        for i in 0..hd {
            kk[h * hd + i] = rbf(n[i]);
        }
    }

    st.kc[li].extend_from_slice(&kk);
    st.vc[li].extend_from_slice(&v_raw);
    let total = pos + 1;
    let group = n_h / n_kv;
    let scale = 1.0 / (hd as f32).sqrt();

    let mut out = vec![0f32; n_h * hd];
    for h in 0..n_h {
        let kv = h / group;
        let mut scores = vec![0f32; total];
        let mut m = f32::NEG_INFINITY;
        for (t, sc) in scores.iter_mut().enumerate() {
            let base = (t * n_kv + kv) * hd;
            let mut dot = 0f32;
            for i in 0..hd {
                dot += st.kc[li][base + i] * q[h * hd + i];
            }
            *sc = dot * scale;
            m = m.max(*sc);
        }
        let mut z = 0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            z += *s;
        }
        for i in 0..hd {
            let mut acc = 0f32;
            for (t, s) in scores.iter().enumerate() {
                acc += s * st.vc[li][(t * n_kv + kv) * hd + i];
            }
            out[h * hd + i] = acc / z;
        }
    }

    let mut gated = vec![0f32; n_h * hd];
    for h in 0..n_h {
        for i in 0..hd {
            let av = rbf(out[h * hd + i]);
            if cfg.attn_output_gate {
                let g = q_raw[h * stride_q + hd + i];
                gated[h * hd + i] = av * rbf(sigmoid(g));
            } else {
                gated[h * hd + i] = av;
            }
        }
    }
    ref_gemv_dense(&a.o, &gated).into_iter().map(rbf).collect()
}

fn ref_mlp(cfg: &Qwen3_5DenseConfig, m: &HostDenseMlp, x: &[f32]) -> Vec<f32> {
    let inter = cfg.intermediate_size;
    let yg = ref_gemv_dense(&m.gate, x);
    let yu = ref_gemv_dense(&m.up, x);
    let act: Vec<f32> = (0..inter)
        .map(|i| rbf(rbf(silu(rbf(yg[i]))) * rbf(yu[i])))
        .collect();
    ref_gemv_dense(&m.down, &act).into_iter().map(rbf).collect()
}

crate::wgpu_state_snapshot::impl_wgpu_state_snapshot!(Qwen3_5DenseWgpu, max_seq);
