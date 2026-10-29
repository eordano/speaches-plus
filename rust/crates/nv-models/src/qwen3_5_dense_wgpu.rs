use crate::{dispatch_pass, submit_passes};
use anyhow::{anyhow, bail, ensure, Context, Result};
use bytemuck::{bytes_of, cast_slice};
use half::bf16;
use nv_config::env_arm_opt_in_eq_some_1_reread_every_call;
use nv_weights::WeightLoader;
use std::env;
use std::sync::{Arc, OnceLock};
use std::time::Instant;
use wgpu::{BindGroup, Buffer, BufferUsages, CommandBuffer, ComputePipeline, ErrorFilter};

use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, compose_enabled, strip_subgroup_fns, WgpuContext};

use crate::gemma4_wgpu_shared::{bf16_bits, bf16_val, err};
use crate::pod;
use crate::qwen3_5_moe::LayerType;
pub use crate::qwen3_5_moe::Qwen3_5DenseConfig;
pub use crate::qwen3_5_moe_wgpu::RefState;
use crate::qwen3_5_moe_wgpu::{
    bytes_to_words, delta_recurrent_variants_source, load_f32_vec, DELTA_RECURRENT_ENTRY,
    DELTA_RECURRENT_LANES_PER_WORKGROUP,
    load_nvfp4, nvfp4_gemv_source, nvfp4_quant_source, nvfp4_v2_route, nvfp4_v2_route_slotshared,
    quant_lane_entry, rbf, ref_delta, ref_gemv_bf16, ref_rmsnorm, sigmoid, silu,
    GemvNvfp4Params, HostBf16Lin, HostDeltaNet, HostNvfp4Lin,
    QuantRowsParams, SiluPairParams, VramReport, NVFP4_BLOCK,
};
use crate::wgpu_ledger::env_on;
use crate::wgpu_ledger::{spliced_chunk, ChunkRowSplice};
use crate::wgpu_lora::{
    build_lora_pipelines, build_lora_site, emit_lora_site_pk, emit_lora_site_pk_shrink_only,
    upload_lora_site, LoraPipelines, LoraShrinkFold, LoraSite, LoraSiteGpu,
};

const MAX_HEAD_DIM: usize = 256;
const MAX_LIN_HEAD_DIM: usize = 128;
const ARGMAX_GROUPS: usize = 256;

const STAGING_FLUSH_BYTES: u64 = 256 << 20;

const GEMV_BF16_WGSL: &str = nv_kernels::wgsl!("../../nv-kernels/wgsl/q3d_gemv_bf16.wgsl");

const GEMV_I8_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_gemv_i8.wgsl");

const DELTA_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_delta.wgsl");

const ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_attn.wgsl");

const MISC_WGSL: &str = concat!(
    include_str!("../../nv-kernels/wgsl/q3d_misc.wgsl"),
    include_str!("../../nv-kernels/wgsl/q3w_argmax.wgsl")
);

const PREFILL_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3d_prefill.wgsl");

const GEMV_W4A8_WGSL: &str = nv_kernels::wgsl!("../../nv-kernels/wgsl/gemv_nvfp4_w4a8.wgsl");

const W4A8_PRODUCER_WGSL: &str =
    nv_kernels::wgsl!("../../nv-kernels/wgsl/w4a8_producer_rowquant.wgsl");

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

pub const SG_DIRECTIVE_IS_BROWSER_ONLY_BECAUSE_NAGA_30_REJECTS_ENABLE_SUBGROUPS_AT_PARSE: &str =
    "Tint requires 'enable subgroups;' before any subgroup builtin and wgpu's webgpu \
     backend hands the WGSL to the browser unparsed, while naga 30 rejects that directive as an \
     unsupported enable-extension yet accepts the builtins bare, so the wasm32 build injects the \
     directive into subgroup-granted modules and the native build keeps its sources byte-\
     identical; an ungranted device on either target gets the subgroup fns stripped and the \
     selection guards keep those arms de-selected";

fn sg_wrap(sg: bool, composed: String) -> String {
    let _ = SG_DIRECTIVE_IS_BROWSER_ONLY_BECAUSE_NAGA_30_REJECTS_ENABLE_SUBGROUPS_AT_PARSE;
    if !sg {
        return strip_subgroup_fns(&composed);
    }
    #[cfg(target_arch = "wasm32")]
    {
        sg_inject(&composed)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        composed
    }
}

fn sg_inject(composed: &str) -> String {
    format!("enable subgroups;\n{composed}")
}

fn sg_compose(sg: bool, body: &str) -> String {
    sg_wrap(sg, compose(body))
}

fn prefill_composed(sg: bool) -> String {
    sg_compose(sg, &format!("{PREFILL_WGSL}\n{SPLICE_WGSL}"))
}

const LMHEAD_EXPAND_WGSL_EXISTS_BECAUSE_EVERY_SHIPPING_NVFP4_GEMV_STORES_BF16_Y_AND_ARGMAX_READS_F32:
    &str = r#"
struct Q3lxParams { n_words: u32, y_off_elems: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> qlx_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> qlx_out: array<f32>;
@group(0) @binding(2) var<uniform> qlx_p: Q3lxParams;
@compute @workgroup_size(256)
fn q3w_lmhead_bf16_expand_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    if (w >= qlx_p.n_words) { return; }
    let v = qlx_in[w];
    qlx_out[qlx_p.y_off_elems + 2u * w] = bitcast<f32>((v & 0xFFFFu) << 16u);
    qlx_out[qlx_p.y_off_elems + 2u * w + 1u] = bitcast<f32>(v & 0xFFFF0000u);
}
"#;

pod! {
    struct LmheadExpandParams {
        n_words: u32,
        y_off_elems: u32,
        pad0: u32,
        pad1: u32,
    }
}

fn gemm_mk_source(m: usize) -> String {
    use std::fmt::Write as _;
    assert!((2..=16).contains(&m), "gemm_mk m must be 2..=16, got {m}");
    let mut b = String::new();
    b.push_str(
        "struct Q3gmParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    w_row_words: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    pad0: u32,
    pad1: u32,
};

@group(0) @binding(0) var<storage, read> gm_w: array<u32>;
@group(0) @binding(1) var<storage, read> gm_x: array<u32>;
@group(0) @binding(2) var<uniform> gm_p: Q3gmParams;
@group(0) @binding(3) var<storage, read_write> gm_y: array<u32>;

var<workgroup> gm_red: array<f32, 256>;

",
    );
    writeln!(b, "@compute @workgroup_size(256)").unwrap();
    writeln!(b, "fn {}(", gemm_mk_entry(m)).unwrap();
    b.push_str(
        "    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let tid = lid.x;
    let half = tid >> 7u;
    let lane = tid & 127u;
    let pair = wid.x + wid.y * gm_p.groups_x;
    let row = pair * 2u + half;
    let live = row < gm_p.n_rows;
    let wbase = select(0u, row * gm_p.w_row_words, live);
    let kw = select(0u, gm_p.k_words, live);
",
    );
    writeln!(
        b,
        "    var acc: array<f32, {m}>;
    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{ acc[mi] = 0.0; }}"
    )
    .unwrap();
    b.push_str(
        "    for (var i = lane; i < kw; i = i + 128u) {
        let ww = gm_w[wbase + i];
",
    );
    writeln!(b, "        for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
    b.push_str(
        "            let xw = gm_x[mi * gm_p.x_stride_words + i];
            acc[mi] = fma(bf16_lo(ww), bf16_lo(xw), acc[mi]);
            acc[mi] = fma(bf16_hi(ww), bf16_hi(xw), acc[mi]);
        }
    }
",
    );
    writeln!(b, "    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
    b.push_str(
        "        gm_red[tid] = acc[mi];
        workgroupBarrier();
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
            if (lane < stride) {
                gm_red[tid] = gm_red[tid] + gm_red[tid + stride];
            }
            workgroupBarrier();
        }
        if (tid == 0u) {
            let lo = gm_red[0];
            var hi = 0.0;
            if (row + 1u < gm_p.n_rows) {
                hi = gm_red[128];
            }
            gm_y[mi * gm_p.y_stride_words + (row >> 1u)] = bf16_pack(lo, hi);
        }
        workgroupBarrier();
    }
}
",
    );
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
        "struct Q3q8mParams {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    groups_per_row: u32,
    group_shift: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> q8m_w: array<u32>;
@group(0) @binding(1) var<storage, read> q8m_s: array<f32>;
@group(0) @binding(2) var<storage, read> q8m_x: array<u32>;
@group(0) @binding(3) var<storage, read_write> q8m_y: array<u32>;
@group(0) @binding(4) var<uniform> q8m_p: Q3q8mParams;

var<workgroup> q8m_pk_bits: array<u32, 8>;

",
    );
    for grouped in [true, false] {
        writeln!(b, "@compute @workgroup_size(256)").unwrap();
        writeln!(b, "fn {}(", gemm_i8_mk_entry(m, grouped)).unwrap();
        b.push_str(
            "    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * q8m_p.groups_x) * 8u + sgid;
    let live = row < q8m_p.n_rows;
    let words = select(0u, q8m_p.k_elems >> 2u, live);
    let wbase = select(0u, row * (q8m_p.k_elems >> 2u), live);
",
        );
        if grouped {
            b.push_str(
                "    let sbase = select(0u, row * q8m_p.groups_per_row, live);
    let gshift = q8m_p.group_shift;
",
            );
        }
        writeln!(
            b,
            "    var acc: array<f32, {m}>;
    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{ acc[mi] = 0.0; }}"
        )
        .unwrap();
        b.push_str(
            "    for (var i = lane; i < words; i = i + 32u) {
        let w = q8m_w[wbase + i];
",
        );
        if grouped {
            b.push_str("        let sc = q8m_s[sbase + (i >> gshift)];\n");
        }
        writeln!(b, "        for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
        b.push_str(
            "            let x0 = q8m_x[mi * q8m_p.x_stride_words + 2u * i];
            let x1 = q8m_x[mi * q8m_p.x_stride_words + 2u * i + 1u];
",
        );
        if grouped {
            b.push_str(
                "            var d = 0.0;
            d = fma(int8_decode(w, 0u), bf16_lo(x0), d);
            d = fma(int8_decode(w, 1u), bf16_hi(x0), d);
            d = fma(int8_decode(w, 2u), bf16_lo(x1), d);
            d = fma(int8_decode(w, 3u), bf16_hi(x1), d);
            acc[mi] = fma(sc, d, acc[mi]);
",
            );
        } else {
            b.push_str(
                "            acc[mi] = fma(int8_decode(w, 0u), bf16_lo(x0), acc[mi]);
            acc[mi] = fma(int8_decode(w, 1u), bf16_hi(x0), acc[mi]);
            acc[mi] = fma(int8_decode(w, 2u), bf16_lo(x1), acc[mi]);
            acc[mi] = fma(int8_decode(w, 3u), bf16_hi(x1), acc[mi]);
",
            );
        }
        b.push_str("        }\n    }\n");
        writeln!(b, "    for (var mi = 0u; mi < {m}u; mi = mi + 1u) {{").unwrap();
        b.push_str(
            "        var a = acc[mi];
        a = a + subgroupShuffleXor(a, 16u);
        a = a + subgroupShuffleXor(a, 8u);
        a = a + subgroupShuffleXor(a, 4u);
        a = a + subgroupShuffleXor(a, 2u);
        a = a + subgroupShuffleXor(a, 1u);
",
        );
        if grouped {
            b.push_str(
                "        if (lane == 0u) {
            q8m_pk_bits[sgid] = bf16_encode(a) & 0xffffu;
        }
",
            );
        } else {
            b.push_str(
                "        if (lane == 0u) {
            q8m_pk_bits[sgid] = bf16_encode(a * q8m_s[select(0u, row, live)]) & 0xffffu;
        }
",
            );
        }
        b.push_str(
            "        workgroupBarrier();
        if ((sgid & 1u) == 0u && lane == 0u && live) {
            var word = q8m_pk_bits[sgid];
            if (row + 1u < q8m_p.n_rows) {
                word = word | (q8m_pk_bits[sgid + 1u] << 16u);
            }
            q8m_y[mi * q8m_p.y_stride_words + (row >> 1u)] = word;
        }
        workgroupBarrier();
    }
}

",
        );
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

pod! {
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

    struct CkParams {
        m_live: u32,
        base: u32,
        pad0: u32,
        pad1: u32,
    }

    struct SpliceParams {
        hidden_words: u32,
        m: u32,
        pad0: u32,
        pad1: u32,
    }

    struct ConvMParams {
        conv_dim: u32,
        kernel: u32,
        x_words: u32,
        mixed_stride: u32,
    }

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

    struct RmsParams {
        hidden: u32,
        batch: u32,
        eps: f32,
        words_per_row: u32,
    }

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

    struct ConvParams {
        conv_dim: u32,
        kernel: u32,
        pad0: u32,
        pad1: u32,
    }

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

    struct GatingParams {
        n_v: u32,
        pad0: u32,
        pad1: u32,
        pad2: u32,
    }

    struct RecurrentParams {
        heads: u32,
        d_k: u32,
        d_v: u32,
        pad0: u32,
    }

    struct DeltaOutParams {
        n_v: u32,
        d_v: u32,
        pad0: u32,
        eps: f32,
    }

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

    struct KvWriteParams {
        words: u32,
        pad0: u32,
        pad1: u32,
        pad2: u32,
    }

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

    struct SiluMulParams {
        n_words: u32,
        pad0: u32,
        pad1: u32,
        pad2: u32,
    }

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

    struct BiasRopeFusedParams {
        n_q_rows: u32,
        n_k_rows: u32,
        n_v_rows: u32,
        head_dim: u32,
        rot_half: u32,
        write_kv: u32,
        pad1: u32,
        pad2: u32,
    }
}

pub const ATTN_SHALLOW_ENV_ONE_DISPATCH_PER_LAYER_REPLACES_FLASH1_FLASH2_GATE_HD128_BF16KV_GATELESS_DEFAULT_OFF:
    &str = "NV_Q3D_ATTN_SHALLOW";

pub const BIAS_ROPE_WRITES_KV_ENV_DROPS_THE_KVWRITE_PASS_LEAVES_ANY_FP8_PF_CACHE_STALE_PAST_THE_PROMPT_DEFAULT_OFF:
    &str = "NV_Q3D_BIAS_ROPE_KVW";

pod! {
    struct AttnShallowParams {
        n_kv: u32,
        head_dim: u32,
        group: u32,
        rot_half: u32,
        scale: f32,
        splits: u32,
        pad2: u32,
        pad3: u32,
    }
}

pod! {
    struct AttnFlashMParams {
        n_kv: u32,
        head_dim: u32,
        group: u32,
        q_row_elems: u32,
        y_row_words: u32,
        pad0: u32,
        pad1: u32,
        scale: f32,
    }

    struct BiasRopeMParams {
        n_rows: u32,
        head_dim: u32,
        src_stride: u32,
        rot_half: u32,
        x_row_elems: u32,
        y_row_elems: u32,
        bias_off: u32,
        pad0: u32,
    }

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

    struct GatherParams {
        row_off: u32,
        n_rows: u32,
        hidden_words: u32,
        vocab: u32,
    }

    struct ArgmaxParams {
        n: u32,
        groups: u32,
        pad0: u32,
        pad1: u32,
    }
}

#[derive(Clone)]
pub enum HostDenseLin {
    Bf16(HostBf16Lin),
    Nvfp4(HostNvfp4Lin),
    AffineW4(HostAffineW4Lin),
    Fp8 { fp8: HostFp8Lin, bf16: HostBf16Lin },
}

#[derive(Clone)]
pub struct HostAffineW4Lin {
    pub words: Vec<u32>,
    pub scales_bf16: Vec<u16>,
    pub biases_bf16: Vec<u16>,
    pub n: usize,
    pub k: usize,
    pub gs: usize,
}

impl HostDenseLin {
    pub fn n(&self) -> usize {
        match self {
            Self::Bf16(l) => l.n,
            Self::Nvfp4(l) => l.n,
            Self::AffineW4(l) => l.n,
            Self::Fp8 { fp8, .. } => fp8.n,
        }
    }

    pub fn k(&self) -> usize {
        match self {
            Self::Bf16(l) => l.k,
            Self::Nvfp4(l) => l.k,
            Self::AffineW4(l) => l.k,
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
    pub qkv_bias_bf16: Vec<u16>,
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

#[derive(Clone, Default)]
pub struct DeltaAffineW4 {
    pub qkv: Option<HostAffineW4Lin>,
    pub z: Option<HostAffineW4Lin>,
    pub out: Option<HostAffineW4Lin>,
}

pub const DELTA_AFFINE_RIDES_A_SIDE_CHANNEL_BECAUSE_HOST_DELTANET_FIELDS_ARE_BF16_STRUCTS: &str =
    "HostDeltaNet keeps mandatory HostBf16Lin projections (shared with the moe graph and the \
     host reference), so mlx-affine checkpoints dequantize those to bf16 at load and carry the \
     packed payloads here; build_delta uploads the affine twin and skips the bf16 upload, \
     mirroring the DeltaFp8 pattern";

pub const FP8_PROJ_STREAMS_CHECKPOINT_BYTES_INSTEAD_OF_A_BF16_UPCAST: &str =
    "the qwen3.8 nvfp4 checkpoints ship the DeltaNet projections as F8_E4M3 with per-row \
     scales; loading them through load_bf16 upcasts 3-bit-mantissa data into an 8-bit-mantissa \
     container and doubles the decode weight stream for zero information; the prefill M-row \
     gemm keeps the bf16 copy, so fp8 residency is additive VRAM, not a swap";

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
    static T: OnceLock<u32> = OnceLock::new();
    *T.get_or_init(|| {
        env::var("NV_Q3D_FLASH_TILE")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .map(|t| t.clamp(0, 8))
            .unwrap_or(0)
    })
}

fn kv_fp8_enabled() -> bool {
    nv_config::env_arm_opt_out_default_on_unless_eq_some_0_cached_first_read_wins!("NV_Q3D_KV_FP8")
}

pub const KV_FP8_TP_LANE_PER_POSITION_STAGE1_OVER_A_SECOND_TRANSPOSED_K_CACHE: &str =
    "the tp stage1 stores the K fp8 bytes a second time in channel-plane-major order \
     (word index (kvh * hd/4 + d4) * slots + pos, k_transposed_word_index) and routes decode \
     stage1 to the fdf*tp* fold arm, whose inner loop gives each subgroup lane its OWN KV \
     position: the q.k dot runs lane-serial over channels with no per-position 5-shuffle \
     butterfly, which FLASH_TILE_STAYS_OFF_THE_COOPERATIVE_DOT_PAYS_THE_REDUCE_NOT_THE_RESCALE \
     identifies as where llama.cpp's fattn-vec edge lives. Synthetic fold6 hd256 ladder: \
     stage1 us/dispatch 971->766 at 172032 depth 64 splits, \
     441->297 at 65536/64, 110->86 at 16384/64; loses 43->55 at 2048/16 where the dispatch is \
     a trivial share of the step. Real-checkpoint A/B: perf/runs.jsonl kv-fp8-tp rows. Costs \
     one extra K-sized fp8 state buffer per full-attn layer (max_seq*n_kv*head_dim bytes) plus \
     one O(1) row-writer dispatch per decode step and one M-row twin per chunk; V and the \
     position-major K cache are unchanged because the chunk attention, mk verify, and nvfp4 \
     arms keep reading them";

pub const TPW_ARM_SWAPS_THE_TP_STAGE1_BODY_NOT_THE_ARM_SELECTION: &str =
    "the fdf*tpw* stage1 entry serves wherever the tp depth gate selects tp: same transposed K cache, same \
     bindings, same grid, and \
     TPW_IS_THE_TP_ARM_WITH_LANE_OWNED_V_WORDS_BITWISE_IDENTICAL_BY_CONSTRUCTION (nv-kernels \
     flash_decode) plus the tp-vs-tpw bitwise asserts in q3d_flash_stage1_rate (single-round \
     and multi-round) carry the proof that only the V load pattern changes -- the tp arm's \
     per-element word fetches issue four redundant loads per V word, tpw loads each word once \
     per lane. M2 Ultra synthetic ladder: \
     us/dispatch 5220 -> 2302 at 172032/64, 1995 -> 894 at 65536/64, 623 -> 291 at 16384/64; \
     e2e deep decode 123.0 -> 76.6 ms/tok at d172032; requires hd % 128 == 0, else the tp \
     body serves unchanged";

pub const KV_FP8_TP_MIN_TOTAL_ENV_TP_RUNS_ONLY_AT_TOTAL_AT_OR_ABOVE_THIS_PLAIN_INTEGER: &str =
    "NV_Q3D_KV_FP8_TP_MIN_TOTAL";

pub const TP_MIN_TOTAL_UNSET_MEANS_THE_DEPTH_GATE_SET_0_FOR_TP_EVERY_STEP_SET_HUGE_FOR_FOLD_ALWAYS:
    &str = "the signed-off default is the depth gate itself: fold stage1 below the knee, tp at or \
     above it, matching whichever pure arm wins at that depth per the tp-depthgated A/B rows in \
     perf/runs.jsonl";

pub const TP_ENGAGES_PAST_THE_MEASURED_CROSSOVER_DEPTH_REANCHORED_BY_THE_Q38_TPKNEE_ROWS: u32 =
    4096;

pub const TPKNEE_4096_IS_THE_TPW_ERA_CROSSOVER_SET_MIN_TOTAL_49152_FOR_THE_PRE_TPW_KNEE: &str =
    "the 49152 knee was measured on the per-element tp body (q38 tpknee rows); the tpw body \
     moved the crossover to the SPLITS_DEEP_KV_FROM_TOTAL=4096 boundary on M2 Ultra -- e2e \
     sd-vs-tpw 40.03/41.06 ms at 256 (sd wins), 44.58/42.52 at 4096, 46.77/44.68 at 8192 \
     (q38rival midknee rows) -- because tpw's win rides the splits-64 regime that engages at \
     exactly that total; NV_Q3D_KV_FP8_TP_MIN_TOTAL pins any knee (49152 restores the pre-tpw \
     default, 0 forces tp/tpw every step); a sibling backend that has not re-stamped its \
     tpw-era knee pins the old value through the same env";

pub const DEPTH_GATE_SELECTS_AT_ENCODE_BECAUSE_STEP_CBS_ARE_REENCODED_EVERY_STEP: &str =
    "Both stage1 arms are recorded adjacently in the static step pass list (q3d-at-flash1 then \
     q3d-at-flash1-tp per full-attn layer) and the per-step command buffer -- which this decoder \
     already re-encodes every step, preenc merely pipelining it one deterministic step ahead -- \
     skips exactly one arm of each pair by total KV depth. Encode-time selection was chosen over \
     an indirect-dispatch pair (adds an args buffer plus one indirect read per arm per layer per \
     step) and over an always-dispatch uniform early-exit (16 full-attn layers x the ~3us \
     dispatch floor lands ~0.05ms/step, a ~0.25% tax exactly where the gate exists to erase a \
     3.6% one) because it costs zero GPU dispatches and zero kernel changes; the A/B rows \
     (build qwen3.8 nvfp4 tp-depthgated vs both pure arms, perf/runs.jsonl) are the measurement \
     that the gated build matches the position-major arm below the knee and the tp arm above it \
     within noise. The knee itself comes from the q38 tpknee rows in perf/runs.jsonl \
     (fold wins at 32768 and below, arms dead even at the knee, tp wins at 65536; the pre-dgap \
     ladder put the knee at 32768 and the dgap-era kernel gains moved it), hence \
     TP_ENGAGES_PAST_THE_MEASURED_CROSSOVER_DEPTH_REANCHORED_BY_THE_Q38_TPKNEE_ROWS. \
     The transposed kc8t cache and both \
     its writer twins stay recorded unconditionally: they are O(1)/step, and keeping them \
     unconditional makes the switch pure dispatch selection with no cache backfill at the \
     crossover. The mtp drafter graph records a single arm (its replay loops never filter).";

fn kv_fp8_tp_min_total() -> u32 {
    match env::var(KV_FP8_TP_MIN_TOTAL_ENV_TP_RUNS_ONLY_AT_TOTAL_AT_OR_ABOVE_THIS_PLAIN_INTEGER) {
        Ok(v) => v.trim().parse::<u32>().unwrap_or_else(|_| {
            panic!(
                "{}={v:?} must be a plain integer total-KV-depth threshold",
                KV_FP8_TP_MIN_TOTAL_ENV_TP_RUNS_ONLY_AT_TOTAL_AT_OR_ABOVE_THIS_PLAIN_INTEGER
            )
        }),
        Err(_) => {
            let _ = TP_MIN_TOTAL_UNSET_MEANS_THE_DEPTH_GATE_SET_0_FOR_TP_EVERY_STEP_SET_HUGE_FOR_FOLD_ALWAYS;
            TP_ENGAGES_PAST_THE_MEASURED_CROSSOVER_DEPTH_REANCHORED_BY_THE_Q38_TPKNEE_ROWS
        }
    }
}

fn is_fold_stage1_label(l: &str) -> bool {
    l == "q3d-at-flash1" || l.starts_with("q3d-at-flash1:")
}

fn is_tp_stage1_label(l: &str) -> bool {
    l == "q3d-at-flash1-tp" || l.starts_with("q3d-at-flash1-tp:")
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
     ppl for zero depth win (runs.jsonl kv-v4k8 rows). v4k8 held the pinned-corpus ppl band at \
     its best face (+0.0038) but re-measures at +0.0057 on the shipped basis and +0.0078 on the \
     pure-fp8 basis (rows 'q38 v4k8 speed', pr-20260826-15*) -- near the k4v4 fence -- and has \
     NO speed win at any depth on the tp-stage1 default: tp_capable requires \
     kv_nvfp4_arm().is_none(), so the arm forfeits the deep-depth winner (-3% @64k, -11% @172k \
     vs shipped) and within the fold family halved V bytes tie at <=64k because the depth tax \
     is K-side score work; a tp+v4 composition caps at ~1.5 ms/tok @172k against the quality \
     penalty and is not built. Do not adopt v4k8 without re-passing ppl on the current floor \
     AND beating the tp default at depth; k4v4 does not hold the ppl gate at \
     any measured sink-exemption size, and the Hadamard-rotation fallback that clause once \
     pointed at is measured and does not rescue it: post-k_norm K carries no channel outliers \
     to disperse (max/median channel amax 2.0-3.5x across the stack), so blocked rotation at \
     16/32/64 leaves e2m1 reconstruction error at the same ~9.5% rel RMS as unrotated \
     (fp8 sits at 2.6%), the teacher-forced ppl deltas of the equal-error rotated arms span \
     +0.002..+0.010 nats over the same-session fp8 control -- realization noise, not a \
     rescue -- and causal per-channel smoothing lowers elementwise error while measuring \
     WORSE ppl because it moves precision off the high-amax channels the q.k dot actually \
     weights (runs.jsonl q38-rotkv rows; tests/qwen38_rotkv_probe.rs is the harness)";

#[derive(Clone, Copy, PartialEq, Eq)]
enum KvNvfp4Arm {
    V4K8,
    K4V4,
}

fn kv_nvfp4_arm() -> Option<KvNvfp4Arm> {
    match env::var(KV_NVFP4_ENV_OFF_BY_DEFAULT_FP8_STAYS_THE_SHIPPED_KV_DECODE_FORMAT)
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

pub const REQUANT_FP8_TO_NVFP4_ENV_MLP_ALWAYS_REQUANTS_A_COMMA_LIST_OR_ALL_ADDS_ATTN_DN_LMHEAD:
    &str = "NV_Q3D_REQUANT_FP8_TO_NVFP4";

pub const REQUANT_HALVES_THE_FP8_DECODE_WEIGHT_STREAM_BUT_CUTS_BELOW_CHECKPOINT_PRECISION_SO_THE_PPL_PAIR_ADJUDICATES:
    &str = "the qwen3.8 nvfp4 checkpoint ships the DeltaNet qkv/z/out projections, attention \
     q/k/v/o, lm_head and its last 8 MLP layers as F8_E4M3, streamed at 8 bpw every decode \
     step; this arm requantizes them at load into the shipping swizzled nvfp4 block-16 \
     format so decode rides the existing nvfp4/w4a8 gemv routes at ~4.5 bpw. nvfp4 holds \
     fewer mantissa bits than the checkpoint's own fp8, so the arm is default OFF and any \
     flip must cite the teacher-forced ppl pair; the DN group keeps its fp8 upload for the \
     coop prefill route, so DN VRAM under this arm is additive, and lm_head logits ride a \
     bf16 stage plus an expand pass because every shipping nvfp4 gemv stores bf16 y";

fn requant_fp8_group_enabled(group: &str) -> bool {
    if group == "mlp" {
        return true;
    }
    let v = env::var(
        REQUANT_FP8_TO_NVFP4_ENV_MLP_ALWAYS_REQUANTS_A_COMMA_LIST_OR_ALL_ADDS_ATTN_DN_LMHEAD,
    )
    .unwrap_or_default();
    match v.trim() {
        "1" | "all" => true,
        list => list.split(',').any(|g| g.trim() == group),
    }
}

static REQUANT_BOOT_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static REQUANT_BOOT_TENSORS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

static REQUANT_BOOT_FP8_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub const REQUANT_ROW_CHUNK_IS_A_MULTIPLE_OF_THE_128_ROW_SCALE_SWIZZLE_TILE_SO_CHUNK_SCALE_BUFFERS_CONCATENATE:
    usize = 1024;

pub fn requant_fp8_rowscale_to_nvfp4(
    packed: &[u32],
    scales: &[f32],
    n: usize,
    k: usize,
) -> Result<HostNvfp4Lin> {
    let t0 = Instant::now();
    ensure!(
        n.is_multiple_of(128) && k.is_multiple_of(NVFP4_BLOCK * 4),
        "requant fp8->nvfp4 needs n % 128 == 0 (the scale swizzle tile) and k % {} == 0 \
         (the quant_rows tile); got n={n} k={k}",
        NVFP4_BLOCK * 4
    );
    let bytes: &[u8] = cast_slice(packed);
    let amax = nv_quant::fp8::amax_e4m3_per_row(bytes, n, k, scales)?;
    let stored_global = if amax.is_finite() && amax > 0.0 {
        (448.0f32 * 6.0) / amax
    } else {
        1.0
    };
    let alpha = if stored_global.is_finite() && stored_global != 0.0 {
        1.0 / stored_global
    } else {
        1.0
    };
    let chunk =
        REQUANT_ROW_CHUNK_IS_A_MULTIPLE_OF_THE_128_ROW_SCALE_SWIZZLE_TILE_SO_CHUNK_SCALE_BUFFERS_CONCATENATE;
    let n_chunks = n.div_ceil(chunk);
    let mut parts: Vec<(Vec<u8>, Vec<u8>)> = vec![(Vec::new(), Vec::new()); n_chunks];
    let workers = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(n_chunks);
    let per_worker = n_chunks.div_ceil(workers);
    let pack_part = |wi: usize, part_slice: &mut [(Vec<u8>, Vec<u8>)]| {
        for (j, slot) in part_slice.iter_mut().enumerate() {
            let r0 = (wi * per_worker + j) * chunk;
            let rows = chunk.min(n - r0);
            let dq = nv_quant::fp8::dequantize_e4m3_per_row(
                &bytes[r0 * k..(r0 + rows) * k],
                rows,
                k,
                &scales[r0..r0 + rows],
            )
            .expect("chunk shape and scale count were checked against n and k above");
            let rowvecs: Vec<Vec<f32>> = dq.chunks_exact(k).map(<[f32]>::to_vec).collect();
            let q =
                nv_quant::nvfp4::Nvfp4Tensor::quantize_rows_with_global(&rowvecs, stored_global);
            let sw = q.scales_swizzled();
            *slot = (q.data, sw);
        }
    };
    #[cfg(target_arch = "wasm32")]
    for (wi, part_slice) in parts.chunks_mut(per_worker).enumerate() {
        pack_part(wi, part_slice);
    }
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::scope(|sc| {
        let pack_part = &pack_part;
        for (wi, part_slice) in parts.chunks_mut(per_worker).enumerate() {
            sc.spawn(move || pack_part(wi, part_slice));
        }
    });
    let mut out_packed = Vec::with_capacity(n * k / 2);
    let mut scales_swizzled =
        Vec::with_capacity(wk::gemv_nvfp4::swizzled_scale_len(n, k / NVFP4_BLOCK));
    for (p, sc) in parts {
        out_packed.extend_from_slice(&p);
        scales_swizzled.extend_from_slice(&sc);
    }
    use std::sync::atomic::Ordering::Relaxed;
    REQUANT_BOOT_NANOS.fetch_add(t0.elapsed().as_nanos() as u64, Relaxed);
    REQUANT_BOOT_TENSORS.fetch_add(1, Relaxed);
    REQUANT_BOOT_FP8_BYTES.fetch_add((n * k) as u64, Relaxed);
    Ok(HostNvfp4Lin {
        packed: out_packed,
        scales_swizzled,
        alpha,
        input_global: 1.0,
        n,
        k,
    })
}

fn requant_boot_report() {
    use std::sync::atomic::Ordering::Relaxed;
    let tensors = REQUANT_BOOT_TENSORS.swap(0, Relaxed);
    if tensors == 0 {
        return;
    }
    let nanos = REQUANT_BOOT_NANOS.swap(0, Relaxed);
    let bytes = REQUANT_BOOT_FP8_BYTES.swap(0, Relaxed);
    eprintln!(
        "[q3d-wgpu] requant fp8->nvfp4: {tensors} tensors, {:.2} GiB fp8 in, {:.2} s host quantize",
        bytes as f64 / (1u64 << 30) as f64,
        nanos as f64 / 1e9
    );
}

pub const PF_COOP_ENV_1_ALSO_ROUTES_SMALL_M_GRAPHS_THROUGH_COOP: &str = "NV_Q3D_PF_COOP";

pub const PF_COOP_ROUTES_PF_PROJECTIONS_THROUGH_16X16_COOPMAT_AND_DROPS_THE_FP8_BF16_TWIN: &str =
    "The chunked-prefill projections route through the 16x16 f16 cooperative-matrix kernels by \
     default: w4a16 (WqFmt::Nvfp4Block16) for nvfp4 MLP weights, w8a16 \
     (WqFmt::Fp8RowscalePlain) for fp8 DN qkv/z/out and attn q/k/v/o projections, and the plain \
     f16 coop GEMM for the small bf16 DN ab projection. The coop route lifts the \
     NV_WGPU_PREFILL_M default from 16 to 512 (ceiling 1024) and skips the fp8 projections' bf16-twin \
     upload, whose only consumer was the m-row bf16 gemm the coop path replaces. Decode passes \
     are byte-identical on both arms; prefill numerics drift within the ppl gate because the \
     a16 side multiplies f16 activations instead of quantized ones; the m<=16 bf16 gemm arm \
     carries the small-m verify graphs and adapters without coop fragments.";

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
    env_arm_opt_in_eq_some_1_reread_every_call(
        PF_ATTN_MK_ENV_DEFAULT_OFF_UNTIL_THE_LADDER_AND_PPL_GATE_ARE_ON_RECORD,
    )
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

pod! {
    struct PfAttnMkQcastParams {
        n_words: u32,
        pad0: u32,
        pad1: u32,
        scale: f32,
    }
}

pub const PF_ATTN_TILED_ENV_1_ALSO_ROUTES_SMALL_M_GRAPHS_THROUGH_TILED_FLASH: &str = "NV_Q3D_PF_ATTN_TILED";

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
     NV_Q3D_PF_ATTN_MK=1 selects the mk arm instead of the tiled one.";

pub const PF_ATTN_COOP_REROUTES_TILED_STAGE1_THROUGH_16X16_F16_COOPMAT: &str =
    "the coop tiled pf stage1 replaces the scalar tiled pf stage1 \
     (q3w_pf_flash1_fp8kv_tiled_slotml_*) with q3w_pf_flash1_fp8kv_tiled_coop_sg: QK^T and P.V \
     run as 16x16x16 f16 cooperative-matrix products over an exactly-decoded f16 stage of the \
     fp8 KV cache (every e4m3 value is f16-representable), Q rides a bf16->f16 cast instead of \
     bf16->f32, and the online softmax keeps one block m/l chain per row instead of slotml's \
     per-slot chains. Scores and the output accumulator stay f32; the same split scratch feeds \
     the unchanged flash_splitk_stage2_mk, so only f16 operand rounding and sum reassociation \
     move the numbers (kernel twin gate: max_rel < 2e-2 through stage2 on random fp8 KV, \
     tests/wgpu_pf_tiled_v2.rs; model band: teacher-forced chunked-prefill ppl within +0.01 \
     nats of the slotml arm on the 27B checkpoint, the bound the slotml sign-off used). The \
     kernel is hd=256/subgroup-32/m%32==0 only and measured 8.6-9.3x over the slotml arm at \
     4k-172k KV on the sm_120 box (runs.jsonl \"qwen3.8 prefill-r1\"); unsupported geometry \
     falls back to the slotml arm loudly.";

fn pf_attn_tiled_enabled() -> bool {
    !pf_attn_mk_enabled()
}

fn host_pf_projections_ride_the_coop_route(h: &HostDenseWeights, tile: Option<u32>) -> bool {
    if tile == Some(8) {
        let affine = |l: &HostDenseLin| matches!(l, HostDenseLin::AffineW4(_));
        return h.layers.iter().all(|l| {
            let mixer_ok = match &l.mixer {
                HostDenseMixer::Delta(_) => false,
                HostDenseMixer::Attn(a) => [&a.q, &a.k, &a.v, &a.o].iter().all(|w| affine(w)),
            };
            mixer_ok
                && [&l.mlp.gate, &l.mlp.up, &l.mlp.down]
                    .iter()
                    .all(|w| affine(w))
        });
    }
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

fn host_carries_affine_w4(h: &HostDenseWeights) -> bool {
    let lin = |l: &HostDenseLin| matches!(l, HostDenseLin::AffineW4(_));
    h.layers.iter().any(|l| {
        lin(&l.mlp.gate)
            || lin(&l.mlp.up)
            || lin(&l.mlp.down)
            || match &l.mixer {
                HostDenseMixer::Attn(a) => lin(&a.q) || lin(&a.k) || lin(&a.v) || lin(&a.o),
                HostDenseMixer::Delta(_) => false,
            }
    })
}

fn loader_is_mlx_affine_probing_layer0_because_mlx_quantizes_the_whole_trunk(
    w: &WeightLoader,
) -> bool {
    [
        "model.language_model.layers.0",
        "language_model.model.layers.0",
        "model.layers.0",
    ]
    .iter()
    .any(|p| w.has(&format!("{p}.mlp.gate_proj.scales")))
}

pub const PF_COOP_AFFINE_ENV_DEFAULT_ON_FOR_THE_QKV_BIAS_ARCH_ONLY_BECAUSE_THE_QWEN3_AFFINE_PREFILL_NUMBERS_ARE_PINNED_ON_THE_M16_GEMM:
    &str = "NV_Q3D_PF_COOP_AFFINE";

fn loader_affine_rides_pf_coop(w: &WeightLoader) -> bool {
    match env::var(PF_COOP_AFFINE_ENV_DEFAULT_ON_FOR_THE_QKV_BIAS_ARCH_ONLY_BECAUSE_THE_QWEN3_AFFINE_PREFILL_NUMBERS_ARE_PINNED_ON_THE_M16_GEMM)
        .ok()
        .as_deref()
    {
        Some("0") => return false,
        Some("1") => return true,
        _ => {}
    }
    [
        "model.language_model.layers.0",
        "language_model.model.layers.0",
        "model.layers.0",
    ]
    .iter()
    .any(|p| w.has(&format!("{p}.self_attn.q_proj.bias")))
}

fn loader_pf_projections_ride_the_coop_route_probing_layer0_of_each_type_because_checkpoints_are_uniform(
    w: &WeightLoader,
    cfg: &Qwen3_5DenseConfig,
    tile: Option<u32>,
) -> bool {
    let fp8 = |name: String| w.st_dtype_of(&name) == Some(nv_weights::StDtype::F8_E4M3);
    let affine_coop = loader_affine_rides_pf_coop(w);
    let affine_only_because_tile8_has_no_wq_staging = tile == Some(8);
    let quantized = |module: String| {
        if affine_only_because_tile8_has_no_wq_staging {
            return affine_coop && w.has(&format!("{module}.scales"));
        }
        w.has(&format!("{module}.weight_packed"))
            || fp8(format!("{module}.weight"))
            || (affine_coop && w.has(&format!("{module}.scales")))
    };
    let mut dn_ok = true;
    let mut attn_ok = true;
    let mut mlp_ok = true;
    for (i, t) in cfg.layer_types.iter().enumerate() {
        let p = [
            format!("model.language_model.layers.{i}"),
            format!("language_model.model.layers.{i}"),
            format!("model.layers.{i}"),
        ]
        .into_iter()
        .find(|p| w.has(&format!("{p}.post_attention_layernorm.weight")))
        .unwrap_or_else(|| format!("model.layers.{i}"));
        match t {
            LayerType::LinearAttention => {
                dn_ok = if affine_only_because_tile8_has_no_wq_staging {
                    affine_coop
                        && w.has(&format!("{p}.linear_attn.in_proj_qkv.scales"))
                        && w.has(&format!("{p}.linear_attn.in_proj_z.scales"))
                        && w.has(&format!("{p}.linear_attn.out_proj.scales"))
                } else {
                    fp8(format!("{p}.linear_attn.in_proj_qkv.weight"))
                        && fp8(format!("{p}.linear_attn.in_proj_z.weight"))
                        && fp8(format!("{p}.linear_attn.out_proj.weight"))
                };
            }
            LayerType::FullAttention => {
                attn_ok = quantized(format!("{p}.self_attn.q_proj"))
                    && quantized(format!("{p}.self_attn.o_proj"));
            }
        }
        mlp_ok = quantized(format!("{p}.mlp.gate_proj")) && quantized(format!("{p}.mlp.down_proj"));
        if !(dn_ok && attn_ok && mlp_ok) {
            return false;
        }
        if i >= 2 {
            break;
        }
    }
    dn_ok && attn_ok && mlp_ok
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

pod! {
    struct PfKvqMParams {
        tokens: u32,
        x_stride_elems: u32,
        pad0: u32,
        pad1: u32,
    }
}

fn pf_coop_tile(ctx: &WgpuContext) -> Option<u32> {
    static T: OnceLock<Option<u32>> = OnceLock::new();
    *T.get_or_init(
        || match wk::gemm_coop_f16::select(ctx, wk::gemm_coop_f16::Operand::F16) {
            Ok(g) => Some(g.tile),
            Err(why) => {
                eprintln!(
                    "[q3d-wgpu] no cooperative-matrix f16 fragment on this adapter ({why});                      the m<=16 gemm arm carries prefill"
                );
                None
            }
        },
    )
}

pub const COOP8_TN_SHRINKS_UNTIL_96_COLUMN_BANDS: u32 = 96;

pub const APPLE_COOP_IS_8X8_SO_THE_AFFINE_PREFILL_RIDES_AN_F16_TWIN_NOT_THE_WQ16_STAGING: &str =
    "metal simdgroup matrices are 8x8; the wq16 in-kernel-dequant coop arms stage one 16-code      weight block per fragment row and require 16x16 fragments, so on an 8x8 adapter the      mlx-affine projections dequantize to an f16 twin at load (prefill-only VRAM, the decode      gemvs keep the packed w4) and ride the plain-f16 tile-8 coop gemm instead";

fn pf_coop_active(ctx: &WgpuContext) -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(
        || match wk::gemm_coop_f16::select(ctx, wk::gemm_coop_f16::Operand::F16) {
            Ok(g) if g.tile == 16 => true,
            Ok(g) => {
                eprintln!(
                    "[q3d-wgpu] the default coop prefill needs 16x16 f16 \
                     fragments for the wq16 staging unpack but the adapter serves only {t}x{t}; \
                     keeping the m<=16 gemm path",
                    t = g.tile
                );
                false
            }
            Err(why) => {
                eprintln!(
                    "[q3d-wgpu] the default coop prefill is unavailable: \
                     {why}; keeping the m<=16 gemm path"
                );
                false
            }
        },
    )
}

pub const PF_SCAN_WY_ENV_1_ALSO_ROUTES_SMALL_M_GRAPHS_THROUGH_THE_WY_SCAN: &str = "NV_Q3D_PF_SCAN_WY";

pub const PF_SCAN_WY_SOLVES_32_TOKEN_SUBCHUNKS_AS_A_UT_TRANSFORM_INSTEAD_OF_TOKEN_SERIAL: &str =
    "The prefill DeltaNet scan routes through q3w_delta_scan_wy by default, which \
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

pub const PF_SCAN_WY_V2_ENV_OPT_IN_SET_1_TO_HOIST_OUTPUTS_OFF_THE_SERIAL_LADDER: &str =
    "NV_Q3D_PF_SCAN_WY_V2";

pub const PF_SCAN_WY_V2_SPLITS_THE_WY_SCAN_INTO_A_SERIAL_LADDER_AND_A_PARALLEL_O_PASS_AND_LOSES:
    &str = "NV_Q3D_PF_SCAN_WY_V2=1 replaces the single q3w_delta_scan_wy dispatch with a \
     q3w_delta_scan_wy_ck ladder plus a q3w_delta_scan_wy_o output pass. The ladder keeps only \
     the serial chain -- u = beta (v - decayed K S_prev) forward-substituted and the state \
     carry -- checkpointing the chunk-start state and cumulative log-gates off the serial path \
     and writing substituted u where outputs will land; the o pass then computes \
     O = exp(cl) q S_ckpt + W u fully parallel over (sub-chunk, head, v-split) with grid z = \
     sub-chunks, replaying the exact op order of the inline arm so out and state gate bitwise \
     against q3w_delta_scan_wy (tests/q3d_delta_scan_wy_v2.rs). The split LOSES on this \
     backend and stays opt-in as the measured negative of the o-hoist idea: the hoistable \
     output work is fused with serial-side loops in the inline arm (aq shares the state-read \
     loop with ak, dqk shares the k.k pair loop with dkk, q shares the k staging), so the \
     ladder sheds only ~15% of the inline stage while the unfused o pass re-pays the shared \
     loads in full, and wgpu serializes the two dispatches into a strict sum -- the q38 \
     wgpu-ohoist rows in perf/runs.jsonl are the A/B (8k 674.9 -> 653.4 tok/s, 32k 585.1 -> \
     569.5; profiled stage 1511 ms inline vs 1282 + 591 ms split). The CUDA twin of the idea \
     won isolated and washed e2e (gdnv3); on wgpu it does not even win isolated. Do not re-run \
     this experiment without first unfusing cheaper than the o pass re-pays.";

fn pf_scan_wy_v2_requested() -> bool {
    env_arm_opt_in_eq_some_1_reread_every_call(
        PF_SCAN_WY_V2_ENV_OPT_IN_SET_1_TO_HOIST_OUTPUTS_OFF_THE_SERIAL_LADDER,
    )
}

pub const WY_SCAN_WORKGROUP_BYTES_TWO_16K_VEC4_TILES_TWO_4K_ROWS_TWO_128B_LANES: u32 = 41216;

pub const WY16_SCAN_WORKGROUP_BYTES_TWO_8K_VEC4_TILES_TWO_1K_TRIANGLES_ONE_2K_SCALAR_U_TWO_64B_LANES: u32 = 20608;

fn pf_scan_wy_route(d_k: usize, d_v: usize, max_workgroup_storage: u32) -> (&'static str, u32) {
    if max_workgroup_storage < WY_SCAN_WORKGROUP_BYTES_TWO_16K_VEC4_TILES_TWO_4K_ROWS_TWO_128B_LANES
    {
        if max_workgroup_storage
            >= WY16_SCAN_WORKGROUP_BYTES_TWO_8K_VEC4_TILES_TWO_1K_TRIANGLES_ONE_2K_SCALAR_U_TWO_64B_LANES
            && d_k == 128
            && d_v == 128
        {
            eprintln!(
                "[q3d-wgpu] wy32 needs {} B of workgroup storage, adapter reports \
                 {max_workgroup_storage}: routing the prefill DeltaNet scan to the half-chunk \
                 q3w_delta_scan_wy16 ({} B)",
                WY_SCAN_WORKGROUP_BYTES_TWO_16K_VEC4_TILES_TWO_4K_ROWS_TWO_128B_LANES,
                WY16_SCAN_WORKGROUP_BYTES_TWO_8K_VEC4_TILES_TWO_1K_TRIANGLES_ONE_2K_SCALAR_U_TWO_64B_LANES
            );
            return (
                "q3w_delta_scan_wy16",
                PF_SCAN_WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS,
            );
        }
        eprintln!(
            "[q3d-wgpu] q3w_delta_scan_wy needs {} B of workgroup storage and this adapter \
             reports {max_workgroup_storage}: Metal ran the over-limit pipeline anyway with \
             aliased tiles and produced 0.63-rel-wrong state (content-blind serving); routing \
             the prefill DeltaNet scan to the token-serial q3w_delta_scan",
            WY_SCAN_WORKGROUP_BYTES_TWO_16K_VEC4_TILES_TWO_4K_ROWS_TWO_128B_LANES
        );
        return ("q3w_delta_scan", 1);
    }
    if d_k == 128 && d_v == 128 {
        (
            "q3w_delta_scan_wy",
            PF_SCAN_WY_VSPLIT_4_WORKGROUPS_PER_HEAD_EACH_OWNS_32_V_COLUMNS,
        )
    } else {
        ("q3w_delta_scan", 1)
    }
}

pub const VF_ATTN_FLASH_ENV_DEFAULT_GATE_AT_THE_MEASURED_KNEE_SMV2_LOOP_OR_MIN_TOTAL_PICK_AN_ARM: &str =
    "NV_Q3D_VF_ATTN_FLASH";

pub const VF_ATTN_FLASH_GIVES_SMALL_M_VERIFY_GRAPHS_SPLIT_K_FLASH_OVER_THE_FP8_KV_CACHE: &str =
    "NV_Q3D_VF_ATTN_FLASH replaces the scores-slab attention of every small-m (2..=8 row) \
     verify/MTP graph with flash-class split-k attention over the fp8 e4m3 KV cache the M=1 \
     decode path already maintains, because the slab arm scans rows x full KV depth with a \
     serial per-thread AV loop and is the dominant term of the verify forward at depth. \
     =smv2 (or =1) records one flash_smv2_stage1_fp8kv_sd + flash_splitk_stage2_mk_u pair per \
     attn layer: per-row causal ends come from the whole-view fd uniform (total = base + m, \
     row qi ends at base+qi+1, the same end the slab computes), one dispatch serves every row. \
     =loop records m per-row dispatches of the same shift-decode fold stage1 the M=1 decode \
     path runs (GQA-folded, sd-tiled where the adapter allows), each row with its own fd \
     uniform (m_rows 1, total = base+row+1) and a 256-byte-aligned row offset into q/out; the \
     tp depth-gated twin stays decode-only because the pf pass lists replay unfiltered. Both \
     arms read chain-row KV that q3w_kv_write_m and q3w_pf_quantize_kv_fp8_m recorded earlier \
     in the same pass list, which is what makes row i see rows < i. Numerics move from the \
     slab's exact-bf16 reads to the fp8+shift-decode class the M=1 decode path already ships, \
     so the gate is the verify4 argmax-match + drift pin, not bit identity. A plain integer \
     value is the depth gate: the graph records BOTH the slab arm and the loop arm adjacently \
     and each verify encode dispatches exactly one of them -- slab while base + live rows is \
     below the value, loop at or above it -- because loop pays a per-row dispatch floor that \
     loses below the measured knee and wins multiples above it (the vflash2 A/B rows in \
     perf/runs.jsonl). =gate selects the measured knee \
     (VF_FLASH_LOOP_ENGAGES_AT_THE_MEASURED_VERIFY_CROSSOVER_TOTAL). Default unset \
     keeps the slab arm; the flip decision stays with the orchestrator.";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum VfAttnFlashArm {
    Smv2,
    RowLoop,
}

fn vf_attn_flash_arm() -> (VfAttnFlashArm, Option<u32>) {
    let raw =
        env::var(VF_ATTN_FLASH_ENV_DEFAULT_GATE_AT_THE_MEASURED_KNEE_SMV2_LOOP_OR_MIN_TOTAL_PICK_AN_ARM)
            .ok();
    match raw.as_deref() {
        None | Some("") | Some("gate") => (
            VfAttnFlashArm::RowLoop,
            Some(VF_FLASH_LOOP_ENGAGES_AT_THE_MEASURED_VERIFY_CROSSOVER_TOTAL),
        ),
        Some("smv2") | Some("1") => (VfAttnFlashArm::Smv2, None),
        Some("loop") => (VfAttnFlashArm::RowLoop, None),
        Some(other) => match other.parse::<u32>() {
            Ok(min_total) if min_total >= 2 => (VfAttnFlashArm::RowLoop, Some(min_total)),
            _ => panic!(
                "{VF_ATTN_FLASH_ENV_DEFAULT_GATE_AT_THE_MEASURED_KNEE_SMV2_LOOP_OR_MIN_TOTAL_PICK_AN_ARM}={other}: \
                 set smv2 (or 1), loop, gate, or a min-total integer >= 2"
            ),
        },
    }
}

pub const VF_FLASH_LOOP_ENGAGES_AT_THE_MEASURED_VERIFY_CROSSOVER_TOTAL: u32 = 512;

pub const VF_FLASH_GATE_SELECTS_AT_ENCODE_BECAUSE_VERIFY_CBS_ARE_REENCODED_EVERY_ROUND: &str =
    "A depth-gated verify graph records the scores-slab pass (q3d-pf-at-decode) and the loop \
     flash passes (q3d-pf-at-vfl-*) adjacently per attn layer in the same pf/vf pass list, and \
     the verify_chain / prefill-chunk encoders -- which already build a fresh command buffer \
     from that list on every call -- skip exactly one arm by total KV depth, mirroring \
     DEPTH_GATE_SELECTS_AT_ENCODE_BECAUSE_STEP_CBS_ARE_REENCODED_EVERY_STEP: zero extra GPU \
     dispatches, zero kernel changes, and the commit scan-replay list carries no attention \
     passes at all so the scan bit-identity contract never sees the arm. The KV writers and \
     fp8 quantize dispatches stay recorded unconditionally on both sides of the knee, so the \
     switch is pure dispatch selection with no cache backfill at the crossover.";

fn vfl_gate_keeps(label: &str, gate_min_total: Option<u32>, total: u32) -> bool {
    let Some(min_total) = gate_min_total else {
        return true;
    };
    if label.starts_with("q3d-pf-at-vfl-") {
        return total >= min_total;
    }
    if label.starts_with("q3d-pf-at-decode") {
        return total < min_total;
    }
    true
}

pub const PF_ATTN_SCORES_SLAB_IS_M_X_NHEADS_X_MAXSEQ_F32_SO_PF_M_CLAMPS_TO_THIS_BUDGET_BYTES: u64 =
    1 << 31;

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
    Loader(&'a WeightLoader),
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

struct Pass {
    pipeline: Arc<ComputePipeline>,
    bind: BindGroup,
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

pub const RESUME_CHECKPOINT_COPIES_DELTANET_STATE_BECAUSE_ONLY_ATTENTION_KV_REWINDS_BY_POS: &str =
    "attention KV rows written past a rewound pos stay resident but are masked out by fd.total, \
     the same invariant verify_chain's partial-accept rollback stands on, so rewinding the \
     attention layers is moving pos; the DeltaNet recurrent and short-conv buffers fold every \
     token they ever saw and hold no position, so the only reachable earlier state is a copied \
     one: save_resume_checkpoint clones every recurrent_states buffer GPU-side at the end of a \
     prompt prefill and rewind_to restores exactly that position. Both refuse while an MTP head \
     is attached (its shifted KV and hidden track the committed stream) and while mrope rows are \
     installed (the rope tables no longer describe the text stream the checkpoint claims).";

pub const GRID_SLOT_CAP_BOUNDS_VRAM_AT_CAP_DELTANET_CLONES_AND_EVICTS_THE_LOWEST_BOUNDARY_BECAUSE_A_DIVERGING_SUFFIX_REWINDS_TO_THE_HIGHEST_BOUNDARY_UNDER_ITS_LCP: usize = 12;

pub const REWIND_DROPS_CHECKPOINTS_ABOVE_THE_TARGET_BECAUSE_THE_RESUMED_STREAM_DIVERGES_THERE:
    &str = "a checkpoint saved above a rewind target captured the OLD stream's state at that \
     position; after the rewind the re-prefilled suffix differs, so admitting such a checkpoint \
     on a later request whose lcp reaches its position would resume a state the served stream \
     never had -- a wrong hit, not a cache miss. rewind_to therefore invalidates every grid slot \
     and the single resume checkpoint above the target.";

struct Verify {
    rows: usize,
    res2_rows: Buffer,
    resadd: Pass,
    chunk_resadd: Pass,
    tok: Buffer,
    row_logits: Option<Buffer>,
    rollback: Vec<Buffer>,
    validated: bool,
    pending: Option<Vec<u32>>,
    pending_rode_vf: bool,
}

pub const PF_M_DEFAULT_512_PPL_PAIR_IS_PLUS_0_0109_NATS_OVER_M256_ACCEPTED_BY_USER_UNDER_THE_WIDENED_0_011_NAT_BAND: usize = 512;

pub const PF_M_EXPLICIT_ENV_CEILING_1024_ON_THE_COOP_ROUTE: usize = 1024;

pub fn prefill_m() -> usize {
    let (default_m, cap) = (
        PF_M_DEFAULT_512_PPL_PAIR_IS_PLUS_0_0109_NATS_OVER_M256_ACCEPTED_BY_USER_UNDER_THE_WIDENED_0_011_NAT_BAND,
        PF_M_EXPLICIT_ENV_CEILING_1024_ON_THE_COOP_ROUTE,
    );
    match env::var("NV_WGPU_PREFILL_M")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
    {
        Some(m) => m.clamp(2, cap),
        None => default_m,
    }
}

pub const VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM: &str =
    "NV_Q3D_VERIFY_M";

pub const VERIFY_GRAPH_COOP_ROWS_ROUND_TO_32_THE_TILED_FLASH_TILE_HEIGHT_AND_TWO_COOP_FRAGMENT_ROWS:
    usize = 32;

pub const VERIFY_M_UNSET_DEFAULT_4_THE_KP1_CHAIN_AT_THE_SHIPPED_K3: usize = 4;

pub const VERIFY_M_DEFAULT_RECORDS_ONLY_BESIDE_A_COOP_PREFILL_GRAPH: &str =
    "legacy m<=16 prefill graphs have no pad pathology worth a second graph, and their callers \
     hold the historic contract verify_max_rows() == prefill m (the tiny verify-chain suites pin \
     it and drive full-width chains), so the unset default must not shrink chain capacity there; \
     an explicit NV_Q3D_VERIFY_M still records beside any strictly larger prefill graph";

pub fn verify_m_dedicated_graph_rows_requested() -> usize {
    env::var(VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(VERIFY_M_UNSET_DEFAULT_4_THE_KP1_CHAIN_AT_THE_SHIPPED_K3)
}

pub fn verify_m_env_is_explicit() -> bool {
    env::var(VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .is_some()
}

pub const MTP_COMMIT_SCAN_REPLAY_RESTORES_THE_SNAPSHOT_AND_REDISPATCHES_ONLY_EACH_DELTANET_CONV_SHIFT_AND_SCAN:
    &str = "every m-row kernel in the verify graph is causal in the row index and the graph \
     shape is fixed, so the DeltaNet scan inputs (q, k, v, g, beta rows and the pre-conv qkv \
     rows) for rows below the accepted prefix are bit-identical between the k+1-row verify \
     forward and the commit-only prefix reforward; attention KV rows written past the accepted \
     prefix are masked out by position exactly as on the reforward path. The scan-replay arm \
     therefore lands those inputs in per-layer buffers during the verify forward, and advance(n) \
     commits a partial accept by restoring the rollback snapshot and re-dispatching only each \
     DeltaNet layer's conv_shift and scan at m_live = n, reproducing the commit-only reforward's \
     committed state bit-for-bit while skipping every projection, attention and MLP pass. The \
     arm needs the dedicated verify graph and is inert without it.";

fn na_enabled() -> bool {
    env_on("NV_WGPU_NA")
}

struct PfCoop {
    tile: u32,
    tm: u32,
    tn: u32,
    sg: u32,
    m_alloc: u32,
    x_f16: Buffer,
    y_f32: Buffer,
    zero: Buffer,
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

pod! {
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
}

struct PfAttnMk {
    q_f32: Buffer,
    scratch: Buffer,
    fd: Vec<Buffer>,
}

struct PfAttnTiled {
    q_f32: Buffer,
    scratch: Buffer,
    fd: Buffer,
    start: Buffer,
    q_f16_coop: Option<Buffer>,
}

struct PfAttnVflash {
    arm: VfAttnFlashArm,
    gate_min_total: Option<u32>,
    q_f32: Buffer,
    scratch: Buffer,
    row_fd: Vec<Buffer>,
}

struct Pf {
    m: usize,
    vf: bool,
    attn_mk: Option<PfAttnMk>,
    attn_tiled: Option<PfAttnTiled>,
    attn_vflash: Option<PfAttnVflash>,
    attn_kvq_fd_start: Option<(Buffer, Buffer)>,
    na: Option<Arc<ComputePipeline>>,
    coop: Option<PfCoop>,
    gemm_src: String,
    gemm_entry: String,
    gemm_i8_src: String,
    ck: Buffer,
    tok: Buffer,
    splice: Buffer,
    mask: Buffer,
    res: Buffer,
    normed: Buffer,
    mix: Buffer,
    mlp_out: Buffer,
    normed_post: Buffer,
    d_qkv: Buffer,
    d_z: Buffer,
    d_ab: Buffer,
    d_mixed: Buffer,
    d_q: Buffer,
    d_k: Buffer,
    d_v: Buffer,
    d_g: Buffer,
    d_beta: Buffer,
    d_core: Buffer,
    d_wy_ckpt: Option<(Buffer, Buffer)>,
    d_gated: Buffer,
    a_qraw: Buffer,
    a_kraw: Buffer,
    a_vraw: Buffer,
    a_q: Buffer,
    a_k: Buffer,
    a_scores: Buffer,
    a_attn: Buffer,
    a_gated: Buffer,
    m_ygate: Buffer,
    m_yup: Buffer,
    m_act: Buffer,
}

struct Builder {
    core: crate::wgpu_ledger::VramLedger,
    passes: Vec<Pass>,
    pf_passes: Vec<Pass>,
    vf_passes: Vec<Pass>,
    vf_commit_passes: Vec<Pass>,
    pf_dst_vf: bool,
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
        binds: &[(u32, &Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<Pass> {
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, label, source, entry)
            .map_err(|e| anyhow!("pipeline {label}::{entry}: {e}"))?;
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        Ok(Pass {
            pipeline,
            bind,
            grid,
            label: if dispatch::profile::enabled() {
                format!("{label}:{entry}")
            } else {
                label.to_string()
            },
        })
    }

    fn push(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let pass = self.make(label, source, entry, binds, grid)?;
        self.passes.push(pass);
        Ok(())
    }

    fn pf_dst(&mut self) -> &mut Vec<Pass> {
        if self.pf_dst_vf {
            &mut self.vf_passes
        } else {
            &mut self.pf_passes
        }
    }

    fn pf_dst_len(&self) -> usize {
        if self.pf_dst_vf {
            self.vf_passes.len()
        } else {
            self.pf_passes.len()
        }
    }

    fn push_pf(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &Buffer)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        if env_arm_opt_in_eq_some_1_reread_every_call("NV_Q3D_DUMP_PF_PASSES") {
            eprintln!(
                "[pf-pass {}{:>3}] {label} :: {entry} grid={grid:?}",
                if self.pf_dst_vf { "vf " } else { "" },
                self.pf_dst_len()
            );
        }
        let pass = self.make(label, source, entry, binds, grid)?;
        self.pf_dst().push(pass);
        Ok(())
    }

    fn push_pf_off(
        &mut self,
        label: &str,
        source: &str,
        entry: &str,
        binds: &[(u32, &Buffer, u64)],
        grid: (u32, u32, u32),
    ) -> Result<()> {
        let pipeline = dispatch::cached_compute_pipeline(self.ctx, label, source, entry)
            .map_err(|e| anyhow!("pipeline {label}::{entry}: {e}"))?;
        let bind = dispatch::bind_group_offsets(self.ctx, &pipeline, binds);
        let pass = Pass {
            pipeline,
            bind,
            grid,
            label: if dispatch::profile::enabled() {
                format!("{label}:{entry}")
            } else {
                label.to_string()
            },
        };
        self.pf_dst().push(pass);
        Ok(())
    }

    fn push_pf_pipeline(
        &mut self,
        label: &str,
        pipeline: Arc<ComputePipeline>,
        binds: &[(u32, &Buffer)],
        grid: (u32, u32, u32),
    ) {
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        let pass = Pass {
            pipeline,
            bind,
            grid,
            label: if dispatch::profile::enabled() {
                label.to_string()
            } else {
                String::new()
            },
        };
        self.pf_dst().push(pass);
    }
}

struct Sources {
    gemv_bf16: String,
    gemv_i8: String,
    gemv_nvfp4: String,
    gemv_w4a8: String,
    w4a8_prod: String,
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
        ("q3d:attn", sg_compose(true, ATTN_WGSL)),
        ("q3d:misc", compose(MISC_WGSL)),
    ]
}

#[doc(hidden)]
pub fn shipped_gemv_i8_source() -> String {
    sg_compose(true, GEMV_I8_WGSL)
}

#[doc(hidden)]
pub fn shipped_delta_source() -> String {
    compose(DELTA_WGSL)
}

#[doc(hidden)]
pub fn shipped_prefill_source() -> String {
    prefill_composed(true)
}

#[doc(hidden)]
pub fn browser_module_sources(sg: bool) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = Sources::with_subgroups(sg)
        .named()
        .into_iter()
        .map(|(tag, src)| {
            let src = if sg && src.contains("subgroup") {
                sg_inject(&src)
            } else {
                src
            };
            (format!("q3d:{tag}"), src)
        })
        .collect();
    for m in [2usize, 8, 16] {
        v.push((format!("q3d:gemm_mk{m}"), gemm_mk_source(m)));
        let i8src = gemm_i8_mk_source(m);
        v.push((
            format!("q3d:gemm_i8_mk{m}"),
            if sg {
                sg_inject(&i8src)
            } else {
                strip_subgroup_fns(&i8src)
            },
        ));
    }
    for (hd, fold) in [(256u32, 6u32), (128, 6)] {
        let body = wk::flash_decode::fold_stage1_source_sd(hd, sg, fold);
        let sd = fold_composed(sg, &body);
        v.push((
            format!("q3d:fold_sd_hd{hd}_f{fold}"),
            if sg { sg_inject(&sd) } else { sd },
        ));
        let body4 = wk::flash_decode::fold_stage1_source_nvfp4(hd, sg, fold, true);
        let nv4 = fold_composed(sg, &body4);
        v.push((
            format!("q3d:fold_nvfp4_hd{hd}_f{fold}"),
            if sg { sg_inject(&nv4) } else { nv4 },
        ));
        if sg {
            let tp = wk::flash_decode::fold_stage1_source_sd_tp(hd, fold);
            v.push((
                format!("q3d:fold_tp_hd{hd}_f{fold}"),
                sg_inject(&fold_composed(true, &tp)),
            ));
            let tpw = wk::flash_decode::fold_stage1_source_sd_tpw(hd, fold);
            v.push((
                format!("q3d:fold_tpw_hd{hd}_f{fold}"),
                sg_inject(&fold_composed(true, &tpw)),
            ));
        }
    }
    v
}

fn fold_composed(sg: bool, body: &str) -> String {
    let full = compose(&format!("{}\n{}", wk::flash_decode::WGSL, body));
    if !sg {
        return full;
    }
    #[cfg(target_arch = "wasm32")]
    {
        sg_inject(&full)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        full
    }
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
    fn new(ctx: &WgpuContext) -> Self {
        Self::with_subgroups(ctx.caps.subgroup)
    }

    fn with_subgroups(sg: bool) -> Self {
        Self {
            gemv_bf16: compose(GEMV_BF16_WGSL),
            gemv_i8: sg_compose(sg, GEMV_I8_WGSL),
            gemv_nvfp4: nvfp4_gemv_source(),
            gemv_w4a8: sg_compose(sg, GEMV_W4A8_WGSL),
            w4a8_prod: compose(W4A8_PRODUCER_WGSL),
            quant: sg_wrap(sg, nvfp4_quant_source()),
            delta: compose(&format!(
                "{}\n{}",
                DELTA_WGSL,
                delta_recurrent_variants_source()
            )),
            attn: sg_compose(sg, ATTN_WGSL),
            flash: compose(wk::flash_decode::WGSL),
            kvq: compose(wk::kv_fp8::WGSL),
            kvq4: compose(wk::kv_nvfp4::WGSL),
            misc: compose(MISC_WGSL),
            rms: compose(wk::rmsnorm::WGSL),
            rmsres: compose(wk::rmsnorm_residual::WGSL),
            resscale: compose(wk::residual_scale::WGSL),
            prefill: prefill_composed(sg),
        }
    }

    fn named(self) -> Vec<(&'static str, String)> {
        vec![
            ("gemv_bf16", self.gemv_bf16),
            ("gemv_i8", self.gemv_i8),
            ("gemv_nvfp4", self.gemv_nvfp4),
            ("gemv_w4a8", self.gemv_w4a8),
            ("w4a8_prod", self.w4a8_prod),
            ("quant", self.quant),
            ("delta", self.delta),
            ("attn", self.attn),
            ("flash", self.flash),
            ("kvq", self.kvq),
            ("kvq4", self.kvq4),
            ("misc", self.misc),
            ("rms", self.rms),
            ("rmsres", self.rmsres),
            ("resscale", self.resscale),
            ("prefill", self.prefill),
        ]
    }
}

struct Bf16Gpu {
    w: Buffer,
    n: usize,
    k: usize,
}

struct Nvfp4Gpu {
    w: Buffer,
    scales: Buffer,
    scales_linear: Option<Buffer>,
    alpha: f32,
    input_global: f32,
    n: usize,
    k: usize,
}

struct I8Gpu {
    w: Buffer,
    s: Buffer,
    n: usize,
    k: usize,
    group: usize,
}

struct AffineW4Gpu {
    w: Buffer,
    scales: Buffer,
    biases: Buffer,
    f16_twin: Option<Buffer>,
    n: usize,
    k: usize,
    gs: usize,
}

enum DenseLinGpu {
    Bf16(Bf16Gpu),
    Nvfp4(Nvfp4Gpu),
    AffineW4(AffineW4Gpu),
    Int8(I8Gpu),
    Fp8(Fp8Gpu, Option<Bf16Gpu>),
}

impl DenseLinGpu {
    fn n(&self) -> usize {
        match self {
            Self::Bf16(w) => w.n,
            Self::Nvfp4(w) => w.n,
            Self::AffineW4(w) => w.n,
            Self::Int8(w) => w.n,
            Self::Fp8(w, _) => w.n,
        }
    }

    fn k(&self) -> usize {
        match self {
            Self::Bf16(w) => w.k,
            Self::Nvfp4(w) => w.k,
            Self::AffineW4(w) => w.k,
            Self::Int8(w) => w.k,
            Self::Fp8(w, _) => w.k,
        }
    }

    fn input_global(&self) -> Option<f32> {
        match self {
            Self::Bf16(_) | Self::Int8(_) | Self::Fp8(..) | Self::AffineW4(_) => None,
            Self::Nvfp4(w) => Some(w.input_global),
        }
    }
}

struct QuantIn {
    xq: Buffer,
    xs: Buffer,
    sel: Buffer,
    alpha_dummy: Buffer,
    xm: Option<Buffer>,
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

pub const MTP_SPECULATOR_LORA_TARGETS_ARE_THE_8_HEAD_PROJECTIONS_NORMS_EMBED_AND_LM_HEAD_ARE_NEVER_TARGETS:
    [&str; 8] = [
    "mtp.fc",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.mlp.down_proj",
];

#[derive(Debug)]
pub struct MtpSpeculatorDelta {
    pub module: String,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
    pub rank: usize,
    pub in_dim: usize,
    pub out_dim: usize,
}

#[derive(Debug)]
pub struct MtpSpeculatorDeltas {
    pub scaling: f64,
    pub entries: Vec<MtpSpeculatorDelta>,
}

pub fn mtp_speculator_deltas_from_adapter_dir(
    adapter_dir: &std::path::Path,
) -> Result<Option<MtpSpeculatorDeltas>> {
    use nv_train::{SPECULATOR_KIND_KEY, SPECULATOR_SUBDIR};
    let main_cfg_path = adapter_dir.join("adapter_config.json");
    let raw = std::fs::read_to_string(&main_cfg_path)
        .with_context(|| format!("read {}", main_cfg_path.display()))?;
    let main_cfg: serde_json::Value =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", main_cfg_path.display()))?;
    let sdir = adapter_dir.join(SPECULATOR_SUBDIR);
    let Some(kind) = main_cfg.get(SPECULATOR_KIND_KEY).and_then(|k| k.as_str()) else {
        ensure!(
            !sdir.exists(),
            "{} exists but the main adapter_config.json carries no {SPECULATOR_KIND_KEY}: the \
             kind is authoritative in the main config (the eagle3 writer never tags the subdir \
             config), so an untagged speculator/ is a malformed artifact, not a plain adapter",
            sdir.display()
        );
        return Ok(None);
    };
    ensure!(
        kind == nv_train::qwen38::QWEN38_MTP_SPECULATOR_KIND,
        "adapter {} carries speculator kind {kind:?} but this engine's drafter is the qwen3.8 \
         MTP head ({:?}); refusing the whole artifact instead of serving the target adapter \
         with an unmerged drafter",
        adapter_dir.display(),
        nv_train::qwen38::QWEN38_MTP_SPECULATOR_KIND
    );
    let scfg = nv_weights::lora_adapter::PeftConfig::from_dir(&sdir)
        .with_context(|| format!("read the speculator peft config in {}", sdir.display()))?;
    let st_path = sdir.join("adapter_model.safetensors");
    let tensors = candle_core::safetensors::load(&st_path, &candle_core::Device::Cpu)
        .with_context(|| format!("read {}", st_path.display()))?;
    let mut pending: std::collections::BTreeMap<
        String,
        (Option<candle_core::Tensor>, Option<candle_core::Tensor>),
    > = std::collections::BTreeMap::new();
    for (name, t) in tensors {
        let stripped = name.strip_prefix("base_model.model.").ok_or_else(|| {
            anyhow!("{st_path:?}: tensor {name} is not a peft base_model.model.* entry")
        })?;
        let (module, is_a) = if let Some(m) = stripped.strip_suffix(".lora_A.weight") {
            (m, true)
        } else if let Some(m) = stripped.strip_suffix(".lora_B.weight") {
            (m, false)
        } else {
            bail!("{st_path:?}: tensor {name} is neither a lora_A nor a lora_B weight");
        };
        ensure!(
            MTP_SPECULATOR_LORA_TARGETS_ARE_THE_8_HEAD_PROJECTIONS_NORMS_EMBED_AND_LM_HEAD_ARE_NEVER_TARGETS
                .contains(&module),
            "{st_path:?}: {module} is not an MTP-head LoRA target; norms load zero-centered as \
             stored+1 and embed/lm_head are shared with the trunk, so a delta naming any of them \
             must refuse the whole artifact, never be skipped"
        );
        let slot_pair = pending.entry(module.to_string()).or_default();
        let slot = if is_a {
            &mut slot_pair.0
        } else {
            &mut slot_pair.1
        };
        ensure!(slot.is_none(), "{st_path:?}: duplicate tensor {name}");
        *slot = Some(t);
    }
    ensure!(
        !pending.is_empty(),
        "{st_path:?}: the speculator artifact holds no LoRA pairs; the trainers refuse to save \
         an empty speculator, so this artifact is corrupt"
    );
    let mut entries = Vec::with_capacity(pending.len());
    for (module, (a, b)) in pending {
        let a = a.ok_or_else(|| anyhow!("{st_path:?}: {module} is missing lora_A"))?;
        let b = b.ok_or_else(|| anyhow!("{st_path:?}: {module} is missing lora_B"))?;
        let (a_r, in_dim) = a.dims2().with_context(|| format!("{module} lora_A dims"))?;
        let (out_dim, b_r) = b.dims2().with_context(|| format!("{module} lora_B dims"))?;
        ensure!(
            a_r == scfg.r && b_r == scfg.r,
            "{module}: lora_A is [{a_r}, {in_dim}] and lora_B is [{out_dim}, {b_r}] but the \
             speculator config says r={}",
            scfg.r
        );
        entries.push(MtpSpeculatorDelta {
            module,
            a: a.to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            b: b.to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
            rank: scfg.r,
            in_dim,
            out_dim,
        });
    }
    Ok(Some(MtpSpeculatorDeltas {
        scaling: scfg.scaling,
        entries,
    }))
}

fn dense_lin_bf16_for_merge<'a>(l: &'a HostDenseLin, module: &str) -> Result<&'a HostBf16Lin> {
    match l {
        HostDenseLin::Bf16(b) => Ok(b),
        _ => bail!(
            "speculator deltas merge into the host bf16 drafter weights before any load-time \
             quant arm, but {module} is already quantized"
        ),
    }
}

fn dense_lin_bf16_for_merge_mut<'a>(
    l: &'a mut HostDenseLin,
    module: &str,
) -> Result<&'a mut HostBf16Lin> {
    match l {
        HostDenseLin::Bf16(b) => Ok(b),
        _ => bail!(
            "speculator deltas merge into the host bf16 drafter weights before any load-time \
             quant arm, but {module} is already quantized"
        ),
    }
}

fn add_delta_skipping_zero_because_adding_0_to_minus_0_would_flip_the_sign_bit(
    w: &mut u16,
    dv: f32,
) {
    if dv != 0.0 {
        *w = bf16::from_f32(bf16::from_bits(*w).to_f32() + dv).to_bits();
    }
}

impl MtpHostWeights {
    fn speculator_target(&self, module: &str) -> Result<&HostBf16Lin> {
        match module {
            "mtp.fc" => Ok(&self.fc),
            "mtp.layers.0.self_attn.q_proj" => dense_lin_bf16_for_merge(&self.attn.q, module),
            "mtp.layers.0.self_attn.k_proj" => dense_lin_bf16_for_merge(&self.attn.k, module),
            "mtp.layers.0.self_attn.v_proj" => dense_lin_bf16_for_merge(&self.attn.v, module),
            "mtp.layers.0.self_attn.o_proj" => dense_lin_bf16_for_merge(&self.attn.o, module),
            "mtp.layers.0.mlp.gate_proj" => dense_lin_bf16_for_merge(&self.mlp.gate, module),
            "mtp.layers.0.mlp.up_proj" => dense_lin_bf16_for_merge(&self.mlp.up, module),
            "mtp.layers.0.mlp.down_proj" => dense_lin_bf16_for_merge(&self.mlp.down, module),
            _ => bail!("{module} is not an MTP-head LoRA target"),
        }
    }

    fn speculator_target_mut(&mut self, module: &str) -> Result<&mut HostBf16Lin> {
        match module {
            "mtp.fc" => Ok(&mut self.fc),
            "mtp.layers.0.self_attn.q_proj" => {
                dense_lin_bf16_for_merge_mut(&mut self.attn.q, module)
            }
            "mtp.layers.0.self_attn.k_proj" => {
                dense_lin_bf16_for_merge_mut(&mut self.attn.k, module)
            }
            "mtp.layers.0.self_attn.v_proj" => {
                dense_lin_bf16_for_merge_mut(&mut self.attn.v, module)
            }
            "mtp.layers.0.self_attn.o_proj" => {
                dense_lin_bf16_for_merge_mut(&mut self.attn.o, module)
            }
            "mtp.layers.0.mlp.gate_proj" => {
                dense_lin_bf16_for_merge_mut(&mut self.mlp.gate, module)
            }
            "mtp.layers.0.mlp.up_proj" => dense_lin_bf16_for_merge_mut(&mut self.mlp.up, module),
            "mtp.layers.0.mlp.down_proj" => {
                dense_lin_bf16_for_merge_mut(&mut self.mlp.down, module)
            }
            _ => bail!("{module} is not an MTP-head LoRA target"),
        }
    }

    pub fn merge_speculator(&mut self, d: &MtpSpeculatorDeltas) -> Result<usize> {
        ensure!(
            !d.entries.is_empty(),
            "an empty speculator delta set has nothing to merge and hides a corrupt artifact"
        );
        for e in &d.entries {
            let lin = self.speculator_target(&e.module)?;
            ensure!(
                lin.n == e.out_dim && lin.k == e.in_dim && e.rank > 0,
                "speculator delta for {} is [{}, {}] against a drafter weight of [{}, {}]: a \
                 mismatched-drafter artifact refuses whole with no partial merge (the gated \
                 q_proj row count comes from the drafter weight itself, never heads*head_dim)",
                e.module,
                e.out_dim,
                e.in_dim,
                lin.n,
                lin.k
            );
        }
        let scale = d.scaling as f32;
        for e in &d.entries {
            let a_t = candle_core::Tensor::from_slice(
                &e.a,
                (e.rank, e.in_dim),
                &candle_core::Device::Cpu,
            )?;
            let b_t = candle_core::Tensor::from_slice(
                &e.b,
                (e.out_dim, e.rank),
                &candle_core::Device::Cpu,
            )?;
            let delta = b_t.matmul(&a_t)?.flatten_all()?.to_vec1::<f32>()?;
            let lin = self.speculator_target_mut(&e.module)?;
            for (w, dv) in lin.w.iter_mut().zip(delta) {
                add_delta_skipping_zero_because_adding_0_to_minus_0_would_flip_the_sign_bit(
                    w,
                    dv * scale,
                );
            }
        }
        Ok(d.entries.len())
    }
}

struct MtpDraft {
    passes: Vec<Pass>,
    kv_end: usize,
    pos_buf: Buffer,
    fd_buf: Buffer,
    hid: Buffer,
    len: usize,
    round_base: Option<usize>,
    validated_kv: bool,
    validated_full: bool,
    _buffers: Vec<Buffer>,
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
    pf_tok: Option<Buffer>,
    pf_ck: Option<Buffer>,
    pf_splice: Option<Buffer>,
    pf_mask: Option<Buffer>,
    pf_attn_mk_fd: Vec<Buffer>,
    pf_vflash_row_fd: Vec<Buffer>,
    pf_attn_out_for_arm_probe: Option<Buffer>,
    vf_attn_out_for_arm_probe: Option<Buffer>,
    pf_vflash_gate_min_total: Option<u32>,
    pf_chunk_fd_and_start: Option<(Buffer, Buffer)>,
    pf_res: Option<Buffer>,
    pf_embed_end: usize,
    vf_passes: Vec<Pass>,
    vf_commit_passes: Vec<Pass>,
    vf_commit_validated: bool,
    vf_m: usize,
    vf_tok: Option<Buffer>,
    vf_ck: Option<Buffer>,
    vf_mask: Option<Buffer>,
    vf_attn_mk_fd: Vec<Buffer>,
    vf_vflash_row_fd: Vec<Buffer>,
    vf_vflash_gate_min_total: Option<u32>,
    vf_chunk_fd_and_start: Option<(Buffer, Buffer)>,
    head_start: usize,
    final_start: usize,
    verify: Option<Verify>,
    _buffers: Vec<Buffer>,
    tok_buf: Buffer,
    pos_buf: Buffer,
    fd_buf: Buffer,
    fd_base: FdParams,
    res: Buffer,
    res2: Buffer,
    final_x: Buffer,
    embed_gather_end: usize,
    mtp: Option<MtpDraft>,
    mtp_replay: bool,
    lora_active: bool,
    _lora_keep: Vec<Box<dyn std::any::Any + Send>>,
    token_out: Buffer,
    logits: Buffer,
    state_buffers: Vec<(Buffer, u64)>,
    recurrent_states: Vec<(Buffer, u64)>,
    resume_checkpoint: Vec<Buffer>,
    resume_checkpoint_pos: Option<usize>,
    grid_checkpoints: Vec<(Option<usize>, Vec<Buffer>)>,
    rope_row_buffers: Vec<(Buffer, Buffer)>,
    attn_kv_bf16_caches: Vec<(Buffer, Buffer)>,
    rope_std_host_cos_sin: (Vec<f32>, Vec<f32>),
    mrope_rows_installed: bool,
    vocab: usize,
    vram: VramReport,
    pending_cb: Option<(bool, CommandBuffer)>,
    tok_stage: Buffer,
    stage1_pairs: Vec<(usize, usize)>,
    tp_min_total: u32,
    last_step_stage1_engaged_tp: Option<bool>,
}

pub fn vram_report_enabled() -> bool {
    crate::wgpu_ledger::vram_report_var_enabled("NV_QWEN35_DENSE_WGPU_VRAM")
}

const QKV_GATED_ENV_FUSION_IS_BIT_IDENTICAL_BY_SHARED_LANE_FNS_BUT_UNMEASURED_SO_DEFAULT_OFF: &str =
    "NV_QWEN35_DENSE_WGPU_QKV_GATED";

fn qkv_gated_enabled() -> bool {
    env_on(QKV_GATED_ENV_FUSION_IS_BIT_IDENTICAL_BY_SHARED_LANE_FNS_BUT_UNMEASURED_SO_DEFAULT_OFF)
}

pub const DN_HEAD_WIDE_ENV_512_LANE_ROW_SPLIT_MEASURED_A_WALL_NULL_ON_THE_DGAP_LADDER_SO_THE_HEAD_IS_NOT_OCCUPANCY_BOUND_REASSOCIATES_SUMS_DEFAULT_OFF:
    &str = "NV_Q3D_DN_HEAD_WIDE";

pub const DN_HEAD_WIDE_REQUIRES_DK_128: usize = 128;

pub const DN_HEAD_WIDE_REQUIRES_DV_128: usize = 128;

fn dn_head_wide_enabled() -> bool {
    env_on(DN_HEAD_WIDE_ENV_512_LANE_ROW_SPLIT_MEASURED_A_WALL_NULL_ON_THE_DGAP_LADDER_SO_THE_HEAD_IS_NOT_OCCUPANCY_BOUND_REASSOCIATES_SUMS_DEFAULT_OFF)
}

pub const FUSE_MLP_GEMV_ENV_GATE_AND_UP_NVFP4_GEMVS_ARE_ONE_MROW2_2W_DISPATCH_PER_MLP_BIT_IDENTICAL_DEFAULT_OFF_SET_1_TO_ENGAGE:
    &str = "NV_Q3D_FUSE_MLP_GEMV";

fn fuse_mlp_gemv_enabled() -> bool {
    env_on(FUSE_MLP_GEMV_ENV_GATE_AND_UP_NVFP4_GEMVS_ARE_ONE_MROW2_2W_DISPATCH_PER_MLP_BIT_IDENTICAL_DEFAULT_OFF_SET_1_TO_ENGAGE)
}

pub const FUSE_KVW_ENV_KV_WRITE_AND_THE_FP8_KV_QUANT_PAIR_ARE_ONE_DISPATCH_PER_ATTN_LAYER_BIT_IDENTICAL_DEFAULT_OFF_SET_1_TO_ENGAGE:
    &str = "NV_Q3D_FUSE_KVW";

fn fuse_kvw_enabled() -> bool {
    env_on(FUSE_KVW_ENV_KV_WRITE_AND_THE_FP8_KV_QUANT_PAIR_ARE_ONE_DISPATCH_PER_ATTN_LAYER_BIT_IDENTICAL_DEFAULT_OFF_SET_1_TO_ENGAGE)
}

const W4A8_GEMV_ENTRY: &str = "gemv_nvfp4_w4a8_mrow2";

const W4A8_GEMV_LORA_FOLD_ENTRY: &str = "gemv_nvfp4_w4a8_mrow2_lora_fold";

const W4A8_RMSRES_ENTRY: &str = "rmsnorm_residual_bf16_rowquant_evenodd";

const W4A8_SILU_ENTRY: &str = "silu_mul_rowquant_evenodd";

const W4A8_SILU_ACT_ENTRY_STORES_BF16_ACT_FOR_THE_DOWN_LORA_SHRINK: &str =
    "silu_mul_rowquant_evenodd_act_bf16";

const W4A8_SILU_NOSMEM_ENTRY: &str = "silu_mul_rowquant_evenodd_nosmem";

const W4A8_SILU_ACT_NOSMEM_ENTRY: &str = "silu_mul_rowquant_evenodd_act_bf16_nosmem";

const W4A8_SILU_SMEM_ENTRY_DECLARED_WORKGROUP_BYTES_40KB_ACT_PLUS_4KB_REDUCE: u32 = 45056;

pub const W4A8_SILU_SMEM_ENV_UNSET_ROUTES_BY_ADAPTER_LIMIT_1_FORCES_SMEM_0_FORCES_NOSMEM: &str =
    "NV_Q3D_W4A8_SILU_SMEM";

pub const W4A8_SILU_NOSMEM_ROUTE_IS_THE_OVER_LIMIT_WORKGROUP_FIX: &str =
    "the smem silu producer stages 40KB of activations in workgroup memory, which beat the \
     redundant-scan variant 12us -> 5us on the adapter it was tuned on -- but M2 Ultra Metal \
     reports max_compute_workgroup_storage_size 32768 and, per the q3w_delta_scan_wy precedent \
     (e8e16ff28), EXECUTES the over-limit pipeline instead of refusing: here it degrades to \
     ~2.63 ms per dispatch, 72.6 pct of the whole w4a8 decode step (q38rival bracket, \
     q38rival-w4a8-profile-d256.log). The nosmem twins recompute the deterministic \
     silu-mul word in the emit loop instead of staging it, so outputs are bit-identical; the \
     route picks nosmem exactly when the declared 45056 bytes exceed the adapter limit, and \
     NV_Q3D_W4A8_SILU_SMEM pins either body for A/B.";

fn w4a8_silu_use_smem(max_workgroup_storage: u32) -> bool {
    match env::var(W4A8_SILU_SMEM_ENV_UNSET_ROUTES_BY_ADAPTER_LIMIT_1_FORCES_SMEM_0_FORCES_NOSMEM)
        .ok()
        .as_deref()
    {
        Some("1") => true,
        Some("0") => false,
        _ => {
            max_workgroup_storage
                >= W4A8_SILU_SMEM_ENTRY_DECLARED_WORKGROUP_BYTES_40KB_ACT_PLUS_4KB_REDUCE
        }
    }
}

pub const LORA_GEMV_FOLD_ATTN_DN_ENV_OPT_IN_MEASURED_PLUS_0_9MS_AT_D256_BECAUSE_2ROW_FP8_WORKGROUPS_RESTAGE_H_4X_DENSER_THAN_W4A8_AND_THE_SHRINK_BECOMES_A_CRITICAL_PATH_PREDECESSOR_OF_THE_WIDEST_GEMVS:
    &str = "NV_Q3D_LORA_GEMV_FOLD_ATTN_DN";

fn lora_gemv_fold_attn_dn_enabled() -> bool {
    env_on(
            LORA_GEMV_FOLD_ATTN_DN_ENV_OPT_IN_MEASURED_PLUS_0_9MS_AT_D256_BECAUSE_2ROW_FP8_WORKGROUPS_RESTAGE_H_4X_DENSER_THAN_W4A8_AND_THE_SHRINK_BECOMES_A_CRITICAL_PATH_PREDECESSOR_OF_THE_WIDEST_GEMVS,
        )
}

pod! {
    struct W48LoraFoldParams {
        rank: u32,
        n_tiles: u32,
        part_base: u32,
        b_base: u32,
        scale: f32,
        pad0: u32,
        pad1: u32,
        pad2: u32,
    }
}

struct LoraFoldSlice<'a> {
    fold: &'a LoraShrinkFold,
    site: &'a LoraSiteGpu,
    slice_id: usize,
}

impl LoraFoldSlice<'_> {
    fn params(&self, label: &str, w_n: usize, w_k: usize) -> W48LoraFoldParams {
        assert_eq!(
            self.site.widths[self.slice_id], w_n,
            "{label}: lora fold slice {} is {} wide, the gemv writes {w_n} rows",
            self.slice_id, self.site.widths[self.slice_id]
        );
        assert_eq!(
            self.site.k, w_k,
            "{label}: lora fold site k {} vs gemv k {w_k}",
            self.site.k
        );
        assert!(
            self.site.rank <= wk::lora::FUSED_MAX_RANK,
            "{label}: lora fold rank {} exceeds the kernel's shared-h array {}",
            self.site.rank,
            wk::lora::FUSED_MAX_RANK
        );
        W48LoraFoldParams {
            rank: self.site.rank as u32,
            n_tiles: self.fold.n_tiles as u32,
            part_base: (self.slice_id * self.fold.n_tiles * self.site.rank) as u32,
            b_base: (self.site.widths[..self.slice_id].iter().sum::<usize>() * self.site.rank)
                as u32,
            scale: 1.0,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }
}

const FP8_GEMV_LORA_FOLD_ENTRY: &str = "q3w_gemv_fp8_rowscale_lora_fold";

const DN_MERGED_GEMV_LORA_FOLD_ENTRY: &str = "q3w_gemv_dn_merged_fp8_qkv_fp8_z_bf16_ab_lora_fold";

pod! {
    struct DnMergedLoraFoldParams {
        rank: u32,
        n_tiles: u32,
        ab_a_rows: u32,
        pad0: u32,
        b_base: [u32; 4],
        scale: f32,
        pad1: u32,
        pad2: u32,
        pad3: u32,
    }
}

struct DnMergedLoraFold<'a> {
    fold: &'a LoraShrinkFold,
    site: &'a LoraSiteGpu,
}

impl DnMergedLoraFold<'_> {
    fn params(
        &self,
        label: &str,
        qkv_n: usize,
        z_n: usize,
        ab_n: usize,
        k: usize,
    ) -> DnMergedLoraFoldParams {
        let w = &self.site.widths;
        assert_eq!(
            w.len(),
            4,
            "{label}: the merged DN fold covers the qkv/z/a/b slice quartet, site has {}",
            w.len()
        );
        assert!(
            w[0] == qkv_n && w[1] == z_n && w[2] + w[3] == ab_n,
            "{label}: lora slice widths {w:?} disagree with merged gemv rows \
             [{qkv_n}, {z_n}, {ab_n}]"
        );
        assert!(
            w[2] == w[3] && w[2].is_multiple_of(2),
            "{label}: the ab pair-packed rows must split into two equal even halves, got {:?}",
            &w[2..]
        );
        assert_eq!(
            self.site.k, k,
            "{label}: lora fold site k {} vs merged gemv k {k}",
            self.site.k
        );
        assert!(
            self.site.rank <= wk::lora::FUSED_MAX_RANK,
            "{label}: lora fold rank {} exceeds the kernel's shared-h array {}",
            self.site.rank,
            wk::lora::FUSED_MAX_RANK
        );
        let mut b_base = [0u32; 4];
        let mut acc = 0usize;
        for (i, wi) in w.iter().enumerate() {
            b_base[i] = (acc * self.site.rank) as u32;
            acc += wi;
        }
        DnMergedLoraFoldParams {
            rank: self.site.rank as u32,
            n_tiles: self.fold.n_tiles as u32,
            ab_a_rows: w[2] as u32,
            pad0: 0,
            b_base,
            scale: 1.0,
            pad1: 0,
            pad2: 0,
            pad3: 0,
        }
    }
}

const W4A8_GEMV_ROWS_PER_GROUP: u32 = 8;

const W4A8_SILU_SMEM_WORDS_MIRRORS_THE_WGSL_40KB_ARRAY_SO_THE_ARM_CLAMPS_INTER_AT_20480: usize =
    10240;

fn mlp_w4a8_build_arm(ctx: &WgpuContext, cfg: &Qwen3_5DenseConfig) -> bool {
    let mut reasons: Vec<&str> = Vec::new();
    if !wk::gemv_nvfp4_v2::subgroup32_ok(ctx) {
        reasons.push("subgroup width 32 unavailable");
    }
    if !cfg.hidden_size.is_multiple_of(32) || !cfg.intermediate_size.is_multiple_of(32) {
        reasons.push("hidden/intermediate not 32-aligned for mrow2 block pairs");
    }
    if cfg.intermediate_size / 2
        > W4A8_SILU_SMEM_WORDS_MIRRORS_THE_WGSL_40KB_ARRAY_SO_THE_ARM_CLAMPS_INTER_AT_20480
    {
        reasons.push("intermediate_size exceeds the silu rowquant 40KB workgroup stage");
    }
    if w8_enabled(ctx, W8Scope::Ffn) {
        reasons.push("NV_Q3D_WGPU_W8 rewrites the FFN weights to int8");
    }
    if reasons.is_empty() {
        return true;
    }
    eprintln!(
        "[q3d-wgpu] the w4a8 MLP route clamped ({}); MLP stays on the nvfp4 route",
        reasons.join("; ")
    );
    false
}

fn mlp_w4a8_layer_arm(build_arm: bool, li: usize, m: &HostDenseMlp) -> bool {
    if !build_arm {
        return false;
    }
    if m.gate.is_nvfp4() && m.up.is_nvfp4() && m.down.is_nvfp4() {
        return true;
    }
    eprintln!(
        "[q3d-wgpu] layer {li} MLP is not all-nvfp4; layer stays off the w4a8 route"
    );
    false
}

pod! {
    struct W4a8ProducerParams {
        k_blocks: u32,
        words_per_row: u32,
        hidden: u32,
        eps: f32,
    }
}

struct W4a8Quant {
    xq: Buffer,
    xs: Buffer,
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
        self.resume_checkpoint_pos = None;
        for slot in &mut self.grid_checkpoints {
            slot.0 = None;
        }
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
        ensure!(
            section.iter().sum::<usize>() == rh,
            "mrope section {section:?} must tile the {rh} rotary half-frequencies of this graph"
        );
        let n = pos.len();
        ensure!(
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
                ensure!(
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
            ensure!(
                (0..self.max_seq as i64).contains(&p),
                "mrope continuation row {i} shifts to {p}, outside the {}-row rope table",
                self.max_seq
            );
            let p = p as usize;
            cos[i * rh..(i + 1) * rh].copy_from_slice(&std_cos[p * rh..(p + 1) * rh]);
            sin[i * rh..(i + 1) * rh].copy_from_slice(&std_sin[p * rh..(p + 1) * rh]);
        }
        for (cb, sb) in &self.rope_row_buffers {
            self.ctx.queue.write_buffer(cb, 0, cast_slice(&cos));
            self.ctx.queue.write_buffer(sb, 0, cast_slice(&sin));
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
            self.ctx.queue.write_buffer(cb, 0, cast_slice(std_cos));
            self.ctx.queue.write_buffer(sb, 0, cast_slice(std_sin));
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
            .ok_or_else(|| anyhow!("no attention layer {layer} with rope rows"))?;
        let rows = self.max_seq * self.rope_rot_half();
        Ok((
            dispatch::read_back::<f32>(self.ctx, cb, rows).map_err(|e| anyhow!("{e}"))?,
            dispatch::read_back::<f32>(self.ctx, sb, rows).map_err(|e| anyhow!("{e}"))?,
        ))
    }

    pub fn attn_kv_layer_count(&self) -> usize {
        self.attn_kv_bf16_caches.len()
    }

    pub fn attn_kv_row_f32s(&self) -> usize {
        self.config.num_key_value_heads * self.config.head_dim
    }

    pub fn debug_read_attn_kv_bf16_rows_for_test(
        &self,
        attn_layer: usize,
        start_pos: usize,
        n_pos: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let (kc, vc) = self.attn_kv_bf16_caches.get(attn_layer).ok_or_else(|| {
            anyhow!(
                "no attention layer {attn_layer}: this trunk carries {} full-attention layers",
                self.attn_kv_bf16_caches.len()
            )
        })?;
        ensure!(
            start_pos + n_pos <= self.max_seq,
            "kv rows {start_pos}..{} past max_seq {}",
            start_pos + n_pos,
            self.max_seq
        );
        let row_words = self.attn_kv_row_f32s() / 2;
        let unpack = |words: Vec<u32>| -> Vec<f32> {
            let mut out = Vec::with_capacity(words.len() * 2);
            for w in words {
                out.push(f32::from_bits((w & 0xffff) << 16));
                out.push(f32::from_bits((w >> 16) << 16));
            }
            out
        };
        let k =
            dispatch::read_back_at::<u32>(self.ctx, kc, start_pos * row_words, n_pos * row_words)
                .map_err(|e| anyhow!("{e}"))?;
        let v =
            dispatch::read_back_at::<u32>(self.ctx, vc, start_pos * row_words, n_pos * row_words)
                .map_err(|e| anyhow!("{e}"))?;
        Ok((unpack(k), unpack(v)))
    }

    fn debug_overwrite_one_attn_cache(
        &mut self,
        attn_layer: usize,
        start_pos: usize,
        rows: &[f32],
        k_not_v: bool,
    ) -> Result<()> {
        ensure!(
            !kv_fp8_enabled(),
            "the rotkv debug overwrite reaches decode only through the bf16 KV cache; with \
             NV_Q3D_KV_FP8 on, stage1 reads the fp8 mirror written before this overwrite and \
             the probe would silently measure nothing -- set NV_Q3D_KV_FP8=0 for mangle arms"
        );
        let row_f32s = self.attn_kv_row_f32s();
        ensure!(
            !rows.is_empty() && rows.len().is_multiple_of(row_f32s),
            "kv overwrite wants whole rows of {row_f32s} f32s; got {}",
            rows.len()
        );
        let n_pos = rows.len() / row_f32s;
        ensure!(
            start_pos + n_pos <= self.max_seq,
            "kv overwrite {start_pos}..{} past max_seq {}",
            start_pos + n_pos,
            self.max_seq
        );
        let (kc, vc) = self.attn_kv_bf16_caches.get(attn_layer).ok_or_else(|| {
            anyhow!(
                "no attention layer {attn_layer}: this trunk carries {} full-attention layers",
                self.attn_kv_bf16_caches.len()
            )
        })?;
        let packed: Vec<u32> = rows
            .chunks_exact(2)
            .map(|p| {
                let lo = bf16::from_f32(p[0]).to_bits() as u32;
                let hi = bf16::from_f32(p[1]).to_bits() as u32;
                lo | (hi << 16)
            })
            .collect();
        let byte_off = (start_pos * row_f32s / 2 * 4) as u64;
        let target = if k_not_v { kc } else { vc };
        self.ctx
            .queue
            .write_buffer(target, byte_off, cast_slice(&packed));
        Ok(())
    }

    pub fn debug_overwrite_attn_k_bf16_rows_for_test(
        &mut self,
        attn_layer: usize,
        start_pos: usize,
        rows: &[f32],
    ) -> Result<()> {
        self.debug_overwrite_one_attn_cache(attn_layer, start_pos, rows, true)
    }

    pub fn debug_overwrite_attn_v_bf16_rows_for_test(
        &mut self,
        attn_layer: usize,
        start_pos: usize,
        rows: &[f32],
    ) -> Result<()> {
        self.debug_overwrite_one_attn_cache(attn_layer, start_pos, rows, false)
    }

    pub fn fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(
        &mut self,
        pos: usize,
    ) -> Result<()> {
        ensure!(
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
        Self::build(config, WeightSource::Host(weights), max_seq, None)
    }

    pub fn new_with_lora(
        config: Qwen3_5DenseConfig,
        weights: &HostDenseWeights,
        max_seq: usize,
        lora: Option<&Q3dLora>,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Host(weights), max_seq, lora)
    }

    pub fn from_loader(
        config: Qwen3_5DenseConfig,
        weights: &WeightLoader,
        max_seq: usize,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Loader(weights), max_seq, None)
    }

    pub fn from_loader_with_lora(
        config: Qwen3_5DenseConfig,
        weights: &WeightLoader,
        max_seq: usize,
        lora: Option<&Q3dLora>,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Loader(weights), max_seq, lora)
    }

    fn build(
        config: Qwen3_5DenseConfig,
        src: WeightSource<'_>,
        max_seq: usize,
        lora: Option<&Q3dLora>,
    ) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow!("wgpu context: {e}"))?;
        let s = Sources::new(ctx);
        let cfg = &config;

        ensure!(max_seq > 0, "max_seq must be positive");
        ensure!(
            cfg.hidden_size.is_multiple_of(4),
            "hidden_size {} must be a multiple of 4",
            cfg.hidden_size
        );
        ensure!(
            cfg.intermediate_size.is_multiple_of(2),
            "intermediate_size {} must be even",
            cfg.intermediate_size
        );
        ensure!(
            cfg.head_dim <= MAX_HEAD_DIM && cfg.head_dim.is_multiple_of(2),
            "head_dim {} must be even and <= {MAX_HEAD_DIM}",
            cfg.head_dim
        );
        ensure!(
            cfg.linear_key_head_dim <= MAX_LIN_HEAD_DIM
                && cfg.linear_value_head_dim <= MAX_LIN_HEAD_DIM
                && cfg.linear_value_head_dim.is_multiple_of(2),
            "linear head dims must be even and <= {MAX_LIN_HEAD_DIM}"
        );
        ensure!(
            cfg.linear_num_value_heads
                .is_multiple_of(cfg.linear_num_key_heads),
            "linear value heads must be a multiple of key heads"
        );
        ensure!(
            cfg.num_attention_heads
                .is_multiple_of(cfg.num_key_value_heads),
            "attention heads must be a multiple of kv heads"
        );
        ensure!(
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
                STAGING_FLUSH_BYTES,
            ),
            passes: Vec::new(),
            pf_passes: Vec::new(),
            vf_passes: Vec::new(),
            vf_commit_passes: Vec::new(),
            pf_dst_vf: false,
        };

        if let Some(l) = lora {
            ensure!(
                l.layer_count() == cfg.num_hidden_layers,
                "lora carries {} layers for {} config layers",
                l.layer_count(),
                cfg.num_hidden_layers
            );
        }
        let lora_keep: std::cell::RefCell<Vec<Box<dyn std::any::Any + Send>>> =
            std::cell::RefCell::new(Vec::new());
        let lora_rt = match lora {
            Some(_) => Some(build_lora_pipelines(ctx)?),
            None => None,
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

        ensure!(
            cfg.linear_conv_kernel_dim >= 1 && cfg.linear_conv_kernel_dim <= 8,
            "linear_conv_kernel_dim {} out of 1..=8",
            cfg.linear_conv_kernel_dim
        );

        ensure!(
            !(pf_attn_mk_enabled()
                && env::var(PF_ATTN_TILED_ENV_1_ALSO_ROUTES_SMALL_M_GRAPHS_THROUGH_TILED_FLASH)
                    .ok()
                    .as_deref()
                    == Some("1")),
            "{} and {} both request the pf attention explicitly; the arms are alternatives, not \
             a ladder, so set at most one",
            PF_ATTN_MK_ENV_DEFAULT_OFF_UNTIL_THE_LADDER_AND_PPL_GATE_ARE_ON_RECORD,
            PF_ATTN_TILED_ENV_1_ALSO_ROUTES_SMALL_M_GRAPHS_THROUGH_TILED_FLASH
        );

        let tiled_routed = |m: usize| {
            pf_attn_tiled_enabled()
                && (!pf_small_m_legacy(m)
                    || env_arm_opt_in_eq_some_1_reread_every_call(
                        PF_ATTN_TILED_ENV_1_ALSO_ROUTES_SMALL_M_GRAPHS_THROUGH_TILED_FLASH,
                    ))
        };
        let qwen2_bias_arch = match &src {
            WeightSource::Host(h) => h.layers.iter().any(
                |l| matches!(&l.mixer, HostDenseMixer::Attn(a) if !a.qkv_bias_bf16.is_empty()),
            ),
            WeightSource::Loader(w) => [
                "model.language_model.layers.0",
                "language_model.model.layers.0",
                "model.layers.0",
            ]
            .iter()
            .any(|p| w.has(&format!("{p}.self_attn.q_proj.bias"))),
        };
        let flash_m_lane = qwen2_bias_arch
            && cfg.head_dim == 128
            && !cfg.attn_output_gate
            && !pf_attn_mk_enabled();
        let coop_adapter = pf_coop_tile(ctx).is_some();
        let projections_ride_coop = prefill_m() > 16
            && coop_adapter
            && match &src {
                WeightSource::Host(h) => {
                    host_pf_projections_ride_the_coop_route(h, pf_coop_tile(ctx))
                }
                WeightSource::Loader(w) => {
                    loader_pf_projections_ride_the_coop_route_probing_layer0_of_each_type_because_checkpoints_are_uniform(
                        w, cfg, pf_coop_tile(ctx),
                    )
                }
            };
        let build_pf_rows_graph = |b: &mut Builder, m_req: usize, vf: bool| -> Option<Pf> {
            let mut m = m_req;
            if m > 16 && !coop_adapter {
                eprintln!(
                    "[q3d-wgpu] prefill m={m} needs a 16x16 f16 coop adapter; clamping to 16"
                );
                m = 16;
            }
            if m > 16 && !projections_ride_coop {
                eprintln!(
                    "[q3d-wgpu] prefill m={m}: bf16-only projections cannot ride the \
                     coop route (the m-row bf16 gemm stops at 16); clamping to 16"
                );
                m = 16;
            }
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
                let coop_lane_open = match pf_coop_tile(ctx) {
                    Some(16) => true,
                    Some(8) => projections_ride_coop,
                    _ => false,
                };
                let use_coop = coop_lane_open
                    && (!small_m
                        || env_arm_opt_in_eq_some_1_reread_every_call(
                            PF_COOP_ENV_1_ALSO_ROUTES_SMALL_M_GRAPHS_THROUGH_COOP,
                        ));
                let use_tiled = tiled_routed(m) && !flash_m_lane;
                let coop = if use_coop {
                    let tile = pf_coop_tile(ctx).expect("use_coop just checked the adapter tile");
                    let m_pad = (m as u32).div_ceil(tile) * tile;
                    let tm = (m_pad / tile).clamp(1, 8);
                    let tn = (16 / tm).clamp(1, 4);
                    let sg = 2u32;
                    let m_alloc = m_pad.div_ceil(tile * tm) * (tile * tm);
                    let max_k = hidden.max(inter).max(value_dim).max(n_h * hd);
                    let max_n = hidden
                        .max(inter)
                        .max(conv_dim)
                        .max(value_dim)
                        .max(2 * n_v)
                        .max(n_h * src_stride_q)
                        .max(n_kv * hd);
                    Some(PfCoop {
                        tile,
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
                    let rows =
                        PF_ATTN_MK_ROWS_PER_DISPATCH_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS;
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
                    let q_f16_coop = match wk::pf_flash_tiled_coop::supported(ctx, hd, m, 0, 0) {
                        Ok(()) => Some(b.zeros("q3d-pf-at-cq-qf16", mu * (n_h * hd * 2) as u64)),
                        Err(why) => {
                            eprintln!(
                                "[q3d-wgpu] the coop tiled pf stage1 is unsupported ({why}); the \
                                 slotml tiled stage1 stays"
                            );
                            None
                        }
                    };
                    Some(PfAttnTiled {
                        q_f32: b.zeros("q3d-pf-at-td-qf32", mu * (n_h * hd * 4) as u64),
                        scratch: b.zeros(
                            "q3d-pf-at-td-scratch",
                            (n_h * m * splits * (hd + 2) * 4) as u64,
                        ),
                        fd: b.uni("q3d-pf-at-td-fd", FdParams::default()),
                        start: b.zeros("q3d-pf-at-td-start", 4),
                        q_f16_coop,
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
                let attn_vflash = match vf_attn_flash_arm() {
                    (arm, gate_min_total)
                        if small_m
                            && attn_mk.is_none()
                            && attn_tiled.is_none()
                            && kv_fp8_enabled() =>
                    {
                        let splits = wk::flash_decode::splits_for(max_seq);
                        let scratch_rows = if arm == VfAttnFlashArm::Smv2 { m } else { 1 };
                        Some(PfAttnVflash {
                            arm,
                            gate_min_total,
                            q_f32: b.zeros("q3d-pf-at-vfl-qf32", mu * (n_h * hd * 4) as u64),
                            scratch: b.zeros(
                                "q3d-pf-at-vfl-scratch",
                                (n_h * scratch_rows * splits * (hd + 2) * 4) as u64,
                            ),
                            row_fd: if arm == VfAttnFlashArm::RowLoop {
                                (0..m)
                                    .map(|_| b.uni("q3d-pf-at-vfl-fd", FdParams::default()))
                                    .collect()
                            } else {
                                Vec::new()
                            },
                        })
                    }
                    _ => None,
                };
                let scores_slab = attn_mk.is_none()
                    && attn_tiled.is_none()
                    && attn_vflash
                        .as_ref()
                        .is_none_or(|v| v.gate_min_total.is_some());
                Some(Pf {
                    m,
                    vf,
                    attn_mk,
                    attn_tiled,
                    attn_vflash,
                    attn_kvq_fd_start,
                    na: if m <= 16 { na } else { None },
                    coop,
                    gemm_src: if m <= 16 {
                        gemm_mk_source(m)
                    } else {
                        String::new()
                    },
                    gemm_entry: if m <= 16 {
                        gemm_mk_entry(m)
                    } else {
                        String::new()
                    },
                    gemm_i8_src: if m <= 16 {
                        sg_wrap(b.ctx.caps.subgroup, gemm_i8_mk_source(m))
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
                    d_wy_ckpt: if pf_scan_wy_v2_requested() && d_k == 128 && d_v == 128 {
                        let chunks = m.div_ceil(32) as u64;
                        Some((
                            b.zeros("q3d-pf-dn-wyck-cl", chunks * (n_v * 32 * 4) as u64),
                            b.zeros("q3d-pf-dn-wyck-s", chunks * (n_v * d_k * d_v * 4) as u64),
                        ))
                    } else {
                        None
                    },
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

        let affine_w4_checkpoint = match &src {
            WeightSource::Host(h) => host_carries_affine_w4(h),
            WeightSource::Loader(w) => {
                loader_is_mlx_affine_probing_layer0_because_mlx_quantizes_the_whole_trunk(w)
            }
        };
        let mut pfs: Vec<Pf> = Vec::new();
        let affine_chunked_opt_in = env_arm_opt_in_eq_some_1_reread_every_call(
            "NV_Q3D_AFFINE_CHUNKED_PF_V2_MEASURES_26TOK_0_97S_1613TOK_27_8MS_VS_STEPPED_1_10S_38_5MS",
        );
        let stepped_default_guards_the_qwen3_arch_basis_only = {
            let qwen2_bias = match &src {
                WeightSource::Host(h) => h.layers.iter().any(
                    |l| matches!(&l.mixer, HostDenseMixer::Attn(a) if !a.qkv_bias_bf16.is_empty()),
                ),
                WeightSource::Loader(w) => [
                    "model.language_model.layers.0",
                    "language_model.model.layers.0",
                    "model.layers.0",
                ]
                .iter()
                .any(|p| w.has(&format!("{p}.self_attn.q_proj.bias"))),
            };
            !qwen2_bias
        };
        if affine_w4_checkpoint
            && !affine_chunked_opt_in
            && stepped_default_guards_the_qwen3_arch_basis_only
        {
            eprintln!(
                "[q3d-wgpu] affine-w4 checkpoint: prefill rides the per-token step path (the \
                 published basis); the v2 chunked graph is faster (0.97s vs 1.10s at 26 prompt \
                 tokens, 27.8 vs 38.5 ms/token at 1613) and is opt-in via the env named above"
            );
        } else {
            if let Some(p) = build_pf_rows_graph(&mut b, prefill_m(), false) {
                pfs.push(p);
            }
            let vf_req = verify_m_dedicated_graph_rows_requested();
            if vf_req > 0 {
                let (main_m, main_coop) =
                    pfs.first().map_or((0, false), |p| (p.m, p.coop.is_some()));
                let vf_m = if vf_req <= SMALL_M_LEGACY_ROUTE_BOUND_VERIFY_GRAPHS_BUILD_M_LE_8 {
                    vf_req.max(2)
                } else if main_coop {
                    vf_req.div_ceil(VERIFY_GRAPH_COOP_ROWS_ROUND_TO_32_THE_TILED_FLASH_TILE_HEIGHT_AND_TWO_COOP_FRAGMENT_ROWS)
                    * VERIFY_GRAPH_COOP_ROWS_ROUND_TO_32_THE_TILED_FLASH_TILE_HEIGHT_AND_TWO_COOP_FRAGMENT_ROWS
                } else {
                    vf_req.clamp(2, 16)
                };
                let explicit = verify_m_env_is_explicit();
                if vf_m < main_m && (explicit || main_coop) {
                    if let Some(v) = build_pf_rows_graph(&mut b, vf_m, true) {
                        pfs.push(v);
                    }
                } else if explicit {
                    eprintln!(
                        "[q3d-wgpu] {}={vf_req} ignored: the dedicated verify graph (rows {vf_m}) \
                     must be strictly smaller than the prefill graph (m {main_m})",
                        VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM
                    );
                } else if main_m >= 2 {
                    eprintln!(
                    "[q3d-wgpu] default {}={vf_req} stays on the padded verify arm (prefill m \
                     {main_m}, coop {main_coop}): {VERIFY_M_DEFAULT_RECORDS_ONLY_BESIDE_A_COOP_PREFILL_GRAPH}",
                    VERIFY_M_ENV_DEFAULT_4_ROW_CHAIN_GRAPH_SET_0_FOR_THE_PF_M_PADDED_ARM
                );
                }
            }
        }
        let pfs = pfs;
        let res = b.zeros("q3d-res", (hidden_words * 4) as u64);
        let res2 = b.zeros("q3d-res2", (hidden_words * 4) as u64);
        let normed = b.zeros("q3d-normed", (hidden_words * 4) as u64);
        let mixed_out = b.zeros("q3d-mix", (hidden_words * 4) as u64);
        let mlp_out = b.zeros("q3d-mlpout", (hidden_words * 4) as u64);

        let embed = src.embed_host_arm(cfg)?;
        let chunk_rows = row_chunk(ctx, hidden);
        let mut off = 0usize;
        while off < vocab {
            let rows = chunk_rows.min(vocab - off);
            let buf = embed.upload_chunk(&mut b, &src, off, rows, hidden)?;
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
            for pf in &pfs {
                b.pf_dst_vf = pf.vf;
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

        for pf in &pfs {
            b.pf_dst_vf = pf.vf;
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

        let mut state_buffers: Vec<(Buffer, u64)> = Vec::new();
        let mut recurrent_states: Vec<(Buffer, u64)> = Vec::new();
        let mut rope_row_buffers: Vec<(Buffer, Buffer)> = Vec::new();
        let mut attn_kv_bf16_caches: Vec<(Buffer, Buffer)> = Vec::new();
        let w4a8_build_arm = mlp_w4a8_build_arm(b.ctx, cfg);

        for li in 0..cfg.num_hidden_layers {
            let layer = src.layer(cfg, li)?;
            let layer_lora = match (lora, &lora_rt) {
                (Some(l), Some(pl)) => Some(LoraSiteCtx {
                    pl,
                    layer: upload_q3d_lora_layer(ctx, &l.layers[li])?,
                    keep: &lora_keep,
                }),
                _ => None,
            };
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
                for pf in &pfs {
                    b.pf_dst_vf = pf.vf;
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
                    let delta_affine = match &src {
                        WeightSource::Loader(w) if affine_w4_checkpoint => {
                            load_delta_affine_w4(cfg, w, li)?
                        }
                        _ => DeltaAffineW4::default(),
                    };
                    build_delta(
                        &mut b,
                        &s,
                        cfg,
                        d,
                        &layer.delta_fp8,
                        &delta_affine,
                        &normed,
                        &mixed_out,
                        &mut state_buffers,
                        &pfs,
                        layer_lora.as_ref(),
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
                        &pfs,
                        &mut rope_row_buffers,
                        &mut attn_kv_bf16_caches,
                        true,
                        layer_lora.as_ref(),
                    )?;
                }
            }

            let post_w = b.upload_u32("q3d-post-ln", &pack_pairs(&layer.post_attn_ln));
            let normed_post = b.zeros("q3d-normed-post", (hidden_words * 4) as u64);
            let w4a8 = if mlp_w4a8_layer_arm(w4a8_build_arm, li, &layer.mlp) {
                Some(push_rmsres_post_w4a8(
                    &mut b,
                    &s,
                    hidden,
                    eps,
                    &mixed_out,
                    &res,
                    &post_w,
                    &normed_post,
                )?)
            } else {
                None
            };
            if w4a8.is_none() {
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
            }
            for pf in &pfs {
                b.pf_dst_vf = pf.vf;
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

            build_mlp(
                &mut b,
                &s,
                cfg,
                &layer.mlp,
                &normed_post,
                &mlp_out,
                &pfs,
                w4a8.as_ref(),
                layer_lora.as_ref(),
            )?;

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
                for pf in &pfs {
                    b.pf_dst_vf = pf.vf;
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

        let verify = match pfs.first() {
            Some(p) => {
                let chain = pfs.last().expect("pfs has a first element");
                let res2_rows = b.zeros("q3d-vf-res2", (p.m * hidden_words * 4) as u64);
                let mk_resadd = |b: &mut Builder, g: &Pf, label: &str| -> Result<Pass> {
                    let vp = b.uni(
                        "q3d-vf-resadd-p",
                        ResScaleParams {
                            n: (g.m * hidden) as u32,
                            n_words: (g.m * hidden_words) as u32,
                            scale: 1.0,
                            ..Default::default()
                        },
                    );
                    let grid = b.grid1((g.m * hidden_words) as u64, 256);
                    b.make(
                        label,
                        &s.resscale,
                        "residual_add_scale_bf16",
                        &[(0, &g.mlp_out), (1, &g.res), (2, &res2_rows), (3, &vp)],
                        grid,
                    )
                };
                let chunk_resadd = mk_resadd(&mut b, p, "q3d-vf-chunk-resadd")?;
                let resadd = mk_resadd(&mut b, chain, "q3d-vf-resadd")?;
                let tok = b.zeros("q3d-vf-tok", (chain.m * 4) as u64);
                Some(Verify {
                    rows: chain.m,
                    res2_rows,
                    resadd,
                    chunk_resadd,
                    tok,
                    row_logits: None,
                    rollback: Vec::new(),
                    validated: false,
                    pending: None,
                    pending_rode_vf: false,
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
        #[cfg(target_arch = "wasm32")]
        let lm_affine_view = src.lm_head_affine_view(cfg)?;
        #[cfg(target_arch = "wasm32")]
        let lm_affine: Option<HostAffineW4Lin> = None;
        #[cfg(not(target_arch = "wasm32"))]
        let lm_affine = src.lm_head_affine(cfg)?;
        let lm_fp8 = src.lm_head_fp8_packed(cfg)?;
        let lmh_requant = requant_fp8_group_enabled("lmhead") && lm_fp8.is_some();
        let lmh_nv4_aligned = vocab.is_multiple_of(128) && hidden.is_multiple_of(NVFP4_BLOCK * 4);
        if lmh_requant && !lmh_nv4_aligned {
            eprintln!(
                "[q3d-wgpu] {}=lmhead clamped (vocab {vocab} % 128 or hidden {hidden} % {} != 0); \
                 lm_head stays fp8",
                REQUANT_FP8_TO_NVFP4_ENV_MLP_ALWAYS_REQUANTS_A_COMMA_LIST_OR_ALL_ADDS_ATTN_DN_LMHEAD,
                NVFP4_BLOCK * 4
            );
        }
        #[cfg(target_arch = "wasm32")]
        let lm_view_handled = if let Some((lm_module, lm_gs)) = &lm_affine_view {
            let WeightSource::Loader(w) = &src else {
                bail!("affine lm_head view arm requires a loader source");
            };
            let w = *w;
            let limit = b
                .ctx
                .caps
                .max_storage_buffer_binding_size
                .clamp(1 << 20, 1u64 << 30);
            let packed_row_bytes = (hidden / 2) as u64;
            let rows_cap = ((limit / packed_row_bytes) as usize).min(vocab).max(1);
            let mut off = 0usize;
            while off < vocab {
                let rows = rows_cap.min(vocab - off);
                let g = upload_affine_w4_rows_from_loader_view_so_wasm32_never_hosts_a_full_copy(
                    &mut b,
                    "q3d-lmhead-aff",
                    w,
                    lm_module,
                    off,
                    rows,
                    hidden,
                    *lm_gs,
                )?;
                let raw = push_gemv_affine_w4_raw_words(&mut b, "q3d-lmhead-aff", &g, &final_x)?;
                push_affine_y_expand_f32(
                    &mut b,
                    "q3d-lmhead-aff-expand",
                    &raw,
                    rows,
                    off,
                    &logits,
                )?;
                off += rows;
            }
            true
        } else {
            false
        };
        #[cfg(not(target_arch = "wasm32"))]
        let lm_view_handled = false;

        if lm_view_handled {
        } else if let Some(l) = &lm_affine {
            ensure!(
                l.n == vocab && l.k == hidden,
                "affine lm_head is [{}, {}], want [{vocab}, {hidden}]",
                l.n,
                l.k
            );
            let limit = b
                .ctx
                .caps
                .max_storage_buffer_binding_size
                .clamp(1 << 20, 1u64 << 30);
            let packed_row_bytes = (hidden / 2) as u64;
            let rows_cap = ((limit / packed_row_bytes) as usize).min(vocab).max(1);
            let row_words = hidden / 8;
            let row_groups = hidden / l.gs;
            let mut off = 0usize;
            while off < vocab {
                let rows = rows_cap.min(vocab - off);
                let whole_tensor_borrow_skips_a_duplicate_of_the_largest_host_copy =
                    off == 0 && rows == vocab;
                let chunk;
                let cref = if whole_tensor_borrow_skips_a_duplicate_of_the_largest_host_copy {
                    l
                } else {
                    chunk = HostAffineW4Lin {
                        words: l.words[off * row_words..(off + rows) * row_words].to_vec(),
                        scales_bf16: l.scales_bf16[off * row_groups..(off + rows) * row_groups]
                            .to_vec(),
                        biases_bf16: l.biases_bf16[off * row_groups..(off + rows) * row_groups]
                            .to_vec(),
                        n: rows,
                        k: hidden,
                        gs: l.gs,
                    };
                    &chunk
                };
                let g = upload_affine_w4(&mut b, "q3d-lmhead-aff", cref, false);
                let raw = push_gemv_affine_w4_raw_words(&mut b, "q3d-lmhead-aff", &g, &final_x)?;
                push_affine_y_expand_f32(
                    &mut b,
                    "q3d-lmhead-aff-expand",
                    &raw,
                    rows,
                    off,
                    &logits,
                )?;
                off += rows;
            }
        } else if lmh_requant && lmh_nv4_aligned {
            let (packed, scales) = lm_fp8.as_ref().expect("lmh_requant checked lm_fp8");
            let row_words = hidden / 4;
            ensure!(
                packed.len() == vocab * row_words,
                "fp8 lm_head has {} words, want {}",
                packed.len(),
                vocab * row_words
            );
            let max_bind = b.ctx.caps.max_storage_buffer_binding_size as usize;
            let rows_cap = ((max_bind / (hidden / 2)) & !127usize).min(vocab);
            ensure!(
                rows_cap >= 128,
                "q3d-lmhead-nv4: max_storage_buffer_binding_size {max_bind} holds fewer than \
                 128 nvfp4 rows of k={hidden}"
            );
            let mut lm_chunks: Vec<(DenseLinGpu, usize, usize)> = Vec::new();
            let mut off = 0usize;
            while off < vocab {
                let rows = rows_cap.min(vocab - off);
                let host = requant_fp8_rowscale_to_nvfp4(
                    &packed[off * row_words..(off + rows) * row_words],
                    &scales[off..off + rows],
                    rows,
                    hidden,
                )?;
                lm_chunks.push((
                    DenseLinGpu::Nvfp4(upload_nvfp4(&mut b, "q3d-lmhead-nv4", &host)),
                    off,
                    rows,
                ));
                off += rows;
            }
            let consumers: Vec<&DenseLinGpu> = lm_chunks.iter().map(|(w, _, _)| w).collect();
            let lq = quant_for(&mut b, &s, "q3d-lmhead-quant", &final_x, hidden, &consumers)?
                .ok_or_else(|| {
                    anyhow!("q3d-lmhead-quant: nvfp4 lm_head chunks must yield a quant plan")
                })?;
            for (wchunk, coff, rows) in &lm_chunks {
                let DenseLinGpu::Nvfp4(wv) = wchunk else {
                    bail!("q3d-lmhead-nv4: chunks are uploaded nvfp4")
                };
                let y_bf16 = b.zeros("q3d-lmhead-ybf16", (*rows * 2) as u64);
                push_gemv_nvfp4(&mut b, &s, "q3d-lmhead-nv4", wv, &lq, &y_bf16)?;
                let xp = b.uni(
                    "q3d-lmhead-exp-p",
                    LmheadExpandParams {
                        n_words: (*rows / 2) as u32,
                        y_off_elems: *coff as u32,
                        ..Default::default()
                    },
                );
                let egrid = b.grid1((*rows / 2) as u64, 256);
                b.push(
                    "q3d-lmhead-expand",
                    LMHEAD_EXPAND_WGSL_EXISTS_BECAUSE_EVERY_SHIPPING_NVFP4_GEMV_STORES_BF16_Y_AND_ARGMAX_READS_F32,
                    "q3w_lmhead_bf16_expand_f32",
                    &[(0, &y_bf16), (1, &logits), (2, &xp)],
                    egrid,
                )?;
            }
        } else if let Some((packed, scales)) = lm_fp8 {
            let row_words = hidden / 4;
            ensure!(
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
            ensure!(
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
            vf_passes,
            vf_commit_passes,
            core,
            ..
        } = b;
        let buffers = core.buffers;

        let (pf_passes, pf_m, pf_tok, pf_ck, pf_splice, pf_mask, pf_attn_mk_fd, pf_tiled, pf_res) =
            match pfs.first() {
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
                None => (
                    Vec::new(),
                    0,
                    None,
                    None,
                    None,
                    None,
                    Vec::new(),
                    None,
                    None,
                ),
            };
        let pf_vflash_row_fd = pfs
            .first()
            .and_then(|p| p.attn_vflash.as_ref())
            .map(|v| v.row_fd.clone())
            .unwrap_or_default();
        let vf_vflash_row_fd = pfs
            .get(1)
            .and_then(|p| p.attn_vflash.as_ref())
            .map(|v| v.row_fd.clone())
            .unwrap_or_default();
        let pf_attn_out_for_arm_probe = pfs.first().map(|p| p.a_attn.clone());
        let vf_attn_out_for_arm_probe = pfs.get(1).map(|p| p.a_attn.clone());
        let pf_vflash_gate_min_total = pfs
            .first()
            .and_then(|p| p.attn_vflash.as_ref())
            .and_then(|v| v.gate_min_total);
        let vf_vflash_gate_min_total = pfs
            .get(1)
            .and_then(|p| p.attn_vflash.as_ref())
            .and_then(|v| v.gate_min_total);
        let (vf_passes, vf_m, vf_tok, vf_ck, vf_mask, vf_attn_mk_fd, vf_tiled) = match pfs.get(1) {
            Some(p) => (
                vf_passes,
                p.m,
                Some(p.tok.clone()),
                Some(p.ck.clone()),
                Some(p.mask.clone()),
                p.attn_mk.as_ref().map(|k| k.fd.clone()).unwrap_or_default(),
                p.attn_tiled
                    .as_ref()
                    .map(|t| (t.fd.clone(), t.start.clone()))
                    .or_else(|| p.attn_kvq_fd_start.clone()),
            ),
            None => (Vec::new(), 0, None, None, None, Vec::new(), None),
        };
        assert!(
            pfs.get(1).is_none() || !vf_passes.is_empty(),
            "a recorded dedicated verify graph must have recorded its passes into the vf list; \
             an empty list here means a push_pf site ran without pf_dst_vf set and leaked verify \
             passes into the prefill graph"
        );
        for (gate, list, name) in [
            (pf_vflash_gate_min_total, &pf_passes, "pf"),
            (vf_vflash_gate_min_total, &vf_passes, "vf"),
        ] {
            if gate.is_none() {
                continue;
            }
            let slab = list
                .iter()
                .filter(|p| p.label.starts_with("q3d-pf-at-decode"))
                .count();
            let vfl = list
                .iter()
                .filter(|p| p.label.starts_with("q3d-pf-at-vfl-flash1"))
                .count();
            assert!(
                slab > 0 && vfl > 0,
                "{name} graph is depth-gated but recorded {slab} slab / {vfl} loop stage1 \
                 passes; a one-armed gate silently runs that arm at every depth, so the A/B it \
                 was set for measures nothing: \
                 {VF_FLASH_GATE_SELECTS_AT_ENCODE_BECAUSE_VERIFY_CBS_ARE_REENCODED_EVERY_ROUND}"
            );
        }

        let tok_stage = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("q3d-token-stage"),
            size: 4,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rope_std_host_cos_sin =
            rope_tables(config.rotary_dim().max(2), config.rope_theta, max_seq);

        let stage1_pairs: Vec<(usize, usize)> = passes
            .windows(2)
            .enumerate()
            .filter(|(_, w)| is_fold_stage1_label(&w[0].label) && is_tp_stage1_label(&w[1].label))
            .map(|(i, _)| (i, i + 1))
            .collect();
        let tp_count = passes
            .iter()
            .filter(|p| is_tp_stage1_label(&p.label))
            .count();
        assert_eq!(
            stage1_pairs.len(),
            tp_count,
            "every tp stage1 in the step list must sit directly after its position-major twin; \
             an unpaired tp arm cannot be depth-gated and would run every step"
        );
        if let Some((&(first_pm, _), &(_, last_tp))) = stage1_pairs.first().zip(stage1_pairs.last())
        {
            assert!(
                first_pm >= embed_gather_end && last_tp < final_start,
                "stage1 pairs must lie inside [embed_gather_end, final_start): the mtp and \
                 verify-chain replays slice [..embed_gather_end] and [final_start..] and never \
                 filter arms"
            );
        }
        let tp_min_total = kv_fp8_tp_min_total();
        if tp_min_total > 0 && stage1_pairs.is_empty() {
            eprintln!(
                "[q3d-wgpu] {}={tp_min_total} has no effect: no dual stage1 pairs are recorded \
                 (tp disabled or not capable), so the single recorded arm runs every step",
                KV_FP8_TP_MIN_TOTAL_ENV_TP_RUNS_ONLY_AT_TOTAL_AT_OR_ABOVE_THIS_PLAIN_INTEGER
            );
        }

        requant_boot_report();

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
            pf_vflash_row_fd,
            pf_attn_out_for_arm_probe,
            vf_attn_out_for_arm_probe,
            pf_vflash_gate_min_total,
            pf_chunk_fd_and_start: pf_tiled,
            pf_res,
            pf_embed_end,
            vf_passes,
            vf_commit_passes,
            vf_commit_validated: false,
            vf_m,
            vf_tok,
            vf_ck,
            vf_mask,
            vf_attn_mk_fd,
            vf_vflash_row_fd,
            vf_vflash_gate_min_total,
            vf_chunk_fd_and_start: vf_tiled,
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
            lora_active: lora.is_some(),
            _lora_keep: lora_keep.into_inner(),
            token_out,
            logits,
            state_buffers,
            recurrent_states,
            resume_checkpoint: Vec::new(),
            resume_checkpoint_pos: None,
            grid_checkpoints: Vec::new(),
            rope_row_buffers,
            attn_kv_bf16_caches,
            rope_std_host_cos_sin,
            mrope_rows_installed: false,
            vocab,
            vram,
            pending_cb: None,
            tok_stage,
            stage1_pairs,
            tp_min_total,
            last_step_stage1_engaged_tp: None,
        })
    }

    fn stage1_tp_engaged_at_total(&self, total: u32) -> bool {
        total >= self.tp_min_total
    }

    fn stage1_skip_idxs(&self, engage_tp: bool) -> Vec<usize> {
        self.stage1_pairs
            .iter()
            .map(|&(pm, tp)| if engage_tp { pm } else { tp })
            .collect()
    }

    fn profiled(&self) -> bool {
        dispatch::profile::enabled() && self.ctx.caps.timestamp_query
    }

    fn encode_step_cb(&self, full: bool, engage_tp: bool) -> CommandBuffer {
        let skip = self.stage1_skip_idxs(engage_tp);
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let passes = if full {
                &self.passes[..]
            } else {
                &self.passes[..self.head_start]
            };
            let mut pass = enc.begin_compute_pass(&Default::default());
            for (i, p) in passes.iter().enumerate() {
                if skip.binary_search(&i).is_ok() {
                    continue;
                }
                dispatch_pass!(pass, p);
            }
        }
        if full {
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
        self.ctx.poll_blocking().map_err(|e| anyhow!("{e}"))?;
        rx.recv()
            .map_err(|e| anyhow!("token stage map callback: {e}"))?
            .map_err(|e| anyhow!("token stage map: {e}"))?;
        let t = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow!("token stage mapped range: {e}"))?;
            cast_slice::<u8, u32>(&view)[0]
        };
        self.tok_stage.unmap();
        Ok(t)
    }

    fn read_token_out(&self) -> Result<u32> {
        dispatch::drive_blocking(self.read_token_out_arm(dispatch::ReadbackArm::Blocking))
    }

    async fn read_token_out_arm(&self, arm: dispatch::ReadbackArm) -> Result<u32> {
        if arm == dispatch::ReadbackArm::Blocking && !self.profiled() {
            return self.read_token_stage();
        }
        let t: Vec<u32> = dispatch::read_back_arm(self.ctx, &self.token_out, 1, arm)
            .await
            .map_err(|e| anyhow!("{e}"))?;
        Ok(t[0])
    }

    fn step_inner(&mut self, token: u32, full: bool) -> Result<()> {
        ensure!((token as usize) < self.vocab, "token {token} out of vocab");
        ensure!(
            self.pos < self.max_seq,
            "kv cache full at {} (max_seq {})",
            self.pos,
            self.max_seq
        );
        self.ctx
            .queue
            .write_buffer(&self.tok_buf, 0, bytes_of(&(token as i32)));
        self.ctx
            .queue
            .write_buffer(&self.pos_buf, 0, bytes_of(&(self.pos as i32)));
        let mut fd = self.fd_base;
        fd.total = (self.pos + 1) as u32;
        self.ctx.queue.write_buffer(&self.fd_buf, 0, bytes_of(&fd));

        let need_scope = if full {
            !self.validated
        } else {
            !self.validated && !self.prefix_validated
        };
        let scope = if need_scope {
            Some(self.ctx.device.push_error_scope(ErrorFilter::Validation))
        } else {
            None
        };
        let engage_tp = self.stage1_tp_engaged_at_total(fd.total);
        if self.profiled() {
            let skip = self.stage1_skip_idxs(engage_tp);
            let passes = if full {
                &self.passes[..]
            } else {
                &self.passes[..self.head_start]
            };
            let raw: Vec<(&ComputePipeline, &BindGroup, (u32, u32, u32))> = passes
                .iter()
                .enumerate()
                .filter(|(i, _)| skip.binary_search(i).is_err())
                .map(|(_, p)| (p.pipeline.as_ref(), &p.bind, p.grid))
                .collect();
            let labels: Vec<String> = passes
                .iter()
                .enumerate()
                .filter(|(i, _)| skip.binary_search(i).is_err())
                .map(|(_, p)| p.label.clone())
                .collect();
            dispatch::submit_profiled_slices(self.ctx, &raw, &labels)
                .map_err(|e| anyhow!("profiled submit: {e}"))?;
        } else {
            let cb = if full {
                match self.pending_cb.take() {
                    Some((tp, cb)) if tp == engage_tp => cb,
                    _ => self.encode_step_cb(true, engage_tp),
                }
            } else {
                self.encode_step_cb(false, engage_tp)
            };
            self.ctx.queue.submit([cb]);
            if full {
                let next_tp = self.stage1_tp_engaged_at_total((self.pos + 2) as u32);
                self.pending_cb = Some((next_tp, self.encode_step_cb(true, next_tp)));
            }
        }
        self.last_step_stage1_engaged_tp = (!self.stage1_pairs.is_empty()).then_some(engage_tp);
        if let Some(scope) = scope {
            if let Some(e) = dispatch::pop_scope(scope) {
                bail!("qwen3_5_dense_wgpu decode step validation: {e}");
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

    pub async fn decode_step_pumped(&mut self, token: u32) -> Result<u32> {
        self.step_inner(token, true)?;
        self.read_token_out_arm(dispatch::ReadbackArm::Pumped).await
    }

    pub fn decode_step_logits(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        dispatch::drive_blocking(
            self.decode_step_logits_arm(token, dispatch::ReadbackArm::Blocking),
        )
    }

    pub async fn decode_step_logits_pumped(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        self.decode_step_logits_arm(token, dispatch::ReadbackArm::Pumped)
            .await
    }

    async fn decode_step_logits_arm(
        &mut self,
        token: u32,
        arm: dispatch::ReadbackArm,
    ) -> Result<(u32, Vec<f32>)> {
        self.step_inner(token, true)?;
        let t = self.read_token_out_arm(arm).await?;
        let l: Vec<f32> = dispatch::read_back_arm(self.ctx, &self.logits, self.vocab, arm)
            .await
            .map_err(|e| anyhow!("{e}"))?;
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
        ensure!(
            m >= 2 && chunk.len() == m,
            "needs the pf graph and a full chunk"
        );
        let hidden = self.config.hidden_size;
        let hidden_words = hidden / 2;
        let splice_buf = self.pf_splice.as_ref().expect("pf splice buffer");
        let mut mask = vec![0u32; m];
        for (rel, row) in splices {
            ensure!(*rel < m && row.len() == hidden, "bad debug splice row");
            mask[*rel] = 1;
            self.ctx.queue.write_buffer(
                splice_buf,
                (*rel * hidden_words * 4) as u64,
                cast_slice(&pack_pairs(row)),
            );
        }
        self.ctx.queue.write_buffer(
            self.pf_mask.as_ref().expect("pf mask buffer"),
            0,
            cast_slice(&mask),
        );
        let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
        self.ctx.queue.write_buffer(
            self.pf_tok.as_ref().expect("pf tok buffer"),
            0,
            cast_slice(&ids),
        );
        submit_passes!(self.ctx, &self.pf_passes[..self.pf_embed_end]);
        let res = self.pf_res.as_ref().expect("pf res buffer");
        let words: Vec<u32> = dispatch::read_back::<u32>(self.ctx, res, m * hidden_words)
            .map_err(|e| anyhow!("{e}"))?;
        let mut out = Vec::with_capacity(m * hidden);
        for w in words {
            out.push((w & 0xffff) as u16);
            out.push((w >> 16) as u16);
        }
        Ok(out)
    }

    pub fn debug_step_pass_labels_for_test(&self) -> Vec<String> {
        self.passes.iter().map(|p| p.label.clone()).collect()
    }

    pub fn debug_step_pass_labels_at_total_for_test(&self, total: u32) -> Vec<String> {
        let skip = self.stage1_skip_idxs(self.stage1_tp_engaged_at_total(total));
        self.passes
            .iter()
            .enumerate()
            .filter(|(i, _)| skip.binary_search(i).is_err())
            .map(|(_, p)| p.label.clone())
            .collect()
    }

    pub fn debug_last_step_stage1_engaged_tp_for_test(&self) -> Option<bool> {
        self.last_step_stage1_engaged_tp
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

    fn write_row_group_fds(&self, fds: &[Buffer], rows: usize, live: usize, base: usize) {
        for (g, buf) in fds.iter().enumerate() {
            let start = g * rows;
            let mr = live.saturating_sub(start).min(rows);
            let mut fd = self.fd_base;
            fd.m_rows = mr as u32;
            fd.total = if mr == 0 {
                0
            } else {
                (base + start + mr) as u32
            };
            self.ctx.queue.write_buffer(buf, 0, bytes_of(&fd));
        }
    }

    fn write_pf_attn_mk_fd(&self, fds: &[Buffer], live: usize, base: usize) {
        self.write_row_group_fds(
            fds,
            PF_ATTN_MK_ROWS_PER_DISPATCH_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS,
            live,
            base,
        );
    }

    fn write_pf_chunk_fd_whole_view_and_start_for_tiled_flash_and_fp8_chunk_quantize(
        &self,
        pair: Option<&(Buffer, Buffer)>,
        m: usize,
        base: usize,
    ) {
        if let Some((fd_buf, start_buf)) = pair {
            let mut fd = self.fd_base;
            fd.m_rows = m as u32;
            fd.total = (base + m) as u32;
            self.ctx.queue.write_buffer(fd_buf, 0, bytes_of(&fd));
            self.ctx
                .queue
                .write_buffer(start_buf, 0, bytes_of(&(base as i32)));
        }
    }

    fn prefill_chunk_masked(
        &mut self,
        chunk: &[u32],
        live: usize,
        splices: &[ChunkRowSplice],
    ) -> Result<()> {
        let m = self.pf_m;
        ensure!(chunk.len() == m && (1..=m).contains(&live), "bad chunk");
        ensure!(
            splices.is_empty() || self.mtp.is_none(),
            "the mtp drafter has no image-splice row convention: its KV row embeds the token id, \
             not the spliced media row; serve mm without the mtp head attached"
        );
        ensure!(
            self.pos + live <= self.max_seq,
            "kv cache full at {} + {live} (max_seq {})",
            self.pos,
            self.max_seq
        );
        for &t in chunk {
            ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        let hidden_words = self.config.hidden_size / 2;
        let mut mask = vec![0u32; m];
        for sp in splices {
            ensure!(
                sp.rel_pos < live,
                "splice rel_pos {} >= live {live}",
                sp.rel_pos
            );
            ensure!(
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
                cast_slice(sp.row_words),
            );
        }
        let mask_buf = self
            .pf_mask
            .as_ref()
            .expect("prefill chunk without pf list");
        self.ctx.queue.write_buffer(mask_buf, 0, cast_slice(&mask));
        let tok = self.pf_tok.as_ref().expect("prefill chunk without pf list");
        let ck = self.pf_ck.as_ref().expect("prefill chunk without pf list");
        let ids: Vec<i32> = chunk.iter().map(|&t| t as i32).collect();
        self.ctx.queue.write_buffer(tok, 0, cast_slice(&ids));
        self.ctx.queue.write_buffer(
            ck,
            0,
            bytes_of(&CkParams {
                m_live: live as u32,
                base: self.pos as u32,
                pad0: 0,
                pad1: 0,
            }),
        );
        self.write_pf_attn_mk_fd(&self.pf_attn_mk_fd, live, self.pos);
        self.write_row_group_fds(&self.pf_vflash_row_fd, 1, live, self.pos);
        self.write_pf_chunk_fd_whole_view_and_start_for_tiled_flash_and_fp8_chunk_quantize(
            self.pf_chunk_fd_and_start.as_ref(),
            self.pf_m,
            self.pos,
        );
        let scope = if self.pf_validated {
            None
        } else {
            Some(self.ctx.device.push_error_scope(ErrorFilter::Validation))
        };
        let gate_total = (self.pos + live) as u32;
        if dispatch::profile::enabled() && self.ctx.caps.timestamp_query {
            let kept: Vec<&Pass> = self
                .pf_passes
                .iter()
                .filter(|p| vfl_gate_keeps(&p.label, self.pf_vflash_gate_min_total, gate_total))
                .collect();
            let raw: Vec<(&ComputePipeline, &BindGroup, (u32, u32, u32))> = kept
                .iter()
                .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
                .collect();
            let labels: Vec<String> = kept.iter().map(|p| p.label.clone()).collect();
            dispatch::submit_profiled_slices(self.ctx, &raw, &labels)
                .map_err(|e| anyhow!("profiled prefill submit: {e}"))?;
        } else {
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in &self.pf_passes {
                    if !vfl_gate_keeps(&p.label, self.pf_vflash_gate_min_total, gate_total) {
                        continue;
                    }
                    dispatch_pass!(pass, p);
                }
            }
            self.ctx.queue.submit([enc.finish()]);
        }
        if let Some(scope) = scope {
            if let Some(e) = dispatch::pop_scope(scope) {
                bail!("qwen3_5_dense_wgpu prefill chunk validation: {e}");
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
        ensure!(
            m >= 2,
            "image-row splice prefill requires the chunked prefill graph and a live pf pass list"
        );
        let hidden = self.config.hidden_size;
        let mut prev_end = 0usize;
        let mut packed: Vec<Vec<u32>> = Vec::with_capacity(splices.len());
        for sp in splices {
            ensure!(
                sp.rows_bf16.len() % hidden == 0,
                "splice rows_bf16 len {} not a multiple of hidden {hidden}",
                sp.rows_bf16.len()
            );
            let n_slots = sp.rows_bf16.len() / hidden;
            ensure!(
                sp.position >= prev_end,
                "image splices must be sorted and non-overlapping"
            );
            ensure!(
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
            ensure!(
                self.pos + live <= self.max_seq,
                "kv cache full at {} + {live} (max_seq {})",
                self.pos,
                self.max_seq
            );
            let (chunk, rows) = spliced_chunk(tokens, splices, &packed, hidden, m, done, live);
            self.prefill_chunk_masked(&chunk, live, &rows)?;
            done += live;
        }
        Ok(done)
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32> {
        dispatch::drive_blocking(self.prefill_arm(tokens, dispatch::ReadbackArm::Blocking))
    }

    pub async fn prefill_pumped(&mut self, tokens: &[u32]) -> Result<u32> {
        self.prefill_arm(tokens, dispatch::ReadbackArm::Pumped)
            .await
    }

    async fn prefill_arm(&mut self, tokens: &[u32], arm: dispatch::ReadbackArm) -> Result<u32> {
        ensure!(!tokens.is_empty(), "prefill needs at least one token");
        let (last, rest) = tokens.split_last().expect("non-empty");
        let done = self.prefill_tokens(rest)?;
        for t in &rest[done..] {
            self.prefill_step(*t)?;
        }
        self.step_inner(*last, true)?;
        self.read_token_out_arm(arm).await
    }

    pub fn verify_max_rows(&self) -> usize {
        self.verify.as_ref().map_or(0, |v| v.rows)
    }

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        Ok(self
            .verify_chain_inner(batch, VerifyTail::ArgmaxRows, false)?
            .0)
    }

    pub fn verify_chain_logits(&mut self, batch: &[u32]) -> Result<(Vec<u32>, Vec<f32>)> {
        self.verify_chain_inner(batch, VerifyTail::ArgmaxAndLogitRows, false)
    }

    pub fn debug_verify_chain_logits_via_the_padded_prefill_graph_for_test(
        &mut self,
        batch: &[u32],
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        self.verify_chain_inner(batch, VerifyTail::ArgmaxAndLogitRows, true)
    }

    pub fn verify_chain_rides_dedicated_graph(&self) -> bool {
        !self.vf_passes.is_empty()
    }

    pub const ATTN_OUT_IS_THE_ARM_PROBE_BECAUSE_THE_GATED_RESIDUAL_ROUNDS_ARM_DRIFT_OUT_OF_TINY_LOGITS:
        &'static str =
        "the attention output gate scales a_attn to well under a percent of the residual on the \
         tiny synthetic models, so the bf16 residual chain rounds any attention-arm change out \
         of the returned logits and a logits oracle passes with the arm switch broken; the \
         verify attention-arm contracts therefore read the M-row graph's a_attn rows, the \
         buffer the slab pass and both vflash arms write directly";

    pub fn debug_verify_attn_out_rows_for_test(&self) -> Result<Vec<f32>> {
        let buf = if !self.vf_passes.is_empty() {
            self.vf_attn_out_for_arm_probe.as_ref()
        } else {
            self.pf_attn_out_for_arm_probe.as_ref()
        }
        .ok_or_else(|| anyhow!("no M-row graph recorded, so no verify attention output"))?;
        dispatch::read_back(self.ctx, buf, (buf.size() / 4) as usize).map_err(|e| anyhow!("{e}"))
    }

    pub fn verify_chain_commit_only(&mut self, batch: &[u32]) -> Result<()> {
        self.verify_chain_inner(
            batch,
            VerifyTail::SkippedBecauseCommitOnlyCallersAlreadyKnowEveryToken,
            false,
        )?;
        Ok(())
    }

    fn verify_chain_inner(
        &mut self,
        batch: &[u32],
        tail: VerifyTail,
        padded_prefill_graph_forced_for_test: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let want_logits = tail == VerifyTail::ArgmaxAndLogitRows;
        let commit_only = tail == VerifyTail::SkippedBecauseCommitOnlyCallersAlreadyKnowEveryToken;
        let vf_on = !padded_prefill_graph_forced_for_test && !self.vf_passes.is_empty();
        let rows = self.verify_max_rows();
        ensure!(
            rows >= 2,
            "verify_chain needs the M-row prefill graph"
        );
        let live = batch.len();
        ensure!(
            (1..=rows).contains(&live),
            "verify_chain batch {live} out of 1..={rows}"
        );
        ensure!(
            self.verify.as_ref().is_some_and(|v| v.pending.is_none()),
            "verify_chain twice without advance(): {VERIFY_CHAIN_COMMITS_BY_SNAPSHOT_AND_M1_REPLAY_BECAUSE_DELTANET_STATE_IS_RECURRENT_NOT_POSITION_MASKED}"
        );
        ensure!(
            self.pos + live <= self.max_seq,
            "kv cache full at {} + {live} (max_seq {})",
            self.pos,
            self.max_seq
        );
        for &t in batch {
            ensure!((t as usize) < self.vocab, "token {t} out of vocab");
        }
        let hidden_words = self.config.hidden_size / 2;
        let row_bytes = (hidden_words * 4) as u64;
        let vocab_bytes = (self.vocab * 4) as u64;
        if self.verify.as_ref().is_some_and(|v| v.rollback.is_empty()) {
            let rollback: Vec<Buffer> = self
                .recurrent_states
                .iter()
                .map(|(_, bytes)| dispatch::storage_zeroed(self.ctx, "q3d-vf-rollback", *bytes))
                .collect();
            self.verify.as_mut().expect("verify list").rollback = rollback;
        }
        if want_logits && self.verify.as_ref().is_some_and(|v| v.row_logits.is_none()) {
            let buf =
                dispatch::storage_zeroed(self.ctx, "q3d-vf-row-logits", rows as u64 * vocab_bytes);
            self.verify.as_mut().expect("verify list").row_logits = Some(buf);
        }

        let graph_m = if vf_on { self.vf_m } else { self.pf_m };
        let mut padded: Vec<u32> = batch.to_vec();
        padded.resize(graph_m, *batch.last().expect("non-empty batch"));
        let ids: Vec<i32> = padded.iter().map(|&t| t as i32).collect();
        let (tok, ck, mask) = if vf_on {
            (
                self.vf_tok.as_ref().expect("verify without vf list"),
                self.vf_ck.as_ref().expect("verify without vf list"),
                self.vf_mask.as_ref().expect("verify without vf list"),
            )
        } else {
            (
                self.pf_tok.as_ref().expect("verify without pf list"),
                self.pf_ck.as_ref().expect("verify without pf list"),
                self.pf_mask.as_ref().expect("verify without pf list"),
            )
        };
        self.ctx
            .queue
            .write_buffer(mask, 0, cast_slice(&vec![0u32; graph_m]));
        self.ctx.queue.write_buffer(tok, 0, cast_slice(&ids));
        self.ctx.queue.write_buffer(
            ck,
            0,
            bytes_of(&CkParams {
                m_live: live as u32,
                base: self.pos as u32,
                pad0: 0,
                pad1: 0,
            }),
        );
        if vf_on {
            self.write_pf_attn_mk_fd(&self.vf_attn_mk_fd, live, self.pos);
            self.write_row_group_fds(&self.vf_vflash_row_fd, 1, live, self.pos);
            self.write_pf_chunk_fd_whole_view_and_start_for_tiled_flash_and_fp8_chunk_quantize(
                self.vf_chunk_fd_and_start.as_ref(),
                self.vf_m,
                self.pos,
            );
        } else {
            self.write_pf_attn_mk_fd(&self.pf_attn_mk_fd, live, self.pos);
            self.write_row_group_fds(&self.pf_vflash_row_fd, 1, live, self.pos);
            self.write_pf_chunk_fd_whole_view_and_start_for_tiled_flash_and_fp8_chunk_quantize(
                self.pf_chunk_fd_and_start.as_ref(),
                self.pf_m,
                self.pos,
            );
        }

        let v = self.verify.as_ref().expect("verify list");
        let scope =
            (!v.validated).then(|| self.ctx.device.push_error_scope(ErrorFilter::Validation));
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for ((src, bytes), dst) in self.recurrent_states.iter().zip(&v.rollback) {
            enc.copy_buffer_to_buffer(src, 0, dst, 0, *bytes);
        }
        {
            let graph_passes: &[Pass] = if vf_on {
                &self.vf_passes
            } else {
                &self.pf_passes
            };
            let gate_min = if vf_on {
                self.vf_vflash_gate_min_total
            } else {
                self.pf_vflash_gate_min_total
            };
            let gate_total = (self.pos + live) as u32;
            let resadd = if vf_on { &v.resadd } else { &v.chunk_resadd };
            let mut pass = enc.begin_compute_pass(&Default::default());
            for p in graph_passes.iter().chain(std::iter::once(resadd)) {
                if !vfl_gate_keeps(&p.label, gate_min, gate_total) {
                    continue;
                }
                dispatch_pass!(pass, p);
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
                        dispatch_pass!(pass, p);
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
            if let Some(e) = dispatch::pop_scope(scope) {
                bail!("qwen3_5_dense_wgpu verify chain validation: {e}");
            }
        }
        let toks: Vec<u32> = if commit_only {
            Vec::new()
        } else {
            dispatch::read_back(self.ctx, &v.tok, live).map_err(|e| anyhow!("{e}"))?
        };
        let logits = match v.row_logits.as_ref().filter(|_| want_logits) {
            Some(rl) => {
                dispatch::read_back(self.ctx, rl, live * self.vocab).map_err(|e| anyhow!("{e}"))?
            }
            None => Vec::new(),
        };
        let vm = self.verify.as_mut().expect("verify list");
        vm.validated = true;
        vm.pending = Some(batch.to_vec());
        vm.pending_rode_vf = vf_on;
        Ok((toks, logits))
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        let rows = self
            .verify
            .as_ref()
            .and_then(|v| v.pending.as_ref().map(|b| b.len()))
            .ok_or_else(|| anyhow!("advance() without a pending verify_chain"))?;
        ensure!(
            n <= rows,
            "advance {n} beyond the {rows} rows verify_chain forwarded"
        );
        let pending_rode_vf = self.verify.as_ref().is_some_and(|v| v.pending_rode_vf);
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
        if self.recurrent_states.is_empty() && self.mtp.is_none() {
            self.pos += n;
            return Ok(());
        }
        if n > 0 && pending_rode_vf && self.mtp_commit_scan_replay_active() {
            return self.commit_accepted_prefix_by_dn_scan_replay(n);
        }
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

    pub fn mtp_commit_scan_replay_active(&self) -> bool {
        !self.vf_commit_passes.is_empty()
    }

    fn commit_accepted_prefix_by_dn_scan_replay(&mut self, n: usize) -> Result<()> {
        ensure!(
            n >= 1 && !self.vf_commit_passes.is_empty(),
            "{MTP_COMMIT_SCAN_REPLAY_RESTORES_THE_SNAPSHOT_AND_REDISPATCHES_ONLY_EACH_DELTANET_CONV_SHIFT_AND_SCAN}"
        );
        let ck = self
            .vf_ck
            .as_ref()
            .expect("scan-replay commit passes exist only beside the dedicated verify graph");
        self.ctx.queue.write_buffer(
            ck,
            0,
            bytes_of(&CkParams {
                m_live: n as u32,
                base: self.pos as u32,
                pad0: 0,
                pad1: 0,
            }),
        );
        let scope = (!self.vf_commit_validated)
            .then(|| self.ctx.device.push_error_scope(ErrorFilter::Validation));
        submit_passes!(self.ctx, &self.vf_commit_passes);
        if let Some(scope) = scope {
            if let Some(e) = dispatch::pop_scope(scope) {
                bail!("qwen3_5_dense_wgpu scan-replay commit validation: {e}");
            }
            self.vf_commit_validated = true;
        }
        self.pos += n;
        Ok(())
    }

    fn checkpoints_refused(&self) -> bool {
        self.mtp.is_some() || self.mrope_rows_installed
    }

    fn ensure_not_mid_verify(&self, what: &str) -> Result<()> {
        ensure!(
            self.verify.as_ref().is_none_or(|v| v.pending.is_none()),
            "{what} mid-verify: advance() has not committed or rolled back"
        );
        Ok(())
    }

    fn copy_recurrent_states(&self, from_live: bool, other: &[Buffer]) {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        for ((live, bytes), o) in self.recurrent_states.iter().zip(other) {
            let (src, dst) = if from_live { (live, o) } else { (o, live) };
            enc.copy_buffer_to_buffer(src, 0, dst, 0, *bytes);
        }
        self.ctx.queue.submit([enc.finish()]);
    }

    fn alloc_checkpoint_clone(&self) -> Vec<Buffer> {
        self.recurrent_states
            .iter()
            .map(|(_, bytes)| dispatch::storage_zeroed(self.ctx, "q3d-resume-ckpt", *bytes))
            .collect()
    }

    pub fn checkpoint_bytes(&self) -> u64 {
        self.recurrent_states.iter().map(|(_, b)| *b).sum()
    }

    pub fn save_resume_checkpoint(&mut self) -> Result<bool> {
        self.ensure_not_mid_verify("save_resume_checkpoint")?;
        if self.checkpoints_refused() {
            return Ok(false);
        }
        if self.resume_checkpoint.is_empty() {
            self.resume_checkpoint = self.alloc_checkpoint_clone();
        }
        self.copy_recurrent_states(true, &self.resume_checkpoint);
        self.resume_checkpoint_pos = Some(self.pos);
        Ok(true)
    }

    pub fn save_grid_checkpoint(&mut self) -> Result<bool> {
        self.ensure_not_mid_verify("save_grid_checkpoint")?;
        if self.checkpoints_refused() {
            return Ok(false);
        }
        if self
            .grid_checkpoints
            .iter()
            .any(|(p, _)| *p == Some(self.pos))
        {
            return Ok(true);
        }
        let slot = if let Some(i) = self.grid_checkpoints.iter().position(|(p, _)| p.is_none()) {
            i
        } else if self.grid_checkpoints.len()
            < GRID_SLOT_CAP_BOUNDS_VRAM_AT_CAP_DELTANET_CLONES_AND_EVICTS_THE_LOWEST_BOUNDARY_BECAUSE_A_DIVERGING_SUFFIX_REWINDS_TO_THE_HIGHEST_BOUNDARY_UNDER_ITS_LCP
        {
            self.grid_checkpoints.push((None, self.alloc_checkpoint_clone()));
            let n = self.grid_checkpoints.len();
            eprintln!(
                "[q3d-wgpu] grid checkpoint slot {n} allocated: {:.1} MiB each, {:.1} MiB grid total",
                self.checkpoint_bytes() as f64 / (1024.0 * 1024.0),
                (n as u64 * self.checkpoint_bytes()) as f64 / (1024.0 * 1024.0)
            );
            n - 1
        } else {
            let (i, evicted) = self
                .grid_checkpoints
                .iter()
                .enumerate()
                .filter_map(|(i, (p, _))| p.map(|p| (i, p)))
                .min_by_key(|(_, p)| *p)
                .expect("a full grid has no vacant slot");
            eprintln!(
                "[q3d-wgpu] grid checkpoint cap {} truncates coverage: evicting boundary {evicted} to save {}",
                GRID_SLOT_CAP_BOUNDS_VRAM_AT_CAP_DELTANET_CLONES_AND_EVICTS_THE_LOWEST_BOUNDARY_BECAUSE_A_DIVERGING_SUFFIX_REWINDS_TO_THE_HIGHEST_BOUNDARY_UNDER_ITS_LCP,
                self.pos
            );
            i
        };
        self.copy_recurrent_states(true, &self.grid_checkpoints[slot].1);
        self.grid_checkpoints[slot].0 = Some(self.pos);
        Ok(true)
    }

    pub fn resume_checkpoint_pos(&self) -> Option<usize> {
        if self.checkpoints_refused() {
            return None;
        }
        self.resume_checkpoint_pos
    }

    pub fn grid_checkpoint_at_or_below(&self, pos: usize) -> Option<usize> {
        if self.checkpoints_refused() {
            return None;
        }
        self.grid_checkpoints
            .iter()
            .filter_map(|(p, _)| *p)
            .filter(|p| *p > 0 && *p <= pos)
            .max()
    }

    pub fn rewind_to(&mut self, pos: usize) -> Result<bool> {
        let single =
            self.resume_checkpoint_pos() == Some(pos) && !self.resume_checkpoint.is_empty();
        let grid = (!single && !self.checkpoints_refused())
            .then(|| {
                self.grid_checkpoints
                    .iter()
                    .position(|(p, _)| *p == Some(pos))
            })
            .flatten();
        if !single && grid.is_none() {
            return Ok(false);
        }
        self.ensure_not_mid_verify("rewind_to")?;
        ensure!(
            pos <= self.pos,
            "rewind_to {pos} is ahead of the live pos {}: the checkpoint outlived a state it \
             never saw",
            self.pos
        );
        match grid {
            Some(i) => self.copy_recurrent_states(false, &self.grid_checkpoints[i].1),
            None => self.copy_recurrent_states(false, &self.resume_checkpoint),
        }
        self.pos = pos;
        if self.resume_checkpoint_pos.is_some_and(|p| p > pos) {
            self.resume_checkpoint_pos = None;
        }
        for slot in &mut self.grid_checkpoints {
            if slot.0.is_some_and(|p| p > pos) {
                slot.0 = None;
            }
        }
        Ok(true)
    }

    pub fn mtp_active(&self) -> bool {
        self.mtp.is_some()
    }

    pub fn mtp_len(&self) -> usize {
        self.mtp.as_ref().map_or(0, |m| m.len)
    }

    pub fn lora_resident(&self) -> bool {
        self.lora_active
    }

    pub fn mtp_attach(&mut self, w: &WeightLoader) -> Result<()> {
        self.mtp_attach_with_adapter(w, None).map(|_| ())
    }

    pub fn mtp_attach_with_adapter(
        &mut self,
        w: &WeightLoader,
        adapter_dir: Option<&std::path::Path>,
    ) -> Result<usize> {
        let cfg = &self.config;
        let hidden = cfg.hidden_size;
        let hd = cfg.head_dim;
        let heads = cfg.num_attention_heads;
        let q_out = heads * hd * if cfg.attn_output_gate { 2 } else { 1 };
        let kv_out = cfg.num_key_value_heads * hd;
        let inter = cfg.intermediate_size;
        let mut host = MtpHostWeights {
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
                o: load_lin(
                    w,
                    "mtp.layers.0.self_attn.o_proj.weight",
                    hidden,
                    heads * hd,
                )?
                .into(),
                q_norm: load_norm_plus_one(w, &["mtp.layers.0.self_attn.q_norm.weight"], hd)?,
                k_norm: load_norm_plus_one(w, &["mtp.layers.0.self_attn.k_norm.weight"], hd)?,
                qkv_bias_bf16: vec![],
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
        let merged = match adapter_dir
            .map(mtp_speculator_deltas_from_adapter_dir)
            .transpose()?
            .flatten()
        {
            Some(d) => host.merge_speculator(&d)?,
            None => 0,
        };
        self.mtp_attach_host(&host)?;
        Ok(merged)
    }

    pub fn mtp_attach_host(&mut self, w: &MtpHostWeights) -> Result<()> {
        ensure!(self.mtp.is_none(), "mtp head already attached");
        ensure!(
            self.verify.is_some(),
            "the mtp drafter needs the M-row prefill graph: drafts are \
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
        ensure!(
            w.fc.n == hidden && w.fc.k == 2 * hidden,
            "mtp fc must be [{hidden}, {}], got [{}, {}]; \
             {MTP_HEAD_CONVENTIONS_ZERO_CENTERED_NORMS_EMB_FIRST_FC_AND_SHIFT_BY_ONE_KV_WITH_A_ZERO_HIDDEN_AT_POS_0}",
            2 * hidden,
            w.fc.n,
            w.fc.k
        );
        ensure!(
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
        ensure!(
            w.pre_fc_norm_embedding.len() == hidden
                && w.pre_fc_norm_hidden.len() == hidden
                && w.input_ln.len() == hidden
                && w.post_attn_ln.len() == hidden
                && w.final_norm.len() == hidden,
            "mtp norm vectors must all be [{hidden}]"
        );

        let s = Sources::new(self.ctx);
        let mut b = Builder {
            core: crate::wgpu_ledger::VramLedger::new(
                self.ctx,
                "q3d-",
                STAGING_FLUSH_BYTES,
            ),
            passes: Vec::new(),
            pf_passes: Vec::new(),
            vf_passes: Vec::new(),
            vf_commit_passes: Vec::new(),
            pf_dst_vf: false,
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

        let mut kv_state: Vec<(Buffer, u64)> = Vec::new();
        let no_pf: [Pf; 0] = [];
        let mut mtp_rope_rows_never_mrope_swapped: Vec<(Buffer, Buffer)> = Vec::new();
        let mut mtp_kv_caches_not_probed_by_the_rotkv_debug_api: Vec<(Buffer, Buffer)> = Vec::new();
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
            &no_pf,
            &mut mtp_rope_rows_never_mrope_swapped,
            &mut mtp_kv_caches_not_probed_by_the_rotkv_debug_api,
            false,
            None,
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
        build_mlp(
            &mut b,
            &s,
            &cfg,
            &w.mlp,
            &mnormed_post,
            &mmlp_out,
            &no_pf,
            None,
            None,
        )?;
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
        ensure!((tok as usize) < self.vocab, "mtp token {tok} out of vocab");
        ensure!(
            pos < self.max_seq,
            "mtp kv full at {pos} (max_seq {})",
            self.max_seq
        );
        let row_bytes = (self.config.hidden_size / 2 * 4) as u64;
        self.ctx
            .queue
            .write_buffer(&self.tok_buf, 0, bytes_of(&(tok as i32)));
        let needs_scope;
        {
            let m = self.mtp.as_ref().expect("mtp_run without an attached head");
            self.ctx
                .queue
                .write_buffer(&m.pos_buf, 0, bytes_of(&(pos as i32)));
            if full {
                let mut fd = self.fd_base;
                fd.total = (pos + 1) as u32;
                self.ctx.queue.write_buffer(&m.fd_buf, 0, bytes_of(&fd));
            }
            needs_scope = if full {
                !m.validated_full
            } else {
                !m.validated_kv
            };
            let scope =
                needs_scope.then(|| self.ctx.device.push_error_scope(ErrorFilter::Validation));
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
                let mtp_slice: &[Pass] = if full {
                    &m.passes
                } else {
                    &m.passes[..m.kv_end]
                };
                let head: &[Pass] = if full {
                    &self.passes[self.head_start..]
                } else {
                    &[]
                };
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in gather.iter().chain(mtp_slice).chain(head) {
                    dispatch_pass!(pass, p);
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
                if let Some(e) = dispatch::pop_scope(scope) {
                    bail!("qwen3_5_dense_wgpu mtp step validation: {e}");
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
        let t: Vec<u32> =
            dispatch::read_back(self.ctx, &self.token_out, 1).map_err(|e| anyhow!("{e}"))?;
        Ok(Some(t[0]))
    }

    pub fn debug_mtp_last_draft_logits_for_test(&self) -> Result<Vec<f32>> {
        ensure!(
            self.mtp.is_some(),
            "no mtp head attached, so the logits buffer holds no draft row"
        );
        dispatch::read_back(self.ctx, &self.logits, self.vocab).map_err(|e| anyhow!("{e}"))
    }

    pub fn mtp_draft_round(&mut self, anchor: u32, k: usize) -> Result<Vec<u32>> {
        ensure!(k >= 1, "mtp draft round needs k >= 1");
        ensure!(
            self.verify.as_ref().is_some_and(|v| v.pending.is_none()),
            "mtp draft round during a pending verify_chain"
        );
        let l = self.mtp_len();
        ensure!(
            self.mtp.is_some() && l == self.pos,
            "mtp kv desync: drafter holds {l} rows but the trunk committed {}; \
             {MTP_HEAD_CONVENTIONS_ZERO_CENTERED_NORMS_EMB_FIRST_FC_AND_SHIFT_BY_ONE_KV_WITH_A_ZERO_HIDDEN_AT_POS_0}",
            self.pos
        );
        ensure!(
            l + k <= self.max_seq && self.pos + k < self.max_seq,
            "mtp round does not fit: committed={l}, k={k}, max_seq={}",
            self.max_seq
        );
        self.mtp.as_mut().expect("mtp attached").round_base = Some(l);
        let mut drafts = Vec::with_capacity(k);
        let mut t = anchor;
        for j in 0..k {
            let pre = if l + j == 0 {
                MtpHid::Zero
            } else {
                MtpHid::Keep
            };
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
            .ok_or_else(|| anyhow!("mtp_post_verify without a preceding mtp_draft_round"))?;
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
        ensure!(
            self.mtp_len() == self.pos,
            "mtp kv catch-up desync: drafter holds {} rows, trunk committed {}",
            self.mtp_len(),
            self.pos
        );
        Ok(())
    }

    fn mtp_sync_committed_step(&mut self, token: u32) -> Result<()> {
        let pos_before = self.pos - 1;
        ensure!(
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
        ensure!(
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
                pass.set_pipeline(&v.chunk_resadd.pipeline);
                pass.set_bind_group(0, &v.chunk_resadd.bind, &[]);
                pass.dispatch_workgroups(
                    v.chunk_resadd.grid.0,
                    v.chunk_resadd.grid.1,
                    v.chunk_resadd.grid.2,
                );
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
    x: &Buffer,
    y: &Buffer,
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
    w: Buffer,
    s: Buffer,
    s_plain: Option<Buffer>,
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
    x: &Buffer,
    y: &Buffer,
    out_f32: bool,
) -> Result<()> {
    push_gemv_fp8_rowscale_lora(b, s, label, w, x, y, out_f32, None)
}

fn push_gemv_fp8_rowscale_lora(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Fp8Gpu,
    x: &Buffer,
    y: &Buffer,
    out_f32: bool,
    lora: Option<LoraFoldSlice<'_>>,
) -> Result<()> {
    ensure!(
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
    match lora {
        Some(lf) => {
            let lp = b.uni("q3d-gemvf8-lp", lf.params(label, w.n, w.k));
            b.push(
                label,
                &s.gemv_bf16,
                FP8_GEMV_LORA_FOLD_ENTRY,
                &[
                    (0, &w.w),
                    (1, x),
                    (2, &p),
                    (3, y),
                    (4, &w.s),
                    (11, lf.site.b.raw()),
                    (12, &lf.fold.partials),
                    (13, &lp),
                ],
                grid,
            )
        }
        None => b.push(
            label,
            &s.gemv_bf16,
            "q3w_gemv_fp8_rowscale",
            &[(0, &w.w), (1, x), (2, &p), (3, y), (4, &w.s)],
            grid,
        ),
    }
}

fn push_gemv_dn_merged(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    qkv: &Fp8Gpu,
    z: &Fp8Gpu,
    ab: &Bf16Gpu,
    x: &Buffer,
    y_qkv: &Buffer,
    y_z: &Buffer,
    y_ab: &Buffer,
    lora: Option<DnMergedLoraFold<'_>>,
) -> Result<()> {
    ensure!(
        qkv.k == z.k && qkv.k == ab.k,
        "{label}: merged DN projections read one normed x, so k must agree; got qkv {} z {} ab {}",
        qkv.k,
        z.k,
        ab.k
    );
    ensure!(
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
    match lora {
        Some(lf) => {
            let lp = b.uni(
                "q3d-dn-merged-lp",
                lf.params(label, qkv.n, z.n, ab.n, qkv.k),
            );
            b.push(
                label,
                &s.gemv_bf16,
                DN_MERGED_GEMV_LORA_FOLD_ENTRY,
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
                    (11, lf.site.b.raw()),
                    (12, &lf.fold.partials),
                    (14, &lp),
                ],
                grid,
            )
        }
        None => b.push(
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
        ),
    }
}

pub const PF_FP8_REPLAY_ROWS_REPLAY_THE_DECODE_GEMV_PER_SLOT: &str =
    "NV_Q3D_PF_FP8_REPLAY=1 makes the M-row graph's DeltaNet projections replay \
     q3w_gemv_fp8_rowscale once per slot with slot offsets, so each row's projection \
     bits equal the decode step it stands in for at fp8 defaults (the verify_chain \
     bit gate's first divergence source, task #13). The bf16-twin gemm stays the \
     default because it streams the weights once for all M rows; the replay streams \
     them M times and only a small verify graph can afford that.";

fn pf_fp8_replay_enabled() -> bool {
    env_arm_opt_in_eq_some_1_reread_every_call("NV_Q3D_PF_FP8_REPLAY")
}

fn push_gemv_fp8_rowscale_pf(
    b: &mut Builder,
    s: &Sources,
    pf: &Pf,
    label: &str,
    w: &Fp8Gpu,
    x: &Buffer,
    x_stride_words: usize,
    y: &Buffer,
    y_stride_words: usize,
) -> Result<()> {
    b.pf_dst_vf = pf.vf;
    ensure!(
        pf.m <= 16,
        "{label}: NV_Q3D_PF_FP8_REPLAY streams the full weight once per slot, which only the \
         small verify graphs can afford; m={} is a coop-scale prefill and the replay is refused",
        pf.m
    );
    ensure!(
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
            let si =
                (((r / 128) * k_tiles + kb / 4) * 32 + r % 32) * 16 + ((r / 32) % 4) * 4 + kb % 4;
            out[r * k_blocks + kb] = l.scales_swizzled[si];
        }
    }
    out
}

fn upload_nvfp4(b: &mut Builder, label: &str, l: &HostNvfp4Lin) -> Nvfp4Gpu {
    let scales_linear = if pf_coop_active(b.ctx) && l.n.is_multiple_of(16) && l.k.is_multiple_of(16)
    {
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
     FP8_PROJ_STREAMS_CHECKPOINT_BYTES_INSTEAD_OF_A_BF16_UPCAST).";

pub fn w8_mode() -> (bool, bool) {
    let v = env::var("NV_Q3D_WGPU_W8").unwrap_or_default();
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
        HostDenseLin::Fp8 { fp8, bf16 } => {
            let g = upload_fp8(b, &format!("{label}-fp8"), fp8);
            let twin = if g.s_plain.is_some() && !pf_fp8_replay_enabled() && pf_rides_coop {
                None
            } else {
                Some(upload_bf16(b, label, bf16))
            };
            DenseLinGpu::Fp8(g, twin)
        }
        HostDenseLin::Nvfp4(x) if w8_enabled(b.ctx, scope) => {
            DenseLinGpu::Int8(upload_nvfp4_i8(b, label, x))
        }
        HostDenseLin::Nvfp4(x) => DenseLinGpu::Nvfp4(upload_nvfp4(b, label, x)),
        HostDenseLin::AffineW4(x) => DenseLinGpu::AffineW4(upload_affine_w4(
            b,
            label,
            x,
            pf_rides_coop && pf_coop_tile(b.ctx) == Some(8) && !pf_coop_affine_wq8_enabled(),
        )),
    }
}

fn push_gemv_i8(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &I8Gpu,
    x: &Buffer,
    y: &Buffer,
) -> Result<()> {
    ensure!(
        b.ctx.caps.subgroup,
        "{label}: every q3d i8 gemv entry rides subgroup butterflies, and this device grants no \
         subgroups, so its module ships stripped of them; serve a non-i8 checkpoint here"
    );
    ensure!(
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
    x: &Buffer,
    k: usize,
    consumers: &[&DenseLinGpu],
) -> Result<Option<QuantIn>> {
    let mut global: Option<f32> = None;
    for c in consumers {
        let Some(g) = c.input_global() else { continue };
        match global {
            None => global = Some(g),
            Some(prev) => ensure!(
                (prev - g).abs() <= 1e-6,
                "{label}: consumers disagree on input_global_scale ({prev} vs {g})"
            ),
        }
    }
    let Some(global) = global else {
        return Ok(None);
    };
    let k_blocks = k / NVFP4_BLOCK;
    ensure!(
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
    y_gate: &Buffer,
    y_up: &Buffer,
    w: &Nvfp4Gpu,
    k: usize,
) -> Result<QuantIn> {
    let k_blocks = k / NVFP4_BLOCK;
    ensure!(
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

fn push_rmsres_post_w4a8(
    b: &mut Builder,
    s: &Sources,
    hidden: usize,
    eps: f32,
    mixed_out: &Buffer,
    res: &Buffer,
    post_w: &Buffer,
    normed_post: &Buffer,
) -> Result<W4a8Quant> {
    let q = W4a8Quant {
        xq: b.zeros("q3d-mlp-q8x", hidden as u64),
        xs: b.zeros("q3d-mlp-q8s", 4),
    };
    let p = b.uni(
        "q3d-rmsres-w48-p",
        W4a8ProducerParams {
            k_blocks: (hidden / NVFP4_BLOCK) as u32,
            words_per_row: (hidden / 2) as u32,
            hidden: hidden as u32,
            eps,
        },
    );
    b.push(
        "q3d-rmsres-post-w48q",
        &s.w4a8_prod,
        W4A8_RMSRES_ENTRY,
        &[
            (0, mixed_out),
            (1, res),
            (2, post_w),
            (3, normed_post),
            (4, &p),
            (5, &q.xq),
            (6, &q.xs),
        ],
        (1, 1, 1),
    )?;
    Ok(q)
}

fn push_silu_mul_rowquant_w4a8(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    y_gate: &Buffer,
    y_up: &Buffer,
    k: usize,
    act_bf16_out: Option<&Buffer>,
) -> Result<W4a8Quant> {
    ensure!(
        k.is_multiple_of(32),
        "{label}: w4a8 silu rowquant emits whole even/odd word groups, k {k} must be 32-aligned"
    );
    ensure!(
        k / 2 <= W4A8_SILU_SMEM_WORDS_MIRRORS_THE_WGSL_40KB_ARRAY_SO_THE_ARM_CLAMPS_INTER_AT_20480,
        "{label}: k {k} exceeds the silu rowquant 40KB workgroup stage"
    );
    let q = W4a8Quant {
        xq: b.zeros(label, k as u64),
        xs: b.zeros(label, 4),
    };
    let p = b.uni(
        label,
        W4a8ProducerParams {
            k_blocks: (k / NVFP4_BLOCK) as u32,
            words_per_row: (k / 2) as u32,
            hidden: k as u32,
            eps: 0.0,
        },
    );
    let smem = w4a8_silu_use_smem(b.ctx.caps.max_compute_workgroup_storage_size);
    match act_bf16_out {
        Some(act) => b.push(
            label,
            &s.w4a8_prod,
            if smem {
                W4A8_SILU_ACT_ENTRY_STORES_BF16_ACT_FOR_THE_DOWN_LORA_SHRINK
            } else {
                W4A8_SILU_ACT_NOSMEM_ENTRY
            },
            &[
                (3, act),
                (4, &p),
                (5, &q.xq),
                (6, &q.xs),
                (7, y_gate),
                (8, y_up),
            ],
            (1, 1, 1),
        )?,
        None => b.push(
            label,
            &s.w4a8_prod,
            if smem {
                W4A8_SILU_ENTRY
            } else {
                W4A8_SILU_NOSMEM_ENTRY
            },
            &[(4, &p), (5, &q.xq), (6, &q.xs), (7, y_gate), (8, y_up)],
            (1, 1, 1),
        )?,
    }
    Ok(q)
}

fn push_gemv_nvfp4_w4a8(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Nvfp4Gpu,
    q: &W4a8Quant,
    y: &Buffer,
    lora: Option<LoraFoldSlice<'_>>,
) -> Result<()> {
    ensure!(
        w.k.is_multiple_of(32),
        "{label}: gemv_nvfp4_w4a8_mrow2 consumes block pairs, k {} must be 32-aligned",
        w.k
    );
    let grid = b.grid1(w.n as u64, W4A8_GEMV_ROWS_PER_GROUP);
    let alpha = w.alpha
        * nvfp4_input_global_folded_into_x_codes_by_q3w_qz_core_so_the_a16_route_multiplies_it(w);
    let p = b.uni(label, wk::gemv_nvfp4::gemv_params(alpha, w.n, w.k, grid.0));
    match lora {
        Some(lf) => {
            let lp = b.uni(label, lf.params(label, w.n, w.k));
            b.push(
                label,
                &s.gemv_w4a8,
                W4A8_GEMV_LORA_FOLD_ENTRY,
                &[
                    (0, &w.w),
                    (1, &w.scales),
                    (2, &q.xq),
                    (3, &q.xs),
                    (4, &p),
                    (5, y),
                    (9, lf.site.b.raw()),
                    (10, &lf.fold.partials),
                    (11, &lp),
                ],
                grid,
            )
        }
        None => b.push(
            label,
            &s.gemv_w4a8,
            W4A8_GEMV_ENTRY,
            &[
                (0, &w.w),
                (1, &w.scales),
                (2, &q.xq),
                (3, &q.xs),
                (4, &p),
                (5, y),
            ],
            grid,
        ),
    }
}

fn push_gemv_nvfp4(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Nvfp4Gpu,
    q: &QuantIn,
    y: &Buffer,
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
    ya: &Buffer,
    yb: &Buffer,
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
    if r.entry != MLP_GEMV_2W_ROUTE_IS_EXACTLY_THE_MROW2_DECODE_ROUTE_ANY_OTHER_ENTRY_KEEPS_THE_PAIR
    {
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvW4A16AffineParams {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmW4A16AffineParams {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
    m_live: u32,
    x_stride_words: u32,
    y_stride_words: u32,
}

const AFFINE_Y_WGSL_EXISTS_BECAUSE_GEMV_W4A16_AFFINE_STORES_ONE_BF16_WORD_PER_ROW_WHILE_THE_GRAPH_READS_BF16_PAIRS:
    &str = r#"
struct AffYParams {
    n: u32,
    y_off: u32,
    groups_x: u32,
    pad0: u32,
};

@group(0) @binding(0) var<storage, read> affy_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> affy_pairs: array<u32>;
@group(0) @binding(2) var<uniform> affy_p: AffYParams;
@group(0) @binding(3) var<storage, read_write> affy_f32: array<f32>;

@compute @workgroup_size(256)
fn q3w_affine_y_pack_pairs(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let i = (wid.x + wid.y * affy_p.groups_x) * 256u + lid.x;
    if (i >= affy_p.n) {
        return;
    }
    affy_pairs[i] = (affy_src[2u * i] & 0xffffu) | (affy_src[2u * i + 1u] << 16u);
}

@compute @workgroup_size(256)
fn q3w_affine_y_expand_f32(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let i = (wid.x + wid.y * affy_p.groups_x) * 256u + lid.x;
    if (i >= affy_p.n) {
        return;
    }
    affy_f32[affy_p.y_off + i] = bitcast<f32>((affy_src[i] & 0xffffu) << 16u);
}
"#;

pod! {
    struct AffYParams {
        n: u32,
        y_off: u32,
        groups_x: u32,
        pad0: u32,
    }
}

fn push_gemv_affine_w4_raw_words(
    b: &mut Builder,
    label: &str,
    w: &AffineW4Gpu,
    x: &Buffer,
) -> Result<Buffer> {
    let raw = b.zeros(&format!("{label}-affy"), (w.n * 4) as u64);
    push_gemv_affine_w4(b, label, w, x, &raw)?;
    Ok(raw)
}

pub const AFFINE_MROW4H_ENV_HALF_WIDTH_WORKGROUP_MEASURED_2X_ON_DOTS_MLP_SHAPES_DEFAULT_OFF_QWEN38_NUMBERS_PINNED_ON_MROW4:
    &str = "NV_Q3D_AFFINE_MROW4H";

fn push_gemv_affine_w4_into_pairs(
    b: &mut Builder,
    label: &str,
    w: &AffineW4Gpu,
    x: &Buffer,
    y_pairs: &Buffer,
) -> Result<()> {
    ensure!(
        b.ctx.caps.subgroup,
        "{label}: every gemv_w4a16_affine entry rides the subgroupShuffleXor butterfly with no \
         workgroup-tree twin, and this device grants no subgroups; an affine-w4 checkpoint \
         cannot decode here (browser pipeline errors are fire-and-forget, so without this \
         refusal the dispatches silently no-op)"
    );
    ensure!(
        w.n.is_multiple_of(2),
        "{label}: the pk entry packs two rows per word; n {} must be even",
        w.n
    );
    let mrow = w.gs.is_multiple_of(32);
    let mrow4h = mrow
        && env_arm_opt_in_eq_some_1_reread_every_call(
            AFFINE_MROW4H_ENV_HALF_WIDTH_WORKGROUP_MEASURED_2X_ON_DOTS_MLP_SHAPES_DEFAULT_OFF_QWEN38_NUMBERS_PINNED_ON_MROW4,
        );
    let rows_per_group = if mrow4h {
        wk::gemv_w4a16_affine::MROW4H_ROWS_PER_GROUP
    } else if mrow {
        wk::gemv_w4a16_affine::MROW4_ROWS_PER_GROUP
    } else {
        wk::gemv_w4a16_affine::ROWS_PER_GROUP
    };
    let entry = if mrow4h {
        wk::gemv_w4a16_affine::MROW4H_PK_ENTRY_REQUIRES_GS_MULTIPLE_OF_32
    } else if mrow {
        wk::gemv_w4a16_affine::MROW4_PK_ENTRY_REQUIRES_GS_MULTIPLE_OF_32
    } else {
        wk::gemv_w4a16_affine::BLOCK_PK_ENTRY
    };
    let grid = b.grid1(w.n as u64, rows_per_group);
    let p = b.uni(
        label,
        GemvW4A16AffineParams {
            n_rows: w.n as u32,
            k_elems: w.k as u32,
            gs: w.gs as u32,
            w_row_words: (w.k / 8) as u32,
            scale_row_stride: (w.k / w.gs) as u32,
            groups_x: grid.0,
            pad0: 0,
            pad1: 0,
        },
    );
    b.push(
        label,
        &wk::gemv_w4a16_affine::source(),
        entry,
        &[
            (0, &w.w),
            (1, &w.scales),
            (2, x),
            (3, y_pairs),
            (4, &p),
            (wk::gemv_w4a16_affine::BIAS_SLOT, &w.biases),
        ],
        (grid.0, grid.1, 1),
    )
}

fn push_affine_y_expand_f32(
    b: &mut Builder,
    label: &str,
    raw: &Buffer,
    rows: usize,
    y_off: usize,
    y_f32: &Buffer,
) -> Result<()> {
    let grid = b.grid1(rows as u64, 256);
    let p = b.uni(
        &format!("{label}-p"),
        AffYParams {
            n: rows as u32,
            y_off: y_off as u32,
            groups_x: grid.0,
            pad0: 0,
        },
    );
    b.push(
        label,
        AFFINE_Y_WGSL_EXISTS_BECAUSE_GEMV_W4A16_AFFINE_STORES_ONE_BF16_WORD_PER_ROW_WHILE_THE_GRAPH_READS_BF16_PAIRS,
        "q3w_affine_y_expand_f32",
        &[(0, raw), (2, &p), (3, y_f32)],
        grid,
    )
}

fn push_gemv_affine_w4(
    b: &mut Builder,
    label: &str,
    w: &AffineW4Gpu,
    x: &Buffer,
    y: &Buffer,
) -> Result<()> {
    ensure!(
        b.ctx.caps.subgroup,
        "{label}: every gemv_w4a16_affine entry rides the subgroupShuffleXor butterfly with no \
         workgroup-tree twin, and this device grants no subgroups; an affine-w4 checkpoint \
         cannot decode here (browser pipeline errors are fire-and-forget, so without this \
         refusal the dispatches silently no-op)"
    );
    let mrow = w.gs.is_multiple_of(32);
    let mrow4h = mrow
        && env_arm_opt_in_eq_some_1_reread_every_call(
            AFFINE_MROW4H_ENV_HALF_WIDTH_WORKGROUP_MEASURED_2X_ON_DOTS_MLP_SHAPES_DEFAULT_OFF_QWEN38_NUMBERS_PINNED_ON_MROW4,
        );
    let rows_per_group = if mrow4h {
        wk::gemv_w4a16_affine::MROW4H_ROWS_PER_GROUP
    } else if mrow {
        wk::gemv_w4a16_affine::MROW4_ROWS_PER_GROUP
    } else {
        wk::gemv_w4a16_affine::ROWS_PER_GROUP
    };
    let entry = if mrow4h {
        wk::gemv_w4a16_affine::MROW4H_PROBE_ENTRY
    } else if mrow {
        wk::gemv_w4a16_affine::MROW4_ENTRY_REQUIRES_GS_MULTIPLE_OF_32
    } else {
        wk::gemv_w4a16_affine::BLOCK_ENTRY
    };
    let grid = b.grid1(w.n as u64, rows_per_group);
    let p = b.uni(
        label,
        GemvW4A16AffineParams {
            n_rows: w.n as u32,
            k_elems: w.k as u32,
            gs: w.gs as u32,
            w_row_words: (w.k / 8) as u32,
            scale_row_stride: (w.k / w.gs) as u32,
            groups_x: grid.0,
            pad0: 0,
            pad1: 0,
        },
    );
    b.push(
        label,
        &wk::gemv_w4a16_affine::source(),
        entry,
        &[
            (0, &w.w),
            (1, &w.scales),
            (2, x),
            (3, y),
            (4, &p),
            (wk::gemv_w4a16_affine::BIAS_SLOT, &w.biases),
        ],
        (grid.0, grid.1, 1),
    )
}

pub const AFFINE_GU_FUSED_ENV_GATE_AND_UP_ARE_ONE_MROW4H_DISPATCH_PER_MLP_DEFAULT_OFF_SET_1_TO_ENGAGE:
    &str = "NV_Q3D_AFFINE_GU_FUSED";

fn push_gemv_affine_w4_gu_pairs(
    b: &mut Builder,
    label: &str,
    a: &AffineW4Gpu,
    u: &AffineW4Gpu,
    x: &Buffer,
    y_gate: &Buffer,
    y_up: &Buffer,
) -> Result<bool> {
    let same = a.k == u.k && a.gs == u.gs;
    let routable = a.gs.is_multiple_of(32) && a.n.is_multiple_of(4) && u.n.is_multiple_of(4);
    if !(same && routable) {
        return Ok(false);
    }
    let n_total = a.n + u.n;
    let grid = b.grid1(n_total as u64, wk::gemv_w4a16_affine::MROW4H_ROWS_PER_GROUP);
    let p = b.uni(
        label,
        GemvW4A16AffineParams {
            n_rows: n_total as u32,
            k_elems: a.k as u32,
            gs: a.gs as u32,
            w_row_words: (a.k / 8) as u32,
            scale_row_stride: (a.k / a.gs) as u32,
            groups_x: grid.0,
            pad0: a.n as u32,
            pad1: 0,
        },
    );
    b.push(
        label,
        &wk::gemv_w4a16_affine::source(),
        wk::gemv_w4a16_affine::GU_PK_ENTRY_SEGMENTS_MUST_BE_MULTIPLES_OF_4_ROWS,
        &[
            (0, &a.w),
            (1, &a.scales),
            (2, x),
            (3, y_gate),
            (4, &p),
            (wk::gemv_w4a16_affine::QKV_Y_K_SLOT, y_up),
            (wk::gemv_w4a16_affine::BIAS_SLOT, &a.biases),
            (wk::gemv_w4a16_affine::GU_W_B_SLOT, &u.w),
            (wk::gemv_w4a16_affine::GU_SCALE_B_SLOT, &u.scales),
            (wk::gemv_w4a16_affine::GU_BIAS_B_SLOT, &u.biases),
        ],
        (grid.0, grid.1, 1),
    )?;
    Ok(true)
}

fn fuse_affine_qkv_hosts(
    q: &HostDenseLin,
    k: &HostDenseLin,
    v: &HostDenseLin,
) -> Option<HostAffineW4Lin> {
    let (HostDenseLin::AffineW4(hq), HostDenseLin::AffineW4(hk), HostDenseLin::AffineW4(hv)) =
        (q, k, v)
    else {
        return None;
    };
    let same = hq.k == hk.k && hk.k == hv.k && hq.gs == hk.gs && hk.gs == hv.gs;
    let routable = hq.gs.is_multiple_of(32)
        && hq.n.is_multiple_of(2)
        && hk.n.is_multiple_of(2)
        && hv.n.is_multiple_of(2);
    if !(same && routable) {
        return None;
    }
    let mut words = hq.words.clone();
    words.extend_from_slice(&hk.words);
    words.extend_from_slice(&hv.words);
    let mut scales_bf16 = hq.scales_bf16.clone();
    scales_bf16.extend_from_slice(&hk.scales_bf16);
    scales_bf16.extend_from_slice(&hv.scales_bf16);
    let mut biases_bf16 = hq.biases_bf16.clone();
    biases_bf16.extend_from_slice(&hk.biases_bf16);
    biases_bf16.extend_from_slice(&hv.biases_bf16);
    Some(HostAffineW4Lin {
        words,
        scales_bf16,
        biases_bf16,
        n: hq.n + hk.n + hv.n,
        k: hq.k,
        gs: hq.gs,
    })
}

fn push_gemv_affine_w4_qkv_pairs(
    b: &mut Builder,
    label: &str,
    w: &AffineW4Gpu,
    nq: usize,
    nk: usize,
    x: &Buffer,
    y_q: &Buffer,
    y_k: &Buffer,
    y_v: &Buffer,
) -> Result<()> {
    let grid = b.grid1(w.n as u64, wk::gemv_w4a16_affine::MROW4_ROWS_PER_GROUP);
    let p = b.uni(
        label,
        GemvW4A16AffineParams {
            n_rows: w.n as u32,
            k_elems: w.k as u32,
            gs: w.gs as u32,
            w_row_words: (w.k / 8) as u32,
            scale_row_stride: (w.k / w.gs) as u32,
            groups_x: grid.0,
            pad0: nq as u32,
            pad1: nk as u32,
        },
    );
    b.push(
        label,
        &wk::gemv_w4a16_affine::source(),
        wk::gemv_w4a16_affine::QKV_PK_ENTRY_SEGMENTS_MUST_BE_EVEN_ROWS,
        &[
            (0, &w.w),
            (1, &w.scales),
            (2, x),
            (3, y_q),
            (4, &p),
            (wk::gemv_w4a16_affine::BIAS_SLOT, &w.biases),
            (wk::gemv_w4a16_affine::QKV_Y_K_SLOT, y_k),
            (wk::gemv_w4a16_affine::QKV_Y_V_SLOT, y_v),
        ],
        (grid.0, grid.1, 1),
    )
}

fn mlx_affine_dequant_f16_pairs(l: &HostAffineW4Lin) -> Vec<u32> {
    let words_per_row = l.k / 8;
    let groups = l.k / l.gs;
    let words_per_group = l.gs / 8;
    let row_pairs = l.k / 2;
    let mut out = vec![0u32; l.n * row_pairs];
    let lanes = std::thread::available_parallelism()
        .map_or(4, |p| p.get())
        .min(12);
    let rows_per_chunk = l.n.div_ceil(lanes).max(1);
    let dequant_chunk = |ci: usize, chunk: &mut [u32]| {
        let row0 = ci * rows_per_chunk;
        for (ri, dst) in chunk.chunks_mut(row_pairs).enumerate() {
            let row = row0 + ri;
            for g in 0..groups {
                let sb = row * groups + g;
                let sc_v = f32::from_bits((l.scales_bf16[sb] as u32) << 16);
                let bi_v = f32::from_bits((l.biases_bf16[sb] as u32) << 16);
                for wg_i in 0..words_per_group {
                    let w_i = g * words_per_group + wg_i;
                    let word = l.words[row * words_per_row + w_i];
                    for pr in 0..4 {
                        let q0 = ((word >> (8 * pr)) & 0xf) as f32;
                        let q1 = ((word >> (8 * pr + 4)) & 0xf) as f32;
                        let lo = half::f16::from_f32(q0 * sc_v + bi_v).to_bits() as u32;
                        let hi = half::f16::from_f32(q1 * sc_v + bi_v).to_bits() as u32;
                        dst[w_i * 4 + pr] = lo | (hi << 16);
                    }
                }
            }
        }
    };
    #[cfg(target_arch = "wasm32")]
    for (ci, chunk) in out.chunks_mut(rows_per_chunk * row_pairs).enumerate() {
        dequant_chunk(ci, chunk);
    }
    #[cfg(not(target_arch = "wasm32"))]
    std::thread::scope(|sc| {
        let dequant_chunk = &dequant_chunk;
        for (ci, chunk) in out.chunks_mut(rows_per_chunk * row_pairs).enumerate() {
            sc.spawn(move || dequant_chunk(ci, chunk));
        }
    });
    out
}

#[cfg(target_arch = "wasm32")]
fn upload_affine_w4_rows_from_loader_view_so_wasm32_never_hosts_a_full_copy(
    b: &mut Builder,
    label: &str,
    w: &WeightLoader,
    module: &str,
    off: usize,
    rows: usize,
    k: usize,
    gs: usize,
) -> Result<AffineW4Gpu> {
    let row_bytes = k / 8 * 4;
    let groups = k / gs;
    let raw_w = w.raw_range_cow(
        &format!("{module}.weight"),
        off * row_bytes,
        rows * row_bytes,
    )?;
    let raw_s = w.raw_range_cow(
        &format!("{module}.scales"),
        off * groups * 2,
        rows * groups * 2,
    )?;
    let raw_b = w.raw_range_cow(
        &format!("{module}.biases"),
        off * groups * 2,
        rows * groups * 2,
    )?;
    let wbuf = b.zeros(&format!("{label}-w4aff"), (rows * row_bytes) as u64);
    b.ctx.queue.write_buffer(&wbuf, 0, &raw_w);
    let widen_u16 = |bytes: &[u8]| -> Vec<u32> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
            .collect()
    };
    let s = widen_u16(&raw_s);
    let bi = widen_u16(&raw_b);
    Ok(AffineW4Gpu {
        w: wbuf,
        scales: b.upload_u32(&format!("{label}-w4aff-s"), &s),
        biases: b.upload_u32(&format!("{label}-w4aff-b"), &bi),
        f16_twin: None,
        n: rows,
        k,
        gs,
    })
}

fn upload_affine_w4(
    b: &mut Builder,
    label: &str,
    x: &HostAffineW4Lin,
    want_f16_twin: bool,
) -> AffineW4Gpu {
    let widen = |v: &[u16]| -> Vec<u32> { v.iter().map(|h| *h as u32).collect() };
    let f16_twin = want_f16_twin.then(|| {
        b.upload_u32(
            &format!("{label}-w4aff-f16"),
            &mlx_affine_dequant_f16_pairs(x),
        )
    });
    AffineW4Gpu {
        w: b.upload_u32(&format!("{label}-w4aff"), &x.words),
        scales: b.upload_u32(&format!("{label}-w4aff-s"), &widen(&x.scales_bf16)),
        biases: b.upload_u32(&format!("{label}-w4aff-b"), &widen(&x.biases_bf16)),
        f16_twin,
        n: x.n,
        k: x.k,
        gs: x.gs,
    }
}

fn push_gemv(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &DenseLinGpu,
    x: &Buffer,
    q: Option<&QuantIn>,
    y: &Buffer,
) -> Result<()> {
    match w {
        DenseLinGpu::Bf16(w) => push_gemv_bf16(b, s, label, w, x, y, false),
        DenseLinGpu::AffineW4(w) => push_gemv_affine_w4_into_pairs(b, label, w, x, y),
        DenseLinGpu::Fp8(w, _) => push_gemv_fp8_rowscale(b, s, label, w, x, y, false),
        DenseLinGpu::Int8(w) => push_gemv_i8(b, s, label, w, x, y),
        DenseLinGpu::Nvfp4(w) => {
            let q = q.ok_or_else(|| anyhow!("{label}: nvfp4 gemv without quantized x"))?;
            push_gemv_nvfp4(b, s, label, w, q, y)
        }
    }
}

pub const PF_COOP_AFFINE_WQ8_ENV_OPT_IN_IN_KERNEL_DEQUANT_REPLACES_THE_F16_TWINS: &str =
    "NV_Q3D_PF_COOP_AFFINE_WQ8";

pub const WQ8_DELETES_THE_F16_TWINS_BY_STAGING_THE_DEQUANT_IN_KERNEL: &str =
    "NV_Q3D_PF_COOP_AFFINE_WQ8=1 routes tile-8 affine coop projections through \
     gemm_coop_f16::source_wq8_affine, which decodes the packed w4 into the workgroup B stage \
     with the host twin's exact mul-then-add rounding \
     (WQ8_AFFINE_STAGES_MATCH_THE_HOST_F16_TWIN_BIT_FOR_BIT), so no f16 twin is uploaded at \
     all -- the twin copies of every affine mat (a full-precision-class shadow of the trunk in \
     GPU-shared memory) disappear while the fragments the MMA consumes stay bit-identical";

fn pf_coop_affine_wq8_enabled() -> bool {
    nv_config::env_arm_opt_in_eq_some_1_cached_first_read_wins!(
        PF_COOP_AFFINE_WQ8_ENV_OPT_IN_IN_KERNEL_DEQUANT_REPLACES_THE_F16_TWINS
    )
}

fn dn_affine_coop_arm(aw: &AffineW4Gpu) -> PfCoopW<'_> {
    if pf_coop_affine_wq8_enabled() {
        PfCoopW::WqAffine {
            w: &aw.w,
            scales: &aw.scales,
            biases: &aw.biases,
            gs: aw.gs,
        }
    } else {
        PfCoopW::F16(
            aw.f16_twin
                .as_ref()
                .expect("the tile-8 DN coop route without wq8 uploads the f16 twin at build time"),
        )
    }
}

fn pf_coop_covers(pf: &Pf, w: &DenseLinGpu) -> bool {
    let Some(cp) = pf.coop.as_ref() else {
        return false;
    };
    match w {
        DenseLinGpu::AffineW4(w) => {
            let shape_ok =
                w.gs.is_multiple_of(16) && w.n.is_multiple_of(16) && w.k.is_multiple_of(16);
            shape_ok && (cp.tile == 16 || w.f16_twin.is_some() || pf_coop_affine_wq8_enabled())
        }
        DenseLinGpu::Fp8(f, _) => {
            cp.tile == 16
                && !pf_fp8_replay_enabled()
                && f.s_plain.is_some()
                && f.n.is_multiple_of(16)
                && f.k.is_multiple_of(16)
        }
        DenseLinGpu::Nvfp4(w) => {
            cp.tile == 16
                && w.scales_linear.is_some()
                && w.n.is_multiple_of(16)
                && w.k.is_multiple_of(16)
        }
        DenseLinGpu::Bf16(_) | DenseLinGpu::Int8(_) => false,
    }
}

fn nvfp4_input_global_folded_into_x_codes_by_q3w_qz_core_so_the_a16_route_multiplies_it(
    w: &Nvfp4Gpu,
) -> f32 {
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
    Wq(wk::gemm_coop_f16::WqFmt, &'a Buffer, &'a Buffer),
    WqAffine {
        w: &'a Buffer,
        scales: &'a Buffer,
        biases: &'a Buffer,
        gs: usize,
    },
    F16(&'a Buffer),
}

pub const PF_GEMM_ROUTE_TABLE_ADJUDICATED_BY_THE_IN_MODEL_PROFILE_NOT_THE_STANDALONE_LADDER: &str =
    "The chunk-prefill projection GEMMs route per (weight-format, n, k, m_alloc) through \
     gemm_coop_f16's Q38 route table; shapes or m_alloc values outside the table ride one \
     tm-derived config (tn=16/tm, sg=2). Table candidates come from the standalone ladder \
     (coop_pf_gemm_route_ladder_at_qwen38_prefill_shapes); the winner per shape comes from \
     the in-model NV_WGPU_PROFILE per-symbol pair, because the standalone SLC-defeat rig \
     mispredicts in-model ranking (it scored the shipped gate/up tm8_tn2_sg2_ku4 at 3.1ms \
     where the model measures 0.93ms, and the m=256 winners lose in-model at m=512 on \
     every tall k>n shape). The table cannot change numerics: every config accumulates \
     the same 16-wide fragments in ascending-k order, asserted bit-equal at the exact \
     shapes and both m_alloc bases by \
     pf_gemm_route_table_configs_are_bit_equal_to_the_default_route. \
     The deleted k>n narrow-column heuristic (NV_Q3D_PF_COOP_NARROW) is \
     superseded: its down-proj pick is the table's m=512 row and loses to the table's \
     m=256 row by 1.26x in-model.";

fn pf_gemm_route_cfg(
    m_alloc: u32,
    kind: Option<wk::gemm_coop_f16::WqFmt>,
    n: usize,
    k: usize,
) -> Option<(u32, u32, u32, u32)> {
    wk::gemm_coop_f16::q38_pf_route(m_alloc, kind, n as u32, k as u32)
}

fn pf_coop_cast_x(
    b: &mut Builder,
    pf: &Pf,
    label: &str,
    x: &Buffer,
    x_stride_words: usize,
    k: usize,
) -> Result<()> {
    b.pf_dst_vf = pf.vf;
    let cp = pf
        .coop
        .as_ref()
        .ok_or_else(|| anyhow!("{label}: coop cast without a coop plan"))?;
    ensure!(
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

fn pf_coop_gemm(
    b: &mut Builder,
    pf: &Pf,
    label: &str,
    w: PfCoopW<'_>,
    n: usize,
    k: usize,
    alpha: f32,
    y: &Buffer,
    y_stride_words: usize,
) -> Result<()> {
    use wk::gemm_coop_f16 as cf;
    b.pf_dst_vf = pf.vf;
    let cp = pf
        .coop
        .as_ref()
        .ok_or_else(|| anyhow!("{label}: coop gemm without a coop plan"))?;
    ensure!(
        n.is_multiple_of(16) && k.is_multiple_of(16),
        "{label}: the coop epilogue stores whole 16x16 fragments; n={n} k={k} must be 16-aligned"
    );
    ensure!(
        y_stride_words * 2 == n,
        "{label}: pf consumers read bf16 rows of n/2 words; y_stride_words {y_stride_words} != {}",
        n / 2
    );
    let kind = match &w {
        PfCoopW::Wq(f, _, _) => Some(*f),
        PfCoopW::WqAffine { .. } => Some(wk::gemm_coop_f16::WqFmt::MlxAffineGe16),
        PfCoopW::F16(_) => None,
    };
    ensure!(
        cp.tile == 16
            || matches!(&w, PfCoopW::F16(_))
            || matches!(&w, PfCoopW::WqAffine { .. }),
        "{label}: the wq16 staging arms need 16x16 fragments; this plan is tile {}          ({APPLE_COOP_IS_8X8_SO_THE_AFFINE_PREFILL_RIDES_AN_F16_TWIN_NOT_THE_WQ16_STAGING})",
        cp.tile
    );
    let wq8 = cp.tile == 8 && matches!(&w, PfCoopW::WqAffine { .. });
    if wq8 {
        ensure!(
            alpha == 1.0,
            "{label}: the wq8 affine arm stores raw f32 rows and folds no alpha"
        );
    }
    let (tm, tn, sg, ku) = match pf_gemm_route_cfg(cp.m_alloc, kind, n, k) {
        Some(cfg) if cp.tile == 16 => cfg,
        _ if cp.tile == 8 => {
            let tn = [4u32, 2, 1]
                .into_iter()
                .find(|t| {
                    (n as u32).div_ceil(8 * t * cp.sg) >= COOP8_TN_SHRINKS_UNTIL_96_COLUMN_BANDS
                })
                .unwrap_or(1);
            (cp.tm, tn, cp.sg, pf_coop_ku(k, tn, cp.sg))
        }
        _ => (cp.tm, cp.tn, cp.sg, pf_coop_ku(k, cp.tn, cp.sg)),
    };
    let y16 = !wq8 && !matches!(&w, PfCoopW::F16(_)) && cp.m_alloc as usize == pf.m;
    let fmt_of = |w: &PfCoopW<'_>| match w {
        PfCoopW::Wq(f, _, _) => *f,
        PfCoopW::WqAffine { .. } => wk::gemm_coop_f16::WqFmt::MlxAffineGe16,
        PfCoopW::F16(_) => unreachable!("the F16 arm builds its own source below"),
    };
    let (src, entry) = match &w {
        _ if wq8 => (
            cf::source_wq8_affine(tm, tn, sg, ku),
            cf::entry_wq8_affine(tm, tn, sg, ku),
        ),
        PfCoopW::F16(_) => {
            let g = cf::CoopGemm::new(cp.tile, cf::Operand::F16);
            (g.source(tm, tn, sg, ku), g.entry(tm, tn, sg, ku))
        }
        w_ if y16 => (
            cf::source_wq16_act_y16(fmt_of(w_), cf::WqAct::F16, tm, tn, sg, ku),
            cf::entry_wq16_act_y16(fmt_of(w_), cf::WqAct::F16, tm, tn, sg, ku),
        ),
        w_ => (
            cf::source_wq16(fmt_of(w_), tm, tn, sg, ku),
            cf::entry_wq16(fmt_of(w_), tm, tn, sg, ku),
        ),
    };
    let cols = cp.tile * tn * sg;
    let bn = (n as u32).div_ceil(cols);
    let bm = cp.m_alloc / (cp.tile * tm);
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
            pad1: match &w {
                PfCoopW::WqAffine { gs, .. } => *gs as u32,
                _ => 0,
            },
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
        if let PfCoopW::WqAffine {
            w: wq,
            scales,
            biases,
            ..
        } = &w
        {
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
                    (5, *scales),
                    (
                        wk::gemm_coop_f16::MLX_AFFINE_BIAS_SLOT_7_BECAUSE_6_IS_THE_QUANT_ACT_SCALE,
                        *biases,
                    ),
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
        PfCoopW::WqAffine {
            w: wq,
            scales,
            biases,
            ..
        } => b.push_pf(
            label,
            &src,
            &entry,
            &[
                (0, *wq),
                (1, &cp.x_f16),
                (2, &cp.y_f32),
                (3, &p),
                (4, &cp.zero),
                (5, *scales),
                (
                    wk::gemm_coop_f16::MLX_AFFINE_BIAS_SLOT_7_BECAUSE_6_IS_THE_QUANT_ACT_SCALE,
                    *biases,
                ),
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
    x: &Buffer,
    k: usize,
    consumers: &[&DenseLinGpu],
) -> Result<Option<QuantIn>> {
    b.pf_dst_vf = pf.vf;
    let mut global: Option<f32> = None;
    for c in consumers {
        let Some(g) = c.input_global() else { continue };
        match global {
            None => global = Some(g),
            Some(prev) => ensure!(
                (prev - g).abs() <= 1e-6,
                "{label}: consumers disagree on input_global_scale ({prev} vs {g})"
            ),
        }
    }
    let Some(global) = global else {
        return Ok(None);
    };
    ensure!(
        pf.m <= 16,
        "{label}: the m-row nvfp4 quant/gemv routes (slotshared included) carry per-slot state \
         sized for m 2..=16; m={} runs only through the NV_Q3D_PF_COOP coop route, which does \
         not quantize activations",
        pf.m
    );
    let m = pf.m;
    let k_blocks = k / NVFP4_BLOCK;
    ensure!(
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
    let xm = {
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
    y: &Buffer,
    y_stride_words: usize,
) -> Result<()> {
    b.pf_dst_vf = pf.vf;
    ensure!(
        pf.m <= 16,
        "{label}: the nvfp4 m-row gemv routes dispatch one z-slice per slot and were only ever \
         built for m 2..=16; m={} runs only through the NV_Q3D_PF_COOP coop route",
        pf.m
    );
    ensure!(
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
        let mut binds: Vec<(u32, &Buffer)> = vec![
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
                anyhow!("{label}: slotshared route without the i8-mapped x buffer")
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

fn push_gemm_pf_affine(
    b: &mut Builder,
    pf: &Pf,
    label: &str,
    w: &AffineW4Gpu,
    x: &Buffer,
    x_stride_words: usize,
    y: &Buffer,
    y_stride_words: usize,
) -> Result<()> {
    b.pf_dst_vf = pf.vf;
    ensure!(
        b.ctx.caps.subgroup,
        "{label}: gemm_w4a16_affine rides subgroup reductions with no subgroup-free twin, and \
         this device grants no subgroups"
    );
    wk::gemm_w4a16_affine::m_rule(pf.m).map_err(|e| anyhow!("{label}: {e}"))?;
    wk::gemm_w4a16_affine::grain_rule(w.gs).map_err(|e| anyhow!("{label}: {e}"))?;
    ensure!(
        w.n.is_multiple_of(2) && x_stride_words.is_multiple_of(4),
        "{label}: mrows_pk packs bf16 row pairs (n {} even) and reads x as vec4 (stride {} % 4)",
        w.n,
        x_stride_words
    );
    let grid = b.grid1(w.n as u64, wk::gemm_w4a16_affine::MROWS_V2_ROWS_PER_GROUP);
    let p = b.uni(
        label,
        GemmW4A16AffineParams {
            n_rows: w.n as u32,
            k_elems: w.k as u32,
            gs: w.gs as u32,
            w_row_words: (w.k / 8) as u32,
            scale_row_stride: (w.k / w.gs) as u32,
            groups_x: grid.0,
            m_live: pf.m as u32,
            x_stride_words: x_stride_words as u32,
            y_stride_words: y_stride_words as u32,
        },
    );
    b.push_pf(
        label,
        &wk::gemm_w4a16_affine::source(),
        wk::gemm_w4a16_affine::MROWS_V2_PK_ENTRY_STAGES_X_IN_THREADGROUP_FOR_64_ROWS,
        &[
            (0, &w.w),
            (1, &w.scales),
            (2, x),
            (3, y),
            (4, &p),
            (wk::gemm_w4a16_affine::BIAS_SLOT, &w.biases),
        ],
        (grid.0, grid.1, 1),
    )
}

fn push_gemm_pf_dense(
    b: &mut Builder,
    s: &Sources,
    pf: &Pf,
    label: &str,
    w: &DenseLinGpu,
    x: &Buffer,
    x_stride_words: usize,
    q: Option<&QuantIn>,
    y: &Buffer,
    y_stride_words: usize,
) -> Result<()> {
    b.pf_dst_vf = pf.vf;
    if pf_coop_covers(pf, w) {
        return match w {
            DenseLinGpu::AffineW4(w) => {
                let tile16 = pf.coop.as_ref().is_some_and(|cp| cp.tile == 16);
                let arm = if tile16 || pf_coop_affine_wq8_enabled() {
                    PfCoopW::WqAffine {
                        w: &w.w,
                        scales: &w.scales,
                        biases: &w.biases,
                        gs: w.gs,
                    }
                } else {
                    PfCoopW::F16(w.f16_twin.as_ref().expect(
                        "pf_coop_covers admits a tile-8 affine only when the f16 twin was uploaded",
                    ))
                };
                pf_coop_gemm(b, pf, label, arm, w.n, w.k, 1.0, y, y_stride_words)
            }
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
                bail!("{label}: pf_coop_covers never selects bf16/int8 weights")
            }
        };
    }
    match w {
        DenseLinGpu::AffineW4(w) => {
            push_gemm_pf_affine(b, pf, label, w, x, x_stride_words, y, y_stride_words)
        }
        DenseLinGpu::Bf16(w) => push_gemm_pf(b, pf, label, w, x, x_stride_words, y, y_stride_words),
        DenseLinGpu::Fp8(f, w) => {
            if pf_fp8_replay_enabled() || w.is_none() && pf.m <= 16 {
                push_gemv_fp8_rowscale_pf(b, s, pf, label, f, x, x_stride_words, y, y_stride_words)
            } else {
                let w = w.as_ref().ok_or_else(|| {
                    anyhow!(
                        "{label}: the fp8 bf16 twin was not uploaded because the coop route was \
                         selected at upload time, yet the coop route does not cover this dispatch"
                    )
                })?;
                push_gemm_pf(b, pf, label, w, x, x_stride_words, y, y_stride_words)
            }
        }
        DenseLinGpu::Nvfp4(w) => {
            let q = q.ok_or_else(|| anyhow!("{label}: nvfp4 M-row gemm without quantized x"))?;
            push_gemm_pf_nvfp4(b, s, pf, label, w, q, y, y_stride_words)
        }
        DenseLinGpu::Int8(w) => {
            push_gemm_pf_i8(b, pf, label, w, x, x_stride_words, y, y_stride_words)
        }
    }
}

fn push_gemm_pf_i8(
    b: &mut Builder,
    pf: &Pf,
    label: &str,
    w: &I8Gpu,
    x: &Buffer,
    x_stride_words: usize,
    y: &Buffer,
    y_stride_words: usize,
) -> Result<()> {
    b.pf_dst_vf = pf.vf;
    ensure!(
        pf.m <= 16,
        "{label}: q3d_gemv_i8*_m<m> keeps one accumulator register per row and is generated \
         only for m 2..=16; m={} needs the NV_Q3D_PF_COOP coop route, which does not carry \
         int8 weights",
        pf.m
    );
    ensure!(
        wk::gemv_nvfp4_v2::subgroup32_ok(b.ctx),
        "{label}: the i8 M-row GEMV reduces with subgroupShuffleXor over exactly 32 lanes, the \
         same predicate that gates the per-token int8 entries; upload_lin only produces Int8 \
         weights where subgroup32_ok holds, so portable adapters take the nvfp4 arm and must \
         never reach this dispatch"
    );
    ensure!(
        w.k.is_multiple_of(4) && x_stride_words * 2 == w.k,
        "{label}: i8 M-row gemm needs k % 4 == 0 and an x row stride of k/2 packed-bf16 words; \
         got k={} x_stride_words={x_stride_words}",
        w.k
    );
    ensure!(
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

fn push_gemm_pf(
    b: &mut Builder,
    pf: &Pf,
    label: &str,
    w: &Bf16Gpu,
    x: &Buffer,
    x_stride_words: usize,
    y: &Buffer,
    y_stride_words: usize,
) -> Result<()> {
    use nv_kernels::wgpu_backend::na_bf16;
    b.pf_dst_vf = pf.vf;
    ensure!(
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

fn build_delta(
    b: &mut Builder,
    s: &Sources,
    cfg: &Qwen3_5DenseConfig,
    d: &HostDeltaNet,
    dfp8: &DeltaFp8,
    daff: &DeltaAffineW4,
    x: &Buffer,
    out: &Buffer,
    states: &mut Vec<(Buffer, u64)>,
    pfs: &[Pf],
    lora: Option<&LoraSiteCtx>,
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
    let coop_on = pfs.first().is_some_and(|p| p.coop.is_some());
    let dn_coop = |g: &Option<Fp8Gpu>| {
        coop_on
            && !pf_fp8_replay_enabled()
            && g.as_ref().is_some_and(|f| {
                f.s_plain.is_some() && f.n.is_multiple_of(16) && f.k.is_multiple_of(16)
            })
    };

    let g_qkv = dfp8.qkv.as_ref().map(|f| upload_fp8(b, "q3d-dn-qkv-fp8", f));
    let g_z = dfp8.z.as_ref().map(|f| upload_fp8(b, "q3d-dn-z-fp8", f));
    let g_out = dfp8.out.as_ref().map(|f| upload_fp8(b, "q3d-dn-oproj-fp8", f));
    let dn_requant = requant_fp8_group_enabled("dn");
    let requant_dn_lin = |f: &Option<HostFp8Lin>| -> Result<Option<HostNvfp4Lin>> {
        match f {
            Some(f)
                if dn_requant && f.n.is_multiple_of(128) && f.k.is_multiple_of(NVFP4_BLOCK * 4) =>
            {
                Ok(Some(requant_fp8_rowscale_to_nvfp4(
                    &f.packed, &f.scales, f.n, f.k,
                )?))
            }
            _ => Ok(None),
        }
    };
    let nv_qkv = requant_dn_lin(&dfp8.qkv)?
        .map(|l| DenseLinGpu::Nvfp4(upload_nvfp4(b, "q3d-dn-qkv-nv4", &l)));
    let nv_z =
        requant_dn_lin(&dfp8.z)?.map(|l| DenseLinGpu::Nvfp4(upload_nvfp4(b, "q3d-dn-z-nv4", &l)));
    let nv_out = requant_dn_lin(&dfp8.out)?
        .map(|l| DenseLinGpu::Nvfp4(upload_nvfp4(b, "q3d-dn-oproj-nv4", &l)));
    let qkv_coop = dn_coop(&g_qkv);
    let z_coop = dn_coop(&g_z);
    let out_coop = dn_coop(&g_out);
    let dn_affine_twin_for_the_tile8_coop_route =
        coop_on && pf_coop_tile(b.ctx) == Some(8) && !pf_coop_affine_wq8_enabled();
    let dn_affine_coop_route = coop_on
        && pf_coop_tile(b.ctx) == Some(8)
        && (dn_affine_twin_for_the_tile8_coop_route || pf_coop_affine_wq8_enabled());
    let a_qkv = daff.qkv.as_ref().map(|l| {
        upload_affine_w4(
            b,
            "q3d-dn-qkv-aff",
            l,
            dn_affine_twin_for_the_tile8_coop_route,
        )
    });
    let a_z = daff.z.as_ref().map(|l| {
        upload_affine_w4(
            b,
            "q3d-dn-z-aff",
            l,
            dn_affine_twin_for_the_tile8_coop_route,
        )
    });
    let a_out = daff.out.as_ref().map(|l| {
        upload_affine_w4(
            b,
            "q3d-dn-oproj-aff",
            l,
            dn_affine_twin_for_the_tile8_coop_route,
        )
    });
    let w_qkv = if qkv_coop || a_qkv.is_some() {
        None
    } else {
        Some(upload_bf16(b, "q3d-dn-qkv", &d.in_proj_qkv))
    };
    let w_z = if z_coop || a_z.is_some() {
        None
    } else {
        Some(upload_bf16(b, "q3d-dn-z", &d.in_proj_z))
    };
    let w_out = if out_coop || a_out.is_some() {
        None
    } else {
        Some(upload_bf16(b, "q3d-dn-out", &d.out_proj))
    };
    let w_ab = upload_bf16(b, "q3d-dn-ab", &d.in_proj_ab);
    let w_ab_f16 = if coop_on && (2 * n_v).is_multiple_of(16) && hidden.is_multiple_of(16) {
        Some(b.upload_u32("q3d-dn-ab-f16", &pack_f16_pairs_from_bf16(&d.in_proj_ab.w)))
    } else {
        None
    };

    let qkv = b.zeros("q3d-dn-qkvbuf", (conv_dim * 2) as u64);
    let z = b.zeros("q3d-dn-zbuf", (value_dim * 2) as u64);
    let ab = b.zeros("q3d-dn-abbuf", (2 * n_v * 2).max(4) as u64);
    let merged_gemv = nv_qkv.is_none()
        && nv_z.is_none()
        && g_qkv.is_some()
        && g_z.is_some()
        && b.ctx.caps.max_storage_buffers_per_shader_stage
            >= DN_MERGED_GEMV_BINDS_NINE_STORAGE_BUFFERS;
    let lora_dn_in = lora.and_then(|lc| lc.layer.dn_in.as_ref().map(|site| (lc, site)));
    let mut dn_in_folded = false;
    if merged_gemv {
        let shrink = match lora_dn_in {
            Some((lc, site)) if lora_gemv_fold_attn_dn_enabled() => {
                Some((lc.push_site_shrink(b, site, x), site))
            }
            _ => None,
        };
        dn_in_folded = shrink.is_some();
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
            shrink
                .as_ref()
                .map(|(f, site)| DnMergedLoraFold { fold: f, site }),
        )?;
    } else {
        let x_consumers: Vec<&DenseLinGpu> = nv_qkv.iter().chain(nv_z.iter()).collect();
        let dn_q = if x_consumers.is_empty() {
            None
        } else {
            quant_for(b, s, "q3d-dn-quant", x, hidden, &x_consumers)?
        };
        match (&a_qkv, &nv_qkv, &g_qkv, &w_qkv) {
            (Some(a), _, _, _) => push_gemv_affine_w4_into_pairs(b, "q3d-dn-qkv", a, x, &qkv)?,
            (None, Some(nv), _, _) => push_gemv(b, s, "q3d-dn-qkv", nv, x, dn_q.as_ref(), &qkv)?,
            (None, None, Some(g), _) => {
                push_gemv_fp8_rowscale(b, s, "q3d-dn-qkv", g, x, &qkv, false)?
            }
            (None, None, None, Some(w)) => push_gemv_bf16(b, s, "q3d-dn-qkv", w, x, &qkv, false)?,
            (None, None, None, None) => {
                bail!("q3d-dn-qkv: neither affine, fp8 nor bf16 weights uploaded")
            }
        }
        match (&a_z, &nv_z, &g_z, &w_z) {
            (Some(a), _, _, _) => push_gemv_affine_w4_into_pairs(b, "q3d-dn-z", a, x, &z)?,
            (None, Some(nv), _, _) => push_gemv(b, s, "q3d-dn-z", nv, x, dn_q.as_ref(), &z)?,
            (None, None, Some(g), _) => push_gemv_fp8_rowscale(b, s, "q3d-dn-z", g, x, &z, false)?,
            (None, None, None, Some(w)) => push_gemv_bf16(b, s, "q3d-dn-z", w, x, &z, false)?,
            (None, None, None, None) => {
                bail!("q3d-dn-z: neither affine, fp8 nor bf16 weights uploaded")
            }
        }
        push_gemv_bf16(b, s, "q3d-dn-ab", &w_ab, x, &ab, false)?;
    }
    if !dn_in_folded {
        if let Some((lc, site)) = lora_dn_in {
            lc.push_site(b, site, x, &[&qkv, &z, &ab]);
        }
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
    let fused_head = d_k <= wg && d_v <= wg && d_v.is_multiple_of(2);
    let wide_head = fused_head
        && dn_head_wide_enabled()
        && d_k == DN_HEAD_WIDE_REQUIRES_DK_128
        && d_v == DN_HEAD_WIDE_REQUIRES_DV_128;
    let (dn_head_label, dn_head_entry) = if wide_head {
        ("q3d-dn-head-fused-wide", "q3w_delta_head_fused_wide")
    } else {
        ("q3d-dn-head-fused", "q3w_delta_head_fused")
    };
    let split_head = fused_head
        && d_k == 128
        && d_v == 128
        && env_arm_opt_in_eq_some_1_reread_every_call("NV_Q3D_DN_HEAD_SPLIT");
    if split_head {
        let core = b.zeros("q3d-dn-core", (n_v * d_v * 4) as u64);
        b.push(
            "q3d-dn-head-split",
            &s.delta,
            "q3w_delta_head_split_dv32",
            &[
                (10, &mixed),
                (14, &dqp),
                (20, &ab),
                (21, &alog),
                (22, &dtb),
                (25, &gp),
                (35, &core),
                (36, &state),
            ],
            (n_v as u32, 4, 1),
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
    } else if fused_head {
        b.push(
            dn_head_label,
            &s.delta,
            dn_head_entry,
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
        let rec_grid_y = (d_v as u32).div_ceil(DELTA_RECURRENT_LANES_PER_WORKGROUP);
        b.push(
            "q3d-dn-recurrent",
            &s.delta,
            DELTA_RECURRENT_ENTRY,
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

    let lora_dn_out = lora.and_then(|lc| lc.layer.dn_out.as_ref().map(|site| (lc, site)));
    let mut dn_out_folded = false;
    match (&a_out, &nv_out, &g_out, &w_out) {
        (Some(a), _, _, _) => push_gemv_affine_w4_into_pairs(b, "q3d-dn-oproj", a, &gated, out)?,
        (None, Some(nv), _, _) => {
            let out_q = quant_for(b, s, "q3d-dn-oquant", &gated, value_dim, &[nv])?;
            push_gemv(b, s, "q3d-dn-oproj", nv, &gated, out_q.as_ref(), out)?;
        }
        (None, None, Some(g), _) => match lora_dn_out {
            Some((lc, site)) if lora_gemv_fold_attn_dn_enabled() => {
                let f = lc.push_site_shrink(b, site, &gated);
                push_gemv_fp8_rowscale_lora(
                    b,
                    s,
                    "q3d-dn-oproj",
                    g,
                    &gated,
                    out,
                    false,
                    Some(LoraFoldSlice {
                        fold: &f,
                        site,
                        slice_id: 0,
                    }),
                )?;
                dn_out_folded = true;
            }
            _ => push_gemv_fp8_rowscale(b, s, "q3d-dn-oproj", g, &gated, out, false)?,
        },
        (None, None, None, Some(w)) => push_gemv_bf16(b, s, "q3d-dn-oproj", w, &gated, out, false)?,
        (None, None, None, None) => {
            bail!("q3d-dn-oproj: neither affine, fp8 nor bf16 weights uploaded")
        }
    }
    if !dn_out_folded {
        if let Some((lc, site)) = lora_dn_out {
            lc.push_site(b, site, &gated, &[out]);
        }
    }

    for pf in pfs {
        b.pf_dst_vf = pf.vf;
        let hidden_words = cfg.hidden_size / 2;
        let conv_words = conv_dim / 2;
        let value_words = value_dim / 2;
        let m = pf.m as u32;
        let commit_replay = pf.vf;
        let mu = pf.m as u64;
        let (d_qkv_b, d_q_b, d_k_b, d_v_b, d_g_b, d_beta_b) = if commit_replay {
            (
                b.zeros("q3d-vfc-dn-qkv", mu * (conv_dim * 2) as u64),
                b.zeros("q3d-vfc-dn-q", mu * (n_v * d_k * 4) as u64),
                b.zeros("q3d-vfc-dn-k", mu * (n_v * d_k * 4) as u64),
                b.zeros("q3d-vfc-dn-v", mu * (n_v * d_v * 4) as u64),
                b.zeros("q3d-vfc-dn-g", mu * (n_v * 4) as u64),
                b.zeros("q3d-vfc-dn-beta", mu * (n_v * 4) as u64),
            )
        } else {
            (
                pf.d_qkv.clone(),
                pf.d_q.clone(),
                pf.d_k.clone(),
                pf.d_v.clone(),
                pf.d_g.clone(),
                pf.d_beta.clone(),
            )
        };
        let pf_on_coop = pf.coop.is_some();
        let dn_affine_coop = dn_affine_coop_route && (a_qkv.is_some() || a_z.is_some());
        if pf_on_coop && (qkv_coop || z_coop || dn_affine_coop || w_ab_f16.is_some()) {
            pf_coop_cast_x(b, pf, "q3d-pf-dn-castx", &pf.normed, hidden_words, hidden)?;
        }
        match (&g_qkv, pf_fp8_replay_enabled() || (!pf_on_coop && qkv_coop)) {
            (Some(g), true) => push_gemv_fp8_rowscale_pf(
                b,
                s,
                pf,
                "q3d-pf-dn-qkv-fp8r",
                g,
                &pf.normed,
                hidden_words,
                &d_qkv_b,
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
                &d_qkv_b,
                conv_words,
            )?,
            _ => match &a_qkv {
                Some(aw) if pf_on_coop && dn_affine_coop_route => pf_coop_gemm(
                    b,
                    pf,
                    "q3d-pf-dn-qkv-coop-aff",
                    dn_affine_coop_arm(aw),
                    conv_dim,
                    hidden,
                    1.0,
                    &d_qkv_b,
                    conv_words,
                )?,
                Some(aw) => push_gemm_pf_affine(
                    b,
                    pf,
                    "q3d-pf-dn-qkv-aff",
                    aw,
                    &pf.normed,
                    hidden_words,
                    &d_qkv_b,
                    conv_words,
                )?,
                None => push_gemm_pf(
                    b,
                    pf,
                    "q3d-pf-dn-qkv",
                    w_qkv.as_ref().expect("no coop route, twin uploaded"),
                    &pf.normed,
                    hidden_words,
                    &d_qkv_b,
                    conv_words,
                )?,
            },
        }
        match (&g_z, pf_fp8_replay_enabled() || (!pf_on_coop && z_coop)) {
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
            _ => match &a_z {
                Some(aw) if pf_on_coop && dn_affine_coop_route => pf_coop_gemm(
                    b,
                    pf,
                    "q3d-pf-dn-z-coop-aff",
                    dn_affine_coop_arm(aw),
                    value_dim,
                    hidden,
                    1.0,
                    &pf.d_z,
                    value_words,
                )?,
                Some(aw) => push_gemm_pf_affine(
                    b,
                    pf,
                    "q3d-pf-dn-z-aff",
                    aw,
                    &pf.normed,
                    hidden_words,
                    &pf.d_z,
                    value_words,
                )?,
                None => push_gemm_pf(
                    b,
                    pf,
                    "q3d-pf-dn-z",
                    w_z.as_ref().expect("no coop route, twin uploaded"),
                    &pf.normed,
                    hidden_words,
                    &pf.d_z,
                    value_words,
                )?,
            },
        }
        match w_ab_f16.as_ref().filter(|_| pf_on_coop) {
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
        if let Some(lc) = lora {
            if let Some(site) = &lc.layer.dn_in {
                lc.push_site_pf(b, pf.m, site, &pf.normed, &[&d_qkv_b, &pf.d_z, &pf.d_ab]);
            }
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
                (10, &d_qkv_b),
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
            &[(10, &d_qkv_b), (12, &conv_state), (14, &cpm), (15, &pf.ck)],
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
                (21, &d_q_b),
                (22, &d_k_b),
                (23, &d_v_b),
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
                (33, &d_g_b),
                (34, &d_beta_b),
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
            && !env_arm_opt_in_eq_some_1_reread_every_call(
                PF_SCAN_WY_ENV_1_ALSO_ROUTES_SMALL_M_GRAPHS_THROUGH_THE_WY_SCAN,
            ) {
            ("q3w_delta_scan", 1)
        } else {
            pf_scan_wy_route(d_k, d_v, b.ctx.caps.max_compute_workgroup_storage_size)
        };
        let wy_v2 = pf
            .d_wy_ckpt
            .as_ref()
            .filter(|_| scan_entry == "q3w_delta_scan_wy");
        if let Some((ck_cl, ck_s)) = wy_v2 {
            let chunks = m.div_ceil(32);
            b.push_pf(
                "q3d-pf-dn-scan",
                &s.prefill,
                "q3w_delta_scan_wy_ck",
                &[
                    (39, ck_cl),
                    (41, &d_k_b),
                    (42, &d_v_b),
                    (43, &d_g_b),
                    (44, &d_beta_b),
                    (45, &pf.d_core),
                    (46, &state),
                    (47, &rpm),
                    (48, &pf.ck),
                    (49, ck_s),
                ],
                (n_v as u32, scan_grid_y, 1),
            )?;
            b.push_pf(
                "q3d-pf-dn-scan-o",
                &s.prefill,
                "q3w_delta_scan_wy_o",
                &[
                    (39, ck_cl),
                    (40, &d_q_b),
                    (41, &d_k_b),
                    (45, &pf.d_core),
                    (47, &rpm),
                    (48, &pf.ck),
                    (49, ck_s),
                ],
                (n_v as u32, scan_grid_y, chunks),
            )?;
        } else {
            b.push_pf(
                "q3d-pf-dn-scan",
                &s.prefill,
                scan_entry,
                &[
                    (40, &d_q_b),
                    (41, &d_k_b),
                    (42, &d_v_b),
                    (43, &d_g_b),
                    (44, &d_beta_b),
                    (45, &pf.d_core),
                    (46, &state),
                    (47, &rpm),
                    (48, &pf.ck),
                ],
                (n_v as u32, scan_grid_y, 1),
            )?;
        }
        if commit_replay {
            let shift = b.make(
                "q3d-vfc-dn-shift",
                &s.prefill,
                "q3w_delta_conv_shift",
                &[(10, &d_qkv_b), (12, &conv_state), (14, &cpm), (15, &pf.ck)],
                cgrid,
            )?;
            b.vf_commit_passes.push(shift);
            if let Some((ck_cl, ck_s)) = wy_v2 {
                let chunks = m.div_ceil(32);
                let scan = b.make(
                    "q3d-vfc-dn-scan",
                    &s.prefill,
                    "q3w_delta_scan_wy_ck",
                    &[
                        (39, ck_cl),
                        (41, &d_k_b),
                        (42, &d_v_b),
                        (43, &d_g_b),
                        (44, &d_beta_b),
                        (45, &pf.d_core),
                        (46, &state),
                        (47, &rpm),
                        (48, &pf.ck),
                        (49, ck_s),
                    ],
                    (n_v as u32, scan_grid_y, 1),
                )?;
                let scan_o = b.make(
                    "q3d-vfc-dn-scan-o",
                    &s.prefill,
                    "q3w_delta_scan_wy_o",
                    &[
                        (39, ck_cl),
                        (40, &d_q_b),
                        (41, &d_k_b),
                        (45, &pf.d_core),
                        (47, &rpm),
                        (48, &pf.ck),
                        (49, ck_s),
                    ],
                    (n_v as u32, scan_grid_y, chunks),
                )?;
                b.vf_commit_passes.push(scan);
                b.vf_commit_passes.push(scan_o);
            } else {
                let scan = b.make(
                    "q3d-vfc-dn-scan",
                    &s.prefill,
                    scan_entry,
                    &[
                        (40, &d_q_b),
                        (41, &d_k_b),
                        (42, &d_v_b),
                        (43, &d_g_b),
                        (44, &d_beta_b),
                        (45, &pf.d_core),
                        (46, &state),
                        (47, &rpm),
                        (48, &pf.ck),
                    ],
                    (n_v as u32, scan_grid_y, 1),
                )?;
                b.vf_commit_passes.push(scan);
            }
        }
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
        match (&g_out, pf_fp8_replay_enabled() || (!pf_on_coop && out_coop)) {
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
            _ => match &a_out {
                Some(aw) if pf_on_coop && dn_affine_coop_route => {
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
                        "q3d-pf-dn-oproj-coop-aff",
                        dn_affine_coop_arm(aw),
                        cfg.hidden_size,
                        value_dim,
                        1.0,
                        &pf.mix,
                        hidden_words,
                    )?
                }
                Some(aw) => push_gemm_pf_affine(
                    b,
                    pf,
                    "q3d-pf-dn-oproj-aff",
                    aw,
                    &pf.d_gated,
                    value_words,
                    &pf.mix,
                    hidden_words,
                )?,
                None => push_gemm_pf(
                    b,
                    pf,
                    "q3d-pf-dn-oproj",
                    w_out.as_ref().expect("no coop route, twin uploaded"),
                    &pf.d_gated,
                    value_words,
                    &pf.mix,
                    hidden_words,
                )?,
            },
        }
        if let Some(lc) = lora {
            if let Some(site) = &lc.layer.dn_out {
                lc.push_site_pf(b, pf.m, site, &pf.d_gated, &[&pf.mix]);
            }
        }
    }
    Ok(())
}

fn build_attn(
    b: &mut Builder,
    s: &Sources,
    cfg: &Qwen3_5DenseConfig,
    a: &HostDenseAttention,
    x: &Buffer,
    out: &Buffer,
    pos_buf: &Buffer,
    fd_buf: &Buffer,
    max_seq: usize,
    states: &mut Vec<(Buffer, u64)>,
    pfs: &[Pf],
    rope_rows: &mut Vec<(Buffer, Buffer)>,
    kv_caches: &mut Vec<(Buffer, Buffer)>,
    dual_stage1: bool,
    lora: Option<&LoraSiteCtx>,
) -> Result<usize> {
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let rot = cfg.rotary_dim();
    ensure!(
        rot.is_multiple_of(2) && rot <= hd,
        "rotary_dim {rot} invalid"
    );

    let pf_rides_coop = pfs.first().is_none_or(|p| p.coop.is_some());
    let wq = upload_lin(b, "q3d-at-wq", &a.q, W8Scope::Attn, pf_rides_coop);
    let wk_ = upload_lin(b, "q3d-at-wk", &a.k, W8Scope::Attn, pf_rides_coop);
    let wv = upload_lin(b, "q3d-at-wv", &a.v, W8Scope::Attn, pf_rides_coop);
    let wo = upload_lin(b, "q3d-at-wo", &a.o, W8Scope::Attn, pf_rides_coop);

    let qkv_q = quant_for(b, s, "q3d-at-quant", x, wq.k(), &[&wq, &wk_, &wv])?;
    let q_raw = b.zeros("q3d-at-qraw", (wq.n() * 2) as u64);
    let k_raw = b.zeros("q3d-at-kraw", (wk_.n() * 2) as u64);
    let v_raw = b.zeros("q3d-at-vraw", (wv.n() * 2) as u64);
    let qkv_fused_arm = env_arm_opt_in_eq_some_1_reread_every_call(
        "NV_Q3D_QKV_FUSED_E2E_NULL_METAL_ALREADY_OVERLAPS_HAZARD_FREE_NEIGHBORS",
    );
    if let Some(fused) = qkv_fused_arm
        .then(|| fuse_affine_qkv_hosts(&a.q, &a.k, &a.v))
        .flatten()
    {
        let nq = a.q.n();
        let nk = a.k.n();
        let fw = upload_affine_w4(b, "q3d-at-wqkv-fused", &fused, false);
        push_gemv_affine_w4_qkv_pairs(b, "q3d-at-qkvproj", &fw, nq, nk, x, &q_raw, &k_raw, &v_raw)?;
        if let Some(lc) = lora {
            if let Some(site) = &lc.layer.qkv {
                lc.push_site(b, site, x, &[&q_raw, &k_raw, &v_raw]);
            }
        }
    } else {
        let lora_qkv = lora.and_then(|lc| lc.layer.qkv.as_ref().map(|site| (lc, site)));
        let qkv_folded = match (lora_qkv, &wq, &wk_, &wv) {
            (
                Some((lc, site)),
                DenseLinGpu::Fp8(gq, _),
                DenseLinGpu::Fp8(gk, _),
                DenseLinGpu::Fp8(gv, _),
            ) if lora_gemv_fold_attn_dn_enabled() => {
                let f = lc.push_site_shrink(b, site, x);
                for (label, g, y, slice_id) in [
                    ("q3d-at-qproj", gq, &q_raw, 0usize),
                    ("q3d-at-kproj", gk, &k_raw, 1),
                    ("q3d-at-vproj", gv, &v_raw, 2),
                ] {
                    push_gemv_fp8_rowscale_lora(
                        b,
                        s,
                        label,
                        g,
                        x,
                        y,
                        false,
                        Some(LoraFoldSlice {
                            fold: &f,
                            site,
                            slice_id,
                        }),
                    )?;
                }
                true
            }
            _ => false,
        };
        if !qkv_folded {
            push_gemv(b, s, "q3d-at-qproj", &wq, x, qkv_q.as_ref(), &q_raw)?;
            push_gemv(b, s, "q3d-at-kproj", &wk_, x, qkv_q.as_ref(), &k_raw)?;
            push_gemv(b, s, "q3d-at-vproj", &wv, x, qkv_q.as_ref(), &v_raw)?;
            if let Some((lc, site)) = lora_qkv {
                lc.push_site(b, site, x, &[&q_raw, &k_raw, &v_raw]);
            }
        }
    }

    let (cos, sin) = rope_tables(rot.max(2), cfg.rope_theta, max_seq);
    let cosb = b.upload_f32("q3d-at-cos", &cos);
    let sinb = b.upload_f32("q3d-at-sin", &sin);
    rope_rows.push((cosb.clone(), sinb.clone()));
    let kv_words = n_kv * hd / 2;
    let cache_bytes = (max_seq * kv_words * 4) as u64;
    let kc = b.zeros("q3d-at-kc", cache_bytes);
    let vc = b.zeros("q3d-at-vc", cache_bytes);
    states.push((kc.clone(), cache_bytes));
    states.push((vc.clone(), cache_bytes));
    kv_caches.push((kc.clone(), vc.clone()));
    let bias_mode = !a.qkv_bias_bf16.is_empty();
    let bias_writes_kv = bias_mode
        && !kv_fp8_enabled()
        && env_arm_opt_in_eq_some_1_reread_every_call(
            BIAS_ROPE_WRITES_KV_ENV_DROPS_THE_KVWRITE_PASS_LEAVES_ANY_FP8_PF_CACHE_STALE_PAST_THE_PROMPT_DEFAULT_OFF,
        );
    ensure!(
        !bias_mode || (a.q_norm.is_empty() && a.k_norm.is_empty() && !cfg.attn_output_gate),
        "qkv bias rides the qwen2 arch: a checkpoint carrying q/k/v projection biases must not \
         also carry q_norm/k_norm or an attention output gate"
    );
    ensure!(
        bias_mode || (!a.q_norm.is_empty() && !a.k_norm.is_empty()),
        "attention layer carries neither qk-norm weights nor qkv biases; the loader should have \
         rejected this checkpoint"
    );
    ensure!(
        !bias_mode || a.qkv_bias_bf16.len() == (n_h + 2 * n_kv) * hd,
        "qkv bias has {} elems, want q||k||v = {}",
        a.qkv_bias_bf16.len(),
        (n_h + 2 * n_kv) * hd
    );
    let (qn, kn) = if bias_mode {
        (None, None)
    } else {
        (
            Some(b.upload_u32("q3d-at-qn", &pack_pairs(&a.q_norm))),
            Some(b.upload_u32("q3d-at-kn", &pack_pairs(&a.k_norm))),
        )
    };
    let qkv_bias = if bias_mode {
        Some(b.upload_u32("q3d-at-qkvb", &pack_pairs(&a.qkv_bias_bf16)))
    } else {
        None
    };
    let attn_mega = bias_writes_kv
        && env_arm_opt_in_eq_some_1_reread_every_call(
            ATTN_SHALLOW_ENV_ONE_DISPATCH_PER_LAYER_REPLACES_FLASH1_FLASH2_GATE_HD128_BF16KV_GATELESS_DEFAULT_OFF,
        )
        && hd == 128
        && !cfg.attn_output_gate;
    let q = b.zeros("q3d-at-q", (n_h * hd * 2) as u64);
    let k = b.zeros("q3d-at-k", (n_kv * hd * 2) as u64);
    let src_stride_q = if cfg.attn_output_gate { 2 * hd } else { hd };
    let q_f32 = b.zeros("q3d-at-qf32", (n_h * hd * 4) as u64);
    let v_post = if bias_mode {
        b.zeros("q3d-at-vb", (wv.n() * 2) as u64)
    } else {
        v_raw.clone()
    };
    let fused_norm = hd <= MAX_HEAD_DIM
        && hd.is_multiple_of(2)
        && b.ctx.caps.max_storage_buffers_per_shader_stage
            >= ATTN_FUSED_NORM_BINDS_TEN_STORAGE_BUFFERS;
    if attn_mega {
    } else if bias_mode {
        let abp = b.uni(
            "q3d-at-bias-rope-p",
            BiasRopeFusedParams {
                n_q_rows: n_h as u32,
                n_k_rows: n_kv as u32,
                n_v_rows: n_kv as u32,
                head_dim: hd as u32,
                rot_half: (rot / 2) as u32,
                write_kv: u32::from(bias_writes_kv),
                ..Default::default()
            },
        );
        b.push(
            "q3d-at-bias-rope",
            &s.attn,
            "q3w_attn_bias_rope_qcast",
            &[
                (0, &q_raw),
                (2, &cosb),
                (3, &sinb),
                (4, pos_buf),
                (5, &q),
                (40, &k_raw),
                (42, &k),
                (43, &q_f32),
                (45, qkv_bias.as_ref().expect("bias_mode just uploaded it")),
                (46, &v_raw),
                (47, &v_post),
                (48, &abp),
                (56, &kc),
                (57, &vc),
            ],
            ((n_h + 2 * n_kv) as u32, 1, 1),
        )?;
    } else if fused_norm {
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
                (1, qn.as_ref().expect("norm arm requires qk-norm weights")),
                (2, &cosb),
                (3, &sinb),
                (4, pos_buf),
                (5, &q),
                (40, &k_raw),
                (41, kn.as_ref().expect("norm arm requires qk-norm weights")),
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
                (1, qn.as_ref().expect("norm arm requires qk-norm weights")),
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
                (1, kn.as_ref().expect("norm arm requires qk-norm weights")),
                (2, &cosb),
                (3, &sinb),
                (4, pos_buf),
                (5, &k),
                (6, &kp),
            ],
            (n_kv as u32, 1, 1),
        )?;
    }

    let kvp = b.uni(
        "q3d-at-kv-p",
        KvWriteParams {
            words: kv_words as u32,
            ..Default::default()
        },
    );
    let kv8_decode = kv_fp8_enabled();
    let tiled_pf = pfs.first().is_some_and(|p| p.attn_tiled.is_some());
    let fused_kvw = !bias_writes_kv
        && fuse_kvw_enabled()
        && (kv8_decode || tiled_pf)
        && b.ctx.caps.max_storage_buffers_per_shader_stage >= KVW_FUSED_BINDS_NINE_STORAGE_BUFFERS;
    if !fused_kvw && !bias_writes_kv {
        let grid = b.grid1(kv_words as u64, 64);
        b.push(
            "q3d-at-kvwrite",
            &s.attn,
            "q3w_kv_write",
            &[
                (10, &k),
                (11, &v_post),
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
                    (9, &v_post),
                    (10, &vc8),
                    (11, &vsc),
                    (12, &kc),
                    (13, &vc),
                ],
                (n_kv as u32, 2, 1),
            )?;
        } else if !bias_writes_kv {
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
                &[
                    (0, &v_post),
                    (1, &vc8),
                    (2, &vsc),
                    (3, pos_buf),
                    (4, &kvq_p),
                ],
                (n_kv as u32, 1, 1),
            )?;
        }
        Some((kc8, vc8, ksc, vsc, kvq_p))
    } else {
        None
    };
    let tp_capable = kv8_decode
        && kv_nvfp4_arm().is_none()
        && b.ctx.caps.subgroup
        && b.ctx.subgroup_width() == Some(32)
        && (n_h / n_kv).min(wk::flash_decode::MAX_GQA_FOLD) > 1;
    if kv_fp8_enabled() && !tp_capable {
        eprintln!(
            "[q3d-wgpu] the transposed-K tp stage1 needs the fp8 decode cache, no nvfp4 KV arm, \
             width-32 subgroups and a foldable GQA group; keeping the position-major stage1"
        );
    }
    let kv_tp = if kv_fp8_enabled() && tp_capable {
        let (_, _, ksc, _, kvq_p) = kv8
            .as_ref()
            .expect("kv8_decode just ensured the fp8 KV cache is allocated");
        let kt_bytes = (max_seq * n_kv * hd) as u64;
        let kc8t = b.zeros("q3d-at-kc8t", kt_bytes);
        assert_eq!(
            kc8t.size(),
            kt_bytes,
            "{}",
            wk::flash_decode::K_TRANSPOSED_BINDING_MUST_BE_EXACT
        );
        states.push((kc8t.clone(), kt_bytes));
        b.push(
            "q3d-at-kvq-kt",
            &s.kvq,
            wk::kv_fp8::QUANTIZE_ENTRY_KT,
            &[(0, &k), (1, &kc8t), (2, ksc), (3, pos_buf), (4, kvq_p)],
            (n_kv as u32, 1, 1),
        )?;
        Some(kc8t)
    } else {
        None
    };
    let kv8_dec = if kv8_decode { kv8.as_ref() } else { None };

    let kv4_arm = kv_nvfp4_arm();
    ensure!(
        kv4_arm.is_none() || kv8_decode,
        "{} rides the fp8 cache for its first {} sink slots and for chunk coherence, so it \
         cannot pair with NV_Q3D_KV_FP8=0; unset one",
        KV_NVFP4_ENV_OFF_BY_DEFAULT_FP8_STAYS_THE_SHIPPED_KV_DECODE_FORMAT,
        wk::kv_nvfp4::KV_NVFP4_SINK_SLOTS_STAY_FP8_TO_ANCHOR_SOFTMAX
    );
    let kv4 = if let Some(arm) = kv4_arm {
        ensure!(
            hd.is_multiple_of(8),
            "kv_nvfp4 packs 8 e2m1 nibbles per u32 word; head_dim {hd} % 8 != 0"
        );
        ensure!(
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
        let f = if nv_config::env_arm_present_at_all_reread_every_call(
            wk::flash_decode::GQA_FOLD_ENV,
        ) {
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
    let gated = b.zeros("q3d-at-gated", (n_h * hd * 2) as u64);
    let shallow_attn = env_arm_opt_in_eq_some_1_reread_every_call(
        ATTN_SHALLOW_ENV_ONE_DISPATCH_PER_LAYER_REPLACES_FLASH1_FLASH2_GATE_HD128_BF16KV_GATELESS_DEFAULT_OFF,
    ) && b.ctx.caps.subgroup
        && hd == 128
        && !cfg.attn_output_gate
        && kv8_dec.is_none()
        && kv4.is_none();
    ensure!(
        !attn_mega || shallow_attn,
        "the mega arm skipped the bias-rope pass expecting to rope in-attention, but the \
         shallow arm disqualified itself (an nvfp4/fp8 KV arm is set); unset it or drop {}",
        ATTN_SHALLOW_ENV_ONE_DISPATCH_PER_LAYER_REPLACES_FLASH1_FLASH2_GATE_HD128_BF16KV_GATELESS_DEFAULT_OFF
    );
    if shallow_attn {
        let asp = b.uni(
            "q3d-at-shallow-p",
            AttnShallowParams {
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                group: group as u32,
                rot_half: (rot / 2) as u32,
                scale: 1.0 / (hd as f32).sqrt(),
                splits,
                ..Default::default()
            },
        );
        if attn_mega && group == 6 {
            let fold_scratch = b.zeros(
                "q3d-at-fold-scr",
                (n_h * splits as usize * (hd + 2) * 4) as u64,
            );
            let bias = qkv_bias.as_ref().expect("bias_mode just uploaded it");
            b.push(
                "q3d-at-fold1",
                &s.attn,
                "q3w_attn_fold_split1",
                &[
                    (0, &q_raw),
                    (2, &cosb),
                    (3, &sinb),
                    (4, pos_buf),
                    (40, &k_raw),
                    (45, bias),
                    (46, &v_raw),
                    (55, &asp),
                    (56, &kc),
                    (57, &vc),
                    (58, &fold_scratch),
                ],
                (n_kv as u32, splits, 1),
            )?;
            b.push(
                "q3d-at-fold2",
                &s.attn,
                "q3w_attn_fold_merge_pk",
                &[(54, &gated), (55, &asp), (58, &fold_scratch)],
                (n_h as u32, 1, 1),
            )?;
        } else if attn_mega {
            b.push(
                "q3d-at-mega",
                &s.attn,
                "q3w_attn_bias_rope_decode_pk",
                &[
                    (0, &q_raw),
                    (2, &cosb),
                    (3, &sinb),
                    (4, pos_buf),
                    (40, &k_raw),
                    (45, qkv_bias.as_ref().expect("bias_mode just uploaded it")),
                    (46, &v_raw),
                    (54, &gated),
                    (55, &asp),
                    (56, &kc),
                    (57, &vc),
                ],
                (n_h as u32, 1, 1),
            )?;
        } else {
            b.push(
                "q3d-at-shallow",
                &s.attn,
                "q3w_attn_decode_shallow_pk",
                &[
                    (50, &q_f32),
                    (56, &kc),
                    (57, &vc),
                    (53, pos_buf),
                    (54, &gated),
                    (55, &asp),
                ],
                (n_h as u32, 1, 1),
            )?;
        }
    } else if let Some((k4, v4w, v4s)) = kv4.as_ref() {
        let (kc8, vc8, ksc, vsc, _) = kv8_dec.expect("the nvfp4 arms ride the fp8 sink cache");
        let sg = b.ctx.caps.subgroup && b.ctx.subgroup_width() == Some(32);
        let body =
            wk::flash_decode::fold_stage1_source_nvfp4(hd as u32, sg, fold as u32, k4.is_some());
        let entry =
            wk::flash_decode::fold_stage1_entry_nvfp4(hd as u32, sg, fold as u32, k4.is_some());
        let src = fold_composed(sg, &body);
        let mut binds: Vec<(u32, &Buffer)> = vec![
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
        let grid = ((n_h / fold) as u32, splits, 1);
        if kv_tp.is_none() || dual_stage1 {
            let (pm_body, pm_entry) = if tile >= 2 && sg {
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
            let pm_src = fold_composed(sg, &pm_body);
            b.push(
                "q3d-at-flash1",
                &pm_src,
                &pm_entry,
                &[
                    (0, &q_f32),
                    (4, fd_buf),
                    (5, kc8),
                    (6, vc8),
                    (7, &scratch),
                    (8, ksc),
                    (9, vsc),
                ],
                grid,
            )?;
        }
        if let Some(kc8t) = kv_tp.as_ref() {
            let tpw = hd.is_multiple_of(128);
            let (tp_body, tp_entry) = if tpw {
                (
                    wk::flash_decode::fold_stage1_source_sd_tpw(hd as u32, fold as u32),
                    wk::flash_decode::fold_stage1_entry_sd_tpw(hd as u32, fold as u32),
                )
            } else {
                (
                    wk::flash_decode::fold_stage1_source_sd_tp(hd as u32, fold as u32),
                    wk::flash_decode::fold_stage1_entry_sd_tp(hd as u32, fold as u32),
                )
            };
            let tp_src = fold_composed(true, &tp_body);
            b.push(
                "q3d-at-flash1-tp",
                &tp_src,
                &tp_entry,
                &[
                    (0, &q_f32),
                    (4, fd_buf),
                    (5, kc8t),
                    (6, vc8),
                    (7, &scratch),
                    (8, ksc),
                    (9, vsc),
                ],
                grid,
            )?;
        }
    } else {
        ensure!(
            kv_tp.is_none(),
            "the tp stage1 only replaces the fold stage1; with fold 1 (NV_WGPU_GQA_FOLD=1 or an \
             indivisible group) the transposed cache would be written and never read"
        );
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
                None => vec![(0, &q_f32), (4, fd_buf), (5, &kc), (6, &vc), (7, &scratch)],
            },
            (n_h as u32, splits, 1),
        )?;
    }
    if !shallow_attn {
        b.push(
            "q3d-at-flash2",
            &s.flash,
            wk::flash_decode::ENTRY_STAGE2,
            &[(3, &attn), (4, fd_buf), (7, &scratch)],
            (n_h as u32, 1, 1),
        )?;

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
    }

    let o_q = quant_for(b, s, "q3d-at-oquant", &gated, wo.k(), &[&wo])?;
    let lora_o = lora.and_then(|lc| lc.layer.o.as_ref().map(|site| (lc, site)));
    let o_folded = match (lora_o, &wo) {
        (Some((lc, site)), DenseLinGpu::Fp8(g, _)) if lora_gemv_fold_attn_dn_enabled() => {
            let f = lc.push_site_shrink(b, site, &gated);
            push_gemv_fp8_rowscale_lora(
                b,
                s,
                "q3d-at-oproj",
                g,
                &gated,
                out,
                false,
                Some(LoraFoldSlice {
                    fold: &f,
                    site,
                    slice_id: 0,
                }),
            )?;
            true
        }
        _ => false,
    };
    if !o_folded {
        push_gemv(b, s, "q3d-at-oproj", &wo, &gated, o_q.as_ref(), out)?;
        if let Some((lc, site)) = lora_o {
            lc.push_site(b, site, &gated, &[out]);
        }
    }

    for pf in pfs {
        b.pf_dst_vf = pf.vf;
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
        if let Some(lc) = lora {
            if let Some(site) = &lc.layer.qkv {
                lc.push_site_pf(
                    b,
                    pf.m,
                    site,
                    &pf.normed,
                    &[&pf.a_qraw, &pf.a_kraw, &pf.a_vraw],
                );
            }
        }
        let pf_v_post = if bias_mode {
            b.zeros("q3d-pf-at-vb", pf.a_vraw.size())
        } else {
            pf.a_vraw.clone()
        };
        if bias_mode {
            let bias = qkv_bias.as_ref().expect("bias_mode just uploaded it");
            for (label, src, out, n_rows, y_elems, bias_off, rh) in [
                (
                    "q3d-pf-at-qbias-rope",
                    &pf.a_qraw,
                    &pf.a_q,
                    n_h,
                    n_h * hd,
                    0usize,
                    rot / 2,
                ),
                (
                    "q3d-pf-at-kbias-rope",
                    &pf.a_kraw,
                    &pf.a_k,
                    n_kv,
                    kv_row_elems,
                    n_h * hd,
                    rot / 2,
                ),
                (
                    "q3d-pf-at-vbias",
                    &pf.a_vraw,
                    &pf_v_post,
                    n_kv,
                    kv_row_elems,
                    (n_h + n_kv) * hd,
                    0,
                ),
            ] {
                let p = b.uni(
                    &format!("{label}-p"),
                    BiasRopeMParams {
                        n_rows: n_rows as u32,
                        head_dim: hd as u32,
                        src_stride: hd as u32,
                        rot_half: rh as u32,
                        x_row_elems: (n_rows * hd) as u32,
                        y_row_elems: y_elems as u32,
                        bias_off: bias_off as u32,
                        ..Default::default()
                    },
                );
                b.push_pf(
                    label,
                    &s.prefill,
                    "q3w_attn_bias_rope_m",
                    &[
                        (60, src),
                        (62, &cosb),
                        (63, &sinb),
                        (64, out),
                        (66, &pf.ck),
                        (67, bias),
                        (68, &p),
                    ],
                    (n_rows as u32, m, 1),
                )?;
            }
        } else {
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
                    (61, qn.as_ref().expect("norm arm requires qk-norm weights")),
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
                    (61, kn.as_ref().expect("norm arm requires qk-norm weights")),
                    (62, &cosb),
                    (63, &sinb),
                    (64, &pf.a_k),
                    (65, &kpm),
                    (66, &pf.ck),
                ],
                (n_kv as u32, m, 1),
            )?;
        }
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
                (71, &pf_v_post),
                (72, &kc),
                (73, &vc),
                (74, &kvm),
                (75, &pf.ck),
            ],
            (kv_grid.0, m, 1),
        )?;
        let pf_kv_write_recorded = b.pf_dst_len();
        let chunk_kvq_start = pf
            .attn_tiled
            .as_ref()
            .map(|td| &td.start)
            .or_else(|| pf.attn_kvq_fd_start.as_ref().map(|(_, start)| start));
        ensure!(
            !kv8_decode || chunk_kvq_start.is_some(),
            "NV_Q3D_KV_FP8 decode reads the fp8 cache for every row, so every chunk arm must \
             record the q3w_pf_quantize_kv_fp8_m pair for chunk rows; an arm that skips it \
             starves decode past the chunk boundary (measured before the pair was recorded on \
             the non-tiled arms: bit-green through 30 prefilled, silent 0.005 logit divergence \
             from 31)"
        );
        if let Some(start) = chunk_kvq_start {
            let (kc8, vc8, ksc, vsc, kvq_p) = kv8.as_ref().ok_or_else(|| {
                anyhow!(
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
                    (0, &pf_v_post),
                    (1, vc8),
                    (2, vsc),
                    (3, start),
                    (4, kvq_p),
                    (9, &pkvm),
                ],
                (n_kv as u32, m, 1),
            )?;
            if let Some(kc8t) = kv_tp.as_ref() {
                b.push_pf(
                    "q3d-pf-at-kvq-kt",
                    &pf_kvq_src,
                    "q3w_pf_quantize_kv_fp8_m_kt",
                    &[
                        (0, &pf.a_k),
                        (1, kc8t),
                        (2, ksc),
                        (3, start),
                        (4, kvq_p),
                        (9, &pkvm),
                    ],
                    (n_kv as u32, m, 1),
                )?;
            }
            if let Some((k4, v4w, v4s)) = kv4.as_ref() {
                ensure!(
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
        ensure!(
            kv4.is_none() || chunk_kvq_start.is_some(),
            "{} decode reads the nvfp4 caches for every non-sink row, so every chunk arm must \
             record the nvfp4 chunk quantizers; an arm without a chunk-quantize start buffer \
             leaves chunk rows unquantized and starves decode past the chunk boundary",
            KV_NVFP4_ENV_OFF_BY_DEFAULT_FP8_STAYS_THE_SHIPPED_KV_DECODE_FORMAT
        );
        if let Some(td) = pf.attn_tiled.as_ref() {
            let (kc8, vc8, ksc, vsc, _) = kv8.as_ref().ok_or_else(|| {
                anyhow!(
                    "the tiled pf attention streams the fp8 KV cache, so build_attn must have \
                     allocated it above whenever pf.attn_tiled is set"
                )
            })?;
            ensure!(
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
            ensure!(
                rows_t == 32 && tiled_src.contains("const FDT_ROWS: u32 = 32u"),
                "the tiled flash tile height is baked as 32 on both sides; \
                 PF_TILED_FLASH_ROWS_ONE_KV_STREAM_SERVES_32_ROWS_AT_4_PER_WARP and the kernel's \
                 FDT_ROWS must move together"
            );
            ensure!(
                8 * hd <= PF_ATTN_TILED_STAGE_ARRAYS_HOLD_FDT_POS_8_TIMES_HEAD_DIM_F32,
                "head_dim {hd} overflows the tiled kernel's 2048-f32 workgroup K/V stages \
                 (FDT_POS 8 x head_dim)"
            );
            let qc_words = (pf.m * n_h * hd / 2) as u64;
            let qc_grid = qc_words.div_ceil(256);
            ensure!(
                qc_grid <= 65535,
                "pf tiled qcast grid {qc_grid} exceeds the 1d dispatch limit"
            );
            let splits = wk::flash_decode::splits_for(max_seq) as u32;
            let tiles = pf.m.div_ceil(rows_t) as u32;
            if let Some(q16) = td.q_f16_coop.as_ref() {
                let coop_src = wk::pf_flash_tiled_coop::source();
                b.push_pf(
                    "q3d-pf-at-cq-qcast",
                    &coop_src,
                    wk::pf_flash_tiled_coop::ENTRY_QCAST,
                    &[(0, q16), (1, &pf.a_q), (4, &td.fd)],
                    (qc_grid as u32, 1, 1),
                )?;
                b.push_pf(
                    "q3d-pf-at-cq-flash1",
                    &coop_src,
                    wk::pf_flash_tiled_coop::ENTRY_STAGE1,
                    &[
                        (0, q16),
                        (4, &td.fd),
                        (5, kc8),
                        (6, vc8),
                        (7, &td.scratch),
                        (8, ksc),
                        (9, vsc),
                    ],
                    (n_h as u32, splits, tiles),
                )?;
            } else {
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
            }
            b.push_pf(
                "q3d-pf-at-td-flash2",
                &s.flash,
                wk::flash_decode::ENTRY_STAGE2_MK,
                &[(3, &pf.a_attn), (4, &td.fd), (7, &td.scratch)],
                (n_h as u32, m, 1),
            )?;
        } else if let Some(mk) = pf.attn_mk.as_ref() {
            ensure!(
                pf_kv_write_recorded > 0,
                "the mk flash prefill attention reads chunk rows 0..base+m_live from the bf16 KV \
                 cache, so q3w_kv_write_m must be recorded in the same pf pass list before the mk \
                 dispatches; sequential dispatch order within one compute pass is what makes those \
                 rows visible"
            );
            let rows = PF_ATTN_MK_ROWS_PER_DISPATCH_BOUND_BY_THE_MK_KERNELS_8_WIDE_MREG_ARRAYS;
            let group_bytes = (rows * n_h * hd * 4) as u64;
            ensure!(
                group_bytes.is_multiple_of(256),
                "pf mk row-group bind offset stride {group_bytes} (8 rows x {n_h} heads x hd {hd} \
                 x 4B) must sit on the 256-byte storage-binding alignment"
            );
            ensure!(
                pf.m.div_ceil(rows) == mk.fd.len(),
                "pf mk fd uniforms {} disagree with ceil(m {}/{rows}) row groups",
                mk.fd.len(),
                pf.m
            );
            let qc_words = (pf.m * n_h * hd / 2) as u64;
            let qc_grid = qc_words.div_ceil(256);
            ensure!(
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
        } else if let Some(vfl) = pf.attn_vflash.as_ref() {
            let (kc8, vc8, ksc, vsc, _) = kv8.as_ref().ok_or_else(|| {
                anyhow!(
                    "{} needs the fp8 KV decode cache (NV_Q3D_KV_FP8 not 0): both flash arms \
                     stream fp8 e4m3 KV; unset one",
                    VF_ATTN_FLASH_ENV_DEFAULT_GATE_AT_THE_MEASURED_KNEE_SMV2_LOOP_OR_MIN_TOTAL_PICK_AN_ARM
                )
            })?;
            ensure!(
                pf_kv_write_recorded > 0 && pf.attn_kvq_fd_start.is_some(),
                "the vflash verify attention reads chain rows 0..base+row+1 from the fp8 KV \
                 cache, so q3w_kv_write_m and the q3w_pf_quantize_kv_fp8_m pair must be recorded \
                 earlier in the same pf pass list; sequential dispatch order within one compute \
                 pass is what makes those rows visible"
            );
            let qc_words = (pf.m * n_h * hd / 2) as u64;
            let qc_grid = qc_words.div_ceil(256);
            ensure!(
                qc_grid <= 65535,
                "vflash qcast grid {qc_grid} exceeds the 1d dispatch limit"
            );
            let qcp = b.uni(
                "q3d-pf-at-vfl-qcast-p",
                PfAttnMkQcastParams {
                    n_words: qc_words as u32,
                    scale: 1.0,
                    ..Default::default()
                },
            );
            b.push_pf(
                "q3d-pf-at-vfl-qcast",
                PF_ATTN_MK_QCAST_WGSL,
                "q3w_pf_attn_mk_qcast",
                &[(0, &pf.a_q), (1, &vfl.q_f32), (2, &qcp)],
                (qc_grid as u32, 1, 1),
            )?;
            let splits = wk::flash_decode::splits_for(max_seq) as u32;
            match vfl.arm {
                VfAttnFlashArm::Smv2 => {
                    ensure!(
                        hd <= wk::flash_decode::MAX_HEAD_DIM_MK
                            && pf.m <= wk::flash_decode::MAX_MK_ROWS,
                        "smv2 handles head_dim<={} and m<={} (got hd {hd}, m {}); use =loop",
                        wk::flash_decode::MAX_HEAD_DIM_MK,
                        wk::flash_decode::MAX_MK_ROWS,
                        pf.m
                    );
                    let (fd_whole, _) = pf
                        .attn_kvq_fd_start
                        .as_ref()
                        .expect("ensured beside pf_kv_write_recorded above");
                    let smv2_src = compose(&format!(
                        "{}\n{}",
                        wk::flash_decode::WGSL,
                        wk::flash_decode::smv2_stage1_source_sd()
                    ));
                    b.push_pf(
                        "q3d-pf-at-vfl-flash1",
                        &smv2_src,
                        wk::flash_decode::ENTRY_SMV2_STAGE1_FP8_SD,
                        &[
                            (0, &vfl.q_f32),
                            (4, fd_whole),
                            (5, kc8),
                            (6, vc8),
                            (7, &vfl.scratch),
                            (8, ksc),
                            (9, vsc),
                        ],
                        (n_h as u32, splits, 1),
                    )?;
                    b.push_pf(
                        "q3d-pf-at-vfl-flash2",
                        &s.flash,
                        wk::flash_decode::ENTRY_STAGE2_MK_U,
                        &[(3, &pf.a_attn), (4, fd_whole), (7, &vfl.scratch)],
                        (n_h as u32, m, 1),
                    )?;
                }
                VfAttnFlashArm::RowLoop => {
                    let row_bytes = (n_h * hd * 4) as u64;
                    ensure!(
                        row_bytes.is_multiple_of(256) && vfl.row_fd.len() == pf.m,
                        "vflash row loop needs 256-byte-aligned q/out row offsets ({row_bytes}) \
                         and one fd uniform per row ({} vs m {})",
                        vfl.row_fd.len(),
                        pf.m
                    );
                    let sg = b.ctx.caps.subgroup && b.ctx.subgroup_width() == Some(32);
                    let tile = flash_tile_env();
                    let (row_src, row_entry, row_gx) = if fold > 1 {
                        let (body, entry) = if tile >= 2 && sg {
                            (
                                wk::flash_decode::fold_stage1_source_sd_tiled(
                                    hd as u32,
                                    fold as u32,
                                    tile,
                                ),
                                wk::flash_decode::fold_stage1_entry_sd_tiled(
                                    hd as u32,
                                    fold as u32,
                                    tile,
                                ),
                            )
                        } else {
                            (
                                wk::flash_decode::fold_stage1_source_sd(hd as u32, sg, fold as u32),
                                wk::flash_decode::fold_stage1_entry_sd(hd as u32, sg, fold as u32),
                            )
                        };
                        (fold_composed(sg, &body), entry, (n_h / fold) as u32)
                    } else {
                        (
                            s.flash.clone(),
                            wk::flash_decode::ENTRY_STAGE1_FP8_SD.to_string(),
                            n_h as u32,
                        )
                    };
                    for (r, fd_r) in vfl.row_fd.iter().enumerate() {
                        let off = r as u64 * row_bytes;
                        b.push_pf_off(
                            "q3d-pf-at-vfl-flash1",
                            &row_src,
                            &row_entry,
                            &[
                                (0, &vfl.q_f32, off),
                                (4, fd_r, 0),
                                (5, kc8, 0),
                                (6, vc8, 0),
                                (7, &vfl.scratch, 0),
                                (8, ksc, 0),
                                (9, vsc, 0),
                            ],
                            (row_gx, splits, 1),
                        )?;
                        b.push_pf_off(
                            "q3d-pf-at-vfl-flash2",
                            &s.flash,
                            wk::flash_decode::ENTRY_STAGE2_MK_U,
                            &[(3, &pf.a_attn, off), (4, fd_r, 0), (7, &vfl.scratch, 0)],
                            (n_h as u32, 1, 1),
                        )?;
                    }
                }
            }
        }
        let flash_m_here = bias_mode
            && b.ctx.caps.subgroup
            && hd == 128
            && !cfg.attn_output_gate
            && pf.attn_mk.is_none()
            && pf.attn_tiled.is_none()
            && pf.attn_vflash.is_none();
        if flash_m_here {
            let afm = b.uni(
                "q3d-pf-at-flashm-p",
                AttnFlashMParams {
                    n_kv: n_kv as u32,
                    head_dim: hd as u32,
                    group: (n_h / n_kv) as u32,
                    q_row_elems: (n_h * hd) as u32,
                    y_row_words: (n_h * hd / 2) as u32,
                    scale: 1.0 / (hd as f32).sqrt(),
                    ..Default::default()
                },
            );
            b.push_pf(
                "q3d-pf-at-flashm",
                &s.prefill,
                "q3w_attn_flash_m_pk",
                &[
                    (95, &pf.a_q),
                    (96, &kc),
                    (97, &vc),
                    (98, &pf.a_gated),
                    (99, &afm),
                    (66, &pf.ck),
                ],
                (n_h as u32, m.div_ceil(8), 1),
            )?;
        } else if pf.attn_mk.is_none()
            && pf.attn_tiled.is_none()
            && pf
                .attn_vflash
                .as_ref()
                .is_none_or(|v| v.gate_min_total.is_some())
        {
            let _ = VF_FLASH_GATE_SELECTS_AT_ENCODE_BECAUSE_VERIFY_CBS_ARE_REENCODED_EVERY_ROUND;
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
        if !flash_m_here {
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
        }
        if pf_coop_covers(pf, &wo) {
            pf_coop_cast_x(b, pf, "q3d-pf-at-ocastx", &pf.a_gated, n_h * hd / 2, wo.k())?;
        }
        let o_qm = if matches!(&wo, DenseLinGpu::Nvfp4(_)) && !pf_coop_covers(pf, &wo) {
            quant_for_m_rows(b, s, pf, "q3d-pf-at-oquant", &pf.a_gated, wo.k(), &[&wo])?
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
        if let Some(lc) = lora {
            if let Some(site) = &lc.layer.o {
                lc.push_site_pf(b, pf.m, site, &pf.a_gated, &[&pf.mix]);
            }
        }
    }
    Ok(kv_written_pass_end)
}

fn build_mlp(
    b: &mut Builder,
    s: &Sources,
    cfg: &Qwen3_5DenseConfig,
    m: &HostDenseMlp,
    x: &Buffer,
    out: &Buffer,
    pfs: &[Pf],
    w4a8: Option<&W4a8Quant>,
    lora: Option<&LoraSiteCtx>,
) -> Result<()> {
    let mlp_lora = lora.filter(|lc| lc.layer.gate_up.is_some() || lc.layer.down.is_some());
    let lora_down = mlp_lora.and_then(|lc| lc.layer.down.as_ref());
    let lora_gate_up = mlp_lora.and_then(|lc| lc.layer.gate_up.as_ref());
    let inter = cfg.intermediate_size;
    ensure!(
        m.gate.n() == inter && m.up.n() == inter && m.down.k() == inter,
        "mlp shapes disagree with intermediate_size {inter}"
    );

    let pf_rides_coop = pfs.first().is_none_or(|p| p.coop.is_some());
    let wg = upload_lin(b, "q3d-mlp-gate", &m.gate, W8Scope::Ffn, pf_rides_coop);
    let wu = upload_lin(b, "q3d-mlp-up", &m.up, W8Scope::Ffn, pf_rides_coop);
    let wd = upload_lin(b, "q3d-mlp-down", &m.down, W8Scope::Ffn, pf_rides_coop);

    let y_gate = b.zeros("q3d-mlp-ygate", (inter * 2) as u64);
    let y_up = b.zeros("q3d-mlp-yup", (inter * 2) as u64);
    if let Some(a) = w4a8 {
        let (DenseLinGpu::Nvfp4(g), DenseLinGpu::Nvfp4(u), DenseLinGpu::Nvfp4(d)) = (&wg, &wu, &wd)
        else {
            bail!(
                "q3d-mlp-w4a8: mlp_w4a8_layer_arm admits only all-nvfp4 MLP weights, \
                 upload_lin disagreed"
            );
        };
        let fold = mlp_lora.is_some();
        let gu_fold = match (mlp_lora, lora_gate_up, fold) {
            (Some(lc), Some(site), true) => Some(lc.push_site_shrink(b, site, x)),
            _ => None,
        };
        match (&gu_fold, lora_gate_up) {
            (Some(f), Some(site)) => {
                push_gemv_nvfp4_w4a8(
                    b,
                    s,
                    "q3d-mlp-gproj-w48",
                    g,
                    a,
                    &y_gate,
                    Some(LoraFoldSlice {
                        fold: f,
                        site,
                        slice_id: 0,
                    }),
                )?;
                push_gemv_nvfp4_w4a8(
                    b,
                    s,
                    "q3d-mlp-uproj-w48",
                    u,
                    a,
                    &y_up,
                    Some(LoraFoldSlice {
                        fold: f,
                        site,
                        slice_id: 1,
                    }),
                )?;
            }
            _ => {
                push_gemv_nvfp4_w4a8(b, s, "q3d-mlp-gproj-w48", g, a, &y_gate, None)?;
                push_gemv_nvfp4_w4a8(b, s, "q3d-mlp-uproj-w48", u, a, &y_up, None)?;
                if let (Some(lc), Some(site)) = (mlp_lora, lora_gate_up) {
                    lc.push_site(b, site, x, &[&y_gate, &y_up]);
                }
            }
        }
        let act_w48 = lora_down.map(|_| b.zeros("q3d-mlp-act-w48", (inter * 2) as u64));
        let down_q = push_silu_mul_rowquant_w4a8(
            b,
            s,
            "q3d-mlp-siluq-w48",
            &y_gate,
            &y_up,
            wd.k(),
            act_w48.as_ref(),
        )?;
        match (mlp_lora, lora_down, &act_w48) {
            (Some(lc), Some(site), Some(act)) => {
                if fold {
                    let f = lc.push_site_shrink(b, site, act);
                    push_gemv_nvfp4_w4a8(
                        b,
                        s,
                        "q3d-mlp-dproj-w48",
                        d,
                        &down_q,
                        out,
                        Some(LoraFoldSlice {
                            fold: &f,
                            site,
                            slice_id: 0,
                        }),
                    )?;
                } else {
                    push_gemv_nvfp4_w4a8(b, s, "q3d-mlp-dproj-w48", d, &down_q, out, None)?;
                    lc.push_site(b, site, act, &[out]);
                }
            }
            _ => push_gemv_nvfp4_w4a8(b, s, "q3d-mlp-dproj-w48", d, &down_q, out, None)?,
        }
    } else {
        let in_q = quant_for(b, s, "q3d-mlp-quant", x, wg.k(), &[&wg, &wu])?;
        let merged_gu = match (&wg, &wu, in_q.as_ref()) {
            (DenseLinGpu::Nvfp4(a), DenseLinGpu::Nvfp4(u), Some(q)) if fuse_mlp_gemv_enabled() => {
                push_gemv_nvfp4_2w(b, "q3d-mlp-guproj", a, u, q, &y_gate, &y_up)?
            }
            (DenseLinGpu::AffineW4(a), DenseLinGpu::AffineW4(u), _)
                if env_arm_opt_in_eq_some_1_reread_every_call(
                    AFFINE_GU_FUSED_ENV_GATE_AND_UP_ARE_ONE_MROW4H_DISPATCH_PER_MLP_DEFAULT_OFF_SET_1_TO_ENGAGE,
                ) =>
            {
                push_gemv_affine_w4_gu_pairs(b, "q3d-mlp-guproj", a, u, x, &y_gate, &y_up)?
            }
            _ => false,
        };
        if !merged_gu {
            push_gemv(b, s, "q3d-mlp-gproj", &wg, x, in_q.as_ref(), &y_gate)?;
            push_gemv(b, s, "q3d-mlp-uproj", &wu, x, in_q.as_ref(), &y_up)?;
        }
        if let Some(lc) = mlp_lora {
            if let Some(site) = &lc.layer.gate_up {
                lc.push_site(b, site, x, &[&y_gate, &y_up]);
            }
        }

        let fused_mlp = if let DenseLinGpu::Nvfp4(w) = &wd {
            w.k.is_multiple_of(NVFP4_BLOCK * 4) && lora_down.is_none()
        } else {
            false
        };
        if fused_mlp {
            let DenseLinGpu::Nvfp4(w) = &wd else {
                bail!("q3d-mlp-siluq: fused_mlp checked the nvfp4 arm")
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
            if let (Some(lc), Some(site)) = (mlp_lora, lora_down) {
                lc.push_site(b, site, &act, &[out]);
            }
        }
    }

    for pf in pfs {
        b.pf_dst_vf = pf.vf;
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
        if let Some(lc) = mlp_lora {
            if let Some(site) = &lc.layer.gate_up {
                lc.push_site_pf(b, pf.m, site, &pf.normed_post, &[&pf.m_ygate, &pf.m_yup]);
            }
        }
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
            quant_for_m_rows(b, s, pf, "q3d-pf-mlp-dquant", &pf.m_act, wd.k(), &[&wd])?
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
        if let (Some(lc), Some(site)) = (mlp_lora, lora_down) {
            lc.push_site_pf(b, pf.m, site, &pf.m_act, &[&pf.mlp_out]);
        }
    }
    Ok(())
}

const EMBED_WEIGHT_NAMES: [&str; 3] = [
    "model.language_model.embed_tokens.weight",
    "language_model.model.embed_tokens.weight",
    "model.embed_tokens.weight",
];

enum EmbedHostArm {
    Whole(Vec<u16>),
    #[cfg(target_arch = "wasm32")]
    AffineView {
        module: String,
        gs: usize,
    },
}

#[cfg(target_arch = "wasm32")]
pub const WASM32_EMBED_DEQUANTS_IN_ROW_SEGMENTS_BECAUSE_THE_WHOLE_BF16_TABLE_OVERFLOWS_32BIT_ISIZE:
    usize = 2048;

impl EmbedHostArm {
    fn upload_chunk(
        &self,
        b: &mut Builder,
        src: &WeightSource<'_>,
        off: usize,
        rows: usize,
        hidden: usize,
    ) -> Result<Buffer> {
        match self {
            Self::Whole(e) => {
                let _ = src;
                Ok(b.upload_u32(
                    "q3d-embed",
                    &pack_pairs(&e[off * hidden..(off + rows) * hidden]),
                ))
            }
            #[cfg(target_arch = "wasm32")]
            Self::AffineView { module, gs } => {
                let WeightSource::Loader(w) = src else {
                    bail!("affine embed view arm requires a loader source");
                };
                ensure!(hidden.is_multiple_of(8));
                let row_bytes = (hidden / 2) * 4;
                let buf = b.zeros("q3d-embed", (rows * row_bytes) as u64);
                let words_of = |bytes: &[u8]| -> Vec<u32> {
                    bytes
                        .chunks_exact(4)
                        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                        .collect()
                };
                let widen = |bytes: &[u8]| -> Vec<f32> {
                    bytes
                        .chunks_exact(2)
                        .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
                        .collect()
                };
                let groups = hidden / gs;
                let wpr = hidden / 8;
                let seg_cap =
                    WASM32_EMBED_DEQUANTS_IN_ROW_SEGMENTS_BECAUSE_THE_WHOLE_BF16_TABLE_OVERFLOWS_32BIT_ISIZE;
                let mut r0 = 0usize;
                while r0 < rows {
                    let seg = seg_cap.min(rows - r0);
                    let a = off + r0;
                    let seg_w =
                        w.raw_range_cow(&format!("{module}.weight"), a * wpr * 4, seg * wpr * 4)?;
                    let seg_s = w.raw_range_cow(
                        &format!("{module}.scales"),
                        a * groups * 2,
                        seg * groups * 2,
                    )?;
                    let seg_b = w.raw_range_cow(
                        &format!("{module}.biases"),
                        a * groups * 2,
                        seg * groups * 2,
                    )?;
                    let f32s = nv_quant::mlx_affine::dequantize_mlx_affine_linear(
                        &words_of(&seg_w),
                        &widen(&seg_s),
                        &widen(&seg_b),
                        seg,
                        hidden,
                        *gs,
                    );
                    let bits: Vec<u16> = f32s.iter().map(|x| bf16_bits(*x)).collect();
                    let packed = pack_pairs(&bits);
                    b.ctx
                        .queue
                        .write_buffer(&buf, (r0 * row_bytes) as u64, cast_slice(&packed));
                    r0 += seg;
                }
                Ok(buf)
            }
        }
    }
}

impl WeightSource<'_> {
    fn embed(&self, cfg: &Qwen3_5DenseConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.embed.clone()),
            Self::Loader(w) => {
                load_bf16_or_mlx_affine(w, &EMBED_WEIGHT_NAMES, &[cfg.vocab_size, cfg.hidden_size])
            }
        }
    }

    fn embed_host_arm(&self, cfg: &Qwen3_5DenseConfig) -> Result<EmbedHostArm> {
        #[cfg(target_arch = "wasm32")]
        if let Self::Loader(w) = self {
            for name in EMBED_WEIGHT_NAMES {
                let Some(module) = name.strip_suffix(".weight") else {
                    continue;
                };
                if w.has(&format!("{module}.scales")) {
                    let gs = affine_w4_gs(w, module, cfg.vocab_size, cfg.hidden_size)?;
                    return Ok(EmbedHostArm::AffineView {
                        module: module.to_string(),
                        gs,
                    });
                }
            }
        }
        let e = self.embed(cfg)?;
        ensure!(
            e.len() == cfg.vocab_size * cfg.hidden_size,
            "embed has {} values, want {}",
            e.len(),
            cfg.vocab_size * cfg.hidden_size
        );
        Ok(EmbedHostArm::Whole(e))
    }

    fn lm_head(&self, cfg: &Qwen3_5DenseConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.lm_head.clone()),
            Self::Loader(w) => {
                if cfg.tie_word_embeddings {
                    self.embed(cfg)
                } else {
                    load_bf16_or_mlx_affine(
                        w,
                        &[
                            "lm_head.weight",
                            "language_model.lm_head.weight",
                            "language_model.model.lm_head.weight",
                        ],
                        &[cfg.vocab_size, cfg.hidden_size],
                    )
                }
            }
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn lm_head_affine_view(&self, cfg: &Qwen3_5DenseConfig) -> Result<Option<(String, usize)>> {
        let Self::Loader(w) = self else {
            return Ok(None);
        };
        if cfg.tie_word_embeddings {
            return Ok(None);
        }
        for module in [
            "lm_head",
            "language_model.lm_head",
            "language_model.model.lm_head",
        ] {
            if w.has(&format!("{module}.scales")) {
                let gs = affine_w4_gs(w, module, cfg.vocab_size, cfg.hidden_size)?;
                return Ok(Some((module.to_string(), gs)));
            }
        }
        Ok(None)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn lm_head_affine(&self, cfg: &Qwen3_5DenseConfig) -> Result<Option<HostAffineW4Lin>> {
        let Self::Loader(w) = self else {
            return Ok(None);
        };
        if cfg.tie_word_embeddings {
            return Ok(None);
        }
        for module in [
            "lm_head",
            "language_model.lm_head",
            "language_model.model.lm_head",
        ] {
            if w.has(&format!("{module}.scales")) {
                return load_affine_w4_lin(w, module, cfg.vocab_size, cfg.hidden_size).map(Some);
            }
        }
        Ok(None)
    }

    fn lm_head_fp8_packed(&self, cfg: &Qwen3_5DenseConfig) -> Result<Option<(Vec<u32>, Vec<f32>)>> {
        let Self::Loader(w) = self else {
            return Ok(None);
        };
        if cfg.tie_word_embeddings
            || w.st_dtype_of("lm_head.weight") != Some(nv_weights::StDtype::F8_E4M3)
        {
            return Ok(None);
        }
        let (vocab, hidden) = (cfg.vocab_size, cfg.hidden_size);
        ensure!(
            hidden % 4 == 0,
            "fp8 lm_head packs 4 e4m3 bytes per word; hidden {hidden} % 4 != 0"
        );
        let qw = w.load_quantized_weight("lm_head.weight", nv_weights::QuantScheme::Fp8E4m3)?;
        ensure!(
            qw.shape == [vocab, hidden],
            "lm_head.weight: fp8 shape {:?} != [{vocab}, {hidden}]",
            qw.shape
        );
        let scales = qw
            .fp8_weight_scale_rows()?
            .unwrap_or_else(|| vec![1.0f32; vocab]);
        ensure!(
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
                &[
                    "model.language_model.norm.weight",
                    "language_model.model.norm.weight",
                    "model.norm.weight",
                ],
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
                    &format!("language_model.model.layers.{idx}.input_layernorm.weight"),
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

fn load_bf16(w: &WeightLoader, names: &[&str], shape: &[usize]) -> Result<Vec<u16>> {
    for n in names {
        if w.has(n) {
            if w.st_dtype_of(n) == Some(nv_weights::StDtype::F8_E4M3) {
                return load_fp8_e4m3_as_bf16(w, n, shape);
            }
            let t = w
                .get(n, candle_core::DType::BF16)
                .with_context(|| format!("load {n}"))?;
            ensure!(t.dims() == shape, "{n}: shape {:?} != {shape:?}", t.dims());
            let v: Vec<bf16> = t.flatten_all()?.to_vec1()?;
            return Ok(v.into_iter().map(|x| x.to_bits()).collect());
        }
    }
    bail!("none of {names:?} found")
}

pub const MLX_VLM_NESTS_THE_TEXT_TRUNK_AS_LANGUAGE_MODEL_MODEL_A_PURE_RENAME_OF_THE_HF_NAMES: &str =
    "mlx-community VLM exports name the text trunk language_model.model.* and the head \
     language_model.lm_head.*, where HF exports use model.language_model.* and a bare lm_head; \
     every loader candidate list carries both spellings, the normalize_module_name precedent";

fn layer_base_prefix(w: &WeightLoader, idx: usize) -> String {
    let candidates = [
        format!("model.language_model.layers.{idx}"),
        format!("language_model.model.layers.{idx}"),
        format!("model.layers.{idx}"),
    ];
    for p in &candidates {
        if w.has(&format!("{p}.post_attention_layernorm.weight")) {
            return p.clone();
        }
    }
    candidates[2].clone()
}

pub fn affine_w4_gs(w: &WeightLoader, module: &str, n: usize, k: usize) -> Result<usize> {
    let name = format!("{module}.weight");
    let packed_dtype = w
        .st_dtype_of(&name)
        .ok_or_else(|| anyhow!("missing dtype for {name}"))?;
    ensure!(
        packed_dtype == nv_weights::StDtype::U32,
        "{name}: mlx-affine packed weight must be U32 (8 LSB-first nibbles per word), got \
         {packed_dtype:?}"
    );
    let shape = w
        .shape_of(&name)
        .ok_or_else(|| anyhow!("missing shape for {name}"))?;
    ensure!(
        k.is_multiple_of(8) && shape == [n, k / 8],
        "{name}: packed shape {shape:?} != [{n}, {}]",
        k / 8
    );
    for part in ["scales", "biases"] {
        let pn = format!("{module}.{part}");
        ensure!(
            w.st_dtype_of(&pn) == Some(nv_weights::StDtype::BF16),
            "{pn}: mlx-affine scales/biases must be BF16, got {:?}",
            w.st_dtype_of(&pn)
        );
    }
    let n_scales = w.raw_len(&format!("{module}.scales"))? / 2;
    let n_biases = w.raw_len(&format!("{module}.biases"))? / 2;
    ensure!(
        n_scales > 0 && n_scales == n_biases,
        "{module}: scales ({n_scales}) and biases ({n_biases}) disagree"
    );
    let groups = n_scales / n;
    ensure!(
        groups > 0 && n_scales == n * groups && k.is_multiple_of(groups),
        "{module}: {n_scales} scales do not tile [{n}, {k}] into whole groups"
    );
    let gs = k / groups;
    ensure!(
        gs.is_multiple_of(8),
        "{module}: group size {gs} is not a multiple of the 8-nibble word the gemv decodes"
    );
    Ok(gs)
}

pub fn load_affine_w4_lin(
    w: &WeightLoader,
    module: &str,
    n: usize,
    k: usize,
) -> Result<HostAffineW4Lin> {
    let gs = affine_w4_gs(w, module, n, k)?;
    let words: Vec<u32> = w
        .raw_bytes_cow(&format!("{module}.weight"))?
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let u16s = |bytes: &[u8]| -> Vec<u16> {
        bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect()
    };
    let scales_bf16 = u16s(&w.raw_bytes_cow(&format!("{module}.scales"))?);
    let biases_bf16 = u16s(&w.raw_bytes_cow(&format!("{module}.biases"))?);
    Ok(HostAffineW4Lin {
        words,
        scales_bf16,
        biases_bf16,
        n,
        k,
        gs,
    })
}

pub fn mlx_affine_dequant_bf16_bits(l: &HostAffineW4Lin) -> Vec<u16> {
    let widen = |v: &[u16]| -> Vec<f32> {
        v.iter()
            .map(|h| f32::from_bits((*h as u32) << 16))
            .collect()
    };
    nv_quant::mlx_affine::dequantize_mlx_affine_linear(
        &l.words,
        &widen(&l.scales_bf16),
        &widen(&l.biases_bf16),
        l.n,
        l.k,
        l.gs,
    )
    .iter()
    .map(|x| bf16_bits(*x))
    .collect()
}

fn load_bf16_or_mlx_affine(w: &WeightLoader, names: &[&str], shape: &[usize]) -> Result<Vec<u16>> {
    for name in names {
        let Some(module) = name.strip_suffix(".weight") else {
            continue;
        };
        if w.has(&format!("{module}.scales")) {
            let n = shape[0];
            let k: usize = shape[1..].iter().product();
            let l = load_affine_w4_lin(w, module, n, k)?;
            return Ok(mlx_affine_dequant_bf16_bits(&l));
        }
    }
    load_bf16(w, names, shape)
}

fn load_delta_affine_w4(
    cfg: &Qwen3_5DenseConfig,
    w: &WeightLoader,
    idx: usize,
) -> Result<DeltaAffineW4> {
    let p = format!("{}.linear_attn", layer_base_prefix(w, idx));
    let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
    let conv_dim = 2 * cfg.linear_num_key_heads * cfg.linear_key_head_dim + value_dim;
    let opt = |module: String, n: usize, k: usize| -> Result<Option<HostAffineW4Lin>> {
        if w.has(&format!("{module}.scales")) {
            load_affine_w4_lin(w, &module, n, k).map(Some)
        } else {
            Ok(None)
        }
    };
    Ok(DeltaAffineW4 {
        qkv: opt(format!("{p}.in_proj_qkv"), conv_dim, cfg.hidden_size)?,
        z: opt(format!("{p}.in_proj_z"), value_dim, cfg.hidden_size)?,
        out: opt(format!("{p}.out_proj"), cfg.hidden_size, value_dim)?,
    })
}

fn load_fp8_e4m3_as_bf16(w: &WeightLoader, name: &str, shape: &[usize]) -> Result<Vec<u16>> {
    let rows = *shape.first().unwrap_or(&1);
    let cols: usize = shape.iter().skip(1).product::<usize>().max(1);
    let qw = w.load_quantized_weight(name, nv_weights::QuantScheme::Fp8E4m3)?;
    ensure!(
        qw.shape == shape,
        "{name}: fp8 shape {:?} != {shape:?}",
        qw.shape
    );
    let scales = qw
        .fp8_weight_scale_rows()?
        .unwrap_or_else(|| vec![1.0f32; rows]);
    let dequant = nv_quant::fp8::dequantize_e4m3_per_row(&qw.packed_bytes, rows, cols, &scales)?;
    Ok(dequant.iter().map(|&x| bf16_bits(x)).collect())
}

fn load_norm_plus_one(w: &WeightLoader, names: &[&str], dim: usize) -> Result<Vec<u16>> {
    let raw = load_bf16(w, names, &[dim])?;
    let mlx_export_already_folded_the_plus_one = names
        .iter()
        .find(|n| w.has(n))
        .is_some_and(|n| n.starts_with("language_model."));
    if mlx_export_already_folded_the_plus_one || !crate::qwen3_5_moe_wgpu::norm_plus_one_enabled() {
        return Ok(raw);
    }
    Ok(raw
        .into_iter()
        .map(|b| bf16_bits(bf16_val(b) + 1.0))
        .collect())
}

pub const MLX_EXPORTS_STORE_ZERO_CENTERED_NORMS_WITH_THE_PLUS_ONE_FOLDED_IN: &str =
    "mlx-lm sanitize adds 1 to every zero-centered rmsnorm gamma at convert time (verified \
     tensor-for-tensor against unsloth/Qwen3.8-27B-NVFP4: input/post/final/q/k norms are \
     exactly hf+1.0, the non-centered linear_attn.norm is bit-identical), so a tensor resolved \
     under the language_model.* spelling skips the loader's +1";

fn load_lin(w: &WeightLoader, name: &str, n: usize, k: usize) -> Result<HostBf16Lin> {
    Ok(HostBf16Lin {
        w: load_bf16_or_mlx_affine(w, &[name], &[n, k])?,
        n,
        k,
    })
}

fn load_fp8_lin(w: &WeightLoader, name: &str, n: usize, k: usize) -> Result<Option<HostFp8Lin>> {
    if w.st_dtype_of(name) != Some(nv_weights::StDtype::F8_E4M3) || !k.is_multiple_of(4) {
        return Ok(None);
    }
    let qw = w.load_quantized_weight(name, nv_weights::QuantScheme::Fp8E4m3)?;
    ensure!(
        qw.shape == [n, k],
        "{name}: fp8 shape {:?} != [{n}, {k}]",
        qw.shape
    );
    let scales = qw
        .fp8_weight_scale_rows()?
        .unwrap_or_else(|| vec![1.0f32; n]);
    ensure!(
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
    w: &WeightLoader,
    module: &str,
    n: usize,
    k: usize,
    requant_group: &str,
) -> Result<HostDenseLin> {
    let dense = format!("{module}.weight");
    if w.has(&format!("{module}.scales")) {
        return Ok(HostDenseLin::AffineW4(load_affine_w4_lin(w, module, n, k)?));
    }
    if let Some(fp8) = load_fp8_lin(w, &dense, n, k)? {
        if requant_fp8_group_enabled(requant_group)
            && fp8.n.is_multiple_of(128)
            && fp8.k.is_multiple_of(NVFP4_BLOCK * 4)
        {
            return Ok(HostDenseLin::Nvfp4(requant_fp8_rowscale_to_nvfp4(
                &fp8.packed,
                &fp8.scales,
                fp8.n,
                fp8.k,
            )?));
        }
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
    bail!("{module}: neither .weight nor .weight_packed present")
}

fn load_layer(cfg: &Qwen3_5DenseConfig, w: &WeightLoader, idx: usize) -> Result<HostDenseLayer> {
    let base = layer_base_prefix(w, idx);
    let input_ln = load_norm_plus_one(
        w,
        &[&format!("{base}.input_layernorm.weight")],
        cfg.hidden_size,
    )?;
    let post_attn_ln = load_norm_plus_one(
        w,
        &[&format!("{base}.post_attention_layernorm.weight")],
        cfg.hidden_size,
    )?;
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
            let a_w = load_bf16_or_mlx_affine(
                w,
                &[&format!("{p}.in_proj_a.weight")],
                &[n_v, cfg.hidden_size],
            )?;
            let b_w = load_bf16_or_mlx_affine(
                w,
                &[&format!("{p}.in_proj_b.weight")],
                &[n_v, cfg.hidden_size],
            )?;
            let mut ab = a_w;
            ab.extend_from_slice(&b_w);
            let conv_name = format!("{p}.conv1d.weight");
            let conv_raw = load_bf16(w, &[&conv_name], &[conv_dim, 1, ks]).or_else(|hf_err| {
                load_bf16(w, &[&conv_name], &[conv_dim, ks, 1]).map_err(|mlx_err| {
                    hf_err.context(format!(
                        "conv1d matches neither the HF [C,1,ks] nor the mlx [C,ks,1] layout \
                         (both flatten to the same C*ks order): {mlx_err}"
                    ))
                })
            })?;
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
            let has_qk_norm = w.has(&format!("{p}.q_norm.weight"));
            let has_qkv_bias = w.has(&format!("{p}.q_proj.bias"));
            ensure!(
                has_qk_norm != has_qkv_bias,
                "{p}: expected exactly one of qk-norm weights (qwen3 arch) or q/k/v projection \
                 biases (qwen2 arch); found norm={has_qk_norm} bias={has_qkv_bias}"
            );
            ensure!(
                !has_qkv_bias || !cfg.attn_output_gate,
                "{p}: qkv biases and attn_output_gate cannot combine"
            );
            let qkv_bias_bf16 = if has_qkv_bias {
                let mut bias = load_bf16(w, &[&format!("{p}.q_proj.bias")], &[q_out])?;
                bias.extend(load_bf16(w, &[&format!("{p}.k_proj.bias")], &[kv_out])?);
                bias.extend(load_bf16(w, &[&format!("{p}.v_proj.bias")], &[kv_out])?);
                bias
            } else {
                vec![]
            };
            let (q_norm, k_norm) = if has_qk_norm {
                (
                    load_norm_plus_one(w, &[&format!("{p}.q_norm.weight")], hd)?,
                    load_norm_plus_one(w, &[&format!("{p}.k_norm.weight")], hd)?,
                )
            } else {
                (vec![], vec![])
            };
            HostDenseMixer::Attn(Box::new(HostDenseAttention {
                q: load_dense_lin(w, &format!("{p}.q_proj"), q_out, cfg.hidden_size, "attn")?,
                k: load_dense_lin(w, &format!("{p}.k_proj"), kv_out, cfg.hidden_size, "attn")?,
                v: load_dense_lin(w, &format!("{p}.v_proj"), kv_out, cfg.hidden_size, "attn")?,
                o: load_dense_lin(
                    w,
                    &format!("{p}.o_proj"),
                    cfg.hidden_size,
                    cfg.num_attention_heads * hd,
                    "attn",
                )?,
                q_norm,
                k_norm,
                qkv_bias_bf16,
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
            gate: load_dense_lin(w, &format!("{mp}.gate_proj"), inter, cfg.hidden_size, "mlp")?,
            up: load_dense_lin(w, &format!("{mp}.up_proj"), inter, cfg.hidden_size, "mlp")?,
            down: load_dense_lin(w, &format!("{mp}.down_proj"), cfg.hidden_size, inter, "mlp")?,
        },
        delta_fp8,
    })
}

fn ref_gemv_dense(w: &HostDenseLin, x: &[f32]) -> Vec<f32> {
    match w {
        HostDenseLin::Bf16(l) => ref_gemv_bf16(l, x),
        HostDenseLin::AffineW4(l) => {
            let widen = |v: &[u16]| -> Vec<f32> {
                v.iter()
                    .map(|h| f32::from_bits((*h as u32) << 16))
                    .collect()
            };
            let w = nv_quant::mlx_affine::dequantize_mlx_affine_linear(
                &l.words,
                &widen(&l.scales_bf16),
                &widen(&l.biases_bf16),
                l.n,
                l.k,
                l.gs,
            );
            (0..l.n)
                .map(|row| (0..l.k).map(|i| w[row * l.k + i] * x[i]).sum())
                .collect()
        }
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

pub fn ref_gemv_dense_for_test(w: &HostDenseLin, x: &[f32]) -> Vec<f32> {
    ref_gemv_dense(w, x)
}

pub fn debug_push_gemv_dense_reading_packed_pairs_for_test(
    l: &HostDenseLin,
    x: &[f32],
) -> Result<Vec<f32>> {
    let ctx = WgpuContext::shared().map_err(|e| anyhow!("wgpu context: {e}"))?;
    let s = Sources::new(ctx);
    let mut b = Builder {
        core: crate::wgpu_ledger::VramLedger::new(
            ctx,
            "q3d-dbg-",
            STAGING_FLUSH_BYTES,
        ),
        passes: Vec::new(),
        pf_passes: Vec::new(),
        vf_passes: Vec::new(),
        vf_commit_passes: Vec::new(),
        pf_dst_vf: false,
    };
    ensure!(
        x.len() == l.k(),
        "x has {} elements, weight k is {}",
        x.len(),
        l.k()
    );
    let n = l.n();
    ensure!(
        n.is_multiple_of(2),
        "packed-pairs readback needs even n, got {n}"
    );
    let x_bits: Vec<u16> = x.iter().map(|v| bf16_bits(*v)).collect();
    let xb = b.upload_u32("q3d-dbg-x", &pack_pairs(&x_bits));
    let w = upload_lin(&mut b, "q3d-dbg-w", l, W8Scope::Ffn, false);
    let y = b.zeros("q3d-dbg-y", (n * 2) as u64);
    let q = quant_for(&mut b, &s, "q3d-dbg-quant", &xb, w.k(), &[&w])?;
    push_gemv(&mut b, &s, "q3d-dbg-gemv", &w, &xb, q.as_ref(), &y)?;
    b.flush_staging();
    submit_passes!(ctx, &b.passes);
    let words: Vec<u32> = dispatch::read_back(ctx, &y, n / 2).map_err(|e| anyhow!("{e}"))?;
    let mut out = Vec::with_capacity(n);
    for word in words {
        out.push(f32::from_bits((word & 0xffff) << 16));
        out.push(f32::from_bits((word >> 16) << 16));
    }
    Ok(out)
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

    let trunk = cfg.trunk();
    let mut res: Vec<f32> = (0..hidden)
        .map(|i| bf16_val(hw.embed[token as usize * hidden + i]))
        .collect();

    for li in 0..cfg.num_hidden_layers {
        let layer = &hw.layers[li];
        let normed = ref_rmsnorm(&res, &layer.input_ln, eps);
        let mixed = match &layer.mixer {
            HostDenseMixer::Delta(d) => ref_delta(&trunk, d, &normed, st, li),
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
    let mut v_raw: Vec<f32> = ref_gemv_dense(&a.v, x).into_iter().map(rbf).collect();

    let bias_mode = !a.qkv_bias_bf16.is_empty();
    let bias_at = |i: usize| bf16_val(a.qkv_bias_bf16[i]);
    if bias_mode {
        let v_off = (n_h + n_kv) * hd;
        for (i, v) in v_raw.iter_mut().enumerate() {
            *v = rbf(*v + bias_at(v_off + i));
        }
    }

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
        let mut n = if bias_mode {
            (0..hd).map(|i| rbf(row[i] + bias_at(h * hd + i))).collect()
        } else {
            ref_rmsnorm(&row, &a.q_norm, eps)
        };
        rope(&mut n, pos);
        for i in 0..hd {
            q[h * hd + i] = rbf(n[i]);
        }
    }
    let mut kk = vec![0f32; n_kv * hd];
    for h in 0..n_kv {
        let row: Vec<f32> = (0..hd).map(|i| k_raw[h * hd + i]).collect();
        let mut n = if bias_mode {
            let off = (n_h + h) * hd;
            (0..hd).map(|i| rbf(row[i] + bias_at(off + i))).collect()
        } else {
            ref_rmsnorm(&row, &a.k_norm, eps)
        };
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

pub const Q3D_LORA_TARGETS: [&str; 12] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_a",
    "linear_attn.in_proj_b",
    "linear_attn.out_proj",
];

pub const Q3D_LORA_Q_GATE_ADAPTER_MUST_COVER_VALUE_AND_GATE_LANES: &str =
    "attn_output_gate doubles q_proj's out width into per-head [value|gate] lanes; a q_proj \
     adapter whose lora_B covers only the value half would pass the slot stack's zero-pad and \
     silently leave the gate lanes un-adapted, so a q_proj adapter must cover the full doubled \
     width";

const Q3D_LORA_PREFIX: &str = "model.language_model.layers";

#[derive(Clone, Debug, Default)]
struct Q3dLoraLayer {
    qkv: Option<LoraSite>,
    o: Option<LoraSite>,
    gate_up: Option<LoraSite>,
    down: Option<LoraSite>,
    dn_in: Option<LoraSite>,
    dn_out: Option<LoraSite>,
}

impl Q3dLoraLayer {
    fn sites(&self) -> impl Iterator<Item = &LoraSite> {
        [
            &self.qkv,
            &self.o,
            &self.gate_up,
            &self.down,
            &self.dn_in,
            &self.dn_out,
        ]
        .into_iter()
        .flatten()
    }
}

pub struct Q3dLora {
    rank: usize,
    matched: usize,
    skipped: Vec<String>,
    layers: Vec<Q3dLoraLayer>,
}

impl Q3dLora {
    pub fn from_peft_dir(
        dir: impl AsRef<std::path::Path>,
        config: &Qwen3_5DenseConfig,
    ) -> Result<Self> {
        let adapter =
            nv_weights::lora_adapter::LoraAdapter::load(dir.as_ref(), &candle_core::Device::Cpu)
                .with_context(|| format!("load lora adapter from {}", dir.as_ref().display()))?;
        Self::from_adapter(&adapter, config)
    }

    pub fn from_adapter(
        adapter: &nv_weights::lora_adapter::LoraAdapter,
        config: &Qwen3_5DenseConfig,
    ) -> Result<Self> {
        let rank = adapter.config.r;
        ensure!(
            rank > 0 && rank <= wk::lora::FUSED_MAX_RANK,
            "lora rank {rank} outside 1..={}",
            wk::lora::FUSED_MAX_RANK
        );
        ensure!(
            config.layer_types.len() == config.num_hidden_layers,
            "config carries {} layer_types for {} layers",
            config.layer_types.len(),
            config.num_hidden_layers
        );
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let hd = config.head_dim;
        let n_h = config.num_attention_heads;
        let kv = config.num_key_value_heads * hd;
        let q_out = n_h * hd * if config.attn_output_gate { 2 } else { 1 };
        let n_vh = config.linear_num_value_heads;
        let dn_key_dim = config.linear_num_key_heads * config.linear_key_head_dim;
        let dn_value_dim = n_vh * config.linear_value_head_dim;
        let dn_conv_dim = 2 * dn_key_dim + dn_value_dim;
        let mut matched = 0usize;
        let mut wanted: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for li in 0..config.num_hidden_layers {
            let p = format!("{Q3D_LORA_PREFIX}.{li}");
            let mut layer = Q3dLoraLayer::default();
            if matches!(config.layer_types[li], LayerType::FullAttention) {
                let q = format!("{p}.self_attn.q_proj");
                let kp = format!("{p}.self_attn.k_proj");
                let vp = format!("{p}.self_attn.v_proj");
                if config.attn_output_gate {
                    if let Some(w) = adapter.loras.get(&q) {
                        ensure!(
                            w.lora_b.dim(0)? == q_out,
                            "{q}: lora_B has {} out rows, the gated projection has {q_out}. \
                             {Q3D_LORA_Q_GATE_ADAPTER_MUST_COVER_VALUE_AND_GATE_LANES}",
                            w.lora_b.dim(0)?,
                        );
                    }
                }
                let qkv_slices: Vec<(&str, usize)> =
                    vec![(q.as_str(), q_out), (kp.as_str(), kv), (vp.as_str(), kv)];
                let qkv_segs: Vec<usize> = qkv_slices.iter().map(|(_, w)| *w).collect();
                let (qkv, n) = build_lora_site(adapter, rank, hidden, &qkv_slices, &qkv_segs)?;
                matched += n;
                layer.qkv = qkv;
                let op = format!("{p}.self_attn.o_proj");
                let (o, n) =
                    build_lora_site(adapter, rank, n_h * hd, &[(op.as_str(), hidden)], &[hidden])?;
                matched += n;
                layer.o = o;
                wanted.insert(q);
                wanted.insert(kp);
                wanted.insert(vp);
                wanted.insert(op);
            } else {
                let qp = format!("{p}.linear_attn.in_proj_qkv");
                let zp = format!("{p}.linear_attn.in_proj_z");
                let ap = format!("{p}.linear_attn.in_proj_a");
                let bp = format!("{p}.linear_attn.in_proj_b");
                let op = format!("{p}.linear_attn.out_proj");
                let (dn_in, n) = build_lora_site(
                    adapter,
                    rank,
                    hidden,
                    &[
                        (qp.as_str(), dn_conv_dim),
                        (zp.as_str(), dn_value_dim),
                        (ap.as_str(), n_vh),
                        (bp.as_str(), n_vh),
                    ],
                    &[dn_conv_dim, dn_value_dim, 2 * n_vh],
                )?;
                matched += n;
                layer.dn_in = dn_in;
                let (dn_out, n) = build_lora_site(
                    adapter,
                    rank,
                    dn_value_dim,
                    &[(op.as_str(), hidden)],
                    &[hidden],
                )?;
                matched += n;
                layer.dn_out = dn_out;
                wanted.insert(qp);
                wanted.insert(zp);
                wanted.insert(ap);
                wanted.insert(bp);
                wanted.insert(op);
            }
            let gp = format!("{p}.mlp.gate_proj");
            let up = format!("{p}.mlp.up_proj");
            let (gate_up, n) = build_lora_site(
                adapter,
                rank,
                hidden,
                &[(gp.as_str(), inter), (up.as_str(), inter)],
                &[inter, inter],
            )?;
            matched += n;
            layer.gate_up = gate_up;
            let dp = format!("{p}.mlp.down_proj");
            let (down, n) =
                build_lora_site(adapter, rank, inter, &[(dp.as_str(), hidden)], &[hidden])?;
            matched += n;
            layer.down = down;
            wanted.insert(gp);
            wanted.insert(up);
            wanted.insert(dp);
            layers.push(layer);
        }
        let layer_ns = format!("{Q3D_LORA_PREFIX}.");
        let known_leaf = |n: &str| Q3D_LORA_TARGETS.iter().any(|t| n.ends_with(t));
        let skipped: Vec<String> = adapter
            .loras
            .keys()
            .filter(|n| n.starts_with(&layer_ns) && known_leaf(n) && !wanted.contains(*n))
            .cloned()
            .collect();
        if !skipped.is_empty() {
            eprintln!(
                "[qwen3_5_dense_wgpu] lora: {} modules have no site in this graph (self_attn \
                 names on linear layers and linear_attn names on full-attention layers have no \
                 projection); first is {}",
                skipped.len(),
                skipped[0]
            );
        }
        ensure!(
            matched > 0,
            "lora adapter matched no qwen3.8-dense projection; it carries {} modules, the first \
             of which is {:?}. Expected names like {Q3D_LORA_PREFIX}.0.self_attn.q_proj",
            adapter.loras.len(),
            adapter.loras.keys().next()
        );
        Ok(Self {
            rank,
            matched,
            skipped,
            layers,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn matched_modules(&self) -> usize {
        self.matched
    }

    pub fn skipped_modules(&self) -> &[String] {
        &self.skipped
    }

    pub fn layer_count(&self) -> usize {
        self.layers.len()
    }

    pub fn layer_pass_count(&self, layer: usize) -> usize {
        2 * self.layers[layer].sites().count()
    }

    pub fn total_pass_count(&self) -> usize {
        (0..self.layers.len())
            .map(|i| self.layer_pass_count(i))
            .sum()
    }
}

#[derive(Default)]
struct Q3dLayerLoraGpu {
    qkv: Option<LoraSiteGpu>,
    o: Option<LoraSiteGpu>,
    gate_up: Option<LoraSiteGpu>,
    down: Option<LoraSiteGpu>,
    dn_in: Option<LoraSiteGpu>,
    dn_out: Option<LoraSiteGpu>,
}

fn upload_q3d_lora_layer(ctx: &WgpuContext, l: &Q3dLoraLayer) -> Result<Q3dLayerLoraGpu> {
    let up = |site: &Option<LoraSite>, label: &str| {
        site.as_ref()
            .map(|s| upload_lora_site(ctx, label, s))
            .transpose()
    };
    Ok(Q3dLayerLoraGpu {
        qkv: up(&l.qkv, "q3d-lora-qkv")?,
        o: up(&l.o, "q3d-lora-o")?,
        gate_up: up(&l.gate_up, "q3d-lora-gate-up")?,
        down: up(&l.down, "q3d-lora-down")?,
        dn_in: up(&l.dn_in, "q3d-lora-dn-in")?,
        dn_out: up(&l.dn_out, "q3d-lora-dn-out")?,
    })
}

struct LoraSiteCtx<'a> {
    pl: &'a LoraPipelines,
    layer: Q3dLayerLoraGpu,
    keep: &'a std::cell::RefCell<Vec<Box<dyn std::any::Any + Send>>>,
}

impl LoraSiteCtx<'_> {
    fn push_site(&self, b: &mut Builder, site: &LoraSiteGpu, x: &Buffer, segs: &[&Buffer]) {
        let ctx = b.ctx;
        let mut emitted: Vec<crate::wgpu_lora::Pass> = Vec::new();
        let mut keep = self.keep.borrow_mut();
        emit_lora_site_pk(ctx, &mut emitted, &mut keep, self.pl, 1, site, x, segs);
        for p in emitted {
            b.passes.push(Pass {
                pipeline: p.pipeline,
                bind: p.bind,
                grid: p.grid,
                label: "q3d-lora".into(),
            });
        }
    }

    fn push_site_shrink(&self, b: &mut Builder, site: &LoraSiteGpu, x: &Buffer) -> LoraShrinkFold {
        let ctx = b.ctx;
        let mut emitted: Vec<crate::wgpu_lora::Pass> = Vec::new();
        let mut keep = self.keep.borrow_mut();
        let fold = emit_lora_site_pk_shrink_only(ctx, &mut emitted, &mut keep, self.pl, 1, site, x);
        for p in emitted {
            b.passes.push(Pass {
                pipeline: p.pipeline,
                bind: p.bind,
                grid: p.grid,
                label: "q3d-lora-shrink".into(),
            });
        }
        fold
    }

    fn push_site_pf(
        &self,
        b: &mut Builder,
        m: usize,
        site: &LoraSiteGpu,
        x: &Buffer,
        segs: &[&Buffer],
    ) {
        let ctx = b.ctx;
        let mut emitted: Vec<crate::wgpu_lora::Pass> = Vec::new();
        let mut keep = self.keep.borrow_mut();
        emit_lora_site_pk(ctx, &mut emitted, &mut keep, self.pl, m, site, x, segs);
        for p in emitted {
            b.pf_dst().push(Pass {
                pipeline: p.pipeline,
                bind: p.bind,
                grid: p.grid,
                label: "q3d-pf-lora".into(),
            });
        }
    }
}

#[doc(hidden)]
pub fn w4a8_lora_fold_probe(
    x_bf16: &[u16],
    a_bf16: &[u16],
    b_bf16: &[u16],
    widths: &[usize],
    slice_id: usize,
    w_packed: &[u32],
    w_scales: &[u32],
    alpha: f32,
    n: usize,
    k: usize,
    rank: usize,
) -> Result<(Vec<u16>, Vec<u16>)> {
    let ctx = WgpuContext::shared().map_err(|e| anyhow!("wgpu context: {e}"))?;
    ensure!(
        x_bf16.len() == k,
        "probe x is {} elements, want {k}",
        x_bf16.len()
    );
    ensure!(
        k.is_multiple_of(32) && n.is_multiple_of(2),
        "probe geometry: k {k} must be 32-aligned and n {n} even"
    );
    ensure!(
        widths[slice_id] == n,
        "probe slice {slice_id} is {} wide, gemv writes {n}",
        widths[slice_id]
    );
    ensure!(
        w_packed.len() == n * k / 8,
        "probe packed weight is {} words, want {}",
        w_packed.len(),
        n * k / 8
    );
    let pl = build_lora_pipelines(ctx)?;
    let host = LoraSite {
        a: a_bf16.to_vec(),
        b: b_bf16.to_vec(),
        widths: widths.to_vec(),
        segs: widths.to_vec(),
        k,
        rank,
    };
    let site = upload_lora_site(ctx, "w48-fold-probe-site", &host)?;
    let src_gemv = compose(GEMV_W4A8_WGSL);
    let mk = |label: &str, entry: &str| {
        dispatch::cached_compute_pipeline(ctx, label, &src_gemv, entry)
            .map_err(|e| anyhow!("pipeline {label}::{entry}: {e}"))
    };
    let q_pipe = mk("w48-fold-probe-qx", "w48_quantize_x_evenodd")?;
    let base_pipe = mk("w48-fold-probe-base", W4A8_GEMV_ENTRY)?;
    let fold_pipe = mk("w48-fold-probe-fold", W4A8_GEMV_LORA_FOLD_ENTRY)?;
    let x_words = dispatch::GpuTensor::<u32>::upload(ctx, "w48-fold-probe-x", &pack_pairs(x_bf16));
    let wq = dispatch::GpuTensor::<u32>::upload(ctx, "w48-fold-probe-w", w_packed);
    let ws = dispatch::GpuTensor::<u32>::upload(ctx, "w48-fold-probe-ws", w_scales);
    let xq = dispatch::GpuTensor::<u32>::zeroed(ctx, "w48-fold-probe-xq", k / 4);
    let xs = dispatch::GpuTensor::<f32>::zeroed(ctx, "w48-fold-probe-xs", 1);
    let groups = dispatch::workgroup_count_1d(ctx, n as u64, W4A8_GEMV_ROWS_PER_GROUP);
    let p = dispatch::GpuUniform::new(
        ctx,
        "w48-fold-probe-p",
        &wk::gemv_nvfp4::gemv_params(alpha, n, k, groups.0),
    );
    let y_base = dispatch::GpuTensor::<u32>::zeroed(ctx, "w48-fold-probe-yb", n / 2);
    let y_fused = dispatch::GpuTensor::<u32>::zeroed(ctx, "w48-fold-probe-yf", n / 2);
    let mut lora_passes: Vec<crate::wgpu_lora::Pass> = Vec::new();
    let mut keep: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
    let fold = emit_lora_site_pk_shrink_only(
        ctx,
        &mut lora_passes,
        &mut keep,
        &pl,
        1,
        &site,
        x_words.raw(),
    );
    let lp = dispatch::GpuUniform::new(
        ctx,
        "w48-fold-probe-lp",
        &LoraFoldSlice {
            fold: &fold,
            site: &site,
            slice_id,
        }
        .params("w48-fold-probe", n, k),
    );
    let q_bind = dispatch::bind_group(
        ctx,
        &q_pipe,
        &[
            (4, p.raw()),
            (6, x_words.raw()),
            (7, xq.raw()),
            (8, xs.raw()),
        ],
    );
    let base_bind = dispatch::bind_group(
        ctx,
        &base_pipe,
        &[
            (0, wq.raw()),
            (1, ws.raw()),
            (2, xq.raw()),
            (3, xs.raw()),
            (4, p.raw()),
            (5, y_base.raw()),
        ],
    );
    let fold_bind = dispatch::bind_group(
        ctx,
        &fold_pipe,
        &[
            (0, wq.raw()),
            (1, ws.raw()),
            (2, xq.raw()),
            (3, xs.raw()),
            (4, p.raw()),
            (5, y_fused.raw()),
            (9, site.b.raw()),
            (10, &fold.partials),
            (11, lp.raw()),
        ],
    );
    let mut refs: Vec<dispatch::PassRef<'_>> = vec![(q_pipe.as_ref(), &q_bind, (1u32, 1u32, 1u32))];
    for pss in &lora_passes {
        refs.push((pss.pipeline.as_ref(), &pss.bind, pss.grid));
    }
    refs.push((base_pipe.as_ref(), &base_bind, groups));
    refs.push((fold_pipe.as_ref(), &fold_bind, groups));
    dispatch::submit_pass_list(ctx, refs);
    ctx.poll_blocking()
        .map_err(|e| anyhow!("probe poll: {e}"))?;
    let unpack = |words: Vec<u32>| -> Vec<u16> {
        let mut out = Vec::with_capacity(n);
        for j in 0..n {
            let word = words[j / 2];
            out.push(if j.is_multiple_of(2) {
                (word & 0xffff) as u16
            } else {
                (word >> 16) as u16
            });
        }
        out
    };
    let yb = unpack(
        y_base
            .download(ctx)
            .map_err(|e| anyhow!("probe download base: {e}"))?,
    );
    let yf = unpack(
        y_fused
            .download(ctx)
            .map_err(|e| anyhow!("probe download fused: {e}"))?,
    );
    Ok((yb, yf))
}

fn unpack_bf16_words(words: &[u32], n: usize) -> Vec<u16> {
    let mut out = Vec::with_capacity(n);
    for j in 0..n {
        let word = words[j / 2];
        out.push(if j.is_multiple_of(2) {
            (word & 0xffff) as u16
        } else {
            (word >> 16) as u16
        });
    }
    out
}

fn probe_shrink_passes(
    ctx: &WgpuContext,
    pl: &crate::wgpu_lora::LoraPipelines,
    site: &LoraSiteGpu,
    x_words: &dispatch::GpuTensor<u32>,
) -> (
    Vec<crate::wgpu_lora::Pass>,
    Vec<Box<dyn std::any::Any + Send>>,
    LoraShrinkFold,
) {
    let mut passes: Vec<crate::wgpu_lora::Pass> = Vec::new();
    let mut keep: Vec<Box<dyn std::any::Any + Send>> = Vec::new();
    let fold =
        emit_lora_site_pk_shrink_only(ctx, &mut passes, &mut keep, pl, 1, site, x_words.raw());
    (passes, keep, fold)
}

#[doc(hidden)]
pub fn fp8_rowscale_lora_fold_probe(
    x_bf16: &[u16],
    a_bf16: &[u16],
    b_bf16: &[u16],
    widths: &[usize],
    slice_id: usize,
    w: &HostFp8Lin,
    rank: usize,
) -> Result<(Vec<u16>, Vec<u16>)> {
    let ctx = WgpuContext::shared().map_err(|e| anyhow!("wgpu context: {e}"))?;
    let (n, k) = (w.n, w.k);
    ensure!(
        x_bf16.len() == k,
        "probe x is {} elements, want {k}",
        x_bf16.len()
    );
    ensure!(
        widths[slice_id] == n,
        "probe slice {slice_id} is {} wide, gemv writes {n}",
        widths[slice_id]
    );
    ensure!(w.packed.len() == n * k / 4, "probe fp8 word count");
    let pl = build_lora_pipelines(ctx)?;
    let host = LoraSite {
        a: a_bf16.to_vec(),
        b: b_bf16.to_vec(),
        widths: widths.to_vec(),
        segs: widths.to_vec(),
        k,
        rank,
    };
    let site = upload_lora_site(ctx, "f8-fold-probe-site", &host)?;
    let src = compose(GEMV_BF16_WGSL);
    let mk = |label: &str, entry: &str| {
        dispatch::cached_compute_pipeline(ctx, label, &src, entry)
            .map_err(|e| anyhow!("pipeline {label}::{entry}: {e}"))
    };
    let base_pipe = mk("f8-fold-probe-base", "q3w_gemv_fp8_rowscale")?;
    let fold_pipe = mk("f8-fold-probe-fold", FP8_GEMV_LORA_FOLD_ENTRY)?;
    let x_words = dispatch::GpuTensor::<u32>::upload(ctx, "f8-fold-probe-x", &pack_pairs(x_bf16));
    let wq = dispatch::GpuTensor::<u32>::upload(ctx, "f8-fold-probe-w", &w.packed);
    let ws = dispatch::GpuTensor::<f32>::upload(
        ctx,
        "f8-fold-probe-ws",
        &nv_kernels::shift_decode_fold::fold_scales_for_e4m3_shift_decode(&w.scales),
    );
    let pairs = n.div_ceil(2);
    let grid = dispatch::workgroup_count_1d(ctx, pairs as u64, 1);
    let row_words = k / 4;
    let p = dispatch::GpuUniform::new(
        ctx,
        "f8-fold-probe-p",
        &GemvBf16Params {
            n_rows: n as u32,
            k_words: row_words as u32,
            groups_x: grid.0,
            out_f32: 0,
            w_row_words: row_words as u32,
            x_off_words: 0,
            y_off_words: 0,
            alpha: 1.0,
            ..Default::default()
        },
    );
    let y_base = dispatch::GpuTensor::<u32>::zeroed(ctx, "f8-fold-probe-yb", n.div_ceil(2));
    let y_fused = dispatch::GpuTensor::<u32>::zeroed(ctx, "f8-fold-probe-yf", n.div_ceil(2));
    let (lora_passes, _keep, fold) = probe_shrink_passes(ctx, &pl, &site, &x_words);
    let lp = dispatch::GpuUniform::new(
        ctx,
        "f8-fold-probe-lp",
        &LoraFoldSlice {
            fold: &fold,
            site: &site,
            slice_id,
        }
        .params("f8-fold-probe", n, k),
    );
    let base_bind = dispatch::bind_group(
        ctx,
        &base_pipe,
        &[
            (0, wq.raw()),
            (1, x_words.raw()),
            (2, p.raw()),
            (3, y_base.raw()),
            (4, ws.raw()),
        ],
    );
    let fold_bind = dispatch::bind_group(
        ctx,
        &fold_pipe,
        &[
            (0, wq.raw()),
            (1, x_words.raw()),
            (2, p.raw()),
            (3, y_fused.raw()),
            (4, ws.raw()),
            (11, site.b.raw()),
            (12, &fold.partials),
            (13, lp.raw()),
        ],
    );
    let mut refs: Vec<dispatch::PassRef<'_>> = Vec::new();
    for pss in &lora_passes {
        refs.push((pss.pipeline.as_ref(), &pss.bind, pss.grid));
    }
    refs.push((base_pipe.as_ref(), &base_bind, grid));
    refs.push((fold_pipe.as_ref(), &fold_bind, grid));
    dispatch::submit_pass_list(ctx, refs);
    ctx.poll_blocking().map_err(err)?;
    let yb = unpack_bf16_words(&y_base.download(ctx).map_err(err)?, n);
    let yf = unpack_bf16_words(&y_fused.download(ctx).map_err(err)?, n);
    Ok((yb, yf))
}

#[doc(hidden)]
pub fn dn_merged_lora_fold_probe(
    x_bf16: &[u16],
    a_bf16: &[u16],
    b_bf16: &[u16],
    widths: &[usize],
    qkv: &HostFp8Lin,
    z: &HostFp8Lin,
    ab: &HostBf16Lin,
    rank: usize,
) -> Result<(Vec<Vec<u16>>, Vec<Vec<u16>>)> {
    let ctx = WgpuContext::shared().map_err(|e| anyhow!("wgpu context: {e}"))?;
    let k = qkv.k;
    ensure!(z.k == k && ab.k == k, "probe k mismatch");
    ensure!(
        x_bf16.len() == k,
        "probe x is {} elements, want {k}",
        x_bf16.len()
    );
    ensure!(
        widths.len() == 4
            && widths[0] == qkv.n
            && widths[1] == z.n
            && widths[2] + widths[3] == ab.n
            && widths[2] == widths[3],
        "probe widths {widths:?} disagree with merged rows [{}, {}, {}]",
        qkv.n,
        z.n,
        ab.n
    );
    let pl = build_lora_pipelines(ctx)?;
    let host = LoraSite {
        a: a_bf16.to_vec(),
        b: b_bf16.to_vec(),
        widths: widths.to_vec(),
        segs: vec![widths[0], widths[1], widths[2] + widths[3]],
        k,
        rank,
    };
    let site = upload_lora_site(ctx, "mg-fold-probe-site", &host)?;
    let src = compose(GEMV_BF16_WGSL);
    let mk = |label: &str, entry: &str| {
        dispatch::cached_compute_pipeline(ctx, label, &src, entry)
            .map_err(|e| anyhow!("pipeline {label}::{entry}: {e}"))
    };
    let base_pipe = mk(
        "mg-fold-probe-base",
        "q3w_gemv_dn_merged_fp8_qkv_fp8_z_bf16_ab",
    )?;
    let fold_pipe = mk("mg-fold-probe-fold", DN_MERGED_GEMV_LORA_FOLD_ENTRY)?;
    let x_words = dispatch::GpuTensor::<u32>::upload(ctx, "mg-fold-probe-x", &pack_pairs(x_bf16));
    let w_qkv = dispatch::GpuTensor::<u32>::upload(ctx, "mg-fold-probe-wq", &qkv.packed);
    let s_qkv = dispatch::GpuTensor::<f32>::upload(
        ctx,
        "mg-fold-probe-sq",
        &nv_kernels::shift_decode_fold::fold_scales_for_e4m3_shift_decode(&qkv.scales),
    );
    let w_z = dispatch::GpuTensor::<u32>::upload(ctx, "mg-fold-probe-wz", &z.packed);
    let s_z = dispatch::GpuTensor::<f32>::upload(
        ctx,
        "mg-fold-probe-sz",
        &nv_kernels::shift_decode_fold::fold_scales_for_e4m3_shift_decode(&z.scales),
    );
    let w_ab = dispatch::GpuTensor::<u32>::upload(ctx, "mg-fold-probe-wab", &pack_pairs(&ab.w));
    let qkv_pairs = qkv.n.div_ceil(2);
    let z_pairs = z.n.div_ceil(2);
    let ab_pairs = ab.n.div_ceil(2);
    let grid = dispatch::workgroup_count_1d(ctx, (qkv_pairs + z_pairs + ab_pairs) as u64, 1);
    let p = dispatch::GpuUniform::new(
        ctx,
        "mg-fold-probe-p",
        &GemvDnMergedParams {
            qkv_pairs: qkv_pairs as u32,
            z_pairs: z_pairs as u32,
            ab_pairs: ab_pairs as u32,
            qkv_rows: qkv.n as u32,
            z_rows: z.n as u32,
            ab_rows: ab.n as u32,
            fp8_row_words: (k / 4) as u32,
            bf16_row_words: (k / 2) as u32,
            groups_x: grid.0,
            ..Default::default()
        },
    );
    let mk_ys = |tag: &str| {
        [
            dispatch::GpuTensor::<u32>::zeroed(ctx, tag, qkv.n.div_ceil(2)),
            dispatch::GpuTensor::<u32>::zeroed(ctx, tag, z.n.div_ceil(2)),
            dispatch::GpuTensor::<u32>::zeroed(ctx, tag, ab.n.div_ceil(2)),
        ]
    };
    let ys_base = mk_ys("mg-fold-probe-yb");
    let ys_fused = mk_ys("mg-fold-probe-yf");
    let (lora_passes, _keep, fold) = probe_shrink_passes(ctx, &pl, &site, &x_words);
    let lp = dispatch::GpuUniform::new(
        ctx,
        "mg-fold-probe-lp",
        &DnMergedLoraFold {
            fold: &fold,
            site: &site,
        }
        .params("mg-fold-probe", qkv.n, z.n, ab.n, k),
    );
    let base_bind = dispatch::bind_group(
        ctx,
        &base_pipe,
        &[
            (0, w_qkv.raw()),
            (1, x_words.raw()),
            (3, ys_base[0].raw()),
            (4, s_qkv.raw()),
            (5, w_z.raw()),
            (6, s_z.raw()),
            (7, ys_base[1].raw()),
            (8, w_ab.raw()),
            (9, ys_base[2].raw()),
            (10, p.raw()),
        ],
    );
    let fold_bind = dispatch::bind_group(
        ctx,
        &fold_pipe,
        &[
            (0, w_qkv.raw()),
            (1, x_words.raw()),
            (3, ys_fused[0].raw()),
            (4, s_qkv.raw()),
            (5, w_z.raw()),
            (6, s_z.raw()),
            (7, ys_fused[1].raw()),
            (8, w_ab.raw()),
            (9, ys_fused[2].raw()),
            (10, p.raw()),
            (11, site.b.raw()),
            (12, &fold.partials),
            (14, lp.raw()),
        ],
    );
    let mut refs: Vec<dispatch::PassRef<'_>> = Vec::new();
    for pss in &lora_passes {
        refs.push((pss.pipeline.as_ref(), &pss.bind, pss.grid));
    }
    refs.push((base_pipe.as_ref(), &base_bind, grid));
    refs.push((fold_pipe.as_ref(), &fold_bind, grid));
    dispatch::submit_pass_list(ctx, refs);
    ctx.poll_blocking().map_err(err)?;
    let dl = |ys: &[dispatch::GpuTensor<u32>; 3]| -> Result<Vec<Vec<u16>>> {
        Ok(vec![
            unpack_bf16_words(&ys[0].download(ctx).map_err(err)?, qkv.n),
            unpack_bf16_words(&ys[1].download(ctx).map_err(err)?, z.n),
            unpack_bf16_words(&ys[2].download(ctx).map_err(err)?, ab.n),
        ])
    };
    Ok((dl(&ys_base)?, dl(&ys_fused)?))
}
