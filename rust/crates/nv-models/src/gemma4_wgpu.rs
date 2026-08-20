use anyhow::{Context, Result};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use nv_kernels::wgpu_backend::dispatch::{self, GpuTensor, GpuUniform};
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};

use crate::gemma4::{Gemma4Config, LayerType};
use crate::gemma4_e4b_wgpu::KvCacheSnapshot;
use crate::qwen3_5_moe_wgpu as q3m;

use crate::gemma4_wgpu_shared::{
    bf16_bits, bytes_to_words, err, pack_pairs, rope_tables, FLASH2_PK_ENTRY, FLASH2_PK_WGSL,
    GEMV_PK3_ENTRY, GEMV_PK_ENTRY, GEMV_PK_WGSL, ROPE_F32_ENTRY, ROPE_F32_WGSL,
};

pub fn flash_splits() -> u32 {
    wk::flash_decode::splits_env() as u32
}

pub const FLASH_SD_DEFAULT_ON_SHIFT_TWIN_WINS_AT_DEPTH_AND_HOLDS_PPL: &str =
    "the exact e4m3_decode is ~12 ops/element in a decode-ALU bound regime; the shift twin \
     flash_splitk_stage1_fp8kv_sd decodes in 2 ops with the 2^120 carry folded into the \
     per-position k/v scales the kernel already multiplies. At deep KV the shift twin wins \
     decode ms/tok and teacher-forced ppl is unchanged (the e4m3-subnormal flush is below \
     noise); current numbers: perf/runs.jsonl. NV_G4_FLASH_SD=0 restores the exact-decode \
     stage1";

fn flash_sd_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NV_G4_FLASH_SD").ok().as_deref() != Some("0"))
}

fn flash1_stage1_entry() -> &'static str {
    if flash_sd_enabled() {
        wk::flash_decode::ENTRY_STAGE1_FP8_SD
    } else {
        wk::flash_decode::ENTRY_STAGE1_FP8
    }
}

pub const MAX_CHAIN: usize = 8;

pub const VERIFY_ROWS_MAX_IS_THE_LONGEST_CHAIN_THE_SPEC_LOOP_SUBMITS: usize = MAX_CHAIN + 1;

pub use crate::embed_row_splice::EmbedRowSplice;

struct ChunkRowSplice<'a> {
    rel_pos: usize,
    row_words: &'a [u32],
}

pub fn preenc_enabled_default() -> bool {
    std::env::var("NV_G4_WGPU_PREENC").ok().as_deref() != Some("0")
}

fn uniform_probe_reps() -> usize {
    std::env::var("NV_G4_WGPU_UNIFORM_PROBE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

pub fn chain_k_from_env() -> usize {
    std::env::var("NV_G4_WGPU_CHAIN")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|k| k.clamp(1, MAX_CHAIN))
        .unwrap_or(2)
}

const GLUE_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4w_glue.wgsl");

pub const MK_MAX: usize = 16;

pub const EMBED_ROW_SPLICE_ENTRY: &str = "g4w_splice_embed_rows";

const EMBED_ROW_SPLICE_WGSL: &str = r#"
struct G4wSpParams { hidden_words: u32, m: u32, pad0: u32, pad1: u32 };
@group(0) @binding(120) var<storage, read> gsp_rows: array<u32>;
@group(0) @binding(121) var<storage, read> gsp_mask: array<u32>;
@group(0) @binding(122) var<storage, read_write> gsp_out: array<u32>;
@group(0) @binding(123) var<uniform> gsp_p: G4wSpParams;

@compute @workgroup_size(256)
fn g4w_splice_embed_rows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let t = wid.y;
    if (t >= gsp_p.m || gsp_mask[t] == 0u) {
        return;
    }
    let w = wid.x * 256u + lid.x;
    if (w >= gsp_p.hidden_words) {
        return;
    }
    gsp_out[t * gsp_p.hidden_words + w] = gsp_rows[t * gsp_p.hidden_words + w];
}
"#;

const MK_PARAMS_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4w_mk_params.wgsl");

fn mk_pk3_store(t: usize, q: &str, k: &str, v: &str, sp: &str) -> String {
    format!(
        concat!(
            "            if (row < {sp}.q_rows) {{\n",
            "                {q}[{t}u * ({sp}.q_rows >> 1u) + (row >> 1u)] = word;\n",
            "            }} else {{\n",
            "                let kr = row - {sp}.q_rows;\n",
            "                if (kr < {sp}.kv_rows) {{\n",
            "                    {k}[{t}u * ({sp}.kv_rows >> 1u) + (kr >> 1u)] = word;\n",
            "                }}\n",
            "                if (row >= {sp}.v_off) {{\n",
            "                    let vr = row - {sp}.v_off;\n",
            "                    if (vr < {sp}.kv_rows) {{\n",
            "                        {v}[{t}u * ({sp}.kv_rows >> 1u) + (vr >> 1u)] = word;\n",
            "                    }}\n",
            "                }}\n",
            "            }}\n",
        ),
        t = t,
        q = q,
        k = k,
        v = v,
        sp = sp
    )
}

fn mk_bf16_source(mk_max: usize) -> String {
    use std::fmt::Write as _;
    assert!((1..=MK_MAX).contains(&mk_max));
    let mut b = String::from(MK_PARAMS_WGSL);
    b.push('\n');
    let pk_store = |t: usize| {
        format!(
            "            gemv_bf16_y[g4w_mk_params.dst_word_off + {t}u * g4w_mk_params.y_stride_words + (row >> 1u)] = word;\n"
        )
    };
    let pk3_store = |t: usize| mk_pk3_store(t, "g4w_y_q", "g4w_y_k", "g4w_y_v", "g4w_split_params");
    for (entry, store) in [
        ("g4w_gemm_bf16_mk_pk", &pk_store as &dyn Fn(usize) -> String),
        ("g4w_gemm_bf16_mk_pk3", &pk3_store),
    ] {
        b.push_str("@compute @workgroup_size(256)\n");
        writeln!(b, "fn {entry}(").unwrap();
        b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
        b.push_str("    let tid = lid.x;\n");
        b.push_str("    let lane = tid & (GEMV_BF16_LANES - 1u);\n");
        b.push_str("    let warp = tid / GEMV_BF16_LANES;\n");
        b.push_str("    let row = gemv_bf16_row(wid, warp);\n");
        b.push_str("    let live = row < gemv_bf16_params.n_rows;\n");
        b.push_str("    let kv = select(0u, gemv_bf16_params.k_elems >> 3u, live);\n");
        b.push_str("    let w_base = select(0u, row * gemv_bf16_params.w_row_words, live);\n");
        b.push_str("    let mm = g4w_mk_params.m;\n");
        b.push_str("    let xs = g4w_mk_params.x_stride_words;\n");
        for t in 0..mk_max {
            writeln!(b, "    var acc{t} = 0.0;").unwrap();
        }
        b.push_str("    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {\n");
        b.push_str("        let wo = w_base + (v << 2u);\n");
        b.push_str("        let xo = v << 2u;\n");
        b.push_str("        for (var j = 0u; j < 4u; j = j + 1u) {\n");
        b.push_str("            let ww = gemv_bf16_w[wo + j];\n");
        b.push_str("            let wl = bf16_lo(ww);\n");
        b.push_str("            let wh = bf16_hi(ww);\n");
        for t in 0..mk_max {
            let xi = if t == 0 {
                "xo + j".to_string()
            } else {
                format!("{t}u * xs + xo + j")
            };
            writeln!(
                b,
                "            if ({t}u < mm) {{ let xw = gemv_bf16_x[{xi}]; acc{t} = acc{t} + (wl * bf16_lo(xw) + wh * bf16_hi(xw)); }}"
            )
            .unwrap();
        }
        b.push_str("        }\n    }\n");
        for t in 0..mk_max {
            writeln!(b, "    if ({t}u < mm) {{").unwrap();
            writeln!(
                b,
                "        let total{t} = gemv_bf16_reduce(tid, lane, acc{t});"
            )
            .unwrap();
            b.push_str("        if (lane == 0u && live && (warp & 1u) == 0u) {\n");
            writeln!(
                b,
                "            let word = g4w_pair_word(tid, total{t}, row + 1u < gemv_bf16_params.n_rows);"
            )
            .unwrap();
            b.push_str(&store(t));
            b.push_str("        }\n");
            b.push_str("        workgroupBarrier();\n");
            b.push_str("    }\n");
        }
        b.push_str("}\n\n");
    }
    b
}

fn mk_q8_source(mk_max: usize) -> String {
    use std::fmt::Write as _;
    assert!((1..=MK_MAX).contains(&mk_max));
    let mut b = String::from(MK_PARAMS_WGSL);
    b.push('\n');
    let pk_store = |t: usize| {
        format!(
            "            qg_y[g4w_mk_params.dst_word_off + {t}u * g4w_mk_params.y_stride_words + (row >> 1u)] = word;\n"
        )
    };
    let pk3_store = |t: usize| mk_pk3_store(t, "qg_y_q", "qg_y_k", "qg_y_v", "qg_split_params");
    for (tag, dot) in [("int8", "qg_dot16_i8"), ("fp8", "qg_dot16_e4m3")] {
        for (suffix, store) in [
            ("pk", &pk_store as &dyn Fn(usize) -> String),
            ("pk3", &pk3_store),
        ] {
            b.push_str("@compute @workgroup_size(256)\n");
            writeln!(b, "fn g4w_gemm_{tag}_mk_{suffix}(").unwrap();
            b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
            b.push_str("    let tid = lid.x;\n");
            b.push_str("    let lane = tid & (QG_LANES - 1u);\n");
            b.push_str("    let warp = tid / QG_LANES;\n");
            b.push_str(
                "    let row = (wid.x + wid.y * qg_params.groups_x) * QG_TREE_ROWS + warp;\n",
            );
            b.push_str("    let live = row < qg_params.n_rows;\n");
            b.push_str("    let kv = select(0u, qg_params.k_elems >> 4u, live);\n");
            b.push_str("    let wbase = select(0u, row * (qg_params.k_elems >> 4u), live);\n");
            b.push_str("    let sbase = select(0u, row * qg_params.scales_per_row, live);\n");
            b.push_str("    let sh = qg_params.group_shift;\n");
            b.push_str("    let mm = g4w_mk_params.m;\n");
            b.push_str("    let xs4 = g4w_mk_params.x_stride_words >> 2u;\n");
            for t in 0..mk_max {
                writeln!(b, "    var acc{t} = 0.0;").unwrap();
            }
            b.push_str("    for (var v = lane; v < kv; v = v + QG_LANES) {\n");
            b.push_str("        let wv = qg_w4[wbase + v];\n");
            b.push_str("        let sc = qg_row_scale[sbase + (v >> sh)];\n");
            for t in 0..mk_max {
                let base = if t == 0 {
                    "".to_string()
                } else {
                    format!("{t}u * xs4 + ")
                };
                writeln!(
                    b,
                    "        if ({t}u < mm) {{ let d = {dot}(wv, qg_x4[{base}2u * v], qg_x4[{base}2u * v + 1u]); acc{t} = fma(sc, d, acc{t}); }}"
                )
                .unwrap();
            }
            b.push_str("    }\n");
            for t in 0..mk_max {
                writeln!(b, "    if ({t}u < mm) {{").unwrap();
                writeln!(b, "        let total{t} = qg_reduce(tid, lane, acc{t});").unwrap();
                b.push_str("        if (lane == 0u && live && (warp & 1u) == 0u) {\n");
                writeln!(b, "            let word = qg_pk_word(tid, row, total{t});").unwrap();
                b.push_str(&store(t));
                b.push_str("        }\n");
                b.push_str("        workgroupBarrier();\n");
                b.push_str("    }\n");
            }
            b.push_str("}\n\n");
        }
    }
    b
}

pub const LMHEAD_I8_MK_PK_ENTRY: &str = "g4w_lmhead_i8_normed_mk_pk";

fn mk_i8_lmhead_source(mk_max: usize) -> String {
    use std::fmt::Write as _;
    assert!(
        (1..=MK_MAX).contains(&mk_max),
        "the int8 lm_head M-row twin unrolls one accumulator per slot; {mk_max} is outside 1..={MK_MAX}"
    );
    let mut b = String::from(MK_PARAMS_WGSL);
    b.push('\n');
    b.push_str("@compute @workgroup_size(256)\n");
    writeln!(b, "fn {LMHEAD_I8_MK_PK_ENTRY}(").unwrap();
    b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
    b.push_str("    let tid = lid.x;\n");
    b.push_str("    let lane = tid & (GEMV_BF16_LANES - 1u);\n");
    b.push_str("    let warp = tid / GEMV_BF16_LANES;\n");
    b.push_str("    let row = (wid.x + wid.y * gi8_params.groups_x) * GEMV_BF16_ROWS + warp;\n");
    b.push_str("    let live = row < gi8_params.n_rows;\n");
    b.push_str("    let kv = select(0u, gi8_params.k_elems >> 4u, live);\n");
    b.push_str("    let w_base = select(0u, row * gi8_params.wq_row_words, live);\n");
    b.push_str("    let mm = g4w_mk_params.m;\n");
    for t in 0..mk_max {
        writeln!(b, "    var acc{t} = 0.0;").unwrap();
    }
    b.push_str("    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {\n");
    b.push_str("        let wo = w_base + (v << 2u);\n");
    b.push_str("        let kb = v << 4u;\n");
    b.push_str("        for (var t = 0u; t < 4u; t = t + 1u) {\n");
    b.push_str("            let word = gi8_wq[wo + t];\n");
    b.push_str("            for (var i = 0u; i < 4u; i = i + 1u) {\n");
    b.push_str("                let f = int8_decode(word, i);\n");
    b.push_str("                let kk = kb + (t << 2u) + i;\n");
    for t in 0..mk_max {
        writeln!(
            b,
            "                if ({t}u < mm) {{ acc{t} = fma(f, gi8_xval({t}u, kk), acc{t}); }}"
        )
        .unwrap();
    }
    b.push_str("            }\n        }\n    }\n");
    b.push_str("    let rs = gi8_row_scale[select(0u, row, live)];\n");
    for t in 0..mk_max {
        writeln!(b, "    if ({t}u < mm) {{").unwrap();
        writeln!(
            b,
            "        let scaled{t} = gi8_scale_mul(gemv_bf16_reduce_seq(tid, lane, acc{t}), rs);"
        )
        .unwrap();
        b.push_str("        workgroupBarrier();\n");
        writeln!(
            b,
            "        if (lane == 0u) {{\n            gemv_bf16_partial[tid] = scaled{t};\n        }}"
        )
        .unwrap();
        b.push_str("        workgroupBarrier();\n");
        b.push_str("        if (lane == 0u && live && (warp & 1u) == 0u) {\n");
        writeln!(b, "            let lo = bf16_encode(scaled{t}) & 0xffffu;").unwrap();
        b.push_str("            let hi_live = row + 1u < gi8_params.n_rows;\n");
        b.push_str(
            "            let hi = bf16_encode(gemv_bf16_partial[tid + GEMV_BF16_LANES]) & 0xffffu;\n",
        );
        b.push_str("            let packed = lo | (select(0u, hi, hi_live) << 16u);\n");
        writeln!(
            b,
            "            gi8_y[g4w_mk_params.dst_word_off + {t}u * g4w_mk_params.y_stride_words + (row >> 1u)] = packed;"
        )
        .unwrap();
        b.push_str("        }\n");
        b.push_str("        workgroupBarrier();\n");
        b.push_str("    }\n");
    }
    b.push_str("}\n");
    b
}

#[doc(hidden)]
pub fn mk_i8_lmhead_shader_source(mk_max: usize) -> String {
    format!(
        "{}\n{}",
        compose(wk::gemv_bf16::WGSL),
        mk_i8_lmhead_source(mk_max)
    )
}

const GEMV4_PK_TREE_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4w_gemv4_pk_tree.wgsl");

const GEMV4_PK_SG_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4w_gemv4_pk_sg.wgsl");

const FP8_PK_BINDINGS_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4w_fp8_pk_bindings.wgsl");

const GEMV_Q8_PK_TREE_PROLOGUE: &str = include_str!("../../nv-kernels/wgsl/g4w_gemv_q8_pk_tree_prologue.wgsl");

const GEMV_Q8_PK_TREE_TEMPLATE: &str = include_str!("../../nv-kernels/wgsl/g4w_gemv_q8_pk_tree_template.wgsl");

const GEMV_Q8_PK_SG_PROLOGUE: &str = include_str!("../../nv-kernels/wgsl/g4w_gemv_q8_pk_sg_prologue.wgsl");

const GEMV_Q8_PK_SG_TEMPLATE: &str = include_str!("../../nv-kernels/wgsl/g4w_gemv_q8_pk_sg_template.wgsl");

const GEMV_Q8_PK_TREE_LEGACY: &str = include_str!("../../nv-kernels/wgsl/g4w_gemv_q8_pk_tree_legacy.wgsl");

const GEMV_Q8_PK_SG_LEGACY: &str = include_str!("../../nv-kernels/wgsl/g4w_gemv_q8_pk_sg_legacy.wgsl");

pub fn attn_fp8_legacy_epilogue() -> u32 {
    attn_variant().legacy_epilogue
}

pub const FP8_PK_ENTRY: &str = "g4w_gemv_fp8_pk";
pub const FP8_PK3_ENTRY: &str = "g4w_gemv_fp8_pk3";
#[doc(hidden)]
pub const INT8_PK_ENTRY: &str = "g4w_gemv_int8_pk";
pub const FP8_LEGACY_PK_ENTRY: &str = "g4w_gemv_legacy_pk";
pub const FP8_LEGACY_PK3_ENTRY: &str = "g4w_gemv_legacy_pk3";
#[doc(hidden)]
pub const FP8_MK_PK_ENTRY: &str = "g4w_gemm_fp8_mk_pk";
#[doc(hidden)]
pub const INT8_MK_PK_ENTRY: &str = "g4w_gemm_int8_mk_pk";

pub fn fp8_pk_shader_source(sg: bool) -> String {
    format!(
        "{}\n{}\n{}",
        wk::quant_gemv::source(),
        FP8_PK_BINDINGS_WGSL,
        q8_pk_wgsl(sg)
    )
}

pub fn fp8_pk_subgroup_path(ctx: &WgpuContext) -> bool {
    fp8_sg(ctx)
}

#[doc(hidden)]
pub fn mk_q8_shader_source(mk_max: usize) -> String {
    format!("{}\n{}", fp8_pk_shader_source(false), mk_q8_source(mk_max))
}

#[doc(hidden)]
pub fn mk_bf16_shader_source(mk_max: usize) -> String {
    format!(
        "{}\n{}\n{}",
        compose(wk::gemv_bf16::WGSL),
        GEMV_PK_WGSL,
        mk_bf16_source(mk_max)
    )
}

#[doc(hidden)]
pub fn glue_shader_source() -> &'static str {
    GLUE_WGSL
}

pub fn fp8_pk_rows_per_group(sg: bool) -> u32 {
    if sg {
        wk::quant_gemv::SG_ROWS_PER_GROUP
    } else {
        wk::quant_gemv::TREE_ROWS_PER_GROUP
    }
}

#[doc(hidden)]
pub fn nozi_audit_sources() -> Vec<(&'static str, String)> {
    let maxw = (MAX_HEAD_DIM_FOR_AUDIT / 2).max(1);
    vec![
        (
            "g4w:head_prep",
            compose(&HEAD_PREP_WGSL.replace("HP_MAXW", &format!("{maxw}u"))),
        ),
        ("g4w:norm_chain", compose(NORM_CHAIN_WGSL)),
        ("g4w:q8_pk_tree", fp8_pk_shader_source(false)),
        ("g4w:q8_pk_sg", fp8_pk_shader_source(true)),
        (
            "g4w:gemv_nvfp4_pk",
            format!("{}\n{}", wk::gemv_nvfp4::gemv_source(), GEMV4_PK_TREE_WGSL),
        ),
        (
            "g4w:quant_row_pk",
            format!("{}\n{}", wk::gemv_nvfp4::quantize_source(), QUANT_PK_WGSL),
        ),
        (
            "g4w:gemv_bf16_pk",
            format!("{}\n{}", compose(wk::gemv_bf16::WGSL), GEMV_PK_WGSL),
        ),
    ]
}

const MAX_HEAD_DIM_FOR_AUDIT: usize = 256;

fn q8_pk_wgsl(sg: bool) -> String {
    let (prologue, template, legacy) = if sg {
        (
            GEMV_Q8_PK_SG_PROLOGUE,
            GEMV_Q8_PK_SG_TEMPLATE,
            GEMV_Q8_PK_SG_LEGACY,
        )
    } else {
        (
            GEMV_Q8_PK_TREE_PROLOGUE,
            GEMV_Q8_PK_TREE_TEMPLATE,
            GEMV_Q8_PK_TREE_LEGACY,
        )
    };
    let mut out = String::from(prologue);
    for (tag, acc) in [("fp8", "qg_group_acc_e4m3"), ("int8", "qg_group_acc_i8")] {
        out.push_str(&template.replace("TAG", tag).replace("ACC", acc));
    }
    out.push_str(legacy);
    out
}

const QUANT_PK_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4w_quant_pk.wgsl");

const HEAD_PREP_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4w_head_prep.wgsl");

const NORM_CHAIN_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4w_norm_chain.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct HpParams {
    n_q: u32,
    n_kv: u32,
    head_dim: u32,
    half_dim: u32,
    eps: f32,
    words: u32,
    out_words: u32,
    ring: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct NcParams {
    hidden: u32,
    words: u32,
    eps: f32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PkOffParams {
    dst_word_off: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitParams {
    q_rows: u32,
    kv_rows: u32,
    v_off: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct MkParams {
    m: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    dst_word_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PackParams {
    src_off: u32,
    dst_off: u32,
    n_words: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct Gather2Params {
    split_row: u32,
    hidden_words: u32,
    vocab: u32,
    pad0: u32,
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
struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ScaleParams {
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
struct GemvBf16Params {
    n_rows: u32,
    k_elems: u32,
    w_row_words: u32,
    groups_x: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RowQuantParams {
    n_rows: u32,
    k_elems: u32,
    src_row_words: u32,
    dst_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvI8Params {
    n_rows: u32,
    k_elems: u32,
    wq_row_words: u32,
    groups_x: u32,
    m_rows: u32,
    x_row_words: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvNvfp4Params {
    alpha: f32,
    n_rows: u32,
    k_blocks: u32,
    k_tiles: u32,
    w_row_words: u32,
    groups_x: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct QuantRowParams {
    global_scale: f32,
    k_blocks: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct RopeParams {
    n_heads: u32,
    half_dim: u32,
    total_words: u32,
    table_rows: u32,
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
struct GeluParams {
    inter: u32,
    inter_words: u32,
    rows: u32,
    tot_pairs: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ArgmaxRowsParams {
    rows: u32,
    n: u32,
    pad0: u32,
    pad1: u32,
}

pub struct HostBf16Lin {
    pub w: Vec<u16>,
    pub n: usize,
    pub k: usize,
}

pub struct HostFp8Lin {
    pub wq: Vec<u32>,
    pub row_scale: Vec<f32>,
    pub n: usize,
    pub k: usize,
    pub group: usize,
    pub fmt: wk::quant_gemv::QFormat,
}

pub struct HostW4a16Lin {
    pub packed: Vec<u32>,
    pub scales: Vec<u16>,
    pub n: usize,
    pub k: usize,
    pub group: usize,
}

pub enum HostProj {
    Bf16(HostBf16Lin),
    Nvfp4(HostNvfp4Lin),
    Fp8(HostFp8Lin),
}

impl HostProj {
    pub fn n(&self) -> usize {
        match self {
            HostProj::Bf16(l) => l.n,
            HostProj::Nvfp4(l) => l.n,
            HostProj::Fp8(l) => l.n,
        }
    }
    pub fn k(&self) -> usize {
        match self {
            HostProj::Bf16(l) => l.k,
            HostProj::Nvfp4(l) => l.k,
            HostProj::Fp8(l) => l.k,
        }
    }
}

pub struct HostLayer {
    pub kind: LayerType,
    pub input_ln: Vec<u16>,
    pub post_attn_ln: Vec<u16>,
    pub pre_ff_ln: Vec<u16>,
    pub post_ff_ln: Vec<u16>,
    pub q_norm: Vec<u16>,
    pub k_norm: Vec<u16>,
    pub layer_scalar: f32,
    pub has_v: bool,
    pub qkv: HostProj,
    pub o: HostProj,
    pub gate_up: HostProj,
    pub down: HostProj,
}

pub struct HostWeights {
    pub embed: Vec<u16>,
    pub final_norm: Vec<u16>,
    pub layers: Vec<HostLayer>,
}

pub use crate::nvfp4_host::{quantize_nvfp4_host, HostNvfp4Lin};

pub fn dequantize_nvfp4_host(lin: &HostNvfp4Lin) -> Vec<f32> {
    let k_blocks = lin.k / 16;
    let k_tiles = k_blocks.div_ceil(4);
    let mut out = vec![0f32; lin.n * lin.k];

    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .min(lin.n.max(1));
    let rows_per = lin.n.div_ceil(threads);
    std::thread::scope(|sc| {
        for (ci, chunk) in out.chunks_mut(rows_per * lin.k).enumerate() {
            let r0 = ci * rows_per;
            sc.spawn(move || {
                for (rr, row) in chunk.chunks_mut(lin.k).enumerate() {
                    let r = r0 + rr;
                    dequant_nvfp4_row(lin, r, k_blocks, k_tiles, row);
                }
            });
        }
    });
    out
}

fn dequant_nvfp4_row(
    lin: &HostNvfp4Lin,
    r: usize,
    k_blocks: usize,
    k_tiles: usize,
    row: &mut [f32],
) {
    {
        for kb in 0..k_blocks {
            let m_tile = r / 128;
            let d2 = (r / 32) % 4;
            let d3 = r % 32;
            let k_tile = kb / 4;
            let d5 = kb % 4;
            let si = ((m_tile * k_tiles + k_tile) * 32 + d3) * 16 + d2 * 4 + d5;
            let sb = lin.scales_swizzled[si] as u32;
            let e = (sb >> 3) & 15;
            let m = sb & 7;
            let s = if e == 0 {
                (m as f32) * 0.001953125f32
            } else {
                f32::from_bits(((e + 120) << 23) | (m << 20))
            };
            for j in 0..16 {
                let idx = kb * 16 + j;
                let byte = lin.packed[r * (lin.k / 2) + idx / 2];
                let nib = if idx % 2 == 0 { byte & 15 } else { byte >> 4 };
                const TABLE: [f32; 16] = [
                    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0,
                    -4.0, -6.0,
                ];
                row[idx] = TABLE[nib as usize] * s * lin.alpha;
            }
        }
    }
}

fn f32_to_bf16_bits_rne(x: f32) -> u16 {
    let u = x.to_bits();
    let round = ((u >> 16) & 1) + 0x7fff;
    (u.wrapping_add(round) >> 16) as u16
}

fn par_bf16_bits(wf: &[f32], gi: f32) -> Vec<u16> {
    let mut bits = vec![0u16; wf.len()];
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .max(1);
    let per = wf.len().div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (ci, chunk) in bits.chunks_mut(per).enumerate() {
            let src = &wf[ci * per..(ci * per + chunk.len())];
            sc.spawn(move || {
                for (d, &v) in chunk.iter_mut().zip(src) {
                    *d = f32_to_bf16_bits_rne(v * gi);
                }
            });
        }
    });
    bits
}

pub const W8_FFN_DEFAULT: &str = "all";

fn gelu_fold_enabled() -> bool {
    std::env::var("NV_G4_WGPU_GELU_FOLD").ok().as_deref() != Some("0")
}

fn w8_ffn_mode() -> (bool, bool, usize) {
    let v = std::env::var("NV_G4_WGPU_W8_FFN").unwrap_or_else(|_| W8_FFN_DEFAULT.to_string());
    let group = std::env::var("NV_G4_WGPU_W8_FFN_GROUP")
        .ok()
        .and_then(|g| g.trim().parse::<usize>().ok())
        .unwrap_or(128);
    match v.trim() {
        "down" => (false, true, group),
        "1" | "all" => (true, true, group),
        _ => (false, false, group),
    }
}

static CONV_DEQ_NS: AtomicU64 = AtomicU64::new(0);
static CONV_BF16_NS: AtomicU64 = AtomicU64::new(0);
static CONV_Q8_NS: AtomicU64 = AtomicU64::new(0);
static CONV_ATTN_Q8_NS: AtomicU64 = AtomicU64::new(0);

fn conv_add(c: &AtomicU64, t: Instant) {
    c.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

fn nvfp4_to_i8(l: &HostNvfp4Lin, group: usize, li: usize, role: ProjRole) -> Option<HostProj> {
    if wk::quant_gemv::group_rule(l.k, group).is_err() || !l.n.is_multiple_of(2) {
        return None;
    }
    let gi = if l.input_global == 0.0 || !l.input_global.is_finite() {
        1.0
    } else {
        l.input_global
    };
    let t = Instant::now();
    let wf = dequantize_nvfp4_host(l);
    conv_add(&CONV_DEQ_NS, t);
    let t = Instant::now();
    let bits = par_bf16_bits(&wf, gi);
    conv_add(&CONV_BF16_NS, t);
    let bits = match w4_preview(&bits, l.n, l.k, li, role) {
        Some(b) => b,
        None => bits,
    };
    let t = Instant::now();
    let q = quantize_q8_host(&bits, l.n, l.k, group, wk::quant_gemv::QFormat::Int8);
    conv_add(&CONV_Q8_NS, t);
    Some(HostProj::Fp8(q))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ProjRole {
    Attn,
    GateUp,
    Down,
}

#[derive(Clone, Copy, Debug)]
struct Q8Plan {
    attn_fp8: bool,
    aq: AttnQuant,
    w8_gate_up: bool,
    w8_down: bool,
    w8_group: usize,
}

impl Q8Plan {
    fn from_env() -> Self {
        let (w8_gate_up, w8_down, w8_group) = w8_ffn_mode();
        Self {
            attn_fp8: attn_fp8_enabled(),
            aq: attn_quant_config(),
            w8_gate_up,
            w8_down,
            w8_group,
        }
    }

    fn plans_fp8(&self, p: &HostProj, li: usize, role: ProjRole) -> bool {
        match (p, role) {
            (HostProj::Bf16(l), ProjRole::Attn) => {
                self.attn_fp8
                    && self.aq.covers(li)
                    && wk::quant_gemv::group_rule(l.k, self.aq.group).is_ok()
                    && l.n % 2 == 0
            }
            (HostProj::Nvfp4(l), ProjRole::GateUp | ProjRole::Down) => {
                let on = match role {
                    ProjRole::GateUp => self.w8_gate_up,
                    _ => self.w8_down,
                };
                on && wk::quant_gemv::group_rule(l.k, self.w8_group).is_ok() && l.n % 2 == 0
            }
            (HostProj::Fp8(_), _) => true,
            _ => false,
        }
    }

    #[allow(clippy::wrong_self_convention)]
    fn to_fp8(&self, p: &HostProj, li: usize, role: ProjRole) -> Option<HostProj> {
        if !self.plans_fp8(p, li, role) {
            return None;
        }
        match p {
            HostProj::Bf16(l) => {
                let owned = w4_preview(&l.w, l.n, l.k, li, role);
                Some(HostProj::Fp8(quantize_q8_host(
                    owned.as_deref().unwrap_or(&l.w),
                    l.n,
                    l.k,
                    self.aq.group,
                    self.aq.fmt,
                )))
            }
            HostProj::Nvfp4(l) => nvfp4_to_i8(l, self.w8_group, li, role),
            HostProj::Fp8(_) => None,
        }
    }

    fn any_fp8(&self, weights: &HostWeights) -> bool {
        weights.layers.iter().enumerate().any(|(li, hl)| {
            [
                (ProjRole::Attn, &hl.qkv),
                (ProjRole::Attn, &hl.o),
                (ProjRole::GateUp, &hl.gate_up),
                (ProjRole::Down, &hl.down),
            ]
            .into_iter()
            .any(|(role, p)| self.plans_fp8(p, li, role))
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum W4ScalePolicy {
    Amax,
    MseSearch,
}

const W4_SCALE_LADDER: [f32; 11] = [
    0.7,
    0.8,
    0.85,
    0.9,
    0.95,
    1.0,
    1.05,
    1.1,
    7.0 / 6.0,
    1.25,
    1.4,
];

fn w4a16_group_scale(chunk: &[u16], policy: W4ScalePolicy) -> u16 {
    let mut amax = 0f32;
    for b in chunk {
        let v = f32::from_bits((*b as u32) << 16).abs();
        if v > amax {
            amax = v;
        }
    }
    if !(amax > 0.0) || !amax.is_finite() {
        return 0;
    }

    let base = amax / 7.0;
    if policy == W4ScalePolicy::Amax {
        return f32_to_bf16_bits_rne(base);
    }
    let mut best_bits = f32_to_bf16_bits_rne(base);
    let mut best_err = f64::INFINITY;
    for t in W4_SCALE_LADDER {
        let bits = f32_to_bf16_bits_rne(base * t);
        let s = f32::from_bits((bits as u32) << 16);
        if !(s > 0.0) {
            continue;
        }
        let mut err = 0f64;
        for b in chunk {
            let v = f32::from_bits((*b as u32) << 16);
            let q = (v / s).round().clamp(-8.0, 7.0);
            let d = (q * s - v) as f64;
            err += d * d;
        }
        if err < best_err {
            best_err = err;
            best_bits = bits;
        }
    }
    best_bits
}

pub fn quantize_w4a16_host(
    w: &[u16],
    n: usize,
    k: usize,
    group: usize,
    policy: W4ScalePolicy,
) -> HostW4a16Lin {
    assert_eq!(w.len(), n * k, "quantize_w4a16_host: w is not n*k");
    assert!(
        group > 0 && k.is_multiple_of(group) && group.is_multiple_of(8) && k.is_multiple_of(32),
        "quantize_w4a16_host: K={k} GS={group} violates K%32==0, K%GS==0, GS%8==0"
    );
    let per_row = k / group;
    let mut packed = vec![0u32; n * k / 8];
    let mut scales = vec![0u16; n * per_row];
    for r in 0..n {
        for g in 0..per_row {
            let off = r * k + g * group;
            let chunk = &w[off..off + group];
            let sbits = w4a16_group_scale(chunk, policy);
            scales[r * per_row + g] = sbits;
            let s = f32::from_bits((sbits as u32) << 16);
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for (i, b) in chunk.iter().enumerate() {
                let v = f32::from_bits((*b as u32) << 16) * inv;
                let q = (v.round().clamp(-8.0, 7.0) + 8.0) as u32;
                let idx = off + i;
                packed[idx / 8] |= q << (4 * (idx % 8));
            }
        }
    }
    HostW4a16Lin {
        packed,
        scales,
        n,
        k,
        group,
    }
}

pub fn dequantize_w4a16_host(l: &HostW4a16Lin) -> Vec<f32> {
    let per_row = l.k / l.group;
    let mut out = vec![0f32; l.n * l.k];
    for r in 0..l.n {
        for g in 0..per_row {
            let s = f32::from_bits((l.scales[r * per_row + g] as u32) << 16);
            for i in 0..l.group {
                let idx = r * l.k + g * l.group + i;
                let q = (l.packed[idx / 8] >> (4 * (idx % 8))) & 15;
                out[idx] = (q as f32 - 8.0) * s;
            }
        }
    }
    out
}

pub fn nvfp4_to_w4a16(
    l: &HostNvfp4Lin,
    group: usize,
    policy: W4ScalePolicy,
) -> Option<HostW4a16Lin> {
    if wk::gemv_w4a16::shape_rule(l.k, group).is_err() {
        return None;
    }
    let gi = if l.input_global == 0.0 || !l.input_global.is_finite() {
        1.0
    } else {
        l.input_global
    };
    let wf = dequantize_nvfp4_host(l);
    let bits: Vec<u16> = wf.iter().map(|&v| f32_to_bf16_bits_rne(v * gi)).collect();
    Some(quantize_w4a16_host(&bits, l.n, l.k, group, policy))
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum W4Method {
    Rtn,
    RtnMse,
    Wmse,
    Awq,
    Gptq,
    AwqGptq,
}

impl W4Method {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s.trim() {
            "rtn" => Self::Rtn,
            "mse" | "rtn-mse" => Self::RtnMse,
            "wmse" => Self::Wmse,
            "awq" => Self::Awq,
            "gptq" => Self::Gptq,
            "awq+gptq" | "awqgptq" => Self::AwqGptq,
            _ => return None,
        })
    }
    pub fn label(&self) -> &'static str {
        match self {
            Self::Rtn => "rtn",
            Self::RtnMse => "rtn-mse",
            Self::Wmse => "wmse",
            Self::Awq => "awq",
            Self::Gptq => "gptq",
            Self::AwqGptq => "awq+gptq",
        }
    }
    pub fn needs_calibration(&self) -> bool {
        matches!(self, Self::Wmse | Self::Awq | Self::Gptq | Self::AwqGptq)
    }
    fn awq(&self) -> bool {
        matches!(self, Self::Awq | Self::AwqGptq)
    }
    fn gptq(&self) -> bool {
        matches!(self, Self::Gptq | Self::AwqGptq)
    }

    fn weighted(&self) -> bool {
        !matches!(self, Self::Rtn | Self::RtnMse)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct W4PtqSpec {
    pub method: W4Method,
    pub group: usize,
    pub gate_up: bool,
    pub down: bool,
    pub attn: bool,
    pub lo: usize,
    pub hi: usize,
    pub alphas: usize,
    pub block: usize,
    pub damp: f32,
}

impl Default for W4PtqSpec {
    fn default() -> Self {
        Self {
            method: W4Method::RtnMse,
            group: 32,
            gate_up: true,
            down: true,
            attn: false,
            lo: 0,
            hi: usize::MAX,
            alphas: 11,
            block: 128,
            damp: 0.01,
        }
    }
}

impl W4PtqSpec {
    pub fn from_env() -> Option<Self> {
        let raw = std::env::var("NV_G4_W4PTQ").ok()?;
        let raw = raw.trim();
        if raw.is_empty() || raw == "off" || raw == "0" {
            return None;
        }
        let mut s = Self::default();
        for kv in raw.split(',').filter(|t| !t.trim().is_empty()) {
            let (k, v) = kv
                .split_once('=')
                .unwrap_or_else(|| panic!("NV_G4_W4PTQ: '{kv}' is not key=value"));
            let (k, v) = (k.trim(), v.trim());
            match k {
                "m" => {
                    s.method =
                        W4Method::parse(v).unwrap_or_else(|| panic!("NV_G4_W4PTQ: m={v} unknown"))
                }
                "r" => {
                    let (mut gu, mut dn, mut at) = (false, false, false);
                    for r in v.split('+') {
                        match r.trim() {
                            "ffn" => {
                                gu = true;
                                dn = true;
                            }
                            "gate_up" | "gu" => gu = true,
                            "down" | "dn" => dn = true,
                            "attn" => at = true,
                            "all" => {
                                gu = true;
                                dn = true;
                                at = true;
                            }
                            o => panic!("NV_G4_W4PTQ: r={o} unknown"),
                        }
                    }
                    s.gate_up = gu;
                    s.down = dn;
                    s.attn = at;
                }
                "g" => s.group = v.parse().expect("NV_G4_W4PTQ: g"),
                "l" => {
                    let (a, b) = v.split_once(':').expect("NV_G4_W4PTQ: l=lo:hi");
                    s.lo = a.parse().expect("l lo");
                    s.hi = b.parse().expect("l hi");
                }
                "a" => s.alphas = v.parse().expect("NV_G4_W4PTQ: a"),
                "b" => s.block = v.parse().expect("NV_G4_W4PTQ: b"),
                "d" => s.damp = v.parse().expect("NV_G4_W4PTQ: d"),
                o => panic!("NV_G4_W4PTQ: key '{o}' unknown"),
            }
        }
        assert!(
            s.group >= 8 && s.group.is_multiple_of(8),
            "NV_G4_W4PTQ: g={} must be a multiple of 8",
            s.group
        );
        assert!(
            !s.method.gptq() || s.block.is_multiple_of(s.group),
            "NV_G4_W4PTQ: gptq block {} must be a multiple of the 4-bit group {}",
            s.block,
            s.group
        );
        Some(s)
    }

    pub fn label(&self) -> String {
        let mut roles: Vec<&str> = Vec::new();
        if self.gate_up {
            roles.push("gate_up");
        }
        if self.down {
            roles.push("down");
        }
        if self.attn {
            roles.push("attn");
        }
        let band = if self.hi == usize::MAX && self.lo == 0 {
            String::new()
        } else {
            format!("@[{},{})", self.lo, self.hi)
        };
        format!(
            "int4/{} {} on {}{band}",
            self.group,
            self.method.label(),
            roles.join("+")
        )
    }

    pub fn bytes_per_weight(&self) -> f64 {
        0.5 + 2.0 / self.group as f64
    }

    fn covers(&self, li: usize, role: ProjRole) -> bool {
        if li < self.lo || li >= self.hi {
            return false;
        }
        match role {
            ProjRole::Attn => self.attn,
            ProjRole::GateUp => self.gate_up,
            ProjRole::Down => self.down,
        }
    }
}

pub struct W4ChanStats {
    pub k: usize,
    pub block: usize,
    pub sumabs: Vec<f32>,
    pub sumsq: Vec<f32>,
    pub absmax: Vec<f32>,
    pub gram: Vec<f32>,
    pub tokens: usize,
}

impl W4ChanStats {
    pub fn new(k: usize, block: usize) -> Self {
        assert!(k.is_multiple_of(block), "K={k} not a multiple of block");
        Self {
            k,
            block,
            sumabs: vec![0.0; k],
            sumsq: vec![0.0; k],
            absmax: vec![0.0; k],
            gram: vec![0.0; k * block],
            tokens: 0,
        }
    }

    pub fn observe(&mut self, x: &[f32]) {
        assert_eq!(x.len(), self.k, "calibration row width");
        for (j, &v) in x.iter().enumerate() {
            let a = v.abs();
            self.sumabs[j] += a;
            self.sumsq[j] += v * v;
            if a > self.absmax[j] {
                self.absmax[j] = a;
            }
        }
        let b = self.block;
        let nb = self.k / b;
        let gram = &mut self.gram;
        let mut chunks: Vec<&mut [f32]> = gram.chunks_mut(b * b).collect();
        let threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(1)
            .max(1);
        let per = nb.div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for (ci, part) in chunks.chunks_mut(per).enumerate() {
                let base = ci * per;
                sc.spawn(move || {
                    for (o, g) in part.iter_mut().enumerate() {
                        let xb = &x[(base + o) * b..(base + o) * b + b];
                        for i in 0..b {
                            let xi = xb[i];
                            if xi == 0.0 {
                                continue;
                            }
                            let row = &mut g[i * b..i * b + b];
                            for (t, xj) in row.iter_mut().zip(xb.iter()) {
                                *t += xi * xj;
                            }
                        }
                    }
                });
            }
        });
        self.tokens += 1;
    }

    fn salience(&self) -> Vec<f32> {
        let t = self.tokens.max(1) as f32;
        self.sumabs.iter().map(|s| s / t).collect()
    }

    fn hdiag(&self) -> Vec<f32> {
        self.sumsq.clone()
    }

    pub fn hdiag_for_test(&self) -> Vec<f32> {
        self.hdiag()
    }
}

pub fn gptq_factor_for_test(gram: &[f32], n: usize, damp: f32) -> Option<Vec<f64>> {
    gptq_factor(gram, n, damp)
}

pub struct W4Calib {
    pub gate_up: Vec<W4ChanStats>,
    pub down: Vec<W4ChanStats>,
    pub tokens: usize,
    pub prompts: usize,
}

static W4_CALIB: Mutex<Option<Arc<W4Calib>>> = Mutex::new(None);
static W4_PROJ: AtomicUsize = AtomicUsize::new(0);
static W4_ELEMS: AtomicU64 = AtomicU64::new(0);
static W4_SE: Mutex<(f64, f64)> = Mutex::new((0.0, 0.0));

pub fn set_w4_calibration(c: Option<Arc<W4Calib>>) {
    *W4_CALIB.lock().unwrap_or_else(|e| e.into_inner()) = c;
}

fn w4_calibration() -> Option<Arc<W4Calib>> {
    W4_CALIB
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .map(Arc::clone)
}

pub fn w4_ptq_report() -> (usize, u64, f64) {
    let (se, sr) = *W4_SE.lock().unwrap_or_else(|e| e.into_inner());
    let rel = if sr > 0.0 { (se / sr).sqrt() } else { 0.0 };
    (
        W4_PROJ.load(Ordering::Relaxed),
        W4_ELEMS.load(Ordering::Relaxed),
        rel,
    )
}

pub fn w4_ptq_reset() {
    W4_PROJ.store(0, Ordering::Relaxed);
    W4_ELEMS.store(0, Ordering::Relaxed);
    *W4_SE.lock().unwrap_or_else(|e| e.into_inner()) = (0.0, 0.0);
}

fn w4_group_scale(chunk: &[f32], h: Option<&[f32]>, ladder: bool) -> f32 {
    let mut amax = 0f32;
    for v in chunk {
        let a = v.abs();
        if a > amax {
            amax = a;
        }
    }
    if !(amax > 0.0) || !amax.is_finite() {
        return 0.0;
    }
    let base = amax / 7.0;
    let bf = |x: f32| f32::from_bits((f32_to_bf16_bits_rne(x) as u32) << 16);
    if !ladder {
        return bf(base);
    }
    let mut best = bf(base);
    let mut best_err = f64::INFINITY;
    for t in W4_SCALE_LADDER {
        let s = bf(base * t);
        if !(s > 0.0) {
            continue;
        }
        let mut err = 0f64;
        for (i, v) in chunk.iter().enumerate() {
            let q = (v / s).round().clamp(-8.0, 7.0);
            let d = (q * s - v) as f64;
            err += d * d * h.map_or(1.0, |w| w[i] as f64);
        }
        if err < best_err {
            best_err = err;
            best = s;
        }
    }
    best
}

fn w4_inv(s: f32) -> f32 {
    if s > 0.0 {
        1.0 / s
    } else {
        0.0
    }
}

fn w4_round(v: f32, s: f32, inv: f32) -> f32 {
    if s > 0.0 {
        (v * inv).round().clamp(-8.0, 7.0) * s
    } else {
        0.0
    }
}

fn w4_rtn_rows(w: &mut [f32], k: usize, group: usize, h: Option<&[f32]>, ladder: bool) {
    for r in w.chunks_mut(k) {
        for g in 0..k / group {
            let hs = h.map(|v| &v[g * group..g * group + group]);
            let chunk = &mut r[g * group..g * group + group];
            let s = w4_group_scale(chunk, hs, ladder);
            let inv = w4_inv(s);
            for v in chunk.iter_mut() {
                *v = w4_round(*v, s, inv);
            }
        }
    }
}

fn chol_lower(a: &mut [f64], n: usize) -> bool {
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for t in 0..j {
                s -= a[i * n + t] * a[j * n + t];
            }
            if i == j {
                if s <= 0.0 {
                    return false;
                }
                a[i * n + j] = s.sqrt();
            } else {
                a[i * n + j] = s / a[j * n + j];
            }
        }
        for j in i + 1..n {
            a[i * n + j] = 0.0;
        }
    }
    true
}

fn gptq_factor(gram: &[f32], n: usize, damp: f32) -> Option<Vec<f64>> {
    let mut h = vec![0f64; n * n];
    let mut mean = 0f64;
    for i in 0..n {
        mean += gram[i * n + i] as f64;
    }
    mean /= n as f64;
    if !(mean > 0.0) {
        return None;
    }
    let lam = damp as f64 * mean;
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = 0.5 * (gram[i * n + j] as f64 + gram[j * n + i] as f64);
        }
        if h[i * n + i] <= 0.0 {
            h[i * n + i] = mean;
        }
        h[i * n + i] += lam;
    }
    let mut l = h.clone();
    if !chol_lower(&mut l, n) {
        return None;
    }
    let mut li = vec![0f64; n * n];
    for i in 0..n {
        li[i * n + i] = 1.0 / l[i * n + i];
        for j in 0..i {
            let mut s = 0f64;
            for t in j..i {
                s += l[i * n + t] * li[t * n + j];
            }
            li[i * n + j] = -s / l[i * n + i];
        }
    }
    let mut hinv = vec![0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut s = 0f64;
            for t in i.max(j)..n {
                s += li[t * n + i] * li[t * n + j];
            }
            hinv[i * n + j] = s;
        }
    }
    let mut lo = hinv;
    if !chol_lower(&mut lo, n) {
        return None;
    }
    let mut u = vec![0f64; n * n];
    for i in 0..n {
        for j in i..n {
            u[i * n + j] = lo[j * n + i];
        }
    }
    Some(u)
}

fn gptq_rows(
    w: &mut [f32],
    k: usize,
    group: usize,
    block: usize,
    h: &[f32],
    factors: &[Option<Vec<f64>>],
) {
    let b = block;
    for row in w.chunks_mut(k) {
        for (bi, fac) in factors.iter().enumerate() {
            let cols = &mut row[bi * b..bi * b + b];
            let Some(u) = fac else {
                let hs = &h[bi * b..bi * b + b];
                for g in 0..b / group {
                    let chunk = &mut cols[g * group..g * group + group];
                    let s = w4_group_scale(chunk, Some(&hs[g * group..g * group + group]), true);
                    let inv = w4_inv(s);
                    for v in chunk.iter_mut() {
                        *v = w4_round(*v, s, inv);
                    }
                }
                continue;
            };
            let (mut scale, mut inv) = (0f32, 0f32);
            for i in 0..b {
                if i.is_multiple_of(group) {
                    let hs = &h[bi * b + i..bi * b + i + group];
                    scale = w4_group_scale(&cols[i..i + group], Some(hs), true);
                    inv = w4_inv(scale);
                }
                let q = w4_round(cols[i], scale, inv);
                let d = u[i * b + i];
                let err = if d.abs() > 1e-12 {
                    (cols[i] - q) as f64 / d
                } else {
                    0.0
                };
                cols[i] = q;
                if err != 0.0 {
                    for j in i + 1..b {
                        cols[j] -= (err * u[i * b + j]) as f32;
                    }
                }
            }
        }
    }
}

fn par_rows<F>(w: &mut [f32], k: usize, f: F)
where
    F: Fn(&mut [f32]) + Sync,
{
    let n = w.len() / k;
    let threads = std::thread::available_parallelism()
        .map(|t| t.get())
        .unwrap_or(1)
        .max(1)
        .min(n.max(1));
    let per = n.div_ceil(threads).max(1) * k;
    std::thread::scope(|sc| {
        for part in w.chunks_mut(per) {
            let f = &f;
            sc.spawn(move || f(part));
        }
    });
}

fn weighted_err(a: &[f32], b: &[f32], k: usize, h: &[f32]) -> f64 {
    let mut e = 0f64;
    for (ra, rb) in a.chunks(k).zip(b.chunks(k)) {
        for j in 0..k {
            let d = (ra[j] - rb[j]) as f64;
            e += d * d * h[j] as f64;
        }
    }
    e
}

fn awq_alpha(w: &[f32], n: usize, k: usize, spec: &W4PtqSpec, st: &W4ChanStats) -> (f32, f64, f64) {
    let sal = st.salience();
    let h = st.hdiag();
    let mut logsum = 0f64;
    let mut live = 0usize;
    for v in &sal {
        if *v > 0.0 {
            logsum += (*v as f64).ln();
            live += 1;
        }
    }
    let gm = if live > 0 {
        (logsum / live as f64).exp() as f32
    } else {
        1.0
    };
    let stride = (n / 2048).max(1);
    let rows: Vec<usize> = (0..n).step_by(stride).collect();
    let mut sub = Vec::with_capacity(rows.len() * k);
    for r in &rows {
        sub.extend_from_slice(&w[r * k..r * k + k]);
    }
    let mut best = (0f32, f64::INFINITY, f64::INFINITY);
    for ai in 0..spec.alphas.max(1) {
        let alpha = ai as f32 / (spec.alphas.max(2) - 1) as f32;
        let s: Vec<f32> = sal
            .iter()
            .map(|v| {
                let r = if *v > 0.0 { *v / gm } else { 1.0 };
                let p = r.powf(alpha);
                if p.is_finite() && p > 1e-4 && p < 1e4 {
                    p
                } else {
                    1.0
                }
            })
            .collect();
        let mut trial = sub.clone();
        for row in trial.chunks_mut(k) {
            for (v, sj) in row.iter_mut().zip(s.iter()) {
                *v *= sj;
            }
        }
        let hs: Vec<f32> = h.iter().zip(s.iter()).map(|(a, b)| a / (b * b)).collect();
        par_rows(&mut trial, k, |part| {
            w4_rtn_rows(part, k, spec.group, Some(&hs), true)
        });
        for row in trial.chunks_mut(k) {
            for (v, sj) in row.iter_mut().zip(s.iter()) {
                *v /= sj;
            }
        }
        let e = weighted_err(&sub, &trial, k, &h);
        if ai == 0 {
            best.2 = e;
        }
        if e < best.1 {
            best = (alpha, e, best.2);
        }
    }
    best
}

pub fn w4_project(w: &mut [f32], n: usize, k: usize, spec: &W4PtqSpec, st: Option<&W4ChanStats>) {
    assert!(
        k.is_multiple_of(spec.group),
        "w4_project: K={k} not a multiple of group {}",
        spec.group
    );
    if spec.method.needs_calibration() && st.is_none() {
        panic!(
            "w4_project: method {} needs calibration for a K={k} input and none was supplied. \
             Falling back to RTN here would measure RTN and print the AWQ label.",
            spec.method.label()
        );
    }
    let mut scales: Option<Vec<f32>> = None;
    if spec.method.awq() {
        let st = st.unwrap();
        let (alpha, e_best, e_rtn) = awq_alpha(w, n, k, spec, st);
        eprintln!(
            "[w4ptq] awq K={k} alpha={alpha:.2} weighted-err {e_best:.6e} vs rtn {e_rtn:.6e} \
             ({:.3}x)",
            e_best / e_rtn.max(f64::MIN_POSITIVE)
        );
        let sal = st.salience();
        let mut logsum = 0f64;
        let mut live = 0usize;
        for v in &sal {
            if *v > 0.0 {
                logsum += (*v as f64).ln();
                live += 1;
            }
        }
        let gm = if live > 0 {
            (logsum / live as f64).exp() as f32
        } else {
            1.0
        };
        let s: Vec<f32> = sal
            .iter()
            .map(|v| {
                let r = if *v > 0.0 { *v / gm } else { 1.0 };
                let p = r.powf(alpha);
                if p.is_finite() && p > 1e-4 && p < 1e4 {
                    p
                } else {
                    1.0
                }
            })
            .collect();
        par_rows(w, k, |part| {
            for row in part.chunks_mut(k) {
                for (v, sj) in row.iter_mut().zip(s.iter()) {
                    *v *= sj;
                }
            }
        });
        scales = Some(s);
    }
    let h: Vec<f32> = match st.filter(|_| spec.method.weighted()) {
        Some(s) => {
            let d = s.hdiag();
            match &scales {
                Some(sc) => d.iter().zip(sc.iter()).map(|(a, b)| a / (b * b)).collect(),
                None => d,
            }
        }
        None => vec![1.0; k],
    };
    if spec.method.gptq() {
        let st = st.unwrap();
        let b = st.block;
        let t = Instant::now();

        let factors: Vec<Option<Vec<f64>>> = (0..k / b)
            .map(|bi| {
                let g = &st.gram[bi * b * b..bi * b * b + b * b];
                match &scales {
                    Some(sc) => {
                        let ss = &sc[bi * b..bi * b + b];
                        let mut gg = vec![0f32; b * b];
                        for i in 0..b {
                            for j in 0..b {
                                gg[i * b + j] = g[i * b + j] / (ss[i] * ss[j]);
                            }
                        }
                        gptq_factor(&gg, b, spec.damp)
                    }
                    None => gptq_factor(g, b, spec.damp),
                }
            })
            .collect();
        let dead = factors.iter().filter(|f| f.is_none()).count();
        eprintln!(
            "[w4ptq] gptq K={k} {} blocks of {b}, {dead} singular (RTN there), factor {:.1}s",
            factors.len(),
            t.elapsed().as_secs_f64()
        );
        par_rows(w, k, |part| gptq_rows(part, k, spec.group, b, &h, &factors));
    } else {
        let ladder = spec.method != W4Method::Rtn;
        let hw = spec.method.weighted().then_some(h.as_slice());
        par_rows(w, k, |part| w4_rtn_rows(part, k, spec.group, hw, ladder));
    }
    if let Some(s) = scales {
        par_rows(w, k, |part| {
            for row in part.chunks_mut(k) {
                for (v, sj) in row.iter_mut().zip(s.iter()) {
                    *v /= sj;
                }
            }
        });
    }
}

fn w4_preview(bits: &[u16], n: usize, k: usize, li: usize, role: ProjRole) -> Option<Vec<u16>> {
    let spec = W4PtqSpec::from_env()?;
    if !spec.covers(li, role) {
        return None;
    }
    let calib = w4_calibration();
    let st = calib.as_ref().and_then(|c| match role {
        ProjRole::GateUp => c.gate_up.get(li),
        ProjRole::Down => c.down.get(li),
        ProjRole::Attn => None,
    });
    if spec.method.needs_calibration() && st.is_none() {
        panic!(
            "NV_G4_W4PTQ method {} covers layer {li} {role:?} but no calibration is loaded for it",
            spec.method.label()
        );
    }
    let mut w: Vec<f32> = bits
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();
    let before = w.clone();
    w4_project(&mut w, n, k, &spec, st);
    let (mut se, mut sr) = (0f64, 0f64);
    for (a, b) in before.iter().zip(w.iter()) {
        let d = (*b - *a) as f64;
        se += d * d;
        sr += (*a as f64) * (*a as f64);
    }
    {
        let mut g = W4_SE.lock().unwrap_or_else(|e| e.into_inner());
        g.0 += se;
        g.1 += sr;
    }
    W4_PROJ.fetch_add(1, Ordering::Relaxed);
    W4_ELEMS.fetch_add(w.len() as u64, Ordering::Relaxed);
    Some(w.iter().map(|v| f32_to_bf16_bits_rne(*v)).collect())
}

pub fn quantize_fp8_host(w: &[u16], n: usize, k: usize) -> HostFp8Lin {
    quantize_q8_host(w, n, k, 0, wk::quant_gemv::QFormat::E4m3)
}

pub fn quantize_q8_host(
    w: &[u16],
    n: usize,
    k: usize,
    group: usize,
    fmt: wk::quant_gemv::QFormat,
) -> HostFp8Lin {
    let (wq, row_scale) = wk::quant_gemv::quantize_groups(w, n, k, group, fmt);
    HostFp8Lin {
        wq,
        row_scale,
        n,
        k,
        group,
        fmt,
    }
}

struct Pass {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind: wgpu::BindGroup,
    grid: (u32, u32, u32),
    label: String,
}

enum GpuProj {
    Bf16 {
        w: GpuTensor<u32>,
        params: GpuUniform<GemvBf16Params>,
        grid: (u32, u32, u32),
    },
    Nvfp4 {
        w: GpuTensor<u32>,
        scales: GpuTensor<u32>,
        gemv_params: GpuUniform<GemvNvfp4Params>,
        quant_params: GpuUniform<QuantRowParams>,
        grid: (u32, u32, u32),
        k: usize,
        n: usize,
        alpha: f32,
        mk_input_globals_one_per_slot: GpuTensor<f32>,
        deep: bool,
        v2: Option<(Arc<wgpu::ComputePipeline>, wk::gemv_nvfp4_v2::V2Kernel)>,
    },
    Fp8 {
        w: GpuTensor<u32>,
        row_scale: GpuTensor<f32>,
        params: GpuUniform<wk::quant_gemv::QuantGemvParams>,
        grid: (u32, u32, u32),
        fmt: wk::quant_gemv::QFormat,
    },
}

struct GpuLayerKv {
    k_fp8: GpuTensor<u32>,
    v_fp8: GpuTensor<u32>,
    k_scales: GpuTensor<f32>,
    v_scales: GpuTensor<f32>,
}

struct Pipelines {
    gather: Arc<wgpu::ComputePipeline>,
    scale: Arc<wgpu::ComputePipeline>,
    rms: Arc<wgpu::ComputePipeline>,
    rmsres: Arc<wgpu::ComputePipeline>,
    resadd: Arc<wgpu::ComputePipeline>,
    cast_f32: Arc<wgpu::ComputePipeline>,
    softcap: Arc<wgpu::ComputePipeline>,
    rope: Arc<wgpu::ComputePipeline>,
    rope_f32: Arc<wgpu::ComputePipeline>,
    kvq: Arc<wgpu::ComputePipeline>,
    flash1: Arc<wgpu::ComputePipeline>,
    flash2_pk: Arc<wgpu::ComputePipeline>,
    gemv_pk: Arc<wgpu::ComputePipeline>,
    gemv_pk3: Arc<wgpu::ComputePipeline>,
    rowquant_i8: Arc<wgpu::ComputePipeline>,
    gemv_i8: Arc<wgpu::ComputePipeline>,
    gemv4_pk: Arc<wgpu::ComputePipeline>,
    gemv4_pk_deep: Option<Arc<wgpu::ComputePipeline>>,
    quant_pk: Arc<wgpu::ComputePipeline>,
    gelu_even: Arc<wgpu::ComputePipeline>,
    am1: Arc<wgpu::ComputePipeline>,
    am2: Arc<wgpu::ComputePipeline>,
    pack16: Arc<wgpu::ComputePipeline>,
    fp8: Option<Fp8Pipelines>,
    fuse: Option<FusePipelines>,
    mk: Option<MkPipelines>,
    splice: Arc<wgpu::ComputePipeline>,
}

struct Fp8Pipelines {
    pk: Arc<wgpu::ComputePipeline>,
    pk3: Arc<wgpu::ComputePipeline>,
    i8_pk: Arc<wgpu::ComputePipeline>,
    i8_pk3: Arc<wgpu::ComputePipeline>,

    i8_gelu: Arc<wgpu::ComputePipeline>,
    rows_per_group: u32,
}

struct MkPipelines {
    gather: Arc<wgpu::ComputePipeline>,
    bf16_pk: Arc<wgpu::ComputePipeline>,
    bf16_pk3: Arc<wgpu::ComputePipeline>,
    q8: Option<MkQ8Pipelines>,
}

struct MkQ8Pipelines {
    i8_pk: Arc<wgpu::ComputePipeline>,
    i8_pk3: Arc<wgpu::ComputePipeline>,
    fp8_pk: Arc<wgpu::ComputePipeline>,
    fp8_pk3: Arc<wgpu::ComputePipeline>,
}

struct Mk4SlotBufs {
    xq: GpuTensor<u32>,
    xs: GpuTensor<u32>,
    xm_i8mapped_for_the_slotshared_gemv: GpuTensor<u32>,
    sel_all_zero_because_gemma4_has_no_experts: GpuTensor<u32>,
    alphas_unread_when_per_expert_alpha_is_zero: GpuTensor<f32>,
}

const MK4_SLOT_V2_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_gemv_nvfp4_v2.wgsl");

struct Mk4V2Route {
    source: String,
    entry: &'static str,
    label: String,
    rows_per_group: u32,
    vec4: bool,
}

fn mk4_slot_v2_route(ctx: &WgpuContext, n: usize, k: usize, slots: usize) -> Option<Mk4V2Route> {
    let k_blocks = k / 16;
    if !nvfp4_v2_enabled(ctx)
        || !k_blocks.is_multiple_of(4)
        || !(n * k_blocks).is_multiple_of(2)
        || n < 2
        || slots == 0
    {
        return None;
    }
    if let Some(r) = q3m::nvfp4_v2_route_slotshared(ctx, n, k, k_blocks, slots) {
        return Some(Mk4V2Route {
            label: format!("g4w-mk4-v2-{}-256x1", r.entry),
            source: r.source,
            entry: r.entry,
            rows_per_group: r.rows_per_group,
            vec4: r.vec4,
        });
    }
    let (kernel, cfg, pk_entry) = wk::gemv_nvfp4_v2::select_pk_slots(n, k, slots)?;
    let entry = match pk_entry {
        wk::gemv_nvfp4_v2::FMLUT_PK_ENTRY => "q3w_gemv_nvfp4_fmlut",
        wk::gemv_nvfp4_v2::FDEC_PK_ENTRY => "q3w_gemv_nvfp4_fdec",
        wk::gemv_nvfp4_v2::WARP_PK_ENTRY => "q3w_gemv_nvfp4_warp",
        _ => return None,
    };
    Some(Mk4V2Route {
        source: compose(&format!(
            "{}\n{}",
            wk::gemv_nvfp4_v2::helpers(cfg),
            MK4_SLOT_V2_WGSL
        )),
        entry,
        label: format!("g4w-mk4-v2-{entry}-{}x{}", cfg.wg, cfg.mr),
        rows_per_group: cfg.rows_per_group(kernel),
        vec4: kernel.vec4_slots(),
    })
}

enum GemvBindDst<'a> {
    Packed(&'a wgpu::Buffer),
    Split {
        q: &'a wgpu::Buffer,
        k: &'a wgpu::Buffer,
        v: &'a wgpu::Buffer,
        sp: &'a wgpu::Buffer,
    },
}

fn gemv_bind_list<'a>(
    w: &'a wgpu::Buffer,
    row_scale: Option<&'a wgpu::Buffer>,
    x: &'a wgpu::Buffer,
    params: &'a wgpu::Buffer,
    dst: GemvBindDst<'a>,
    tail: Option<(u32, &'a wgpu::Buffer)>,
) -> Vec<(u32, &'a wgpu::Buffer)> {
    let mut binds = vec![(0u32, w)];
    let mut slot = 1u32;
    if let Some(rs) = row_scale {
        binds.push((slot, rs));
        slot += 1;
    }
    binds.push((slot, x));
    match dst {
        GemvBindDst::Packed(y) => {
            binds.push((slot + 1, y));
            binds.push((slot + 2, params));
        }
        GemvBindDst::Split { q, k, v, sp } => {
            binds.push((slot + 2, params));
            binds.extend([(31, q), (32, k), (33, v), (34, sp)]);
        }
    }
    binds.extend(tail);
    binds
}

type LmHeadI8Bufs = (
    GpuTensor<u32>,
    GpuTensor<u32>,
    GpuTensor<f32>,
    GpuTensor<f32>,
    GpuTensor<u32>,
    GpuTensor<f32>,
);

struct MrowHeadBufs<'a> {
    hidden_in: &'a wgpu::Buffer,
    normed: &'a wgpu::Buffer,
    logits_pk: &'a wgpu::Buffer,
    logits_f32: &'a wgpu::Buffer,
    am_val: &'a wgpu::Buffer,
    am_idx: &'a wgpu::Buffer,
    token_out: &'a wgpu::Buffer,
}

struct MrowHeadWeights<'a> {
    final_norm: &'a GpuTensor<u32>,
    embed_lo: &'a GpuTensor<u32>,
    embed_hi: &'a GpuTensor<u32>,
    i8: &'a Option<LmHeadI8Bufs>,
}

#[allow(clippy::too_many_arguments)]
fn push_mrow_head(
    b: &mut Builder,
    ctx: &'static WgpuContext,
    pl: &Pipelines,
    config: &Gemma4Config,
    rows: usize,
    bufs: &MrowHeadBufs,
    w: &MrowHeadWeights,
) -> Result<()> {
    let hidden = config.hidden_size;
    let vocab = config.vocab_size;
    let split_row = vocab / 2;
    let eps = config.rms_norm_eps as f32;
    b.rms(
        bufs.hidden_in,
        w.final_norm.raw(),
        bufs.normed,
        rows,
        hidden,
        eps,
    );
    let lm_grid = b.grid_1d(split_row as u64, wk::gemv_bf16::ROWS_PER_GROUP);
    if let Some((wq_lo, wq_hi, rs_lo, rs_hi, wn_ones, _rstd_one)) = w.i8 {
        let lm_i8_p = b.uni(
            "g4w-mrow-lm-i8-p",
            GemvI8Params {
                n_rows: split_row as u32,
                k_elems: hidden as u32,
                wq_row_words: (hidden / 4) as u32,
                groups_x: lm_grid.0,
                m_rows: rows as u32,
                x_row_words: (hidden / 2) as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        let rstd_ones_noop_because_final_norm_already_ran =
            GpuTensor::<f32>::upload(ctx, "g4w-mrow-lm-rstd1", &vec![1.0f32; rows]);
        let lm_i8_mk = mk_pipeline(
            ctx,
            "g4w-lmhead-i8-mk-pk",
            &mk_i8_lmhead_shader_source(rows),
            LMHEAD_I8_MK_PK_ENTRY,
        )?;
        for (wq, rs, word_off) in [(wq_lo, rs_lo, 0usize), (wq_hi, rs_hi, split_row / 2)] {
            let mkp = b.uni(
                "g4w-mrow-lm-i8-mk-p",
                MkParams {
                    m: rows as u32,
                    x_stride_words: (hidden / 2) as u32,
                    y_stride_words: (vocab / 2) as u32,
                    dst_word_off: word_off as u32,
                },
            );
            b.push(
                lm_i8_mk.clone(),
                &[
                    (13, wq.raw()),
                    (14, rs.raw()),
                    (15, bufs.normed),
                    (16, wn_ones.raw()),
                    (17, rstd_ones_noop_because_final_norm_already_ran.raw()),
                    (18, bufs.logits_pk),
                    (19, &lm_i8_p),
                    (35, &mkp),
                ],
                lm_grid,
            );
        }
        b.keep
            .push(Box::new(rstd_ones_noop_because_final_norm_already_ran));
    } else {
        let lm_p = b.uni(
            "g4w-mrow-lm-p",
            GemvBf16Params {
                n_rows: split_row as u32,
                k_elems: hidden as u32,
                w_row_words: (hidden / 2) as u32,
                groups_x: lm_grid.0,
            },
        );
        let mk_bf16 = pl.mk.as_ref().expect("mk pipelines").bf16_pk.clone();
        for (half_w, word_off) in [(w.embed_lo, 0usize), (w.embed_hi, split_row / 2)] {
            let mkp = b.uni(
                "g4w-mrow-lm-mk-p",
                MkParams {
                    m: rows as u32,
                    x_stride_words: (hidden / 2) as u32,
                    y_stride_words: (vocab / 2) as u32,
                    dst_word_off: word_off as u32,
                },
            );
            let binds = gemv_bind_list(
                half_w.raw(),
                None,
                bufs.normed,
                &lm_p,
                GemvBindDst::Packed(bufs.logits_pk),
                Some((35, &mkp)),
            );
            b.push(mk_bf16.clone(), &binds, lm_grid);
        }
    }
    let softcap_on =
        config.final_logit_softcapping > 0.0 && config.final_logit_softcapping.is_finite();
    let cap_p = b.uni(
        "g4w-mrow-cap-p",
        ScaleParams {
            n: (rows * vocab) as u32,
            n_words: (rows * vocab / 2) as u32,
            scale: 0.0,
            cap: config.final_logit_softcapping,
            inv_cap: if softcap_on {
                1.0 / config.final_logit_softcapping
            } else {
                0.0
            },
            pad0: 0,
            pad1: 0,
            pad2: 0,
        },
    );
    let capgrid = b.grid_1d((rows * vocab / 2) as u64, 256);
    let cap_pl = if softcap_on {
        pl.softcap.clone()
    } else {
        pl.cast_f32.clone()
    };
    b.push(
        cap_pl,
        &[(0, bufs.logits_pk), (3, &cap_p), (4, bufs.logits_f32)],
        capgrid,
    );
    let am_p = b.uni(
        "g4w-mrow-am-p",
        ArgmaxRowsParams {
            rows: rows as u32,
            n: vocab as u32,
            pad0: 0,
            pad1: 0,
        },
    );
    let am1_pl = pl.am1.clone();
    b.push(
        am1_pl,
        &[
            (54, bufs.logits_f32),
            (55, bufs.am_val),
            (56, bufs.am_idx),
            (58, &am_p),
        ],
        (wk::graph_decode::ARGMAX_BLOCKS as u32, rows as u32, 1),
    );
    let am2_grid = b.grid_1d(rows as u64, 1);
    let am2_pl = pl.am2.clone();
    b.push(
        am2_pl,
        &[
            (55, bufs.am_val),
            (56, bufs.am_idx),
            (57, bufs.token_out),
            (58, &am_p),
        ],
        am2_grid,
    );
    Ok(())
}

fn q8_pick(
    fmt: wk::quant_gemv::QFormat,
    e4m3: &Arc<wgpu::ComputePipeline>,
    int8: &Arc<wgpu::ComputePipeline>,
) -> Arc<wgpu::ComputePipeline> {
    match fmt {
        wk::quant_gemv::QFormat::E4m3 => e4m3.clone(),
        wk::quant_gemv::QFormat::Int8 => int8.clone(),
    }
}

impl MkQ8Pipelines {
    fn pk_for(&self, fmt: wk::quant_gemv::QFormat) -> Arc<wgpu::ComputePipeline> {
        q8_pick(fmt, &self.fp8_pk, &self.i8_pk)
    }
    fn pk3_for(&self, fmt: wk::quant_gemv::QFormat) -> Arc<wgpu::ComputePipeline> {
        q8_pick(fmt, &self.fp8_pk3, &self.i8_pk3)
    }
}

pub fn batch_slots_default() -> usize {
    match std::env::var("NV_WGPU_BATCH_SLOTS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(s) if s >= 2 => s.min(MK_MAX),
        _ => 0,
    }
}

pub fn prefill_m() -> usize {
    match std::env::var("NV_G4_WGPU_PREFILL_M")
        .or_else(|_| std::env::var("NV_WGPU_PREFILL_M"))
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(0) | Some(1) => 0,
        Some(m) => m.clamp(2, MK_MAX),
        None => MK_MAX,
    }
}

const PF_FLASH2_PK_MK_WGSL: &str = include_str!("../../nv-kernels/wgsl/g4w_pf_flash2_pk_mk.wgsl");

pub const PF_FLASH_TILE_ROWS_BAKED_AS_FDT_ROWS_32_IN_THE_SHARED_TILED_KERNEL: usize = 32;

pub const PF_FLASH_MAX_HEAD_DIM_FDT_KV_STAGE_HOLDS_8_POSITIONS_OF_256_FLOATS: usize = 256;

pub fn pf_flash_full_attention_enabled_via_nv_g4_wgpu_pf_flash_default_off_pending_nll_adjudication(
) -> bool {
    matches!(
        std::env::var("NV_G4_WGPU_PF_FLASH").ok().as_deref(),
        Some("1") | Some("on")
    )
}

pub fn pf_flash_portable_arm_forced_via_nv_g4_wgpu_pf_flash_portable_for_testing_on_sg_adapters(
) -> bool {
    matches!(
        std::env::var("NV_G4_WGPU_PF_FLASH_PORTABLE").ok().as_deref(),
        Some("1") | Some("on")
    )
}

pub fn pf_flash_pipeline_specs_stage1_tiled_slotml_arm_matching_the_qwen_nll_signoff_then_stage2_pk_mk(
    sg: bool,
) -> [(String, &'static str, &'static str); 2] {
    let src1 = crate::qwen3_5_moe_wgpu::pf_flash_tiled_source(sg);
    let (label1, entry1) = if sg {
        (
            "g4w-pf-flash1-tiled-sg",
            "q3w_pf_flash1_fp8kv_tiled_slotml_sg",
        )
    } else {
        (
            "g4w-pf-flash1-tiled-wg",
            "q3w_pf_flash1_fp8kv_tiled_slotml_wg",
        )
    };
    let src2 = format!("{}\n{}", compose(wk::flash_decode::WGSL), PF_FLASH2_PK_MK_WGSL);
    [
        (src1, label1, entry1),
        (src2, "g4w-pf-flash2-pk-mk", "g4w_flash_splitk_stage2_pk_mk"),
    ]
}

pub const PF_WIDE_M_MAX_VIA_MK_MAX_ROW_GEMM_TILES_AT_256B_ALIGNED_OFFSETS: usize = 4 * MK_MAX;

pub const PF_GEMM_TILE_ROWS_BOUND_BY_ONE_F32_ACCUMULATOR_PER_SLOT_IN_REGISTERS: usize = MK_MAX;

pub fn prefill_m_wide_via_nv_g4_wgpu_pf_m_in_mk_max_row_tiles() -> usize {
    match std::env::var("NV_G4_WGPU_PF_M")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(m) if m > MK_MAX => {
            m.min(PF_WIDE_M_MAX_VIA_MK_MAX_ROW_GEMM_TILES_AT_256B_ALIGNED_OFFSETS) / MK_MAX
                * MK_MAX
        }
        Some(0) | Some(1) => 0,
        Some(m) => m.clamp(2, MK_MAX),
        None => prefill_m(),
    }
}

pub const PF_COOP_TN2_SG2_KU4_STAGES_4096_F16_WELL_UNDER_THE_24576_BUDGET: (u32, u32, u32) =
    (2, 2, 4);

pub fn pf_coop_w4a16_ffn_opt_in_default_off_because_only_measured_on_this_blackwell_adapter(
) -> bool {
    std::env::var("NV_G4_PF_COOP").ok().as_deref() == Some("1")
}

pub const PF_FIXED_PASSES_PER_DENSE_LAYER_RMS8_ROPE2_KVQ2_GELU1: usize = 13;

pub const PF_EMBED_PASSES_GATHER_SCALE_PLUS_MM_SPLICE: usize = 3;

pub const PF_PROJECTIONS_PER_LAYER_QKV_O_GATEUP_DOWN: usize = 4;

pub fn pf_gemm_dispatches_per_projection_one_tile_per_mk_max_rows(m: usize) -> usize {
    m.div_ceil(PF_GEMM_TILE_ROWS_BOUND_BY_ONE_F32_ACCUMULATOR_PER_SLOT_IN_REGISTERS)
}

pub fn pf_passes_per_chunk_when_no_projection_is_nvfp4(layers: usize, m: usize) -> usize {
    layers
        * (PF_FIXED_PASSES_PER_DENSE_LAYER_RMS8_ROPE2_KVQ2_GELU1
            + PF_PROJECTIONS_PER_LAYER_QKV_O_GATEUP_DOWN
                * pf_gemm_dispatches_per_projection_one_tile_per_mk_max_rows(m)
            + 2 * m)
        + PF_EMBED_PASSES_GATHER_SCALE_PLUS_MM_SPLICE
}

pub const SLIDING_KV_RING_DEFAULT_ON: bool = false;

pub fn sliding_kv_ring_enabled() -> bool {
    match std::env::var("NV_G4_WGPU_KV_RING") {
        Ok(v) => {
            let t = v.trim();
            !t.is_empty() && t != "0"
        }
        Err(_) => SLIDING_KV_RING_DEFAULT_ON,
    }
}

pub const SLIDING_KV_RING_HEADROOM_SLOTS_MATCHING_CUDA_KV_FP8_RING_SLOTS: usize = 128;

pub fn sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom(
    window: usize,
    prefill_chunk: usize,
) -> usize {
    window + prefill_chunk.max(1) + SLIDING_KV_RING_HEADROOM_SLOTS_MATCHING_CUDA_KV_FP8_RING_SLOTS
}

fn mk_pipeline(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
) -> Result<Arc<wgpu::ComputePipeline>> {
    dispatch::cached_compute_pipeline(ctx, label, source, entry)
        .map_err(|e| anyhow::anyhow!("pipeline {label}: {e}"))
}

fn build_mk_pipelines(ctx: &WgpuContext, m: usize, fp8: bool) -> Result<MkPipelines> {
    let mk = |label: &str, source: &str, entry: &str| mk_pipeline(ctx, label, source, entry);
    let src_b = mk_bf16_shader_source(m);
    let q8 = if fp8 {
        let src_q = mk_q8_shader_source(m);
        Some(MkQ8Pipelines {
            i8_pk: mk("g4w-gemm-int8-mk-pk", &src_q, "g4w_gemm_int8_mk_pk")?,
            i8_pk3: mk("g4w-gemm-int8-mk-pk3", &src_q, "g4w_gemm_int8_mk_pk3")?,
            fp8_pk: mk("g4w-gemm-fp8-mk-pk", &src_q, "g4w_gemm_fp8_mk_pk")?,
            fp8_pk3: mk("g4w-gemm-fp8-mk-pk3", &src_q, "g4w_gemm_fp8_mk_pk3")?,
        })
    } else {
        None
    };
    Ok(MkPipelines {
        gather: mk("g4w-gather-mk", glue_shader_source(), "gather2_bf16_mk")?,
        bf16_pk: mk("g4w-gemm-bf16-mk-pk", &src_b, "g4w_gemm_bf16_mk_pk")?,
        bf16_pk3: mk("g4w-gemm-bf16-mk-pk3", &src_b, "g4w_gemm_bf16_mk_pk3")?,
        q8,
    })
}

impl Fp8Pipelines {
    fn pk_for(&self, fmt: wk::quant_gemv::QFormat) -> Arc<wgpu::ComputePipeline> {
        q8_pick(fmt, &self.pk, &self.i8_pk)
    }
    fn pk3_for(&self, fmt: wk::quant_gemv::QFormat) -> Arc<wgpu::ComputePipeline> {
        q8_pick(fmt, &self.pk3, &self.i8_pk3)
    }
}

struct FusePipelines {
    head_prep: Arc<wgpu::ComputePipeline>,
    norm_res_norm: Arc<wgpu::ComputePipeline>,
    norm_add_norm: Arc<wgpu::ComputePipeline>,
}

pub fn fuse_workgroup_bytes(hd_max: usize) -> u32 {
    let maxw = (hd_max / 2).max(1) as u32;
    2 * 256 * 4 + 2 * 4 + 2 * maxw * 4
}

pub const FUSE_HEAD_PREP: u32 = 1;
pub const FUSE_NORM_RES_NORM: u32 = 2;
pub const FUSE_NORM_ADD_NORM: u32 = 4;
pub const FUSE_ALL: u32 = FUSE_HEAD_PREP | FUSE_NORM_RES_NORM | FUSE_NORM_ADD_NORM;

pub fn fuse_mask(ctx: &WgpuContext, hd_max: usize) -> u32 {
    if !ctx
        .caps
        .workgroup_storage_fits(fuse_workgroup_bytes(hd_max))
    {
        return 0;
    }
    match std::env::var("NV_WGPU_FUSE") {
        Ok(v) => v.trim().parse::<u32>().unwrap_or(FUSE_ALL) & FUSE_ALL,
        Err(_) => FUSE_ALL,
    }
}

fn build_fuse_pipelines(ctx: &WgpuContext, hd_max: usize) -> Result<FusePipelines> {
    let maxw = (hd_max / 2).max(1);
    anyhow::ensure!(
        HEAD_PREP_WGSL.matches("HP_MAXW").count() == 2,
        "head-prep shader must declare HP_MAXW exactly twice"
    );
    let hp = compose(&HEAD_PREP_WGSL.replace("HP_MAXW", &format!("{maxw}u")));
    let nc = compose(NORM_CHAIN_WGSL);
    let mk = |label: &str, source: &str, entry: &str| mk_pipeline(ctx, label, source, entry);
    Ok(FusePipelines {
        head_prep: mk("g4w-head-prep", &hp, "g4w_head_prep")?,
        norm_res_norm: mk("g4w-norm-res-norm", &nc, "g4w_norm_res_norm")?,
        norm_add_norm: mk("g4w-norm-add-norm", &nc, "g4w_norm_add_norm")?,
    })
}

pub const NVFP4_SHAPE_PICK_DEFAULT_ON: bool = false;

fn nvfp4_shape_pick_enabled() -> bool {
    match std::env::var("NV_WGPU_NVFP4_SHAPE_PICK") {
        Ok(v) => v != "0",
        Err(_) => NVFP4_SHAPE_PICK_DEFAULT_ON,
    }
}

fn nvfp4_variant(ctx: &WgpuContext) -> wk::gemv_nvfp4::GemvVariant {
    if std::env::var("NV_WGPU_NVFP4_TREE").is_ok_and(|v| v != "0") {
        wk::gemv_nvfp4::GemvVariant::Tree
    } else {
        wk::gemv_nvfp4::select_variant(ctx)
    }
}

pub const ATTN_FP8_DEFAULT_ON: bool = true;

pub const ATTN_FP8_FMT_DEFAULT: wk::quant_gemv::QFormat = wk::quant_gemv::QFormat::Int8;

pub const NVFP4_V2_DEFAULT_ON: bool = false;

pub fn nvfp4_v2_enabled(ctx: &WgpuContext) -> bool {
    if std::env::var("NV_WGPU_NVFP4_TREE").is_ok_and(|v| v != "0") {
        return false;
    }
    if !wk::gemv_nvfp4_v2::subgroup32_ok(ctx) {
        return false;
    }
    match std::env::var("NV_WGPU_NVFP4_V2") {
        Ok(v) => v != "0",
        Err(_) => NVFP4_V2_DEFAULT_ON,
    }
}

fn nvfp4_v2_pipeline(
    ctx: &WgpuContext,
    n: usize,
    k: usize,
) -> Option<(Arc<wgpu::ComputePipeline>, wk::gemv_nvfp4_v2::V2Kernel, u32)> {
    let (kernel, cfg, entry) = wk::gemv_nvfp4_v2::select_pk(n, k)?;
    let src = wk::gemv_nvfp4_v2::source(cfg);
    let label = format!("g4w-gemv4-v2-{entry}-{}x{}", cfg.wg, cfg.mr);
    match dispatch::cached_compute_pipeline(ctx, &label, &src, entry) {
        Ok(p) => Some((p, kernel, cfg.rows_per_group(kernel))),
        Err(e) => {
            eprintln!("[gemma4_wgpu] nvfp4 v2 pipeline {entry} unavailable ({e}); keeping tree/sg");
            None
        }
    }
}

pub fn nvfp4_v2_boot_line(requested: bool, routed: usize, total: usize) -> Option<String> {
    if !requested {
        return None;
    }
    Some(if routed == 0 {
        format!("[gemma4_wgpu] nvfp4 v2 requested but 0 of {total} nvfp4 projections routed to it")
    } else {
        format!("[gemma4_wgpu] nvfp4 v2 engaged on {routed} of {total} nvfp4 projections")
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnVariant {
    pub on: bool,
    pub quant: AttnQuant,
    pub legacy_epilogue: u32,
}

pub const ATTN_VARIANT_DEFAULT: AttnVariant = AttnVariant {
    on: ATTN_FP8_DEFAULT_ON,
    quant: AttnQuant {
        fmt: ATTN_FP8_FMT_DEFAULT,
        group: 128,
        lo: 0,
        hi: usize::MAX,
    },
    legacy_epilogue: 0,
};

static ATTN_VARIANT_OVERRIDE: std::sync::Mutex<Option<AttnVariant>> = std::sync::Mutex::new(None);

pub fn set_attn_variant(v: Option<AttnVariant>) {
    *ATTN_VARIANT_OVERRIDE.lock().unwrap() = v;
}

fn attn_variant() -> AttnVariant {
    ATTN_VARIANT_OVERRIDE
        .lock()
        .unwrap()
        .unwrap_or(ATTN_VARIANT_DEFAULT)
}

pub fn attn_fp8_enabled() -> bool {
    attn_variant().on
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnQuant {
    pub fmt: wk::quant_gemv::QFormat,
    pub group: usize,
    pub lo: usize,
    pub hi: usize,
}

impl Default for AttnQuant {
    fn default() -> Self {
        Self {
            fmt: ATTN_FP8_FMT_DEFAULT,
            group: 0,
            lo: 0,
            hi: usize::MAX,
        }
    }
}

impl AttnQuant {
    pub fn covers(&self, layer: usize) -> bool {
        layer >= self.lo && layer < self.hi
    }
    pub fn label(&self) -> String {
        let g = if self.group == 0 {
            "row".to_string()
        } else {
            self.group.to_string()
        };
        if self.hi == usize::MAX && self.lo == 0 {
            format!("{}/{g}", self.fmt.label())
        } else {
            format!("{}/{g}@[{},{})", self.fmt.label(), self.lo, self.hi)
        }
    }
}

pub fn weight_format_boot_line() -> String {
    let attn = if attn_fp8_enabled() {
        attn_quant_config().label()
    } else {
        "checkpoint".to_string()
    };
    let (w8_gu, w8_dn, g) = w8_ffn_mode();
    let ffn = |on: bool| {
        if on {
            format!("int8/{g} if nvfp4")
        } else {
            "checkpoint".to_string()
        }
    };
    let lmhead = if std::env::var("NV_WGPU_LMHEAD_INT8").is_ok_and(|v| v != "0") {
        "int8"
    } else {
        "checkpoint"
    };
    let w4 = match W4PtqSpec::from_env() {
        Some(s) => format!(
            " w4_ptq={} ({:.4} B/weight if served w4a16)",
            s.label(),
            s.bytes_per_weight()
        ),
        None => String::new(),
    };
    format!(
        "[gemma4_wgpu] weight formats: attn.qkv+o={attn} ffn.gate_up={} ffn.down={} \
         lm_head={lmhead}{w4} (attn variant is hardcoded per arch; override with \
         NV_G4_WGPU_W8_FFN=off / NV_WGPU_LMHEAD_INT8=0)",
        ffn(w8_gu),
        ffn(w8_dn)
    )
}

pub fn attn_quant_config() -> AttnQuant {
    attn_variant().quant
}

fn fp8_sg(ctx: &WgpuContext) -> bool {
    wk::gemv_nvfp4::subgroup_ok(ctx)
}

fn build_fp8_pipelines(ctx: &WgpuContext) -> Result<Fp8Pipelines> {
    let sg = fp8_sg(ctx);

    eprintln!(
        "[gemma4_wgpu] fp8/int8 epilogue: {} ({} rows/group)",
        if sg { "subgroup" } else { "tree" },
        if sg {
            wk::quant_gemv::SG_ROWS_PER_GROUP
        } else {
            wk::quant_gemv::TREE_ROWS_PER_GROUP
        }
    );
    let src = fp8_pk_shader_source(sg);
    let mk = |label: &str, entry: &str| mk_pipeline(ctx, label, &src, entry);
    let legacy = attn_fp8_legacy_epilogue();
    let (e_pk, e_pk3) = if legacy != 0 {
        eprintln!("[gemma4_wgpu] using the LEGACY row-scale fp8 epilogue (mode {legacy})");
        ("g4w_gemv_legacy_pk", "g4w_gemv_legacy_pk3")
    } else {
        ("g4w_gemv_fp8_pk", "g4w_gemv_fp8_pk3")
    };
    Ok(Fp8Pipelines {
        pk: mk("g4w-gemv-fp8-pk", e_pk)?,
        pk3: mk("g4w-gemv-fp8-pk3", e_pk3)?,
        i8_pk: mk("g4w-gemv-int8-pk", "g4w_gemv_int8_pk")?,
        i8_pk3: mk("g4w-gemv-int8-pk3", "g4w_gemv_int8_pk3")?,
        i8_gelu: mk(
            "g4w-gemv-int8-gelu",
            if sg {
                wk::quant_gemv::INT8_GROUP_GELU_SG_ENTRY
            } else {
                wk::quant_gemv::INT8_GROUP_GELU_ENTRY
            },
        )?,
        rows_per_group: if sg {
            wk::quant_gemv::SG_ROWS_PER_GROUP
        } else {
            wk::quant_gemv::TREE_ROWS_PER_GROUP
        },
    })
}

fn build_pipelines(
    ctx: &WgpuContext,
    variant: wk::gemv_nvfp4::GemvVariant,
    fp8: bool,
    fuse_hd_max: Option<usize>,
    mk_rows: usize,
) -> Result<Pipelines> {
    let mk = |label: &str, source: &str, entry: &str| mk_pipeline(ctx, label, source, entry);
    let src_rs = compose(wk::residual_scale::WGSL);
    let src_rms = compose(wk::rmsnorm::WGSL);
    let src_rmsres = compose(wk::rmsnorm_residual::WGSL);
    let src_rope = format!("{}\n{}", compose(wk::rope_bf16::WGSL), ROPE_F32_WGSL);
    let src_kv = compose(wk::kv_fp8::WGSL);
    let src_flash = format!("{}\n{}", compose(wk::flash_decode::WGSL), FLASH2_PK_WGSL);
    let src_flash2_folded = {
        let folded = src_flash.replacen(
            "let splits = 16u;",
            &format!("let splits = {}u;", flash_splits()),
            1,
        );
        anyhow::ensure!(
            flash_splits() == 16 || folded != src_flash,
            "flash2 split-count anchor `let splits = 16u;` missing from g4shared_flash2_pk; \
             the folded stage2 would silently merge the wrong number of partials"
        );
        folded
    };
    let src_gemvb = format!("{}\n{}", compose(wk::gemv_bf16::WGSL), GEMV_PK_WGSL);
    let src_gd = compose(wk::graph_decode::WGSL);
    let src_gemv4_pk = match variant {
        wk::gemv_nvfp4::GemvVariant::Tree => {
            format!("{}\n{}", wk::gemv_nvfp4::gemv_source(), GEMV4_PK_TREE_WGSL)
        }
        _ => {
            format!("{}\n{}", wk::gemv_nvfp4::sg_gemv_source(), GEMV4_PK_SG_WGSL)
        }
    };
    let src_quant4 = format!("{}\n{}", wk::gemv_nvfp4::quantize_source(), QUANT_PK_WGSL);
    let src_gelu = wk::gelu_tanh_mul::source();
    Ok(Pipelines {
        gather: mk("g4w-gather", glue_shader_source(), "gather2_bf16")?,
        scale: mk("g4w-scale", &src_rs, "scale_bf16")?,
        rms: mk("g4w-rms", &src_rms, "rmsnorm_bf16")?,
        rmsres: mk("g4w-rmsres", &src_rmsres, "rmsnorm_residual_bf16")?,
        resadd: mk("g4w-resadd", &src_rs, "residual_add_scale_bf16")?,
        cast_f32: mk("g4w-cast", &src_rs, "cast_bf16_to_f32")?,
        softcap: mk("g4w-softcap", &src_rs, "tanh_softcap_bf16_to_f32")?,
        rope: mk("g4w-rope", &src_rope, "rope_bf16")?,
        rope_f32: mk("g4w-rope-f32", &src_rope, ROPE_F32_ENTRY)?,
        kvq: mk("g4w-kvq", &src_kv, wk::kv_fp8::QUANTIZE_ENTRY)?,
        flash1: mk("g4w-flash1", &src_flash, flash1_stage1_entry())?,
        flash2_pk: mk("g4w-flash2-pk", &src_flash2_folded, FLASH2_PK_ENTRY)?,
        gemv_pk: mk("g4w-gemv8-pk", &src_gemvb, GEMV_PK_ENTRY)?,
        gemv_pk3: mk("g4w-gemv8-pk3", &src_gemvb, GEMV_PK3_ENTRY)?,
        rowquant_i8: mk("g4w-rowquant-i8", &src_gemvb, wk::gemv_bf16::ROWQUANT_ENTRY)?,
        gemv_i8: mk("g4w-gemv-i8", &src_gemvb, wk::gemv_bf16::I8_NORMED_ENTRY)?,
        gemv4_pk: mk("g4w-gemv4-pk", &src_gemv4_pk, "g4w_gemv_nvfp4_pk")?,
        gemv4_pk_deep: match variant {
            wk::gemv_nvfp4::GemvVariant::Tree => None,
            _ => Some(mk(
                "g4w-gemv4-pk-deep",
                &format!("{}\n{}", wk::gemv_nvfp4::gemv_source(), GEMV4_PK_TREE_WGSL),
                "g4w_gemv_nvfp4_pk",
            )?),
        },
        quant_pk: mk("g4w-quant4-pk", &src_quant4, "g4w_quant_row_pk")?,
        gelu_even: mk("g4w-gelu", &src_gelu, wk::gelu_tanh_mul::ENTRY_FUSED_EVEN)?,
        am1: mk("g4w-am1", &src_gd, "argmax_f32_rows_stage1")?,
        am2: mk("g4w-am2", &src_gd, "argmax_f32_rows_stage2")?,
        pack16: mk("g4w-pack16", GLUE_WGSL, "pack_lo16")?,
        fp8: if fp8 {
            Some(build_fp8_pipelines(ctx)?)
        } else {
            None
        },
        fuse: match fuse_hd_max {
            Some(hd) => Some(build_fuse_pipelines(ctx, hd)?),
            None => None,
        },
        mk: if mk_rows > 0 {
            Some(build_mk_pipelines(ctx, mk_rows, fp8)?)
        } else {
            None
        },
        splice: mk(
            "g4w-splice-embed-rows",
            EMBED_ROW_SPLICE_WGSL,
            EMBED_ROW_SPLICE_ENTRY,
        )?,
    })
}

struct LayerGpu {
    kind: LayerType,
    has_v: bool,
    layer_scalar: f32,
    qkv: GpuProj,
    o: GpuProj,
    gate_up: GpuProj,
    down: GpuProj,
    gate_up_coop: Option<CoopLin>,
    down_coop: Option<CoopLin>,
    ln_pa: GpuTensor<u32>,
    ln_pf: GpuTensor<u32>,
    ln_po: GpuTensor<u32>,
    qn: GpuTensor<u32>,
    kn: GpuTensor<u32>,
    vn: GpuTensor<u32>,
}

struct BtBufs {
    core: PfBufs,
    logits_pk: GpuTensor<u32>,
    am_val: GpuTensor<f32>,
    am_idx: GpuTensor<i32>,
}

struct PfBufs {
    splice_rows: GpuTensor<u32>,
    splice_mask: GpuTensor<u32>,
    hid_a: GpuTensor<u32>,
    hid_b: GpuTensor<u32>,
    t0: GpuTensor<u32>,
    t1: GpuTensor<u32>,
    qa: GpuTensor<u32>,
    qb: GpuTensor<u32>,
    ka: GpuTensor<u32>,
    kb: GpuTensor<u32>,
    va: GpuTensor<u32>,
    vb: GpuTensor<u32>,
    q_f32: GpuTensor<f32>,
    attn_pack: GpuTensor<u32>,
    gu_pack: GpuTensor<u32>,
    act_pack: GpuTensor<u32>,
}

fn prefill_is_live(pf_m: usize, pf_passes: &[Pass]) -> bool {
    pf_m > 0 && !pf_passes.is_empty()
}

struct VerifyState {
    rows: usize,
    passes: Vec<Pass>,
    logits_f32: GpuTensor<f32>,
    token_out: GpuTensor<u32>,
    validated: bool,
}

struct PrefillState {
    m: usize,
    passes: Vec<Pass>,
    tok_idx: GpuTensor<i32>,
    rope_pos: GpuTensor<i32>,
    kv_start: GpuTensor<i32>,
    fd_s: Vec<GpuUniform<FdParams>>,
    fd_f: Vec<GpuUniform<FdParams>>,
    fd_flash: Option<GpuUniform<FdParams>>,
    splice_rows: wgpu::Buffer,
    splice_mask: wgpu::Buffer,
    splice_mask_live: bool,
    validated: bool,
}

struct BatchState {
    slots: usize,
    passes: Vec<Pass>,
    tok_idx: GpuTensor<i32>,
    rope_pos: GpuTensor<i32>,
    kv_start: Vec<GpuTensor<i32>>,
    fd_s: Vec<GpuUniform<FdParams>>,
    fd_f: Vec<GpuUniform<FdParams>>,
    token_out: GpuTensor<u32>,
    logits_f32: GpuTensor<f32>,
    validated: bool,
}

pub struct Gemma4Wgpu {
    ctx: &'static WgpuContext,
    config: Gemma4Config,
    max_seq: usize,
    pos: usize,
    kv_base: usize,
    slot_pos: Vec<usize>,
    validated: bool,
    prefill: Option<PrefillState>,
    pf_coop_sites: usize,
    verify: Option<VerifyState>,
    batch: Option<BatchState>,
    weight_bytes: u64,
    nvfp4_v2: (usize, usize),
    passes: Vec<Pass>,
    head_start: usize,
    prefix_validated: bool,
    tok_idx: GpuTensor<i32>,
    rope_pos: GpuTensor<i32>,
    kv_start: GpuTensor<i32>,
    fd_sliding: GpuUniform<FdParams>,
    fd_full: GpuUniform<FdParams>,
    fd_sliding_base: FdParams,
    fd_full_base: FdParams,
    token_out: GpuTensor<u32>,
    chain_out: GpuTensor<u32>,
    logits_f32: GpuTensor<f32>,
    vocab: usize,
    preenc: bool,
    pending_cb: Option<wgpu::CommandBuffer>,
    preenc_hits: u64,
    chain_steps: u64,
    uniform_probe: usize,
    w4_cap: Option<W4Capture>,
    kv_layers: Vec<GpuLayerKv>,
    _keep: Vec<Box<dyn std::any::Any>>,
}

struct W4Capture {
    buf: GpuTensor<u32>,
    layers: usize,
    gu_words: usize,
    dn_words: usize,
}

enum GemvDst<'b> {
    Packed {
        y: &'b wgpu::Buffer,
        word_off: usize,
    },
    SplitQkv {
        q: &'b wgpu::Buffer,
        k: &'b wgpu::Buffer,
        v: &'b wgpu::Buffer,
        q_rows: usize,
        kv_rows: usize,
        v_off: usize,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PassDst {
    Decode,
    Prefill,
    Batch,
    Verify,
}

struct Builder<'a> {
    ctx: &'static WgpuContext,
    pl: &'a Pipelines,
    passes: Vec<Pass>,
    pf_passes: Vec<Pass>,
    bt_passes: Vec<Pass>,
    vf_passes: Vec<Pass>,
    dst: PassDst,
    keep: Vec<Box<dyn std::any::Any>>,
    weight_bytes: u64,
    nvfp4_projs: usize,
    nvfp4_v2_routed: usize,
    mk4: Option<Mk4SlotBufs>,
}

enum GemvDstMk<'b> {
    Packed {
        y: &'b wgpu::Buffer,
        word_off: usize,
        y_stride_words: usize,
    },
    SplitQkv {
        q: &'b wgpu::Buffer,
        k: &'b wgpu::Buffer,
        v: &'b wgpu::Buffer,
        q_rows: usize,
        kv_rows: usize,
        v_off: usize,
    },
}

impl<'a> Builder<'a> {
    fn push_offsets(
        &mut self,
        pipeline: Arc<wgpu::ComputePipeline>,
        binds: &[(u32, &wgpu::Buffer, u64)],
        grid: (u32, u32, u32),
    ) {
        let bind = dispatch::bind_group_offsets(self.ctx, &pipeline, binds);
        let label = if dispatch::profile::enabled() {
            format!(
                "{}[{}x{}x{}]",
                dispatch::profile::pipeline_name(&pipeline)
                    .unwrap_or_else(|| "unnamed".to_string()),
                grid.0,
                grid.1,
                grid.2
            )
        } else {
            String::new()
        };
        let p = Pass {
            pipeline,
            bind,
            grid,
            label,
        };
        match self.dst {
            PassDst::Decode => self.passes.push(p),
            PassDst::Prefill => self.pf_passes.push(p),
            PassDst::Batch => self.bt_passes.push(p),
            PassDst::Verify => self.vf_passes.push(p),
        }
    }

    fn push(
        &mut self,
        pipeline: Arc<wgpu::ComputePipeline>,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) {
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        let label = if dispatch::profile::enabled() {
            format!(
                "{}[{}x{}x{}]",
                dispatch::profile::pipeline_name(&pipeline)
                    .unwrap_or_else(|| "unnamed".to_string()),
                grid.0,
                grid.1,
                grid.2
            )
        } else {
            String::new()
        };
        let p = Pass {
            pipeline,
            bind,
            grid,
            label,
        };
        match self.dst {
            PassDst::Decode => self.passes.push(p),
            PassDst::Prefill => self.pf_passes.push(p),
            PassDst::Batch => self.bt_passes.push(p),
            PassDst::Verify => self.vf_passes.push(p),
        }
    }

    fn uni<T: bytemuck::Pod + 'static>(&mut self, label: &str, value: T) -> wgpu::Buffer {
        let u = GpuUniform::new(self.ctx, label, &value);
        let raw = u.raw().clone();
        self.keep.push(Box::new(u));
        raw
    }

    fn grid_1d(&self, invocations: u64, wg: u32) -> (u32, u32, u32) {
        dispatch::workgroup_count_1d(self.ctx, invocations, wg)
    }

    fn rms(
        &mut self,
        x: &wgpu::Buffer,
        w: &wgpu::Buffer,
        y: &wgpu::Buffer,
        batch: usize,
        hidden: usize,
        eps: f32,
    ) {
        let p = self.uni(
            "g4w-rms-p",
            RmsParams {
                hidden: hidden as u32,
                batch: batch as u32,
                eps,
                words_per_row: (hidden / 2) as u32,
            },
        );
        let grid = self.grid_1d(batch as u64, 1);
        let pipeline = self.pl.rms.clone();
        self.push(pipeline, &[(0, x), (1, w), (2, y), (3, &p)], grid);
    }

    fn pack16(
        &mut self,
        src: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        src_off: usize,
        dst_off_words: usize,
        n_elems: usize,
    ) {
        let n_words = n_elems / 2;
        let p = self.uni(
            "g4w-pack-p",
            PackParams {
                src_off: src_off as u32,
                dst_off: dst_off_words as u32,
                n_words: n_words as u32,
                pad0: 0,
            },
        );
        let grid = self.grid_1d(n_words as u64, 256);
        let pipeline = self.pl.pack16.clone();
        self.push(pipeline, &[(0, src), (1, dst), (2, &p)], grid);
    }

    #[allow(clippy::too_many_arguments)]
    fn pf_coop_w4a16(
        &mut self,
        cc: &PfCoopCtx,
        lin: &CoopLin,
        x_packed: &wgpu::Buffer,
        m: usize,
        x_stride_words: usize,
        y: &wgpu::Buffer,
        word_off: usize,
        y_stride_words: usize,
    ) {
        let (n, k) = (lin.n as u32, lin.k as u32);
        assert!(
            m * lin.k / 2 <= cc.xf16.len() && m * lin.n <= cc.yf32.len(),
            "coop scratch sized for the largest FFN shape must cover m={m} n={n} k={k}"
        );
        let xp = self.uni(
            "g4w-pf-coop-x16-p",
            PfCoopX16Params {
                row_words: k / 2,
                rows: m as u32,
                x_stride_words: x_stride_words as u32,
                dst_stride_elems: k,
            },
        );
        let x16_pl = cc.x16.clone();
        self.push(
            x16_pl,
            &[(0, x_packed), (1, cc.xf16.raw()), (2, &xp)],
            ((k / 2).div_ceil(256), m as u32, 1),
        );
        let cols = 16 * cc.tn * cc.sg;
        let bm = (m as u32).div_ceil(16 * cc.tm);
        let bn = n.div_ceil(cols);
        let grid = self.grid_1d(bm as u64 * bn as u64, 1);
        let gp = self.uni(
            "g4w-pf-coop-p",
            wk::gemm_coop_f16::CoopGemmParams {
                n_rows: n,
                k_elems: k,
                m_rows: m as u32,
                blocks_n: bn,
                y_stride: n,
                groups_x: grid.0,
                pad0: 0,
                pad1: 0,
            },
        );
        let gemm_pl = cc.gemm.clone();
        self.push(
            gemm_pl,
            &[
                (0, lin.w.raw()),
                (1, cc.xf16.raw()),
                (2, cc.yf32.raw()),
                (3, &gp),
                (4, cc.zero.raw()),
                (5, lin.sf.raw()),
            ],
            grid,
        );
        let pp = self.uni(
            "g4w-pf-coop-pack-p",
            PfCoopPackParams {
                pairs_per_row: n / 2,
                rows: m as u32,
                y_stride_words: y_stride_words as u32,
                dst_word_off: word_off as u32,
                src_stride_elems: n,
                alpha: lin.alpha,
                pad0: 0,
                pad1: 0,
            },
        );
        let pack_pl = cc.pack.clone();
        self.push(
            pack_pl,
            &[(0, cc.yf32.raw()), (1, y), (2, &pp)],
            ((n / 2).div_ceil(256), m as u32, 1),
        );
    }

    fn gemv_mk(
        &mut self,
        proj: &GpuProj,
        x_packed: &wgpu::Buffer,
        m: usize,
        x_stride_words: usize,
        dst: GemvDstMk,
    ) {
        if m > MK_MAX && !matches!(proj, GpuProj::Nvfp4 { .. }) {
            assert!(
                m.is_multiple_of(MK_MAX),
                "wide chunks dispatch the M-row GEMMs in {MK_MAX}-row tiles, so m={m} must be a multiple of {MK_MAX}"
            );
            for tile in 0..m / MK_MAX {
                self.gemv_mk_tile(proj, x_packed, MK_MAX, x_stride_words, &dst, tile * MK_MAX);
            }
            return;
        }
        self.gemv_mk_tile(proj, x_packed, m, x_stride_words, &dst, 0);
    }

    fn gemv_mk_tile(
        &mut self,
        proj: &GpuProj,
        x_packed: &wgpu::Buffer,
        m: usize,
        x_stride_words: usize,
        dst: &GemvDstMk,
        row_base: usize,
    ) {
        let (y_stride_words, dst_word_off) = match dst {
            GemvDstMk::Packed {
                word_off,
                y_stride_words,
                ..
            } => (*y_stride_words, *word_off + row_base * *y_stride_words),
            GemvDstMk::SplitQkv { .. } => (0, 0),
        };
        let mkp = self.uni(
            "g4w-mk-p",
            MkParams {
                m: m as u32,
                x_stride_words: x_stride_words as u32,
                y_stride_words: y_stride_words as u32,
                dst_word_off: dst_word_off as u32,
            },
        );
        let sp = match &dst {
            GemvDstMk::SplitQkv {
                q_rows,
                kv_rows,
                v_off,
                ..
            } => Some(self.uni(
                "g4w-mk-split-p",
                SplitParams {
                    q_rows: *q_rows as u32,
                    kv_rows: *kv_rows as u32,
                    v_off: *v_off as u32,
                    pad0: 0,
                },
            )),
            GemvDstMk::Packed { .. } => None,
        };
        let mkpl = self.pl.mk.as_ref().expect("gemv_mk without mk pipelines");
        let (pipeline, w, row_scale, params, grid) = match proj {
            GpuProj::Bf16 { w, params, grid } => {
                let pipeline = match &dst {
                    GemvDstMk::Packed { .. } => mkpl.bf16_pk.clone(),
                    GemvDstMk::SplitQkv { .. } => mkpl.bf16_pk3.clone(),
                };
                (pipeline, w.raw(), None, params.raw(), *grid)
            }
            GpuProj::Fp8 {
                w,
                row_scale,
                params,
                grid,
                fmt,
            } => {
                assert_eq!(
                    x_stride_words % 4,
                    0,
                    "q8 M-row GEMV indexes x as vec4; row stride {x_stride_words} words is not a multiple of 4"
                );
                let q8 = mkpl.q8.as_ref().expect("mk q8 pipelines missing");
                let pipeline = match &dst {
                    GemvDstMk::Packed { .. } => q8.pk_for(*fmt),
                    GemvDstMk::SplitQkv { .. } => q8.pk3_for(*fmt),
                };
                (
                    pipeline,
                    w.raw(),
                    Some(row_scale.raw()),
                    params.raw(),
                    *grid,
                )
            }
            GpuProj::Nvfp4 {
                w,
                scales,
                k,
                n,
                alpha,
                mk_input_globals_one_per_slot,
                ..
            } => {
                let (y, word_off, y_stride) = match &dst {
                    GemvDstMk::Packed {
                        y,
                        word_off,
                        y_stride_words,
                    } => (*y, *word_off, *y_stride_words),
                    GemvDstMk::SplitQkv { .. } => {
                        unreachable!(
                            "nvfp4 qkv is rejected at construction, so the M-row split-qkv dst never carries nvfp4"
                        )
                    }
                };
                assert_eq!(
                    word_off, 0,
                    "the slot-strided nvfp4 M-row gemv writes each slot at word offset 0 only"
                );
                assert_eq!(
                    y_stride,
                    n / 2,
                    "the slot-strided nvfp4 M-row gemv writes one packed row of n/2 words per slot, so the caller's y stride must equal n/2"
                );
                let k_blocks = k / 16;
                assert!(
                    k_blocks.is_multiple_of(4) && n.is_multiple_of(2),
                    "nvfp4 shape n={n} k={k} misses the slot-strided M-row layout; the boot gate must have rejected this graph"
                );
                assert!(
                    x_stride_words.is_multiple_of(8),
                    "the slot quant reads x as packed bf16 pairs over k % 16 == 0 elements, so the slot stride {x_stride_words} words must cover whole blocks"
                );
                let (xq, xs, xm, sel, alphas) = {
                    let m4 = self
                        .mk4
                        .as_ref()
                        .expect("nvfp4 M-row gemv without mk4 slot buffers");
                    assert!(
                        m * (k / 8) <= m4.xq.len()
                            && m * (k_blocks / 4) <= m4.xs.len()
                            && m * (k / 4) <= m4.xm_i8mapped_for_the_slotshared_gemv.len()
                            && m <= m4.sel_all_zero_because_gemma4_has_no_experts.len(),
                        "mk4 slot buffers sized for quant_k_max cannot hold m={m} k={k}"
                    );
                    (
                        m4.xq.raw().clone(),
                        m4.xs.raw().clone(),
                        m4.xm_i8mapped_for_the_slotshared_gemv.raw().clone(),
                        m4.sel_all_zero_because_gemma4_has_no_experts.raw().clone(),
                        m4.alphas_unread_when_per_expert_alpha_is_zero.raw().clone(),
                    )
                };
                let qp = self.uni(
                    "g4w-mk4-quant-p",
                    q3m::QuantRowsParams {
                        k_blocks: k_blocks as u32,
                        n_slots: m as u32,
                        use_sel: 0,
                        x_slot_stride_elems: (x_stride_words * 2) as u32,
                    },
                );
                let quant = mk_pipeline(
                    self.ctx,
                    "g4w-mk4-quant-rows",
                    &q3m::nvfp4_quant_source(),
                    "q3w_quant_rows",
                )
                .expect("mk4 slot quant pipeline");
                self.push(
                    quant,
                    &[
                        (10, x_packed),
                        (11, &qp),
                        (12, &xq),
                        (13, &xs),
                        (14, &sel),
                        (15, mk_input_globals_one_per_slot.raw()),
                    ],
                    ((k_blocks as u32).div_ceil(256).max(1), m as u32, 1),
                );
                let route = mk4_slot_v2_route(self.ctx, *n, *k, m);
                let grid = match &route {
                    Some(r) => self.grid_1d(*n as u64, r.rows_per_group),
                    None => self.grid_1d(n.div_ceil(2) as u64, 1),
                };
                let one_weight_sweep = route
                    .as_ref()
                    .is_some_and(|r| r.entry == q3m::SLOTSHARED_ENTRY);
                let gp = self.uni(
                    "g4w-mk4-gemv-p",
                    q3m::GemvNvfp4Params {
                        alpha: *alpha,
                        n_rows: *n as u32,
                        k_blocks: k_blocks as u32,
                        k_tiles: wk::gemv_nvfp4::k_tiles(k_blocks) as u32,
                        groups_x: grid.0,
                        w_e_stride_vec2: 0,
                        sf_e_stride_bytes: 0,
                        x_slot_stride_vec2: k_blocks as u32,
                        xsf_slot_stride_bytes: k_blocks as u32,
                        y_slot_stride_words: (n / 2) as u32,
                        per_expert_alpha: 0,
                        m_slots_sharing_expert_zero: if one_weight_sweep { m as u32 } else { 0 },
                    },
                );
                if one_weight_sweep {
                    let map = mk_pipeline(
                        self.ctx,
                        "g4w-mk4-i8map-x",
                        &q3m::nvfp4_slotshared_sources(),
                        q3m::SLOTSHARED_MAP_ENTRY,
                    )
                    .expect("mk4 i8map pipeline");
                    self.push(
                        map,
                        &[(12, &xq), (14, &gp), (20, &xm)],
                        ((k_blocks as u32).div_ceil(256).max(1), m as u32, 1),
                    );
                }
                match route {
                    Some(r) => {
                        let gemv = mk_pipeline(self.ctx, &r.label, &r.source, r.entry)
                            .expect("mk4 v2 slot pipeline");
                        let (w_slot, x_slot) = if r.vec4 { (18, 19) } else { (10, 12) };
                        let grid_z = if one_weight_sweep { 1 } else { m as u32 };
                        let mut binds: Vec<(u32, &wgpu::Buffer)> = vec![
                            (w_slot, w.raw()),
                            (11, scales.raw()),
                            (13, &xs),
                            (14, &gp),
                            (15, y),
                            (16, &sel),
                            (17, &alphas),
                        ];
                        if one_weight_sweep {
                            binds.push((20, &xm));
                        } else {
                            binds.push((x_slot, &xq));
                        }
                        self.push(gemv, &binds, (grid.0, grid.1, grid_z));
                    }
                    None => {
                        let gemv = mk_pipeline(
                            self.ctx,
                            "g4w-mk4-gemv4-slot",
                            &q3m::nvfp4_gemv_source(),
                            "q3w_gemv_nvfp4",
                        )
                        .expect("mk4 slot gemv pipeline");
                        self.push(
                            gemv,
                            &[
                                (10, w.raw()),
                                (11, scales.raw()),
                                (12, &xq),
                                (13, &xs),
                                (14, &gp),
                                (15, y),
                                (16, &sel),
                                (17, &alphas),
                            ],
                            (grid.0, grid.1, m as u32),
                        );
                    }
                }
                return;
            }
        };
        let bind_dst = match &dst {
            GemvDstMk::Packed { y, .. } => GemvBindDst::Packed(y),
            GemvDstMk::SplitQkv { q, k, v, .. } => GemvBindDst::Split {
                q,
                k,
                v,
                sp: sp.as_ref().expect("split params"),
            },
        };
        let binds = gemv_bind_list(w, row_scale, x_packed, params, bind_dst, Some((35, &mkp)));
        if row_base == 0 {
            self.push(pipeline, &binds, grid);
            return;
        }
        let x_off_bytes = (row_base * x_stride_words * 4) as u64;
        let (q_off_bytes, kv_off_bytes) = match dst {
            GemvDstMk::SplitQkv {
                q_rows, kv_rows, ..
            } => (
                (row_base * (q_rows / 2) * 4) as u64,
                (row_base * (kv_rows / 2) * 4) as u64,
            ),
            GemvDstMk::Packed { .. } => (0, 0),
        };
        for off in [x_off_bytes, q_off_bytes, kv_off_bytes] {
            assert!(
                off.is_multiple_of(256),
                "a {MK_MAX}-row GEMM tile bind offset of {off} bytes breaks the 256B \
                 storage-offset rule; the boot gate must have clamped this graph to m<={MK_MAX}"
            );
        }
        let x_slot = if row_scale.is_some() { 2u32 } else { 1u32 };
        let with_offs: Vec<(u32, &wgpu::Buffer, u64)> = binds
            .iter()
            .map(|(slot, buf)| {
                let off = match *slot {
                    s if s == x_slot => x_off_bytes,
                    31 => q_off_bytes,
                    32 | 33 => kv_off_bytes,
                    _ => 0,
                };
                (*slot, *buf, off)
            })
            .collect();
        self.push_offsets(pipeline, &with_offs, grid);
    }

    fn gemv(
        &mut self,
        proj: &GpuProj,
        x_packed: &wgpu::Buffer,
        dst: GemvDst,
        aux: Option<(&wgpu::Buffer, &wgpu::Buffer)>,
    ) {
        let (pipeline, w, row_scale, params, grid, off_label, split_label) = match proj {
            GpuProj::Bf16 { w, params, grid } => {
                let pipeline = match &dst {
                    GemvDst::Packed { .. } => self.pl.gemv_pk.clone(),
                    GemvDst::SplitQkv { .. } => self.pl.gemv_pk3.clone(),
                };
                (
                    pipeline,
                    w.raw(),
                    None,
                    params.raw(),
                    *grid,
                    "g4w-pk-off",
                    "g4w-split-p",
                )
            }
            GpuProj::Fp8 {
                w,
                row_scale,
                params,
                grid,
                fmt,
            } => {
                let fp8 = self.pl.fp8.as_ref().expect("fp8 pipelines missing");
                let pipeline = match &dst {
                    GemvDst::Packed { .. } => fp8.pk_for(*fmt),
                    GemvDst::SplitQkv { .. } => fp8.pk3_for(*fmt),
                };
                (
                    pipeline,
                    w.raw(),
                    Some(row_scale.raw()),
                    params.raw(),
                    *grid,
                    "g4w-fp8-off",
                    "g4w-fp8-split-p",
                )
            }
            GpuProj::Nvfp4 {
                w,
                scales,
                gemv_params,
                quant_params,
                grid,
                k,
                deep,
                v2,
                ..
            } => {
                let (xq, xs_pack) = aux.expect("nvfp4 gemv needs quant buffers");
                let k_blocks = k / 16;
                let qgrid = self.grid_1d(k_blocks as u64, 256);
                let quant = self.pl.quant_pk.clone();
                self.push(
                    quant,
                    &[
                        (0, x_packed),
                        (1, quant_params.raw()),
                        (2, xq),
                        (3, xs_pack),
                    ],
                    qgrid,
                );
                let y = match dst {
                    GemvDst::Packed { y, word_off } => {
                        assert_eq!(word_off, 0, "nvfp4 packed gemv writes at offset 0 only");
                        y
                    }
                    GemvDst::SplitQkv { .. } => {
                        unreachable!("nvfp4 qkv scatter is rejected at construction")
                    }
                };
                match v2 {
                    Some((pipeline, kernel)) => {
                        let (w_slot, x_slot) = if kernel.vec4_slots() {
                            (wk::gemv_nvfp4_v2::W4_SLOT, wk::gemv_nvfp4_v2::X4_SLOT)
                        } else {
                            (wk::gemv_nvfp4_v2::W2_SLOT, wk::gemv_nvfp4_v2::X2_SLOT)
                        };
                        self.push(
                            pipeline.clone(),
                            &[
                                (w_slot, w.raw()),
                                (wk::gemv_nvfp4_v2::WS_SLOT, scales.raw()),
                                (x_slot, xq),
                                (wk::gemv_nvfp4_v2::XS_SLOT, xs_pack),
                                (wk::gemv_nvfp4_v2::PARAMS_SLOT, gemv_params.raw()),
                                (wk::gemv_nvfp4_v2::Y_SLOT, y),
                            ],
                            *grid,
                        );
                    }
                    None => {
                        let gemv = if *deep {
                            self.pl
                                .gemv4_pk_deep
                                .as_ref()
                                .expect("deep pick set with no deep pipeline built")
                                .clone()
                        } else {
                            self.pl.gemv4_pk.clone()
                        };
                        self.push(
                            gemv,
                            &[
                                (0, w.raw()),
                                (1, scales.raw()),
                                (2, xq),
                                (3, xs_pack),
                                (4, gemv_params.raw()),
                                (6, y),
                            ],
                            *grid,
                        );
                    }
                }
                return;
            }
        };
        match dst {
            GemvDst::Packed { y, word_off } => {
                let off = self.uni(
                    off_label,
                    PkOffParams {
                        dst_word_off: word_off as u32,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    },
                );
                let binds = gemv_bind_list(
                    w,
                    row_scale,
                    x_packed,
                    params,
                    GemvBindDst::Packed(y),
                    Some((30, &off)),
                );
                self.push(pipeline, &binds, grid);
            }
            GemvDst::SplitQkv {
                q,
                k,
                v,
                q_rows,
                kv_rows,
                v_off,
            } => {
                let sp = self.uni(
                    split_label,
                    SplitParams {
                        q_rows: q_rows as u32,
                        kv_rows: kv_rows as u32,
                        v_off: v_off as u32,
                        pad0: 0,
                    },
                );
                let binds = gemv_bind_list(
                    w,
                    row_scale,
                    x_packed,
                    params,
                    GemvBindDst::Split { q, k, v, sp: &sp },
                    None,
                );
                self.push(pipeline, &binds, grid);
            }
        }
    }
}

struct CoopLin {
    w: GpuTensor<u32>,
    sf: GpuTensor<u32>,
    n: usize,
    k: usize,
    alpha: f32,
}

struct PfCoopCtx {
    gemm: Arc<wgpu::ComputePipeline>,
    x16: Arc<wgpu::ComputePipeline>,
    pack: Arc<wgpu::ComputePipeline>,
    xf16: GpuTensor<u32>,
    yf32: GpuTensor<f32>,
    zero: GpuTensor<f32>,
    tm: u32,
    tn: u32,
    sg: u32,
    ku: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PfCoopX16Params {
    row_words: u32,
    rows: u32,
    x_stride_words: u32,
    dst_stride_elems: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PfCoopPackParams {
    pairs_per_row: u32,
    rows: u32,
    y_stride_words: u32,
    dst_word_off: u32,
    src_stride_elems: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
}

const PF_COOP_X16_WGSL: &str = "\
enable f16;
struct X16Params { row_words: u32, rows: u32, x_stride_words: u32, dst_stride_elems: u32 };
@group(0) @binding(0) var<storage, read> cx_src: array<u32>;
@group(0) @binding(1) var<storage, read_write> cx_dst: array<f16>;
@group(0) @binding(2) var<uniform> cx_p: X16Params;
@compute @workgroup_size(256)
fn g4w_pf_bf16pairs_to_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let r = gid.y;
    if (w >= cx_p.row_words || r >= cx_p.rows) { return; }
    let v = cx_src[r * cx_p.x_stride_words + w];
    let base = r * cx_p.dst_stride_elems + w * 2u;
    cx_dst[base] = f16(bitcast<f32>(v << 16u));
    cx_dst[base + 1u] = f16(bitcast<f32>(v & 0xffff0000u));
}
";

const PF_COOP_PACK_WGSL: &str = "\
struct CoopPackParams { pairs_per_row: u32, rows: u32, y_stride_words: u32, dst_word_off: u32, src_stride_elems: u32, alpha: f32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> cp_src: array<f32>;
@group(0) @binding(1) var<storage, read_write> cp_dst: array<u32>;
@group(0) @binding(2) var<uniform> cp_p: CoopPackParams;
fn cp_bf16_encode(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    let r = 0x7fffu + ((b >> 16u) & 1u);
    return select((b + r) >> 16u, 0x7fc0u, x != x);
}
@compute @workgroup_size(256)
fn g4w_pf_coop_alpha_pack_bf16(@builtin(global_invocation_id) gid: vec3<u32>) {
    let w = gid.x;
    let r = gid.y;
    if (w >= cp_p.pairs_per_row || r >= cp_p.rows) { return; }
    let base = r * cp_p.src_stride_elems + w * 2u;
    let lo = cp_bf16_encode(cp_src[base] * cp_p.alpha) & 0xffffu;
    let hi = cp_bf16_encode(cp_src[base + 1u] * cp_p.alpha) & 0xffffu;
    cp_dst[cp_p.dst_word_off + r * cp_p.y_stride_words + w] = lo | (hi << 16u);
}
";

pub fn unswizzle_nvfp4_scales_row_major_for_the_coop_kernels_plain_sf_index(
    l: &HostNvfp4Lin,
) -> Vec<u8> {
    let k_blocks = l.k / 16;
    let k_tiles = k_blocks.div_ceil(4);
    let mut out = vec![0u8; l.n * k_blocks];
    for r in 0..l.n {
        let m_tile = r / 128;
        let d2 = (r / 32) % 4;
        let d3 = r % 32;
        for kb in 0..k_blocks {
            let si = ((m_tile * k_tiles + kb / 4) * 32 + d3) * 16 + d2 * 4 + kb % 4;
            out[r * k_blocks + kb] = l.scales_swizzled[si];
        }
    }
    out
}

fn upload_coop_lin_when_host_kept_the_nvfp4_original(
    ctx: &WgpuContext,
    proj: &HostProj,
) -> Option<CoopLin> {
    let HostProj::Nvfp4(l) = proj else {
        return None;
    };
    let (_, _, ku) = PF_COOP_TN2_SG2_KU4_STAGES_4096_F16_WELL_UNDER_THE_24576_BUDGET;
    if !l.k.is_multiple_of(16 * ku as usize) || !l.n.is_multiple_of(64) {
        return None;
    }
    let gi_folded_because_the_coop_arm_reads_unquantized_f16_activations_like_the_i8_arm =
        if l.input_global == 0.0 || !l.input_global.is_finite() {
            1.0
        } else {
            l.input_global
        };
    Some(CoopLin {
        w: GpuTensor::upload(ctx, "g4w-pf-coop-w4", &bytes_to_words(&l.packed)),
        sf: GpuTensor::upload(
            ctx,
            "g4w-pf-coop-sf",
            &bytes_to_words(&unswizzle_nvfp4_scales_row_major_for_the_coop_kernels_plain_sf_index(l)),
        ),
        n: l.n,
        k: l.k,
        alpha: l.alpha
            * gi_folded_because_the_coop_arm_reads_unquantized_f16_activations_like_the_i8_arm,
    })
}

fn upload_proj(
    ctx: &WgpuContext,
    b: &mut Builder,
    proj: &HostProj,
    variant: wk::gemv_nvfp4::GemvVariant,
) -> Result<GpuProj> {
    match proj {
        HostProj::Bf16(l) => {
            anyhow::ensure!(l.k % 8 == 0, "bf16 gemv needs k % 8 == 0, got {}", l.k);
            anyhow::ensure!(l.w.len() == l.n * l.k, "bf16 weight length mismatch");
            b.weight_bytes += 2 * (l.n as u64) * (l.k as u64);
            let w = GpuTensor::upload(ctx, "g4w-linw", &pack_pairs(&l.w));
            let grid = b.grid_1d(l.n as u64, wk::gemv_bf16::ROWS_PER_GROUP);
            let params = GpuUniform::new(
                ctx,
                "g4w-linp",
                &GemvBf16Params {
                    n_rows: l.n as u32,
                    k_elems: l.k as u32,
                    w_row_words: (l.k / 2) as u32,
                    groups_x: grid.0,
                },
            );
            Ok(GpuProj::Bf16 { w, params, grid })
        }
        HostProj::Nvfp4(l) => {
            anyhow::ensure!(l.k % 16 == 0, "nvfp4 gemv needs k % 16 == 0, got {}", l.k);
            anyhow::ensure!(
                l.packed.len() == l.n * l.k / 2,
                "nvfp4 packed length mismatch"
            );
            let k_blocks = l.k / 16;
            let want = wk::gemv_nvfp4::swizzled_scale_len(l.n, k_blocks);
            anyhow::ensure!(
                l.scales_swizzled.len() >= want,
                "nvfp4 swizzled scales: got {} want {}",
                l.scales_swizzled.len(),
                want
            );
            b.weight_bytes += (l.packed.len() as u64) + (l.n as u64) * (k_blocks as u64);
            let w = GpuTensor::upload(ctx, "g4w-lin4w", &bytes_to_words(&l.packed));
            let scales = GpuTensor::upload(ctx, "g4w-lin4s", &bytes_to_words(&l.scales_swizzled));
            let v2 = if nvfp4_v2_enabled(ctx) {
                nvfp4_v2_pipeline(ctx, l.n, l.k)
            } else {
                None
            };
            b.nvfp4_projs += 1;
            b.nvfp4_v2_routed += usize::from(v2.is_some());
            let proj_variant = if nvfp4_shape_pick_enabled()
                && !matches!(variant, wk::gemv_nvfp4::GemvVariant::Tree)
            {
                wk::gemv_nvfp4::select_variant_for_shape(ctx, l.n, l.k)
            } else {
                variant
            };
            let deep = !matches!(variant, wk::gemv_nvfp4::GemvVariant::Tree)
                && matches!(proj_variant, wk::gemv_nvfp4::GemvVariant::Tree);
            let rows_per_group = match &v2 {
                Some((_, _, rpg)) => *rpg,
                None => proj_variant.rows_per_group(),
            };
            let v2 = v2.map(|(p, kernel, _)| (p, kernel));
            let grid = b.grid_1d(l.n as u64, rows_per_group);
            let gemv_params = GpuUniform::new(
                ctx,
                "g4w-lin4p",
                &GemvNvfp4Params {
                    alpha: l.alpha,
                    n_rows: l.n as u32,
                    k_blocks: k_blocks as u32,
                    k_tiles: wk::gemv_nvfp4::k_tiles(k_blocks) as u32,
                    w_row_words: (l.k / 8) as u32,
                    groups_x: grid.0,
                    pad0: 0,
                    pad1: 0,
                },
            );
            let quant_params = GpuUniform::new(
                ctx,
                "g4w-lin4q",
                &QuantRowParams {
                    global_scale: l.input_global,
                    k_blocks: k_blocks as u32,
                    pad0: 0,
                    pad1: 0,
                },
            );
            Ok(GpuProj::Nvfp4 {
                w,
                scales,
                gemv_params,
                quant_params,
                grid,
                k: l.k,
                n: l.n,
                alpha: l.alpha,
                mk_input_globals_one_per_slot: GpuTensor::upload(
                    ctx,
                    "g4w-lin4-mk-glob",
                    &vec![l.input_global; MK_MAX],
                ),
                deep,
                v2,
            })
        }
        HostProj::Fp8(l) => {
            let rows_per_group =
                b.pl.fp8.as_ref().map(|p| p.rows_per_group).ok_or_else(|| {
                    anyhow::anyhow!("fp8 projection uploaded without fp8 pipelines")
                })?;
            wk::quant_gemv::group_rule(l.k, l.group).map_err(err)?;
            let per_row = wk::quant_gemv::scales_per_row(l.k, l.group);
            anyhow::ensure!(
                l.wq.len() == l.n * l.k / 4,
                "fp8 packed length {} != n*k/4 = {}",
                l.wq.len(),
                l.n * l.k / 4
            );
            anyhow::ensure!(
                l.row_scale.len() == l.n * per_row,
                "q8 scale length {} != n*scales_per_row = {}",
                l.row_scale.len(),
                l.n * per_row
            );
            anyhow::ensure!(
                l.n % 2 == 0,
                "fp8 packed gemv needs n % 2 == 0, got {}",
                l.n
            );
            b.weight_bytes += (l.n as u64) * (l.k as u64) + 4 * (l.n as u64) * (per_row as u64);
            let w = GpuTensor::upload(ctx, "g4w-lin8w", &l.wq);
            let row_scale = GpuTensor::upload(ctx, "g4w-lin8s", &l.row_scale);
            let grid = b.grid_1d(l.n as u64, rows_per_group);
            let params = GpuUniform::new(ctx, "g4w-lin8p", &{
                let mut p = wk::quant_gemv::params_for(l.n, l.k, l.group, grid.0);
                p.pad1 = attn_fp8_legacy_epilogue();
                p
            });
            Ok(GpuProj::Fp8 {
                w,
                row_scale,
                params,
                grid,
                fmt: l.fmt,
            })
        }
    }
}

impl Gemma4Wgpu {
    pub fn new(config: Gemma4Config, weights: &HostWeights, max_seq: usize) -> Result<Self> {
        Self::build(config, weights, max_seq, batch_slots_default())
    }

    pub fn new_batched(
        config: Gemma4Config,
        weights: &HostWeights,
        max_seq: usize,
        slots: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            slots <= MK_MAX,
            "batch decode wants {slots} slots; the M-row GEMV twins are generated up to MK_MAX={MK_MAX} rows"
        );
        Self::build(config, weights, max_seq, if slots >= 2 { slots } else { 0 })
    }

    fn build(
        config: Gemma4Config,
        weights: &HostWeights,
        max_seq: usize,
        batch_slots: usize,
    ) -> Result<Self> {
        let t_boot = Instant::now();
        anyhow::ensure!(
            config.num_kv_shared_layers == 0,
            "gemma4_wgpu: per-layer KV sharing (num_kv_shared_layers={}) is not wired on the wgpu path; \
             the CUDA gemma4.rs fp8 decode cache also allocates per-layer KV with no sharing",
            config.num_kv_shared_layers
        );
        anyhow::ensure!(
            !config.has_per_layer_embeddings(),
            "gemma4_wgpu: per-layer embeddings unsupported"
        );
        anyhow::ensure!(
            weights.layers.len() == config.num_hidden_layers,
            "gemma4_wgpu: {} host layers for {} config layers",
            weights.layers.len(),
            config.num_hidden_layers
        );
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
        let variant = nvfp4_variant(ctx);
        let plan = Q8Plan::from_env();
        let attn_fp8 = plan.attn_fp8;
        eprintln!("{}", weight_format_boot_line());

        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let eps = config.rms_norm_eps as f32;
        let n_q = config.num_attention_heads;
        anyhow::ensure!(
            hidden.is_multiple_of(8) && inter.is_multiple_of(2) && vocab.is_multiple_of(8)
        );

        let hd_s = config.head_dim_for(LayerType::SlidingAttention);
        let hd_f = config.head_dim_for(LayerType::FullAttention);
        let nkv_s = config.num_kv_heads_for(LayerType::SlidingAttention);
        let nkv_f = config.num_kv_heads_for(LayerType::FullAttention);
        let hd_max = hd_s.max(hd_f);
        let q_dim_max = n_q * hd_max;
        let kv_dim_max = (nkv_s * hd_s).max(nkv_f * hd_f);
        anyhow::ensure!(hd_max <= wk::flash_decode::MAX_HEAD_DIM);

        let fuse = if hd_s.is_multiple_of(4) && hd_f.is_multiple_of(4) {
            fuse_mask(ctx, hd_max)
        } else {
            0
        };

        let mut pf_m = prefill_m_wide_via_nv_g4_wgpu_pf_m_in_mk_max_row_tiles().min(max_seq);
        if pf_m > MK_MAX {
            pf_m -= pf_m % MK_MAX;
            let tile_offsets_256b_aligned = (nkv_s * hd_s).is_multiple_of(8)
                && (nkv_f * hd_f).is_multiple_of(8)
                && inter.is_multiple_of(8);
            if !tile_offsets_256b_aligned {
                eprintln!(
                    "[gemma4_wgpu] wide prefill m clamped to {MK_MAX}: a {MK_MAX}-row GEMM tile \
                     bind offset (32*kv_dim or 32*inter bytes) misses the 256B storage-offset rule"
                );
                pf_m = MK_MAX;
            }
        }
        if pf_m > 0 && attn_fp8 && attn_fp8_legacy_epilogue() != 0 {
            eprintln!(
                "[gemma4_wgpu] chunked prefill off: the legacy fp8 epilogue scales after the reduction and has no M-row twin"
            );
            pf_m = 0;
        }

        if pf_m > 0
            && (!(n_q * hd_s).is_multiple_of(128)
                || !(n_q * hd_f).is_multiple_of(128)
                || !hidden.is_multiple_of(16))
        {
            eprintln!(
                "[gemma4_wgpu] chunked prefill off: q_dim {}/{} or hidden {hidden} miss the 256B storage-offset rule",
                n_q * hd_s,
                n_q * hd_f
            );
            pf_m = 0;
        }

        let mut bt_slots = batch_slots.min(MK_MAX);
        if bt_slots >= 2 && attn_fp8 && attn_fp8_legacy_epilogue() != 0 {
            eprintln!(
                "[gemma4_wgpu] batch decode off: the legacy fp8 epilogue scales after the reduction and has no M-row twin"
            );
            bt_slots = 0;
        }
        if bt_slots >= 2
            && (!(n_q * hd_s).is_multiple_of(128)
                || !(n_q * hd_f).is_multiple_of(128)
                || !hidden.is_multiple_of(16))
        {
            eprintln!(
                "[gemma4_wgpu] batch decode off: q_dim {}/{} or hidden {hidden} miss the 256B storage-offset rule",
                n_q * hd_s,
                n_q * hd_f
            );
            bt_slots = 0;
        }
        if bt_slots >= 2
            && (!(nkv_s * hd_s).is_multiple_of(128) || !(nkv_f * hd_f).is_multiple_of(128))
        {
            eprintln!(
                "[gemma4_wgpu] batch decode off: kv_dim {}/{} misses the 256B storage-offset rule the per-slot kv write needs",
                nkv_s * hd_s,
                nkv_f * hd_f
            );
            bt_slots = 0;
        }
        if bt_slots >= 2 && max_seq * bt_slots > u32::MAX as usize {
            eprintln!("[gemma4_wgpu] batch decode off: {bt_slots} x max_seq {max_seq} overflows the u32 cache row index");
            bt_slots = 0;
        }
        let sliding_ring_rows: Option<usize> = if sliding_kv_ring_enabled() {
            let rows = sliding_kv_ring_rows_window_plus_prefill_chunk_plus_headroom(
                config.sliding_window,
                pf_m,
            );
            if max_seq > rows {
                eprintln!(
                    "[gemma4_wgpu] sliding kv ring: {rows} rows/layer (window {} + chunk {} + headroom {}) instead of {max_seq}",
                    config.sliding_window,
                    pf_m.max(1),
                    SLIDING_KV_RING_HEADROOM_SLOTS_MATCHING_CUDA_KV_FP8_RING_SLOTS
                );
                Some(rows)
            } else {
                eprintln!(
                    "[gemma4_wgpu] sliding kv ring off: max_seq {max_seq} <= ring {rows}, full-depth rows are already smaller"
                );
                None
            }
        } else {
            None
        };
        if bt_slots >= 2 && sliding_ring_rows.is_some() {
            eprintln!(
                "[gemma4_wgpu] batch decode off: the sliding kv ring wraps absolute slot indices, so per-slot bases at j*max_seq would alias"
            );
            bt_slots = 0;
        }
        let any_q8 = plan.any_fp8(weights);
        let t_pl = Instant::now();
        let mk_gemm_tile_rows_never_exceed_the_shader_unroll = pf_m.min(MK_MAX).max(bt_slots);
        let pl = build_pipelines(
            ctx,
            variant,
            any_q8,
            (fuse != 0).then_some(hd_max),
            mk_gemm_tile_rows_never_exceed_the_shader_unroll,
        )?;
        let pipelines_s = t_pl.elapsed().as_secs_f64();
        let mut b = Builder {
            ctx,
            pl: &pl,
            passes: Vec::new(),
            pf_passes: Vec::new(),
            bt_passes: Vec::new(),
            vf_passes: Vec::new(),
            dst: PassDst::Decode,
            keep: Vec::new(),
            weight_bytes: 0,
            nvfp4_projs: 0,
            nvfp4_v2_routed: 0,
            mk4: None,
        };

        let o_nvfp4 = std::env::var("NV_WGPU_O_NVFP4").is_ok_and(|v| v != "0");
        let quant_k_max = weights
            .layers
            .iter()
            .flat_map(|l| [l.gate_up.k(), l.down.k(), l.o.k()])
            .max()
            .unwrap_or(hidden)
            .max(hidden);

        anyhow::ensure!(
            weights.embed.len() == vocab * hidden,
            "embed length {} != {}",
            weights.embed.len(),
            vocab * hidden
        );
        anyhow::ensure!(vocab.is_multiple_of(4), "vocab must be a multiple of 4");
        let split_row = vocab / 2;
        let half_bytes = (split_row * hidden / 2 * 4) as u64;
        anyhow::ensure!(
            half_bytes <= ctx.caps.max_storage_buffer_binding_size,
            "embed half needs {half_bytes} bytes in one storage binding; device allows {}",
            ctx.caps.max_storage_buffer_binding_size
        );
        let embed_lo = GpuTensor::upload(
            ctx,
            "g4w-embed-lo",
            &pack_pairs(&weights.embed[..split_row * hidden]),
        );
        let embed_hi = GpuTensor::upload(
            ctx,
            "g4w-embed-hi",
            &pack_pairs(&weights.embed[split_row * hidden..]),
        );
        ctx.queue.submit(std::iter::empty());
        ctx.poll_blocking().map_err(err)?;
        let final_norm = GpuTensor::upload(ctx, "g4w-final-norm", &pack_pairs(&weights.final_norm));

        let lmhead_i8 = std::env::var("NV_WGPU_LMHEAD_INT8").is_ok_and(|v| v != "0");
        let kv_slots = bt_slots.max(1);
        let lmhead_i8_bufs = if lmhead_i8 {
            anyhow::ensure!(
                hidden.is_multiple_of(16),
                "int8 lm_head needs hidden % 16 == 0"
            );
            let wq_lo = GpuTensor::<u32>::zeroed(ctx, "g4w-lm-i8-lo", split_row * hidden / 4);
            let wq_hi = GpuTensor::<u32>::zeroed(ctx, "g4w-lm-i8-hi", split_row * hidden / 4);
            let rs_lo = GpuTensor::<f32>::zeroed(ctx, "g4w-lm-rs-lo", split_row);
            let rs_hi = GpuTensor::<f32>::zeroed(ctx, "g4w-lm-rs-hi", split_row);
            let rq_grid = dispatch::workgroup_count_1d(ctx, split_row as u64, 1);
            let rq_p = GpuUniform::new(
                ctx,
                "g4w-rq-p",
                &RowQuantParams {
                    n_rows: split_row as u32,
                    k_elems: hidden as u32,
                    src_row_words: (hidden / 2) as u32,
                    dst_row_words: (hidden / 4) as u32,
                    groups_x: rq_grid.0,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                },
            );
            let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
            for (src, dst, sc) in [(&embed_lo, &wq_lo, &rs_lo), (&embed_hi, &wq_hi, &rs_hi)] {
                let bind = dispatch::bind_group(
                    ctx,
                    &pl.rowquant_i8,
                    &[
                        (9, src.raw()),
                        (10, dst.raw()),
                        (11, sc.raw()),
                        (12, rq_p.raw()),
                    ],
                );
                let mut enc = ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pl.rowquant_i8);
                    pass.set_bind_group(0, &bind, &[]);
                    pass.dispatch_workgroups(rq_grid.0, rq_grid.1, rq_grid.2);
                }
                ctx.queue.submit([enc.finish()]);
                ctx.poll_blocking().map_err(err)?;
            }
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_wgpu lm_head int8 rowquant: {e}");
            }
            let ones = vec![bf16_bits(1.0); hidden];
            let wn_ones = GpuTensor::upload(ctx, "g4w-lm-wn1", &pack_pairs(&ones));
            let rstd_one = GpuTensor::<f32>::upload(ctx, "g4w-lm-rstd1", &[1.0f32]);
            Some((wq_lo, wq_hi, rs_lo, rs_hi, wn_ones, rstd_one))
        } else {
            None
        };
        b.weight_bytes += if lmhead_i8_bufs.is_some() {
            (vocab as u64) * (hidden as u64) + 4 * (vocab as u64)
        } else {
            2 * (vocab as u64) * (hidden as u64)
        };

        let hid_a = GpuTensor::<u32>::zeroed(ctx, "g4w-hid-a", hidden / 2);
        let hid_b = GpuTensor::<u32>::zeroed(ctx, "g4w-hid-b", hidden / 2);
        let t0 = GpuTensor::<u32>::zeroed(ctx, "g4w-t0", hidden / 2);
        let t1 = GpuTensor::<u32>::zeroed(ctx, "g4w-t1", hidden / 2);

        let cap_align = ctx.device.limits().min_storage_buffer_offset_alignment as usize / 4;
        let w4_cap = std::env::var("NV_G4_W4_CAPTURE")
            .ok()
            .filter(|v| v != "0")
            .map(|_| {
                let gu_words = (hidden / 2).next_multiple_of(cap_align);
                let dn_words = (inter / 2).next_multiple_of(cap_align);
                let layers = weights.layers.len();
                W4Capture {
                    buf: GpuTensor::<u32>::zeroed(ctx, "g4w-w4cap", layers * (gu_words + dn_words)),
                    layers,
                    gu_words,
                    dn_words,
                }
            });
        let qa = GpuTensor::<u32>::zeroed(ctx, "g4w-qa", q_dim_max / 2);
        let qb = GpuTensor::<u32>::zeroed(ctx, "g4w-qb", q_dim_max / 2);
        let ka = GpuTensor::<u32>::zeroed(ctx, "g4w-ka", kv_dim_max / 2);
        let kb = GpuTensor::<u32>::zeroed(ctx, "g4w-kb", kv_dim_max / 2);
        let va = GpuTensor::<u32>::zeroed(ctx, "g4w-va", kv_dim_max / 2);
        let vb = GpuTensor::<u32>::zeroed(ctx, "g4w-vb", kv_dim_max / 2);
        let q_f32 = GpuTensor::<f32>::zeroed(ctx, "g4w-qf32", q_dim_max);
        let attn_pack = GpuTensor::<u32>::zeroed(ctx, "g4w-attn-pack", q_dim_max / 2);
        let gu_pack = GpuTensor::<u32>::zeroed(ctx, "g4w-gu-pack", inter.max(hidden));
        let act_pack = GpuTensor::<u32>::zeroed(ctx, "g4w-act-pack", inter.max(hidden) / 2);
        let xq = GpuTensor::<u32>::zeroed(ctx, "g4w-xq", quant_k_max / 8);
        let xs_pack = GpuTensor::<u32>::zeroed(ctx, "g4w-xs-pack", (quant_k_max / 16).div_ceil(4));
        let logits_un = if lmhead_i8_bufs.is_some() {
            Some((
                GpuTensor::<u32>::zeroed(ctx, "g4w-logits-un-lo", vocab / 2),
                GpuTensor::<u32>::zeroed(ctx, "g4w-logits-un-hi", vocab / 2),
            ))
        } else {
            None
        };
        let logits_pk = GpuTensor::<u32>::zeroed(ctx, "g4w-logits-pk", vocab / 2);
        let logits_f32 = GpuTensor::<f32>::zeroed(ctx, "g4w-logits-f32", vocab);
        let am_val = GpuTensor::<f32>::zeroed(ctx, "g4w-am-val", wk::graph_decode::ARGMAX_BLOCKS);
        let am_idx = GpuTensor::<i32>::zeroed(ctx, "g4w-am-idx", wk::graph_decode::ARGMAX_BLOCKS);
        let token_out = GpuTensor::<u32>::zeroed(ctx, "g4w-token-out", 1);
        let chain_out = GpuTensor::<u32>::zeroed(ctx, "g4w-chain-out", MAX_CHAIN);
        let scratch_len = n_q * flash_splits() as usize * (hd_max + 2);
        let scratch = GpuTensor::<f32>::zeroed(ctx, "g4w-flash-scratch", scratch_len);

        let tok_idx = GpuTensor::<i32>::upload(ctx, "g4w-tok-idx", &[0]);
        let rope_pos = GpuTensor::<i32>::upload(ctx, "g4w-rope-pos", &[0]);
        let kv_start = GpuTensor::<i32>::upload(ctx, "g4w-kv-start", &[0]);

        let ring_for = |kind: LayerType| -> u32 {
            match (kind, sliding_ring_rows) {
                (LayerType::SlidingAttention, Some(rows)) => rows as u32,
                _ => 0,
            }
        };
        let kv_rows_for = |kind: LayerType| -> usize {
            match (kind, sliding_ring_rows) {
                (LayerType::SlidingAttention, Some(rows)) => rows,
                _ => max_seq * kv_slots,
            }
        };
        let mk_fd = |kind: LayerType| FdParams {
            n_heads: n_q as u32,
            n_kv: config.num_kv_heads_for(kind) as u32,
            head_dim: config.head_dim_for(kind) as u32,
            total: 0,
            start: 0,
            splits: flash_splits(),
            ring: ring_for(kind),
            out_bf16: 1,
            scaling: 1.0,
            pad0: 0,
            fused: 0,
            pad2: 0,
            m_rows: 1,
            window: 0,
            pad3: 0,
            pad4: 0,
        };
        let fd_sliding_base = mk_fd(LayerType::SlidingAttention);
        let fd_full_base = mk_fd(LayerType::FullAttention);
        let fd_sliding = GpuUniform::new(ctx, "g4w-fd-s", &fd_sliding_base);
        let fd_full = GpuUniform::new(ctx, "g4w-fd-f", &fd_full_base);

        let pf_bufs = (pf_m > 0).then(|| {
            let h = pf_m * hidden / 2;
            PfBufs {
                splice_rows: GpuTensor::zeroed(ctx, "g4w-pf-splice-rows", h),
                splice_mask: GpuTensor::upload(ctx, "g4w-pf-splice-mask", &vec![0u32; pf_m.max(4)]),
                hid_a: GpuTensor::zeroed(ctx, "g4w-pf-hid-a", h),
                hid_b: GpuTensor::zeroed(ctx, "g4w-pf-hid-b", h),
                t0: GpuTensor::zeroed(ctx, "g4w-pf-t0", h),
                t1: GpuTensor::zeroed(ctx, "g4w-pf-t1", h),
                qa: GpuTensor::zeroed(ctx, "g4w-pf-qa", pf_m * q_dim_max / 2),
                qb: GpuTensor::zeroed(ctx, "g4w-pf-qb", pf_m * q_dim_max / 2),
                ka: GpuTensor::zeroed(ctx, "g4w-pf-ka", pf_m * kv_dim_max / 2),
                kb: GpuTensor::zeroed(ctx, "g4w-pf-kb", pf_m * kv_dim_max / 2),
                va: GpuTensor::zeroed(ctx, "g4w-pf-va", pf_m * kv_dim_max / 2),
                vb: GpuTensor::zeroed(ctx, "g4w-pf-vb", pf_m * kv_dim_max / 2),
                q_f32: GpuTensor::zeroed(ctx, "g4w-pf-qf32", pf_m * q_dim_max),
                attn_pack: GpuTensor::zeroed(ctx, "g4w-pf-attn-pack", pf_m * q_dim_max / 2),
                gu_pack: GpuTensor::zeroed(ctx, "g4w-pf-gu-pack", pf_m * inter.max(hidden)),
                act_pack: GpuTensor::zeroed(ctx, "g4w-pf-act-pack", pf_m * inter.max(hidden) / 2),
            }
        });
        let pf_tok_idx = GpuTensor::<i32>::upload(ctx, "g4w-pf-tok", &vec![0i32; pf_m.max(1)]);
        let pf_rope_pos = GpuTensor::<i32>::upload(ctx, "g4w-pf-rope", &vec![0i32; pf_m.max(1)]);
        let pf_kv_start = GpuTensor::<i32>::upload(ctx, "g4w-pf-kvs", &[0]);
        let pf_fd_s: Vec<GpuUniform<FdParams>> = (0..pf_m)
            .map(|_| GpuUniform::new(ctx, "g4w-pf-fd-s", &fd_sliding_base))
            .collect();
        let pf_fd_f: Vec<GpuUniform<FdParams>> = (0..pf_m)
            .map(|_| GpuUniform::new(ctx, "g4w-pf-fd-f", &fd_full_base))
            .collect();

        let bt_bufs = (bt_slots >= 2).then(|| {
            let s = bt_slots;
            let h = s * hidden / 2;
            BtBufs {
                core: PfBufs {
                    splice_rows: GpuTensor::zeroed(ctx, "g4w-bt-splice-rows", h),
                    splice_mask: GpuTensor::upload(ctx, "g4w-bt-splice-mask", &vec![0u32; s.max(4)]),
                    hid_a: GpuTensor::zeroed(ctx, "g4w-bt-hid-a", h),
                    hid_b: GpuTensor::zeroed(ctx, "g4w-bt-hid-b", h),
                    t0: GpuTensor::zeroed(ctx, "g4w-bt-t0", h),
                    t1: GpuTensor::zeroed(ctx, "g4w-bt-t1", h),
                    qa: GpuTensor::zeroed(ctx, "g4w-bt-qa", s * q_dim_max / 2),
                    qb: GpuTensor::zeroed(ctx, "g4w-bt-qb", s * q_dim_max / 2),
                    ka: GpuTensor::zeroed(ctx, "g4w-bt-ka", s * kv_dim_max / 2),
                    kb: GpuTensor::zeroed(ctx, "g4w-bt-kb", s * kv_dim_max / 2),
                    va: GpuTensor::zeroed(ctx, "g4w-bt-va", s * kv_dim_max / 2),
                    vb: GpuTensor::zeroed(ctx, "g4w-bt-vb", s * kv_dim_max / 2),
                    q_f32: GpuTensor::zeroed(ctx, "g4w-bt-qf32", s * q_dim_max),
                    attn_pack: GpuTensor::zeroed(ctx, "g4w-bt-attn-pack", s * q_dim_max / 2),
                    gu_pack: GpuTensor::zeroed(ctx, "g4w-bt-gu-pack", s * inter.max(hidden)),
                    act_pack: GpuTensor::zeroed(ctx, "g4w-bt-act-pack", s * inter.max(hidden) / 2),
                },
                logits_pk: GpuTensor::zeroed(ctx, "g4w-bt-logits-pk", s * vocab / 2),
                am_val: GpuTensor::zeroed(
                    ctx,
                    "g4w-bt-am-val",
                    s * wk::graph_decode::ARGMAX_BLOCKS,
                ),
                am_idx: GpuTensor::zeroed(
                    ctx,
                    "g4w-bt-am-idx",
                    s * wk::graph_decode::ARGMAX_BLOCKS,
                ),
            }
        });
        let bt_logits_f32 =
            GpuTensor::<f32>::zeroed(ctx, "g4w-bt-logits-f32", bt_slots.max(1) * vocab);
        let bt_tok_idx = GpuTensor::<i32>::upload(ctx, "g4w-bt-tok", &vec![0i32; bt_slots.max(1)]);
        let bt_rope_pos =
            GpuTensor::<i32>::upload(ctx, "g4w-bt-rope", &vec![0i32; bt_slots.max(1)]);
        let bt_kv_start: Vec<GpuTensor<i32>> = (0..bt_slots)
            .map(|j| GpuTensor::<i32>::upload(ctx, "g4w-bt-kvs", &[(j * max_seq) as i32]))
            .collect();
        let bt_token_out = GpuTensor::<u32>::zeroed(ctx, "g4w-bt-token-out", bt_slots.max(1));
        let bt_fd_s: Vec<GpuUniform<FdParams>> = (0..bt_slots)
            .map(|_| GpuUniform::new(ctx, "g4w-bt-fd-s", &fd_sliding_base))
            .collect();
        let bt_fd_f: Vec<GpuUniform<FdParams>> = (0..bt_slots)
            .map(|_| GpuUniform::new(ctx, "g4w-bt-fd-f", &fd_full_base))
            .collect();

        let mut rope_bufs = Vec::new();
        for kind in [LayerType::SlidingAttention, LayerType::FullAttention] {
            let hd = config.head_dim_for(kind);
            let (cos, sin) = rope_tables(
                hd,
                config.rope_theta_for(kind),
                config.rope_partial_factor_for(kind),
                max_seq,
            );
            rope_bufs.push((
                GpuTensor::upload(ctx, "g4w-rope-cos", &cos),
                GpuTensor::upload(ctx, "g4w-rope-sin", &sin),
            ));
        }

        let mut kv_layers = Vec::with_capacity(config.num_hidden_layers);
        for kind in &config.layer_types {
            let hd = config.head_dim_for(*kind);
            let nkv = config.num_kv_heads_for(*kind);
            let rows = kv_rows_for(*kind);
            kv_layers.push(GpuLayerKv {
                k_fp8: GpuTensor::zeroed(ctx, "g4w-kc", rows * nkv * hd / 4),
                v_fp8: GpuTensor::zeroed(ctx, "g4w-vc", rows * nkv * hd / 4),
                k_scales: GpuTensor::zeroed(ctx, "g4w-ks", rows * nkv),
                v_scales: GpuTensor::zeroed(ctx, "g4w-vs", rows * nkv),
            });
        }

        let gather_p = b.uni(
            "g4w-gather-p",
            Gather2Params {
                split_row: split_row as u32,
                hidden_words: (hidden / 2) as u32,
                vocab: vocab as u32,
                pad0: 0,
            },
        );
        b.push(
            pl.gather.clone(),
            &[
                (3, embed_lo.raw()),
                (4, embed_hi.raw()),
                (5, tok_idx.raw()),
                (6, t0.raw()),
                (7, &gather_p),
            ],
            (1, 1, 1),
        );
        let embed_scale = (hidden as f32).sqrt();
        let scale_p = b.uni(
            "g4w-embed-scale-p",
            ScaleParams {
                n: hidden as u32,
                n_words: (hidden / 2) as u32,
                scale: embed_scale,
                cap: 0.0,
                inv_cap: 0.0,
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        let sgrid = b.grid_1d((hidden / 2) as u64, 256);
        b.push(
            pl.scale.clone(),
            &[(0, t0.raw()), (2, hid_a.raw()), (3, &scale_p)],
            sgrid,
        );

        let ln_in_all: Vec<GpuTensor<u32>> = weights
            .layers
            .iter()
            .map(|l| GpuTensor::upload(ctx, "g4w-ln-in", &pack_pairs(&l.input_ln)))
            .collect();

        let mut layer_gpu: Vec<LayerGpu> = Vec::with_capacity(weights.layers.len());
        for c in [&CONV_DEQ_NS, &CONV_BF16_NS, &CONV_Q8_NS, &CONV_ATTN_Q8_NS] {
            c.store(0, Ordering::Relaxed);
        }
        let mut conv_s = 0f64;
        let mut upload_s = 0f64;
        let pf_coop_wanted =
            pf_coop_w4a16_ffn_opt_in_default_off_because_only_measured_on_this_blackwell_adapter()
                && pf_m >= 16
                && pf_m.is_multiple_of(16)
                && pf_m / 16 <= 8
                && ctx.caps.coop_gemm_tile().is_some();
        if pf_coop_w4a16_ffn_opt_in_default_off_because_only_measured_on_this_blackwell_adapter()
            && !pf_coop_wanted
        {
            eprintln!(
                "[gemma4_wgpu] NV_G4_PF_COOP=1 ignored: needs a 16x16x16 f16xf16->f32 coop tile \
                 ({:?}) and a prefill chunk m in 16..=128 that is a multiple of 16 (m={pf_m})",
                ctx.caps.coop_gemm_tile()
            );
        }
        let t_layers = Instant::now();
        for (li, hl) in weights.layers.iter().enumerate() {
            let kind = hl.kind;
            let hd = config.head_dim_for(kind);
            let nkv = config.num_kv_heads_for(kind);
            let q_dim = n_q * hd;
            let kv_dim = nkv * hd;
            anyhow::ensure!(
                hl.qkv.n() == q_dim + kv_dim * if hl.has_v { 2 } else { 1 },
                "layer {li}: qkv rows {} mismatch",
                hl.qkv.n()
            );
            anyhow::ensure!(hl.qkv.k() == hidden && hl.o.k() == q_dim && hl.o.n() == hidden);
            anyhow::ensure!(hl.gate_up.n() == 2 * inter && hl.gate_up.k() == hidden);
            anyhow::ensure!(hl.down.n() == hidden && hl.down.k() == inter);
            anyhow::ensure!(
                !matches!(hl.qkv, HostProj::Nvfp4(_)),
                "layer {li}: nvfp4 qkv is not wired for the fused split epilogue"
            );

            let t_c = Instant::now();
            let t_aq = Instant::now();
            let qkv_owned = plan.to_fp8(&hl.qkv, li, ProjRole::Attn);
            let o_fp8 = plan.to_fp8(&hl.o, li, ProjRole::Attn);
            if qkv_owned.is_some() || o_fp8.is_some() {
                conv_add(&CONV_ATTN_Q8_NS, t_aq);
            }
            let o_owned;
            let o_src = match (&hl.o, o_nvfp4, o_fp8) {
                (_, _, Some(f)) => {
                    o_owned = f;
                    &o_owned
                }
                (HostProj::Bf16(l), true, None) if l.k % 16 == 0 && l.n % 16 == 0 => {
                    o_owned = HostProj::Nvfp4(quantize_nvfp4_host(&l.w, l.n, l.k));
                    &o_owned
                }
                _ => &hl.o,
            };
            let gu_owned = plan.to_fp8(&hl.gate_up, li, ProjRole::GateUp);
            let dn_owned = plan.to_fp8(&hl.down, li, ProjRole::Down);
            conv_s += t_c.elapsed().as_secs_f64();
            let t_u = Instant::now();
            let qkv = upload_proj(ctx, &mut b, qkv_owned.as_ref().unwrap_or(&hl.qkv), variant)?;
            let o = upload_proj(ctx, &mut b, o_src, variant)?;
            let gate_up = upload_proj(
                ctx,
                &mut b,
                gu_owned.as_ref().unwrap_or(&hl.gate_up),
                variant,
            )?;
            let down = upload_proj(ctx, &mut b, dn_owned.as_ref().unwrap_or(&hl.down), variant)?;
            let (gate_up_coop, down_coop) = if pf_coop_wanted {
                (
                    upload_coop_lin_when_host_kept_the_nvfp4_original(ctx, &hl.gate_up),
                    upload_coop_lin_when_host_kept_the_nvfp4_original(ctx, &hl.down),
                )
            } else {
                (None, None)
            };
            upload_s += t_u.elapsed().as_secs_f64();

            let ln_in = &ln_in_all[li];
            let ln_pa = GpuTensor::upload(ctx, "g4w-ln-pa", &pack_pairs(&hl.post_attn_ln));
            let ln_pf = GpuTensor::upload(ctx, "g4w-ln-pf", &pack_pairs(&hl.pre_ff_ln));
            let ln_po = GpuTensor::upload(ctx, "g4w-ln-po", &pack_pairs(&hl.post_ff_ln));
            let qn = GpuTensor::upload(ctx, "g4w-qn", &pack_pairs(&hl.q_norm));
            let kn = GpuTensor::upload(ctx, "g4w-kn", &pack_pairs(&hl.k_norm));
            let ones: Vec<u16> = vec![bf16_bits(1.0); hd];
            let vn = GpuTensor::upload(ctx, "g4w-vn", &pack_pairs(&ones));

            let (x_in, x_out) = if li % 2 == 0 {
                (hid_a.raw(), hid_b.raw())
            } else {
                (hid_b.raw(), hid_a.raw())
            };

            if fuse & FUSE_NORM_ADD_NORM == 0 || li == 0 {
                b.rms(x_in, ln_in.raw(), t0.raw(), 1, hidden, eps);
            }
            let v_off = if hl.has_v { q_dim + kv_dim } else { q_dim };
            b.gemv(
                &qkv,
                t0.raw(),
                GemvDst::SplitQkv {
                    q: qa.raw(),
                    k: ka.raw(),
                    v: va.raw(),
                    q_rows: q_dim,
                    kv_rows: kv_dim,
                    v_off,
                },
                Some((xq.raw(), xs_pack.raw())),
            );

            let kidx = match kind {
                LayerType::SlidingAttention => 0usize,
                LayerType::FullAttention => 1,
            };
            let (cos, sin) = (&rope_bufs[kidx].0, &rope_bufs[kidx].1);
            let half = hd / 2;
            let kv = &kv_layers[li];

            if let Some(fp) = pl.fuse.as_ref().filter(|_| fuse & FUSE_HEAD_PREP != 0) {
                let hp_p = b.uni(
                    "g4w-hp-p",
                    HpParams {
                        n_q: n_q as u32,
                        n_kv: nkv as u32,
                        head_dim: hd as u32,
                        half_dim: half as u32,
                        eps,
                        words: half as u32,
                        out_words: (hd / 4) as u32,
                        ring: ring_for(kind),
                    },
                );
                let hgrid = b.grid_1d((n_q + 2 * nkv) as u64, 1);
                b.push(
                    fp.head_prep.clone(),
                    &[
                        (0, qa.raw()),
                        (1, ka.raw()),
                        (2, va.raw()),
                        (3, qn.raw()),
                        (4, kn.raw()),
                        (5, vn.raw()),
                        (6, q_f32.raw()),
                        (7, kv.k_fp8.raw()),
                        (8, kv.k_scales.raw()),
                        (9, kv.v_fp8.raw()),
                        (10, kv.v_scales.raw()),
                        (11, cos.raw()),
                        (12, sin.raw()),
                        (13, rope_pos.raw()),
                        (14, kv_start.raw()),
                        (15, &hp_p),
                    ],
                    hgrid,
                );
            } else {
                b.rms(qa.raw(), qn.raw(), qb.raw(), n_q, hd, eps);
                b.rms(ka.raw(), kn.raw(), kb.raw(), nkv, hd, eps);
                b.rms(va.raw(), vn.raw(), vb.raw(), nkv, hd, eps);

                let rope_q_p = b.uni(
                    "g4w-rope-q-p",
                    RopeParams {
                        n_heads: n_q as u32,
                        half_dim: half as u32,
                        total_words: (n_q * half) as u32,
                        table_rows: max_seq as u32,
                    },
                );
                let rgq = b.grid_1d((n_q * half) as u64, 256);
                b.push(
                    pl.rope_f32.clone(),
                    &[
                        (0, qb.raw()),
                        (2, cos.raw()),
                        (3, sin.raw()),
                        (4, rope_pos.raw()),
                        (5, &rope_q_p),
                        (6, q_f32.raw()),
                    ],
                    rgq,
                );
                let rope_k_p = b.uni(
                    "g4w-rope-k-p",
                    RopeParams {
                        n_heads: nkv as u32,
                        half_dim: half as u32,
                        total_words: (nkv * half) as u32,
                        table_rows: max_seq as u32,
                    },
                );
                let rgk = b.grid_1d((nkv * half) as u64, 256);
                b.push(
                    pl.rope.clone(),
                    &[
                        (0, kb.raw()),
                        (1, ka.raw()),
                        (2, cos.raw()),
                        (3, sin.raw()),
                        (4, rope_pos.raw()),
                        (5, &rope_k_p),
                    ],
                    rgk,
                );

                let kvq_p = b.uni(
                    "g4w-kvq-p",
                    KvFp8Params {
                        n_tokens: 1,
                        n_kv: nkv as u32,
                        head_dim: hd as u32,
                        ring: ring_for(kind),
                        pairs: nkv as u32,
                        start: 0,
                        slots: kv_rows_for(kind) as u32,
                        reserved: 0,
                    },
                );
                b.push(
                    pl.kvq.clone(),
                    &[
                        (0, ka.raw()),
                        (1, kv.k_fp8.raw()),
                        (2, kv.k_scales.raw()),
                        (3, kv_start.raw()),
                        (4, &kvq_p),
                    ],
                    (nkv as u32, 1, 1),
                );
                b.push(
                    pl.kvq.clone(),
                    &[
                        (0, vb.raw()),
                        (1, kv.v_fp8.raw()),
                        (2, kv.v_scales.raw()),
                        (3, kv_start.raw()),
                        (4, &kvq_p),
                    ],
                    (nkv as u32, 1, 1),
                );
            }

            let fd = match kind {
                LayerType::SlidingAttention => fd_sliding.raw(),
                LayerType::FullAttention => fd_full.raw(),
            };
            b.push(
                pl.flash1.clone(),
                &[
                    (0, q_f32.raw()),
                    (4, fd),
                    (5, kv.k_fp8.raw()),
                    (6, kv.v_fp8.raw()),
                    (7, scratch.raw()),
                    (8, kv.k_scales.raw()),
                    (9, kv.v_scales.raw()),
                ],
                (n_q as u32, flash_splits(), 1),
            );
            b.push(
                pl.flash2_pk.clone(),
                &[(3, attn_pack.raw()), (4, fd), (7, scratch.raw())],
                (n_q as u32, 1, 1),
            );
            b.gemv(
                &o,
                attn_pack.raw(),
                GemvDst::Packed {
                    y: t1.raw(),
                    word_off: 0,
                },
                Some((xq.raw(), xs_pack.raw())),
            );
            if let Some(fp) = pl.fuse.as_ref().filter(|_| fuse & FUSE_NORM_RES_NORM != 0) {
                let nc_p = b.uni(
                    "g4w-nrn-p",
                    NcParams {
                        hidden: hidden as u32,
                        words: (hidden / 2) as u32,
                        eps,
                        scale: 0.0,
                    },
                );
                b.push(
                    fp.norm_res_norm.clone(),
                    &[
                        (20, t1.raw()),
                        (21, ln_pa.raw()),
                        (22, t0.raw()),
                        (23, x_in),
                        (24, ln_pf.raw()),
                        (26, &nc_p),
                    ],
                    (1, 1, 1),
                );
            } else {
                b.rms(t1.raw(), ln_pa.raw(), t0.raw(), 1, hidden, eps);

                let rmsres_p = b.uni(
                    "g4w-rmsres-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: 1,
                        eps,
                        words_per_row: (hidden / 2) as u32,
                    },
                );
                b.push(
                    pl.rmsres.clone(),
                    &[
                        (0, t0.raw()),
                        (1, x_in),
                        (2, ln_pf.raw()),
                        (3, t1.raw()),
                        (4, &rmsres_p),
                    ],
                    (1, 1, 1),
                );
            }

            if let Some(cap) = w4_cap.as_ref() {
                let p = b.uni(
                    "g4w-w4cap-gu",
                    ScaleParams {
                        n: hidden as u32,
                        n_words: (hidden / 2) as u32,
                        scale: 1.0,
                        cap: 0.0,
                        inv_cap: 0.0,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    },
                );
                let grid = b.grid_1d((hidden / 2) as u64, 256);
                let off = (li * (cap.gu_words + cap.dn_words) * 4) as u64;
                let pipeline = pl.scale.clone();
                b.push_offsets(
                    pipeline,
                    &[(0, t1.raw(), 0), (2, cap.buf.raw(), off), (3, &p, 0)],
                    grid,
                );
            }

            let folded = match (&gate_up, pl.fp8.as_ref()) {
                (
                    GpuProj::Fp8 {
                        w,
                        row_scale,
                        params,
                        grid,
                        fmt: wk::quant_gemv::QFormat::Int8,
                    },
                    Some(fp8),
                ) if gelu_fold_enabled() && wk::quant_gemv::gelu_fold_rule(2 * inter).is_ok() => {
                    b.push(
                        fp8.i8_gelu.clone(),
                        &[
                            (0, w.raw()),
                            (1, row_scale.raw()),
                            (2, t1.raw()),
                            (3, act_pack.raw()),
                            (4, params.raw()),
                        ],
                        *grid,
                    );
                    true
                }
                _ => false,
            };
            if !folded {
                b.gemv(
                    &gate_up,
                    t1.raw(),
                    GemvDst::Packed {
                        y: gu_pack.raw(),
                        word_off: 0,
                    },
                    Some((xq.raw(), xs_pack.raw())),
                );
                let gelu_p = b.uni(
                    "g4w-gelu-p",
                    GeluParams {
                        inter: inter as u32,
                        inter_words: (inter / 2) as u32,
                        rows: 1,
                        tot_pairs: inter as u32,
                    },
                );
                let ggrid = b.grid_1d((inter / 2) as u64, 256);
                b.push(
                    pl.gelu_even.clone(),
                    &[(3, gu_pack.raw()), (4, act_pack.raw()), (5, &gelu_p)],
                    (ggrid.0, 1, 1),
                );
            }
            if let Some(cap) = w4_cap.as_ref() {
                let p = b.uni(
                    "g4w-w4cap-dn",
                    ScaleParams {
                        n: inter as u32,
                        n_words: (inter / 2) as u32,
                        scale: 1.0,
                        cap: 0.0,
                        inv_cap: 0.0,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    },
                );
                let grid = b.grid_1d((inter / 2) as u64, 256);
                let off = (li * (cap.gu_words + cap.dn_words) * 4 + cap.gu_words * 4) as u64;
                let pipeline = pl.scale.clone();
                b.push_offsets(
                    pipeline,
                    &[(0, act_pack.raw(), 0), (2, cap.buf.raw(), off), (3, &p, 0)],
                    grid,
                );
            }
            b.gemv(
                &down,
                act_pack.raw(),
                GemvDst::Packed {
                    y: t0.raw(),
                    word_off: 0,
                },
                Some((xq.raw(), xs_pack.raw())),
            );
            if let Some(fp) = pl.fuse.as_ref().filter(|_| fuse & FUSE_NORM_ADD_NORM != 0) {
                let next_w = if li + 1 < weights.layers.len() {
                    ln_in_all[li + 1].raw()
                } else {
                    final_norm.raw()
                };
                let nc_p = b.uni(
                    "g4w-nan-p",
                    NcParams {
                        hidden: hidden as u32,
                        words: (hidden / 2) as u32,
                        eps,
                        scale: hl.layer_scalar,
                    },
                );
                b.push(
                    fp.norm_add_norm.clone(),
                    &[
                        (20, t0.raw()),
                        (21, ln_po.raw()),
                        (22, t1.raw()),
                        (23, x_in),
                        (24, next_w),
                        (25, x_out),
                        (26, &nc_p),
                    ],
                    (1, 1, 1),
                );
            } else {
                b.rms(t0.raw(), ln_po.raw(), t1.raw(), 1, hidden, eps);

                let res_p = b.uni(
                    "g4w-res-p",
                    ScaleParams {
                        n: hidden as u32,
                        n_words: (hidden / 2) as u32,
                        scale: hl.layer_scalar,
                        cap: 0.0,
                        inv_cap: 0.0,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    },
                );
                let rgrid = b.grid_1d((hidden / 2) as u64, 256);
                b.push(
                    pl.resadd.clone(),
                    &[(0, x_in), (1, t1.raw()), (2, x_out), (3, &res_p)],
                    rgrid,
                );
            }

            layer_gpu.push(LayerGpu {
                kind,
                has_v: hl.has_v,
                layer_scalar: hl.layer_scalar,
                qkv,
                o,
                gate_up,
                down,
                gate_up_coop,
                down_coop,
                ln_pa,
                ln_pf,
                ln_po,
                qn,
                kn,
                vn,
            });

            let t_u = Instant::now();
            ctx.queue.submit(std::iter::empty());
            ctx.poll_blocking().map_err(err)?;
            upload_s += t_u.elapsed().as_secs_f64();
        }
        let ns = |c: &AtomicU64| c.load(Ordering::Relaxed) as f64 / 1e9;
        eprintln!(
            "[gemma4_wgpu] boot phases: pipelines {pipelines_s:.1}s, layers {:.1}s \
             (conv {conv_s:.1}s = ffn-deq {:.1} + ffn-bf16 {:.1} + ffn-q8 {:.1} + attn-q8 {:.1}; \
             upload {upload_s:.1}s), total-so-far {:.1}s",
            t_layers.elapsed().as_secs_f64(),
            ns(&CONV_DEQ_NS),
            ns(&CONV_BF16_NS),
            ns(&CONV_Q8_NS),
            ns(&CONV_ATTN_Q8_NS),
            t_boot.elapsed().as_secs_f64()
        );

        let nvfp4_survived = layer_gpu.iter().any(|l| {
            [&l.qkv, &l.o, &l.gate_up, &l.down]
                .iter()
                .any(|p| matches!(p, GpuProj::Nvfp4 { .. }))
        });
        let nvfp4_mk_unservable = layer_gpu.iter().any(|l| {
            [&l.qkv, &l.o, &l.gate_up, &l.down]
                .iter()
                .any(|p| match p {
                    GpuProj::Nvfp4 { n, k, .. } => {
                        !(k / 16).is_multiple_of(4) || !n.is_multiple_of(2)
                    }
                    _ => false,
                })
        });
        if nvfp4_mk_unservable {
            if pf_m > 0 {
                eprintln!(
                    "[gemma4_wgpu] chunked prefill off: an nvfp4 projection shape (k_blocks % 4 != 0 or odd n) misses the slot-strided M-row quant layout"
                );
                pf_m = 0;
            }
            if bt_slots >= 2 {
                eprintln!(
                    "[gemma4_wgpu] batch decode off: an nvfp4 projection shape (k_blocks % 4 != 0 or odd n) misses the slot-strided M-row quant layout"
                );
                bt_slots = 0;
            }
        }
        if nvfp4_survived && (pf_m > 0 || bt_slots >= 2) {
            let mk4_rows = pf_m.max(bt_slots);
            b.mk4 = Some(Mk4SlotBufs {
                xq: GpuTensor::<u32>::zeroed(ctx, "g4w-mk4-xq", mk4_rows * (quant_k_max / 8)),
                xs: GpuTensor::<u32>::zeroed(
                    ctx,
                    "g4w-mk4-xs",
                    mk4_rows * (quant_k_max / 16).div_ceil(4),
                ),
                xm_i8mapped_for_the_slotshared_gemv: GpuTensor::<u32>::zeroed(
                    ctx,
                    "g4w-mk4-xm",
                    mk4_rows * (quant_k_max / 4),
                ),
                sel_all_zero_because_gemma4_has_no_experts: GpuTensor::upload(
                    ctx,
                    "g4w-mk4-sel",
                    &vec![0u32; mk4_rows],
                ),
                alphas_unread_when_per_expert_alpha_is_zero: GpuTensor::upload(
                    ctx,
                    "g4w-mk4-alphas",
                    &[1.0f32],
                ),
            });
        }
        let pf_flash_requested = pf_m > 0
            && pf_flash_full_attention_enabled_via_nv_g4_wgpu_pf_flash_default_off_pending_nll_adjudication();
        let pf_flash_shape_ok = hd_f.is_multiple_of(4)
            && hd_f <= PF_FLASH_MAX_HEAD_DIM_FDT_KV_STAGE_HOLDS_8_POSITIONS_OF_256_FLOATS;
        if pf_flash_requested && !pf_flash_shape_ok {
            eprintln!(
                "[gemma4_wgpu] NV_G4_WGPU_PF_FLASH=1 ignored: full-attention head_dim {hd_f} misses the tiled kernel's vec4 staging bound of {}",
                PF_FLASH_MAX_HEAD_DIM_FDT_KV_STAGE_HOLDS_8_POSITIONS_OF_256_FLOATS
            );
        }
        let pf_flash_on = pf_flash_requested && pf_flash_shape_ok;
        let pf_fd_flash = pf_flash_on.then(|| {
            let mut f = fd_full_base;
            f.m_rows = pf_m as u32;
            GpuUniform::new(ctx, "g4w-pf-fd-flash", &f)
        });
        let pf_flash_scratch = pf_flash_on.then(|| {
            GpuTensor::<f32>::zeroed(
                ctx,
                "g4w-pf-flash-scratch",
                n_q * pf_m * flash_splits() as usize * (hd_f + 2),
            )
        });
        let pf_flash_pipes = if pf_flash_on {
            let sg = wk::gemv_nvfp4_v2::subgroup32_ok(ctx)
                && !pf_flash_portable_arm_forced_via_nv_g4_wgpu_pf_flash_portable_for_testing_on_sg_adapters();
            let [(src1, label1, entry1), (src2, label2, entry2)] =
                pf_flash_pipeline_specs_stage1_tiled_slotml_arm_matching_the_qwen_nll_signoff_then_stage2_pk_mk(sg);
            eprintln!(
                "[gemma4_wgpu] prefill full-attention flash: tiled slotml {} arm (NV_G4_WGPU_PF_FLASH)",
                if sg { "subgroup" } else { "portable" }
            );
            Some((
                mk_pipeline(ctx, label1, &src1, entry1)?,
                mk_pipeline(ctx, label2, &src2, entry2)?,
            ))
        } else {
            None
        };
        let mut pf_coop_sites = 0usize;
        if let (true, Some(pfb)) = (pf_m > 0, pf_bufs.as_ref()) {
            let m = pf_m;
            b.dst = PassDst::Prefill;
            let pf_coop = if pf_coop_wanted
                && layer_gpu
                    .iter()
                    .any(|l| l.gate_up_coop.is_some() || l.down_coop.is_some())
            {
                let tm = (m / 16) as u32;
                let (tn, sg, ku) = PF_COOP_TN2_SG2_KU4_STAGES_4096_F16_WELL_UNDER_THE_24576_BUDGET;
                let src = wk::gemm_coop_f16::source_wq16(
                    wk::gemm_coop_f16::WqFmt::Nvfp4Block16,
                    tm,
                    tn,
                    sg,
                    ku,
                );
                let entry = wk::gemm_coop_f16::entry_wq16(
                    wk::gemm_coop_f16::WqFmt::Nvfp4Block16,
                    tm,
                    tn,
                    sg,
                    ku,
                );
                let kmax = hidden.max(inter);
                let nmax = (2 * inter).max(hidden);
                if m < 32 {
                    eprintln!(
                        "[gemma4_wgpu] prefill FFN coop w4a16 at m={m} pays a full weight sweep \
                         per 16 rows and measured SLOWER than the i8 mk tiles (72.5 vs 104.8 \
                         tok/s at 2048 on the 31B); set NV_G4_WGPU_PF_M=64 for the measured win"
                    );
                }
                eprintln!(
                    "[gemma4_wgpu] prefill FFN coop w4a16 arm on (NV_G4_PF_COOP): m={m} tm={tm} tn={tn} sg={sg} ku={ku}; \
                     covered layers={} of {}",
                    layer_gpu
                        .iter()
                        .filter(|l| l.gate_up_coop.is_some() && l.down_coop.is_some())
                        .count(),
                    layer_gpu.len()
                );
                Some(PfCoopCtx {
                    gemm: mk_pipeline(ctx, "g4w-pf-coop-w4a16", &src, &entry)?,
                    x16: mk_pipeline(ctx, "g4w-pf-coop-x16", PF_COOP_X16_WGSL, "g4w_pf_bf16pairs_to_f16")?,
                    pack: mk_pipeline(
                        ctx,
                        "g4w-pf-coop-pack",
                        PF_COOP_PACK_WGSL,
                        "g4w_pf_coop_alpha_pack_bf16",
                    )?,
                    xf16: GpuTensor::<u32>::zeroed(ctx, "g4w-pf-coop-xf16", m * kmax / 2),
                    yf32: GpuTensor::<f32>::zeroed(ctx, "g4w-pf-coop-yf32", m * nmax),
                    zero: GpuTensor::<f32>::zeroed(ctx, "g4w-pf-coop-zero", 256),
                    tm,
                    tn,
                    sg,
                    ku,
                })
            } else {
                None
            };
            if pf_coop.is_some() {
                pf_coop_sites = layer_gpu
                    .iter()
                    .map(|l| {
                        usize::from(l.gate_up_coop.is_some()) + usize::from(l.down_coop.is_some())
                    })
                    .sum();
            }
            let gather_mk = pl.mk.as_ref().expect("mk pipelines").gather.clone();
            b.push(
                gather_mk,
                &[
                    (3, embed_lo.raw()),
                    (4, embed_hi.raw()),
                    (5, pf_tok_idx.raw()),
                    (6, pfb.t0.raw()),
                    (7, &gather_p),
                ],
                (m as u32, 1, 1),
            );
            let pf_scale_p = b.uni(
                "g4w-pf-embed-scale-p",
                ScaleParams {
                    n: (m * hidden) as u32,
                    n_words: (m * hidden / 2) as u32,
                    scale: embed_scale,
                    cap: 0.0,
                    inv_cap: 0.0,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                },
            );
            let psgrid = b.grid_1d((m * hidden / 2) as u64, 256);
            let scale_pl = pl.scale.clone();
            b.push(
                scale_pl,
                &[(0, pfb.t0.raw()), (2, pfb.hid_a.raw()), (3, &pf_scale_p)],
                psgrid,
            );
            let pf_splice_p = b.uni(
                "g4w-pf-splice-p",
                SpliceParams {
                    hidden_words: (hidden / 2) as u32,
                    m: m as u32,
                    pad0: 0,
                    pad1: 0,
                },
            );
            let splice_pl = pl.splice.clone();
            b.push(
                splice_pl,
                &[
                    (120, pfb.splice_rows.raw()),
                    (121, pfb.splice_mask.raw()),
                    (122, pfb.hid_a.raw()),
                    (123, &pf_splice_p),
                ],
                ((hidden / 2).div_ceil(256) as u32, m as u32, 1),
            );

            for (li, lg) in layer_gpu.iter().enumerate() {
                let kind = lg.kind;
                let hd = config.head_dim_for(kind);
                let nkv = config.num_kv_heads_for(kind);
                let q_dim = n_q * hd;
                let kv_dim = nkv * hd;
                let half = hd / 2;
                let kv = &kv_layers[li];
                let kidx = match kind {
                    LayerType::SlidingAttention => 0usize,
                    LayerType::FullAttention => 1,
                };
                let (cos, sin) = (&rope_bufs[kidx].0, &rope_bufs[kidx].1);
                let (x_in, x_out) = if li % 2 == 0 {
                    (pfb.hid_a.raw(), pfb.hid_b.raw())
                } else {
                    (pfb.hid_b.raw(), pfb.hid_a.raw())
                };

                b.rms(x_in, ln_in_all[li].raw(), pfb.t0.raw(), m, hidden, eps);
                let v_off = if lg.has_v { q_dim + kv_dim } else { q_dim };
                b.gemv_mk(
                    &lg.qkv,
                    pfb.t0.raw(),
                    m,
                    hidden / 2,
                    GemvDstMk::SplitQkv {
                        q: pfb.qa.raw(),
                        k: pfb.ka.raw(),
                        v: pfb.va.raw(),
                        q_rows: q_dim,
                        kv_rows: kv_dim,
                        v_off,
                    },
                );
                b.rms(pfb.qa.raw(), lg.qn.raw(), pfb.qb.raw(), m * n_q, hd, eps);
                b.rms(pfb.ka.raw(), lg.kn.raw(), pfb.kb.raw(), m * nkv, hd, eps);
                b.rms(pfb.va.raw(), lg.vn.raw(), pfb.vb.raw(), m * nkv, hd, eps);

                let rope_q_p = b.uni(
                    "g4w-pf-rope-q-p",
                    RopeParams {
                        n_heads: n_q as u32,
                        half_dim: half as u32,
                        total_words: (m * n_q * half) as u32,
                        table_rows: max_seq as u32,
                    },
                );
                let rgq = b.grid_1d((m * n_q * half) as u64, 256);
                let rope_f32_pl = pl.rope_f32.clone();
                b.push(
                    rope_f32_pl,
                    &[
                        (0, pfb.qb.raw()),
                        (2, cos.raw()),
                        (3, sin.raw()),
                        (4, pf_rope_pos.raw()),
                        (5, &rope_q_p),
                        (6, pfb.q_f32.raw()),
                    ],
                    rgq,
                );
                let rope_k_p = b.uni(
                    "g4w-pf-rope-k-p",
                    RopeParams {
                        n_heads: nkv as u32,
                        half_dim: half as u32,
                        total_words: (m * nkv * half) as u32,
                        table_rows: max_seq as u32,
                    },
                );
                let rgk = b.grid_1d((m * nkv * half) as u64, 256);
                let rope_pl = pl.rope.clone();
                b.push(
                    rope_pl,
                    &[
                        (0, pfb.kb.raw()),
                        (1, pfb.ka.raw()),
                        (2, cos.raw()),
                        (3, sin.raw()),
                        (4, pf_rope_pos.raw()),
                        (5, &rope_k_p),
                    ],
                    rgk,
                );
                let kvq_p = b.uni(
                    "g4w-pf-kvq-p",
                    KvFp8Params {
                        n_tokens: m as u32,
                        n_kv: nkv as u32,
                        head_dim: hd as u32,
                        ring: ring_for(kind),
                        pairs: (m * nkv) as u32,
                        start: 0,
                        slots: kv_rows_for(kind) as u32,
                        reserved: 0,
                    },
                );
                let kvq_pl = pl.kvq.clone();
                b.push(
                    kvq_pl.clone(),
                    &[
                        (0, pfb.ka.raw()),
                        (1, kv.k_fp8.raw()),
                        (2, kv.k_scales.raw()),
                        (3, pf_kv_start.raw()),
                        (4, &kvq_p),
                    ],
                    ((m * nkv) as u32, 1, 1),
                );
                b.push(
                    kvq_pl,
                    &[
                        (0, pfb.vb.raw()),
                        (1, kv.v_fp8.raw()),
                        (2, kv.v_scales.raw()),
                        (3, pf_kv_start.raw()),
                        (4, &kvq_p),
                    ],
                    ((m * nkv) as u32, 1, 1),
                );

                let flash_arm = match kind {
                    LayerType::FullAttention => pf_flash_pipes
                        .as_ref()
                        .zip(pf_fd_flash.as_ref())
                        .zip(pf_flash_scratch.as_ref()),
                    LayerType::SlidingAttention => None,
                };
                match flash_arm {
                    Some((((f1, f2), fdu), scr)) => {
                        let tiles =
                            m.div_ceil(PF_FLASH_TILE_ROWS_BAKED_AS_FDT_ROWS_32_IN_THE_SHARED_TILED_KERNEL);
                        b.push(
                            f1.clone(),
                            &[
                                (0, pfb.q_f32.raw()),
                                (4, fdu.raw()),
                                (5, kv.k_fp8.raw()),
                                (6, kv.v_fp8.raw()),
                                (7, scr.raw()),
                                (8, kv.k_scales.raw()),
                                (9, kv.v_scales.raw()),
                            ],
                            (n_q as u32, flash_splits(), tiles as u32),
                        );
                        b.push(
                            f2.clone(),
                            &[(3, pfb.attn_pack.raw()), (4, fdu.raw()), (7, scr.raw())],
                            (n_q as u32, m as u32, 1),
                        );
                    }
                    None => {
                        for t in 0..m {
                            let fd = match kind {
                                LayerType::SlidingAttention => pf_fd_s[t].raw(),
                                LayerType::FullAttention => pf_fd_f[t].raw(),
                            };
                            let flash1_pl = pl.flash1.clone();
                            b.push_offsets(
                                flash1_pl,
                                &[
                                    (0, pfb.q_f32.raw(), (t * q_dim * 4) as u64),
                                    (4, fd, 0),
                                    (5, kv.k_fp8.raw(), 0),
                                    (6, kv.v_fp8.raw(), 0),
                                    (7, scratch.raw(), 0),
                                    (8, kv.k_scales.raw(), 0),
                                    (9, kv.v_scales.raw(), 0),
                                ],
                                (n_q as u32, flash_splits(), 1),
                            );
                            let flash2_pl = pl.flash2_pk.clone();
                            b.push_offsets(
                                flash2_pl,
                                &[
                                    (3, pfb.attn_pack.raw(), (t * (q_dim / 2) * 4) as u64),
                                    (4, fd, 0),
                                    (7, scratch.raw(), 0),
                                ],
                                (n_q as u32, 1, 1),
                            );
                        }
                    }
                }

                b.gemv_mk(
                    &lg.o,
                    pfb.attn_pack.raw(),
                    m,
                    q_dim / 2,
                    GemvDstMk::Packed {
                        y: pfb.t1.raw(),
                        word_off: 0,
                        y_stride_words: hidden / 2,
                    },
                );
                b.rms(pfb.t1.raw(), lg.ln_pa.raw(), pfb.t0.raw(), m, hidden, eps);
                let rmsres_p = b.uni(
                    "g4w-pf-rmsres-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: m as u32,
                        eps,
                        words_per_row: (hidden / 2) as u32,
                    },
                );
                let rmsres_pl = pl.rmsres.clone();
                b.push(
                    rmsres_pl,
                    &[
                        (0, pfb.t0.raw()),
                        (1, x_in),
                        (2, lg.ln_pf.raw()),
                        (3, pfb.t1.raw()),
                        (4, &rmsres_p),
                    ],
                    (m as u32, 1, 1),
                );
                match (&pf_coop, &lg.gate_up_coop) {
                    (Some(cc), Some(cl)) => {
                        b.pf_coop_w4a16(cc, cl, pfb.t1.raw(), m, hidden / 2, pfb.gu_pack.raw(), 0, inter)
                    }
                    _ => b.gemv_mk(
                        &lg.gate_up,
                        pfb.t1.raw(),
                        m,
                        hidden / 2,
                        GemvDstMk::Packed {
                            y: pfb.gu_pack.raw(),
                            word_off: 0,
                            y_stride_words: inter,
                        },
                    ),
                }
                let gelu_p = b.uni(
                    "g4w-pf-gelu-p",
                    GeluParams {
                        inter: inter as u32,
                        inter_words: (inter / 2) as u32,
                        rows: m as u32,
                        tot_pairs: (m * inter) as u32,
                    },
                );
                let ggrid = b.grid_1d((inter / 2) as u64, 256);
                let gelu_pl = pl.gelu_even.clone();
                b.push(
                    gelu_pl,
                    &[
                        (3, pfb.gu_pack.raw()),
                        (4, pfb.act_pack.raw()),
                        (5, &gelu_p),
                    ],
                    (ggrid.0, m as u32, 1),
                );
                match (&pf_coop, &lg.down_coop) {
                    (Some(cc), Some(cl)) => {
                        b.pf_coop_w4a16(cc, cl, pfb.act_pack.raw(), m, inter / 2, pfb.t0.raw(), 0, hidden / 2)
                    }
                    _ => b.gemv_mk(
                        &lg.down,
                        pfb.act_pack.raw(),
                        m,
                        inter / 2,
                        GemvDstMk::Packed {
                            y: pfb.t0.raw(),
                            word_off: 0,
                            y_stride_words: hidden / 2,
                        },
                    ),
                }
                b.rms(pfb.t0.raw(), lg.ln_po.raw(), pfb.t1.raw(), m, hidden, eps);
                let res_p = b.uni(
                    "g4w-pf-res-p",
                    ScaleParams {
                        n: (m * hidden) as u32,
                        n_words: (m * hidden / 2) as u32,
                        scale: lg.layer_scalar,
                        cap: 0.0,
                        inv_cap: 0.0,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    },
                );
                let rgrid = b.grid_1d((m * hidden / 2) as u64, 256);
                let resadd_pl = pl.resadd.clone();
                b.push(
                    resadd_pl,
                    &[(0, x_in), (1, pfb.t1.raw()), (2, x_out), (3, &res_p)],
                    rgrid,
                );
            }
            b.dst = PassDst::Decode;
        }

        if let (true, Some(btb)) = (bt_slots >= 2, bt_bufs.as_ref()) {
            let s = bt_slots;
            let bt = &btb.core;
            b.dst = PassDst::Batch;
            let gather_mk = pl.mk.as_ref().expect("mk pipelines").gather.clone();
            b.push(
                gather_mk,
                &[
                    (3, embed_lo.raw()),
                    (4, embed_hi.raw()),
                    (5, bt_tok_idx.raw()),
                    (6, bt.t0.raw()),
                    (7, &gather_p),
                ],
                (s as u32, 1, 1),
            );
            let bt_scale_p = b.uni(
                "g4w-bt-embed-scale-p",
                ScaleParams {
                    n: (s * hidden) as u32,
                    n_words: (s * hidden / 2) as u32,
                    scale: embed_scale,
                    cap: 0.0,
                    inv_cap: 0.0,
                    pad0: 0,
                    pad1: 0,
                    pad2: 0,
                },
            );
            let bsgrid = b.grid_1d((s * hidden / 2) as u64, 256);
            let scale_pl = pl.scale.clone();
            b.push(
                scale_pl,
                &[(0, bt.t0.raw()), (2, bt.hid_a.raw()), (3, &bt_scale_p)],
                bsgrid,
            );

            for (li, lg) in layer_gpu.iter().enumerate() {
                let kind = lg.kind;
                let hd = config.head_dim_for(kind);
                let nkv = config.num_kv_heads_for(kind);
                let q_dim = n_q * hd;
                let kv_dim = nkv * hd;
                let half = hd / 2;
                let kv = &kv_layers[li];
                let kidx = match kind {
                    LayerType::SlidingAttention => 0usize,
                    LayerType::FullAttention => 1,
                };
                let (cos, sin) = (&rope_bufs[kidx].0, &rope_bufs[kidx].1);
                let (x_in, x_out) = if li % 2 == 0 {
                    (bt.hid_a.raw(), bt.hid_b.raw())
                } else {
                    (bt.hid_b.raw(), bt.hid_a.raw())
                };

                b.rms(x_in, ln_in_all[li].raw(), bt.t0.raw(), s, hidden, eps);
                let v_off = if lg.has_v { q_dim + kv_dim } else { q_dim };
                b.gemv_mk(
                    &lg.qkv,
                    bt.t0.raw(),
                    s,
                    hidden / 2,
                    GemvDstMk::SplitQkv {
                        q: bt.qa.raw(),
                        k: bt.ka.raw(),
                        v: bt.va.raw(),
                        q_rows: q_dim,
                        kv_rows: kv_dim,
                        v_off,
                    },
                );
                b.rms(bt.qa.raw(), lg.qn.raw(), bt.qb.raw(), s * n_q, hd, eps);
                b.rms(bt.ka.raw(), lg.kn.raw(), bt.kb.raw(), s * nkv, hd, eps);
                b.rms(bt.va.raw(), lg.vn.raw(), bt.vb.raw(), s * nkv, hd, eps);

                let rope_q_p = b.uni(
                    "g4w-bt-rope-q-p",
                    RopeParams {
                        n_heads: n_q as u32,
                        half_dim: half as u32,
                        total_words: (s * n_q * half) as u32,
                        table_rows: max_seq as u32,
                    },
                );
                let rgq = b.grid_1d((s * n_q * half) as u64, 256);
                let rope_f32_pl = pl.rope_f32.clone();
                b.push(
                    rope_f32_pl,
                    &[
                        (0, bt.qb.raw()),
                        (2, cos.raw()),
                        (3, sin.raw()),
                        (4, bt_rope_pos.raw()),
                        (5, &rope_q_p),
                        (6, bt.q_f32.raw()),
                    ],
                    rgq,
                );
                let rope_k_p = b.uni(
                    "g4w-bt-rope-k-p",
                    RopeParams {
                        n_heads: nkv as u32,
                        half_dim: half as u32,
                        total_words: (s * nkv * half) as u32,
                        table_rows: max_seq as u32,
                    },
                );
                let rgk = b.grid_1d((s * nkv * half) as u64, 256);
                let rope_pl = pl.rope.clone();
                b.push(
                    rope_pl,
                    &[
                        (0, bt.kb.raw()),
                        (1, bt.ka.raw()),
                        (2, cos.raw()),
                        (3, sin.raw()),
                        (4, bt_rope_pos.raw()),
                        (5, &rope_k_p),
                    ],
                    rgk,
                );

                let kvq_p = b.uni(
                    "g4w-bt-kvq-p",
                    KvFp8Params {
                        n_tokens: 1,
                        n_kv: nkv as u32,
                        head_dim: hd as u32,
                        ring: 0,
                        pairs: nkv as u32,
                        start: 0,
                        slots: (max_seq * kv_slots) as u32,
                        reserved: 0,
                    },
                );
                for j in 0..s {
                    let row_off = (j * kv_dim / 2 * 4) as u64;
                    let kvq_pl = pl.kvq.clone();
                    b.push_offsets(
                        kvq_pl,
                        &[
                            (0, bt.ka.raw(), row_off),
                            (1, kv.k_fp8.raw(), 0),
                            (2, kv.k_scales.raw(), 0),
                            (3, bt_kv_start[j].raw(), 0),
                            (4, &kvq_p, 0),
                        ],
                        (nkv as u32, 1, 1),
                    );
                    let kvq_pl = pl.kvq.clone();
                    b.push_offsets(
                        kvq_pl,
                        &[
                            (0, bt.vb.raw(), row_off),
                            (1, kv.v_fp8.raw(), 0),
                            (2, kv.v_scales.raw(), 0),
                            (3, bt_kv_start[j].raw(), 0),
                            (4, &kvq_p, 0),
                        ],
                        (nkv as u32, 1, 1),
                    );
                }

                for j in 0..s {
                    let fd = match kind {
                        LayerType::SlidingAttention => bt_fd_s[j].raw(),
                        LayerType::FullAttention => bt_fd_f[j].raw(),
                    };
                    let flash1_pl = pl.flash1.clone();
                    b.push_offsets(
                        flash1_pl,
                        &[
                            (0, bt.q_f32.raw(), (j * q_dim * 4) as u64),
                            (4, fd, 0),
                            (5, kv.k_fp8.raw(), 0),
                            (6, kv.v_fp8.raw(), 0),
                            (7, scratch.raw(), 0),
                            (8, kv.k_scales.raw(), 0),
                            (9, kv.v_scales.raw(), 0),
                        ],
                        (n_q as u32, flash_splits(), 1),
                    );
                    let flash2_pl = pl.flash2_pk.clone();
                    b.push_offsets(
                        flash2_pl,
                        &[
                            (3, bt.attn_pack.raw(), (j * (q_dim / 2) * 4) as u64),
                            (4, fd, 0),
                            (7, scratch.raw(), 0),
                        ],
                        (n_q as u32, 1, 1),
                    );
                }

                b.gemv_mk(
                    &lg.o,
                    bt.attn_pack.raw(),
                    s,
                    q_dim / 2,
                    GemvDstMk::Packed {
                        y: bt.t1.raw(),
                        word_off: 0,
                        y_stride_words: hidden / 2,
                    },
                );
                b.rms(bt.t1.raw(), lg.ln_pa.raw(), bt.t0.raw(), s, hidden, eps);
                let rmsres_p = b.uni(
                    "g4w-bt-rmsres-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: s as u32,
                        eps,
                        words_per_row: (hidden / 2) as u32,
                    },
                );
                let rmsres_pl = pl.rmsres.clone();
                b.push(
                    rmsres_pl,
                    &[
                        (0, bt.t0.raw()),
                        (1, x_in),
                        (2, lg.ln_pf.raw()),
                        (3, bt.t1.raw()),
                        (4, &rmsres_p),
                    ],
                    (s as u32, 1, 1),
                );
                b.gemv_mk(
                    &lg.gate_up,
                    bt.t1.raw(),
                    s,
                    hidden / 2,
                    GemvDstMk::Packed {
                        y: bt.gu_pack.raw(),
                        word_off: 0,
                        y_stride_words: inter,
                    },
                );
                let gelu_p = b.uni(
                    "g4w-bt-gelu-p",
                    GeluParams {
                        inter: inter as u32,
                        inter_words: (inter / 2) as u32,
                        rows: s as u32,
                        tot_pairs: (s * inter) as u32,
                    },
                );
                let ggrid = b.grid_1d((inter / 2) as u64, 256);
                let gelu_pl = pl.gelu_even.clone();
                b.push(
                    gelu_pl,
                    &[(3, bt.gu_pack.raw()), (4, bt.act_pack.raw()), (5, &gelu_p)],
                    (ggrid.0, s as u32, 1),
                );
                b.gemv_mk(
                    &lg.down,
                    bt.act_pack.raw(),
                    s,
                    inter / 2,
                    GemvDstMk::Packed {
                        y: bt.t0.raw(),
                        word_off: 0,
                        y_stride_words: hidden / 2,
                    },
                );
                b.rms(bt.t0.raw(), lg.ln_po.raw(), bt.t1.raw(), s, hidden, eps);
                let res_p = b.uni(
                    "g4w-bt-res-p",
                    ScaleParams {
                        n: (s * hidden) as u32,
                        n_words: (s * hidden / 2) as u32,
                        scale: lg.layer_scalar,
                        cap: 0.0,
                        inv_cap: 0.0,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    },
                );
                let rgrid = b.grid_1d((s * hidden / 2) as u64, 256);
                let resadd_pl = pl.resadd.clone();
                b.push(
                    resadd_pl,
                    &[(0, x_in), (1, bt.t1.raw()), (2, x_out), (3, &res_p)],
                    rgrid,
                );
            }

            let bt_final = if config.num_hidden_layers.is_multiple_of(2) {
                bt.hid_a.raw()
            } else {
                bt.hid_b.raw()
            };
            push_mrow_head(
                &mut b,
                ctx,
                &pl,
                &config,
                s,
                &MrowHeadBufs {
                    hidden_in: bt_final,
                    normed: bt.t0.raw(),
                    logits_pk: btb.logits_pk.raw(),
                    logits_f32: bt_logits_f32.raw(),
                    am_val: btb.am_val.raw(),
                    am_idx: btb.am_idx.raw(),
                    token_out: bt_token_out.raw(),
                },
                &MrowHeadWeights {
                    final_norm: &final_norm,
                    embed_lo: &embed_lo,
                    embed_hi: &embed_hi,
                    i8: &lmhead_i8_bufs,
                },
            )?;
            b.dst = PassDst::Decode;
        }

        let verify = match pf_bufs.as_ref() {
            Some(pfb) => {
                let rows = pf_m.min(VERIFY_ROWS_MAX_IS_THE_LONGEST_CHAIN_THE_SPEC_LOOP_SUBMITS);
                let logits_pk = GpuTensor::<u32>::zeroed(ctx, "g4w-vf-logits-pk", rows * vocab / 2);
                let logits_f32 = GpuTensor::<f32>::zeroed(ctx, "g4w-vf-logits-f32", rows * vocab);
                let am_val = GpuTensor::<f32>::zeroed(
                    ctx,
                    "g4w-vf-am-val",
                    rows * wk::graph_decode::ARGMAX_BLOCKS,
                );
                let am_idx = GpuTensor::<i32>::zeroed(
                    ctx,
                    "g4w-vf-am-idx",
                    rows * wk::graph_decode::ARGMAX_BLOCKS,
                );
                let token_out = GpuTensor::<u32>::zeroed(ctx, "g4w-vf-token-out", rows);
                b.dst = PassDst::Verify;
                let pf_final = if config.num_hidden_layers.is_multiple_of(2) {
                    pfb.hid_a.raw()
                } else {
                    pfb.hid_b.raw()
                };
                push_mrow_head(
                    &mut b,
                    ctx,
                    &pl,
                    &config,
                    rows,
                    &MrowHeadBufs {
                        hidden_in: pf_final,
                        normed: pfb.t0.raw(),
                        logits_pk: logits_pk.raw(),
                        logits_f32: logits_f32.raw(),
                        am_val: am_val.raw(),
                        am_idx: am_idx.raw(),
                        token_out: token_out.raw(),
                    },
                    &MrowHeadWeights {
                        final_norm: &final_norm,
                        embed_lo: &embed_lo,
                        embed_hi: &embed_hi,
                        i8: &lmhead_i8_bufs,
                    },
                )?;
                b.dst = PassDst::Decode;
                b.keep.push(Box::new((am_val, am_idx, logits_pk)));
                Some(VerifyState {
                    rows,
                    passes: Vec::new(),
                    logits_f32,
                    token_out,
                    validated: false,
                })
            }
            None => None,
        };
        b.keep.push(Box::new(layer_gpu));

        let head_start = b.passes.len();
        let final_hid = if config.num_hidden_layers.is_multiple_of(2) {
            hid_a.raw()
        } else {
            hid_b.raw()
        };
        if fuse & FUSE_NORM_ADD_NORM == 0 {
            b.rms(final_hid, final_norm.raw(), t0.raw(), 1, hidden, eps);
        } else {
            let _ = final_hid;
        }
        if let Some((wq_lo, wq_hi, rs_lo, rs_hi, wn_ones, rstd_one)) = &lmhead_i8_bufs {
            let lm_grid = b.grid_1d(split_row as u64, wk::gemv_bf16::ROWS_PER_GROUP);
            let lm_p = b.uni(
                "g4w-lm-i8-p",
                GemvI8Params {
                    n_rows: split_row as u32,
                    k_elems: hidden as u32,
                    wq_row_words: (hidden / 4) as u32,
                    groups_x: lm_grid.0,
                    m_rows: 1,
                    x_row_words: (hidden / 2) as u32,
                    pad0: 0,
                    pad1: 0,
                },
            );
            let (logits_un_lo, logits_un_hi) = logits_un.as_ref().unwrap();
            for (wq, rs, half_y) in [(wq_lo, rs_lo, logits_un_lo), (wq_hi, rs_hi, logits_un_hi)] {
                b.push(
                    pl.gemv_i8.clone(),
                    &[
                        (13, wq.raw()),
                        (14, rs.raw()),
                        (15, t0.raw()),
                        (16, wn_ones.raw()),
                        (17, rstd_one.raw()),
                        (18, half_y.raw()),
                        (19, &lm_p),
                    ],
                    lm_grid,
                );
            }
            b.pack16(logits_un_lo.raw(), logits_pk.raw(), 0, 0, split_row);
            b.pack16(
                logits_un_hi.raw(),
                logits_pk.raw(),
                0,
                split_row / 2,
                split_row,
            );
        } else {
            let lm_grid = b.grid_1d(split_row as u64, wk::gemv_bf16::ROWS_PER_GROUP);
            let lm_p = b.uni(
                "g4w-lm-p",
                GemvBf16Params {
                    n_rows: split_row as u32,
                    k_elems: hidden as u32,
                    w_row_words: (hidden / 2) as u32,
                    groups_x: lm_grid.0,
                },
            );
            for (half_w, word_off) in [(&embed_lo, 0usize), (&embed_hi, split_row / 2)] {
                let off = b.uni(
                    "g4w-lm-pk-off",
                    PkOffParams {
                        dst_word_off: word_off as u32,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    },
                );
                b.push(
                    pl.gemv_pk.clone(),
                    &[
                        (0, half_w.raw()),
                        (1, t0.raw()),
                        (2, logits_pk.raw()),
                        (3, &lm_p),
                        (30, &off),
                    ],
                    lm_grid,
                );
            }
        }
        let cap = config.final_logit_softcapping;
        let softcap_on = cap > 0.0 && cap.is_finite();
        let cap_p = b.uni(
            "g4w-cap-p",
            ScaleParams {
                n: vocab as u32,
                n_words: (vocab / 2) as u32,
                scale: 0.0,
                cap,
                inv_cap: if softcap_on { 1.0 / cap } else { 0.0 },
                pad0: 0,
                pad1: 0,
                pad2: 0,
            },
        );
        let capgrid = b.grid_1d((vocab / 2) as u64, 256);
        let cap_pl = if softcap_on {
            &pl.softcap
        } else {
            &pl.cast_f32
        };
        b.push(
            cap_pl.clone(),
            &[(0, logits_pk.raw()), (3, &cap_p), (4, logits_f32.raw())],
            capgrid,
        );
        let am_p = b.uni(
            "g4w-am-p",
            ArgmaxRowsParams {
                rows: 1,
                n: vocab as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        b.push(
            pl.am1.clone(),
            &[
                (54, logits_f32.raw()),
                (55, am_val.raw()),
                (56, am_idx.raw()),
                (58, &am_p),
            ],
            (wk::graph_decode::ARGMAX_BLOCKS as u32, 1, 1),
        );
        b.push(
            pl.am2.clone(),
            &[
                (55, am_val.raw()),
                (56, am_idx.raw()),
                (57, token_out.raw()),
                (58, &am_p),
            ],
            (1, 1, 1),
        );

        b.keep.push(Box::new((embed_lo, embed_hi, final_norm)));
        b.keep.push(Box::new(ln_in_all));
        if let Some(bufs) = lmhead_i8_bufs {
            b.keep.push(Box::new(bufs));
        }
        b.keep
            .push(Box::new((hid_a, hid_b, t0, t1, qa, qb, ka, kb, va, vb)));
        b.keep
            .push(Box::new((q_f32, attn_pack, gu_pack, act_pack, xq, xs_pack)));
        b.keep
            .push(Box::new((logits_un, logits_pk, am_val, am_idx, scratch)));
        b.keep.push(Box::new(rope_bufs));
        let kv_handles: Vec<GpuLayerKv> = kv_layers
            .iter()
            .map(|kv| GpuLayerKv {
                k_fp8: kv.k_fp8.clone(),
                v_fp8: kv.v_fp8.clone(),
                k_scales: kv.k_scales.clone(),
                v_scales: kv.v_scales.clone(),
            })
            .collect();
        b.keep.push(Box::new(kv_layers));

        let pf_splice = pf_bufs
            .as_ref()
            .map(|pfb| (pfb.splice_rows.raw().clone(), pfb.splice_mask.raw().clone()));
        b.keep.push(Box::new(pf_bufs));
        b.keep.push(Box::new(pf_flash_scratch));
        b.keep.push(Box::new(bt_bufs));
        let mk4_kept_alive_with_the_graph = b.mk4.take();
        b.keep.push(Box::new(mk4_kept_alive_with_the_graph));
        let Builder {
            passes,
            pf_passes,
            bt_passes,
            vf_passes,
            keep,
            weight_bytes,
            nvfp4_projs,
            nvfp4_v2_routed,
            ..
        } = b;
        let verify = match (verify, prefill_is_live(pf_m, &pf_passes), vf_passes) {
            (Some(v), true, vp) if !vp.is_empty() => Some(VerifyState { passes: vp, ..v }),
            _ => None,
        };
        let prefill = match (pf_splice, prefill_is_live(pf_m, &pf_passes)) {
            (Some((splice_rows, splice_mask)), true) => Some(PrefillState {
                m: pf_m,
                passes: pf_passes,
                tok_idx: pf_tok_idx,
                rope_pos: pf_rope_pos,
                kv_start: pf_kv_start,
                fd_s: pf_fd_s,
                fd_f: pf_fd_f,
                fd_flash: pf_fd_flash,
                splice_rows,
                splice_mask,
                splice_mask_live: false,
                validated: false,
            }),
            _ => None,
        };
        let batch = (bt_slots >= 2 && !bt_passes.is_empty()).then_some(BatchState {
            slots: bt_slots,
            passes: bt_passes,
            tok_idx: bt_tok_idx,
            rope_pos: bt_rope_pos,
            kv_start: bt_kv_start,
            fd_s: bt_fd_s,
            fd_f: bt_fd_f,
            token_out: bt_token_out,
            logits_f32: bt_logits_f32,
            validated: false,
        });
        eprintln!(
            "[gemma4_wgpu] chunked prefill: {}",
            match &prefill {
                Some(p) => format!("m={} rows, {} passes/chunk", p.m, p.passes.len()),
                None => "off (one decode graph per prompt token)".to_string(),
            }
        );
        eprintln!(
            "[gemma4_wgpu] batch decode: {}",
            match &batch {
                Some(p) => format!(
                    "{} slots x {max_seq} kv rows, {} passes/step",
                    p.slots,
                    p.passes.len()
                ),
                None => "off (single-stream kv cache)".to_string(),
            }
        );
        if let Some(line) = nvfp4_v2_boot_line(nvfp4_v2_enabled(ctx), nvfp4_v2_routed, nvfp4_projs)
        {
            eprintln!("{line}");
        }
        Ok(Self {
            ctx,
            config,
            max_seq,
            pos: 0,
            kv_base: 0,
            slot_pos: vec![0; kv_slots],
            validated: false,
            prefill,
            pf_coop_sites,
            verify,
            batch,
            weight_bytes,
            nvfp4_v2: (nvfp4_v2_routed, nvfp4_projs),
            passes,
            head_start,
            prefix_validated: false,
            tok_idx,
            rope_pos,
            kv_start,
            fd_sliding,
            fd_full,
            fd_sliding_base,
            fd_full_base,
            token_out,
            chain_out,
            logits_f32,
            vocab,
            preenc: preenc_enabled_default(),
            pending_cb: None,
            preenc_hits: 0,
            chain_steps: 0,
            uniform_probe: uniform_probe_reps(),
            w4_cap,
            kv_layers: kv_handles,
            _keep: keep,
        })
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    pub fn w4_capture_live(&self) -> bool {
        self.w4_cap.is_some()
    }

    pub fn w4_collect(&self, calib: &mut W4Calib) -> Result<()> {
        let cap = self
            .w4_cap
            .as_ref()
            .context("w4_collect: NV_G4_W4_CAPTURE was not set when this graph was built")?;
        let words = cap.buf.download(self.ctx).map_err(err)?;
        let hidden = self.config.hidden_size;
        let inter = self.config.intermediate_size;
        anyhow::ensure!(
            calib.gate_up.len() == cap.layers && calib.down.len() == cap.layers,
            "w4_collect: calibration has {} layers, graph has {}",
            calib.gate_up.len(),
            cap.layers
        );
        let unpack = |src: &[u32], n: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(n);
            for w in src.iter().take(n / 2) {
                out.push(f32::from_bits((*w & 0xffff) << 16));
                out.push(f32::from_bits(*w & 0xffff_0000));
            }
            out
        };
        let stride = cap.gu_words + cap.dn_words;
        for li in 0..cap.layers {
            let gu = unpack(&words[li * stride..li * stride + cap.gu_words], hidden);
            let dn = unpack(
                &words[li * stride + cap.gu_words..li * stride + stride],
                inter,
            );
            calib.gate_up[li].observe(&gu);
            calib.down[li].observe(&dn);
        }
        calib.tokens += 1;
        Ok(())
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn weight_bytes_per_token(&self) -> u64 {
        self.weight_bytes
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn head_start(&self) -> usize {
        self.head_start
    }

    pub fn nvfp4_v2_projections(&self) -> (usize, usize) {
        self.nvfp4_v2
    }

    pub fn reset(&mut self) {
        self.pos = 0;
        self.sync_active();
    }

    pub fn truncate_to(&mut self, pos: usize) -> Result<()> {
        anyhow::ensure!(pos <= self.pos, "truncate_to {pos} beyond pos {}", self.pos);
        self.pos = pos;
        self.sync_active();
        Ok(())
    }

    pub fn kv_layer_count(&self) -> usize {
        self.kv_layers.len()
    }

    pub fn kv_layer_lens(&self, li: usize) -> Option<[usize; 4]> {
        let kv = self.kv_layers.get(li)?;
        Some([
            kv.k_fp8.len(),
            kv.v_fp8.len(),
            kv.k_scales.len(),
            kv.v_scales.len(),
        ])
    }

    pub fn kv_cache_snapshot(&self, li: usize) -> Result<Option<KvCacheSnapshot>> {
        let Some(kv) = self.kv_layers.get(li) else {
            return Ok(None);
        };
        Ok(Some((
            kv.k_fp8.download(self.ctx).map_err(err)?,
            kv.v_fp8.download(self.ctx).map_err(err)?,
            kv.k_scales.download(self.ctx).map_err(err)?,
            kv.v_scales.download(self.ctx).map_err(err)?,
        )))
    }

    pub fn kv_cache_restore(&mut self, li: usize, snap: &KvCacheSnapshot) -> Result<bool> {
        let Some(kv) = self.kv_layers.get(li) else {
            return Ok(false);
        };
        kv.k_fp8.write(self.ctx, &snap.0).map_err(err)?;
        kv.v_fp8.write(self.ctx, &snap.1).map_err(err)?;
        kv.k_scales.write(self.ctx, &snap.2).map_err(err)?;
        kv.v_scales.write(self.ctx, &snap.3).map_err(err)?;
        Ok(true)
    }

    pub fn restore_pos(&mut self, pos: usize) -> Result<()> {
        anyhow::ensure!(
            pos <= self.max_seq,
            "restore_pos {pos} past max_seq {}",
            self.max_seq
        );
        self.pos = pos;
        self.sync_active();
        Ok(())
    }

    pub fn rewind_limits(&self) -> crate::prefix_reuse::RewindLimits {
        crate::prefix_reuse::RewindLimits::positional(
            self.fd_sliding_base.ring as usize,
            self.config.sliding_window,
            self.prefill_chunk_len(),
        )
    }

    pub fn rewind_to(&mut self, pos: usize) -> Result<bool> {
        if !self.rewind_limits().admits(self.pos, pos) {
            return Ok(false);
        }
        self.truncate_to(pos)?;
        Ok(true)
    }

    fn window_start(total: usize, window: usize) -> usize {
        if window > 0 && total > window {
            total - window
        } else {
            0
        }
    }

    fn encode_cb(&self, full: bool) -> wgpu::CommandBuffer {
        let passes = if full {
            &self.passes[..]
        } else {
            &self.passes[..self.head_start]
        };
        if dispatch::profile::enabled() {
            let labels: Vec<&str> = passes.iter().map(|p| p.label.as_str()).collect();
            dispatch::encode_pass_list_labeled(
                self.ctx,
                passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
                &labels,
            )
        } else {
            dispatch::encode_pass_list(
                self.ctx,
                passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
            )
        }
    }

    pub fn preenc(&self) -> bool {
        self.preenc
    }

    pub fn set_preenc(&mut self, on: bool) {
        self.preenc = on;
        if !on {
            self.pending_cb = None;
        }
    }

    pub fn host_shape_counters(&self) -> (u64, u64) {
        (self.preenc_hits, self.chain_steps)
    }

    pub fn set_uniform_probe(&mut self, reps: usize) {
        self.uniform_probe = reps;
    }

    fn write_pos_uniforms(&self, at: usize) -> Result<()> {
        for _ in 0..self.uniform_probe {
            self.write_pos_uniforms_once(at)?;
        }
        self.write_pos_uniforms_once(at)
    }

    fn write_pos_uniforms_once(&self, at: usize) -> Result<()> {
        let base = self.kv_base;
        let total = at + 1;
        self.rope_pos.write(self.ctx, &[at as i32]).map_err(err)?;
        self.kv_start
            .write(self.ctx, &[(base + at) as i32])
            .map_err(err)?;
        let mut fd_s = self.fd_sliding_base;
        fd_s.total = (base + total) as u32;
        fd_s.start = (base + Self::window_start(total, self.config.sliding_window)) as u32;
        self.fd_sliding.write(self.ctx, &fd_s);
        let mut fd_f = self.fd_full_base;
        fd_f.total = (base + total) as u32;
        fd_f.start = base as u32;
        self.fd_full.write(self.ctx, &fd_f);
        Ok(())
    }

    fn step_inner(&mut self, token: u32) -> Result<()> {
        self.step_inner_full(token, true)
    }

    fn step_inner_full(&mut self, token: u32, full: bool) -> Result<()> {
        anyhow::ensure!((token as usize) < self.vocab, "token {token} out of vocab");
        anyhow::ensure!(self.pos < self.max_seq, "kv cache full at {}", self.pos);
        self.tok_idx.write(self.ctx, &[token as i32]).map_err(err)?;
        self.write_pos_uniforms(self.pos)?;

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
        let reuse = full && !dispatch::profile::enabled();
        if !reuse {
            self.pending_cb = None;
        }
        let cb = match if reuse { self.pending_cb.take() } else { None } {
            Some(cb) => {
                self.preenc_hits += 1;
                cb
            }
            None => self.encode_cb(full),
        };
        self.ctx.queue.submit([cb]);
        if reuse && self.preenc {
            self.pending_cb = Some(self.encode_cb(true));
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_wgpu decode step validation: {e}");
            }
            if full {
                self.validated = true;
            }
            self.prefix_validated = true;
        }
        self.pos += 1;
        Ok(())
    }

    pub fn decode_chain(&mut self, token: u32, k: usize) -> Result<Vec<u32>> {
        anyhow::ensure!(
            (1..=MAX_CHAIN).contains(&k),
            "decode_chain k {k} outside 1..={MAX_CHAIN}"
        );
        if k == 1 || !self.validated || dispatch::profile::enabled() {
            let mut out = Vec::with_capacity(k);
            let mut t = token;
            for _ in 0..k {
                t = self.decode_step(t)?;
                out.push(t);
            }
            return Ok(out);
        }
        anyhow::ensure!((token as usize) < self.vocab, "token {token} out of vocab");
        anyhow::ensure!(
            self.pos + k <= self.max_seq,
            "kv cache full at {} + {k}",
            self.pos
        );
        for i in 0..k {
            if i == 0 {
                self.tok_idx.write(self.ctx, &[token as i32]).map_err(err)?;
            }
            self.write_pos_uniforms(self.pos)?;
            let cb = match self.pending_cb.take() {
                Some(cb) => {
                    self.preenc_hits += 1;
                    cb
                }
                None => self.encode_cb(true),
            };
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(
                self.token_out.raw(),
                0,
                self.chain_out.raw(),
                (i * 4) as u64,
                4,
            );
            if i + 1 < k {
                enc.copy_buffer_to_buffer(self.token_out.raw(), 0, self.tok_idx.raw(), 0, 4);
            }
            self.ctx.queue.submit([cb, enc.finish()]);
            if self.preenc {
                self.pending_cb = Some(self.encode_cb(true));
            }
            self.pos += 1;
            self.chain_steps += 1;
        }
        self.chain_out.download_range(self.ctx, 0, k).map_err(err)
    }

    pub fn decode_step(&mut self, token: u32) -> Result<u32> {
        self.step_inner(token)?;
        let t = self.token_out.download(self.ctx).map_err(err)?;
        Ok(t[0])
    }

    pub fn prefill_step(&mut self, token: u32) -> Result<()> {
        self.step_inner_full(token, false)
    }

    pub fn prefill_chunk_len(&self) -> usize {
        self.prefill.as_ref().map(|p| p.m).unwrap_or(0)
    }

    pub fn prefill_pass_count(&self) -> usize {
        self.prefill.as_ref().map(|p| p.passes.len()).unwrap_or(0)
    }

    pub fn prefill_coop_ffn_gemm_sites(&self) -> usize {
        self.pf_coop_sites
    }

    fn prefill_chunk_advance(&mut self, tokens: &[u32], live: usize) -> Result<()> {
        self.prefill_chunk_masked(tokens, live, &[])
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
        let kv_base = self.kv_base;
        let window = self.config.sliding_window;
        let base_s = self.fd_sliding_base;
        let base_f = self.fd_full_base;
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
        pf.tok_idx.write(ctx, &ids).map_err(err)?;
        pf.rope_pos.write(ctx, &poss).map_err(err)?;
        pf.kv_start
            .write(ctx, &[(kv_base + pos0) as i32])
            .map_err(err)?;
        for t in 0..m {
            let total = pos0 + t + 1;
            let mut fs = base_s;
            fs.total = (kv_base + total) as u32;
            fs.start = (kv_base + Self::window_start(total, window)) as u32;
            pf.fd_s[t].write(ctx, &fs);
            let mut ff = base_f;
            ff.total = (kv_base + total) as u32;
            ff.start = kv_base as u32;
            pf.fd_f[t].write(ctx, &ff);
        }
        if let Some(fdu) = pf.fd_flash.as_ref() {
            anyhow::ensure!(
                kv_base == 0,
                "the tiled prefill flash arm derives its causal window from fd_params.window alone, \
                 so kv slot base {kv_base} != 0 needs NV_G4_WGPU_PF_FLASH=0 or slot 0"
            );
            let mut ff = base_f;
            ff.m_rows = m as u32;
            ff.total = (pos0 + m) as u32;
            ff.start = 0;
            fdu.write(ctx, &ff);
        }
        let hidden_words = self.config.hidden_size / 2;
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

    fn submit_prefill_passes(&mut self) -> Result<()> {
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
        if dispatch::profile::enabled() && ctx.caps.timestamp_query {
            let raw: Vec<(&wgpu::ComputePipeline, &wgpu::BindGroup, (u32, u32, u32))> =
                pf.passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)).collect();
            let labels: Vec<String> = pf.passes.iter().map(|p| p.label.clone()).collect();
            dispatch::submit_profiled_slices(ctx, &raw, &labels).map_err(|e| {
                anyhow::anyhow!("profiled prefill chunk submit (sliced past the 1792-pass query ceiling): {e}")
            })?;
        } else {
            let cb = dispatch::encode_pass_list(
                ctx,
                pf.passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
            );
            ctx.queue.submit([cb]);
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_wgpu prefill chunk validation: {e}");
            }
            pf.validated = true;
        }
        Ok(())
    }

    fn prefill_chunk_masked(
        &mut self,
        tokens: &[u32],
        live: usize,
        splices: &[ChunkRowSplice],
    ) -> Result<()> {
        self.write_prefill_inputs(tokens, live, splices)?;
        self.submit_prefill_passes()?;
        self.pos += live;
        Ok(())
    }

    pub fn prefill_chunk(&mut self, tokens: &[u32]) -> Result<()> {
        let m = self.prefill_chunk_len();
        anyhow::ensure!(m > 0, "chunked prefill is disabled on this graph");
        self.prefill_chunk_advance(tokens, m)
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

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        let rows = self.verify_max_rows();
        anyhow::ensure!(
            rows > 0,
            "verify_chain needs the m-row prefill graph and its lm_head epilogue"
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
        self.submit_prefill_passes()?;
        let ctx = self.ctx;
        let vs = self
            .verify
            .as_mut()
            .expect("verify_max_rows > 0 proved the verify epilogue exists");
        let scope = if vs.validated {
            None
        } else {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        };
        let cb = dispatch::encode_pass_list(
            ctx,
            vs.passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
        );
        ctx.queue.submit([cb]);
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_wgpu verify chain validation: {e}");
            }
            vs.validated = true;
        }
        let toks = vs.token_out.download(ctx).map_err(err)?;
        Ok(toks[..mb].to_vec())
    }

    pub fn verify_row_logits(&self, row: usize) -> Result<Vec<f32>> {
        let vs = self
            .verify
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("verify epilogue disabled"))?;
        anyhow::ensure!(row < vs.rows, "verify row {row} out of 0..{}", vs.rows);
        let all = vs.logits_f32.download(self.ctx).map_err(err)?;
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
            "embed-row splice prefill requires the m-row prefill graph (NV_G4_WGPU_PREFILL_M >= 2)"
        );
        let hidden = self.config.hidden_size;
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
            let chunk: Vec<u32> = tokens[done..done + m].to_vec();
            self.prefill_chunk(&chunk)?;
            done += m;
        }
        let left = tokens.len() - done;
        if left >= 2 && self.pos + m <= self.max_seq {
            let mut padded: Vec<u32> = tokens[done..].to_vec();
            let pad = *padded.last().expect("non-empty tail");
            padded.resize(m, pad);
            self.prefill_chunk_advance(&padded, left)?;
            done += left;
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

    pub fn head_pass_count(&self) -> usize {
        self.passes.len() - self.head_start
    }

    pub fn batch_slots(&self) -> usize {
        self.batch.as_ref().map(|b| b.slots).unwrap_or(0)
    }

    pub fn batch_pass_count(&self) -> usize {
        self.batch.as_ref().map(|b| b.passes.len()).unwrap_or(0)
    }

    pub fn slot_pos(&self, slot: usize) -> usize {
        if slot == self.active_slot() {
            self.pos
        } else {
            self.slot_pos.get(slot).copied().unwrap_or(0)
        }
    }

    fn active_slot(&self) -> usize {
        self.kv_base / self.max_seq
    }

    fn sync_active(&mut self) {
        let a = self.active_slot();
        self.slot_pos[a] = self.pos;
    }

    pub fn select_slot(&mut self, slot: usize) -> Result<()> {
        anyhow::ensure!(
            slot < self.slot_pos.len(),
            "slot {slot} beyond the {} this graph allocates",
            self.slot_pos.len()
        );
        self.sync_active();
        self.kv_base = slot * self.max_seq;
        self.pos = self.slot_pos[slot];
        Ok(())
    }

    pub fn reset_slot(&mut self, slot: usize) -> Result<()> {
        self.select_slot(slot)?;
        self.pos = 0;
        self.slot_pos[slot] = 0;
        Ok(())
    }

    pub fn prefill_slot(&mut self, slot: usize, tokens: &[u32]) -> Result<u32> {
        self.reset_slot(slot)?;
        self.prefill(tokens)
    }

    fn write_batch_uniforms(&mut self, tokens: &[u32]) -> Result<()> {
        self.sync_active();
        let window = self.config.sliding_window;
        let base_s = self.fd_sliding_base;
        let base_f = self.fd_full_base;
        let max_seq = self.max_seq;
        let live = tokens.len();
        let pos = self.slot_pos.clone();
        let ctx = self.ctx;
        let bt = self.batch.as_ref().expect("batch graph");
        let s = bt.slots;
        let mut ids = vec![0i32; s];
        let mut poss = vec![0i32; s];
        for j in 0..s {
            let (tok, p) = if j < live {
                (tokens[j], pos[j])
            } else {
                (tokens[0], pos[j].min(max_seq - 1))
            };
            ids[j] = tok as i32;
            poss[j] = p as i32;
            let kb = j * max_seq;
            bt.kv_start[j].write(ctx, &[(kb + p) as i32]).map_err(err)?;
            let total = p + 1;
            let mut fs = base_s;
            fs.total = (kb + total) as u32;
            fs.start = (kb + Self::window_start(total, window)) as u32;
            bt.fd_s[j].write(ctx, &fs);
            let mut ff = base_f;
            ff.total = (kb + total) as u32;
            ff.start = kb as u32;
            bt.fd_f[j].write(ctx, &ff);
        }
        bt.tok_idx.write(ctx, &ids).map_err(err)?;
        bt.rope_pos.write(ctx, &poss).map_err(err)?;
        Ok(())
    }

    pub fn batch_logits(&self) -> Result<Vec<f32>> {
        let bt = self
            .batch
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no batch graph on this model"))?;
        bt.logits_f32.download(self.ctx).map_err(err)
    }

    pub fn decode_step_batch(&mut self, tokens: &[u32]) -> Result<Vec<u32>> {
        anyhow::ensure!(!tokens.is_empty(), "decode_step_batch needs a token");
        if tokens.len() == 1 {
            self.select_slot(0)?;
            return Ok(vec![self.decode_step(tokens[0])?]);
        }
        let slots = self.batch_slots();
        anyhow::ensure!(
            slots >= tokens.len(),
            "decode_step_batch got {} sequences; this graph holds {slots} \
             (build with new_batched or NV_WGPU_BATCH_SLOTS, and check the [gemma4_wgpu] boot line for a disabler)",
            tokens.len()
        );
        for (j, &t) in tokens.iter().enumerate() {
            anyhow::ensure!(
                (t as usize) < self.vocab,
                "slot {j}: token {t} out of vocab"
            );
            anyhow::ensure!(
                self.slot_pos(j) < self.max_seq,
                "slot {j}: kv cache full at {}",
                self.slot_pos(j)
            );
        }
        self.write_batch_uniforms(tokens)?;
        let ctx = self.ctx;
        let bt = self.batch.as_mut().expect("batch graph");
        let scope = if bt.validated {
            None
        } else {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        };
        let cb = if dispatch::profile::enabled() {
            let labels: Vec<&str> = bt.passes.iter().map(|p| p.label.as_str()).collect();
            dispatch::encode_pass_list_labeled(
                ctx,
                bt.passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
                &labels,
            )
        } else {
            dispatch::encode_pass_list(
                ctx,
                bt.passes.iter().map(|p| (&*p.pipeline, &p.bind, p.grid)),
            )
        };
        ctx.queue.submit([cb]);
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_wgpu batch decode validation: {e}");
            }
            bt.validated = true;
        }
        let out = bt
            .token_out
            .download_range(ctx, 0, tokens.len())
            .map_err(err)?;
        for j in 0..tokens.len() {
            self.slot_pos[j] += 1;
        }
        self.pos = self.slot_pos[self.active_slot()];
        Ok(out)
    }

    pub fn decode_step_logits(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        self.step_inner(token)?;
        let t = self.token_out.download(self.ctx).map_err(err)?;
        let logits = self.logits_f32.download(self.ctx).map_err(err)?;
        Ok((t[0], logits))
    }
}

fn tensor_to_bf16_bits(t: &candle_core::Tensor) -> Result<Vec<u16>> {
    let v: Vec<half::bf16> = t
        .to_dtype(candle_core::DType::BF16)?
        .flatten_all()?
        .to_vec1()?;
    Ok(v.into_iter().map(|x| x.to_bits()).collect())
}

fn scalar_f32(weights: &nv_weights::WeightLoader, name: &str) -> Result<f32> {
    let t = weights.get(name, candle_core::DType::F32)?;
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    Ok(*v.first().unwrap_or(&1.0))
}

fn load_nvfp4_fused_pair(
    weights: &nv_weights::WeightLoader,
    module_a: &str,
    module_b: &str,
    n_each: usize,
    k: usize,
) -> Result<Option<HostNvfp4Lin>> {
    for m in [module_a, module_b] {
        if !weights.has(&format!("{m}.weight_scale_2"))
            || !weights.has(&format!("{m}.weight_scale"))
        {
            return Ok(None);
        }
    }
    anyhow::ensure!(
        n_each.is_multiple_of(128),
        "fused nvfp4 pair needs n % 128 == 0"
    );
    let mut packed = Vec::with_capacity(n_each * k);
    let mut scales_lin = Vec::with_capacity(2 * n_each * k / 16);
    let mut globals = Vec::new();
    for m in [module_a, module_b] {
        let pname = format!("{m}.weight");
        let shape = weights
            .shape_of(&pname)
            .ok_or_else(|| anyhow::anyhow!("missing {pname}"))?;
        anyhow::ensure!(
            shape == vec![n_each, k / 2],
            "{pname}: shape {shape:?} != [{n_each}, {}]",
            k / 2
        );
        packed.extend_from_slice(weights.raw_bytes(&pname)?);
        scales_lin.extend_from_slice(weights.raw_bytes(&format!("{m}.weight_scale"))?);
        globals.push((
            scalar_f32(weights, &format!("{m}.weight_scale_2"))?,
            scalar_f32(weights, &format!("{m}.input_scale"))?,
        ));
    }
    anyhow::ensure!(
        globals[0] == globals[1],
        "fused nvfp4 pair global scales differ: {:?}",
        globals
    );
    let safe_recip = |x: f32| {
        if x == 0.0 || !x.is_finite() {
            1.0
        } else {
            1.0 / x
        }
    };
    let stored_w = safe_recip(globals[0].0);
    let stored_x = safe_recip(globals[0].1);
    let n = 2 * n_each;
    Ok(Some(HostNvfp4Lin {
        packed,
        scales_swizzled: nv_quant::nvfp4::swizzle_scales(&scales_lin, n, k / 16),
        alpha: safe_recip(stored_w) * safe_recip(stored_x),
        input_global: stored_x,
        n,
        k,
    }))
}

fn load_nvfp4_single(
    weights: &nv_weights::WeightLoader,
    module: &str,
    n: usize,
    k: usize,
) -> Result<Option<HostNvfp4Lin>> {
    if !weights.has(&format!("{module}.weight_scale_2"))
        || !weights.has(&format!("{module}.weight_scale"))
    {
        return Ok(None);
    }
    let pname = format!("{module}.weight");
    let shape = weights
        .shape_of(&pname)
        .ok_or_else(|| anyhow::anyhow!("missing {pname}"))?;
    anyhow::ensure!(
        shape == vec![n, k / 2],
        "{pname}: shape {shape:?} != [{n}, {}]",
        k / 2
    );
    let packed = weights.raw_bytes(&pname)?.to_vec();
    let scales_lin = weights.raw_bytes(&format!("{module}.weight_scale"))?;
    let safe_recip = |x: f32| {
        if x == 0.0 || !x.is_finite() {
            1.0
        } else {
            1.0 / x
        }
    };
    let stored_w = safe_recip(scalar_f32(weights, &format!("{module}.weight_scale_2"))?);
    let stored_x = safe_recip(scalar_f32(weights, &format!("{module}.input_scale"))?);
    Ok(Some(HostNvfp4Lin {
        packed,
        scales_swizzled: nv_quant::nvfp4::swizzle_scales(scales_lin, n, k / 16),
        alpha: safe_recip(stored_w) * safe_recip(stored_x),
        input_global: stored_x,
        n,
        k,
    }))
}

pub fn host_weights_from_loader(
    config: &Gemma4Config,
    weights: &nv_weights::WeightLoader,
) -> Result<HostWeights> {
    use candle_core::DType;
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let embed = tensor_to_bf16_bits(
        &weights
            .get("model.language_model.embed_tokens.weight", DType::BF16)
            .context("load embed")?,
    )?;
    let final_norm = tensor_to_bf16_bits(
        &weights
            .get("model.language_model.norm.weight", DType::BF16)
            .context("load final norm")?,
    )?;
    anyhow::ensure!(
        config.tie_word_embeddings,
        "gemma4_wgpu loader: untied lm_head not wired"
    );

    let get_bits = |name: &str| -> Result<Vec<u16>> {
        tensor_to_bf16_bits(
            &weights
                .get(name, DType::BF16)
                .with_context(|| format!("load {name}"))?,
        )
    };

    let mut layers = Vec::with_capacity(config.num_hidden_layers);
    for i in 0..config.num_hidden_layers {
        let kind = config.layer_kind(i);
        let p = format!("model.language_model.layers.{i}");
        let hd = config.head_dim_for(kind);
        let nkv = config.num_kv_heads_for(kind);
        let n_q = config.num_attention_heads;
        let q_dim = n_q * hd;
        let kv_dim = nkv * hd;
        let has_v = !matches!(
            (kind, config.attention_k_eq_v),
            (LayerType::FullAttention, true)
        );

        let mut qkv_w = get_bits(&format!("{p}.self_attn.q_proj.weight"))?;
        qkv_w.extend(get_bits(&format!("{p}.self_attn.k_proj.weight"))?);
        if has_v {
            qkv_w.extend(get_bits(&format!("{p}.self_attn.v_proj.weight"))?);
        }
        let qkv_rows = q_dim + kv_dim * if has_v { 2 } else { 1 };
        anyhow::ensure!(qkv_w.len() == qkv_rows * hidden, "layer {i} qkv shape");

        let gate_up = match load_nvfp4_fused_pair(
            weights,
            &format!("{p}.mlp.gate_proj"),
            &format!("{p}.mlp.up_proj"),
            inter,
            hidden,
        )? {
            Some(l) => HostProj::Nvfp4(l),
            None => {
                let mut w = get_bits(&format!("{p}.mlp.gate_proj.weight"))?;
                w.extend(get_bits(&format!("{p}.mlp.up_proj.weight"))?);
                HostProj::Bf16(HostBf16Lin {
                    w,
                    n: 2 * inter,
                    k: hidden,
                })
            }
        };
        let down = match load_nvfp4_single(weights, &format!("{p}.mlp.down_proj"), hidden, inter)? {
            Some(l) => HostProj::Nvfp4(l),
            None => HostProj::Bf16(HostBf16Lin {
                w: get_bits(&format!("{p}.mlp.down_proj.weight"))?,
                n: hidden,
                k: inter,
            }),
        };

        let scalar_t = weights.get(&format!("{p}.layer_scalar"), DType::BF16)?;
        let scalar_bits = tensor_to_bf16_bits(&scalar_t)?;
        let layer_scalar = half::bf16::from_bits(scalar_bits[0]).to_f32();

        layers.push(HostLayer {
            kind,
            input_ln: get_bits(&format!("{p}.input_layernorm.weight"))?,
            post_attn_ln: get_bits(&format!("{p}.post_attention_layernorm.weight"))?,
            pre_ff_ln: get_bits(&format!("{p}.pre_feedforward_layernorm.weight"))?,
            post_ff_ln: get_bits(&format!("{p}.post_feedforward_layernorm.weight"))?,
            q_norm: get_bits(&format!("{p}.self_attn.q_norm.weight"))?,
            k_norm: get_bits(&format!("{p}.self_attn.k_norm.weight"))?,
            layer_scalar,
            has_v,
            qkv: HostProj::Bf16(HostBf16Lin {
                w: qkv_w,
                n: qkv_rows,
                k: hidden,
            }),
            o: HostProj::Bf16(HostBf16Lin {
                w: get_bits(&format!("{p}.self_attn.o_proj.weight"))?,
                n: hidden,
                k: q_dim,
            }),
            gate_up,
            down,
        });
        eprintln!(
            "[gemma4_wgpu] loaded layer {i}/{}",
            config.num_hidden_layers
        );
    }

    Ok(HostWeights {
        embed,
        final_norm,
        layers,
    })
}
