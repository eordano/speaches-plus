use anyhow::Result;

use nv_kernels::wgpu_backend::dispatch;
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::{compose, WgpuContext};
use nv_quant::mxfp4::BLOCK_SIZE as MX_BLOCK;

pub use crate::gpt_oss::{
    reference_step, rope_tables, stack_mx_host, yarn_inv_freq, GptOssConfig, GptOssLayerType,
    HostAttn, HostBf16Lin, HostLayer, HostMoe, HostMxStack, HostWeights, RefState, SWIGLU_ALPHA,
};
pub use crate::qwen3_5_dense_wgpu::ImageRowSplice;

use crate::gpt_oss::{bf16_val, load_bf16, load_layer};
use crate::gemma4_wgpu_shared::pack_pairs;
use crate::wgpu_ledger::VramLedger;
const MAX_HEAD_DIM: usize = 256;
const MAX_TOPK: usize = 16;
const ARGMAX_GROUPS: usize = 256;
const STAGING_FLUSH_BYTES: u64 = 256 << 20;

const GEMV_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_gemv.wgsl");

const MX_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_mx.wgsl");

const MX_XV4_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_mx_xv4.wgsl");

const MX_SG_SECTION_SPLIT: &str = "const GOW_MX_SECTION_SG";
pub const MX_SCALAR_ENTRY: &str = "gow_gemv_mx";
pub const MX_SG_ENTRY: &str = "gow_gemv_mx_sg";
const MX_SCALAR_ROWS_PER_WG: u32 = 2;
const MX_SG_ROWS_PER_WG_AT_SG32_OVER_WG256: u32 = 8;

pub const MX_SG_GATE_ENV: &str = "NV_GPTOSS_WGPU_MX_SG_REORDERED_SUM";

pub const GOW_ARM_STACK_IS_THE_DEFAULT_FLASH_SLIDING_FUSED_GEMV_SG_AND_MX_SG_TOGETHER_CUT_KV8_DECODE_7P9_TO_6P5_MS_AT_256_AND_8P3_TO_7P0_AT_8K_PPL_65P7_IN_THE_58_TO_92_BAND_AND_ZERO_RESTORES_EACH_SERIAL_ARM: bool = true;

fn arm_env_enabled_with_the_stack_default(env: &str) -> bool {
    match std::env::var(env) {
        Ok(v) => !matches!(v.trim(), "0" | "off" | "false"),
        Err(_) => GOW_ARM_STACK_IS_THE_DEFAULT_FLASH_SLIDING_FUSED_GEMV_SG_AND_MX_SG_TOGETHER_CUT_KV8_DECODE_7P9_TO_6P5_MS_AT_256_AND_8P3_TO_7P0_AT_8K_PPL_65P7_IN_THE_58_TO_92_BAND_AND_ZERO_RESTORES_EACH_SERIAL_ARM,
    }
}

pub fn mx_sg_reordered_sum_enabled() -> bool {
    arm_env_enabled_with_the_stack_default(MX_SG_GATE_ENV)
}

pub const MX_SG_BYTE_LUT_ENV: &str = "NV_GOW_MX_SG_BYTE_LUT";

pub const MX_SG_LUT_ENTRY: &str = "gow_gemv_mx_sg_lut";

pub const MX_SG_XV4_ENV: &str = "NV_GOW_MX_SG_XV4";

pub const MX_SG_XV4_ENTRY: &str = "gow_gemv_mx_sg_xv4";

pub const MX_SG_XV4_R2_ENV: &str = "NV_GOW_MX_SG_XV4_R2";

pub const MX_SG_XV4_R2_ENTRY: &str = "gow_gemv_mx_sg_xv4_r2";

pub const MX_SG_XV4_LUT_ENTRY: &str = "gow_gemv_mx_sg_xv4_lut";

pub const MX_SG_XV4_R2_LUT_ENTRY: &str = "gow_gemv_mx_sg_xv4_r2_lut";

pub const MX_SG_XSTAGE_ENV: &str = "NV_GOW_MX_SG_XSTAGE";

pub const MX_SG_XV4_LUT_XS_ENTRY: &str = "gow_gemv_mx_sg_xv4_lut_xs";

pub const MX_SG_GUPAIR_ENV: &str = "NV_GOW_MX_SG_GUPAIR";

pub const MX_SG_XV4_R2_LUT_XS_ENTRY: &str = "gow_gemv_mx_sg_xv4_r2_lut_xs";

pub const MX_SG_GUSW_ENV: &str = "NV_GOW_MX_SG_GUSW";

pub const MX_SG_XV4_R4_LUT_XS_SWIGLU_ENTRY: &str = "gow_gemv_mx_sg_xv4_r4_lut_xs_swiglu";

const MX_SG_R4SW_ROWS_PER_WG: u32 = 32;

const MX_XS_K_BLOCK_CAP_MATCHES_THE_512_VEC4_SHARED_ARRAY: usize = 128;

const MX_SG_XV4_R2_ROWS_PER_WG: u32 = 16;

pub const MOE_TOPK_SG32_ENV: &str = "NV_GOW_MOE_TOPK_SG32";

pub const MOE_TOPK_SG32_ENTRY: &str = "gow_router_topk_sg32";

pub const MOE_ROUTER_TOPK_FUSE_ENV: &str = "NV_GOW_MOE_ROUTER_TOPK_FUSE";

pub const MOE_DNCOMB_FUSE_ENV: &str = "NV_GOW_MOE_DNCOMB_FUSE";

pub const NULL_DISPATCH_ENV: &str = "NV_GOW_NULLDISP";

pub const NULL_DISPATCH_ENTRY: &str =
    "gow_null_dispatch_read_modify_writes_its_own_scratch_so_wgpu_must_serialise_it";

pub fn null_dispatch_repeat_prices_a_dependent_dispatch_inside_this_graph_instead_of_a_microbench(
) -> usize {
    std::env::var(NULL_DISPATCH_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

pub const MX_SG_XV4_R2_LUT_XS_DNCOMB_ENTRY: &str = "gow_gemv_mx_sg_xv4_r2_lut_xs_dncomb";

pub const MX_SG_XV4_R2_LUT_XS_DNCOMB_WG64_ENTRY: &str =
    "gow_gemv_mx_sg_xv4_r2_lut_xs_dncomb_wg64";

const DNCOMB_WG64_ROWS_PER_WG_RESTORES_THE_SPLIT_PASSES_720_RESIDENT_WORKGROUPS: usize = 4;

pub const MOE_DNCOMB_WG64_ENV: &str = "NV_GOW_MOE_DNCOMB_WG64";

pub const MX_SG_XV4_R2_LUT_DNCOMB_SGSLOT_ENTRY: &str =
    "gow_gemv_mx_sg_xv4_r2_lut_dncomb_sgslot";

pub const MOE_DNCOMB_SGSLOT_ENV: &str = "NV_GOW_MOE_DNCOMB_SGSLOT";

const DNCOMB_SGSLOT_KEEPS_EVERY_ROW_SLOT_PAIR_ON_ITS_OWN_SUBGROUP_SO_THE_RESIDENT_THREAD_COUNT_AND_MEMORY_LEVEL_PARALLELISM_MATCH_THE_SPLIT_PASS: usize = 4;

fn arm_env_opt_in_default_off(env: &str) -> bool {
    std::env::var(env).is_ok_and(|v| v.trim() == "1")
}

pub const GOW_MOE_FUSE_STACK_XV4_LUT_XSTAGE_GUPAIR_GUSW_DNCOMB_SGSLOT_SHIPS_ON_ROWS_GPTOSS_FUSE_GATED_LADDER_175P7_166P8_107P6_TOK_S_AT_256_8K_122K_CAMPAIGN_PLUS_19P9_35P6_58P2_PCT_AND_ZERO_RESTORES_EACH_ARM: bool = true;

fn arm_env_enabled_with_the_moe_fuse_stack_default(env: &str) -> bool {
    match std::env::var(env) {
        Ok(v) => !matches!(v.trim(), "0" | "off" | "false"),
        Err(_) => GOW_MOE_FUSE_STACK_XV4_LUT_XSTAGE_GUPAIR_GUSW_DNCOMB_SGSLOT_SHIPS_ON_ROWS_GPTOSS_FUSE_GATED_LADDER_175P7_166P8_107P6_TOK_S_AT_256_8K_122K_CAMPAIGN_PLUS_19P9_35P6_58P2_PCT_AND_ZERO_RESTORES_EACH_ARM,
    }
}

pub fn mx_sg_byte_lut_enabled() -> bool {
    arm_env_enabled_with_the_moe_fuse_stack_default(MX_SG_BYTE_LUT_ENV)
}

pub fn mx_sg_xv4_enabled() -> bool {
    arm_env_enabled_with_the_moe_fuse_stack_default(MX_SG_XV4_ENV)
}

pub fn mx_sg_xv4_r2_enabled() -> bool {
    arm_env_opt_in_default_off(MX_SG_XV4_R2_ENV)
}

pub fn moe_topk_sg32_enabled() -> bool {
    arm_env_opt_in_default_off(MOE_TOPK_SG32_ENV)
}

pub fn mx_sg_xstage_enabled() -> bool {
    arm_env_enabled_with_the_moe_fuse_stack_default(MX_SG_XSTAGE_ENV)
}

pub fn mx_sg_gupair_enabled_two_adjacent_rows_share_one_staged_x_so_the_interleaved_gate_up_pair_rides_one_x_pass(
) -> bool {
    arm_env_enabled_with_the_moe_fuse_stack_default(MX_SG_GUPAIR_ENV)
}

pub fn mx_sg_gusw_enabled_one_subgroup_owns_two_gate_up_pairs_and_writes_the_swiglu_act_word_directly_skipping_the_swiglu_pass(
) -> bool {
    arm_env_enabled_with_the_moe_fuse_stack_default(MX_SG_GUSW_ENV)
}

pub const MX_KPRE_ENV: &str = "NV_GOW_MX_KPRE";

pub fn mx_kpre_enabled_baked_block_entries_issue_the_expert_weight_and_scale_stream_before_the_fma_chain(
) -> bool {
    arm_env_opt_in_default_off(MX_KPRE_ENV)
}

fn mx_kpre_r2_xs_entry_name(blocks: usize) -> String {
    format!("gow_gemv_mx_sg_xv4_r2_lut_xs_kpre{blocks}")
}

fn mx_kpre_r4sw_entry_name(blocks: usize) -> String {
    format!("gow_gemv_mx_sg_xv4_r4_lut_xs_swiglu_kpre{blocks}")
}

const MX_KPRE_XS_PREAMBLE_MATCHES_THE_LUT_AND_XSTAGE_PROLOGUE_OF_THE_BASE_XS_ENTRIES: &str = "    gow_mx_xv4_byte_lut_every_e2m1_value_is_exact_in_f16[lid.x] =\n        pack2x16float(vec2<f32>(gow_mx_e2m1(lid.x, 0u), gow_mx_e2m1(lid.x, 1u)));\n    let slot = wid.z;\n    let xqbase = slot * (gow_mx_p.x_slot_stride_words >> 2u);\n    let xwords4 = gow_mx_p.k_blocks * 4u;\n    for (var i = lid.x; i < xwords4; i = i + 256u) {\n        gow_mx_xsh[i] = gow_mx_xv4[xqbase + i];\n    }\n    workgroupBarrier();\n\n    var e = slot;\n    if (gow_mx_p.use_sel == 1u) {\n        e = gow_mx_sel[slot];\n    }\n";

const MX_KPRE_ENTRY_BUILTINS: &str = "    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(subgroup_id) sgid: u32,\n    @builtin(subgroup_invocation_id) lane: u32\n) {\n";

fn mx_kpre_dot_chain(r: usize, wv: &str, sw: &str, si: &str, indent: &str) -> String {
    format!(
        "{indent}var d{r} = gow_mx_lut_dot8({wv}.x, xa.x, xa.y, xa.z, xa.w);\n\
         {indent}d{r} = d{r} + gow_mx_lut_dot8({wv}.y, xb.x, xb.y, xb.z, xb.w);\n\
         {indent}d{r} = d{r} + gow_mx_lut_dot8({wv}.z, xc.x, xc.y, xc.z, xc.w);\n\
         {indent}d{r} = d{r} + gow_mx_lut_dot8({wv}.w, xd.x, xd.y, xd.z, xd.w);\n\
         {indent}acc{r} = fma(gow_e8m0(byte_at({sw}, {si})), d{r}, acc{r});\n"
    )
}

fn mx_kpre_x_loads(j_off: usize, indent: &str) -> String {
    format!(
        "{indent}let xq = (lane + {j_off}u) * 4u;\n\
         {indent}let xa = gow_mx_xsh[xq];\n\
         {indent}let xb = gow_mx_xsh[xq + 1u];\n\
         {indent}let xc = gow_mx_xsh[xq + 2u];\n\
         {indent}let xd = gow_mx_xsh[xq + 3u];\n"
    )
}

fn mx_kpre_r2_xs_source_issuing_both_rows_streams_before_the_fma_chains(blocks: usize) -> String {
    use std::fmt::Write as _;
    assert!(blocks >= 1, "kpre needs at least one mx block, got {blocks}");
    let n_full = blocks / 32;
    let rem = blocks % 32;
    let mut b = String::new();
    writeln!(
        b,
        "@compute @workgroup_size(256)\nfn {}(",
        mx_kpre_r2_xs_entry_name(blocks)
    )
    .unwrap();
    b.push_str(MX_KPRE_ENTRY_BUILTINS);
    b.push_str(MX_KPRE_XS_PREAMBLE_MATCHES_THE_LUT_AND_XSTAGE_PROLOGUE_OF_THE_BASE_XS_ENTRIES);
    b.push_str(
        "    let row0 = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_R2_ROWS_PER_WG + sgid * 2u;\n\
         \x20   let row1 = row0 + 1u;\n\
         \x20   let live0 = row0 < gow_mx_p.n_rows;\n\
         \x20   let live1 = row1 < gow_mx_p.n_rows;\n\
         \x20   let wbase0 = e * gow_mx_p.w_e_stride_v4 + select(0u, row0 * gow_mx_p.k_blocks, live0);\n\
         \x20   let wbase1 = e * gow_mx_p.w_e_stride_v4 + select(0u, row1 * gow_mx_p.k_blocks, live1);\n\
         \x20   let sfbase0 = e * gow_mx_p.sf_e_stride_bytes + select(0u, row0 * gow_mx_p.k_blocks, live0);\n\
         \x20   let sfbase1 = e * gow_mx_p.sf_e_stride_bytes + select(0u, row1 * gow_mx_p.k_blocks, live1);\n\
         \x20   var acc0 = 0.0;\n\
         \x20   var acc1 = 0.0;\n\
         \x20   if (live0) {\n",
    );
    for (r, base, sf) in [(0usize, "wbase0", "sfbase0"), (1, "wbase1", "sfbase1")] {
        let guard = r == 1;
        if guard {
            for j in 0..n_full {
                writeln!(b, "        var wv{r}_{j}: vec4<u32>;").unwrap();
                writeln!(b, "        var sw{r}_{j}: u32;").unwrap();
            }
        }
        if rem > 0 || guard {
            if rem > 0 {
                writeln!(b, "        var wv{r}_r: vec4<u32>;").unwrap();
                writeln!(b, "        var sw{r}_r: u32;").unwrap();
            }
        }
        let (open, pad) = if guard {
            (format!("        if (live1) {{\n"), "            ")
        } else {
            (String::new(), "        ")
        };
        b.push_str(&open);
        for j in 0..n_full {
            let decl = if guard {
                format!("wv{r}_{j} = ")
            } else {
                format!("let wv{r}_{j} = ")
            };
            writeln!(b, "{pad}{decl}gow_mx_w[{base} + (lane + {}u)];", 32 * j).unwrap();
            let sdecl = if guard {
                format!("sw{r}_{j} = ")
            } else {
                format!("let sw{r}_{j} = ")
            };
            writeln!(
                b,
                "{pad}{sdecl}gow_mx_sf[({sf} + (lane + {}u)) >> 2u];",
                32 * j
            )
            .unwrap();
        }
        if rem > 0 {
            writeln!(b, "{pad}if (lane < {rem}u) {{").unwrap();
            writeln!(
                b,
                "{pad}    wv{r}_r = gow_mx_w[{base} + (lane + {}u)];",
                32 * n_full
            )
            .unwrap();
            writeln!(
                b,
                "{pad}    sw{r}_r = gow_mx_sf[({sf} + (lane + {}u)) >> 2u];",
                32 * n_full
            )
            .unwrap();
            writeln!(b, "{pad}}}").unwrap();
        }
        if guard {
            b.push_str("        }\n");
        }
    }
    for j in 0..n_full {
        b.push_str("        {\n");
        b.push_str(&mx_kpre_x_loads(32 * j, "            "));
        b.push_str(&mx_kpre_dot_chain(
            0,
            &format!("wv0_{j}"),
            &format!("sw0_{j}"),
            &format!("sfbase0 + (lane + {}u)", 32 * j),
            "            ",
        ));
        b.push_str("            if (live1) {\n");
        b.push_str(&mx_kpre_dot_chain(
            1,
            &format!("wv1_{j}"),
            &format!("sw1_{j}"),
            &format!("sfbase1 + (lane + {}u)", 32 * j),
            "                ",
        ));
        b.push_str("            }\n        }\n");
    }
    if rem > 0 {
        writeln!(b, "        if (lane < {rem}u) {{").unwrap();
        b.push_str(&mx_kpre_x_loads(32 * n_full, "            "));
        b.push_str(&mx_kpre_dot_chain(
            0,
            "wv0_r",
            "sw0_r",
            &format!("sfbase0 + (lane + {}u)", 32 * n_full),
            "            ",
        ));
        b.push_str("            if (live1) {\n");
        b.push_str(&mx_kpre_dot_chain(
            1,
            "wv1_r",
            "sw1_r",
            &format!("sfbase1 + (lane + {}u)", 32 * n_full),
            "                ",
        ));
        b.push_str("            }\n        }\n");
    }
    b.push_str(
        "    }\n\
         \x20   let total0 = gow_mx_sg_butterfly32(acc0);\n\
         \x20   let total1 = gow_mx_sg_butterfly32(acc1);\n\
         \x20   if (lane == 0u && live0) {\n\
         \x20       gow_mx_out_row(slot, e, row0, total0);\n\
         \x20   }\n\
         \x20   if (lane == 0u && live1) {\n\
         \x20       gow_mx_out_row(slot, e, row1, total1);\n\
         \x20   }\n\
         }\n\n",
    );
    b
}

fn mx_kpre_r4sw_source_issuing_all_four_rows_streams_before_the_fma_chains(blocks: usize) -> String {
    use std::fmt::Write as _;
    assert!(blocks >= 1, "kpre needs at least one mx block, got {blocks}");
    let n_full = blocks / 32;
    let rem = blocks % 32;
    let mut b = String::new();
    writeln!(
        b,
        "@compute @workgroup_size(256)\nfn {}(",
        mx_kpre_r4sw_entry_name(blocks)
    )
    .unwrap();
    b.push_str(MX_KPRE_ENTRY_BUILTINS);
    b.push_str(MX_KPRE_XS_PREAMBLE_MATCHES_THE_LUT_AND_XSTAGE_PROLOGUE_OF_THE_BASE_XS_ENTRIES);
    b.push_str(
        "    let r0 = (wid.x + wid.y * gow_mx_p.groups_x) * GOW_MX_SG_R4SW_ROWS_PER_WG + sgid * 4u;\n\
         \x20   let live = r0 + 3u < gow_mx_p.n_rows;\n\
         \x20   let ebase_w = e * gow_mx_p.w_e_stride_v4;\n\
         \x20   let ebase_sf = e * gow_mx_p.sf_e_stride_bytes;\n\
         \x20   let wrow = select(0u, r0 * gow_mx_p.k_blocks, live);\n\
         \x20   var acc0 = 0.0;\n\
         \x20   var acc1 = 0.0;\n\
         \x20   var acc2 = 0.0;\n\
         \x20   var acc3 = 0.0;\n\
         \x20   if (live) {\n",
    );
    for r in 0..4usize {
        for j in 0..n_full {
            writeln!(
                b,
                "        let wv{r}_{j} = gow_mx_w[ebase_w + wrow + {}u + (lane + {}u)];",
                r * blocks,
                32 * j
            )
            .unwrap();
            writeln!(
                b,
                "        let sw{r}_{j} = gow_mx_sf[(ebase_sf + wrow + {}u + (lane + {}u)) >> 2u];",
                r * blocks,
                32 * j
            )
            .unwrap();
        }
        if rem > 0 {
            writeln!(b, "        var wv{r}_r: vec4<u32>;").unwrap();
            writeln!(b, "        var sw{r}_r: u32;").unwrap();
            writeln!(b, "        if (lane < {rem}u) {{").unwrap();
            writeln!(
                b,
                "            wv{r}_r = gow_mx_w[ebase_w + wrow + {}u + (lane + {}u)];",
                r * blocks,
                32 * n_full
            )
            .unwrap();
            writeln!(
                b,
                "            sw{r}_r = gow_mx_sf[(ebase_sf + wrow + {}u + (lane + {}u)) >> 2u];",
                r * blocks,
                32 * n_full
            )
            .unwrap();
            b.push_str("        }\n");
        }
    }
    for j in 0..n_full {
        b.push_str("        {\n");
        b.push_str(&mx_kpre_x_loads(32 * j, "            "));
        for r in 0..4usize {
            b.push_str(&mx_kpre_dot_chain(
                r,
                &format!("wv{r}_{j}"),
                &format!("sw{r}_{j}"),
                &format!("ebase_sf + wrow + {}u + (lane + {}u)", r * blocks, 32 * j),
                "            ",
            ));
        }
        b.push_str("        }\n");
    }
    if rem > 0 {
        writeln!(b, "        if (lane < {rem}u) {{").unwrap();
        b.push_str(&mx_kpre_x_loads(32 * n_full, "            "));
        for r in 0..4usize {
            b.push_str(&mx_kpre_dot_chain(
                r,
                &format!("wv{r}_r"),
                &format!("sw{r}_r"),
                &format!(
                    "ebase_sf + wrow + {}u + (lane + {}u)",
                    r * blocks,
                    32 * n_full
                ),
                "            ",
            ));
        }
        b.push_str("        }\n");
    }
    b.push_str(
        "    }\n\
         \x20   let t0 = gow_mx_sg_butterfly32(acc0);\n\
         \x20   let t1 = gow_mx_sg_butterfly32(acc1);\n\
         \x20   let t2 = gow_mx_sg_butterfly32(acc2);\n\
         \x20   let t3 = gow_mx_sg_butterfly32(acc3);\n\
         \x20   if (lane == 0u && live) {\n\
         \x20       let a0 = gow_mx_r4sw_swiglu_one_matches_gow_swiglu_one_bitwise(\n\
         \x20           gow_mx_biased_row(e, r0, t0),\n\
         \x20           gow_mx_biased_row(e, r0 + 1u, t1)\n\
         \x20       );\n\
         \x20       let a1 = gow_mx_r4sw_swiglu_one_matches_gow_swiglu_one_bitwise(\n\
         \x20           gow_mx_biased_row(e, r0 + 2u, t2),\n\
         \x20           gow_mx_biased_row(e, r0 + 3u, t3)\n\
         \x20       );\n\
         \x20       gow_mx_y[slot * gow_mx_p.y_slot_stride + (r0 >> 2u)] = bitcast<f32>(bf16_pack(a0, a1));\n\
         \x20   }\n\
         }\n\n",
    );
    b
}

pub const GEMV_SG_XSTAGE_ENV: &str = "NV_GOW_GEMV_SG_XSTAGE";

pub const GEMV_SG_XS_ENTRY: &str = "gow_fu_gemv_bf16_sg_xs";

const GEMV_SG_XS_KV4_CAP_MATCHES_THE_512_VEC4_SHARED_ARRAY: usize = 512;

pub fn gemv_sg_xstage_enabled() -> bool {
    arm_env_opt_in_default_off(GEMV_SG_XSTAGE_ENV)
}

pub const QKV_FUSE_KV8_ENV: &str = "NV_GOW_QKV_FUSE_KV8";

pub const QKV_FUSE_KV8_ROWS_ENTRY: &str = "gow_fu_rope_kvrows";

pub fn qkv_fuse_kv8_enabled_one_fused_qkv_gemv_plus_one_rope_row_writer_replaces_the_kv8_layers_six_dispatch_split_chain(
) -> bool {
    arm_env_opt_in_default_off(QKV_FUSE_KV8_ENV)
}

pub const LMHEAD_I8_ENV: &str = "NV_GOW_LMHEAD_I8";

pub const LMHEAD_I8_ENTRY: &str = "gow_lmhead_i8_sg";

pub fn lmhead_i8_enabled_rowwise_int8_head_halves_vocab_gemv_bytes_and_is_ppl_gated_not_bitwise(
) -> bool {
    arm_env_opt_in_default_off(LMHEAD_I8_ENV)
}

pub const GEMV_SG_R2_ENV: &str = "NV_GOW_GEMV_SG_R2";

pub const GEMV_SG_R2_ENTRY: &str = "gow_fu_gemv_bf16_sg_r2";

const GEMV_SG_R2_ROWS_PER_WG_16_TWO_CONSECUTIVE_ROWS_PER_SUBGROUP_KEEP_EACH_ROWS_LANE_CHAIN: usize = 16;

pub fn gemv_sg_r2_enabled() -> bool {
    arm_env_opt_in_default_off(GEMV_SG_R2_ENV)
}

fn gemv_bf16_sg_entry_and_rows_per_wg(s: &Sources, k_v4: usize) -> (String, usize) {
    if s.gemv_kpre_kv4s.contains(&k_v4) {
        (
            gemv_sg_kpre_entry_name(k_v4),
            GEMV_SG_ROWS_PER_WG_8_SUBGROUPS_OF_32_OVER_WG256,
        )
    } else if gemv_sg_r2_enabled() {
        (
            GEMV_SG_R2_ENTRY.to_string(),
            GEMV_SG_R2_ROWS_PER_WG_16_TWO_CONSECUTIVE_ROWS_PER_SUBGROUP_KEEP_EACH_ROWS_LANE_CHAIN,
        )
    } else if gemv_sg_xstage_enabled() && k_v4 <= GEMV_SG_XS_KV4_CAP_MATCHES_THE_512_VEC4_SHARED_ARRAY
    {
        (
            GEMV_SG_XS_ENTRY.to_string(),
            GEMV_SG_ROWS_PER_WG_8_SUBGROUPS_OF_32_OVER_WG256,
        )
    } else {
        (
            "gow_fu_gemv_bf16_sg".to_string(),
            GEMV_SG_ROWS_PER_WG_8_SUBGROUPS_OF_32_OVER_WG256,
        )
    }
}

pub const GEMV_SG_KPRE_ENV: &str = "NV_GOW_GEMV_SG_KPRE";

pub fn gemv_sg_kpre_enabled_baked_k_entries_issue_every_weight_load_before_the_fma_chain() -> bool {
    arm_env_opt_in_default_off(GEMV_SG_KPRE_ENV)
}

fn gemv_sg_kpre_entry_name(k_v4: usize) -> String {
    format!("gow_fu_gemv_bf16_sg_kpre{k_v4}")
}

fn gemv_sg_kpre_source_issuing_every_weight_load_before_the_fma_chain(k_v4: usize) -> String {
    use std::fmt::Write as _;
    assert!(k_v4 >= 1, "kpre needs at least one vec4 of k, got {k_v4}");
    let n_full = k_v4 / 32;
    let rem = k_v4 % 32;
    let mut b = String::new();
    writeln!(
        b,
        "fn gsg_row_acc_kpre{k_v4}(row: u32, live: bool, lane: u32) -> f32 {{"
    )
    .unwrap();
    b.push_str("    var acc = 0.0;\n    if (live) {\n");
    b.push_str("        let wbase = row * gsg_p.w_row_v4;\n");
    b.push_str("        let xo = gsg_p.x_off_v4;\n");
    for j in 0..n_full {
        writeln!(b, "        let wv{j} = gsg_w[wbase + (lane + {}u)];", 32 * j).unwrap();
    }
    if rem > 0 {
        b.push_str("        var wvr: vec4<u32>;\n");
        writeln!(b, "        if (lane < {rem}u) {{").unwrap();
        writeln!(
            b,
            "            wvr = gsg_w[wbase + (lane + {}u)];",
            32 * n_full
        )
        .unwrap();
        b.push_str("        }\n");
    }
    for j in 0..n_full {
        writeln!(
            b,
            "        acc = acc + gsg_dot8(wv{j}, gsg_x[xo + (lane + {}u)]);",
            32 * j
        )
        .unwrap();
    }
    if rem > 0 {
        writeln!(b, "        if (lane < {rem}u) {{").unwrap();
        writeln!(
            b,
            "            acc = acc + gsg_dot8(wvr, gsg_x[xo + (lane + {}u)]);",
            32 * n_full
        )
        .unwrap();
        b.push_str("        }\n");
    }
    b.push_str("    }\n    return acc;\n}\n\n");
    writeln!(
        b,
        "@compute @workgroup_size(256)\nfn {}(",
        gemv_sg_kpre_entry_name(k_v4)
    )
    .unwrap();
    b.push_str(
        "    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(subgroup_id) sgid: u32,\n    @builtin(subgroup_invocation_id) lane: u32\n) {\n    let pair_base = (wid.x + wid.y * gsg_p.groups_x) * 8u;\n    let row = pair_base + sgid;\n    let live = row < gsg_p.n_rows;\n",
    );
    writeln!(
        b,
        "    let total = gsg_butterfly32(gsg_row_acc_kpre{k_v4}(row, live, lane));"
    )
    .unwrap();
    b.push_str(
        "    if (gsg_p.out_f32 == 1u) {\n        if (lane == 0u && live) {\n            gsg_y[gsg_p.y_off_words + row] = bitcast<u32>(total * gsg_p.alpha + gsg_bias(row));\n        }\n        return;\n    }\n    if (lane == 0u) {\n        gsg_row[sgid] = total;\n    }\n    workgroupBarrier();\n    let tid = lid.x;\n    if (tid < 4u) {\n        let r0 = pair_base + 2u * tid;\n        if (r0 < gsg_p.n_rows) {\n            let lo = gsg_row[2u * tid] * gsg_p.alpha + gsg_bias(r0);\n            var hi = 0.0;\n            if (r0 + 1u < gsg_p.n_rows) {\n                hi = gsg_row[2u * tid + 1u] * gsg_p.alpha + gsg_bias(r0 + 1u);\n            }\n            gsg_y[gsg_p.y_off_words + (r0 >> 1u)] = bf16_pack(lo, hi);\n        }\n    }\n}\n",
    );
    b
}

pub fn moe_router_topk_fuse_enabled() -> bool {
    arm_env_opt_in_default_off(MOE_ROUTER_TOPK_FUSE_ENV)
}

pub fn moe_dncomb_fuse_enabled_the_down_gemv_keeps_each_expert_row_in_a_register_and_folds_the_same_ascending_fma_chain_gow_moe_combine_would_have_run(
) -> bool {
    arm_env_enabled_with_the_moe_fuse_stack_default(MOE_DNCOMB_FUSE_ENV)
}

pub const RMSRES_SGLAT_ENV: &str = "NV_GOW_RMSRES_SGLAT";

pub const RMSRES_SGLAT_ENTRY: &str = "gow_rmsres_bf16_sglat";

pub fn rmsres_sglat_enabled_scratch_tree_collapses_to_one_barrier_plus_subgroup_shuffles() -> bool {
    arm_env_opt_in_default_off(RMSRES_SGLAT_ENV)
}

const RMSRES_SGLAT_WGSL_REPRODUCES_THE_SHARED_TREE_ASSOCIATION_SO_THE_RMS_BITS_MATCH: &str = r#"
struct RmsResParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
};

@group(0) @binding(0) var<storage, read> rmsres_x: array<u32>;
@group(0) @binding(1) var<storage, read_write> rmsres_res: array<u32>;
@group(0) @binding(2) var<storage, read> rmsres_w: array<u32>;
@group(0) @binding(3) var<storage, read_write> rmsres_out: array<u32>;
@group(0) @binding(4) var<uniform> rmsres_params: RmsResParams;

var<workgroup> rmsres_scratch: array<f32, 256>;
var<workgroup> rmsres_shared: f32;

@compute @workgroup_size(256)
fn gow_rmsres_bf16_sglat(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = wg.x + wg.y * nwg.x;
    if (row >= rmsres_params.batch) {
        return;
    }
    let lid = tid.x;
    let words = rmsres_params.words_per_row;
    let base = row * words;

    var local = 0.0;
    for (var i = lid; i < words; i = i + 256u) {
        let xw = rmsres_x[base + i];
        let rw = rmsres_res[base + i];
        let lo = bf16_lo(xw) + bf16_lo(rw);
        let hi = bf16_hi(xw) + bf16_hi(rw);
        rmsres_res[base + i] = bf16_pack(lo, hi);
        local = local + lo * lo + hi * hi;
    }

    rmsres_scratch[lid] = local;
    workgroupBarrier();
    if (sgid == 0u) {
        let a = (rmsres_scratch[lane] + rmsres_scratch[lane + 128u])
            + (rmsres_scratch[lane + 64u] + rmsres_scratch[lane + 192u]);
        let b = (rmsres_scratch[lane + 32u] + rmsres_scratch[lane + 160u])
            + (rmsres_scratch[lane + 96u] + rmsres_scratch[lane + 224u]);
        var t = a + b;
        t = t + subgroupShuffleXor(t, 16u);
        t = t + subgroupShuffleXor(t, 8u);
        t = t + subgroupShuffleXor(t, 4u);
        t = t + subgroupShuffleXor(t, 2u);
        t = t + subgroupShuffleXor(t, 1u);
        if (lane == 0u) {
            rmsres_shared = inverseSqrt(t / f32(rmsres_params.hidden) + rmsres_params.eps);
        }
    }
    workgroupBarrier();
    let rms = rmsres_shared;

    for (var i = lid; i < words; i = i + 256u) {
        let sw = rmsres_res[base + i];
        let ww = rmsres_w[i];
        let lo = bf16_lo(sw) * rms * bf16_lo(ww);
        let hi = bf16_hi(sw) * rms * bf16_hi(ww);
        rmsres_out[base + i] = bf16_pack(lo, hi);
    }
}
"#;

pub const FLASH_STAGE2_SCC_ENV: &str = "NV_GOW_FLASH_STAGE2_SCC";

pub const FLASH_STAGE2_SCC_ENTRY: &str = "gow_flash_stage2_sink_pk_scc";

pub fn flash_stage2_scc_enabled() -> bool {
    arm_env_opt_in_default_off(FLASH_STAGE2_SCC_ENV)
}

pub const FLASH_STAGE2_PAR_ENV: &str = "NV_GOW_FLASH_STAGE2_PAR";

pub const FLASH_STAGE2_PAR_ENTRY: &str = "gow_flash_stage2_sink_pk_par";

pub const FLASH_STAGE2_PAR_SPLIT_CACHE_512_MUST_MATCH_THE_WGSL_WORKGROUP_ARRAY_BOUND: u32 = 512;

const _: () = assert!(
    wk::flash_decode::MAX_SPLITS as u32
        <= FLASH_STAGE2_PAR_SPLIT_CACHE_512_MUST_MATCH_THE_WGSL_WORKGROUP_ARRAY_BOUND
);

pub fn flash_stage2_module_source_carries_the_serial_scc_and_par_entries_side_by_side() -> &'static str
{
    FLASH_WGSL
}

pub fn flash_stage2_par_a_serial_split_scan_over_64_workgroups_is_the_price_of_every_wider_stage1_grid(
) -> bool {
    arm_env_opt_in_default_off(FLASH_STAGE2_PAR_ENV)
}

fn flash_stage2_entry_scc_caches_the_split_scales_and_par_also_splits_the_serial_scan_over_the_eight_warps(
) -> &'static str {
    if flash_stage2_par_a_serial_split_scan_over_64_workgroups_is_the_price_of_every_wider_stage1_grid()
    {
        FLASH_STAGE2_PAR_ENTRY
    } else if flash_stage2_scc_enabled() {
        FLASH_STAGE2_SCC_ENTRY
    } else {
        "gow_flash_stage2_sink_pk"
    }
}

fn mx_scalar_section() -> &'static str {
    MX_WGSL
        .split_once(MX_SG_SECTION_SPLIT)
        .map_or(MX_WGSL, |(head, _)| head)
}

const ATTN_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_attn.wgsl");

pub const FUSED_STEP_ENV: &str = "NV_GPTOSS_WGPU_FUSED_STEP";

pub fn fused_step_enabled() -> bool {
    arm_env_enabled_with_the_stack_default(FUSED_STEP_ENV)
}

pub const GEMV_SG_ENV: &str = "NV_GPTOSS_WGPU_GEMV_SG";

pub fn gemv_sg_enabled() -> bool {
    arm_env_enabled_with_the_stack_default(GEMV_SG_ENV)
}

const GEMV_SG_ROWS_PER_WG_8_SUBGROUPS_OF_32_OVER_WG256: usize = 8;

const FUSED_WGSL: &str = r#"
struct GfuRopeParams {
    n_h: u32,
    n_kv: u32,
    head_dim: u32,
    rot_half: u32,
};

@group(0) @binding(0) var<storage, read> gfu_src: array<u32>;
@group(0) @binding(1) var<storage, read> gfu_cos: array<f32>;
@group(0) @binding(2) var<storage, read> gfu_sin: array<f32>;
@group(0) @binding(3) var<storage, read> gfu_pos: array<i32>;
@group(0) @binding(4) var<storage, read_write> gfu_q: array<u32>;
@group(0) @binding(5) var<storage, read_write> gfu_kc: array<u32>;
@group(0) @binding(6) var<storage, read_write> gfu_vc: array<u32>;
@group(0) @binding(7) var<uniform> gfu_p: GfuRopeParams;

fn gfu_at(r: u32, e: u32) -> f32 {
    let idx = r * gfu_p.head_dim + e;
    return bf16_decode(u16_at(gfu_src[idx >> 1u], idx));
}

fn gfu_rot(r: u32, e: u32, p: u32) -> f32 {
    let rh = gfu_p.rot_half;
    if (e < rh) {
        let c = gfu_cos[p * rh + e];
        let s = gfu_sin[p * rh + e];
        return gfu_at(r, e) * c - gfu_at(r, e + rh) * s;
    }
    let i = e - rh;
    let c = gfu_cos[p * rh + i];
    let s = gfu_sin[p * rh + i];
    return gfu_at(r, e) * c + gfu_at(r, i) * s;
}

@compute @workgroup_size(32)
fn gow_fu_rope_kvw(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.x;
    let hd = gfu_p.head_dim;
    var p = 0u;
    if (gfu_pos[0] > 0) {
        p = u32(gfu_pos[0]);
    }
    if (r < gfu_p.n_h) {
        for (var e0 = 2u * lid.x; e0 < hd; e0 = e0 + 64u) {
            let o0 = gfu_rot(r, e0, p);
            let o1 = gfu_rot(r, e0 + 1u, p);
            gfu_q[(r * hd + e0) >> 1u] = bf16_pack(o0, o1);
        }
        return;
    }
    let rk = r - gfu_p.n_h;
    let hw = hd >> 1u;
    let base = p * gfu_p.n_kv * hw + rk * hw;
    let krow = gfu_p.n_h + rk;
    for (var e0 = 2u * lid.x; e0 < hd; e0 = e0 + 64u) {
        let o0 = gfu_rot(krow, e0, p);
        let o1 = gfu_rot(krow, e0 + 1u, p);
        gfu_kc[base + (e0 >> 1u)] = bf16_pack(o0, o1);
    }
    let voff = (gfu_p.n_h + gfu_p.n_kv) * hw;
    for (var w = lid.x; w < hw; w = w + 32u) {
        gfu_vc[base + w] = gfu_src[voff + rk * hw + w];
    }
}

@compute @workgroup_size(32)
fn gow_fu_rope_kvrows(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let r = wid.x;
    let hd = gfu_p.head_dim;
    var p = 0u;
    if (gfu_pos[0] > 0) {
        p = u32(gfu_pos[0]);
    }
    if (r < gfu_p.n_h) {
        for (var e0 = 2u * lid.x; e0 < hd; e0 = e0 + 64u) {
            let o0 = gfu_rot(r, e0, p);
            let o1 = gfu_rot(r, e0 + 1u, p);
            gfu_q[(r * hd + e0) >> 1u] = bf16_pack(o0, o1);
        }
        return;
    }
    let rk = r - gfu_p.n_h;
    let hw = hd >> 1u;
    let krow = gfu_p.n_h + rk;
    for (var e0 = 2u * lid.x; e0 < hd; e0 = e0 + 64u) {
        let o0 = gfu_rot(krow, e0, p);
        let o1 = gfu_rot(krow, e0 + 1u, p);
        gfu_kc[rk * hw + (e0 >> 1u)] = bf16_pack(o0, o1);
    }
    let voff = (gfu_p.n_h + gfu_p.n_kv) * hw;
    for (var w = lid.x; w < hw; w = w + 32u) {
        gfu_vc[rk * hw + w] = gfu_src[voff + rk * hw + w];
    }
}

struct GfaParams {
    n_heads: u32,
    n_kv: u32,
    head_dim: u32,
    max_seq: u32,
    group: u32,
    window: u32,
    pad0: u32,
    scale: f32,
};

@group(0) @binding(8) var<storage, read> gfa_q: array<u32>;
@group(0) @binding(9) var<storage, read> gfa_kc: array<u32>;
@group(0) @binding(10) var<storage, read> gfa_vc: array<u32>;
@group(0) @binding(11) var<storage, read_write> gfa_scores: array<f32>;
@group(0) @binding(12) var<storage, read_write> gfa_out_pk: array<u32>;
@group(0) @binding(13) var<storage, read> gfa_pos: array<i32>;
@group(0) @binding(14) var<uniform> gfa_p: GfaParams;
@group(0) @binding(15) var<storage, read> gfa_sinks: array<f32>;

var<workgroup> gfa_qs: array<f32, 256>;
var<workgroup> gfa_red: array<f32, 256>;
var<workgroup> gfa_osh: array<f32, 256>;
var<workgroup> gfa_m: f32;
var<workgroup> gfa_z: f32;

@compute @workgroup_size(256)
fn gow_fu_attn_decode_pk(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let h = wid.x;
    let tid = lid.x;
    let hd = gfa_p.head_dim;
    var p = 0u;
    if (gfa_pos[0] > 0) {
        p = u32(gfa_pos[0]);
    }
    let total = p + 1u;
    var start = 0u;
    if (gfa_p.window > 0u && total > gfa_p.window) {
        start = total - gfa_p.window;
    }
    let kv = h / gfa_p.group;
    let srow = h * gfa_p.max_seq;
    let sink = gfa_sinks[h];

    for (var d = tid; d < hd; d = d + 256u) {
        let idx = h * hd + d;
        gfa_qs[d] = bf16_decode(u16_at(gfa_q[idx >> 1u], idx));
    }
    workgroupBarrier();

    var lmax = -3.4028235e38;
    for (var t = start + tid; t < total; t = t + 256u) {
        let kbase = (t * gfa_p.n_kv + kv) * hd;
        var dot = 0.0;
        for (var d = 0u; d < hd; d = d + 1u) {
            let idx = kbase + d;
            dot = fma(bf16_decode(u16_at(gfa_kc[idx >> 1u], idx)), gfa_qs[d], dot);
        }
        let s = dot * gfa_p.scale;
        gfa_scores[srow + t] = s;
        lmax = max(lmax, s);
    }
    gfa_red[tid] = lmax;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gfa_red[tid] = max(gfa_red[tid], gfa_red[tid + s]);
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        gfa_m = max(gfa_red[0], sink);
    }
    workgroupBarrier();
    let m = gfa_m;

    var lsum = 0.0;
    for (var t = start + tid; t < total; t = t + 256u) {
        let e = exp(gfa_scores[srow + t] - m);
        gfa_scores[srow + t] = e;
        lsum = lsum + e;
    }
    workgroupBarrier();
    gfa_red[tid] = lsum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            gfa_red[tid] = gfa_red[tid] + gfa_red[tid + s];
        }
        workgroupBarrier();
    }
    if (tid == 0u) {
        gfa_z = gfa_red[0] + exp(sink - m);
    }
    workgroupBarrier();
    let z = gfa_z;

    if (tid < hd) {
        var acc = 0.0;
        for (var t = start; t < total; t = t + 1u) {
            let idx = (t * gfa_p.n_kv + kv) * hd + tid;
            acc = fma(gfa_scores[srow + t], bf16_decode(u16_at(gfa_vc[idx >> 1u], idx)), acc);
        }
        gfa_osh[tid] = acc / z;
    }
    workgroupBarrier();
    if (tid < (hd >> 1u)) {
        gfa_out_pk[((h * hd) >> 1u) + tid] = bf16_pack(gfa_osh[2u * tid], gfa_osh[2u * tid + 1u]);
    }
}

struct GsgParams {
    n_rows: u32,
    k_v4: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_v4: u32,
    x_off_v4: u32,
    y_off_words: u32,
    has_bias: u32,
    alpha: f32,
    topk_k: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(30) var<storage, read> gsg_w: array<vec4<u32>>;
@group(0) @binding(31) var<storage, read> gsg_x: array<vec4<u32>>;
@group(0) @binding(32) var<uniform> gsg_p: GsgParams;
@group(0) @binding(33) var<storage, read_write> gsg_y: array<u32>;
@group(0) @binding(34) var<storage, read> gsg_b: array<u32>;
@group(0) @binding(35) var<storage, read_write> gsg_topk_ids: array<u32>;
@group(0) @binding(36) var<storage, read_write> gsg_topk_w: array<f32>;

var<workgroup> gsg_row: array<f32, 8>;

fn gsg_bias(row: u32) -> f32 {
    if (gsg_p.has_bias == 0u) {
        return 0.0;
    }
    return bf16_decode(u16_at(gsg_b[row >> 1u], row));
}

fn gsg_butterfly32(v: f32) -> f32 {
    var a = v;
    a = a + subgroupShuffleXor(a, 16u);
    a = a + subgroupShuffleXor(a, 8u);
    a = a + subgroupShuffleXor(a, 4u);
    a = a + subgroupShuffleXor(a, 2u);
    a = a + subgroupShuffleXor(a, 1u);
    return a;
}

fn gsg_dot8(wv: vec4<u32>, xv: vec4<u32>) -> f32 {
    var a = fma(bf16_lo(wv.x), bf16_lo(xv.x), 0.0);
    a = fma(bf16_hi(wv.x), bf16_hi(xv.x), a);
    a = fma(bf16_lo(wv.y), bf16_lo(xv.y), a);
    a = fma(bf16_hi(wv.y), bf16_hi(xv.y), a);
    a = fma(bf16_lo(wv.z), bf16_lo(xv.z), a);
    a = fma(bf16_hi(wv.z), bf16_hi(xv.z), a);
    a = fma(bf16_lo(wv.w), bf16_lo(xv.w), a);
    a = fma(bf16_hi(wv.w), bf16_hi(xv.w), a);
    return a;
}

@compute @workgroup_size(256)
fn gow_fu_gemv_bf16_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let pair_base = (wid.x + wid.y * gsg_p.groups_x) * 8u;
    let row = pair_base + sgid;
    let live = row < gsg_p.n_rows;
    let kv4 = select(0u, gsg_p.k_v4, live);
    let wbase = select(0u, row * gsg_p.w_row_v4, live);
    var acc = 0.0;
    for (var i = lane; i < kv4; i = i + 32u) {
        acc = acc + gsg_dot8(gsg_w[wbase + i], gsg_x[gsg_p.x_off_v4 + i]);
    }
    let total = gsg_butterfly32(acc);

    if (gsg_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            gsg_y[gsg_p.y_off_words + row] = bitcast<u32>(total * gsg_p.alpha + gsg_bias(row));
        }
        return;
    }
    if (lane == 0u) {
        gsg_row[sgid] = total;
    }
    workgroupBarrier();
    let tid = lid.x;
    if (tid < 4u) {
        let r0 = pair_base + 2u * tid;
        if (r0 < gsg_p.n_rows) {
            let lo = gsg_row[2u * tid] * gsg_p.alpha + gsg_bias(r0);
            var hi = 0.0;
            if (r0 + 1u < gsg_p.n_rows) {
                hi = gsg_row[2u * tid + 1u] * gsg_p.alpha + gsg_bias(r0 + 1u);
            }
            gsg_y[gsg_p.y_off_words + (r0 >> 1u)] = bf16_pack(lo, hi);
        }
    }
}

var<workgroup> gsg_rt_logits_each_row_dot_keeps_the_lane32_stride_and_butterfly_of_gow_fu_gemv_bf16_sg_so_fused_bits_equal_the_two_pass_chain: array<f32, 64>;

var<workgroup> gsg_xsh: array<vec4<u32>, 512>;

@compute @workgroup_size(256)
fn gow_fu_gemv_bf16_sg_xs(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    for (var i = lid.x; i < gsg_p.k_v4; i = i + 256u) {
        gsg_xsh[i] = gsg_x[gsg_p.x_off_v4 + i];
    }
    workgroupBarrier();
    let pair_base = (wid.x + wid.y * gsg_p.groups_x) * 8u;
    let row = pair_base + sgid;
    let live = row < gsg_p.n_rows;
    let kv4 = select(0u, gsg_p.k_v4, live);
    let wbase = select(0u, row * gsg_p.w_row_v4, live);
    var acc = 0.0;
    for (var i = lane; i < kv4; i = i + 32u) {
        acc = acc + gsg_dot8(gsg_w[wbase + i], gsg_xsh[i]);
    }
    let total = gsg_butterfly32(acc);

    if (gsg_p.out_f32 == 1u) {
        if (lane == 0u && live) {
            gsg_y[gsg_p.y_off_words + row] = bitcast<u32>(total * gsg_p.alpha + gsg_bias(row));
        }
        return;
    }
    if (lane == 0u) {
        gsg_row[sgid] = total;
    }
    workgroupBarrier();
    let tid = lid.x;
    if (tid < 4u) {
        let r0 = pair_base + 2u * tid;
        if (r0 < gsg_p.n_rows) {
            let lo = gsg_row[2u * tid] * gsg_p.alpha + gsg_bias(r0);
            var hi = 0.0;
            if (r0 + 1u < gsg_p.n_rows) {
                hi = gsg_row[2u * tid + 1u] * gsg_p.alpha + gsg_bias(r0 + 1u);
            }
            gsg_y[gsg_p.y_off_words + (r0 >> 1u)] = bf16_pack(lo, hi);
        }
    }
}

@group(0) @binding(37) var<storage, read> gsg_i8_row_scales: array<f32>;

fn gsg_i8_dot16(wv: vec4<u32>, xa: vec4<u32>, xb: vec4<u32>) -> f32 {
    var a = fma(int8_decode(wv.x, 0u), bf16_lo(xa.x), 0.0);
    a = fma(int8_decode(wv.x, 1u), bf16_hi(xa.x), a);
    a = fma(int8_decode(wv.x, 2u), bf16_lo(xa.y), a);
    a = fma(int8_decode(wv.x, 3u), bf16_hi(xa.y), a);
    a = fma(int8_decode(wv.y, 0u), bf16_lo(xa.z), a);
    a = fma(int8_decode(wv.y, 1u), bf16_hi(xa.z), a);
    a = fma(int8_decode(wv.y, 2u), bf16_lo(xa.w), a);
    a = fma(int8_decode(wv.y, 3u), bf16_hi(xa.w), a);
    a = fma(int8_decode(wv.z, 0u), bf16_lo(xb.x), a);
    a = fma(int8_decode(wv.z, 1u), bf16_hi(xb.x), a);
    a = fma(int8_decode(wv.z, 2u), bf16_lo(xb.y), a);
    a = fma(int8_decode(wv.z, 3u), bf16_hi(xb.y), a);
    a = fma(int8_decode(wv.w, 0u), bf16_lo(xb.z), a);
    a = fma(int8_decode(wv.w, 1u), bf16_hi(xb.z), a);
    a = fma(int8_decode(wv.w, 2u), bf16_lo(xb.w), a);
    a = fma(int8_decode(wv.w, 3u), bf16_hi(xb.w), a);
    return a;
}

@compute @workgroup_size(256)
fn gow_lmhead_i8_sg(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let row = (wid.x + wid.y * gsg_p.groups_x) * 8u + sgid;
    let live = row < gsg_p.n_rows;
    let kv16 = select(0u, gsg_p.w_row_v4, live);
    let wbase = select(0u, row * gsg_p.w_row_v4, live);
    var acc = 0.0;
    for (var i = lane; i < kv16; i = i + 32u) {
        acc = acc + gsg_i8_dot16(
            gsg_w[wbase + i],
            gsg_x[gsg_p.x_off_v4 + 2u * i],
            gsg_x[gsg_p.x_off_v4 + 2u * i + 1u]
        );
    }
    let total = gsg_butterfly32(acc);
    if (lane == 0u && live) {
        gsg_y[gsg_p.y_off_words + row] = bitcast<u32>(total * gsg_i8_row_scales[row]);
    }
}

@compute @workgroup_size(256)
fn gow_fu_gemv_bf16_sg_r2(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    let base16 = (wid.x + wid.y * gsg_p.groups_x) * 16u;
    let row0 = base16 + sgid * 2u;
    let row1 = row0 + 1u;
    let live0 = row0 < gsg_p.n_rows;
    let live1 = row1 < gsg_p.n_rows;
    let kv4 = select(0u, gsg_p.k_v4, live0);
    let wbase0 = select(0u, row0 * gsg_p.w_row_v4, live0);
    let wbase1 = select(0u, row1 * gsg_p.w_row_v4, live1);
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var i = lane; i < kv4; i = i + 32u) {
        let xv = gsg_x[gsg_p.x_off_v4 + i];
        acc0 = acc0 + gsg_dot8(gsg_w[wbase0 + i], xv);
        if (live1) {
            acc1 = acc1 + gsg_dot8(gsg_w[wbase1 + i], xv);
        }
    }
    let t0 = gsg_butterfly32(acc0);
    let t1 = gsg_butterfly32(acc1);
    if (lane != 0u) {
        return;
    }
    if (gsg_p.out_f32 == 1u) {
        if (live0) {
            gsg_y[gsg_p.y_off_words + row0] = bitcast<u32>(t0 * gsg_p.alpha + gsg_bias(row0));
        }
        if (live1) {
            gsg_y[gsg_p.y_off_words + row1] = bitcast<u32>(t1 * gsg_p.alpha + gsg_bias(row1));
        }
        return;
    }
    if (live0) {
        let lo = t0 * gsg_p.alpha + gsg_bias(row0);
        var hi = 0.0;
        if (live1) {
            hi = t1 * gsg_p.alpha + gsg_bias(row1);
        }
        gsg_y[gsg_p.y_off_words + (row0 >> 1u)] = bf16_pack(lo, hi);
    }
}

@compute @workgroup_size(256)
fn gow_fu_router_topk(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) sgid: u32,
    @builtin(subgroup_invocation_id) lane: u32
) {
    for (var rb = 0u; rb < gsg_p.n_rows; rb = rb + 8u) {
        let row = rb + sgid;
        let live = row < gsg_p.n_rows;
        let kv4 = select(0u, gsg_p.k_v4, live);
        let wbase = select(0u, row * gsg_p.w_row_v4, live);
        var acc = 0.0;
        for (var i = lane; i < kv4; i = i + 32u) {
            acc = acc + gsg_dot8(gsg_w[wbase + i], gsg_x[gsg_p.x_off_v4 + i]);
        }
        let total = gsg_butterfly32(acc);
        if (lane == 0u && live) {
            let v = total * gsg_p.alpha + gsg_bias(row);
            gsg_rt_logits_each_row_dot_keeps_the_lane32_stride_and_butterfly_of_gow_fu_gemv_bf16_sg_so_fused_bits_equal_the_two_pass_chain[row] = v;
            gsg_y[gsg_p.y_off_words + row] = bitcast<u32>(v);
        }
    }
    workgroupBarrier();
    if (lid.x != 0u) {
        return;
    }
    var taken: array<u32, 64>;
    var chosen: array<f32, 16>;
    for (var i = 0u; i < gsg_p.n_rows; i = i + 1u) {
        taken[i] = 0u;
    }
    for (var j = 0u; j < gsg_p.topk_k; j = j + 1u) {
        var best = -3.4028235e38;
        var bi = 0u;
        var found = false;
        for (var i = 0u; i < gsg_p.n_rows; i = i + 1u) {
            if (taken[i] == 1u) {
                continue;
            }
            let v = gsg_rt_logits_each_row_dot_keeps_the_lane32_stride_and_butterfly_of_gow_fu_gemv_bf16_sg_so_fused_bits_equal_the_two_pass_chain[i];
            if (!found || v > best) {
                best = v;
                bi = i;
                found = true;
            }
        }
        taken[bi] = 1u;
        gsg_topk_ids[j] = bi;
        chosen[j] = best;
    }
    var m = chosen[0];
    for (var j = 1u; j < gsg_p.topk_k; j = j + 1u) {
        m = max(m, chosen[j]);
    }
    var s = 0.0;
    for (var j = 0u; j < gsg_p.topk_k; j = j + 1u) {
        let e = exp(chosen[j] - m);
        chosen[j] = e;
        s = s + e;
    }
    for (var j = 0u; j < gsg_p.topk_k; j = j + 1u) {
        gsg_topk_w[j] = chosen[j] / s;
    }
}
"#;

const FLASH_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_flash_decode.wgsl");

pub const FLASH_DECODE_ENV: &str = "NV_GPTOSS_WGPU_FLASH_DECODE";

pub const FLASH_SPLITS_64_KVSHARE_GRID_IS_N_KV_X_SPLITS_AND_43TOKS_AT_120K_BEAT_S16_BY_1P6X: u32 = 64;

const FLASH_WORKGROUP_BYTES_QSH256_RED256_SM8_SL8_SACC2048_F32: u32 =
    (256 + 256 + 8 + 8 + 2048) * 4;

const KVSHARE_STAGE1_IS_SPECIALIZED_TO_HD_64: usize = 64;

const KVSHARE_STAGE1_IS_SPECIALIZED_TO_GROUP_8: usize = 8;

pub const FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH: u32 = 256;

pub fn flash_decode_enabled() -> bool {
    arm_env_enabled_with_the_stack_default(FLASH_DECODE_ENV)
}

pub const FLASH_SLIDING_ENV: &str = "NV_GPTOSS_WGPU_FLASH_SLIDING";

pub fn flash_sliding_enabled() -> bool {
    arm_env_enabled_with_the_stack_default(FLASH_SLIDING_ENV)
}

pub fn flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build() -> u32 {
    match std::env::var("NV_GPTOSS_WGPU_FLASH_SPLITS") {
        Ok(v) => {
            let n: u32 = v.trim().parse().unwrap_or_else(|_| {
                panic!("NV_GPTOSS_WGPU_FLASH_SPLITS={v}: expected an integer split count")
            });
            assert!(
                n >= 1 && n <= FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH,
                "NV_GPTOSS_WGPU_FLASH_SPLITS={n} out of 1..={}: stage2 scans splits serially per \
                 head and the scratch buffer carries splits*(head_dim+2) f32 rows per head",
                FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH
            );
            n
        }
        Err(_) => FLASH_SPLITS_64_KVSHARE_GRID_IS_N_KV_X_SPLITS_AND_43TOKS_AT_120K_BEAT_S16_BY_1P6X,
    }
}

pub const FLASH_DEPTHGATE_ENV: &str = "NV_GOW_FLASH_DEPTHGATE";

pub const FLASH_DEPTHGATE_SPLITS_ENV: &str = "NV_GOW_FLASH_DEPTHGATE_SPLITS";

pub const FLASH_DEPTHGATE_KNEE_TOKENS_1280_IS_WHERE_SPLITS_8_AND_64_TIE_AND_BELOW_IT_8_WINS: u32 =
    1280;

pub const FLASH_DEPTHGATE_BAND1_SPLITS_24_IS_ONE_WORKGROUP_PER_SM_AT_8_KV_HEADS_ON_188_SMS: u32 =
    24;

pub const FLASH_DEPTHGATE_KNEE2_TOKENS_3584_IS_WHERE_47_OVERTAKES_24_ON_THE_SAME_LADDER: u32 = 3584;

pub const FLASH_DEPTHGATE_BAND2_SPLITS_47_IS_TWO_WORKGROUPS_PER_SM_AT_8_KV_HEADS_ON_188_SMS: u32 =
    47;

pub const FLASH_DEPTHGATE_KNEE3_TOKENS_32768_IS_WHERE_64_TIES_47_AND_WINS_EVERY_DEPTH_BEYOND: u32 =
    32768;

pub fn flash_depthgate_enabled_shallow_tuned_splits_starve_the_grid_at_depth() -> bool {
    arm_env_opt_in_default_off(FLASH_DEPTHGATE_ENV)
}

fn flash_depthgate_forced_deep_splits_env_overrides_every_band_above_the_knee() -> Option<u32> {
    let v = std::env::var(FLASH_DEPTHGATE_SPLITS_ENV).ok()?;
    let n: u32 = v.trim().parse().unwrap_or_else(|_| {
        panic!("{FLASH_DEPTHGATE_SPLITS_ENV}={v}: expected an integer split count")
    });
    assert!(
        n >= 1 && n <= FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH,
        "{FLASH_DEPTHGATE_SPLITS_ENV}={n} out of 1..={}: stage2 scans splits serially per head \
         and the scratch buffer carries splits*(head_dim+2) f32 rows per head",
        FLASH_SPLITS_MAX_256_BOUNDS_STAGE2_SERIAL_SCAN_AND_SCRATCH
    );
    Some(n)
}

pub fn flash_depthgate_splits_for_context_tokens(tokens: u32, shallow_splits: u32) -> u32 {
    if tokens < FLASH_DEPTHGATE_KNEE_TOKENS_1280_IS_WHERE_SPLITS_8_AND_64_TIE_AND_BELOW_IT_8_WINS {
        return shallow_splits;
    }
    if let Some(forced) = flash_depthgate_forced_deep_splits_env_overrides_every_band_above_the_knee()
    {
        return forced;
    }
    if tokens < FLASH_DEPTHGATE_KNEE2_TOKENS_3584_IS_WHERE_47_OVERTAKES_24_ON_THE_SAME_LADDER {
        FLASH_DEPTHGATE_BAND1_SPLITS_24_IS_ONE_WORKGROUP_PER_SM_AT_8_KV_HEADS_ON_188_SMS
    } else if tokens
        < FLASH_DEPTHGATE_KNEE3_TOKENS_32768_IS_WHERE_64_TIES_47_AND_WINS_EVERY_DEPTH_BEYOND
    {
        FLASH_DEPTHGATE_BAND2_SPLITS_47_IS_TWO_WORKGROUPS_PER_SM_AT_8_KV_HEADS_ON_188_SMS
    } else {
        FLASH_SPLITS_64_KVSHARE_GRID_IS_N_KV_X_SPLITS_AND_43TOKS_AT_120K_BEAT_S16_BY_1P6X
    }
}

pub fn flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(
    shallow_splits: u32,
) -> u32 {
    if !flash_depthgate_enabled_shallow_tuned_splits_starve_the_grid_at_depth() {
        return shallow_splits;
    }
    let banded = match flash_depthgate_forced_deep_splits_env_overrides_every_band_above_the_knee() {
        Some(forced) => forced,
        None => FLASH_SPLITS_64_KVSHARE_GRID_IS_N_KV_X_SPLITS_AND_43TOKS_AT_120K_BEAT_S16_BY_1P6X
            .max(FLASH_DEPTHGATE_BAND1_SPLITS_24_IS_ONE_WORKGROUP_PER_SM_AT_8_KV_HEADS_ON_188_SMS)
            .max(FLASH_DEPTHGATE_BAND2_SPLITS_47_IS_TWO_WORKGROUPS_PER_SM_AT_8_KV_HEADS_ON_188_SMS),
    };
    shallow_splits.max(banded)
}

pub const FLASH_KPRE_ENV: &str = "NV_GOW_FLASH_KPRE";

pub const FLASH_KPRE_UNROLL_ENV: &str = "NV_GOW_FLASH_KPRE_UNROLL";

pub const FLASH_KPRE_UNROLL_4_ROUNDS_ISSUES_EVERY_K_V_AND_SCALE_LOAD_AND_EVERY_SCORE_REDUCE_BEFORE_THE_SERIAL_SOFTMAX: u32 = 4;

pub const FLASH_KPRE_MAX_UNROLL_8_BOUNDS_THE_LIVE_SCORE_FILE_AT_FOLD_TIMES_UNROLL: u32 = 8;

pub const FLASH_KPRE_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS: u32 = 128;

pub fn flash_kpre_banked_negative_preloading_four_rounds_costs_more_warp_residency_than_the_loads_in_flight_buy() -> bool {
    arm_env_opt_in_default_off(FLASH_KPRE_ENV)
}

pub fn flash_kpre_unroll() -> u32 {
    let n = match std::env::var(FLASH_KPRE_UNROLL_ENV) {
        Ok(v) => v
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{FLASH_KPRE_UNROLL_ENV}={v}: expected an integer unroll")),
        Err(_) => {
            FLASH_KPRE_UNROLL_4_ROUNDS_ISSUES_EVERY_K_V_AND_SCALE_LOAD_AND_EVERY_SCORE_REDUCE_BEFORE_THE_SERIAL_SOFTMAX
        }
    };
    assert!(
        n >= 1 && n <= FLASH_KPRE_MAX_UNROLL_8_BOUNDS_THE_LIVE_SCORE_FILE_AT_FOLD_TIMES_UNROLL,
        "{FLASH_KPRE_UNROLL_ENV}={n} out of 1..={}: every unrolled round keeps one score per \
         folded query head live across the preload, so the register file bounds the unroll",
        FLASH_KPRE_MAX_UNROLL_8_BOUNDS_THE_LIVE_SCORE_FILE_AT_FOLD_TIMES_UNROLL
    );
    n
}

pub const FLASH_BLOCKSPLIT_ENV: &str = "NV_GOW_FLASH_BLOCKSPLIT";

pub fn flash_blocksplit_banked_negative_cyclic_splits_sweep_the_kv_stream_in_lockstep_and_contiguous_blocks_do_not(
) -> bool {
    arm_env_opt_in_default_off(FLASH_BLOCKSPLIT_ENV)
}

pub fn flash_kpre_stage1_entry_sd(hd: u32, fold: u32, unroll: u32, block: bool) -> String {
    format!(
        "gow_{}_kpre{unroll}{}",
        wk::flash_decode::fold_stage1_entry_sd(hd, true, fold),
        if block { "_blk" } else { "" }
    )
}

pub fn flash_kpre_stage1_source_sd(hd: u32, fold: u32, unroll: u32, block: bool) -> String {
    use std::fmt::Write as _;
    assert!(
        hd.is_multiple_of(32)
            && hd >= 32
            && hd <= FLASH_KPRE_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS,
        "the gow kpre stage1 gives each of the first head_dim/4 lanes exactly one packed \
         fp8 word, so head_dim must be a multiple of 32 up to \
         {FLASH_KPRE_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS}, got {hd}"
    );
    let shared = wk::flash_decode::fold_stage1_entry_sd(hd, true, fold);
    let p = shared
        .strip_suffix("_stage1_fp8_sd")
        .unwrap_or_else(|| {
            panic!(
                "the gow kpre stage1 reuses the shared fold kernel's workgroup arrays, reduce \
                 and epilogue by name; `{shared}` no longer ends in _stage1_fp8_sd so the \
                 prefix cannot be derived and the generated kernel would reference nothing"
            )
        })
        .to_string();
    let entry = flash_kpre_stage1_entry_sd(hd, fold, unroll, block);
    let accs = hd / 32;
    let sd = " * bitcast<f32>(0x7B800000u)";
    let mut b = String::with_capacity(32768);
    writeln!(b, "@compute @workgroup_size(256)").unwrap();
    writeln!(
        b,
        "fn {entry}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) tid: \
         vec3<u32>) {{"
    )
    .unwrap();
    writeln!(b, "    let h0 = wg.x * {fold}u;").unwrap();
    writeln!(b, "    let split = wg.y;").unwrap();
    writeln!(b, "    if (h0 >= fd_params.n_heads) {{ return; }}").unwrap();
    writeln!(b, "    let hd = fd_params.head_dim;").unwrap();
    writeln!(b, "    let nkv = fd_params.n_kv;").unwrap();
    writeln!(b, "    let group = fd_params.n_heads / nkv;").unwrap();
    writeln!(b, "    let kvh = h0 / group;").unwrap();
    writeln!(b, "    let lid = tid.x;").unwrap();
    writeln!(b, "    let lane = lid & 31u;").unwrap();
    writeln!(b, "    let warp = lid >> 5u;").unwrap();
    writeln!(
        b,
        "    for (var d = lid; d < hd * {fold}u; d = d + FD_BLOCK) {{ {p}_qsh[d] = fd_q[h0 * hd + \
         d]; }}"
    )
    .unwrap();
    writeln!(b, "    workgroupBarrier();").unwrap();
    for j in 0..fold {
        writeln!(b, "    var m{j} = fd_neg_inf();").unwrap();
        writeln!(b, "    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "    var a{j}_{i} = 0.0;").unwrap();
        }
    }
    if block {
        writeln!(b, "    let span = fd_params.total - fd_params.start;").unwrap();
        writeln!(
            b,
            "    let per = (span + fd_params.splits - 1u) / fd_params.splits;"
        )
        .unwrap();
        writeln!(b, "    let base = fd_params.start + split * per;").unwrap();
        writeln!(b, "    let total = min(base + per, fd_params.total);").unwrap();
        writeln!(b, "    let stride = FD_WARPS;").unwrap();
    } else {
        writeln!(b, "    let total = fd_params.total;").unwrap();
        writeln!(b, "    let base = fd_params.start + split * FD_WARPS;").unwrap();
        writeln!(b, "    let stride = fd_params.splits * FD_WARPS;").unwrap();
    }
    writeln!(b, "    var rounds = 0u;").unwrap();
    writeln!(
        b,
        "    if (total > base) {{ rounds = (total - base + stride - 1u) / stride; }}"
    )
    .unwrap();
    writeln!(b, "    let n4 = hd >> 2u;").unwrap();
    writeln!(b, "    let kpre_blocks = rounds / {unroll}u;").unwrap();

    let preload = |b: &mut String, t: u32, r_expr: &str| {
        writeln!(b, "        let p{t} = base + warp + ({r_expr}) * stride;").unwrap();
        writeln!(b, "        let live{t} = p{t} < total;").unwrap();
        writeln!(b, "        var sp{t} = p{t};").unwrap();
        writeln!(
            b,
            "        if (fd_params.ring > 0u) {{ sp{t} = p{t} % fd_params.ring; }}"
        )
        .unwrap();
        writeln!(b, "        let kb{t} = (sp{t} * nkv + kvh) * hd;").unwrap();
        writeln!(b, "        var kw{t} = 0u;").unwrap();
        writeln!(b, "        var ks{t} = 0.0;").unwrap();
        writeln!(b, "        var vs{t} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "        var vw{t}_{i} = 0u;").unwrap();
        }
        writeln!(b, "        if (live{t}) {{").unwrap();
        writeln!(
            b,
            "            ks{t} = fd_k_scales[sp{t} * nkv + kvh]{sd};"
        )
        .unwrap();
        writeln!(
            b,
            "            vs{t} = fd_v_scales[sp{t} * nkv + kvh]{sd};"
        )
        .unwrap();
        writeln!(
            b,
            "            if (lane < n4) {{ kw{t} = fd_k_words[(kb{t} >> 2u) + lane]; }}"
        )
        .unwrap();
        for i in 0..accs {
            writeln!(
                b,
                "            if (lane + {i}u * FD_LANES < hd) {{ vw{t}_{i} = fd_v_words[(kb{t} + \
                 lane + {i}u * FD_LANES) >> 2u]; }}"
            )
            .unwrap();
        }
        writeln!(b, "        }}").unwrap();
    };

    let dot = |b: &mut String, t: u32| {
        for j in 0..fold {
            writeln!(b, "        var pt{t}_{j} = 0.0;").unwrap();
        }
        writeln!(b, "        if (live{t} && lane < n4) {{").unwrap();
        for i in 0..4u32 {
            writeln!(
                b,
                "            let e{t}_{i} = e4m3_shift_decode_scale_must_carry_2pow120(byte_at(kw{t}, {i}u));"
            )
            .unwrap();
        }
        writeln!(b, "            let qb{t} = lane << 2u;").unwrap();
        for j in 0..fold {
            writeln!(b, "            {{").unwrap();
            writeln!(b, "                let qo = {j}u * hd + qb{t};").unwrap();
            writeln!(b, "                var s = {p}_qsh[qo + 1u] * e{t}_1;").unwrap();
            writeln!(b, "                s = fma({p}_qsh[qo], e{t}_0, s);").unwrap();
            writeln!(b, "                s = fma({p}_qsh[qo + 2u], e{t}_2, s);").unwrap();
            writeln!(b, "                s = fma({p}_qsh[qo + 3u], e{t}_3, s);").unwrap();
            writeln!(b, "                pt{t}_{j} = pt{t}_{j} + s;").unwrap();
            writeln!(b, "            }}").unwrap();
        }
        writeln!(b, "        }}").unwrap();
    };

    let reduce = |b: &mut String, t: u32| {
        for j in 0..fold {
            writeln!(
                b,
                "        let sc{t}_{j} = ({p}_reduce(lid, pt{t}_{j}) * ks{t}) * fd_params.scaling;"
            )
            .unwrap();
        }
    };

    let update = |b: &mut String, t: u32| {
        writeln!(b, "        if (live{t}) {{").unwrap();
        for j in 0..fold {
            writeln!(b, "            let mn{j} = max(m{j}, sc{t}_{j});").unwrap();
            writeln!(b, "            let cr{j} = fd_exp(m{j} - mn{j});").unwrap();
            writeln!(b, "            let wt{j} = fd_exp(sc{t}_{j} - mn{j});").unwrap();
            writeln!(b, "            l{j} = fma(l{j}, cr{j}, wt{j});").unwrap();
            writeln!(b, "            let wv{j} = wt{j} * vs{t};").unwrap();
        }
        for i in 0..accs {
            writeln!(b, "            {{").unwrap();
            writeln!(b, "                let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(b, "                if (d < hd) {{").unwrap();
            writeln!(
                b,
                "                    let vx = e4m3_shift_decode_scale_must_carry_2pow120(byte_at(vw{t}_{i}, d));"
            )
            .unwrap();
            for j in 0..fold {
                writeln!(
                    b,
                    "                    a{j}_{i} = fma(wv{j}, vx, a{j}_{i} * cr{j});"
                )
                .unwrap();
            }
            writeln!(b, "                }}").unwrap();
            writeln!(b, "            }}").unwrap();
        }
        for j in 0..fold {
            writeln!(b, "            m{j} = mn{j};").unwrap();
        }
        writeln!(b, "        }}").unwrap();
    };

    writeln!(
        b,
        "    for (var rb = 0u; rb < kpre_blocks; rb = rb + 1u) {{"
    )
    .unwrap();
    for t in 0..unroll {
        preload(&mut b, t, &format!("rb * {unroll}u + {t}u"));
    }
    for t in 0..unroll {
        dot(&mut b, t);
    }
    for t in 0..unroll {
        reduce(&mut b, t);
    }
    for t in 0..unroll {
        update(&mut b, t);
    }
    writeln!(b, "    }}").unwrap();
    writeln!(
        b,
        "    for (var rr = kpre_blocks * {unroll}u; rr < rounds; rr = rr + 1u) {{"
    )
    .unwrap();
    preload(&mut b, unroll, "rr");
    dot(&mut b, unroll);
    reduce(&mut b, unroll);
    update(&mut b, unroll);
    writeln!(b, "    }}").unwrap();
    for j in 0..fold {
        writeln!(b, "    workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(b, "    {{").unwrap();
            writeln!(b, "        let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(
                b,
                "        if (d < hd) {{ {p}_sacc[warp * {hd}u + d] = a{j}_{i}; }}"
            )
            .unwrap();
            writeln!(b, "    }}").unwrap();
        }
        writeln!(
            b,
            "    {p}_epilogue(lid, lane, warp, hd, ((h0 + {j}u) * fd_params.splits + split) * (hd \
             + 2u), m{j}, l{j});"
        )
        .unwrap();
    }
    writeln!(b, "}}").unwrap();
    assert!(
        b.matches(&format!("{p}_reduce(")).count() as u32 == fold * (unroll + 1)
            && b.contains(&format!("{p}_epilogue("))
            && b.matches("0x7B800000").count() as u32 == 2 * (unroll + 1),
        "the gow kpre stage1 must call the shared fold reduce once per folded head per unrolled \
         round plus the tail, reuse the shared epilogue, and fold 2pow120 into both kv scales and \
         every decoded byte stream; a missed anchor silently changes the reduction or the fp8 \
         magnitudes"
    );
    b
}

pub const FLASH_LANEPOS_F16DOT_ENV: &str = "NV_GOW_FLASH_LANEPOS_F16DOT";

pub fn lanepos_f16dot_enabled() -> bool {
    arm_env_opt_in_default_off(FLASH_LANEPOS_F16DOT_ENV)
}

pub const FLASH_LANEPOS_ENV: &str = "NV_GOW_FLASH_LANEPOS";
pub const FLASH_LANEPOS_MIN_TOTAL_ENV: &str = "NV_GOW_FLASH_LANEPOS_MIN_TOTAL";

pub const LANEPOS_ENGAGES_AT_THE_MEASURED_8192_CROSSOVER_FDF_WINS_SHALLOW_LANEPOS_FROM_16K: u32 =
    8192;

pub const LANEPOS_GATED_ROUTE_SHIPS_ON_ROWS_GPTOSS_FUSE_GATED_LADDER_175P7_166P8_107P6_TOK_S_AT_256_8K_122K_SHALLOW_ARGMAX_AB_FAILS_BELOW_THE_KNEE_SO_THE_GATE_NEVER_EXPOSES_IT_AND_ZERO_RESTORES_FDF_ONLY: bool =
    true;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FlashLaneposMode {
    Off,
    Always,
    Gate(u32),
}

pub fn flash_lanepos_mode_unset_gates_at_the_measured_knee_1_always_0_restores_fdf_only(
) -> FlashLaneposMode {
    let gate_at_the_measured_knee = || {
        let knee = std::env::var(FLASH_LANEPOS_MIN_TOTAL_ENV)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(
                LANEPOS_ENGAGES_AT_THE_MEASURED_8192_CROSSOVER_FDF_WINS_SHALLOW_LANEPOS_FROM_16K,
            );
        FlashLaneposMode::Gate(knee)
    };
    match std::env::var(FLASH_LANEPOS_ENV).ok().as_deref() {
        Some("1") => FlashLaneposMode::Always,
        Some("gate") => gate_at_the_measured_knee(),
        None if LANEPOS_GATED_ROUTE_SHIPS_ON_ROWS_GPTOSS_FUSE_GATED_LADDER_175P7_166P8_107P6_TOK_S_AT_256_8K_122K_SHALLOW_ARGMAX_AB_FAILS_BELOW_THE_KNEE_SO_THE_GATE_NEVER_EXPOSES_IT_AND_ZERO_RESTORES_FDF_ONLY => {
            gate_at_the_measured_knee()
        }
        _ => FlashLaneposMode::Off,
    }
}

pub fn flash_lanepos_one_lane_owns_one_positions_whole_dot_so_no_per_position_reduce_exists(
) -> bool {
    flash_lanepos_mode_unset_gates_at_the_measured_knee_1_always_0_restores_fdf_only()
        == FlashLaneposMode::Always
}

pub const FLASH_BLKSM_ENV: &str = "NV_GOW_FLASH_BLKSM";

pub fn flash_blksm_banked_negative_the_4x_score_and_scale_register_file_costs_more_residency_than_the_shared_softmax_update_buys(
) -> bool {
    arm_env_opt_in_default_off(FLASH_BLKSM_ENV)
}

pub const FLASH_ROWPAIR_ENV: &str = "NV_GOW_FLASH_ROWPAIR";

pub const FLASH_ROWPAIR_LANES_PER_ROW_16_IS_HEAD_DIM_64_PACKED_FOUR_FP8_TO_A_WORD: u32 = 16;

pub fn flash_rowpair_banked_neutral_two_rows_per_warp_round_fills_the_idle_lanes_and_halves_the_rounds_for_no_wall() -> bool {
    arm_env_opt_in_default_off(FLASH_ROWPAIR_ENV)
}

pub fn flash_rowpair_stage1_entry_sd(hd: u32, fold: u32) -> String {
    format!(
        "gow_{}_rowpair",
        wk::flash_decode::fold_stage1_entry_sd(hd, true, fold)
    )
}

pub fn flash_rowpair_stage1_source_sd(hd: u32, fold: u32) -> String {
    use std::fmt::Write as _;
    let lanes = FLASH_ROWPAIR_LANES_PER_ROW_16_IS_HEAD_DIM_64_PACKED_FOUR_FP8_TO_A_WORD;
    assert!(
        hd.is_multiple_of(4 * lanes) && hd >= 4 * lanes,
        "the gow rowpair stage1 splits one KV row over {lanes} lanes at four packed fp8 per \
         word, so head_dim must be a positive multiple of {}, got {hd}",
        4 * lanes
    );
    let shared = wk::flash_decode::fold_stage1_entry_sd(hd, true, fold);
    let p = shared
        .strip_suffix("_stage1_fp8_sd")
        .unwrap_or_else(|| {
            panic!(
                "the gow rowpair stage1 reuses the shared fold kernel's workgroup arrays and \
                 epilogue by name; `{shared}` no longer ends in _stage1_fp8_sd so the prefix \
                 cannot be derived and the generated kernel would reference nothing"
            )
        })
        .to_string();
    let entry = flash_rowpair_stage1_entry_sd(hd, fold);
    let accs = hd / 32;
    let sd = " * bitcast<f32>(0x7B800000u)";
    let mut b = String::with_capacity(32768);
    writeln!(b, "fn {entry}_red16(x: f32) -> f32 {{").unwrap();
    writeln!(b, "    var a = x;").unwrap();
    for o in [8u32, 4, 2, 1] {
        writeln!(b, "    a = a + subgroupShuffleXor(a, {o}u);").unwrap();
    }
    writeln!(b, "    return a;\n}}\n").unwrap();
    writeln!(b, "@compute @workgroup_size(256)").unwrap();
    writeln!(
        b,
        "fn {entry}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) tid: \
         vec3<u32>) {{"
    )
    .unwrap();
    writeln!(b, "    let h0 = wg.x * {fold}u;").unwrap();
    writeln!(b, "    let split = wg.y;").unwrap();
    writeln!(b, "    if (h0 >= fd_params.n_heads) {{ return; }}").unwrap();
    writeln!(b, "    let hd = fd_params.head_dim;").unwrap();
    writeln!(b, "    let nkv = fd_params.n_kv;").unwrap();
    writeln!(b, "    let group = fd_params.n_heads / nkv;").unwrap();
    writeln!(b, "    let kvh = h0 / group;").unwrap();
    writeln!(b, "    let lid = tid.x;").unwrap();
    writeln!(b, "    let lane = lid & 31u;").unwrap();
    writeln!(b, "    let warp = lid >> 5u;").unwrap();
    writeln!(
        b,
        "    for (var d = lid; d < hd * {fold}u; d = d + FD_BLOCK) {{ {p}_qsh[d] = fd_q[h0 * hd + \
         d]; }}"
    )
    .unwrap();
    writeln!(b, "    workgroupBarrier();").unwrap();
    for j in 0..fold {
        writeln!(b, "    var m{j} = fd_neg_inf();").unwrap();
        writeln!(b, "    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "    var a{j}_{i} = 0.0;").unwrap();
        }
    }
    writeln!(b, "    let total = fd_params.total;").unwrap();
    writeln!(b, "    let base = fd_params.start + split * FD_WARPS * 2u;").unwrap();
    writeln!(b, "    let stride = fd_params.splits * FD_WARPS * 2u;").unwrap();
    writeln!(b, "    var rounds = 0u;").unwrap();
    writeln!(
        b,
        "    if (total > base) {{ rounds = (total - base + stride - 1u) / stride; }}"
    )
    .unwrap();
    writeln!(b, "    let wpl = (hd >> 2u) / {lanes}u;").unwrap();
    writeln!(b, "    let sub = lane & {}u;", lanes - 1).unwrap();
    writeln!(b, "    let hi = lane >= {lanes}u;").unwrap();
    writeln!(b, "    for (var r = 0u; r < rounds; r = r + 1u) {{").unwrap();
    writeln!(b, "        let q0 = base + warp * 2u + r * stride;").unwrap();
    writeln!(b, "        let qm = select(q0, q0 + 1u, hi);").unwrap();
    writeln!(b, "        let livem = qm < total;").unwrap();
    writeln!(b, "        var spm = qm;").unwrap();
    writeln!(
        b,
        "        if (fd_params.ring > 0u) {{ spm = qm % fd_params.ring; }}"
    )
    .unwrap();
    writeln!(b, "        let kbm = (spm * nkv + kvh) * hd;").unwrap();
    for j in 0..fold {
        writeln!(b, "        var pt{j} = 0.0;").unwrap();
    }
    writeln!(b, "        if (livem) {{").unwrap();
    writeln!(
        b,
        "            for (var t = 0u; t < wpl; t = t + 1u) {{"
    )
    .unwrap();
    writeln!(b, "                let wi = sub * wpl + t;").unwrap();
    writeln!(b, "                let kwv = fd_k_words[(kbm >> 2u) + wi];").unwrap();
    for i in 0..4u32 {
        writeln!(
            b,
            "                let e{i} = e4m3_shift_decode_scale_must_carry_2pow120(byte_at(kwv, {i}u));"
        )
        .unwrap();
    }
    for j in 0..fold {
        writeln!(b, "                {{").unwrap();
        writeln!(b, "                    let qo = {j}u * hd + (wi << 2u);").unwrap();
        writeln!(b, "                    var s = {p}_qsh[qo + 1u] * e1;").unwrap();
        writeln!(b, "                    s = fma({p}_qsh[qo], e0, s);").unwrap();
        writeln!(b, "                    s = fma({p}_qsh[qo + 2u], e2, s);").unwrap();
        writeln!(b, "                    s = fma({p}_qsh[qo + 3u], e3, s);").unwrap();
        writeln!(b, "                    pt{j} = pt{j} + s;").unwrap();
        writeln!(b, "                }}").unwrap();
    }
    writeln!(b, "            }}").unwrap();
    writeln!(b, "        }}").unwrap();
    for j in 0..fold {
        writeln!(b, "        let rd{j} = {entry}_red16(pt{j});").unwrap();
        writeln!(
            b,
            "        let ot{j} = subgroupShuffleXor(rd{j}, {lanes}u);"
        )
        .unwrap();
    }
    for i in 0..2u32 {
        writeln!(b, "        let q{i}p = q0 + {i}u;").unwrap();
        writeln!(b, "        let live{i} = q{i}p < total;").unwrap();
        writeln!(b, "        var sp{i} = q{i}p;").unwrap();
        writeln!(
            b,
            "        if (fd_params.ring > 0u) {{ sp{i} = q{i}p % fd_params.ring; }}"
        )
        .unwrap();
        writeln!(b, "        var ks{i} = 0.0;").unwrap();
        writeln!(b, "        var vs{i} = 0.0;").unwrap();
        for a in 0..accs {
            writeln!(b, "        var vw{i}_{a} = 0u;").unwrap();
        }
        writeln!(b, "        if (live{i}) {{").unwrap();
        writeln!(b, "            ks{i} = fd_k_scales[sp{i} * nkv + kvh]{sd};").unwrap();
        writeln!(b, "            vs{i} = fd_v_scales[sp{i} * nkv + kvh]{sd};").unwrap();
        for a in 0..accs {
            writeln!(
                b,
                "            if (lane + {a}u * FD_LANES < hd) {{ vw{i}_{a} = fd_v_words[((sp{i} * \
                 nkv + kvh) * hd + lane + {a}u * FD_LANES) >> 2u]; }}"
            )
            .unwrap();
        }
        writeln!(b, "        }}").unwrap();
    }
    for i in 0..2u32 {
        for j in 0..fold {
            let pick = if i == 0 {
                format!("select(rd{j}, ot{j}, hi)")
            } else {
                format!("select(ot{j}, rd{j}, hi)")
            };
            writeln!(
                b,
                "        let sc{i}_{j} = ({pick} * ks{i}) * fd_params.scaling;"
            )
            .unwrap();
        }
        writeln!(b, "        if (live{i}) {{").unwrap();
        for j in 0..fold {
            writeln!(b, "            let mn{j} = max(m{j}, sc{i}_{j});").unwrap();
            writeln!(b, "            let cr{j} = fd_exp(m{j} - mn{j});").unwrap();
            writeln!(b, "            let wt{j} = fd_exp(sc{i}_{j} - mn{j});").unwrap();
            writeln!(b, "            l{j} = fma(l{j}, cr{j}, wt{j});").unwrap();
            writeln!(b, "            let wv{j} = wt{j} * vs{i};").unwrap();
        }
        for a in 0..accs {
            writeln!(b, "            {{").unwrap();
            writeln!(b, "                let d = lane + {a}u * FD_LANES;").unwrap();
            writeln!(b, "                if (d < hd) {{").unwrap();
            writeln!(
                b,
                "                    let vx = e4m3_shift_decode_scale_must_carry_2pow120(byte_at(vw{i}_{a}, d));"
            )
            .unwrap();
            for j in 0..fold {
                writeln!(
                    b,
                    "                    a{j}_{a} = fma(wv{j}, vx, a{j}_{a} * cr{j});"
                )
                .unwrap();
            }
            writeln!(b, "                }}").unwrap();
            writeln!(b, "            }}").unwrap();
        }
        for j in 0..fold {
            writeln!(b, "            m{j} = mn{j};").unwrap();
        }
        writeln!(b, "        }}").unwrap();
    }
    writeln!(b, "    }}").unwrap();
    for j in 0..fold {
        writeln!(b, "    workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(b, "    {{").unwrap();
            writeln!(b, "        let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(
                b,
                "        if (d < hd) {{ {p}_sacc[warp * {hd}u + d] = a{j}_{i}; }}"
            )
            .unwrap();
            writeln!(b, "    }}").unwrap();
        }
        writeln!(
            b,
            "    {p}_epilogue(lid, lane, warp, hd, ((h0 + {j}u) * fd_params.splits + split) * (hd \
             + 2u), m{j}, l{j});"
        )
        .unwrap();
    }
    writeln!(b, "}}").unwrap();
    assert!(
        b.matches(&format!("{entry}_red16(")).count() as u32 == fold + 1
            && b.matches(&format!("subgroupShuffleXor(rd")).count() as u32 == fold
            && b.contains(&format!("{p}_epilogue("))
            && b.matches("0x7B800000").count() == 4,
        "the gow rowpair stage1 must reduce each folded head over the 16 lanes that own one KV \
         row, exchange the pair's scores across the halves once per head, reuse the shared \
         epilogue, and fold 2pow120 into both scales of both rows plus the decoded K stream; a \
         missed anchor silently mixes the two rows' scores"
    );
    b
}

pub const FLASH_ACC_ENV: &str = "NV_GOW_FLASH_ACC";

pub const FLASH_QREG_ENV: &str = "NV_GOW_FLASH_QREG";

pub const FLASH_SMFAST_ENV: &str = "NV_GOW_FLASH_SMFAST";

pub const FLASH_WGACC_ENV: &str = "NV_GOW_FLASH_WGACC";

pub const FLASH_ACC_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS: u32 = 128;

pub fn flash_acc_stage1_hoists_every_loop_invariant_uniform_out_of_the_round_loop() -> bool {
    arm_env_opt_in_default_off(FLASH_ACC_ENV)
        || flash_qreg_the_folded_query_rows_are_loop_invariant_and_cost_one_workgroup_load_per_component_per_head_per_round()
        || flash_smfast_a_running_max_that_does_not_move_makes_every_rescale_an_exact_multiply_by_one()
        || flash_wgacc_the_per_thread_accumulator_file_moves_to_workgroup_memory_to_buy_resident_warps()
}

pub fn flash_wgacc_the_per_thread_accumulator_file_moves_to_workgroup_memory_to_buy_resident_warps(
) -> bool {
    arm_env_opt_in_default_off(FLASH_WGACC_ENV)
}

pub fn flash_qreg_the_folded_query_rows_are_loop_invariant_and_cost_one_workgroup_load_per_component_per_head_per_round(
) -> bool {
    arm_env_opt_in_default_off(FLASH_QREG_ENV)
}

pub fn flash_smfast_a_running_max_that_does_not_move_makes_every_rescale_an_exact_multiply_by_one(
) -> bool {
    arm_env_opt_in_default_off(FLASH_SMFAST_ENV)
}

pub fn flash_acc_stage1_entry_sd(
    hd: u32,
    fold: u32,
    qreg: bool,
    smfast: bool,
    wgacc: bool,
) -> String {
    format!(
        "gow_{}_acc{}{}{}",
        wk::flash_decode::fold_stage1_entry_sd(hd, true, fold),
        if qreg { "q" } else { "" },
        if smfast { "s" } else { "" },
        if wgacc { "w" } else { "" }
    )
}

pub fn flash_acc_stage1_source_sd(
    hd: u32,
    fold: u32,
    qreg: bool,
    smfast: bool,
    wgacc: bool,
) -> String {
    use std::fmt::Write as _;
    assert!(
        hd.is_multiple_of(32)
            && hd >= 32
            && hd <= FLASH_ACC_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS,
        "the gow accumulator-file stage1 gives each of the first head_dim/4 lanes exactly one \
         packed fp8 word and holds that lane's four query components per folded head, so \
         head_dim must be a multiple of 32 up to \
         {FLASH_ACC_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS}, got {hd}"
    );
    let shared = wk::flash_decode::fold_stage1_entry_sd(hd, true, fold);
    let p = shared
        .strip_suffix("_stage1_fp8_sd")
        .unwrap_or_else(|| {
            panic!(
                "the gow accumulator-file stage1 reuses the shared fold kernel's workgroup \
                 arrays, reduce and epilogue by name; `{shared}` no longer ends in \
                 _stage1_fp8_sd so the prefix cannot be derived and the generated kernel would \
                 reference nothing"
            )
        })
        .to_string();
    let entry = flash_acc_stage1_entry_sd(hd, fold, qreg, smfast, wgacc);
    let accs = hd / 32;
    let sd = " * bitcast<f32>(0x7B800000u)";
    let acc = |j: u32, i: u32| {
        if wgacc {
            format!("{entry}_wacc[{}u + lid]", (j * accs + i) * 256)
        } else {
            format!("a{j}_{i}")
        }
    };
    let mut b = String::with_capacity(65536);
    if wgacc {
        writeln!(
            b,
            "var<workgroup> {entry}_wacc: array<f32, {}>;",
            fold * accs * 256
        )
        .unwrap();
    }
    writeln!(b, "@compute @workgroup_size(256)").unwrap();
    writeln!(
        b,
        "fn {entry}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) tid: \
         vec3<u32>) {{"
    )
    .unwrap();
    writeln!(b, "    let h0 = wg.x * {fold}u;").unwrap();
    writeln!(b, "    let split = wg.y;").unwrap();
    writeln!(b, "    if (h0 >= fd_params.n_heads) {{ return; }}").unwrap();
    writeln!(b, "    let hd = fd_params.head_dim;").unwrap();
    writeln!(b, "    let nkv = fd_params.n_kv;").unwrap();
    writeln!(b, "    let group = fd_params.n_heads / nkv;").unwrap();
    writeln!(b, "    let kvh = h0 / group;").unwrap();
    writeln!(b, "    let lid = tid.x;").unwrap();
    writeln!(b, "    let lane = lid & 31u;").unwrap();
    writeln!(b, "    let warp = lid >> 5u;").unwrap();
    writeln!(b, "    let scaling = fd_params.scaling;").unwrap();
    writeln!(b, "    let ring = fd_params.ring;").unwrap();
    writeln!(b, "    let n4 = hd >> 2u;").unwrap();
    writeln!(b, "    let hot = lane < n4;").unwrap();
    writeln!(
        b,
        "    for (var d = lid; d < hd * {fold}u; d = d + FD_BLOCK) {{ {p}_qsh[d] = fd_q[h0 * hd + \
         d]; }}"
    )
    .unwrap();
    writeln!(b, "    workgroupBarrier();").unwrap();
    writeln!(b, "    let qb = select(0u, lane << 2u, hot);").unwrap();
    if qreg {
        for j in 0..fold {
            for c in 0..4u32 {
                writeln!(
                    b,
                    "    let qr{j}_{c} = {p}_qsh[{j}u * hd + qb + {c}u];"
                )
                .unwrap();
            }
        }
    }
    for j in 0..fold {
        writeln!(b, "    var m{j} = fd_neg_inf();").unwrap();
        writeln!(b, "    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            let decl = if wgacc { "   " } else { "    var" };
            writeln!(b, "{decl} {} = 0.0;", acc(j, i)).unwrap();
        }
    }
    writeln!(b, "    let total = fd_params.total;").unwrap();
    writeln!(b, "    let base = fd_params.start + split * FD_WARPS;").unwrap();
    writeln!(b, "    let stride = fd_params.splits * FD_WARPS;").unwrap();
    writeln!(b, "    var rounds = 0u;").unwrap();
    writeln!(
        b,
        "    if (total > base) {{ rounds = (total - base + stride - 1u) / stride; }}"
    )
    .unwrap();
    writeln!(b, "    for (var r = 0u; r < rounds; r = r + 1u) {{").unwrap();
    writeln!(b, "        let pp = base + warp + r * stride;").unwrap();
    writeln!(b, "        let live = pp < total;").unwrap();
    writeln!(b, "        var sp = pp;").unwrap();
    writeln!(b, "        if (ring > 0u) {{ sp = pp % ring; }}").unwrap();
    writeln!(b, "        let kbase = (sp * nkv + kvh) * hd;").unwrap();
    for j in 0..fold {
        writeln!(b, "        var pt{j} = 0.0;").unwrap();
    }
    writeln!(b, "        var ks = 0.0;").unwrap();
    writeln!(b, "        if (live) {{").unwrap();
    writeln!(b, "            ks = fd_k_scales[sp * nkv + kvh]{sd};").unwrap();
    writeln!(b, "            if (hot) {{").unwrap();
    writeln!(
        b,
        "                let kw = fd_k_words[(kbase >> 2u) + lane];"
    )
    .unwrap();
    for c in 0..4u32 {
        writeln!(
            b,
            "                let e{c} = e4m3_shift_decode_scale_must_carry_2pow120(byte_at(kw, \
             {c}u));"
        )
        .unwrap();
    }
    for j in 0..fold {
        let q = |c: u32| {
            if qreg {
                format!("qr{j}_{c}")
            } else if c == 0 {
                format!("{p}_qsh[qo]")
            } else {
                format!("{p}_qsh[qo + {c}u]")
            }
        };
        writeln!(b, "                {{").unwrap();
        if !qreg {
            writeln!(b, "                    let qo = {j}u * hd + qb;").unwrap();
        }
        writeln!(b, "                    var t = {} * e1;", q(1)).unwrap();
        writeln!(b, "                    t = fma({}, e0, t);", q(0)).unwrap();
        writeln!(b, "                    t = fma({}, e2, t);", q(2)).unwrap();
        writeln!(b, "                    t = fma({}, e3, t);", q(3)).unwrap();
        writeln!(b, "                    pt{j} = pt{j} + t;").unwrap();
        writeln!(b, "                }}").unwrap();
    }
    writeln!(b, "            }}").unwrap();
    writeln!(b, "        }}").unwrap();
    for j in 0..fold {
        writeln!(
            b,
            "        let sc{j} = ({p}_reduce(lid, pt{j}) * ks) * scaling;"
        )
        .unwrap();
    }
    writeln!(b, "        if (live) {{").unwrap();
    writeln!(b, "            let vsc = fd_v_scales[sp * nkv + kvh]{sd};").unwrap();
    let mut general = String::new();
    for j in 0..fold {
        writeln!(general, "            let mn{j} = max(m{j}, sc{j});").unwrap();
        writeln!(general, "            let cr{j} = fd_exp(m{j} - mn{j});").unwrap();
        writeln!(general, "            let wt{j} = fd_exp(sc{j} - mn{j});").unwrap();
        writeln!(general, "            l{j} = fma(l{j}, cr{j}, wt{j});").unwrap();
        writeln!(general, "            let wv{j} = wt{j} * vsc;").unwrap();
    }
    for i in 0..accs {
        writeln!(general, "            {{").unwrap();
        writeln!(general, "                let d = lane + {i}u * FD_LANES;").unwrap();
        writeln!(general, "                if (d < hd) {{").unwrap();
        writeln!(
            general,
            "                    let vx = fd_v_fp8_sd(kbase + d);"
        )
        .unwrap();
        for j in 0..fold {
            writeln!(
                general,
                "                    {0} = fma(wv{j}, vx, {0} * cr{j});",
                acc(j, i)
            )
            .unwrap();
        }
        writeln!(general, "                }}").unwrap();
        writeln!(general, "            }}").unwrap();
    }
    for j in 0..fold {
        writeln!(general, "            m{j} = mn{j};").unwrap();
    }
    if smfast {
        let grow = (0..fold)
            .map(|j| format!("(sc{j} > m{j})"))
            .collect::<Vec<_>>()
            .join(" || ");
        writeln!(b, "            if ({grow}) {{").unwrap();
        b.push_str(&general);
        writeln!(b, "            }} else {{").unwrap();
        for j in 0..fold {
            writeln!(b, "            let wt{j} = fd_exp(sc{j} - m{j});").unwrap();
            writeln!(b, "            l{j} = l{j} + wt{j};").unwrap();
            writeln!(b, "            let wv{j} = wt{j} * vsc;").unwrap();
        }
        for i in 0..accs {
            writeln!(b, "            {{").unwrap();
            writeln!(b, "                let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(b, "                if (d < hd) {{").unwrap();
            writeln!(b, "                    let vx = fd_v_fp8_sd(kbase + d);").unwrap();
            for j in 0..fold {
                writeln!(
                    b,
                    "                    {0} = fma(wv{j}, vx, {0});",
                    acc(j, i)
                )
                .unwrap();
            }
            writeln!(b, "                }}").unwrap();
            writeln!(b, "            }}").unwrap();
        }
        writeln!(b, "            }}").unwrap();
    } else {
        b.push_str(&general);
    }
    writeln!(b, "        }}").unwrap();
    writeln!(b, "    }}").unwrap();
    for j in 0..fold {
        writeln!(b, "    workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(b, "    {{").unwrap();
            writeln!(b, "        let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(
                b,
                "        if (d < hd) {{ {p}_sacc[warp * {hd}u + d] = {}; }}",
                acc(j, i)
            )
            .unwrap();
            writeln!(b, "    }}").unwrap();
        }
        writeln!(
            b,
            "    {p}_epilogue(lid, lane, warp, hd, ((h0 + {j}u) * fd_params.splits + split) * (hd \
             + 2u), m{j}, l{j});"
        )
        .unwrap();
    }
    writeln!(b, "}}").unwrap();
    let qsh_reads_per_round = if qreg { 0 } else { fold * 4 };
    assert!(
        b.matches(&format!("{p}_reduce(")).count() as u32 == fold
            && b.contains(&format!("{p}_epilogue("))
            && b.matches("0x7B800000").count() == 2
            && b.matches(&format!("{p}_qsh[qo")).count() as u32 == qsh_reads_per_round
            && b.matches("fd_exp(").count() as u32 == if smfast { 3 * fold } else { 2 * fold },
        "the gow accumulator-file stage1 must reduce each folded head once per round, reuse the \
         shared epilogue, fold 2pow120 into both kv scales, read the staged query rows from \
         workgroup memory exactly {qsh_reads_per_round} times per round, and spend one \
         correction exp per head only on the branch where the running max actually moves; a \
         missed anchor silently changes the reduction or leaves the arm inert"
    );
    b
}

pub const FLASH_SHIPPED_RESIDENT_WORKGROUPS_PER_SM_IN_HUNDREDTHS_IS_512_ON_188_SMS: u32 = 272;

pub fn flash_accumulators_in_workgroup_memory_bytes_per_workgroup(hd: u32, fold: u32) -> u32 {
    let per_thread_accumulator_file = fold * (hd / 32);
    4 * (256 * per_thread_accumulator_file + hd * fold + hd * 8)
}

pub const FLASH_KVLINE_ENV: &str = "NV_GOW_FLASH_KVLINE";

pub const FLASH_KVLINE_SLOTS_ENV: &str = "NV_GOW_FLASH_KVLINE_SLOTS";

pub const FLASH_KVLINE_SLOTS_2_KEEPS_THE_RESIDENT_WARP_COUNT_AT_THE_SHIPPED_4096_WHILE_SPLITS_STAYS_INSIDE_THE_256_CAP: u32 = 2;

pub const FLASH_KVLINE_MAX_SLOTS_4_BOUNDS_THE_WORKGROUP_AT_1024_INVOCATIONS: u32 = 4;

pub fn flash_kvline_one_warp_per_kv_head_makes_a_workgroup_read_the_whole_slot_stride_of_a_position(
) -> bool {
    arm_env_opt_in_default_off(FLASH_KVLINE_ENV)
}

pub fn flash_kvline_slots() -> u32 {
    let n = match std::env::var(FLASH_KVLINE_SLOTS_ENV) {
        Ok(v) => v
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("{FLASH_KVLINE_SLOTS_ENV}={v}: expected an integer")),
        Err(_) => {
            FLASH_KVLINE_SLOTS_2_KEEPS_THE_RESIDENT_WARP_COUNT_AT_THE_SHIPPED_4096_WHILE_SPLITS_STAYS_INSIDE_THE_256_CAP
        }
    };
    assert!(
        n >= 1 && n <= FLASH_KVLINE_MAX_SLOTS_4_BOUNDS_THE_WORKGROUP_AT_1024_INVOCATIONS,
        "{FLASH_KVLINE_SLOTS_ENV}={n} out of 1..={}: the kvline workgroup is 32 x n_kv x slots \
         invocations and a device workgroup holds at most 1024",
        FLASH_KVLINE_MAX_SLOTS_4_BOUNDS_THE_WORKGROUP_AT_1024_INVOCATIONS
    );
    n
}

pub fn flash_kvline_stage1_entry_sd(hd: u32, nkv: u32, group: u32, slots: u32) -> String {
    format!("gow_fdf{hd}kvl{nkv}x{group}s{slots}_stage1_fp8_sd")
}

pub fn flash_kvline_workgroup_invocations(nkv: u32, slots: u32) -> u32 {
    32 * nkv * slots
}

pub fn flash_kvline_workgroup_storage_bytes(hd: u32, nkv: u32, group: u32, slots: u32) -> u32 {
    let warps = nkv * slots;
    4 * (nkv * group * hd + warps * hd + 2 * warps)
}

pub fn flash_kvline_stage1_source_sd(hd: u32, nkv: u32, group: u32, slots: u32) -> String {
    use std::fmt::Write as _;
    assert!(
        hd.is_multiple_of(32)
            && hd >= 32
            && hd <= FLASH_ACC_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS
            && nkv >= 1
            && group >= 1,
        "the gow kvline stage1 gives each of the first head_dim/4 lanes one packed fp8 word, so \
         head_dim must be a multiple of 32 up to \
         {FLASH_ACC_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS}, got {hd}"
    );
    let shared = wk::flash_decode::fold_stage1_entry_sd(hd, true, group);
    let sp = shared
        .strip_suffix("_stage1_fp8_sd")
        .unwrap_or_else(|| {
            panic!(
                "the gow kvline stage1 reuses the shared fold kernel's subgroup reduce by name; \
                 `{shared}` no longer ends in _stage1_fp8_sd so the prefix cannot be derived"
            )
        })
        .to_string();
    let entry = flash_kvline_stage1_entry_sd(hd, nkv, group, slots);
    let p = entry.strip_suffix("_stage1_fp8_sd").unwrap().to_string();
    let accs = hd / 32;
    let nh = nkv * group;
    let warps = nkv * slots;
    let wgsz = flash_kvline_workgroup_invocations(nkv, slots);
    let sd = " * bitcast<f32>(0x7B800000u)";
    let mut b = String::with_capacity(65536);
    writeln!(b, "var<workgroup> {p}_qsh: array<f32, {}>;", nh * hd).unwrap();
    writeln!(b, "var<workgroup> {p}_sacc: array<f32, {}>;", warps * hd).unwrap();
    writeln!(b, "var<workgroup> {p}_sm: array<f32, {warps}>;").unwrap();
    writeln!(b, "var<workgroup> {p}_sl: array<f32, {warps}>;").unwrap();
    writeln!(b, "@compute @workgroup_size({wgsz})").unwrap();
    writeln!(
        b,
        "fn {entry}(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) tid: \
         vec3<u32>) {{"
    )
    .unwrap();
    writeln!(
        b,
        "    if (fd_params.n_kv != {nkv}u || fd_params.n_heads != {nh}u) {{ return; }}"
    )
    .unwrap();
    writeln!(b, "    let split = wg.y;").unwrap();
    writeln!(b, "    let hd = fd_params.head_dim;").unwrap();
    writeln!(b, "    let nkv = fd_params.n_kv;").unwrap();
    writeln!(b, "    let lid = tid.x;").unwrap();
    writeln!(b, "    let lane = lid & 31u;").unwrap();
    writeln!(b, "    let warp = lid >> 5u;").unwrap();
    writeln!(b, "    let kvh = warp % {nkv}u;").unwrap();
    writeln!(b, "    let slot = warp / {nkv}u;").unwrap();
    writeln!(b, "    let h0 = kvh * {group}u;").unwrap();
    writeln!(b, "    let scaling = fd_params.scaling;").unwrap();
    writeln!(b, "    let ring = fd_params.ring;").unwrap();
    writeln!(b, "    let n4 = hd >> 2u;").unwrap();
    writeln!(b, "    let hot = lane < n4;").unwrap();
    writeln!(
        b,
        "    for (var d = lid; d < hd * {nh}u; d = d + {wgsz}u) {{ {p}_qsh[d] = fd_q[d]; }}"
    )
    .unwrap();
    writeln!(b, "    workgroupBarrier();").unwrap();
    writeln!(b, "    let qb = select(0u, lane << 2u, hot);").unwrap();
    for j in 0..group {
        writeln!(b, "    var m{j} = fd_neg_inf();").unwrap();
        writeln!(b, "    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "    var a{j}_{i} = 0.0;").unwrap();
        }
    }
    writeln!(b, "    let total = fd_params.total;").unwrap();
    writeln!(b, "    let base = fd_params.start + split * {slots}u;").unwrap();
    writeln!(b, "    let stride = fd_params.splits * {slots}u;").unwrap();
    writeln!(b, "    var rounds = 0u;").unwrap();
    writeln!(
        b,
        "    if (total > base) {{ rounds = (total - base + stride - 1u) / stride; }}"
    )
    .unwrap();
    writeln!(b, "    for (var r = 0u; r < rounds; r = r + 1u) {{").unwrap();
    writeln!(b, "        let pp = base + slot + r * stride;").unwrap();
    writeln!(b, "        let live = pp < total;").unwrap();
    writeln!(b, "        var sp = pp;").unwrap();
    writeln!(b, "        if (ring > 0u) {{ sp = pp % ring; }}").unwrap();
    writeln!(b, "        let kbase = (sp * nkv + kvh) * hd;").unwrap();
    for j in 0..group {
        writeln!(b, "        var pt{j} = 0.0;").unwrap();
    }
    writeln!(b, "        var ks = 0.0;").unwrap();
    writeln!(b, "        if (live) {{").unwrap();
    writeln!(b, "            ks = fd_k_scales[sp * nkv + kvh]{sd};").unwrap();
    writeln!(b, "            if (hot) {{").unwrap();
    writeln!(b, "                let kw = fd_k_words[(kbase >> 2u) + lane];").unwrap();
    for c in 0..4u32 {
        writeln!(
            b,
            "                let e{c} = e4m3_shift_decode_scale_must_carry_2pow120(byte_at(kw, \
             {c}u));"
        )
        .unwrap();
    }
    for j in 0..group {
        writeln!(b, "                {{").unwrap();
        writeln!(b, "                    let qo = (h0 + {j}u) * hd + qb;").unwrap();
        writeln!(b, "                    var t = {p}_qsh[qo + 1u] * e1;").unwrap();
        writeln!(b, "                    t = fma({p}_qsh[qo], e0, t);").unwrap();
        writeln!(b, "                    t = fma({p}_qsh[qo + 2u], e2, t);").unwrap();
        writeln!(b, "                    t = fma({p}_qsh[qo + 3u], e3, t);").unwrap();
        writeln!(b, "                    pt{j} = pt{j} + t;").unwrap();
        writeln!(b, "                }}").unwrap();
    }
    writeln!(b, "            }}").unwrap();
    writeln!(b, "        }}").unwrap();
    for j in 0..group {
        writeln!(
            b,
            "        let sc{j} = ({sp}_reduce(lid, pt{j}) * ks) * scaling;"
        )
        .unwrap();
    }
    writeln!(b, "        if (live) {{").unwrap();
    writeln!(b, "            let vsc = fd_v_scales[sp * nkv + kvh]{sd};").unwrap();
    for j in 0..group {
        writeln!(b, "            let mn{j} = max(m{j}, sc{j});").unwrap();
        writeln!(b, "            let cr{j} = fd_exp(m{j} - mn{j});").unwrap();
        writeln!(b, "            let wt{j} = fd_exp(sc{j} - mn{j});").unwrap();
        writeln!(b, "            l{j} = fma(l{j}, cr{j}, wt{j});").unwrap();
        writeln!(b, "            let wv{j} = wt{j} * vsc;").unwrap();
    }
    for i in 0..accs {
        writeln!(b, "            {{").unwrap();
        writeln!(b, "                let d = lane + {i}u * FD_LANES;").unwrap();
        writeln!(b, "                if (d < hd) {{").unwrap();
        writeln!(b, "                    let vx = fd_v_fp8_sd(kbase + d);").unwrap();
        for j in 0..group {
            writeln!(
                b,
                "                    a{j}_{i} = fma(wv{j}, vx, a{j}_{i} * cr{j});"
            )
            .unwrap();
        }
        writeln!(b, "                }}").unwrap();
        writeln!(b, "            }}").unwrap();
    }
    for j in 0..group {
        writeln!(b, "            m{j} = mn{j};").unwrap();
    }
    writeln!(b, "        }}").unwrap();
    writeln!(b, "    }}").unwrap();
    let peer = format!("kvh + s * {nkv}u");
    for j in 0..group {
        writeln!(b, "    workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(b, "    {{").unwrap();
            writeln!(b, "        let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(
                b,
                "        if (d < hd) {{ {p}_sacc[warp * {hd}u + d] = a{j}_{i}; }}"
            )
            .unwrap();
            writeln!(b, "    }}").unwrap();
        }
        writeln!(
            b,
            "    if (lane == 0u) {{ {p}_sm[warp] = m{j}; {p}_sl[warp] = l{j}; }}"
        )
        .unwrap();
        writeln!(b, "    workgroupBarrier();").unwrap();
        writeln!(b, "    if (slot == 0u) {{").unwrap();
        writeln!(b, "        var m_blk = fd_neg_inf();").unwrap();
        writeln!(
            b,
            "        for (var s = 0u; s < {slots}u; s = s + 1u) {{ m_blk = max(m_blk, \
             {p}_sm[{peer}]); }}"
        )
        .unwrap();
        writeln!(
            b,
            "        let dst = ((h0 + {j}u) * fd_params.splits + split) * (hd + 2u);"
        )
        .unwrap();
        writeln!(b, "        if (lane == 0u) {{").unwrap();
        writeln!(b, "            var l_blk = 0.0;").unwrap();
        writeln!(
            b,
            "            for (var s = 0u; s < {slots}u; s = s + 1u) {{ if ({p}_sm[{peer}] > \
             fd_neg_inf()) {{ l_blk = l_blk + fd_round({p}_sl[{peer}] * fd_exp({p}_sm[{peer}] - \
             m_blk)); }} }}"
        )
        .unwrap();
        writeln!(b, "            fd_scratch[dst] = m_blk;").unwrap();
        writeln!(b, "            fd_scratch[dst + 1u] = l_blk;").unwrap();
        writeln!(b, "        }}").unwrap();
        writeln!(b, "        for (var d = lane; d < hd; d = d + FD_LANES) {{").unwrap();
        writeln!(b, "            var a = 0.0;").unwrap();
        writeln!(
            b,
            "            for (var s = 0u; s < {slots}u; s = s + 1u) {{ if ({p}_sm[{peer}] > \
             fd_neg_inf()) {{ a = a + fd_round({p}_sacc[{peer} * {hd}u + d] * fd_exp({p}_sm[{peer}] \
             - m_blk)); }} }}"
        )
        .unwrap();
        writeln!(b, "            fd_scratch[dst + 2u + d] = a;").unwrap();
        writeln!(b, "        }}").unwrap();
        writeln!(b, "    }}").unwrap();
    }
    writeln!(b, "}}").unwrap();
    assert!(
        b.matches(&format!("{sp}_reduce(")).count() as u32 == group
            && b.matches("0x7B800000").count() == 2
            && b.contains(&format!("let kvh = warp % {nkv}u;"))
            && b.matches("fd_scratch[dst").count() as u32 == 3 * group,
        "the gow kvline stage1 must map one warp to one kv head so the {nkv} warps of a slot \
         request the whole {}-byte position stride in one round, reduce each folded head once \
         per round through the shared subgroup butterfly, fold 2pow120 into both kv scales, and \
         write m, l and the accumulator row of every folded head",
        nkv * hd
    );
    b
}

pub const KV_FP8_ENV: &str = "NV_GOW_KV_FP8";

pub const GOW_KV_FP8_DEFAULT_ON_THE_DEPTH_STACK_DOMINATES_SERIAL_AT_EVERY_MEASURED_DEPTH: bool = true;

pub fn kv_fp8_enabled() -> bool {
    match std::env::var(KV_FP8_ENV) {
        Ok(v) => !matches!(v.trim(), "0" | "off" | "false"),
        Err(_) => GOW_KV_FP8_DEFAULT_ON_THE_DEPTH_STACK_DOMINATES_SERIAL_AT_EVERY_MEASURED_DEPTH,
    }
}

const PF_KVQ_WGSL: &str = include_str!("../../nv-kernels/wgsl/q3m_pf_kvq.wgsl");

const MOE_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_moe.wgsl");

const PREFILL_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_prefill.wgsl");

const SPLICE_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_splice.wgsl");

const VERIFY_WGSL: &str = include_str!("../../nv-kernels/wgsl/gow_verify.wgsl");

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
    has_bias: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvMxParams {
    n_rows: u32,
    k_blocks: u32,
    groups_x: u32,
    has_bias: u32,
    w_e_stride_v4: u32,
    sf_e_stride_bytes: u32,
    bias_e_stride: u32,
    x_slot_stride_words: u32,
    y_slot_stride: u32,
    use_sel: u32,
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
struct RopeParams {
    n_rows: u32,
    head_dim: u32,
    rot_half: u32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FusedRopeParams {
    n_h: u32,
    n_kv: u32,
    head_dim: u32,
    rot_half: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvSgParams {
    n_rows: u32,
    k_v4: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_v4: u32,
    x_off_v4: u32,
    y_off_words: u32,
    has_bias: u32,
    alpha: f32,
    topk_k: u32,
    pad1: u32,
    pad2: u32,
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
    window: u32,
    pad0: u32,
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
    pad0: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SharedFdParams {
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
struct PfKvqParams {
    tokens: u32,
    x_stride_elems: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PackParams {
    n_words: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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
struct SwigluParams {
    n_words: u32,
    inter_words: u32,
    two_inter: u32,
    pad0: u32,
    limit: f32,
    alpha: f32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CombineParams {
    hidden_words: u32,
    k: u32,
    slot_stride: u32,
    pad0: u32,
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
struct PfGatherParams {
    row_off: u32,
    n_rows: u32,
    hidden_words: u32,
    vocab: u32,
    m: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfGemvBf16Params {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_row_words: u32,
    y_row_words: u32,
    has_bias: u32,
    alpha: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct PfGemvMxParams {
    n_rows: u32,
    k_blocks: u32,
    groups_x: u32,
    has_bias: u32,
    w_e_stride_v4: u32,
    sf_e_stride_bytes: u32,
    bias_e_stride: u32,
    x_slot_stride_words: u32,
    x_tok_stride_words: u32,
    y_slot_stride: u32,
    k_top: u32,
    pad0: u32,
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
struct PfCombineParams {
    hidden_words: u32,
    k: u32,
    slot_stride: u32,
    m: u32,
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
struct VerifyLmParams {
    n_rows: u32,
    k_words: u32,
    groups_x: u32,
    out_f32: u32,
    w_row_words: u32,
    x_row_words: u32,
    y_row_words: u32,
    has_bias: u32,
    alpha: f32,
    y_off_words: u32,
    pad0: u32,
    pad1: u32,
}

pub enum WeightSource<'a> {
    Host(&'a HostWeights),
    Loader(&'a nv_weights::WeightLoader),
}

fn bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    nv_kernels::wgpu_backend::pack::pack_u8_words_padded_to_multiple(bytes, 4)
}

struct Pass {
    pipeline: std::sync::Arc<wgpu::ComputePipeline>,
    bind: wgpu::BindGroup,
    grid: (u32, u32, u32),
    label: String,
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

pub use crate::wgpu_ledger::VramReport;

struct Builder {
    core: VramLedger,
    passes: Vec<Pass>,
    pf_passes: Vec<Pass>,
    v_passes: Vec<Pass>,
    to_prefill: bool,
    to_verify: bool,
    probes: Vec<(String, wgpu::Buffer, usize, bool)>,
    pf_kvq_pairs: usize,
    pf_kv_write_full_layers: usize,
    lanepos_alt_flash1_passes: Vec<Pass>,
    depthgate_flash1_pass_indices_full_attention_only_sliding_cost_does_not_scale_with_depth:
        Vec<usize>,
    storage_buffer_registry_indexes_every_hazard_tracked_buffer_the_graph_binds: Vec<wgpu::Buffer>,
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
        if self.to_verify {
            self.v_passes.push(pass);
        } else if self.to_prefill {
            self.pf_passes.push(pass);
        } else {
            self.passes.push(pass);
        }
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

    fn probe(&mut self, name: &str, buf: &wgpu::Buffer, elems: usize, bf16: bool) {
        if self.probes.iter().any(|(n, _, _, _)| n == name) {
            return;
        }
        self.probes
            .push((name.to_string(), buf.clone(), elems, bf16));
    }
}

struct Sources {
    gemv: String,
    mx: String,
    mx_entry: &'static str,
    mx_rows_per_wg: u32,
    attn: String,
    flash: String,
    flash_stage1_entry: &'static str,
    kvq: String,
    pf_kvq: String,
    moe: String,
    rms: String,
    rmsres: String,
    resscale: String,
    prefill: String,
    verify: String,
    fused: String,
    fused_on: bool,
    gemv_sg_on: bool,
    gemv_kpre_kv4s: Vec<usize>,
    mx_kpre_blocks: Vec<usize>,
    rmsres_sglat_on: bool,
    rmsres_sglat: String,
}

#[doc(hidden)]
pub fn verify_audit_sources() -> Vec<(&'static str, String)> {
    vec![
        ("gow:verify", compose(VERIFY_WGSL)),
        (
            "gow:prefill+splice",
            compose(&format!("{PREFILL_WGSL}\n{SPLICE_WGSL}")),
        ),
    ]
}

#[doc(hidden)]
pub fn nozi_audit_sources() -> Vec<(&'static str, String)> {
    vec![
        ("gow:gemv", compose(GEMV_WGSL)),
        ("gow:mx", compose(MX_WGSL)),
        ("gow:attn", compose(ATTN_WGSL)),
        ("gow:moe", compose(MOE_WGSL)),
    ]
}

impl Sources {
    fn new(ctx: &WgpuContext, cfg: &GptOssConfig) -> Self {
        let sg = mx_sg_reordered_sum_enabled() && wk::gemv_nvfp4::sg32_ok(ctx);
        let gemv_sg_on = gemv_sg_enabled() && wk::gemv_nvfp4::sg32_ok(ctx);
        let mut gemv_kpre_kv4s: Vec<usize> = Vec::new();
        if gemv_sg_kpre_enabled_baked_k_entries_issue_every_weight_load_before_the_fma_chain()
            && gemv_sg_on
        {
            for k in [cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim] {
                if k >= 8 && k.is_multiple_of(8) && !gemv_kpre_kv4s.contains(&(k / 8)) {
                    gemv_kpre_kv4s.push(k / 8);
                }
            }
        }
        let mut fused_src = String::from(FUSED_WGSL);
        for &kv4 in &gemv_kpre_kv4s {
            fused_src.push_str(&gemv_sg_kpre_source_issuing_every_weight_load_before_the_fma_chain(
                kv4,
            ));
        }
        let mut mx_kpre_blocks: Vec<usize> = Vec::new();
        if mx_kpre_enabled_baked_block_entries_issue_the_expert_weight_and_scale_stream_before_the_fma_chain()
            && sg
            && mx_sg_xv4_enabled()
            && mx_sg_byte_lut_enabled()
            && mx_sg_xstage_enabled()
        {
            for k in [cfg.hidden_size, cfg.intermediate_size] {
                if k >= MX_BLOCK
                    && k.is_multiple_of(MX_BLOCK)
                    && k / MX_BLOCK <= MX_XS_K_BLOCK_CAP_MATCHES_THE_512_VEC4_SHARED_ARRAY
                    && !mx_kpre_blocks.contains(&(k / MX_BLOCK))
                {
                    mx_kpre_blocks.push(k / MX_BLOCK);
                }
            }
        }
        let mx_base: &str = if sg && (mx_sg_xv4_r2_enabled() || mx_sg_xv4_enabled()) {
            MX_XV4_WGSL
        } else if sg {
            MX_WGSL
        } else {
            mx_scalar_section()
        };
        let mut mx_src = String::from(mx_base);
        for &blocks in &mx_kpre_blocks {
            mx_src.push_str(&mx_kpre_r2_xs_source_issuing_both_rows_streams_before_the_fma_chains(
                blocks,
            ));
            mx_src.push_str(&mx_kpre_r4sw_source_issuing_all_four_rows_streams_before_the_fma_chains(
                blocks,
            ));
        }
        Self {
            gemv: compose(GEMV_WGSL),
            mx: compose(&mx_src),
            mx_entry: match (
                sg,
                mx_sg_xv4_r2_enabled(),
                mx_sg_xv4_enabled(),
                mx_sg_byte_lut_enabled(),
            ) {
                (true, true, _, true) => MX_SG_XV4_R2_LUT_ENTRY,
                (true, true, _, false) => MX_SG_XV4_R2_ENTRY,
                (true, false, true, true)
                    if mx_sg_xstage_enabled()
                        && mx_sg_gupair_enabled_two_adjacent_rows_share_one_staged_x_so_the_interleaved_gate_up_pair_rides_one_x_pass() =>
                {
                    MX_SG_XV4_R2_LUT_XS_ENTRY
                }
                (true, false, true, true) if mx_sg_xstage_enabled() => MX_SG_XV4_LUT_XS_ENTRY,
                (true, false, true, true) => MX_SG_XV4_LUT_ENTRY,
                (true, false, true, false) => MX_SG_XV4_ENTRY,
                (true, false, false, true) => MX_SG_LUT_ENTRY,
                (true, false, false, false) => MX_SG_ENTRY,
                (false, ..) => MX_SCALAR_ENTRY,
            },
            mx_rows_per_wg: if sg
                && (mx_sg_xv4_r2_enabled()
                    || (mx_sg_xv4_enabled()
                        && mx_sg_byte_lut_enabled()
                        && mx_sg_xstage_enabled()
                        && mx_sg_gupair_enabled_two_adjacent_rows_share_one_staged_x_so_the_interleaved_gate_up_pair_rides_one_x_pass()))
            {
                MX_SG_XV4_R2_ROWS_PER_WG
            } else if sg {
                MX_SG_ROWS_PER_WG_AT_SG32_OVER_WG256
            } else {
                MX_SCALAR_ROWS_PER_WG
            },
            attn: compose(ATTN_WGSL),
            flash: compose(FLASH_WGSL),
            flash_stage1_entry: if wk::gemv_nvfp4::sg32_ok(ctx) {
                "gow_flash_stage1_sg"
            } else {
                "gow_flash_stage1"
            },
            kvq: compose(wk::kv_fp8::WGSL),
            pf_kvq: compose(&format!("{}\n{}", wk::kv_fp8::WGSL, PF_KVQ_WGSL)),
            moe: compose(MOE_WGSL),
            rms: compose(wk::rmsnorm::WGSL),
            rmsres: compose(wk::rmsnorm_residual::WGSL),
            resscale: compose(wk::residual_scale::WGSL),
            prefill: compose(&format!("{PREFILL_WGSL}\n{SPLICE_WGSL}")),
            verify: compose(VERIFY_WGSL),
            fused: compose(&fused_src),
            fused_on: fused_step_enabled(),
            gemv_sg_on,
            gemv_kpre_kv4s,
            mx_kpre_blocks,
            rmsres_sglat_on:
                rmsres_sglat_enabled_scratch_tree_collapses_to_one_barrier_plus_subgroup_shuffles()
                    && wk::gemv_nvfp4::sg32_ok(ctx),
            rmsres_sglat: compose(
                RMSRES_SGLAT_WGSL_REPRODUCES_THE_SHARED_TREE_ASSOCIATION_SO_THE_RMS_BITS_MATCH,
            ),
        }
    }
}

struct Bf16Gpu {
    w: wgpu::Buffer,
    bias: Option<wgpu::Buffer>,
    n: usize,
    k: usize,
}

struct MxGpu {
    w: wgpu::Buffer,
    scales: wgpu::Buffer,
    bias: wgpu::Buffer,
    e: usize,
    n: usize,
    k: usize,
}

struct VerifyState {
    rows: usize,
    head_start: usize,
    passes: Vec<Pass>,
    logits: wgpu::Buffer,
    tokens: wgpu::Buffer,
    validated: bool,
}

struct ChunkRowSplice<'a> {
    rel_pos: usize,
    row_words: &'a [u32],
}

pub struct GptOssWgpu {
    ctx: &'static WgpuContext,
    config: GptOssConfig,
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
    verify: Option<VerifyState>,
    head_start: usize,
    _buffers: Vec<wgpu::Buffer>,
    tok_buf: wgpu::Buffer,
    pos_buf: wgpu::Buffer,
    flash_fd: Option<(wgpu::Buffer, FlashDecodeParams)>,
    kv8_fd: Option<(wgpu::Buffer, SharedFdParams)>,
    kv8_gfd: Option<(wgpu::Buffer, FlashDecodeParams)>,
    depthgate_flash1_passes: Vec<usize>,
    depthgate_on: bool,
    lanepos_alt_flash1: Vec<Pass>,
    lanepos_gate_knee: Option<u32>,
    lanepos_deep_active: bool,
    depthgate_shallow_splits: u32,
    pf_kvq: Option<wgpu::Buffer>,
    flash_sw: Option<(wgpu::Buffer, FlashDecodeParams, u32)>,
    token_out: wgpu::Buffer,
    logits: wgpu::Buffer,
    state_buffers: Vec<(wgpu::Buffer, u64)>,
    probes: Vec<(String, wgpu::Buffer, usize, bool)>,
    vocab: usize,
    vram: VramReport,
    mx_entry: &'static str,
}

pub fn vram_report_enabled() -> bool {
    crate::wgpu_ledger::vram_report_var_enabled("NV_GPTOSS_WGPU_VRAM")
}

pub fn staging_flush_enabled() -> bool {
    !matches!(
        std::env::var("NV_GPTOSS_WGPU_STAGING_FLUSH")
            .ok()
            .as_deref(),
        Some("0") | Some("off") | Some("false")
    )
}

impl GptOssWgpu {
    pub fn config(&self) -> &GptOssConfig {
        &self.config
    }

    pub fn vram_report(&self) -> &VramReport {
        &self.vram
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn mx_gemv_entry(&self) -> &'static str {
        self.mx_entry
    }

    pub fn decode_pass_labels(&self) -> Vec<String> {
        self.passes.iter().map(|p| p.label.clone()).collect()
    }

    pub fn decode_pass_labels_and_grids_every_entry_is_one_dependent_dispatch(
        &self,
    ) -> Vec<(String, (u32, u32, u32))> {
        self.passes
            .iter()
            .map(|p| (p.label.clone(), p.grid))
            .collect()
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

    pub fn decode_head_start_passes_below_it_also_run_on_a_prefill_only_step(&self) -> usize {
        self.head_start
    }

    #[doc(hidden)]
    pub fn depthgate_full_attention_flash1_grid_y_as_last_encoded(&self) -> Option<u32> {
        self.depthgate_flash1_passes
            .first()
            .map(|&i| self.passes[i].grid.1)
    }

    #[doc(hidden)]
    pub fn full_attention_flash1_entry(&self) -> Option<String> {
        self.depthgate_flash1_passes
            .first()
            .map(|&i| self.passes[i].label.clone())
    }

    pub fn current_pos(&self) -> usize {
        self.pos
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
            "synthetic fill pos {pos} past max_seq {}; the flat kv caches index slots by pos \
             and the decode window start is a pure function of pos, so pos alone is the \
             cache-depth state",
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
        Ok(())
    }

    pub fn new(config: GptOssConfig, weights: &HostWeights, max_seq: usize) -> Result<Self> {
        Self::build(config, WeightSource::Host(weights), max_seq)
    }

    pub fn from_loader(
        config: GptOssConfig,
        weights: &nv_weights::WeightLoader,
        max_seq: usize,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Loader(weights), max_seq)
    }

    fn build(config: GptOssConfig, src: WeightSource<'_>, max_seq: usize) -> Result<Self> {
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
        let s = Sources::new(ctx, &config);
        let mx_entry = s.mx_entry;
        let cfg = &config;

        anyhow::ensure!(max_seq > 0, "max_seq must be positive");
        anyhow::ensure!(
            cfg.hidden_size.is_multiple_of(2 * MX_BLOCK),
            "hidden_size {} must be a multiple of {}",
            cfg.hidden_size,
            2 * MX_BLOCK
        );
        anyhow::ensure!(
            cfg.intermediate_size.is_multiple_of(MX_BLOCK),
            "intermediate_size {} must be a multiple of {MX_BLOCK}",
            cfg.intermediate_size
        );
        anyhow::ensure!(
            (2 * cfg.intermediate_size * cfg.hidden_size / MX_BLOCK).is_multiple_of(4)
                && (cfg.hidden_size * cfg.intermediate_size / MX_BLOCK).is_multiple_of(4),
            "expert scale bytes must align to whole u32 words"
        );
        anyhow::ensure!(
            cfg.num_local_experts <= 256,
            "router top-k kernel caps num_local_experts at 256, got {}",
            cfg.num_local_experts
        );
        anyhow::ensure!(
            cfg.num_experts_per_tok >= 1 && cfg.num_experts_per_tok <= MAX_TOPK,
            "num_experts_per_tok {} out of range 1..={MAX_TOPK}",
            cfg.num_experts_per_tok
        );
        anyhow::ensure!(
            cfg.head_dim <= MAX_HEAD_DIM && cfg.head_dim.is_multiple_of(4),
            "head_dim {} must be a multiple of 4 and <= {MAX_HEAD_DIM}",
            cfg.head_dim
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
        anyhow::ensure!(cfg.sliding_window > 0, "sliding_window must be positive");

        let hidden = cfg.hidden_size;
        let hidden_words = hidden / 2;
        let eps = cfg.rms_norm_eps as f32;
        let vocab = cfg.vocab_size;

        let mut b = Builder {
            core: VramLedger::new(ctx, "gow-", staging_flush_enabled, STAGING_FLUSH_BYTES),
            passes: Vec::new(),
            pf_passes: Vec::new(),
            v_passes: Vec::new(),
            to_prefill: false,
            to_verify: false,
            probes: Vec::new(),
            pf_kvq_pairs: 0,
            pf_kv_write_full_layers: 0,
            lanepos_alt_flash1_passes: Vec::new(),
            depthgate_flash1_pass_indices_full_attention_only_sliding_cost_does_not_scale_with_depth:
                Vec::new(),
            storage_buffer_registry_indexes_every_hazard_tracked_buffer_the_graph_binds: Vec::new(),
        };

        let tok_buf = b.upload_u32("gow-tok", &[0u32]);
        let pos_buf = b.upload_u32("gow-pos", &[0u32]);
        let flash_fd: Option<(wgpu::Buffer, FlashDecodeParams)> =
            flash_decode_enabled().then(|| {
                let p = FlashDecodeParams {
                    n_heads: cfg.num_attention_heads as u32,
                    n_kv: cfg.num_key_value_heads as u32,
                    head_dim: cfg.head_dim as u32,
                    splits: flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
                    total: 0,
                    start: 0,
                    pad0: 0,
                    scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                };
                (b.uni("gow-at-fl-p", p), p)
            });
        let kv8_geometry_ok = cfg.head_dim.is_multiple_of(32)
            && cfg.head_dim <= wk::flash_decode::MAX_HEAD_DIM;
        if kv_fp8_enabled() && !kv8_geometry_ok {
            eprintln!(
                "[gow-wgpu] kv-fp8 depth stack needs head_dim a multiple of 32 up to the fold \
                 generator's cap; head_dim {} keeps the serial bf16 decode arm",
                cfg.head_dim
            );
        }
        let kv8_shared: Option<Kv8Shared> = (kv_fp8_enabled() && kv8_geometry_ok)
            .then(|| -> Result<Kv8Shared> {
                let group = cfg.num_attention_heads / cfg.num_key_value_heads;
                let fold = {
                    let f = if nv_config::env_arm_present_at_all_reread_every_call(
                        wk::flash_decode::GQA_FOLD_ENV,
                    ) {
                        wk::flash_decode::gqa_fold_env(group)
                    } else {
                        group.min(wk::flash_decode::MAX_GQA_FOLD)
                    };
                    if f > 1
                        && group.is_multiple_of(f)
                        && cfg.num_attention_heads.is_multiple_of(f)
                    {
                        f
                    } else {
                        1
                    }
                };
                let sg = ctx.caps.subgroup && ctx.subgroup_width() == Some(32);
                let splits =
                    flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build();
                let fd = SharedFdParams {
                    n_heads: cfg.num_attention_heads as u32,
                    n_kv: cfg.num_key_value_heads as u32,
                    head_dim: cfg.head_dim as u32,
                    splits,
                    scaling: 1.0 / (cfg.head_dim as f32).sqrt(),
                    ..Default::default()
                };
                let gfd = FlashDecodeParams {
                    n_heads: cfg.num_attention_heads as u32,
                    n_kv: cfg.num_key_value_heads as u32,
                    head_dim: cfg.head_dim as u32,
                    splits,
                    scale: 1.0 / (cfg.head_dim as f32).sqrt(),
                    ..Default::default()
                };
                let hd32 = cfg.head_dim as u32;
                let mut kv8_fold_stage1_source = format!(
                    "{}\n{}",
                    wk::flash_decode::WGSL,
                    wk::flash_decode::fold_stage1_source_sd(hd32, sg, fold as u32)
                );
                let mut kv8_fold_stage1_entry =
                    wk::flash_decode::fold_stage1_entry_sd(hd32, sg, fold as u32);
                anyhow::ensure!(
                    [
                        flash_kpre_banked_negative_preloading_four_rounds_costs_more_warp_residency_than_the_loads_in_flight_buy(),
                        flash_rowpair_banked_neutral_two_rows_per_warp_round_fills_the_idle_lanes_and_halves_the_rounds_for_no_wall(),
                        flash_acc_stage1_hoists_every_loop_invariant_uniform_out_of_the_round_loop(),
                        flash_kvline_one_warp_per_kv_head_makes_a_workgroup_read_the_whole_slot_stride_of_a_position(),
                    ]
                    .into_iter()
                    .filter(|on| *on)
                    .count()
                        <= 1,
                    "{FLASH_KPRE_ENV}, {FLASH_ROWPAIR_ENV}, {FLASH_ACC_ENV} and \
                     {FLASH_KVLINE_ENV} each replace the stage1 entry with a different \
                     lane-to-operand partition; set at most one"
                );
                let mut stage1_grid_x = (cfg.num_attention_heads / fold) as u32;
                if flash_kvline_one_warp_per_kv_head_makes_a_workgroup_read_the_whole_slot_stride_of_a_position()
                    && sg
                    && hd32.is_multiple_of(32)
                    && hd32 <= FLASH_ACC_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS
                {
                    let nkv = cfg.num_key_value_heads as u32;
                    let slots = flash_kvline_slots();
                    let invocations = flash_kvline_workgroup_invocations(nkv, slots);
                    let storage = flash_kvline_workgroup_storage_bytes(hd32, nkv, group as u32, slots);
                    anyhow::ensure!(
                        ctx.caps.max_compute_invocations_per_workgroup >= invocations
                            && ctx.caps.max_compute_workgroup_size_x >= invocations
                            && ctx.caps.workgroup_storage_fits(storage),
                        "{FLASH_KVLINE_ENV} needs a {invocations}-invocation workgroup and \
                         {storage} workgroup bytes because one workgroup stages every query head \
                         and gives one warp to each of the {nkv} kv heads; device allows {} \
                         invocations / {} bytes",
                        ctx.caps.max_compute_invocations_per_workgroup,
                        ctx.caps.max_compute_workgroup_storage_size
                    );
                    kv8_fold_stage1_source.push('\n');
                    kv8_fold_stage1_source.push_str(&flash_kvline_stage1_source_sd(
                        hd32,
                        nkv,
                        group as u32,
                        slots,
                    ));
                    kv8_fold_stage1_entry =
                        flash_kvline_stage1_entry_sd(hd32, nkv, group as u32, slots);
                    stage1_grid_x = 1;
                } else if flash_acc_stage1_hoists_every_loop_invariant_uniform_out_of_the_round_loop()
                    && sg
                    && hd32.is_multiple_of(32)
                    && hd32 <= FLASH_ACC_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS
                {
                    let qreg = flash_qreg_the_folded_query_rows_are_loop_invariant_and_cost_one_workgroup_load_per_component_per_head_per_round();
                    let smfast = flash_smfast_a_running_max_that_does_not_move_makes_every_rescale_an_exact_multiply_by_one();
                    let wgacc = flash_wgacc_the_per_thread_accumulator_file_moves_to_workgroup_memory_to_buy_resident_warps();
                    kv8_fold_stage1_source.push('\n');
                    kv8_fold_stage1_source.push_str(&flash_acc_stage1_source_sd(
                        hd32,
                        fold as u32,
                        qreg,
                        smfast,
                        wgacc,
                    ));
                    kv8_fold_stage1_entry =
                        flash_acc_stage1_entry_sd(hd32, fold as u32, qreg, smfast, wgacc);
                } else if flash_lanepos_one_lane_owns_one_positions_whole_dot_so_no_per_position_reduce_exists()
                    && sg
                    && hd32.is_multiple_of(32)
                {
                    kv8_fold_stage1_source.push('\n');
                    if lanepos_f16dot_enabled() && ctx.caps.shader_f16 {
                        kv8_fold_stage1_source.push_str(
                            &wk::flash_decode::fold_stage1_source_lanepos_h2_sd(
                                hd32,
                                fold as u32,
                            ),
                        );
                        kv8_fold_stage1_entry =
                            wk::flash_decode::fold_stage1_entry_lanepos_h2_sd(
                                hd32,
                                fold as u32,
                            );
                    } else {
                        kv8_fold_stage1_source.push_str(
                            &wk::flash_decode::fold_stage1_source_lanepos_sd(hd32, fold as u32),
                        );
                        kv8_fold_stage1_entry =
                            wk::flash_decode::fold_stage1_entry_lanepos_sd(hd32, fold as u32);
                    }
                } else if flash_blksm_banked_negative_the_4x_score_and_scale_register_file_costs_more_residency_than_the_shared_softmax_update_buys()
                    && sg
                    && hd32.is_multiple_of(4)
                {
                    kv8_fold_stage1_source.push('\n');
                    kv8_fold_stage1_source.push_str(
                        &wk::flash_decode::fold_stage1_source_blksm_sd(hd32, fold as u32),
                    );
                    kv8_fold_stage1_entry =
                        wk::flash_decode::fold_stage1_entry_blksm_sd(hd32, fold as u32);
                } else if flash_rowpair_banked_neutral_two_rows_per_warp_round_fills_the_idle_lanes_and_halves_the_rounds_for_no_wall()
                    && sg
                    && hd32.is_multiple_of(
                        4 * FLASH_ROWPAIR_LANES_PER_ROW_16_IS_HEAD_DIM_64_PACKED_FOUR_FP8_TO_A_WORD,
                    )
                {
                    kv8_fold_stage1_source.push('\n');
                    kv8_fold_stage1_source
                        .push_str(&flash_rowpair_stage1_source_sd(hd32, fold as u32));
                    kv8_fold_stage1_entry = flash_rowpair_stage1_entry_sd(hd32, fold as u32);
                } else if (flash_kpre_banked_negative_preloading_four_rounds_costs_more_warp_residency_than_the_loads_in_flight_buy()
                    || flash_blocksplit_banked_negative_cyclic_splits_sweep_the_kv_stream_in_lockstep_and_contiguous_blocks_do_not())
                    && sg
                    && hd32.is_multiple_of(32)
                    && hd32 <= FLASH_KPRE_STAGE1_IS_SPECIALIZED_TO_HEAD_DIM_AT_MOST_FOUR_LANE_WORDS
                {
                    let unroll =
                        if flash_kpre_banked_negative_preloading_four_rounds_costs_more_warp_residency_than_the_loads_in_flight_buy()
                        {
                            flash_kpre_unroll()
                        } else {
                            1
                        };
                    let blk =
                        flash_blocksplit_banked_negative_cyclic_splits_sweep_the_kv_stream_in_lockstep_and_contiguous_blocks_do_not();
                    kv8_fold_stage1_source.push('\n');
                    kv8_fold_stage1_source.push_str(&flash_kpre_stage1_source_sd(
                        hd32,
                        fold as u32,
                        unroll,
                        blk,
                    ));
                    kv8_fold_stage1_entry =
                        flash_kpre_stage1_entry_sd(hd32, fold as u32, unroll, blk);
                }
                Ok(Kv8Shared {
                    fd_buf: b.uni("gow-at-fd8-p", fd),
                    fd_base: fd,
                    gfd_buf: b.uni("gow-at-fd8-s2-p", gfd),
                    gfd_base: gfd,
                    lanepos_alt_entry: match flash_lanepos_mode_unset_gates_at_the_measured_knee_1_always_0_restores_fdf_only()
                    {
                        FlashLaneposMode::Gate(_) if sg && hd32.is_multiple_of(32) => {
                            kv8_fold_stage1_source.push('\n');
                            kv8_fold_stage1_source.push_str(
                                &wk::flash_decode::fold_stage1_source_lanepos_sd(
                                    hd32,
                                    fold as u32,
                                ),
                            );
                            Some(wk::flash_decode::fold_stage1_entry_lanepos_sd(
                                hd32,
                                fold as u32,
                            ))
                        }
                        _ => None,
                    },
                    fold_src: {
                        let c = compose(&kv8_fold_stage1_source);
                        if lanepos_f16dot_enabled()
                            && ctx.caps.shader_f16
                            && kv8_fold_stage1_entry.contains("_h2")
                        {
                            format!("enable f16;\n{c}")
                        } else {
                            c
                        }
                    },
                    fold_entry: kv8_fold_stage1_entry,
                    splits,
                    scratch_splits:
                        flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(
                            splits,
                        ),
                    stage1_grid_x_is_one_when_a_workgroup_covers_every_kv_head: stage1_grid_x,
                })
            })
            .transpose()?;
        let flash_sw: Option<(wgpu::Buffer, FlashDecodeParams, u32)> = match &flash_fd {
            Some((_, base))
                if flash_sliding_enabled()
                    && cfg
                        .layer_types
                        .iter()
                        .any(|t| matches!(t, GptOssLayerType::Sliding)) =>
            {
                let mut p = *base;
                p.splits = p
                    .splits
                    .min((cfg.sliding_window as u32).div_ceil(8))
                    .max(1);
                Some((b.uni("gow-at-fl-sw-p", p), p, cfg.sliding_window as u32))
            }
            _ => None,
        };
        let mut pf_m = prefill_m();
        if pf_m > max_seq {
            pf_m = 0;
        }
        let pf = (pf_m > 0).then(|| alloc_pf_scratch(&mut b, cfg, max_seq, pf_m));
        let pf_kvq_buf = (kv8_shared.is_some() && pf.is_some()).then(|| {
            b.uni(
                "gow-pf-kvq-mp",
                PfKvqParams {
                    tokens: 0,
                    x_stride_elems: (cfg.num_key_value_heads * cfg.head_dim) as u32,
                    ..Default::default()
                },
            )
        });

        let res = b.zeros("gow-res", (hidden_words * 4) as u64);
        let res2 = b.zeros("gow-res2", (hidden_words * 4) as u64);
        let normed = b.zeros("gow-normed", (hidden_words * 4) as u64);
        let mixed_out = b.zeros("gow-mix", (hidden_words * 4) as u64);
        let moe_out = b.zeros("gow-moeout", (hidden_words * 4) as u64);

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
                "gow-embed",
                &pack_pairs(&embed[off * hidden..(off + rows) * hidden]),
            );
            let p = b.uni(
                "gow-embed-p",
                GatherParams {
                    row_off: off as u32,
                    n_rows: rows as u32,
                    hidden_words: hidden_words as u32,
                    vocab: vocab as u32,
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.push(
                "gow-gather",
                &s.moe,
                "gow_gather_embed",
                &[(30, &buf), (31, &tok_buf), (32, &res), (33, &p)],
                grid,
            )?;
            if let Some(sc) = &pf {
                let pp = b.uni(
                    "gow-pf-embed-p",
                    PfGatherParams {
                        row_off: off as u32,
                        n_rows: rows as u32,
                        hidden_words: hidden_words as u32,
                        vocab: vocab as u32,
                        m: pf_m as u32,
                        ..Default::default()
                    },
                );
                b.to_prefill = true;
                b.push(
                    "gow-pf-gather",
                    &s.prefill,
                    "gow_pf_gather_embed",
                    &[(0, &buf), (1, &sc.tok), (2, &sc.res), (3, &pp)],
                    ((hidden_words as u32).div_ceil(256), pf_m as u32, 1),
                )?;
                b.to_prefill = false;
            }
            off += rows;
        }
        drop(embed);

        if let Some(sc) = &pf {
            let sp = b.uni(
                "gow-pf-splice-p",
                SpliceParams {
                    hidden_words: hidden_words as u32,
                    m: pf_m as u32,
                    ..Default::default()
                },
            );
            let grid = b.grid1(hidden_words as u64, 256);
            b.to_prefill = true;
            b.push(
                "gow-pf-splice",
                &s.prefill,
                "gow_pf_splice_embed_rows",
                &[(2, &sc.res), (4, &sc.splice), (5, &sc.mask), (6, &sp)],
                (grid.0, pf_m as u32, 1),
            )?;
            b.to_prefill = false;
        }

        let (cos, sin) = rope_tables(cfg, max_seq);
        let cosb = b.upload_f32("gow-cos", &cos);
        let sinb = b.upload_f32("gow-sin", &sin);

        let mut state_buffers: Vec<(wgpu::Buffer, u64)> = Vec::new();

        for li in 0..cfg.num_hidden_layers {
            let layer = src.layer(cfg, li)?;
            let ln_w = b.upload_u32("gow-ln", &pack_pairs(&layer.input_ln));
            if li == 0 {
                let p = b.uni(
                    "gow-rms-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: 1,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                b.push(
                    "gow-rms0",
                    &s.rms,
                    "rmsnorm_bf16",
                    &[(0, &res), (1, &ln_w), (2, &normed), (3, &p)],
                    (1, 1, 1),
                )?;
            }

            let ag = build_attn(
                &mut b,
                &s,
                cfg,
                &layer,
                li,
                &normed,
                &mixed_out,
                &pos_buf,
                &cosb,
                &sinb,
                max_seq,
                &mut state_buffers,
                flash_fd.as_ref().map(|(buf, _)| buf),
                kv8_shared.as_ref(),
                flash_sw.as_ref().map(|(buf, p, _)| (buf, p.splits)),
            )?;

            let post_w = b.upload_u32("gow-post-ln", &pack_pairs(&layer.post_attn_ln));
            let normed_post = b.zeros("gow-normed-post", (hidden_words * 4) as u64);
            let rp = b.uni(
                "gow-rmsres-p",
                RmsParams {
                    hidden: hidden as u32,
                    batch: 1,
                    eps,
                    words_per_row: hidden_words as u32,
                },
            );
            let (rmsres_src, rmsres_entry) = if s.rmsres_sglat_on {
                (&s.rmsres_sglat, RMSRES_SGLAT_ENTRY)
            } else {
                (&s.rmsres, "rmsnorm_residual_bf16")
            };
            b.push(
                "gow-rmsres-post",
                rmsres_src,
                rmsres_entry,
                &[
                    (0, &mixed_out),
                    (1, &res),
                    (2, &post_w),
                    (3, &normed_post),
                    (4, &rp),
                ],
                (1, 1, 1),
            )?;

            let mg = build_moe(&mut b, &s, cfg, &layer.moe, &normed_post, &moe_out)?;

            if let Some(sc) = &pf {
                b.to_prefill = true;
                if li == 0 {
                    pf_norm(
                        &mut b,
                        &s,
                        "gow-pf-rms0",
                        &sc.res,
                        &ln_w,
                        &sc.normed,
                        hidden,
                        eps,
                        pf_m,
                    )?;
                }
                build_attn_prefill(
                    &mut b,
                    &s,
                    cfg,
                    &ag,
                    sc,
                    &cosb,
                    &sinb,
                    &sc.normed,
                    &sc.mixed,
                    max_seq,
                    &pos_buf,
                    pf_kvq_buf.as_ref(),
                )?;
                pf_norm_residual(
                    &mut b,
                    &s,
                    "gow-pf-rmsres-post",
                    &sc.mixed,
                    &sc.res,
                    &post_w,
                    &sc.normed_post,
                    hidden,
                    eps,
                    pf_m,
                )?;
                build_moe_prefill(&mut b, &s, cfg, &mg, sc, &sc.normed_post, &sc.moe_out)?;
                b.to_prefill = false;
            }

            if li + 1 < cfg.num_hidden_layers {
                let next = src.layer_input_ln(cfg, li + 1)?;
                let nw = b.upload_u32("gow-next-ln", &pack_pairs(&next));
                let rp2 = b.uni(
                    "gow-rmsres2-p",
                    RmsParams {
                        hidden: hidden as u32,
                        batch: 1,
                        eps,
                        words_per_row: hidden_words as u32,
                    },
                );
                let (rmsres2_src, rmsres2_entry) = if s.rmsres_sglat_on {
                    (&s.rmsres_sglat, RMSRES_SGLAT_ENTRY)
                } else {
                    (&s.rmsres, "rmsnorm_residual_bf16")
                };
                b.push(
                    "gow-rmsres-next",
                    rmsres2_src,
                    rmsres2_entry,
                    &[(0, &moe_out), (1, &res), (2, &nw), (3, &normed), (4, &rp2)],
                    (1, 1, 1),
                )?;
                if let Some(sc) = &pf {
                    b.to_prefill = true;
                    pf_norm_residual(
                        &mut b,
                        &s,
                        "gow-pf-rmsres-next",
                        &sc.moe_out,
                        &sc.res,
                        &nw,
                        &sc.normed,
                        hidden,
                        eps,
                        pf_m,
                    )?;
                    b.to_prefill = false;
                }
            } else {
                let sp = b.uni(
                    "gow-resadd-p",
                    ResScaleParams {
                        n: hidden as u32,
                        n_words: hidden_words as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let grid = b.grid1(hidden_words as u64, 256);
                b.push(
                    "gow-resadd",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &moe_out), (1, &res), (2, &res2), (3, &sp)],
                    grid,
                )?;
            }
        }

        let final_w = b.upload_u32("gow-final-ln", &pack_pairs(&src.final_norm(cfg)?));
        let final_x = b.zeros("gow-final-x", (hidden_words * 4) as u64);
        let fp = b.uni(
            "gow-final-p",
            RmsParams {
                hidden: hidden as u32,
                batch: 1,
                eps,
                words_per_row: hidden_words as u32,
            },
        );
        b.push(
            "gow-final-rms",
            &s.rms,
            "rmsnorm_bf16",
            &[(0, &res2), (1, &final_w), (2, &final_x), (3, &fp)],
            (1, 1, 1),
        )?;

        let head_start = b.passes.len();
        let logits = b.zeros("gow-logits", (vocab * 4) as u64);
        let lm = src.lm_head(cfg)?;
        anyhow::ensure!(
            lm.len() == vocab * hidden,
            "lm_head has {} values, want {}",
            lm.len(),
            vocab * hidden
        );
        let dummy_bias = b.upload_u32("gow-dummy-bias", &[0u32]);
        let head_i8 = lmhead_i8_enabled_rowwise_int8_head_halves_vocab_gemv_bytes_and_is_ppl_gated_not_bitwise()
            && s.gemv_sg_on
            && hidden.is_multiple_of(16);
        let mut lm_chunks: Vec<(wgpu::Buffer, usize, usize)> = Vec::new();
        let mut off = 0usize;
        while head_i8 && off < vocab {
            let rows = (chunk_rows * 2).min(vocab - off);
            let (packed, scales) = wk::quant_gemv::quantize_rows_int8(
                &lm[off * hidden..(off + rows) * hidden],
                rows,
                hidden,
            );
            let wbuf = b.upload_u32("gow-lmhead-i8-w", &packed);
            let sbuf = b.upload_f32("gow-lmhead-i8-s", &scales);
            let groups = rows.div_ceil(8);
            let grid = b.grid1(groups as u64, 1);
            let p = b.uni(
                "gow-lmhead-i8-p",
                GemvSgParams {
                    n_rows: rows as u32,
                    k_v4: (hidden / 8) as u32,
                    groups_x: grid.0,
                    out_f32: 1,
                    w_row_v4: (hidden / 16) as u32,
                    x_off_v4: 0,
                    y_off_words: off as u32,
                    has_bias: 0,
                    alpha: 1.0,
                    ..Default::default()
                },
            );
            b.push(
                "gow-lmhead",
                &s.fused,
                LMHEAD_I8_ENTRY,
                &[(30, &wbuf), (31, &final_x), (32, &p), (33, &logits), (37, &sbuf)],
                grid,
            )?;
            off += rows;
        }
        while off < vocab {
            let rows = (chunk_rows.min(vocab - off)) & !1usize;
            let rows = if rows == 0 { vocab - off } else { rows };
            let wbuf = b.upload_u32(
                "gow-lmhead",
                &pack_pairs(&lm[off * hidden..(off + rows) * hidden]),
            );
            lm_chunks.push((wbuf.clone(), off, rows));
            if s.gemv_sg_on && hidden.is_multiple_of(8) {
                let (entry, rows_per_wg) = gemv_bf16_sg_entry_and_rows_per_wg(&s, hidden / 8);
                let groups = rows.div_ceil(rows_per_wg);
                let grid = b.grid1(groups as u64, 1);
                let p = b.uni(
                    "gow-lmhead-sg-p",
                    GemvSgParams {
                        n_rows: rows as u32,
                        k_v4: (hidden / 8) as u32,
                        groups_x: grid.0,
                        out_f32: 1,
                        w_row_v4: (hidden / 8) as u32,
                        x_off_v4: 0,
                        y_off_words: off as u32,
                        has_bias: 0,
                        alpha: 1.0,
                        ..Default::default()
                    },
                );
                b.push(
                    "gow-lmhead",
                    &s.fused,
                    &entry,
                    &[
                        (30, &wbuf),
                        (31, &final_x),
                        (32, &p),
                        (33, &logits),
                        (34, &dummy_bias),
                    ],
                    grid,
                )?;
            } else {
                let pairs = rows.div_ceil(2);
                let grid = b.grid1(pairs as u64, 1);
                let p = b.uni(
                    "gow-lmhead-p",
                    GemvBf16Params {
                        n_rows: rows as u32,
                        k_words: hidden_words as u32,
                        groups_x: grid.0,
                        out_f32: 1,
                        w_row_words: hidden_words as u32,
                        x_off_words: 0,
                        y_off_words: off as u32,
                        has_bias: 0,
                        alpha: 1.0,
                        ..Default::default()
                    },
                );
                b.push(
                    "gow-lmhead",
                    &s.gemv,
                    "gow_gemv_bf16",
                    &[
                        (0, &wbuf),
                        (1, &final_x),
                        (2, &p),
                        (3, &logits),
                        (4, &dummy_bias),
                    ],
                    grid,
                )?;
            }
            off += rows;
        }
        drop(lm);

        let pv = b.zeros("gow-am-pv", (ARGMAX_GROUPS * 4) as u64);
        let pi = b.zeros("gow-am-pi", (ARGMAX_GROUPS * 4) as u64);
        let token_out = b.zeros("gow-token", 4);
        let ap = b.uni(
            "gow-am-p",
            ArgmaxParams {
                n: vocab as u32,
                groups: ARGMAX_GROUPS as u32,
                ..Default::default()
            },
        );
        b.push(
            "gow-am1",
            &s.moe,
            "gow_argmax_stage1",
            &[(40, &logits), (41, &pv), (42, &pi), (44, &ap)],
            (ARGMAX_GROUPS as u32, 1, 1),
        )?;
        b.push(
            "gow-am2",
            &s.moe,
            "gow_argmax_stage2",
            &[(41, &pv), (42, &pi), (43, &token_out), (44, &ap)],
            (1, 1, 1),
        )?;

        let verify = match pf.as_ref() {
            Some(sc)
                if !b.pf_passes.is_empty()
                    && verify_rows(sc.m) > 0
                    && !lm_chunks.is_empty() =>
            {
                let rows = verify_rows(sc.m);
                b.to_verify = true;
                let v_res2 = b.zeros("gow-v-res2", (rows * hidden_words * 4) as u64);
                let v_final_x = b.zeros("gow-v-finalx", (rows * hidden_words * 4) as u64);
                let v_logits = b.zeros("gow-v-logits", (rows * vocab * 4) as u64);
                let v_pv = b.zeros("gow-v-am-pv", (rows * ARGMAX_GROUPS * 4) as u64);
                let v_pi = b.zeros("gow-v-am-pi", (rows * ARGMAX_GROUPS * 4) as u64);
                let v_tokens = b.zeros("gow-v-tokens", (rows * 4) as u64);
                let rsp = b.uni(
                    "gow-v-resadd-p",
                    ResScaleParams {
                        n: (rows * hidden) as u32,
                        n_words: (rows * hidden_words) as u32,
                        scale: 1.0,
                        ..Default::default()
                    },
                );
                let grid = b.grid1((rows * hidden_words) as u64, 256);
                b.push(
                    "gow-v-resadd",
                    &s.resscale,
                    "residual_add_scale_bf16",
                    &[(0, &sc.moe_out), (1, &sc.res), (2, &v_res2), (3, &rsp)],
                    grid,
                )?;
                pf_norm(
                    &mut b,
                    &s,
                    "gow-v-final-rms",
                    &v_res2,
                    &final_w,
                    &v_final_x,
                    hidden,
                    eps,
                    rows,
                )?;
                let head_start = b.v_passes.len();
                for (wbuf, off, n_rows) in &lm_chunks {
                    let pairs = n_rows.div_ceil(2);
                    let grid = b.grid1(pairs as u64, 1);
                    let p = b.uni(
                        "gow-v-lmhead-p",
                        VerifyLmParams {
                            n_rows: *n_rows as u32,
                            k_words: hidden_words as u32,
                            groups_x: grid.0,
                            out_f32: 1,
                            w_row_words: hidden_words as u32,
                            x_row_words: hidden_words as u32,
                            y_row_words: vocab as u32,
                            has_bias: 0,
                            alpha: 1.0,
                            y_off_words: *off as u32,
                            ..Default::default()
                        },
                    );
                    b.push(
                        "gow-v-lmhead",
                        &s.verify,
                        "gow_v_lmhead",
                        &[
                            (0, wbuf),
                            (1, &v_final_x),
                            (2, &p),
                            (3, &v_logits),
                            (4, &dummy_bias),
                        ],
                        (grid.0, grid.1, rows as u32),
                    )?;
                }
                let vap = b.uni(
                    "gow-v-am-p",
                    ArgmaxParams {
                        n: vocab as u32,
                        groups: ARGMAX_GROUPS as u32,
                        ..Default::default()
                    },
                );
                b.push(
                    "gow-v-am1",
                    &s.verify,
                    "gow_v_argmax_stage1",
                    &[(10, &v_logits), (11, &v_pv), (12, &v_pi), (14, &vap)],
                    (ARGMAX_GROUPS as u32, 1, rows as u32),
                )?;
                b.push(
                    "gow-v-am2",
                    &s.verify,
                    "gow_v_argmax_stage2",
                    &[(11, &v_pv), (12, &v_pi), (13, &v_tokens), (14, &vap)],
                    (1, 1, rows as u32),
                )?;
                b.to_verify = false;
                Some(VerifyState {
                    rows,
                    head_start,
                    passes: std::mem::take(&mut b.v_passes),
                    logits: v_logits,
                    tokens: v_tokens,
                    validated: false,
                })
            }
            _ => None,
        };

        if kv8_shared.is_some() && !b.pf_passes.is_empty() {
            anyhow::ensure!(
                b.pf_kvq_pairs == b.pf_kv_write_full_layers,
                "{KV_FP8_ENV} decode reads the fp8 cache for every full-attention row, so every \
                 prefill arm that writes full-layer KV must also record its quantize pair; \
                 counted {} quantize pairs against {} full-layer prefill kv writes -- an arm \
                 that skips the pair would leave chunk rows unread by decode",
                b.pf_kvq_pairs,
                b.pf_kv_write_full_layers
            );
        }
        let nulldisp = null_dispatch_repeat_prices_a_dependent_dispatch_inside_this_graph_instead_of_a_microbench();
        if nulldisp > 0 {
            b.to_prefill = false;
            b.to_verify = false;
            let scratch = b.zeros("gow-nulldisp-scratch", 4);
            let encoded = b.passes.len();
            for _ in 0..(encoded * nulldisp) {
                b.push(
                    "gow-nulldisp",
                    &s.moe,
                    NULL_DISPATCH_ENTRY,
                    &[(50, &scratch)],
                    (1, 1, 1),
                )?;
            }
        }

        b.flush_staging();
        let vram = b.report();
        if vram_report_enabled() {
            eprint!("[gow-wgpu] {}", vram.render());
        }

        let Builder {
            core,
            passes,
            pf_passes,
            probes,
            depthgate_flash1_pass_indices_full_attention_only_sliding_cost_does_not_scale_with_depth:
                depthgate_flash1_passes,
            ..
        } = b;
        let buffers = core.buffers;

        let (pf_passes, pf_m, pf_tok, pf_ck, pf_splice, pf_mask) = match pf {
            Some(sc) if !pf_passes.is_empty() => (
                pf_passes,
                sc.m,
                Some(sc.tok.clone()),
                Some(sc.ck.clone()),
                Some(sc.splice.clone()),
                Some(sc.mask.clone()),
            ),
            _ => (Vec::new(), 0, None, None, None, None),
        };
        let verify = if pf_m == 0 { None } else { verify };

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
            verify,
            head_start,
            _buffers: buffers,
            probes,
            tok_buf,
            pos_buf,
            flash_fd,
            kv8_fd: kv8_shared
                .as_ref()
                .map(|k8| (k8.fd_buf.clone(), k8.fd_base)),
            kv8_gfd: kv8_shared
                .as_ref()
                .map(|k8| (k8.gfd_buf.clone(), k8.gfd_base)),
            depthgate_flash1_passes,
            depthgate_on: flash_depthgate_enabled_shallow_tuned_splits_starve_the_grid_at_depth(),
            lanepos_alt_flash1: b.lanepos_alt_flash1_passes,
            lanepos_gate_knee: match flash_lanepos_mode_unset_gates_at_the_measured_knee_1_always_0_restores_fdf_only()
            {
                FlashLaneposMode::Gate(k) => Some(k),
                _ => None,
            },
            lanepos_deep_active: false,
            depthgate_shallow_splits: flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
            pf_kvq: pf_kvq_buf,
            flash_sw,
            token_out,
            logits,
            state_buffers,
            vocab,
            vram,
            mx_entry,
        })
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
        if let Some(knee) = self.lanepos_gate_knee {
            let want_deep = (self.pos + 1) as u32
                >= knee
                && !self.lanepos_alt_flash1.is_empty();
            if want_deep != self.lanepos_deep_active {
                for (slot, i) in self.depthgate_flash1_passes.iter().enumerate() {
                    std::mem::swap(&mut self.passes[*i], &mut self.lanepos_alt_flash1[slot]);
                }
                self.lanepos_deep_active = want_deep;
            }
        }
        let depthgate_splits = (self.depthgate_on
            && !self.depthgate_flash1_passes.is_empty())
        .then(|| {
            flash_depthgate_splits_for_context_tokens(
                (self.pos + 1) as u32,
                self.depthgate_shallow_splits,
            )
        });
        if let Some((buf, base)) = &self.flash_fd {
            let mut p = *base;
            p.total = (self.pos + 1) as u32;
            if let Some(sp) = depthgate_splits {
                p.splits = sp;
            }
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }
        if let Some((buf, base)) = &self.kv8_fd {
            let mut p = *base;
            p.total = (self.pos + 1) as u32;
            if let Some(sp) = depthgate_splits {
                p.splits = sp;
            }
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }
        if let (Some(sp), Some((buf, base))) = (depthgate_splits, &self.kv8_gfd) {
            let mut p = *base;
            p.splits = sp;
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }
        if let Some(sp) = depthgate_splits {
            for i in 0..self.depthgate_flash1_passes.len() {
                let pass = self.depthgate_flash1_passes[i];
                self.passes[pass].grid.1 = sp;
            }
        }
        if let Some((buf, base, window)) = &self.flash_sw {
            let mut p = *base;
            let total = (self.pos + 1) as u32;
            p.total = total;
            p.start = total.saturating_sub(*window);
            self.ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }

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
        } else if let Some(k) = dispatch::submit_chunk_overlapping_encode_with_execution() {
            for chunk in passes.chunks(k) {
                let mut enc = self.ctx.device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_compute_pass(&Default::default());
                    for p in chunk {
                        pass.set_pipeline(&p.pipeline);
                        pass.set_bind_group(0, &p.bind, &[]);
                        pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                    }
                }
                self.ctx.queue.submit([enc.finish()]);
            }
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
                anyhow::bail!("gpt_oss_wgpu decode step validation: {e}");
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
        if self.pf_passes.is_empty() {
            0
        } else {
            self.pf_m
        }
    }

    pub fn prefill_pass_count(&self) -> usize {
        self.pf_passes.len()
    }

    fn write_chunk_inputs(
        &mut self,
        chunk: &[u32],
        live: usize,
        splices: &[ChunkRowSplice],
    ) -> Result<()> {
        let m = self.pf_m;
        anyhow::ensure!(
            chunk.len() == m && (1..=m).contains(&live),
            "prefill chunk is {} tokens with {live} live, want {m} with 1..={m} live",
            chunk.len()
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
            anyhow::ensure!(
                sp.rel_pos < live,
                "splice rel_pos {} is not a live row of this chunk (live {live})",
                sp.rel_pos
            );
            anyhow::ensure!(
                sp.row_words.len() == hidden_words,
                "splice row has {} words, want {hidden_words}",
                sp.row_words.len()
            );
            mask[sp.rel_pos] = 1;
            let splice = self
                .pf_splice
                .as_ref()
                .expect("splice prefill without pf list");
            self.ctx.queue.write_buffer(
                splice,
                (sp.rel_pos * hidden_words * 4) as u64,
                bytemuck::cast_slice(sp.row_words),
            );
        }
        let mask_buf = self
            .pf_mask
            .as_ref()
            .expect("prefill chunk without pf list");
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
        if let Some(pf_kvq) = &self.pf_kvq {
            self.ctx
                .queue
                .write_buffer(&self.pos_buf, 0, bytemuck::bytes_of(&(self.pos as i32)));
            self.ctx.queue.write_buffer(
                pf_kvq,
                0,
                bytemuck::bytes_of(&PfKvqParams {
                    tokens: live as u32,
                    x_stride_elems: (self.config.num_key_value_heads * self.config.head_dim)
                        as u32,
                    ..Default::default()
                }),
            );
        }
        Ok(())
    }

    fn prefill_chunk(&mut self, chunk: &[u32], live: usize) -> Result<()> {
        self.prefill_chunk_masked(chunk, live, &[])
    }

    fn prefill_chunk_masked(
        &mut self,
        chunk: &[u32],
        live: usize,
        splices: &[ChunkRowSplice],
    ) -> Result<()> {
        self.write_chunk_inputs(chunk, live, splices)?;
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
                anyhow::bail!("gpt_oss_wgpu prefill chunk validation: {e}");
            }
            self.pf_validated = true;
        }
        self.pos += live;
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
            "embedding-row splice prefill requires the chunked prefill graph: \
             NV_GPTOSS_WGPU_PREFILL_M>=2 and a live pf pass list"
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
                "embedding-row splices must be sorted and non-overlapping"
            );
            anyhow::ensure!(
                sp.position + n_slots <= tokens.len(),
                "splice at {} with {n_slots} rows exceeds {} tokens",
                sp.position,
                tokens.len()
            );
            prev_end = sp.position + n_slots;
            packed.push(pack_pairs(&sp.rows_bf16));
        }
        let mut done = 0usize;
        while done < tokens.len() {
            let live = m.min(tokens.len() - done);
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

    pub fn verify_max_rows(&self) -> usize {
        self.verify.as_ref().map(|v| v.rows).unwrap_or(0)
    }

    pub fn advance(&mut self, n: usize) -> Result<()> {
        anyhow::ensure!(
            self.pos + n <= self.max_seq,
            "advance {n} from {} past max_seq {}",
            self.pos,
            self.max_seq
        );
        self.pos += n;
        Ok(())
    }

    fn verify_forward(&mut self, batch: &[u32]) -> Result<usize> {
        anyhow::ensure!(
            !(flash_decode_enabled()
                || fused_step_enabled()
                || gemv_sg_enabled()
                || mx_sg_reordered_sum_enabled()),
            "the M-row verify trunk is bit-identical to the serial decode arms only, so the \
             verify surface refuses the shipped arm-stack default; spec/verify consumers must \
             set {FLASH_DECODE_ENV}=0 {FUSED_STEP_ENV}=0 {GEMV_SG_ENV}=0 {MX_SG_GATE_ENV}=0 \
             until the verify epilogue grows arm-consistent rows (no serving route consumes \
             gpt-oss verify today)"
        );
        let rows = self.verify_max_rows();
        anyhow::ensure!(
            rows > 0,
            "verify_chain needs the M-row prefill graph and its verify epilogue: \
             NV_GPTOSS_WGPU_PREFILL_M >= 2 and {VERIFY_M_ENV} != 0"
        );
        let live = batch.len();
        anyhow::ensure!(
            (1..=rows).contains(&live),
            "verify_chain batch of {live} out of 1..={rows}"
        );
        let mut padded = batch.to_vec();
        let pad = *padded.last().expect("non-empty batch");
        padded.resize(self.pf_m, pad);
        self.write_chunk_inputs(&padded, live, &[])?;
        let ctx = self.ctx;
        let verified = self.verify.as_ref().is_some_and(|v| v.validated);
        let scope = if self.pf_validated && verified {
            None
        } else {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        };
        {
            let vs = self
                .verify
                .as_ref()
                .expect("verify rows are nonzero only with a verify epilogue");
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for p in &self.pf_passes {
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
                for (i, p) in vs.passes.iter().enumerate() {
                    let z = if i >= vs.head_start {
                        live as u32
                    } else {
                        p.grid.2
                    };
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, z);
                }
            }
            ctx.queue.submit([enc.finish()]);
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gpt_oss_wgpu verify chain validation: {e}");
            }
            self.pf_validated = true;
            if let Some(vs) = self.verify.as_mut() {
                vs.validated = true;
            }
        }
        Ok(live)
    }

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        let live = self.verify_forward(batch)?;
        let vs = self.verify.as_ref().expect("verify epilogue");
        dispatch::read_back::<u32>(self.ctx, &vs.tokens, live)
            .map_err(|e| anyhow::anyhow!("verify token readback: {e}"))
    }

    pub fn verify_chain_logits(&mut self, batch: &[u32]) -> Result<(Vec<u32>, Vec<f32>)> {
        let live = self.verify_forward(batch)?;
        let vocab = self.vocab;
        let vs = self.verify.as_ref().expect("verify epilogue");
        let toks = dispatch::read_back::<u32>(self.ctx, &vs.tokens, live)
            .map_err(|e| anyhow::anyhow!("verify token readback: {e}"))?;
        let logits = dispatch::read_back::<f32>(self.ctx, &vs.logits, live * vocab)
            .map_err(|e| anyhow::anyhow!("verify logits readback: {e}"))?;
        Ok((toks, logits))
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

    #[doc(hidden)]
    pub fn debug_probe(&self, name: &str) -> Option<Vec<f32>> {
        let (_, buf, elems, bf16) = self.probes.iter().find(|(n, _, _, _)| n == name)?;
        if *bf16 {
            let words: Vec<u32> = dispatch::read_back(self.ctx, buf, elems.div_ceil(2)).ok()?;
            let mut out = Vec::with_capacity(*elems);
            for i in 0..*elems {
                out.push(bf16_val((words[i / 2] >> (16 * (i % 2))) as u16));
            }
            Some(out)
        } else {
            dispatch::read_back(self.ctx, buf, *elems).ok()
        }
    }

    #[doc(hidden)]
    pub fn debug_probe_names(&self) -> Vec<String> {
        self.probes.iter().map(|(n, _, _, _)| n.clone()).collect()
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

#[allow(clippy::too_many_arguments)]
fn push_gemv_bf16(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    out_f32: bool,
    dummy_bias: &wgpu::Buffer,
) -> Result<()> {
    if s.gemv_sg_on && w.k.is_multiple_of(8) {
        let (entry, rows_per_wg) = gemv_bf16_sg_entry_and_rows_per_wg(s, w.k / 8);
        let groups = w.n.div_ceil(rows_per_wg);
        let grid = b.grid1(groups as u64, 1);
        let p = b.uni(
            "gow-gsg-p",
            GemvSgParams {
                n_rows: w.n as u32,
                k_v4: (w.k / 8) as u32,
                groups_x: grid.0,
                out_f32: u32::from(out_f32),
                w_row_v4: (w.k / 8) as u32,
                x_off_v4: 0,
                y_off_words: 0,
                has_bias: u32::from(w.bias.is_some()),
                alpha: 1.0,
                ..Default::default()
            },
        );
        let bias = w.bias.as_ref().unwrap_or(dummy_bias);
        return b.push(
            label,
            &s.fused,
            &entry,
            &[(30, &w.w), (31, x), (32, &p), (33, y), (34, bias)],
            grid,
        );
    }
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "gow-gemvb-p",
        GemvBf16Params {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: (w.k / 2) as u32,
            x_off_words: 0,
            y_off_words: 0,
            has_bias: u32::from(w.bias.is_some()),
            alpha: 1.0,
            ..Default::default()
        },
    );
    let bias = w.bias.as_ref().unwrap_or(dummy_bias);
    b.push(
        label,
        &s.gemv,
        "gow_gemv_bf16",
        &[(0, &w.w), (1, x), (2, &p), (3, y), (4, bias)],
        grid,
    )
}

#[allow(clippy::too_many_arguments)]
fn push_gemv_mx(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &MxGpu,
    x: &wgpu::Buffer,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    slots: usize,
    x_slot_stride_words: usize,
) -> Result<()> {
    let k_blocks = w.k / MX_BLOCK;
    let groups = w.n.div_ceil(s.mx_rows_per_wg as usize);
    let grid = b.grid1(groups as u64, 1);
    let p = b.uni(
        "gow-gemvmx-p",
        GemvMxParams {
            n_rows: w.n as u32,
            k_blocks: k_blocks as u32,
            groups_x: grid.0,
            has_bias: 1,
            w_e_stride_v4: (w.n * k_blocks) as u32,
            sf_e_stride_bytes: (w.n * k_blocks) as u32,
            bias_e_stride: w.n as u32,
            x_slot_stride_words: x_slot_stride_words as u32,
            y_slot_stride: w.n as u32,
            use_sel: 1,
            ..Default::default()
        },
    );
    anyhow::ensure!(slots <= w.e, "more slots than experts");
    let entry: String = if s.mx_kpre_blocks.contains(&k_blocks)
        && s.mx_entry == MX_SG_XV4_R2_LUT_XS_ENTRY
    {
        mx_kpre_r2_xs_entry_name(k_blocks)
    } else {
        s.mx_entry.to_string()
    };
    anyhow::ensure!(
        !entry.starts_with(MX_SG_XV4_ENTRY) || x_slot_stride_words.is_multiple_of(4),
        "the xv4 mx arms rebind x as vec4 words, so slot stride {x_slot_stride_words} must be a \
         multiple of 4"
    );
    anyhow::ensure!(
        !entry.contains("_xs")
            || k_blocks <= MX_XS_K_BLOCK_CAP_MATCHES_THE_512_VEC4_SHARED_ARRAY,
        "the xstage arms hold all of x in an 8 KiB shared array, so k_blocks {k_blocks} must \
         stay within the staged capacity"
    );
    b.push(
        label,
        &s.mx,
        &entry,
        &[
            (10, &w.w),
            (11, &w.scales),
            (12, x),
            (13, &p),
            (14, y),
            (15, sel),
            (16, &w.bias),
        ],
        (grid.0, grid.1, slots as u32),
    )
}

pub const PREFILL_M_MAX: usize = 64;

pub fn prefill_m() -> usize {
    match std::env::var("NV_GPTOSS_WGPU_PREFILL_M")
        .or_else(|_| std::env::var("NV_WGPU_PREFILL_M"))
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(0) => 0,
        Some(m) => m.clamp(2, PREFILL_M_MAX),
        None => 16,
    }
}

pub const VERIFY_M_ENV: &str = "NV_GPTOSS_WGPU_VERIFY_M";

pub const VERIFY_EPILOGUE_COSTS_ROWS_TIMES_VOCAB_F32_OF_LOGITS: &str =
    "the verify epilogue holds one f32 logit row per verify row (rows * vocab * 4 bytes) plus two \
     bf16 hidden scratch rows; at the published 20b geometry (vocab 201088) that is 804 KiB per \
     row, so the default verify width equals the prefill width and NV_GPTOSS_WGPU_VERIFY_M=0 \
     removes the epilogue entirely";

fn verify_rows(prefill_m: usize) -> usize {
    match std::env::var(VERIFY_M_ENV)
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
    {
        Some(0) => 0,
        Some(r) => r.min(prefill_m),
        None => prefill_m,
    }
}

struct Kv8Shared {
    lanepos_alt_entry: Option<String>,
    fd_buf: wgpu::Buffer,
    fd_base: SharedFdParams,
    gfd_buf: wgpu::Buffer,
    gfd_base: FlashDecodeParams,
    fold_src: String,
    fold_entry: String,
    splits: u32,
    scratch_splits: u32,
    stage1_grid_x_is_one_when_a_workgroup_covers_every_kv_head: u32,
}

struct Kv8Attn {
    kc8: wgpu::Buffer,
    vc8: wgpu::Buffer,
    ksc: wgpu::Buffer,
    vsc: wgpu::Buffer,
    kvq_p: wgpu::Buffer,
}

struct AttnGpu {
    wq: Bf16Gpu,
    wk: Bf16Gpu,
    wv: Bf16Gpu,
    wo: Bf16Gpu,
    kc: wgpu::Buffer,
    vc: wgpu::Buffer,
    sinks: wgpu::Buffer,
    dummy_bias: wgpu::Buffer,
    window: usize,
    kv8: Option<Kv8Attn>,
}

struct MoeGpu {
    router: Bf16Gpu,
    gu: MxGpu,
    dn: MxGpu,
    dummy_bias: wgpu::Buffer,
}

struct PfScratch {
    m: usize,
    tok: wgpu::Buffer,
    ck: wgpu::Buffer,
    splice: wgpu::Buffer,
    mask: wgpu::Buffer,
    res: wgpu::Buffer,
    normed: wgpu::Buffer,
    normed_post: wgpu::Buffer,
    mixed: wgpu::Buffer,
    moe_out: wgpu::Buffer,
    q_raw: wgpu::Buffer,
    k_raw: wgpu::Buffer,
    v_raw: wgpu::Buffer,
    q: wgpu::Buffer,
    k: wgpu::Buffer,
    scores: wgpu::Buffer,
    attn: wgpu::Buffer,
    attn_bf16: wgpu::Buffer,
    rlogits: wgpu::Buffer,
    ids: wgpu::Buffer,
    wts: wgpu::Buffer,
    y_gu: wgpu::Buffer,
    act: wgpu::Buffer,
    y_down: wgpu::Buffer,
}

fn alloc_pf_scratch(b: &mut Builder, cfg: &GptOssConfig, max_seq: usize, m: usize) -> PfScratch {
    let hidden_words = cfg.hidden_size / 2;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let hd = cfg.head_dim;
    let k_top = cfg.num_experts_per_tok;
    let inter = cfg.intermediate_size;
    let mw = (m * hidden_words * 4) as u64;
    PfScratch {
        m,
        tok: b.upload_u32("gow-pf-tok", &vec![0u32; m]),
        ck: b.uni("gow-pf-ck", CkParams::default()),
        splice: b.zeros("gow-pf-splice", mw),
        mask: b.upload_u32("gow-pf-mask", &vec![0u32; m]),
        res: b.zeros("gow-pf-res", mw),
        normed: b.zeros("gow-pf-normed", mw),
        normed_post: b.zeros("gow-pf-normed-post", mw),
        mixed: b.zeros("gow-pf-mix", mw),
        moe_out: b.zeros("gow-pf-moeout", mw),
        q_raw: b.zeros("gow-pf-qraw", (m * n_h * hd * 2) as u64),
        k_raw: b.zeros("gow-pf-kraw", (m * n_kv * hd * 2) as u64),
        v_raw: b.zeros("gow-pf-vraw", (m * n_kv * hd * 2) as u64),
        q: b.zeros("gow-pf-q", (m * n_h * hd * 2) as u64),
        k: b.zeros("gow-pf-k", (m * n_kv * hd * 2) as u64),
        scores: b.zeros("gow-pf-scores", (m * n_h * max_seq * 4) as u64),
        attn: b.zeros("gow-pf-attn", (m * n_h * hd * 4) as u64),
        attn_bf16: b.zeros("gow-pf-packed", (m * n_h * hd * 2) as u64),
        rlogits: b.zeros("gow-pf-rlogits", (m * cfg.num_local_experts * 4) as u64),
        ids: b.zeros("gow-pf-ids", (m * k_top * 4) as u64),
        wts: b.zeros("gow-pf-wts", (m * k_top * 4) as u64),
        y_gu: b.zeros("gow-pf-ygu", (m * k_top * 2 * inter * 4) as u64),
        act: b.zeros("gow-pf-act", (m * k_top * inter * 2) as u64),
        y_down: b.zeros("gow-pf-ydown", (m * k_top * cfg.hidden_size * 4) as u64),
    }
}

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
        "gow-pf-rms-p",
        RmsParams {
            hidden: hidden as u32,
            batch: m as u32,
            eps,
            words_per_row: (hidden / 2) as u32,
        },
    );
    b.push(
        label,
        &s.rms,
        "rmsnorm_bf16",
        &[(0, x), (1, w), (2, y), (3, &p)],
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
        "gow-pf-rmsres-p",
        RmsParams {
            hidden: hidden as u32,
            batch: m as u32,
            eps,
            words_per_row: (hidden / 2) as u32,
        },
    );
    b.push(
        label,
        &s.rmsres,
        "rmsnorm_residual_bf16",
        &[(0, x), (1, res), (2, w), (3, y), (4, &p)],
        (m as u32, 1, 1),
    )
}

#[allow(clippy::too_many_arguments)]
fn pf_gemv_bf16(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &Bf16Gpu,
    x: &wgpu::Buffer,
    x_row_words: usize,
    y: &wgpu::Buffer,
    y_row_words: usize,
    out_f32: bool,
    dummy_bias: &wgpu::Buffer,
    m: usize,
) -> Result<()> {
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "gow-pf-gemvb-p",
        PfGemvBf16Params {
            n_rows: w.n as u32,
            k_words: (w.k / 2) as u32,
            groups_x: grid.0,
            out_f32: u32::from(out_f32),
            w_row_words: (w.k / 2) as u32,
            x_row_words: x_row_words as u32,
            y_row_words: y_row_words as u32,
            has_bias: u32::from(w.bias.is_some()),
            alpha: 1.0,
            ..Default::default()
        },
    );
    let bias = w.bias.as_ref().unwrap_or(dummy_bias);
    b.push(
        label,
        &s.prefill,
        "gow_pf_gemv_bf16",
        &[(10, &w.w), (11, x), (12, &p), (13, y), (14, bias)],
        (grid.0, grid.1, m as u32),
    )
}

#[allow(clippy::too_many_arguments)]
fn pf_gemv_mx(
    b: &mut Builder,
    s: &Sources,
    label: &str,
    w: &MxGpu,
    x: &wgpu::Buffer,
    x_tok_stride_words: usize,
    x_slot_stride_words: usize,
    y: &wgpu::Buffer,
    sel: &wgpu::Buffer,
    m: usize,
    k_top: usize,
) -> Result<()> {
    let k_blocks = w.k / MX_BLOCK;
    let pairs = w.n.div_ceil(2);
    let grid = b.grid1(pairs as u64, 1);
    let p = b.uni(
        "gow-pf-gemvmx-p",
        PfGemvMxParams {
            n_rows: w.n as u32,
            k_blocks: k_blocks as u32,
            groups_x: grid.0,
            has_bias: 1,
            w_e_stride_v4: (w.n * k_blocks) as u32,
            sf_e_stride_bytes: (w.n * k_blocks) as u32,
            bias_e_stride: w.n as u32,
            x_slot_stride_words: x_slot_stride_words as u32,
            x_tok_stride_words: x_tok_stride_words as u32,
            y_slot_stride: w.n as u32,
            k_top: k_top as u32,
            ..Default::default()
        },
    );
    anyhow::ensure!(
        k_top <= w.e,
        "router picks {k_top} of {} experts; the slot axis carries m*k_top rows, but every slot \
         must still name a real expert",
        w.e
    );
    b.push(
        label,
        &s.prefill,
        "gow_pf_gemv_mx",
        &[
            (60, &w.w),
            (61, &w.scales),
            (62, x),
            (63, &p),
            (64, y),
            (65, sel),
            (66, &w.bias),
        ],
        (grid.0, grid.1, (m * k_top) as u32),
    )
}

fn qkv_concat_host(a: &HostAttn) -> Result<HostBf16Lin> {
    let (q, k, v) = (&a.q, &a.k, &a.v);
    anyhow::ensure!(
        q.k == k.k && q.k == v.k,
        "fused qkv gemv needs one shared input width, got {} / {} / {}",
        q.k,
        k.k,
        v.k
    );
    anyhow::ensure!(
        q.n.is_multiple_of(2) && k.n.is_multiple_of(2),
        "fused qkv output packs bf16 pairs, so the q and k sections must hold even row counts \
         ({} / {}) or the section boundaries straddle a packed word",
        q.n,
        k.n
    );
    let any_bias = !(q.bias.is_empty() && k.bias.is_empty() && v.bias.is_empty());
    anyhow::ensure!(
        !any_bias || (!q.bias.is_empty() && !k.bias.is_empty() && !v.bias.is_empty()),
        "fused qkv concatenates biases positionally, so either every projection carries one or none does"
    );
    let mut w = Vec::with_capacity((q.n + k.n + v.n) * q.k);
    w.extend_from_slice(&q.w);
    w.extend_from_slice(&k.w);
    w.extend_from_slice(&v.w);
    let mut bias = Vec::new();
    if any_bias {
        bias.extend_from_slice(&q.bias);
        bias.extend_from_slice(&k.bias);
        bias.extend_from_slice(&v.bias);
    }
    let lin = HostBf16Lin {
        w,
        bias,
        n: q.n + k.n + v.n,
        k: q.k,
    };
    Ok(lin)
}

fn upload_bf16(b: &mut Builder, label: &str, l: &HostBf16Lin) -> Bf16Gpu {
    let bias = if l.bias.is_empty() {
        None
    } else {
        Some(b.upload_u32(&format!("{label}-b"), &pack_pairs(&l.bias)))
    };
    Bf16Gpu {
        w: b.upload_u32(label, &pack_pairs(&l.w)),
        bias,
        n: l.n,
        k: l.k,
    }
}
fn upload_mx(b: &mut Builder, label: &str, st: &HostMxStack) -> MxGpu {
    MxGpu {
        w: b.upload_u32(label, &bytes_to_words(&st.blocks)),
        scales: b.upload_u32(&format!("{label}-sf"), &bytes_to_words(&st.scales)),
        bias: b.upload_u32(&format!("{label}-bias"), &pack_pairs(&st.bias)),
        e: st.e,
        n: st.n,
        k: st.k,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_attn(
    b: &mut Builder,
    s: &Sources,
    cfg: &GptOssConfig,
    layer: &HostLayer,
    li: usize,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    pos_buf: &wgpu::Buffer,
    cosb: &wgpu::Buffer,
    sinb: &wgpu::Buffer,
    max_seq: usize,
    states: &mut Vec<(wgpu::Buffer, u64)>,
    flash_fd: Option<&wgpu::Buffer>,
    kv8: Option<&Kv8Shared>,
    flash_sw: Option<(&wgpu::Buffer, u32)>,
) -> Result<AttnGpu> {
    let a = &layer.attn;
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let window = match cfg.layer_types[li] {
        GptOssLayerType::Sliding => cfg.sliding_window,
        GptOssLayerType::Full => 0,
    };
    let use_kv8 = kv8.is_some() && window == 0;
    let fuse_qkv_except_where_kv8_quantizes_the_separate_k_and_v_rows = s.fused_on && !use_kv8;
    let fuse_qkv_rows_for_kv8 = s.fused_on
        && use_kv8
        && qkv_fuse_kv8_enabled_one_fused_qkv_gemv_plus_one_rope_row_writer_replaces_the_kv8_layers_six_dispatch_split_chain();
    anyhow::ensure!(
        a.sinks.len() == n_h,
        "sinks length {} != {n_h}",
        a.sinks.len()
    );

    let wq = upload_bf16(b, "gow-at-q", &a.q);
    let wk_ = upload_bf16(b, "gow-at-k", &a.k);
    let wv = upload_bf16(b, "gow-at-v", &a.v);
    let wo = upload_bf16(b, "gow-at-o", &a.o);
    let dummy_bias = b.upload_u32("gow-at-dummy", &[0u32]);

    let kv_words = n_kv * hd / 2;
    let cache_bytes = (max_seq * kv_words * 4) as u64;
    let kc = b.zeros("gow-at-kc", cache_bytes);
    let vc = b.zeros("gow-at-vc", cache_bytes);
    states.push((kc.clone(), cache_bytes));
    states.push((vc.clone(), cache_bytes));

    let q = b.zeros("gow-at-q", (n_h * hd * 2) as u64);
    let mut unfused_k_and_v_rows: Option<(wgpu::Buffer, wgpu::Buffer)> = None;
    if fuse_qkv_except_where_kv8_quantizes_the_separate_k_and_v_rows {
        let qkv_lin = qkv_concat_host(a)?;
        let qkv_raw = b.zeros("gow-at-qkvraw", (qkv_lin.n * 2) as u64);
        let wqkv = upload_bf16(b, "gow-at-qkv", &qkv_lin);
        push_gemv_bf16(b, s, "gow-at-qkvproj", &wqkv, x, &qkv_raw, false, &dummy_bias)?;
        let frp = b.uni(
            "gow-at-furope-p",
            FusedRopeParams {
                n_h: n_h as u32,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                rot_half: (hd / 2) as u32,
            },
        );
        b.push(
            "gow-at-furopekv",
            &s.fused,
            "gow_fu_rope_kvw",
            &[
                (0, &qkv_raw),
                (1, cosb),
                (2, sinb),
                (3, pos_buf),
                (4, &q),
                (5, &kc),
                (6, &vc),
                (7, &frp),
            ],
            ((n_h + n_kv) as u32, 1, 1),
        )?;
    } else if fuse_qkv_rows_for_kv8 {
        let qkv_lin = qkv_concat_host(a)?;
        let qkv_raw = b.zeros("gow-at-qkvraw", (qkv_lin.n * 2) as u64);
        let wqkv = upload_bf16(b, "gow-at-qkv", &qkv_lin);
        push_gemv_bf16(b, s, "gow-at-qkvproj", &wqkv, x, &qkv_raw, false, &dummy_bias)?;
        let k_rows = b.zeros("gow-at-k", (n_kv * hd * 2) as u64);
        let v_rows = b.zeros("gow-at-vraw", (n_kv * hd * 2) as u64);
        let frp = b.uni(
            "gow-at-furope-p",
            FusedRopeParams {
                n_h: n_h as u32,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                rot_half: (hd / 2) as u32,
            },
        );
        b.push(
            "gow-at-furopekvrows",
            &s.fused,
            QKV_FUSE_KV8_ROWS_ENTRY,
            &[
                (0, &qkv_raw),
                (1, cosb),
                (2, sinb),
                (3, pos_buf),
                (4, &q),
                (5, &k_rows),
                (6, &v_rows),
                (7, &frp),
            ],
            ((n_h + n_kv) as u32, 1, 1),
        )?;
        unfused_k_and_v_rows = Some((k_rows, v_rows));
    } else {
        let q_raw = b.zeros("gow-at-qraw", (wq.n * 2) as u64);
        let k_raw = b.zeros("gow-at-kraw", (wk_.n * 2) as u64);
        let v_raw = b.zeros("gow-at-vraw", (wv.n * 2) as u64);
        push_gemv_bf16(b, s, "gow-at-qproj", &wq, x, &q_raw, false, &dummy_bias)?;
        push_gemv_bf16(b, s, "gow-at-kproj", &wk_, x, &k_raw, false, &dummy_bias)?;
        push_gemv_bf16(b, s, "gow-at-vproj", &wv, x, &v_raw, false, &dummy_bias)?;

        let k = b.zeros("gow-at-k", (n_kv * hd * 2) as u64);
        let qp = b.uni(
            "gow-at-rope-p",
            RopeParams {
                n_rows: n_h as u32,
                head_dim: hd as u32,
                rot_half: (hd / 2) as u32,
                pad0: 0,
            },
        );
        b.push(
            "gow-at-ropeq",
            &s.attn,
            "gow_rope",
            &[
                (0, &q_raw),
                (1, cosb),
                (2, sinb),
                (3, pos_buf),
                (4, &q),
                (5, &qp),
            ],
            (n_h as u32, 1, 1),
        )?;
        let kp = b.uni(
            "gow-at-ropek-p",
            RopeParams {
                n_rows: n_kv as u32,
                head_dim: hd as u32,
                rot_half: (hd / 2) as u32,
                pad0: 0,
            },
        );
        b.push(
            "gow-at-ropek",
            &s.attn,
            "gow_rope",
            &[
                (0, &k_raw),
                (1, cosb),
                (2, sinb),
                (3, pos_buf),
                (4, &k),
                (5, &kp),
            ],
            (n_kv as u32, 1, 1),
        )?;

        let kvp = b.uni(
            "gow-at-kv-p",
            KvWriteParams {
                words: kv_words as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1(kv_words as u64, 64);
        b.push(
            "gow-at-kvwrite",
            &s.attn,
            "gow_kv_write",
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
        b.probe("kraw0", &k_raw, n_kv * hd, true);
        unfused_k_and_v_rows = Some((k, v_raw));
    }

    let sinks = b.upload_f32("gow-at-sinks", &a.sinks);
    b.probe("q0", &q, n_h * hd, true);
    b.probe("sinks0", &sinks, n_h, false);
    b.probe("mix0", out, cfg.hidden_size, true);
    let attn_bf16 = b.zeros("gow-at-packed", (n_h * hd * 2) as u64);
    b.probe(&format!("attnpk{li}"), &attn_bf16, n_h * hd, true);
    let flash_layer = if use_kv8 {
        None
    } else if window == 0 {
        flash_fd.map(|f| {
            (
                f,
                flash_splits_grid_y_scratch_rows_and_stage1_stride_read_once_at_build(),
            )
        })
    } else {
        flash_sw
    };
    let mut kv8_attn: Option<Kv8Attn> = None;
    if let (Some(k8), true) = (kv8, use_kv8) {
        let Some((k_rows, v_rows)) = unfused_k_and_v_rows.as_ref() else {
            anyhow::bail!(
                "the kv8 arm quantizes the unfused per-layer k and v rows, so the fused qkv arm \
                 must stay off on every layer kv8 covers"
            );
        };
        anyhow::ensure!(
            hd.is_multiple_of(32),
            "the shared fp8 fold stage1 packs 4 e4m3 bytes per word and strides head_dim in \
             32-lane strips, got {hd}"
        );
        let cache8_bytes = (max_seq * n_kv * hd) as u64;
        let scale_bytes = (max_seq * n_kv * 4) as u64;
        let kc8 = b.zeros("gow-at-kc8", cache8_bytes);
        let vc8 = b.zeros("gow-at-vc8", cache8_bytes);
        let ksc = b.zeros("gow-at-ks8", scale_bytes);
        let vsc = b.zeros("gow-at-vs8", scale_bytes);
        states.push((kc8.clone(), cache8_bytes));
        states.push((vc8.clone(), cache8_bytes));
        states.push((ksc.clone(), scale_bytes));
        states.push((vsc.clone(), scale_bytes));
        let kvq_p = b.uni(
            "gow-at-kvq-p",
            KvFp8Params {
                n_tokens: 1,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                pairs: n_kv as u32,
                slots: max_seq as u32,
                ..Default::default()
            },
        );
        b.push(
            "gow-at-kvq-k",
            &s.kvq,
            wk::kv_fp8::QUANTIZE_ENTRY,
            &[(0, k_rows), (1, &kc8), (2, &ksc), (3, pos_buf), (4, &kvq_p)],
            (n_kv as u32, 1, 1),
        )?;
        b.push(
            "gow-at-kvq-v",
            &s.kvq,
            wk::kv_fp8::QUANTIZE_ENTRY,
            &[(0, v_rows), (1, &vc8), (2, &vsc), (3, pos_buf), (4, &kvq_p)],
            (n_kv as u32, 1, 1),
        )?;
        let q_f32 = b.zeros("gow-at-qf32", (n_h * hd * 4) as u64);
        let cast_p = b.uni(
            "gow-at-qcast-p",
            ResScaleParams {
                n: (n_h * hd) as u32,
                n_words: (n_h * hd / 2) as u32,
                ..Default::default()
            },
        );
        let cast_grid = b.grid1((n_h * hd / 2) as u64, 256);
        b.push(
            "gow-at-qcast",
            &s.resscale,
            "cast_bf16_to_f32",
            &[(0, &q), (3, &cast_p), (4, &q_f32)],
            cast_grid,
        )?;
        let splits = k8.splits;
        let scratch = b.zeros(
            "gow-at-fscr8",
            (n_h * k8.scratch_splits as usize * (hd + 2) * 4) as u64,
        );
        let flash1_pass_index = b.passes.len();
        b.depthgate_flash1_pass_indices_full_attention_only_sliding_cost_does_not_scale_with_depth
            .push(flash1_pass_index);
        b.push(
            "gow-at-flash1",
            &k8.fold_src,
            &k8.fold_entry,
            &[
                (0, &q_f32),
                (4, &k8.fd_buf),
                (5, &kc8),
                (6, &vc8),
                (7, &scratch),
                (8, &ksc),
                (9, &vsc),
            ],
            (
                k8.stage1_grid_x_is_one_when_a_workgroup_covers_every_kv_head,
                splits,
                1,
            ),
        )?;
        if let Some(alt_entry) = &k8.lanepos_alt_entry {
            let alt = b.built_pass(
                "gow-at-flash1-deep",
                &k8.fold_src,
                alt_entry,
                &[
                    (0, &q_f32),
                    (4, &k8.fd_buf),
                    (5, &kc8),
                    (6, &vc8),
                    (7, &scratch),
                    (8, &ksc),
                    (9, &vsc),
                ],
                (
                    k8.stage1_grid_x_is_one_when_a_workgroup_covers_every_kv_head,
                    splits,
                    1,
                ),
            )?;
            b.lanepos_alt_flash1_passes.push(alt);
        }
        b.push(
            "gow-at-flash2",
            &s.flash,
            flash_stage2_entry_scc_caches_the_split_scales_and_par_also_splits_the_serial_scan_over_the_eight_warps(),
            &[(3, &scratch), (4, &attn_bf16), (5, &sinks), (6, &k8.gfd_buf)],
            (n_h as u32, 1, 1),
        )?;
        kv8_attn = Some(Kv8Attn {
            kc8,
            vc8,
            ksc,
            vsc,
            kvq_p,
        });
    } else if let Some((fp, splits)) = flash_layer {
        anyhow::ensure!(
            hd <= MAX_HEAD_DIM && hd.is_multiple_of(2),
            "gow flash stage1 accumulates head_dim in 32-lane strips up to {MAX_HEAD_DIM} and \
             stage2 packs bf16 pairs, got {hd}"
        );
        anyhow::ensure!(
            b.ctx.caps.max_compute_invocations_per_workgroup >= 256
                && b.ctx.caps.max_compute_workgroup_size_x >= 256
                && b.ctx.caps.workgroup_storage_fits(
                    FLASH_WORKGROUP_BYTES_QSH256_RED256_SM8_SL8_SACC2048_F32,
                ),
            "{FLASH_DECODE_ENV} flash decode (the shipped default) needs a 256-invocation \
             workgroup and {} workgroup bytes; device allows {} invocations / {} bytes -- set \
             {FLASH_DECODE_ENV}=0 to restore the serial arm",
            FLASH_WORKGROUP_BYTES_QSH256_RED256_SM8_SL8_SACC2048_F32,
            b.ctx.caps.max_compute_invocations_per_workgroup,
            b.ctx.caps.max_compute_workgroup_storage_size
        );
        let scratch_splits = if window == 0 {
            flash_depthgate_scratch_splits_must_cover_every_band_the_gate_can_pick(splits)
        } else {
            splits
        };
        let scratch = b.zeros(
            "gow-at-fscr",
            (n_h * scratch_splits as usize * (hd + 2) * 4) as u64,
        );
        let kvshare = s.flash_stage1_entry == "gow_flash_stage1_sg"
            && hd == KVSHARE_STAGE1_IS_SPECIALIZED_TO_HD_64
            && n_h / n_kv == KVSHARE_STAGE1_IS_SPECIALIZED_TO_GROUP_8;
        let (stage1_entry, grid_x) = if kvshare {
            ("gow_flash_stage1_kvshare_sg_group8_hd64", n_kv as u32)
        } else {
            (s.flash_stage1_entry, n_h as u32)
        };
        if window == 0 {
            let flash1_pass_index = b.passes.len();
            b.depthgate_flash1_pass_indices_full_attention_only_sliding_cost_does_not_scale_with_depth
                .push(flash1_pass_index);
        }
        b.push(
            "gow-at-flash1",
            &s.flash,
            stage1_entry,
            &[(0, &q), (1, &kc), (2, &vc), (3, &scratch), (6, fp)],
            (grid_x, splits, 1),
        )?;
        b.push(
            "gow-at-flash2",
            &s.flash,
            flash_stage2_entry_scc_caches_the_split_scales_and_par_also_splits_the_serial_scan_over_the_eight_warps(),
            &[(3, &scratch), (4, &attn_bf16), (5, &sinks), (6, fp)],
            (n_h as u32, 1, 1),
        )?;
    } else if s.fused_on {
        let scores = b.zeros("gow-at-scores", (n_h * max_seq * 4) as u64);
        let adp = b.uni(
            "gow-at-dec-p",
            AttnDecodeParams {
                n_heads: n_h as u32,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                max_seq: max_seq as u32,
                group: (n_h / n_kv) as u32,
                window: window as u32,
                scale: 1.0 / (hd as f32).sqrt(),
                ..Default::default()
            },
        );
        b.push(
            "gow-at-fudecpk",
            &s.fused,
            "gow_fu_attn_decode_pk",
            &[
                (8, &q),
                (9, &kc),
                (10, &vc),
                (11, &scores),
                (12, &attn_bf16),
                (13, pos_buf),
                (14, &adp),
                (15, &sinks),
            ],
            (n_h as u32, 1, 1),
        )?;
    } else {
        let scores = b.zeros("gow-at-scores", (n_h * max_seq * 4) as u64);
        let attn = b.zeros("gow-at-out", (n_h * hd * 4) as u64);
        let adp = b.uni(
            "gow-at-dec-p",
            AttnDecodeParams {
                n_heads: n_h as u32,
                n_kv: n_kv as u32,
                head_dim: hd as u32,
                max_seq: max_seq as u32,
                group: (n_h / n_kv) as u32,
                window: window as u32,
                scale: 1.0 / (hd as f32).sqrt(),
                ..Default::default()
            },
        );
        b.push(
            "gow-at-decode",
            &s.attn,
            "gow_attn_decode",
            &[
                (20, &q),
                (21, &kc),
                (22, &vc),
                (23, &scores),
                (24, &attn),
                (25, pos_buf),
                (26, &adp),
                (27, &sinks),
            ],
            (n_h as u32, 1, 1),
        )?;
        b.probe("attn0", &attn, n_h * hd, false);
        let pkp = b.uni(
            "gow-at-pack-p",
            PackParams {
                n_words: (n_h * hd / 2) as u32,
                ..Default::default()
            },
        );
        let grid = b.grid1((n_h * hd / 2) as u64, 64);
        b.push(
            "gow-at-pack",
            &s.attn,
            "gow_pack_bf16",
            &[(30, &attn), (31, &attn_bf16), (32, &pkp)],
            grid,
        )?;
    }

    push_gemv_bf16(
        b,
        s,
        "gow-at-oproj",
        &wo,
        &attn_bf16,
        out,
        false,
        &dummy_bias,
    )?;

    Ok(AttnGpu {
        wq,
        wk: wk_,
        wv,
        wo,
        kc,
        vc,
        sinks,
        dummy_bias,
        window,
        kv8: kv8_attn,
    })
}

#[allow(clippy::too_many_arguments)]
fn build_attn_prefill(
    b: &mut Builder,
    s: &Sources,
    cfg: &GptOssConfig,
    ag: &AttnGpu,
    sc: &PfScratch,
    cosb: &wgpu::Buffer,
    sinb: &wgpu::Buffer,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
    max_seq: usize,
    pos_buf: &wgpu::Buffer,
    pf_kvq: Option<&wgpu::Buffer>,
) -> Result<()> {
    let m = sc.m;
    let hd = cfg.head_dim;
    let n_h = cfg.num_attention_heads;
    let n_kv = cfg.num_key_value_heads;
    let hidden_words = cfg.hidden_size / 2;

    pf_gemv_bf16(
        b,
        s,
        "gow-pf-qproj",
        &ag.wq,
        x,
        hidden_words,
        &sc.q_raw,
        n_h * hd / 2,
        false,
        &ag.dummy_bias,
        m,
    )?;
    pf_gemv_bf16(
        b,
        s,
        "gow-pf-kproj",
        &ag.wk,
        x,
        hidden_words,
        &sc.k_raw,
        n_kv * hd / 2,
        false,
        &ag.dummy_bias,
        m,
    )?;
    pf_gemv_bf16(
        b,
        s,
        "gow-pf-vproj",
        &ag.wv,
        x,
        hidden_words,
        &sc.v_raw,
        n_kv * hd / 2,
        false,
        &ag.dummy_bias,
        m,
    )?;

    let qp = b.uni(
        "gow-pf-ropeq-p",
        RopeParams {
            n_rows: n_h as u32,
            head_dim: hd as u32,
            rot_half: (hd / 2) as u32,
            pad0: 0,
        },
    );
    b.push(
        "gow-pf-ropeq",
        &s.prefill,
        "gow_pf_rope",
        &[
            (20, &sc.q_raw),
            (21, cosb),
            (22, sinb),
            (23, &sc.q),
            (24, &qp),
            (25, &sc.ck),
        ],
        (n_h as u32, m as u32, 1),
    )?;
    let kp = b.uni(
        "gow-pf-ropek-p",
        RopeParams {
            n_rows: n_kv as u32,
            head_dim: hd as u32,
            rot_half: (hd / 2) as u32,
            pad0: 0,
        },
    );
    b.push(
        "gow-pf-ropek",
        &s.prefill,
        "gow_pf_rope",
        &[
            (20, &sc.k_raw),
            (21, cosb),
            (22, sinb),
            (23, &sc.k),
            (24, &kp),
            (25, &sc.ck),
        ],
        (n_kv as u32, m as u32, 1),
    )?;

    let kv_words = n_kv * hd / 2;
    let kvp = b.uni(
        "gow-pf-kv-p",
        KvWriteParams {
            words: kv_words as u32,
            ..Default::default()
        },
    );
    b.push(
        "gow-pf-kvwrite",
        &s.prefill,
        "gow_pf_kv_write",
        &[
            (30, &sc.k),
            (31, &sc.v_raw),
            (32, &ag.kc),
            (33, &ag.vc),
            (34, &kvp),
            (35, &sc.ck),
        ],
        ((kv_words as u32).div_ceil(64), m as u32, 1),
    )?;
    if ag.window == 0 {
        b.pf_kv_write_full_layers += 1;
    }
    if let Some(k8) = ag.kv8.as_ref() {
        let pf_kvq = pf_kvq.ok_or_else(|| {
            anyhow::anyhow!(
                "{KV_FP8_ENV} full-layer prefill needs the per-chunk kvq uniform; \
                 build() allocates it whenever kv8 and the prefill graph coexist"
            )
        })?;
        b.push(
            "gow-pf-kvq-k",
            &s.pf_kvq,
            "q3w_pf_quantize_kv_fp8_m",
            &[
                (0, &sc.k),
                (1, &k8.kc8),
                (2, &k8.ksc),
                (3, pos_buf),
                (4, &k8.kvq_p),
                (9, pf_kvq),
            ],
            (n_kv as u32, m as u32, 1),
        )?;
        b.push(
            "gow-pf-kvq-v",
            &s.pf_kvq,
            "q3w_pf_quantize_kv_fp8_m",
            &[
                (0, &sc.v_raw),
                (1, &k8.vc8),
                (2, &k8.vsc),
                (3, pos_buf),
                (4, &k8.kvq_p),
                (9, pf_kvq),
            ],
            (n_kv as u32, m as u32, 1),
        )?;
        b.pf_kvq_pairs += 1;
    }

    let adp = b.uni(
        "gow-pf-attn-p",
        AttnDecodeParams {
            n_heads: n_h as u32,
            n_kv: n_kv as u32,
            head_dim: hd as u32,
            max_seq: max_seq as u32,
            group: (n_h / n_kv) as u32,
            window: ag.window as u32,
            scale: 1.0 / (hd as f32).sqrt(),
            ..Default::default()
        },
    );
    b.push(
        "gow-pf-attn",
        &s.prefill,
        "gow_pf_attn",
        &[
            (40, &sc.q),
            (41, &ag.kc),
            (42, &ag.vc),
            (43, &sc.scores),
            (44, &sc.attn),
            (45, &adp),
            (46, &ag.sinks),
            (47, &sc.ck),
        ],
        (n_h as u32, m as u32, 1),
    )?;

    let pkp = b.uni(
        "gow-pf-pack-p",
        PackParams {
            n_words: (m * n_h * hd / 2) as u32,
            ..Default::default()
        },
    );
    let grid = b.grid1((m * n_h * hd / 2) as u64, 64);
    b.push(
        "gow-pf-pack",
        &s.attn,
        "gow_pack_bf16",
        &[(30, &sc.attn), (31, &sc.attn_bf16), (32, &pkp)],
        grid,
    )?;

    pf_gemv_bf16(
        b,
        s,
        "gow-pf-oproj",
        &ag.wo,
        &sc.attn_bf16,
        n_h * hd / 2,
        out,
        hidden_words,
        false,
        &ag.dummy_bias,
        m,
    )
}

fn build_moe(
    b: &mut Builder,
    s: &Sources,
    cfg: &GptOssConfig,
    m: &HostMoe,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
) -> Result<MoeGpu> {
    let hidden = cfg.hidden_size;
    let hidden_words = hidden / 2;
    let inter = cfg.intermediate_size;
    let e = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;

    anyhow::ensure!(
        m.gate_up.e == e && m.down.e == e,
        "expert stack count mismatch"
    );
    anyhow::ensure!(
        m.gate_up.n == 2 * inter && m.gate_up.k == hidden,
        "gate_up stack is [{}, {}], want [{}, {hidden}]",
        m.gate_up.n,
        m.gate_up.k,
        2 * inter
    );
    anyhow::ensure!(
        m.down.n == hidden && m.down.k == inter,
        "down stack is [{}, {}], want [{hidden}, {inter}]",
        m.down.n,
        m.down.k
    );

    let router = upload_bf16(b, "gow-moe-router", &m.router);
    let dummy_bias = b.upload_u32("gow-moe-dummy", &[0u32]);
    let rlogits = b.zeros("gow-moe-rlogits", (e * 4) as u64);
    let ids = b.zeros("gow-moe-ids", (k_top * 4) as u64);
    let wts = b.zeros("gow-moe-wts", (k_top * 4) as u64);
    let router_topk_fused = moe_router_topk_fuse_enabled()
        && s.gemv_sg_on
        && e <= 64
        && k_top <= MAX_TOPK
        && hidden.is_multiple_of(8);
    if router_topk_fused {
        let p = b.uni(
            "gow-moe-rtf-p",
            GemvSgParams {
                n_rows: e as u32,
                k_v4: (hidden / 8) as u32,
                groups_x: 1,
                out_f32: 1,
                w_row_v4: (hidden / 8) as u32,
                has_bias: u32::from(router.bias.is_some()),
                alpha: 1.0,
                topk_k: k_top as u32,
                ..Default::default()
            },
        );
        let bias = router.bias.as_ref().unwrap_or(&dummy_bias);
        b.push(
            "gow-moe-router-topk",
            &s.fused,
            "gow_fu_router_topk",
            &[
                (30, &router.w),
                (31, x),
                (32, &p),
                (33, &rlogits),
                (34, bias),
                (35, &ids),
                (36, &wts),
            ],
            (1, 1, 1),
        )?;
    } else {
        push_gemv_bf16(
            b,
            s,
            "gow-moe-router",
            &router,
            x,
            &rlogits,
            true,
            &dummy_bias,
        )?;
        let rp = b.uni(
            "gow-moe-router-p",
            RouterParams {
                n_experts: e as u32,
                k: k_top as u32,
                ..Default::default()
            },
        );
        let topk_sg32 = moe_topk_sg32_enabled() && e <= 32;
        b.push(
            "gow-moe-topk",
            &s.moe,
            if topk_sg32 {
                MOE_TOPK_SG32_ENTRY
            } else {
                "gow_router_topk"
            },
            &[(0, &rlogits), (1, &ids), (2, &wts), (3, &rp)],
            (1, 1, 1),
        )?;
    }

    let gu = upload_mx(b, "gow-moe-gu", &m.gate_up);
    let dn = upload_mx(b, "gow-moe-dn", &m.down);

    let act = b.zeros("gow-moe-act", (k_top * inter * 2) as u64);
    let gusw = mx_sg_gusw_enabled_one_subgroup_owns_two_gate_up_pairs_and_writes_the_swiglu_act_word_directly_skipping_the_swiglu_pass()
        && s.mx_entry.ends_with("_xs")
        && (2 * inter).is_multiple_of(MX_SG_R4SW_ROWS_PER_WG as usize);
    if gusw {
        let n_rows = 2 * inter;
        let k_blocks = gu.k / MX_BLOCK;
        let groups = n_rows.div_ceil(MX_SG_R4SW_ROWS_PER_WG as usize);
        let grid = b.grid1(groups as u64, 1);
        let p = b.uni(
            "gow-gemvmx-gusw-p",
            GemvMxParams {
                n_rows: n_rows as u32,
                k_blocks: k_blocks as u32,
                groups_x: grid.0,
                has_bias: 1,
                w_e_stride_v4: (gu.n * k_blocks) as u32,
                sf_e_stride_bytes: (gu.n * k_blocks) as u32,
                bias_e_stride: gu.n as u32,
                x_slot_stride_words: 0,
                y_slot_stride: (inter / 2) as u32,
                use_sel: 1,
                pad0: cfg.swiglu_limit.to_bits(),
                pad1: SWIGLU_ALPHA.to_bits(),
            },
        );
        anyhow::ensure!(
            k_blocks <= MX_XS_K_BLOCK_CAP_MATCHES_THE_512_VEC4_SHARED_ARRAY,
            "the fused gu+swiglu arm stages all of x in the 8 KiB shared array, so k_blocks \
             {k_blocks} must stay within the staged capacity"
        );
        let gu_entry: String = if s.mx_kpre_blocks.contains(&k_blocks) {
            mx_kpre_r4sw_entry_name(k_blocks)
        } else {
            MX_SG_XV4_R4_LUT_XS_SWIGLU_ENTRY.to_string()
        };
        b.push(
            "gow-moe-gemv-gu",
            &s.mx,
            &gu_entry,
            &[
                (10, &gu.w),
                (11, &gu.scales),
                (12, x),
                (13, &p),
                (14, &act),
                (15, &ids),
                (16, &gu.bias),
            ],
            (grid.0, grid.1, k_top as u32),
        )?;
    } else {
        let y_gu = b.zeros("gow-moe-ygu", (k_top * 2 * inter * 4) as u64);
        push_gemv_mx(b, s, "gow-moe-gemv-gu", &gu, x, &y_gu, &ids, k_top, 0)?;
        let swp = b.uni(
            "gow-moe-swiglu-p",
            SwigluParams {
                n_words: (k_top * inter / 2) as u32,
                inter_words: (inter / 2) as u32,
                two_inter: (2 * inter) as u32,
                limit: cfg.swiglu_limit,
                alpha: SWIGLU_ALPHA,
                ..Default::default()
            },
        );
        let grid = b.grid1((k_top * inter / 2) as u64, 64);
        b.push(
            "gow-moe-swiglu",
            &s.moe,
            "gow_swiglu",
            &[(10, &y_gu), (11, &act), (12, &swp)],
            grid,
        )?;
    }

    let dn_k_blocks = dn.k / MX_BLOCK;
    let dncomb = moe_dncomb_fuse_enabled_the_down_gemv_keeps_each_expert_row_in_a_register_and_folds_the_same_ascending_fma_chain_gow_moe_combine_would_have_run()
        && s.mx_entry == MX_SG_XV4_R2_LUT_XS_ENTRY
        && !s.mx_kpre_blocks.contains(&dn_k_blocks)
        && dn_k_blocks <= MX_XS_K_BLOCK_CAP_MATCHES_THE_512_VEC4_SHARED_ARRAY
        && hidden.is_multiple_of(MX_SG_XV4_R2_ROWS_PER_WG as usize)
        && (inter / 2).is_multiple_of(4);
    b.probe("rlogits0", &rlogits, e, false);
    b.probe("wts0", &wts, k_top, false);
    if dncomb {
        let wg64 = arm_env_opt_in_default_off(MOE_DNCOMB_WG64_ENV);
        let sgslot = if std::env::var(MOE_DNCOMB_SGSLOT_ENV).is_ok() {
            arm_env_enabled_with_the_moe_fuse_stack_default(MOE_DNCOMB_SGSLOT_ENV)
        } else {
            GOW_MOE_FUSE_STACK_XV4_LUT_XSTAGE_GUPAIR_GUSW_DNCOMB_SGSLOT_SHIPS_ON_ROWS_GPTOSS_FUSE_GATED_LADDER_175P7_166P8_107P6_TOK_S_AT_256_8K_122K_CAMPAIGN_PLUS_19P9_35P6_58P2_PCT_AND_ZERO_RESTORES_EACH_ARM
                && k_top
                    == DNCOMB_SGSLOT_KEEPS_EVERY_ROW_SLOT_PAIR_ON_ITS_OWN_SUBGROUP_SO_THE_RESIDENT_THREAD_COUNT_AND_MEMORY_LEVEL_PARALLELISM_MATCH_THE_SPLIT_PASS
        };
        anyhow::ensure!(
            !sgslot
                || k_top
                    == DNCOMB_SGSLOT_KEEPS_EVERY_ROW_SLOT_PAIR_ON_ITS_OWN_SUBGROUP_SO_THE_RESIDENT_THREAD_COUNT_AND_MEMORY_LEVEL_PARALLELISM_MATCH_THE_SPLIT_PASS,
            "the sgslot dncomb variant maps the 8 subgroups of a 256-thread workgroup onto \
             k_top expert slots x 2 row pairs, so it needs exactly 4 slots, not {k_top}"
        );
        let rows_per_wg = if wg64 || sgslot {
            DNCOMB_WG64_ROWS_PER_WG_RESTORES_THE_SPLIT_PASSES_720_RESIDENT_WORKGROUPS
        } else {
            MX_SG_XV4_R2_ROWS_PER_WG as usize
        };
        let groups = hidden.div_ceil(rows_per_wg);
        let grid = b.grid1(groups as u64, 1);
        anyhow::ensure!(
            hidden.is_multiple_of(rows_per_wg) && rows_per_wg.is_multiple_of(2),
            "each dncomb subgroup owns an even row pair that becomes one packed bf16 output \
             word, so hidden {hidden} must divide into whole {rows_per_wg}-row workgroups"
        );
        let p = b.uni(
            "gow-gemvmx-dncomb-p",
            GemvMxParams {
                n_rows: dn.n as u32,
                k_blocks: dn_k_blocks as u32,
                groups_x: grid.0,
                has_bias: 1,
                w_e_stride_v4: (dn.n * dn_k_blocks) as u32,
                sf_e_stride_bytes: (dn.n * dn_k_blocks) as u32,
                bias_e_stride: dn.n as u32,
                x_slot_stride_words: (inter / 2) as u32,
                y_slot_stride: dn.n as u32,
                use_sel: 1,
                pad0: k_top as u32,
                pad1: 0,
            },
        );
        b.push(
            "gow-moe-gemv-dncomb",
            &s.mx,
            match (sgslot, wg64) {
                (true, _) => MX_SG_XV4_R2_LUT_DNCOMB_SGSLOT_ENTRY,
                (false, true) => MX_SG_XV4_R2_LUT_XS_DNCOMB_WG64_ENTRY,
                (false, false) => MX_SG_XV4_R2_LUT_XS_DNCOMB_ENTRY,
            },
            &[
                (10, &dn.w),
                (11, &dn.scales),
                (12, &act),
                (13, &p),
                (15, &ids),
                (16, &dn.bias),
                (17, &wts),
                (18, out),
            ],
            (grid.0, grid.1, 1),
        )?;
    } else {
        let y_down = b.zeros("gow-moe-ydown", (k_top * hidden * 4) as u64);
        push_gemv_mx(
            b,
            s,
            "gow-moe-gemv-dn",
            &dn,
            &act,
            &y_down,
            &ids,
            k_top,
            inter / 2,
        )?;

        b.probe("ydown0", &y_down, k_top * hidden, false);
        let cbp = b.uni(
            "gow-moe-comb-p",
            CombineParams {
                hidden_words: hidden_words as u32,
                k: k_top as u32,
                slot_stride: hidden as u32,
                pad0: 0,
            },
        );
        let grid = b.grid1(hidden_words as u64, 64);
        b.push(
            "gow-moe-combine",
            &s.moe,
            "gow_moe_combine",
            &[(20, &y_down), (21, &wts), (22, out), (23, &cbp)],
            grid,
        )?;
    }

    Ok(MoeGpu {
        router,
        gu,
        dn,
        dummy_bias,
    })
}

fn build_moe_prefill(
    b: &mut Builder,
    s: &Sources,
    cfg: &GptOssConfig,
    mg: &MoeGpu,
    sc: &PfScratch,
    x: &wgpu::Buffer,
    out: &wgpu::Buffer,
) -> Result<()> {
    let m = sc.m;
    let hidden = cfg.hidden_size;
    let hidden_words = hidden / 2;
    let inter = cfg.intermediate_size;
    let e = cfg.num_local_experts;
    let k_top = cfg.num_experts_per_tok;

    pf_gemv_bf16(
        b,
        s,
        "gow-pf-moe-router",
        &mg.router,
        x,
        hidden_words,
        &sc.rlogits,
        e,
        true,
        &mg.dummy_bias,
        m,
    )?;

    let rp = b.uni(
        "gow-pf-router-p",
        PfRouterParams {
            n_experts: e as u32,
            k: k_top as u32,
            m: m as u32,
            pad0: 0,
        },
    );
    b.push(
        "gow-pf-moe-topk",
        &s.prefill,
        "gow_pf_router_topk",
        &[(50, &sc.rlogits), (51, &sc.ids), (52, &sc.wts), (53, &rp)],
        (m as u32, 1, 1),
    )?;

    pf_gemv_mx(
        b,
        s,
        "gow-pf-moe-gemv-gu",
        &mg.gu,
        x,
        hidden_words,
        0,
        &sc.y_gu,
        &sc.ids,
        m,
        k_top,
    )?;

    let slots = m * k_top;
    let swp = b.uni(
        "gow-pf-swiglu-p",
        SwigluParams {
            n_words: (slots * inter / 2) as u32,
            inter_words: (inter / 2) as u32,
            two_inter: (2 * inter) as u32,
            limit: cfg.swiglu_limit,
            alpha: SWIGLU_ALPHA,
            ..Default::default()
        },
    );
    let grid = b.grid1((slots * inter / 2) as u64, 64);
    b.push(
        "gow-pf-swiglu",
        &s.moe,
        "gow_swiglu",
        &[(10, &sc.y_gu), (11, &sc.act), (12, &swp)],
        grid,
    )?;

    pf_gemv_mx(
        b,
        s,
        "gow-pf-moe-gemv-dn",
        &mg.dn,
        &sc.act,
        0,
        inter / 2,
        &sc.y_down,
        &sc.ids,
        m,
        k_top,
    )?;

    let cbp = b.uni(
        "gow-pf-comb-p",
        PfCombineParams {
            hidden_words: hidden_words as u32,
            k: k_top as u32,
            slot_stride: hidden as u32,
            m: m as u32,
        },
    );
    b.push(
        "gow-pf-moe-combine",
        &s.prefill,
        "gow_pf_moe_combine",
        &[(70, &sc.y_down), (71, &sc.wts), (72, out), (73, &cbp)],
        ((hidden_words as u32).div_ceil(64), m as u32, 1),
    )
}

impl WeightSource<'_> {
    fn embed(&self, cfg: &GptOssConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.embed.clone()),
            Self::Loader(w) => load_bf16(
                w,
                &["model.embed_tokens.weight"],
                &[cfg.vocab_size, cfg.hidden_size],
            ),
        }
    }

    fn lm_head(&self, cfg: &GptOssConfig) -> Result<Vec<u16>> {
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

    fn final_norm(&self, cfg: &GptOssConfig) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.final_norm.clone()),
            Self::Loader(w) => load_bf16(w, &["model.norm.weight"], &[cfg.hidden_size]),
        }
    }

    fn layer_input_ln(&self, cfg: &GptOssConfig, idx: usize) -> Result<Vec<u16>> {
        match self {
            Self::Host(h) => Ok(h.layers[idx].input_ln.clone()),
            Self::Loader(w) => load_bf16(
                w,
                &[&format!("model.layers.{idx}.input_layernorm.weight")],
                &[cfg.hidden_size],
            ),
        }
    }

    fn layer(&self, cfg: &GptOssConfig, idx: usize) -> Result<HostLayer> {
        match self {
            Self::Host(h) => Ok(h.layers[idx].clone()),
            Self::Loader(w) => load_layer(cfg, w, idx),
        }
    }
}

crate::wgpu_state_snapshot::impl_wgpu_state_snapshot!(GptOssWgpu, max_seq);
