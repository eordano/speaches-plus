#![allow(clippy::too_many_arguments)]

use crate::wgpu_backend::device::WgpuContext;
use crate::wgpu_backend::{compose, dispatch, Result, WgpuError};
pub(crate) use crate::wgpu_backend::pack::{pack_u16_odd_tail_zeroed_min_one_word as pack_u16, pack_u8_min_one_word as pack_u8};
use crate::wgpu_backend::pack::{unpack_u16_by_element as unpack_u16};

pub const WGSL: &str = include_str!("../../../wgsl/flash_decode.wgsl");

pub const WORKGROUP_SIZE: u32 = 256;
pub const WARPS: usize = 8;
pub const LANES: usize = 32;
pub const MAX_HEAD_DIM: usize = 512;
pub const DEFAULT_SPLITS: usize = 16;

pub const ENTRY_DECODE_F32: &str = "flash_decode_f32";
pub const ENTRY_STAGE1_BF16: &str = "flash_splitk_stage1_bf16kv";
pub const ENTRY_STAGE1_FP8: &str = "flash_splitk_stage1_fp8kv";
pub const ENTRY_STAGE1_FP8_SD: &str = "flash_splitk_stage1_fp8kv_sd";
pub const ENTRY_STAGE2: &str = "flash_splitk_stage2";
pub const ENTRY_WRITE_KV_F32: &str = "write_kv_from_f32";
pub const ENTRY_WRITE_KV_BF16: &str = "write_kv_from_bf16";
pub const ENTRY_STAGE1_BF16_MK: &str = "flash_splitk_stage1_bf16kv_mk";
pub const ENTRY_STAGE1_FP8_MK: &str = "flash_splitk_stage1_fp8kv_mk";
pub const ENTRY_STAGE2_MK: &str = "flash_splitk_stage2_mk";
pub const ENTRY_STAGE1_BF16_MK_U: &str = "flash_splitk_stage1_bf16kv_mk_u";
pub const ENTRY_STAGE1_FP8_MK_U: &str = "flash_splitk_stage1_fp8kv_mk_u";
pub const ENTRY_STAGE2_U: &str = "flash_splitk_stage2_u";
pub const ENTRY_STAGE2_MK_U: &str = "flash_splitk_stage2_mk_u";
pub const MAX_HEAD_DIM_MK: usize = 256;
pub const MAX_MK_ROWS: usize = 8;

const SCRATCH_BYTES: u32 = (MAX_HEAD_DIM as u32) * 4
    + WORKGROUP_SIZE * 4
    + (WARPS as u32) * 8
    + (WARPS as u32) * (MAX_HEAD_DIM as u32) * 4;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
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

pub fn normalize_splits(splits: usize) -> usize {
    match splits {
        8 | 16 | 32 => splits,
        _ => DEFAULT_SPLITS,
    }
}

pub const MAX_SPLITS: usize = 256;

pub const MAX_GQA_FOLD: usize = 8;

pub const GQA_FOLD_ENV: &str = "NV_WGPU_GQA_FOLD";

pub const SPLITS_ENV: &str = "NV_WGPU_FLASH_SPLITS";

pub const ENTRY_STAGE2_PK_RT: &str = "flash_splitk_stage2_pk_rt";

fn env_count(name: &str) -> Option<usize> {
    let raw = std::env::var(name).ok()?;
    let t = raw.trim().to_string();
    if t.is_empty() {
        return None;
    }
    Some(t.parse::<usize>().unwrap_or_else(|_| {
        panic!("{name}={t} must be a positive decimal count");
    }))
}

pub const DEFAULT_GQA_FOLD: usize = 2;

pub fn gqa_fold_env(group: usize) -> usize {
    let Some(n) = env_count(GQA_FOLD_ENV) else {
        return if group > 0 && group.is_multiple_of(DEFAULT_GQA_FOLD) {
            DEFAULT_GQA_FOLD
        } else {
            1
        };
    };
    if n <= 1 {
        return 1;
    }
    assert!(
        n <= MAX_GQA_FOLD,
        "{GQA_FOLD_ENV}={n} exceeds {MAX_GQA_FOLD}; each folded query head costs head_dim/32 more live \
         accumulators per thread and the register file is the binding constraint"
    );
    assert!(
        group > 0 && group.is_multiple_of(n),
        "{GQA_FOLD_ENV}={n} must divide the GQA group {group}: a workgroup may only fold query heads \
         that share one KV head, so the folded grid n_heads/{n} has to stay inside kv-head boundaries"
    );
    n
}

#[cfg(test)]
mod gqa_fold_default_tests {
    use super::*;

    #[test]
    fn the_default_fold_is_two_where_the_group_allows_it() {
        if std::env::var(GQA_FOLD_ENV).is_ok() {
            return;
        }
        assert_eq!(gqa_fold_env(2), 2);
        assert_eq!(gqa_fold_env(4), 2);
        assert_eq!(gqa_fold_env(8), 2);
    }

    #[test]
    fn the_default_never_folds_more_than_two() {
        if std::env::var(GQA_FOLD_ENV).is_ok() {
            return;
        }
        for group in [2usize, 4, 6, 8, 16] {
            assert!(
                gqa_fold_env(group) <= DEFAULT_GQA_FOLD,
                "group {group} folded {} by default; 4 and 8 were measured slower \
                 at every context",
                gqa_fold_env(group)
            );
        }
    }

    #[test]
    fn an_indivisible_group_falls_back_to_one() {
        if std::env::var(GQA_FOLD_ENV).is_ok() {
            return;
        }
        for group in [1usize, 3, 5, 7] {
            assert_eq!(gqa_fold_env(group), 1, "group {group}");
        }
        assert_eq!(gqa_fold_env(0), 1, "a zero group must not divide-by-zero");
    }
}

pub fn splits_env() -> usize {
    let Some(n) = env_count(SPLITS_ENV) else {
        return DEFAULT_SPLITS;
    };
    assert!(
        n >= 1 && n <= MAX_SPLITS,
        "{SPLITS_ENV}={n} must be 1..={MAX_SPLITS}; the split-k scratch is provisioned for \
         {MAX_SPLITS} partials per head and stage2 reduces them serially"
    );
    n
}

fn fold_prefix(hd_max: u32, sg: bool, fold: u32) -> String {
    format!("fdf{hd_max}{}{fold}", if sg { "sg" } else { "wb" })
}

pub fn fold_stage1_entry(hd_max: u32, sg: bool, fold: u32) -> String {
    format!("{}_stage1_fp8", fold_prefix(hd_max, sg, fold))
}

const FOLD_REDUCE_SG: &str = include_str!("../../../wgsl/flash_decode_fold_reduce_sg.wgsl");

const FOLD_REDUCE_WB: &str = include_str!("../../../wgsl/flash_decode_fold_reduce_wb.wgsl");

const FOLD_EPILOGUE: &str = include_str!("../../../wgsl/flash_decode_fold_epilogue.wgsl");

const FOLD_HEAD: &str = include_str!("../../../wgsl/flash_decode_fold_head.wgsl");

const FOLD_ROUNDS: &str = include_str!("../../../wgsl/flash_decode_fold_rounds.wgsl");

pub fn fold_stage1_entry_sd(hd_max: u32, sg: bool, fold: u32) -> String {
    format!("{}_stage1_fp8_sd", fold_prefix(hd_max, sg, fold))
}

pub fn fold_stage1_source_sd(hd_max: u32, sg: bool, fold: u32) -> String {
    let base = fold_stage1_source(hd_max, sg, fold);
    let entry = fold_stage1_entry(hd_max, sg, fold);
    let entry_sd = fold_stage1_entry_sd(hd_max, sg, fold);
    let sd = base
        .replacen(&format!("fn {entry}("), &format!("fn {entry_sd}("), 1)
        .replace("fd_k_fp8(", "fd_k_fp8_sd(")
        .replace("fd_v_fp8(", "fd_v_fp8_sd(")
        .replacen(
            "ks = fd_k_scales[sp * nkv + kvh];",
            "ks = fd_k_scales[sp * nkv + kvh] * bitcast<f32>(0x7B800000u);",
            1,
        )
        .replacen(
            "let vsc = fd_v_scales[sp * nkv + kvh];",
            "let vsc = fd_v_scales[sp * nkv + kvh] * bitcast<f32>(0x7B800000u);",
            1,
        );
    assert!(
        sd.contains(&entry_sd) && sd.contains("fd_k_fp8_sd(") && sd.contains("0x7B800000"),
        "fold shift-decode rewrite missed an anchor; the generated kernel would silently \
         apply exact-decode magnitudes against 2pow120-folded scales"
    );
    sd
}

pub const ENTRY_STAGE1_FP8_MK_SD: &str = "flash_splitk_stage1_fp8kv_mk_sd";
pub const ENTRY_STAGE1_FP8_MK_U_SD: &str = "flash_splitk_stage1_fp8kv_mk_u_sd";
pub const ENTRY_SMV2_STAGE1_FP8: &str = "flash_smv2_stage1_fp8kv";
pub const ENTRY_SMV2_STAGE1_FP8_SD: &str = "flash_smv2_stage1_fp8kv_sd";

const SD_SCALE_FOLD_2POW120: &str = " * bitcast<f32>(0x7B800000u)";

const MK_SD_K_SCALE_ANCHOR: &str = "ks = fd_k_scales[sp * nkv + kvh];";
const MK_SD_V_SCALE_ANCHOR: &str = "vs = fd_v_scales[sp * nkv + kvh];";
const SMV2_SD_V_SCALE_ANCHOR: &str = "let w_v = w * fd_v_scales[sp * nkv + kvh];";

fn stock_stage1_block(entry: &str) -> &'static str {
    let decl = format!("fn {entry}(");
    assert_eq!(
        WGSL.matches(decl.as_str()).count(),
        1,
        "stage1 shift-decode twin: `{decl}` must appear exactly once in flash_decode.wgsl; a renamed or \
         duplicated stock entry would make the twin silently rewrite the wrong kernel body"
    );
    let fn_pos = WGSL.find(decl.as_str()).unwrap();
    let head = WGSL[..fn_pos].rfind("@compute").unwrap_or_else(|| {
        panic!(
            "stage1 shift-decode twin: no @compute attribute precedes {entry}; the extracted twin \
             would not be an entry point and its pipeline creation would fail at model load"
        )
    });
    let end = WGSL[fn_pos..].find("\n}").unwrap_or_else(|| {
        panic!(
            "stage1 shift-decode twin: {entry} has no column-zero closing brace; the extracted twin \
             would be truncated mid-body"
        )
    });
    &WGSL[head..fn_pos + end + 2]
}

fn stage1_twin_sd(entry: &str, k_scale_anchor: &str, v_scale_anchor: &str) -> String {
    let block = stock_stage1_block(entry);
    for (which, scales, anchor) in [
        ("k", "fd_k_scales", k_scale_anchor),
        ("v", "fd_v_scales", v_scale_anchor),
    ] {
        assert_eq!(
            block.matches(anchor).count(),
            1,
            "stage1 shift-decode twin for {entry}: {which}-scale anchor `{anchor}` must appear exactly \
             once; a missed anchor silently applies exact-decode magnitudes against 2pow120-folded scales"
        );
        assert_eq!(
            block.matches(scales).count(),
            1,
            "stage1 shift-decode twin for {entry}: a {scales} read outside the anchored line would \
             escape the 2pow120 fold and scale shift-decoded values 2pow120 too small"
        );
    }
    assert!(
        block.contains("fd_k_fp8(") && block.contains("fd_v_fp8("),
        "stage1 shift-decode twin for {entry}: the stock body no longer calls the exact fp8 decoders, \
         so a twin would change nothing while claiming the shift-decode speedup"
    );
    let folded = |anchor: &str| {
        format!(
            "{}{SD_SCALE_FOLD_2POW120};",
            anchor.strip_suffix(';').expect("scale anchors end the statement")
        )
    };
    let sd = block
        .replacen(&decl_of(entry), &decl_of(&format!("{entry}_sd")), 1)
        .replace("fd_k_fp8(", "fd_k_fp8_sd(")
        .replace("fd_v_fp8(", "fd_v_fp8_sd(")
        .replacen(k_scale_anchor, &folded(k_scale_anchor), 1)
        .replacen(v_scale_anchor, &folded(v_scale_anchor), 1);
    assert!(
        sd.contains(&decl_of(&format!("{entry}_sd")))
            && sd.contains("fd_k_fp8_sd(")
            && sd.contains("fd_v_fp8_sd(")
            && sd.matches("0x7B800000").count() == 2,
        "stage1 shift-decode rewrite for {entry} missed an anchor; the generated kernel would silently \
         apply exact-decode magnitudes against 2pow120-folded scales"
    );
    sd
}

fn decl_of(entry: &str) -> String {
    format!("fn {entry}(")
}

pub fn mk_stage1_source_sd() -> String {
    stage1_twin_sd(ENTRY_STAGE1_FP8_MK, MK_SD_K_SCALE_ANCHOR, MK_SD_V_SCALE_ANCHOR)
}

pub fn mk_u_stage1_source_sd() -> String {
    stage1_twin_sd(ENTRY_STAGE1_FP8_MK_U, MK_SD_K_SCALE_ANCHOR, MK_SD_V_SCALE_ANCHOR)
}

pub fn smv2_stage1_source_sd() -> String {
    stage1_twin_sd(ENTRY_SMV2_STAGE1_FP8, MK_SD_K_SCALE_ANCHOR, SMV2_SD_V_SCALE_ANCHOR)
}

pub fn fold_stage1_source(hd_max: u32, sg: bool, fold: u32) -> String {
    use std::fmt::Write;
    assert!(
        hd_max > 0 && hd_max % 32 == 0 && hd_max as usize <= MAX_HEAD_DIM,
        "fold stage1 hd_max {hd_max} must be a positive multiple of 32 up to {MAX_HEAD_DIM}"
    );
    assert!(
        fold >= 1 && fold as usize <= MAX_GQA_FOLD,
        "fold stage1 folds {fold} query heads per workgroup; 1..={MAX_GQA_FOLD} keeps the per-thread \
         accumulator file bounded"
    );
    let accs = hd_max / 32;
    let p = fold_prefix(hd_max, sg, fold);
    let entry = fold_stage1_entry(hd_max, sg, fold);
    let sub = |s: &str| {
        s.replace("{P}", &p)
            .replace("{E}", &entry)
            .replace("{F}", &fold.to_string())
            .replace("{HD}", &hd_max.to_string())
    };
    let mut b = String::with_capacity(16384);

    writeln!(b, "var<workgroup> {p}_qsh: array<f32, {}>;", hd_max * fold).unwrap();
    writeln!(b, "var<workgroup> {p}_sacc: array<f32, {}>;", hd_max * 8).unwrap();
    writeln!(b, "var<workgroup> {p}_sm: array<f32, 8>;").unwrap();
    writeln!(b, "var<workgroup> {p}_sl: array<f32, 8>;").unwrap();
    b.push_str(&sub(if sg { FOLD_REDUCE_SG } else { FOLD_REDUCE_WB }));
    b.push_str(&sub(FOLD_EPILOGUE));
    b.push_str(&sub(FOLD_HEAD));
    for j in 0..fold {
        writeln!(b, "    var m{j} = fd_neg_inf();").unwrap();
        writeln!(b, "    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "    var a{j}_{i} = 0.0;").unwrap();
        }
    }
    b.push_str(&sub(FOLD_ROUNDS));
    for j in 0..fold {
        writeln!(b, "        var pt{j} = 0.0;").unwrap();
    }
    writeln!(b, "        var ks = 0.0;").unwrap();
    writeln!(b, "        if (live) {{").unwrap();
    writeln!(b, "            let kbase = (sp * nkv + kvh) * hd;").unwrap();
    writeln!(b, "            ks = fd_k_scales[sp * nkv + kvh];").unwrap();
    writeln!(b, "            if (use_vec4) {{").unwrap();
    writeln!(b, "                let n4 = hd >> 2u;").unwrap();
    writeln!(
        b,
        "                for (var jv = lane; jv < n4; jv = jv + FD_LANES) {{"
    )
    .unwrap();
    writeln!(b, "                    let qb = jv * 4u;").unwrap();
    writeln!(b, "                    let kb = kbase + qb;").unwrap();
    for i in 0..4 {
        writeln!(
            b,
            "                    let e{i} = fd_k_fp8(kb + {i}u);"
        )
        .unwrap();
    }
    for j in 0..fold {
        writeln!(b, "                    {{").unwrap();
        writeln!(b, "                        let qo = {j}u * hd + qb;").unwrap();
        writeln!(b, "                        var t = {p}_qsh[qo + 1u] * e1;").unwrap();
        writeln!(b, "                        t = fma({p}_qsh[qo], e0, t);").unwrap();
        writeln!(b, "                        t = fma({p}_qsh[qo + 2u], e2, t);").unwrap();
        writeln!(b, "                        t = fma({p}_qsh[qo + 3u], e3, t);").unwrap();
        writeln!(b, "                        pt{j} = pt{j} + t;").unwrap();
        writeln!(b, "                    }}").unwrap();
    }
    writeln!(b, "                }}").unwrap();
    writeln!(b, "            }} else {{").unwrap();
    writeln!(
        b,
        "                for (var d = lane; d < hd; d = d + FD_LANES) {{"
    )
    .unwrap();
    writeln!(b, "                    let kx = fd_k_fp8(kbase + d);").unwrap();
    for j in 0..fold {
        writeln!(
            b,
            "                    pt{j} = fma({p}_qsh[{j}u * hd + d], kx, pt{j});"
        )
        .unwrap();
    }
    writeln!(b, "                }}").unwrap();
    writeln!(b, "            }}").unwrap();
    writeln!(b, "        }}").unwrap();
    for j in 0..fold {
        writeln!(
            b,
            "        let sc{j} = ({p}_reduce(lid, pt{j}) * ks) * fd_params.scaling;"
        )
        .unwrap();
    }
    writeln!(b, "        if (live) {{").unwrap();
    writeln!(b, "            let vbase = (sp * nkv + kvh) * hd;").unwrap();
    writeln!(b, "            let vsc = fd_v_scales[sp * nkv + kvh];").unwrap();
    for j in 0..fold {
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
        writeln!(b, "                    let vx = fd_v_fp8(vbase + d);").unwrap();
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
    writeln!(b, "    }}\n").unwrap();
    for j in 0..fold {
        writeln!(b, "    workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(b, "    {{").unwrap();
            writeln!(b, "        let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(b, "        if (d < hd) {{").unwrap();
            writeln!(
                b,
                "            {p}_sacc[warp * {hd_max}u + d] = a{j}_{i};"
            )
            .unwrap();
            writeln!(b, "        }}").unwrap();
            writeln!(b, "    }}").unwrap();
        }
        writeln!(
            b,
            "    {p}_epilogue(lid, lane, warp, hd, ((h0 + {j}u) * fd_params.splits + split) * (hd + \
             2u), m{j}, l{j});"
        )
        .unwrap();
    }
    writeln!(b, "}}").unwrap();
    b
}

pub const KV_NVFP4_STAGE1_BINDS_TEN_STORAGE_BUFFERS: u32 = 10;

fn nvfp4_arm_tag(k4: bool) -> &'static str {
    if k4 {
        "k4v4"
    } else {
        "v4k8"
    }
}

pub fn fold_stage1_entry_nvfp4(hd_max: u32, sg: bool, fold: u32, k4: bool) -> String {
    format!(
        "fdf{hd_max}{}{fold}{}_stage1_nvfp4",
        if sg { "sg" } else { "wb" },
        nvfp4_arm_tag(k4)
    )
}

pub fn fold_stage1_source_nvfp4(hd_max: u32, sg: bool, fold: u32, k4: bool) -> String {
    use std::fmt::Write;
    assert!(
        hd_max > 0 && hd_max % 32 == 0 && hd_max as usize <= MAX_HEAD_DIM,
        "nvfp4 fold stage1 hd_max {hd_max} must be a positive multiple of 32 up to {MAX_HEAD_DIM}"
    );
    assert!(
        hd_max % 8 == 0,
        "nvfp4 fold stage1 reads 8 e2m1 nibbles per u32 word; hd {hd_max} % 8 != 0 has no word tiling"
    );
    assert!(
        fold >= 1 && fold as usize <= MAX_GQA_FOLD,
        "nvfp4 fold stage1 folds {fold} query heads per workgroup; 1..={MAX_GQA_FOLD} keeps the \
         per-thread accumulator file bounded"
    );
    let sink = crate::wgpu_backend::kernels::kv_nvfp4::KV_NVFP4_SINK_SLOTS_STAY_FP8_TO_ANCHOR_SOFTMAX;
    let bt =
        crate::wgpu_backend::kernels::kv_nvfp4::KV_NVFP4_K_BLOCK_TOKENS_A_BLOCK_FINALIZES_THE_STEP_ITS_LAST_TOKEN_LANDS;
    let accs = hd_max / 32;
    let p = format!(
        "fdf{hd_max}{}{fold}{}",
        if sg { "sg" } else { "wb" },
        nvfp4_arm_tag(k4)
    );
    let entry = fold_stage1_entry_nvfp4(hd_max, sg, fold, k4);
    let sub = |s: &str| {
        s.replace("{P}", &p)
            .replace("{E}", &entry)
            .replace("{F}", &fold.to_string())
            .replace("{HD}", &hd_max.to_string())
    };
    let mut b = String::with_capacity(24576);
    writeln!(b, "@group(0) @binding(15) var<storage, read> fd_v4_words: array<u32>;").unwrap();
    writeln!(b, "@group(0) @binding(17) var<storage, read> fd_v4_scales: array<f32>;").unwrap();
    if k4 {
        writeln!(b, "@group(0) @binding(14) var<storage, read> fd_k4_words: array<u32>;").unwrap();
        writeln!(b, "@group(0) @binding(16) var<storage, read> fd_k4_scales: array<f32>;").unwrap();
    }
    writeln!(b, "fn {p}_e2m1x2(nib: u32) -> f32 {{").unwrap();
    writeln!(b, "    let n = nib & 15u;").unwrap();
    writeln!(b, "    let m = n & 1u;").unwrap();
    writeln!(b, "    let e = (n >> 1u) & 3u;").unwrap();
    writeln!(b, "    let t = select(((2u + m) << e) >> 1u, m, e == 0u);").unwrap();
    writeln!(b, "    let ti = select(i32(t), -i32(t), (n & 8u) != 0u);").unwrap();
    writeln!(b, "    return f32(ti);").unwrap();
    writeln!(b, "}}").unwrap();
    writeln!(b, "var<workgroup> {p}_qsh: array<f32, {}>;", hd_max * fold).unwrap();
    writeln!(b, "var<workgroup> {p}_sacc: array<f32, {}>;", hd_max * 8).unwrap();
    writeln!(b, "var<workgroup> {p}_sm: array<f32, 8>;").unwrap();
    writeln!(b, "var<workgroup> {p}_sl: array<f32, 8>;").unwrap();
    b.push_str(&sub(if sg { FOLD_REDUCE_SG } else { FOLD_REDUCE_WB }));
    b.push_str(&sub(FOLD_EPILOGUE));
    b.push_str(&sub(FOLD_HEAD));
    for j in 0..fold {
        writeln!(b, "    var m{j} = fd_neg_inf();").unwrap();
        writeln!(b, "    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "    var a{j}_{i} = 0.0;").unwrap();
        }
    }
    b.push_str(&sub(FOLD_ROUNDS));
    for j in 0..fold {
        writeln!(b, "        var pt{j} = 0.0;").unwrap();
    }
    writeln!(b, "        var ks = 0.0;").unwrap();
    writeln!(b, "        let snk = sp < {sink}u;").unwrap();
    let emit_k_fp8_vec4 = |b: &mut String, indent: &str| {
        writeln!(b, "{indent}let n4 = hd >> 2u;").unwrap();
        writeln!(b, "{indent}for (var jv = lane; jv < n4; jv = jv + FD_LANES) {{").unwrap();
        writeln!(b, "{indent}    let qb = jv * 4u;").unwrap();
        writeln!(b, "{indent}    let kb = kbase + qb;").unwrap();
        for i in 0..4 {
            writeln!(b, "{indent}    let e{i} = fd_k_fp8(kb + {i}u);").unwrap();
        }
        for j in 0..fold {
            writeln!(b, "{indent}    {{").unwrap();
            writeln!(b, "{indent}        let qo = {j}u * hd + qb;").unwrap();
            writeln!(b, "{indent}        var t = {p}_qsh[qo + 1u] * e1;").unwrap();
            writeln!(b, "{indent}        t = fma({p}_qsh[qo], e0, t);").unwrap();
            writeln!(b, "{indent}        t = fma({p}_qsh[qo + 2u], e2, t);").unwrap();
            writeln!(b, "{indent}        t = fma({p}_qsh[qo + 3u], e3, t);").unwrap();
            writeln!(b, "{indent}        pt{j} = pt{j} + t;").unwrap();
            writeln!(b, "{indent}    }}").unwrap();
        }
        writeln!(b, "{indent}}}").unwrap();
    };
    writeln!(b, "        if (live) {{").unwrap();
    writeln!(b, "            let kbase = (sp * nkv + kvh) * hd;").unwrap();
    if k4 {
        writeln!(b, "            if (snk) {{").unwrap();
        writeln!(b, "                ks = fd_k_scales[sp * nkv + kvh];").unwrap();
        emit_k_fp8_vec4(&mut b, "                ");
        writeln!(b, "            }} else {{").unwrap();
        writeln!(b, "                ks = 1.0;").unwrap();
        writeln!(b, "                let ksb = ((sp / {bt}u) * nkv + kvh) * hd;").unwrap();
        writeln!(b, "                let n8 = hd >> 3u;").unwrap();
        writeln!(b, "                for (var jw = lane; jw < n8; jw = jw + FD_LANES) {{").unwrap();
        writeln!(b, "                    let w = fd_k4_words[(kbase >> 3u) + jw];").unwrap();
        writeln!(b, "                    let qb = jw * 8u;").unwrap();
        for i in 0..8u32 {
            writeln!(
                b,
                "                    let e{i} = {p}_e2m1x2(w >> {}u) * (fd_k4_scales[ksb + qb + {i}u] * 0.5);",
                4 * i
            )
            .unwrap();
        }
        for j in 0..fold {
            writeln!(b, "                    {{").unwrap();
            writeln!(b, "                        let qo = {j}u * hd + qb;").unwrap();
            writeln!(b, "                        var t = {p}_qsh[qo + 1u] * e1;").unwrap();
            writeln!(b, "                        t = fma({p}_qsh[qo], e0, t);").unwrap();
            for i in 2..8 {
                writeln!(
                    b,
                    "                        t = fma({p}_qsh[qo + {i}u], e{i}, t);"
                )
                .unwrap();
            }
            writeln!(b, "                        pt{j} = pt{j} + t;").unwrap();
            writeln!(b, "                    }}").unwrap();
        }
        writeln!(b, "                }}").unwrap();
        writeln!(b, "            }}").unwrap();
    } else {
        writeln!(b, "            ks = fd_k_scales[sp * nkv + kvh];").unwrap();
        emit_k_fp8_vec4(&mut b, "            ");
    }
    writeln!(b, "        }}").unwrap();
    for j in 0..fold {
        writeln!(
            b,
            "        let sc{j} = ({p}_reduce(lid, pt{j}) * ks) * fd_params.scaling;"
        )
        .unwrap();
    }
    writeln!(b, "        if (live) {{").unwrap();
    writeln!(b, "            let vbase = (sp * nkv + kvh) * hd;").unwrap();
    writeln!(b, "            var vsc = 0.0;").unwrap();
    writeln!(b, "            if (snk) {{").unwrap();
    writeln!(b, "                vsc = fd_v_scales[sp * nkv + kvh];").unwrap();
    writeln!(b, "            }} else {{").unwrap();
    writeln!(b, "                vsc = fd_v4_scales[sp * nkv + kvh] * 0.5;").unwrap();
    writeln!(b, "            }}").unwrap();
    for j in 0..fold {
        writeln!(b, "            let mn{j} = max(m{j}, sc{j});").unwrap();
        writeln!(b, "            let cr{j} = fd_exp(m{j} - mn{j});").unwrap();
        writeln!(b, "            let wt{j} = fd_exp(sc{j} - mn{j});").unwrap();
        writeln!(b, "            l{j} = fma(l{j}, cr{j}, wt{j});").unwrap();
        writeln!(b, "            let wv{j} = wt{j} * vsc;").unwrap();
    }
    writeln!(b, "            if (snk) {{").unwrap();
    for i in 0..accs {
        writeln!(b, "                {{").unwrap();
        writeln!(b, "                    let d = lane + {i}u * FD_LANES;").unwrap();
        writeln!(b, "                    if (d < hd) {{").unwrap();
        writeln!(b, "                        let vx = fd_v_fp8(vbase + d);").unwrap();
        for j in 0..fold {
            writeln!(
                b,
                "                        a{j}_{i} = fma(wv{j}, vx, a{j}_{i} * cr{j});"
            )
            .unwrap();
        }
        writeln!(b, "                    }}").unwrap();
        writeln!(b, "                }}").unwrap();
    }
    writeln!(b, "            }} else {{").unwrap();
    for i in 0..accs {
        writeln!(b, "                {{").unwrap();
        writeln!(b, "                    let d = lane + {i}u * FD_LANES;").unwrap();
        writeln!(b, "                    if (d < hd) {{").unwrap();
        writeln!(b, "                        let vi = vbase + d;").unwrap();
        writeln!(
            b,
            "                        let vx = {p}_e2m1x2(fd_v4_words[vi >> 3u] >> (4u * (vi & 7u)));"
        )
        .unwrap();
        for j in 0..fold {
            writeln!(
                b,
                "                        a{j}_{i} = fma(wv{j}, vx, a{j}_{i} * cr{j});"
            )
            .unwrap();
        }
        writeln!(b, "                    }}").unwrap();
        writeln!(b, "                }}").unwrap();
    }
    writeln!(b, "            }}").unwrap();
    for j in 0..fold {
        writeln!(b, "            m{j} = mn{j};").unwrap();
    }
    writeln!(b, "        }}").unwrap();
    writeln!(b, "    }}\n").unwrap();
    for j in 0..fold {
        writeln!(b, "    workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(b, "    {{").unwrap();
            writeln!(b, "        let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(b, "        if (d < hd) {{").unwrap();
            writeln!(b, "            {p}_sacc[warp * {hd_max}u + d] = a{j}_{i};").unwrap();
            writeln!(b, "        }}").unwrap();
            writeln!(b, "    }}").unwrap();
        }
        writeln!(
            b,
            "    {p}_epilogue(lid, lane, warp, hd, ((h0 + {j}u) * fd_params.splits + split) * (hd + \
             2u), m{j}, l{j});"
        )
        .unwrap();
    }
    writeln!(b, "}}").unwrap();
    assert!(
        b.contains("_e2m1x2(") && b.contains("fd_v4_words") && b.contains("* 0.5"),
        "nvfp4 fold rewrite lost its int8x2 e2m1 V reads or the 0.5 fold that makes them exact; \
         the generated kernel would silently measure the fp8 arm under an nvfp4 name"
    );
    assert!(
        !k4 || (b.contains("fd_k4_words") && b.contains("fd_k4_scales") && b.contains("ks = 1.0;")),
        "k4v4 fold rewrite lost its per-channel K reads or still factors a row scale out of the \
         dot; per-channel scales multiply per element and the row scale must be identity"
    );
    assert!(
        !b.contains("_sd(") && !b.contains("0x7B800000"),
        "nvfp4 fold generated a shift-decode read; sd K measured worse ppl for zero depth win on \
         this arm (runs.jsonl kv-v4k8 pair), and e2m1 sd is unsound outright because the adapter \
         flushes f32 denormals, which lands on e2m1's 0.5 codes"
    );
    b
}

pub fn fold_stage1_entry_sd_tiled(hd_max: u32, fold: u32, tile: u32) -> String {
    format!("fdf{hd_max}sg{fold}t{tile}_stage1_fp8_sd")
}

pub fn fold_stage1_source_sd_tiled(hd_max: u32, fold: u32, tile: u32) -> String {
    use std::fmt::Write;
    assert!(
        hd_max > 0 && hd_max % 32 == 0 && hd_max as usize <= MAX_HEAD_DIM,
        "tiled fold stage1 hd_max {hd_max} must be a positive multiple of 32 up to {MAX_HEAD_DIM}"
    );
    assert!(
        hd_max % 4 == 0,
        "tiled fold stage1 emits only the vec4 lane-owns-word K path; hd {hd_max} % 4 != 0 \
         would read one u32 from four lanes redundantly (the v1 defect)"
    );
    assert!(
        fold >= 1 && fold as usize <= MAX_GQA_FOLD,
        "tiled fold stage1 folds {fold} query heads; 1..={MAX_GQA_FOLD}"
    );
    assert!(
        (2..=8).contains(&tile) && fold * tile <= 48,
        "tile {tile} must be 2..=8 and fold*tile <= 48 or the per-lane score register file \
         spills and the tile amortization inverts"
    );
    let accs = hd_max / 32;
    let p = format!("fdt{hd_max}sg{fold}t{tile}");
    let entry = fold_stage1_entry_sd_tiled(hd_max, fold, tile);
    let sub = |s: &str| {
        s.replace("{P}", &p)
            .replace("{E}", &entry)
            .replace("{F}", &fold.to_string())
            .replace("{HD}", &hd_max.to_string())
    };
    let mut b = String::with_capacity(32768);
    writeln!(b, "var<workgroup> {p}_qsh: array<f32, {}>;", hd_max * fold).unwrap();
    writeln!(b, "var<workgroup> {p}_sacc: array<f32, {}>;", hd_max * 8).unwrap();
    writeln!(b, "var<workgroup> {p}_sm: array<f32, 8>;").unwrap();
    writeln!(b, "var<workgroup> {p}_sl: array<f32, 8>;").unwrap();
    b.push_str(&sub(FOLD_REDUCE_SG));
    b.push_str(&sub(FOLD_EPILOGUE));
    b.push_str(&sub(FOLD_HEAD));
    for j in 0..fold {
        writeln!(b, "    var m{j} = fd_neg_inf();").unwrap();
        writeln!(b, "    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "    var a{j}_{i} = 0.0;").unwrap();
        }
    }
    writeln!(b, "    let total = fd_params.total;").unwrap();
    writeln!(b, "    let base = fd_params.start + split * (FD_WARPS * {tile}u);").unwrap();
    writeln!(b, "    let stride = fd_params.splits * FD_WARPS * {tile}u;").unwrap();
    writeln!(b, "    var rounds = 0u;").unwrap();
    writeln!(b, "    if (total > base) {{").unwrap();
    writeln!(b, "        rounds = (total - base + stride - 1u) / stride;").unwrap();
    writeln!(b, "    }}").unwrap();
    writeln!(b, "    for (var r = 0u; r < rounds; r = r + 1u) {{").unwrap();
    writeln!(b, "        let p0 = base + warp * {tile}u + r * stride;").unwrap();
    for t in 0..tile {
        writeln!(b, "        let pp{t} = p0 + {t}u;").unwrap();
        writeln!(b, "        let lv{t} = pp{t} < total;").unwrap();
        writeln!(b, "        var sp{t} = pp{t};").unwrap();
        writeln!(b, "        if (fd_params.ring > 0u) {{ sp{t} = pp{t} % fd_params.ring; }}").unwrap();
    }
    writeln!(b, "        if (lv0) {{").unwrap();
    for t in 0..tile {
        for j in 0..fold {
            writeln!(b, "            var pt{j}_{t} = 0.0;").unwrap();
        }
        writeln!(b, "            var ks{t} = 0.0;").unwrap();
        writeln!(b, "            var vs{t} = 0.0;").unwrap();
        writeln!(b, "            if (lv{t}) {{").unwrap();
        writeln!(b, "                let kb{t} = (sp{t} * nkv + kvh) * hd;").unwrap();
        writeln!(
            b,
            "                ks{t} = fd_k_scales[sp{t} * nkv + kvh] * bitcast<f32>(0x7B800000u);"
        )
        .unwrap();
        writeln!(
            b,
            "                vs{t} = fd_v_scales[sp{t} * nkv + kvh] * bitcast<f32>(0x7B800000u);"
        )
        .unwrap();
        writeln!(b, "                let n4 = hd >> 2u;").unwrap();
        writeln!(
            b,
            "                for (var jv = lane; jv < n4; jv = jv + FD_LANES) {{"
        )
        .unwrap();
        writeln!(b, "                    let qb = jv * 4u;").unwrap();
        writeln!(b, "                    let kk = kb{t} + qb;").unwrap();
        for i in 0..4 {
            writeln!(b, "                    let e{i} = fd_k_fp8_sd(kk + {i}u);").unwrap();
        }
        for j in 0..fold {
            writeln!(b, "                    {{").unwrap();
            writeln!(b, "                        let qo = {j}u * hd + qb;").unwrap();
            writeln!(b, "                        var tt = {p}_qsh[qo + 1u] * e1;").unwrap();
            writeln!(b, "                        tt = fma({p}_qsh[qo], e0, tt);").unwrap();
            writeln!(b, "                        tt = fma({p}_qsh[qo + 2u], e2, tt);").unwrap();
            writeln!(b, "                        tt = fma({p}_qsh[qo + 3u], e3, tt);").unwrap();
            writeln!(b, "                        pt{j}_{t} = pt{j}_{t} + tt;").unwrap();
            writeln!(b, "                    }}").unwrap();
        }
        writeln!(b, "                }}").unwrap();
        writeln!(b, "            }}").unwrap();
        for j in 0..fold {
            writeln!(
                b,
                "            let rs{j}_{t} = ({p}_reduce(lid, pt{j}_{t}) * ks{t}) * fd_params.scaling;"
            )
            .unwrap();
            writeln!(
                b,
                "            let sc{j}_{t} = select(fd_neg_inf(), rs{j}_{t}, lv{t});"
            )
            .unwrap();
        }
    }
    for j in 0..fold {
        write!(b, "            let mx{j} = max(m{j}, ").unwrap();
        let mut expr = format!("sc{j}_0");
        for t in 1..tile {
            expr = format!("max({expr}, sc{j}_{t})");
        }
        writeln!(b, "{expr});").unwrap();
        writeln!(b, "            let corr{j} = fd_exp(m{j} - mx{j});").unwrap();
        for t in 0..tile {
            writeln!(
                b,
                "            let w{j}_{t} = select(0.0, fd_exp(sc{j}_{t} - mx{j}), lv{t});"
            )
            .unwrap();
        }
        write!(b, "            l{j} = fma(l{j}, corr{j}, ").unwrap();
        let mut sum = format!("w{j}_0");
        for t in 1..tile {
            sum = format!("({sum} + w{j}_{t})");
        }
        writeln!(b, "{sum});").unwrap();
        writeln!(b, "            m{j} = mx{j};").unwrap();
    }
    for t in 0..tile {
        writeln!(b, "            let vb{t} = (sp{t} * nkv + kvh) * hd;").unwrap();
        for j in 0..fold {
            writeln!(b, "            let wv{j}_{t} = w{j}_{t} * vs{t};").unwrap();
        }
    }
    for i in 0..accs {
        writeln!(b, "            {{").unwrap();
        writeln!(b, "                let d = lane + {i}u * FD_LANES;").unwrap();
        writeln!(b, "                if (d < hd) {{").unwrap();
        for j in 0..fold {
            writeln!(b, "                    a{j}_{i} = a{j}_{i} * corr{j};").unwrap();
        }
        for t in 0..tile {
            writeln!(
                b,
                "                    let vx{t} = select(0.0, fd_v_fp8_sd(vb{t} + d), lv{t});"
            )
            .unwrap();
            for j in 0..fold {
                writeln!(
                    b,
                    "                    a{j}_{i} = fma(wv{j}_{t}, vx{t}, a{j}_{i});"
                )
                .unwrap();
            }
        }
        writeln!(b, "                }}").unwrap();
        writeln!(b, "            }}").unwrap();
    }
    writeln!(b, "        }}").unwrap();
    writeln!(b, "    }}\n").unwrap();
    for j in 0..fold {
        writeln!(b, "    workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(b, "    {{").unwrap();
            writeln!(b, "        let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(b, "        if (d < hd) {{").unwrap();
            writeln!(b, "            {p}_sacc[warp * {hd_max}u + d] = a{j}_{i};").unwrap();
            writeln!(b, "        }}").unwrap();
            writeln!(b, "    }}").unwrap();
        }
        writeln!(
            b,
            "    {p}_epilogue(lid, lane, warp, hd, ((h0 + {j}u) * fd_params.splits + split) * (hd + \
             2u), m{j}, l{j});"
        )
        .unwrap();
    }
    writeln!(b, "}}").unwrap();
    b
}

const FOLD_REDUCE_SA: &str = include_str!("../../../wgsl/flash_decode_fold_reduce_sa.wgsl");

pub fn fold_stage1_entry_sd_ra(hd_max: u32, fold: u32) -> String {
    format!("fdf{hd_max}ra{fold}_stage1_fp8_sd")
}

pub fn fold_stage1_source_sd_ra(hd_max: u32, fold: u32) -> String {
    let sg_prefix = fold_prefix(hd_max, true, fold);
    let ra_prefix = format!("fdf{hd_max}ra{fold}");
    let base = fold_stage1_source_sd(hd_max, true, fold).replace(&sg_prefix, &ra_prefix);
    let butterfly = FOLD_REDUCE_SG.replace("{P}", &ra_prefix);
    let single_op = FOLD_REDUCE_SA.replace("{P}", &ra_prefix);
    assert_eq!(
        base.matches(&butterfly).count(),
        1,
        "reduce-lite rewrite: the 5-shuffle butterfly must appear exactly once in the sg fold \
         source; a missed anchor would ship a renamed butterfly and measure nothing"
    );
    let out = base.replacen(&butterfly, &single_op, 1);
    assert!(
        out.contains("subgroupAdd") && !out.contains("subgroupShuffleXor"),
        "reduce-lite rewrite left butterfly shuffles behind; the variant would not isolate the \
         reduce cost"
    );
    out
}

pub fn fold_stage1_entry_sd_tp(hd_max: u32, fold: u32) -> String {
    format!("fdf{hd_max}tp{fold}_stage1_fp8_sd")
}

pub const K_TRANSPOSED_BINDING_MUST_BE_EXACT: &str =
    "the tp stage1 derives the slot stride from arrayLength(&fd_k_words), so the K binding must \
     be exactly slots * n_kv * head_dim / 4 words; a padded binding would shear every plane";

pub fn fold_stage1_source_sd_tp(hd_max: u32, fold: u32) -> String {
    use std::fmt::Write;
    assert!(
        hd_max > 0 && hd_max % 32 == 0 && hd_max as usize <= MAX_HEAD_DIM,
        "tp fold stage1 hd_max {hd_max} must be a positive multiple of 32 up to {MAX_HEAD_DIM}"
    );
    assert!(
        hd_max % 4 == 0,
        "tp fold stage1 reads K one whole u32 word per lane; hd {hd_max} % 4 != 0 has no word tiling"
    );
    assert!(
        fold >= 1 && fold as usize <= MAX_GQA_FOLD,
        "tp fold stage1 folds {fold} query heads; 1..={MAX_GQA_FOLD} keeps score+acc registers bounded"
    );
    let accs = hd_max / 32;
    let p = format!("fdf{hd_max}tp{fold}");
    let entry = fold_stage1_entry_sd_tp(hd_max, fold);
    let sub = |s: &str| {
        s.replace("{P}", &p)
            .replace("{E}", &entry)
            .replace("{F}", &fold.to_string())
            .replace("{HD}", &hd_max.to_string())
    };
    let mut b = String::with_capacity(24576);
    writeln!(b, "var<workgroup> {p}_qsh: array<f32, {}>;", hd_max * fold).unwrap();
    writeln!(b, "var<workgroup> {p}_sacc: array<f32, {}>;", hd_max * 8).unwrap();
    writeln!(b, "var<workgroup> {p}_sm: array<f32, 8>;").unwrap();
    writeln!(b, "var<workgroup> {p}_sl: array<f32, 8>;").unwrap();
    b.push_str(&sub(FOLD_EPILOGUE));
    b.push_str(&sub(FOLD_HEAD));
    for j in 0..fold {
        writeln!(b, "    var m{j} = fd_neg_inf();").unwrap();
        writeln!(b, "    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "    var a{j}_{i} = 0.0;").unwrap();
        }
    }
    writeln!(b, "    let total = fd_params.total;").unwrap();
    writeln!(
        b,
        "    let kt_slots = (arrayLength(&fd_k_words) * 4u) / (nkv * hd);"
    )
    .unwrap();
    writeln!(b, "    let kplane = kvh * (hd >> 2u);").unwrap();
    writeln!(b, "    let base = fd_params.start + split * FD_BLOCK;").unwrap();
    writeln!(b, "    let stride = fd_params.splits * FD_BLOCK;").unwrap();
    writeln!(b, "    var rounds = 0u;").unwrap();
    writeln!(b, "    if (total > base) {{").unwrap();
    writeln!(b, "        rounds = (total - base + stride - 1u) / stride;").unwrap();
    writeln!(b, "    }}").unwrap();
    writeln!(b, "    for (var r = 0u; r < rounds; r = r + 1u) {{").unwrap();
    writeln!(b, "        let pw = base + warp * FD_LANES + r * stride;").unwrap();
    writeln!(b, "        if (pw < total) {{").unwrap();
    writeln!(b, "            let pp = pw + lane;").unwrap();
    writeln!(b, "            let live = pp < total;").unwrap();
    writeln!(b, "            var sp = pp;").unwrap();
    writeln!(
        b,
        "            if (fd_params.ring > 0u) {{ sp = pp % fd_params.ring; }}"
    )
    .unwrap();
    writeln!(b, "            var ks = 0.0;").unwrap();
    writeln!(b, "            var vs = 0.0;").unwrap();
    writeln!(b, "            if (live) {{").unwrap();
    writeln!(
        b,
        "                ks = fd_k_scales[sp * nkv + kvh] * bitcast<f32>(0x7B800000u);"
    )
    .unwrap();
    writeln!(
        b,
        "                vs = fd_v_scales[sp * nkv + kvh] * bitcast<f32>(0x7B800000u);"
    )
    .unwrap();
    writeln!(b, "            }}").unwrap();
    for j in 0..fold {
        writeln!(b, "            var s{j} = 0.0;").unwrap();
    }
    writeln!(b, "            if (live) {{").unwrap();
    writeln!(
        b,
        "                for (var d4 = 0u; d4 < (hd >> 2u); d4 = d4 + 1u) {{"
    )
    .unwrap();
    writeln!(
        b,
        "                    let kw = fd_k_words[(kplane + d4) * kt_slots + sp];"
    )
    .unwrap();
    for i in 0..4u32 {
        writeln!(
            b,
            "                    let e{i} = e4m3_shift_decode_scale_must_carry_2pow120(byte_at(kw, {i}u));"
        )
        .unwrap();
    }
    writeln!(b, "                    let qb = d4 << 2u;").unwrap();
    for j in 0..fold {
        writeln!(b, "                    {{").unwrap();
        writeln!(b, "                        let qo = {j}u * hd + qb;").unwrap();
        writeln!(b, "                        var t = {p}_qsh[qo + 1u] * e1;").unwrap();
        writeln!(b, "                        t = fma({p}_qsh[qo], e0, t);").unwrap();
        writeln!(b, "                        t = fma({p}_qsh[qo + 2u], e2, t);").unwrap();
        writeln!(b, "                        t = fma({p}_qsh[qo + 3u], e3, t);").unwrap();
        writeln!(b, "                        s{j} = s{j} + t;").unwrap();
        writeln!(b, "                    }}").unwrap();
    }
    writeln!(b, "                }}").unwrap();
    writeln!(b, "            }}").unwrap();
    for j in 0..fold {
        writeln!(
            b,
            "            let sc{j} = select(fd_neg_inf(), (s{j} * ks) * fd_params.scaling, live);"
        )
        .unwrap();
        writeln!(b, "            let mn{j} = max(m{j}, subgroupMax(sc{j}));").unwrap();
        writeln!(b, "            let cr{j} = fd_exp(m{j} - mn{j});").unwrap();
        writeln!(
            b,
            "            let w{j} = select(0.0, fd_exp(sc{j} - mn{j}), live);"
        )
        .unwrap();
        writeln!(b, "            l{j} = fma(l{j}, cr{j}, subgroupAdd(w{j}));").unwrap();
        writeln!(b, "            m{j} = mn{j};").unwrap();
        writeln!(b, "            let wv{j} = w{j} * vs;").unwrap();
    }
    for i in 0..accs {
        for j in 0..fold {
            writeln!(b, "            a{j}_{i} = a{j}_{i} * cr{j};").unwrap();
        }
    }
    writeln!(
        b,
        "            for (var pi = 0u; pi < FD_LANES; pi = pi + 1u) {{"
    )
    .unwrap();
    writeln!(b, "                let spv = subgroupShuffle(sp, pi);").unwrap();
    for j in 0..fold {
        writeln!(b, "                let bw{j} = subgroupShuffle(wv{j}, pi);").unwrap();
    }
    let mut bsum = "bw0".to_string();
    for j in 1..fold {
        bsum = format!("({bsum} + bw{j})");
    }
    writeln!(b, "                if ({bsum} > 0.0) {{").unwrap();
    writeln!(b, "                    let vbase = (spv * nkv + kvh) * hd;").unwrap();
    for i in 0..accs {
        writeln!(b, "                    {{").unwrap();
        writeln!(b, "                        let d = lane + {i}u * FD_LANES;").unwrap();
        writeln!(b, "                        if (d < hd) {{").unwrap();
        writeln!(
            b,
            "                            let vx = fd_v_fp8_sd(vbase + d);"
        )
        .unwrap();
        for j in 0..fold {
            writeln!(
                b,
                "                            a{j}_{i} = fma(bw{j}, vx, a{j}_{i});"
            )
            .unwrap();
        }
        writeln!(b, "                        }}").unwrap();
        writeln!(b, "                    }}").unwrap();
    }
    writeln!(b, "                }}").unwrap();
    writeln!(b, "            }}").unwrap();
    writeln!(b, "        }}").unwrap();
    writeln!(b, "    }}\n").unwrap();
    for j in 0..fold {
        writeln!(b, "    workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(b, "    {{").unwrap();
            writeln!(b, "        let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(b, "        if (d < hd) {{").unwrap();
            writeln!(b, "            {p}_sacc[warp * {hd_max}u + d] = a{j}_{i};").unwrap();
            writeln!(b, "        }}").unwrap();
            writeln!(b, "    }}").unwrap();
        }
        writeln!(
            b,
            "    {p}_epilogue(lid, lane, warp, hd, ((h0 + {j}u) * fd_params.splits + split) * (hd + \
             2u), m{j}, l{j});"
        )
        .unwrap();
    }
    writeln!(b, "}}").unwrap();
    b
}

pub fn k_transposed_word_index(slots: usize, n_kv: usize, head_dim: usize, kvh: usize, pos: usize, d4: usize) -> usize {
    assert!(head_dim % 4 == 0 && d4 < head_dim / 4 && kvh < n_kv && pos < slots);
    (kvh * (head_dim / 4) + d4) * slots + pos
}

pub const SPLITS_MEASURED_BEST_SHORT_KV: usize = 16;

pub const SPLITS_MEASURED_BEST_DEEP_KV: usize = 64;

pub const SPLITS_DEEP_KV_FROM_TOTAL: usize = 4096;

pub fn splits_for(total: usize) -> usize {
    if env_count(SPLITS_ENV).is_some() {
        return splits_env();
    }
    if total >= SPLITS_DEEP_KV_FROM_TOTAL {
        SPLITS_MEASURED_BEST_DEEP_KV
    } else {
        SPLITS_MEASURED_BEST_SHORT_KV
    }
}

#[cfg(test)]
mod splits_for_tests {
    use super::*;

    #[test]
    fn splits_follow_the_measured_knee_between_2k_and_4k() {
        if std::env::var(SPLITS_ENV).is_ok() {
            return;
        }
        assert_eq!(splits_for(0), SPLITS_MEASURED_BEST_SHORT_KV);
        assert_eq!(splits_for(256), SPLITS_MEASURED_BEST_SHORT_KV);
        assert_eq!(splits_for(2048), SPLITS_MEASURED_BEST_SHORT_KV);
        assert_eq!(splits_for(4096), SPLITS_MEASURED_BEST_DEEP_KV);
        assert_eq!(splits_for(168 * 1024), SPLITS_MEASURED_BEST_DEEP_KV);
    }

    #[test]
    fn splits_stay_inside_the_scratch_provisioning() {
        for t in [0usize, 1, 2047, 4096, 1 << 20] {
            assert!(splits_for(t) <= MAX_SPLITS);
        }
    }
}

fn check_device(ctx: &WgpuContext) -> Result<()> {
    dispatch::require_workgroup_and_scratch(ctx, "flash_decode", WORKGROUP_SIZE, SCRATCH_BYTES)
}

fn check_grid(ctx: &WgpuContext, n_heads: usize, splits: usize) -> Result<()> {
    let limit = ctx.caps.max_compute_workgroups_per_dimension as usize;
    if n_heads > limit || splits > limit {
        return Err(WgpuError::Unsupported(format!(
            "flash_decode grid {n_heads}x{splits} exceeds max_compute_workgroups_per_dimension {limit}"
        )));
    }
    Ok(())
}

fn check_geometry(n_heads: usize, n_kv_heads: usize, head_dim: usize) -> Result<()> {
    if head_dim > MAX_HEAD_DIM {
        return Err(WgpuError::Unsupported(format!(
            "flash_decode head_dim {head_dim} exceeds {MAX_HEAD_DIM}"
        )));
    }
    if !n_heads.is_multiple_of(n_kv_heads) {
        return Err(WgpuError::Shape(format!(
            "flash_decode n_heads {n_heads} is not a multiple of n_kv_heads {n_kv_heads}"
        )));
    }
    Ok(())
}

fn window_start(total: usize, window: usize) -> usize {
    if window > 0 && total > window {
        total - window
    } else {
        0
    }
}

fn total_from(n_total: &[i32]) -> Result<usize> {
    let Some(first) = n_total.first() else {
        return Err(WgpuError::Shape(
            "flash_decode n_total is empty".to_string(),
        ));
    };
    Ok((*first).max(0) as usize)
}

fn slot_capacity(len: usize, per_slot: usize, what: &str) -> Result<usize> {
    if per_slot == 0 {
        return Err(WgpuError::Shape(format!("{what}: zero elements per slot")));
    }
    if !len.is_multiple_of(per_slot) {
        return Err(WgpuError::Shape(format!(
            "{what}: length {len} is not a multiple of {per_slot}"
        )));
    }
    Ok(len / per_slot)
}

pub(crate) fn words_to_f32(words: &[u32], out: &mut [f32]) {
    for (dst, src) in out.iter_mut().zip(words.iter()) {
        *dst = f32::from_bits(*src);
    }
}

pub(crate) fn words_to_u16(words: &[u32], out: &mut [u16]) {
    for (dst, src) in out.iter_mut().zip(words.iter()) {
        *dst = (*src & 0xffff) as u16;
    }
}

pub fn flash_splitk_scratch_elems(n_heads: usize, head_dim: usize, splits: usize) -> Result<usize> {
    if head_dim > MAX_HEAD_DIM {
        return Err(WgpuError::Unsupported(format!(
            "flash_decode head_dim {head_dim} exceeds {MAX_HEAD_DIM}"
        )));
    }
    Ok(n_heads * normalize_splits(splits) * (head_dim + 2))
}

fn decode_dev_common(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    n_total: &[i32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    out_bf16: bool,
) -> Result<Vec<u32>> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(Vec::new());
    }
    check_geometry(n_heads, n_kv_heads, head_dim)?;
    check_device(ctx)?;
    check_grid(ctx, n_heads, 1)?;
    let total = total_from(n_total)?;
    dispatch::check_len("flash_decode q", q.len(), n_heads * head_dim)?;
    let slots = slot_capacity(k_cache.len(), n_kv_heads * head_dim, "flash_decode k_cache")?;
    dispatch::check_len("flash_decode v_cache", v_cache.len(), k_cache.len())?;
    if total > slots {
        return Err(WgpuError::Shape(format!(
            "flash_decode k_cache holds {slots} slots but n_total is {total}"
        )));
    }

    let params = FdParams {
        n_heads: n_heads as u32,
        n_kv: n_kv_heads as u32,
        head_dim: head_dim as u32,
        total: total as u32,
        start: window_start(total, window) as u32,
        splits: 1,
        ring: 0,
        out_bf16: u32::from(out_bf16),
        scaling,
        pad0: 0,
        fused: 0,
        pad2: 0,
        m_rows: 1,
        window: 0,
        pad3: 0,
        pad4: 0,
    };

    let qb = dispatch::storage_from_slice(ctx, "flash-q", q);
    let kb = dispatch::storage_from_slice(ctx, "flash-k", k_cache);
    let vb = dispatch::storage_from_slice(ctx, "flash-v", v_cache);
    let ob = dispatch::storage_zeroed(ctx, "flash-out", (n_heads * head_dim * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "flash-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_flash_decode_dev_f32",
        &compose(WGSL),
        ENTRY_DECODE_F32,
        &[(0, &qb), (1, &kb), (2, &vb), (3, &ob), (4, &pb)],
        (n_heads as u32, 1, 1),
    )?;

    dispatch::read_back::<u32>(ctx, &ob, n_heads * head_dim)
}

pub fn flash_decode_dev_f32(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [f32],
    n_total: &[i32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    _splits: usize,
) -> Result<()> {
    dispatch::check_len("flash_decode out", out.len(), n_heads * head_dim)?;
    let words = decode_dev_common(
        ctx, q, k_cache, v_cache, n_total, n_heads, n_kv_heads, head_dim, window, scaling, false,
    )?;
    if words.is_empty() {
        return Ok(());
    }
    words_to_f32(&words, out);
    Ok(())
}

pub fn flash_decode_dev_f32_bf16out(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    out: &mut [u16],
    n_total: &[i32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    _splits: usize,
) -> Result<()> {
    dispatch::check_len("flash_decode out", out.len(), n_heads * head_dim)?;
    let words = decode_dev_common(
        ctx, q, k_cache, v_cache, n_total, n_heads, n_kv_heads, head_dim, window, scaling, true,
    )?;
    if words.is_empty() {
        return Ok(());
    }
    words_to_u16(&words, out);
    Ok(())
}

struct SplitkPlan {
    params: FdParams,
    scratch_elems: usize,
    n_heads: usize,
    head_dim: usize,
}

fn splitk_plan(
    ctx: &WgpuContext,
    q_len: usize,
    kv_len_elems: usize,
    out_len: usize,
    scratch_len: usize,
    n_total: &[i32],
    delta: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
    fused: bool,
) -> Result<SplitkPlan> {
    check_geometry(n_heads, n_kv_heads, head_dim)?;
    check_device(ctx)?;
    let splits = normalize_splits(splits);
    check_grid(ctx, n_heads, splits)?;
    if ring > 0 && window == 0 {
        return Err(WgpuError::Shape(
            "flash_decode ring>0 requires window>0".to_string(),
        ));
    }
    let total = total_from(n_total)?.saturating_sub(delta);
    dispatch::check_len("flash_decode q", q_len, n_heads * head_dim)?;
    dispatch::check_len("flash_decode out", out_len, n_heads * head_dim)?;
    let slots = slot_capacity(kv_len_elems, n_kv_heads * head_dim, "flash_decode k_cache")?;
    if ring > 0 {
        if ring > slots {
            return Err(WgpuError::Shape(format!(
                "flash_decode ring {ring} exceeds cache slots {slots}"
            )));
        }
    } else if total > slots {
        return Err(WgpuError::Shape(format!(
            "flash_decode k_cache holds {slots} slots but n_total is {total}"
        )));
    }
    let scratch_elems = n_heads * splits * (head_dim + 2);
    if scratch_len < scratch_elems {
        return Err(WgpuError::Shape(format!(
            "flash_decode scratch: got {scratch_len} want at least {scratch_elems}"
        )));
    }
    Ok(SplitkPlan {
        params: FdParams {
            n_heads: n_heads as u32,
            n_kv: n_kv_heads as u32,
            head_dim: head_dim as u32,
            total: total as u32,
            start: window_start(total, window) as u32,
            splits: splits as u32,
            ring: ring as u32,
            out_bf16: 1,
            scaling,
            pad0: 0,
            fused: u32::from(fused),
            pad2: 0,
            m_rows: 1,
            window: 0,
            pad3: 0,
            pad4: 0,
        },
        scratch_elems,
        n_heads,
        head_dim,
    })
}

fn run_splitk_bf16(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    out: &mut [u16],
    scratch: &mut [f32],
    n_total: &[i32],
    delta: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
    fused: bool,
) -> Result<()> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len("flash_decode v_cache", v_cache.len(), k_cache.len())?;
    let plan = splitk_plan(
        ctx,
        q.len(),
        k_cache.len(),
        out.len(),
        scratch.len(),
        n_total,
        delta,
        n_heads,
        n_kv_heads,
        head_dim,
        window,
        scaling,
        splits,
        ring,
        fused,
    )?;

    let qb = dispatch::storage_from_slice(ctx, "flash-q", q);
    let kb = dispatch::storage_from_slice(ctx, "flash-k-bf16", &pack_u16(k_cache));
    let vb = dispatch::storage_from_slice(ctx, "flash-v-bf16", &pack_u16(v_cache));
    let sb = dispatch::storage_zeroed(ctx, "flash-scratch", (plan.scratch_elems * 4) as u64);
    let ob = dispatch::storage_zeroed(ctx, "flash-out", (plan.n_heads * plan.head_dim * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "flash-params", &plan.params);
    let source = compose(WGSL);

    dispatch::run(
        ctx,
        "nv_kernels_flash_splitk_stage1_bf16kv",
        &source,
        ENTRY_STAGE1_BF16,
        &[(0, &qb), (4, &pb), (5, &kb), (6, &vb), (7, &sb)],
        (plan.n_heads as u32, plan.params.splits, 1),
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_flash_splitk_stage2",
        &source,
        ENTRY_STAGE2_U,
        &[(3, &ob), (4, &pb), (7, &sb)],
        (plan.n_heads as u32, 1, 1),
    )?;

    let got_scratch: Vec<f32> = dispatch::read_back(ctx, &sb, plan.scratch_elems)?;
    scratch[..plan.scratch_elems].copy_from_slice(&got_scratch);
    let words: Vec<u32> = dispatch::read_back(ctx, &ob, plan.n_heads * plan.head_dim)?;
    words_to_u16(&words, out);
    Ok(())
}

pub fn flash_decode_splitk_bf16kv(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    out: &mut [u16],
    scratch: &mut [f32],
    n_total: &[i32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
) -> Result<()> {
    run_splitk_bf16(
        ctx, q, k_cache, v_cache, out, scratch, n_total, 0, n_heads, n_kv_heads, head_dim, window,
        scaling, splits, ring, false,
    )
}

pub fn flash_decode_fused_bf16kv(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    out: &mut [u16],
    scratch: &mut [f32],
    n_total: &[i32],
    delta: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
) -> Result<()> {
    run_splitk_bf16(
        ctx, q, k_cache, v_cache, out, scratch, n_total, delta, n_heads, n_kv_heads, head_dim,
        window, scaling, splits, ring, true,
    )
}

pub fn flash_decode_fused_fp8kv(
    ctx: &WgpuContext,
    q: &[u16],
    k_fp8: &[u8],
    v_fp8: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    out: &mut [u16],
    scratch: &mut [f32],
    n_total: &[i32],
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
) -> Result<()> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len("flash_decode v_fp8", v_fp8.len(), k_fp8.len())?;
    let plan = splitk_plan(
        ctx,
        q.len(),
        k_fp8.len(),
        out.len(),
        scratch.len(),
        n_total,
        0,
        n_heads,
        n_kv_heads,
        head_dim,
        window,
        scaling,
        splits,
        ring,
        false,
    )?;
    let slots = k_fp8.len() / (n_kv_heads * head_dim);
    dispatch::check_len("flash_decode k_scales", k_scales.len(), slots * n_kv_heads)?;
    dispatch::check_len("flash_decode v_scales", v_scales.len(), slots * n_kv_heads)?;

    let q_f32: Vec<f32> = q
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();

    let qb = dispatch::storage_from_slice(ctx, "flash-q", &q_f32);
    let kb = dispatch::storage_from_slice(ctx, "flash-k-fp8", &pack_u8(k_fp8));
    let vb = dispatch::storage_from_slice(ctx, "flash-v-fp8", &pack_u8(v_fp8));
    let ksb = dispatch::storage_from_slice(ctx, "flash-k-scales", k_scales);
    let vsb = dispatch::storage_from_slice(ctx, "flash-v-scales", v_scales);
    let sb = dispatch::storage_zeroed(ctx, "flash-scratch", (plan.scratch_elems * 4) as u64);
    let ob = dispatch::storage_zeroed(ctx, "flash-out", (plan.n_heads * plan.head_dim * 4) as u64);
    let pb = dispatch::uniform_from(ctx, "flash-params", &plan.params);
    let source = compose(WGSL);

    dispatch::run(
        ctx,
        "nv_kernels_flash_splitk_stage1_fp8kv",
        &source,
        ENTRY_STAGE1_FP8,
        &[
            (0, &qb),
            (4, &pb),
            (5, &kb),
            (6, &vb),
            (7, &sb),
            (8, &ksb),
            (9, &vsb),
        ],
        (plan.n_heads as u32, plan.params.splits, 1),
    )?;
    dispatch::run(
        ctx,
        "nv_kernels_flash_splitk_stage2",
        &source,
        ENTRY_STAGE2_U,
        &[(3, &ob), (4, &pb), (7, &sb)],
        (plan.n_heads as u32, 1, 1),
    )?;

    let got_scratch: Vec<f32> = dispatch::read_back(ctx, &sb, plan.scratch_elems)?;
    scratch[..plan.scratch_elems].copy_from_slice(&got_scratch);
    let words: Vec<u32> = dispatch::read_back(ctx, &ob, plan.n_heads * plan.head_dim)?;
    words_to_u16(&words, out);
    Ok(())
}

pub(crate) const SCRATCH_BYTES_MK: u32 =
    2048 * 4 + WORKGROUP_SIZE * 4 + (WARPS as u32) * 8 + (WARPS as u32) * (MAX_HEAD_DIM as u32) * 4;

pub fn flash_splitk_scratch_elems_mk(
    n_heads: usize,
    head_dim: usize,
    m: usize,
    splits: usize,
) -> Result<usize> {
    if head_dim > MAX_HEAD_DIM_MK {
        return Err(WgpuError::Unsupported(format!(
            "flash_decode_mk head_dim {head_dim} exceeds {MAX_HEAD_DIM_MK}"
        )));
    }
    if !(1..=MAX_MK_ROWS).contains(&m) {
        return Err(WgpuError::Shape(format!(
            "flash_decode_mk m {m} out of 1..={MAX_MK_ROWS}"
        )));
    }
    Ok(n_heads * m * normalize_splits(splits) * (head_dim + 2))
}

struct MkPlan {
    params: FdParams,
    scratch_elems: usize,
    n_heads: usize,
    head_dim: usize,
    m: usize,
}

fn mk_plan(
    ctx: &WgpuContext,
    q_len: usize,
    kv_len_elems: usize,
    out_len: usize,
    scratch_len: usize,
    n_total: &[i32],
    delta: i32,
    m: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
) -> Result<MkPlan> {
    check_geometry(n_heads, n_kv_heads, head_dim)?;
    if head_dim > MAX_HEAD_DIM_MK {
        return Err(WgpuError::Unsupported(format!(
            "flash_decode_mk head_dim {head_dim} exceeds {MAX_HEAD_DIM_MK}"
        )));
    }
    if !(1..=MAX_MK_ROWS).contains(&m) {
        return Err(WgpuError::Shape(format!(
            "flash_decode_mk m {m} out of 1..={MAX_MK_ROWS}"
        )));
    }
    check_device(ctx)?;
    if !ctx.caps.workgroup_storage_fits(SCRATCH_BYTES_MK) {
        return Err(WgpuError::Unsupported(format!(
            "flash_decode_mk scratch needs {SCRATCH_BYTES_MK} bytes of workgroup storage; device allows {}",
            ctx.caps.max_compute_workgroup_storage_size
        )));
    }
    let splits = normalize_splits(splits);
    check_grid(ctx, n_heads, splits.max(m))?;
    if ring > 0 && window == 0 {
        return Err(WgpuError::Shape(
            "flash_decode_mk ring>0 requires window>0".to_string(),
        ));
    }
    let Some(pos0) = n_total.first() else {
        return Err(WgpuError::Shape(
            "flash_decode_mk n_total is empty".to_string(),
        ));
    };
    let total = (i64::from(*pos0) - i64::from(delta)).max(0) as usize;
    dispatch::check_len("flash_decode_mk q", q_len, m * n_heads * head_dim)?;
    dispatch::check_len("flash_decode_mk out", out_len, m * n_heads * head_dim)?;
    let slots = slot_capacity(
        kv_len_elems,
        n_kv_heads * head_dim,
        "flash_decode_mk k_cache",
    )?;
    if ring > 0 {
        if ring > slots {
            return Err(WgpuError::Shape(format!(
                "flash_decode_mk ring {ring} exceeds cache slots {slots}"
            )));
        }
    } else if total > slots {
        return Err(WgpuError::Shape(format!(
            "flash_decode_mk k_cache holds {slots} slots but n_total is {total}"
        )));
    }
    let scratch_elems = n_heads * m * splits * (head_dim + 2);
    if scratch_len < scratch_elems {
        return Err(WgpuError::Shape(format!(
            "flash_decode_mk scratch: got {scratch_len} want at least {scratch_elems}"
        )));
    }
    Ok(MkPlan {
        params: FdParams {
            n_heads: n_heads as u32,
            n_kv: n_kv_heads as u32,
            head_dim: head_dim as u32,
            total: total as u32,
            start: 0,
            splits: splits as u32,
            ring: ring as u32,
            out_bf16: 1,
            scaling,
            pad0: 0,
            fused: 1,
            pad2: 0,
            m_rows: m as u32,
            window: window as u32,
            pad3: 0,
            pad4: 0,
        },
        scratch_elems,
        n_heads,
        head_dim,
        m,
    })
}

fn run_mk_stages(
    ctx: &WgpuContext,
    plan: &MkPlan,
    stage1_entry: &str,
    stage1_bindings: &[(u32, &wgpu::Buffer)],
    sb: &wgpu::Buffer,
    ob: &wgpu::Buffer,
    pb: &wgpu::Buffer,
    out: &mut [u16],
    scratch: &mut [f32],
) -> Result<()> {
    let source = compose(WGSL);
    dispatch::run(
        ctx,
        stage1_entry,
        &source,
        stage1_entry,
        stage1_bindings,
        (plan.n_heads as u32, plan.params.splits, 1),
    )?;
    dispatch::run(
        ctx,
        ENTRY_STAGE2_MK_U,
        &source,
        ENTRY_STAGE2_MK_U,
        &[(3, ob), (4, pb), (7, sb)],
        (plan.n_heads as u32, plan.m as u32, 1),
    )?;
    let got_scratch: Vec<f32> = dispatch::read_back(ctx, sb, plan.scratch_elems)?;
    scratch[..plan.scratch_elems].copy_from_slice(&got_scratch);
    let words: Vec<u32> = dispatch::read_back(ctx, ob, plan.m * plan.n_heads * plan.head_dim)?;
    words_to_u16(&words, out);
    Ok(())
}

pub fn flash_decode_fused_bf16kv_mk(
    ctx: &WgpuContext,
    q: &[f32],
    k_cache: &[u16],
    v_cache: &[u16],
    out: &mut [u16],
    scratch: &mut [f32],
    n_total: &[i32],
    delta: i32,
    m: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    splits: usize,
) -> Result<()> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len("flash_decode_mk v_cache", v_cache.len(), k_cache.len())?;
    let plan = mk_plan(
        ctx,
        q.len(),
        k_cache.len(),
        out.len(),
        scratch.len(),
        n_total,
        delta,
        m,
        n_heads,
        n_kv_heads,
        head_dim,
        window,
        1.0,
        splits,
        0,
    )?;

    let qb = dispatch::storage_from_slice(ctx, "flash-mk-q", q);
    let kb = dispatch::storage_from_slice(ctx, "flash-mk-k-bf16", &pack_u16(k_cache));
    let vb = dispatch::storage_from_slice(ctx, "flash-mk-v-bf16", &pack_u16(v_cache));
    let sb = dispatch::storage_zeroed(ctx, "flash-mk-scratch", (plan.scratch_elems * 4) as u64);
    let ob = dispatch::storage_zeroed(
        ctx,
        "flash-mk-out",
        (plan.m * plan.n_heads * plan.head_dim * 4) as u64,
    );
    let pb = dispatch::uniform_from(ctx, "flash-mk-params", &plan.params);
    run_mk_stages(
        ctx,
        &plan,
        ENTRY_STAGE1_BF16_MK_U,
        &[(0, &qb), (4, &pb), (5, &kb), (6, &vb), (7, &sb)],
        &sb,
        &ob,
        &pb,
        out,
        scratch,
    )
}

pub fn flash_decode_fused_fp8kv_mk(
    ctx: &WgpuContext,
    q: &[u16],
    k_fp8: &[u8],
    v_fp8: &[u8],
    k_scales: &[f32],
    v_scales: &[f32],
    out: &mut [u16],
    scratch: &mut [f32],
    n_total: &[i32],
    delta: i32,
    m: usize,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    window: usize,
    scaling: f32,
    splits: usize,
    ring: usize,
) -> Result<()> {
    if n_heads == 0 || n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len("flash_decode_mk v_fp8", v_fp8.len(), k_fp8.len())?;
    let plan = mk_plan(
        ctx,
        q.len(),
        k_fp8.len(),
        out.len(),
        scratch.len(),
        n_total,
        delta,
        m,
        n_heads,
        n_kv_heads,
        head_dim,
        window,
        scaling,
        splits,
        ring,
    )?;
    let slots = k_fp8.len() / (n_kv_heads * head_dim);
    dispatch::check_len(
        "flash_decode_mk k_scales",
        k_scales.len(),
        slots * n_kv_heads,
    )?;
    dispatch::check_len(
        "flash_decode_mk v_scales",
        v_scales.len(),
        slots * n_kv_heads,
    )?;

    let q_f32: Vec<f32> = q
        .iter()
        .map(|b| f32::from_bits((*b as u32) << 16))
        .collect();

    let qb = dispatch::storage_from_slice(ctx, "flash-mk-q", &q_f32);
    let kb = dispatch::storage_from_slice(ctx, "flash-mk-k-fp8", &pack_u8(k_fp8));
    let vb = dispatch::storage_from_slice(ctx, "flash-mk-v-fp8", &pack_u8(v_fp8));
    let ksb = dispatch::storage_from_slice(ctx, "flash-mk-k-scales", k_scales);
    let vsb = dispatch::storage_from_slice(ctx, "flash-mk-v-scales", v_scales);
    let sb = dispatch::storage_zeroed(ctx, "flash-mk-scratch", (plan.scratch_elems * 4) as u64);
    let ob = dispatch::storage_zeroed(
        ctx,
        "flash-mk-out",
        (plan.m * plan.n_heads * plan.head_dim * 4) as u64,
    );
    let pb = dispatch::uniform_from(ctx, "flash-mk-params", &plan.params);
    run_mk_stages(
        ctx,
        &plan,
        ENTRY_STAGE1_FP8_MK_U,
        &[
            (0, &qb),
            (4, &pb),
            (5, &kb),
            (6, &vb),
            (7, &sb),
            (8, &ksb),
            (9, &vsb),
        ],
        &sb,
        &ob,
        &pb,
        out,
        scratch,
    )
}

fn write_kv_params(
    ctx: &WgpuContext,
    cache_len: usize,
    pos: &[i32],
    n_kv_heads: usize,
    head_dim: usize,
    ring: usize,
) -> Result<FdParams> {
    let slots = slot_capacity(cache_len, n_kv_heads * head_dim, "write_kv_bf16 cache")?;
    let pos0 = total_from(pos)?;
    if ring > 0 {
        if ring > slots {
            return Err(WgpuError::Shape(format!(
                "write_kv_bf16 ring {ring} exceeds cache slots {slots}"
            )));
        }
    } else if pos0 > slots {
        return Err(WgpuError::Shape(format!(
            "write_kv_bf16 cache holds {slots} slots but pos is {pos0}"
        )));
    }
    check_grid(ctx, n_kv_heads, 1)?;
    Ok(FdParams {
        n_heads: n_kv_heads as u32,
        n_kv: n_kv_heads as u32,
        head_dim: head_dim as u32,
        total: pos0 as u32,
        start: 0,
        splits: 1,
        ring: ring as u32,
        out_bf16: 1,
        scaling: 1.0,
        pad0: 0,
        fused: 0,
        pad2: 0,
        m_rows: 1,
        window: 0,
        pad3: 0,
        pad4: 0,
    })
}

pub fn write_kv_bf16(
    ctx: &WgpuContext,
    k_src: &[u16],
    v_src: &[u16],
    k_cache: &mut [u16],
    v_cache: &mut [u16],
    pos: &[i32],
    n_kv_heads: usize,
    head_dim: usize,
    ring: usize,
) -> Result<()> {
    if n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len("write_kv_bf16 k_src", k_src.len(), n_kv_heads * head_dim)?;
    dispatch::check_len("write_kv_bf16 v_src", v_src.len(), n_kv_heads * head_dim)?;
    dispatch::check_len("write_kv_bf16 v_cache", v_cache.len(), k_cache.len())?;
    let params = write_kv_params(ctx, k_cache.len(), pos, n_kv_heads, head_dim, ring)?;
    if params.total == 0 {
        return Ok(());
    }

    let kw = dispatch::storage_from_slice(ctx, "wkv-src-k", &pack_u16(k_src));
    let vw = dispatch::storage_from_slice(ctx, "wkv-src-v", &pack_u16(v_src));
    let ck = dispatch::storage_from_slice(ctx, "wkv-cache-k", &pack_u16(k_cache));
    let cv = dispatch::storage_from_slice(ctx, "wkv-cache-v", &pack_u16(v_cache));
    let pb = dispatch::uniform_from(ctx, "wkv-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_write_kv_bf16",
        &compose(WGSL),
        ENTRY_WRITE_KV_BF16,
        &[(4, &pb), (5, &kw), (6, &vw), (12, &ck), (13, &cv)],
        (1, 1, 1),
    )?;

    let words = k_cache.len().div_ceil(2);
    let got_k: Vec<u32> = dispatch::read_back(ctx, &ck, words)?;
    let got_v: Vec<u32> = dispatch::read_back(ctx, &cv, words)?;
    unpack_u16(&got_k, k_cache);
    unpack_u16(&got_v, v_cache);
    Ok(())
}

pub fn write_kv_bf16_from_f32(
    ctx: &WgpuContext,
    k_src: &[f32],
    v_src: &[f32],
    k_cache: &mut [u16],
    v_cache: &mut [u16],
    pos: &[i32],
    n_kv_heads: usize,
    head_dim: usize,
    ring: usize,
) -> Result<()> {
    if n_kv_heads == 0 || head_dim == 0 {
        return Ok(());
    }
    dispatch::check_len("write_kv_bf16 k_src", k_src.len(), n_kv_heads * head_dim)?;
    dispatch::check_len("write_kv_bf16 v_src", v_src.len(), n_kv_heads * head_dim)?;
    dispatch::check_len("write_kv_bf16 v_cache", v_cache.len(), k_cache.len())?;
    let params = write_kv_params(ctx, k_cache.len(), pos, n_kv_heads, head_dim, ring)?;
    if params.total == 0 {
        return Ok(());
    }

    let kw = dispatch::storage_from_slice(ctx, "wkv-src-k", k_src);
    let vw = dispatch::storage_from_slice(ctx, "wkv-src-v", v_src);
    let ck = dispatch::storage_from_slice(ctx, "wkv-cache-k", &pack_u16(k_cache));
    let cv = dispatch::storage_from_slice(ctx, "wkv-cache-v", &pack_u16(v_cache));
    let pb = dispatch::uniform_from(ctx, "wkv-params", &params);

    dispatch::run(
        ctx,
        "nv_kernels_write_kv_bf16_f32",
        &compose(WGSL),
        ENTRY_WRITE_KV_F32,
        &[(4, &pb), (10, &kw), (11, &vw), (12, &ck), (13, &cv)],
        (1, 1, 1),
    )?;

    let words = k_cache.len().div_ceil(2);
    let got_k: Vec<u32> = dispatch::read_back(ctx, &ck, words)?;
    let got_v: Vec<u32> = dispatch::read_back(ctx, &cv, words)?;
    unpack_u16(&got_k, k_cache);
    unpack_u16(&got_v, v_cache);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_are_normalized_like_the_cuda_env_hook() {
        assert_eq!(normalize_splits(0), 16);
        assert_eq!(normalize_splits(8), 8);
        assert_eq!(normalize_splits(16), 16);
        assert_eq!(normalize_splits(32), 32);
        assert_eq!(normalize_splits(7), 16);
    }

    #[test]
    fn scratch_elems_match_the_cuda_formula() {
        assert_eq!(
            flash_splitk_scratch_elems(4, 128, 16).unwrap(),
            4 * 16 * 130
        );
        assert_eq!(flash_splitk_scratch_elems(3, 64, 8).unwrap(), 3 * 8 * 66);
        assert_eq!(flash_splitk_scratch_elems(1, 256, 0).unwrap(), 16 * 258);
    }

    #[test]
    fn window_start_clamps_like_cuda() {
        assert_eq!(window_start(100, 0), 0);
        assert_eq!(window_start(100, 200), 0);
        assert_eq!(window_start(100, 40), 60);
    }

    #[test]
    fn params_are_uniform_buffer_sized() {
        assert_eq!(std::mem::size_of::<FdParams>() % 16, 0);
    }

    #[test]
    fn pack_round_trips_u16() {
        let src: Vec<u16> = (0..9u16).collect();
        let words = pack_u16(&src);
        let mut back = vec![0u16; src.len()];
        unpack_u16(&words, &mut back);
        assert_eq!(back, src);
    }

    #[test]
    fn mk_scratch_elems_match_the_cuda_formula() {
        assert_eq!(
            flash_splitk_scratch_elems_mk(8, 256, 4, 16).unwrap(),
            8 * 4 * 16 * 258
        );
        assert_eq!(
            flash_splitk_scratch_elems_mk(4, 128, 1, 0).unwrap(),
            4 * 16 * 130
        );
        assert!(matches!(
            flash_splitk_scratch_elems_mk(4, 512, 2, 16).unwrap_err(),
            WgpuError::Unsupported(_)
        ));
        assert!(matches!(
            flash_splitk_scratch_elems_mk(4, 128, 0, 16).unwrap_err(),
            WgpuError::Shape(_)
        ));
        assert!(matches!(
            flash_splitk_scratch_elems_mk(4, 128, 9, 16).unwrap_err(),
            WgpuError::Shape(_)
        ));
    }

    #[test]
    fn geometry_rejects_ragged_head_groups() {
        assert!(matches!(
            check_geometry(6, 4, 64).unwrap_err(),
            WgpuError::Shape(_)
        ));
        assert!(matches!(
            check_geometry(8, 4, 1024).unwrap_err(),
            WgpuError::Unsupported(_)
        ));
    }
}
