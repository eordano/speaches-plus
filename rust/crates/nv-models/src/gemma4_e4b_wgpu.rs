use anyhow::{Context, Result};
use std::sync::Arc;

use nv_kernels::wgpu_backend::dispatch::{self, GpuTensor, GpuUniform};
use nv_kernels::wgpu_backend::kernels as wk;
use nv_kernels::wgpu_backend::na;
use nv_kernels::wgpu_backend::na_attn;
use nv_kernels::wgpu_backend::{compose, WgpuContext};

use crate::gemma4::{Gemma4Config, LayerType};
pub use crate::gemma4_wgpu_shared::pack_pairs;
use crate::gemma4_wgpu_shared::{
    bf16_bits, err, rope_tables, FLASH2_PK_ENTRY, FLASH2_PK_WGSL, GEMV_PK3_ENTRY, GEMV_PK_ENTRY,
    GEMV_PK_WGSL, ROPE_F32_ENTRY, ROPE_F32_WGSL,
};

pub const FLASH_SPLITS: u32 = 16;

pub const MAX_TABLE_CHUNKS: usize = 8;

const CHUNK_BYTE_CAP: u64 = 1 << 30;

const GATHER_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_gather.wgsl");

const AXPBY_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_axpby.wgsl");

const GATEMUL_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_gatemul.wgsl");

const GEMV_PK_MK_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_gemv_pk_mk.wgsl");

fn gemv_pk_wgsl() -> String {
    [GEMV_PK_WGSL, GEMV_PK_MK_WGSL].concat()
}

const GEMV_W4_PK_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_gemv_w4_pk.wgsl");

const FLASH2_PK_MK_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_flash2_pk_mk.wgsl");

fn flash2_pk_wgsl() -> String {
    [FLASH2_PK_WGSL, FLASH2_PK_MK_WGSL].concat()
}

pub fn flash1_e4b_source(hd_max: u32, sg: bool) -> String {
    assert!(
        hd_max > 0 && hd_max % 32 == 0 && hd_max <= 512,
        "flash1 hd_max {hd_max} must be a positive multiple of 32 up to 512"
    );
    let accs = hd_max / 32;
    let t = flash1_e4b_tag(hd_max, sg);
    let entry = flash1_e4b_entry(hd_max, sg);
    let mut b = String::with_capacity(8192);
    use std::fmt::Write;

    writeln!(b, "var<workgroup> f1{t}_qsh: array<f32, {hd_max}>;").unwrap();
    writeln!(b, "var<workgroup> f1{t}_sacc: array<f32, {}>;", hd_max * 8).unwrap();
    writeln!(b, "var<workgroup> f1{t}_sm: array<f32, 8>;").unwrap();
    writeln!(b, "var<workgroup> f1{t}_sl: array<f32, 8>;").unwrap();
    if sg {
        writeln!(
            b,
            "\nfn f1{t}_reduce(lid: u32, x: f32) -> f32 {{\n    var a = x;\n    a = a + \
             subgroupShuffleXor(a, 16u);\n    a = a + subgroupShuffleXor(a, 8u);\n    a = a + \
             subgroupShuffleXor(a, 4u);\n    a = a + subgroupShuffleXor(a, 2u);\n    a = a + \
             subgroupShuffleXor(a, 1u);\n    return a;\n}}"
        )
        .unwrap();
    } else {
        writeln!(b, "var<workgroup> f1{t}_red: array<f32, 256>;").unwrap();
        writeln!(
            b,
            "\nfn f1{t}_reduce(lid: u32, x: f32) -> f32 {{\n    f1{t}_red[lid] = x;\n    \
             workgroupBarrier();\n    for (var o = 16u; o > 0u; o = o >> 1u) {{\n        let \
             other = f1{t}_red[lid ^ o];\n        workgroupBarrier();\n        f1{t}_red[lid] = \
             f1{t}_red[lid] + other;\n        workgroupBarrier();\n    }}\n    return \
             f1{t}_red[lid];\n}}"
        )
        .unwrap();
    }

    writeln!(
        b,
        "\nfn f1{t}_epilogue(lid: u32, lane: u32, warp: u32, hd: u32, slot: u32, m: f32, l: f32) \
         {{\n    if (lane == 0u) {{\n        f1{t}_sm[warp] = m;\n        f1{t}_sl[warp] = l;\n    \
         }}\n    workgroupBarrier();\n    if (warp == 0u) {{\n        var m_blk = \
         fd_neg_inf();\n        for (var w = 0u; w < FD_WARPS; w = w + 1u) {{\n            m_blk = \
         max(m_blk, f1{t}_sm[w]);\n        }}\n        var l_blk = 0.0;\n        for (var w = 0u; \
         w < FD_WARPS; w = w + 1u) {{\n            if (f1{t}_sm[w] > fd_neg_inf()) {{\n           \
         l_blk = l_blk + fd_round(f1{t}_sl[w] * fd_exp(f1{t}_sm[w] - m_blk));\n            }}\n    \
         }}\n        if (lane == 0u) {{\n            fd_scratch[slot] = m_blk;\n            \
         fd_scratch[slot + 1u] = l_blk;\n        }}\n    }}\n    var m_blk = fd_neg_inf();\n    \
         for (var w = 0u; w < FD_WARPS; w = w + 1u) {{\n        m_blk = max(m_blk, \
         f1{t}_sm[w]);\n    }}\n    for (var d = lid; d < hd; d = d + FD_BLOCK) {{\n        var a \
         = 0.0;\n        for (var w = 0u; w < FD_WARPS; w = w + 1u) {{\n            if \
         (f1{t}_sm[w] > fd_neg_inf()) {{\n                a = a + fd_round(f1{t}_sacc[w * \
         {hd_max}u + d] * fd_exp(f1{t}_sm[w] - m_blk));\n            }}\n        }}\n        \
         fd_scratch[slot + 2u + d] = a;\n    }}\n}}"
    )
    .unwrap();

    writeln!(b, "\n@compute @workgroup_size(256)").unwrap();
    writeln!(
        b,
        "fn {entry}(\n    @builtin(workgroup_id) wg: vec3<u32>,\n    \
         @builtin(local_invocation_id) tid: vec3<u32>\n) {{"
    )
    .unwrap();
    writeln!(
        b,
        "    let h = wg.x;\n    let split = wg.y;\n    if (h >= fd_params.n_heads) {{\n        \
         return;\n    }}\n    let hd = fd_params.head_dim;\n    let nkv = fd_params.n_kv;\n    let \
         group = fd_params.n_heads / nkv;\n    let kvh = h / group;\n    let lid = tid.x;\n    let \
         lane = lid & 31u;\n    let warp = lid >> 5u;\n\n    for (var d = lid; d < hd; d = d + \
         FD_BLOCK) {{\n        f1{t}_qsh[d] = fd_q[h * hd + d];\n    }}\n    workgroupBarrier();\n"
    )
    .unwrap();
    for i in 0..accs {
        writeln!(b, "    var acc{i} = 0.0;").unwrap();
    }
    writeln!(
        b,
        "    var m = fd_neg_inf();\n    var l = 0.0;\n\n    let total = fd_params.total;\n    let \
         base = fd_params.start + split * FD_WARPS;\n    let stride = fd_params.splits * \
         FD_WARPS;\n    var rounds = 0u;\n    if (total > base) {{\n        rounds = (total - base \
         + stride - 1u) / stride;\n    }}\n    let use_vec4 = (hd & 3u) == 0u;\n\n    for (var r = \
         0u; r < rounds; r = r + 1u) {{\n        let p = base + warp + r * stride;\n        let \
         live = p < total;\n        var sp = p;\n        if (fd_params.ring > 0u) {{\n            \
         sp = p % fd_params.ring;\n        }}\n        var partial = 0.0;\n        var ks = 0.0;\n \
         if (live) {{\n            let kbase = (sp * nkv + kvh) * hd;\n            ks = \
         fd_k_scales[sp * nkv + kvh];\n            if (use_vec4) {{\n                let n4 = hd >> \
         2u;\n                for (var j = lane; j < n4; j = j + FD_LANES) {{\n                    \
         let qb = j * 4u;\n                    let kb = kbase + qb;\n                    let f0 = \
         fd_k_fp8(kb);\n                    let f1 = fd_k_fp8(kb + 1u);\n                    let \
         f2 = fd_k_fp8(kb + 2u);\n                    let f3 = fd_k_fp8(kb + 3u);\n                \
         var t = f1{t}_qsh[qb + 1u] * f1;\n                    t = fma(f1{t}_qsh[qb], f0, t);\n    \
         t = fma(f1{t}_qsh[qb + 2u], f2, t);\n                    t = fma(f1{t}_qsh[qb + 3u], f3, \
         t);\n                    partial = partial + t;\n                }}\n            }} else \
         {{\n                for (var d = lane; d < hd; d = d + FD_LANES) {{\n                    \
         partial = fma(f1{t}_qsh[d], fd_k_fp8(kbase + d), partial);\n                }}\n          \
         }}\n        }}\n        let score = (f1{t}_reduce(lid, partial) * ks) * \
         fd_params.scaling;\n        if (live) {{\n            let m_new = max(m, score);\n        \
         let corr = fd_exp(m - m_new);\n            let w = fd_exp(score - m_new);\n            l = \
         fma(l, corr, w);\n            let vbase = (sp * nkv + kvh) * hd;\n            let w_v = w \
         * fd_v_scales[sp * nkv + kvh];"
    )
    .unwrap();
    for i in 0..accs {
        writeln!(
            b,
            "            {{\n                let d = lane + {i}u * FD_LANES;\n                if \
             (d < hd) {{\n                    acc{i} = fma(w_v, fd_v_fp8(vbase + d), acc{i} * \
             corr);\n                }}\n            }}"
        )
        .unwrap();
    }
    writeln!(b, "            m = m_new;\n        }}\n    }}\n").unwrap();
    for i in 0..accs {
        writeln!(
            b,
            "    {{\n        let d = lane + {i}u * FD_LANES;\n        if (d < hd) {{\n            \
             f1{t}_sacc[warp * {hd_max}u + d] = acc{i};\n        }}\n    }}"
        )
        .unwrap();
    }
    writeln!(
        b,
        "    let slot = (h * fd_params.splits + split) * (hd + 2u);\n    f1{t}_epilogue(lid, lane, \
         warp, hd, slot, m, l);\n}}"
    )
    .unwrap();
    b
}

fn flash1_pick(
    built: &[(u32, Arc<wgpu::ComputePipeline>)],
    hd: usize,
) -> Arc<wgpu::ComputePipeline> {
    built
        .iter()
        .find(|(cap, _)| *cap as usize >= hd)
        .map(|(_, p)| p.clone())
        .unwrap_or_else(|| {
            panic!(
                "no flash stage1 pipeline covers head_dim {hd}; built {:?}",
                built.iter().map(|(c, _)| *c).collect::<Vec<_>>()
            )
        })
}

fn flash1_for(pl: &Pipelines, hd: usize) -> Arc<wgpu::ComputePipeline> {
    flash1_pick(&pl.flash1, hd)
}

fn flash1_e4b_tag(hd_max: u32, sg: bool) -> String {
    format!("{}{}", hd_max, if sg { "sg" } else { "wb" })
}

pub fn flash1_e4b_entry(hd_max: u32, sg: bool) -> String {
    format!("e4b_flash1_{}_fp8", flash1_e4b_tag(hd_max, sg))
}

pub const FLASH_SD_ENV: &str = "NV_E4B_FLASH_SD";

pub const FLASH_SD_DEFAULT_ON_SHIFT_DECODE_WINS_AT_DEPTH_AND_HOLDS_PPL_PER_PERF_RUNS_JSONL: bool =
    true;

pub fn flash_sd_enabled() -> bool {
    if std::env::var(FLASH_SD_ENV).ok().as_deref() == Some("0") {
        return false;
    }
    FLASH_SD_DEFAULT_ON_SHIFT_DECODE_WINS_AT_DEPTH_AND_HOLDS_PPL_PER_PERF_RUNS_JSONL
}

const E4B_SD_K_SCALE_ANCHOR: &str = "ks = fd_k_scales[sp * nkv + kvh];";

const E4B_SD_W_V_SCALE_ANCHOR: &str = "let w_v = w * fd_v_scales[sp * nkv + kvh];";

const E4B_SD_VSC_SCALE_ANCHOR: &str = "let vsc = fd_v_scales[sp * nkv + kvh];";

fn flash_sd_rewrite(base: &str, entry: &str, k_anchor: &str, v_anchor: &str) -> String {
    for (scales, anchor) in [("fd_k_scales", k_anchor), ("fd_v_scales", v_anchor)] {
        assert_eq!(
            base.matches(anchor).count(),
            1,
            "e4b shift-decode rewrite for {entry}: scale anchor `{anchor}` must appear exactly \
             once; a missed anchor silently applies exact-decode magnitudes against \
             2pow120-folded scales"
        );
        assert_eq!(
            base.matches(scales).count(),
            1,
            "e4b shift-decode rewrite for {entry}: a {scales} read outside the anchored line \
             would escape the 2pow120 fold and scale shift-decoded values 2pow120 too small"
        );
    }
    assert!(
        base.contains("fd_k_fp8(") && base.contains("fd_v_fp8("),
        "e4b shift-decode rewrite for {entry}: the exact fp8 decoder calls are gone, so the twin \
         would change nothing while claiming the shift-decode speedup"
    );
    let folded = |anchor: &str| {
        format!(
            "{} * bitcast<f32>(0x7B800000u);",
            anchor
                .strip_suffix(';')
                .expect("scale anchors end the statement")
        )
    };
    let sd = base
        .replacen(&format!("fn {entry}("), &format!("fn {entry}_sd("), 1)
        .replace("fd_k_fp8(", "fd_k_fp8_sd(")
        .replace("fd_v_fp8(", "fd_v_fp8_sd(")
        .replacen(k_anchor, &folded(k_anchor), 1)
        .replacen(v_anchor, &folded(v_anchor), 1);
    assert!(
        sd.contains(&format!("fn {entry}_sd("))
            && sd.contains("fd_k_fp8_sd(")
            && sd.contains("fd_v_fp8_sd(")
            && sd.matches("0x7B800000").count() == 2,
        "e4b shift-decode rewrite for {entry} missed an anchor; the generated kernel would \
         silently apply exact-decode magnitudes against 2pow120-folded scales"
    );
    sd
}

pub fn flash1_e4b_entry_sd(hd_max: u32, sg: bool) -> String {
    format!("{}_sd", flash1_e4b_entry(hd_max, sg))
}

pub fn flash1_e4b_source_sd(hd_max: u32, sg: bool) -> String {
    flash_sd_rewrite(
        &flash1_e4b_source(hd_max, sg),
        &flash1_e4b_entry(hd_max, sg),
        E4B_SD_K_SCALE_ANCHOR,
        E4B_SD_W_V_SCALE_ANCHOR,
    )
}

pub const FLASH_ROWS_STAGE1_ENTRY: &str = "g4w_flash_rows_stage1_fp8";

pub const FLASH_ROWS_STAGE1_ENTRY_SD: &str = "g4w_flash_rows_stage1_fp8_sd";

pub fn flash_rows_stage1_source_sd() -> String {
    let decl = format!("fn {FLASH_ROWS_STAGE1_ENTRY}(");
    assert_eq!(
        FLASH2_PK_MK_WGSL.matches(decl.as_str()).count(),
        1,
        "mk stage1 shift-decode twin: `{decl}` must appear exactly once in e4b_flash2_pk_mk.wgsl; \
         a renamed or duplicated stock entry would make the twin silently rewrite the wrong \
         kernel body"
    );
    let fn_pos = FLASH2_PK_MK_WGSL.find(decl.as_str()).unwrap();
    let head = FLASH2_PK_MK_WGSL[..fn_pos].rfind("@compute").unwrap_or_else(|| {
        panic!(
            "mk stage1 shift-decode twin: no @compute attribute precedes \
             {FLASH_ROWS_STAGE1_ENTRY}; the extracted twin would not be an entry point and its \
             pipeline creation would fail at model load"
        )
    });
    let end = FLASH2_PK_MK_WGSL[fn_pos..].find("\n}").unwrap_or_else(|| {
        panic!(
            "mk stage1 shift-decode twin: {FLASH_ROWS_STAGE1_ENTRY} has no column-zero closing \
             brace; the extracted twin would be truncated mid-body"
        )
    });
    flash_sd_rewrite(
        &FLASH2_PK_MK_WGSL[head..fn_pos + end + 2],
        FLASH_ROWS_STAGE1_ENTRY,
        E4B_SD_K_SCALE_ANCHOR,
        E4B_SD_W_V_SCALE_ANCHOR,
    )
}

pub fn decode_splits() -> u32 {
    wk::flash_decode::splits_env() as u32
}

pub const DEEP_SPLIT_ARM_RULE: &str = "depth-adaptive split-k, fold2 real-weight \
     e4b_deep_kv_decode_rate ms/tok shallow16 vs deep64: depth 512 = 6.40 vs 6.70, 2048 = 6.66 \
     vs 6.75, 4096 = 7.07 vs 6.86, 8000 = 7.79 vs 7.05 -- the deep arm engages once total kv \
     depth exceeds NV_E4B_SPLIT_DEPTH (default 2048, the last depth where shallow still wins); \
     NV_E4B_SPLIT_DEPTH=0 restores the single-list shallow decoder";

pub const E4B_SPLIT_DEPTH_ENV: &str = "NV_E4B_SPLIT_DEPTH";

pub const E4B_DEEP_SPLITS_ENV: &str = "NV_E4B_DEEP_SPLITS";

pub const E4B_DEEP_FULL_ONLY_ENV: &str = "NV_E4B_DEEP_FULL_ONLY";

pub fn deep_full_only() -> bool {
    std::env::var(E4B_DEEP_FULL_ONLY_ENV).ok().as_deref() == Some("1")
}

pub const E4B_DEEP_FOLD_ENV: &str = "NV_E4B_DEEP_FOLD";

pub fn deep_fold_env(gqa_group: u32) -> u32 {
    let Some(n) = deep_env_u32(E4B_DEEP_FOLD_ENV) else {
        return 0;
    };
    if n <= 1 {
        return 0;
    }
    assert!(
        gqa_group > 0 && gqa_group % n == 0,
        "{E4B_DEEP_FOLD_ENV}={n} must divide the GQA group {gqa_group}: a workgroup may only fold \
         query heads that share one KV head"
    );
    n
}

pub const E4B_DEEP_STAGE2_LEGACY16_ENV: &str = "NV_E4B_DEEP_STAGE2_LEGACY16";

pub const DEEP_STAGE2_RULE: &str = "the shared g4w_flash_splitk_stage2_pk body is hand-unrolled \
     for exactly 16 partials, so folding `let splits` past 16 makes it merge only the first 16 \
     of N split results -- full-attention past NV_E4B_SPLIT_DEPTH then reads a quarter of the \
     context and sliding windows their first 128 tokens (argmax probes at pos 3050: legacy \
     agrees with the shallow reference 6/16, this kernel 15/16); e4b_flash2_deep_source is the \
     any-splits stage2 the deep arm dispatches instead, and NV_E4B_DEEP_STAGE2_LEGACY16=1 \
     restores the truncating fold for measurement. Cost of the fix at deep splits 64: within \
     run variance (12.70 vs 13.03 ms/tok at depth 122880, same-harness repeats). Numbers \
     published from the truncating path: the DEEP_SPLIT_ARM_RULE ladder cells and the \
     RECORD-BAR gemma4-E4B deep row (63.4 ms/tok at 122880)";

pub fn deep_stage2_legacy16() -> bool {
    std::env::var(E4B_DEEP_STAGE2_LEGACY16_ENV).ok().as_deref() == Some("1")
}

pub const FLASH2_DEEP_ENTRY: &str = "e4b_flash2_pk_deep_rt";

pub fn e4b_flash2_deep_source(splits: u32) -> String {
    assert!(
        (1..=wk::flash_decode::MAX_SPLITS as u32).contains(&splits),
        "deep stage2 splits {splits} outside the provisioned scratch; {DEEP_STAGE2_RULE}"
    );
    format!(
        r#"
var<workgroup> f2d_m: array<f32, 256>;
var<workgroup> f2d_ssc: array<f32, 256>;
var<workgroup> f2d_l: array<f32, 256>;

@compute @workgroup_size(256)
fn {FLASH2_DEEP_ENTRY}(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) tid: vec3<u32>
) {{
    let h = wg.x;
    if (h >= fd_params.n_heads) {{
        return;
    }}
    let hd = fd_params.head_dim;
    let splits = {splits}u;
    let stride = hd + 2u;
    let base = h * splits * stride;
    let lid = tid.x;
    var m_i = fd_neg_inf();
    if (lid < splits) {{
        m_i = fd_scratch[base + lid * stride];
    }}
    f2d_m[lid] = m_i;
    workgroupBarrier();
    var m_glob = fd_neg_inf();
    for (var s = 0u; s < splits; s = s + 1u) {{
        m_glob = max(m_glob, f2d_m[s]);
    }}
    var sc = 0.0;
    if (lid < splits && m_i > fd_neg_inf()) {{
        sc = fd_exp(m_i - m_glob);
    }}
    f2d_ssc[lid] = sc;
    var lc = 0.0;
    if (lid < splits) {{
        lc = fd_scratch[base + lid * stride + 1u] * sc;
    }}
    f2d_l[lid] = lc;
    workgroupBarrier();
    var l_glob = 0.0;
    for (var s = 0u; s < splits; s = s + 1u) {{
        l_glob = l_glob + f2d_l[s];
    }}
    var inv_l = 0.0;
    if (l_glob > 0.0) {{
        inv_l = fd_recip(l_glob);
    }}
    let hw = hd >> 1u;
    for (var w = lid; w < hw; w = w + FD_BLOCK) {{
        let d0 = w * 2u;
        var a0 = 0.0;
        var a1 = 0.0;
        for (var s = 0u; s < splits; s = s + 1u) {{
            let sb = base + s * stride + 2u + d0;
            let ssc = f2d_ssc[s];
            a0 = fma(fd_scratch[sb], ssc, a0);
            a1 = fma(fd_scratch[sb + 1u], ssc, a1);
        }}
        fd_out[h * hw + w] = (bf16_encode(a0 * inv_l) & 0xffffu)
            | ((bf16_encode(a1 * inv_l) & 0xffffu) << 16u);
    }}
}}
"#
    )
}

pub const E4B_DEEP_TPW_ENV: &str = "NV_E4B_DEEP_TPW";

pub const E4B_DEEP_TPW_MAX: u32 = 4;

pub fn deep_tpw_env() -> u32 {
    let Some(n) = deep_env_u32(E4B_DEEP_TPW_ENV) else {
        return 1;
    };
    if n <= 1 {
        return 1;
    }
    assert!(
        n <= E4B_DEEP_TPW_MAX,
        "{E4B_DEEP_TPW_ENV}={n} exceeds {E4B_DEEP_TPW_MAX}; each in-flight token per warp costs \
         fold x tokens live score registers plus tokens decoded k/v words"
    );
    n
}

pub fn deep_tpw_entry(hd_max: u32, fold: u32, tpw: u32, sd: bool) -> String {
    format!(
        "e4b_dfold{fold}t{tpw}_{hd_max}_stage1_fp8{}",
        if sd { "_sd" } else { "" }
    )
}

pub fn deep_tpw_source(hd_max: u32, fold: u32, tpw: u32, sd: bool) -> String {
    use std::fmt::Write;
    assert!(
        hd_max > 0 && hd_max % 32 == 0 && hd_max <= 512,
        "deep tpw stage1 hd_max {hd_max} must be a positive multiple of 32 up to 512"
    );
    assert!(
        (1..=wk::flash_decode::MAX_GQA_FOLD as u32).contains(&fold),
        "deep tpw stage1 folds {fold} query heads per workgroup"
    );
    assert!((2..=E4B_DEEP_TPW_MAX).contains(&tpw), "deep tpw {tpw} outside 2..={E4B_DEEP_TPW_MAX}");
    let accs = hd_max / 32;
    let entry = deep_tpw_entry(hd_max, fold, tpw, sd);
    let p = format!("df{fold}t{tpw}h{hd_max}{}", if sd { "s" } else { "e" });
    let (kf, vf) = if sd {
        ("fd_k_fp8_sd", "fd_v_fp8_sd")
    } else {
        ("fd_k_fp8", "fd_v_fp8")
    };
    let scale_fold = if sd { " * bitcast<f32>(0x7B800000u)" } else { "" };
    let mut b = String::with_capacity(32768);
    writeln!(b, "var<workgroup> {p}_qsh: array<f32, {}>;", hd_max * fold).unwrap();
    writeln!(b, "var<workgroup> {p}_sacc: array<f32, {}>;", hd_max * 8).unwrap();
    writeln!(b, "var<workgroup> {p}_sm: array<f32, 8>;").unwrap();
    writeln!(b, "var<workgroup> {p}_sl: array<f32, 8>;").unwrap();
    writeln!(
        b,
        "\nfn {p}_reduce(lid: u32, x: f32) -> f32 {{\n    var a = x;\n    a = a + \
         subgroupShuffleXor(a, 16u);\n    a = a + subgroupShuffleXor(a, 8u);\n    a = a + \
         subgroupShuffleXor(a, 4u);\n    a = a + subgroupShuffleXor(a, 2u);\n    a = a + \
         subgroupShuffleXor(a, 1u);\n    return a;\n}}"
    )
    .unwrap();
    writeln!(
        b,
        "\nfn {p}_epilogue(lid: u32, lane: u32, warp: u32, hd: u32, slot: u32, m: f32, l: f32) \
         {{\n    if (lane == 0u) {{\n        {p}_sm[warp] = m;\n        {p}_sl[warp] = l;\n    \
         }}\n    workgroupBarrier();\n    if (warp == 0u) {{\n        var m_blk = \
         fd_neg_inf();\n        for (var w = 0u; w < FD_WARPS; w = w + 1u) {{\n            m_blk \
         = max(m_blk, {p}_sm[w]);\n        }}\n        var l_blk = 0.0;\n        for (var w = 0u; \
         w < FD_WARPS; w = w + 1u) {{\n            if ({p}_sm[w] > fd_neg_inf()) {{\n             \
         l_blk = l_blk + fd_round({p}_sl[w] * fd_exp({p}_sm[w] - m_blk));\n            }}\n       \
         }}\n        if (lane == 0u) {{\n            fd_scratch[slot] = m_blk;\n            \
         fd_scratch[slot + 1u] = l_blk;\n        }}\n    }}\n    var m_blk = fd_neg_inf();\n    \
         for (var w = 0u; w < FD_WARPS; w = w + 1u) {{\n        m_blk = max(m_blk, \
         {p}_sm[w]);\n    }}\n    for (var d = lid; d < hd; d = d + FD_BLOCK) {{\n        var a = \
         0.0;\n        for (var w = 0u; w < FD_WARPS; w = w + 1u) {{\n            if ({p}_sm[w] > \
         fd_neg_inf()) {{\n                a = a + fd_round({p}_sacc[w * {hd_max}u + d] * \
         fd_exp({p}_sm[w] - m_blk));\n            }}\n        }}\n        fd_scratch[slot + 2u + \
         d] = a;\n    }}\n}}"
    )
    .unwrap();
    writeln!(
        b,
        "\n@compute @workgroup_size(256)\nfn {entry}(\n    @builtin(workgroup_id) wg: \
         vec3<u32>,\n    @builtin(local_invocation_id) tid: vec3<u32>\n) {{\n    let h0 = wg.x * \
         {fold}u;\n    let split = wg.y;\n    if (h0 >= fd_params.n_heads) {{\n        return;\n  \
         }}\n    let hd = fd_params.head_dim;\n    let nkv = fd_params.n_kv;\n    let group = \
         fd_params.n_heads / nkv;\n    let kvh = h0 / group;\n    let lid = tid.x;\n    let lane \
         = lid & 31u;\n    let warp = lid >> 5u;\n\n    for (var d = lid; d < hd * {fold}u; d = d \
         + FD_BLOCK) {{\n        {p}_qsh[d] = fd_q[h0 * hd + d];\n    }}\n    workgroupBarrier();"
    )
    .unwrap();
    for j in 0..fold {
        writeln!(b, "    var m{j} = fd_neg_inf();\n    var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "    var a{j}_{i} = 0.0;").unwrap();
        }
    }
    writeln!(
        b,
        "    let total = fd_params.total;\n    let base = fd_params.start + split * FD_WARPS;\n  \
         let stride = fd_params.splits * FD_WARPS;\n    var rounds = 0u;\n    if (total > base) \
         {{\n        rounds = (total - base + stride - 1u) / stride;\n    }}\n    let trounds = \
         (rounds + {tpw}u - 1u) / {tpw}u;\n    let n4 = hd >> 2u;\n\n    for (var r = 0u; r < \
         trounds; r = r + 1u) {{"
    )
    .unwrap();
    for t in 0..tpw {
        writeln!(
            b,
            "        let p{t} = base + warp + (r * {tpw}u + {t}u) * stride;\n        let live{t} \
             = p{t} < total;\n        var sp{t} = 0u;\n        if (live{t}) {{\n            sp{t} \
             = p{t};\n            if (fd_params.ring > 0u) {{\n                sp{t} = p{t} % \
             fd_params.ring;\n            }}\n        }}\n        var ks{t} = 0.0;\n        if \
             (live{t}) {{\n            ks{t} = fd_k_scales[sp{t} * nkv + kvh]{scale_fold};\n      \
             }}\n        let kb{t} = (sp{t} * nkv + kvh) * hd;"
        )
        .unwrap();
    }
    for j in 0..fold {
        for t in 0..tpw {
            writeln!(b, "        var pt{j}_{t} = 0.0;").unwrap();
        }
    }
    writeln!(
        b,
        "        for (var jv = lane; jv < n4; jv = jv + FD_LANES) {{\n            let qb = jv * \
         4u;"
    )
    .unwrap();
    for t in 0..tpw {
        for i in 0..4 {
            writeln!(b, "            let f{t}_{i} = {kf}(kb{t} + qb + {i}u);").unwrap();
        }
    }
    for j in 0..fold {
        for t in 0..tpw {
            writeln!(
                b,
                "            {{\n                let qo = {j}u * hd + qb;\n                var tv \
                 = {p}_qsh[qo + 1u] * f{t}_1;\n                tv = fma({p}_qsh[qo], f{t}_0, \
                 tv);\n                tv = fma({p}_qsh[qo + 2u], f{t}_2, tv);\n                \
                 tv = fma({p}_qsh[qo + 3u], f{t}_3, tv);\n                pt{j}_{t} = pt{j}_{t} + \
                 tv;\n            }}"
            )
            .unwrap();
        }
    }
    writeln!(b, "        }}").unwrap();
    for j in 0..fold {
        for t in 0..tpw {
            writeln!(
                b,
                "        let sc{j}_{t} = ({p}_reduce(lid, pt{j}_{t}) * ks{t}) * \
                 fd_params.scaling;"
            )
            .unwrap();
        }
    }
    let any_live = (0..tpw)
        .map(|t| format!("live{t}"))
        .collect::<Vec<_>>()
        .join(" || ");
    writeln!(b, "        if ({any_live}) {{").unwrap();
    for t in 0..tpw {
        writeln!(
            b,
            "            let vb{t} = (sp{t} * nkv + kvh) * hd;\n            let vs{t} = \
             fd_v_scales[sp{t} * nkv + kvh]{scale_fold};"
        )
        .unwrap();
    }
    for j in 0..fold {
        writeln!(b, "            var mn{j} = m{j};").unwrap();
        for t in 0..tpw {
            writeln!(
                b,
                "            if (live{t}) {{\n                mn{j} = max(mn{j}, sc{j}_{t});\n    \
                 }}"
            )
            .unwrap();
        }
        writeln!(b, "            let cr{j} = fd_exp(m{j} - mn{j});").unwrap();
        for t in 0..tpw {
            writeln!(
                b,
                "            let w{j}_{t} = select(0.0, fd_exp(sc{j}_{t} - mn{j}), live{t}) * \
                 vs{t};\n            let lw{j}_{t} = select(0.0, fd_exp(sc{j}_{t} - mn{j}), \
                 live{t});"
            )
            .unwrap();
        }
        let lsum = (0..tpw)
            .map(|t| format!("lw{j}_{t}"))
            .collect::<Vec<_>>()
            .join(" + ");
        writeln!(b, "            l{j} = fma(l{j}, cr{j}, {lsum});").unwrap();
        writeln!(b, "            m{j} = mn{j};").unwrap();
    }
    for i in 0..accs {
        writeln!(
            b,
            "            {{\n                let d = lane + {i}u * FD_LANES;\n                if \
             (d < hd) {{"
        )
        .unwrap();
        for t in 0..tpw {
            writeln!(b, "                    let v{t} = {vf}(vb{t} + d);").unwrap();
        }
        for j in 0..fold {
            let mut acc = format!("a{j}_{i} * cr{j}");
            for t in 0..tpw {
                acc = format!("fma(w{j}_{t}, v{t}, {acc})");
            }
            writeln!(b, "                    a{j}_{i} = {acc};").unwrap();
        }
        writeln!(b, "                }}\n            }}").unwrap();
    }
    writeln!(b, "        }}\n    }}").unwrap();
    for j in 0..fold {
        writeln!(b, "    workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(
                b,
                "    {{\n        let d = lane + {i}u * FD_LANES;\n        if (d < hd) {{\n        \
                 {p}_sacc[warp * {hd_max}u + d] = a{j}_{i};\n        }}\n    }}"
            )
            .unwrap();
        }
        writeln!(
            b,
            "    {p}_epilogue(lid, lane, warp, hd, ((h0 + {j}u) * fd_params.splits + split) * (hd \
             + 2u), m{j}, l{j});"
        )
        .unwrap();
    }
    writeln!(b, "}}").unwrap();
    b
}

const SPLIT_DEPTH_DEFAULT_LAST_SHALLOW_WIN: u32 = 2048;

const DEEP_SPLITS_DEFAULT_LADDER_WINNER_PAST_2048: u32 = 64;

fn deep_env_u32(name: &str) -> Option<u32> {
    let raw = std::env::var(name).ok()?;
    let t = raw.trim();
    if t.is_empty() {
        return None;
    }
    Some(t.parse().unwrap_or_else(|_| {
        panic!("{name}={t} must be a decimal token count; {DEEP_SPLIT_ARM_RULE}")
    }))
}

pub fn split_depth() -> u32 {
    deep_env_u32(E4B_SPLIT_DEPTH_ENV).unwrap_or(SPLIT_DEPTH_DEFAULT_LAST_SHALLOW_WIN)
}

pub fn deep_splits() -> u32 {
    deep_env_u32(E4B_DEEP_SPLITS_ENV)
        .unwrap_or(DEEP_SPLITS_DEFAULT_LADDER_WINNER_PAST_2048)
        .clamp(1, wk::flash_decode::MAX_SPLITS as u32)
}

pub fn deep_split_arm() -> Option<(u32, u32)> {
    let depth = split_depth();
    if depth == 0 || deep_splits() == decode_splits() {
        return None;
    }
    Some((depth, deep_splits()))
}

pub fn scratch_splits() -> u32 {
    let deep = deep_split_arm().map(|(_, s)| s).unwrap_or(0);
    decode_splits().max(deep).max(FLASH_SPLITS)
}

pub const PREFILL_QTILE_ENV: &str = "NV_E4B_WGPU_PREFILL_QTILE";

pub const PREFILL_QTILE_MAX: u32 = 8;

const MK_QSH_ELEMS: u32 = 2048;

pub fn prefill_qtile() -> u32 {
    let Ok(raw) = std::env::var(PREFILL_QTILE_ENV) else {
        return 1;
    };
    let t = raw.trim().to_string();
    if t.is_empty() {
        return 1;
    }
    let n: u32 = t.parse().unwrap_or_else(|_| {
        panic!("{PREFILL_QTILE_ENV}={t} must be a decimal count of query rows per K/V pass")
    });
    if n <= 1 {
        return 1;
    }
    assert!(
        n <= PREFILL_QTILE_MAX,
        "{PREFILL_QTILE_ENV}={n} exceeds {PREFILL_QTILE_MAX}; each tiled row holds head_dim/32 more \
         live accumulators per thread and shares the {MK_QSH_ELEMS}-element mk query staging"
    );
    n
}

pub fn flash1_mk_qtile_entry(hd_max: u32, tile: u32) -> String {
    format!("g4w_flash_rows_stage1_fp8_q{hd_max}_{tile}")
}

pub fn flash1_mk_qtile_source(hd_max: u32, tile: u32) -> String {
    use std::fmt::Write;
    assert!(
        hd_max > 0 && hd_max % 32 == 0 && hd_max <= 512,
        "mk qtile hd_max {hd_max} must be a positive multiple of 32 up to 512"
    );
    assert!(
        tile >= 1 && tile <= PREFILL_QTILE_MAX && tile * hd_max <= MK_QSH_ELEMS,
        "mk qtile {tile} x hd_max {hd_max} must fit the {MK_QSH_ELEMS}-element fd_qsh_mk staging"
    );
    let accs = hd_max / 32;
    let entry = flash1_mk_qtile_entry(hd_max, tile);
    let mut b = String::with_capacity(16384);

    writeln!(b, "\n@compute @workgroup_size(256)").unwrap();
    writeln!(
        b,
        "fn {entry}(\n    @builtin(workgroup_id) wg: vec3<u32>,\n    \
         @builtin(local_invocation_id) tid: vec3<u32>\n) {{"
    )
    .unwrap();
    writeln!(
        b,
        "    let h = wg.x;\n    let split = wg.y;\n    if (h >= fd_params.n_heads) {{\n        \
         return;\n    }}\n    let hd = fd_params.head_dim;\n    let nkv = fd_params.n_kv;\n    let \
         group = fd_params.n_heads / nkv;\n    let kvh = h / group;\n    let lid = tid.x;\n    let \
         lane = lid & 31u;\n    let warp = lid >> 5u;\n    let mr = fd_params.m_rows;\n    let \
         use_vec4 = (hd & 3u) == 0u;\n    let stride = fd_params.splits * FD_WARPS;\n    let base = \
         split * FD_WARPS;\n"
    )
    .unwrap();
    writeln!(b, "    for (var q0 = 0u; q0 < mr; q0 = q0 + {tile}u) {{").unwrap();
    writeln!(
        b,
        "        workgroupBarrier();\n        for (var t = lid; t < {tile}u * hd; t = t + \
         FD_BLOCK) {{\n            let jj = t / hd;\n            let d = t - jj * hd;\n            \
         let qi = q0 + jj;\n            var qv = 0.0;\n            if (qi < mr) {{\n                \
         qv = fd_q[(qi * fd_params.n_heads + h) * hd + d];\n            }}\n            \
         fd_qsh_mk[t] = qv;\n        }}\n        workgroupBarrier();"
    )
    .unwrap();
    for j in 0..tile {
        writeln!(b, "        var m{j} = fd_neg_inf();").unwrap();
        writeln!(b, "        var l{j} = 0.0;").unwrap();
        for i in 0..accs {
            writeln!(b, "        var a{j}_{i} = 0.0;").unwrap();
        }
    }
    writeln!(
        b,
        "        let qhi = min(q0 + {tile}u - 1u, mr - 1u);\n        let tot_hi = fd_params.total - \
         (mr - 1u - qhi);\n        var rounds = 0u;\n        if (tot_hi > base) {{\n            \
         rounds = (tot_hi - base + stride - 1u) / stride;\n        }}"
    )
    .unwrap();
    writeln!(
        b,
        "        for (var r = 0u; r < rounds; r = r + 1u) {{\n            let p = base + warp + r * \
         stride;\n            let any = p < tot_hi;\n            var sp = p;\n            if \
         (fd_params.ring > 0u) {{\n                sp = p % fd_params.ring;\n            }}"
    )
    .unwrap();
    for j in 0..tile {
        writeln!(b, "            var pt{j} = 0.0;").unwrap();
    }
    writeln!(b, "            var ks = 0.0;").unwrap();
    writeln!(b, "            if (any) {{").unwrap();
    writeln!(b, "                let kbase = (sp * nkv + kvh) * hd;").unwrap();
    writeln!(b, "                ks = fd_k_scales[sp * nkv + kvh];").unwrap();
    writeln!(b, "                if (use_vec4) {{").unwrap();
    writeln!(b, "                    let n4 = hd >> 2u;").unwrap();
    writeln!(
        b,
        "                    for (var jv = lane; jv < n4; jv = jv + FD_LANES) {{"
    )
    .unwrap();
    writeln!(b, "                        let qb = jv * 4u;").unwrap();
    writeln!(b, "                        let kb = kbase + qb;").unwrap();
    for i in 0..4 {
        writeln!(b, "                        let e{i} = fd_k_fp8(kb + {i}u);").unwrap();
    }
    for j in 0..tile {
        writeln!(b, "                        {{").unwrap();
        writeln!(b, "                            let qo = {j}u * hd + qb;").unwrap();
        writeln!(b, "                            var t = fd_qsh_mk[qo + 1u] * e1;").unwrap();
        writeln!(b, "                            t = fma(fd_qsh_mk[qo], e0, t);").unwrap();
        writeln!(b, "                            t = fma(fd_qsh_mk[qo + 2u], e2, t);").unwrap();
        writeln!(b, "                            t = fma(fd_qsh_mk[qo + 3u], e3, t);").unwrap();
        writeln!(b, "                            pt{j} = pt{j} + t;").unwrap();
        writeln!(b, "                        }}").unwrap();
    }
    writeln!(b, "                    }}").unwrap();
    writeln!(b, "                }} else {{").unwrap();
    writeln!(
        b,
        "                    for (var d = lane; d < hd; d = d + FD_LANES) {{"
    )
    .unwrap();
    writeln!(b, "                        let kx = fd_k_fp8(kbase + d);").unwrap();
    for j in 0..tile {
        writeln!(
            b,
            "                        pt{j} = fma(fd_qsh_mk[{j}u * hd + d], kx, pt{j});"
        )
        .unwrap();
    }
    writeln!(b, "                    }}").unwrap();
    writeln!(b, "                }}").unwrap();
    writeln!(b, "            }}").unwrap();
    for j in 0..tile {
        writeln!(
            b,
            "            let sc{j} = (fd_warp_sum(lid, pt{j}) * ks) * fd_params.scaling;"
        )
        .unwrap();
    }
    writeln!(b, "            if (any) {{").unwrap();
    writeln!(b, "                let vbase = (sp * nkv + kvh) * hd;").unwrap();
    writeln!(b, "                let vsc = fd_v_scales[sp * nkv + kvh];").unwrap();
    for j in 0..tile {
        writeln!(b, "                let qj{j} = min(q0 + {j}u, mr - 1u);").unwrap();
        writeln!(
            b,
            "                let lv{j} = (q0 + {j}u < mr) && (p < fd_params.total - (mr - 1u - \
             qj{j}));"
        )
        .unwrap();
        writeln!(b, "                var cr{j} = 1.0;").unwrap();
        writeln!(b, "                var wv{j} = 0.0;").unwrap();
        writeln!(b, "                if (lv{j}) {{").unwrap();
        writeln!(b, "                    let mn = max(m{j}, sc{j});").unwrap();
        writeln!(b, "                    cr{j} = fd_exp(m{j} - mn);").unwrap();
        writeln!(b, "                    let wt = fd_exp(sc{j} - mn);").unwrap();
        writeln!(b, "                    l{j} = fma(l{j}, cr{j}, wt);").unwrap();
        writeln!(b, "                    wv{j} = wt * vsc;").unwrap();
        writeln!(b, "                    m{j} = mn;").unwrap();
        writeln!(b, "                }}").unwrap();
    }
    for i in 0..accs {
        writeln!(b, "                {{").unwrap();
        writeln!(b, "                    let d = lane + {i}u * FD_LANES;").unwrap();
        writeln!(b, "                    if (d < hd) {{").unwrap();
        writeln!(b, "                        let vx = fd_v_fp8(vbase + d);").unwrap();
        for j in 0..tile {
            writeln!(b, "                        if (lv{j}) {{").unwrap();
            writeln!(
                b,
                "                            a{j}_{i} = fma(wv{j}, vx, a{j}_{i} * cr{j});"
            )
            .unwrap();
            writeln!(b, "                        }}").unwrap();
        }
        writeln!(b, "                    }}").unwrap();
        writeln!(b, "                }}").unwrap();
    }
    writeln!(b, "            }}").unwrap();
    writeln!(b, "        }}\n").unwrap();
    for j in 0..tile {
        writeln!(b, "        if (q0 + {j}u < mr) {{").unwrap();
        writeln!(b, "            workgroupBarrier();").unwrap();
        for i in 0..accs {
            writeln!(b, "            {{").unwrap();
            writeln!(b, "                let d = lane + {i}u * FD_LANES;").unwrap();
            writeln!(b, "                if (d < hd) {{").unwrap();
            writeln!(
                b,
                "                    fd_sacc[warp * FD_MAX_HD + d] = a{j}_{i};"
            )
            .unwrap();
            writeln!(b, "                }}").unwrap();
            writeln!(b, "            }}").unwrap();
        }
        writeln!(
            b,
            "            let slot{j} = ((h * mr + q0 + {j}u) * fd_params.splits + split) * (hd + \
             2u);\n            fd_stage1_epilogue(lid, lane, warp, hd, slot{j}, m{j}, l{j});"
        )
        .unwrap();
        writeln!(b, "        }}").unwrap();
    }
    writeln!(b, "    }}\n}}").unwrap();
    b
}

pub fn flash1_mk_qtile_entry_sd(hd_max: u32, tile: u32) -> String {
    format!("{}_sd", flash1_mk_qtile_entry(hd_max, tile))
}

pub fn flash1_mk_qtile_source_sd(hd_max: u32, tile: u32) -> String {
    flash_sd_rewrite(
        &flash1_mk_qtile_source(hd_max, tile),
        &flash1_mk_qtile_entry(hd_max, tile),
        E4B_SD_K_SCALE_ANCHOR,
        E4B_SD_VSC_SCALE_ANCHOR,
    )
}

pub fn gqa_group_of(config: &Gemma4Config) -> usize {
    let s = config.num_attention_heads / config.num_kv_heads_for(LayerType::SlidingAttention);
    let f = config.num_attention_heads / config.num_kv_heads_for(LayerType::FullAttention);
    let (mut a, mut b) = (s.max(f), s.min(f));
    while b > 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

pub fn flash1_head_dims(config: &Gemma4Config) -> Vec<u32> {
    let mut v = vec![
        config.head_dim_for(LayerType::SlidingAttention) as u32,
        config.head_dim_for(LayerType::FullAttention) as u32,
    ];
    v.sort_unstable();
    v.dedup();
    v
}

pub fn flash1_sg_supported(ctx: &WgpuContext) -> bool {
    matches!(ctx.subgroup_width(), Some(w) if w >= 32 && w % 32 == 0)
}

fn flash1_sg_enabled(ctx: &WgpuContext) -> bool {
    if std::env::var("NV_E4B_WGPU_FLASH1_SG").ok().as_deref() == Some("0") {
        return false;
    }
    flash1_sg_supported(ctx)
}

fn flash1_hd_specialize_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_FLASH1_HD").ok().as_deref() != Some("0")
}

const LMHEAD_I8_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_lmhead_i8.wgsl");

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct I8hParams {
    n_rows: u32,
    k_elems: u32,
    groups_x: u32,
    dst_word_off: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GcParams {
    rows_per_chunk: u32,
    row_words: u32,
    total_rows: u32,
    n_chunks: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct AxParams {
    n_words: u32,
    sa: f32,
    sb: f32,
    pad0: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GmParams {
    n_words: u32,
    pli_word_off: u32,
    tok_words: u32,
    pli_stride: u32,
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
struct MkParams {
    m: u32,
    x_stride_words: u32,
    y_stride_words: u32,
    dst_word_off: u32,
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
struct RmsParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct FncParams {
    hidden: u32,
    batch: u32,
    eps: f32,
    words_per_row: u32,
    scale: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
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
struct GemvW4Params {
    n_rows: u32,
    k_elems: u32,
    gs: u32,
    w_row_words: u32,
    scale_row_stride: u32,
    groups_x: u32,
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

pub struct HostW4 {
    pub packed: Vec<u32>,
    pub scales: Vec<u16>,
    pub gs: usize,
}

pub struct HostLin {
    pub w: Vec<u16>,
    pub n: usize,
    pub k: usize,
    pub q: Option<HostW4>,
}

impl HostLin {
    pub fn new(w: Vec<u16>, n: usize, k: usize) -> Self {
        Self { w, n, k, q: None }
    }

    pub fn new_w4(packed: Vec<u32>, scales: Vec<u16>, gs: usize, n: usize, k: usize) -> Self {
        Self {
            w: Vec::new(),
            n,
            k,
            q: Some(HostW4 { packed, scales, gs }),
        }
    }
}

pub struct E4bHostLayer {
    pub kind: LayerType,

    pub kv_source: Option<usize>,

    pub input_ln: Vec<u16>,
    pub post_attn_ln: Vec<u16>,
    pub pre_ff_ln: Vec<u16>,
    pub post_ff_ln: Vec<u16>,
    pub post_per_layer_input_norm: Vec<u16>,
    pub q_norm: Vec<u16>,

    pub k_norm: Vec<u16>,
    pub layer_scalar: f32,
    pub has_v: bool,

    pub qkv: HostLin,
    pub o: HostLin,
    pub gate_up: HostLin,
    pub down: HostLin,
    pub per_layer_input_gate: HostLin,
    pub per_layer_projection: HostLin,
}

pub struct E4bHostWeights {
    pub embed: Vec<u16>,
    pub embed_per_layer: Vec<u16>,
    pub per_layer_model_projection: HostLin,
    pub per_layer_projection_norm: Vec<u16>,
    pub final_norm: Vec<u16>,
    pub layers: Vec<E4bHostLayer>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkPlan {
    pub rows_per_chunk: usize,
    pub n_chunks: usize,
}

pub fn plan_chunks(rows: usize, row_bytes: usize, limit: u64) -> Result<ChunkPlan> {
    anyhow::ensure!(
        rows >= 2 && rows.is_multiple_of(2),
        "table rows must be even, got {rows}"
    );
    anyhow::ensure!(row_bytes > 0, "row_bytes must be > 0");
    let cap = limit.min(CHUNK_BYTE_CAP);
    let max_rows = ((cap / row_bytes as u64) as usize) & !1usize;
    anyhow::ensure!(
        max_rows >= 2,
        "a single {row_bytes}-byte row does not fit a {cap}-byte storage binding"
    );
    let mut rows_per_chunk = rows.div_ceil(rows.div_ceil(max_rows));
    if rows_per_chunk % 2 == 1 {
        rows_per_chunk += 1;
    }
    if rows_per_chunk > max_rows {
        rows_per_chunk = max_rows;
    }
    let n_chunks = rows.div_ceil(rows_per_chunk);
    anyhow::ensure!(
        n_chunks <= MAX_TABLE_CHUNKS,
        "table needs {n_chunks} chunks, the gather shader binds at most {MAX_TABLE_CHUNKS}"
    );
    Ok(ChunkPlan {
        rows_per_chunk,
        n_chunks,
    })
}

#[derive(Clone)]
struct Pass {
    pipeline: Arc<wgpu::ComputePipeline>,
    bind: wgpu::BindGroup,
    grid: (u32, u32, u32),

    bound_bytes: u64,
    widest_bytes: u64,
}

fn bind_bytes<'a>(bufs: impl Iterator<Item = &'a wgpu::Buffer>) -> (u64, u64) {
    let mut sum = 0u64;
    let mut widest = 0u64;
    for b in bufs {
        sum += b.size();
        widest = widest.max(b.size());
    }
    (sum, widest)
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SplitArm {
    Shallow,
    Deep,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProfMode {
    Off,
    PassTotal,
    PerDispatch,
}

pub const MAX_CHAIN: usize = 8;

pub fn chain_k_from_env() -> usize {
    std::env::var("NV_WGPU_CHAIN")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|k| k.clamp(1, MAX_CHAIN))
        .unwrap_or(1)
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum W4Variant {
    Block,
    V4,
    Sg16,
}

enum GpuProj {
    Bf16 {
        w: GpuTensor<u32>,
        params: GpuUniform<GemvBf16Params>,
        grid: (u32, u32, u32),
        n: usize,
    },
    I8 {
        w: GpuTensor<u32>,
        scales: GpuTensor<f32>,
        params: GpuUniform<GemvBf16Params>,
        grid: (u32, u32, u32),
        n: usize,
        group: usize,
    },
    W4 {
        packed: GpuTensor<u32>,
        scales: GpuTensor<u32>,
        params: GpuUniform<GemvW4Params>,
        grid: (u32, u32, u32),
        variant: W4Variant,
        n: usize,
        k: usize,
        gs: usize,
    },
}

struct GpuLayerKv {
    k_fp8: GpuTensor<u32>,
    v_fp8: GpuTensor<u32>,
    k_scales: GpuTensor<f32>,
    v_scales: GpuTensor<f32>,
}

enum BigBf16<'a> {
    Bits(std::borrow::Cow<'a, [u16]>),
    Raw(&'a [u8]),
}

impl BigBf16<'_> {
    fn len(&self) -> usize {
        match self {
            Self::Bits(v) => v.len(),
            Self::Raw(bytes) => bytes.len() / 2,
        }
    }

    fn upload_chunk(
        &self,
        ctx: &WgpuContext,
        label: &str,
        start: usize,
        elems: usize,
    ) -> wgpu::Buffer {
        match self {
            Self::Bits(v) => {
                dispatch::storage_from_slice(ctx, label, &pack_pairs(&v[start..start + elems]))
            }
            Self::Raw(bytes) => {
                storage_from_bytes(ctx, label, &bytes[start * 2..(start + elems) * 2])
            }
        }
    }

    fn bits_chunk(&self, start: usize, elems: usize) -> std::borrow::Cow<'_, [u16]> {
        match self {
            Self::Bits(v) => match v {
                std::borrow::Cow::Borrowed(s) => {
                    std::borrow::Cow::Borrowed(&s[start..start + elems])
                }
                std::borrow::Cow::Owned(s) => {
                    std::borrow::Cow::Owned(s[start..start + elems].to_vec())
                }
            },
            Self::Raw(bytes) => {
                let raw = &bytes[start * 2..(start + elems) * 2];
                let mut out = vec![0u16; elems];
                for (i, o) in out.iter_mut().enumerate() {
                    *o = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                }
                std::borrow::Cow::Owned(out)
            }
        }
    }
}

fn storage_from_bytes(ctx: &WgpuContext, label: &str, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len().max(4) as u64).next_multiple_of(8);
    let buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: true,
    });
    {
        let mut view = buf
            .slice(..)
            .get_mapped_range_mut()
            .expect("map upload buffer at creation");
        view.slice(..data.len()).copy_from_slice(data);
    }
    buf.unmap();
    buf
}

struct GpuTable {
    chunks: Vec<wgpu::Buffer>,
    plan: ChunkPlan,
    row_words: usize,
    rows: usize,
}

impl GpuTable {
    fn upload(
        ctx: &WgpuContext,
        label: &str,
        data: &BigBf16<'_>,
        rows: usize,
        row_elems: usize,
    ) -> Result<Self> {
        anyhow::ensure!(
            data.len() == rows * row_elems,
            "{label}: {} elements for {rows}x{row_elems}",
            data.len()
        );
        anyhow::ensure!(
            row_elems.is_multiple_of(2),
            "{label}: row_elems must be even"
        );
        let row_words = row_elems / 2;
        let limit = ctx
            .caps
            .max_storage_buffer_binding_size
            .min(ctx.caps.max_buffer_size);
        let plan = plan_chunks(rows, row_words * 4, limit)
            .with_context(|| format!("chunk plan for {label}"))?;
        let mut chunks = Vec::with_capacity(plan.n_chunks);
        for c in 0..plan.n_chunks {
            let lo = c * plan.rows_per_chunk;
            let hi = ((c + 1) * plan.rows_per_chunk).min(rows);
            chunks.push(data.upload_chunk(ctx, label, lo * row_elems, (hi - lo) * row_elems));
            ctx.queue.submit(std::iter::empty());
            ctx.poll_blocking().map_err(err)?;
        }
        Ok(Self {
            chunks,
            plan,
            row_words,
            rows,
        })
    }

    fn binds(&self) -> Vec<(u32, &wgpu::Buffer)> {
        let mut v = Vec::with_capacity(MAX_TABLE_CHUNKS);
        for i in 0..MAX_TABLE_CHUNKS {
            let c = self.chunks.get(i).unwrap_or(&self.chunks[0]);
            v.push((i as u32, c));
        }
        v
    }

    fn params(&self) -> GcParams {
        GcParams {
            rows_per_chunk: self.plan.rows_per_chunk as u32,
            row_words: self.row_words as u32,
            total_rows: self.rows as u32,
            n_chunks: self.plan.n_chunks as u32,
        }
    }

    fn chunk_rows(&self, c: usize) -> usize {
        let lo = c * self.plan.rows_per_chunk;
        ((c + 1) * self.plan.rows_per_chunk).min(self.rows) - lo
    }
}

struct Pipelines {
    w4_grain: wk::gemv_w4a16::ScaleGrain,
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

    flash1: Vec<(u32, Arc<wgpu::ComputePipeline>)>,
    flash1_fold: Vec<(u32, Arc<wgpu::ComputePipeline>)>,
    flash1_fold_deep: Vec<(u32, Arc<wgpu::ComputePipeline>)>,
    gqa_fold: u32,
    deep_fold: u32,
    flash1_entry: String,
    flash2_pk: Arc<wgpu::ComputePipeline>,
    flash2_pk_deep: Option<Arc<wgpu::ComputePipeline>>,
    gemv_pk: Arc<wgpu::ComputePipeline>,
    gemv_pk3: Arc<wgpu::ComputePipeline>,
    gemv_w4_pk: Arc<wgpu::ComputePipeline>,
    gemv_w4_pk3: Arc<wgpu::ComputePipeline>,
    gemv_w4_v4_pk: Arc<wgpu::ComputePipeline>,
    gemv_w4_v4_pk3: Arc<wgpu::ComputePipeline>,
    gemv_w4_sg_pk: Option<Arc<wgpu::ComputePipeline>>,
    gemv_w4_sg_pk3: Option<Arc<wgpu::ComputePipeline>>,
    gemv_w4_sg_pkm: Option<Arc<wgpu::ComputePipeline>>,
    gemv_w4_sg_pkm3: Option<Arc<wgpu::ComputePipeline>>,
    gemv_i8_pk: Option<Arc<wgpu::ComputePipeline>>,
    gemv_i8_pk3: Option<Arc<wgpu::ComputePipeline>>,
    gemv_i8g_pk: Option<Arc<wgpu::ComputePipeline>>,
    gemv_i8g_pk3: Option<Arc<wgpu::ComputePipeline>>,
    lmhead_sg: Option<Arc<wgpu::ComputePipeline>>,
    lmhead_i8: Option<Arc<wgpu::ComputePipeline>>,
    gelu_even: Arc<wgpu::ComputePipeline>,
    axpby: Arc<wgpu::ComputePipeline>,
    gatemul: Arc<wgpu::ComputePipeline>,
    am1: Arc<wgpu::ComputePipeline>,
    am2: Arc<wgpu::ComputePipeline>,
    mk: Option<MkPipelines>,
    mk_verify: Option<MkPipelines>,
    fnc: Option<FncPipelines>,
    fac: Option<FacPipelines>,
}

struct FncPipelines {
    a: Arc<wgpu::ComputePipeline>,
    b: Arc<wgpu::ComputePipeline>,
    c: Arc<wgpu::ComputePipeline>,

    unrolled: [bool; 3],
}

struct FacPipelines {
    q: Arc<wgpu::ComputePipeline>,
    k: Arc<wgpu::ComputePipeline>,
    v: Arc<wgpu::ComputePipeline>,
}

struct MkPipelines {
    rows: usize,
    gather: Arc<wgpu::ComputePipeline>,
    gemm_bf16_pk: Arc<wgpu::ComputePipeline>,
    gemm_bf16_pk3: Arc<wgpu::ComputePipeline>,
    gemm_w4_pk: Arc<wgpu::ComputePipeline>,
    gemm_w4_pk3: Arc<wgpu::ComputePipeline>,
    gemm_w4_v4_pk: Arc<wgpu::ComputePipeline>,
    gemm_w4_v4_pk3: Arc<wgpu::ComputePipeline>,
    gemm_w4_sg_pk: Option<Arc<wgpu::ComputePipeline>>,
    gemm_w4_sg_pk3: Option<Arc<wgpu::ComputePipeline>>,
    gemm_i8_pk: Option<Arc<wgpu::ComputePipeline>>,
    gemm_i8_pk3: Option<Arc<wgpu::ComputePipeline>>,
    gemm_i8g_pk: Option<Arc<wgpu::ComputePipeline>>,
    gemm_i8g_pk3: Option<Arc<wgpu::ComputePipeline>>,
    flash1: Arc<wgpu::ComputePipeline>,
    flash1_qtile: Option<Arc<wgpu::ComputePipeline>>,
    flash2_pk: Arc<wgpu::ComputePipeline>,
    gatemul: Arc<wgpu::ComputePipeline>,
}

fn sg16_enabled(ctx: &WgpuContext) -> bool {
    if std::env::var("NV_E4B_WGPU_W4_SG").ok().as_deref() == Some("0") {
        return false;
    }
    wk::gemv_w4a16::sg_pk_supported(ctx.subgroup_width())
}

fn w4_mr() -> u32 {
    let mr = std::env::var("NV_E4B_W4_MR")
        .ok()
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(1);
    if matches!(mr, 1 | 2 | 4 | 8) {
        mr
    } else {
        1
    }
}

fn w4_route_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_W4_ROUTE").ok().as_deref() == Some("1")
}

fn w4_grain_enabled() -> bool {
    std::env::var("NV_E4B_W4_GRAIN").ok().as_deref() != Some("0")
}

fn st_elems(l: &nv_weights::WeightLoader, name: &str) -> Option<usize> {
    l.shape_of(name).map(|s| s.iter().product())
}

fn w4_linear_names(n_layers: usize) -> Vec<String> {
    let mut v = vec![format!("{PREFIX}.per_layer_model_projection")];
    for i in 0..n_layers {
        let p = format!("{PREFIX}.layers.{i}");
        for suffix in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "per_layer_input_gate",
            "per_layer_projection",
        ] {
            v.push(format!("{p}.{suffix}"));
        }
    }
    v
}

fn uniform_w4_group_size(n_layers: usize, src: &WeightSource<'_>) -> Option<usize> {
    let mut seen: Option<usize> = None;
    let mut agree = |gs: usize| {
        if gs == 0 || seen.is_some_and(|s| s != gs) {
            seen = Some(0);
        } else {
            seen = Some(gs);
        }
    };
    match src {
        WeightSource::Host(w) => {
            let lins = std::iter::once(&w.per_layer_model_projection).chain(
                w.layers.iter().flat_map(|l| {
                    [
                        &l.qkv,
                        &l.o,
                        &l.gate_up,
                        &l.down,
                        &l.per_layer_input_gate,
                        &l.per_layer_projection,
                    ]
                }),
            );
            for l in lins {
                if let Some(q) = &l.q {
                    agree(q.gs);
                }
            }
        }
        WeightSource::Loader(l) => {
            for name in w4_linear_names(n_layers) {
                let packed = format!("{name}.weight_packed");
                if !l.has(&packed) {
                    continue;
                }
                let (Some(pw), Some(se)) = (
                    st_elems(l, &packed),
                    st_elems(l, &format!("{name}.weight_scale")),
                ) else {
                    return None;
                };
                if se == 0 || !(8 * pw).is_multiple_of(se) {
                    return None;
                }
                agree(8 * pw / se);
            }
        }
    }
    seen.filter(|gs| *gs != 0)
}

fn checkpoint_w4_grain(
    n_layers: usize,
    src: &WeightSource<'_>,
) -> (wk::gemv_w4a16::ScaleGrain, Option<usize>) {
    let gs = uniform_w4_group_size(n_layers, src);
    let grain = match gs {
        Some(g) if w4_grain_enabled() && g >= 32 => {
            wk::gemv_w4a16::ScaleGrain::fastest_for_group_size(g)
                .unwrap_or(wk::gemv_w4a16::ScaleGrain::Ge32)
        }
        _ => wk::gemv_w4a16::ScaleGrain::Ge32,
    };
    (grain, gs)
}

fn na_route_enabled() -> bool {
    std::env::var("NV_WGPU_NA").ok().as_deref() == Some("1")
}

fn na_route_note(reason: &str) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!("[gemma4_e4b_wgpu] NV_WGPU_NA=1 but tensor-ops route not taken: {reason}");
    });
}

fn na_attn_enabled() -> bool {
    std::env::var("NV_WGPU_NA_ATTN").ok().as_deref() == Some("1")
}

fn na_attn_note(reason: &str) {
    static ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!(
            "[gemma4_e4b_wgpu] NV_WGPU_NA_ATTN=1 but tensor-ops attention not taken: {reason}"
        );
    });
}

fn w4_prefer_v4(n: usize, k: usize) -> bool {
    wk::gemv_w4a16::prefers_v4(n, k)
}

fn lmhead_i8_enabled() -> bool {
    std::env::var("NV_E4B_LMHEAD_INT8").ok().as_deref() == Some("1")
}

fn lmhead_sg_enabled(ctx: &WgpuContext) -> bool {
    std::env::var("NV_E4B_LMHEAD_SG").ok().as_deref() == Some("1") && wk::gemv_bf16::sg32_ok(ctx)
}

fn lmhead_sg_wg() -> u32 {
    std::env::var("NV_E4B_LMHEAD_SG_WG")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256)
}

fn flash_nozi_enabled() -> bool {
    dispatch::nozi_env_enabled()
        && std::env::var("NV_E4B_WGPU_FLASH_NOZI").ok().as_deref() != Some("0")
}

fn nozi_all_enabled() -> bool {
    dispatch::nozi_env_enabled()
        && std::env::var("NV_E4B_WGPU_NOZI_ALL").ok().as_deref() != Some("0")
}

fn raw_nozi_pipeline(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
) -> Result<Arc<wgpu::ComputePipeline>> {
    let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = ctx
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
    let pipeline = ctx
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some(entry),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[],
                zero_initialize_workgroup_memory: false,
            },
            cache: None,
        });
    if let Some(e) = pollster::block_on(scope.pop()) {
        anyhow::bail!("pipeline {label} (nozi): {e}");
    }
    Ok(Arc::new(pipeline))
}

fn nozi_all_pipeline(
    ctx: &WgpuContext,
    label: &str,
    source: &str,
    entry: &str,
) -> Result<Arc<wgpu::ComputePipeline>> {
    if nozi_all_enabled() {
        return raw_nozi_pipeline(ctx, label, source, entry);
    }
    dispatch::cached_compute_pipeline(ctx, label, source, entry)
        .map_err(|e| anyhow::anyhow!("pipeline {label}: {e}"))
}

fn fuse_norms_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_FUSE_NORMS").ok().as_deref() != Some("0")
}

pub const FNCU_ENTRY_A: &str = "e4b_fncu_a";
pub const FNCU_ENTRY_B: &str = "e4b_fncu_b";
pub const FNCU_ENTRY_C: &str = "e4b_fncu_c";
const FNCU_BLOCK: usize = 256;

fn fnc_unroll_mask() -> [bool; 3] {
    match std::env::var("NV_E4B_WGPU_FNC_UNROLL").ok().as_deref() {
        Some("0") => [false; 3],
        Some(v) if v.contains(['a', 'b', 'c']) => {
            [v.contains('a'), v.contains('b'), v.contains('c')]
        }
        _ => [true; 3],
    }
}

const FNCU_PRELUDE: &str = include_str!("../../nv-kernels/wgsl/e4b_fncu_prelude.wgsl");

pub fn fnc_unrolled_source(hidden: usize) -> Option<String> {
    use std::fmt::Write as _;
    if hidden == 0 || !hidden.is_multiple_of(2 * FNCU_BLOCK) {
        return None;
    }
    let words = hidden / 2;
    let it1 = hidden / FNCU_BLOCK;
    let it2 = words / FNCU_BLOCK;
    let half = FNCU_BLOCK / 2;

    let mut s = String::from(FNCU_PRELUDE);

    let pass1 = |s: &mut String| {
        for k in 0..it1 {
            let _ = writeln!(s, "    let a{k} = fncu_in[hw + {}u];", k * half);
        }
        let _ = writeln!(s, "    var acc = 0.0;");
        for k in 0..it1 {
            let _ = writeln!(
                s,
                "    let v{k} = select(bf16_lo(a{k}), bf16_hi(a{k}), odd);\n    \
                 acc = fma(v{k}, v{k}, acc);"
            );
        }
        let _ = writeln!(s, "    let rms1 = fncu_rms_reduce(lid, acc);");
    };
    let load_xg = |s: &mut String| {
        for k in 0..it2 {
            let off = k * FNCU_BLOCK;
            let _ = writeln!(s, "    let x{k} = fncu_in[base + lid + {off}u];");
            let _ = writeln!(s, "    let g{k} = fncu_w1[lid + {off}u];");
            let _ = writeln!(s, "    let r{k} = fncu_res[base + lid + {off}u];");
        }
    };
    let tail = |s: &mut String, src: &str, dst: &str| {
        let _ = writeln!(
            s,
            "    for (var i = lid; i < fncu_params.words_per_row; i = i + 256u) {{\n        \
             let sw = {src}[base + i];\n        \
             let ww = fncu_w2[i];\n        \
             let lo = bf16_lo(sw) * rms2 * bf16_lo(ww);\n        \
             let hi = bf16_hi(sw) * rms2 * bf16_hi(ww);\n        \
             {dst}[base + i] = bf16_pack(lo, hi);\n    }}"
        );
    };
    let head = |s: &mut String, name: &str| {
        let _ = writeln!(
            s,
            "\n@compute @workgroup_size(256)\nfn {name}(\n    \
             @builtin(workgroup_id) wg: vec3<u32>,\n    \
             @builtin(num_workgroups) nwg: vec3<u32>,\n    \
             @builtin(local_invocation_id) tid: vec3<u32>\n) {{\n    \
             let row = wg.x + wg.y * nwg.x;\n    \
             if (row >= fncu_params.batch) {{ return; }}\n    \
             let lid = tid.x;\n    \
             let base = row * {words}u;\n    \
             let hw = base + (lid >> 1u);\n    \
             let odd = (lid & 1u) == 1u;"
        );
    };

    head(&mut s, FNCU_ENTRY_A);
    pass1(&mut s);
    load_xg(&mut s);
    let _ = writeln!(&mut s, "    var acc2 = 0.0;");
    for k in 0..it2 {
        let off = k * FNCU_BLOCK;
        let _ = writeln!(
            &mut s,
            "    let tlo{k} = bf16_lo(x{k}) * rms1 * bf16_lo(g{k});\n    \
             let thi{k} = bf16_hi(x{k}) * rms1 * bf16_hi(g{k});\n    \
             let tw{k} = bf16_pack(tlo{k}, thi{k});\n    \
             let lo{k} = bf16_lo(tw{k}) + bf16_lo(r{k});\n    \
             let hi{k} = bf16_hi(tw{k}) + bf16_hi(r{k});\n    \
             let sw{k} = bf16_pack(lo{k}, hi{k});\n    \
             fncu_res[base + lid + {off}u] = sw{k};\n    \
             acc2 = acc2 + lo{k} * lo{k} + hi{k} * hi{k};"
        );
    }
    let _ = writeln!(&mut s, "    let rms2 = fncu_res_reduce(lid, acc2);");
    tail(&mut s, "fncu_res", "fncu_out");
    let _ = writeln!(&mut s, "}}");

    head(&mut s, FNCU_ENTRY_B);
    pass1(&mut s);
    load_xg(&mut s);
    let _ = writeln!(&mut s, "    let scale = fncu_params.scale;");
    for k in 0..it2 {
        let off = k * FNCU_BLOCK;
        let _ = writeln!(
            &mut s,
            "    let tlo{k} = bf16_lo(x{k}) * rms1 * bf16_lo(g{k});\n    \
             let thi{k} = bf16_hi(x{k}) * rms1 * bf16_hi(g{k});\n    \
             let tw{k} = bf16_pack(tlo{k}, thi{k});\n    \
             let lo{k} = (bf16_lo(r{k}) + bf16_lo(tw{k})) * scale;\n    \
             let hi{k} = (bf16_hi(r{k}) + bf16_hi(tw{k})) * scale;\n    \
             fncu_out[base + lid + {off}u] = bf16_pack(lo{k}, hi{k});"
        );
    }
    let _ = writeln!(&mut s, "}}");

    head(&mut s, FNCU_ENTRY_C);
    pass1(&mut s);
    load_xg(&mut s);
    let _ = writeln!(&mut s, "    let scale = fncu_params.scale;");
    for k in 0..it2 {
        let off = k * FNCU_BLOCK;
        let _ = writeln!(
            &mut s,
            "    let tlo{k} = bf16_lo(x{k}) * rms1 * bf16_lo(g{k});\n    \
             let thi{k} = bf16_hi(x{k}) * rms1 * bf16_hi(g{k});\n    \
             let tw{k} = bf16_pack(tlo{k}, thi{k});\n    \
             let lo{k} = (bf16_lo(r{k}) + bf16_lo(tw{k})) * scale;\n    \
             let hi{k} = (bf16_hi(r{k}) + bf16_hi(tw{k})) * scale;\n    \
             fncu_out[base + lid + {off}u] = bf16_pack(lo{k}, hi{k});"
        );
    }
    let _ = writeln!(&mut s, "    storageBarrier();");
    for k in 0..it1 {
        let _ = writeln!(&mut s, "    let d{k} = fncu_out[hw + {}u];", k * half);
    }
    let _ = writeln!(&mut s, "    var acc3 = 0.0;");
    for k in 0..it1 {
        let _ = writeln!(
            &mut s,
            "    let u{k} = select(bf16_lo(d{k}), bf16_hi(d{k}), odd);\n    \
             acc3 = fma(u{k}, u{k}, acc3);"
        );
    }
    let _ = writeln!(&mut s, "    let rms2 = fncu_rms_reduce(lid, acc3);");
    tail(&mut s, "fncu_out", "fncu_out2");
    let _ = writeln!(&mut s, "}}");
    Some(s)
}

fn fuse_attn_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_FUSE_ATTN").ok().as_deref() != Some("0")
}

fn fuse_head_argmax_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_FUSE_HEAD").ok().as_deref() == Some("1")
}

pub const PREFILL_M_MAX: usize = 128;
const PREFILL_SLAB: usize = wk::gemv_w4a16::SG_MK_MAX as usize;

fn prefill_m_from_env() -> usize {
    if std::env::var("NV_E4B_WGPU_PREFILL").ok().as_deref() == Some("0") {
        return 0;
    }
    std::env::var("NV_WGPU_PREFILL_M")
        .or_else(|_| std::env::var("NV_E4B_WGPU_PREFILL_M"))
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(10)
        .clamp(2, PREFILL_M_MAX)
}

fn prefill_tail_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_PREFILL_TAIL").ok().as_deref() != Some("0")
}

fn verify_m_from_env() -> Option<usize> {
    std::env::var("NV_E4B_WGPU_VERIFY_M")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .map(|v| v.clamp(1, PREFILL_M_MAX))
}

fn mk_widen(src: &str, mk_max: usize) -> String {
    if mk_max <= 8 {
        return src.to_string();
    }
    src.replace("array<f32, 8>", &format!("array<f32, {mk_max}>"))
        .replace("t < 8u", &format!("t < {mk_max}u"))
}

fn mk_unroll_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_MK_UNROLL").ok().as_deref() != Some("0")
}

fn w8_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_W8").ok().as_deref() == Some("1")
}

fn w8_group() -> usize {
    match std::env::var("NV_E4B_WGPU_W8_GROUP").ok().as_deref() {
        Some("128") => 128,
        _ => 0,
    }
}

fn i8_inner_dot(b: &mut String, indent: &str, acc: &str, xw: [&str; 4]) {
    use std::fmt::Write as _;
    for (wi, wname) in ["w0", "w1"].iter().enumerate() {
        for e in 0..4usize {
            let x = xw[wi * 2 + e / 2];
            let half = if e % 2 == 0 { "bf16_lo" } else { "bf16_hi" };
            writeln!(
                b,
                "{indent}{acc} = fma(int8_decode({wname}, {e}u), {half}({x}), {acc});"
            )
            .unwrap();
        }
    }
}

fn gemv_i8_pk_source() -> String {
    use std::fmt::Write as _;
    let mut b = String::new();
    b.push_str("struct G4wPkParams {\n    dst_word_off: u32,\n    pad0: u32,\n    pad1: u32,\n    pad2: u32,\n};\n\n");
    b.push_str("struct G4wSplitParams {\n    q_rows: u32,\n    kv_rows: u32,\n    v_off: u32,\n    pad0: u32,\n};\n\n");
    b.push_str("@group(0) @binding(22) var<storage, read> g4w_i8_scales: array<f32>;\n");
    b.push_str("@group(0) @binding(30) var<uniform> g4w_pk_params: G4wPkParams;\n");
    b.push_str("@group(0) @binding(31) var<storage, read_write> g4w_y_q: array<u32>;\n");
    b.push_str("@group(0) @binding(32) var<storage, read_write> g4w_y_k: array<u32>;\n");
    b.push_str("@group(0) @binding(33) var<storage, read_write> g4w_y_v: array<u32>;\n");
    b.push_str("@group(0) @binding(34) var<uniform> g4w_split_params: G4wSplitParams;\n\n");
    let pk_store = "        gemv_bf16_y[g4w_pk_params.dst_word_off + (row >> 1u)] = word;\n";
    let pk3_store = concat!(
        "        if (row < g4w_split_params.q_rows) {\n",
        "            g4w_y_q[row >> 1u] = word;\n",
        "        } else {\n",
        "            let kr = row - g4w_split_params.q_rows;\n",
        "            if (kr < g4w_split_params.kv_rows) {\n",
        "                g4w_y_k[kr >> 1u] = word;\n",
        "            }\n",
        "            if (row >= g4w_split_params.v_off) {\n",
        "                let vr = row - g4w_split_params.v_off;\n",
        "                if (vr < g4w_split_params.kv_rows) {\n",
        "                    g4w_y_v[vr >> 1u] = word;\n",
        "                }\n",
        "            }\n",
        "        }\n",
    );
    for (entry, store) in [
        ("g4w_gemv_i8_pk", pk_store),
        ("g4w_gemv_i8_pk3", pk3_store),
        ("g4w_gemv_i8g_pk", pk_store),
        ("g4w_gemv_i8g_pk3", pk3_store),
    ] {
        let grouped = entry.contains("i8g");
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
        if grouped {
            b.push_str(
                "    let sbase = select(0u, row * (gemv_bf16_params.k_elems >> 7u), live);\n",
            );
        }
        b.push_str("    var acc = 0.0;\n");
        b.push_str("    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {\n");
        b.push_str("        let wo = w_base + (v << 1u);\n");
        b.push_str("        let xo = v << 2u;\n");
        b.push_str("        let w0 = gemv_bf16_w[wo];\n");
        b.push_str("        let w1 = gemv_bf16_w[wo + 1u];\n");
        b.push_str("        let x0 = gemv_bf16_x[xo];\n");
        b.push_str("        let x1 = gemv_bf16_x[xo + 1u];\n");
        b.push_str("        let x2 = gemv_bf16_x[xo + 2u];\n");
        b.push_str("        let x3 = gemv_bf16_x[xo + 3u];\n");
        if grouped {
            b.push_str("        var d = 0.0;\n");
            i8_inner_dot(&mut b, "        ", "d", ["x0", "x1", "x2", "x3"]);
            b.push_str("        acc = fma(g4w_i8_scales[sbase + (v >> 4u)], d, acc);\n");
        } else {
            i8_inner_dot(&mut b, "        ", "acc", ["x0", "x1", "x2", "x3"]);
        }
        b.push_str("    }\n");
        b.push_str("    let sum = gemv_bf16_reduce(tid, lane, acc);\n");
        if grouped {
            b.push_str("    let total = sum;\n");
        } else {
            b.push_str("    let total = sum * g4w_i8_scales[select(0u, row, live)];\n");
        }
        b.push_str("    workgroupBarrier();\n");
        b.push_str("    if (lane == 0u) {\n        gemv_bf16_partial[tid] = total;\n    }\n");
        b.push_str("    workgroupBarrier();\n");
        b.push_str("    if (lane == 0u && live && (warp & 1u) == 0u) {\n");
        b.push_str("        let lo = bf16_encode(total) & 0xffffu;\n");
        b.push_str("        let hi_live = row + 1u < gemv_bf16_params.n_rows;\n");
        b.push_str(
            "        let hi = bf16_encode(gemv_bf16_partial[tid + GEMV_BF16_LANES]) & 0xffffu;\n",
        );
        b.push_str("        let word = lo | (select(0u, hi, hi_live) << 16u);\n");
        b.push_str(store);
        b.push_str("    }\n}\n\n");
    }
    b
}

fn mk_unrolled_i8_source(mk_max: usize) -> String {
    use std::fmt::Write as _;
    let mut b = String::new();
    b.push_str("struct G4wSplitParams {\n    q_rows: u32,\n    kv_rows: u32,\n    v_off: u32,\n    pad0: u32,\n};\n\n");
    b.push_str("struct G4wMkParams {\n    m: u32,\n    x_stride_words: u32,\n    y_stride_words: u32,\n    dst_word_off: u32,\n};\n\n");
    b.push_str("@group(0) @binding(22) var<storage, read> g4w_i8_scales: array<f32>;\n");
    b.push_str("@group(0) @binding(31) var<storage, read_write> g4w_y_q: array<u32>;\n");
    b.push_str("@group(0) @binding(32) var<storage, read_write> g4w_y_k: array<u32>;\n");
    b.push_str("@group(0) @binding(33) var<storage, read_write> g4w_y_v: array<u32>;\n");
    b.push_str("@group(0) @binding(34) var<uniform> g4w_split_params: G4wSplitParams;\n");
    b.push_str("@group(0) @binding(35) var<uniform> g4w_mk_params: G4wMkParams;\n\n");
    let pk_store = |t: usize| {
        format!(
            "            gemv_bf16_y[g4w_mk_params.dst_word_off + {t}u * g4w_mk_params.y_stride_words + (row >> 1u)] = word;\n"
        )
    };
    let pk3_store = |t: usize| {
        format!(
            concat!(
                "            if (row < g4w_split_params.q_rows) {{\n",
                "                g4w_y_q[{t}u * (g4w_split_params.q_rows >> 1u) + (row >> 1u)] = word;\n",
                "            }} else {{\n",
                "                let kr = row - g4w_split_params.q_rows;\n",
                "                if (kr < g4w_split_params.kv_rows) {{\n",
                "                    g4w_y_k[{t}u * (g4w_split_params.kv_rows >> 1u) + (kr >> 1u)] = word;\n",
                "                }}\n",
                "                if (row >= g4w_split_params.v_off) {{\n",
                "                    let vr = row - g4w_split_params.v_off;\n",
                "                    if (vr < g4w_split_params.kv_rows) {{\n",
                "                        g4w_y_v[{t}u * (g4w_split_params.kv_rows >> 1u) + (vr >> 1u)] = word;\n",
                "                    }}\n",
                "                }}\n",
                "            }}\n",
            ),
            t = t
        )
    };
    for (entry, store) in [
        ("g4w_gemm_i8_mk_pk", &pk_store as &dyn Fn(usize) -> String),
        ("g4w_gemm_i8_mk_pk3", &pk3_store),
        ("g4w_gemm_i8g_mk_pk", &pk_store),
        ("g4w_gemm_i8g_mk_pk3", &pk3_store),
    ] {
        let grouped = entry.contains("i8g");
        b.push_str("@compute @workgroup_size(256)\n");
        writeln!(b, "fn {entry}(").unwrap();
        b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
        b.push_str("    let tid = lid.x;\n");
        b.push_str("    let lane = tid & (GEMV_BF16_LANES - 1u);\n");
        b.push_str("    let warp = tid / GEMV_BF16_LANES;\n");
        b.push_str("    let row = wid.x * GEMV_BF16_ROWS + warp;\n");
        b.push_str("    let live = row < gemv_bf16_params.n_rows;\n");
        b.push_str("    let kv = select(0u, gemv_bf16_params.k_elems >> 3u, live);\n");
        b.push_str("    let w_base = select(0u, row * gemv_bf16_params.w_row_words, live);\n");
        b.push_str("    let mm = g4w_mk_params.m;\n");
        b.push_str("    let xs = g4w_mk_params.x_stride_words;\n");
        for t in 0..mk_max {
            writeln!(b, "    var acc{t} = 0.0;").unwrap();
        }
        if grouped {
            b.push_str(
                "    let sbase = select(0u, row * (gemv_bf16_params.k_elems >> 7u), live);\n",
            );
        }
        b.push_str("    for (var v = lane; v < kv; v = v + GEMV_BF16_LANES) {\n");
        b.push_str("        let wo = w_base + (v << 1u);\n");
        b.push_str("        let xo = v << 2u;\n");
        b.push_str("        let w0 = gemv_bf16_w[wo];\n");
        b.push_str("        let w1 = gemv_bf16_w[wo + 1u];\n");
        if grouped {
            b.push_str("        let sg = g4w_i8_scales[sbase + (v >> 4u)];\n");
        }
        for t in 0..mk_max {
            let base = if t == 0 {
                "xo".to_string()
            } else {
                format!("{t}u * xs + xo")
            };
            writeln!(b, "        if ({t}u < mm) {{").unwrap();
            writeln!(b, "            let x0 = gemv_bf16_x[{base}];").unwrap();
            writeln!(b, "            let x1 = gemv_bf16_x[{base} + 1u];").unwrap();
            writeln!(b, "            let x2 = gemv_bf16_x[{base} + 2u];").unwrap();
            writeln!(b, "            let x3 = gemv_bf16_x[{base} + 3u];").unwrap();
            if grouped {
                b.push_str("            var d = 0.0;\n");
                i8_inner_dot(&mut b, "            ", "d", ["x0", "x1", "x2", "x3"]);
                writeln!(b, "            acc{t} = fma(sg, d, acc{t});").unwrap();
            } else {
                i8_inner_dot(
                    &mut b,
                    "            ",
                    &format!("acc{t}"),
                    ["x0", "x1", "x2", "x3"],
                );
            }
            b.push_str("        }\n");
        }
        b.push_str("    }\n");
        if grouped {
            b.push_str("    let sc = 1.0;\n");
        } else {
            b.push_str("    let sc = g4w_i8_scales[select(0u, row, live)];\n");
        }
        for t in 0..mk_max {
            writeln!(b, "    if ({t}u < mm) {{").unwrap();
            writeln!(
                b,
                "        let total = gemv_bf16_reduce(tid, lane, acc{t}) * sc;"
            )
            .unwrap();
            b.push_str("        workgroupBarrier();\n");
            b.push_str("        if (lane == 0u) {\n            gemv_bf16_partial[tid] = total;\n        }\n");
            b.push_str("        workgroupBarrier();\n");
            b.push_str("        if (lane == 0u && live && (warp & 1u) == 0u) {\n");
            b.push_str("            let lo = bf16_encode(total) & 0xffffu;\n");
            b.push_str("            let hi_live = row + 1u < gemv_bf16_params.n_rows;\n");
            b.push_str("            let hi = bf16_encode(gemv_bf16_partial[tid + GEMV_BF16_LANES]) & 0xffffu;\n");
            b.push_str("            let word = lo | (select(0u, hi, hi_live) << 16u);\n");
            b.push_str(&store(t));
            b.push_str("        }\n");
            b.push_str("        workgroupBarrier();\n");
            b.push_str("    }\n");
        }
        b.push_str("}\n\n");
    }
    b
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct SmkLiveParams {
    n_rows: u32,
    k_elems: u32,
    row_words: u32,
    groups_x: u32,
    m_live: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

const SMK_LIVE_ENTRY: &str = "e4b_verify_smk_live";

fn verify_smk_live_source(m: usize) -> String {
    use std::fmt::Write as _;
    let mut b = String::new();
    b.push_str(
        "struct SmkLiveParams {\n    n_rows: u32,\n    k_elems: u32,\n    row_words: u32,\n    groups_x: u32,\n    m_live: u32,\n    pad0: u32,\n    pad1: u32,\n    pad2: u32,\n};\n\n",
    );
    b.push_str("@group(0) @binding(0) var<storage, read> smk_w: array<u32>;\n");
    b.push_str("@group(0) @binding(1) var<storage, read> smk_x: array<u32>;\n");
    b.push_str("@group(0) @binding(2) var<storage, read_write> smk_y: array<u32>;\n");
    b.push_str("@group(0) @binding(3) var<uniform> smk_params: SmkLiveParams;\n\n");
    b.push_str("const SMK_LANES: u32 = 32u;\nconst SMK_ROWS: u32 = 8u;\n\n");
    b.push_str("var<workgroup> smk_partial: array<f32, 256>;\n\n");
    b.push_str(
        "fn smk_row(wid: vec3<u32>, warp: u32) -> u32 {\n    return (wid.x + wid.y * smk_params.groups_x) * SMK_ROWS + warp;\n}\n\n",
    );
    b.push_str(
        "fn smk_reduce(tid: u32, lane: u32, acc: f32) -> f32 {\n    workgroupBarrier();\n    smk_partial[tid] = acc;\n    workgroupBarrier();\n    for (var stride = SMK_LANES >> 1u; stride > 0u; stride = stride >> 1u) {\n        if (lane < stride) {\n            smk_partial[tid] = smk_partial[tid] + smk_partial[tid + stride];\n        }\n        workgroupBarrier();\n    }\n    return smk_partial[tid - lane];\n}\n\n",
    );
    b.push_str("@compute @workgroup_size(256)\n");
    writeln!(
        b,
        "fn {SMK_LIVE_ENTRY}(\n    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {{"
    )
    .unwrap();
    b.push_str("    let tid = lid.x;\n");
    b.push_str("    let lane = tid & (SMK_LANES - 1u);\n");
    b.push_str("    let warp = tid / SMK_LANES;\n");
    b.push_str("    let row = smk_row(wid, warp);\n");
    b.push_str("    let live = row < smk_params.n_rows;\n");
    b.push_str("    let kv = select(0u, smk_params.k_elems >> 3u, live);\n");
    b.push_str("    let w_base = select(0u, row * smk_params.row_words, live);\n");
    b.push_str("    let ml = smk_params.m_live;\n");
    for t in 0..m {
        writeln!(b, "    var acc{t} = 0.0;").unwrap();
    }
    b.push_str("    for (var v = lane; v < kv; v = v + SMK_LANES) {\n");
    b.push_str("        let wo = w_base + (v << 2u);\n");
    b.push_str("        let xo = v << 2u;\n");
    b.push_str("        for (var j = 0u; j < 4u; j = j + 1u) {\n");
    b.push_str("            let ww = smk_w[wo + j];\n");
    b.push_str("            let wl = bf16_lo(ww);\n");
    b.push_str("            let wh = bf16_hi(ww);\n");
    for t in 0..m {
        let x_index = if t == 0 {
            "xo + j".to_string()
        } else {
            format!("{t}u * smk_params.row_words + xo + j")
        };
        writeln!(
            b,
            "            if ({t}u < ml) {{ let xw = smk_x[{x_index}]; acc{t} = acc{t} + (wl * bf16_lo(xw) + wh * bf16_hi(xw)); }}"
        )
        .unwrap();
    }
    b.push_str("        }\n    }\n");
    for t in 0..m {
        writeln!(b, "    if ({t}u < ml) {{").unwrap();
        writeln!(b, "        let total = smk_reduce(tid, lane, acc{t});").unwrap();
        writeln!(
            b,
            "        if (lane == 0u && live) {{\n            smk_y[{t}u * smk_params.n_rows + row] = bf16_encode(total);\n        }}"
        )
        .unwrap();
        b.push_str("    }\n");
    }
    b.push_str("}\n");
    compose(&b)
}

fn mk_unrolled_source(mk_max: usize) -> String {
    use std::fmt::Write as _;
    let mut b = String::new();
    b.push_str(
        "struct G4wSplitParams {\n    q_rows: u32,\n    kv_rows: u32,\n    v_off: u32,\n    pad0: u32,\n};\n\n",
    );
    b.push_str(
        "struct G4wMkParams {\n    m: u32,\n    x_stride_words: u32,\n    y_stride_words: u32,\n    dst_word_off: u32,\n};\n\n",
    );
    b.push_str("@group(0) @binding(31) var<storage, read_write> g4w_y_q: array<u32>;\n");
    b.push_str("@group(0) @binding(32) var<storage, read_write> g4w_y_k: array<u32>;\n");
    b.push_str("@group(0) @binding(33) var<storage, read_write> g4w_y_v: array<u32>;\n");
    b.push_str("@group(0) @binding(34) var<uniform> g4w_split_params: G4wSplitParams;\n");
    b.push_str("@group(0) @binding(35) var<uniform> g4w_mk_params: G4wMkParams;\n\n");
    b.push_str(
        "fn g4w_pair_word(tid: u32, total: f32, hi_live: bool) -> u32 {\n    let lo = bf16_encode(total) & 0xffffu;\n    let hi = bf16_encode(gemv_bf16_partial[tid + GEMV_BF16_LANES]) & 0xffffu;\n    return lo | (select(0u, hi, hi_live) << 16u);\n}\n\n",
    );
    let pk_store = |t: usize| {
        format!(
            "            gemv_bf16_y[g4w_mk_params.dst_word_off + {t}u * g4w_mk_params.y_stride_words + (row >> 1u)] = word;\n"
        )
    };
    let pk3_store = |t: usize| {
        format!(
            concat!(
                "            if (row < g4w_split_params.q_rows) {{\n",
                "                g4w_y_q[{t}u * (g4w_split_params.q_rows >> 1u) + (row >> 1u)] = word;\n",
                "            }} else {{\n",
                "                let kr = row - g4w_split_params.q_rows;\n",
                "                if (kr < g4w_split_params.kv_rows) {{\n",
                "                    g4w_y_k[{t}u * (g4w_split_params.kv_rows >> 1u) + (kr >> 1u)] = word;\n",
                "                }}\n",
                "                if (row >= g4w_split_params.v_off) {{\n",
                "                    let vr = row - g4w_split_params.v_off;\n",
                "                    if (vr < g4w_split_params.kv_rows) {{\n",
                "                        g4w_y_v[{t}u * (g4w_split_params.kv_rows >> 1u) + (vr >> 1u)] = word;\n",
                "                    }}\n",
                "                }}\n",
                "            }}\n",
            ),
            t = t
        )
    };
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
        b.push_str("    let row = wid.x * GEMV_BF16_ROWS + warp;\n");
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
            let x_index = if t == 0 {
                "xo + j".to_string()
            } else if t == 1 {
                "xs + xo + j".to_string()
            } else {
                format!("{t}u * xs + xo + j")
            };
            writeln!(
                b,
                "            if ({t}u < mm) {{ let xw = gemv_bf16_x[{x_index}]; acc{t} = acc{t} + (wl * bf16_lo(xw) + wh * bf16_hi(xw)); }}"
            )
            .unwrap();
        }
        b.push_str("        }\n    }\n");
        for t in 0..mk_max {
            writeln!(b, "    if ({t}u < mm) {{").unwrap();
            writeln!(
                b,
                "        let total = gemv_bf16_reduce(tid, lane, acc{t});"
            )
            .unwrap();
            b.push_str("        if (lane == 0u && live && (warp & 1u) == 0u) {\n");
            b.push_str(
                "            let word = g4w_pair_word(tid, total, row + 1u < gemv_bf16_params.n_rows);\n",
            );
            b.push_str(&store(t));
            b.push_str("        }\n");
            b.push_str("        workgroupBarrier();\n");
            b.push_str("    }\n");
        }
        b.push_str("}\n\n");
    }
    b
}

fn w4_mk_unrolled_source(mk_max: usize) -> String {
    use std::fmt::Write as _;
    let mut b = String::new();
    b.push_str(
        "struct G4wSplitParams {\n    q_rows: u32,\n    kv_rows: u32,\n    v_off: u32,\n    pad0: u32,\n};\n\n",
    );
    b.push_str(
        "struct G4wMkParams {\n    m: u32,\n    x_stride_words: u32,\n    y_stride_words: u32,\n    dst_word_off: u32,\n};\n\n",
    );
    b.push_str("@group(0) @binding(31) var<storage, read_write> g4w_y_q: array<u32>;\n");
    b.push_str("@group(0) @binding(32) var<storage, read_write> g4w_y_k: array<u32>;\n");
    b.push_str("@group(0) @binding(33) var<storage, read_write> g4w_y_v: array<u32>;\n");
    b.push_str("@group(0) @binding(34) var<uniform> g4w_split_params: G4wSplitParams;\n");
    b.push_str("@group(0) @binding(35) var<uniform> g4w_mk_params: G4wMkParams;\n\n");
    b.push_str(
        "fn g4w_w4_pair_word(tid: u32, total: f32, hi_live: bool) -> u32 {\n    let lo = bf16_encode(total) & 0xffffu;\n    let hi = bf16_encode(w4a16_partial[tid + W4A16_LANES]) & 0xffffu;\n    return lo | (select(0u, hi, hi_live) << 16u);\n}\n\n",
    );
    b.push_str(
        "fn w4a16_dot8_xoff(pv: u32, kb: u32, xoff_words: u32, acc_in: f32) -> f32 {\n    var acc = acc_in;\n    let xb = (kb >> 1u) + xoff_words;\n    for (var i = 0u; i < 4u; i = i + 1u) {\n        let xp = w4a16_x_pair(xb + i);\n        acc = fma(w4a16_q(pv, 2u * i), xp.x, acc);\n        acc = fma(w4a16_q(pv, 2u * i + 1u), xp.y, acc);\n    }\n    return acc;\n}\n\n",
    );
    let xoff = |t: usize| {
        if t == 0 {
            "0u".to_string()
        } else {
            format!("{t}u * xs")
        }
    };
    let block_rows = |b: &mut String| {
        b.push_str("    let kv = select(0u, w4a16_params.k_elems >> 5u, live);\n");
        b.push_str("    let wbase = select(0u, row * w4a16_params.w_row_words, live);\n");
        b.push_str("    let sbase = select(0u, row * w4a16_params.scale_row_stride, live);\n");
        b.push_str("    let gs = w4a16_params.gs;\n");
        b.push_str("    let mm = g4w_mk_params.m;\n");
        b.push_str("    let xs = g4w_mk_params.x_stride_words;\n");
        for t in 0..mk_max {
            writeln!(b, "    var acc{t} = 0.0;").unwrap();
        }
        b.push_str("    if (gs >= 32u) {\n");
        b.push_str("        for (var v = lane; v < kv; v = v + W4A16_LANES) {\n");
        b.push_str("            let kbase = v * 32u;\n");
        b.push_str("            let sc = w4a16_scale_at(sbase, kbase / gs);\n");
        for t in 0..mk_max {
            writeln!(b, "            var blk{t} = 0.0;").unwrap();
        }
        b.push_str("            for (var j = 0u; j < 4u; j = j + 1u) {\n");
        b.push_str("                let pv = w4a16_packed[wbase + v * 4u + j];\n");
        b.push_str("                let kb = kbase + j * 8u;\n");
        for t in 0..mk_max {
            writeln!(
                b,
                "                if ({t}u < mm) {{ blk{t} = w4a16_dot8_xoff(pv, kb, {}, blk{t}); }}",
                xoff(t)
            )
            .unwrap();
        }
        b.push_str("            }\n");
        for t in 0..mk_max {
            writeln!(
                b,
                "            if ({t}u < mm) {{ acc{t} = fma(sc, blk{t}, acc{t}); }}"
            )
            .unwrap();
        }
        b.push_str("        }\n");
        b.push_str("    } else {\n");
        b.push_str("        for (var v = lane; v < kv; v = v + W4A16_LANES) {\n");
        b.push_str("            let kbase = v * 32u;\n");
        b.push_str("            for (var j = 0u; j < 4u; j = j + 1u) {\n");
        b.push_str("                let pv = w4a16_packed[wbase + v * 4u + j];\n");
        b.push_str("                let kb = kbase + j * 8u;\n");
        b.push_str("                let sc = w4a16_scale_at(sbase, kb / gs);\n");
        for t in 0..mk_max {
            writeln!(
                b,
                "                if ({t}u < mm) {{ let a{t} = w4a16_dot8_xoff(pv, kb, {}, 0.0); acc{t} = fma(a{t}, sc, acc{t}); }}",
                xoff(t)
            )
            .unwrap();
        }
        b.push_str("            }\n");
        b.push_str("        }\n");
        b.push_str("    }\n");
    };
    let v4_rows = |b: &mut String| {
        b.push_str("    let kv = select(0u, w4a16_params.k_elems >> 5u, live);\n");
        b.push_str("    let wbase4 = select(0u, row * (w4a16_params.w_row_words >> 2u), live);\n");
        b.push_str("    let sbase = select(0u, row * w4a16_params.scale_row_stride, live);\n");
        b.push_str("    let gs = w4a16_params.gs;\n");
        b.push_str("    let mm = g4w_mk_params.m;\n");
        b.push_str("    let xs4 = g4w_mk_params.x_stride_words >> 2u;\n");
        for t in 0..mk_max {
            writeln!(b, "    var acc{t} = 0.0;").unwrap();
        }
        b.push_str("    for (var v = lane; v < kv; v = v + W4A16_LANES) {\n");
        b.push_str("        let sc = w4a16_scale_at(sbase, (v << 5u) / gs);\n");
        b.push_str("        let wv = w4a16_packed4[wbase4 + v];\n");
        for t in 0..mk_max {
            let xb = if t == 0 {
                "(v << 2u)".to_string()
            } else {
                format!("{t}u * xs4 + (v << 2u)")
            };
            writeln!(
                b,
                "        if ({t}u < mm) {{ acc{t} = fma(sc, w4a16_dot32_v4(wv, {xb}), acc{t}); }}"
            )
            .unwrap();
        }
        b.push_str("    }\n");
    };
    let pk_store = |t: usize| {
        format!(
            "                w4a16_y[g4w_mk_params.dst_word_off + {t}u * g4w_mk_params.y_stride_words + (row >> 1u)] = word;\n"
        )
    };
    let pk3_store = |t: usize| {
        format!(
            concat!(
                "                if (row < g4w_split_params.q_rows) {{\n",
                "                    g4w_y_q[{t}u * (g4w_split_params.q_rows >> 1u) + (row >> 1u)] = word;\n",
                "                }} else {{\n",
                "                    let kr = row - g4w_split_params.q_rows;\n",
                "                    if (kr < g4w_split_params.kv_rows) {{\n",
                "                        g4w_y_k[{t}u * (g4w_split_params.kv_rows >> 1u) + (kr >> 1u)] = word;\n",
                "                    }}\n",
                "                    if (row >= g4w_split_params.v_off) {{\n",
                "                        let vr = row - g4w_split_params.v_off;\n",
                "                        if (vr < g4w_split_params.kv_rows) {{\n",
                "                            g4w_y_v[{t}u * (g4w_split_params.kv_rows >> 1u) + (vr >> 1u)] = word;\n",
                "                        }}\n",
                "                    }}\n",
                "                }}\n",
            ),
            t = t
        )
    };
    let rows_fns: [(&str, &dyn Fn(&mut String)); 2] = [
        ("block", &block_rows as &dyn Fn(&mut String)),
        ("v4", &v4_rows),
    ];
    let stores: [(&str, &dyn Fn(usize) -> String); 2] = [
        ("pk", &pk_store as &dyn Fn(usize) -> String),
        ("pk3", &pk3_store),
    ];
    for (rows_tag, rows) in rows_fns {
        for (store_tag, store) in stores {
            b.push_str("@compute @workgroup_size(256)\n");
            writeln!(b, "fn g4w_gemm_w4a16_{rows_tag}_mk_{store_tag}(").unwrap();
            b.push_str("    @builtin(workgroup_id) wid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>\n) {\n");
            b.push_str("    let tid = lid.x;\n");
            b.push_str("    let lane = tid & (W4A16_LANES - 1u);\n");
            b.push_str("    let warp = tid / W4A16_LANES;\n");
            b.push_str("    let row = wid.x * W4A16_ROWS + warp;\n");
            b.push_str("    let live = row < w4a16_params.n_rows;\n");
            rows(&mut b);
            for t in 0..mk_max {
                writeln!(b, "    if ({t}u < mm) {{").unwrap();
                writeln!(
                    b,
                    "        let total = w4a16_lane_reduce(tid, lane, acc{t});"
                )
                .unwrap();
                b.push_str("        if (lane == 0u && live && (warp & 1u) == 0u) {\n");
                b.push_str("            let word = g4w_w4_pair_word(tid, total, row + 1u < w4a16_params.n_rows);\n");
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

fn build_pipelines(
    ctx: &WgpuContext,
    sg16: bool,
    w4_grain: wk::gemv_w4a16::ScaleGrain,
    lmhead_i8: bool,
    lmhead_sg: bool,
    prefill_m: usize,
    verify_m: Option<usize>,
    fuse_norms: bool,
    fuse_attn: bool,
    hidden: usize,
    flash1_hds: &[u32],
    gqa_group: usize,
) -> Result<Pipelines> {
    let raw_nozi =
        |label: &str, source: &str, entry: &str| raw_nozi_pipeline(ctx, label, source, entry);
    let mk = |label: &str, source: &str, entry: &str| nozi_all_pipeline(ctx, label, source, entry);
    let mk_nozi = |label: &str, source: &str, entry: &str| {
        if flash_nozi_enabled() {
            return raw_nozi(label, source, entry);
        }
        mk(label, source, entry)
    };
    let src_rs = compose(wk::residual_scale::WGSL);
    let src_rms = compose(wk::rmsnorm::WGSL);
    let src_rmsres = compose(wk::rmsnorm_residual::WGSL);
    let src_rope = format!("{}\n{}", compose(wk::rope_bf16::WGSL), ROPE_F32_WGSL);
    let src_kv = compose(wk::kv_fp8::WGSL);
    let src_flash = format!("{}\n{}", compose(wk::flash_decode::WGSL), flash2_pk_wgsl());
    let fold_flash2 = |splits: u32| -> Result<String> {
        let folded = src_flash.replacen(
            "let splits = 16u;",
            &format!("let splits = {splits}u;"),
            1,
        );
        anyhow::ensure!(
            splits == 16 || folded != src_flash,
            "flash2 split-count anchor `let splits = 16u;` missing from g4shared_flash2_pk; \
             the folded stage2 would silently merge the wrong number of partials"
        );
        Ok(folded)
    };
    let src_flash2_folded = fold_flash2(decode_splits())?;
    let src_gemvb = format!("{}\n{}", compose(wk::gemv_bf16::WGSL), gemv_pk_wgsl());
    let src_gemvw4 = format!("{}\n{}", compose(wk::gemv_w4a16::WGSL), GEMV_W4_PK_WGSL);
    let src_gd = compose(wk::graph_decode::WGSL);
    let src_gelu = wk::gelu_tanh_mul::source();
    let src_gather = compose(GATHER_WGSL);
    let src_ax = compose(AXPBY_WGSL);
    let src_gm = compose(GATEMUL_WGSL);
    let (gemv_w4_sg_pk, gemv_w4_sg_pk3) = if sg16 {
        let src_sg = wk::gemv_w4a16::sg_pk_source_grain(w4_grain);
        (
            Some(mk(
                "e4bw-gemv-w4-sg-pk",
                &src_sg,
                wk::gemv_w4a16::SG_PK_ENTRY,
            )?),
            Some(mk(
                "e4bw-gemv-w4-sg-pk3",
                &src_sg,
                wk::gemv_w4a16::SG_PK3_ENTRY,
            )?),
        )
    } else {
        (None, None)
    };
    let (gemv_w4_sg_pkm, gemv_w4_sg_pkm3) = if sg16 && w4_mr() > 1 {
        let src_sgm = wk::gemv_w4a16::sg_pk_mr_source_grain(w4_mr(), w4_grain);
        (
            Some(mk(
                "e4bw-gemv-w4-sg-pkm",
                &src_sgm,
                wk::gemv_w4a16::SG_PKM_ENTRY,
            )?),
            Some(mk(
                "e4bw-gemv-w4-sg-pkm3",
                &src_sgm,
                wk::gemv_w4a16::SG_PKM3_ENTRY,
            )?),
        )
    } else {
        (None, None)
    };
    let fnc = if fuse_norms {
        let want = fnc_unroll_mask();
        let fast = want
            .iter()
            .any(|w| *w)
            .then(|| fnc_unrolled_source(hidden))
            .flatten();
        let src_fnc = compose(wk::fused_norm_chain::WGSL);
        let one = |i: usize,
                   rolled_entry: &str,
                   fast_entry: &str|
         -> Result<Arc<wgpu::ComputePipeline>> {
            match fast.as_ref().filter(|_| want[i]) {
                Some(body) => mk(fast_entry, &compose(body), fast_entry),
                None => mk(rolled_entry, &src_fnc, rolled_entry),
            }
        };
        Some(FncPipelines {
            a: one(0, wk::fused_norm_chain::ENTRY_RMS_RES_RMS, FNCU_ENTRY_A)?,
            b: one(1, wk::fused_norm_chain::ENTRY_RES_OF_RMS, FNCU_ENTRY_B)?,
            c: one(
                2,
                wk::fused_norm_chain::ENTRY_RMS_RES_RMS_NEXT,
                FNCU_ENTRY_C,
            )?,
            unrolled: if fast.is_some() { want } else { [false; 3] },
        })
    } else {
        None
    };
    let fac = if fuse_attn {
        let src_fac = compose(wk::fused_attn_chain::WGSL);
        Some(FacPipelines {
            q: mk("e4bw-fac-q", &src_fac, wk::fused_attn_chain::ENTRY_Q)?,
            k: mk("e4bw-fac-k", &src_fac, wk::fused_attn_chain::ENTRY_K)?,
            v: mk("e4bw-fac-v", &src_fac, wk::fused_attn_chain::ENTRY_V)?,
        })
    } else {
        None
    };
    let lmhead_i8_pl = if lmhead_i8 {
        Some(mk(
            "e4bw-lmhead-i8",
            &compose(LMHEAD_I8_WGSL),
            "e4b_lmhead_i8_pk",
        )?)
    } else {
        None
    };
    let lmhead_sg_pl = if lmhead_sg {
        let (entry, _) = wk::gemv_bf16::sg_pk_entry(lmhead_sg_wg());
        Some(mk(
            "e4bw-lmhead-sg-pk",
            &wk::gemv_bf16::sg_pk_source(),
            entry,
        )?)
    } else {
        None
    };
    let sd = flash_sd_enabled();
    let build_mk = |mk_rows: usize| -> Result<MkPipelines> {
        let src_gemvb_mk = format!(
            "{}\n{}",
            compose(wk::gemv_bf16::WGSL),
            if mk_unroll_enabled() {
                mk_unrolled_source(mk_rows)
            } else {
                mk_widen(&gemv_pk_wgsl(), mk_rows)
            }
        );
        let src_gemvw4_mk = format!(
            "{}\n{}",
            compose(wk::gemv_w4a16::WGSL),
            if mk_unroll_enabled() {
                w4_mk_unrolled_source(mk_rows)
            } else {
                mk_widen(GEMV_W4_PK_WGSL, mk_rows)
            }
        );
        let (gemm_w4_sg_pk, gemm_w4_sg_pk3) = if sg16 {
            let src_sg_mk = if mk_unroll_enabled() {
                wk::gemv_w4a16::sg_mk_unrolled_source_grain(mk_rows as u32, w4_grain)
            } else {
                wk::gemv_w4a16::sg_mk_source_grain(mk_rows as u32, w4_grain)
            };
            (
                Some(mk(
                    "e4bw-gemm-w4-sg-mk-pk",
                    &src_sg_mk,
                    wk::gemv_w4a16::SG_MK_PK_ENTRY,
                )?),
                Some(mk(
                    "e4bw-gemm-w4-sg-mk-pk3",
                    &src_sg_mk,
                    wk::gemv_w4a16::SG_MK_PK3_ENTRY,
                )?),
            )
        } else {
            (None, None)
        };
        let (gemm_i8_pk, gemm_i8_pk3, gemm_i8g_pk, gemm_i8g_pk3) = if w8_enabled() {
            let src_i8_mk = format!(
                "{}\n{}",
                compose(wk::gemv_bf16::WGSL),
                mk_unrolled_i8_source(mk_rows)
            );
            let (g_pk, g_pk3) = if w8_group() > 0 {
                (
                    Some(mk("e4bw-gemm-i8g-mk-pk", &src_i8_mk, "g4w_gemm_i8g_mk_pk")?),
                    Some(mk(
                        "e4bw-gemm-i8g-mk-pk3",
                        &src_i8_mk,
                        "g4w_gemm_i8g_mk_pk3",
                    )?),
                )
            } else {
                (None, None)
            };
            (
                Some(mk("e4bw-gemm-i8-mk-pk", &src_i8_mk, "g4w_gemm_i8_mk_pk")?),
                Some(mk("e4bw-gemm-i8-mk-pk3", &src_i8_mk, "g4w_gemm_i8_mk_pk3")?),
                g_pk,
                g_pk3,
            )
        } else {
            (None, None, None, None)
        };
        Ok(MkPipelines {
            rows: mk_rows,
            gather: mk("e4bw-gather-mk", &src_gather, "e4b_gather_chunks_mk")?,
            gemm_bf16_pk: mk("e4bw-gemm-bf16-mk-pk", &src_gemvb_mk, "g4w_gemm_bf16_mk_pk")?,
            gemm_bf16_pk3: mk(
                "e4bw-gemm-bf16-mk-pk3",
                &src_gemvb_mk,
                "g4w_gemm_bf16_mk_pk3",
            )?,
            gemm_i8_pk,
            gemm_i8_pk3,
            gemm_i8g_pk,
            gemm_i8g_pk3,
            gemm_w4_pk: mk(
                "e4bw-gemm-w4-mk-pk",
                &src_gemvw4_mk,
                "g4w_gemm_w4a16_block_mk_pk",
            )?,
            gemm_w4_pk3: mk(
                "e4bw-gemm-w4-mk-pk3",
                &src_gemvw4_mk,
                "g4w_gemm_w4a16_block_mk_pk3",
            )?,
            gemm_w4_v4_pk: mk(
                "e4bw-gemm-w4-v4-mk-pk",
                &src_gemvw4_mk,
                "g4w_gemm_w4a16_v4_mk_pk",
            )?,
            gemm_w4_v4_pk3: mk(
                "e4bw-gemm-w4-v4-mk-pk3",
                &src_gemvw4_mk,
                "g4w_gemm_w4a16_v4_mk_pk3",
            )?,
            gemm_w4_sg_pk,
            gemm_w4_sg_pk3,
            flash1: if sd {
                let src = format!("{}\n{}", src_flash, flash_rows_stage1_source_sd());
                mk("e4bw-flash-rows1-sd", &src, FLASH_ROWS_STAGE1_ENTRY_SD)?
            } else {
                mk("e4bw-flash-rows1", &src_flash, FLASH_ROWS_STAGE1_ENTRY)?
            },
            flash1_qtile: match prefill_qtile() {
                1 => None,
                tile => {
                    let hdq = *flash1_hds
                        .iter()
                        .max()
                        .expect("flash1 needs at least one head_dim");
                    let (qsrc, qentry) = if sd {
                        (
                            flash1_mk_qtile_source_sd(hdq, tile),
                            flash1_mk_qtile_entry_sd(hdq, tile),
                        )
                    } else {
                        (
                            flash1_mk_qtile_source(hdq, tile),
                            flash1_mk_qtile_entry(hdq, tile),
                        )
                    };
                    let src = format!("{}\n{}", src_flash, qsrc);
                    Some(mk("e4bw-flash-rows1-qtile", &src, &qentry)?)
                }
            },
            flash2_pk: mk(
                "e4bw-flash-rows2-pk",
                &src_flash,
                "g4w_flash_rows_stage2_pk",
            )?,
            gatemul: mk("e4bw-gatemul-mk", &src_gm, "e4b_gate_mul_bf16_mk")?,
        })
    };
    let mk_rows = prefill_m.min(PREFILL_SLAB);
    let mk_pl = if prefill_m > 0 {
        Some(build_mk(mk_rows)?)
    } else {
        None
    };
    let sg = flash1_sg_enabled(ctx);
    let widest = *flash1_hds
        .iter()
        .max()
        .expect("flash1 needs at least one head_dim");
    assert!(
        widest as usize <= wk::flash_decode::MAX_HEAD_DIM,
        "flash1 head_dim {widest} exceeds the shared cache geometry {}",
        wk::flash_decode::MAX_HEAD_DIM
    );
    let mut hds: Vec<u32> = if flash1_hd_specialize_enabled() {
        let mut v = flash1_hds.to_vec();
        v.sort_unstable();
        v.dedup();
        v
    } else {
        vec![widest]
    };
    if hds.last() != Some(&widest) {
        hds.push(widest);
    }
    let mut flash1: Vec<(u32, Arc<wgpu::ComputePipeline>)> = Vec::with_capacity(hds.len());
    for hd in hds {
        let (entry, gen) = if sd {
            (flash1_e4b_entry_sd(hd, sg), flash1_e4b_source_sd(hd, sg))
        } else {
            (flash1_e4b_entry(hd, sg), flash1_e4b_source(hd, sg))
        };
        let src = format!("{}\n{}", compose(wk::flash_decode::WGSL), gen);
        flash1.push((hd, mk_nozi(&format!("e4bw-flash1-{hd}"), &src, &entry)?));
    }
    let gqa_fold = wk::flash_decode::gqa_fold_env(gqa_group) as u32;
    let mut flash1_fold: Vec<(u32, Arc<wgpu::ComputePipeline>)> = Vec::new();
    if gqa_fold > 1 {
        for (hd, _) in &flash1 {
            let (entry, gen) = if sd {
                (
                    wk::flash_decode::fold_stage1_entry_sd(*hd, sg, gqa_fold),
                    wk::flash_decode::fold_stage1_source_sd(*hd, sg, gqa_fold),
                )
            } else {
                (
                    wk::flash_decode::fold_stage1_entry(*hd, sg, gqa_fold),
                    wk::flash_decode::fold_stage1_source(*hd, sg, gqa_fold),
                )
            };
            let src = format!("{}\n{}", compose(wk::flash_decode::WGSL), gen);
            flash1_fold.push((
                *hd,
                mk_nozi(&format!("e4bw-flash1-fold{gqa_fold}-{hd}"), &src, &entry)?,
            ));
        }
    }
    let tpw = deep_tpw_env();
    let deep_fold = match deep_fold_env(gqa_group as u32) {
        0 if tpw > 1 => gqa_fold.max(1),
        f => f,
    };
    let mut flash1_fold_deep: Vec<(u32, Arc<wgpu::ComputePipeline>)> = Vec::new();
    if deep_fold > 1 || tpw > 1 {
        assert!(
            sg,
            "{E4B_DEEP_FOLD_ENV}/{E4B_DEEP_TPW_ENV} need 32-lane subgroups; only the subgroup \
             fold reduce is wired for the deep split arm"
        );
        for (hd, _) in &flash1 {
            let (entry, gen) = if tpw > 1 {
                (
                    deep_tpw_entry(*hd, deep_fold.max(1), tpw, sd),
                    deep_tpw_source(*hd, deep_fold.max(1), tpw, sd),
                )
            } else if sd {
                (
                    wk::flash_decode::fold_stage1_entry_sd(*hd, sg, deep_fold),
                    wk::flash_decode::fold_stage1_source_sd(*hd, sg, deep_fold),
                )
            } else {
                (
                    wk::flash_decode::fold_stage1_entry(*hd, sg, deep_fold),
                    wk::flash_decode::fold_stage1_source(*hd, sg, deep_fold),
                )
            };
            let src = format!("{}\n{}", compose(wk::flash_decode::WGSL), gen);
            flash1_fold_deep.push((
                *hd,
                mk_nozi(&format!("e4bw-flash1-dfold{deep_fold}t{tpw}-{hd}"), &src, &entry)?,
            ));
        }
        eprintln!(
            "[gemma4_e4b_wgpu] deep split arm stage1: fold {deep_fold} query heads, {tpw} \
             token(s) per warp on full-attention layers ({E4B_DEEP_FOLD_ENV}, {E4B_DEEP_TPW_ENV})"
        );
    }
    let deep_fold = if deep_fold > 1 || tpw > 1 {
        deep_fold.max(1)
    } else {
        0
    };
    let flash1_entry = if sd {
        flash1_e4b_entry_sd(widest, sg)
    } else {
        flash1_e4b_entry(widest, sg)
    };
    eprintln!(
        "[gemma4_e4b_wgpu] flash stage1: {} pipeline(s) at head_dim {:?}, reduce {}, gqa fold {}, \
         decode {}",
        flash1.len(),
        flash1.iter().map(|(h, _)| *h).collect::<Vec<_>>(),
        if sg { "subgroup" } else { "workgroup-barrier" },
        gqa_fold,
        if sd { "e4m3-shift" } else { "e4m3-exact" }
    );
    let mk_verify_pl = match verify_m {
        Some(v) if prefill_m > 0 && v < mk_rows => Some(build_mk(v)?),
        _ => None,
    };
    Ok(Pipelines {
        w4_grain,
        gather: mk("e4bw-gather", &src_gather, "e4b_gather_chunks")?,
        scale: mk("e4bw-scale", &src_rs, "scale_bf16")?,
        rms: mk("e4bw-rms", &src_rms, "rmsnorm_bf16")?,
        rmsres: mk("e4bw-rmsres", &src_rmsres, "rmsnorm_residual_bf16")?,
        resadd: mk("e4bw-resadd", &src_rs, "residual_add_scale_bf16")?,
        cast_f32: mk("e4bw-cast", &src_rs, "cast_bf16_to_f32")?,
        softcap: mk("e4bw-softcap", &src_rs, "tanh_softcap_bf16_to_f32")?,
        rope: mk("e4bw-rope", &src_rope, "rope_bf16")?,
        rope_f32: mk("e4bw-rope-f32", &src_rope, ROPE_F32_ENTRY)?,
        kvq: mk("e4bw-kvq", &src_kv, wk::kv_fp8::QUANTIZE_ENTRY)?,
        flash1,
        flash1_fold,
        flash1_fold_deep,
        gqa_fold,
        deep_fold,
        flash1_entry,
        flash2_pk: mk("e4bw-flash2-pk", &src_flash2_folded, FLASH2_PK_ENTRY)?,
        flash2_pk_deep: match deep_split_arm() {
            Some((_, splits)) if deep_stage2_legacy16() => Some(mk(
                "e4bw-flash2-pk-deep",
                &fold_flash2(splits)?,
                FLASH2_PK_ENTRY,
            )?),
            Some((_, splits)) => Some(mk(
                "e4bw-flash2-pk-deep",
                &format!("{src_flash}\n{}", e4b_flash2_deep_source(splits)),
                FLASH2_DEEP_ENTRY,
            )?),
            None => None,
        },
        gemv_i8_pk: if w8_enabled() {
            let src_i8 = format!("{}\n{}", compose(wk::gemv_bf16::WGSL), gemv_i8_pk_source());
            Some(mk("e4bw-gemv-i8-pk", &src_i8, "g4w_gemv_i8_pk")?)
        } else {
            None
        },
        gemv_i8_pk3: if w8_enabled() {
            let src_i8 = format!("{}\n{}", compose(wk::gemv_bf16::WGSL), gemv_i8_pk_source());
            Some(mk("e4bw-gemv-i8-pk3", &src_i8, "g4w_gemv_i8_pk3")?)
        } else {
            None
        },
        gemv_i8g_pk: if w8_enabled() && w8_group() > 0 {
            let src_i8 = format!("{}\n{}", compose(wk::gemv_bf16::WGSL), gemv_i8_pk_source());
            Some(mk("e4bw-gemv-i8g-pk", &src_i8, "g4w_gemv_i8g_pk")?)
        } else {
            None
        },
        gemv_i8g_pk3: if w8_enabled() && w8_group() > 0 {
            let src_i8 = format!("{}\n{}", compose(wk::gemv_bf16::WGSL), gemv_i8_pk_source());
            Some(mk("e4bw-gemv-i8g-pk3", &src_i8, "g4w_gemv_i8g_pk3")?)
        } else {
            None
        },
        gemv_pk: mk("e4bw-gemv8-pk", &src_gemvb, GEMV_PK_ENTRY)?,
        gemv_pk3: mk("e4bw-gemv8-pk3", &src_gemvb, GEMV_PK3_ENTRY)?,
        gemv_w4_pk: mk("e4bw-gemv-w4-pk", &src_gemvw4, "g4w_gemv_w4a16_block_pk")?,
        gemv_w4_pk3: mk("e4bw-gemv-w4-pk3", &src_gemvw4, "g4w_gemv_w4a16_block_pk3")?,
        gemv_w4_v4_pk: mk("e4bw-gemv-w4-v4-pk", &src_gemvw4, "g4w_gemv_w4a16_v4_pk")?,
        gemv_w4_v4_pk3: mk("e4bw-gemv-w4-v4-pk3", &src_gemvw4, "g4w_gemv_w4a16_v4_pk3")?,
        gemv_w4_sg_pk,
        gemv_w4_sg_pk3,
        gemv_w4_sg_pkm,
        gemv_w4_sg_pkm3,
        lmhead_sg: lmhead_sg_pl,
        lmhead_i8: lmhead_i8_pl,
        gelu_even: mk("e4bw-gelu", &src_gelu, wk::gelu_tanh_mul::ENTRY_FUSED_EVEN)?,
        axpby: mk("e4bw-axpby", &src_ax, "e4b_axpby_bf16")?,
        gatemul: mk("e4bw-gatemul", &src_gm, "e4b_gate_mul_bf16")?,
        am1: mk("e4bw-am1", &src_gd, "argmax_f32_rows_stage1")?,
        am2: mk("e4bw-am2", &src_gd, "argmax_f32_rows_stage2")?,
        mk: mk_pl,
        mk_verify: mk_verify_pl,
        fnc,
        fac,
    })
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

struct Builder<'a> {
    ctx: &'static WgpuContext,
    pl: &'a Pipelines,
    passes: Vec<Pass>,
    prefill_passes: Vec<Pass>,
    verify_prefill_passes: Vec<Pass>,
    to_prefill: bool,
    to_verify: bool,
    keep: Vec<Box<dyn std::any::Any>>,
    weight_bytes: u64,
    sg16: bool,

    w4_census: [usize; 3],

    mk_prefill_unis: Vec<(wgpu::Buffer, MkParams, usize)>,
    mk_verify_unis: Vec<(wgpu::Buffer, MkParams, usize)>,
    deep_attn: Vec<(usize, Pass)>,
}

impl Builder<'_> {
    fn push(
        &mut self,
        pipeline: Arc<wgpu::ComputePipeline>,
        binds: &[(u32, &wgpu::Buffer)],
        grid: (u32, u32, u32),
    ) {
        let bind = dispatch::bind_group(self.ctx, &pipeline, binds);
        let (bound_bytes, widest_bytes) = bind_bytes(binds.iter().map(|(_, b)| *b));
        let pass = Pass {
            pipeline,
            bind,
            grid,
            bound_bytes,
            widest_bytes,
        };
        if self.to_verify {
            self.verify_prefill_passes.push(pass);
        } else if self.to_prefill {
            self.prefill_passes.push(pass);
        } else {
            self.passes.push(pass);
        }
    }

    fn push_deep_decode_attn(
        &mut self,
        s1_binds: &[(u32, &wgpu::Buffer)],
        s2_binds: &[(u32, &wgpu::Buffer)],
        n_q: u32,
        hd: usize,
        full_attn: bool,
    ) {
        let Some(deep_pl) = self.pl.flash2_pk_deep.clone() else {
            return;
        };
        let (_, deep) = deep_split_arm()
            .expect("flash2_pk_deep was built, so deep_split_arm() must still resolve");
        assert!(
            !self.to_prefill && !self.to_verify,
            "deep split-arm twins are decode-only; the m>1 prefill/verify flash keeps \
             FLASH_SPLITS"
        );
        let i2 = self.passes.len() - 1;
        let i1 = i2 - 1;
        if !full_attn && deep_full_only() {
            self.deep_attn.push((i1, self.passes[i1].clone()));
            self.deep_attn.push((i2, self.passes[i2].clone()));
            return;
        }
        let mut s1 = self.passes[i1].clone();
        assert_eq!(
            s1.grid.1,
            decode_splits(),
            "deep twin must shadow the decode flash stage1 dispatch; {DEEP_SPLIT_ARM_RULE}"
        );
        s1.grid.1 = deep;
        if full_attn && !self.pl.flash1_fold_deep.is_empty() {
            let fpl = flash1_pick(&self.pl.flash1_fold_deep, hd);
            s1.bind = dispatch::bind_group(self.ctx, &fpl, s1_binds);
            s1.pipeline = fpl;
            s1.grid.0 = n_q / self.pl.deep_fold.max(1);
        }
        let bind = dispatch::bind_group(self.ctx, &deep_pl, s2_binds);
        let (bound_bytes, widest_bytes) = bind_bytes(s2_binds.iter().map(|(_, b)| *b));
        let s2 = Pass {
            pipeline: deep_pl,
            bind,
            grid: (n_q, 1, 1),
            bound_bytes,
            widest_bytes,
        };
        self.deep_attn.push((i1, s1));
        self.deep_attn.push((i2, s2));
    }

    fn push_off(
        &mut self,
        pipeline: Arc<wgpu::ComputePipeline>,
        binds: &[(u32, &wgpu::Buffer, u64)],
        grid: (u32, u32, u32),
    ) {
        let bind = dispatch::bind_group_offsets(self.ctx, &pipeline, binds);
        let (bound_bytes, widest_bytes) = bind_bytes(binds.iter().map(|(_, b, _)| *b));
        let pass = Pass {
            pipeline,
            bind,
            grid,
            bound_bytes,
            widest_bytes,
        };
        if self.to_verify {
            self.verify_prefill_passes.push(pass);
        } else if self.to_prefill {
            self.prefill_passes.push(pass);
        } else {
            self.passes.push(pass);
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

    fn wants_verify_list(&self) -> bool {
        na_route_enabled() || na_attn_enabled() || self.pl.mk_verify.is_some()
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
            "e4bw-rms-p",
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

    fn fnc_uni(&mut self, batch: usize, hidden: usize, eps: f32, scale: f32) -> wgpu::Buffer {
        self.uni(
            "e4bw-fnc-p",
            FncParams {
                hidden: hidden as u32,
                batch: batch as u32,
                eps,
                words_per_row: (hidden / 2) as u32,
                scale,
                ..Default::default()
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn fnc_a(
        &mut self,
        x: &wgpu::Buffer,
        res: &wgpu::Buffer,
        w1: &wgpu::Buffer,
        w2: &wgpu::Buffer,
        out: &wgpu::Buffer,
        batch: usize,
        hidden: usize,
        eps: f32,
    ) {
        let p = self.fnc_uni(batch, hidden, eps, 1.0);
        let grid = self.grid_1d(batch as u64, 1);
        let pipeline = self.pl.fnc.as_ref().expect("fnc pipelines").a.clone();
        self.push(
            pipeline,
            &[(0, x), (1, res), (2, w1), (3, w2), (4, out), (5, &p)],
            grid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn fnc_b(
        &mut self,
        x: &wgpu::Buffer,
        res: &wgpu::Buffer,
        w: &wgpu::Buffer,
        out: &wgpu::Buffer,
        batch: usize,
        hidden: usize,
        eps: f32,
        scale: f32,
    ) {
        let p = self.fnc_uni(batch, hidden, eps, scale);
        let grid = self.grid_1d(batch as u64, 1);
        let pipeline = self.pl.fnc.as_ref().expect("fnc pipelines").b.clone();
        self.push(
            pipeline,
            &[(0, x), (1, res), (2, w), (4, out), (5, &p)],
            grid,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn fnc_c(
        &mut self,
        x: &wgpu::Buffer,
        res: &wgpu::Buffer,
        w1: &wgpu::Buffer,
        w2: &wgpu::Buffer,
        out: &wgpu::Buffer,
        out2: &wgpu::Buffer,
        batch: usize,
        hidden: usize,
        eps: f32,
        scale: f32,
    ) {
        let p = self.fnc_uni(batch, hidden, eps, scale);
        let grid = self.grid_1d(batch as u64, 1);
        let pipeline = self.pl.fnc.as_ref().expect("fnc pipelines").c.clone();
        self.push(
            pipeline,
            &[
                (0, x),
                (1, res),
                (2, w1),
                (3, w2),
                (4, out),
                (5, &p),
                (6, out2),
            ],
            grid,
        );
    }

    fn resadd(
        &mut self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        y: &wgpu::Buffer,
        n: usize,
        scale: f32,
    ) {
        let p = self.uni(
            "e4bw-res-p",
            ScaleParams {
                n: n as u32,
                n_words: (n / 2) as u32,
                scale,
                ..Default::default()
            },
        );
        let grid = self.grid_1d((n / 2) as u64, 256);
        let pipeline = self.pl.resadd.clone();
        self.push(pipeline, &[(0, a), (1, b), (2, y), (3, &p)], grid);
    }

    fn lora_site(
        &mut self,
        pl: &LoraPipelines,
        scratch: &wgpu::Buffer,
        m: usize,
        site: &LoraSiteGpu,
        x: &wgpu::Buffer,
        segs: &[&wgpu::Buffer],
    ) {
        let ctx = self.ctx;
        let out = if self.to_verify {
            &mut self.verify_prefill_passes
        } else if self.to_prefill {
            &mut self.prefill_passes
        } else {
            &mut self.passes
        };
        emit_lora_site(ctx, out, &mut self.keep, pl, scratch, m, site, x, segs);
    }

    fn gemv(&mut self, proj: &GpuProj, x_packed: &wgpu::Buffer, dst: GemvDst) {
        match dst {
            GemvDst::Packed { y, word_off } => {
                let off = self.uni(
                    "e4bw-pk-off",
                    PkOffParams {
                        dst_word_off: word_off as u32,
                        ..Default::default()
                    },
                );
                match proj {
                    GpuProj::Bf16 {
                        w, params, grid, ..
                    } => {
                        let pipeline = self.pl.gemv_pk.clone();
                        self.push(
                            pipeline,
                            &[
                                (0, w.raw()),
                                (1, x_packed),
                                (2, y),
                                (3, params.raw()),
                                (30, &off),
                            ],
                            *grid,
                        );
                    }
                    GpuProj::I8 {
                        w,
                        scales,
                        params,
                        grid,
                        group,
                        ..
                    } => {
                        let pipeline = if *group > 0 {
                            self.pl
                                .gemv_i8g_pk
                                .as_ref()
                                .expect("i8g proj without i8g pipelines")
                                .clone()
                        } else {
                            self.pl
                                .gemv_i8_pk
                                .as_ref()
                                .expect("i8 proj without i8 pipelines")
                                .clone()
                        };
                        self.push(
                            pipeline,
                            &[
                                (0, w.raw()),
                                (1, x_packed),
                                (2, y),
                                (3, params.raw()),
                                (22, scales.raw()),
                                (30, &off),
                            ],
                            *grid,
                        );
                    }
                    GpuProj::W4 {
                        packed,
                        scales,
                        params,
                        grid,
                        variant,
                        ..
                    } => match variant {
                        W4Variant::Sg16 => {
                            let pipeline = self
                                .pl
                                .gemv_w4_sg_pkm
                                .as_ref()
                                .or(self.pl.gemv_w4_sg_pk.as_ref())
                                .expect("sg16 routing without sg pipeline")
                                .clone();
                            self.push(
                                pipeline,
                                &[
                                    (1, scales.raw()),
                                    (3, y),
                                    (4, params.raw()),
                                    (6, packed.raw()),
                                    (7, x_packed),
                                    (30, &off),
                                ],
                                *grid,
                            );
                        }
                        W4Variant::V4 => {
                            let pipeline = self.pl.gemv_w4_v4_pk.clone();
                            self.push(
                                pipeline,
                                &[
                                    (1, scales.raw()),
                                    (3, y),
                                    (4, params.raw()),
                                    (6, packed.raw()),
                                    (7, x_packed),
                                    (30, &off),
                                ],
                                *grid,
                            );
                        }
                        W4Variant::Block => {
                            let pipeline = self.pl.gemv_w4_pk.clone();
                            self.push(
                                pipeline,
                                &[
                                    (0, packed.raw()),
                                    (1, scales.raw()),
                                    (2, x_packed),
                                    (3, y),
                                    (4, params.raw()),
                                    (30, &off),
                                ],
                                *grid,
                            );
                        }
                    },
                }
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
                    "e4bw-split-p",
                    SplitParams {
                        q_rows: q_rows as u32,
                        kv_rows: kv_rows as u32,
                        v_off: v_off as u32,
                        pad0: 0,
                    },
                );
                match proj {
                    GpuProj::Bf16 {
                        w, params, grid, ..
                    } => {
                        let pipeline = self.pl.gemv_pk3.clone();
                        self.push(
                            pipeline,
                            &[
                                (0, w.raw()),
                                (1, x_packed),
                                (3, params.raw()),
                                (31, q),
                                (32, k),
                                (33, v),
                                (34, &sp),
                            ],
                            *grid,
                        );
                    }
                    GpuProj::I8 {
                        w,
                        scales,
                        params,
                        grid,
                        group,
                        ..
                    } => {
                        let pipeline = if *group > 0 {
                            self.pl
                                .gemv_i8g_pk3
                                .as_ref()
                                .expect("i8g proj without i8g pipelines")
                                .clone()
                        } else {
                            self.pl
                                .gemv_i8_pk3
                                .as_ref()
                                .expect("i8 proj without i8 pipelines")
                                .clone()
                        };
                        self.push(
                            pipeline,
                            &[
                                (0, w.raw()),
                                (1, x_packed),
                                (3, params.raw()),
                                (22, scales.raw()),
                                (31, q),
                                (32, k),
                                (33, v),
                                (34, &sp),
                            ],
                            *grid,
                        );
                    }
                    GpuProj::W4 {
                        packed,
                        scales,
                        params,
                        grid,
                        variant,
                        ..
                    } => match variant {
                        W4Variant::Sg16 => {
                            let pipeline = self
                                .pl
                                .gemv_w4_sg_pkm3
                                .as_ref()
                                .or(self.pl.gemv_w4_sg_pk3.as_ref())
                                .expect("sg16 routing without sg pipeline")
                                .clone();
                            self.push(
                                pipeline,
                                &[
                                    (1, scales.raw()),
                                    (4, params.raw()),
                                    (6, packed.raw()),
                                    (7, x_packed),
                                    (31, q),
                                    (32, k),
                                    (33, v),
                                    (34, &sp),
                                ],
                                *grid,
                            );
                        }
                        W4Variant::V4 => {
                            let pipeline = self.pl.gemv_w4_v4_pk3.clone();
                            self.push(
                                pipeline,
                                &[
                                    (1, scales.raw()),
                                    (4, params.raw()),
                                    (6, packed.raw()),
                                    (7, x_packed),
                                    (31, q),
                                    (32, k),
                                    (33, v),
                                    (34, &sp),
                                ],
                                *grid,
                            );
                        }
                        W4Variant::Block => {
                            let pipeline = self.pl.gemv_w4_pk3.clone();
                            self.push(
                                pipeline,
                                &[
                                    (0, packed.raw()),
                                    (1, scales.raw()),
                                    (2, x_packed),
                                    (4, params.raw()),
                                    (31, q),
                                    (32, k),
                                    (33, v),
                                    (34, &sp),
                                ],
                                *grid,
                            );
                        }
                    },
                }
            }
        }
    }
}

impl Builder<'_> {
    #[allow(clippy::too_many_arguments)]
    fn emit_na_w4_mk(
        &mut self,
        packed: &wgpu::Buffer,
        scales: &wgpu::Buffer,
        variant: W4Variant,
        n: usize,
        k: usize,
        gs: usize,
        x_packed: &wgpu::Buffer,
        x_off: u64,
        row0: usize,
        m: usize,
        mkp: &wgpu::Buffer,
        dst: &GemvDstMk,
    ) -> bool {
        if m > na::TILE_M as usize {
            na_route_note("prefill m exceeds the tensor-ops tile height");
            return false;
        }
        if !k.is_multiple_of(32) || gs < 32 || !gs.is_multiple_of(32) || !n.is_multiple_of(2) {
            na_route_note("projection shape unfit for tensor-ops tiles");
            return false;
        }
        if let GemvDstMk::SplitQkv {
            q_rows,
            kv_rows,
            v_off,
            ..
        } = dst
        {
            if !q_rows.is_multiple_of(2) || !kv_rows.is_multiple_of(2) || !v_off.is_multiple_of(2) {
                na_route_note("qkv split boundaries are odd");
                return false;
            }
        }
        let pipeline = match dst {
            GemvDstMk::Packed { .. } => na::pk_pipeline(self.ctx),
            GemvDstMk::SplitQkv { .. } => na::pk3_pipeline(self.ctx),
        };
        let pipeline = match pipeline {
            Ok(p) => p,
            Err(e) => {
                na_route_note(&format!("pipeline unavailable: {e}"));
                return false;
            }
        };
        let scale_elem_stride = match variant {
            W4Variant::Sg16 => 1u32,
            W4Variant::Block | W4Variant::V4 => 2u32,
        };
        let (q_rows, kv_rows, v_off) = match dst {
            GemvDstMk::SplitQkv {
                q_rows,
                kv_rows,
                v_off,
                ..
            } => (*q_rows, *kv_rows, *v_off),
            GemvDstMk::Packed { .. } => (0, 0, 0),
        };
        let nap = self.uni(
            "e4bw-na-p",
            na::NaStaticParams {
                n_rows: n as u32,
                k_elems: k as u32,
                gs: gs as u32,
                scale_row_stride: (k / gs) as u32,
                scale_elem_stride,
                q_rows: q_rows as u32,
                kv_rows: kv_rows as u32,
                v_off: v_off as u32,
            },
        );
        let grid = (na::grid_x(n as u32), 1, 1);
        match dst {
            GemvDstMk::Packed { y, .. } => {
                self.push_off(
                    pipeline,
                    &[
                        (0, packed, 0),
                        (1, scales, 0),
                        (2, x_packed, x_off),
                        (3, y, 0),
                        (4, mkp, 0),
                        (5, &nap, 0),
                    ],
                    grid,
                );
            }
            GemvDstMk::SplitQkv { q, k: kb, v, .. } => {
                let q_off = (row0 * (q_rows / 2) * 4) as u64;
                let kv_off = (row0 * (kv_rows / 2) * 4) as u64;
                self.push_off(
                    pipeline,
                    &[
                        (0, packed, 0),
                        (1, scales, 0),
                        (2, x_packed, x_off),
                        (3, q, q_off),
                        (4, kb, kv_off),
                        (5, v, kv_off),
                        (6, mkp, 0),
                        (7, &nap, 0),
                    ],
                    grid,
                );
            }
        }
        true
    }

    fn gemv_mk(
        &mut self,
        proj: &GpuProj,
        x_packed: &wgpu::Buffer,
        m: usize,
        x_stride_words: usize,
        dst: GemvDstMk,
    ) {
        let m = match (self.to_verify, self.pl.mk_verify.as_ref()) {
            (true, Some(v)) => m.min(v.rows),
            _ => m,
        };
        let mut row0 = 0usize;
        while row0 < m {
            let rows = PREFILL_SLAB.min(m - row0);
            self.gemv_mk_slab(proj, x_packed, row0, rows, x_stride_words, &dst);
            row0 += rows;
        }
    }

    fn gemv_mk_slab(
        &mut self,
        proj: &GpuProj,
        x_packed: &wgpu::Buffer,
        row0: usize,
        m: usize,
        x_stride_words: usize,
        dst: &GemvDstMk,
    ) {
        let mkpl = if self.to_verify {
            self.pl.mk_verify.as_ref().or(self.pl.mk.as_ref())
        } else {
            self.pl.mk.as_ref()
        }
        .expect("gemv_mk without mk pipelines");
        let x_off = (row0 * x_stride_words * 4) as u64;
        let (y_stride_words, dst_word_off, q_off, kv_off) = match dst {
            GemvDstMk::Packed {
                y_stride_words,
                word_off,
                ..
            } => (
                *y_stride_words,
                *word_off + row0 * *y_stride_words,
                0u64,
                0u64,
            ),
            GemvDstMk::SplitQkv {
                q_rows, kv_rows, ..
            } => (
                0,
                0,
                (row0 * (*q_rows / 2) * 4) as u64,
                (row0 * (*kv_rows / 2) * 4) as u64,
            ),
        };
        let mkp_val = MkParams {
            m: m as u32,
            x_stride_words: x_stride_words as u32,
            y_stride_words: y_stride_words as u32,
            dst_word_off: dst_word_off as u32,
        };
        let mkp = self.uni("e4bw-mk-p", mkp_val);
        if self.to_verify {
            self.mk_verify_unis.push((mkp.clone(), mkp_val, row0));
        } else if self.to_prefill {
            self.mk_prefill_unis.push((mkp.clone(), mkp_val, row0));
        }
        if self.to_prefill && !self.to_verify && na_route_enabled() {
            if let GpuProj::W4 {
                packed,
                scales,
                variant,
                n,
                k,
                gs,
                ..
            } = proj
            {
                let (packed, scales) = (packed.raw().clone(), scales.raw().clone());
                if self.emit_na_w4_mk(
                    &packed, &scales, *variant, *n, *k, *gs, x_packed, x_off, row0, m, &mkp, dst,
                ) {
                    return;
                }
            } else {
                na_route_note("projection is not w4a16-packed");
            }
        }
        let n = match proj {
            GpuProj::Bf16 { n, .. } | GpuProj::W4 { n, .. } | GpuProj::I8 { n, .. } => *n,
        };
        let sg = matches!(
            proj,
            GpuProj::W4 {
                variant: W4Variant::Sg16,
                ..
            }
        );
        let rows_per_group = if sg {
            wk::gemv_w4a16::SG_PK_ROWS as usize
        } else {
            8
        };
        let groups = n.div_ceil(rows_per_group) as u32;
        assert!(
            groups <= self.ctx.caps.max_compute_workgroups_per_dimension,
            "gemv_mk grid {groups} exceeds device limit"
        );
        let grid = (groups, 1, 1);
        let sp = match dst {
            GemvDstMk::SplitQkv {
                q_rows,
                kv_rows,
                v_off,
                ..
            } => Some(self.uni(
                "e4bw-mk-split-p",
                SplitParams {
                    q_rows: *q_rows as u32,
                    kv_rows: *kv_rows as u32,
                    v_off: *v_off as u32,
                    pad0: 0,
                },
            )),
            GemvDstMk::Packed { .. } => None,
        };
        match proj {
            GpuProj::Bf16 { w, params, .. } => match dst {
                GemvDstMk::Packed { y, .. } => {
                    let pipeline = mkpl.gemm_bf16_pk.clone();
                    self.push_off(
                        pipeline,
                        &[
                            (0, w.raw(), 0),
                            (1, x_packed, x_off),
                            (2, y, 0),
                            (3, params.raw(), 0),
                            (35, &mkp, 0),
                        ],
                        grid,
                    );
                }
                GemvDstMk::SplitQkv { q, k, v, .. } => {
                    let pipeline = mkpl.gemm_bf16_pk3.clone();
                    self.push_off(
                        pipeline,
                        &[
                            (0, w.raw(), 0),
                            (1, x_packed, x_off),
                            (3, params.raw(), 0),
                            (31, q, q_off),
                            (32, k, kv_off),
                            (33, v, kv_off),
                            (34, sp.as_ref().unwrap(), 0),
                            (35, &mkp, 0),
                        ],
                        grid,
                    );
                }
            },
            GpuProj::I8 {
                w,
                scales,
                params,
                group,
                ..
            } => match dst {
                GemvDstMk::Packed { y, .. } => {
                    let pipeline = if *group > 0 {
                        mkpl.gemm_i8g_pk
                            .as_ref()
                            .expect("i8g proj without i8g mk pipelines")
                            .clone()
                    } else {
                        mkpl.gemm_i8_pk
                            .as_ref()
                            .expect("i8 proj without i8 mk pipelines")
                            .clone()
                    };
                    self.push_off(
                        pipeline,
                        &[
                            (0, w.raw(), 0),
                            (1, x_packed, x_off),
                            (2, y, 0),
                            (3, params.raw(), 0),
                            (22, scales.raw(), 0),
                            (35, &mkp, 0),
                        ],
                        grid,
                    );
                }
                GemvDstMk::SplitQkv { q, k, v, .. } => {
                    let pipeline = if *group > 0 {
                        mkpl.gemm_i8g_pk3
                            .as_ref()
                            .expect("i8g proj without i8g mk pipelines")
                            .clone()
                    } else {
                        mkpl.gemm_i8_pk3
                            .as_ref()
                            .expect("i8 proj without i8 mk pipelines")
                            .clone()
                    };
                    self.push_off(
                        pipeline,
                        &[
                            (0, w.raw(), 0),
                            (1, x_packed, x_off),
                            (3, params.raw(), 0),
                            (22, scales.raw(), 0),
                            (31, q, q_off),
                            (32, k, kv_off),
                            (33, v, kv_off),
                            (34, sp.as_ref().unwrap(), 0),
                            (35, &mkp, 0),
                        ],
                        grid,
                    );
                }
            },
            GpuProj::W4 {
                packed,
                scales,
                params,
                variant,
                ..
            } => {
                let v4 = !matches!(variant, W4Variant::Block);
                match dst {
                    GemvDstMk::Packed { y, .. } => {
                        if sg {
                            let pipeline = mkpl
                                .gemm_w4_sg_pk
                                .as_ref()
                                .expect("sg16 mk routing without sg pipeline")
                                .clone();
                            self.push_off(
                                pipeline,
                                &[
                                    (1, scales.raw(), 0),
                                    (3, y, 0),
                                    (4, params.raw(), 0),
                                    (6, packed.raw(), 0),
                                    (7, x_packed, x_off),
                                    (35, &mkp, 0),
                                ],
                                grid,
                            );
                        } else if v4 {
                            let pipeline = mkpl.gemm_w4_v4_pk.clone();
                            self.push_off(
                                pipeline,
                                &[
                                    (1, scales.raw(), 0),
                                    (3, y, 0),
                                    (4, params.raw(), 0),
                                    (6, packed.raw(), 0),
                                    (7, x_packed, x_off),
                                    (35, &mkp, 0),
                                ],
                                grid,
                            );
                        } else {
                            let pipeline = mkpl.gemm_w4_pk.clone();
                            self.push_off(
                                pipeline,
                                &[
                                    (0, packed.raw(), 0),
                                    (1, scales.raw(), 0),
                                    (2, x_packed, x_off),
                                    (3, y, 0),
                                    (4, params.raw(), 0),
                                    (35, &mkp, 0),
                                ],
                                grid,
                            );
                        }
                    }
                    GemvDstMk::SplitQkv { q, k, v, .. } => {
                        if sg {
                            let pipeline = mkpl
                                .gemm_w4_sg_pk3
                                .as_ref()
                                .expect("sg16 mk routing without sg pipeline")
                                .clone();
                            self.push_off(
                                pipeline,
                                &[
                                    (1, scales.raw(), 0),
                                    (4, params.raw(), 0),
                                    (6, packed.raw(), 0),
                                    (7, x_packed, x_off),
                                    (31, q, q_off),
                                    (32, k, kv_off),
                                    (33, v, kv_off),
                                    (34, sp.as_ref().unwrap(), 0),
                                    (35, &mkp, 0),
                                ],
                                grid,
                            );
                        } else if v4 {
                            let pipeline = mkpl.gemm_w4_v4_pk3.clone();
                            self.push_off(
                                pipeline,
                                &[
                                    (1, scales.raw(), 0),
                                    (4, params.raw(), 0),
                                    (6, packed.raw(), 0),
                                    (7, x_packed, x_off),
                                    (31, q, q_off),
                                    (32, k, kv_off),
                                    (33, v, kv_off),
                                    (34, sp.as_ref().unwrap(), 0),
                                    (35, &mkp, 0),
                                ],
                                grid,
                            );
                        } else {
                            let pipeline = mkpl.gemm_w4_pk3.clone();
                            self.push_off(
                                pipeline,
                                &[
                                    (0, packed.raw(), 0),
                                    (1, scales.raw(), 0),
                                    (2, x_packed, x_off),
                                    (4, params.raw(), 0),
                                    (31, q, q_off),
                                    (32, k, kv_off),
                                    (33, v, kv_off),
                                    (34, sp.as_ref().unwrap(), 0),
                                    (35, &mkp, 0),
                                ],
                                grid,
                            );
                        }
                    }
                }
            }
        }
    }
}

fn upload_proj(ctx: &WgpuContext, b: &mut Builder, label: &str, l: &HostLin) -> Result<GpuProj> {
    anyhow::ensure!(
        l.n.is_multiple_of(2),
        "{label}: packed gemv needs n % 2 == 0, got {}",
        l.n
    );
    if let Some(q) = &l.q {
        wk::gemv_w4a16::shape_rule(l.k, q.gs).map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
        anyhow::ensure!(
            q.packed.len() == l.n * l.k / 8,
            "{label}: packed length {} != {}x{}/8",
            q.packed.len(),
            l.n,
            l.k
        );
        anyhow::ensure!(
            q.scales.len() == l.n * (l.k / q.gs),
            "{label}: scale length {} != {}x{}/{}",
            q.scales.len(),
            l.n,
            l.k,
            q.gs
        );
        let packed = GpuTensor::upload(ctx, label, &q.packed);
        let force_block = std::env::var("NV_E4B_WGPU_W4_BLOCK").ok().as_deref() == Some("1");
        let grain = b.pl.w4_grain;
        let sg_ok =
            b.sg16 && grain.accepts(q.gs) && !(w4_route_enabled() && w4_prefer_v4(l.n, l.k));
        let variant = match wk::gemv_w4a16::w4_route(l.n, l.k, q.gs, sg_ok, force_block) {
            wk::gemv_w4a16::W4Route::Block => W4Variant::Block,
            wk::gemv_w4a16::W4Route::V4 => W4Variant::V4,
            wk::gemv_w4a16::W4Route::Sg16 => W4Variant::Sg16,
        };
        if variant == W4Variant::Sg16 {
            wk::gemv_w4a16::require_grain(grain, q.gs)
                .map_err(|e| anyhow::anyhow!("{label}: {e}"))?;
        }
        b.w4_census[match variant {
            W4Variant::Block => 0,
            W4Variant::V4 => 1,
            W4Variant::Sg16 => 2,
        }] += 1;
        let scale_words = if variant == W4Variant::Sg16 {
            wk::gemv_w4a16::pack_scale_words(&q.scales)
        } else {
            q.scales.iter().map(|&s| s as u32).collect()
        };
        b.weight_bytes += 4 * (q.packed.len() as u64) + 4 * (scale_words.len() as u64);
        let scales = GpuTensor::upload(ctx, label, &scale_words);
        let rows_per_group = if variant == W4Variant::Sg16 {
            wk::gemv_w4a16::SG_PK_ROWS * w4_mr()
        } else {
            wk::gemv_w4a16::ROWS_PER_GROUP
        };
        let grid = b.grid_1d(l.n as u64, rows_per_group);
        let params = GpuUniform::new(
            ctx,
            label,
            &GemvW4Params {
                n_rows: l.n as u32,
                k_elems: l.k as u32,
                gs: q.gs as u32,
                w_row_words: (l.k / 8) as u32,
                scale_row_stride: (l.k / q.gs) as u32,
                groups_x: grid.0,
            },
        );
        return Ok(GpuProj::W4 {
            packed,
            scales,
            params,
            grid,
            variant,
            n: l.n,
            k: l.k,
            gs: q.gs,
        });
    }
    anyhow::ensure!(
        l.k.is_multiple_of(8),
        "{label}: bf16 gemv needs k % 8 == 0, got {}",
        l.k
    );
    anyhow::ensure!(
        l.w.len() == l.n * l.k,
        "{label}: weight length {} != {}x{}",
        l.w.len(),
        l.n,
        l.k
    );
    if w8_enabled() {
        let group = if w8_group() > 0 && l.k.is_multiple_of(w8_group()) && l.k >= w8_group() {
            w8_group()
        } else {
            0
        };
        let (packed, sc) = if group > 0 {
            wk::quant_gemv::quantize_groups(&l.w, l.n, l.k, group, wk::quant_gemv::QFormat::Int8)
        } else {
            wk::quant_gemv::quantize_rows_int8(&l.w, l.n, l.k)
        };
        b.weight_bytes += (l.n as u64) * (l.k as u64) + 4 * (sc.len() as u64);
        let w = GpuTensor::upload(ctx, label, &packed);
        let scales = GpuTensor::upload(ctx, label, &sc);
        let grid = b.grid_1d(l.n as u64, wk::gemv_bf16::ROWS_PER_GROUP);
        let params = GpuUniform::new(
            ctx,
            label,
            &GemvBf16Params {
                n_rows: l.n as u32,
                k_elems: l.k as u32,
                w_row_words: (l.k / 4) as u32,
                groups_x: grid.0,
            },
        );
        return Ok(GpuProj::I8 {
            w,
            scales,
            params,
            grid,
            n: l.n,
            group,
        });
    }
    b.weight_bytes += 2 * (l.n as u64) * (l.k as u64);
    let w = GpuTensor::upload(ctx, label, &pack_pairs(&l.w));
    let grid = b.grid_1d(l.n as u64, wk::gemv_bf16::ROWS_PER_GROUP);
    let params = GpuUniform::new(
        ctx,
        label,
        &GemvBf16Params {
            n_rows: l.n as u32,
            k_elems: l.k as u32,
            w_row_words: (l.k / 2) as u32,
            groups_x: grid.0,
        },
    );
    Ok(GpuProj::Bf16 {
        w,
        params,
        grid,
        n: l.n,
    })
}

struct EmitCfg {
    hidden: usize,
    inter: usize,
    eps: f32,
    n_q: usize,
    n_layers: usize,
    hpl: usize,
    ple_row: usize,
    max_seq: usize,
}

struct Bufs {
    m: usize,
    hid_a: GpuTensor<u32>,
    hid_b: GpuTensor<u32>,
    t0: GpuTensor<u32>,
    t1: GpuTensor<u32>,
    t2: GpuTensor<u32>,
    mid: GpuTensor<u32>,
    ple_raw: GpuTensor<u32>,
    ple_proj: GpuTensor<u32>,
    ple_normed: GpuTensor<u32>,
    pli: GpuTensor<u32>,
    pl_gate: GpuTensor<u32>,
    pl_gated: GpuTensor<u32>,
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
    scratch: GpuTensor<f32>,
}

fn alloc_bufs(
    ctx: &WgpuContext,
    m: usize,
    cfg: &EmitCfg,
    q_dim_max: usize,
    kv_dim_max: usize,
    hd_max: usize,
) -> Bufs {
    let h = m * cfg.hidden / 2;
    Bufs {
        m,
        hid_a: GpuTensor::zeroed(ctx, "e4bw-hid-a", h),
        hid_b: GpuTensor::zeroed(ctx, "e4bw-hid-b", h),
        t0: GpuTensor::zeroed(ctx, "e4bw-t0", h),
        t1: GpuTensor::zeroed(ctx, "e4bw-t1", h),
        t2: GpuTensor::zeroed(ctx, "e4bw-t2", h),
        mid: GpuTensor::zeroed(ctx, "e4bw-mid", h),
        ple_raw: GpuTensor::zeroed(ctx, "e4bw-ple-raw", m * cfg.ple_row / 2),
        ple_proj: GpuTensor::zeroed(ctx, "e4bw-ple-proj", m * cfg.ple_row / 2),
        ple_normed: GpuTensor::zeroed(ctx, "e4bw-ple-normed", m * cfg.ple_row / 2),
        pli: GpuTensor::zeroed(ctx, "e4bw-pli", m * cfg.ple_row / 2),
        pl_gate: GpuTensor::zeroed(ctx, "e4bw-pl-gate", m * cfg.hpl / 2),
        pl_gated: GpuTensor::zeroed(ctx, "e4bw-pl-gated", m * cfg.hpl / 2),
        qa: GpuTensor::zeroed(ctx, "e4bw-qa", m * q_dim_max / 2),
        qb: GpuTensor::zeroed(ctx, "e4bw-qb", m * q_dim_max / 2),
        ka: GpuTensor::zeroed(ctx, "e4bw-ka", m * kv_dim_max / 2),
        kb: GpuTensor::zeroed(ctx, "e4bw-kb", m * kv_dim_max / 2),
        va: GpuTensor::zeroed(ctx, "e4bw-va", m * kv_dim_max / 2),
        vb: GpuTensor::zeroed(ctx, "e4bw-vb", m * kv_dim_max / 2),
        q_f32: GpuTensor::zeroed(ctx, "e4bw-qf32", m * q_dim_max),
        attn_pack: GpuTensor::zeroed(ctx, "e4bw-attn-pack", m * q_dim_max / 2),
        gu_pack: GpuTensor::zeroed(ctx, "e4bw-gu-pack", m * cfg.inter.max(cfg.hidden)),
        act_pack: GpuTensor::zeroed(ctx, "e4bw-act-pack", m * cfg.inter.max(cfg.hidden) / 2),
        scratch: GpuTensor::zeroed(
            ctx,
            "e4bw-flash-scratch",
            cfg.n_q * m * scratch_splits() as usize * (hd_max + 2),
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_prologue(
    b: &mut Builder,
    bufs: &Bufs,
    cfg: &EmitCfg,
    embed_tab: &GpuTable,
    ple_tab: &GpuTable,
    plmp: &GpuProj,
    plp_norm: &wgpu::Buffer,
    tok_idx: &wgpu::Buffer,
) {
    let m = bufs.m;
    let hidden = cfg.hidden;
    let gather_pl = if m == 1 {
        b.pl.gather.clone()
    } else {
        b.pl.mk
            .as_ref()
            .expect("prefill gather pipeline")
            .gather
            .clone()
    };
    let embed_p = b.uni("e4bw-embed-gp", embed_tab.params());
    {
        let mut binds = embed_tab.binds();
        binds.push((8, tok_idx));
        binds.push((9, bufs.t0.raw()));
        binds.push((10, &embed_p));
        b.push(gather_pl.clone(), &binds, (m as u32, 1, 1));
    }
    let embed_scale = (hidden as f32).sqrt();
    let scale_p = b.uni(
        "e4bw-embed-scale-p",
        ScaleParams {
            n: (m * hidden) as u32,
            n_words: (m * hidden / 2) as u32,
            scale: embed_scale,
            ..Default::default()
        },
    );
    let sgrid = b.grid_1d((m * hidden / 2) as u64, 256);
    let scale_pl = b.pl.scale.clone();
    b.push(
        scale_pl,
        &[(0, bufs.t0.raw()), (2, bufs.hid_a.raw()), (3, &scale_p)],
        sgrid,
    );

    let ple_p = b.uni("e4bw-ple-gp", ple_tab.params());
    {
        let mut binds = ple_tab.binds();
        binds.push((8, tok_idx));
        binds.push((9, bufs.ple_raw.raw()));
        binds.push((10, &ple_p));
        b.push(gather_pl, &binds, (m as u32, 1, 1));
    }
    if m == 1 {
        b.gemv(
            plmp,
            bufs.hid_a.raw(),
            GemvDst::Packed {
                y: bufs.ple_proj.raw(),
                word_off: 0,
            },
        );
    } else {
        b.gemv_mk(
            plmp,
            bufs.hid_a.raw(),
            m,
            hidden / 2,
            GemvDstMk::Packed {
                y: bufs.ple_proj.raw(),
                word_off: 0,
                y_stride_words: cfg.ple_row / 2,
            },
        );
    }
    b.rms(
        bufs.ple_proj.raw(),
        plp_norm,
        bufs.ple_normed.raw(),
        m * cfg.n_layers,
        cfg.hpl,
        cfg.eps,
    );
    {
        let ax_p = b.uni(
            "e4bw-pli-p",
            AxParams {
                n_words: (m * cfg.ple_row / 2) as u32,
                sa: std::f32::consts::FRAC_1_SQRT_2,
                sb: (cfg.hpl as f32).sqrt() * std::f32::consts::FRAC_1_SQRT_2,
                pad0: 0,
            },
        );
        let grid = b.grid_1d((m * cfg.ple_row / 2) as u64, 256);
        let axpby_pl = b.pl.axpby.clone();
        b.push(
            axpby_pl,
            &[
                (0, bufs.ple_normed.raw()),
                (1, bufs.ple_raw.raw()),
                (2, bufs.pli.raw()),
                (3, &ax_p),
            ],
            grid,
        );
    }
}

struct LayerEmit<'x> {
    bufs: &'x Bufs,
    cfg: &'x EmitCfg,
    li: usize,
    shared: Option<usize>,
    full_attn: bool,
    has_v: bool,
    layer_scalar: f32,
    hd: usize,
    nkv: usize,
    q_dim: usize,
    kv_dim: usize,
    qkv: &'x GpuProj,
    o: &'x GpuProj,
    gate_up: &'x GpuProj,
    down: &'x GpuProj,
    pl_gate_proj: &'x GpuProj,
    pl_proj: &'x GpuProj,
    ln_in: &'x wgpu::Buffer,
    ln_pa: &'x wgpu::Buffer,
    ln_pf: &'x wgpu::Buffer,
    ln_po: &'x wgpu::Buffer,
    ln_pli: &'x wgpu::Buffer,
    ln_next: &'x wgpu::Buffer,
    qn: &'x wgpu::Buffer,
    kn: &'x wgpu::Buffer,
    vn: &'x wgpu::Buffer,
    cos: &'x wgpu::Buffer,
    sin: &'x wgpu::Buffer,
    kv: &'x GpuLayerKv,
    fd: &'x wgpu::Buffer,
    rope_pos: &'x wgpu::Buffer,
    kv_start: &'x wgpu::Buffer,
    lora: Option<&'x LayerLoraGpu>,
    lora_pl: Option<&'x LoraPipelines>,
    lora_scratch: Option<&'x wgpu::Buffer>,
}

impl LayerEmit<'_> {
    fn lora_parts(&self) -> Option<(&LoraPipelines, &wgpu::Buffer, &LayerLoraGpu)> {
        match (self.lora_pl, self.lora_scratch, self.lora) {
            (Some(pl), Some(s), Some(l)) => Some((pl, s, l)),
            _ => None,
        }
    }
}

fn emit_na_attn_prefill(b: &mut Builder, a: &LayerEmit, m: usize) -> bool {
    let bufs = a.bufs;
    let n_q = a.cfg.n_q;
    let pipeline = if a.hd == na_attn::HEAD_DIM as usize {
        na_attn::pipeline(b.ctx)
    } else if a.hd == na_attn::HEAD_DIM_G as usize {
        na_attn::pipeline_g(b.ctx)
    } else {
        na_attn_note("head_dim is neither 256 nor 512");
        return false;
    };
    if a.nkv == 0 || !n_q.is_multiple_of(a.nkv) {
        na_attn_note("gqa head mapping unfit");
        return false;
    }
    let pipeline = match pipeline {
        Ok(p) => p,
        Err(e) => {
            na_attn_note(&format!("pipeline unavailable: {e}"));
            return false;
        }
    };
    b.push(
        pipeline,
        &[
            (0, bufs.q_f32.raw()),
            (1, a.kv.k_fp8.raw()),
            (2, a.kv.v_fp8.raw()),
            (3, a.kv.k_scales.raw()),
            (4, a.kv.v_scales.raw()),
            (5, bufs.attn_pack.raw()),
            (6, a.fd),
        ],
        na_attn::grid(n_q as u32, m as u32),
    );
    true
}

fn emit_layer(b: &mut Builder, a: &LayerEmit) {
    let bufs = a.bufs;
    let cfg = a.cfg;
    let m = bufs.m;
    let hidden = cfg.hidden;
    let inter = cfg.inter;
    let eps = cfg.eps;
    let n_q = cfg.n_q;
    let hpl = cfg.hpl;
    let (x_in, x_out) = if a.li.is_multiple_of(2) {
        (bufs.hid_a.raw(), bufs.hid_b.raw())
    } else {
        (bufs.hid_b.raw(), bufs.hid_a.raw())
    };

    let fuse = m == 1 && b.pl.fnc.is_some();
    if !fuse || a.li == 0 {
        b.rms(x_in, a.ln_in, bufs.t0.raw(), m, hidden, eps);
    }
    match a.shared {
        Some(_) => {
            if m == 1 {
                b.gemv(
                    a.qkv,
                    bufs.t0.raw(),
                    GemvDst::Packed {
                        y: bufs.qa.raw(),
                        word_off: 0,
                    },
                );
            } else {
                b.gemv_mk(
                    a.qkv,
                    bufs.t0.raw(),
                    m,
                    hidden / 2,
                    GemvDstMk::Packed {
                        y: bufs.qa.raw(),
                        word_off: 0,
                        y_stride_words: a.q_dim / 2,
                    },
                );
            }
        }
        None => {
            let v_off = if a.has_v { a.q_dim + a.kv_dim } else { a.q_dim };
            if m == 1 {
                b.gemv(
                    a.qkv,
                    bufs.t0.raw(),
                    GemvDst::SplitQkv {
                        q: bufs.qa.raw(),
                        k: bufs.ka.raw(),
                        v: bufs.va.raw(),
                        q_rows: a.q_dim,
                        kv_rows: a.kv_dim,
                        v_off,
                    },
                );
            } else {
                b.gemv_mk(
                    a.qkv,
                    bufs.t0.raw(),
                    m,
                    hidden / 2,
                    GemvDstMk::SplitQkv {
                        q: bufs.qa.raw(),
                        k: bufs.ka.raw(),
                        v: bufs.va.raw(),
                        q_rows: a.q_dim,
                        kv_rows: a.kv_dim,
                        v_off,
                    },
                );
            }
        }
    }
    if let Some((pl, scratch, ll)) = a.lora_parts() {
        if let Some(site) = &ll.qkv {
            let segs: Vec<&wgpu::Buffer> = match a.shared {
                Some(_) => vec![bufs.qa.raw()],
                None => vec![bufs.qa.raw(), bufs.ka.raw(), bufs.va.raw()],
            };
            b.lora_site(pl, scratch, m, site, bufs.t0.raw(), &segs);
        }
    }
    let half = a.hd / 2;
    let fuse_attn = m == 1
        && b.pl.fac.is_some()
        && a.hd.is_multiple_of(4)
        && a.hd <= wk::fused_attn_chain::MAX_HEAD_DIM;
    if fuse_attn {
        let mk_fac = |b: &mut Builder, heads: usize| {
            b.uni(
                "e4bw-fac-p",
                wk::fused_attn_chain::FacParams {
                    n_heads: heads as u32,
                    head_dim: a.hd as u32,
                    half_dim: half as u32,
                    eps,
                    rows: (m * heads) as u32,
                    ring: 0,
                    pad0: 0,
                    pad1: 0,
                },
            )
        };
        let fac_q_p = mk_fac(b, n_q);
        let fac_q_pl = b.pl.fac.as_ref().unwrap().q.clone();
        b.push(
            fac_q_pl,
            &[
                (0, bufs.qa.raw()),
                (1, a.qn),
                (2, a.cos),
                (3, a.sin),
                (4, a.rope_pos),
                (6, bufs.q_f32.raw()),
                (9, &fac_q_p),
            ],
            ((m * n_q) as u32, 1, 1),
        );
        if a.shared.is_none() {
            let fac_kv_p = mk_fac(b, a.nkv);
            let fac_k_pl = b.pl.fac.as_ref().unwrap().k.clone();
            b.push(
                fac_k_pl,
                &[
                    (0, bufs.ka.raw()),
                    (1, a.kn),
                    (2, a.cos),
                    (3, a.sin),
                    (4, a.rope_pos),
                    (5, a.kv_start),
                    (7, a.kv.k_fp8.raw()),
                    (8, a.kv.k_scales.raw()),
                    (9, &fac_kv_p),
                ],
                ((m * a.nkv) as u32, 1, 1),
            );
            let fac_v_pl = b.pl.fac.as_ref().unwrap().v.clone();
            b.push(
                fac_v_pl,
                &[
                    (0, bufs.va.raw()),
                    (1, a.vn),
                    (5, a.kv_start),
                    (7, a.kv.v_fp8.raw()),
                    (8, a.kv.v_scales.raw()),
                    (9, &fac_kv_p),
                ],
                ((m * a.nkv) as u32, 1, 1),
            );
        }
    } else {
        b.rms(bufs.qa.raw(), a.qn, bufs.qb.raw(), m * n_q, a.hd, eps);
        if a.shared.is_none() {
            b.rms(bufs.ka.raw(), a.kn, bufs.kb.raw(), m * a.nkv, a.hd, eps);
            b.rms(bufs.va.raw(), a.vn, bufs.vb.raw(), m * a.nkv, a.hd, eps);
        }

        let rope_q_p = b.uni(
            "e4bw-rope-q-p",
            RopeParams {
                n_heads: n_q as u32,
                half_dim: half as u32,
                total_words: (m * n_q * half) as u32,
                table_rows: cfg.max_seq as u32,
            },
        );
        let rgq = b.grid_1d((m * n_q * half) as u64, 256);
        let rope_f32_pl = b.pl.rope_f32.clone();
        b.push(
            rope_f32_pl,
            &[
                (0, bufs.qb.raw()),
                (2, a.cos),
                (3, a.sin),
                (4, a.rope_pos),
                (5, &rope_q_p),
                (6, bufs.q_f32.raw()),
            ],
            rgq,
        );

        if a.shared.is_none() {
            let rope_k_p = b.uni(
                "e4bw-rope-k-p",
                RopeParams {
                    n_heads: a.nkv as u32,
                    half_dim: half as u32,
                    total_words: (m * a.nkv * half) as u32,
                    table_rows: cfg.max_seq as u32,
                },
            );
            let rgk = b.grid_1d((m * a.nkv * half) as u64, 256);
            let rope_pl = b.pl.rope.clone();
            b.push(
                rope_pl,
                &[
                    (0, bufs.kb.raw()),
                    (1, bufs.ka.raw()),
                    (2, a.cos),
                    (3, a.sin),
                    (4, a.rope_pos),
                    (5, &rope_k_p),
                ],
                rgk,
            );

            let kvq_p = b.uni(
                "e4bw-kvq-p",
                KvFp8Params {
                    n_tokens: m as u32,
                    n_kv: a.nkv as u32,
                    head_dim: a.hd as u32,
                    ring: 0,
                    pairs: (m * a.nkv) as u32,
                    start: 0,
                    slots: cfg.max_seq as u32,
                    reserved: 0,
                },
            );
            let kvq_pl = b.pl.kvq.clone();
            b.push(
                kvq_pl.clone(),
                &[
                    (0, bufs.ka.raw()),
                    (1, a.kv.k_fp8.raw()),
                    (2, a.kv.k_scales.raw()),
                    (3, a.kv_start),
                    (4, &kvq_p),
                ],
                ((m * a.nkv) as u32, 1, 1),
            );
            b.push(
                kvq_pl,
                &[
                    (0, bufs.vb.raw()),
                    (1, a.kv.v_fp8.raw()),
                    (2, a.kv.v_scales.raw()),
                    (3, a.kv_start),
                    (4, &kvq_p),
                ],
                ((m * a.nkv) as u32, 1, 1),
            );
        }
    }

    if m == 1 {
        let fold = b.pl.gqa_fold;
        let flash1_pl = if fold > 1 {
            flash1_pick(&b.pl.flash1_fold, a.hd)
        } else {
            flash1_for(b.pl, a.hd)
        };
        let s1_binds = [
            (0, bufs.q_f32.raw()),
            (4, a.fd),
            (5, a.kv.k_fp8.raw()),
            (6, a.kv.v_fp8.raw()),
            (7, bufs.scratch.raw()),
            (8, a.kv.k_scales.raw()),
            (9, a.kv.v_scales.raw()),
        ];
        b.push(flash1_pl, &s1_binds, (n_q as u32 / fold, decode_splits(), 1));
        let flash2_pl = b.pl.flash2_pk.clone();
        let s2_binds = [
            (3, bufs.attn_pack.raw()),
            (4, a.fd),
            (7, bufs.scratch.raw()),
        ];
        b.push(flash2_pl, &s2_binds, (n_q as u32, 1, 1));
        b.push_deep_decode_attn(&s1_binds, &s2_binds, n_q as u32, a.hd, a.full_attn);
    } else if !(b.to_prefill && !b.to_verify && na_attn_enabled() && emit_na_attn_prefill(b, a, m))
    {
        let mkpl = b.pl.mk.as_ref().expect("prefill flash pipelines");
        let flash1_pl = match mkpl.flash1_qtile.as_ref().filter(|_| a.full_attn) {
            Some(p) => p.clone(),
            None => mkpl.flash1.clone(),
        };
        let flash2_pl = mkpl.flash2_pk.clone();
        b.push(
            flash1_pl,
            &[
                (0, bufs.q_f32.raw()),
                (4, a.fd),
                (5, a.kv.k_fp8.raw()),
                (6, a.kv.v_fp8.raw()),
                (7, bufs.scratch.raw()),
                (8, a.kv.k_scales.raw()),
                (9, a.kv.v_scales.raw()),
            ],
            (n_q as u32, FLASH_SPLITS, 1),
        );
        b.push(
            flash2_pl,
            &[
                (3, bufs.attn_pack.raw()),
                (4, a.fd),
                (7, bufs.scratch.raw()),
            ],
            (n_q as u32, m as u32, 1),
        );
    }

    if m == 1 {
        b.gemv(
            a.o,
            bufs.attn_pack.raw(),
            GemvDst::Packed {
                y: bufs.t1.raw(),
                word_off: 0,
            },
        );
    } else {
        b.gemv_mk(
            a.o,
            bufs.attn_pack.raw(),
            m,
            a.q_dim / 2,
            GemvDstMk::Packed {
                y: bufs.t1.raw(),
                word_off: 0,
                y_stride_words: hidden / 2,
            },
        );
    }
    if let Some((pl, scratch, ll)) = a.lora_parts() {
        if let Some(site) = &ll.o {
            b.lora_site(pl, scratch, m, site, bufs.attn_pack.raw(), &[bufs.t1.raw()]);
        }
    }
    let mlp_in = if fuse {
        b.fnc_a(
            bufs.t1.raw(),
            x_in,
            a.ln_pa,
            a.ln_pf,
            bufs.t2.raw(),
            m,
            hidden,
            eps,
        );
        bufs.t2.raw()
    } else {
        b.rms(bufs.t1.raw(), a.ln_pa, bufs.t0.raw(), m, hidden, eps);

        let rmsres_p = b.uni(
            "e4bw-rmsres-p",
            RmsParams {
                hidden: hidden as u32,
                batch: m as u32,
                eps,
                words_per_row: (hidden / 2) as u32,
            },
        );
        let rmsres_pl = b.pl.rmsres.clone();
        b.push(
            rmsres_pl,
            &[
                (0, bufs.t0.raw()),
                (1, x_in),
                (2, a.ln_pf),
                (3, bufs.t1.raw()),
                (4, &rmsres_p),
            ],
            (m as u32, 1, 1),
        );
        bufs.t1.raw()
    };

    if m == 1 {
        b.gemv(
            a.gate_up,
            mlp_in,
            GemvDst::Packed {
                y: bufs.gu_pack.raw(),
                word_off: 0,
            },
        );
    } else {
        b.gemv_mk(
            a.gate_up,
            mlp_in,
            m,
            hidden / 2,
            GemvDstMk::Packed {
                y: bufs.gu_pack.raw(),
                word_off: 0,
                y_stride_words: inter,
            },
        );
    }
    if let Some((pl, scratch, ll)) = a.lora_parts() {
        if let Some(site) = &ll.gate_up {
            b.lora_site(pl, scratch, m, site, mlp_in, &[bufs.gu_pack.raw()]);
        }
    }
    let gelu_p = b.uni(
        "e4bw-gelu-p",
        GeluParams {
            inter: inter as u32,
            inter_words: (inter / 2) as u32,
            rows: m as u32,
            tot_pairs: (m * inter) as u32,
        },
    );
    let ggrid = b.grid_1d((inter / 2) as u64, 256);
    let gelu_pl = b.pl.gelu_even.clone();
    b.push(
        gelu_pl,
        &[
            (3, bufs.gu_pack.raw()),
            (4, bufs.act_pack.raw()),
            (5, &gelu_p),
        ],
        (ggrid.0, m as u32, 1),
    );
    if m == 1 {
        b.gemv(
            a.down,
            bufs.act_pack.raw(),
            GemvDst::Packed {
                y: bufs.t0.raw(),
                word_off: 0,
            },
        );
    } else {
        b.gemv_mk(
            a.down,
            bufs.act_pack.raw(),
            m,
            inter / 2,
            GemvDstMk::Packed {
                y: bufs.t0.raw(),
                word_off: 0,
                y_stride_words: hidden / 2,
            },
        );
    }
    if let Some((pl, scratch, ll)) = a.lora_parts() {
        if let Some(site) = &ll.down {
            b.lora_site(pl, scratch, m, site, bufs.act_pack.raw(), &[bufs.t0.raw()]);
        }
    }
    if fuse {
        b.fnc_b(
            bufs.t0.raw(),
            x_in,
            a.ln_po,
            bufs.mid.raw(),
            m,
            hidden,
            eps,
            1.0,
        );
    } else {
        b.rms(bufs.t0.raw(), a.ln_po, bufs.t1.raw(), m, hidden, eps);
        b.resadd(x_in, bufs.t1.raw(), bufs.mid.raw(), m * hidden, 1.0);
    }

    if m == 1 {
        b.gemv(
            a.pl_gate_proj,
            bufs.mid.raw(),
            GemvDst::Packed {
                y: bufs.pl_gate.raw(),
                word_off: 0,
            },
        );
    } else {
        b.gemv_mk(
            a.pl_gate_proj,
            bufs.mid.raw(),
            m,
            hidden / 2,
            GemvDstMk::Packed {
                y: bufs.pl_gate.raw(),
                word_off: 0,
                y_stride_words: hpl / 2,
            },
        );
    }
    let gm_p = b.uni(
        "e4bw-gm-p",
        GmParams {
            n_words: (m * hpl / 2) as u32,
            pli_word_off: (a.li * hpl / 2) as u32,
            tok_words: (hpl / 2) as u32,
            pli_stride: (cfg.ple_row / 2) as u32,
        },
    );
    let gmgrid = b.grid_1d((m * hpl / 2) as u64, 256);
    let gm_pl = if m == 1 {
        b.pl.gatemul.clone()
    } else {
        b.pl.mk
            .as_ref()
            .expect("prefill gatemul pipeline")
            .gatemul
            .clone()
    };
    b.push(
        gm_pl,
        &[
            (0, bufs.pl_gate.raw()),
            (1, bufs.pli.raw()),
            (2, bufs.pl_gated.raw()),
            (3, &gm_p),
        ],
        gmgrid,
    );
    if m == 1 {
        b.gemv(
            a.pl_proj,
            bufs.pl_gated.raw(),
            GemvDst::Packed {
                y: bufs.t1.raw(),
                word_off: 0,
            },
        );
    } else {
        b.gemv_mk(
            a.pl_proj,
            bufs.pl_gated.raw(),
            m,
            hpl / 2,
            GemvDstMk::Packed {
                y: bufs.t1.raw(),
                word_off: 0,
                y_stride_words: hidden / 2,
            },
        );
    }
    if fuse {
        b.fnc_c(
            bufs.t1.raw(),
            bufs.mid.raw(),
            a.ln_pli,
            a.ln_next,
            x_out,
            bufs.t0.raw(),
            m,
            hidden,
            eps,
            a.layer_scalar,
        );
    } else {
        b.rms(bufs.t1.raw(), a.ln_pli, bufs.t2.raw(), m, hidden, eps);
        b.resadd(
            bufs.mid.raw(),
            bufs.t2.raw(),
            x_out,
            m * hidden,
            a.layer_scalar,
        );
    }
}

struct PrefillState {
    m: usize,
    tail: bool,
    passes: Vec<Pass>,
    labels: Vec<String>,
    fd_sliding: GpuUniform<FdParams>,
    fd_full: GpuUniform<FdParams>,
    fd_sliding_base: FdParams,
    fd_full_base: FdParams,
    validated: bool,
    pending_cb: Option<wgpu::CommandBuffer>,
    mk_unis: Vec<(wgpu::Buffer, MkParams, usize)>,
    mk_verify_unis: Vec<(wgpu::Buffer, MkParams, usize)>,
    rows_live: usize,
    rows_live_verify: usize,
    live_rows_enabled: bool,
}

fn verify_live_rows_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_VERIFY_LIVE_ROWS")
        .ok()
        .as_deref()
        != Some("0")
}

impl PrefillState {
    fn set_live_rows(&mut self, ctx: &WgpuContext, rows: usize) {
        debug_assert!((1..=self.m).contains(&rows));
        if rows == self.rows_live {
            return;
        }
        write_mk_live(ctx, &self.mk_unis, rows);
        self.rows_live = rows;
    }

    fn set_verify_live_rows(&mut self, ctx: &WgpuContext, rows: usize) {
        if self.mk_verify_unis.is_empty() {
            return self.set_live_rows(ctx, rows);
        }
        if rows == self.rows_live_verify {
            return;
        }
        write_mk_live(ctx, &self.mk_verify_unis, rows);
        self.rows_live_verify = rows;
    }
}

fn write_mk_live(ctx: &WgpuContext, unis: &[(wgpu::Buffer, MkParams, usize)], rows: usize) {
    for (buf, base, row0) in unis {
        let mut p = *base;
        p.m = (base.m as usize).min(rows.saturating_sub(*row0)) as u32;
        ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
    }
}

pub const VERIFY_ATTN_SEAM_TODO: &str = "verify_chain rides the bit-validated prefill flash for its M-row attention; the small-M swap is FALSIFIED: the afd-parity fp8 kernel (wk::attn_decode_small_m_fp8) wired here regressed structured spec decode and broke serving losslessness via an ulp-level argmax flip vs the decode split-k flash (measurements: perf/runs.jsonl); any future swap must be per-row bit-identical to the decode flash arithmetic";

struct VerifyState {
    rows: usize,
    passes: Vec<Pass>,
    labels: Vec<String>,
    pf_passes: Option<Vec<Pass>>,
    pf_labels: Vec<String>,
    validated: bool,
    pending_cb: Option<wgpu::CommandBuffer>,
    pending_pf_cb: Option<wgpu::CommandBuffer>,
    hid: wgpu::Buffer,
    hid_row_words: usize,
    sm_unis: Vec<(wgpu::Buffer, SmkLiveParams)>,
    rows_live: usize,
}

impl VerifyState {
    fn set_live_rows(&mut self, ctx: &WgpuContext, rows: usize) {
        let rows = rows.min(self.rows);
        if rows == self.rows_live {
            return;
        }
        for (buf, base) in &self.sm_unis {
            let mut p = *base;
            p.m_live = rows as u32;
            ctx.queue.write_buffer(buf, 0, bytemuck::bytes_of(&p));
        }
        self.rows_live = rows;
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct VerifyCapParams {
    n_rows: u32,
    row_off: u32,
    vocab: u32,
    m: u32,
    cap: f32,
    inv_cap: f32,
    softcap: u32,
    pad0: u32,
}

const VERIFY_CAP_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_verify_cap.wgsl");

fn pipeline_name(pl: &Pipelines, p: &Arc<wgpu::ComputePipeline>) -> &'static str {
    if let Some(name) = na::pipeline_label(p) {
        return name;
    }
    if let Some(name) = na_attn::pipeline_label(p) {
        return name;
    }
    let eq = |a: &Arc<wgpu::ComputePipeline>| Arc::ptr_eq(a, p);
    let opt = |o: &Option<Arc<wgpu::ComputePipeline>>| o.as_ref().is_some_and(eq);
    if eq(&pl.gather) {
        return "embed_gather";
    }
    if eq(&pl.scale) {
        return "embed_scale";
    }
    if eq(&pl.rms) {
        return "rmsnorm";
    }
    if eq(&pl.rmsres) {
        return "rmsnorm_residual";
    }
    if eq(&pl.resadd) {
        return "residual_add";
    }
    if eq(&pl.cast_f32) {
        return "cast_f32";
    }
    if eq(&pl.softcap) {
        return "softcap";
    }
    if eq(&pl.rope) {
        return "rope";
    }
    if eq(&pl.rope_f32) {
        return "rope_f32";
    }
    if eq(&pl.kvq) {
        return "kv_fp8_quant";
    }
    if pl
        .flash1
        .iter()
        .chain(pl.flash1_fold.iter())
        .chain(pl.flash1_fold_deep.iter())
        .any(|(_, p)| eq(p))
    {
        return "flash_stage1";
    }
    if eq(&pl.flash2_pk) || opt(&pl.flash2_pk_deep) {
        return "flash_stage2";
    }
    if eq(&pl.gemv_pk) {
        return "gemv_bf16";
    }
    if eq(&pl.gemv_pk3) {
        return "gemv_bf16_qkv";
    }
    if eq(&pl.gemv_w4_pk) {
        return "gemv_w4_block";
    }
    if eq(&pl.gemv_w4_pk3) {
        return "gemv_w4_block_qkv";
    }
    if eq(&pl.gemv_w4_v4_pk) {
        return "gemv_w4_v4";
    }
    if eq(&pl.gemv_w4_v4_pk3) {
        return "gemv_w4_v4_qkv";
    }
    if opt(&pl.gemv_w4_sg_pk) {
        return "gemv_w4_sg16";
    }
    if opt(&pl.gemv_w4_sg_pk3) {
        return "gemv_w4_sg16_qkv";
    }
    if opt(&pl.gemv_w4_sg_pkm) {
        return "gemv_w4_sg16";
    }
    if opt(&pl.gemv_w4_sg_pkm3) {
        return "gemv_w4_sg16_qkv";
    }
    if opt(&pl.lmhead_sg) {
        return "lmhead_bf16_sg";
    }
    if opt(&pl.lmhead_i8) {
        return "lmhead_int8";
    }
    if eq(&pl.gelu_even) {
        return "gelu_mul";
    }
    if eq(&pl.axpby) {
        return "axpby";
    }
    if eq(&pl.gatemul) {
        return "gatemul";
    }
    if eq(&pl.am1) {
        return "argmax_stage1";
    }
    if eq(&pl.am2) {
        return "argmax_stage2";
    }
    if let Some(f) = &pl.fnc {
        if eq(&f.a) {
            return "fused_norm_a";
        }
        if eq(&f.b) {
            return "fused_norm_b";
        }
        if eq(&f.c) {
            return "fused_norm_c";
        }
    }
    if let Some(f) = &pl.fac {
        if eq(&f.q) {
            return "fused_attn_q";
        }
        if eq(&f.k) {
            return "fused_attn_k";
        }
        if eq(&f.v) {
            return "fused_attn_v";
        }
    }
    for mk in [pl.mk.as_ref(), pl.mk_verify.as_ref()]
        .into_iter()
        .flatten()
    {
        if eq(&mk.gather) {
            return "mk_gather";
        }
        if eq(&mk.gemm_bf16_pk) {
            return "mk_gemm_bf16";
        }
        if eq(&mk.gemm_bf16_pk3) {
            return "mk_gemm_bf16_qkv";
        }
        if eq(&mk.gemm_w4_pk) || eq(&mk.gemm_w4_v4_pk) {
            return "mk_gemm_w4";
        }
        if eq(&mk.gemm_w4_pk3) || eq(&mk.gemm_w4_v4_pk3) {
            return "mk_gemm_w4_qkv";
        }
        if opt(&mk.gemm_w4_sg_pk) {
            return "mk_gemm_w4_sg16";
        }
        if opt(&mk.gemm_w4_sg_pk3) {
            return "mk_gemm_w4_sg16_qkv";
        }
        if opt(&mk.gemm_i8_pk) || opt(&mk.gemm_i8g_pk) {
            return "mk_gemm_i8";
        }
        if opt(&mk.gemm_i8_pk3) || opt(&mk.gemm_i8g_pk3) {
            return "mk_gemm_i8_qkv";
        }
        if eq(&mk.flash1) {
            return "mk_flash_stage1";
        }
        if eq(&mk.flash2_pk) {
            return "mk_flash_stage2";
        }
        if eq(&mk.gatemul) {
            return "mk_gatemul";
        }
    }
    "other"
}

fn pass_labels(pl: &Pipelines, passes: &[Pass]) -> Vec<String> {
    passes
        .iter()
        .map(|p| {
            format!(
                "{} [{}x{}x{}]",
                pipeline_name(pl, &p.pipeline),
                p.grid.0,
                p.grid.1,
                p.grid.2
            )
        })
        .collect()
}

fn encode_passes(ctx: &WgpuContext, passes: &[Pass]) -> wgpu::CommandBuffer {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for p in passes {
            pass.set_pipeline(&p.pipeline);
            pass.set_bind_group(0, &p.bind, &[]);
            pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
        }
    }
    enc.finish()
}

fn encode_passes_staged(
    ctx: &WgpuContext,
    passes: &[Pass],
    src: &wgpu::Buffer,
    stage: &wgpu::Buffer,
    bytes: u64,
) -> wgpu::CommandBuffer {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
        let mut pass = enc.begin_compute_pass(&Default::default());
        for p in passes {
            pass.set_pipeline(&p.pipeline);
            pass.set_bind_group(0, &p.bind, &[]);
            pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
        }
    }
    enc.copy_buffer_to_buffer(src, 0, stage, 0, bytes);
    enc.finish()
}

const TOK_STAGE_BYTES: u64 = 64;

fn staged_read_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_STAGED_READ").ok().as_deref() != Some("0")
}

fn spin_read_enabled() -> bool {
    std::env::var("NV_E4B_WGPU_SPIN_READ").ok().as_deref() == Some("1")
}

const SPIN_READ_BUDGET: std::time::Duration = std::time::Duration::from_millis(50);

pub type KvCacheSnapshot = (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>);

pub struct Gemma4E4bWgpu {
    ctx: &'static WgpuContext,
    config: Gemma4Config,
    max_seq: usize,
    pos: usize,
    kv_epoch: u64,
    validated: bool,
    weight_bytes: u64,
    passes: Vec<Pass>,
    prefill: Option<PrefillState>,
    verify: Option<VerifyState>,
    tok_idx: GpuTensor<i32>,
    rope_pos: GpuTensor<i32>,
    kv_start: GpuTensor<i32>,
    fd_sliding: GpuUniform<FdParams>,
    fd_full: GpuUniform<FdParams>,
    fd_sliding_base: FdParams,
    fd_full_base: FdParams,
    token_out: GpuTensor<u32>,
    tok_stage: wgpu::Buffer,
    staged_read: bool,
    spin_read: bool,
    pipe_stage: [wgpu::Buffer; 2],
    pipe_cb: [Option<(SplitArm, wgpu::CommandBuffer)>; 2],
    pipe_parity: usize,
    pipe_read: usize,
    pipe_inflight: usize,
    chain_out: GpuTensor<u32>,
    logits_f32: GpuTensor<f32>,
    kv_layers: Vec<Option<GpuLayerKv>>,
    embed_tab: GpuTable,
    decode_hid: wgpu::Buffer,
    decode_hid_row_words: usize,
    vocab: usize,
    enc_stats: bool,
    enc_us_sum: f64,
    enc_us_min: f64,
    enc_steps: u64,
    preenc: bool,
    pending_cb: Option<(SplitArm, wgpu::CommandBuffer)>,
    pass_labels: Vec<String>,
    deep_split_depth: u32,
    deep_splits_baked: u32,
    deep_overlay: Vec<Option<Pass>>,

    base_passes: usize,
    prof: ProfMode,
    prof_timeline: Vec<(usize, f64, f64)>,
    prof_pass_total_ns: f64,
    lora_passes: usize,
    w4_grain: wk::gemv_w4a16::ScaleGrain,
    w4_census: [usize; 3],
    fnc_unrolled: Option<[bool; 3]>,
    flash1_entry: String,
    flash1_hds: Vec<(u32, usize)>,
    _keep: Vec<Box<dyn std::any::Any>>,
}

pub enum WeightSource<'a> {
    Host(&'a E4bHostWeights),
    Loader(&'a nv_weights::WeightLoader),
}

impl WeightSource<'_> {
    fn embed(&self, config: &Gemma4Config) -> Result<BigBf16<'_>> {
        match self {
            Self::Host(h) => Ok(BigBf16::Bits(std::borrow::Cow::Borrowed(&h.embed))),
            Self::Loader(w) => {
                let name = format!("{PREFIX}.embed_tokens.weight");
                let shape = [config.vocab_size, config.hidden_size];
                if w.st_dtype_of(&name) == Some(nv_weights::StDtype::BF16)
                    && w.shape_of(&name).as_deref() == Some(&shape[..])
                {
                    Ok(BigBf16::Raw(w.raw_bytes(&name)?))
                } else {
                    Ok(BigBf16::Bits(std::borrow::Cow::Owned(bf16_bits_of(
                        w, &name,
                    )?)))
                }
            }
        }
    }

    fn embed_per_layer(&self, config: &Gemma4Config) -> Result<BigBf16<'_>> {
        match self {
            Self::Host(h) => Ok(BigBf16::Bits(std::borrow::Cow::Borrowed(
                &h.embed_per_layer,
            ))),
            Self::Loader(w) => {
                let name = format!("{PREFIX}.embed_tokens_per_layer.weight");
                let elems = config.vocab_size_per_layer()
                    * config.num_hidden_layers
                    * config.hidden_size_per_layer_input;
                if w.st_dtype_of(&name) == Some(nv_weights::StDtype::BF16)
                    && w.shape_of(&name).map(|s| s.iter().product::<usize>()) == Some(elems)
                {
                    Ok(BigBf16::Raw(w.raw_bytes(&name)?))
                } else {
                    Ok(BigBf16::Bits(std::borrow::Cow::Owned(bf16_bits_of(
                        w, &name,
                    )?)))
                }
            }
        }
    }

    fn final_norm(&self) -> Result<std::borrow::Cow<'_, [u16]>> {
        match self {
            Self::Host(h) => Ok(std::borrow::Cow::Borrowed(&h.final_norm[..])),
            Self::Loader(w) => Ok(std::borrow::Cow::Owned(bf16_bits_of(
                w,
                &format!("{PREFIX}.norm.weight"),
            )?)),
        }
    }

    fn per_layer_projection_norm(&self) -> Result<std::borrow::Cow<'_, [u16]>> {
        match self {
            Self::Host(h) => Ok(std::borrow::Cow::Borrowed(&h.per_layer_projection_norm[..])),
            Self::Loader(w) => Ok(std::borrow::Cow::Owned(bf16_bits_of(
                w,
                &format!("{PREFIX}.per_layer_projection_norm.weight"),
            )?)),
        }
    }
}

impl Gemma4E4bWgpu {
    pub fn new(config: Gemma4Config, weights: &E4bHostWeights, max_seq: usize) -> Result<Self> {
        Self::build(config, WeightSource::Host(weights), max_seq, None)
    }

    pub fn new_with_lora(
        config: Gemma4Config,
        weights: &E4bHostWeights,
        max_seq: usize,
        lora: Option<&E4bLora>,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Host(weights), max_seq, lora)
    }

    pub fn from_loader(
        config: Gemma4Config,
        weights: &nv_weights::WeightLoader,
        max_seq: usize,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Loader(weights), max_seq, None)
    }

    pub fn from_loader_with_lora(
        config: Gemma4Config,
        weights: &nv_weights::WeightLoader,
        max_seq: usize,
        lora: Option<&E4bLora>,
    ) -> Result<Self> {
        Self::build(config, WeightSource::Loader(weights), max_seq, lora)
    }

    fn build(
        config: Gemma4Config,
        src: WeightSource<'_>,
        max_seq: usize,
        lora: Option<&E4bLora>,
    ) -> Result<Self> {
        anyhow::ensure!(
            config.has_per_layer_embeddings(),
            "gemma4_e4b_wgpu needs hidden_size_per_layer_input > 0; use gemma4_wgpu for plain Gemma4"
        );
        match &src {
            WeightSource::Host(w) => anyhow::ensure!(
                w.layers.len() == config.num_hidden_layers,
                "gemma4_e4b_wgpu: {} host layers for {} config layers",
                w.layers.len(),
                config.num_hidden_layers
            ),
            WeightSource::Loader(_) => anyhow::ensure!(
                config.tie_word_embeddings,
                "gemma4_e4b_wgpu loader: untied lm_head not wired"
            ),
        }
        let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
        let sg16 = sg16_enabled(ctx);
        let (w4_grain, ckpt_gs) = checkpoint_w4_grain(config.num_hidden_layers, &src);
        let lmhead_i8 = lmhead_i8_enabled();
        let lmhead_sg = !lmhead_i8 && lmhead_sg_enabled(ctx);
        let fuse_norms = fuse_norms_enabled();
        let fuse_attn = fuse_attn_enabled();
        eprintln!(
            "[gemma4_e4b_wgpu] w4 scale grain: {w4_grain:?} (checkpoint group size {ckpt_gs:?})"
        );
        eprintln!(
            "[gemma4_e4b_wgpu] w4 gemv variant: {} (subgroup width {:?}), lm_head: {}, norm fusion: {}, attn fusion: {}",
            if sg16 { "sg16" } else { "v4/block" },
            ctx.subgroup_width(),
            if lmhead_i8 {
                "int8"
            } else if lmhead_sg {
                "bf16-sg"
            } else {
                "bf16"
            },
            if fuse_norms { "on" } else { "off" },
            if fuse_attn { "on" } else { "off" }
        );
        let mut prefill_m = prefill_m_from_env();
        if max_seq < prefill_m {
            prefill_m = 0;
        }
        if prefill_m > PREFILL_SLAB {
            let align = ctx.device.limits().min_storage_buffer_offset_alignment as u64;
            let hd_s = config.head_dim_for(LayerType::SlidingAttention);
            let hd_f = config.head_dim_for(LayerType::FullAttention);
            let nq = config.num_attention_heads;
            let strides = [
                config.hidden_size / 2,
                nq * hd_s / 2,
                nq * hd_f / 2,
                config.num_kv_heads_for(LayerType::SlidingAttention) * hd_s / 2,
                config.num_kv_heads_for(LayerType::FullAttention) * hd_f / 2,
                config.intermediate_size / 2,
                config.hidden_size_per_layer_input / 2,
            ];
            let ok = strides
                .iter()
                .all(|&w| (PREFILL_SLAB as u64 * 4 * w as u64).is_multiple_of(align));
            if !ok {
                eprintln!(
                    "[gemma4_e4b_wgpu] prefill m {prefill_m} clamped to {PREFILL_SLAB}: slab bind offsets misaligned for this shape"
                );
                prefill_m = PREFILL_SLAB;
            }
        }
        let verify_m = verify_m_from_env();
        let pl = build_pipelines(
            ctx,
            sg16,
            w4_grain,
            lmhead_i8,
            lmhead_sg,
            prefill_m,
            verify_m,
            fuse_norms,
            fuse_attn,
            config.hidden_size,
            &flash1_head_dims(&config),
            gqa_group_of(&config),
        )?;
        let fnc_unrolled = pl.fnc.as_ref().map(|f| f.unrolled);
        eprintln!(
            "[gemma4_e4b_wgpu] norm chain: {}",
            match fnc_unrolled {
                None => "unfused (rms + rmsres per site)".to_string(),
                Some(m) => format!(
                    "fused; a/b/c unrolled at hidden = {}/{}/{}",
                    m[0], m[1], m[2]
                ),
            }
        );
        if let Some(v) = pl.mk_verify.as_ref() {
            eprintln!(
                "[gemma4_e4b_wgpu] verify m-row width {} (prefill {}), separate verify pass list",
                v.rows,
                prefill_m.min(PREFILL_SLAB)
            );
        }
        if let Some(l) = lora {
            anyhow::ensure!(
                l.layers.len() == config.num_hidden_layers,
                "lora adapter carries {} layers for {} config layers",
                l.layers.len(),
                config.num_hidden_layers
            );
            eprintln!(
                "[gemma4_e4b_wgpu] lora: rank {} matched {} modules over {} layers, +{} passes/token",
                l.rank,
                l.matched,
                l.layers.len(),
                l.total_pass_count()
            );
        }
        let lora_pl = match lora {
            Some(_) => Some(build_lora_pipelines(ctx)?),
            None => None,
        };
        let mut b = Builder {
            ctx,
            pl: &pl,
            passes: Vec::new(),
            prefill_passes: Vec::new(),
            verify_prefill_passes: Vec::new(),
            to_prefill: false,
            to_verify: false,
            keep: Vec::new(),
            weight_bytes: 0,
            sg16,
            w4_census: [0; 3],
            mk_prefill_unis: Vec::new(),
            mk_verify_unis: Vec::new(),
            deep_attn: Vec::new(),
        };

        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let eps = config.rms_norm_eps as f32;
        let n_q = config.num_attention_heads;
        let n_layers = config.num_hidden_layers;
        let hpl = config.hidden_size_per_layer_input;
        let ple_row = n_layers * hpl;
        anyhow::ensure!(
            hidden.is_multiple_of(8)
                && inter.is_multiple_of(2)
                && vocab.is_multiple_of(8)
                && hpl.is_multiple_of(8),
            "gemma4_e4b_wgpu shape rule: hidden%8, inter%2, vocab%8, hpl%8"
        );
        anyhow::ensure!(ple_row.is_multiple_of(2), "n_layers*hpl must be even");

        let hd_s = config.head_dim_for(LayerType::SlidingAttention);
        let hd_f = config.head_dim_for(LayerType::FullAttention);
        let nkv_s = config.num_kv_heads_for(LayerType::SlidingAttention);
        let nkv_f = config.num_kv_heads_for(LayerType::FullAttention);
        let hd_max = hd_s.max(hd_f);
        let q_dim_max = n_q * hd_max;
        let kv_dim_max = (nkv_s * hd_s).max(nkv_f * hd_f);
        anyhow::ensure!(hd_max <= wk::flash_decode::MAX_HEAD_DIM);

        let embed_src = src.embed(&config)?;
        let ple_src = src.embed_per_layer(&config)?;
        let embed_tab = GpuTable::upload(ctx, "e4bw-embed", &embed_src, vocab, hidden)?;
        let ple_tab = GpuTable::upload(
            ctx,
            "e4bw-ple",
            &ple_src,
            config.vocab_size_per_layer(),
            ple_row,
        )?;
        if lmhead_i8 {
            anyhow::ensure!(
                hidden.is_multiple_of(16),
                "int8 lm_head needs hidden % 16 == 0, got {hidden}"
            );
            b.weight_bytes += (vocab as u64) * (hidden as u64) + 4 * (vocab as u64);
        } else {
            b.weight_bytes += 2 * (vocab as u64) * (hidden as u64);
        }
        b.weight_bytes += 2 * (ple_row as u64);
        let mut lm_i8_chunks: Vec<(GpuTensor<u32>, GpuTensor<f32>)> = Vec::new();
        if lmhead_i8 {
            for c in 0..embed_tab.plan.n_chunks {
                let lo = c * embed_tab.plan.rows_per_chunk;
                let rows = embed_tab.chunk_rows(c);
                let bits = embed_src.bits_chunk(lo * hidden, rows * hidden);
                let (packed, scales) = wk::quant_gemv::quantize_rows_int8(&bits, rows, hidden);
                let w = GpuTensor::upload(ctx, "e4bw-lm-i8-w", &packed);
                let s = GpuTensor::upload(ctx, "e4bw-lm-i8-s", &scales);
                ctx.queue.submit(std::iter::empty());
                ctx.poll_blocking().map_err(err)?;
                lm_i8_chunks.push((w, s));
            }
        }

        let final_norm = GpuTensor::upload(ctx, "e4bw-final-norm", &pack_pairs(&src.final_norm()?));
        let plp_norm = GpuTensor::upload(
            ctx,
            "e4bw-plp-norm",
            &pack_pairs(&src.per_layer_projection_norm()?),
        );
        let plmp_owned;
        let plmp_host: &HostLin = match &src {
            WeightSource::Host(w) => &w.per_layer_model_projection,
            WeightSource::Loader(l) => {
                plmp_owned = load_lin(
                    l,
                    &format!("{PREFIX}.per_layer_model_projection"),
                    ple_row,
                    hidden,
                )?;
                &plmp_owned
            }
        };
        anyhow::ensure!(
            plmp_host.n == ple_row && plmp_host.k == hidden,
            "per_layer_model_projection must be [{ple_row}, {hidden}], got [{}, {}]",
            plmp_host.n,
            plmp_host.k
        );
        let plmp = upload_proj(ctx, &mut b, "e4bw-plmp", plmp_host)?;

        let cfg = EmitCfg {
            hidden,
            inter,
            eps,
            n_q,
            n_layers,
            hpl,
            ple_row,
            max_seq,
        };
        let bufs = alloc_bufs(ctx, 1, &cfg, q_dim_max, kv_dim_max, hd_max);
        let pf_bufs = if prefill_m > 0 {
            Some(alloc_bufs(
                ctx, prefill_m, &cfg, q_dim_max, kv_dim_max, hd_max,
            ))
        } else {
            None
        };
        let lora_scratch = lora.map(|l| {
            let w = l.max_row_width().max(2);
            (
                GpuTensor::<u32>::zeroed(ctx, "e4bw-lora-scratch", w),
                prefill_m.max(1).saturating_mul(w),
            )
        });
        let (lora_scratch, pf_lora_scratch) = match lora_scratch {
            Some((decode, pf_len)) => {
                let pf = if prefill_m > 0 {
                    Some(GpuTensor::<u32>::zeroed(
                        ctx,
                        "e4bw-lora-scratch-pf",
                        pf_len,
                    ))
                } else {
                    None
                };
                (Some(decode), pf)
            }
            None => (None, None),
        };
        let logits_pk = GpuTensor::<u32>::zeroed(ctx, "e4bw-logits-pk", vocab / 2);
        let logits_f32 = GpuTensor::<f32>::zeroed(ctx, "e4bw-logits-f32", vocab);
        let am_val = GpuTensor::<f32>::zeroed(ctx, "e4bw-am-val", wk::graph_decode::ARGMAX_BLOCKS);
        let am_idx = GpuTensor::<i32>::zeroed(ctx, "e4bw-am-idx", wk::graph_decode::ARGMAX_BLOCKS);
        let token_out =
            GpuTensor::<u32>::zeroed(ctx, "e4bw-token-out", wk::gemm_bf16_small_m::MAX_M as usize);
        let mk_stage = |label| {
            ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: TOK_STAGE_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let tok_stage = mk_stage("e4bw-token-stage");
        let pipe_stage = [mk_stage("e4bw-pipe-stage-0"), mk_stage("e4bw-pipe-stage-1")];
        let chain_out = GpuTensor::<u32>::zeroed(ctx, "e4bw-chain-out", MAX_CHAIN);

        let mrows = prefill_m.max(1);
        let tok_idx = GpuTensor::<i32>::upload(ctx, "e4bw-tok-idx", &vec![0; mrows]);
        let rope_pos = GpuTensor::<i32>::upload(ctx, "e4bw-rope-pos", &vec![0; mrows]);
        let kv_start = GpuTensor::<i32>::upload(ctx, "e4bw-kv-start", &[0]);

        let mk_fd = |kind: LayerType| FdParams {
            n_heads: n_q as u32,
            n_kv: config.num_kv_heads_for(kind) as u32,
            head_dim: config.head_dim_for(kind) as u32,
            total: 0,
            start: 0,
            splits: FLASH_SPLITS,
            ring: 0,
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
        let mut fd_sliding_base = mk_fd(LayerType::SlidingAttention);
        let mut fd_full_base = mk_fd(LayerType::FullAttention);
        fd_sliding_base.splits = decode_splits();
        fd_full_base.splits = decode_splits();
        let fd_sliding = GpuUniform::new(ctx, "e4bw-fd-s", &fd_sliding_base);
        let fd_full = GpuUniform::new(ctx, "e4bw-fd-f", &fd_full_base);
        let pf_fd = if prefill_m > 0 {
            let mut s = mk_fd(LayerType::SlidingAttention);
            s.m_rows = prefill_m as u32;
            s.window = config.sliding_window as u32;
            let mut f = mk_fd(LayerType::FullAttention);
            f.m_rows = prefill_m as u32;
            Some((
                GpuUniform::new(ctx, "e4bw-pf-fd-s", &s),
                GpuUniform::new(ctx, "e4bw-pf-fd-f", &f),
                s,
                f,
            ))
        } else {
            None
        };

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
                GpuTensor::upload(ctx, "e4bw-rope-cos", &cos),
                GpuTensor::upload(ctx, "e4bw-rope-sin", &sin),
            ));
        }

        let mut kv_layers: Vec<Option<GpuLayerKv>> = Vec::with_capacity(n_layers);
        for (li, kind) in config.layer_types.iter().enumerate() {
            if config.kv_source_layer(li).is_some() {
                kv_layers.push(None);
                continue;
            }
            let hd = config.head_dim_for(*kind);
            let nkv = config.num_kv_heads_for(*kind);
            kv_layers.push(Some(GpuLayerKv {
                k_fp8: GpuTensor::zeroed(ctx, "e4bw-kc", max_seq * nkv * hd / 4),
                v_fp8: GpuTensor::zeroed(ctx, "e4bw-vc", max_seq * nkv * hd / 4),
                k_scales: GpuTensor::zeroed(ctx, "e4bw-ks", max_seq * nkv),
                v_scales: GpuTensor::zeroed(ctx, "e4bw-vs", max_seq * nkv),
            }));
        }

        emit_prologue(
            &mut b,
            &bufs,
            &cfg,
            &embed_tab,
            &ple_tab,
            &plmp,
            plp_norm.raw(),
            tok_idx.raw(),
        );
        if let Some(pf) = &pf_bufs {
            b.to_prefill = true;
            emit_prologue(
                &mut b,
                pf,
                &cfg,
                &embed_tab,
                &ple_tab,
                &plmp,
                plp_norm.raw(),
                tok_idx.raw(),
            );
            if b.wants_verify_list() {
                b.to_verify = true;
                emit_prologue(
                    &mut b,
                    pf,
                    &cfg,
                    &embed_tab,
                    &ple_tab,
                    &plmp,
                    plp_norm.raw(),
                    tok_idx.raw(),
                );
                b.to_verify = false;
            }
            b.to_prefill = false;
        }

        let mut ln_in_all: Vec<GpuTensor<u32>> = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let bits: std::borrow::Cow<'_, [u16]> = match &src {
                WeightSource::Host(w) => std::borrow::Cow::Borrowed(&w.layers[li].input_ln[..]),
                WeightSource::Loader(l) => std::borrow::Cow::Owned(bf16_bits_of(
                    l,
                    &format!("{PREFIX}.layers.{li}.input_layernorm.weight"),
                )?),
            };
            anyhow::ensure!(
                bits.len() == hidden,
                "layer {li}: input_layernorm must be {hidden} long, got {}",
                bits.len()
            );
            ln_in_all.push(GpuTensor::upload(ctx, "e4bw-ln-in", &pack_pairs(&bits)));
        }

        for li in 0..n_layers {
            let owned_layer;
            let hl: &E4bHostLayer = match &src {
                WeightSource::Host(w) => &w.layers[li],
                WeightSource::Loader(l) => {
                    owned_layer = load_e4b_layer_from_loader(&config, l, li)?;
                    eprintln!("[gemma4_e4b_wgpu] streamed layer {li}/{n_layers}");
                    &owned_layer
                }
            };
            let kind = hl.kind;
            let hd = config.head_dim_for(kind);
            let nkv = config.num_kv_heads_for(kind);
            let q_dim = n_q * hd;
            let kv_dim = nkv * hd;
            let shared = hl.kv_source;
            anyhow::ensure!(
                shared == config.kv_source_layer(li),
                "layer {li}: host kv_source {:?} disagrees with config {:?}",
                shared,
                config.kv_source_layer(li)
            );
            let want_qkv_rows = match shared {
                Some(_) => q_dim,
                None => q_dim + kv_dim * if hl.has_v { 2 } else { 1 },
            };
            anyhow::ensure!(
                hl.qkv.n == want_qkv_rows && hl.qkv.k == hidden,
                "layer {li}: qkv is [{}, {}], want [{want_qkv_rows}, {hidden}]",
                hl.qkv.n,
                hl.qkv.k
            );
            anyhow::ensure!(
                hl.o.k == q_dim && hl.o.n == hidden,
                "layer {li}: o_proj is [{}, {}], want [{hidden}, {q_dim}]",
                hl.o.n,
                hl.o.k
            );
            anyhow::ensure!(hl.gate_up.n == 2 * inter && hl.gate_up.k == hidden);
            anyhow::ensure!(hl.down.n == hidden && hl.down.k == inter);
            anyhow::ensure!(
                hl.per_layer_input_gate.n == hpl && hl.per_layer_input_gate.k == hidden,
                "layer {li}: per_layer_input_gate must be [{hpl}, {hidden}]"
            );
            anyhow::ensure!(
                hl.per_layer_projection.n == hidden && hl.per_layer_projection.k == hpl,
                "layer {li}: per_layer_projection must be [{hidden}, {hpl}]"
            );

            let qkv = upload_proj(ctx, &mut b, "e4bw-qkv", &hl.qkv)?;
            let o = upload_proj(ctx, &mut b, "e4bw-o", &hl.o)?;
            let gate_up = upload_proj(ctx, &mut b, "e4bw-gate-up", &hl.gate_up)?;
            let down = upload_proj(ctx, &mut b, "e4bw-down", &hl.down)?;
            let pl_gate_proj = upload_proj(ctx, &mut b, "e4bw-plig", &hl.per_layer_input_gate)?;
            let pl_proj = upload_proj(ctx, &mut b, "e4bw-plp", &hl.per_layer_projection)?;

            let ln_pa = GpuTensor::upload(ctx, "e4bw-ln-pa", &pack_pairs(&hl.post_attn_ln));
            let ln_pf = GpuTensor::upload(ctx, "e4bw-ln-pf", &pack_pairs(&hl.pre_ff_ln));
            let ln_po = GpuTensor::upload(ctx, "e4bw-ln-po", &pack_pairs(&hl.post_ff_ln));
            let ln_pli = GpuTensor::upload(
                ctx,
                "e4bw-ln-pli",
                &pack_pairs(&hl.post_per_layer_input_norm),
            );
            let qn = GpuTensor::upload(ctx, "e4bw-qn", &pack_pairs(&hl.q_norm));
            let ones: Vec<u16> = vec![bf16_bits(1.0); hd];
            let kn = if shared.is_some() {
                GpuTensor::upload(ctx, "e4bw-kn", &pack_pairs(&ones))
            } else {
                anyhow::ensure!(
                    hl.k_norm.len() == hd,
                    "layer {li}: k_norm must be {hd} long"
                );
                GpuTensor::upload(ctx, "e4bw-kn", &pack_pairs(&hl.k_norm))
            };
            let vn = GpuTensor::upload(ctx, "e4bw-vn", &pack_pairs(&ones));

            let kidx = match kind {
                LayerType::SlidingAttention => 0usize,
                LayerType::FullAttention => 1,
            };
            let (cos, sin) = (&rope_bufs[kidx].0, &rope_bufs[kidx].1);
            let kv_idx = shared.unwrap_or(li);
            let kv = kv_layers[kv_idx]
                .as_ref()
                .with_context(|| format!("layer {li}: kv source {kv_idx} has no cache"))?;
            let fd = match kind {
                LayerType::SlidingAttention => fd_sliding.raw(),
                LayerType::FullAttention => fd_full.raw(),
            };
            let mut le = LayerEmit {
                bufs: &bufs,
                cfg: &cfg,
                li,
                shared,
                full_attn: matches!(kind, LayerType::FullAttention),
                has_v: hl.has_v,
                layer_scalar: hl.layer_scalar,
                hd,
                nkv,
                q_dim,
                kv_dim,
                qkv: &qkv,
                o: &o,
                gate_up: &gate_up,
                down: &down,
                pl_gate_proj: &pl_gate_proj,
                pl_proj: &pl_proj,
                ln_in: ln_in_all[li].raw(),
                ln_pa: ln_pa.raw(),
                ln_pf: ln_pf.raw(),
                ln_po: ln_po.raw(),
                ln_pli: ln_pli.raw(),
                ln_next: if li + 1 < n_layers {
                    ln_in_all[li + 1].raw()
                } else {
                    final_norm.raw()
                },
                qn: qn.raw(),
                kn: kn.raw(),
                vn: vn.raw(),
                cos: cos.raw(),
                sin: sin.raw(),
                kv,
                fd,
                rope_pos: rope_pos.raw(),
                kv_start: kv_start.raw(),
                lora: None,
                lora_pl: lora_pl.as_ref(),
                lora_scratch: lora_scratch.as_ref().map(|t| t.raw()),
            };
            let layer_lora = match lora {
                Some(l) => Some(upload_lora_layer(ctx, &l.layers[li])?),
                None => None,
            };
            le.lora = layer_lora.as_ref();
            emit_layer(&mut b, &le);
            if let Some(pf) = &pf_bufs {
                let (pf_s, pf_f, _, _) = pf_fd.as_ref().unwrap();
                le.bufs = pf;
                le.fd = match kind {
                    LayerType::SlidingAttention => pf_s.raw(),
                    LayerType::FullAttention => pf_f.raw(),
                };
                le.lora_scratch = pf_lora_scratch.as_ref().map(|t| t.raw());
                b.to_prefill = true;
                emit_layer(&mut b, &le);
                if b.wants_verify_list() {
                    b.to_verify = true;
                    emit_layer(&mut b, &le);
                    b.to_verify = false;
                }
                b.to_prefill = false;
            }

            b.keep.push(Box::new(layer_lora));
            b.keep.push(Box::new((qkv, o, gate_up, down)));
            b.keep.push(Box::new((pl_gate_proj, pl_proj)));
            b.keep
                .push(Box::new((ln_pa, ln_pf, ln_po, ln_pli, qn, kn, vn)));

            ctx.queue.submit(std::iter::empty());
            ctx.poll_blocking().map_err(err)?;
        }

        let decode_hid = if n_layers.is_multiple_of(2) {
            bufs.hid_a.raw().clone()
        } else {
            bufs.hid_b.raw().clone()
        };
        if b.pl.fnc.is_none() {
            let final_hid = if n_layers.is_multiple_of(2) {
                bufs.hid_a.raw()
            } else {
                bufs.hid_b.raw()
            };
            b.rms(final_hid, final_norm.raw(), bufs.t0.raw(), 1, hidden, eps);
        }

        let mut row_off = 0usize;
        let lm_sg_rows = wk::gemv_bf16::sg_pk_entry(lmhead_sg_wg()).1;
        #[allow(clippy::needless_range_loop)]
        for c in 0..embed_tab.plan.n_chunks {
            let rows = embed_tab.chunk_rows(c);
            anyhow::ensure!(rows % 2 == 0, "lm_head chunk {c} has odd row count {rows}");
            let lm_grid = b.grid_1d(rows as u64, wk::gemv_bf16::ROWS_PER_GROUP);
            if lmhead_i8 {
                let (wq, sc) = &lm_i8_chunks[c];
                let lm_p = b.uni(
                    "e4bw-lm-i8-p",
                    I8hParams {
                        n_rows: rows as u32,
                        k_elems: hidden as u32,
                        groups_x: lm_grid.0,
                        dst_word_off: (row_off / 2) as u32,
                    },
                );
                let pipeline = pl
                    .lmhead_i8
                    .as_ref()
                    .expect("int8 lm_head routing without pipeline")
                    .clone();
                b.push(
                    pipeline,
                    &[
                        (0, wq.raw()),
                        (1, sc.raw()),
                        (2, bufs.t0.raw()),
                        (3, logits_pk.raw()),
                        (4, &lm_p),
                    ],
                    lm_grid,
                );
            } else if let Some(sg_pl) = pl.lmhead_sg.as_ref() {
                let sg_grid = b.grid_1d(rows as u64, lm_sg_rows);
                let lm_p = b.uni(
                    "e4bw-lm-sg-p",
                    GemvBf16Params {
                        n_rows: rows as u32,
                        k_elems: hidden as u32,
                        w_row_words: (hidden / 2) as u32,
                        groups_x: sg_grid.0,
                    },
                );
                let off = b.uni(
                    "e4bw-lm-sg-off",
                    PkOffParams {
                        dst_word_off: (row_off / 2) as u32,
                        ..Default::default()
                    },
                );
                b.push(
                    sg_pl.clone(),
                    &[
                        (0, &embed_tab.chunks[c]),
                        (1, bufs.t0.raw()),
                        (2, logits_pk.raw()),
                        (3, &lm_p),
                        (30, &off),
                    ],
                    sg_grid,
                );
            } else {
                let lm_p = b.uni(
                    "e4bw-lm-p",
                    GemvBf16Params {
                        n_rows: rows as u32,
                        k_elems: hidden as u32,
                        w_row_words: (hidden / 2) as u32,
                        groups_x: lm_grid.0,
                    },
                );
                let off = b.uni(
                    "e4bw-lm-pk-off",
                    PkOffParams {
                        dst_word_off: (row_off / 2) as u32,
                        ..Default::default()
                    },
                );
                b.push(
                    pl.gemv_pk.clone(),
                    &[
                        (0, &embed_tab.chunks[c]),
                        (1, bufs.t0.raw()),
                        (2, logits_pk.raw()),
                        (3, &lm_p),
                        (30, &off),
                    ],
                    lm_grid,
                );
            }
            row_off += rows;
        }
        anyhow::ensure!(
            row_off == vocab,
            "lm_head chunks covered {row_off} of {vocab}"
        );

        let cap = config.final_logit_softcapping;
        let softcap_on = cap > 0.0 && cap.is_finite();
        let am_p = b.uni(
            "e4bw-am-p",
            ArgmaxRowsParams {
                rows: 1,
                n: vocab as u32,
                pad0: 0,
                pad1: 0,
            },
        );
        if fuse_head_argmax_enabled() {
            let amc_pl = dispatch::cached_compute_pipeline(
                ctx,
                "e4bw-am1-cap",
                &compose(wk::graph_decode::WGSL),
                wk::graph_decode::ARGMAX_SOFTCAP_STAGE1_ENTRY,
            )
            .map_err(err)?;
            let amc_p = b.uni(
                "e4bw-amc-p",
                wk::graph_decode::argmax_softcap_cap_params(vocab, cap),
            );
            b.push(
                amc_pl,
                &[
                    (65, logits_pk.raw()),
                    (66, logits_f32.raw()),
                    (55, am_val.raw()),
                    (56, am_idx.raw()),
                    (67, &amc_p),
                ],
                (wk::graph_decode::ARGMAX_BLOCKS as u32, 1, 1),
            );
        } else {
            let cap_p = b.uni(
                "e4bw-cap-p",
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
        }
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

        let verify = match pf_bufs.as_ref() {
            Some(pf) if !lmhead_i8 && !lmhead_sg && hidden.is_multiple_of(8) => {
                let vrows = pl
                    .mk_verify
                    .as_ref()
                    .map(|v| v.rows)
                    .unwrap_or(prefill_m)
                    .min(prefill_m)
                    .min(wk::gemm_bf16_small_m::MAX_M as usize);
                let vstart = b.prefill_passes.len();
                b.to_prefill = true;
                let final_hid = if n_layers.is_multiple_of(2) {
                    pf.hid_a.raw()
                } else {
                    pf.hid_b.raw()
                };
                b.rms(
                    final_hid,
                    final_norm.raw(),
                    pf.t0.raw(),
                    prefill_m,
                    hidden,
                    eps,
                );
                let sm_src = verify_smk_live_source(vrows);
                let sm_pl =
                    nozi_all_pipeline(ctx, "e4bw-verify-lm-smk-live", &sm_src, SMK_LIVE_ENTRY)?;
                let mut sm_unis: Vec<(wgpu::Buffer, SmkLiveParams)> = Vec::new();
                let cap_pl = dispatch::cached_compute_pipeline(
                    ctx,
                    "e4bw-verify-cap",
                    &compose(VERIFY_CAP_WGSL),
                    "e4b_verify_cap_rows",
                )
                .map_err(err)?;
                let v_logits = GpuTensor::<f32>::zeroed(ctx, "e4bw-verify-logits", vrows * vocab);
                let v_val = GpuTensor::<f32>::zeroed(
                    ctx,
                    "e4bw-verify-am-val",
                    vrows * wk::graph_decode::ARGMAX_BLOCKS,
                );
                let v_idx = GpuTensor::<i32>::zeroed(
                    ctx,
                    "e4bw-verify-am-idx",
                    vrows * wk::graph_decode::ARGMAX_BLOCKS,
                );
                let mut v_y = Vec::new();
                let mut v_off = 0usize;
                for c in 0..embed_tab.plan.n_chunks {
                    let rows_c = embed_tab.chunk_rows(c);
                    let y = GpuTensor::<u32>::zeroed(ctx, "e4bw-verify-y", vrows * rows_c);
                    let grid = dispatch::workgroup_count_1d(
                        ctx,
                        rows_c as u64,
                        wk::gemm_bf16_small_m::ROWS_PER_GROUP,
                    );
                    let sm_p_val = SmkLiveParams {
                        n_rows: rows_c as u32,
                        k_elems: hidden as u32,
                        row_words: (hidden / 2) as u32,
                        groups_x: grid.0,
                        m_live: vrows as u32,
                        pad0: 0,
                        pad1: 0,
                        pad2: 0,
                    };
                    let sm_p = b.uni("e4bw-verify-smk-p", sm_p_val);
                    sm_unis.push((sm_p.clone(), sm_p_val));
                    b.push(
                        sm_pl.clone(),
                        &[
                            (0, &embed_tab.chunks[c]),
                            (1, pf.t0.raw()),
                            (2, y.raw()),
                            (3, &sm_p),
                        ],
                        grid,
                    );
                    let cap_p = b.uni(
                        "e4bw-verify-cap-p",
                        VerifyCapParams {
                            n_rows: rows_c as u32,
                            row_off: v_off as u32,
                            vocab: vocab as u32,
                            m: vrows as u32,
                            cap,
                            inv_cap: if softcap_on { 1.0 / cap } else { 0.0 },
                            softcap: softcap_on as u32,
                            pad0: 0,
                        },
                    );
                    let cgrid = b.grid_1d((vrows * rows_c) as u64, 256);
                    b.push(
                        cap_pl.clone(),
                        &[(0, y.raw()), (1, v_logits.raw()), (2, &cap_p)],
                        cgrid,
                    );
                    v_y.push(y);
                    v_off += rows_c;
                }
                let vam_p = b.uni(
                    "e4bw-verify-am-p",
                    ArgmaxRowsParams {
                        rows: vrows as u32,
                        n: vocab as u32,
                        pad0: 0,
                        pad1: 0,
                    },
                );
                b.push(
                    pl.am1.clone(),
                    &[
                        (54, v_logits.raw()),
                        (55, v_val.raw()),
                        (56, v_idx.raw()),
                        (58, &vam_p),
                    ],
                    (wk::graph_decode::ARGMAX_BLOCKS as u32, vrows as u32, 1),
                );
                b.push(
                    pl.am2.clone(),
                    &[
                        (55, v_val.raw()),
                        (56, v_idx.raw()),
                        (57, token_out.raw()),
                        (58, &vam_p),
                    ],
                    (vrows as u32, 1, 1),
                );
                b.to_prefill = false;
                let vpasses = b.prefill_passes.split_off(vstart);
                let vlabels = pass_labels(&pl, &vpasses);
                let vpf = if b.verify_prefill_passes.is_empty() {
                    None
                } else {
                    Some(std::mem::take(&mut b.verify_prefill_passes))
                };
                let vpf_labels = vpf
                    .as_ref()
                    .map(|p| pass_labels(&pl, p))
                    .unwrap_or_default();
                b.keep.push(Box::new((v_logits, v_val, v_idx, v_y)));
                Some(VerifyState {
                    rows: vrows,
                    passes: vpasses,
                    labels: vlabels,
                    pf_passes: vpf,
                    pf_labels: vpf_labels,
                    validated: false,
                    pending_cb: None,
                    pending_pf_cb: None,
                    hid: final_hid.clone(),
                    hid_row_words: hidden / 2,
                    sm_unis,
                    rows_live: vrows,
                })
            }
            _ => None,
        };

        b.keep
            .push(Box::new((lora_scratch, pf_lora_scratch, lora_pl)));
        b.keep.push(Box::new(ple_tab));
        b.keep.push(Box::new((final_norm, plp_norm, plmp)));
        b.keep.push(Box::new(ln_in_all));
        b.keep.push(Box::new(bufs));
        b.keep.push(Box::new(pf_bufs));
        b.keep.push(Box::new((logits_pk, am_val, am_idx)));
        b.keep.push(Box::new(rope_bufs));
        b.keep.push(Box::new(lm_i8_chunks));

        let Builder {
            passes,
            prefill_passes,
            keep,
            weight_bytes,
            w4_census,
            mk_prefill_unis,
            mk_verify_unis,
            deep_attn,
            ..
        } = b;
        let deep_arm = deep_split_arm();
        if deep_arm.is_some() {
            assert_eq!(
                deep_attn.len(),
                2 * config.num_hidden_layers,
                "deep split arm must shadow exactly stage1+stage2 in each of {} decode layers; \
                 {DEEP_SPLIT_ARM_RULE}",
                config.num_hidden_layers
            );
        } else {
            assert!(
                deep_attn.is_empty(),
                "deep twins recorded while the deep split arm is disabled"
            );
        }
        let mut deep_overlay: Vec<Option<Pass>> = vec![None; passes.len()];
        for (i, p) in deep_attn {
            assert!(
                deep_overlay[i].replace(p).is_none(),
                "deep split-arm overlay would shadow decode pass {i} twice"
            );
        }
        if let Some((depth, splits)) = deep_arm {
            eprintln!(
                "[gemma4_e4b_wgpu] decode split-k arms: shallow {} / deep {splits} past total \
                 {depth}",
                decode_splits()
            );
        }
        eprintln!(
            "[gemma4_e4b_wgpu] w4 projections routed: {} block / {} v4 / {} sg16",
            w4_census[0], w4_census[1], w4_census[2]
        );
        let labels = pass_labels(&pl, &passes);
        let pf_labels = pass_labels(&pl, &prefill_passes);
        let flash1_hds: Vec<(u32, usize)> = pl
            .flash1
            .iter()
            .map(|(hd, p)| {
                let n = passes
                    .iter()
                    .filter(|q| Arc::ptr_eq(&q.pipeline, p))
                    .count()
                    + pl
                        .flash1_fold
                        .iter()
                        .filter(|(fhd, _)| fhd == hd)
                        .map(|(_, fp)| {
                            passes
                                .iter()
                                .filter(|q| Arc::ptr_eq(&q.pipeline, fp))
                                .count()
                        })
                        .sum::<usize>();
                (*hd, n)
            })
            .collect();
        let dispatched: usize = flash1_hds.iter().map(|(_, n)| n).sum();
        assert_eq!(
            dispatched, config.num_hidden_layers,
            "flash stage1 reaches {dispatched} of {} decode layers; census {flash1_hds:?} -- \
             this census must count BOTH pl.flash1 and pl.flash1_fold, because with gqa_fold > 1 \
             decode dispatches the folded stage1 and a census that follows one spelling of the \
             dispatch reports zero reached layers the moment the other becomes the default",
            config.num_hidden_layers
        );
        eprintln!("[gemma4_e4b_wgpu] flash stage1 dispatches by head_dim: {flash1_hds:?}");
        let prefill = pf_fd.map(|(fd_s, fd_f, fd_s_base, fd_f_base)| PrefillState {
            m: prefill_m,
            tail: prefill_tail_enabled(),
            passes: prefill_passes,
            labels: pf_labels,
            fd_sliding: fd_s,
            fd_full: fd_f,
            fd_sliding_base: fd_s_base,
            fd_full_base: fd_f_base,
            validated: false,
            pending_cb: None,
            mk_unis: mk_prefill_unis,
            mk_verify_unis,
            rows_live: prefill_m,
            rows_live_verify: prefill_m,
            live_rows_enabled: verify_live_rows_enabled(),
        });
        Ok(Self {
            ctx,
            config,
            max_seq,
            pos: 0,
            kv_epoch: 0,
            validated: false,
            weight_bytes,
            passes,
            prefill,
            verify,
            tok_idx,
            rope_pos,
            kv_start,
            fd_sliding,
            fd_full,
            fd_sliding_base,
            fd_full_base,
            token_out,
            tok_stage,
            staged_read: staged_read_enabled(),
            spin_read: spin_read_enabled(),
            pipe_stage,
            pipe_cb: [None, None],
            pipe_parity: 0,
            pipe_read: 0,
            pipe_inflight: 0,
            chain_out,
            logits_f32,
            kv_layers,
            embed_tab,
            decode_hid,
            decode_hid_row_words: hidden / 2,
            vocab,
            enc_stats: std::env::var("NV_E4B_WGPU_ENCODE_STATS").ok().as_deref() == Some("1"),
            enc_us_sum: 0.0,
            enc_us_min: f64::INFINITY,
            enc_steps: 0,
            preenc: std::env::var("NV_E4B_WGPU_PREENC").ok().as_deref() != Some("0"),
            pending_cb: None,
            deep_split_depth: deep_arm.map(|(d, _)| d).unwrap_or(0),
            deep_splits_baked: deep_arm.map(|(_, s)| s).unwrap_or_else(decode_splits),
            deep_overlay,
            base_passes: labels.len(),
            pass_labels: labels,
            prof: if dispatch::profile::enabled() && ctx.caps.timestamp_query {
                ProfMode::PerDispatch
            } else {
                ProfMode::Off
            },
            prof_timeline: Vec::new(),
            prof_pass_total_ns: 0.0,
            lora_passes: lora.map(|l| l.total_pass_count()).unwrap_or(0),
            w4_grain,
            w4_census,
            fnc_unrolled,
            flash1_entry: pl.flash1_entry.clone(),
            flash1_hds,
            _keep: keep,
        })
    }

    fn arm_for_total(&self, total: usize) -> SplitArm {
        if self.deep_split_depth > 0 && total > self.deep_split_depth as usize {
            SplitArm::Deep
        } else {
            SplitArm::Shallow
        }
    }

    fn active_arm(&self) -> SplitArm {
        self.arm_for_total(self.pos + 1)
    }

    fn arm_splits(&self, arm: SplitArm) -> u32 {
        match arm {
            SplitArm::Shallow => self.fd_full_base.splits,
            SplitArm::Deep => self.deep_splits_baked,
        }
    }

    fn arm_pass(&self, i: usize, arm: SplitArm) -> &Pass {
        if arm == SplitArm::Deep {
            if let Some(Some(p)) = self.deep_overlay.get(i) {
                return p;
            }
        }
        &self.passes[i]
    }

    fn take_arm_cb(
        cache: &mut Option<(SplitArm, wgpu::CommandBuffer)>,
        arm: SplitArm,
    ) -> Option<wgpu::CommandBuffer> {
        match cache.take() {
            Some((cached, cb)) if cached == arm => Some(cb),
            _ => None,
        }
    }

    fn encode_cb(&self, arm: SplitArm) -> wgpu::CommandBuffer {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for i in 0..self.passes.len() {
                let p = self.arm_pass(i, arm);
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        if self.staged_read {
            enc.copy_buffer_to_buffer(self.token_out.raw(), 0, &self.tok_stage, 0, 4);
        }
        enc.finish()
    }

    fn read_stage(&self, n: usize) -> Result<Vec<u32>> {
        self.map_stage(&self.tok_stage, n, self.spin_read)
    }

    fn map_stage(&self, buf: &wgpu::Buffer, n: usize, spin: bool) -> Result<Vec<u32>> {
        let slice = buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        let mapped = if spin {
            let t0 = std::time::Instant::now();
            loop {
                let _ = self.ctx.device.poll(wgpu::PollType::Poll);
                match rx.try_recv() {
                    Ok(r) => break r,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        anyhow::bail!("token stage map callback dropped")
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {}
                }
                if t0.elapsed() > SPIN_READ_BUDGET {
                    self.ctx.poll_blocking().map_err(err)?;
                    break rx
                        .recv()
                        .map_err(|e| anyhow::anyhow!("token stage map callback: {e}"))?;
                }
                std::hint::spin_loop();
            }
        } else {
            self.ctx.poll_blocking().map_err(err)?;
            rx.recv()
                .map_err(|e| anyhow::anyhow!("token stage map callback: {e}"))?
        };
        mapped.map_err(|e| anyhow::anyhow!("token stage map: {e}"))?;
        let out = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("token stage mapped range: {e}"))?;
            bytemuck::cast_slice::<u8, u32>(&view)[..n].to_vec()
        };
        buf.unmap();
        Ok(out)
    }

    fn encode_pipe_cb(&self, parity: usize, arm: SplitArm) -> wgpu::CommandBuffer {
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            for i in 0..self.passes.len() {
                let p = self.arm_pass(i, arm);
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        let src = self.token_out.raw();
        enc.copy_buffer_to_buffer(src, 0, &self.pipe_stage[parity], 0, 4);
        enc.copy_buffer_to_buffer(src, 0, self.tok_idx.raw(), 0, 4);
        enc.finish()
    }

    fn pipe_submit(&mut self, token: Option<u32>) -> Result<()> {
        anyhow::ensure!(self.pos < self.max_seq, "kv cache full at {}", self.pos);
        self.write_step_uniforms(token)?;
        let arm = self.active_arm();
        let p = self.pipe_parity;
        let cb = match Self::take_arm_cb(&mut self.pipe_cb[p], arm) {
            Some(cb) => cb,
            None => self.encode_pipe_cb(p, arm),
        };
        self.ctx.queue.submit([cb]);
        if self.preenc {
            let next = self.arm_for_total(self.pos + 3);
            self.pipe_cb[p] = Some((next, self.encode_pipe_cb(p, next)));
        }
        self.pipe_parity ^= 1;
        self.pipe_inflight += 1;
        self.pos += 1;
        Ok(())
    }

    fn pipe_take(&mut self) -> Result<u32> {
        anyhow::ensure!(self.pipe_inflight > 0, "decode pipe is empty");
        let p = self.pipe_read;
        let v = self.map_stage(&self.pipe_stage[p], 1, true)?[0];
        self.pipe_read ^= 1;
        self.pipe_inflight -= 1;
        Ok(v)
    }

    pub fn decode_pipe_inflight(&self) -> usize {
        self.pipe_inflight
    }

    pub fn decode_step_pipelined(&mut self, token: Option<u32>) -> Result<u32> {
        anyhow::ensure!(
            self.staged_read,
            "decode pipe needs the staged readback (NV_E4B_WGPU_STAGED_READ=0 disables it)"
        );
        if self.prof != ProfMode::Off || !self.validated {
            let t = token.context("decode pipe cannot start from an empty pipe without a token")?;
            anyhow::ensure!(self.pipe_inflight == 0, "decode pipe already primed");
            return self.decode_step(t);
        }
        match token {
            Some(t) => {
                anyhow::ensure!(self.pipe_inflight == 0, "decode pipe already primed");
                anyhow::ensure!((t as usize) < self.vocab, "token {t} out of vocab");
                self.pipe_submit(Some(t))?;
            }
            None => anyhow::ensure!(self.pipe_inflight == 1, "decode pipe is not primed"),
        }
        if self.pos < self.max_seq {
            self.pipe_submit(None)?;
        }
        self.pipe_take()
    }

    pub fn decode_pipe_abort(&mut self) -> Result<usize> {
        let d = self.pipe_inflight;
        if d == 0 {
            return Ok(0);
        }
        self.ctx.poll_blocking().map_err(err)?;
        self.pipe_inflight = 0;
        self.pipe_read = self.pipe_parity;
        self.pos -= d;
        self.kv_epoch += 1;
        Ok(d)
    }

    fn read_token_out(&self, n: usize) -> Result<Vec<u32>> {
        if self.staged_read {
            self.read_stage(n)
        } else {
            let t = self.token_out.download(self.ctx).map_err(err)?;
            Ok(t[..n].to_vec())
        }
    }

    pub fn set_prof_mode(&mut self, mode: ProfMode) -> bool {
        if mode != ProfMode::Off && !self.ctx.caps.timestamp_query {
            self.prof = ProfMode::Off;
            return false;
        }
        self.prof = mode;
        self.pending_cb = None;
        true
    }

    pub fn prof_mode(&self) -> ProfMode {
        self.prof
    }

    pub fn set_staged_read(&mut self, on: bool) {
        if on == self.staged_read {
            return;
        }
        self.staged_read = on;
        self.pending_cb = None;
        if let Some(vs) = self.verify.as_mut() {
            vs.pending_cb = None;
        }
    }

    pub fn staged_read(&self) -> bool {
        self.staged_read
    }

    pub fn set_spin_read(&mut self, on: bool) {
        self.spin_read = on;
    }

    pub fn spin_read(&self) -> bool {
        self.spin_read
    }

    pub fn prof_timeline(&self) -> Vec<(&str, f64, f64)> {
        self.prof_timeline
            .iter()
            .map(|&(i, b, e)| (self.pass_labels[i].as_str(), b, e))
            .collect()
    }

    pub fn prof_pass_total_ns(&self) -> f64 {
        self.prof_pass_total_ns
    }

    pub fn pass_label(&self, i: usize) -> &str {
        &self.pass_labels[i]
    }

    pub fn pass_grid(&self, i: usize) -> (u32, u32, u32) {
        self.passes[i].grid
    }

    pub fn set_preenc(&mut self, on: bool) {
        self.preenc = on;
        if !on {
            self.pending_cb = None;
        }
    }

    pub fn preenc(&self) -> bool {
        self.preenc
    }

    pub fn pass_bound_bytes(&self, i: usize) -> (u64, u64) {
        (self.passes[i].bound_bytes, self.passes[i].widest_bytes)
    }

    pub fn probe_at(&mut self, token: u32, pos: usize) -> Result<()> {
        anyhow::ensure!(
            pos < self.max_seq,
            "probe pos {pos} beyond {}",
            self.max_seq
        );
        self.pos = pos;
        self.write_step_uniforms(Some(token))
    }

    pub fn probe_prefix(&self, n: usize) -> Result<()> {
        anyhow::ensure!(
            n <= self.passes.len(),
            "prefix {n} beyond {} passes",
            self.passes.len()
        );
        let arm = self.active_arm();
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            for i in 0..n {
                let p = self.arm_pass(i, arm);
                cp.set_pipeline(&p.pipeline);
                cp.set_bind_group(0, &p.bind, &[]);
                cp.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        self.ctx.queue.submit([enc.finish()]);
        self.ctx.poll_blocking().map_err(err)
    }

    pub fn probe_encode(&self, n: usize) -> Result<()> {
        anyhow::ensure!(
            n <= self.passes.len(),
            "prefix {n} beyond {} passes",
            self.passes.len()
        );
        let arm = self.active_arm();
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        {
            let mut cp = enc.begin_compute_pass(&Default::default());
            for i in 0..n {
                let p = self.arm_pass(i, arm);
                cp.set_pipeline(&p.pipeline);
                cp.set_bind_group(0, &p.bind, &[]);
                cp.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        }
        drop(enc.finish());
        Ok(())
    }

    pub fn probe_append(&mut self, needle: &str, copies: usize) -> usize {
        self.probe_append_class(needle, None, copies)
    }

    pub fn probe_append_class(
        &mut self,
        needle: &str,
        widest: Option<u64>,
        copies: usize,
    ) -> usize {
        self.passes.truncate(self.base_passes);
        let src: Vec<Pass> = (0..self.base_passes)
            .filter(|&i| {
                self.pass_labels[i].contains(needle)
                    && widest.is_none_or(|w| self.passes[i].widest_bytes == w)
            })
            .map(|i| self.passes[i].clone())
            .collect();
        for _ in 0..copies {
            self.passes.extend(src.iter().cloned());
        }
        self.pending_cb = None;
        src.len() * copies
    }

    pub fn probe_append_clear(&mut self) {
        self.passes.truncate(self.base_passes);
        self.pending_cb = None;
    }

    fn submit_prof(&mut self) -> Result<()> {
        let n = self.passes.len();
        let per_dispatch = self.prof == ProfMode::PerDispatch;
        let queries = if per_dispatch { 2 * n as u32 } else { 2 };
        anyhow::ensure!(
            queries <= wgpu::QUERY_SET_MAX_QUERIES,
            "profiling {n} passes needs {queries} timestamp queries, limit {}",
            wgpu::QUERY_SET_MAX_QUERIES
        );
        let bytes = queries as u64 * 8;
        let qs = self.ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("e4bw-prof"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        });
        let resolve = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("e4bw-prof-resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("e4bw-prof-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let arm = self.active_arm();
        let mut enc = self.ctx.device.create_command_encoder(&Default::default());
        if per_dispatch {
            for i in 0..n {
                let p = self.arm_pass(i, arm);
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs,
                        beginning_of_pass_write_index: Some((i * 2) as u32),
                        end_of_pass_write_index: Some((i * 2 + 1) as u32),
                    }),
                });
                pass.set_pipeline(&p.pipeline);
                pass.set_bind_group(0, &p.bind, &[]);
                pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
            }
        } else {
            {
                let _marker = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: None,
                    }),
                });
            }
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                for i in 0..n {
                    let p = self.arm_pass(i, arm);
                    pass.set_pipeline(&p.pipeline);
                    pass.set_bind_group(0, &p.bind, &[]);
                    pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
                }
            }
            {
                let _marker = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs,
                        beginning_of_pass_write_index: None,
                        end_of_pass_write_index: Some(1),
                    }),
                });
            }
        }
        if self.staged_read {
            enc.copy_buffer_to_buffer(self.token_out.raw(), 0, &self.tok_stage, 0, 4);
        }
        enc.resolve_query_set(&qs, 0..queries, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
        self.ctx.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.ctx.poll_blocking().map_err(err)?;
        rx.recv()
            .map_err(|e| anyhow::anyhow!("prof map callback: {e}"))?
            .map_err(|e| anyhow::anyhow!("prof map: {e}"))?;
        let ts: Vec<u64> = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("prof mapped range: {e}"))?;
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        staging.unmap();
        let period = self.ctx.queue.get_timestamp_period() as f64;
        if per_dispatch {
            self.prof_timeline.clear();
            let t0 = ts[0];
            for i in 0..n {
                let ns = ts[i * 2 + 1].saturating_sub(ts[i * 2]) as f64 * period;
                dispatch::profile::record(&self.pass_labels[i], ns);
                self.prof_timeline.push((
                    i,
                    ts[i * 2].saturating_sub(t0) as f64 * period,
                    ts[i * 2 + 1].saturating_sub(t0) as f64 * period,
                ));
            }
        } else {
            let ns = ts[1].saturating_sub(ts[0]) as f64 * period;
            self.prof_pass_total_ns = ns;
            dispatch::profile::record("e4b_decode_whole_pass", ns);
        }
        Ok(())
    }

    pub fn config(&self) -> &Gemma4Config {
        &self.config
    }

    pub fn current_pos(&self) -> usize {
        self.pos
    }

    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    pub fn weight_bytes_per_token(&self) -> u64 {
        self.weight_bytes
    }

    pub fn w4_scale_grain(&self) -> wk::gemv_w4a16::ScaleGrain {
        self.w4_grain
    }

    pub fn w4_route_census(&self) -> (usize, usize, usize) {
        (self.w4_census[0], self.w4_census[1], self.w4_census[2])
    }

    pub fn lora_passes(&self) -> usize {
        self.lora_passes
    }

    pub fn fnc_unrolled(&self) -> Option<[bool; 3]> {
        self.fnc_unrolled
    }

    pub fn flash1_route(&self) -> (&str, &[(u32, usize)]) {
        (&self.flash1_entry, &self.flash1_hds)
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn prefill_pass_count(&self) -> usize {
        self.prefill.as_ref().map(|p| p.passes.len()).unwrap_or(0)
    }

    pub fn prefill_chunk_len(&self) -> usize {
        self.prefill.as_ref().map(|p| p.m).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        let _ = self.decode_pipe_abort();
        self.pos = 0;
        self.kv_epoch += 1;
    }

    pub fn kv_cache_snapshot(&self, li: usize) -> Result<Option<KvCacheSnapshot>> {
        anyhow::ensure!(
            self.pipe_inflight == 0,
            "kv snapshot would include the speculative step; abort the pipe first"
        );
        let Some(kv) = self.kv_layers.get(li).and_then(|o| o.as_ref()) else {
            return Ok(None);
        };
        self.sync()?;
        Ok(Some((
            kv.k_fp8.download(self.ctx).map_err(err)?,
            kv.v_fp8.download(self.ctx).map_err(err)?,
            kv.k_scales.download(self.ctx).map_err(err)?,
            kv.v_scales.download(self.ctx).map_err(err)?,
        )))
    }

    pub fn kv_cache_snapshot_range(
        &self,
        li: usize,
        start: usize,
        len: usize,
    ) -> Result<Option<KvCacheSnapshot>> {
        let Some(kv) = self.kv_layers.get(li).and_then(|o| o.as_ref()) else {
            return Ok(None);
        };
        anyhow::ensure!(
            start + len <= self.max_seq,
            "kv snapshot range {start}+{len} outside max_seq {}",
            self.max_seq
        );
        let words_per_slot = kv.k_fp8.len() / self.max_seq;
        let scales_per_slot = kv.k_scales.len() / self.max_seq;
        self.sync()?;
        Ok(Some((
            kv.k_fp8
                .download_range(self.ctx, start * words_per_slot, len * words_per_slot)
                .map_err(err)?,
            kv.v_fp8
                .download_range(self.ctx, start * words_per_slot, len * words_per_slot)
                .map_err(err)?,
            kv.k_scales
                .download_range(self.ctx, start * scales_per_slot, len * scales_per_slot)
                .map_err(err)?,
            kv.v_scales
                .download_range(self.ctx, start * scales_per_slot, len * scales_per_slot)
                .map_err(err)?,
        )))
    }

    pub fn kv_epoch(&self) -> u64 {
        self.kv_epoch
    }

    pub fn kv_layer_count(&self) -> usize {
        self.kv_layers.len()
    }

    pub fn kv_layer_lens(&self, li: usize) -> Option<[usize; 4]> {
        let kv = self.kv_layers.get(li).and_then(|o| o.as_ref())?;
        Some([
            kv.k_fp8.len(),
            kv.v_fp8.len(),
            kv.k_scales.len(),
            kv.v_scales.len(),
        ])
    }

    pub fn kv_cache_restore(&mut self, li: usize, snap: &KvCacheSnapshot) -> Result<bool> {
        self.decode_pipe_abort()?;
        let Some(kv) = self.kv_layers.get(li).and_then(|o| o.as_ref()) else {
            return Ok(false);
        };
        kv.k_fp8.write(self.ctx, &snap.0).map_err(err)?;
        kv.v_fp8.write(self.ctx, &snap.1).map_err(err)?;
        kv.k_scales.write(self.ctx, &snap.2).map_err(err)?;
        kv.v_scales.write(self.ctx, &snap.3).map_err(err)?;
        Ok(true)
    }

    pub fn restore_pos(&mut self, pos: usize) -> Result<()> {
        self.decode_pipe_abort()?;
        anyhow::ensure!(
            pos <= self.max_seq,
            "restore_pos {pos} past max_seq {}",
            self.max_seq
        );
        self.pos = pos;
        self.kv_epoch += 1;
        Ok(())
    }

    pub fn kv_cache_gpu(
        &self,
        li: usize,
    ) -> Option<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer)> {
        let kv = self.kv_layers.get(li).and_then(|o| o.as_ref())?;
        Some((
            kv.k_fp8.raw(),
            kv.v_fp8.raw(),
            kv.k_scales.raw(),
            kv.v_scales.raw(),
        ))
    }

    pub fn embed_table_gpu(&self) -> (&[wgpu::Buffer], usize) {
        (&self.embed_tab.chunks, self.embed_tab.plan.rows_per_chunk)
    }

    pub fn sync(&self) -> Result<()> {
        self.ctx.queue.submit(std::iter::empty());
        self.ctx.poll_blocking().map_err(err)
    }

    fn window_start(total: usize, window: usize) -> usize {
        if window > 0 && total > window {
            total - window
        } else {
            0
        }
    }

    fn write_step_uniforms(&self, token: Option<u32>) -> Result<()> {
        let pos = self.pos as i32;
        let total = self.pos + 1;
        if let Some(t) = token {
            let mut ids = vec![0i32; self.tok_idx.len()];
            ids[0] = t as i32;
            self.tok_idx.write(self.ctx, &ids).map_err(err)?;
        }
        let mut poss = vec![0i32; self.rope_pos.len()];
        poss[0] = pos;
        self.rope_pos.write(self.ctx, &poss).map_err(err)?;
        self.kv_start.write(self.ctx, &[pos]).map_err(err)?;
        let splits = self.arm_splits(self.arm_for_total(total));
        let mut fd_s = self.fd_sliding_base;
        fd_s.total = total as u32;
        fd_s.start = Self::window_start(total, self.config.sliding_window) as u32;
        fd_s.splits = if deep_full_only() {
            self.fd_full_base.splits
        } else {
            splits
        };
        self.fd_sliding.write(self.ctx, &fd_s);
        let mut fd_f = self.fd_full_base;
        fd_f.total = total as u32;
        fd_f.start = 0;
        fd_f.splits = splits;
        self.fd_full.write(self.ctx, &fd_f);
        Ok(())
    }

    fn step_inner(&mut self, token: u32) -> Result<()> {
        anyhow::ensure!((token as usize) < self.vocab, "token {token} out of vocab");
        anyhow::ensure!(self.pos < self.max_seq, "kv cache full at {}", self.pos);
        anyhow::ensure!(
            self.pipe_inflight == 0,
            "decode pipe still has {} step(s) in flight; call decode_pipe_abort first",
            self.pipe_inflight
        );
        self.write_step_uniforms(Some(token))?;

        let scope = if !self.validated {
            Some(
                self.ctx
                    .device
                    .push_error_scope(wgpu::ErrorFilter::Validation),
            )
        } else {
            None
        };
        if self.prof != ProfMode::Off && self.ctx.caps.timestamp_query {
            self.pending_cb = None;
            self.submit_prof()?;
            if let Some(scope) = scope {
                if let Some(e) = pollster::block_on(scope.pop()) {
                    anyhow::bail!("gemma4_e4b_wgpu decode step validation: {e}");
                }
                self.validated = true;
            }
            self.pos += 1;
            return Ok(());
        }
        let t_enc = if self.enc_stats {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let arm = self.active_arm();
        let cb = match Self::take_arm_cb(&mut self.pending_cb, arm) {
            Some(cb) => cb,
            None => self.encode_cb(arm),
        };
        self.ctx.queue.submit([cb]);
        if let Some(t0) = t_enc {
            let us = t0.elapsed().as_secs_f64() * 1e6;
            self.enc_us_sum += us;
            self.enc_us_min = self.enc_us_min.min(us);
            self.enc_steps += 1;
            if self.enc_steps.is_multiple_of(100) {
                eprintln!(
                    "[gemma4_e4b_wgpu] critical-path encode+submit ({} passes, preenc={}): min {:.1} us mean {:.1} us over {} steps",
                    self.passes.len(),
                    self.preenc,
                    self.enc_us_min,
                    self.enc_us_sum / self.enc_steps as f64,
                    self.enc_steps
                );
            }
        }
        if self.preenc {
            let next = self.arm_for_total(self.pos + 2);
            self.pending_cb = Some((next, self.encode_cb(next)));
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_e4b_wgpu decode step validation: {e}");
            }
            self.validated = true;
        }
        self.pos += 1;
        Ok(())
    }

    pub fn decode_step(&mut self, token: u32) -> Result<u32> {
        self.step_inner(token)?;
        Ok(self.read_token_out(1)?[0])
    }

    pub fn decode_step_logits(&mut self, token: u32) -> Result<(u32, Vec<f32>)> {
        self.step_inner(token)?;
        let t = self.read_token_out(1)?;
        let logits = self.logits_f32.download(self.ctx).map_err(err)?;
        Ok((t[0], logits))
    }

    pub fn decode_chain(&mut self, token: u32, k: usize) -> Result<Vec<u32>> {
        anyhow::ensure!(
            (1..=MAX_CHAIN).contains(&k),
            "decode_chain k {k} outside 1..={MAX_CHAIN}"
        );
        if k == 1 || !self.validated || self.prof != ProfMode::Off {
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
            self.write_step_uniforms(if i == 0 { Some(token) } else { None })?;
            let arm = self.active_arm();
            let cb = match Self::take_arm_cb(&mut self.pending_cb, arm) {
                Some(cb) => cb,
                None => self.encode_cb(arm),
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
            self.pos += 1;
        }
        if self.preenc {
            let next = self.arm_for_total(self.pos + 1);
            self.pending_cb = Some((next, self.encode_cb(next)));
        }
        self.chain_out.download_range(self.ctx, 0, k).map_err(err)
    }

    pub fn prefill_chunk(&mut self, tokens: &[u32]) -> Result<()> {
        self.prefill_chunk_advance(tokens, tokens.len())
    }

    fn prefill_chunk_advance(&mut self, tokens: &[u32], advance: usize) -> Result<()> {
        let ctx = self.ctx;
        let vocab = self.vocab;
        let max_seq = self.max_seq;
        let pos0 = self.pos;
        let preenc = self.preenc;
        let Some(pf) = self.prefill.as_mut() else {
            anyhow::bail!("prefill pass list disabled");
        };
        let m = pf.m;
        anyhow::ensure!(
            tokens.len() == m,
            "prefill_chunk wants exactly {m} tokens, got {}",
            tokens.len()
        );
        anyhow::ensure!(
            (1..=m).contains(&advance),
            "prefill advance {advance} out of 1..={m}"
        );
        for &t in tokens {
            anyhow::ensure!((t as usize) < vocab, "token {t} out of vocab");
        }
        anyhow::ensure!(pos0 + m <= max_seq, "kv cache full at {pos0} + {m}");
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let poss: Vec<i32> = (0..m).map(|i| (pos0 + i) as i32).collect();
        self.tok_idx.write(ctx, &ids).map_err(err)?;
        self.rope_pos.write(ctx, &poss).map_err(err)?;
        self.kv_start.write(ctx, &[pos0 as i32]).map_err(err)?;
        let total = (pos0 + m) as u32;
        let mut fs = pf.fd_sliding_base;
        fs.total = total;
        pf.fd_sliding.write(ctx, &fs);
        let mut ff = pf.fd_full_base;
        ff.total = total;
        pf.fd_full.write(ctx, &ff);

        pf.set_live_rows(ctx, m);

        let scope = if !pf.validated {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        } else {
            None
        };
        let cb = match pf.pending_cb.take() {
            Some(cb) => cb,
            None => encode_passes(ctx, &pf.passes),
        };
        ctx.queue.submit([cb]);
        if preenc {
            pf.pending_cb = Some(encode_passes(ctx, &pf.passes));
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_e4b_wgpu prefill chunk validation: {e}");
            }
            pf.validated = true;
        }
        self.pos += advance;
        Ok(())
    }

    pub fn prefill_chunk_profiled(&mut self, tokens: &[u32]) -> Result<Vec<(String, f64)>> {
        anyhow::ensure!(
            self.ctx.caps.timestamp_query,
            "prefill_chunk_profiled needs timestamp queries"
        );
        let ctx = self.ctx;
        let vocab = self.vocab;
        let max_seq = self.max_seq;
        let pos0 = self.pos;
        let Some(pf) = self.prefill.as_mut() else {
            anyhow::bail!("prefill pass list disabled");
        };
        let m = pf.m;
        anyhow::ensure!(
            tokens.len() == m,
            "prefill_chunk_profiled wants exactly {m} tokens, got {}",
            tokens.len()
        );
        for &t in tokens {
            anyhow::ensure!((t as usize) < vocab, "token {t} out of vocab");
        }
        anyhow::ensure!(pos0 + m <= max_seq, "kv cache full at {pos0} + {m}");
        let ids: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
        let poss: Vec<i32> = (0..m).map(|i| (pos0 + i) as i32).collect();
        self.tok_idx.write(ctx, &ids).map_err(err)?;
        self.rope_pos.write(ctx, &poss).map_err(err)?;
        self.kv_start.write(ctx, &[pos0 as i32]).map_err(err)?;
        let total = (pos0 + m) as u32;
        let mut fs = pf.fd_sliding_base;
        fs.total = total;
        pf.fd_sliding.write(ctx, &fs);
        let mut ff = pf.fd_full_base;
        ff.total = total;
        pf.fd_full.write(ctx, &ff);
        pf.set_live_rows(ctx, m);
        pf.pending_cb = None;

        let n = pf.passes.len();
        let queries = 2 * n as u32;
        anyhow::ensure!(
            queries <= wgpu::QUERY_SET_MAX_QUERIES,
            "profiling {n} prefill passes needs {queries} timestamp queries"
        );
        let bytes = queries as u64 * 8;
        let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("e4bw-pf-prof"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("e4bw-pf-prof-resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("e4bw-pf-prof-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for (i, p) in pf.passes.iter().enumerate() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &qs,
                    beginning_of_pass_write_index: Some((i * 2) as u32),
                    end_of_pass_write_index: Some((i * 2 + 1) as u32),
                }),
            });
            pass.set_pipeline(&p.pipeline);
            pass.set_bind_group(0, &p.bind, &[]);
            pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
        }
        enc.resolve_query_set(&qs, 0..queries, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
        ctx.queue.submit([enc.finish()]);

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.poll_blocking().map_err(err)?;
        rx.recv()
            .map_err(|e| anyhow::anyhow!("prefill prof map callback: {e}"))?
            .map_err(|e| anyhow::anyhow!("prefill prof map: {e}"))?;
        let ts: Vec<u64> = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("prefill prof mapped range: {e}"))?;
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        staging.unmap();
        let period = ctx.queue.get_timestamp_period() as f64;
        let out = pf
            .labels
            .iter()
            .enumerate()
            .map(|(i, l)| {
                (
                    l.clone(),
                    ts[i * 2 + 1].saturating_sub(ts[i * 2]) as f64 * period,
                )
            })
            .collect();
        self.pos += m;
        Ok(out)
    }

    pub fn prefill_tokens(&mut self, tokens: &[u32]) -> Result<usize> {
        let m = self.prefill_chunk_len();
        if m == 0 {
            return Ok(0);
        }
        let tail = self.prefill.as_ref().map(|p| p.tail).unwrap_or(false);
        let mut done = 0;
        while tokens.len() - done >= m {
            self.prefill_chunk(&tokens[done..done + m])?;
            done += m;
        }
        let left = tokens.len() - done;
        if tail && left > 0 && self.pos + m <= self.max_seq {
            let mut padded = Vec::with_capacity(m);
            padded.extend_from_slice(&tokens[done..]);
            let pad = *padded.last().unwrap();
            padded.resize(m, pad);
            self.prefill_chunk_advance(&padded, left)?;
            done += left;
        }
        Ok(done)
    }

    pub fn prefill_prompt(&mut self, prompt: &[u32]) -> Result<u32> {
        anyhow::ensure!(!prompt.is_empty(), "empty prompt");
        let done = self.prefill_tokens(&prompt[..prompt.len() - 1])?;
        let mut next = 0u32;
        for &t in &prompt[done..] {
            next = self.decode_step(t)?;
        }
        Ok(next)
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
        self.decode_pipe_abort()?;
        anyhow::ensure!(pos <= self.pos, "truncate_to {pos} beyond pos {}", self.pos);
        if pos < self.pos {
            self.kv_epoch += 1;
        }
        self.pos = pos;
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

    pub fn verify_chain(&mut self, batch: &[u32]) -> Result<Vec<u32>> {
        let ctx = self.ctx;
        let vocab = self.vocab;
        let max_seq = self.max_seq;
        let pos0 = self.pos;
        let preenc = self.preenc;
        let staged = self.staged_read;
        let token_out = self.token_out.raw();
        let tok_stage = &self.tok_stage;
        let Some(pf) = self.prefill.as_mut() else {
            anyhow::bail!("verify_chain needs the prefill pass list");
        };
        let Some(vs) = self.verify.as_mut() else {
            anyhow::bail!("verify_chain disabled: needs the default bf16 lm_head epilogue");
        };
        let m = pf.m;
        let mb = batch.len();
        anyhow::ensure!(
            (1..=vs.rows).contains(&mb),
            "verify_chain batch {mb} out of 1..={}",
            vs.rows
        );
        for &t in batch {
            anyhow::ensure!((t as usize) < vocab, "token {t} out of vocab");
        }
        anyhow::ensure!(pos0 + m <= max_seq, "kv cache full at {pos0} + {m}");
        let live = if pf.live_rows_enabled { mb } else { m };
        let mut ids: Vec<i32> = batch.iter().map(|&t| t as i32).collect();
        let last = *ids.last().unwrap();
        ids.resize(m, last);
        let poss: Vec<i32> = (0..m).map(|i| (pos0 + i) as i32).collect();
        self.tok_idx.write(ctx, &ids).map_err(err)?;
        self.rope_pos.write(ctx, &poss).map_err(err)?;
        self.kv_start.write(ctx, &[pos0 as i32]).map_err(err)?;

        let total = (pos0 + live) as u32;
        let mut fs = pf.fd_sliding_base;
        fs.total = total;
        fs.m_rows = live as u32;
        pf.fd_sliding.write(ctx, &fs);
        let mut ff = pf.fd_full_base;
        ff.total = total;
        ff.m_rows = live as u32;
        pf.fd_full.write(ctx, &ff);
        pf.set_verify_live_rows(ctx, live);
        vs.set_live_rows(ctx, if pf.live_rows_enabled { mb } else { vs.rows });

        let scope = if !vs.validated {
            Some(ctx.device.push_error_scope(wgpu::ErrorFilter::Validation))
        } else {
            None
        };
        let cb_main = match vs.pf_passes.as_ref() {
            Some(vpp) => match vs.pending_pf_cb.take() {
                Some(cb) => cb,
                None => encode_passes(ctx, vpp),
            },
            None => match pf.pending_cb.take() {
                Some(cb) => cb,
                None => encode_passes(ctx, &pf.passes),
            },
        };
        let stage_bytes = (vs.rows * 4) as u64;
        let epi_cb = |vs: &VerifyState| {
            if staged {
                encode_passes_staged(ctx, &vs.passes, token_out, tok_stage, stage_bytes)
            } else {
                encode_passes(ctx, &vs.passes)
            }
        };
        let cb_epi = match vs.pending_cb.take() {
            Some(cb) => cb,
            None => epi_cb(vs),
        };
        ctx.queue.submit([cb_main, cb_epi]);
        if preenc {
            match vs.pf_passes.as_ref() {
                Some(vpp) => vs.pending_pf_cb = Some(encode_passes(ctx, vpp)),
                None => pf.pending_cb = Some(encode_passes(ctx, &pf.passes)),
            }
            vs.pending_cb = Some(epi_cb(vs));
        }
        if let Some(scope) = scope {
            if let Some(e) = pollster::block_on(scope.pop()) {
                anyhow::bail!("gemma4_e4b_wgpu verify chain validation: {e}");
            }
            vs.validated = true;
        }
        let toks = self.read_token_out(mb)?;
        Ok(toks)
    }

    pub fn verify_chain_profiled(
        &mut self,
        batch: &[u32],
    ) -> Result<(Vec<u32>, Vec<(String, f64)>)> {
        anyhow::ensure!(
            self.ctx.caps.timestamp_query,
            "verify_chain_profiled needs TIMESTAMP_QUERY"
        );
        let ctx = self.ctx;
        let vocab = self.vocab;
        let max_seq = self.max_seq;
        let pos0 = self.pos;
        let preenc = self.preenc;
        let Some(pf) = self.prefill.as_mut() else {
            anyhow::bail!("verify_chain needs the prefill pass list");
        };
        let Some(vs) = self.verify.as_mut() else {
            anyhow::bail!("verify_chain disabled: needs the default bf16 lm_head epilogue");
        };
        let m = pf.m;
        let mb = batch.len();
        anyhow::ensure!(
            (1..=vs.rows).contains(&mb),
            "verify_chain batch {mb} out of 1..={}",
            vs.rows
        );
        for &t in batch {
            anyhow::ensure!((t as usize) < vocab, "token {t} out of vocab");
        }
        anyhow::ensure!(pos0 + m <= max_seq, "kv cache full at {pos0} + {m}");
        let live = if pf.live_rows_enabled { mb } else { m };
        let mut ids: Vec<i32> = batch.iter().map(|&t| t as i32).collect();
        let last = *ids.last().unwrap();
        ids.resize(m, last);
        let poss: Vec<i32> = (0..m).map(|i| (pos0 + i) as i32).collect();
        self.tok_idx.write(ctx, &ids).map_err(err)?;
        self.rope_pos.write(ctx, &poss).map_err(err)?;
        self.kv_start.write(ctx, &[pos0 as i32]).map_err(err)?;
        let total = (pos0 + live) as u32;
        let mut fs = pf.fd_sliding_base;
        fs.total = total;
        fs.m_rows = live as u32;
        pf.fd_sliding.write(ctx, &fs);
        let mut ff = pf.fd_full_base;
        ff.total = total;
        ff.m_rows = live as u32;
        pf.fd_full.write(ctx, &ff);
        pf.set_verify_live_rows(ctx, live);
        vs.set_live_rows(ctx, if pf.live_rows_enabled { mb } else { vs.rows });

        let main_passes: &[Pass] = match vs.pf_passes.as_ref() {
            Some(vpp) => vpp,
            None => &pf.passes,
        };
        let main_labels: &[String] = if vs.pf_passes.is_some() {
            &vs.pf_labels
        } else {
            &pf.labels
        };
        let n = main_passes.len() + vs.passes.len();
        let queries = 2 * n as u32;
        anyhow::ensure!(
            queries <= wgpu::QUERY_SET_MAX_QUERIES,
            "profiling {n} verify passes needs {queries} timestamp queries, limit {}",
            wgpu::QUERY_SET_MAX_QUERIES
        );
        let bytes = queries as u64 * 8;
        let qs = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("e4bw-verify-prof"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        });
        let resolve = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("e4bw-verify-prof-resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("e4bw-verify-prof-staging"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        pf.pending_cb = None;
        vs.pending_cb = None;
        vs.pending_pf_cb = None;
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for (i, p) in main_passes.iter().chain(vs.passes.iter()).enumerate() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &qs,
                    beginning_of_pass_write_index: Some((i * 2) as u32),
                    end_of_pass_write_index: Some((i * 2 + 1) as u32),
                }),
            });
            pass.set_pipeline(&p.pipeline);
            pass.set_bind_group(0, &p.bind, &[]);
            pass.dispatch_workgroups(p.grid.0, p.grid.1, p.grid.2);
        }
        enc.resolve_query_set(&qs, 0..queries, &resolve, 0);
        enc.copy_buffer_to_buffer(&resolve, 0, &staging, 0, bytes);
        ctx.queue.submit([enc.finish()]);
        if preenc {
            match vs.pf_passes.as_ref() {
                Some(vpp) => vs.pending_pf_cb = Some(encode_passes(ctx, vpp)),
                None => pf.pending_cb = Some(encode_passes(ctx, &pf.passes)),
            }
            vs.pending_cb = Some(encode_passes(ctx, &vs.passes));
        }
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        ctx.poll_blocking().map_err(err)?;
        rx.recv()
            .map_err(|e| anyhow::anyhow!("verify prof map callback: {e}"))?
            .map_err(|e| anyhow::anyhow!("verify prof map: {e}"))?;
        let ts: Vec<u64> = {
            let view = slice
                .get_mapped_range()
                .map_err(|e| anyhow::anyhow!("verify prof mapped range: {e}"))?;
            bytemuck::cast_slice::<u8, u64>(&view).to_vec()
        };
        staging.unmap();
        let period = ctx.queue.get_timestamp_period() as f64;
        let mut prof = Vec::with_capacity(n);
        for (i, label) in main_labels.iter().chain(vs.labels.iter()).enumerate() {
            let ns = ts[i * 2 + 1].saturating_sub(ts[i * 2]) as f64 * period;
            prof.push((label.clone(), ns));
        }
        let toks = self.token_out.download(ctx).map_err(err)?;
        Ok((toks[..mb].to_vec(), prof))
    }

    pub fn decode_hid_gpu(&self) -> (&wgpu::Buffer, usize) {
        (&self.decode_hid, self.decode_hid_row_words)
    }

    pub fn verify_hid_gpu(&self) -> Option<(&wgpu::Buffer, usize, usize)> {
        self.verify
            .as_ref()
            .map(|vs| (&vs.hid, vs.hid_row_words, vs.rows))
    }

    pub fn decode_hidden_row(&self) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.pipe_inflight == 0,
            "decode_hidden_row would read the speculative step's hidden row; abort the pipe first"
        );
        self.sync()?;
        let words: Vec<u32> =
            dispatch::read_back(self.ctx, &self.decode_hid, self.decode_hid_row_words)
                .map_err(err)?;
        let mut out = Vec::with_capacity(self.decode_hid_row_words * 2);
        for &w in &words {
            out.push(f32::from_bits((w & 0xffff) << 16));
            out.push(f32::from_bits(w & 0xffff_0000));
        }
        Ok(out)
    }

    pub fn verify_hidden_row(&self, row: usize) -> Result<Vec<f32>> {
        let Some(vs) = self.verify.as_ref() else {
            anyhow::bail!("verify hidden readback needs the verify epilogue");
        };
        anyhow::ensure!(
            row < vs.rows,
            "verify hidden row {row} out of 0..{}",
            vs.rows
        );
        self.sync()?;
        let words: Vec<u32> =
            dispatch::read_back(self.ctx, &vs.hid, (row + 1) * vs.hid_row_words).map_err(err)?;
        let start = row * vs.hid_row_words;
        let mut out = Vec::with_capacity(vs.hid_row_words * 2);
        for &w in &words[start..] {
            out.push(f32::from_bits((w & 0xffff) << 16));
            out.push(f32::from_bits(w & 0xffff_0000));
        }
        Ok(out)
    }
}

const PREFIX: &str = "model.language_model";

fn bf16_bits_of(weights: &nv_weights::WeightLoader, name: &str) -> Result<Vec<u16>> {
    if weights.st_dtype_of(name) == Some(nv_weights::StDtype::BF16) {
        let raw = weights
            .raw_bytes(name)
            .with_context(|| format!("raw_bytes {name}"))?;
        anyhow::ensure!(raw.len() % 2 == 0, "{name}: odd byte length");
        let mut out = vec![0u16; raw.len() / 2];
        for (i, o) in out.iter_mut().enumerate() {
            *o = u16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
        }
        return Ok(out);
    }
    let t = weights
        .get(name, candle_core::DType::BF16)
        .with_context(|| format!("load {name}"))?;
    let v: Vec<half::bf16> = t.flatten_all()?.to_vec1()?;
    Ok(v.into_iter().map(|x| x.to_bits()).collect())
}

fn u32_words_of(weights: &nv_weights::WeightLoader, name: &str) -> Result<Vec<u32>> {
    let raw = weights
        .raw_bytes(name)
        .with_context(|| format!("raw_bytes {name}"))?;
    anyhow::ensure!(raw.len() % 4 == 0, "{name}: byte length not word-aligned");
    let mut out = vec![0u32; raw.len() / 4];
    for (i, o) in out.iter_mut().enumerate() {
        *o = u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]]);
    }
    Ok(out)
}

fn load_lin(weights: &nv_weights::WeightLoader, name: &str, n: usize, k: usize) -> Result<HostLin> {
    let packed_name = format!("{name}.weight_packed");
    if weights.has(&packed_name) {
        let packed = u32_words_of(weights, &packed_name)?;
        anyhow::ensure!(
            packed.len() == n * k / 8,
            "{name}: packed length {} != {n}x{k}/8",
            packed.len()
        );
        let scales = bf16_bits_of(weights, &format!("{name}.weight_scale"))?;
        anyhow::ensure!(
            !scales.is_empty() && scales.len() % n == 0 && k.is_multiple_of(scales.len() / n),
            "{name}: scale length {} does not divide {n}x{k}",
            scales.len()
        );
        let gs = k / (scales.len() / n);
        return Ok(HostLin::new_w4(packed, scales, gs, n, k));
    }
    Ok(HostLin::new(
        bf16_bits_of(weights, &format!("{name}.weight"))?,
        n,
        k,
    ))
}

fn concat_rows(parts: Vec<HostLin>) -> Result<HostLin> {
    let k = parts[0].k;
    anyhow::ensure!(parts.iter().all(|p| p.k == k), "fused parts must share k");
    let n: usize = parts.iter().map(|p| p.n).sum();
    let quant = parts[0].q.is_some();
    anyhow::ensure!(
        parts.iter().all(|p| p.q.is_some() == quant),
        "fused parts must share storage format"
    );
    if quant {
        let gs = parts[0].q.as_ref().map(|q| q.gs).unwrap_or(0);
        let mut packed = Vec::with_capacity(n * k / 8);
        let mut scales = Vec::with_capacity(n * (k / gs));
        for p in parts {
            let q = p.q.context("fused part lost quant payload")?;
            anyhow::ensure!(q.gs == gs, "fused parts must share group size");
            packed.extend_from_slice(&q.packed);
            scales.extend_from_slice(&q.scales);
        }
        return Ok(HostLin::new_w4(packed, scales, gs, n, k));
    }
    let mut w = Vec::with_capacity(n * k);
    for p in parts {
        w.extend_from_slice(&p.w);
    }
    Ok(HostLin::new(w, n, k))
}

fn load_e4b_layer_from_loader(
    config: &Gemma4Config,
    weights: &nv_weights::WeightLoader,
    i: usize,
) -> Result<E4bHostLayer> {
    let hidden = config.hidden_size;
    let inter = config.intermediate_size;
    let hpl = config.hidden_size_per_layer_input;
    let kind = config.layer_kind(i);
    let p = format!("{PREFIX}.layers.{i}");
    let hd = config.head_dim_for(kind);
    let nkv = config.num_kv_heads_for(kind);
    let n_q = config.num_attention_heads;
    let q_dim = n_q * hd;
    let kv_dim = nkv * hd;
    let kv_source = config.kv_source_layer(i);
    let has_v = !matches!(
        (kind, config.attention_k_eq_v),
        (LayerType::FullAttention, true)
    );

    let mut qkv_parts = vec![load_lin(
        weights,
        &format!("{p}.self_attn.q_proj"),
        q_dim,
        hidden,
    )?];
    let mut k_norm = Vec::new();
    if kv_source.is_none() {
        qkv_parts.push(load_lin(
            weights,
            &format!("{p}.self_attn.k_proj"),
            kv_dim,
            hidden,
        )?);
        if has_v {
            qkv_parts.push(load_lin(
                weights,
                &format!("{p}.self_attn.v_proj"),
                kv_dim,
                hidden,
            )?);
        }
        k_norm = bf16_bits_of(weights, &format!("{p}.self_attn.k_norm.weight"))?;
    }
    let qkv = concat_rows(qkv_parts).with_context(|| format!("layer {i}: qkv"))?;

    let scalar_bits = bf16_bits_of(weights, &format!("{p}.layer_scalar"))?;
    let layer_scalar =
        half::bf16::from_bits(*scalar_bits.first().context("empty layer_scalar")?).to_f32();

    let gate_up = concat_rows(vec![
        load_lin(weights, &format!("{p}.mlp.gate_proj"), inter, hidden)?,
        load_lin(weights, &format!("{p}.mlp.up_proj"), inter, hidden)?,
    ])
    .with_context(|| format!("layer {i}: gate_up"))?;

    Ok(E4bHostLayer {
        kind,
        kv_source,
        input_ln: bf16_bits_of(weights, &format!("{p}.input_layernorm.weight"))?,
        post_attn_ln: bf16_bits_of(weights, &format!("{p}.post_attention_layernorm.weight"))?,
        pre_ff_ln: bf16_bits_of(weights, &format!("{p}.pre_feedforward_layernorm.weight"))?,
        post_ff_ln: bf16_bits_of(weights, &format!("{p}.post_feedforward_layernorm.weight"))?,
        post_per_layer_input_norm: bf16_bits_of(
            weights,
            &format!("{p}.post_per_layer_input_norm.weight"),
        )?,
        q_norm: bf16_bits_of(weights, &format!("{p}.self_attn.q_norm.weight"))?,
        k_norm,
        layer_scalar,
        has_v,
        qkv,
        o: load_lin(weights, &format!("{p}.self_attn.o_proj"), hidden, q_dim)?,
        gate_up,
        down: load_lin(weights, &format!("{p}.mlp.down_proj"), hidden, inter)?,
        per_layer_input_gate: load_lin(weights, &format!("{p}.per_layer_input_gate"), hpl, hidden)?,
        per_layer_projection: load_lin(weights, &format!("{p}.per_layer_projection"), hidden, hpl)?,
    })
}

pub fn e4b_host_weights_from_loader(
    config: &Gemma4Config,
    weights: &nv_weights::WeightLoader,
) -> Result<E4bHostWeights> {
    anyhow::ensure!(
        config.has_per_layer_embeddings(),
        "e4b_host_weights_from_loader needs hidden_size_per_layer_input > 0"
    );
    anyhow::ensure!(
        config.tie_word_embeddings,
        "gemma4_e4b_wgpu loader: untied lm_head not wired"
    );
    let hidden = config.hidden_size;
    let hpl = config.hidden_size_per_layer_input;
    let n_layers = config.num_hidden_layers;

    let embed = bf16_bits_of(weights, &format!("{PREFIX}.embed_tokens.weight"))?;
    let embed_per_layer =
        bf16_bits_of(weights, &format!("{PREFIX}.embed_tokens_per_layer.weight"))?;
    let per_layer_model_projection = load_lin(
        weights,
        &format!("{PREFIX}.per_layer_model_projection"),
        n_layers * hpl,
        hidden,
    )?;
    let per_layer_projection_norm = bf16_bits_of(
        weights,
        &format!("{PREFIX}.per_layer_projection_norm.weight"),
    )?;
    let final_norm = bf16_bits_of(weights, &format!("{PREFIX}.norm.weight"))?;

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        layers.push(load_e4b_layer_from_loader(config, weights, i)?);
        eprintln!("[gemma4_e4b_wgpu] loaded layer {i}/{n_layers}");
    }

    Ok(E4bHostWeights {
        embed,
        embed_per_layer,
        per_layer_model_projection,
        per_layer_projection_norm,
        final_norm,
        layers,
    })
}

const LORA_CVT_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_lora_cvt.wgsl");

const LORA_WIDEN_ENTRY: &str = "e4b_lora_widen";
const LORA_REPACK_ENTRY: &str = "e4b_lora_repack";

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LoraCvtParams {
    m: u32,
    width: u32,
    pk_row_words: u32,
    wide_row_elems: u32,
    wide_col_off: u32,
    total: u32,
    pad0: u32,
    pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LoraFusedParams {
    m: u32,
    rank: u32,
    k: u32,
    a_slice_stride: u32,
    a_d0_stride: u32,
    y_row_stride: u32,
    win_off: u32,
    win_len: u32,
    scale: f32,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    off_b_d0: u32,
}

struct LoraMetaLayout {
    data: Vec<i32>,
    off_counts: u32,
    off_start: u32,
    off_active: u32,
    off_slice_n: u32,
    off_slice_start: u32,
    off_b_off: u32,
    off_b_d0: u32,
}

fn build_lora_meta(meta: &wk::lora::LoraMeta, widths: &[usize], rank: usize) -> LoraMetaLayout {
    let mut data: Vec<i32> = Vec::new();
    data.extend_from_slice(&meta.token_indices_sorted);
    let off_counts = data.len() as u32;
    data.extend_from_slice(&meta.num_tokens_per_lora);
    let off_start = data.len() as u32;
    data.extend_from_slice(&meta.lora_token_start_loc);
    let off_active = data.len() as u32;
    data.extend_from_slice(&meta.active_lora_ids);
    let off_slice_n = data.len() as u32;
    for &w in widths {
        data.push(w as i32);
    }
    let off_slice_start = data.len() as u32;
    let mut acc = 0usize;
    for &w in widths {
        data.push(acc as i32);
        acc += w;
    }
    let off_b_off = data.len() as u32;
    let mut b_acc = 0usize;
    for &w in widths {
        data.push(b_acc as i32);
        b_acc += meta.max_loras * w * rank;
    }
    let off_b_d0 = data.len() as u32;
    for &w in widths {
        data.push((w * rank) as i32);
    }
    LoraMetaLayout {
        data,
        off_counts,
        off_start,
        off_active,
        off_slice_n,
        off_slice_start,
        off_b_off,
        off_b_d0,
    }
}

struct LoraPipelines {
    fused: Arc<wgpu::ComputePipeline>,
    widen: Arc<wgpu::ComputePipeline>,
    repack: Arc<wgpu::ComputePipeline>,
}

fn build_lora_pipelines(ctx: &WgpuContext) -> Result<LoraPipelines> {
    let mk = |label: &str, source: &str, entry: &str| {
        dispatch::cached_compute_pipeline(ctx, label, source, entry)
            .map_err(|e| anyhow::anyhow!("pipeline {label}: {e}"))
    };
    let src_fused = compose(wk::lora::FUSED_WGSL);
    let src_cvt = compose(LORA_CVT_WGSL);
    Ok(LoraPipelines {
        fused: mk("e4bw-lora-fused", &src_fused, wk::lora::FUSED_ENTRY)?,
        widen: mk("e4bw-lora-widen", &src_cvt, LORA_WIDEN_ENTRY)?,
        repack: mk("e4bw-lora-repack", &src_cvt, LORA_REPACK_ENTRY)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn emit_lora_site(
    ctx: &WgpuContext,
    out: &mut Vec<Pass>,
    keep: &mut Vec<Box<dyn std::any::Any>>,
    pl: &LoraPipelines,
    scratch: &wgpu::Buffer,
    m: usize,
    site: &LoraSiteGpu,
    x: &wgpu::Buffer,
    segs: &[&wgpu::Buffer],
) {
    assert_eq!(
        segs.len(),
        site.segs.len(),
        "lora site: {} destination buffers for {} segments",
        segs.len(),
        site.segs.len()
    );
    let total_w: usize = site.segs.iter().sum();
    fn cvt(
        ctx: &WgpuContext,
        out: &mut Vec<Pass>,
        keep: &mut Vec<Box<dyn std::any::Any>>,
        pipeline: &Arc<wgpu::ComputePipeline>,
        scratch: &wgpu::Buffer,
        m: usize,
        total_w: usize,
        seg_widths: &[usize],
        segs: &[&wgpu::Buffer],
        widen: bool,
    ) {
        let mut off = 0usize;
        for (buf, w) in segs.iter().zip(seg_widths.iter()) {
            let total = if widen { m * w } else { m * w / 2 };
            let p = GpuUniform::new(
                ctx,
                "e4bw-lora-cvt-p",
                &LoraCvtParams {
                    m: m as u32,
                    width: *w as u32,
                    pk_row_words: (w / 2) as u32,
                    wide_row_elems: total_w as u32,
                    wide_col_off: off as u32,
                    total: total as u32,
                    ..Default::default()
                },
            );
            let grid = dispatch::workgroup_count_1d(ctx, total as u64, 256);
            let binds: [(u32, &wgpu::Buffer); 3] = [(0, buf), (1, scratch), (2, p.raw())];
            let bind = dispatch::bind_group(ctx, pipeline, &binds);
            let (bound_bytes, widest_bytes) = bind_bytes(binds.iter().map(|(_, b)| *b));
            out.push(Pass {
                pipeline: pipeline.clone(),
                bind,
                grid,
                bound_bytes,
                widest_bytes,
            });
            keep.push(Box::new(p));
            off += w;
        }
    }
    cvt(
        ctx, out, keep, &pl.widen, scratch, m, total_w, &site.segs, segs, true,
    );

    let meta = wk::lora::LoraMeta::prepare(&vec![0i32; m], 1);
    let ml = build_lora_meta(&meta, &site.widths, site.rank);
    let meta_buf = GpuTensor::<i32>::upload(ctx, "e4bw-lora-meta", &ml.data);
    let params = GpuUniform::new(
        ctx,
        "e4bw-lora-fused-p",
        &LoraFusedParams {
            m: m as u32,
            rank: site.rank as u32,
            k: site.k as u32,
            a_slice_stride: (site.rank * site.k) as u32,
            a_d0_stride: (site.rank * site.k) as u32,
            y_row_stride: total_w as u32,
            win_off: 0,
            win_len: total_w as u32,
            scale: 1.0,
            off_counts: ml.off_counts,
            off_start: ml.off_start,
            off_active: ml.off_active,
            off_slice_n: ml.off_slice_n,
            off_slice_start: ml.off_slice_start,
            off_b_off: ml.off_b_off,
            off_b_d0: ml.off_b_d0,
        },
    );
    let max_n = *site.widths.iter().max().expect("lora site has slices");
    let grid = (
        (m * max_n.div_ceil(wk::lora::FUSED_N_CHUNK as usize)) as u32,
        site.widths.len() as u32,
        meta.grid_loras() as u32,
    );
    let limit = ctx.caps.max_compute_workgroups_per_dimension;
    assert!(
        grid.0 <= limit && grid.1 <= limit && grid.2 <= limit,
        "lora fused grid {grid:?} exceeds device limit {limit}"
    );
    let binds: [(u32, &wgpu::Buffer); 6] = [
        (0, x),
        (1, site.a.raw()),
        (2, site.b.raw()),
        (3, scratch),
        (4, meta_buf.raw()),
        (5, params.raw()),
    ];
    let bind = dispatch::bind_group(ctx, &pl.fused, &binds);
    let (bound_bytes, widest_bytes) = bind_bytes(binds.iter().map(|(_, b)| *b));
    out.push(Pass {
        pipeline: pl.fused.clone(),
        bind,
        grid,
        bound_bytes,
        widest_bytes,
    });
    keep.push(Box::new(meta_buf));
    keep.push(Box::new(params));

    cvt(
        ctx, out, keep, &pl.repack, scratch, m, total_w, &site.segs, segs, false,
    );
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn lora_site_probe(
    x_bf16: &[u16],
    a_bf16: &[u16],
    b_bf16: &[u16],
    y_segments: &mut [Vec<u16>],
    widths: &[usize],
    segs: &[usize],
    m: usize,
    rank: usize,
    k: usize,
) -> Result<()> {
    let ctx = WgpuContext::shared().map_err(|e| anyhow::anyhow!("wgpu context: {e}"))?;
    let pl = build_lora_pipelines(ctx)?;
    let host = E4bLoraSite {
        a: a_bf16.to_vec(),
        b: b_bf16.to_vec(),
        widths: widths.to_vec(),
        segs: segs.to_vec(),
        k,
        rank,
    };
    let site = upload_lora_site(ctx, "e4bw-lora-probe", &host)?;
    anyhow::ensure!(
        x_bf16.len() == m * k,
        "lora probe x is {} elements, want {}",
        x_bf16.len(),
        m * k
    );
    anyhow::ensure!(y_segments.len() == segs.len(), "lora probe segment count");
    let x = GpuTensor::upload(ctx, "e4bw-lora-probe-x", &pack_pairs(x_bf16));
    let total_w: usize = segs.iter().sum();
    let scratch = GpuTensor::<u32>::zeroed(ctx, "e4bw-lora-probe-scratch", m * total_w);
    let mut ys = Vec::with_capacity(segs.len());
    for (i, w) in segs.iter().enumerate() {
        anyhow::ensure!(
            y_segments[i].len() == m * w,
            "lora probe segment {i} is {} elements, want {}",
            y_segments[i].len(),
            m * w
        );
        ys.push(GpuTensor::upload(
            ctx,
            "e4bw-lora-probe-y",
            &pack_pairs(&y_segments[i]),
        ));
    }
    let mut passes: Vec<Pass> = Vec::new();
    let mut keep: Vec<Box<dyn std::any::Any>> = Vec::new();
    let seg_refs: Vec<&wgpu::Buffer> = ys.iter().map(|t| t.raw()).collect();
    emit_lora_site(
        ctx,
        &mut passes,
        &mut keep,
        &pl,
        scratch.raw(),
        m,
        &site,
        x.raw(),
        &seg_refs,
    );
    let refs: Vec<dispatch::PassRef<'_>> = passes
        .iter()
        .map(|p| (p.pipeline.as_ref(), &p.bind, p.grid))
        .collect();
    dispatch::submit_pass_list(ctx, refs);
    ctx.poll_blocking().map_err(err)?;
    for (i, w) in segs.iter().enumerate() {
        let words = ys[i].download(ctx).map_err(err)?;
        for j in 0..(m * w) {
            let word = words[j / 2];
            y_segments[i][j] = if j.is_multiple_of(2) {
                (word & 0xffff) as u16
            } else {
                (word >> 16) as u16
            };
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct E4bLoraSite {
    pub a: Vec<u16>,
    pub b: Vec<u16>,
    pub widths: Vec<usize>,
    pub segs: Vec<usize>,
    pub k: usize,
    pub rank: usize,
}

impl E4bLoraSite {
    fn validate(&self, label: &str) -> Result<()> {
        anyhow::ensure!(!self.widths.is_empty(), "{label}: no lora slices");
        anyhow::ensure!(!self.segs.is_empty(), "{label}: no lora segments");
        anyhow::ensure!(
            self.segs.iter().sum::<usize>() == self.widths.iter().sum::<usize>(),
            "{label}: segment widths {:?} do not cover slice widths {:?}",
            self.segs,
            self.widths
        );
        anyhow::ensure!(self.rank > 0 && self.k > 0, "{label}: zero lora rank/k");
        anyhow::ensure!(
            self.rank <= wk::lora::FUSED_MAX_RANK,
            "{label}: rank {} exceeds kernel limit {}",
            self.rank,
            wk::lora::FUSED_MAX_RANK
        );
        anyhow::ensure!(
            self.a.len() == self.widths.len() * self.rank * self.k,
            "{label}: lora_a is {} elements, want {}",
            self.a.len(),
            self.widths.len() * self.rank * self.k
        );
        let want_b: usize = self.widths.iter().map(|w| w * self.rank).sum();
        anyhow::ensure!(
            self.b.len() == want_b,
            "{label}: lora_b is {} elements, want {want_b}",
            self.b.len()
        );
        anyhow::ensure!(self.k.is_multiple_of(2), "{label}: lora k must be even");
        for w in self.widths.iter().chain(self.segs.iter()) {
            anyhow::ensure!(
                w.is_multiple_of(2),
                "{label}: lora slice width must be even"
            );
        }
        Ok(())
    }

    fn total_width(&self) -> usize {
        self.widths.iter().sum()
    }

    fn pass_count(&self) -> usize {
        1 + 2 * self.segs.len()
    }
}

#[derive(Clone, Debug, Default)]
struct E4bLoraLayer {
    qkv: Option<E4bLoraSite>,
    o: Option<E4bLoraSite>,
    gate_up: Option<E4bLoraSite>,
    down: Option<E4bLoraSite>,
}

pub struct E4bLora {
    rank: usize,
    matched: usize,
    skipped: Vec<String>,
    layers: Vec<E4bLoraLayer>,
}

pub const E4B_LORA_TARGETS: [&str; 7] = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

impl E4bLora {
    pub fn from_peft_dir(dir: impl AsRef<std::path::Path>, config: &Gemma4Config) -> Result<Self> {
        let adapter =
            nv_weights::lora_adapter::LoraAdapter::load(dir.as_ref(), &candle_core::Device::Cpu)
                .with_context(|| format!("load lora adapter from {}", dir.as_ref().display()))?;
        Self::from_adapter(&adapter, config)
    }

    pub fn from_adapter(
        adapter: &nv_weights::lora_adapter::LoraAdapter,
        config: &Gemma4Config,
    ) -> Result<Self> {
        let rank = adapter.config.r;
        anyhow::ensure!(
            rank > 0 && rank <= wk::lora::FUSED_MAX_RANK,
            "lora rank {rank} outside 1..={}",
            wk::lora::FUSED_MAX_RANK
        );
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let n_q = config.num_attention_heads;
        let mut matched = 0usize;
        let mut wanted: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for li in 0..config.num_hidden_layers {
            let kind = config.layer_kind(li);
            let hd = config.head_dim_for(kind);
            let nkv = config.num_kv_heads_for(kind);
            let q_dim = n_q * hd;
            let kv_dim = nkv * hd;
            let shared = config.kv_source_layer(li);
            let has_v = !matches!(
                (kind, config.attention_k_eq_v),
                (LayerType::FullAttention, true)
            );
            let p = format!("{PREFIX}.layers.{li}");
            let q = format!("{p}.self_attn.q_proj");
            let kp = format!("{p}.self_attn.k_proj");
            let vp = format!("{p}.self_attn.v_proj");
            let qkv_slices: Vec<(&str, usize)> = match shared {
                Some(_) => vec![(q.as_str(), q_dim)],
                None if has_v => vec![
                    (q.as_str(), q_dim),
                    (kp.as_str(), kv_dim),
                    (vp.as_str(), kv_dim),
                ],
                None => vec![
                    (q.as_str(), q_dim),
                    (kp.as_str(), kv_dim),
                    (kp.as_str(), kv_dim),
                ],
            };
            let qkv_segs: Vec<usize> = qkv_slices.iter().map(|(_, w)| *w).collect();
            let (qkv, n) = build_lora_site(adapter, rank, hidden, &qkv_slices, &qkv_segs)?;
            matched += n;
            let op = format!("{p}.self_attn.o_proj");
            let (o, n) =
                build_lora_site(adapter, rank, q_dim, &[(op.as_str(), hidden)], &[hidden])?;
            matched += n;
            let gp = format!("{p}.mlp.gate_proj");
            let up = format!("{p}.mlp.up_proj");
            let (gate_up, n) = build_lora_site(
                adapter,
                rank,
                hidden,
                &[(gp.as_str(), inter), (up.as_str(), inter)],
                &[2 * inter],
            )?;
            matched += n;
            let dp = format!("{p}.mlp.down_proj");
            let (down, n) =
                build_lora_site(adapter, rank, inter, &[(dp.as_str(), hidden)], &[hidden])?;
            matched += n;
            for (name, _) in qkv_slices.iter() {
                wanted.insert((*name).to_string());
            }
            wanted.insert(op);
            wanted.insert(gp);
            wanted.insert(up);
            wanted.insert(dp);
            layers.push(E4bLoraLayer {
                qkv,
                o,
                gate_up,
                down,
            });
        }
        let layer_ns = format!("{PREFIX}.layers.");
        let skipped: Vec<String> = adapter
            .loras
            .keys()
            .filter(|n| {
                n.starts_with(&layer_ns)
                    && E4B_LORA_TARGETS.iter().any(|s| n.ends_with(s))
                    && !wanted.contains(*n)
            })
            .cloned()
            .collect();
        if !skipped.is_empty() {
            eprintln!(
                "[gemma4_e4b_wgpu] lora: {} text-tower modules have no projection in this graph \
                 (kv-shared layers compute no k/v); first is {}",
                skipped.len(),
                skipped[0]
            );
        }
        anyhow::ensure!(
            matched > 0,
            "lora adapter matched no gemma4-e4b text projection; it carries {} modules, \
             the first of which is {:?}. Expected names like {PREFIX}.layers.0.self_attn.q_proj",
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

    fn max_row_width(&self) -> usize {
        let mut m = 0usize;
        for l in &self.layers {
            for s in [&l.qkv, &l.o, &l.gate_up, &l.down].into_iter().flatten() {
                m = m.max(s.total_width());
            }
        }
        m
    }

    pub fn layer_pass_count(&self, layer: usize) -> usize {
        let l = &self.layers[layer];
        let mut n = 0usize;
        for s in [&l.qkv, &l.o, &l.gate_up, &l.down].into_iter().flatten() {
            n += s.pass_count();
        }
        n
    }

    pub fn total_pass_count(&self) -> usize {
        (0..self.layers.len())
            .map(|i| self.layer_pass_count(i))
            .sum()
    }
}

fn build_lora_site(
    adapter: &nv_weights::lora_adapter::LoraAdapter,
    rank: usize,
    k: usize,
    slices: &[(&str, usize)],
    segs: &[usize],
) -> Result<(Option<E4bLoraSite>, usize)> {
    use candle_core::{DType, Device};
    use nv_layers::lora_slots::LoraSlotStack;

    let mut matched = 0usize;
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut widths = Vec::with_capacity(slices.len());
    for (name, width) in slices {
        let stack = LoraSlotStack::new(1, rank, k, *width, DType::BF16, &Device::Cpu)
            .map_err(|e| anyhow::anyhow!("lora slot stack for {name}: {e}"))?;
        if let Some(w) = adapter.loras.get(*name) {
            anyhow::ensure!(
                !w.is_embedding,
                "{name}: embedding lora is not supported by the wgpu graph"
            );
            stack
                .set_lora(0, &w.lora_a, &w.lora_b)
                .map_err(|e| anyhow::anyhow!("set lora for {name}: {e}"))?;
            matched += 1;
        }
        a.extend_from_slice(&bf16_bits_of_tensor(stack.lora_a_stacked())?);
        b.extend_from_slice(&bf16_bits_of_tensor(stack.lora_b_stacked())?);
        widths.push(*width);
    }
    if matched == 0 {
        return Ok((None, 0));
    }
    let site = E4bLoraSite {
        a,
        b,
        widths,
        segs: segs.to_vec(),
        k,
        rank,
    };
    site.validate("lora site")?;
    Ok((Some(site), matched))
}

fn bf16_bits_of_tensor(t: &candle_core::Tensor) -> Result<Vec<u16>> {
    let v: Vec<half::bf16> = t.flatten_all()?.to_vec1()?;
    Ok(v.into_iter().map(|x| x.to_bits()).collect())
}

struct LoraSiteGpu {
    a: GpuTensor<u32>,
    b: GpuTensor<u32>,
    widths: Vec<usize>,
    segs: Vec<usize>,
    k: usize,
    rank: usize,
}

#[derive(Default)]
struct LayerLoraGpu {
    qkv: Option<LoraSiteGpu>,
    o: Option<LoraSiteGpu>,
    gate_up: Option<LoraSiteGpu>,
    down: Option<LoraSiteGpu>,
}

fn upload_lora_site(ctx: &WgpuContext, label: &str, site: &E4bLoraSite) -> Result<LoraSiteGpu> {
    site.validate(label)?;
    Ok(LoraSiteGpu {
        a: GpuTensor::upload(ctx, label, &pack_pairs(&site.a)),
        b: GpuTensor::upload(ctx, label, &pack_pairs(&site.b)),
        widths: site.widths.clone(),
        segs: site.segs.clone(),
        k: site.k,
        rank: site.rank,
    })
}

fn upload_lora_layer(ctx: &WgpuContext, l: &E4bLoraLayer) -> Result<LayerLoraGpu> {
    Ok(LayerLoraGpu {
        qkv: l
            .qkv
            .as_ref()
            .map(|s| upload_lora_site(ctx, "e4bw-lora-qkv", s))
            .transpose()?,
        o: l.o
            .as_ref()
            .map(|s| upload_lora_site(ctx, "e4bw-lora-o", s))
            .transpose()?,
        gate_up: l
            .gate_up
            .as_ref()
            .map(|s| upload_lora_site(ctx, "e4bw-lora-gate-up", s))
            .transpose()?,
        down: l
            .down
            .as_ref()
            .map(|s| upload_lora_site(ctx, "e4bw-lora-down", s))
            .transpose()?,
    })
}

#[cfg(test)]
mod w4_mk_unroll_tests {
    use super::{mk_widen, w4_mk_unrolled_source, GEMV_W4_PK_WGSL};

    const ENTRIES: [&str; 4] = [
        "g4w_gemm_w4a16_block_mk_pk",
        "g4w_gemm_w4a16_block_mk_pk3",
        "g4w_gemm_w4a16_v4_mk_pk",
        "g4w_gemm_w4a16_v4_mk_pk3",
    ];

    #[test]
    fn unrolled_twin_declares_every_entry_the_widened_source_does() {
        for m in [2usize, 8, 10, 16] {
            let widened = mk_widen(GEMV_W4_PK_WGSL, m);
            let unrolled = w4_mk_unrolled_source(m);
            for e in ENTRIES {
                assert!(
                    widened.contains(&format!("fn {e}(")),
                    "widened m={m} missing {e}"
                );
                assert!(
                    unrolled.contains(&format!("fn {e}(")),
                    "unrolled m={m} missing {e}"
                );
            }
        }
    }

    #[test]
    fn unrolled_twin_has_no_function_space_array() {
        for m in [2usize, 8, 10, 16] {
            let unrolled = w4_mk_unrolled_source(m);
            assert!(
                !unrolled.contains("array<f32,"),
                "m={m}: unrolled source still declares a function-space array"
            );
            assert!(
                !unrolled.contains("ptr<function"),
                "m={m}: unrolled source still passes an accumulator by pointer"
            );
            for t in 0..m {
                assert!(
                    unrolled.contains(&format!("var acc{t} = 0.0;")),
                    "m={m}: no acc{t}"
                );
            }
            assert!(
                !unrolled.contains(&format!("var acc{m} = 0.0;")),
                "m={m}: acc{m} overshoot"
            );
        }
    }

    #[test]
    fn widened_source_carries_the_spill_shape_the_twin_removes() {
        let widened = mk_widen(GEMV_W4_PK_WGSL, 10);
        assert!(widened.contains("ptr<function, array<f32, 10>>"));
        assert!(widened.contains("var blk: array<f32, 10>;"));
    }
}

#[cfg(test)]
mod w4_route_tests {
    use super::*;

    #[test]
    fn w4_route_predicate_matches_the_measured_e4b_ladder() {
        assert!(w4_prefer_v4(2560, 10240));
        assert!(!w4_prefer_v4(20480, 2560));
        assert!(!w4_prefer_v4(3072, 2560));
        assert!(!w4_prefer_v4(2560, 2048));
        assert!(!w4_prefer_v4(2560, 4096));
        assert!(w4_prefer_v4(5376, 21504));
    }

    fn w4(gs: usize, n: usize, k: usize) -> HostLin {
        HostLin::new_w4(vec![0; n * k / 8], vec![0; n * (k / gs)], gs, n, k)
    }

    fn layer(gs: usize) -> E4bHostLayer {
        let [qkv, o, gate_up, down, per_layer_input_gate, per_layer_projection] =
            std::array::from_fn(|_| w4(gs, 64, 256));
        E4bHostLayer {
            kind: LayerType::FullAttention,
            kv_source: None,
            input_ln: Vec::new(),
            post_attn_ln: Vec::new(),
            pre_ff_ln: Vec::new(),
            post_ff_ln: Vec::new(),
            post_per_layer_input_norm: Vec::new(),
            q_norm: Vec::new(),
            k_norm: Vec::new(),
            layer_scalar: 1.0,
            has_v: true,
            qkv,
            o,
            gate_up,
            down,
            per_layer_input_gate,
            per_layer_projection,
        }
    }

    fn host(layers: Vec<E4bHostLayer>, plmp: HostLin) -> E4bHostWeights {
        E4bHostWeights {
            embed: Vec::new(),
            embed_per_layer: Vec::new(),
            per_layer_model_projection: plmp,
            per_layer_projection_norm: Vec::new(),
            final_norm: Vec::new(),
            layers,
        }
    }

    fn gs_of(w: &E4bHostWeights) -> Option<usize> {
        uniform_w4_group_size(w.layers.len(), &WeightSource::Host(w))
    }

    #[test]
    fn one_group_size_per_checkpoint_or_no_baked_shift() {
        let w = host(vec![layer(32), layer(32)], w4(32, 64, 256));
        assert_eq!(gs_of(&w), Some(32));

        let mut mixed = host(vec![layer(32), layer(32)], w4(32, 64, 256));
        mixed.layers[1].down = w4(64, 64, 256);
        assert_eq!(gs_of(&mixed), None);

        let mut plmp_off = host(vec![layer(32)], w4(128, 64, 256));
        plmp_off.layers[0].o = w4(32, 64, 256);
        assert_eq!(gs_of(&plmp_off), None);

        let bf16 = {
            let mut w = host(vec![layer(64)], w4(64, 64, 256));
            w.layers[0].qkv = HostLin::new(vec![0; 64 * 256], 64, 256);
            w
        };
        assert_eq!(
            gs_of(&bf16),
            Some(64),
            "a bf16 tensor carries no group size"
        );

        let mut none = host(vec![layer(64)], HostLin::new(vec![0; 64 * 256], 64, 256));
        for l in none.layers.iter_mut() {
            for t in [
                &mut l.qkv,
                &mut l.o,
                &mut l.gate_up,
                &mut l.down,
                &mut l.per_layer_input_gate,
                &mut l.per_layer_projection,
            ] {
                *t = HostLin::new(vec![0; 64 * 256], 64, 256);
            }
        }
        assert_eq!(gs_of(&none), None);
    }

    #[test]
    fn the_baked_grain_accepts_exactly_what_it_was_built_for() {
        use wk::gemv_w4a16::ScaleGrain;
        for (gs, want) in [
            (32usize, ScaleGrain::Ge32Fixed(0)),
            (64, ScaleGrain::Ge32Fixed(1)),
            (128, ScaleGrain::Ge32Fixed(2)),
            (96, ScaleGrain::Ge32),
        ] {
            let w = host(vec![layer(gs)], w4(gs, 64, 256));
            let (grain, seen) = checkpoint_w4_grain(1, &WeightSource::Host(&w));
            assert_eq!(seen, Some(gs));
            assert_eq!(grain, want, "gs={gs}");
            assert!(grain.accepts(gs));
            assert!(wk::gemv_w4a16::require_grain(grain, gs).is_ok());
        }
        let g16 = host(vec![layer(16)], w4(16, 64, 256));
        assert_eq!(
            checkpoint_w4_grain(1, &WeightSource::Host(&g16)),
            (ScaleGrain::Ge32, Some(16))
        );
        let mixed = {
            let mut w = host(vec![layer(32)], w4(64, 64, 256));
            w.layers[0].down = w4(32, 64, 256);
            w
        };
        assert_eq!(
            checkpoint_w4_grain(1, &WeightSource::Host(&mixed)),
            (ScaleGrain::Ge32, None)
        );
    }

    #[test]
    fn a_group_size_no_sg_body_can_express_must_lose_sg_eligibility() {
        use wk::gemv_w4a16::ScaleGrain;
        assert!(wk::gemv_w4a16::shape_rule(96, 48).is_ok());
        assert_eq!(ScaleGrain::for_group_size(48), None);
        assert_eq!(ScaleGrain::fastest_for_group_size(48), None);
        assert!(!ScaleGrain::Ge32.accepts(48));
        let w = host(vec![layer(48)], w4(48, 64, 96 * 4));
        let (grain, seen) = checkpoint_w4_grain(1, &WeightSource::Host(&w));
        assert_eq!((grain, seen), (ScaleGrain::Ge32, Some(48)));
        assert!(wk::gemv_w4a16::require_grain(grain, 48).is_err());
    }
}

#[cfg(test)]
mod verify_smk_nozi_tests {
    use super::*;

    const SMK_POISON_PROLOGUE: &str =
        "    smk_partial[lid.x] = bitcast<f32>(0x7fc0deadu | smk_y[0]);\n    workgroupBarrier();\n";

    const SMK_TRIPWIRE_WGSL: &str = include_str!("../../nv-kernels/wgsl/e4b_smk_tripwire.wgsl");

    fn poisoned_twin(src: &str, entry: &str, twin: &str) -> String {
        let sig = format!("fn {entry}(");
        assert!(src.contains(&sig), "{entry} is not in the generated source");
        let out = src.replacen(&sig, &format!("fn {twin}("), 1);
        let open = ") {\n".to_string();
        let at = out.find(&format!("fn {twin}(")).expect("twin");
        let brace = out[at..].find(&open).expect("entry body") + at + open.len();
        let mut poisoned = String::with_capacity(out.len() + 128);
        poisoned.push_str(&out[..brace]);
        poisoned.push_str(SMK_POISON_PROLOGUE);
        poisoned.push_str(&out[brace..]);
        poisoned
    }

    fn ctx(what: &str) -> &'static WgpuContext {
        let ctx = WgpuContext::shared()
            .unwrap_or_else(|e| panic!("{what}: no wgpu adapter, this test cannot pass: {e}"));
        eprintln!("{what}: {}", ctx.summary());
        ctx
    }

    fn pipeline(
        ctx: &WgpuContext,
        label: &str,
        src: &str,
        entry: &str,
        zero_init: bool,
    ) -> wgpu::ComputePipeline {
        let scope = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
        let pl = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &[],
                    zero_initialize_workgroup_memory: zero_init,
                },
                cache: None,
            });
        assert!(
            pollster::block_on(scope.pop()).is_none(),
            "{label}:{entry} failed to compile"
        );
        pl
    }

    fn lcg(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u32
    }

    fn bf16_pairs(n: usize, seed: &mut u64) -> Vec<u32> {
        (0..n).map(|_| lcg(seed) & 0x3f7f_3f7f).collect()
    }

    fn params(n_rows: usize, k: usize, m: usize, groups_x: u32) -> SmkLiveParams {
        SmkLiveParams {
            n_rows: n_rows as u32,
            k_elems: k as u32,
            row_words: (k / 2) as u32,
            groups_x,
            m_live: m as u32,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        }
    }

    #[test]
    fn the_generated_verify_entry_is_not_reachable_through_the_audited_list() {
        assert!(
            !dispatch::nozi_entry_listed(SMK_LIVE_ENTRY, true),
            "{SMK_LIVE_ENTRY} joined NOZI_AUDITED_ENTRIES; NV_E4B_WGPU_NOZI_ALL=0 no longer \
             restores zero-init for it"
        );
        assert!(
            verify_smk_live_source(4).contains(&format!("fn {SMK_LIVE_ENTRY}(")),
            "the generated source no longer declares the entry the graph asks for"
        );
        assert!(
            verify_smk_live_source(4).contains("var<workgroup> smk_partial"),
            "the entry stopped using workgroup memory; if that is deliberate the audit \
             and the poison twin above need to say so rather than pass vacuously"
        );
    }

    #[test]
    fn verify_smk_live_is_write_before_read_under_poison() {
        let ctx = ctx("nozi_e4b_verify_smk_live");
        let k = 2560usize;
        let row_words = k / 2;
        let mut cells = 0usize;
        let mut tripwires = 0usize;
        for m in [1usize, 2, 4, 5, 8, 9] {
            let clean = verify_smk_live_source(m);
            let twin = poisoned_twin(&clean, SMK_LIVE_ENTRY, "smk_poisoned_twin");
            let trip_src = format!("{clean}\n{SMK_TRIPWIRE_WGSL}");
            let pl_clean = pipeline(ctx, "smk-clean", &clean, SMK_LIVE_ENTRY, false);
            let pl_twin = pipeline(ctx, "smk-twin", &twin, "smk_poisoned_twin", false);
            let pl_trip_clean = pipeline(ctx, "smk-trip-clean", &trip_src, "smk_trip_clean", true);
            let pl_trip_poisoned = pipeline(
                ctx,
                "smk-trip-poisoned",
                &trip_src,
                "smk_trip_poisoned",
                false,
            );
            for n_rows in [512usize, 517, 3] {
                let mut seed = 0x5e_6b_00_11u64 ^ (m as u64) ^ ((n_rows as u64) << 32);
                let w = bf16_pairs(n_rows * row_words, &mut seed);
                let x = bf16_pairs(m * row_words, &mut seed);
                let grid = dispatch::workgroup_count_1d(
                    ctx,
                    n_rows as u64,
                    wk::gemm_bf16_small_m::ROWS_PER_GROUP,
                );
                let p = dispatch::uniform_from(ctx, "smk-p", &params(n_rows, k, m, grid.0));
                let wb = dispatch::storage_from_slice(ctx, "smk-w", &w);
                let xb = dispatch::storage_from_slice(ctx, "smk-x", &x);
                let y_len = m * n_rows;
                let run = |pl: &wgpu::ComputePipeline, trip: bool| -> Vec<u32> {
                    let y = dispatch::storage_zeroed(ctx, "smk-y", (y_len * 4) as u64);
                    let binds: Vec<(u32, &wgpu::Buffer)> = if trip {
                        vec![(2, &y), (3, &p)]
                    } else {
                        vec![(0, &wb), (1, &xb), (2, &y), (3, &p)]
                    };
                    let bind = dispatch::bind_group(ctx, pl, &binds);
                    let mut enc = ctx.device.create_command_encoder(&Default::default());
                    {
                        let mut pass = enc.begin_compute_pass(&Default::default());
                        pass.set_pipeline(pl);
                        pass.set_bind_group(0, &bind, &[]);
                        pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                    }
                    ctx.queue.submit([enc.finish()]);
                    ctx.poll_blocking().expect("poll");
                    dispatch::read_back(ctx, &y, y_len).expect("read_back")
                };

                let t_clean = run(&pl_trip_clean, true);
                let t_poisoned = run(&pl_trip_poisoned, true);
                let moved = t_clean
                    .iter()
                    .zip(t_poisoned.iter())
                    .filter(|(a, b)| a != b)
                    .count();
                assert!(
                    moved > 0,
                    "m={m} n_rows={n_rows}: the poison prologue changed nothing for an entry \
                     that deliberately reads a slot it never wrote, so the parity result \
                     below would be vacuous"
                );
                tripwires += 1;

                let zi = run(&pl_clean, false);
                let nozi = run(&pl_twin, false);
                assert!(
                    zi.iter().any(|v| *v != 0),
                    "m={m} n_rows={n_rows}: the clean arm wrote nothing, parity is vacuous"
                );
                let bad: Vec<String> = zi
                    .iter()
                    .zip(nozi.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .take(4)
                    .map(|(i, (a, b))| format!("[{i}] clean {a:08x} vs poisoned {b:08x}"))
                    .collect();
                assert!(
                    bad.is_empty(),
                    "e4b_verify_smk_live m={m} n_rows={n_rows} read workgroup memory it had \
                     not written: {bad:?}"
                );
                cells += 1;
            }
        }
        assert_eq!(cells, 18, "ran {cells} parity cells");
        eprintln!(
            "e4b_verify_smk_live: {cells} cells bit-identical with every word of smk_partial \
             arriving as a signalling NaN, {tripwires} tripwires moved"
        );
    }

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    }

    #[derive(Clone, Copy)]
    struct Stat {
        lo: f64,
        med: f64,
        hi: f64,
    }

    struct Ab {
        ctx: &'static WgpuContext,
        qs: wgpu::QuerySet,
        resolve: wgpu::Buffer,
        staging: wgpu::Buffer,
        period: f64,
    }

    impl Ab {
        fn new(ctx: &'static WgpuContext) -> Self {
            assert!(
                ctx.caps.timestamp_query,
                "this measurement needs TIMESTAMP_QUERY; without it there is no number to \
                 report and a green run would mean nothing"
            );
            Self {
                ctx,
                qs: ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("smk-ab-qs"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                }),
                resolve: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("smk-ab-res"),
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
                staging: ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("smk-ab-stg"),
                    size: 16,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                period: ctx.queue.get_timestamp_period() as f64,
            }
        }

        fn once(
            &self,
            pl: &wgpu::ComputePipeline,
            bind: &wgpu::BindGroup,
            grid: (u32, u32, u32),
            reps: usize,
        ) -> f64 {
            let mut enc = self.ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                        query_set: &self.qs,
                        beginning_of_pass_write_index: Some(0),
                        end_of_pass_write_index: Some(1),
                    }),
                });
                for _ in 0..reps {
                    pass.set_pipeline(pl);
                    pass.set_bind_group(0, bind, &[]);
                    pass.dispatch_workgroups(grid.0, grid.1, grid.2);
                }
            }
            enc.resolve_query_set(&self.qs, 0..2, &self.resolve, 0);
            enc.copy_buffer_to_buffer(&self.resolve, 0, &self.staging, 0, 16);
            self.ctx.queue.submit([enc.finish()]);
            let slice = self.staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.ctx.poll_blocking().expect("poll");
            rx.recv().expect("map cb").expect("map");
            let ticks: Vec<u64> = {
                let v = slice.get_mapped_range().expect("range");
                bytemuck::cast_slice::<u8, u64>(&v).to_vec()
            };
            self.staging.unmap();
            ticks[1].saturating_sub(ticks[0]) as f64 * self.period / 1e6
        }

        fn sweep(
            &self,
            arms: &[(&str, &wgpu::ComputePipeline, &wgpu::BindGroup)],
            grid: (u32, u32, u32),
            reps_in_pass: usize,
            warm: usize,
            iters: usize,
        ) -> Vec<Stat> {
            let mut samples: Vec<Vec<f64>> = vec![Vec::new(); arms.len()];
            for it in 0..(warm + iters) {
                let order: Vec<usize> = (0..arms.len()).chain((0..arms.len()).rev()).collect();
                for i in order {
                    let ms = self.once(arms[i].1, arms[i].2, grid, reps_in_pass);
                    if it >= warm {
                        samples[i].push(ms / reps_in_pass as f64);
                    }
                }
            }
            samples
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    assert!(
                        s.iter().all(|v| *v > 0.0 && v.is_finite()),
                        "{}: timestamp resolve produced a non-positive interval",
                        arms[i].0
                    );
                    Stat {
                        lo: s.iter().cloned().fold(f64::MAX, f64::min),
                        med: median(s.clone()),
                        hi: s.iter().cloned().fold(0.0, f64::max),
                    }
                })
                .collect()
        }
    }

    #[test]
    fn verify_smk_live_memset_cost_at_serving_shape() {
        let ctx = ctx("nozi_e4b_verify_smk_live_ab");
        let ab = Ab::new(ctx);
        let k = 2560usize;
        let row_words = k / 2;
        let vocab = 262144usize;
        let chunks = 2usize;
        let n_rows = vocab / chunks;
        let m = 5usize;
        let src = verify_smk_live_source(m);
        let trip_src = format!("{src}\n{SMK_TRIPWIRE_WGSL}");

        let grid =
            dispatch::workgroup_count_1d(ctx, n_rows as u64, wk::gemm_bf16_small_m::ROWS_PER_GROUP);
        let p = dispatch::uniform_from(ctx, "smk-ab-p", &params(n_rows, k, m, grid.0));
        let mut seed = 0x1d_1d_be_efu64;
        let x = bf16_pairs(m * row_words, &mut seed);
        let w = bf16_pairs(1 << 20, &mut seed);
        let wb = dispatch::storage_zeroed(ctx, "smk-ab-w", (n_rows * row_words * 4) as u64);
        ctx.queue.write_buffer(&wb, 0, bytemuck::cast_slice(&w));
        let xb = dispatch::storage_from_slice(ctx, "smk-ab-x", &x);
        let y = dispatch::storage_zeroed(ctx, "smk-ab-y", (m * n_rows * 4) as u64);
        let binds: Vec<(u32, &wgpu::Buffer)> = vec![(0, &wb), (1, &xb), (2, &y), (3, &p)];

        let zi_flags = [true, false, true, false, true, false];
        let pls: Vec<wgpu::ComputePipeline> = zi_flags
            .iter()
            .enumerate()
            .map(|(i, zi)| pipeline(ctx, &format!("smk-ab-{i}"), &src, SMK_LIVE_ENTRY, *zi))
            .collect();
        let groups: Vec<wgpu::BindGroup> = pls
            .iter()
            .map(|pl| dispatch::bind_group(ctx, pl, &binds))
            .collect();
        let names = [
            "zero-init ON   #1",
            "zero-init OFF  #1",
            "zero-init ON   #2",
            "zero-init OFF  #2",
            "zero-init ON   #3",
            "zero-init OFF  #3",
        ];
        let arms: Vec<(&str, &wgpu::ComputePipeline, &wgpu::BindGroup)> =
            (0..6).map(|i| (names[i], &pls[i], &groups[i])).collect();
        let rounds_in_pass = 8usize;
        let st = ab.sweep(&arms, grid, chunks * rounds_in_pass, 2, 15);
        eprintln!(
            "  shape: {} lm_head rows over {chunks} chunks x {} groups, hidden {k}, m={m}; \
             each sample is {rounds_in_pass} verify rounds",
            vocab, grid.0
        );
        for (i, name) in names.iter().enumerate() {
            eprintln!(
                "  {name}  median {:.4} ms/chunk  (min {:.4}, max {:.4})",
                st[i].med, st[i].lo, st[i].hi
            );
        }
        let cond = |on: bool| -> (f64, f64) {
            let mut v: Vec<f64> = zi_flags
                .iter()
                .enumerate()
                .filter(|(_, z)| **z == on)
                .map(|(i, _)| st[i].med)
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (
                v[v.len() / 2],
                (v[v.len() - 1] - v[0]) / v[v.len() / 2] * 100.0,
            )
        };
        let (zi, zi_spread) = cond(true);
        let (nozi, nozi_spread) = cond(false);
        let effect = (zi - nozi) / zi * 100.0;
        let null = zi_spread.max(nozi_spread);
        eprintln!(
            "  memset on e4b_verify_smk_live: {:+.4} ms/verify round ({effect:+.2}%); \
             null (spread within a condition, 3 pipeline objects each) \
             {zi_spread:.2}% / {nozi_spread:.2}%",
            (zi - nozi) * chunks as f64
        );
        eprintln!(
            "  reading: the memset is at the edge of what this box can resolve -- one KiB of \
             workgroup memory per group against forty KiB of weight stream per group. Six \
             arms in one process put it at a few percent with a null of the same order; a \
             two-arm version of this harness flipped its sign run to run. Route the entry \
             for consistency with its siblings and for the audit; quote a serving number \
             only from a quiesced box, and only if {effect:+.2}% still clears {null:.2}%."
        );

        let trip_clean = pipeline(ctx, "smk-trip-clean", &trip_src, "smk_trip_clean", true);
        let trip_poisoned = pipeline(
            ctx,
            "smk-trip-poisoned",
            &trip_src,
            "smk_trip_poisoned",
            false,
        );
        let trip = |pl: &wgpu::ComputePipeline| -> Vec<u32> {
            let ty = dispatch::storage_zeroed(ctx, "smk-trip-y", (n_rows * 4) as u64);
            let bind = dispatch::bind_group(ctx, pl, &[(2, &ty), (3, &p)]);
            let mut enc = ctx.device.create_command_encoder(&Default::default());
            {
                let mut pass = enc.begin_compute_pass(&Default::default());
                pass.set_pipeline(pl);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(grid.0, grid.1, grid.2);
            }
            ctx.queue.submit([enc.finish()]);
            ctx.poll_blocking().expect("poll");
            dispatch::read_back::<u32>(ctx, &ty, n_rows).expect("trip readback")
        };
        let a = trip(&trip_clean);
        let b = trip(&trip_poisoned);
        let moved = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
        assert!(
            moved > 0,
            "zero_initialize_workgroup_memory did not reach the compiler: an entry that reads \
             workgroup memory it never wrote produced identical output with the flag on and \
             a poisoned prologue, so the arms above are one pipeline measured six times"
        );
        eprintln!(
            "  engagement control: the flag is live -- an unwritten workgroup read moved on \
             {moved}/{n_rows} rows"
        );
    }
}
