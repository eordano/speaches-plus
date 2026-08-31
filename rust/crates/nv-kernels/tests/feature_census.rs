use std::collections::BTreeSet;
use std::path::PathBuf;

fn compiled_features() -> Vec<&'static str> {
    let mut on: Vec<&'static str> = Vec::new();
    if cfg!(feature = "cuda") {
        on.push("cuda");
    }
    if cfg!(feature = "wgpu") {
        on.push("wgpu");
    }
    on
}

#[test]
fn features_compiled_into_this_binary() {
    let on = compiled_features();
    eprintln!(
        "nv-kernels tests compiled with features [{}]",
        if on.is_empty() {
            String::from("NONE")
        } else {
            on.join(", ")
        }
    );
    eprintln!(
        "cuda-only suites: {}; wgpu-only suites: {}; parity_*/scaling_* (needs both): {}",
        if cfg!(feature = "cuda") {
            "COMPILED"
        } else {
            "ABSENT"
        },
        if cfg!(feature = "wgpu") {
            "COMPILED"
        } else {
            "ABSENT"
        },
        if cfg!(all(feature = "cuda", feature = "wgpu")) {
            "COMPILED"
        } else {
            "ABSENT"
        },
    );
}

const SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES: &[&str] = &[];

const THE_ONLY_OPT_IN_IS_CARGOS_OWN_IGNORED_FLAG_NO_ENV_VAR_GATES_THIS_SUITE: &str = "--ignored";

const SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED: &[(&str, &str)] = &[
    ("flash_fp8_bandwidth.rs", "NV_FLASH_BW_BENCH"),
    (
        "gdn_chunk_gemm.rs",
        "NV_GDN_CHUNK_GEMM_TEST NV_GDN_CHUNK_GEMM_BENCH",
    ),
    (
        "gdn_chunked_parity.rs",
        "NV_GDN_CHUNK_TEST NV_GDN_CHUNK_BENCH",
    ),
    ("gdn_tree_parity.rs", "NV_GDN_CHUNK_TEST"),
    (
        "gdn_scan_v2.rs",
        "NV_GDN_SCANV2_TEST NV_GDN_SCANV2_BENCH",
    ),
    ("gemv_e4m3_group_identity.rs", "NV_E4M3_GROUP_TEST"),
    ("gemv_e4m3_group_ladder.rs", "NV_E4M3_GROUP_LADDER"),
    ("gdn_state_traffic_shape.rs", "NV_KERNELS_BENCH"),
    ("gemv_e4m3_mk_m_sweep_bench.rs", "NV_KERNELS_BENCH"),
    (
        "kv_fp8_paged_long_context.rs",
        THE_ONLY_OPT_IN_IS_CARGOS_OWN_IGNORED_FLAG_NO_ENV_VAR_GATES_THIS_SUITE,
    ),
    ("moe_grouped_fp4_gemv_bench.rs", "NV_MOE_M1_BENCH"),
    ("moe_grouped_fp4_gemv_parity.rs", "NV_MOE_M1_TEST"),
    ("q38_attn_lane_group_ladder.rs", "NV_Q38_ATTN_LANE_LADDER"),
    ("q38_decode_bucket_floor_bench.rs", "NV_KERNELS_BENCH"),
    ("q38_flash_depth_floor.rs", "NV_Q38_FLASH_FLOOR_TEST"),
    ("wgpu_d2d_copy_ceiling_probe.rs", "NV_D2D_CEILING_PROBE"),
    ("wgpu_dispatch_floor_probe.rs", "NV_DISPATCH_FLOOR_PROBE"),
    ("wgpu_dispatch_serialization_bench.rs", "NV_WGPU_SERBENCH"),
    ("wgpu_dispatch_timing_audit.rs", "NV_DISPATCH_AUDIT"),
];

const TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST: &[&str] = &[
    "#[test]",
    "#[test(",
    "#[tokio::test",
    "#[rstest",
    "#[test_case",
    "#[proptest",
    "#[bench",
];

fn is_a_test_attribute(t: &str) -> bool {
    TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST
        .iter()
        .any(|a| t.starts_with(a))
}

struct SourceFn {
    name: String,
    is_test: bool,
    is_ignored: bool,
    is_zero_arg_bool_fn: bool,
    body: Vec<String>,
    body_never_closed: bool,
}

fn braces_outside_char_literals(l: &str) -> (usize, usize) {
    let stripped = l.replace("'{'", "").replace("'}'", "");
    (
        stripped.matches('{').count(),
        stripped.matches('}').count(),
    )
}

fn source_fns(src: &str) -> Vec<SourceFn> {
    let lines: Vec<&str> = src.lines().collect();
    let mut out: Vec<SourceFn> = Vec::new();
    let mut attr_depth = 0i32;
    let mut in_attr = false;
    let mut saw_test = false;
    let mut saw_ignore = false;
    let mut i = 0usize;
    while i < lines.len() {
        let t = lines[i].trim();
        if in_attr {
            attr_depth += t.matches('[').count() as i32 - t.matches(']').count() as i32;
            in_attr = attr_depth > 0;
            i += 1;
            continue;
        }
        if t.starts_with("#[") {
            if is_a_test_attribute(t) {
                saw_test = true;
            }
            if t.starts_with("#[ignore") {
                saw_ignore = true;
            }
            attr_depth = t.matches('[').count() as i32 - t.matches(']').count() as i32;
            in_attr = attr_depth > 0;
            i += 1;
            continue;
        }
        if t.is_empty() {
            i += 1;
            continue;
        }
        let starts_a_fn = t.starts_with("fn ")
            || t.starts_with("pub fn ")
            || t.starts_with("async fn ")
            || t.starts_with("pub async fn ");
        if !starts_a_fn {
            saw_test = false;
            saw_ignore = false;
            i += 1;
            continue;
        }
        let name = t
            .split("fn ")
            .nth(1)
            .unwrap_or("")
            .split('(')
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        let mut body: Vec<String> = Vec::new();
        let mut signature = String::new();
        let mut d = 0i32;
        let mut started = false;
        let mut j = i;
        while j < lines.len() {
            let l = lines[j];
            if !started {
                signature.push_str(l);
            }
            body.push(l.to_string());
            let (opens, closes) = braces_outside_char_literals(l);
            d += opens as i32 - closes as i32;
            if opens > 0 {
                started = true;
            }
            if started && d <= 0 {
                break;
            }
            j += 1;
        }
        let closed = started && d <= 0;
        out.push(SourceFn {
            name,
            is_test: saw_test,
            is_ignored: saw_test && saw_ignore,
            is_zero_arg_bool_fn: !saw_test
                && signature.contains("()")
                && signature.contains("-> bool"),
            body,
            body_never_closed: !closed,
        });
        saw_test = false;
        saw_ignore = false;
        i = if closed { j + 1 } else { i + 1 };
    }
    out
}

fn declared_and_ignored_test_fns(src: &str) -> (usize, usize) {
    let fns = source_fns(src);
    (
        fns.iter().filter(|f| f.is_test).count(),
        fns.iter().filter(|f| f.is_ignored).count(),
    )
}

fn env_flag_helper_names(fns: &[SourceFn]) -> BTreeSet<String> {
    fns.iter()
        .filter(|f| f.is_zero_arg_bool_fn && f.body.iter().any(|l| l.contains("env::var")))
        .map(|f| f.name.clone())
        .collect()
}

const A_BLOCK_OPENS_WITH: char = '{';
const A_BLOCK_CLOSES_WITH: char = '}';

fn enclosing_guard(body: &[String], k: usize) -> String {
    let mut d = 0i32;
    let mut open = 0usize;
    for j in (0..=k).rev() {
        d += body[j].matches('}').count() as i32 - body[j].matches('{').count() as i32;
        if d < 0 {
            open = j;
            break;
        }
    }
    let mut first = open;
    while first > 0 && open - first < 8 {
        let prev = body[first - 1].trim_end();
        let ends_a_finished_statement = prev.is_empty()
            || prev.ends_with(';')
            || prev.ends_with(A_BLOCK_OPENS_WITH)
            || prev.ends_with(A_BLOCK_CLOSES_WITH);
        if ends_a_finished_statement {
            break;
        }
        first -= 1;
    }
    body[first..=open]
        .iter()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join(" ")
}

fn tests_that_return_green_when_an_opt_in_env_var_is_absent(src: &str) -> Vec<String> {
    let fns = source_fns(src);
    let flags = env_flag_helper_names(&fns);
    let mut out: Vec<String> = Vec::new();
    for f in fns.iter().filter(|f| f.is_test) {
        if f.body_never_closed {
            out.push(format!(
                "{}: this census could not find the end of the function body, so its returns were \
                 never examined; an unbalanced brace inside a string literal defeats the walk and \
                 an unexaminable suite must be an error, not a silent pass",
                f.name
            ));
            continue;
        }
        for (k, l) in f.body.iter().enumerate() {
            let t = l.trim();
            if t != "return;" && t != "return Ok(());" {
                continue;
            }
            let guard = enclosing_guard(&f.body, k);
            let gates_on_env = guard.contains("env::var")
                || flags.iter().any(|n| guard.contains(&format!("{n}()")));
            if !gates_on_env {
                continue;
            }
            let fires_only_when_the_variable_is_deliberately_set = (guard.contains("==")
                || guard.contains("is_ok()"))
                && !guard.contains("!=")
                && !guard.contains("is_err()")
                && !guard.contains("is_none()")
                && !flags.iter().any(|n| guard.contains(&format!("!{n}()")));
            if fires_only_when_the_variable_is_deliberately_set {
                continue;
            }
            out.push(format!("{} returns under the guard `{}`", f.name, guard.trim()));
        }
    }
    out
}

fn suite_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

const A_WALK_THAT_READS_FEWER_FILES_THAN_THIS_IS_BROKEN_NOT_CLEAN: usize = 140;

fn suite_sources() -> Vec<(String, String)> {
    let dir = suite_dir();
    let mut out: Vec<(String, String)> = Vec::new();
    let mut push = |p: &PathBuf| {
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            return;
        }
        let name = p
            .strip_prefix(&dir)
            .unwrap_or(p)
            .to_string_lossy()
            .to_string();
        let src = std::fs::read_to_string(p)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
        out.push((name, src));
    };
    for e in std::fs::read_dir(&dir).expect("suite directory").flatten() {
        let p = e.path();
        if p.is_dir() {
            for e2 in std::fs::read_dir(&p).expect("suite subdirectory").flatten() {
                push(&e2.path());
            }
        } else {
            push(&p);
        }
    }
    out.sort();
    out
}

#[derive(Default)]
struct Corpus {
    cuda_only: Vec<String>,
    cuda_and_wgpu: Vec<String>,
    wgpu_only: Vec<String>,
    unconditional: Vec<String>,
    cfg_header_hiding_below_the_first_line: Vec<String>,
    helpers: Vec<String>,
    files_declaring_no_tests_and_not_declared_helpers: Vec<String>,
    every_test_ignored: Vec<String>,
    subdirectory_targets_a_file_walk_cannot_see: Vec<String>,
    subdirectory_helper_modules: Vec<String>,
    subdirectories_cargo_builds_nothing_from: Vec<String>,
}

fn corpus() -> Corpus {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let rd = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read the suite directory {}: {e}", dir.display()));
    let mut c = Corpus::default();
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            let dname = p
                .file_name()
                .and_then(|s| s.to_str())
                .unwrap_or_default()
                .to_string();
            if p.join("main.rs").exists() {
                c.subdirectory_targets_a_file_walk_cannot_see.push(dname);
            } else if p.join("mod.rs").exists() {
                c.subdirectory_helper_modules.push(dname);
            } else {
                c.subdirectories_cargo_builds_nothing_from.push(dname);
            }
            continue;
        }
        if p.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let name = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or_default()
            .to_string();
        let src = std::fs::read_to_string(&p)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display()));
        let is_helper = SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES
            .contains(&name.as_str());
        let (declared, ignored) = declared_and_ignored_test_fns(&src);
        if is_helper {
            c.helpers.push(name.clone());
        } else if declared == 0 {
            c.files_declaring_no_tests_and_not_declared_helpers
                .push(name.clone());
        } else if declared == ignored {
            c.every_test_ignored.push(name.clone());
        }
        for (i, line) in src.lines().enumerate().skip(1) {
            if line.trim_start().starts_with("#![cfg(") {
                c.cfg_header_hiding_below_the_first_line.push(format!(
                    "{name}:{}: {}",
                    i + 1,
                    line.trim()
                ));
            }
        }
        if is_helper {
            continue;
        }
        let first = src.lines().next().unwrap_or_default().trim().to_string();
        match first.as_str() {
            "#![cfg(all(feature = \"cuda\", feature = \"wgpu\"))]" => c.cuda_and_wgpu.push(name),
            "#![cfg(feature = \"cuda\")]" => c.cuda_only.push(name),
            "#![cfg(feature = \"wgpu\")]" => c.wgpu_only.push(name),
            _ => c.unconditional.push(name),
        }
    }
    for v in [
        &mut c.cuda_only,
        &mut c.cuda_and_wgpu,
        &mut c.wgpu_only,
        &mut c.unconditional,
        &mut c.cfg_header_hiding_below_the_first_line,
        &mut c.helpers,
        &mut c.files_declaring_no_tests_and_not_declared_helpers,
        &mut c.every_test_ignored,
        &mut c.subdirectory_targets_a_file_walk_cannot_see,
        &mut c.subdirectory_helper_modules,
        &mut c.subdirectories_cargo_builds_nothing_from,
    ] {
        v.sort();
    }
    c
}

const CENSUS_CUDA_ONLY: &[&str] = &[
    "advkern_argmax_adv.rs",
    "advkern_splitk_race.rs",
    "advkern_window_boundary.rs",
    "cutlass_fp4_gemm_sm120.rs",
    "depthwise_conv1d_silu.rs",
    "flash_derivev_fp8.rs",
    "flash_fp8_bandwidth.rs",
    "flash_fp8_kvshare_parity.rs",
    "flash_splits_picker.rs",
    "fp8_mk_ratio_oracle.rs",
    "gather_rows_bf16.rs",
    "gdn_chunk_gemm.rs",
    "gdn_chunked_parity.rs",
    "gdn_qkvz_conv_fuse_parity.rs",
    "gdn_scan_v2.rs",
    "gdn_split_decode_parity.rs",
    "gdn_state_traffic_shape.rs",
    "gdn_state_vec4_arm.rs",
    "gdn_tree_parity.rs",
    "gemv_bf16_normed_dynk_parity.rs",
    "gemv_e4m3_group_identity.rs",
    "gemv_e4m3_group_ladder.rs",
    "gemv_e4m3_mk_dual_parity.rs",
    "gemv_e4m3_mk_m_sweep_bench.rs",
    "gemv_e4m3_mk_parity.rs",
    "gemv_e4m3_mk_row0_m_invariance.rs",
    "gemv_e4m3_qkv_one_parity.rs",
    "gemv_nvfp4_w4a16_decode_parity.rs",
    "gemv_nvfp4_w4a8_decode_parity.rs",
    "gemv_nvfp4_w4a8_dual_argmax_fused_parity.rs",
    "gemv_w4a16_cpu_ref.rs",
    "gemv_w4a16_m1_proto.rs",
    "gqa_hd512_verify.rs",
    "guard_refusal_conv.rs",
    "guard_refusal_gemv.rs",
    "guard_refusal_lora.rs",
    "guard_refusal_verify.rs",
    "kv_derive_v.rs",
    "kv_fp8_paged_long_context.rs",
    "laguna_flash_decode_gqa.rs",
    "lora_fused.rs",
    "lora_grouped.rs",
    "marlin_w4a16_cpu_ref.rs",
    "marlin_w4a8_fp8_cpu_ref.rs",
    "moe_gemv.rs",
    "moe_grouped_fp4_gemv_bench.rs",
    "moe_grouped_fp4_gemv_parity.rs",
    "moe_grouped_fp4_multi_expert.rs",
    "moe_mchain_mrow_parity.rs",
    "moe_permute.rs",
    "moe_route_gather_quant_small_m.rs",
    "moe_unpermute_scatter.rs",
    "parity_verify_fused.rs",
    "pdl_gemv_bf16_ab.rs",
    "q38_attn_lane_group_ladder.rs",
    "q38_decode_bucket_floor_bench.rs",
    "q38_flash_depth_floor.rs",
    "quantize_nvfp4_bf16.rs",
    "quantize_nvfp4_bf16_per_expert.rs",
    "quantize_nvfp4_bf16_rows.rs",
    "rmsnorm.rs",
    "rmsnorm_residual_writeout_parity.rs",
    "rope.rs",
    "sampler.rs",
    "silu.rs",
    "silu_mul_quantize_nvfp4.rs",
    "smem_optin_never_lowers.rs",
    "tree_verify_attn.rs",
    "ue4m3_log2f_boundary.rs",
];

const CENSUS_CUDA_AND_WGPU: &[&str] = &[
    "parity_gdn.rs",
    "parity_gemv_bf16_i8.rs",
    "parity_kv_fp8_paged.rs",
];

const CENSUS_WGPU_ONLY: &[&str] = &[
    "flash_decode_sd_twins.rs",
    "fp8_contract_e4m3.rs",
    "gdn_scan_w32.rs",
    "kernel_forge_gemv_w4a16.rs",
    "qwen35_opt_gemv_route.rs",
    "wgpu_argmax_softcap_fold.rs",
    "wgpu_attn_decode.rs",
    "wgpu_attn_small_m.rs",
    "wgpu_attn_small_m_fp8.rs",
    "wgpu_attn_small_m_v2.rs",
    "wgpu_coop_matrix_probe.rs",
    "wgpu_correct_gemv_nvfp4_variants.rs",
    "wgpu_d2d_copy_ceiling_probe.rs",
    "wgpu_depthwise_conv1d_silu_bf16.rs",
    "wgpu_dequant.rs",
    "wgpu_dispatch_floor_probe.rs",
    "wgpu_dispatch_serialization_bench.rs",
    "wgpu_dispatch_timing_audit.rs",
    "wgpu_flash_attn_fp8_oracle.rs",
    "wgpu_flash_decode.rs",
    "wgpu_flash_stage1_mk_unroll.rs",
    "wgpu_flash_stage2_unroll.rs",
    "wgpu_flash_zeroinit_ab.rs",
    "wgpu_fp8_group_gemv.rs",
    "wgpu_fused_attn_chain.rs",
    "wgpu_fused_norm_chain.rs",
    "wgpu_gather_rows_bf16.rs",
    "wgpu_gelu_tanh_mul.rs",
    "wgpu_gemm_bf16_small_m.rs",
    "wgpu_gemm_coop_prefill.rs",
    "wgpu_gemm_nvfp4.rs",
    "wgpu_gemm_w4a16_small_m.rs",
    "wgpu_gemv_bf16.rs",
    "wgpu_gemv_bf16_sg_pk.rs",
    "wgpu_gemv_i8_epilogue.rs",
    "wgpu_gemv_multicol.rs",
    "wgpu_gemv_nvfp4.rs",
    "wgpu_gemv_nvfp4_decode.rs",
    "wgpu_gemv_nvfp4_scale_layout.rs",
    "wgpu_gemv_nvfp4_v2_routed_pk.rs",
    "wgpu_gemv_nvfp4_variants.rs",
    "wgpu_gemv_nvfp4_w4a8.rs",
    "wgpu_gemv_q8dot.rs",
    "wgpu_gemv_w4a16.rs",
    "wgpu_gemv_w4a16_31b.rs",
    "wgpu_gemv_w4a16_group_sizes.rs",
    "wgpu_gemv_w4a16_sg_pk.rs",
    "wgpu_graph_decode.rs",
    "wgpu_group_gelu_fold.rs",
    "wgpu_kv_fp8.rs",
    "wgpu_kv_nvfp4.rs",
    "wgpu_lora.rs",
    "wgpu_moe_grouped.rs",
    "wgpu_moe_permute.rs",
    "wgpu_moe_unpermute_scatter.rs",
    "wgpu_msl_memory_census.rs",
    "wgpu_na_attn.rs",
    "wgpu_na_gemm.rs",
    "wgpu_na_gemm_bf16.rs",
    "wgpu_nozi_poison_parity_audit.rs",
    "wgpu_pf_tiled_v2.rs",
    "wgpu_q3d_fp8_lmhead.rs",
    "wgpu_quant_gemv_unpack_probe.rs",
    "wgpu_quantize_nvfp4_bf16.rs",
    "wgpu_quantize_nvfp4_bf16_cpu_ref.rs",
    "wgpu_quantize_nvfp4_ue4m3_subnormal.rs",
    "wgpu_residual_scale.rs",
    "wgpu_rmsnorm.rs",
    "wgpu_rmsnorm_residual.rs",
    "wgpu_rope.rs",
    "wgpu_rope_bf16.rs",
    "wgpu_sampler.rs",
    "wgpu_shift_decode.rs",
    "wgpu_silu.rs",
    "wgpu_spirv_passthrough_probe.rs",
    "wgpu_spirv_safety_tax_census.rs",
    "wgpu_subgroup_portability.rs",
    "wgpu_verify_fused_norms.rs",
    "wgpu_w4a8_producer_rowquant.rs",
    "wgpu_wmma_accum_model.rs",
];

const CENSUS_UNCONDITIONAL_THESE_SUITES_READ_SOURCE_TEXT_NOT_A_DEVICE: &[&str] = &[
    "cuda_ue4m3_subnormal_codec.rs",
    "feature_census.rs",
    "fp8_rowscale_oracle_fuzz.rs",
    "gdn_chunk_gemm_host_oracle.rs",
    "gdn_host_oracle_fuzz.rs",
    "laguna_fp8_contract.rs",
    "prompt_fidelity.rs",
    "shift_decode_oracle.rs",
    "w4a16_oracle_fuzz.rs",
    "wgsl_ue4m3_subnormal_census.rs",
];

const EXPECT_CUDA_ONLY: usize = CENSUS_CUDA_ONLY.len();
const EXPECT_CUDA_AND_WGPU: usize = CENSUS_CUDA_AND_WGPU.len();
const EXPECT_WGPU_ONLY: usize = CENSUS_WGPU_ONLY.len();
const EXPECT_UNCONDITIONAL: usize =
    CENSUS_UNCONDITIONAL_THESE_SUITES_READ_SOURCE_TEXT_NOT_A_DEVICE.len();

const UNCONDITIONAL_SUITES_STAY_UNDER_A_TENTH_OF_THE_CORPUS: usize = 10;

fn census_delta(
    class: &str,
    recorded_in: &str,
    on_disk: &[String],
    recorded: &[&str],
) -> Option<String> {
    let disk: BTreeSet<&str> = on_disk.iter().map(String::as_str).collect();
    let rec: BTreeSet<&str> = recorded.iter().copied().collect();
    let added: Vec<&str> = disk.difference(&rec).copied().collect();
    let dropped: Vec<&str> = rec.difference(&disk).copied().collect();
    if added.is_empty() && dropped.is_empty() {
        return None;
    }
    let mut m = format!(
        "{class}: {} suites on disk, {} recorded in {recorded_in}",
        disk.len(),
        rec.len()
    );
    for n in added {
        m.push_str(&format!(
            "\n      + {n} is on disk and not in {recorded_in}. The whole fix is one line: add \
             \"{n}\", to {recorded_in}. Name the commit that added it with\n          git log \
             --oneline --diff-filter=A -- rust/crates/nv-kernels/tests/{n}"
        ));
    }
    for n in dropped {
        m.push_str(&format!(
            "\n      - {n} is in {recorded_in} and not on disk. The whole fix is one line: delete \
             \"{n}\", from {recorded_in}. Name the commit that removed it with\n          git log \
             --oneline --diff-filter=D -- rust/crates/nv-kernels/tests/{n}"
        ));
    }
    Some(m)
}

fn out_of_order(recorded_in: &str, recorded: &[&str]) -> Option<String> {
    for w in recorded.windows(2) {
        if w[0] >= w[1] {
            return Some(format!(
                "{recorded_in} is not in strictly ascending order: {:?} precedes {:?}. Keep it \
                 sorted, or two branches that each add a suite will both append and the duplicate \
                 will read as a correct count.",
                w[0], w[1]
            ));
        }
    }
    None
}

#[test]
fn the_suite_corpus_is_the_one_this_census_names_file_by_file() {
    let c = corpus();
    let classes: [(&str, &str, &[String], &[&str]); 4] = [
        (
            "cuda-only",
            "CENSUS_CUDA_ONLY",
            c.cuda_only.as_slice(),
            CENSUS_CUDA_ONLY,
        ),
        (
            "cuda+wgpu",
            "CENSUS_CUDA_AND_WGPU",
            c.cuda_and_wgpu.as_slice(),
            CENSUS_CUDA_AND_WGPU,
        ),
        (
            "wgpu-only",
            "CENSUS_WGPU_ONLY",
            c.wgpu_only.as_slice(),
            CENSUS_WGPU_ONLY,
        ),
        (
            "unconditional",
            "CENSUS_UNCONDITIONAL_THESE_SUITES_READ_SOURCE_TEXT_NOT_A_DEVICE",
            c.unconditional.as_slice(),
            CENSUS_UNCONDITIONAL_THESE_SUITES_READ_SOURCE_TEXT_NOT_A_DEVICE,
        ),
    ];
    let total =
        c.cuda_only.len() + c.cuda_and_wgpu.len() + c.wgpu_only.len() + c.unconditional.len();
    eprintln!(
        "nv-kernels suite corpus: {total} files on disk = {} cuda-only + {} cuda+wgpu + {} \
         wgpu-only + {} unconditional; this census records {} + {} + {} + {}",
        c.cuda_only.len(),
        c.cuda_and_wgpu.len(),
        c.wgpu_only.len(),
        c.unconditional.len(),
        EXPECT_CUDA_ONLY,
        EXPECT_CUDA_AND_WGPU,
        EXPECT_WGPU_ONLY,
        EXPECT_UNCONDITIONAL,
    );
    eprintln!("  cuda+wgpu: {:?}", c.cuda_and_wgpu);
    eprintln!("  unconditional: {:?}", c.unconditional);

    let mut wrong: Vec<String> = Vec::new();
    for (class, recorded_in, on_disk, recorded) in classes {
        wrong.extend(out_of_order(recorded_in, recorded));
        wrong.extend(census_delta(class, recorded_in, on_disk, recorded));
    }
    assert!(
        wrong.is_empty(),
        "the suite corpus moved without this census moving with it.\n  {}\n\nEvery line above is \
         its own fix and its own investigation: the name to add or delete, the constant to edit, \
         and the git command that names the commit responsible. Make the edit in the SAME commit \
         as the suite and put those names in the message. This census records file NAMES rather \
         than counts precisely so that going stale costs one line instead of a bisect -- it has \
         been red on landed work four times, and every time the count alone said only that a \
         number had moved.",
        wrong.join("\n  ")
    );
}

#[test]
fn the_unconditional_class_stays_small_and_no_cfg_header_hides_below_the_first_line() {
    let c = corpus();
    assert!(
        c.cfg_header_hiding_below_the_first_line.is_empty(),
        "an inner `#![cfg]` attribute appears below line 1, where this census cannot see it: \
         {:?}.\nClassification reads the FIRST line only, so such a suite is filed \
         `unconditional` and its class silently loses a member -- which is what a `//!` header \
         above the attribute, or a spelling this census does not match, looks like from here. \
         Move the attribute to line 1, or teach `corpus()` the new spelling.",
        c.cfg_header_hiding_below_the_first_line,
    );
    assert!(
        c.unconditional.len() * UNCONDITIONAL_SUITES_STAY_UNDER_A_TENTH_OF_THE_CORPUS
            <= c.cuda_only.len()
                + c.cuda_and_wgpu.len()
                + c.wgpu_only.len()
                + c.unconditional.len(),
        "{} of the suites in this directory compile unconditionally: {:?}. That class exists for \
         suites whose oracle is SOURCE TEXT -- they scan cuda/ or wgsl/ or prompts and must still \
         speak when the matching feature is off -- and a suite that touches a device does not \
         belong in it. Give the new one the `#![cfg]` header its class carries, or say in the \
         commit message why it reads text rather than a GPU.",
        c.unconditional.len(),
        c.unconditional,
    );
    assert!(
        c.wgpu_only.len() > 40 && c.cuda_only.len() > 20,
        "the two large classes collapsed ({} wgpu-only, {} cuda-only), which is what a changed \
         header spelling looks like from here",
        c.wgpu_only.len(),
        c.cuda_only.len()
    );
    eprintln!(
        "subdirectories of tests/: shared mod.rs (compiled into each dependent suite, never a \
         target) {:?}; neither main.rs nor mod.rs, so cargo builds nothing from them {:?}",
        c.subdirectory_helper_modules, c.subdirectories_cargo_builds_nothing_from,
    );
    assert!(
        c.subdirectory_targets_a_file_walk_cannot_see.is_empty(),
        "these subdirectories of tests/ hold a main.rs: {:?}.\ncargo builds tests/<dir>/main.rs as \
         its own test target, and this census walks FILES in tests/ -- so such a target is exactly \
         the suite-that-never-runs this file exists to make impossible, hiding one directory level \
         below where anyone would look. Either flatten it to tests/<dir>.rs, or teach corpus() to \
         descend and classify it.",
        c.subdirectory_targets_a_file_walk_cannot_see,
    );
}

#[test]
fn every_file_excluded_as_a_shared_helper_declares_no_tests_and_is_path_included_by_a_real_suite() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let mut wrong: Vec<String> = Vec::new();
    for helper in SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES {
        let p = dir.join(helper);
        let Ok(src) = std::fs::read_to_string(&p) else {
            wrong.push(format!(
                "{helper} is excluded from the census as a shared helper but does not exist"
            ));
            continue;
        };
        let (declared, _) = declared_and_ignored_test_fns(&src);
        if declared != 0 {
            wrong.push(format!(
                "{helper} is excluded from the census as a shared helper but declares {declared} \
                 #[test] fn(s); a file that asserts is a suite and belongs in a census class"
            ));
        }
        let include = format!("#[path = \"{helper}\"]");
        let included_by: Vec<String> = std::fs::read_dir(&dir)
            .expect("suite directory")
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("rs"))
            .filter(|p| p.file_name().and_then(|s| s.to_str()) != Some(helper))
            .filter(|p| {
                std::fs::read_to_string(p)
                    .map(|s| s.contains(&include))
                    .unwrap_or(false)
            })
            .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(String::from))
            .collect();
        if included_by.is_empty() {
            wrong.push(format!(
                "{helper} is excluded from the census as a shared helper but no suite carries \
                 {include}; an unreferenced helper is dead weight cargo still builds as its own \
                 test target, and excluding it hides that"
            ));
        } else {
            eprintln!("{helper} is path-included by {included_by:?}");
        }
    }
    assert!(wrong.is_empty(), "{}", wrong.join("\n  "));

    let c = corpus();
    assert!(
        c.files_declaring_no_tests_and_not_declared_helpers.is_empty(),
        "these files in tests/ declare no #[test] fn this census can see, yet are not named in \
         SHARED_HELPER_MODULES_PATH_INCLUDED_BY_SUITES_AND_DECLARING_NO_TESTS_THEMSELVES: \
         {:?}.\ncargo builds every tests/*.rs as its own binary, so such a file prints `ok. 0 \
         passed` and reads as a passing suite forever. Either it is a helper -- name it, and this \
         census will prove the claim -- or it declares its tests through an attribute spelling \
         outside TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST, which \
         already carries #[test], #[tokio::test], #[rstest], #[test_case], #[proptest] and \
         #[bench] -- add the new spelling there rather than excluding the file.",
        c.files_declaring_no_tests_and_not_declared_helpers,
    );
}

#[test]
fn every_suite_whose_tests_are_all_ignored_is_named_here_with_the_env_var_that_opts_it_in() {
    let c = corpus();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let recorded: BTreeSet<&str> = SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED
        .iter()
        .map(|(n, _)| *n)
        .collect();
    let found: BTreeSet<&str> = c.every_test_ignored.iter().map(String::as_str).collect();
    let mut wrong: Vec<String> = Vec::new();
    for n in found.difference(&recorded) {
        wrong.push(format!(
            "+ {n}: every #[test] in it is #[ignore]d, so `nvk.sh test` prints `ok. 0 passed` for \
             it and a green crate says nothing about the property it gates. Add it to \
             SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED with the env \
             var that runs it."
        ));
    }
    for n in recorded.difference(&found) {
        wrong.push(format!(
            "- {n}: recorded as entirely-#[ignore]d, but it now has at least one test that runs \
             by default (or the file is gone). Drop the entry."
        ));
    }
    for (n, vars) in SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED {
        let Ok(src) = std::fs::read_to_string(dir.join(n)) else {
            continue;
        };
        for var in vars.split_whitespace() {
            if var == THE_ONLY_OPT_IN_IS_CARGOS_OWN_IGNORED_FLAG_NO_ENV_VAR_GATES_THIS_SUITE {
                continue;
            }
            if !src.contains(var) {
                wrong.push(format!(
                    "! {n} is recorded as opting in through {var}, which does not appear in its \
                     source. A documented opt-in nobody can perform is worse than none."
                ));
            }
        }
    }
    assert!(
        wrong.is_empty(),
        "the entirely-ignored suites moved:\n  {}\n\n{} of this crate's {} suites are in this \
         class: present, registered, compiled, and still proving nothing until someone sets a \
         variable. Most are doubly silent -- the #[ignore] hides them from a plain run, and then \
         the env check inside prints `SKIP: ...` and RETURNS, so even `--ignored` reports a pass \
         having executed no kernel. flash_fp8_bandwidth and q38_attn_lane_group_ladder panic \
         instead, which is the shape the rest should take.",
        wrong.join("\n  "),
        SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED.len(),
        c.cuda_only.len() + c.cuda_and_wgpu.len() + c.wgpu_only.len() + c.unconditional.len(),
    );
    eprintln!(
        "PRESENT BUT SILENT UNLESS OPTED IN: {:?}",
        SUITES_WHOSE_EVERY_TEST_IS_IGNORED_SO_A_PLAIN_RUN_PRINTS_OK_0_PASSED
    );
}

#[test]
fn no_suite_reports_a_pass_by_returning_when_an_opt_in_env_var_it_gates_itself_on_is_absent() {
    let sources = suite_sources();
    assert!(
        sources.len() >= A_WALK_THAT_READS_FEWER_FILES_THAN_THIS_IS_BROKEN_NOT_CLEAN,
        "this gate read only {} files under {}; a walk that finds nothing proves nothing",
        sources.len(),
        suite_dir().display(),
    );
    let mut wrong: Vec<String> = Vec::new();
    for (name, src) in &sources {
        for m in tests_that_return_green_when_an_opt_in_env_var_is_absent(src) {
            wrong.push(format!("{name}: {m}"));
        }
    }
    assert!(
        wrong.is_empty(),
        "these tests return instead of failing when the environment variable they gate \
         themselves on is absent, so the runner counts them as PASSED having executed \
         nothing:\n  {}\n\nA suite whose precondition is absent must fail or be ignored, never \
         return green. An `#[ignore]`d suite at least prints `0 passed`, which this census can \
         assert on; a skip-and-return prints `1 passed` and reads as coverage. The fix is one of \
         two shapes: panic! with a message naming the variable (correct when the suite is \
         already #[ignore]d, because --ignored IS the opt-in and a second gate behind it can \
         only make --ignored lie), or #[ignore] plus that same panic (correct when the suite \
         still runs by default). A guard that fires only when a variable is deliberately SET -- \
         an explicit NV_*_ALLOW_SKIP opt-out -- is not this defect and is not reported here.",
        wrong.join("\n  ")
    );
}

#[test]
fn the_skip_return_scanner_reads_the_absent_direction_and_leaves_a_deliberate_opt_out_alone() {
    let fixture = "fn device_or_none() -> Option<usize> {\n\
                   \x20   None\n\
                   }\n\
                   \n\
                   fn bench_gated() -> bool {\n\
                   \x20   std::env::var(\"NV_X_BENCH\").as_deref() == Ok(\"1\")\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn skips_on_a_direct_env_absence() {\n\
                   \x20   if std::env::var(\"NV_X_TEST\").as_deref() != Ok(\"1\") {\n\
                   \x20       eprintln!(\"SKIP: set NV_X_TEST=1\");\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn skips_through_a_bool_returning_env_flag_helper() {\n\
                   \x20   if !bench_gated() {\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn skips_only_when_a_variable_is_deliberately_set() {\n\
                   \x20   if std::env::var(\"NV_X_ALLOW_SKIP\").as_deref() == Ok(\"1\") {\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn returns_early_on_a_fact_about_the_machine_not_a_variable() {\n\
                   \x20   let n = 3usize;\n\
                   \x20   if n == 0 {\n\
                   \x20       return;\n\
                   \x20   }\n\
                   \x20   assert!(true);\n\
                   }\n\
                   \n\
                   #[test]\n\
                   fn asserts_on_the_variable_then_returns_on_an_absent_device() {\n\
                   \x20   assert_eq!(\n\
                   \x20       std::env::var(\"NV_X_BENCH\").ok().as_deref(),\n\
                   \x20       Some(\"1\"),\n\
                   \x20       \"set NV_X_BENCH=1 -- this loads the GPU\"\n\
                   \x20   );\n\
                   \x20   let Some(_d) = device_or_none() else {\n\
                   \x20       return;\n\
                   \x20   };\n\
                   \x20   assert!(true);\n\
                   }\n";
    let found = tests_that_return_green_when_an_opt_in_env_var_is_absent(fixture);
    assert!(
        found.iter().any(|f| f.contains("skips_on_a_direct_env_absence")),
        "the scanner must see a direct env::var guard whose absent branch returns: {found:?}"
    );
    assert!(
        found
            .iter()
            .any(|f| f.contains("skips_through_a_bool_returning_env_flag_helper")),
        "a zero-argument helper returning bool from an env::var read is the same gate one call \
         away; a scanner that only reads env::var textually misses every `if !bench_gated()`: \
         {found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|f| f.contains("skips_only_when_a_variable_is_deliberately_set")),
        "a guard that fires only when a variable IS set is a deliberate, named opt-out, not a \
         suite going green because a precondition is missing: {found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|f| f.contains("returns_early_on_a_fact_about_the_machine_not_a_variable")),
        "an early return that names no environment variable is out of this gate's scope: \
         {found:?}"
    );
    assert!(
        !found
            .iter()
            .any(|f| f.contains("asserts_on_the_variable_then_returns_on_an_absent_device")),
        "the guard this scanner reads is the block the return sits in, not the six lines above \
         it. A suite that asserts on its variable and then returns because the DEVICE is absent \
         is already the correct shape; reading the assert_eq! above the `let ... else` as the \
         guard makes every such suite a false positive, which is how a gate stops being \
         believed: {found:?}"
    );
    assert_eq!(found.len(), 2, "exactly two of the five fixtures qualify: {found:?}");
}

#[test]
fn the_attribute_scanner_reads_multi_line_ignore_attributes_and_ignores_them_inside_strings() {
    let fixture = "#![cfg(feature = \"cuda\")]\n\
                   #[test]\n\
                   fn plain() {}\n\
                   \n\
                   #[test]\n\
                   #[ignore]\n\
                   fn bare_ignore() {}\n\
                   \n\
                   #[test]\n\
                   #[ignore = \"needs a device; set NV_X=1 -- proves \\\n\
                   something across two lines\"]\n\
                   fn multi_line_ignore() {}\n\
                   \n\
                   fn helper_that_only_mentions_attributes() {\n\
                       panic!(\"this is already #[ignore]d, so --ignored IS the opt-in\");\n\
                   }\n";
    assert_eq!(
        declared_and_ignored_test_fns(fixture),
        (3, 2),
        "the scanner must count three #[test] fns of which two are ignored: a bare #[ignore], a \
         multi-line #[ignore = \"...\"] whose closing bracket lands on a later line, and it must \
         not be fooled by a `#[ignore]` mentioned inside a panic message"
    );

    let attribute_macro_forms = "#[tokio::test]\n\
                                 async fn plain_tokio() {}\n\
                                 \n\
                                 #[tokio::test(flavor = \"multi_thread\", worker_threads = 2)]\n\
                                 #[ignore = \"boots the real binary\"]\n\
                                 async fn configured_tokio() {}\n\
                                 \n\
                                 #[rstest]\n\
                                 fn a_fixture_driven_case() {}\n\
                                 \n\
                                 #[test_case(1)]\n\
                                 fn a_table_driven_case() {}\n\
                                 \n\
                                 fn a_plain_helper() {}\n";
    let char_literal_braces = "fn depth_of(s: &str) -> usize {\n\
                               \x20   let mut depth = 0usize;\n\
                               \x20   for b in s.bytes() {\n\
                               \x20       match b {\n\
                               \x20           b'{' => depth += 1,\n\
                               \x20           b'}' => depth -= 1,\n\
                               \x20           _ => {}\n\
                               \x20       }\n\
                               \x20   }\n\
                               \x20   depth\n\
                               }\n\
                               \n\
                               #[test]\n\
                               fn a_suite_declared_after_that_helper() {\n\
                               \x20   assert_eq!(depth_of(\"\"), 0);\n\
                               }\n";
    assert_eq!(
        declared_and_ignored_test_fns(char_literal_braces),
        (1, 0),
        "a `b'{{'` or `'}}'` char literal is not a block delimiter. Counting it as one leaves the \
         helper above looking unterminated, the walk swallows every test below it, and the file \
         reads as a test-free helper -- which is how nv-kernels' cuda_ue4m3_subnormal_codec.rs, \
         a suite with eight tests, went invisible the first time this scanner ran."
    );
    assert_eq!(
        declared_and_ignored_test_fns(attribute_macro_forms),
        (4, 1),
        "cargo runs #[tokio::test] fns as tests, so a file whose only tests carry that attribute \
         is a real suite -- not the test-free helper a plain-#[test] scanner reads it as. The \
         rust root crate has 12 such files, 8 of them with no plain #[test] at all, and a census \
         written against the old scanner would have excluded or red-flagged every one. \
         #[rstest], #[test_case], #[proptest] and #[bench] are counted for the same reason: \
         TEST_ATTRIBUTE_SPELLINGS_CARGO_RUNS_AS_A_TEST_NOT_ONLY_PLAIN_TEST is the whole list."
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
#[allow(non_snake_case)]
fn the_cuda_half_of_nv_kernels_is_ABSENT_from_this_binary_SKIPPED_no_cuda_feature() {
    let c = corpus();
    eprintln!(
        "All {} #![cfg(feature = \"cuda\")] suites in nv-kernels/tests compiled to nothing in \
         this run, plus the {} that need cuda AND wgpu. A `0 passed` or a small pass count from \
         this binary is NOT evidence that any CUDA kernel gate held. Re-run with \
         NVK_FEATURES=cuda,wgpu.\n\
         The list is DERIVED, not written down: this sentence used to name suites in prose and \
         was wrong twice, advertising graph, flash_decode_mk and moe_grouped_fp4_* after a cull \
         deleted them.\n  cuda-only: {:?}\n  cuda+wgpu: {:?}",
        c.cuda_only.len(),
        c.cuda_and_wgpu.len(),
        c.cuda_only,
        c.cuda_and_wgpu,
    );
}

#[cfg(not(feature = "wgpu"))]
#[test]
#[allow(non_snake_case)]
fn the_wgpu_half_of_nv_kernels_is_ABSENT_from_this_binary_SKIPPED_no_wgpu_feature() {
    let c = corpus();
    eprintln!(
        "All {} #![cfg(feature = \"wgpu\")] suites in nv-kernels/tests compiled to nothing in \
         this run. A `0 passed` or a small pass count from this binary is NOT evidence that any \
         wgpu kernel gate held. Re-run with NVK_FEATURES=cuda,wgpu.\n  wgpu-only: {:?}",
        c.wgpu_only.len(),
        c.wgpu_only,
    );
}

#[cfg(not(all(feature = "cuda", feature = "wgpu")))]
#[test]
#[allow(non_snake_case)]
fn every_parity_and_scaling_suite_is_ABSENT_from_this_binary_SKIPPED_needs_cuda_and_wgpu() {
    let c = corpus();
    eprintln!(
        "The {} cfg(all(cuda, wgpu)) suites compiled to nothing in this run: {:?}. Those are the \
         cross-backend bit-exactness gates; nothing in this binary speaks for them. Re-run with \
         NVK_FEATURES=cuda,wgpu.\n\
         Do NOT reach for a `parity_*` glob instead: parity_* is no longer uniformly \
         cross-backend -- parity_verify_fused.rs is cfg(cuda) alone -- so the glob overcounts \
         the cross-backend corpus. Every scaling_* and cuda_wgpu_* suite has been deleted, and \
         naming them here defended an empty set.",
        c.cuda_and_wgpu.len(),
        c.cuda_and_wgpu,
    );
}
