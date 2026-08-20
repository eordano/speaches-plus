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

#[derive(Default)]
struct Corpus {
    cuda_only: Vec<String>,
    cuda_and_wgpu: Vec<String>,
    wgpu_only: Vec<String>,
    unconditional: Vec<String>,
    cfg_header_hiding_below_the_first_line: Vec<String>,
}

fn corpus() -> Corpus {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests");
    let rd = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("cannot read the suite directory {}: {e}", dir.display()));
    let mut c = Corpus::default();
    for e in rd.flatten() {
        let p = e.path();
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
        for (i, line) in src.lines().enumerate().skip(1) {
            if line.trim_start().starts_with("#![cfg(") {
                c.cfg_header_hiding_below_the_first_line.push(format!(
                    "{name}:{}: {}",
                    i + 1,
                    line.trim()
                ));
            }
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
    "fp8_mk_ratio_oracle.rs",
    "gather_rows_bf16.rs",
    "gdn_chunked_parity.rs",
    "gdn_qkvz_conv_fuse_parity.rs",
    "gdn_split_decode_parity.rs",
    "gemv_bf16_normed_dynk_parity.rs",
    "gemv_e4m3_mk_m_sweep_bench.rs",
    "gemv_e4m3_mk_parity.rs",
    "gemv_e4m3_qkv_one_parity.rs",
    "gemv_nvfp4_w4a16_decode_parity.rs",
    "gemv_nvfp4_w4a8_decode_parity.rs",
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
    "moe_gemv.rs",
    "moe_grouped_fp4_gemv_bench.rs",
    "moe_grouped_fp4_gemv_parity.rs",
    "moe_grouped_fp4_multi_expert.rs",
    "moe_permute.rs",
    "moe_unpermute_scatter.rs",
    "parity_verify_fused.rs",
    "pdl_gemv_bf16_ab.rs",
    "q38_decode_bucket_floor_bench.rs",
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
    "wgpu_subgroup_portability.rs",
    "wgpu_verify_fused_norms.rs",
    "wgpu_wmma_accum_model.rs",
];

const CENSUS_UNCONDITIONAL_THESE_SUITES_READ_SOURCE_TEXT_NOT_A_DEVICE: &[&str] = &[
    "cuda_ue4m3_subnormal_codec.rs",
    "feature_census.rs",
    "fp8_rowscale_oracle_fuzz.rs",
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
