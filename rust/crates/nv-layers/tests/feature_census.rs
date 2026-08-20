#![allow(dead_code)]

const WGPU_SUITES: &[&str] = &[
    "wgpu_sampling_parity",
    "wgpu_correct_moe_row_padding",
    "wgpu_tensor",
    "moe_wgpu",
    "lora_wgpu_runtime",
];

const CUDA_SUITES: &[&str] = &[
    "flash_attn",
    "mlp",
    "fp8_shape_probe",
    "gdn_fused_decode",
    "nvfp4_staging_lifecycle",
    "moe",
    "linear_quant",
    "linear",
    "rmsnorm",
    "linear_offset_views",
    "rope",
];

const BOTH_SUITES: &[&str] = &["wgpu_correct_moe_cuda_shape_sweep"];

const ALWAYS_SUITES: &[&str] = &[
    "backend_select",
    "paper_sampling",
    "rope_table_precision",
    "rope_table_env_knob",
    "lora_slots",
];

const CULLED_SUITES: &[&str] = &[
    "block",
    "fa2_hdim512_determinism",
    "flash_determinism",
    "layer_mixer",
    "linear_attn",
    "lora_linear",
    "moe_host_vs_grouped_numerics",
    "pretranspose_prefill_cost",
    "wgpu_sampling_cost",
];

#[test]
fn features_compiled_into_this_test_run() {
    let cuda = cfg!(feature = "cuda");
    let wgpu = cfg!(feature = "wgpu");
    eprintln!("nv-layers tests compiled with cuda={cuda} wgpu={wgpu}");
    eprintln!(
        "nvk.sh defaults NVK_PKG=nv-layers to `cuda` ALONE, so the whole wgpu surface -- \
         including lora_wgpu_runtime's own falsification test \
         (falsification_single_weight_perturbation_breaks_the_gate, the thing that proves the \
         other nine can fail) -- compiles to nothing and prints `0 passed` in 0.00s. Pass \
         NVK_FEATURES=cuda,wgpu."
    );
    assert!(
        cuda || wgpu,
        "neither `cuda` nor `wgpu` is enabled, so 24 of this crate's suites compiled to empty \
         binaries. A run in this configuration is not evidence of anything."
    );
}

#[cfg(not(feature = "wgpu"))]
#[test]
#[allow(non_snake_case)]
fn the_wgpu_suites_were_CFGD_OUT_of_this_binary_SKIPPED_no_wgpu_feature() {
    eprintln!(
        "SKIPPED, NOT PASSED: these suites are #![cfg(feature = \"wgpu\")] and compiled to \
         nothing in this run: {WGPU_SUITES:?}. Re-run with NVK_FEATURES=cuda,wgpu."
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
#[allow(non_snake_case)]
fn the_cuda_suites_were_CFGD_OUT_of_this_binary_SKIPPED_no_cuda_feature() {
    eprintln!(
        "SKIPPED, NOT PASSED: these suites are #![cfg(feature = \"cuda\")] and compiled to \
         nothing in this run: {CUDA_SUITES:?}. `NVK_FEATURES=wgpu` is the sanctioned fast-edit \
         loop, so this is the configuration in which the entire nv-layers CUDA surface silently \
         becomes `0 passed`."
    );
}

#[cfg(not(all(feature = "cuda", feature = "wgpu")))]
#[test]
#[allow(non_snake_case)]
fn the_cross_backend_moe_sweep_was_CFGD_OUT_SKIPPED_needs_cuda_and_wgpu() {
    eprintln!(
        "SKIPPED, NOT PASSED: {BOTH_SUITES:?} is #![cfg(all(cuda, wgpu))] -- the textbook parity_* \
         trap. With one feature it compiles to nothing and prints `0 passed` in 0.00s. Both \
         devices exist on this box."
    );
}

#[test]
fn every_suite_this_census_names_still_exists() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let missing: Vec<&str> = WGPU_SUITES
        .iter()
        .chain(CUDA_SUITES)
        .chain(BOTH_SUITES)
        .chain(ALWAYS_SUITES)
        .copied()
        .filter(|s| {
            !dir.join(format!("{}.rs", s.split("::").next().unwrap_or(s)))
                .exists()
        })
        .collect();
    assert!(
        missing.is_empty(),
        "this census names {} suite(s) that no longer exist: {missing:?}. Remove them here in the \
         same commit that removes the files, and record them in CULLED_SUITES, so the coverage \
         loss is visible rather than absorbed.",
        missing.len()
    );
}

#[test]
fn the_culled_suites_are_still_gone_and_their_coverage_is_still_missing() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let resurrected: Vec<&str> = CULLED_SUITES
        .iter()
        .copied()
        .filter(|s| dir.join(format!("{s}.rs")).exists())
        .collect();
    assert!(
        resurrected.is_empty(),
        "CULLED_SUITES names {resurrected:?}, which exist again. Move them into the matching \
         feature list in the same commit that restored them."
    );
    eprintln!(
        "UNGATED SINCE e6d7905b9, NOT COVERED: {CULLED_SUITES:?}. This census reports what is \
         missing, not only what is present."
    );
}

#[test]
fn some_suite_in_this_crate_runs_without_any_feature() {
    assert!(
        !ALWAYS_SUITES.is_empty(),
        "every suite in this crate is feature-gated, so a build with no features reports `ok` \
         while running nothing"
    );
    eprintln!("compiled in every configuration: {ALWAYS_SUITES:?}");
}
