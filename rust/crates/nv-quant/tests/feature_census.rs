#![allow(dead_code)]

const CUDA_SUITES: &[&str] = &[
    "fp8",
    "matmul",
    "matmul_bt",
    "matmul_bt_adversarial",
    "nvfp4",
    "nvfp4_true_m",
    "nvfp4_default_routing_split",
    "graph_capture_probe",
];

const ALWAYS_SUITES: &[&str] = &["mxfp4", "ue4m3_fuzz"];

const CULLED_SUITES: &[&str] = &[
    "matmul_bt_determinism",
    "nvfp4_tile_bench",
    "nvfp4_lt_algo_bench",
    "bf16_splitk_bench",
    "quantization_config",
];

fn suite_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn missing(names: &[&str]) -> Vec<String> {
    names
        .iter()
        .filter(|s| !suite_dir().join(format!("{s}.rs")).exists())
        .map(|s| (*s).to_string())
        .collect()
}

#[test]
fn features_compiled_into_this_test_run() {
    let cuda = cfg!(feature = "cuda");
    eprintln!("nv-quant tests compiled with cuda={cuda} (this crate has no wgpu feature)");
    eprintln!(
        "Every GEMM/quantization suite in this crate is #![cfg(feature = \"cuda\")]. Only \
         {ALWAYS_SUITES:?} survive a build without it."
    );
    assert!(
        !ALWAYS_SUITES.is_empty(),
        "no suite in this crate is compiled unconditionally, so a run without `cuda` executes \
         nothing and still exits 0"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
#[allow(non_snake_case)]
fn the_cuda_gemm_suites_were_CFGD_OUT_of_this_binary_SKIPPED_no_cuda_feature() {
    eprintln!(
        "SKIPPED, NOT PASSED: these suites compiled to nothing in this run: {CUDA_SUITES:?}. \
         `NVK_FEATURES=wgpu` is the documented fast-edit loop, and in that configuration every \
         one of them reports `0 passed` in 0.00s -- a skip that reads as a pass. Re-run with \
         NVK_FEATURES=cuda for any fp8/nvfp4 claim."
    );
}

#[test]
fn every_suite_this_census_names_still_exists() {
    let gone = missing(&[CUDA_SUITES, ALWAYS_SUITES].concat());
    assert!(
        gone.is_empty(),
        "this census names {} suite(s) that no longer exist: {gone:?}. Remove them here in the \
         same commit that removes the files, and record them in CULLED_SUITES, so the coverage \
         loss is visible rather than absorbed.",
        gone.len()
    );
}

#[test]
fn the_culled_suites_are_still_gone_and_their_coverage_is_still_missing() {
    let resurrected: Vec<&str> = CULLED_SUITES
        .iter()
        .copied()
        .filter(|s| suite_dir().join(format!("{s}.rs")).exists())
        .collect();
    assert!(
        resurrected.is_empty(),
        "CULLED_SUITES names {resurrected:?}, which exist again. Move them into CUDA_SUITES or \
         ALWAYS_SUITES in the same commit that restored them."
    );
    eprintln!(
        "UNGATED SINCE e6d7905b9, NOT COVERED: {CULLED_SUITES:?}. This census reports what is \
         missing, not only what is present."
    );
}
