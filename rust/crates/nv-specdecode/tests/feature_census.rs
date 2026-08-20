#![allow(dead_code)]

const WGPU_SUITES: &[&str] = &[
    "wgpu_spec",
    "wgpu_spec_host_ref",
    "wgpu_spec_smallm",
    "wgpu_spec_real_lockstep",
    "wgpu_spec_real_gemma4",
    "lora_spec_wgpu",
    "lora_spec_backend::wgpu_hook_through_linear_forward_no_cuda_required",
];

const CUDA_SUITES: &[&str] = &["chain_graph", "dflash_gemma4_chain_geometry"];

const BOTH_SUITES: &[&str] = &["lora_spec_backend::cuda_and_wgpu_lora_deltas_are_bit_identical"];

#[test]
fn features_compiled_into_this_test_run() {
    let cuda = cfg!(feature = "cuda");
    let wgpu = cfg!(feature = "wgpu");
    eprintln!("nv-specdecode tests compiled with cuda={cuda} wgpu={wgpu}");
    eprintln!(
        "nvk.sh defaults every NVK_PKG other than nv-kernels to `cuda` ALONE, so the natural \
         `NVK_PKG=nv-specdecode nvk.sh test` silently compiles every wgpu suite to nothing and \
         prints `0 passed` in 0.00s. Pass NVK_FEATURES=cuda,wgpu for the full surface."
    );
    assert!(
        cuda || wgpu,
        "neither `cuda` nor `wgpu` is enabled, so almost every suite in this crate compiled to an \
         empty binary. A run in this configuration is not evidence of anything."
    );
}

#[cfg(not(feature = "wgpu"))]
#[test]
#[allow(non_snake_case)]
fn the_wgpu_spec_suites_were_CFGD_OUT_of_this_binary_SKIPPED_no_wgpu_feature() {
    eprintln!(
        "SKIPPED, NOT PASSED: these suites are #![cfg(feature = \"wgpu\")] and were compiled to \
         nothing in this run: {WGPU_SUITES:?}. wgpu_spec_host_ref is the ONLY host-f64 oracle on \
         the 13 shipped sp_* kernels -- every other numeric gate in this crate compares one GPU \
         path against another, which cannot see a uniformly wrong kernel (measured: the host-ref \
         gate catches 13/13 planted kernel bugs, everything else in the crate catches 1/13). \
         Losing it to a missing feature flag is the single most expensive skip here. Separately, \
         wgpu_spec_real_lockstep's six tests are pure CPU table decoders and \
         wgpu_spec_real_gemma4's two official_template tests need no GPU at all -- the cfg header \
         alone is what hides them, because the backend-agnostic types they use live in the \
         wgpu-gated nv_specdecode::wgpu_spec module. Re-run with NVK_FEATURES=cuda,wgpu."
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
#[allow(non_snake_case)]
fn the_cuda_spec_suites_were_CFGD_OUT_of_this_binary_SKIPPED_no_cuda_feature() {
    eprintln!(
        "SKIPPED, NOT PASSED: these suites are #![cfg(feature = \"cuda\")] and were compiled to \
         nothing in this run: {CUDA_SUITES:?}. Re-run with NVK_FEATURES=cuda (or cuda,wgpu)."
    );
}

#[cfg(not(all(feature = "cuda", feature = "wgpu")))]
#[test]
#[allow(non_snake_case)]
fn the_cross_backend_lora_delta_check_was_CFGD_OUT_SKIPPED_needs_cuda_and_wgpu() {
    eprintln!(
        "SKIPPED, NOT PASSED: {BOTH_SUITES:?} needs BOTH features. lora_spec_backend uses per-test \
         #[cfg] rather than a file header, so with one feature the suite still reports a non-zero \
         pass count from its CPU test while the bit-exactness claim -- the whole point of the file \
         -- is erased."
    );
}

#[test]
fn every_suite_this_census_names_still_exists() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    let missing: Vec<&str> = WGPU_SUITES
        .iter()
        .chain(CUDA_SUITES)
        .chain(BOTH_SUITES)
        .copied()
        .filter(|s| {
            let stem = s.split("::").next().unwrap_or(s);
            !dir.join(format!("{stem}.rs")).exists()
        })
        .collect();
    assert!(
        missing.is_empty(),
        "this census names {} suite(s) that no longer exist: {missing:?}. Delete them from the \
         lists -- and lower any count they were propping up in the same commit, so the loss is \
         visible rather than absorbed.",
        missing.len()
    );
}
