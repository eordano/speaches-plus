
#[path = "../../nv-models/tests/prompt_scan/mod.rs"]
mod prompt_scan;

const MIN_WALKED: usize = 40;

#[test]
fn no_real_weights_harness_hand_builds_a_gemma4_prompt() {
    prompt_scan::CrateScan {
        crate_name: "nv-kernels",
        manifest_dir: env!("CARGO_MANIFEST_DIR"),
        self_file: file!(),
        min_walked: MIN_WALKED,
        allowed: &[],
    }
    .run();
}
