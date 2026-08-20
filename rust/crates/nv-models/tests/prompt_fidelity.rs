
mod prompt_scan;

const MIN_WALKED: usize = 40;

const ALLOWED: [(&str, &str); 1] = [(
    "fp8_contract_freerun.rs",
    "the_affix_deriver_recovers_the_gemma4_turn_markers_from_renders_alone feeds a synthetic \
     render to the affix deriver and asserts it recovers <|turn>/<turn|> from it. The literal IS \
     the fixture under test -- it even carries the 26B/31B <|channel>thought opener -- and \
     templating it would delete what the test proves. No model output is scored on it.",
)];

#[test]
fn no_real_weights_harness_hand_builds_a_gemma4_prompt() {
    prompt_scan::CrateScan {
        crate_name: "nv-models",
        manifest_dir: env!("CARGO_MANIFEST_DIR"),
        self_file: file!(),
        min_walked: MIN_WALKED,
        allowed: &ALLOWED,
    }
    .run();
}
