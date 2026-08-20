use nv_models::gemma4::PREFILL_W4A4_MIN_M;
use std::path::PathBuf;

const SCHEDULER: &str = "crates/nv-engine/src/scheduler.rs";

const SCHEDULER_FLOOR_CONST: &str =
    "PREFILL_CHUNK_ROWS_BELOW_WHICH_THE_MODEL_SWITCHES_PRECISION_256: usize = ";

fn rust_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.pop();
    p.pop();
    p
}

#[test]
fn the_scheduler_floor_is_the_row_count_at_which_prefill_switches_precision() {
    let path = rust_root().join(SCHEDULER);
    let src = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "census cannot read {}: {e}. A missing source is a finding, not a pass",
            path.display()
        )
    });
    let at = src.find(SCHEDULER_FLOOR_CONST).unwrap_or_else(|| {
        panic!(
            "{SCHEDULER} no longer declares {SCHEDULER_FLOOR_CONST}..., so this census is \
             checking nothing. nv-engine does not depend on nv-models, so a textual census is \
             the only thing holding the scheduler's floor to the model's switch point"
        )
    });
    let tail = &src[at + SCHEDULER_FLOOR_CONST.len()..];
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    let floor: usize = digits.parse().unwrap_or_else(|e| {
        panic!("cannot parse the scheduler floor from {digits:?}: {e}")
    });
    assert_eq!(
        floor, PREFILL_W4A4_MIN_M,
        "the scheduler floors prefill chunks at {floor} rows but the model switches its QKV \
         projection to W4A4 at m >= {PREFILL_W4A4_MIN_M} (prefill_w4a4_selects, on unless \
         NV_PREFILL_W4A4=0). Those must be the same number: the floor exists so that every \
         chunk of a prompt lands on the same side of that switch whatever else is in flight. \
         Measured on Gemma4-31B-NVFP4 at 2048 prompt tokens, chunks 1792,256 agree to 0e0 \
         while 1793,255 differ by 1.19e-1 on layer 0 K. Note the switch is not the only \
         m-dependent path -- with NV_PREFILL_W4A4=0 chunk 128 still differs from chunk 1024 \
         by 1.02e-1, and fused_qkv_bitwise_safe takes its own branch for m in 2..=16 -- so \
         moving this constant needs a fresh sweep, not arithmetic"
    );
}
