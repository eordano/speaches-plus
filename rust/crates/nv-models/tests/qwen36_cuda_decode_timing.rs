#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::Arm;
use common::build;
use common::percentile_of_sorted;
use common::snapshot;
use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;

const TIMED_FREE_RUNNING_DECODE_STEPS_96_SO_THE_MEDIAN_COVERS_WELL_OVER_64: usize = 96;
const MIN_TIMED_STEPS_64_BELOW_WHICH_A_MEDIAN_IS_NOT_QUOTABLE: usize = 64;
const WARMUP_DECODE_STEPS_EXCLUDED_FROM_TIMING: usize = 4;

const ARMS: [Arm; 3] = [
    Arm {
        label: "grouped+routed",
        grouped: true,
        routing: true,
    },
    Arm {
        label: "grouped+unrouted",
        grouped: true,
        routing: false,
    },
    Arm {
        label: "plain",
        grouped: false,
        routing: false,
    },
];

#[test]
#[ignore = "loads the ~22 GB Qwen3.6 checkpoint; set NV_QWEN36_TIMING=1; timing only, the #95 free-running degeneracy (distinct ratio 0.062) is expected and deliberately not asserted"]
fn qwen36_nvfp4_cuda_graphed_free_running_decode_times_decode_without_judging_output_quality_see_task_95(
) {
    if std::env::var("NV_QWEN36_TIMING").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_TIMING=1 to run (it must never silently skip)");
    }
    let dir = snapshot();
    assert!(
        dir.join("model.safetensors").is_file()
            || dir.join("model.safetensors.index.json").is_file(),
        "checkpoint {dir:?} has no weights"
    );
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let q = std::env::var("NV_QWEN36_Q")
        .unwrap_or_else(|_| "What is the capital of France? Answer in one short sentence.".into());
    let prompt_text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let prompt: Vec<u32> = tok
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let want = std::env::var("NV_QWEN36_ARM").unwrap_or_else(|_| "grouped+routed".into());
    let arm = ARMS
        .iter()
        .find(|a| a.label == want)
        .unwrap_or_else(|| panic!("unknown NV_QWEN36_ARM={want}"));

    let t_load = Instant::now();
    let mut eng = build(&dir, &device, arm);
    let load_s = t_load.elapsed().as_secs_f64();

    let t0 = Instant::now();
    let row = eng.prefill(&prompt).expect("prefill");
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;
    assert_eq!(row.len(), eng.vocab_size(), "logit row width");

    let mut cur = argmax(&row);
    let mut times_ms: Vec<f64> = Vec::new();
    let total_steps = WARMUP_DECODE_STEPS_EXCLUDED_FROM_TIMING
        + TIMED_FREE_RUNNING_DECODE_STEPS_96_SO_THE_MEDIAN_COVERS_WELL_OVER_64;
    for step in 0..total_steps {
        let t = Instant::now();
        eng.forward_decode(cur)
            .unwrap_or_else(|e| panic!("decode step {step}: {e:#}"));
        let r = eng.logits_host().expect("logits");
        let dt_ms = t.elapsed().as_secs_f64() * 1e3;
        if step >= WARMUP_DECODE_STEPS_EXCLUDED_FROM_TIMING {
            times_ms.push(dt_ms);
        }
        cur = argmax(&r);
    }

    assert!(
        times_ms.len() >= MIN_TIMED_STEPS_64_BELOW_WHICH_A_MEDIAN_IS_NOT_QUOTABLE,
        "only {} timed steps",
        times_ms.len()
    );
    let mut sorted = times_ms.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = percentile_of_sorted(&sorted, 0.5);
    let p10 = percentile_of_sorted(&sorted, 0.1);
    let p90 = percentile_of_sorted(&sorted, 0.9);
    assert!(
        med.is_finite() && med > 0.0,
        "median decode step time is not a positive number: {med}"
    );
    eprintln!(
        "TIMING qwen36-nvfp4-cuda basis=graphed_free_running_decode_argmax_feed_includes_logits_to_host arm={} capture_active={} load={load_s:.1}s prompt_toks={} prefill={prefill_ms:.1}ms timed_steps={} warmup_excluded={} median={med:.3}ms/tok p10={p10:.3} p90={p90:.3} tok_per_s={:.1} output_quality=NOT_JUDGED_see_task_95",
        arm.label,
        eng.capture_active(),
        prompt.len(),
        times_ms.len(),
        WARMUP_DECODE_STEPS_EXCLUDED_FROM_TIMING,
        1000.0 / med
    );
}
