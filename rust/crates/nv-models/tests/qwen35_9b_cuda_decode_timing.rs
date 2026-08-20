#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::percentile_of_sorted;
use candle_core::{Device, Tensor};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;
use tokenizers::Tokenizer;
use common::ig1_qwen35_9b_nvfp4_snapshot_dir_env_override_then_home_hub;

const TIMED_DECODE_STEPS_96_SO_THE_MEDIAN_COVERS_WELL_OVER_64: usize = 96;
const MIN_TIMED_STEPS_64_BELOW_WHICH_A_MEDIAN_IS_NOT_QUOTABLE: usize = 64;
const WARMUP_DECODE_STEPS_EXCLUDED_FROM_TIMING: usize = 4;

fn last_row_f32(logits: &Tensor, seq: usize) -> Vec<f32> {
    let flat: Vec<f32> = logits
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("host");
    assert_eq!(flat.len() % seq, 0, "logit rows not divisible by seq len");
    let vocab = flat.len() / seq;
    flat[(seq - 1) * vocab..].to_vec()
}

#[test]
#[ignore = "loads the 9.6 GiB ig1 Qwen3.5-9B NVFP4 checkpoint; set NV_QWEN35_TIMING=1; per-step model timing, unlike the suspect serving figure that was a warm-request wall clock over the whole serving stack"]
fn qwen35_9b_nvfp4_cuda_eager_dense_decode_step_median_including_logits_to_host_same_model_build_as_the_opt_in_cuda_serving_arm(
) {
    if std::env::var("NV_QWEN35_TIMING").as_deref() != Ok("1") {
        panic!("set NV_QWEN35_TIMING=1 to run (it must never silently skip)");
    }
    let dir = ig1_qwen35_9b_nvfp4_snapshot_dir_env_override_then_home_hub();
    assert!(
        dir.join("model.safetensors").is_file()
            || dir.join("model.safetensors.index.json").is_file(),
        "checkpoint {dir:?} has no weights"
    );
    let tok = Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let q = std::env::var("NV_QWEN35_Q")
        .unwrap_or_else(|_| "What is the capital of France? Answer in one short sentence.".into());
    let prompt_text =
        format!("<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    let prompt: Vec<u32> = tok
        .encode(prompt_text.as_str(), false)
        .expect("encode")
        .get_ids()
        .to_vec();

    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let t_load = Instant::now();
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("model");
    let load_s = t_load.elapsed().as_secs_f64();
    assert_eq!(
        model.dense_intermediate(),
        Some(cfg.intermediate_size),
        "from_loader_dense_quantized did not carry intermediate_size, so this is not the dense arm \
         the opt-in NV_QWEN35_DENSE_CUDA_SERVE serving path builds"
    );

    let total_steps = WARMUP_DECODE_STEPS_EXCLUDED_FROM_TIMING
        + TIMED_DECODE_STEPS_96_SO_THE_MEDIAN_COVERS_WELL_OVER_64;
    let k = prompt.len();
    let mut cache = model
        .new_kv_cache(k + total_steps + 8)
        .expect("kv cache");
    let t0 = Instant::now();
    let tokens = Tensor::from_vec(prompt.clone(), (1usize, k), &device).expect("tokens");
    let positions =
        Tensor::from_vec((0..k as i32).collect::<Vec<_>>(), k, &device).expect("positions");
    let logits = model
        .forward_with_cache(&tokens, &positions, &mut cache)
        .expect("prefill forward");
    let mut cur = argmax(&last_row_f32(&logits, k));
    let prefill_ms = t0.elapsed().as_secs_f64() * 1e3;

    let mut times_ms: Vec<f64> = Vec::new();
    for step in 0..total_steps {
        let p = k + step;
        let t = Instant::now();
        let tokens = Tensor::from_vec(vec![cur], (1usize, 1usize), &device).expect("token");
        let positions = Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
        let logits = model
            .forward_with_cache(&tokens, &positions, &mut cache)
            .unwrap_or_else(|e| panic!("decode step {step}: {e:#}"));
        let row = last_row_f32(&logits, 1);
        let dt_ms = t.elapsed().as_secs_f64() * 1e3;
        if step >= WARMUP_DECODE_STEPS_EXCLUDED_FROM_TIMING {
            times_ms.push(dt_ms);
        }
        cur = argmax(&row);
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
        "TIMING qwen35-9b-nvfp4-cuda basis=eager_dense_forward_with_cache_argmax_feed_includes_logits_to_host_no_graph_capture_same_build_as_NV_QWEN35_DENSE_CUDA_SERVE load={load_s:.1}s prompt_toks={} prefill={prefill_ms:.1}ms timed_steps={} warmup_excluded={} median={med:.3}ms/tok p10={p10:.3} p90={p90:.3} tok_per_s={:.1}",
        prompt.len(),
        times_ms.len(),
        WARMUP_DECODE_STEPS_EXCLUDED_FROM_TIMING,
        1000.0 / med
    );
}
