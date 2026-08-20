#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::ctx_tokens_from_env_default_256_8k_196k;
use common::percentile_of_sorted;
use candle_core::{DType, Device, Tensor};
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;

mod ctx_timing_common;
use common::decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state;
use common::qwen36_nvfp4_snapshot_dir_env_override_then_home_hub;

const PREFILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN36_CTX_SCALING_PPL_BLOCK: usize = 512;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS: usize = 16;

fn dump_every_timed_step_when_env_asks_because_medians_hide_graph_recapture_spikes(
    depth: usize,
    step_ms: &[f64],
) {
    if std::env::var("NV_Q36_STEP_DUMP").ok().as_deref() != Some("1") {
        return;
    }
    for (i, ms) in step_ms.iter().enumerate() {
        eprintln!("STEP-DUMP qwen36-cuda-serving depth={depth} step={i} ms={ms:.3}");
    }
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4; set NV_QWEN36_SERVING_TEST=1 -- serving-class decode ms/token vs KV depth: chunked prefill primes the cache with the PLAIN dispatch and grouped MoE is installed only afterwards, because eager grouped prefill returns all-NaN logits (see graph_engine::THE_EAGER_ALL_NAN_IS_INSTALL_GROUPED_MOE_NOT_THIS_DISABLE_AND_NEEDS_A_HOST_STALL); then 64 graphed free-running decode steps are timed; the #95 degeneracy is deliberately not judged; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn qwen36_cuda_serving_path_decode_ms_per_token_vs_context_depth_graphed_grouped_moe_chunk_primed()
{
    if std::env::var("NV_QWEN36_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_QWEN36_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; a 196k serving gate that silently reports ok would hide \
                 exactly the failure class task #106 exists to catch"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN36_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = qwen36_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().max_position_embeddings;
    let per_depth_extra_slots =
        ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
        + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        + KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS;
    assert!(
        max_depth + per_depth_extra_slots < max_pos,
        "depth {max_depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );
    let cache_slots = max_depth + per_depth_extra_slots;
    let mut eng = GraphedQwen3Moe::new(model, &device, cache_slots)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {cache_slots} kv slots: {e:#}"));

    for &depth in &depths {
        eng.set_moe_dispatch(None);
        eng.reset().expect("engine reset");
        assert_eq!(eng.current_pos(), 0, "reset must rewind the engine");

        let prime_start = Instant::now();
        let mut pos = 0usize;
        let mut last_row: Vec<f32> = Vec::new();
        while pos < depth {
            let n = PREFILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN36_CTX_SCALING_PPL_BLOCK
                .min(depth - pos);
            let ids: Vec<u32> = (0..n).map(|i| 2000 + ((pos + i) as u32 % 30000)).collect();
            last_row = eng
                .prefill(&ids)
                .unwrap_or_else(|e| panic!("prefill chunk at pos {pos}: {e:#}"));
            pos += n;
        }
        let prime_s = prime_start.elapsed().as_secs_f64();
        assert_eq!(
            eng.current_pos(),
            depth,
            "chunked prefill must leave the engine at the primed depth"
        );
        let nan_in_last_row = last_row.iter().filter(|v| v.is_nan()).count();
        assert_eq!(
            nan_in_last_row, 0,
            "plain-dispatch chunked prefill produced {nan_in_last_row} NaN logits at depth \
             {depth}; the all-NaN failure belongs to grouped eager prefill, which this test \
             avoids by installing grouped MoE only after priming"
        );

        eng.install_grouped_moe().expect("grouped moe install");

        let mut cur = argmax(&last_row);
        let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                eng.forward_decode(cur)
                    .unwrap_or_else(|e| panic!("serving decode at depth {depth}: {e:#}"));
                let r = eng.logits_host().expect("logits to host");
                cur = argmax(&r);
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        assert!(
            eng.capture_active(),
            "the serving basis of this gate is the CAPTURED graphed decode and the \
             engine fell back to uncaptured at depth {depth}; the engine printed its \
             blocker to stderr just above -- diagnose that blocker, do not relabel the \
             number as serving-class"
        );

        let mut sorted = step_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = percentile_of_sorted(&sorted, 0.5);
        let p10 = percentile_of_sorted(&sorted, 0.1);
        let p90 = percentile_of_sorted(&sorted, 0.9);
        assert!(
            median.is_finite() && median > 0.0,
            "median decode step time is not a positive number: {median}"
        );
        dump_every_timed_step_when_env_asks_because_medians_hide_graph_recapture_spikes(
            depth, &step_ms,
        );
        eprintln!(
            "CTX-SCALING qwen36-cuda-serving depth={depth} basis=graphed_grouped_moe_free_running_decode_argmax_feed_includes_logits_host decode_kernel={} capture_active={} median_ms_tok={median:.3} p10={p10:.3} p90={p90:.3} tok_s={:.1} prime_s={prime_s:.1} prefill_chunk={} steps={} warmup_steps={warmup_steps} output_quality=NOT_JUDGED_see_task_95",
            decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state(),
            eng.capture_active(),
            1000.0 / median,
            PREFILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN36_CTX_SCALING_PPL_BLOCK,
            step_ms.len()
        );
    }
}

const SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_REAL_PRIME_CHUNK_ANY_SIZE_FITS_THE_FP8_APPEND: usize =
    512;

fn fixed_seed_position_hashed_values_because_moe_decode_cost_is_weakly_routing_dependent_and_an_all_zero_cache_would_collapse_routing_toward_a_single_expert(
    len: usize,
) -> Vec<f32> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((state >> 40) as f32) / (1u64 << 24) as f32 - 0.5) * 0.5
        })
        .collect()
}

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4; set NV_QWEN36_SERVING_TEST=1 -- same graphed serving decode ladder but the fp8 KV is filled synthetically and current_pos advanced the way prefill would (decode ms/token reads cache SIZE; routing is only weakly value-dependent, so the fill is fixed-seed pseudo-random rather than zeros); the capture_active assertion is kept; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn qwen36_cuda_serving_path_decode_ms_per_token_vs_context_depth_synthetic_cache_fill_graphed_grouped_moe(
) {
    if std::env::var("NV_QWEN36_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_QWEN36_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; a 196k serving gate that silently reports ok would hide \
                 exactly the failure class task #106 exists to catch"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN36_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = qwen36_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().max_position_embeddings;
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let per_depth_extra_slots =
        ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
        + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        + KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS;
    assert!(
        max_depth + per_depth_extra_slots < max_pos,
        "depth {max_depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );
    let cache_slots = max_depth + per_depth_extra_slots;
    let mut eng = GraphedQwen3Moe::new(model, &device, cache_slots)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {cache_slots} kv slots: {e:#}"));

    let chunk = SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_REAL_PRIME_CHUNK_ANY_SIZE_FITS_THE_FP8_APPEND;
    let vals = fixed_seed_position_hashed_values_because_moe_decode_cost_is_weakly_routing_dependent_and_an_all_zero_cache_would_collapse_routing_toward_a_single_expert(chunk * n_kv * hd);
    let k_template = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("k bf16");
    let v_template = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("v bf16");

    for &depth in &depths {
        eng.set_moe_dispatch(None);
        eng.reset().expect("engine reset");
        assert_eq!(eng.current_pos(), 0, "reset must rewind the engine");

        let fill_start = Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            if n == chunk {
                eng.prime_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
                    &k_template,
                    &v_template,
                )
                .unwrap_or_else(|e| panic!("synthetic fill chunk at pos {pos}: {e:#}"));
            } else {
                let kn = k_template.narrow(1, 0, n).expect("k tail");
                let vn = v_template.narrow(1, 0, n).expect("v tail");
                eng.prime_kv_depth_synthetically_for_ctx_timing_decode_reads_cache_size_not_values(
                    &kn, &vn,
                )
                .unwrap_or_else(|e| panic!("synthetic fill tail at pos {pos}: {e:#}"));
            }
            pos += n;
        }
        eng.synchronize().expect("sync before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            eng.current_pos(),
            depth,
            "synthetic fill must advance current_pos exactly the way prefill would"
        );

        eng.install_grouped_moe().expect("grouped moe install");

        let mut cur = 2000u32;
        let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                eng.forward_decode(cur)
                    .unwrap_or_else(|e| panic!("serving decode at depth {depth}: {e:#}"));
                let r = eng.logits_host().expect("logits to host");
                cur = argmax(&r);
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        assert!(
            eng.capture_active(),
            "the serving basis of this gate is the CAPTURED graphed decode and the \
             engine fell back to uncaptured at depth {depth}; the engine printed its \
             blocker to stderr just above -- diagnose that blocker, do not relabel the \
             number as serving-class"
        );

        let mut sorted = step_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = percentile_of_sorted(&sorted, 0.5);
        let p10 = percentile_of_sorted(&sorted, 0.1);
        let p90 = percentile_of_sorted(&sorted, 0.9);
        assert!(
            median.is_finite() && median > 0.0,
            "median decode step time is not a positive number: {median}"
        );
        dump_every_timed_step_when_env_asks_because_medians_hide_graph_recapture_spikes(
            depth, &step_ms,
        );
        eprintln!(
            "CTX-SCALING qwen36-cuda-serving-synthfill depth={depth} basis=graphed_grouped_moe_free_running_decode_argmax_feed_includes_logits_host_synthetic_kv_fill decode_kernel={} capture_active={} median_ms_tok={median:.3} p10={p10:.3} p90={p90:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps} output_quality=NOT_JUDGED_see_task_95",
            decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state(),
            eng.capture_active(),
            1000.0 / median,
            step_ms.len()
        );
    }
}
