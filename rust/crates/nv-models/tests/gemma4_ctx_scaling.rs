#![cfg(feature = "cuda")]

mod common;
use common::ctx_tokens_from_env_default_256_8k_168k;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Cache, Gemma4Config};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

mod ctx_timing_common;
use common::gemma4_snapshot_dir_env_override_then_home_hub as snapshot_dir;

const PREFILL_CHUNK_256_THE_SLIDING_RING_ACCEPTS_ONLY_SLACK_SIZED_WRITES_ONCE_FULL: usize = 256;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK: usize =
    ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
        + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        + 16;

#[test]
#[ignore = "loads the 31B; set NV_GEMMA4_CTX_TEST=1 -- decode ms/token as a function of KV depth (256/8k/168k), eager cuda path"]
fn gemma4_cuda_decode_ms_per_token_vs_context_depth_eager_path() {
    if std::env::var("NV_GEMMA4_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GEMMA4_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model =
        Gemma4::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().max_position_embeddings;
    assert!(
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK < max_pos,
        "requested depth {max_depth} exceeds max_position_embeddings {max_pos}"
    );

    for &depth in &depths {
        let mut cache = model
            .new_kv_cache(depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK)
            .expect("kv cache");
        let chunk = PREFILL_CHUNK_256_THE_SLIDING_RING_ACCEPTS_ONLY_SLACK_SIZED_WRITES_ONCE_FULL;
        let prefill_start = std::time::Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            let ids: Vec<u32> = (0..n).map(|i| 2000 + ((pos + i) as u32 % 30000)).collect();
            let tokens = Tensor::from_vec(ids, (1usize, n), &device).expect("tokens");
            let positions = Tensor::from_vec(
                (pos as i32..(pos + n) as i32).collect::<Vec<_>>(),
                n,
                &device,
            )
            .expect("positions");
            model
                .forward_with_cache(&tokens, &positions, &mut cache)
                .unwrap_or_else(|e| panic!("prefill chunk at pos {pos}: {e:#}"));
            pos += n;
        }
        let prefill_s = prefill_start.elapsed().as_secs_f64();

        let mut p = depth;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                let tokens = Tensor::from_vec(
                    vec![2000u32 + (p as u32 % 30000)],
                    (1usize, 1usize),
                    &device,
                )
                .expect("token");
                let positions =
                    Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
                let logits = model
                    .forward_with_cache(&tokens, &positions, &mut cache)
                    .unwrap_or_else(|e| panic!("decode at depth {p}: {e:#}"));
                logits
                    .to_dtype(candle_core::DType::F32)
                    .expect("f32")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("sync to host");
                p += 1;
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = step_ms[step_ms.len() / 2];
        let mean: f64 = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
        eprintln!(
            "CTX-SCALING gemma4-cuda-eager depth={depth} median_ms_tok={median:.3} mean_ms_tok={mean:.3} tok_s={:.1} prefill_s={prefill_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the 31B; set NV_GEMMA4_CTX_TEST=1 -- same depth ladder but the KV is filled synthetically (decode speed does not read cache values), so 168k costs seconds of setup, not an hour of eager prefill"]
fn gemma4_cuda_decode_ms_per_token_vs_context_depth_synthetic_cache_fill() {
    if std::env::var("NV_GEMMA4_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GEMMA4_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Gemma4Config::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Gemma4::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let n_layers = model.config().num_hidden_layers;
    for &depth in &depths {
        let mut cache = model
            .new_kv_cache(depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK)
            .expect("kv cache");
        let chunk = PREFILL_CHUNK_256_THE_SLIDING_RING_ACCEPTS_ONLY_SLACK_SIZED_WRITES_ONCE_FULL;
        let fill_start = std::time::Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            cache
                .prepare_for_decode(pos, pos + n)
                .expect("prepare_for_decode");
            for li in 0..n_layers {
                let kind = model.config().layer_kind(li);
                let hd = model.config().head_dim_for(kind);
                let n_kv = model.config().num_kv_heads_for(kind);
                let k = Tensor::zeros((1usize, n, n_kv, hd), DType::BF16, &device).expect("k");
                let v = Tensor::zeros((1usize, n, n_kv, hd), DType::BF16, &device).expect("v");
                cache.write_at(li, &k, &v).expect("write_at");
            }
            cache.advance(n);
            pos += n;
        }
        let fill_s = fill_start.elapsed().as_secs_f64();

        let mut p = depth;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                let tokens = Tensor::from_vec(
                    vec![2000u32 + (p as u32 % 30000)],
                    (1usize, 1usize),
                    &device,
                )
                .expect("token");
                let positions =
                    Tensor::from_vec(vec![p as i32], 1usize, &device).expect("position");
                let logits = model
                    .forward_with_cache(&tokens, &positions, &mut cache)
                    .unwrap_or_else(|e| panic!("decode at depth {p}: {e:#}"));
                logits
                    .to_dtype(DType::F32)
                    .expect("f32")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("sync to host");
                p += 1;
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = step_ms[step_ms.len() / 2];
        eprintln!(
            "CTX-SCALING gemma4-cuda-eager-synthfill depth={depth} median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}
