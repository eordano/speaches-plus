#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4Cache, LayerType};
use nv_models::laguna::{Laguna, LagunaConfig};
use nv_models::laguna_fp8::LagunaKvCacheFp8;
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

mod ctx_timing_common;

const PREFILL_CHUNK_256_LAGUNA_KV_RING_SLIDING_CAP_IS_WINDOW_PLUS_COMPACT_SLACK_256_AND_AFTER_A_SHIFT_KEEPS_WINDOW_ROWS_SO_A_STEADY_STATE_APPEND_ABOVE_256_TRIPS_THE_SLIDING_CAPACITY_BAIL:
    usize = 256;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK: usize =
    ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
        + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        + 16;
const DEPTH_196K_IS_200704_TOKENS_THE_CONFORMANCE_RUNG_EVERY_262144_MAX_POS_MODEL_MUST_PASS:
    usize = 196 * 1024;

fn laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_LAGUNA_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--poolside--Laguna-XS-2.1-NVFP4/snapshots");
    let mut candidates: Vec<PathBuf> = std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("laguna snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.join("config.json").is_file()
                && (p.join("model.safetensors").is_file()
                    || p.join("model.safetensors.index.json").is_file())
        })
        .collect();
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .expect("no Laguna-XS-2.1 NVFP4 snapshot with weights under HOME hub; set NV_LAGUNA_DIR")
}

fn ctx_tokens_from_env_default_256_8k_196k() -> Vec<usize> {
    match std::env::var("NV_CTX_TOKENS") {
        Ok(v) => v
            .split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix('k') {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect(),
        Err(_) => vec![
            256,
            8 * 1024,
            DEPTH_196K_IS_200704_TOKENS_THE_CONFORMANCE_RUNG_EVERY_262144_MAX_POS_MODEL_MUST_PASS,
        ],
    }
}

fn assert_fp8_decode_routes_to_gscores_scratch_at_the_196k_rung(
    cfg: &LagunaConfig,
    device: &Device,
) {
    let rung =
        DEPTH_196K_IS_200704_TOKENS_THE_CONFORMANCE_RUNG_EVERY_262144_MAX_POS_MODEL_MUST_PASS;
    let smem_cap = LagunaKvCacheFp8::max_seq_len_for_fp8_decode(cfg.head_dim);
    assert!(
        rung > smem_cap,
        "196k rung {rung} unexpectedly fits the fp8 decode smem path (cap {smem_cap} at head_dim {}), the gscores probe below would prove nothing",
        cfg.head_dim
    );
    let mut probe_cfg = cfg.clone();
    probe_cfg.layer_types = vec![LayerType::FullAttention, LayerType::SlidingAttention];
    probe_cfg.num_hidden_layers = probe_cfg.layer_types.len();
    let probe = LagunaKvCacheFp8::new(&probe_cfg, rung, device, DType::BF16)
        .expect("fp8 kv cache probe at 200704");
    assert!(
        probe.uses_score_scratch(),
        "fp8 kv cache at max_seq_len {rung} > smem cap {smem_cap} must allocate the gscores scratch so decode routes to attention_fp8_decode_gscores instead of the 48KB-smem kernel"
    );
    assert_eq!(probe.max_seq_len(), rung, "fp8 probe cache max_seq_len");
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4 checkpoint; set NV_LAGUNA_CTX_TEST=1 -- decode ms/token vs KV depth (max_pos 262144 fits the full 256/8k/196k ladder), chunked-prefill primed at 256 because the cuda kv ring's sliding layers cap at window+256, eager cuda path, bf16 ring cache; run ONE depth per process via NV_CTX_TOKENS"]
fn laguna_cuda_decode_ms_per_token_vs_context_depth_eager_path_chunked_prefill_primed() {
    if std::env::var("NV_LAGUNA_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_LAGUNA_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Laguna::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    assert_fp8_decode_routes_to_gscores_scratch_at_the_196k_rung(model.config(), &device);

    let depths = ctx_tokens_from_env_default_256_8k_196k();
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
        let chunk = PREFILL_CHUNK_256_LAGUNA_KV_RING_SLIDING_CAP_IS_WINDOW_PLUS_COMPACT_SLACK_256_AND_AFTER_A_SHIFT_KEEPS_WINDOW_ROWS_SO_A_STEADY_STATE_APPEND_ABOVE_256_TRIPS_THE_SLIDING_CAPACITY_BAIL;
        let prime_start = std::time::Instant::now();
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
        let prime_s = prime_start.elapsed().as_secs_f64();

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
            "CTX-SCALING laguna-cuda-eager depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the Laguna-XS-2.1 NVFP4 checkpoint; set NV_LAGUNA_CTX_TEST=1 -- same synthetic depth ladder as the eager synthfill instrument but decoding through LagunaStepGraph, the whole-step M=1 capture that serving's spec path uses by default (graph-vs-eager identity is asserted by laguna_serve_spec_matches_normal_greedy); run ONE depth per process via NV_CTX_TOKENS"]
fn laguna_cuda_decode_ms_per_token_vs_context_depth_step_graph_whole_step_capture_synthetic_cache_fill(
) {
    if std::env::var("NV_LAGUNA_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_LAGUNA_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = std::sync::Arc::new(
        Laguna::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model"),
    );
    drop(weights);

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().max_position_embeddings;
    assert!(
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK < max_pos,
        "requested depth {max_depth} exceeds max_position_embeddings {max_pos}"
    );
    let n_layers = model.config().num_hidden_layers;
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let chunk = PREFILL_CHUNK_256_LAGUNA_KV_RING_SLIDING_CAP_IS_WINDOW_PLUS_COMPACT_SLACK_256_AND_AFTER_A_SHIFT_KEEPS_WINDOW_ROWS_SO_A_STEADY_STATE_APPEND_ABOVE_256_TRIPS_THE_SLIDING_CAPACITY_BAIL;
    let vals =
        fixed_seed_small_nonzero_values_so_laguna_moe_routing_is_not_degenerate_single_expert(
            chunk * n_kv * hd,
        );
    let k_template = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("k bf16");
    let v_template = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("v bf16");

    for &depth in &depths {
        let mut cache = model
            .new_kv_cache(depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK)
            .expect("kv cache");
        let fill_start = std::time::Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            cache
                .prepare_for_decode(pos, pos + n)
                .expect("prepare_for_decode");
            for li in 0..n_layers {
                if n == chunk {
                    cache
                        .write_at(li, &k_template, &v_template)
                        .expect("write_at");
                } else {
                    let kn = k_template.narrow(1, 0, n).expect("k tail");
                    let vn = v_template.narrow(1, 0, n).expect("v tail");
                    cache.write_at(li, &kn, &vn).expect("write_at tail");
                }
            }
            cache.advance(n);
            pos += n;
        }
        device
            .synchronize()
            .expect("sync before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            cache.current_len(),
            depth,
            "synthetic fill must leave the cache at the requested depth"
        );

        let mut sg =
            nv_models::laguna_step_graph::LagunaStepGraph::new(std::sync::Arc::clone(&model), cache)
                .expect("step graph over the synthetically filled ring cache");
        let mut p = depth;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                sg.step(2000u32 + (p as u32 % 30000))
                    .unwrap_or_else(|e| panic!("graphed decode at depth {p}: {e:#}"));
                sg.argmax_device()
                    .unwrap_or_else(|e| panic!("argmax readback at depth {p}: {e:#}"));
                p += 1;
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = step_ms[step_ms.len() / 2];
        eprintln!(
            "CTX-SCALING laguna-cuda-step-graph-synthfill depth={depth} median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

fn fixed_seed_small_nonzero_values_so_laguna_moe_routing_is_not_degenerate_single_expert(
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
#[ignore = "loads the Laguna-XS-2.1 NVFP4 checkpoint; set NV_LAGUNA_CTX_TEST=1 -- same depth ladder but ONLY the attention-KV rows are synthetically deep (filled via prepare_for_decode/write_at/advance in ring-legal 256 chunks; decode ms/token reads cache SIZE, not values), so the 196k point costs seconds instead of an hour of chunked prefill; run ONE depth per process via NV_CTX_TOKENS"]
fn laguna_cuda_decode_ms_per_token_vs_context_depth_synthetic_cache_fill_only_attention_kv_rows_are_synthetic(
) {
    if std::env::var("NV_LAGUNA_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_LAGUNA_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = laguna_xs21_nvfp4_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = LagunaConfig::from_hf_json_str(&raw_cfg).expect("parse config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Laguna::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    drop(weights);

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().max_position_embeddings;
    assert!(
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK < max_pos,
        "requested depth {max_depth} exceeds max_position_embeddings {max_pos}"
    );
    let n_layers = model.config().num_hidden_layers;
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let chunk = PREFILL_CHUNK_256_LAGUNA_KV_RING_SLIDING_CAP_IS_WINDOW_PLUS_COMPACT_SLACK_256_AND_AFTER_A_SHIFT_KEEPS_WINDOW_ROWS_SO_A_STEADY_STATE_APPEND_ABOVE_256_TRIPS_THE_SLIDING_CAPACITY_BAIL;
    let vals =
        fixed_seed_small_nonzero_values_so_laguna_moe_routing_is_not_degenerate_single_expert(
            chunk * n_kv * hd,
        );
    let k_template = Tensor::from_vec(vals.clone(), (1usize, chunk, n_kv, hd), &device)
        .expect("k template")
        .to_dtype(DType::BF16)
        .expect("k bf16");
    let v_template = Tensor::from_vec(vals, (1usize, chunk, n_kv, hd), &device)
        .expect("v template")
        .to_dtype(DType::BF16)
        .expect("v bf16");

    for &depth in &depths {
        let mut cache = model
            .new_kv_cache(depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK)
            .expect("kv cache");
        let fill_start = std::time::Instant::now();
        let mut pos = 0usize;
        while pos < depth {
            let n = chunk.min(depth - pos);
            cache
                .prepare_for_decode(pos, pos + n)
                .expect("prepare_for_decode");
            for li in 0..n_layers {
                if n == chunk {
                    cache
                        .write_at(li, &k_template, &v_template)
                        .expect("write_at");
                } else {
                    let kn = k_template.narrow(1, 0, n).expect("k tail");
                    let vn = v_template.narrow(1, 0, n).expect("v tail");
                    cache.write_at(li, &kn, &vn).expect("write_at tail");
                }
            }
            cache.advance(n);
            pos += n;
        }
        device
            .synchronize()
            .expect("sync before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            cache.current_len(),
            depth,
            "synthetic fill must leave the cache at the requested depth"
        );

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
            "CTX-SCALING laguna-cuda-eager-synthfill depth={depth} median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}
