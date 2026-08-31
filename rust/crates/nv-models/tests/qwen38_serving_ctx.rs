#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use common::ctx_tokens_from_env_default_256_8k_196k;
use common::percentile_of_sorted;
use common::serving_feed_argmax_and_basis_label_env_selects_the_pre_2026_08_20_scalar;
use common::serving_feed_device_argmax_env_default_off_because_every_published_ladder_number_is_a_host_feed;
use candle_core::{DType, Device, Tensor};
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{LayerType, Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;

mod ctx_timing_common;
mod hub_snapshot;
mod prime_ckpt_common;
use common::decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state;
use common::qwen38_snapshot_dir_env_override_then_home_hub;

const PRIME_CKPT_FILL_MODE_SYNTHFILL512_V1_FIXED_SEED_CHUNKED: &str = "synthfill512v1";

const SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK: usize = 512;
const STREAM_IDENTITY_GATE_32_STEPS_OLD_AND_NEW_ARGMAX_MUST_AGREE_ON_REAL_LOGITS_ONCE_PER_SESSION:
    usize = 32;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS: usize = 16;

fn fixed_seed_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax(
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
#[ignore = "loads the ~22.6 GB Qwen3.8-27B NVFP4 on the eager cuda dense arm; set NV_QWEN38_SERVING_TEST=1 -- decode ms/token vs KV depth on the 262144-max-pos 256/8k/196k ladder; the fp8 KV at every full-attention slot is filled synthetically and current_len advanced the way prefill would (decode cost reads cache SIZE, gdn state has no depth dimension); run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn qwen38_cuda_dense_decode_ms_per_token_vs_context_depth_synthetic_cache_fill_eager() {
    if std::env::var("NV_QWEN38_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_QWEN38_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; a 196k serving gate that silently reports ok would hide \
                 exactly the failure class task #106 exists to catch"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN38_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect(
            "build Qwen3.8-27B on the eager cuda dense arm; a trip here on the real unsloth \
             checkpoint is the track-1 NVFP4 format gap (mixed fp8 attn + nvfp4 mlp), not a \
             ladder bug",
        );
    drop(weights);
    assert_eq!(
        model.dense_intermediate(),
        Some(cfg.intermediate_size),
        "from_loader_dense_quantized did not carry intermediate_size, so this is not the dense arm"
    );

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

    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let chunk = SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK;
    let vals = fixed_seed_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax(
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

    let ckpt_dir = prime_ckpt_common::prime_ckpt_dir_env_off_by_default_so_the_ladder_defaults_never_change();
    for &depth in &depths {
        let mut cache = model
            .new_kv_cache(depth + per_depth_extra_slots)
            .expect("kv cache");
        let fingerprint =
            prime_ckpt_common::fingerprint_of_checkpoint_dims_depth_fillmode_and_cache_layout_version(
                &raw_cfg,
                model.config().num_hidden_layers,
                n_kv,
                hd,
                depth,
                PRIME_CKPT_FILL_MODE_SYNTHFILL512_V1_FIXED_SEED_CHUNKED,
            );
        let mut prime = |cache: &mut nv_models::qwen3_5_moe::Qwen3MoeKvCache| -> anyhow::Result<()> {
            let mut pos = 0usize;
            while pos < depth {
                let n = chunk.min(depth - pos);
                let (kn, vn) = if n == chunk {
                    (k_template.clone(), v_template.clone())
                } else {
                    (
                        k_template.narrow(1, 0, n).expect("k tail"),
                        v_template.narrow(1, 0, n).expect("v tail"),
                    )
                };
                cache
                    .write_synthetic_rows_at_every_full_attention_slot_for_depth_timing_decode_reads_cache_size_not_values(
                        pos, &kn, &vn,
                    )
                    .unwrap_or_else(|e| panic!("synthetic fill chunk at pos {pos}: {e:#}"));
                pos += n;
            }
            device
                .synchronize()
                .expect("sync before stopping the fill clock");
            Ok(())
        };
        let full_attn_slots = model
            .config()
            .layer_types
            .iter()
            .filter(|t| matches!(t, LayerType::FullAttention))
            .count();
        let expected_file_bytes =
            prime_ckpt_common::expected_ckpt_file_bytes_fp8_kv_rows_scales_plus_lin_state_slack(
                full_attn_slots,
                depth,
                n_kv,
                hd,
            );
        let fill_start = Instant::now();
        let (prime_source, _restore_or_prime_s) =
            prime_ckpt_common::restore_or_prime_then_dump_holding_the_flock(
                &mut cache,
                ckpt_dir.as_deref(),
                &fingerprint,
                expected_file_bytes,
                &mut prime,
            )
            .unwrap_or_else(|e| panic!("restore-or-prime at depth {depth}: {e:#}"));
        device
            .synchronize()
            .expect("sync before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            cache.current_len(),
            depth,
            "synthetic fill must advance current_len exactly the way prefill would"
        );

        let mut step_pos = depth;
        let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                let tokens = Tensor::from_vec(
                    vec![2000u32 + (step_pos as u32 % 30000)],
                    (1usize, 1usize),
                    &device,
                )
                .expect("token");
                let positions =
                    Tensor::from_vec(vec![step_pos as i32], 1usize, &device).expect("position");
                let logits = model
                    .forward_with_cache(&tokens, &positions, &mut cache)
                    .unwrap_or_else(|e| panic!("decode at depth {step_pos}: {e:#}"));
                logits
                    .to_dtype(DType::F32)
                    .expect("f32")
                    .flatten_all()
                    .expect("flatten")
                    .to_vec1::<f32>()
                    .expect("sync to host");
                step_pos += 1;
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
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
        eprintln!(
            "CTX-SCALING qwen38-cuda-synthfill depth={depth} basis=eager_dense_forward_with_cache_includes_logits_host_synthetic_kv_fill median_ms_tok={median:.3} p10={p10:.3} p90={p90:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps} prime_source={}",
            1000.0 / median,
            step_ms.len(),
            prime_source.label()
        );
    }
}

#[test]
#[ignore = "loads the ~54 GB Qwen3.8-27B NVFP4 dense arm GRAPHED; set NV_QWEN38_SERVING_TEST=1 -- serving-class decode ms/token vs KV depth on the captured graphed engine: the fp8 KV is filled synthetically and current_pos advanced the way prefill would; the dense trunk needs no MoE dispatch so install_grouped_moe only arms capture; capture_active is asserted; set NV_Q36_GRAPHED_DECODE_FIX=1 to route decode_attention_fp8 to flash_decode_fused_fp8kv (24q/4kv hd256 fits the splitk kMaxHD=512 kernel); run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn qwen38_cuda_dense_serving_decode_ms_per_token_vs_context_depth_synthetic_cache_fill_graphed() {
    if std::env::var("NV_QWEN38_SERVING_TEST").ok().as_deref() != Some("1") {
        if std::env::var("NV_MODELS_ALLOW_SKIP").as_deref() != Ok("1") {
            panic!(
                "set NV_QWEN38_SERVING_TEST=1 to run this gate, or NV_MODELS_ALLOW_SKIP=1 to \
                 skip it on purpose; a 196k serving gate that silently reports ok would hide \
                 exactly the failure class task #106 exists to catch"
            );
        }
        eprintln!("SKIP (NV_MODELS_ALLOW_SKIP=1): set NV_QWEN38_SERVING_TEST=1 to run");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = qwen38_snapshot_dir_env_override_then_home_hub();
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, &device)
        .expect("build Qwen3.8-27B on the dense arm for the graphed serving ladder");
    drop(weights);
    assert!(
        model.is_dense(),
        "from_loader_dense_quantized must yield the dense arm; the graphed dense branch \
         (no MoE dispatch needed for capture) is the whole point of this gate"
    );

    let depths = ctx_tokens_from_env_default_256_8k_196k();
    let max_depth = depths.iter().copied().max().unwrap();
    let max_pos = model.config().max_position_embeddings;
    let n_kv = model.config().num_key_value_heads;
    let hd = model.config().head_dim;
    let per_depth_extra_slots =
        ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
            + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
            + KV_SLOT_HEADROOM_16_BEYOND_THE_TIMED_STEPS
            + STREAM_IDENTITY_GATE_32_STEPS_OLD_AND_NEW_ARGMAX_MUST_AGREE_ON_REAL_LOGITS_ONCE_PER_SESSION;
    assert!(
        max_depth + per_depth_extra_slots < max_pos,
        "depth {max_depth} + decode steps exceeds max_position_embeddings {max_pos}; a model \
         declaring {max_pos} must serve this depth, so a trip here is a config bug"
    );
    let cache_slots = max_depth + per_depth_extra_slots;
    nv_models::qwen3_5_moe::trunk_events::prebuild_if_armed_because_the_ladder_cannot_allocate_mid_capture(
        &device,
        model.config().num_hidden_layers,
    );
    let mut eng = GraphedQwen3Moe::new(model, &device, cache_slots)
        .unwrap_or_else(|e| panic!("GraphedQwen3Moe with {cache_slots} kv slots: {e:#}"));

    let chunk = SYNTHETIC_FILL_CHUNK_512_MATCHES_THE_PROVEN_QWEN35_DENSE_PREFILL_BLOCK;
    let vals = fixed_seed_values_because_an_all_zero_cache_is_an_unrealistically_easy_softmax(
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

    let (feed_argmax, host_feed_basis) =
        serving_feed_argmax_and_basis_label_env_selects_the_pre_2026_08_20_scalar();
    let device_feed =
        serving_feed_device_argmax_env_default_off_because_every_published_ladder_number_is_a_host_feed();
    let feed_basis = if device_feed {
        "engine_device_argmax_f32_rows_4_byte_exit"
    } else {
        host_feed_basis
    };
    let mut stream_identity_gate_ran = false;
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

        eng.install_grouped_moe()
            .expect("dense branch of install_grouped_moe arms capture without a dispatch");
        assert!(
            eng.moe_dispatch_ref().is_none(),
            "the dense arm must not carry a MoE dispatch; one here means the dense branch \
             regressed into building GroupedMoeDispatch over plain MLPs"
        );

        let mut cur = 2000u32;
        if !stream_identity_gate_ran {
            for _ in
                0..STREAM_IDENTITY_GATE_32_STEPS_OLD_AND_NEW_ARGMAX_MUST_AGREE_ON_REAL_LOGITS_ONCE_PER_SESSION
            {
                eng.forward_decode(cur)
                    .unwrap_or_else(|e| panic!("stream-identity gate decode at depth {depth}: {e:#}"));
                let device_tok = if device_feed {
                    Some(eng.next_token_argmax_on_device().unwrap_or_else(|e| {
                        panic!("stream-identity gate device argmax at depth {depth}: {e:#}")
                    }))
                } else {
                    None
                };
                let r = eng.logits_host().expect("gate logits to host");
                let old_tok = argmax(&r);
                let new_tok = feed_argmax(&r);
                assert_eq!(
                    old_tok, new_tok,
                    "the {host_feed_basis} argmax diverged from the pre-2026-08-20 scalar on real \
                     logits at gate pos {}; the depth ladder's measured token stream must be \
                     IDENTICAL across feed paths or old/new medians are not comparable",
                    eng.current_pos()
                );
                if let Some(dt) = device_tok {
                    assert_eq!(
                        dt, new_tok,
                        "the {feed_basis} diverged from the host feed on real logits at gate pos \
                         {}; next_token_argmax_on_device must reproduce argmax_checked semantics \
                         (cast_bf16_f32 widens exactly, argmax_f32_rows skips non-finite and \
                         breaks ties lowest-index) or the device-fed ladder measures a different \
                         token stream",
                        eng.current_pos()
                    );
                }
                cur = new_tok;
            }
            stream_identity_gate_ran = true;
        }
        let (warmup_steps, step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                eng.forward_decode(cur)
                    .unwrap_or_else(|e| panic!("serving decode at depth {depth}: {e:#}"));
                cur = if device_feed {
                    eng.next_token_argmax_on_device()
                        .unwrap_or_else(|e| panic!("device argmax at depth {depth}: {e:#}"))
                } else {
                    feed_argmax(&eng.logits_host().expect("logits to host"))
                };
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
        eng.underlying()
            .dump_trunk_phase_ladder_after_replay_when_nv_q36_trunk_events_1();
        for (tok, census) in eng.captured_graph_node_type_census_per_shape_token() {
            eprintln!("[q38-serving-ctx] census shape_token={tok} {census:?}");
        }
        eprintln!(
            "CTX-SCALING qwen38-cuda-serving-synthfill depth={depth} basis=graphed_dense_fused_gdn_free_running_decode_argmax_feed_includes_logits_host_synthetic_kv_fill argmax_feed={feed_basis} decode_kernel={} capture_active={} median_ms_tok={median:.3} p10={p10:.3} p90={p90:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps}",
            decode_fix_label_reports_the_nv_q36_graphed_decode_fix_gate_state(),
            eng.capture_active(),
            1000.0 / median,
            step_ms.len()
        );
    }
}
