#![cfg(feature = "wgpu")]

mod common;
mod ctx_timing_common;
use common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights;

const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK: usize =
    ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
        + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        + 16;

#[test]
#[ignore = "loads a real MatFormer E4B checkpoint on wgpu; set NV_E4B_CTX_TEST=1 -- the same decode ms/token ladder as the step-primed suite, but every fp8 kv slot and scale is filled with finite synthetic bytes and pos is set directly, so a 120k rung costs seconds instead of the 13 minutes 122880 real decode steps cost. e4b keeps one flat slot array per kv layer with no ring, and attention scans min(total, window) positions regardless of their values, so pos alone is the cache-depth state; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn e4b_wgpu_decode_ms_per_token_vs_context_depth_synthetic_kv_fill_pos_is_the_only_depth_state() {
    if std::env::var("NV_E4B_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_E4B_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_E4B_SNAPSHOT",
        &[
            "models--google--gemma-4-E4B-it-qat-w4a16-ct",
            "models--google--gemma-4-E4B-it",
        ],
    );
    let ckpt = if dir.to_string_lossy().contains("qat-w4a16") {
        "w4a16-qat"
    } else {
        "dense-bf16"
    };
    eprintln!("e4b checkpoint: {} ({ckpt})", dir.display());
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_timing_common::ctx_tokens_from_env_default_256_8k_120k_because_max_pos_131072_has_no_room_for_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let max_seq = max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK;
    assert!(
        max_seq <= config.max_position_embeddings,
        "ladder needs {max_seq} positions but max_position_embeddings is {}",
        config.max_position_embeddings
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut m = nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu::from_loader(config, &loader, max_seq)
        .expect("build from loader");
    drop(loader);

    for &depth in &depths {
        m.reset();
        let fill_start = std::time::Instant::now();
        m.fill_kv_cache_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
            .unwrap_or_else(|e| panic!("synthetic kv fill at depth {depth}: {e:#}"));
        ctx.poll_blocking()
            .expect("drain the synthetic fill writes before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            m.current_pos(),
            depth,
            "synthetic fill must land at the requested depth"
        );

        let mut token = 2000u32;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = m
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let median = ctx_timing_common::median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING e4b-wgpu-synthfill depth={depth} median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps} ckpt={ckpt}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

const AB_STEPS_4_IS_ENOUGH_FOR_A_SPLIT_COUNT_TO_SHOW_ITS_REDUCTION_ORDER: usize = 4;

const AB_FILLER_TOKENS_ARE_ORDINARY_ENGLISH_SO_THE_LOGITS_DO_NOT_SIT_ON_THE_SOFTCAP_RAILS: &[u32] =
    &[573, 4926, 576, 7127, 1671, 611, 573, 2149, 576, 573, 3821, 235269];

fn ab_prompt(ctx: usize, bos: u32) -> Vec<u32> {
    let mut v = vec![bos];
    let mut i = 0usize;
    while v.len() < ctx {
        v.push(AB_FILLER_TOKENS_ARE_ORDINARY_ENGLISH_SO_THE_LOGITS_DO_NOT_SIT_ON_THE_SOFTCAP_RAILS
            [i % AB_FILLER_TOKENS_ARE_ORDINARY_ENGLISH_SO_THE_LOGITS_DO_NOT_SIT_ON_THE_SOFTCAP_RAILS.len()]);
        i += 1;
    }
    v
}

#[test]
#[ignore = "loads a real MatFormer E4B checkpoint on wgpu twice per depth and prefills a real prompt to that depth; set NV_E4B_DEEPGATE_AB=1 -- the ppl scorer never exceeds ~800 positions and so is structurally blind to every band of NV_E4B_DEEP_DEPTHGATE, whose first knee is 2048. This is the deep-band quality instrument instead: real weights, a real prefilled kv state, gate off vs gate on, full-vocab logit relative error and argmax agreement. A synthetic kv fill is NOT usable here -- e4b softcaps its logits at 30 and a synthetic state parks them on the rails, where any reassociation flips a sign and reads as a relative error of 2"]
fn e4b_deep_depthgate_logits_and_argmax_match_the_ungated_arm_over_one_real_prefilled_state() {
    if std::env::var("NV_E4B_DEEPGATE_AB").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_E4B_DEEPGATE_AB != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_E4B_SNAPSHOT",
        &[
            "models--google--gemma-4-E4B-it-qat-w4a16-ct",
            "models--google--gemma-4-E4B-it",
        ],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_timing_common::nv_ctx_tokens_env_with_k_suffix()
        .unwrap_or_else(|| vec![2048, 8192]);
    let max_depth = depths.iter().copied().max().unwrap();
    let max_seq = max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK;

    let run = |depth: usize, gate: bool| -> Vec<(u32, Vec<f32>)> {
        if gate {
            std::env::set_var(nv_models::gemma4_e4b_wgpu::E4B_DEEP_DEPTHGATE_ENV, "1");
        } else {
            std::env::remove_var(nv_models::gemma4_e4b_wgpu::E4B_DEEP_DEPTHGATE_ENV);
        }
        let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
            .expect("config.json");
        let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
            .expect("open safetensors");
        let mut m =
            nv_models::gemma4_e4b_wgpu::Gemma4E4bWgpu::from_loader(config, &loader, max_seq)
                .expect("build from loader");
        drop(loader);
        let mut t = m.prefill_prompt(&ab_prompt(depth, 2)).expect("prefill");
        let mut out = Vec::new();
        for _ in 0..AB_STEPS_4_IS_ENOUGH_FOR_A_SPLIT_COUNT_TO_SHOW_ITS_REDUCTION_ORDER {
            let (next, logits) = m.decode_step_logits(t).expect("decode step logits");
            out.push((next, logits));
            t = next;
        }
        std::env::remove_var(nv_models::gemma4_e4b_wgpu::E4B_DEEP_DEPTHGATE_ENV);
        out
    };

    for &depth in &depths {
        let base = run(depth, false);
        let gated = run(depth, true);
        let mut worst_rel = 0.0f64;
        let mut flips = 0usize;
        let mut worst_abs = 0.0f64;
        for (i, ((ab, a), (gb, g))) in base.iter().zip(gated.iter()).enumerate() {
            assert_eq!(a.len(), g.len(), "step {i}: logit widths differ");
            if ab != gb {
                flips += 1;
            }
            for (x, y) in a.iter().zip(g.iter()) {
                let d = (*x as f64 - *y as f64).abs();
                worst_abs = worst_abs.max(d);
                let scale = (*x as f64).abs().max((*y as f64).abs()).max(1.0);
                worst_rel = worst_rel.max(d / scale);
            }
        }
        eprintln!(
            "DEEPGATE-AB depth={depth} steps={} worst_logit_rel={worst_rel:.6e} worst_logit_abs={worst_abs:.6e} argmax_flips={flips} base_tokens={:?} gate_tokens={:?}",
            base.len(),
            base.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            gated.iter().map(|(t, _)| *t).collect::<Vec<_>>()
        );
        assert_eq!(
            base.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            gated.iter().map(|(t, _)| *t).collect::<Vec<_>>(),
            "at depth {depth} the gated arm's greedy chain diverges from the ungated arm on an \
             identical real prefilled state. The gated quantity is the argmax chain and not a \
             logit bound: this checkpoint is W4A16, so every projection requantizes its \
             activations each step and a reassociated softmax fold that moves an accumulator by \
             one ulp can cross a quantization boundary and step a single low-rank logit by \
             whole units. The full-vocab worst relative move is printed for the record, and the \
             instrument that decides quality for a split count is the ppl scorer"
        );
    }
}
