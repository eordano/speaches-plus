#![cfg(feature = "wgpu")]

mod common;
use common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights;

mod ctx_timing_common;

const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK: usize =
    ctx_timing_common::WORST_CASE_PLATEAU_WARMUP_200_STEPS_SIZES_KV_SLOT_HEADROOM
        + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN
        + 16;

fn nv_ctx_tokens_env_with_k_suffix() -> Option<Vec<usize>> {
    std::env::var("NV_CTX_TOKENS").ok().map(|v| {
        v.split(',')
            .map(|s| {
                let s = s.trim();
                let (num, mult) = match s.strip_suffix(['k', 'K']) {
                    Some(n) => (n, 1024),
                    None => (s, 1),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect()
    })
}

fn ctx_tokens_from_env_default_256_8k_120k_because_max_pos_131072_has_no_room_for_168k(
) -> Vec<usize> {
    nv_ctx_tokens_env_with_k_suffix().unwrap_or_else(|| vec![256, 8192, 120 * 1024])
}

fn median_ms_of_timed_steps(step_ms: &mut Vec<f64>) -> f64 {
    step_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    step_ms[step_ms.len() / 2]
}

fn print_wgpu_profile_share_table_then_reset(tag: &str) {
    use nv_kernels::wgpu_backend::dispatch::profile;
    if !profile::enabled() {
        return;
    }
    let rows = profile::report();
    let n: u64 = rows.iter().map(|r| r.1).sum();
    let total: f64 = rows.iter().map(|r| r.2).sum();
    eprintln!(
        "[wgpu-prof] {tag}: {} labels {n} dispatches {:.1} ms GPU",
        rows.len(),
        total / 1e6
    );
    for (label, count, ns) in rows.into_iter().take(48) {
        eprintln!(
            "[wgpu-prof] {tag} | {label} n={count} total_ms={:.1} share={:.1}%",
            ns / 1e6,
            100.0 * ns / total.max(1.0)
        );
    }
    profile::reset();
}

const AB_STEPS_48_CROSS_SPLIT_AND_ROUND_BOUNDARIES_AT_DEPTH: usize = 48;

#[test]
#[ignore = "loads the ~13 GB gpt-oss-20b MXFP4 on wgpu twice; set NV_GPTOSS_CTX_TEST=1 -- greedy argmax A/B of NV_GPTOSS_WGPU_FLASH_DECODE=1 against the serial arm from the same synthetic state at depth (default 8192, override via single-entry NV_CTX_TOKENS); the flash split/round merge math must be argmax-invariant"]
fn gptoss_wgpu_flash_decode_argmax_matches_the_serial_arm_from_the_same_synthetic_state() {
    if std::env::var("NV_GPTOSS_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GPTOSS_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_GPTOSS_DIR",
        &["models--openai--gpt-oss-20b"],
    );
    let depth = nv_ctx_tokens_env_with_k_suffix()
        .map(|v| v[0])
        .unwrap_or(8192);
    let cfg = nv_models::gpt_oss_wgpu::GptOssConfig::from_hf_json_file(dir.join("config.json"))
        .expect("config.json");
    let max_seq = depth + AB_STEPS_48_CROSS_SPLIT_AND_ROUND_BOUNDARIES_AT_DEPTH + 16;
    assert!(max_seq <= cfg.max_position_embeddings);
    assert!(
        matches!(
            cfg.layer_types[1],
            nv_models::gpt_oss_wgpu::GptOssLayerType::Full
        ),
        "the probe gate reads layer 1 as the first full-attention layer"
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let ab_env = std::env::var("NV_GPTOSS_AB_ENV")
        .unwrap_or_else(|_| nv_models::gpt_oss_wgpu::FLASH_DECODE_ENV.to_string());
    let mut forced: Vec<u32> = vec![2000];
    let mut arms: Vec<Vec<Vec<f32>>> = Vec::new();
    let mut probes: Vec<(&str, usize, Vec<f32>)> = Vec::new();
    for arm in ["0", "1"] {
        std::env::set_var(&ab_env, arm);
        let mut gpu =
            nv_models::gpt_oss_wgpu::GptOssWgpu::from_loader(cfg.clone(), &loader, max_seq)
                .unwrap_or_else(|e| panic!("build from loader (arm {arm}): {e:#}"));
        gpu.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
            .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
        ctx.poll_blocking().expect("drain synthetic fill writes");
        let mut steps: Vec<Vec<f32>> = Vec::new();
        for i in 0..AB_STEPS_48_CROSS_SPLIT_AND_ROUND_BOUNDARIES_AT_DEPTH {
            let (argmax, logits) = gpu
                .decode_step_logits(forced[i])
                .unwrap_or_else(|e| panic!("decode step at depth {depth} (arm {arm}): {e:#}"));
            steps.push(logits);
            if arm == "0" {
                forced.push(argmax);
            }
            if i == 0 {
                for li in [1usize, 3, 23] {
                    let p = gpu
                        .debug_probe(&format!("attnpk{li}"))
                        .unwrap_or_else(|| panic!("attnpk{li} probe missing"));
                    probes.push((arm, li, p));
                }
            }
        }
        arms.push(steps);
    }
    std::env::remove_var(&ab_env);
    let mut first_full_layer_rel = f32::INFINITY;
    for li in [1usize, 3, 23] {
        let s = &probes.iter().find(|(a, l, _)| *a == "0" && *l == li).unwrap().2;
        let f = &probes.iter().find(|(a, l, _)| *a == "1" && *l == li).unwrap().2;
        let norm = s.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        let max_abs = s
            .iter()
            .zip(f.iter())
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        eprintln!(
            "AB-PROBE layer {li} attn_bf16 step0: max_abs {max_abs:.6} peak {norm:.4} rel {:.6}",
            max_abs / norm
        );
        if li == 1 {
            first_full_layer_rel = max_abs / norm;
        }
    }
    let mut flips = 0usize;
    let mut min_flip_margin = f32::INFINITY;
    let mut worst_rel = 0.0f32;
    for (i, (s, f)) in arms[0].iter().zip(arms[1].iter()).enumerate() {
        let norm = s.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
        let max_abs = s
            .iter()
            .zip(f.iter())
            .fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
        worst_rel = worst_rel.max(max_abs / norm);
        let am = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        let (a_s, a_f) = (am(s), am(f));
        if a_s != a_f {
            flips += 1;
            let margin = s[a_s] - s[a_f];
            min_flip_margin = min_flip_margin.min(margin);
            eprintln!(
                "AB-FLIP step {i}: serial argmax {a_s} ({}) vs flash {a_f} ({}), serial margin {margin:.5}, max_abs {max_abs:.5}",
                s[a_s], s[a_f]
            );
        }
    }
    eprintln!(
        "AB-LOGITS gptoss-wgpu depth={depth} steps={} worst_rel={worst_rel:.6} flips={flips} min_flip_margin={min_flip_margin:.5}",
        AB_STEPS_48_CROSS_SPLIT_AND_ROUND_BOUNDARIES_AT_DEPTH
    );
    assert!(
        first_full_layer_rel < KERNEL_PARITY_BAND_1E3_LAYER1_INPUTS_ARE_IDENTICAL_BOTH_ARMS_SO_ONLY_THE_TOUCHED_ATTENTION_KERNEL_DIFFERS,
        "the first full-attention layer sees identical inputs in both arms, so its attn output \
         must match to reduction-order noise: rel {first_full_layer_rel} at depth {depth}; \
         downstream logit drift is NOT gated here because the accepted MX_SG reordered-sum arm \
         (attention untouched, probe rel 0.0) produces the same worst_rel ~0.3 and flips the \
         same near-tie steps on this synthetic state -- run NV_GPTOSS_AB_ENV=\
         NV_GPTOSS_WGPU_MX_SG_REORDERED_SUM to reproduce the null"
    );
}

const KERNEL_PARITY_BAND_1E3_LAYER1_INPUTS_ARE_IDENTICAL_BOTH_ARMS_SO_ONLY_THE_TOUCHED_ATTENTION_KERNEL_DIFFERS: f32 = 1e-3;

#[test]
#[ignore = "loads the ~13 GB gpt-oss-20b MXFP4 on wgpu; set NV_GPTOSS_CTX_TEST=1 -- decode ms/token vs KV depth on the 256/8k/120k ladder (max_pos 131072), every state buffer filled with finite synthetic bytes and pos set directly (the flat kv caches index slots by pos and the sliding-window start is a pure function of pos; MoE routing is only weakly value-dependent and the byte pattern keeps it non-degenerate), so the deep rung costs seconds instead of a step-primed hour; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn gptoss_wgpu_decode_ms_per_token_vs_context_depth_synthetic_state_fill_pos_is_the_only_depth_state(
) {
    if std::env::var("NV_GPTOSS_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GPTOSS_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_GPTOSS_DIR",
        &["models--openai--gpt-oss-20b"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_120k_because_max_pos_131072_has_no_room_for_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let cfg = nv_models::gpt_oss_wgpu::GptOssConfig::from_hf_json_file(dir.join("config.json"))
        .expect("config.json");
    let max_seq = max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK;
    assert!(
        max_seq <= cfg.max_position_embeddings,
        "ladder needs {max_seq} positions but max_position_embeddings is {}",
        cfg.max_position_embeddings
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut gpu = nv_models::gpt_oss_wgpu::GptOssWgpu::from_loader(cfg, &loader, max_seq)
        .expect("build from loader");
    drop(loader);

    for &depth in &depths {
        gpu.reset().expect("reset between depth arms");
        let fill_start = std::time::Instant::now();
        gpu.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
            .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
        ctx.poll_blocking()
            .expect("drain synthetic fill writes before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(
            gpu.current_pos(),
            depth,
            "synthetic fill must land at the requested depth"
        );

        let mut token = 2000u32;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = gpu
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING gptoss-wgpu-synthfill depth={depth} median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} max_seq={max_seq} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
        print_wgpu_profile_share_table_then_reset(&format!("gptoss-wgpu-synthfill-{depth}"));
    }
}
