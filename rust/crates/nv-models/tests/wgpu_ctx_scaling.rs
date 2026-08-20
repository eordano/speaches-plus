#![cfg(feature = "wgpu")]

use std::path::PathBuf;

mod common;
mod ctx_timing_common;
mod official_template;
use common::snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights;

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
                let (num, mult) = match s.strip_suffix('k') {
                    Some(n) => (n, 1024usize),
                    None => (s, 1usize),
                };
                num.parse::<usize>().expect("NV_CTX_TOKENS entry") * mult
            })
            .collect()
    })
}

fn ctx_tokens_from_env_default_256_8k_168k() -> Vec<usize> {
    nv_ctx_tokens_env_with_k_suffix().unwrap_or_else(|| vec![256, 8 * 1024, 168 * 1024])
}

fn ctx_tokens_from_env_default_256_8k_120k_because_max_pos_131072_has_no_room_for_168k(
) -> Vec<usize> {
    nv_ctx_tokens_env_with_k_suffix().unwrap_or_else(|| vec![256, 8 * 1024, 120 * 1024])
}

fn ctx_tokens_from_env_default_256_8k_196k_the_task_106_gate_ladder_for_262144_max_pos_models(
) -> Vec<usize> {
    nv_ctx_tokens_env_with_k_suffix().unwrap_or_else(|| vec![256, 8 * 1024, 196 * 1024])
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

#[test]
#[ignore = "loads the 31B on wgpu; set NV_GEMMA4_CTX_TEST=1 -- decode ms/token vs KV depth, primed by stepping (the 31B wgpu port has no M-row prefill)"]
fn gemma4_wgpu_decode_ms_per_token_vs_context_depth_step_primed() {
    if std::env::var("NV_GEMMA4_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GEMMA4_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let home = std::env::var("HOME").expect("HOME");
    let dir = match std::env::var("NV_G4_SNAPSHOT") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let base = PathBuf::from(&home)
                .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
            let mut c: Vec<PathBuf> = std::fs::read_dir(&base)
                .unwrap_or_else(|e| panic!("gemma4 snapshots dir {base:?} missing: {e}"))
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.join("config.json").is_file())
                .collect();
            c.sort();
            c.into_iter().next().expect("no gemma4 snapshot; set NV_G4_SNAPSHOT")
        }
    };
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("weights");
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader)
        .expect("host weight staging");
    drop(loader);
    let mut m = nv_models::gemma4_wgpu::Gemma4Wgpu::new(
        config,
        &host,
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK,
    )
    .expect("wgpu model");
    drop(host);

    for &depth in &depths {
        m.reset();
        let prime_start = std::time::Instant::now();
        let mut token = 2000u32;
        for p in 0..depth {
            token = m
                .decode_step(2000 + (p as u32 % 30000))
                .unwrap_or_else(|e| panic!("prime step {p}: {e:#}"));
        }
        let prime_s = prime_start.elapsed().as_secs_f64();
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = m
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING gemma4-wgpu depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the 31B on wgpu; set NV_GEMMA4_CTX_PREFILL_TEST=1 -- chunked-prefill tok/s plus decode ms/token (mean of 64) at each NV_CTX_TOKENS depth, synthetic repeat-token prompt"]
fn gemma4_wgpu_chunked_prefill_tok_s_then_decode_ms_tok_vs_context_depth() {
    if std::env::var("NV_GEMMA4_CTX_PREFILL_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GEMMA4_CTX_PREFILL_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let home = std::env::var("HOME").expect("HOME");
    let dir = match std::env::var("NV_G4_SNAPSHOT") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let base = PathBuf::from(&home)
                .join(".cache/huggingface/hub/models--nvidia--Gemma-4-31B-IT-NVFP4/snapshots");
            let mut c: Vec<PathBuf> = std::fs::read_dir(&base)
                .unwrap_or_else(|e| panic!("gemma4 snapshots dir {base:?} missing: {e}"))
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.join("config.json").is_file())
                .collect();
            c.sort();
            c.into_iter().next().expect("no gemma4 snapshot; set NV_G4_SNAPSHOT")
        }
    };
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("weights");
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader)
        .expect("host weight staging");
    drop(loader);
    let mut m = nv_models::gemma4_wgpu::Gemma4Wgpu::new(
        config,
        &host,
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK,
    )
    .expect("wgpu model");
    drop(host);
    eprintln!(
        "formats: {} | prefill chunk m={} passes/chunk={}",
        nv_models::gemma4_wgpu::weight_format_boot_line(),
        m.prefill_chunk_len(),
        m.prefill_pass_count()
    );
    assert!(
        m.prefill_chunk_len() >= 2,
        "chunked prefill is off on this config; a step-primed prefill number would not be a prefill number"
    );

    for &depth in &depths {
        m.reset();
        nv_kernels::wgpu_backend::dispatch::profile::reset();
        let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        let t0 = std::time::Instant::now();
        let done = m.prefill_tokens(&ids).unwrap_or_else(|e| panic!("prefill_tokens at depth {depth}: {e:#}"));
        for (j, t) in ids[done..].iter().enumerate() {
            m.prefill_step(*t).unwrap_or_else(|e| panic!("prefill tail step {}: {e:#}", done + j));
        }
        ctx.poll_blocking().expect("drain prefill work before stopping the clock");
        let prefill_s = t0.elapsed().as_secs_f64();
        assert_eq!(m.current_pos(), depth, "prefill must land exactly at the requested depth");
        print_wgpu_profile_share_table_then_reset(&format!("gemma4-wgpu prefill depth={depth}"));

        let mut token = 2000u32;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = m
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let mean = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-PREFILL gemma4-wgpu depth={depth} prefill_s={prefill_s:.3} prefill_tok_s={:.1} chunk_done={done} decode_mean_ms_tok={mean:.3} decode_median_ms_tok={median:.3} steps={} warmup_steps={warmup_steps}",
            depth as f64 / prefill_s,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the 31B on wgpu; set NV_GEMMA4_CTX_CHUNKED_TEST=1 -- decode ms/token vs KV depth primed by chunked prefill instead of stepping (step-priming 168k takes 4h+, ring-ladder-168k.log); emits the CTX-SCALING line with prime_s = chunked prefill wall time; sanity: at 8k median_ms_tok must match the step-primed test (current numbers: perf/runs.jsonl)"]
fn gemma4_wgpu_decode_ms_per_token_vs_context_depth_chunked_prefill_primed() {
    if std::env::var("NV_GEMMA4_CTX_CHUNKED_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GEMMA4_CTX_CHUNKED_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4_SNAPSHOT",
        &["models--nvidia--Gemma-4-31B-IT-NVFP4"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader)
        .expect("host weight staging");
    drop(loader);
    let mut m = nv_models::gemma4_wgpu::Gemma4Wgpu::new(
        config,
        &host,
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK,
    )
    .expect("wgpu model");
    drop(host);
    eprintln!(
        "formats: {} | prefill chunk m={} passes/chunk={}",
        nv_models::gemma4_wgpu::weight_format_boot_line(),
        m.prefill_chunk_len(),
        m.prefill_pass_count()
    );
    assert!(
        m.prefill_chunk_len() >= 2,
        "chunked prefill is off on this config; priming would degrade to step priming and the 168k rung is infeasible again"
    );

    for &depth in &depths {
        m.reset();
        let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        let prime_start = std::time::Instant::now();
        let done = m
            .prefill_tokens(&ids)
            .unwrap_or_else(|e| panic!("prefill_tokens at depth {depth}: {e:#}"));
        for (j, t) in ids[done..].iter().enumerate() {
            m.prefill_step(*t)
                .unwrap_or_else(|e| panic!("prefill tail step {}: {e:#}", done + j));
        }
        ctx.poll_blocking()
            .expect("drain prefill work before stopping the prime clock");
        let prime_s = prime_start.elapsed().as_secs_f64();
        assert_eq!(
            m.current_pos(),
            depth,
            "chunked prefill must land exactly at the requested depth"
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
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING gemma4-wgpu depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} prime_tok_s={:.1} prime=chunked chunk_done={done} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            depth as f64 / prime_s,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the 26B-A4B MoE on wgpu; set NV_GEMMA4_MOE_CTX_TEST=1 -- chunked-prefill tok/s plus decode ms/token (mean of 64) at each NV_CTX_TOKENS depth, synthetic repeat-token prompt"]
fn gemma4_moe_wgpu_chunked_prefill_tok_s_then_decode_ms_tok_vs_context_depth() {
    if std::env::var("NV_GEMMA4_MOE_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GEMMA4_MOE_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let home = std::env::var("HOME").expect("HOME");
    let dir = match std::env::var("NV_GEMMA4_MOE_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            let base = PathBuf::from(&home)
                .join(".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots");
            let mut c: Vec<PathBuf> = std::fs::read_dir(&base)
                .unwrap_or_else(|e| panic!("gemma4-moe snapshots dir {base:?} missing: {e}"))
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.join("config.json").is_file())
                .collect();
            c.sort();
            c.into_iter()
                .next()
                .expect("no gemma4-moe snapshot; set NV_GEMMA4_MOE_DIR")
        }
    };
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let cfg = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let max_seq = max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK;
    assert!(
        max_seq <= cfg.base.max_position_embeddings,
        "max_seq {max_seq} exceeds max_position_embeddings {}; drop the offending NV_CTX_TOKENS depth",
        cfg.base.max_position_embeddings
    );
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let t0 = std::time::Instant::now();
    let mut m = nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu::from_loader(cfg, &loader, max_seq)
        .expect("wgpu moe model");
    drop(loader);
    let wired_gib = m.load_report().wired_bytes as f64 / (1u64 << 30) as f64;
    eprintln!(
        "built in {:.1}s wired={wired_gib:.2} GiB | prefill chunk m={} passes/chunk={} decode passes/token={}",
        t0.elapsed().as_secs_f64(),
        m.prefill_chunk_len(),
        m.prefill_pass_count(),
        m.pass_count()
    );
    assert!(
        m.prefill_chunk_len() >= 2,
        "chunked prefill is off on this config; a step-primed prefill number would not be a prefill number"
    );

    for &depth in &depths {
        m.reset().expect("reset between depth arms");
        let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        let t0 = std::time::Instant::now();
        let done = m
            .prefill_tokens(&ids)
            .unwrap_or_else(|e| panic!("prefill_tokens at depth {depth}: {e:#}"));
        for (j, t) in ids[done..].iter().enumerate() {
            m.prefill_step(*t)
                .unwrap_or_else(|e| panic!("prefill tail step {}: {e:#}", done + j));
        }
        ctx.poll_blocking()
            .expect("drain prefill work before stopping the clock");
        let prefill_s = t0.elapsed().as_secs_f64();
        assert_eq!(
            m.current_pos(),
            depth,
            "prefill must land exactly at the requested depth"
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
        let mean = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-PREFILL gemma4-moe-wgpu depth={depth} prefill_s={prefill_s:.3} prefill_tok_s={:.1} chunk_done={done} decode_mean_ms_tok={mean:.3} decode_median_ms_tok={median:.3} steps={} warmup_steps={warmup_steps}",
            depth as f64 / prefill_s,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the 26B-A4B MoE on wgpu; set NV_G4MOE_CTX_CHUNKED_TEST=1 -- decode ms/token vs KV depth primed by chunked prefill (one submission per m-token chunk; step-priming deep rungs is infeasible); deep rungs also need NV_G4MOE_KV_RING=1 or the flat sliding caches (~42 GiB bf16 KV at 196k) OOM the card"]
fn g4moe_wgpu_decode_ms_per_token_vs_context_depth_chunked_prefill_primed() {
    if std::env::var("NV_G4MOE_CTX_CHUNKED_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_G4MOE_CTX_CHUNKED_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4MOE_SNAPSHOT",
        &["models--google--gemma-4-26B-A4B-it"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let config = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let max_seq = max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK;
    assert!(
        max_seq <= config.base.max_position_embeddings,
        "ladder needs {max_seq} positions but max_position_embeddings is {}",
        config.base.max_position_embeddings
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let t_build = std::time::Instant::now();
    let mut m = nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu::from_loader(config, &loader, max_seq)
        .expect("build from loader");
    drop(loader);
    let wired_gib = m.load_report().wired_bytes as f64 / (1u64 << 30) as f64;
    eprintln!(
        "built in {:.1}s wired={wired_gib:.2} GiB ring={} | prefill chunk m={} passes/chunk={} decode passes/token={}",
        t_build.elapsed().as_secs_f64(),
        nv_models::gemma4_moe_wgpu::sliding_kv_ring_enabled(),
        m.prefill_chunk_len(),
        m.prefill_pass_count(),
        m.pass_count()
    );
    assert!(
        m.prefill_chunk_len() >= 2,
        "chunked prefill is off on this config; priming would degrade to step priming and the deep rungs are infeasible again"
    );

    for &depth in &depths {
        m.reset().expect("reset between depth arms");
        let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        let prime_start = std::time::Instant::now();
        let done = m
            .prefill_tokens(&ids)
            .unwrap_or_else(|e| panic!("prefill_tokens at depth {depth}: {e:#}"));
        for (j, t) in ids[done..].iter().enumerate() {
            m.prefill_step(*t)
                .unwrap_or_else(|e| panic!("prefill tail step {}: {e:#}", done + j));
        }
        ctx.poll_blocking()
            .expect("drain prefill work before stopping the prime clock");
        let prime_s = prime_start.elapsed().as_secs_f64();
        assert_eq!(
            m.current_pos(),
            depth,
            "chunked prefill must land exactly at the requested depth"
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
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING g4moe-wgpu depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} prime_tok_s={:.1} prime=chunked chunk_done={done} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            depth as f64 / prime_s,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the qwen3.6-35B MoE on wgpu; set NV_QWEN36_WGPU_TEST=1 -- decode ms/token vs KV depth; GDN linear layers should keep this near-flat"]
fn qwen36_wgpu_decode_ms_per_token_vs_context_depth_step_primed() {
    if std::env::var("NV_QWEN36_WGPU_TEST").is_err() {
        eprintln!("skip: NV_QWEN36_WGPU_TEST not set");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let home = std::env::var("HOME").expect("HOME");
    let dir = std::env::var("NV_QWEN36_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(&home).join(
            ".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/e850c696e6d75f965367e816c16bc7dacd955ffa",
        )
    });
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let cfg = nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut gpu = nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::from_loader(
        cfg,
        &loader,
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK,
    )
    .expect("build from loader");
    drop(loader);

    for &depth in &depths {
        gpu.reset().expect("reset between depth arms");
        let prime_start = std::time::Instant::now();
        let mut token = 2000u32;
        for p in 0..depth {
            token = gpu
                .decode_step(2000 + (p as u32 % 30000))
                .unwrap_or_else(|e| panic!("prime step {p}: {e:#}"));
        }
        let prime_s = prime_start.elapsed().as_secs_f64();
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
            "CTX-SCALING qwen36-wgpu depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the qwen3.6-35B MoE on wgpu; set NV_QWEN36_CTX_PREFILL_TEST=1 -- chunked-prefill tok/s plus decode ms/token (mean of 64) at each NV_CTX_TOKENS depth, synthetic repeat-token prompt; NV_WGPU_PROFILE=1 prints the per-label prefill share table (M-row chunks only: under profiling prefill_tokens hands the per-token tail back to prefill_step, so pick depths divisible by the M-row chunk)"]
fn qwen36_wgpu_chunked_prefill_tok_s_then_decode_ms_tok_vs_context_depth() {
    if std::env::var("NV_QWEN36_CTX_PREFILL_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_QWEN36_CTX_PREFILL_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_QWEN36_DIR",
        &["models--RedHatAI--Qwen3.6-35B-A3B-NVFP4"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let cfg = nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut m = nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::from_loader(
        cfg,
        &loader,
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK,
    )
    .expect("build from loader");
    drop(loader);
    let (pf_dense, pf_moe) = m.prefill_mrow_pass_mix();
    eprintln!(
        "prefill mrow m={} passes/chunk={} mix=({pf_dense},{pf_moe}) legacy chunk m={}",
        m.prefill_mrow_chunk_len(),
        m.prefill_mrow_pass_count(),
        m.prefill_chunk_len()
    );
    assert!(
        m.prefill_mrow_chunk_len() >= 2,
        "the M-row prefill list is off on this config; a per-token-replay prefill number would not answer the M-row prefill question"
    );

    for &depth in &depths {
        m.reset().expect("reset between depth arms");
        nv_kernels::wgpu_backend::dispatch::profile::reset();
        let ids: Vec<u32> = (0..depth).map(|i| 2000 + (i as u32 % 30000)).collect();
        let t0 = std::time::Instant::now();
        let done = m
            .prefill_tokens(&ids)
            .unwrap_or_else(|e| panic!("prefill_tokens at depth {depth}: {e:#}"));
        for (j, t) in ids[done..].iter().enumerate() {
            m.prefill_step(*t)
                .unwrap_or_else(|e| panic!("prefill tail step {}: {e:#}", done + j));
        }
        ctx.poll_blocking()
            .expect("drain prefill work before stopping the clock");
        let prefill_s = t0.elapsed().as_secs_f64();
        assert_eq!(
            m.current_pos(),
            depth,
            "prefill must land exactly at the requested depth"
        );
        print_wgpu_profile_share_table_then_reset(&format!("qwen36-wgpu prefill depth={depth}"));

        let mut token = 2000u32;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = m
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let mean = step_ms.iter().sum::<f64>() / step_ms.len() as f64;
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-PREFILL qwen36-wgpu depth={depth} prefill_s={prefill_s:.3} prefill_tok_s={:.1} chunk_done={done} decode_mean_ms_tok={mean:.3} decode_median_ms_tok={median:.3} steps={} warmup_steps={warmup_steps}",
            depth as f64 / prefill_s,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the 31B on wgpu three times; set NV_G4_PF_COOP_REAL=1 and NV_PPL_CORPUS -- the \
            coop prefill arm's quality gate: the serving ppl instrument scores via decode_step \
            only and never runs the prefill graph, so this gate prefills the chat-wrapped \
            continuation prompt per arm (same wrapping as the shipping ppl suite; raw unwrapped \
            text scores 14.4 nats on this IT checkpoint and discriminates nothing) and \
            teacher-forces the 512-token continuation; a stepped-prompt control proves the \
            prefill->decode handoff, and a real prefill defect moves NLL by whole nats"]
fn gemma4_real_pf_coop_prefill_primed_teacher_forced_nll_tail_vs_mk_arm() {
    if std::env::var("NV_G4_PF_COOP_REAL").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_G4_PF_COOP_REAL != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4_SNAPSHOT",
        &["models--nvidia--Gemma-4-31B-IT-NVFP4"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let corpus_path = std::env::var("NV_PPL_CORPUS").expect("set NV_PPL_CORPUS");
    let corpus = std::fs::read_to_string(&corpus_path).expect("read NV_PPL_CORPUS");
    let tok = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let bos = tok.token_to_id("<bos>").expect("bos");
    let ctx_tokens = 256usize;
    let cont = 512usize;
    let all: Vec<u32> = tok
        .encode(corpus.as_str(), false)
        .expect("encode corpus")
        .get_ids()[..ctx_tokens + cont]
        .to_vec();
    let ctx_text = tok.decode(&all[..ctx_tokens], false).expect("decode context slice");
    let user = format!(
        "Continue the following text, staying in the same style, with no commentary:\n\n{ctx_text}"
    );
    let chat = official_template::OfficialTemplate::load(&dir).render_user(&user);
    let ids: Vec<u32> = tok
        .encode(chat.as_str(), false)
        .expect("tokenize chat")
        .get_ids()
        .to_vec();
    assert_eq!(
        ids.first().copied(),
        Some(bos),
        "the official render must begin with <bos>; hand-prepending it would double-count"
    );
    let mut ids = ids;
    let prompt = ids.len();
    ids.extend_from_slice(&all[ctx_tokens..]);
    let tail = cont;

    let run_arm = |coop: bool, stepped_prompt_control: bool| -> (f64, usize, usize) {
        if coop {
            std::env::set_var("NV_G4_PF_COOP", "1");
        } else {
            std::env::remove_var("NV_G4_PF_COOP");
        }
        let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
            .expect("config.json");
        let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
            .expect("weights");
        let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader)
            .expect("host staging");
        drop(loader);
        let mut m = nv_models::gemma4_wgpu::Gemma4Wgpu::new(config, &host, ids.len() + 8)
            .expect("wgpu model");
        drop(host);
        std::env::remove_var("NV_G4_PF_COOP");
        let sites = m.prefill_coop_ffn_gemm_sites();
        assert_eq!(
            sites > 0,
            coop,
            "coop-arm engagement must follow the env (coop={coop}, sites={sites}); \
             a decline reason was printed on stderr during build"
        );
        let done = if stepped_prompt_control {
            for t in &ids[..prompt - 1] {
                let _ = m.decode_step(*t).expect("stepped-control decode step");
            }
            0
        } else {
            let done = m.prefill_tokens(&ids[..prompt - 1]).expect("prefill_tokens");
            for t in &ids[done..prompt - 1] {
                m.prefill_step(*t).expect("prefill tail step");
            }
            done
        };
        assert_eq!(m.current_pos(), prompt - 1, "prompt must land at prompt-1");
        let mut nll = 0f64;
        let mut top1 = 0usize;
        for p in prompt - 1..prompt - 1 + tail {
            let (_next, row) = m.decode_step_logits(ids[p]).expect("teacher-forced step");
            let target = ids[p + 1] as usize;
            let maxv = row.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
            let sum: f64 = row.iter().map(|v| ((*v - maxv) as f64).exp()).sum();
            nll += -(((row[target] - maxv) as f64) - sum.ln());
            let am = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .unwrap();
            top1 += usize::from(am == target);
        }
        (nll / tail as f64, top1, done)
    };

    let (nll_step, top1_step, _) = run_arm(false, true);
    let (nll_base, top1_base, done_base) = run_arm(false, false);
    let (nll_coop, top1_coop, done_coop) = run_arm(true, false);
    let delta = (nll_coop - nll_base).abs();
    eprintln!(
        "PF-COOP-NLL gemma4-wgpu prompt={prompt} tail={tail} basis=prefill_primed_teacher_forced_decode_tail_full_vocab \
         nll_stepped_control={nll_step:.5} nll_base={nll_base:.5} nll_coop={nll_coop:.5} delta={delta:.5} \
         top1 stepped={top1_step} base={top1_base} coop={top1_coop} \
         chunk_done base={done_base} coop={done_coop}"
    );
    assert!(
        (nll_base - nll_step).abs() <= 0.05,
        "the mk prefill arm diverges from the stepped-prompt control by {:.4} nats; the \
         prefill->decode handoff itself is broken and no coop comparison on top of it is valid",
        (nll_base - nll_step).abs()
    );
    assert!(
        done_coop >= 16,
        "the coop arm never ran a chunked prefill; this gate would be vacuous"
    );
    assert!(
        nll_base.is_finite() && nll_coop.is_finite(),
        "non-finite NLL; the arms cannot be compared"
    );
    assert!(
        nll_base < 6.0,
        "chat-wrapped baseline NLL {nll_base:.3} is not a working language model; the \
         instrument has no discriminating power at this operating point (raw unwrapped text \
         scored 14.4 nats on this IT checkpoint -- worse than uniform)"
    );
    assert!(
        delta <= 0.15,
        "coop-arm prefill moved teacher-forced NLL by {delta:.4} nats (base {nll_base:.4}); \
         a stride/scale defect moves whole nats, quantization-format drift moves centinats"
    );
}

mod g4_pf_coop_tiny {
    use nv_models::gemma4::Gemma4Config;
    use nv_models::gemma4_wgpu::{quantize_nvfp4_host, Gemma4Wgpu};

    struct Lcg(u64);

    impl Lcg {
        fn next_f32(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (self.0 >> 32) as u32;
            (bits as f32 / u32::MAX as f32 - 0.5) * 0.2
        }
        fn bf16_vec(&mut self, n: usize) -> Vec<u16> {
            (0..n)
                .map(|_| half::bf16::from_f32(self.next_f32()).to_bits())
                .collect()
        }
        fn bf16_vec_around_one(&mut self, n: usize) -> Vec<u16> {
            (0..n)
                .map(|_| half::bf16::from_f32(1.0 + self.next_f32()).to_bits())
                .collect()
        }
    }

    fn tiny_config_all_ffn_shapes_multiples_of_64_so_the_coop_guard_accepts() -> Gemma4Config {
        let raw = r#"{
  "text_config": {
    "hidden_size": 512,
    "intermediate_size": 1024,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 128,
    "global_head_dim": 256,
    "vocab_size": 2048,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "sliding_window": 4096,
    "final_logit_softcapping": 0.0,
    "layer_types": ["sliding_attention", "sliding_attention", "full_attention", "sliding_attention"],
    "attention_k_eq_v": false,
    "hidden_activation": "gelu_pytorch_tanh",
    "num_kv_shared_layers": 0,
    "rope_parameters": {
      "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
      "sliding_attention": {"rope_theta": 10000.0}
    }
  },
  "tie_word_embeddings": true
}"#;
        Gemma4Config::from_hf_json_str(raw).expect("tiny config")
    }

    use crate::common::gemma4_wgpu_host_weights as host_weights;

    fn prefill_then_last_logits(m: &mut Gemma4Wgpu, ids: &[u32]) -> (u32, Vec<f32>, usize) {
        m.reset();
        let (last, rest) = ids.split_last().expect("prompt");
        let done = m.prefill_tokens(rest).expect("prefill_tokens");
        for t in &rest[done..] {
            m.prefill_step(*t).expect("prefill step");
        }
        let (next, logits) = m.decode_step_logits(*last).expect("last prompt token");
        (next, logits, done)
    }

    #[test]
    fn unswizzled_plain_scales_reproduce_the_swizzled_host_dequant_bit_exactly() {
        let mut rng = Lcg(0x51ca1e5);
        for (n, k) in [(128usize, 64usize), (512, 1024), (2048, 512)] {
            let w = rng.bf16_vec(n * k);
            let lin = quantize_nvfp4_host(&w, n, k);
            let deq = nv_models::gemma4_wgpu::dequantize_nvfp4_host(&lin);
            let plain = nv_models::gemma4_wgpu::unswizzle_nvfp4_scales_row_major_for_the_coop_kernels_plain_sf_index(&lin);
            let k_blocks = k / 16;
            const TABLE: [f32; 16] = [
                0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0,
                -6.0,
            ];
            let mut worst = 0usize;
            for r in 0..n {
                for kb in 0..k_blocks {
                    let sb = plain[r * k_blocks + kb] as u32;
                    let e = (sb >> 3) & 15;
                    let m = sb & 7;
                    let s = if e == 0 {
                        (m as f32) * 0.001953125f32
                    } else {
                        f32::from_bits(((e + 120) << 23) | (m << 20))
                    };
                    for j in 0..16 {
                        let idx = kb * 16 + j;
                        let byte = lin.packed[r * (k / 2) + idx / 2];
                        let nib = if idx % 2 == 0 { byte & 15 } else { byte >> 4 };
                        let v = TABLE[nib as usize] * s * lin.alpha;
                        if v.to_bits() != deq[r * k + idx].to_bits() {
                            worst += 1;
                        }
                    }
                }
            }
            assert_eq!(
                worst, 0,
                "{n}x{k}: plain-scale dequant diverges from the swizzled host dequant; \
                 the unswizzle index map is wrong"
            );
        }
    }

    #[test]
    fn gemma4_pf_coop_w4a16_arm_engages_and_tracks_the_mk_arm_and_leaves_decode_bit_identical() {
        if std::env::var("NV_G4_PF_COOP_TINY").ok().as_deref() != Some("1") {
            eprintln!("skip: NV_G4_PF_COOP_TINY != 1");
            return;
        }
        let _one_gpu_test_at_a_time = crate::ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
        let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
            .expect("wgpu adapter required: this gated suite must never silently skip");
        eprintln!("adapter: {}", ctx.summary());
        assert!(
            ctx.caps.coop_gemm_tile().is_some(),
            "no 16x16x16 f16 coop tile on this adapter; the coop arm cannot be gated here"
        );
        let config = tiny_config_all_ffn_shapes_multiples_of_64_so_the_coop_guard_accepts();
        let w = host_weights(&config, 0x9e3779b9);

        std::env::remove_var("NV_G4_PF_COOP");
        let mut base = Gemma4Wgpu::new(config.clone(), &w, 512).expect("baseline build");
        assert_eq!(
            base.prefill_coop_ffn_gemm_sites(),
            0,
            "baseline must not route any FFN GEMM through the coop arm"
        );
        let cm = base.prefill_chunk_len();
        assert!(cm >= 2, "chunked prefill off; the arm comparison would be vacuous");
        let pp = cm * 3 + 5;
        let ids: Vec<u32> = (0..pp).map(|i| ((i * 7919 + 13) % 2048) as u32).collect();
        let (next0, l0, done0) = prefill_then_last_logits(&mut base, &ids);
        base.reset();
        let (dtok0, dlog0) = base.decode_step_logits(7).expect("baseline fresh decode");

        std::env::set_var("NV_G4_WGPU_W8_FFN", "off");
        let mut base4 = Gemma4Wgpu::new(config.clone(), &w, 512).expect("nvfp4-ffn build");
        std::env::remove_var("NV_G4_WGPU_W8_FFN");
        let (next4, l4, _done4) = prefill_then_last_logits(&mut base4, &ids);
        drop(base4);

        std::env::set_var("NV_G4_PF_COOP", "1");
        let mut coop = Gemma4Wgpu::new(config, &w, 512).expect("coop-arm build");
        std::env::remove_var("NV_G4_PF_COOP");
        assert_eq!(
            coop.prefill_coop_ffn_gemm_sites(),
            8,
            "all 4 tiny layers x (gate_up, down) must route through the coop arm; \
             a partial count means the nvfp4 originals or shape guards silently declined"
        );
        let (next1, l1, done1) = prefill_then_last_logits(&mut coop, &ids);
        assert!(
            done0 >= cm && done1 >= cm,
            "both arms must actually run chunked prefill (done0={done0} done1={done1} cm={cm})"
        );
        coop.reset();
        let (dtok1, dlog1) = coop.decode_step_logits(7).expect("coop-arm fresh decode");
        assert_eq!(dtok0, dtok1, "fresh-state decode token must be identical across arms");
        let decode_diff = dlog0
            .iter()
            .zip(dlog1.iter())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
        assert_eq!(
            decode_diff, 0,
            "decode logits must stay BIT-identical: the coop arm may only touch the prefill list"
        );

        let max_abs = l0.iter().fold(0f32, |a, b| a.max(b.abs()));
        let delta = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max)
        };
        let d_coop_vs_int8 = delta(&l0, &l1);
        let d_coop_vs_nvfp4 = delta(&l4, &l1);
        let d_nvfp4_vs_int8 = delta(&l0, &l4);
        let distinct = l0.iter().map(|v| v.to_bits()).collect::<std::collections::HashSet<_>>().len();
        eprintln!(
            "[g4-pf-coop-tiny] pp={pp} cm={cm} next int8={next0} nvfp4={next4} coop={next1} \
             max_abs={max_abs:.3} d(coop,int8)={d_coop_vs_int8:.4} d(coop,nvfp4)={d_coop_vs_nvfp4:.4} \
             d(nvfp4,int8)={d_nvfp4_vs_int8:.4} distinct={distinct}"
        );
        assert!(
            distinct > 512,
            "degenerate logits ({distinct} distinct); the tolerance compare would be vacuous"
        );
        assert!(
            d_coop_vs_nvfp4 <= d_nvfp4_vs_int8.max(0.02 * max_abs.max(1.0)) * 2.0,
            "the coop arm shares the nvfp4-native arm's exact weight values, so its drift \
             ({d_coop_vs_nvfp4:.4}) must sit inside the codebase's own accepted \
             quant-format spread (nvfp4-vs-int8 arms differ by {d_nvfp4_vs_int8:.4}); a stride, \
             row-map or scale defect moves logits by whole units, not by an activation-precision \
             margin"
        );
    }
}

#[test]
#[ignore = "loads the ~13 GB gpt-oss-20b MXFP4 on wgpu; set NV_GPTOSS_CTX_TEST=1 -- decode ms/token vs KV depth, step-primed; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race: state can leak between depth arms sharing a process)"]
fn gptoss_wgpu_decode_ms_per_token_vs_context_depth_step_primed_max_pos_131072_caps_the_ladder_at_120k(
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
        let prime_start = std::time::Instant::now();
        let mut token = 2000u32;
        for p in 0..depth {
            token = gpu
                .decode_step(2000 + (p as u32 % 30000))
                .unwrap_or_else(|e| panic!("prime step {p}: {e:#}"));
        }
        let prime_s = prime_start.elapsed().as_secs_f64();
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
            "CTX-SCALING gptoss-wgpu depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads a real MatFormer E4B checkpoint on wgpu (W4A16 qat pack preferred, dense bf16 fallback -- the ckpt= field says which loaded); set NV_E4B_CTX_TEST=1 -- decode ms/token vs KV depth, step-primed; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn e4b_wgpu_decode_ms_per_token_vs_context_depth_step_primed_max_pos_131072_caps_the_ladder_at_120k(
) {
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
    let depths = ctx_tokens_from_env_default_256_8k_120k_because_max_pos_131072_has_no_room_for_168k();
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
        let prime_start = std::time::Instant::now();
        let mut token = 2000u32;
        for p in 0..depth {
            token = m
                .decode_step(2000 + (p as u32 % 30000))
                .unwrap_or_else(|e| panic!("prime step {p}: {e:#}"));
        }
        let prime_s = prime_start.elapsed().as_secs_f64();
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = m
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING e4b-wgpu depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={} warmup_steps={warmup_steps} ckpt={ckpt}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the real ~26B-A4B MoE on wgpu; set NV_G4MOE_CTX_TEST=1 -- decode ms/token vs KV depth (max_pos 262144 fits the full 256/8k/168k ladder), step-primed; run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn g4moe_wgpu_decode_ms_per_token_vs_context_depth_step_primed() {
    if std::env::var("NV_G4MOE_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_G4MOE_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4MOE_SNAPSHOT",
        &["models--google--gemma-4-26B-A4B-it"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let config = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let max_seq = max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK;
    assert!(
        max_seq <= config.base.max_position_embeddings,
        "ladder needs {max_seq} positions but max_position_embeddings is {}",
        config.base.max_position_embeddings
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut m = nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu::from_loader(config, &loader, max_seq)
        .expect("build from loader");
    drop(loader);

    for &depth in &depths {
        m.reset().expect("reset between depth arms");
        let prime_start = std::time::Instant::now();
        let mut token = 2000u32;
        for p in 0..depth {
            token = m
                .decode_step(2000 + (p as u32 % 30000))
                .unwrap_or_else(|e| panic!("prime step {p}: {e:#}"));
        }
        let prime_s = prime_start.elapsed().as_secs_f64();
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = m
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING g4moe-wgpu depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

fn synthetic_fp8_packed_words_every_byte_0x30_to_0x3e_finite_small_no_nan_under_e4m3(
    len_words: usize,
) -> Vec<u32> {
    let mut state = 0x9e3779b97f4a7c15u64;
    (0..len_words)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let b = |shift: u64| 0x30u32 + ((state >> shift) % 15) as u32;
            b(8) | (b(18) << 8) | (b(28) << 16) | (b(38) << 24)
        })
        .collect()
}

fn synthetic_kv_scales_small_positive_varied_so_dequant_stays_finite(len: usize) -> Vec<f32> {
    let mut state = 0x517cc1b727220a95u64;
    (0..len)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0.004f32 + ((state >> 40) % 1024) as f32 * 1.5e-5
        })
        .collect()
}

#[test]
#[ignore = "loads the 31B on wgpu; set NV_GEMMA4_CTX_TEST=1 -- same decode ms/token ladder but the fp8 KV buffers are filled synthetically via kv_cache_restore and the depth restored via restore_pos (decode reads cache SIZE through pos-derived uniforms; ring wrap state is a pure function of pos), so the deep rung costs seconds instead of hours; run under NV_G4_WGPU_KV_RING both on and off -- ring on is the arm that fits 168k"]
fn gemma4_wgpu_decode_ms_per_token_vs_context_depth_synthetic_cache_fill_ring_wrap_is_a_pure_function_of_pos(
) {
    if std::env::var("NV_GEMMA4_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_GEMMA4_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4_SNAPSHOT",
        &["models--nvidia--Gemma-4-31B-IT-NVFP4"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let ring = nv_models::gemma4_wgpu::sliding_kv_ring_enabled();
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let config = nv_models::gemma4::Gemma4Config::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let loader =
        nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu).expect("weights");
    let host = nv_models::gemma4_wgpu::host_weights_from_loader(&config, &loader)
        .expect("host weight staging");
    drop(loader);
    let mut m = nv_models::gemma4_wgpu::Gemma4Wgpu::new(
        config,
        &host,
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK,
    )
    .expect("wgpu model");
    drop(host);

    let mut max_words = 0usize;
    let mut max_scales = 0usize;
    for li in 0..m.kv_layer_count() {
        let lens = m.kv_layer_lens(li).expect("kv layer lens");
        max_words = max_words.max(lens[0]).max(lens[1]);
        max_scales = max_scales.max(lens[2]).max(lens[3]);
    }
    let word_template =
        synthetic_fp8_packed_words_every_byte_0x30_to_0x3e_finite_small_no_nan_under_e4m3(
            max_words,
        );
    let scale_template =
        synthetic_kv_scales_small_positive_varied_so_dequant_stays_finite(max_scales);

    for &depth in &depths {
        m.reset();
        let fill_start = std::time::Instant::now();
        for li in 0..m.kv_layer_count() {
            let lens = m.kv_layer_lens(li).expect("kv layer lens");
            let snap = (
                word_template[..lens[0]].to_vec(),
                word_template[..lens[1]].to_vec(),
                scale_template[..lens[2]].to_vec(),
                scale_template[..lens[3]].to_vec(),
            );
            let wrote = m
                .kv_cache_restore(li, &snap)
                .unwrap_or_else(|e| panic!("synthetic kv restore at layer {li}: {e:#}"));
            assert!(wrote, "layer {li} has no kv buffers to fill");
        }
        m.restore_pos(depth).expect("restore_pos to synthetic depth");
        ctx.poll_blocking()
            .expect("drain synthetic fill writes before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(m.current_pos(), depth, "synthetic fill must land at the requested depth");

        let mut token = 2000u32;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = m
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING gemma4-wgpu-synthfill depth={depth} median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} ring={ring} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the ~26B-A4B MoE on wgpu; set NV_G4MOE_CTX_TEST=1 -- same decode ms/token ladder but every state buffer is filled with finite synthetic bytes and pos set directly (window start and ring wrap are pure functions of pos; MoE routing is only weakly value-dependent and the byte pattern keeps it non-degenerate), so the deep rung costs seconds; deep rungs still need NV_G4MOE_KV_RING=1 or the flat sliding caches OOM the card"]
fn g4moe_wgpu_decode_ms_per_token_vs_context_depth_synthetic_state_fill_pos_is_the_only_depth_state(
) {
    if std::env::var("NV_G4MOE_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_G4MOE_CTX_TEST != 1");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4MOE_SNAPSHOT",
        &["models--google--gemma-4-26B-A4B-it"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let config = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let max_seq = max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK;
    assert!(
        max_seq <= config.base.max_position_embeddings,
        "ladder needs {max_seq} positions but max_position_embeddings is {}",
        config.base.max_position_embeddings
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut m = nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu::from_loader(config, &loader, max_seq)
        .expect("build from loader");
    drop(loader);
    let ring = nv_models::gemma4_moe_wgpu::sliding_kv_ring_enabled();
    let flash = nv_models::gemma4_moe_wgpu::flash_decode_enabled();

    for &depth in &depths {
        m.reset().expect("reset between depth arms");
        let fill_start = std::time::Instant::now();
        m.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
            .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
        ctx.poll_blocking()
            .expect("drain synthetic fill writes before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(m.current_pos(), depth, "synthetic fill must land at the requested depth");

        let mut token = 2000u32;
        let (warmup_steps, mut step_ms) = ctx_timing_common::warmup_to_plateau_then_time_steps(
            || {
                token = m
                    .decode_step(token)
                    .unwrap_or_else(|e| panic!("timed step at depth {depth}: {e:#}"));
            },
            TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN,
        );
        let median = median_ms_of_timed_steps(&mut step_ms);
        eprintln!(
            "CTX-SCALING g4moe-wgpu-synthfill depth={depth} median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} ring={ring} flash={flash} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
        print_wgpu_profile_share_table_then_reset(&format!("g4moe-synthfill-{depth}"));
    }
}

#[test]
#[ignore = "loads the ~26B-A4B MoE on wgpu; set NV_G4MOE_CTX_TEST=1 AND NV_G4MOE_KIND_LABELS=1 -- prices the full-attention vs sliding share of the decode token by replicating the kind-labeled attention passes once per timed step over a synthetic state fill; deep rungs still need NV_G4MOE_KV_RING=1 or the flat sliding caches OOM the card; NV_G4MOE_FLASH_DECODE picks which full-attention arm gets priced"]
fn g4moe_wgpu_full_vs_sliding_decode_share_priced_by_replicating_kind_labeled_passes() {
    if std::env::var("NV_G4MOE_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_G4MOE_CTX_TEST != 1");
        return;
    }
    assert!(
        nv_models::gemma4_moe_wgpu::layer_kind_decode_labels_enabled(),
        "set NV_G4MOE_KIND_LABELS=1: the -full/-sliding suffix is baked into the pass labels \
         at build time, so this probe cannot see layer kinds without it"
    );
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_G4MOE_SNAPSHOT",
        &["models--google--gemma-4-26B-A4B-it"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let config = nv_models::gemma4_moe::Gemma4MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config.json");
    let max_seq = max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK;
    assert!(
        max_seq <= config.base.max_position_embeddings,
        "ladder needs {max_seq} positions but max_position_embeddings is {}",
        config.base.max_position_embeddings
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut m = nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu::from_loader(config, &loader, max_seq)
        .expect("build from loader");
    drop(loader);
    let ring = nv_models::gemma4_moe_wgpu::sliding_kv_ring_enabled();
    let flash = nv_models::gemma4_moe_wgpu::flash_decode_enabled();

    let mut classes: Vec<(String, usize)> = Vec::new();
    for (label, _, _, _) in m.pass_rows() {
        if label.starts_with("g4m-at-decode") || label.starts_with("g4m-at-flash") {
            match classes.iter_mut().find(|(l, _)| *l == label) {
                Some((_, n)) => *n += 1,
                None => classes.push((label, 1)),
            }
        }
    }
    assert!(
        classes.iter().any(|(l, _)| l.ends_with("-full"))
            && classes.iter().any(|(l, _)| l.ends_with("-sliding")),
        "no kind-suffixed attention classes in the recorded passes: {classes:?}"
    );

    const KIND_PROBE_WARMUP_STEPS: usize = 8;
    const KIND_PROBE_PAIRED_REPS: usize = 5;
    for &depth in &depths {
        m.reset().expect("reset between depth arms");
        m.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
            .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
        ctx.poll_blocking()
            .expect("drain synthetic fill writes before timing");
        let mut token = 2000u32;
        for _ in 0..KIND_PROBE_WARMUP_STEPS {
            token = m.decode_step(token).expect("warm decode");
        }
        for (class, n) in &classes {
            let mut diffs: Vec<f64> = Vec::with_capacity(KIND_PROBE_PAIRED_REPS);
            let mut bases: Vec<f64> = Vec::with_capacity(KIND_PROBE_PAIRED_REPS);
            for _ in 0..KIND_PROBE_PAIRED_REPS {
                let t0 = std::time::Instant::now();
                token = m
                    .decode_step_replicated(token, None, 0)
                    .expect("paired baseline step");
                let base_ms = t0.elapsed().as_secs_f64() * 1e3;
                let t1 = std::time::Instant::now();
                token = m
                    .decode_step_replicated(token, Some(class.as_str()), 1)
                    .expect("replicated step");
                let plus_ms = t1.elapsed().as_secs_f64() * 1e3;
                bases.push(base_ms);
                diffs.push(plus_ms - base_ms);
            }
            let base = median_ms_of_timed_steps(&mut bases);
            let share = median_ms_of_timed_steps(&mut diffs);
            eprintln!(
                "KIND-SHARE g4moe-wgpu depth={depth} class={class} n={n} base_ms_tok={base:.3} class_ms_tok={share:.3} ({:.1}% of token) ring={ring} flash={flash}",
                share / base * 100.0
            );
        }
    }
}

#[test]
#[ignore = "loads the qwen3.6-35B MoE on wgpu; set NV_QWEN36_WGPU_TEST=1 -- same decode ms/token ladder but every state buffer is filled with finite synthetic bytes and pos set directly (attention window is a pure function of pos, gdn state has no depth dimension, and the byte pattern keeps MoE routing non-degenerate), so deep rungs cost seconds instead of step-priming"]
fn qwen36_wgpu_decode_ms_per_token_vs_context_depth_synthetic_state_fill_pos_is_the_only_depth_state(
) {
    if std::env::var("NV_QWEN36_WGPU_TEST").is_err() {
        eprintln!("skip: NV_QWEN36_WGPU_TEST not set");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_QWEN36_DIR",
        &["models--RedHatAI--Qwen3.6-35B-A3B-NVFP4"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths = ctx_tokens_from_env_default_256_8k_168k();
    let max_depth = depths.iter().copied().max().unwrap();

    let cfg = nv_models::qwen3_5_moe::Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut gpu = nv_models::qwen3_5_moe_wgpu::Qwen3MoeWgpu::from_loader(
        cfg,
        &loader,
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK,
    )
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
        assert_eq!(gpu.current_pos(), depth, "synthetic fill must land at the requested depth");

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
            "CTX-SCALING qwen36-wgpu-synthfill depth={depth} median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
    }
}

#[test]
#[ignore = "loads the ~22.6 GB Qwen3.8-27B dense hybrid on wgpu; set NV_QWEN38_WGPU_TEST=1 -- decode ms/token vs KV depth on the 262144-max-pos 256/8k/196k ladder; state buffers are filled with finite synthetic bytes and pos set directly (16 full-attn layers read a window that is a pure function of pos, 48 gdn layers carry no depth dimension); run ONE depth per process via NV_CTX_TOKENS (the #96 residual race)"]
fn qwen38_wgpu_decode_ms_per_token_vs_context_depth_synthetic_state_fill_pos_is_the_only_depth_state(
) {
    if std::env::var("NV_QWEN38_WGPU_TEST").is_err() {
        eprintln!("skip: NV_QWEN38_WGPU_TEST not set");
        return;
    }
    let _one_gpu_test_at_a_time = ctx_timing_common::serialize_tests_sharing_the_gpu_because_libtest_threads_double_book_the_card_reading_2x_slow_and_a_sibling_engines_teardown_surfaces_a_one_shot_cuda_error_invalid_value();
    let dir = snapshot_dir_env_override_then_home_hub_first_repo_with_config_and_weights(
        "NV_QWEN38_DIR",
        &["models--unsloth--Qwen3.8-27B-NVFP4"],
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());
    let depths =
        ctx_tokens_from_env_default_256_8k_196k_the_task_106_gate_ladder_for_262144_max_pos_models();
    let max_depth = depths.iter().copied().max().unwrap();

    let cfg = nv_models::qwen3_5_moe::Qwen3_5DenseConfig::from_hf_json_file(&dir.join("config.json"))
        .expect("config");
    assert!(
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK
            < cfg.max_position_embeddings,
        "a model declaring max_pos {} must serve depth {max_depth}, so a trip here is a config bug",
        cfg.max_position_embeddings
    );
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut gpu = nv_models::qwen3_5_dense_wgpu::Qwen3_5DenseWgpu::from_loader(
        cfg,
        &loader,
        max_depth + DECODE_SLOTS_BEYOND_DEPTH_WORST_CASE_WARMUP_PLUS_TIMED_PLUS_16_SLACK,
    )
    .expect(
        "build Qwen3.8-27B on the qwen3_5 dense wgpu decoder; a trip here on the real unsloth \
         checkpoint is the track-1 NVFP4 format gap (mixed fp8 attn + nvfp4 mlp), not a ladder bug",
    );
    drop(loader);

    for &depth in &depths {
        gpu.reset().expect("reset between depth arms");
        let fill_start = std::time::Instant::now();
        gpu.fill_state_buffers_with_finite_synthetic_bytes_and_set_pos_for_depth_timing_decode_reads_kv_size_not_values(depth)
            .unwrap_or_else(|e| panic!("synthetic state fill at depth {depth}: {e:#}"));
        ctx.poll_blocking()
            .expect("drain synthetic fill writes before stopping the fill clock");
        let fill_s = fill_start.elapsed().as_secs_f64();
        assert_eq!(gpu.current_pos(), depth, "synthetic fill must land at the requested depth");

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
            "CTX-SCALING qwen38-wgpu-synthfill depth={depth} basis=eager_dense_wgpu_decode_step_synthetic_state_fill median_ms_tok={median:.3} tok_s={:.1} fill_s={fill_s:.1} steps={} warmup_steps={warmup_steps}",
            1000.0 / median,
            step_ms.len()
        );
        print_wgpu_profile_share_table_then_reset(&format!("qwen38-wgpu decode depth={depth}"));
    }
}
