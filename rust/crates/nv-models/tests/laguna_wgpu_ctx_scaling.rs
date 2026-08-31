#![cfg(feature = "laguna-wip")]

mod ctx_timing_common;
mod hub_snapshot;

use std::path::PathBuf;

use nv_models::laguna::LagunaConfig;
use nv_models::laguna_wgpu::LagunaWgpu;

const LAGUNA_REPO: &str = "poolside/Laguna-XS-2.1-NVFP4";

fn ckpt_dir() -> Option<PathBuf> {
    hub_snapshot::dir_from_env_or_hub(
        "NV_LAGUNA_DIR",
        LAGUNA_REPO,
        &["config.json", "*.safetensors"],
    )
}

const WARMUP_STEPS: u32 = 8;
const TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN: usize = 64;
const DECODE_SLOTS_BEYOND_DEPTH_WARMUP_PLUS_TIMED_PLUS_16_SLACK: usize =
    WARMUP_STEPS as usize + TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN + 16;

#[test]
#[ignore = "loads the ~21 GiB Laguna-XS-2.1-NVFP4 on wgpu; set NV_LAGUNA_WGPU_CTX_TEST=1 -- decode ms/token vs KV depth, primed by stepping (the wgpu port has plain M=1 decode only); run ONE depth per process via NV_CTX_TOKENS (the #96 residual race); the 3:1 sliding(512)/full layer mix should keep the depth slope shallow next to an all-full-attention model"]
fn laguna_wgpu_decode_ms_per_token_vs_context_depth_step_primed() {
    if std::env::var("NV_LAGUNA_WGPU_CTX_TEST").ok().as_deref() != Some("1") {
        eprintln!("skip: NV_LAGUNA_WGPU_CTX_TEST != 1");
        return;
    }
    let dir = ckpt_dir().expect(
        "no Laguna-XS-2.1-NVFP4 snapshot; set NV_LAGUNA_DIR -- this gate must never silently skip",
    );
    let ctx = nv_kernels::wgpu_backend::WgpuContext::shared()
        .expect("wgpu adapter required: this gated suite must never silently skip");
    eprintln!("adapter: {}", ctx.summary());

    let depth: usize = std::env::var("NV_CTX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let max_seq = depth + DECODE_SLOTS_BEYOND_DEPTH_WARMUP_PLUS_TIMED_PLUS_16_SLACK;

    let cfg = LagunaConfig::from_hf_json_file(&dir.join("config.json")).expect("config.json");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let mut m = LagunaWgpu::from_loader(cfg, &loader, max_seq).expect("build from loader");
    drop(loader);

    let prime_start = std::time::Instant::now();
    let mut token = 2000u32;
    for p in 0..depth {
        token = m
            .decode_step(2000 + (p as u32 % 30000))
            .unwrap_or_else(|e| panic!("prime step {p}: {e:#}"));
    }
    let prime_s = prime_start.elapsed().as_secs_f64();

    for _ in 0..WARMUP_STEPS {
        token = m.decode_step(token).expect("warmup step");
    }
    let mut step_ms: Vec<f64> = Vec::with_capacity(TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN);
    for _ in 0..TIMED_DECODE_STEPS_64_ENOUGH_FOR_A_STABLE_MEDIAN {
        let t0 = std::time::Instant::now();
        token = m.decode_step(token).expect("timed step");
        step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    step_ms.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
    let median = step_ms[step_ms.len() / 2];
    eprintln!(
        "CTX-SCALING laguna-wgpu depth={depth} median_ms_tok={median:.3} tok_s={:.1} prime_s={prime_s:.1} steps={} warmup_steps={WARMUP_STEPS}",
        1000.0 / median,
        step_ms.len()
    );
    ctx_timing_common::emit_nv_measure_v1_alongside_the_legacy_line_which_stays_for_one_release(
        ctx_timing_common::NvMeasureCtxLine {
            instrument: "ctx-scaling",
            model_dir: &dir,
            backend: "wgpu",
            device: ctx_timing_common::wgpu_device_name_for_the_nv_measure_device_basis_field(),
            tokens_primed_kv_depth_of_this_run_never_a_capacity: depth,
            steps: step_ms.len(),
            warmup: WARMUP_STEPS as usize,
            value: median,
            unit: "ms/tok",
            basis: "eager_wgpu_decode_step_step_primed",
        },
        &[("prime_s", format!("{prime_s:.1}"))],
    );
}
