#![cfg(feature = "wgpu")]

use nv_models::qwen3_5_moe_wgpu as q3w;
use nv_models::qwen3_5_moe::Qwen3MoeConfig;

const STEPS_64_MATCHES_THE_GPTOSS_DECOMPOSITION_BASIS: usize = 64;

#[test]
#[ignore = "loads the ~22 GB Qwen3.6-35B NVFP4 on wgpu; set NV_QWEN36_WGPU_TEST=1 and \
            NV_WGPU_PROFILE=1 -- decode-step per-kernel share table for cross-box \
            triangulation (the Metal profiler kills that adapter, so the Vulkan ranking is \
            the only attribution available for the shared graph); read-only companion to the \
            fenced qwen36 suites, prints shares via the public dispatch::profile API"]
fn qwen36_wgpu_decode_step_per_kernel_share_table_for_cross_box_triangulation() {
    if std::env::var("NV_QWEN36_WGPU_TEST").is_err() {
        eprintln!("[skip] NV_QWEN36_WGPU_TEST not set");
        return;
    }
    let dir = std::path::PathBuf::from(std::env::var("NV_QWEN36_DIR").unwrap_or_else(|_| {
        format!(
            "{}/.cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots/965bfb0e24d08e295cd641a15c7f231554078d0d",
            std::env::var("HOME").unwrap_or_default()
        )
    }));
    let cfg = Qwen3MoeConfig::from_hf_json_file(&dir.join("config.json")).expect("config");
    let loader = nv_weights::WeightLoader::open_dir(&dir, &candle_core::Device::Cpu)
        .expect("open safetensors");
    let max_seq = 512usize;
    let mut gpu =
        q3w::Qwen3MoeWgpu::from_loader(cfg, &loader, max_seq).expect("build from loader");
    eprintln!("[q36-prof] built, {} passes/token", gpu.pass_count());

    let tok =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer");
    let prompt = tok
        .encode("The capital of France is", false)
        .expect("encode")
        .get_ids()
        .to_vec();
    let mut last = *prompt.first().expect("nonempty prompt");
    for &t in &prompt {
        last = gpu.decode_step(t).expect("prime step");
    }

    {
        use nv_kernels::wgpu_backend::dispatch::profile;
        if profile::enabled() {
            let _ = profile::report();
        }
    }

    let t0 = std::time::Instant::now();
    for _ in 0..STEPS_64_MATCHES_THE_GPTOSS_DECOMPOSITION_BASIS {
        last = gpu.decode_step(last).expect("decode step");
    }
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[q36-prof] {} steps, {:.2} ms/tok wall (profiled -- shares are the signal, absolutes \
         inflated)",
        STEPS_64_MATCHES_THE_GPTOSS_DECOMPOSITION_BASIS,
        1e3 * wall / STEPS_64_MATCHES_THE_GPTOSS_DECOMPOSITION_BASIS as f64
    );

    use nv_kernels::wgpu_backend::dispatch::profile;
    if !profile::enabled() {
        panic!(
            "NV_WGPU_PROFILE is not 1, so no share table exists: this probe's only output is \
             the attribution table and running it unprofiled would print a vacuous green"
        );
    }
    let rows = profile::report();
    let total: f64 = rows.iter().map(|r| r.2).sum();
    let n: u64 = rows.iter().map(|r| r.1).sum();
    eprintln!(
        "[q36-prof] table: {} labels {} dispatches {:.1} ms GPU over {} steps",
        rows.len(),
        n,
        total,
        STEPS_64_MATCHES_THE_GPTOSS_DECOMPOSITION_BASIS
    );
    let mut sorted = rows.clone();
    sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    for (label, count, ms) in &sorted {
        eprintln!(
            "[q36-prof] | {label} n={count} total_ms={ms:.1} share={:.1}%",
            100.0 * ms / total
        );
    }
}
