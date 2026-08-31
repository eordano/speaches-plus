#![cfg(feature = "wgpu")]

use candle_core::Device;
use nv_models::gemma4_moe::Gemma4MoeConfig;
use nv_models::gemma4_moe_wgpu::Gemma4MoeWgpu;
use nv_weights::WeightLoader;

const STEPS_64_MATCHES_THE_FLEET_DECOMPOSITION_BASIS: usize = 64;

#[test]
#[ignore = "loads the 26B-A4B MoE on wgpu; set NV_GEMMA4_MOE_WGPU_TEST=1 and NV_WGPU_PROFILE=1 \
            -- decode-step per-kernel share table for the 2.7x same-API frontier (the \
            g4moe ctx instrument does not print shares); read-only companion probe, prints \
            via the public dispatch::profile API; prime suspect from the kernel source: the \
            w4e expert GEMV reduces each output row through a 256-thread workgroup-shared \
            tree with barriers, the reduction-cost class the gptoss campaign convicted"]
fn g4moe_wgpu_decode_step_per_kernel_share_table_for_the_grouped_dispatch_frontier() {
    if std::env::var("NV_GEMMA4_MOE_WGPU_TEST").is_err() {
        eprintln!("[skip] NV_GEMMA4_MOE_WGPU_TEST not set");
        return;
    }
    let snap = match std::env::var("NV_GEMMA4_MOE_DIR") {
        Ok(d) => std::path::PathBuf::from(d),
        Err(_) => {
            let root = std::path::PathBuf::from(std::env::var("HOME").unwrap())
                .join(".cache/huggingface/hub/models--google--gemma-4-26B-A4B-it/snapshots");
            std::fs::read_dir(&root)
                .unwrap_or_else(|e| panic!("no snapshot dir at {}: {e}", root.display()))
                .filter_map(|d| d.ok())
                .map(|d| d.path())
                .find(|p| p.join("config.json").exists())
                .expect("no snapshot with config.json")
        }
    };
    let cfg = Gemma4MoeConfig::from_hf_json_file(&snap.join("config.json")).expect("config");
    let loader = WeightLoader::open_dir(&snap, &Device::Cpu).expect("open safetensors");
    let mut gpu = Gemma4MoeWgpu::from_loader(cfg, &loader, 512).expect("build");
    eprintln!("[g4moe-prof] built, {} passes/token", gpu.pass_count());

    let tok =
        tokenizers::Tokenizer::from_file(snap.join("tokenizer.json")).expect("tokenizer");
    let mut ids: Vec<u32> = tok
        .encode("The capital of France is", false)
        .expect("encode")
        .get_ids()
        .to_vec();
    ids.insert(0, 2);
    let mut last = *ids.first().expect("nonempty prompt");
    for &t in &ids {
        last = gpu.decode_step(t).expect("prime step");
    }

    {
        use nv_kernels::wgpu_backend::dispatch::profile;
        if profile::enabled() {
            let _ = profile::report();
        }
    }

    let t0 = std::time::Instant::now();
    for _ in 0..STEPS_64_MATCHES_THE_FLEET_DECOMPOSITION_BASIS {
        last = gpu.decode_step(last).expect("decode step");
    }
    let wall = t0.elapsed().as_secs_f64();
    eprintln!(
        "[g4moe-prof] {} steps, {:.2} ms/tok wall (profiled -- shares are the signal, \
         absolutes inflated)",
        STEPS_64_MATCHES_THE_FLEET_DECOMPOSITION_BASIS,
        1e3 * wall / STEPS_64_MATCHES_THE_FLEET_DECOMPOSITION_BASIS as f64
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
        "[g4moe-prof] table: {} labels {} dispatches {:.1} raw-units over {} steps",
        rows.len(),
        n,
        total,
        STEPS_64_MATCHES_THE_FLEET_DECOMPOSITION_BASIS
    );
    let mut sorted = rows.clone();
    sorted.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    for (label, count, ms) in &sorted {
        eprintln!(
            "[g4moe-prof] | {label} n={count} share={:.1}%",
            100.0 * ms / total
        );
    }
}
