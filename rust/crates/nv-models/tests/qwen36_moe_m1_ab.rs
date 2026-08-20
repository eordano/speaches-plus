#![cfg(feature = "cuda")]

mod common;
use common::argmax;
use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;
use std::time::Instant;

fn snapshot() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/snapshots");
    std::fs::read_dir(base)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.join("config.json").is_file())
}

#[test]
#[ignore = "loads the ~20 GB Qwen3.6 checkpoint; set NV_QWEN36_M1_AB=1"]
fn qwen36_decode_m1_ab_arm() {
    if std::env::var("NV_QWEN36_M1_AB").as_deref() != Ok("1") {
        panic!("set NV_QWEN36_M1_AB=1 to run (it must never silently skip)");
    }
    let gemv_env = std::env::var("NV_MOE_FP4_DECODE_GEMV").unwrap_or_default();
    let dir = snapshot().expect("qwen snapshot");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    let mut eng = GraphedQwen3Moe::new(model, &device, 512).expect("engine");
    eng.install_grouped_moe().expect("grouped");
    eng.set_device_routing(true);
    eng.reset().expect("reset");

    let prompt: Vec<u32> = vec![151644u32, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30];
    let steps: usize = std::env::var("NV_QWEN36_M1_AB_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let warm: usize = 16;

    let row = eng.prefill(&prompt).expect("prefill");
    let mut tok = argmax(&row);
    let mut toks: Vec<u32> = vec![tok];
    let mut times: Vec<f64> = Vec::with_capacity(steps);
    for i in 0..steps {
        let t0 = Instant::now();
        eng.forward_decode(tok)
            .unwrap_or_else(|e| panic!("graphed decode step {i} failed: {e:#}"));
        let r = eng
            .logits_host()
            .unwrap_or_else(|e| panic!("logits step {i}: {e:#}"));
        times.push(t0.elapsed().as_secs_f64() * 1e3);
        tok = argmax(&r);
        toks.push(tok);
    }
    assert!(
        eng.capture_active(),
        "graphed decode did not actually capture — arm did not exercise the graph path"
    );

    let pre_nan = row.iter().filter(|v| v.is_nan()).count();
    assert_eq!(
        pre_nan, 0,
        "prefill produced {pre_nan}/{} NaN logits, so every ms/tok below would be timing a decode \
         that computed nothing. See #63: grouped MoE on a non-default stream.",
        row.len()
    );
    let distinct = toks.iter().collect::<std::collections::HashSet<_>>().len();
    assert!(
        distinct > 1,
        "greedy decode emitted a single repeated token {:?} over {} steps -- degenerate output, so \
         the timings describe a broken path",
        toks.first(),
        toks.len()
    );

    let mut steady: Vec<f64> = times[warm..].to_vec();
    steady.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = steady[steady.len() / 2];
    let mean = steady.iter().sum::<f64>() / steady.len() as f64;
    let warm_mean = times[..warm].iter().sum::<f64>() / warm as f64;
    eprintln!(
        "AB basis: RedHatAI/Qwen3.6-35B-A3B-NVFP4 {dir:?}, graphed decode + logits_host D2H + host argmax per step, bs=1 greedy, prompt=10 toks, steps={steps} (warmup {warm} discarded), NV_MOE_FP4_DECODE_GEMV={gemv_env:?}"
    );
    eprintln!(
        "AB arm gemv_env={gemv_env:?}: steady median={med:.3} ms/tok mean={mean:.3} ms/tok warmup_mean={warm_mean:.3} ms/tok"
    );
    eprintln!(
        "AB tokens gemv_env={gemv_env:?}: {}",
        toks.iter()
            .map(|t| t.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
}
