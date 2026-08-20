#![cfg(feature = "cuda")]

use candle_core::Device;
use nv_models::graph_engine::GraphedQwen3Moe;
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3MoeConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

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

fn stats(row: &[f32]) -> (usize, usize, u32, f32) {
    let nan = row.iter().filter(|v| v.is_nan()).count();
    let inf = row.iter().filter(|v| v.is_infinite()).count();
    let (mut bi, mut bv) = (0u32, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > bv {
            bv = v;
            bi = i as u32;
        }
    }
    (nan, inf, bi, bv)
}

#[test]
#[ignore = "loads the ~20 GB Qwen3.6 checkpoint; set NV_QWEN_GRAPH_BISECT=1"]
fn reset_then_prefill_is_clean_after_graphed_decode() {
    if std::env::var("NV_QWEN_GRAPH_BISECT").as_deref() != Ok("1") {
        panic!("set NV_QWEN_GRAPH_BISECT=1 to run (it must never silently skip)");
    }
    let dir = snapshot().expect("qwen snapshot");
    let device = Device::new_cuda_with_stream(0).expect("cuda");
    let raw = std::fs::read_to_string(dir.join("config.json")).expect("config");
    let cfg = Qwen3MoeConfig::from_hf_json_str(&raw).expect("config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw).expect("qcfg");
    let weights = WeightLoader::open_dir(&dir, &device).expect("weights");
    let model = Qwen3Moe::from_loader_quantized(cfg, &weights, &qcfg, &device).expect("model");
    let mut eng = GraphedQwen3Moe::new(model, &device, 512).expect("engine");
    eng.install_grouped_moe().expect("grouped");

    let prompt: Vec<u32> = vec![151644u32, 872, 198, 3838, 374, 279, 6722, 315, 9625, 30];
    let steps = 8usize;

    for (arm, use_graph) in [("A:eager-decode", false), ("B:graphed-decode", true)] {
        eng.set_device_routing(use_graph);
        eng.reset().expect("pre-arm reset");
        let row1 = eng.prefill(&prompt).expect("prefill 1");
        let (nan1, inf1, argmax1, v1) = stats(&row1);
        let mut tok = argmax1;
        for i in 0..steps {
            let r = if use_graph {
                eng.forward_decode(tok)
                    .unwrap_or_else(|e| panic!("{arm}: graphed decode step {i} failed: {e:#}"));
                eng.logits_host()
                    .unwrap_or_else(|e| panic!("{arm}: logits step {i}: {e:#}"))
            } else {
                eng.forward_decode_logits_vec(tok)
                    .unwrap_or_else(|e| panic!("{arm}: eager step {i}: {e:#}"))
            };
            let (nan, inf, top, v) = stats(&r);
            eprintln!("[bisect] {arm}: step {i} in={tok} nan={nan} inf={inf} argmax={top}@{v:.3}");
            tok = top;
        }
        let captured = eng.capture_active();
        assert_eq!(
            captured, use_graph,
            "{arm}: capture_active={captured} — the arm did not exercise the path it claims"
        );
        eng.reset().expect("mid-arm reset");
        let row2 = eng.prefill(&prompt).expect("prefill 2");
        let (nan2, inf2, argmax2, v2) = stats(&row2);
        eprintln!(
            "[bisect] {arm}: captured_during_decode={captured} \
             prefill1 nan={nan1} inf={inf1} argmax={argmax1}@{v1:.3} | \
             prefill2 nan={nan2} inf={inf2} argmax={argmax2}@{v2:.3} match={}",
            argmax1 == argmax2
        );
        assert_eq!(nan2, 0, "{arm}: prefill AFTER reset has {nan2} NaN logits");
        assert_eq!(
            argmax1, argmax2,
            "{arm}: same prompt after reset changed the prefill argmax"
        );
    }
}
