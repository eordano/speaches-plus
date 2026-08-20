#![cfg(feature = "cuda")]

mod common;
use common::write_tiny_model;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_models::gemma4_graph::GraphedGemma4Decoder;
use nv_weights::WeightLoader;
use std::collections::HashMap;

fn load(device: &Device) -> Gemma4 {
    static PROBE_FIXTURE: std::sync::OnceLock<std::path::PathBuf> =
        std::sync::OnceLock::new();
    let dir = PROBE_FIXTURE
        .get_or_init(|| {
            let d = std::env::temp_dir()
                .join(format!("{}-{}", "nv-gemma4-graph-logits-tiny", std::process::id()));
            write_tiny_model(&d);
            d
        })
        .clone();
    let cfg =
        Gemma4Config::from_hf_json_str(&std::fs::read_to_string(dir.join("config.json")).unwrap())
            .unwrap();
    let weights = WeightLoader::open_dir(&dir, device).unwrap();
    Gemma4::from_loader(cfg, &weights, device).unwrap()
}

fn gpu_graph_gated() -> bool {
    std::env::var("NV_GPU_GRAPH_TEST").as_deref() == Ok("1")
}

#[test]
#[ignore]
fn gpu_graph_is_a_number_on_the_replay_path() {
    if !gpu_graph_gated() {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_GPU_GRAPH_TEST=1");
    }

    unsafe {
        std::env::set_var("NV_PROF_DECODE_PHASES", "1");
        std::env::set_var("NV_PROF_DECODE_EVERY", "1000000");
    }
    assert!(
        nv_models::decode_probe::enabled(),
        "the probe did not turn on, so this test would pass without measuring anything"
    );
    let Ok(device) = Device::new_cuda(0) else {
        panic!("PRECONDITION NOT MET: no CUDA device");
    };
    let model = load(&device);

    let before = nv_models::decode_probe::gpu_coverage();
    {
        let cache = model.new_kv_cache_fp8(64).expect("kv cache");
        let mut dec = GraphedGemma4Decoder::new(&model, cache, &device).expect("decoder");
        let mut tok = 3u32;
        for _ in 0..8 {
            tok = dec.forward_decode_logits(tok).expect("decode step");
        }
    }
    let after = nv_models::decode_probe::gpu_coverage();

    let steps = after.1 - before.1;
    let timed = after.0 - before.0;
    eprintln!("[gpu-graph] {timed} of {steps} decode steps carried a gpu_graph time");
    assert!(
        steps >= 8,
        "only {steps} steps recorded; the probe is not recording and this gate is inert"
    );
    assert_eq!(
        timed, steps,
        "gpu_graph was missing on {} of {steps} steps -- it shows as NaN in the probe table \
         and nothing else notices",
        steps - timed
    );
}
