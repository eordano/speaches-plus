#![cfg(feature = "cuda")]

mod common;
use common::VOCAB_512 as VOCAB;
use common::write_tiny_model;
use candle_core::{DType, Device, Tensor};
use nv_models::gemma4::{Gemma4, Gemma4Config};
use nv_models::gemma4_graph::GraphedGemma4Decoder;
use nv_weights::WeightLoader;
use std::collections::HashMap;

fn gated() -> bool {
    std::env::var("NV_GEMMA4_GRAPH_LOGITS_TEST").ok().as_deref() == Some("1")
}

fn load(device: &Device) -> Gemma4 {
    static INTO_FIXTURE: std::sync::OnceLock<std::path::PathBuf> =
        std::sync::OnceLock::new();
    let dir = INTO_FIXTURE
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

fn make_decoders<'m>(
    model: &'m Gemma4,
    device: &Device,
    n: usize,
) -> Vec<GraphedGemma4Decoder<'m>> {
    let caches: Vec<_> = (0..n)
        .map(|_| model.new_kv_cache_fp8(64).expect("kv cache"))
        .collect();
    caches
        .into_iter()
        .map(|c| GraphedGemma4Decoder::new(model, c, device).expect("decoder"))
        .collect()
}

fn run_prefix(dec: &mut GraphedGemma4Decoder<'_>, tokens: &[u32]) -> Vec<Vec<f32>> {
    tokens
        .iter()
        .map(|&t| dec.forward_decode_logits_into(t).expect("decode").to_vec())
        .collect()
}

#[test]
#[ignore]
fn logits_into_returns_the_current_step_not_the_previous_one() {
    if !gated() {
        eprintln!("skip: set NV_GEMMA4_GRAPH_LOGITS_TEST=1 to run");
        return;
    }
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let model = load(&device);
    let tokens: [u32; 6] = [3, 17, 5, 41, 9, 23];

    let mut decs = make_decoders(&model, &device, 1 + tokens.len());
    let rows = run_prefix(&mut decs[0], &tokens);
    assert_eq!(rows.len(), tokens.len());
    for (i, r) in rows.iter().enumerate() {
        assert_eq!(r.len(), VOCAB, "step {i}: row is not one full vocab wide");
        assert!(
            r.iter().all(|v| v.is_finite()),
            "step {i}: the pinned D2H produced non-finite values, so it was read \
             before the copy landed"
        );
        assert!(
            r.iter().any(|&v| v != 0.0),
            "step {i}: the whole row is zero -- the D2H never landed"
        );
    }

    for i in 1..rows.len() {
        assert!(
            rows[i] != rows[i - 1],
            "step {i} returned a bit-identical row to step {}: the pinned buffer \
             was not refreshed",
            i - 1
        );
    }

    for k in 0..tokens.len() {
        let fresh = run_prefix(&mut decs[1 + k], &tokens[..=k]);
        let want = fresh.last().unwrap();
        assert_eq!(
            &rows[k],
            want,
            "row {k} of the full run does not match the last row of an \
             independent {}-token replay: the returned borrow is stale",
            k + 1
        );
    }
    eprintln!(
        "logits_into_returns_the_current_step_not_the_previous_one: {} steps, \
         vocab {VOCAB}, prefix-replay exact",
        tokens.len()
    );
}

#[test]
#[ignore]
fn logits_vec_wrapper_matches_the_pinned_path_bitwise() {
    if !gated() {
        eprintln!("skip: set NV_GEMMA4_GRAPH_LOGITS_TEST=1 to run");
        return;
    }
    let Ok(device) = Device::new_cuda(0) else {
        eprintln!("skip: no CUDA device");
        return;
    };
    let model = load(&device);
    let tokens: [u32; 4] = [3, 17, 5, 41];

    let mut decs = make_decoders(&model, &device, 2);
    let via_into = run_prefix(&mut decs[0], &tokens);
    let via_vec: Vec<Vec<f32>> = tokens
        .iter()
        .map(|&t| decs[1].forward_decode_logits_vec(t).expect("decode"))
        .collect();

    assert_eq!(
        via_vec, via_into,
        "the Vec wrapper diverged from the borrow"
    );
    eprintln!(
        "logits_vec_wrapper_matches_the_pinned_path_bitwise: {} steps OK",
        tokens.len()
    );
}
