#![cfg(feature = "cuda")]

use candle_core::{DType, Device, Tensor};
use nv_layers::RmsNorm;

const EPS: f64 = 1e-6;
const HEAD_DIM_FULL: usize = 512;

#[test]
fn v_norm_output_is_the_k_norm_output_divided_by_the_scalar() {
    let device = Device::Cpu;
    let scalar = 0.0615f32;
    let rows = 64usize;

    let x = Tensor::rand(-4f32, 4f32, (rows, HEAD_DIM_FULL), &device).unwrap();

    let ones = Tensor::ones(HEAD_DIM_FULL, DType::F32, &device).unwrap();
    let v_norm = RmsNorm::new(ones.clone(), EPS);
    let k_norm = RmsNorm::new((ones * scalar as f64).unwrap(), EPS);

    let v = v_norm.forward(&x).unwrap();
    let k = k_norm.forward(&x).unwrap();
    let derived = (k / scalar as f64).unwrap();

    let e: Vec<f32> = (derived - &v)
        .unwrap()
        .abs()
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1()
        .unwrap();
    let r: Vec<f32> = v.abs().unwrap().flatten_all().unwrap().to_vec1().unwrap();
    let num: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
    let den: f32 = r.iter().map(|x| x * x).sum::<f32>().sqrt();
    let rel = num / den;

    assert!(
        rel < 1e-6,
        "v_norm(x) != k_norm(x)/{scalar}: rel-rms {rel:e}. The V-from-K derivation \
         rests on v_norm carrying weight 1 and k_norm carrying a scalar; one of \
         those changed."
    );
}

#[test]
#[ignore]
fn every_full_layer_k_norm_weight_is_one_scalar() {
    if std::env::var("NV_KV_SHARE_TEST").as_deref() != Ok("1") {
        panic!("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: set NV_KV_SHARE_TEST=1");
    }
    let dir = std::env::var("NV_CHAT_MODEL_DIR")
        .expect("PRECONDITION NOT MET, THIS TEST EXECUTED NOTHING: NV_CHAT_MODEL_DIR unset");
    let raw = std::fs::read_to_string(format!("{dir}/config.json")).expect("config.json");
    let cfg: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let text = cfg.get("text_config").unwrap_or(&cfg);

    assert_eq!(
        text.get("attention_k_eq_v").and_then(|v| v.as_bool()),
        Some(true),
        "this checkpoint does not share K and V on full layers; the whole \
         derivation is void for it"
    );

    let types: Vec<String> = text
        .get("layer_types")
        .and_then(|v| v.as_array())
        .expect("layer_types")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();

    let device = Device::Cpu;
    let weights = nv_weights::WeightLoader::open_dir(std::path::Path::new(&dir), &device).unwrap();

    let mut checked = 0usize;
    for (layer, kind) in types.iter().enumerate() {
        if kind != "full_attention" {
            continue;
        }
        let name = format!("model.language_model.layers.{layer}.self_attn.k_norm.weight");
        let w: Vec<f32> = weights
            .get(&name, DType::F32)
            .unwrap_or_else(|e| panic!("{name}: {e}"))
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(w.len(), HEAD_DIM_FULL, "{name} width");
        let lo = w.iter().cloned().fold(f32::INFINITY, f32::min);
        let hi = w.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert_eq!(
            lo, hi,
            "{name} is a per-dim vector ({lo}..{hi}), not a scalar. Deriving V \
             from K would need a 512-wide reciprocal and a division by {lo}."
        );
        assert!(
            lo > 1e-2,
            "{name} scalar {lo} is small enough to amplify fp8 KV error"
        );
        checked += 1;
    }
    assert!(
        checked >= 10,
        "expected at least 10 full-attention layers, checked {checked}"
    );
}
