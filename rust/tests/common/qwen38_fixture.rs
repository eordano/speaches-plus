use candle_core::{Device, Tensor};
use nv_models::qwen3_5_moe::{Qwen3Moe, Qwen3_5DenseConfig};
use nv_weights::{QuantizationConfig, WeightLoader};
use std::path::PathBuf;

pub fn qwen38_nvfp4_snapshot_dir_env_override_then_home_hub() -> PathBuf {
    if let Ok(d) = std::env::var("NV_QWEN38_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").expect("HOME");
    let base = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("qwen3.8 snapshots dir {base:?} missing: {e}"))
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| {
            p.join("config.json").is_file()
                && p.join("tokenizer.json").is_file()
                && p.join("chat_template.jinja").is_file()
                && (p.join("model.safetensors").is_file()
                    || p.join("model.safetensors.index.json").is_file())
        })
        .expect("no complete Qwen3.8-27B-NVFP4 snapshot under HOME hub; set NV_QWEN38_DIR")
}

pub fn load_qwen38_dense_on_the_cuda_serving_arm(dir: &PathBuf, device: &Device) -> Qwen3Moe {
    let raw_cfg = std::fs::read_to_string(dir.join("config.json")).expect("config.json");
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).expect("parse qwen3.8 dense config");
    let qcfg = QuantizationConfig::from_hf_json_str(&raw_cfg).expect("parse qcfg");
    let weights = WeightLoader::open_dir(dir, device).expect("weights");
    let model =
        Qwen3Moe::from_loader_dense_quantized(cfg.clone(), &weights, &qcfg, device).expect("model");
    assert_eq!(
        model.dense_intermediate(),
        Some(cfg.intermediate_size),
        "from_loader_dense_quantized did not carry intermediate_size, so this is not the dense \
         arm the cuda serving opt-in builds"
    );
    model
}

pub fn host_row_f32(logits: &Tensor) -> Vec<f32> {
    logits
        .to_dtype(candle_core::DType::F32)
        .expect("f32")
        .flatten_all()
        .expect("flatten")
        .to_vec1()
        .expect("host")
}

pub fn eos_ids_from_generation_config(dir: &PathBuf) -> Vec<u32> {
    let raw = std::fs::read_to_string(dir.join("generation_config.json"))
        .expect("generation_config.json");
    let v: serde_json::Value = serde_json::from_str(&raw).expect("parse generation_config.json");
    let eos = match v.get("eos_token_id") {
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_u64())
            .map(|x| x as u32)
            .collect(),
        Some(serde_json::Value::Number(n)) => {
            n.as_u64().map(|x| x as u32).into_iter().collect()
        }
        _ => Vec::new(),
    };
    assert!(
        !eos.is_empty(),
        "generation_config.json carries no eos_token_id; free-running without a stop set never \
         terminates"
    );
    eos
}
