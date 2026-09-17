use anyhow::Result;
use safetensors::tensor::{Dtype, TensorView};

use super::lora::LoraVar;

pub const PEFT_BYTES_MATCH_NV_TRAIN_SAVE_PEFT_RAW_SO_DGLORA_HOT_LOADS_THEM_UNCHANGED: &str =
    "the adapter pair is byte-compatible with nv_train::save_peft_raw: bf16 lora_A [r,k] and \
     lora_B [n,r] tensors under base_model.model.<module>.lora_{A,B}.weight with the trainer's \
     language_model.model.* module names, which nv_weights::lora_adapter::normalize_module_name \
     maps onto the serving graph's model.language_model.* namespace; the browser writes these \
     bytes to OPFS, the native gate writes them to a dir and round-trips DgLora::from_peft_dir";

fn bf16_le_bytes(vals: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vals.len() * 2);
    for v in vals {
        out.extend_from_slice(&half::bf16::from_f32(*v).to_bits().to_le_bytes());
    }
    out
}

pub fn adapter_safetensors_bytes(loras: &[LoraVar]) -> Result<Vec<u8>> {
    let _ = PEFT_BYTES_MATCH_NV_TRAIN_SAVE_PEFT_RAW_SO_DGLORA_HOT_LOADS_THEM_UNCHANGED;
    let mut blobs: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::with_capacity(loras.len() * 2);
    for lv in loras {
        blobs.push((
            format!("base_model.model.{}.lora_A.weight", lv.module),
            vec![lv.r, lv.k],
            bf16_le_bytes(&lv.a),
        ));
        blobs.push((
            format!("base_model.model.{}.lora_B.weight", lv.module),
            vec![lv.n, lv.r],
            bf16_le_bytes(&lv.b),
        ));
    }
    let views = blobs
        .iter()
        .map(|(name, shape, bytes)| {
            TensorView::new(Dtype::BF16, shape.clone(), bytes).map(|v| (name.as_str(), v))
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(safetensors::serialize(views, None)?)
}

pub fn adapter_config_json_bytes(
    loras: &[LoraVar],
    r: usize,
    alpha: f64,
    base_model_name_or_path: &str,
) -> Vec<u8> {
    let mut targets: Vec<String> = Vec::new();
    for lv in loras {
        let leaf = lv
            .module
            .rsplit('.')
            .next()
            .unwrap_or(&lv.module)
            .to_string();
        if !targets.contains(&leaf) {
            targets.push(leaf);
        }
    }
    targets.sort();
    serde_json::json!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": r,
        "lora_alpha": alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "use_rslora": false,
        "use_dora": false,
        "modules_to_save": serde_json::Value::Null,
        "fan_in_fan_out": false,
        "inference_mode": true,
        "target_modules": targets,
        "base_model_name_or_path": base_model_name_or_path,
    })
    .to_string()
    .into_bytes()
}
