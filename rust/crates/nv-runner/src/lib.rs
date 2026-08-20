use anyhow::Result;
use std::path::Path;

pub use candle_core::{DType, Device, Tensor};
pub use nv_tokenizer::Tokenizer;

mod config;
mod model;
mod runner;

pub use config::Qwen3Config;
pub use model::{KvCache, Qwen3};
pub use runner::{FinishReason, GenerationResult, GreedyRunner};

pub fn discover_model_files(model_dir: &Path) -> Result<ModelPaths> {
    let config_path = model_dir.join("config.json");
    if !config_path.exists() {
        anyhow::bail!("missing config.json under {}", model_dir.display());
    }
    let tokenizer_path = model_dir.join("tokenizer.json");
    if !tokenizer_path.exists() {
        anyhow::bail!("missing tokenizer.json under {}", model_dir.display());
    }
    let single = model_dir.join("model.safetensors");
    let index = model_dir.join("model.safetensors.index.json");
    let weight_files: Vec<std::path::PathBuf> = if single.exists() {
        vec![single]
    } else if index.exists() {
        let text = std::fs::read_to_string(&index)?;
        let parsed: serde_json::Value = serde_json::from_str(&text)?;
        let map = parsed
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| anyhow::anyhow!("model.safetensors.index.json missing weight_map"))?;
        let mut set = std::collections::BTreeSet::new();
        for v in map.values() {
            if let Some(s) = v.as_str() {
                set.insert(model_dir.join(s));
            }
        }
        set.into_iter().collect()
    } else {
        anyhow::bail!(
            "no model.safetensors or model.safetensors.index.json under {}",
            model_dir.display()
        );
    };
    Ok(ModelPaths {
        config: config_path,
        tokenizer: tokenizer_path,
        weights: weight_files,
    })
}

pub struct ModelPaths {
    pub config: std::path::PathBuf,
    pub tokenizer: std::path::PathBuf,
    pub weights: Vec<std::path::PathBuf>,
}
