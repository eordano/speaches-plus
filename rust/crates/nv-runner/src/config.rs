use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

#[derive(Clone, Debug, Deserialize)]
pub struct Qwen3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_tie")]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<EosField>,
}

fn default_rms_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_max_pos() -> usize {
    32768
}
fn default_tie() -> bool {
    false
}

#[derive(Clone, Debug, Deserialize)]
#[serde(untagged)]
pub enum EosField {
    Single(u32),
    Many(Vec<u32>),
}

impl EosField {
    pub fn first(&self) -> Option<u32> {
        match self {
            EosField::Single(v) => Some(*v),
            EosField::Many(v) => v.first().copied(),
        }
    }

    pub fn matches(&self, tok: u32) -> bool {
        match self {
            EosField::Single(v) => *v == tok,
            EosField::Many(v) => v.contains(&tok),
        }
    }
}

impl Qwen3Config {
    pub fn from_hf_json_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        let cfg: Self = serde_json::from_str(&text)?;
        Ok(cfg)
    }

    pub fn eos_token(&self) -> Option<u32> {
        self.eos_token_id.as_ref().and_then(|f| f.first())
    }

    pub fn is_eos(&self, tok: u32) -> bool {
        self.eos_token_id
            .as_ref()
            .map(|f| f.matches(tok))
            .unwrap_or(false)
    }
}
