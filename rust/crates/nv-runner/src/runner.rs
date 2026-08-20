use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use std::path::Path;

use crate::config::Qwen3Config;
use crate::model::{KvCache, Qwen3};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    MaxTokens,
    Eos,
}

pub struct GenerationResult {
    pub prompt_tokens: Vec<u32>,
    pub output_tokens: Vec<u32>,
    pub output_text: String,
    pub finish_reason: FinishReason,
}

pub struct GreedyRunner {
    model: Qwen3,
    tokenizer: nv_tokenizer::Tokenizer,
    cache: KvCache,
    device: Device,
    dtype: DType,
}

impl GreedyRunner {
    #[cfg(feature = "cuda")]
    pub fn from_pretrained(model_dir: &Path, device: &Device) -> Result<Self> {
        let paths = crate::discover_model_files(model_dir)?;
        let config = Qwen3Config::from_hf_json_file(&paths.config)?;
        let tokenizer = nv_tokenizer::load_tokenizer(&paths.tokenizer)?;
        let model = Qwen3::from_pretrained(config.clone(), &paths.weights, device)?;
        let cache = KvCache::new(
            model.config(),
            config.max_position_embeddings.min(8192),
            device,
            DType::BF16,
        )?;
        Ok(Self {
            model,
            tokenizer,
            cache,
            device: device.clone(),
            dtype: DType::BF16,
        })
    }

    #[cfg(not(feature = "cuda"))]
    pub fn from_pretrained(_model_dir: &Path, _device: &Device) -> Result<Self> {
        anyhow::bail!("nv-runner requires --features cuda")
    }

    pub fn from_assembled(
        model: Qwen3,
        tokenizer: nv_tokenizer::Tokenizer,
        device: &Device,
    ) -> Result<Self> {
        let cache = KvCache::new(
            model.config(),
            model.config().max_position_embeddings.min(8192),
            device,
            DType::BF16,
        )?;
        Ok(Self {
            model,
            tokenizer,
            cache,
            device: device.clone(),
            dtype: DType::BF16,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn config(&self) -> &Qwen3Config {
        self.model.config()
    }

    pub fn tokenizer(&self) -> &nv_tokenizer::Tokenizer {
        &self.tokenizer
    }

    pub fn generate(&mut self, prompt: &str, max_new_tokens: usize) -> Result<GenerationResult> {
        let encoded = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let prompt_tokens: Vec<u32> = encoded.get_ids().to_vec();
        self.generate_tokens(&prompt_tokens, max_new_tokens)
    }

    pub fn generate_tokens(
        &mut self,
        prompt_tokens: &[u32],
        max_new_tokens: usize,
    ) -> Result<GenerationResult> {
        if prompt_tokens.is_empty() {
            anyhow::bail!("empty prompt_tokens");
        }
        self.cache.reset();

        let mut output_tokens: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let cfg = self.model.config().clone();

        let prompt_len = prompt_tokens.len();
        let prompt_t = Tensor::from_vec(prompt_tokens.to_vec(), (1, prompt_len), &self.device)?;
        let positions = Tensor::from_vec(
            (0..prompt_len as i32).collect::<Vec<_>>(),
            prompt_len,
            &self.device,
        )?;

        let logits = self.model.forward(&prompt_t, &positions, &mut self.cache)?;
        let next_token = argmax_last_position(&logits)?;
        output_tokens.push(next_token);

        let mut finish = FinishReason::MaxTokens;
        if cfg.is_eos(next_token) {
            finish = FinishReason::Eos;
        } else {
            for step in 1..max_new_tokens {
                let pos = (prompt_len + step - 1) as i32;
                let tok_t = Tensor::from_vec(vec![output_tokens[step - 1]], (1, 1), &self.device)?;
                let pos_t = Tensor::from_vec(vec![pos], 1, &self.device)?;
                let logits = self.model.forward(&tok_t, &pos_t, &mut self.cache)?;
                let nt = argmax_last_position(&logits)?;
                output_tokens.push(nt);
                if cfg.is_eos(nt) {
                    finish = FinishReason::Eos;
                    break;
                }
            }
        }

        let output_text = self
            .tokenizer
            .decode(&output_tokens, true)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;

        Ok(GenerationResult {
            prompt_tokens: prompt_tokens.to_vec(),
            output_tokens,
            output_text,
            finish_reason: finish,
        })
    }
}

fn argmax_last_position(logits: &Tensor) -> Result<u32> {
    let dims = logits.dims();
    if dims.len() != 3 {
        anyhow::bail!("expected (B, T, V) logits, got {:?}", dims);
    }
    let (_b, t, _v) = (dims[0], dims[1], dims[2]);
    let last = logits.narrow(1, t - 1, 1)?.squeeze(1)?;
    let last_f32 = last.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let v: Vec<f32> = last_f32.squeeze(0)?.to_vec1()?;
    let mut best_idx = 0u32;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > best_val {
            best_val = x;
            best_idx = i as u32;
        }
    }
    Ok(best_idx)
}
