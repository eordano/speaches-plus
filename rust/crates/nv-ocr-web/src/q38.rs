use anyhow::{Context, Result};
use std::collections::BTreeMap;

use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_weights::WeightLoader;

pub const Q38_PARITY_PROMPT_IDS_ARE_FIXED_SO_BOTH_SIDES_FORWARD_IDENTICAL_TOKENS_WITHOUT_A_TOKENIZER:
    [u32; 12] = [
    248044, 3838, 374, 279, 12483, 315, 220, 16, 15, 489, 220, 16,
];

pub const Q38_PARITY_DECODE_STEPS_READ_FULL_LOGITS_EACH_STEP: usize = 3;

pub const Q38_PARITY_MAX_SEQ_SLACK_KEEPS_THE_KV_AND_SCRATCH_SMALL: usize = 4;

pub const Q38_SPLICE_KV_SPAN_IS_THE_BROWSER_PRACTICAL_CONTEXT: usize = 4096;

pub const Q38_SPLICE_DECODE_STEPS_TIME_A_SHORT_GREEDY_SEQUENCE: usize = 32;

pub const Q38_SPLICE_CENSUS_STEPS_READ_FULL_LOGITS_THE_REST_DECODE_TOKEN_ONLY: usize = 3;

pub const Q38_FULL_DECODE_STEPS_TIME_A_REAL_ANSWER: usize = 64;

pub const Q38_FULL_PROMPT_IS_CHAT_TEMPLATED_TEXT_TOKENIZED_BY_THE_NATIVE_REFERENCE_WRITER: &str =
    "<|im_start|>user\nExplain in one sentence why the sky is blue.<|im_end|>\n<|im_start|>assistant\n";

pub const Q38_FULL_REFUTE_STEM_NAMES_THE_SECOND_FIXTURES_FILES: &str = "ref-full-refute";

pub const Q38_FULL_REFUTING_PROMPT_CROSSES_A_GREEDY_NEAR_TIE_UNDER_TINT_VS_NAGA_DRIFT_AT_STEP_19:
    &str =
    "<|im_start|>user\nName the largest planet in our solar system and give one striking fact about it.<|im_end|>\n<|im_start|>assistant\n";

pub struct Q38ParityModel {
    model: Qwen3_5DenseWgpu,
    logits_le: Vec<u8>,
    last_token: Option<u32>,
    vocab: usize,
}

pub struct Q38ParityBoot {
    pub caps_summary: String,
    pub vocab: usize,
    pub layer_types: Vec<String>,
}

pub async fn q38_parity_boot(
    mut source: BTreeMap<String, Vec<u8>>,
    max_seq: usize,
) -> Result<(Q38ParityModel, Q38ParityBoot)> {
    let config_json = String::from_utf8(
        source
            .remove(crate::CONFIG_JSON_NAME)
            .context("q38 parity source carries no config.json")?,
    )
    .context("q38 config.json is not utf-8")?;
    let shards: Vec<(String, Vec<u8>)> = source
        .into_iter()
        .filter(|(name, _)| name.ends_with(".safetensors"))
        .collect();
    anyhow::ensure!(!shards.is_empty(), "q38 parity source carries no shards");
    let loader = WeightLoader::open_from_named_bytes(shards, &candle_core::Device::Cpu)?;
    q38_boot_from_loader(&config_json, &loader, max_seq).await
}

pub async fn q38_boot_from_loader(
    config_json: &str,
    loader: &WeightLoader,
    max_seq: usize,
) -> Result<(Q38ParityModel, Q38ParityBoot)> {
    let ctx = crate::ensure_browser_gpu().await?;
    let cfg = Qwen3_5DenseConfig::from_hf_json_str(config_json)?;
    let vocab = cfg.vocab_size;
    let layer_types = cfg.layer_types.iter().map(|t| format!("{t:?}")).collect();
    let model = Qwen3_5DenseWgpu::from_loader(cfg, loader, max_seq)?;
    let boot = Q38ParityBoot {
        caps_summary: ctx.summary(),
        vocab,
        layer_types,
    };
    Ok((
        Q38ParityModel {
            model,
            logits_le: Vec::new(),
            last_token: None,
            vocab,
        },
        boot,
    ))
}

impl Q38ParityModel {
    pub async fn prefill(&mut self, prompt: &[u32]) -> Result<u32> {
        let first = self.model.prefill_pumped(prompt).await?;
        self.last_token = Some(first);
        Ok(first)
    }

    pub async fn decode_step_quiet(&mut self) -> Result<u32> {
        let prev = self
            .last_token
            .context("q38 parity decode before prefill")?;
        let tok = self.model.decode_step_pumped(prev).await?;
        self.last_token = Some(tok);
        Ok(tok)
    }

    pub async fn decode_step(&mut self) -> Result<u32> {
        let prev = self
            .last_token
            .context("q38 parity decode before prefill")?;
        let (tok, logits) = self.model.decode_step_logits_pumped(prev).await?;
        anyhow::ensure!(
            logits.len() == self.vocab,
            "step logits carry {} values for vocab {}",
            logits.len(),
            self.vocab
        );
        for v in &logits {
            self.logits_le.extend_from_slice(&v.to_le_bytes());
        }
        self.last_token = Some(tok);
        Ok(tok)
    }

    pub fn reset_for_census_determinism_replay(&mut self) -> Result<()> {
        self.model.reset()?;
        self.logits_le.clear();
        self.last_token = None;
        Ok(())
    }

    pub fn step_pass_labels(&self) -> Vec<String> {
        self.model.debug_step_pass_labels_for_test()
    }

    pub fn logits_le(&self) -> &[u8] {
        &self.logits_le
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }
}
