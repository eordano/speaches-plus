use std::path::Path;

use anyhow::{anyhow, Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_weights::WeightLoader;
use tokenizers::Tokenizer;

use super::vision::{GotLmNames, GotVision};
use crate::deepseek_ocr::decoder::{banned_tokens_windowed_ngram, detect_loop};
use crate::deepseek_ocr::preprocess::{resize_rgb, RgbImage};
use crate::dots_ocr::decoder::{
    argmax_banned, DotsDecoder, DotsDecoderConfig, DotsKvCache, GenerateOptions, GenerateOutcome,
};

pub const GOT_IMAGE_TOKENS: usize = 256;
pub const GOT_SYSTEM: &str =
    "You should follow the instructions carefully and explain your answers in detail.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GotMode {
    Plain,
    Format,
}

impl GotMode {
    pub fn instruction(self) -> &'static str {
        match self {
            GotMode::Plain => "OCR: ",
            GotMode::Format => "OCR with format: ",
        }
    }
}

#[derive(Debug, Clone)]
pub struct GotPageResult {
    pub text: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub looped: bool,
    pub hit_eos: bool,
}

pub struct GotOcrPipeline {
    spec: nv_ocr::PreprocessSpec,
    vision: GotVision,
    decoder: DotsDecoder,
    tokenizer: Tokenizer,
    image_token: u32,
    eos_ids: Vec<u32>,
    device: Device,
    dtype: DType,
}

fn rope_span_from_env() -> usize {
    std::env::var("NV_GOT_MAX_SEQ")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(8192)
}

fn fast_kv_grow_by_concat_matches_accurate_numerics_at_1_33x() -> bool {
    std::env::var("NV_GOT_FAST_KV").as_deref() == Ok("1")
}

fn text_config_json(raw: &str) -> Result<String> {
    let mut v: serde_json::Value = serde_json::from_str(raw).context("parse config.json")?;
    let tc = v
        .get_mut("text_config")
        .and_then(|t| t.as_object_mut())
        .context("config.json missing text_config object")?;
    tc.entry("rms_norm_eps")
        .or_insert_with(|| serde_json::json!(1e-6));
    tc.entry("max_position_embeddings")
        .or_insert_with(|| serde_json::json!(32768));
    serde_json::to_string(tc).context("re-serialize text_config")
}

impl GotOcrPipeline {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("read {}/config.json", dir.display()))?;
        let cfg = nv_ocr::ModelOcrConfig::from_json_str(&raw).map_err(|e| anyhow!("{e}"))?;

        let weight_path = dir.join(nv_ocr::WEIGHT_FILE);
        let len = std::fs::metadata(&weight_path)
            .with_context(|| format!("stat {}", weight_path.display()))?
            .len();
        anyhow::ensure!(
            len == nv_ocr::WEIGHT_FILE_BYTES_INCLUDING_SAFETENSORS_HEADER,
            "{} is {len} bytes, expected {} (safetensors header + {} bytes bf16 payload)",
            weight_path.display(),
            nv_ocr::WEIGHT_FILE_BYTES_INCLUDING_SAFETENSORS_HEADER,
            nv_ocr::WEIGHT_BYTES_BF16
        );

        let weights = WeightLoader::open_dir(dir, device)
            .with_context(|| format!("open GOT-OCR2 checkpoint dir {}", dir.display()))?;
        let names = weights.names();
        anyhow::ensure!(
            names.len() == nv_ocr::WEIGHT_TENSOR_COUNT,
            "checkpoint has {} tensors, expected {}",
            names.len(),
            nv_ocr::WEIGHT_TENSOR_COUNT
        );
        nv_ocr::verify_weight_map(&cfg, &names).map_err(|e| anyhow!("{e}"))?;

        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vision = GotVision::from_loader(&weights, dtype).context("load GOT vision tower")?;

        let text_cfg = DotsDecoderConfig::from_hf_json_str(&text_config_json(&raw)?)
            .context("parse GOT text_config as Qwen2 decoder config")?;
        anyhow::ensure!(
            text_cfg.hidden_size == 1024,
            "GOT text_config hidden_size {} must be 1024 to match the projector output",
            text_cfg.hidden_size
        );
        anyhow::ensure!(
            text_cfg.tie_word_embeddings,
            "GOT ties word embeddings; a checkpoint with a separate lm_head breaks the 471-tensor gate"
        );
        let decoder = DotsDecoder::from_loader(
            text_cfg,
            &GotLmNames(&weights),
            device,
            dtype,
            rope_span_from_env(),
        )
        .context("load GOT Qwen2 decoder")?;

        let tokenizer = nv_tokenizer::load_tokenizer(&dir.join("tokenizer.json"))
            .context("load GOT tokenizer.json")?;
        let enc = |s: &str| -> Result<Vec<u32>> {
            Ok(tokenizer
                .encode(s, false)
                .map_err(|e| anyhow!("tokenizer encode {s:?}: {e}"))?
                .get_ids()
                .to_vec())
        };
        let imgpad = enc("<imgpad>")?;
        anyhow::ensure!(
            imgpad == [cfg.image_token_index],
            "<imgpad> must encode to the config image_token_index {}, got {imgpad:?}",
            cfg.image_token_index
        );
        let image_token = cfg.image_token_index;
        let one = |s: &str| -> Result<u32> {
            let e = enc(s)?;
            anyhow::ensure!(e.len() == 1, "{s} must be a single token, got {e:?}");
            Ok(e[0])
        };
        let im_end = one("<|im_end|>")?;
        let eot = one("<|endoftext|>")?;
        let eos_ids = vec![im_end, eot];

        Ok(Self {
            spec: nv_ocr::PreprocessSpec::got_ocr2(),
            vision,
            decoder,
            tokenizer,
            image_token,
            eos_ids,
            device: device.clone(),
            dtype,
        })
    }

    fn preprocess(&self, img: &RgbImage) -> Vec<f32> {
        let r = resize_rgb(img, self.spec.width, self.spec.height);
        let n = r.w * r.h;
        let mean = self.spec.image_mean;
        let std = self.spec.image_std;
        let mut out = vec![0f32; 3 * n];
        for c in 0..3 {
            for i in 0..n {
                let v = r.data[i * 3 + c] as f32 * self.spec.rescale_factor;
                out[c * n + i] = (v - mean[c]) / std[c];
            }
        }
        out
    }

    fn prompt_ids(&self, mode: GotMode) -> Result<(Vec<u32>, usize)> {
        let pads = "<imgpad>".repeat(GOT_IMAGE_TOKENS);
        let text = format!(
            "<|im_start|>system\n{GOT_SYSTEM}<|im_end|><|im_start|>user\n<img>{pads}</img>\n {}<|im_end|><|im_start|>assistant\n",
            mode.instruction()
        );
        let ids = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("tokenizer encode prompt: {e}"))?
            .get_ids()
            .to_vec();
        let vision_start = ids
            .iter()
            .position(|&t| t == self.image_token)
            .context("rendered prompt has no image token")?;
        anyhow::ensure!(
            vision_start + GOT_IMAGE_TOKENS <= ids.len()
                && ids[vision_start..vision_start + GOT_IMAGE_TOKENS]
                    .iter()
                    .all(|&t| t == self.image_token),
            "image token run at {vision_start} is not {GOT_IMAGE_TOKENS} contiguous imgpad tokens"
        );
        anyhow::ensure!(
            vision_start == 0 || ids[vision_start - 1] != self.image_token,
            "image token run starts before {vision_start}"
        );
        anyhow::ensure!(
            vision_start + GOT_IMAGE_TOKENS >= ids.len()
                || ids[vision_start + GOT_IMAGE_TOKENS] != self.image_token,
            "image token run exceeds {GOT_IMAGE_TOKENS} tokens"
        );
        Ok((ids, vision_start))
    }

    pub fn encode_image(&self, img: &RgbImage) -> Result<Tensor> {
        let pixels = Tensor::from_vec(self.preprocess(img), (1, 3, 1024, 1024), &Device::Cpu)?
            .to_device(&self.device)?
            .to_dtype(self.dtype)?;
        self.vision.forward(&pixels)
    }

    pub fn recognize(
        &self,
        img: &RgbImage,
        mode: GotMode,
        max_new_tokens: usize,
    ) -> Result<GotPageResult> {
        let feats = self.encode_image(img)?;
        anyhow::ensure!(
            feats.dims2()? == (GOT_IMAGE_TOKENS, self.decoder.config().hidden_size),
            "GOT vision produced {:?}, expected [{GOT_IMAGE_TOKENS}, {}]",
            feats.dims(),
            self.decoder.config().hidden_size
        );
        let (tokens, vision_start) = self.prompt_ids(mode)?;
        let opts = GenerateOptions {
            max_new_tokens,
            eos_token_ids: self.eos_ids.clone(),
            ..Default::default()
        };
        let outcome = if fast_kv_grow_by_concat_matches_accurate_numerics_at_1_33x() {
            self.generate_grow_kv(&tokens, vision_start, &feats, &opts)?
        } else {
            self.decoder
                .generate(&tokens, vision_start, Some(&feats), &opts)?
        };
        let text = self
            .tokenizer
            .decode(&outcome.tokens, true)
            .map_err(|e| anyhow!("tokenizer decode: {e}"))?;
        Ok(GotPageResult {
            text,
            prompt_tokens: tokens.len(),
            generated_tokens: outcome.tokens.len(),
            looped: outcome.loop_detection.is_some(),
            hit_eos: outcome.hit_eos,
        })
    }

    fn generate_grow_kv(
        &self,
        prompt_tokens: &[u32],
        vision_start: usize,
        vision_features: &Tensor,
        opts: &GenerateOptions,
    ) -> Result<GenerateOutcome> {
        let d = &self.decoder;
        let cfg = d.config();
        let prompt_len = prompt_tokens.len();
        anyhow::ensure!(
            prompt_len < d.rope_span(),
            "prompt of {prompt_len} tokens exceeds the rope span {}",
            d.rope_span()
        );
        let budget = (prompt_len + opts.max_new_tokens + 1).min(d.rope_span());
        let mut cache =
            DotsKvCache::new_with_mode(cfg, budget, d.device(), d.dtype(), true)?;
        let n = vision_features.dim(0)?;
        anyhow::ensure!(
            vision_start + n <= prompt_len,
            "vision span {vision_start}+{n} exceeds prompt {prompt_len}"
        );
        let feats = vision_features
            .reshape((1, n, cfg.hidden_size))?
            .to_dtype(d.dtype())?;
        let embeds = d.embed_tokens(prompt_tokens)?.slice_assign(
            &[0..1, vision_start..vision_start + n, 0..cfg.hidden_size],
            &feats,
        )?;
        let mut logits = d.last_logits(&embeds, 0, &mut cache)?;
        let mut generated: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        for _ in 0..opts.max_new_tokens {
            let banned = match opts.ngram_size {
                Some(k) if k > 0 => banned_tokens_windowed_ngram(
                    &generated,
                    k,
                    opts.ngram_window,
                    &opts.ngram_whitelist,
                ),
                _ => Vec::new(),
            };
            let next = argmax_banned(&logits, &banned);
            if opts.eos_token_ids.contains(&next) {
                hit_eos = true;
                break;
            }
            generated.push(next);
            if opts.stop_on_loop
                && generated.len().is_multiple_of(64)
                && detect_loop(&generated).is_some()
            {
                break;
            }
            let pos = prompt_len + generated.len() - 1;
            if pos + 1 >= cache.max_seq_len() {
                break;
            }
            let step = d.embed_tokens(&[next])?;
            logits = d.last_logits(&step, pos, &mut cache)?;
        }
        let loop_detection = detect_loop(&generated);
        Ok(GenerateOutcome {
            tokens: generated,
            loop_detection,
            hit_eos,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
