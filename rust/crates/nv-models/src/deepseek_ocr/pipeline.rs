use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use nv_weights::WeightLoader;
use tokenizers::Tokenizer;

use super::decoder::{
    build_prompt_tokens, strip_grounding_tokens, DeepseekOcrDecoder, DeepseekOcrDecoderConfig,
    GenerateOptions, LoopDetection, PROMPT_FREE_OCR, PROMPT_GROUNDING_MARKDOWN,
};
use super::preprocess::{prepare, PreparedViews, ResolutionMode, RgbImage};
use super::{DeepSeekOcr2Vision, VisionConfig};

#[cfg(feature = "cuda")]
use super::decoder_graph::DsocrDecodeGraph;
#[cfg(feature = "cuda")]
use std::sync::Mutex;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DecoderPrecision {
    #[default]
    Bf16,
    Nvfp4,
    W4,
    W8,
}

pub const DSOCR_QUANT_ENV_NVFP4_IS_THE_CUDA_DEFAULT_PER_CER_AB_0_0286_BF16_0_0288_NVFP4_IN_PERF_RUNS_AND_BF16_IS_THE_ESCAPE:
    &str = "NV_OCR_DEEPSEEK_QUANT";

impl DecoderPrecision {
    pub fn from_quant_env_value(raw: Option<&str>, nvfp4_capable: bool) -> Self {
        match raw {
            Some("nvfp4") => Self::Nvfp4,
            Some("bf16") => Self::Bf16,
            Some("w4") => Self::W4,
            Some("w8") => Self::W8,
            _ if nvfp4_capable => Self::Nvfp4,
            _ => Self::Bf16,
        }
    }

    pub fn for_device(device: &Device) -> Self {
        Self::from_quant_env_value(
            std::env::var(DSOCR_QUANT_ENV_NVFP4_IS_THE_CUDA_DEFAULT_PER_CER_AB_0_0286_BF16_0_0288_NVFP4_IN_PERF_RUNS_AND_BF16_IS_THE_ESCAPE)
                .ok()
                .as_deref(),
            cfg!(feature = "cuda") && device.is_cuda(),
        )
    }
}

pub struct DeepSeekOcr2Pipeline {
    vision: DeepSeekOcr2Vision,
    decoder: Arc<DeepseekOcrDecoder>,
    tokenizer: Tokenizer,
    device: Device,
    #[cfg(feature = "cuda")]
    graph: Mutex<Option<DsocrDecodeGraph>>,
    #[cfg(feature = "cuda")]
    graph_eligible_checked_once_so_nvfp4_never_reattempts_capture_per_page: bool,
}

fn retry_alternate_prompt() -> bool {
    nv_config::env_arm_opt_out_default_on_unless_eq_some_0_reread_every_call("NV_DSOCR_LOOP_RETRY")
}

impl DeepSeekOcr2Pipeline {
    pub fn load(dir: &Path, device: &Device, precision: DecoderPrecision) -> Result<Self> {
        #[cfg(not(feature = "cuda"))]
        if precision == DecoderPrecision::Nvfp4 {
            anyhow::bail!(
                "deepseek-ocr2: DecoderPrecision::Nvfp4 requires the cuda feature; \
                 rebuild with --features cuda or use DecoderPrecision::Bf16 \
                 (runs on CPU in f32 on non-cuda builds)"
            );
        }
        let weights = WeightLoader::open_dir(dir, device)
            .with_context(|| format!("open checkpoint dir {}", dir.display()))?;
        let cfg = DeepseekOcrDecoderConfig::from_hf_json_file(&dir.join("config.json"))?;
        let vis_dtype = match std::env::var("NV_DSOCR_VIS_DTYPE").ok().as_deref() {
            Some("f32") => DType::F32,
            Some("bf16") => DType::BF16,
            _ if device.is_cuda() => DType::BF16,
            _ => DType::F32,
        };
        let vision = DeepSeekOcr2Vision::from_loader(
            &weights,
            "model.",
            VisionConfig::deepseek_ocr2(),
            device,
            vis_dtype,
        )
        .context("load DeepSeek-OCR-2 vision tower")?;
        let decoder = match precision {
            DecoderPrecision::Bf16 => DeepseekOcrDecoder::from_loader(cfg, &weights, device)
                .context("load DeepSeek-OCR-2 decoder bf16")?,
            DecoderPrecision::Nvfp4 => {
                #[cfg(feature = "cuda")]
                {
                    DeepseekOcrDecoder::from_loader_nvfp4(cfg, &weights, device)
                        .context("load DeepSeek-OCR-2 decoder nvfp4")?
                }
                #[cfg(not(feature = "cuda"))]
                anyhow::bail!("nvfp4 decoder precision requires the cuda feature")
            }
            DecoderPrecision::W4 => DeepseekOcrDecoder::from_loader_ggml(
                cfg,
                &weights,
                device,
                super::decoder::env_decoder_dtype(),
                candle_core::quantized::GgmlDType::Q4K,
            )
            .context("load DeepSeek-OCR-2 decoder w4 (ggml Q4K)")?,
            DecoderPrecision::W8 => DeepseekOcrDecoder::from_loader_ggml(
                cfg,
                &weights,
                device,
                super::decoder::env_decoder_dtype(),
                candle_core::quantized::GgmlDType::Q8_0,
            )
            .context("load DeepSeek-OCR-2 decoder w8 (ggml Q8_0)")?,
        };
        let tokenizer = nv_tokenizer::load_tokenizer(&dir.join("tokenizer.json"))?;
        let decoder = Arc::new(decoder);
        #[cfg(feature = "cuda")]
        let graph_eligible = super::decoder_graph::graph_supported(&decoder).is_ok();
        let me = Self {
            vision,
            decoder,
            tokenizer,
            device: device.clone(),
            #[cfg(feature = "cuda")]
            graph: Mutex::new(None),
            #[cfg(feature = "cuda")]
            graph_eligible_checked_once_so_nvfp4_never_reattempts_capture_per_page: graph_eligible,
        };
        me.warmup();
        Ok(me)
    }

    fn warmup_modes() -> Vec<ResolutionMode> {
        match std::env::var("NV_DSOCR_WARMUP").as_deref() {
            Ok("0") => Vec::new(),
            Ok("all") => vec![
                ResolutionMode::Gundam,
                ResolutionMode::Base1024,
                ResolutionMode::Base768,
            ],
            Ok(other) if !other.is_empty() => match other {
                "base1024" => vec![ResolutionMode::Base1024],
                "base768" => vec![ResolutionMode::Base768],
                _ => vec![ResolutionMode::Gundam],
            },
            _ => vec![ResolutionMode::Gundam],
        }
    }

    fn warmup(&self) {
        let modes = Self::warmup_modes();
        if modes.is_empty() {
            return;
        }
        if !self.device.is_cuda() && std::env::var("NV_DSOCR_WARMUP").is_err() {
            return;
        }
        let img = RgbImage::from_fn(1024, 1024, |x, y| {
            let v = (((x ^ y) & 0xff) as u8).wrapping_add(32);
            [v, v, v]
        });
        let t0 = std::time::Instant::now();
        let opts = GenerateOptions {
            max_new_tokens: 1,
            ..Default::default()
        };
        for mode in modes {
            if let Err(e) = self.generate_tokens(&img, PROMPT_FREE_OCR, mode, &opts) {
                eprintln!("[dsocr] warmup ({mode:?}) failed, first request will pay it: {e:#}");
                return;
            }
        }
        eprintln!("[dsocr] warmed in {:.1}s", t0.elapsed().as_secs_f64());
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn decoder(&self) -> &DeepseekOcrDecoder {
        &self.decoder
    }

    pub fn vision(&self) -> &DeepSeekOcr2Vision {
        &self.vision
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn encode_text(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn decoder_arc(&self) -> Arc<DeepseekOcrDecoder> {
        self.decoder.clone()
    }

    fn generate_once(
        &self,
        features: &Tensor,
        n_vision_tokens: usize,
        prompt: &str,
        opts: &GenerateOptions,
    ) -> Result<(Vec<u32>, Option<LoopDetection>)> {
        let tokens = build_prompt_tokens(|s| self.encode_text(s), prompt, n_vision_tokens)?;
        #[cfg(feature = "cuda")]
        if super::decoder_graph::graph_enabled()
            && self.device.is_cuda()
            && self.graph_eligible_checked_once_so_nvfp4_never_reattempts_capture_per_page
        {
            let mut slot = self
                .graph
                .lock()
                .map_err(|e| anyhow::anyhow!("dsocr graph lock poisoned: {e}"))?;
            if slot.is_none() {
                match DsocrDecodeGraph::new(
                    self.decoder.clone(),
                    self.decoder.config().max_position_embeddings,
                ) {
                    Ok(g) => *slot = Some(g),
                    Err(e) => {
                        eprintln!("[dsocr] graph init failed, falling back to eager: {e:#}")
                    }
                }
            }
            if let Some(g) = slot.as_mut() {
                let outcome = g.generate(&tokens, Some(features), opts)?;
                return Ok((outcome.tokens, outcome.loop_detection));
            }
        }
        let outcome = self
            .decoder
            .generate_detected(&tokens, Some(features), opts)?;
        Ok((outcome.tokens, outcome.loop_detection))
    }

    pub fn generate_tokens_flagged(
        &self,
        img: &RgbImage,
        prompt: &str,
        mode: ResolutionMode,
        opts: &GenerateOptions,
    ) -> Result<(Vec<u32>, PreparedViews, bool)> {
        let prep = prepare(img, mode)?;
        let features = self.vision.encode_prepared(&prep)?;
        let (mut out, det) = self.generate_once(&features, prep.vision_tokens(), prompt, opts)?;
        let Some(d) = det else {
            return Ok((out, prep, false));
        };
        out.truncate(d.onset);
        if prompt.trim() == PROMPT_FREE_OCR && retry_alternate_prompt() {
            let (mut retry, retry_det) = self.generate_once(
                &features,
                prep.vision_tokens(),
                PROMPT_GROUNDING_MARKDOWN,
                opts,
            )?;
            if let Some(rd) = retry_det {
                retry.truncate(rd.onset);
            }
            let retry = strip_grounding_tokens(&retry);
            if retry.len() > out.len() {
                return Ok((retry, prep, true));
            }
        }
        Ok((out, prep, true))
    }

    pub fn generate_tokens(
        &self,
        img: &RgbImage,
        prompt: &str,
        mode: ResolutionMode,
        opts: &GenerateOptions,
    ) -> Result<(Vec<u32>, PreparedViews)> {
        let (out, prep, _) = self.generate_tokens_flagged(img, prompt, mode, opts)?;
        Ok((out, prep))
    }

    pub fn recognize_flagged(
        &self,
        img: &RgbImage,
        prompt: &str,
        mode: ResolutionMode,
        opts: &GenerateOptions,
    ) -> Result<(String, bool)> {
        let (out, _, looped) = self.generate_tokens_flagged(img, prompt, mode, opts)?;
        let text = self
            .tokenizer
            .decode(&out, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        Ok((text, looped))
    }

    pub fn recognize(
        &self,
        img: &RgbImage,
        prompt: &str,
        mode: ResolutionMode,
        opts: &GenerateOptions,
    ) -> Result<String> {
        let (text, _) = self.recognize_flagged(img, prompt, mode, opts)?;
        Ok(text)
    }
}

#[cfg(test)]
mod precision_default_tests {
    use super::DecoderPrecision;

    #[test]
    fn unset_env_defaults_nvfp4_where_capable_and_bf16_elsewhere() {
        assert_eq!(
            DecoderPrecision::from_quant_env_value(None, true),
            DecoderPrecision::Nvfp4
        );
        assert_eq!(
            DecoderPrecision::from_quant_env_value(None, false),
            DecoderPrecision::Bf16
        );
    }

    #[test]
    fn explicit_nvfp4_opts_in_and_fails_loudly_off_cuda_instead_of_silently_downgrading() {
        assert_eq!(
            DecoderPrecision::from_quant_env_value(Some("nvfp4"), false),
            DecoderPrecision::Nvfp4
        );
    }

    #[test]
    fn bf16_escape_restores_the_graphed_bf16_decoder_even_where_nvfp4_is_capable() {
        assert_eq!(
            DecoderPrecision::from_quant_env_value(Some("bf16"), true),
            DecoderPrecision::Bf16
        );
    }

    #[test]
    fn unknown_values_follow_the_capability_default_like_unset() {
        assert_eq!(
            DecoderPrecision::from_quant_env_value(Some("int8"), true),
            DecoderPrecision::Nvfp4
        );
        assert_eq!(
            DecoderPrecision::from_quant_env_value(Some("int8"), false),
            DecoderPrecision::Bf16
        );
    }
}

#[cfg(all(test, not(feature = "cuda")))]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_precision_fails_cleanly_without_cuda() {
        let Err(err) = DeepSeekOcr2Pipeline::load(
            Path::new("/nonexistent/dsocr-checkpoint"),
            &Device::Cpu,
            DecoderPrecision::Nvfp4,
        ) else {
            panic!("load unexpectedly succeeded")
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("cuda"),
            "error should name the cuda feature: {msg}"
        );
        assert!(
            msg.contains("Bf16"),
            "error should point at the working precision: {msg}"
        );
    }

    #[test]
    fn bf16_precision_reports_missing_checkpoint_not_panic() {
        let Err(err) = DeepSeekOcr2Pipeline::load(
            Path::new("/nonexistent/dsocr-checkpoint"),
            &Device::Cpu,
            DecoderPrecision::Bf16,
        ) else {
            panic!("load unexpectedly succeeded")
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("checkpoint"),
            "error should mention the checkpoint dir: {msg}"
        );
    }
}
