use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device};
use nv_weights::WeightLoader;
use tokenizers::Tokenizer;

use super::decoder::{
    build_prompt_tokens, DotsDecoder, DotsDecoderConfig, GenerateOptions, PromptStyle,
};
use super::parse::{parse_layout_json, plain_text_fallback, LayoutPage};
use super::preprocess::{prepare, PixelBudget, PreparedImage};
use super::vision::{DotsVisionConfig, DotsVisionTower};
use crate::deepseek_ocr::preprocess::RgbImage;

pub const PROMPT_LAYOUT_ALL_EN: &str = "Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.\n\n1. Bbox format: [x1, y1, x2, y2]\n\n2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n3. Text Extraction & Formatting Rules:\n    - Picture: For the 'Picture' category, the text field should be omitted.\n    - Formula: Format its text as LaTeX.\n    - Table: Format its text as HTML.\n    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n4. Constraints:\n    - The output text must be the original text from the image, with no translation.\n    - All layout elements must be sorted according to human reading order.\n\n5. Final Output: The entire output must be a single JSON object.\n";

pub const PROMPT_LAYOUT_ONLY_EN: &str = "Please output the layout information from this PDF image, including each layout's bbox and its category. The bbox should be in the format [x1, y1, x2, y2]. The layout categories for the PDF document include ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title']. Do not output the corresponding text. The layout result should be in JSON format.";

pub const PROMPT_OCR: &str = "Extract the text content from this image.";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DotsMode {
    #[default]
    LayoutAll,
    LayoutOnly,
    PlainOcr,
}

impl DotsMode {
    pub fn prompt(self) -> &'static str {
        match self {
            DotsMode::LayoutAll => PROMPT_LAYOUT_ALL_EN,
            DotsMode::LayoutOnly => PROMPT_LAYOUT_ONLY_EN,
            DotsMode::PlainOcr => PROMPT_OCR,
        }
    }

    pub fn emits_json(self) -> bool {
        !matches!(self, DotsMode::PlainOcr)
    }
}

#[derive(Debug, Clone)]
pub struct DotsPageResult {
    pub page: LayoutPage,
    pub text: String,
    pub raw: String,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub looped: bool,
    pub hit_eos: bool,
    pub grid: (usize, usize),
}

pub struct DotsOcrPipeline {
    vision: DotsVisionTower,
    decoder: DotsDecoder,
    tokenizer: Tokenizer,
    budget: PixelBudget,
    style: PromptStyle,
    device: Device,
}

fn default_rope_span() -> usize {
    std::env::var("NV_DOTS_MAX_SEQ")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(32768)
}

impl DotsOcrPipeline {
    pub fn load(dir: &Path, device: &Device) -> Result<Self> {
        let weights = WeightLoader::open_dir(dir, device)
            .with_context(|| format!("open dots.ocr checkpoint dir {}", dir.display()))?;
        let cfg = DotsDecoderConfig::from_hf_json_file(&dir.join("config.json"))
            .context("load dots.ocr config.json")?;
        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vision = DotsVisionTower::from_loader(
            &weights,
            "vision_tower.",
            DotsVisionConfig::default(),
            device,
            dtype,
        )
        .context("load dots.ocr vision tower")?;
        let decoder = DotsDecoder::from_loader(cfg, &weights, device, dtype, default_rope_span())
            .context("load dots.ocr decoder")?;
        let tokenizer = nv_tokenizer::load_tokenizer(&dir.join("tokenizer.json"))
            .context("load dots.ocr tokenizer.json")?;
        Ok(Self {
            vision,
            decoder,
            tokenizer,
            budget: PixelBudget::from_env(),
            style: PromptStyle::from_env(),
            device: device.clone(),
        })
    }

    pub fn prompt_style(&self) -> PromptStyle {
        self.style
    }

    pub fn set_prompt_style(&mut self, style: PromptStyle) {
        self.style = style;
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn decoder(&self) -> &DotsDecoder {
        &self.decoder
    }

    pub fn vision(&self) -> &DotsVisionTower {
        &self.vision
    }

    pub fn pixel_budget(&self) -> PixelBudget {
        self.budget
    }

    pub fn set_pixel_budget(&mut self, budget: PixelBudget) {
        self.budget = budget;
    }

    pub fn encode_text(&self, s: &str) -> Result<Vec<u32>> {
        Ok(self
            .tokenizer
            .encode(s, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode: {e}"))?
            .get_ids()
            .to_vec())
    }

    pub fn prepare(&self, img: &RgbImage) -> Result<PreparedImage> {
        prepare(img, self.budget)
    }

    pub fn recognize(
        &self,
        img: &RgbImage,
        mode: DotsMode,
        opts: &GenerateOptions,
    ) -> Result<DotsPageResult> {
        let prep = self.prepare(img)?;
        let feats = self.vision.encode(&prep)?;
        let n_vis = prep.num_vision_tokens();
        anyhow::ensure!(
            feats.dim(0)? == n_vis,
            "vision produced {} tokens, expected {n_vis}",
            feats.dim(0)?
        );
        let (tokens, vision_start) =
            build_prompt_tokens(|s| self.encode_text(s), mode.prompt(), n_vis, self.style)?;
        let outcome = self
            .decoder
            .generate(&tokens, vision_start, Some(&feats), opts)?;
        let mut out_tokens = outcome.tokens;
        let looped = outcome.loop_detection.is_some();
        if let Some(d) = &outcome.loop_detection {
            if d.period > 0 || !mode.emits_json() {
                out_tokens.truncate(d.onset);
            }
        }
        let raw = self
            .tokenizer
            .decode(&out_tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode: {e}"))?;
        let mut page = if mode.emits_json() {
            parse_layout_json(&raw)
        } else {
            plain_text_fallback(&raw)
        };
        page.truncated |= !outcome.hit_eos;
        page.rescale(
            prep.orig_w as f32 / prep.resized_w as f32,
            prep.orig_h as f32 / prep.resized_h as f32,
        );
        let text = page.to_plain_text();
        Ok(DotsPageResult {
            page,
            text,
            raw,
            prompt_tokens: tokens.len(),
            generated_tokens: out_tokens.len(),
            looped,
            hit_eos: outcome.hit_eos,
            grid: (prep.grid_h, prep.grid_w),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_checkpoint_reports_the_directory() {
        let Err(err) = DotsOcrPipeline::load(Path::new("/nonexistent/dots-ocr"), &Device::Cpu)
        else {
            panic!("load unexpectedly succeeded")
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains("dots.ocr"),
            "error should name the model: {msg}"
        );
    }

    #[test]
    fn prompt_bodies_carry_the_canonical_recipe() {
        assert!(DotsMode::LayoutAll.prompt().contains("human reading order"));
        assert!(DotsMode::LayoutAll.prompt().contains("[x1, y1, x2, y2]"));
        assert!(DotsMode::LayoutOnly
            .prompt()
            .contains("Do not output the corresponding text"));
        assert!(DotsMode::LayoutAll.emits_json());
        assert!(!DotsMode::PlainOcr.emits_json());
    }
}
