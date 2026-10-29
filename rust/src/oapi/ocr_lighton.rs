use std::path::Path;

use anyhow::{ensure, Context, Result};
use candle_core::Device;
use nv_models::deepseek_ocr::decoder::{detect_loop, LOOP_CHECK_STRIDE};
use nv_models::deepseek_ocr::RgbImage;
use nv_models::qwen3_5_dense_wgpu::{Qwen3_5DenseConfig, Qwen3_5DenseWgpu};
use nv_models::qwen3_mm_splice::build_mrope_positions_matching_hf_get_rope_index;

use super::chat::GEMMA4_IMAGE_MARKER;
use super::chat_multimodal_qwen3::{expand_marker_prompt, prep_images, Qwen3MmSpec, Qwen3VisionMm};

pub const LIGHTON_LONGEST_EDGE_PX_IS_THE_MODEL_CARD_RENDER_TARGET: usize = 2048;
pub const LIGHTON_GROUNDING_PROMPT_IS_THE_ONLY_TRAINED_INSTRUCTION_BESIDE_THE_EMPTY_ONE: &str =
    "grounding";
pub const LIGHTON_BOXES_ARE_NORMALIZED_TO_THIS_PAGE_EXTENT: f32 = 1000.0;
pub const LIGHTON_MAX_SEQ_COVERS_A_2048_EDGE_PAGE_OF_4096_VISION_TOKENS_PLUS_ITS_TEXT: usize =
    16384;
const LIGHTON_BLOCKS_THAT_DESCRIBE_A_PICTURE_INSTEAD_OF_TRANSCRIBING_PAGE_TEXT: [&str; 3] =
    ["image", "header_image", "footer_image"];

#[derive(Debug, Clone, PartialEq)]
pub struct LightOnBlock {
    pub label: String,
    pub bbox: Option<[f32; 4]>,
    pub text: String,
}

impl LightOnBlock {
    pub fn is_page_text(&self) -> bool {
        !LIGHTON_BLOCKS_THAT_DESCRIBE_A_PICTURE_INSTEAD_OF_TRANSCRIBING_PAGE_TEXT
            .contains(&self.label.trim_end_matches('+'))
    }
}

pub struct LightOnPage {
    pub text: String,
    pub blocks: Vec<LightOnBlock>,
    pub hit_eos: bool,
    pub looped: bool,
    pub prompt_tokens: usize,
    pub vision_tokens: usize,
    pub generated_tokens: usize,
    pub encode_ms: f64,
    pub prefill_ms: f64,
    pub decode_ms: f64,
}

pub struct LightOnOcrPipeline {
    tokenizer: tokenizers::Tokenizer,
    vision: Qwen3VisionMm,
    decoder: Qwen3_5DenseWgpu,
    eos: Vec<u32>,
}

impl LightOnOcrPipeline {
    pub fn load(dir: &Path, decoder_weights: Option<&Path>, vision_device: Device) -> Result<Self> {
        let raw_cfg = std::fs::read_to_string(dir.join("config.json"))
            .with_context(|| format!("read {}/config.json", dir.display()))?;
        let tokenizer = tokenizers::Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(anyhow::Error::msg)
            .with_context(|| format!("load {}/tokenizer.json", dir.display()))?;
        let spec = Qwen3MmSpec::from_model_dir(dir, &raw_cfg, &tokenizer)?.with_context(|| {
            format!(
                "{}: config.json has no vision_config, so this is not a LightOnOCR checkpoint",
                dir.display()
            )
        })?;
        let vision = Qwen3VisionMm::load_on(spec, dir, vision_device)?;
        let cfg = Qwen3_5DenseConfig::from_hf_json_str(&raw_cfg).context("parse qwen3.5 dense")?;
        let weights_dir = decoder_weights.unwrap_or(dir);
        let loader = nv_weights::WeightLoader::open_dir(weights_dir, &Device::Cpu)
            .with_context(|| format!("open decoder weights in {}", weights_dir.display()))?;
        let decoder = Qwen3_5DenseWgpu::from_loader(
            cfg,
            &loader,
            LIGHTON_MAX_SEQ_COVERS_A_2048_EDGE_PAGE_OF_4096_VISION_TOKENS_PLUS_ITS_TEXT,
        )
        .context("build Qwen3_5DenseWgpu for LightOnOCR")?;
        ensure!(
            decoder.prefill_chunk_len() >= 2,
            "{}: the decoder built no chunked prefill graph, and image rows enter only through \
             it; an MLX affine-w4 checkpoint prefills per token unless \
             NV_Q3D_AFFINE_CHUNKED_PF_V2_MEASURES_26TOK_0_97S_1613TOK_27_8MS_VS_STEPPED_1_10S_38_5MS=1",
            weights_dir.display()
        );
        let eos = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tokenizer.token_to_id(t))
            .collect::<Vec<_>>();
        ensure!(!eos.is_empty(), "tokenizer has neither <|im_end|> nor <|endoftext|>");
        Ok(Self {
            tokenizer,
            vision,
            decoder,
            eos,
        })
    }

    pub fn recognize(
        &mut self,
        img: &RgbImage,
        grounding: bool,
        max_new_tokens: usize,
    ) -> Result<LightOnPage> {
        let spec = self.vision.spec().clone();
        let page = fit_longest_edge(img)?;
        let prepped = prep_images(&spec, std::slice::from_ref(&page))?;
        let instruction = if grounding {
            LIGHTON_GROUNDING_PROMPT_IS_THE_ONLY_TRAINED_INSTRUCTION_BESIDE_THE_EMPTY_ONE
        } else {
            ""
        };
        let prompt = format!(
            "<|im_start|>user\n{GEMMA4_IMAGE_MARKER}{instruction}<|im_end|>\n\
             <|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        let prompt = expand_marker_prompt(&spec, &prompt, &prepped)?;
        let ids = self
            .tokenizer
            .encode(prompt.as_str(), false)
            .map_err(anyhow::Error::msg)?
            .get_ids()
            .to_vec();
        let t0 = std::time::Instant::now();
        let embeds = vec![self.vision.encode(&prepped[0])?];
        let splices = self.vision.splice_rows(&ids, &embeds)?;
        let encode_ms = t0.elapsed().as_secs_f64() * 1e3;
        let t1 = std::time::Instant::now();
        self.decoder.reset()?;
        if let Some(section) = spec.mrope_section {
            let (gh, gw) = prepped[0].grid();
            let merge = prepped[0].merge;
            let mp = build_mrope_positions_matching_hf_get_rope_index(
                &ids,
                spec.image_token_id,
                &[(1, gh / merge, gw / merge)],
            )?;
            if !mp.is_text_degenerate() {
                self.decoder
                    .install_mrope_rows_for_prompt_and_shift_continuation_rows_by_delta(
                        &mp, section,
                    )?;
            }
        }
        let last = ids.len() - 1;
        let done = self
            .decoder
            .prefill_tokens_with_image_rows(&ids[..last], &splices)?;
        ensure!(done == last, "image prefill consumed {done} of {last} prompt tokens");
        let budget = max_new_tokens.min(
            LIGHTON_MAX_SEQ_COVERS_A_2048_EDGE_PAGE_OF_4096_VISION_TOKENS_PLUS_ITS_TEXT
                .saturating_sub(ids.len() + 1),
        );
        let mut out: Vec<u32> = Vec::new();
        let mut hit_eos = false;
        let mut tok = self.decoder.decode_step(ids[last])?;
        let prefill_ms = t1.elapsed().as_secs_f64() * 1e3;
        let t2 = std::time::Instant::now();
        while out.len() < budget {
            if self.eos.contains(&tok) {
                hit_eos = true;
                break;
            }
            out.push(tok);
            if out.len().is_multiple_of(LOOP_CHECK_STRIDE) && detect_loop(&out).is_some() {
                break;
            }
            if out.len() < budget {
                tok = self.decoder.decode_step(tok)?;
            }
        }
        let generated_tokens = out.len();
        let decode_ms = t2.elapsed().as_secs_f64() * 1e3;
        let looped = match detect_loop(&out) {
            Some(d) => {
                out.truncate(d.onset);
                true
            }
            None => false,
        };
        let raw = self
            .tokenizer
            .decode(&out, true)
            .map_err(anyhow::Error::msg)?;
        let (text, blocks) = if grounding {
            let blocks = parse_grounding(&raw, img.w as f32, img.h as f32);
            let text = blocks
                .iter()
                .filter(|b| b.is_page_text() && !b.text.is_empty())
                .map(|b| b.text.as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            (text, blocks)
        } else {
            (raw.trim().to_string(), Vec::new())
        };
        Ok(LightOnPage {
            text,
            blocks,
            hit_eos,
            looped,
            prompt_tokens: ids.len(),
            vision_tokens: prepped[0].merged_tokens(),
            generated_tokens,
            encode_ms,
            prefill_ms,
            decode_ms,
        })
    }
}

fn fit_longest_edge(img: &RgbImage) -> Result<image::RgbImage> {
    let (w0, h0) = (img.w, img.h);
    let edge = LIGHTON_LONGEST_EDGE_PX_IS_THE_MODEL_CARD_RENDER_TARGET;
    let (w, h, data) = if w0.max(h0) > edge {
        let s = edge as f64 / w0.max(h0) as f64;
        let w = ((w0 as f64 * s).round() as usize).max(1);
        let h = ((h0 as f64 * s).round() as usize).max(1);
        (w, h, nv_omni::resize_rgb8_bicubic_matching_pil(&img.data, w0, h0, w, h))
    } else {
        (w0, h0, img.data.clone())
    };
    image::RgbImage::from_raw(w as u32, h as u32, data)
        .with_context(|| format!("rgb buffer does not match {w}x{h}"))
}

fn parse_placeholder(line: &str) -> Option<(String, [f32; 4], &str)> {
    let rest = line.trim_start().strip_prefix("![")?;
    let (label, rest) = rest.split_once("](")?;
    let (coords, tail) = rest.split_once(')')?;
    let v: Vec<f32> = coords
        .split(',')
        .map(|c| c.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .ok()?;
    let bbox = <[f32; 4]>::try_from(v).ok()?;
    if label.is_empty() || label.contains(char::is_whitespace) {
        return None;
    }
    Some((label.to_string(), bbox, tail.trim()))
}

fn flush_block_body(blocks: &mut Vec<LightOnBlock>, body: &mut Vec<&str>) {
    let text = body.join("\n").trim().to_string();
    body.clear();
    match blocks.last_mut() {
        Some(b) if b.text.is_empty() => b.text = text,
        _ if !text.is_empty() => blocks.push(LightOnBlock {
            label: "text".into(),
            bbox: None,
            text,
        }),
        _ => {}
    }
}

pub fn parse_grounding(raw: &str, page_w: f32, page_h: f32) -> Vec<LightOnBlock> {
    let sx = page_w / LIGHTON_BOXES_ARE_NORMALIZED_TO_THIS_PAGE_EXTENT;
    let sy = page_h / LIGHTON_BOXES_ARE_NORMALIZED_TO_THIS_PAGE_EXTENT;
    let mut blocks: Vec<LightOnBlock> = Vec::new();
    let mut body: Vec<&str> = Vec::new();
    let flush = flush_block_body;
    for line in raw.lines() {
        match parse_placeholder(line) {
            Some((label, bbox, tail)) => {
                flush(&mut blocks, &mut body);
                blocks.push(LightOnBlock {
                    label,
                    bbox: Some([bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy]),
                    text: String::new(),
                });
                if !tail.is_empty() {
                    body.push(tail);
                }
            }
            None => body.push(line),
        }
    }
    flush(&mut blocks, &mut body);
    blocks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grounding_blocks_carry_labels_boxes_text_inline_placeholders_and_continuations() {
        let raw = "![title](80,70,920,140)\n# Company report 2025\n\n![text](80,175,920,225)\nRevenue rose 20%.\nSecond line.\n\n![image](80,285,460,545)\nA solar-powered factory.\n";
        let b = parse_grounding(raw, 2000.0, 1000.0);
        assert_eq!(b.len(), 3);
        assert_eq!(b[0].label, "title");
        assert_eq!(b[0].bbox, Some([160.0, 70.0, 1840.0, 140.0]));
        assert_eq!(b[0].text, "# Company report 2025");
        assert_eq!(b[1].text, "Revenue rose 20%.\nSecond line.");
        assert!(b[1].is_page_text());
        assert!(!b[2].is_page_text());
        {
            let raw = "![title](54,37,492,61) # NORTHWIND SUPPLY CO.\n\n![header](677,42,872,98) **INVOICE #2041** Date: 2026-07-18\n\n![table](61,196,940,500) <table>\n<tr><td>a</td></tr>\n</table>";
            let b = parse_grounding(raw, 1000.0, 1000.0);
            assert_eq!(b.len(), 3);
            assert_eq!(b[0].label, "title");
            assert_eq!(b[0].text, "# NORTHWIND SUPPLY CO.");
            assert_eq!(b[1].text, "**INVOICE #2041** Date: 2026-07-18");
            assert_eq!(b[2].text, "<table>\n<tr><td>a</td></tr>\n</table>");
            assert_eq!(b[2].bbox, Some([61.0, 196.0, 940.0, 500.0]));
        }
        {
            let raw = "stray preface\n![text+](0,0,1000,1000)\ncontinued";
            let b = parse_grounding(raw, 1000.0, 1000.0);
            assert_eq!(b.len(), 2);
            assert_eq!((b[0].label.as_str(), b[0].bbox), ("text", None));
            assert_eq!(b[1].label, "text+");
            assert!(b[1].is_page_text());
        }
    }

    #[test]
    fn markdown_images_with_alt_text_are_not_placeholders() {
        assert!(parse_placeholder("![a logo](https://x/y.png)").is_none());
        assert!(parse_placeholder("![chart](1,2,3)").is_none());
        assert!(parse_placeholder("  ![chart](1, 2, 3, 4)  ").is_some());
    }
}
