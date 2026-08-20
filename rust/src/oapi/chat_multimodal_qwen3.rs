use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};

use crate::oapi::chat::GEMMA4_IMAGE_MARKER;

#[derive(Clone)]
pub struct Qwen3MmSpec {
    pub image_token_id: u32,
    pub vision_start: String,
    pub vision_end: String,
    pub image_pad: String,
    pub patch: usize,
    pub merge: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub mean: [f32; 3],
    pub std: [f32; 3],
    pub mrope_section: Option<[usize; 3]>,
}

impl Qwen3MmSpec {
    pub fn factor(&self) -> usize {
        self.patch * self.merge
    }

    pub fn from_model_dir(
        model_dir: &Path,
        raw_cfg: &str,
        tokenizer: &tokenizers::Tokenizer,
    ) -> Result<Option<Self>> {
        let cfg: serde_json::Value =
            serde_json::from_str(raw_cfg).context("parse config.json for qwen3 mm spec")?;
        if cfg.get("vision_config").is_none() {
            return Ok(None);
        }
        let read_id = |k: &str| -> Result<u32> {
            cfg.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as u32)
                .ok_or_else(|| anyhow::anyhow!("config.json: missing top-level {k}"))
        };
        let image_token_id = read_id("image_token_id")?;
        let vision_start_id = read_id("vision_start_token_id")?;
        let vision_end_id = read_id("vision_end_token_id")?;
        let resolve = |id: u32, what: &str| -> Result<String> {
            tokenizer
                .id_to_token(id)
                .ok_or_else(|| anyhow::anyhow!("tokenizer has no token for {what} id {id}"))
        };
        let image_pad = resolve(image_token_id, "image_token_id")?;
        let vision_start = resolve(vision_start_id, "vision_start_token_id")?;
        let vision_end = resolve(vision_end_id, "vision_end_token_id")?;

        let pp_path = model_dir.join("preprocessor_config.json");
        let pp_raw = std::fs::read_to_string(&pp_path).with_context(|| {
            format!(
                "read {} (the qwen3.8 checkpoint ships a preprocessor_config.json)",
                pp_path.display()
            )
        })?;
        let pp: serde_json::Value =
            serde_json::from_str(&pp_raw).with_context(|| format!("parse {}", pp_path.display()))?;
        let get_usize = |k: &str, default: usize| -> usize {
            pp.get(k)
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };
        let patch = get_usize("patch_size", 16);
        let merge = get_usize("merge_size", 2);
        let size = pp.get("size");
        let min_pixels = pp
            .get("min_pixels")
            .and_then(|v| v.as_u64())
            .or_else(|| size.and_then(|s| s.get("shortest_edge")).and_then(|v| v.as_u64()))
            .map(|v| v as usize)
            .ok_or_else(|| anyhow::anyhow!("preprocessor_config.json: no min_pixels/shortest_edge"))?;
        let max_pixels = pp
            .get("max_pixels")
            .and_then(|v| v.as_u64())
            .or_else(|| size.and_then(|s| s.get("longest_edge")).and_then(|v| v.as_u64()))
            .map(|v| v as usize)
            .ok_or_else(|| anyhow::anyhow!("preprocessor_config.json: no max_pixels/longest_edge"))?;
        let triple = |k: &str, default: f32| -> [f32; 3] {
            match pp.get(k).and_then(|v| v.as_array()) {
                Some(a) if a.len() == 3 => [
                    a[0].as_f64().unwrap_or(default as f64) as f32,
                    a[1].as_f64().unwrap_or(default as f64) as f32,
                    a[2].as_f64().unwrap_or(default as f64) as f32,
                ],
                Some(a) if a.len() == 1 => {
                    let v = a[0].as_f64().unwrap_or(default as f64) as f32;
                    [v, v, v]
                }
                _ => [default, default, default],
            }
        };
        let mean = triple("image_mean", 0.5);
        let std = triple("image_std", 0.5);

        let mrope_section =
            nv_models::qwen3_mm_splice::mrope_section_from_hf_json_str(raw_cfg).ok();

        Ok(Some(Self {
            image_token_id,
            vision_start,
            vision_end,
            image_pad,
            patch,
            merge,
            min_pixels,
            max_pixels,
            mean,
            std,
            mrope_section,
        }))
    }
}

pub fn smart_resize(
    h: usize,
    w: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(h > 0 && w > 0, "smart_resize: zero dimension");
    let hf = h as f64;
    let wf = w as f64;
    let ratio = hf.max(wf) / hf.min(wf);
    anyhow::ensure!(
        ratio <= 200.0,
        "smart_resize: aspect ratio {ratio:.1} exceeds 200"
    );
    let ff = factor as f64;
    let round_mult = |x: f64| -> usize { ((x / ff).round() * ff).max(ff) as usize };
    let floor_mult = |x: f64| -> usize { ((x / ff).floor() * ff).max(ff) as usize };
    let ceil_mult = |x: f64| -> usize { ((x / ff).ceil() * ff).max(ff) as usize };
    let mut h_bar = round_mult(hf);
    let mut w_bar = round_mult(wf);
    if h_bar * w_bar > max_pixels {
        let beta = ((hf * wf) / max_pixels as f64).sqrt();
        h_bar = floor_mult(hf / beta);
        w_bar = floor_mult(wf / beta);
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / (hf * wf)).sqrt();
        h_bar = ceil_mult(hf * beta);
        w_bar = ceil_mult(wf * beta);
    }
    Ok((h_bar, w_bar))
}

pub struct PreppedImage {
    pub pixels: Vec<f32>,
    pub h: usize,
    pub w: usize,
    pub patch: usize,
    pub merge: usize,
}

impl PreppedImage {
    pub fn grid(&self) -> (usize, usize) {
        (self.h / self.patch, self.w / self.patch)
    }

    pub fn merged_tokens(&self) -> usize {
        let (gh, gw) = self.grid();
        gh * gw / (self.merge * self.merge)
    }
}

pub fn prep_images(spec: &Qwen3MmSpec, images: &[image::RgbImage]) -> Result<Vec<PreppedImage>> {
    let factor = spec.factor();
    let mut out = Vec::with_capacity(images.len());
    for img in images {
        let (w0, h0) = (img.width() as usize, img.height() as usize);
        let (h, w) = smart_resize(h0, w0, factor, spec.min_pixels, spec.max_pixels)?;
        let resized = image::imageops::resize(
            img,
            w as u32,
            h as u32,
            image::imageops::FilterType::CatmullRom,
        );
        let mut pixels = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let p = resized.get_pixel(x as u32, y as u32);
                for c in 0..3 {
                    let v = (p.0[c] as f32 / 255.0 - spec.mean[c]) / spec.std[c];
                    pixels[c * h * w + y * w + x] = v;
                }
            }
        }
        out.push(PreppedImage {
            pixels,
            h,
            w,
            patch: spec.patch,
            merge: spec.merge,
        });
    }
    Ok(out)
}

pub fn expand_marker_prompt(
    spec: &Qwen3MmSpec,
    prompt: &str,
    prepped: &[PreppedImage],
) -> Result<String> {
    let count = prompt.matches(GEMMA4_IMAGE_MARKER).count();
    anyhow::ensure!(
        count == prepped.len(),
        "prompt has {count} image markers but {} images were prepared",
        prepped.len()
    );
    let mut out = String::with_capacity(prompt.len());
    let mut rest = prompt;
    for p in prepped {
        let idx = rest
            .find(GEMMA4_IMAGE_MARKER)
            .expect("marker count checked above");
        out.push_str(&rest[..idx]);
        out.push_str(&spec.vision_start);
        for _ in 0..p.merged_tokens() {
            out.push_str(&spec.image_pad);
        }
        out.push_str(&spec.vision_end);
        rest = &rest[idx + GEMMA4_IMAGE_MARKER.len()..];
    }
    out.push_str(rest);
    Ok(out)
}

fn f32_to_bf16_bits_round_nearest_even(x: f32) -> u16 {
    let bits = x.to_bits();
    if bits & 0x7fff_ffff > 0x7f80_0000 {
        return 0x7fc0;
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

pub struct Qwen3VisionMm {
    spec: Qwen3MmSpec,
    tower: nv_omni::Qwen3VisionTower,
    device: Device,
}

impl Qwen3VisionMm {
    pub fn spec(&self) -> &Qwen3MmSpec {
        &self.spec
    }

    pub fn load(spec: Qwen3MmSpec, model_dir: &Path) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).context("qwen3 vision tower needs a cuda device")?;
        #[cfg(not(feature = "cuda"))]
        let device = Device::Cpu;
        let weights = nv_weights::WeightLoader::open_dir(model_dir, &device)?;
        let cfg = nv_omni::Qwen3VisionConfig::from_hf_config_json(model_dir.join("config.json"))?;
        let mut tower = nv_omni::Qwen3VisionTower::new_empty(cfg, &device)?;
        tower.load_weights(&weights)?;
        Ok(Self {
            spec,
            tower,
            device,
        })
    }

    pub fn encode(&self, img: &PreppedImage) -> Result<Tensor> {
        let t = Tensor::from_vec(img.pixels.clone(), (1, 3, img.h, img.w), &self.device)?;
        self.tower.forward(&t)
    }

    pub fn splice_rows(
        &self,
        prompt_ids: &[u32],
        embeds: &[Tensor],
    ) -> Result<Vec<nv_models::embed_row_splice::EmbedRowSplice>> {
        let splices =
            self.tower
                .build_splices(prompt_ids, self.spec.image_token_id, embeds)?;
        let mut out = Vec::with_capacity(splices.len());
        for sp in splices {
            let f: Vec<f32> = sp
                .embedding
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            let rows_bf16: Vec<u16> = f.iter().map(|&x| f32_to_bf16_bits_round_nearest_even(x)).collect();
            out.push(nv_models::embed_row_splice::EmbedRowSplice {
                position: sp.position,
                rows_bf16,
            });
        }
        Ok(out)
    }
}
