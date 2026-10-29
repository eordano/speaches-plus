use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::oapi::chat_multimodal_qwen3::{prep_images, PreppedImage, Qwen3MmSpec};
use crate::oapi::cold_tier::ColdSlot;
use nv_models::qwen3_mm_splice::{
    build_mrope_positions_for_vision_runs, prefix_key_with_vision_content, Qwen3MropePositions,
};
use nv_models::qwen4_exp::{Qwen4Exp, Qwen4ExpMm};

pub(crate) const NV_Q4EXP_VISION_DEFAULT_1_ENCODES_IMAGES_NATIVELY_0_SENDS_THEM_THROUGH_THE_OCR_BRIDGE: &str =
    "NV_Q4EXP_VISION";

pub(crate) const NV_Q4EXP_VISION_MAX_PIXELS_DEFAULT_4194304_CAPS_ONE_IMAGE_AT_4096_TOKENS_AND_ABOUT_1_5_GIB_OF_TOWER_ACTIVATIONS: &str =
    "NV_Q4EXP_VISION_MAX_PIXELS";

const DEFAULT_MAX_PIXELS: usize = 4_194_304;

pub(crate) const VISION_TOWER_IS_A_COLD_TIER_RESIDENT_SO_A_TEXT_ONLY_DAY_PAYS_NO_VRAM: &str =
    "the 27-block ViT and merger hold 0.84 GiB of bf16 weights; the engine registers them as the \
     qwen4exp-vision cold-tier resident, loaded on the first image request under the engine permit \
     (after ensure_free makes room by evicting other idle residents) and evicted after \
     NV_COLD_TIER_IDLE_SECS idle or under VRAM pressure, so the 80+ GiB text engine never carries \
     the tower when nobody sends images";

const COLD_TIER_NAME: &str = "qwen4exp-vision";
const TOWER_WEIGHT_BYTES_0_84_GIB: u64 = 448_931_056 * 2;
const F32_ACTIVATION_BYTES_PER_PATCH_ABOUT_96_KIB_COUNTING_THE_FC1_GELU_TEMPORARIES: u64 = 96 * 1024;

pub(crate) struct Qwen4ExpVisionTower {
    tower: nv_omni::Qwen3VisionTower,
}

pub(crate) struct Qwen4ExpVision {
    spec: Qwen3MmSpec,
    video_token_id: u32,
    section: [usize; 3],
    slot: Arc<ColdSlot<Qwen4ExpVisionTower>>,
}

pub(crate) struct PreparedVision {
    pub(crate) prompt: String,
    images: Vec<PreppedImage>,
    hashes: Vec<u64>,
}

static VISIONS: Mutex<Vec<(usize, Arc<Qwen4ExpVision>)>> = Mutex::new(Vec::new());

fn model_key(model: &Arc<Qwen4Exp>) -> usize {
    Arc::as_ptr(model) as usize
}

pub(crate) fn vision_for(model: &Arc<Qwen4Exp>) -> Option<Arc<Qwen4ExpVision>> {
    let k = model_key(model);
    VISIONS.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(m, _)| *m == k).map(|(_, v)| v.clone())
}

pub(crate) fn vision_enabled_from(raw: Option<&str>) -> bool {
    raw != Some("0")
}

fn max_pixels_from(raw: Option<&str>) -> usize {
    raw.and_then(|s| s.parse().ok()).filter(|v| *v > 0).unwrap_or(DEFAULT_MAX_PIXELS)
}

pub(crate) fn register_vision(
    model: &Arc<Qwen4Exp>,
    model_dir: &Path,
    raw_cfg: &str,
    tokenizer: &tokenizers::Tokenizer,
    device: &Device,
) -> Result<bool> {
    let _ = VISION_TOWER_IS_A_COLD_TIER_RESIDENT_SO_A_TEXT_ONLY_DAY_PAYS_NO_VRAM;
    if !vision_enabled_from(std::env::var(NV_Q4EXP_VISION_DEFAULT_1_ENCODES_IMAGES_NATIVELY_0_SENDS_THEM_THROUGH_THE_OCR_BRIDGE).ok().as_deref()) {
        return Ok(false);
    }
    let Some(mut spec) = Qwen3MmSpec::from_model_dir(model_dir, raw_cfg, tokenizer)? else {
        return Ok(false);
    };
    spec.max_pixels = spec.max_pixels.min(max_pixels_from(
        std::env::var(NV_Q4EXP_VISION_MAX_PIXELS_DEFAULT_4194304_CAPS_ONE_IMAGE_AT_4096_TOKENS_AND_ABOUT_1_5_GIB_OF_TOWER_ACTIVATIONS).ok().as_deref(),
    ));
    anyhow::ensure!(spec.min_pixels <= spec.max_pixels, "vision min_pixels {} above max_pixels {}", spec.min_pixels, spec.max_pixels);
    let section = spec
        .mrope_section
        .context("qwen4_exp vision needs text_config.rope_parameters.mrope_section with mrope_interleaved")?;
    let cfg_json: serde_json::Value = serde_json::from_str(raw_cfg)?;
    let video_token_id = cfg_json["video_token_id"].as_u64().context("config.json: video_token_id")? as u32;
    let vcfg = nv_omni::Qwen3VisionConfig::from_hf_config_json(model_dir.join("config.json"))?;
    anyhow::ensure!(
        vcfg.out_hidden_size == model.config().hidden_size,
        "vision merger emits {} wide rows, the text model is {} wide",
        vcfg.out_hidden_size,
        model.config().hidden_size
    );
    let (dir, dev) = (model_dir.to_path_buf(), device.clone());
    let slot = ColdSlot::demand(
        COLD_TIER_NAME,
        TOWER_WEIGHT_BYTES_0_84_GIB,
        Arc::new(move || match load_tower(&dir, vcfg.clone(), &dev) {
            Ok(t) => Some(Arc::new(t)),
            Err(e) => {
                tracing::warn!(error = %format!("{e:#}"), "qwen4_exp vision tower load failed");
                None
            }
        }),
    );
    tracing::info!(
        max_pixels = spec.max_pixels,
        min_pixels = spec.min_pixels,
        "qwen4_exp native vision registered: images encode through the checkpoint's ViT ({COLD_TIER_NAME} cold-tier resident); NV_Q4EXP_VISION=0 restores the OCR bridge"
    );
    VISIONS.lock().unwrap_or_else(|e| e.into_inner()).push((
        model_key(model),
        Arc::new(Qwen4ExpVision { spec, video_token_id, section, slot }),
    ));
    Ok(true)
}

fn load_tower(dir: &Path, mut cfg: nv_omni::Qwen3VisionConfig, device: &Device) -> Result<Qwen4ExpVisionTower> {
    let t0 = std::time::Instant::now();
    cfg.dtype = DType::BF16;
    let weights = nv_weights::WeightLoader::open_dir(dir, device)?;
    let tower = nv_omni::Qwen3VisionTower::load_onto(cfg, &weights, device, true)?;
    eprintln!("[qwen4_exp] vision tower loaded in {:.2}s (bf16 weights, f32 activations)", t0.elapsed().as_secs_f64());
    Ok(Qwen4ExpVisionTower { tower })
}

fn content_hash(img: &image::RgbImage, p: &PreppedImage) -> u64 {
    use sha2::Digest as _;
    let mut h = sha2::Sha256::new();
    for v in [img.width() as u64, img.height() as u64, p.w as u64, p.h as u64] {
        h.update(v.to_le_bytes());
    }
    h.update(img.as_raw());
    u64::from_le_bytes(h.finalize()[..8].try_into().expect("sha256 is 32 bytes"))
}

impl Qwen4ExpVision {
    pub(crate) fn prepare(&self, prompt: &str, images: &[image::RgbImage]) -> Result<PreparedVision> {
        let prepped = prep_images(&self.spec, images)?;
        let hashes = images.iter().zip(&prepped).map(|(i, p)| content_hash(i, p)).collect();
        let prompt = crate::oapi::chat_multimodal_qwen3::expand_marker_prompt(&self.spec, prompt, &prepped)?;
        Ok(PreparedVision { prompt, images: prepped, hashes })
    }

    fn grids(prep: &PreparedVision) -> Vec<(usize, usize, usize)> {
        prep.images
            .iter()
            .map(|p| {
                let (gh, gw) = p.grid();
                (1, gh / p.merge, gw / p.merge)
            })
            .collect()
    }

    pub(crate) fn key_and_positions(&self, prep: &PreparedVision, prompt_ids: &[u32]) -> Result<(Vec<u32>, Qwen3MropePositions)> {
        let ids = [self.spec.image_token_id, self.video_token_id];
        let positions = build_mrope_positions_for_vision_runs(prompt_ids, &ids, &Self::grids(prep))
            .context("the rendered prompt's image_pad runs do not match the request's image_url parts")?;
        let key = prefix_key_with_vision_content(prompt_ids, &ids, &prep.hashes)?;
        Ok((key, positions))
    }

    pub(crate) fn activation_bytes(prep: &PreparedVision) -> u64 {
        let patches = prep.images.iter().map(|p| (p.h / p.patch) * (p.w / p.patch)).max().unwrap_or(0) as u64;
        patches * F32_ACTIVATION_BYTES_PER_PATCH_ABOUT_96_KIB_COUNTING_THE_FC1_GELU_TEMPORARIES
    }

    pub(crate) async fn encode(
        &self,
        prep: &PreparedVision,
        prompt_ids: &[u32],
        key: Vec<u32>,
        positions: Qwen3MropePositions,
        device: &Device,
    ) -> Result<Qwen4ExpMm> {
        let need = TOWER_WEIGHT_BYTES_0_84_GIB + Self::activation_bytes(prep);
        tokio::task::spawn_blocking(move || crate::oapi::cold_tier::ensure_free(need, "qwen4exp-vision-encode", Some(COLD_TIER_NAME)))
            .await
            .ok();
        let tower = self.slot.get().await.context(
            "the qwen4_exp vision tower could not be loaded (see the cold-tier log); NV_Q4EXP_VISION=0 serves images through the OCR bridge",
        )?;
        let t0 = std::time::Instant::now();
        let mut starts = Vec::with_capacity(prep.images.len());
        let mut i = 0usize;
        while i < prompt_ids.len() {
            let id = prompt_ids[i];
            if id == self.spec.image_token_id || id == self.video_token_id {
                starts.push(i);
                while i < prompt_ids.len() && prompt_ids[i] == id {
                    i += 1;
                }
            } else {
                i += 1;
            }
        }
        anyhow::ensure!(starts.len() == prep.images.len(), "{} image runs for {} images", starts.len(), prep.images.len());
        let mut rows = Vec::with_capacity(prep.images.len());
        for (p, at) in prep.images.iter().zip(starts) {
            let px = Tensor::from_vec(p.pixels.clone(), (1, 3, p.h, p.w), device)?;
            let emb = tower.tower.forward(&px)?.to_dtype(DType::BF16)?;
            anyhow::ensure!(emb.dim(0)? == p.merged_tokens(), "tower emitted {} rows for {} image tokens", emb.dim(0)?, p.merged_tokens());
            rows.push((at, emb));
        }
        tracing::info!(
            images = prep.images.len(),
            tokens = rows.iter().map(|(_, r)| r.dim(0).unwrap_or(0)).sum::<usize>(),
            encode_ms = format!("{:.1}", 1000.0 * t0.elapsed().as_secs_f64()).as_str(),
            "qwen4_exp images encoded"
        );
        Ok(Qwen4ExpMm { positions, section: self.section, rows, key })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_defaults_on_and_only_zero_restores_the_bridge() {
        assert!(vision_enabled_from(None));
        assert!(vision_enabled_from(Some("1")));
        assert!(!vision_enabled_from(Some("0")));
    }

    #[test]
    fn max_pixels_defaults_to_2048_squared_and_rejects_garbage() {
        assert_eq!(max_pixels_from(None), 2048 * 2048);
        assert_eq!(max_pixels_from(Some("1048576")), 1 << 20);
        assert_eq!(max_pixels_from(Some("0")), DEFAULT_MAX_PIXELS);
        assert_eq!(max_pixels_from(Some("lots")), DEFAULT_MAX_PIXELS);
    }
}
