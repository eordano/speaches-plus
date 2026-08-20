use std::path::Path;

use anyhow::{Context as _, Result};
use candle_core::{DType, Device};

use nv_models::embed_row_splice::{rows_to_bf16, EmbedRowSplice};
use nv_models::gemma4_mm_splice::{placeholder_runs, Modality};

use crate::oapi::chat_multimodal::{
    plan_from_marked_tokens, run_towers, Gemma4MmTowers, MmMedia, MmPlan,
};
use crate::oapi::chat_multimodal_qwen3::{Qwen3MmSpec, Qwen3VisionMm};

use super::WgpuModelKind;

pub const EVERY_KIND_EITHER_SERVES_MEDIA_OR_NAMES_ITSELF_IN_A_REFUSAL: &str = "a wgpu kind that cannot splice embedding rows must never fall through to a text-only prefill of a prompt whose image and audio parts were replaced by marker tokens: the model would read the literal marker and answer about nothing. Every arm of embed_row_route and refuse_media therefore returns a reason that names the kind and says which half is missing -- the decoder entry, the tower, or the checkpoint config.";

pub const GEMMA4_TOWER_ROWS_ARE_ALREADY_HIDDEN_SPACE_SO_THE_SPLICE_LANDS_AFTER_THE_EMBED_SCALE: &str = "chat_multimodal::mm_embeddings scales the TEXT rows by embed_scale and leaves the tower rows untouched, so the wgpu splice must overwrite gathered rows after the embed-scale pass, which is exactly where gemma4_wgpu::EMBED_ROW_SPLICE_ENTRY and its gemma4_moe twin run. That makes the wgpu rows the same numbers the cuda gemma4 path feeds forward_with_cache_last_embeds.";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmRoute {
    Qwen3Vision,
    Gemma4Towers,
}

pub fn embed_row_route(kind: WgpuModelKind) -> Result<MmRoute, &'static str> {
    match kind {
        WgpuModelKind::Qwen3_5Dense => Ok(MmRoute::Qwen3Vision),
        WgpuModelKind::GptOss => Err(
            "gpt-oss (nv-models::gpt_oss_wgpu) takes embedding rows in its prefill graph \
             (prefill_tokens_with_image_rows, pinned by \
             crates/nv-models/tests/gpt_oss_wgpu_verify.rs) but no gpt-oss checkpoint ships a \
             vision_config or an audio_config and this tree carries no tower that emits \
             gpt-oss-shaped rows, so there is nothing to splice.",
        ),
        WgpuModelKind::Gemma4Dense | WgpuModelKind::Gemma4Moe => Ok(MmRoute::Gemma4Towers),
        WgpuModelKind::Gemma4E4b => Err(
            "gemma4-e4b (nv-models::gemma4_e4b_wgpu) ships vision AND audio towers in its \
             checkpoint, but its wgpu decoder exposes no embed-row prefill entry \
             (prefill_tokens_with_embed_rows): media rows have no way into its prefill graph. \
             Serve this checkpoint's images/audio on the cuda gemma4 engine, or land the e4b \
             embed-row splice next to the gemma4_wgpu one first.",
        ),
        WgpuModelKind::Qwen3_5Moe => Err(
            "qwen3.5-moe (nv-models::qwen3_5_moe_wgpu) has a decoder-level embed-row splice \
             (prefill_with_splices, pinned by crates/nv-models/tests/qwen36_moe_vision_splice.rs) \
             but the serving seam loads no vision tower for this kind and no image oracle has \
             been recorded against a real qwen3.6-moe checkpoint, so serving it would be an \
             unmeasured path.",
        ),
        WgpuModelKind::Laguna => Err(
            "laguna-xs (nv-models::laguna_wgpu) is a text-only checkpoint: it carries no \
             vision_config and no audio_config, so there is no tower to turn media into rows.",
        ),
    }
}

pub fn media_present(media: &MmMedia) -> bool {
    !media.images.is_empty() || !media.audios.is_empty()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmCaps {
    pub images: bool,
    pub audio: bool,
}

pub fn refuse_media(
    kind: WgpuModelKind,
    caps: Option<MmCaps>,
    has_images: bool,
    has_audio: bool,
) -> Option<String> {
    if !has_images && !has_audio {
        return None;
    }
    let label = kind.label();
    if let Err(reason) = embed_row_route(kind) {
        return Some(format!("image and audio parts cannot be served here: {reason}"));
    }
    let Some(caps) = caps else {
        return Some(format!(
            "{label} loaded no multimodal tower for this checkpoint, so image and audio parts \
             cannot be encoded; the request would otherwise be served as text-only with the \
             media markers left in the prompt"
        ));
    };
    if has_images && !caps.images {
        return Some(format!(
            "{label} loaded no vision tower for this checkpoint (config.json has no usable \
             vision_config), so image parts cannot be encoded"
        ));
    }
    if has_audio && !caps.audio {
        return Some(format!(
            "{label} loaded no audio tower for this checkpoint (config.json audio_config is \
             null or the tower is disabled by NV_MM_TOWERS), so input_audio parts cannot be \
             encoded"
        ));
    }
    None
}

pub const A_GEMMA4_TOWER_THAT_FAILS_TO_LOAD_MUST_NOT_TAKE_DOWN_TEXT_SERVING: &str = "every gemma4 checkpoint this seam serves carries a vision_config, so probing it turns the mm towers into a load-time dependency of a route that was text-only until now. A tower that fails to load therefore leaves the engine up with no runtime and refuse_media answers every media request by name; the qwen3 route keeps its original fatal behaviour because its tower load is the one that has been exercised end to end.";

pub enum MmSpec {
    Qwen3(Box<Qwen3MmSpec>),
    Gemma4 { text_hidden: Option<usize> },
}

impl MmSpec {
    pub fn tower_load_failure_is_fatal(&self) -> bool {
        matches!(self, Self::Qwen3(_))
    }
}

pub fn text_hidden_size(raw_cfg: &str) -> Option<usize> {
    let v: serde_json::Value = serde_json::from_str(raw_cfg).ok()?;
    v.get("text_config")
        .and_then(|t| t.get("hidden_size"))
        .or_else(|| v.get("hidden_size"))
        .and_then(|h| h.as_u64())
        .map(|h| h as usize)
}

pub fn detect(
    kind: WgpuModelKind,
    model_dir: &Path,
    raw_cfg: &str,
    tokenizer: &tokenizers::Tokenizer,
) -> Result<Option<MmSpec>> {
    match embed_row_route(kind) {
        Err(_) => Ok(None),
        Ok(MmRoute::Qwen3Vision) => Ok(Qwen3MmSpec::from_model_dir(model_dir, raw_cfg, tokenizer)?
            .map(|s| MmSpec::Qwen3(Box::new(s)))),
        Ok(MmRoute::Gemma4Towers) => {
            let v: serde_json::Value = serde_json::from_str(raw_cfg)
                .context("parse config.json for the gemma4 wgpu multimodal probe")?;
            let present = |k: &str| !matches!(v.get(k), None | Some(serde_json::Value::Null));
            if present("vision_config") || present("audio_config") {
                Ok(Some(MmSpec::Gemma4 {
                    text_hidden: text_hidden_size(raw_cfg),
                }))
            } else {
                Ok(None)
            }
        }
    }
}

pub struct Gemma4Mm {
    towers: Gemma4MmTowers,
    device: Device,
    text_hidden: Option<usize>,
}

impl Gemma4Mm {
    pub fn load(model_dir: &Path, text_hidden: Option<usize>) -> Result<Self> {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0).context("gemma4 mm towers need a cuda device")?;
        #[cfg(not(feature = "cuda"))]
        let device = Device::Cpu;
        let towers = Gemma4MmTowers::from_model_dir(model_dir, &device)
            .context("load gemma4 mm towers for the wgpu decoder")?;
        Ok(Self {
            towers,
            device,
            text_hidden,
        })
    }

    pub fn caps(&self) -> MmCaps {
        MmCaps {
            images: self.towers.vision.is_some(),
            audio: self.towers.audio.is_some(),
        }
    }

    pub fn plan(&self, prompt_ids: &[u32], media: &MmMedia) -> Result<MmPlan> {
        plan_from_marked_tokens(&self.towers, prompt_ids, media, &self.device)
    }

    pub fn embed_rows(&self, plan: &MmPlan) -> Result<Vec<EmbedRowSplice>> {
        let items = run_towers(&self.towers, plan, DType::F32)?;
        let runs = placeholder_runs(&plan.tokens);
        anyhow::ensure!(
            items.len() == runs.len(),
            "gemma4 towers produced {} embedding item(s) for {} placeholder run(s)",
            items.len(),
            runs.len()
        );
        let mut order: Vec<usize> = (0..items.len()).collect();
        order.sort_by_key(|&i| items[i].position);
        let mut out = Vec::with_capacity(runs.len());
        for (run, &idx) in runs.iter().zip(order.iter()) {
            let item = &items[idx];
            anyhow::ensure!(
                item.position == run.start && item.modality == run.modality,
                "gemma4 {:?} embedding at position {} does not align with the {:?} placeholder \
                 run starting at {}",
                item.modality,
                item.position,
                run.modality,
                run.start
            );
            let (rows, hidden) = item.embedding.dims2().with_context(|| {
                format!(
                    "gemma4 {:?} embedding at position {} must be [rows, hidden]",
                    run.modality, run.start
                )
            })?;
            if let Some(want) = self.text_hidden {
                anyhow::ensure!(
                    hidden == want,
                    "gemma4 {:?} tower emits {hidden}-wide rows but the text decoder is {want} \
                     wide; a splice of the wrong width would land as a partial run of rows \
                     instead of failing",
                    run.modality
                );
            }
            anyhow::ensure!(
                rows == run.len,
                "gemma4 {:?} tower produced {rows} row(s) for a placeholder run of {} at {}",
                run.modality,
                run.len,
                run.start
            );
            let flat: Vec<f32> = item
                .embedding
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            out.push(EmbedRowSplice {
                position: run.start,
                rows_bf16: rows_to_bf16(&flat),
            });
        }
        Ok(out)
    }

    pub fn modality_of_token(&self, id: u32) -> Option<Modality> {
        match id {
            nv_models::gemma4_mm_splice::GEMMA4_IMAGE_TOKEN_ID => Some(Modality::Image),
            nv_models::gemma4_mm_splice::GEMMA4_AUDIO_TOKEN_ID => Some(Modality::Audio),
            _ => None,
        }
    }
}

pub enum MmRuntime {
    Qwen3(Box<Qwen3VisionMm>),
    Gemma4(Box<Gemma4Mm>),
}

impl MmRuntime {
    pub fn load(spec: MmSpec, model_dir: &Path) -> Result<Self> {
        match spec {
            MmSpec::Qwen3(s) => Ok(Self::Qwen3(Box::new(Qwen3VisionMm::load(*s, model_dir)?))),
            MmSpec::Gemma4 { text_hidden } => Ok(Self::Gemma4(Box::new(Gemma4Mm::load(
                model_dir,
                text_hidden,
            )?))),
        }
    }

    pub fn caps(&self) -> MmCaps {
        match self {
            Self::Qwen3(_) => MmCaps {
                images: true,
                audio: false,
            },
            Self::Gemma4(g) => g.caps(),
        }
    }

    pub fn qwen3(&self) -> Option<&Qwen3VisionMm> {
        match self {
            Self::Qwen3(m) => Some(m),
            Self::Gemma4(_) => None,
        }
    }

    pub fn gemma4(&self) -> Option<&Gemma4Mm> {
        match self {
            Self::Gemma4(m) => Some(m),
            Self::Qwen3(_) => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL_KINDS: [WgpuModelKind; 7] = [
        WgpuModelKind::Gemma4Dense,
        WgpuModelKind::Gemma4E4b,
        WgpuModelKind::Gemma4Moe,
        WgpuModelKind::Qwen3_5Moe,
        WgpuModelKind::Qwen3_5Dense,
        WgpuModelKind::GptOss,
        WgpuModelKind::Laguna,
    ];

    fn kind_word(kind: WgpuModelKind) -> &'static str {
        kind.label()
            .split_whitespace()
            .next()
            .expect("kind labels start with the kind name")
    }

    #[test]
    fn every_kind_that_refuses_media_names_itself_and_says_what_is_missing() {
        for kind in ALL_KINDS {
            let Err(reason) = embed_row_route(kind) else {
                continue;
            };
            assert!(
                reason.contains(kind_word(kind)),
                "{kind:?} refusal does not name the kind: {reason}"
            );
            assert!(
                reason.contains("decoder")
                    || reason.contains("tower")
                    || reason.contains("config"),
                "{kind:?} refusal does not say which half is missing: {reason}"
            );
        }
    }

    #[test]
    fn a_media_request_to_a_kind_without_a_splice_is_refused_not_degraded() {
        for kind in ALL_KINDS {
            if embed_row_route(kind).is_ok() {
                continue;
            }
            for (img, aud) in [(true, false), (false, true), (true, true)] {
                let refusal = refuse_media(kind, None, img, aud)
                    .unwrap_or_else(|| panic!("{kind:?} silently accepted media"));
                assert!(
                    refusal.contains(kind_word(kind)),
                    "{kind:?} refusal does not name the kind: {refusal}"
                );
            }
        }
    }

    #[test]
    fn a_text_only_request_is_never_refused_on_any_kind() {
        for kind in ALL_KINDS {
            assert!(refuse_media(kind, None, false, false).is_none());
            let caps = MmCaps {
                images: true,
                audio: true,
            };
            assert!(refuse_media(kind, Some(caps), false, false).is_none());
        }
    }

    #[test]
    fn a_kind_with_a_splice_still_refuses_the_modality_its_towers_cannot_encode() {
        let images_only = MmCaps {
            images: true,
            audio: false,
        };
        for kind in ALL_KINDS {
            if embed_row_route(kind).is_err() {
                continue;
            }
            assert!(
                refuse_media(kind, Some(images_only), true, false).is_none(),
                "{kind:?} refused an image it can encode"
            );
            let refusal = refuse_media(kind, Some(images_only), false, true)
                .unwrap_or_else(|| panic!("{kind:?} accepted audio with no audio tower"));
            assert!(refusal.contains("audio"), "{refusal}");
            assert!(
                refusal.contains(kind_word(kind)),
                "{kind:?} audio refusal does not name the kind: {refusal}"
            );
            let both = refuse_media(kind, Some(images_only), true, true)
                .unwrap_or_else(|| panic!("{kind:?} accepted audio with no audio tower"));
            assert!(both.contains("audio"), "{both}");
        }
    }

    #[test]
    fn a_loaded_runtime_with_no_tower_at_all_refuses_before_the_missing_tower_message() {
        let none = MmCaps {
            images: false,
            audio: false,
        };
        let refusal = refuse_media(WgpuModelKind::Gemma4Dense, Some(none), true, false)
            .expect("no vision tower must refuse");
        assert!(refusal.contains("vision_config"), "{refusal}");
    }

    #[test]
    fn the_qwen3_runtime_reports_images_without_audio() {
        let caps = MmCaps {
            images: true,
            audio: false,
        };
        assert!(refuse_media(WgpuModelKind::Qwen3_5Dense, Some(caps), true, false).is_none());
        let refusal = refuse_media(WgpuModelKind::Qwen3_5Dense, Some(caps), true, true)
            .expect("qwen3.8 has no audio tower");
        assert!(refusal.contains("qwen3.5-dense"), "{refusal}");
    }
}
