use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use tokio::sync::mpsc;

use nv_omni::{
    audio_tokens_for_mel_frames, build_mrope_positions, whisper_log_mel_128, AuTConfig,
    AudioEncoder, ModalitySplice, OmniDeepstack, OmniKvCache, OmniPositions, OmniSpecialIds,
    OmniThinker, OmniThinkerConfig, OmniVisionEncoder,
};

use crate::oapi::chat::{ChatEvent, ChatGenerateRequest};
use crate::oapi::chat_engine::sampling::ChatSampler;
use crate::oapi::chat_engine::spec_window::kv_window;
use crate::oapi::chat_engine::stream::{
    effective_max_new, push_event_async, push_event_blocking, sse_send_timeout, IncrementalDetok,
    StreamEmitter,
};
use crate::oapi::chat_engine::spec_env::acquire_chat_permit;

pub(crate) const OMNI_IMAGE_MARKER: &str = "<|vision_start|><|image_pad|><|vision_end|>";
pub(crate) const OMNI_AUDIO_MARKER: &str = "<|audio_start|><|audio_pad|><|audio_end|>";

pub(crate) struct OmniShared {
    pub thinker: OmniThinker,
    pub vision: OmniVisionEncoder,
    pub audio: AudioEncoder,
    pub ids: OmniSpecialIds,
    pub device: Device,
}

pub(crate) fn build_omni(model_dir: &Path, device: &Device) -> Result<Arc<OmniShared>> {
    let t0 = std::time::Instant::now();
    let config = model_dir.join("config.json");
    let ids = OmniSpecialIds::from_hf_config_json(&config)?;
    let weights = nv_weights::WeightLoader::open_dir(model_dir, device)
        .with_context(|| format!("open omni weights dir {}", model_dir.display()))?;

    let mut thinker = OmniThinker::new(OmniThinkerConfig::from_hf_config_json(&config)?, device)?;
    let n_thinker = thinker.load_weights(&weights)?;
    if n_thinker != 18867 {
        anyhow::bail!("omni thinker loaded {n_thinker} tensors, expected 18867");
    }
    let mut vision = OmniVisionEncoder::from_hf_config_json(&config, device)?;
    let n_vision = vision.load_weights(&weights)?;
    if n_vision != 351 {
        anyhow::bail!("omni vision loaded {n_vision} tensors, expected 351");
    }
    let mut audio = AudioEncoder::new(&AuTConfig::from_hf_config_json(&config)?, device)?;
    let n_audio = audio.load_weights(&weights)?;
    if n_audio != 525 {
        anyhow::bail!("omni audio loaded {n_audio} tensors, expected 525");
    }

    eprintln!(
        "[omni] loaded {} (thinker={n_thinker} vision={n_vision} audio={n_audio}) in {:.1}s",
        model_dir.display(),
        t0.elapsed().as_secs_f32()
    );
    Ok(Arc::new(OmniShared {
        thinker,
        vision,
        audio,
        ids,
        device: device.clone(),
    }))
}

struct MmExpansion {
    tokens: Vec<u32>,
    splices: Vec<ModalitySplice>,
    image_grids: Vec<(usize, usize, usize)>,
    audio_lens: Vec<usize>,
    deepstack: Option<OmniDeepstack>,
}

fn expand_media(model: &OmniShared, tokens: &[u32], req: &ChatGenerateRequest) -> Result<MmExpansion> {
    let ids = &model.ids;
    let device = &model.device;
    let empty_images: Vec<image::RgbImage> = Vec::new();
    let empty_audios: Vec<Vec<f32>> = Vec::new();
    let (images, audios) = match &req.mm {
        Some(m) => (&m.images, &m.audios),
        None => (&empty_images, &empty_audios),
    };

    let mut out = Vec::with_capacity(tokens.len());
    let mut splices = Vec::new();
    let mut image_grids = Vec::new();
    let mut audio_lens = Vec::new();
    let mut ds_rows: Vec<u32> = Vec::new();
    let mut ds_layers: Vec<Vec<Tensor>> = Vec::new();
    let mut img_i = 0usize;
    let mut aud_i = 0usize;

    for &tok in tokens {
        if tok == ids.image_pad {
            let img = images
                .get(img_i)
                .context("prompt has more image markers than image_url parts")?;
            img_i += 1;
            let (w0, h0) = (img.width() as usize, img.height() as usize);
            let (h1, w1) = model.vision.smart_resize(h0, w0);
            let resized = image::imageops::resize(
                img,
                w1 as u32,
                h1 as u32,
                image::imageops::FilterType::CatmullRom,
            );
            let (patches, grid) = model.vision.patchify_rgb(resized.as_raw(), w1, h1, device)?;
            let (emb, deep) = model.vision.forward(&patches, grid)?;
            let n = emb.dim(0)?;
            let pos = out.len();
            out.extend(std::iter::repeat(ids.image_pad).take(n));
            for r in pos..pos + n {
                ds_rows.push(r as u32);
            }
            if ds_layers.is_empty() {
                ds_layers = vec![Vec::new(); deep.len()];
            }
            for (li, d) in deep.into_iter().enumerate() {
                ds_layers[li].push(d);
            }
            image_grids.push((1, grid.1 / 2, grid.2 / 2));
            splices.push(ModalitySplice { position: pos, embedding: emb });
        } else if tok == ids.audio_pad {
            let samples = audios
                .get(aud_i)
                .context("prompt has more audio markers than input_audio parts")?;
            aud_i += 1;
            let (mel, frames) = whisper_log_mel_128(samples)?;
            let mel = Tensor::from_vec(mel, (128, frames), device)?;
            let emb = model.audio.forward(&mel)?;
            let n = emb.dim(0)?;
            if n != audio_tokens_for_mel_frames(frames) {
                anyhow::bail!("audio produced {n} tokens != output-length law");
            }
            let pos = out.len();
            out.extend(std::iter::repeat(ids.audio_pad).take(n));
            audio_lens.push(n);
            splices.push(ModalitySplice { position: pos, embedding: emb });
        } else {
            out.push(tok);
        }
    }
    if img_i != images.len() {
        anyhow::bail!("request carries {} images but prompt has {img_i} image markers", images.len());
    }
    if aud_i != audios.len() {
        anyhow::bail!("request carries {} audios but prompt has {aud_i} audio markers", audios.len());
    }

    let deepstack = if ds_rows.is_empty() {
        None
    } else {
        let rows = Tensor::from_vec(ds_rows.clone(), ds_rows.len(), device)?;
        let mut embeds = Vec::with_capacity(ds_layers.len());
        for layer in ds_layers {
            let refs: Vec<&Tensor> = layer.iter().collect();
            embeds.push(Tensor::cat(&refs, 0)?);
        }
        Some(OmniDeepstack { rows, embeds })
    };

    Ok(MmExpansion {
        tokens: out,
        splices,
        image_grids,
        audio_lens,
        deepstack,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_sampling_omni(
    model: Arc<OmniShared>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    _device: Device,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    tx: &mpsc::Sender<ChatEvent>,
) -> Result<()> {
    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let raw_ids: Vec<u32> = encoded.get_ids().to_vec();

    let expansion = expand_media(&model, &raw_ids, &req)?;
    let tokens = expansion.tokens;
    let prompt_tokens = tokens.len() as u32;

    let max_new = effective_max_new(req.max_new_tokens, default_max_new);
    let Some((_kv_needed, max_new)) = kv_window(tokens.len(), max_new, kv_max_seq_len) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {kv_max_seq_len}-token KV window",
            tokens.len()
        );
    };

    if tx.send(ChatEvent::Started { prompt_tokens }).await.is_err() {
        return Ok(());
    }
    if tokens.is_empty() {
        let _ = tx
            .send(ChatEvent::Done {
                finish_reason: "length".into(),
                completion_tokens: 0,
            })
            .await;
        return Ok(());
    }

    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(&tokens);

    let (positions, mut next_pos) =
        build_mrope_positions(&tokens, &model.ids, &expansion.image_grids, &expansion.audio_lens)?;

    let _permit = acquire_chat_permit().await?;
    let send_timeout = sse_send_timeout();

    let mut cache = OmniKvCache::new(model.thinker.num_layers());
    let mut emitter = StreamEmitter::new(&req.stop);
    let mut detok = IncrementalDetok::new(tokenizer.clone());
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();
    let mut aborted = false;

    tokio::task::block_in_place(|| -> Result<()> {
        let x = model.thinker.embed_with_splices(&tokens, &expansion.splices)?.unsqueeze(0)?;
        let logits = model.thinker.forward_step(
            &x,
            &positions,
            &mut cache,
            expansion.deepstack.as_ref(),
        )?;
        let last_row: Vec<f32> = logits.to_vec1()?;
        let mut sampled = sampler.sample(&last_row);
        if sampled.exhausted {
            anyhow::bail!("no legal token at prefill: sampling mask left every candidate at -inf");
        }
        let mut last = sampled.token;

        for step in 0..max_new {
            if tx.is_closed() {
                aborted = true;
                break;
            }
            if eos_ids.contains(&last) {
                finish_reason = "stop".into();
                break;
            }
            completion_tokens += 1;
            let new_text = detok.push(last)?;
            let (piece, stop_hit) = emitter.step(new_text);
            if !piece.is_empty() {
                let outcome = push_event_blocking(tx, ChatEvent::TextDelta(piece), send_timeout);
                if outcome != crate::oapi::chat_engine::stream::SsePush::Sent {
                    aborted = true;
                    break;
                }
            }
            if stop_hit {
                finish_reason = "stop".into();
                break;
            }
            if step + 1 >= max_new {
                break;
            }

            let xi = model.thinker.embed_with_splices(&[last], &[])?.unsqueeze(0)?;
            let pi = OmniPositions::uniform(&[next_pos]);
            next_pos += 1;
            let logits = model.thinker.forward_step(&xi, &pi, &mut cache, None)?;
            let row: Vec<f32> = logits.to_vec1()?;
            sampled = sampler.sample(&row);
            if sampled.exhausted {
                anyhow::bail!("no legal token at step {step}");
            }
            last = sampled.token;
        }
        Ok(())
    })?;

    if !aborted {
        let _ = push_event_async(
            tx,
            ChatEvent::Done {
                finish_reason,
                completion_tokens,
            },
            send_timeout,
        )
        .await;
    }
    Ok(())
}
