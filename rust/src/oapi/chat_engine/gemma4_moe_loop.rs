#[cfg(feature = "cuda")]
use super::*;

#[cfg(feature = "cuda")]
const PREFILL_CHUNK: usize = 256;

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_sampling_gemma4_moe(
    model: Arc<nv_models::gemma4_moe::Gemma4Moe>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    bos_token_id: Option<u32>,
    mm_towers: Option<Arc<crate::oapi::chat_multimodal::Gemma4MmTowers>>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    use candle_core::Tensor;

    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    splice_bos_at_position_0_only(&mut prompt_ids, bos_token_id, eos_ids);

    let mm_embeds = match req
        .mm
        .as_ref()
        .filter(|m| !(m.images.is_empty() && m.audios.is_empty()))
    {
        Some(media) => {
            let towers = mm_towers.as_deref().ok_or_else(|| {
                anyhow::anyhow!(
                    "request carries image/audio parts but this engine loaded no mm towers"
                )
            })?;
            let plan = crate::oapi::chat_multimodal::plan_from_marked_tokens(
                towers,
                &prompt_ids,
                media,
                &device,
            )?;
            let embeds = crate::oapi::chat_multimodal::mm_embeddings(
                towers,
                &plan,
                model.embed_weight(),
                model.embed_scale() as f64,
            )?;
            prompt_ids = plan.tokens;
            Some(embeds)
        }
        None => None,
    };
    let prompt_tokens = prompt_ids.len() as u32;
    let max_new = effective_max_new(req.max_new_tokens, default_max_new);

    if tx.send(ChatEvent::Started { prompt_tokens }).await.is_err() {
        return Ok(());
    }
    if prompt_ids.is_empty() {
        let _ = tx
            .send(ChatEvent::Done {
                finish_reason: "length".into(),
                completion_tokens: 0,
            })
            .await;
        return Ok(());
    }

    let Some((cache_len, max_new)) = kv_window(prompt_ids.len(), max_new, kv_max_seq_len) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {kv_max_seq_len}-token KV window",
            prompt_ids.len()
        );
    };
    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(&prompt_ids);

    let mut cache = model.new_kv_cache(cache_len)?;

    let mut last_logits = None;
    let mut off = 0usize;
    while off < prompt_ids.len() {
        let c = (prompt_ids.len() - off).min(PREFILL_CHUNK);
        let chunk = &prompt_ids[off..off + c];
        let tokens_t = Tensor::from_vec(chunk.to_vec(), (1usize, c), &device)?;
        let pos: Vec<i32> = (off as i32..(off + c) as i32).collect();
        let pos_t = Tensor::from_vec(pos, c, &device)?;
        last_logits = Some(match &mm_embeds {
            Some(e) => model.forward_with_cache_embeds(
                &tokens_t,
                &e.narrow(0, off, c)?,
                &pos_t,
                &mut cache,
            )?,
            None => model.forward_with_cache(&tokens_t, &pos_t, &mut cache)?,
        });
        off += c;
    }
    let last_row = last_row_logits_3d(&last_logits.expect("non-empty prompt"))?;
    let mut last_out = sampler.sample(&last_row);
    let mut last = last_out.token;
    if last_out.exhausted {
        anyhow::bail!("no legal token at prefill: the sampling mask left every candidate at -inf");
    }

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    let mut emitter = StreamEmitter::new(&req.stop);
    let mut detok = IncrementalDetok::new(tokenizer.clone());
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();

    for step in 0..max_new {
        if eos_ids.contains(&last) {
            finish_reason = "stop".into();
            break;
        }
        generated_ids.push(last);
        completion_tokens = generated_ids.len() as u32;

        let new_text = detok.push(last)?;
        let (piece, stop_hit) = emitter.step(new_text);
        if !piece.is_empty() && tx.send(ChatEvent::TextDelta(piece)).await.is_err() {
            return Ok(());
        }
        if req.logprobs
            && tx
                .send(ChatEvent::Logprob(build_logprob_entry(
                    &tokenizer, &last_out,
                )))
                .await
                .is_err()
        {
            return Ok(());
        }

        if stop_hit {
            finish_reason = "stop".into();
            break;
        }
        if step + 1 >= max_new {
            break;
        }
        if cache.current_len() >= cache_len {
            finish_reason = "length".into();
            break;
        }

        let pos = (prompt_ids.len() + step) as i32;
        let next_t = Tensor::from_vec(vec![last], (1usize, 1usize), &device)?;
        let pos_t = Tensor::from_vec(vec![pos], 1usize, &device)?;
        let step_logits = model.forward_with_cache(&next_t, &pos_t, &mut cache)?;
        let step_row = last_row_logits_3d(&step_logits)?;
        drop(step_logits);
        last_out = sampler.sample(&step_row);
        last = last_out.token;
        if last_out.exhausted {
            anyhow::bail!(
                "no legal token at step {step}: the sampling mask left every candidate at -inf"
            );
        }
    }

    if !generated_ids.is_empty() {
        if let Ok(full) = crate::oapi::chat_engine::stream::decode_keeping_wire(&tokenizer, &generated_ids) {
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                let _ = tx.send(ChatEvent::TextDelta(tail)).await;
            }
        }
    }

    let _ = tx
        .send(ChatEvent::Done {
            finish_reason,
            completion_tokens,
        })
        .await;
    Ok(())
}
