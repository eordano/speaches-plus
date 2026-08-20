#[cfg(feature = "cuda")]
use super::*;

#[cfg(feature = "cuda")]
pub(crate) const GPT_OSS_CUDA_PREFILL_CHUNK_IS_SMALL_BECAUSE_EAGER_SCORES_ARE_M_TIMES_T: &str =
    "the cuda gpt-oss forward scores eagerly (sinks rule out flash), so one prefill chunk \
     materializes heads * chunk * context f32 scores plus the same again as probabilities. At \
     openai/gpt-oss-20b's 64 heads that is 64 * chunk * context * 8 bytes: a 128-token chunk \
     against an 8k context is about 0.5 GB of transient scratch, and doubling the chunk doubles \
     it. 128 is the chunk that keeps the transient under a gigabyte at the contexts this opt-in \
     is expected to serve.";

#[cfg(feature = "cuda")]
const PREFILL_CHUNK: usize = 128;

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_sampling_gpt_oss(
    model: Arc<nv_models::gpt_oss_cuda::GptOssCuda>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let _ = GPT_OSS_CUDA_PREFILL_CHUNK_IS_SMALL_BECAUSE_EAGER_SCORES_ARE_M_TIMES_T;
    if req
        .mm
        .as_ref()
        .is_some_and(|m| !(m.images.is_empty() && m.audios.is_empty()))
    {
        anyhow::bail!(
            "gpt-oss is a text-only checkpoint on both backends; image and audio parts have no \
             tower to encode them"
        );
    }

    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
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
    let mut logits = Vec::new();
    let mut off = 0usize;
    while off < prompt_ids.len() {
        let c = (prompt_ids.len() - off).min(PREFILL_CHUNK);
        let positions: Vec<u32> = (off as u32..(off + c) as u32).collect();
        logits = model.forward_last_logits(&prompt_ids[off..off + c], &positions, &mut cache)?;
        off += c;
    }

    let mut last_out = sampler.sample(&logits);
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

        let pos = (prompt_ids.len() + step) as u32;
        let step_logits = model.forward_last_logits(&[last], &[pos], &mut cache)?;
        last_out = sampler.sample(&step_logits);
        last = last_out.token;
        if last_out.exhausted {
            anyhow::bail!(
                "no legal token at step {step}: the sampling mask left every candidate at -inf"
            );
        }
    }

    if !generated_ids.is_empty() {
        if let Ok(full) =
            crate::oapi::chat_engine::stream::decode_keeping_wire(&tokenizer, &generated_ids)
        {
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
