#[cfg(feature = "cuda")]
use super::*;
#[cfg(feature = "cuda")]
pub(crate) fn render_qwen3_5_moe_prompt(messages: &[ChatMessageIn]) -> String {
    let mut out = String::new();
    let mut idx = 0usize;
    if !messages.is_empty() && (messages[0].role == "system" || messages[0].role == "developer") {
        out.push_str("<|im_start|>system\n");
        out.push_str(messages[0].text().trim());
        out.push_str("<|im_end|>\n");
        idx = 1;
    }
    for m in &messages[idx..] {
        let role = if m.role == "developer" {
            "system"
        } else {
            m.role.as_str()
        };
        out.push_str("<|im_start|>");
        out.push_str(role);
        out.push('\n');
        out.push_str(m.text().trim());
        out.push_str("<|im_end|>\n");
    }
    out.push_str("<|im_start|>assistant\n<think>\n");
    out
}

#[cfg(feature = "cuda")]
pub(crate) async fn run_sampling_qwen3_5_moe(
    model: QwenMoeShared,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    moe_dispatch: Option<Arc<nv_models::qwen3_5_moe::GroupedMoeDispatch>>,
    qwen_mtp: Option<Arc<nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    match model {
        QwenMoeShared::Eager(m) => {
            run_sampling_qwen3_5_moe_eager(
                m,
                tokenizer,
                device,
                req,
                kv_max_seq_len,
                default_max_new,
                eos_ids,
                moe_dispatch,
                qwen_mtp,
                tx,
            )
            .await
        }
        QwenMoeShared::Graphed(g) => {
            run_sampling_qwen3_5_moe_graphed(g, tokenizer, req, default_max_new, eos_ids, tx).await
        }
        QwenMoeShared::Batch(sched) => sched.submit(req, tx.clone()),
    }
}

#[cfg(feature = "cuda")]
async fn run_sampling_qwen3_5_moe_graphed(
    model: Arc<tokio::sync::Mutex<nv_models::graph_engine::GraphedQwen3Moe>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    req: ChatGenerateRequest,
    default_max_new: usize,
    eos_ids: &[u32],
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
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

    let mut engine = model.lock().await;
    let window = engine
        .cache()
        .max_seq_len()
        .min(engine.underlying().config().max_position_embeddings);
    let Some((_cache_len, max_new)) = kv_window(prompt_ids.len(), max_new, window) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {window}-token KV window",
            prompt_ids.len()
        );
    };
    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(&prompt_ids);

    engine.reset()?;
    let row = engine.prefill(&prompt_ids)?;
    let mut last_out = sampler.sample(&row);
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
        if engine.current_pos() >= window {
            finish_reason = "length".into();
            break;
        }

        engine.forward_decode(last)?;
        let row = engine.logits_host()?;
        last_out = sampler.sample(&row);
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

#[cfg(feature = "cuda")]
async fn tokenize_and_send_started(
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<Option<Vec<u32>>> {
    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    let prompt_tokens = prompt_ids.len() as u32;
    if tx.send(ChatEvent::Started { prompt_tokens }).await.is_err() {
        return Ok(None);
    }
    if prompt_ids.is_empty() {
        let _ = tx
            .send(ChatEvent::Done {
                finish_reason: "length".into(),
                completion_tokens: 0,
            })
            .await;
        return Ok(None);
    }
    Ok(Some(prompt_ids))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
async fn run_sampling_qwen3_5_moe_eager(
    model: Arc<tokio::sync::Mutex<nv_models::qwen3_5_moe::Qwen3Moe>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    moe_dispatch: Option<Arc<nv_models::qwen3_5_moe::GroupedMoeDispatch>>,
    qwen_mtp: Option<Arc<nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let Some(prompt_ids) = tokenize_and_send_started(&tokenizer, &req, tx).await? else {
        return Ok(());
    };
    let model = model.lock().await;
    run_qwen38_dense_solo_body(
        &model,
        &tokenizer,
        &device,
        &req,
        &prompt_ids,
        kv_max_seq_len,
        default_max_new,
        eos_ids,
        moe_dispatch,
        qwen_mtp.as_deref(),
        tx,
    )
    .await
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_qwen38_solo_via_scheduler(
    model: &nv_models::qwen3_5_moe::Qwen3Moe,
    mtp: Option<&nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    device: &candle_core::Device,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let Some(prompt_ids) = tokenize_and_send_started(tokenizer, &req, tx).await? else {
        return Ok(());
    };
    run_qwen38_dense_solo_body(
        model,
        tokenizer,
        device,
        &req,
        &prompt_ids,
        kv_max_seq_len,
        default_max_new,
        eos_ids,
        None,
        mtp,
        tx,
    )
    .await
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
async fn run_qwen38_dense_solo_body(
    model: &nv_models::qwen3_5_moe::Qwen3Moe,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    device: &candle_core::Device,
    req: &ChatGenerateRequest,
    prompt_ids: &[u32],
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    moe_dispatch: Option<Arc<nv_models::qwen3_5_moe::GroupedMoeDispatch>>,
    qwen_mtp: Option<&nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    use candle_core::Tensor;

    let max_new = effective_max_new(req.max_new_tokens, default_max_new);
    let window = kv_max_seq_len.min(model.config().max_position_embeddings);
    let Some((cache_len, max_new)) = kv_window(prompt_ids.len(), max_new, window) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {window}-token KV window",
            prompt_ids.len()
        );
    };
    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(&prompt_ids);

    if let Some(mtp) = qwen_mtp {
        let k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
        let one_round_must_fit_or_the_normal_loop_serves_the_window_edge =
            prompt_ids.len() + k + 1 <= (cache_len + k).min(window);
        if sampler.fast_greedy() && one_round_must_fit_or_the_normal_loop_serves_the_window_edge {
            anyhow::ensure!(
                moe_dispatch.is_none(),
                "the MTP drafter is loaded only for the dense arm, which never builds a MoE \
                 dispatch; seeing one here means the drafter was attached to the wrong model"
            );
            return run_qwen38_mtp_greedy_rounds(
                &model, mtp, &tokenizer, &req, &prompt_ids, cache_len, window, max_new, eos_ids,
                tx,
            )
            .await;
        }
    }

    let mut cache = model.new_kv_cache(cache_len)?;
    let disp: Option<&dyn nv_models::qwen3_5_moe::MoeDispatch> = moe_dispatch
        .as_deref()
        .map(|d| d as &dyn nv_models::qwen3_5_moe::MoeDispatch);

    let prefill_tokens =
        Tensor::from_vec(prompt_ids.to_vec(), (1usize, prompt_ids.len()), device)?;
    let positions: Vec<i32> = (0..prompt_ids.len() as i32).collect();
    let positions_t = Tensor::from_vec(positions, prompt_ids.len(), &device)?;
    let logits =
        model.forward_with_cache_dispatched(&prefill_tokens, &positions_t, &mut cache, disp)?;
    let last_row = last_row_logits_3d(&logits)?;
    drop(logits);
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
        if cache.current_len() >= cache.max_seq_len() {
            finish_reason = "length".into();
            break;
        }

        let pos = (prompt_ids.len() + step) as i32;
        let next_t = Tensor::from_vec(vec![last], (1usize, 1usize), &device)?;
        let pos_t = Tensor::from_vec(vec![pos], 1usize, &device)?;
        let step_logits = model.forward_with_cache_dispatched(&next_t, &pos_t, &mut cache, disp)?;
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

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
async fn run_qwen38_mtp_greedy_rounds(
    model: &nv_models::qwen3_5_moe::Qwen3Moe,
    mtp: &nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    prompt_ids: &[u32],
    cache_len: usize,
    window: usize,
    max_new: usize,
    eos_ids: &[u32],
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
    let max_seq = (cache_len + k).min(window);
    let mut session = nv_specdecode::qwen38_mtp::Qwen38MtpDecodeSession::start(
        model, mtp, k, prompt_ids, max_seq,
    )?;

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    let mut emitter = StreamEmitter::new(&req.stop);
    let mut detok = IncrementalDetok::new(tokenizer.clone());
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();
    let mut pending: Vec<u32> = vec![session.anchor()];

    'stream: loop {
        for tok in pending.drain(..) {
            if eos_ids.contains(&tok) {
                finish_reason = "stop".into();
                break 'stream;
            }
            generated_ids.push(tok);
            completion_tokens = generated_ids.len() as u32;
            let new_text = detok.push(tok)?;
            let (piece, stop_hit) = emitter.step(new_text);
            if !piece.is_empty() && tx.send(ChatEvent::TextDelta(piece)).await.is_err() {
                return Ok(());
            }
            if stop_hit {
                finish_reason = "stop".into();
                break 'stream;
            }
            if generated_ids.len() >= max_new {
                break 'stream;
            }
        }
        if !session.round_fits() {
            finish_reason = "length".into();
            break 'stream;
        }
        pending = session.round()?;
    }

    let s = &session.stats;
    tracing::info!(
        rounds = s.rounds,
        drafted = s.drafted,
        accepted = s.accepted,
        accept_rate = format!("{:.3}", s.accept_rate()).as_str(),
        tokens_per_round = format!("{:.2}", s.tokens_per_round()).as_str(),
        k,
        completion_tokens,
        "qwen3.8 mtp self-speculative request done"
    );

    if !generated_ids.is_empty() {
        if let Ok(full) =
            crate::oapi::chat_engine::stream::decode_keeping_wire(tokenizer, &generated_ids)
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

#[cfg(feature = "cuda")]
pub(crate) async fn run_sampling_qwen3(
    model: Arc<tokio::sync::Mutex<nv_models::qwen3::Qwen3>>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    use candle_core::Tensor;

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

    let model = model.lock().await;
    let Some((cache_len, max_new)) = kv_window(prompt_ids.len(), max_new, kv_max_seq_len) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {kv_max_seq_len}-token KV window",
            prompt_ids.len()
        );
    };
    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(&prompt_ids);

    let mut cache = model.new_kv_cache(cache_len)?;

    let prefill_tokens = Tensor::from_vec(prompt_ids.clone(), (1usize, prompt_ids.len()), &device)?;
    let positions: Vec<u32> = (0..prompt_ids.len() as u32).collect();
    let positions_t = Tensor::from_vec(positions, prompt_ids.len(), &device)?;
    let logits = model.forward(&prefill_tokens, &positions_t, &mut cache)?;
    let last_row = last_row_logits_3d(&logits)?;
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

        if !piece.is_empty() {
            if tx.send(ChatEvent::TextDelta(piece)).await.is_err() {
                return Ok(());
            }
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

        let pos = (prompt_ids.len() + step) as u32;
        let next = Tensor::from_vec(vec![last], (1usize, 1usize), &device)?;
        let next_pos = Tensor::from_vec(vec![pos], 1usize, &device)?;
        let step_logits = model.forward(&next, &next_pos, &mut cache)?;
        let step_row = last_row_logits_3d(&step_logits)?;
        last_out = sampler.sample(&step_row);
        last = last_out.token;
        if last_out.exhausted {
            anyhow::bail!(
                "no legal token at step {step}: the sampling mask left every candidate at -inf"
            );
        }
        drop(step_logits);
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
