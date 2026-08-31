#[cfg(feature = "cuda")]
use super::*;

#[cfg(feature = "cuda")]
pub(crate) const SOLO_BODY_SENDS_TIME_OUT_BECAUSE_UNDER_NV_Q38_BATCH_1_RUN_SOLO_RUNS_THESE_BODIES_ON_THE_SINGLE_THREAD_DISPATCHER_WHERE_ONE_NON_READING_CLIENT_PARKED_EVERY_QUEUED_JOB_AND_ON_THE_ENGINE_RUNTIME_THEY_HOLD_THE_MODEL_MUTEX_COST_THE_STALLED_CLIENT_LOSES_ITS_STREAM_AFTER_NV_SSE_SEND_TIMEOUT_MS:
    () = ();

#[cfg(feature = "cuda")]
fn solo_send_timeout_via_scheduler_zero_restores_the_old_infinite_wait_matching_sched_push(
) -> Option<std::time::Duration> {
    let () = SOLO_BODY_SENDS_TIME_OUT_BECAUSE_UNDER_NV_Q38_BATCH_1_RUN_SOLO_RUNS_THESE_BODIES_ON_THE_SINGLE_THREAD_DISPATCHER_WHERE_ONE_NON_READING_CLIENT_PARKED_EVERY_QUEUED_JOB_AND_ON_THE_ENGINE_RUNTIME_THEY_HOLD_THE_MODEL_MUTEX_COST_THE_STALLED_CLIENT_LOSES_ITS_STREAM_AFTER_NV_SSE_SEND_TIMEOUT_MS;
    sse_send_timeout_opt_for_the_batch_sched_zero_disables_unlike_the_solo_paths_1ms_clamp()
}

#[cfg(feature = "cuda")]
fn solo_send_timeout_on_the_engine_runtime_zero_clamps_to_1ms_like_the_gemma4_paths(
) -> Option<std::time::Duration> {
    let () = SOLO_BODY_SENDS_TIME_OUT_BECAUSE_UNDER_NV_Q38_BATCH_1_RUN_SOLO_RUNS_THESE_BODIES_ON_THE_SINGLE_THREAD_DISPATCHER_WHERE_ONE_NON_READING_CLIENT_PARKED_EVERY_QUEUED_JOB_AND_ON_THE_ENGINE_RUNTIME_THEY_HOLD_THE_MODEL_MUTEX_COST_THE_STALLED_CLIENT_LOSES_ITS_STREAM_AFTER_NV_SSE_SEND_TIMEOUT_MS;
    Some(sse_send_timeout())
}

#[cfg(feature = "cuda")]
async fn solo_push(
    tx: &mpsc::Sender<ChatEvent>,
    ev: ChatEvent,
    timeout: Option<std::time::Duration>,
) -> SsePush {
    let out = push_event_async_opt(tx, ev, timeout).await;
    log_sse_abort(out, "qwen-solo");
    out
}

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
    qwen_mtp: Option<QwenMtpHead>,
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
            run_sampling_qwen3_5_moe_graphed(
                g,
                tokenizer,
                req,
                default_max_new,
                eos_ids,
                qwen_mtp,
                tx,
            )
            .await
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
    qwen_mtp: Option<QwenMtpHead>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let t_gen = std::time::Instant::now();
    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    let prompt_tokens = prompt_ids.len() as u32;
    crate::oapi::chat::oapi_phase_wall(t_gen, "gen_tokenized");

    let max_new = effective_max_new(req.max_new_tokens, default_max_new);
    let send_timeout = solo_send_timeout_on_the_engine_runtime_zero_clamps_to_1ms_like_the_gemma4_paths();

    if solo_push(tx, ChatEvent::Started { prompt_tokens }, send_timeout).await != SsePush::Sent {
        return Ok(());
    }
    yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;
    if prompt_ids.is_empty() {
        let _ = solo_push(
            tx,
            ChatEvent::Done {
                finish_reason: "length".into(),
                completion_tokens: 0,
            },
            send_timeout,
        )
        .await;
        return Ok(());
    }

    let mut engine = model.lock().await;
    crate::oapi::chat::oapi_phase_wall(t_gen, "gen_engine_lock_acquired");
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

    if let Some(QwenMtpHead::Moe(mtp)) = &qwen_mtp {
        let k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
        let one_round_must_fit_or_the_normal_loop_serves_the_window_edge =
            prompt_ids.len() + k < window;
        if sampler.fast_greedy() && one_round_must_fit_or_the_normal_loop_serves_the_window_edge {
            let session = nv_models::qwen3_5_mtp::MtpMoeGraphedDecodeSession::start(
                &mut engine,
                mtp,
                k,
                &prompt_ids,
            )?;
            return run_qwen36_moe_mtp_greedy_rounds(
                session,
                &tokenizer,
                &req,
                max_new,
                eos_ids,
                k,
                tx,
                send_timeout,
            )
            .await;
        }
    }
    if let Some(QwenMtpHead::Dense(mtp)) = &qwen_mtp {
        let k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
        let one_round_must_fit_or_the_normal_loop_serves_the_window_edge =
            prompt_ids.len() + k < window;
        if sampler.fast_greedy() && one_round_must_fit_or_the_normal_loop_serves_the_window_edge {
            let session = nv_specdecode::qwen38_mtp::Qwen38MtpGraphedDecodeSession::start(
                &mut engine,
                mtp,
                k,
                &prompt_ids,
            )?;
            crate::oapi::chat::oapi_phase_wall(t_gen, "gen_q38_session_started");
            return run_qwen38_mtp_rounds_streaming(
                session,
                &tokenizer,
                &req,
                max_new,
                eos_ids,
                k,
                tx,
                send_timeout,
            )
            .await;
        }
    }

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

        if !piece.is_empty()
            && solo_push(tx, ChatEvent::TextDelta(piece), send_timeout).await != SsePush::Sent
        {
            return Ok(());
        }
        if req.logprobs
            && solo_push(
                tx,
                ChatEvent::Logprob(build_logprob_entry(&tokenizer, &last_out)),
                send_timeout,
            )
            .await
                != SsePush::Sent
        {
            return Ok(());
        }
        yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;

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
                let _ = solo_push(tx, ChatEvent::TextDelta(tail), send_timeout).await;
            }
        }
    }

    let _ = solo_push(
        tx,
        ChatEvent::Done {
            finish_reason,
            completion_tokens,
        },
        send_timeout,
    )
    .await;
    Ok(())
}

#[cfg(feature = "cuda")]
async fn tokenize_and_send_started(
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    tx: &mpsc::Sender<ChatEvent>,
    send_timeout: Option<std::time::Duration>,
) -> anyhow::Result<Option<Vec<u32>>> {
    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    let prompt_tokens = prompt_ids.len() as u32;
    if solo_push(tx, ChatEvent::Started { prompt_tokens }, send_timeout).await != SsePush::Sent {
        return Ok(None);
    }
    yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;
    if prompt_ids.is_empty() {
        let _ = solo_push(
            tx,
            ChatEvent::Done {
                finish_reason: "length".into(),
                completion_tokens: 0,
            },
            send_timeout,
        )
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
    qwen_mtp: Option<QwenMtpHead>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let send_timeout = solo_send_timeout_on_the_engine_runtime_zero_clamps_to_1ms_like_the_gemma4_paths();
    let Some(prompt_ids) = tokenize_and_send_started(&tokenizer, &req, tx, send_timeout).await?
    else {
        return Ok(());
    };
    let model = model.lock().await;
    let disp: Option<&dyn nv_models::qwen3_5_moe::MoeDispatch> = moe_dispatch
        .as_deref()
        .map(|d| d as &dyn nv_models::qwen3_5_moe::MoeDispatch);
    if let Some(QwenMtpHead::Moe(mtp)) = &qwen_mtp {
        return run_qwen36_moe_eager_mtp_or_plain(
            &model,
            &tokenizer,
            &device,
            &req,
            &prompt_ids,
            kv_max_seq_len,
            default_max_new,
            eos_ids,
            disp,
            mtp,
            tx,
            send_timeout,
        )
        .await;
    }
    let dense_mtp = match &qwen_mtp {
        Some(QwenMtpHead::Dense(m)) => Some(m.as_ref()),
        _ => None,
    };
    run_qwen38_dense_solo_body(
        &model,
        &tokenizer,
        &device,
        &req,
        &prompt_ids,
        kv_max_seq_len,
        default_max_new,
        eos_ids,
        disp,
        dense_mtp,
        tx,
        send_timeout,
    )
    .await
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_qwen36_solo_via_scheduler(
    model: &nv_models::qwen3_5_moe::Qwen3Moe,
    mtp: Option<&nv_models::qwen3_5_mtp::MtpHead>,
    moe_dispatch: Option<&dyn nv_models::qwen3_5_moe::MoeDispatch>,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    device: &candle_core::Device,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let send_timeout =
        solo_send_timeout_via_scheduler_zero_restores_the_old_infinite_wait_matching_sched_push();
    let Some(prompt_ids) = tokenize_and_send_started(tokenizer, &req, tx, send_timeout).await?
    else {
        return Ok(());
    };
    if let Some(mtp) = mtp {
        return run_qwen36_moe_eager_mtp_or_plain(
            model,
            tokenizer,
            device,
            &req,
            &prompt_ids,
            kv_max_seq_len,
            default_max_new,
            eos_ids,
            moe_dispatch,
            mtp,
            tx,
            send_timeout,
        )
        .await;
    }
    run_qwen38_dense_solo_body(
        model,
        tokenizer,
        device,
        &req,
        &prompt_ids,
        kv_max_seq_len,
        default_max_new,
        eos_ids,
        moe_dispatch,
        None,
        tx,
        send_timeout,
    )
    .await
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
async fn run_qwen36_moe_eager_mtp_or_plain(
    model: &nv_models::qwen3_5_moe::Qwen3Moe,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    device: &candle_core::Device,
    req: &ChatGenerateRequest,
    prompt_ids: &[u32],
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    moe_dispatch: Option<&dyn nv_models::qwen3_5_moe::MoeDispatch>,
    mtp: &nv_models::qwen3_5_mtp::MtpHead,
    tx: &mpsc::Sender<ChatEvent>,
    send_timeout: Option<std::time::Duration>,
) -> anyhow::Result<()> {
    let max_new = effective_max_new(req.max_new_tokens, default_max_new);
    let window = kv_max_seq_len.min(model.config().max_position_embeddings);
    let Some((cache_len, max_new)) = kv_window(prompt_ids.len(), max_new, window) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {window}-token KV window",
            prompt_ids.len()
        );
    };
    let mut sampler = ChatSampler::for_request(req, tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(prompt_ids);

    let k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
    let max_seq = (cache_len + k).min(window);
    let one_round_must_fit_or_the_normal_loop_serves_the_window_edge =
        prompt_ids.len() + k < max_seq;
    if sampler.fast_greedy() && one_round_must_fit_or_the_normal_loop_serves_the_window_edge {
        let session = nv_models::qwen3_5_mtp::MtpMoeEagerDecodeSession::start(
            model,
            mtp,
            k,
            prompt_ids,
            max_seq,
            moe_dispatch,
        )?;
        return run_qwen36_moe_mtp_greedy_rounds(
            session,
            tokenizer,
            req,
            max_new,
            eos_ids,
            k,
            tx,
            send_timeout,
        )
        .await;
    }
    run_qwen38_dense_solo_body(
        model,
        tokenizer,
        device,
        req,
        prompt_ids,
        kv_max_seq_len,
        default_max_new,
        eos_ids,
        moe_dispatch,
        None,
        tx,
        send_timeout,
    )
    .await
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
async fn run_qwen36_moe_mtp_greedy_rounds<S: nv_models::qwen3_5_mtp::MtpMoeServingRounds>(
    mut session: S,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    max_new: usize,
    eos_ids: &[u32],
    k: usize,
    tx: &mpsc::Sender<ChatEvent>,
    send_timeout: Option<std::time::Duration>,
) -> anyhow::Result<()> {
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
            if !piece.is_empty()
                && solo_push(tx, ChatEvent::TextDelta(piece), send_timeout).await != SsePush::Sent
            {
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
        yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;
        if !session.round_fits() {
            finish_reason = "length".into();
            break 'stream;
        }
        pending = session.round()?;
    }

    let s = session.stats();
    let p_min = nv_models::qwen3_5_mtp::q36_mtp_draft_p_min_from_env()
        .map(|p| format!("{p:.2}"))
        .unwrap_or_else(|| "off".into());
    tracing::info!(
        arm = session.arm_name(),
        capture_active = session.capture_active(),
        rounds = s.rounds,
        drafted = s.drafted,
        accepted = s.accepted,
        accept_rate = format!("{:.3}", s.accept_rate()).as_str(),
        tokens_per_round = format!("{:.2}", s.tokens_per_round()).as_str(),
        k,
        p_min = p_min.as_str(),
        completion_tokens,
        "qwen3.6 moe mtp self-speculative request done"
    );

    if !generated_ids.is_empty() {
        if let Ok(full) =
            crate::oapi::chat_engine::stream::decode_keeping_wire(tokenizer, &generated_ids)
        {
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                let _ = solo_push(tx, ChatEvent::TextDelta(tail), send_timeout).await;
            }
        }
    }

    let _ = solo_push(
        tx,
        ChatEvent::Done {
            finish_reason,
            completion_tokens,
        },
        send_timeout,
    )
    .await;
    Ok(())
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
    let send_timeout =
        solo_send_timeout_via_scheduler_zero_restores_the_old_infinite_wait_matching_sched_push();
    let Some(prompt_ids) = tokenize_and_send_started(tokenizer, &req, tx, send_timeout).await?
    else {
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
        send_timeout,
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
    moe_dispatch: Option<&dyn nv_models::qwen3_5_moe::MoeDispatch>,
    qwen_mtp: Option<&nv_specdecode::qwen38_mtp::Qwen38DenseMtpHead>,
    tx: &mpsc::Sender<ChatEvent>,
    send_timeout: Option<std::time::Duration>,
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
    let mut sampler = ChatSampler::for_request(req, tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(prompt_ids);

    if let Some(mtp) = qwen_mtp {
        let k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
        let one_round_must_fit_or_the_normal_loop_serves_the_window_edge =
            prompt_ids.len() + k < (cache_len + k).min(window);
        if sampler.fast_greedy() && one_round_must_fit_or_the_normal_loop_serves_the_window_edge {
            anyhow::ensure!(
                moe_dispatch.is_none(),
                "the MTP drafter is loaded only for the dense arm, which never builds a MoE \
                 dispatch; seeing one here means the drafter was attached to the wrong model"
            );
            return run_qwen38_mtp_greedy_rounds(
                model,
                mtp,
                tokenizer,
                req,
                prompt_ids,
                cache_len,
                window,
                max_new,
                eos_ids,
                tx,
                send_timeout,
            )
            .await;
        }
    }

    let mut cache = model.new_kv_cache(cache_len)?;
    let disp = moe_dispatch;

    let logits = match nv_models::qwen3_5_moe::q38_prefill_chunk_tokens_from_env()
        .filter(|&c| prompt_ids.len() > c)
    {
        Some(chunk) => model
            .prefill_chunks_dispatched_serving_last_row_logits_bounding_transients_to_one_chunk(
                prompt_ids, chunk, &mut cache, disp,
            )?,
        None => {
            let prefill_tokens =
                Tensor::from_vec(prompt_ids.to_vec(), (1usize, prompt_ids.len()), device)?;
            let positions: Vec<i32> = (0..prompt_ids.len() as i32).collect();
            let positions_t = Tensor::from_vec(positions, prompt_ids.len(), device)?;
            model.forward_with_cache_dispatched(&prefill_tokens, &positions_t, &mut cache, disp)?
        }
    };
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

        if !piece.is_empty()
            && solo_push(tx, ChatEvent::TextDelta(piece), send_timeout).await != SsePush::Sent
        {
            return Ok(());
        }
        if req.logprobs
            && solo_push(
                tx,
                ChatEvent::Logprob(build_logprob_entry(tokenizer, &last_out)),
                send_timeout,
            )
            .await
                != SsePush::Sent
        {
            return Ok(());
        }
        yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;

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
        let next_t = Tensor::from_vec(vec![last], (1usize, 1usize), device)?;
        let pos_t = Tensor::from_vec(vec![pos], 1usize, device)?;
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
        if let Ok(full) = crate::oapi::chat_engine::stream::decode_keeping_wire(tokenizer, &generated_ids) {
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                let _ = solo_push(tx, ChatEvent::TextDelta(tail), send_timeout).await;
            }
        }
    }

    let _ = solo_push(
        tx,
        ChatEvent::Done {
            finish_reason,
            completion_tokens,
        },
        send_timeout,
    )
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
    send_timeout: Option<std::time::Duration>,
) -> anyhow::Result<()> {
    let k = nv_specdecode::qwen38_mtp::mtp_chain_depth_from_env();
    let max_seq = (cache_len + k).min(window);
    let session = nv_specdecode::qwen38_mtp::Qwen38MtpDecodeSession::start(
        model, mtp, k, prompt_ids, max_seq,
    )?;
    run_qwen38_mtp_rounds_streaming(
        session,
        tokenizer,
        req,
        max_new,
        eos_ids,
        k,
        tx,
        send_timeout,
    )
    .await
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
async fn run_qwen38_mtp_rounds_streaming<S: nv_models::qwen3_5_mtp::MtpMoeServingRounds>(
    mut session: S,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    max_new: usize,
    eos_ids: &[u32],
    k: usize,
    tx: &mpsc::Sender<ChatEvent>,
    send_timeout: Option<std::time::Duration>,
) -> anyhow::Result<()> {
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
            if !piece.is_empty()
                && solo_push(tx, ChatEvent::TextDelta(piece), send_timeout).await != SsePush::Sent
            {
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
        yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;
        if !session.round_fits() {
            finish_reason = "length".into();
            break 'stream;
        }
        pending = session.round()?;
    }

    let s = session.stats();
    tracing::info!(
        arm = session.arm_name(),
        capture_active = session.capture_active(),
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
                let _ = solo_push(tx, ChatEvent::TextDelta(tail), send_timeout).await;
            }
        }
    }

    let _ = solo_push(
        tx,
        ChatEvent::Done {
            finish_reason,
            completion_tokens,
        },
        send_timeout,
    )
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
    let send_timeout = solo_send_timeout_on_the_engine_runtime_zero_clamps_to_1ms_like_the_gemma4_paths();

    if solo_push(tx, ChatEvent::Started { prompt_tokens }, send_timeout).await != SsePush::Sent {
        return Ok(());
    }
    yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;

    if prompt_ids.is_empty() {
        let _ = solo_push(
            tx,
            ChatEvent::Done {
                finish_reason: "length".into(),
                completion_tokens: 0,
            },
            send_timeout,
        )
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

        if !piece.is_empty()
            && solo_push(tx, ChatEvent::TextDelta(piece), send_timeout).await != SsePush::Sent
        {
            return Ok(());
        }
        if req.logprobs
            && solo_push(
                tx,
                ChatEvent::Logprob(build_logprob_entry(&tokenizer, &last_out)),
                send_timeout,
            )
            .await
                != SsePush::Sent
        {
            return Ok(());
        }
        yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;

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
                let _ = solo_push(tx, ChatEvent::TextDelta(tail), send_timeout).await;
            }
        }
    }

    let _ = solo_push(
        tx,
        ChatEvent::Done {
            finish_reason,
            completion_tokens,
        },
        send_timeout,
    )
    .await;
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod solo_send_discipline_tests {
    use super::*;

    #[tokio::test]
    async fn a_stalled_channel_times_out_solo_push_instead_of_parking_the_calling_runtime() {
        let (tx, _rx_held_open_and_never_read) = mpsc::channel::<ChatEvent>(1);
        assert_eq!(
            solo_push(
                &tx,
                ChatEvent::TextDelta("fill".into()),
                Some(std::time::Duration::from_millis(10)),
            )
            .await,
            SsePush::Sent
        );
        let t0 = std::time::Instant::now();
        assert_eq!(
            solo_push(
                &tx,
                ChatEvent::TextDelta("blocked".into()),
                Some(std::time::Duration::from_millis(60)),
            )
            .await,
            SsePush::TimedOut
        );
        let dt = t0.elapsed();
        assert!(
            dt >= std::time::Duration::from_millis(55),
            "returned before the deadline: {dt:?}"
        );
        assert!(dt < std::time::Duration::from_secs(5), "overslept: {dt:?}");
    }

    #[tokio::test]
    async fn a_closed_channel_aborts_solo_push_immediately_even_on_the_infinite_wait_hatch() {
        let (tx, rx) = mpsc::channel::<ChatEvent>(1);
        drop(rx);
        assert_eq!(
            solo_push(&tx, ChatEvent::TextDelta("a".into()), None).await,
            SsePush::Closed
        );
    }

    #[test]
    fn the_scheduler_hatch_maps_zero_to_infinite_while_the_engine_hatch_never_does() {
        assert_eq!(
            sse_timeout_opt_from_ms_zero_keeps_the_old_infinite_wait(0),
            None
        );
        assert_eq!(
            sse_timeout_opt_from_ms_zero_keeps_the_old_infinite_wait(10_000),
            Some(std::time::Duration::from_millis(10_000))
        );
        if std::env::var_os("NV_SSE_SEND_TIMEOUT_MS").is_none() {
            assert_eq!(
                solo_send_timeout_via_scheduler_zero_restores_the_old_infinite_wait_matching_sched_push(),
                Some(std::time::Duration::from_millis(10_000))
            );
            assert_eq!(
                solo_send_timeout_on_the_engine_runtime_zero_clamps_to_1ms_like_the_gemma4_paths(),
                Some(std::time::Duration::from_millis(10_000))
            );
        }
    }
}
