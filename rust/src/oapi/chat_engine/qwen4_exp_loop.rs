#[cfg(feature = "cuda")]
use super::*;

#[cfg(any(feature = "cuda", test))]
pub(super) async fn acquire_qwen4_exp_permit(
    queue: std::time::Duration,
) -> anyhow::Result<tokio::sync::OwnedSemaphorePermit> {
    static GATE: std::sync::OnceLock<std::sync::Arc<tokio::sync::Semaphore>> =
        std::sync::OnceLock::new();
    let sem = GATE.get_or_init(|| {
        tracing::info!(
            permits = 1,
            "qwen4_exp chat serialized across session reuse and CUDA graph capture"
        );
        std::sync::Arc::new(tokio::sync::Semaphore::new(1))
    });
    super::spec_env::acquire_permit_bounded(sem.clone(), 1, queue).await
}

#[cfg(test)]
mod concurrency_tests {
    use super::acquire_qwen4_exp_permit;
    use crate::oapi::chat::{engine_event_error_response, ChatEvent, EngineBusy};
    use crate::oapi::chat_engine::stream::{push_or_abort, sse_send_timeout};
    use std::time::Duration;

    static THE_PROCESS_WIDE_PERMIT_IS_SHARED_BY_THESE_TESTS: tokio::sync::Mutex<()> =
        tokio::sync::Mutex::const_new(());

    #[tokio::test(start_paused = true)]
    async fn qwen4_exp_serializes_sheds_and_recovers_after_errors_and_cancellation() {
        let _serial = THE_PROCESS_WIDE_PERMIT_IS_SHARED_BY_THESE_TESTS.lock().await;
        let queue = Duration::from_millis(100);
        let held = acquire_qwen4_exp_permit(queue).await.unwrap();
        let burst = futures_util::future::join_all(
            (0..12).map(|_| acquire_qwen4_exp_permit(queue)),
        )
        .await;
        for result in burst {
            let err = result.expect_err("a second Qwen4Exp request must never enter CUDA");
            let busy = err.downcast_ref::<EngineBusy>().unwrap();
            assert_eq!((busy.permits, busy.waited_ms), (1, 100));
            let response = engine_event_error_response(err.to_string());
            assert_eq!(response.status(), axum::http::StatusCode::SERVICE_UNAVAILABLE);
            assert_eq!(response.headers()["retry-after"], "1");
        }
        drop(held);

        let failed: anyhow::Result<()> = async {
            let _permit = acquire_qwen4_exp_permit(queue).await?;
            anyhow::bail!("simulated session failure")
        }
        .await;
        assert!(failed.is_err());

        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let running = tokio::spawn(async move {
            let _permit = acquire_qwen4_exp_permit(queue).await.unwrap();
            entered_tx.send(()).unwrap();
            std::future::pending::<()>().await;
        });
        entered_rx.await.unwrap();
        let waiter = tokio::spawn(acquire_qwen4_exp_permit(queue));
        tokio::task::yield_now().await;
        assert!(
            !waiter.is_finished(),
            "a queued request cannot overlap the active session"
        );
        running.abort();
        assert!(running.await.unwrap_err().is_cancelled());
        let resumed = waiter
            .await
            .unwrap()
            .expect("cancellation must release the slot");
        drop(resumed);
        drop(
            acquire_qwen4_exp_permit(queue)
                .await
                .expect("the service must remain reusable"),
        );
    }

    #[tokio::test(start_paused = true)]
    async fn a_never_read_stream_releases_the_qwen4_exp_permit_within_the_send_timeout() {
        let _serial = THE_PROCESS_WIDE_PERMIT_IS_SHARED_BY_THESE_TESTS.lock().await;
        let queue = Duration::from_millis(100);
        let (tx, _held_open_and_never_read) = tokio::sync::mpsc::channel::<ChatEvent>(1);
        tx.send(ChatEvent::TextDelta("fills the channel".into())).await.unwrap();
        let send_timeout = sse_send_timeout();
        let body = async {
            let _permit = acquire_qwen4_exp_permit(queue).await.unwrap();
            push_or_abort(&tx, ChatEvent::TextDelta("x".into()), send_timeout, "qwen4_exp").await
        };
        let delivered = tokio::time::timeout(send_timeout + Duration::from_secs(1), body)
            .await
            .expect(
                "a qwen4_exp body parked on a client that stopped reading while holding the \
                 model's only permit, so every other request is shed with 503",
            );
        assert!(!delivered, "nobody read the channel, so the send cannot have succeeded");
        drop(
            acquire_qwen4_exp_permit(queue)
                .await
                .expect("the next request acquires the permit once the stalled body returns"),
        );
    }
}

#[cfg(feature = "cuda")]
const Q4: &str = "qwen4_exp";

#[cfg(feature = "cuda")]
pub(crate) const QWEN4_EXP_LONG_SESSION_GROWS_IN_32768_TOKEN_STEPS_SO_A_GROWING_CONVERSATION_REALLOCATES_ONCE_PER_STEP: usize = 32768;

#[cfg(feature = "cuda")]
fn session_len(cache_len: usize, kv_max_seq_len: usize) -> usize {
    let min = nv_models::qwen4_exp::SESSION_MIN_LEN_8192_SHORT_REQUESTS_SHARE_ONE_STATE_SO_THEY_NEVER_OVERWRITE_THE_LONG_SESSIONS_PREFIX;
    if cache_len <= min {
        return min;
    }
    cache_len
        .next_multiple_of(QWEN4_EXP_LONG_SESSION_GROWS_IN_32768_TOKEN_STEPS_SO_A_GROWING_CONVERSATION_REALLOCATES_ONCE_PER_STEP)
        .min(kv_max_seq_len)
        .max(cache_len)
}

#[cfg(all(test, feature = "cuda"))]
mod session_len_tests {
    use super::session_len;

    #[test]
    fn short_requests_share_the_small_state_and_long_ones_grow_in_steps_inside_the_window() {
        assert_eq!(session_len(700, 262144), 8192);
        assert_eq!(session_len(8192, 262144), 8192);
        assert_eq!(session_len(8193, 262144), 32768);
        assert_eq!(session_len(141767, 262144), 163840);
        assert_eq!(session_len(250000, 262144), 262144);
        assert_eq!(session_len(40000, 50000), 50000, "the step never outgrows the engine window");
    }
}

#[cfg(feature = "cuda")]
pub(crate) const QWEN4_EXP_ADAPTER_REQUESTS_TAKE_A_FRESH_NON_SPECULATIVE_SESSION_BECAUSE_THE_SESSION_CACHE_AND_THE_MTP_HEAD_BELONG_TO_THE_BASE: &str =
    "the per-prompt session cache and its disk snapshots hold base-model KV, the MTP draft head \
     was trained for the base, and the verify-rows CUDA graph would capture m>1 LoRA deltas the \
     m=1 serving kernel cannot express; an adapter request therefore attaches the version it \
     resolved at its start under the engine permit, takes a k=0 session tagged id@version \
     (reused only under the same tag, NV_Q4EXP_LORA_SESSION_CACHE=1), decodes through the \
     captured m=1 graph or eagerly when the version carries expert LoRA, and never hands its \
     session to the base cache";

#[cfg(feature = "cuda")]
pub(crate) const NV_Q4EXP_SPEC_K_DEFAULT_2_DRAFT_TOKENS_PER_ROUND_0_DISABLES: &str = "NV_Q4EXP_SPEC_K";

#[cfg(feature = "cuda")]
fn spec_k_from_env() -> usize {
    std::env::var(NV_Q4EXP_SPEC_K_DEFAULT_2_DRAFT_TOKENS_PER_ROUND_0_DISABLES)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2)
}

#[cfg(feature = "cuda")]
pub(crate) async fn run_sampling_qwen4_exp(
    model: Arc<nv_models::qwen4_exp::Qwen4Exp>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    adapter: Option<Arc<nv_specdecode::qwen4_exp_adapters::AdapterVersion>>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let media = req.mm.as_ref().filter(|m| !(m.images.is_empty() && m.audios.is_empty()));
    let vision = match media {
        Some(m) => {
            anyhow::ensure!(
                m.audios.is_empty(),
                "qwen4_exp has no audio encoder; input_audio parts are transcribed by the perception bridge before they reach the engine"
            );
            let v = super::qwen4_exp_mm::vision_for(&model).ok_or_else(|| {
                anyhow::anyhow!("qwen4_exp vision is off (NV_Q4EXP_VISION=0): image parts go through the OCR bridge")
            })?;
            let prep = v.prepare(&req.prompt, &m.images)?;
            Some((v, prep))
        }
        None => None,
    };
    let prompt_text = vision.as_ref().map_or(req.prompt.as_str(), |(_, p)| p.prompt.as_str());
    let encoded = tokenizer
        .encode(prompt_text, false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    let (key, positions) = match &vision {
        Some((v, prep)) => {
            let (k, p) = v.key_and_positions(prep, &prompt_ids)?;
            (k, Some(p))
        }
        None => (prompt_ids.clone(), None),
    };
    let prompt_tokens = prompt_ids.len() as u32;
    let max_new = effective_max_new(req.max_new_tokens, default_max_new);

    if prompt_ids.is_empty() {
        if tx.send(ChatEvent::Started { prompt_tokens }).await.is_ok() {
            let _ = tx
                .send(ChatEvent::Done {
                    finish_reason: "length".into(),
                    completion_tokens: 0,
                })
                .await;
        }
        return Ok(());
    }

    let Some((cache_len, max_new)) = kv_window(prompt_ids.len(), max_new, kv_max_seq_len) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {kv_max_seq_len}-token window",
            prompt_ids.len()
        );
    };
    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(&prompt_ids);

    let _permit = acquire_qwen4_exp_permit(std::time::Duration::from_millis(chat_queue_ms(
        std::env::var("NV_CHAT_QUEUE_MS").ok().as_deref(),
    )))
    .await?;
    let want_len = session_len(cache_len, kv_max_seq_len);
    if let Some(adapter) = adapter {
        struct DetachOnDrop<'a>(&'a nv_models::qwen4_exp::Qwen4Exp);
        impl Drop for DetachOnDrop<'_> {
            fn drop(&mut self) {
                nv_specdecode::lora_spec::Qwen4ExpServingAdapter::detach(self.0);
            }
        }
        let _ = QWEN4_EXP_ADAPTER_REQUESTS_TAKE_A_FRESH_NON_SPECULATIVE_SESSION_BECAUSE_THE_SESSION_CACHE_AND_THE_MTP_HEAD_BELONG_TO_THE_BASE;
        adapter.serving.attach(&model)?;
        let _detach = DetachOnDrop(&model);
        let mut session = model.take_adapter_session(&adapter.tag(), want_len)?;
        let mm = match (&vision, positions.clone()) {
            (Some((v, prep)), Some(p)) => Some(v.encode(prep, &prompt_ids, key.clone(), p, model.device()).await?),
            _ => None,
        };
        model.session_install_mm(&mut session, mm.as_ref())?;
        if !push_or_abort(tx, ChatEvent::Started { prompt_tokens }, sse_send_timeout(), Q4).await {
            return Ok(());
        }
        let out = run_plain(&model, &mut session, &tokenizer, &req, &prompt_ids, cache_len, max_new, eos_ids, &mut sampler, tx).await;
        model.put_adapter_session(session);
        return out;
    }
    let k = spec_k_from_env();
    let t_take = std::time::Instant::now();
    let (mut session, source) = model.take_session_traced(&key, want_len, k)?;
    let cached = session.reusable(&key);
    let mm = match (&vision, positions) {
        (Some((v, prep)), Some(p)) => match v.encode(prep, &prompt_ids, key.clone(), p, model.device()).await {
            Ok(mm) => Some(mm),
            Err(e) => {
                model.put_session(&key, session);
                return Err(e);
            }
        },
        _ => None,
    };
    if let Err(e) = model.session_install_mm(&mut session, mm.as_ref()) {
        model.put_session(&key, session);
        return Err(e);
    }
    let short = prompt_ids.len()
        <= nv_models::qwen4_exp::SESSION_MIN_LEN_8192_SHORT_REQUESTS_SHARE_ONE_STATE_SO_THEY_NEVER_OVERWRITE_THE_LONG_SESSIONS_PREFIX;
    nv_models::session_lru::METRICS.record_request(prompt_ids.len(), cached, source, short);
    tracing::info!(
        take_session_ms = format!("{:.1}", 1000.0 * t_take.elapsed().as_secs_f64()).as_str(),
        prompt_tokens,
        cache_len,
        session_len = session.state.max_len(),
        cached_tokens = cached,
        cache_source = ?source,
        "qwen4_exp request start"
    );
    let spec = session.spec_k() >= 1
        && sampler.speculation_keeps_the_greedy_output()
        && prompt_ids.len() + session.spec_k() < cache_len;
    let pick_raw_argmax = sampler.fast_greedy();
    let res = async {
        let cached_event = ChatEvent::PromptCached {
            cached_tokens: cached as u32,
        };
        if cached > 0 && !push_or_abort(tx, cached_event, sse_send_timeout(), Q4).await {
            return Ok(());
        }
        if !push_or_abort(tx, ChatEvent::Started { prompt_tokens }, sse_send_timeout(), Q4).await {
            return Ok(());
        }
        yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;
        if spec {
            let picker = (!pick_raw_argmax).then_some(&mut sampler);
            run_greedy_spec_rounds(&model, &mut session, &tokenizer, &req, &prompt_ids, max_new, eos_ids, picker, tx).await
        } else {
            run_plain(&model, &mut session, &tokenizer, &req, &prompt_ids, cache_len, max_new, eos_ids, &mut sampler, tx).await
        }
    }
    .await;
    model.put_session(&key, session);
    res
}

#[cfg(feature = "cuda")]
async fn run_greedy_spec_rounds(
    model: &nv_models::qwen4_exp::Qwen4Exp,
    session: &mut nv_models::qwen4_exp::Qwen4ExpSession,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    prompt_ids: &[u32],
    max_new: usize,
    eos_ids: &[u32],
    mut sampler: Option<&mut ChatSampler>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let t_req = std::time::Instant::now();
    let send_timeout = sse_send_timeout();
    let anchor = match sampler.as_deref_mut() {
        Some(s) => model.session_begin_with(session, prompt_ids, &mut SamplerPick(s))?,
        None => model.session_begin(session, prompt_ids)?,
    };
    let begin_ms = 1000.0 * t_req.elapsed().as_secs_f64();
    let mut first_delta_ms: Option<f64> = None;
    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    let mut emitter = StreamEmitter::new(&req.stop);
    let mut detok = IncrementalDetok::new(tokenizer.clone());
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();
    let mut pending: Vec<u32> = vec![anchor];
    'stream: loop {
        for tok in pending.drain(..) {
            if eos_ids.contains(&tok) {
                finish_reason = "stop".into();
                break 'stream;
            }
            if let Some(s) = sampler.as_deref_mut() {
                anyhow::ensure!(
                    s.commit_greedy(tok),
                    "speculative token {tok} left the grammar: drafts are accepted only when they \
                     equal the masked target pick, so this is a picker bug"
                );
            }
            generated_ids.push(tok);
            completion_tokens = generated_ids.len() as u32;
            let new_text = detok.push(tok)?;
            let (piece, stop_hit) = emitter.step(new_text);
            if !piece.is_empty() {
                first_delta_ms.get_or_insert_with(|| 1000.0 * t_req.elapsed().as_secs_f64());
                if !push_or_abort(tx, ChatEvent::TextDelta(piece), send_timeout, Q4).await {
                    return Ok(());
                }
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
        pending = match sampler.as_deref_mut() {
            Some(s) => model.session_round_with(session, &mut SamplerPick(s))?,
            None => model.session_round(session)?,
        };
    }
    let s = &session.stats;
    tracing::info!(
        rounds = s.rounds,
        drafted = s.drafted,
        accepted = s.accepted,
        accept_rate = format!("{:.3}", s.accept_rate()).as_str(),
        tokens_per_round = format!("{:.2}", s.tokens_per_round()).as_str(),
        k = session.spec_k(),
        completion_tokens,
        begin_ms = format!("{begin_ms:.1}").as_str(),
        first_delta_ms = format!("{:.1}", first_delta_ms.unwrap_or(-1.0)).as_str(),
        gen_ms = format!("{:.1}", 1000.0 * t_req.elapsed().as_secs_f64()).as_str(),
        "qwen4_exp mtp self-speculative request done"
    );
    if !generated_ids.is_empty() {
        if let Ok(full) =
            crate::oapi::chat_engine::stream::decode_keeping_wire(tokenizer, &generated_ids)
        {
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                let _ = push_or_abort(tx, ChatEvent::TextDelta(tail), send_timeout, Q4).await;
            }
        }
    }
    let _ = send_stopped_by_then_done(
        tx,
        &mut emitter,
        finish_reason,
        completion_tokens,
        Some(send_timeout),
    )
    .await;
    Ok(())
}

#[cfg(feature = "cuda")]
async fn run_plain(
    model: &nv_models::qwen4_exp::Qwen4Exp,
    session: &mut nv_models::qwen4_exp::Qwen4ExpSession,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    prompt_ids: &[u32],
    cache_len: usize,
    max_new: usize,
    eos_ids: &[u32],
    sampler: &mut ChatSampler,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let (logits, _) = model.session_prefill(session, prompt_ids)?;
    let send_timeout = sse_send_timeout();

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
        if !piece.is_empty()
            && !push_or_abort(tx, ChatEvent::TextDelta(piece), send_timeout, Q4).await
        {
            return Ok(());
        }
        if req.logprobs
            && !push_or_abort(
                tx,
                ChatEvent::Logprob(build_logprob_entry(tokenizer, &last_out)),
                send_timeout,
                Q4,
            )
            .await
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
        if session.state.pos >= cache_len {
            finish_reason = "length".into();
            break;
        }

        let nv_models::qwen4_exp::Qwen4ExpSession { state, graph, .. } = &mut *session;
        let step_logits = model.decode_graphed(last, state, graph)?;
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
            crate::oapi::chat_engine::stream::decode_keeping_wire(tokenizer, &generated_ids)
        {
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                let _ = push_or_abort(tx, ChatEvent::TextDelta(tail), send_timeout, Q4).await;
            }
        }
    }

    let _ = send_stopped_by_then_done(
        tx,
        &mut emitter,
        finish_reason,
        completion_tokens,
        Some(send_timeout),
    )
    .await;
    Ok(())
}
