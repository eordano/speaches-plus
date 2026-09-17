#[cfg(feature = "cuda")]
use super::*;

#[cfg(any(test, feature = "cuda"))]
const NV_Q38_SOLO_DEVICE_ARGMAX_DEFAULT_ON_USER_SIGN_OFF_2026_09_12_ZERO_RESTORES_HOST_FEED: &str =
    "NV_Q38_SOLO_DEVICE_ARGMAX";

#[cfg(any(test, feature = "cuda"))]
fn solo_device_argmax_default_on_user_sign_off_2026_09_12(raw: Option<&std::ffi::OsStr>) -> bool {
    raw.is_none() || raw == Some(std::ffi::OsStr::new("1"))
}

#[cfg(test)]
mod solo_device_argmax_default_tests {
    use super::solo_device_argmax_default_on_user_sign_off_2026_09_12 as selected;
    use std::ffi::OsStr;

    #[test]
    fn only_unset_changes_and_zero_restores_host_feed() {
        assert!(selected(None));
        assert!(selected(Some(OsStr::new("1"))));
        for raw in ["0", "", " 1", "1 ", "true", "gate"] {
            assert!(
                !selected(Some(OsStr::new(raw))),
                "explicit {raw:?} stays off"
            );
        }
    }

    #[cfg(unix)]
    #[test]
    fn non_unicode_remains_explicit_off() {
        use std::os::unix::ffi::OsStrExt;
        assert!(!selected(Some(OsStr::from_bytes(&[0xff]))));
    }
}

#[cfg(any(test, feature = "cuda"))]
fn qwen_emitted_token_ids_fnv1a64_u32_le(tokens: &[u32]) -> u64 {
    tokens
        .iter()
        .flat_map(|token| token.to_le_bytes())
        .fold(0xcbf29ce484222325u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
}

#[cfg(any(test, feature = "cuda"))]
#[derive(Default)]
struct QwenReadyBlocks {
    first: Option<(std::time::Instant, u64)>,
    last: Option<std::time::Instant>,
    first_block_tokens: u32,
    tokens: u32,
}

#[cfg(any(test, feature = "cuda"))]
impl QwenReadyBlocks {
    fn retain(&mut self, ready: std::time::Instant, block: u64) {
        let first = *self.first.get_or_insert((ready, block));
        self.first_block_tokens += u32::from(block == first.1);
        self.last = Some(ready);
        self.tokens += 1;
    }

    fn generation_seconds(&self) -> Option<f64> {
        let elapsed = self.last?.checked_duration_since(self.first?.0)?;
        (!elapsed.is_zero()).then_some(elapsed.as_secs_f64())
    }
}

#[cfg(feature = "cuda")]
struct QwenGenerationTrace {
    id: u64,
    route: &'static str,
    depth: usize,
    allocated_context: usize,
    context_limit: usize,
    engine_arm: Option<&'static str>,
    capture_active: Option<bool>,
    block: u64,
    ready: std::time::Instant,
    blocks: QwenReadyBlocks,
}

#[cfg(feature = "cuda")]
impl QwenGenerationTrace {
    fn selected(
        dense: bool,
        route: &'static str,
        depth: usize,
        allocated_context: usize,
        context_limit: usize,
    ) -> Option<Self> {
        if !dense || !crate::oapi::chat::oapi_phase_walls_selected_from_env() {
            return None;
        }
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Some(Self {
            id: NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            route,
            depth,
            allocated_context,
            context_limit,
            engine_arm: None,
            capture_active: None,
            block: 0,
            ready: std::time::Instant::now(),
            blocks: QwenReadyBlocks::default(),
        })
    }

    fn retain(&mut self) {
        self.blocks.retain(self.ready, self.block);
    }

    fn next_block(&mut self) {
        self.ready = std::time::Instant::now();
        self.block += 1;
    }

    fn report(self, completion_tokens: u32, finish_reason: &str) {
        let seconds = self.blocks.generation_seconds();
        eprintln!(
            "Q38-GENERATION {}",
            serde_json::json!({
                "id": self.id,
                "route": self.route,
                "engine_arm": self.engine_arm,
                "capture_active_at_finish": self.capture_active,
                "prompt_tokens": self.depth,
                "allocated_context": self.allocated_context,
                "context_limit": self.context_limit,
                "retained_tokens": self.blocks.tokens,
                "completion_tokens": completion_tokens,
                "counts_match": self.blocks.tokens == completion_tokens,
                "first_block_tokens": self.blocks.first_block_tokens,
                "host_ready_generation_seconds": seconds,
                "host_ready_generation_tok_s": seconds.map(|s|
                    (self.blocks.tokens - self.blocks.first_block_tokens) as f64 / s),
                "finish_reason": finish_reason,
                "terminal_event_enqueued": true,
            })
        );
    }
}

#[cfg(feature = "cuda")]
fn qwen_generation_trace_deliverable(tail_sent: bool, done: SsePush) -> bool {
    tail_sent && done == SsePush::Sent
}

#[cfg(test)]
mod qwen_ready_blocks_tests {
    use super::QwenReadyBlocks;
    use std::time::{Duration, Instant};

    #[test]
    fn excludes_every_retained_token_in_the_first_ready_block() {
        let t = Instant::now();
        let mut blocks = QwenReadyBlocks::default();
        for (ready, block) in [
            (t, 0),
            (t, 0),
            (t + Duration::from_millis(100), 1),
            (t + Duration::from_millis(100), 1),
        ] {
            blocks.retain(ready, block);
        }
        assert_eq!(blocks.tokens, 4);
        assert_eq!(blocks.first_block_tokens, 2);
        assert_eq!(blocks.generation_seconds(), Some(0.1));
        assert_eq!(
            (blocks.tokens - blocks.first_block_tokens) as f64 / 0.1,
            20.0
        );
    }

    #[test]
    fn no_interval_is_reported_for_empty_or_single_block_output() {
        let mut blocks = QwenReadyBlocks::default();
        assert_eq!(blocks.generation_seconds(), None);
        let t = Instant::now();
        blocks.retain(t, 0);
        blocks.retain(t, 0);
        assert_eq!(blocks.generation_seconds(), None);
        assert_eq!(blocks.tokens, blocks.first_block_tokens);
    }

    #[test]
    fn distinct_blocks_with_equal_clock_values_do_not_merge_their_counts() {
        let mut blocks = QwenReadyBlocks::default();
        let t = Instant::now();
        blocks.retain(t, 0);
        blocks.retain(t, 1);
        assert_eq!(blocks.first_block_tokens, 1);
        assert_eq!(blocks.tokens, 2);
        assert_eq!(blocks.generation_seconds(), None);
    }
}

#[cfg(any(test, feature = "cuda"))]
struct SoloQueue {
    waiting: std::sync::Arc<tokio::sync::Semaphore>,
    capacity: usize,
    timeout: std::time::Duration,
}

#[cfg(any(test, feature = "cuda"))]
impl SoloQueue {
    fn new(capacity: usize, timeout: std::time::Duration) -> Self {
        Self {
            waiting: std::sync::Arc::new(tokio::sync::Semaphore::new(capacity)),
            capacity,
            timeout,
        }
    }

    async fn lock<T: Send>(
        &self,
        model: std::sync::Arc<tokio::sync::Mutex<T>>,
        tx: &tokio::sync::mpsc::Sender<crate::oapi::chat::ChatEvent>,
    ) -> anyhow::Result<Option<tokio::sync::OwnedMutexGuard<T>>> {
        if tx.is_closed() {
            return Ok(None);
        }
        if let Ok(guard) = model.clone().try_lock_owned() {
            return Ok(Some(guard));
        }
        let busy = |waited_ms| crate::oapi::chat::EngineBusy::new(self.capacity, waited_ms);
        let _waiting = self
            .waiting
            .clone()
            .try_acquire_owned()
            .map_err(|_| busy(0))?;
        tokio::select! {
            biased;
            _ = tx.closed() => Ok(None),
            result = tokio::time::timeout(self.timeout, model.lock_owned()) => {
                result.map(Some).map_err(|_| busy(self.timeout.as_millis() as u64).into())
            }
        }
    }
}

#[cfg(feature = "cuda")]
fn solo_queue() -> &'static SoloQueue {
    static QUEUE: std::sync::OnceLock<SoloQueue> = std::sync::OnceLock::new();
    QUEUE.get_or_init(|| {
        SoloQueue::new(
            chat_permits(std::env::var("NV_CHAT_CONCURRENCY").ok().as_deref()),
            std::time::Duration::from_millis(chat_queue_ms(
                std::env::var("NV_CHAT_QUEUE_MS").ok().as_deref(),
            )),
        )
    })
}

#[cfg(test)]
mod solo_queue_tests {
    use super::SoloQueue;
    use crate::oapi::chat::{ChatEvent, EngineBusy};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::{mpsc, Mutex};

    #[tokio::test(start_paused = true)]
    async fn solo_queue_bounds_waiters_and_recovers_after_release_and_timeout() {
        let queue = SoloQueue::new(1, Duration::from_millis(50));
        let model = Arc::new(Mutex::new(()));
        let (tx, _rx) = mpsc::channel::<ChatEvent>(1);
        let active = queue.lock(model.clone(), &tx).await.unwrap().unwrap();
        let mut waiting = Box::pin(queue.lock(model.clone(), &tx));
        assert!(futures_util::poll!(waiting.as_mut()).is_pending());
        let error = queue.lock(model.clone(), &tx).await.unwrap_err();
        assert_eq!(error.downcast_ref::<EngineBusy>().unwrap().waited_ms, 0);
        let error = waiting.await.unwrap_err();
        assert_eq!(error.downcast_ref::<EngineBusy>().unwrap().waited_ms, 50);
        assert_eq!(queue.waiting.available_permits(), 1);

        let mut waiting = Box::pin(queue.lock(model.clone(), &tx));
        assert!(futures_util::poll!(waiting.as_mut()).is_pending());
        drop(active);
        drop(waiting.await.unwrap().unwrap());
        drop(queue.lock(model, &tx).await.unwrap().unwrap());
        assert_eq!(queue.waiting.available_permits(), 1);
    }

    #[tokio::test(start_paused = true)]
    async fn a_disconnected_solo_waiter_releases_capacity_without_taking_the_model() {
        let queue = SoloQueue::new(1, Duration::from_secs(60));
        let model = Arc::new(Mutex::new(()));
        let active = model.clone().lock_owned().await;
        let (tx, rx) = mpsc::channel::<ChatEvent>(1);
        let mut waiting = Box::pin(queue.lock(model.clone(), &tx));
        assert!(futures_util::poll!(waiting.as_mut()).is_pending());
        drop(rx);
        assert!(waiting.await.unwrap().is_none());
        assert_eq!(queue.waiting.available_permits(), 1);
        assert!(model.try_lock().is_err());
        drop(active);
        assert!(queue.lock(model, &tx).await.unwrap().is_none());
    }
}

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
async fn solo_started(
    tx: &mpsc::Sender<ChatEvent>,
    prompt_tokens: u32,
    timeout: Option<std::time::Duration>,
) -> bool {
    if solo_push(tx, ChatEvent::Started { prompt_tokens }, timeout).await != SsePush::Sent {
        return false;
    }
    yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;
    true
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
fn qwen_solo_sample_device_token_or_checked_host(
    sampler: &mut ChatSampler,
    device_token: Option<u32>,
    host_logits: impl FnOnce() -> anyhow::Result<Vec<f32>>,
) -> anyhow::Result<SampleOutput> {
    if let Some(token) = device_token.filter(|&token| token != 0 && sampler.fast_greedy()) {
        sampler.record_token(token);
        return Ok(SampleOutput {
            token,
            logprob: None,
            top: Vec::new(),
            exhausted: false,
        });
    }
    Ok(sampler.sample(&host_logits()?))
}

#[cfg(all(test, feature = "cuda"))]
mod solo_device_greedy_tests {
    use super::*;

    fn greedy() -> ChatSampler {
        ChatSampler::new(
            nv_layers::sampler::SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
            1,
            None,
            Vec::new(),
            false,
            0,
        )
    }

    #[test]
    fn device_zero_preserves_invalid_rows_ties_and_host_errors() {
        for row in [
            vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY],
            vec![3.0, 3.0, f32::NAN],
            vec![f32::INFINITY, 3.0, 3.0],
        ] {
            let expected = greedy().sample(&row);
            let actual =
                qwen_solo_sample_device_token_or_checked_host(&mut greedy(), Some(0), || Ok(row))
                    .unwrap();
            assert_eq!(actual.token, expected.token);
            assert_eq!(actual.exhausted, expected.exhausted);
        }
        assert!(qwen_solo_sample_device_token_or_checked_host(
            &mut greedy(),
            Some(0),
            || anyhow::bail!("host copy failed"),
        )
        .is_err());
    }

    #[test]
    fn device_nonzero_skips_host_copy_only_for_unmodified_greedy_sampling() {
        let mut sampler = greedy();
        let result = qwen_solo_sample_device_token_or_checked_host(&mut sampler, Some(7), || {
            panic!("finite nonzero winner must not copy logits")
        })
        .unwrap();
        assert_eq!(result.token, 7);
        assert!(!result.exhausted);
        assert_eq!(sampler.counts.get(&7), Some(&1));
        for mode in 0..3 {
            let mut sampler = greedy();
            match mode {
                0 => sampler.logprobs = true,
                1 => sampler.logit_bias.push((0, 10.0)),
                _ => sampler.params.presence_penalty = 1.0,
            }
            let result =
                qwen_solo_sample_device_token_or_checked_host(&mut sampler, Some(7), || {
                    Ok(vec![3.0, 1.0])
                })
                .unwrap();
            assert_eq!(result.token, 0);
            assert!(!result.exhausted);
        }
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
    let Some(mut engine) = solo_queue().lock(model, tx).await? else {
        return Ok(());
    };
    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    let prompt_tokens = prompt_ids.len() as u32;
    crate::oapi::chat::oapi_phase_wall(t_gen, "gen_tokenized");

    let max_new = effective_max_new(req.max_new_tokens, default_max_new);
    let send_timeout =
        solo_send_timeout_on_the_engine_runtime_zero_clamps_to_1ms_like_the_gemma4_paths();
    let allocated_context = engine.cache().max_seq_len();
    let window = allocated_context.min(engine.underlying().config().max_position_embeddings);
    let Some((_cache_len, max_new)) = kv_window(prompt_ids.len(), max_new, window) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {window}-token KV window",
            prompt_ids.len()
        );
    };

    if prompt_ids.is_empty() {
        if !solo_started(tx, prompt_tokens, send_timeout).await {
            return Ok(());
        }
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

    crate::oapi::chat::oapi_phase_wall(t_gen, "gen_engine_lock_acquired");
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
            if !solo_started(tx, prompt_tokens, send_timeout).await {
                return Ok(());
            }
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
            let generation_trace = QwenGenerationTrace::selected(
                true,
                "mtp-graphed-solo",
                prompt_ids.len(),
                allocated_context,
                window,
            );
            crate::oapi::chat::oapi_phase_wall(t_gen, "gen_q38_session_started");
            if !solo_started(tx, prompt_tokens, send_timeout).await {
                return Ok(());
            }
            return run_qwen38_mtp_rounds_streaming(
                session,
                &tokenizer,
                &req,
                max_new,
                eos_ids,
                k,
                tx,
                send_timeout,
                generation_trace,
            )
            .await;
        }
    }

    let device_greedy = engine.underlying().dense_intermediate().is_some()
        && sampler.fast_greedy()
        && solo_device_argmax_default_on_user_sign_off_2026_09_12(
            std::env::var_os(
                NV_Q38_SOLO_DEVICE_ARGMAX_DEFAULT_ON_USER_SIGN_OFF_2026_09_12_ZERO_RESTORES_HOST_FEED,
            ).as_deref(),
        );
    tracing::debug!(
        device_argmax = device_greedy,
        "qwen CUDA solo decode sampling"
    );
    engine.reset()?;
    let row = engine.prefill(&prompt_ids)?;
    let mut last_out = sampler.sample(&row);
    let mut last = last_out.token;
    if last_out.exhausted {
        anyhow::bail!("no legal token at prefill: the sampling mask left every candidate at -inf");
    }
    let mut generation_trace = QwenGenerationTrace::selected(
        engine.underlying().dense_intermediate().is_some(),
        "plain-graphed-solo",
        prompt_ids.len(),
        allocated_context,
        window,
    );
    if let Some(trace) = &mut generation_trace {
        trace.capture_active = Some(engine.capture_active());
    }
    if !solo_started(tx, prompt_tokens, send_timeout).await {
        return Ok(());
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
        if let Some(trace) = &mut generation_trace {
            trace.retain();
        }
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
        let device_token = if device_greedy {
            Some(engine.next_token_argmax_on_device()?)
        } else {
            None
        };
        last_out =
            qwen_solo_sample_device_token_or_checked_host(&mut sampler, device_token, || {
                engine.logits_host()
            })?;
        last = last_out.token;
        if let Some(trace) = &mut generation_trace {
            trace.next_block();
        }
        if last_out.exhausted {
            anyhow::bail!(
                "no legal token at step {step}: the sampling mask left every candidate at -inf"
            );
        }
    }

    let mut tail_sent = generated_ids.is_empty();
    if !generated_ids.is_empty() {
        if let Ok(full) =
            crate::oapi::chat_engine::stream::decode_keeping_wire(&tokenizer, &generated_ids)
        {
            tail_sent = true;
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                tail_sent =
                    solo_push(tx, ChatEvent::TextDelta(tail), send_timeout).await == SsePush::Sent;
            }
        }
    }

    let done = solo_push(
        tx,
        ChatEvent::Done {
            finish_reason: finish_reason.clone(),
            completion_tokens,
        },
        send_timeout,
    )
    .await;
    if qwen_generation_trace_deliverable(tail_sent, done) {
        if let Some(mut trace) = generation_trace {
            trace.capture_active = Some(engine.capture_active());
            trace.report(completion_tokens, &finish_reason);
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
async fn tokenize_before_prefill(
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    tx: &mpsc::Sender<ChatEvent>,
    send_timeout: Option<std::time::Duration>,
    window: usize,
) -> anyhow::Result<Option<Vec<u32>>> {
    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    let prompt_tokens = prompt_ids.len() as u32;
    anyhow::ensure!(
        prompt_ids.len() < window,
        "prompt of {prompt_tokens} tokens does not fit the {window}-token KV window"
    );
    if prompt_ids.is_empty() {
        if !solo_started(tx, prompt_tokens, send_timeout).await {
            return Ok(None);
        }
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
    let Some(model) = solo_queue().lock(model, tx).await? else {
        return Ok(());
    };
    let send_timeout =
        solo_send_timeout_on_the_engine_runtime_zero_clamps_to_1ms_like_the_gemma4_paths();
    let window = kv_max_seq_len.min(model.config().max_position_embeddings);
    let Some(prompt_ids) =
        tokenize_before_prefill(&tokenizer, &req, tx, send_timeout, window).await?
    else {
        return Ok(());
    };
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
    let window = kv_max_seq_len.min(model.config().max_position_embeddings);
    let Some(prompt_ids) =
        tokenize_before_prefill(tokenizer, &req, tx, send_timeout, window).await?
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
        if !solo_started(tx, prompt_ids.len() as u32, send_timeout).await {
            return Ok(());
        }
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
    let window = kv_max_seq_len.min(model.config().max_position_embeddings);
    let Some(prompt_ids) =
        tokenize_before_prefill(tokenizer, &req, tx, send_timeout, window).await?
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
    let mut generation_trace = QwenGenerationTrace::selected(
        model.dense_intermediate().is_some(),
        "plain-eager-solo",
        prompt_ids.len(),
        cache.max_seq_len(),
        window,
    );
    if let Some(trace) = &mut generation_trace {
        trace.capture_active = Some(false);
    }
    if !solo_started(tx, prompt_ids.len() as u32, send_timeout).await {
        return Ok(());
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
        if let Some(trace) = &mut generation_trace {
            trace.retain();
        }
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
        if let Some(trace) = &mut generation_trace {
            trace.next_block();
        }
        if last_out.exhausted {
            anyhow::bail!(
                "no legal token at step {step}: the sampling mask left every candidate at -inf"
            );
        }
    }

    let mut tail_sent = generated_ids.is_empty();
    if !generated_ids.is_empty() {
        if let Ok(full) =
            crate::oapi::chat_engine::stream::decode_keeping_wire(tokenizer, &generated_ids)
        {
            tail_sent = true;
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                tail_sent =
                    solo_push(tx, ChatEvent::TextDelta(tail), send_timeout).await == SsePush::Sent;
            }
        }
    }

    let done = solo_push(
        tx,
        ChatEvent::Done {
            finish_reason: finish_reason.clone(),
            completion_tokens,
        },
        send_timeout,
    )
    .await;
    if qwen_generation_trace_deliverable(tail_sent, done) {
        if let Some(trace) = generation_trace {
            trace.report(completion_tokens, &finish_reason);
        }
    }
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
    let generation_trace =
        QwenGenerationTrace::selected(true, "mtp-eager-solo", prompt_ids.len(), max_seq, window);
    if !solo_started(tx, prompt_ids.len() as u32, send_timeout).await {
        return Ok(());
    }
    run_qwen38_mtp_rounds_streaming(
        session,
        tokenizer,
        req,
        max_new,
        eos_ids,
        k,
        tx,
        send_timeout,
        generation_trace,
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
    mut generation_trace: Option<QwenGenerationTrace>,
) -> anyhow::Result<()> {
    if let Some(trace) = &mut generation_trace {
        trace.engine_arm = Some(session.arm_name());
        trace.capture_active = Some(session.capture_active());
    }
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
            if let Some(trace) = &mut generation_trace {
                trace.retain();
            }
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
        if let Some(trace) = &mut generation_trace {
            trace.next_block();
        }
    }

    let s = session.stats();
    let suffix = session.suffix_rounds_drafted_accepted();
    let (suffix_rounds, suffix_drafted, suffix_accepted) = suffix.unwrap_or_default();
    let tracking = session.acceptance_tracking_k2_k3_switches();
    let (adaptive_k2_rounds, adaptive_k3_rounds, adaptive_switches) = tracking.unwrap_or_default();
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
        suffix_enabled = suffix.is_some(),
        suffix_rounds,
        suffix_drafted,
        suffix_accepted,
        accept_tracking_enabled = tracking.is_some(),
        adaptive_k2_rounds,
        adaptive_k3_rounds,
        adaptive_switches,
        draft_ms = s.draft_ms,
        verify_ms = s.verify_ms,
        commit_ms = s.commit_ms,
        emitted_fnv1a64 = format!(
            "{:016x}",
            qwen_emitted_token_ids_fnv1a64_u32_le(&generated_ids)
        )
        .as_str(),
        "qwen3.8 mtp self-speculative request done"
    );

    let mut tail_sent = generated_ids.is_empty();
    if !generated_ids.is_empty() {
        if let Ok(full) =
            crate::oapi::chat_engine::stream::decode_keeping_wire(tokenizer, &generated_ids)
        {
            tail_sent = true;
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                tail_sent =
                    solo_push(tx, ChatEvent::TextDelta(tail), send_timeout).await == SsePush::Sent;
            }
        }
    }

    let done = solo_push(
        tx,
        ChatEvent::Done {
            finish_reason: finish_reason.clone(),
            completion_tokens,
        },
        send_timeout,
    )
    .await;
    if qwen_generation_trace_deliverable(tail_sent, done) {
        if let Some(mut trace) = generation_trace {
            trace.capture_active = Some(session.capture_active());
            trace.report(completion_tokens, &finish_reason);
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "cuda"))]
mod qwen_generation_streaming_tests {
    use super::*;
    use nv_models::qwen3_5_mtp::{MtpMoeServingRounds, MtpSpecStats};

    #[test]
    fn emitted_token_hash_preserves_u32_byte_order_and_empty_stream() {
        assert_eq!(
            qwen_emitted_token_ids_fnv1a64_u32_le(&[]),
            0xcbf29ce484222325
        );
        assert_eq!(
            qwen_emitted_token_ids_fnv1a64_u32_le(&[0]),
            0x4d25767f9dce13f5
        );
        assert_eq!(
            qwen_emitted_token_ids_fnv1a64_u32_le(&[0x01020304]),
            0xb345225e3644edb5
        );
        assert_eq!(
            qwen_emitted_token_ids_fnv1a64_u32_le(&[1, 256, 65536, u32::MAX]),
            0xd171144022ec4b52
        );
        assert_eq!(
            FakeRounds::new(0, []).suffix_rounds_drafted_accepted(),
            None
        );
    }

    struct FakeRounds {
        anchor: u32,
        rounds: std::collections::VecDeque<Vec<u32>>,
        stats: MtpSpecStats,
    }

    impl FakeRounds {
        fn new(anchor: u32, rounds: impl IntoIterator<Item = Vec<u32>>) -> Self {
            Self {
                anchor,
                rounds: rounds.into_iter().collect(),
                stats: MtpSpecStats::default(),
            }
        }
    }

    impl MtpMoeServingRounds for FakeRounds {
        fn anchor(&self) -> u32 {
            self.anchor
        }

        fn round_fits(&self) -> bool {
            !self.rounds.is_empty()
        }

        fn round(&mut self) -> anyhow::Result<Vec<u32>> {
            let out = self.rounds.pop_front().unwrap_or_default();
            self.stats.rounds += 1;
            self.stats.emitted += out.len();
            Ok(out)
        }

        fn stats(&self) -> &MtpSpecStats {
            &self.stats
        }

        fn arm_name(&self) -> &'static str {
            "fake-rounds"
        }

        fn capture_active(&self) -> bool {
            false
        }
    }

    fn tiny_tokenizer() -> Arc<tokenizers::Tokenizer> {
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                ["alpha", "beta", "gamma", "<eos>"]
                    .iter()
                    .enumerate()
                    .map(|(i, token)| ((*token).to_string(), i as u32))
                    .collect(),
            )
            .build()
            .unwrap();
        Arc::new(tokenizers::Tokenizer::new(model))
    }

    fn request(max_new_tokens: usize) -> ChatGenerateRequest {
        ChatGenerateRequest {
            prompt: "alpha".into(),
            max_new_tokens,
            stop: vec![],
            seed: Some(0),
            temperature: Some(0.0),
            top_p: None,
            top_k: None,
            min_p: None,
            presence_penalty: None,
            frequency_penalty: None,
            repetition_penalty: None,
            guided: None,
            guided_think_close: None,
            logit_bias: vec![],
            logprobs: false,
            top_logprobs: 0,
            kv_resume: None,
            kv_store: None,
            mm: None,
        }
    }

    fn trace() -> QwenGenerationTrace {
        QwenGenerationTrace {
            id: 1,
            route: "fake-mtp",
            depth: 1,
            allocated_context: 16,
            context_limit: 16,
            engine_arm: None,
            capture_active: None,
            block: 0,
            ready: std::time::Instant::now(),
            blocks: QwenReadyBlocks::default(),
        }
    }

    fn event_shape(event: ChatEvent) -> (&'static str, Option<(String, u32)>) {
        match event {
            ChatEvent::TextDelta(_) => ("text", None),
            ChatEvent::Done {
                finish_reason,
                completion_tokens,
            } => ("done", Some((finish_reason, completion_tokens))),
            _ => ("other", None),
        }
    }

    async fn budget_events(with_trace: bool) -> Vec<(&'static str, Option<(String, u32)>)> {
        let (tx, mut rx) = mpsc::channel(16);
        run_qwen38_mtp_rounds_streaming(
            FakeRounds::new(0, [vec![1, 2]]),
            &tiny_tokenizer(),
            &request(2),
            2,
            &[3],
            2,
            &tx,
            None,
            with_trace.then(trace),
        )
        .await
        .unwrap();
        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event_shape(event));
        }
        events
    }

    #[tokio::test]
    async fn trace_preserves_stream_events_and_output_budget_truncation() {
        let off = budget_events(false).await;
        let on = budget_events(true).await;
        assert_eq!(on, off);
        assert_eq!(on.last(), Some(&("done", Some(("length".into(), 2)))));
    }

    #[tokio::test]
    async fn eos_before_retention_reports_zero_tokens_and_stop() {
        let (tx, mut rx) = mpsc::channel(4);
        run_qwen38_mtp_rounds_streaming(
            FakeRounds::new(3, std::iter::empty()),
            &tiny_tokenizer(),
            &request(4),
            4,
            &[3],
            2,
            &tx,
            None,
            Some(trace()),
        )
        .await
        .unwrap();
        assert!(matches!(
            rx.recv().await,
            Some(ChatEvent::Done {
                ref finish_reason,
                completion_tokens: 0,
            }) if finish_reason == "stop"
        ));
    }

    #[tokio::test]
    async fn failed_terminal_delivery_is_not_reportable() {
        let (tx, rx) = mpsc::channel(1);
        drop(rx);
        run_qwen38_mtp_rounds_streaming(
            FakeRounds::new(3, std::iter::empty()),
            &tiny_tokenizer(),
            &request(4),
            4,
            &[3],
            2,
            &tx,
            None,
            Some(trace()),
        )
        .await
        .unwrap();
        assert!(!qwen_generation_trace_deliverable(true, SsePush::Closed));
        assert!(!qwen_generation_trace_deliverable(true, SsePush::TimedOut));
    }
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
    let send_timeout =
        solo_send_timeout_on_the_engine_runtime_zero_clamps_to_1ms_like_the_gemma4_paths();

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
        if let Ok(full) =
            crate::oapi::chat_engine::stream::decode_keeping_wire(&tokenizer, &generated_ids)
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
