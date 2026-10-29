#[cfg(feature = "cuda")]
use super::*;
#[cfg(feature = "cuda")]
use crate::oapi::generation_limits::request_window;

#[cfg(feature = "cuda")]
pub(crate) const LAGUNA_WORKER_RETIRED: &str = "engine busy: Laguna worker retired; try later.";

#[cfg(feature = "cuda")]
pub(crate) struct LagunaAdmission {
    active: Arc<tokio::sync::RwLock<()>>,
    waiting: Arc<tokio::sync::Semaphore>,
    capacity: usize,
    queue: std::time::Duration,
}

#[cfg(feature = "cuda")]
impl LagunaAdmission {
    fn new(capacity: usize, queue: std::time::Duration) -> Self {
        Self {
            active: Arc::new(tokio::sync::RwLock::new(())),
            waiting: Arc::new(tokio::sync::Semaphore::new(capacity)),
            capacity,
            queue,
        }
    }

    pub(crate) fn from_env() -> Self {
        Self::new(
            chat_permits(std::env::var("NV_CHAT_CONCURRENCY").ok().as_deref()),
            std::time::Duration::from_millis(chat_queue_ms(
                std::env::var("NV_CHAT_QUEUE_MS").ok().as_deref(),
            )),
        )
    }

    fn deadline(&self) -> tokio::time::Instant {
        tokio::time::Instant::now() + self.queue
    }

    fn ensure_available(&self) -> anyhow::Result<()> {
        anyhow::ensure!(!self.waiting.is_closed(), LAGUNA_WORKER_RETIRED);
        Ok(())
    }

    fn retire(&self) {
        self.waiting.close();
    }

    pub(crate) async fn acquire_exclusive(
        &self,
    ) -> anyhow::Result<tokio::sync::OwnedRwLockWriteGuard<()>> {
        self.ensure_available()?;
        let permit = self.active.clone().write_owned().await;
        self.ensure_available()?;
        Ok(permit)
    }

    async fn acquire(
        &self,
        tx: &mpsc::Sender<ChatEvent>,
    ) -> anyhow::Result<Option<tokio::sync::OwnedRwLockWriteGuard<()>>> {
        if tx.is_closed() {
            return Ok(None);
        }
        self.ensure_available()?;
        if let Ok(permit) = self.active.clone().try_write_owned() {
            self.ensure_available()?;
            return Ok(Some(permit));
        }
        self.wait_until(tx, self.deadline(), self.active.clone().write_owned())
            .await
    }

    async fn acquire_spec(
        &self,
        tx: &mpsc::Sender<ChatEvent>,
        deadline: tokio::time::Instant,
    ) -> anyhow::Result<Option<tokio::sync::OwnedRwLockReadGuard<()>>> {
        if tx.is_closed() {
            return Ok(None);
        }
        self.ensure_available()?;
        if let Ok(permit) = self.active.clone().try_read_owned() {
            self.ensure_available()?;
            return Ok(Some(permit));
        }
        self.wait_until(tx, deadline, self.active.clone().read_owned())
            .await
    }

    async fn wait_until<T>(
        &self,
        tx: &mpsc::Sender<ChatEvent>,
        deadline: tokio::time::Instant,
        acquire: impl std::future::Future<Output = T>,
    ) -> anyhow::Result<Option<T>> {
        let busy = |ms| crate::oapi::chat::EngineBusy::new(self.capacity, ms);
        let _waiting = match self.waiting.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(_) => {
                self.ensure_available()?;
                return Err(busy(0).into());
            }
        };
        tokio::select! {
            biased;
            _ = tx.closed() => Ok(None),
            result = tokio::time::timeout_at(deadline, acquire) => {
                self.ensure_available()?;
                result.map(Some).map_err(|_| busy(self.queue.as_millis() as u64).into())
            }
        }
    }
}

#[cfg(feature = "cuda")]
struct LagunaSpecReservation {
    admission: Arc<LagunaAdmission>,
    _permit: tokio::sync::OwnedRwLockReadGuard<()>,
}

#[cfg(feature = "cuda")]
impl LagunaSpecReservation {
    fn observe(&self, event: &nv_models::laguna_serve::SpecServeEvent) {
        if matches!(event, nv_models::laguna_serve::SpecServeEvent::Error(message) if message == LAGUNA_WORKER_RETIRED)
        {
            self.admission.retire();
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for LagunaSpecReservation {
    fn drop(&mut self) {
        if std::thread::panicking() {
            self.admission.retire();
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn laguna_serve_draft_enabled() -> bool {
    std::env::var("NV_LAGUNA_SERVE_DRAFT")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
}

#[cfg(feature = "cuda")]
pub(crate) fn laguna_dflash_dir() -> Option<PathBuf> {
    if let Ok(d) = std::env::var("NV_LAGUNA_DFLASH_DIR") {
        let p = PathBuf::from(d);
        if p.is_dir() {
            return Some(p);
        }
    }
    let home = std::env::var("HOME").ok()?;
    let p = PathBuf::from(home)
        .join(".cache/huggingface/hub/models--poolside--Laguna-XS-2.1-DFlash/snapshots/main");
    p.is_dir().then_some(p)
}

#[cfg(feature = "cuda")]
pub(crate) const WHY_THE_FALLBACK_RENDERER_IS_THINKING_OFF: &str =
    "This built-in renderer runs only when chat_template.jinja fails to load; the served path uses \
     NvEngineChat::official_template() when it is available. It deliberately emits \
     <assistant></think>, i.e. thinking-OFF, even though the snapshot's generation_config.json \
     sets default_chat_template_kwargs.enable_thinking=true. Reason: the fallback has no reasoning \
     parser wired to it, so a thinking-ON generation prompt would stream raw reasoning into the \
     assistant content field. The divergence from the shipped default is intentional and it means \
     any quality number taken through this fallback is NOT a measurement of the served \
     configuration. Measure through the official template (see \
     rust/tests/laguna_serve_spec.rs).";

#[cfg(feature = "cuda")]
pub(crate) fn render_laguna_prompt(messages: &[ChatMessageIn]) -> String {
    tracing::debug!("{WHY_THE_FALLBACK_RENDERER_IS_THINKING_OFF}");
    let mut out = String::new();
    out.push_str("〈|EOS|〉");
    let mut idx = 0usize;
    let system_text = if !messages.is_empty()
        && (messages[0].role == "system" || messages[0].role == "developer")
    {
        idx = 1;
        messages[0].text().trim().to_string()
    } else {
        "You are a helpful, conversationally-fluent assistant made by Poolside. \
         You are here to be helpful to users through natural language conversations."
            .to_string()
    };
    if !system_text.is_empty() {
        out.push_str("<system>");
        out.push_str(&system_text);
        out.push_str("</system>\n");
    }
    for m in &messages[idx..] {
        match m.role.as_str() {
            "assistant" => {
                out.push_str("<assistant></think>");
                out.push_str(m.text().trim());
                out.push_str("</assistant>\n");
            }
            "tool" => {
                out.push_str("<tool_response>");
                out.push_str(m.text().trim());
                out.push_str("</tool_response>\n");
            }
            "system" | "developer" => {
                out.push_str("<system>");
                out.push_str(m.text().trim());
                out.push_str("</system>\n");
            }
            _ => {
                out.push_str("<user>");
                out.push_str(m.text().trim());
                out.push_str("</user>\n");
            }
        }
    }
    out.push_str("<assistant></think>");
    out
}

#[cfg(feature = "cuda")]
impl LagunaSpecServe {
    fn request_is_eligible(
        &self,
        req: &ChatGenerateRequest,
        prompt_tokens: usize,
        max_new: usize,
    ) -> bool {
        let params = sampling_params_from(req);
        params.is_greedy()
            && !params.has_penalties()
            && req.guided.is_none()
            && req.logit_bias.is_empty()
            && !req.logprobs
            && self
                .tail
                .fits(prompt_tokens, max_new, self.num_spec, self.max_seq)
    }
}

#[cfg(feature = "cuda")]
pub(crate) struct LagunaText {
    decoder: nv_tokenizer::IncrementalDecoder,
    decoded: String,
    emitter: StreamEmitter,
}

#[cfg(feature = "cuda")]
impl LagunaText {
    pub(crate) fn new(stop: &[String]) -> Self {
        Self {
            decoder: nv_tokenizer::IncrementalDecoder::new(),
            decoded: String::new(),
            emitter: StreamEmitter::new(stop),
        }
    }

    pub(crate) fn push(
        &mut self,
        tokenizer: &tokenizers::Tokenizer,
        id: u32,
    ) -> anyhow::Result<(String, bool)> {
        let Some(piece) = self.decoder.push(tokenizer, id)? else {
            return Ok((String::new(), false));
        };
        self.decoded.push_str(&piece);
        Ok(self.emitter.step(&self.decoded))
    }

    pub(crate) fn finish(&mut self, tokenizer: &tokenizers::Tokenizer) -> anyhow::Result<String> {
        if let Some(tail) = self.decoder.flush(tokenizer)? {
            self.decoded.push_str(&tail);
        }
        Ok(self.emitter.finish(&self.decoded))
    }
}

#[cfg(feature = "cuda")]
const EAGER: &str = "laguna-eager";

#[cfg(feature = "cuda")]
pub(crate) async fn run_sampling_laguna(
    model: LagunaShared,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    spec: Option<Arc<LagunaSpecServe>>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    use candle_core::Tensor;

    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    let prompt_tokens = prompt_ids.len() as u32;

    let max_new = effective_max_new(req.max_new_tokens, default_max_new);

    if prompt_ids.is_empty() || max_new == 0 {
        if tx.send(ChatEvent::Started { prompt_tokens }).await.is_err() {
            return Ok(());
        }
        let _ = tx
            .send(ChatEvent::Done {
                finish_reason: "length".into(),
                completion_tokens: 0,
            })
            .await;
        return Ok(());
    }

    let cache_len = request_window(
        prompt_ids.len(),
        max_new,
        kv_max_seq_len,
        model.config().max_position_embeddings,
    )?;

    if let Some(spec) = spec.as_ref() {
        let eligible = spec.request_is_eligible(&req, prompt_ids.len(), max_new);
        if eligible {
            let deadline = model.1.deadline();
            let Some(permit) = model.1.acquire_spec(tx, deadline).await? else {
                return Ok(());
            };
            run_laguna_spec_stream(
                spec,
                &tokenizer,
                &req,
                &prompt_ids,
                max_new,
                eos_ids,
                tx,
                permit,
                model.1.clone(),
                deadline,
            )
            .await?;
            return Ok(());
        }
    }

    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;
    let Some(_permit) = model.1.acquire(tx).await? else {
        return Ok(());
    };
    let mut cache: Box<dyn nv_models::gemma4::Gemma4Cache + Send> =
        if nv_config::env_arm_present_at_all_reread_every_call("NV_LAGUNA_FP8_KV") {
            match model.new_kv_cache_fp8(cache_len) {
                Ok(c) => Box::new(c),
                Err(e) => {
                    warn!("NV_LAGUNA_FP8_KV set but fp8 KV cache unavailable, using bf16: {e}");
                    Box::new(model.new_kv_cache(cache_len)?)
                }
            }
        } else {
            Box::new(model.new_kv_cache(cache_len)?)
        };

    let mut logits = None;
    let mut offset = 0usize;
    while offset < prompt_ids.len() {
        if tx.is_closed() {
            return Ok(());
        }
        let n = model.next_prefill_chunk_len(prompt_ids.len() - offset);
        let chunk_tokens = Tensor::from_vec(
            prompt_ids[offset..offset + n].to_vec(),
            (1usize, n),
            &device,
        )?;
        let positions: Vec<i32> = (offset as i32..(offset + n) as i32).collect();
        let positions_t = Tensor::from_vec(positions, n, &device)?;
        logits = model
            .prefill_with_cache_aux_scoped(
                &chunk_tokens,
                &positions_t,
                &mut cache,
                &[],
                false,
                offset + n == prompt_ids.len(),
            )?
            .0;
        offset += n;
        yield_so_the_woken_reader_escapes_this_workers_unstealable_lifo_slot().await;
    }
    let logits = logits.ok_or_else(|| anyhow::anyhow!("empty prompt after tokenize"))?;
    let last_row = last_row_logits_3d(&logits)?;
    drop(logits);
    let mut last_out = sampler.sample(&last_row);
    let mut last = last_out.token;
    if last_out.exhausted {
        anyhow::bail!("no legal token at prefill: the sampling mask left every candidate at -inf");
    }

    let cuda_dev = match &device {
        candle_core::Device::Cuda(d) => Some(d.clone()),
        _ => None,
    };
    let mut argmax_scratch = match &cuda_dev {
        Some(d) if sampler.fast_greedy() => Some(LagunaArgmaxScratch::new(d)?),
        _ => None,
    };
    let mut text = LagunaText::new(&req.stop);
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();
    let send_timeout = sse_send_timeout();

    if !push_or_abort(tx, ChatEvent::Started { prompt_tokens }, send_timeout, EAGER).await {
        return Ok(());
    }

    for step in 0..max_new {
        if tx.is_closed() {
            return Ok(());
        }
        if eos_ids.contains(&last) {
            finish_reason = "stop".into();
            break;
        }
        completion_tokens += 1;

        let (piece, stop_hit) = text.push(&tokenizer, last)?;

        if !piece.is_empty()
            && !push_or_abort(tx, ChatEvent::TextDelta(piece), send_timeout, EAGER).await
        {
            return Ok(());
        }
        if req.logprobs
            && !push_or_abort(
                tx,
                ChatEvent::Logprob(build_logprob_entry(&tokenizer, &last_out)),
                send_timeout,
                EAGER,
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
        if cache_len > 0 && nv_models::gemma4::Gemma4Cache::current_len(&cache) >= cache_len {
            finish_reason = "length".into();
            break;
        }

        let pos = (prompt_ids.len() + step) as i32;
        let next_t = Tensor::from_vec(vec![last], (1usize, 1usize), &device)?;
        let pos_t = Tensor::from_vec(vec![pos], 1usize, &device)?;
        let step_logits = model.forward_with_cache(&next_t, &pos_t, &mut cache)?;
        let fast_tok = match (argmax_scratch.as_mut(), &cuda_dev) {
            (Some(sc), Some(dev))
                if matches!(
                    step_logits.dtype(),
                    candle_core::DType::BF16 | candle_core::DType::F32
                ) =>
            {
                Some(laguna_device_argmax(&step_logits, dev, sc)?)
            }
            _ => None,
        };
        last_out = match fast_tok {
            Some(tok) => {
                sampler.record_token(tok);
                SampleOutput {
                    token: tok,
                    logprob: None,
                    top: Vec::new(),
                    exhausted: false,
                }
            }
            None => {
                let step_row = last_row_logits_3d(&step_logits)?;
                sampler.sample(&step_row)
            }
        };
        drop(step_logits);
        last = last_out.token;
        if last_out.exhausted {
            anyhow::bail!(
                "no legal token at step {step}: the sampling mask left every candidate at -inf"
            );
        }
    }

    let tail = text.finish(&tokenizer)?;
    if !tail.is_empty() && !push_or_abort(tx, ChatEvent::TextDelta(tail), send_timeout, EAGER).await
    {
        return Ok(());
    }

    let _ = send_stopped_by_then_done(
        tx,
        &mut text.emitter,
        finish_reason,
        completion_tokens,
        Some(send_timeout),
    )
    .await;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) async fn run_laguna_spec_stream(
    spec: &LagunaSpecServe,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    prompt_ids: &[u32],
    max_new: usize,
    eos_ids: &[u32],
    tx: &mpsc::Sender<ChatEvent>,
    permit: tokio::sync::OwnedRwLockReadGuard<()>,
    admission: Arc<LagunaAdmission>,
    deadline: tokio::time::Instant,
) -> anyhow::Result<()> {
    use nv_models::laguna_serve::{SpecServeEvent, SpecServeJob};

    let (ev_tx, mut ev_rx) = tokio::sync::mpsc::channel::<SpecServeEvent>(64);
    let (admitted_tx, admitted_rx) = tokio::sync::oneshot::channel();
    let mut admitted_tx = Some(admitted_tx);
    let mut delivery_failed = false;
    let client_tx = tx.clone();
    admission.ensure_available()?;
    let reservation = LagunaSpecReservation {
        admission,
        _permit: permit,
    };
    let job = SpecServeJob {
        prompt_ids: prompt_ids.to_vec(),
        max_new,
        eos_ids: eos_ids.to_vec(),
        emit: Box::new(move |ev| {
            reservation.observe(&ev);
            if delivery_failed || client_tx.is_closed() || ev_tx.is_closed() {
                return false;
            }
            if matches!(&ev, SpecServeEvent::Tokens(tokens) if tokens.is_empty()) {
                if let Some(admitted_tx) = admitted_tx.take() {
                    return admitted_tx.send(()).is_ok();
                }
                return true;
            }
            if ev_tx.try_send(ev).is_err() {
                delivery_failed = true;
                return false;
            }
            true
        }),
    };
    spec.jobs
        .try_send(job)
        .map_err(|_| crate::oapi::chat::EngineBusy::new(spec.capacity, 0))?;
    tokio::select! {
        biased;
        _ = tx.closed() => return Ok(()),
        admitted = tokio::time::timeout_at(deadline, admitted_rx) => match admitted {
            Ok(Ok(())) => {},
            Ok(Err(_)) => anyhow::bail!(crate::oapi::chat::ENGINE_INTERRUPTED),
            Err(_) => return Err(crate::oapi::chat::EngineBusy::new(spec.capacity, spec.queue.as_millis() as u64).into()),
        }
    }

    let mut text = LagunaText::new(&req.stop);
    let mut completion_tokens: u32 = 0;
    let mut finish_reason: Option<String> = None;
    let mut started = false;
    let mut done = false;
    let mut local_stop = false;

    'outer: loop {
        let event = tokio::select! {
            biased;
            _ = tx.closed() => return Ok(()),
            event = ev_rx.recv() => event,
        };
        let Some(ev) = event else {
            break;
        };
        match ev {
            SpecServeEvent::Tokens(toks) => {
                if toks.is_empty() || finish_reason.is_some() {
                    continue;
                }
                if !started {
                    if tx
                        .send(ChatEvent::Started {
                            prompt_tokens: prompt_ids.len() as u32,
                        })
                        .await
                        .is_err()
                    {
                        return Ok(());
                    }
                    started = true;
                }
                for t in toks {
                    if eos_ids.contains(&t) {
                        finish_reason = Some("stop".into());
                        break;
                    }
                    completion_tokens += 1;
                    let (piece, stop_hit) = text.push(tokenizer, t)?;
                    if !piece.is_empty() && tx.send(ChatEvent::TextDelta(piece)).await.is_err() {
                        return Ok(());
                    }
                    if stop_hit {
                        finish_reason = Some("stop".into());
                        local_stop = true;
                        break 'outer;
                    }
                    if completion_tokens as usize >= max_new {
                        finish_reason = Some("length".into());
                        break;
                    }
                }
            }
            SpecServeEvent::PromptCached(cached) => {
                if !started
                    && cached > 0
                    && tx
                        .send(ChatEvent::PromptCached {
                            cached_tokens: cached as u32,
                        })
                        .await
                        .is_err()
                {
                    return Ok(());
                }
            }
            SpecServeEvent::Done => {
                done = true;
                break 'outer;
            }
            SpecServeEvent::Error(e) => {
                anyhow::bail!(e);
            }
        }
    }
    if (!done && !local_stop)
        || (done && finish_reason.is_none() && max_new > 0 && !prompt_ids.is_empty())
    {
        anyhow::bail!(crate::oapi::chat::ENGINE_INTERRUPTED);
    }
    if !started
        && tx
            .send(ChatEvent::Started {
                prompt_tokens: prompt_ids.len() as u32,
            })
            .await
            .is_err()
    {
        return Ok(());
    }

    let tail = text.finish(tokenizer)?;
    if !tail.is_empty() && tx.send(ChatEvent::TextDelta(tail)).await.is_err() {
        return Ok(());
    }
    let _ = send_stopped_by_then_done(
        tx,
        &mut text.emitter,
        finish_reason.unwrap_or_else(|| "length".into()),
        completion_tokens,
        None,
    )
    .await;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) struct LagunaArgmaxScratch {
    part_val: nv_layers::cudarc::driver::CudaSlice<f32>,
    part_idx: nv_layers::cudarc::driver::CudaSlice<i32>,
    token_out: nv_layers::cudarc::driver::CudaSlice<u32>,
}

#[cfg(feature = "cuda")]
impl LagunaArgmaxScratch {
    fn new(dev: &candle_core::CudaDevice) -> anyhow::Result<Self> {
        let stream = nv_layers::cuda_stream::current_stream(dev);
        let parts = nv_kernels::cuda::argmax_parts();
        Ok(Self {
            part_val: stream
                .alloc_zeros::<f32>(parts)
                .map_err(|e| anyhow::anyhow!("alloc part_val: {e:?}"))?,
            part_idx: stream
                .alloc_zeros::<i32>(parts)
                .map_err(|e| anyhow::anyhow!("alloc part_idx: {e:?}"))?,
            token_out: stream
                .alloc_zeros::<u32>(1)
                .map_err(|e| anyhow::anyhow!("alloc token_out: {e:?}"))?,
        })
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn laguna_device_argmax(
    logits: &candle_core::Tensor,
    dev: &candle_core::CudaDevice,
    scratch: &mut LagunaArgmaxScratch,
) -> anyhow::Result<u32> {
    use nv_layers::cudarc::driver::{DevicePtr, DevicePtrMut};
    let dims = logits.dims();
    anyhow::ensure!(
        dims.len() == 3 && dims[0] == 1,
        "expected logits shape [1, seq, vocab], got {:?}",
        dims
    );
    let last = logits.i((0usize, dims[1] - 1, ..))?.contiguous()?;
    let last = if last.dtype() == candle_core::DType::BF16 {
        last
    } else {
        last.to_dtype(candle_core::DType::BF16)?
    };
    let n = last.elem_count();
    let stream = nv_layers::cuda_stream::current_stream(dev);
    let rc = {
        let (storage, layout) = last.storage_and_layout();
        let cuda = match &*storage {
            candle_core::Storage::Cuda(s) => s,
            _ => anyhow::bail!("logits not on cuda"),
        };
        let slice = match &cuda.slice {
            candle_core::cuda::CudaStorageSlice::BF16(s) => s,
            _ => anyhow::bail!("device argmax expects bf16 logits"),
        };
        let (lp, _gl) = slice.device_ptr(&stream);
        let lp = lp + (layout.start_offset() * 2) as u64;
        let (vp, _gv) = scratch.part_val.device_ptr_mut(&stream);
        let (ip, _gi) = scratch.part_idx.device_ptr_mut(&stream);
        let (tp, _gt) = scratch.token_out.device_ptr_mut(&stream);
        unsafe {
            nv_kernels::cuda::argmax_bf16(
                stream.cu_stream() as *mut _,
                lp as *const u16,
                n as i32,
                vp as *mut f32,
                ip as *mut i32,
                std::ptr::null(),
                tp as *mut u32,
                std::ptr::null_mut(),
                0,
            )
        }
    };
    anyhow::ensure!(rc == 0, "argmax_bf16 returned {rc}");
    let out = stream
        .clone_dtoh(&scratch.token_out)
        .map_err(|e| anyhow::anyhow!("dtoh token: {e:?}"))?;
    stream
        .synchronize()
        .map_err(|e| anyhow::anyhow!("sync: {e:?}"))?;
    Ok(out[0])
}

#[cfg(all(test, feature = "cuda"))]
mod overload_tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use axum::response::Response;
    use axum::routing::post;
    use axum::Router;
    use nv_models::laguna_serve::{SpecServeEvent, SpecServeJob};
    use serde_json::{json, Value};
    use std::time::Duration;
    use tower::ServiceExt;

    const SURFACES: [(&str, u16); 3] = [
        ("/v1/chat/completions", 503),
        ("/v1/responses", 503),
        ("/v1/messages", 529),
    ];

    struct FixtureEngine {
        spec: Arc<LagunaSpecServe>,
        tokenizer: Arc<tokenizers::Tokenizer>,
        admission: Arc<LagunaAdmission>,
        eager: std::sync::mpsc::SyncSender<(
            tokio::sync::OwnedRwLockWriteGuard<()>,
            mpsc::Sender<ChatEvent>,
        )>,
    }

    #[async_trait::async_trait]
    impl ChatEngine for FixtureEngine {
        fn model_id(&self) -> &str {
            "laguna-overload"
        }

        async fn generate(
            &self,
            req: ChatGenerateRequest,
            tx: mpsc::Sender<ChatEvent>,
        ) -> anyhow::Result<()> {
            let spec = self.spec.clone();
            let tokenizer = self.tokenizer.clone();
            let admission = self.admission.clone();
            let eager = self.eager.clone();
            tokio::spawn(async move {
                let result: anyhow::Result<()> = async {
                    request_window(1, req.max_new_tokens, spec.max_seq, spec.max_seq)?;
                    if req.temperature.is_some_and(|t| t > 0.0) {
                        let Some(permit) = admission.acquire(&tx).await? else {
                            return Ok(());
                        };
                        eager
                            .try_send((permit, tx.clone()))
                            .map_err(|_| crate::oapi::chat::EngineBusy::new(1, 0))?;
                        return Ok(());
                    }
                    let deadline = admission.deadline();
                    let Some(permit) = admission.acquire_spec(&tx, deadline).await? else {
                        return Ok(());
                    };
                    run_laguna_spec_stream(
                        &spec,
                        &tokenizer,
                        &req,
                        &[0],
                        req.max_new_tokens,
                        &[2],
                        &tx,
                        permit,
                        admission.clone(),
                        deadline,
                    )
                    .await
                }
                .await;
                if let Err(error) = result {
                    let _ = tx.send(ChatEvent::Error(format!("{error:#}"))).await;
                }
            });
            Ok(())
        }
    }

    fn capacity_fixture(tail: LagunaWorkerTail) -> LagunaSpecServe {
        let (jobs, _) = std::sync::mpsc::sync_channel(1);
        LagunaSpecServe {
            jobs: LagunaSpecWorker::from_test_sender(jobs),
            max_seq: 262144,
            num_spec: 2,
            tail,
            capacity: 2,
            queue: Duration::from_millis(30),
        }
    }

    fn capacity_greedy_request() -> ChatGenerateRequest {
        ChatGenerateRequest {
            prompt: "prompt".into(),
            max_new_tokens: 256,
            stop: vec![],
            seed: None,
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

    #[test]
    fn laguna_prepared_worker_capacity_tail_keeps_exact_cap_on_worker() {
        let req = capacity_greedy_request();
        for tail in [
            LagunaWorkerTail::PreparedPlain,
            LagunaWorkerTail::PreparedSpec,
        ] {
            let spec = capacity_fixture(tail);
            for (prompt, output, expected) in [
                (168000, 256, true),
                (261884, 256, true),
                (261885, 256, true),
                (261888, 256, true),
                (261889, 256, false),
                (262142, 2, true),
                (262143, 2, false),
                (usize::MAX, 256, false),
                (1, usize::MAX, false),
                (0, 256, false),
                (262144, 1, false),
            ] {
                assert_eq!(
                    spec.request_is_eligible(&req, prompt, output),
                    expected,
                    "worker capacity route {tail:?}, prompt={prompt}, output={output}"
                );
            }
            assert!(request_window(261888, 256, 262144, 262144).is_ok());
            assert!(request_window(261889, 256, 262144, 262144).is_err());
            assert!(spec.request_is_eligible(&req, 262142, 1));
            assert_eq!(
                spec.request_is_eligible(&req, 262143, 1),
                tail == LagunaWorkerTail::PreparedSpec,
                "prepared plain admission requires at least two remaining cache slots"
            );
        }
    }

    #[test]
    fn laguna_legacy_worker_capacity_tail_preserves_headroom_and_sampling_guards() {
        let req = capacity_greedy_request();
        let mut legacy = capacity_fixture(LagunaWorkerTail::Reserved);
        for (k, accepted_prompt, refused_prompt) in [
            (0, 261886, 261887),
            (2, 261884, 261885),
            (4, 261882, 261883),
        ] {
            legacy.num_spec = k;
            assert!(legacy.request_is_eligible(&req, accepted_prompt, 256));
            assert!(!legacy.request_is_eligible(&req, refused_prompt, 256));
            assert!(!legacy.request_is_eligible(&req, 261888, 256));
        }
        legacy.num_spec = usize::MAX;
        assert!(!legacy.request_is_eligible(&req, 1, 1));
        for tail in [
            LagunaWorkerTail::Reserved,
            LagunaWorkerTail::PreparedPlain,
            LagunaWorkerTail::PreparedSpec,
        ] {
            let spec = capacity_fixture(tail);
            let mut req = capacity_greedy_request();
            assert!(spec.request_is_eligible(&req, 168000, 256));
            req.temperature = Some(0.7);
            assert!(!spec.request_is_eligible(&req, 168000, 256));
            req.temperature = Some(0.0);
            req.presence_penalty = Some(0.5);
            assert!(!spec.request_is_eligible(&req, 168000, 256));
            req.presence_penalty = None;
            req.guided = Some(nv_grammar::GrammarSpec::Regex("answer".into()));
            assert!(!spec.request_is_eligible(&req, 168000, 256));
            req.guided = None;
            req.logit_bias = vec![(1, 0.5)];
            assert!(!spec.request_is_eligible(&req, 168000, 256));
            req.logit_bias.clear();
            req.logprobs = true;
            assert!(!spec.request_is_eligible(&req, 168000, 256));
            req.logprobs = false;
            assert!(spec.request_is_eligible(&req, 168000, 256));
        }
    }

    fn fixture(queue_ms: u64, max_seq: usize) -> (Router, std::sync::mpsc::Receiver<SpecServeJob>) {
        let (app, jobs, _, _) = mixed_fixture(queue_ms, max_seq);
        (app, jobs)
    }

    fn mixed_fixture(
        queue_ms: u64,
        max_seq: usize,
    ) -> (
        Router,
        std::sync::mpsc::Receiver<SpecServeJob>,
        std::sync::mpsc::Receiver<(
            tokio::sync::OwnedRwLockWriteGuard<()>,
            mpsc::Sender<ChatEvent>,
        )>,
        Arc<LagunaAdmission>,
    ) {
        let (jobs, rx) = std::sync::mpsc::sync_channel(1);
        let (eager, eager_rx) = std::sync::mpsc::sync_channel(1);
        let spec = Arc::new(LagunaSpecServe {
            jobs: LagunaSpecWorker::from_test_sender(jobs),
            max_seq,
            num_spec: 4,
            tail: LagunaWorkerTail::Reserved,
            capacity: 1,
            queue: Duration::from_millis(queue_ms),
        });
        let admission = Arc::new(LagunaAdmission::new(1, Duration::from_millis(queue_ms)));
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("prompt".into(), 0),
                    ("answer".into(), 1),
                    ("end".into(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .build()
            .unwrap();
        let engine = Arc::new(FixtureEngine {
            spec,
            tokenizer: Arc::new(tokenizers::Tokenizer::new(model)),
            admission: admission.clone(),
            eager,
        });
        let app = Router::new()
            .route(
                "/v1/chat/completions",
                post(crate::oapi::chat::handle_chat_completions),
            )
            .route(
                "/v1/responses",
                post(crate::oapi::responses::handle_responses),
            )
            .route("/v1/messages", post(crate::oapi::messages::handle_messages))
            .with_state(crate::oapi::chat::ChatAppState {
                registry: ChatRegistry::single(engine),
            });
        (app, rx, eager_rx, admission)
    }

    async fn request(app: Router, path: &str, stream: bool, deadline: Option<u64>) -> Response {
        mixed_request(app, path, stream, deadline, false).await
    }

    async fn mixed_request(
        app: Router,
        path: &str,
        stream: bool,
        deadline: Option<u64>,
        eager: bool,
    ) -> Response {
        let mut payload = if path == "/v1/responses" {
            json!({"model":"laguna-overload", "input":"hello", "max_output_tokens":8, "stream":stream})
        } else {
            json!({"model":"laguna-overload", "messages":[{"role":"user","content":"hello"}], "max_tokens":8, "stream":stream})
        };
        payload["temperature"] = json!(if eager { 0.7 } else { 0.0 });
        let mut request = Request::builder()
            .method("POST")
            .uri(path)
            .header("content-type", "application/json")
            .header("anthropic-version", "2023-06-01");
        if let Some(ms) = deadline {
            request = request.header("x-request-timeout-ms", ms.to_string());
        }
        tokio::time::timeout(
            Duration::from_secs(2),
            app.oneshot(request.body(Body::from(payload.to_string())).unwrap()),
        )
        .await
        .unwrap()
        .unwrap()
    }

    async fn body(response: Response) -> String {
        let bytes = tokio::time::timeout(
            Duration::from_secs(2),
            axum::body::to_bytes(response.into_body(), 1 << 20),
        )
        .await
        .unwrap()
        .unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    async fn busy(response: Response, status: u16) {
        let actual = response.status().as_u16();
        let headers = response.headers().clone();
        let text = body(response).await;
        assert_eq!(actual, status, "{text}");
        assert_eq!(headers.get("retry-after").unwrap(), "1", "{text}");
        assert_eq!(
            headers.get("content-type").unwrap(),
            "application/json",
            "failure committed SSE headers: {text}"
        );
        let value: Value = serde_json::from_str(&text).unwrap();
        assert!(
            value["error"]["message"]
                .as_str()
                .unwrap()
                .contains("try later"),
            "{text}"
        );
    }

    async fn take_job<T>(rx: &std::sync::mpsc::Receiver<T>) -> T {
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                match rx.try_recv() {
                    Ok(job) => return job,
                    Err(std::sync::mpsc::TryRecvError::Empty) => tokio::task::yield_now().await,
                    Err(error) => panic!("worker queue closed: {error}"),
                }
            }
        })
        .await
        .unwrap()
    }

    fn admit(job: &mut SpecServeJob) -> bool {
        job.client_is_live()
    }

    #[tokio::test]
    async fn laguna_worker_observes_client_close_without_waiting_for_the_bridge_to_run() {
        for already_admitted in [false, true] {
            let gate = Arc::new(LagunaAdmission::new(1, Duration::from_secs(1)));
            let (tx, rx) = mpsc::channel(1);
            let deadline = gate.deadline();
            let permit = gate.acquire_spec(&tx, deadline).await.unwrap().unwrap();
            let (jobs, queued) = std::sync::mpsc::sync_channel(1);
            let spec = LagunaSpecServe {
                jobs: LagunaSpecWorker::from_test_sender(jobs),
                max_seq: 64,
                num_spec: 4,
                tail: LagunaWorkerTail::Reserved,
                capacity: 1,
                queue: gate.queue,
            };
            let tokenizer = Arc::new(tokenizers::Tokenizer::new(
                tokenizers::models::wordlevel::WordLevel::builder()
                    .build()
                    .unwrap(),
            ));
            let req = ChatGenerateRequest {
                prompt: "prompt".into(),
                max_new_tokens: 8,
                stop: vec![],
                seed: None,
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
            };
            let mut bridge = Box::pin(run_laguna_spec_stream(
                &spec,
                &tokenizer,
                &req,
                &[0],
                8,
                &[2],
                &tx,
                permit,
                gate.clone(),
                deadline,
            ));
            assert!(futures_util::poll!(bridge.as_mut()).is_pending());
            let mut job = queued.try_recv().unwrap();
            if already_admitted {
                assert!(admit(&mut job));
            }
            drop(rx);
            assert!(!admit(&mut job), "closed HTTP receiver must cancel queued and admitted work before the bridge is polled again");
            assert!(
                gate.active.try_write().is_err(),
                "client cancellation must retain the worker reservation"
            );
            drop(bridge);
            assert!(
                gate.active.try_write().is_err(),
                "bridge cleanup must retain the worker reservation"
            );
            drop(job);
            assert!(gate.active.try_write().is_ok());
        }
    }

    #[tokio::test]
    async fn laguna_http_queue_is_bounded_cancels_waiters_and_recovers() {
        for (path, status) in SURFACES {
            let (app, rx) = fixture(20, 64);
            let (a, b) = tokio::join!(
                request(app.clone(), path, true, None),
                request(app.clone(), path, false, None)
            );
            busy(a, status).await;
            busy(b, status).await;
            let mut expired = take_job(&rx).await;
            assert!(
                !admit(&mut expired),
                "expired queued work must never start prefill"
            );
            assert!(
                rx.try_recv().is_err(),
                "bounded queue retained more than one job"
            );
            drop(expired);
            let recovery = tokio::spawn(request(app, path, false, None));
            let mut job = take_job(&rx).await;
            assert!(admit(&mut job));
            assert!((job.emit)(SpecServeEvent::Tokens(vec![1, 2])));
            assert!((job.emit)(SpecServeEvent::Done));
            let response = recovery.await.unwrap();
            assert_eq!(response.status(), 200);
            assert!(body(response).await.contains("answer"));
        }
    }

    #[tokio::test]
    async fn laguna_http_prefill_deadline_and_oom_precede_stream_headers() {
        for (path, status) in SURFACES {
            for oom in [false, true] {
                let (app, rx) = fixture(500, 64);
                let response = tokio::spawn(request(app, path, true, Some(20)));
                let mut job = take_job(&rx).await;
                assert!(admit(&mut job));
                if oom {
                    assert!((job.emit)(SpecServeEvent::Error(
                        "CUDA_ERROR_OUT_OF_MEMORY".into()
                    )));
                }
                busy(response.await.unwrap(), status).await;
                assert!(
                    !admit(&mut job),
                    "caller exit must cancel admitted prefill at the next worker check"
                );
            }
        }
    }

    #[tokio::test]
    async fn laguna_http_typed_execution_failure_is_retryable_before_and_after_output() {
        for (path, status) in SURFACES {
            for stream in [false, true] {
                for after_output in [false, true] {
                    let (app, jobs, _, gate) = mixed_fixture(500, 64);
                    let first = tokio::spawn(request(app.clone(), path, stream, None));
                    let mut job = take_job(&jobs).await;
                    assert!(admit(&mut job));
                    if after_output {
                        assert!((job.emit)(SpecServeEvent::Tokens(vec![1])));
                    }
                    let error =
                        anyhow::anyhow!("Laguna FMHAv2 FP8 requires an explicit CUDA stream")
                            .context(nv_models::laguna_serve::LagunaRequestExecutionFailure);
                    assert!((job.emit)(SpecServeEvent::Error(format!("{error:#}"))));
                    let response = first.await.unwrap();
                    if stream && after_output {
                        assert_eq!(response.status(), 200);
                        assert!(response
                            .headers()
                            .get("content-type")
                            .unwrap()
                            .to_str()
                            .unwrap()
                            .starts_with("text/event-stream"));
                        let text = body(response).await;
                        assert!(
                            text.contains("answer")
                                && text.contains("requires an explicit CUDA stream"),
                            "{text}"
                        );
                        assert!(
                            text.contains("try later")
                                && (text.contains("engine_busy")
                                    || text.contains("overloaded_error")),
                            "{text}"
                        );
                        assert!(
                            !text.contains("response.completed") && !text.contains("message_stop"),
                            "{text}"
                        );
                        assert!(
                            !text.contains("\"finish_reason\":\"stop\"")
                                && !text.contains("\"finish_reason\":\"length\""),
                            "{text}"
                        );
                        if path == "/v1/chat/completions" {
                            assert!(text.contains("\"retry_after\":1"), "{text}");
                            assert_eq!(text.matches("data: [DONE]").count(), 1, "{text}");
                        }
                    } else {
                        busy(response, status).await;
                    }
                    assert!(
                        !gate.waiting.is_closed(),
                        "a request failure is not an authoritative retirement marker"
                    );
                    assert!(
                        gate.active.try_write().is_err(),
                        "error delivery released a still-owned GPU reservation"
                    );
                    if stream && after_output {
                        let (spec_waiter, eager_waiter) = tokio::join!(
                            request(app.clone(), path, false, Some(20)),
                            mixed_request(app.clone(), path, false, Some(20), true),
                        );
                        busy(spec_waiter, status).await;
                        busy(eager_waiter, status).await;
                        let mut expired = take_job(&jobs).await;
                        assert!(!admit(&mut expired), "a queued caller that timed out must not execute after a stalled worker returns");
                        drop(expired);
                        assert!(
                            gate.active.try_write().is_err(),
                            "waiter refusal released the failing worker's reservation"
                        );
                    }
                    drop(job);
                    let recovery = tokio::spawn(request(app.clone(), path, false, None));
                    let mut next = take_job(&jobs).await;
                    assert!(admit(&mut next));
                    assert!((next.emit)(SpecServeEvent::Tokens(vec![1, 2])));
                    assert!((next.emit)(SpecServeEvent::Done));
                    let response = recovery.await.unwrap();
                    assert_eq!(response.status(), 200);
                    assert!(body(response).await.contains("answer"));
                    drop(next);
                }
            }
        }
    }

    #[tokio::test]
    async fn laguna_http_missing_done_is_an_error_and_done_releases_response() {
        for (path, status) in SURFACES {
            for stream in [false, true] {
                let (app, rx) = fixture(500, 64);
                let response = tokio::spawn(request(app, path, stream, None));
                let mut job = take_job(&rx).await;
                assert!(admit(&mut job));
                assert!((job.emit)(SpecServeEvent::Tokens(vec![1])));
                drop(job);
                let response = response.await.unwrap();
                if stream {
                    assert_eq!(response.status(), 200);
                    let text = body(response).await;
                    assert!(text.contains("try later"), "{text}");
                    assert!(
                        !text.contains("response.completed") && !text.contains("message_stop"),
                        "{text}"
                    );
                } else {
                    busy(response, status).await;
                }
            }
        }
    }

    #[tokio::test]
    async fn laguna_http_early_done_and_dropped_token_delivery_never_become_success() {
        for (path, status) in SURFACES {
            for overflow in [false, true] {
                let (app, rx) = fixture(500, 64);
                let response = tokio::spawn(request(app, path, false, None));
                let mut job = take_job(&rx).await;
                assert!(admit(&mut job));
                if overflow {
                    for _ in 0..64 {
                        assert!((job.emit)(SpecServeEvent::Tokens(vec![1])));
                    }
                    assert!(!(job.emit)(SpecServeEvent::Tokens(vec![1])));
                    tokio::task::yield_now().await;
                    assert!(
                        !(job.emit)(SpecServeEvent::Done),
                        "a dropped token must permanently poison this response"
                    );
                } else {
                    assert!((job.emit)(SpecServeEvent::Tokens(vec![1])));
                    assert!((job.emit)(SpecServeEvent::Done));
                }
                drop(job);
                busy(response.await.unwrap(), status).await;
            }
        }
    }

    #[tokio::test]
    async fn laguna_http_configured_context_ceiling_remains_a_client_error() {
        for (path, _) in SURFACES {
            let (app, rx) = fixture(20, 8);
            let response = request(app, path, true, None).await;
            assert_eq!(response.status(), 400);
            assert!(body(response).await.contains("KV window"));
            assert!(rx.try_recv().is_err());
        }
        assert_eq!(request_window(168000, 64, 262144, 262144).unwrap(), 262144);
        assert!(request_window(168000, 64, 8192, 262144).is_err());
    }

    #[tokio::test]
    async fn laguna_mixed_routes_hold_reservations_until_workers_exit_and_recover() {
        for (path, status) in SURFACES {
            let (app, jobs, eager_jobs, gate) = mixed_fixture(30, 64);
            let first = tokio::spawn(request(app.clone(), path, true, Some(15)));
            let mut spec_job = take_job(&jobs).await;
            assert!(admit(&mut spec_job));
            busy(first.await.unwrap(), status).await;
            assert!(!admit(&mut spec_job));
            assert!(
                gate.active.try_write().is_err(),
                "canceled spec worker released its reservation before exit"
            );
            busy(
                mixed_request(app.clone(), path, false, None, true).await,
                status,
            )
            .await;
            assert!(
                eager_jobs.try_recv().is_err(),
                "eager work entered while the spec worker was still running"
            );
            drop(spec_job);

            let response = tokio::spawn(mixed_request(app.clone(), path, false, None, true));
            let eager_job = take_job(&eager_jobs).await;
            eager_job
                .1
                .send(ChatEvent::Started { prompt_tokens: 1 })
                .await
                .unwrap();
            eager_job
                .1
                .send(ChatEvent::TextDelta("answer".into()))
                .await
                .unwrap();
            eager_job
                .1
                .send(ChatEvent::Done {
                    finish_reason: "stop".into(),
                    completion_tokens: 1,
                })
                .await
                .unwrap();
            let response = response.await.unwrap();
            assert_eq!(response.status(), 200);
            assert!(body(response).await.contains("answer"));
            busy(request(app.clone(), path, true, None).await, status).await;
            assert!(
                jobs.try_recv().is_err(),
                "spec prefill entered while the eager worker retained shared scratch"
            );
            drop(eager_job);

            let recovery = tokio::spawn(request(app, path, false, None));
            let mut job = take_job(&jobs).await;
            assert!(admit(&mut job));
            assert!((job.emit)(SpecServeEvent::Tokens(vec![1, 2])));
            assert!((job.emit)(SpecServeEvent::Done));
            let response = recovery.await.unwrap();
            assert_eq!(response.status(), 200);
            assert!(body(response).await.contains("answer"));
            assert!(
                gate.active.try_write().is_err(),
                "Done must not release a job that its worker still owns"
            );
            drop(job);
            assert!(gate.active.try_write().is_ok());
        }
    }

    async fn wait_for_mixed_waiter(gate: &LagunaAdmission) {
        tokio::time::timeout(Duration::from_secs(1), async {
            while gate.waiting.available_permits() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("eager request never reached the shared admission wait");
    }

    #[tokio::test]
    async fn laguna_retirement_blocks_waiting_and_fresh_routes_until_a_new_engine() {
        for (path, status) in SURFACES {
            for stream in [false, true] {
                let (app, jobs, eager_jobs, gate) = mixed_fixture(500, 64);
                let first = tokio::spawn(request(app.clone(), path, stream, None));
                let mut job = take_job(&jobs).await;
                assert!(admit(&mut job));
                assert!((job.emit)(SpecServeEvent::Tokens(vec![1])));
                let waiting = tokio::spawn(mixed_request(app.clone(), path, false, None, true));
                wait_for_mixed_waiter(&gate).await;
                assert!((job.emit)(SpecServeEvent::Error(
                    LAGUNA_WORKER_RETIRED.into()
                )));
                assert!(
                    gate.waiting.is_closed(),
                    "retirement must publish synchronously before the callback returns"
                );
                assert!(
                    gate.active.try_write().is_err(),
                    "Error cannot release a worker's reservation"
                );
                let response = first.await.unwrap();
                if stream {
                    assert_eq!(response.status(), 200);
                    let text = body(response).await;
                    assert!(text.contains(LAGUNA_WORKER_RETIRED), "{text}");
                    assert!(
                        text.contains("engine_busy") || text.contains("overloaded_error"),
                        "{text}"
                    );
                    assert!(
                        !text.contains("response.completed")
                            && !text.contains("message_stop")
                            && !text.contains("\"finish_reason\":\"stop\""),
                        "{text}"
                    );
                } else {
                    busy(response, status).await;
                }
                for eager in [false, true] {
                    busy(
                        mixed_request(app.clone(), path, true, None, eager).await,
                        status,
                    )
                    .await;
                }
                assert!(jobs.try_recv().is_err() && eager_jobs.try_recv().is_err());
                drop(job);
                busy(waiting.await.unwrap(), status).await;
                assert!(
                    gate.active.try_write().is_ok(),
                    "worker exit must still release the physical lock"
                );
                for eager in [false, true] {
                    busy(
                        mixed_request(app.clone(), path, false, None, eager).await,
                        status,
                    )
                    .await;
                }
                assert!(
                    jobs.try_recv().is_err() && eager_jobs.try_recv().is_err(),
                    "retired admission let work reach a backend"
                );

                let (fresh, fresh_jobs, _, fresh_gate) = mixed_fixture(500, 64);
                assert!(
                    !fresh_gate.waiting.is_closed(),
                    "a new engine needs independent healthy admission"
                );
                let recovery = tokio::spawn(request(fresh, path, false, None));
                let mut fresh_job = take_job(&fresh_jobs).await;
                assert!(admit(&mut fresh_job));
                assert!((fresh_job.emit)(SpecServeEvent::Tokens(vec![1, 2])));
                assert!((fresh_job.emit)(SpecServeEvent::Done));
                let response = recovery.await.unwrap();
                assert_eq!(response.status(), 200);
                assert!(body(response).await.contains("answer"));
                assert!(
                    gate.waiting.is_closed(),
                    "replacement must not reset the old engine"
                );
            }
        }
    }

    #[tokio::test]
    async fn laguna_retirement_is_observed_even_after_delivery_failure_or_client_exit() {
        for (path, status) in SURFACES {
            for overflow in [false, true] {
                let (app, jobs, eager_jobs, gate) = mixed_fixture(500, 64);
                let first = tokio::spawn(request(
                    app.clone(),
                    path,
                    false,
                    if overflow { None } else { Some(20) },
                ));
                let mut job = take_job(&jobs).await;
                assert!(admit(&mut job));
                if overflow {
                    for _ in 0..64 {
                        assert!((job.emit)(SpecServeEvent::Tokens(vec![1])));
                    }
                    assert!(!(job.emit)(SpecServeEvent::Tokens(vec![1])));
                    assert!(!(job.emit)(SpecServeEvent::Error(
                        LAGUNA_WORKER_RETIRED.into()
                    )));
                    assert!(
                        gate.waiting.is_closed(),
                        "delivery failure suppressed retirement"
                    );
                    drop(job);
                    busy(first.await.unwrap(), status).await;
                } else {
                    busy(first.await.unwrap(), status).await;
                    tokio::time::timeout(Duration::from_secs(1), async {
                        while admit(&mut job) {
                            tokio::task::yield_now().await;
                        }
                    })
                    .await
                    .expect("timed-out caller did not close its event receiver");
                    assert!(!(job.emit)(SpecServeEvent::Error(
                        LAGUNA_WORKER_RETIRED.into()
                    )));
                    assert!(
                        gate.waiting.is_closed(),
                        "client exit suppressed retirement"
                    );
                    assert!(
                        gate.active.try_write().is_err(),
                        "closed caller released a still-owned reservation"
                    );
                    drop(job);
                }
                for eager in [false, true] {
                    busy(
                        mixed_request(app.clone(), path, false, None, eager).await,
                        status,
                    )
                    .await;
                }
                assert!(jobs.try_recv().is_err() && eager_jobs.try_recv().is_err());
            }
        }
    }

    #[tokio::test]
    async fn laguna_only_the_exact_retirement_marker_closes_shared_admission() {
        for (path, status) in SURFACES {
            let (app, jobs, _, gate) = mixed_fixture(500, 64);
            for (message, retryable) in [
                (
                    "cuda error: an illegal memory access was encountered",
                    false,
                ),
                ("tokenizer: unknown token", false),
                ("CUDA_ERROR_OUT_OF_MEMORY", true),
                (
                    "engine busy: Laguna worker retired; try later. trailing diagnostic",
                    true,
                ),
            ] {
                let first = tokio::spawn(request(app.clone(), path, false, None));
                let mut job = take_job(&jobs).await;
                assert!(admit(&mut job));
                assert!((job.emit)(SpecServeEvent::Error(message.into())));
                let response = first.await.unwrap();
                if retryable {
                    busy(response, status).await;
                } else {
                    assert_eq!(response.status(), 500);
                    assert!(response.headers().get("retry-after").is_none());
                    assert!(body(response).await.contains(message));
                }
                assert!(
                    !gate.waiting.is_closed(),
                    "unmarked error changed engine health: {message}"
                );
                drop(job);
            }
            let recovery = tokio::spawn(request(app, path, false, None));
            let mut job = take_job(&jobs).await;
            assert!(admit(&mut job));
            assert!((job.emit)(SpecServeEvent::Tokens(vec![1, 2])));
            assert!((job.emit)(SpecServeEvent::Done));
            assert_eq!(recovery.await.unwrap().status(), 200);
        }
    }

    #[tokio::test]
    async fn laguna_retirement_rechecks_a_reader_after_its_waited_acquisition() {
        let gate = LagunaAdmission::new(1, Duration::from_millis(500));
        let (tx, _rx) = mpsc::channel(1);
        let writer = gate.acquire(&tx).await.unwrap().unwrap();
        let mut reader = Box::pin(gate.acquire_spec(&tx, gate.deadline()));
        assert!(futures_util::poll!(reader.as_mut()).is_pending());
        gate.retire();
        drop(writer);
        assert_eq!(reader.await.unwrap_err().to_string(), LAGUNA_WORKER_RETIRED);
        assert!(
            gate.active.try_write().is_ok(),
            "a rejected waited acquisition leaked its lock"
        );
        assert_eq!(
            gate.acquire(&tx).await.unwrap_err().to_string(),
            LAGUNA_WORKER_RETIRED
        );
        assert_eq!(
            gate.acquire_spec(&tx, gate.deadline())
                .await
                .unwrap_err()
                .to_string(),
            LAGUNA_WORKER_RETIRED
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn laguna_worker_unwind_retires_before_its_reservation_releases() {
        let (app, jobs, eager_jobs, gate) = mixed_fixture(500, 64);
        let first = tokio::spawn(request(app.clone(), "/v1/chat/completions", false, None));
        let mut job = take_job(&jobs).await;
        assert!(admit(&mut job));
        let waiting = tokio::spawn(mixed_request(
            app.clone(),
            "/v1/chat/completions",
            false,
            None,
            true,
        ));
        wait_for_mixed_waiter(&gate).await;
        let worker = std::thread::spawn(move || {
            let _job = job;
            panic!("controlled worker unwind after admission");
        });
        assert!(worker.join().is_err());
        assert!(gate.waiting.is_closed());
        busy(first.await.unwrap(), 503).await;
        busy(waiting.await.unwrap(), 503).await;
        assert!(
            eager_jobs.try_recv().is_err(),
            "eager waiter entered between unwind and retirement"
        );
        assert!(gate.active.try_write().is_ok());
        busy(request(app, "/v1/chat/completions", true, None).await, 503).await;
        assert!(jobs.try_recv().is_err());
    }

    #[tokio::test]
    async fn laguna_spec_reservations_preserve_batching_and_bound_mixed_waiters() {
        let gate = LagunaAdmission::new(1, Duration::from_millis(30));
        let (tx, _rx) = mpsc::channel(1);
        let first = gate
            .acquire_spec(&tx, gate.deadline())
            .await
            .unwrap()
            .unwrap();
        let second = gate
            .acquire_spec(&tx, gate.deadline())
            .await
            .unwrap()
            .unwrap();
        let mut writer = Box::pin(gate.acquire(&tx));
        assert!(futures_util::poll!(writer.as_mut()).is_pending());
        let error = gate.acquire_spec(&tx, gate.deadline()).await.unwrap_err();
        assert_eq!(
            error
                .downcast_ref::<crate::oapi::chat::EngineBusy>()
                .unwrap()
                .waited_ms,
            0,
            "readers must respect the waiting writer and bounded queue"
        );
        drop(first);
        assert!(futures_util::poll!(writer.as_mut()).is_pending());
        drop(second);
        let held = writer.await.unwrap().unwrap();
        assert!(gate.active.try_read().is_err());
        drop(held);
        drop(
            gate.acquire_spec(&tx, gate.deadline())
                .await
                .unwrap()
                .unwrap(),
        );
    }

    #[tokio::test]
    async fn laguna_spec_lock_and_worker_ack_share_one_deadline() {
        let gate = Arc::new(LagunaAdmission::new(1, Duration::from_millis(30)));
        let (tx, _rx) = mpsc::channel(1);
        let writer = gate.acquire(&tx).await.unwrap().unwrap();
        let deadline = gate.deadline();
        let mut reader = Box::pin(gate.acquire_spec(&tx, deadline));
        assert!(futures_util::poll!(reader.as_mut()).is_pending());
        tokio::time::sleep(Duration::from_millis(5)).await;
        drop(writer);
        let permit = reader.await.unwrap().unwrap();
        tokio::time::sleep_until(deadline).await;
        let (jobs, queued) = std::sync::mpsc::sync_channel(1);
        let spec = LagunaSpecServe {
            jobs: LagunaSpecWorker::from_test_sender(jobs),
            max_seq: 64,
            num_spec: 4,
            tail: LagunaWorkerTail::Reserved,
            capacity: 1,
            queue: gate.queue,
        };
        let tokenizer = Arc::new(tokenizers::Tokenizer::new(
            tokenizers::models::wordlevel::WordLevel::builder()
                .build()
                .unwrap(),
        ));
        let req = ChatGenerateRequest {
            prompt: "prompt".into(),
            max_new_tokens: 8,
            stop: vec![],
            seed: None,
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
        };
        let error = tokio::time::timeout(
            Duration::from_millis(15),
            run_laguna_spec_stream(
                &spec,
                &tokenizer,
                &req,
                &[0],
                8,
                &[2],
                &tx,
                permit,
                gate.clone(),
                deadline,
            ),
        )
        .await
        .expect("worker admission started a second queue budget")
        .unwrap_err();
        assert_eq!(
            error
                .downcast_ref::<crate::oapi::chat::EngineBusy>()
                .unwrap()
                .waited_ms,
            30
        );
        let mut expired = take_job(&queued).await;
        assert!(!admit(&mut expired));
        assert!(
            gate.active.try_write().is_err(),
            "expired queued worker must retain the reservation until cleanup"
        );
        drop(expired);
        assert!(gate.active.try_write().is_ok());
    }

    #[tokio::test]
    async fn laguna_eager_permit_survives_caller_exit_until_worker_release() {
        let gate = LagunaAdmission::new(1, Duration::from_millis(20));
        let (tx, rx) = mpsc::channel(1);
        let held = gate.acquire(&tx).await.unwrap().unwrap();
        let (next_tx, _next_rx) = mpsc::channel(1);
        let mut waiting = Box::pin(gate.acquire(&next_tx));
        assert!(futures_util::poll!(waiting.as_mut()).is_pending());
        let error = gate.acquire(&next_tx).await.unwrap_err();
        assert_eq!(
            error
                .downcast_ref::<crate::oapi::chat::EngineBusy>()
                .unwrap()
                .waited_ms,
            0
        );
        drop(rx);
        assert!(
            gate.active.try_write().is_err(),
            "caller exit must not release a still-running GPU worker's permit"
        );
        assert!(waiting
            .await
            .unwrap_err()
            .downcast_ref::<crate::oapi::chat::EngineBusy>()
            .is_some());
        drop(held);
        drop(gate.acquire(&next_tx).await.unwrap().unwrap());
        assert!(gate.acquire(&tx).await.unwrap().is_none());
    }
}

#[cfg(all(test, feature = "cuda"))]
mod text_tests {
    use super::LagunaText;

    fn bytelevel_tokenizer() -> tokenizers::Tokenizer {
        let json = serde_json::json!({
            "version": "1.0",
            "added_tokens": [],
            "model": {
                "type": "WordLevel",
                "vocab": {
                    "<unk>": 0, "h": 1, "ðŁ": 2, "ĺĢ": 3, "âĤ": 4, "¬": 5,
                    "EN": 6, "OUGHĠdone": 7, "D": 8, "sayĠE": 9, "N": 10, "DĠnow": 11,
                    "Hello": 12, "oneĠ": 13, "twoĠ": 14, "three": 15, "x": 16, "aby": 17
                },
                "unk_token": "<unk>"
            },
            "decoder": {"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}
        });
        tokenizers::Tokenizer::from_bytes(serde_json::to_vec(&json).unwrap()).unwrap()
    }

    fn stream(text: &mut LagunaText, tok: &tokenizers::Tokenizer, ids: &[u32]) -> (String, bool) {
        let mut out = String::new();
        for &id in ids {
            let (piece, hit) = text.push(tok, id).expect("push");
            out.push_str(&piece);
            if hit {
                return (out, true);
            }
        }
        out.push_str(&text.finish(tok).expect("finish"));
        (out, false)
    }

    #[test]
    fn laguna_streams_split_characters_whole_and_excludes_stops_found_mid_piece() {
        let tok = bytelevel_tokenizer();
        assert_eq!(tok.decode(&[1, 2], true).unwrap(), "h\u{FFFD}");
        for (ids, whole) in [([1u32, 2, 3], "h\u{1F600}"), ([1, 4, 5], "h\u{20AC}")] {
            let mut text = LagunaText::new(&[]);
            assert_eq!(
                stream(&mut text, &tok, &ids),
                (whole.to_string(), false),
                "a character split across tokens must reach the client whole"
            );
        }
        {
            let tok = bytelevel_tokenizer();
            let cases: [(&str, &[u32], &str, bool); 6] = [
                ("END", &[6, 7], "ENOUGH done", false),
                ("END", &[6, 8], "", true),
                ("END", &[9, 10, 11], "say ", true),
                ("Hell", &[12], "", true),
                ("ab", &[16, 17], "x", true),
                ("three", &[13, 14, 15], "one two ", true),
            ];
            for (stop, ids, want, hit) in cases {
                let mut text = LagunaText::new(&[stop.to_string()]);
                assert_eq!(
                    stream(&mut text, &tok, ids),
                    (want.to_string(), hit),
                    "stop {stop:?} over ids {ids:?}: Laguna must stream exactly what every other CUDA \
                     family streams, the text before the earliest stop, never the stop itself"
                );
            }
        }
    }

}
