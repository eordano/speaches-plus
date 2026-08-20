#[cfg(feature = "cuda")]
use super::*;
#[cfg(feature = "cuda")]
pub(crate) fn laguna_serve_spec_enabled() -> bool {
    std::env::var("NV_LAGUNA_SERVE_SPEC")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
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
#[allow(clippy::too_many_arguments)]
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

    if let Some(spec) = spec.as_ref().filter(|_| laguna_serve_spec_enabled()) {
        let params = sampling_params_from(&req);
        let eligible = params.is_greedy()
            && !params.has_penalties()
            && req.guided.is_none()
            && req.logit_bias.is_empty()
            && !req.logprobs
            && std::env::var_os("NV_LAGUNA_HOST_SAMPLE").is_none()
            && prompt_ids.len() + max_new + spec.num_spec + 2 <= spec.max_seq;
        if eligible
            && run_laguna_spec_stream(spec, &tokenizer, &req, &prompt_ids, max_new, eos_ids, tx)
                .await?
                .is_some()
        {
            return Ok(());
        }
    }

    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;

    let cache_len = kv_max_seq_len
        .max(prompt_ids.len() + max_new + 1)
        .min(model.config().max_position_embeddings);
    if prompt_ids.len() >= cache_len {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {cache_len}-token KV window",
            prompt_ids.len()
        );
    }
    let mut cache: Box<dyn nv_models::gemma4::Gemma4Cache + Send> =
        if std::env::var_os("NV_LAGUNA_FP8_KV").is_some() {
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

    const LAGUNA_PREFILL_CHUNK: usize = 256;
    let mut logits = None;
    let mut offset = 0usize;
    while offset < prompt_ids.len() {
        let n = LAGUNA_PREFILL_CHUNK.min(prompt_ids.len() - offset);
        let chunk_tokens = Tensor::from_vec(
            prompt_ids[offset..offset + n].to_vec(),
            (1usize, n),
            &device,
        )?;
        let positions: Vec<i32> = (offset as i32..(offset + n) as i32).collect();
        let positions_t = Tensor::from_vec(positions, n, &device)?;
        logits = Some(model.forward_with_cache(&chunk_tokens, &positions_t, &mut cache)?);
        offset += n;
    }
    let logits = logits.ok_or_else(|| anyhow::anyhow!("empty prompt after tokenize"))?;
    let last_row = last_row_logits_3d(&logits)?;
    drop(logits);
    let mut last_out = sampler.sample(&last_row);
    let mut last = last_out.token;
    if last_out.exhausted {
        anyhow::bail!("no legal token at prefill: the sampling mask left every candidate at -inf");
    }

    let host_sample = std::env::var_os("NV_LAGUNA_HOST_SAMPLE").is_some();
    let cuda_dev = match &device {
        candle_core::Device::Cuda(d) => Some(d.clone()),
        _ => None,
    };
    let mut argmax_scratch = match &cuda_dev {
        Some(d) if sampler.fast_greedy() && !host_sample => Some(LagunaArgmaxScratch::new(d)?),
        _ => None,
    };
    let mut inc_decoder = (!host_sample).then(nv_tokenizer::IncrementalDecoder::new);

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    let mut emitted_text = String::new();
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();
    let stop_strings = req.stop.clone();

    for step in 0..max_new {
        if eos_ids.contains(&last) {
            finish_reason = "stop".into();
            break;
        }
        generated_ids.push(last);
        completion_tokens = generated_ids.len() as u32;

        let piece = match inc_decoder.as_mut() {
            Some(dec) => {
                let p = dec.push(&tokenizer, last)?.unwrap_or_default();
                emitted_text.push_str(&p);
                p
            }
            None => {
                let new_text = tokenizer
                    .decode(&generated_ids, true)
                    .map_err(|e| anyhow::anyhow!("detokenize: {e}"))?;
                let p = if new_text.len() > emitted_text.len() {
                    new_text[emitted_text.len()..].to_string()
                } else {
                    String::new()
                };
                emitted_text = new_text;
                p
            }
        };

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

        if stop_strings
            .iter()
            .any(|s| !s.is_empty() && emitted_text.ends_with(s))
        {
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

    if let Some(dec) = inc_decoder.as_mut() {
        if let Some(p) = dec.flush(&tokenizer)? {
            if !p.is_empty() {
                emitted_text.push_str(&p);
                if tx.send(ChatEvent::TextDelta(p)).await.is_err() {
                    return Ok(());
                }
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
pub(crate) async fn run_laguna_spec_stream(
    spec: &LagunaSpecServe,
    tokenizer: &Arc<tokenizers::Tokenizer>,
    req: &ChatGenerateRequest,
    prompt_ids: &[u32],
    max_new: usize,
    eos_ids: &[u32],
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<Option<()>> {
    use nv_models::laguna_serve::{SpecServeEvent, SpecServeJob};

    let (ev_tx, mut ev_rx) = tokio::sync::mpsc::unbounded_channel::<SpecServeEvent>();
    let job = SpecServeJob {
        prompt_ids: prompt_ids.to_vec(),
        prompt_text: req.prompt.clone(),
        max_new,
        eos_ids: eos_ids.to_vec(),
        emit: Box::new(move |ev| ev_tx.send(ev).is_ok()),
    };
    {
        let sender = spec.jobs.lock().expect("laguna spec job sender poisoned");
        if sender.send(job).is_err() {
            return Ok(None);
        }
    }

    let mut inc_decoder = nv_tokenizer::IncrementalDecoder::new();
    let mut emitted_text = String::new();
    let mut completion_tokens: u32 = 0;
    let mut finish_reason: Option<String> = None;
    let mut got_any = false;
    let stop_strings = &req.stop;

    'outer: while let Some(ev) = ev_rx.recv().await {
        match ev {
            SpecServeEvent::Tokens(toks) => {
                got_any = true;
                for t in toks {
                    if eos_ids.contains(&t) {
                        finish_reason = Some("stop".into());
                        break 'outer;
                    }
                    completion_tokens += 1;
                    let piece = inc_decoder.push(tokenizer, t)?.unwrap_or_default();
                    if !piece.is_empty() {
                        emitted_text.push_str(&piece);
                        if tx.send(ChatEvent::TextDelta(piece)).await.is_err() {
                            return Ok(Some(()));
                        }
                    }
                    if stop_strings
                        .iter()
                        .any(|s| !s.is_empty() && emitted_text.ends_with(s))
                    {
                        finish_reason = Some("stop".into());
                        break 'outer;
                    }
                    if completion_tokens as usize >= max_new {
                        finish_reason = Some("length".into());
                        break 'outer;
                    }
                }
            }
            SpecServeEvent::Done => break 'outer,
            SpecServeEvent::Error(e) => {
                if !got_any {
                    warn!("laguna spec serving failed before first token, per-request path: {e}");
                    return Ok(None);
                }
                let _ = tx.send(ChatEvent::Error(e)).await;
                return Ok(Some(()));
            }
        }
    }
    if !got_any && finish_reason.is_none() {
        warn!("laguna spec serving stream closed before first token, per-request path");
        return Ok(None);
    }

    if let Some(p) = inc_decoder.flush(tokenizer)? {
        if !p.is_empty() {
            emitted_text.push_str(&p);
            if tx.send(ChatEvent::TextDelta(p)).await.is_err() {
                return Ok(Some(()));
            }
        }
    }
    let _ = tx
        .send(ChatEvent::Done {
            finish_reason: finish_reason.unwrap_or_else(|| "length".into()),
            completion_tokens,
        })
        .await;
    Ok(Some(()))
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
