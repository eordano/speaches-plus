#[cfg(feature = "cuda")]
use super::*;

#[cfg(feature = "cuda")]
pub(crate) const BOS_IS_A_POSITION_0_ROLE_AND_THE_SAME_ID_LATER_IS_A_STOP: &str =
    "BOS is a role that only position 0 of a prompt can hold. Checkpoints declare one id in both \
     roles: poolside/Laguna-XS-2.1-NVFP4 declares id 2 as bos_token_id and as a member of \
     eos_token_id [2, 24], and a cached Qwen checkpoint carries bos_token_id 248044 inside its \
     own eos set. Such an id is a beginning-of-sequence marker at position 0 and a stop at every \
     later position, so the engine may leave it at position 0 only when the checkpoint's own \
     rendered chat template put it there. Splicing a second copy in front demotes the template's \
     copy to position 1, where that id means stop; adopting one the template never emitted opens \
     the prompt on a stop id the model never saw in training. Refusing the splice never removes \
     the id from the EOS set -- that membership is the checkpoint's own declaration and stopping \
     depends on it. Both sides of that membership test must be the checkpoint's own words, which \
     is why bos is Option<u32>: a checkpoint that declares no bos_token_id in \
     generation_config.json, config.json or their text_config gets none spliced. A fabricated \
     pair -- the `parse_bos_id(config).unwrap_or(2)` weighed against \
     `parse_eos_ids(config).unwrap_or([1, 106])` this engine shipped -- can never collide, so \
     the rule answered `splice it` on every checkpoint and reported nothing.";

#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PromptHead {
    TemplateEmittedBosAtPosition0,
    EnginePrependedBosAtPosition0,
    RefusedToSpliceAnIdTheCheckpointAlsoDeclaresEos,
    CheckpointDeclaresNoBos,
}

#[cfg(feature = "cuda")]
pub(crate) fn splice_bos_at_position_0_only(
    prompt_ids: &mut Vec<u32>,
    bos_token_id: Option<u32>,
    eos_ids: &[u32],
) -> PromptHead {
    let Some(bos_token_id) = bos_token_id else {
        return PromptHead::CheckpointDeclaresNoBos;
    };
    if prompt_ids.first().copied() == Some(bos_token_id) {
        return PromptHead::TemplateEmittedBosAtPosition0;
    }
    if eos_ids.contains(&bos_token_id) {
        tracing::warn!(
            bos_token_id,
            eos_ids = ?eos_ids,
            head = ?prompt_ids.first().copied(),
            rule = BOS_IS_A_POSITION_0_ROLE_AND_THE_SAME_ID_LATER_IS_A_STOP,
            "declared bos_token_id is also an EOS member and the rendered prompt does not open on it; not splicing it"
        );
        return PromptHead::RefusedToSpliceAnIdTheCheckpointAlsoDeclaresEos;
    }
    prompt_ids.insert(0, bos_token_id);
    PromptHead::EnginePrependedBosAtPosition0
}

#[cfg(feature = "cuda")]
pub(crate) async fn run_gemma4_via_engine(
    handle: Arc<nv_engine::BatchEngineHandle>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    req: ChatGenerateRequest,
    default_max_new: usize,
    eos_ids: &[u32],
    bos_token_id: Option<u32>,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;
    let mut prompt_ids: Vec<u32> = encoded.get_ids().to_vec();
    splice_bos_at_position_0_only(&mut prompt_ids, bos_token_id, eos_ids);
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

    let sampling = nv_engine::SamplingConfig {
        temperature: req.temperature.unwrap_or(0.0).max(0.0),
        top_p: req.top_p,
        top_k: req.top_k,
        min_p: req.min_p,
        seed: Some(req.seed.unwrap_or_else(os_random_u64)),
        presence_penalty: req.presence_penalty,
        frequency_penalty: req.frequency_penalty,
        repetition_penalty: req.repetition_penalty,
    };

    let _admit_guard = crate::oapi::admission::admit_or_bail_measured(0, 0, "gemma4-batch").await?;

    let mut rx = handle.submit(prompt_ids, max_new, eos_ids.to_vec(), sampling);

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    let mut emitter = StreamEmitter::new(&req.stop);
    let mut detok = IncrementalDetok::new(tokenizer.clone());
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();
    let mut seq_id: Option<u64> = None;
    let mut stopped_early = false;

    while let Some(ev) = rx.recv().await {
        match ev {
            nv_engine::BatchEvent::Started { seq_id: sid, .. } => {
                seq_id = Some(sid);
            }
            nv_engine::BatchEvent::Token { token, .. } => {
                if stopped_early {
                    continue;
                }
                generated_ids.push(token);
                completion_tokens = generated_ids.len() as u32;
                let new_text = detok.push(token)?;
                let (piece, stop_hit) = emitter.step(new_text);
                if !piece.is_empty() && tx.send(ChatEvent::TextDelta(piece)).await.is_err() {
                    if let Some(sid) = seq_id {
                        handle.abort(sid);
                    }
                    return Ok(());
                }
                if stop_hit {
                    finish_reason = "stop".into();
                    stopped_early = true;
                    if let Some(sid) = seq_id {
                        handle.abort(sid);
                    }
                }
            }
            nv_engine::BatchEvent::Done { reason, .. } => {
                if !stopped_early {
                    finish_reason = match reason {
                        nv_engine::FinishReason::Eos => "stop".into(),
                        nv_engine::FinishReason::MaxTokens => "length".into(),
                        nv_engine::FinishReason::Aborted => finish_reason.clone(),
                    };
                }
                break;
            }
            nv_engine::BatchEvent::Error { message, .. } => {
                anyhow::bail!("batch engine: {message}");
            }
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
pub(crate) async fn run_sampling_gemma4(
    model: Arc<nv_models::gemma4::Gemma4>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    req: ChatGenerateRequest,
    kv_max_seq_len: usize,
    default_max_new: usize,
    eos_ids: &[u32],
    bos_token_id: Option<u32>,
    eagle3: Option<Arc<Eagle3Shared>>,
    dflash: Option<Arc<DFlashShared>>,
    mm_towers: Option<Arc<crate::oapi::chat_multimodal::Gemma4MmTowers>>,
    snap: &SpecEnvSnapshot,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    use candle_core::Tensor;

    let encoded = tokenizer
        .encode(req.prompt.as_str(), false)
        .map_err(|e| anyhow::anyhow!("tokenize: {e}"))?;

    let mut raw_ids: Vec<u32> = encoded.get_ids().to_vec();
    splice_bos_at_position_0_only(&mut raw_ids, bos_token_id, eos_ids);
    let mut prompt_ids = raw_ids;

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
    let prompt_ids = prompt_ids;
    let prompt_tokens = prompt_ids.len() as u32;

    let max_new = effective_max_new(req.max_new_tokens, default_max_new);
    let Some((kv_needed, max_new)) = kv_window(prompt_ids.len(), max_new, kv_max_seq_len) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {kv_max_seq_len}-token KV window",
            prompt_ids.len()
        );
    };

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

    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(&prompt_ids);

    if env_flag_enabled(snap.prof_chat.as_deref()) {
        eprintln!("[NV_PROF_CHAT][env] {}", snap.profile_line());
    }

    if spec_gate_for_request(
        nv_no_spec(snap.no_spec.as_deref()),
        env_flag_enabled(snap.use_eagle3.as_deref()),
        sampler.params.is_greedy(),
    ) && sampler.guided.is_none()
        && sampler.logit_bias.is_empty()
        && !req.logprobs
        && mm_embeds.is_none()
        && prompt_ids.len() < spec_ctx_disable(snap.spec_ctx_disable.as_deref())
    {
        let drafter_kind = nv_drafter_kind(snap.drafter.as_deref());
        let class = classify_prompt(&req.prompt);
        let ctx_gate = if drafter_kind == "auto" {
            drafter_auto_switch_tokens(snap.drafter_auto_switch_tokens.as_deref())
        } else {
            route_ctx_gate(snap.route_ctx_gate.as_deref())
        };
        let arm = resolve_drafter_arm(
            drafter_kind,
            class,
            prompt_ids.len(),
            ctx_gate,
            dflash.is_some(),
            eagle3.is_some(),
            arm_ema_get(DrafterArm::DFlash),
            arm_ema_get(DrafterArm::Eagle3),
        );
        if let Some(routed) = arm {
            record_last_routed_drafter_arm(routed);
        }
        if dflash.is_some() && eagle3.is_some() {
            tracing::info!(
                ?arm,
                drafter = drafter_kind,
                ?class,
                prompt_tokens = prompt_ids.len(),
                ctx_gate,
                dflash_ema = arm_ema_get(DrafterArm::DFlash),
                eagle3_ema = arm_ema_get(DrafterArm::Eagle3),
                "drafter routing: routed request"
            );
            if env_flag_enabled(snap.prof_chat.as_deref()) {
                eprintln!(
                    "[NV_PROF_CHAT][route] drafter={} arm={} prompt_tokens={} gate_tokens={}",
                    drafter_kind,
                    arm.map(drafter_arm_name).unwrap_or("none"),
                    prompt_ids.len(),
                    ctx_gate
                );
            }
        }
        if arm == Some(DrafterArm::DFlash) {
            let state = dflash.clone().expect("dflash arm routed while loaded");
            return run_sampling_gemma4_spec_dflash(
                model,
                tokenizer,
                device,
                prompt_ids,
                max_new,
                kv_max_seq_len,
                eos_ids,
                req.stop.clone(),
                state,
                class,
                sampler,
                snap,
                tx,
            )
            .await;
        }
        if let Some(state) = eagle3.clone() {
            return run_sampling_gemma4_spec(
                model,
                tokenizer,
                device,
                prompt_ids,
                max_new,
                kv_max_seq_len,
                eos_ids,
                req.stop.clone(),
                state,
                sampler,
                snap,
                tx,
            )
            .await;
        }
    }

    let _admit_guard = {
        let b = nv_models::gemma4::kv_budget(
            model.config(),
            kv_needed,
            nv_models::gemma4::verify_kv_use_fp8(),
            nv_models::gemma4::kv_ring_enabled(),
            0,
        );
        crate::oapi::admission::admit_or_bail(0, b.decode_total() as u64, "gemma4-nonspec").await?
    };

    let mut permit = Some(acquire_chat_permit().await?);

    let prof = env_flag_enabled(snap.prof_chat.as_deref());
    let t_start = std::time::Instant::now();

    let t_lock = t_start.elapsed();

    if let Err(e) = device.synchronize() {
        warn!(error = %e, "pre-generation device sync surfaced a stale error (cleared); continuing");
        let _ = device.synchronize();
    }
    let mut cache = model
        .new_kv_cache_fp8_windowed(kv_needed)
        .map_err(|e| anyhow::anyhow!("alloc fp8 kv cache: {e}"))?;
    let t_cache = t_start.elapsed() - t_lock;

    let prefill_chunk = nv_models::gemma4::VERIFY_PREFILL_CHUNK;
    let t_pre0 = std::time::Instant::now();
    let det_debug = snap.debug_determinism;
    let mut logits_opt: Option<Tensor> = None;
    let mut off = 0usize;
    while off < prompt_ids.len() {
        let c = (prompt_ids.len() - off).min(prefill_chunk);
        let prefill_tokens =
            Tensor::from_vec(prompt_ids[off..off + c].to_vec(), (1usize, c), &device)?;
        let positions: Vec<i32> = (off as i32..(off + c) as i32).collect();
        let positions_t = Tensor::from_vec(positions, c, &device)?;
        logits_opt = Some(match &mm_embeds {
            Some(e) => model
                .forward_with_cache_last_embeds(
                    &prefill_tokens,
                    &e.narrow(0, off, c)?,
                    &positions_t,
                    &mut cache,
                )
                .map_err(|e| anyhow::anyhow!("prefill forward_with_cache_last_embeds: {e}"))?,
            None if det_debug => {
                let mut hook = DetHashHook;
                model
                    .forward_with_cache_hooked(&prefill_tokens, &positions_t, &mut cache, &mut hook)
                    .map_err(|e| anyhow::anyhow!("prefill forward_with_cache_hooked: {e}"))?
            }
            None => model
                .forward_with_cache_last(&prefill_tokens, &positions_t, &mut cache)
                .map_err(|e| anyhow::anyhow!("prefill forward_with_cache: {e}"))?,
        });
        off += c;
    }
    let logits = logits_opt.ok_or_else(|| anyhow::anyhow!("empty prompt: no prefill logits"))?;
    let last_row = last_row_logits_3d(&logits)?;
    if det_debug {
        eprintln!(
            "[NV_DEBUG_DETERMINISM] prefill-last-row n={} hash={:016x}",
            last_row.len(),
            det_hash_f32(&last_row)
        );
    }
    let mut last_out = sampler.sample(&last_row);
    let mut last = last_out.token;
    if last_out.exhausted {
        anyhow::bail!("no legal token at prefill: the sampling mask left every candidate at -inf");
    }
    drop(logits);
    let t_prefill = t_pre0.elapsed();

    if prof {
        eprintln!(
            "[NV_PROF_CHAT] lock={:.2}ms cache_alloc={:.2}ms prefill={:.2}ms prompt_tokens={}",
            t_lock.as_secs_f64() * 1000.0,
            t_cache.as_secs_f64() * 1000.0,
            t_prefill.as_secs_f64() * 1000.0,
            prompt_ids.len()
        );
    }

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    let mut emitter = StreamEmitter::new(&req.stop);
    let mut detok = IncrementalDetok::new(tokenizer.clone());
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();

    let total_decode_ms;
    let mut total_detok_ms = 0.0f64;
    let mut total_send_ms = 0.0f64;
    let mut total_fwd_ms = 0.0f64;
    let total_sample_ms = 0.0f64;
    let decode_loop_start = std::time::Instant::now();

    let use_graph = sampler.params.is_greedy()
        && !sampler.params.has_penalties()
        && sampler.guided.is_none()
        && sampler.logit_bias.is_empty()
        && !req.logprobs;

    let send_timeout = sse_send_timeout();
    let mut aborted = false;

    if use_graph {
        tokio::task::block_in_place(|| -> anyhow::Result<()> {
            let mut graphed =
                nv_models::gemma4_graph::GraphedGemma4Decoder::new(&*model, cache, &device)?;
            for step in 0..max_new {
                if tx.is_closed() {
                    log_sse_abort(SsePush::Closed, "gemma4-nonspec");
                    aborted = true;
                    return Ok(());
                }
                if eos_ids.contains(&last) {
                    finish_reason = "stop".into();
                    break;
                }
                generated_ids.push(last);
                completion_tokens = generated_ids.len() as u32;

                let t0 = std::time::Instant::now();
                let new_text = detok.push(last)?;
                total_detok_ms += t0.elapsed().as_secs_f64() * 1000.0;
                let (piece, stop_hit) = emitter.step(new_text);

                if !piece.is_empty() {
                    let t1 = std::time::Instant::now();
                    let outcome =
                        push_event_blocking(tx, ChatEvent::TextDelta(piece), send_timeout);
                    if outcome != SsePush::Sent {
                        log_sse_abort(outcome, "gemma4-nonspec");
                        aborted = true;
                        return Ok(());
                    }
                    total_send_ms += t1.elapsed().as_secs_f64() * 1000.0;
                }

                if stop_hit {
                    finish_reason = "stop".into();
                    break;
                }
                if step + 1 >= max_new {
                    break;
                }

                let t2 = std::time::Instant::now();
                last = graphed.forward_decode_logits(last)?;
                total_fwd_ms += t2.elapsed().as_secs_f64() * 1000.0;
            }
            Ok(())
        })?;
    } else {
        tokio::task::block_in_place(|| -> anyhow::Result<()> {
            use anyhow::Context as _;
            if snap.debug_graph {
                eprintln!(
                    "[guided] prompt_len={} kv_max_seq_len={} max_new={} guided={}",
                    prompt_ids.len(),
                    kv_max_seq_len,
                    max_new,
                    req.guided.is_some()
                );
            }
            let mut graphed =
                nv_models::gemma4_graph::GraphedGemma4Decoder::new(&*model, cache, &device)
                    .context("construct GraphedGemma4Decoder (guided path)")?;
            for step in 0..max_new {
                if tx.is_closed() {
                    log_sse_abort(SsePush::Closed, "gemma4-guided");
                    aborted = true;
                    return Ok(());
                }
                if eos_ids.contains(&last) {
                    finish_reason = "stop".into();
                    break;
                }
                generated_ids.push(last);
                completion_tokens = generated_ids.len() as u32;

                let new_text = detok.push(last)?;
                let (piece, stop_hit) = emitter.step(new_text);

                if !piece.is_empty() {
                    let outcome =
                        push_event_blocking(tx, ChatEvent::TextDelta(piece), send_timeout);
                    if outcome != SsePush::Sent {
                        log_sse_abort(outcome, "gemma4-guided");
                        aborted = true;
                        return Ok(());
                    }
                }
                if req.logprobs {
                    let outcome = push_event_blocking(
                        tx,
                        ChatEvent::Logprob(build_logprob_entry(&tokenizer, &last_out)),
                        send_timeout,
                    );
                    if outcome != SsePush::Sent {
                        log_sse_abort(outcome, "gemma4-guided");
                        aborted = true;
                        return Ok(());
                    }
                }

                if stop_hit {
                    finish_reason = "stop".into();
                    break;
                }
                if step + 1 >= max_new {
                    break;
                }

                let logits = graphed
                    .forward_decode_logits_into(last)
                    .with_context(|| format!("guided graph decode step={step}"))?;
                last_out = sampler.sample(logits);
                last = last_out.token;
                if last_out.exhausted {
                    anyhow::bail!("no legal token at step {step}: the sampling mask left every candidate at -inf");
                }
            }
            Ok(())
        })?;
    }
    total_decode_ms = decode_loop_start.elapsed().as_secs_f64() * 1000.0;

    drop(permit.take());

    if !aborted && !generated_ids.is_empty() {
        if let Ok(full) = crate::oapi::chat_engine::stream::decode_keeping_wire(&tokenizer, &generated_ids) {
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                let _ = push_event_async(tx, ChatEvent::TextDelta(tail), send_timeout).await;
            }
        }
    }

    if prof {
        let n = completion_tokens.max(1) as f64;
        eprintln!(
            "[NV_PROF_CHAT] decode_total={:.0}ms ({} tok = {:.1} tok/s) | per-tok: fwd={:.2}ms detok={:.2}ms send={:.2}ms sample={:.2}ms other={:.2}ms",
            total_decode_ms,
            completion_tokens,
            (n / total_decode_ms) * 1000.0,
            total_fwd_ms / n,
            total_detok_ms / n,
            total_send_ms / n,
            total_sample_ms / n,
            (total_decode_ms - total_fwd_ms - total_detok_ms - total_send_ms - total_sample_ms) / n,
        );
    }

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

#[cfg(feature = "cuda")]
pub(crate) async fn run_sampling_gemma4_spec(
    model: Arc<nv_models::gemma4::Gemma4>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    prompt_ids: Vec<u32>,
    max_new: usize,
    kv_max_seq_len: usize,
    eos_ids: &[u32],
    stop_strings: Vec<String>,
    eagle3: Arc<Eagle3Shared>,
    sampler: ChatSampler,
    snap: &SpecEnvSnapshot,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    let force_ungraphed = env_flag_enabled(snap.eagle3_ungraphed.as_deref());
    let use_tree = snap.eagle3_tree;

    if (force_ungraphed || use_tree) && sampler.params.is_greedy() {
        return run_sampling_gemma4_spec_ungraphed(
            model,
            tokenizer,
            device,
            prompt_ids,
            max_new,
            eos_ids,
            stop_strings,
            eagle3,
            sampler,
            snap,
            tx,
        )
        .await;
    }
    run_sampling_gemma4_spec_graphed(
        model,
        tokenizer,
        device,
        prompt_ids,
        max_new,
        kv_max_seq_len,
        eos_ids,
        stop_strings,
        eagle3,
        sampler,
        snap,
        tx,
    )
    .await
}

#[cfg(feature = "cuda")]
pub(crate) async fn run_sampling_gemma4_spec_graphed(
    model: Arc<nv_models::gemma4::Gemma4>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    prompt_ids: Vec<u32>,
    max_new: usize,
    kv_max_seq_len: usize,
    eos_ids: &[u32],
    stop_strings: Vec<String>,
    eagle3: Arc<Eagle3Shared>,
    mut sampler: ChatSampler,
    snap: &SpecEnvSnapshot,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    use nv_specdecode::chain::{
        accept_prefix_argmax, aux_row_extract, build_chain_batch, chain_positions, lower_tri_mask,
        ChainJudgment, ChainVerifier,
    };

    let prof = env_flag_enabled(snap.prof_chat.as_deref());

    let model_arc = model;
    let model = &*model_arc;

    let k_env = eagle3_k(snap.eagle3_k.as_deref(), prompt_ids.len());
    let adaptive = adaptive_k_enabled(snap.adaptive_k.as_deref()) && {
        let dk = !snap.eagle3_no_drafter_kv;
        resolve_cond_mode(snap.eagle3_cond.as_deref(), dk).0 == "shift"
    };
    let k = if adaptive {
        adaptive_k_graph(snap.adaptive_k_max.as_deref(), k_env)
    } else {
        k_env
    };
    let Some((max_seq, max_new)) = spec_verify_window(prompt_ids.len(), max_new, k, kv_max_seq_len)
    else {
        anyhow::bail!(
            "spec verify cache for {} prompt + {max_new} new + k={k} does not fit the {kv_max_seq_len}-token KV window",
            prompt_ids.len()
        );
    };
    debug_assert!(max_seq <= kv_max_seq_len);

    let graph_cache_enabled = !snap.eagle3_no_graph_cache;
    let (sticky, extra) = {
        let cache_cap = verify_cache_capacity(max_seq);
        let b = nv_models::gemma4::kv_budget_capped(
            model.config(),
            cache_cap,
            nv_models::gemma4::verify_kv_use_fp8(),
            nv_models::gemma4::kv_ring_enabled(),
            eagle3.proposer.scorer().config().kv_out_dim(),
            capped_drafter_kv_rows(cache_cap, drafter_kv_cap_env()),
        );
        let hd512_scratch = nv_models::gemma4::gqa512_verify_scratch_bytes(model.config()) as u64;
        (
            b.verify_total() as u64 + hd512_scratch,
            b.drafter_kv_bytes as u64,
        )
    };

    let shared = eagle3;
    let (mut taken_gv, had_lease, mut taken_chain) = {
        let mut pool = shared.pool.lock().await;
        let gv = pool.verify.take();
        let had = gv.is_some();
        if had {
            pool.lease_out = true;
        }
        (gv, had, pool.chain.take())
    };

    let admit_res = if graph_cache_enabled && had_lease {
        crate::oapi::admission::admit_or_bail(sticky, extra, "gemma4-spec").await
    } else {
        crate::oapi::admission::admit_or_bail(0, sticky + extra, "gemma4-spec").await
    };
    let permit_res = match &admit_res {
        Ok(_) => Some(acquire_chat_permit().await),
        Err(_) => None,
    };
    let (mut admit_guard, permit) = match (admit_res, permit_res) {
        (Ok(g), Some(Ok(p))) => (g, p),
        (admit_res, permit_res) => {
            let mut pool = shared.pool.lock().await;
            if had_lease {
                pool.lease_out = false;
            }
            if pool.verify.is_none() {
                pool.verify = taken_gv.take();
            }
            if pool.chain.is_none() {
                pool.chain = taken_chain.take();
            }
            drop(pool);
            admit_res?;
            match permit_res {
                Some(Err(e)) => return Err(e),
                _ => anyhow::bail!("spec admission/permit acquisition failed"),
            }
        }
    };

    if let Err(e) = device.synchronize() {
        warn!(error = %e, "pre-spec device sync surfaced a stale error (cleared); continuing");
        let _ = device.synchronize();
    }
    let aux_layers = shared.aux_layers.clone();
    let n_layers_aux = aux_layers.len();
    let fc_in_dim = shared.proposer.scorer().config().fc_in_dim();
    let vocab = model.config().vocab_size;
    let hidden = model.config().hidden_size;

    let dev = match &device {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("graphed spec requires a CUDA device"),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);

    anyhow::ensure!(
        fc_in_dim == n_layers_aux * hidden,
        "eagle3 fc_in {fc_in_dim} != aux layers {n_layers_aux} * hidden {hidden}"
    );

    let mut context: Vec<u32> = prompt_ids.clone();
    let initial_len = context.len();

    let core = (|| -> anyhow::Result<(
        nv_models::gemma4_graph::GraphedGemma4Verify<Arc<nv_models::gemma4::Gemma4>>,
        String,
        u32,
        bool,
    )> {

    let mut gv = take_reusable_or_build(
        &mut taken_gv,
        |gv| {
            verify_graph_reusable(gv.k(), gv.cache_capacity(), k, max_seq)
                && Arc::ptr_eq(gv.model(), &model_arc)
        },
        || {
            if let Err(e) = device.synchronize() {
                warn!(error = %e, "verifier teardown surfaced a stale error (cleared); continuing");
                let _ = device.synchronize();
            }

            let cache = model
                .new_verify_cache(verify_cache_capacity(max_seq))
                .map_err(|e| anyhow::anyhow!("alloc verify cache: {e}"))?;
            nv_models::gemma4_graph::GraphedGemma4Verify::new(
                model_arc.clone(),
                cache,
                &device,
                k,
                aux_layers.clone(),
            )
            .map_err(|e| anyhow::anyhow!("construct GraphedGemma4Verify: {e}"))
        },
    )?;

    let prefill_chunk = spec_prefill_chunk(snap.spec_prefill_chunk.as_deref())
        .min(nv_models::gemma4::VERIFY_PREFILL_CHUNK);

    let use_drafter_kv = !snap.eagle3_no_drafter_kv;
    let mut drafter_kv = nv_specdecode::eagle3_loader::DrafterKvCache::new();

    let (cond_mode, shift_forced, downgraded) = resolve_cond_mode(
        snap.eagle3_cond.as_deref(),
        use_drafter_kv,
    );
    if downgraded {
        tracing::warn!(
            "NV_EAGLE3_NO_DRAFTER_KV set: cached conditioning modes unavailable, \
             using legacy pairing (NV_EAGLE3_COND=shift-force runs shift via a \
             per-round throwaway drafter KV for A/B diagnostics)"
        );
    }

    let stream_aux = use_drafter_kv;
    let shift_prefill = cond_mode == "shift";
    use nv_specdecode::eagle3_loader::DRAFTER_ENCODE_CHUNK;
    let preencode_limit =
        (initial_len.saturating_sub(1) / DRAFTER_ENCODE_CHUNK) * DRAFTER_ENCODE_CHUNK;
    let defer_drafter = stream_aux
        && spec_defer_drafter_from(snap.spec_defer_drafter.as_deref());

    let prof_prefill = snap.prof_prefill;
    let prefill_start = std::time::Instant::now();
    let mut prefill_verify_ms = 0.0f64;
    let mut prefill_drafter_ms = 0.0f64;

    let mut aux_proj: Option<candle_core::Tensor> = None;
    let mut aux_base = 0usize;
    let mut committed = 0usize;
    let mut seed: Vec<f32> = Vec::new();
    let mut aux_pending: Vec<f32> = if stream_aux {
        Vec::new()
    } else {
        Vec::with_capacity(initial_len.saturating_mul(fc_in_dim))
    };
    while committed < initial_len {
        let t_chunk = std::time::Instant::now();
        let len = prefill_chunk.min(initial_len - committed);
        let last = committed + len == initial_len;
        let pmask_d = stream
            .clone_htod(&lower_tri_mask(len))
            .map_err(|e| anyhow::anyhow!("prefill mask htod: {e:?}"))?;
        let ppos: Vec<i32> = (committed as i32..(committed + len) as i32).collect();
        let (chunk_logits, chunk_aux) = model
            .forward_verify_tail(
                &context[committed..committed + len],
                &ppos,
                &pmask_d,
                committed,
                &aux_layers,
                gv.cache_mut(),
                if last { 1 } else { 0 },
            )
            .map_err(|e| anyhow::anyhow!("spec prefill forward_verify (chunk at {committed}): {e}"))?;
        if let Some(lg) = chunk_logits {
            seed = lg
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
        }
        let chunk_aux_cat =
            candle_core::Tensor::cat(&chunk_aux.iter().collect::<Vec<_>>()[..], 1)?;
        let t_verify_done = if prof_prefill {
            let _ = device.synchronize();
            prefill_verify_ms += t_chunk.elapsed().as_secs_f64() * 1000.0;
            Some(std::time::Instant::now())
        } else {
            None
        };
        if stream_aux {
            let projected = shared
                .proposer
                .scorer()
                .project_aux(&chunk_aux_cat)
                .map_err(|e| anyhow::anyhow!("project_aux (prefill chunk at {committed}): {e}"))?;
            aux_proj = Some(match aux_proj.take() {
                Some(prev) => candle_core::Tensor::cat(&[&prev, &projected], 0)?,
                None => projected,
            });
            committed += len;
            let target = if defer_drafter {
                0
            } else {
                ((committed / DRAFTER_ENCODE_CHUNK) * DRAFTER_ENCODE_CHUNK).min(preencode_limit)
            };
            if target > drafter_kv.len() {
                let aux_t = aux_proj.take().expect("aux_proj set above");
                shared
                    .proposer
                    .scorer()
                    .preencode_context(
                        &mut drafter_kv,
                        &context,
                        &aux_t,
                        aux_base,
                        target,
                        shift_prefill,
                    )
                    .map_err(|e| anyhow::anyhow!("drafter preencode to {target}: {e}"))?;
                let drop = target - aux_base;
                let rows = aux_t.dims()[0];
                aux_proj = Some(aux_t.narrow(0, drop, rows - drop)?);
                aux_base = target;
            }
            if let Some(t) = t_verify_done {
                let _ = device.synchronize();
                let d = t.elapsed().as_secs_f64() * 1000.0;
                prefill_drafter_ms += d;
                eprintln!(
                    "[NV_PROF_PREFILL] chunk at {} len {len}: verify+seed {:.1} ms drafter {:.1} ms",
                    committed - len,
                    (t - t_chunk).as_secs_f64() * 1000.0,
                    d
                );
            }
        } else {
            let chunk_aux_host: Vec<f32> = chunk_aux_cat
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
            aux_pending.extend_from_slice(&chunk_aux_host);
            committed += len;
            if let Some(t) = t_verify_done {
                let d = t.elapsed().as_secs_f64() * 1000.0;
                prefill_drafter_ms += d;
                eprintln!(
                    "[NV_PROF_PREFILL] chunk at {} len {len}: verify+seed {:.1} ms aux-d2h {:.1} ms",
                    committed - len,
                    (t - t_chunk).as_secs_f64() * 1000.0,
                    d
                );
            }
        }
    }
    if prof_prefill {
        let total_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[NV_PROF_PREFILL] SUMMARY prompt={initial_len} chunks={} total {:.1} ms ({:.1} tok/s) verify {:.1} ms drafter {:.1} ms other {:.1} ms",
            initial_len.div_ceil(prefill_chunk.max(1)),
            total_ms,
            initial_len as f64 / (total_ms / 1000.0),
            prefill_verify_ms,
            prefill_drafter_ms,
            total_ms - prefill_verify_ms - prefill_drafter_ms
        );
    }
    anyhow::ensure!(
        seed.len() == vocab,
        "spec prefill seed row: expected {vocab} logits, got {}",
        seed.len()
    );
    if stream_aux {
        let tail_rows = aux_proj.as_ref().map(|t| t.dims()[0]).unwrap_or(0);
        anyhow::ensure!(
            aux_base + tail_rows == initial_len && drafter_kv.len() == aux_base,
            "spec prefill aux tail: encoded {} + tail rows [{aux_base}, {}) do not cover the {initial_len}-token prompt",
            drafter_kv.len(),
            aux_base + tail_rows
        );
    } else {
        anyhow::ensure!(
            aux_pending.len() == initial_len * fc_in_dim,
            "spec prefill aux rows: expected {} floats, got {}",
            initial_len * fc_in_dim,
            aux_pending.len()
        );
    }

    let gpu_accept =
        sampler.pure_greedy() && !snap.spec_no_gpu_accept;

    let use_graph_chain = use_drafter_kv
        && cond_mode == "shift"
        && k >= 2
        && eagle3_graph_chain_from(snap.eagle3_graph_chain.as_deref())
        && !snap.eagle3_no_device_chain;
    if use_graph_chain {
        let kd = k - 1;
        let drafter_cap = drafter_kv.kv_cap();
        let slack = nv_specdecode::eagle3_loader::DRAFTER_KV_CAP_SLACK;
        let chain_cap =
            nv_specdecode::chain::chain_graph_cap(max_seq, k, drafter_cap, slack);
        let rebuild = match &taken_chain {
            Some(cg) => {
                cg.kd() != kd
                    || cg.cap()
                        < nv_specdecode::chain::chain_graph_cap(max_seq, k, drafter_cap, slack)
            }
            None => true,
        };
        if rebuild {
            let built = shared.proposer.scorer().new_chain_graph(chain_cap, kd);
            match built {
                Ok(cg) => {
                    tracing::info!(kd, cap = chain_cap, "draft-chain CUDA graph allocated");
                    taken_chain = Some(cg);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "draft-chain graph alloc failed; using eager chain");
                    taken_chain = None;
                }
            }
        }
    }

    device
        .synchronize()
        .map_err(|e| anyhow::anyhow!("spec prefill sync: {e}"))?;

    let cmask = lower_tri_mask(k);

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    let mut emitter = StreamEmitter::new(&stop_strings);
    let mut detok = IncrementalDetok::new(tokenizer.clone());
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();
    let mut done = false;

    let mut total_rounds: u64 = 0;
    let mut total_drafts_accepted: u64 = 0;
    let mut adak = adaptive.then(|| AdaptiveK::new(k, k_env));
    let mut k_hist: Vec<u64> = vec![0; k + 1];

    let suffix_enabled = suffix_drafter_enabled(snap.suffix_drafter.as_deref())
        && use_drafter_kv
        && cond_mode == "shift";
    let suffix_min = suffix_min_match(snap.suffix_min_match.as_deref());
    let mut sam = suffix_enabled.then(nv_specdecode::SuffixAutomaton::new);
    let mut sam_fed = 0usize;
    let mut drafter_ema = nv_specdecode::AcceptEma::new(0.2, 2.0);
    let mut suffix_rounds: u64 = 0;
    let mut suffix_accepted: u64 = 0;

    let mut pos_accepted: Vec<u64> = vec![0; k.saturating_sub(1)];
    let mut pos_rejected: Vec<u64> = vec![0; k.saturating_sub(1)];
    let mut total_emitted: u64 = 0;
    let mut total_draft_ms = 0.0f64;
    let mut total_verify_ms = 0.0f64;
    let loop_start = std::time::Instant::now();
    let mut pending: Vec<String> = Vec::new();
    let send_timeout = sse_send_timeout();
    let mut aborted = false;

    tokio::task::block_in_place(|| -> anyhow::Result<()> {
    let mut bonus = sampler.draw_from_logits(&seed);

    let mut first_tok_pre_emitted = false;
    let mut first_tok_stop_hit = false;
    if max_new > 0 && !eos_ids.contains(&bonus) {
        let new_text = detok.push(bonus)?;
        let (piece, stop_hit) = emitter.step(new_text);
        first_tok_stop_hit = stop_hit;
        first_tok_pre_emitted = true;
        if !piece.is_empty() {
            let outcome = push_event_blocking(tx, ChatEvent::TextDelta(piece), send_timeout);
            if outcome != SsePush::Sent {
                log_sse_abort(outcome, "gemma4-spec");
                aborted = true;
                return Ok(());
            }
        }
    }

    if defer_drafter && preencode_limit > drafter_kv.len() {
        let t_arm = std::time::Instant::now();
        let aux_t = aux_proj
            .take()
            .ok_or_else(|| anyhow::anyhow!("aux projection missing before drafter arming"))?;
        shared
            .proposer
            .scorer()
            .preencode_context(
                &mut drafter_kv,
                &context,
                &aux_t,
                aux_base,
                preencode_limit,
                shift_prefill,
            )
            .map_err(|e| anyhow::anyhow!("deferred drafter preencode to {preencode_limit}: {e}"))?;
        let drop = preencode_limit - aux_base;
        let rows = aux_t.dims()[0];
        aux_proj = Some(aux_t.narrow(0, drop, rows - drop)?);
        aux_base = preencode_limit;
        if prof_prefill {
            let _ = device.synchronize();
            eprintln!(
                "[NV_PROF_PREFILL] deferred drafter arm to {preencode_limit}: {:.1} ms",
                t_arm.elapsed().as_secs_f64() * 1000.0
            );
        }
    }

    while !done && generated_ids.len() < max_new {
        if tx.is_closed() {
            log_sse_abort(SsePush::Closed, "gemma4-spec");
            aborted = true;
            return Ok(());
        }

        if !aux_pending.is_empty() {
            let rows = aux_pending.len() / fc_in_dim;
            let new_aux = candle_core::Tensor::from_vec(
                std::mem::take(&mut aux_pending),
                (rows, fc_in_dim),
                &device,
            )?;
            let projected = shared
                .proposer
                .scorer()
                .project_aux(&new_aux)
                .map_err(|e| anyhow::anyhow!("project_aux: {e}"))?;
            aux_proj = Some(match aux_proj.take() {
                Some(prev) => candle_core::Tensor::cat(&[&prev, &projected], 0)?,
                None => projected,
            });
        }
        let aux_t = aux_proj
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("aux projection missing"))?;
        let k_eff = adak.as_ref().map_or(k, AdaptiveK::k_eff);
        if let Some(h) = k_hist.get_mut(k_eff) {
            *h += 1;
        }

        let mut suffix_draft: Option<Vec<u32>> = None;
        if let Some(sam) = sam.as_mut() {
            sam.extend_slice(&context[sam_fed..]);
            sam.extend(bonus);
            sam_fed = context.len() + 1;
            if let Some(p) = sam.propose(k - 1, suffix_min) {
                if nv_specdecode::suffix_arm_wins(
                    p.tokens.len(),
                    suffix_min,
                    p.match_len,
                    drafter_ema.value(),
                ) {
                    suffix_draft = Some(p.tokens);
                }
            }
        }
        let from_suffix = suffix_draft.is_some();

        let t_draft = std::time::Instant::now();
        let mut draft = if let Some(sd) = suffix_draft.take() {
            pad_suffix_draft(sd, k, bonus)
        } else if use_drafter_kv {
            match cond_mode.as_str() {

                "shift" => {
                    match (use_graph_chain, taken_chain.as_mut()) {
                        (true, Some(cg)) => shared
                            .proposer
                            .scorer()
                            .chain_draft_cached_shift_graphed_tail(
                                &mut drafter_kv,
                                cg,
                                &context,
                                aux_t,
                                aux_base,
                                k_eff - 1,
                                bonus,
                            )
                            .map_err(|e| {
                                anyhow::anyhow!("chain_draft_cached_shift_graphed: {e}")
                            })?,
                        _ => shared
                            .proposer
                            .scorer()
                            .chain_draft_cached_cond_tail(
                                &mut drafter_kv,
                                &context,
                                aux_t,
                                aux_base,
                                k_eff - 1,
                                Some(bonus),
                                true,
                            )
                            .map_err(|e| anyhow::anyhow!("chain_draft_cached_cond(shift): {e}"))?,
                    }
                }
                "bonus" => shared
                    .proposer
                    .scorer()
                    .chain_draft_cached_cond_tail(
                        &mut drafter_kv,
                        &context,
                        aux_t,
                        aux_base,
                        k,
                        Some(bonus),
                        false,
                    )
                    .map_err(|e| anyhow::anyhow!("chain_draft_cached_cond(bonus): {e}"))?,
                _ => shared
                    .proposer
                    .scorer()
                    .chain_draft_cached_cond_tail(&mut drafter_kv, &context, aux_t, aux_base, k, None, false)
                    .map_err(|e| anyhow::anyhow!("chain_draft_cached: {e}"))?,
            }
        } else if shift_forced {

            let mut throwaway = nv_specdecode::eagle3_loader::DrafterKvCache::new();
            shared
                .proposer
                .scorer()
                .chain_draft_cached_cond(&mut throwaway, &context, aux_t, k - 1, Some(bonus), true)
                .map_err(|e| anyhow::anyhow!("chain_draft_cached_cond(shift-force): {e}"))?
        } else {
            shared
                .proposer
                .scorer()
                .chain_draft_projected(&context, aux_t, k)
                .map_err(|e| anyhow::anyhow!("chain_draft: {e}"))?
        };

        let draft_ms_obs = t_draft.elapsed().as_secs_f64() * 1000.0;
        if adak.is_some() && draft.len() + 1 < k {
            let pad = draft.last().copied().unwrap_or(bonus);
            draft.resize(k - 1, pad);
        }

        if use_drafter_kv && !from_suffix {
            aux_proj = None;
            aux_base = context.len();
        }

        if prof {
            let _ = device.synchronize();
            total_draft_ms += t_draft.elapsed().as_secs_f64() * 1000.0;
        }

        anyhow::ensure!(
            committed == context.len(),
            "spec round desync: committed={committed} but context has {} tokens",
            context.len()
        );
        let batch = build_chain_batch(bonus, &draft, k, cond_mode == "shift")?;
        let positions = chain_positions(committed, k);

        let t_verify = std::time::Instant::now();
        let verify_out = ChainVerifier::verify_chain(
            &mut gv,
            &batch,
            &positions,
            &cmask,
            committed,
            !gpu_accept,
        )
        .map_err(|e| anyhow::anyhow!("graphed chain verify: {e}"))?;
        let verify_ms_obs = t_verify.elapsed().as_secs_f64() * 1000.0;
        let gaux = verify_out.aux;
        let (verify_logits, chain_acc) = match verify_out.judgment {
            ChainJudgment::Argmax(amax) => (None, Some(accept_prefix_argmax(&batch, &amax)?)),
            ChainJudgment::Logits { data, .. } => (Some(data), None),
        };
        if prof {
            let _ = device.synchronize();
            total_verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
        }
        total_rounds += 1;

        let mut next_bonus: Option<u32> = None;
        let mut round_accepted = 0usize;
        let mut i = 0usize;
        loop {
            let tok = if i == 0 {
                bonus
            } else {
                let outcome = match &chain_acc {

                    Some(acc) => {
                        if i < acc.commit_len {
                            DraftOutcome::Accept
                        } else {
                            DraftOutcome::Reject(acc.next_bonus)
                        }
                    }
                    None => {
                        let logits = verify_logits.as_ref().expect("cpu accept path has logits");
                        sampler.accept_draft(&logits[(i - 1) * vocab..i * vocab], batch[i])
                    }
                };
                match outcome {
                    DraftOutcome::Accept => {
                        total_drafts_accepted += 1;
                        pos_accepted[i - 1] += 1;
                        round_accepted += 1;
                        batch[i]
                    }
                    DraftOutcome::Reject(repl) => {
                        pos_rejected[i - 1] += 1;
                        next_bonus = Some(repl);
                        break;
                    }
                }
            };

            if generated_ids.len() >= max_new {
                finish_reason = "length".into();
                done = true;
                break;
            }
            context.push(tok);
            aux_pending.extend_from_slice(&aux_row_extract(&gaux, n_layers_aux, k, hidden, i)?);
            committed += 1;
            sampler.commit(tok);

            if eos_ids.contains(&tok) {
                finish_reason = "stop".into();
                done = true;
                break;
            }

            generated_ids.push(tok);
            completion_tokens = generated_ids.len() as u32;
            total_emitted += 1;

            let stop_hit = if i == 0 && std::mem::take(&mut first_tok_pre_emitted) {
                first_tok_stop_hit
            } else {
                let new_text = detok.push(tok)?;
                let (piece, stop_hit) = emitter.step(new_text);
                if !piece.is_empty() {
                    pending.push(piece);
                }
                stop_hit
            };
            if stop_hit {
                finish_reason = "stop".into();
                done = true;
                break;
            }

            i += 1;
            if i >= k {
                break;
            }
        }

        if from_suffix {
            suffix_rounds += 1;
            suffix_accepted += round_accepted as u64;
        } else {
            drafter_ema.observe(round_accepted);
            if let Some(a) = adak.as_mut() {
                a.observe(
                    k_eff.saturating_sub(1),
                    round_accepted,
                    draft_ms_obs,
                    verify_ms_obs,
                );
            }
        }

        if !done {
            bonus = match next_bonus {
                Some(repl) => repl,
                None => match &chain_acc {

                    Some(acc) => acc.next_bonus,
                    None => {
                        let logits = verify_logits.as_ref().expect("cpu accept path has logits");
                        sampler.draw_from_logits(&logits[(k - 1) * vocab..k * vocab])
                    }
                },
            };
        }

        for piece in pending.drain(..) {
            let outcome = push_event_blocking(tx, ChatEvent::TextDelta(piece), send_timeout);
            if outcome != SsePush::Sent {
                log_sse_abort(outcome, "gemma4-spec");
                aborted = true;
                return Ok(());
            }
        }
    }

    if !generated_ids.is_empty() {
        if let Ok(full) = crate::oapi::chat_engine::stream::decode_keeping_wire(&tokenizer, &generated_ids) {
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                let outcome = push_event_blocking(tx, ChatEvent::TextDelta(tail), send_timeout);
                if outcome != SsePush::Sent {
                    log_sse_abort(outcome, "gemma4-spec");
                    aborted = true;
                }
            }
        }
    }
    Ok(())
    })?;
    device.synchronize().ok();

    {
        let model_rounds = total_rounds.saturating_sub(suffix_rounds);
        if model_rounds > 0 {
            arm_ema_observe(
                DrafterArm::Eagle3,
                total_drafts_accepted.saturating_sub(suffix_accepted) as f64
                    / model_rounds as f64,
            );
        }
    }

    if prof {
        let loop_ms = loop_start.elapsed().as_secs_f64() * 1000.0;
        let rounds = total_rounds.max(1) as f64;
        if suffix_enabled {
            eprintln!(
                "[NV_PROF_CHAT][spec] SUFFIX rounds={suffix_rounds}/{total_rounds} accepted={suffix_accepted} acc/suffix_round={:.2} drafter_ema={:.2}",
                suffix_accepted as f64 / suffix_rounds.max(1) as f64,
                drafter_ema.value(),
            );
        }
        eprintln!(
            "[NV_PROF_CHAT][spec] GRAPHED SUMMARY gpu_accept={gpu_accept} rounds={} emitted={} drafts_accepted={} tokens/round={:.2} draft-accept={:.2} tok/s={:.1} ms_per_tok={:.2} draft_ms/round={:.2} verify_ms/round={:.2}",
            total_rounds,
            total_emitted,
            total_drafts_accepted,
            total_emitted as f64 / rounds,
            total_drafts_accepted as f64 / (rounds * (k - 1) as f64),
            (total_emitted as f64 / loop_ms) * 1000.0,
            loop_ms / total_emitted.max(1) as f64,
            total_draft_ms / rounds,
            total_verify_ms / rounds,
        );

        let per_round: Vec<String> = pos_accepted
            .iter()
            .map(|a| format!("{:.3}", *a as f64 / rounds))
            .collect();
        let cond: Vec<String> = pos_accepted
            .iter()
            .zip(pos_rejected.iter())
            .map(|(a, r)| {
                let reached = a + r;
                if reached == 0 {
                    "-".to_string()
                } else {
                    format!("{:.3}", *a as f64 / reached as f64)
                }
            })
            .collect();
        let counts: Vec<String> = pos_accepted
            .iter()
            .zip(pos_rejected.iter())
            .map(|(a, r)| format!("{a}/{r}"))
            .collect();
        if let Some(a) = adak.as_ref() {
            let hist: Vec<String> = k_hist
                .iter()
                .enumerate()
                .filter(|(_, n)| **n > 0)
                .map(|(kk, n)| format!("{kk}:{n}"))
                .collect();
            eprintln!(
                "[NV_PROF_CHAT][spec] ADAPTIVE K k_graph={} k_final={} p_ema={:.3} d_graph={:.2} d_eager={:.2} verify_ema={:.2} k_hist=[{}]",
                k,
                a.k_eff(),
                a.p_ema,
                a.d_graph_ms,
                a.d_eager_ms,
                a.verify_ms,
                hist.join(","),
            );
        }
        eprintln!(
            "[NV_PROF_CHAT][spec] ACCEPT POS cond_mode={} k={} rounds={} acc_pos=[{}] cond=[{}] n_acc/rej=[{}]",
            if cond_mode.is_empty() { "default" } else { &cond_mode },
            k,
            total_rounds,
            per_round.join(","),
            cond.join(","),
            counts.join(","),
        );
    }

    Ok((gv, finish_reason, completion_tokens, aborted))
    })();

    drop(permit);

    let (finish_reason, completion_tokens, aborted) = {
        let mut pool = shared.pool.lock().await;
        if had_lease {
            pool.lease_out = false;
        }
        if pool.chain.is_none() {
            pool.chain = taken_chain.take();
        }
        match core {
            Ok((gv, finish_reason, completion_tokens, aborted)) => {
                if graph_cache_enabled && pool.verify.is_none() && !pool.lease_out {
                    pool.verify = Some(gv);
                    if let Some(g) = admit_guard.as_mut() {
                        g.set_sticky(sticky);
                    }
                } else {
                    drop(gv);
                    if let Some(g) = admit_guard.as_mut() {
                        g.set_sticky(0);
                    }
                }
                (finish_reason, completion_tokens, aborted)
            }
            Err(e) => {
                if let Some(g) = admit_guard.as_mut() {
                    g.set_sticky(0);
                }
                return Err(e);
            }
        }
    };

    if !aborted {
        let _ = push_event_async(
            tx,
            ChatEvent::Done {
                finish_reason,
                completion_tokens,
            },
            sse_send_timeout(),
        )
        .await;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) async fn run_sampling_gemma4_spec_dflash(
    model: Arc<nv_models::gemma4::Gemma4>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    prompt_ids: Vec<u32>,
    max_new: usize,
    kv_max_seq_len: usize,
    eos_ids: &[u32],
    stop_strings: Vec<String>,
    dflash: Arc<DFlashShared>,
    class: PromptClass,
    mut sampler: ChatSampler,
    snap: &SpecEnvSnapshot,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    use nv_specdecode::chain::{
        accept_prefix_argmax, aux_row_extract, build_chain_batch, chain_positions, lower_tri_mask,
        ChainJudgment, ChainVerifier,
    };

    let prof = env_flag_enabled(snap.prof_chat.as_deref());

    let model_arc = model;
    let model = &*model_arc;

    let block_size = dflash.drafter.config().block_size;
    let k = {
        let base = dflash_k(snap.dflash_k.as_deref(), block_size);
        if class == PromptClass::Prose {
            dflash_prose_k(snap.dflash_k_prose.as_deref(), block_size, base)
        } else {
            base
        }
    };
    let Some((max_seq, max_new)) = spec_verify_window(prompt_ids.len(), max_new, k, kv_max_seq_len)
    else {
        anyhow::bail!(
            "spec verify cache for {} prompt + {max_new} new + k={k} does not fit the {kv_max_seq_len}-token KV window",
            prompt_ids.len()
        );
    };
    debug_assert!(max_seq <= kv_max_seq_len);

    let graph_cache_enabled = !snap.eagle3_no_graph_cache;
    let (sticky, extra) = {
        let b = nv_models::gemma4::kv_budget(
            model.config(),
            verify_cache_capacity(max_seq),
            nv_models::gemma4::verify_kv_use_fp8(),
            nv_models::gemma4::kv_ring_enabled(),
            {
                let c = dflash.drafter.config();
                c.num_hidden_layers * c.kv_out_dim()
            },
        );
        let hd512_scratch = nv_models::gemma4::gqa512_verify_scratch_bytes(model.config()) as u64;
        (
            b.verify_total() as u64 + hd512_scratch,
            b.drafter_kv_bytes as u64,
        )
    };

    let shared = dflash;
    let (mut taken_gv, had_lease, mut taken_chain, mut taken_dd) = {
        let mut pool = shared.pool.lock().await;
        let gv = pool.verify.take();
        let had = gv.is_some();
        if had {
            pool.lease_out = true;
        }
        (gv, had, pool.chain.take(), pool.dflash_draft.take())
    };

    let admit_res = if graph_cache_enabled && had_lease {
        crate::oapi::admission::admit_or_bail(sticky, extra, "gemma4-spec-dflash").await
    } else {
        crate::oapi::admission::admit_or_bail(0, sticky + extra, "gemma4-spec-dflash").await
    };
    let permit_res = match &admit_res {
        Ok(_) => Some(acquire_chat_permit().await),
        Err(_) => None,
    };
    let (mut admit_guard, permit) = match (admit_res, permit_res) {
        (Ok(g), Some(Ok(p))) => (g, p),
        (admit_res, permit_res) => {
            let mut pool = shared.pool.lock().await;
            if had_lease {
                pool.lease_out = false;
            }
            if pool.verify.is_none() {
                pool.verify = taken_gv.take();
            }
            if pool.chain.is_none() {
                pool.chain = taken_chain.take();
            }
            if pool.dflash_draft.is_none() {
                pool.dflash_draft = taken_dd.take();
            }
            drop(pool);
            admit_res?;
            match permit_res {
                Some(Err(e)) => return Err(e),
                _ => anyhow::bail!("spec admission/permit acquisition failed"),
            }
        }
    };

    if let Err(e) = device.synchronize() {
        warn!(error = %e, "pre-spec device sync surfaced a stale error (cleared); continuing");
        let _ = device.synchronize();
    }
    let aux_layers = shared.aux_layers.clone();
    let n_layers_aux = aux_layers.len();
    let fc_in_dim = shared.drafter.config().fc_in_dim();
    let vocab = model.config().vocab_size;
    let hidden = model.config().hidden_size;

    let dev = match &device {
        candle_core::Device::Cuda(d) => d.clone(),
        _ => anyhow::bail!("graphed spec requires a CUDA device"),
    };
    let stream = nv_layers::cuda_stream::current_stream(&dev);

    anyhow::ensure!(
        fc_in_dim == n_layers_aux * hidden,
        "dflash fc_in {fc_in_dim} != aux layers {n_layers_aux} * hidden {hidden}"
    );

    let mut context: Vec<u32> = prompt_ids.clone();
    let initial_len = context.len();

    let core = (|| -> anyhow::Result<(
        nv_models::gemma4_graph::GraphedGemma4Verify<Arc<nv_models::gemma4::Gemma4>>,
        String,
        u32,
        bool,
    )> {

    let mut gv = take_reusable_or_build(
        &mut taken_gv,
        |gv| {
            verify_graph_reusable(gv.k(), gv.cache_capacity(), k, max_seq)
                && Arc::ptr_eq(gv.model(), &model_arc)
        },
        || {
            if let Err(e) = device.synchronize() {
                warn!(error = %e, "verifier teardown surfaced a stale error (cleared); continuing");
                let _ = device.synchronize();
            }

            let cache = model
                .new_verify_cache(verify_cache_capacity(max_seq))
                .map_err(|e| anyhow::anyhow!("alloc verify cache: {e}"))?;
            nv_models::gemma4_graph::GraphedGemma4Verify::new(
                model_arc.clone(),
                cache,
                &device,
                k,
                aux_layers.clone(),
            )
            .map_err(|e| anyhow::anyhow!("construct GraphedGemma4Verify: {e}"))
        },
    )?;

    let prefill_chunk = spec_prefill_chunk(snap.spec_prefill_chunk.as_deref())
        .min(nv_models::gemma4::VERIFY_PREFILL_CHUNK);

    let prof_prefill = snap.prof_prefill;
    let prefill_start = std::time::Instant::now();
    let mut prefill_verify_ms = 0.0f64;
    let mut prefill_drafter_ms = 0.0f64;

    let graph_eager_body = snap.dflash_graph_eager;
    let use_graph = !snap.dflash_no_graph
        && shared.drafter.dtype() == candle_core::DType::BF16;
    let qr = shared.drafter.config().query_rows();
    let mut dflash_graph: Option<nv_specdecode::dflash::DFlashBlockGraph> = None;
    let mut dctx = nv_specdecode::dflash::DFlashContextKv::empty();
    if use_graph {
        let needed = max_seq + qr;
        match taken_dd.take() {
            Some((mut ctx, bg))
                if !bg.disabled() && bg.cap() >= needed && ctx.capacity() >= needed =>
            {
                ctx.reset();
                dctx = ctx;
                dflash_graph = Some(bg);
            }
            stale => {
                drop(stale);
                let graph_cap = kv_max_seq_len.max(max_seq) + qr + 8;
                let built = shared.drafter.new_context_kv(graph_cap).and_then(|ctx| {
                    Ok((ctx, shared.drafter.new_block_graph(graph_cap)?))
                });
                match built {
                    Ok((ctx, bg)) => {
                        dctx = ctx;
                        dflash_graph = Some(bg);
                    }
                    Err(e) => {
                        warn!(error = %e, "dflash block graph unavailable; using eager drafter");
                    }
                }
            }
        }
    }
    let mut committed = 0usize;
    let mut seed: Vec<f32> = Vec::new();
    let mut aux_pending: Vec<f32> = Vec::new();
    while committed < initial_len {
        let t_chunk = std::time::Instant::now();
        let len = prefill_chunk.min(initial_len - committed);
        let last = committed + len == initial_len;
        let pmask_d = stream
            .clone_htod(&lower_tri_mask(len))
            .map_err(|e| anyhow::anyhow!("prefill mask htod: {e:?}"))?;
        let ppos: Vec<i32> = (committed as i32..(committed + len) as i32).collect();
        let (chunk_logits, chunk_aux) = model
            .forward_verify_tail(
                &context[committed..committed + len],
                &ppos,
                &pmask_d,
                committed,
                &aux_layers,
                gv.cache_mut(),
                if last { 1 } else { 0 },
            )
            .map_err(|e| anyhow::anyhow!("spec prefill forward_verify (chunk at {committed}): {e}"))?;
        if let Some(lg) = chunk_logits {
            seed = lg
                .to_dtype(candle_core::DType::F32)?
                .flatten_all()?
                .to_vec1()?;
        }
        let chunk_aux_cat =
            candle_core::Tensor::cat(&chunk_aux.iter().collect::<Vec<_>>()[..], 1)?;
        let t_verify_done = if prof_prefill {
            let _ = device.synchronize();
            prefill_verify_ms += t_chunk.elapsed().as_secs_f64() * 1000.0;
            Some(std::time::Instant::now())
        } else {
            None
        };
        let projected = shared
            .drafter
            .project_aux(&chunk_aux_cat)
            .map_err(|e| anyhow::anyhow!("project_aux (prefill chunk at {committed}): {e}"))?;
        let cpos: Vec<u32> = (committed as u32..(committed + len) as u32).collect();
        shared
            .drafter
            .append_context_kv(&mut dctx, &projected, &cpos)
            .map_err(|e| anyhow::anyhow!("drafter ctx append (prefill chunk at {committed}): {e}"))?;
        committed += len;
        if let Some(t) = t_verify_done {
            let _ = device.synchronize();
            let d = t.elapsed().as_secs_f64() * 1000.0;
            prefill_drafter_ms += d;
            eprintln!(
                "[NV_PROF_PREFILL] chunk at {} len {len}: verify+seed {:.1} ms drafter {:.1} ms",
                committed - len,
                (t - t_chunk).as_secs_f64() * 1000.0,
                d
            );
        }
    }
    if prof_prefill {
        let total_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "[NV_PROF_PREFILL] SUMMARY prompt={initial_len} chunks={} total {:.1} ms ({:.1} tok/s) verify {:.1} ms drafter {:.1} ms other {:.1} ms",
            initial_len.div_ceil(prefill_chunk.max(1)),
            total_ms,
            initial_len as f64 / (total_ms / 1000.0),
            prefill_verify_ms,
            prefill_drafter_ms,
            total_ms - prefill_verify_ms - prefill_drafter_ms
        );
    }
    anyhow::ensure!(
        seed.len() == vocab,
        "spec prefill seed row: expected {vocab} logits, got {}",
        seed.len()
    );
    anyhow::ensure!(
        dctx.len() == initial_len,
        "spec prefill drafter ctx: {} rows for the {initial_len}-token prompt",
        dctx.len()
    );

    let gpu_accept =
        sampler.pure_greedy() && !snap.spec_no_gpu_accept;

    device
        .synchronize()
        .map_err(|e| anyhow::anyhow!("spec prefill sync: {e}"))?;

    let cmask = lower_tri_mask(k);

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    let mut emitter = StreamEmitter::new(&stop_strings);
    let mut detok = IncrementalDetok::new(tokenizer.clone());
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();
    let mut done = false;

    let mut total_rounds: u64 = 0;
    let mut total_drafts_accepted: u64 = 0;

    let mut pos_accepted: Vec<u64> = vec![0; k.saturating_sub(1)];
    let mut pos_rejected: Vec<u64> = vec![0; k.saturating_sub(1)];
    let mut total_emitted: u64 = 0;
    let mut total_draft_ms = 0.0f64;
    let mut total_verify_ms = 0.0f64;
    let loop_start = std::time::Instant::now();
    let mut pending: Vec<String> = Vec::new();
    let send_timeout = sse_send_timeout();
    let mut aborted = false;

    let suffix_enabled =
        suffix_drafter_enabled(snap.suffix_drafter.as_deref());
    let suffix_min = suffix_min_match(snap.suffix_min_match.as_deref());
    let mut sam = suffix_enabled.then(nv_specdecode::SuffixAutomaton::new);
    let mut sam_fed = 0usize;
    let mut drafter_ema = nv_specdecode::AcceptEma::new(0.2, 2.0);
    let mut suffix_rounds: u64 = 0;
    let mut suffix_accepted: u64 = 0;

    tokio::task::block_in_place(|| -> anyhow::Result<()> {
    let mut bonus = sampler.draw_from_logits(&seed);

    let mut first_tok_pre_emitted = false;
    let mut first_tok_stop_hit = false;
    if max_new > 0 && !eos_ids.contains(&bonus) {
        let new_text = detok.push(bonus)?;
        let (piece, stop_hit) = emitter.step(new_text);
        first_tok_stop_hit = stop_hit;
        first_tok_pre_emitted = true;
        if !piece.is_empty() {
            let outcome = push_event_blocking(tx, ChatEvent::TextDelta(piece), send_timeout);
            if outcome != SsePush::Sent {
                log_sse_abort(outcome, "gemma4-spec-dflash");
                aborted = true;
                return Ok(());
            }
        }
    }

    while !done && generated_ids.len() < max_new {
        if tx.is_closed() {
            log_sse_abort(SsePush::Closed, "gemma4-spec-dflash");
            aborted = true;
            return Ok(());
        }

        if !aux_pending.is_empty() {
            let rows = aux_pending.len() / fc_in_dim;
            let new_aux = candle_core::Tensor::from_vec(
                std::mem::take(&mut aux_pending),
                (rows, fc_in_dim),
                &device,
            )?;
            let projected = shared
                .drafter
                .project_aux(&new_aux)
                .map_err(|e| anyhow::anyhow!("project_aux: {e}"))?;
            let base = dctx.len();
            let cpos: Vec<u32> = (base as u32..(base + rows) as u32).collect();
            shared
                .drafter
                .append_context_kv(&mut dctx, &projected, &cpos)
                .map_err(|e| anyhow::anyhow!("drafter ctx append: {e}"))?;
        }
        anyhow::ensure!(
            committed == context.len() && dctx.len() == committed,
            "spec round desync: committed={committed}, context={}, drafter ctx={}",
            context.len(),
            dctx.len()
        );

        let mut suffix_draft: Option<Vec<u32>> = None;
        if let Some(sam) = sam.as_mut() {
            sam.extend_slice(&context[sam_fed..]);
            sam.extend(bonus);
            sam_fed = context.len() + 1;
            if let Some(p) = sam.propose(k - 1, suffix_min) {
                if nv_specdecode::suffix_arm_wins(
                    p.tokens.len(),
                    suffix_min,
                    p.match_len,
                    drafter_ema.value(),
                ) {
                    suffix_draft = Some(p.tokens);
                }
            }
        }
        let from_suffix = suffix_draft.is_some();

        let t_draft = std::time::Instant::now();
        let draft = if let Some(sd) = suffix_draft.take() {
            pad_suffix_draft(sd, k, bonus)
        } else {
            let graphed = match dflash_graph.as_mut() {
                Some(bg) if !bg.disabled() => {
                    match shared
                        .drafter
                        .draft_block_graphed(&dctx, bg, bonus, graph_eager_body)
                    {
                        Ok(d) => Some(d),
                        Err(e) => {
                            warn!(
                                error = %e,
                                "dflash graphed draft failed; falling back to eager permanently"
                            );
                            bg.disable();
                            None
                        }
                    }
                }
                _ => None,
            };
            match graphed {
                Some(d) => d,
                None => shared
                    .drafter
                    .draft_block(&dctx, bonus)
                    .map_err(|e| anyhow::anyhow!("dflash draft_block: {e}"))?,
            }
        };
        if prof {
            let _ = device.synchronize();
            total_draft_ms += t_draft.elapsed().as_secs_f64() * 1000.0;
        }

        let batch = build_chain_batch(bonus, &draft, k, true)?;
        let positions = chain_positions(committed, k);

        let t_verify = std::time::Instant::now();
        let verify_out = ChainVerifier::verify_chain(
            &mut gv,
            &batch,
            &positions,
            &cmask,
            committed,
            !gpu_accept,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "graphed chain verify (the VERIFIER CUDA graph, which runs regardless of \
                 NV_DFLASH_NO_GRAPH -- that knob only disables the drafter block graph): {e}"
            )
        })?;
        let gaux = verify_out.aux;
        let (verify_logits, chain_acc) = match verify_out.judgment {
            ChainJudgment::Argmax(amax) => (None, Some(accept_prefix_argmax(&batch, &amax)?)),
            ChainJudgment::Logits { data, .. } => (Some(data), None),
        };
        if prof {
            let _ = device.synchronize();
            total_verify_ms += t_verify.elapsed().as_secs_f64() * 1000.0;
        }
        total_rounds += 1;

        let mut next_bonus: Option<u32> = None;
        let mut round_accepted = 0usize;
        let mut i = 0usize;
        loop {
            let tok = if i == 0 {
                bonus
            } else {
                let outcome = match &chain_acc {

                    Some(acc) => {
                        if i < acc.commit_len {
                            DraftOutcome::Accept
                        } else {
                            DraftOutcome::Reject(acc.next_bonus)
                        }
                    }
                    None => {
                        let logits = verify_logits.as_ref().expect("cpu accept path has logits");
                        sampler.accept_draft(&logits[(i - 1) * vocab..i * vocab], batch[i])
                    }
                };
                match outcome {
                    DraftOutcome::Accept => {
                        total_drafts_accepted += 1;
                        pos_accepted[i - 1] += 1;
                        round_accepted += 1;
                        batch[i]
                    }
                    DraftOutcome::Reject(repl) => {
                        pos_rejected[i - 1] += 1;
                        next_bonus = Some(repl);
                        break;
                    }
                }
            };

            if generated_ids.len() >= max_new {
                finish_reason = "length".into();
                done = true;
                break;
            }
            context.push(tok);
            aux_pending.extend_from_slice(&aux_row_extract(&gaux, n_layers_aux, k, hidden, i)?);
            committed += 1;
            sampler.commit(tok);

            if eos_ids.contains(&tok) {
                finish_reason = "stop".into();
                done = true;
                break;
            }

            generated_ids.push(tok);
            completion_tokens = generated_ids.len() as u32;
            total_emitted += 1;

            let stop_hit = if i == 0 && std::mem::take(&mut first_tok_pre_emitted) {
                first_tok_stop_hit
            } else {
                let new_text = detok.push(tok)?;
                let (piece, stop_hit) = emitter.step(new_text);
                if !piece.is_empty() {
                    pending.push(piece);
                }
                stop_hit
            };
            if stop_hit {
                finish_reason = "stop".into();
                done = true;
                break;
            }

            i += 1;
            if i >= k {
                break;
            }
        }

        if from_suffix {
            suffix_rounds += 1;
            suffix_accepted += round_accepted as u64;
        } else {
            drafter_ema.observe(round_accepted);
        }

        if !done {
            bonus = match next_bonus {
                Some(repl) => repl,
                None => match &chain_acc {

                    Some(acc) => acc.next_bonus,
                    None => {
                        let logits = verify_logits.as_ref().expect("cpu accept path has logits");
                        sampler.draw_from_logits(&logits[(k - 1) * vocab..k * vocab])
                    }
                },
            };
        }

        for piece in pending.drain(..) {
            let outcome = push_event_blocking(tx, ChatEvent::TextDelta(piece), send_timeout);
            if outcome != SsePush::Sent {
                log_sse_abort(outcome, "gemma4-spec-dflash");
                aborted = true;
                return Ok(());
            }
        }
    }

    if !generated_ids.is_empty() {
        if let Ok(full) = crate::oapi::chat_engine::stream::decode_keeping_wire(&tokenizer, &generated_ids) {
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                let outcome = push_event_blocking(tx, ChatEvent::TextDelta(tail), send_timeout);
                if outcome != SsePush::Sent {
                    log_sse_abort(outcome, "gemma4-spec-dflash");
                    aborted = true;
                }
            }
        }
    }
    Ok(())
    })?;
    device.synchronize().ok();

    {
        let model_rounds = total_rounds.saturating_sub(suffix_rounds);
        if model_rounds > 0 {
            arm_ema_observe(
                DrafterArm::DFlash,
                total_drafts_accepted.saturating_sub(suffix_accepted) as f64
                    / model_rounds as f64,
            );
        }
    }

    if prof {
        let loop_ms = loop_start.elapsed().as_secs_f64() * 1000.0;
        let rounds = total_rounds.max(1) as f64;
        if suffix_enabled {
            eprintln!(
                "[NV_PROF_CHAT][spec-dflash] SUFFIX rounds={suffix_rounds}/{total_rounds} accepted={suffix_accepted} acc/suffix_round={:.2} drafter_ema={:.2}",
                suffix_accepted as f64 / suffix_rounds.max(1) as f64,
                drafter_ema.value(),
            );
        }
        eprintln!(
            "[NV_PROF_CHAT][spec-dflash] GRAPHED SUMMARY gpu_accept={gpu_accept} block_size={block_size} rounds={} emitted={} drafts_accepted={} tokens/round={:.2} draft-accept={:.2} tok/s={:.1} ms_per_tok={:.2} draft_ms/round={:.2} verify_ms/round={:.2}",
            total_rounds,
            total_emitted,
            total_drafts_accepted,
            total_emitted as f64 / rounds,
            total_drafts_accepted as f64 / (rounds * (k - 1) as f64),
            (total_emitted as f64 / loop_ms) * 1000.0,
            loop_ms / total_emitted.max(1) as f64,
            total_draft_ms / rounds,
            total_verify_ms / rounds,
        );

        let per_round: Vec<String> = pos_accepted
            .iter()
            .map(|a| format!("{:.3}", *a as f64 / rounds))
            .collect();
        let cond: Vec<String> = pos_accepted
            .iter()
            .zip(pos_rejected.iter())
            .map(|(a, r)| {
                let reached = a + r;
                if reached == 0 {
                    "-".to_string()
                } else {
                    format!("{:.3}", *a as f64 / reached as f64)
                }
            })
            .collect();
        let counts: Vec<String> = pos_accepted
            .iter()
            .zip(pos_rejected.iter())
            .map(|(a, r)| format!("{a}/{r}"))
            .collect();
        eprintln!(
            "[NV_PROF_CHAT][spec-dflash] ACCEPT POS k={} rounds={} acc_pos=[{}] cond=[{}] n_acc/rej=[{}]",
            k,
            total_rounds,
            per_round.join(","),
            cond.join(","),
            counts.join(","),
        );
    }

    if let Some(bg) = dflash_graph.take() {
        if !bg.disabled() {
            taken_dd = Some((dctx, bg));
        }
    }

    Ok((gv, finish_reason, completion_tokens, aborted))
    })();

    drop(permit);

    let (finish_reason, completion_tokens, aborted) = {
        let mut pool = shared.pool.lock().await;
        if had_lease {
            pool.lease_out = false;
        }
        if pool.chain.is_none() {
            pool.chain = taken_chain.take();
        }
        if pool.dflash_draft.is_none() {
            pool.dflash_draft = taken_dd.take();
        }
        match core {
            Ok((gv, finish_reason, completion_tokens, aborted)) => {
                if graph_cache_enabled && pool.verify.is_none() && !pool.lease_out {
                    pool.verify = Some(gv);
                    if let Some(g) = admit_guard.as_mut() {
                        g.set_sticky(sticky);
                    }
                } else {
                    drop(gv);
                    if let Some(g) = admit_guard.as_mut() {
                        g.set_sticky(0);
                    }
                }
                (finish_reason, completion_tokens, aborted)
            }
            Err(e) => {
                if let Some(g) = admit_guard.as_mut() {
                    g.set_sticky(0);
                }
                return Err(e);
            }
        }
    };

    if !aborted {
        let _ = push_event_async(
            tx,
            ChatEvent::Done {
                finish_reason,
                completion_tokens,
            },
            sse_send_timeout(),
        )
        .await;
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) async fn run_sampling_gemma4_spec_ungraphed(
    model: Arc<nv_models::gemma4::Gemma4>,
    tokenizer: Arc<tokenizers::Tokenizer>,
    device: candle_core::Device,
    prompt_ids: Vec<u32>,
    max_new: usize,
    eos_ids: &[u32],
    stop_strings: Vec<String>,
    eagle3: Arc<Eagle3Shared>,

    _sampler: ChatSampler,
    snap: &SpecEnvSnapshot,
    tx: &mpsc::Sender<ChatEvent>,
) -> anyhow::Result<()> {
    use nv_specdecode::eagle3::{flatten_with_mask, DraftTree};

    let prof = env_flag_enabled(snap.prof_chat.as_deref());

    if let Err(e) = device.synchronize() {
        warn!(error = %e, "pre-spec device sync surfaced a stale error (cleared); continuing");
        let _ = device.synchronize();
    }
    let _admit_guard = {
        let ctx = prompt_ids.len().saturating_add(max_new).saturating_add(1);
        let b = nv_models::gemma4::kv_budget_capped(
            model.config(),
            ctx,
            nv_models::gemma4::verify_kv_use_fp8(),
            nv_models::gemma4::kv_ring_enabled(),
            crate::oapi::admission::drafter_row_elems(),
            capped_drafter_kv_rows(ctx, drafter_kv_cap_env()),
        );
        let hd512_scratch = nv_models::gemma4::gqa512_verify_scratch_bytes(model.config());
        crate::oapi::admission::admit_or_bail(
            0,
            (b.verify_total() + b.drafter_kv_bytes + hd512_scratch) as u64,
            "gemma4-spec-ungraphed",
        )
        .await?
    };

    let _permit = acquire_chat_permit().await?;

    let eagle = &*eagle3;
    let aux_layers = eagle.aux_layers.clone();

    let mut verifier = nv_specdecode::gemma4_verifier::Gemma4Verifier::new(
        &*model,
        device.clone(),
        aux_layers.clone(),
    );

    let use_tree = snap.eagle3_tree;
    let spec_k = eagle3_k(snap.eagle3_k.as_deref(), prompt_ids.len());
    let tree_branch = eagle.proposer.config().branch_factor.max(1);
    let tree_depth = eagle.proposer.config().max_depth.max(1);
    let tree_budget = eagle.proposer.config().total_budget.max(1);
    let fc_in_dim = eagle.proposer.scorer().config().fc_in_dim();

    let mut context: Vec<u32> = prompt_ids.clone();

    let aux_for_context = |ctx: &[u32]| -> anyhow::Result<candle_core::Tensor> {
        let seq = ctx.len();
        let tokens = candle_core::Tensor::from_vec(ctx.to_vec(), (1usize, seq), &device)?;
        let positions: Vec<i32> = (0..seq as i32).collect();
        let pos = candle_core::Tensor::from_vec(positions, seq, &device)?;
        let (_lg, hs) = model.forward_with_aux_hidden(&tokens, &pos, &aux_layers)?;
        let squeezed: Vec<candle_core::Tensor> =
            hs.iter().map(|h| h.squeeze(0)).collect::<Result<_, _>>()?;
        let aux = candle_core::Tensor::cat(&squeezed.iter().collect::<Vec<_>>()[..], 1)?;
        anyhow::ensure!(
            aux.dims() == [seq, fc_in_dim],
            "aux_for_context: expected [{seq}, {fc_in_dim}], got {:?}",
            aux.dims()
        );
        Ok(aux)
    };

    let mut aux_state = aux_for_context(&context)?;

    let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
    let mut emitter = StreamEmitter::new(&stop_strings);
    let mut detok = IncrementalDetok::new(tokenizer.clone());
    let mut completion_tokens: u32 = 0;
    let mut finish_reason = "length".to_string();
    let mut done = false;

    let mut total_cycles: u64 = 0;
    let mut total_proposed: u64 = 0;
    let mut total_accepted: u64 = 0;
    let mut total_emitted: u64 = 0;
    let mut total_propose_ms = 0.0f64;
    let mut total_verify_ms = 0.0f64;
    let loop_start = std::time::Instant::now();
    let send_timeout = sse_send_timeout();
    let mut aborted = false;

    while !done && generated_ids.len() < max_new {
        if tx.is_closed() {
            log_sse_abort(SsePush::Closed, "gemma4-spec-ungraphed");
            aborted = true;
            break;
        }
        let t_prop = std::time::Instant::now();
        let tree_target = if use_tree {
            eagle.proposer.scorer().tree_draft(
                &context,
                &aux_state,
                tree_branch,
                tree_depth,
                tree_budget,
            )?
        } else {
            let chain = eagle
                .proposer
                .scorer()
                .chain_draft(&context, &aux_state, spec_k)?;
            let n = chain.len();
            DraftTree {
                tokens: chain,
                parents: (0..n)
                    .map(|i| if i == 0 { None } else { Some(i - 1) })
                    .collect(),
                depths: (1..=n).collect(),
            }
        };
        let propose_ms = t_prop.elapsed().as_secs_f64() * 1000.0;
        total_propose_ms += propose_ms;

        let n_proposed = tree_target.tokens.len();
        if n_proposed == 0 {
            break;
        }

        let (_toks, mask) = flatten_with_mask(&tree_target);

        let t_ver = std::time::Instant::now();
        let result = verifier.verify_tree(&context, &tree_target, Some(&mask))?;
        let verify_ms = t_ver.elapsed().as_secs_f64() * 1000.0;
        total_verify_ms += verify_ms;

        total_cycles += 1;
        total_proposed += n_proposed as u64;
        total_accepted += result.num_accepted as u64;
        total_emitted += result.emitted.len() as u64;

        if prof {
            eprintln!(
                "[NV_PROF_CHAT][spec] cycle={} proposed={} accepted={} emitted={} propose={:.2}ms verify={:.2}ms",
                total_cycles,
                n_proposed,
                result.num_accepted,
                result.emitted.len(),
                propose_ms,
                verify_ms,
            );
        }

        for &tok in &result.emitted {
            if generated_ids.len() >= max_new {
                finish_reason = "length".into();
                done = true;
                break;
            }
            if eos_ids.contains(&tok) {
                finish_reason = "stop".into();
                done = true;
                break;
            }
            context.push(tok);
            generated_ids.push(tok);
            completion_tokens = generated_ids.len() as u32;

            let new_text = detok.push(tok)?;
            let (piece, stop_hit) = emitter.step(new_text);

            if !piece.is_empty() {
                let outcome = push_event_async(tx, ChatEvent::TextDelta(piece), send_timeout).await;
                if outcome != SsePush::Sent {
                    log_sse_abort(outcome, "gemma4-spec-ungraphed");
                    aborted = true;
                    done = true;
                    break;
                }
            }

            if stop_hit {
                finish_reason = "stop".into();
                done = true;
                break;
            }
        }

        if !done {
            aux_state = aux_for_context(&context)?;
        }
    }

    if !aborted && !generated_ids.is_empty() {
        if let Ok(full) = crate::oapi::chat_engine::stream::decode_keeping_wire(&tokenizer, &generated_ids) {
            let tail = emitter.finish(&full);
            if !tail.is_empty() {
                let _ = push_event_async(tx, ChatEvent::TextDelta(tail), send_timeout).await;
            }
        }
    }

    let loop_ms = loop_start.elapsed().as_secs_f64() * 1000.0;
    if prof {
        let cycles = total_cycles.max(1) as f64;
        let proposed = total_proposed.max(1) as f64;
        let emitted = total_emitted.max(1) as f64;
        eprintln!(
            "[NV_PROF_CHAT][spec] SUMMARY cycles={} proposed={} accepted={} emitted={} accept_rate={:.1}% tok/s={:.1} avg_propose={:.2}ms avg_verify={:.2}ms ms_per_tok={:.2}",
            total_cycles,
            total_proposed,
            total_accepted,
            total_emitted,
            (total_accepted as f64 / proposed) * 100.0,
            (total_emitted as f64 / loop_ms) * 1000.0,
            total_propose_ms / cycles,
            total_verify_ms / cycles,
            loop_ms / emitted,
        );
    }

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

#[cfg(feature = "cuda")]
pub(crate) async fn run_sampling_gemma4_e4b(
    model: Arc<nv_models::gemma4_e4b::Gemma4E4b>,
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
    const PREFILL_CHUNK: usize = 512;

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
                model.embed_table(),
                model.embed_normalizer(),
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

    let Some((_cache_len, max_new)) = kv_window(prompt_ids.len(), max_new, kv_max_seq_len) else {
        anyhow::bail!(
            "prompt of {} tokens does not fit the {kv_max_seq_len}-token KV window",
            prompt_ids.len()
        );
    };
    let mut sampler = ChatSampler::for_request(&req, &tokenizer, eos_ids, max_new)?;
    sampler.seed_prompt(&prompt_ids);

    let mut cache: Vec<Option<(candle_core::Tensor, candle_core::Tensor)>> =
        vec![None; model.config().num_hidden_layers];
    let mut last_logits = None;
    let mut off = 0usize;
    while off < prompt_ids.len() {
        let c = (prompt_ids.len() - off).min(PREFILL_CHUNK);
        last_logits = Some(match &mm_embeds {
            Some(e) => model.forward_step_embeds(
                &prompt_ids[off..off + c],
                &e.narrow(0, off, c)?,
                off,
                &mut cache,
            )?,
            None => model.forward_step(&prompt_ids[off..off + c], off, &mut cache)?,
        });
        off += c;
    }
    let row: Vec<f32> = last_logits
        .expect("non-empty prompt")
        .to_dtype(candle_core::DType::F32)?
        .to_vec1()?;
    let mut last_out = sampler.sample(&row);
    let mut last = last_out.token;
    if last_out.exhausted {
        anyhow::bail!("no legal token at prefill: the sampling mask left every candidate at -inf");
    }

    let graph_eligible = sampling_params_from(&req).is_greedy()
        && !req.logprobs
        && req.guided.is_none()
        && req.logit_bias.is_empty()
        && std::env::var("NV_E4B_SERVE_GRAPH").ok().as_deref() != Some("0");
    if graph_eligible {
        const GRAPH_BATCH: usize = 16;
        let max_len = (prompt_ids.len() + max_new + GRAPH_BATCH + 8)
            .next_power_of_two()
            .max(64);
        match model.graphed_decoder(&cache, prompt_ids.len(), max_len) {
            Ok(mut dec) => {
                let mut generated_ids: Vec<u32> = Vec::with_capacity(max_new);
                let mut emitter = StreamEmitter::new(&req.stop);
                let mut detok = IncrementalDetok::new(tokenizer.clone());
                let mut completion_tokens: u32 = 0;
                let mut finish_reason = "length".to_string();
                let mut pending: std::collections::VecDeque<u32> = Default::default();
                let mut warmed = false;
                loop {
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
                    if stop_hit {
                        finish_reason = "stop".into();
                        break;
                    }
                    if generated_ids.len() >= max_new {
                        break;
                    }
                    last = match pending.pop_front() {
                        Some(t) => t,
                        None if !warmed => {
                            warmed = true;
                            dec.warm_step(last)?
                        }
                        None => {
                            let k = GRAPH_BATCH.min(max_new - generated_ids.len());
                            pending.extend(dec.replay_batch(last, k)?);
                            match pending.pop_front() {
                                Some(t) => t,
                                None => break,
                            }
                        }
                    };
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
                return Ok(());
            }
            Err(e) => {
                tracing::warn!(error = %e, "e4b graphed decoder unavailable; serving eager");
            }
        }
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

        let past = prompt_ids.len() + step;
        let step_logits = model.forward_step(&[last], past, &mut cache)?;
        let step_row: Vec<f32> = step_logits.to_dtype(candle_core::DType::F32)?.to_vec1()?;
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

#[cfg(all(test, feature = "cuda"))]
mod prompt_head_position_tests {
    use super::*;

    const GEMMA_BOS: u32 = 2;
    const GEMMA_EOS: [u32; 2] = [1, 106];
    const LAGUNA_BOS: u32 = 2;
    const LAGUNA_EOS: [u32; 2] = [2, 24];
    const QWEN_GENERATION_CONFIG_BOS: u32 = 248044;
    const QWEN_EOS: [u32; 2] = [248044, 248046];

    #[test]
    fn a_gemma_shaped_bos_outside_the_eos_set_is_prepended_when_the_template_omitted_it() {
        let mut ids = vec![105u32, 2364, 107];
        let head = splice_bos_at_position_0_only(&mut ids, Some(GEMMA_BOS), &GEMMA_EOS);
        assert_eq!(head, PromptHead::EnginePrependedBosAtPosition0);
        assert_eq!(ids, vec![2, 105, 2364, 107]);
    }

    #[test]
    fn a_template_that_already_opened_on_bos_is_not_given_a_second_copy() {
        let mut ids = vec![GEMMA_BOS, 105, 2364, 107];
        let before = ids.clone();
        let head = splice_bos_at_position_0_only(&mut ids, Some(GEMMA_BOS), &GEMMA_EOS);
        assert_eq!(head, PromptHead::TemplateEmittedBosAtPosition0);
        assert_eq!(ids, before);
    }

    #[test]
    fn a_dual_role_id_the_template_emitted_itself_stays_at_position_0_and_is_not_duplicated() {
        let mut ids = vec![LAGUNA_BOS, 9204, 24];
        let before = ids.clone();
        let head = splice_bos_at_position_0_only(&mut ids, Some(LAGUNA_BOS), &LAGUNA_EOS);
        assert_eq!(head, PromptHead::TemplateEmittedBosAtPosition0);
        assert_eq!(
            ids, before,
            "{BOS_IS_A_POSITION_0_ROLE_AND_THE_SAME_ID_LATER_IS_A_STOP}"
        );
        assert_ne!(
            ids.get(1).copied(),
            Some(LAGUNA_BOS),
            "{BOS_IS_A_POSITION_0_ROLE_AND_THE_SAME_ID_LATER_IS_A_STOP}"
        );
    }

    #[test]
    fn a_dual_role_id_the_template_did_not_emit_is_never_spliced_in() {
        let mut ids = vec![9204u32, 610, 24];
        let before = ids.clone();
        let head = splice_bos_at_position_0_only(&mut ids, Some(LAGUNA_BOS), &LAGUNA_EOS);
        assert_eq!(
            head,
            PromptHead::RefusedToSpliceAnIdTheCheckpointAlsoDeclaresEos
        );
        assert_eq!(ids, before);
    }

    #[test]
    fn the_qwen_generation_config_bos_that_lives_in_its_own_eos_set_never_reaches_position_0() {
        let mut ids = vec![151644u32, 872, 198];
        let head =
            splice_bos_at_position_0_only(&mut ids, Some(QWEN_GENERATION_CONFIG_BOS), &QWEN_EOS);
        assert_eq!(
            head,
            PromptHead::RefusedToSpliceAnIdTheCheckpointAlsoDeclaresEos
        );
        assert!(
            !QWEN_EOS.contains(&ids[0]),
            "{BOS_IS_A_POSITION_0_ROLE_AND_THE_SAME_ID_LATER_IS_A_STOP}"
        );
    }

    #[test]
    fn a_checkpoint_that_declares_no_bos_gets_none_invented_for_it() {
        let mut ids = vec![105u32, 2364, 107];
        let before = ids.clone();
        let head = splice_bos_at_position_0_only(&mut ids, None, &GEMMA_EOS);
        assert_eq!(head, PromptHead::CheckpointDeclaresNoBos);
        assert_eq!(
            ids, before,
            "an undeclared bos_token_id is absent, not 2: {BOS_IS_A_POSITION_0_ROLE_AND_THE_SAME_ID_LATER_IS_A_STOP}"
        );
    }

    #[test]
    fn refusing_the_splice_leaves_the_checkpoints_own_eos_set_intact() {
        let eos: Vec<u32> = LAGUNA_EOS.to_vec();
        let mut ids = vec![9204u32, 610];
        splice_bos_at_position_0_only(&mut ids, Some(LAGUNA_BOS), &eos);
        assert!(
            eos.contains(&LAGUNA_BOS),
            "id {LAGUNA_BOS} must remain a stop for generated tokens: it is the checkpoint's own \
             declaration and removing it breaks real stopping"
        );
    }
}
