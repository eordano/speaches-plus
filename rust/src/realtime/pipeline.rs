use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde_json::json;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::conversation::llm::{self, ChatMessage};
use crate::defaults;
use crate::eou::{EouConfig, EouModel};
use crate::errors::code as errcode;
use crate::stt::WhisperHandle;

use super::audio_out;
use super::audio_out_ws::{AudioPacer, WsAudioPacer};
use super::inspector::InspectorEvent;
use super::session::{CancelReason, FailReason, ResponseLane, Session};
use super::state::{ConversationItem, RespPhase, SealedBuffer, VadPhase};
use super::transport::OutboundAudioSpec;
use super::wire::{ResponseStatus, ResponseStatusReason};
use super::Intent;
use crate::inspect::AudioStore;

#[tracing::instrument(
    target = "speaches/realtime",
    name = "eou.dispatch",
    skip_all,
    fields(session_id = %session.id.as_str(), item_id = %item_id, kind = cfg.kind.as_str())
)]
pub(super) async fn run_eou_dispatch(
    session: Arc<Session>,
    cfg: EouConfig,
    model: Arc<dyn EouModel>,
    context: String,
    input_chars: u32,
    item_id: String,
    audio: Vec<f32>,
    speech_samples: usize,
    audio_ms: u64,
    suppress_response: bool,
) {
    let started = std::time::Instant::now();
    let hard_cap_ms = cfg.silence_hard_cap_ms as u64;
    let hard_cap_deadline =
        tokio::time::Instant::now() + std::time::Duration::from_millis(hard_cap_ms);
    let inference_timeout = std::time::Duration::from_millis(cfg.inference_timeout_ms as u64);

    let model_for_score = model.clone();
    let context_for_score = context.clone();
    let audio_for_score = audio.clone();
    let mut audio = audio;
    audio.truncate(speech_samples.min(audio.len()));
    let classifier = tokio::time::timeout(
        inference_timeout,
        tokio::task::spawn_blocking(move || {
            model_for_score.score_with_audio(
                &context_for_score,
                &audio_for_score,
                crate::eou::audio::SAMPLE_RATE,
            )
        }),
    );

    enum EouOutcome {
        HardCapDuringEou,
        Verdict {
            p: f32,
            cancelled_by: &'static str,
            fast_commit: bool,
        },
    }

    let outcome = match crate::eou::race_hard_cap(hard_cap_deadline, classifier).await {
        crate::eou::HardCapRace::HardCap => EouOutcome::HardCapDuringEou,
        crate::eou::HardCapRace::Completed(scored) => match scored {
            Ok(Ok(p)) if p.is_finite() && (0.0..=1.0).contains(&p) => EouOutcome::Verdict {
                p,
                cancelled_by: "none",
                fast_commit: false,
            },
            Ok(Ok(_)) => EouOutcome::Verdict {
                p: 1.0,
                cancelled_by: "garbage_prob",
                fast_commit: true,
            },
            Ok(Err(_)) => EouOutcome::Verdict {
                p: 1.0,
                cancelled_by: "error",
                fast_commit: true,
            },
            Err(_) => EouOutcome::Verdict {
                p: 1.0,
                cancelled_by: "timeout",
                fast_commit: true,
            },
        },
    };

    let (p, cancelled_by, fast_commit, hard_cap_during_eou) = match outcome {
        EouOutcome::HardCapDuringEou => {
            session.inspector.emit(InspectorEvent::EouHardCapFired {
                session_id: session.id.as_str().to_string(),
                item_id: item_id.clone(),
                phase: super::inspector::HardCapPhase::DuringEou,
                score: None,
            });

            (1.0_f32, "hard_cap", true, true)
        }
        EouOutcome::Verdict {
            p,
            cancelled_by,
            fast_commit,
        } => (p, cancelled_by, fast_commit, false),
    };

    let delay_ms = if fast_commit {
        cfg.min_delay_ms as u64
    } else if p < cfg.p_threshold {
        cfg.max_delay_ms as u64
    } else {
        crate::eou::sigmoid_lerp(
            p,
            cfg.p_threshold,
            1.0,
            cfg.max_delay_ms as u64,
            cfg.min_delay_ms as u64,
        )
    };

    if let Some(head) = crate::eou::text_head::shadow_head() {
        let p_head = head.prob(&context);
        tracing::info!(
            item_id = %item_id,
            p_head,
            p_fused = p,
            cancelled_by,
            audio_ms,
            "eou text head shadow"
        );
    }

    session.inspector.emit(InspectorEvent::EouScored {
        session_id: session.id.as_str().to_string(),
        kind: cfg.kind.as_str(),
        score: p,
        eager_score: None,
        threshold: cfg.p_threshold,
        language: None,
        input_chars: Some(input_chars),
        input_audio_ms: Some(audio_ms as u32),
        delay_ms,
        elapsed_ms: started.elapsed().as_millis() as u64,
        cancelled_by,
        hard_cap_fired: hard_cap_during_eou,
    });

    if !cfg.eager_disabled() && p >= cfg.eager_p_threshold && !suppress_response {
        super::eou_eager::try_eager_dispatch(&session, &cfg, p, audio.clone()).await;
    }

    if !hard_cap_during_eou {
        let cap_during_wait = matches!(
            crate::eou::race_hard_cap(
                hard_cap_deadline,
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)),
            )
            .await,
            crate::eou::HardCapRace::HardCap,
        );
        if cap_during_wait {
            session.inspector.emit(InspectorEvent::EouHardCapFired {
                session_id: session.id.as_str().to_string(),
                item_id: item_id.clone(),
                phase: super::inspector::HardCapPhase::DuringWait,
                score: Some(p),
            });
        }
    }

    session.clear_commit_timer().await;
    commit_after_eou(&session, item_id, audio, audio_ms, suppress_response).await;
}

#[tracing::instrument(
    target = "speaches/realtime",
    name = "turn",
    skip_all,
    fields(session_id = %session.id.as_str(), item_id = %item_id, audio_ms = audio_ms)
)]
pub(super) async fn commit_after_eou(
    session: &Arc<Session>,
    item_id: String,
    audio: Vec<f32>,
    audio_ms: u64,
    suppress_response: bool,
) {
    let (predicted_runner, predicted_llm_handle, predicted_wire_id) = {
        let mut state = session.state.lock().await;
        let already_present = state.conversation.iter().any(|i| i.id == item_id);
        if !already_present {
            state
                .conversation
                .push(ConversationItem::new_user_audio(item_id.clone()));
        }
        if matches!(state.vad, VadPhase::Stopped { .. }) {
            state.vad = VadPhase::Silent;
        }
        let predicted_wire_id = match &state.resp {
            RespPhase::Predicted { id, .. } => Some(id.clone()),
            _ => None,
        };
        let (predicted_runner, predicted_llm_handle) =
            if matches!(state.resp, RespPhase::Predicted { .. }) {
                state
                    .resp_retire_predicted_full()
                    .ok()
                    .unwrap_or((None, None))
            } else {
                (None, None)
            };
        super::session::check_or_react(session, &state);
        (predicted_runner, predicted_llm_handle, predicted_wire_id)
    };
    let predicted_id_for_log = predicted_wire_id.as_ref().map(|id| id.as_str().to_string());
    {
        let mut last = session.last_eager_dispatch_at.lock().await;
        *last = None;
    }
    let predicted_transcript = if let Some(runner) = predicted_runner.as_ref() {
        match super::eou_predicted::await_predicted_stt(runner).await {
            Ok(t) => Some(t),
            Err(err) => {
                warn!(error = %err, "speculative STT failed; falling back to fresh STT");
                None
            }
        }
    } else {
        None
    };
    let predicted_llm_text: Option<String> = match predicted_llm_handle {
        Some(handle) => {
            let runner = handle.into_runner();
            let timeout = std::time::Duration::from_millis(
                (session.eou_config.silence_hard_cap_ms as u64).max(1),
            );
            let _ = tokio::time::timeout(timeout, runner.wait_finished()).await;
            let response_id_str = predicted_id_for_log.clone().unwrap_or_default();
            if runner.overflowed() {
                let chars = runner.chars_seen();
                let dropped = runner.dropped_count();
                session
                    .inspector
                    .emit(InspectorEvent::EouPredictedOverflow {
                        session_id: session.id.as_str().to_string(),
                        response_id: response_id_str.clone(),
                        dropped_tokens: dropped,
                    });
                session
                    .inspector
                    .emit(InspectorEvent::EouPredictedRollback {
                        session_id: session.id.as_str().to_string(),
                        response_id: response_id_str,
                        reason: "predicted_overflow",
                        llm_chars_thrown: chars,
                    });
                runner.abort();
                None
            } else {
                let text = runner.snapshot_text().await;
                runner.abort();
                if text.is_empty() {
                    None
                } else {
                    Some(text)
                }
            }
        }
        None => None,
    };
    {
        session
            .send_to_client(&json!({
                "type": "input_audio_buffer.committed",
                "item_id": item_id,
            }))
            .await;
        session
            .send_to_client(&json!({
                "type": "conversation.item.added",
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "role": "user",
                    "status": "in_progress",
                    "content": [{"type": "input_audio"}],
                },
            }))
            .await;
    }
    let wants_response = session.intent == Intent::Conversation
        && !suppress_response
        && session
            .turn_detection
            .create_response
            .load(Ordering::Relaxed);
    let response_id = promoted_turn_wire_id(
        predicted_llm_text.is_some(),
        predicted_wire_id,
        session.id_source.response(),
    );
    let assistant_item_id = session.id_source.item();
    let session_for_task = session.clone();
    let utterance = move |response: Option<ResponseLane>| async move {
        let _ = process_utterance(
            &session_for_task,
            item_id,
            audio,
            audio_ms,
            suppress_response,
            predicted_transcript,
            predicted_llm_text,
            response,
        )
        .await;
    };
    let mut state = session.state.lock().await;
    let admitted = wants_response
        && state
            .resp_admit()
            .inspect_err(|violation| {
                info!(
                    ?violation,
                    "committed turn is transcribed without a response: the session already holds one"
                )
            })
            .is_ok();
    if admitted {
        let _ = session.spawn_response_lane(&mut state, response_id, assistant_item_id, |lane| {
            utterance(Some(lane))
        });
    } else {
        tokio::spawn(session.cancel.wrap_unit(utterance(None)));
    }
}

pub(super) async fn commit_bargein(session: &Arc<Session>, item_id: &str, audio_start_ms: u64) {
    if let Some(snap) = session.cancel_current_response().await {
        session.inspector.emit(InspectorEvent::BargeinFired {
            session_id: session.id.as_str().to_string(),
            played_ms: snap.played_ms,
        });
        if snap.wire_opened {
            super::events::emit_cancelled_brackets(session, &snap, CancelReason::BargeIn).await;
        } else {
            info!(
                cancelled_id = %snap.response_id,
                "barge-in: suppressed close cascade for never-opened response (W1/W2)",
            );
        }
        session.apply_truncate_to_assistant_item(&snap).await;
        super::events::emit_server_truncate(session, &snap).await;
        info!(
            cancelled_id = %snap.response_id,
            played_ms = snap.played_ms,
            "barge-in: cancelled response",
        );
    }
    {
        let mut state = session.state.lock().await;
        state.vad = VadPhase::Speaking {
            item_id: crate::types::ItemId::new(item_id.to_string()),
            audio_start_ms: crate::types::Millis(audio_start_ms),
        };
        super::session::check_or_react(session, &state);
    }
    session
        .send_to_client(&json!({
            "type": "input_audio_buffer.speech_started",
            "item_id": item_id,
            "audio_start_ms": audio_start_ms,
        }))
        .await;
}

pub(super) async fn process_utterance(
    session: &Arc<Session>,
    item_id: String,
    audio: Vec<f32>,
    audio_ms: u64,
    suppress_response: bool,
    cached_transcript: Option<String>,
    predicted_llm_text: Option<String>,
    response: Option<ResponseLane>,
) -> anyhow::Result<()> {
    {
        let sealed = SealedBuffer {
            item_id: item_id.clone(),
            audio: audio.clone(),
            audio_start_ms: 0,
            audio_end_ms: audio_ms,
        };
        let mut state = session.state.lock().await;
        state.store_sealed_buffer(sealed);
    }
    let speaker_audio = super::speaker_id::SpeakerIdentifier::global(&session.models)
        .map(|identifier| (identifier, audio.clone()));
    let mut confidence = super::wire::TranscriptConfidence::default();
    let transcript = if let Some(cached) = cached_transcript {
        debug!(%item_id, len = cached.len(), "using speculative STT result");
        cached
    } else {
        let stt_result = match match session.models.whisper() {
            Ok(w) => {
                run_stt_full(
                    w,
                    crate::types::MonoF32At16k::new(audio),
                    session.stt_language(),
                )
                .await
            }
            Err(e) => Err(e),
        } {
            Ok(t) => t,
            Err(err) => {
                warn!(error = %err, %item_id, "STT failed; emitting transcription.failed");
                session
                    .emit(super::wire::OutboundEvent::TranscriptionFailed(
                        super::wire::TranscriptionFailedEvent {
                            item_id: item_id.clone().into(),
                            content_index: 0,
                            error: json!({
                                "code": errcode::STT_FAILED,
                                "message": err.to_string(),
                            }),
                        },
                    ))
                    .await;
                session.mark_user_item_incomplete(&item_id).await;
                let mut state = session.state.lock().await;
                state.drop_sealed_buffer(&item_id);
                return Ok(());
            }
        };
        let thresholds = session.noise_gate_thresholds();
        if let Some(rejection) = crate::stt::noise_gate::evaluate(
            stt_result.no_speech_prob,
            stt_result.avg_logprob,
            audio_ms as u32,
            thresholds,
        ) {
            info!(
                %item_id,
                reason = rejection.as_str(),
                avg_logprob = ?stt_result.avg_logprob,
                no_speech_prob = ?stt_result.no_speech_prob,
                duration_ms = audio_ms,
                "noise gate rejected transcript"
            );
            let mut state = session.state.lock().await;
            state.drop_sealed_buffer(&item_id);
            return Ok(());
        }
        confidence = super::wire::TranscriptConfidence::from_stt(&stt_result);
        stt_result.text
    };
    if transcript.is_empty() {
        debug!("empty transcript (silence gate); dropping utterance");
        let mut state = session.state.lock().await;
        state.drop_sealed_buffer(&item_id);
        return Ok(());
    }
    info!(
        transcript_chars = transcript.chars().count(),
        "transcription complete"
    );
    debug!(transcript = %transcript, "transcription complete (full)");

    session.inspector.emit(InspectorEvent::SttFinal {
        session_id: session.id.as_str().to_string(),
        item_id: item_id.clone(),
        text: transcript.clone(),
        audio_start_ms: 0,
        audio_end_ms: audio_ms,
    });
    session.inspector.emit(InspectorEvent::TurnUserCommitted {
        session_id: session.id.as_str().to_string(),
        item_id: item_id.clone(),
    });

    session
        .complete_user_item_transcript(&item_id, transcript.clone())
        .await;

    let speaker = match speaker_audio {
        Some((identifier, audio)) => {
            tokio::task::spawn_blocking(move || identifier.identify(&audio))
                .await
                .map_err(anyhow::Error::from)
                .and_then(|r| r)
                .unwrap_or_else(|err| {
                    warn!(error = %err, %item_id, "speaker identification failed");
                    None
                })
        }
        None => None,
    };
    if let Some(m) = &speaker {
        info!(%item_id, speaker = %m.id, name = ?m.name, score = m.score, "speaker identified");
    }
    let mut completed = json!({
        "type": "conversation.item.input_audio_transcription.completed",
        "item_id": item_id,
        "content_index": 0,
        "transcript": transcript,
        "speaker": speaker,
    });
    confidence.attach(&mut completed);
    session.send_to_client(&completed).await;
    {
        let mut state = session.state.lock().await;
        state.drop_sealed_buffer(&item_id);
    }

    if suppress_response {
        info!(
            %item_id,
            audio_ms,
            "backchannel: response suppressed (audio_ms < min_speech_for_response_ms)",
        );
        session
            .inspector
            .emit(InspectorEvent::BackchannelSuppressed {
                session_id: session.id.as_str().to_string(),
                item_id: item_id.clone(),
                audio_ms,
                transcript: Some(transcript),
            });
        return Ok(());
    }

    let Some(lane) = response else {
        debug!(%item_id, "committed turn has no response lane");
        return Ok(());
    };
    run_response(
        session,
        lane,
        transcript,
        predicted_llm_text.map(PrefilledText::Predicted),
        ResponseOverrides::default(),
    )
    .await
}

#[derive(Debug, Clone)]
pub(super) enum PrefilledText {
    Predicted(String),
    ClientSupplied(String),
}

impl PrefilledText {
    pub(super) fn text(&self) -> &str {
        match self {
            PrefilledText::Predicted(t) | PrefilledText::ClientSupplied(t) => t.as_str(),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub(super) struct ResponseOverrides {
    pub instructions: Option<String>,
    #[allow(dead_code)]
    pub modalities: Option<Vec<String>>,
}

#[tracing::instrument(
    target = "speaches/realtime",
    name = "llm.dispatch",
    skip_all,
    fields(session_id = %session.id.as_str(), response_id = %lane.id, item_id = %lane.assistant_item_id)
)]
pub(super) async fn run_response(
    session: &Arc<Session>,
    lane: ResponseLane,
    last_user_text: String,
    prefilled: Option<PrefilledText>,
    overrides: ResponseOverrides,
) -> anyhow::Result<()> {
    if !session.announce_response(&lane).await {
        return Ok(());
    }
    let ResponseLane {
        id,
        assistant_item_id,
        played_ms,
        transcript_so_far,
        ..
    } = lane;
    let response_id = id.as_str();
    let assistant_item_id = assistant_item_id.into_string();

    let client_supplied = matches!(prefilled, Some(PrefilledText::ClientSupplied(_)));
    let utterance_started = std::time::Instant::now();
    let mut llm_stream = if let Some(predicted) = prefilled.as_ref() {
        match predicted {
            PrefilledText::Predicted(t) => info!(
                chars = t.chars().count(),
                "promoting eager prediction: replaying buffered LLM output"
            ),
            PrefilledText::ClientSupplied(t) => info!(
                chars = t.chars().count(),
                "speaking client-supplied assistant text"
            ),
        }
        let (tx, rx) = mpsc::channel::<anyhow::Result<String>>(8);
        let predicted_text = predicted.text().to_string();
        tokio::spawn(async move {
            let _ = tx.send(Ok(predicted_text)).await;
        });
        rx
    } else {
        let Some(cfg) = session.llm_config_for_turn() else {
            warn!("conversation intent but CHAT_COMPLETION_BASE_URL not set");
            super::events::fail_response(
                session,
                response_id,
                &assistant_item_id,
                None,
                FailReason::LlmError,
                &played_ms,
            )
            .await;
            return Ok(());
        };
        let instructions = match overrides.instructions.as_deref() {
            Some(s) => Some(s.to_string()),
            None => session.instructions().await,
        };
        let messages = session.build_chat_messages(instructions.as_deref()).await;
        let messages = if messages
            .iter()
            .any(|m| m.role == "user" && m.content == last_user_text)
        {
            messages
        } else {
            let mut m = messages;
            m.push(ChatMessage {
                role: "user".into(),
                content: last_user_text.clone(),
            });
            m
        };
        let model_name = cfg.model.clone();
        session.inspector.emit(InspectorEvent::LlmRequest {
            session_id: session.id.as_str().to_string(),
            response_id: response_id.to_string(),
            model: model_name,
        });
        llm::complete_stream_messages(&cfg, messages, Some(response_id.to_string()))
    };

    let Some(kokoro) = &session.models.kokoro else {
        warn!("conversation intent but Kokoro TTS not loaded");
        super::events::emit_error(
            session,
            errcode::MODEL_LOAD_FAILED,
            "TTS model is not loaded; this response cannot produce audio",
            None,
            None,
        )
        .await;
        let mut full = String::new();
        while let Some(item) = llm_stream.recv().await {
            match item {
                Ok(d) => full.push_str(&d),
                Err(err) => {
                    warn!(error = %err, "LLM upstream failed; emitting response.done(failed)");
                    super::events::fail_response(
                        session,
                        response_id,
                        &assistant_item_id,
                        None,
                        FailReason::LlmError,
                        &played_ms,
                    )
                    .await;
                    return Ok(());
                }
            }
        }
        super::events::fail_response(
            session,
            response_id,
            &assistant_item_id,
            Some(full),
            FailReason::TtsError,
            &played_ms,
        )
        .await;
        return Ok(());
    };

    let pacer: Option<AudioPacer> = session.outbound_audio.as_ref().map(|spec| match spec {
        OutboundAudioSpec::Webrtc(track) => AudioPacer::Webrtc(audio_out::OutboundPacer::start(
            track.clone(),
            played_ms.clone(),
            session.outbound_queue_cap_ms,
        )),
        OutboundAudioSpec::WebSocket { ws_send, format } => {
            AudioPacer::WebSocket(WsAudioPacer::start(
                ws_send.clone(),
                session.event_seq.clone(),
                played_ms.clone(),
                format,
                response_id,
                &assistant_item_id,
            ))
        }
    });
    if pacer.is_none() {
        warn!("conversation intent but no outbound audio sink");
    }

    let (sentence_tx, mut sentence_rx) = mpsc::channel::<String>(64);
    let (voice, language) = session.tts_settings();
    let tts_speed = session.tts_speed();
    let kokoro_clone = kokoro.clone();
    let audio_store = session.audio_store.clone();
    let tts_inspector = session.inspector.clone();
    let tts_session_id = session.id.as_str().to_string();
    let tts_response_id = response_id.to_string();
    let tts_voice = voice.clone();
    let tts_body = async move {
        let mut pacer = pacer;
        let mut first_audio_logged = false;
        let mut planned_ms: u64 = 0;
        let mut chunk_idx: u32 = 0;
        while let Some(sentence) = sentence_rx.recv().await {
            tts_inspector.emit(InspectorEvent::TtsPhraseStart {
                session_id: tts_session_id.clone(),
                response_id: tts_response_id.clone(),
                text: sentence.clone(),
                voice: tts_voice.clone().unwrap_or_default(),
            });
            match synthesize_and_play(
                &kokoro_clone,
                &sentence,
                voice.clone(),
                language,
                tts_speed,
                pacer.as_mut(),
                utterance_started,
                &mut first_audio_logged,
                Some(audio_store.as_ref()),
            )
            .await
            {
                SynthesizePlayOutcome::Played { duration_ms } => {
                    tts_inspector.emit(InspectorEvent::TtsChunk {
                        session_id: tts_session_id.clone(),
                        response_id: tts_response_id.clone(),
                        chunk_idx,
                        ms_audio: duration_ms,
                        first: chunk_idx == 0,
                    });
                    chunk_idx += 1;
                    planned_ms += duration_ms;
                }
                SynthesizePlayOutcome::QueueFull { queued_ms, cap_ms } => {
                    return TtsWorkerOutcome::QueueFull {
                        pacer,
                        planned_ms,
                        queued_ms,
                        cap_ms,
                    };
                }
            }
        }
        TtsWorkerOutcome::Drained { pacer, planned_ms }
    };
    let tts_body = session.cancel.wrap(tts_body);
    let mut tts_worker = session.tts_abort.spawn(response_id, async move {
        tts_body.await.unwrap_or(TtsWorkerOutcome::Drained {
            pacer: None,
            planned_ms: 0,
        })
    });

    let mut chunker = llm::SentenceChunker::new();
    let mut full_response = String::new();
    let mut streaming_marked = false;
    let mut sentence_tx_opt = Some(sentence_tx);
    let mut llm_failed = false;
    let mut llm_first_token_emitted = false;
    let llm_start = std::time::Instant::now();

    let outcome: TtsWorkerOutcome = loop {
        tokio::select! {
            biased;

            res = &mut tts_worker => {
                break res.unwrap_or(TtsWorkerOutcome::Drained { pacer: None, planned_ms: 0 });
            }
            item = llm_stream.recv(), if sentence_tx_opt.is_some() => {
                let Some(item) = item else {

                    if let Some(tail) = chunker.take_flush() {
                        if !streaming_marked {
                            session.mark_streaming(response_id, &played_ms).await;
                        }
                        if let Some(tx) = sentence_tx_opt.as_ref() {
                            let _ = tx.send(tail).await;
                        }
                    }
                    sentence_tx_opt = None;
                    break (&mut tts_worker).await.unwrap_or(TtsWorkerOutcome::Drained { pacer: None, planned_ms: 0 });
                };
                let delta = match item {
                    Ok(d) => d,
                    Err(err) => {
                        warn!(error = %err, "LLM upstream failed mid-stream; emitting response.done(failed)");
                        sentence_tx_opt = None;
                        llm_failed = true;
                        break (&mut tts_worker).await.unwrap_or(TtsWorkerOutcome::Drained { pacer: None, planned_ms: 0 });
                    }
                };
                full_response.push_str(&delta);
                {
                    let mut t = transcript_so_far.lock().await;
                    t.push_str(&delta);
                }
                if !delta.is_empty() && !llm_first_token_emitted {
                    llm_first_token_emitted = true;
                    session.inspector.emit(InspectorEvent::LlmFirstToken {
                        session_id: session.id.as_str().to_string(),
                        response_id: response_id.to_string(),
                        elapsed_ms: llm_start.elapsed().as_millis() as u64,
                    });
                }
                if !delta.is_empty() {
                    if !streaming_marked {
                        session.mark_streaming(response_id, &played_ms).await;
                        streaming_marked = true;
                    }
                    super::events::emit_audio_transcript_delta(
                        session,
                        response_id,
                        &assistant_item_id,
                        &delta,
                    )
                    .await;
                }
                for sentence in chunker.feed(&delta) {
                    let Some(tx) = sentence_tx_opt.as_ref() else { break };
                    if tx.send(sentence).await.is_err() {

                        sentence_tx_opt = None;
                        break;
                    }
                }
            }
        }
    };

    drop(sentence_tx_opt);
    session.tts_abort.release(response_id);

    session.inspector.emit(InspectorEvent::LlmDone {
        session_id: session.id.as_str().to_string(),
        response_id: response_id.to_string(),
        reply_chars: full_response.len(),
        elapsed_ms: llm_start.elapsed().as_millis() as u64,
    });

    if llm_failed {
        let mut pacer = outcome.into_pacer();
        if let Some(p) = pacer.as_mut() {
            let _ = p.flush().await;
        }
        super::events::fail_response(
            session,
            response_id,
            &assistant_item_id,
            Some(full_response),
            FailReason::LlmError,
            &played_ms,
        )
        .await;
        return Ok(());
    }

    let (mut pacer, planned_ms) = match outcome {
        TtsWorkerOutcome::QueueFull {
            queued_ms, cap_ms, ..
        } => {
            handle_client_too_slow(
                session,
                response_id,
                &assistant_item_id,
                &mut llm_stream,
                full_response,
                queued_ms,
                cap_ms,
                played_ms.load(Ordering::Relaxed),
            )
            .await;
            return Ok(());
        }
        TtsWorkerOutcome::Drained { pacer, planned_ms } => (pacer, planned_ms),
    };

    if full_response.is_empty() {
        if let Some(p) = pacer.as_mut() {
            if let Err(err) = p.flush().await {
                warn!(error = %err, "outbound audio flush failed");
            }
        }
        super::events::fail_response(
            session,
            response_id,
            &assistant_item_id,
            None,
            FailReason::LlmError,
            &played_ms,
        )
        .await;
        return Ok(());
    }

    let drain_status =
        drain_pacer(session, response_id, pacer.as_mut(), &played_ms, planned_ms).await;

    info!(
        total_ms = utterance_started.elapsed().as_millis() as u64,
        planned_ms,
        played_ms = played_ms.load(Ordering::Relaxed),
        status = drain_status,
        "response complete",
    );

    let played = played_ms.load(Ordering::Relaxed);
    let close = match drain_status {
        "completed" => super::events::completed_response(
            response_id,
            &assistant_item_id,
            &full_response,
            played,
        ),
        _ => super::events::interrupted_response(
            response_id,
            &assistant_item_id,
            &full_response,
            ResponseStatus::Incomplete,
            ResponseStatusReason::DrainCap,
            played,
        ),
    };
    let assistant_item = (!client_supplied).then(|| {
        ConversationItem::new_assistant_audio(assistant_item_id.clone(), full_response, played)
    });
    session
        .close_response(response_id, assistant_item, close)
        .await;
    Ok(())
}

pub(super) async fn drain_pacer(
    session: &Session,
    response_id: &str,
    pacer: Option<&mut AudioPacer>,
    played_ms: &Arc<AtomicU64>,
    planned_ms: u64,
) -> &'static str {
    let Some(p) = pacer else {
        return "completed";
    };
    let played_now = played_ms.load(Ordering::Relaxed);
    if played_now >= planned_ms {
        if let Err(err) = p.flush().await {
            warn!(error = %err, "outbound audio flush failed");
        }
        return "completed";
    }
    let drain_cap_ms = (2 * planned_ms).clamp(
        defaults::response::DRAIN_CAP_FLOOR_MS,
        defaults::response::DRAIN_CAP_CEILING_MS,
    );
    session.inspector.emit(InspectorEvent::DrainStart {
        session_id: session.id.as_str().to_string(),
        response_id: response_id.to_string(),
        planned_ms,
    });
    let flush_fut = p.flush();
    let timeout_dur = std::time::Duration::from_millis(drain_cap_ms);
    let status: &'static str = match tokio::time::timeout(timeout_dur, flush_fut).await {
        Ok(Ok(())) => "completed",
        Ok(Err(err)) => {
            warn!(error = %err, "outbound audio flush failed during drain");
            "completed"
        }
        Err(_) => {
            warn!(
                planned_ms,
                played_ms = played_ms.load(Ordering::Relaxed),
                drain_cap_ms,
                "drain cap expired; terminating response as incomplete",
            );
            "incomplete"
        }
    };
    let played_after = played_ms.load(Ordering::Relaxed);
    session.inspector.emit(InspectorEvent::DrainComplete {
        session_id: session.id.as_str().to_string(),
        response_id: response_id.to_string(),
        played_ms: played_after,
        status,
    });
    status
}

const TTS_ABORT_QUIESCE_MS: u64 = 250;

struct TtsAbortEntry {
    response_id: String,
    abort: tokio::task::AbortHandle,
    quiesced: tokio::sync::oneshot::Receiver<std::convert::Infallible>,
}

#[derive(Default)]
pub(super) struct TtsAbort {
    slot: std::sync::Mutex<Option<TtsAbortEntry>>,
}

impl TtsAbort {
    pub(super) fn new() -> Self {
        Self::default()
    }

    pub(super) fn spawn<F>(&self, response_id: &str, fut: F) -> tokio::task::JoinHandle<F::Output>
    where
        F: std::future::Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let (quiesce_tx, quiesced) = tokio::sync::oneshot::channel::<std::convert::Infallible>();
        let handle = tokio::spawn(async move {
            let _quiesce_tx = quiesce_tx;
            fut.await
        });
        let entry = TtsAbortEntry {
            response_id: response_id.to_string(),
            abort: handle.abort_handle(),
            quiesced,
        };
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(prev) = slot.replace(entry) {
            warn!(
                stale_response_id = %prev.response_id,
                new_response_id = %response_id,
                "TTS abort slot still held by an earlier response; dropping the stale handle",
            );
        }
        handle
    }

    pub(super) fn release(&self, response_id: &str) {
        let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
        let matches = slot
            .as_ref()
            .map(|e| e.response_id == response_id)
            .unwrap_or(false);
        if matches {
            *slot = None;
        }
    }

    pub(super) async fn cancel(&self, response_id: &str) -> bool {
        let entry = {
            let mut slot = self.slot.lock().unwrap_or_else(|e| e.into_inner());
            match slot.as_ref() {
                Some(e) if e.response_id == response_id => slot.take(),
                _ => None,
            }
        };
        let Some(entry) = entry else {
            return false;
        };
        entry.abort.abort();
        let quiesced = tokio::time::timeout(
            std::time::Duration::from_millis(TTS_ABORT_QUIESCE_MS),
            entry.quiesced,
        )
        .await
        .is_ok();
        if !quiesced {
            warn!(
                %response_id,
                quiesce_ms = TTS_ABORT_QUIESCE_MS,
                "TTS worker did not quiesce within the abort window",
            );
        }
        true
    }
}

enum SynthesizePlayOutcome {
    Played { duration_ms: u64 },
    QueueFull { queued_ms: u64, cap_ms: u64 },
}

enum TtsWorkerOutcome {
    Drained {
        pacer: Option<AudioPacer>,
        planned_ms: u64,
    },
    QueueFull {
        pacer: Option<AudioPacer>,
        #[allow(dead_code)]
        planned_ms: u64,
        queued_ms: u64,
        cap_ms: u64,
    },
}

impl TtsWorkerOutcome {
    fn into_pacer(self) -> Option<AudioPacer> {
        match self {
            TtsWorkerOutcome::Drained { pacer, .. } => pacer,
            TtsWorkerOutcome::QueueFull { pacer, .. } => pacer,
        }
    }
}

async fn handle_client_too_slow(
    session: &Arc<Session>,
    response_id: &str,
    item_id: &str,
    llm_stream: &mut tokio::sync::mpsc::Receiver<anyhow::Result<String>>,
    transcript_so_far: String,
    queued_ms: u64,
    cap_ms: u64,
    audio_end_ms: u64,
) {
    session
        .inspector
        .emit(InspectorEvent::OutboundQueueExceeded {
            session_id: session.id.as_str().to_string(),
            response_id: response_id.to_string(),
            queued_ms,
            cap_ms,
        });
    let message = format!("outbound queue cap exceeded ({queued_ms} ms buffered)");
    super::events::emit_error(session, errcode::CLIENT_TOO_SLOW, &message, None, None).await;

    llm_stream.close();
    let transcript = if transcript_so_far.is_empty() {
        None
    } else {
        Some(transcript_so_far)
    };
    session
        .close_response(
            response_id,
            None,
            vec![super::events::failed_response(
                response_id,
                item_id,
                transcript,
                FailReason::ClientTooSlow,
                audio_end_ms,
            )],
        )
        .await;
}

#[tracing::instrument(
    target = "speaches/realtime",
    name = "tts.dispatch",
    skip_all,
    fields(chunk_chars = sentence.len(), voice = voice.as_deref().unwrap_or(""))
)]
async fn synthesize_and_play(
    kokoro: &crate::tts::KokoroHandle,
    sentence: &str,
    voice: Option<String>,
    language: &'static str,
    speed: f32,
    pacer: Option<&mut AudioPacer>,
    utterance_started: std::time::Instant,
    first_audio_logged: &mut bool,
    audio_store: Option<&AudioStore>,
) -> SynthesizePlayOutcome {
    let kokoro = kokoro.clone();
    let synth_text = sentence.to_string();
    let synth_started = std::time::Instant::now();
    let result = tokio::task::spawn_blocking(move || {
        kokoro.synthesize(&synth_text, voice.as_deref(), Some(language), speed)
    })
    .await;
    let audio_24k = match result {
        Ok(Ok(a)) => a,
        Ok(Err(err)) => {
            warn!(error = %err, sentence, "kokoro synth failed; skipping sentence");
            return SynthesizePlayOutcome::Played { duration_ms: 0 };
        }
        Err(err) => {
            warn!(error = %err, "kokoro synth join failed");
            return SynthesizePlayOutcome::Played { duration_ms: 0 };
        }
    };
    let duration_ms = audio_24k.duration_ms().raw();
    if let Some(store) = audio_store {
        store.append_tts_out_f32(audio_24k.samples());
    }
    let Some(pacer) = pacer else {
        return SynthesizePlayOutcome::Played { duration_ms };
    };
    if !*first_audio_logged {
        info!(
            first_audio_byte_ms = utterance_started.elapsed().as_millis() as u64,
            synth_ms = synth_started.elapsed().as_millis() as u64,
            sentence_chars = sentence.len(),
            "first audio frame about to write"
        );
        *first_audio_logged = true;
    }
    match pacer.play(audio_24k).await {
        Ok(()) => SynthesizePlayOutcome::Played { duration_ms },
        Err(audio_out::OutboundPushError::QueueFull { queued_ms, cap_ms }) => {
            warn!(
                queued_ms,
                cap_ms, "outbound queue cap exceeded; aborting response (client_too_slow)"
            );
            SynthesizePlayOutcome::QueueFull { queued_ms, cap_ms }
        }
        Err(err) => {
            warn!(error = %err, "outbound audio play failed");
            SynthesizePlayOutcome::Played { duration_ms }
        }
    }
}

#[tracing::instrument(
    target = "speaches/realtime",
    name = "stt.dispatch",
    skip_all,
    fields(samples = audio.len())
)]
pub(super) async fn run_stt(
    whisper: &WhisperHandle,
    audio: crate::types::MonoF32At16k,
    language: Option<String>,
) -> anyhow::Result<String> {
    Ok(run_stt_full(whisper, audio, language).await?.text)
}

pub(super) async fn run_stt_full(
    whisper: &WhisperHandle,
    audio: crate::types::MonoF32At16k,
    language: Option<String>,
) -> anyhow::Result<crate::stt::TranscriptionResult> {
    let whisper = whisper.clone();
    let samples = audio.into_vec();
    let opts = crate::stt::SttRequestOptions {
        language,
        initial_prompt: None,
    };
    let result = tokio::task::spawn_blocking(move || {
        whisper.transcribe_full_with_opts(&samples, crate::stt::WhisperTask::Transcribe, &opts)
    })
    .await??;
    Ok(result)
}

pub(super) fn promoted_turn_wire_id(
    promoted: bool,
    predicted: Option<crate::types::ResponseId>,
    fresh: crate::types::ResponseId,
) -> crate::types::ResponseId {
    match (promoted, predicted) {
        (true, Some(id)) => id,
        _ => fresh,
    }
}

#[cfg(test)]
mod wire_id_tests {
    use super::promoted_turn_wire_id;
    use crate::types::ResponseId;

    #[test]
    fn promoted_turns_keep_the_predicted_wire_id_and_rolled_back_turns_mint_a_fresh_one() {
        let predicted = ResponseId::new("resp_predicted");
        let fresh = ResponseId::new("resp_fresh");
        assert_eq!(
            promoted_turn_wire_id(true, Some(predicted.clone()), fresh.clone()),
            predicted,
        );
        {
            let predicted = ResponseId::new("resp_predicted");
            let fresh = ResponseId::new("resp_fresh");
            assert_eq!(
                promoted_turn_wire_id(false, Some(predicted), fresh.clone()),
                fresh,
            );
            assert_eq!(promoted_turn_wire_id(true, None, fresh.clone()), fresh);
        }
    }

}
