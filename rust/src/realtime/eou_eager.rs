use std::sync::Arc;

use crate::conversation::llm::ChatMessage;
use crate::eou::EouConfig;

use super::inspector::InspectorEvent;
use super::session::Session;
use super::state::RespPhase;
use super::Intent;

pub(super) async fn try_eager_dispatch(
    session: &Arc<Session>,
    cfg: &EouConfig,
    score: f32,
    audio: Vec<f32>,
) {
    {
        let mut last = session.last_eager_dispatch_at.lock().await;
        let now = std::time::Instant::now();
        if let Some(prev) = *last {
            let throttle = std::time::Duration::from_millis(cfg.eager_interval_ms.max(1) as u64);
            if now.duration_since(prev) < throttle {
                session.inspector.emit(InspectorEvent::PredictedSuppressed {
                    session_id: session.id.as_str().to_string(),
                    score,
                    inflight: 1,
                });
                return;
            }
        }
        *last = Some(now);
    }

    let predicted_id = session.id_source.response();
    let predicted_item_id = session.id_source.item();
    let Ok(whisper) = session.models.whisper() else {
        tracing::error!("eager end-of-turn STT requested but speech-to-text is unavailable");
        return;
    };
    let stt_runner = super::eou_predicted::spawn_predicted_stt(
        &session.cancel,
        whisper.clone(),
        crate::types::MonoF32At16k::new(audio),
    );
    let llm_runner_opt = if session.intent == Intent::Conversation {
        if let Some(llm_cfg) = session.llm_config.clone() {
            let instructions = session.instructions().await;
            let mut messages = session.build_chat_messages(instructions.as_deref()).await;
            let placeholder = format!(
                "[predicted-end-of-turn p={:.2}; partial transcription pending]",
                score
            );
            messages.push(ChatMessage {
                role: "user".into(),
                content: placeholder,
            });
            Some(super::eou_predicted::spawn_predicted_llm(
                &session.cancel,
                llm_cfg,
                messages,
                cfg.predicted_token_buffer_cap,
            ))
        } else {
            None
        }
    } else {
        None
    };

    let llm_handle_opt = llm_runner_opt.map(super::state::PredictedLlmRunnerHandle::from_runner);

    let started_ok = {
        let mut state = session.state.lock().await;
        if matches!(state.resp, RespPhase::None) {
            let started = state.resp_start_predicted_with_llm(
                predicted_id.clone(),
                predicted_item_id.clone(),
                score,
                Some(stt_runner),
                llm_handle_opt,
            );
            match started {
                Ok(epoch) => {
                    session.inspector.emit(InspectorEvent::PredictedPromoted {
                        session_id: session.id.as_str().to_string(),
                        response_id: predicted_id.as_str().to_string(),
                        score,
                    });
                    session.inspector.emit(InspectorEvent::EouEagerDispatch {
                        session_id: session.id.as_str().to_string(),
                        response_id: predicted_id.as_str().to_string(),
                        item_id: predicted_item_id.as_str().to_string(),
                        score,
                        threshold: cfg.eager_p_threshold,
                        epoch: epoch.raw(),
                    });
                    true
                }
                Err(_) => false,
            }
        } else {
            session.inspector.emit(InspectorEvent::PredictedSuppressed {
                session_id: session.id.as_str().to_string(),
                score,
                inflight: 1,
            });
            false
        }
    };
    if !started_ok {
        let mut last = session.last_eager_dispatch_at.lock().await;
        *last = None;
    }
}
