use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde_json::json;
use tracing::warn;

use crate::errors;

use super::session::{CancelReason, CancelledSnapshot, FailReason, Session};
use super::state::{ConversationItem, ItemContent, ItemRole};

pub(super) async fn emit_response_open_brackets(
    session: &Session,
    response_id: &str,
    item_id: &str,
) {
    session
        .emit_event(json!({
            "type": "response.created",
            "response": { "id": response_id, "object": "realtime.response", "status": "in_progress" },
        }))
        .await;
    session
        .emit_event(json!({
            "type": "response.output_item.added",
            "response_id": response_id,
            "output_index": 0,
            "item": {
                "id": item_id,
                "object": "realtime.item",
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        }))
        .await;
    session
        .emit_event(json!({
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": { "type": "audio", "transcript": "" },
        }))
        .await;
}

pub(super) async fn fail_response(
    session: &Session,
    response_id: &str,
    transcript: Option<String>,
    reason: FailReason,
    played_ms: &Arc<AtomicU64>,
) {
    emit_response_done(
        session,
        response_id,
        "failed",
        transcript,
        Some(reason),
        played_ms.load(Ordering::Relaxed),
    )
    .await;
}

pub(super) async fn emit_response_brackets(
    session: &Session,
    response_id: &str,
    item_id: &str,
    transcript: &str,
    item_status: &'static str,
) {
    use super::wire::OutboundEvent::*;
    session
        .emit(ResponseOutputAudioTranscriptDone {
            response_id: response_id.into(),
            item_id: item_id.into(),
            output_index: 0,
            content_index: 0,
            transcript: transcript.to_string(),
        })
        .await;
    session
        .emit(ResponseOutputAudioDone {
            response_id: response_id.into(),
            item_id: item_id.into(),
            output_index: 0,
            content_index: 0,
        })
        .await;
    session
        .emit(ResponseContentPartDone {
            response_id: response_id.into(),
            item_id: item_id.into(),
            output_index: 0,
            content_index: 0,
            part: json!({ "type": "audio", "transcript": transcript }),
        })
        .await;
    session
        .emit(ResponseOutputItemDone {
            response_id: response_id.into(),
            output_index: 0,
            item: json!({
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": item_status,
                "content": [{ "type": "audio", "transcript": transcript }],
            }),
        })
        .await;
}

pub(super) async fn emit_bracket_close(
    session: &Session,
    response_id: &str,
    item_id: &str,
    transcript: &str,
) {
    emit_response_brackets(session, response_id, item_id, transcript, "completed").await;
}

pub(super) async fn emit_audio_transcript_delta(
    session: &Session,
    response_id: &str,
    item_id: &str,
    sentence: &str,
) {
    session
        .emit(
            super::wire::OutboundEvent::ResponseOutputAudioTranscriptDelta {
                response_id: response_id.into(),
                item_id: item_id.into(),
                output_index: 0,
                content_index: 0,
                delta: sentence.to_string(),
            },
        )
        .await;
}

pub(super) async fn emit_response_done(
    session: &Session,
    response_id: &str,
    status: &str,
    transcript: Option<String>,
    fail_reason: Option<FailReason>,
    audio_end_ms: u64,
) {
    use super::wire::{ResponsePayload, ResponseStatus, ResponseStatusDetails};
    let item_id_owned: String = {
        let state = session.state.lock().await;
        state
            .resp
            .item_id()
            .map(|i| i.as_str().to_string())
            .unwrap_or_else(|| session.id_source.item().as_str().to_string())
    };
    let mut output = json!({
        "id": item_id_owned,
        "type": "message",
        "role": "assistant",
    });
    if let Some(t) = transcript {
        output["content"] = json!([{ "type": "audio", "transcript": t }]);
    }

    let parsed_status = match status {
        "completed" => ResponseStatus::Completed,
        "cancelled" => ResponseStatus::Cancelled,
        "incomplete" => ResponseStatus::Incomplete,
        "failed" => ResponseStatus::Failed,
        other => {
            warn!(
                status = other,
                "unknown response status; defaulting to failed"
            );
            ResponseStatus::Failed
        }
    };

    let status_details = match parsed_status {
        ResponseStatus::Failed => Some(ResponseStatusDetails {
            reason: fail_reason.unwrap_or(FailReason::LlmError).into(),
            error: None,
        }),
        _ => None,
    };

    let payload = ResponsePayload {
        id: crate::types::ResponseId::new(response_id),
        object: "realtime.response",
        status: parsed_status,
        audio_end_ms,
        output: vec![output],
        status_details,
    };
    session
        .emit(super::wire::OutboundEvent::ResponseDone { response: payload })
        .await;
}

pub(super) async fn emit_error(
    session: &Session,
    code: &str,
    message: &str,
    event_id: Option<&str>,
    param: Option<&str>,
) {
    errors::debug_assert_known_code(code);
    if session.event_sink().await.is_none() {
        return;
    }
    let payload = super::wire::ErrorPayload {
        type_: errors::error_type_for(code).to_string(),
        code: code.to_string(),
        message: message.to_string(),
        event_id: event_id.map(|s| s.to_string()),
        param: param.map(|s| s.to_string()),
    };
    session
        .emit(super::wire::OutboundEvent::Error { error: payload })
        .await;
}

pub(super) async fn emit_incomplete_brackets(
    session: &Session,
    response_id: &str,
    item_id: &str,
    transcript: &str,
    played_ms: u64,
) {
    use super::wire::{
        ResponsePayload, ResponseStatus, ResponseStatusDetails, ResponseStatusReason,
    };
    emit_response_brackets(session, response_id, item_id, transcript, "incomplete").await;
    let payload = ResponsePayload {
        id: crate::types::ResponseId::new(response_id),
        object: "realtime.response",
        status: ResponseStatus::Incomplete,
        audio_end_ms: played_ms,
        output: vec![assistant_audio_item_json(item_id, transcript, "incomplete")],
        status_details: Some(ResponseStatusDetails {
            reason: ResponseStatusReason::DrainCap,
            error: None,
        }),
    };
    session
        .emit(super::wire::OutboundEvent::ResponseDone { response: payload })
        .await;
}

pub(super) async fn emit_cancelled_brackets(
    session: &Session,
    snap: &CancelledSnapshot,
    reason: CancelReason,
) {
    use super::wire::{
        ResponsePayload, ResponseStatus, ResponseStatusDetails, ResponseStatusReason,
    };
    emit_response_brackets(
        session,
        snap.response_id.as_str(),
        snap.assistant_item_id.as_str(),
        &snap.transcript,
        "incomplete",
    )
    .await;
    let cancel_reason = match reason {
        CancelReason::ClientCancelled => ResponseStatusReason::ClientCancelled,
        CancelReason::BargeIn => ResponseStatusReason::BargeIn,
    };
    let payload = ResponsePayload {
        id: snap.response_id.clone(),
        object: "realtime.response",
        status: ResponseStatus::Cancelled,
        audio_end_ms: snap.played_ms,
        output: vec![assistant_audio_item_json(
            snap.assistant_item_id.as_str(),
            &snap.transcript,
            "incomplete",
        )],
        status_details: Some(ResponseStatusDetails {
            reason: cancel_reason,
            error: None,
        }),
    };
    session
        .emit(super::wire::OutboundEvent::ResponseDone { response: payload })
        .await;
}

pub(super) fn assistant_audio_item_json(
    item_id: &str,
    transcript: &str,
    status: &'static str,
) -> serde_json::Value {
    json!({
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{ "type": "audio", "transcript": transcript }],
    })
}

pub(super) async fn emit_server_truncate(session: &Session, snap: &CancelledSnapshot) {
    if snap.played_ms == 0 {
        return;
    }
    session
        .send_to_client(&json!({
            "type": "conversation.item.assistant_truncated",
            "item_id": snap.assistant_item_id,
            "audio_end_ms": snap.played_ms,
            "transcript": snap.transcript,
        }))
        .await;
}

pub(super) fn extract_text_from_content(content: Option<&serde_json::Value>) -> Option<String> {
    let arr = content?.as_array()?;
    let mut text = String::new();
    for entry in arr {
        let part_type = entry.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match part_type {
            "input_text" | "text" | "output_text" => {
                if let Some(t) = entry.get("text").and_then(|v| v.as_str()) {
                    if !text.is_empty() {
                        text.push(' ');
                    }
                    text.push_str(t);
                }
            }
            _ => {}
        }
    }
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

pub(super) fn item_to_json(item: &ConversationItem) -> serde_json::Value {
    let content = match &item.content {
        ItemContent::UserAudio {
            transcript,
            audio_end_ms,
        } => {
            let mut obj = serde_json::Map::new();
            obj.insert("type".into(), json!("input_audio"));
            if let Some(t) = transcript {
                obj.insert("transcript".into(), json!(t));
            }
            if let Some(ms) = audio_end_ms {
                obj.insert("audio_end_ms".into(), json!(ms));
            }
            json!([serde_json::Value::Object(obj)])
        }
        ItemContent::AssistantAudio {
            transcript,
            audio_ms,
        } => json!([{
            "type": "audio",
            "transcript": transcript,
            "audio_ms": audio_ms,
        }]),
        ItemContent::Text(t) => {
            let part_type = if matches!(item.role, ItemRole::User) {
                "input_text"
            } else {
                "text"
            };
            json!([{ "type": part_type, "text": t }])
        }
    };
    json!({
        "id": item.id,
        "object": "realtime.item",
        "type": "message",
        "role": item.role.as_str(),
        "status": item.status.as_str(),
        "content": content,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_text_accepts_all_message_part_types() {
        assert_eq!(
            extract_text_from_content(Some(&json!([{"type":"output_text","text":"hi"}]))),
            Some("hi".to_string())
        );
        assert_eq!(
            extract_text_from_content(Some(&json!([{"type":"input_text","text":"hi"}]))),
            Some("hi".to_string())
        );
        assert_eq!(
            extract_text_from_content(Some(&json!([{"type":"text","text":"hi"}]))),
            Some("hi".to_string())
        );
        assert_eq!(
            extract_text_from_content(Some(&json!([
                {"type":"output_text","text":"one"},
                {"type":"text","text":"two"}
            ]))),
            Some("one two".to_string())
        );
        assert_eq!(
            extract_text_from_content(Some(&json!([{"type":"input_audio","transcript":"hi"}]))),
            None
        );
        assert_eq!(extract_text_from_content(None), None);
    }
}
