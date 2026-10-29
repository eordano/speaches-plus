use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use serde_json::json;

use crate::errors;

use super::session::{CancelReason, CancelledSnapshot, FailReason, Session};
use super::state::{ConversationItem, ItemContent, ItemRole};
use super::wire::{OutboundEvent, ResponseStatus, ResponseStatusReason};

pub(super) fn response_open_brackets(response_id: &str, item_id: &str) -> [serde_json::Value; 3] {
    [
        json!({
            "type": "response.created",
            "response": { "id": response_id, "object": "realtime.response", "status": "in_progress" },
        }),
        json!({
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
        }),
        json!({
            "type": "response.content_part.added",
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": { "type": "audio", "transcript": "" },
        }),
    ]
}

pub(super) async fn fail_response(
    session: &Arc<Session>,
    response_id: &str,
    item_id: &str,
    transcript: Option<String>,
    reason: FailReason,
    played_ms: &Arc<AtomicU64>,
) {
    session
        .close_response(
            response_id,
            None,
            vec![failed_response(
                response_id,
                item_id,
                transcript,
                reason,
                played_ms.load(Ordering::Relaxed),
            )],
        )
        .await;
}

fn response_brackets(
    response_id: &str,
    item_id: &str,
    transcript: &str,
    item_status: &'static str,
) -> Vec<OutboundEvent> {
    vec![
        OutboundEvent::ResponseOutputAudioTranscriptDone(
            super::wire::ResponseOutputAudioTranscriptDoneEvent {
                response_id: response_id.into(),
                item_id: item_id.into(),
                output_index: 0,
                content_index: 0,
                transcript: transcript.to_string(),
            },
        ),
        OutboundEvent::ResponseOutputAudioDone(super::wire::ResponseOutputAudioDoneEvent {
            response_id: response_id.into(),
            item_id: item_id.into(),
            output_index: 0,
            content_index: 0,
        }),
        OutboundEvent::ResponseContentPartDone(super::wire::ResponseContentPartDoneEvent {
            response_id: response_id.into(),
            item_id: item_id.into(),
            output_index: 0,
            content_index: 0,
            part: json!({ "type": "audio", "transcript": transcript }),
        }),
        OutboundEvent::ResponseOutputItemDone(super::wire::ResponseOutputItemDoneEvent {
            response_id: response_id.into(),
            output_index: 0,
            item: json!({
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": item_status,
                "content": [{ "type": "audio", "transcript": transcript }],
            }),
        }),
    ]
}

fn response_done(
    response_id: &str,
    status: ResponseStatus,
    reason: Option<ResponseStatusReason>,
    output: serde_json::Value,
    audio_end_ms: u64,
) -> OutboundEvent {
    OutboundEvent::ResponseDone(super::wire::ResponseDoneEvent {
        response: super::wire::ResponsePayload {
            id: crate::types::ResponseId::new(response_id),
            object: "realtime.response",
            status,
            audio_end_ms,
            output: vec![output],
            status_details: reason.map(|reason| super::wire::ResponseStatusDetails {
                reason,
                error: None,
            }),
        },
    })
}

fn done_output(item_id: &str, transcript: Option<String>) -> serde_json::Value {
    let mut output = json!({
        "id": item_id,
        "type": "message",
        "role": "assistant",
    });
    if let Some(t) = transcript {
        output["content"] = json!([{ "type": "audio", "transcript": t }]);
    }
    output
}

pub(super) fn failed_response(
    response_id: &str,
    item_id: &str,
    transcript: Option<String>,
    reason: FailReason,
    audio_end_ms: u64,
) -> OutboundEvent {
    response_done(
        response_id,
        ResponseStatus::Failed,
        Some(reason.into()),
        done_output(item_id, transcript),
        audio_end_ms,
    )
}

pub(super) fn completed_response(
    response_id: &str,
    item_id: &str,
    transcript: &str,
    audio_end_ms: u64,
) -> Vec<OutboundEvent> {
    let mut events = response_brackets(response_id, item_id, transcript, "completed");
    events.push(response_done(
        response_id,
        ResponseStatus::Completed,
        None,
        done_output(item_id, Some(transcript.to_string())),
        audio_end_ms,
    ));
    events
}

pub(super) fn interrupted_response(
    response_id: &str,
    item_id: &str,
    transcript: &str,
    status: ResponseStatus,
    reason: ResponseStatusReason,
    audio_end_ms: u64,
) -> Vec<OutboundEvent> {
    let mut events = response_brackets(response_id, item_id, transcript, "incomplete");
    events.push(response_done(
        response_id,
        status,
        Some(reason),
        assistant_audio_item_json(item_id, transcript, "incomplete"),
        audio_end_ms,
    ));
    events
}

pub(super) async fn emit_audio_transcript_delta(
    session: &Session,
    response_id: &str,
    item_id: &str,
    sentence: &str,
) {
    session
        .emit(
            super::wire::OutboundEvent::ResponseOutputAudioTranscriptDelta(
                super::wire::ResponseOutputAudioTranscriptDeltaEvent {
                    response_id: response_id.into(),
                    item_id: item_id.into(),
                    output_index: 0,
                    content_index: 0,
                    delta: sentence.to_string(),
                },
            ),
        )
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
        .emit(super::wire::OutboundEvent::Error(super::wire::ErrorEvent {
            error: payload,
        }))
        .await;
}

pub(super) async fn emit_cancelled_brackets(
    session: &Session,
    snap: &CancelledSnapshot,
    reason: CancelReason,
) {
    let reason = match reason {
        CancelReason::ClientCancelled => ResponseStatusReason::ClientCancelled,
        CancelReason::BargeIn => ResponseStatusReason::BargeIn,
    };
    for ev in interrupted_response(
        snap.response_id.as_str(),
        snap.assistant_item_id.as_str(),
        &snap.transcript,
        ResponseStatus::Cancelled,
        reason,
        snap.played_ms,
    ) {
        session.emit(ev).await;
    }
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
