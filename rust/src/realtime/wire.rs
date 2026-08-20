#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};

use serde::Serialize;
use serde_json::Value;

use super::state::Topic;
use crate::types::{EventId, ItemId, ResponseId};

pub const EVENT_ID_FIELD: &str = "event_id";

pub fn format_event_id(n: u64) -> EventId {
    EventId::new(format!("evt_{n:024}"))
}

#[derive(Debug, Default)]
pub struct EventSeq {
    next: AtomicU64,
}

impl EventSeq {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn issued(&self) -> u64 {
        self.next.load(Ordering::SeqCst)
    }

    pub fn next_id(&self) -> EventId {
        format_event_id(self.next.fetch_add(1, Ordering::SeqCst))
    }

    pub fn stamp(&self, value: &mut Value) -> Option<EventId> {
        let obj = value.as_object_mut()?;
        let id = self.next_id();
        obj.insert(
            EVENT_ID_FIELD.to_string(),
            Value::String(id.as_str().to_string()),
        );
        Some(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Completed,
    Cancelled,
    Incomplete,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub enum ResponseStatusReason {
    #[serde(rename = "drain_cap")]
    DrainCap,
    #[serde(rename = "token_limit")]
    TokenLimit,
    #[serde(rename = "llm_error")]
    LlmError,
    #[serde(rename = "tts_error")]
    TtsError,
    #[serde(rename = "client_too_slow")]
    ClientTooSlow,
    #[serde(rename = "barge_in")]
    BargeIn,
    #[serde(rename = "client_cancelled")]
    ClientCancelled,
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseStatusDetails {
    pub reason: ResponseStatusReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub error: Option<ErrorPayload>,
}

fn realtime_response_object() -> &'static str {
    "realtime.response"
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponsePayload {
    pub id: ResponseId,
    #[serde(default = "realtime_response_object")]
    pub object: &'static str,
    pub status: ResponseStatus,
    pub audio_end_ms: u64,
    pub output: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub status_details: Option<ResponseStatusDetails>,
}

impl Default for ResponsePayload {
    fn default() -> Self {
        Self {
            id: ResponseId::new(""),
            object: "realtime.response",
            status: ResponseStatus::Completed,
            audio_end_ms: 0,
            output: Vec::new(),
            status_details: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ErrorPayload {
    #[serde(rename = "type")]
    pub type_: String,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub event_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub param: Option<String>,
}

impl ErrorPayload {
    pub fn for_code(code: impl Into<String>, message: impl Into<String>) -> Self {
        let code = code.into();
        let type_ = crate::errors::error_type_for(&code).to_string();
        Self {
            type_,
            code,
            message: message.into(),
            event_id: None,
            param: None,
        }
    }
}

#[derive(Debug, Clone, Serialize)]
#[cfg_attr(
    feature = "ts-bindings",
    derive(ts_rs::TS),
    ts(export, rename = "RealtimeOutboundEvent")
)]
#[serde(tag = "type")]
pub enum OutboundEvent {
    #[serde(rename = "session.created")]
    SessionCreated { session: Value },

    #[serde(rename = "session.updated")]
    SessionUpdated { session: Value },

    #[serde(rename = "session.done")]
    SessionDone { reason: String },

    #[serde(rename = "input_audio_buffer.speech_started")]
    SpeechStarted {
        item_id: ItemId,
        audio_start_ms: u64,
    },

    #[serde(rename = "input_audio_buffer.speech_stopped")]
    SpeechStopped { item_id: ItemId, audio_end_ms: u64 },

    #[serde(rename = "input_audio_buffer.committed")]
    BufferCommitted { item_id: ItemId },

    #[serde(rename = "input_audio_buffer.cleared")]
    BufferCleared,

    #[serde(rename = "input_audio_buffer.partial_transcription")]
    PartialTranscription {
        item_id: ItemId,
        transcript: String,
        audio_end_ms: u64,
    },

    #[serde(rename = "conversation.item.added")]
    ItemAdded { item: Value },

    #[serde(rename = "conversation.item.deleted")]
    ItemDeleted { item_id: ItemId },

    #[serde(rename = "conversation.item.truncated")]
    ItemTruncatedClientAck {
        item_id: ItemId,
        content_index: u64,
        audio_end_ms: u64,
    },

    #[serde(rename = "conversation.item.assistant_truncated")]
    AssistantTruncated {
        event_id: EventId,
        item_id: ItemId,
        audio_end_ms: u64,
        transcript: String,
    },

    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    TranscriptionCompleted {
        item_id: ItemId,
        content_index: u64,
        transcript: String,
    },

    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    TranscriptionDelta {
        item_id: ItemId,
        content_index: u64,
        delta: String,
    },

    #[serde(rename = "conversation.item.input_audio_transcription.failed")]
    TranscriptionFailed {
        item_id: ItemId,
        content_index: u64,
        error: Value,
    },

    #[serde(rename = "conversation.item.done")]
    ItemDone { item: Value },

    #[serde(rename = "conversation.item.retrieved")]
    ItemRetrieved { item: Value },

    #[serde(rename = "response.created")]
    ResponseCreated { response: Value },

    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded {
        response_id: ResponseId,
        output_index: u64,
        item: Value,
    },

    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone {
        response_id: ResponseId,
        output_index: u64,
        item: Value,
    },

    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        content_index: u64,
        part: Value,
    },

    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        content_index: u64,
        part: Value,
    },

    #[serde(rename = "response.output_audio_transcript.delta")]
    ResponseOutputAudioTranscriptDelta {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        content_index: u64,
        delta: String,
    },

    #[serde(rename = "response.output_audio_transcript.done")]
    ResponseOutputAudioTranscriptDone {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        content_index: u64,
        transcript: String,
    },

    #[serde(rename = "response.output_audio.delta")]
    ResponseOutputAudioDelta {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        content_index: u64,
        delta: String,
    },

    #[serde(rename = "response.output_audio.done")]
    ResponseOutputAudioDone {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        content_index: u64,
    },

    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        content_index: u64,
        delta: String,
    },

    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        content_index: u64,
        text: String,
    },

    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        call_id: String,
        delta: String,
    },

    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        call_id: String,
        arguments: String,
    },

    #[serde(rename = "response.tool_progress")]
    ResponseToolProgress {
        response_id: ResponseId,
        item_id: ItemId,
        output_index: u64,
        progress: Value,
    },

    #[serde(rename = "response.cancelled")]
    ResponseCancelled { response_id: ResponseId },

    #[serde(rename = "response.done")]
    ResponseDone { response: ResponsePayload },

    #[serde(rename = "output_audio_buffer.cleared")]
    OutputAudioBufferCleared,

    #[serde(rename = "output_audio_buffer.started")]
    OutputAudioBufferStarted { response_id: ResponseId },

    #[serde(rename = "output_audio_buffer.stopped")]
    OutputAudioBufferStopped { response_id: ResponseId },

    #[serde(rename = "rate_limits.updated")]
    RateLimitsUpdated { rate_limits: Value },

    #[serde(rename = "error")]
    Error { error: ErrorPayload },

    #[serde(rename = "conversation.item.diarization")]
    Diarization {
        item_id: ItemId,
        audio_end_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        elapsed_ms: Option<u64>,
        segments: Vec<Value>,
    },
}

impl OutboundEvent {
    pub fn topic(&self) -> Topic {
        Topic::classify(self.type_name())
    }

    pub fn to_wire_value(&self, seq: &EventSeq) -> Result<Value, serde_json::Error> {
        let mut value = serde_json::to_value(self)?;
        seq.stamp(&mut value);
        Ok(value)
    }

    pub fn type_name(&self) -> &'static str {
        use OutboundEvent::*;
        match self {
            SessionCreated { .. } => "session.created",
            SessionUpdated { .. } => "session.updated",
            SessionDone { .. } => "session.done",
            SpeechStarted { .. } => "input_audio_buffer.speech_started",
            SpeechStopped { .. } => "input_audio_buffer.speech_stopped",
            BufferCommitted { .. } => "input_audio_buffer.committed",
            BufferCleared => "input_audio_buffer.cleared",
            PartialTranscription { .. } => "input_audio_buffer.partial_transcription",
            ItemAdded { .. } => "conversation.item.added",
            ItemDeleted { .. } => "conversation.item.deleted",
            ItemTruncatedClientAck { .. } => "conversation.item.truncated",
            AssistantTruncated { .. } => "conversation.item.assistant_truncated",
            TranscriptionCompleted { .. } => {
                "conversation.item.input_audio_transcription.completed"
            }
            TranscriptionDelta { .. } => "conversation.item.input_audio_transcription.delta",
            TranscriptionFailed { .. } => "conversation.item.input_audio_transcription.failed",
            ItemDone { .. } => "conversation.item.done",
            ItemRetrieved { .. } => "conversation.item.retrieved",
            ResponseCreated { .. } => "response.created",
            ResponseOutputItemAdded { .. } => "response.output_item.added",
            ResponseOutputItemDone { .. } => "response.output_item.done",
            ResponseContentPartAdded { .. } => "response.content_part.added",
            ResponseContentPartDone { .. } => "response.content_part.done",
            ResponseOutputAudioTranscriptDelta { .. } => "response.output_audio_transcript.delta",
            ResponseOutputAudioTranscriptDone { .. } => "response.output_audio_transcript.done",
            ResponseOutputAudioDelta { .. } => "response.output_audio.delta",
            ResponseOutputAudioDone { .. } => "response.output_audio.done",
            ResponseOutputTextDelta { .. } => "response.output_text.delta",
            ResponseOutputTextDone { .. } => "response.output_text.done",
            ResponseFunctionCallArgumentsDelta { .. } => "response.function_call_arguments.delta",
            ResponseFunctionCallArgumentsDone { .. } => "response.function_call_arguments.done",
            ResponseToolProgress { .. } => "response.tool_progress",
            ResponseCancelled { .. } => "response.cancelled",
            ResponseDone { .. } => "response.done",
            OutputAudioBufferCleared => "output_audio_buffer.cleared",
            OutputAudioBufferStarted { .. } => "output_audio_buffer.started",
            OutputAudioBufferStopped { .. } => "output_audio_buffer.stopped",
            RateLimitsUpdated { .. } => "rate_limits.updated",
            Error { .. } => "error",
            Diarization { .. } => "conversation.item.diarization",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn type_name_matches_serde_tag() {
        let ev = OutboundEvent::ResponseOutputAudioDone {
            response_id: "r".into(),
            item_id: "i".into(),
            output_index: 0,
            content_index: 0,
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v.get("type").and_then(|t| t.as_str()), Some(ev.type_name()));
    }

    fn sample_response_payload(status: ResponseStatus, audio_end_ms: u64) -> ResponsePayload {
        ResponsePayload {
            id: ResponseId::new("resp_x"),
            object: "realtime.response",
            status,
            audio_end_ms,
            output: vec![],
            status_details: None,
        }
    }

    #[test]
    fn topic_classifies_response_variants() {
        let ev = OutboundEvent::ResponseDone {
            response: sample_response_payload(ResponseStatus::Completed, 0),
        };
        assert_eq!(ev.topic(), Topic::Response);
        let ev2 = OutboundEvent::SessionCreated {
            session: json!({"id": "s"}),
        };
        assert_eq!(ev2.topic(), Topic::Session);
        let ev3 = OutboundEvent::AssistantTruncated {
            event_id: "e".into(),
            item_id: "i".into(),
            audio_end_ms: 0,
            transcript: "".into(),
        };
        assert_eq!(ev3.topic(), Topic::Item);
    }

    #[test]
    fn response_done_audio_end_ms_present_for_every_status() {
        for (status, status_str) in [
            (ResponseStatus::Completed, "completed"),
            (ResponseStatus::Cancelled, "cancelled"),
            (ResponseStatus::Incomplete, "incomplete"),
            (ResponseStatus::Failed, "failed"),
        ] {
            let ev = OutboundEvent::ResponseDone {
                response: sample_response_payload(status, 1234),
            };
            let v = serde_json::to_value(&ev).unwrap();
            assert_eq!(v["type"].as_str(), Some("response.done"));
            assert_eq!(
                v["response"]["audio_end_ms"].as_u64(),
                Some(1234),
                "audio_end_ms missing for status={status_str}",
            );
            assert_eq!(
                v["response"]["status"].as_str(),
                Some(status_str),
                "wrong status",
            );
        }
    }

    #[test]
    fn response_done_omits_status_details_when_completed_or_cancelled() {
        let ev = OutboundEvent::ResponseDone {
            response: sample_response_payload(ResponseStatus::Completed, 0),
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert!(v["response"].get("status_details").is_none());
    }

    #[test]
    fn response_status_reason_serializes_snake_case() {
        let mut p = sample_response_payload(ResponseStatus::Incomplete, 5_000);
        p.status_details = Some(ResponseStatusDetails {
            reason: ResponseStatusReason::DrainCap,
            error: None,
        });
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v["status_details"]["reason"].as_str(), Some("drain_cap"));
    }

    #[test]
    fn error_payload_serializes_with_code_and_message_required() {
        let ev = OutboundEvent::Error {
            error: ErrorPayload {
                type_: "invalid_request_error".into(),
                code: "invalid_request_error".into(),
                message: "bad".into(),
                event_id: None,
                param: None,
            },
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"].as_str(), Some("error"));
        assert_eq!(v["error"]["type"].as_str(), Some("invalid_request_error"));
        assert_eq!(v["error"]["code"].as_str(), Some("invalid_request_error"));
        assert_eq!(v["error"]["message"].as_str(), Some("bad"));
        assert!(v["error"].get("event_id").is_none());
        assert!(v["error"].get("param").is_none());
    }

    #[test]
    fn error_payload_for_code_resolves_type_field() {
        let p = ErrorPayload::for_code("vad_failed", "boom");
        assert_eq!(p.type_, "server_error");
        let p2 = ErrorPayload::for_code("response_already_active", "nope");
        assert_eq!(p2.type_, "invalid_request_error");
    }

    #[test]
    fn typed_ids_serialize_as_strings_in_outbound_events() {
        let ev = OutboundEvent::ResponseOutputAudioDone {
            response_id: ResponseId::new("resp_1"),
            item_id: ItemId::new("item_a"),
            output_index: 0,
            content_index: 0,
        };
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["response_id"].as_str(), Some("resp_1"));
        assert_eq!(v["item_id"].as_str(), Some("item_a"));
        assert_eq!(v["type"].as_str(), Some("response.output_audio.done"));

        let trunc = OutboundEvent::AssistantTruncated {
            event_id: EventId::new("evt_1"),
            item_id: ItemId::new("item_a"),
            audio_end_ms: 1234,
            transcript: "partial".into(),
        };
        let tv = serde_json::to_value(&trunc).unwrap();
        assert_eq!(tv["event_id"].as_str(), Some("evt_1"));
        assert_eq!(tv["item_id"].as_str(), Some("item_a"));
        assert_eq!(tv["audio_end_ms"].as_u64(), Some(1234));
    }

    #[test]
    fn event_seq_is_strictly_increasing_and_prefixed() {
        let seq = EventSeq::new();
        let a = seq.next_id();
        let b = seq.next_id();
        assert_eq!(a.as_str(), "evt_000000000000000000000000");
        assert_eq!(b.as_str(), "evt_000000000000000000000001");
        assert!(a.as_str() < b.as_str());
        assert_eq!(seq.issued(), 2);
    }

    #[test]
    fn stamp_overwrites_any_caller_supplied_event_id() {
        let seq = EventSeq::new();
        let ev = OutboundEvent::AssistantTruncated {
            event_id: EventId::new("evt_caller_supplied"),
            item_id: "i".into(),
            audio_end_ms: 7,
            transcript: "x".into(),
        };
        let v = ev.to_wire_value(&seq).expect("wire value");
        assert_eq!(v["event_id"].as_str(), Some("evt_000000000000000000000000"));
        assert_eq!(v["audio_end_ms"].as_u64(), Some(7));
    }

    #[test]
    fn stamp_leaves_nested_error_event_id_alone() {
        let seq = EventSeq::new();
        let ev = OutboundEvent::Error {
            error: ErrorPayload {
                type_: "invalid_request_error".into(),
                code: "invalid_request_error".into(),
                message: "bad".into(),
                event_id: Some("evt_from_client".into()),
                param: None,
            },
        };
        let v = ev.to_wire_value(&seq).expect("wire value");
        assert_eq!(v["event_id"].as_str(), Some("evt_000000000000000000000000"));
        assert_eq!(v["error"]["event_id"].as_str(), Some("evt_from_client"));
    }

    #[test]
    fn to_wire_value_is_to_value_plus_stamp() {
        for ev in [
            OutboundEvent::BufferCleared,
            OutboundEvent::SessionDone { reason: "x".into() },
            OutboundEvent::ResponseCancelled {
                response_id: "r".into(),
            },
        ] {
            let via_helper = ev.to_wire_value(&EventSeq::new()).expect("wire value");
            let mut via_emit = serde_json::to_value(&ev).expect("serialize");
            EventSeq::new().stamp(&mut via_emit);
            assert_eq!(via_helper, via_emit, "drift for {}", ev.type_name());
        }
    }

    #[test]
    fn stamp_does_not_consume_a_number_for_non_objects() {
        let seq = EventSeq::new();
        let mut v = Value::String("not an event".into());
        assert!(seq.stamp(&mut v).is_none());
        assert_eq!(seq.issued(), 0);
    }

    #[test]
    fn type_name_for_each_variant_unique() {
        let names = [
            "session.created",
            "session.updated",
            "session.done",
            "input_audio_buffer.speech_started",
            "input_audio_buffer.speech_stopped",
            "input_audio_buffer.committed",
            "input_audio_buffer.cleared",
            "input_audio_buffer.partial_transcription",
            "conversation.item.added",
            "conversation.item.deleted",
            "conversation.item.truncated",
            "conversation.item.assistant_truncated",
            "conversation.item.input_audio_transcription.completed",
            "conversation.item.input_audio_transcription.delta",
            "conversation.item.input_audio_transcription.failed",
            "conversation.item.done",
            "conversation.item.retrieved",
            "response.created",
            "response.output_item.added",
            "response.output_item.done",
            "response.content_part.added",
            "response.content_part.done",
            "response.output_audio_transcript.delta",
            "response.output_audio_transcript.done",
            "response.output_audio.delta",
            "response.output_audio.done",
            "response.output_text.delta",
            "response.output_text.done",
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
            "response.tool_progress",
            "response.cancelled",
            "response.done",
            "output_audio_buffer.cleared",
            "output_audio_buffer.started",
            "output_audio_buffer.stopped",
            "rate_limits.updated",
            "error",
            "conversation.item.diarization",
        ];
        let mut seen = std::collections::HashSet::new();
        for n in names {
            assert!(seen.insert(n), "duplicate name {n}");
        }
    }
}
