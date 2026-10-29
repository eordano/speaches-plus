
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

#[typeshare::typeshare]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(rename_all = "snake_case")]
pub enum ResponseStatus {
    Completed,
    Cancelled,
    Incomplete,
    Failed,
}

#[typeshare::typeshare]
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

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseStatusDetails {
    pub reason: ResponseStatusReason,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub error: Option<ErrorPayload>,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponsePayload {
    pub id: ResponseId,
    #[typeshare(skip)]
    pub object: &'static str,
    pub status: ResponseStatus,
    #[typeshare(serialized_as = "U53")]
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

#[typeshare::typeshare]
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

#[typeshare::typeshare]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    Running,
    Done,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ToolProgressEntry {
    pub name: String,
    pub status: ToolStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub args: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub error: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    pub summary: Option<String>,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct SessionCreatedEvent {
    pub session: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct SessionUpdatedEvent {
    pub session: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct SessionDoneEvent {
    pub reason: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct SpeechStartedEvent {
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub audio_start_ms: u64,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct SpeechStoppedEvent {
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub audio_end_ms: u64,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct BufferCommittedEvent {
    pub item_id: ItemId,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct PartialTranscriptionEvent {
    pub item_id: ItemId,
    pub transcript: String,
    #[typeshare(serialized_as = "U53")]
    pub audio_end_ms: u64,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ItemAddedEvent {
    pub item: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ItemDeletedEvent {
    pub item_id: ItemId,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ItemTruncatedClientAckEvent {
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    #[typeshare(serialized_as = "U53")]
    pub audio_end_ms: u64,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct AssistantTruncatedEvent {
    pub event_id: EventId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub audio_end_ms: u64,
    pub transcript: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct TranscriptionCompletedEvent {
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub transcript: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_logprob: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_speech_prob: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<TranscriptTokenLogprob>>,
}

impl TranscriptionCompletedEvent {
    pub fn with_confidence(mut self, confidence: TranscriptConfidence) -> Self {
        self.avg_logprob = confidence.avg_logprob;
        self.no_speech_prob = confidence.no_speech_prob;
        self.compression_ratio = confidence.compression_ratio;
        self.logprobs = confidence.logprobs;
        self
    }
}

#[typeshare::typeshare]
#[derive(Debug, Clone, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct TranscriptTokenLogprob {
    pub token: String,
    pub logprob: f32,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct TranscriptConfidence {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_logprob: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_speech_prob: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub compression_ratio: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Vec<TranscriptTokenLogprob>>,
}

impl TranscriptConfidence {
    pub fn from_stt(result: &crate::stt::TranscriptionResult) -> Self {
        Self {
            avg_logprob: result.avg_logprob,
            no_speech_prob: result.no_speech_prob,
            compression_ratio: result.compression_ratio,
            logprobs: (!result.tokens.is_empty()).then(|| {
                result
                    .tokens
                    .iter()
                    .map(|t| TranscriptTokenLogprob {
                        token: t.token.clone(),
                        logprob: t.logprob,
                    })
                    .collect()
            }),
        }
    }

    pub fn attach(&self, event: &mut Value) {
        if let (Value::Object(target), Value::Object(fields)) =
            (event, serde_json::to_value(self).unwrap_or(Value::Null))
        {
            target.extend(fields);
        }
    }
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct TranscriptionDeltaEvent {
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub delta: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct TranscriptionFailedEvent {
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub error: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ItemDoneEvent {
    pub item: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ItemRetrievedEvent {
    pub item: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseCreatedEvent {
    pub response: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputItemAddedEvent {
    pub response_id: ResponseId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    pub item: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputItemDoneEvent {
    pub response_id: ResponseId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    pub item: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseContentPartAddedEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub part: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseContentPartDoneEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub part: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputAudioTranscriptDeltaEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub delta: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputAudioTranscriptDoneEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub transcript: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputAudioDeltaEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub delta: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputAudioDoneEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputTextDeltaEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub delta: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseOutputTextDoneEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    #[typeshare(serialized_as = "U53")]
    pub content_index: u64,
    pub text: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseFunctionCallArgumentsDeltaEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    pub call_id: String,
    pub delta: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseFunctionCallArgumentsDoneEvent {
    pub response_id: ResponseId,
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub output_index: u64,
    pub call_id: String,
    pub arguments: String,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseToolProgressEvent {
    pub response_id: ResponseId,
    pub tools: Vec<ToolProgressEntry>,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseCancelledEvent {
    pub response_id: ResponseId,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ResponseDoneEvent {
    pub response: ResponsePayload,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct OutputAudioBufferStartedEvent {
    pub response_id: ResponseId,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct OutputAudioBufferStoppedEvent {
    pub response_id: ResponseId,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct RateLimitsUpdatedEvent {
    pub rate_limits: Value,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct ErrorEvent {
    pub error: ErrorPayload,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub struct DiarizationEvent {
    pub item_id: ItemId,
    #[typeshare(serialized_as = "U53")]
    pub audio_end_ms: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "ts-bindings", ts(optional))]
    #[typeshare(serialized_as = "Option<U53>")]
    pub elapsed_ms: Option<u64>,
    pub segments: Vec<Value>,
}

#[typeshare::typeshare]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[cfg_attr(feature = "ts-bindings", derive(ts_rs::TS), ts(export))]
pub enum RealtimeServerEventType {
    #[serde(rename = "session.created")]
    SessionCreated,
    #[serde(rename = "session.updated")]
    SessionUpdated,
    #[serde(rename = "session.done")]
    SessionDone,
    #[serde(rename = "input_audio_buffer.speech_started")]
    SpeechStarted,
    #[serde(rename = "input_audio_buffer.speech_stopped")]
    SpeechStopped,
    #[serde(rename = "input_audio_buffer.committed")]
    BufferCommitted,
    #[serde(rename = "input_audio_buffer.cleared")]
    BufferCleared,
    #[serde(rename = "input_audio_buffer.partial_transcription")]
    PartialTranscription,
    #[serde(rename = "conversation.item.added")]
    ItemAdded,
    #[serde(rename = "conversation.item.deleted")]
    ItemDeleted,
    #[serde(rename = "conversation.item.truncated")]
    ItemTruncatedClientAck,
    #[serde(rename = "conversation.item.assistant_truncated")]
    AssistantTruncated,
    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    TranscriptionCompleted,
    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    TranscriptionDelta,
    #[serde(rename = "conversation.item.input_audio_transcription.failed")]
    TranscriptionFailed,
    #[serde(rename = "conversation.item.done")]
    ItemDone,
    #[serde(rename = "conversation.item.retrieved")]
    ItemRetrieved,
    #[serde(rename = "response.created")]
    ResponseCreated,
    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded,
    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone,
    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded,
    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone,
    #[serde(rename = "response.output_audio_transcript.delta")]
    ResponseOutputAudioTranscriptDelta,
    #[serde(rename = "response.output_audio_transcript.done")]
    ResponseOutputAudioTranscriptDone,
    #[serde(rename = "response.output_audio.delta")]
    ResponseOutputAudioDelta,
    #[serde(rename = "response.output_audio.done")]
    ResponseOutputAudioDone,
    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta,
    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone,
    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta,
    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone,
    #[serde(rename = "response.tool_progress")]
    ResponseToolProgress,
    #[serde(rename = "response.cancelled")]
    ResponseCancelled,
    #[serde(rename = "response.done")]
    ResponseDone,
    #[serde(rename = "output_audio_buffer.cleared")]
    OutputAudioBufferCleared,
    #[serde(rename = "output_audio_buffer.started")]
    OutputAudioBufferStarted,
    #[serde(rename = "output_audio_buffer.stopped")]
    OutputAudioBufferStopped,
    #[serde(rename = "rate_limits.updated")]
    RateLimitsUpdated,
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "conversation.item.diarization")]
    Diarization,
}

impl RealtimeServerEventType {
    pub const ALL: [RealtimeServerEventType; 39] = {
        use RealtimeServerEventType::*;
        [
            SessionCreated,
            SessionUpdated,
            SessionDone,
            SpeechStarted,
            SpeechStopped,
            BufferCommitted,
            BufferCleared,
            PartialTranscription,
            ItemAdded,
            ItemDeleted,
            ItemTruncatedClientAck,
            AssistantTruncated,
            TranscriptionCompleted,
            TranscriptionDelta,
            TranscriptionFailed,
            ItemDone,
            ItemRetrieved,
            ResponseCreated,
            ResponseOutputItemAdded,
            ResponseOutputItemDone,
            ResponseContentPartAdded,
            ResponseContentPartDone,
            ResponseOutputAudioTranscriptDelta,
            ResponseOutputAudioTranscriptDone,
            ResponseOutputAudioDelta,
            ResponseOutputAudioDone,
            ResponseOutputTextDelta,
            ResponseOutputTextDone,
            ResponseFunctionCallArgumentsDelta,
            ResponseFunctionCallArgumentsDone,
            ResponseToolProgress,
            ResponseCancelled,
            ResponseDone,
            OutputAudioBufferCleared,
            OutputAudioBufferStarted,
            OutputAudioBufferStopped,
            RateLimitsUpdated,
            Error,
            Diarization,
        ]
    };

    pub fn wire_name(&self) -> String {
        serde_json::to_value(self)
            .expect("unit enum serializes")
            .as_str()
            .expect("unit enum serializes to a string")
            .to_string()
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
    SessionCreated(SessionCreatedEvent),

    #[serde(rename = "session.updated")]
    SessionUpdated(SessionUpdatedEvent),

    #[serde(rename = "session.done")]
    SessionDone(SessionDoneEvent),

    #[serde(rename = "input_audio_buffer.speech_started")]
    SpeechStarted(SpeechStartedEvent),

    #[serde(rename = "input_audio_buffer.speech_stopped")]
    SpeechStopped(SpeechStoppedEvent),

    #[serde(rename = "input_audio_buffer.committed")]
    BufferCommitted(BufferCommittedEvent),

    #[serde(rename = "input_audio_buffer.cleared")]
    BufferCleared,

    #[serde(rename = "input_audio_buffer.partial_transcription")]
    PartialTranscription(PartialTranscriptionEvent),

    #[serde(rename = "conversation.item.added")]
    ItemAdded(ItemAddedEvent),

    #[serde(rename = "conversation.item.deleted")]
    ItemDeleted(ItemDeletedEvent),

    #[serde(rename = "conversation.item.truncated")]
    ItemTruncatedClientAck(ItemTruncatedClientAckEvent),

    #[serde(rename = "conversation.item.assistant_truncated")]
    AssistantTruncated(AssistantTruncatedEvent),

    #[serde(rename = "conversation.item.input_audio_transcription.completed")]
    TranscriptionCompleted(TranscriptionCompletedEvent),

    #[serde(rename = "conversation.item.input_audio_transcription.delta")]
    TranscriptionDelta(TranscriptionDeltaEvent),

    #[serde(rename = "conversation.item.input_audio_transcription.failed")]
    TranscriptionFailed(TranscriptionFailedEvent),

    #[serde(rename = "conversation.item.done")]
    ItemDone(ItemDoneEvent),

    #[serde(rename = "conversation.item.retrieved")]
    ItemRetrieved(ItemRetrievedEvent),

    #[serde(rename = "response.created")]
    ResponseCreated(ResponseCreatedEvent),

    #[serde(rename = "response.output_item.added")]
    ResponseOutputItemAdded(ResponseOutputItemAddedEvent),

    #[serde(rename = "response.output_item.done")]
    ResponseOutputItemDone(ResponseOutputItemDoneEvent),

    #[serde(rename = "response.content_part.added")]
    ResponseContentPartAdded(ResponseContentPartAddedEvent),

    #[serde(rename = "response.content_part.done")]
    ResponseContentPartDone(ResponseContentPartDoneEvent),

    #[serde(rename = "response.output_audio_transcript.delta")]
    ResponseOutputAudioTranscriptDelta(ResponseOutputAudioTranscriptDeltaEvent),

    #[serde(rename = "response.output_audio_transcript.done")]
    ResponseOutputAudioTranscriptDone(ResponseOutputAudioTranscriptDoneEvent),

    #[serde(rename = "response.output_audio.delta")]
    ResponseOutputAudioDelta(ResponseOutputAudioDeltaEvent),

    #[serde(rename = "response.output_audio.done")]
    ResponseOutputAudioDone(ResponseOutputAudioDoneEvent),

    #[serde(rename = "response.output_text.delta")]
    ResponseOutputTextDelta(ResponseOutputTextDeltaEvent),

    #[serde(rename = "response.output_text.done")]
    ResponseOutputTextDone(ResponseOutputTextDoneEvent),

    #[serde(rename = "response.function_call_arguments.delta")]
    ResponseFunctionCallArgumentsDelta(ResponseFunctionCallArgumentsDeltaEvent),

    #[serde(rename = "response.function_call_arguments.done")]
    ResponseFunctionCallArgumentsDone(ResponseFunctionCallArgumentsDoneEvent),

    #[serde(rename = "response.tool_progress")]
    ResponseToolProgress(ResponseToolProgressEvent),

    #[serde(rename = "response.cancelled")]
    ResponseCancelled(ResponseCancelledEvent),

    #[serde(rename = "response.done")]
    ResponseDone(ResponseDoneEvent),

    #[serde(rename = "output_audio_buffer.cleared")]
    OutputAudioBufferCleared,

    #[serde(rename = "output_audio_buffer.started")]
    OutputAudioBufferStarted(OutputAudioBufferStartedEvent),

    #[serde(rename = "output_audio_buffer.stopped")]
    OutputAudioBufferStopped(OutputAudioBufferStoppedEvent),

    #[serde(rename = "rate_limits.updated")]
    RateLimitsUpdated(RateLimitsUpdatedEvent),

    #[serde(rename = "error")]
    Error(ErrorEvent),

    #[serde(rename = "conversation.item.diarization")]
    Diarization(DiarizationEvent),
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

    pub fn event_type(&self) -> RealtimeServerEventType {
        use OutboundEvent::*;
        match self {
            SessionCreated(..) => RealtimeServerEventType::SessionCreated,
            SessionUpdated(..) => RealtimeServerEventType::SessionUpdated,
            SessionDone(..) => RealtimeServerEventType::SessionDone,
            SpeechStarted(..) => RealtimeServerEventType::SpeechStarted,
            SpeechStopped(..) => RealtimeServerEventType::SpeechStopped,
            BufferCommitted(..) => RealtimeServerEventType::BufferCommitted,
            BufferCleared => RealtimeServerEventType::BufferCleared,
            PartialTranscription(..) => RealtimeServerEventType::PartialTranscription,
            ItemAdded(..) => RealtimeServerEventType::ItemAdded,
            ItemDeleted(..) => RealtimeServerEventType::ItemDeleted,
            ItemTruncatedClientAck(..) => RealtimeServerEventType::ItemTruncatedClientAck,
            AssistantTruncated(..) => RealtimeServerEventType::AssistantTruncated,
            TranscriptionCompleted(..) => RealtimeServerEventType::TranscriptionCompleted,
            TranscriptionDelta(..) => RealtimeServerEventType::TranscriptionDelta,
            TranscriptionFailed(..) => RealtimeServerEventType::TranscriptionFailed,
            ItemDone(..) => RealtimeServerEventType::ItemDone,
            ItemRetrieved(..) => RealtimeServerEventType::ItemRetrieved,
            ResponseCreated(..) => RealtimeServerEventType::ResponseCreated,
            ResponseOutputItemAdded(..) => RealtimeServerEventType::ResponseOutputItemAdded,
            ResponseOutputItemDone(..) => RealtimeServerEventType::ResponseOutputItemDone,
            ResponseContentPartAdded(..) => RealtimeServerEventType::ResponseContentPartAdded,
            ResponseContentPartDone(..) => RealtimeServerEventType::ResponseContentPartDone,
            ResponseOutputAudioTranscriptDelta(..) => {
                RealtimeServerEventType::ResponseOutputAudioTranscriptDelta
            }
            ResponseOutputAudioTranscriptDone(..) => {
                RealtimeServerEventType::ResponseOutputAudioTranscriptDone
            }
            ResponseOutputAudioDelta(..) => RealtimeServerEventType::ResponseOutputAudioDelta,
            ResponseOutputAudioDone(..) => RealtimeServerEventType::ResponseOutputAudioDone,
            ResponseOutputTextDelta(..) => RealtimeServerEventType::ResponseOutputTextDelta,
            ResponseOutputTextDone(..) => RealtimeServerEventType::ResponseOutputTextDone,
            ResponseFunctionCallArgumentsDelta(..) => {
                RealtimeServerEventType::ResponseFunctionCallArgumentsDelta
            }
            ResponseFunctionCallArgumentsDone(..) => {
                RealtimeServerEventType::ResponseFunctionCallArgumentsDone
            }
            ResponseToolProgress(..) => RealtimeServerEventType::ResponseToolProgress,
            ResponseCancelled(..) => RealtimeServerEventType::ResponseCancelled,
            ResponseDone(..) => RealtimeServerEventType::ResponseDone,
            OutputAudioBufferCleared => RealtimeServerEventType::OutputAudioBufferCleared,
            OutputAudioBufferStarted(..) => RealtimeServerEventType::OutputAudioBufferStarted,
            OutputAudioBufferStopped(..) => RealtimeServerEventType::OutputAudioBufferStopped,
            RateLimitsUpdated(..) => RealtimeServerEventType::RateLimitsUpdated,
            Error(..) => RealtimeServerEventType::Error,
            Diarization(..) => RealtimeServerEventType::Diarization,
        }
    }

    pub fn type_name(&self) -> &'static str {
        use OutboundEvent::*;
        match self {
            SessionCreated(..) => "session.created",
            SessionUpdated(..) => "session.updated",
            SessionDone(..) => "session.done",
            SpeechStarted(..) => "input_audio_buffer.speech_started",
            SpeechStopped(..) => "input_audio_buffer.speech_stopped",
            BufferCommitted(..) => "input_audio_buffer.committed",
            BufferCleared => "input_audio_buffer.cleared",
            PartialTranscription(..) => "input_audio_buffer.partial_transcription",
            ItemAdded(..) => "conversation.item.added",
            ItemDeleted(..) => "conversation.item.deleted",
            ItemTruncatedClientAck(..) => "conversation.item.truncated",
            AssistantTruncated(..) => "conversation.item.assistant_truncated",
            TranscriptionCompleted(..) => "conversation.item.input_audio_transcription.completed",
            TranscriptionDelta(..) => "conversation.item.input_audio_transcription.delta",
            TranscriptionFailed(..) => "conversation.item.input_audio_transcription.failed",
            ItemDone(..) => "conversation.item.done",
            ItemRetrieved(..) => "conversation.item.retrieved",
            ResponseCreated(..) => "response.created",
            ResponseOutputItemAdded(..) => "response.output_item.added",
            ResponseOutputItemDone(..) => "response.output_item.done",
            ResponseContentPartAdded(..) => "response.content_part.added",
            ResponseContentPartDone(..) => "response.content_part.done",
            ResponseOutputAudioTranscriptDelta(..) => "response.output_audio_transcript.delta",
            ResponseOutputAudioTranscriptDone(..) => "response.output_audio_transcript.done",
            ResponseOutputAudioDelta(..) => "response.output_audio.delta",
            ResponseOutputAudioDone(..) => "response.output_audio.done",
            ResponseOutputTextDelta(..) => "response.output_text.delta",
            ResponseOutputTextDone(..) => "response.output_text.done",
            ResponseFunctionCallArgumentsDelta(..) => "response.function_call_arguments.delta",
            ResponseFunctionCallArgumentsDone(..) => "response.function_call_arguments.done",
            ResponseToolProgress(..) => "response.tool_progress",
            ResponseCancelled(..) => "response.cancelled",
            ResponseDone(..) => "response.done",
            OutputAudioBufferCleared => "output_audio_buffer.cleared",
            OutputAudioBufferStarted(..) => "output_audio_buffer.started",
            OutputAudioBufferStopped(..) => "output_audio_buffer.stopped",
            RateLimitsUpdated(..) => "rate_limits.updated",
            Error(..) => "error",
            Diarization(..) => "conversation.item.diarization",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
    fn response_done_carries_audio_end_ms_and_status_details_only_when_not_completed() {
        for (status, status_str) in [
            (ResponseStatus::Completed, "completed"),
            (ResponseStatus::Cancelled, "cancelled"),
            (ResponseStatus::Incomplete, "incomplete"),
            (ResponseStatus::Failed, "failed"),
        ] {
            let ev = OutboundEvent::ResponseDone(ResponseDoneEvent {
                response: sample_response_payload(status, 1234),
            });
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
        {
            let ev = OutboundEvent::ResponseDone(ResponseDoneEvent {
                response: sample_response_payload(ResponseStatus::Completed, 0),
            });
            let v = serde_json::to_value(&ev).unwrap();
            assert!(v["response"].get("status_details").is_none());
        }
        {
            let mut p = sample_response_payload(ResponseStatus::Incomplete, 5_000);
            p.status_details = Some(ResponseStatusDetails {
                reason: ResponseStatusReason::DrainCap,
                error: None,
            });
            let v = serde_json::to_value(&p).unwrap();
            assert_eq!(v["status_details"]["reason"].as_str(), Some("drain_cap"));
        }
    }

    #[test]
    fn tool_progress_carries_response_id_and_per_tool_entries_matching_live_emitters() {
        let ev = OutboundEvent::ResponseToolProgress(ResponseToolProgressEvent {
            response_id: ResponseId::new("resp_1"),
            tools: vec![
                ToolProgressEntry {
                    name: "get_current_datetime".into(),
                    status: ToolStatus::Running,
                    args: None,
                    result: None,
                    error: None,
                    summary: None,
                },
                ToolProgressEntry {
                    name: "search".into(),
                    status: ToolStatus::Done,
                    args: Some(json!({"q": "weather"})),
                    result: Some("62F".into()),
                    error: Some(false),
                    summary: Some("so, 62 and sunny".into()),
                },
            ],
        });
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"].as_str(), Some("response.tool_progress"));
        assert_eq!(v["response_id"].as_str(), Some("resp_1"));
        let tools = v["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0]["name"].as_str(), Some("get_current_datetime"));
        assert_eq!(tools[0]["status"].as_str(), Some("running"));
        assert!(
            tools[0].get("args").is_none() && tools[0].get("result").is_none(),
            "running entries omit optional fields, matching the mock emitter"
        );
        assert_eq!(tools[1]["status"].as_str(), Some("done"));
        assert_eq!(tools[1]["args"]["q"].as_str(), Some("weather"));
        assert_eq!(tools[1]["summary"].as_str(), Some("so, 62 and sunny"));
        assert!(
            v.get("item_id").is_none() && v.get("output_index").is_none(),
            "the live wire (mock-speaches-server, Rock RealtimeFrameRouter) has no item_id or \
             output_index on response.tool_progress"
        );
    }

    #[test]
    fn error_payload_serializes_with_code_and_message_required() {
        let ev = OutboundEvent::Error(ErrorEvent {
            error: ErrorPayload {
                type_: "invalid_request_error".into(),
                code: "invalid_request_error".into(),
                message: "bad".into(),
                event_id: None,
                param: None,
            },
        });
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["type"].as_str(), Some("error"));
        assert_eq!(v["error"]["type"].as_str(), Some("invalid_request_error"));
        assert_eq!(v["error"]["code"].as_str(), Some("invalid_request_error"));
        assert_eq!(v["error"]["message"].as_str(), Some("bad"));
        assert!(v["error"].get("event_id").is_none());
        assert!(v["error"].get("param").is_none());
        {
            let p = ErrorPayload::for_code("vad_failed", "boom");
            assert_eq!(p.type_, "server_error");
            let p2 = ErrorPayload::for_code("response_already_active", "nope");
            assert_eq!(p2.type_, "invalid_request_error");
        }
    }

    #[test]
    fn typed_ids_serialize_as_strings_and_type_name_matches_the_serde_tag() {
        let ev = OutboundEvent::ResponseOutputAudioDone(ResponseOutputAudioDoneEvent {
            response_id: ResponseId::new("resp_1"),
            item_id: ItemId::new("item_a"),
            output_index: 0,
            content_index: 0,
        });
        let v = serde_json::to_value(&ev).unwrap();
        assert_eq!(v["response_id"].as_str(), Some("resp_1"));
        assert_eq!(v["item_id"].as_str(), Some("item_a"));
        assert_eq!(v["type"].as_str(), Some("response.output_audio.done"));

        let trunc = OutboundEvent::AssistantTruncated(AssistantTruncatedEvent {
            event_id: EventId::new("evt_1"),
            item_id: ItemId::new("item_a"),
            audio_end_ms: 1234,
            transcript: "partial".into(),
        });
        let tv = serde_json::to_value(&trunc).unwrap();
        assert_eq!(tv["event_id"].as_str(), Some("evt_1"));
        assert_eq!(tv["item_id"].as_str(), Some("item_a"));
        assert_eq!(tv["audio_end_ms"].as_u64(), Some(1234));
        {
            let ev = OutboundEvent::ResponseOutputAudioDone(ResponseOutputAudioDoneEvent {
                response_id: "r".into(),
                item_id: "i".into(),
                output_index: 0,
                content_index: 0,
            });
            let v = serde_json::to_value(&ev).unwrap();
            assert_eq!(v.get("type").and_then(|t| t.as_str()), Some(ev.type_name()));
        }
    }

    #[test]
    fn event_seq_is_strictly_increasing_and_stamp_overwrites_only_the_top_level_event_id() {
        let seq = EventSeq::new();
        let a = seq.next_id();
        let b = seq.next_id();
        assert_eq!(a.as_str(), "evt_000000000000000000000000");
        assert_eq!(b.as_str(), "evt_000000000000000000000001");
        assert!(a.as_str() < b.as_str());
        assert_eq!(seq.issued(), 2);
        {
            let seq = EventSeq::new();
            let ev = OutboundEvent::AssistantTruncated(AssistantTruncatedEvent {
                event_id: EventId::new("evt_caller_supplied"),
                item_id: "i".into(),
                audio_end_ms: 7,
                transcript: "x".into(),
            });
            let v = ev.to_wire_value(&seq).expect("wire value");
            assert_eq!(v["event_id"].as_str(), Some("evt_000000000000000000000000"));
            assert_eq!(v["audio_end_ms"].as_u64(), Some(7));
        }
        {
            let seq = EventSeq::new();
            let ev = OutboundEvent::Error(ErrorEvent {
                error: ErrorPayload {
                    type_: "invalid_request_error".into(),
                    code: "invalid_request_error".into(),
                    message: "bad".into(),
                    event_id: Some("evt_from_client".into()),
                    param: None,
                },
            });
            let v = ev.to_wire_value(&seq).expect("wire value");
            assert_eq!(v["event_id"].as_str(), Some("evt_000000000000000000000000"));
            assert_eq!(v["error"]["event_id"].as_str(), Some("evt_from_client"));
        }
        {
            let seq = EventSeq::new();
            let mut v = Value::String("not an event".into());
            assert!(seq.stamp(&mut v).is_none());
            assert_eq!(seq.issued(), 0);
        }
    }

    #[test]
    fn transcript_confidence_rides_the_completed_event_only_when_known() {
        let mut bare = serde_json::json!({"type": "conversation.item.input_audio_transcription.completed", "transcript": "t"});
        TranscriptConfidence::default().attach(&mut bare);
        assert_eq!(
            bare.as_object().unwrap().len(),
            2,
            "unknown confidence adds no field: {bare}"
        );
        let stt = crate::stt::TranscriptionResult {
            avg_logprob: Some(-0.25),
            no_speech_prob: Some(0.5),
            tokens: vec![crate::stt::TokenLogprob {
                token: " hi".into(),
                logprob: -0.5,
            }],
            ..Default::default()
        };
        let mut full = serde_json::json!({"type": "x"});
        TranscriptConfidence::from_stt(&stt).attach(&mut full);
        assert_eq!(full["avg_logprob"], -0.25);
        assert_eq!(full["no_speech_prob"], 0.5);
        assert!(full.get("compression_ratio").is_none());
        assert_eq!(full["logprobs"][0]["token"], " hi");
        assert_eq!(full["logprobs"][0]["logprob"], -0.5);
        let typed = serde_json::to_value(
            TranscriptionCompletedEvent {
                item_id: ItemId::new("item_a"),
                content_index: 0,
                transcript: "t".into(),
                avg_logprob: None,
                no_speech_prob: None,
                compression_ratio: None,
                logprobs: None,
            }
            .with_confidence(TranscriptConfidence::from_stt(&stt)),
        )
        .unwrap();
        assert_eq!(
            typed["logprobs"][0]["logprob"], -0.5,
            "typed event carries the confidence: {typed}"
        );
        assert!(
            typed.get("compression_ratio").is_none(),
            "unknown fields stay off the wire: {typed}"
        );
    }

    #[test]
    fn typeshared_event_type_enum_matches_type_name_for_every_variant() {
        let mut seen = std::collections::HashSet::new();
        for t in RealtimeServerEventType::ALL {
            assert!(
                seen.insert(t.wire_name()),
                "duplicate name {}",
                t.wire_name()
            );
        }
        assert_eq!(seen.len(), 39);
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
        for n in names {
            assert!(
                seen.contains(n),
                "type_name vocabulary entry {n} missing from RealtimeServerEventType"
            );
        }
    }
}
