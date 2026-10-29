from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from .state import Topic

class ResponseStatus(enum.Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    INCOMPLETE = "incomplete"
    FAILED = "failed"

class ResponseStatusReason(enum.Enum):
    DRAIN_CAP = "drain_cap"
    TOKEN_LIMIT = "token_limit"
    LLM_ERROR = "llm_error"
    TTS_ERROR = "tts_error"
    CLIENT_TOO_SLOW = "client_too_slow"
    BARGE_IN = "barge_in"
    CLIENT_CANCELLED = "client_cancelled"

@dataclass
class ErrorPayload:
    type_: str
    code: str
    message: str
    event_id: str | None = None
    param: str | None = None

    @classmethod
    def for_code(cls, code: str, message: str) -> ErrorPayload:
        from .errors import error_type_for

        return cls(type_=error_type_for(code), code=code, message=message)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.type_,
            "code": self.code,
            "message": self.message,
        }
        if self.event_id is not None:
            out["event_id"] = self.event_id
        if self.param is not None:
            out["param"] = self.param
        return out

@dataclass
class ResponseStatusDetails:
    reason: ResponseStatusReason
    error: ErrorPayload | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"reason": self.reason.value}
        if self.error is not None:
            out["error"] = self.error.to_json()
        return out

@dataclass
class ResponsePayload:
    id: str
    audio_end_ms: int
    object: str = "realtime.response"
    status: ResponseStatus = ResponseStatus.COMPLETED
    output: list[Any] = field(default_factory=list)
    status_details: ResponseStatusDetails | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "status": self.status.value,
            "audio_end_ms": int(self.audio_end_ms),
            "output": list(self.output),
        }
        if self.status_details is not None:
            out["status_details"] = self.status_details.to_json()
        return out

class _Tag(str, enum.Enum):
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    SESSION_DONE = "session.done"

    SPEECH_STARTED = "input_audio_buffer.speech_started"
    SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    BUFFER_COMMITTED = "input_audio_buffer.committed"
    BUFFER_CLEARED = "input_audio_buffer.cleared"
    PARTIAL_TRANSCRIPTION = "input_audio_buffer.partial_transcription"

    ITEM_ADDED = "conversation.item.added"
    ITEM_DELETED = "conversation.item.deleted"
    ITEM_TRUNCATED_CLIENT_ACK = "conversation.item.truncated"
    ASSISTANT_TRUNCATED = "conversation.item.assistant_truncated"
    TRANSCRIPTION_COMPLETED = "conversation.item.input_audio_transcription.completed"
    TRANSCRIPTION_DELTA = "conversation.item.input_audio_transcription.delta"
    TRANSCRIPTION_FAILED = "conversation.item.input_audio_transcription.failed"
    ITEM_DONE = "conversation.item.done"
    ITEM_RETRIEVED = "conversation.item.retrieved"

    RESPONSE_CREATED = "response.created"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
    RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
    RESPONSE_OUTPUT_AUDIO_DELTA = "response.output_audio.delta"
    RESPONSE_OUTPUT_TEXT_DELTA = "response.output_text.delta"
    RESPONSE_OUTPUT_TEXT_DONE = "response.output_text.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    RESPONSE_OUTPUT_AUDIO_DONE = "response.output_audio.done"
    RESPONSE_TOOL_PROGRESS = "response.tool_progress"
    RESPONSE_DONE = "response.done"
    RESPONSE_CANCELLED = "response.cancelled"

    OUTPUT_AUDIO_BUFFER_CLEARED = "output_audio_buffer.cleared"
    OUTPUT_AUDIO_BUFFER_STARTED = "output_audio_buffer.started"
    OUTPUT_AUDIO_BUFFER_STOPPED = "output_audio_buffer.stopped"

    RATE_LIMITS_UPDATED = "rate_limits.updated"

    ERROR = "error"

@dataclass
class OutboundEvent:
    tag: _Tag
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def session_created(cls, session: dict[str, Any]) -> OutboundEvent:
        return cls(_Tag.SESSION_CREATED, {"session": session})

    @classmethod
    def session_updated(cls, session: dict[str, Any]) -> OutboundEvent:
        return cls(_Tag.SESSION_UPDATED, {"session": session})

    @classmethod
    def session_done(cls, reason: str) -> OutboundEvent:
        return cls(_Tag.SESSION_DONE, {"reason": reason})

    @classmethod
    def speech_started(cls, item_id: str, audio_start_ms: int) -> OutboundEvent:
        return cls(_Tag.SPEECH_STARTED, {"item_id": item_id, "audio_start_ms": int(audio_start_ms)})

    @classmethod
    def speech_stopped(cls, item_id: str, audio_end_ms: int) -> OutboundEvent:
        return cls(_Tag.SPEECH_STOPPED, {"item_id": item_id, "audio_end_ms": int(audio_end_ms)})

    @classmethod
    def buffer_committed(cls, item_id: str) -> OutboundEvent:
        return cls(_Tag.BUFFER_COMMITTED, {"item_id": item_id})

    @classmethod
    def buffer_cleared(cls) -> OutboundEvent:
        return cls(_Tag.BUFFER_CLEARED, {})

    @classmethod
    def partial_transcription(cls, item_id: str, transcript: str, audio_end_ms: int) -> OutboundEvent:
        return cls(
            _Tag.PARTIAL_TRANSCRIPTION,
            {"item_id": item_id, "transcript": transcript, "audio_end_ms": int(audio_end_ms)},
        )

    @classmethod
    def item_added(cls, item: dict[str, Any]) -> OutboundEvent:
        return cls(_Tag.ITEM_ADDED, {"item": item})

    @classmethod
    def item_deleted(cls, item_id: str) -> OutboundEvent:
        return cls(_Tag.ITEM_DELETED, {"item_id": item_id})

    @classmethod
    def item_truncated_client_ack(
        cls, item_id: str, content_index: int, audio_end_ms: int
    ) -> OutboundEvent:
        return cls(
            _Tag.ITEM_TRUNCATED_CLIENT_ACK,
            {
                "item_id": item_id,
                "content_index": int(content_index),
                "audio_end_ms": int(audio_end_ms),
            },
        )

    @classmethod
    def assistant_truncated(
        cls, event_id: str, item_id: str, audio_end_ms: int, transcript: str
    ) -> OutboundEvent:
        return cls(
            _Tag.ASSISTANT_TRUNCATED,
            {
                "event_id": event_id,
                "item_id": item_id,
                "audio_end_ms": int(audio_end_ms),
                "transcript": transcript,
            },
        )

    @classmethod
    def transcription_completed(
        cls, item_id: str, content_index: int, transcript: str
    ) -> OutboundEvent:
        return cls(
            _Tag.TRANSCRIPTION_COMPLETED,
            {"item_id": item_id, "content_index": int(content_index), "transcript": transcript},
        )

    @classmethod
    def transcription_failed(
        cls, item_id: str, content_index: int, error: dict[str, Any]
    ) -> OutboundEvent:
        return cls(
            _Tag.TRANSCRIPTION_FAILED,
            {"item_id": item_id, "content_index": int(content_index), "error": error},
        )

    @classmethod
    def response_created(cls, response: dict[str, Any]) -> OutboundEvent:
        return cls(_Tag.RESPONSE_CREATED, {"response": response})

    @classmethod
    def response_output_item_added(
        cls, response_id: str, output_index: int, item: dict[str, Any]
    ) -> OutboundEvent:
        return cls(
            _Tag.RESPONSE_OUTPUT_ITEM_ADDED,
            {"response_id": response_id, "output_index": int(output_index), "item": item},
        )

    @classmethod
    def response_output_item_done(
        cls, response_id: str, output_index: int, item: dict[str, Any]
    ) -> OutboundEvent:
        return cls(
            _Tag.RESPONSE_OUTPUT_ITEM_DONE,
            {"response_id": response_id, "output_index": int(output_index), "item": item},
        )

    @classmethod
    def response_content_part_added(
        cls,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        part: dict[str, Any],
    ) -> OutboundEvent:
        return cls(
            _Tag.RESPONSE_CONTENT_PART_ADDED,
            {
                "response_id": response_id,
                "item_id": item_id,
                "output_index": int(output_index),
                "content_index": int(content_index),
                "part": part,
            },
        )

    @classmethod
    def response_content_part_done(
        cls,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        part: dict[str, Any],
    ) -> OutboundEvent:
        return cls(
            _Tag.RESPONSE_CONTENT_PART_DONE,
            {
                "response_id": response_id,
                "item_id": item_id,
                "output_index": int(output_index),
                "content_index": int(content_index),
                "part": part,
            },
        )

    @classmethod
    def response_output_audio_transcript_delta(
        cls,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        delta: str,
    ) -> OutboundEvent:
        return cls(
            _Tag.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA,
            {
                "response_id": response_id,
                "item_id": item_id,
                "output_index": int(output_index),
                "content_index": int(content_index),
                "delta": delta,
            },
        )

    @classmethod
    def response_output_audio_transcript_done(
        cls,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        transcript: str,
    ) -> OutboundEvent:
        return cls(
            _Tag.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE,
            {
                "response_id": response_id,
                "item_id": item_id,
                "output_index": int(output_index),
                "content_index": int(content_index),
                "transcript": transcript,
            },
        )

    @classmethod
    def response_output_audio_delta(
        cls,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
        delta: str,
    ) -> OutboundEvent:
        return cls(
            _Tag.RESPONSE_OUTPUT_AUDIO_DELTA,
            {
                "response_id": response_id,
                "item_id": item_id,
                "output_index": int(output_index),
                "content_index": int(content_index),
                "delta": delta,
            },
        )

    @classmethod
    def response_output_audio_done(
        cls,
        response_id: str,
        item_id: str,
        output_index: int,
        content_index: int,
    ) -> OutboundEvent:
        return cls(
            _Tag.RESPONSE_OUTPUT_AUDIO_DONE,
            {
                "response_id": response_id,
                "item_id": item_id,
                "output_index": int(output_index),
                "content_index": int(content_index),
            },
        )

    @classmethod
    def response_done(cls, response: ResponsePayload) -> OutboundEvent:
        return cls(_Tag.RESPONSE_DONE, {"response": response})

    @classmethod
    def error(cls, payload: ErrorPayload) -> OutboundEvent:
        return cls(_Tag.ERROR, {"error": payload})

    def type_name(self) -> str:
        return self.tag.value

    def topic(self) -> Topic:
        return Topic.classify(self.tag.value)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.tag.value}
        for k, v in self.payload.items():
            if isinstance(v, ResponsePayload):
                out[k] = v.to_json()
            elif isinstance(v, ErrorPayload):
                out[k] = v.to_json()
            else:
                out[k] = v
        return out

def serialize_outbound_event(ev: OutboundEvent) -> dict[str, Any]:
    return ev.to_json()
