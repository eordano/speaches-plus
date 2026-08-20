from __future__ import annotations

import enum
import json
from typing import Any

from .state import ConversationItem, ItemContent, ItemRole, ItemStatus
from .wire import (
    ErrorPayload,
    OutboundEvent,
    ResponsePayload,
    ResponseStatus,
    ResponseStatusDetails,
    ResponseStatusReason,
)

class ClientEventType(str, enum.Enum):
    SESSION_UPDATE = "session.update"
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    CONVERSATION_ITEM_RETRIEVE = "conversation.item.retrieve"
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"

class ServerEventType(str, enum.Enum):
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
    ITEM_TRUNCATED = "conversation.item.truncated"
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
    DIARIZATION = "conversation.item.diarization"

def parse_client_event(text: str) -> tuple[ClientEventType | None, dict[str, Any]]:
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None, {}
    if not isinstance(obj, dict):
        return None, {}
    t = obj.get("type")
    if not isinstance(t, str):
        return None, obj
    try:
        return ClientEventType(t), obj
    except ValueError:
        return None, obj

def assistant_audio_item_json(item_id: str, transcript: str, status: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "object": "realtime.item",
        "type": "message",
        "role": "assistant",
        "status": status,
        "content": [{"type": "audio", "transcript": transcript}],
    }

def item_to_json(item: ConversationItem) -> dict[str, Any]:
    if item.content.is_user_audio():
        obj: dict[str, Any] = {"type": "input_audio"}
        if item.content.transcript is not None:
            obj["transcript"] = item.content.transcript
        if item.content.audio_end_ms is not None:
            obj["audio_end_ms"] = int(item.content.audio_end_ms)
        content: list[Any] = [obj]
    elif item.content.is_assistant_audio():
        content = [
            {
                "type": "audio",
                "transcript": item.content.transcript or "",
                "audio_ms": int(item.content.audio_ms or 0),
            }
        ]
    else:
        part_type = "input_text" if item.role is ItemRole.USER else "text"
        content = [{"type": part_type, "text": item.content.text or ""}]
    return {
        "id": item.id,
        "object": "realtime.item",
        "type": "message",
        "role": item.role.as_str(),
        "status": item.status.as_str(),
        "content": content,
    }

def extract_text_from_content(content: Any) -> str | None:
    if not isinstance(content, list):
        return None
    text = ""
    for entry in content:
        if not isinstance(entry, dict):
            continue
        part_type = entry.get("type", "")
        if part_type in ("input_text", "text"):
            t = entry.get("text")
            if isinstance(t, str):
                if text:
                    text += " "
                text += t
    return text or None

def make_response_open_brackets(response_id: str, item_id: str) -> list[dict[str, Any]]:
    return [
        {
            "type": ServerEventType.RESPONSE_CREATED.value,
            "response": {
                "id": response_id,
                "object": "realtime.response",
                "status": "in_progress",
            },
        },
        {
            "type": ServerEventType.RESPONSE_OUTPUT_ITEM_ADDED.value,
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
        },
        {
            "type": ServerEventType.RESPONSE_CONTENT_PART_ADDED.value,
            "response_id": response_id,
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {"type": "audio", "transcript": ""},
        },
    ]

def make_response_brackets(
    response_id: str, item_id: str, transcript: str, item_status: str
) -> list[OutboundEvent]:
    return [
        OutboundEvent.response_output_audio_transcript_done(response_id, item_id, 0, 0, transcript),
        OutboundEvent.response_output_audio_done(response_id, item_id, 0, 0),
        OutboundEvent.response_content_part_done(
            response_id, item_id, 0, 0, {"type": "audio", "transcript": transcript}
        ),
        OutboundEvent.response_output_item_done(
            response_id,
            0,
            {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": item_status,
                "content": [{"type": "audio", "transcript": transcript}],
            },
        ),
    ]

def make_response_done(
    response_id: str,
    item_id: str,
    status: str,
    transcript: str | None,
    fail_reason: ResponseStatusReason | None,
    audio_end_ms: int,
) -> OutboundEvent:
    output: dict[str, Any] = {
        "id": item_id,
        "type": "message",
        "role": "assistant",
    }
    if transcript is not None:
        output["content"] = [{"type": "audio", "transcript": transcript}]

    parsed_status = {
        "completed": ResponseStatus.COMPLETED,
        "cancelled": ResponseStatus.CANCELLED,
        "incomplete": ResponseStatus.INCOMPLETE,
        "failed": ResponseStatus.FAILED,
    }.get(status, ResponseStatus.FAILED)

    status_details: ResponseStatusDetails | None = None
    if parsed_status is ResponseStatus.FAILED:
        reason = fail_reason if fail_reason is not None else ResponseStatusReason.LLM_ERROR
        status_details = ResponseStatusDetails(reason=reason)

    payload = ResponsePayload(
        id=response_id,
        audio_end_ms=int(audio_end_ms),
        status=parsed_status,
        output=[output],
        status_details=status_details,
    )
    return OutboundEvent.response_done(payload)

def make_incomplete_brackets(
    response_id: str, item_id: str, transcript: str, played_ms: int
) -> tuple[list[OutboundEvent], OutboundEvent]:
    brackets = make_response_brackets(response_id, item_id, transcript, "incomplete")
    payload = ResponsePayload(
        id=response_id,
        audio_end_ms=int(played_ms),
        status=ResponseStatus.INCOMPLETE,
        output=[assistant_audio_item_json(item_id, transcript, "incomplete")],
        status_details=ResponseStatusDetails(reason=ResponseStatusReason.DRAIN_CAP),
    )
    return brackets, OutboundEvent.response_done(payload)

def make_cancelled_brackets(
    response_id: str,
    item_id: str,
    transcript: str,
    played_ms: int,
    reason: ResponseStatusReason,
    *,
    transcript_done_emitted: bool = False,
    audio_done_emitted: bool = False,
) -> tuple[list[OutboundEvent], OutboundEvent]:
    """Return (brackets, response.done).

    `transcript_done_emitted` / `audio_done_emitted` let the caller suppress
    duplicate close events when the cancelled response had already emitted them
    on the wire (e.g. LLM stream finished and transcript.done fired before the
    user barge-in cancelled mid-TTS). Each stream's `.done` event is emitted at
    most once per response per the realtime wire invariant.
    """
    full = make_response_brackets(response_id, item_id, transcript, "incomplete")
    brackets: list[OutboundEvent] = []
    for ev in full:
        tag = ev.type_name()
        if transcript_done_emitted and tag == "response.output_audio_transcript.done":
            continue
        if audio_done_emitted and tag == "response.output_audio.done":
            continue
        brackets.append(ev)
    payload = ResponsePayload(
        id=response_id,
        audio_end_ms=int(played_ms),
        status=ResponseStatus.CANCELLED,
        output=[assistant_audio_item_json(item_id, transcript, "incomplete")],
        status_details=ResponseStatusDetails(reason=reason),
    )
    return brackets, OutboundEvent.response_done(payload)

def make_error_event(code: str, message: str, event_id: str | None, param: str | None) -> OutboundEvent:
    from .errors import debug_assert_known_code, error_type_for

    debug_assert_known_code(code)
    return OutboundEvent.error(
        ErrorPayload(
            type_=error_type_for(code),
            code=code,
            message=message,
            event_id=event_id,
            param=param,
        )
    )

def make_server_truncate_event(
    event_id: str, item_id: str, audio_end_ms: int, transcript: str
) -> dict[str, Any] | None:
    if audio_end_ms == 0:
        return None
    return {
        "event_id": event_id,
        "type": ServerEventType.ASSISTANT_TRUNCATED.value,
        "item_id": item_id,
        "audio_end_ms": int(audio_end_ms),
        "transcript": transcript,
    }

__all__ = [
    "ClientEventType",
    "ServerEventType",
    "assistant_audio_item_json",
    "extract_text_from_content",
    "item_to_json",
    "make_cancelled_brackets",
    "make_error_event",
    "make_incomplete_brackets",
    "make_response_brackets",
    "make_response_done",
    "make_response_open_brackets",
    "make_server_truncate_event",
    "parse_client_event",
]
