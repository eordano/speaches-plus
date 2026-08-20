from __future__ import annotations

class code:
    INVALID_REQUEST_ERROR = "invalid_request_error"
    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    SESSION_UPDATE_INVALID = "session_update_invalid"
    RESPONSE_ALREADY_ACTIVE = "response_already_active"
    RESPONSE_CANCEL_NOT_ACTIVE = "response_cancel_not_active"
    INPUT_AUDIO_BUFFER_COMMIT_EMPTY = "input_audio_buffer_commit_empty"
    CLIENT_TOO_SLOW = "client_too_slow"
    INTERNAL_STATE_ERROR = "internal_state_error"
    VAD_FAILED = "vad_failed"
    STT_FAILED = "stt_failed"

_KNOWN: set[str] = {
    code.INVALID_REQUEST_ERROR,
    code.UNKNOWN_EVENT_TYPE,
    code.SESSION_UPDATE_INVALID,
    code.RESPONSE_ALREADY_ACTIVE,
    code.RESPONSE_CANCEL_NOT_ACTIVE,
    code.INPUT_AUDIO_BUFFER_COMMIT_EMPTY,
    code.CLIENT_TOO_SLOW,
    code.INTERNAL_STATE_ERROR,
    code.VAD_FAILED,
    code.STT_FAILED,
}

def is_known_code(c: str) -> bool:
    return c in _KNOWN

def error_type_for(c: str) -> str:
    if c in (
        code.INVALID_REQUEST_ERROR,
        code.UNKNOWN_EVENT_TYPE,
        code.SESSION_UPDATE_INVALID,
        code.RESPONSE_ALREADY_ACTIVE,
        code.RESPONSE_CANCEL_NOT_ACTIVE,
        code.INPUT_AUDIO_BUFFER_COMMIT_EMPTY,
        code.CLIENT_TOO_SLOW,
    ):
        return "invalid_request_error"
    if c in (code.INTERNAL_STATE_ERROR, code.VAD_FAILED, code.STT_FAILED):
        return "server_error"
    return "invalid_request_error"

def debug_assert_known_code(c: str) -> None:
    if not is_known_code(c):
        raise AssertionError(f"unknown error code: {c}")
