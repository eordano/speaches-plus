from __future__ import annotations

class code:
    INVALID_REQUEST_ERROR = "invalid_request_error"
    UNKNOWN_EVENT_TYPE = "unknown_event_type"
    SESSION_NOT_ACTIVE = "session_not_active"
    SESSION_UPDATE_INVALID = "session_update_invalid"
    RESPONSE_ALREADY_ACTIVE = "response_already_active"
    RESPONSE_CANCEL_NOT_ACTIVE = "response_cancel_not_active"
    INPUT_AUDIO_BUFFER_COMMIT_EMPTY = "input_audio_buffer_commit_empty"
    CLIENT_TOO_SLOW = "client_too_slow"
    INTERNAL_STATE_ERROR = "internal_state_error"
    VAD_FAILED = "vad_failed"
    STT_FAILED = "stt_failed"
    MODEL_LOAD_FAILED = "model_load_failed"

KNOWN_CODES: tuple[str, ...] = (
    code.INVALID_REQUEST_ERROR,
    code.UNKNOWN_EVENT_TYPE,
    code.SESSION_NOT_ACTIVE,
    code.SESSION_UPDATE_INVALID,
    code.RESPONSE_ALREADY_ACTIVE,
    code.RESPONSE_CANCEL_NOT_ACTIVE,
    code.INPUT_AUDIO_BUFFER_COMMIT_EMPTY,
    code.CLIENT_TOO_SLOW,
    code.INTERNAL_STATE_ERROR,
    code.VAD_FAILED,
    code.STT_FAILED,
    code.MODEL_LOAD_FAILED,
)

_KNOWN: frozenset[str] = frozenset(KNOWN_CODES)

_INVALID_REQUEST: frozenset[str] = frozenset({
    code.INVALID_REQUEST_ERROR,
    code.UNKNOWN_EVENT_TYPE,
    code.SESSION_NOT_ACTIVE,
    code.SESSION_UPDATE_INVALID,
    code.RESPONSE_ALREADY_ACTIVE,
    code.RESPONSE_CANCEL_NOT_ACTIVE,
    code.INPUT_AUDIO_BUFFER_COMMIT_EMPTY,
    code.CLIENT_TOO_SLOW,
})

_SERVER_ERROR: frozenset[str] = frozenset({
    code.INTERNAL_STATE_ERROR,
    code.VAD_FAILED,
    code.STT_FAILED,
    code.MODEL_LOAD_FAILED,
})

def is_known_code(c: str) -> bool:
    return c in _KNOWN

def debug_assert_known_code(c: str) -> None:
    if __debug__ and not is_known_code(c):
        raise AssertionError(
            f"unknown error code {c!r}: add it to errors.code (RFC v3 §10.5)"
        )

def error_type_for(c: str) -> str:
    if c in _SERVER_ERROR:
        return "server_error"
    return "invalid_request_error"

def envelope(
    message: str,
    err_type: str | None = None,
    param: str | None = None,
    code_value: str | None = None,
) -> dict[str, dict[str, str | None]]:
    if err_type is None:
        err_type = error_type_for(code_value) if code_value else "invalid_request_error"
    return {
        "error": {
            "message": message,
            "type": err_type,
            "param": param,
            "code": code_value,
        }
    }
