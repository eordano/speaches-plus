from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from . import (
    AUDIO_FORMAT_SUPPORTED,
    buffer_defaults,
    eou_defaults,
    turn_detection,
)
from .errors import code as errcode
from .session import TurnDetectionKind, validate_session_max_duration_s
from .wire import OutboundEvent

if TYPE_CHECKING:
    from .session import Session

FieldErr = tuple[str, str]

@dataclass
class StagedInstructions:
    set_to: str | None = None
    clear: bool = False

    @classmethod
    def from_value(cls, v: Any) -> StagedInstructions | FieldErr:
        if v is None:
            return cls(clear=True)
        if isinstance(v, str):
            return cls(set_to=v)
        return ("session.instructions", "instructions: must be a string or null")

@dataclass
class StagedTurnDetection:
    kind: TurnDetectionKind | None = None
    threshold: float | None = None
    neg_threshold: tuple[bool, float | None] | None = None
    min_speech_duration_ms: int | None = None
    prefix_padding_ms: int | None = None
    silence_duration_ms: int | None = None
    barge_in_delay_ms: int | None = None
    create_response: bool | None = None

@dataclass
class StagedSessionUpdate:
    instructions: StagedInstructions | None = None
    turn_detection: StagedTurnDetection | None = None
    session_max_duration_s: int | None = None
    voice: tuple[bool, str | None] | None = None
    min_speech_ms: int | None = None
    min_speech_for_response_ms: int | None = None
    no_speech_prob_threshold: tuple[bool, float | None] | None = None
    avg_logprob_threshold: tuple[bool, float | None] | None = None
    sealed_buffer_retention_count: int | None = None
    input_audio_format: str | None = None
    output_audio_format: str | None = None

def _parse_bounded_u64(v: Any, lo: int, hi: int, path: str, name: str) -> int | FieldErr:
    if isinstance(v, bool) or not isinstance(v, int):
        if not isinstance(v, int):
            return (path, f"{name}: must be in [{lo},{hi}]")
    if isinstance(v, bool):
        return (path, f"{name}: must be in [{lo},{hi}]")
    if v < lo or v > hi:
        return (path, f"{name}: must be in [{lo},{hi}]")
    return int(v)

def _parse_optional_unit_interval(
    v: Any, path: str, name: str
) -> tuple[bool, float | None] | FieldErr:
    if v is None:
        return (True, None)
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return (path, f"{name}: must be a number or null")
    f = float(v)
    if f != f or f < 0.0 or f > 1.0:
        return (path, f"{name}: must be in [0,1]")
    return (True, f)

def _validate_audio_format(v: Any) -> str | str:
    if not isinstance(v, str):
        return ""
    if v in AUDIO_FORMAT_SUPPORTED:
        return v
    return ""

def parse_session_update(session_obj: dict[str, Any]) -> StagedSessionUpdate | FieldErr:
    from .v2_compat import normalize_session_object

    session_obj = normalize_session_object(session_obj)
    staged = StagedSessionUpdate()

    if "instructions" in session_obj:
        r = StagedInstructions.from_value(session_obj["instructions"])
        if isinstance(r, tuple):
            return r
        staged.instructions = r

    if "turn_detection" in session_obj:
        r2 = _parse_turn_detection_update(session_obj["turn_detection"])
        if isinstance(r2, tuple) and len(r2) == 2 and isinstance(r2[0], str):
            return r2
        staged.turn_detection = r2

    if "session_max_duration_s" in session_obj:
        v = session_obj["session_max_duration_s"]
        if not isinstance(v, int) or isinstance(v, bool):
            return ("session.session_max_duration_s",
                    f"session_max_duration_s: must be in [1,{eou_defaults.SESSION_MAX_DURATION_S_MAX}]")
        try:
            staged.session_max_duration_s = validate_session_max_duration_s(v)
        except ValueError:
            return ("session.session_max_duration_s",
                    f"session_max_duration_s: must be in [1,{eou_defaults.SESSION_MAX_DURATION_S_MAX}]")

    if "voice" in session_obj:
        v = session_obj["voice"]
        if v is None:
            staged.voice = (True, None)
        elif isinstance(v, str):
            staged.voice = (True, v)
        else:
            return ("session.voice", "voice: must be a string or null")

    if "min_speech_ms" in session_obj:
        r3 = _parse_bounded_u64(
            session_obj["min_speech_ms"], 0, buffer_defaults.MIN_SPEECH_MS_MAX,
            "session.min_speech_ms", "min_speech_ms",
        )
        if isinstance(r3, tuple):
            return r3
        staged.min_speech_ms = r3

    if "min_speech_for_response_ms" in session_obj:
        r4 = _parse_bounded_u64(
            session_obj["min_speech_for_response_ms"], 0,
            buffer_defaults.MIN_SPEECH_FOR_RESPONSE_MS_MAX,
            "session.min_speech_for_response_ms", "min_speech_for_response_ms",
        )
        if isinstance(r4, tuple):
            return r4
        staged.min_speech_for_response_ms = r4

    if "no_speech_prob_threshold" in session_obj:
        r5 = _parse_optional_unit_interval(
            session_obj["no_speech_prob_threshold"],
            "session.no_speech_prob_threshold", "no_speech_prob_threshold",
        )
        if isinstance(r5, tuple) and len(r5) == 2 and isinstance(r5[0], str):
            return r5
        staged.no_speech_prob_threshold = r5

    if "avg_logprob_threshold" in session_obj:
        v = session_obj["avg_logprob_threshold"]
        if v is None:
            staged.avg_logprob_threshold = (True, None)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            f = float(v)
            if f != f:
                return ("session.avg_logprob_threshold", "avg_logprob_threshold: must be finite")
            staged.avg_logprob_threshold = (True, f)
        else:
            return ("session.avg_logprob_threshold", "avg_logprob_threshold: must be a number or null")

    if "sealed_buffer_retention_count" in session_obj:
        r6 = _parse_bounded_u64(
            session_obj["sealed_buffer_retention_count"], 0,
            buffer_defaults.SEALED_BUFFER_RETENTION_COUNT_MAX,
            "session.sealed_buffer_retention_count", "sealed_buffer_retention_count",
        )
        if isinstance(r6, tuple):
            return r6
        staged.sealed_buffer_retention_count = int(r6)

    if "input_audio_format" in session_obj:
        v = _validate_audio_format(session_obj["input_audio_format"])
        if not v:
            return ("session.input_audio_format",
                    f"input_audio_format: unsupported (supported: {list(AUDIO_FORMAT_SUPPORTED)})")
        staged.input_audio_format = v

    if "output_audio_format" in session_obj:
        v = _validate_audio_format(session_obj["output_audio_format"])
        if not v:
            return ("session.output_audio_format",
                    f"output_audio_format: unsupported (supported: {list(AUDIO_FORMAT_SUPPORTED)})")
        staged.output_audio_format = v

    return staged

def _parse_turn_detection_update(td: Any) -> StagedTurnDetection | FieldErr:
    staged = StagedTurnDetection()
    if td is None:
        return staged
    if not isinstance(td, dict):
        return ("session.turn_detection", "turn_detection: must be an object")

    if "type" in td:
        s = td["type"]
        if not isinstance(s, str):
            return ("session.turn_detection.type", "turn_detection.type: must be a string")
        kind = TurnDetectionKind.parse(s)
        if kind is None:
            return ("session.turn_detection.type",
                    "turn_detection.type: must be 'server_vad' or 'none'")
        staged.kind = kind

    if "threshold" in td:
        v = td["threshold"]
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return ("session.turn_detection.threshold",
                    "turn_detection.threshold: must be a number")
        f = float(v)
        if f != f or f < 0.0 or f > 1.0:
            return ("session.turn_detection.threshold",
                    "turn_detection.threshold: must be in [0,1]")
        staged.threshold = f

    if "neg_threshold" in td:
        v = td["neg_threshold"]
        if v is None:
            staged.neg_threshold = (True, None)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            f = float(v)
            if f != f or f < 0.0 or f > 1.0:
                return ("session.turn_detection.neg_threshold",
                        "turn_detection.neg_threshold: must be in [0,1]")
            staged.neg_threshold = (True, f)
        else:
            return ("session.turn_detection.neg_threshold",
                    "turn_detection.neg_threshold: must be a number or null")

    if "min_speech_duration_ms" in td:
        n = td["min_speech_duration_ms"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return ("session.turn_detection.min_speech_duration_ms",
                    f"turn_detection.min_speech_duration_ms: must be in [0,{buffer_defaults.MIN_SPEECH_MS_MAX}]")
        if n > buffer_defaults.MIN_SPEECH_MS_MAX:
            return ("session.turn_detection.min_speech_duration_ms",
                    f"turn_detection.min_speech_duration_ms: must be in [0,{buffer_defaults.MIN_SPEECH_MS_MAX}]")
        staged.min_speech_duration_ms = n

    if "prefix_padding_ms" in td:
        n = td["prefix_padding_ms"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return ("session.turn_detection.prefix_padding_ms",
                    "turn_detection.prefix_padding_ms: must be unsigned int")
        if n > turn_detection.PREFIX_PADDING_MS_MAX:
            return ("session.turn_detection.prefix_padding_ms",
                    "turn_detection.prefix_padding_ms: must be in [0,1000]")
        staged.prefix_padding_ms = n

    if "silence_duration_ms" in td:
        n = td["silence_duration_ms"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return ("session.turn_detection.silence_duration_ms",
                    "turn_detection.silence_duration_ms: must be unsigned int")
        lo, hi = turn_detection.SILENCE_DURATION_MS_MIN, turn_detection.SILENCE_DURATION_MS_MAX
        if n < lo or n > hi:
            return ("session.turn_detection.silence_duration_ms",
                    f"turn_detection.silence_duration_ms: must be in [{lo},{hi}]")
        staged.silence_duration_ms = n

    if "barge_in_delay_ms" in td:
        n = td["barge_in_delay_ms"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            return ("session.turn_detection.barge_in_delay_ms",
                    "turn_detection.barge_in_delay_ms: must be unsigned int")
        if n > turn_detection.BARGE_IN_DELAY_MS_MAX:
            return ("session.turn_detection.barge_in_delay_ms",
                    "turn_detection.barge_in_delay_ms: must be in [0,1000]")
        staged.barge_in_delay_ms = n

    if "create_response" in td:
        v = td["create_response"]
        if not isinstance(v, bool):
            return ("session.turn_detection.create_response",
                    "turn_detection.create_response: must be a boolean")
        staged.create_response = v

    return staged

async def handle_session_update(
    session: "Session", obj: dict[str, Any], event_id: str | None
) -> None:
    sess_obj = obj.get("session")
    if not isinstance(sess_obj, dict):
        await session._emit_error(
            errcode.SESSION_UPDATE_INVALID,
            "session.update: missing session object",
            event_id,
            "session",
        )
        return
    parsed = parse_session_update(sess_obj)
    if isinstance(parsed, tuple):
        path, msg = parsed
        await session._emit_error(errcode.SESSION_UPDATE_INVALID, msg, event_id, path)
        return

    async with session._state_lock:
        if parsed.instructions is not None:
            session.state.instructions = (
                None if parsed.instructions.clear else parsed.instructions.set_to
            )
        if parsed.session_max_duration_s is not None:
            session.session_max_duration_s = parsed.session_max_duration_s
        if parsed.voice is not None:
            session.voice = parsed.voice[1]
        if parsed.min_speech_ms is not None:
            session.min_speech_ms = parsed.min_speech_ms
        if parsed.min_speech_for_response_ms is not None:
            session.min_speech_for_response_ms = parsed.min_speech_for_response_ms
        if parsed.no_speech_prob_threshold is not None:
            session.no_speech_prob_threshold = parsed.no_speech_prob_threshold[1]
        if parsed.avg_logprob_threshold is not None:
            session.avg_logprob_threshold = parsed.avg_logprob_threshold[1]
        if parsed.sealed_buffer_retention_count is not None:
            session.state.sealed_buffer_retention_count = parsed.sealed_buffer_retention_count
        if parsed.input_audio_format is not None:
            session.input_audio_format = parsed.input_audio_format
        if parsed.output_audio_format is not None:
            session.output_audio_format = parsed.output_audio_format
        if parsed.turn_detection is not None:
            td = parsed.turn_detection
            cfg = session.turn_detection
            if td.kind is not None:
                cfg.kind = td.kind
            if td.threshold is not None:
                cfg.threshold = td.threshold
            if td.neg_threshold is not None:
                cfg.neg_threshold = td.neg_threshold[1]
            if td.min_speech_duration_ms is not None:
                cfg.min_speech_duration_ms = td.min_speech_duration_ms
            if td.prefix_padding_ms is not None:
                cfg.prefix_padding_ms = td.prefix_padding_ms
            if td.silence_duration_ms is not None:
                cfg.silence_duration_ms = td.silence_duration_ms
            if td.barge_in_delay_ms is not None:
                cfg.barge_in_delay_ms = td.barge_in_delay_ms
            if td.create_response is not None:
                cfg.create_response = td.create_response

    snapshot = await session.current_session_view()
    await session.emit(OutboundEvent.session_updated(snapshot))
