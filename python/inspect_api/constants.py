from __future__ import annotations

ERR_KINDS: tuple[str, ...] = (
    "error",
    "raised",
    "dropped",
    "failed",
    "phrase_error",
    "bargein_missed",
)

LANES: tuple[str, ...] = (
    "audio_level",
    "vad",
    "stt",
    "turn",
    "bargein",
    "eou",
    "diarization",
    "llm",
    "response",
    "tool",
    "tts_req",
    "tts_chunk",
    "tts_pacer",
    "wire",
    "state",
    "error",
)

RELAY_CAP: int = 1024
REPLAY_CAP: int = RELAY_CAP * 4

DEFAULT_RETENTION_COUNT: int = 200
DEFAULT_RETENTION_BYTES: int = 500_000_000
DEFAULT_RETENTION_DAYS: int = 30

def is_error_kind(kind: str) -> bool:
    return kind in ERR_KINDS

def is_known_lane(lane: str) -> bool:
    return lane in LANES
