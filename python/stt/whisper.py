from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

import env

class Backend(str, Enum):
    CT2 = "ct2"
    WHISPER_CPP = "whisper_cpp"

    @classmethod
    def from_env(cls) -> "Backend":
        raw = env.read_str(env.STT_BACKEND, "").strip().lower()
        if raw in {"ct2", "ctranslate2", "faster-whisper"}:
            return cls.CT2
        return cls.WHISPER_CPP

    def as_str(self) -> str:
        return self.value

@dataclass
class TimedSegment:
    t_start_ms: int = 0
    t_end_ms: int = 0
    text: str = ""
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

@dataclass
class TranscriptionResult:
    text: str = ""
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float | None = None
    segments: list[TimedSegment] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "TranscriptionResult":
        return cls()

    @classmethod
    def from_text(cls, s: str) -> "TranscriptionResult":
        return cls(text=s)

@runtime_checkable
class WhisperBackend(Protocol):
    model_id: str

    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        *,
        language: str | None = None,
        prompt: str | None = None,
        with_timestamps: bool = False,
        task: str = "transcribe",
    ) -> TranscriptionResult: ...

def peak_amplitude(samples: np.ndarray) -> float:
    if len(samples) == 0:
        return 0.0
    return float(np.max(np.abs(samples)))

def join_segments(segments: Iterable[str]) -> str:
    out: list[str] = []
    for seg in segments:
        trimmed = seg.strip()
        if not trimmed:
            continue
        out.append(trimmed)
    return " ".join(out)

def strip_special_tokens(s: str) -> str:
    out: list[str] = []
    depth = 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">" and depth > 0:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)

def parse_timestamp_token(tok: str) -> int | None:
    if not tok.startswith("<|") or not tok.endswith("|>"):
        return None
    inner = tok[2:-2]
    if "." not in inner:
        return None
    whole, frac = inner.split(".", 1)
    if not whole or not frac:
        return None
    if not whole.isdigit() or not frac.isdigit():
        return None
    secs = int(whole)
    frac_val = int(frac)
    n = len(frac)
    if n == 1:
        frac_ms = frac_val * 100
    elif n == 2:
        frac_ms = frac_val * 10
    elif n == 3:
        frac_ms = frac_val
    elif n > 3:
        frac_ms = frac_val // (10 ** (n - 3))
    else:
        frac_ms = 0
    return secs * 1000 + frac_ms

def classify_timestamp(tok: str, tok_id: int | None, ts_begin_id: int | None) -> int | None:
    if tok_id is not None and ts_begin_id is not None:
        if ts_begin_id <= tok_id < ts_begin_id + 1501:
            step = (tok_id - ts_begin_id) * 20
            return min(step, 0xFFFFFFFF)
        return None
    return parse_timestamp_token(tok)

@dataclass
class _Ct2Segment:
    t_start_ms: int
    t_end_ms: int
    text_tokens: list[str]

def split_ct2_segments(
    tokens: Sequence[str],
    token_ids: Sequence[int],
    ts_begin_id: int | None,
    audio_ms: int,
) -> list[_Ct2Segment]:
    segments: list[_Ct2Segment] = []
    current_start: int | None = None
    current_tokens: list[str] = []
    n = len(tokens)
    for i in range(n):
        tok = tokens[i]
        tid = token_ids[i] if i < len(token_ids) else None
        ts = classify_timestamp(tok, tid, ts_begin_id)
        if ts is not None:
            ts_ms = min(ts, audio_ms)
            if current_start is None:
                current_start = ts_ms
            else:
                start = current_start
                valid = ts_ms > start and len(current_tokens) > 0
                if valid:
                    segments.append(_Ct2Segment(
                        t_start_ms=min(start, audio_ms),
                        t_end_ms=ts_ms,
                        text_tokens=current_tokens,
                    ))
                    current_tokens = []
                elif current_tokens:
                    current_tokens = []
                current_start = ts_ms
        else:
            if current_start is not None:
                current_tokens.append(tok)
    return segments
