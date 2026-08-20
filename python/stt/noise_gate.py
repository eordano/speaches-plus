from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

FULL_MS = 1500
OFF_MS = 5000
LOOSE_FLOOR = -3.0

def effective_avg_logprob_threshold(base: float | None, duration_ms: int) -> float | None:
    if base is None:
        return None
    if duration_ms <= FULL_MS:
        return base
    if duration_ms >= OFF_MS:
        return None
    frac = (duration_ms - FULL_MS) / (OFF_MS - FULL_MS)
    return base + frac * (LOOSE_FLOOR - base)

class NoiseRejection(Enum):
    NO_SPEECH_PROB = "no_speech_prob"
    AVG_LOGPROB = "avg_logprob"

    def as_str(self) -> str:
        return self.value

@dataclass(frozen=True)
class GateThresholds:
    no_speech_prob_threshold: float | None = None
    avg_logprob_threshold: float | None = None

    @classmethod
    def disabled(cls) -> GateThresholds:
        return cls(no_speech_prob_threshold=None, avg_logprob_threshold=None)

def evaluate(
    avg_no_speech_prob: float | None,
    avg_logprob: float | None,
    duration_ms: int,
    thresholds: GateThresholds,
) -> NoiseRejection | None:
    if avg_no_speech_prob is not None and thresholds.no_speech_prob_threshold is not None:
        if avg_no_speech_prob > thresholds.no_speech_prob_threshold:
            return NoiseRejection.NO_SPEECH_PROB
    if avg_logprob is not None:
        eff = effective_avg_logprob_threshold(thresholds.avg_logprob_threshold, duration_ms)
        if eff is not None and avg_logprob < eff:
            return NoiseRejection.AVG_LOGPROB
    return None
