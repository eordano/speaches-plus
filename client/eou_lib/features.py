"""Port of GatedFusionFeatures from go/internal/eou/gated_fusion.go and
rust/.../eou.rs. Same eight-element vector layout; same one-hot semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import log1p
from typing import Tuple

from .heuristic import (
    CONTINUATIONS,
    ends_strong_terminator,
    ends_soft_terminator,
    last_word_is_continuation,
)

@dataclass(frozen=True)
class GatedFusionFeatures:
    audio_ms: int = 0
    partial_chars: int = 0
    partial_ends_with_strong_terminator: bool = False
    partial_ends_with_soft_terminator: bool = False
    partial_last_word_is_continuation: bool = False

    def vector(self, p_text: float, p_audio: float) -> Tuple[float, ...]:
        """8-dim vector: [bias, p_text, p_audio, log_sec, log_chars,
        strong_term, soft_term, continuation_last_word]. Order MUST match
        GatedFusionWeights field order on both Go and Rust sides.
        """
        log_sec = log1p(self.audio_ms / 1000.0)
        log_chars = log1p(self.partial_chars)
        b = lambda x: 1.0 if x else 0.0
        return (
            1.0,
            _clamp01(p_text),
            _clamp01(p_audio),
            log_sec,
            log_chars,
            b(self.partial_ends_with_strong_terminator),
            b(self.partial_ends_with_soft_terminator),
            b(self.partial_last_word_is_continuation),
        )

def extract_gated_fusion_features(partial: str, audio_ms: int) -> GatedFusionFeatures:
    trimmed = (partial or "").strip()
    return GatedFusionFeatures(
        audio_ms=audio_ms,
        partial_chars=len(trimmed),
        partial_ends_with_strong_terminator=ends_strong_terminator(trimmed),
        partial_ends_with_soft_terminator=ends_soft_terminator(trimmed),
        partial_last_word_is_continuation=last_word_is_continuation(trimmed),
    )

def _clamp01(p: float) -> float:
    import math
    if math.isnan(p) or math.isinf(p):
        return 0.0
    if p < 0:
        return 0.0
    if p > 1:
        return 1.0
    return p
