"""Gated fusion combine + trained weights.

Mirrors go/internal/eou/gated_fusion.go (FuseScoresGated /
DefaultGatedFusionWeights) and rust/.../eou.rs::combine_fusion_gated /
DEFAULT_GATED_FUSION_WEIGHTS -- three implementations, one set of numbers.
The training pipeline (client/gated_fusion/train.py) writes the same
literal into all three sources after each re-fit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .features import GatedFusionFeatures, _clamp01

@dataclass(frozen=True)
class GatedFusionWeights:
    bias: float
    w_p_text: float
    w_p_audio: float
    w_audio_log_sec: float
    w_partial_log_chars: float
    w_strong_terminator: float
    w_soft_terminator: float
    w_continuation_last_word: float
    trained_samples: int
    trained_acc: float

    def gate(self, p_text: float, p_audio: float, feat: GatedFusionFeatures) -> float:
        x = feat.vector(p_text, p_audio)
        z = (
            self.bias * x[0]
            + self.w_p_text * x[1]
            + self.w_p_audio * x[2]
            + self.w_audio_log_sec * x[3]
            + self.w_partial_log_chars * x[4]
            + self.w_strong_terminator * x[5]
            + self.w_soft_terminator * x[6]
            + self.w_continuation_last_word * x[7]
        )
        return 1.0 / (1.0 + math.exp(-z))

DEFAULT_GATED_FUSION_WEIGHTS = GatedFusionWeights(
    bias=0.866202,
    w_p_text=0.283641,
    w_p_audio=0.018662,
    w_audio_log_sec=0.560501,
    w_partial_log_chars=1.195453,
    w_strong_terminator=0.258435,
    w_soft_terminator=0.003248,
    w_continuation_last_word=0.081883,
    trained_samples=350,
    trained_acc=0.9314,
)

def _is_garbage(p: float) -> bool:
    return math.isnan(p) or math.isinf(p) or p < 0 or p > 1

def combine_fusion(p_text: float, p_audio: float, rule: str, weight_text: float = 0.5) -> float:
    """Mirrors the closed-form rules in go/internal/eou/eou.go::FuseScores
    and rust/.../eou.rs::combine_fusion. For rule='gated' without
    features supplied, degrades to weighted-0.5 (paper §3.3 untrained
    baseline) -- callers wanting input-conditioned gating must use
    combine_fusion_with_features.
    """
    text_failed = _is_garbage(p_text)
    audio_failed = _is_garbage(p_audio)
    if text_failed and audio_failed:
        return 1.0
    if text_failed:
        return _clamp01(p_audio)
    if audio_failed:
        return _clamp01(p_text)
    pt = _clamp01(p_text)
    pa = _clamp01(p_audio)
    if rule == "max":
        combined = max(pt, pa)
    elif rule == "mean":
        combined = (pt + pa) * 0.5
    elif rule == "weighted":
        w = max(0.0, min(1.0, weight_text))
        combined = w * pt + (1.0 - w) * pa
    elif rule == "gated":
        combined = (pt + pa) * 0.5
    else:
        combined = 1.0 - (1.0 - pt) * (1.0 - pa)
    return _clamp01(combined)

def combine_fusion_gated(p_text: float, p_audio: float,
                          feat: GatedFusionFeatures,
                          weights: GatedFusionWeights = DEFAULT_GATED_FUSION_WEIGHTS) -> float:
    if _is_garbage(p_text) and _is_garbage(p_audio):
        return 1.0
    if _is_garbage(p_text):
        return _clamp01(p_audio)
    if _is_garbage(p_audio):
        return _clamp01(p_text)
    pt = _clamp01(p_text)
    pa = _clamp01(p_audio)
    g = weights.gate(pt, pa, feat)
    return _clamp01(g * pa + (1.0 - g) * pt)

def combine_fusion_with_features(rule: str, p_text: float, p_audio: float,
                                  weight_text: float,
                                  feat: GatedFusionFeatures,
                                  weights: GatedFusionWeights = DEFAULT_GATED_FUSION_WEIGHTS) -> float:
    if rule == "gated":
        return combine_fusion_gated(p_text, p_audio, feat, weights)
    return combine_fusion(p_text, p_audio, rule, weight_text)
