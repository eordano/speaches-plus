from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Optional

from .types import EouModel

class FusionRule(enum.Enum):
    NOISY_OR = "noisy_or"
    MAX = "max"
    MEAN = "mean"
    WEIGHTED = "weighted"
    GATED = "gated"

    def as_str(self) -> str:
        return self.value

    @classmethod
    def parse(cls, s: str) -> Optional["FusionRule"]:
        v = s.strip().lower()
        if v in ("noisy_or", "noisy-or", "noisyor"):
            return cls.NOISY_OR
        if v == "max":
            return cls.MAX
        if v in ("mean", "avg", "average"):
            return cls.MEAN
        if v == "weighted":
            return cls.WEIGHTED
        if v == "gated":
            return cls.GATED
        return None

    @classmethod
    def default(cls) -> "FusionRule":
        return cls.GATED

CONTINUATION_WORDS = (
    "and",
    "or",
    "but",
    "with",
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "is",
    "was",
    "are",
    "were",
    "because",
    "since",
    "if",
    "when",
    "while",
    "as",
    "than",
    "that",
    "which",
    "who",
    "whom",
    "whose",
)

@dataclass
class GatedFusionFeatures:
    audio_ms: int = 0
    partial_chars: int = 0
    partial_ends_with_strong_terminator: bool = False
    partial_ends_with_soft_terminator: bool = False
    partial_last_word_is_continuation: bool = False

    def vector(self, p_text: float, p_audio: float) -> tuple[float, ...]:
        log_sec = math.log(1.0 + float(self.audio_ms) / 1000.0)
        log_chars = math.log(1.0 + float(self.partial_chars))

        def b(x: bool) -> float:
            return 1.0 if x else 0.0

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

def extract_gated_fusion_features(
    partial: str, audio_ms: int
) -> GatedFusionFeatures:
    trimmed = partial.strip()
    feat = GatedFusionFeatures(
        audio_ms=audio_ms,
        partial_chars=len(trimmed.encode("utf-8")),
    )
    if trimmed:
        last = trimmed[-1]
        if last in (".", "!", "?"):
            feat.partial_ends_with_strong_terminator = True
        elif last in (",", ";", ":", "-"):
            feat.partial_ends_with_soft_terminator = True
    feat.partial_last_word_is_continuation = _last_word_is_continuation(trimmed)
    return feat

def _last_word_is_continuation(s: str) -> bool:
    if not s:
        return False
    chars = list(s)
    end = len(chars)
    while end > 0:
        c = chars[end - 1]
        if c.isalnum() or c == "'" or c == "-":
            break
        end -= 1
    start = end
    while start > 0:
        c = chars[start - 1]
        if c.isalnum() or c == "'" or c == "-":
            start -= 1
        else:
            break
    if start >= end:
        return False
    word = "".join(chars[start:end]).lower()
    return word in CONTINUATION_WORDS

@dataclass
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

    def gate(
        self, p_text: float, p_audio: float, feat: GatedFusionFeatures
    ) -> float:
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

def _clamp01(p: float) -> float:
    if not math.isfinite(p):
        return 0.0
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p

def is_garbage_prob(p: float) -> bool:
    if not math.isfinite(p):
        return True
    return p < 0.0 or p > 1.0

def combine_fusion(
    p_text: float, p_audio: float, rule: FusionRule, weight_text: float
) -> float:
    text_failed = is_garbage_prob(p_text)
    audio_failed = is_garbage_prob(p_audio)
    if text_failed and audio_failed:
        return 1.0
    if text_failed:
        return _clamp01(p_audio)
    if audio_failed:
        return _clamp01(p_text)
    pt = _clamp01(p_text)
    pa = _clamp01(p_audio)
    if rule is FusionRule.NOISY_OR:
        combined = 1.0 - (1.0 - pt) * (1.0 - pa)
    elif rule is FusionRule.MAX:
        combined = pt if pt > pa else pa
    elif rule is FusionRule.MEAN:
        combined = (pt + pa) * 0.5
    elif rule is FusionRule.WEIGHTED:
        w = _clamp01(weight_text)
        combined = w * pt + (1.0 - w) * pa
    elif rule is FusionRule.GATED:
        combined = (pt + pa) * 0.5
    else:
        combined = (pt + pa) * 0.5
    return _clamp01(combined)

def combine_fusion_gated(
    p_text: float,
    p_audio: float,
    feat: GatedFusionFeatures,
    weights: GatedFusionWeights,
) -> float:
    text_failed = is_garbage_prob(p_text)
    audio_failed = is_garbage_prob(p_audio)
    if text_failed and audio_failed:
        return 1.0
    if text_failed:
        return _clamp01(p_audio)
    if audio_failed:
        return _clamp01(p_text)
    pt = _clamp01(p_text)
    pa = _clamp01(p_audio)
    g = weights.gate(pt, pa, feat)
    return _clamp01(g * pa + (1.0 - g) * pt)

def combine_fusion_with_features(
    p_text: float,
    p_audio: float,
    rule: FusionRule,
    weight_text: float,
    feat: GatedFusionFeatures,
    weights: GatedFusionWeights,
) -> float:
    if rule is FusionRule.GATED:
        return combine_fusion_gated(p_text, p_audio, feat, weights)
    return combine_fusion(p_text, p_audio, rule, weight_text)

class FusionEouModel(EouModel):
    def __init__(
        self,
        text: EouModel,
        audio: EouModel,
        rule: FusionRule,
        weight_text: float,
    ) -> None:
        self.text = text
        self.audio = audio
        self.rule = rule
        self.weight_text = weight_text

    def score_pair(self, context: str) -> tuple[float, float]:
        return (self.text.score(context), self.audio.score(context))

    def score_pair_with_audio(
        self, context: str, audio, sample_rate: int
    ) -> tuple[float, float]:
        return (
            self.text.score(context),
            self.audio.score_with_audio(context, audio, sample_rate),
        )

    def score(self, context: str) -> float:
        p_text, p_audio = self.score_pair(context)
        feat = extract_gated_fusion_features(context, 0)
        return combine_fusion_with_features(
            p_text,
            p_audio,
            self.rule,
            self.weight_text,
            feat,
            DEFAULT_GATED_FUSION_WEIGHTS,
        )

    def score_with_audio(self, context: str, audio, sample_rate: int) -> float:
        p_text, p_audio = self.score_pair_with_audio(context, audio, sample_rate)
        if sample_rate > 0:
            try:
                length = len(audio)
            except TypeError:
                length = int(getattr(audio, "size", 0))
            audio_ms = int((length * 1000) // sample_rate)
        else:
            audio_ms = 0
        feat = extract_gated_fusion_features(context, audio_ms)
        return combine_fusion_with_features(
            p_text,
            p_audio,
            self.rule,
            self.weight_text,
            feat,
            DEFAULT_GATED_FUSION_WEIGHTS,
        )
