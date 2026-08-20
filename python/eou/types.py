from __future__ import annotations

import enum
import math
from typing import Generic, Optional, TypeVar

from . import constants

class Eagerness(enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"

    @classmethod
    def parse(cls, s: str) -> Optional["Eagerness"]:
        v = s.strip().lower()
        if v == "low":
            return cls.LOW
        if v in ("medium", "med"):
            return cls.MEDIUM
        if v == "high":
            return cls.HIGH
        if v == "auto":
            return cls.AUTO
        return None

    def triple(self) -> tuple[float, int, int]:
        if self is Eagerness.LOW:
            return constants.EAGERNESS_LOW
        if self is Eagerness.MEDIUM:
            return constants.EAGERNESS_MEDIUM
        if self is Eagerness.HIGH:
            return constants.EAGERNESS_HIGH
        return constants.EAGERNESS_MEDIUM

class EouKind(enum.Enum):
    VAD = "vad"
    TEXT = "text"
    AUDIO = "audio"
    FUSION = "fusion"
    HEURISTIC = "heuristic"
    INTEGRATED = "integrated"

    def as_str(self) -> str:
        return self.value

    @classmethod
    def parse(cls, s: str) -> Optional["EouKind"]:
        v = s.strip().lower()
        for k in cls:
            if k.value == v:
                return k
        return None

    def calls_classifier(self) -> bool:
        return self is not EouKind.VAD

    def is_v3_spec(self) -> bool:
        return self in (EouKind.VAD, EouKind.TEXT, EouKind.AUDIO, EouKind.FUSION)

EouKind.V3_SPEC = (EouKind.VAD, EouKind.TEXT, EouKind.AUDIO, EouKind.FUSION)
EouKind.EXTENSIONS = (EouKind.HEURISTIC, EouKind.INTEGRATED)

class AudioPadAlignment(enum.Enum):
    LEADING = "leading"
    TRAILING = "trailing"

    def as_str(self) -> str:
        return self.value

    @classmethod
    def parse(cls, s: str) -> Optional["AudioPadAlignment"]:
        v = s.strip().lower()
        if v == "leading":
            return cls.LEADING
        if v == "trailing":
            return cls.TRAILING
        return None

T = TypeVar("T")

class HardCapRace(Generic[T]):
    def __init__(self, hard_cap: bool, value: Optional[T] = None):
        self.hard_cap = hard_cap
        self.value = value

    @classmethod
    def cap(cls) -> "HardCapRace[T]":
        return cls(True, None)

    @classmethod
    def completed(cls, v: T) -> "HardCapRace[T]":
        return cls(False, v)

class EouModel:
    def score(self, context: str) -> float:
        raise NotImplementedError

    def score_with_audio(self, context: str, audio, sample_rate: int) -> float:
        return self.score(context)

class StubEouModel(EouModel):
    def score(self, context: str) -> float:
        return 1.0

def sigmoid_lerp(
    p: float,
    p_threshold: float,
    p_max: float,
    max_delay_ms: int,
    min_delay_ms: int,
) -> int:
    if p <= p_threshold:
        return int(max_delay_ms)
    if p >= p_max:
        return int(min_delay_ms)
    span = max(p_max - p_threshold, _F32_EPSILON)
    x = (p - p_threshold) / span
    k = constants.CURVE_K
    z = k * (x - 0.5)
    s = 1.0 / (1.0 + math.exp(-z))
    s0 = 1.0 / (1.0 + math.exp(k * 0.5))
    s1 = 1.0 / (1.0 + math.exp(-k * 0.5))
    t = (s - s0) / (s1 - s0)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    mx = float(max_delay_ms)
    mn = float(min_delay_ms)
    delay = mx + (mn - mx) * t
    rounded = round(delay)
    if rounded < 0:
        rounded = 0
    return int(rounded)

_F32_EPSILON = 1.1920929e-7
