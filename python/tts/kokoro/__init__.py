from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .npz import Voice, load_voices
from .text import (
    DEFAULT_LANGUAGE,
    DEFAULT_VOICE,
    KOKORO_LANGUAGES,
    KOKORO_SAMPLE_RATE,
    MAX_SAMPLE_RATE,
    MIN_SAMPLE_RATE,
    SPEED_MAX,
    SPEED_MIN,
    f32_to_s16le,
    is_openai_voice_alias,
    normalize_for_tts,
    strip_emojis,
    strip_markdown_emphasis,
)

if TYPE_CHECKING:
    from .model import KOKORO_HF_REPO, KokoroTTS

_LAZY_FROM_MODEL = frozenset({"KOKORO_HF_REPO", "KokoroTTS"})

def __getattr__(name: str) -> Any:
    if name in _LAZY_FROM_MODEL:
        from . import model
        return getattr(model, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_LANGUAGE",
    "DEFAULT_VOICE",
    "KOKORO_HF_REPO",
    "KOKORO_LANGUAGES",
    "KOKORO_SAMPLE_RATE",
    "KokoroTTS",
    "MAX_SAMPLE_RATE",
    "MIN_SAMPLE_RATE",
    "SPEED_MAX",
    "SPEED_MIN",
    "Voice",
    "f32_to_s16le",
    "is_openai_voice_alias",
    "load_voices",
    "normalize_for_tts",
    "strip_emojis",
    "strip_markdown_emphasis",
]
