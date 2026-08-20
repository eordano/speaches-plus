from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tts.kokoro.text import KOKORO_LANGUAGES

WHISPER_LANGUAGES = (
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca",
    "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms",
    "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg", "lt", "la",
    "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be",
    "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn",
    "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
    "ba", "jw", "su", "yue",
)

__all__ = [
    "KOKORO_LANGUAGES",
    "ListModelsResponse",
    "Model",
    "WHISPER_LANGUAGES",
]

@dataclass
class Model:
    id: str
    owned_by: str
    task: str
    created: int = 1
    languages: list[str] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "object": "model",
            "created": self.created,
            "owned_by": self.owned_by,
            "language": self.languages,
            "task": self.task,
        }
        out.update(self.extras)
        return out

@dataclass
class ListModelsResponse:
    data: list[Model]
    object: str = "list"

    def to_dict(self) -> dict[str, Any]:
        return {"object": self.object, "data": [m.to_dict() for m in self.data]}
