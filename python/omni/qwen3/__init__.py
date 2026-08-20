from __future__ import annotations

from . import _mps_compat
from .model import (
    DEFAULT_SPEAKER,
    INPUT_AUDIO_SR,
    OUTPUT_AUDIO_SR,
    SUPPORTED_SPEAKERS,
    ChatResult,
    Qwen3OmniWrapper,
)

_mps_compat.install()

__all__ = [
    "ChatResult",
    "DEFAULT_SPEAKER",
    "INPUT_AUDIO_SR",
    "OUTPUT_AUDIO_SR",
    "Qwen3OmniWrapper",
    "SUPPORTED_SPEAKERS",
]
