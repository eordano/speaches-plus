from __future__ import annotations

EXTENSION_AVAILABLE: bool = False
EXTENSION_IMPORT_ERROR: str | None = None

try:
    from . import _whisper as _whisper
    from ._whisper import WhisperContext as WhisperContext

    EXTENSION_AVAILABLE = True
except ImportError as _e:
    _whisper = None
    WhisperContext = None
    EXTENSION_IMPORT_ERROR = str(_e)

__all__ = [
    "EXTENSION_AVAILABLE",
    "EXTENSION_IMPORT_ERROR",
    "WhisperContext",
    "_whisper",
]
