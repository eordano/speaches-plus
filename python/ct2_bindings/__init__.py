from __future__ import annotations

EXTENSION_AVAILABLE: bool = False
EXTENSION_IMPORT_ERROR: str | None = None

try:
    from . import _ct2 as _ct2
    from ._ct2 import Ct2Whisper as Ct2Whisper

    EXTENSION_AVAILABLE = True
except ImportError as _e:
    _ct2 = None
    Ct2Whisper = None
    EXTENSION_IMPORT_ERROR = str(_e)

__all__ = [
    "EXTENSION_AVAILABLE",
    "EXTENSION_IMPORT_ERROR",
    "Ct2Whisper",
    "_ct2",
]
