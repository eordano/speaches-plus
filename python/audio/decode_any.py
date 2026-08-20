from __future__ import annotations

import numpy as np

from .avdecode import decode_via_symphonia
from .types import BYTES_PER_S16, MIME_RAW, MIME_RAW_PCM, S16_SCALE
from .wav import decode_wav_to_16k_mono

def decode_any_to_16k_mono(
    data: bytes | bytearray | memoryview, mime: str | None = None
) -> np.ndarray:
    mime_lc = mime.strip().lower() if isinstance(mime, str) else None
    if mime_lc in (MIME_RAW_PCM, MIME_RAW):
        b = bytes(data)
        n = len(b) // BYTES_PER_S16
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        arr = np.frombuffer(b[: n * BYTES_PER_S16], dtype="<i2").astype(np.float32)
        return arr / np.float32(S16_SCALE)
    try:
        return decode_wav_to_16k_mono(data)
    except Exception:
        pass
    return decode_via_symphonia(bytes(data))

__all__ = ["decode_any_to_16k_mono"]
