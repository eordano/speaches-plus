from __future__ import annotations

TARGET_SAMPLE_RATE: int = 16_000
S16_SCALE: float = 32_768.0
S24_SCALE: float = 8_388_608.0
S32_SCALE: float = 2_147_483_648.0
BYTES_PER_S16: int = 2
MIME_RAW_PCM: str = "audio/pcm"
MIME_RAW: str = "audio/raw"

MIN_DECODE_SAMPLE_RATE: int = 1_000
MAX_DECODE_SAMPLE_RATE: int = 384_000

__all__ = [
    "TARGET_SAMPLE_RATE",
    "S16_SCALE",
    "S24_SCALE",
    "S32_SCALE",
    "BYTES_PER_S16",
    "MIME_RAW_PCM",
    "MIME_RAW",
    "MIN_DECODE_SAMPLE_RATE",
    "MAX_DECODE_SAMPLE_RATE",
]
