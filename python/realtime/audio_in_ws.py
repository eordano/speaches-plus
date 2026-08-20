from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .session import Session

log = logging.getLogger("realtime.audio_in_ws")

TARGET_HZ = 16_000
SCALE_I16 = 32_768.0

class IngestError(Exception):
    pass

class _UnsupportedFormat(IngestError):
    pass

_FORMATS: dict[str, tuple[str, int]] = {
    "pcm16": ("pcm16le", 24_000),
    "": ("pcm16le", 24_000),
    "pcm16_8k": ("pcm16le", 8_000),
    "pcm16_16k": ("pcm16le", 16_000),
    "pcm16_24k": ("pcm16le", 24_000),
    "pcm16_44k1": ("pcm16le", 44_100),
    "pcm16_48k": ("pcm16le", 48_000),
    "g711_ulaw": ("ulaw", 8_000),
    "g711_alaw": ("alaw", 8_000),
}

@dataclass
class WsAudioIngest:
    codec: str
    src_hz: int
    src_position: float = 0.0
    last_sample: float = 0.0

    @classmethod
    def new(cls, format: str) -> WsAudioIngest:
        if format not in _FORMATS:
            raise _UnsupportedFormat(f"unsupported input_audio_format: {format}")
        codec, src_hz = _FORMATS[format]
        return cls(codec=codec, src_hz=src_hz)

    def ingest_b64(self, b64: str) -> list[float]:
        try:
            payload = base64.b64decode(b64.strip(), validate=False)
        except Exception as err:
            raise IngestError(f"base64 decode failed: {err}") from err
        f32_in = self._decode_bytes(payload)
        if self.src_hz == TARGET_HZ:
            if f32_in:
                self.last_sample = f32_in[-1]
            return f32_in
        return self._linear_resample(f32_in)

    def _decode_bytes(self, payload: bytes) -> list[float]:
        if self.codec == "pcm16le":
            if len(payload) % 2 != 0:
                raise IngestError(f"PCM16 payload has odd byte count: {len(payload)}")
            out: list[float] = []
            for i in range(0, len(payload), 2):
                v = int.from_bytes(payload[i : i + 2], "little", signed=True)
                out.append(v / SCALE_I16)
            return out
        if self.codec == "ulaw":
            from audio.g711 import ulaw_decode_byte

            return [ulaw_decode_byte(b) / SCALE_I16 for b in payload]
        if self.codec == "alaw":
            from audio.g711 import alaw_decode_byte

            return [alaw_decode_byte(b) / SCALE_I16 for b in payload]
        raise IngestError(f"unsupported codec: {self.codec}")

    def _linear_resample(self, src: list[float]) -> list[float]:
        if not src:
            return []
        ratio = TARGET_HZ / float(self.src_hz)
        out: list[float] = []
        pos = self.src_position
        step = 1.0 / ratio
        last = self.last_sample
        n = len(src)
        while pos < n:
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            if lo >= n:
                s = last
            else:
                a = src[lo]
                b = src[hi]
                s = a + (b - a) * frac
            out.append(s)
            last = s
            pos += step
        self.src_position = pos - n
        self.last_sample = last
        return out

async def handle_audio_append(session: "Session", b64: str) -> None:
    fmt = session.input_audio_format
    try:
        ingest = WsAudioIngest.new(fmt)
        samples = ingest.ingest_b64(b64)
    except _UnsupportedFormat as err:
        await session._emit_error("invalid_request_error", str(err), None, "input_audio_format")
        return
    except IngestError as err:
        log.warning("audio ingest failed: %s", err)
        return
    capture = getattr(session, "capture_inbound_f32", None)
    if capture is not None and samples:
        try:
            capture(samples)
        except Exception as err:
            log.warning("inbound audio capture failed: %s", err)
    runner = getattr(session, "vad_runner", None)
    if runner is not None and samples:
        try:
            import numpy as np

            runner.push_samples(np.asarray(samples, dtype=np.float32))
        except Exception as err:
            log.warning("vad_runner.push_samples (ws) failed: %s", err)
