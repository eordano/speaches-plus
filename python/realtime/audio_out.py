from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from scipy.signal import resample_poly as _scipy_resample_poly
except ImportError:
    _scipy_resample_poly = None

from . import audio_defaults, wire_defaults

log = logging.getLogger("realtime.audio_out")

OUT_SAMPLE_RATE = audio_defaults.OUT_SAMPLE_RATE
FRAME_MS = audio_defaults.FRAME_MS
FRAME_SAMPLES = audio_defaults.FRAME_SAMPLES
OPUS_SAMPLE_RATE_HZ = audio_defaults.OPUS_SAMPLE_RATE_HZ
DEFAULT_OUTBOUND_QUEUE_CAP_MS = wire_defaults.OUTBOUND_QUEUE_CAP_MS
DEFAULT_OUTBOUND_QUEUE_CAP_EVENTS = wire_defaults.OUTBOUND_QUEUE_CAP_EVENTS

OPUS_BITRATE_BPS = 64_000
OPUS_COMPLEXITY = 5

class OutboundPushError(Exception):
    pass

class QueueFull(OutboundPushError):
    def __init__(self, queued_ms: int, cap_ms: int):
        self.queued_ms = queued_ms
        self.cap_ms = cap_ms
        super().__init__(
            f"outbound queue cap exceeded ({queued_ms} ms buffered, cap {cap_ms} ms)"
        )

@dataclass
class QueueGate:
    cap_ms: int
    queued_ms: int = 0

    def try_push(self, chunk_ms: int) -> None:
        projected = self.queued_ms + chunk_ms
        if projected > self.cap_ms:
            raise QueueFull(queued_ms=projected, cap_ms=self.cap_ms)
        self.queued_ms = projected

    def on_frame_sent(self) -> None:
        self.queued_ms = max(0, self.queued_ms - FRAME_MS)

    def reset(self) -> None:
        self.queued_ms = 0

def read_queue_cap_ms_from_env() -> int:
    import env as env_mod

    return env_mod.read_int(env_mod.OUTBOUND_QUEUE_CAP_MS, DEFAULT_OUTBOUND_QUEUE_CAP_MS)

def read_queue_cap_events_from_env() -> int:
    import env as env_mod

    return env_mod.read_int(env_mod.OUTBOUND_QUEUE_CAP, DEFAULT_OUTBOUND_QUEUE_CAP_EVENTS)

def _resample_24k_to_48k(arr):
    if _scipy_resample_poly is not None:
        return _scipy_resample_poly(arr, up=2, down=1).astype(np.float32)
    try:
        import librosa

        return librosa.resample(
            arr,
            orig_sr=audio_defaults.TTS_SAMPLE_RATE,
            target_sr=OUT_SAMPLE_RATE,
        ).astype(np.float32)
    except ImportError:
        ratio = OUT_SAMPLE_RATE / audio_defaults.TTS_SAMPLE_RATE
        n_out = int(round(len(arr) * ratio))
        x_in = np.linspace(0, len(arr) - 1, n_out)
        return np.interp(x_in, np.arange(len(arr)), arr).astype(np.float32)

def _f32_to_s16le_bytes(frame) -> bytes:
    arr = np.asarray(frame, dtype=np.float32)
    arr = np.clip(arr, -1.0, 1.0)
    v = np.rint(arr * 32767.0).astype(np.int32)
    v = np.clip(v, -32768, 32767).astype("<i2")
    return v.tobytes()

class OutboundPacer:
    def __init__(self, track: Any, played_ms_ref: list[int], queue_cap_ms: int):
        self.track = track
        self.played_ms_ref = played_ms_ref
        self.gate = QueueGate(cap_ms=queue_cap_ms)
        self.frames_written = 0
        self._start: float | None = None
        self._cancelled = False
        self._capture: Any = None
        self._last_pcm_bytes: bytes | None = None
        self._frame_callback: Any = None

    def attach_capture(self, capture: Any) -> None:
        self._capture = capture

    def attach_frame_callback(self, callback: Any) -> None:
        self._frame_callback = callback

    async def play(self, audio_24k_samples) -> None:
        if self._cancelled:
            return
        if audio_24k_samples is None:
            return
        if isinstance(audio_24k_samples, np.ndarray):
            if audio_24k_samples.size == 0:
                return
        elif not audio_24k_samples:
            return
        if self._capture is not None:
            try:
                self._capture(audio_24k_samples)
            except Exception as err:
                log.warning("outbound audio capture failed: %s", err)
        chunk_ms = int(len(audio_24k_samples) * 1000 / audio_defaults.TTS_SAMPLE_RATE)
        self.gate.try_push(chunk_ms)
        arr = np.asarray(audio_24k_samples, dtype=np.float32)
        resampled = _resample_24k_to_48k(arr)
        cursor = 0
        while cursor + FRAME_SAMPLES <= len(resampled):
            if self._cancelled:
                return
            frame = resampled[cursor : cursor + FRAME_SAMPLES]
            await self._write_pcm_frame(frame)
            cursor += FRAME_SAMPLES

    async def flush(self) -> None:
        if self._cancelled:
            return
        if self.frames_written == 0:
            return
        silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        try:
            await self._write_pcm_frame(silence)
        except OutboundPushError as err:
            log.debug("flush: tail-silence push skipped: %s", err)
        end = getattr(self.track, "end_of_stream", None)
        if callable(end):
            try:
                res = end()
                if asyncio.iscoroutine(res):
                    await res
            except Exception as err:
                log.debug("track.end_of_stream raised: %s", err)

    def cancel(self) -> None:
        self._cancelled = True
        self.gate.reset()
        if self.played_ms_ref:
            self.played_ms_ref[0] = self.frames_written * FRAME_MS
        drop = getattr(self.track, "drop_queued", None)
        if callable(drop):
            try:
                drop()
            except Exception as err:
                log.debug("track.drop_queued raised: %s", err)

    def last_pcm_bytes(self) -> bytes | None:
        return self._last_pcm_bytes

    async def _write_pcm_frame(self, frame) -> None:
        if self._cancelled:
            return
        pcm_bytes = _f32_to_s16le_bytes(frame)
        self._last_pcm_bytes = pcm_bytes
        if self._frame_callback is not None:
            try:
                res = self._frame_callback(pcm_bytes)
                if asyncio.iscoroutine(res):
                    await res
            except Exception as err:
                log.debug("frame_callback raised: %s", err)
        push = getattr(self.track, "push_opus_frame", None)
        if callable(push):
            res = push(pcm_bytes, FRAME_MS)
            if asyncio.iscoroutine(res):
                await res
        else:
            queue = getattr(self.track, "queue", None)
            if queue is not None:
                put = getattr(queue, "put", None)
                if put is not None:
                    res = put((pcm_bytes, FRAME_MS))
                    if asyncio.iscoroutine(res):
                        await res
                else:
                    queue.append((pcm_bytes, FRAME_MS))
            else:
                push_any = getattr(self.track, "push", None)
                if callable(push_any):
                    res = push_any(pcm_bytes)
                    if asyncio.iscoroutine(res):
                        await res
        if self._start is None:
            self._start = time.monotonic()
        self.frames_written += 1
        if self.played_ms_ref is not None and self.played_ms_ref:
            self.played_ms_ref[0] = self.frames_written * FRAME_MS
        self.gate.on_frame_sent()
        target = self._start + (self.frames_written * FRAME_MS) / 1000.0
        now = time.monotonic()
        if target > now:
            try:
                await asyncio.sleep(target - now)
            except asyncio.CancelledError:
                self._cancelled = True
                raise

__all__ = [
    "DEFAULT_OUTBOUND_QUEUE_CAP_EVENTS",
    "DEFAULT_OUTBOUND_QUEUE_CAP_MS",
    "FRAME_MS",
    "FRAME_SAMPLES",
    "OPUS_SAMPLE_RATE_HZ",
    "OUT_SAMPLE_RATE",
    "OutboundPacer",
    "OutboundPushError",
    "QueueFull",
    "QueueGate",
    "read_queue_cap_events_from_env",
    "read_queue_cap_ms_from_env",
]
