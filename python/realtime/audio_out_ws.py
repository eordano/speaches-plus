from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field

from . import audio_defaults
from .audio_out import OutboundPushError

log = logging.getLogger("realtime.audio_out_ws")

KOKORO_HZ = audio_defaults.TTS_SAMPLE_RATE
FRAME_MS = audio_defaults.FRAME_MS

def _wire_for(format: str) -> tuple[str, int]:
    table = {
        "pcm16_8k": ("pcm16le", 8_000),
        "pcm16_16k": ("pcm16le", 16_000),
        "pcm16_24k": ("pcm16le", 24_000),
        "pcm16_44k1": ("pcm16le", 44_100),
        "pcm16_48k": ("pcm16le", 48_000),
        "g711_ulaw": ("ulaw", 8_000),
        "g711_alaw": ("alaw", 8_000),
    }
    return table.get(format, ("pcm16le", 24_000))

@dataclass
class WsAudioPacer:
    ws_send: asyncio.Queue
    id_event_factory: callable
    played_ms_ref: list[int]
    codec: str
    sample_rate: int
    frame_samples: int
    frames_written: int = 0
    _start: float | None = None
    carry: list[float] = field(default_factory=list)
    last_sample: float = 0.0
    src_position: float = 0.0

    @classmethod
    def start(
        cls,
        ws_send: asyncio.Queue,
        id_event_factory: callable,
        played_ms_ref: list[int],
        format: str,
    ) -> "WsAudioPacer":
        codec, sr = _wire_for(format)
        frame_samples = sr * FRAME_MS // 1000
        return cls(
            ws_send=ws_send,
            id_event_factory=id_event_factory,
            played_ms_ref=played_ms_ref,
            codec=codec,
            sample_rate=sr,
            frame_samples=frame_samples,
        )

    async def play(self, audio_24k_samples: list[float]) -> None:
        if not audio_24k_samples:
            return
        if self.sample_rate == KOKORO_HZ:
            resampled = list(audio_24k_samples)
        else:
            resampled = self._linear_resample(audio_24k_samples)
        self.carry.extend(resampled)
        while len(self.carry) >= self.frame_samples:
            await self._emit_one_frame()

    async def flush(self) -> None:
        if not self.carry:
            return
        while len(self.carry) < self.frame_samples:
            self.carry.append(0.0)
        await self._emit_one_frame()

    async def _emit_one_frame(self) -> None:
        frame = self.carry[: self.frame_samples]
        self.carry = self.carry[self.frame_samples :]
        if self.codec == "pcm16le":
            buf = bytearray(self.frame_samples * 2)
            for i, s in enumerate(frame):
                v = int(max(-1.0, min(1.0, s)) * 32_767.0)
                if v < 0:
                    v += 0x10000
                buf[i * 2] = v & 0xFF
                buf[i * 2 + 1] = (v >> 8) & 0xFF
            payload = bytes(buf)
        elif self.codec in ("ulaw", "alaw"):
            from audio.g711 import alaw_encode_sample, ulaw_encode_sample

            enc = ulaw_encode_sample if self.codec == "ulaw" else alaw_encode_sample
            buf = bytearray(self.frame_samples)
            for i, s in enumerate(frame):
                v = int(max(-1.0, min(1.0, s)) * 32_767.0)
                buf[i] = enc(v)
            payload = bytes(buf)
        else:
            return

        event = {
            "event_id": self.id_event_factory(),
            "type": "response.output_audio.delta",
            "delta": base64.b64encode(payload).decode("ascii"),
        }
        try:
            text = json.dumps(event, separators=(",", ":"))
        except (TypeError, ValueError) as err:
            log.warning("audio.delta json serialize failed: %s", err)
            return
        try:
            await self.ws_send.put(text)
        except Exception as err:
            log.warning("ws writer dropped while sending audio.delta: %s", err)
            return

        if self._start is None:
            self._start = time.monotonic()
        self.frames_written += 1
        if self.played_ms_ref:
            self.played_ms_ref[0] = self.frames_written * FRAME_MS
        target = self._start + (self.frames_written * FRAME_MS) / 1000.0
        now = time.monotonic()
        if target > now:
            await asyncio.sleep(target - now)

    def _linear_resample(self, src: list[float]) -> list[float]:
        if not src:
            return []
        ratio = self.sample_rate / float(KOKORO_HZ)
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

@dataclass
class AudioPacer:
    webrtc: object | None = None
    websocket: WsAudioPacer | None = None

    @classmethod
    def from_webrtc(cls, p) -> "AudioPacer":
        return cls(webrtc=p)

    @classmethod
    def from_websocket(cls, p: WsAudioPacer) -> "AudioPacer":
        return cls(websocket=p)

    async def play(self, audio_24k_samples) -> None:
        if self.webrtc is not None:
            await self.webrtc.play(audio_24k_samples)
            return
        if self.websocket is not None:
            await self.websocket.play(audio_24k_samples)

    async def flush(self) -> None:
        if self.webrtc is not None:
            await self.webrtc.flush()
            return
        if self.websocket is not None:
            await self.websocket.flush()

__all__ = [
    "AudioPacer",
    "OutboundPushError",
    "WsAudioPacer",
]
