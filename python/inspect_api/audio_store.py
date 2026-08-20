from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO

import numpy as np

logger = logging.getLogger(__name__)

class Channel(Enum):
    MIC_IN = "mic_in"
    TTS_OUT = "tts_out"

    @classmethod
    def parse(cls, s: str) -> Channel | None:
        if s == "mic_in":
            return cls.MIC_IN
        if s == "tts_out":
            return cls.TTS_OUT
        return None

    def as_str(self) -> str:
        return self.value

    def sample_rate(self) -> int:
        if self is Channel.MIC_IN:
            return 16_000
        return 24_000

@dataclass
class _TrackState:
    fh: BinaryIO | None
    first_ns: int | None
    total_samples: int

class _Track:
    def __init__(self, session_id: str, channel: Channel, directory: Path) -> None:
        self.channel = channel
        self.path = directory / f"{session_id}.audio_{channel.as_str()}.raw"
        self._lock = threading.Lock()
        fh: BinaryIO | None
        try:
            fh = open(self.path, "ab")
        except OSError as err:
            logger.warning("open audio track %s: %s", self.path, err)
            fh = None
        self._state = _TrackState(fh=fh, first_ns=None, total_samples=0)

    def append_pcm16(self, pcm: bytes, session_start_ns: int) -> None:
        if not pcm:
            return
        with self._lock:
            if self._state.first_ns is None:
                self._state.first_ns = time.monotonic_ns() - session_start_ns
            if self._state.fh is None:
                return
            try:
                self._state.fh.write(pcm)
            except OSError as err:
                logger.warning("write audio track: %s", err)
                return
            self._state.total_samples += len(pcm) // 2

    def append_f32(self, samples, session_start_ns: int) -> None:
        if isinstance(samples, (bytes, bytearray)):
            if not samples:
                return
            self.append_pcm16(bytes(samples), session_start_ns)
            return
        if isinstance(samples, np.ndarray):
            if samples.size == 0:
                return
            arr = samples
        else:
            if not samples:
                return
            arr = np.asarray(samples, dtype=np.float32)
            if arr.size == 0:
                return
        arr = np.asarray(arr, dtype=np.float32)
        clipped = np.clip(arr, -1.0, 1.0)
        pcm = np.rint(clipped * 32767.0).astype("<i2").tobytes()
        self.append_pcm16(pcm, session_start_ns)

    def offset_ms(self, session_start_ns: int) -> int:
        with self._lock:
            first = self._state.first_ns
        if first is None:
            return 0
        diff = max(0, first)
        return diff // 1_000_000

    def slice(self, from_ms: int, to_ms: int) -> bytes:
        sr = self.channel.sample_rate()
        byte_offset = (from_ms * sr * 2) // 1000
        end_offset = (to_ms * sr * 2) // 1000 if to_ms > 0 else None
        with self._lock:
            try:
                with open(self.path, "rb") as fh:
                    fh.seek(byte_offset)
                    if end_offset is None:
                        return fh.read()
                    n = max(0, end_offset - byte_offset)
                    return fh.read(n)
            except OSError as err:
                logger.warning("open audio track for slice %s: %s", self.path, err)
                return b""

    def close(self) -> None:
        with self._lock:
            if self._state.fh is not None:
                try:
                    self._state.fh.flush()
                    self._state.fh.close()
                except OSError:
                    pass
                self._state.fh = None

    def total_samples(self) -> int:
        with self._lock:
            return self._state.total_samples

class AudioStore:
    def __init__(self, session_id: str, session_dir: Path) -> None:
        if session_dir is None:
            raise ValueError("AudioStore requires a non-None session_dir; pass INSPECT_SESSION_DIR or skip audio capture")
        self.session_id = session_id
        self.session_dir = session_dir
        try:
            session_dir.mkdir(parents=True, exist_ok=True)
        except OSError as err:
            logger.warning("create audio session dir %s: %s", session_dir, err)
        self.session_start_wall = time.time()
        self.session_start_ns = time.monotonic_ns()
        self._mic_in = _Track(session_id, Channel.MIC_IN, session_dir)
        self._tts_out = _Track(session_id, Channel.TTS_OUT, session_dir)

    def append_mic_in_f32(self, samples) -> None:
        self._mic_in.append_f32(samples, self.session_start_ns)

    def append_tts_out_f32(self, samples) -> None:
        self._tts_out.append_f32(samples, self.session_start_ns)

    def append_tts_out_pcm16(self, pcm: bytes) -> None:
        self._tts_out.append_pcm16(pcm, self.session_start_ns)

    def append_mic_in_pcm16(self, pcm: bytes) -> None:
        self._mic_in.append_pcm16(pcm, self.session_start_ns)

    def _track(self, channel: Channel) -> _Track:
        if channel is Channel.MIC_IN:
            return self._mic_in
        return self._tts_out

    def track_offset_ms(self, channel: Channel) -> int:
        return self._track(channel).offset_ms(self.session_start_ns)

    def slice(self, channel: Channel, from_ms: int, to_ms: int) -> bytes:
        track = self._track(channel)
        offset = track.offset_ms(self.session_start_ns)
        adj_from = max(0, from_ms - offset)
        adj_to = max(0, to_ms - offset) if to_ms > 0 else 0
        return track.slice(adj_from, adj_to)

    def close(self) -> None:
        if self.session_dir is not None:
            sidecar = self.session_dir / f"{self.session_id}.audio.json"
            body = {
                "session_id": self.session_id,
                "started_at": self.session_start_wall,
                "tracks": {
                    "mic_in": {
                        "sample_rate": Channel.MIC_IN.sample_rate(),
                        "samples": self._mic_in.total_samples(),
                        "offset_ms": self._mic_in.offset_ms(self.session_start_ns),
                    },
                    "tts_out": {
                        "sample_rate": Channel.TTS_OUT.sample_rate(),
                        "samples": self._tts_out.total_samples(),
                        "offset_ms": self._tts_out.offset_ms(self.session_start_ns),
                    },
                },
            }
            try:
                sidecar.write_text(json.dumps(body, separators=(",", ":")))
            except OSError as err:
                logger.warning("write audio sidecar %s: %s", sidecar, err)
        self._mic_in.close()
        self._tts_out.close()

def wav_header(num_samples: int, sample_rate: int) -> bytes:
    byte_rate = sample_rate * 2
    block_align = 2
    data_bytes = num_samples * 2
    buf = bytearray()
    buf.extend(b"RIFF")
    buf.extend((36 + data_bytes).to_bytes(4, "little"))
    buf.extend(b"WAVE")
    buf.extend(b"fmt ")
    buf.extend((16).to_bytes(4, "little"))
    buf.extend((1).to_bytes(2, "little"))
    buf.extend((1).to_bytes(2, "little"))
    buf.extend(sample_rate.to_bytes(4, "little"))
    buf.extend(byte_rate.to_bytes(4, "little"))
    buf.extend(block_align.to_bytes(2, "little"))
    buf.extend((16).to_bytes(2, "little"))
    buf.extend(b"data")
    buf.extend(data_bytes.to_bytes(4, "little"))
    return bytes(buf)
