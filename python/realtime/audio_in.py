from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any

import numpy as np

from . import audio_defaults

log = logging.getLogger("realtime.audio_in")

_MAX_BUFFER_SAMPLES_16K = 16_000 * 30

def _load_opus_decoder_factory() -> Any:
    try:
        import opuslib

        return opuslib.Decoder
    except ImportError as err:
        raise RuntimeError(
            "AudioIngest requires the 'opuslib' package (>=3.0). "
            "Install it via `pip install opuslib` (libopus must be present)."
        ) from err

def _build_polyphase_kernel(up: int, down: int, half_taps: int = 32, beta: float = 8.6) -> np.ndarray:
    n_taps_per_phase = 2 * half_taps + 1
    n_taps = n_taps_per_phase * up
    cutoff = 1.0 / max(up, down)
    n = np.arange(n_taps, dtype=np.float64) - (n_taps - 1) / 2.0
    sinc = np.sinc(cutoff * n)
    window = np.kaiser(n_taps, beta)
    h = sinc * window
    h *= cutoff
    return h.astype(np.float32)

_RESAMPLE_KERNEL_48_TO_16: np.ndarray | None = None

def _polyphase_resample_48_to_16(mono_48k: np.ndarray, tail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    global _RESAMPLE_KERNEL_48_TO_16
    if _RESAMPLE_KERNEL_48_TO_16 is None:
        _RESAMPLE_KERNEL_48_TO_16 = _build_polyphase_kernel(up=1, down=3, half_taps=32, beta=8.6)
    h = _RESAMPLE_KERNEL_48_TO_16
    n_taps = h.shape[0]
    half = (n_taps - 1) // 2
    if tail.size:
        signal = np.concatenate([tail, mono_48k]).astype(np.float32, copy=False)
    else:
        signal = mono_48k.astype(np.float32, copy=False)
    if signal.size < n_taps:
        return np.empty(0, dtype=np.float32), signal
    convolved = np.convolve(signal, h, mode="valid")
    decimated = convolved[::3].astype(np.float32, copy=False)
    consumed = (decimated.shape[0] - 1) * 3 + 1
    new_tail_start = consumed
    new_tail = signal[new_tail_start:].astype(np.float32, copy=False)
    if new_tail.size > n_taps + 3:
        new_tail = new_tail[-(n_taps + 3) :].copy()
    return decimated, new_tail

def _resample_with_scipy(mono_48k: np.ndarray) -> np.ndarray | None:
    try:
        from scipy.signal import resample_poly
    except ImportError:
        return None
    out = resample_poly(mono_48k.astype(np.float32, copy=False), up=1, down=3, window=("kaiser", 8.6))
    return out.astype(np.float32, copy=False)

class AudioIngest:
    def __init__(self, channels: int, frame_samples: int | None = None, max_buffer_samples: int | None = None) -> None:
        if channels not in (1, 2):
            raise ValueError(f"unsupported opus channel count: {channels}")
        decoder_cls = _load_opus_decoder_factory()
        self.channels = channels
        self._opus_decoder = decoder_cls(audio_defaults.OPUS_SAMPLE_RATE_HZ, channels)
        self._frame_samples = int(frame_samples) if frame_samples else audio_defaults.MAX_DECODE_FRAMES
        self._buf_chunks: deque[np.ndarray] = deque()
        self._buf_size: int = 0
        self._tail_48k = np.empty(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._total_in_48k = 0
        self._total_out_16k = 0
        self._scipy_available: bool | None = None
        self._max_buffer_samples = int(max_buffer_samples) if max_buffer_samples else _MAX_BUFFER_SAMPLES_16K
        self.dropped_samples: int = 0

    def process_opus(self, payload: bytes, decode_fec: bool = False) -> None:
        if not payload:
            return
        try:
            pcm_bytes = self._opus_decoder.decode(
                bytes(payload), self._frame_samples, decode_fec=decode_fec
            )
        except Exception as err:
            log.warning("opus decode failed: %s", err)
            return
        arr = np.frombuffer(pcm_bytes, dtype=np.int16)
        if self.channels == 2:
            if arr.size % 2 != 0:
                log.warning("opus stereo frame produced odd sample count: %d", arr.size)
                arr = arr[: arr.size - (arr.size % 2)]
            stereo = arr.reshape(-1, 2).astype(np.float32)
            mono = stereo.mean(axis=1) / 32768.0
        else:
            mono = arr.astype(np.float32) / 32768.0
        self._ingest_mono_48k(mono)

    def process_av_frame(self, frame: Any) -> None:
        sample_rate = int(getattr(frame, "sample_rate", audio_defaults.OPUS_SAMPLE_RATE_HZ))
        layout = getattr(frame, "layout", None)
        layout_name = getattr(layout, "name", None) if layout is not None else None
        ndarr = frame.to_ndarray()
        if not isinstance(ndarr, np.ndarray):
            ndarr = np.asarray(ndarr)
        mono_native = self._frame_to_mono_float32(ndarr, layout_name)
        if sample_rate == int(audio_defaults.INPUT_HZ):
            self._ingest_mono_48k(mono_native)
            return
        resampled = self._resample_arbitrary(mono_native, sample_rate, int(audio_defaults.INPUT_HZ))
        self._ingest_mono_48k(resampled)

    def process(self, opus_payload: bytes) -> None:
        self.process_opus(opus_payload)

    def take(self) -> list[float]:
        return self.take_array().tolist()

    def take_array(self) -> np.ndarray:
        with self._lock:
            chunks = self._buf_chunks
            self._buf_chunks = deque()
            self._buf_size = 0
        if not chunks:
            return np.empty(0, dtype=np.float32)
        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(list(chunks))

    def get_total_samples_consumed(self) -> int:
        with self._lock:
            return self._total_out_16k

    def get_total_input_samples(self) -> int:
        with self._lock:
            return self._total_in_48k

    @staticmethod
    def _frame_to_mono_float32(ndarr: np.ndarray, layout_name: str | None) -> np.ndarray:
        if ndarr.dtype == np.int16:
            data = ndarr.astype(np.float32) / 32768.0
        elif ndarr.dtype == np.int32:
            data = ndarr.astype(np.float32) / 2147483648.0
        elif ndarr.dtype == np.uint8:
            data = (ndarr.astype(np.float32) - 128.0) / 128.0
        elif ndarr.dtype in (np.float32, np.float64):
            data = ndarr.astype(np.float32, copy=False)
        else:
            data = ndarr.astype(np.float32)
        if data.ndim == 1:
            if layout_name and "stereo" in layout_name:
                if data.size % 2 == 0:
                    data = data.reshape(-1, 2).mean(axis=1)
            return data.astype(np.float32, copy=False)
        if data.ndim == 2:
            if data.shape[0] <= 8 and data.shape[1] >= data.shape[0]:
                return data.mean(axis=0).astype(np.float32, copy=False)
            return data.mean(axis=1).astype(np.float32, copy=False)
        return data.reshape(-1).astype(np.float32, copy=False)

    def _ingest_mono_48k(self, mono_48k: np.ndarray) -> None:
        if mono_48k.size == 0:
            return
        with self._lock:
            self._total_in_48k += int(mono_48k.size)
            decimated, new_tail = self._do_resample(mono_48k)
            self._tail_48k = new_tail
            if decimated.size:
                self._buf_chunks.append(decimated)
                self._buf_size += int(decimated.size)
                self._total_out_16k += int(decimated.size)
                self._enforce_buffer_cap_locked()

    def _enforce_buffer_cap_locked(self) -> None:
        cap = self._max_buffer_samples
        if cap <= 0 or self._buf_size <= cap:
            return
        dropped = 0
        while self._buf_chunks and self._buf_size > cap:
            head = self._buf_chunks[0]
            head_n = int(head.shape[0])
            overflow = self._buf_size - cap
            if head_n <= overflow:
                self._buf_chunks.popleft()
                self._buf_size -= head_n
                dropped += head_n
            else:
                trimmed = head[overflow:]
                self._buf_chunks[0] = trimmed
                self._buf_size -= overflow
                dropped += overflow
        if dropped:
            self.dropped_samples += dropped
            log.warning("AudioIngest buffer cap %d exceeded; dropped %d oldest samples (total dropped=%d)", cap, dropped, self.dropped_samples)

    def _do_resample(self, mono_48k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._scipy_available is None:
            self._scipy_available = _resample_with_scipy(np.zeros(96, dtype=np.float32)) is not None
        if self._scipy_available:
            joined = (
                np.concatenate([self._tail_48k, mono_48k])
                if self._tail_48k.size
                else mono_48k
            )
            chunk_size = 9600
            if joined.size < chunk_size:
                return np.empty(0, dtype=np.float32), joined.astype(np.float32, copy=False)
            usable = (joined.size // 3) * 3
            head = joined[:usable]
            new_tail = joined[usable:].astype(np.float32, copy=False)
            out = _resample_with_scipy(head)
            assert out is not None
            return out, new_tail
        return _polyphase_resample_48_to_16(mono_48k, self._tail_48k)

    @staticmethod
    def _resample_arbitrary(signal: np.ndarray, src_hz: int, dst_hz: int) -> np.ndarray:
        if signal.size == 0 or src_hz == dst_hz:
            return signal.astype(np.float32, copy=False)
        try:
            from scipy.signal import resample_poly
            from math import gcd

            g = gcd(src_hz, dst_hz)
            up = dst_hz // g
            down = src_hz // g
            return resample_poly(signal.astype(np.float32, copy=False), up=up, down=down).astype(
                np.float32, copy=False
            )
        except ImportError:
            pass
        try:
            import librosa

            return librosa.resample(
                signal.astype(np.float32, copy=False),
                orig_sr=src_hz,
                target_sr=dst_hz,
                res_type="kaiser_best",
            ).astype(np.float32, copy=False)
        except ImportError:
            pass
        n_out = int(round(signal.size * dst_hz / src_hz))
        if n_out <= 0:
            return np.empty(0, dtype=np.float32)
        x_in = np.linspace(0.0, signal.size - 1.0, n_out)
        return np.interp(x_in, np.arange(signal.size), signal).astype(np.float32)
