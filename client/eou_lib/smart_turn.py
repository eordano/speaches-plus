"""smart-turn-v3 mel preprocessing + onnxruntime wrapper, ported from
go/internal/eou/audio.go and rust/src/realtime/eou_audio.rs.

The mel pipeline is byte-close to both ports (parity asserted in the Go
golden test). Use this from any Python tool that needs to score audio
without spinning up a server.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP = 160
TARGET_SAMPLES = 8 * SAMPLE_RATE
N_FRAMES = TARGET_SAMPLES // HOP

def _hz_to_mel(f: float) -> float:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = np.log(6.4) / 27.0
    return min_log_mel + np.log(f / min_log_hz) / log_step if f >= min_log_hz else f / f_sp

def _mel_to_hz(m: float) -> float:
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    log_step = np.log(6.4) / 27.0
    return min_log_hz * np.exp((m - min_log_mel) * log_step) if m >= min_log_mel else f_sp * m

def _build_mel_filters() -> np.ndarray:
    n_bins = N_FFT // 2 + 1
    m_min, m_max = _hz_to_mel(0.0), _hz_to_mel(SAMPLE_RATE / 2.0)
    mel_pts = np.array([m_min + (m_max - m_min) * i / (N_MELS + 1) for i in range(N_MELS + 2)])
    hz_pts = np.array([_mel_to_hz(m) for m in mel_pts])
    fft_freqs = np.array([i * SAMPLE_RATE / N_FFT for i in range(n_bins)])
    filters = np.zeros((N_MELS, n_bins), dtype=np.float32)
    for m in range(N_MELS):
        lo, ce, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        lower_slope = max(ce - lo, 1e-30)
        upper_slope = max(hi - ce, 1e-30)
        enorm = 2.0 / max(hi - lo, 1e-30)
        for k, freq in enumerate(fft_freqs):
            w = 0.0
            if lo <= freq <= ce:
                w = (freq - lo) / lower_slope
            elif ce < freq <= hi:
                w = (hi - freq) / upper_slope
            filters[m, k] = w * enorm
    return filters

_HANN = (0.5 * (1 - np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT))).astype(np.float32)
_MEL_FILTERS = _build_mel_filters()

def prepare_audio(audio: np.ndarray, audio_window_ms: int = 8000,
                   pad_alignment: str = "leading") -> np.ndarray:
    """Trim/pad/clamp to exactly TARGET_SAMPLES (8 s @ 16 kHz)."""
    target = TARGET_SAMPLES
    max_window = min(audio_window_ms * SAMPLE_RATE // 1000, target)
    src = audio[-max_window:] if len(audio) > max_window else audio
    out = np.zeros(target, dtype=np.float32)
    if len(src) >= target:
        out[:] = src[-target:]
    elif pad_alignment == "trailing":
        out[: len(src)] = src
    else:
        out[target - len(src):] = src
    out = np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    return out

def log_mel(audio: np.ndarray) -> np.ndarray:
    """Reflective-pad -> STFT -> mel multiply -> log10 -> max-anchored
    floor + (v+4)/4 normalize. Returns a [N_MELS, N_FRAMES] f32 array.
    """
    pad = N_FFT // 2
    n = len(audio)
    padded = np.empty(n + N_FFT, dtype=np.float32)
    for i in range(pad):
        padded[i] = audio[pad - i]
    padded[pad:pad + n] = audio
    for i in range(pad):
        src = max(0, n - 2 - i)
        padded[pad + n + i] = audio[src]
    n_bins = N_FFT // 2 + 1
    power = np.empty((N_FRAMES, n_bins), dtype=np.float32)
    for f in range(N_FRAMES):
        start = f * HOP
        frame = padded[start:start + N_FFT].astype(np.float64) * _HANN
        spec = np.fft.rfft(frame)
        power[f] = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)
    mel = (_MEL_FILTERS @ power.T).astype(np.float32)
    eps = 1e-10
    log10 = np.log10(np.maximum(mel, eps))
    floor = log10.max() - 8.0
    log10 = np.maximum(log10, floor)
    return ((log10 + 4.0) / 4.0).astype(np.float32)

@dataclass
class SmartTurn:
    """onnxruntime wrapper around smart-turn-v3. Construct once, call
    .score(audio_f32_16k) per request -- single-threaded, holds an
    InferenceSession.
    """
    session: object
    audio_window_ms: int = 8000
    pad_alignment: str = "leading"

    @classmethod
    def load(cls, model_path: str, audio_window_ms: int = 8000,
              pad_alignment: str = "leading", intra_op_num_threads: int = 1) -> "SmartTurn":
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op_num_threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(model_path, sess_options=so,
                                     providers=["CPUExecutionProvider"])
        return cls(session=sess, audio_window_ms=audio_window_ms,
                    pad_alignment=pad_alignment)

    def score(self, audio_f32: np.ndarray) -> float:
        prepared = prepare_audio(audio_f32, self.audio_window_ms, self.pad_alignment)
        mel = log_mel(prepared)[None, ...].astype(np.float32)
        out = self.session.run(["logits"], {"input_features": mel})
        raw = float(out[0].flatten()[0])
        if not np.isfinite(raw):
            return float("nan")
        if 0.0 <= raw <= 1.0:
            return raw
        return 1.0 / (1.0 + np.exp(-raw))
