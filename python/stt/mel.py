from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .constants import (
    DEFAULT_N_MELS,
    WHISPER_HOP_LENGTH,
    WHISPER_NB_FRAMES,
    WHISPER_NFFT,
    WHISPER_PAD_SAMPLES,
    WHISPER_SAMPLING_HZ,
)

N_FRAMES = WHISPER_NB_FRAMES
N_FFT = WHISPER_NFFT
HOP_LENGTH = WHISPER_HOP_LENGTH
SAMPLE_RATE = WHISPER_SAMPLING_HZ
TARGET_SAMPLES = WHISPER_PAD_SAMPLES

_MEL_F_SP = 200.0 / 3.0
_MEL_MIN_LOG_HZ = 1_000.0

def _mel_min_log_mel() -> float:
    return _MEL_MIN_LOG_HZ / _MEL_F_SP

def _mel_logstep() -> float:
    return math.log(6.4) / 27.0

def hz_to_mel(f: float) -> float:
    if f >= _MEL_MIN_LOG_HZ:
        return _mel_min_log_mel() + math.log(f / _MEL_MIN_LOG_HZ) / _mel_logstep()
    return f / _MEL_F_SP

def mel_to_hz(m: float) -> float:
    if m >= _mel_min_log_mel():
        return _MEL_MIN_LOG_HZ * math.exp((m - _mel_min_log_mel()) * _mel_logstep())
    return _MEL_F_SP * m

def pad_or_truncate_to_30s(audio: np.ndarray) -> np.ndarray:
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    out = np.zeros(TARGET_SAMPLES, dtype=np.float32)
    n = min(len(audio), TARGET_SAMPLES)
    out[:n] = audio[:n]
    return out

def build_hann_window() -> np.ndarray:
    i = np.arange(N_FFT, dtype=np.float64)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / float(N_FFT))
    return w.astype(np.float32)

def build_mel_filters(n_mels: int, n_fft: int = N_FFT, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n_bins = n_fft // 2 + 1
    f_min = 0.0
    f_max = float(sample_rate) / 2.0
    m_min = hz_to_mel(f_min)
    m_max = hz_to_mel(f_max)

    mel_points = np.zeros(n_mels + 2, dtype=np.float64)
    for i in range(n_mels + 2):
        frac = float(i) / (float(n_mels) + 1.0)
        mel_points[i] = m_min + (m_max - m_min) * frac
    hz_points = np.array([mel_to_hz(float(m)) for m in mel_points], dtype=np.float64)

    fft_freqs = np.arange(n_bins, dtype=np.float64) * float(sample_rate) / float(n_fft)

    filters = np.zeros((n_mels, n_bins), dtype=np.float32)
    eps = float(np.finfo(np.float32).eps)
    for m in range(n_mels):
        lower = float(hz_points[m])
        center = float(hz_points[m + 1])
        upper = float(hz_points[m + 2])
        lower_slope = max(center - lower, eps)
        upper_slope = max(upper - center, eps)
        enorm = 2.0 / max(upper - lower, eps)
        for k in range(n_bins):
            freq = float(fft_freqs[k])
            weight = 0.0
            if lower <= freq <= center:
                weight = (freq - lower) / lower_slope
            elif center < freq <= upper:
                weight = (upper - freq) / upper_slope
            filters[m, k] = float(weight * enorm)
    return filters

@dataclass
class MelExtractor:
    n_mels: int = DEFAULT_N_MELS

    def __post_init__(self) -> None:
        self._filters = build_mel_filters(self.n_mels)
        self._hann = build_hann_window()

    @property
    def filters(self) -> np.ndarray:
        return self._filters

    @property
    def hann(self) -> np.ndarray:
        return self._hann

    def log_mel(self, audio_30s: np.ndarray) -> np.ndarray:
        if audio_30s.dtype != np.float32:
            audio_30s = audio_30s.astype(np.float32)
        if len(audio_30s) != TARGET_SAMPLES:
            raise ValueError(f"expected {TARGET_SAMPLES} samples (30s @ 16kHz), got {len(audio_30s)}")

        pad = N_FFT // 2
        padded = np.zeros(len(audio_30s) + N_FFT, dtype=np.float32)
        for i in range(pad):
            padded[i] = audio_30s[pad - i]
        padded[pad:pad + len(audio_30s)] = audio_30s
        for i in range(pad):
            src = max(len(audio_30s) - 2 - i, 0)
            padded[pad + len(audio_30s) + i] = audio_30s[src]

        n_bins = N_FFT // 2 + 1
        frame_starts = np.arange(N_FRAMES) * HOP_LENGTH
        frames = np.empty((N_FRAMES, N_FFT), dtype=np.float32)
        for i, start in enumerate(frame_starts):
            frames[i] = padded[start:start + N_FFT] * self._hann
        spec = np.fft.rfft(frames, n=N_FFT, axis=1)
        power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)

        mel = self._filters @ power.T

        eps = 1e-10
        mel = np.maximum(mel, eps)
        mel = np.log10(mel)
        max_val = float(mel.max())
        floor = max_val - 8.0
        mel = np.maximum(mel, floor)
        mel = (mel + 4.0) / 4.0
        return mel.astype(np.float32)

    def extract(self, samples: np.ndarray) -> np.ndarray:
        return self.log_mel(pad_or_truncate_to_30s(samples))

def log_mel_spectrogram(audio: np.ndarray, n_mels: int = DEFAULT_N_MELS) -> np.ndarray:
    return MelExtractor(n_mels=n_mels).extract(audio)
