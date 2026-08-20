from __future__ import annotations

import numpy as np

PRE_EMPHASIS = 0.97
SAMPLE_RATE = 16_000.0
LOW_FREQ_HZ = 20.0
HIGH_FREQ_HZ = 7600.0
LOG_FLOOR = 1e-10
POVEY_EXP = 0.85

class FBank:
    def __init__(self, num_mels: int, frame_length: int, frame_shift: int):
        self.num_mels = num_mels
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.n_fft = _next_power_of_two(frame_length)
        self._window = _povey_window(frame_length)
        self._mel_filters = _build_mel_filters(
            num_mels, self.n_fft, SAMPLE_RATE, LOW_FREQ_HZ, HIGH_FREQ_HZ,
        )

    def compute(self, audio: np.ndarray) -> np.ndarray:
        if audio.shape[0] < self.frame_length:
            raise ValueError(
                f"fbank: audio too short ({audio.shape[0]} < {self.frame_length})"
            )
        num_frames = 1 + (audio.shape[0] - self.frame_length) // self.frame_shift
        out = np.zeros((num_frames, self.num_mels), dtype=np.float32)
        spectrum_bins = self.n_fft // 2 + 1
        for frame_i in range(num_frames):
            start = frame_i * self.frame_shift
            frame_buf = np.zeros(self.n_fft, dtype=np.float32)
            prev0 = audio[0] if start == 0 else audio[start - 1]
            frame_buf[0] = audio[start] - PRE_EMPHASIS * prev0
            window_slice = audio[start + 1 : start + self.frame_length]
            prev_slice = audio[start : start + self.frame_length - 1]
            frame_buf[1 : self.frame_length] = window_slice - PRE_EMPHASIS * prev_slice
            frame_buf[: self.frame_length] *= self._window
            spectrum = np.fft.rfft(frame_buf, n=self.n_fft)
            power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float32)
            for m, taps in enumerate(self._mel_filters):
                acc = 0.0
                for bin_idx, weight in taps:
                    if bin_idx < spectrum_bins:
                        acc += power[bin_idx] * weight
                out[frame_i, m] = float(np.log(max(acc, LOG_FLOOR)))
        _cmn_in_place(out)
        return out.reshape(-1)

def _next_power_of_two(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p

def _povey_window(n: int) -> np.ndarray:
    denom = max(n - 1, 1)
    indices = np.arange(n, dtype=np.float32)
    raised = 0.5 - 0.5 * np.cos(2.0 * np.pi * indices / denom)
    return np.maximum(raised, 0.0) ** POVEY_EXP

def _hz_to_mel(hz: float) -> float:
    return 1127.0 * float(np.log(1.0 + hz / 700.0))

def _mel_to_hz(mel: float) -> float:
    return 700.0 * (float(np.exp(mel / 1127.0)) - 1.0)

def _build_mel_filters(
    num_mels: int,
    n_fft: int,
    sample_rate: float,
    low_hz: float,
    high_hz: float,
) -> list[list[tuple[int, float]]]:
    num_bins = n_fft // 2 + 1
    low_mel = _hz_to_mel(low_hz)
    high_mel = _hz_to_mel(high_hz)
    mel_points = [
        _mel_to_hz(low_mel + (high_mel - low_mel) * i / (num_mels + 1))
        for i in range(num_mels + 2)
    ]
    bins = [hz * n_fft / sample_rate for hz in mel_points]
    filters: list[list[tuple[int, float]]] = [[] for _ in range(num_mels)]
    for m in range(num_mels):
        left = bins[m]
        center = bins[m + 1]
        right = bins[m + 2]
        lo = int(np.floor(left))
        hi = int(np.ceil(right))
        for k in range(lo, hi + 1):
            if k < 0 or k >= num_bins:
                continue
            kf = float(k)
            if kf < center:
                w = (kf - left) / (center - left) if center > left else 0.0
            elif kf <= right:
                w = (right - kf) / (right - center) if right > center else 0.0
            else:
                w = 0.0
            if w > 0.0:
                filters[m].append((k, float(w)))
    return filters

def _cmn_in_place(feats: np.ndarray) -> None:
    if feats.size == 0:
        return
    mean = feats.mean(axis=0)
    feats -= mean
