from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Optional, Union

import numpy as np

import env as env_keys

from . import constants
from .types import AudioPadAlignment, EouModel

SAMPLE_RATE = constants.AUDIO_SAMPLE_RATE
N_MELS = constants.AUDIO_N_MELS
N_FFT = constants.AUDIO_N_FFT
HOP_LENGTH = constants.AUDIO_HOP_LENGTH
CHUNK_LENGTH_S = constants.AUDIO_CHUNK_LENGTH_S
TARGET_SAMPLES = constants.AUDIO_TARGET_SAMPLES
N_FRAMES = constants.AUDIO_N_FRAMES

_F32_EPSILON = 1.1920929e-7

MEL_F_SP = 200.0 / 3.0
MEL_MIN_LOG_HZ = 1000.0

def _mel_min_log_mel() -> float:
    return MEL_MIN_LOG_HZ / MEL_F_SP

def _mel_logstep() -> float:
    return math.log(6.4) / 27.0

def hz_to_mel(f: float) -> float:
    if f >= MEL_MIN_LOG_HZ:
        return _mel_min_log_mel() + math.log(f / MEL_MIN_LOG_HZ) / _mel_logstep()
    return f / MEL_F_SP

def mel_to_hz(m: float) -> float:
    if m >= _mel_min_log_mel():
        return MEL_MIN_LOG_HZ * math.exp((m - _mel_min_log_mel()) * _mel_logstep())
    return MEL_F_SP * m

def build_hann_window() -> np.ndarray:
    w = np.zeros(N_FFT, dtype=np.float32)
    for i in range(N_FFT):
        phase = 2.0 * math.pi * float(i) / float(N_FFT)
        w[i] = 0.5 - 0.5 * math.cos(phase)
    return w

def build_mel_filters() -> np.ndarray:
    n_bins = N_FFT // 2 + 1
    f_min = 0.0
    f_max = float(SAMPLE_RATE) / 2.0
    m_min = hz_to_mel(f_min)
    m_max = hz_to_mel(f_max)
    mel_points = np.zeros(N_MELS + 2, dtype=np.float32)
    for i in range(N_MELS + 2):
        frac = float(i) / (float(N_MELS) + 1.0)
        mel_points[i] = m_min + (m_max - m_min) * frac
    hz_points = np.array([mel_to_hz(float(m)) for m in mel_points], dtype=np.float32)
    fft_freqs = np.zeros(n_bins, dtype=np.float32)
    for i in range(n_bins):
        fft_freqs[i] = float(i) * float(SAMPLE_RATE) / float(N_FFT)
    filters = np.zeros(N_MELS * n_bins, dtype=np.float32)
    for m in range(N_MELS):
        lower = float(hz_points[m])
        center = float(hz_points[m + 1])
        upper = float(hz_points[m + 2])
        lower_slope = max(center - lower, _F32_EPSILON)
        upper_slope = max(upper - center, _F32_EPSILON)
        enorm = 2.0 / max(upper - lower, _F32_EPSILON)
        for k in range(n_bins):
            freq = float(fft_freqs[k])
            weight = 0.0
            if lower <= freq <= center:
                weight = (freq - lower) / lower_slope
            elif center < freq <= upper:
                weight = (upper - freq) / upper_slope
            filters[m * n_bins + k] = weight * enorm
    return filters

def prepare_audio(
    audio: np.ndarray,
    audio_window_ms: int,
    pad_alignment: AudioPadAlignment,
) -> np.ndarray:
    target = TARGET_SAMPLES
    max_window = (int(audio_window_ms) * int(SAMPLE_RATE)) // 1000
    if max_window > target:
        max_window = target
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] > max_window:
        arr = arr[arr.shape[0] - max_window :]
    arr = arr.copy()
    if arr.shape[0] < target:
        pad = target - arr.shape[0]
        if pad_alignment is AudioPadAlignment.LEADING:
            arr = np.concatenate([np.zeros(pad, dtype=np.float32), arr])
        else:
            arr = np.concatenate([arr, np.zeros(pad, dtype=np.float32)])
    elif arr.shape[0] > target:
        arr = arr[arr.shape[0] - target :]
    out = np.empty(target, dtype=np.float32)
    for i in range(target):
        v = float(arr[i])
        if not math.isfinite(v):
            out[i] = 0.0
        elif v > 1.0:
            out[i] = 1.0
        elif v < -1.0:
            out[i] = -1.0
        else:
            out[i] = v
    return out

def log_mel_spectrogram(
    audio: np.ndarray, hann: np.ndarray, mel_filters: np.ndarray
) -> np.ndarray:
    assert hann.shape[0] == N_FFT
    n_bins = N_FFT // 2 + 1
    assert mel_filters.shape[0] == N_MELS * n_bins

    pad = N_FFT // 2
    audio_arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = audio_arr.shape[0]
    padded = np.zeros(n + N_FFT, dtype=np.float32)
    for i in range(pad):
        padded[i] = float(audio_arr[pad - i])
    padded[pad : pad + n] = audio_arr
    for i in range(pad):
        src = max(n - 2 - i, 0)
        padded[pad + n + i] = float(audio_arr[src])

    power = np.zeros(n_bins * N_FRAMES, dtype=np.float32)
    for frame in range(N_FRAMES):
        start = frame * HOP_LENGTH
        windowed = padded[start : start + N_FFT] * hann
        spec = np.fft.rfft(windowed, n=N_FFT)
        re = spec.real.astype(np.float32)
        im = spec.imag.astype(np.float32)
        mag = re * re + im * im
        power[frame * n_bins : (frame + 1) * n_bins] = mag

    mel = np.zeros(N_MELS * N_FRAMES, dtype=np.float32)
    for m in range(N_MELS):
        for frame in range(N_FRAMES):
            s = 0.0
            base_f = mel_filters[m * n_bins : (m + 1) * n_bins]
            base_p = power[frame * n_bins : (frame + 1) * n_bins]
            s = float(np.dot(base_f, base_p))
            mel[m * N_FRAMES + frame] = s

    eps = 1e-10
    log_mel = np.log10(np.maximum(mel, eps)).astype(np.float32)
    max_val = float(np.max(log_mel))
    floor = max_val - 8.0
    log_mel = np.maximum(log_mel, floor)
    log_mel = (log_mel + 4.0) / 4.0
    return log_mel.astype(np.float32)

def _normalize_output(raw: float) -> float:
    if not math.isfinite(raw):
        return raw
    if 0.0 <= raw <= 1.0:
        return raw
    return _sigmoid(raw)

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

class AudioEouModel(EouModel):
    def __init__(
        self,
        session,
        audio_window_ms: int,
        pad_alignment: AudioPadAlignment,
    ) -> None:
        self._session = session
        self._lock = threading.Lock()
        self._audio_window_ms = int(audio_window_ms)
        self._pad_alignment = pad_alignment
        self._mel_filters = build_mel_filters()
        self._hann = build_hann_window()

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        audio_window_ms: int,
        pad_alignment: AudioPadAlignment,
    ) -> "AudioEouModel":
        import onnxruntime as ort

        path = Path(model_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(path),
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )
        return cls(session, audio_window_ms, pad_alignment)

    def audio_window_ms(self) -> int:
        return self._audio_window_ms

    def pad_alignment(self) -> AudioPadAlignment:
        return self._pad_alignment

    def run(self, audio: np.ndarray, sample_rate: int) -> float:
        if int(sample_rate) != int(SAMPLE_RATE):
            raise ValueError(
                f"smart-turn expects {SAMPLE_RATE} Hz, got {sample_rate}"
            )
        prepared = prepare_audio(audio, self._audio_window_ms, self._pad_alignment)
        mel = log_mel_spectrogram(prepared, self._hann, self._mel_filters)
        if mel.shape[0] != N_MELS * N_FRAMES:
            raise ValueError(
                f"mel size {mel.shape[0]} != {N_MELS * N_FRAMES}"
            )
        tensor = mel.reshape(1, N_MELS, N_FRAMES).astype(np.float32, copy=False)
        with self._lock:
            outputs = self._session.run(None, {"input_features": tensor})
        if not outputs:
            raise ValueError("smart-turn produced no outputs")
        first = np.asarray(outputs[0])
        flat = first.reshape(-1)
        if flat.size == 0:
            raise ValueError("smart-turn empty output")
        raw = float(flat[0])
        return _normalize_output(raw)

    def score(self, context: str) -> float:
        return float("nan")

    def score_with_audio(self, context: str, audio, sample_rate: int) -> float:
        try:
            p = self.run(np.asarray(audio, dtype=np.float32), int(sample_rate))
        except Exception:
            return float("nan")
        if math.isfinite(p) and 0.0 <= p <= 1.0:
            return p
        return float("nan")

def try_load_from_env(
    cfg_window_ms: int, cfg_alignment: AudioPadAlignment
) -> Optional[AudioEouModel]:
    path = env_keys.read_str_or_none(env_keys.EOU_AUDIO_MODEL_PATH)
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return AudioEouModel.load(p, cfg_window_ms, cfg_alignment)
    except Exception:
        return None

def shared_audio_eou_model(
    window_ms: int, alignment: AudioPadAlignment
) -> Optional[EouModel]:
    return try_load_from_env(window_ms, alignment)

def resolve_audio_eou_paths() -> Optional[str]:
    return env_keys.read_str_or_none(env_keys.EOU_AUDIO_MODEL_PATH)
