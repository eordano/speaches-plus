
import base64
import io
import urllib.request
from typing import Any, Union
from urllib.parse import urlparse

import librosa
import numpy as np
import soundfile as sf

AudioLike = Union[
    str,
    tuple[np.ndarray, int],
]

SAMPLE_RATE = 16000

def ensure_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else [x]

def is_url(s: str) -> bool:
    try:
        u = urlparse(s)
        return u.scheme in ("http", "https") and bool(u.netloc)
    except Exception:
        return False

def is_probably_base64(s: str) -> bool:
    if s.startswith("data:audio"):
        return True
    if ("/" not in s and "\\" not in s) and len(s) > 256:
        return True
    return False

def decode_base64_bytes(b64: str) -> bytes:
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)

def load_audio_any(x: str) -> tuple[np.ndarray, int]:
    if is_url(x):
        with urllib.request.urlopen(x) as resp:
            audio_bytes = resp.read()
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    elif is_probably_base64(x):
        audio_bytes = decode_base64_bytes(x)
        with io.BytesIO(audio_bytes) as f:
            audio, sr = sf.read(f, dtype="float32", always_2d=False)
    else:
        audio, sr = librosa.load(x, sr=None, mono=False)
    return np.asarray(audio, dtype=np.float32), int(sr)

def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:

        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
        return np.mean(audio, axis=-1).astype(np.float32)
    raise ValueError(f"Unsupported audio ndim={audio.ndim}")

def float_range_normalize(audio: np.ndarray) -> np.ndarray:
    audio = audio.astype(np.float32)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if peak == 0.0:
        return audio
    if peak > 1.0:
        audio = audio / peak
    return np.clip(audio, -1.0, 1.0)

def normalize_audio_input(a: AudioLike) -> np.ndarray:
    if isinstance(a, str):
        audio, sr = load_audio_any(a)
    elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
        audio, sr = a[0], int(a[1])
    else:
        raise TypeError(f"Unsupported audio input type: {type(a)}")
    audio = to_mono(np.asarray(audio))
    if sr != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE).astype(np.float32)
    return float_range_normalize(audio)

def normalize_audios(audios: AudioLike | list[AudioLike]) -> list[np.ndarray]:
    return [normalize_audio_input(a) for a in ensure_list(audios)]
