from __future__ import annotations

import io
import shutil
import subprocess

import numpy as np
import soundfile as sf

from .resample import downmix_and_resample_f32
from .types import TARGET_SAMPLE_RATE

def _decode_via_soundfile(data: bytes) -> np.ndarray:
    with sf.SoundFile(io.BytesIO(data)) as handle:
        sr_in = int(handle.samplerate)
        channels = int(handle.channels)
        block = handle.read(dtype="float32", always_2d=True)
    if block.size == 0:
        return np.zeros(0, dtype=np.float32)
    interleaved = block.reshape(-1).astype(np.float32, copy=False)
    return downmix_and_resample_f32(
        interleaved, channels, sr_in, int(TARGET_SAMPLE_RATE)
    )

def _decode_via_ffmpeg(data: bytes) -> np.ndarray:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not on PATH")
    proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-ac",
            "1",
            "-ar",
            str(int(TARGET_SAMPLE_RATE)),
            "pipe:1",
        ],
        input=data,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg decode failed: {stderr or 'unknown error'}")
    return np.frombuffer(proc.stdout, dtype="<f4").astype(np.float32, copy=True)

def decode_via_symphonia(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("avdecode: empty input")
    try:
        return _decode_via_soundfile(data)
    except Exception:
        pass
    return _decode_via_ffmpeg(data)

__all__ = ["decode_via_symphonia"]
