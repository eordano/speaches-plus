from __future__ import annotations

import numpy as np

from .types import MAX_DECODE_SAMPLE_RATE, MIN_DECODE_SAMPLE_RATE

def downmix_and_resample_f32(
    interleaved,
    channels: int,
    sr_in: int,
    sr_out: int,
) -> np.ndarray:
    if (
        not (MIN_DECODE_SAMPLE_RATE <= sr_in <= MAX_DECODE_SAMPLE_RATE)
        or not (MIN_DECODE_SAMPLE_RATE <= sr_out <= MAX_DECODE_SAMPLE_RATE)
        or channels == 0
    ):
        return np.zeros(0, dtype=np.float32)

    arr = np.asarray(interleaved, dtype=np.float32).reshape(-1)
    if channels == 1:
        mono = arr.copy()
    else:
        n_frames = arr.shape[0] // channels
        if n_frames == 0:
            mono = np.zeros(0, dtype=np.float32)
        else:
            framed = arr[: n_frames * channels].reshape(n_frames, channels)
            mono = framed.sum(axis=1, dtype=np.float32) / np.float32(channels)

    if sr_in == sr_out:
        return mono

    n_in = mono.shape[0]
    if n_in == 0:
        return np.zeros(0, dtype=np.float32)

    n_out = int((int(n_in) * int(sr_out)) // int(sr_in))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)

    i = np.arange(n_out, dtype=np.float64)
    pos = i * (float(sr_in) / float(sr_out))
    lo = np.floor(pos).astype(np.int64)
    t = pos - lo.astype(np.float64)
    hi = np.minimum(lo + 1, n_in - 1)
    lo = np.clip(lo, 0, n_in - 1)
    a = mono[lo].astype(np.float64)
    b = mono[hi].astype(np.float64)
    out = (a * (1.0 - t) + b * t).astype(np.float32)
    return out

__all__ = ["downmix_and_resample_f32"]
