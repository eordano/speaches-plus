from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import torch

CHUNK_FRAMES = 25
LEFT_CONTEXT_FRAMES = 72
TARGET_SAMPLE_RATE = 24000
CODEC_FRAME_RATE_HZ = 12
SAMPLES_PER_FRAME = TARGET_SAMPLE_RATE // CODEC_FRAME_RATE_HZ

def _to_int16_pcm(wav: np.ndarray) -> bytes:
    pcm = np.clip(np.asarray(wav, dtype=np.float32), -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16).tobytes()

def chunked_decode_pcm(
    model: Any,
    codec_tokens: torch.Tensor,
    *,
    chunk_frames: int = CHUNK_FRAMES,
    left_context_frames: int = LEFT_CONTEXT_FRAMES,
) -> Iterator[bytes]:

    if isinstance(codec_tokens, list):
        if len(codec_tokens) != 1:
            msg = "streaming decoder accepts a single codec sequence per call"
            raise ValueError(msg)
        codec_tokens = codec_tokens[0]

    n_frames = codec_tokens.shape[-1]

    for chunk_start in range(0, n_frames, chunk_frames):
        chunk_end = min(chunk_start + chunk_frames, n_frames)
        ctx_start = max(0, chunk_start - left_context_frames)

        window = codec_tokens[..., ctx_start:chunk_end]
        wavs, _sr = model.speech_tokenizer.decode([{"audio_codes": window}])
        wav = wavs[0]
        if hasattr(wav, "cpu"):
            wav = wav.cpu().numpy()
        wav = np.asarray(wav, dtype=np.float32)

        if chunk_start > 0:
            trim_samples = (chunk_start - ctx_start) * SAMPLES_PER_FRAME
            wav = wav[trim_samples:]

        if wav.size > 0:
            yield _to_int16_pcm(wav)
