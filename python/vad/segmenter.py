from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .constants import (
    MIN_SILENCE_AT_MAX_SPEECH_MS,
    MIN_SPEECH_DURATION_MS,
    MAX_SPEECH_DURATION_S,
    NEG_THRESHOLD_DELTA,
    NEG_THRESHOLD_FLOOR,
    PREFIX_PADDING_MS,
    SAMPLE_RATE,
    SILENCE_DURATION_MS,
    SPEECH_THRESHOLD,
    WINDOW_SAMPLES,
)

@dataclass(frozen=True)
class SpeechTimestamp:
    start: int
    end: int

@dataclass
class VadOptions:
    threshold: float = SPEECH_THRESHOLD
    neg_threshold: float | None = None
    min_speech_duration_ms: int = MIN_SPEECH_DURATION_MS
    max_speech_duration_s: float = MAX_SPEECH_DURATION_S
    min_silence_duration_ms: int = SILENCE_DURATION_MS
    speech_pad_ms: int = PREFIX_PADDING_MS

class VadInfer(Protocol):
    def process_window(self, window: np.ndarray) -> float: ...
    def reset(self) -> None: ...

def get_speech_timestamps(
    model: VadInfer,
    audio: np.ndarray,
    opts: VadOptions,
    sample_rate: int = SAMPLE_RATE,
) -> list[SpeechTimestamp]:
    window_size = WINDOW_SAMPLES
    audio_length = int(audio.shape[0])

    pad = (window_size - audio_length % window_size) % window_size
    model.reset()
    total_windows = (audio_length + pad) // window_size
    probs: list[float] = []
    window_buf = np.zeros(window_size, dtype=np.float32)
    audio_f32 = np.ascontiguousarray(audio, dtype=np.float32)
    for w in range(total_windows):
        start = w * window_size
        end = min(start + window_size, audio_length)
        copy_len = end - start
        if copy_len > 0:
            window_buf[:copy_len] = audio_f32[start:end]
        if copy_len < window_size:
            window_buf[copy_len:] = 0.0
        probs.append(model.process_window(window_buf))
    model.reset()

    return speech_timestamps_from_probs(probs, audio_length, opts, sample_rate)

def speech_timestamps_from_probs(
    probs: list[float] | np.ndarray,
    audio_length: int,
    opts: VadOptions,
    sample_rate: int = SAMPLE_RATE,
) -> list[SpeechTimestamp]:
    if opts.neg_threshold is not None:
        neg_threshold = opts.neg_threshold
    else:
        neg_threshold = max(opts.threshold - NEG_THRESHOLD_DELTA, NEG_THRESHOLD_FLOOR)

    window_size = WINDOW_SAMPLES
    min_speech_samples = sample_rate * opts.min_speech_duration_ms // 1000
    speech_pad_samples = sample_rate * opts.speech_pad_ms // 1000
    max_speech_raw = int(sample_rate * opts.max_speech_duration_s) - window_size - 2 * speech_pad_samples
    max_speech_samples = max(max_speech_raw, 0)
    min_silence_samples = sample_rate * opts.min_silence_duration_ms // 1000
    min_silence_samples_at_max_speech = sample_rate * MIN_SILENCE_AT_MAX_SPEECH_MS // 1000

    speeches: list[dict] = []
    triggered = False
    current_start = 0
    have_current = False
    temp_end = 0
    prev_end = 0
    next_start = 0

    for i, prob in enumerate(probs):
        prob = float(prob)
        pos = window_size * i

        if prob >= opts.threshold and temp_end != 0:
            temp_end = 0
            if next_start < prev_end:
                next_start = pos

        if prob >= opts.threshold and not triggered:
            triggered = True
            current_start = pos
            have_current = True
            continue

        if triggered and pos - current_start > max_speech_samples:
            if prev_end != 0:
                speeches.append({"start": current_start, "end": prev_end})
                have_current = False
                if next_start < prev_end:
                    triggered = False
                else:
                    current_start = next_start
                    have_current = True
                prev_end = 0
                next_start = 0
                temp_end = 0
            else:
                speeches.append({"start": current_start, "end": pos})
                have_current = False
                prev_end = 0
                next_start = 0
                temp_end = 0
                triggered = False
                continue

        if prob < neg_threshold and triggered:
            if temp_end == 0:
                temp_end = pos
            if max(pos - temp_end, 0) > min_silence_samples_at_max_speech:
                prev_end = temp_end
            if max(pos - temp_end, 0) < min_silence_samples:
                continue
            seg_end = temp_end
            if have_current and seg_end > current_start and seg_end - current_start > min_speech_samples:
                speeches.append({"start": current_start, "end": seg_end})
            have_current = False
            prev_end = 0
            next_start = 0
            temp_end = 0
            triggered = False
            continue

    if have_current and audio_length > current_start and audio_length - current_start > min_speech_samples:
        speeches.append({"start": current_start, "end": audio_length})

    n = len(speeches)
    for i in range(n):
        if i == 0:
            speeches[i]["start"] = max(speeches[i]["start"] - speech_pad_samples, 0)
        if i != n - 1:
            next_start_pos = speeches[i + 1]["start"]
            cur_end = speeches[i]["end"]
            silence = max(next_start_pos - cur_end, 0)
            if silence < 2 * speech_pad_samples:
                half = silence // 2
                speeches[i]["end"] += half
                speeches[i + 1]["start"] = max(speeches[i + 1]["start"] - half, 0)
            else:
                speeches[i]["end"] = min(speeches[i]["end"] + speech_pad_samples, audio_length)
                speeches[i + 1]["start"] = max(speeches[i + 1]["start"] - speech_pad_samples, 0)
        else:
            speeches[i]["end"] = min(speeches[i]["end"] + speech_pad_samples, audio_length)

    return [SpeechTimestamp(start=s["start"], end=s["end"]) for s in speeches]

def to_ms_speech_timestamps(timestamps: list[SpeechTimestamp]) -> list[SpeechTimestamp]:
    div = SAMPLE_RATE // 1000
    return [SpeechTimestamp(start=t.start // div, end=t.end // div) for t in timestamps]
