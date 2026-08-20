from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constants import (
    SILENCE_PEAK_THRESHOLD,
    WHISPER_CHUNK_SECS,
    WHISPER_SAMPLING_HZ,
)
from .mel import MelExtractor
from .noise_gate import GateThresholds, NoiseRejection, evaluate
from .whisper import (
    TimedSegment,
    TranscriptionResult,
    WhisperBackend,
    join_segments,
    peak_amplitude,
)

@dataclass
class ChunkResult:
    offset_ms: int
    result: TranscriptionResult
    rejection: NoiseRejection | None = None

def _chunk_audio(audio: np.ndarray, sample_rate: int) -> list[tuple[int, np.ndarray]]:
    if sample_rate != WHISPER_SAMPLING_HZ:
        raise ValueError(f"transcribe_long expects {WHISPER_SAMPLING_HZ}Hz, got {sample_rate}")
    chunk_samples = WHISPER_CHUNK_SECS * sample_rate
    if len(audio) <= chunk_samples:
        return [(0, audio)]
    out: list[tuple[int, np.ndarray]] = []
    pos = 0
    while pos < len(audio):
        end = min(pos + chunk_samples, len(audio))
        offset_ms = (pos * 1000) // sample_rate
        out.append((offset_ms, audio[pos:end]))
        pos = end
    return out

def _shift_segments(segs: list[TimedSegment], offset_ms: int) -> list[TimedSegment]:
    return [
        TimedSegment(
            t_start_ms=s.t_start_ms + offset_ms,
            t_end_ms=s.t_end_ms + offset_ms,
            text=s.text,
            avg_logprob=s.avg_logprob,
            no_speech_prob=s.no_speech_prob,
        )
        for s in segs
    ]

def _aggregate_chunk_stats(
    results: list[TranscriptionResult],
    durations_ms: list[int],
) -> tuple[float | None, float | None]:
    lp_sum = 0.0
    lp_w = 0.0
    nsp_sum = 0.0
    nsp_w = 0.0
    for r, dur in zip(results, durations_ms):
        d = float(max(dur, 1))
        if r.avg_logprob is not None:
            lp_sum += float(r.avg_logprob) * d
            lp_w += d
        if r.no_speech_prob is not None:
            nsp_sum += float(r.no_speech_prob) * d
            nsp_w += d
    lp = (lp_sum / lp_w) if lp_w > 0 else None
    nsp = (nsp_sum / nsp_w) if nsp_w > 0 else None
    return lp, nsp

def transcribe_long(
    backend: WhisperBackend,
    audio: np.ndarray,
    sample_rate: int = WHISPER_SAMPLING_HZ,
    language: str | None = None,
    prompt: str | None = None,
    mel_extractor: MelExtractor | None = None,
    n_mels: int = 80,
    gate: GateThresholds | None = None,
    silence_peak_threshold: float = SILENCE_PEAK_THRESHOLD,
    task: str = "transcribe",
) -> TranscriptionResult:
    del mel_extractor, n_mels
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if peak_amplitude(audio) < silence_peak_threshold:
        return TranscriptionResult.empty()

    chunks = _chunk_audio(audio, sample_rate)
    gate = gate if gate is not None else GateThresholds.disabled()

    accepted_segments: list[TimedSegment] = []
    accepted_results: list[TranscriptionResult] = []
    accepted_durations: list[int] = []
    accepted_texts: list[str] = []

    for offset_ms, chunk in chunks:
        if peak_amplitude(chunk) < silence_peak_threshold:
            continue
        result = backend.transcribe(
            chunk,
            sample_rate=sample_rate,
            language=language,
            prompt=prompt,
            task=task,
        )
        chunk_dur_ms = (len(chunk) * 1000) // sample_rate
        rejection = evaluate(
            avg_no_speech_prob=result.no_speech_prob,
            avg_logprob=result.avg_logprob,
            duration_ms=chunk_dur_ms,
            thresholds=gate,
        )
        if rejection is not None:
            continue
        if not result.text.strip() and not result.segments:
            continue
        shifted = _shift_segments(result.segments, offset_ms)
        accepted_segments.extend(shifted)
        accepted_results.append(result)
        accepted_durations.append(chunk_dur_ms)
        if result.text.strip():
            accepted_texts.append(result.text.strip())

    text = join_segments(accepted_texts) if accepted_texts else join_segments(s.text for s in accepted_segments)
    avg_lp, avg_nsp = _aggregate_chunk_stats(accepted_results, accepted_durations)
    return TranscriptionResult(
        text=text,
        avg_logprob=avg_lp,
        no_speech_prob=avg_nsp,
        compression_ratio=None,
        segments=accepted_segments,
    )
