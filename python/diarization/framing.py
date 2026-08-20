from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .powerset import Multilabel

COALESCE_GAP_MS = 250
SEGMENTATION_SAMPLE_RATE = 16_000

@dataclass(frozen=True)
class Chunk:
    samples: np.ndarray
    t_offset_ms: int

@dataclass(frozen=True)
class Span:
    sample_start: int
    sample_end: int
    t_start_ms: int
    t_end_ms: int
    local_speaker: int
    overlap: bool

@dataclass(frozen=True)
class ChunkSpans:
    chunk_index: int
    spans: list[Span]

def slide_chunks(
    audio: np.ndarray,
    sample_rate: int,
    chunk_seconds: float,
    hop_ratio: float,
) -> list[Chunk]:
    chunk_samples = int(chunk_seconds * sample_rate)
    hop_samples = max(1, int((chunk_seconds * hop_ratio) * sample_rate))
    if audio.shape[0] < chunk_samples:
        padded = np.zeros(chunk_samples, dtype=np.float32)
        n = min(audio.shape[0], chunk_samples)
        padded[:n] = audio[:n]
        return [Chunk(samples=padded, t_offset_ms=0)]
    out: list[Chunk] = []
    start = 0
    while start + chunk_samples <= audio.shape[0]:
        t_offset_ms = (start * 1000) // sample_rate
        out.append(Chunk(
            samples=np.array(audio[start : start + chunk_samples], dtype=np.float32, copy=True),
            t_offset_ms=t_offset_ms,
        ))
        start += hop_samples
    return out

def median_filter_multihot(input: Multilabel, window: int) -> Multilabel:
    if window <= 1:
        return input
    half = window // 2
    out = np.zeros(input.frames * input.speakers, dtype=np.uint8)
    grid = input.data.reshape(input.frames, input.speakers)
    for f in range(input.frames):
        lo = max(0, f - half)
        hi = min(input.frames, f + half + 1)
        window_slice = grid[lo:hi]
        ones = (window_slice != 0).sum(axis=0)
        majority = (ones * 2 > (hi - lo)).astype(np.uint8)
        out[f * input.speakers : (f + 1) * input.speakers] = majority
    return Multilabel(frames=input.frames, speakers=input.speakers, data=out)

def extract_spans(
    multihot: Multilabel,
    frame_rate_hz: int,
    t_offset_ms: int,
    min_frames: int,
) -> list[Span]:
    frame_ms = 1000.0 / float(frame_rate_hz)
    samples_per_frame = SEGMENTATION_SAMPLE_RATE // int(frame_rate_hz)
    grid = multihot.data.reshape(multihot.frames, multihot.speakers)
    overlap_mask = (grid != 0).sum(axis=1) >= 2
    out: list[Span] = []
    for s in range(multihot.speakers):
        run_start: int | None = None
        for f in range(multihot.frames):
            active = grid[f, s] != 0
            if run_start is None and active:
                run_start = f
            elif run_start is not None and not active:
                _push_span(out, run_start, f, s, overlap_mask, frame_ms,
                           t_offset_ms, samples_per_frame, min_frames)
                run_start = None
        if run_start is not None:
            _push_span(out, run_start, multihot.frames, s, overlap_mask, frame_ms,
                       t_offset_ms, samples_per_frame, min_frames)
    return out

def _push_span(
    out: list[Span],
    start: int,
    end: int,
    speaker: int,
    overlap_mask: np.ndarray,
    frame_ms: float,
    t_offset_ms: int,
    samples_per_frame: int,
    min_frames: int,
) -> None:
    if end <= start:
        return
    length = end - start
    if length < min_frames:
        return
    overlap_frames = int(overlap_mask[start:end].sum())
    is_overlap = overlap_frames * 2 > length
    t_start_ms = t_offset_ms + int(start * frame_ms)
    t_end_ms = t_offset_ms + int(end * frame_ms)
    sample_offset = (t_offset_ms * SEGMENTATION_SAMPLE_RATE) // 1000
    out.append(Span(
        sample_start=sample_offset + start * samples_per_frame,
        sample_end=sample_offset + end * samples_per_frame,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        local_speaker=speaker,
        overlap=is_overlap,
    ))

def coalesce_segments(segments: list) -> list:
    if not segments:
        return segments
    from .types import DiarSegment
    sorted_segs = sorted(segments, key=lambda s: s.t_start_ms)
    out: list[DiarSegment] = []
    for s in sorted_segs:
        if out:
            last = out[-1]
            if last.speaker == s.speaker and s.t_start_ms <= last.t_end_ms + COALESCE_GAP_MS:
                out[-1] = DiarSegment(
                    speaker=last.speaker,
                    t_start_ms=last.t_start_ms,
                    t_end_ms=max(last.t_end_ms, s.t_end_ms),
                    confidence=max(last.confidence, s.confidence),
                )
                continue
        out.append(s)
    return out
