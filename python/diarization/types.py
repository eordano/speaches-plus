from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

import env
from .clustering import ClusterId, OnlineClusterer
from .framing import coalesce_segments, extract_spans, median_filter_multihot, slide_chunks
from .powerset import PowersetDecoder
from .segmentation import SegmentationModel
from .embedding import EmbeddingModel

DEFAULT_CHUNK_SECONDS = 5.0
DEFAULT_HOP_RATIO = 0.1
DEFAULT_MEDIAN_FILTER_WINDOW = 11
DEFAULT_MIN_SPAN_FRAMES = 8
DEFAULT_CLUSTERING_THRESHOLD = 0.55
DEFAULT_MAX_SPEAKERS = 16

@dataclass(frozen=True)
class DiarSegment:
    speaker: ClusterId
    t_start_ms: int
    t_end_ms: int
    confidence: float

@dataclass
class DiarConfig:
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS
    hop_ratio: float = DEFAULT_HOP_RATIO
    median_filter_window: int = DEFAULT_MEDIAN_FILTER_WINDOW
    min_span_frames: int = DEFAULT_MIN_SPAN_FRAMES
    clustering_threshold: float = DEFAULT_CLUSTERING_THRESHOLD
    max_speakers: int = DEFAULT_MAX_SPEAKERS

    @classmethod
    def from_env(cls) -> DiarConfig:
        return cls(
            chunk_seconds=DEFAULT_CHUNK_SECONDS,
            hop_ratio=DEFAULT_HOP_RATIO,
            median_filter_window=env.read_int(env.DIAR_MEDIAN_FILTER_FRAMES, DEFAULT_MEDIAN_FILTER_WINDOW),
            min_span_frames=env.read_int(env.DIAR_MIN_SPAN_FRAMES, DEFAULT_MIN_SPAN_FRAMES),
            clustering_threshold=_clamp01(env.read_float(env.DIAR_THRESHOLD, DEFAULT_CLUSTERING_THRESHOLD)),
            max_speakers=max(1, env.read_int(env.DIAR_MAX_SPEAKERS, DEFAULT_MAX_SPEAKERS)),
        )

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))

class Diarizer:
    def __init__(
        self,
        seg: SegmentationModel,
        emb: EmbeddingModel,
        cfg: DiarConfig | None = None,
    ):
        self._cfg = cfg or DiarConfig.from_env()
        self._seg = seg
        self._emb = emb
        self._decoder = PowersetDecoder(
            seg.max_speakers_per_chunk,
            seg.max_speakers_per_frame,
        )
        self._clusterer = OnlineClusterer(
            self._cfg.clustering_threshold,
            self._cfg.max_speakers,
        )
        self._session_start_ms: int | None = None

    @property
    def cfg(self) -> DiarConfig:
        return self._cfg

    def reset(self) -> None:
        self._clusterer.reset()
        self._session_start_ms = None

    def diarize_utterance(
        self, audio: np.ndarray, t_start_ms: int = 0,
    ) -> list[DiarSegment]:
        if self._session_start_ms is None:
            self._session_start_ms = t_start_ms
        chunks = slide_chunks(
            audio,
            self._seg.sample_rate,
            self._cfg.chunk_seconds,
            self._cfg.hop_ratio,
        )
        emitted: list[DiarSegment] = []
        for chunk in chunks:
            logits = self._seg.run(chunk.samples)
            multihot = self._decoder.to_multilabel_hard(logits)
            smoothed = median_filter_multihot(multihot, self._cfg.median_filter_window)
            chunk_spans = extract_spans(
                smoothed,
                self._seg.frame_rate_hz,
                chunk.t_offset_ms,
                self._cfg.min_span_frames,
            )
            for span in chunk_spans:
                span_audio = audio[span.sample_start : min(span.sample_end, audio.shape[0])]
                if span_audio.shape[0] < self._emb.min_input_samples:
                    continue
                emb = self._emb.embed(span_audio)
                cluster_id, score = self._clusterer.assign(emb)
                emitted.append(DiarSegment(
                    speaker=cluster_id,
                    t_start_ms=t_start_ms + span.t_start_ms,
                    t_end_ms=t_start_ms + span.t_end_ms,
                    confidence=score,
                ))
        return coalesce_segments(emitted)

@dataclass
class _DiarizerHandles:
    diarizer: Diarizer | None = None
    seg_model_path: str = ""
    emb_model_path: str = ""
    error: str | None = field(default=None)
