from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

SAMPLE_RATE = 16_000
FRAME_RATE_HZ = 50
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE_HZ

DEFAULT_MAX_SPEAKERS_PER_CHUNK = 4
DEFAULT_MAX_SPEAKERS_PER_FRAME = 2

WAVEFORM_INPUT_KEY = "waveform"

@dataclass(frozen=True)
class SegmentationLogits:
    frames: int
    classes: int
    data: np.ndarray

    def row(self, frame: int) -> np.ndarray:
        return self.data[frame * self.classes : (frame + 1) * self.classes]

class SegmentationModel:
    def __init__(
        self,
        session: "ort.InferenceSession",
        max_speakers_per_chunk: int = DEFAULT_MAX_SPEAKERS_PER_CHUNK,
        max_speakers_per_frame: int = DEFAULT_MAX_SPEAKERS_PER_FRAME,
    ):
        self._session = session
        self._lock = threading.Lock()
        self._max_speakers_per_chunk = max_speakers_per_chunk
        self._max_speakers_per_frame = max_speakers_per_frame

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        max_speakers_per_chunk: int = DEFAULT_MAX_SPEAKERS_PER_CHUNK,
        max_speakers_per_frame: int = DEFAULT_MAX_SPEAKERS_PER_FRAME,
        providers: list[str] | None = None,
    ) -> SegmentationModel:
        import onnxruntime as ort
        path = Path(model_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(path),
            sess_options=sess_options,
            providers=providers or ["CPUExecutionProvider"],
        )
        return cls(session, max_speakers_per_chunk, max_speakers_per_frame)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def frame_rate_hz(self) -> int:
        return FRAME_RATE_HZ

    @property
    def max_speakers_per_chunk(self) -> int:
        return self._max_speakers_per_chunk

    @property
    def max_speakers_per_frame(self) -> int:
        return self._max_speakers_per_frame

    def run(self, samples: np.ndarray) -> SegmentationLogits:
        if samples.size == 0:
            raise ValueError("segmentation: empty input")
        n = samples.shape[0]
        waveform = np.ascontiguousarray(samples, dtype=np.float32).reshape(1, 1, n)
        with self._lock:
            outputs = self._session.run(None, {WAVEFORM_INPUT_KEY: waveform})
        first = outputs[0]
        if first.ndim != 3:
            raise ValueError(
                f"segmentation: expected 3D output, got shape {first.shape}"
            )
        frames = int(first.shape[1])
        classes = int(first.shape[2])
        if frames * classes != first.size:
            raise ValueError(
                f"segmentation: shape {first.shape} disagrees with {first.size} elements"
            )
        return SegmentationLogits(
            frames=frames,
            classes=classes,
            data=first.astype(np.float32, copy=False).reshape(-1),
        )
