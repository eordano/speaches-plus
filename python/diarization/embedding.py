from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .fbank import FBank

if TYPE_CHECKING:
    import onnxruntime as ort

SAMPLE_RATE = 16_000
EMBEDDING_DIM = 256
FRAME_LENGTH_SAMPLES = 400
FRAME_SHIFT_SAMPLES = 160
NUM_MEL_BINS = 80
MIN_INPUT_SAMPLES = 16_000

DEFAULT_INPUT_NAME = "feats"
DEFAULT_OUTPUT_NAME = "embs"

class EmbeddingModel:
    def __init__(
        self,
        session: "ort.InferenceSession",
        input_name: str = DEFAULT_INPUT_NAME,
        output_name: str = DEFAULT_OUTPUT_NAME,
    ):
        self._session = session
        self._lock = threading.Lock()
        self._input_name = input_name
        self._output_name = output_name
        self._fbank = FBank(NUM_MEL_BINS, FRAME_LENGTH_SAMPLES, FRAME_SHIFT_SAMPLES)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        input_name: str = DEFAULT_INPUT_NAME,
        output_name: str = DEFAULT_OUTPUT_NAME,
        providers: list[str] | None = None,
    ) -> EmbeddingModel:
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
        return cls(session, input_name, output_name)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    @property
    def min_input_samples(self) -> int:
        return MIN_INPUT_SAMPLES

    def embed(self, samples: np.ndarray) -> np.ndarray:
        if samples.shape[0] < FRAME_LENGTH_SAMPLES:
            raise ValueError(
                f"embedding: input {samples.shape[0]} samples shorter than "
                f"frame length {FRAME_LENGTH_SAMPLES}"
            )
        feats_flat = self._fbank.compute(samples)
        frames = feats_flat.size // NUM_MEL_BINS
        feats = feats_flat.reshape(1, frames, NUM_MEL_BINS).astype(np.float32, copy=False)
        with self._lock:
            outputs = self._session.run(None, {self._input_name: feats})
        emb = outputs[0]
        if emb.size == 0 or emb.shape[-1] != EMBEDDING_DIM:
            raise ValueError(
                f"embedding: expected last dim {EMBEDDING_DIM}, got shape {emb.shape}"
            )
        flat = emb.astype(np.float32, copy=False).reshape(-1)
        return _l2_normalize(flat)

def _l2_normalize(v: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(v))
    if norm < 1e-9:
        norm = 1e-9
    return v / norm

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"cosine_sim: shape mismatch {a.shape} vs {b.shape}")
    return float(np.dot(a, b))
