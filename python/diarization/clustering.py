from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedding import cosine_sim

ClusterId = int

@dataclass
class _Centroid:
    id: int
    vec: np.ndarray
    count: int

class OnlineClusterer:
    def __init__(
        self,
        threshold: float,
        max_speakers: int,
        ema_smoothing: float = 0.9,
    ):
        self._centroids: list[_Centroid] = []
        self._next_id: int = 0
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.ema_smoothing = max(0.0, min(0.999, ema_smoothing))

    def with_ema(self, ema_smoothing: float) -> OnlineClusterer:
        self.ema_smoothing = max(0.0, min(0.999, ema_smoothing))
        return self

    def reset(self) -> None:
        self._centroids.clear()
        self._next_id = 0

    def num_clusters(self) -> int:
        return len(self._centroids)

    def assign(self, emb: np.ndarray) -> tuple[ClusterId, float]:
        best = self._best_match(emb)
        if best is not None and best[1] >= self.threshold:
            idx, sim = best
            self._update_centroid(idx, emb)
            return self._centroids[idx].id, sim
        if len(self._centroids) < self.max_speakers:
            new_id = self._next_id
            self._next_id += 1
            self._centroids.append(_Centroid(id=new_id, vec=np.array(emb, copy=True), count=1))
            return new_id, best[1] if best is not None else 1.0
        idx, sim = best  # type: ignore[misc]
        self._update_centroid(idx, emb)
        return self._centroids[idx].id, sim

    def lookup(self, emb: np.ndarray) -> tuple[ClusterId, float] | None:
        best = self._best_match(emb)
        if best is None or best[1] < self.threshold:
            return None
        idx, sim = best
        return self._centroids[idx].id, sim

    def _best_match(self, emb: np.ndarray) -> tuple[int, float] | None:
        best: tuple[int, float] | None = None
        for i, c in enumerate(self._centroids):
            if c.vec.shape[0] != emb.shape[0]:
                continue
            sim = cosine_sim(c.vec, emb)
            if best is None or sim > best[1]:
                best = (i, sim)
        return best

    def _update_centroid(self, idx: int, emb: np.ndarray) -> None:
        c = self._centroids[idx]
        alpha = self.ema_smoothing
        c.vec = alpha * c.vec + (1.0 - alpha) * emb
        norm = float(np.linalg.norm(c.vec))
        if norm < 1e-9:
            norm = 1e-9
        c.vec = c.vec / norm
        c.count += 1
