from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

NEG_INF = -1e30

@dataclass(frozen=True)
class _Tag:
    prefix: str
    cls: str

def _split(label: str) -> _Tag:
    if label == "O":
        return _Tag("O", "")
    dash = label.find("-")
    if dash < 0:
        return _Tag("?", "")
    p = label[:dash]
    if p not in ("B", "I", "E", "S"):
        p = "?"
    return _Tag(p, label[dash + 1 :])

def _allowed(a: _Tag, b: _Tag) -> bool:
    if a.prefix in ("O", "E", "S"):
        return b.prefix in ("O", "B", "S")
    if a.prefix in ("B", "I"):
        return b.prefix in ("I", "E") and b.cls == a.cls
    return False

def _build_start(labels: Sequence[str]) -> np.ndarray:
    out = np.zeros(len(labels), dtype=np.float32)
    for i, lbl in enumerate(labels):
        p = _split(lbl).prefix
        if p in ("I", "E"):
            out[i] = NEG_INF
    return out

def _build_transitions(labels: Sequence[str]) -> np.ndarray:
    tags = [_split(l) for l in labels]
    n = len(labels)
    out = np.full((n, n), NEG_INF, dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if _allowed(tags[i], tags[j]):
                out[i, j] = 0.0
    return out

def _log_softmax_row(row: np.ndarray) -> np.ndarray:
    m = float(row.max())
    e = np.exp(row - m)
    return (row - (math.log(float(e.sum())) + m)).astype(np.float32)

def viterbi_decode(logits: np.ndarray, labels: Sequence[str]) -> np.ndarray:
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2-D [T, L], got {logits.shape}")
    T, L = logits.shape
    if L != len(labels):
        raise ValueError(f"logits L={L} != len(labels)={len(labels)}")
    if T == 0:
        return np.zeros(0, dtype=np.int32)

    trans = _build_transitions(labels)
    start = _build_start(labels)

    delta = start + _log_softmax_row(logits[0])
    bp = np.zeros((T, L), dtype=np.int32)

    for t in range(1, T):
        lp = _log_softmax_row(logits[t])
        scores = delta[:, None] + trans
        best_prev = scores.argmax(axis=0)
        best = scores[best_prev, np.arange(L)]
        delta = (best + lp).astype(np.float32)
        bp[t] = best_prev

    out = np.zeros(T, dtype=np.int32)
    out[T - 1] = int(delta.argmax())
    for t in range(T - 1, 0, -1):
        out[t - 1] = bp[t, out[t]]
    return out
