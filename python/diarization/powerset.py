from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .segmentation import SegmentationLogits

@dataclass(frozen=True)
class Multilabel:
    frames: int
    speakers: int
    data: np.ndarray

    def row(self, frame: int) -> np.ndarray:
        return self.data[frame * self.speakers : (frame + 1) * self.speakers]

class PowersetDecoder:
    def __init__(self, max_speakers_per_chunk: int, max_speakers_per_frame: int):
        self.max_speakers_per_chunk = max_speakers_per_chunk
        self.max_speakers_per_frame = max_speakers_per_frame
        self._mapping = _build_mapping(max_speakers_per_chunk, max_speakers_per_frame)

    def num_classes(self) -> int:
        return len(self._mapping)

    @property
    def mapping(self) -> list[list[int]]:
        return self._mapping

    def to_multilabel_hard(self, logits: SegmentationLogits) -> Multilabel:
        if logits.classes != self.num_classes():
            raise ValueError(
                f"logits.classes ({logits.classes}) != decoder.num_classes "
                f"({self.num_classes()}) -- topology mismatch"
            )
        speakers = self.max_speakers_per_chunk
        data = np.zeros(logits.frames * speakers, dtype=np.uint8)
        flat = logits.data.reshape(logits.frames, logits.classes)
        argmax = np.argmax(flat, axis=1)
        for frame_idx, cls in enumerate(argmax):
            for spk in self._mapping[int(cls)]:
                data[frame_idx * speakers + spk] = 1
        return Multilabel(frames=logits.frames, speakers=speakers, data=data)

def _build_mapping(num_classes: int, max_set_size: int) -> list[list[int]]:
    out: list[list[int]] = []
    for size in range(max_set_size + 1):
        for combo in _combinations(num_classes, size):
            out.append(combo)
    return out

def _combinations(n: int, k: int) -> list[list[int]]:
    result: list[list[int]] = []
    _pick(0, n, k, [], result)
    return result

def _pick(start: int, n: int, k: int, buf: list[int], out: list[list[int]]) -> None:
    if len(buf) == k:
        out.append(list(buf))
        return
    for i in range(start, n):
        buf.append(i)
        _pick(i + 1, n, k, buf, out)
        buf.pop()
