from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

@dataclass
class Voice:
    shape: list[int]
    data: np.ndarray = field(repr=False)

    def row(self, index: int) -> np.ndarray:
        if not self.shape:
            raise ValueError("voice has no shape")
        if index >= self.shape[0]:
            raise IndexError(f"index {index} >= leading dim {self.shape[0]}")
        if index < 0:
            raise IndexError(f"index {index} < 0")
        row_size = 1
        for d in self.shape[1:]:
            row_size *= d
        flat = self.data.reshape(-1)
        off = index * row_size
        return np.asarray(flat[off:off + row_size])

def parse_npy(b: bytes) -> Voice:
    arr = np.load(io.BytesIO(b))
    if arr.dtype != np.dtype("<f4") and arr.dtype != np.float32:
        raise ValueError(f"unsupported dtype {arr.dtype!r} (need <f4)")
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)
    arr = arr.astype(np.float32, copy=False)
    return Voice(shape=list(arr.shape), data=arr)

def load_voices(path: Path) -> dict[str, Voice]:
    out: dict[str, Voice] = {}
    with zipfile.ZipFile(path, "r") as zf:
        for name in zf.namelist():
            if not name.endswith(".npy"):
                continue
            stem = name[: -len(".npy")]
            with zf.open(name) as entry:
                body = entry.read()
            out[stem] = parse_npy(body)
    return out
