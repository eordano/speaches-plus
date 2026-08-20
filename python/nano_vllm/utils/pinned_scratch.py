from __future__ import annotations

import torch

_USE_PIN = torch.cuda.is_available()

class PinnedScratch:
    def __init__(self) -> None:
        self._buffers: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def view(self, name: str, dtype: torch.dtype, size: int) -> torch.Tensor:
        key = (name, dtype)
        buf = self._buffers.get(key)
        if buf is None or buf.numel() < size:
            cap = 1
            while cap < size:
                cap *= 2
            buf = torch.empty(cap, dtype=dtype, pin_memory=_USE_PIN)
            self._buffers[key] = buf
        return buf[:size]

_SCRATCH = PinnedScratch()

def host_view(name: str, dtype: torch.dtype, size: int) -> torch.Tensor:
    return _SCRATCH.view(name, dtype, size)
