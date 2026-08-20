from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional

@dataclass
class IntegratedVerdict:
    p_eot: float
    p_eager_eot: float
    transcript_so_far: str

class IntegratedEouBackend:
    def step(self, audio_ms_so_far: int) -> Optional[IntegratedVerdict]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

class FakeIntegratedBackend(IntegratedEouBackend):
    def __init__(self, schedule: list[tuple[int, IntegratedVerdict]]) -> None:
        s = list(schedule)
        s.sort(key=lambda x: x[0])
        self._schedule = s
        self._cursor = 0
        self._lock = threading.Lock()

    @classmethod
    def smoke_default(cls) -> "FakeIntegratedBackend":
        return cls(
            [
                (
                    500,
                    IntegratedVerdict(
                        p_eot=0.1, p_eager_eot=0.2, transcript_so_far="hi"
                    ),
                ),
                (
                    1500,
                    IntegratedVerdict(
                        p_eot=0.3,
                        p_eager_eot=0.6,
                        transcript_so_far="hi there",
                    ),
                ),
                (
                    2500,
                    IntegratedVerdict(
                        p_eot=0.85,
                        p_eager_eot=0.9,
                        transcript_so_far="hi there friend",
                    ),
                ),
            ]
        )

    def step(self, audio_ms_so_far: int) -> Optional[IntegratedVerdict]:
        with self._lock:
            emit: Optional[IntegratedVerdict] = None
            while (
                self._cursor < len(self._schedule)
                and self._schedule[self._cursor][0] <= audio_ms_so_far
            ):
                emit = self._schedule[self._cursor][1]
                self._cursor += 1
            return emit

    def reset(self) -> None:
        with self._lock:
            self._cursor = 0
