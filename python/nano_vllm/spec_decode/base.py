from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nano_vllm.engine.sequence import Sequence

class Proposer(ABC):

    @abstractmethod
    def propose(
        self,
        seqs: list[Sequence],
        runner_state: dict[str, Any],
    ) -> dict[int, list[int]]:

        ...
