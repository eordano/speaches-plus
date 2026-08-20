from __future__ import annotations

import struct
from typing import Any

from nano_vllm.engine.sequence import Sequence
from nano_vllm.spec_decode.base import Proposer

_TOKEN_BYTES = 4
_TOKEN_FORMAT = ">i"

class NgramProposer(Proposer):

    def __init__(self, engine_default_enabled: bool = False):
        self.engine_default_enabled = engine_default_enabled

    @staticmethod
    def propose_tokens(
        token_ids: list[int],
        max_n: int = 8,
        min_n: int = 2,
        num_drafts: int = 5,
    ) -> list[int]:
        total = len(token_ids)
        if total < min_n + 1 or num_drafts <= 0:
            return []
        buf = struct.pack(f">{total}i", *token_ids)
        upper_n = min(max_n, total - 1)
        for n in range(upper_n, min_n - 1, -1):
            suffix_bytes = buf[(total - n) * _TOKEN_BYTES:]
            search_end = (total - n) * _TOKEN_BYTES
            pos = buf.rfind(suffix_bytes, 0, search_end)
            if pos == -1:
                continue
            start = pos // _TOKEN_BYTES
            draft_start = start + n
            draft_end = min(draft_start + num_drafts, total - n)
            if draft_start >= draft_end:
                return []
            return token_ids[draft_start:draft_end]
        return []

    def propose(
        self,
        seqs: list[Sequence],
        runner_state: dict[str, Any],
    ) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for seq in seqs:
            if seq.is_finished:
                continue
            if not (self.engine_default_enabled or seq.enable_ngram_spec_decode):
                continue
            drafts = self.propose_tokens(
                seq.token_ids,
                max_n=seq.ngram_max_n,
                min_n=seq.ngram_min_n,
                num_drafts=seq.ngram_num_drafts,
            )
            out[seq.seq_id] = drafts
        return out

if __name__ == "__main__":
    assert NgramProposer.propose_tokens([1, 2, 3, 4, 5, 1, 2, 3]) == [4, 5]
    assert NgramProposer.propose_tokens([1, 2, 3], min_n=2, max_n=4, num_drafts=3) == []
    assert NgramProposer.propose_tokens([], num_drafts=3) == []
    assert NgramProposer.propose_tokens([7], num_drafts=3) == []
    assert NgramProposer.propose_tokens([1, 2, 3, 1, 2], min_n=2, max_n=4, num_drafts=3) == [3]
    print("OK")
