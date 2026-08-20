from collections import deque
from typing import cast

from nano_vllm.config import Config
from nano_vllm.engine.block_manager import BlockManager
from nano_vllm.engine.sequence import Sequence, SequenceStatus

SCHEDULE_MODE_PREFILL = "prefill"
SCHEDULE_MODE_DECODE = "decode"
SCHEDULE_MODE_VERIFY = "verify"

class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.num_running_with_drafts: int = 0

    def note_drafts_set(self, had_drafts: bool, has_drafts: bool) -> None:
        if has_drafts and not had_drafts:
            self.num_running_with_drafts += 1
        elif had_drafts and not has_drafts:
            self.num_running_with_drafts -= 1

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], str]:
        scheduled_seqs = []
        num_batched_tokens = 0

        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, SCHEDULE_MODE_PREFILL

        if not self.running:
            return [], SCHEDULE_MODE_DECODE

        mode = SCHEDULE_MODE_VERIFY if self.num_running_with_drafts > 0 else SCHEDULE_MODE_DECODE
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            if mode == SCHEDULE_MODE_VERIFY:
                slots_needed = 1 + len(seq.draft_tokens)
            else:
                slots_needed = 1
            while not self.block_manager.can_append_n(seq, slots_needed):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = slots_needed
                seq.is_prefill = False
                self.block_manager.may_append_n(seq, slots_needed)
                scheduled_seqs.append(seq)
        if not scheduled_seqs:
            return [], mode
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, mode

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        had_drafts = bool(seq.draft_tokens)
        seq.clear_drafts()
        if had_drafts:
            self.num_running_with_drafts -= 1
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence],
        token_ids: list[int] | list[tuple[int, int, list[int]]],
        mode: str,
    ) -> list[tuple[int, int]]:
        if mode == SCHEDULE_MODE_VERIFY:
            verify_outputs = cast(list[tuple[int, int, list[int]]], token_ids)
            return self._postprocess_verify(seqs, verify_outputs)
        plain_tokens = cast(list[int], token_ids)
        return self._postprocess_simple(seqs, plain_tokens, mode == SCHEDULE_MODE_PREFILL)

    def _postprocess_simple(
        self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool
    ) -> list[tuple[int, int]]:
        accepted: list[tuple[int, int]] = []
        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            accepted.append((seq.seq_id, token_id))
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
        return accepted

    def _postprocess_verify(
        self,
        seqs: list[Sequence],
        verify_outputs: list[tuple[int, int, list[int]]],
    ) -> list[tuple[int, int]]:
        accepted: list[tuple[int, int]] = []
        for seq, (accepted_count, bonus_token_id, accepted_drafts) in zip(seqs, verify_outputs):
            advance = accepted_count + 1
            seq.accepted_token_count = accepted_count
            if seq.draft_tokens:
                self.num_running_with_drafts -= 1
            seq.clear_drafts()
            finished = False
            for token_id in accepted_drafts + [bonus_token_id]:
                seq.append_token(token_id)
                accepted.append((seq.seq_id, token_id))
                eos_hit = not seq.ignore_eos and token_id == self.eos
                if eos_hit or seq.num_completion_tokens == seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    finished = True
                    break
            if finished:
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
                continue
            seq.num_scheduled_tokens = advance
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += advance
            seq.num_scheduled_tokens = 0
        return accepted
