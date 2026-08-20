from copy import copy
from enum import Enum, auto
from itertools import count

from nano_vllm.sampling_params import SamplingParams

class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams | None = None):
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.enable_ngram_spec_decode = sampling_params.enable_ngram_spec_decode
        self.ngram_max_n = sampling_params.ngram_max_n
        self.ngram_min_n = sampling_params.ngram_min_n
        self.ngram_num_drafts = sampling_params.ngram_num_drafts
        self.draft_tokens: list[int] = []
        self.accepted_token_count: int = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key: int | slice) -> int | list[int]:
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def set_drafts(self, drafts: list[int]) -> None:
        self.draft_tokens = list(drafts)

    def clear_drafts(self) -> None:
        self.draft_tokens = []

    def verify_inputs(self) -> tuple[list[int], list[int]]:
        start = self.num_cached_tokens
        tokens = [self.last_token, *self.draft_tokens]
        positions = list(range(start, start + len(tokens)))
        return tokens, positions

    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
            self.draft_tokens,
        )

    def __setstate__(self, state):
        (
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_cached_tokens,
            self.num_scheduled_tokens,
            self.block_table,
            last_state,
            self.draft_tokens,
        ) = state
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
