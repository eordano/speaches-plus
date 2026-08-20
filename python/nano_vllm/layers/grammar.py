from __future__ import annotations

import json
from typing import Any

import torch

from nano_vllm.sampling_params import SamplingParams
from nano_vllm.utils.pinned_scratch import host_view

try:
    import xgrammar
    _HAS_XGRAMMAR = True
except ImportError:
    xgrammar = None
    _HAS_XGRAMMAR = False

XGRAMMAR_REQUIRED_MSG = (
    "guided decoding requires the 'xgrammar' package; install speaches-plus-python with "
    "the grammar extras or add xgrammar to your environment"
)
BITMASK_DTYPE = torch.int32
NEG_INF_FILL = float("-inf")

_pinned_bitmask_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}

def _get_pinned_bitmask(rows: int, vocab_size: int) -> torch.Tensor:
    if not _HAS_XGRAMMAR:
        raise RuntimeError(XGRAMMAR_REQUIRED_MSG)
    cache_key = (vocab_size, BITMASK_DTYPE)
    cached = _pinned_bitmask_cache.get(cache_key)
    if cached is None or cached.size(0) < rows:
        new_rows = 1
        while new_rows < rows:
            new_rows *= 2
        new_shape = xgrammar.get_bitmask_shape(new_rows, vocab_size)
        use_pin = torch.cuda.is_available()
        cached = torch.full(new_shape, -1, dtype=BITMASK_DTYPE, pin_memory=use_pin)
        _pinned_bitmask_cache[cache_key] = cached
    cached.fill_(-1)
    return cached.narrow(0, 0, rows)

class GrammarBackend:

    def __init__(self, tokenizer, vocab_size: int) -> None:
        if not _HAS_XGRAMMAR:
            self._compiler: Any = None
            self._tokenizer_info: Any = None
        else:
            self._tokenizer_info = xgrammar.TokenizerInfo.from_huggingface(tokenizer, vocab_size=vocab_size)
            self._compiler = xgrammar.GrammarCompiler(self._tokenizer_info)
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def compile(self, sampling_params: SamplingParams) -> xgrammar.GrammarMatcher | None:
        if not sampling_params.has_guided_decoding:
            return None
        if not _HAS_XGRAMMAR:
            raise RuntimeError(XGRAMMAR_REQUIRED_MSG)
        if sampling_params.guided_json is not None:
            schema = sampling_params.guided_json
            if isinstance(schema, dict):
                schema = json.dumps(schema)
            grammar = self._compiler.compile_json_schema(schema)
        elif sampling_params.guided_grammar is not None:
            grammar = self._compiler.compile_grammar(sampling_params.guided_grammar)
        elif sampling_params.guided_choice is not None:
            grammar = self._compiler.compile_grammar(_choice_grammar(sampling_params.guided_choice))
        elif sampling_params.guided_regex is not None:
            grammar = self._compiler.compile_regex(sampling_params.guided_regex)
        else:
            return None
        return xgrammar.GrammarMatcher(grammar)

def apply_grammar_mask(
    logits: torch.Tensor,
    matchers: list[xgrammar.GrammarMatcher | None],
    vocab_size: int,
) -> torch.Tensor:
    if not _HAS_XGRAMMAR:
        return logits
    active_indices = [batch_idx for batch_idx, matcher in enumerate(matchers) if matcher is not None]
    if not active_indices:
        return logits
    bitmask = _get_pinned_bitmask(len(active_indices), vocab_size)
    for slot, batch_idx in enumerate(active_indices):
        matchers[batch_idx].fill_next_token_bitmask(bitmask, slot)
    bitmask = bitmask.to(logits.device, non_blocking=True)
    indices_h = host_view("grammar_active_idx", torch.long, len(active_indices))
    indices_h.numpy()[:] = active_indices
    indices_tensor = indices_h.to(logits.device, non_blocking=True)
    xgrammar.apply_token_bitmask_inplace(logits, bitmask, indices=indices_tensor)
    return logits

def apply_grammar_mask_verify(
    logits: torch.Tensor,
    matchers_per_seq: list[xgrammar.GrammarMatcher | None],
    draft_tokens_per_seq: list[list[int]],
    vocab_size: int,
) -> torch.Tensor:
    if not _HAS_XGRAMMAR:
        return logits
    if not matchers_per_seq or not any(m is not None for m in matchers_per_seq):
        return logits
    if not hasattr(xgrammar.GrammarMatcher, "rollback"):
        raise NotImplementedError(
            "verify-mode grammar masking requires xgrammar.GrammarMatcher.rollback "
            "(xgrammar >= 0.1.0); the installed version lacks the API."
        )
    per_seq_lengths = [1 + len(drafts) for drafts in draft_tokens_per_seq]
    row_indices: list[int] = []
    row_matchers: list[Any] = []
    base = 0
    for seq_idx, matcher in enumerate(matchers_per_seq):
        length = per_seq_lengths[seq_idx]
        if matcher is None:
            base += length
            continue
        for k in range(length):
            row_indices.append(base + k)
            row_matchers.append(matcher)
        base += length
    if not row_indices:
        return logits
    bitmask = _get_pinned_bitmask(len(row_indices), vocab_size)
    slot = 0
    for seq_idx, matcher in enumerate(matchers_per_seq):
        if matcher is None:
            continue
        drafts = draft_tokens_per_seq[seq_idx]
        matcher.fill_next_token_bitmask(bitmask, slot)
        slot += 1
        advanced = 0
        for draft_token in drafts:
            ok = matcher.accept_token(draft_token)
            if not ok:
                break
            advanced += 1
            matcher.fill_next_token_bitmask(bitmask, slot)
            slot += 1
        if advanced < len(drafts):
            for _ in range(len(drafts) - advanced):
                matcher.fill_next_token_bitmask(bitmask, slot)
                slot += 1
        if advanced > 0:
            matcher.rollback(advanced)
    bitmask = bitmask.to(logits.device, non_blocking=True)
    rows_h = host_view("grammar_verify_rows", torch.long, len(row_indices))
    rows_h.numpy()[:] = row_indices
    indices_tensor = rows_h.to(logits.device, non_blocking=True)
    xgrammar.apply_token_bitmask_inplace(logits, bitmask, indices=indices_tensor)
    return logits

def accept_tokens(
    matchers: list[xgrammar.GrammarMatcher | None],
    token_ids: list[int],
) -> None:
    if not _HAS_XGRAMMAR:
        return
    for matcher, token_id in zip(matchers, token_ids):
        if matcher is None:
            continue
        if not matcher.accept_token(token_id):
            raise RuntimeError(
                f"grammar matcher rejected committed token {token_id}; "
                "the masking layer let an invalid token through (sampler/grammar desync)"
            )

def _choice_grammar(choices: list[str]) -> str:
    if not choices:
        raise ValueError("guided_choice cannot be empty")
    alternatives = " | ".join(json.dumps(choice) for choice in choices)
    return f'root ::= {alternatives}'

def has_xgrammar() -> bool:
    return _HAS_XGRAMMAR
