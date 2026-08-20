from dataclasses import dataclass

import torch

@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    kv_indptr: torch.Tensor | None = None
    kv_indices: torch.Tensor | None = None
    kv_last_page_len: torch.Tensor | None = None
    qo_indptr: torch.Tensor | None = None
    batch_indices: torch.Tensor | None = None
    positions_in_page: torch.Tensor | None = None
    verify_mode: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    context_lens=None,
    block_tables=None,
    kv_indptr=None,
    kv_indices=None,
    kv_last_page_len=None,
    qo_indptr=None,
    batch_indices=None,
    positions_in_page=None,
    verify_mode=False,
):
    global _CONTEXT
    _CONTEXT = Context(
        is_prefill,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        slot_mapping,
        context_lens,
        block_tables,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        qo_indptr,
        batch_indices,
        positions_in_page,
        verify_mode,
    )

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
