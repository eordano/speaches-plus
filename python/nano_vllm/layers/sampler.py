import os

import torch
from torch import nn

GUMBEL_FLOOR = 1e-10
TOP_K_DISABLED_SENTINEL = -1
TOP_P_DISABLED_SENTINEL = 1.0
NEG_INF_FILL = float("-inf")
LAST_DIM = -1
ARGMAX_KEEPDIM = True
COMPILE_DISABLE_ENV_VAR = "NANO_VLLM_TORCH_COMPILE"
COMPILE_DISABLED_VALUE = "0"

def _torch_compile_enabled() -> bool:
    return os.environ.get(COMPILE_DISABLE_ENV_VAR) != COMPILE_DISABLED_VALUE

def _eager_sample(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float() / temperatures.unsqueeze(dim=1)
    logits = _apply_top_k(logits, top_k)
    logits = _apply_top_p(logits, top_p)
    probs = torch.softmax(logits, dim=LAST_DIM)
    gumbel_noise = torch.empty_like(probs).exponential_(1).clamp_min_(GUMBEL_FLOOR)
    return (probs / gumbel_noise).argmax(dim=LAST_DIM, keepdim=ARGMAX_KEEPDIM)

def _eager_sample_temp_only(
    logits: torch.Tensor,
    temperatures: torch.Tensor,
) -> torch.Tensor:
    logits = logits.float() / temperatures.unsqueeze(dim=1)
    probs = torch.softmax(logits, dim=LAST_DIM)
    gumbel_noise = torch.empty_like(probs).exponential_(1).clamp_min_(GUMBEL_FLOOR)
    return (probs / gumbel_noise).argmax(dim=LAST_DIM, keepdim=ARGMAX_KEEPDIM)

class Sampler(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._compiled_sample = (
            torch.compile(_eager_sample, dynamic=True) if _torch_compile_enabled() else None
        )
        self._compile_failed = False

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        any_top_k: bool = True,
        any_top_p: bool = True,
    ) -> torch.Tensor:
        if not any_top_k and not any_top_p:
            return _eager_sample_temp_only(logits, temperatures)
        if self._compiled_sample is not None and not self._compile_failed:
            try:
                return self._compiled_sample(logits, temperatures, top_k, top_p)
            except Exception:
                self._compile_failed = True
        return _eager_sample(logits, temperatures, top_k, top_p)

def _apply_top_k(logits: torch.Tensor, top_k: torch.Tensor) -> torch.Tensor:
    needs_top_k = (top_k != TOP_K_DISABLED_SENTINEL).any()
    if not needs_top_k:
        return logits
    vocab_size = logits.size(LAST_DIM)
    safe_k = torch.where(
        top_k == TOP_K_DISABLED_SENTINEL,
        torch.full_like(top_k, vocab_size),
        top_k,
    ).clamp(max=vocab_size)
    max_k = int(safe_k.max().item())
    topk_values, _ = torch.topk(logits, max_k, dim=LAST_DIM)
    threshold_index = (safe_k - 1).clamp(min=0).unsqueeze(1).long()
    threshold = topk_values.gather(LAST_DIM, threshold_index)
    keep_logits = logits >= threshold
    return torch.where(keep_logits, logits, torch.full_like(logits, NEG_INF_FILL))

def _apply_top_p(logits: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    needs_top_p = (top_p < TOP_P_DISABLED_SENTINEL).any()
    if not needs_top_p:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, dim=LAST_DIM, descending=True)
    sorted_probs = torch.softmax(sorted_logits, dim=LAST_DIM)
    cumulative_probs = sorted_probs.cumsum(dim=LAST_DIM)
    prev_cumulative = cumulative_probs - sorted_probs
    keep_sorted = prev_cumulative < top_p.unsqueeze(1)
    keep_unsorted = torch.empty_like(keep_sorted)
    keep_unsorted.scatter_(LAST_DIM, sorted_indices, keep_sorted)
    return torch.where(keep_unsorted, logits, torch.full_like(logits, NEG_INF_FILL))
