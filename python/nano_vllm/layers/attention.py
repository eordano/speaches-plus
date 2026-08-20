from __future__ import annotations

import os

import torch
from torch import nn

from nano_vllm.utils.context import get_context

ATTN_BACKEND_ENV_VAR = "NANO_VLLM_ATTN_BACKEND"
ATTN_BACKEND_AUTO = "auto"
ATTN_BACKEND_FLASHINFER = "flashinfer"
ATTN_BACKEND_FLASH_ATTN = "flash_attn"
ATTN_BACKEND_TRITON = "triton"
ATTN_BACKEND_NONE = "none"
ATTN_BACKEND_VALID = (
    ATTN_BACKEND_AUTO,
    ATTN_BACKEND_FLASHINFER,
    ATTN_BACKEND_FLASH_ATTN,
    ATTN_BACKEND_TRITON,
)

FLASHINFER_MIN_SM_MAJOR = 8
FLASHINFER_KV_LAYOUT = "NHD"
KV_TENSOR_DIMS = 2
KV_INDEX_KEYS = 0
KV_INDEX_VALUES = 1

FP8_E4M3_KEYS = ("fp8", "fp8_e4m3")
FP8_E5M2_KEYS = ("fp8_e5m2",)

try:
    import triton
    import triton.language as tl
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _HAS_FLASH_ATTN = True
except ImportError as _flash_attn_err:
    _HAS_FLASH_ATTN = False
    _FLASH_ATTN_IMPORT_ERROR = _flash_attn_err
    triton = None
    tl = None
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None

try:
    import flashinfer
    _HAS_FLASHINFER = True
except ImportError as _flashinfer_err:
    _HAS_FLASHINFER = False
    _FLASHINFER_IMPORT_ERROR = _flashinfer_err
    flashinfer = None

_RESOLVED_BACKEND: str | None = None

def _cuda_supports_flashinfer() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _minor = torch.cuda.get_device_capability(0)
    except (RuntimeError, AssertionError):
        return False
    return major >= FLASHINFER_MIN_SM_MAJOR

def _resolve_backend() -> str:

    global _RESOLVED_BACKEND
    if _RESOLVED_BACKEND is not None:
        return _RESOLVED_BACKEND
    requested = os.environ.get(ATTN_BACKEND_ENV_VAR, ATTN_BACKEND_AUTO).lower()
    if requested not in ATTN_BACKEND_VALID:
        raise ValueError(
            f"{ATTN_BACKEND_ENV_VAR}={requested!r} not in {ATTN_BACKEND_VALID}"
        )
    if requested == ATTN_BACKEND_FLASHINFER:
        if not _HAS_FLASHINFER:
            raise RuntimeError(
                "Backend 'flashinfer' explicitly requested but flashinfer not "
                f"installed: {_FLASHINFER_IMPORT_ERROR}"
            )
        _RESOLVED_BACKEND = ATTN_BACKEND_FLASHINFER
        return _RESOLVED_BACKEND
    if requested == ATTN_BACKEND_FLASH_ATTN:
        if not _HAS_FLASH_ATTN:
            raise RuntimeError(
                "Backend 'flash_attn' explicitly requested but flash-attn not "
                f"installed: {_FLASH_ATTN_IMPORT_ERROR}"
            )
        _RESOLVED_BACKEND = ATTN_BACKEND_FLASH_ATTN
        return _RESOLVED_BACKEND
    if requested == ATTN_BACKEND_TRITON:
        raise NotImplementedError("Triton backend not implemented yet")
    if _HAS_FLASHINFER and _cuda_supports_flashinfer():
        _RESOLVED_BACKEND = ATTN_BACKEND_FLASHINFER
        return _RESOLVED_BACKEND
    if _HAS_FLASH_ATTN:
        _RESOLVED_BACKEND = ATTN_BACKEND_FLASH_ATTN
        return _RESOLVED_BACKEND
    _RESOLVED_BACKEND = ATTN_BACKEND_NONE
    return _RESOLVED_BACKEND

def reset_backend_cache() -> None:
    global _RESOLVED_BACKEND
    _RESOLVED_BACKEND = None

def has_flashinfer() -> bool:
    return _HAS_FLASHINFER

def has_flash_attn() -> bool:
    return _HAS_FLASH_ATTN

def fp8_dtype_for(kv_cache_dtype: str | None) -> torch.dtype | None:
    if kv_cache_dtype is None:
        return None
    key = kv_cache_dtype.lower()
    if key in FP8_E4M3_KEYS:
        return torch.float8_e4m3fn
    if key in FP8_E5M2_KEYS:
        return torch.float8_e5m2
    return None

def make_kv_scale_remap(
    proj_suffixes: tuple[str, ...] = ("k_proj", "v_proj"),
    attn_attr: str = "attn",
):

    proj_to_target = {
        "k_proj": "k_scale",
        "v_proj": "v_scale",
    }

    def rule(weight_name: str) -> str | None:
        for proj in proj_suffixes:
            suffix = f".{proj}.output_scale"
            if weight_name.endswith(suffix):
                target = proj_to_target.get(proj)
                if target is None:
                    return None
                prefix = weight_name[: -len(suffix)]
                return f"{prefix}.{attn_attr}.{target}"
        return None

    return rule

if _HAS_FLASH_ATTN:
    @triton.jit
    def store_kvcache_kernel(
        key_ptr,
        key_stride,
        value_ptr,
        value_stride,
        k_cache_ptr,
        v_cache_ptr,
        slot_mapping_ptr,
        D: tl.constexpr,
    ):
        idx = tl.program_id(0)
        slot = tl.load(slot_mapping_ptr + idx)
        if slot == -1: return
        key_offsets = idx * key_stride + tl.arange(0, D)
        value_offsets = idx * value_stride + tl.arange(0, D)
        key = tl.load(key_ptr + key_offsets)
        value = tl.load(value_ptr + value_offsets)
        cache_offsets = slot * D + tl.arange(0, D)
        tl.store(k_cache_ptr + cache_offsets, key)
        tl.store(v_cache_ptr + cache_offsets, value)
else:
    store_kvcache_kernel = None

def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    num_tokens, num_heads, head_dim = key.shape
    flat_dim = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == flat_dim and v_cache.stride(1) == flat_dim
    assert slot_mapping.numel() == num_tokens
    store_kvcache_kernel[(num_tokens,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, flat_dim
    )

class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        self.kv_paged = torch.tensor([])
        self.register_buffer("k_scale", torch.ones(1), persistent=False)
        self.register_buffer("v_scale", torch.ones(1), persistent=False)
        self.register_buffer("q_scale", torch.ones(1), persistent=False)
        self.register_buffer("prob_scale", torch.ones(1), persistent=False)
        self._prefill_wrapper = None
        self._decode_wrapper = None
        self._kv_cache_dtype: torch.dtype | None = None

    def attach_flashinfer_wrappers(
        self,
        prefill_wrapper,
        decode_wrapper,
        kv_paged: torch.Tensor,
        kv_cache_dtype: torch.dtype | None,
    ) -> None:
        self._prefill_wrapper = prefill_wrapper
        self._decode_wrapper = decode_wrapper
        self.kv_paged = kv_paged
        self._kv_cache_dtype = kv_cache_dtype

    def _forward_flash_attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        if not _HAS_FLASH_ATTN:
            raise RuntimeError(
                "nano_vllm.Attention flash-attn backend requires flash-attn + triton "
                f"(CUDA only). Import failed: {_FLASH_ATTN_IMPORT_ERROR}. Install "
                "the [gpu] extra and run on a CUDA device."
            )
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            output = flash_attn_varlen_func(
                q, k, v,
                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale, causal=True, block_table=context.block_tables,
            )
        else:
            output = flash_attn_with_kvcache(
                q.unsqueeze(1), k_cache, v_cache,
                cache_seqlens=context.context_lens, block_table=context.block_tables,
                softmax_scale=self.scale, causal=True,
            )
        return output

    def _forward_flashinfer(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        kv_paged = self.kv_paged
        if kv_paged.numel():
            flashinfer.append_paged_kv_cache(
                k, v,
                batch_indices=context.batch_indices,
                positions=context.positions_in_page,
                paged_kv_cache=kv_paged,
                kv_indices=context.kv_indices,
                kv_indptr=context.kv_indptr,
                kv_last_page_len=context.kv_last_page_len,
                kv_layout=FLASHINFER_KV_LAYOUT,
            )
        if context.is_prefill:
            wrapper = self._prefill_wrapper
            if wrapper is None:
                raise RuntimeError("FlashInfer prefill wrapper not attached")
            output = wrapper.run(q, kv_paged, k_scale=self.k_scale, v_scale=self.v_scale)
        else:
            wrapper = self._decode_wrapper
            if wrapper is None:
                raise RuntimeError("FlashInfer decode wrapper not attached")
            output = wrapper.run(q, kv_paged, k_scale=self.k_scale, v_scale=self.v_scale)
        return output

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        backend = _resolve_backend()
        if backend == ATTN_BACKEND_FLASHINFER:
            return self._forward_flashinfer(q, k, v)
        if backend == ATTN_BACKEND_FLASH_ATTN:
            return self._forward_flash_attn(q, k, v)
        raise RuntimeError(
            "No attention backend available. Install flashinfer (CUDA SM>=80) "
            "or flash-attn + triton, or set "
            f"{ATTN_BACKEND_ENV_VAR} to override. flashinfer error: "
            f"{_FLASHINFER_IMPORT_ERROR if not _HAS_FLASHINFER else 'OK'}; "
            f"flash-attn error: "
            f"{_FLASH_ATTN_IMPORT_ERROR if not _HAS_FLASH_ATTN else 'OK'}."
        )
