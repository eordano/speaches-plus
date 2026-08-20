import os
from dataclasses import dataclass
from typing import Literal

from transformers import AutoConfig

KVCACHE_BLOCK_SIZE_ALIGNMENT = 256
MIN_TENSOR_PARALLEL_SIZE = 1
MAX_TENSOR_PARALLEL_SIZE = 8
TORCH_COMPILE_ENV_VAR = "NANO_VLLM_TORCH_COMPILE"
TORCH_COMPILE_DISABLED_VALUE = "0"

ATTN_BACKEND_AUTO = "auto"
ATTN_BACKEND_FLASHINFER = "flashinfer"
ATTN_BACKEND_FLASH_ATTN = "flash_attn"
ATTN_BACKEND_TRITON = "triton"
ATTN_BACKEND_CHOICES = (
    ATTN_BACKEND_AUTO,
    ATTN_BACKEND_FLASHINFER,
    ATTN_BACKEND_FLASH_ATTN,
    ATTN_BACKEND_TRITON,
)

KV_CACHE_DTYPE_AUTO = "auto"
KV_CACHE_DTYPE_FP8 = "fp8"
KV_CACHE_DTYPE_FP8_E4M3 = "fp8_e4m3"
KV_CACHE_DTYPE_FP8_E5M2 = "fp8_e5m2"
KV_CACHE_DTYPE_CHOICES = (
    KV_CACHE_DTYPE_AUTO,
    KV_CACHE_DTYPE_FP8,
    KV_CACHE_DTYPE_FP8_E4M3,
    KV_CACHE_DTYPE_FP8_E5M2,
)

MAX_GPU_MEMORY_ENV_VAR = "NANO_VLLM_MAX_GPU_MEMORY_GB"
NGRAM_SPEC_DECODE_ENV_VAR = "NANO_VLLM_NGRAM_SPEC_DECODE"
NGRAM_SPEC_DECODE_ENABLED_VALUE = "1"
BYTES_PER_GIB = 1024 ** 3

@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    max_gpu_memory_gb: float | None = None
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    quantization: str | None = None
    use_torch_compile: bool = True
    attn_backend: Literal["auto", "flashinfer", "flash_attn", "triton"] = ATTN_BACKEND_AUTO
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_e5m2"] = KV_CACHE_DTYPE_AUTO
    enable_ngram_spec_decode: bool = False
    eagle3_aux_hidden_layer_ids: list[int] | None = None
    enable_eagle3_spec_decode: bool = False
    eagle3_speculator_path: str | None = None
    eagle3_num_drafts: int = 1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % KVCACHE_BLOCK_SIZE_ALIGNMENT == 0
        assert MIN_TENSOR_PARALLEL_SIZE <= self.tensor_parallel_size <= MAX_TENSOR_PARALLEL_SIZE
        assert self.attn_backend in ATTN_BACKEND_CHOICES
        assert self.kv_cache_dtype in KV_CACHE_DTYPE_CHOICES
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        if os.environ.get(TORCH_COMPILE_ENV_VAR) == TORCH_COMPILE_DISABLED_VALUE:
            self.use_torch_compile = False
        max_gpu_env = os.environ.get(MAX_GPU_MEMORY_ENV_VAR)
        if max_gpu_env and self.max_gpu_memory_gb is None:
            self.max_gpu_memory_gb = float(max_gpu_env)
        if self.max_gpu_memory_gb is not None and self.max_gpu_memory_gb <= 0:
            raise ValueError(
                f"max_gpu_memory_gb must be positive, got {self.max_gpu_memory_gb}"
            )
        if os.environ.get(NGRAM_SPEC_DECODE_ENV_VAR) == NGRAM_SPEC_DECODE_ENABLED_VALUE:
            self.enable_ngram_spec_decode = True
        if self.enable_eagle3_spec_decode:
            if not self.eagle3_speculator_path:
                raise ValueError(
                    "enable_eagle3_spec_decode=True requires eagle3_speculator_path to be set "
                    "(HF repo id or local directory containing the EAGLE-3 draft checkpoint)."
                )
            if not self.eagle3_aux_hidden_layer_ids:
                raise ValueError(
                    "enable_eagle3_spec_decode=True requires eagle3_aux_hidden_layer_ids to be set "
                    "(list of target-model layer indices to expose as aux hidden states)."
                )

    @property
    def max_gpu_memory_bytes(self) -> int | None:
        if self.max_gpu_memory_gb is None:
            return None
        return int(self.max_gpu_memory_gb * BYTES_PER_GIB)
