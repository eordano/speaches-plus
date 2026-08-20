import inspect
import pickle
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Event
from typing import Any

import torch
import torch.distributed as dist

from nano_vllm.config import (
    BYTES_PER_GIB,
    KV_CACHE_DTYPE_AUTO,
    Config,
)
from nano_vllm.engine.scheduler import (
    SCHEDULE_MODE_DECODE,
    SCHEDULE_MODE_PREFILL,
    SCHEDULE_MODE_VERIFY,
)
from nano_vllm.engine.sequence import Sequence
from nano_vllm.layers.attention import (
    ATTN_BACKEND_FLASHINFER,
    FLASHINFER_KV_LAYOUT,
    KV_INDEX_KEYS,
    KV_INDEX_VALUES,
    KV_TENSOR_DIMS,
    _resolve_backend,
    fp8_dtype_for,
)
from nano_vllm.layers.grammar import apply_grammar_mask, apply_grammar_mask_verify
from nano_vllm.layers.quantization import QuantizationConfig
from nano_vllm.layers.quantization.compressed_tensors import CompressedTensorsConfig
from nano_vllm.layers.quantization.fp8 import Fp8Config
from nano_vllm.layers.sampler import Sampler
from nano_vllm.models.registry import resolve_model_class
from nano_vllm.utils.context import get_context, reset_context, set_context
from nano_vllm.utils.loader import load_model
from nano_vllm.utils.pinned_scratch import host_view

try:
    import flashinfer as _flashinfer_runtime
    _HAS_FLASHINFER_RUNTIME = True
except ImportError:
    _HAS_FLASHINFER_RUNTIME = False
    _flashinfer_runtime = None

QUANTIZATION_FP8 = "fp8"
QUANTIZATION_COMPRESSED_TENSORS = "compressed-tensors"
HF_QUANTIZATION_CONFIG_ATTR = "quantization_config"
HF_QUANT_METHOD_KEY = "quant_method"
WARMUP_DECODE_BATCH_SIZE = 1
WARMUP_DECODE_FAKE_BLOCK = 0
FLASHINFER_WORKSPACE_BYTES = 128 * 1024 * 1024
FLASHINFER_WORKSPACE_DTYPE = torch.uint8
DECODE_GRAPH_MAX_BS = 512
DECODE_GRAPH_TIER_BREAKPOINTS = (
    (8, 1),
    (16, 2),
    (32, 4),
    (64, 8),
    (128, 16),
    (256, 32),
    (DECODE_GRAPH_MAX_BS, 64),
)
PREFILL_INPUT_THRESHOLD = 512

def _decode_graph_tiers(max_bs: int) -> list[int]:
    tiers: list[int] = []
    lower = 1
    for upper, step in DECODE_GRAPH_TIER_BREAKPOINTS:
        ceiling = min(upper, max_bs)
        if lower > ceiling:
            break
        first_in_range = lower if step == 1 else lower + (step - 1)
        for batch_size in range(first_in_range, ceiling + 1, step):
            tiers.append(batch_size)
        if tiers and tiers[-1] < ceiling:
            tiers.append(ceiling)
        lower = upper + 1
    if tiers[-1] != max_bs:
        tiers.append(max_bs)
    return tiers
DIST_BACKEND = "nccl"
DIST_INIT_URL = "tcp://localhost:2333"
SHM_NAME = "nano_vllm"
SHM_SIZE_BYTES = 2 ** 20
SHM_HEADER_BYTES = 4
SHM_HEADER_BYTEORDER = "little"

def _build_quant_config(config: Config) -> QuantizationConfig | None:
    quantization = config.quantization
    hf_quant_config = _hf_quantization_config(config)
    if quantization is None and hf_quant_config is not None:
        if hf_quant_config.get(HF_QUANT_METHOD_KEY) == QUANTIZATION_COMPRESSED_TENSORS:
            return CompressedTensorsConfig.from_config(hf_quant_config)
        if hf_quant_config.get(HF_QUANT_METHOD_KEY) == QUANTIZATION_FP8:
            return Fp8Config.from_config(hf_quant_config)
    if quantization is None:
        return None
    if quantization == QUANTIZATION_FP8:
        return Fp8Config.from_config(hf_quant_config or {})
    if quantization == QUANTIZATION_COMPRESSED_TENSORS:
        if hf_quant_config is None:
            raise ValueError(
                "quantization='compressed-tensors' requires the model's hf_config "
                "to expose a quantization_config block."
            )
        return CompressedTensorsConfig.from_config(hf_quant_config)
    raise ValueError(f"Unknown quantization scheme: {quantization!r}")

def _hf_quantization_config(config: Config) -> dict[str, Any] | None:
    hf_config = config.hf_config
    if hf_config is None:
        return None
    raw = getattr(hf_config, HF_QUANTIZATION_CONFIG_ATTR, None)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "to_dict"):
        return raw.to_dict()
    return dict(raw)

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        assert config.hf_config is not None
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        self.attn_backend = _resolve_backend()
        self.kv_cache_dtype = (
            fp8_dtype_for(config.kv_cache_dtype) if config.kv_cache_dtype != KV_CACHE_DTYPE_AUTO else None
        )
        self.flashinfer_workspace: torch.Tensor | None = None
        self.flashinfer_prefill_wrappers: list[Any] = []
        self.flashinfer_decode_wrappers: list[Any] = []
        self.attention_layers: list[Any] = []

        if self.kv_cache_dtype is not None and self.attn_backend != ATTN_BACKEND_FLASHINFER:
            raise RuntimeError(
                f"kv_cache_dtype={config.kv_cache_dtype!r} requires the flashinfer "
                f"backend; resolved backend was {self.attn_backend!r}."
            )

        dist.init_process_group(DIST_BACKEND, DIST_INIT_URL, world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        quant_config = _build_quant_config(config)
        target_cls = resolve_model_class(hf_config)
        model_kwargs: dict[str, Any] = dict(
            quant_config=quant_config,
            use_torch_compile=config.use_torch_compile,
        )
        aux_layer_ids = config.eagle3_aux_hidden_layer_ids
        target_params = inspect.signature(target_cls.__init__).parameters
        target_supports_aux = "aux_hidden_layer_ids" in target_params
        if aux_layer_ids is not None and target_supports_aux:
            model_kwargs["aux_hidden_layer_ids"] = list(aux_layer_ids)
        elif aux_layer_ids is not None and config.enable_eagle3_spec_decode:
            raise RuntimeError(
                "EAGLE-3 spec decode requires a target with "
                "aux_hidden_layer_ids support; current model class "
                f"{target_cls.__name__} does not implement the hook."
            )
        self.model = target_cls(hf_config, **model_kwargs)
        self.aux_hidden_layer_ids = aux_layer_ids if target_supports_aux else None
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        self._warmup_decode_shape()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name=SHM_NAME, create=True, size=SHM_SIZE_BYTES)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name=SHM_NAME)
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        header_size = int.from_bytes(self.shm.buf[0:SHM_HEADER_BYTES], SHM_HEADER_BYTEORDER)
        method_name, *args = pickle.loads(self.shm.buf[SHM_HEADER_BYTES:header_size + SHM_HEADER_BYTES])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        assert self.shm.buf is not None
        data = pickle.dumps([method_name, *args])
        size = len(data)
        self.shm.buf[0:SHM_HEADER_BYTES] = size.to_bytes(SHM_HEADER_BYTES, SHM_HEADER_BYTEORDER)
        self.shm.buf[SHM_HEADER_BYTES:size + SHM_HEADER_BYTES] = data
        assert isinstance(self.event, list)
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, SCHEDULE_MODE_PREFILL)
        torch.cuda.empty_cache()

    def _warmup_decode_shape(self):
        decode_input_ids = torch.zeros(WARMUP_DECODE_BATCH_SIZE, dtype=torch.int64, device="cuda")
        decode_positions = torch.zeros(WARMUP_DECODE_BATCH_SIZE, dtype=torch.int64, device="cuda")
        slot_mapping = torch.full(
            (WARMUP_DECODE_BATCH_SIZE,), WARMUP_DECODE_FAKE_BLOCK, dtype=torch.int32, device="cuda"
        )
        context_lens = torch.ones(WARMUP_DECODE_BATCH_SIZE, dtype=torch.int32, device="cuda")
        block_tables = torch.zeros(WARMUP_DECODE_BATCH_SIZE, 1, dtype=torch.int32, device="cuda")
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        with torch.inference_mode():
            self.model(decode_input_ids, decode_positions)
        reset_context()

    def collect_attention_layers(self):
        layers = []
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                layers.append(module)
        return layers

    def _kv_storage_dtype(self) -> torch.dtype:
        if self.kv_cache_dtype is not None:
            return self.kv_cache_dtype
        assert self.config.hf_config is not None
        return self.config.hf_config.dtype

    def allocate_kv_cache(self):
        config = self.config
        assert config.hf_config is not None
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        attention_layers = self.collect_attention_layers()
        self.attention_layers = attention_layers
        if attention_layers:
            per_layer_kv_heads = [layer.num_kv_heads for layer in attention_layers]
            per_layer_head_dims = [layer.head_dim for layer in attention_layers]
        else:
            fallback_num_kv_heads = hf_config.num_key_value_heads // self.world_size
            fallback_head_dim = getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            )
            per_layer_kv_heads = [fallback_num_kv_heads] * hf_config.num_hidden_layers
            per_layer_head_dims = [fallback_head_dim] * hf_config.num_hidden_layers
        max_num_kv_heads = max(per_layer_kv_heads)
        max_head_dim = max(per_layer_head_dims)
        storage_dtype = self._kv_storage_dtype()
        block_bytes = (
            KV_TENSOR_DIMS
            * hf_config.num_hidden_layers
            * self.block_size
            * max_num_kv_heads
            * max_head_dim
            * storage_dtype.itemsize
        )
        budget_from_utilization = total * config.gpu_memory_utilization
        budget_from_explicit_cap = config.max_gpu_memory_bytes
        if budget_from_explicit_cap is not None:
            effective_budget = min(budget_from_utilization, budget_from_explicit_cap)
        else:
            effective_budget = budget_from_utilization
        config.num_kvcache_blocks = int(
            effective_budget - used - peak + current
        ) // block_bytes
        assert config.num_kvcache_blocks > 0, (
            f"No KV cache budget left after model load. effective_budget="
            f"{effective_budget / BYTES_PER_GIB:.2f} GiB, used={used / BYTES_PER_GIB:.2f}, "
            f"peak={peak / BYTES_PER_GIB:.2f}, current={current / BYTES_PER_GIB:.2f}. "
            f"Lower max_gpu_memory_gb or raise gpu_memory_utilization."
        )

        if self.attn_backend == ATTN_BACKEND_FLASHINFER:
            self.kv_cache = torch.empty(
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                KV_TENSOR_DIMS,
                self.block_size,
                max_num_kv_heads,
                max_head_dim,
                dtype=storage_dtype,
            )
            for layer_index, layer in enumerate(attention_layers):
                layer_num_kv_heads = per_layer_kv_heads[layer_index]
                layer_head_dim = per_layer_head_dims[layer_index]
                layer_kv = self.kv_cache[
                    layer_index, :, :, :, :layer_num_kv_heads, :layer_head_dim
                ]
                layer.k_cache = layer_kv[:, KV_INDEX_KEYS]
                layer.v_cache = layer_kv[:, KV_INDEX_VALUES]
            self._init_flashinfer_state(attention_layers)
        else:
            self.kv_cache = torch.empty(
                KV_TENSOR_DIMS,
                hf_config.num_hidden_layers,
                config.num_kvcache_blocks,
                self.block_size,
                max_num_kv_heads,
                max_head_dim,
                dtype=storage_dtype,
            )
            for layer_index, layer in enumerate(attention_layers):
                layer_num_kv_heads = per_layer_kv_heads[layer_index]
                layer_head_dim = per_layer_head_dims[layer_index]
                layer.k_cache = self.kv_cache[
                    KV_INDEX_KEYS, layer_index, :, :, :layer_num_kv_heads, :layer_head_dim
                ]
                layer.v_cache = self.kv_cache[
                    KV_INDEX_VALUES, layer_index, :, :, :layer_num_kv_heads, :layer_head_dim
                ]

    def _init_flashinfer_state(self, attention_layers: list) -> None:
        if not _HAS_FLASHINFER_RUNTIME:
            raise RuntimeError("flashinfer runtime not importable; cannot init wrappers")
        self.flashinfer_workspace = torch.empty(
            FLASHINFER_WORKSPACE_BYTES,
            dtype=FLASHINFER_WORKSPACE_DTYPE,
            device="cuda",
        )
        self.flashinfer_prefill_wrappers = []
        self.flashinfer_decode_wrappers = []
        for layer_index, layer in enumerate(attention_layers):
            prefill_wrapper = _flashinfer_runtime.BatchPrefillWithPagedKVCacheWrapper(
                self.flashinfer_workspace, FLASHINFER_KV_LAYOUT
            )
            decode_wrapper = _flashinfer_runtime.BatchDecodeWithPagedKVCacheWrapper(
                self.flashinfer_workspace, FLASHINFER_KV_LAYOUT
            )
            layer_kv_view = self.kv_cache[layer_index]
            layer.attach_flashinfer_wrappers(
                prefill_wrapper,
                decode_wrapper,
                layer_kv_view,
                self.kv_cache_dtype,
            )
            self.flashinfer_prefill_wrappers.append(prefill_wrapper)
            self.flashinfer_decode_wrappers.append(decode_wrapper)

    def _build_flashinfer_prefill_meta(self, seqs: list[Sequence]):
        qo_indptr = [0]
        kv_indptr = [0]
        kv_indices_flat: list[int] = []
        kv_last_page_len: list[int] = []
        batch_indices: list[int] = []
        positions_in_page: list[int] = []
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            qo_indptr.append(qo_indptr[-1] + seqlen_q)
            num_pages = (end + self.block_size - 1) // self.block_size
            pages = seq.block_table[:num_pages] if seq.block_table else []
            kv_indices_flat.extend(pages)
            kv_indptr.append(kv_indptr[-1] + len(pages))
            last_page_tokens = end - (num_pages - 1) * self.block_size if num_pages > 0 else 0
            kv_last_page_len.append(last_page_tokens)
            for token_offset in range(seqlen_q):
                absolute_pos = start + token_offset
                page_local_index = absolute_pos // self.block_size
                page_offset = absolute_pos % self.block_size
                page_global = seq.block_table[page_local_index] if seq.block_table else -1
                batch_indices.append(page_global)
                positions_in_page.append(page_offset)
        return (
            torch.tensor(qo_indptr, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(kv_indptr, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(kv_indices_flat, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(kv_last_page_len, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(batch_indices, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(positions_in_page, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
        )

    def _build_flashinfer_decode_meta(self, seqs: list[Sequence]):
        kv_indptr = [0]
        kv_indices_flat: list[int] = []
        kv_last_page_len: list[int] = []
        batch_indices: list[int] = []
        positions_in_page: list[int] = []
        for seq in seqs:
            num_pages = (len(seq) + self.block_size - 1) // self.block_size
            pages = seq.block_table[:num_pages]
            kv_indices_flat.extend(pages)
            kv_indptr.append(kv_indptr[-1] + len(pages))
            last_page_tokens = len(seq) - (num_pages - 1) * self.block_size
            kv_last_page_len.append(last_page_tokens)
            last_token_pos = len(seq) - 1
            page_local_index = last_token_pos // self.block_size
            batch_indices.append(seq.block_table[page_local_index])
            positions_in_page.append(last_token_pos % self.block_size)
        return (
            torch.tensor(kv_indptr, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(kv_indices_flat, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(kv_last_page_len, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(batch_indices, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
            torch.tensor(positions_in_page, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True),
        )

    def _plan_flashinfer_prefill(self, qo_indptr, kv_indptr, kv_indices, kv_last_page_len) -> None:
        if not self.attention_layers:
            return
        sample_layer = self.attention_layers[0]
        assert self.config.hf_config is not None
        kv_data_type = self.kv_cache_dtype if self.kv_cache_dtype is not None else self.config.hf_config.dtype
        for wrapper in self.flashinfer_prefill_wrappers:
            wrapper.plan(
                qo_indptr=qo_indptr,
                paged_kv_indptr=kv_indptr,
                paged_kv_indices=kv_indices,
                paged_kv_last_page_len=kv_last_page_len,
                num_qo_heads=sample_layer.num_heads,
                num_kv_heads=sample_layer.num_kv_heads,
                head_dim_qk=sample_layer.head_dim,
                page_size=self.block_size,
                causal=True,
                q_data_type=self.config.hf_config.dtype,
                kv_data_type=kv_data_type,
            )

    def _plan_flashinfer_decode(self, kv_indptr, kv_indices, kv_last_page_len) -> None:
        if not self.attention_layers:
            return
        sample_layer = self.attention_layers[0]
        assert self.config.hf_config is not None
        kv_data_type = self.kv_cache_dtype if self.kv_cache_dtype is not None else self.config.hf_config.dtype
        for wrapper in self.flashinfer_decode_wrappers:
            wrapper.plan(
                indptr=kv_indptr,
                indices=kv_indices,
                last_page_len=kv_last_page_len,
                num_qo_heads=sample_layer.num_heads,
                num_kv_heads=sample_layer.num_kv_heads,
                head_dim=sample_layer.head_dim,
                page_size=self.block_size,
                q_data_type=self.config.hf_config.dtype,
                kv_data_type=kv_data_type,
            )

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for block_index in range(start_block, end_block):
                slot_start = seq.block_table[block_index] * self.block_size
                if block_index == start_block:
                    slot_start += start % self.block_size
                if block_index != end_block - 1:
                    slot_end = seq.block_table[block_index] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[block_index] * self.block_size + end - block_index * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        if self.attn_backend == ATTN_BACKEND_FLASHINFER:
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len, batch_indices, positions_in_page = (
                self._build_flashinfer_prefill_meta(seqs)
            )
            if any(seq.block_table for seq in seqs):
                self._plan_flashinfer_prefill(qo_indptr, kv_indptr, kv_indices, kv_last_page_len)
            set_context(
                True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                slot_mapping, None, block_tables,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_last_page_len=kv_last_page_len,
                qo_indptr=qo_indptr,
                batch_indices=batch_indices,
                positions_in_page=positions_in_page,
            )
        else:
            set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        n = len(seqs)
        bs = self.block_size
        input_ids_h = host_view("decode_input_ids", torch.int64, n)
        positions_h = host_view("decode_positions", torch.int64, n)
        slot_h = host_view("decode_slot", torch.int32, n)
        ctx_h = host_view("decode_ctx", torch.int32, n)
        in_np = input_ids_h.numpy()
        pos_np = positions_h.numpy()
        slot_np = slot_h.numpy()
        ctx_np = ctx_h.numpy()
        for i, seq in enumerate(seqs):
            last_pos = len(seq) - 1
            in_np[i] = seq.last_token
            pos_np[i] = last_pos
            ctx_np[i] = len(seq)
            slot_np[i] = seq.block_table[last_pos // bs] * bs + last_pos % bs
        input_ids = input_ids_h.cuda(non_blocking=True)
        positions = positions_h.cuda(non_blocking=True)
        slot_mapping = slot_h.cuda(non_blocking=True)
        context_lens = ctx_h.cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        if self.attn_backend == ATTN_BACKEND_FLASHINFER:
            kv_indptr, kv_indices, kv_last_page_len, batch_indices, positions_in_page = (
                self._build_flashinfer_decode_meta(seqs)
            )
            self._plan_flashinfer_decode(kv_indptr, kv_indices, kv_last_page_len)
            set_context(
                False,
                slot_mapping=slot_mapping,
                context_lens=context_lens,
                block_tables=block_tables,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_last_page_len=kv_last_page_len,
                batch_indices=batch_indices,
                positions_in_page=positions_in_page,
            )
        else:
            set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_verify(self, seqs: list[Sequence]):
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        for seq in seqs:
            tokens, pos = seq.verify_inputs()
            seqlen_q = len(tokens)
            start = pos[0]
            end = pos[-1] + 1
            seqlen_k = end
            input_ids.extend(tokens)
            positions.extend(pos)
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for block_index in range(start_block, end_block):
                slot_start = seq.block_table[block_index] * self.block_size
                if block_index == start_block:
                    slot_start += start % self.block_size
                if block_index != end_block - 1:
                    slot_end = seq.block_table[block_index] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[block_index] * self.block_size + end - block_index * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        block_tables = self.prepare_block_tables(seqs)

        def _to_cuda(name: str, dtype: torch.dtype, values: list[int]) -> torch.Tensor:
            buf = host_view(name, dtype, len(values))
            buf.numpy()[:] = values
            return buf.cuda(non_blocking=True)

        input_ids_t = _to_cuda("verify_input_ids", torch.int64, input_ids)
        positions_t = _to_cuda("verify_positions", torch.int64, positions)
        cu_seqlens_q_t = _to_cuda("verify_cu_q", torch.int32, cu_seqlens_q)
        cu_seqlens_k_t = _to_cuda("verify_cu_k", torch.int32, cu_seqlens_k)
        slot_mapping_t = _to_cuda("verify_slot", torch.int32, slot_mapping)
        if self.attn_backend == ATTN_BACKEND_FLASHINFER:
            qo_indptr, kv_indptr, kv_indices, kv_last_page_len, batch_indices, positions_in_page = (
                self._build_flashinfer_prefill_meta(seqs)
            )
            self._plan_flashinfer_prefill(qo_indptr, kv_indptr, kv_indices, kv_last_page_len)
            set_context(
                True, cu_seqlens_q_t, cu_seqlens_k_t, max_seqlen_q, max_seqlen_k,
                slot_mapping_t, None, block_tables,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_last_page_len=kv_last_page_len,
                qo_indptr=qo_indptr,
                batch_indices=batch_indices,
                positions_in_page=positions_in_page,
                verify_mode=True,
            )
        else:
            set_context(
                True, cu_seqlens_q_t, cu_seqlens_k_t, max_seqlen_q, max_seqlen_k,
                slot_mapping_t, None, block_tables,
                verify_mode=True,
            )
        return input_ids_t, positions_t

    def prepare_sample(self, seqs: list[Sequence]):
        n = len(seqs)
        temp_h = host_view("sample_temp", torch.float32, n)
        topk_h = host_view("sample_topk", torch.int32, n)
        topp_h = host_view("sample_topp", torch.float32, n)
        temp_np = temp_h.numpy()
        topk_np = topk_h.numpy()
        topp_np = topp_h.numpy()
        any_top_k = False
        any_top_p = False
        for i, seq in enumerate(seqs):
            temp_np[i] = seq.temperature
            topk_np[i] = seq.top_k
            topp_np[i] = seq.top_p
            if seq.top_k != -1:
                any_top_k = True
            if seq.top_p < 1.0:
                any_top_p = True
        return (
            temp_h.cuda(non_blocking=True),
            topk_h.cuda(non_blocking=True),
            topp_h.cuda(non_blocking=True),
            any_top_k,
            any_top_p,
        )

    def _slice_last_aux(
        self,
        aux_hidden_states: torch.Tensor,
        flat: bool,
        per_seq_offsets: list[int] | None = None,
    ) -> torch.Tensor:
        if not flat:
            return aux_hidden_states
        context = get_context()
        cu_seqlens_q = getattr(context, "cu_seqlens_q", None)
        if cu_seqlens_q is None:
            return aux_hidden_states
        if per_seq_offsets is None:
            indices = cu_seqlens_q[1:] - 1
        else:
            starts = cu_seqlens_q[:-1]
            offsets_t = torch.tensor(per_seq_offsets, dtype=starts.dtype, device=starts.device)
            indices = starts + offsets_t
        return aux_hidden_states.index_select(0, indices.to(aux_hidden_states.device))

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        runner_state: dict,
    ):
        if self.aux_hidden_layer_ids is not None:
            is_prefill = True
        if is_prefill or self.enforce_eager or input_ids.size(0) > PREFILL_INPUT_THRESHOLD:
            output = self.model(input_ids, positions)
            if self.aux_hidden_layer_ids is not None:
                hidden_states, aux_hidden_states = output
                runner_state["raw_aux_hidden_states"] = aux_hidden_states
                runner_state["aux_flat"] = True
                return self.model.compute_logits(hidden_states)
            return self.model.compute_logits(output)
        if self.attn_backend == ATTN_BACKEND_FLASHINFER:
            output = self.model(input_ids, positions)
            if self.aux_hidden_layer_ids is not None:
                hidden_states, aux_hidden_states = output
                runner_state["raw_aux_hidden_states"] = aux_hidden_states
                runner_state["aux_flat"] = False
                return self.model.compute_logits(hidden_states)
            return self.model.compute_logits(output)
        batch_size = input_ids.size(0)
        context = get_context()
        graph = self.graphs[next(graph_bs for graph_bs in self.graph_bs if graph_bs >= batch_size)]
        graph_vars = self.graph_vars
        graph_vars["input_ids"][:batch_size] = input_ids
        graph_vars["positions"][:batch_size] = positions
        graph_vars["slot_mapping"].fill_(-1)
        graph_vars["slot_mapping"][:batch_size] = context.slot_mapping
        graph_vars["context_lens"].zero_()
        graph_vars["context_lens"][:batch_size] = context.context_lens
        graph_vars["block_tables"][:batch_size, :context.block_tables.size(1)] = context.block_tables
        graph.replay()
        return self.model.compute_logits(graph_vars["outputs"][:batch_size])

    def run(
        self,
        seqs: list[Sequence],
        mode: str,
        matchers: list | None = None,
    ):
        runner_state: dict = {}
        if mode == SCHEDULE_MODE_VERIFY:
            input_ids, positions = self.prepare_verify(seqs)
        elif mode == SCHEDULE_MODE_PREFILL:
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)
        sample_inputs = self.prepare_sample(seqs) if self.rank == 0 else None
        is_prefill_kernel = mode != SCHEDULE_MODE_DECODE
        logits = self.run_model(input_ids, positions, is_prefill_kernel, runner_state)
        per_seq_aux_offsets: list[int] | None = None
        if self.rank == 0:
            assert sample_inputs is not None
            temperatures, top_k, top_p, any_top_k, any_top_p = sample_inputs
            if mode == SCHEDULE_MODE_VERIFY:
                output = self._sample_verify(
                    seqs, logits, temperatures, top_k, top_p, matchers, any_top_k, any_top_p
                )
                per_seq_aux_offsets = [accepted_count for accepted_count, _, _ in output]
            else:
                if matchers is not None and any(matcher is not None for matcher in matchers):
                    logits = apply_grammar_mask(logits, matchers, logits.size(-1))
                output = self.sampler(
                    logits, temperatures, top_k, top_p, any_top_k=any_top_k, any_top_p=any_top_p
                ).squeeze(-1).tolist()
        else:
            output = None
        if "raw_aux_hidden_states" in runner_state:
            raw_aux = runner_state.pop("raw_aux_hidden_states")
            aux_flat = runner_state.pop("aux_flat")
            runner_state["last_aux_hidden_states"] = self._slice_last_aux(
                raw_aux, flat=aux_flat, per_seq_offsets=per_seq_aux_offsets
            )
        reset_context()
        return output, runner_state

    def _sample_verify(
        self,
        seqs: list[Sequence],
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        matchers: list | None,
        any_top_k: bool = True,
        any_top_p: bool = True,
    ) -> list[tuple[int, int, list[int]]]:
        per_seq_lengths = [1 + len(seq.draft_tokens) for seq in seqs]
        if matchers is not None and any(m is not None for m in matchers):
            draft_tokens_per_seq = [list(seq.draft_tokens) for seq in seqs]
            logits = apply_grammar_mask_verify(
                logits, matchers, draft_tokens_per_seq, logits.size(-1),
            )
        repeats_h = host_view("verify_repeats", torch.int64, len(per_seq_lengths))
        repeats_h.numpy()[:] = per_seq_lengths
        repeats = repeats_h.to(temperatures.device, non_blocking=True)
        expanded_temperatures = temperatures.repeat_interleave(repeats)
        expanded_top_k = top_k.repeat_interleave(repeats)
        expanded_top_p = top_p.repeat_interleave(repeats)
        sampled = self.sampler(
            logits, expanded_temperatures, expanded_top_k, expanded_top_p,
            any_top_k=any_top_k, any_top_p=any_top_p,
        ).squeeze(-1).tolist()
        outputs: list[tuple[int, int, list[int]]] = []
        cursor = 0
        for seq, length in zip(seqs, per_seq_lengths):
            seq_sampled = sampled[cursor:cursor + length]
            cursor += length
            drafts = seq.draft_tokens
            accepted_drafts: list[int] = []
            for i, draft in enumerate(drafts):
                if seq_sampled[i] == draft:
                    accepted_drafts.append(draft)
                else:
                    break
            accepted_count = len(accepted_drafts)
            bonus_token_id = seq_sampled[accepted_count]
            outputs.append((accepted_count, bonus_token_id, accepted_drafts))
        return outputs

    @torch.inference_mode()
    def capture_cudagraph(self):
        if self.attn_backend == ATTN_BACKEND_FLASHINFER:
            self.graphs = {}
            self.graph_pool = None
            self.graph_bs = []
            self.graph_vars = {}
            return
        config = self.config
        assert config.hf_config is not None
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, DECODE_GRAPH_MAX_BS)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = _decode_graph_tiers(max_bs)
        self.graphs = {}
        self.graph_pool = None

        for batch_size in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:batch_size], context_lens=context_lens[:batch_size], block_tables=block_tables[:batch_size])
            outputs[:batch_size] = self.model(input_ids[:batch_size], positions[:batch_size])
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:batch_size] = self.model(input_ids[:batch_size], positions[:batch_size])
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[batch_size] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
