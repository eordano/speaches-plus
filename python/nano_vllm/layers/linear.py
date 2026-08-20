from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
from torch import nn

from nano_vllm.layers.quantization import (
    LinearMethodBase,
    QuantizationConfig,
    UnquantizedLinearMethod,
    set_weight_attrs,
)

PACKED_OUTPUT_DIM = 0
INPUT_DIM_FOR_ROW_PARALLEL = 1
DEFAULT_PREFIX = ""
SHARD_ID_Q = "q"
SHARD_ID_K = "k"
SHARD_ID_V = "v"
QKV_SHARD_IDS = (SHARD_ID_Q, SHARD_ID_K, SHARD_ID_V)
NUM_KV_PROJECTIONS = 2
BIAS_VECTOR_NDIM = 1

def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0
    return numerator // denominator

def select_quant_method(
    layer: nn.Module,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> LinearMethodBase:
    if quant_config is None:
        return UnquantizedLinearMethod()
    return quant_config.get_quant_method(layer, prefix) or UnquantizedLinearMethod()

class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool,
        tp_dim: int | None,
        params_dtype: torch.dtype | None,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.params_dtype = params_dtype if params_dtype is not None else torch.get_default_dtype()
        self.quant_config = quant_config
        self.prefix = prefix
        self.quant_method = select_quant_method(self, quant_config, prefix)

    def _init_params(
        self,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        bias: bool,
        bias_size: int,
    ) -> None:
        self.output_partition_sizes = output_partition_sizes
        self.quant_method.create_weights(
            layer=self,
            input_size_per_partition=input_size_per_partition,
            output_partition_sizes=output_partition_sizes,
            input_size=self.input_size,
            output_size=self.output_size,
            params_dtype=self.params_dtype,
            weight_loader=self.weight_loader,
        )
        self._register_bias(bias_size, bias)

    def _register_bias(self, bias_size: int, bias: bool) -> None:
        if not bias:
            self.register_parameter("bias", None)
            return
        bias_param = nn.Parameter(
            torch.empty(bias_size, dtype=self.params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(bias_param, {"output_dim": PACKED_OUTPUT_DIM, "weight_loader": self.weight_loader})
        self.register_parameter("bias", bias_param)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, *args: Any) -> None:
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        super().__init__(input_size, output_size, bias, None, params_dtype, quant_config, prefix)
        self._init_params(input_size, [output_size], bias, output_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x, bias=self.bias)

class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = DEFAULT_PREFIX,
        output_sizes: list[int] | None = None,
    ) -> None:
        super().__init__(input_size, output_size, bias, PACKED_OUTPUT_DIM, params_dtype, quant_config, prefix)
        per_rank_total = divide(output_size, self.tp_size)
        if output_sizes is None:
            partitions = [per_rank_total]
        else:
            partitions = [divide(size, self.tp_size) for size in output_sizes]
        self.output_size_per_partition = per_rank_total
        self._init_params(input_size, partitions, bias, per_rank_total)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.quant_method.apply(self, x, bias=self.bias)

class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        self.output_sizes = output_sizes
        super().__init__(
            input_size, sum(output_sizes), bias, params_dtype, quant_config, prefix, output_sizes,
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int,
    ) -> None:
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.total_num_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        q_proj_size = total_num_heads * head_size
        kv_proj_size = total_num_kv_heads * head_size
        super().__init__(
            hidden_size,
            q_proj_size + NUM_KV_PROJECTIONS * kv_proj_size,
            bias,
            params_dtype,
            quant_config,
            prefix,
            output_sizes=[q_proj_size, kv_proj_size, kv_proj_size],
        )

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ) -> None:
        assert loaded_shard_id in QKV_SHARD_IDS
        param_data = param.data
        q_shard_size = self.num_heads * self.head_size
        kv_shard_size = self.num_kv_heads * self.head_size
        if loaded_shard_id == SHARD_ID_Q:
            shard_size, shard_offset = q_shard_size, 0
        elif loaded_shard_id == SHARD_ID_K:
            shard_size, shard_offset = kv_shard_size, q_shard_size
        else:
            shard_size, shard_offset = kv_shard_size, q_shard_size + kv_shard_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)

class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        params_dtype: torch.dtype | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = DEFAULT_PREFIX,
    ) -> None:
        super().__init__(
            input_size, output_size, bias, INPUT_DIM_FOR_ROW_PARALLEL, params_dtype, quant_config, prefix,
        )
        self.input_size_per_partition = divide(input_size, self.tp_size)
        self._init_params(self.input_size_per_partition, [output_size], bias, output_size)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        param_data = param.data
        if param_data.ndim == BIAS_VECTOR_NDIM:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias_for_apply = self.bias if self.tp_rank == 0 else None
        y = self.quant_method.apply(self, x, bias=bias_for_apply)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
