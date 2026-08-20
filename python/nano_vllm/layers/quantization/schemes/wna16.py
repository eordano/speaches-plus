from __future__ import annotations

from typing import Any

import torch
from torch import nn

from nano_vllm.layers.quantization import (
    LinearMethodBase,
    set_weight_attrs,
)
from nano_vllm.layers.quantization.utils import marlin

WEIGHT_PACKED_PARAM = "weight_packed"
WEIGHT_SCALE_PARAM = "weight_scale"
WEIGHT_ZERO_POINT_PARAM = "weight_zero_point"
WEIGHT_SHAPE_PARAM = "weight_shape"

WEIGHT_PACKED_INPUT_DIM = 1
WEIGHT_PACKED_OUTPUT_DIM = 0
WEIGHT_PACKED_PACKED_DIM = 1
WEIGHT_SCALE_INPUT_DIM = 1
WEIGHT_SCALE_OUTPUT_DIM = 0
WEIGHT_SHAPE_NDIM = 2

DEFAULT_NUM_BITS = 4
WEIGHT_PACKED_DTYPE = torch.int32
ZERO_POINT_DTYPE = torch.int32
WEIGHT_SHAPE_DTYPE = torch.int64

GROUP_SIZE_CHANNELWISE = -1

class CompressedTensorsWNA16Method(LinearMethodBase):

    def __init__(
        self,
        group_size: int,
        num_bits: int = DEFAULT_NUM_BITS,
        symmetric: bool = True,
        has_zero_point: bool | None = None,
    ) -> None:
        if num_bits != DEFAULT_NUM_BITS:
            raise NotImplementedError(
                f"WNA16 num_bits={num_bits} not implemented; only 4-bit is supported."
            )
        self.num_bits = num_bits
        self.group_size = group_size
        self.symmetric = symmetric
        self.has_zero_point = (not symmetric) if has_zero_point is None else has_zero_point
        self.pack_factor = marlin.get_pack_factor(num_bits)

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs: Any,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        if input_size_per_partition % self.pack_factor != 0:
            raise ValueError(
                f"input_size_per_partition={input_size_per_partition} is not divisible "
                f"by pack_factor={self.pack_factor}"
            )
        if input_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"input_size_per_partition={input_size_per_partition} is not divisible "
                f"by group_size={self.group_size}"
            )
        num_groups_per_partition = input_size_per_partition // self.group_size

        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.quant_num_bits = self.num_bits
        layer.quant_group_size = self.group_size
        layer.quant_has_zero_point = self.has_zero_point

        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_packed = nn.Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=WEIGHT_PACKED_DTYPE,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight_packed,
            {
                "input_dim": WEIGHT_PACKED_INPUT_DIM,
                "output_dim": WEIGHT_PACKED_OUTPUT_DIM,
                "packed_dim": WEIGHT_PACKED_PACKED_DIM,
                "pack_factor": self.pack_factor,
            },
        )
        if weight_loader is not None:
            set_weight_attrs(weight_packed, {"weight_loader": weight_loader})
        layer.register_parameter(WEIGHT_PACKED_PARAM, weight_packed)

        weight_scale = nn.Parameter(
            torch.empty(
                output_size_per_partition,
                num_groups_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            weight_scale,
            {
                "input_dim": WEIGHT_SCALE_INPUT_DIM,
                "output_dim": WEIGHT_SCALE_OUTPUT_DIM,
            },
        )
        if weight_loader is not None:
            set_weight_attrs(weight_scale, {"weight_loader": weight_loader})
        layer.register_parameter(WEIGHT_SCALE_PARAM, weight_scale)

        if self.has_zero_point:
            weight_zero_point = nn.Parameter(
                torch.zeros(
                    output_size_per_partition // self.pack_factor,
                    num_groups_per_partition,
                    dtype=ZERO_POINT_DTYPE,
                ),
                requires_grad=False,
            )
            set_weight_attrs(
                weight_zero_point,
                {
                    "input_dim": WEIGHT_SCALE_INPUT_DIM,
                    "output_dim": WEIGHT_PACKED_OUTPUT_DIM,
                    "packed_dim": WEIGHT_PACKED_OUTPUT_DIM,
                    "pack_factor": self.pack_factor,
                },
            )
            if weight_loader is not None:
                set_weight_attrs(weight_zero_point, {"weight_loader": weight_loader})
            layer.register_parameter(WEIGHT_ZERO_POINT_PARAM, weight_zero_point)

        weight_shape = nn.Parameter(
            torch.empty(WEIGHT_SHAPE_NDIM, dtype=WEIGHT_SHAPE_DTYPE),
            requires_grad=False,
        )
        set_weight_attrs(weight_shape, {"weight_loader": _weight_shape_loader})
        layer.register_parameter(WEIGHT_SHAPE_PARAM, weight_shape)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        marlin.prepare_gptq_layer_for_marlin(
            layer,
            num_bits=self.num_bits,
            group_size=self.group_size,
            has_zero_point=self.has_zero_point,
            input_size_per_partition=layer.input_size_per_partition,
            output_size_per_partition=layer.output_size_per_partition,
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return marlin.apply_gptq_marlin_linear(
            input_tensor=x,
            weight=getattr(layer, marlin.WEIGHT_PARAM),
            weight_scale=getattr(layer, marlin.SCALE_PARAM),
            weight_zero_point=getattr(layer, marlin.ZERO_POINT_PARAM),
            workspace=getattr(layer, marlin.WORKSPACE_ATTR),
            g_idx=getattr(layer, marlin.G_IDX_ATTR),
            g_idx_sort_indices=getattr(layer, marlin.G_IDX_SORT_INDICES_ATTR),
            wtype=getattr(layer, marlin.WTYPE_ATTR),
            output_size_per_partition=layer.output_size_per_partition,
            input_size_per_partition=layer.input_size_per_partition,
            bias=bias,
        )

def _weight_shape_loader(param: nn.Parameter, loaded_weight: torch.Tensor, *args: Any) -> None:

    param.data.copy_(loaded_weight.to(param.dtype))

__all__ = ["CompressedTensorsWNA16Method"]
