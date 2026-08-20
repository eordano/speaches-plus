from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

WEIGHT_PARAM_NAME = "weight"
DEFAULT_PREFIX = ""
PACKED_OUTPUT_DIM = 0
INPUT_DIM = 1
MIN_CAPABILITY_UNQUANTIZED = 0

class LinearMethodBase(ABC):

    @abstractmethod
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
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        return None

class UnquantizedLinearMethod(LinearMethodBase):

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
        weight = nn.Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(weight, {"input_dim": INPUT_DIM, "output_dim": PACKED_OUTPUT_DIM})
        set_weight_attrs(weight, extra_weight_attrs)
        layer.register_parameter(WEIGHT_PARAM_NAME, weight)

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)

class QuantizationConfig(ABC):

    @classmethod
    @abstractmethod
    def from_config(cls, config: dict[str, Any]) -> QuantizationConfig:
        raise NotImplementedError

    @abstractmethod
    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> LinearMethodBase | None:
        raise NotImplementedError

    @abstractmethod
    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        raise NotImplementedError

    @abstractmethod
    def get_min_capability(self) -> int:
        raise NotImplementedError

def set_weight_attrs(weight: nn.Parameter, attrs: dict[str, Any] | None) -> None:

    if not attrs:
        return
    for attr_name, attr_value in attrs.items():
        assert not hasattr(weight, attr_name), f"{attr_name} already set on parameter"
        setattr(weight, attr_name, attr_value)

__all__ = [
    "LinearMethodBase",
    "QuantizationConfig",
    "UnquantizedLinearMethod",
    "set_weight_attrs",
]
