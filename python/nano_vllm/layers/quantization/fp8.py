from __future__ import annotations

from typing import Any

import torch
from torch import nn

from nano_vllm.layers.quantization import (
    INPUT_DIM,
    PACKED_OUTPUT_DIM,
    LinearMethodBase,
    QuantizationConfig,
    set_weight_attrs,
)

FP8_E4M3_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn
OUT_DTYPE = torch.bfloat16
INPUT_AMAX_DIM = -1
INPUT_AMAX_KEEPDIM = True
WEIGHT_PARAM_NAME = "weight"
WEIGHT_SCALE_ATTR = "weight_scale"
SCALE_TENSOR_SHAPE: tuple[int, ...] = ()
MIN_FP8_CAPABILITY = 89
QUANT_METHOD_KEY = "quant_method"
QUANT_METHOD_FP8 = "fp8"
ACTIVATION_SCHEME_KEY = "activation_scheme"
ACTIVATION_SCHEME_DYNAMIC = "dynamic"
SKIP_PREFIX_SUBSTRINGS = ("lm_head", "embed_tokens")
SKIP_SUFFIX_SUBSTRINGS = (".router.proj", ".gate")
INPUT_SCALE_EPSILON = 1e-12

class Fp8Config(QuantizationConfig):

    def __init__(self, activation_scheme: str = ACTIVATION_SCHEME_DYNAMIC) -> None:
        if activation_scheme != ACTIVATION_SCHEME_DYNAMIC:
            raise ValueError(
                f"nano_vllm fp8 supports only dynamic activations, got {activation_scheme!r}"
            )
        self.activation_scheme = activation_scheme

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Fp8Config:
        if config:
            quant_method = config.get(QUANT_METHOD_KEY)
            if quant_method == QUANT_METHOD_FP8 and config.get(ACTIVATION_SCHEME_KEY) != ACTIVATION_SCHEME_DYNAMIC:
                raise ValueError(
                    "Pre-quantized FP8 checkpoints are not supported on this engine "
                    "(produces garbage on SM120). Pass --quantization fp8 with a BF16 checkpoint instead."
                )
        return cls(activation_scheme=ACTIVATION_SCHEME_DYNAMIC)

    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> LinearMethodBase | None:
        if _should_skip(prefix):
            return None
        return Fp8OnlineLinearMethod()

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    def get_min_capability(self) -> int:
        return MIN_FP8_CAPABILITY

def _should_skip(prefix: str) -> bool:
    for substring in SKIP_PREFIX_SUBSTRINGS:
        if substring in prefix:
            return True
    for substring in SKIP_SUFFIX_SUBSTRINGS:
        if prefix.endswith(substring):
            return True
    return False

class Fp8OnlineLinearMethod(LinearMethodBase):

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

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        original_weight = layer.weight.data
        amax = original_weight.abs().amax().to(torch.float32)
        weight_scale = (amax / FP8_E4M3_MAX).clamp(min=INPUT_SCALE_EPSILON)
        scaled = original_weight.to(torch.float32) / weight_scale
        quantized = scaled.to(FP8_DTYPE)
        del layer.weight
        layer.register_parameter(
            WEIGHT_PARAM_NAME,
            nn.Parameter(quantized, requires_grad=False),
        )
        layer.register_buffer(
            WEIGHT_SCALE_ATTR,
            weight_scale.reshape(SCALE_TENSOR_SHAPE),
            persistent=False,
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        x_amax = flat_x.abs().amax(dim=INPUT_AMAX_DIM, keepdim=INPUT_AMAX_KEEPDIM).to(torch.float32)
        input_scale = (x_amax / FP8_E4M3_MAX).clamp(min=INPUT_SCALE_EPSILON)
        x_quantized = (flat_x.to(torch.float32) / input_scale).to(FP8_DTYPE)
        weight_scale = layer.weight_scale
        output = torch._scaled_mm(
            x_quantized,
            layer.weight.t(),
            scale_a=input_scale,
            scale_b=weight_scale,
            bias=bias,
            out_dtype=OUT_DTYPE,
        )
        return output.reshape(*original_shape[:-1], output.shape[-1])

__all__ = ["Fp8Config", "Fp8OnlineLinearMethod"]
