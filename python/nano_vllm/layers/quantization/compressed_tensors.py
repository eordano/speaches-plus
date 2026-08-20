from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from nano_vllm.layers.quantization import (
    LinearMethodBase,
    QuantizationConfig,
)
from nano_vllm.layers.quantization.schemes.wna16 import CompressedTensorsWNA16Method

CONFIG_GROUPS_KEY = "config_groups"
TARGETS_KEY = "targets"
WEIGHTS_KEY = "weights"
NUM_BITS_KEY = "num_bits"
GROUP_SIZE_KEY = "group_size"
SYMMETRIC_KEY = "symmetric"
STRATEGY_KEY = "strategy"
TYPE_KEY = "type"
INPUT_ACTIVATIONS_KEY = "input_activations"
IGNORE_KEY = "ignore"
FORMAT_KEY = "format"
QUANT_METHOD_KEY = "quant_method"

QUANT_METHOD_VALUE = "compressed-tensors"
FORMAT_PACK_QUANTIZED = "pack-quantized"
TYPE_INT = "int"
TYPE_FLOAT = "float"
STRATEGY_GROUP = "group"
STRATEGY_CHANNEL = "channel"
STRATEGY_TENSOR = "tensor"

REGEX_PREFIX = "re:"
WILDCARD_LINEAR_TARGET = "Linear"

GROUP_SIZE_CHANNELWISE = -1
SUPPORTED_NUM_BITS = (4,)
MIN_CAPABILITY_AWQ_MARLIN = 80

@dataclass(frozen=True)
class _SchemeSpec:
    num_bits: int
    group_size: int
    symmetric: bool
    has_zero_point: bool
    strategy: str
    target_pattern: re.Pattern[str] | None
    target_class_name: str | None

class CompressedTensorsConfig(QuantizationConfig):

    def __init__(
        self,
        scheme_specs: list[_SchemeSpec],
        ignore_patterns: list[re.Pattern[str]],
        ignore_exact: list[str],
        quant_format: str,
    ) -> None:
        self.scheme_specs = scheme_specs
        self.ignore_patterns = ignore_patterns
        self.ignore_exact = ignore_exact
        self.quant_format = quant_format

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CompressedTensorsConfig:
        quant_format = config.get(FORMAT_KEY, FORMAT_PACK_QUANTIZED)
        if config.get(QUANT_METHOD_KEY, QUANT_METHOD_VALUE) != QUANT_METHOD_VALUE:
            raise ValueError(
                f"Expected quant_method={QUANT_METHOD_VALUE!r}, got {config.get(QUANT_METHOD_KEY)!r}"
            )

        scheme_specs: list[_SchemeSpec] = []
        groups = config.get(CONFIG_GROUPS_KEY, {}) or {}
        for group_key, group in groups.items():
            weights = group.get(WEIGHTS_KEY) or {}
            num_bits = int(weights.get(NUM_BITS_KEY, 0))
            if num_bits not in SUPPORTED_NUM_BITS:
                raise NotImplementedError(
                    f"compressed-tensors num_bits={num_bits} (group {group_key!r}) "
                    f"not implemented; supported = {SUPPORTED_NUM_BITS}."
                )
            if weights.get(TYPE_KEY, TYPE_INT) != TYPE_INT:
                raise NotImplementedError(
                    f"compressed-tensors weight type={weights.get(TYPE_KEY)!r} not implemented; "
                    "only integer W4A16 is supported."
                )
            if group.get(INPUT_ACTIVATIONS_KEY) is not None:
                raise NotImplementedError(
                    "compressed-tensors with quantized input activations (W4A8/W8A8/NVFP4) "
                    "is not implemented; only W4A16 is supported."
                )
            strategy = weights.get(STRATEGY_KEY, STRATEGY_GROUP)
            symmetric = bool(weights.get(SYMMETRIC_KEY, True))
            has_zero_point = not symmetric
            group_size_raw = weights.get(GROUP_SIZE_KEY)
            if strategy == STRATEGY_GROUP:
                if group_size_raw is None:
                    raise ValueError(
                        f"group strategy requires group_size (group {group_key!r})"
                    )
                group_size = int(group_size_raw)
            elif strategy == STRATEGY_CHANNEL:
                group_size = GROUP_SIZE_CHANNELWISE
            else:
                raise NotImplementedError(
                    f"compressed-tensors strategy={strategy!r} not implemented."
                )

            for target in group.get(TARGETS_KEY, []) or []:
                pattern, class_name = _parse_target(target)
                scheme_specs.append(
                    _SchemeSpec(
                        num_bits=num_bits,
                        group_size=group_size,
                        symmetric=symmetric,
                        has_zero_point=has_zero_point,
                        strategy=strategy,
                        target_pattern=pattern,
                        target_class_name=class_name,
                    )
                )

        ignore_patterns: list[re.Pattern[str]] = []
        ignore_exact: list[str] = []
        for ignore_entry in config.get(IGNORE_KEY, []) or []:
            pattern, class_name = _parse_target(ignore_entry)
            if pattern is not None:
                ignore_patterns.append(pattern)
            elif class_name is not None:
                ignore_exact.append(class_name)
            else:
                ignore_exact.append(ignore_entry)

        return cls(
            scheme_specs=scheme_specs,
            ignore_patterns=ignore_patterns,
            ignore_exact=ignore_exact,
            quant_format=quant_format,
        )

    def get_quant_method(
        self,
        layer: nn.Module,
        prefix: str,
    ) -> LinearMethodBase | None:
        if is_layer_skipped(prefix, self.ignore_exact, self.ignore_patterns, _packed_modules_mapping(layer)):
            return None
        for spec in self.scheme_specs:
            if not _spec_matches(spec, prefix):
                continue
            return CompressedTensorsWNA16Method(
                group_size=spec.group_size,
                num_bits=spec.num_bits,
                symmetric=spec.symmetric,
                has_zero_point=spec.has_zero_point,
            )
        return None

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.bfloat16, torch.float16]

    def get_min_capability(self) -> int:
        return MIN_CAPABILITY_AWQ_MARLIN

def _parse_target(target: str) -> tuple[re.Pattern[str] | None, str | None]:
    if target.startswith(REGEX_PREFIX):
        return re.compile(target[len(REGEX_PREFIX):]), None
    if "." not in target and target != WILDCARD_LINEAR_TARGET:
        return None, target
    if target == WILDCARD_LINEAR_TARGET:
        return None, WILDCARD_LINEAR_TARGET
    return re.compile(re.escape(target)), None

def _spec_matches(spec: _SchemeSpec, prefix: str) -> bool:
    if spec.target_class_name == WILDCARD_LINEAR_TARGET:
        return True
    if spec.target_pattern is not None:
        return spec.target_pattern.search(prefix) is not None
    return False

def is_layer_skipped(
    prefix: str,
    ignore_exact: list[str],
    ignore_patterns: list[re.Pattern[str]],
    packed_modules_mapping: Mapping[str, list[str]],
) -> bool:

    proj_name = prefix.rsplit(".", 1)[-1] if "." in prefix else prefix
    if proj_name in packed_modules_mapping:
        shard_prefixes = [
            prefix.rsplit(".", 1)[0] + "." + shard_name if "." in prefix else shard_name
            for shard_name in packed_modules_mapping[proj_name]
        ]
        decisions = [_matches_ignore(shard_prefix, ignore_exact, ignore_patterns) for shard_prefix in shard_prefixes]
        if all(decisions):
            return True
        if any(decisions):
            raise ValueError(
                f"Mixed quantization for fused layer {prefix!r}: some shards in ignore list, "
                f"others not. Cannot mix quant for a fused module. Shards: {shard_prefixes}"
            )
        return False
    return _matches_ignore(prefix, ignore_exact, ignore_patterns)

def _matches_ignore(
    prefix: str,
    ignore_exact: list[str],
    ignore_patterns: list[re.Pattern[str]],
) -> bool:
    if prefix in ignore_exact:
        return True
    for pattern in ignore_patterns:
        if pattern.search(prefix) is not None:
            return True
    return False

def _packed_modules_mapping(layer: nn.Module | None) -> Mapping[str, list[str]]:

    if layer is None:
        return _DEFAULT_PACKED_MAPPING
    mapping = getattr(layer, "_compressed_tensors_packed_mapping", None)
    if mapping is not None:
        return mapping
    return _DEFAULT_PACKED_MAPPING

_DEFAULT_PACKED_MAPPING: Mapping[str, list[str]] = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}

__all__ = [
    "CompressedTensorsConfig",
    "is_layer_skipped",
]
