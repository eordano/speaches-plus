from __future__ import annotations

import os
from collections.abc import Callable
from glob import glob

import torch
from safetensors import safe_open
from torch import nn

SAFETENSORS_GLOB = "*.safetensors"
SAFETENSORS_FRAMEWORK = "pt"
LOAD_DEVICE = "cpu"
POST_LOAD_HOOK_NAME = "process_weights_after_loading"
QUANT_METHOD_ATTR = "quant_method"
WEIGHT_REMAP_ATTR = "weight_name_remap_rules"

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    param.data.copy_(loaded_weight)

def _resolve_packed(
    weight_name: str,
    packed_modules_mapping: dict,
) -> tuple[str, str | int] | None:
    for source_key, (target_key, shard_id) in packed_modules_mapping.items():
        if source_key in weight_name:
            return weight_name.replace(source_key, target_key), shard_id
    return None

def _apply_remap_rules(
    weight_name: str,
    remap_rules: list[Callable[[str], str | None]],
) -> str | None:
    for rule in remap_rules:
        remapped = rule(weight_name)
        if remapped is not None:
            return remapped
    return weight_name

def _dispatch_weight(
    model: nn.Module,
    weight_name: str,
    tensor: torch.Tensor,
    packed_modules_mapping: dict,
) -> None:
    packed = _resolve_packed(weight_name, packed_modules_mapping)
    if packed is not None:
        param_name, shard_id = packed
        param = model.get_parameter(param_name)
        param.weight_loader(param, tensor, shard_id)
        return
    param = model.get_parameter(weight_name)
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, tensor)

def _run_post_load_hooks(model: nn.Module) -> None:
    for module in model.modules():
        quant_method = getattr(module, QUANT_METHOD_ATTR, None)
        if quant_method is not None:
            hook = getattr(quant_method, POST_LOAD_HOOK_NAME, None)
            if callable(hook):
                hook(module)
        direct_hook = getattr(module, POST_LOAD_HOOK_NAME, None)
        if callable(direct_hook):
            direct_hook()

def load_model(model: nn.Module, path: str) -> None:
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    remap_rules = getattr(model, WEIGHT_REMAP_ATTR, [])
    for file in glob(os.path.join(path, SAFETENSORS_GLOB)):
        with safe_open(file, SAFETENSORS_FRAMEWORK, LOAD_DEVICE) as handle:
            for weight_name in handle.keys():
                effective_name = _apply_remap_rules(weight_name, remap_rules)
                if effective_name is None:
                    continue
                _dispatch_weight(model, effective_name, handle.get_tensor(weight_name), packed_modules_mapping)
    _run_post_load_hooks(model)
