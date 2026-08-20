from __future__ import annotations

from nano_vllm.models.gemma4_moe import Gemma4ForCausalLM
from nano_vllm.models.qwen3 import Qwen3ForCausalLM

MODEL_REGISTRY: dict[str, type] = {
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
    "Gemma4ForCausalLM": Gemma4ForCausalLM,
}

def resolve_model_class(hf_config) -> type:
    architectures = getattr(hf_config, "architectures", None)
    if not architectures:
        raise ValueError(
            "hf_config has no 'architectures' field; cannot resolve model class. "
            f"Available registry keys: {sorted(MODEL_REGISTRY)}"
        )
    arch = architectures[0]
    cls = MODEL_REGISTRY.get(arch)
    if cls is None:
        raise ValueError(
            f"Architecture {arch!r} is not registered in nano_vllm.models.registry. "
            f"Available keys: {sorted(MODEL_REGISTRY)}"
        )
    return cls
