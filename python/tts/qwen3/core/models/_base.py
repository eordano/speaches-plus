
from __future__ import annotations

from torch import nn
from transformers.modeling_utils import (PreTrainedModel)
from transformers.utils import logging

from .configuration_qwen3_tts import (Qwen3TTSConfig)
from ._norms_rope import Qwen3TTSRMSNorm

logger = logging.get_logger(__name__)

DEFAULT_INIT_STD = 0.02

class Qwen3TTSPreTrainedModel(PreTrainedModel):
    config_class = Qwen3TTSConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3TTSDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = False
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = getattr(self.config, "initializer_range", DEFAULT_INIT_STD)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

class Qwen3TTSTalkerTextPreTrainedModel(PreTrainedModel):
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = False
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, Qwen3TTSRMSNorm):
            module.weight.data.fill_(1.0)
