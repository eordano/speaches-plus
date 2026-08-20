
from __future__ import annotations

import torch
from torch import nn
from transformers.integrations import use_kernel_forward_from_hub
from transformers.modeling_rope_utils import (ROPE_INIT_FUNCTIONS,
                                              dynamic_rope_update)
from transformers.utils import logging

from .configuration_qwen3_tts import (Qwen3TTSConfig,
                                       Qwen3TTSTalkerConfig)

logger = logging.get_logger(__name__)

def _nano_default_rope_init(config, device=None, seq_len=None, layer_type=None):
    import torch as _torch
    base = getattr(config, 'rope_theta', 10000.0)
    partial = getattr(config, 'partial_rotary_factor', 1.0)
    head_dim = getattr(config, 'head_dim', None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial)
    inv = 1.0 / (base ** (_torch.arange(0, dim, 2, dtype=_torch.int64).to(device=device, dtype=_torch.float) / dim))
    return inv, 1.0

if 'default' not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS['default'] = _nano_default_rope_init
class Qwen3TTSTalkerRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3TTSTalkerConfig, device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, hidden_states, position_ids):

        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float()

        device_type = hidden_states.device.type if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)

class Qwen3TTSRotaryEmbedding(nn.Module):
    def __init__(self, config: Qwen3TTSConfig, device=None):
        super().__init__()

        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, hidden_states, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(hidden_states.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = hidden_states.device.type if isinstance(hidden_states.device.type, str) and hidden_states.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=hidden_states.dtype), sin.to(dtype=hidden_states.dtype)

@use_kernel_forward_from_hub("RMSNorm")
class Qwen3TTSRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

def rotate_half(x):
    first_half = x[..., : x.shape[-1] // 2]
    second_half = x[..., x.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)

def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1):

    if mrope_interleaved:

        def apply_interleaved_rope(freqs, modality_num):
            interleaved = freqs[0].clone()
            index_ranges = []
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * modality_num
                index_ranges.append((beg_idx, end_idx))
            for beg_idx, end_idx in index_ranges:
                interleaved[..., beg_idx:end_idx:modality_num] = freqs[beg_idx, ..., beg_idx:end_idx:modality_num]
            return interleaved

        dim = cos.shape[-1]
        modality_num = len(mrope_section)
        cos = torch.cat([apply_interleaved_rope(cos[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([apply_interleaved_rope(sin[..., : dim // 2], modality_num)] * 2, dim=-1).unsqueeze(
            unsqueeze_dim
        )
    else:
        mrope_section = mrope_section * 2
        cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )
        sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(
            unsqueeze_dim
        )

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):

    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
