from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

@dataclass
class Eagle3Config:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    target_vocab_size: int
    draft_vocab_size: int
    target_hidden_size: int | None = None
    eagle_aux_hidden_state_layer_ids: list[int] = field(default_factory=list)
    norm_before_residual: bool = False
    norm_before_fc: bool = False
    tie_word_embeddings: bool = False

    @classmethod
    def from_hf_dir(cls, path: str | os.PathLike) -> Eagle3Config:
        with open(Path(path) / "config.json") as f:
            raw = json.load(f)
        tl = dict(raw.get("transformer_layer_config", {}))
        rope_params = tl.get("rope_parameters") or {}
        rope_theta = rope_params.get("rope_theta", tl.get("rope_theta", 10000.0))
        head_dim = tl.get("head_dim") or (tl["hidden_size"] // tl["num_attention_heads"])
        merged = {
            "hidden_size": tl["hidden_size"],
            "intermediate_size": tl["intermediate_size"],
            "num_attention_heads": tl["num_attention_heads"],
            "num_key_value_heads": tl.get("num_key_value_heads", tl["num_attention_heads"]),
            "head_dim": head_dim,
            "rms_norm_eps": tl.get("rms_norm_eps", 1e-6),
            "rope_theta": float(rope_theta),
            "max_position_embeddings": tl.get("max_position_embeddings", 4096),
            "target_vocab_size": tl["vocab_size"],
            "draft_vocab_size": raw.get("draft_vocab_size", 32000),
            "target_hidden_size": raw.get("target_hidden_size"),
            "eagle_aux_hidden_state_layer_ids": list(
                raw.get("eagle_aux_hidden_state_layer_ids") or []
            ),
            "norm_before_residual": bool(raw.get("norm_before_residual", False)),
            "norm_before_fc": bool(raw.get("norm_before_fc", False)),
            "tie_word_embeddings": bool(
                raw.get("tie_word_embeddings", tl.get("tie_word_embeddings", False))
            ),
        }
        return cls(**merged)

class Eagle3RMSNorm(nn.Module):

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x32 = x.float()
        var = x32.pow(2).mean(-1, keepdim=True)
        x32 = x32 * torch.rsqrt(var + self.eps)
        return (x32.to(orig_dtype)) * self.weight

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos + _rotate_half(q) * sin), (k * cos + _rotate_half(k) * sin)

class Eagle3RotaryEmbedding(nn.Module):

    def __init__(self, head_dim: int, max_position: int, base: float):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position = max_position

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        from typing import cast as _cast
        inv_freq = _cast(torch.Tensor, self.inv_freq).to(positions.device)
        freqs = positions[..., None].float() * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class Eagle3MLP(nn.Module):

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Eagle3Attention(nn.Module):

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        attn_in = 2 * config.hidden_size
        self.q_proj = nn.Linear(attn_in, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(attn_in, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(attn_in, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        if self.num_kv_heads != self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.scale)
        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.num_heads * self.head_dim)
        return self.o_proj(out)

class Eagle3DecoderLayer(nn.Module):

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.norm_before_residual = config.norm_before_residual
        self.input_layernorm = Eagle3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.hidden_norm = Eagle3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Eagle3Attention(config)
        self.post_attention_layernorm = Eagle3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = Eagle3MLP(config.hidden_size, config.intermediate_size)

    def forward(
        self,
        embeds: torch.Tensor,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden
        embeds_n = self.input_layernorm(embeds)
        hidden_n = self.hidden_norm(hidden)
        if self.norm_before_residual:
            residual = hidden_n
        x = torch.cat([embeds_n, hidden_n], dim=-1)
        x = self.self_attn(x, cos, sin)
        x = residual + x
        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x

class Eagle3DraftModel(nn.Module):

    def __init__(self, config: Eagle3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.target_vocab_size, config.hidden_size)
        self.fc = nn.Linear(3 * config.hidden_size, config.hidden_size, bias=False)
        if config.norm_before_fc:
            self.input_norm = Eagle3RMSNorm(3 * config.hidden_size, config.rms_norm_eps)
        else:
            self.input_norm = None
        self.midlayer = Eagle3DecoderLayer(config)
        self.norm = Eagle3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.draft_vocab_size, bias=False)
        self.rotary_emb = Eagle3RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer(
            "d2t", torch.zeros(config.draft_vocab_size, dtype=torch.long), persistent=True
        )
        self.register_buffer(
            "t2d", torch.zeros(config.target_vocab_size, dtype=torch.bool), persistent=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        fuse_aux: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if fuse_aux:
            if self.input_norm is not None:
                hidden_states = self.input_norm(hidden_states)
            h = self.fc(hidden_states)
        else:
            h = hidden_states
        e = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(positions)
        midlayer_out = self.midlayer(e, h, cos, sin)
        x = self.norm(midlayer_out)
        logits = self.lm_head(x)
        return logits, midlayer_out

    def propose_token_ids(self, logits: torch.Tensor) -> torch.Tensor:
        draft_ids = logits.argmax(dim=-1)
        return draft_ids + self.d2t[draft_ids]

_KEY_RENAMES = {
    "layers.0.": "midlayer.",
}

def _remap_key(name: str) -> str:
    for src, dst in _KEY_RENAMES.items():
        if name.startswith(src):
            return dst + name[len(src):]
    return name

def load_eagle3_from_hf(repo_id_or_path: str) -> Eagle3DraftModel:
    if "/" in repo_id_or_path and not Path(repo_id_or_path).exists():
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id_or_path)
    else:
        path = repo_id_or_path
    cfg = Eagle3Config.from_hf_dir(path)
    model = Eagle3DraftModel(cfg)
    raw = load_file(str(Path(path) / "model.safetensors"))
    remapped: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        remapped[_remap_key(k)] = v
    own = dict(model.state_dict())
    matched: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for k, v in remapped.items():
        if k in own:
            if tuple(own[k].shape) != tuple(v.shape):
                raise RuntimeError(
                    f"shape mismatch for {k}: ckpt {tuple(v.shape)} vs model {tuple(own[k].shape)}"
                )
            matched[k] = v.to(own[k].dtype) if own[k].dtype != v.dtype else v
        else:
            unmapped.append(k)
    missing = [k for k in own if k not in matched]
    if unmapped:
        raise RuntimeError(f"unmapped checkpoint keys: {unmapped}")
    if missing:
        raise RuntimeError(f"missing model keys (no checkpoint): {missing}")
    model.load_state_dict(matched, strict=True)
    return model
