import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from nano_vllm.layers.attention import Attention
from nano_vllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nano_vllm.layers.layernorm import RMSNorm
from nano_vllm.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from nano_vllm.layers.quantization import QuantizationConfig
from nano_vllm.layers.rotary_embedding import RotaryEmbedding, get_rope

GEMMA4_DEFAULT_RMS_NORM_EPS = 1e-06
GEMMA4_DEFAULT_ROPE_THETA_SLIDING = 10000.0
GEMMA4_DEFAULT_ROPE_THETA_FULL = 1000000.0
GEMMA4_FULL_ATTENTION_PARTIAL_ROTARY_FACTOR = 0.25
GEMMA4_DEFAULT_TOP_K_EXPERTS = 8
GEMMA4_DEFAULT_NUM_EXPERTS = 128
GEMMA4_RENORM_EPSILON = 0.0
GEMMA4_LAYER_TYPE_FULL = "full_attention"
GEMMA4_LAYER_TYPE_SLIDING = "sliding_attention"
GEMMA4_TORCH_COMPILE_MODE = "default"
GEMMA4_TORCH_COMPILE_DYNAMIC = True
GEMMA4_TORCH_COMPILE_FULLGRAPH = False
GEMMA4_PREFIX_SEPARATOR = "."

def _join_prefix(parent: str, child: str) -> str:
    if not parent:
        return child
    return f"{parent}{GEMMA4_PREFIX_SEPARATOR}{child}"

def _get_text_config(config):
    if hasattr(config, "text_config"):
        return config.text_config
    return config

def _resolve_rope_for_layer(text_config, layer_type: str) -> tuple[float, float]:
    rope_parameters = getattr(text_config, "rope_parameters", None)
    if rope_parameters is None:
        if layer_type == GEMMA4_LAYER_TYPE_FULL:
            return GEMMA4_DEFAULT_ROPE_THETA_FULL, 1.0
        return GEMMA4_DEFAULT_ROPE_THETA_SLIDING, 1.0
    layer_rope = rope_parameters.get(layer_type, rope_parameters)
    rope_theta = float(layer_rope.get("rope_theta", GEMMA4_DEFAULT_ROPE_THETA_SLIDING))
    partial_rotary_factor = float(
        layer_rope.get(
            "partial_rotary_factor",
            GEMMA4_FULL_ATTENTION_PARTIAL_ROTARY_FACTOR
            if layer_type == GEMMA4_LAYER_TYPE_FULL
            else 1.0,
        )
    )
    return rope_theta, partial_rotary_factor

class PartialRotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.rotary_emb = RotaryEmbedding(
            rotary_dim, rotary_dim, max_position_embeddings, rope_theta
        )

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.rotary_dim == self.head_dim:
            return self.rotary_emb(positions, query, key)
        query_rotated, query_passthrough = query.split(
            [self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1
        )
        key_rotated, key_passthrough = key.split(
            [self.rotary_dim, self.head_dim - self.rotary_dim], dim=-1
        )
        query_rotated, key_rotated = self.rotary_emb(
            positions, query_rotated, key_rotated
        )
        query = torch.cat([query_rotated, query_passthrough], dim=-1)
        key = torch.cat([key_rotated, key_passthrough], dim=-1)
        return query, key

def _build_rope(
    head_dim: int,
    max_position: int,
    rope_theta: float,
    partial_rotary_factor: float,
):
    if partial_rotary_factor >= 1.0:
        return get_rope(head_dim, head_dim, max_position, rope_theta)
    rotary_dim = int(head_dim * partial_rotary_factor)
    rotary_dim -= rotary_dim % 2
    return PartialRotaryEmbedding(head_dim, rotary_dim, max_position, rope_theta)

class Gemma4MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "gate_up_proj"),
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "down_proj"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)

class Gemma4Expert(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [moe_intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "gate_up_proj"),
        )
        self.down_proj = RowParallelLinear(
            moe_intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "down_proj"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(hidden_states)
        gate, up = gate_up.chunk(2, dim=-1)
        return self.down_proj(F.gelu(gate, approximate="tanh") * up)

class Gemma4Router(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        rms_norm_eps: float,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)
        self.scale = nn.Parameter(torch.ones(hidden_size))
        self.register_buffer(
            "root_size",
            torch.tensor(hidden_size**-0.5),
            persistent=False,
        )
        self.proj = ReplicatedLinear(
            hidden_size,
            num_experts,
            bias=False,
            prefix=_join_prefix(prefix, "proj"),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(hidden_states)
        scaled = normalized * self.root_size.to(normalized.dtype)
        scaled = scaled * self.scale.to(scaled.dtype)
        return self.proj(scaled.to(self.proj.weight.dtype)).to(torch.float32)

class Gemma4SparseMoEBlock(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        moe_intermediate_size: int,
        num_experts: int,
        top_k_experts: int,
        rms_norm_eps: float,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k_experts = top_k_experts
        self.router = Gemma4Router(
            hidden_size,
            num_experts,
            rms_norm_eps,
            prefix=_join_prefix(prefix, "router"),
        )
        self.per_expert_scale = nn.Parameter(torch.ones(num_experts))
        self.experts = nn.ModuleList(
            [
                Gemma4Expert(
                    hidden_size,
                    moe_intermediate_size,
                    quant_config=quant_config,
                    prefix=_join_prefix(prefix, f"experts.{expert_index}"),
                )
                for expert_index in range(num_experts)
            ]
        )

    def apply_router(
        self,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        topk_logits, topk_ids = torch.topk(
            router_logits, k=self.top_k_experts, dim=-1
        )
        probabilities = F.softmax(router_logits, dim=-1)
        gather_weights = probabilities.gather(-1, topk_ids)
        normalizer = gather_weights.sum(dim=-1, keepdim=True)
        normalizer = torch.where(
            normalizer > GEMMA4_RENORM_EPSILON,
            normalizer,
            torch.ones_like(normalizer),
        )
        topk_weights = gather_weights / normalizer
        expert_scales = self.per_expert_scale[topk_ids].to(topk_weights.dtype)
        return topk_weights * expert_scales, topk_ids

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        flat_hidden = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.router(flat_hidden)
        topk_weights, topk_ids = self.apply_router(router_logits)
        output = torch.zeros_like(flat_hidden)
        for expert_index in range(self.num_experts):
            token_mask = (topk_ids == expert_index).any(dim=-1)
            if not torch.any(token_mask):
                continue
            expert_input = flat_hidden[token_mask]
            expert_output = self.experts[expert_index](expert_input)
            slot_mask = topk_ids[token_mask] == expert_index
            slot_weight = (topk_weights[token_mask] * slot_mask.to(topk_weights.dtype)).sum(dim=-1, keepdim=True)
            output[token_mask] += expert_output * slot_weight.to(expert_output.dtype)
        return output.reshape(original_shape)

class Gemma4Attention(nn.Module):

    def __init__(
        self,
        text_config,
        layer_type: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.layer_type = layer_type
        self.is_full_attention = layer_type == GEMMA4_LAYER_TYPE_FULL
        if self.is_full_attention:
            head_dim = getattr(text_config, "global_head_dim", text_config.head_dim)
            num_kv_heads = getattr(
                text_config,
                "num_global_key_value_heads",
                text_config.num_key_value_heads,
            )
        else:
            head_dim = text_config.head_dim
            num_kv_heads = text_config.num_key_value_heads
        hidden_size = text_config.hidden_size
        num_heads = text_config.num_attention_heads
        rms_norm_eps = getattr(text_config, "rms_norm_eps", GEMMA4_DEFAULT_RMS_NORM_EPS)

        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = 1.0
        self.sliding_window = (
            getattr(text_config, "sliding_window", None)
            if not self.is_full_attention
            else None
        )

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(text_config, "attention_bias", False),
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "qkv_proj"),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=getattr(text_config, "attention_bias", False),
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "o_proj"),
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        rope_theta, partial_rotary_factor = _resolve_rope_for_layer(
            text_config, layer_type
        )
        max_position = text_config.max_position_embeddings
        self.rotary_emb = _build_rope(
            self.head_dim, max_position, rope_theta, partial_rotary_factor
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        v = self.v_norm(v)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output.flatten(1, -1))

class Gemma4DecoderLayer(nn.Module):

    def __init__(
        self,
        text_config,
        layer_index: int,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_types = getattr(text_config, "layer_types", None)
        layer_type = (
            layer_types[layer_index]
            if layer_types is not None
            else GEMMA4_LAYER_TYPE_SLIDING
        )
        self.layer_index = layer_index
        self.layer_type = layer_type
        rms_norm_eps = getattr(text_config, "rms_norm_eps", GEMMA4_DEFAULT_RMS_NORM_EPS)

        self.self_attn = Gemma4Attention(
            text_config,
            layer_type,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "self_attn"),
        )
        self.mlp = Gemma4MLP(
            hidden_size=text_config.hidden_size,
            intermediate_size=text_config.intermediate_size,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "mlp"),
        )
        self.input_layernorm = RMSNorm(text_config.hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(text_config.hidden_size, eps=rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(text_config.hidden_size, eps=rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(text_config.hidden_size, eps=rms_norm_eps)

        self.enable_moe_block = getattr(text_config, "enable_moe_block", False)
        if self.enable_moe_block:
            num_experts = getattr(text_config, "num_experts", GEMMA4_DEFAULT_NUM_EXPERTS)
            top_k_experts = getattr(
                text_config, "top_k_experts", GEMMA4_DEFAULT_TOP_K_EXPERTS
            )
            moe_intermediate_size = getattr(
                text_config, "moe_intermediate_size", text_config.intermediate_size
            )
            self.moe = Gemma4SparseMoEBlock(
                hidden_size=text_config.hidden_size,
                moe_intermediate_size=moe_intermediate_size,
                num_experts=num_experts,
                top_k_experts=top_k_experts,
                rms_norm_eps=rms_norm_eps,
                quant_config=quant_config,
                prefix=_join_prefix(prefix, "moe"),
            )
            self.pre_feedforward_layernorm_2 = RMSNorm(
                text_config.hidden_size, eps=rms_norm_eps
            )
            self.post_feedforward_layernorm_1 = RMSNorm(
                text_config.hidden_size, eps=rms_norm_eps
            )
            self.post_feedforward_layernorm_2 = RMSNorm(
                text_config.hidden_size, eps=rms_norm_eps
            )
        else:
            self.moe = None

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        mlp_input = self.pre_feedforward_layernorm(hidden_states)
        mlp_output = self.mlp(mlp_input)

        if self.enable_moe_block:
            dense_output = self.post_feedforward_layernorm_1(mlp_output)
            moe_input = self.pre_feedforward_layernorm_2(residual)
            moe_output = self.post_feedforward_layernorm_2(self.moe(moe_input))
            combined = dense_output + moe_output
        else:
            combined = mlp_output

        hidden_states = self.post_feedforward_layernorm(combined) + residual
        return hidden_states

class Gemma4Model(nn.Module):

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "model",
        use_torch_compile: bool = False,
        aux_hidden_layer_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        text_config = _get_text_config(config)
        self.config = text_config
        self.embed_tokens = VocabParallelEmbedding(
            text_config.vocab_size, text_config.hidden_size
        )
        self.layers = nn.ModuleList(
            [
                Gemma4DecoderLayer(
                    text_config,
                    layer_index,
                    quant_config=quant_config,
                    prefix=_join_prefix(prefix, f"layers.{layer_index}"),
                )
                for layer_index in range(text_config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            text_config.hidden_size,
            eps=getattr(text_config, "rms_norm_eps", GEMMA4_DEFAULT_RMS_NORM_EPS),
        )
        self.register_buffer(
            "normalizer",
            torch.tensor(text_config.hidden_size**0.5),
            persistent=False,
        )
        self.aux_hidden_layer_ids = (
            tuple(aux_hidden_layer_ids) if aux_hidden_layer_ids else None
        )
        if self.aux_hidden_layer_ids is not None:
            num_layers = text_config.num_hidden_layers
            for layer_id in self.aux_hidden_layer_ids:
                if not (0 <= layer_id < num_layers):
                    raise ValueError(
                        f"aux_hidden_layer_ids contains out-of-range index {layer_id}; "
                        f"model has {num_layers} layers."
                    )
        if use_torch_compile:
            self.forward = torch.compile(
                self.forward,
                mode=GEMMA4_TORCH_COMPILE_MODE,
                dynamic=GEMMA4_TORCH_COMPILE_DYNAMIC,
                fullgraph=GEMMA4_TORCH_COMPILE_FULLGRAPH,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.embed_tokens(input_ids) * self.normalizer.to(
            self.embed_tokens.weight.dtype
        )
        if self.aux_hidden_layer_ids is None:
            for layer in self.layers:
                hidden_states = layer(positions, hidden_states)
            return self.norm(hidden_states)
        aux_set = set(self.aux_hidden_layer_ids)
        snapshots: dict[int, torch.Tensor] = {}
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer(positions, hidden_states)
            if layer_index in aux_set:
                snapshots[layer_index] = hidden_states
        ordered = [snapshots[i] for i in self.aux_hidden_layer_ids]
        aux_hidden_states = torch.cat(ordered, dim=-1)
        return self.norm(hidden_states), aux_hidden_states

class Gemma4ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_torch_compile: bool = False,
        aux_hidden_layer_ids: list[int] | None = None,
    ) -> None:
        super().__init__()
        text_config = _get_text_config(config)
        self.config = text_config
        self._mirror_text_config_onto_root(config, text_config)
        self.model = Gemma4Model(
            config,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "model"),
            use_torch_compile=use_torch_compile,
            aux_hidden_layer_ids=aux_hidden_layer_ids,
        )
        self.aux_hidden_layer_ids = self.model.aux_hidden_layer_ids
        self.lm_head = ParallelLMHead(text_config.vocab_size, text_config.hidden_size)
        if getattr(text_config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    @staticmethod
    def _mirror_text_config_onto_root(root_config, text_config):
        for attribute_name in (
            "num_hidden_layers",
            "num_key_value_heads",
            "num_attention_heads",
            "hidden_size",
            "head_dim",
        ):
            if hasattr(text_config, attribute_name) and not hasattr(
                root_config, attribute_name
            ):
                setattr(root_config, attribute_name, getattr(text_config, attribute_name))
        if not hasattr(root_config, "dtype"):
            root_config.dtype = getattr(text_config, "dtype", torch.bfloat16)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
