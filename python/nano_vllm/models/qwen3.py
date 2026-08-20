import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config

from nano_vllm.layers.activation import SiluAndMul
from nano_vllm.layers.attention import Attention
from nano_vllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from nano_vllm.layers.layernorm import RMSNorm
from nano_vllm.layers.linear import MergedColumnParallelLinear, QKVParallelLinear, RowParallelLinear
from nano_vllm.layers.quantization import QuantizationConfig
from nano_vllm.layers.rotary_embedding import get_rope

QWEN3_DEFAULT_MAX_POSITION = 4096 * 32
QWEN3_DEFAULT_RMS_NORM_EPS = 1e-06
QWEN3_DEFAULT_ROPE_THETA = 10000
QWEN3_DEFAULT_ROPE_THETA_LARGE = 1000000
QWEN3_DEFAULT_QKV_BIAS = True
QWEN3_HIDDEN_ACT_SILU = "silu"
QWEN3_TORCH_COMPILE_MODE = "default"
QWEN3_TORCH_COMPILE_DYNAMIC = True
QWEN3_TORCH_COMPILE_FULLGRAPH = False
QWEN3_PREFIX_SEPARATOR = "."

def _join_prefix(parent: str, child: str) -> str:
    if not parent:
        return child
    return f"{parent}{QWEN3_PREFIX_SEPARATOR}{child}"

class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = QWEN3_DEFAULT_MAX_POSITION,
        head_dim: int | None = None,
        rms_norm_eps: float = QWEN3_DEFAULT_RMS_NORM_EPS,
        qkv_bias: bool = False,
        rope_theta: float = QWEN3_DEFAULT_ROPE_THETA,
        rope_scaling: dict | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "qkv_proj"),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "o_proj"),
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

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
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        return self.o_proj(attn_output.flatten(1, -1))

class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
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
        assert hidden_act == QWEN3_HIDDEN_ACT_SILU
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states):
        gate_up = self.gate_up_proj(hidden_states)
        activated = self.act_fn(gate_up)
        return self.down_proj(activated)

class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, 'attention_bias', QWEN3_DEFAULT_QKV_BIAS),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", QWEN3_DEFAULT_ROPE_THETA_LARGE),
            rope_scaling=getattr(config, "rope_scaling", None),
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "self_attn"),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "mlp"),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "model",
        use_torch_compile: bool = False,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(
                    config,
                    quant_config=quant_config,
                    prefix=_join_prefix(prefix, f"layers.{layer_index}"),
                )
                for layer_index in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if use_torch_compile:
            self.forward = torch.compile(
                self.forward,
                mode=QWEN3_TORCH_COMPILE_MODE,
                dynamic=QWEN3_TORCH_COMPILE_DYNAMIC,
                fullgraph=QWEN3_TORCH_COMPILE_FULLGRAPH,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

class Qwen3ForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        use_torch_compile: bool = False,
    ) -> None:
        super().__init__()
        self.model = Qwen3Model(
            config,
            quant_config=quant_config,
            prefix=_join_prefix(prefix, "model"),
            use_torch_compile=use_torch_compile,
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
