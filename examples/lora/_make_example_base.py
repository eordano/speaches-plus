#!/usr/bin/env python3
"""Write a TINY synthetic Gemma4Moe base (config.json + model.safetensors) so the
LoRA quickstart runs end-to-end in ~2 minutes on ANY box -- no torch, no GPU, no
download (pure Python stdlib). This is a throwaway ~0.4M-param toy model: it makes
the pipeline (data -> nvk-train -> servable adapter -> load/route check) demoable.

For a REAL adapter, pass `nvk-train --base` a genuine Gemma checkpoint instead:
a `.gguf` file, or a dir with config.json + model.safetensors. See
docs/book/06.6-lora-training.md. The tensor names/shapes here mirror the T3 end-to-end gate
(rust/crates/nv-models/tests/nvk_train_cli.rs) exactly, so the loader accepts it.

Usage:  python3 examples/lora/_make_example_base.py <out-dir>
"""
import json
import os
import random
import struct
import sys
from array import array

HIDDEN = 64
INTER = 96
N_LAYERS = 3
N_Q = 4
N_KV = 2
N_GLOBAL_KV = 1
HEAD_DIM = 16
GLOBAL_HEAD_DIM = 32
VOCAB = 160
N_EXPERTS = 8
TOP_K = 2
MOE_INTER = 24

CONFIG = {
    "model_type": "gemma4",
    "tie_word_embeddings": True,
    "text_config": {
        "attention_k_eq_v": True,
        "enable_moe_block": True,
        "final_logit_softcapping": 30.0,
        "global_head_dim": GLOBAL_HEAD_DIM,
        "head_dim": HEAD_DIM,
        "hidden_activation": "gelu_pytorch_tanh",
        "hidden_size": HIDDEN,
        "intermediate_size": INTER,
        "layer_types": ["sliding_attention", "sliding_attention", "full_attention"],
        "max_position_embeddings": 64,
        "moe_intermediate_size": MOE_INTER,
        "num_attention_heads": N_Q,
        "num_experts": N_EXPERTS,
        "num_global_key_value_heads": N_GLOBAL_KV,
        "num_hidden_layers": N_LAYERS,
        "num_key_value_heads": N_KV,
        "rms_norm_eps": 1e-06,
        "rope_parameters": {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0},
            "sliding_attention": {"rope_theta": 10000.0},
        },
        "sliding_window": 8,
        "tie_word_embeddings": True,
        "top_k_experts": TOP_K,
        "vocab_size": VOCAB,
    },
}

RNG = random.Random(0x9E3779B9)

def rand(shape, scale):
    n = 1
    for d in shape:
        n *= d
    return array("f", [RNG.uniform(-scale, scale) for _ in range(n)])

def ones(n):
    return array("f", [1.0] * n)

def build_tensors():
    """(name, shape, float-array) in the exact set the model expects."""
    t = []
    t.append(("model.language_model.embed_tokens.weight", [VOCAB, HIDDEN], rand([VOCAB, HIDDEN], 1.0)))
    t.append(("model.language_model.norm.weight", [HIDDEN], ones(HIDDEN)))
    for i in range(N_LAYERS):
        p = f"model.language_model.layers.{i}"
        full = i == N_LAYERS - 1
        hd, n_kv = (GLOBAL_HEAD_DIM, N_GLOBAL_KV) if full else (HEAD_DIM, N_KV)
        for norm in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
            "post_feedforward_layernorm_1",
            "pre_feedforward_layernorm_2",
            "post_feedforward_layernorm_2",
        ]:
            t.append((f"{p}.{norm}.weight", [HIDDEN], ones(HIDDEN)))
        t.append((f"{p}.layer_scalar", [1], ones(1)))
        t.append((f"{p}.self_attn.q_proj.weight", [N_Q * hd, HIDDEN], rand([N_Q * hd, HIDDEN], 0.3)))
        t.append((f"{p}.self_attn.k_proj.weight", [n_kv * hd, HIDDEN], rand([n_kv * hd, HIDDEN], 0.3)))
        if not full:
            t.append((f"{p}.self_attn.v_proj.weight", [n_kv * hd, HIDDEN], rand([n_kv * hd, HIDDEN], 0.3)))
        t.append((f"{p}.self_attn.o_proj.weight", [HIDDEN, N_Q * hd], rand([HIDDEN, N_Q * hd], 0.3)))
        t.append((f"{p}.self_attn.q_norm.weight", [hd], ones(hd)))
        t.append((f"{p}.self_attn.k_norm.weight", [hd], ones(hd)))
        t.append((f"{p}.mlp.gate_proj.weight", [INTER, HIDDEN], rand([INTER, HIDDEN], 0.3)))
        t.append((f"{p}.mlp.up_proj.weight", [INTER, HIDDEN], rand([INTER, HIDDEN], 0.3)))
        t.append((f"{p}.mlp.down_proj.weight", [HIDDEN, INTER], rand([HIDDEN, INTER], 0.3)))
        t.append((f"{p}.router.proj.weight", [N_EXPERTS, HIDDEN], rand([N_EXPERTS, HIDDEN], 0.3)))
        t.append((f"{p}.router.scale", [HIDDEN], ones(HIDDEN)))
        t.append((f"{p}.router.per_expert_scale", [N_EXPERTS], ones(N_EXPERTS)))
        t.append((f"{p}.experts.gate_up_proj", [N_EXPERTS, 2 * MOE_INTER, HIDDEN], rand([N_EXPERTS, 2 * MOE_INTER, HIDDEN], 0.3)))
        t.append((f"{p}.experts.down_proj", [N_EXPERTS, HIDDEN, MOE_INTER], rand([N_EXPERTS, HIDDEN, MOE_INTER], 0.3)))
    return t

def write_safetensors(path, tensors):
    """Minimal safetensors writer: 8-byte LE header length, JSON header, F32 data."""
    header = {}
    data = bytearray()
    for name, shape, arr in tensors:
        raw = arr.tobytes()
        start = len(data)
        data += raw
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [start, len(data)]}
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (-(len(hjson)) % 8)
    hjson += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(data)

def main():
    if len(sys.argv) != 2:
        print("usage: python3 examples/lora/_make_example_base.py <out-dir>", file=sys.stderr)
        sys.exit(2)
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(CONFIG, f, indent=2)
    tensors = build_tensors()
    write_safetensors(os.path.join(out, "model.safetensors"), tensors)
    n_params = sum(len(a) for _, _, a in tensors)
    print(f"wrote example base -> {out}  ({len(tensors)} tensors, {n_params} params, F32)")

if __name__ == "__main__":
    main()
