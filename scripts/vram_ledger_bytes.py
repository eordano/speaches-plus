import glob
import json
import os
import struct

DT = {
    "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "I8": 1, "U8": 1,
    "I32": 4, "I64": 8, "U32": 4, "F64": 8, "BOOL": 1, "F4": 0.5,
    "F6_E3M2": 0.75, "F6_E2M3": 0.75, "U16": 2, "I16": 2,
}

HUB = os.environ.get("HF_HUB_CACHE_FOR_LEDGER", os.path.expanduser("~/.cache/huggingface/hub"))

NVFP4_RESIDENT_BYTES_PER_BF16_BYTE = 9.0 / 32.0

def snap(repo):
    dirs = sorted(glob.glob(os.path.join(HUB, repo, "snapshots", "*")))
    assert dirs, f"no snapshot for {repo} under {HUB}"
    return dirs[0]

def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))

def tensors(snapdir):
    out = {}
    for p in sorted(glob.glob(os.path.join(snapdir, "*.safetensors"))):
        for name, meta in header(p).items():
            if name == "__metadata__":
                continue
            numel = 1
            for d in meta["shape"]:
                numel *= d
            out[name] = (meta["dtype"], meta["shape"], numel * DT[meta["dtype"]])
    return out

def classify(name):
    n = name.lower()
    if "visual" in n or "vision" in n or "audio" in n or "multi_modal" in n or "multimodal" in n or "mm_" in n:
        return "multimodal"
    if "mtp" in n:
        return "mtp"
    if "embed" in n:
        return "embed"
    if "lm_head" in n:
        return "lm_head"
    if any(s in n for s in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "attn", "in_proj", "out_proj", "ba_proj", "a_log", "dt_bias", "conv1d")):
        return "attn_plus_gdn"
    if any(s in n for s in ("gate_up", "gate_proj", "up_proj", "down_proj", "mlp", "ffn", "fc1", "fc2")):
        return "mlp"
    if "norm" in n:
        return "norms"
    return "other"

def cats(ts):
    out = {}
    for name, (_dt, _shape, b) in ts.items():
        out[classify(name)] = out.get(classify(name), 0) + b
    return out

def gib(x):
    return x / (1 << 30)

def kv_full_attn_bytes(layers, kv_heads, head_dim, seq, dtype_bytes):
    return layers * kv_heads * head_dim * seq * dtype_bytes * 2

def gdn_state_bytes(layers, n_v_heads, d_k, d_v, conv_dim, conv_kernel, state_dtype_bytes, conv_dtype_bytes):
    recurrent = layers * n_v_heads * d_k * d_v * state_dtype_bytes
    conv = layers * conv_dim * (conv_kernel - 1) * conv_dtype_bytes
    return recurrent + conv

QWEN38 = {
    "repo": "models--unsloth--Qwen3.8-27B-NVFP4",
    "layers": 64, "full_attn_layers": 16, "gdn_layers": 48,
    "kv_heads": 4, "head_dim": 256,
    "linear_num_value_heads": 48, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
    "linear_num_key_heads": 16, "conv_kernel": 4,
    "hidden": 5120, "vocab": 248320, "max_pos": 262144,
}
QWEN38["gdn_conv_dim"] = (
    2 * QWEN38["linear_num_key_heads"] * QWEN38["linear_key_head_dim"]
    + QWEN38["linear_num_value_heads"] * QWEN38["linear_value_head_dim"]
)

GEMMA431B = {
    "repo": "models--nvidia--Gemma-4-31B-IT-NVFP4",
    "layers": 60, "full_attn_layers": 10, "sliding_layers": 50, "sliding_window": 1024,
    "kv_heads": 16, "head_dim": 256,
    "hidden": 5376, "vocab": 262144, "max_pos": 262144,
}

EAGLE3 = {"repo": "models--RedHatAI--gemma-4-31B-it-speculator.eagle3"}

AUX_SAFETENSORS = {
    "qwen3-embedding-0.6b": "models--Qwen--Qwen3-Embedding-0.6B",
    "qwen3-tts-0.6b": "models--Qwen--Qwen3-TTS-12Hz-0.6B-Base",
    "dots-ocr": "models--rednote-hilab--dots.ocr",
    "deepseek-ocr-2": "models--deepseek-ai--DeepSeek-OCR-2",
}

AUX_FILES_LAZY_OR_CPU = {
    "whisper-ct2-large-v3-turbo": ("models--deepdml--faster-whisper-large-v3-turbo-ct2", "model.bin"),
    "kokoro-82m-onnx": ("models--speaches-ai--Kokoro-82M-v1.0-ONNX", "model.onnx"),
    "privacy-filter-q4-onnx": ("models--openai--privacy-filter", "onnx/model_q4.onnx_data"),
}

def report():
    out = {"basis": {"hub": HUB, "unit": "bytes; GiB = 2^30"}}

    q = tensors(snap(QWEN38["repo"]))
    qc = cats(q)
    out["qwen38_ckpt_categories_gib"] = {k: round(gib(v), 3) for k, v in sorted(qc.items(), key=lambda kv: -kv[1])}
    text_resident = sum(v for k, v in qc.items() if k not in ("multimodal", "mtp"))
    out["qwen38_text_resident_no_drafter_gib"] = round(gib(text_resident), 3)
    mtp_bf16 = sum(b for n, (dt, _s, b) in q.items() if classify(n) == "mtp" and dt == "BF16")
    mtp_rest = qc.get("mtp", 0) - mtp_bf16
    lm_head_bytes = qc.get("lm_head", 0)
    out["qwen38_mtp_drafter_gib"] = {
        "ckpt_mtp_tensors": round(gib(qc.get("mtp", 0)), 3),
        "fp8_resident_arm_NV_Q38_DRAFT_FP8_replaces_bf16_with_fp8": round(gib(mtp_bf16 / 2 + mtp_rest), 3),
        "lmhead_nvfp4_second_resident_copy_NV_Q38_DRAFT_LMHEAD_NVFP4": round(gib(lm_head_bytes * NVFP4_RESIDENT_BYTES_PER_BF16_BYTE * 2), 3),
    }
    for seq in (32768, 131072, 262144):
        out[f"qwen38_kv_full16_at_{seq}_gib"] = {
            "formula": "16 layers x 4 kv_heads x 256 hd x seq x dtype x 2(K+V)",
            "bf16": round(gib(kv_full_attn_bytes(16, 4, 256, seq, 2)), 3),
            "fp8": round(gib(kv_full_attn_bytes(16, 4, 256, seq, 1)), 3),
        }
    out["qwen38_gdn_state_per_lane_gib"] = {
        "formula": "48 layers x (48 vheads x 128 dk x 128 dv x f32 + conv_dim x (k-1) x bf16)",
        "value": round(gib(gdn_state_bytes(48, 48, 128, 128, QWEN38["gdn_conv_dim"], 4, 4, 2)), 3),
    }

    g = tensors(snap(GEMMA431B["repo"]))
    gc = cats(g)
    out["gemma431b_ckpt_categories_gib"] = {k: round(gib(v), 3) for k, v in sorted(gc.items(), key=lambda kv: -kv[1])}
    attn = gc.get("attn_plus_gdn", 0)
    i8_head = GEMMA431B["vocab"] * GEMMA431B["hidden"]
    base_resident = sum(v for k, v in gc.items() if k != "multimodal")
    out["gemma431b_text_resident_default_gib"] = {
        "ckpt_no_mm": round(gib(base_resident), 3),
        "plus_i8_lmhead_copy_unless_NV_VERIFY_LMHEAD_INT8_0": round(gib(i8_head), 3),
        "total": round(gib(base_resident + i8_head), 3),
    }
    out["gemma431b_attn_proj_quant_nvfp4_arm_gib"] = {
        "attn_bf16_ckpt": round(gib(attn), 3),
        "attn_nvfp4_resident": round(gib(attn * NVFP4_RESIDENT_BYTES_PER_BF16_BYTE), 3),
        "delta_freed": round(gib(attn * (1 - NVFP4_RESIDENT_BYTES_PER_BF16_BYTE)), 3),
    }
    for seq in (32768, 131072, 262144):
        full = {
            "formula": "10 full layers x 16 kv_heads x 256 hd x seq x dtype x 2",
            "bf16": round(gib(kv_full_attn_bytes(10, 16, 256, seq, 2)), 3),
            "fp8": round(gib(kv_full_attn_bytes(10, 16, 256, seq, 1)), 3),
        }
        out[f"gemma431b_kv_full10_at_{seq}_gib"] = full
    out["gemma431b_kv_sliding50_ring_gib"] = {
        "formula": "50 sliding layers x 16 kv_heads x 256 hd x window1024 x dtype x 2 (ring caps at window)",
        "bf16": round(gib(kv_full_attn_bytes(50, 16, 256, 1024, 2)), 3),
        "fp8": round(gib(kv_full_attn_bytes(50, 16, 256, 1024, 1)), 3),
    }

    e = tensors(snap(EAGLE3["repo"]))
    ec = cats(e)
    embed = ec.get("embed", 0)
    out["eagle3_drafter_gib"] = {
        "ckpt_total": round(gib(sum(ec.values())), 3),
        "embed_shared_with_target_when_dims_match": round(gib(embed), 3),
        "resident_excl_shared_embed": round(gib(sum(ec.values()) - embed), 3),
    }

    for name, repo in AUX_SAFETENSORS.items():
        ts = tensors(snap(repo))
        out[f"aux_{name}_ckpt_gib"] = round(gib(sum(b for _d, _s, b in ts.values())), 3)

    for name, (repo, rel) in AUX_FILES_LAZY_OR_CPU.items():
        p = os.path.join(snap(repo), rel)
        out[f"aux_{name}_file_gib"] = round(gib(os.path.getsize(p)), 3) if os.path.exists(p) else None

    return out

if __name__ == "__main__":
    print(json.dumps(report(), indent=1))
