#!/usr/bin/env python3
"""Export the Qwen3.8 vision tower (model.visual.*) to ONNX for ort GPU serving.

The serving reference is the candle implementation in
rust/crates/nv-omni/src/qwen3_vision.rs; this exporter mirrors its exact
semantics so the ONNX graph can replace the CPU tower behind the existing
EmbedRowSplice seam ([N, out_hidden_size] f32 rows):

  - temporal stacking: each input channel repeated temporal_patch_size times
    (channel-major interleave, matching the candle per-channel cat order)
  - patch embed: conv2d, weight reshaped (h, c*tp, p, p), stride = patch_size
  - pos embed: bilinear resize of the square 48x48 table with align_corners
    semantics (candle linspace_align: src = i*(side-1)/(g-1))
  - 27 blocks: LayerNorm(eps from config, default 1e-6) -> fused qkv ->
    non-causal sdpa (scale 1/sqrt(head_dim)) -> proj -> residual;
    LayerNorm -> fc1 -> gelu_pytorch_tanh -> fc2 -> residual
  - spatial merge: patches regrouped into merge x merge blocks
    (merge_order_indices), then LayerNorm -> fc1 -> gelu -> fc2
  - output: [gh*gw/merge^2, out_hidden_size] f32

Weights are read as bf16 from the checkpoint shards (the tower is on the quant
ignore list) and upcast to f32; the graph computes in f32. The candle tower
computes matmuls in bf16, so parity against it is judged by per-row cosine,
not bitwise equality (rust/tests/qwen38_vision_ort.rs owns that gate).

The export is fixed-shape: one graph per (height, width). Serving arbitrary
smart-resize grids needs either a bucket set of exports or a dynamic-shape
(dynamo) export; that wiring is out of scope here.

Usage:
    python3 scripts/export_qwen38_vision_onnx.py \
        --model-dir ~/.cache/huggingface/hub/models--unsloth--Qwen3.8-27B-NVFP4/snapshots/<sha> \
        --out /path/qwen38-vision-448.onnx [--height 448 --width 448] [--skip-check]

Requires torch + safetensors; the self-check additionally uses onnxruntime
(CPU) and asserts torch-vs-onnx max |diff| on a synthetic photo.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def resolve_model_dir(arg: str | None) -> Path:
    if arg:
        p = Path(arg).expanduser()
        assert p.is_dir(), f"--model-dir {p} is not a directory"
        return p
    env = os.environ.get("NV_QWEN38_DIR", "")
    if env:
        p = Path(env).expanduser()
        assert p.is_dir(), f"NV_QWEN38_DIR={env} is not a directory"
        return p
    hub = Path("~/.cache/huggingface/hub").expanduser()
    snaps = sorted(hub.glob("models--unsloth--Qwen3.8-27B-NVFP4/snapshots/*/config.json"))
    assert snaps, f"no Qwen3.8-27B-NVFP4 snapshot under {hub}; pass --model-dir or set NV_QWEN38_DIR"
    return snaps[-1].parent


def load_visual_state(model_dir: Path) -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            if name.startswith("model.visual."):
                shards.setdefault(shard, []).append(name)
    else:
        single = model_dir / "model.safetensors"
        assert single.is_file(), f"neither index json nor model.safetensors under {model_dir}"
        shards = {"model.safetensors": []}

    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        with safe_open(model_dir / shard, framework="pt", device="cpu") as f:
            keys = names if names else [k for k in f.keys() if k.startswith("model.visual.")]
            for k in keys:
                state[k] = f.get_tensor(k).to(torch.float32)
    assert state, f"no model.visual.* tensors found in {model_dir}"
    return state


def merge_order_indices(gh: int, gw: int, merge: int) -> torch.Tensor:
    bh, bw, group = gh // merge, gw // merge, merge * merge
    idx = torch.zeros(gh * gw, dtype=torch.long)
    for br in range(bh):
        for bc in range(bw):
            for dr in range(merge):
                for dc in range(merge):
                    out = (br * bw + bc) * group + dr * merge + dc
                    src = (merge * br + dr) * gw + (merge * bc + dc)
                    idx[out] = src
    return idx


def gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    return F.gelu(x, approximate="tanh")


class Block(nn.Module):
    def __init__(self, h: int, inter: int, num_heads: int, eps: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(h, eps=eps)
        self.norm2 = nn.LayerNorm(h, eps=eps)
        self.qkv = nn.Linear(h, 3 * h)
        self.proj = nn.Linear(h, h)
        self.fc1 = nn.Linear(h, inter)
        self.fc2 = nn.Linear(inter, h)
        self.num_heads = num_heads
        self.head_dim = h // num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        nh, hd = self.num_heads, self.head_dim
        qkv = self.qkv(self.norm1(x)).view(b, t, 3, nh, hd)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) * (1.0 / math.sqrt(hd))
        attn = torch.matmul(torch.softmax(scores, dim=-1), v)
        attn = attn.transpose(1, 2).reshape(b, t, h)
        x = x + self.proj(attn)
        return x + self.fc2(gelu_pytorch_tanh(self.fc1(self.norm2(x))))


class VisionTower(nn.Module):
    def __init__(self, vcfg: dict, height: int, width: int):
        super().__init__()
        self.h = vcfg["hidden_size"]
        self.depth = vcfg["depth"]
        self.num_heads = vcfg["num_heads"]
        self.inter = vcfg["intermediate_size"]
        self.c = vcfg.get("in_channels", 3)
        self.p = vcfg["patch_size"]
        self.tp = vcfg.get("temporal_patch_size", 2)
        self.merge = vcfg.get("spatial_merge_size", 2)
        self.npos = vcfg["num_position_embeddings"]
        self.out_h = vcfg["out_hidden_size"]
        self.eps = vcfg.get("layer_norm_eps", 1e-6)
        side = round(math.sqrt(self.npos))
        assert side * side == self.npos, f"pos table {self.npos} is not a perfect square"
        self.side = side
        factor = self.p * self.merge
        assert height % factor == 0 and width % factor == 0, (
            f"export size {height}x{width} must be a multiple of patch*merge={factor}"
        )
        self.gh, self.gw = height // self.p, width // self.p

        self.patch_weight = nn.Parameter(torch.zeros(self.h, self.c, self.tp, self.p, self.p))
        self.patch_bias = nn.Parameter(torch.zeros(self.h))
        self.pos_embed = nn.Parameter(torch.zeros(self.npos, self.h))
        self.blocks = nn.ModuleList(
            Block(self.h, self.inter, self.num_heads, self.eps) for _ in range(self.depth)
        )
        merged = self.h * self.merge * self.merge
        self.merger_norm = nn.LayerNorm(self.h, eps=self.eps)
        self.merger_fc1 = nn.Linear(merged, merged)
        self.merger_fc2 = nn.Linear(merged, self.out_h)
        self.register_buffer("merge_idx", merge_order_indices(self.gh, self.gw, self.merge))

    def load_checkpoint(self, state: dict[str, torch.Tensor]) -> None:
        pre = "model.visual."
        used = set()

        def take(name: str, shape: tuple) -> torch.Tensor:
            t = state[pre + name]
            assert tuple(t.shape) == shape, f"{pre+name}: expected {shape}, got {tuple(t.shape)}"
            used.add(pre + name)
            return t

        h, c, tp, p = self.h, self.c, self.tp, self.p
        with torch.no_grad():
            self.patch_weight.copy_(take("patch_embed.proj.weight", (h, c, tp, p, p)))
            self.patch_bias.copy_(take("patch_embed.proj.bias", (h,)))
            self.pos_embed.copy_(take("pos_embed.weight", (self.npos, h)))
            for i, blk in enumerate(self.blocks):
                bp = f"blocks.{i}."
                blk.norm1.weight.copy_(take(bp + "norm1.weight", (h,)))
                blk.norm1.bias.copy_(take(bp + "norm1.bias", (h,)))
                blk.norm2.weight.copy_(take(bp + "norm2.weight", (h,)))
                blk.norm2.bias.copy_(take(bp + "norm2.bias", (h,)))
                blk.qkv.weight.copy_(take(bp + "attn.qkv.weight", (3 * h, h)))
                blk.qkv.bias.copy_(take(bp + "attn.qkv.bias", (3 * h,)))
                blk.proj.weight.copy_(take(bp + "attn.proj.weight", (h, h)))
                blk.proj.bias.copy_(take(bp + "attn.proj.bias", (h,)))
                blk.fc1.weight.copy_(take(bp + "mlp.linear_fc1.weight", (self.inter, h)))
                blk.fc1.bias.copy_(take(bp + "mlp.linear_fc1.bias", (self.inter,)))
                blk.fc2.weight.copy_(take(bp + "mlp.linear_fc2.weight", (h, self.inter)))
                blk.fc2.bias.copy_(take(bp + "mlp.linear_fc2.bias", (h,)))
            merged = h * self.merge * self.merge
            self.merger_norm.weight.copy_(take("merger.norm.weight", (h,)))
            self.merger_norm.bias.copy_(take("merger.norm.bias", (h,)))
            self.merger_fc1.weight.copy_(take("merger.linear_fc1.weight", (merged, merged)))
            self.merger_fc1.bias.copy_(take("merger.linear_fc1.bias", (merged,)))
            self.merger_fc2.weight.copy_(take("merger.linear_fc2.weight", (self.out_h, merged)))
            self.merger_fc2.bias.copy_(take("merger.linear_fc2.bias", (self.out_h,)))
        unused = sorted(set(state) - used)
        assert not unused, f"checkpoint has {len(unused)} unconsumed visual tensors: {unused[:5]}"

    def interpolated_pos(self) -> torch.Tensor:
        gh, gw, side = self.gh, self.gw, self.side
        if gh == side and gw == side:
            return self.pos_embed
        grid = self.pos_embed.t().reshape(1, self.h, side, side)
        out = F.interpolate(grid, size=(gh, gw), mode="bilinear", align_corners=True)
        return out.reshape(self.h, gh * gw).t()

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b = pixel_values.shape[0]
        x = pixel_values.repeat_interleave(self.tp, dim=1)
        w = self.patch_weight.reshape(self.h, self.c * self.tp, self.p, self.p)
        y = F.conv2d(x, w, self.patch_bias, stride=self.p)
        x = y.flatten(2).transpose(1, 2)
        x = x + self.interpolated_pos().unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        x = x[:, self.merge_idx, :]
        n_merged = (self.gh * self.gw) // (self.merge * self.merge)
        x = self.merger_norm(x).reshape(b, n_merged, self.h * self.merge * self.merge)
        x = self.merger_fc2(gelu_pytorch_tanh(self.merger_fc1(x)))
        return x.reshape(b * n_merged, self.out_h)


def synthetic_photo(h: int, w: int) -> torch.Tensor:
    ys = torch.arange(h, dtype=torch.float32).view(h, 1)
    xs = torch.arange(w, dtype=torch.float32).view(1, w)
    fr = (xs / w + ys / h) * 0.5
    fg = ((xs * 3 + ys) % 251) / 251.0
    fb = ((xs + ys * 2) % 241) / 241.0
    img = torch.stack([fr.expand(h, w), fg.expand(h, w), fb.expand(h, w)], dim=0)
    return ((img - 0.5) / 0.5).unsqueeze(0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--height", type=int, default=448)
    ap.add_argument("--width", type=int, default=448)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--check-tol", type=float, default=1e-2)
    ap.add_argument("--check-cos-floor", type=float, default=0.9999)
    args = ap.parse_args()

    model_dir = resolve_model_dir(args.model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())
    vcfg = cfg["vision_config"]
    print(f"model_dir={model_dir}", flush=True)
    print(f"vision_config depth={vcfg['depth']} hidden={vcfg['hidden_size']} out={vcfg['out_hidden_size']}", flush=True)

    tower = VisionTower(vcfg, args.height, args.width)
    state = load_visual_state(model_dir)
    print(f"loaded {len(state)} visual tensors", flush=True)
    tower.load_checkpoint(state)
    tower.eval()

    example = synthetic_photo(args.height, args.width)
    with torch.no_grad():
        ref = tower(example)
    n_rows = (args.height // vcfg["patch_size"]) * (args.width // vcfg["patch_size"]) // 4
    assert tuple(ref.shape) == (n_rows, vcfg["out_hidden_size"]), f"got {tuple(ref.shape)}"
    assert torch.isfinite(ref).all(), "torch reference has non-finite values"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        tower,
        (example,),
        str(args.out),
        input_names=["pixel_values"],
        output_names=["embeddings"],
        opset_version=args.opset,
        dynamo=False,
    )
    size_mb = args.out.stat().st_size / 1e6
    print(f"exported {args.out} ({size_mb:.1f} MB) input [1,3,{args.height},{args.width}] output [{n_rows},{vcfg['out_hidden_size']}]", flush=True)

    if args.skip_check:
        return 0

    import numpy as np
    import onnxruntime as rt

    sess = rt.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    (out,) = sess.run(["embeddings"], {"pixel_values": example.numpy()})
    diff = np.abs(out - ref.numpy())
    ref_np = ref.numpy()
    cos = (out * ref_np).sum(axis=1) / (
        np.linalg.norm(out, axis=1) * np.linalg.norm(ref_np, axis=1)
    )
    print(f"onnxruntime-cpu vs torch: max|diff|={diff.max():.3e} mean|diff|={diff.mean():.3e} min_row_cos={cos.min():.6f}", flush=True)
    assert diff.max() < args.check_tol, (
        f"export self-check failed: max|diff|={diff.max():.3e} >= {args.check_tol}"
    )
    assert cos.min() >= args.check_cos_floor, (
        f"export self-check failed: min_row_cos={cos.min():.6f} < {args.check_cos_floor}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
