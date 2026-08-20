#!/usr/bin/env python3
"""
One-shot exporter for DiariZen-Large-s80-v2's segmentation forward pass.

The DiariZen authors ship PyTorch only. Speaches-plus runs ONNX at runtime.
Run this once after `fetch-models.sh` to produce models/diarizen-segmentation.onnx;
no Python is needed thereafter.

Usage:
    python3 scripts/export-diarizen-onnx.py \
        models/diarizen-large-s80-v2 \
        models/diarizen-segmentation.onnx

Requirements (one-time, in a Python venv):
    pip install torch torchaudio diarizen toml

License notes:
    Code: MIT. Weights: CC-BY-NC-4.0 -- see models/diarizen-large-s80-v2/MODEL_LICENSE
    in the upstream HF repo. By running this exporter you accept the
    non-commercial terms attached to the weights.
"""

import argparse
import sys
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="Path to diarizen-large-s80-v2 dir")
    parser.add_argument("output", type=Path, help="Output .onnx path")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=5.0,
        help="Input chunk length in seconds (default 5.0)",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Sample rate in Hz (default 16000)",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset")
    args = parser.parse_args()

    try:
        import torch
        import toml
        from diarizen.models.eend.model_wavlm_conformer import Model as DiariZenModel
    except ImportError as e:
        print(f"missing dependency: {e}", file=sys.stderr)
        print("install: pip install torch torchaudio diarizen toml", file=sys.stderr)
        return 2

    config_path = args.model_dir / "config.toml"
    weights_path = args.model_dir / "pytorch_model.bin"
    if not config_path.exists():
        print(f"missing {config_path}", file=sys.stderr)
        return 1
    if not weights_path.exists():
        print(f"missing {weights_path}", file=sys.stderr)
        return 1

    config = toml.load(config_path)
    model_args = config.get("model", {}).get("args", {})
    print(f"loading DiariZen with config: {model_args}")

    model = DiariZenModel(**model_args)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  warn: missing keys: {len(missing)} (showing 3): {missing[:3]}")
    if unexpected:
        print(f"  warn: unexpected keys: {len(unexpected)} (showing 3): {unexpected[:3]}")

    model.eval()

    num_samples = int(args.chunk_seconds * args.sample_rate)
    dummy = torch.randn(1, 1, num_samples, dtype=torch.float32)

    with torch.no_grad():
        out = model(dummy)
    print(f"forward output shape: {tuple(out.shape)}  (expect [1, T_frames, C_powerset])")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"exporting to {args.output} (opset {args.opset})...")

    import torch.nn.functional as _F

    _orig_layer_norm = _F.layer_norm

    def _static_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
        normalized_shape = tuple(int(s) for s in normalized_shape)
        return _orig_layer_norm(input, normalized_shape, weight, bias, eps)

    _F.layer_norm = _static_layer_norm

    torch.onnx.export(
        model,
        (dummy,),
        str(args.output),
        input_names=["waveform"],
        output_names=["scores"],
        dynamic_axes={
            "waveform": {0: "batch"},
            "scores": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )

    size_mb = args.output.stat().st_size / 1e6
    print(f"wrote {args.output} ({size_mb:.1f} MB)")
    print()
    print("verify with onnxruntime:")
    print(f"  python3 -c \"import onnxruntime as ort, numpy as np; "
          f"s=ort.InferenceSession('{args.output}'); "
          f"x=np.random.randn(1,1,{num_samples}).astype('float32'); "
          f"y=s.run(None,{{'waveform':x}})[0]; print(y.shape)\"")
    return 0

if __name__ == "__main__":
    sys.exit(main())
