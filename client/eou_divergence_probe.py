#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy>=2.0",
#   "onnxruntime>=1.20",
# ]
# ///
"""eou_divergence_probe -- run smart-turn-v3 directly on captured mic_in.raw
recordings from inspector dirs, at a sweep of trailing-window positions.
Used for post-mortem: when EOU verdicts disagree across transports / runs,
pull the mic_in audio and ask "would the model say EOT here?" without the
VAD / commit pipeline in the way.

Replaces the older Go cmd/eou-divergence-probe -- same outputs, no Go
toolchain required.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eou_lib import SmartTurn

SAMPLE_RATE = 16000

def load_raw_s16(path: str) -> np.ndarray:
    """mic_in.raw is little-endian s16le @ 16 kHz mono. Return float32 in [-1, 1]."""
    raw = Path(path).read_bytes()
    int16 = np.frombuffer(raw, dtype="<i2")
    return (int16.astype(np.float32) / 32767.0).copy()

def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model",
                   default=str(Path(__file__).resolve().parents[1] / "rust/models/smart-turn-v3.onnx"))
    p.add_argument("paths", nargs="*",
                   help="mic_in.raw paths to probe; if empty, defaults to a "
                        "fixed set under /tmp/eou-runs[-rust]/audio/.")
    args = p.parse_args()

    smart = SmartTurn.load(args.model)

    silence = np.zeros(8 * SAMPLE_RATE, dtype=np.float32)
    print(f"PURE SILENCE (8 s zeros)  -> smart-turn = {smart.score(silence):.4f}")
    print(f"NIL AUDIO (empty array)   -> smart-turn = "
          f"{smart.score(np.zeros(0, dtype=np.float32)):.4f}")
    print()

    paths = list(args.paths)
    if not paths:
        for root in ("/tmp/eou-runs", "/tmp/eou-runs-rust"):
            for fixture in ("canonical", "continuation", "hesitation"):
                d = Path(root) / "audio" / fixture / "inspector"
                if d.exists():
                    for f in d.glob("*.audio_mic_in.raw"):
                        paths.append(str(f))

    print(f"{'recording':60s} {'len_ms':>8s} {'rms':>8s} {'@end':>10s}  sweep (-ms_from_end)")
    for path in paths:
        try:
            samples = load_raw_s16(path)
        except FileNotFoundError:
            print(f"{path:60s}  (not found)")
            continue
        len_ms = len(samples) * 1000 / SAMPLE_RATE
        end_score = smart.score(samples)
        sweep_parts = []
        for off in (6000, 4000, 2000, 1000, 500, 0):
            end_idx = len(samples) - int(off * SAMPLE_RATE / 1000)
            if end_idx < int(0.5 * SAMPLE_RATE):
                continue
            start_idx = max(0, end_idx - 8 * SAMPLE_RATE)
            score = smart.score(samples[start_idx:end_idx])
            sweep_parts.append(f"-{off:4d}:{score:.3f}")
        label = path
        if len(label) > 60:
            label = "..." + label[-57:]
        print(f"{label:60s} {len_ms:>8.1f} {rms(samples):>8.4f} {end_score:>10.4f}  "
              f"{' '.join(sweep_parts)}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
