#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = []
# ///
"""gate_probe -- emit the gated-fusion combined score Python produces for
a fixed set of (partial, audio_ms, p_text, p_audio) tuples. Used to seed
the Go and Rust parity tests so all three implementations evaluate to
the same number on the same input.

If you re-train (client/gated_fusion/train.py), update the literals in
ALL THREE sources:

    client/eou_lib/gate.py::DEFAULT_GATED_FUSION_WEIGHTS
    go/internal/eou/gated_fusion.go::DefaultGatedFusionWeights
    rust/src/realtime/eou.rs::DEFAULT_GATED_FUSION_WEIGHTS

then run this probe again and paste the new expected values into
go/internal/eou/gated_fusion_test.go::TestGatedRustGoPythonParity (if
you add one) and rust/.../eou.rs::gated_rust_go_byte_for_byte_parity.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eou_lib import (
    DEFAULT_GATED_FUSION_WEIGHTS,
    combine_fusion_gated,
    extract_gated_fusion_features,
)

CASES = [
    ("That's right.", 1500, 0.95, 0.99),
    ("Yes.", 1500, 0.95, 0.95),
    ("and the next thing", 1500, 0.55, 0.05),
    ("the cat is on the", 1500, 0.25, 0.05),
    ("looking forward to it", 1500, 0.55, 0.50),
]

def main() -> int:
    w = DEFAULT_GATED_FUSION_WEIGHTS
    for partial, ms, pt, pa in CASES:
        feat = extract_gated_fusion_features(partial, ms)
        got = combine_fusion_gated(pt, pa, feat, w)
        print(f"{partial!r:32s} ms={ms} pt={pt:.2f} pa={pa:.2f} -> {got:.6f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
