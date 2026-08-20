#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "numpy>=2.0",
# ]
# ///
"""train -- fit the gated-fusion gate on a feature JSONL produced by
extract_features.py. Output: trained weights printed as Python, Go and
Rust literals so a single re-train run can be pasted into all three
production sources:

    client/eou_lib/gate.py::DEFAULT_GATED_FUSION_WEIGHTS
    go/internal/eou/gated_fusion.go::DefaultGatedFusionWeights
    rust/src/realtime/eou.rs::DEFAULT_GATED_FUSION_WEIGHTS

Loss: binary cross-entropy on the COMBINED score r = g·p_audio +
(1-g)·p_text, NOT on g itself. The gate is a weighting coefficient,
not a label predictor -- see the chain-rule derivation below.

Gradients (per-sample):

    r       = g·pa + (1-g)·pt        with g = σ(z), z = θ·x
    dg/dz   = g(1-g)
    dr/dg   = pa - pt
    dL/dr   = (r - y) / (r·(1-r))     # BCE on r as a probability
    dL/dz   = dL/dr · dr/dg · dg/dz
    dL/dθj  = dL/dz · xj
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eou_lib import GatedFusionFeatures

logger = logging.getLogger("train")

def feature_vector(row: dict) -> np.ndarray:
    feat = GatedFusionFeatures(
        audio_ms=row["audio_ms"],
        partial_chars=row["partial_chars"],
        partial_ends_with_strong_terminator=row["ends_strong_terminator"],
        partial_ends_with_soft_terminator=row["ends_soft_terminator"],
        partial_last_word_is_continuation=row["last_word_continuation"],
    )
    return np.array(feat.vector(row["p_text_heuristic"], row["p_audio_smartturn"]),
                     dtype=np.float64)

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))

def train_logistic(features: np.ndarray, labels: np.ndarray,
                    lr: float, l2: float, epochs: int) -> np.ndarray:
    rng = np.random.default_rng(7)
    theta = (rng.uniform(-0.5, 0.5, size=features.shape[1]) * 0.02)
    n = float(len(features))
    for _ in range(epochs):
        z = features @ theta
        g = sigmoid(z)
        pa = features[:, 2]
        pt = features[:, 1]
        r = g * pa + (1 - g) * pt
        eps = 1e-9
        r = np.clip(r, eps, 1 - eps)
        dL_dr = (r - labels) / (r * (1 - r))
        dr_dg = pa - pt
        dg_dz = g * (1 - g)
        dL_dz = dL_dr * dr_dg * dg_dz
        grad = (features.T @ dL_dz) / n
        reg = np.copy(theta)
        reg[0] = 0
        grad = grad + l2 * reg
        theta -= lr * grad
    return theta

def eval_acc(features: np.ndarray, labels: np.ndarray, theta: np.ndarray) -> float:
    z = features @ theta
    g = sigmoid(z)
    pa = features[:, 2]
    pt = features[:, 1]
    r = g * pa + (1 - g) * pt
    pred = (r >= 0.5).astype(np.int64)
    return float(np.mean(pred == labels))

def eval_baseline(name: str, p_text: np.ndarray, p_audio: np.ndarray,
                   labels: np.ndarray) -> float:
    if name == "noisy_or":
        s = 1 - (1 - p_text) * (1 - p_audio)
    elif name == "weighted-0.5" or name == "mean":
        s = (p_text + p_audio) / 2
    elif name == "max":
        s = np.maximum(p_text, p_audio)
    elif name == "audio-only":
        s = p_audio
    elif name == "text-only":
        s = p_text
    else:
        raise ValueError(name)
    pred = (s >= 0.5).astype(np.int64)
    return float(np.mean(pred == labels))

def kfold(n: int, k: int, seed: int = 11):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    for i in range(k):
        test = idx[i::k]
        train = np.array([j for j in range(n) if j not in set(test.tolist())], dtype=np.int64)
        yield train, test

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--in", dest="in_path", default="/tmp/gated-fusion-real.jsonl")
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--l2", type=float, default=0.001)
    p.add_argument("--folds", type=int, default=5)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    rows = []
    with open(args.in_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            pa = r.get("p_audio_smartturn")
            if pa is None or not math.isfinite(pa):
                continue
            rows.append(r)
    if not rows:
        logger.error(f"no rows in {args.in_path}")
        return 1

    features = np.stack([feature_vector(r) for r in rows])
    labels = np.array([float(r["label"]) for r in rows])
    pos = int(labels.sum())
    logger.info(f"rows: {len(rows)} (pos={pos}, neg={len(rows) - pos})")

    if args.folds > 1:
        logger.info(f"=== {args.folds}-fold cross-validation (gated) ===")
        accs = []
        for fi, (train_idx, test_idx) in enumerate(kfold(len(rows), args.folds)):
            theta = train_logistic(features[train_idx], labels[train_idx],
                                    args.lr, args.l2, args.epochs)
            acc = eval_acc(features[test_idx], labels[test_idx], theta)
            logger.info(f"  fold {fi}: held-out acc = {acc * 100:.1f}%")
            accs.append(acc)
        m, s = float(np.mean(accs)), float(np.std(accs))
        logger.info(f"  mean held-out acc: {100 * m:.1f}% +/- {100 * s:.1f}%")

    pt = features[:, 1]
    pa = features[:, 2]
    logger.info(f"=== Same-set accuracy (full {len(rows)}-row corpus) ===")
    for name in ("audio-only", "text-only", "noisy_or", "weighted-0.5", "max"):
        logger.info(f"  {name:14s} {100 * eval_baseline(name, pt, pa, labels):.1f}%")
    theta = train_logistic(features, labels, args.lr, args.l2, args.epochs)
    full_acc = eval_acc(features, labels, theta)
    logger.info(f"  {'gated (this)':14s} {100 * full_acc:.1f}%")

    print()
    print(f"// Trained on a real-data corpus from pipecat-ai/smart-turn-data-v3-test.")
    print(f"// Corpus: {len(rows)} clips, {pos} positive / {len(rows) - pos} negative.")
    print()
    fields = [
        ("bias",                   theta[0]),
        ("w_p_text",               theta[1]),
        ("w_p_audio",              theta[2]),
        ("w_audio_log_sec",        theta[3]),
        ("w_partial_log_chars",    theta[4]),
        ("w_strong_terminator",    theta[5]),
        ("w_soft_terminator",      theta[6]),
        ("w_continuation_last_word", theta[7]),
    ]

    print("# === Python (client/eou_lib/gate.py) ===")
    print("DEFAULT_GATED_FUSION_WEIGHTS = GatedFusionWeights(")
    for k, v in fields:
        print(f"    {k}={v:.6f},")
    print(f"    trained_samples={len(rows)},")
    print(f"    trained_acc={full_acc:.4f},")
    print(")")
    print()

    print("// === Go (go/internal/eou/gated_fusion.go) ===")
    print("var DefaultGatedFusionWeights = GatedFusionWeights{")
    go_field = {
        "bias": "Bias",
        "w_p_text": "WPText",
        "w_p_audio": "WPAudio",
        "w_audio_log_sec": "WAudioLogSec",
        "w_partial_log_chars": "WPartialLogChars",
        "w_strong_terminator": "WStrongTerminator",
        "w_soft_terminator": "WSoftTerminator",
        "w_continuation_last_word": "WContinuationLastWord",
    }
    for k, v in fields:
        print(f"\t{go_field[k]}: {v:.6f},")
    print(f"\tTrainedSamples: {len(rows)},")
    print(f"\tTrainedAcc: {full_acc:.4f},")
    print("}")
    print()

    print("// === Rust (rust/src/realtime/eou.rs) ===")
    print("pub const DEFAULT_GATED_FUSION_WEIGHTS: GatedFusionWeights = GatedFusionWeights {")
    for k, v in fields:
        print(f"    {k}: {v:.6f},")
    print(f"    trained_samples: {len(rows)},")
    print(f"    trained_acc: {full_acc:.4f},")
    print("};")
    return 0

if __name__ == "__main__":
    sys.exit(main())
