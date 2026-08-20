"""Train and price the transcript text head for EOU endpointing.

Hashed char/word n-gram logistic regression over the whisper transcript,
trained on fusion-feature shards (extract via fusion_extract.py).  Reports
AUC against the corpus baselines (p_audio, the p_text heuristic), the
audio+text fused AUC, and the known whisper failure subset: incomplete
turns that whisper ends with a strong terminator.

Run with PYTHONHASHSEED fixed -- the feature hash must be stable:
  PYTHONHASHSEED=0 python text_head_probe.py

Env:
  TH_FEATURES     shard dir (default: this session's fusion-features)
  TH_TRAIN_GLOB   default "train-*.csv"
  TH_EVAL_GLOB    default "test-*.csv"
"""

import csv
import glob
import os
import re

import numpy as np

FEAT = os.environ.get(
    "TH_FEATURES",
    "/tmp/eot-scratch/fusion-features",
)
TRAIN_GLOBS = os.environ.get("TH_TRAIN_GLOB", "train-*.csv").split(",")
EVAL_GLOBS = os.environ.get("TH_EVAL_GLOB", "test-*.csv").split(",")
DIM = 2**18


def hashed_features(text):
    t = text.strip().lower()
    idx = []
    tail = t[-48:]
    for n in (2, 3, 4):
        for i in range(max(0, len(tail) - n + 1)):
            idx.append(hash(("c", n, tail[i : i + n])) % DIM)
    words = re.findall(r"[\w'\-]+|[.!?,;:…]", t)
    for w in words[-6:]:
        idx.append(hash(("w", w)) % DIM)
    for i in range(max(0, len(words) - 3), len(words) - 1):
        idx.append(hash(("b", words[i], words[i + 1])) % DIM)
    if t:
        idx.append(hash(("last_char", t[-1])) % DIM)
        idx.append(hash(("ellipsis", t.endswith("...") or t.endswith("…"))) % DIM)
    idx.append(hash(("nwords_bucket", min(len(words) // 4, 12))) % DIM)
    return idx


def load(globs):
    X, rows = [], []
    for g in globs:
        for f in sorted(glob.glob(f"{FEAT}/{g.strip()}")):
            with open(f) as fh:
                for r in csv.DictReader(fh):
                    X.append(hashed_features(r.get("text", "")))
                    rows.append(r)
    y = np.array([int(r["label"]) for r in rows], dtype=np.float64)
    return X, y, rows


def auc(scores, labels):
    order = np.argsort(-scores)
    l = labels[order]
    tps = np.cumsum(l) / max(l.sum(), 1)
    fps = np.cumsum(1 - l) / max((1 - l).sum(), 1)
    return float(np.trapezoid(tps, fps))


def train(X, Y, epochs=4, lr=0.15, lam=1e-6):
    w = np.zeros(DIM)
    b = 0.0
    rng = np.random.default_rng(0)
    order = np.arange(len(Y))
    for _ in range(epochs):
        rng.shuffle(order)
        for i in order:
            z = b + w[X[i]].sum()
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - Y[i]
            w[X[i]] -= lr * (g + lam * w[X[i]])
            b -= lr * g
        lr *= 0.5
    return w, b


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_2d(z1, z2, y, epochs=200, lr=0.05):
    a = np.zeros(3)
    for _ in range(epochs):
        z = a[0] + a[1] * z1 + a[2] * z2
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - y
        a[0] -= lr * g.mean()
        a[1] -= lr * (g * z1).mean()
        a[2] -= lr * (g * z2).mean()
    return a


def main():
    assert os.environ.get("PYTHONHASHSEED") == "0", "run with PYTHONHASHSEED=0"
    Xtr, Ytr, _ = load(TRAIN_GLOBS)
    Xte, Yte, rows_te = load(EVAL_GLOBS)
    print(f"train={len(Ytr)} eval={len(Yte)} (shard-level split)")

    w, b = train(Xtr, Ytr)
    z_text = np.array([b + w[x].sum() for x in Xte])

    p_audio = np.array([float(r["p_audio"]) for r in rows_te])
    p_text_heur = np.array([float(r["p_text"]) for r in rows_te])
    strong = np.array([int(r["strong"]) for r in rows_te])

    print(f"text-head AUC       {auc(z_text, Yte):.4f}")
    print(f"p_text heuristic    {auc(p_text_heur, Yte):.4f}")
    print(f"p_audio             {auc(p_audio, Yte):.4f}")

    z_audio_tr = logit(np.array([float(r['p_audio']) for r in load(TRAIN_GLOBS)[2]]))
    z_text_tr = np.array([b + w[x].sum() for x in Xtr])
    a = fit_2d(z_audio_tr, z_text_tr, Ytr)
    z_fused = a[0] + a[1] * logit(p_audio) + a[2] * z_text
    print(f"audio+text fused    {auc(z_fused, Yte):.4f}  (weights {a})")

    for lo, hi in [(0.2, 0.8), (0.35, 0.65)]:
        band = (p_audio > lo) & (p_audio < hi)
        n = int(band.sum())
        if n < 50:
            continue
        print(
            f"audio-uncertain band {lo}<p_audio<{hi}: n={n} "
            f"(complete {int(Yte[band].sum())}) | "
            f"audio-only AUC {auc(p_audio[band], Yte[band]):.4f} | "
            f"text-head AUC {auc(z_text[band], Yte[band]):.4f} | "
            f"fused AUC {auc(z_fused[band], Yte[band]):.4f}"
        )

    hard = (Yte == 0) & (strong == 1)
    print(
        f"whisper-trap subset (incomplete turns ending in a strong terminator): "
        f"n={int(hard.sum())}"
    )
    for name, s in [("text-head", z_text), ("p_text", p_text_heur), ("p_audio", p_audio)]:
        mask = hard | (Yte == 1)
        print(f"  {name:9} AUC on trap-vs-complete: {auc(s[mask], Yte[mask]):.4f}")


if __name__ == "__main__":
    main()
