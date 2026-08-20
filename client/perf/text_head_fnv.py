"""FNV-hashed text head trainer -- the stable-hash twin of text_head_probe.py.

The feature hash is FNV-1a (portable; rust/src/eou/text_head.rs implements the
identical function), so the trained weights ship as a binary the rust side
loads directly. The committed artifact and its golden probabilities live at
rust/tests/data/eou_text_head_fnv_v1.bin + rust/tests/eou_text_head_golden.rs.

Env: TH_FEATURES (shard dir from fusion_extract.py), TH_OUT (weights path).
"""

import csv
import glob
import os
import re
import struct

import numpy as np

FEAT = os.environ.get("TH_FEATURES", "fusion-features")
DIM = 2**18
OUT = os.environ.get("TH_OUT", "text-head-fnv.bin")


def fnv1a(data):
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def bucket(kind, payload):
    return fnv1a(kind.encode() + b"\x1f" + payload.encode("utf-8")) % DIM


TOKEN_RE = re.compile(r"[\w'\-]+|[.!?,;:…]")


def hashed_features(text):
    t = text.strip().lower()
    idx = []
    tail = t[-48:]
    for n in (2, 3, 4):
        for i in range(max(0, len(tail) - n + 1)):
            idx.append(bucket(f"c{n}", tail[i:i + n]))
    words = TOKEN_RE.findall(t)
    for w in words[-6:]:
        idx.append(bucket("w", w))
    for i in range(max(0, len(words) - 3), len(words) - 1):
        idx.append(bucket("b", words[i] + "\x1e" + words[i + 1]))
    if t:
        idx.append(bucket("lc", t[-1]))
        idx.append(bucket("el", "1" if (t.endswith("...") or t.endswith("…")) else "0"))
    idx.append(bucket("nw", str(min(len(words) // 4, 12))))
    return idx


def load(pattern):
    X, Y = [], []
    for f in sorted(glob.glob(f"{FEAT}/{pattern}")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                X.append(hashed_features(r.get("text", "")))
                Y.append(int(r["label"]))
    return X, np.array(Y, dtype=np.float64)


def auc(scores, labels):
    order = np.argsort(-scores)
    l = labels[order]
    tps = np.cumsum(l) / max(l.sum(), 1)
    fps = np.cumsum(1 - l) / max((1 - l).sum(), 1)
    return float(np.trapezoid(tps, fps))


def main():
    Xtr, Ytr = load("train-*.csv")
    Xte, Yte = load("test-*.csv")
    w = np.zeros(DIM)
    b = 0.0
    lr, lam = 0.15, 1e-6
    rng = np.random.default_rng(0)
    order = np.arange(len(Ytr))
    for epoch in range(4):
        rng.shuffle(order)
        for i in order:
            z = b + w[Xtr[i]].sum()
            p = 1.0 / (1.0 + np.exp(-z))
            g = p - Ytr[i]
            w[Xtr[i]] -= lr * (g + lam * w[Xtr[i]])
            b -= lr * g
        lr *= 0.5

    scores = np.array([b + w[x].sum() for x in Xte])
    p = 1.0 / (1.0 + np.exp(-scores))
    print(f"fnv head: AUC {auc(scores, Yte):.4f}")
    for thr in (0.8, 0.9, 0.95, 0.98):
        m = p >= thr
        if m.sum() == 0: continue
        prec = (Yte[m] == 1).mean()
        cov = m.mean()
        print(f"  p>={thr}: precision {prec*100:.2f}%  coverage {cov*100:.1f}%")

    with open(OUT, "wb") as f:
        f.write(b"STH1")
        f.write(struct.pack("<If", DIM, b))
        f.write(w.astype(np.float32).tobytes())
    print(f"weights -> {OUT} ({4 + 8 + DIM*4} bytes)")

    sample = "so I was thinking maybe we could"
    print("golden:", [f"{i}" for i in sorted(hashed_features(sample))[:5]],
          f"score {b + w[hashed_features(sample)].sum():.6f}")


if __name__ == "__main__":
    main()
