"""Fit the gated-fusion weights on extracted feature CSVs.

Text features are RE-DERIVED from the stored transcript column using the
post-ellipsis-fix definitions (rust commit 841a15909): trailing '...' or
U+2026 scores hesitation-class in p_text and maps to continuation=true with
no terminator flags. The gate replicates rust combine_fusion_gated:
g = sigmoid(w . v), p = g*p_audio + (1-g)*p_text,
v = [1, p_text, p_audio, ln(1+sec), ln(1+chars), strong, soft, continuation].
"""

import csv
import glob
import json
import re
import sys

import numpy as np

FEAT_DIR = "/tmp/eot-scratch/fusion-features"
OUT = "/tmp/eot-scratch/fusion-weights-new.json"

CURRENT = np.array([0.866202, 0.283641, 0.018662, 0.560501, 1.195453, 0.258435, 0.003248, 0.081883])
NAMES = ["bias", "w_p_text", "w_p_audio", "w_audio_log_sec", "w_partial_log_chars",
         "w_strong_terminator", "w_soft_terminator", "w_continuation_last_word"]

HESITATION = {"uh", "um", "uhh", "umm", "er", "erm", "hmm", "like", "so"}
CONTINUATIONS = {"and", "or", "but", "with", "the", "a", "an", "to", "of", "for", "is",
                 "was", "are", "were", "because", "since", "if", "when", "while", "as",
                 "than", "that", "which", "who", "whom", "whose"}

def last_word(s):
    t = s.rstrip(" \t\n\r.!?,;:")
    parts = [p for p in re.split(r"[^\w'\-]", t) if p]
    return parts[-1] if parts else ""

def derive(text, audio_ms):
    t = text.strip()
    ellipsis = t.endswith("...") or t.endswith("…")
    if not t:
        p_text = 0.1
    elif ellipsis:
        p_text = 0.15
    elif t[-1] in ".!?":
        p_text = 0.95
    elif t[-1] in ",;:-":
        p_text = 0.25
    else:
        lw = last_word(t).lower()
        if not lw:
            p_text = 0.3
        elif lw in HESITATION:
            p_text = 0.15
        elif lw in CONTINUATIONS:
            p_text = 0.2
        else:
            p_text = 0.6
    if ellipsis:
        strong, soft, cont = False, False, True
    else:
        strong = bool(t) and t[-1] in ".!?"
        soft = bool(t) and not strong and t[-1] in ",;:-"
        m = list(re.finditer(r"[\w'\-]+", t))
        cont = bool(m) and m[-1].group(0).lower() in CONTINUATIONS
    log_sec = np.log(1.0 + audio_ms / 1000.0)
    log_chars = np.log(1.0 + len(t))
    return p_text, [1.0, p_text, 0.0, log_sec, log_chars, float(strong), float(soft), float(cont)]

def load(pattern):
    V, PT, PA, Y = [], [], [], []
    for f in sorted(glob.glob(f"{FEAT_DIR}/{pattern}")):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                pa = float(r["p_audio"])
                p_text, v = derive(r.get("text", ""), int(r["audio_ms"]))
                v[2] = min(max(pa, 0.0), 1.0)
                v[1] = min(max(p_text, 0.0), 1.0)
                V.append(v)
                PT.append(p_text)
                PA.append(pa)
                Y.append(int(r["label"]))
    return np.array(V), np.array(PT), np.array(PA), np.array(Y, dtype=np.float64)

def fused(w, V, PT, PA):
    g = 1.0 / (1.0 + np.exp(-(V @ w)))
    return np.clip(g * np.clip(PA, 0, 1) + (1 - g) * np.clip(PT, 0, 1), 1e-7, 1 - 1e-7)

def metrics(p, y):
    pred = p > 0.5
    acc = float((pred == (y == 1)).mean())
    cd = float(pred[y == 1].mean()) if (y == 1).any() else 0.0
    fc = float(pred[y == 0].mean()) if (y == 0).any() else 0.0
    return acc, cd, fc

def main():
    Vtr, PTtr, PAtr, Ytr = load("train-*.csv")
    Vte, PTte, PAte, Yte = load("test-*.csv")
    if len(Yte) == 0:
        print("NOTE: no test CSVs yet -- holding out every 5th train row (preview only)")
        hold = np.arange(len(Ytr)) % 5 == 0
        Vte, PTte, PAte, Yte = Vtr[hold], PTtr[hold], PAtr[hold], Ytr[hold]
        Vtr, PTtr, PAtr, Ytr = Vtr[~hold], PTtr[~hold], PAtr[~hold], Ytr[~hold]
    print(f"train rows={len(Ytr)} test rows={len(Yte)}")
    if len(Ytr) < 1000:
        print("not enough training data yet")
        sys.exit(1)

    w = CURRENT.copy()
    m = np.zeros(8)
    v_adam = np.zeros(8)
    lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
    for t in range(1, 2001):
        g_gate = 1.0 / (1.0 + np.exp(-(Vtr @ w)))
        pa = np.clip(PAtr, 0, 1)
        pt = np.clip(PTtr, 0, 1)
        p = np.clip(g_gate * pa + (1 - g_gate) * pt, 1e-7, 1 - 1e-7)
        dl_dp = (p - Ytr) / (p * (1 - p)) / len(Ytr)
        dp_dg = pa - pt
        dg_dz = g_gate * (1 - g_gate)
        grad = Vtr.T @ (dl_dp * dp_dg * dg_dz)
        m = b1 * m + (1 - b1) * grad
        v_adam = b2 * v_adam + (1 - b2) * grad**2
        w -= lr * (m / (1 - b1**t)) / (np.sqrt(v_adam / (1 - b2**t)) + eps)
        w[1] = min(w[1], 0.0)
        w[2] = max(w[2], 0.0)

    results = {}
    for name, p in (
        ("new", fused(w, Vte, PTte, PAte)),
        ("current", fused(CURRENT, Vte, PTte, PAte)),
        ("audio_only", np.clip(PAte, 1e-7, 1 - 1e-7)),
        ("text_only", np.clip(PTte, 1e-7, 1 - 1e-7)),
    ):
        acc, cd, fc = metrics(p, Yte)
        results[name] = {"acc": acc, "complete_detected": cd, "false_cutoff": fc}
        print(f"{name:11s}: acc {acc*100:5.2f}%  complete-detected {cd*100:5.2f}%  false-cutoff {fc*100:5.2f}%")

    tr_acc = metrics(fused(w, Vtr, PTtr, PAtr), Ytr)[0]
    out = {
        "weights": dict(zip(NAMES, [round(float(x), 6) for x in w])),
        "trained_samples": int(len(Ytr)),
        "trained_acc": round(tr_acc, 4),
        "held_out": results,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"weights -> {OUT}")

if __name__ == "__main__":
    main()
