"""Independently verify fusion-weights-new.json before it touches fusion.rs.

Deliberately NOT a reuse of fusion_train.py: features and the gate are
re-implemented from rust/src/eou (fusion.rs GatedFusionFeatures::vector +
GatedFusionWeights::gate + combine_fusion_gated, heuristic.rs score_text with
the ellipsis rule), in f32 like the rust code, row-by-row without vectorized
shortcuts. Checks: (1) claimed held-out metrics reproduce within 0.005;
(2) new weights beat both current weights and audio-only on accuracy;
(3) the gate is not degenerate (g must not exceed 0.95 on >99% of rows);
(4) train/test file hygiene (only expected shard names, each >1000 rows);
(5) failures.log is empty and shard count is complete.
Exit 0 = verified; exit 1 = refuted, with reasons printed.
"""

import csv
import glob
import json
import math
import os
import re
import sys

FEAT = "/tmp/eot-scratch/fusion-features"
WEIGHTS = "/tmp/eot-scratch/fusion-weights-new.json"

CURRENT = dict(bias=0.866202, w_p_text=0.283641, w_p_audio=0.018662,
               w_audio_log_sec=0.560501, w_partial_log_chars=1.195453,
               w_strong_terminator=0.258435, w_soft_terminator=0.003248,
               w_continuation_last_word=0.081883)

HESITATION = {"uh", "um", "uhh", "umm", "er", "erm", "hmm", "like", "so"}
CONTINUATIONS = {"and", "or", "but", "with", "the", "a", "an", "to", "of", "for",
                 "is", "was", "are", "were", "because", "since", "if", "when",
                 "while", "as", "than", "that", "which", "who", "whom", "whose"}

problems = []

def f32(x):
    import struct
    return struct.unpack("f", struct.pack("f", x))[0]

def score_text(s):
    s = s.strip()
    if not s:
        return 0.1
    if s.endswith("...") or s.endswith("…"):
        return 0.15
    c = s[-1]
    if c in ".!?":
        return 0.95
    if c in ",;:-":
        return 0.25
    t = s.rstrip(" \t\n\r.!?,;:")
    parts = [p for p in re.split(r"[^\w'\-]", t) if p]
    lw = parts[-1].lower() if parts else ""
    if not lw:
        return 0.3
    if lw in HESITATION:
        return 0.15
    if lw in CONTINUATIONS:
        return 0.2
    return 0.6

def features(text, audio_ms):
    t = text.strip()
    if t.endswith("...") or t.endswith("…"):
        strong, soft, cont = False, False, True
    else:
        strong = bool(t) and t[-1] in ".!?"
        soft = bool(t) and not strong and t[-1] in ",;:-"
        m = list(re.finditer(r"[\w'\-]+", t))
        cont = bool(m) and m[-1].group(0).lower() in CONTINUATIONS
    return strong, soft, cont, len(t)

def fused_prob(w, text, audio_ms, p_audio):
    pt = min(max(f32(score_text(text)), 0.0), 1.0)
    pa = min(max(f32(p_audio), 0.0), 1.0)
    strong, soft, cont, chars = features(text, audio_ms)
    log_sec = f32(math.log(1.0 + audio_ms / 1000.0))
    log_chars = f32(math.log(1.0 + chars))
    z = (w["bias"] + w["w_p_text"] * pt + w["w_p_audio"] * pa
         + w["w_audio_log_sec"] * log_sec + w["w_partial_log_chars"] * log_chars
         + w["w_strong_terminator"] * strong + w["w_soft_terminator"] * soft
         + w["w_continuation_last_word"] * cont)
    g = 1.0 / (1.0 + math.exp(-z))
    return g * pa + (1.0 - g) * pt, g

def main():
    with open(WEIGHTS) as f:
        blob = json.load(f)
    w_new = blob["weights"]
    claimed = blob["held_out"]

    train_files = sorted(glob.glob(f"{FEAT}/train-*.csv"))
    test_files = sorted(glob.glob(f"{FEAT}/test-*.csv"))
    if len(train_files) != 83:
        problems.append(f"expected 83 train csvs, found {len(train_files)}")
    if len(test_files) != 10:
        problems.append(f"expected 10 test csvs, found {len(test_files)}")
    for f in train_files + test_files:
        base = os.path.basename(f)
        if not re.fullmatch(r"(train|test)-\d\d\.csv", base):
            problems.append(f"unexpected file {base}")
    flog = f"{FEAT}/failures.log"
    if os.path.exists(flog) and os.path.getsize(flog) > 0:
        problems.append("failures.log is non-empty")

    rows = []
    for f in test_files:
        n = 0
        with open(f) as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                n += 1
        if n < 1000:
            problems.append(f"{os.path.basename(f)} has only {n} rows")

    stats = {}
    for name, wts in (("new", w_new), ("current", CURRENT)):
        tp = fc = npos = nneg = 0
        g_hi = 0
        for r in rows:
            p, g = fused_prob(wts, r.get("text", ""), int(r["audio_ms"]), float(r["p_audio"]))
            lab = r["label"] == "1"
            if name == "new" and g > 0.95:
                g_hi += 1
            if lab:
                npos += 1
                tp += p > 0.5
            else:
                nneg += 1
                fc += p > 0.5
        acc = (tp + (nneg - fc)) / max(npos + nneg, 1)
        stats[name] = dict(acc=acc, cd=tp / max(npos, 1), fc=fc / max(nneg, 1))
        if name == "new" and len(rows) and g_hi / len(rows) > 0.99:
            problems.append(f"gate degenerate: g>0.95 on {g_hi/len(rows):.1%} of rows (audio-only in disguise)")

    aud_acc = sum((float(r["p_audio"]) > 0.5) == (r["label"] == "1") for r in rows) / max(len(rows), 1)

    for name in ("new", "current"):
        for k, ck in (("acc", "acc"), ("cd", "complete_detected"), ("fc", "false_cutoff")):
            got, want = stats[name][k], claimed[name][ck if ck in claimed[name] else k]
            if abs(got - want) > 0.005:
                problems.append(f"{name}.{k}: recomputed {got:.4f} vs claimed {want:.4f}")

    if not (stats["new"]["acc"] > stats["current"]["acc"] and stats["new"]["acc"] > aud_acc):
        problems.append(
            f"new acc {stats['new']['acc']:.4f} does not beat current "
            f"{stats['current']['acc']:.4f} and audio-only {aud_acc:.4f}"
        )

    print(f"test rows={len(rows)}")
    for name in ("new", "current"):
        s = stats[name]
        print(f"{name:8s}: acc {s['acc']*100:.2f}% cd {s['cd']*100:.2f}% fc {s['fc']*100:.2f}%")
    print(f"audio-only acc {aud_acc*100:.2f}%")
    if problems:
        print("REFUTED:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("VERIFIED")

if __name__ == "__main__":
    main()
