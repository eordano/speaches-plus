import io
import struct
import sys

import numpy as np
import pyarrow.parquet as pq
import onnxruntime as ort
import soundfile as sf

sys.path.insert(0, "/tmp/eot-scratch")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from text_head_fnv import hashed_features
import eot_silence_ab as F

SMART_TURN = "/tmp/models/smart-turn-v3.2-cpu.onnx"
HEAD = "/tmp/eot-scratch/text-head-fnv.bin"
SR = 16000
DECIDE_AT = 0.2

raw = open(HEAD, "rb").read()
_, bias = struct.unpack_from("<If", raw, 4)
W = np.frombuffer(raw, dtype=np.float32, offset=12)

def p_text(text):
    z = bias + W[hashed_features(text)].sum()
    return 1.0 / (1.0 + np.exp(-z))

so = ort.SessionOptions()
so.intra_op_num_threads = 8
st = ort.InferenceSession(SMART_TURN, so, providers=["CPUExecutionProvider"])
st_in = st.get_inputs()[0].name

DIGIT_WORDS = ("number", "code", "phone", "card", "spell", "digits", "email", "address", "confirm")

def context_feats(msgs):
    agent = ""
    for m in reversed(msgs):
        if m["role"] == "assistant":
            agent = m["content"].strip()
            break
    a = agent.lower()
    return [
        1.0 if agent.endswith("?") else 0.0,
        1.0 if any(k in a for k in DIGIT_WORDS) else 0.0,
        1.0 if a.startswith(("what", "which", "how", "when", "where", "who", "can ", "could ", "would ", "do ", "did ", "is ", "are ")) else 0.0,
        min(len(a.split()) / 30.0, 2.0),
        1.0 if "yes or no" in a or a.endswith("correct?") else 0.0,
    ]

t = pq.read_table("/tmp/eot-scratch/eot-en.parquet")
rows = t.to_pylist()
X_pa, X_pt, X_ctx, Y, CLIP = [], [], [], [], []
for ci, r in enumerate(rows):
    try:
        a, rate = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
        if a.ndim > 1:
            a = a.mean(axis=1)
        if rate != SR:
            x = np.linspace(0, len(a) - 1, int(len(a) * SR / rate))
            a = np.interp(x, np.arange(len(a)), a).astype(np.float32)
    except Exception:
        continue
    ctx = context_feats(r["messages"])
    spans = r["silence_spans"]
    if not spans:
        continue
    for si, s in enumerate(spans):
        final = si == len(spans) - 1
        if s["end"] - s["start"] < DECIDE_AT and not final:
            continue
        d = s["start"] + DECIDE_AT
        if d * SR > len(a):
            d = len(a) / SR
        clip = a[: int(d * SR)]
        if len(clip) < SR // 2:
            continue
        pa = float(st.run(None, {st_in: F.log_mel(F.prepare(clip))[None]})[0].reshape(-1)[0])
        text = " ".join(w["word"] for w in r["words"] if w["end"] <= d)
        X_pa.append(pa)
        X_pt.append(p_text(text))
        X_ctx.append(ctx)
        Y.append(1 if final else 0)
        CLIP.append(ci)

X_pa, X_pt, X_ctx, Y, CLIP = map(np.array, (X_pa, X_pt, X_ctx, Y, CLIP))
print(f"points={len(Y)} eot={Y.sum()} hold={(Y==0).sum()} clips={len(set(CLIP.tolist()))}")

def auc(s, y):
    o = np.argsort(-s)
    l = y[o]
    tp = np.cumsum(l) / max(l.sum(), 1)
    fp = np.cumsum(1 - l) / max((1 - l).sum(), 1)
    return float(np.trapezoid(tp, fp))

def cv_logistic(feats, y, clips, folds=5):
    scores = np.zeros(len(y))
    for k in range(folds):
        te = clips % folds == k
        tr = ~te
        Xtr, ytr = feats[tr], y[tr]
        w = np.zeros(feats.shape[1] + 1)
        for _ in range(3000):
            z = w[0] + Xtr @ w[1:]
            p = 1 / (1 + np.exp(-z))
            g = p - ytr
            gw = np.concatenate([[g.mean()], Xtr.T @ g / len(ytr) + 1e-3 * w[1:]])
            w -= 0.3 * gw
        scores[te] = w[0] + feats[te] @ w[1:]
    return scores

f_audio = X_pa.reshape(-1, 1)
f_at = np.column_stack([X_pa, X_pt])
f_atc = np.column_stack([X_pa, X_pt, X_ctx])
print(f"audio-only          AUC {auc(X_pa, Y):.4f}")
print(f"audio+turntext (cv) AUC {auc(cv_logistic(f_at, Y, CLIP), Y):.4f}")
print(f"audio+text+context  AUC {auc(cv_logistic(f_atc, Y, CLIP), Y):.4f}")

wrong = ((X_pa > 0.5) != (Y == 1))
print(f"\naudio-wrong subset n={wrong.sum()} ({wrong.mean()*100:.1f}%): base eot-rate {Y[wrong].mean()*100:.1f}%")
s_atc = cv_logistic(f_atc, Y, CLIP)
s_at = cv_logistic(f_at, Y, CLIP)
print(f"  turntext AUC on audio-wrong: {auc(s_at[wrong], Y[wrong]):.4f}")
print(f"  +context AUC on audio-wrong: {auc(s_atc[wrong], Y[wrong]):.4f}")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def point_metrics(p, y):
    cd = float((p[y == 1] > 0.5).mean() * 100) if (y == 1).any() else 0.0
    fc = float((p[y == 0] > 0.5).mean() * 100) if (y == 0).any() else 0.0
    return cd, fc


def band_gated(pa, pt, y, clips, lo, hi, folds=5):
    out = pa.copy()
    band = (pa > lo) & (pa < hi)
    for k in range(folds):
        te = clips % folds == k
        tr = ~te & band
        if tr.sum() < 30:
            continue
        z1, z2, yt = logit(pa[tr]), logit(pt[tr]), y[tr]
        w = np.zeros(3)
        for _ in range(3000):
            z = w[0] + w[1] * z1 + w[2] * z2
            p = 1 / (1 + np.exp(-z))
            g = p - yt
            w -= 0.3 * np.array([g.mean(), (g * z1).mean(), (g * z2).mean()])
        sel = te & band
        fused = 1 / (1 + np.exp(-(w[0] + w[1] * logit(pa[sel]) + w[2] * logit(pt[sel]))))
        out[sel] = np.clip(fused, lo + 1e-4, hi - 1e-4)
    return out, band


for lo, hi in [(0.35, 0.65), (0.2, 0.8)]:
    p_bg, band = band_gated(X_pa, X_pt, Y, CLIP, lo, hi)
    cd0, fc0 = point_metrics(X_pa, Y)
    cd1, fc1 = point_metrics(p_bg, Y)
    p_veto = np.where(band, np.minimum(X_pa, p_bg), X_pa)
    cd2, fc2 = point_metrics(p_veto, Y)
    print(
        f"\nband-gated fusion {lo}<p_audio<{hi}: band n={int(band.sum())} "
        f"({band.mean()*100:.1f}% of points, eot-rate {Y[band].mean()*100:.1f}%)"
    )
    print(f"  overall AUC   audio {auc(X_pa, Y):.4f} -> band-gated {auc(p_bg, Y):.4f}")
    if band.sum() >= 30:
        print(
            f"  in-band AUC   audio {auc(X_pa[band], Y[band]):.4f} -> "
            f"band-gated {auc(p_bg[band], Y[band]):.4f}"
        )
    print(f"  point@0.5     audio CD {cd0:.2f} FC {fc0:.2f} -> band-gated CD {cd1:.2f} FC {fc1:.2f}")
    print(f"  veto-only@0.5 (text may only hold a cut): CD {cd2:.2f} FC {fc2:.2f}")
