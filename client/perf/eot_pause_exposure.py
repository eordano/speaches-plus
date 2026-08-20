"""End-to-end exposure measurement for the silence-end 350ms -> 200ms flip.

The model-layer A/B (eot_silence_ab.py) showed smart-turn verdicts are
silence-length-invariant on turn-final clips.  What that cannot answer: a
shorter silence-end consults the classifier at MID-UTTERANCE pauses that the
longer one never sees.  This probe enumerates real intra-speech pauses in the
smart-turn-data clips via silero VAD, and for every internal pause long
enough to trigger a candidate silence-end, scores smart-turn at exactly the
moment the endpointer would consult it (pause start + candidate).  A
"complete" verdict at an internal pause is a would-be false cutoff.

Reported per candidate: consultations and would-be cutoffs per audio-minute,
plus the [200,350)ms band alone -- the marginal exposure the flip adds.

Env:
  EOT_PE_SHARDS       glob of .arrow (HF datasets IPC) or .parquet shards
  EOT_PE_SMART_TURN   smart-turn onnx (mel [1,80,800] input)
  EOT_PE_SILERO       silero vad onnx
  EOT_PE_SILENCES_MS  comma list, default "350,200"
  EOT_PE_LANG         language filter, default "eng"
  EOT_PE_MAX_ROWS     row cap across shards, default 2500
"""

import glob
import os

import numpy as np
import onnxruntime as ort

import eot_silence_ab as ab

SR = ab.SR
VAD_WINDOW, VAD_CONTEXT = ab.VAD_WINDOW, ab.VAD_CONTEXT
WINDOW_MS = VAD_WINDOW * 1000 // SR


def iter_rows(paths, columns):
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    for p in paths:
        if p.endswith(".parquet"):
            tbl = pq.read_table(p, columns=columns)
        else:
            try:
                tbl = ipc.open_stream(p).read_all()
            except Exception:
                tbl = ipc.open_file(p).read_all()
            tbl = tbl.select(columns)
        cols = [tbl.column(c).to_pylist() for c in columns]
        yield from zip(*cols)


def vad_speech_mask(vad, a):
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr = np.array(SR, dtype=np.int64)
    ctx = np.zeros(VAD_CONTEXT, dtype=np.float32)
    mask = []
    for i in range(len(a) // VAD_WINDOW):
        frame = a[i * VAD_WINDOW : (i + 1) * VAD_WINDOW].astype(np.float32)
        out = vad.run(
            None, {"input": np.concatenate([ctx, frame])[None, :], "state": state, "sr": sr}
        )
        state = out[1]
        ctx = frame[-VAD_CONTEXT:]
        mask.append(out[0].item() > 0.5)
    return mask


def internal_gaps(mask):
    on = [i for i, m in enumerate(mask) if m]
    if not on:
        return [], 0
    first, last = on[0], on[-1]
    gaps = []
    run = 0
    for i in range(first, last + 1):
        if mask[i]:
            if run:
                gaps.append((i - run, run))
            run = 0
        else:
            run += 1
    return gaps, (last + 1 - first)


def main():
    paths = sorted(glob.glob(os.environ["EOT_PE_SHARDS"]))
    assert paths, "EOT_PE_SHARDS matched nothing"
    st = ort.InferenceSession(
        os.environ["EOT_PE_SMART_TURN"], providers=["CPUExecutionProvider"]
    )
    st_in = st.get_inputs()[0].name
    vad = ort.InferenceSession(os.environ["EOT_PE_SILERO"], providers=["CPUExecutionProvider"])
    silences = sorted(
        int(s) for s in os.environ.get("EOT_PE_SILENCES_MS", "350,200").split(",")
    )
    lang_filter = os.environ.get("EOT_PE_LANG", "eng")
    max_rows = int(os.environ.get("EOT_PE_MAX_ROWS", "2500"))
    lo, hi = silences[0], silences[-1]

    def score(clip):
        return float(st.run(None, {st_in: ab.log_mel(ab.prepare(clip))[None]})[0].reshape(-1)[0])

    stats = {ms: dict(consult=0, cut=0) for ms in silences}
    band = dict(consult=0, cut=0)
    band_ps = []
    used = 0
    span_minutes = 0.0
    gap_hist = dict()
    for arec, lang in iter_rows(paths, ["audio", "language"]):
        if used >= max_rows:
            break
        if lang != lang_filter:
            continue
        try:
            a = ab.decode(arec["bytes"])
        except Exception:
            continue
        mask = vad_speech_mask(vad, a)
        gaps, span_windows = internal_gaps(mask)
        if span_windows == 0:
            continue
        used += 1
        span_minutes += span_windows * WINDOW_MS / 60000.0
        for start_w, run_w in gaps:
            gap_ms = run_w * WINDOW_MS
            bucket = min(gap_ms // 100 * 100, 1000)
            gap_hist[bucket] = gap_hist.get(bucket, 0) + 1
            for ms in silences:
                if gap_ms < ms:
                    continue
                consult_end = start_w * VAD_WINDOW + ms * SR // 1000
                p = score(a[:consult_end])
                stats[ms]["consult"] += 1
                stats[ms]["cut"] += p > 0.5
                if ms == lo and lo <= gap_ms < hi:
                    band["consult"] += 1
                    band["cut"] += p > 0.5
                    band_ps.append(p)

    print(
        f"samples used={used} lang={lang_filter} in-span audio={span_minutes:.1f} min "
        f"internal-gap histogram (ms bucket: n): "
        + " ".join(f"{k}:{v}" for k, v in sorted(gap_hist.items()))
    )
    for ms in silences:
        s = stats[ms]
        rate_c = s["consult"] / max(span_minutes, 1e-9)
        rate_x = s["cut"] / max(span_minutes, 1e-9)
        frac = s["cut"] / max(s["consult"], 1) * 100
        print(
            f"silence-end {ms:>3}ms: {s['consult']} mid-utterance consultations "
            f"({rate_c:.2f}/min), would-cut {s['cut']} ({frac:.1f}% of consultations, "
            f"{rate_x:.3f}/min)"
        )
    frac = band["cut"] / max(band["consult"], 1) * 100
    print(
        f"NEW band [{lo},{hi})ms only: {band['consult']} consultations, "
        f"would-cut {band['cut']} ({frac:.1f}%) -- this is the marginal false-cut "
        f"exposure the {hi}->{lo}ms flip adds"
    )
    if band_ps:
        ps = np.array(band_ps)
        sweep = " ".join(
            f"p>{t}: {int((ps > t).sum())} ({(ps > t).mean() * 100:.1f}%)"
            for t in (0.5, 0.7, 0.9, 0.95, 0.99)
        )
        print(
            f"eager-threshold sweep over the new band (cut only when smart-turn exceeds t "
            f"at {lo}ms, else wait for {hi}ms): {sweep}"
        )


if __name__ == "__main__":
    main()
