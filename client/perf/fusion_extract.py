"""Extract gated-fusion training features from one smart-turn-data shard.

Replicates rust/src/eou exactly: HeuristicEouModel::score_text for p_text,
extract_gated_fusion_features for the boolean/length features, smart-turn
v3.2-cpu int8 (the deployed file) for p_audio, faster-whisper large-v3-turbo
CT2 fp16 CUDA (the deployed STT) for transcripts.

Usage: fusion_extract.py {train|test} <shard_idx> <n_shards> <out_csv>
"""

import csv
import io
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np
import onnxruntime as ort
import pyarrow.parquet as pq
import soundfile as sf
from faster_whisper import WhisperModel

import eot_silence_ab as F

SMART_TURN = "/tmp/models/smart-turn-v3.2-cpu.onnx"
WHISPER_CT2 = "/tmp/models/faster-whisper-large-v3-turbo-ct2"
SR = 16000

LANG = {"eng": "en", "spa": "es", "deu": "de", "kor": "ko", "ita": "it", "ukr": "uk",
        "dan": "da", "fra": "fr", "por": "pt", "ind": "id", "zho": "zh", "jpn": "ja",
        "fin": "fi", "pol": "pl", "ara": "ar", "tur": "tr", "rus": "ru", "nld": "nl",
        "vie": "vi", "ben": "bn", "nor": "no", "hin": "hi", "mar": "mr"}

HESITATION = {"uh", "um", "uhh", "umm", "er", "erm", "hmm", "like", "so"}
CONTINUATIONS = {"and", "or", "but", "with", "the", "a", "an", "to", "of", "for", "is",
                 "was", "are", "were", "because", "since", "if", "when", "while", "as",
                 "than", "that", "which", "who", "whom", "whose"}

import re

def last_word(s):
    t = s.rstrip(" \t\n\r.!?,;:")
    parts = [p for p in re.split(r"[^\w'\-]", t) if p]
    return parts[-1] if parts else ""

def score_text(s):
    s = s.strip()
    if not s:
        return 0.1
    last_char = s[-1]
    lw = last_word(s).lower()
    if last_char in ".!?":
        return 0.95
    if last_char in ",;:-":
        return 0.25
    if not lw:
        return 0.3
    if lw in HESITATION:
        return 0.15
    if lw in CONTINUATIONS:
        return 0.2
    return 0.6

def text_features(partial):
    t = partial.strip()
    strong = soft = False
    if t:
        if t[-1] in ".!?":
            strong = True
        elif t[-1] in ",;:-":
            soft = True
    import re
    m = list(re.finditer(r"[\w'\-]+", t))
    cont = bool(m) and m[-1].group(0).lower() in CONTINUATIONS
    return len(t), strong, soft, cont

def main():
    split, idx, n, out_csv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    repo = f"pipecat-ai/smart-turn-data-v3.2-{split}"
    fname = f"train-{idx:05d}-of-{n:05d}.parquet"
    import os
    import urllib.request
    local = f"{out_csv}.parquet"
    if not os.path.exists(local):
        urllib.request.urlretrieve(
            f"https://huggingface.co/datasets/{repo}/resolve/main/data/{fname}", local
        )

    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    st = ort.InferenceSession(SMART_TURN, so, providers=["CPUExecutionProvider"])
    st_in = st.get_inputs()[0].name
    wh = WhisperModel(WHISPER_CT2, device="cuda", compute_type="float16")

    tbl = pq.read_table(local, columns=["audio", "endpoint_bool", "language", "midfiller",
                                        "endfiller", "synthetic", "dataset"])
    rows_out = []
    for arec, label, lang, midf, endf, syn, ds in zip(
        tbl.column("audio").to_pylist(), tbl.column("endpoint_bool").to_pylist(),
        tbl.column("language").to_pylist(), tbl.column("midfiller").to_pylist(),
        tbl.column("endfiller").to_pylist(), tbl.column("synthetic").to_pylist(),
        tbl.column("dataset").to_pylist(),
    ):
        try:
            a, rate = sf.read(io.BytesIO(arec["bytes"]), dtype="float32")
            if a.ndim > 1:
                a = a.mean(axis=1)
            if rate != SR:
                x = np.linspace(0, len(a) - 1, int(len(a) * SR / rate))
                a = np.interp(x, np.arange(len(a)), a).astype(np.float32)
        except Exception:
            continue
        p_audio = float(st.run(None, {st_in: F.log_mel(F.prepare(a))[None]})[0].reshape(-1)[0])
        try:
            segs, _ = wh.transcribe(a, language=LANG.get(lang), beam_size=1,
                                    vad_filter=False, condition_on_previous_text=False)
            text = " ".join(s.text for s in segs).strip()
        except Exception:
            text = ""
        p_text = score_text(text)
        chars, strong, soft_t, cont = text_features(text)
        audio_ms = int(len(a) * 1000 / SR)
        rows_out.append([int(bool(label)), lang, ds, int(bool(midf)), int(bool(endf)),
                         int(bool(syn)), audio_ms, f"{p_audio:.6f}", f"{p_text:.4f}",
                         chars, int(strong), int(soft_t), int(cont), text])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "language", "dataset", "midfiller", "endfiller", "synthetic",
                    "audio_ms", "p_audio", "p_text", "partial_chars", "strong", "soft",
                    "continuation", "text"])
        w.writerows(rows_out)
    import os
    os.unlink(local)
    print(f"{split}/{idx}: {len(rows_out)} rows -> {out_csv}")

if __name__ == "__main__":
    main()
