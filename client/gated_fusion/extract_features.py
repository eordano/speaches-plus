#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "huggingface_hub>=0.27",
#   "soundfile>=0.13",
#   "numpy>=2.0",
#   "onnxruntime>=1.20",
#   "faster-whisper>=1.1",
#   "pyarrow>=17",
# ]
# ///
"""extract_features -- pull a real-data sample from
pipecat-ai/smart-turn-data-v3 and produce a JSONL of feature rows for
client/gated_fusion/train.py to fit a logistic regression on.

Per-row schema:

    {
      "id": "...",                       # row id from the dataset
      "label": 0 | 1,                    # endpoint_bool (1=user finished)
      "language": "eng" | "spa" | ...,
      "midfiller": bool,
      "endfiller": bool,
      "synthetic": bool,
      "source": "...",                   # `dataset` field from upstream
      "transcript": "...",                # whisper output on the clip
      "p_text_heuristic": float,         # heuristic on `transcript`
      "p_audio_smartturn": float,        # smart-turn-v3 output
      "audio_ms": int,
      "partial_chars": int,
      "ends_strong_terminator": bool,
      "ends_soft_terminator": bool,
      "last_word_continuation": bool
    }

Why these features: they map 1:1 to GatedFusionFeatures + the head
probabilities expected by combine_fusion_gated, so a regression on them
yields weights that drop straight into DEFAULT_GATED_FUSION_WEIGHTS on
all three implementations (Python, Go, Rust).
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eou_lib import (
    SmartTurn,
    extract_gated_fusion_features,
    heuristic_score,
)
from eou_lib.heuristic import (
    ends_strong_terminator,
    ends_soft_terminator,
    last_word_is_continuation,
)

logger = logging.getLogger("extract")

DATASET_REPO = "pipecat-ai/smart-turn-data-v3-test"
SHARD_FILES = [f"data/train-{i:05d}-of-00008.parquet" for i in range(8)]

ISO_3TO2 = {
    "eng": "en", "spa": "es", "fra": "fr", "deu": "de", "ita": "it",
    "por": "pt", "nld": "nl", "rus": "ru", "ukr": "uk", "pol": "pl",
    "ces": "cs", "tur": "tr", "ara": "ar", "hin": "hi", "ben": "bn",
    "vie": "vi", "ind": "id", "zho": "zh", "kor": "ko", "jpn": "ja",
    "fin": "fi", "swe": "sv", "nor": "no", "nob": "no", "nno": "no",
    "dan": "da", "ell": "el", "heb": "he", "hun": "hu", "ron": "ro",
    "tha": "th", "fas": "fa", "urd": "ur", "tam": "ta", "tel": "te",
    "mar": "mr", "mal": "ml", "kan": "kn", "guj": "gu",
}

SAMPLE_RATE = 16000

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smart-turn",
                   default=str(Path(__file__).resolve().parents[2] / "rust/models/smart-turn-v3.onnx"))
    p.add_argument("--whisper-model", default="deepdml/faster-whisper-large-v3-turbo-ct2")
    p.add_argument("--max-rows", type=int, default=2000,
                   help="cap (random sample) of dataset rows; -1 for all")
    p.add_argument("--shard", type=int, default=0,
                   help="which of the 8 test parquet shards to use (0..7)")
    p.add_argument("--out", default="/tmp/gated-fusion-real-features.jsonl")
    p.add_argument("--language-filter", default="eng",
                   help="comma-separated ISO-639-3 language codes to keep "
                        "(matches the dataset's `language` field); '*' for all")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    from huggingface_hub import hf_hub_download
    logger.info(f"downloading {DATASET_REPO} shard {args.shard} ...")
    parquet_path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset",
                                    filename=SHARD_FILES[args.shard])
    logger.info(f"  cached at {parquet_path}")

    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    logger.info(f"  rows in shard: {table.num_rows}, columns: {table.column_names}")

    indices = list(range(table.num_rows))
    if args.language_filter and args.language_filter != "*":
        wanted = set(args.language_filter.split(","))
        languages = table.column("language").to_pylist()
        indices = [i for i in indices if languages[i] in wanted]
        logger.info(f"  after language filter ({args.language_filter}): {len(indices)}")

    rng = np.random.default_rng(42)
    if args.max_rows > 0 and len(indices) > args.max_rows:
        indices = list(rng.choice(indices, size=args.max_rows, replace=False))
        logger.info(f"  sampled to {len(indices)} rows")

    smart = SmartTurn.load(args.smart_turn)
    logger.info("smart-turn ready")

    from faster_whisper import WhisperModel
    whisper = WhisperModel(args.whisper_model, device="cpu", compute_type="int8")
    logger.info("whisper ready")

    out_f = open(args.out, "w")
    written = 0
    skipped = 0
    t0 = time.monotonic()

    audio_col = table.column("audio").to_pylist()
    id_col = table.column("id").to_pylist()
    lang_col = table.column("language").to_pylist()
    endpoint_col = table.column("endpoint_bool").to_pylist()
    midfiller_col = table.column("midfiller").to_pylist()
    endfiller_col = table.column("endfiller").to_pylist()
    synthetic_col = table.column("synthetic").to_pylist()
    dataset_col = table.column("dataset").to_pylist()

    for i in indices:
        row_id = id_col[i]
        try:
            entry = audio_col[i]
            buf = io.BytesIO(entry["bytes"])
            audio_arr, sr = sf.read(buf, dtype="float32", always_2d=False)
            if audio_arr.ndim > 1:
                audio_arr = audio_arr[:, 0]
            if sr != SAMPLE_RATE:
                duration = len(audio_arr) / sr
                n_out = int(duration * SAMPLE_RATE)
                x_in = np.arange(len(audio_arr))
                x_out = np.linspace(0, len(audio_arr) - 1, n_out)
                audio_arr = np.interp(x_out, x_in, audio_arr).astype(np.float32)

            audio_ms = int(len(audio_arr) * 1000 / SAMPLE_RATE)
            if audio_ms < 200:
                skipped += 1
                continue

            p_audio = smart.score(audio_arr)
            if not np.isfinite(p_audio):
                skipped += 1
                continue

            iso3 = lang_col[i] or ""
            iso2 = ISO_3TO2.get(iso3, None)
            segments, _info = whisper.transcribe(audio_arr, language=iso2,
                                                  beam_size=1, vad_filter=False)
            transcript = " ".join(seg.text.strip() for seg in segments).strip()

            p_text = heuristic_score(transcript)
            feat = extract_gated_fusion_features(transcript, audio_ms)

            row_out = {
                "id": row_id,
                "label": int(bool(endpoint_col[i])),
                "language": lang_col[i] or "",
                "midfiller": bool(midfiller_col[i]),
                "endfiller": bool(endfiller_col[i]),
                "synthetic": bool(synthetic_col[i]),
                "source": dataset_col[i] or "",
                "transcript": transcript,
                "p_text_heuristic": p_text,
                "p_audio_smartturn": float(p_audio),
                "audio_ms": audio_ms,
                "partial_chars": feat.partial_chars,
                "ends_strong_terminator": ends_strong_terminator(transcript),
                "ends_soft_terminator": ends_soft_terminator(transcript),
                "last_word_continuation": last_word_is_continuation(transcript),
            }
            out_f.write(json.dumps(row_out) + "\n")
            written += 1

            if written % 50 == 0:
                elapsed = time.monotonic() - t0
                rate = written / max(elapsed, 1e-3)
                logger.info(f"  [{written}/{len(indices)}] {rate:.1f} rows/s  "
                            f"label={row_out['label']} pT={p_text:.2f} "
                            f"pA={p_audio:.2f}  last={transcript!r}")

        except Exception as exc:
            logger.warning(f"row {i} ({row_id}) skipped: {exc}")
            skipped += 1

    out_f.close()
    logger.info(f"done: wrote {written} rows, skipped {skipped}, "
                f"in {time.monotonic() - t0:.1f}s")
    logger.info(f"output: {args.out}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
