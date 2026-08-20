#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.28"]
# ///
"""HTTP shape-parity test for `/v1/audio/transcriptions?response_format=diarized_json`.

Both the Rust and Go servers expose this endpoint and are expected to agree
on the response JSON shape per `docs/book/02.1-model-compat-matrix.md` § "diarized_json response shape":

    {
      "text": str,
      "avg_logprob": float | null,
      "no_speech_prob": float | null,
      "segments": [
        {
          "type": "transcript.text.segment",        # OpenAI-required
          "id":   "seg_NNN",                         # OpenAI-required
          "speaker": "SPEAKER_NN" | null,
          "start": float,                            # seconds
          "end":   float,                            # seconds
          "duration": float,                         # seconds
          "text": str,
          "avg_logprob":   float | null,
          "no_speech_prob":float | null,
          "confidence":    float | null,
        }, ...
      ]
    }

Usage:
    BENCH_TARGET=http://127.0.0.1:18801 python3 client/test_diarized_json.py     # Rust
    BENCH_TARGET=http://127.0.0.1:18802 python3 client/test_diarized_json.py     # Go (--stt-backend ct2)

Exit code: 0 on pass, 1 on shape mismatch, 2 on usage error.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import httpx

REQUIRED_TOP_KEYS = {"text", "avg_logprob", "no_speech_prob", "segments"}
REQUIRED_SEG_KEYS = {
    "type",
    "id",
    "speaker",
    "start",
    "end",
    "duration",
    "text",
    "avg_logprob",
    "no_speech_prob",
    "confidence",
}
OPENAI_TYPE = "transcript.text.segment"

def fail(msg: str, code: int = 1) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)
    sys.exit(code)

def check_shape(body: dict) -> list[str]:
    """Return a list of contract violations; empty means pass."""
    errors: list[str] = []
    missing_top = REQUIRED_TOP_KEYS - body.keys()
    if missing_top:
        errors.append(f"top-level missing keys: {sorted(missing_top)}")
    if not isinstance(body.get("text"), str):
        errors.append(f"top.text must be str, got {type(body.get('text')).__name__}")
    if body.get("avg_logprob") is not None and not isinstance(
        body["avg_logprob"], (int, float)
    ):
        errors.append("top.avg_logprob must be number or null")
    if body.get("no_speech_prob") is not None and not isinstance(
        body["no_speech_prob"], (int, float)
    ):
        errors.append("top.no_speech_prob must be number or null")
    segs = body.get("segments")
    if not isinstance(segs, list):
        errors.append(f"top.segments must be list, got {type(segs).__name__}")
        return errors

    for i, seg in enumerate(segs):
        if not isinstance(seg, dict):
            errors.append(f"segments[{i}] not an object")
            continue
        missing = REQUIRED_SEG_KEYS - seg.keys()
        if missing:
            errors.append(f"segments[{i}] missing keys: {sorted(missing)}")
            continue
        if seg["type"] != OPENAI_TYPE:
            errors.append(
                f"segments[{i}].type must be {OPENAI_TYPE!r}, got {seg['type']!r}"
            )
        if not isinstance(seg["id"], str) or not seg["id"].startswith("seg_"):
            errors.append(f"segments[{i}].id must start with 'seg_', got {seg['id']!r}")
        if seg["speaker"] is not None and not (
            isinstance(seg["speaker"], str) and seg["speaker"].startswith("SPEAKER_")
        ):
            errors.append(
                f"segments[{i}].speaker must be null or 'SPEAKER_NN', got {seg['speaker']!r}"
            )
        for k in ("start", "end", "duration"):
            if not isinstance(seg[k], (int, float)):
                errors.append(
                    f"segments[{i}].{k} must be number, got {type(seg[k]).__name__}"
                )
        if not isinstance(seg["text"], str):
            errors.append(
                f"segments[{i}].text must be str, got {type(seg['text']).__name__}"
            )
        if seg["end"] < seg["start"]:
            errors.append(
                f"segments[{i}] end ({seg['end']}) < start ({seg['start']})"
            )
    return errors

def find_fixture() -> Path:
    """Find any cached WAV fixture; gen-fixtures.py populates /tmp/sp-fixtures."""
    for p in (
        Path("/tmp/sp-fixtures/p01.wav"),
        Path("client/fixtures/ref_quick_brown_fox.wav"),
    ):
        if p.exists():
            return p
    raise SystemExit("[FAIL] no fixture wav available. Run client/perf/gen-fixtures.py first.")

def main() -> int:
    target = os.environ.get("BENCH_TARGET", "http://127.0.0.1:18801")
    backend = os.environ.get("BENCH_MODEL", "")
    fixture = find_fixture()

    files = {"file": (fixture.name, fixture.read_bytes(), "audio/wav")}
    data = {"response_format": "diarized_json"}
    if backend:
        data["model"] = backend

    print(f"POST {target}/v1/audio/transcriptions  fixture={fixture.name}  backend={backend or '(default)'}")
    try:
        r = httpx.post(
            f"{target}/v1/audio/transcriptions",
            files=files,
            data=data,
            timeout=180.0,
        )
    except Exception as e:
        fail(f"request failed: {e}")

    if r.status_code != 200:
        fail(f"HTTP {r.status_code}: {r.text[:500]}")

    try:
        body = r.json()
    except json.JSONDecodeError as e:
        fail(f"response not JSON: {e}; body={r.text[:200]!r}")

    errors = check_shape(body)
    if errors:
        for e in errors:
            print(f"  [SHAPE] {e}", file=sys.stderr)
        fail(f"{len(errors)} shape violation(s)")

    n_segments = len(body["segments"])
    speakers = sorted(
        {s["speaker"] for s in body["segments"] if s["speaker"] is not None}
    )
    print(
        f"[PASS] shape ok  text={body['text'][:50]!r}  segments={n_segments}  "
        f"speakers={speakers or '(none)'}"
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
