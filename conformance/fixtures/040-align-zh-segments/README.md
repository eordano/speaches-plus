# 040-align-zh-segments

Mandarin (`language=zh`) diarized-segment alignment via
`/v1/audio/diarization`.

## Input artifact

`audio.wav` — 3 s, 16 kHz mono, 2 sine bursts (392 / 466.16 Hz). The
audio is placeholder; we only assert JSON skeleton shape.

## What it asserts

- Route returns 200 with `Content-Type: application/json`.
- Body has top-level `segments` array.
- 2 segments (`±1` tolerance), each with `id`, `start`, `end`, `text`,
  in ascending `start` order, with per-segment `start <= end`.
- `language == "zh"` round-trips (smoke test for non-Latin script handling).

## Placeholder mode

Without a real STT/aligner, this route may not return the diarized
shape at all; the fixture is `skip_when_no_model: true`. Once wired with
a Mandarin-capable model the per-segment text becomes a canonical assertion
under T5 tolerance.

## Spec pins

- §11.4 — 040-align-* family
- `/v1/audio/diarization` (see `rust/src/main.rs` route registration)
- `nv_aligner::pipeline`
