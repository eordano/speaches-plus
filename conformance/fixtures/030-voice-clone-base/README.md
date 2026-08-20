# 030-voice-clone-base

Synthesize a known short utterance with the built-in `Base` Qwen3-TTS voice.

## What it asserts

- Structural: 200 OK, Content-Type `audio/wav`, valid RIFF/WAVE header,
  24 kHz sample rate (matches `NV_TTS_SAMPLE_RATE` in
  `rust/src/oapi/audio_speech.rs`), mono.
- Audio envelope: duration in `[1.5s, 5.5s]` for the pinned input + speed=1.0.
- Token-count proxy: the BPE+12Hz frame count falls in `[40, 220]` for the
  given input (the canonical `nv_tts::tokenizer` is loaded with the model;
  in placeholder mode this check is skipped).

## Why not byte-exact?

The Qwen3-TTS codec is permitted to introduce hardware/timing-dependent
LSB noise even when the symbol stream is deterministic. We therefore
compare via the audio envelope + token-count, not byte equality, per
PRD §10.1 (T5 tolerance).

## Spec pins

- §11.4 — 030-voice-clone-* family
- §11.5 — Qwen3-TTS Base RTF ≤ 0.15× (perf check is separate)
- `rust/src/oapi/audio_speech.rs::handle`
- `nv_tts::tokenizer`, `nv_tts::talker::TtsStream`
