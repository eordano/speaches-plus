# 030-voice-clone-custom-from-3sec

Voice profile created from a 3-second reference WAV.

## Input artifact

`reference_440hz_3s_16k_mono.wav` — a 440 Hz sine wave, 3 seconds, mono,
16 kHz, 16-bit PCM. Regenerate with `./generate_reference_wav.py` (it's
deterministic; the file is checked in for offline use).

## What it asserts

1. POST `/v1/voice-profiles` with `file=<the WAV>` returns 201, the profile
   has `embedding_dim == 1024` (the ECAPA-TDNN `enc_dim` of the Qwen3-TTS
   Base speaker encoder) and `schema_version == 2`. The embedding is the
   raw encoder output — the reference implementation does not L2-normalize,
   so no norm assertion is made.
2. GET `/v1/voice-profiles/test-custom-from-3sec` returns the profile with
   `embedding_state == "encoded"`.
3. DELETE cleans up.

## Checkpoint requirement

The speaker encoder is loaded from the TTS checkpoint itself: the
Qwen3-TTS-12Hz `*-Base` checkpoints carry a 76-tensor `speaker_encoder.*`
block (ECAPA-TDNN, 128-mel @ 24 kHz → 1024-d x-vector) in
`model.safetensors`. CustomVoice/VoiceDesign checkpoints do NOT — with one
of those loaded via `NV_TTS_TALKER_DIR`, enrollment returns 503 with code
`no_speaker_encoder` (there is no placeholder-zeros mode), and this fixture
is skipped. Synthesis with an enrolled profile voice likewise requires the
Base checkpoint; on CustomVoice it returns 400 `voice_profile_unsupported`.

## Spec pins

- §11.4 — 030-voice-clone-* family
- `nv_tts::SpeakerEncoder`, `nv_tts::spk_mel`, `nv_tts::VoiceProfileStore`
- `rust/src/oapi/voice_profiles.rs::handle_create`
