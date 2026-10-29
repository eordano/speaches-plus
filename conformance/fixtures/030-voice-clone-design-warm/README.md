# 030-voice-clone-design-warm

A voice designed from a description, then cloned:

1. POST `/v1/audio/speech` to `qwen3-tts-voicedesign` with the description in
   `instructions` and the draw in `voice` (a decimal seed; LiteLLM forwards
   `voice` and drops `seed`).
2. POST `/v1/voice-profiles` with that sample (`file_from_step: 1`).
3. POST `/v1/audio/speech` to the 0.6B Base with `voice=test-design-warm`,
   reading a longer paragraph.
4. DELETE the profile.

`contrast` adds a deep male and a high female description read with the same
draw.

## What it asserts

The consumer is
`rust/src/oapi/audio_speech_nvtts.rs::tests::real_voice_design_follows_the_description_repeats_by_seed_and_clones_onto_base`
(`#[ignore]`, `NV_TTS_VOICE_DESIGN_TEST=1`, `NV_TTS_VOICE_DESIGN_DIR`,
`NV_TTS_STT_URL`). It drives the engine in process (enrollment uses the same
decode and `extract_embedding` as the route) and checks `gates`:

- every sample round-trips its text through STT within `max_word_error_rate`;
- the deep male description's median F0 is at most `low_max_median_f0_hz`, the
  high female one's at least `high_min_median_f0_hz`;
- in the Base checkpoint's speaker-encoder space the two male descriptions sit
  closer than the deep male and the high female by `contrast_min_ecapa_margin`,
  and the clone sits closer to the designed sample than the Base default voice
  does by `clone_min_ecapa_margin_over_default_voice`;
- the clone's median F0 stays within `clone_max_median_f0_deviation` of the
  designed sample's;
- the same description, text and draw give bit-identical audio;
- the text and instruction-prompt token ids equal `python_reference`, taken from
  the Python reference's processor.

The speaker-encoder cosines of this checkpoint are compressed near 1, so the
gates are margins between pairs, not absolute similarities.
