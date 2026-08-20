# 030-voice-clone-design-warm

VoiceDesign profile with `warm=1.0`. Multi-step fixture:

1. POST `/v1/voice-profiles` with `name=test-design-warm` and
   `design_params={"warm": 1.0}` (multipart form).
2. POST `/v1/audio/speech` using `voice="test-design-warm"`.
3. DELETE `/v1/voice-profiles/test-design-warm`.

## What it asserts

- Round-trip: the `design_params` we POST come back from the create response
  (and from a later GET, if added) byte-equal. This is the regression gate
  against silent drop of unknown design fields in
  `nv_tts::VoiceProfile::design_params`.
- Cleanup: DELETE returns 200 or 204.
- Audio envelope on the produced WAV (same envelope shape as
  `030-voice-clone-base`).

## Placeholder mode

`design_params` is opaque JSON today (see
`rust/src/oapi/voice_profiles.rs::handle_create`). The structural round-trip
assertion is meaningful even without a VoiceDesign-conditioned model. Once
the model treats `warm` as a real conditioning axis the audio envelope (and
optionally a per-axis A/B perceptual delta) becomes the canonical check.

## Spec pins

- §11.4 — 030-voice-clone-* family
- `nv_tts::VoiceProfile`, `nv_tts::VoiceProfileStore`
- `rust/src/oapi/voice_profiles.rs`
