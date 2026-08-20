# 008 — per-status audio_end_ms (cancelled) (RFC v3 §8.5 / §15.7)

Companion to 007: the cancelled-status side of W4. Barge-in cancels a
streaming response; `response.done(cancelled)` MUST carry `audio_end_ms`
equal to the pacer snapshot. Distinct from W7 because this fixture
asserts the field is structurally present even when `played_ms == 0`
(no `assistant_truncated` is emitted in that case, but `audio_end_ms`
still appears on `response.done`).

Pins:
- v3 §8.5 (audio_end_ms required on cancelled)
- v3 §9.3 (barge-in commitment carries played_ms snapshot)
- v3 §15.2 (W4 invariant)
