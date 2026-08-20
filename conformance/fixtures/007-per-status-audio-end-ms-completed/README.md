# 007 — per-status audio_end_ms (completed) (RFC v3 §8.5 / §15.7 / D.4)

Asserts the v3 §8.5 / W4 contract: a clean `completed` response MUST
carry `audio_end_ms` on the wire.

Doubles as the regression gate for the v2 Go bug where `audio_end_ms`
was emitted only on `response.done(status ∈ {cancelled, incomplete})`
and the implementation's own `AssertTraceInvariants` mirrored the
bug — both wrong in the same direction (RFC v3 App. D.4).

Pins:
- v3 §8.5 (audio_end_ms required on ALL response.done statuses)
- v3 §15.2 (W4 invariant)
- v3 App. D.4 (Go v2 mirror-bug postmortem)
