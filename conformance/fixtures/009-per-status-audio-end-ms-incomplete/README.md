# 009 — per-status audio_end_ms (incomplete) (RFC v3 §8.5 / §15.7)

`response.done(status=incomplete)` (drain-cap expiry) MUST carry
`audio_end_ms` and `status_details.reason="drain_cap"`. v2 emitted
audio_end_ms only conditionally; v3 mandates the field unconditionally.

Pins:
- v3 §8.3 (drain_cap expiry → status=incomplete, reason=drain_cap)
- v3 §8.5 (audio_end_ms required on incomplete)
- v3 §15.2 (W4 invariant)
