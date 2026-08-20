# 012 — per-status audio_end_ms (failed) (RFC v3 §8.5 / §15.7)

Closes the per-status W4 set begun by fixtures 007 / 008 / 009. A
response that fails mid-stream (LLM upstream error, TTS error,
client_too_slow) MUST emit `response.done(status=failed)` carrying
`audio_end_ms` equal to the pacer snapshot at failure time.

Pins:
- v3 §8.5 (audio_end_ms required on ALL response.done statuses,
  including `failed`)
- v3 §10.5 (`status_details.reason` carries the underlying classification
  — `llm_error` here)
- v3 §15.2 (W4 invariant)

The §15.7 regression assertion `TestAssertTraceInvariants_W4PerStatus_*`
already exercises the trace-level shape; this fixture pins the wire-side
emit path so a future bug in the response-failure builder shows up
loud rather than only failing the table-driven trace assertion.
