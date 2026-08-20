# 005 — manual response.create with instructions override

Client-driven `response.create` (no preceding user speech) with an
explicit `instructions` override on the response body, exercising the
C1 dispatch path.

Pins (RFC v3):
- §8.1 Creation (`response.create.instructions` overrides session-level
  instructions for this response only).
- §10.1 Inbound events: `response.create` dispatch parity — not a no-op;
  it transitions RespNone → RespCreated with the supplied id/item_id.
- W1 (paired response.created / response.done).
- W6 (no response.* after response.done).

The expected trace echoes `instructions` on the synthesized
`response.created` event so the override is visible in the wire log.
This conformance fixture does not exercise `modalities` because the
synthesizer keeps audio-bytes shape stable across all scenarios.
