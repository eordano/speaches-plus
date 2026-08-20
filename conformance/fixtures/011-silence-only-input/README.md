# 011 — silence-only input (RFC v3 §15.3)

No voiced audio for the session duration. The phase machine must not
emit any commit, conversation item, or response — `session.created`
is the only wire event for the session lifetime.

Pins:
- v3 §3.1 (no buffer rotation without a voiced span)
- v3 §6 (no EOU dispatch without `speech_stopped`)
- v3 §7.1 (commit requires a non-empty buffer)
- v3 §15.3 (silence-only-input fixture)

The wire trace is exactly one event: `session.created`. Implementations
that synthesize spurious `input_audio_buffer.committed` or
`response.created` while VAD is silent fail this fixture immediately.
