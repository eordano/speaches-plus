# 004 — single-word backchannel below min_speech_for_response_ms

Short utterance (e.g. "uh-huh", "yeah") whose audio span is **above**
`min_speech_ms` (default 100, §17.4) but **below**
`min_speech_for_response_ms` (default 600, §17.4). Per RFC v3 §7.1 the
commit MUST succeed (the buffer is large enough to commit), and per §7.2
step 4 the auto-response MUST be suppressed.

Pins:
- v3 §7.1 (commit accepted because audio_ms ≥ min_speech_ms)
- v3 §7.2 (auto-response suppressed because audio_ms < min_speech_for_response_ms)
- v3 §17.4 (`min_speech_ms` default 100, `min_speech_for_response_ms` default 600)

The wire trace shows the full commit path (`input_audio_buffer.committed`
+ `conversation.item.added` + `input_audio_transcription.completed`)
followed by NO `response.created` / `response.done`. Production
additionally records `inspector.backchannel_suppressed` which is below
the wire layer and not represented in the canonical trace.
