# 001 — clean utterance

Single user turn followed by a complete assistant response, no barge-in.

Pins (RFC v3):
- §6.6 Trigger and cancellation (commit fires after EOU silence with stable item_id / audio_end_ms)
- §7.2 Transcription (transcription_complete → input_audio_transcription.completed)
- §8.1 Creation, §8.2 Streaming, §8.4 Terminal (response.created → audio.delta(*) → audio.done → response.done(completed))
- W1 (paired response.created / response.done)
- W3 (committed appears between speech_stopped and conversation.item.added)
- W6 (no response.* after response.done)

The wire trace canonicalizes IDs to counter-style (sess_1, item_1=user_a,
resp_1=resp_a) per RFC v3 §15.4 Trace canonicalization.
