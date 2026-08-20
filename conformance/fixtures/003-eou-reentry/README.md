# 003 — EOU re-entry mid commit_timer

User pauses long enough to put the buffer in BufStopped (commit_timer
armed) but resumes speaking before the timer fires. The phase machine
(state.go::onVadSpeechStart while BufStopped) rolls back to BufVoiced
WITHOUT advancing the item_id or audio_start_ms.

Pins (RFC v3):
- §6.6 Trigger and cancellation (commit_timer cancelled, item_id and audio_start_ms preserved)
- §2.2 VadPhase / §2.3 RespPhase (CommitTimer is a structural slot on VadStopped; re-entry
  cancels by replacing the variant)
- W3 (single committed for the merged turn)

Expected: exactly one input_audio_buffer.speech_started for the merged
turn (the second vad_speech_start does NOT emit speech_started because
the phase reuses the existing item_id and audio_start_ms). The trace
contains two speech_stopped events — one per VAD pause — because the
test driver records each VAD edge as it crosses speaking → stopped.
The committed/conversation.item.added pair fires once at end-of-turn.
