# 002 — barge-in mid-streaming

User speaks while the assistant is still streaming an in-flight response.
Pins (RFC v3):
- §9.1 Atomicity: barge-in cancels the active response, snapshot of
  played_ms is emitted as audio_end_ms in both response.done(cancelled)
  and conversation.item.assistant_truncated.
- §2.4 Invariants (I1): VadSpeaking incompatible with {Created, Streaming, Drain};
  the cancel happens within the same critical section as VadSpeaking arming.
- W4 (response.done(cancelled) carries audio_end_ms)
- W6 (no response.* after response.done for the cancelled id)
- W7 (assistant_truncated paired with cancelled response.done)

Note: this scenario synthesizes audio_end_ms = 0 because the replay loop
has not delivered any played_ms snapshot before the barge-in. A real run
would carry the played_ms captured under the lock at cancel time.
