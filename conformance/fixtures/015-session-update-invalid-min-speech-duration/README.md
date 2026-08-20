# 015 — session.update rejects out-of-range min_speech_duration_ms

Client sends `session.update` with `turn_detection.min_speech_duration_ms`
above the 60_000 ms cap. The sweep-based VAD filter drops speech segments
shorter than this threshold (port of Python speaches' Silero post-filter);
absurdly large values would suppress all real speech, so the parser
rejects them at the wire boundary.

Pins:
- §11.2 Updates (rejected updates do not mutate session state)
- §10.5 Error envelope
- §11.4 Reconciliation
- §17.2 `turn_detection.min_speech_duration_ms ∈ [0, 60_000]`
