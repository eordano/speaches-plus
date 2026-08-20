# 014 — session.update rejects out-of-range neg_threshold

Client sends `session.update` with `turn_detection.neg_threshold` outside
`[0, 1]`. Same atomic-reject + reflective-echo pattern as fixtures 006/013.

The `neg_threshold` field controls the hysteresis floor in the sweep-based
VAD detector (the "leave speech" gate, distinct from `threshold` which is
the "enter speech" gate). `null` means "auto" (computed as
`max(threshold - 0.15, 0.01)` per Python speaches parity).

Pins:
- §11.2 Updates (rejected updates do not mutate session state)
- §10.5 Error envelope
- §11.4 Reconciliation
- §17.2 `turn_detection.neg_threshold ∈ [0, 1] | null` (null = auto)
