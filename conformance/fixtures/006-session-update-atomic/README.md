# 006 — session.update atomic rejection with reflective echo

Client sends an invalid `session.update` (e.g. `turn_detection.threshold`
out of [0,1]). Per RFC v3 §11.2 + §10.5, the server MUST emit:
1. An `error` event with code `session_update_invalid` describing the
   failed field, AND
2. A `session.updated` echo carrying the SERVER's authoritative,
   unchanged config so the client can reconcile.

Pins (RFC v3):
- §11.2 Updates (rejected updates do not mutate session state)
- §10.5 Error envelope (`session_update_invalid` error code shape)
- §11.4 Reconciliation (the `session.updated` echo is emitted EVEN ON REJECT —
  clients rely on it to detect that their requested update was discarded).
