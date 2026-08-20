# 013 — session.update rejects out-of-range no_speech_prob_threshold

Client sends `session.update` with `no_speech_prob_threshold > 1.0`
(or `< 0.0`). Per RFC v3 §11.2 (input validation) + §17.4 (gate
parameter bounds), the server MUST:
1. Emit an `error` event with code `session_update_invalid` describing
   the failed field.
2. Emit a `session.updated` echo carrying the SERVER's authoritative,
   unchanged config (the previously-active gate config is preserved).

Pins:
- §11.2 Updates (rejected updates do not mutate session state)
- §10.5 Error envelope (`session_update_invalid`)
- §11.4 Reconciliation (echo even on reject)
- §17.4 `no_speech_prob_threshold ∈ [0, 1]`
