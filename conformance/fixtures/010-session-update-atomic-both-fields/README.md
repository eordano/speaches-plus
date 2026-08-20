# 010 — session.update atomic — valid+invalid pair (RFC v3 §11.2.1 / §15.7 / D.5)

Companion to 006 with a stricter shape: a session.update body
containing one valid field followed by one invalid field MUST be
rejected atomically. The server MUST emit
`error{code: session_update_invalid}` AND a `session.updated` echo
that reflects NO field change (the v2 Go and Rust bug — see App. D.5 —
wrote fields as it parsed and left partial state on validation error).

Pins:
- v3 §11.2.1 (parse-all → validate-all → commit-all)
- v3 App. D.5 (both v2 implementations had the bug)
