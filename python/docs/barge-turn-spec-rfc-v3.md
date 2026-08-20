# Moved -- see `docs/book/07.1-barge-turn-spec-rfc-v3.md`

The normative realtime session spec (RFC v3) lives at the repo root:

> [`../../docs/book/07.1-barge-turn-spec-rfc-v3.md`](../../docs/book/07.1-barge-turn-spec-rfc-v3.md)

**There is exactly one copy, and this is not it.** The file that used to sit
here was a 2247-line subset of the canonical 2558-line document. Among other
gaps it was missing **§0.3 "Event-name evolution and aliases"** entirely, so a
reader following the Python-side copy would have implemented the wire protocol
without the alias table -- while `go/IMPLEMENTATION.md`, `rust/IMPLEMENTATION.md`
and `conformance/README.md` all track the canonical file and its §-numbering.

Two divergent copies of a normative spec is worse than one copy in the wrong
directory, so the copy was replaced with this pointer on 2026-08-07 rather than
resynchronised. Do not re-add a second copy; link to the canonical path.
