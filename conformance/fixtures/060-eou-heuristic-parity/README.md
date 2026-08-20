# 060-eou-heuristic-parity

Cross-implementation parity corpus for the rule-based text-EOU heuristic.
Rust is canonical; Go and Python must return identical scores for every case
in `fixture.json::cases`.

Declarative fixture — there is no endpoint. The cases are read directly by the
per-language tests, so `run_endpoint_fixture.py` validates the manifest shape
and reports it as skipped.

## Consumed by

- `go/internal/eou/parity_corpus_test.go`
- the Rust and Python EOU heuristic tests named in
  `fixture.json::canonical_impl` / `also_implemented_by`

## Scope

Scores are in lockstep across implementations; branch *selection* is not. Go
carries per-language hesitation/continuation tables and CJK terminators that
Rust and Python lack, and the three HESITATIONS word lists still differ. Every
case in the corpus is chosen to avoid those divergences, so this fixture gates
the numeric contract only — see `fixture.json::notes`.
