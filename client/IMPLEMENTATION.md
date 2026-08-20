# Client-side tooling -- implementation notes

Sources under `client/` are comment-free per the project-wide discipline
(`go/IMPLEMENTATION.md` §0); shebangs, PEP 723 inline-script metadata and
docstrings are language constructs, not comments, and stay. The non-obvious
WHYs live here.

## What's in this directory

Black-box test, fuzz, and training tools that drive a running speaches-plus
server (Go or Rust). None run inside the server process. Two flavours:

- **Black-box tests / probes** -- `test_e2e.py`, `test_e2e_full.py`,
  `test_diarized_json.py`, `check_openai_surface.py`,
  `check_speech_endpoint.py`, `eou_divergence_probe.py`, `eou_matrix.py`,
  `fuzz_e2e.py`, `fake_llm.py`, `eou_fixture.py`. Spawn the server, send
  requests, assert on wire-protocol invariants. Per-scenario coverage lives
  in `conformance/fixtures/`, **not** here -- both Go and Rust replay that
  corpus in-process. Add new scenarios as fixtures.
- **EOU model tooling** -- `eou_lib/`, `gated_fusion/`. Canonical Python
  ports of the production gates plus the training pipeline that emits the
  numbers compiled into the Go and Rust binaries.

## `fuzz_e2e.py` -- RFC v3 invariant checker

`check_invariants` enforces eight wire-protocol invariants -- the same ones
the Rust and Go servers self-check internally (RFC v3 §9.6); the fuzzer
catches them when the server fails to. Numbered as in the script's module
docstring so log messages cross-reference cleanly:

1. **Liveness.** After `pc.close()` and a 2 s grace window the server
   must emit `session.done` or the transport must close. A hang means the
   server is leaking a task -- a real bug, not a fuzz artefact. The check
   fires only **after** the harness's own `pc.close()`; sessions that
   simply run to their natural duration are not flagged.
2. **No `internal_state_error`.** RFC v3 §9.6 reserves that code for
   server-side invariant-violation bugs. Surfacing one means an
   assertion in `state.go` / `state.rs` tripped.
3. **`response.created` <-> `response.done` matched.** Every started
   response must reach a terminal.
4. **`audio_end_ms` monotonic per `item_id`.** Audio cursors don't go
   backwards within a logical user-turn.
5. **Every event has a `type` field.** Well-formedness.
6. **`response.done` at most once per `response_id`** (RFC v3 §8.4).
7. **No `response.output_audio.delta` after `response.output_audio.done`**
   for the same `(response_id, item_id)` pair (RFC v3 §8.2).
8. **No panics / unexpected process death** during the run (checked by
   the harness via subprocess exit-code surveillance, not in
   `check_invariants`).

Fragment reassembly (RFC v3 §10.4) is applied before invariants run.

## Gated-fusion weights -- `eou_lib/gate.py::DEFAULT_GATED_FUSION_WEIGHTS`

Three implementations, **one set of numbers**:

- `client/eou_lib/gate.py::DEFAULT_GATED_FUSION_WEIGHTS`
- `go/internal/eou/gated_fusion.go::DefaultGatedFusionWeights`
- `rust/src/eou/...::DEFAULT_GATED_FUSION_WEIGHTS`

The literal was trained on a 350-row English-language sample of
`pipecat-ai/smart-turn-data-v3-test`. 5-fold cross-validation held-out
accuracy: **93.1 % +/- 1.7 %**. `client/gated_fusion/train.py` is the
training pipeline; after each re-fit it must overwrite the constant in
**all three** sources -- Python literal, Go struct, Rust struct -- in
lockstep.

## OpenAI-surface probe -- `check_openai_surface.py`

Walks the documented OpenAI REST surface against an arbitrary target
server. Path parameters (`{model}`, `{file_id}`, ...) are filled with
syntactically valid placeholders that almost certainly won't match real
entities. `404` on those paths is ambiguous -- route missing or entity
missing -- so the verdict-mapping treats `404` on a `{param}` path as
"present and behaving" iff the route is in the documented OpenAPI spec.
Surface coverage only, not end-to-end correctness.

## Diarized-JSON contract -- `test_diarized_json.py`

Asserts the `segment` shape returned by `/v1/audio/transcriptions` with
`response_format=diarized_json` matches OpenAI's published spec keys
(`type`, `id`, `start`, `end`, `text`, `speaker`). Server-side
implementation: `go/internal/stt/http.go::serveDiarized`; segment-assignment
algorithm (midpoint-in-cluster, fallback to nearest cluster) in
`go/IMPLEMENTATION.md` §19.6.
