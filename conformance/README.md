# Conformance Corpus

Centralized fixture corpus shared by all implementations (Go, Rust, ...) per
`docs/book/07.1-barge-turn-spec-rfc-v3.md` §15.6 (the §-references in this
README point at that spec unless noted) plus the endpoint families from
`docs/book/01.1-native-rust-port-prd.md` §11.4. Layout, fixture kinds, loader
convention, canonicalization rules and runner semantics are documented in
`docs/book/08-build-and-testing.md` § "The conformance corpus"; this README
carries the per-band inventory and the add-a-fixture procedures.

Naming: every fixture is a directory directly under `fixtures/`, named
`NNN-<family>-<name>` -- except the 001-015 Realtime band, which predates the
family segment and uses `NNN-<name>`. No nested or unnumbered fixture trees.
Family-level docs are `fixtures/README-<NNN>-<family>.md`; the 050 and 060
bands hold a single fixture each and document themselves in that fixture's
`README.md`. The canonicalization reference implementation is
`go/internal/realtime/conformance_test.go::CanonicalizeTrace` (IDs become
`sess_N` / `item_N` / `resp_N` in first-appearance order).

## Bands

| band | family | kind | fixtures | consumed by |
| --- | --- | --- | --- | --- |
| 001-015 | realtime barge-turn | wire-trace | 15 | Go + Rust conformance gates, `run_fixture.py` |
| 020 | `020-chat-completions` | endpoint manifest | 4 | `run_endpoint_fixture.py` |
| 030 | `030-voice-clone` | endpoint manifest | 3 | `run_endpoint_fixture.py` |
| 040 | `040-align` | endpoint manifest | 3 | `run_endpoint_fixture.py` |
| 050 | `050-diarization` | declarative manifest | 1 | `rust/src/diarization/hop_sweep.rs` |
| 060 | `060-eou` | declarative manifest | 1 | `go/internal/eou/parity_corpus_test.go` + Rust/Python EOU tests |
| 070 | `070-ocr` | declarative manifest | 11 | `rust/crates/nv-ocr/tests/` |
| 071 | `071-ocr-layout` | declarative manifest | 5 | `rust/crates/nv-ocr/tests/`, `rust/crates/nv-models/tests/deepseek_ocr_*.rs` |

Declarative `fixture.json` payload keys are family-specific: `cases` for 060,
`gates` + `expected_text` for 070/071, `input_artifacts` + `consumed_by` for
050.

## Scenarios (001-015)

| ID  | Name                                  | Focus                                                              |
| --- | ------------------------------------- | ------------------------------------------------------------------ |
| 001 | clean-utterance                       | Clean turn, full assistant response                                |
| 002 | barge-in-streaming                    | User barges in mid-streaming assistant response                    |
| 003 | eou-reentry                           | Speaking -> Stopped -> Speaking re-entry                             |
| 004 | backchannel                           | Short utterance below `min_speech_for_response_ms`                 |
| 005 | manual-response-create                | Client-driven `response.create` with instructions override         |
| 006 | session-update-atomic                 | Invalid `session.update` rejected, reflective `session.updated` echo |
| 007 | per-status-audio-end-ms-completed     | W4 / §8.5 -- `audio_end_ms` on `response.done(status=completed)`    |
| 008 | per-status-audio-end-ms-cancelled     | W4 / §8.5 -- `audio_end_ms` on `response.done(status=cancelled)`    |
| 009 | per-status-audio-end-ms-incomplete    | W4 / §8.5 -- `audio_end_ms` on `response.done(status=incomplete)`   |
| 010 | session-update-atomic-both-fields     | §11.2.1 -- atomic accept/reject when one field is valid + one invalid |
| 011 | silence-only-input                    | §15.3 -- silent session: only `session.created`, nothing else       |
| 012 | per-status-audio-end-ms-failed        | W4 / §8.5 -- `audio_end_ms` on `response.done(status=failed)`       |
| 013 | session-update-invalid-no-speech-prob | §11.2 / §17.4 -- out-of-range `no_speech_prob_threshold` rejected   |
| 014 | session-update-invalid-neg-threshold  | §11.2 / §17.4 -- out-of-range `turn_detection.neg_threshold` rejected |
| 015 | session-update-invalid-min-speech-duration | §11.2 / §17.4 -- `min_speech_duration_ms` above the 60 s cap rejected |

## Fixtures (020 / 030 / 040)

| ID                                | Endpoint                                    |
| --------------------------------- | ------------------------------------------- |
| `020-chat-greedy-basic`           | POST /v1/chat/completions                   |
| `020-chat-sampled-temp07`         | POST /v1/chat/completions                   |
| `020-chat-structured-json`        | POST /v1/chat/completions                   |
| `020-chat-tool-call-stub`         | POST /v1/chat/completions (streaming)       |
| `030-voice-clone-base`            | POST /v1/audio/speech                       |
| `030-voice-clone-design-warm`     | POST /v1/voice-profiles -> /v1/audio/speech  |
| `030-voice-clone-custom-from-3sec`| POST /v1/voice-profiles -> GET -> DELETE      |
| `040-align-en-words`              | POST /v1/audio/transcriptions               |
| `040-align-zh-segments`           | POST /v1/audio/diarization                  |
| `040-align-multi-lang-de-es-fr`   | POST /v1/audio/transcriptions x 3           |

## Running

```sh
./conformance/runner/run_fixture.py fixtures/001-clean-utterance
./conformance/runner/run_fixture.py --all
./conformance/runner/run_fixture.py fixtures/002-barge-in-streaming --strict
./conformance/runner/run_endpoint_fixture.py
./conformance/runner/run_endpoint_fixture.py fixtures/020-chat-greedy-basic
```

Per-language gates (authoritative for the wire-trace band only -- they skip
directories lacking `expected.jsonl`):

- Go: `cd go && go test ./internal/realtime/ -run TestConformanceCorpus -v`
- Rust: `cd rust && cargo test --test conformance` -- note
  `rust/tests/conformance.rs` was culled from the vendored tree (431c0d5fb,
  2026-08-10); until restored, Go is the only live wire-trace gate here.

Band-specific gates:

- 060: `cd go && go test ./internal/eou/...`
- 070 / 071: `cargo test -p nv-ocr`, `cargo test -p nv-models deepseek_ocr`

## Adding a new fixture

Common to every band: create `conformance/fixtures/<NNN>-<family>-<name>/`,
set `name` to the directory name and `family` to the band's registered family,
write a `README.md`, and put input artifacts in the same directory (generator
scripts go in `conformance/tools/`, never as a test-time dependency).

### Wire-trace corpus (001-015)

1. Pick the next free `NNN` and create `conformance/fixtures/NNN-<name>/`.
2. Write `input.jsonl` (phase ops) and `expected.jsonl` (canonical trace).
3. Add `README.md` with a one-line summary and spec pins (§6.x, §8.x, Wn).
4. Run both per-language gates above; both must pass before the fixture
   lands.

### Endpoint corpus (020 / 030 / 040 -- PRD §11.4)

1. Re-use one of the reserved families: `020-chat-completions`,
   `030-voice-clone`, `040-align`.
2. Write `fixture.json` with `name`, `family`, `description`, an `endpoint`
   block or a `steps` array, `expected_response`, `comparison_strategy`,
   `ref_outputs`, and `skip_when_no_model`.
3. Document input artifacts under `input_artifacts`.
4. Add a `README.md` describing what the fixture asserts in placeholder
   mode vs once real models are wired.
5. Run `./conformance/runner/run_endpoint_fixture.py`; structural
   validation must pass.

### Declarative corpus (050 / 060 / 070 / 071)

1. Re-use a registered family, or add a new band to
   `run_endpoint_fixture.py::FAMILY_BANDS` before adding fixtures under it.
2. Write `fixture.json` with the common head (`name`, `family`,
   `description`, `comparison_strategy`, `skip_when_no_model`) plus the
   family's payload keys.
3. Commit input artifacts so the consuming test is hermetic.
4. Point the consuming Go/Rust test at
   `<repo>/conformance/fixtures/<dir>` -- no per-family subdirectory.
5. Run `./conformance/runner/run_endpoint_fixture.py` plus the band's own
   gate.
