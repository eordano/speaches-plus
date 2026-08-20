# 040-align-* family

Forced-alignment / diarization fixtures for `/v1/audio/transcriptions`
and `/v1/audio/diarization`.

## Fixtures (3)

| ID                              | Coverage                                                          |
| ------------------------------- | ----------------------------------------------------------------- |
| `040-align-en-words`            | English per-word SRT; ±100 ms timing tolerance                    |
| `040-align-zh-segments`         | Mandarin diarized_json; segment-count + ordering                  |
| `040-align-multi-lang-de-es-fr` | Smoke test — de / es / fr language hints, SRT                     |

## "Passing" in placeholder mode

All three are `skip_when_no_model: true`. The placeholder runner verifies:

- The fixture's `request_multipart` parses (i.e. the WAV decoder accepts the
  artifact).
- Input artifacts exist on disk and have the documented duration / sample
  rate (basic RIFF header check).

It does NOT POST against a live server in placeholder mode; the live-server
mode is gated on `--target <url>` and a model being loaded.

## "Passing" once real models land

| Fixture                          | Canonical assertion                                                |
| -------------------------------- | ------------------------------------------------------------------ |
| `040-align-en-words`             | SRT with 3 ± 1 segments, monotonic, timing within ±100 ms          |
| `040-align-zh-segments`          | diarized_json with 2 ± 1 segments, ordered, text under T5          |
| `040-align-multi-lang-de-es-fr`  | All three language hints accepted; each returns parseable SRT      |

## Input artifacts

All synthesized programmatically (`generate_audio.py` in each fixture
directory). The placeholder audio is a sine burst pattern — the aligner
accepts any decodable WAV, so audio-vs-transcript correctness is not part
of the placeholder assertion; only SRT/diarized-JSON *shape* is.

| Artifact                                                  | Duration | Notes                                |
| --------------------------------------------------------- | -------- | ------------------------------------ |
| `040-align-en-words/audio.wav`                            | 2.0 s    | 3 sine bursts (440/523.25/659.25 Hz) |
| `040-align-zh-segments/audio.wav`                         | 3.0 s    | 2 sine bursts (392/466.16 Hz)        |
| `040-align-multi-lang-de-es-fr/audio_{de,es,fr}.wav`      | 1.5 s ea | single sine each                     |

## Discovery

Same as 020/030: `fixture.json` files, picked up by
`conformance/runner/run_endpoint_fixture.py`. The legacy Realtime runners
skip these directories because they lack `expected.jsonl`.
