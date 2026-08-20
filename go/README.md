# speaches-plus / go

A Go server wire-compatible with the speaches `/v1/realtime` endpoint.
`go/IMPLEMENTATION.md` is the canonical as-built reference: file map
(§1), rationale, constants, and the RFC v3 feature/compliance table
(§17).

## Conformance

The canonical harness is shared across implementations: `client/`
drives the language-agnostic e2e suite over the wire; `conformance/`
holds the §15.6 fixture corpus + canonical assertion library that both
Go and Rust replay in-process (corpus layout and bands:
`docs/book/08-build-and-testing.md` § "The conformance corpus" and
`conformance/README.md`).

| Tool | Purpose |
|---|---|
| `client/test_e2e.py` | Single-leg STT round-trip |
| `client/test_e2e_full.py` | Full conversation (orchestrator spawns subprocess) |
| `client/fuzz_e2e.py` | Black-box fuzzer + RFC v3 invariant checker (§9.6, §8.2, §8.4, §10.x) |
| `client/fake_llm.py` | Deterministic LLM stub (validates received text + emits fixed response) |
| `client/fixtures/` | Synthetic audio WAVs |
| `conformance/lib/trace_invariants.py` | W1-W8 wire + I1-I11 state-machine invariants checker |
| `conformance/lib/trace_diff.py` | Canonical trace diff against expected |
| `conformance/runner/run_fixture.py` | Standalone fixture runner (no per-language harness needed) |

Run the e2e suite against this server with `--target` or
`--speaches-binary`; run the in-process corpus replay with
`go test ./internal/realtime/ -run TestConformanceCorpus -v`.
Spec violations surface as wire-trace diffs against the canonical
expected. Both `client/test_e2e.py` and `client/test_e2e_full.py` pass
(transcription leg, fake-LLM forward leg, TTS round-trip leg).

## Build

cgo is required: whisper.cpp for STT, libopus for outbound audio,
onnxruntime for Silero VAD + Kokoro TTS, libespeak-ng for the Kokoro
phonemizer. The root `flake.nix` wires everything up:

```sh
nix develop          # cpu shell (or metal on Darwin)
go build -o bin/server ./cmd/server

# CUDA: linux-cuda profile uses CUDA-enabled whisper-cpp/ctranslate2 and
# defaults CT2_DEVICE=cuda. See ../README.md for the full recipe.
nix develop ../#cuda
go build -o bin/server-cuda ./cmd/server
```

Inside the shell, `KOKORO_MODEL` / `KOKORO_VOICES` / `SILERO_VAD_MODEL`
/ `CT2_MODEL` resolve from `$HF_HUB_CACHE` if it points at a populated
HF cache. Anything that's already exported wins.

## Run

```sh
# Whisper ggml weights aren't in HF cache, download once:
mkdir -p models
curl -L -o models/ggml-tiny.en.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin

# Single-leg STT (intent=transcription). KOKORO_* not required.
./bin/server --whisper-model models/ggml-tiny.en.bin --addr :8765
./client/test_e2e.py --target http://localhost:8765

# Full conversation (intent=conversation). The dev shell exports
# KOKORO_MODEL / KOKORO_VOICES / ESPEAK_DATA_PATH / SILERO_VAD_MODEL.
./bin/server --addr :8765
./client/test_e2e_full.py --speaches-binary $PWD/bin/server

# CUDA:
./bin/server-cuda --addr :8765 --stt-backend ct2
# nvidia-smi --query-compute-apps shows server-cuda holding ~2.7 GiB.

# macOS Metal: nixpkgs ships a Metal-enabled libwhisper.dylib. Pick the
# whisper.cpp backend; ct2 has no Metal kernels and runs CPU-only here.
./bin/server --addr :8765 --stt-backend whisper_cpp \
             --whisper-model models/ggml-large-v3-turbo.bin
# init log should include `use gpu = 1` and `using embedded metal library`.
```

## Architecture

Audio path: inbound RTP opus 48 kHz -> pion/opus pure-Go decode (mono
s16) -> linear resample to 16 kHz -> Silero VAD (32 ms windows) ->
speech_start / speech_end -> flush utterance -> whisper.cpp (cgo) ->
`transcription.completed`; conversation intent continues: POST
`CHAT_COMPLETION_BASE_URL` -> Kokoro ONNX (espeak-ng -> IPA -> Kokoro
vocab -> ORT -> 24 kHz f32) -> resample 24 -> 48 kHz -> libopus 20 ms
frames (cgo) -> pion `TrackLocalStaticSample.WriteSample`, paced at
wallclock -> `response.done`.

## Architectural notes

- `pc.OnDataChannel` not `CreateDataChannel`: speaches reacts to the
  client's channel; mirroring that keeps both ends on the same SCTP
  stream so `session.created` reaches the client.
- Mono-output opus decoder. `pion/opus.NewDecoder()` defaults to
  48 kHz mono and downmixes packets that declare stereo. The decode
  path honors that default rather than `track.Codec().Channels`, since
  treating the mono output as L/R-interleaved averages adjacent samples
  and corrupts the audio.
- `onnxruntime_go` must track the nix-built `libonnxruntime`'s ORT C
  API level (the binding was pinned at v1.18 when the shipped ORT
  1.24.4 topped out at API v24 and newer bindings required v25; go.mod
  carries the current pin).
- Silero `sr` input takes shape `[1]`, not `[]`. Even though the
  model declares the sr tensor as a 0-dim scalar, yalue's binding
  rejects empty shapes; ORT accepts a 1-D 1-element tensor.
- Phonemizer is process-global. espeak-ng's engine is a singleton
  (it uses C globals for state). `kokoro.Close()` deliberately does
  not terminate it, so repeated `NewKokoro` calls within one process
  (e.g. test files) remain safe.
- Kokoro tokens cross-checked: `internal/tts/vocab_test.go` asserts
  byte-identical token IDs against the Python `kokoro_onnx` reference,
  so phoneme->ID drift surfaces immediately.
- whisper.cpp `use_gpu` left at library default. `whisper_cgo.c` used
  to hard-code `cparams.use_gpu = false`; removing that lets
  whisper.cpp bind the GPU backend already linked into
  `libwhisper.dylib`/`.so` -- Metal on Darwin, cuBLAS on `nix develop
  .#cuda`, no-op on linux-cpu. Net effect on Apple Silicon is ~13x
  c=1 throughput vs the ct2 (CPU) backend; see
  `docs/book/08.4-PERFORMANCE.md` § "STT smoke benchmark".
