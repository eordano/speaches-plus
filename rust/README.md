# speaches-plus / rust

Rust implementation of the speaches realtime endpoint.
`client/test_e2e.py` (single-leg STT round-trip) and
`client/test_e2e_full.py` (full conversation loop) both pass with
arbitrary TTS response text via libespeak-ng phonemization.

`IMPLEMENTATION.md` is the canonical as-built reference.
`../docs/book/01.4-STATUS.md` tracks PRD-phase progress. `../CLAUDE.md`
has the working conventions + operational gotchas. Build discipline
(nvk.sh, lanes, measurement traps) is owned by the root `README.md`
§"Building and testing", `../CLAUDE.md`, and
`../docs/book/08-build-and-testing.md` -- not duplicated here; this
file keeps only the dev-shell internals and e2e client invocations.

## Conformance

`client/` drives the language-agnostic e2e suite over the wire;
`conformance/` holds the §15.6 fixture corpus + canonical assertion
library that both Rust and Go replay in-process (corpus layout: root
`README.md` and `../docs/book/08-build-and-testing.md`).

| Tool | Purpose |
|---|---|
| `client/test_e2e.py` | Single-leg STT round-trip |
| `client/test_e2e_full.py` | Full conversation (orchestrator spawns subprocess) |
| `client/fuzz_e2e.py` | Black-box fuzzer + RFC v3 invariant checker (§9.6, §8.2, §8.4, §10.x) |
| `client/fake_llm.py` | Deterministic LLM stub (validates received text + emits fixed response) |
| `client/fixtures/` | Synthetic audio WAVs |
| `conformance/fixtures/` | Fixture corpus (input.jsonl + expected.jsonl pairs) |
| `conformance/lib/trace_invariants.py` | W1-W8 wire invariants and I1-I11 state-machine invariants checker |
| `conformance/lib/trace_diff.py` | Canonical trace diff against expected |
| `conformance/runner/run_fixture.py` | Standalone fixture runner (no per-language harness needed) |

Run the e2e suite against this server with `--target` or
`--speaches-binary`; run the in-process corpus replay with
`cargo test --test conformance`. Spec-normative failures surface as
wire-trace diffs against the canonical expected.

## Build profiles (dev-shell internals)

The root `flake.nix` exposes dev shells that each set
`$SPEACHES_CARGO_FEATURE`, the cargo feature gating the acceleration
backend:

| Shell | Cargo feature | Backend |
|---|---|---|
| `nix develop` (Darwin) | `metal` | Metal whisper.cpp + Accelerate ct2 |
| `nix develop` (Linux)  | (none)  | CPU-only whisper.cpp + ct2 |
| `nix develop .#cuda`   | `cuda`  | cuBLAS whisper.cpp + ct2 with `cuda-dynamic-loading` |

The cuda shell drives `mkShell` off `cudaPackages.backendStdenv`,
exports `CUDA_PATH` / `CUDA_ARCH_LIST`, prepends the cuda_cudart
`lib/stubs` to `LIBRARY_PATH`, and runs `scripts/patch-ct2rs-cuda.sh`
from the `shellHook` so ct2rs's vendored CTranslate2 4.7.1 builds
against CUDA >= 12.8's reorganized thrust headers. See the root
`README.md` for details.

## Models -- pinned via `nix-hug`

The model corpus is pinned declaratively via the
[nix-hug](https://github.com/eordano/nix-hug) flake;
`flake.nix::mkSpeachesHubCache` is the canonical pin set (profiles:
`../CLAUDE.md`). `flake.nix` exposes a `speaches-models-hub` package --
a store-path Hub cache (one snapshot per revision under
`models--<org>--<repo>/snapshots/<rev>/`) that any process pointed at
via `$HF_HUB_CACHE` reads without network access. The Qwen 3.6 pin
excludes `model_visual.safetensors` and `model_mtp.safetensors` by
default -- pass `withMultimodal = true;` to
`lib.${system}.mkSpeachesHubCache` (or build
`.#speaches-models-hub-mm`) to include them. First build downloads
~25 GB of LFS blobs; rebuilds are content-addressed and free.

```sh
nix build .#speaches-models-hub
ls result/        # -> models--RedHatAI--Qwen3.6-35B-A3B-NVFP4/, ...
export HF_HUB_CACHE=$PWD/result
export TRANSFORMERS_OFFLINE=1
```

Both `nix develop` and `nix develop .#cuda` set `$HF_HUB_CACHE`,
`$NV_MODELS_HUB`, and `$TRANSFORMERS_OFFLINE` automatically and resolve
per-model snapshot directories into these env vars (each defaults into
the hub, each overridable):

| Env var | Role |
|---|---|
| `NV_CHAT_MODEL_DIR` | Qwen 3.6 snapshot consumed by `NvEngineChat` |
| `CT2_MODEL` | Whisper-CT2 snapshot consumed by `ct2rs` |
| `KOKORO_MODEL` / `KOKORO_VOICES` | Kokoro ONNX (legacy hub entries) |
| `SILERO_VAD_MODEL` | Silero VAD ONNX |
| `NV_TTS_MODEL_DIR` | Qwen3-TTS snapshot |
| `NV_EMBED_MODEL_DIR` | Qwen3-Embedding snapshot |
| `NV_EAGLE3_DRAFT_DIR` | Eagle3 draft (Gemma-4-only) |
| `NV_GEMMA4_VERIFIER_DIR` | Gemma-4 NVFP4 verifier |
| `NV_SPEAKER_MODEL_DIR` | WeSpeaker snapshot |

The same env vars are baked into the `speaches-plus` and
`speaches-plus-cuda` `buildRustPackage` outputs -- `nix build
.#speaches-plus-cuda` produces a binary that already knows where its
models live.

## Quick start

Rust builds go through `rust/scripts/nvk.sh` (root `README.md` /
`../CLAUDE.md`); the steps below are the realtime-server run + e2e
recipe.

```sh
nix develop               # auto-picks cpu/metal; .#cuda for GPU; realises the hub
nix run .#fetch-models    # no-op once the nix-hug hub is built

cargo build --features "$SPEACHES_CARGO_FEATURE"
cargo run   --features "$SPEACHES_CARGO_FEATURE" -- --port 8000

# conformance corpus (no models, no network):
cargo test --features "$SPEACHES_CARGO_FEATURE" --test conformance

# end-to-end (another shell):
./client/test_e2e.py --target http://localhost:8000 --transcription-model tiny.en
./client/test_e2e_full.py --speaches-binary $(pwd)/target/debug/speaches-plus \
                          --transcription-model tiny.en \
                          --response "any arbitrary response works"
```

The server listens on `127.0.0.1:8000` by default; `--host`/`--port`
and `UVICORN_HOST`/`UVICORN_PORT` env vars are accepted (the latter
mirrors speaches' Python uvicorn convention). For
`intent=conversation` the server reads `CHAT_COMPLETION_BASE_URL`
(e.g. `http://localhost:11434/v1`), `CHAT_COMPLETION_API_KEY`, and
`ESPEAK_DATA_PATH` from the environment; the dev shell sets
`ESPEAK_DATA_PATH` automatically.

## Status

Both e2e suites pass end-to-end (transcription leg, fake-LLM forward
leg, TTS leg). p50 transcription latency on M-series CPU: ~400 ms;
conversation round-trip including LLM + TTS: ~1.2 s.

## What's implemented

`IMPLEMENTATION.md` §Layout maps every module; its §"RFC v3
conformance" lists the implemented spec features (barge-in delay +
suppression, server truncate, queue-cap backpressure, Predicted phase,
text/audio/fusion EOU, drain cap, session hard timeout, error registry,
inspector relay + OTel).

## Build prerequisites / interop

Nix package prerequisites and the `.cargo/config.toml` workarounds:
`IMPLEMENTATION.md` §"Build prerequisites". ONNX/model contracts
(Silero context window, Kokoro style slice, protobuf clash):
`IMPLEMENTATION.md` §"Subtle ONNX/model contracts". Two interop quirks
worth knowing here:

- aiortc puts a separate `ice-ufrag`/`ice-pwd` per m-line even when
  bundled, which webrtc-rs rejects -- see `sdp_filter`.
- Streaming TTS WAVs emit `0xFFFFFFFF` for the RIFF + data chunk
  sizes; hound rejects that, so the `/v1/audio/transcriptions` handler
  patches both before parsing (cf. `docs/book/07-speech-stack.md` §One
  canonical audio format).
