# speaches-plus

A native inference server in Rust. One Axum binary serves an
OpenAI-compatible HTTP API -- chat completions, speech synthesis,
transcription and translation, text and audio embeddings, speaker
diarization, OCR, PII classification/redaction, and a realtime WebRTC
voice protocol -- on top of 18 in-house crates that implement the model
stack directly: CUDA and WGSL kernels, quantized linear algebra, model
graphs, a KV-cache/scheduler engine, speculative decoding, tokenizers,
and safetensors weight loading. No PyTorch, no Python in the Rust
serving path (`python/` ships a separate, PyTorch-based server).

The project started as a Rust/Go rewrite of the upstream `speaches`
realtime endpoint with a language-agnostic conformance rig. That part still exists and still matters (see
"Realtime protocol and conformance" below), but the bulk of the repo is
now the inference engine itself.

Canonical references, in order of authority:

- [`docs/book/05.7-apple-silicon-inference-architecture.md`](docs/book/05.7-apple-silicon-inference-architecture.md)
  -- **read this first before measuring, quantizing, or optimizing anything
  on Apple silicon.** How to run LLMs well on Ultra-class Apple silicon: the crossover
  rule that decides whether fewer bits is actually faster (here it usually
  is not -- int8 beats every 4-bit kernel in the survey, and the 31B got
  1.96x faster by reading 43.5% *more* bytes), why our own per-dispatch
  profiler overstates small passes ~10x and once invented a 154 ms/token
  overhead that did not exist, the standing list of "obviously better"
  changes that measured worse, the kernel rules, the per-token cost model
  and its consequences for spec decode and batching, and where we stand
  versus the field.
- [`docs/book/01.4-STATUS.md`](docs/book/01.4-STATUS.md) -- what is actually implemented,
  phase by phase, including what is *not* verified.
- [`docs/book/08.4-PERFORMANCE.md`](docs/book/08.4-PERFORMANCE.md) -- every measured number,
  both backends, with the conditions that make it meaningful.
- [`docs/book/08.1-quality-harness.md`](docs/book/08.1-quality-harness.md) -- how output
  quality is measured, and the chat-template rules that decide whether a
  measurement means anything.
- [`docs/book/05.1-wgpu-status.md`](docs/book/05.1-wgpu-status.md) -- the wgpu/Vulkan port:
  kernel parity, serving, and remaining work.
- [`docs/book/05.6-macos-port-status.md`](docs/book/05.6-macos-port-status.md) --
  the single Apple-silicon document: what serves, the measured matrix, and
  the known limitations.
- [`docs/book/04.1-fp8.md`](docs/book/04.1-fp8.md) -- the cross-backend fp8 contract and its
  default-flip gate.
- [`docs/book/01.1-native-rust-port-prd.md`](docs/book/01.1-native-rust-port-prd.md) -- the
  normative PRD.
- [`docs/book/07.1-barge-turn-spec-rfc-v3.md`](docs/book/07.1-barge-turn-spec-rfc-v3.md) --
  the normative realtime protocol spec.
- [`CLAUDE.md`](CLAUDE.md) -- working conventions and operational
  gotchas for this machine.

## API surface

23 OpenAI/Anthropic-compatible routes:

| Area | Routes |
|---|---|
| Chat / text | `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/responses/{id}`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, `/v1/embeddings` |
| Audio | `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/embeddings`, `/v1/audio/diarization`, `/v1/voice-profiles` |
| Vision | `/v1/ocr` |
| PII | `/v1/pii/classify`, `/v1/pii/classify/batch`, `/v1/pii/redact/analyze`, `/v1/pii/redact/render` |
| Realtime | `/v1/realtime` (WebRTC), `/v1/realtime/capabilities`, `/v1/inspect/sessions`, `/v1/inspect/sessions/history` |

Those 23 are the OpenAI/Anthropic-shaped surface, not the whole surface:
the binary mounts **35 distinct paths / 40 method+path pairs**. The rest are
operational or non-OpenAI -- `/health`, `/health/ready`, `/health/sessions`,
`/version`, `/metrics`, `/v1/internal/chat-engines`,
`/v1/voice-profiles/{name}`, three more `/v1/inspect/*` paths, and the
`/v1/fine_tuning/jobs` surface (`rust/src/oapi/fine_tuning.rs`, undocumented
and unaudited).

Which chat/embedding/STT/TTS checkpoints each route can actually serve,
and why: [`docs/book/02.1-model-compat-matrix.md`](docs/book/02.1-model-compat-matrix.md)
is the static trace of the CUDA dispatch;
[`docs/book/05.6-macos-port-status.md`](docs/book/05.6-macos-port-status.md) carries the
Apple-silicon measured matrix.

## Engine

- **Backends.** `nv-layers::backend::BackendKind` has three: **Cuda**,
  **Wgpu**, **Cpu**. CUDA is the mature path -- graph capture, paged KV,
  speculative decoding all live there. The wgpu path is real and serves
  chat with coherent output on five checkpoints across four model graphs
  (Gemma4-E4B bf16 and qat-w4a16-ct, Gemma4-31B-NVFP4, Qwen3.6-35B-A3B-NVFP4,
  Laguna-XS-2.1-NVFP4 -- all re-verified over HTTP on Apple silicon,
  [`docs/book/05.6-macos-port-status.md`](docs/book/05.6-macos-port-status.md)), but it is
  newer and less proven ([`docs/book/05.1-wgpu-status.md`](docs/book/05.1-wgpu-status.md)).
  Backend selection is `NV_SERVE_BACKEND=cuda|wgpu|auto`; `auto` is
  cuda-first and only falls back to wgpu when CUDA reports a concrete
  reason it cannot serve the model. The **`metal` cargo feature is real**
  -- wired through `candle-core/metal` in eight crates plus the binary, and
  the whole Apple-silicon serving surface runs on it; Metal compute reaches
  the GPU through wgpu/WGSL, so `nv-kernels`' own `metal` feature is an
  empty stub (there are no in-tree `.metal` kernels). `rocm` is a stub
  everywhere.
- **Quantization.** NVFP4 (e2m1 values + ue4m3 block scales, group 16),
  MXFP4 (group 32, e8m0 scales), FP8 e4m3, int8, w4a16/Marlin, bf16.
- **Speculative decoding.** Eagle3, DFlash, DSpark, MTP, an
  ngram/suffix-automaton drafter, and a chain speculative loop on wgpu.
  On wgpu/Metal it is measured as a **net loss** and gates itself off --
  `verify_chain` costs 2-5x a decode step, so break-even needs tau~2-7
  while the best drafter reaches 1.55-1.73
  ([`docs/book/05.6-macos-port-status.md`](docs/book/05.6-macos-port-status.md)).
- **OCR.** A classical CPU pipeline (`nv-ocr`) plus DeepSeek-OCR-2,
  both served at `/v1/ocr` ([`docs/book/06.3-ocr.md`](docs/book/06.3-ocr.md)).
- **Diarization.** DiariZen segmentation + WeSpeaker ResNet293-LM
  embeddings + online cosine clustering, at `/v1/audio/diarization`,
  as `response_format=diarized_json` on transcriptions, and as realtime
  data-channel events. **The segmentation ONNX is not shipped** -- until
  `rust/scripts/export-diarizen-onnx.py` has been run, `/v1/audio/diarization`
  answers `503 model_not_loaded` (correctly) and `diarized_json` returns
  every `speaker` as `null` with no failure signal (a real defect, tracked
  in [`docs/book/05.6-macos-port-status.md`](docs/book/05.6-macos-port-status.md)). The
  speaker-embedding half works on its own at `/v1/audio/embeddings`.

## Performance

Numbers go stale within hours here, so this README quotes none.
[`docs/book/08.4-PERFORMANCE.md`](docs/book/08.4-PERFORMANCE.md) is the rule for
quoting one (a number without its basis tuple is not a number), and the
per-platform measured records live in the book's chapter-5 and chapter-8
subchapters. Per [`docs/book/01.4-STATUS.md`](docs/book/01.4-STATUS.md), 0 of 9
PRD §11.5 performance gates have been measured -- what exists are real bs=1
decode figures with per-run provenance. Treat any number found outside those
documents as unverified.

## Building and testing

**Always build through `rust/scripts/nvk.sh`.** It owns the nix
devshell (cached `nix print-dev-env`, so `.cu` files are not recompiled
on every invocation), `TMPDIR`, `CUDA_ARCH_LIST`, a per-lane
`CARGO_TARGET_DIR`, and parity-gate env. Bare `cargo` or ad-hoc
`nix develop --command cargo` invocations are slower and subtly
different; see `CLAUDE.md` for the full rationale.

```sh
# default: cuda,wgpu features, nv-kernels package
rust/scripts/nvk.sh test --test parity_gdn -- --nocapture

# fast edit loop: wgpu only, skips every nvcc invocation
NVK_FEATURES=wgpu rust/scripts/nvk.sh test --test wgpu_rope

# another crate
NVK_PKG=nv-models rust/scripts/nvk.sh test --test gemma4_moe

# concurrent agents/sessions: one lane each
NVK_LANE=mylane rust/scripts/nvk.sh test --test <suite>
```

`rust/scripts/nvk.sh --help` lists all knobs. Real-weight tests are
`#[ignore]` + feature-gated and opt in via `NV_*_TEST=1` env vars.
Note that `parity_*` suites require both `cuda` and `wgpu` features --
running them with `NVK_FEATURES=wgpu` compiles them to nothing and
reports a vacuous pass.

### Running the server

The server binary lives under `rust/`. It reads models from `nix-hug`
symlinks pinned in `flake.nix`, so run it from a hub shell (which
realizes the pinned HF weights, ~25 GB) or fetch models first:

```sh
nix run .#fetch-models              # one-time weight fetch
nix develop .#cuda                  # CUDA shell + pinned model corpus
cd rust && cargo run --features cuda -- --port 8000
```

`nix develop` / `nix develop .#cuda` are the plain (no-hub) shells; see
[`rust/README.md`](rust/README.md) for shell internals and CUDA build
details.

## Realtime protocol and conformance

The realtime side implements a WebRTC voice loop with barge-in:
server-side VAD, streaming STT, end-of-utterance detection, chat
upstream, streaming TTS, and interruption semantics. The normative
contract is [`docs/book/07.1-barge-turn-spec-rfc-v3.md`](docs/book/07.1-barge-turn-spec-rfc-v3.md).

Two implementations exist -- the Rust server and a self-contained Go
server (`go/`) -- and both replay the shared fixture corpus at
`conformance/` in-process against a canonical assertion library. Each
`conformance/fixtures/NNN-<name>/` directory is one scenario
(`input.jsonl` ops in, `expected.jsonl` canonical wire trace out);
scenario-level protocol coverage lives there, not in ad-hoc scripts.
The 020/030/040 fixture families are `skip_when_no_model: true`
placeholders pending model-backed runs.

`client/` holds Python e2e drivers (built on `aiortc`, the same WebRTC
stack upstream speaches uses) plus a black-box fuzzer that enforces the
RFC's wire invariants. Every script is a self-contained PEP 723 file --
`uv run` resolves deps on first use.

```sh
./conformance/runner/run_fixture.py --all             # language-agnostic
cd go   && go test ./internal/realtime/ -run TestConformanceCorpus -v
cd rust && cargo test --test conformance

./client/test_e2e.py --target http://localhost:8000   # over-the-wire STT round-trip
./client/test_e2e_full.py                             # full conversation loop
./client/fuzz_e2e.py                                  # invariant fuzzer
```

Both servers report their feature surface at
`GET /v1/realtime/capabilities`. Per-implementation as-built references:
[`rust/IMPLEMENTATION.md`](rust/IMPLEMENTATION.md),
[`go/IMPLEMENTATION.md`](go/IMPLEMENTATION.md).

## Layout

```
speaches-plus/
  rust/
    src/                 Axum HTTP binary (routes, chat engine dispatch, realtime)
    crates/
      nv-kernels         CUDA (.cu) + WGSL kernels, parity harness
      nv-layers          linear/attention/MoE layers, backend selection
      nv-models          model graphs (Gemma4, Qwen3/3.5/3.6-MoE, Laguna, DeepSeek-OCR-2, ...)
      nv-engine          scheduler, block manager, paged KV, CUDA graphs
      nv-specdecode      Eagle3 / DFlash / DSpark / MTP / ngram drafters
      nv-quant           NVFP4, MXFP4, FP8, int8, Marlin
      nv-weights         safetensors + quant-config loading
      nv-tokenizer       tokenizers + incremental detokenization
      nv-grammar         guided decoding (JSON-schema -> regex -> DFA)
      nv-tts             Qwen3-TTS pipeline
      nv-omni            audio/vision encoders, talker, vocoder
      nv-aligner         forced alignment, timestamps, SRT/VTT
      nv-ocr             classical CPU OCR pipeline
      nv-punkt           sentence segmentation
      nv-config, nv-runner
  go/                    Go realtime server (self-contained: pion, whisper.cpp, Kokoro)
  python/                Qwen3-Omni multimodal HTTP server (PyTorch; Apache-2.0, derived from upstream speaches)
  rocq/                  Rocq (Coq) proofs + generators: launch geometry, roofline, KV budgets
  conformance/           shared fixture corpus + canonical assertions + runner
  client/                Python e2e drivers + fuzzer (aiortc)
  studio/                nur studio: React/Vite web UI (proxies /v1 to the server; NUR_API overrides)
  inspector/             realtime session inspector web UI
  examples/              worked examples (LoRA training notebook + data)
  scripts/               repo tooling (strip-comments.py, bench.py, chat.py, dev-env.sh)
  docs/book/             the book: narrative chapters + every working doc as dotted subchapters
  flake.nix              dev shells + pinned model corpus (nix-hug)
```

## License

Source code: **AGPL-3.0-or-later** (see [`LICENSE`](LICENSE)), except
`python/`, which derives from upstream speaches and stays **Apache-2.0**
(see [`python/LICENSE`](python/LICENSE)).

Model weights are governed by their own licenses -- see
[`NOTICE`](NOTICE) for the per-model list. The most restrictive bundle
is the DiariZen segmentation model (**CC-BY-NC-4.0, non-commercial
only**); commercial deployments must replace it. Every other shipped
weight permits commercial use.
