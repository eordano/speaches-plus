# Architecture

## What the system is

One Axum binary serves an OpenAI-shaped HTTP API — chat completions, speech
synthesis, transcription, translation, text and audio embeddings, speaker
diarization, OCR, PII classification, a realtime WebRTC voice protocol — on an
in-tree model stack: kernels, quantized linear algebra, model graphs, the
KV-cache scheduler, speculative decoding, tokenizers and weight loading are all
Rust crates in the server's workspace, per **G6 — single binary**, "No Python
at runtime. No sidecar." (01.1-native-rust-port-prd.md).

## One process, one binary

`rust/Cargo.toml` declares one workspace (`members = [".", "crates/*"]`) whose
root package `speaches-plus` is both a library (`rust/src/lib.rs`) and a binary
(`rust/src/main.rs`); every model crate is a path dependency of it, so `cargo
build` yields one executable owning the whole stack. The HTTP/WS server runs on
the tokio multi-thread runtime; `BatchEngineHandle::spawn`
(`nv-engine/src/batch_runtime.rs`) runs the batch engine on a dedicated OS
thread named `nv-batch-engine`, building the stepper inside that thread so the
device context stays there, taking commands over an unbounded `mpsc` channel
and replying per request on a bounded `mpsc::Receiver<EngineEvent>`; handlers
never touch the device. One process, not the parent's process-per-rank: one
allocator, one log stream (PRD §3). `AppState` (`rust/src/lib.rs`) is the whole
shared surface — a `Models` handle, an optional default `dyn ChatEngine`,
`ChatRegistry`, TTS talker and speaker encoder; every field but `models` is
optional, so the binary boots and health-checks with any subset of the corpus.

## The crate graph

`rust/crates/` holds 19 crates, layered so each may only use the one below it.

| Crate | Layer | Exists for |
|---|---|---|
| `nv-config` | 0 | `Dtype`, `Backend`, `EngineConfig`. One file, no heavy deps. |
| `nv-lookup` | 0 | Suffix automaton / n-gram matching. **Zero dependencies**, so it compiles everywhere. |
| `nv-punkt` | 0 | Punkt sentence segmentation plus a `punkt-train` binary; cuts TTS text into utterances. |
| `nv-grammar` | 0 | Guided decoding: JSON Schema → regex → DFA over `regex-automata`, then a token mask over the vocabulary's raw bytes. |
| `nv-tokenizer` | 0 | Wrapper over `tokenizers` plus a minijinja chat-template renderer. |
| `nv-imgdec` | 0 | The one image decoder every surface goes through — containers, EXIF orientation, alpha matting, decompression-bomb bounds. |
| `nv-kernels` | 1 | The only non-trivial `build.rs`. Owns `.cu`, `.hip.cpp` and `.wgsl` sources and the `extern "C"` FFI surface. `links = "nv_kernels"`. |
| `nv-weights` | 1 | safetensors + `memmap2` + GGUF loading, quant-config parsing, LoRA adapter reading. |
| `nv-quant` | 2 | NVFP4, MXFP4, FP8, int8 formats and the matmul dispatcher. |
| `nv-layers` | 2 | Linear, RMSNorm, RoPE, attention, MoE, conv, sampler, LoRA slots; `backend.rs` defines `BackendKind` and probes what a device can run. |
| `nv-train` | 2 | LoRA trainable modules and PEFT entries on candle `Var`s. |
| `nv-models` | 3 | Model graphs: Gemma4 (dense/MoE/E4B/vision/audio/GGUF/graph), Qwen3, Qwen3.5-MoE and its dense-hybrid sibling, gpt-oss, Laguna, DeepSeek-OCR, dots.ocr, GOT-OCR, per-backend wgpu twins, an `nvk-train` binary. |
| `nv-engine` | 3 | Scheduler, block manager, paged KV, sequence state, the batch engine and its worker thread. |
| `nv-specdecode` | 4 | Eagle3, DFlash, MTP, chain and n-gram drafters plus the verify path. |
| `nv-omni` | 4 | Audio encoder (AuT), vision tower, thinker, talker, vocoder, learned velocity field codec. |
| `nv-tts` | 4 | Qwen3-TTS talker, codec decoder, speaker encoder, streaming, voice profiles. |
| `nv-aligner` | 4 | Forced alignment: Viterbi DP over logprobs, then SRT/VTT/diarized-JSON emission. |
| `nv-ocr` | — | Classical CPU OCR. No candle, no GPU — independent of the tensor stack. |
| `nv-runner` | — | Standalone Qwen3 greedy runner, not linked into the binary. The smallest complete decode path; the reference implementation. |

The graph is acyclic, `nv-config` and `nv-weights` at its base
(02-model-loading.md); `rust/crates/*` is the authority on the count.

## Backends are a feature axis, not a crate axis

Backend support is a cargo feature fanning out from the root manifest:

```
cuda = [..., "nv-kernels/cuda", "nv-engine/cuda", "nv-models/cuda",
        "nv-specdecode/cuda", "nv-omni/cuda", "nv-tts/cuda",
        "nv-aligner/cuda", "candle-core/cuda", ...]
```

`metal` and `wgpu` fan out identically, and a crate's `cuda` feature enables
only its dependencies' `cuda` features, so the feature graph mirrors the crate
graph: `nv-kernels`'s `build.rs` spawns `nvcc` only under `CARGO_FEATURE_CUDA`.
`BackendKind` (`nv_layers::backend`) is `Cuda` | `Wgpu` | `Cpu`, `BackendSel`
parses `NV_KERNELS_BACKEND`, and the serving selector
`rust/src/oapi/backend_select.rs` (`NV_SERVE_BACKEND=cuda|wgpu|auto`) is
cuda-first: `auto` falls back to wgpu only on a concrete CUDA reason it cannot
serve the model, an explicit `wgpu` never falls back, and `WgpuEvidence`
separates a decoder verified on real weights from one inferred from its
architecture family, so **the surface claims no more than has been tested**
(05-backends.md). `nv-kernels` carries `cuda/` (with a vendored Marlin
subtree), `cuda_sm120/` for Blackwell FP4, `hip/` (a ROCm stub), and `wgsl/`
driven by `src/wgpu_backend/`; Metal reaches the GPU through wgpu/WGSL, so
`nv-kernels`'s own `metal` feature is empty (04-kernels-and-quantization.md).

## The HTTP binary

`rust/src/` is organized by protocol surface, not by model. `oapi/` holds the
OpenAI-shaped routes: `chat.rs` (request/response types, tool-call rewriting,
thinking-tag splitting, multimodal extraction, the `ChatEngine` trait),
`chat_engine/` (CUDA engine build, per-family decode loops `gemma4_loop.rs` /
`laguna_loop.rs` / `qwen.rs`, sampling, spec-decode windowing, SSE streaming),
`chat_engine_wgpu.rs` (the wgpu engine), and `build.rs`'s `detect_family`,
which maps `config.json` to a `ModelFamily` — **the single dispatch point
between serving and model graphs** (02-model-loading.md,
06-serving-surface.md). `realtime/` is the WebRTC/WebSocket voice loop —
transport, session state machine, framing, audio in/out, barge-in, EOU
detection, inspector emission (07.1-barge-turn-spec-rfc-v3.md). `audio/`,
`stt/`, `tts/`, `vad/`, `eou/` and `diarization/` are the speech plumbing both
the batch routes and the realtime loop consume (07-speech-stack.md); `pii/` is
classifier, span mapper, Viterbi decode and redaction renderer; `inspect/` is
per-session NDJSON + audio capture, retention and `/v1/inspect/*`; `models.rs`,
`defaults.rs`, `errors.rs`, `ids.rs`, `otel.rs` and `trace.rs` are model
catalog, env defaults, error envelopes, ID generation and OTLP tracing.

`main.rs` assembles one `axum::Router`, merging chat, voice-profile, PII, OCR,
backend-reporting and fine-tuning sub-routers, each omitted when its model is
absent. Also mounted: `/health`, `/health/ready`, `/health/sessions`,
`/version`, `/metrics`, `/v1/backends`, `/v1/internal/chat-engines`,
`/v1/fine_tuning/jobs`. `oapi/fine_tuning.rs` ties `nv-models`' LoRA trainer
(`train_runner::run`) to `oapi/lora.rs`'s adapter catalog, the one place the
tree crosses the PRD's "inference only" non-goal.

## The parallel trees

**`python/`** is the parity reference: `nano_vllm/`, the vLLM-style engine
ported module for module (`engine/` onto `nv-engine`, `layers/` onto
`nv-layers` + `nv-quant`, `spec_decode/` onto `nv-specdecode`), with a `pyo3`
dev-dependency embedding CPython for parity tests that diff at named hook
points; PRD §8 keeps it, **not** deprecated. **`go/`** independently implements
the realtime endpoint only (pion WebRTC, whisper.cpp and Kokoro via cgo, ONNX
Runtime), with no chat engine; a Go port of the inference stack is an explicit
PRD non-goal. **`conformance/`** is the contract between them: wire-trace
fixtures replayed in-process by both, endpoint manifests driven over HTTP, and
declarative manifests consumed by named Rust and Go tests, over
`conformance/lib/`'s assertion library (trace canonicalization, diffing, the
W1–W8 wire and I1–I11 state-machine invariants), **so an implementation cannot
pass by agreeing with itself** (08-build-and-testing.md). **`client/`** holds
Python and Node e2e drivers driving the server over `aiortc`, upstream
`speaches`'s own WebRTC stack; **`inspector/`** is a dependency-free static
frontend over `/v1/inspect/*`, not served by the binary; `rocq/` holds Coq/Rocq
proofs of scheduling and roofline properties; `examples/` a LoRA training
notebook.

## Build and model provisioning

`flake.nix` is the outer boundary: dev shells, packages, and the model corpus
pinned via `nix-hug.lib.fetchModel` with content-addressed file-tree hashes.
Its shells export `HF_HUB_CACHE`, `TRANSFORMERS_OFFLINE` and per-model
directory variables into a store-path Hub cache, **so the runtime never touches
the network and a checkpoint is a build input rather than an ambient fact.**
`rust/build.rs` compiles the espeak-ng glue for TTS phonemization and resolves
dylibs on Darwin; `nv-kernels/build.rs` generates bindgen bindings from
`include/nv_kernels.h`, then — only under the `cuda` feature — spawns `nvcc`
per `.cu` file, honouring `CUDA_ARCH_LIST`, `CUTLASS_DIR`, `NCCL_ROOT` and
friends. Build goes through `rust/scripts/nvk.sh`, and 08-build-and-testing.md
also covers the failure mode this repo repeats: tests that compile to nothing
or return early and still report a pass.
