# speaches-plus, explained

How this server is built and why. The numbered chapters are the narrative:
mechanism, not measurements -- no chapter quotes a timing, a throughput figure
or a benchmark result.

Everything else under `docs/` lives here too, as dotted subchapters
(`NN.M-<name>.md`): the working documents -- status files, RFCs, design docs,
parity matrices, measured surveys -- attached to the chapter they report on.
Subchapters MAY quote numbers; each carries its own provenance discipline.
The chapter is where you learn how a thing works; its subchapters are where
you learn what state it is in today.

Read the chapters in order the first time. Each chapter names the files that
implement what it describes, so it doubles as a map of the tree.

> **There is no provenance register, and a number cites its own basis or is
> marked UNVERIFIED.** There is no `INDEX.md`, no `§`-numbered section register,
> no `F-NN` flag ids --
> so a citation naming any of those is citing something that does not exist and
> reads as evidence while being none. A number earns its place one of two ways:
> it names a **suite in the tree that reproduces it**, or it carries its **basis
> tuple** (checkpoint, harness, backend, batch, token count, sha, log path)
> inline, in the paragraph making the claim, where it cannot be pruned away from
> it. A number with neither is labelled **UNVERIFIED at HEAD** at its site,
> together with what would establish it; that label is a standing invitation to
> re-measure, not a licence to quote.

| # | Chapter | What it answers |
|---|---|---|
| 1 | [Architecture](01-architecture.md) | What the crates are, why the workspace is split the way it is, and why `python/`, `go/`, `client/`, `conformance/` and `inspector/` all exist beside the Rust tree |
| 2 | [Model loading](02-model-loading.md) | How a checkpoint on disk becomes a served model: registry, family dispatch, the tensor-name contract, quantization detection, chat templates and reasoning splitting |
| 3 | [Speculative decoding](03-speculative-decoding.md) | The draft/verify/accept loop, the drafter families, why acceptance preserves output, and what each eligibility condition protects |
| 4 | [Kernels and quantization](04-kernels-and-quantization.md) | The CUDA build, the FFI surface, the number formats, MoE dispatch, KV layout, and why kernels are hand-written |
| 5 | [Backends](05-backends.md) | Why a WGSL backend exists next to CUDA, how the two are kept honest, and the feature-gating trap that makes an empty test suite look green |
| 6 | [Serving surface](06-serving-surface.md) | Every HTTP endpoint, what backs it, where it diverges from OpenAI on purpose, and how a missing model degrades one route instead of the process |
| 7 | [Speech stack](07-speech-stack.md) | STT, TTS, VAD, turn detection, diarization and alignment, and how the realtime loop ties them to the LLM |
| 8 | [Build and testing](08-build-and-testing.md) | The flake, why `nvk.sh` replaces bare cargo, the test taxonomy, and how to tell a passing suite from an empty one |

## Subchapters

**1 -- Architecture**

- [1.1 Charter](01.1-native-rust-port-prd.md) -- the normative goals, the acceptance bar, and the T0-T5 numerics contract
- [1.4 STATUS](01.4-STATUS.md) -- the canonical implementation-status snapshot against the PRD

**2 -- Model loading**

- [2.1 Model compatibility matrix](02.1-model-compat-matrix.md) -- what this server can actually serve, audited against the tree
- [2.2 GGUF format notes](02.2-gguf-format.md) -- the GGUF loader, format notes and upstream citations

**3 -- Speculative decoding**

- [3.1 MTP drafter contract](03.1-mtp-drafter-notes.md) -- the shapes, constants and dataflow the gemma-4-E4B assistant drafter must match

**4 -- Kernels and quantization**

- [4.1 FP8 contract](04.1-fp8.md) -- cross-backend contract, defaults and known issues; `include_str!`'d by `fp8_contract_e4m3.rs`
- [4.2 FP8 epilogue mechanism](04.2-fp8-epilogue-mechanism.md) -- what the 2026-08 wgpu fix actually changed
- [4.3 Quantization balance](04.3-quantization-balance.md) -- per-component performance/quality recipe
- [4.5 WGSL -> naga -> MSL kernel rules](04.5-wgsl-naga-msl-kernel-rules.md) -- design rules so the next Apple-GPU kernel is fast the first time (§R6 carries the measured 4/8-bit format ranking on Metal)

**5 -- Backends**

- [5.1 wgpu status](05.1-wgpu-status.md) -- canonical wgpu document: what works, how correct, what remains
- [5.2 Kernel parity](05.2-kernel-parity-matrix.md) -- what would catch a wrong number: the three cross-backend suites, the host oracles that carry the rest, and the kernels with no value gate at all
- [5.5 ROCm port](05.5-rocm-port-status.md) -- the HIP backend: wave-width rules, the C-ABI gap, and the NVFP4-on-AMD decision
- [5.6 macOS port](05.6-macos-port-status.md) -- the single Apple-silicon document: status, measured matrix, hardware verdicts (the perf rules live in 5.7 and 4.5)
- [5.7 Apple-silicon rules](05.7-apple-silicon-inference-architecture.md) -- why things are the way they are; read before any perf/quant/kernel work
- [5.8 Laguna on wgpu](05.8-laguna-wgpu-status.md) -- the Laguna XS 2.1 port: shapes, reused kernels, the partial-rope trap, and the expert gather
- [5.9 Prefix cache](05.9-prefix-cache-scope.md) -- when KV reuse is legal, the bounds a rewind must clear, and what the key must contain
- [5.10 KV disk persistence](05.10-kv-disk-persistence.md) -- warm tokens across restarts (all wgpu decoders, `NV_KV_CACHE_DIR`)
- [5.11 KV prime checkpointing](05.11-kv-prime-checkpointing.md) -- deep-context primes saved and kill-resumable (`NV_KV_CKPT_DIR`), fingerprint-keyed, refuse-on-mismatch
- [5.12 Kernel rationale](05.12-kernel-rationale.md) -- the measured facts behind kernel design choices: grid-axis guard law, flash-decode bandwidth structure, smem opt-in, WGSL occupancy rules, harness invariants

**6 -- Serving surface**

- [6.1 CUDA serving architecture](06.1-serving-architecture.md) -- paged KV, continuous batching, the graphed Eagle3 verify loop
- [6.2 Concurrency and ordering](06.2-concurrency-and-ordering.md) -- the realtime-lane contract; read before parallelizing anything there
- [6.3 OCR](06.3-ocr.md) -- three OCR models behind one trait, and the verdict vs vLLM
- [6.4 Notebook digitizer](06.4-notebooks.md) -- scanned notebook pages to per-page markdown
- [6.5 LoRA design](06.5-lora-design.md) -- normative multi-adapter serving design, vLLM-cited; §12 is the wgpu/Metal path
- [6.6 LoRA training](06.6-lora-training.md) -- train an adapter and serve it here, end to end

**7 -- Speech stack**

- [7.1 Realtime session RFC v3](07.1-barge-turn-spec-rfc-v3.md) -- the `/v1/realtime` session state machine, normative constants

**8 -- Build and testing**

- [8.1 Quality harness](08.1-quality-harness.md) -- the only sanctioned way to measure output quality, and what each model's prompt framing costs
- [8.3 Rocq proof notes](08.3-rocq-proof-notes.md) -- what code each proof mirrors and what is deliberately not claimed
- [8.4 PERFORMANCE](08.4-PERFORMANCE.md) -- how to quote a number: the basis-tuple rule

## Reading paths

- **Serving a model** -- 2, then 6.
- **Making decode faster** -- 3, then 4.
- **Porting to non-CUDA hardware** -- 5, then 4.
- **Changing a kernel** -- 4, then 8 for how to build and verify it.
- **Touching audio** -- 7, then 6.
- **"What state is X in?"** -- the subchapters of X's chapter, starting from [1.4 STATUS](01.4-STATUS.md).

## Two conventions worth knowing before you edit

Comments are stripped by policy; `scripts/strip-comments.py` is the canonical
formatter and the comment-free state is intended, not neglect. And a test suite
compiled without the cargo feature it needs prints a pass while executing
nothing -- chapter 8 explains how to detect that, and chapter 5 explains why the
backend split makes it easy to hit.
