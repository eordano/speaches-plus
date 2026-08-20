# Chapter 5 — Two Backends

The engine carries every GPU kernel twice: once as CUDA C++ under
`rust/crates/nv-kernels/cuda/`, once as WGSL under
`rust/crates/nv-kernels/wgsl/`. Chapter 4 described what those kernels compute.
This chapter is about the fact that there are two of them, and the machinery that
stops the second copy from silently rotting. Per-kernel gates are
`docs/book/05.2-kernel-parity-matrix.md`; wgpu port status is
`docs/book/05.1-wgpu-status.md`; the Metal and HIP adapters are
`docs/book/05.6-macos-port-status.md` and `docs/book/05.5-rocm-port-status.md`.

## Why a second backend

The CUDA path is bound to NVIDIA hardware three times over: `cudarc` for the
driver API, `nvcc` at build time, and — for the two SM120 files under
`cuda_sm120/` — CUTLASS/FlashInfer tensor-core mainloops that exist for one
vendor's silicon only.

wgpu is the escape. `nv-kernels/Cargo.toml` pulls `wgpu` with the backend
features `vulkan`, `metal` and `dx12` compiled in simultaneously; which one runs
is an adapter choice made at startup, not a build choice. One WGSL corpus
therefore reaches NVIDIA and AMD through Vulkan, Apple silicon through Metal, and
Windows through DX12, with no CUDA toolchain anywhere in the build. **That is the
entire payoff: portability of the *decode path*, not speed.**

There is a third native backend and it is not wgpu: the `rocm` feature compiles
checked-in HIP sources from `rust/crates/nv-kernels/hip/`, with unported kernels
quarantined under `hip/pending/` and named loudly by a build warning. The
distinction matters when reasoning about AMD -- wgpu reaches AMD through Vulkan
today, HIP reaches it natively but is unfinished, and **neither gets NVFP4 for
free**: that format is 16-element blocks with an E4M3 scale, while AMD's matrix
units implement OCP MXFP4 with 32-element blocks and an E8M0 power-of-two scale.
They are not interchangeable, which is why in-shader dequantization to f16 is the
only route that works on more than one vendor
(`docs/book/05.1-wgpu-status.md` §2.4).

## Where the second copy lives

**Kernels.** `rust/crates/nv-kernels/src/wgpu_backend/` is the whole wgpu
runtime: `device.rs` (adapter selection, feature request, a shader-driven probe
for the real subgroup width), `qualify.rs` (`Capabilities` scraped from wgpu
`Features`/`Limits`/`DownlevelCapabilities`, plus `reduction_strategy()` →
subgroup vs. workgroup tree and `gemm_strategy()` → coop-matrix vs. scalar),
`dispatch.rs` (buffers, bind groups, pipelines, pass encoding, readback), and
`kernels/` — one module per kernel, each holding
`pub const WGSL: &str = include_str!("../../../wgsl/<name>.wgsl")` plus the shape
checks and the launch. Every shader is composed against a shared prelude:
`dequant.rs` exposes `wgsl/dequant.wgsl` as `DEQUANT_WGSL` and `compose()`
prepends it, so `ue4m3_decode`, `e4m3_decode`, `int4_decode_u4b8` and the
software `nv_tanhf` are one definition for the whole corpus.

**Models.** `rust/crates/nv-models/src/lib.rs` gates seven modules on
`feature = "wgpu"` — `gemma4_wgpu`, `gemma4_e4b_wgpu`, `gemma4_moe_wgpu`,
`qwen3_5_moe_wgpu`, `qwen3_5_dense_wgpu`, `gpt_oss_wgpu`,
`gemma4_assistant_wgpu` — and `laguna_wgpu/` on `feature = "laguna-wip"`. Eight
more (`gemma4_graph`, `graph_engine`, `laguna_fa2`, `laguna_fp8`, `laguna_graph`,
`laguna_serve`, `laguna_step_graph`, `qwen3_5_mtp`) are `feature = "cuda"`. The
remainder — `gemma4`, `gemma4_e4b`, `gemma4_moe`, `qwen3_5_moe`, `laguna`, the
vision and audio towers — are ungated candle modules and are what the CUDA
serving path actually runs.

**Serving.** `rust/src/oapi/chat_engine_wgpu.rs` (plus `spec.rs`, `learned.rs`,
`mem_fit.rs`) is the wgpu chat engine, declared under `#[cfg(feature = "wgpu")]`.
Its CUDA counterpart is `rust/src/oapi/chat_engine.rs` and `chat_engine/`.

## The same model, expressed twice

The duplication is deliberately partial: a wgpu decoder reuses the CUDA-side
model's *configuration* types and reimplements only the execution.
`gemma4_wgpu.rs` and `gemma4_e4b_wgpu.rs` import `Gemma4Config` and `LayerType`
from `gemma4.rs`; `qwen3_5_moe_wgpu.rs` imports `Qwen3MoeConfig` and `LayerType`
from `qwen3_5_moe.rs`; `qwen3_5_dense_wgpu.rs` goes further and imports its NVFP4
loading, GEMV source generation and host-side quantization helpers directly from
`qwen3_5_moe_wgpu.rs`; `laguna_wgpu/config.rs` re-exports `LagunaConfig`,
`LagunaGating`, `MlpLayerType` and `yarn_inv_freq` from `laguna.rs`. **So config
parsing, checkpoint layout knowledge and layer-type schedules have one owner;
only the forward pass is written twice**, which is why a checkpoint that loads on
one backend generally classifies identically on the other.

The wgpu forward pass is structurally different from candle's: the resident
decoders stage weights from a `nv_weights::WeightLoader` on `Device::Cpu`,
convert to host bit vectors, upload once, build a list of compute pipelines and
bind groups at construction, and replay that list per token inside one command
encoder, reading back four bytes. `laguna_wgpu` makes the shape explicit --
`gpu.rs` defines `Pass { pipeline, bind, grid, entry }` and a `Builder`
accumulating `Vec<Pass>`, and `dispatch.rs` supplies `encode_pass_list`,
`submit_pass_list` and a `Recorded`/`replay` pair. No candle `Device` appears on
any wgpu forward path; the test that asserted that by reading the sources went
with the culled inventory slice, so it is a convention now rather than a gate.

Model-specific WGSL used to live as string constants inside `nv-models`,
invisible to any `ls nv-kernels/wgsl/` census; every static shader is now a file
under `nv-kernels/wgsl/`, which is why that directory is several times larger
than the kernel-module count suggests — the `e4b_*`, `g4w_*`, `g4m_*`, `q3m_*`,
`q3d_*` and `laguna_*` files are model glue, not `nv-kernels` kernels, and most
carry no suite of their own. What is still assembled in Rust is enumerated by
`wgsl_ue4m3_subnormal_census.rs`, precisely so a shader cannot hide from a source
census by being built with `writeln!`.

**The seam back to shared code is narrow and load-bearing.** Every decoder
exposes `decode_step(token: u32) -> u32`,
`decode_step_logits(token) -> (u32, Vec<f32>)`, `prefill_step`, `prefill_tokens`,
`reset`, `pass_count` and `current_pos`. The entire host-side generation stack in
`chat_engine.rs` — sampling parameters, `ChatSampler`, stop scanning, incremental
detokenization, logprob construction, guided decoding, `ChatRegistry` — consumes
`&[f32]` logits and nothing else, so it is backend-neutral and both engines share
it. Reasoning-trace splitting is likewise in the shared `oapi::chat` handler,
above both engines.

## Keeping the twins honest

Three distinct mechanisms, often conflated.

**CUDA↔WGSL kernel parity** is deliberately the *smallest* of the three: three
suites (`parity_gdn`, `parity_gemv_bf16_i8`, `parity_kv_fp8_paged`), each opening
with `#![cfg(all(feature = "cuda", feature = "wgpu"))]` and a local `backends()`
helper that acquires a `CudaStream` on device 0 *and* the shared `WgpuContext` in
one process, then drives the same inputs through the real `nv_kernels::cuda::*`
FFI and through WGSL. The count is asserted at run time by
`feature_census::EXPECT_CUDA_AND_WGPU`, not by this sentence, and a `parity_*`
glob overcounts it — `parity_verify_fused` is CUDA-only despite the name.
Cross-backend agreement is a weak oracle by construction (two implementations can
share a bug, and in this tree they have), so it is used only where the shipping
backend has no independent host reference;
`docs/book/05.2-kernel-parity-matrix.md` names the gate for each kernel and the
kernels that have none. Two divergences are deliberate and pinned by tests: where
CUDA kernels do no bounds checking and would read out of range, the wgpu wrappers
return `WgpuError::Shape` rather than reproduce undefined behaviour; and
`moe_permute` is deterministic on wgpu where CUDA's `atomicAdd` slot assignment
is not.

**wgpu-only correctness** lives in `rust/crates/nv-kernels/tests/wgpu_*.rs`,
gated `#![cfg(feature = "wgpu")]`, comparing WGSL against CPU oracles. This is
the only tier that can run on a machine with no CUDA at all, and it is what Apple
silicon exercises (`docs/book/05.6-macos-port-status.md` §A2).

**A third layer used to exist and no longer does.** `rust/tests/parity_inventory`
asserted structural facts about the wgpu port by reading source text, and was
removed with the rest of the slice that could not go red on a wrong number — its
assertions were string matches over sources the code had already moved past. What
it pinned about the capability registry is pinned again by a suite that fails on
a wrong answer rather than a moved string:
`nv-layers/tests/backend_registry_moe_grouped_nvfp4.rs` holds
`KernelId::MarlinGemmW4a16` as the only entry `kind_supports` reports missing on
wgpu, and holds `KernelId::MoeGroupedGemmNvfp4` present because
`moe_grouped_gemm.wgsl` implements it.

## Feature gating, and the trap

In `rust/Cargo.toml`:

```
wgpu = ["nv-kernels/wgpu", "nv-layers/wgpu", "nv-models/wgpu", "nv-models/laguna-wip"]
cuda = ["nv-kernels/cuda", "nv-models/cuda", "nv-specdecode/cuda", ...]
metal = ["nv-models/metal", "candle-core/metal", ...]
```

Three consequences that catch people:

1. **`metal` is not the WGSL path.** The top-level `metal` feature routes
   *candle* to Apple's Metal and does not imply `wgpu`; the Metal WGSL backend
   arrives via the `wgpu` crate's own always-compiled `metal` backend feature. On
   macOS you want `--features metal,wgpu`; with `metal` alone the wgpu chat
   engine compiles out entirely (`docs/book/05.6-macos-port-status.md` §A1).
2. **`nv-kernels`' own `metal` feature is inert.** It is declared in
   `crates/nv-kernels/Cargo.toml` and gates no code under `src/`.
3. **`laguna-wip` implies `wgpu`, not the reverse** in `nv-models`
   (`laguna-wip = ["wgpu"]`); the top-level `wgpu` feature adds it back
   explicitly, which is what makes `WgpuModelKind::Laguna` reachable.

Build-time backend compilation is driven from `crates/nv-kernels/build.rs`, which
calls `build_cuda()` only when `CARGO_FEATURE_CUDA` is set while bindgen over
`include/nv_kernels.h` runs unconditionally, so the FFI signatures exist even in
a CUDA-less build and only the objects are missing. A `wgpu`-only build therefore
invokes no `nvcc`, which is why `NVK_FEATURES=wgpu` is the fast edit loop.

**Now the trap.** A Rust integration test whose crate-level attribute is
`#![cfg(all(feature = "cuda", feature = "wgpu"))]` compiles to an empty binary
under one feature: cargo runs it, it contains zero tests, it prints
`0 passed; 0 failed; 0 ignored` in `0.00s`, and the run is green. Nothing
distinguishes that from a suite that genuinely had nothing to do, except that a
real `#[ignore]` skip reports a **non-zero ignored count**. Concretely,
`NVK_FEATURES=wgpu rust/scripts/nvk.sh test --test parity_rope` tests nothing
whatsoever and reports success. The countermeasures:
`NV_KERNELS_PARITY_REQUIRE=1` turns a *runtime* skip (no CUDA device, no wgpu
adapter, adapter not qualified) into a panic inside `backends()` and `nvk.sh`
exports it by default -- but it cannot help with the cfg case, because there is
no test left to panic; `NV_KERNELS_WGPU_REQUIRE=1` does the same for the `wgpu_*`
suites' `ctx_or_skip()`; and every real wgpu run prints an adapter banner from
`WgpuContext::summary()` before doing work, so **a wgpu result with no banner is
a skip regardless of what the result line says.** The rule that falls out,
repeated in `CLAUDE.md` and `docs/book/05.1-wgpu-status.md` §10: read a suite's
`#![cfg(...)]` header before trusting a green run, and treat `0 passed` in
`0.00s`, or `1 passed` in `0.00s` for anything that touches a GPU, as a skip
until proven otherwise. `08-build-and-testing.md` covers `nvk.sh` and lane
discipline in full.

## Selecting a backend

Two independent selection points, at different altitudes.

**Kernel level.** `nv-layers/src/backend.rs` defines
`BackendKind::{Cuda, Wgpu, Cpu}` and `BackendSel::{Auto, Cuda, Wgpu, Cpu}`,
parsed from `NV_KERNELS_BACKEND` (`BACKEND_ENV`). `resolve_from` takes probe
closures: `Auto` tries `probe_cuda()` first and falls back; an explicit selection
fails loudly rather than degrading. Both probes are cfg-aware and return the
honest reason `"nv-layers compiled without the <x> feature"` when the feature is
absent, so a misconfigured build reports a build problem instead of a hardware
one. `probe_wgpu` additionally requires `ctx.qualify().qualified`, so an adapter
that exists but lacks the needed limits is not selected. `Backend::open` then
constructs the concrete backend, and `kind_supports(kind, KernelId)` answers per
kernel: on wgpu everything except `MarlinGemmW4a16`, the one entry of
`KernelId::ALL` with no WGSL module — `gemm_w4a16_small_m` and `gemv_w4a16` are
not marlin. `nv-layers/tests/backend_registry_moe_grouped_nvfp4.rs` fails if that
missing set grows or shrinks.

**Serving level.** `rust/src/main.rs` chooses the registry at compile time
(`oapi::chat_engine_wgpu::registry_from_env_with_wgpu()` under
`#[cfg(feature = "wgpu")]`, plain `oapi::chat_engine::registry_from_env()`
otherwise). `WgpuRegistryPlan::from_env` then reads `NV_SERVE_BACKEND`,
`NV_WGPU_CHAT_MODEL_DIRS`, `NV_CHAT_MODEL_DIRS` and `NV_CHAT_MODEL_DIR` and
returns one of three plans:

- `Delegate` — hand off to `chat_engine::registry_from_env()`. **This is the
  default, so compiling the `wgpu` feature in cannot change a deployment that
  never asked for it.**
- `Extend(dirs)` — keep the base registry and add wgpu engines beside it, with
  colliding model ids aliased by a `#wgpu` suffix. On a build without `cuda` this
  is downgraded to `Replace`, because there the base registry *is* the wgpu path
  and extending it would load every checkpoint twice.
- `Replace(dirs)` — serve only wgpu engines. If every directory fails to load
  this **panics rather than starting**, symmetric with the CUDA path; the
  alternative was a server answering `/health` with 200 and having no
  `/v1/chat/completions` for a load balancer to find.

`rust/src/oapi/backend_select.rs` is the third surface and it *is* routed:
`main.rs` merges `backends_report_router()` at `BACKENDS_REPORT_ROUTE`
(`/v1/backends`) and adds `REALTIME_CAPABILITIES_WITH_BACKENDS_ROUTE`, so
`AUTO_POLICY`, the per-class decoder table and the per-backend unservable reasons
are answered live rather than only asserted by `tests/backend_wgpu_gate.rs`.
`docs/book/05.1-wgpu-status.md` §5.2 is the narrative view of that table; **read
the route, not either page, when the answer matters.**

Per-model routing inside the engine is `classify_wgpu_model`
(`chat_engine_wgpu.rs`), which reads `architectures` / `model_type` from
`config.json` and returns one of seven `WgpuModelKind`s — `Gemma4Dense`,
`Gemma4E4b`, `Gemma4Moe`, `Qwen3_5Moe`, `Qwen3_5Dense`, `GptOss`, `Laguna` —
matching the seven `Decoder` variants.

### Routing hazards worth knowing before you touch classification

Both classifiers are string/config matches, not enums straight from the
checkpoint, so ordering and discriminator choice carry real weight:

- **DiffusionGemma must be checked before the gemma arms.** Its id contains both
  `"gemma"` and `"a4b"`, so an id-substring classifier checking the gemma/MoE arms
  first would claim it as `Gemma4Moe` and report it servable; its transformer *is*
  gemma4-26B-A4B-shaped, but it decodes by block diffusion, not autoregressively,
  and would emit garbage. Both `backend_select::ModelClass::classify` and
  `classify_wgpu_model` check for it first and refuse by name, because its
  `text_config` is near-identical to a supported model's (28 of 29 shared keys
  equal, seven keys absent -- `docs/diffusion-gemma.md`) and the generic message
  would send an operator hunting for a decoder that is already present.
- **Gemma-4 MoE (26B-A4B) is told apart from dense/E4B gemma-4 only by config**,
  never by id: `enable_moe_block` picks MoE, `has_per_layer_embeddings()` then
  picks E4B vs dense. `chat_engine::detect_family` (CUDA) and
  `classify_wgpu_model` (wgpu) apply the same rule; routing on the id instead
  loads a model with no expert weights, or the wrong per-layer-embedding stack,
  and decodes garbage silently. Its cuda serving is wired --
  `detect_family` splits gemma4 on `enable_moe_block` into
  `ModelFamily::Gemma4Moe`, `build.rs` constructs `nv_models::gemma4_moe::Gemma4Moe`
  from safetensors, and `run_sampling_gemma4_moe` serves it with chunked prefill
  and eager host-routed experts, no CUDA graphs and no speculative decoding --
  and `cuda_model_unservable_reason` returns `None` for the class to match.
- **Gemma-4 E4B cuda serving is wired but gated off by default.** Both
  `chat_engine::build`'s `ModelFamily::Gemma4E4b` arm and
  `backend_select::cuda_model_unservable_reason` require `NV_E4B_CUDA_SERVE=1`, on
  purpose: flipping a serving default is a queued decision, not a capability gap,
  and **the capability report must describe exactly what `try_load` will do under
  the same environment**, or an operator gets a green backends-report for a model
  that then fails to load. Standalone w4a16 decode measured faster on cuda than on
  wgpu (measured on the development box).

## What is on wgpu, and what stays CUDA-only

The honest shape is *not* subset/superset: `gpt_oss` has a wgpu decoder while its
cuda arm is opt-in and deliberately more expensive, and Qwen3.5-dense loads on
wgpu while the cuda registry refuses the family outright
(`backend_select::QWEN35_DENSE_NO_CUDA` records why, and that the failure CUDA
produces today accuses the wrong party). **Any sentence of the form "wgpu is CUDA
minus X" is wrong**, and the per-capability answer is
`docs/book/05.1-wgpu-status.md` §5 rather than a second copy here.

The structural asymmetries that survive any capability round, each for a reason
rather than an oversight: **CUDA graph capture** and everything built on it
(Eagle3, DFlash verification) has no wgpu analog -- pre-recorded command buffers
cover replay but not the dynamic-shape tree-verify half; **paged KV and continuous
batching** are cuda-only because wgpu KV is one contiguous buffer sized at
construction, with `max_seq` baked into uniforms and RoPE tables and further
bounded by `max_storage_buffer_binding_size`, and both unblock from the same
missing piece, an M>1 wgpu forward; **Marlin** and **CUTLASS-speed grouped MoE
GEMM** stay cuda-only because their value is tensor-core scheduling WGSL cannot
express (`wgsl` has no device pointers and no way to reconstruct CuTe layout
descriptors); **multimodal** is blocked downstream of the towers, because the
resident decoders' only entry point takes a token id with no way to inject a
spliced embedding row; **STT and TTS** are third-party C++ (`whisper-rs`,
`ct2rs`) that falls back to CPU rather than to wgpu; and **LoRA on the resident
decoders** has real, tested wgpu kernels (`nv-kernels/tests/wgpu_lora.rs`) that
nothing splices into a resident pass list. Conversely **speculative decoding on
wgpu is E4B-only** (`spec_route_eligible` requires
`kind == WgpuModelKind::Gemma4E4b`), and **chunked prefill is delegated by all
seven kinds** through `Decoder::prefill_chunk_len` -- though the `Qwen3_5Dense`
path does not engage on an NVFP4 checkpoint, so **a wgpu serving
figure is not a decode figure**.

## The Vulkan loader, or your wgpu tests run nothing

wgpu enumerates adapters through the platform loader — on this box
`libvulkan.so.1` plus an ICD manifest, which the dev shell historically shipped
neither of. The failure mode is the worst possible one: `WgpuContext::shared()`
returns `NoAdapter`, `ctx_or_skip()` prints a `skip:` line and returns `None`,
the test body never executes, and the harness reports **`1 passed`** — a non-zero
pass count, so it does not even resemble the cfg'd-out case above.

`rust/scripts/nvk.sh` wires both halves, and **the ordering is load-bearing: the
exports must come *after* the cached `nix print-dev-env` snapshot is sourced**,
because sourcing it overwrites caller environment.

```sh
VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
LD_LIBRARY_PATH=<newest /nix/store/*-vulkan-loader-*/lib>:$LD_LIBRARY_PATH
```

`NVK_VULKAN_LOADER` overrides the loader directory. One further hazard recorded
in `docs/book/05.1-wgpu-status.md` §10: most `vulkan-loader` store paths on this
machine are **32-bit**, and a 32-bit loader fails with no diagnostic at all — not
even under `VK_LOADER_DEBUG` — so the path must be a *verified* 64-bit one. When
running a test binary directly rather than through `nvk.sh`, the exports have to
be reproduced by hand. The consequence for reading the archives: **any wgpu
timing or parity claim dated before the loader was wired was measured on
nothing**, and the adapter banner in the output is the discriminator.

## Connections

- `04-kernels-and-quantization.md` — what the kernels compute, the NVFP4 /
  w4a16 / FP8 formats both backends decode, and why in-shader dequantization is
  the portable choice.
- `08-build-and-testing.md` — `nvk.sh`, lanes, feature sets, and the full
  catalogue of ways a test in this repo can report success without executing.
- `docs/book/05.1-wgpu-status.md`, `docs/book/05.2-kernel-parity-matrix.md`,
  `docs/book/05.5-rocm-port-status.md`, `docs/book/05.6-macos-port-status.md`.
