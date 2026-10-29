# Build and Testing

The previous chapters described what the system is. This one describes how to
make it exist on a machine and how to find out whether it works. Two ideas carry
most of the weight: the environment is a nix flake, not a README of `apt install`
lines; and a green test in this repo is not by itself evidence that anything ran.

## The flake is the environment

`flake.nix` is the single source of truth for toolchains, native libraries,
model weights, and the env vars that bind them together. `.envrc` does nothing
but `use flake .#cuda` with `watch_file flake.nix flake.lock` — direnv
re-evaluates only when those two files change.

Dev shells are generated combinatorially. `mkShell` takes
`withCUDA` and a `profile`, and `devShells` instantiates it across every profile
name, twice on Linux:

- `.#default` — CPU, `full` profile.
- `.#no-models` — CPU, `profile = null`: toolchain only, no weights realized.
- `.#cuda`, `.#cuda-no-models` — the same pair with CUDA.
- `.#<profile>` and `.#cuda-<profile>` for each profile in `profiles`: `full`, `minimal`, `chat-only`, `qwen36`, `qwen35-dense`,
  `qwen35-dense-nvfp4`, `tts-only`, `tts-clone`, `audio-only`, `audio-light`,
  `mm`.
- `.#python` — a separate shell built from `pythonDevShell`, carrying the torch/
  transformers env plus pybind11 build helpers for the in-tree `ct2_bindings/`
  and `whisper_bindings/` extensions, and a `shellHook` that symlinks ONNX and
  GGML model files into `rust/models/` and `go/models/` so the Rust and Go test
  runners find them offline. The hook's `_ln` wrapper swallows `ln -sfn` errors:
  a symlink that already points into the read-only nix store fails with EPERM
  rather than being overwritten, and if it already points at the right store
  path the test runners are served either way. It skips a tree entirely when the
  checkout is missing. On Linux it also exports `LD_LIBRARY_PATH` for
  ctranslate2 and whisper-cpp: setuptools does not emit `-Wl,-rpath` for nix
  store paths, so without it `_ct2.so`/`_whisper.so` fail at import with
  "libctranslate2.so.4: cannot open shared object file"; macOS needs nothing —
  dylibs encode their absolute install_name.

Under CUDA the shell is built with `pkgs.mkShell.override { stdenv =
cudaPackages.backendStdenv; }` because nvcc 12.x rejects gcc ≥ 15 and
`backendStdenv` pins gcc 14. The shell's `env` block is where every native
dependency named in 04-kernels-and-quantization.md and 05-backends.md gets its
path: `LIBCLANG_PATH` for bindgen, `ORT_DYLIB_PATH`/`ONNXRUNTIME_LIB` for the
ONNX runtime backends, `ESPEAK_DATA_PATH`, `CUDA_PATH`, `CUTLASS_DIR` (an
out-of-nixpkgs CUTLASS v4.4.2 pin, since nixpkgs' 3.9.2 lacks the blockscaled
GEMV header the NVFP4 M=1 decode path needs), `FLASHINFER_DIR`, `NCCL_ROOT`,
`CUDNN_ROOT`, `CUDNN_FRONTEND_DIR`, and `CMAKE_CUDA_ARCHITECTURES`. It also sets
`SPEACHES_PROFILE` and `SPEACHES_CARGO_FEATURE`, which is how the shell tells you
which cargo feature the platform expects (`cuda` on Linux+CUDA, `metal` on
darwin, empty on Linux CPU).

Two escape hatches exist for entering the environment without paying flake
evaluation per command: `scripts/dev-env.sh` (source it; snapshots
`nix print-dev-env` into `.direnv/devenv-<profile>.env`, invalidated by
`flake.nix`/`flake.lock` mtime) and `rust/scripts/nvk.sh`, described below.

Go's cgo packages are the exception: `internal/stt`, `internal/tts` and
`internal/audio` need a live `nix develop` for header and library resolution — a
sourced env snapshot is not enough.

## Pinned weights: `fetchModel` and the hub cache

Weights are treated as build inputs, not as something you download. `mkModels` declares each model exactly once with `url`, `rev`,
`fileTreeHash`, optional include/exclude `filters`, and — crucially — the list of
env vars the binary reads to locate it. Each entry's `drv` is
`nix-hug.lib.<system>.fetchModel`, so a checkpoint is a content-addressed
derivation like any other. `snapshot` records the
`models--<org>--<repo>/snapshots/<rev>` path that HF-cache-shaped consumers
expect.

`filters` matter for more than size: `qwen36-text` excludes
`model_visual.safetensors` and `model_mtp.safetensors` to get a text-only view of
the same revision that `qwen36-mm` includes in full. Two profiles, one pin.

A profile is a list of model names. `mkSpeachesModelsHub` maps a profile through `mkModels` and hands the derivations to
`nix-hug.lib.buildCache`, producing a directory laid out exactly like a
HuggingFace hub cache. `mkProfileEnv` then walks the same list and emits, for
each model, every one of its declared `envVars` pointing at
`${hub}/${model.snapshot}`. The shell sets `HF_HUB_CACHE` to the hub and
`TRANSFORMERS_OFFLINE=1`, then merges `profileEnv` on top. That is the whole
mechanism: *declare a model once, list it in a profile, and both the hub path and
the `NV_*_DIR` variables the engine reads appear in the shell.*

The hubs are also first-class packages — `packages.<system>.speaches-models-hub-<profile>` for
every profile — and `lib.<system>.mkSpeachesHubCache` exposes the builder to
downstream flakes, alongside `mkSpeachesPlus`, `mkSpeachesPlusGo`, `profiles`,
and `models`.

Entering a hub-bearing shell realizes real weights into the store. `.#no-models`
and `.#cuda-no-models` exist precisely so you can compile without that.

The hub overrides your env, not the other way round. The shell (and the
snapshot `nvk.sh` sources) assigns `NV_CHAT_MODEL_DIR`, `NV_EMBEDDING_MODEL_DIR`
and `HF_HUB_CACHE`. Sourcing happens after caller env, so a
`NV_CHAT_MODEL_DIR=... nvk.sh ...` prefix is silently discarded and you measure a
different checkpoint than you named. Read the `loading … from` boot line, or
source first and override second.

## What the flake builds

`packages.<system>` carries `speaches-plus` (= `speaches-plus-full`),
`speaches-plus-<profile>` and, on Linux, `speaches-plus-cuda` plus
`speaches-plus-cuda-<profile>`; the Go server as `speaches-plus-go` /
`speaches-plus-go-cuda`; the Python multimodal nano-vLLM as
`speaches-plus-python`, `speaches-plus-python-pkg` (and `-cuda` variants) with
`hf-cache`; every `speaches-models-hub-<profile>`; and the DiariZen ONNX
toolchain (`diarizen-onnx-exporter`, `diarizen-segmentation-onnx`) built from a
separate nixos-24.05 pin because DiariZen vendors a pyannote.audio 3.1 fork the
unstable python set cannot satisfy. The ONNX artifact is produced reproducibly in
a sandbox from the nix-hug weights — no manual export step, no network.

**A profile picks weights, not code, so every `speaches-plus-<profile>` of one
backend is one derivation.** `HF_HUB_CACHE`, `SPEACHES_PROFILE_NAME` and each
`NV_*_DIR` are read with `std::env::var` at run time and no `build.rs` declares
`rerun-if-env-changed` for one, so `mkSpeachesPlus` keeps every model path out of
the derivation: `speaches-plus-cuda-full` and `speaches-plus-cuda-chat-only` are
the same store path, all twelve profiles share one `craneLib.buildDepsOnly` per
backend (cpu / cuda / vulkan) instead of twelve, a dependency compile never waits
on a 25 GB hub, and repinning a model rebuilds no Rust. The profile's own hub and
env are still on the package — `pkg.modelsHub`, `pkg.profileEnv`, `pkg.profile`
via passthru, realized only when read — and that is what a unit wires; nothing
reads them out of the build. Building a profile package therefore does not
fetch its models: build `speaches-models-hub-<profile>` for those.

The application build's `preBuild` deletes every `*/out/build/CMakeCache.txt`
under the cargo target dir before compiling: the deps-only stage leaves each
cmake build script's cache pinned to its own sandbox dir, and when a fingerprint
change re-runs one in the application sandbox (cc 1.9 re-ran sentencepiece-sys),
cmake refuses the foreign cache; dropping the caches makes a re-run reconfigure
in place.

`apps.<system>` are `strip-comments` (also `default`) and `fetch-models`.

## `rust/scripts/nvk.sh`, and what breaks without it

Rust work goes through `rust/scripts/nvk.sh`. Not `nix develop`, not bare
`cargo`. The wrapper is not a convenience alias; it owns five things that are
each individually load-bearing.

1. A cached dev environment. It runs `nix print-dev-env` once into
   `$NVK_CACHE/devenv-<shell>.sh`, keyed on an md5 of `flake.nix` + `flake.lock`
   + the nix-hug override, and sources that file thereafter. Re-entering
   `nix develop` yields a subtly different environment each time, which trips
   `nv-kernels/build.rs`'s `rerun-if-env-changed` lines (it declares
   `CUDA_ARCH_LIST`, `NVK_NVCC_WRAPPER` and others at `crates/nv-kernels/build.rs` (`rerun-if-env-changed`)) and
   recompiles every `.cu` on every invocation. `NVK_REFRESH=1` forces
   regeneration.

2. `CUDA_ARCH_LIST`, by assignment. `crates/nv-kernels/build.rs`'s arch list defaults to `"8.9;12.0"`,
   and so does the dev shell, because `cudaCapabilities` in `flake.nix` keeps
   `8.9` for the nixpkgs CUDA package set's binary-cache hits. On a single-SM120
   box that compiles a dead Ada architecture into every object. `nvk.sh`
   assigns `CUDA_ARCH_LIST` after sourcing the env and derives
   `CMAKE_CUDA_ARCHITECTURES` from it, because a `${VAR:-default}` cannot
   override a variable the sourced snapshot already set to a non-empty value.
   Override with `NVK_CUDA_ARCH`.

3. A Vulkan loader. The dev shell ships none, so wgpu enumerates zero
   adapters and every wgpu GPU test returns early — while still printing a
   passing result. `nvk.sh` exports `VK_ICD_FILENAMES` from
   `/run/opengl-driver` and prepends the store's vulkan-loader to
   `LD_LIBRARY_PATH`, and it must do so after sourcing the snapshot for the
   same overwrite reason as above. Override with `NVK_VULKAN_LOADER`. A wgpu
   parity or timing claim produced without a loader present rests on nothing —
   see 05-backends.md.

4. A per-lane `CARGO_TARGET_DIR`. `$NVK_CACHE/tgt-$NVK_LANE`, default lane
   `base`.

5. ccache. Since every lane has its own target dir, N lanes recompile the
   same `.cu` sources N times. `nvk.sh` sets `NVK_NVCC_WRAPPER` (consumed by
   `crates/nv-kernels/build.rs`'s `nvcc_launcher`, which generates a shim in `OUT_DIR` — the
   `cc` crate calls `cc::Build::compiler()` explicitly and does not consult
   `RUSTC_WRAPPER`) plus `CMAKE_{CUDA,C,CXX}_COMPILER_LAUNCHER` for the
   cmake-driven sub-builds. It is ccache and not sccache: sccache parses
   `nvcc --dryrun` output and treats any line not matching `^([_A-Z]+)=` as a
   subcommand, and nixpkgs' `nvcc.profile` emits
   `#$ compiler-bindir=/nix/store/…`, which fails that regex and kills every
   compile. Disable with `NVK_CCACHE=0`.

6. sccache for rustc — opt-in, and not what it looks like. `NVK_SCCACHE=1`
   sets `RUSTC_WRAPPER` (plus `CARGO_INCREMENTAL=0`, which sccache requires).
   The obvious hope — lane #2 reuses lane #1's compiled crates — is falsified
   by measurement: sccache's rust cache key includes the per-lane target-dir
   path, so a fresh lane hits 0% of a cache another lane just filled and comes
   out slower than bare (nv-models `--release` lane-cold: 5m52s vs 5m31s).
   The one case that pays is rebuilding a lane under the same name after
   its target dir was reaped: 100% hit rate, 4m39s vs 5m31s (~16% — the
   remaining wall is linking, build scripts, and proc-macro crates, which
   sccache does not cache). Hence default 0; `NVK_SCCACHE_DIR` (default
   `$HOME/build/sccache`), `NVK_SCCACHE_SIZE`, `NVK_SCCACHE_BIN`
   (provision via `nix build --inputs-from . nixpkgs#sccache`). The real
   cross-lane economics live in ccache (above) and in reusing lane names.

It also sets `TMPDIR` into the cache dir (`/tmp` here is a root-mounted tmpfs;
filling it breaks the machine, not just the build) and defaults
`NV_KERNELS_PARITY_REQUIRE=1`, which is discussed under the trap below.

Selection knobs: `NVK_PKG` (default `nv-kernels`), `NVK_FEATURES` (default
`cuda,wgpu` for `nv-kernels`, `cuda` otherwise, empty string for none), `NVK_JOBS`,
`NVK_LANE`, `NVK_SHELL`, `NVK_CACHE`. The devshell attribute follows the
features: anything containing `cuda` selects `.#cuda`, otherwise `.#default`.

```sh
rust/scripts/nvk.sh test --test rope -- --nocapture
NVK_LANE=rope rust/scripts/nvk.sh test --test rope
NVK_FEATURES=wgpu rust/scripts/nvk.sh test --test wgpu_rope
NVK_PKG=nv-models rust/scripts/nvk.sh test --release --test laguna_fa2_contract
rust/scripts/nvk.sh --help
```

`NVK_FEATURES=wgpu` skips every nvcc invocation and is the fast edit loop for
backend work — but see the trap: it also silences every suite cfg'd on `cuda`,
including the cross-backend ones.

The CUDA build is large and memory-hungry. `NVK_JOBS` defaults deliberately low;
raising it, or running several lanes at once, can drive load high enough to
starve the machine. Prefer fewer concurrent lanes over more parallelism inside
one.

## The per-lane convention

Concurrent agents work directly in the tree on `main` — no branches, no
worktrees — and are isolated by disjoint file ownership plus **disjoint
lanes**. Every concurrent worker must set its own `NVK_LANE`; sharing a lane
means two cargo processes fighting over one target dir.

The cost of that isolation is disk: nothing is shared between lanes except the
ccache (sccache does not cross lanes — see the build point above), and per-lane
target dirs are the dominant disk consumer in this repo — far ahead of model
weights. Reuse lane names where you can, and prefer a filesystem with headroom
(`NVK_CACHE` accepts any path).
On a snapshotting filesystem, deleting stale target dirs frees nothing until the
snapshots referencing them expire — check the pool, not `df`.

`nvk.sh lanes` is the ledger for that disk: it lists every `tgt-*` dir with
size, idle days, and the PIDs of any live consumer (matched via `pgrep` on the
target-dir path and the `NVK_LANE=` token, plus `/proc/*/environ`).
`nvk.sh lanes --reap DAYS` deletes lanes idle longer than `DAYS`;
`--dry-run` prints the doomed list and deletes nothing. Three lanes are never
reaped: `tgt-base`, the caller's current `NVK_LANE`, and any lane a live
process references. Reaping is a manual act — nothing in the repo runs it on a
timer, because deleting build state is a decision, not a side effect.

## Test taxonomy

Four kinds of test live in this tree, with different costs and different
preconditions.

Unit and source-invariant tests. Ordinary `#[test]` functions in `src/`, plus
a category of inventory tests that assert properties of the source itself:
`rust/tests/backends_report_route.rs` reads `main.rs` to prove the binary mounts
the backends report. These are cheap and need no GPU.

Numerical parity. There is no golden-fixture tier ladder; see
the private status ledger's Numerics row for what its absence ungates. The repo-wide rule
it modeled still stands: without the parent `nano-qwen3-omni` repo present, scaffolds get
architectural verification — real weights load with the right shapes, forward
produces the right output shape — and nothing may be called numerical parity.

Backend parity is thin. `rust/scripts/parity-audit.sh` ratchets the suites it
lists (today `wgpu_quantize_nvfp4_bf16_cpu_ref`); the `wgpu_*` suites in
`rust/crates/nv-kernels/tests/` hold wgpu kernels to host oracles. These are the
executable form of the equivalence claims in 04-kernels-and-quantization.md and
05-backends.md.

Conformance fixtures. A language-agnostic corpus under `conformance/`,
described in its own section below.

Real-weight tests. Anything that loads a checkpoint or touches the GPU is
`#[ignore]`d, usually `#[cfg(feature = "cuda")]`, and additionally gated on an
opt-in environment variable. The canonical shape, from
`crates/nv-models/tests/gemma4_vision_graph_capture.rs` (the file is
`#![cfg(feature = "cuda")]`):

```rust
#[test]
#[ignore]
fn e4b_real_tower_graph_parity_and_encode_wall() {
    if std::env::var("NV_GEMMA4_VISION_REAL_TEST").ok().as_deref() != Some("1") {
        panic!("… NV_GEMMA4_VISION_REAL_TEST=1 is not set. This is a SKIP, not a pass.");
    }
    …
}
```

There are many such variables — `NV_LAGUNA_TEST`, `NV_WGPU_SERVE_TEST`,
`NV_GEMMA4_WGPU_TEST`, `NV_E4B_WGPU_TEST`, `NV_TOOLS_REAL_TEST`, and more. Each
names the assets it needs. None of them belong in a CI run without a GPU.

A gate that `panic!`s when its variable is unset is stronger than one that
prints `skip` and returns, because the second shape is indistinguishable from a
pass in the harness output. `rust/tests/guided_json_stress.rs` still has the
skip shape; `rust/tests/gemma4_mm_e2e.rs` and the example above have the panic
shape.

Laguna prefix checkpoints. Validation is specific to the recorded serving
composition, KV/draft arm and chunk size.
[`05.9-prefix-cache-scope.md`](05.9-prefix-cache-scope.md#running-the-gates)
provides the exact commands and coverage boundaries. The planner's
`laguna_prefix::tests` covers shared-allocation accounting, byte/count admission,
LRU selection and backtracking without GPU execution. The three ignored
`prefix_cuda_gate` fixtures need `NV_LAGUNA_PREFIX_CUDA_TEST=1` and actual CUDA;
they cover BF16/FixedOne target snapshots and ring draft context, not full-model
numerics.

The separate shared-prefill lifecycle suite
(`laguna_serve::shared_prefill_handoff_cuda_tests`) requires
`NV_LAGUNA_B2_HANDOFF_TEST=1`, the pinned target directory and content manifest,
and the matching prepared `m2` or packed-K2 `m2-spec` settings. Run its ignored
tests individually through `nvk.sh` on an available CUDA device. It checks source
ownership, cancellation, draft-context isolation, decode progress between
prefill chunks, buffer retention, physical graph masks and teardown. Its
full-capacity allocations contain shorter test prompts; real near-cap HTTP
execution, throughput and model quality require separate validation. The
`retirement_producer_tests` and HTTP `laguna_loop::overload_tests` cover returned
errors, client-close observation, reservations and capability-based context-tail
routing without claiming recovery from a CUDA call that never returns.

Unset `NV_LAGUNA_PREFILL_BF16_HANDOFF` for the ordinary prefix gates. The
default-off handoff requires explicit `NV_LAGUNA_PREFIX_REUSE=0`:
ordinary checkpoint operations address the step cache, while handoff prefill
writes a separate BF16 source. Source-unaware reuse
must refuse at startup. Its combined central build and handoff-plus-prefix
composition remain unqualified; the ordinary prefix tests do not cover them.

Laguna optional decode kernels. `NV_LAGUNA_VERIFY_ROUTER_GEMV=1` selects
the fused router and activation staging for eligible verifier rows 2–8.
Its arithmetic differs from the replaced GEMM, and no suite in the tree covers
router staging. Verifier rows 1-8 on the rows-w8 path always run the fused FP8
q|k|v|gate and norm+rope row kernels, and the gated o_proj is fused whenever its
shape fits (the unfused q8 rows o_proj is the n_q > 64 or head_dim % 8
fallback); `NV_LAGUNA_QKVO_RSTD_FUSE=1` additionally folds the rstd reduction into the
projection at M1. The suite covering row projections (`laguna_r3_rows_gemv`,
`NV_LAGUNA_ROWS_GEMV_TEST=1`) checks every row 1..8 bit for bit against the
m=1 decode kernels and compares fused and separate normalization for FP8
and int8 weights. `NV_LAGUNA_VERIFY_ROUTER_GEMV` and the rstd fold default
off; both are read at graph construction. Preserve actual route markers and validate
the composed binary through `nvk.sh`; an older host result does not qualify a
selective source port or the overall model trajectory.

Multimodal chat serving. 06-serving-surface.md owns the engine behaviour
(`Gemma4MmTowers::from_model_dir`, `NvEngineChat::supports_mm_input`,
`NV_MM_TOWERS_DIR`, `NV_MM_TOWERS=vision|audio`); what belongs here is its gate.
Media requests take the eager per-request loop on every arm — `mm_present`
excludes them from the batch-engine and speculative-decode paths in
`NvEngineChat::generate`, and `run_sampling_gemma4`'s spec branch additionally
refuses when embeds are present — so text-only requests still reach exactly the
path they reached before mm existed. That is what `rust/tests/gemma4_mm_e2e.rs`
asserts (`NV_GEMMA4_MM_E2E_TEST=1`, plus `NV_GEMMA4_DENSE_DIR` /
`NV_GEMMA4_MOE_DIR`): the same text prompt must answer identically before and
after an image request goes through the same engine, and the image request must
bill materially more prompt tokens than the text one or the soft-token run never
expanded. `NV_GEMMA4_MM_TEXT_BASELINE`, set to a reply recorded on a pre-change
build, turns the within-process check into a byte-identity regression gate;
without it the suite says so on stderr rather than claiming the stronger
property.

## The trap: a green run that executed nothing

**This is the dominant failure mode in this repository, ahead of any actual
bug.** Five distinct mechanisms produce a passing report from a suite that ran
no work:

1. `cfg`'d out. A cross-backend suite is cfg'd on both features
   (`#[cfg(all(feature = "cuda", feature = "wgpu"))]`, as in
   `fp8_contract_e4m3.rs`). Run one under
   `NVK_FEATURES=wgpu` and it compiles to an empty binary that reports zero tests
   passing in no time. That is a skip wearing a pass's clothes. Never use the
   feature override on a CUDA parity gate.
2. Env-gated early return. The `#[ignore]` + `NV_*_TEST` pattern above
   returns before doing anything, and the harness counts the test as passed.
   A non-zero pass count, so it does not even look like a skip.
3. No adapter / no device. Without the Vulkan loader `nvk.sh` wires, every
   wgpu test that requests an adapter returns early and reports ok — the whole
   wgpu surface measured on nothing.
4. Missing dependency guard. `let Some(models) = models() else { return };`
   is invisible in output, and silently skips every suite needing a models
   directory.
5. A fixture that is free. A "heavy" wgpu kernel written as a constant-trip
   affine recurrence has a closed form; the shader compiler reduced it to
   constant work, and the test concluded the profiler was broken.

Detection, in order of cost:

- Read the suite's `#![cfg(...)]` header before believing the result.
- Grep the output for `skip` / `SKIP`. Both the env gates and the adapter checks
  print one.
- Sanity-check the shape of the result: a suite that spawns a process, loads a
  checkpoint or touches a GPU cannot legitimately report a single instant pass.
- Set the require-flags. `NV_KERNELS_PARITY_REQUIRE=1` (which `nvk.sh` sets by
  default) converts "no CUDA device 0" from a silent early return into a
  `panic!` — see the `require()`/`backends()` helpers in
  `crates/nv-kernels/tests/common/mod.rs` and
  `crates/nv-layers/tests/backend_select.rs`. This is the only one of the five
  mechanisms with a mechanical off switch; use it.
- Prefer suites that assert their own inputs, so an empty fixture directory
  cannot pass as "every fixture passes".

And before claiming a fix or a regression: baseline on a reverted tree.
Multiple "regressions" here were tests that had never previously run and only
started executing because of the change under test. The procedure is
`git diff -- <files> > x.patch; git checkout -- <files>; run; git apply x.patch`.

## The conformance corpus

`conformance/` is a language-agnostic fixture corpus shared by the Rust and Go
implementations — the concrete artifact behind the "two servers, one contract"
claim in 01-architecture.md. Layout is flat and numbered: one directory per
fixture at `conformance/fixtures/NNN-<family>-<name>/`, with family-level docs at
`fixtures/README-<NNN>-<family>.md`, the canonical assertion library in `lib/`,
standalone runners in `runner/`, and generators in `tools/` (generators are never
a test-time dependency; artifacts are committed so consuming tests stay
hermetic).

Bands and kinds:

- 001–015, wire-trace. `input.jsonl` (phase ops driving the realtime state
  machine) + `expected.jsonl` (the canonical wire trace) + `README.md` pinning
  spec sections. Compared after canonicalization: session/item/response IDs are
  renumbered in first-appearance order and volatile fields (timestamps, nonces,
  audio bytes) stripped, so traces are comparable across implementations.
- 020 / 030 / 040, endpoint manifests. `fixture.json` with an `endpoint`
  block or a `steps` array, `expected_response`, `comparison_strategy`,
  `ref_outputs`, `skip_when_no_model`. Drivable over HTTP.
- 050 / 060 / 070 / 071, declarative manifests. Consumed in-process by Go and
  Rust tests rather than over the wire — diarization hop sweep, EOU parity,
  OCR and OCR-layout gates.

Both languages locate the corpus by walking up from their package directory to a
sibling `conformance/`: Go via `internal/realtime/conformance_test.go::corpusRoot`,
Rust via `rust/tests/conformance.rs::repo_conformance_root`, which climbs
`CARGO_MANIFEST_DIR`'s ancestors. Crate-local consumers join
`../../../conformance/fixtures` — never a per-family subdirectory.

Running them:

```sh
cd go   && go test ./internal/realtime/ -run TestConformanceCorpus -v
cd rust && cargo test --test conformance
./conformance/runner/run_fixture.py --all
./conformance/runner/run_endpoint_fixture.py
```

The two runners are complementary and are `uv` scripts requiring no project
install. `run_fixture.py` drives the wire-trace corpus through
`lib/trace_invariants.py` directly, optionally restricted to the W-invariant
subset (`W1`…`W8`) with `--strict`; it validates `expected.jsonl` against the
canonical assertions without needing any implementation, which makes it the cheap
gate. `run_endpoint_fixture.py` structurally validates every `fixture.json` in
the corpus: name equals directory name, family matches the numeric band, required
fields present, every path in `input_artifacts` and every `field@` multipart
reference exists on disk, declared WAV headers match. In placeholder mode
declarative fixtures are validated and reported as skipped, since there is
nothing to drive.

`skip_when_no_model: true` marks a fixture that needs model assets; which asset
is named in the fixture's `description`. The 020/030/040 families currently
exist as such placeholders — worth knowing before reading a green corpus run as
end-to-end coverage.

The per-language gates remain authoritative for the wire-trace band only: they
skip directories lacking `expected.jsonl`, so bands 020 and up do not affect
them.

## No comments, and the canonical formatter

The repo convention is no comments by default. `scripts/strip-comments.py`
is the canonical formatter and is run on new files before committing; past agents
wrote dense docstrings and expected-key tables, and their removal is the desired
state, not a loss.

The tool handles `.go`, `.rs` and `.py` (anything else is ignored), and it is
written to be safe rather than clever. It preserves cgo declaration blocks (any
`/* … */` containing `#cgo`, `#include`, or `extern `), all string, char, rune,
raw and byte-string literals, Rust lifetime tokens (`'a`) as distinct from char
literals (`'x'`), nested Rust block comments, the Python shebang and PEP 723
inline-script metadata, and Python docstrings — which are string literals, not
comments. An inline block comment is a token separator, so
`_close_block_comment` re-emits a single space when removing it would glue two
identifier characters together. Runs of blank lines are collapsed.

```sh
nix run .#strip-comments -- <paths...>            # or .#default
nix develop --command python3 scripts/strip-comments.py <paths...>
python3 scripts/strip-comments.py --check .       # exit 1 if anything would change
```

`--check` writes nothing and exits 1 if any file would be modified — the form to
use as a gate. `SKIP_DIRS` excludes `.git`, `target`, `vendor`, `node_modules`,
`.direnv`, `.venv`, `__pycache__`.

Where explanation is genuinely load-bearing, it goes into `docs/` — which is what
this book is — not into the source. The handful of comments that survive in
`flake.nix` and `nvk.sh` are of a specific type: they record a falsified
hypothesis or an upstream bug (why sccache cannot wrap nvcc, why NCCL is disabled
in onnxruntime, why CUTLASS is pinned out of tree, why `:-` failed to override
`CUDA_ARCH_LIST`). That is the bar.
