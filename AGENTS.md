# AGENTS.md -- context for future sessions

Read `docs/book/05.7-apple-silicon-inference-architecture.md` before any
performance or kernel work. Measured numbers live outside this repository;
prose states the relationship a measurement evidenced, never the figure.

## Repo shape

- `rust/crates/nv-*` -- 18 native crates (kernels, layers, models, engine,
  specdecode, ..., train); `rust/src/` -- the Axum 0.8 HTTP binary over them.
- `conformance/` -- fixtures shared with the Go implementation; most carry
  `skip_when_no_model: true` -- grep that key before reading green as coverage.
- `flake.nix` pins the model corpus. Dev shells: `<profile>` / `cuda-<profile>`,
  `default`, `no-models`, `local-hub`, `python`, `cuda`, `cuda-no-models`; there
  is no `.#cuda-hub` or `.#default-hub`.
- `scripts/strip-comments.py` -- canonical formatter; run it on new files before
  commit. `--self-test` asserts exactly what survives; `--check` writes nothing.

## Working conventions

- **Never add comments.** `strip-comments.py` deletes `//`, `///` and `//!` alike.
  Rationale that must survive goes where a build or test reads it: a **name**, an
  **assertion message**, or **`docs/`**. Put a restriction's reason in its
  constant's name (`COOP_LADDER_SPEEDUPS_ON_RECORD_ARE_M2_ULTRA`,
  `RING_REWIND_RESERVED_SLOTS`), not in a `///` above `const N = 61`.
- Exception: a doc comment with a fence cargo builds or runs (```` ``` ````,
  ```` ```no_run ````, ```` ```compile_fail ````) is a **test**; the stripper keeps
  it and deleting it drops coverage. ```` ```text ```` and ```` ```ignore ```` are
  prose and go. Never use a doctest to smuggle prose past the stripper.
- Always optimize for small code.
- Add knobs sparingly. When a new path is proven, remove the code it supersedes
  -- in its own backend only: CUDA and wgpu are parallel backends, and deleting
  the other backend's path removes the only one that hardware has.
- Edit, don't rewrite: `nv-layers/src/linear.rs`, `nv-quant/src/nvfp4.rs` and
  `nv-models/src/gemma4.rs` carry perf-critical, hand-tuned invariants.
- Don't claim numerical parity: without the parent Python repo, scaffolds get
  architectural verification only -- say so in commit messages.
- Real-weight tests are `#[ignore]`, cuda-gated, and opt in via `NV_*_TEST=1`.
- No worktrees or branches: work on `main`. Isolate concurrent agents by disjoint
  file ownership -- `ls` every path, assert each is in exactly one list, serialize
  tasks sharing a file. A file outside your list: stop and report.
- Write gates as "the suite covering <property>" (current name in parentheses); a
  missing artifact is a finding, not a pass -- report a VOID INSTRUCTION.
- **A round leaves a rule, not a record.** Docs state what the system IS, in the
  present tense, as something a reader can act on -- never what a round did,
  found or planned. "X was measured at N" and "this note records" are changelog
  voice: rewrite to the rule or delete the page. When a doc's subject ships, the
  doc describes the shipped thing or goes. Prefer deleting a record to patching
  it; grep for inbound references first.
- **Cite code by symbol, not line number** (`file.rs:2299` rots within hours):
  name the function, constant, test or env var, and the file. Open any existing
  `path:line` before quoting it. (Published numbers inside documents are still
  enumerated by line, per the next rule.)
- Default flips follow the bitwise-or-instrument law. A performance default may
  flip only when the change reproduces the family's canonical trajectory or
  output bitwise with its suites green; a change that reassociates stays opt-in
  until a named corpus-level instrument (the markup-neutral CER scorecard, the
  twelve-zeros stamp, word-diff adjudication) blesses it, and the flip's commit
  names the instrument. A correctness default changes only to fix a demonstrated
  defect. Either way the change ships in place: delete the superseded path and
  any env var that would restore it in the same change -- git is the rollback.
  The commit states the cost and lists every published number measured on the
  old path, by document and line. No env var exists to bring back old behavior.

## Building: always `rust/scripts/nvk.sh`

Never bare `cargo` or per-command `nix develop` for Rust. The wrapper owns the
cached devshell, `TMPDIR`, `CUDA_ARCH_LIST=12.0` (single Blackwell sm_120 box),
per-lane `CARGO_TARGET_DIR`, ccache, and the Vulkan loader wgpu tests need.

```sh
rust/scripts/nvk.sh test --test gdn_scan_w32 -- --nocapture   # default lane, cuda,wgpu
NVK_LANE=<lane> rust/scripts/nvk.sh test --test <suite>      # MANDATORY when concurrent
NVK_FEATURES=wgpu rust/scripts/nvk.sh test --test wgpu_rope  # skips all nvcc
NVK_LANE=x NVK_PKG=nv-models rust/scripts/nvk.sh test --test laguna_wgpu_moe
```

- Switch crates with `NVK_PKG`, never an appended `-p`: nvk.sh emits its own
  `-p nv-kernels`, so appending runs two packages with wrong feature resolution.
- Feature defaults are per package: only nv-kernels gets `cuda,wgpu`; every other
  `NVK_PKG` defaults to `cuda`. Pass `NVK_FEATURES=cuda,wgpu` for wgpu suites
  elsewhere; `laguna_wgpu_*` needs `cuda,wgpu,laguna-wip`.
- To run a prebuilt test binary directly (profiler, interposer), first
  `source $HOME/.cache/cargo-tmp/devenv-__cuda.sh`.
- Wait on builds by redirecting to a per-lane log outside the source tree, then
  ONE `until grep -qE '^(error|    Finished)' <log>; do sleep 15; done` or
  `run_in_background`. Hooks deny sleep-then-read polls and piping builds into
  head/tail.
- Non-Rust work uses `nix develop` directly (Go's cgo needs the live shell);
  prefer `--offline` unless a flake input changed.

## Trusting a run, and machine facts

- A red may hide more reds: bare `cargo test` stops at the first failing binary
  ("276 passed, 2 failed" over 36 binaries was really 459/3 over 132) -- pass
  `--no-fail-fast` for a count (sweep-workspace.sh does).
- A green may have run nothing: `cfg`'d-out suites print `0 passed` in 0.00s,
  env-gated early returns print `1 passed`, and a missing GPU adapter or models
  dir skips silently -- grep output for `skip`, read the suite's `#![cfg]`, check
  count and elapsed time. A test whose oracle is the implementation, or whose
  bound derives from the value it bounds, survives every mutation -- ask what the
  reference IS before reading green as coverage.
- Baseline before claiming a fix or a break. A number without its basis tuple
  (checkpoint, harness, backend, batch, token count, sha, log path) is not a number.
- `NV_DRAFTER=dflash` runs non-speculative if `NV_DFLASH_DRAFT_DIR` is unset.
  Discover model ids from `/v1/models`; idle-gate on `memory.used` and `uptime`,
  not `utilization.gpu`.
- Keep build artifacts and logs out of the source tree. Disk truth is `zpool list`
  + `zfs get usedbysnapshots`, never `df`; never destroy snapshots unasked.
- Reuse `NVK_LANE` names -- stale lane target dirs dominate disk; audit with
  `nvk.sh lanes`, reclaim manually via `nvk.sh lanes --reap DAYS [--dry-run]`.
- nvk.sh restores caller-set `NV_CHAT_MODEL_DIR(S)`/`NV_WGPU_CHAT_MODEL_DIRS`/
  `HF_HUB_CACHE` over the cached devenv; sourcing `devenv-__cuda.sh` directly still
  overrides caller env (verify the `loading ... from` boot line). Check
  `model-*.safetensors` exists before planning around a checkpoint.
