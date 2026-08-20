# CLAUDE.md -- conventions for contributors and agents

`docs/book/01.4-STATUS.md` is the canonical what's-implemented / what's-open list;
skim it first, never copy it here. Read `docs/book/05.7-apple-silicon-inference-architecture.md`
before any performance or kernel work; `docs/book/08.4-PERFORMANCE.md` governs quoting numbers.

## Repo shape

- `rust/crates/nv-*` -- 18 native crates (kernels, layers, models, engine,
  specdecode, ..., train). `rust/src/` -- Axum 0.8 HTTP binary consuming them.
- `conformance/` -- fixture corpus shared with the Go implementation; most fixtures
  carry `skip_when_no_model: true` -- grep that key before reading green as coverage.
- `flake.nix` pins the model corpus. Dev shells: `<profile>` / `cuda-<profile>`
  plus `default`, `no-models`, `local-hub`, `python`, `cuda`, `cuda-no-models`;
  there is no `.#cuda-hub` or `.#default-hub`.
- `scripts/strip-comments.py` -- canonical formatter; run on new files before commit.
  It deletes `///` and `//!`, not just `//`. `--self-test` asserts exactly what
  survives; `--check` writes nothing.

## Working conventions

- **Never add comments, and know why: nothing in a comment is durable.**
  `strip-comments.py` deletes every comment form a reader would call
  documentation -- `//`, `///` and `//!` alike -- so a comment is a note to
  whoever is looking right now and to nobody after. Writing one is not merely
  against convention, it is writing something with a scheduled deletion date.
  Rationale that must survive goes in one of three places, all of which a build
  or a test reads: a **name**, an **assertion message**, or **`docs/`**. Encode
  restrictions in constants and put the reason in the constant's name --
  `COOP_LADDER_SPEEDUPS_ON_RECORD_ARE_FROM_AN_APPLE_SILICON_ADAPTER` and `RING_REWIND_RESERVED_SLOTS`
  are the pattern; a `///` above `const N = 61` is not, because the next format
  pass removes it and the number is left looking arbitrary.
- The one exception is a doc comment containing a fence cargo builds or runs
  (```` ``` ````, ```` ```no_run ````, ```` ```compile_fail ````): that is a
  **test**, the stripper preserves it, and deleting it would drop coverage no
  test count reports missing. ```` ```text ```` and ```` ```ignore ```` are prose
  and go. Do not reach for a doctest as a way to smuggle prose past the stripper.
- Always optimize for small code.
- Add knobs sparingly. When a new path is proven, remove the code it supersedes.
  CUDA and wgpu are parallel backends, not a ladder: a CUDA improvement
  supersedes only the older CUDA path, and a wgpu improvement supersedes the
  older wgpu path -- deleting the other backend's implementation because yours
  got faster removes the only path that hardware has.
- Edit, don't rewrite: `nv-layers/src/linear.rs`, `nv-quant/src/nvfp4.rs`, and
  `nv-models/src/gemma4.rs` carry perf-critical, hand-tuned invariants.
- Don't claim numerical parity: without the parent Python repo, scaffolds get
  architectural verification only -- say so in commit messages.
- Real-weight tests are `#[ignore]`, cuda-gated, and opt in via `NV_*_TEST=1`.
- No worktrees or branches: work on `main`; isolate concurrent agents by disjoint
  file ownership -- `ls` every path, assert each appears in exactly one list, and
  serialize tasks sharing a file. A file outside your list: stop and report.
- Write gates as "the suite covering <property>" (current name in parentheses); a
  missing artifact is a finding, not a pass -- report a VOID INSTRUCTION.
- **A round leaves a rule, not a record.** Docs state what the system IS, in the
  present tense, as something a reader can act on -- never what a round did, found
  or planned to do. "X was measured at N", "this note records", "before any of
  this is built" are all the changelog voice: rewrite to the rule, or delete the
  page. When a doc's subject ships, the doc becomes a description of the shipped
  thing or it goes; a plan kept past its build is a false map. Prefer deleting a
  record to patching it, and grep for inbound references before you delete.
- **Cite code by symbol, not by line number.** `file.rs:2299` rots within hours
  in this repo and then points at a `}` or, worse, at an unrelated call that
  reads as if it were the subject. Name the function, constant, test or env var
  and give the file without a line. Before quoting any existing `path:line`,
  open it -- a citation you did not check is a claim you did not make. (Line
  numbers are still how you enumerate *published numbers inside documents*, per
  the performance-default rule below.)
- Never flip a performance default. A correctness default may change only to fix a
  demonstrated defect: keep the old path behind an env var, state the cost, and
  list every published number measured on the old path, by document and line.

## Building: always `rust/scripts/nvk.sh`

Never bare `cargo` or per-command `nix develop` for Rust. The wrapper owns the
cached devshell, `TMPDIR`, `CUDA_ARCH_LIST=12.0` (a Blackwell sm_120 target),
per-lane `CARGO_TARGET_DIR`, ccache, and the Vulkan loader wgpu tests need.

```sh
rust/scripts/nvk.sh test --test parity_gdn -- --nocapture    # default lane, cuda,wgpu
NVK_LANE=<lane> rust/scripts/nvk.sh test --test <suite>      # MANDATORY when concurrent
NVK_FEATURES=wgpu rust/scripts/nvk.sh test --test wgpu_rope  # skips all nvcc
NVK_LANE=x NVK_PKG=nv-models rust/scripts/nvk.sh test --test laguna_wgpu_model
```

- Switch crates with `NVK_PKG`, never an appended `-p`: nvk.sh emits its own
  `-p nv-kernels`, so appending runs two packages with wrong feature resolution.
- Feature defaults are per-package: only nv-kernels gets `cuda,wgpu`; every other
  `NVK_PKG` defaults to `cuda` alone. Pass `NVK_FEATURES=cuda,wgpu` explicitly for
  wgpu suites elsewhere; `laguna_wgpu_*` needs `cuda,wgpu,laguna-wip`.
- To run a prebuilt test binary directly (profiler, interposer), first
  `source $HOME/.cache/cargo-tmp/devenv-__cuda.sh`.
- Wait on builds by redirecting to a per-lane build log outside the source tree,
  then ONE `until grep -qE '^(error|    Finished)' <log>; do sleep 15; done` --
  never bare sleep-then-read polls, and never pipe build output into head/tail.
- Non-Rust work uses `nix develop` directly (Go's cgo needs the live shell).
  Prefer `nix develop --offline` unless a flake input actually changed.

## Trusting a run, and machine facts

A red test here may have hidden more reds: bare `cargo test` stops at the first
failing binary, so a crate reported as "276 passed, 2 failed" over 36 binaries
was really 459/3 over 132 -- always pass `--no-fail-fast` for a count
(sweep-workspace.sh already does). A green test here may have run nothing: `cfg`'d-out suites print `0 passed` in
0.00s, env-gated early returns print `1 passed`, and a missing GPU adapter or
models dir skips silently -- grep output for `skip`, read the suite's `#![cfg]`
header, check count and elapsed time are plausible. A test whose oracle is the
implementation, or whose bound is derived from the value it bounds, survives
every mutation -- ask what the reference IS before reading green as coverage.
Baseline before claiming a fix or a break. A number without its basis tuple (checkpoint, harness, backend,
batch, token count, sha, log path) is not a number. `NV_DRAFTER=dflash` runs
non-speculative if `NV_DFLASH_DRAFT_DIR` is unset. Discover model ids from
`/v1/models`; idle-gate on `memory.used` and `uptime`, not `utilization.gpu`.
Keep build artifacts and logs out of the source tree.
Reuse `NVK_LANE` names -- stale lane target dirs dominate disk use. The cached
cuda devenv exports `NV_CHAT_MODEL_DIR`/`HF_HUB_CACHE` over caller env (verify the
`loading ... from` boot line); check `model-*.safetensors` exists before planning
around a checkpoint.
