# Kernels and Quantization

The numerical core is three crates, stacked, with dependencies pointing one way
(`nv-layers` → `nv-quant` → `nv-kernels`):

- `rust/crates/nv-kernels` — the CUDA (and WGSL, and a partial HIP) source tree,
  a hand-written C header, and the generated FFI surface. No policy.
- `rust/crates/nv-quant` — number formats, their host reference encoders, and the
  cuBLASLt / CUTLASS GEMM runners. Knows what a scale means.
- `rust/crates/nv-layers` — `Linear`, `Mlp`, `MoeBlock` and the grouped MoE
  dispatch. Decides which kernel runs for a given shape.

A model in `nv-models` never calls a kernel entry point for a plain projection;
it calls `Linear::forward` and the storage variant picks the path. It *does* call
kernel entry points directly for anything fused or capture-sensitive, which is
most of the decode loop.

## How build.rs compiles the .cu sources

`nv-kernels/build.rs` does two independent jobs.

**Bindings, always.** `emit_bindings` runs unconditionally — even with no CUDA
feature and no toolkit — running bindgen over `include/nv_kernels.h` with
`allowlist_function("nv_kernels_.*")` and `layout_tests(false)` into
`$OUT_DIR/bindings.rs`. `src/lib.rs` includes that as `pub mod sys` under
`cfg(any(feature = "cuda", feature = "rocm"))`. The header is hand-written and is
the authority; bindgen only transcribes it.

**Device code, only under `cuda`.** `build_cuda` first removes `CXXFLAGS`,
`CFLAGS`, `HOST_CXXFLAGS`, `HOST_CFLAGS` and their target-suffixed forms from the
environment, so the `cc` crate cannot forward host-compiler flags into `nvcc`. It
then locates the toolkit from `CUDA_PATH`, falling back to `/usr/local/cuda` and
`/opt/cuda`. `NVK_NVCC_WRAPPER` optionally interposes a launcher: build.rs writes
an `sh` shim into `$OUT_DIR/nvcc-launcher/nvcc` that execs `<launcher> <nvcc>
"$@"` and hands *that* to `cc::Build::compiler`, so a caching or distributing
wrapper drops in without the `cc` crate seeing a different compiler identity. A
launcher not on `PATH` emits a `cargo:warning` and uses `nvcc` directly.

`CUDA_ARCH_LIST` defaults to `"8.9;12.0"`, splits on `;` or `,`, and each entry
becomes one `-gencode=arch=compute_<A>,code=sm_<A>`. It controls exactly one
thing: which device architectures every `.cu` in `cuda/` is compiled for. Each
extra entry is a full extra device compilation of every translation unit, which
is why `rust/scripts/nvk.sh` narrows it locally (`08-build-and-testing.md`).

**Two compilation units.** `cuda/` plus `cpp/` compile into `libnv_kernels.a`
with `-std=c++17`, `-Xcompiler=-fPIC`, `-Xcompiler=-fno-strict-aliasing`,
`--expt-relaxed-constexpr`, `--expt-extended-lambda` and the `cuda/marlin` +
`cuda/marlin/generated` include dirs. `cuda_sm120/` compiles separately into
`libnv_kernels_sm120.a`, and only when `120` appears in the arch list, with
`-gencode=arch=compute_120a,code=sm_120a`, the `120f` variant, and
`-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1`. The separation is forced by content: those
files instantiate CUTLASS/FlashInfer block-scaled FP4 templates targeting the
arch-conditional `a`/`f` architectures, which the generic arch list cannot
express.

**Where cutlass and flashinfer come from.** `collect_includes` assembles the `-I`
list from environment variables, never from a vendored copy: `CUTLASS_DIR` →
`include/` and `tools/util/include/`; `FLASHINFER_DIR` → `include/` plus its
bundled `3rdparty/cutlass/tools/util/include` when present; `CUDNN_FRONTEND_DIR`,
`CUDNN_ROOT`, `NCCL_ROOT` → their `include/`. `flake.nix` sets all of these from
pinned nixpkgs derivations and derives `CUDA_ARCH_LIST` and
`CMAKE_CUDA_ARCHITECTURES` from one `cudaCapabilities` list, so the CUTLASS
version is a flake input, not a submodule. `cuda/cutlass_probe.cu` is a
deliberately trivial translation unit whose only purpose is to fail the build
loudly if those include paths are unresolved, without paying template
instantiation cost. Linking emits `cudart`, `cublas`, `cublasLt`, `cudnn`,
`nvrtc` and `nccl` as dylibs; `Cargo.toml` declares `links = "nv_kernels"`.

**What build.rs does not do.** It watches `hip/` and re-runs on `ROCM_PATH`,
`HIP_PATH`, `ROCM_ARCH_LIST` and `ROCM_DEVICE_LIB_PATH` — but there is no HIP
compilation path. The `rocm` feature only selects the bindgen `sys` module; the
`.hip.cpp` files are not compiled by this script. See `05-backends.md` and
`05.5-rocm-port-status.md`.

## The FFI surface

`include/nv_kernels.h` declares every entry point as
`int nv_kernels_<name>(void* stream, ...)` returning `0` on success. Two
conventions explain most of the signatures:

**The stream is always an explicit argument.** There is no implicit or global
stream. A launch on the legacy null stream cannot participate in CUDA graph
capture, which is why several kernels exist at all.

**Sizes and positions that change per step arrive as device pointers.** `pos`,
`n_total_dev`, `n_committed`, `start_dev`, `cache_pos`, `ring_meta`, `pos_base`
are `const int*` into device memory, not host `int`s; the kernel reads the value
at entry. That is what lets one captured graph replay across decode steps with a
growing context: the host updates the device int between replays instead of
re-capturing. `nv_kernels_attention_fp8_decode` takes both `n_total_dev` (read by
the kernel) and `max_total` (a host upper bound sizing the dynamic shared-memory
allocation) precisely because one is per-replay and the other is per-capture.

`src/lib.rs` wraps every generated declaration in `pub mod cuda` with an
`unsafe fn` of the same shape. A second `pub mod cuda` under
`cfg(not(any(feature = "cuda", feature = "rocm")))` provides identical names
returning `-1` / `Err(-1)`, so downstream crates compile without a GPU toolchain
and fail at runtime rather than at type-check time. `src/graph.rs` holds
`CudaGraphRunner`, which caches `CUgraph`/`CUgraphExec` pairs by key and exposes a
process-wide `capture_lock()`; capture is not re-entrant across threads.

## LinearKind

`nv_quant::LinearKind` is the tag; `nv_layers::linear::LinearStorage` is the
payload. Four variants are declared; three are constructed.

**`Bf16`** — `LinearStorage::Bf16 { weight, weight_t }`, where `weight_t` is a
pre-transposed contiguous copy built by `Linear::new`.
`Linear::new_no_pretranspose` skips it: the differentiable training path reads
only `weight`, and on a large dense model the transposed copy doubles the
resident bf16 footprint. `matmul_bf16` (`nv-layers/src/linear.rs`) dispatches:
non-CUDA or non-bf16 input → `matmul_fallback` (candle matmul, f32 upcast on
CPU); `leading == 1`, `in_features` even, not forced dense → hand-written
`nv_kernels_gemv_bf16`; `weight_t` present →
`TensorCoreGemm::bf16_matmul_row_major_offs`; 2..=16 rows with `NV_BF16_SMALLM_DET`
(default on) → `bf16_matmul_row_major_bt_det_offs`, the untimed deterministic
algo; otherwise `bf16_matmul_row_major_bt_offs`. `forward_rows` and
`forward_dense_det` are narrower entry points over the same storage: a row-window
GEMM (LoRA-windowed and vocabulary-sliced calls) and an always-deterministic one.

**`Fp8E4m3 { a_scale, b_scale }`** — W8A8 e4m3. `Fp8Storage` holds the weight
bytes, a per-output-row f32 scale vector on device (`b_scale_rows_dev`), its
per-tensor collapse (`b_scale_dev`, the max over rows — exactly the scale a
whole-tensor quantization would produce, since each row scale is `row_amax/448`),
and an `Fp8GemmRunner`. At GEMM level this is cuBLASLt with scale pointers:
`matmul_e4m3_row_scaled` for per-row (cuBLASLt `OUTER_VEC_32F`) and
`matmul_e4m3_weight_row` for per-tensor, reconstructing
`value ≈ f32(e4m3_byte) * scale`.

`matmul_fp8` quantizes the activation *on the host* every call — copies `x` to
host, runs `quantize_e4m3_per_row`, uploads bytes and scales — which makes it
unsuitable for graph capture, and is why `nv_kernels_quantize_fp8_pertensor_bf16`
and `nv_kernels_gemv_e4m3_w8a8` exist as the device-side capturable
counterparts. Granularity policy lives in `fp8_scale_mode_from`: per-output-row
is the default; `NV_FP8_SCALE_MODE=tensor|per_tensor|per-tensor|0` selects one
scale for the whole weight, and that path *refuses* a checkpoint whose stored
`weight_scale` varies across rows rather than silently coarsening it
(`fp8_weight_payload`). Checkpoint-supplied scales are honoured verbatim, never
recomputed from the bf16 amax.

Per-row construction probes cuBLASLt at load (`probe_per_row_scale_support`, one
`cublasLtMatmulAlgoGetHeuristic` call, cached per device ordinal) and **refuses
at boot** where the per-row `OUTER_VEC_32F` mode is unserved — which is every
shape on this box's SM 12 (`fp8_shape_probe` asserts hardware agreement in both
directions) — instead of booting clean and dying on the first request. The
refusal names `NV_FP8_SCALE_MODE=tensor` as the escape. **The two refusals
compose**: a row-varying checkpoint on a per-row-unserved device cannot run fp8
at all here, and `supports_fp8()` (arch major only) is NOT the capability
predicate.

**`Nvfp4`** — W4A4 block-scaled FP4. `Nvfp4Storage` holds packed e2m1 nibbles,
the swizzled ue4m3 block scales, a host and device copy of `weight_alpha`, the
checkpoint's `input_stored_global`, an `Nvfp4GemmRunner`, and a
per-(stream, m_padded) staging cache for quantized activations. At GEMM level
this is either the CUTLASS SM120 block-scaled FP4 kernel or cuBLASLt's
`CUDA_R_4F_E2M1` + `VEC16_UE4M3` mode; both take the same operand layout — A
row-major packed FP4, B column-major packed FP4, two ue4m3 scale buffers in the
interleaved NVFP4 layout, one f32 global scale folded in as alpha, bf16 output.

**`Mxfp4` is declared and never constructed.** There is no
`LinearStorage::Mxfp4`; `LinearKind::Mxfp4` appears only in `match` arms.
`nv-quant/src/mxfp4.rs` is a complete host-side reference (e8m0 scale codec,
32-element blocks, 16-byte packing, `from_gpt_oss_row_major` ingest, CPU matmul)
exercised by `nv-quant/tests/mxfp4.rs` and by WGSL kernels, but there is no CUDA
MXFP4 GEMM and no way to build an MXFP4 `Linear`.

## QuantScheme: two enums, one live

`nv_quant::QuantScheme` has eight variants (`None`, `Fp8E4m3`, `Fp8E5m2`,
`Nvfp4`, `Mxfp4`, `AwqInt4`, `GptqInt4`, `Marlin`), its only producer is
`LinearKind::scheme()`, that method has **zero call sites in the workspace**, and
four of the eight are unreachable from any `LinearKind` even in principle. It is
a declared vocabulary with no consumer.

`nv_weights::QuantScheme` has three (`None`, `Fp8E4m3`, `Nvfp4`) and is the one
constructed and matched — by checkpoint sniffing in `nv-weights/src/lib.rs`, by
`Linear::from_quantized_weight_fp8`, by `MoeBlock::from_loader_quantized`, and by
the loaders in `nv-models/src/{gemma4,qwen3,laguna}.rs`.
`implemented_group_size` returns `Some(16)` for `Nvfp4` and `None` otherwise; a
non-16 NVFP4 group size is a hard error, because 16 is baked into the Rust
reference, the CUDA quantize kernels and cuBLASLt's `VEC16_UE4M3` mode alike.
Int4 exists as a kernel path (Marlin, below) but is reached through
`gemma4_e4b.rs`'s own `QLinear` enum, not through either `QuantScheme`.

## NVFP4: the format

`nv-quant/src/nvfp4.rs`. `BLOCK_SIZE = 16`, `MIN_TILE = 128`. NVFP4 is NVIDIA's
format, distinct from OCP MXFP4 (`05-backends.md`): 16-element blocks rather than
32, an E4M3 block scale rather than E8M0, and a second per-tensor FP32 scale
layered on top — NVIDIA, ["Introducing NVFP4 for Efficient and Accurate
Low-Precision Inference"](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)
(2025-06-24). The E4M3 byte layout is the OCP 8-bit spec, same as `fp8.rs`'s
E4M3 (`04.1-fp8.md` §7.1); NVFP4's block scale is the *unsigned* variant.

**Elements** are e2m1: sign bit plus the magnitude codebook
`{0, 0.5, 1, 1.5, 2, 3, 4, 6}`, two nibbles per byte, low nibble first
(`pack_e2m1_pair`), so a row of `K` values occupies `K/2` bytes.

**Block scales** are ue4m3 — unsigned e4m3, one per 16 elements. Biased exponent
0 is the subnormal encoding (`mantissa * 2^-9`, no implicit leading one), min
normal is `2^-6`, the value clamps at 448, and `0x7F` is remapped to `0x7E`. The
subnormal branch is the part that is easy to get wrong: `encode_ue4m3` /
`decode_ue4m3` are the oracle, the CUDA `encode_ue4m3_dev` / `decode_ue4m3_dev`
in `cuda/quantize_nvfp4_bf16.cu` mirror them (using `frexpf`/`ldexpf` rather than
`log2f`, which rounds up near powers of two), and `nv-quant/tests/ue4m3_fuzz.rs`
checks every byte against the device. `nv-quant/src/fp8.rs`'s
`tests::device_decode_ue4m3_matches_rust_oracle_*` guard the same branch a second
way, because `quantize_nvfp4_bf16.cu` cannot be compiled from a Rust test binary
at all: the device helpers are transliterated into Rust, cross-checked against
the oracle, plus a source-text check that `decode_ue4m3_dev` still branches on
`exp==0`. Without that branch byte `0x02` decodes ~2.5x high
(`(1 + 2/8) * 2^-7` instead of `2 * 2^-9`) — silently, since the pre-fix encode
and decode agreed with each other and only disagreed with the hardware.

**Two-level scaling**, `quantize_block_with_global`:

```
local_scale  = amax / 6                 (amax over the 16 values)
stored_scale = stored_global * local_scale
scale_byte   = encode_ue4m3(stored_scale)
inv          = stored_global / decode_ue4m3(scale_byte)
nibble[i]    = encode_e2m1(clamp(v[i] * inv, -6, 6))
```

The `inv` term divides by the *re-decoded* scale, not the ideal one, so the
encoder's own rounding is compensated in the elements. The effective value the
GEMM contracts against is
`decode_e2m1(nibble) * decode_ue4m3(block_scale) * weight_alpha`, with
`weight_alpha = 1/(stored_weight_global * stored_input_global)` — exactly what
`dequantize_packed_linear` / `dequantize_packed_swizzled` compute and what
`Linear::dequant_weight` returns.

**Swizzle.** CUTLASS and cuBLASLt both want block scales tiled, not row-major.
`swizzle_scales` maps `(m, kb)` to
`((m/128 * k_tiles + kb/4) * 32 + m%32) * 16 + ((m/32)%4) * 4 + kb%4`, padding to
whole 128-row × 4-block tiles; `unswizzle_scales` is its exact inverse and is
what `dequant_weight` uses. The device quantize kernels write the swizzled layout
directly. The one exception is `nv_kernels_nvfp4_quantize_row_bf16`, which writes
a flat layout because its only consumer is `nv_kernels_nvfp4_gemv_bf16` — and
that CUDA M=1 GEMV pair is referenced only by parity tests and mirrored by a WGSL
port in `src/wgpu_backend/kernels/gemv_nvfp4.rs`; the CUDA serving path does not
call it.

## Where quantized weights come from

Three routes into `LinearStorage::Nvfp4`, all in `nv-layers/src/moe.rs` and
`linear.rs`.

**From disk.** `nvfp4_linear_from_disk_inner` reads four tensors named by a
`Nvfp4Suffixes` constant, because the two checkpoint families disagree:

| | compressed-tensors (Qwen) | modelopt (Gemma) |
|---|---|---|
| packed nibbles | `weight_packed` | `weight` |
| block scales | `weight_scale` | `weight_scale` |
| weight global | `weight_global_scale` | `weight_scale_2` |
| activation global | `input_global_scale` | `input_scale` |
| global is inverse | yes | no |

`global_scale_is_inverse` reconciles the two: modelopt stores the reciprocal of
what compressed-tensors stores, so one family gets a `safe_recip`. Raw scale
bytes are swizzled on the host and uploaded; `weight_alpha` is the product of the
two reciprocals. `nvfp4_linear_from_disk_fused_pair` concatenates gate and up
into one `Linear`, refusing if their global scales differ and requiring
`out_features_each % 128 == 0` so the swizzled scale blocks concatenate cleanly.
Modules whose dimensions fall below `MIN_TILE` are dequantized to bf16 instead
(`dequantize_nvfp4_to_bf16`) — there is no small-tile NVFP4 GEMM.

**On device from bf16.** `from_bf16_quantized_nvfp4_dev` computes the amax, sets
`stored_weight_global = 448*6/amax`, and runs `nv_kernels_quantize_nvfp4_bf16`.

**On host from bf16.** `from_bf16_quantized_nvfp4` builds an `Nvfp4Tensor` and
uploads `data` + `scales_swizzled()`. Slower, but it is the same arithmetic as
the reference, which makes it the fixture for parity tests.

Activations are quantized per call, never stored: `matmul_nvfp4_impl` runs
`quantize_nvfp4_bf16`, `..._rows`, or the fused `rmsnorm_quantize_nvfp4_bf16`
immediately before the GEMM.

## The NVFP4 GEMM runner

`Nvfp4GemmRunner` (`nv-quant/src/nvfp4.rs`, `mod cuda`) owns a cuBLASLt handle, a
descriptor cache keyed on `(m, n, k, a_scale_ptr)`, and a per-stream 64 MiB
workspace.

**Workspace lifetime is a capture constraint.** `workspace_handle` refuses to
allocate while the stream is capturing and tells the caller to run
`ensure_workspace_for_stream` first. `release_stream_resources` drops the
workspace and the handle cache *and* bumps a per-stream epoch, because CUDA
reuses stream addresses: `Nvfp4Storage::a_staging` is keyed on
`(cu_stream as usize, m_padded)` and each entry records its epoch, so a recycled
stream pointer cannot resurrect a buffer belonging to a destroyed stream.

**Backend selection.** `SPEACHES_NVFP4_BACKEND` = `cublaslt` | `cutlass` | `auto`
(default). `cutlass_supports_shape` requires `m ≤ 128 or m % 128 == 0`,
`n % 128 == 0`, `k % 128 == 0` — the SM120 tile is 128×128×128. Auto uses CUTLASS
when the shape fits and cuBLASLt otherwise; the cuBLASLt path requires `m ≥ 128`,
which is why `matmul_nvfp4_impl` pads `m_padded = max(m, 128)` unless
`NV_NVFP4_TRUE_M` is set *and* the shape is CUTLASS-eligible.

**Padding avoidance.** When `m_padded > m_logical` the runner takes a "skip pad"
route: a zero-initialised staging buffer per `(stream, m_padded)`, quantizing
only the live rows with `quantize_nvfp4_bf16_rows` and re-zeroing when the row
high-water mark drops (`hwm_rows`). `NV_NVFP4_QUANT_FULLPAD` disables it.

**Kernel selection inside CUTLASS.** `cuda_sm120/cutlass_fp4_gemm.cu`
instantiates FlashInfer's `genericFp4GemmKernelLauncher` and
`genericFp4GemmKernelLauncherStreamK` for bf16 output at three tiles —
128×128×128, 128×128×256, 128×256×128.
`nv_kernels_cutlass_fp4_gemm_sm120_bf16` is the plain 128³ entry, `_streamk` its
Stream-K sibling, `_tiled` multiplexes `(tile, stream_k)` and exposes a forced
S-way split-K that sets `args.scheduler.splits` directly, leaving the reduction
mode deterministic. `cutlass_launch_impl` picks between them from
`NV_NVFP4_STREAMK` (never/always/auto), `NV_NVFP4_K256`,
`NV_NVFP4_STREAMK_K256` and `NV_NVFP4_STREAMK_DOWN_SPLITS`, plus a
`NV_NVFP4_SKINNY_LT` reroute sending narrow shapes back to cuBLASLt. **A Stream-K
launch failure sets `streamk_failed` and is never retried for the lifetime of the
runner.**

**Fused pre-norm.** `Linear::prenorm_nvfp4_eligible` / `forward_prenorm_nvfp4`
take the *pre*-norm activation and the RMSNorm gain and call
`nv_kernels_rmsnorm_quantize_nvfp4_bf16`, which normalizes in shared memory and
emits packed FP4 + swizzled scales without the normed row ever reaching HBM. It
is byte-identical to `rmsnorm_bf16` followed by `quantize_nvfp4_bf16`, and the
kernel returns non-zero when `K*2` bytes exceeds the portable 48 KiB dynamic-smem
limit so the caller falls back to the unfused pair. Eligibility also requires no
bias and no attached LoRA.

## FP8, INT8 and INT4 outside the Linear path

`nv-quant/src/fp8.rs` defines `E4M3_MAX = 448.0` and `scale = amax/448`, with
`Fp8ScaleMode::{PerTensor, PerOuterRow}` (per-row default) and host encoders
`quantize_e4m3_per_tensor`, `quantize_e4m3_per_row`,
`quantize_e4m3_with_row_scales`; `cpu_e4m3_matmul_row_scaled` is the host oracle
for the device GEMM. The cross-backend fp8 contract is `04.1-fp8.md`.

Device counterparts in `nv-kernels`: `nv_kernels_rowquant_e4m3` (per-output-row
absmax bf16 → e4m3 bytes plus f32 row scales);
`nv_kernels_quantize_fp8_pertensor_bf16` (bit-identical device counterpart of the
host per-tensor quantize, writing the folded `a_scale` to device memory);
`nv_kernels_gemv_e4m3_mk_h` (M=1..16 e4m3-weight GEMV with the activation RMSNorm
fused into the staging and a per-row dequant epilogue); and
`nv_kernels_gemv_e4m3_w8a8` (capturable per-tensor W8A8 GEMV, both operands
e4m3, fp32 accumulate — explicitly a graph-replay-safe drop-in for the cuBLASLt
fp8 GEMM).

INT8 is the same shape one step coarser: `nv_kernels_rowquant_i8` (per-row
absmax) feeding `gemv_i8_normed`, `gemv_i8_normed_mk` and `gemv_i8_mk_h`.
`nv_kernels_gemv_i8_normed_mk_max_m(K)` reports the largest M a single launch
supports at a given K, or 0 if K is unsupported, so the caller can decide whether
to chunk. These are reached from Laguna's lm_head (`nv-models/src/laguna.rs`,
gated on `NV_LAGUNA_LMHEAD_INT8`, e4m3 variant as fallback), not from `Linear`.

INT4 w4a16 has two implementations: `nv_kernels_gemv_w4a16`, a plain row-major
packed-int4 GEMV over `[N, K/8]` u32 with `[N, K/GS]` bf16 group scales plus a
`_gelu_pli` variant fusing Gemma's PLE gate epilogue; and the vendored vLLM
Marlin GEMM in `cuda/marlin/` (`cuda/marlin/INTEGRATION.md`) — compressed-tensors
w4a16, symmetric 4-bit offset-binary nibbles (`stored = signed + 8`),
`group_size = 32`, no zero-points, no act-order. Using Marlin requires a repack
(`nv_kernels_marlin_repack_w4a16`, `[k/8, n]` → the Marlin tile layout) and a
workspace of `nv_kernels_marlin_workspace_elems()` int32s zeroed once; a
`_prezeroed` variant exists for callers that already zero the output, and
`nv_kernels_multi_zero_bf16` zeroes many scattered regions in one launch to make
that cheap. `nv-models/src/gemma4_e4b.rs` chooses between the two per shape
(Marlin needs `in % 32 == 0` and `out % 64 == 0`), with `NV_E4B_FORCE_GEMV` to
pin the choice.

## The MoE path

Routing and dispatch exist at three levels of specialization.

**Reference: `MoeBlock::forward`** (`nv-layers/src/moe.rs`). Gate projection →
f32 → `sort_last_dim` → take top `k` → softmax over the winners only (so weights
sum to one over the selected experts, not over all of them). Then a host loop:
bucket rows by expert, `index_select` the rows, run the expert's `Mlp`, scale by
the routing weight, `index_add` back; the shared expert is added on top. This
path pulls `top_idx` and `top_weights` to the host and checks every routed id
against `num_experts` with an error message naming the actual failure mode — a
router output read during graph capture is garbage, and the check exists so it
surfaces as a bounds error rather than an out-of-range weight read. The shared
expert is a plain `Mlp` plus a one-row gate `Linear`; `shared_contribution`
computes `sigmoid(gate_logits) * shared_out` in candle and
`shared_contribution_device` replaces that pair with
`nv_kernels_mul_sigmoid_rowgate_f32` so the step is capturable.

**Device routing.** `cuda/moe_route.cu` implements `nv_kernels_moe_route_topk`
with two modes: **mode 0** selects by raw logits with weights = softmax over the
K winners; **mode 1** computes `scores = sigmoid(logits)`, selects by
`score + bias[e]` (the per-expert selection bias may be null), and takes weights
from the *unbiased* scores, optionally sum-normalized, times `routed_scaling`. An
optional `softcap` applies `softcap * tanh(x/softcap)` to the logits first. The
kernel is one block per token with expert scores in shared memory.

`cuda/moe_permute.cu` implements the sort into expert order as three kernels —
count, single-block exclusive scan, assign — **deliberately not fused**, because
counting needs cross-block atomics, the scan needs one block to see the whole
count vector, and assign needs the scan's result before it can atomically bump
per-expert cursors. The count buffer is reused as the cursor buffer.

**Grouped NVFP4 experts** (`nv-layers/src/moe_grouped.rs`).
`MoeGroupedWeights::build_from_experts` reads each expert's NVFP4 parts back to
host and concatenates them into one buffer per projection (`gate_w`, `up_w`,
`down_w`, their swizzled scale buffers, per-expert `alphas`), keeps each expert's
`input_global_scale` on host and device, and rejects an expert whose gate and up
disagree on it. After this the whole expert bank is three contiguous allocations
addressable by expert index. `forward_grouped` (prefill-shaped) then:
`plan_routing` buckets `(token, slot)` pairs by expert on the host and lays out
one `MIN_TILE = 128`-row slot per *active* expert (`src_idx` is the gather index
with `-1` for pad rows, `inv_perm` its inverse); `gather_rows_bf16` materialises
the sorted padded activation with `-1` rows zero-filled;
`quantize_nvfp4_bf16_per_expert` quantizes each 128-row tile with *that* expert's
`input_global_scale`; three grouped CUTLASS GEMMs run through
`grouped_gemm_chunked` at most `MAX_A_PER_GROUPED_CALL = 8` experts per launch;
`silu_mul_quantize_nvfp4_bf16_per_expert` fuses `SiLU(gate) * up` with
re-quantization so the intermediate never lands in HBM as bf16; and
`moe_unpermute_scatter` applies the routing weights and scatters back to token
order. `forward_grouped` refuses if any expert received more than 128 rows —
`MoeBlock::try_forward_grouped` checks that first — and the whole grouped path is
opt-in behind `SPEACHES_MOE_GROUPED`.

**Slot-batched decode.** `GroupedDecodeContext` is the capturable form: at
construction it fixes `n_tokens * k` tiles (capped at
`MAX_TILES_PER_GROUPED_CALL = 256`), allocates every intermediate once, and
builds a *constant* `src_idx`/`inv_perm` where tile `t` reads token `t / k` and
writes rows starting at `t * 128`. Nothing about the layout depends on which
experts were routed to — only the weight pointers do, and those are selected on
device by `active_expert_indices`. `forward_grouped_decode` therefore runs
entirely on device: route → gather each slot's `input_global_scale` by routed id
(`gather_f32_by_ids`) → `gather_rows_bf16_strided` → quantize → three grouped
GEMMs → `silu_mul_quantize_..._strided` → scatter. The `_strided` variants exist
because at M=1 only the first row of each 128-row tile is live; they launch one
block per tile and leave pad rows untouched.

`grouped_gemm_decode` can select
`nv_kernels_cutlass_moe_grouped_fp4_gemm_sm120_bf16_decode`, gated on
`NV_MOE_FP4_DECODE_TILE` and a CTA-count threshold. **That entry point is
currently identical to the prefill one**, because on SM120 the grouped
block-scaled path loads the scale-factor block by TMA at a fixed 128-wide layout
and a CTA N tile below 128 fails CUTLASS's static assertions; `DecodeCfg` is
pinned to the same 128×128×128 tile as `PrefillCfg`, with `static_assert`s
checking the two configs agree on every layout and stride type. **The header
comment in `include/nv_kernels.h` describing the decode entry as a skinny-N
`128x32x128` tile is stale**; the two wrapper entry points are kept so a future
CUTLASS with flexible grouped tiles can be dropped in without touching Rust.

**bf16 MoE** — two separate mechanisms, neither NVFP4. `Bf16GroupedExperts`
(`moe_bf16_grouped.rs`) stacks `[E, 2*inter, hidden]` and `[E, hidden, inter]`
and runs a host-side per-expert loop of candle matmuls; it is the Qwen3.5 MTP
head's expert bank. The indexed GEMV kernels are the fast path:
`nv_kernels_moe_gemv_swiglu_bf16_m1` reads the *selected* experts' stacked
weights directly by id — no gather, no permutation — computing
`silu(gate_e @ x) * (up_e @ x)` per slot and zeroing out-of-range ids, and
`nv_kernels_moe_gemv_down_tail_bf16_m1` does the down projection while folding
the weighted sum over slots, the shared-expert contribution and the residual add
into the same kernel. The `_mb` forms batch `b` decode slots over `gridDim.z`
with per-slot accumulation order unchanged, so they are bit-identical to `b`
sequential `_m1` calls. `nv_kernels_moe_unpermute_scatter_tail` is the same idea
for the NVFP4 path: scatter, weight, add shared expert, add residual, cast —
bit-exact with the unfused chain, one launch.

## Attention kernels and the KV cache

**Layout.** Every KV slab is `[slot][kv_head][head_dim]`, with scales (when
present) at `[slot][kv_head]`. `slot` is a linear position for a fixed cache and
`(start + i) % ring` for a ring. There is no head-major or paged-block-inner
variant except in the explicitly paged kernels, where a logical position `p` maps
to `block_table[p / block_size] * block_size + (p % block_size)` inside a shared
pool sized `[num_blocks * block_size, n_kv, head_dim]`. Ring caches exist because
sliding-window layers only ever attend a bounded suffix; `nv-models/src/gemma4.rs`
sizes them as `kv_fp8_ring_slots(sliding_window) = sliding_window +
VERIFY_PREFILL_CHUNK + VERIFY_RING_HEADROOM`, and only sliding layers get one.

**bf16 KV.** `cuda/kv_ring.cu` provides `kv_ring_append_bf16` (write slot from a
device int, so it captures) and `kv_shift_bf16` (in-place row-block move, with
the caller responsible for chunking so source and destination ranges are disjoint
within a launch). `nv_kernels_attention_bf16_decode_ring` reads a 2-element
device `ring_meta = [ring_start, stored]` and attends the last
`min(stored, window)` tokens in chronological order.

`cuda/flash_decode.cu` is the main decode attention: one block per query head,
the attended position range striped across all warps of the block, each warp
keeping its own online-softmax state (running max `m`, denominator `l`, a value
accumulator across its 32 lanes) and reducing each `q·k` with a single
warp-shuffle butterfly. There is no `__syncthreads` per position; the only
block-wide sync is the final flash merge across warps. It replaced a naive kernel
that did one block-wide reduction *per cached position*, which made decode
latency-bound on the sync chain rather than on memory. Identity with that naive
kernel is preserved and load-bearing: scores are `q·k` with no `1/sqrt(d)`
factor, GQA maps `kvh = h / (NH/NKV)`, the window start is computed the same way,
and an empty range yields zeros rather than NaN (`l <= 0 → inv_l = 0`). Variants:
`flash_decode_splitk_bf16kv` (two-stage, caller-owned f32 scratch sized by
`flash_splitk_scratch_elems`), `flash_decode_fused_bf16kv` (single launch, atomic
fan-in, the last split block per head merging in-launch; the `fan_in` counter
array is self-restoring so it need only be zeroed once), and `_mk` forms taking
`M` query rows for the speculative verify shapes
(`03-speculative-decoding.md`).

**fp8 KV.** `cuda/kv_fp8.cu` quantizes per `(token, kv_head)`:
`amax = max_d |x[t,h,d]|`, `scale = amax/448`, `x_fp8 = e4m3(x / scale)`. The
write slot is read from `start_dev`, a one-element device int, and the caller
passes the *full* slab base — the kernel computes the destination offset itself,
which is what allows one captured graph to write a different slot on each replay.
`dequantize_kv_fp8` is the inverse and is used at prefill to materialise a
contiguous bf16 window. `cuda/kv_fp8_paged.cu` adds the block-table indirection
plus `copy_kv_block_fp8`, duplicating one physical pool block for copy-on-write
of a tail block shared between sequences.

`cuda/attention_fp8_decode.cu` attends the fp8 slab without a dequantize step:
one block per query head, three passes — scores (`HEAD_DIM` threads each
contributing one `q[d]*k[d]` term, warp-shuffle then cross-warp reduce), softmax
over `n_total`, and `out[d] = Σ scores[i] * dequant(V[i,d])`. Scores live in
dynamic shared memory sized from `max_total`; `_gscores` is the identical kernel
with the scores array in caller-owned global memory, for contexts where the smem
allocation would exceed the portable 48 KiB limit. Both read `n_total` from a
device int. FP8 decode goes through `__nv_cvt_fp8_to_halfraw` plus a
`cvt.f32.f16` PTX, mirroring vLLM's conversion.

**What fp8 KV trades.** Storage is one byte per element instead of two, plus one
f32 per `(slot, kv_head)`. The quantization granularity is a whole `head_dim`
vector, so a head whose components span a wide dynamic range absorbs more error
than a finer block scheme would — the same amax-per-vector convention vLLM uses,
not a block-scaled format like NVFP4. Two consequences in the code: the prefill
path still dequantizes to bf16 rather than running eager attention on fp8, so the
slab is a *storage* format only the specialized decode and verify kernels consume
directly; and the fp8 verify kernels pin their contraction order explicitly
(`gqa512_verify_fp8` folds `v_scale` into the softmax weight with a fixed
`__fmaf_rn`/`__fmul_rn` order) so the result is reproducible rather than merely
close.

**Sparse prefill scoring (XAttention).** `gemma4.rs`'s `xattn_prefill_bias`
family implements the block-selection rule from **XAttention** (arXiv 2503.16428,
ICML'25, [mit-han-lab/x-attention](https://github.com/mit-han-lab/x-attention)),
env-gated via `NV_XATTN_PREFILL` (with a runtime override so one test process can
run dense and sparse prefills back to back) and applied only to full-attention
layers during prefill (`seq > 1`). For each `(Q-block, K-block)` pair the paper's
cheap proxy for attention mass is an antidiagonal-strided sum of the QK scores
rather than the full block — a Θ(1/stride) shortcut that still correlates with
the true softmax mass because attention scores are locally smooth along the
antidiagonal. Blocks are ranked by that proxy and kept until cumulative mass
clears a threshold, always keeping the sink block (`bk = 0`) and the diagonal
block; pruned blocks contribute an additive `-inf` bias.

This implementation is a **masked-dense reference, not the paper's fused
kernel**: the antidiagonal reduction is read off scores already computed densely,
so *which* blocks get selected is faithful to XAttention and is what the quality
gate and the achievable-sparsity measurement exercise, but the paper's actual
FLOP saving — reducing Q and K *before* the matmul — is not realized; that
reduction is the open kernel seam. A second deliberate simplification: block
importance is head-averaged into one shared `[1, seq, stored]` mask because that
is the shape candle's prefill additive mask accepts, where the paper scores per
head. `xattn_prefill_bias_mass` (the "v2.14b" scorer) computes the paper's real
signal — per-head softmax attention mass over candidate K-blocks, combined across
heads — processing one head at a time to keep peak host memory at one head's
`[seq, stored]`. Default off; `NV_XATTN_PREFILL=0` restores byte-identical dense
prefill.

**Verify-path attention.** `03-speculative-decoding.md` owns these; they are
*built* here. `tree_verify_attn_bf16` / `_fp8` let K query tokens attend the
committed prefix `[0, *n_committed)` plus the tree positions selected by a
`[K, K]` byte mask. `gqa512_verify_bf16` / `_fp8` is the head-group-batched chain
verify for `head_dim = 512` at group size 8: grid `(NKV, splits)`, each block
streaming its kv-head's range once from HBM through a `cp.async` shared-memory
tile shared by all eight q-heads of the group, one warp per q-head, all `M` chain
queries, with the queries staged in dynamic shared memory for `M >= 5` so the
register file holds only accumulators. `kv_append_*` / `kv_compact_*` append at a
device row offset and compact an accepted tree path `[base + path[i]] ->
[base + i]` through a scratch buffer; `verify_qkv_prep` is one kernel for per-head
RMSNorm, RoPE and the fp8 cache write; and `dflash_accept_f32` fuses the per-row
first-max argmax over the verify logits with the greedy accept chain against the
draft tokens, emitting `[accepted_count, tokens...]`.

## Why hand-written CUDA rather than library calls

Four distinct reasons. Where a library *is* the right answer it is used:
cuBLASLt for the general bf16 and fp8 GEMMs, CUTLASS/FlashInfer templates for the
FP4 block-scaled GEMMs, vLLM's Marlin for int4.

1. **Graph capturability.** candle's element-wise ops go through the legacy null
   stream, which cannot be captured, and `narrow().contiguous()` uploads layout
   metadata from a temporary host `Vec` per call — a captured graph would replay
   that memcpy from freed host memory. `residual_add_scale_bf16`,
   `scale_out_bf16`, `scale_inplace_bf16`, `copy_cols_bf16`,
   `gelu_tanh_mul_fused_bf16`, `tanh_softcap_bf16_to_f32` and
   `mul_sigmoid_rowgate_f32` each replace one such candle chain with a single
   stream-aware launch computing the same arithmetic; `gemv_e4m3_w8a8` is the same
   argument applied to a GEMM.
2. **Device-resident control values.** No library GEMM or attention entry point
   accepts "the sequence length is at this address." Everything in the decode loop
   that must read a per-step position — `incr_pos`, `write_kv_*`, `flash_decode_*`,
   `attention_fp8_decode`, `kv_append_*`, `qkv_prep` — takes an `int*` for that
   reason alone.
3. **Shape mismatch with library tiling.** A tiled GEMM charges for its tile. At
   M=1 decode the NVFP4 CUTLASS kernel pads to a 128-row tile and the cuBLASLt
   setup cost dominates a memory-bound bf16 projection, so `gemv_bf16`,
   `gemv_nvfp4`, `gemv_w4a16`, `gemv_i8_*`, `gemv_e4m3_*` and the MoE GEMVs do the
   M=1 (and small-M) case directly.
4. **Fusion across a boundary the library splits.**
   `rmsnorm_quantize_nvfp4_bf16` (norm + quantize), `qkv_prep` and
   `verify_qkv_prep` (norm + RoPE + cache write),
   `silu_mul_quantize_nvfp4_bf16_per_expert` (activation + requantize),
   `moe_unpermute_scatter_tail` and `moe_gemv_down_tail_*` (GEMV + weighted sum +
   shared expert + residual), `rmsnorm_add_scale_bf16` (residual + norm + the
   *next* norm's apply). Each carries a documented bit-exactness claim against the
   unfused chain, which is what makes them safe to swap in under a determinism
   gate.

## Determinism and algorithm pinning

`nv-quant/src/matmul.rs` carries the deterministic bf16 path. `NV_DETERMINISTIC=1`
(exactly `"1"`) selects it, and `splitk_timing_selection_enabled(splitk,
deterministic)` disables the *timed* split-K algorithm search when it is on — a
timed search is by construction non-reproducible. `NV_BF16_SPLITK=0` disables
split-K outright. A separate cuBLASLt handle (`DET_HANDLE_CACHE`) is kept for the
deterministic path and torn down alongside the normal one in
`release_stream_state`.

`nv-quant/src/algo_pin.rs` parses `MxNxK=IDX;...` specs into a
`(m, n, k) → algo index` map, malformed entries logged and skipped.
`NV_BF16_ALGO_PIN` and `NV_NVFP4_LT_ALGO_PIN` feed it, and
`NV_NVFP4_LT_ALGO_LOG` dumps the chosen algo's config
(id/tile/splitk/reduction/stages/cluster) so a pin can be derived from a run.
`fused_qkv_bitwise_safe(m, has_v)` encodes which fused-QKV shapes the GEMM path
may take without changing bits. The bit-exactness claims scattered through the
header (`_mb` equals `b` × `_m1`, fused norm+quantize equals the unfused pair,
the decode grouped GEMM equals the prefill one) are what the parity suites check;
see `08-build-and-testing.md`.

## Connections

- `03-speculative-decoding.md` — the tree-verify and chain-verify attention
  kernels, the fp8 verify KV, `dflash_accept_f32`, and the `_mk` split-K flash
  decode variants are built here and consumed there.
- `05-backends.md` — `nv_layers::backend::KernelId` enumerates this chapter's
  kernels and `kind_supports` declares which exist per backend; the wgpu backend
  reimplements most of them in WGSL, minus `MarlinGemmW4a16`.
- `08-build-and-testing.md` — `CUDA_ARCH_LIST`, `NVK_NVCC_WRAPPER`, `nvk.sh`, and
  the `parity_*` suites that hold the bit-exactness claims made above.
