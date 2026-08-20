# Marlin w4a16 -- Rust integration contract

Raw-pointer C entry points wrapping vLLM's Marlin int4 GEMM with **no
libtorch**. Vendored in `entry.cu`; declared in `include/nv_kernels.h`.
Target: sm_89 + sm_120 (Blackwell), CUDA 12.9.

Quant format: compressed-tensors **w4a16**, 4-bit *symmetric* int (vLLM
scalar type `kU4B8`), `group_size = 32`, bf16 or fp16 activations. No
zero-points, no act-order / g_idx / desc_act.

> u4b8 / offset-8: each 4-bit weight is stored as an **unsigned**
> nibble in `0..15` equal to `signed_value + 8`. The GEMM subtracts 8
> during dequant. Neither the repack nor the GEMM rescales the numeric
> value -- the bits handed to repack must already be offset-binary
> nibbles, and offset-8 is preserved end-to-end.

## C ABI

```c
int nv_kernels_marlin_workspace_elems(int* out_elems);

int nv_kernels_marlin_repack_w4a16(
    void* stream, const void* b_q_packed, void* b_q_marlin_out,
    int k, int n, int num_bits /* = 4 */);

int nv_kernels_marlin_gemm_w4a16(
    void* stream, const void* a_bf16, const void* b_q_marlin,
    const void* b_scales, void* c_out, void* workspace,
    int m, int n, int k, int group_size, int a_is_bf16 /* 1=bf16,0=fp16 */);
```

All return `0` on success, negative on bad-args / unsupported-shape /
cuda error. `stream` is a `cudaStream_t` (null = default stream).

Dimension convention (Marlin / vLLM): **K = in_features,
N = out_features, M = batch*seq (activation rows)**.
`C[m,n] = A[m,k] @ W^T` with the logical weight `W[n, k]` (out x in) --
same as a torch `nn.Linear`.

## Weight layout: what you have vs. what repack wants

Compressed-tensors gives you:

- `weight_packed`: `[out_features, in_features/8]` `int32`, packed
  along the **in** (K) axis, 8 int4 per int32, **low-nibble first**
  (nibble `j`, bits `[4j, 4j+4)`, is the weight at `in = 8*col + j`),
  offset-8 values.
- `weight_scale`: `[out_features, in_features/group_size]` bf16.
- `group_size = 32`.

`nv_kernels_marlin_repack_w4a16` wants `b_q_packed` as
`[k/8, n] = [in_features/8, out_features]` `int32`, K-major, with the
**same** intra-int32 packing. The repack kernel only rearranges which
int32 / bit position a nibble lands in for the tensor-core tile; it
never changes nibble values.

Since both layouts pack identically along K, the only transform is a
plain 2-D transpose -- **no re-packing inside the int32 word**:

```
b_q_packed = weight_packed.T    # [in/8, out] int32, contiguous
```

Done once at load time (host or transpose kernel), then uploaded. If
an exporter already stores `[in/8, out]` K-major, skip the transpose.

Repack output `b_q_marlin`: `[k/16, n*16/8] = [in/16, out*2]` `int32`,
contiguous -- same total element count (`k*n/8` int32s, allocate
`k*n/8 * 4` bytes on device), just retiled. Produced **once per weight
at load time** and cached; the forward pass only calls the GEMM.

## Scales layout

Marlin wants scales as `[num_groups, n] = [k/group_size,
out_features]`, contiguous, in the **same float dtype as the
activations** (`num_groups = 1` when `group_size == -1`, per-channel).
Your `weight_scale` is `[n, k/gs]`, so: transpose, **then a 64-wide
interleave permutation**:

```
s        = weight_scale.T                       # [k/gs, out]  bf16
b_scales = s.reshape(-1, 64)[:, scale_perm].reshape(k/gs, n).contiguous()

scale_perm        = [i + 8*j for i in range(8) for j in range(8)]   # grouped
scale_perm_single = [2*i + j for i in range(4)
                             for j in (0,1,8,9,16,17,24,25)]        # per-channel
```

> **CORRECTED 2026-08-09.** This section previously said "no
> marlin-internal permutation of the scale array". **That was wrong**:
> Marlin consumes scale fragments in tensor-core lane order, not plain
> `[group, n]` order, and an integrator following the old text would
> have gotten silently wrong scales -- all values present and finite,
> just attached to the wrong output columns, degrading quality with no
> error. vLLM applies exactly this permutation in
> `marlin_permute_scales`
> (`vllm/model_executor/layers/quantization/utils/marlin_utils.py`).
> The shipping caller was always correct --
> `nv-models/src/gemma4_e4b.rs::MarlinLinear::from_raw` builds
> `perm = [i + 8*j]` and `index_select`s the transposed scales through
> it -- so serving was never affected; only this document was.

Pick the permutation by grain, the way vLLM does:

| condition | permutation |
|---|---|
| `group_size != -1` **and** `group_size < k` | `scale_perm` (64-wide) |
| `group_size == -1` **or** `group_size == k` | `scale_perm_single` |

`MarlinLinear::from_raw` applies the **grouped** permutation
unconditionally -- correct for every shipping checkpoint
(`gemma-4-E4B-it-qat-w4a16-ct` is `group_size: 32` with `k >= 2560`),
but a caller setting `group_size == k` (legal, the GEMM accepts it)
would silently take the wrong branch. Do not add a
per-channel/`group_size == k` Marlin path without also selecting
`scale_perm_single`.

No numeric change to scale values; the reshape/permute is done once at
load time. dtype note: fp32 scales on disk must be cast to bf16/fp16
to match the activation dtype before upload.

## Workspace

- `nv_kernels_marlin_workspace_elems(&elems)` -> `elems == SM count`
  of the current device (e.g. 170 on a Blackwell part). Allocate
  `elems` **int32** on device.
- **`cudaMemset(ws, 0, elems*4)` exactly once** before the first GEMM.
  The kernel uses it as inter-block locks and returns all entries to
  `0` on normal completion, so one buffer per device (or per stream)
  is reused across forwards **without re-zeroing** for the model's
  lifetime. Re-zero only if a launch failed mid-flight.
- Not sized by N -- purely the SM lock array. (The fp32 reduce scratch
  `c_tmp` is allocated/freed internally by the GEMM each call.)

## Forward-pass sequence (per `W4a16Linear`)

Load time (once): transpose `weight_packed` -> upload -> allocate
`b_q_marlin` (`k*n/8` int32) -> `nv_kernels_marlin_repack_w4a16(stream,
b_q_packed, b_q_marlin, k=in, n=out, 4)` -> free `b_q_packed` ->
build + upload `b_scales` (transpose + `scale_perm`, REQUIRED) ->
query/allocate/zero the workspace.

Per forward (`x: [m, in]` bf16): ensure `x` row-major, 16-byte
aligned, `stride(0)=k`, `k % 8 == 0`; allocate `c_out = [m, out]`
(activation dtype, 16-byte aligned);
`nv_kernels_marlin_gemm_w4a16(stream, x, b_q_marlin, b_scales, c_out,
workspace, m, n=out, k=in, group_size=32, a_is_bf16=1)`. `c_out` holds
`x @ W^T`; add bias separately (the wrapper passes none).

## Shape / alignment constraints (enforced; violations return nonzero)

| Quantity | Constraint |
|----------|------------|
| `k` (in_features)  | multiple of `tile_size = 16` |
| `n` (out_features) | multiple of `min_thread_n = 64` |
| `group_size`       | one of **32, 64, 128** or `-1`; must divide `k` |
| `a` activations    | row-major, 16-byte aligned, `stride0 = k`, `k%8==0` |
| `c_out`            | row-major `[m,n]`, 16-byte aligned |
| `b_scales` rows    | `k / group_size` |
| repack `k`         | multiple of `tile_k_size = 16` |
| repack `n`         | multiple of `tile_n_size = 64` |
| workspace          | `>= SM count` int32, zeroed once |

`m == 0` is a no-op (returns 0); `m` has no alignment requirement --
Marlin splits it internally.

### Which group sizes actually have a kernel

`generated/generate_kernels.py` instantiates `group_blocks in
[-1, 0, 2, 4, 8]` with `group_blocks == group_size / 16`, so the
compiled set is `-1`, 32, 64, 128 -- **`group_size = 16` has no Marlin
kernel.** For gs=16, `get_marlin_kernel` falls through every branch of
`generated/kernel_selector.h`, returns `MarlinDefault`, and
`marlin_mm_raw` returns `-1` -- a clean rejection, not a wrong answer.
This matters because the Metal/wgpu lanes standardise on gs=16 and
gs=32: **gs=32 is the only group size both backends can serve today.**
Adding gs=16 means instantiating `group_blocks = 1`, which multiplies
the generated kernel count and drives `marlin_template.h`'s staging
arithmetic (`thread_k_blocks / group_blocks`, the
`group_blocks < thread_k_blocks` branches at lines 565, 994, 1011)
into a regime upstream vLLM does not ship. Do not do it blind -- it
needs a numerical gate on real hardware. `gemv_w4a16.cu`'s block/row
kernel already handles gs=16 correctly and is the supported CUDA gs=16
path.

## Scalar-type selection (done inside `entry.cu`, for reference)

| role | value |
|------|-------|
| `a_type` (activation) | `vllm::kBFloat16` or `vllm::kFloat16` |
| `b_type` (weight)     | `vllm::kU4B8` (4-bit sym, offset-8) |
| `c_type` / `s_type`   | same as `a_type` |
| `has_zp`              | `false` |
| `has_act_order`       | `false` (g_idx / perm = null) |
| `is_k_full`           | `true` (no-op when act_order off) |
| `use_fp32_reduce`     | `true` |
| `use_atomic_add`      | `false` |

## Build wiring

`build.rs` recursively collects `cuda/**/*.cu`, so
`cuda/marlin/entry.cu` and the generated `cuda/marlin/generated/*.cu`
are picked up automatically. nvcc adds each `.cu`'s own directory to
the include path, so `entry.cu`'s `#include "marlin.cuh"` /
`"kernel.h"` resolve against the sibling-vendored headers; if those
headers move to a shared `include/`, add that dir to
`collect_includes`.

## Assumed sibling symbols (must exist when entry.cu compiles)

From `marlin.cuh`: namespace `marlin`; constants `tile_size`,
`tile_k_size`, `tile_n_size`, `min_thread_n`, `max_thread_n`,
`default_threads`, `repack_threads`, `repack_stages`; device helpers
`div_ceil`, `cp_async4`, `cp_async_fence`, `cp_async_wait<>`.
From `scalar_type.h`: `vllm::ScalarType`, `vllm::kBFloat16`,
`vllm::kFloat16`, `vllm::kU4B8`.
From `kernel.h`: the `MarlinDefault` sentinel and
`get_marlin_kernel(...)`.
From `kernel_selector.h`: `thread_config_t`, `exec_config_t`,
`is_valid_config(...)`, `determine_exec_config(...)`.

If a sibling already vendors a non-torch `marlin::marlin_mm`
translation unit, drop the `marlin_mm_raw` body in `entry.cu` and call
that instead -- the extern "C" wrappers are the load-bearing part.

---

## Accuracy contract

Marlin dequantizes and applies the group scale **in bf16** -- `__hmul2(frag_b, s)`
in `marlin_template.h` -- and only then feeds an MMA that accumulates in fp32
(`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`). Error is therefore
dominated by **operand rounding, not accumulation order**, and a half-ulp bf16
rounding of the scaled weight is its honest floor. fp32 accumulation over 1024
terms cannot produce the ~1e-2 absolute error observed; reduced-precision
intermediates can, and do.

`marlin_w4a16_matches_cpu_reference` therefore bounds the error in **bf16
operand-rounding units**:

    |got - want| / (u * sum_j |a_j * w_j|),  u = 2^-8

with the denominator from `Weight::abs_dot`. The bound is **1.0 unit**: above one
full bf16 ulp the error is larger than bf16 operands can explain, so the kernel
really is wrong.

### Why not a relative bound

The gate used to be `|got - want| / max(|want|, 0.5)`. The signed result grows
like `sqrt(k)` while the operand-rounding bound grows like `k`, so a fixed
relative threshold silently tightens as k grows with the kernel unchanged. Same
kernel, same seeds, one box:

| k    | old max rel err | rounding units |
|------|-----------------|----------------|
| 256  | 8.2e-3 .. 1.0e-2 | 0.232 .. 0.466 |
| 512  | 1.2e-2 .. 1.6e-2 | 0.235 .. 0.338 |
| 1024 | 1.3e-2 .. 2.5e-2 | 0.140 .. 0.320 |
| 2048 | 2.7e-2 .. 4.8e-2 | 0.148 .. 0.262 |

The old column climbs 6x across the sweep and blew its 2e-2 bound from k=1024 up.
The new one is flat in k and peaks at 0.466 -- essentially the 0.5 ceiling a
single half-ulp rounding allows. Marlin is tight against theory.

RMS normalization does **not** fix this: RMS grows like `sqrt(k)` too, so it
leaves a `sqrt(k)` trend behind. `|A|.|W|` is the textbook bound and the only
form that comes out k-independent.

### Two things not to do

- **Do not reinstate `max_rel` as a gate.** It is still printed, as a diagnostic
  only, so numbers stay comparable with pre-2026-08-10 logs.
- **Do not raise the 1.0 bound to get green.** Above one ulp the error exceeds
  what bf16 operands can explain and the kernel is wrong. That is the whole point
  of expressing the bound in ulps rather than fitting a constant.

`the_rounding_unit_metric_is_calibrated_and_can_fail` pins the metric on the CPU
with no GPU: 0.5/1/2/4 ulps of error read as 0.5/1/2/4 units, the bound trips
exactly when the error exceeds one ulp, and an exact match reads 0.0.
