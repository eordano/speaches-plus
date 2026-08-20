# hip/pending -- what is here and why it is not built

`build.rs::build_rocm` globs `hip/*.cpp` **shallowly**, so nothing in
this directory is compiled; the build emits a `cargo:warning` naming
every file found here. A file leaves this directory by being `git
mv`-ed one level up; no build change is required.

Port-wide context -- strategy, wavefront-64 audit, C ABI coverage,
NVFP4-vs-MXFP4 decision, per-arch matrix-core dispatch -- lives in
`docs/book/05.5-rocm-port-status.md` (below: "05.5"). This note keeps
only what is specific to porting group **G5**, its five sources:

| CUDA source | state | file here |
| --- | --- | --- |
| `cuda/gdn_recurrent.cu` | **ported, compiles, promotable** | `gdn_recurrent.hip.cpp` |
| `cuda/cutlass_probe.cu` | **ported, compiles, promotable** | `cutlass_probe.hip.cpp` |
| `cuda_sm120/moe_grouped_fp4_gemm.cu` | rewrite required; ABI stub + preserved layout kernel | `moe_grouped_fp4_gemm.hip.cpp` |
| `cuda_sm120/cutlass_fp4_gemm.cu` | rewrite required; ABI stub + NVFP4 decode prologue | `cutlass_fp4_gemm.hip.cpp` |
| `cuda_sm120/gemv_blockscaled_probe.cu` | **delete under `rocm`** -- see §4 | none |

Verification performed (ROCm 7.2.3 / LLVM 22.0.0 from the nix store,
no `/opt/rocm`, no ROCm-capable dGPU on this machine):

```
hipcc -c -O2 -std=c++17 -Wall --offload-arch=$ARCH \
  --rocm-device-lib-path=<rocm-device-libs>/amdgcn/bitcode -I include -I hip
```

for `$ARCH` in gfx90a, gfx942, gfx950, gfx1100, gfx1036 -- 4 files x 5
arches, 0 failures, 0 warnings. `nm -g --defined-only` over the
resulting archive lists exactly the six `nv_kernels_*` symbols
declared in `include/nv_kernels.h`. `--rocm-device-lib-path` is not
optional in this nixpkgs (`rocm-device-libs` is a separate store path
from `clr`); `cargo check -p nv-kernels --features rocm` fails with
*"cannot find ROCm device library"* unless `ROCM_DEVICE_LIB_PATH` is
exported alongside `ROCM_PATH`, and with both set it finishes clean
and prints the pending-file warning listing all fifteen entries here.

`gdn_recurrent.hip.cpp` was additionally **executed** on the machine's
gfx1036 (RDNA2 Raphael iGPU, 2 CUs, wave32) against a CPU reference --
§2. The other three files are stubs with nothing to execute. No CDNA
device exists here, so MFMA, wave64 and gfx950 LDS behaviour are
compiled only, never run.

## 1. hipify tool choice

`hipify-perl` emits invalid C++ for `cudaFuncSetAttribute` on a
templated `__global__` -- it splits on the comma inside the template
argument list, yielding `gdn_recurrent_kernel_f32<128), 128>`, and
reports exit 0 because perl does not parse C++; `hipify-clang` (given
`--cuda-path`, `--clang-resource-directory` and host libstdc++
includes) emits the correct form. Demonstration in 05.5 §2. Rule for
this repo: **any file that launches or references a templated
`__global__` must go through hipify-clang, or have its
`hipFuncSetAttribute` call hand-checked.** `gdn_recurrent.cu` is the
only such file in the corpus -- which is why the wire-phase perl run
over all 29 `cuda/*.cu` reported exit 0.

## 2. `gdn_recurrent` -- the LDS blocker, and the fix

The CUDA host code asks for

```
smem = (K * V + 2 * K) * sizeof(float) = (128*128 + 256) * 4 = 66_560 bytes
```

and relies on `cudaFuncAttributeMaxDynamicSharedMemorySize` to unlock
Hopper/Blackwell's ~227 KB opt-in dynamic shared memory. **There is no
AMD equivalent.** LDS is 64 KiB per workgroup on every shipping AMD
generation except gfx950 (160 KiB); `hipFuncSetAttribute` cannot raise
it past the hardware cap, so the launch fails outright -- including on
the gfx1036 in this box. 66 560 overshoots by exactly 1 024 bytes, so
the tempting micro-fix (move `k_buf`/`q_buf` out of LDS) lands at
exactly 65 536: zero headroom, one workgroup per CU.

The fix implemented here is **exact tiling over the V axis**. In the
inner loops, `state[kk * V + my_v]`, `kv_mem`, `delta` and `out_v` are
all indexed by `my_v` only, and `k_buf`/`q_buf` are broadcast; there is
*no* cross-channel communication, so splitting V across workgroups is
bit-exact, not an approximation. The kernel is templated on `V_TILE`,
owns `state[K_DIM][V_TILE]`, and the grid becomes `B * H * (V /
V_TILE)`:

| `V_TILE` | LDS bytes | workgroups/CU at 64 KiB | grid |
| --- | --- | --- | --- |
| 128 | 66 560 | 0 (gfx950 only) | `B*H` |
| 64 | 33 792 | 1 | `B*H*2` |
| 32 | 17 408 | 3 | `B*H*4` |

The host now **queries** `hipDeviceAttributeMaxSharedMemoryPerBlock`
and picks the largest tile that fits, falling back to the pre-existing
`-2` return (previously only for `K != 128 || V != 128`) if the query
fails or nothing fits -- a diagnosable code rather than an opaque
launch failure. A 64 KiB device selects `V_TILE = 64`; only a device
*reporting* more than 66 560 would take the untiled 128 path (gfx950
has 160 KiB of LDS, but whether HIP surfaces that through this
attribute is untested here).

### measured on the local gfx1036

```
lds=65536  smem128=66560  smem64=33792  smem32=17408
untiled V_TILE=128 launch            -> hipGetLastError()=1 (hipErrorInvalidValue)
V_TILE=64 vs V_TILE=32               -> 0/3840 outputs differ, bitwise identical
nv_kernels_gdn_recurrent_f32 (B=2,T=5,H=3,K=V=128), tile chosen by query = 64
  vs float64 CPU reference           -> max relative error 7.43e-4, rc=0
  vs float32 CPU reference           -> max relative error 1.23e-3
```

This pins down: (1) the blocker is real on hardware, not a spec-sheet
inference -- a 64 KiB-LDS device rejects the untiled launch; (2) the
tiling is exact -- `V_TILE = 64` and `32` agree bit-for-bit, as "no
cross-channel communication" predicts; (3) the residual CPU difference
is fp32 rounding in the T-recurrence and FMA contraction, not a logic
error -- the GPU sits *closer* to a float64 reference (7.4e-4) than
the float32 CPU reference does (1.2e-3). An earlier run with
`HSA_OVERRIDE_GFX_VERSION=""` set produced `hipErrorNoDevice`, and the
dispatcher returned `-2` instead of crashing or launching with a bogus
tile -- the intended behaviour of the added query guard.

Occupancy is honestly still poor. The recurrence is serial in T, so
the only parallelism is `B * H * tiles`; at `V_TILE = 64` the
workgroup is 64 threads -- one wave64 on CDNA, two wave32 waves on
RDNA -- and 33 792 bytes of LDS admits one workgroup per CU. If a real
gfx942 benchmark shows LDS-occupancy binding, the next move is
`V_TILE = 32` (3 workgroups/CU) *plus* pairing two lanes per output
channel with a two-step cross-lane reduction of `kv_mem`; that
reduction is wavefront-width sensitive and must be written against
`warpSize`, not 32. Deliberately not done here.

### wavefront-width audit for this kernel

`grep` for `warpSize`, `__shfl`, `__ballot`, `__activemask`, `% 32`,
`>> 5`, `& 31` in `cuda/gdn_recurrent.cu` returns **nothing**. The
kernel is pure `__syncthreads()` + LDS: no 32-lane assumption to
repair, correct on both wavefront widths as written. Two consequences:

- `__syncthreads()` is retained at every original site even though at
  `V_TILE = 64` on CDNA the whole workgroup is one wave and the
  barrier is a no-op. Removing it would be correct on wave64 and
  **wrong** on wave32 (gfx10xx/11xx/12xx run the same 64 threads as
  two waves). Do not "optimise" it away.
- `__launch_bounds__(V_TILE)` was added so the compiler sizes
  registers for the actual workgroup, now that `V_TILE` is smaller
  than the old `V`.

## 3. `cutlass_probe` -- trivially satisfiable

The CUDA version fails on ROCm only because of `#include
"cutlass/cutlass.h"` and `#include "flashinfer/math.cuh"`. Its body
computes two compile-time constants -- `cutlass::Status::kSuccess` is
enumerator `0`, and `(int)(6.0f * 32.0f)` is `192` -- so the HIP
version writes `0` and `192` from named constants with no third-party
headers. Behaviour is identical for the
`nv_kernels::cutlass_flashinfer_probe()` caller in `src/lib.rs`. If a
reviewer wants the probe to report something true about the *AMD*
build, the natural repurposing is hipblaslt's or Composable Kernel's
version -- but both are unrealised `.drv`s in this store (05.5 §6), so
that is a flake change, not a code change.

## 4. The three `cuda_sm120` files are rewrites, not ports

Running `hipify-perl` on them is worse than useless, and the output is
deliberately **not** checked in. Measured on this tree:

- `moe_grouped_fp4_gemm.cu` -- 22 diff lines, all `cudaStream_t` ->
  `hipStream_t` and `cudaGetLastError` -> `hipGetLastError`; every
  `cutlass/`, `cute/` and `CollectiveBuilder` line untouched, so the
  file still cannot compile. The diff creates the *impression* of a
  port.
- `cutlass_fp4_gemm.cu` -- 62 diff lines; the damaging ones rename
  `__nv_bfloat16` -> `__hip_bfloat16` **inside**
  `INSTANTIATE_FP4_GEMM_KERNEL_LAUNCHER(...)`, a FlashInfer macro that
  does not exist on AMD, while the macro, the
  `flashinfer/gemm/fp4_gemm_template_sm120.h` include and every
  `cute::Int<>` tile parameter stay NVIDIA-only.
- `gemv_blockscaled_probe.cu` -- **zero-line diff**; its entire
  content is CUTLASS type aliases.

CUTLASS has no AMD backend. The analogues are Composable Kernel
(tile-level, closest structural match to CUTLASS collectives) and
hipBLASLt (library-level, closest match to the *shape* of these entry
points, which already take `workspace` / `workspace_bytes` /
`required_workspace`).

### what the stubs preserve

`moe_grouped_fp4_gemm.hip.cpp` keeps `get_group_gemm_starts` nearly
verbatim, because that kernel is the written form of the MoE
memory-layout contract with the Rust caller and it survives the
backend change intact:

- `a_ptrs[e] = a_base + expert_offset * (k/2)` -- A byte-addressed, 2
  e2m1 elements per byte, rows of `k/2` bytes.
- `b_ptrs[e] = b_base + e_global * n * (k/2)` -- B indexed by *global*
  expert id, A by the compacted `expert_offsets` prefix sum.
- `a_scales_ptrs[e] = a_scales_base + sf_offset * (k/16)` -- one ue4m3
  scale per 16-element NVFP4 block, hence `group_size = 16`.
- `alpha_ptrs[e] = alphas_base + e_global` -- per-tensor FP32 global
  scale, also global-indexed.

Two changes were required:

1. **Bug fix, carried over from CUDA.** The original guard at L112-113
   was `int e = threadIdx.x; if (e >= gridDim.x * blockDim.x)
   return;` -- vacuous, since `threadIdx.x` is always below
   `blockDim.x`. It is now `e = blockIdx.x * blockDim.x + threadIdx.x;
   if (e >= num_experts) return;` with `num_experts` added as a kernel
   parameter. The launch is `<<<1, num_experts>>>`, so the original
   was safe only because the grid was one block; it would read out of
   bounds the moment the launch went multi-block. **The same fix is
   still needed in the CUDA file** (out of this group's scope; also
   flagged in 05.5 §6).
2. `layout_sfa` / `layout_sfb` are **dropped**. They were
   `ScaleConfig::tile_atom_to_shape_SFA/SFB`, CUTLASS's `Sm1xx`
   block-scaled scale-factor swizzle; there is no AMD counterpart and
   no way to fake one -- whoever writes the CK or hipBLASLt path must
   derive the equivalent SF layout for the AMD block-scaled MFMA and
   re-emit it here. `StrideA/B/C` became a plain `GroupStride{major,
   minor, batch}` POD carrying the same three values
   `cute::make_stride(v, _1{}, _0{})` encoded.

`cutlass_fp4_gemm.hip.cpp` keeps all three entry points (`_bf16`,
`_bf16_streamk`, `_bf16_tiled`) with byte-identical signatures, sets
`*required_workspace = 0` and returns `-1001`
(`kRocmNotImplemented`). The `-3` return for an out-of-range
`tile*2 + stream_k` variant is preserved so the argument-validation
contract still holds. It also carries a working
`nvfp4_decode_to_bf16_prologue` kernel (§5), the piece the real
implementation will reuse.

### `gemv_blockscaled_probe.cu` should be deleted under `rocm`

Its only symbol appears exactly once in the tree, at its own
definition; it is absent from `include/nv_kernels.h`, so no Rust
caller can exist (05.5 §2.1). Its body is a link-time canary for
`cutlass::gemm::device::GemvBlockScaled`, which has no AMD analogue.
No file is provided; the `rocm` build should simply lack the symbol.
If a build-time canary for AMD matrix cores is wanted instead, the
per-arch MFMA/WMMA builtin dispatch verified to compile with this
toolchain -- including the two traps (the RDNA WMMA builtin bakes the
wavefront width into its name, and the accumulator fragment is `f32x8`
on wave32 vs `f32x4` on wave64) and the rocWMMA recommendation -- is
recorded in 05.5 §8.2. gfx1036 (RDNA2) has *neither* MFMA nor WMMA
and needs a scalar `#else` fallback, not an `#error`.

## 5. NVFP4 on AMD

Decision, rationale and format comparison: 05.5 §8 (decode NVFP4 to
bf16 in a mainloop prologue -- lossless, works on every target;
offline MXFP4 transcode recorded as a lossy, gfx950-only,
measure-before-claiming option). The prologue lives in
`cutlass_fp4_gemm.hip.cpp`, with E2M1/UE4M3 decode tables lifted
verbatim from the proven `cuda/gemv_nvfp4.cu` (`decode_e2m1_dev`,
`decode_ue4m3_dev`) so the two paths cannot drift. Caveat, so nobody
mistakes it for finished: the prologue indexes the scale tensor
**linearly** (`scales[i / 16]`), but real NVFP4 checkpoints store
scales in the 128x4 swizzle of `swizzled_scale_dst()` in
`cuda/gemv_nvfp4.cu` -- whoever wires it into a CK or hipBLASLt
mainloop must apply the swizzle on the read side or de-swizzle at load
time. The decode arithmetic is correct; the addressing is a
placeholder.

## 6. Tile shapes and `stream_k` do not carry over

The six-configuration CUTLASS SM120 variant matrix at
`cutlass_fp4_gemm.cu` L127-172 means nothing on AMD: MFMA wants
different K multiples, LDS schedules and swizzles (re-sweep on real
gfx942/gfx950 hardware, do not translate), and stream-K has no direct
AMD analogue -- collapse the `_streamk` entry into the base one on
ROCm, keeping the symbol only for C ABI stability. Details: 05.5 §8.2.

## 7. Out of scope, restated

`cuda/marlin/` is vendored upstream Marlin with no AMD equivalent of
any kind; its five `nv_kernels_marlin_*` symbols will never resolve
under the `rocm` feature (05.5 §4.3).
