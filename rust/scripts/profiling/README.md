# CUPTI kernel tracer

`ncu` and `nsys` are not in the devshell, and lanes have inferred
kernel behaviour instead of measuring because of that. **CUPTI is
present** in the devshell's merged CUDA, which is enough.

`nvtrace.cpp` is a `CUDA_INJECTION64_PATH` shim on the CUPTI Activity
API: every kernel and memcpy with GPU start/end, **grid dimensions**,
bytes moved and `graphId` -- so it sees *inside* a captured CUDA
graph, where this repo's decode steps live.

```sh
sh build-nvtrace.sh                                   # -> libnvtrace.so
CUDA_INJECTION64_PATH=$PWD/libnvtrace.so <any cuda binary>
python3 analyze.py <trace> <batch-sizes> [vocab]      # per-kernel table
sh run-trace.sh                                       # trace + analyze
```

`analyze.py` segments steps on the logits DtoH memcpy of
`b * vocab * 4` bytes. Vocab defaults to 129280 (DSOCR); a Gemma4
verify trace needs 262144 (third arg or `NVTRACE_VOCAB`) with `b` =
verify width (5 for k=4), or it finds zero marks and reports nothing.

Perturbation measured on a DSOCR b=8 decode step: wall +7%, GPU-side
+1%.

Grid dimensions are the reason this exists: #111 spent a campaign
believing DSOCR decode was bandwidth-bound at 30% of roofline; the
trace showed `flash_fwd_kernel` on a **1x1x10 grid** -- 10 blocks on
188 SMs, 5.3% residency, grid-starved, which no bandwidth number could
show. Batching the launch took b=8 from 7.537 to 4.546 ms/step.
Corollary: step-level "achieved GB/s" averages kernels with wildly
different attainable ceilings and is not a valid target -- judge each
kernel against what its own grid can sustain.

## Achieved-bandwidth lane (#42)

Two halves, per-kernel only -- never a step average (see above):

1. **Out-of-graph:** `nv-specdecode/tests/sol_roofline.rs` phase B
   (`NV_SOL_TEST=1 NV_SOL_PHASES=ab`, `#[ignore]`, cuda) emits per-GEMM
   `SOL,gemm_m5_achieved_GBs` and `SOL,gemm_m5_attainable_frac` lines --
   bytes computed from each Linear's *actual stored* buffers (nvfp4
   packed + scales, or bf16), divided by the cold-loop ms, against a
   same-run `read_contig` ceiling. Cold is the honest number: most
   per-layer weights fit L2 warm. Also emitted: an explicit `lm_head`
   case with its projected `gemm_m5_lowbit_floor_ms`/`delta_ms`
   (slice 1 ceiling), and a `fusion_ms` separate-vs-fused
   rmsnorm->quantize pair on `gate_up` via `forward_prenorm_nvfp4`
   (slice 2 ceiling, measured not modeled).
2. **In-graph:** nvtrace against a serving session, segmented with
   `analyze.py <trace> graphs` -- NOT the DtoH mode: the serving verify
   path does device-side accept and never copies `b*vocab` logits to
   host, so logits-DtoH segmentation finds zero marks there. Every
   in-graph kernel carries its `graphId`; graph mode clusters replays
   on >200us gaps. Driver: `rust/scripts/verify-trace.sh` (boots the
   server under the shim, warm + long greedy request, TERM -- the shim
   flushes on SIGTERM).

**Measured 2026-08-09 (94 verify-graph replays, Gemma-4-31B spec
serving, short ctx):** span med 27.87 ms, busy 26.90, **GPU idle
inside the graph 0.97 ms (3.5%)** -- the Lt-stall slice's <=1.0 ms
bound confirmed in-graph; perfect stall removal buys under 1 ms.
lm_head is the 1.73 ms `256x64` bf16 GEMM (matches the out-of-graph
1.728 exactly). The round is dominated by the two bf16 wmma GEMM
families at 5.91+5.60 = **11.5 ms/replay** -- the attention
projections summed over layers, kept bf16 because the #57 eval refused
attn-proj quant. Together with fp4 at 5.45+3.56, the 18.4 ms verify
target is arithmetically unreachable without either attn-proj quant
(quality-refused) or a faster bf16 GEMM for those shapes. Drafter
chain graph: 2.20 ms/replay, 2.04 of it the gemv_bf16.

A `SOL,` line that never appears means the env gate or model dir
failed; `1 passed` in 0.00s is a skip, not a pass.
