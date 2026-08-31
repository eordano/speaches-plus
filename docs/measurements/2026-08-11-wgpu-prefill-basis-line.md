# wgpu prefill: the basis line a chunked-prefill port is judged against

Round of 2026-08-11, task #28. **No decoder was changed this round.** Nothing
here is a port; this page exists so that a later port can be proven or refuted
against numbers that already have their basis attached.

`../book/08.4-PERFORMANCE.md` governs every number below: checkpoint, backend, machine,
prompt length, decode-vs-end-to-end, and load are stated for each row, and the
archive (this file) is where measurement rounds live.

## 0. Why this page exists

Task #28 quotes one sizing number for the port and sources it to
`docs/measurements/*exec-shape*`. **That file does not exist** and never did:
`docs/measurements/` did not exist in the tree before this page created it
(`docs/` held `book/`, `diffusion-gemma.md`, `rocq-proof-notes.md`). The task's
sizing number is therefore unsourced and is not repeated here.

The task's decoder list is also wrong in three places; §2 gives the
code-verified list.

## 1. Basis tuple (applies to every number on this page)

| item | value |
|---|---|
| box | blackwell; NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition, 97,887 MiB, driver 590.48.01 |
| backend | wgpu on Vulkan (`/run/opengl-driver/.../nvidia_icd.json`), single adapter |
| binary | `$HOME/.cache/cargo-tmp/tgt-round14/release/speaches-plus`, cargo **release**, `--features wgpu` (no cuda), built 2026-08-10 23:48 |
| source | this repository at measurement time (2026-08-10); `git status --short` showed only test files dirty, no `src/` |
| co-tenant | 13,208 MiB resident on the GPU before **every** boot below, GPU util 0% at each idle check: production `speaches-plus` 8,486 MiB + a debug `speaches-plus` 4,638 MiB + `kwin_wayland` 28 MiB. This is not the ~56 GB the round brief assumed -- it was 13.2 GB all night. |
| load average (1 min) | recorded per request; range over the round 2.99 - 14.96. See §6 for what load does to these numbers (≤3.5%). |
| KV window | `NV_WGPU_CHAT_MAX_SEQ=10240` on every boot, so rows are comparable; `DEFAULT_MAX_SEQ` is 2048 and would have refused the 8k prompts |
| harness | HTTP `POST /v1/chat/completions`, `temperature=0`, `stream=false`, `max_tokens=8` (1 for the Qwen3.5 rows) |
| prompt | official chat template, rendered **server-side** by the engine -- `WgpuChatEngine::load_with_lora` refuses to serve a hand-rolled or ChatML-fallback prompt (`chat_engine_wgpu.rs:701-706`). The user turn is a fixed instruction plus the first N chars of `docs/book/*.md` concatenated in name order at this sha. |
| quantity | `prefill_ms` as the server itself measures it (timer opened at `chat_engine_wgpu.rs:2267`, closed at `:2287`, logged at `:2569`). It excludes HTTP, tokenisation and decode; client wall exceeded it by 37-461 ms. |
| repeats | 2 per point; the table reports the mean and the observed spread |
| raw logs | internal measurement logs (round14, 2026-08-10); not preserved in this repository |

## 2. Which decoders have chunked prefill (code-verified, HEAD 2026-08-10)

`Decoder::prefill_chunk_len` (`rust/src/oapi/chat_engine_wgpu.rs:364`) is the
serving seam: a non-zero answer means the engine calls `prefill_tokens` for the
whole prompt minus its last token (`:2271`), and zero means it steps the prompt
one token per command-buffer replay (`:2284`).

| wgpu kind | chunked prefill in code | chunk `m` (default) | knob | engages on the checkpoint this box has |
|---|---|---|---|---|
| Gemma4Dense (`gemma4_wgpu.rs`) | **yes** | 16 (`MK_MAX`, `:2837`) | `NV_G4_WGPU_PREFILL_M` / `NV_WGPU_PREFILL_M` | **yes** -- boot line `chunked prefill: m=16 rows, 2942 passes/chunk` |
| Gemma4E4b (`gemma4_e4b_wgpu.rs`) | **yes** | 10 (`prefill_m_from_env`, `:2601`), capped at 16 (`PREFILL_SLAB = SG_MK_MAX`) | `NV_E4B_WGPU_PREFILL=0`, `NV_E4B_WGPU_PREFILL_M` | not booted this round |
| Gemma4Moe (`gemma4_moe_wgpu.rs`) | **yes** (landed 2026-08-09) | 8, max 64 (`:3113`) | `NV_G4MOE_WGPU_PREFILL_M` | not booted this round |
| Qwen3_5Dense (`qwen3_5_dense_wgpu.rs`) | **yes, but dead here** | 16 (`:1816`) | `NV_WGPU_PREFILL_M` | **no** -- the graph drops the prefill pass list unless *all* attention and MLP projections are bf16 (`:3873`, `:4121`). The only Qwen3.5 dense checkpoint on this box is NVFP4, so boot prints `[q3d-wgpu] chunked prefill disabled: nvfp4 mlp projections` and `prefill_chunk_len()` returns 0. |
| Qwen3_5Moe (`qwen3_5_moe_wgpu.rs`) | **no** | -- | -- | per-token (`:4852` -> `step_inner`, one submit per token) |
| GptOss (`gpt_oss_wgpu.rs`) | **no** | -- | -- | per-token (`:1763`) |
| Laguna (`laguna_wgpu/mod.rs`) | **no** | -- | -- | per-token (`:443`) |

Corrections this forces:

- **The task brief is wrong three ways.** It says Gemma4Dense and Gemma4Moe
  "landed it" (true), that Qwen3_5Dense has not (false -- the code is there, it
  is the *checkpoint dtype* that disables it), and it omits both Gemma4E4b
  (which has had it longest) and Laguna (which has not).
- `docs/book/05-backends.md` and `docs/book/05.1-wgpu-status.md` §5 now carry
  four-of-seven and the NVFP4 caveat; they previously read two kinds and one
  kind respectively.
- `docs/book/01.4-STATUS.md`'s wgpu table is the closest to correct (it already
  files "wgpu chunked prefill for qwen3_5_moe / gpt_oss / laguna" as open) but
  its Qwen35Dense row reads "yes" without the NVFP4 caveat.

Two shape facts a port needs:

- The per-token fallback skips the lm-head: `prefill_step` replays
  `passes[..head_start]`, one command buffer per token, with no host readback.
  Only the prompt's last token runs the full graph.
- **Except on E4B**, where the serving seam routes `prefill_step` to
  `decode_step` (`chat_engine_wgpu.rs:399`), so an E4B prefill token that misses
  the chunked path pays the lm-head *and* an argmax readback. E4B's non-chunked
  cost is therefore not comparable to the other kinds'.

## 3. The A/B: same model, same box, chunked vs per-token

`nvidia/Gemma-4-31B-IT-NVFP4` (the flake-pinned chat checkpoint), 60 layers,
hidden 5376, vocab 262,144, modelopt NVFP4 -> `WgpuModelKind::Gemma4Dense`.
Boot line: `passes_per_token=548`, `chunked prefill: m=16 rows, 2942
passes/chunk`. The only difference between the two columns is
`NV_WGPU_CHAT_CHUNKED_PREFILL=0`, verified present in `/proc/<pid>/environ` of
the serving process before the per-token column was taken.

| prompt tokens | per-token replay (ms) | chunked m=16 (ms) | speed-up | per-token ms/tok | chunked ms/tok |
|---:|---:|---:|---:|---:|---:|
| 480 | 13,302.8 (±2.5%) | 4,886.2 (±4.4%) | **2.72x** | 27.71 | 10.18 |
| 1,065 | 30,790.2 (±1.9%) | 11,616.5 (±3.8%) | **2.65x** | 28.91 | 10.91 |
| 2,213 | 66,999.1 (±1.2%) | 25,821.2 (±2.8%) | **2.59x** | 30.28 | 11.67 |
| 4,258 | 134,127.2 (±0.03%) | 52,952.4 (±1.6%) | **2.53x** | 31.50 | 12.44 |
| 8,824 | 294,820.5 (±0.04%) | 122,816.3 (±0.3%) | **2.40x** | 33.41 | 13.92 |

Both columns produced **identical greedy continuations** -- all 8 sampled
tokens, at all five lengths (480 -> "This document serves as a guide to the";
2,213 -> "This document serves as a comprehensive guide to"; 8,824 ->
"`speaches-plus` is a"), which is the cheapest available check that the chunked
path is not silently wrong.

**The measured win of the shipped chunked prefill is 2.4-2.7x, not the order of
magnitude "one command buffer per 16 tokens" suggests.** §5 says why, and that
ceiling is the single most useful number on this page for sizing the port.

## 4. Decoders with no chunked path, measured

Same box, same harness, same KV window.

`ig1/Qwen3.5-9B-NVFP4` -> `Qwen3_5Dense`, 32 layers, `passes_per_token=568`,
chunked prefill compiled out by the NVFP4 projections (§2).

| prompt tokens | prefill (ms) | spread | ms/tok |
|---:|---:|---:|---:|
| 476 | 5,663.2 | 0.10% | 11.90 |
| 1,028 | 13,434.3 | 0.04% | 13.07 |
| 2,107 | 32,417.5 | 0.08% | 15.39 |
| 4,098 | 80,500.2 | 0.02% | 19.64 |
| 8,459 | 244,825.2 | 0.01% | 28.94 |

`poolside/Laguna-XS-2.1-NVFP4` -> `Laguna`, 40 layers, `passes_per_token=1154`,
no `prefill_tokens` at the model layer at all.

| prompt tokens | prefill (ms) | spread | ms/tok |
|---:|---:|---:|---:|
| 491 | 6,942.3 | 0.15% | 14.14 |
| 2,087 | 47,107.5 | 0.09% | 22.57 |
| 8,213 | 324,364.5 | 0.01% | 39.49 |

## 5. Where the time goes, and why chunking only buys 2.4-2.7x

### 5.1 Cost model

Every series fits `T(N) = c0*N + c1*N^2` (least squares on `T/N` vs `N`), i.e. a
fixed per-token cost plus a term linear in position:

| series | c0 (ms/token, position-independent) | c1 (ms/token^2) | fit residuals |
|---|---:|---:|---|
| Gemma-4-31B NVFP4, per-token | 28.24 | 0.000631 | -2.1% .. +3.0% |
| Gemma-4-31B NVFP4, chunked m=16 | 10.42 | 0.000417 | -2.8% .. +4.3% |
| Qwen3.5-9B NVFP4, per-token | 10.88 | 0.002136 | -0.1% .. +0.1% |
| Laguna-XS-2.1 NVFP4, per-token | 14.11 | 0.003138 | -8.5% .. +10.7% (3 points only) |

`c0` is the position-independent part -- the weight sweep, paid once per prompt
token on the replay path and amortised over the chunk where one exists; `c1` is
attention over the KV that has accumulated so far. Chunking on the 31B cuts
`c0` by 2.71x and `c1` by only 1.51x -- consistent with §5.3, where the chunked
graph batches the projections but still issues attention per row.

### 5.2 It is not command-buffer submission overhead

With `NV_WGPU_PROFILE=1` on the 31B, a 31-step decode replay (the same M=1 pass
list the per-token prefill replays, plus the lm-head) reported:

```
wall 1167.246 ms / 31 graph steps; GPU-attributed 890.854 ms = 76.3% of wall;
dispatches seen 16988 of 16988 expected (0 missed); timestamp_query=true
```

Read that carefully: profiling itself inflates the wall (37.65 ms/token
profiled vs 26.2-27.5 ms/token unprofiled at the same ~480-token prompt, because
each dispatch becomes its own timestamped compute pass), so **the GPU-busy time
alone, 28.7 ms/token, already exceeds the whole unprofiled per-token wall**. The 23.7% "not
GPU-attributed" is an upper bound produced by the instrument, not headroom.

Composition of that GPU time (per-dispatch profile, same run):

| pass | dispatches/step | GPU ms | share |
|---|---:|---:|---:|
| `gemv_int8_group_gelu_sg` | 60 | 288.25 | 32.4% |
| `g4w_gemv_int8_pk` | 120 | 230.84 | 25.9% |
| `g4w_gemv_int8_pk3` (lm-head) | 50 | 99.41 | 11.2% |
| `flash_splitk_stage1_fp8kv` | 60 | 65.72 | 7.4% |
| `g4w_gemv_bf16_vec8_pk` (lm-head) | 2 | 54.11 | 6.1% |
| norms (`norm_add_norm` + `norm_res_norm`) | 120 | 93.55 | 10.5% |
| everything else | 136 | 58.99 | 6.6% |

**So the per-token replay is not paying for command buffers, it is paying to
re-read the weights once per prompt token at arithmetic intensity 1.** Task
#28's title ("one token per command-buffer replay") describes the shape
correctly but mis-attributes the cost; a port that only batches *submissions*
would buy approximately nothing.

### 5.3 The dispatch budget explains the 2.4-2.7x exactly

From the profile, the 548 dispatches of a decode step decompose as 60 layers x 7
(2 norms, `gemv_int8_gelu`, 2 `gemv_int8_pk`, `flash1`, `flash2`) + 2 entry
passes + a ~126-dispatch lm-head block. A per-token *prefill* step replays
`passes[..head_start]`, i.e. **422 dispatches/token** (±1, depending on which
side of `head_start` the final RMS norm falls).

The chunked graph issues **2,942 dispatches per 16-token chunk = 183.9
dispatches/token**, because `gemma4_wgpu.rs:5050` builds attention **per row**
inside the chunk (`for t in 0..m` pushing `flash1`+`flash2` per token) while the
projections and MLP run as batched M-row passes:

| per layer | per-token replay | chunked m=16 | per prompt token |
|---|---:|---:|---|
| non-attention (norms, projections, MLP) | 5 | 17 per chunk | 5 -> 1.06 (**4.7x fewer**) |
| attention (`flash1`, `flash2`) | 2 | 32 per chunk | 2 -> 2 (**unchanged**) |
| total | 7 | 49 per chunk | 7 -> 3.06 (**2.29x fewer**) |

Wall time tracks the dispatch count at a nearly constant cost per dispatch:

| | dispatches/prompt token | ms/token at N=480 | us/dispatch | ms/token at N~8.8k | us/dispatch |
|---|---:|---:|---:|---:|---:|
| per-token replay | 422 | 27.71 | 65.7 | 33.41 | 79.2 |
| chunked m=16 | 183.9 | 10.18 | 55.4 | 13.92 | 75.7 |

**Consequence for the port: the reachable factor is set by the attention floor,
not by `m`.** At m=16 attention is already 65% of the chunked dispatch budget,
so raising `m` alone asymptotes at ~7/2 = 3.5x. Batching attention over the
chunk is where the next factor lives. Any port that claims more than ~2.3-2.7x
without touching attention is claiming something this box did not do.

## 6. Controls, and what load does

- **Null control (A/B/A).** The chunked configuration was measured in two
  separate boots (`28-g31-chunkedA.log`, `28-g31-chunkedA2.log`) with the
  per-token boot in between. Matched points differ by 0.03% / 0.3% / 0.8%
  (8,824 / 2,213 / 480 tokens). Worst single-request spread anywhere in the
  round is 4.4%. The claimed effect is 140-172%: 175x the matched-boot null and
  32x the worst single-request spread.
- **Load.** `28-g31-chunkedA2` rep 0 ran at load average 14.7 and rep 1 at 4.6;
  the two differ by 3.2% at 480 tokens and 0.7% at 8,824. On this box, with the
  work GPU-resident, a 3x swing in load average moves these numbers by ≤3.5% --
  it does not survive as an explanation for anything above.
- **Rule deviation, stated.** `../book/08.4-PERFORMANCE.md` requires
  configurations to be compared interleaved in one process. `chunked_prefill_enabled()`
  (`chat_engine_wgpu.rs:1786`) reads a process env var, so chunked and
  per-token cannot coexist in one process; the A/B here is across boots, which
  the A/B/A null above bounds at ≤0.8%.
- **Flag reached the builder.** `NV_WGPU_CHAT_CHUNKED_PREFILL=0` was confirmed
  in `/proc/<pid>/environ` of the serving process, the boot line still printed
  the chunked graph (`m=16 rows, 2942 passes/chunk`, since the knob is a serving
  seam and not a graph switch), and the effect is 175x the null.

## 7. The table a port is judged against

No row here is a raw measurement: 512 / 2,048 / 8,192 are not lengths the
tokenizer handed back, so every cell is the §5.1 fit evaluated at that length
(**fit** = interpolation between measured points in §3/§4, carrying that
series' residuals: ±4% for the 31B, ±0.1% for Qwen3.5, ±11% for Laguna).
"After a port" for kinds with no chunked path applies the 31B's *measured*
ratio at the same length (2.69x / 2.62x / 2.41x) and is therefore a *projection
that assumes the port reproduces `gemma4_wgpu`'s shape* -- batched M-row
projections, per-row attention. It is not a promise.

| decoder / checkpoint | today at 512 | at 2,048 | at 8,192 | after a `gemma4_wgpu`-shaped port |
|---|---:|---:|---:|---|
| **Gemma4Dense** `nvidia/Gemma-4-31B-IT-NVFP4` -- already ported | 5.4 s (chunked, **fit**) | 23.1 s | 113.3 s | already banked: per-token would be 14.6 / 60.5 / 273.7 s, so the shipped path already saves 9.2 / 37.4 / 160.3 s (both columns measured in §3) |
| **Qwen3_5Dense** `ig1/Qwen3.5-9B-NVFP4` -- code present, dead on NVFP4 | 6.1 s (**fit**) | 31.2 s | 232.5 s | 2.3 / 11.9 / 96.5 s projected; the work is an M-row arm for NVFP4 projections, not a new prefill graph |
| **Laguna** `poolside/Laguna-XS-2.1-NVFP4` -- no path | 8.0 s (**fit**) | 42.1 s | 326.2 s | 3.0 / 16.1 / 135.4 s projected |
| **GptOss** `openai/gpt-oss-20b` -- no path | not measured | not measured | not measured | unknown; checkpoint is cached (36 GB) and boots, it was time that ran out |
| **Qwen3_5Moe** -- no path | **cannot measure** | | | no checkpoint of that family exists on this box (§8) |
| **Gemma4E4b** -- has the path | not measured | not measured | not measured | already ported; note its non-chunked fallback pays the lm-head per token (§2) |
| **Gemma4Moe** -- has the path | not measured | not measured | not measured | already ported at m=8 |

The measured points behind these fits are in §3 and §4 (480 / 1,065 / 2,213 /
4,258 / 8,824 on the 31B, 476 - 8,459 on Qwen3.5, 491 / 2,087 / 8,213 on
Laguna). Prefer them to the fits whenever a measured length is close enough.

## 8. What I could not measure, and why

1. **Qwen3_5Moe: unmeasured, but measurable.** This section originally said no
   Qwen3.5/3.6 MoE snapshot exists on this box; that is stale.
   `RedHatAI/Qwen3.6-35B-A3B-NVFP4` is cached (21 GB, `model.safetensors` +
   `model_mtp.safetensors`, snapshot `e850c696`), so a `Qwen3_5Moe`
   chunked-prefill port can be measured against this page. `Qwen/Qwen3.5-35B-A3B`
   and `Qwen/Qwen3.5-397B-A17B` remain metadata-only stubs. Every Qwen MoE cell
   above is blank, not estimated.
2. **GptOss, Gemma4E4b, Gemma4Moe: not booted.** Time, not capability. Each is
   one boot of the same harness; the commands are in §9.
3. **Prefill-phase GPU-attributed time is unobtainable without a code change.**
   The server resets the dispatch profiler at the decode boundary
   (`chat_engine_wgpu.rs:2337`), so every prefill dispatch is discarded before
   `report()`. §5.2 substitutes the decode replay -- the same M=1 pass list plus
   the lm-head -- and says so. A prefill-phase profile needs a `profile::reset()`
   before `t_prefill`, which is a decoder-adjacent edit this round was not
   allowed to make.
4. **The submission-overhead / kernel-time split is bounded, not resolved.**
   Profiling inflates wall by ~35%, so §5.2 yields "≤23.7%, and probably far
   less" rather than a number.
5. **No CUDA comparison.** The binary was built `--features wgpu` with no cuda
   backend, so nothing here says anything about the CUDA prefill path.
6. **One KV window, one corpus.** Every boot used `max_seq=10240`; RoPE tables
   and KV buffers are sized at load, so another window may move `c1`. All
   prompts are English technical prose from `docs/book`; prefill cost should be
   content-independent, but that was not tested.
7. **Laguna's fit rests on three points** and its residuals are ±11%; its
   512/2,048/8,192 row is the weakest projection on this page. Laguna also
   returned empty assistant text at every length, consistent with the open BOS
   == EOS defect (task #58); that affects decode, not the prefill timer used
   here, but it means these runs are not a coherence check for Laguna.
8. **`nvidia-smi` cannot see wgpu memory per process.** Vulkan compute
   allocations do not appear in `--query-compute-apps`; the occupancy column in
   the JSON is whole-device `memory.used` (baseline 13,208 MiB, peak 54,419 MiB
   with the 31B resident).

## 9. Reproducing this

```sh
# one release binary, wgpu only, no nvcc
NVK_LANE=round14 NVK_PKG=speaches-plus NVK_FEATURES=wgpu \
  rust/scripts/nvk.sh build --release --bin speaches-plus

# boot: source the cached devshell for the Vulkan loader, then
NV_SERVE_BACKEND=wgpu \
NV_WGPU_CHAT_MODEL_DIRS=<snapshot dir> \
NV_WGPU_CHAT_MAX_SEQ=10240 \
RUST_LOG=info \
  <target>/release/speaches-plus --host 127.0.0.1 --port 1399

# per-token column: add NV_WGPU_CHAT_CHUNKED_PREFILL=0 and verify it landed
tr '\0' '\n' < /proc/$(pgrep -f release/speaches-plus)/environ | grep CHUNKED

# per-dispatch GPU profile (decode phase only): add NV_WGPU_PROFILE=1
```

Then POST `/v1/chat/completions` with `temperature=0` and read `prefill_ms` off
the server's own `wgpu chat request complete` line; do not time it from the
client, which adds 37-461 ms of HTTP and tokenisation.

Grep every boot for the lines that decide which path ran:

- `chunked prefill: m=16 rows, 2942 passes/chunk` (gemma4 dense, engaged)
- `[q3d-wgpu] chunked prefill disabled: nvfp4 mlp projections` (qwen3.5 dense,
  compiled out)
- `[gemma4_wgpu] chunked prefill off: an nvfp4 projection survived to the graph
  and has no M-row arm` (the other way it can vanish)
- `wgpu decoder resident ... passes_per_token=N` (the dispatch budget every
  number in §5.3 is built on)

A port is proven against this page when, at the same checkpoint, window and
prompt lengths, it moves the §4 rows toward the §7 projections **and** its boot
line shows a `passes_per_token` / passes-per-chunk pair that explains the move.
