# DiffusionGemma -- what is established, and why it is a note and not a task

`google/diffusiongemma-26B-A4B-it`, released 2026-06-10, Apache 2.0.
Google's first open-weight text-diffusion model.

**Status: not a task.** Task #45 ("Support DiffusionGemma: block-diffusion
decode loop over a 256-token canvas") is demoted to this note. It cannot
be started on this box (no checkpoint cached; 18.8 GB for the NVFP4
variant, 51.6 GB for bf16), **nothing at HEAD is red because of it**, the
refusal is precise and test-gated so nobody can be mis-served, and
Google's own published benchmarks put the model below standard Gemma 4 on
quality. Every other open item in the tracker is a wrong number or a
latent corruption on a model that *is* cached here; this one is a new
capability. It ranks below all of them.

This file exists so the research does not get re-derived. Everything below
was measured -- from the real `config.json` / index / scheduler JSON over
HTTP, or from the locally cached `google/gemma-4-26B-A4B-it` sibling --
not read off a blog post. Where something is inferred it says so.

Verified 2026-08-11 against the 2026-08-10 tree.

## The hazard, and where the refusal lives

The model's transformer is gemma-4-26B-A4B-shaped. Its weights would load
and its forward pass would run, and the output would be garbage, because
it was trained to denoise a masked canvas rather than to predict the next
token. That is the #34 failure mode -- a model silently mis-served by a
decoder built for a different objective -- with a much better disguise.

Step one landed 2026-08-09 ("recognise DiffusionGemma and refuse it
precisely, before it can be mis-served") and is intact at HEAD:

All of it lives on the `ModelClass::DiffusionGemma` variant in
`rust/src/oapi/backend_select.rs`:

- The id match sits **before** the gemma arms. This ordering is
  load-bearing: `diffusiongemma-26B-A4B-it` contains both `"gemma"` and
  `"a4b"`, so an id-substring classifier that checked the MoE arm first
  would return `Gemma4Moe` and report the model servable.
- Canonical id `"diffusion-gemma-26b-a4b"`.
- `wgpu_decoder` returns `None`, and `wgpu_absent_note` explains the
  identical-config trap instead of saying "unsupported".
- `KNOWN_MODELS` carries rows for the google and nvidia repos.
- `detect_family` (`rust/src/oapi/chat_engine/build.rs`) has no arm, which
  is the cuda refusal.
- `classify_wgpu_model` (`rust/src/oapi/chat_engine_wgpu.rs`) bails by
  name, reading `canvas_length` out of the config rather than hardcoding
  256.

`rust/tests/backend_wgpu_gate.rs` asserts the refusal names the model and
that it is servable on neither backend, so this stays true without anyone
re-reading the file.

Gated by `rust/tests/backend_wgpu_gate.rs`:
`:23-24` (classification), `:146-151` (the refusal text must contain
"not autoregressive" and "block-diffusion"), `:290-299` (auto-select must
aggregate *both* backends' reasons and return `NoBackend`, since the model
is refused on cuda and wgpu alike). Not culled by the 2026-08-10 test cull.

`docs/book/01.4-STATUS.md` records the refusal per backend and lists the
decode loop as open; `docs/book/05-backends.md` (routing hazards) explains the
named refusal and the ordering hazard.

## The four unknowns that are now answered

Task #45 listed five unknowns and said "do not infer these from blog
posts". Four are answered from the real artifacts. Do not re-derive them.

**1. The mask token is `<mask>`, id 4.** `tokenizer_config.json` gives
`mask_token: "<mask>"`. Its `added_tokens_decoder` is empty, so the id
comes from `tokenizer.json` -- a 32 MB LFS file, but only its head is
needed, and an HTTP range request over
`.../resolve/main/tokenizer.json` returns the `added_tokens` array
directly: `0 <pad>, 1 <eos>, 2 <bos>, 3 <unk>, 4 <mask>`. This is
**confirmed from DiffusionGemma's own tokenizer**, not inferred from the
sibling -- and it agrees with the cached
`google/gemma-4-26B-A4B-it/tokenizer.json`, which has the identical
24-entry `added_tokens` table.

**2. The schedule is two configs, not one.** `scheduler/scheduler_config.json`:

```json
{"_class_name": "BlockRefinementScheduler", "_diffusers_version": "0.39.0.dev0",
 "block_length": 32, "num_inference_steps": 32,
 "threshold": 0.95, "minimal_topk": 1, "editing_threshold": null}
```

`generation_config.json`:

```json
{"max_denoising_steps": 48, "max_new_tokens": 256,
 "sampler_config": {"_cls_name": "EntropyBoundSamplerConfig", "entropy_bound": 0.1},
 "confidence_threshold": 0.005, "stability_threshold": 1,
 "t_max": 0.8, "t_min": 0.4,
 "eos_token_id": [1, 106, 50], "pad_token_id": 0,
 "transformers_version": "5.8.0.dev0"}
```

Note `eos_token_id` is **`[1, 106, 50]`** here versus `[1, 106]` in
`config.json`. Resolving those ids against the shared Gemma vocab:
1 = `<eos>`, 106 = `<turn|>`, 50 = `<|tool_response>`. A third terminator
the task text does not know about, and an *opening* tool marker at that.

`model_index.json` ties it together:

```json
{"_class_name": "DiffusionGemmaPipeline",
 "model": ["transformers", "DiffusionGemmaForBlockDiffusion"],
 "processor": ["transformers", "Gemma4Processor"],
 "scheduler": ["diffusers", "BlockRefinementScheduler"]}
```

**3. The NVFP4 quant layout is this repo's preferred shape, not the
Qwen3.5-9B pathology.** `nvidia/diffusiongemma-26B-A4B-it-NVFP4`:
`model.safetensors.index.json` metadata is
`total_parameters 14,404,786,224 / total_size 18,818,050,776` over 47,067
tensors, and `quantization_config` is
`{quant_algo: NVFP4, quant_method: modelopt, producer: modelopt
0.45.0.dev127, group_size 16 weights and input_activations,
kv_cache_scheme: fp8 e4m3}` with

```
ignore: ["lm_head", "*embed_vision*", "*mlp*", "*router*",
         "*self_attn*", "*self_conditioning*", "*vision_tower*"]
```

Those are wildcards, so **only `experts.*` is NVFP4**; attention, the
dense MLP, the router and the whole vision tower stay bf16. Same shape as
this repo's `Gemma-4-31B-IT-NVFP4` (attention bf16, FFN nvfp4). 18.8 GB
fits alongside a co-tenant on this card. The Qwen3.5-9B lesson (231
`ignore` entries leaving the visual tower unquantized and the file 11.24 GB
instead of ~7 GB) does **not** apply here.

**4. Encoder and decoder share the transformer; the "encoder" is the
vision path plus one scalar per layer.** From the bf16
`model.safetensors.index.json` (1047 tensors, `total_parameters
25,823,778,864`, `total_size 51,647,562,456`):

- `model.decoder.*` -- 661 tensors. Per-layer names are **identical** to
  `gemma4_moe`'s: `experts.gate_up_proj`, `experts.down_proj`,
  `router.{proj.weight,scale,per_expert_scale}`, the seven norms,
  `layer_scalar`, `self_attn.{q,k,o}_proj` + `{q,k}_norm`, and the dense
  `mlp.{gate,up,down}_proj`.
- `model.encoder.*` -- 386 tensors: `vision_tower.*` (355),
  `embed_vision.embedding_projection.weight` (1), and
  **`language_model.layers.N.layer_scalar` (30)**. There is no second
  text transformer. The encoder's view of the language model differs from
  the decoder's by exactly **one scalar per layer**.

So the model costs 26 B params, not 52 B, and the memory case is far
better than "encoder + decoder" suggests. Arithmetic:
`1047 = 1013 (the gemma-4-26B tensor count) + 4 + 30`.

The 34 tensors `gemma4_moe` has no loader for are those 30 encoder
`layer_scalar`s plus 4 decoder-only ones:
`model.decoder.self_conditioning.{pre_norm,gate_proj,up_proj,down_proj}` --
a single gated MLP, not per-layer.

**Still genuinely unknown:** the exact per-step unmasking rule -- how
`entropy_bound 0.1`, `threshold 0.95`, `minimal_topk 1`,
`confidence_threshold 0.005`, `stability_threshold 1` and `t_max 0.8 ..
t_min 0.4` compose into a choice of which positions to unmask. That needs
the `BlockRefinementScheduler` and `DiffusionGemmaForBlockDiffusion`
sources from diffusers 0.39.0.dev0 / transformers 5.8.0.dev0. It is a
source read: no GPU, no weights. Also open: whether attention is causal
*across* blocks, which is the single biggest perf question (below).

## The decode shape

- A **256-token canvas** (`config.json: canvas_length 256`, matched by
  `generation_config.json: max_new_tokens 256`), denoised rather than
  extended.
- Refined in **32-token blocks** (`block_length 32`), **32 inference
  steps** per the scheduler, with a **48-step ceiling**
  (`max_denoising_steps`). The task text's "denoised in parallel,
  typically 12-16 steps" understates the structure: blocks are
  sequential, and 8 of them tile the canvas.
- Per step, unmask the **lowest-entropy** positions subject to an
  entropy bound (`EntropyBoundSamplerConfig`, `entropy_bound 0.1`) with a
  confidence `threshold 0.95` and `minimal_topk 1` as the floor, i.e. at
  least one position is committed per step so the loop cannot stall.
- **Bidirectional attention over the canvas**, with an autoregressive
  encoder for the prompt.

The block structure is the answer-shape to the KV-cache question: because
blocks are refined **sequentially**, the prompt and already-committed
blocks plausibly keep a normal causal KV cache, and only the 32-token
active block needs bidirectional attention. That is a 32-wide window, not
a 256-wide one -- a much smaller ask of the attention kernels. It is a
strong inference from the scheduler config, **not** a verified property of
the modeling code.

One trap worth recording: `text_config.use_bidirectional_attention` is
`"vision"`, and it is `"vision"` in the cached `gemma-4-26B-A4B-it` too.
The canvas bidirectionality is **not** declared in the config. Do not go
looking for it there.

## Three premise corrections to task #45

**A. "text_config is field-for-field IDENTICAL to gemma-4-26B-A4B-it" is
false**, though it is close. Diffing DiffusionGemma's live `text_config`
against the cached `gemma-4-26B-A4B-it` one: 28 of 29 shared keys are
equal, `model_type` differs (`diffusion_gemma_text` vs `gemma4_text`), and
DiffusionGemma **omits seven keys** that gemma-4 carries:
`attention_k_eq_v` (true in G4), `enable_moe_block` (true),
`num_kv_shared_layers`, `use_cache`, `hidden_size_per_layer_input`,
`vocab_size_per_layer_input`, `use_double_wide_mlp`. DiffusionGemma
introduces no key of its own.

Two of those omissions bite `Gemma4Config` specifically:

- `attention_k_eq_v` (`nv-models/src/gemma4.rs:79`) is a plain `bool`
  with **no `#[serde(default)]`** -- so DiffusionGemma's `text_config`
  will not deserialize at all. "Reuse the gemma4 MoE text config" is not
  a drop-in.
- `enable_moe_block` (`:91`) **does** have `#[serde(default)]`, so it
  would quietly deserialize to `false` -- and `false` routes to the dense
  decoder. That is the worse of the two: a silent misroute rather than a
  parse error, on a model that certainly is MoE (it has
  `experts.gate_up_proj` on all 30 layers).

Behaviourally `k_eq_v` *is* true for DiffusionGemma: exactly 25 of 30
decoder layers carry `self_attn.v_proj`, and the five that do not are
layers **5, 11, 17, 23, 29** -- precisely the `full_attention` entries in
`layer_types`. The cached `gemma-4-26B-A4B-it` has the identical
25-of-30 / same-five-indices layout.

**B. The weight-loading path is not reusable as-is.** The namespace is
`model.decoder.layers.N.*` where `gemma4_moe` expects
`model.language_model.layers.N.*`, plus the 34 unmapped tensors above.
Per-layer names inside the prefix are otherwise identical, so this is a
prefix remap plus four new loaders, not a rewrite.

**C. The scheduler is a shipped artifact, not something to design.** See
the two configs above. Increments B-E of task #45 were all built on
premises A-C, which is why the note supersedes the task rather than
merely deprioritising it.

## What it would cost to start

- **Fetch.** Nothing is cached: `ls -d ~/.cache/huggingface/hub/*diffusiongemma*`
  fails, and `grep -c diffusiongemma flake.nix` is 0. 18.8 GB for
  `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (the variant matching this
  repo's preferred format), 51.6 GB for `google/diffusiongemma-26B-A4B-it`
  bf16. All four published repos resolve
  (HTTP 200 on `/raw/main/config.json`, checked 2026-08-11):
  `google/diffusiongemma-26B-A4B-it`,
  `nvidia/diffusiongemma-26B-A4B-it-NVFP4`,
  `RedHatAI/diffusiongemma-26B-A4B-it-NVFP4`,
  `RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic`.
  Pinning uses `flake.nix`'s `fileTreeHash` mechanism (`flake.nix:272-305`).
- **Reuse base.** `nv-models/src/gemma4_moe.rs` (819 lines) and
  `gemma4_moe_wgpu.rs` (5045 lines). The transformer, MoE routing, RoPE
  and norms carry over.
- **New work.** A config type that does not fight `Gemma4Config`'s
  non-defaulted fields; a `model.decoder.*` prefix remap plus loaders for
  `self_conditioning` and the encoder `layer_scalar`s; the canvas loop
  itself; and the attention-mask work (bidirectional within a 32-token
  block). The last is where the existing flash / paged-KV kernels may not
  apply.
- **Effort: LARGE.** Multi-file, behaviour-changing, needs a campaign.
- **The number to beat** is the vendor's ~4x-autoregressive claim
  (Google reports >1000 tok/s on a single H100). Anything less means the
  parallelism is not being realised, and the model is already below
  standard Gemma 4 on quality, so a slow implementation buys nothing.

**Gating rule, unchanged and non-negotiable:** keep the precise refusal in
place until the bidirectional-mask work is proven. A nearly-identical
`text_config` means a half-finished implementation loads and emits
plausible-looking garbage instead of failing loudly.

## The block-diffusion connection: DFlash

This repo already ships the same mechanism, applied to the *drafter*
instead of the target. `docs/book/03-speculative-decoding.md` § "DFlash"
records it: DFlash's parallel drafting is exactly the
`[anchor, MASK, MASK, ...]` block, and the round is not autoregressive
within itself. Two implementations live here --
`rust/crates/nv-specdecode/src/dflash.rs` (standalone, carries its own
`embed_tokens` and `lm_head`) and
`rust/crates/nv-models/src/laguna_dflash.rs` (borrows the target's
`embed_weight()` / `lm_head()`).

The practical consequence: the masked-block forward, the "which positions
do I commit this step" question, and the mask-token plumbing are **not
new to this codebase**. A DiffusionGemma canvas loop is DFlash's block
applied to the target model at 32-token width with a scheduler on top.
Anyone starting #45 should read the DFlash implementation first -- it is
the closest existing thing by a wide margin, and it runs.

Trap that transfers with the mechanism: `NV_DRAFTER=dflash` silently runs
**non-speculative** if `NV_DFLASH_DRAFT_DIR` is unset, and a sweep once
produced 14 plausible cells with zero spec decode before anyone noticed.
A canvas loop that silently degrades to autoregressive decode would be the
same failure with a worse blast radius.

## Sources, each checked

Upstream (all HTTP 200 on 2026-08-11; artifacts quoted verbatim above):
`google/diffusiongemma-26B-A4B-it` (config, generation_config, scheduler
config, model_index, tokenizer_config, safetensors index, tokenizer.json
head), `nvidia/diffusiongemma-26B-A4B-it-NVFP4` (quantization_config, index),
`ai.google.dev/gemma/docs/diffusiongemma` (**prose, not an artifact** -- its
"typically 12-16 steps" disagrees with the shipped `num_inference_steps: 32`;
prefer the JSON), and arXiv:2602.06036 / `github.com/z-lab/dflash`.

In-repo: the refusal and its gate (`backend_select.rs`,
`chat_engine_wgpu.rs`, `tests/backend_wgpu_gate.rs`); the two serde fields in
`nv-models/src/gemma4.rs` (`attention_k_eq_v`, `enable_moe_block`) that decide
loud-fail vs quiet-misroute; the reuse base (`gemma4_moe.rs`,
`gemma4_moe_wgpu.rs`); the shipping block-diffusion mechanism
(`nv-specdecode/src/dflash.rs`, `nv-models/src/laguna_dflash.rs`,
`docs/book/03-speculative-decoding.md`).

**Known dangling citation, not mine to fix:**
`rust/src/oapi/backend_select.rs` (and task #45's own text) still call the
`text_config` "field-for-field identical"; per correction A it is 28-of-29
with seven keys absent. The conclusion (name the refusal) is unaffected.
