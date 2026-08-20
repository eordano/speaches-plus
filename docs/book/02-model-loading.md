# Chapter 2 — From Checkpoint to Served Chat Model

This chapter follows the one path that turns a directory of files on disk into
an entry in `/v1/models` and a `ChatEngine` a request can resolve to. **Every
gate on that path exists because some earlier version of it failed silently**,
and the code says so in its error strings.

## Two environment variables, and why the plural exists

`registry_from_env` (`chat_engine/build.rs`) has two arms.
**`NV_CHAT_MODEL_DIRS`** (plural, checked first) is split on `,` or `:`,
trimmed, empties dropped, and handed to `build_registry_strict` with the
non-panicking `try_load_engine_dir`. **`NV_CHAT_MODEL_DIR`** (singular, the
fallback) loads exactly one directory through the *panicking* `load_engine_dir`
and wraps it in `ChatRegistry::single`. `main.rs` calls `registry_from_env()` on
a CUDA build and `chat_engine_wgpu::registry_from_env_with_wgpu()` under the
`wgpu` feature, which can extend or replace the CUDA plan
(`WgpuRegistryPlan::decide`).

The plural is not sugar for the singular. One process holds one CUDA context and
one set of weights in VRAM, so serving more than one model means loading them
all at boot — there is no lazy load, no unload, no swap. Once multiple models
share a boot, **a single bad directory must not kill the others**, which is the
entire reason the plural arm uses a different loader. The singular arm keeps the
older, stricter contract: you asked for one model, you get it or the process
dies. On a build without the `cuda` feature both variables produce a warning and
no chat routes, and `build_readiness` treats either as "chat configured", so a
chat-less boot with one set shows up as configured-but-not-live rather than as
an absent subsystem.

## `ChatRegistry`

`ChatRegistry` is `engines: Arc<HashMap<String, Arc<dyn ChatEngine>>>` plus a
`default_id` and an `order` vector. `from_engines` returns `None` for an empty
vector — that `Option` is what makes "no chat" a representable state rather than
a stub engine. **The default model is `engines[0].model_id()`**, the first entry
that actually loaded. `order` keeps first-insertion order for `/v1/models`; a
duplicate id is pushed once, but `HashMap::insert` replaces the value, so the
*last* engine with that id is served.

`resolve(model)` is `resolve_with(model, allow_unknown_model())`: `None` or
`Some("")` gives the default engine (making `model` optional for a single-model
deployment); then an exact id hit; then `model_ids::canonical_model_id(m)`
followed by a second exact lookup; then, only if
`NV_CHAT_ALLOW_UNKNOWN_MODEL` is truthy **and exactly one engine is loaded**,
the default engine with a one-shot-per-id warning; otherwise `None`, which
`chat.rs` turns into a 404 `model_not_found` (06-serving-surface.md).

**The escape hatch is deliberately crippled.** It never fires with more than one
engine (`the_escape_hatch_never_fires_with_more_than_one_engine`), because with
two models loaded "serve whatever we have" is not a defensible reading of a
typo. Even single-engine, the warning spells out the cost: the response echoes
the *served* id, so a client typo becomes indistinguishable from a hit. The
default — off — is OpenAI's behaviour, and once an adapter registry exists this
arm becomes a correctness bug (01.4-STATUS.md).

Canonicalisation exists because the id a user sees and the path an operator
configures are different strings. `model_ids.rs` maps a Nix store path
`…/<32-char lowercase hash>-hf-model-<org>-<model>[-<40-hex rev>]` and a hub
snapshot `…/models--<org>--<name>/snapshots/<rev>` onto `org/model`; anything
else keeps its basename, and `canonical_model_id` returns `None` for ids that
are already pretty. So a client may address a model by store path, by that
path's basename, or by the canonical `org/model` id.

## Every configured engine must load

`build_registry_strict(dirs, load)` makes **any entry that fails to load
fatal**, with a panic naming the directory and carrying the full `{err:#}`
chain. A model that is configured but missing from `/v1/models` is a silent
capability regression: callers routing to it get a resolution failure at request
time, far from the boot log that explains why. An empty entry list is fatal for
the same reason — to disable chat, unset `NV_CHAT_MODEL_DIRS` rather than
pointing it at nothing. The generic `load` parameter makes the policy testable
without a GPU: `registry_policy_tests` drives it with closures over fake paths,
pinning that one bad directory aborts even when the others are good.

`try_load_engine_dir` and `load_engine_dir` are the same call with different
failure contracts. **Both refuse to substitute a stub engine** — the panic text
says "Refusing to start with a stub chat engine". `try_load_engine_dir` also
spawns a background thread building a throwaway `nv_grammar::JsonConstraint` so
the grammar machinery's one-time initialisation happens off the first request's
critical path.

## Inside `NvEngineChat::try_load`

`try_load` is a pure file-existence gate before any GPU work: `config.json` and
`tokenizer.json` must be regular files, and `has_safetensors` must find
`model.safetensors`, `model.safetensors.index.json`, or any `*.safetensors`.

A GGUF-only directory fails here, before `detect_family` runs — it ships no
`config.json`. **The wgpu engine does serve one end to end** (`gguf_config_json`
synthesizes an HF-shape config, `classify_wgpu_model` routes it to
`WgpuModelKind::Gemma4Moe`, `build_decoder_with_lora` constructs
`Gemma4MoeWgpu::from_gguf`); CUDA refuses with a message naming the gap and
pointing at wgpu. Wiring the CUDA seam needs its own `LoadedModel` variant and
decode loop — `Gemma4` and `Gemma4Moe` are distinct structs with distinct decode
paths, so `Gemma4` cannot consume a GGUF MoE checkpoint "unchanged". The GGUF
ships no `tokenizer.json` either, but the flake-pinned gemma-4 E4B
`tokenizer.json` is a verified id-for-id drop-in (all 262144 pieces exact,
controls `<|turn>`=105 / `<turn|>`=106 included), proven both ways by
`nv-models/tests/gguf_tokenizer_identity.rs`.

`try_load_inner` (CUDA only) then, in order: reads `config.json` as a string;
`detect_family`; `ChatTemplate::load` (an `Option` — absence is logged, not
fatal); opens CUDA device 0 and disables CUDA event tracking on its stream;
`WeightLoader::open_dir`; loads the tokenizer and runs
`nv_tokenizer::sanitize_for_serving`; computes the model id with
`model_ids::model_id_for_dir`; and dispatches on family.

### `detect_family` and the family enum

`detect_family(&raw_cfg)` reads `architectures` (an array) first, lowercases each
entry, and prefix-matches; failing that it tries `model_type` the same way.
**The match order is load-bearing**: `qwen3omni` and `qwen3_5moe`/`qwen3.5moe`
are tested before `qwen3_5`/`qwen3.5`, which is tested before `qwen3`, because a
shorter prefix would otherwise swallow the longer architecture and route it to
the dense loader. Recognised prefixes are `qwen3omni`/`qwen3_omni`,
`qwen3_5moe`/`qwen3.5moe`, `gemma4`, `qwen3_5`/`qwen3.5`, `qwen3`, `laguna`,
`gptoss`/`gpt_oss`.

Two arms are opt-in, and `detect_family` bails with the gate's name when the
opt-in is absent: `gpt_oss_family` requires `NV_GPTOSS_CUDA_SERVE` (the CUDA
path dequants mxfp4 to bf16 at load, costing roughly 3x the native resident
footprint, so wgpu stays the default), and `qwen3_5_dense_family` requires
`NV_QWEN35_DENSE_CUDA_SERVE`. The dense-hybrid arm reuses
`ModelFamily::Qwen3_5Moe`, and `try_load` distinguishes it by parsing
`Qwen3_5DenseConfig` and asserting `model.dense_intermediate()` matches
`config.intermediate_size`.

**An unrecognised architecture is not guessed at**: `detect_family` bails with
the list of prefixes it accepts, and that error either skips one directory
(plural arm) or panics the boot (singular arm). There is no "try the closest
loader" path — every family arm assumes a specific config schema, tensor-name
layout and KV geometry, and a wrong guess would surface as a shape mismatch or,
worse, as plausible-looking garbage.

`ModelFamily` and `LoadedModel` are parallel enums:

| `ModelFamily` | `LoadedModel` payload | Concurrency |
| --- | --- | --- |
| `Qwen3` | `Arc<Mutex<Qwen3>>` | serialised |
| `Gemma4` | `Arc<Gemma4>` | shared, no lock |
| `Gemma4E4b` | `Arc<Gemma4E4b>` | shared |
| `Gemma4Moe` | `Arc<Gemma4Moe>` | shared |
| `Qwen3_5Moe` | `QwenMoeShared::{Eager, Graphed, Batch}` | serialised, except `Batch` (a `Qwen38BatchScheduler`) |
| `Laguna` | `LagunaShared` (`Arc<Laguna>` + `unsafe impl Send/Sync`) | shared |
| `Omni` | `Arc<OmniShared>` | shared |
| `GptOss` | `Arc<GptOssCuda>` | shared |

`generate()` matches on the `(family, loaded)` pair with an
`engine family/model variant mismatch` error for the impossible combinations —
the two enums are kept in step by construction, and the arm exists so a future
divergence fails loudly instead of mis-dispatching.

Each arm also fixes `eos_token_ids`, `bos_token_id`, `default_max_new_tokens`
and `kv_max_seq_len`. **EOS resolution is per family because the checkpoints
disagree about where it lives** — Qwen3 takes it from the typed config, Gemma4
from `parse_eos_ids` over the raw JSON with a hardcoded default, Laguna from a
config list or the raw JSON, and Qwen3.5-MoE from
`resolve_qwen3_5_moe_eos_ids`, which unions `tokenizer_config.json:eos_token`
mapped through the tokenizer, the literal `<|im_end|>` id, and the config field.

`kv_max_seq_len_for` caps at 8192 by default; the Gemma4 variant defaults to the
checkpoint's full `max_position_embeddings` and then runs `fit_gemma4_kv_max`,
halving down to `KV_FIT_FLOOR` until the KV budget plus the measured weight
residency fits `NV_VRAM_BUDGET_GIB`. **`decide_gemma4_kv_max` returns a `KvFit`
enum rather than a bare number** specifically so a *failed* `nvidia-smi` probe
(`ProbeFailed`) is distinguishable from a genuine fit — when they were the same
value, auto-fit silently became a no-op and the boot failed later inside an
allocation with no explanation. The Gemma4 arm additionally loads the Eagle3 and
DFlash drafters and sets `spec_status`, and the Qwen3.8 arm loads the MTP head
and boot-fails when `NV_DRAFTER=mtp` and the head is missing
(03-speculative-decoding.md).

## Weights: `nv-weights`

`WeightLoader::open_dir` resolves the shard set two ways. With
`model.safetensors.index.json` it takes the distinct values of `weight_map` as
the shard list, erroring if the index references a missing shard or names a
tensor the shard does not contain. Without an index it globs `*.safetensors`,
sorts, and unions every shard's headers — and **a name appearing in two shards
is a hard error**, because with no index nothing arbitrates which copy is
authoritative. Shards that are Git-LFS pointers (≤1024 bytes, starting with
`version https://git-lfs.github.com/spec/`) are skipped with a note and their
tensors dropped, so a partially-fetched checkpoint fails at the first missing
tensor *by name* rather than at a safetensors header parse of ASCII text.

Each shard is `mmap`ed once; `load_shard` records `(dtype, shape, offset_start,
offset_end)` per tensor, and `get(name, dtype)` slices the mapping, builds the
tensor with `Tensor::from_raw_buffer`, and converts only if the on-disk dtype
differs (`BOOL` canonicalises to `U8`). `map_st_to_candle` accepts exactly
`BF16, F16, F32, I64, U8` and bails on anything else — **an FP8 or FP4 payload
cannot come through `get` at all**, which is why the quantized path uses
`raw_bytes`. `TensorSource` (`get` / `has`) is implemented by both
`WeightLoader` and `GgufLoader`, so a model loader is written once against
either (02.2-gguf-format.md).

### Tensor names are a contract

Model loaders ask for exact strings — `nv_models::gemma4` wants
`model.language_model.embed_tokens.weight`, `lm_head.weight`, per layer
`{prefix}.self_attn.o_proj.weight`, and so on. For NVFP4 it uses
`nv_layers::moe::Nvfp4Suffixes::GEMMA_MODELOPT`:

```
GEMMA_MODELOPT:            packed "weight",        block "weight_scale", global "weight_scale_2",  input "input_scale",         inverse=false
QWEN_COMPRESSED_TENSORS:   packed "weight_packed", block "weight_scale", global "weight_global_scale", input "input_global_scale", inverse=true
```

These are two on-disk conventions for the same numeric format. **A checkpoint
whose tensors are named for the other convention will not load through the
generic path.** The Gemma4 projection loader tests for `{module}.weight` and
`{module}.weight_scale_2`; a compressed-tensors pack-quantized checkpoint has
neither (its payload is `{module}.weight_packed`), so the loader falls back to
the plain bf16 name — also absent — and fails with `tensor not found: …`.
Loaders that *do* understand that layout name it explicitly: `nv_layers::moe`
probes `{module}.weight_packed`, and `gemma4_e4b::dequant_pack_quantized`
handles the w4a16 pack-quantized E4B checkpoint. **Support for a checkpoint is a
property of the (family loader × naming convention) pair, not of the format
alone**; 02.1-model-compat-matrix.md is the per-checkpoint table.

### Quantization detection

`QuantizationConfig::from_hf_json_str` reads the top-level `quantization_config`
key (absent → `QuantScheme::None`) and runs `parse_hf_value` then `validate`.
`parse_hf_value` tries, in order: `quant_method == "modelopt"` with `quant_algo`
containing `NVFP4`/`FP4` → `Nvfp4` or `FP8` → `Fp8E4m3`; `format` containing
`nvfp4`/`fp4` → `Nvfp4`, or `quantization_type == "fp8"` / `format` containing
`fp8` → `Fp8E4m3`; `config_groups.*.weights` with `type: "float"` and `num_bits`
4 → `Nvfp4`, 8 → `Fp8E4m3`; otherwise `Self::none()`. **That last step is the
trap**: an unrecognised `quant_method` falls through silently to
`QuantScheme::None`, so a checkpoint declaring AWQ or GPTQ is treated as
unquantized, the loader goes looking for bf16 `.weight` tensors it does not
have, and the failure surfaces as a missing-tensor error one layer from its
cause.

What *is* rejected is a config that would change numerics behind your back.
`validate()` refuses NVFP4 with any `group_size` other than `NVFP4_GROUP_SIZE`
(= `nv_quant::nvfp4::BLOCK_SIZE`), naming MXFP4 in the message, because that
field was once parsed and never read while the block size stayed hardcoded — a
4-bit-float checkpoint with 32-element groups loaded as NVFP4 and mis-scaled
every block. FP8 with any declared `group_size` is refused too, because this
build scales FP8 per output row, not per block. `ignore` / `ignored_modules`
entries are matched exactly or as a `prefix*` glob by `is_module_ignored`; an
ignored module loads unquantized.

`load_quantized_weight` pulls the packed payload with `raw_bytes` and hunts the
scale under several candidate names. The FP8 arm calls `fp8_weight_scale_rows()`
**at load time**, canonicalising `weight_scale` to one f32 per output row and
refusing anything neither per-tensor nor per-row (a per-column or block-wise
grid is a different numeric contract and is not averaged into one), along with
non-finite and non-positive scales. **Doing this at load rather than at the
first matmul is the point**: this value used to be discarded by the runtime
path, so it had never been shape-checked at all.

## Tokenizer

`nv_tokenizer::sanitize_for_serving` clears truncation and padding on the loaded
`tokenizers::Tokenizer`, because some shipped `tokenizer.json` files configure
truncation and honouring it would silently cap prompt length at the checkpoint
author's fine-tuning window.

`IncrementalDecoder` is the streaming counterpart: it decodes
`ids[prefix_offset..]` against `ids[prefix_offset..read_offset]` and emits the
delta only when the longer decode is longer, ends on a char boundary, and does
not end in U+FFFD, so **a stream never emits a replacement character for a token
that was merely incomplete.** `flush()` releases whatever is left. Note that
`nv-tokenizer` also exposes a small `ChatTemplate` of its own; the serving path
does not use it, it uses `oapi/chat_template.rs`.

## Chat templates

`ChatTemplate::load(model_dir)` returns `Option<Arc<ChatTemplate>>` and records
every attempt — directory plus error — in a process-global list read later by
`load_attempt_for` / `load_was_attempted`, which is what the fallback policy
keys on. Source resolution: `chat_template.jinja` if present and non-blank, else
`tokenizer_config.json:chat_template`, which may be a string or an array of
`{name, template}` objects (the entry named `default` wins, otherwise the first
with a template).

The template compiles with minijinja plus four shims: `pycompat`'s
`unknown_method_callback` so Python string methods resolve; `raise_exception`,
which HF templates call to reject malformed message sequences and which becomes
a render failure; `strftime_now`, backed by a self-contained UTC civil-date
formatter rather than a date library; and `strip_generation_tags`, which
rewrites HF's `{% generation %}` markers into Jinja comments, since minijinja
does not know those tags and they carry no inference-time meaning. `bos_token`
and `eos_token` come from `tokenizer_config.json` and are injected alongside
`messages`, `tools` and `add_generation_prompt`. Template kwargs layer:
`default_chat_template_kwargs` from `generation_config.json` then
`tokenizer_config.json` (first wins), then `NV_CHAT_TEMPLATE_KWARGS` (a JSON
object; invalid values logged and ignored), then per-request
`chat_template_kwargs`.

**Three predicates read the template rather than a hardcoded model list**:
`declares_thinking_switch()` (source mentions `enable_thinking`),
`uses_tool_responses()`, and `supports_tools()` — the last renders a probe
conversation twice, with and without a synthetic tool, and reports whether the
output differed, so a template that ignores `tools` reports `false` and
`render_official_with_kwargs` falls back to flattening tools into a synthetic
system message with a warning.

### The built-in renderer and why it is refused by default

`ChatEngine::render_chat` tries `official_template()` first; if it is absent or
errors, it calls `note_builtin_fallback(reason)` and falls through to
`render_prompt`, dispatched per family (`render_chat_prompt` — ChatML — for
Qwen3, `render_gemma4_prompt`, `render_qwen3_5_moe_prompt`,
`render_laguna_prompt`). `render_chat_checked_kwargs` wraps that: if a fallback
note was set *and* the template is required, it returns an error the handler
turns into a 500 `chat_template_missing` rather than serving a wrongly-shaped
prompt. `template_required_for(model_id)` resolves the policy:
`NV_REQUIRE_CHAT_TEMPLATE` truthy means required always;
`NV_ALLOW_CHATML_FALLBACK` truthy or `NV_REQUIRE_CHAT_TEMPLATE` explicitly off
means not required; otherwise required iff a template load was attempted for
that model id, so model-backed engines are strict and test/echo engines are not.
**The reason for the default is mechanical, not aesthetic**: ChatML's
`<|im_start|>role … <|im_end|>` is not Gemma-4's `<|turn>role … <turn|>` and it
does not emit Qwen3.6's `<think>` opener, and feeding a model control tokens it
was not trained on changes both what it generates and when it stops. The
fallback is an opt-in escape hatch, not a graceful degradation.

### Reasoning / think splitting

`ThinkPostProcess.active` is `engine.thinking_split_supported()` (the template
declares the switch) and `opened` is whether the rendered prompt itself already
opened the block — which some templates do as part of the generation prompt,
after which the model continues inside the thought without re-emitting
`<think>`. A model whose template does not declare the switch keeps any
`<think>` text in `content` untouched.

Non-streaming, `split_thinking(text, opened)` requires a leading `<think>` after
`trim_start` when `opened == false`, else returns `None` and splits nothing —
**that is what keeps a stray `</think>` inside a code block from being read as a
delimiter.** The body up to the first `</think>` becomes `reasoning_content` and
the left-trimmed remainder becomes `content`; with no `</think>` at all
everything is reasoning and content is empty, so **truncated reasoning is never
presented as an answer.** Streaming uses `ThinkingStream`, a three-state machine
(`Undecided` → `Reasoning` or `Content`) whose load-bearing detail is
`emittable_len`: it always retains the last `len("</think>") - 1` bytes (backed
off to a char boundary) of pending reasoning, **so a delimiter split across two
deltas is never leaked into the stream.** On the transition to `Content` it sets
`content_lead` to swallow leading whitespace, so streamed content matches what
the non-streaming splitter would have produced, and `finish()` drains whatever
is pending.

## What the registry publishes

`models_handler.rs` iterates `registry.model_ids()` and emits one `/v1/models`
row per engine: `id`, `owned_by` (the segment before `/`, else the whole id),
`task: "chat"`, and an `extras.spec_decode` of `on` / `degraded` / `off`. LoRA
adapter rows are merged into a matching registry row rather than duplicating it,
so a boot-registered adapter keeps both its `spec_decode` and its `parent` +
`lora` provenance extras. **This is the one place a client can discover the
exact ids `resolve` will accept** — a hand-written id in a config file is the
classic way to get a 404 from a server that is working perfectly.

## Connections

01-architecture.md (the crate split this path runs across);
03-speculative-decoding.md (the drafter load, the `spec_status` gate,
`NV_DRAFTER` / `NV_EAGLE3_REQUIRED` / `NV_DFLASH_REQUIRED`);
06-serving-surface.md (how `resolve` returning `None` becomes a 404, how a
required-but-missing template becomes a 500, and how `reasoning_content` appears
on the wire).
