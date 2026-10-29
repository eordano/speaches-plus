# Chapter 6 — The serving surface

Which routes exist, what backs each one, where the surface deliberately stops
being OpenAI-compatible, and what happens to an endpoint whose model never
loaded. The whole server is one Axum binary, `rust/src/main.rs`: no gateway, no
per-model process, no sidecar.

## The router is assembled from what actually loaded

Unconditional, on `AppState`: `/health`, `/health/ready`, `/version`,
`/metrics`, `/health/sessions`, `/v1/internal/chat-engines`, `/v1/realtime`
(POST for SDP, GET for the WebSocket upgrade), `/v1/realtime/capabilities`,
`/v1/audio/transcriptions`, `/v1/audio/translations`, `/v1/audio/diarization`,
`/v1/audio/embeddings`, `/v1/embeddings`, `/v1/models`, `/v1/inspect/*`; then
merged unconditionally, the `/v1/audio/speech` router,
`oapi::ocr::router_from_env()`, `oapi::fine_tuning::router()` and
`backends_report_router()` (`GET /v1/backends`).

Conditional: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`,
`/v1/messages` and `/v1/messages/count_tokens` exist only if
`chat_engine::registry_from_env()` returned a registry — otherwise `main.rs`
logs that they are disabled and never merges them, so they 404 at the router
level. `/v1/voice-profiles` requires `VoiceProfileStore::open` to have
succeeded, and the four `/v1/pii/*` routes require `REDACT_MODEL_DIR` set, the
classifier layout present, and `PiiClassifier::load` succeeded. The three
`/v1/systemone*` routes require `NV_LAYA_MODEL_DIR` set with
`laya/model.safetensors` under it, or `NV_KEV_MODEL_DIRS` naming at least one
exported Kev checkpoint.

`/v1/messages` (`oapi/messages.rs`) is an Anthropic Messages adapter over the
same engines: content blocks (`text`, `tool_use`, `tool_result`) translate to
`ChatMessageIn`, `system` becomes the leading system turn, tools map onto the
prompt-injected tool machinery, and streaming emits the named-event SSE
lifecycle (`message_start`, `ping`, `content_block_start/delta/stop`,
`message_delta`, `message_stop`, `error`) rather than OpenAI's anonymous `data:`
frames. `reasoning_content` surfaces as a `thinking` block with an empty
signature; errors use the Anthropic envelope and engine-busy sheds map to 529
`overloaded_error`; `usage` carries `input_tokens`/`output_tokens` plus
`cache_read_input_tokens` wired to the prefix-reuse machinery, mirrored on
`/v1/chat/completions` as `usage.prompt_tokens_details.cached_tokens`. A
stop-string hit is attributed only where the engine records it — the wgpu path
and the echo engine emit `ChatEvent::StoppedBy` and report
`stop_reason: "stop_sequence"`, the CUDA loops do not (the private status ledger).
`/v1/messages/count_tokens` renders the same prompt off the request path
(`spawn_blocking`) and counts it with the model dir's `tokenizer.json`, or
answers 501 honestly when no tokenizer is on disk. `/v1/responses` is the
OpenAI Responses adapter, with `store` / `previous_response_id` backed by per-id
KV snapshots (05.10-kv-disk-persistence.md).

Three middleware layers wrap the merged router, outermost last:
`DefaultBodyLimit::max(200 MiB)`, then `auth_mw`, `metrics_mw`, `cors_mw`.
`auth_mw` enforces `SPEACHES_API_KEY` with a constant-time compare against
either `Authorization: Bearer` or the Anthropic-style `x-api-key`, exempting
only `/health`, `/health/ready`, `/metrics` and `/version`; on `/v1/messages*`
the 401 body is the Anthropic envelope. **If no key is set and
`SPEACHES_CORS_ORIGIN` is still `*`, boot emits an explicit warning** that every
route is unauthenticated and callable from any web origin.

## Where OpenAI compatibility is deliberately broken

The surface is OpenAI-shaped so existing clients work, but **it refuses rather
than pretends in every case where honouring the request literally would produce
a plausible-looking wrong answer.** An unsupported field is a 4xx naming the
field and the alternative, never a silently ignored parameter.

`oapi/chat.rs`: `messages[].content` accepts a string or an array of parts —
`text`, `image_url` and `input_audio` are handled (below) and any other part
type deserializes to `ContentPart::Unsupported`; `first_unsupported_part` finds it
and the multimodal bridge in `chat.rs` either serves the part or answers
`400 unsupported_content_part`, stating outright that the request was rejected
rather than answered without the part.
`max_tokens`/`max_completion_tokens` clamp into `1..=MAX_MAX_TOKENS` (8192),
defaulting to `DEFAULT_MAX_TOKENS` (256); `n` caps at 16, `best_of` is accepted
only when it equals `n`, `top_logprobs` caps at 20. Extra request fields with no
OpenAI counterpart: `top_k`, `min_p`, `repetition_penalty`, `guided_json`,
`guided_regex`, `guided_choice`, `chat_template_kwargs`, `enable_thinking`,
`timeout`. Extra response fields: `reasoning_content` on the message and on
streamed deltas, carrying the `<think>`-fenced span split out of the model's own
text, and `system_fingerprint` synthesised from the model id hashed with the
crate version. One extra response header: `x-spec-decode`.

Reasoning tokens spend `max_tokens` first. On a reasoning-on model (Qwen3.8
thinks by default), everything the model emits inside its `<think>` fence counts
against `max_tokens` before any visible text does, and it is returned as
`reasoning_content`, not `content` — so a 350-token budget on a thinking model
typically ends `finish_reason: "length"` with a long `reasoning_content` and a
short or empty `content`. A client that wants N visible tokens from a thinking
model budgets `max_tokens` for thinking plus answer (thinking spans of several
hundred tokens are normal), or disables thinking for that request with
`enable_thinking: false` (`chat_template_kwargs` reaches the template for
finer control). `usage.completion_tokens` counts both spans; the split is
visible only in the response body.

`reasoning_effort` speaks OpenAI's spelling through
`chat_template::request_reasoning_effort`:

- **`none`** turns thinking off, as `enable_thinking: false` would. An explicit
  `enable_thinking` wins over it.
- **`minimal`** is the template's `low`.
- **`low|medium|high|xhigh`** reach a template that declares `reasoning_effort`
  (Qwen3.8; the served default is `medium`).
- Anything else is a 400.

`chat_template_kwargs` win over all of these.

`check_stt_params` on `/v1/audio/transcriptions` validates `language` against
the whisper language table (unknown codes are a 400 naming the field) and
forwards it — whisper-cpp receives it directly, ct2 skips detection and forces
the matching language token, parakeet accepts only `en`. A non-empty `prompt`
is forwarded as the decoder's initial prompt (whisper-cpp `prompt_tokens`, ct2
`<|startofprev|>` prefix, truncated to the last
`INITIAL_PROMPT_TAIL_TOKENS_HALF_WHISPER_TEXT_CTX_MINUS_ONE` tokens); a backend
that cannot honour it (parakeet, a ct2 tokenizer without `<|startofprev|>`)
answers 400 `unsupported_parameter` naming the field rather than dropping it.
A multipart `model` field naming anything but the loaded STT model (generic
`whisper`/`whisper-1` match any whisper-family backend) is a 404
`model_not_found` naming what is served. Any non-zero `temperature` is still
refused — decoding is greedy. It adds a non-OpenAI
`response_format=diarized_json`.

`oapi/audio_speech.rs` refuses an unknown `model` with `404 model_not_found`
instead of rendering it with whichever engine happens to be loaded — the error
says so explicitly, because earlier builds did substitute. `speed != 1.0` is
refused on the nv-tts path with `speed_unsupported` and a pointer at the Kokoro
model id, and `response_format` is `wav|pcm` with `mp3` an explicit refusal.
Route selection is `resolve_speech_route`, a pure function over (requested id,
loaded talker id, kokoro loaded, talker bootstrap failure) — 07-speech-stack.md.

`oapi/text_embeddings.rs` accepts only `input` as a string or array of strings;
`encoding_format` and `dimensions` are not implemented, and a requested `model`
that does not match the loaded embedder is logged and ignored, with the response
reporting the model that actually produced the vectors.

`/v1/models` serialises the OpenAI keys plus `language`, `task`, an optional
`max_model_len`, and per-model `extras` (chat rows carry `spec_decode`; adapter
rows carry `parent` and a `lora` object; TTS rows carry `sample_rate` and
`voices`). The list is built strictly from load state — a row appears only
if its subsystem is live, asserted by
`embeddings_and_pii_rows_follow_load_state_in_both_directions`.

## A chat request, end to end

`POST /v1/chat/completions` → `oapi::chat::handle_chat_completions`.
`ClientDeadline::from_request` merges the body's `timeout` with the
`x-request-timeout-ms` header, clamped into
`[50 ms, NV_MAX_REQUEST_TIMEOUT_MS]`, and the `x-spec-decode` value is computed
up front so it can be stamped on whatever comes back, errors included.
`ChatRegistry::resolve` looks the id up and retries through
`canonical_model_id`, 404ing `model_not_found` unless exactly one engine is
registered (02-model-loading.md). `tool_choice` naming an unknown function,
`n > 16`, `best_of != n`, `top_logprobs > 20` and a malformed `guided_json` are
all 400s before any GPU work, and `render_chat_checked_kwargs` prefers the
checkpoint's official chat template and refuses rather than guessing when the
template is required and absent.

Then the handler builds a `ChatGenerateRequest` and calls
`ChatEngine::generate(req, tx)`. The trait is deliberately narrow: the
engine returns `Ok(())` as soon as it has started and streams
`ChatEvent::{PromptCached, Started, TextDelta, Logprob, StoppedBy, Done, Error}`
down an `mpsc::Sender<ChatEvent>`, so HTTP never touches a model handle. The
decode loop runs on its own thread with its own model ownership; `Started`
carries `prompt_tokens`, `Done` the finish reason and `completion_tokens`.

`first_event` applies the client deadline only to the first `ChatEvent` — a
`ChatEvent::Error` whose text starts with the engine-busy prefix becomes
`503 engine_busy`, a timeout becomes a deadline shed, and **once the first token
is out the stream is committed.** Streaming goes to `run_streaming`, writing
`data: {json}\n\n` frames into a channel wrapped by `ReceiverStream` as the
response body; the frame order per choice is a role-only opening chunk, content
chunks, then — if tools are active — one chunk carrying the parsed `tool_calls`,
then a chunk with `finish_reason`, then an optional usage-only chunk when
`stream_options.include_usage` is set, then `data: [DONE]`. Non-streaming drains
the same events into one `chat.completion` body. **`n > 1` is served by
repetition, not by a batched sampler**: both runners loop `0..n`, calling
`generate` again per choice with `seed = base_seed + i`, and the streaming
runner reuses the already-started first channel for `i == 0`.

`POST /v1/completions` is the same machinery with a text prompt instead of
messages, importing `ChatAppState`, `ChatGenerateRequest`,
`resolve_guided_fields` and the SSE helpers directly from `chat.rs`, and adding
`prompt` as string or array, `suffix`, `echo`, `stop_token_ids`, and an integer
`logprobs`.

## Multimodal input

Image and audio chat parts are served by two mechanisms chosen per engine via
`ChatEngine::supports_mm_input` and `supports_audio_input`; `resolve_media_parts`
(`chat.rs`) makes the choice once for `/v1/chat/completions`, `/v1/responses`
(`input_image` parts) and `/v1/messages` (user `image` blocks, base64 or a
data-URL `url` source). **Native towers:** `extract_mm_media` converts mm parts
to `<|image>`/`<|audio>` marker text and decodes the media at the HTTP layer,
the official template renders the marked messages, and the engine expands each
marker and splices tower rows into the embedded ids. An engine with vision but
no audio encoder (Flash-Next) first has its `input_audio` parts transcribed by
the bridge, so audio keeps working when images go native. **The perception
bridge (everything else):** `bridge_mm_parts` loops back over HTTP to the
server's own `/v1/ocr` (dots, `mode=plain`) and `/v1/audio/transcriptions`
(whisper), injecting the result as `[image, transcribed by ocr]…` text parts.
The 400 remains only for unknown part kinds. A `/v1/responses` sidecar stores
image parts as the caller sent them, so `previous_response_id` resends the
image and the follow-up's prefix matches.

**Gemma-4 family.** `plan_from_marked_tokens` (`chat_multimodal.rs`) expands
each marker into BOI/soft-token/EOI runs, `mm_embeddings` splices tower outputs,
and `Gemma4E4b::forward_step_embeds` prefills from the spliced embeddings.
Towers load at engine boot (`Gemma4MmTowers::from_model_dir`), warning and
degrading to text-only on failure; a checkpoint without a tower gets
`require_vision()` / `require_audio()` naming the model.

**Qwen3.8-Flash-Next (`qwen4_exp`).** The checkpoint's own 27-block ViT and 2x2
merger (`nv_omni::Qwen3VisionTower`, the Qwen3.5 vision model: patch 16,
temporal patch 2, bilinear-resampled 48x48 position table, 2D vision rope,
merger LayerNorm then exact-erf GELU to 2560 wide) run with bf16 weights and an
f32 residual stream (`load_onto(.., f32_activations)`; HF's own bf16 run drifts
to cosine 0.993-0.999 from its f32 run, the f32 stream matches HF f32 at
cosine 1.000000). Preprocessing reproduces HF's Qwen2-VL processor bit for bit:
`smart_resize_matching_hf_qwen2_vl` (Python round-half-even) and
`resize_rgb8_bicubic_matching_pil` (PIL's fixed-point bicubic, horizontal pass
first). `chat_engine/qwen4_exp_mm.rs` expands each marker to
`<|vision_start|>` + one `<|image_pad|>` per merged token + `<|vision_end|>`,
encodes the images under the engine permit, and hands the engine a
`Qwen4ExpMm`: the bf16 rows, which replace the token embedding in all four
hyper-connection streams (`mm_splice_rows`, also into the MTP head's embedding
input); the HF `get_rope_index` positions (`build_mrope_positions_for_vision_runs`);
and the prefix key. Every `Qwen4ExpState` owns a per-token rope table
(`PER_STATE_ROPE_TABLE_BECAUSE_MROPE_ROWS_DIFFER_PER_PROMPT_AND_CAPTURED_GRAPHS_BAKE_ITS_ADDRESS`):
the interleaved [11, 11, 10] t/h/w rows for the prompt, then base row
`i + rope_delta` for every later token, so prefill, captured decode, MTP
drafts and verify, and the QSA indexer's block keys all read HF's per-token
rows; a text request restores the base rows, which keeps text logits
bit-identical. The prefix key replaces each image token with a high-bit id
derived from the image's SHA-256 (`prefix_key_with_vision_content`), so the
session cache, prefix checkpoints and parked or spilled sessions resume a
prefix only under the same image. PLE n-grams read the real `<|image_pad|>`
ids, as HF's `ple_input_ids` do. MTP self-speculation runs over image prompts
(the draft head's embedding input takes the same rows). Runtime-LoRA adapter
ids do not report `supports_mm_input` yet, so their images take the OCR
bridge. The tower is the `qwen4exp-vision`
cold-tier resident (0.84 GiB, loaded on the first image request after
`ensure_free` makes room, evicted when idle or under pressure), and
`NV_Q4EXP_VISION_MAX_PIXELS` (default 4194304, so at most 4096 image tokens
and about 1.5 GiB of transient activations per image) caps the resize.
`NV_Q4EXP_VISION=0` restores the OCR bridge. The suites covering this:
`nv-omni` `qwen4_exp_vision_hf_ref` (tower and preprocessing against an HF
dump), `nv-models` `qwen3_mm_splice` (positions and rope rows against HF) and
`qwen4_exp_vision_mm` (splice, chunking, graphed decode rows, prefix keys on
the tiny fixture), and on a card that holds Flash-Next `qwen4_exp_vision_full`
(`NV_Q4EXP_VISION_E2E=1`: six capability-gate images, greedy answers, MTP k=2
equal to the plain decode).

The rules `chat_multimodal.rs` enforces, which any future wiring inherits:
images must be inline data URLs (`decode_image_ref` accepts only
`data:<mime>;base64,<payload>`, a remote URL or filesystem path is refused, and
the unit test asserts the error does not echo the rejected path back, so **the
server never fetches on a caller's behalf** — no SSRF surface, no
path-disclosure oracle); audio must be base64 WAV (`decode_audio_input`
refuses any other `format`, mixes to mono, resamples to the tower's rate, and
the audio is clipped to `MAX_AUDIO_SAMPLES`, 30 s); `run_towers` **checks the produced row count
against the reserved run length** and fails loudly on mismatch rather than
splicing a wrong-sized block; and for a text-only plan the splice is a no-op,
bit-identical to plain embedding — a property the test suite pins. Two known
limits are in the private status ledger: the w4a16-ct E4B checkpoint is
text-only in practice, and the Gemma audio tower has architectural
verification only.

## Tool calling

Assistant messages accept both `reasoning` and `reasoning_content`. These
fields survive request parsing and official-template rendering separately
from answer text, so Laguna can retain thinking across tool results and later
user turns. Callers must return the reasoning received with each assistant
message; it cannot be reconstructed from the final answer.

Official templates use `template_tojson` for MiniJinja JSON serialization.
It accepts `ensure_ascii=false` for Unicode text and `ensure_ascii=true` for
ASCII escapes, including surrogate pairs, while retaining the built-in filter's
indentation and HTML escaping. Without `indent` it separates items and keys with
`", "` and `": "`, as Python's `json.dumps` and transformers' own `tojson` do.
Laguna's native history template uses this option for non-string tool
arguments: numbers, booleans, nulls, arrays and objects must render without
switching to the generic tool prompt. The model-independent
`laguna_tool_history_serializes_non_string_arguments_without_fallback` test
covers that history branch; first-turn tool tests cannot exercise it.


Tools are prompt-level, then parsed back out of the text; there is no separate
tool-call decoding channel. On the way in, `build_tool_messages` is the fallback
used when no official template renders tools: it prepends a system message
written by `tool_preamble` stating the
`<tool_call>{"name", "arguments"}</tool_call>` contract, hardens it for
`tool_choice: required` or a named function, and lists each tool's JSON schema.
Prior assistant `tool_calls` are re-serialised into the same block form, and
`role: "tool"`/`"function"` messages fold into user turns as
`Tool result (<who>): ...`, because the built-in renderer has no tool role.

On the way out, `tool_parse.rs::parse_tool_calls` tolerates three dialects:
`<tool_call>` blocks containing a JSON object with `name` and `arguments`
(`arguments` may itself be a string or an object); the Qwen XML dialect
`<function=name><parameter=key>value</parameter></function>`, with or without a
wrapper, where `xml_param_value` converts a parameter to a JSON number/bool/null
only when the value round-trips exactly, so `007` stays the string `"007"`;
and a forced call, where `tool_choice` naming a function (or `required` with a
single tool) treats the whole output as that call's arguments and extracts the
outermost balanced JSON object. A forced call whose object never closes (the
token cap cut it) is returned as content with its `finish_reason`, never as a
call with invented `{}` arguments. `rewrite_native_tool_calls` normalises a
model-native `<|tool_call>` encoding into the `<tool_call>` form first, so the
parser sees one dialect. Any parsed call flips `finish_reason` to `tool_calls`
and moves the text into `tool_calls[].function.arguments`, with the remaining
prose as `content`; in the streaming runner this is necessarily buffered
(`TextDelta`s accumulate into a string and the calls are emitted in one late
chunk), so **tool calling and incremental content streaming are mutually
exclusive by construction.**

How each `tool_choice` is enforced (`chat.rs::resolve_tool_policy`, shared by
`/v1/chat/completions`, `/v1/responses` and `/v1/messages`):

| `tool_choice` | grammar | output the model writes |
|---|---|---|
| `auto`, `none` | none | free text; calls are parsed if present |
| a named function, or `required`/`any` with one tool | that tool's `parameters` (`tool_args_grammar`) | the arguments object only |
| `required`/`any` with several tools | `required_calls_grammar` (a `CompiledRegex`) | one or more `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks, each an alternation over the tools with that tool's argument schema |

The argument schema is the tool's own `parameters`, compiled by the same
compiler as `response_format` (below): optional parameters stay optional, and
a keyword guided decoding cannot enforce is a 400 naming the tool and the
keyword. `auto` is never constrained, so a model may answer in prose or call
several tools in parallel. Only `required` with several tools admits parallel
forced calls; a forced named function produces exactly one.

## Guided and structured output

Every request spelling maps onto one `nv_grammar::GrammarSpec`:

| endpoint | field | spec |
|---|---|---|
| chat, completions | `response_format: {"type":"json_schema", "json_schema": {"schema": ...}}` | `JsonSchema` (the inner `schema`, else the whole object) |
| chat, completions | `response_format: {"type":"json_object"}` | `JsonSchema({"type":"object"})` |
| chat, completions | `guided_json` (object or JSON string) | `JsonSchema` |
| chat, completions | `guided_regex` | `Regex` |
| chat, completions | `guided_choice` (non-empty) | `Regex(choice_to_regex(..))` |
| responses | `text.format` `json_schema` / `json_object` (`text` is free) | as chat (`responses.rs::requested_text_format`) |
| messages | `output_format` or `output_config.format`, `{"type":"json_schema","schema":...}` | `JsonSchema` (`messages.rs::requested_output_format`) |

The three variants differ in whether the bytes U+0001 and U+0002 are the
any-JSON markers described below:

- `JsonSchema` compiles through `schema_to_regex` and keeps the markers
  (`GuidedDecoder::from_schema`).
- `Regex` is a request regex (`guided_regex`, `guided_choice`): its DFA alone
  decides what is legal (`GuidedDecoder::from_regex`, `json_markers` false). A
  class such as `[^0-9]` that also admits U+0001 never opens free JSON, and a
  U+0001 or U+0002 in a pattern or a choice is that byte.
- `CompiledRegex` is a regex assembled from `schema_to_regex` output
  (`chat.rs::required_calls_grammar`) and keeps the markers
  (`GuidedDecoder::from_compiled_regex`). A grammar that embeds compiled
  schemas must use it, or a tool with no or free-form `parameters` loses its
  free JSON.

The handler compiles the grammar before the engine sees it
(`nv_grammar::check_grammar`), so an invalid regex, an unsupported keyword or a
grammar too large to compile is a 400 naming the field and the reason, not an
engine 500 after admission.

### What a schema enforces

`nv-grammar/src/schema.rs` compiles a schema to an anchored regex, which is
determinized into a DFA capped at
`DFA_SIZE_LIMIT_64_MIB_REFUSES_SCHEMAS_THAT_EXPLODE_WHEN_DETERMINIZED`.

Enforced:

- `type` (a string or an array of types), `nullable`, `enum`, `const`;
- `anyOf` and `oneOf`, as an alternation: a value that matches two `oneOf`
  branches is accepted, so mutual exclusion is the caller's to design;
- local `$ref` (`#/$defs/..`, `#/definitions/..`, any JSON pointer), inlined,
  non-recursive, at most `MAX_REF_DEPTH_16_BOUNDS_NESTED_DEFINITIONS` deep;
  `allOf` with exactly one member;
- objects: `properties`, `required`, `additionalProperties` (`false`, or a
  schema for a free-keyed map when there are no `properties`). Keys are emitted
  in a fixed order: the `required` keys in the order the `required` array
  lists them, then the optional ones in key order, each of which may be
  omitted. With no `required` array every listed property is written;
- arrays: `items`, `prefixItems` (tuples; draft-4 `items: [..]` with
  `additionalItems`), `minItems`, `maxItems`;
- strings: `minLength`/`maxLength` (counted in JSON characters, an escape
  counts once), `pattern`, `format` (`date`, `time`, `date-time`, `uuid`,
  `email`, `ipv4`, `hostname`, `uri`). At most one of the three per string.
  A `pattern` is parsed with `regex-syntax`; its quotes, backslashes and
  control characters are admitted only in their JSON-escaped form, and an
  unanchored pattern searches as JSON Schema says (`^`/`$` around the whole
  pattern anchor it);
- integers and numbers: `minimum`, `maximum`, `exclusiveMinimum`,
  `exclusiveMaximum` (draft-4 boolean or later numeric form), compiled to
  digit-range regexes. A bounded number is written without an exponent.

Refused with a 400 naming the keyword: `multipleOf`, `uniqueItems: true`,
`not`, `if`/`then`/`else`, `patternProperties`, `propertyNames`,
`minProperties`/`maxProperties`, `contains`, `dependentRequired`, recursive
or remote `$ref`, `allOf` with several members, an unknown `format`, a
fractional bound on a `number`, counted repetitions above
`MAX_COUNTED_REPEAT_256_KEEPS_MAXITEMS_AND_MAXLENGTH_DFAS_SMALL`, and any other
keyword the compiler does not know. Annotations (`title`, `description`,
`default`, `examples`, `$schema`, `$id`, `$comment`, `deprecated`, `readOnly`,
`writeOnly`, `x-*`) are ignored, as JSON Schema defines them.

A schema node that leaves its JSON free compiles to a marker byte no JSON text
can contain: an object with no `properties`, an array with no `items`, a node
with no type-specific keyword. The markers are `ANY_JSON_VALUE` and
`ANY_JSON_OBJECT` in `nv-grammar`, and in a `JsonSchema` or `CompiledRegex`
grammar the decoder hands each byte the DFA refuses at them to the exact JSON
pushdown recognizer in `nv-grammar/src/json.rs`. It accepts:

- nesting up to 64 levels, the stack being one `u64`;
- whitespace runs and number digits bounded like the schema regex;
- strings checked for escapes and UTF-8.

A regex cannot count brackets, so this is what lets `json_object` mode, and a
schema like `{"ops": {"type": "array", "items": {"type": "object"}}}`, take
replies of any depth.

### The vocabulary the mask is built from

`chat_engine/sampling.rs::token_bytes_for_guided` gives every token id the
exact bytes it appends to the output. For a byte-level BPE tokenizer (Qwen,
Flash-Next) each vocabulary string is mapped back through the GPT-2 byte
alphabet, so a token holding half of a multi-byte character is its raw bytes
and a leading-space token keeps its space; a lone `decode()` would turn the
half character into U+FFFD and wrongly let the grammar treat it as a whole
character. Added tokens that are not special (`<think>`, `</think>`,
`<tool_call>`) are their text; special tokens (`<|im_start|>`, vision and
audio pads) emit nothing and are never legal inside a grammar; EOS ids are
legal exactly when the armed grammar can terminate, including EOS ids in the
padded `lm_head` tail. `guided_vocab_bytes` caches the table per tokenizer instance
and a fingerprint of its vocabulary, so two models never share a table.

### Enforcement and speculation

`build_guided` compiles the spec into a `GuidedDecoder` (the DFA, the JSON
recognizer riding along in its state, and per-state mask rows cached across
requests, bounded by `ROW_CACHE_4096_MASK_ROWS_THEN_CLEARED_A_ROW_IS_ONE_BIT_PER_VOCAB_TOKEN`).
Grammar state lives in the request's `ChatSampler`, never in a session, so a
resumed or parked prefix-cache session starts every request at the grammar's
start state. Inside `sample()`, a guided decoder forces the copy-and-modify
branch: logits are copied, biases and penalties applied,
`g.apply_mask(&mut lg)` removes every token that would leave the grammar, a
token is sampled, and `g.advance(tok)` walks the DFA. Logprobs are recorded
against the raw logits. Guided decoding keeps a request off the
continuous-batching path, because the batch sampler does not thread
per-request DFA state.

Qwen3.8-Flash-Next (qwen4_exp) keeps its MTP self-speculation under a grammar.
When the request is greedy with no penalties and no logprobs,
`qwen4_exp_loop.rs` passes the sampler to
`Qwen4Exp::session_begin_with`/`session_round_with` as a
`nv_models::spec_pick::SpecPick` (`sampling.rs::SamplerPick`). Each draft is
the masked argmax at the grammar state reached by the drafts before it; each
verify row is checked against the masked argmax at the state reached by the
accepted prefix (`spec_pick::accept_drafts`); the probe advances a checkpoint
and rolls back (`GuidedRun::checkpoint`/`rollback`), and only emitted tokens
commit. The output is therefore the token sequence the non-speculative guided
decode produces from the same logits, which
`guided_spec_tests::grammar_plus_speculation_emits_exactly_the_non_speculative_greedy_tokens`
checks for k = 1..3 over schemas, regexes and a thinking prefix. Sampled
(temperature > 0) or penalised guided requests, and runtime-LoRA adapter
requests, decode without speculation.

### Thinking models

When the official template leaves thinking on, the grammar is deferred until
the model's own close marker (`</think>` for Qwen and Flash-Next;
`guided_think_close_marker`): the reasoning is free text and comes back in
`reasoning_content`, and the constraint starts at the first token after the
marker. Until that marker arms the grammar every EOS id is masked
(`GuidedDecoder::apply_mask` while `deferred()`, reached through
`GuidedRun::apply_mask`), the ones in the padded `lm_head` tail included, so a
thought cannot end the request before the answer starts. `enable_thinking: false` and `reasoning_effort: "none"` render the
closed think block into the prompt, so the grammar binds from the first token.
A thought may run for `max_tokens` minus `GUIDED_THINK_TAIL_RESERVE` tokens
(half of `max_tokens` when it is smaller than the reserve); at that point the
close marker is the only legal token, and the answer gets the rest. A
`max_tokens` too small for the answer returns the truncated answer with
`finish_reason: "length"`, as an unguided request does.

## `/v1/ocr`

Five independent backends sit behind one `nv_ocr::OcrEngine` trait object,
detailed in 06.3-ocr.md: `tesseract`/`classical` (CPU, line-parallel),
`deepseek`, `dots`/`dots.ocr` (a layout model emitting
`{bbox, category, text}`), `got`/`got-ocr2`, and `lighton`/`lightonocr` (a
second layout model, same `{bbox, category, text}` shape). `router_from_env()` attempts
every load and always returns a router; a backend that fails to load is simply
absent from `OcrAppState`, so the route always exists and degrades per backend.

Choosing a backend when the request does not name one is
`pick_default_backend`, which refuses to guess: `NV_OCR_DEFAULT_BACKEND` naming
a loaded backend wins; else exactly one backend loaded wins; else two or more
loaded with none named is `400 ocr_backend_ambiguous` listing the loaded
backends and both ways to disambiguate; else `503 ocr_backend_not_loaded` naming
the env vars. The resolved default is logged once at boot and again per request
with a `default_source` of `request`, `NV_OCR_DEFAULT_BACKEND` or
`only-loaded-backend`. Positional fallthrough is not used for the decision: a
harness that believed it was measuring DeepSeek-OCR-2 could be answered by the
classical path with a 200 and no signal.

Formats are gated on backend capability, checked before a permit is taken or
a model touched: `default_format_for` picks `json-boxes` for tesseract, `text`
or `markdown` for deepseek (by mode), `layout-json` for dots; then
`format=json-boxes` requires tesseract (the generative backends emit no word
boxes), `format=layout-json` and `mode=layout*` require dots, `mode=markdown` is
refused for tesseract, and `resolution` applies to deepseek only.

Each backend is gated by a `SurfaceGate` (`oapi/gate.rs`) — a semaphore plus a
bounded queue window, shedding `503 surface_busy` when saturated for the whole
window — with the generative gate (`NV_OCR_CONCURRENCY`, default 1;
`NV_OCR_QUEUE_MS`) covering the model backends and the classical path **ungated
by default**, opting in via `NV_OCR_CLASSICAL_CONCURRENCY` (06.2 §4b). **The
permit is acquired before `spawn_blocking`, never inside it**: a waiter that
queues inside the blocking closure holds a blocking-pool thread while it waits,
and one hammered endpoint would starve every other surface. The deepseek backend
submits an `OcrJob` to a `DsocrScheduler`, which routes either to the bs=N
batched decode engine whenever the decoder supports the bf16 batch decode graph
(CUDA only), and otherwise to the single-sequence scheduler, which decodes NVFP4
eagerly.

## Realtime sessions

`/v1/realtime` carries both transports on one path: `POST` with
`Content-Type: application/sdp` is the WebRTC offer/answer handshake, `GET` is
the WebSocket upgrade, anything else on the POST is `415`. A `RealtimeQuery`
carries `intent`, `model`, `transcription_model`, `voice` and `speech_model` on
the query string; `intent` defaults to `transcription` and selects whether an
outbound audio lane is created at all. Both transports check the
concurrent-session cap first and refuse with `503 concurrent session cap
exceeded` before building a session.

The ordering contract governing everything a realtime session emits — event
sequencing, cancel semantics, backpressure, and the explicit non-guarantee
about audio/transcript interleaving — is 06.2-concurrency-and-ordering.md. **Its
§2 is the stated precondition for turning batched decode on by default**:
fanning a batched step's tokens out from N tasks into one per-session socket
reintroduces the outbound race it documents.

`oapi/backend_select.rs` computes the machine-readable report of which backend
can serve which model class, with a per-class reason string for every refusal.
It is served at `GET /v1/backends` and folded into
`/v1/realtime/capabilities/backends`, covered by
`rust/tests/backends_report_route.rs`; its verdicts and the three selector/engine
divergences live in the private status ledger.

## PII redaction

Four routes, all requiring `REDACT_MODEL_DIR` to have yielded a loaded
classifier: `POST /v1/pii/classify` (`{"text"}` → `{"spans": [{start,
endExclusive, label}]}`), `POST /v1/pii/classify/batch` (capped at 32 texts),
`POST /v1/pii/redact/analyze` (multipart image → OCR tokens, PII spans, and the
span→pixel-rect mapping), and `POST /v1/pii/redact/render` (multipart image plus
rects and a fill mode → the redacted image bytes). **The split between `analyze`
and `render` is the point**: the caller sees the proposed rectangles and
decides, rather than getting an image back with unreviewable edits already
burned in. Both text handlers run the ONNX classifier inside `spawn_blocking`.

## Layer capture and the SAE pipeline

`POST /v1/capture` (`oapi/capture.rs`, engines in `oapi/chat_engine/capture.rs`)
returns the residual stream after one decoder block: for each text or token-id
input, the per-token rows' `mean`, the `last` row, and with `tokens: true` every
row. The request names `model` and `layer`; the response echoes `model`, `layer`,
`num_layers` and `hidden_size`. `/v1/laguna/capture` is the same handler.

- **A named model must be served exactly.** The capture route never takes the
  `NV_CHAT_ALLOW_UNKNOWN_MODEL` fallback: activations of another model are not
  comparable, so an unknown id is a 404 `model_not_found` listing what is served.
  Without `model` the default engine answers.
- **What is captured is the residual after block `layer`** (attention and FFN or
  MoE output already added), computed by running only blocks `0..=layer` on a
  private cache; serving state is untouched. Per family
  (`residual_rows_after_layer`): Laguna (study capture state), Gemma 4 dense and
  Gemma 4 MoE (bf16 `Gemma4KvCache`, the serving prefill chunk), Qwen 3.5/3.6/3.8
  on the solo eager or graphed engine (the engine's MoE dispatch, if any), and
  Qwen3.8-Flash-Next (`qwen4_exp`), whose residual is `hc_count`
  hyper-connection streams: capture reports their mean, one `hidden_size` row per
  token.
- **Gemma inputs carry BOS for context**, but the BOS row is not reported, so its
  attention-sink norm never dominates a mean.
- **Capture waits for the family's serving slot** (Laguna's exclusive permit, the
  Qwen solo mutex, the qwen4_exp and gemma4 chat permits) for up to
  `CAPTURE_WAITS_120_S_FOR_THE_SERVING_SLOT_BEFORE_A_503_THE_CLIENT_RETRIES`.
- **Everything else is a 501** naming the family and the model:
  `NV_Q38_BATCH=1` (the lane scheduler owns the model), Gemma 4 E4B, Omni,
  GPT-OSS, plain Qwen3 and the wgpu backend. An input over the context is the 400
  `each input must hold 1..=N tokens`, which cccp's scorer splits on.

`GET /v1/capture` lists every chat engine as `{model, family, capture,
num_layers, hidden_size}` or with the `reason` it cannot capture.

`nv-sae` (`rust/crates/nv-sae`) consumes the route. A dictionary belongs to one
(model, layer): `capture` records both in `capture.json`, `merge` refuses to mix
them, `train` writes `model`, `layer` and `hidden_size` into `result.json`, and
`score` refuses states whose width differs from the dictionary. `capture` stops on
the first 404/501 refusal and after
`CAPTURE_GIVES_UP_AFTER_8_CONSECUTIVE_RECORD_FAILURES`, instead of skipping the
rest of the corpus.

## System One decisions

`POST /v1/systemone` (`oapi/systemone.rs` over the `nv-laya` crate) answers
typed questions about a state in one non-autoregressive forward pass. **The
wire contract is TypeSafe's Jev API, field for field**, so a Jev client works
by changing its base URL and nothing else: `Authorization: Bearer`, a body of
`{"state", "model", "questions"}` where each question is a `noul`, a `choice`
or a `score`, and a response of `{"model", "answers", "usage"}`. A `noul`
answer carries `noul` alone, a `choice` carries `choice`, `probabilities` and
`confidence`, a `score` adds `legend` -- nothing more, because Jev's own client
reads a `confidence` field when one is present and would misread Laya's
0.5-to-1 noul confidence as its 0-to-1 one. A request that fails validation is
a 422; weights that will not load are a 503.

The models are the three [Laya](https://laya.convaiinnovations.com/)
checkpoints, laid out under `NV_LAYA_MODEL_DIR` as `laya/`,
`laya-multilingual/` and `laya-typed-decisions/`. `model` selects one by name;
`jev-latest`, `jev-preview`, `auto` or no model at all routes by the script and
language of the state, because the English checkpoint collapses to chance off
Latin script while reporting high confidence. `POST /v1/systemone/detailed`
is the same call with Laya's additions -- noul confidence, the action head's
`act_probability`, the `routing` decision and `timing` -- and both accept
`"preset"` in place of `questions` (`GET /v1/systemone/presets`) and
`{"requests": [...]}` for a list.

Object key order is part of the input. The state is serialized the way
Python's `json.dumps` would and the options of a `choice` are scored by
position, so the crate parses requests into its own order-preserving `Json`
rather than `serde_json::Value`, whose maps sort keys in this workspace.
Enabling `serde_json/preserve_order` instead would reorder every other
response the server emits.

The forward pass has three layouts, picked by the device. A `cuda` build
runs F16 through `flash_attn_varlen_windowed`: the rows of a pass are
concatenated with no padding at all, and ModernBERT's inclusive local window is
the kernel's own window argument. A `metal` build runs F16 on the candle Metal
device and a padded layout whose attention is `candle_nn::ops::sdpa` -- the
steel kernel `mx.fast.scaled_dot_product_attention` dispatches, so the Apple
path of [laya-mlx](https://github.com/mizorewww/laya-mlx) is this crate's
Metal path rather than a second runtime. Everything else is CPU F32 with a
plain matmul-softmax. All three layouts pass `tests/real_checkpoints.rs` on
their own hardware. The rules every layout keeps:

- A padded pass is split into length groups (`A_PADDED_GROUP_ADMITS_ROWS_…`
  in `model.rs`): rows join a group only when they are at least three quarters
  of its longest row, so one long state never pads every short question to its
  length. The varlen layout has no padding and is one group.
- Mask biases are finite. A padded query whose local window holds no valid
  key would softmax an all-`-inf` row to NaN, and the NaN reaches every valid
  row through the next layer's keys. `MASKED_SCORE_BIAS_IS_FINITE_…` underflows
  to the same exact zero for valid rows and keeps the unread rows finite.
- The last decision-head layer computes only what is read. It attends over
  the full sequence, then runs its output projection and FFN on the `[CLS]` and
  option-marker rows alone -- the only rows the scorer and the action head
  consume. This pays on batched and long passes; a single short question is
  launch-bound and does not move.
- Reuse stops at tokens. Question prefixes are cached tokenized and the
  state is tokenized once per request; hidden states are never reused across
  questions, because the first encoder layer is global and bidirectional, so a
  state token's representation depends on the question beside it.
- q and k are roped in one launch. On CUDA a single short question is
  bound by how many kernels it launches, not by their arithmetic: candle's
  per-op host time and the device's own per-kernel floor are of the same
  order, so the lever there is kernels removed.
- A pass whose shape has proven it repeats replays a captured CUDA graph
  (`model/graph.rs`, CUDA only). A varlen pass is
  padded with one unread sequence to the next multiple of
  `A_GRAPHED_PASS_IS_PADDED_…` tokens, run once eager over static buffers,
  captured with `CudaGraphRunner`, and replayed by every later pass of the same
  shape -- token bucket, sequences, option markers and final-head width;
  flash-attn takes the checkpoint's `max_len`, so the longest row is not part
  of the shape. Each capture is held bitwise to that eager pass, a replica
  keeps `LIVE_GRAPHS_A_REPLICA_KEEPS_…` graphs and destroys the least recently
  replayed, and a pass above `A_PASS_THAT_PADS_BEYOND_…`, or any capture
  failure, runs the eager path. A replay removes the host share of a pass and
  leaves the device floor, so it pays most on a single short question and
  fades into the null control as the pass turns compute-bound.
  **A shape is captured only on its `REQUESTS_THAT_PROVE_A_SHAPE_REPEATS_…`th
  request**; until then it runs the unpadded eager pass, bit for bit. A capture
  costs more than two eager passes and a replay returns a fraction of one, so
  the threshold is the number of replays that pay a capture back where replays
  pay at all -- the rent-then-buy rule, which bounds the loss on any request
  sequence to a constant factor of the best choice in hindsight. Requests are
  counted for the `UNPROVEN_SHAPES_A_REPLICA_COUNTS_…` most recently requested
  shapes and the least recently requested is forgotten, so a workload whose
  shapes never repeat captures nothing and runs at the eager pass's speed, and
  a graph the replica destroyed must prove itself again. `GraphStats` counts
  captures, replays, evictions and the passes turned away unproven. The cost
  of the default: padding changes the GEMM shapes, so a replayed pass keeps
  every decision inside upstream's FP16 tolerance but is not bitwise the
  unpadded eager pass, and an answer can move in its fourth decimal when its
  shape crosses the threshold. The first capture turns cudarc event tracking
  off for the CUDA context, as every graph engine in this workspace does.
- The Metal layout hands candle no permuted or last-dim-strided tensor.
  candle's generic strided Metal kernels run an order of magnitude behind
  their contiguous equivalents, while a block copy of adjacent columns is
  cheap -- so the q|k columns are block-copied once and roped as `2*heads`
  heads of one `rope_thd`, `sdpa` receives q, k and v as strided views (the
  steel kernel takes strides), and the GeGLU halves are made contiguous before
  `gelu_erf` and the multiply. CUDA does not copy the halves: there each copy
  is a launch, and launches are what a short pass is made of. What remains of
  a single short question on Metal is the four small-M GEMMs per layer, and
  `CANDLE_METAL_COMPUTE_PER_BUFFER` moves nothing.
- **On Metal the forward pass routes its grid-starved GEMMs through split-k,
  and only its own.** The vendored `candle-metal-kernels` patch carries MLX's
  split-k GEMM (05.7-apple-silicon-inference-architecture.md, the grid-starved
  law): a product with `K >= max(M, N)` and a starved tile grid is cut into K
  slices stacked in grid-z of one dispatch, and the partials are summed by a
  second kernel. Laya's attention and MLP output projections are such products
  on a short pass; the wide input projections are not and stay fused. Summing
  partials reassociates the K sum, so a routed product differs from the fused
  one in its last bits. The switch is therefore a scope, not a process
  setting: `with_split_k_on_this_thread_unless_the_environment_decides` in
  `kernels/mlx_gemm.rs` raises a thread-local count for the duration of a
  closure, candle encodes a GEMM on the thread that calls `matmul`, and
  `DecisionModel::forward` wraps its pass in that scope when the device is
  Metal. Every other candle-Metal family in the server process -- speech,
  alignment, omni, the LoRA trainers whose trajectories are promised bitwise --
  encodes outside the scope and keeps the fused GEMM bit for bit; none of them
  has an instrument that could bless a reassociation, and Laya does.
  `CANDLE_METAL_GEMM_SPLITK` is read once per process and overrides the scope
  in both directions: `0` keeps the fused GEMM even inside Laya's scope, `1`
  routes every family's eligible GEMMs, unset leaves the decision to the scope.
  The cost of the default: a merged pass and its case-by-case passes route
  different products, so they agree to upstream's FP16 tolerance rather than
  bitwise, and the partials live in one per-thread scratch buffer sized to the
  largest routed product. `tests/metal_split_k_scope.rs` holds the scope: 84
  deterministic products are bitwise unmoved outside it, on a thread beside
  it and after it, only routable products move inside it, and those stay in
  the fused GEMM's error class against a CPU F32 product.

Parity is held by `tests/real_checkpoints.rs` (ignored; `NV_LAYA_TEST=1`
and `NV_LAYA_MODEL_DIR`). Its questions are laya-mlx's parity cases, and its
reference is the torch `RLAgent` that ships inside the checkpoint, recorded by
`tests/fixtures/record_upstream_reference.py`: CPU F32 must match it on
decision, token count and four-decimal probabilities, the accelerator must
decide as CPU F32 does inside upstream's FP16 tolerance, and pruning and
pass-merging must keep every decision. On CUDA it also proves every parity
shape, replays the captured graphs, and holds the replays to both the eager
pass and the CPU F32 reference. That suite is the instrument that rules on any
default in this crate that reassociates; the CUDA graph default and the Metal
split-k default both stand on it, green with the default and with each
opt-out on the hardware the default is for. The timing probes live in the same
file and run under `nvk.sh probe`:
`final_head_pruning_and_default_fast_path_paired_ladder` times a full-width pair as its null control, the pruned pass with every fast
path opted out, and the pruned pass as the environment leaves it; on CUDA
`shape_churn_paired_ladder` feeds both a workload whose shapes never repeat
and fails if the default captures once or trails the eager pass.
The checkpoints fine-tune natively on CUDA and Metal and are served merged:
[6.6](06.6-lora-training.md#laya-decision-checkpoints-nvk-laya-lora-train).
`rust/scripts/systemone-live-check.py BASE_URL` holds a running server to the
same reference over HTTP: every parity case against torch, a concurrent burst
that must reproduce the sequential answers (the merging worker), and the
strict Jev shape. Run it after a deploy.

CLM checkpoints share the route. `NV_CLM_MODEL_DIRS` loads native CLM
projection heads and their Qwen3-8B encoder, with cached action projections.
See [Contrastive tool decisions](06.12-clm-tooluse.md) for training, export,
token limits and serving.

Kev checkpoints share the route. [Kev](https://github.com/jaredpalmer/kev)
answers the same wire contract with a pointer head over a Qwen3.5 decoder
(`oapi/systemone_kev.rs`). `NV_KEV_MODEL_DIRS` is a comma-separated list of
checkpoint directories written by `scripts/kev_export.py`; each loads on first
use as its own cold slot, and a request whose `model` equals a directory's
`kev.json` `name` is answered by that checkpoint. Every other `model` goes to
Laya, and with no Laya directory configured it is a 404 that names the Kev
models served.

The exporter merges Kev's LoRA adapter into the base weights (fp32 sum, cast
back to the base dtype, the order `kev.serve` merges in) and writes an ordinary
Qwen3.5 checkpoint beside `kev_head.safetensors` and `kev.json` (the calibration
temperature, the five delimiter token ids and the serving context). It refuses
adapters a merge cannot represent: DoRA, rsLoRA, trained delimiter embeddings,
option isolation. The backbone is `Qwen3Moe::from_loader_dense_quantized`, and
`Qwen3Moe::forward_hidden` returns the final-normed hidden states the head
reads. Every question is its own causal row, the state followed by
`<q> instructions <opt> option </opt>... <decide>` at contiguous positions, and
the head scores each `</opt>` state against `<decide>`'s. Requests render the way
Kev's `api.render` and `option_text` do, Python `str()` of scalars included, and
caller text has `<|name|>` rewritten to `<¦name¦>` before tokenizing so it can
never spell a delimiter. Option order is part of the question: Kev's answers
move when options are reordered, so request JSON is read with the
order-preserving `nv_laya::json::Json`, never `serde_json::Value`, whose maps
sort.

Parity is held by `tests/systemone_kev_parity.rs` (ignored; `NV_KEV_TEST=1`
and `NV_KEV_TEST_DIR`). `kev_export.py --oracle` records Kev's own fp32
predictor on a request file as `kev_oracle.json`, and the suite sends every
request through `/v1/systemone` and requires the same option keys in the same
order, no changed answer outside near-ties, and no probability further from
Kev's than Kev's own published bf16 serving drift. The backbone's residual
stream tracks torch bf16 layer by layer; the requests whose probabilities move
most are the ones that move most under Kev's own bf16 path.

Concurrency is merging, not replication: handlers tokenize on the blocking
pool and enqueue, one worker drains the queue, and every request
that arrived while a pass ran joins the next pass up to 16384 tokens. A pass is
launch-bound when small and compute-bound when large, so a second replica on
the same GPU buys nothing and the server runs one. The
service lives in a cold-tier slot (`systemone-laya`), loads on first use,
preloads the checkpoints named in `NV_LAYA_PRELOAD` (default `english`) and
is evicted when idle like any other resident.

## Admission control and continuous batching

Two independent limiters guard the chat lane, and they are not interchangeable.
The VRAM admission gate (`oapi/admission.rs`) is the primary one:
`init_gemma4` arms a `VramGate` with a per-request `NV_ADMIT_TRANSIENT_GIB` pad
and an `NV_ADMIT_QUEUE_MS` wait window, and the gate recomputes its capacity at
every admission as `NV_VRAM_BUDGET_GIB − max(measured static, ledger)`
(`GateCapacity::bytes`). The measured static is the process VRAM read once,
after the gemma4 batch pool and the mm towers load
(`chat_engine/build.rs`, `our_process_vram_bytes`; `DEFAULT_STATIC_GIB` when it
cannot be read). The ledger is every recorded vram-plan item outside the gemma4
kv-reserve (`vram_plan::recorded_planned_bytes_outside_the_gemma4_kv_reserve`),
so another engine's planned request reserve, and an engine booted or dropped
after gemma4, moves the capacity instead of being spent by gemma4 requests.
`admit(sticky, extra, label)` charges
`sticky − already_retained + extra + pad`, admitting if it fits and otherwise
waiting on a `Notify` until the deadline. Three properties: **a sole request is
always admitted**, even over budget, with a warning — the startup gate is the
real bound, and refusing the only in-flight request would make the server
unusable rather than safe; sticky retention means a spec-decode lease that
allocates a persistent structure calls `set_sticky`, and on drop the gate keeps
a high-water `retained` charge so the next request is charged only the delta;
and RAII release means `VramGuard::drop` returns the charge and wakes
waiters, so a panicking holder cannot leak capacity (tested by
`panic_in_holder_releases_via_raii`). Rejection produces an `anyhow::Error` that
carries `chat::EngineBusy` through its context chain, prefixed
`vram-admission-reject:` with the arithmetic preserved, so any caller surfacing
it synchronously answers `503 engine_busy`; call sites in
`chat_engine/gemma4_loop.rs` are labelled per path (`gemma4-batch`,
`gemma4-nonspec`, `gemma4-spec`, `gemma4-spec-dflash`), each estimating its own
charge from the decode geometry it is about to allocate.

`NV_CHAT_CONCURRENCY` is a separate semaphore, and its logging is explicit that
it is a backstop, not a VRAM proxy: it is sized high, and setting it below
the default makes it — not the admission gate — the binding limit.
`NV_ADMIT_DISABLE` turns the primary gate off and warns that the semaphore is
then the only thing between the server and CUDA OOM.

Continuous batching is the orchestration underneath, designed in
06.1-serving-architecture.md §2 and reached through
`gemma4_loop.rs::run_gemma4_via_engine`. It exists because chat generation used
to serialize on a per-model `tokio::Mutex` held for an entire prompt-to-EOS
loop. A `nv_engine::BatchEngine` owns the model on a dedicated OS thread behind
a `BatchEngineHandle`: the handler tokenizes, prepends BOS if absent, submits
`(prompt_tokens, max_new, eos_ids, SamplingConfig)`, and forwards the returned
`BatchEvent` stream onto the same `ChatEvent` channel through the same
incremental-detokenizer and stop-string helpers the serial loop uses, **so
client-visible behaviour is identical.**

The engine path is opt-in via `NV_BATCH_ENGINE` and deliberately does not claim
every request: Eagle3 spec-decode, guided decoding, `logit_bias` and logprobs
stay on the direct sampling loop, because the batch sampler does not thread
their per-request state. `oapi/batch_chat.rs` holds the `BatchStepper`
implementations — `Gemma4BatchStepper` (serial fallback) and
`Gemma4PagedBatchStepper`, which owns the shared paged fp8 pool and per-sequence
caches and makes exactly one `forward_decode_batched` call per decode step.
Batched decode is not bitwise deterministic with respect to batch composition,
so exact-match tests are gated to the non-batched path. Every other surface's
concurrency model is tabulated in 06.2 §4b, including the two results worth
remembering here: STT deliberately takes no lock at all, and making
`/v1/embeddings` concurrent was built, measured and reverted.

## The cold tier

The generative OCR backends (DeepSeek-OCR-2, dots.ocr, GOT), the text
embedding model and the Qwen3-TTS VoiceDesign talker are the cold tier: configured at boot, loaded by the first
request that needs them, and unloaded after `NV_COLD_TIER_IDLE_SECS` seconds
(default 600; `0` keeps them resident) without a request. `cold_tier::ColdSlot`
owns one resident and the registry in `oapi::cold_tier` is the eviction order,
least recently used first. Laguna with its caches and graphs, and the realtime
speech stack (whisper, kokoro, VAD, diarization) are never in it: the chat engine
is the product and takes minutes to come back, and a websocket session cannot
wait for a reload.

Eviction drops the shared reference. An in-flight request keeps its own `Arc`
and finishes, so the memory returns when the last holder drops;
`ColdSlot::evict` synchronizes the CUDA context afterwards so the freed bytes
are visible to `cuMemGetInfo` before the next decision.

Pressure is checked before the two known-large transients, not after an
out-of-memory error. `ensure_free_async` reads free device memory and evicts
cold residents in LRU order until `NV_COLD_TIER_CHAT_MIN_FREE_GIB` (default 2)
is free before a chat request starts, and `NV_COLD_TIER_FIT_MIN_FREE_GIB`
(default 6) before a voice fit; a cold resident's own load first evicts its
siblings until its ledger estimate fits. The check costs nothing while the tier
is empty. A CUDA out-of-memory inside an ONNX run or a graph replay is not
retried; the shed and its `503` remain the answer there.

`NV_COLD_TIER_PRELOAD=1` loads every cold resident at boot and keeps the idle
unload, for hosts where first-request latency (13 s for dots, 39 s for
DeepSeek-OCR-2 on an RTX PRO 6000) is unacceptable. A cold resident whose load fails
answers `503 ocr_backend_load_failed` (`model_not_loaded` for embeddings,
`voice_design_unavailable` for the VoiceDesign talker, which also refuses to
load unless its booked size fits beside chat's floor) and is retried after
`RETRY_AFTER_FAILED_LOAD`; `/v1/models` advertises a
configured embedding model until its load has failed. The boot ledger charges
the cold tier at its resident estimate because the plan is the everything-loaded
worst case; `nvidia-smi` shows the difference.

## Readiness, and how a missing model degrades one endpoint

`main.rs` distinguishes three states per subsystem: configured (its env var or
file is present), live (it actually loaded), and required.
`build_readiness` records all nine — vad, stt, chat, tts_talker, tts_kokoro,
speaker_encoder, voice_profiles, pii, diarization — after every load has been
attempted, and `readiness_report` marks the process not-ready only for
subsystems that are `required && configured && !live`. **The asymmetry is the
design**: not configured is not a failure (a server started without
`NV_CHAT_MODEL_DIR` is ready, it simply has no chat routes), while configured
but dead is a failure and `/health/ready` returns `503` with the `down` list,
because someone asked for that model and did not get it. `speaker_encoder` is
the one non-required entry, because a CustomVoice or VoiceDesign checkpoint
legitimately has no speaker-encoder tensors. `/health` is a static `ok` and is
not the readiness probe; before the readiness vector is populated
`/health/ready` answers `503 starting`.

That answer is reachable during the load, because the listener binds before
any model is loaded. While loading, the socket serves a warming router: the
four unauthenticated infrastructure routes — `/health`, `/health/ready`,
`/metrics`, `/version`, the same set `auth_mw` exempts — answer for real, and
every other path answers `503` with `Retry-After` and
`code: model_load_in_progress`. Binding after the load instead would leave
nothing listening for as long as the weights take, which a reverse proxy can
only report as a refused connection — `502`, indistinguishable from a crashed
server, and the reason a slow boot reads as an outage. `bind_listener` retries
briefly so the handoff from the warming router to the real one survives the
moment the first socket closes.

Below the process level the same principle repeats per endpoint, and it is why
there are two distinct failure shapes. **A route that cannot exist without its
model is not mounted** — no chat registry means no `/v1/chat/completions` and a
router-level 404, likewise voice profiles and the PII family. **A route that can
exist with a subset of its backends is always mounted and answers 503 per
request** — `/v1/ocr` returns `503 ocr_backend_not_loaded` naming the env var
for the backend the caller asked for (`503 ocr_backend_load_failed` when the
backend is configured but its cold-tier load failed), `/v1/embeddings` returns
`503 model_not_loaded` naming `NV_EMBEDDING_MODEL_DIR`, and `/v1/audio/speech`
returns `503 tts_not_configured` naming both the talker env var and the Kokoro
file layout.

Nothing in the boot path aborts the process for a missing model. The only
hard boot gates are `ORT_DYLIB_PATH` pointing at a non-existent file and
`audio_eou_boot_gate()` when the audio end-of-utterance model is both wanted and
explicitly marked required. Every other load failure is a `warn!` plus a
disabled feature, so one bad checkpoint costs one endpoint rather than the
server.

## CUDA build cache boundary

The `nv-cuda-kernels` Nix package owns `nv-kernels` CUDA compilation, generated
bindings and linker directives. Its source contains native headers, CUDA/C++
sources and the build script, plus the workspace dependency manifests needed
by Cargo. Application Rust sources do not participate in its derivation.
Compiler/toolkit versions, GPU architectures and native source changes do.

CUDA application packages set `NV_KERNELS_PREBUILT` to this immutable output.
The crate build script copies its bindings and emits its linker directives;
it does not run NVCC or bindgen on this path. Local builds without that
variable retain the source compilation path. The package is also available
as `speaches-plus-cuda.cudaKernels` and is shared by CUDA workspace binaries
using the same toolchain and architecture settings. Model and request-handler
edits still rebuild their Rust code, but reuse the native archives.

The server and workspace binaries use one `cargoArtifacts` dependency build
per toolchain and backend, selected with `--workspace`. The CUDA host constructs both
the server and SAE with its host package set and CUDA toolkit; NixOS assertions
keep their Rust dependency and native-kernel derivations identical. Cargo may
still compile feature-specific variants needed by an individual binary.
