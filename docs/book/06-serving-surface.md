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
classifier layout present, and `PiiClassifier::load` succeeded.

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
`stop_reason: "stop_sequence"`, the CUDA loops do not (01.4-STATUS.md).
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
type deserializes to `ContentPart::Unsupported`, which
`reject_unsupported_parts` answers with `400 unsupported_content_part`, stating
outright that the request was rejected rather than answered without the part.
`max_tokens`/`max_completion_tokens` clamp into `1..=MAX_MAX_TOKENS` (8192),
defaulting to `DEFAULT_MAX_TOKENS` (256); `n` caps at 16, `best_of` is accepted
only when it equals `n`, `top_logprobs` caps at 20. Extra request fields with no
OpenAI counterpart: `top_k`, `min_p`, `repetition_penalty`, `guided_json`,
`guided_regex`, `guided_choice`, `chat_template_kwargs`, `enable_thinking`,
`timeout`. Extra response fields: `reasoning_content` on the message and on
streamed deltas, carrying the `<think>`-fenced span split out of the model's own
text, and `system_fingerprint` synthesised from the model id hashed with the
crate version. One extra response header: `x-spec-decode`.

`check_stt_params` rejects `language` (anything but empty or `auto`), any
non-empty `prompt`, and any non-zero `temperature` on
`/v1/audio/transcriptions` — the STT backend auto-detects language, takes no
decoder prompt, and decodes greedily, so accepting those fields would be a lie.
(That refusal breaks two shipped conformance fixtures; see 01.4-STATUS.md.) It
adds a non-OpenAI `response_format=diarized_json`.

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
`voices`). **The list is built strictly from load state** — a row appears only
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
`ChatEngine::generate(req, tx)`. **The trait is deliberately narrow**: the
engine returns `Ok(())` as soon as it has *started* and streams
`ChatEvent::{PromptCached, Started, TextDelta, Logprob, StoppedBy, Done, Error}`
down an `mpsc::Sender<ChatEvent>`, so HTTP never touches a model handle. The
decode loop runs on its own thread with its own model ownership; `Started`
carries `prompt_tokens`, `Done` the finish reason and `completion_tokens`.

`first_event` applies the client deadline only to the *first* `ChatEvent` — a
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
`ChatEngine::supports_mm_input`. **Native towers (Gemma-4 family):**
`extract_mm_media` converts mm parts to `<|image>`/`<|audio>` marker text and
decodes the media at the HTTP layer; the official template renders the marked
messages; `plan_from_marked_tokens` (`chat_multimodal.rs`) expands each marker
into BOI/soft-token/EOI runs; `mm_embeddings` splices tower outputs into the
embedded ids; and `Gemma4E4b::forward_step_embeds` prefills from the spliced
embeddings. Towers load at engine boot (`Gemma4MmTowers::from_model_dir`),
warning and degrading to text-only on failure. **The perception bridge
(everything else):** `bridge_mm_parts` loops back over HTTP to the server's own
`/v1/ocr` (dots, `mode=plain`) and `/v1/audio/transcriptions` (whisper),
injecting the result as `[image, transcribed by ocr]…` text parts — document and
speech perception on a text-only engine with zero extra VRAM. The 400 remains
only for unknown part kinds.

The rules `chat_multimodal.rs` enforces, which any future wiring inherits:
**images must be inline data URLs** (`decode_image_ref` accepts only
`data:<mime>;base64,<payload>`, a remote URL or filesystem path is refused, and
the unit test asserts the error does not echo the rejected path back, so **the
server never fetches on a caller's behalf** — no SSRF surface, no
path-disclosure oracle); **audio must be base64 WAV** (`decode_audio_input`
refuses any other `format`, mixes to mono, resamples to the tower's rate, and
`plan_prompt` clips to 30 s); `run_towers` **checks the produced row count
against the reserved run length** and fails loudly on mismatch rather than
splicing a wrong-sized block; and for a text-only plan the splice is a no-op,
bit-identical to plain embedding — a property the test suite pins.

**The vision path is tied to one model family.** `Gemma4MmTowers` loads
`Gemma4VisionTower` from a checkpoint's `vision_config` and `Gemma4AudioTower`
from its `audio_config`; a checkpoint without one gets `require_vision()` /
`require_audio()` naming the model. There is no generic multimodal abstraction —
another family means another tower pair and another splice contract. Two known
limits are in 01.4-STATUS.md: the w4a16-ct E4B checkpoint is text-only in
practice, and the audio tower has architectural verification only.

## Tool calling

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
only when the value round-trips exactly, **so `007` stays the string `"007"`**;
and a forced call, where `tool_choice` naming a function (or `required` with a
single tool) treats the whole output as that call's arguments and extracts the
outermost balanced JSON object. `rewrite_native_tool_calls` normalises a
model-native `<|tool_call>` encoding into the `<tool_call>` form first, so the
parser sees one dialect. Any parsed call flips `finish_reason` to `tool_calls`
and moves the text into `tool_calls[].function.arguments`, with the remaining
prose as `content`; in the streaming runner this is necessarily buffered
(`TextDelta`s accumulate into a string and the calls are emitted in one late
chunk), so **tool calling and incremental content streaming are mutually
exclusive by construction.**

## Guided and structured output

`nv_grammar`'s public input is `GrammarSpec::{JsonSchema(Value), Regex(String)}`,
and `chat.rs::resolve_guided_fields` is the single funnel mapping every request
spelling onto one spec, in priority order: `response_format:
{"type":"json_schema", ...}` → `JsonSchema` (the inner `schema` if present, else
the whole object); `response_format: {"type":"json_object"}` →
`Regex(json_object_regex(3))`; `guided_json` (object or JSON string) →
`JsonSchema`; `guided_regex` → `Regex`; `guided_choice` →
`Regex(choice_to_regex(..))`, with an empty list a 400. `completions.rs` calls
the same function, so both endpoints share the semantics.

If no grammar was requested but `tool_choice` forces a single named function,
`tool_args_grammar` synthesises one from that tool's `parameters` through
`sanitize_args_schema`: properties are kept, every property is made required,
nested objects and arrays recurse, `enum`/`const` pass through untouched, and
anything else collapses to a plain typed leaf. **A forced tool call is
schema-constrained at the token level, not merely requested in the prompt.**

Enforcement happens in the sampler, not the handler.
`chat_engine/sampling.rs::build_guided` compiles the spec into a `GuidedDecoder`
(an anchored DFA over the vocabulary's byte expansions, cached per tokenizer via
`guided_vocab_bytes`). Inside `sample()`, a guided decoder — like penalties or
`logit_bias` — forces the copy-and-modify branch: logits are copied, biases and
penalties applied, `g.apply_mask(&mut lg)` zeroes every token that would leave
the DFA's language, a token is sampled, and `g.advance(tok)` walks the DFA.
**Logprobs are recorded against the *raw* logits, not the masked ones.** Guided
decoding is one of the request shapes that keeps a request off the
continuous-batching path, because the batch sampler does not thread per-request
DFA state.

## `/v1/ocr`

Four independent backends sit behind one `nv_ocr::OcrEngine` trait object,
detailed in 06.3-ocr.md: `tesseract`/`classical` (CPU, line-parallel),
`deepseek`, `dots`/`dots.ocr` (a *layout* model emitting
`{bbox, category, text}`), and `got`/`got-ocr2`. `router_from_env()` attempts
every load and always returns a router; a backend that fails to load is simply
absent from `OcrAppState`, so the route always exists and degrades per backend.

**Choosing a backend when the request does not name one** is
`pick_default_backend`, which refuses to guess: `NV_OCR_DEFAULT_BACKEND` naming
a loaded backend wins; else exactly one backend loaded wins; else two or more
loaded with none named is `400 ocr_backend_ambiguous` listing the loaded
backends and both ways to disambiguate; else `503 ocr_backend_not_loaded` naming
the env vars. The resolved default is logged once at boot and again per request
with a `default_source` of `request`, `NV_OCR_DEFAULT_BACKEND` or
`only-loaded-backend`. **Positional fallthrough is not used for the decision**: a
harness that believed it was measuring DeepSeek-OCR-2 could be answered by the
classical path with a 200 and no signal.

**Formats are gated on backend capability**, checked before a permit is taken or
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
batched decode engine (CUDA-only, default on, `NV_OCR_BSN=0` opts out) or to the
legacy single-sequence scheduler.

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
sequencing, cancel semantics, backpressure, and the explicit *non*-guarantee
about audio/transcript interleaving — is 06.2-concurrency-and-ordering.md. **Its
§2 is the stated precondition for turning batched decode on by default**:
fanning a batched step's tokens out from N tasks into one per-session socket
reintroduces the outbound race it documents.

`oapi/backend_select.rs` computes the machine-readable report of which backend
can serve which model class, with a per-class reason string for every refusal.
It is served at `GET /v1/backends` and folded into
`/v1/realtime/capabilities/backends`, covered by
`rust/tests/backend_select_e2e.rs`; its verdicts and the three selector/engine
divergences live in 01.4-STATUS.md.

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

## Admission control and continuous batching

Two independent limiters guard the chat lane, and they are not interchangeable.
**The VRAM admission gate** (`oapi/admission.rs`) is the primary one:
`init_gemma4` arms a `VramGate` with `capacity = NV_VRAM_BUDGET_GIB − measured
static weights`, a per-request `NV_ADMIT_TRANSIENT_GIB` pad, and an
`NV_ADMIT_QUEUE_MS` wait window, and `admit(sticky, extra, label)` charges
`sticky − already_retained + extra + pad`, admitting if it fits and otherwise
waiting on a `Notify` until the deadline. Three properties: **a sole request is
always admitted**, even over budget, with a warning — the startup gate is the
real bound, and refusing the only in-flight request would make the server
unusable rather than safe; **sticky retention** means a spec-decode lease that
allocates a persistent structure calls `set_sticky`, and on drop the gate keeps
a high-water `retained` charge so the next request is charged only the delta;
and **RAII release** means `VramGuard::drop` returns the charge and wakes
waiters, so a panicking holder cannot leak capacity (tested by
`panic_in_holder_releases_via_raii`). Rejection produces an `anyhow::Error` that
*carries* `chat::EngineBusy` through its context chain, prefixed
`vram-admission-reject:` with the arithmetic preserved, so any caller surfacing
it synchronously answers `503 engine_busy`; call sites in
`chat_engine/gemma4_loop.rs` are labelled per path (`gemma4-batch`,
`gemma4-nonspec`, `gemma4-spec`, `gemma4-spec-dflash`), each estimating its own
charge from the decode geometry it is about to allocate.

`NV_CHAT_CONCURRENCY` is a separate semaphore, and its logging is explicit that
it is a **backstop, not a VRAM proxy**: it is sized high, and setting it below
the default makes it — not the admission gate — the binding limit.
`NV_ADMIT_DISABLE` turns the primary gate off and warns that the semaphore is
then the only thing between the server and CUDA OOM.

**Continuous batching** is the orchestration underneath, designed in
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

## Readiness, and how a missing model degrades one endpoint

`main.rs` distinguishes three states per subsystem: *configured* (its env var or
file is present), *live* (it actually loaded), and *required*.
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

Below the process level the same principle repeats per endpoint, and it is why
there are two distinct failure shapes. **A route that cannot exist without its
model is not mounted** — no chat registry means no `/v1/chat/completions` and a
router-level 404, likewise voice profiles and the PII family. **A route that can
exist with a subset of its backends is always mounted and answers 503 per
request** — `/v1/ocr` returns `503 ocr_backend_not_loaded` naming the env var
for the backend the caller asked for, `/v1/embeddings` returns
`503 model_not_loaded` naming `NV_EMBEDDING_MODEL_DIR`, and `/v1/audio/speech`
returns `503 tts_not_configured` naming both the talker env var and the Kokoro
file layout.

**Nothing in the boot path aborts the process for a missing model.** The only
hard boot gates are `ORT_DYLIB_PATH` pointing at a non-existent file and
`audio_eou_boot_gate()` when the audio end-of-utterance model is both wanted and
explicitly marked required. Every other load failure is a `warn!` plus a
disabled feature, so one bad checkpoint costs one endpoint rather than the
server.
