# Implementation

The multimedia nano-vLLM. One process, one OpenAI-compatible HTTP surface, multimodal in (text + audio + image + video) -> text + audio out. We consume; we don't generate. (Why `python/` exists at all -- reference implementation for parity, not a deployment target -- is covered in `docs/book/01-architecture.md` § "The parallel trees".)

## Six planes

| Plane | Wrapper | Models | Load timing | Job |
|---|---|---|---|---|
| Qwen3-TTS | `tts.qwen3` (vendored fork) | One `Qwen3-TTS-12Hz-*` per `task_type` | Eager at boot, listed in `QWEN3_TTS_MODELS` | TTS -- `CustomVoice` / `VoiceDesign` / `Base` (voice cloning). |
| Omni | `omni.qwen3` (façade) | `Qwen3-Omni-30B-A3B-Instruct` | Lazy on first chat/ASR request | Multimodal chat; transcription + translation. With talker -> speaks back. |
| Aligner | `aligner` (vendored fork) | `Qwen3-ForcedAligner-0.6B` | Eager at boot | Word-level timestamps for SRT / VTT captions and per-segment text on `diarized_json`. |
| Gemma | `omni.gemma` (façade) | `google/gemma-4-E4B-it` (test) or `gemma-4-31B-it` (prod) | Lazy on first request | Alternative multimodal chat path. No talker. |
| Kokoro | `tts.kokoro` (in-tree rewrite of `kokoro-onnx`) | `onnx-community/Kokoro-82M-v1.0-ONNX` | Eager at boot when `KOKORO_ENABLE=1` | Tiny 82M ONNX TTS, runs on CPU at real-time. 55 voices, 8 languages. |
| Diarizer | `diarization` (in-tree port of `speaches-plus/rust/src/diarization`) | DiariZen-v2 segmentation + WeSpeaker embedding (both ONNX) | Eager at boot when both `DIAR_SEGMENTATION_MODEL_FILE` and `DIAR_EMBEDDING_MODEL_FILE` are set | Per-utterance speaker turn segmentation for `response_format=diarized_json`. |

## Routing

| Endpoint | Default | Switch via |
|---|---|---|
| `POST /v1/audio/speech` | Qwen3-TTS | `task_type="Kokoro"` or `model: "kokoro-..."` -> Kokoro |
| `POST /v1/chat/completions` | Omni | `model: "gemma-..."` -> Gemma |
| `POST /v1/audio/transcriptions` | Omni | `model: "gemma-..."` -> Gemma |
| `POST /v1/audio/translations` | Omni | `model: "gemma-..."` -> Gemma |
| `POST /v1/voice-profiles` | Qwen3-TTS Base | -- |
| `GET /v1/models` | All loaded planes | `?task=text-to-speech\|automatic-speech-recognition\|chat-completion\|forced-alignment` filter |
| `GET /v1/models/{model_id}` | Single entry | 404 with `not_found_error` envelope on miss |

`response_format: srt` / `vtt` on `/v1/audio/transcriptions` calls into the aligner regardless of the chat plane. SRT/VTT with `language=auto` first asks Omni for the language, then aligns.

`/v1/models` returns OpenAI-shaped entries (`id`, `object: "model"`, `created`, `owned_by`, `language`, `task`, plus per-plane extras like `sample_rate` and `voices`). One entry per `(model_id, task)` pair -- Omni and Gemma each appear under both `chat-completion` and `automatic-speech-recognition`. Task strings come from `oapi.task` constants (e.g. `task.CHAT`, `task.ASR`, `task.TTS`) and match speaches-plus's `oapi::task::*`.

## `diarization/` -- speaker turn segmentation

In-tree Python port of `speaches-plus/rust/src/diarization/`. File layout, type names, function names, and constants mirror upstream so a developer can read either tree and know where things live:

| Upstream Rust file | Our Python file | Exports |
|---|---|---|
| `mod.rs` | `__init__.py` + `types.py` | `Diarizer`, `DiarConfig`, `DiarSegment`, `ClusterId` |
| `segmentation.rs` | `segmentation.py` | `SegmentationModel`, `SegmentationLogits`, `SAMPLE_RATE`, `FRAME_RATE_HZ`, `DEFAULT_MAX_SPEAKERS_PER_CHUNK`, `DEFAULT_MAX_SPEAKERS_PER_FRAME` |
| `embedding.rs` | `embedding.py` | `EmbeddingModel`, `cosine_sim`, `EMBEDDING_DIM`, `FRAME_LENGTH_SAMPLES`, `FRAME_SHIFT_SAMPLES`, `NUM_MEL_BINS`, `MIN_INPUT_SAMPLES` |
| `powerset.rs` | `powerset.py` | `PowersetDecoder`, `Multilabel` (DiariZen v2 -> 11 classes; pyannote 3-spk -> 7) |
| `clustering.rs` | `clustering.py` | `OnlineClusterer` (cosine-sim threshold, EMA-smoothed centroids, `max_speakers` cap) |
| `framing.rs` | `framing.py` | `slide_chunks`, `median_filter_multihot`, `extract_spans`, `coalesce_segments`, `Chunk`, `Span`, `ChunkSpans` |
| `fbank.rs` | `fbank.py` | `FBank` (Povey window + Kaldi-style mel + per-utterance CMN) |

Pipeline matches upstream's `Diarizer::diarize_utterance`:

1. `slide_chunks(audio, sample_rate, chunk_seconds, hop_ratio)` -- overlapping `chunk_seconds`-second windows; short utterances zero-padded to one chunk.
2. `SegmentationModel.run(chunk)` -> `SegmentationLogits[frames, classes]`.
3. `PowersetDecoder.to_multilabel_hard(logits)` -> `Multilabel[frames, max_speakers_per_chunk]` via `argmax -> mapping[cls]`.
4. `median_filter_multihot(multihot, window)` -- drops singleton blips per frame, per speaker (window: `DIAR_MEDIAN_FILTER_FRAMES`).
5. `extract_spans(...)` -- runs of `1` per local speaker, dropped if shorter than `min_span_frames` (`DIAR_MIN_SPAN_FRAMES`).
6. For each span: pull `samples[span.sample_start..sample_end]`, skip if shorter than `MIN_INPUT_SAMPLES` (16 000 = 1 s), feed to `EmbeddingModel.embed`.
7. `OnlineClusterer.assign(emb)` -- cosine-sim >= `clustering_threshold` joins, else new cluster; capped at `max_speakers`.
8. `coalesce_segments(segments)` -- merges adjacent same-speaker segments separated by <= 250 ms.

### Env vars

| Var | Default | Purpose |
|---|---|---|
| `DIAR_SEGMENTATION_MODEL_FILE` | unset -> diarizer disabled | Path to the DiariZen v2 segmentation ONNX. |
| `DIAR_EMBEDDING_MODEL_FILE` | unset -> diarizer disabled | Path to the WeSpeaker embedding ONNX (256-d). |
| `DIAR_THRESHOLD` | `0.55` | Cosine similarity threshold for joining an existing cluster. |
| `DIAR_MAX_SPEAKERS` | `16` | Hard cap on cluster count per session. |
| `DIAR_MIN_SPAN_FRAMES` | `8` | Drop spans shorter than this (default = 160 ms at 50 Hz frame rate). |
| `DIAR_MEDIAN_FILTER_FRAMES` | `11` | Window size for the per-frame median smoother. |

Both model files must be set together; if either is missing the diarizer is disabled and `diarized_json` falls back to a single `SPEAKER_00` segment covering the whole utterance. `/health` surfaces `diarizer.{configured, loaded, load_error}` so operators can tell which mode is active.

### Wire shape (parity with upstream `realtime/diarization.rs`)

`/v1/audio/transcriptions` with `response_format=diarized_json` returns `{"text", "avg_logprob": null, "no_speech_prob": null, "segments": [...]}` where each segment is `{"speaker": "SPEAKER_NN", "start", "end", "duration", "text", "avg_logprob": null, "no_speech_prob": null, "confidence"}` (times in seconds, confidence 0..1).

Per-segment `text` comes from running the forced aligner once on the full transcript and bucketing alignment items into the diarizer's spans by midpoint (matches the upstream Rust/Go intersection algorithm). `avg_logprob`/`no_speech_prob` are `null` because Omni and Gemma don't expose token-level probs the way Whisper does. `confidence` is the cosine-similarity score from `OnlineClusterer.assign`.

## `tts/kokoro/` -- Kokoro 82M ONNX TTS

In-tree Python port of `speaches-plus/rust/src/tts/`. File layout, type names, function names, and constants mirror upstream so the diff between the two trees stays readable:

| Upstream Rust file | Our Python file | Exports |
|---|---|---|
| `mod.rs` (`KokoroHandle`) | `model.py` (`KokoroTTS`) | Class with `synthesize`, `stream`, `voices_list`, `has_voice`. ONNX session + voice cache + the per-utterance `split_phoneme_chunks` chunker. Plus `KOKORO_HF_REPO` and `DEFAULT_PRELOAD_VOICES`. Reads `env.KOKORO_MODEL_FILE` and `env.KOKORO_ONNX_PROVIDER`. |
| `text.rs` | `text.py` | All numeric/string defaults (`KOKORO_SAMPLE_RATE`, `MIN_SAMPLE_RATE`, `MAX_SAMPLE_RATE`, `SPEED_MIN`, `SPEED_MAX`, `DEFAULT_SPEED`, `MAX_CHUNK_CHARS`, `DEFAULT_VOICE`, `DEFAULT_LANGUAGE`), `OPENAI_VOICE_ALIASES` + `is_openai_voice_alias()`, `ResponseFormat`/`StreamFormat` enums with `mime_type()`, and the text-side helpers (`strip_emojis`, `strip_markdown_emphasis`, `normalize_for_tts`, `split_sentences`, `split_into_chunks`, `f32_to_s16le`). |
| `vocab.rs` | `vocab.py` | `VOCAB` (the v1.0 char->id table), `MAX_PHONEME_LENGTH`, `PAD_TOKEN_ID`, `tokenize(phonemes)`, `clean_phonemes(phonemes)` (with the same `kəkˈ...` substitutions and `r->ɹ, x->k, ɬ->l, ʲ->j` rune map; the substitution and rune-map tables are hoisted to module-level constants `_PHONEME_REPLACEMENTS` and `_PHONEME_CHAR_MAP`, intentionally diverging from upstream's inline literals to avoid per-call dict reconstruction in Python). |
| `phonemize.rs` | `phonemize.py` | `configure_espeak()` (FFI library + data path resolution; reads `env.PHONEMIZER_ESPEAK_LIBRARY` / `env.ESPEAK_DATA_PATH`) and `phonemize(text, language)`. |

All TTS-text helpers used by `server.py` (`strip_emojis`, `strip_markdown_emphasis`, `normalize_for_tts`, `f32_to_s16le`, `is_openai_voice_alias`) are imported from `tts.kokoro`, not re-implemented; `_clean_speech_input` is a one-liner around the imported pipeline (the old `_split_into_chunks` / `_split_sentences` / `_numpy_to_pcm_bytes` / `_strip_*` private helpers are gone).

`KokoroTTS` mirrors upstream `KokoroHandle`'s public surface (the class name follows our Pythonic convention `Qwen3TTSModel`/`Qwen3OmniWrapper`; upstream's `*Handle` suffix is a Rust idiom for `Arc<Mutex<T>>` ownership that doesn't translate):

```python
KokoroTTS(voices_dir, preload_voices) -> instance
  .synthesize(text, voice, *, speed, lang) -> (np.ndarray, sample_rate)
  .stream(text, voice, *, speed, lang)     -> Iterator[(np.ndarray, sample_rate)]
  .voices_list() -> list[str]               # upstream KokoroHandle::voice_names() (renamed for Python idiom)
  .has_voice(name) -> bool                  # mirrors upstream KokoroHandle::has_voice
  .voices -> dict[str, np.ndarray]          # preloaded voice tensors
```

## `audio/` -- multimodal input

Single source of truth for the bytes-to-tensors path. Both Omni and Gemma `_build_inputs` pull `normalize_parts` + `process_mm_info` from here.

```python
load_audio(spec, target_sr=16000) -> np.ndarray             # mono float32
load_image(spec) -> PIL.Image                               # RGB
load_video(spec, max_frames=32, target_fps=None) -> [PIL]   # pre-sampled frames
read_bytes_or_b64(spec) -> bytes                            # data: / http(s) / file:// / path / bare base64
normalize_parts(conversation) -> conversation               # OpenAI shape -> Qwen shape
process_mm_info(conversation) -> (audios, images, videos)   # parallel lists in conversation order
```

Part shapes accepted on the wire:

| OpenAI shape | Qwen shape |
|---|---|
| `{"type":"input_audio","input_audio":{"data":"<b64>"}}` | `{"type":"audio","audio":""}` |
| `{"type":"audio","audio":"<path|url|data:>"}` | unchanged |
| `{"type":"image_url","image_url":{"url":"<url|data:|path>"}}` | `{"type":"image","image":""}` |
| `{"type":"image","image":"<path|url|data:>"}` | unchanged |
| `{"type":"video_url","video_url":{"url":"<url|data:|path>"}}` | `{"type":"video","video":""}` |
| `{"type":"video","video":"<path|url|data:>"}` | unchanged |

Video sampling: `target_fps` set -> stride `source_fps / target_fps`; else uniform `max_frames` evenly spaced. Bounded memory.

## `nano_vllm/` -- serving engine

Vendored from [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm). vLLM-class continuous batching in ~1.2k lines: paged KV cache, prefix caching with xxhash-keyed blocks, chunked prefill, decode CUDA graphs, tensor parallelism, GPU memory auto-sizing, Triton kernel for KV writes.

Engine runs full-speed on CUDA only. On MPS and CPU flash-attn + Triton are unavailable (CUDA-only); imports are guarded with try/except so the package stays importable, and `nano_vllm.LLM(...)` raises a clear error from `Attention.forward` directing to the `[gpu]` extra. The other planes work everywhere (CPU slow).

### Engine contract for porting a model

A model runs on the engine if it:

1. Subclasses nano_vllm's TP-aware layers -- `QKVParallelLinear` / `MergedColumnParallelLinear` / `RowParallelLinear`, `VocabParallelEmbedding` / `ParallelLMHead`, `RMSNorm`, `Attention`, `get_rope`.
2. Defines `packed_modules_mapping: dict[str, tuple[str, str]]` so the loader knows how to fan out fused safetensors weights into per-shard layers.
3. Provides `num_key_value_heads`, `head_dim`, `num_hidden_layers`, `dtype` on the `hf_config`.
4. Exposes every `Attention` layer's `k_cache` / `v_cache` slot at module-walk order; the runner slices its global KV-cache buffer per-layer by visit order.

`nano_vllm/models/qwen3.py` is the reference.

### Plane -> engine readiness

| Plane | Engine-portable? | Notes |
|---|---|---|
| Qwen3-TTS | Hard | Talker is autoregressive but tightly coupled with the codec/vocoder pipeline. Skip until needed. |
| Omni (chat + ASR) | Yes -- port the thinker (text-decoder portion). Encoders run eager and produce embeddings. | Highest-value port. Enables prefix-caching across multi-turn chat. |
| Aligner | No | Forward-only, no autoregressive generation. |
| Gemma 4 | Yes, same pattern as Omni thinker. | Lower priority. |
| Kokoro | No | ONNX, not torch. ORT handles its own batching. |

## Speculative decode

### EAGLE-3 chain proposal (`nano_vllm/spec_decode/`)

`EagleProposer.propose_tokens` runs a K-step chain, not K independent proposals.

- Step 0 (`fuse_aux=True`): `hidden_states` is the target's auxiliary hidden state of shape `[..., 3 * hidden_size]`, gathered from the target model's three EAGLE-3 hook layers. Goes through the optional `input_norm` and the `fc` projection down to `hidden_size`.
- Steps 1..K-1 (`fuse_aux=False`): `hidden_states` is the draft's own midlayer output from the previous step (already at `hidden_size`). `input_norm`/`fc` are bypassed. Matches EAGLE-3's training pattern where chain-step aux is the draft's residual, not a fresh target aux.

`Eagle3DraftModel.forward` returns `(logits, midlayer_out)` so the proposer can feed the residual back. `propose_token_ids` re-maps draft-vocab argmax through `d2t` to target-vocab IDs.

**Aux row contract** (`EagleProposer.propose`): `runner_state["last_aux_hidden_states"]` has one row per seq in the input batch order -- including finished seqs. Active seqs are filtered by index, not by `zip(active, aux)`. Older code that zipped aux against the active subset crashed when any seq finished mid-step.

**Partial finish**: a seq that just finished still occupies a row in `aux` but has `is_finished=True`. We `clear_drafts()` for finished seqs and only forward active ones into the draft model.

**Partial acceptance row index** (`ModelRunner._slice_last_aux`, `run`): when verify accepts M of K drafts, the next round's "last hidden state" must be sliced from the M-th accepted row, not the last row of the verify batch. `run` builds `per_seq_aux_offsets = [accepted_count for ...]` and threads it through `_slice_last_aux`.

**Chain hoist**: the chain loop appends `next_tokens` and only calls `.tolist()` once after the loop. Collapses K device->host syncs per step into one.

**Pin-memory guard**: `pin_memory=True` on a non-CUDA build (CPU, MPS) does not error -- it hangs silently. Anywhere we allocate a CPU tensor that will be H2D-copied, gate `pin_memory` on `device.type == "cuda"` (or `torch.cuda.is_available()` for module-globals like the bitmask cache).

### N-gram drafter (`nano_vllm/spec_decode/ngram.py`)

Prompt-lookup style: find the longest suffix of the current sequence that appeared earlier and propose the tokens that followed it.

- Why `bytes.rfind` over a Python loop: `struct.pack(f">{total}i", ...)` packs token IDs as 4-byte big-endian ints; `bytes.rfind` then drops to libc `memmem`. Measured ~7x faster than the equivalent Python loop on a 4096-token cold-scan worst case (smoke test verifies live).
- Why big-endian: lexicographic byte order over big-endian-packed ints matches integer comparison. Little-endian would match a different token sequence.
- 31-bit token IDs only (signed int32). Real-world vocabularies fit; if this ever changes, switch to `q` (int64).
- Don't propose into the live suffix: drafts beyond `total - n` would re-predict the unverified suffix. `draft_end = min(draft_start + num_drafts, total - n)`.

## Scheduler 3-mode dispatch (`nano_vllm/engine/scheduler.py`)

Each `Scheduler.schedule()` returns a single mode for the whole batch: `prefill`, `decode`, or `verify`. Mixed batches would require separate attention kernels per row.

- Prefill drains the `waiting` queue. Chunked prefill is allowed only for the first seq in a step (otherwise prefill of seq A would starve decodes of B..N indefinitely).
- Decode vs verify is decided by `num_running_with_drafts > 0`. We keep this as a counter, not an O(N) scan: any code that adds or clears `Sequence.draft_tokens` must call `Scheduler.note_drafts_set(had, has)` (or `preempt`, which decrements directly). Forgetting this leaves the scheduler in the wrong mode forever -- silent correctness bug.

### KV-cache invariant on partial acceptance

Verify writes K+1 slots `[L .. L+K]`. After accepting M drafts plus the bonus, we advance `num_cached_tokens` to `L + M + 1`. Slots `L+M+2 .. L+K` still hold stale KV from the rejected drafts -- but they're never attended to, because the next step's `context_lens` is `L+M+2` (the post-accept length), capping attention before the stale slots. The next decode pass writes slot `L+M+1`, overwriting the first stale row.

Therefore: don't try to "clean up" the stale KV. They're harmless under the context-len invariant; cleanup adds cost without correctness benefit.

### Postprocess type contract

`Scheduler.postprocess(seqs, token_ids, mode)` takes a typed union:

- `mode == VERIFY` -> `token_ids: list[tuple[accepted_count, bonus, accepted_drafts]]`
- otherwise -> `token_ids: list[int]`

Mode disambiguates. Callers (`LLMEngine.step`) hand us the model_runner output that follows the mode contract; `cast()` narrows so each helper sees its precise type. ty cannot infer this -- the cast is load-bearing.

## Grammar / structured output (`nano_vllm/layers/grammar.py`)

xgrammar-backed FSM masks logits to the set of tokens valid under a JSON schema, regex, choice list, or BNF grammar.

### Pinned bitmask workspace

`xgrammar.allocate_token_bitmask(rows, vocab)` returns a fresh unpinned tensor every call. We manage our own pinned CPU buffer (`_pinned_bitmask_cache`) so the H2D copy can overlap compute via `non_blocking=True`.

- Cache key is `(vocab_size, BITMASK_DTYPE)` -- different vocabs get separate buffers; same-vocab calls reuse.
- Power-of-two growth: when a request needs more rows than cached, round up to the next power of two and re-allocate. Amortizes realloc cost.
- Pin only on CUDA: `torch.full(..., pin_memory=True)` on CPU/MPS hangs silently -- gate on `torch.cuda.is_available()`.

### Verify-mode masking semantics

`apply_grammar_mask_verify` is the load-bearing correctness fix. For a seq with K drafts, K+1 rows of logits correspond to draft positions L+0..L+K. Each row needs the matcher's mask AT THAT POSITION:

- Row 0 uses the matcher's current state.
- Row k>0 uses state advanced by drafts `[d_0, ..., d_{k-1}]`.

Implementation:

1. Fill row 0 from current state.
2. For each draft d_k: `accept_token(d_k)`; on success fill row k+1.
3. After the loop: `matcher.rollback(advanced)` to return the matcher to its pre-verify state. The engine commits the actually-accepted prefix later via `accept_tokens()` in `LLMEngine.step`, which is what advances the matcher canonically.

If any draft is rejected by the matcher mid-loop, remaining rows are filled from the rejected state (the model will re-sample at the rejection point anyway, so any mask is fine for those rows).

`xgrammar.GrammarMatcher.rollback` requires xgrammar >= 0.1.0; we hard-fail with `NotImplementedError` if absent rather than silently producing wrong masks.

### Tensor parallelism + grammar = unsupported

`xgrammar.GrammarMatcher` instances are not picklable and cannot be shipped to subprocess ranks via the SHM transport. Matchers stay on the engine (in-process with rank 0). `LLMEngine.add_request` raises `NotImplementedError` if both `tensor_parallel_size > 1` and a grammar are requested. Fix path: teach matchers to pickle, or move masking entirely to rank 0 with a logits gather.

## Model registry & CUDA graphs (`nano_vllm/engine/model_runner.py`, `models/registry.py`)

`resolve_model_class(hf_config)` keys on `hf_config.architectures[0]` to map HF arch names to our model classes. Adding a new model means registering the class here, not editing `model_runner.py`.

`aux_hidden_layer_ids` is passed only when the target class declares it in its `__init__` signature (introspected via `inspect.signature`). EAGLE-3 spec decode requires this hook; turning it on against a model that lacks the hook gives a clean `RuntimeError` at construction.

### CUDA graph aux guard

CUDA graphs can't return tuple outputs. Models with EAGLE-3 aux hooks return `(hidden_states, aux_hidden_states)`; those models always force the eager path, never the captured-graph path. `run_model` flips `is_prefill = True` when `aux_hidden_layer_ids is not None` to route through the eager branch.

Tier breakpoints (`DECODE_GRAPH_TIER_BREAKPOINTS`) define which batch sizes get captured: dense at small sizes (1-8), sparser higher up. The next-tier graph is replayed and sliced to actual batch size. Capture cost is paid once at startup; replay is microseconds.

## Per-step Context (`nano_vllm/utils/context.py`)

`set_context()` overwrites the module-global `_CONTEXT` with a new `@dataclass(slots=True)` instance -- a side channel attention kernels read via `get_context()` to avoid plumbing through every layer.

Reset after every step (`reset_context()` at the bottom of `ModelRunner.run`). Forgetting this leaves stale tensors visible to the next step's attention kernel -- silent correctness bug under tp>1 or back-to-back prefill/decode transitions.

`verify_mode: bool` is the only flag distinguishing verify from prefill in the kernel path (since both use the prefill kernel with `cu_seqlens_q` metadata).

## Server internals

### SSRF policy

Two distinct boundaries fetch user-supplied URIs; both apply the same allowlist.

`server._decode_ref_audio` -- `ref_audio` on `/v1/audio/speech` and `/v1/voice-profiles`:

- `data:audio/...;base64,<payload>` -- strict base64 validation, capped at `REF_AUDIO_MAX_BYTES` (32 MiB).
- `https://...` -- fetched with `urllib.request.urlopen`, timeout 10 s, capped at `REF_AUDIO_MAX_BYTES` (read at most cap+1 bytes, fail if exceeded).

`audio.loaders.read_bytes_or_b64` -- multimodal parts on `/v1/chat/completions`, `/v1/audio/transcriptions`, `/v1/audio/translations`. Same allowlist:

- `data:<mime>;base64,<payload>` -- strict base64, capped at `MULTIMODAL_MAX_BYTES` (64 MiB). `;base64,` required; raw text payloads rejected.
- `https://...` -- `urlopen` with `MULTIMODAL_FETCH_TIMEOUT_SECONDS` (10 s), capped at `MULTIMODAL_MAX_BYTES`.
- Bare base64 -- accepted only when >= `BARE_BASE64_MIN_LENGTH` (256) chars and every char is in the strict base64 alphabet (`A-Za-z0-9+/=`); decoded with `validate=True`, capped at `MULTIMODAL_MAX_BYTES`. The length floor + alphabet test rules out absolute paths and most filesystem strings; the field-extractor changes below close the rest.

Both boundaries reject `file://`, `http://`, `ftp://`/`ftps://`, `gopher://`, leading `/` (absolute paths), and `..` (traversal). Empty strings and NUL-byte injections are rejected. Don't add schemes without thinking about: outbound network reach, file disclosure, internal service probing.

`_audio_spec_from_part` / `_image_spec_from_part` / `_video_spec_from_part` extract `data` / `url` / `audio` / `image` / `video` only -- the legacy `path` field is no longer honored on the wire (it was a backdoor for arbitrary local-file reads).

The `HOST != 127.0.0.1` startup warning in `lifespan` reminds operators that all of the above is unauthenticated -- bind to localhost or front with an auth proxy.

### Tool-call parsing (`server._parse_tool_calls`)

Four input formats are recognized, in priority order:

1. `<tool_call>{json}</tool_call>` -- Qwen3 native.
2. ` ```tool_code\n<python>\n``` ` -- Gemma's format. Walked with `ast` for `ast.Call` nodes whose name matches a registered tool. Args via `ast.literal_eval` (no eval, no execution).
3. ` ```json\n{...}\n``` ` -- Markdown-fenced JSON.
4. Raw `{...}` JSON -- last resort, scanned with `json.JSONDecoder.raw_decode`.

Earlier matches mark spans so later parsers don't double-count. All four funnel through `_coerce_tool_call_dict` which validates the name is a known tool and that arguments serialize to a JSON string (the OpenAI tool-call shape demands `arguments: str`, not `arguments: object`).

`_strip_tool_calls_from_text` re-scans with the same regexes/AST walker to remove matched spans from the assistant content; cleaned text goes into `message["content"]` while structured calls go into `tool_calls`.

`tool_choice="required"` raises 422 if no parseable call comes back. Proper fix is grammar-enforced tool calling (Phase 5B); operators must lean on a stronger system prompt or retry.

### ChatResponseFormat OpenAI envelope (breaking change)

`ChatResponseFormat.json_schema` takes a `ChatJsonSchemaSpec`, not a bare dict -- the OpenAI shape `{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}`. The previous shape (bare dict as `json_schema`) is rejected by pydantic with 422. Deliberate break to align with OpenAI clients; smoke test asserts both behaviors.

`_guided_json_from_request` extracts `spec.schema` and forwards to the chat wrapper as `guided_json`. `json_object` returns `{}` (permissive); `text` or no `response_format` returns `None`.

### Speaches-plus wire compatibility

Contract details that land at the wire and aren't visible from the route handlers alone:

- OpenAI voice aliases (`OPENAI_VOICE_ALIASES` + `is_openai_voice_alias()` in `tts.kokoro.text`: alloy, ash, ballad, coral, echo, sage, shimmer, verse) on `/v1/audio/speech` route to Kokoro and resolve to its default voice. Kept for parity with speaches-plus, which advertises the same set.
- Error envelope unwrapping (`_http_exception_handler`). Audio endpoints raise `HTTPException` with an OpenAI-shaped detail (`{"error": {message, type, param, code}}`); the handler emits that body unwrapped so speaches-plus / OpenAI clients can parse it directly. All other detail shapes (plain string, dict without `"error"`) keep FastAPI's default `{"detail": ...}` envelope.
- `stream_format` vs `stream` (`_resolve_stream_format`). speaches-plus uses `stream_format: "sse" | "audio"`; nano historically used `stream: bool` paired with `response_format: "pcm"`. Both shapes are accepted; `stream_format` wins when set.
- `diarized_json` on `/v1/audio/transcriptions`: real per-speaker segments via `_diarized_segments_for(...)` when `_diarizer` is loaded; otherwise the single-segment fallback (`SPEAKER_00`, `start=0.0`, `end=0.0`) -- keeps the contract honest without faking diarization. Shape in § "Wire shape" above.

### Naming parity with `speaches-plus/rust/src/tts/text.rs`

Numeric/string defaults for the speech endpoint live next to the kokoro implementation (`tts.kokoro`) and are re-exported:

- `SPEED_MIN` / `SPEED_MAX` (was `SPEECH_SPEED_MIN/MAX`)
- `MIN_SAMPLE_RATE` / `MAX_SAMPLE_RATE` (was `SPEECH_SAMPLE_RATE_MIN/MAX`)
- `MAX_CHUNK_CHARS` (was `SPEECH_MAX_CHUNK_CHARS`)
- `KOKORO_LANGUAGES = ("en", "es", "fr", "hi", "it", "ja", "pt", "zh")` -- Kokoro v1.0's actual 8-language coverage. (Upstream's earlier 5-entry list with `ko` was a bug since fixed in both directions; Kokoro v1.0 has no Korean voice.)
- `KOKORO_VOICE_PREFIX_TO_LANG` -- single source of truth for `voice_id[0] -> ISO language code` (a/b->en, e->es, f->fr, h->hi, i->it, j->ja, p->pt, z->zh).

### Event-shape parity with `speaches-plus/rust/src/oapi` and `tts/http.rs`

Wire shapes for streaming events, validation entries, and the diarization payload match upstream verbatim; every event we emit parses with the same client code that parses speaches-plus.

- TTS SSE on `/v1/audio/speech` (`_sse_speech_events`): per-chunk `{"type": "speech.audio.delta", "audio": "<base64 s16 PCM>"}`; terminal `{"type": "speech.audio.done", "token_usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}`; framed as `data: <json>\n\n` per the WHATWG `text/event-stream` spec.
- OpenAI error envelope (`_openai_error_payload`): `{"error": {"message", "type", "param", "code"}}` -- identical key order and semantics to upstream `oapi::openai_error`. Audio-endpoint `HTTPException`s raise this shape; `_http_exception_handler` unwraps it from FastAPI's `{"detail": ...}` envelope.
- FastAPI validation entries (`_missing_field`, inline `less_than_equal`): `{"type": "missing"|"less_than_equal", "loc": ["body", "<field>"], "msg": "<text>", "input"?: <value>}`. Emitted as 422 with body `{"detail": [<entry>, ...]}` matching `oapi::fastapi_validation_error`. `_missing_field(loc)` mirrors `oapi::missing_field` exactly.
- Diarized transcription: the per-segment shape (§ "Wire shape" above) matches upstream `realtime/diarization.rs`'s `conversation.item.diarization` event; we wrap under `{"text": <full>, "segments": [...]}` since the REST endpoint also returns the transcript, and per-segment `id` / `type` fields were dropped to match upstream.
- Verbose JSON transcription (`response_format=verbose_json`): OpenAI Whisper-shape `{"task": "transcribe", "language", "duration", "text", "segments": [], "words": []}`. We don't compute per-segment timestamps yet (no Whisper-style alignment in the chat planes), so `segments`/`words` ship empty; `duration` reads from soundfile and falls back to `null`.

### `oapi/` -- OpenAI compatibility module

Python port of `speaches-plus/rust/src/oapi/` and `speaches-plus/go/internal/oapi/`. Same file layout, same names, same wire shapes:

| Upstream | Our `oapi/` | Exports |
|---|---|---|
| `mod.rs::kind` mod / Go `TypeInvalidRequest` etc. | `kind.py` | `INVALID_REQUEST`, `AUTH`, `NOT_FOUND`, `SERVER`, `SERVICE_UNAVAIL` |
| `mod.rs::task` mod / Go `TaskASR` etc. | `task.py` | `ASR`, `TTS`, `VAD` (upstream) + `CHAT`, `FORCED_ALIGNMENT` (our extensions) |
| `mod.rs::openai_error` / Go `WriteError` | `errors.py` | `openai_error(message, type, param, code)` returns the dict; `raise_openai_error(status, ...)` raises `HTTPException` (FastAPI-friendly variant) |
| `mod.rs::missing_field` / Go `FastAPIErrorEntry{Type:"missing"}` | `errors.py` | `missing_field(loc)` -> `{"type": "missing", "loc": loc, "msg": "Field required"}` |
| `mod.rs::fastapi_validation_error` / Go `WriteValidationError` | `errors.py` | `fastapi_validation_error(entries)` (returns `HTTPException`); `raise_fastapi_validation_error(...)` |
| `mod.rs::Model` / Go `Model` | `models.py` | `@dataclass Model(id, owned_by, task, created=1, languages=None, extras={})` with `to_dict()` matching upstream's `MarshalJSON` field order |
| `mod.rs::ListModelsResponse` / Go `ListModelsResponse` | `models.py` | `@dataclass ListModelsResponse(data, object="list")` with `to_dict()` |
| `mod.rs::WHISPER_LANGUAGES` / Go `WhisperLanguages` | `models.py` | 99-entry tuple, identical to upstream |
| `mod.rs::KOKORO_LANGUAGES` / Go `KokoroLanguages` | re-exported from `tts.kokoro.text` | The 8-language Kokoro v1.0 set |
| `models_handler.rs::handle_list_models` / Go `NewModelsHandler` | `models_handler.py` | `hf_owner(model_id)`, `filter_by_task(models, task)`, `list_models_response(models, task)` |

`server.py` reads from these as `oapi.kind.X`, `oapi.task.X`, `oapi.raise_openai_error(...)`, `oapi.missing_field(...)`, `oapi.Model(...)`, `oapi.list_models_response(...)`. `_build_models()` in `server.py` is the per-deployment listing builder (we have planes upstream doesn't -- Omni/Gemma chat+ASR, Aligner -- so the builder is project-specific).

The two task extensions (`CHAT`, `FORCED_ALIGNMENT`) cover planes speaches-plus doesn't have. Upstream's `VAD` is exposed for parity even though we don't ship a VAD model.

### Lifecycle & batching

- TTS models load eagerly in `lifespan` (`_load_tts_models`). Profiled cold start: ~3.5 s on Mac.
- Omni and Gemma load lazily on first request, lock-protected (`_omni_lock`, `_gemma_lock`). Failures cache an error string; subsequent requests get 503 with the original message.
- Aligner and Kokoro load eagerly if their env vars are set.
- `_models_by_task`: each task type (CustomVoice, VoiceDesign, Base, Kokoro) gets the first loaded model whose ID resolves to that task via `_resolve_task_type`. To override priority, set `QWEN3_TTS_MODELS` to a comma-separated list ordered per task.
- Voice profiles are speaker prompts precomputed from a `ref_audio`, cached under a name and bound to a specific base model. If that model isn't loaded, the profile refuses (409). Lock-protected for create/delete and for read.
- `_Batcher`: when `BATCH_WINDOW_MS > 0`, requests for the same `(model_id, task)` are coalesced for that many ms, then run as a single batched generate. Batch key is `(id(model), task)`. Used only for non-streaming CustomVoice / VoiceDesign.

## Type checking -- what's strict, what's tolerated

`pyproject.toml` configures ruff (`E,W,F,I,B,UP,SIM,TCH,PT,RUF,PERF,S`) and ty (Astral) as the type checker.

### Boundary asserts pattern

We don't try to over-type the HuggingFace and torch surface. Instead, assert at boundaries and let downstream code work with narrowed types:

- `assert tokenizer is not None` after `AutoTokenizer.from_pretrained(...)` (return type is Optional in HF stubs even though None means a hard load failure).
- `assert config.hf_config is not None` whenever code accesses HF config attrs (`Config.__post_init__` always sets it; we just need the narrowing).
- `assert isinstance(decoded, str)` after `tokenizer.decode(...)` (the stub returns a union).
- `assert isinstance(self.event, list)` in `ModelRunner.write_shm` (the field is `Event | list[Event]` depending on rank -- only rank 0 ever calls write_shm and there it's a list).
- `assert self.shm.buf is not None` (`SharedMemory.buf: memoryview | None` in stubs but always set after `SharedMemory(create=True)`).

### Inline directive policy

No inline `# noqa`, `# ty: ignore`, `# type: ignore`, `# pyright: ignore`, `# coding=...`, or `# fmt: ...` directives in source. Suppressions live in `pyproject.toml` under `[tool.ruff.lint.per-file-ignores]` and `[tool.ty.rules]`. Known ecosystem-typing gaps demoted there:

- `invalid-assignment` -> `warn` -- pydantic `Field(default_factory=...)` and `xgrammar = None` after ImportError.
- `unresolved-attribute` -> `warn` -- dynamic HF `AutoConfig` attrs (`vocab_size`, `dtype`, `hidden_size`).
- `possibly-unbound-attribute` -> `warn` -- `multiprocessing.Event | list[Event]` union on `ModelRunner.event`.
- `no-matching-overload` -> `warn` -- `@dataclass(slots=True)` and torch in-place op stub gaps.
- `too-many-positional-arguments` -> `warn` -- ty's stricter slots-dataclass checking vs. positional `Context(...)` construction.
- `call-non-callable`, `not-subscriptable`, `invalid-argument-type` -> `warn` -- HF `AutoTokenizer` / `AutoConfig` Optional-union stubs; we guard with `assert` at boundaries.

`server.py` carries `S110` + `S603` per-file (best-effort cleanup paths and ffmpeg subprocess invocations). `test_review_fixes.py` carries `S, E402, F401`.

### Buffer access on `nn.Module`

`nn.Module.__getattr__` returns `Tensor | Module` for registered buffers (because subclassing modules can replace the buffer with a child module). For tensor-only buffers (`inv_freq`, `d2t`, `t2d`), use `typing.cast(torch.Tensor, self.<buffer>)` at the access site.

### Torch overload gaps

`torch.Tensor.narrow`, `torch.Tensor.scatter_`, `torch.Tensor.zero_` and similar in-place ops occasionally trip ty stub overloads. Warnings, not errors; the runtime is correct.

### Current state

- ty: 0 errors, ~46 warnings (all in the categories above).
- ruff: ~57 cosmetic items (mostly `UP007` Optional remnants in vendored tts/qwen3 paths, `S105` false positives on env-var name strings).

If you bump xgrammar / transformers / torch and warning counts shift, that's expected. Goal is "no errors", not "no warnings".

## Smoke test (`test_review_fixes.py`)

Round-1 smoke test. Exercises: SSRF rejection of every blocked scheme; registry resolution and Proposer ABC; ChatJsonSchemaSpec envelope (new shape parses, old shape rejected); EAGLE partial-finish without crash, K=5 chain hoist, K=1 regression; `Context.verify_indices` removal (was never read after the shape change); n-gram `bytes.rfind` correctness against a Python-loop oracle on 50 random inputs plus the live 7x speedup measurement; scheduler `note_drafts_set` counter increments/decrements; pinned bitmask cache reuse (1000 same-shape calls -> 0 reallocs); end-to-end xgrammar mask via the cache; all 4 tool-call parser formats + strip.

Run with `nix develop --command python3 test_review_fixes.py`. Output ends with `=== ALL SMOKE TESTS PASSED ===` on success.

## Known limitations

- No CUDA on the dev machine. Smoke tests exercise CPU code paths (proposers, parsers, masking, registry, SSRF). Real KV cache, CUDA-graph capture, flashinfer wrappers, and actual `model.forward` only run on GPU. The flamegraph at `review_fixes_flamegraph.html` captured imports, not steady-state decode -- useful for spotting import bloat, useless for spotting decode hotspots.
- `nix-builder` remote CUDA build host not reachable from this machine. Building the CUDA closure requires either local CUDA or a working remote builder.
- Streaming chat completions not implemented (`/v1/chat/completions` with `stream=true` returns 501).
- Audio-out chat works only on Qwen3-Omni; Gemma path returns 400 for `modalities=["audio"]`.
- Guided decoding via the Omni chat wrapper raises 501. Use the Gemma path or wait for Phase 5B-2.
- xgrammar matchers + tp>1 -> NotImplementedError (matchers don't pickle).
- `tool_choice="required"` falls back to 422 if the model doesn't emit parseable calls. Proper fix is grammar-enforced tool calling.

## Quick reference: where to add things

| Need | File |
|---|---|
| New target model | `nano_vllm/models/<name>.py` + `registry.py` |
| New proposer | `nano_vllm/spec_decode/<name>.py` implementing `Proposer` |
| New attention backend | `nano_vllm/layers/attention.py` |
| New quant scheme | `nano_vllm/layers/quantization/` |
| New endpoint | `server.py` (FastAPI route) |
| New chat tool format | `server._parse_tool_calls` + `_strip_tool_calls_from_text` |
| New ref_audio scheme | `server._decode_ref_audio` (think hard about SSRF) |
| New env var | `env.py` (single source of truth -- name = value, mirrors upstream `defaults::env`) + the table in §"Environment variables" |
| New ty rule tweak | `pyproject.toml` `[tool.ty.rules]` |

## Bundled HuggingFace assets

The flake bundles every HF repo we ship via [`nix-hug.fetchModel`](../nix-hug), pinned by commit hash + filetree sha. Runtime never hits the network.

```bash
nix run path:../nix-hug -- fetch <org>/<repo>
```

Paste the printed expression into `mkPackages` in `flake.nix` and have the wrapper export the corresponding env vars.

All models referenced by the source (chat, ASR, TTS, aligner, Gemma, Kokoro) and by `speaches-plus/` (Whisper, smart-turn, DiariZen, WeSpeaker, Silero) are declared in `nix/models.nix` and assembled into a single Hugging-Face hub cache via `nix-hug-lib.buildCache`. The wrapper and `devShell` export `HF_HUB_CACHE`, `HF_HUB_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1`, so every `hf_hub_download` / `snapshot_download` / `AutoX.from_pretrained` resolves locally with no network. The `speaches-plus/{rust,go}/models/` directories are populated with symlinks during the devShell's `shellHook`. Run `bash scripts/fetch-models.sh` once after clone to pin every `rev` + `fileTreeHash` in `nix/models.nix`.

## Environment variables

All env-var names are centralized in `env.py` (mirrors `speaches-plus/rust/src/defaults.rs::env`). Constants follow the upstream pattern `name = value` (the constant name IS the env var name), so `os.environ.get(env.QWEN3_TTS_MODELS, ...)` is the canonical lookup; consumers `import env` (module-local `ENV_*` constants no longer exist). Full operator-facing surface (25 names + the 4 HF-cache vars set by the flake):

| Var | Default | Plane | Purpose |
|---|---|---|---|
| `QWEN3_TTS_MODELS` (or `QWEN3_TTS_MODEL`) | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | TTS | Comma-separated HF ids to preload, one per task type. |
| `QWEN3_TTS_DEVICE` | `auto` | all torch | `auto` / `cuda:0` / `mps` / `cpu`. |
| `QWEN3_TTS_DTYPE` | `bfloat16` | all torch | `bfloat16` / `float16` / `float32`. |
| `QWEN3_TTS_HOST` | `127.0.0.1` | server | Bind address. |
| `QWEN3_TTS_PORT` | `8091` | server | Bind port. |
| `QWEN3_TTS_BATCH_WINDOW_MS` | `0` (off) | TTS | Coalesce concurrent same-task requests. |
| `QWEN3_OMNI_MODEL` | `""` (disabled) | Omni | HF id; empty disables chat + transcription. |
| `QWEN3_OMNI_DISABLE_TALKER` | `0` | Omni | `1` to skip ~10 GB of audio-decoder weights. |
| `QWEN3_ALIGNER_MODEL` | `Qwen/Qwen3-ForcedAligner-0.6B` | aligner | Empty disables SRT/VTT (those formats then 503). |
| `GEMMA_MODEL` | `""` (disabled) | Gemma | HF id (e.g. `google/gemma-4-E4B-it`). |
| `GEMMA_ATTN_IMPL` | auto-pick | Gemma | Override: `sdpa` / `flash_attention_2` / `eager`. |
| `GEMMA_COMPILE` | `0` | Gemma | `1` wraps the model in `torch.compile`. Skipped on MPS. |
| `KOKORO_ENABLE` | `0` | Kokoro | `1` to load Kokoro at boot. |
| `KOKORO_MODEL_FILE` | (HF cache) | Kokoro | Optional override path to the ONNX model file. Default: `hf_hub_download(...)` resolves from `HF_HUB_CACHE`. |
| `KOKORO_VOICES_DIR` | (set by flake) | Kokoro | Directory of per-voice `.bin` files; enables full 55-voice offline listing. Without it, `voices_list()` returns only preloaded voices. |
| `HF_HUB_CACHE` | (set by flake) | all HF | Hub cache directory (assembled by `nix-hug-lib.buildCache`). |
| `HF_HUB_OFFLINE` | `1` (set by flake) | all HF | Block every network call; resolve from cache only. |
| `TRANSFORMERS_OFFLINE` | `1` (set by flake) | all HF | Same, for `transformers.AutoX.from_pretrained`. |
| `KOKORO_ONNX_PROVIDER` | auto-pick (CoreML/CUDA/CPU) | Kokoro | Comma-separated EP list. |
| `PHONEMIZER_ESPEAK_LIBRARY` | (set by flake) | Kokoro | Path to libespeak-ng. |
| `ESPEAK_DATA_PATH` | (set by flake) | Kokoro | Path to espeak-ng-data. |

## Build

`pyproject.toml` is the single source of truth for deps. `flake.nix` builds the wheel via `buildPythonPackage` and ships:

- `speaches-plus-python` (CPU wrapper) -- default, has espeak-ng + Kokoro assets wired in via env vars.
- `speaches-plus-python-cuda` -- same, with CUDA-built torch.
- `speaches-plus-python-pkg` (and `-cuda`) -- bare Python package output.
- `kokoro-assets` -- the bundled HF model directory.

`nix develop` provides the same env wiring plus `ffmpeg` / `sox` / `curl` / `file` for the e2e test.

The `[gpu]` optional pip extra adds `flash-attn` + `triton` -- required by the nano_vllm engine, only useful on a CUDA box.

### Tested-with (last verified by e2e)

| Dep | Version |
|---|---|
| python | 3.13.12 |
| torch | 2.11.0 |
| transformers | 5.5.4 |
| accelerate | 1.12.0 |
| numpy | 2.4.2 |
| huggingface_hub | 1.10.2 |
| fastapi | 0.128.0 |
| uvicorn | 0.40.0 |
| pydantic | 2.12.5 |
| librosa | 0.11.0 |
| soundfile | 0.13.1 |
| pillow | 12.1.1 |
| torchvision | 0.26.0 |
| onnxruntime | 1.24.4 |
| phonemizer | 3.3.0 |
| xxhash | 3.6.0 |
| imageio | 2.37.2 |
| av | 16.1.0 |
| python-multipart | 0.0.21 |

Refresh when you bump `flake.lock` and re-run the e2e.

## Top-level utility ports (`ids.py` / `errors.py` / `otel.py` / `trace.py`)

Faithful Python ports of `speaches-plus/rust/src/{ids,errors,otel,trace}.rs`.

- `ids.py` -- `RandomIdSource` (uuid4-hex) + `CounterIdSource` (24-digit zero-padded, atomic via `threading.Lock`) implementing the same four-prefix surface (`sess_`, `item_`, `resp_`, `evt_`). Module-level `next_*_id()` helpers wrap a default `RandomIdSource`.
- `errors.py` -- Canonical RFC v3 §10.5 error-code registry. Keeps the upstream Rust split between codes that map to `invalid_request_error` vs `server_error`. `realtime/errors.py` already mirrored a subset; we keep both rather than break the existing `realtime.errors` import sites in `test_review_fixes.py` and `realtime/`. The top-level `errors.py` is the superset (adds `SESSION_NOT_ACTIVE`, `MODEL_LOAD_FAILED`) and also exposes an `envelope()` helper for the OpenAI `{error: {message, type, param, code}}` JSON shape -- `oapi/errors.py` keeps the `HTTPException`-bound flavor used by FastAPI handlers.
- `otel.py` -- `init()` is a no-op when `OTEL_EXPORTER_OTLP_ENDPOINT` is unset or when the `opentelemetry-sdk`/`opentelemetry-exporter-otlp` packages are missing (try/except ImportError). `shutdown()` is idempotent. Wired into `server.py` lifespan.
- `trace.py` -- Faithful port of `canonicalize_trace` / `trace_diff` (used for trace-replay equality in tests: redacts timestamps to 0, rounds floats to 3 decimals, renumbers IDs). Also adds `init()` / `span(name, **attrs)` context manager / `@traced(name)` decorator built on top of `otel.py`, all guarded by `is_enabled()` so they degrade to no-ops without opentelemetry installed.
- `types.py` not created -- the upstream `types.rs` newtypes (`SessionId` / `ItemId` / `ResponseId` / `EventId` / `MonoF32At*` / `StereoS16At48k` / `Millis` / `DurationMs` / `Epoch`) are not used elsewhere in the Python tree; Python uses raw `str` for IDs and `np.ndarray` / `torch.Tensor` for audio. Re-introduce only when a call site needs the type-safety.

## Native bindings (`ct2_bindings/`, `whisper_bindings/`)

Two in-tree pybind11 extensions bridge `libctranslate2` (CTranslate2 Whisper) and `libwhisper` (whisper.cpp). Intentionally separate buildables -- the main `speaches-plus-python` package installs cleanly without them, and `stt/whisper.py` falls back to PyPI wheels when the native module isn't importable. Build details and rationale: `ct2_bindings/README.md`, `whisper_bindings/README.md`.

`nix develop` provides libctranslate2 + whisper-cpp + pybind11 headers and exports the hint vars `CT2_INCLUDE_DIR` / `CT2_LIBRARY_DIR` (ct2_bindings) and `WHISPER_INCLUDE_DIR` / `WHISPER_LIBRARY_DIR` (whisper_bindings). `bash scripts/build_bindings.sh` probes these (plus `/usr/{,local/}include`, `/opt/homebrew/include` fallbacks), builds both extensions in-place via `setup.py build_ext --inplace`, and import-tests the resulting `.so` files; it exits non-zero with a pointed message if either header isn't reachable.

### Backend selection at runtime

`stt/whisper.py:Backend.from_env()` reads `STT_BACKEND` (canonical name mirrored in `env.py`):

- `STT_BACKEND=ct2` (or `ctranslate2`, `faster-whisper`) -> CTranslate2 path.
- Anything else (default) -> whisper.cpp path.

The top-level `server.py` adds `STT_BACKEND=qwen3_omni` as the *default* selector for the omni model; the whisper aliases above are only honored in the dedicated whisper code path.

### Nixpkgs source

Both `pkgs.ctranslate2` and `pkgs.whisper-cpp` resolve from the pinned `nixpkgs` input (`github:NixOS/nixpkgs/nixpkgs-unstable`) as-is.

## Perf-P0 hot-path notes (round-3 review fixes)

Four hot-path allocations were eliminated:

- `realtime/pipeline.py` no longer calls `audio_chunk.tolist()` before `pacer.play(...)`. The pacer accepts ndarrays directly (it was already re-wrapping via `np.asarray(...)`), so the per-chunk box of ~6000 PyFloats per TTS chunk is gone. `OutboundPacer.play` got an ndarray-aware empty-check (`isinstance(x, np.ndarray) and x.size == 0`) replacing the old `if not audio_24k_samples:` which would have raised on an ndarray. Public signature unchanged -- lists still work.
- `audio/g711.py` ships two module-scope LUTs (`_ULAW_ENCODE_LUT`, `_ALAW_ENCODE_LUT`, both `uint8[65536]`) keyed by the `int16.view(uint16)` bit pattern. `f32_to_ulaw_bytes` / `f32_to_alaw_bytes` now do `LUT[v.view(np.uint16)].tobytes()` -- byte-for-byte identical to the old per-sample loop across the full int16 sweep (verified in `test_review_fixes.py::test_fix_perf_p0s`), ~80x faster on a 10k-sample buffer.
- `inspect_api/audio_store.py:_Track.append_pcm16` no longer flushes per write; flush remains in `close()`. `append_f32` does the f32->s16 conversion via numpy in one shot (`np.rint(np.clip(arr, -1, 1) * 32767.0).astype('<i2').tobytes()`) instead of a Python loop. Semantic flex: the old per-sample loop up-cast to float64 before the multiply; the vectorized path stays float32, so inputs whose float64 multiply lands near `n + 0.5` may round to a different LSB (1-bit, ~30 dB below FS). This buffer is inspector capture only (not user-facing audio), so the precision drop is well below the noise floor -- deemed acceptable for the speedup. Accepts ndarray / list / bytes / bytearray; the only callers pass list or ndarray. `Session.capture_inbound_f32` / `capture_outbound_f32` now pass values through unchanged (the old `list(samples)` coercion is gone).
- `realtime/audio_out.py` hoisted `numpy as np` and `scipy.signal.resample_poly` to module top (the librosa fallback inside `_resample_24k_to_48k` stays lazy since librosa is optional). `realtime/audio_in.py` was already clean.

Bonus: `realtime/audio_in.AudioIngest` switched its 16k output buffer from `np.concatenate([_buf, decimated])` (O(N^2) under consumer stalls) to a `deque[np.ndarray]` with a single `np.concatenate` on `take_array()`. Public surface (`take`, `take_array`, `get_total_samples_consumed`, `get_total_input_samples`) unchanged.
