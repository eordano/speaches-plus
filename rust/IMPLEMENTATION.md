# Implementation notes

Source files are comment-free repo-wide (`scripts/strip-comments.py`
is the canonical formatter); for the realtime lane, the *why* behind
subtle decisions, the spec references, and the constraints invisible
from the code alone live here. Spec references are to
`docs/book/07.1-barge-turn-spec-rfc-v3.md` (RFC v3) unless noted; the
spec is the authority and is not reproduced. Any drift between code
behaviour and this document is reconciled here.
`docs/book/07-speech-stack.md` is the narrative companion.

## Layout

The realtime lane's helpers were promoted out of `src/realtime/` to
top-level modules as the OpenAI-compat surface grew around them; this
table is the current shape.

```
src/main.rs                axum HTTP binary: routes, AppState, realtime_post, transcriptions_post
src/defaults.rs            single source of truth for all numeric defaults
src/errors.rs              reserved error-code registry
src/ids.rs                 IdSource trait (random vs counter for replay); default
                           RandomIdSource, CounterIdSource for replay tests, injected
                           via Session::with_dependencies (§13)
src/types.rs               Millis / DurationMs / Epoch / typed Ids / typed audio
src/trace.rs               canonicalize_trace + trace_diff (RFC v3 §15.4)
src/otel.rs                OTel span helpers + trace context propagation
src/models.rs              process-global lazy model registry
src/soak.rs                soak harness: SoakConfig/SoakResult + RSS / open-fd probes
src/audio/                 decoders + resampling (opus, ogg, webm, wav, g711, avdecode)
src/conversation/          llm.rs -- OpenAI-compatible SSE forwarder
src/vad/                   Silero v5 ONNX + speech segmenter + supervisor (VadModel, VadInfer)
src/stt/                   dual-backend Whisper + mel frontend + silence noise gate
src/tts/                   Kokoro 82M v1.0 ONNX: mod.rs (synth), chunk.rs + text.rs
                           (chunker, emoji/markdown strip), phonemize.rs +
                           phonemize_glue.c (libespeak-ng FFI), vocab.rs
                           (phoneme -> token-ID + clean_phonemes), npz.rs
                           (voices.bin reader), http.rs (/v1/audio/speech)
src/eou/                   end-of-utterance gate (heuristic/integrated/fusion + text + audio)
  mod.rs                   EouModel trait, Eagerness, sigmoid_lerp, race_hard_cap
  fusion.rs                Gated (default) / Noisy-OR / Max / Mean / Weighted fusion + features
  heuristic.rs             punctuation/length heuristic baseline
  integrated.rs            IntegratedEouBackend trait + FakeIntegratedBackend
  loader.rs                EouConfig + env-var loader
  audio.rs                 smart-turn-v3 audio EOU classifier (§6.2.2)
  onnx.rs                  LiveKit turn-detector ONNX session + shared handle
  bpe.rs                   GPT-2/Qwen byte-level BPE
  byte_map.rs              256->char mapping for byte-level BPE
  chat_template.rs         Qwen chat template renderer
  special_trie.rs          special-token longest-match split
src/inspect/               inspector lane (§14.1) -- speaches-compatible
  mod.rs                   re-exports + env config (session_dir, retention)
  types.rs                 WireEvent + Corr; constants.rs holds LANES + ERR_KINDS
  relay.rs                 per-session relay (replay buffer, NDJSON, broadcast, error mirror)
  registry.rs              process-global session_id -> relay map; SessionMeta listing
  audio_store.rs           mic_in (16 kHz s16) / tts_out (24 kHz s16) raw files + sidecar
  retention.rs             cleanup_on_startup (count/bytes/days)
  routes.rs                axum handlers for /v1/inspect/{sessions,history,audio,WS}
src/oapi/                  OpenAI-compat surface beyond realtime: chat engines
                           (chat_engine*, batch_chat, completions), transcriptions,
                           audio_speech*, models_handler, ocr*, text_embeddings,
                           tool_parse, lora, fine_tuning, voice_profiles, ...
                           mod.rs owns the error-envelope helpers (see "Wire shapes")
src/diarization/           diarization engine (segmentation + embeddings + clustering)
src/pii/                   PII classify/redact surface
src/realtime/
  mod.rs                   PC/SDP exchange, on_track / on_data_channel wiring,
                           Intent, capabilities_json
  session.rs               per-connection orchestration (STT -> LLM -> TTS -> events)
  state.rs                 SessionState, phase enums, ConversationItem store, invariants
  pipeline.rs              response pipeline: chunker -> TTS worker -> pacer handoff
  events.rs                response bracket / done / transcript-delta emission helpers
  session_update.rs        StagedSessionUpdate parse-all -> validate-all -> commit-all (§11.2.1)
  cancel.rs                SessionCancel + LaneGuard in-flight lane accounting
  eou_eager.rs             try_eager_dispatch (Appendix C/D throttle + spawn)
  eou_predicted.rs         speculative STT/LLM runners + PredictedTokenBuffer (§D.2)
  eou_integrated.rs        IntegratedVerdictAction plumbing
  diarization.rs           realtime diarization event lane (run_diarization)
  audio_in.rs              opus -> 48 kHz mono -> 16 kHz mono f32
  audio_in_ws.rs           WS ingest: base64 PCM/G.711 decode + resampler carry
  audio_out.rs             24 kHz mono -> 48 kHz mono -> opus -> outbound track
  audio_out_ws.rs          WsAudioPacer; AudioPacer abstracts WebRTC vs WS output
  transport.rs             EventSink / OutboundAudioSpec (WebRTC vs WebSocket)
  websocket.rs             /v1/realtime WebSocket endpoint (realtime_ws)
  v2_compat.rs             nested <-> flat session.update shape normalisation
  framing.rs               full_message / partial_message envelope
  sdp_filter.rs            aiortc -> webrtc-rs SDP normaliser
  inspector.rs             InspectorEvent enum + sink trait + InspectorEvent -> wire mapping
  wire.rs                  OutboundEvent enum (typed alphabet, serde tag="type")
  fuzz.rs                  random-walk property fuzzer over SessionState (§15.5)
  order_harness.rs         wire-order Violation harness shared by ordering tests
```

There is no `eou/languages.rs`: per-language thresholds and the
smart-turn constants live in `defaults.rs`, `eou/mod.rs`, `eou/audio.rs`.

## Conventions

- **`defaults.rs` is the single source of truth for every numeric
  default.** Nothing else hard-codes one; test code reads the same
  constants; changing a default is a one-file edit. Organisation (mirrors
  RFC §17 plus internal sections):

  | Module | Source of truth |
  |---|---|
  | `session` | §17.1 (`MAX_DURATION_S`, `MAX_DURATION_HARD_CAP_S`) |
  | `turn_detection` | §17.2 + §4.5 ranges |
  | `eou` | §17.3 incl. eagerness triples and curve_k |
  | `buffer` | §17.4 |
  | `response` | §17.5 (drain cap floor/ceiling) |
  | `wire` | §17.6 |
  | `inspector` | §17.7 (incl. `relay_cap`) |
  | `inspect` | session_dir + retention_count/bytes/days |
  | `vad` / `audio` / `stt` / `kokoro` | implementation internals (sample rates, frame sizes, model hyperparameters) |
  | `env` | environment-variable names; consumers reference these symbols, so a rename touches one line |

- **No source-level comments**; explanations live here. (The old
  in-process conformance bridge `tests/conformance.rs` was culled in
  `431c0d5fb` -- see §15.6 below.)

## RFC v3 conformance

`GET /v1/realtime/capabilities` returns:

```json
{ "rfc_version": "v3",
  "features": { "eou_kinds": ["vad","text","audio","fusion"],
                "input_audio_formats": [...], "output_audio_formats": [...] },
  "extensions": { "eou_kinds": ["heuristic","integrated"],
                  "eager_eou": true, "integrated_eou": true,
                  "predicted_resp_phase": true } }
```

`EouKind::V3_SPEC` is the canonical `[Vad, Text, Audio, Fusion]`
ordering used by this endpoint; `EouKind::EXTENSIONS` holds `Heuristic`
and `Integrated` (deferred from v3 by §16), which advertise under
`capabilities.extensions`, never `capabilities.eou_kinds`.
`EouKind::is_v3_spec()` is the gate.

### Spec compliance status

Live and tested:

- §2.1 `SessionPhase::{Pending, Active{created_at_ms},
  Terminated{reason}}`. Termination reasons propagate from
  `terminate_with_error` (vad / stt / internal / model_load) and the
  `MaxDuration` hard timeout.
- §2.4 invariants -- I1, I5, I7, I9 enforced at runtime; I3 is a
  typestate property of `OpenBuffer/SealedBuffer`.
  `IllegalRespTransition` rejects illegal `from->to` pairs.
- §2.6 sum-type encoding -- `RespPhase` is a Rust `enum`; `Predicted`
  deliberately omits any `wire_emitted` flag (I7 is structural).
- §3.2 buffer rotation is bound to commit via the sealed-buffers map;
  the user item is appended at the commit point, not on
  `vad.speech_stopped` (§I6 -- see Conversation store).
- §3.4 sealed-buffer retention map (FIFO `K = 4`, drop on
  transcription complete).
- §4.1 VAD frame contract; `VadModel::reset()` zeroes state on Stopped.
- §4.5 hot reload -- `threshold` / `prefix_padding_ms` /
  `silence_duration_ms` / `barge_in_delay_ms` / `create_response` /
  `type` all live via `Arc<TurnDetectionConfig>`. `type=none` mid-update
  collapses any in-flight Speaking buffer to Stopped synchronously and
  disarms `commit_timer`.
- §6.2 EOU kinds. `kind=vad` waits Silero's `silence_duration_ms`;
  `kind=heuristic` uses `HeuristicEouModel`; `kind=text` uses the
  LiveKit `turn-detector` ONNX (`TextEouModel`, `eou/onnx.rs`, from
  `EOU_MODEL_PATH`); `kind=audio` uses `pipecat-ai/smart-turn-v3` ONNX
  (`AudioEouModel`, `eou/audio.rs`, from `EOU_AUDIO_MODEL_PATH`).
- §6.2.2 audio EOU. smart-turn expects a Whisper-Tiny log-mel
  spectrogram `[1, 80, 800]` (8 s x 16 kHz / hop 160), input tensor
  `input_features`; output `logits` is one scalar of RAW logits -- apply
  sigmoid. Preprocessing in `eou/audio.rs`: take the last
  `eou.audio_window_ms` of buffered 16 kHz mono f32; clamp to `[-1, 1]`;
  if shorter than 8 s, **zero-pad on the left** so the audio sits at the
  *end* of the window (`eou.audio_pad_alignment = leading`, smart-turn's
  required alignment); reflect-padded STFT (`n_fft = 400`, `hop = 160`,
  Hann) -> 80-bin mel -> `log10(max(mel, 1e-10))` clipped to `max - 8`
  -> `(x + 4) / 4` (standard whisper normalisation). Session behind
  `Arc<Mutex<ort::Session>>`, lazily loaded by
  `eou::audio::try_load_from_env`. `EouModel::score_with_audio(context,
  audio, sample_rate)` defaults to delegating to `score(&str)`; the
  dispatch in `run_eou_dispatch` always calls `score_with_audio`
  (passing `eou::audio::SAMPLE_RATE`) through
  `tokio::task::spawn_blocking`.
- §6.2.3 `kind=fusion`: `FusionEouModel::score_with_audio` runs the
  text head via `score(context)` and the audio head via
  `score_with_audio`, then `combine_fusion_with_features` applies the
  configured rule (`FusionRule::default()` is `Gated`):

  | Rule | Formula |
  |---|---|
  | `gated` (default) | `g * p_audio + (1 - g) * p_text` where `g` is a logistic gate over an 8-feature vector (bias, p_text, p_audio, log audio seconds, log partial chars, strong/soft terminator, continuation-last-word); `DEFAULT_GATED_FUSION_WEIGHTS` was trained on 350 samples, 93.14% acc, and audio_ms is threaded through `extract_gated_fusion_features` |
  | `noisy_or` | `1 - (1 - p_text)(1 - p_audio)` |
  | `max` | `max(p_text, p_audio)` |
  | `mean` | `(p_text + p_audio) / 2` |
  | `weighted` | `w * p_text + (1 - w) * p_audio` (`w = fusion_weight_text`, default 0.5) |

  Per §6.5 a head returning NaN/Inf/out-of-`[0, 1]` is failed
  (`is_garbage_prob`): the other head's score is the verdict; both
  failed -> `p = 1`. Env: `EOU_FUSION_RULE`, `EOU_FUSION_WEIGHT_TEXT`.
- §6.3 hard cap -- see "Hard-cap race" below.
- §6.4 + §6.4.1 -- `delay_curve` via `sigmoid_lerp`; per-language
  thresholds via `EouConfig::threshold_for_language`.
- §6.4.2 defaults pinned via `defaults::eou::*`.
- §6.5 fast-commit on uncertainty -- see "Failure policy" below.
- §6.6 trigger on `vad.speech_stopped`, hard cap and verdict delay
  racing per §6.3.
- §6.7 partial-transcription task aborted on Speaking -> Stopped via the
  VAD supervisor's `partial_task` slot; next Stopped -> Speaking
  re-entry restarts naturally.
- §7.1 commit -- buffer rotates to a `SealedBuffer` and
  `conversation.item.created` is emitted; rejected commits below
  `min_speech_ms` emit `input_audio_buffer.speech_stopped` *before* the
  error (matches the §004-backchannel fixture) and restore
  `Stopped -> Silent`.
- §7.2 `conversation.item.input_audio_transcription.failed` on STT
  error; item flipped to `Incomplete`; no auto-response.
- §7.3 played_ms ordering -- see "played_ms post-abort snapshot" below.
- §7.4 backpressure -- audio gate (`OUTBOUND_QUEUE_CAP_MS`) plus
  event-count gate (`OUTBOUND_QUEUE_CAP = 256`) via
  `Session.outbound_inflight: AtomicU32`. See "Backpressure" below.
- §8.3 drain cap formula `clamp(2 * planned_ms,
  defaults::response::DRAIN_CAP_FLOOR_MS,
  defaults::response::DRAIN_CAP_CEILING_MS)`.
- §8.5 `audio_end_ms` on every `response.done` (all four statuses).
  `response.done.response.output[0].id` is read from the live
  `RespPhase::item_id`, so it agrees with the earlier
  `response.output_item.added/done` ids of the same response.
- §9.3 server-issued `conversation.item.assistant_truncated` distinct
  from client-issued `conversation.item.truncate`; guarded by
  `played_ms_snapshot > 0` (W7).
- §9.5 simultaneous barge-in single-slot replacement.
- §10.3 wire invariants W1-W8 in `conformance/lib/trace_invariants.py`
  (canonical, shared with Go).
- §10.4 envelope: `full_message` carries `id`; `partial_message`
  carries `id`, `fragment_index`, `total_fragments`. Fragment cap from
  `defaults::wire::DATA_CHANNEL_FRAGMENT_MAX`.
- §10.5 reserved error-code registry incl. `stt_failed`.
- §11.2.1 atomic apply -- `handle_session_update` is parse-all ->
  validate-all -> commit-all; `StagedSessionUpdate` collects parsed
  fields and commits only after every field validates, so a
  partially-applied update is unrepresentable.
- §11.3 / §11.4 hard timeout -- fires after `session_max_duration_s`
  (default `defaults::session::MAX_DURATION_S = 1800`, env
  `SESSION_MAX_DURATION_S`). Clients MAY shorten via `session.update`;
  values outside `[1, defaults::session::MAX_DURATION_HARD_CAP_S =
  3600]` are rejected with `session_update_invalid`. Reschedule
  preserves elapsed time (`compute_remaining_timeout_ms`); a new value
  placing the fire time in the past fires immediately. Emits
  `session.done(reason="max_duration")`, terminates, closes PC.
- §13 typed Ids (`SessionId/ItemId/ResponseId/EventId`) via `IdSource`.
- §14.1 inspector lane -- see its own section below.
- §14.2 transitions toggle (`INSPECTOR_TRANSITIONS=1` +
  `INSPECTOR_TRANSITIONS_SAMPLE_RATE`).
- §14.4 capabilities endpoint at `GET /v1/realtime/capabilities`.
- §15.4 `src/trace.rs::canonicalize_trace` + `trace_diff`
  (counter ids, ts -> 0, float rounding to 3 dp, audio payload
  length-only).
- §15.5 random-walk property fuzzer (`realtime/fuzz.rs`) -- 5000 steps
  x multiple seeds, `check_state` after every transition.
- §15.6 conformance fixtures at `<repo>/conformance/fixtures/` with
  `conformance/lib/trace_invariants.py` (W1-W8) as the canonical
  assertor. The Rust in-process bridge (`tests/conformance.rs`, a pyo3
  `auto-initialize` FFI into that module) was culled in `431c0d5fb`;
  Go's `TestConformanceCorpus` is currently the only live wire-trace
  gate. If the bridge is restored: pyo3 needs `PYO3_PYTHON` at build
  time, and on macOS with the Xcode Python, `DYLD_FRAMEWORK_PATH`
  pointing at its framework prefix so the dylib resolves at load time.

## Inspector lane (§14.1)

Full HTTP/WebSocket transport. `realtime/inspect/` is structurally
compatible with `speaches/src/speaches/inspect/`; the `WireEvent` JSON
shape matches `speaches.types.inspect.InspectorEvent` (`session_id,
seq, ts_mono_ns, ts_wall, lane, kind, corr, span_id, payload`).
`Session::inspector` is a `FanoutSink` of `(TracingSink | NoopSink,
RelayInspectorSink)`; `RelayInspectorSink::emit` translates each
variant via `to_wire()` to the speaches lane/kind alphabet. NDJSON
persists under `INSPECT_SESSION_DIR` (default
`~/.cache/speaches/sessions`); raw audio per channel:

| Channel | Capture point | Format on disk |
|---|---|---|
| `mic_in` | `attach_audio_track` after AudioIngest produces 16 kHz f32 | s16 LE 16 kHz |
| `tts_out` | `synthesize_and_play` with the 24 kHz Kokoro output before resampling | s16 LE 24 kHz |

HTTP endpoints (compatible with the dashboard at `/inspector/`):

| Path | Behavior |
|---|---|
| `GET /v1/inspect/sessions` | live SessionMeta list from registry |
| `GET /v1/inspect/sessions/history` | NDJSON files on disk, mtime desc |
| `GET /v1/inspect/sessions/history/{sid}` | application/x-ndjson stream |
| `GET /v1/inspect/sessions/{sid}/audio?channel=&from_ms=&to_ms=` | audio/wav slice (live AudioStore for in-memory sessions, raw + sidecar offset_ms for ended ones) |
| `WS /v1/inspect/{sid}/stream` | replay-buffer snapshot then live broadcast; falls back to NDJSON replay-from-disk + close when session is gone |

Lane mapping (RFC v3 §14.1.2 / speaches `LaneId`):

| Rust `InspectorEvent` variant | lane | kind |
|---|---|---|
| `VadConfirmedStart` | `vad` | `confirmed_start` |
| `VadConfirmedStop` | `vad` | `stopped` |
| `BargeinPending` | `bargein` | `bargein_pending` |
| `BargeinFired` | `bargein` | `bargein_fired` |
| `BargeinSuppressed` | `bargein` | `bargein_cancelled` |
| `EouScored` | `eou` | `scored` |
| `EouHardCapFired` | `eou` | `hard_cap_fired` |
| `EouPredictedOverflow` / `Rollback` | `response` | `predicted_overflow` / `predicted_rollback` |
| `PacerPlayedMs` | `tts_pacer` | `played_ms` |
| `StateTransition` | `state` | `transition` |
| `DrainStart` / `DrainComplete` | `response` | `drain_start` / `drain_complete` |
| `PartialTranscription` | `stt` | `partial` |
| `Predicted{Rollback,Suppressed,Promoted}` | `response` | `predicted_*` |
| `OutboundQueueExceeded` | `wire` | `err` |
| `InvariantViolation` | `error` | `invariant_violation` |
| `VadFailed` | `error` | `vad_failed` |
| `BackchannelSuppressed` | `turn` | `backchannel_suppressed` |

Plus `wire.in` / `wire.out` mirrors from `Session::handle_dc_message`
and `Session::emit_event` / `Session::emit`, carrying `event_type` and
`bytes`. Errors whose kind is in `ERR_KINDS`
(`error|raised|dropped|failed|...`) auto-mirror to the `error` lane via
`InspectorRelay::publish`. Retention runs once at first-session
creation via `inspect::run_startup_cleanup()`; tunable via
`INSPECT_RETENTION_{COUNT,BYTES,DAYS}`.

## State machine -- `state.rs`

`Session` owns one `Mutex<SessionState>`; every transition flows
through it, followed by `check_state(state)` (§2.4). `SessionState`
carries:

- `session: SessionPhase`, `vad: VadPhase`, `resp: RespPhase`
- `instructions, pc, dc, current_response, last_epoch`
- `timeout_task` (§11.3), `commit_timer` (§6.3), `bargein_task` (§9.2),
  `pending_bargein`
- `conversation: Vec<ConversationItem>`
- `current_speech_item: Option<String>` (partial-STT bookkeeping)
- `sealed_buffers: VecDeque<SealedBuffer>` (FIFO §3.4 retention)

Each non-`None` `RespPhase` variant carries `id, item_id, epoch`;
`Streaming` and `Drain` carry `played_ms: Arc<AtomicU64>` and
`planned_ms`. Cross-struct id mismatch is unrepresentable.

`SealedBuffer { item_id, audio, audio_start_ms, audio_end_ms }` is
stored on commit, dropped when the corresponding
`input_audio_transcription.completed` is emitted (§3.4 rule 1), FIFO
evicted at `defaults::buffer::SEALED_BUFFER_RETENTION_COUNT` (rule 2).

### Transitions

| Transition | Method |
|---|---|
| `None -> Created` (typed STT/LLM path) | `resp_create_from_none(id, item_id, runtime)` -- bumps epoch |
| `None -> Predicted` (eager EOU) | `resp_start_predicted(id, item_id, eou_score, runner)` -- bumps epoch |
| `Predicted -> Created` (promote on commit) | `resp_promote_predicted_to_created(runtime)` -- preserves id+epoch (I9) |
| `Created -> Streaming` | `resp_advance_to_streaming(played_ms)` |
| `Streaming -> Drain` | `resp_drain(planned_ms)` |
| `* -> None` (terminal) | `resp_retire_to_none()` |
| `Predicted -> None` (rollback) | `resp_retire_predicted()` |

Illegal pairs return `InvariantViolation::IllegalRespTransition { from,
to }` and terminate the session with `internal_state_error`.

### Invariant enforcement (§2.4 + §9.6)

Every mutation under `Mutex<SessionState>` ends with
`check_or_react(session, &state)`; on `Err(InvariantViolation)`:
`tracing::error!`, `InspectorEvent::InvariantViolation`, spawn
`terminate_with_error(... INTERNAL_STATE_ERROR ...)`.

I1 (mutual exclusion), I5 (epoch monotonicity), I7 (Predicted
invisibility), I9 (Predicted retirement), `ResponseRuntimeMismatch`,
`IllegalRespTransition` are runtime-checked. I2/I3/I4 are wire-level,
enforced by the canonical assertor against captured traces; I3 is also
a typestate property of `OpenBuffer/SealedBuffer`.

## Inbound event handling (RFC §10.1)

| Inbound type | Behaviour |
|---|---|
| `session.update` | Parse-all -> validate-all -> commit-all (§11.2.1). Reply `session.updated` echoing `instructions` + the resolved `turn_detection` snapshot. Bad shape -> `session_update_invalid` with `param`. |
| `input_audio_buffer.clear` | Emit `input_audio_buffer.cleared`. |
| `input_audio_buffer.commit` | Reject with `input_audio_buffer_commit_empty` (manual commit incompatible with VAD-driven WebRTC buffer). |
| `response.cancel` | Snapshot in-flight, abort, emit §10.2 cancellation bracket + truncate. No active response -> `response_cancel_not_active`. |
| `conversation.item.create` | Append `Text` item; echo `conversation.item.created`. |
| `conversation.item.delete` | Remove by id; emit `conversation.item.deleted`. Missing -> `invalid_request_error`. |
| `conversation.item.truncate` | Clamp `AssistantAudio.audio_ms`; emit `conversation.item.truncated`. |
| `response.create` | Conversation-intent only. Reject with `response_already_active` if one is in flight. Builds chat history from instructions + transcript-bearing items, runs the same `run_response` used post-STT. |
| anything else | `unknown_event_type`. |
| non-JSON | `invalid_request_error`. |

## Conversation store (RFC §3.3)

```
ConversationItem { id, role: User|Assistant|System, status: InProgress|Completed|Incomplete,
                   content: UserAudio { transcript, audio_end_ms }
                          | AssistantAudio { transcript, audio_ms }
                          | Text(String) }
```

Append points:

- **Commit (`commit_after_eou`)**: append `UserAudio { InProgress }`
  alongside `conversation.item.created`. The append happens here, not
  on `vad.speech_stopped`, so a cancelled commit timer leaves no
  phantom `in_progress` entries (§I6).
- **STT completion (`process_utterance`)**: mutate user item to
  `Completed`, fill `transcript`, alongside
  `conversation.item.input_audio_transcription.completed`.
- **Response success (`run_response`)**: append `AssistantAudio` with
  full transcript and `played_ms` snapshot.
- **Inbound `conversation.item.create`**: append `Text` (defaults to
  `Completed`); delete / truncate as in the table above.

## Barge-in

Per-utterance work runs inside a `JoinHandle` on
`SessionState.current_response`. On `VadEvent::SpeechStarted`:
`cancel_current_response()` takes the handle and aborts; the canceller
emits the §10.2 cancelled bracket; `conversation.item.assistant_truncated`
follows the bracket close when `played_ms_snapshot > 0` (§9.3, W7);
then `input_audio_buffer.speech_started { item_id }` for the new
utterance. `docs/book/07-speech-stack.md` §Barge-in narrates the full
commit sequence. `tokio::task::spawn_blocking` for Kokoro synth runs
the underlying thread to completion; on cancel the result is discarded.

### Barge-in delay + suppression (§9.2-§9.4)

`BARGE_IN_DELAY_MS` (clamped to
`defaults::turn_detection::BARGE_IN_DELAY_MS_MAX`, default
`defaults::turn_detection::BARGE_IN_DELAY_MS = 0`) defers cancel
commitment so a brief noise burst doesn't tear down the response.

- `== 0`: `SpeechStarted` immediately calls `cancel_current_response()`.
- `> 0`: emit `BargeinPending`, store `pending_bargein`, spawn a delay
  task that calls `take_pending_bargein_if(item_id)`. If still set, run
  `commit_bargein`; if cleared, exit silently.

Suppression (§9.4): `SpeechCommitted` for the same `item_id` during the
delay window clears the slot, emits `BargeinSuppressed`, and the
handler returns early -- no `committed`, no `created`; the active
response continues.

### `status_details.reason` (§8.4)

| Path | `status` | `status_details.reason` |
|---|---|---|
| `response.cancel` honored | `cancelled` | `client_cancelled` |
| Barge-in (VAD) committed | `cancelled` | `barge_in` |
| Drain cap expired | `incomplete` | `drain_cap` |
| LLM upstream / config / empty | `failed` | `llm_error` |
| Kokoro TTS unavailable / errored | `failed` | `tts_error` |
| Outbound queue cap exceeded | `failed` | `client_too_slow` |
| Natural completion | `completed` | (omitted) |

## End-of-utterance gate (RFC §6)

- `trait EouModel { fn score(&self, context: &str) -> f32 }`.
  `HeuristicEouModel` is a hesitation/continuation lookup over the last
  word / trailing punctuation. `TextEouModel` (`eou/onnx.rs`) is the
  LiveKit `turn-detector` (Qwen2.5-0.5B distil INT8 ONNX, §6.2.1):
  byte-level BPE in `eou/bpe.rs`, the 256->char byte-permutation table
  in `eou/byte_map.rs`, longest-match special-token splitter in
  `eou/special_trie.rs`, Qwen chat template
  (`<|im_start|>role\ncontent<|im_end|>\n...`) in
  `eou/chat_template.rs`. `score(context)` renders the partial through
  the chat template (with empty turns, since the trait passes a single
  string), encodes, left-truncates to `EOU_MAX_CONTEXT_TOKENS`
  (default `defaults::eou::MAX_CONTEXT_TOKENS = 128`, per §6.2.1), and
  runs one ORT pass on `defaults::eou::INPUT_IDS` /
  `defaults::eou::ATTENTION_MASK` to get
  `softmax(logits[-1])[<|im_end|>]`. Loaded once per process via
  `eou::onnx::shared_text_eou_model()` (`OnceLock`), reading
  `EOU_MODEL_PATH`, `EOU_TOKENIZER_PATH` (default
  `<dir(model)>/tokenizer.json`), `EOU_MAX_CONTEXT_TOKENS`.
- `EouConfig` (`eou/loader.rs`) carries every knob, populated by
  `from_env` from `defaults::eou::*`.
- `sigmoid_lerp(p, p_threshold, p_max, max_delay_ms, min_delay_ms)`
  maps `p in [p_threshold, p_max]` to `[min_delay_ms, max_delay_ms]`
  via the §6.4 logistic curve with `k = defaults::eou::CURVE_K = 12`.

### Eagerness (§6.4.2 + §17.3)

`defaults::eou::eagerness::*` carries the Low/Medium/High
`(p_threshold, min_delay_ms, max_delay_ms)` triples; Auto resolves to
Medium (the default). `EOU_EAGERNESS={low,medium,high,auto}` populates
the triple AND wins over the per-knob
`EOU_P_THRESHOLD`/`EOU_MIN_DELAY_MS`/`EOU_MAX_DELAY_MS`.

### Per-language thresholds (§6.4.1)

`EOU_THRESHOLDS=lang:0.5,fr:0.7,es:0.4` populates
`EouConfig::thresholds`; `threshold_for_language(Option<&str>)` falls
back to the global `p_threshold` for unknown languages.

### Backchannel filter (§6.5 / §17.4)

`min_speech_for_response_ms` (default 600). Utterances in
`[MIN_SPEECH_MS, min_speech_for_response_ms)` commit (recorded in the
conversation, transcript filled) but skip the auto-response leg;
inspector emits `BackchannelSuppressed`.

### Failure policy (§6.5)

`run_eou_dispatch` collapses every uncertainty source to fast-commit
(`p = 1, delay_ms = min_delay_ms`); the inspector `cancelled_by` field
carries the discriminator:

| Cause | `cancelled_by` |
|---|---|
| Hard cap fires while EOU computing | `hard_cap` (phase `during_eou`) |
| Inference timeout | `timeout` |
| Inference returned an error | `error` |
| NaN / out-of-range probability | `garbage_prob` |

The hard cap remains the outer bound but rarely fires when
`min_delay_ms` is far below `silence_hard_cap_ms` (default 5000).

### Eager LLM dispatch (Appendix C / D -- Predicted)

When the EOU score reaches `eager_p_threshold` mid-utterance,
`run_eou_dispatch` calls `try_eager_dispatch`, which:

1. Throttles via `last_eager_dispatch_at`; next attempt only after
   `eager_interval_ms`.
2. Spawns a speculative STT runner over the buffered audio
   (`spawn_predicted_stt`).
3. Spawns an LLM runner (`spawn_predicted_llm`) streaming chat
   completions into a `PredictedLlmShared` buffer bounded by
   `predicted_token_buffer_cap`.
4. Records `RespPhase::Predicted { runner, llm_runner, .. }`; I7
   (`Predicted => wireEmitted == false`) holds because
   `Session::emit{,_event}` short-circuits when topic is `Response` and
   the phase is `Predicted`.
5. Emits `EouEagerDispatch` (lane `eou`, kind `eager_dispatch`) plus
   the pre-existing `PredictedPromoted` event for back-compat.

Knobs in `defaults::eou::*` with env overrides `EOU_EAGER_P_THRESHOLD`,
`EOU_EAGER_MAX_INFLIGHT`, `EOU_EAGER_INTERVAL_MS`,
`EOU_PREDICTED_TOKEN_BUFFER_CAP` (defaults 0.5 / 1 / 250 ms / 256
tokens). Disable with `EOU_EAGER_P_THRESHOLD=1.0` (the
`EAGER_P_THRESHOLD_DISABLED` sentinel); `eager_disabled()` is true for
any non-finite or `>= 1.0` value.

#### Promotion (`Predicted -> Created`)

In `commit_after_eou`, `resp_retire_predicted_full` returns both
runners. The LLM runner is awaited up to `silence_hard_cap_ms`; on
success the buffered text replaces the live LLM call inside
`run_response` -- a single-shot `mpsc` channel feeds the existing
chunker -> TTS -> wire-emit loop. Epoch and ids are preserved across
the transition (I9), so the wire sees the same `response_id` on
`response.created` and beyond.

#### Rollback paths

| Trigger | `EouPredictedRollback.reason` | Source |
|---|---|---|
| `vad.speech_started` re-enters during Predicted | `speech_resumed` | `handle_vad_event` |
| `response.cancel` during Predicted | `cancel_event` | `handle_response_cancel` |
| `EouKind::Integrated` turn resumes | `turn_resumed` | `handle_stt_turn_resumed` |
| Token buffer exceeds cap | `predicted_overflow` (preceded by `EouPredictedOverflow`) | `commit_after_eou` |
| Partial vs. final transcript Jaccard < `1 - EAGER_TRANSCRIPT_MISMATCH_RATIO` | `transcript_mismatch` | `commit_after_eou` |

Rollback aborts both runners (`PredictedRunner::task.abort()`,
`PredictedLlmRunner::abort`), bumps the epoch via
`resp_retire_predicted_full`, clears the throttle, and emits both
`PredictedRollback` (back-compat) and `EouPredictedRollback` (spec
lane). The mismatch heuristic is character-set Jaccard on
whitespace-stripped, lowercased strings; threshold
`EAGER_TRANSCRIPT_MISMATCH_RATIO = 0.5` (sets must overlap >= 50%).

## Hard-cap race (§6.3)

`eou::race_hard_cap(deadline, future)` is the §6.3 mandatory-pattern
helper: a deadline derived **once** at `vad.speech_stopped` is observed
twice -- while EOU is computing, then again while the verdict-derived
`delay_ms` sleeps (`tokio::select!` of `sleep_until(hard_cap_deadline)`
against each). Both observations bias toward the deadline arm so a
stalled classifier still triggers `eou.hard_cap_fired` -- emitted at
two observation points with `phase in {during_eou, during_wait}` per
§6.3.1. The `clamp` form (`delay = min(delay, hard_cap)`) is rejected
by the spec because it cannot guard a stalled classifier; the
`regress-v2-rust-no-hard-cap` test asserts the bias.

### Commit timer slot

`Session::install_commit_timer` is single-slot on purpose: the slot
holds the `JoinHandle` of the spawned `run_eou_dispatch` task, which
itself races `commit_timer` against `hard_cap_timer` via
`race_hard_cap` (twice, as above). Aborting the JoinHandle cancels
both arms atomically -- required by §6.6 when `vad.speech_started`
re-enters during the race. A two-slot design would risk abort-order
races between the two arms.

## Drain phase (§8.3)

After the LLM stream ends, `played_ms` (wall-clock-confirmed playback)
usually trails `planned_ms` (total ms synthesised). In `drain_pacer`:
`drain_cap_ms = clamp(2 * planned_ms, DRAIN_CAP_FLOOR_MS,
DRAIN_CAP_CEILING_MS)`. If `played_ms >= planned_ms`: flush, complete.
Else emit `DrainStart` and wrap `pacer.flush()` in
`tokio::time::timeout(drain_cap_ms)`: natural drain -> `completed`;
cap timeout -> `incomplete` + `status_details.reason="drain_cap"`.
Drain status is inspector-only, never on the wire.

## Partial transcription (§5)

Off by default; `PARTIAL_STT_ENABLED=1` enables.
`VadProcessor::current_speech_audio()` returns `Some((item_id, audio))`
while Speaking. `spawn_vad_task` keeps a `total_samples` counter; every
`defaults::buffer::PARTIAL_INTERVAL_MS` (500 ms) of audio, if Speaking
and no partial in flight (`partial_in_progress: AtomicBool`, one at a
time), snapshot and spawn `run_partial_transcription`. Results are
checked against `current_speech_item()` and dropped if stale.

## Error envelope (§10.5)

`emit_error(session, code, message, event_id?, param?)` writes
`{ "type": "error", "error": { "code", "message", "event_id"?,
"param"? } }`. Every `code` comes from `src/errors.rs`'s `code`
module; a literal string in `session.rs` would trip
`errors::debug_assert_known_code` in debug / `cargo test`.

| Code | Class | Trigger |
|---|---|---|
| `invalid_request_error` | 4xx | malformed JSON, missing/invalid fields |
| `unknown_event_type` | 4xx | inbound `type` not in §10.1 |
| `session_update_invalid` | 4xx | `session.update` shape rejected |
| `response_already_active` | 4xx | `response.create` while `RespPhase != None` |
| `response_cancel_not_active` | 4xx | `response.cancel` with no in-flight |
| `input_audio_buffer_commit_empty` | 4xx | manual commit on WebRTC, OR VAD commit below `MIN_SPEECH_MS` |
| `client_too_slow` | 4xx | outbound queue cap exceeded (§7.4) |
| `internal_state_error` | 5xx | invariant violation (§9.6, terminates session) |
| `vad_failed` | 5xx | VAD inference threshold exceeded (§4.4, terminates session) |

## Outbound pacing -- `audio_out.rs`

`OutboundPacer::write_sample` paces with `tokio::time::sleep_until` to
keep the receiver's jitter buffer happy. Wake target =
`start + (frames_written + 1) * FRAME_MS` -- deadline-based, not
fixed-delay, so a slow encode doesn't compound slip into later frames.

### played_ms post-abort snapshot (§7.3, RFC C.7)

`OutboundPacer::played_ms` is an `Arc<AtomicU64>`.

- **Bump after `write_sample` returns, not before** -- an abort landing
  mid-frame must not count that frame as drained.
- **`Ordering::Release` on the writer; `Acquire` at the cancel snapshot
  site.** `cancel_current_response` reads `played_ms` inside the same
  critical section that takes `current_response` and sets
  `resp = None`; `JoinHandle::abort()` + `await` establishes
  happens-before, so the snapshot taken under the session mutex
  reflects exactly what shipped. Reading after the abort outside the
  lock would race the pacer.

### Backpressure (§7.4)

`OutboundPacer` carries a `QueueGate` tracking `queued_ms` (at the 5 s
cap that's ~120 KB of pending audio). `try_push(chunk_ms)` at the head
of `play()` returns `QueueFull` without enqueuing when
`queued_ms + chunk_ms > queue_cap_ms`; `on_frame_sent()` at the tail of
`write_encoded_frame` decrements by `FRAME_MS` (saturating). Cap from
`OUTBOUND_QUEUE_CAP_MS` (default `defaults::wire::OUTBOUND_QUEUE_CAP_MS`).

On `QueueFull`, `synthesize_and_play` surfaces it and `run_response`
calls `handle_client_too_slow`: emit `OutboundQueueExceeded` to the
inspector, emit `error { code: "client_too_slow" }` on the data
channel, abort the LLM upstream, emit `response.done(failed,
status_details.reason="client_too_slow")`.

## Failure-mode handling

- LLM upstream 5xx / unreachable / empty stream ->
  `response.done(failed, reason="llm_error")`, session stays alive.
- LLM upstream slow -> `complete_stream_messages` blocks the
  per-utterance task; the next utterance still gets STT in parallel
  because `transcribe` runs on `spawn_blocking`.
- Kokoro synth panic / failure -> `response.done(failed,
  reason="tts_error")`.
- Malformed SDP offer -> HTTP 400 from `realtime_post`.
- Whisper silence hallucinations -> pre-gated at peak amplitude <
  `defaults::stt::SILENCE_PEAK_THRESHOLD = 0.005` in
  `WhisperHandle::transcribe`.

### Failure -> terminate (§4.4, §9.6)

- **§9.6 invariant violation** -- `state::check_state` returning `Err`
  emits `InvariantViolation` and spawns `terminate_with_error` with
  `INTERNAL_STATE_ERROR`. Detection runs synchronously under the mutex;
  termination is spawned so `pc.close().await` never runs with the lock
  held.
- **§4.4 VAD failure** -- `vad_supervisor_step` counts consecutive ONNX
  errors; at `defaults::vad::FAILURE_THRESHOLD = 3` it emits
  `VadEvent::Failed { reason }`; the handler emits `VadFailed` then
  `terminate_with_error` with `VAD_FAILED`. `vad_supervisor_step` is a
  pure function generic over a `VadInfer` trait so unit tests drive it
  with a deterministic mock.

## Subtle ONNX/model contracts

### Silero VAD v5 -- `src/vad/`

The frame contract (64-sample context window prepended to each
512-sample window from the previous chunk's tail; omitting it yields
the idle baseline ~0.0005 regardless of input) is stated in
`docs/book/07-speech-stack.md` §Voice activity detection. Specific to
this implementation:

- `sr` is a rank-0 (scalar) tensor, not a 1-element 1-D tensor: pass
  `((), [16000i64])` to `Tensor::from_array`.
- `state` is `[2, batch=1, 128]` f32, persisted across calls within an
  utterance and zeroed on speech end.
- Threshold/timing constants live in `defaults::turn_detection`
  (threshold 0.5, prefix padding 300 ms, silence duration 350 ms);
  test paths fall back to `vad::SPEECH_THRESHOLD`, which mirrors the
  default.

### Kokoro 82M v1.0 ONNX -- `src/tts/`

Named inputs `tokens` (int64 `[1, n+2]`), `style` (f32 `[1, 256]`),
`speed` (f32 `[1]`); output `audio` (f32 `[audio_length]`) at
`defaults::audio::TTS_SAMPLE_RATE = 24_000`.

- Token list MUST be padded with leading and trailing `0` (`pad`):
  `[0, ...tokens..., 0]` -- unpadded input produces garbage at clip
  edges.
- `style` is NOT the whole voice pack. The pack is `[510, 1, 256]`;
  pick `voice_pack[n]` where `n` is the un-padded token count.
- `clean_phonemes` (`vocab.rs`) post-phonemize rune substitutions:
  `r -> ɹ`, `ʲ -> j`, `x -> k`, `ɬ -> l`, plus
  `kəkˈoːɹoʊ -> kˈoʊkəɹoʊ`.
- VOCAB index assignment: pad -> 0, punctuation -> 1..16, ASCII letters
  -> 17..68, then IPA -- symbol order matches Python's insertion
  sequence verbatim.

### Whisper STT -- `src/stt/`

Two backends in one binary, picked at startup via
`STT_BACKEND={whisper-cpp,ct2}` (default `whisper-cpp`):

| Backend | Crate | Format | macOS GPU | 2.6 s utterance, debug build |
|---|---|---|---|---|
| `whisper-cpp` | `whisper-rs` | GGML (`ggml-large-v3-turbo.bin`) | yes (Metal) | ~9 s |
| `ct2` | `ct2rs` | CT2 dir (`whisper-ct2/`) | no (CPU only) | ~30-40 s |

`ct2`: `Whisper::generate(samples, lang, timestamp, opts)` runs
mel-spec -> encode -> detect-lang -> decode -> tokenizer.decode
end-to-end and takes `&self` (no Mutex on the inference path);
`beam_size = 1` (greedy) for realtime. Model dir requires `model.bin`,
`config.json`, `tokenizer.json`, `vocabulary.json`,
`preprocessor_config.json`. Raw output may include `<|...|>` markers;
`strip_special_tokens` peels them.

`whisper-cpp`: the Metal feature pulls whisper.cpp's GGML/Metal
kernels; a `WhisperState` is created per call (concurrent decode
against a shared `WhisperContext`). Model search order:
`ggml-large-v3-turbo.bin` -> `ggml-large-v3.bin` -> `ggml-tiny.en.bin`
(first existing wins).

#### ort + ct2 protobuf clash (load-dynamic)

`ct2rs` pulls `sentencepiece-sys` (via `whisper` -> `all-tokenizers`),
which static-links protobuf-lite; `onnxruntime` 1.24 also embeds
protobuf. With both static-linked, `ort::Session::commit_from_file`
fails to parse the silero VAD model. Fix: `ort`'s `load-dynamic`
feature -- onnxruntime is dlopen'd from its own dylib, keeping its
protobuf private. `main.rs` auto-points `ORT_DYLIB_PATH` at
`~/.nix-profile/lib/libonnxruntime.dylib` when unset. Applies
regardless of `STT_BACKEND` -- sentencepiece is linked either way.

### ONNX session locking

`ort::session::Session::run` takes `&mut self`, so the VAD and Kokoro
sessions live behind `Mutex<Session>` shared via `Arc`. Lock duration
is sub-millisecond on M-series, so contention is invisible up to many
tens of concurrent realtime sessions.

## libespeak-ng FFI

`espeak_TextToPhonemes` takes `void**` (a moving cursor through the
input); `tts/phonemize_glue.c` (~50 LOC) hides that, looping over clauses
until the cursor is NULL and accumulating IPA into a caller buffer
with overflow detection. Engine state is process-global (espeak's API
leaves no choice), so calls go through a `Mutex<State>` that also
caches the last-set voice to skip redundant `SetVoiceByName`. We never
call `espeak_Terminate` -- teardown breaks subsequent calls in the same
process.

## NPZ reader -- `src/tts/npz.rs`

`voices.bin` is a `.npz` (ZIP of `.npy`), one array per voice. Each
`.npy`: magic `\x93NUMPY`, version, header length, header dict
(`{'descr': '<f4', 'fortran_order': False, 'shape': (510, 1, 256), }`),
then raw LE f32 data. Only `<f4` + `fortran_order: False` is supported
-- the only combination Kokoro ships.

## Build prerequisites -- `build.rs` + `.cargo/config.toml`

Builds on Nix-darwin. The Nix-wrapped `clang` strips system search
paths, so `build.rs` walks `/nix/store` for `libiconv`, `libopus`,
`libespeak-ng` and emits `cargo:rustc-link-search=native=...`.
`.cargo/config.toml`:

| Setting | Why |
|---|---|
| `CMAKE_POLICY_VERSION_MINIMUM=3.5` | `audiopus_sys` ships an old `cmake_minimum_required(2.x)` in bundled libopus; CMake 4 refuses without it. |
| `LIBRARY_PATH=~/.nix-profile/lib` | Sub-build scripts (ring, ureq, ort-sys, ...) link against libiconv. |
| `CFLAGS`/`CXXFLAGS`/`BINDGEN_EXTRA_CLANG_ARGS` `=-Wno-elaborated-enum-base` | macOS SDK headers use a C++11 forward-enum form; Nix-wrapped clang escalates it to an error on Accelerate-using deps (C, C++/CTranslate2, and bindgen respectively). |

Required Nix packages: `cmake`, `libiconv`, `espeak-ng`, `onnxruntime`
(`nix profile add nixpkgs#<pkg>`). The espeak data dir is found at
runtime via `ESPEAK_DATA_PATH`, defaulting to
`~/.nix-profile/share/espeak-ng-data`.

## Lifetime / cleanup

- Sessions live in a process-global
  `Mutex<HashMap<String, Arc<Session>>>`. `handle_offer` inserts after
  SDP negotiation; `pc.on_peer_connection_state_change` removes on
  `Closed | Failed | Disconnected`, dropping the last `Arc<Session>`
  and with it the `RTCPeerConnection` and its closures.
  `GET /health/sessions` returns `{"live_sessions": N}`.
- Both ORT sessions and the espeak engine live for the process lifetime
  via `OnceLock` -- load-once, share-everywhere.

## Allocation hot paths

- `AudioIngest::process` reuses `decode_workspace: Vec<i16>` (~23 KB)
  and `mono_48k_workspace: Vec<f32>` across opus packets -- at 50 pkt/s
  these would otherwise produce ~1.15 MB/s of garbage.
- `audio_out::play_through` streams 24 kHz f32 through the resampler in
  480-sample (20 ms) chunks, accumulates 48 kHz output in a small
  `carry: Vec<f32>`, emits one opus frame per >= 960 samples. Working
  set ~14 KB regardless of utterance length.
- VAD copies `self.state` per window because `Tensor::from_array`
  takes ownership; the new state from `outputs["stateN"]` is copied
  back. State is 1 KB, so this is in the noise.

## Differential testing harness

`client/test_e2e.py --record-trace PATH` and
`client/test_e2e_full.py --record-trace PATH` capture the inbound
config + every outbound data-channel event with relative `ts_ms` + a
final `result` block. `client/trace_diff.py A B` canonicalises
sess_/item_/resp_ UUIDs and diffs the outbound event sequences.
`conformance/lib/trace_invariants.py` runs W1-W8 + supplementary
assertions against the trace -- the canonical implementation, invoked
today by the Go conformance gate (the Rust bridge was culled, §15.6).

## OpenAI-compat surface (`/v1/models`, `/v1/audio/{transcriptions,speech}`)

These endpoints exist so the server sits behind LiteLLM or any OpenAI
SDK client without translation:

- `GET /v1/models` -- `src/oapi/models_handler.rs`. Returns
  `{"object":"list","data":[...]}` listing the loaded executors. Each
  entry: `id`, `object="model"`, `created`, `owned_by`, `language`
  (always present, `null` for VAD), `task`, plus per-model extras
  (Kokoro carries `sample_rate` + `voices`). Honours `?task=`. Whisper
  id derives from whatever loaded -- HF-shaped `user/repo` strings pass
  through with `owned_by=user`; local file paths get
  `owned_by=speaches-plus`. The handler reads
  `WhisperHandle::model_id()`, which stores the basename / dir name
  alongside the backend.
- `POST /v1/audio/transcriptions` -- `transcriptions_post` in
  `src/main.rs`, multipart upload, `text` and `json` response formats;
  `verbose_json` / `srt` / `vtt` are the open gap.
- `POST /v1/audio/speech` -- `src/tts/http.rs`. JSON body,
  Kokoro synthesis chunked through ffmpeg per format. Matches Python
  speaches except one deliberate divergence: out-of-range `speed`
  returns **400** with an OpenAI error envelope, where speaches
  validates inside the streaming generator and returns 200 + zero
  bytes -- we fail loudly so SDK callers see the mistake. The chunker,
  emoji/markdown stripper, and f32->s16 helpers live in
  `src/tts/{chunk,text}.rs`. SSE mode emits `speech.audio.delta` (base64
  s16le) then `speech.audio.done` (zero token usage).

## Wire shapes (`src/oapi/mod.rs`)

Every 4xx/5xx JSON response goes through one of two helpers:

- `oapi::openai_error(status, msg, type, param?, code?)` -- emits
  `{"error":{"message","type","param"?,"code"?}}`; `type` is one of
  `oapi::kind::INVALID_REQUEST` / `AUTH` / `NOT_FOUND` / `SERVER` /
  `SERVICE_UNAVAIL`. Used by `transcriptions_post`, `realtime_post`,
  `speech_http`, and any non-pydantic-shaped error path.
- `oapi::fastapi_validation_error(entries)` -- the FastAPI 422 shape
  `{"detail":[{"type","loc","msg","input"?},...]}`. Used by
  `speech_http` for missing required fields and `transcriptions_post`
  for missing `file`.

The module also defines `Model` / `ListModelsResponse` plus the
`WHISPER_LANGUAGES` and `KOKORO_LANGUAGES` constants for the listing
handler.

## Wire-event invariants (compile-time)

Encoded in `wire.rs` as types rather than runtime checks:

- `ResponsePayload.audio_end_ms` is non-optional `u64`. Per §8.5 / W4
  every `response.done` carries `audio_end_ms = played_ms` for all four
  statuses; the v2 implementations shipped `Option<u64>` and lost the
  field on `completed` (the `regress-v2-go-w4-completed` fixture
  catches it). Non-optional makes omission a compile error.
- `ResponseStatusReason` is a closed enum (`drain_cap` / `token_limit`
  / `llm_error` / `tts_error` / `client_too_slow` / `barge_in` /
  `client_cancelled`); a new reason requires a spec amendment AND an
  enum variant.
- All ID fields on `OutboundEvent` use the typed newtypes from
  `types.rs`, so swapping them at a callsite is a compile error.

## Concurrent fuzz harness (`fuzz.rs`)

Two coverage variants run side by side because production wraps
`SessionState` in `tokio::sync::Mutex` (`session.rs`), not
`std::sync::Mutex`:

- `concurrent_invariants_hold_8_workers_5000_ops` -- 8 OS threads x
  5_000 ops on one `std::sync::Mutex<SessionState>`; proves §2.4 holds
  under arbitrary interleavings.
- `concurrent_invariants_hold_tokio_8_tasks_5000_ops` -- 8 tokio tasks
  x 5_000 ops on `tokio::sync::Mutex<SessionState>`; catches a
  transition method re-acquiring the same lock (deadlock) under
  tokio's cancellable, non-poisoning semantics.
- `multi_seed_invariants_hold_6x6x1500` -- 6 seeds x 6 workers x 1_500
  ops for scheduler diversity (mirrors Go's `MultiSeed`).
- `install_replace_abort_pattern_no_lost_or_double_mutation` --
  exercises the install/replace/abort discipline used by
  `install_commit_timer` and `set_pending_bargein` so a refactor of the
  slot helpers can't silently lose a transition or double-fire.

Wire-trace coherence under concurrent emission, lock-graph cycles, and
real-timer races (commit vs hard-cap, bargein-delay vs cancel) are
covered elsewhere: the sequential fuzzer's emission-capture path,
`eou::tests::race_hard_cap_*` paused-time tests, and the runtime-level
bargein tests respectively.

## Source quirks

### NaN-encoded `Option<f32>` in `realtime/session.rs`

`SessionConfig.no_speech_prob_threshold_bits` and
`avg_logprob_threshold_bits` are `AtomicU32`. A NaN bit-pattern means
`None` (gate disabled); any other f32 is the active threshold. NaN is a
safe sentinel because comparisons against NaN always fail -- an
accidental load-as-f32 still does the right thing under the gate's
`>` / `<` checks. Rust analogue of the Go side's three-state pointer
encoding for the same Whisper noise-gate fields.

`TurnDetectionConfig.neg_threshold_bits == 0` is a different sentinel:
"auto" (= `max(threshold - 0.15, 0.01)`); any other value is an
explicit threshold in `[0, 1]`.

### Audio-in `Sender` drop on `Session::terminate`

The terminate path explicitly nils `audio_in_tx` and `ws_ingest` after
flipping to `Terminated`. Without dropping the `Sender`, the spawned
VAD task's `rx.recv()` never returns `None`; the task keeps its
captured `Arc<Session>` alive and leaks one task + one session per
disconnect.

### `WsAudioIngest` resampler carry state

In `realtime/audio_in_ws.rs`, `src_position` and `last_sample` carry
**across** `append` events so chunk boundaries don't click
mid-utterance. Fuzz-asserted invariants:

- Constructor and `ingest_b64` never panic -- `Err` on malformed input
  is fine; garbage formats (`"opus"`, `"wav"`, `"pcm32"`,
  `"../../etc/passwd"`, `"💀"`) are rejected at construction.
- Carry state stays finite -- no NaN/Inf accumulation after thousands
  of random `ingest_b64` calls.

`fuzz_extreme_payloads` runs a 1 MiB raw PCM16 chunk: no panic on
large-but-legal payloads, and the 24 kHz -> 16 kHz resample ratio
matches `in_samples * 16/24` within 4 samples.

### Sweep VAD constants -- `defaults.rs::vad_window`

Mirror Python's `silero_vad_v5.get_speech_timestamps` constants; must
stay in lockstep with the Go side (`go/internal/vad/constants.go`) and
the Python upstream:

- `MAX_VAD_WINDOW_SAMPLES = 3 s` (= Python's
  `MAX_VAD_WINDOW_SIZE_SAMPLES`).
- `MIN_SILENCE_AT_MAX_SPEECH_MS = 98` (= Python's
  `min_silence_samples_at_max_speech` at 16 kHz).
- `NEG_THRESHOLD_DELTA = 0.15`, `NEG_THRESHOLD_FLOOR = 0.01` implement
  Python's `neg_threshold = max(threshold - DELTA, FLOOR)` hysteresis
  default.
