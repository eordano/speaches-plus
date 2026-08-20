# speaches-plus Go server -- Implementation guide

Canonical reference for the Go `/v1/realtime` server. Source files
contain no comments; all rationale, invariants, configuration, and
design decisions live here.

The implementation tracks `docs/book/07.1-barge-turn-spec-rfc-v3.md`
(RFC v3); section numbers below cite that spec. All four normative EOU
modes (`vad`, `text`, `audio`, `fusion`) are wired; `heuristic` and
`integrated` are extensions (§6 / §16).

## 0. Documentation discipline

1. No comments in source code. Files under `internal/realtime/` and
   `internal/eou/` (excluding `*_test.go`) are comment-free.
   Identifiers carry the meaning; non-obvious WHYs go here, not inline.
2. All tunable constants live in one file per package:
   `internal/realtime/constants.go` and `internal/eou/constants.go` are
   the sole sources of truth for defaults, magic numbers, and rate
   constants. No source file outside those two may declare a numeric or
   string default.

Type-bound enums (sealed-sum discriminators, `iota`-based phase kinds,
`ServerEventType` / `ClientEventType` wire alphabets) live with their
type definitions in `state.go` and `events.go`. Those are not "tunables";
they are the type itself.

## 1. File map

```
internal/
  realtime/                  /v1/realtime server, state machine, wire protocol
    constants.go             ALL defaults, rate constants, magic numbers (§16.1)
    capabilities.go          §14.4 GET /v1/realtime/capabilities + format helpers
    handler.go               Server, Config, makeSessionConfig, HandleRealtime
    websocket.go             HandleRealtimeWS (GET -> WS upgrade), wsTransport
    session.go               sessionConfig, sessionPipeline, EOU integration, applySessionUpdate
    state.go                 phaseState, sealed sum types, transition methods, invariants I1..I11
    audio_in.go              opus -> PCM -> VAD; speech_start / speech_stop dispatch
    audio_out.go             outboundAudio: TTS -> Opus track + PlayedMs cursor (RTC mode)
    audio_out_ws.go          WS-mode outbound audio path (response.audio.delta frames)
    audio_store_registry.go  per-session inspector audio-store handle + lifecycle
    pipeline.go              runResponse (LLM -> TTS -> wire); drain phase; emit*
    eager.go                 speculative LLM dispatch
    client_events.go         §10.1 inbound dispatcher + audio.append + commit
    events.go                JSON DTOs + ClientEventType / ServerEventType enums
    framing.go               §10.4 fragmenting; backpressure; client_too_slow
    inspector.go             §14.1 inspector lane (bridges runtime to internal/inspect)
    inspect_routes.go        §14.1 GET /v1/inspect/{sessions,history,audio,WS} handlers
    trace.go                 §15.2/§15.4 CanonicalizeTrace + AssertTraceInvariants
    chunker.go               sentence chunker for TTS feeding
    sdp_filter.go            OPUS-only SDP filtering + extractOpusChannels
    vad_adapter.go           adapter from VAD to internal vadDecision
    ids.go                   IdSource interface; randomIDs; DeterministicIDs
    types.go                 SessionID/ItemID/ResponseID/Epoch/Millis newtypes
    integrated.go            §6.6-style integrated STT+EOU consumer
    tracing.go               OTel span helper
  eou/                       End-of-utterance gate
    constants.go             ALL defaults + heuristic scores + ONNX I/O names + chat-template tokens
    eou.go                   Kind, Verdict, Request, Model, SigmoidLerpK
    loader.go                Config, Eagerness, Load() factory
    heuristic.go             Text-side heuristic that ships with the binary
    onnx.go                  LiveKit turn-detector ONNX wrapper
    bpe.go                   GPT-2/Qwen byte-level BPE
    byte_map.go              256->rune mapping for byte-level BPE
    special_trie.go          Special-token longest-match split
    chat_template.go         Qwen chat-template renderer
    languages.go             Per-language threshold table
    integrated_fake.go       deterministic STT+EOU fake for integrated_test.go
  vad/
    silero.go                Silero v5 ONNX wrapper with 64-sample context
    constants.go             VAD tunables
  audio/
    resample.go              linear 48 -> 16 kHz mono downsampler
    polyphase.go             polyphase resampler used for outbound 24 -> 48 kHz
    g711.go                  μ-law / A-law codecs (WS audio formats)
    opus_enc.go              cgo libopus encoder (outbound RTP)
    wav.go                   WAV decoder for /v1/audio/transcriptions
    types.go                 PCM buffer types
  conversation/
    llm.go                   OpenAI-compatible chat HTTP client (SSE streaming)
  stt/
    stt.go                   Transcriber interface; backend dispatch
    http.go                  /v1/audio/transcriptions handler
    whisper_cgo.{c,go}       whisper.cpp binding
    ct2_cgo.{cc,go}          CTranslate2 alternate backend
    mel.go                   mel-spectrogram fallback
    bpe.go                   Whisper tokenizer pieces
    constants.go             STT tunables
  tts/
    tts.go                   Synthesizer interface
    kokoro.go                Kokoro ONNX inference + voice style picker
    npz.go                   voices.bin (.npz) parser
    phonemize_cgo.{c,go}     espeak-ng -> IPA (FFI)
    vocab.go                 Kokoro phoneme -> token-ID table
    http.go                  /v1/audio/speech handler
    constants.go             TTS tunables
  inspect/                   §14.1 diagnostic-relay package (separate from realtime/inspector.go)
    relay.go                 per-session relay (replay buffer, NDJSON, broadcast)
    registry.go              process-global session_id -> relay map
    audio_store.go           mic_in / tts_out raw streams + sidecar
    retention.go             cleanup_on_startup (count/bytes/days)
    emit.go                  typed emit helpers (eou.*, bargein.*, state.transition...)
    json_helpers.go / types.go / util.go / constants.go
cmd/server/main.go           CLI flags / env vars; wires realtime.Server
```

## 2. State machine (RFC v3 §2)

A session owns one `phaseState` (state.go) protected by a single mutex.
Every transition runs under the mutex and is followed by `checkInvariants`.

### 2.1 Sealed sum-type phases (§2.6)

```
SessionPhase   : Pending | Active | Terminated
VadPhase       : Silent | Speaking | Stopped
RespPhase      : None | Predicted | Created | Streaming | Drain | Finalized
InputBuffer    : Empty | Voiced | Stopped | Committed
```

Each is an interface with a closed set of variants and a `Kind()`
accessor (per RFC v3 §2.6). The "top" label
(idle/listen/process/generate/drain) is derived from the four phases by
`derivedTopName` -- never stored.

VAD speaking is tracked separately so the `Predicted` phase can coexist
with VAD `Speaking` (Predicted is invisible per I7).

`RespPredicted` is a speaches extension (eager LLM dispatch) that
satisfies I1, I7, and I9 and emits no wire events.

### 2.2 Invariants (`checkInvariants` in state.go)

- **I1** Mutual exclusion: `vad.speaking => resp ∉ {Created, Streaming, Drain}`. Predicted is exempt because it emits no wire event.
- **I2** Single-slot response: at most one Created/Streaming/Drain. A second `response.create` raises `response_already_active`.
- **I3** Buffer immutability: once a buffer is rotated (Committed) it accepts no further appends.
- **I4** Response terminality: every `response.created(R)` is paired with exactly one terminal `response.done(R)` (W1 on the wire).
- **I5** Epoch monotonicity: `respPhase.epoch` is incremented on every retire (cancel, complete, drain-cap, Predicted rollback).
- **I6** Buffer-conversation consistency: a `Voiced` user buffer with `item_id=A` does not coexist with `conv[A].status=in_progress`.
- **I7** Predicted invisibility: `Predicted => wireEmitted == false`.
- **I8** Rotation-at-commit: buffer rotation happens synchronously with commit, not on `Speaking -> Stopped`. Enables the §6.7 EOU re-entry path that reuses the same buffer's item_id and audio_start_ms.
- **I9** Predicted retirement: `Predicted -> None` increments epoch and aborts the runner; `Predicted -> Created` preserves id and epoch and reuses the runner.
- **I10** Session monotonicity: `sessionPhase` follows `Pending -> Active -> Terminated`; no other order, no rewind.
- **I11** Conversation/wire consistency: every accepted `conversation.item.create` is followed by exactly one `conversation.item.created` referencing the same id (W8 on the wire).

`debugInvariants = true` (constants.go); in tests where no `violationHook`
is installed, a violation panics so the property fuzzer surfaces them. In
the live pipeline, `violationHook` emits
`error{code: internal_state_error}` and `session.done{reason:
internal_state_error}` and terminates the transport (§9.6).

## 3. Audio buffer model (§3)

`inputBuffer` rotates synchronously with **commit**, not on
Speaking->Stopped. The Stopped buffer survives the commit_timer window so
that a `vad.speech_started` during the timer can re-enter the same
`item_id` and `audio_start_ms` (§6.6).

`phaseState.sealedBufs` retains rotated buffers keyed by `item_id`
(§3.4). FIFO-bounded by `sealed_buffer_retention_count` (default
`defaultSealedBufferRetention = 4`); evicted oldest-first when the cap
is exceeded. Entries are dropped on
`conversation.item.input_audio_transcription.completed` (§7.2) regardless.

## 4. VAD (§4)

Silero v5 (`internal/vad/silero.go`) follows the RFC §4.1 frame
contract (also `docs/book/07-speech-stack.md` § "Voice activity
detection"): 512-sample (32 ms) windows @ 16 kHz, a 64-sample context
window prepended from the previous chunk (without it the model emits
its idle baseline, ~5e-4, regardless of input), `[2,1,128]` `state`
persisted within an utterance and zeroed on speech-end.

Threshold defaults (constants.go): `defaultVADThreshold = 0.5`,
`defaultVADPrefixPaddingMs = 300`, `defaultVADSilenceDurationMs = 350`.
VAD failure terminates the session with `code=vad_failed` (5xx). When
Silero is unavailable, a watchdog falls back to RMS-floor silence
detection (`hasNonSilenceWith`, threshold `defaultNonSilenceThreshold
= 0.005`, tick `silenceWatchdogTickMs = 200`).

## 5. Partial transcription (§5)

`startPartialLoop` runs while VAD is Speaking; cadence
`PartialTickMs` (default `defaultPartialTickMs = 500`). Each tick
re-runs STT on the current buffer and emits
`input_audio_buffer.partial_transcription`. The loop is bounded to one
task per session and is restarted on §6.6 Stopped->Speaking re-entry.

## 6. End-of-utterance gate (§6)

### 6.1 Modes

`eou.Kind ∈ {vad, heuristic, text, audio, fusion, integrated}`.

| Kind         | Source                                                         | Notes                                              |
|--------------|----------------------------------------------------------------|----------------------------------------------------|
| `vad`        | (none -- silence-only, score=1.0)                               | RFC v3 spec default.                               |
| `heuristic`  | `internal/eou/heuristic.go`                                    | Rule-based partial-transcript scorer. Extension.   |
| `text`       | `internal/eou/onnx.go` (LiveKit Qwen2.5)                       | RFC v3 §6.2.1.                                     |
| `audio`      | `internal/eou/audio.go` (`AudioONNXModel`) running smart-turn-v3 ONNX over an 8 s @ 16 kHz Whisper-Tiny log-mel `[1, 80, 800]`. Loaded from `--eou-audio-model` / `EOU_AUDIO_MODEL_PATH`; falls back to heuristic if unset/missing. | RFC v3 §6.2.2.                                     |
| `fusion`     | `runEOUFusion` -- parallel text+audio with §6.2.3 combiner      | RFC v3 §6.2.3. Rule via `eou.fusion_rule`.         |
| `integrated` | `internal/eou/integrated_fake.go` + custom impl                | Speaches extension (§16).                          |

The default is `KindVad` (matches RFC v3).

### 6.1.1 Fusion combiner (§6.2.3)

`runEOUFusion` (session.go) dispatches the text and audio heads in
parallel goroutines and combines their probabilities via
`eou.FuseScores(rule, pText, pAudio, weightText)`:

| Rule (`eou.fusion_rule`) | Formula                                  |
|--------------------------|------------------------------------------|
| `noisy_or`               | `1 - (1 - p_text)(1 - p_audio)`          |
| `max`                    | `max(p_text, p_audio)`                   |
| `mean`                   | `(p_text + p_audio) / 2`                 |
| `weighted`               | `w·p_text + (1-w)·p_audio`, `w = fusion_weight_text` |
| `gated` (default)        | `g·p_audio + (1-g)·p_text`, `g = σ(W·x)` (§6.1.1.1, Chen et al. 2022) |

Per §6.5, a head returning timeout / error / NaN / Inf / out-of-`[0, 1]`
is treated as `p = 1`; the surviving head's score becomes the verdict.
The hard-cap parallel timer races both heads' completion; if either head
times out beyond `silence_hard_cap_ms`, fusion returns
`phase = "during_eou"` and commits at the cap.

The combined score then flows through `SigmoidLerpK` (§6.4) like any
single-head verdict.

#### 6.1.1.1 Gated fusion (Chen et al. 2022)

The `gated` rule implements input-conditioned multimodal fusion after
Chen et al., "Gated Multimodal Fusion with Contrastive Learning for
Turn-Taking Prediction," ICASSP 2022 (arXiv:2204.10172). The gate is a
trained logistic projection over
`x = [1, p_text, p_audio, log1p(audio_ms/1000), log1p(partial_chars),
1[strong-terminator], 1[soft-terminator], 1[continuation-last-word]]`,
yielding `g = σ(W·x)`; the combined score is the `weighted` blend shape
with `g` recomputed per call. The paper gates over penultimate-layer
representations `[r_as, r_ts]`; that access isn't available through the
head ONNX surfaces, so the features proxy the representation (head
outputs plus structural cues). All-zero weights recover `weighted` at
`w = 0.5` exactly -- the paper's degenerate baseline.

Weights live in `eou.DefaultGatedFusionWeights`, fit on a 350-row
English sample of the public `pipecat-ai/smart-turn-data-v3-test`
dataset (labels hand-curated by the smart-turn-v3 authors; mostly
chirp3-synthesized audio plus a tail of human recordings). Pipeline
under `client/`:

1. `client/gated_fusion/extract_features.py` pulls a parquet shard via
   `huggingface_hub`; per row: decode audio -> faster-whisper
   transcript -> smart-turn-v3 `p_audio` -> heuristic `p_text` ->
   JSONL feature row.
2. `client/gated_fusion/train.py` fits a logistic regression on the
   *combined* score (BCE on `r = g·p_audio + (1-g)·p_text`, NOT on `g`
   -- chain-rule derivation in the script header). 5-fold
   cross-validation held-out accuracy is the honest number. Output is a
   Go + Rust + Python literal so one re-train pastes into all three.
3. `client/gated_fusion/gate_probe.py` emits the canonical
   (input, score) tuples seeding the Go and Rust parity tests; run it
   whenever weights change.

Same-set baselines on the 350-row corpus: audio-only 93.1%, text-only
52.0%, `noisy_or` 52.0%, `weighted` (w=0.5) 90.9%, `max` 52.0%,
`gated` 93.1% (xval 93.1% +/- 1.7%). Smart-turn-v3 was trained on this
dataset, so its accuracy is the realistic upper bound; the gate learns
to defer to it when audio is in-distribution. The gradient on the
combined score does this naturally: when heads disagree
(`pa = 0.05`, `pt = 0.95`), pushing `g -> 1` makes `r -> 0.05` and
matches the label; when they agree any blend works.

Re-train (`uv run --script client/gated_fusion/train.py`) whenever the
heuristic rules, the audio model file, or the feature set changes --
the gate is a function of the full upstream pipeline.

Other scaffolding scripts (also under `client/`, all Python):

| Script                              | Replaces (was)                |
|-------------------------------------|-------------------------------|
| `client/eou_fixture.py`             | `go/cmd/eou-fixture`          |
| `client/eou_matrix.py`              | `go/cmd/eou-runner` + `client/eou_inspector_matrix_rust.py` |
| `client/eou_divergence_probe.py`    | `go/cmd/eou-divergence-probe` |
| `client/gated_fusion/gate_probe.py` | `go/cmd/gate-probe`           |
| `client/gated_fusion/train.py`      | `go/cmd/train-gated-fusion-real` |

`client/eou_lib/` (smart_turn / heuristic / features / gate) is the
single Python source of truth for the EOU math; both the trainer and
the probes share it.

### 6.2 Heuristic scores

Values in §16.1 (`eou/constants.go`). Classes: `Empty` = empty /
whitespace-only; `StrongTerminator` = `.!?` and CJK analogues;
`SoftTerminator` = `,;:` and CJK analogues; `Hesitation` = "um"/"uh"
per language; `Continuation` = "and"/"the" per language; `Default` =
anything else. Per-language tables (`hesitationsByLang`,
`continuationsByLang`) for en/es/fr/de/it/pt; ja/zh/ko fall back to en.

### 6.3 ONNX path

`internal/eou/onnx.go` wraps LiveKit's `turn-detector` (Qwen2.5-0.5B
distil, INT8 ONNX). Pre-tokenization by GPT-2 byte-level BPE (`bpe.go`),
special-token longest-match split (`special_trie.go`), Qwen chat-template
framing (`chat_template.go`), softmax over the last-token logits row.
`MaxContextTokens` (default `defaultMaxContextTokens = 1024`) truncates
rightmost.

ONNX I/O names live in `eou/constants.go`:
- `onnxInputIDs = "input_ids"`
- `onnxAttentionMask = "attention_mask"`
- `onnxOutputLogits = "logits"`

Chat-template special tokens:
- `ImStart = "<|im_start|>"`
- `ImEnd   = "<|im_end|>"`

### 6.4 Hard-cap parallel timer (RFC v3 §6.3, mandatory pattern)

The hard cap is a deadline derived **once** at `vad.speech_stopped` and
observed at **two** points:

1. **During EOU compute** -- `runEOU` races the inference goroutine
   against `hardCapDeadline`. If the cap fires first, returns
   `verdict.phase = "during_eou"` and emits inspector
   `eou.hard_cap_fired{phase:"during_eou", score:null}`.
2. **During verdict-derived sleep** -- `runPartialAndScheduleCommit`
   builds two timers (`commitTimer` for the verdict delay,
   `commitHardCapTimer` for the cap) and races them via a shared
   `sync.Once`-guarded `commitFire`. Whichever fires first commits;
   the cap fire emits `eou.hard_cap_fired{phase:"during_wait", score:p}`.

### 6.5 Failure policy (RFC v3 §6.5)

Any uncertainty source -- inference timeout, inference error, garbage
probability (NaN/Inf/oob), tokenizer failure -- yields the configured
failure verdict, default `p=1`, `delay=min_delay`. A single failing
fusion head gets `p=1` and the surviving head's score becomes the
verdict (§6.2.3). Tunable per session via `eou.failure_p_default`
(default 1.0) and `eou.failure_delay` (default `"min"`); operators
preferring slow-commit set both to the opposite end (p=0, delay=max).

### 6.6 Cancellability

`vad.speech_started` during commit_timer:

1. `cancelCommitTimer()` cancels both the delay timer and the parallel
   hard-cap timer; the in-flight partial-STT work is cancelled too.
2. State machine reuses `item_id` and `audio_start_ms` (§3.2 + I8).
3. Any eager `Predicted` runner is rolled back via
   `rollbackPredictedIfAny("speech_resumed")`.
4. `startPartialLoop(itemID)` restarts the periodic STT.

### 6.7 Backchannel filter (RFC v3 §7.1 + §7.2)

`min_speech_for_response_ms` (default `defaultMinSpeechForResponseMs =
600`) is the auto-response gate. Buffers shorter than this **commit**
(for the conversation record) but suppress `create_response`. Buffers
shorter than `min_speech_ms` (default `defaultMinSpeechMs = 100`) are
rejected with `input_audio_buffer_commit_empty`. Telemetry on the
backchannel path: `inspector.backchannel_suppressed`.

### 6.8 Eagerness profiles (extension, RFC-neutral)

`eou.Eagerness ∈ {low, medium, high, auto}` overwrites `p_threshold /
min_delay_ms / max_delay_ms` (values: §16.1 Eagerness block; `auto` is
per-language).

## 7. Commit, transcription, played_ms, backpressure (§7)

### 7.1 Commit (`fireCommitTimer` in session.go)

1. Reject if `audio_ms < MinSpeechMs` -> `input_audio_buffer_commit_empty`.
2. `phase.onCommitTimerFire` rotates buffer to Committed and creates
   the user item.
3. Emit `input_audio_buffer.committed` + `conversation.item.created`.
4. If `< MinSpeechForResponseMs`, set `suppressResponse=true`.
5. Dispatch transcription (`runFromPartial` if cached, else
   `runTranscription`).

### 7.2 Transcription

On STT completion: emit
`conversation.item.input_audio_transcription.completed`, mark item
`completed`. If `autoResp` is true, call `promotePredicted` first; if
no eager runner is in flight, fall through to `runResponse`.

On STT failure: emit
`conversation.item.input_audio_transcription.failed`, mark item
`incomplete`, no response auto-created.

### 7.3 played_ms (RFC v3 §7.3 / Appendix C.7)

`outboundAudio.playedMs` is an `atomic.Int64` incremented **after**
`Track.WriteSample` returns OK. Synthesis time is too early; SCTP-acks
are too late. The barge-in path takes the snapshot via a callable
under the same lock that decides to cancel (TOCTOU-free).

### 7.4 Drain (§8.3)

`drainResponse` waits for `playedMs >= plannedMs` bounded by:
```
drain_cap = clamp(2 * plannedMs, drainCapFloorMs, drainCapCeilingMs)
```

Defaults: `defaultDrainCapFloorMs = 5000`, `defaultDrainCapCeilingMs = 60000`.
Polled every `drainPollIntervalMs = 20` ms.

### 7.5 Backpressure (§7.4)

Two layers:

- **Audio frames**: `Track.WriteSample` blocks naturally; encoder
  throttles. `playedMs` continues to reflect successful writes.
- **Non-audio events**: `sendFragmentedWith` checks
  `DataChannel.BufferedAmount() > OutboundBufferLimit`
  (default `defaultOutboundBufferLimit = 1 MiB`) and returns
  `errClientTooSlow`. `safeSend` catches that, emits
  `error{code:"client_too_slow"}`, terminates the session, closes
  transport (4xx semantics in error type, but session terminal).

## 8. Response lifecycle (§8)

### 8.1 Per-output streaming bracket

Bracket order per RFC §8.2: `output_item.added`, `content_part.added`,
`audio_transcript.delta*` / `audio.delta*`, then the matching `.done`s
in reverse nesting order, ending in `response.done`.
`response.audio.delta` carries base64 PCM16 LE on the data channel
regardless of transport (§10.2 -- required so text-only inspectors see
a complete record).

### 8.2 Terminal events (RFC v3 §8.5 / W4)

`response.done` carries:

```
{ id, object, status, status_details?, audio_end_ms }
status        ∈ completed | cancelled | incomplete | failed
status_details.reason ∈ drain_cap | token_limit | llm_error | tts_error
audio_end_ms  REQUIRED on ALL four statuses (no omitempty)
```

W4 is the regression gate that v2 failed: the field is on the response
DTO without `omitempty`, and `AssertTraceInvariants` checks every
status for it.

## 9. Barge-in (§9)

### 9.1 Atomicity (§9.1 + Appendix C.7)

`phaseState.onVadSpeechStart` runs under the session mutex and returns
a `bargeInEffect` (cancel snapshot + cancelTimer flag). The caller
emits without re-entering the lock. The `playedMs` snapshot is taken
**inside** the same critical section via a snapshot callable, never
read after lock release.

### 9.2 Delay

When `barge_in_delay_ms > 0` and a response is active,
`armBargeInTask(itemID, startMs)` runs the cancel through a goroutine
that sleeps the delay. Inspector emits `bargein.pending` when armed.

### 9.3 Commitment + assistant_truncated (RFC v3 §9.3 / W7)

On `handleBargeIn`:

1. `played_ms` is snapshotted from the outbound pacer.
2. Emit `response.done{status:cancelled, audio_end_ms}`.
3. **If `played_ms > 0`**, truncate the assistant `conversation.item`
   server-side and emit
   `conversation.item.assistant_truncated{audio_end_ms, transcript}`
   -- distinct from the client-issued `conversation.item.truncate` /
   server-echoed `conversation.item.truncated`.
4. Emit the buffered `input_audio_buffer.speech_started`.

Predicted barge-ins skip step 3 (no audio left the wire -- I7).

### 9.4 Suppression

`cancelBargeInTask` is called on `vad.speech_stopped`. If the task is
still sleeping, it is cancelled -- no client event, response
unaffected. Inspector emits `bargein.suppressed`.

## 10. Wire protocol (§10)

### 10.1 Inbound dispatcher (client_events.go)

`validClientEvents` whitelist: `session.update`,
`input_audio_buffer.{append,commit,clear}`,
`conversation.item.{create,delete,truncate}`,
`response.{create,cancel}`. Anything else -> `unknown_event_type`.
Pre-Active (no `session.created`) -> `session_not_active`.

### 10.2 Wire invariants W1..W8 (`AssertTraceInvariants` in trace.go)

W1-W8 as defined in RFC v3 §10.3. Go asserts W1, W2, W3, W4, W6, W7,
W8; W5 (partial_transcription / speech_stopped ordering) is guidance,
not asserted; W9-W12 are not checked. The alias event name used on the
wire is `conversation.item.created` (RFC §0.3).

### 10.3 Framing (§10.4)

`sendFragmentedWith(ch, event, eventID, fragMax, bufferLimit)`
implements the RFC §10.4 `full_message` / `partial_message` envelopes
(contiguous, ordered fragments). `envelopeBudget = 100` accounts for
the wrapper JSON keys. WebSocket sends raw JSON text frames, no
fragmenting; WS write timeout `wsWriteTimeoutSec = 5`.

### 10.4 Reserved error codes (`errorTypeFor` in pipeline.go)

Code semantics per the RFC §10.5 registry. Go maps the 4xx codes
(`invalid_request_error`, `unknown_event_type`, `session_not_active`,
`session_update_invalid`, `response_already_active`,
`response_cancel_not_active`, `input_audio_buffer_commit_empty`,
`client_too_slow`) to type `invalid_request_error`, and the 5xx codes
(`internal_state_error`, `vad_failed`, `stt_failed`,
`model_load_failed`) to type `server_error`. The RFC's
`eou_kind_unsupported` / `eou_fusion_rule_unsupported` are not in this
map. 5xx codes terminate the session after the error is flushed.

## 11. Session configuration (§11)

`session.created` carries the initial Session object. `session.update`
goes through `applySessionUpdate` (session.go), which implements RFC v3
§11.2.1's mandatory parse-validate-commit pattern: Phase 1 validates
every field into locals without touching `p.session` (any range/enum
failure returns immediately); Phase 2 writes and cannot fail. Partial
application (the v2 anti-pattern, App. D.5) is forbidden.

### 11.1 Validated session-mutable fields

| Field                                       | Range / values                  | Default constant                |
|---------------------------------------------|---------------------------------|---------------------------------|
| `instructions`                              | non-empty string                | (empty)                         |
| `voice`                                     | string                          | `defaultVoice = "af_heart"`     |
| `turn_detection.type`                       | `server_vad` \| `none`          | `defaultTurnDetectionType`      |
| `turn_detection.threshold`                  | `[0, 1]`                        | `defaultVADThreshold = 0.5`     |
| `turn_detection.prefix_padding_ms`          | `[0, 1000]`                     | `defaultVADPrefixPaddingMs = 300` |
| `turn_detection.silence_duration_ms`        | `[50, 5000]`                    | `defaultVADSilenceDurationMs = 350` |
| `turn_detection.barge_in_delay_ms`          | `[0, 1000]`                     | 0                               |
| `eou.kind`                                  | enum                            | `KindVad`                       |
| `eou.min_delay_ms`                          | `>= 0`                           | `defaultEOUMinDelayMs = 500`    |
| `eou.max_delay_ms`                          | `>= 0`                           | `defaultEOUMaxDelayMs = 3000`   |
| `eou.curve_k`                               | `(0, 30]`                       | `defaultEOUCurveK = 12.0`       |
| `eou.silence_hard_cap_ms`                   | `[0, 60000]`                    | `defaultHardCapMs = 5000`       |
| `eou.inference_timeout_ms`                  | `[0, 10000]`                    | `defaultInferenceTimeoutMs = 100` |
| `eou.context_turns`                         | `[0, 64]`                       | `defaultEOUContextTurns = 4`    |
| `eou.p_threshold`                           | `[0, 1]`                        | `defaultPThreshold = 0.5`       |
| `eou.failure_p_default`                     | `0.0` \| `1.0`                  | `defaultEOUFailureP = 1.0`      |
| `eou.failure_delay`                         | `min` \| `max`                  | `defaultEOUFailureDelay = "min"` |
| `eou.fusion_rule`                           | `noisy_or` \| `max` \| `mean` \| `weighted` \| `gated` | `defaultFusionRule = "gated"` |
| `eou.fusion_weight_text`                    | `[0, 1]`                        | `defaultFusionWeightText = 0.5` |
| `session_max_duration_s`                    | `[1, 86400]`                    | `defaultSessionMaxDurSec = 1800` |
| `min_speech_ms`                             | `[0, 60000]`                    | `defaultMinSpeechMs = 100`      |
| `min_speech_for_response_ms`                | `[0, 60000]`                    | `defaultMinSpeechForResponseMs = 600` |
| `sealed_buffer_retention_count`             | `[0, 1024]`                     | `defaultSealedBufferRetention = 4` |
| `input_audio_format`                        | enum from §14.4 capabilities    | `defaultInputAudioFormat = "pcm16"` |
| `output_audio_format`                       | enum from §14.4 capabilities    | `defaultOutputAudioFormat = "pcm16"` |

### 11.2 Process-scoped fields (rejected via `session.update`)

Per RFC v3 §17.8 / §17.9. The `sessionUpdateBody.UnmarshalJSON` collects
any of these top-level keys into `body.ProcessScoped`; Phase 1
validation rejects with `session_update_invalid` and `param: <field>`.

```
vad_model
session_max_duration_hard_cap_s
chat_completion_base_url
chat_completion_api_key
default_realtime_model
default_realtime_stt_model
default_realtime_partial_stt_model
default_speech_model
default_voice
gpu_mem_limit_bytes
```

### 11.3 Hard timeout

`SessionMaxDurSec` (default 1800 s) fires
`session.done(reason="max_duration")` then closes the transport. Clients
MAY shorten via `session.update`; the new value applies at the next
session start.

## 12. Transport (§12)

### 12.1 WebRTC

`POST /v1/realtime` with `Content-Type: application/sdp`. Body is the
client offer; response is the server answer.

`filterOpusOnly` (sdp_filter.go) strips non-OPUS codecs.
`extractOpusChannels` parses the offer's first OPUS rtpmap line and
passes channel count to `newOutboundAudio(channels)`.

Audio ingress: opus -> s16 stereo -> mono 16 kHz f32.
Audio egress: f32 mono at TTS rate -> 48 kHz {mono | stereo} opus on a
single audio track. Egress rate is `rtpOutSampleRate = 48000`,
frame size `opusFrameMs = 20` (constants.go).

### 12.2 WebSocket

`GET /v1/realtime` with WebSocket upgrade. Same state machine and
event semantics as WebRTC; only audio carriage differs.

## 13. Identifiers (§13)

`internal/realtime/ids.go`:

- `IdSource` interface returns `Event/Item/Resp/Sess()` strings.
- `randomIDs` (default) emits `evt_/item_/resp_/sess_<24-char hex>`.
- `DeterministicIDs` is a counter-based generator for tests
  (deterministic-replay traces become byte-identical after canonicalization).
- `setDefaultIDs` is the test-only injection point.

## 14. Telemetry / observability (§14)

### 14.1 Inspector lane

The inspector lane (RFC v3 §14.1) is a HTTP/WebSocket transport,
structurally compatible with `speaches/src/speaches/inspect/` and the
Rust implementation's `realtime/inspect/`. File layout: §1 (the WAV
header helper and `.raw` + sidecar writer live in
`inspect/audio_store.go`).

HTTP endpoints (mounted in `cmd/server/main.go`):

| Path | Behavior |
|---|---|
| `GET /v1/inspect/sessions`                         | live `SessionMeta` list from registry |
| `GET /v1/inspect/sessions/history`                 | NDJSON files on disk, sorted by mtime desc |
| `GET /v1/inspect/sessions/history/{sid}`           | application/x-ndjson stream of one session |
| `GET /v1/inspect/sessions/{sid}/audio?channel=&from_ms=&to_ms=` | audio/wav slice -- live `AudioStore` for in-memory sessions, `.raw` + sidecar `offset_ms` for ended ones |
| `GET /v1/inspect/{sid}` / `GET /v1/inspect/{sid}/stream` | replay-buffer snapshot then live broadcast; falls back to NDJSON replay-from-disk + close when session is gone |

NDJSON is persisted under `INSPECT_SESSION_DIR` (default
`~/.cache/speaches/sessions`); raw audio captured per channel:

| Channel | Capture point | On-disk format |
|---|---|---|
| `mic_in`  | `audio_in.go` after VAD-side resample to 16 kHz | s16 LE 16 kHz `.audio_mic_in.raw` |
| `tts_out` | `audio_out.go` before opus encode at TTS rate    | s16 LE 24 kHz `.audio_tts_out.raw` |

Retention runs once at first-session creation
(`inspect.RunStartupCleanup`); tunable via
`INSPECT_RETENTION_{COUNT,BYTES,DAYS}` env vars and the `relay_cap`
constant.

Lane mapping (RFC v3 §14.1.2 / speaches `LaneId`):

| Go `Inspector` event | lane | kind |
|---|---|---|
| `vad.confirmed_start`        | `vad`        | `confirmed_start` |
| `vad.confirmed_stop`         | `vad`        | `stopped`         |
| `bargein.pending`            | `bargein`    | `bargein_pending` |
| `bargein.fired`              | `bargein`    | `bargein_fired`   |
| `bargein.suppressed`         | `bargein`    | `bargein_cancelled` |
| `pacer.played_ms`            | `tts_pacer`  | `played_ms`       |
| `eou.hard_cap_fired`         | `eou`        | `hard_cap_fired`  |
| `eou.failure`                | `eou`        | `failure`         |
| `eou.eager_dispatch`         | `eou`        | `eager_dispatch`  |
| `eou.predicted_overflow`     | `response`   | `predicted_overflow` |
| `eou.predicted_rollback`     | `response`   | `predicted_rollback` |
| `state.transition`           | `state`      | `transition`      |
| `error.invariant_violation`  | `error`      | `invariant_violation` |
| `backchannel.suppressed`     | `turn`       | `backchannel_suppressed` |

Plus `wire.in` / `wire.out` mirrors emitted by inbound/outbound JSON
event handlers carrying `event_type` and `bytes`. Errors with kinds in
`ERR_KINDS` (`error|raised|dropped|failed|...`) auto-mirror to the
`error` lane via `relay.publish`.

### 14.2 slog backend

The default `slogInspector` (in `inspector.go`) emits structured
records with key `inspector.<event>`.
`inspector.transitions_sample_rate` is honored stochastically
(rand-gated drop). The relay-backed sink and the slog sink fan out
via `multiInspector` so live debugging and durable logging coexist.

### 14.3 OTel

OTel spans live on session/turn/LLM/STT/TTS/EOU dispatch entry points
(`startSpan`, tracer name `tracerName = "speaches/realtime"`). The
inspector envelope carries an optional `span_id` slot when an OTel
trace is active.

### 14.4 Capabilities endpoint (§14.4)

`GET /v1/realtime/capabilities` returns the `rfc_version` and the
supported `eou_kinds`, `input_audio_formats`, `output_audio_formats`,
and `fusion_rules`. Implemented in `capabilities.go`.

### 14.5 OpenAI-compat endpoints (`/v1/models`, `/v1/audio/{transcriptions,speech}`)

Three endpoints exist primarily so the server can sit behind LiteLLM
or any OpenAI SDK client without translation:

- `GET /v1/models` -- `internal/oapi/models_handler.go`. Returns the
  speaches-shape list (`{"object":"list","data":[...]}`) of executors
  the server has loaded. Each entry carries `id`, `object="model"`,
  `created`, `owned_by`, `language` (always present, possibly null),
  `task` (`automatic-speech-recognition` / `text-to-speech` /
  `voice-activity-detection`), plus per-model extras (Kokoro adds
  `sample_rate` and `voices`). The `?task=` query param filters the
  list. Whisper id is derived from whatever was loaded -- HF-shaped
  strings (`user/repo`) pass through with `owned_by=user`; local
  paths get `owned_by=speaches-plus` and a basename id.
- `POST /v1/audio/transcriptions` -- `internal/stt/http.go`. Multipart
  upload, `text` and `json` response formats. `verbose_json`/`srt`/
  `vtt` are the open gap.
- `POST /v1/audio/speech` -- `internal/tts/http.go`. JSON body, Kokoro
  synthesis chunked through ffmpeg per format. Behaviour matches
  Python speaches except for one deliberate divergence: out-of-range
  `speed` returns **400** with an OpenAI error envelope; speaches
  validates inside the streaming generator and returns 200 + zero
  bytes (we fail loudly so SDK callers see the mistake). See
  `client/check_speech_endpoint.py --allow-python-quirk` for the
  matrix of cases.

### 14.6 Error envelopes (`internal/oapi/errors.go`)

Every 4xx/5xx response from this server is one of two JSON shapes:

- OpenAI envelope (`oapi.WriteError`): `{"error": {"message", "type",
  "param"?, "code"?}}` -- used for everything that isn't a per-field
  Pydantic-style validation. `type` is one of `oapi.TypeInvalidRequest`
  / `TypeAuthError` / `TypeNotFound` / `TypeServerError` /
  `TypeServiceUnavail`.
- FastAPI 422 (`oapi.WriteValidationError`): `{"detail":[{"type",
  "loc","msg","input"?}, ...]}` -- used for missing-required-field /
  enum-out-of-range / range-violation on JSON request bodies. Match
  speaches' Pydantic shape so the OpenAI Python SDK surfaces the
  field path under `BadRequestError.body['detail']`.

The handler in `internal/realtime/handler.go` maps known SDP-parse
errors to 400 + `code="sdp_invalid"`; unknown failures go to 500 +
`code="negotiate_failed"`. Inspect routes use `code="session_not_found"`
/ `"audio_not_found"` for 404s.

## 15. Testability (§15)

### 15.1 Differential traces (§15.4)

`CanonicalizeTrace(events)` (trace.go):

- Wall-clock (`ts_ms`, `created_at`) -> ordinal index.
- Random ids -> `<TYPE>_N`, preserving cross-event identity.
- Stochastic floats (`eou.score`, `vad.probability`, `score`, `p`)
  rounded to 3 decimals.
- Audio base64 -> `{audio_bytes: <length>}`.

`TraceDiff(a, b)` returns the first index where two canonical traces
diverge, or -1.

### 15.2 Wire-invariant library (§15.2)

`AssertTraceInvariants(trace) []string` (in `trace.go`) checks W1, W2,
W3, W4, W6, W7, W8 on a serialized trace and returns a slice of
human-readable violations. `internal/realtime/canonical_bridge_test.go`
also runs every conformance trace through the canonical Python
implementation under `<repo>/conformance/lib/trace_invariants.py` via
subprocess, so a Go assertion bug mirroring a Go emitter bug is
independently caught.

### 15.3 Conformance corpus (§15.6)

Fifteen wire-trace fixtures (001-015, `input.jsonl` + `expected.jsonl`)
live at `<repo>/conformance/fixtures/`, alongside the endpoint and
declarative manifest bands (020-071) described in
`conformance/README.md`:

| ID  | Name                                       | Pins                                                  |
|-----|--------------------------------------------|-------------------------------------------------------|
| 001 | clean-utterance                            | W1, W3, W6 -- full happy turn                          |
| 002 | barge-in-streaming                         | W4, W6, W7 -- barge-in mid-stream (also pins §15.3 "barge-in straddling drain") |
| 003 | eou-reentry                                | §6.6 re-entry preserves item_id (also pins §15.3 "mid-utterance pause" -- wire-equivalent at the phase layer) |
| 004 | backchannel                                | §7.1 + §7.2 -- short utterance commits, no auto-response (pins §15.3 "short utterance below response threshold") |
| 005 | manual-response-create                     | C1 dispatch + `instructions` override                 |
| 006 | session-update-atomic                      | §11.2.1 atomic update rejection + reflective echo     |
| 007 | per-status-audio-end-ms-completed          | §8.5 / W4 on `completed` (closes the v2 D.4 self-hiding bug) |
| 008 | per-status-audio-end-ms-cancelled          | §8.5 / W4 on `cancelled`                              |
| 009 | per-status-audio-end-ms-incomplete         | §8.5 / W4 on `incomplete` (drain_cap)                 |
| 010 | session-update-atomic-both-fields          | §11.2.1 / D.5 -- valid+invalid pair, atomic rejection  |
| 011 | silence-only-input                         | §15.3 -- markActive only; trace = session.created      |
| 012 | per-status-audio-end-ms-failed             | §8.5 / W4 on `failed` (closes the per-status set)     |
| 013 | session-update-invalid-no-speech-prob      | §11.2 + §17.4 -- out-of-range `no_speech_prob_threshold` rejected atomically |
| 014 | session-update-invalid-neg-threshold       | §11.2 -- `turn_detection.neg_threshold` outside `[0, 1]` rejected |
| 015 | session-update-invalid-min-speech-duration | §11.2 -- `min_speech_duration_ms` above the 60 s cap rejected |

The §15.3 scenarios that exercise real-timer behavior live as runtime
tests in §15.7 instead of wire-trace fixtures:

| Scenario                    | Test                                                                |
|-----------------------------|---------------------------------------------------------------------|
| hard-cap-eou-stall          | `TestRunEOU_HardCapFiresDuringEOU` + the two `RegressV2*` companions |
| hard-cap-low-eou-score      | `TestRunPartialAndScheduleCommit_HardCapFiresDuringWait`            |
| simultaneous-bargein (§9.5) | `TestArmBargeInTask_SimultaneousReplacesSlot` (`bargein_test.go`)   |

`go test ./internal/realtime/ -run TestConformanceCorpus` replays each
through a synthesizer and diffs against the canonical expected trace.
**A failing fixture is a release blocker.**

§15.7 runtime regression tests:

| Test                                          | Pins                                            |
|-----------------------------------------------|-------------------------------------------------|
| `TestRunEOU_HardCapFiresDuringEOU`            | §6.3 / D.2 -- clamp pattern would never fire     |
| `TestRunEOU_RegressV2RustNoHardCap`           | §15.7 alias of the above (Rust v2 postmortem)   |
| `TestRunEOU_RegressV2GoClampNotTimer`         | §15.7 alias (Go v2 postmortem)                  |
| `TestRunEOUFusion_NoisyOrCombines`            | §6.2.3 -- combiner formula                       |
| `TestRunEOUFusion_DegradesWhenAudioHeadFails` | §6.2.3 + §6.5 -- graceful degradation            |

### 15.4 Property fuzzer (§15.5)

`TestPhase_FuzzInvariantsHold` runs 5,000 random ops with seed=1.
`TestPhase_FuzzInvariantsHold_MultiSeed` runs 1,000 ops x 6 seeds.
`checkInvariants` runs after each step; failure aborts with seed +
step index for replay.

## 16. Configuration reference

### 16.1 Constants -- single source of truth

**Two files. No defaults anywhere else.**

#### `internal/realtime/constants.go`

| Group        | Constant                          | Default | Spec ref            |
|--------------|-----------------------------------|---------|---------------------|
| Audio rates  | `opusSampleRate`                  | 48000   | WebRTC standard     |
|              | `whisperSampleRate`               | 16000   | §3.0                |
|              | `maxOpusFrameInt16`               | 5760    | 120 ms @ 48 kHz     |
|              | `rtpOutSampleRate`                | 48000   | §12.4               |
|              | `opusFrameMs`                     | 20      | Standard frame      |
|              | `opusEncodeScratchCap`            | 1500    | MTU-bounded buf     |
| EOU          | `defaultEOUMinDelayMs`            | 500     | §6.4.2              |
|              | `defaultEOUMaxDelayMs`            | 3000    | §6.4.2              |
|              | `defaultHardCapMs`                | 5000    | §6.3                |
|              | `defaultEOUCurveK`                | 12.0    | §6.4.2              |
|              | `defaultEOUContextTurns`          | 4       | §6.4.2              |
|              | `defaultEOUFailureP`              | 1.0     | §6.5                |
|              | `defaultEOUFailureDelay`          | "min"   | §6.5                |
|              | `eouHistoryFallbackTurns`         | 6       | fallback when cfg=0 |
|              | `defaultEOUAudioWindowMs`         | 8000    | §6.2.2              |
| Buffer/STT   | `defaultMinSpeechMs`              | 100     | §7.1                |
|              | `defaultMinSpeechForResponseMs`   | 600     | §7.2 / §17.4        |
|              | `defaultPartialTickMs`            | 500     | §5                  |
|              | `defaultStartSpeechSamples`       | 800     | 50 ms @ 16 kHz      |
|              | `defaultSealedBufferRetention`    | 4       | §3.4                |
| Session/resp | `defaultSessionMaxDurSec`         | 1800    | §11.3               |
|              | `defaultLLMTimeoutSec`            | 60      | operational         |
|              | `defaultDrainCapFloorMs`          | 5000    | §8.3                |
|              | `defaultDrainCapCeilingMs`        | 60000   | §8.3                |
|              | `drainPollIntervalMs`             | 20      | §8.3 polling        |
| Wire         | `defaultOutboundQueueCap`         | 256     | §7.4                |
|              | `defaultDataChannelFragmentMax`   | 900     | §10.4               |
|              | `defaultOutboundBufferLimit`      | 1 MiB   | §7.4                |
|              | `envelopeBudget`                  | 100     | §10.4 wrapper       |
|              | `wsWriteTimeoutSec`               | 5       | WS write deadline   |
| VAD          | `defaultVADThreshold`             | 0.5     | §4.2                |
|              | `defaultVADSilenceDurationMs`     | 350     | §4.2                |
|              | `defaultVADPrefixPaddingMs`       | 300     | §4.2                |
|              | `defaultVADLessSilenceMs`         | 1500    | watchdog fallback   |
|              | `defaultNonSilenceThreshold`      | 0.005   | watchdog RMS floor  |
|              | `silenceWatchdogTickMs`           | 200     | watchdog cadence    |
|              | `defaultVADModel`                 | silero_v5 | §4.1              |
|              | `defaultTurnDetectionType`        | server_vad | §4.5             |
| Eager        | `defaultPredictedTokenBufferCap`  | 256     | extension           |
|              | `defaultEagerMaxInflight`         | 1       | extension           |
|              | `defaultEagerIntervalMs`          | 250     | extension           |
| Inspector    | `defaultInspectorTransitions`     | true    | §14.2               |
|              | `defaultInspectorSampleRate`      | 1.0     | §14.2               |
| Misc         | `sentenceChunkerMinChars`         | 120     | TTS chunk granularity |
|              | `defaultVoice`                    | af_heart | TTS default voice  |
|              | `tracerName`                      | speaches/realtime | OTel       |
|              | `debugInvariants`                 | true    | §2.4                |
|              | `defaultInferenceTimeoutMs`       | 100     | §6.4.2 (realtime mirror) |
| Capabilities | `capabilityRFCVersion`            | "v3"    | §14.4 endpoint payload |
|              | `defaultInputAudioFormat`         | pcm16   | §3.3 fallback       |
|              | `defaultOutputAudioFormat`        | pcm16   | §12.4 fallback      |
|              | `defaultFusionRule`               | gated    | §6.2.3             |
|              | `defaultFusionWeightText`         | 0.5     | §6.2.3 weighted     |
| Inspector WS | `inspectorBusBufferPerSub`        | 256     | §14.1 backpressure  |
|              | `inspectorWSWriteTimeout`         | 5       | §14.1 write deadline |

#### `internal/eou/constants.go`

| Group           | Constant                            | Default     |
|-----------------|-------------------------------------|-------------|
| Curve           | `DefaultCurveK`                     | 12.0        |
| Defaults        | `defaultMinDelayMs`                 | 500         |
|                 | `defaultMaxDelayMs`                 | 3000        |
|                 | `defaultHardCapMs`                  | 5000        |
|                 | `defaultInferenceTimeoutMs`         | 100         |
|                 | `defaultContextTurns`               | 4           |
|                 | `defaultEagerMaxInflight`           | 1           |
|                 | `defaultEagerIntervalMs`            | 250         |
|                 | `defaultMinSpeechForCommit`         | 600         |
|                 | `defaultMaxContextTokens`           | 1024        |
|                 | `defaultPThreshold`                 | 0.5         |
|                 | `defaultEotThreshold`               | 0.7         |
|                 | `defaultEagerEotThreshold`          | 0.5         |
|                 | `defaultAudioWindowMs`              | 8000        |
|                 | `defaultFusionRule`                 | gated       |
|                 | `defaultFusionWeightText`           | 0.5         |
| Heuristic       | `heuristicScoreEmpty`               | 0.5         |
|                 | `heuristicScoreStrongTerminator`    | 0.95        |
|                 | `heuristicScoreSoftTerminator`      | 0.65        |
|                 | `heuristicScoreHesitation`          | 0.10        |
|                 | `heuristicScoreContinuation`        | 0.25        |
|                 | `heuristicScoreDefault`             | 0.55        |
| Eagerness       | `eagernessLowPThreshold`            | 0.7         |
|                 | `eagernessLowMinDelayMs`            | 800         |
|                 | `eagernessLowMaxDelayMs`            | 3000        |
|                 | `eagernessMediumPThreshold`         | 0.5         |
|                 | `eagernessMediumMinDelayMs`         | 500         |
|                 | `eagernessMediumMaxDelayMs`         | 2500        |
|                 | `eagernessHighPThreshold`           | 0.4         |
|                 | `eagernessHighMinDelayMs`           | 300         |
|                 | `eagernessHighMaxDelayMs`           | 1500        |
| ONNX            | `onnxInputIDs`                      | "input_ids" |
|                 | `onnxAttentionMask`                 | "attention_mask" |
|                 | `onnxOutputLogits`                  | "logits"    |
| Chat template   | `ImStart`                           | `<|im_start|>` |
|                 | `ImEnd`                             | `<|im_end|>`   |

### 16.2 Type-bound enums (NOT in constants.go)

These live with their type because they ARE the type:

- `internal/realtime/state.go`: `sessKind*`, `vadKind*`, `bufKind*`,
  `respPhaseKind*`, `responseStatus*`, `itemStatus*`, `TerminationReason*`.
- `internal/realtime/events.go`: `ClientEventType` / `ServerEventType`
  wire alphabet enums.
- `internal/realtime/types.go`: `SampleRate` consts (`SR16k/24k/48k`).
- `internal/realtime/audio_in.go`: `vadDecision` (no defaults; just the
  three iota values).
- `internal/eou/eou.go`: `Kind` enum string consts (`KindVad`,
  `KindHeuristic`, `KindText`, `KindAudio`, `KindFusion`, `KindIntegrated`)
  and `FusionRule` enum (`FusionNoisyOr`, `FusionMax`, `FusionMean`,
  `FusionWeighted`).
- `internal/eou/loader.go`: `Eagerness` enum.

### 16.3 CLI / env wiring

`cmd/server/main.go` exposes flags / env vars for every constant in
§16.1 that operators tune at startup, threading them through
`realtime.Config` into `sessionConfig` via
`Server.makeSessionConfig`.

## 17. RFC v3 compliance status

| §                                            | Status   |
|----------------------------------------------|----------|
| §0.2 Deviation register (assistant_truncated, audio_end_ms-on-all, client_too_slow, eou.* namespace) | PASS |
| §2 Session model + I1..I11 invariants        | PASS     |
| §3 Buffer + sealed map                       | PASS     |
| §4 VAD                                       | PASS     |
| §5 Partial transcription                     | PASS     |
| §6 EOU (vad, text, audio, fusion mandatory; heuristic, integrated extension) | PASS |
| §6.2.3 Fusion combiner (noisy_or/max/mean/weighted/gated) | PASS |
| §6.3 Hard-cap parallel timer with `phase` discriminator | PASS  |
| §6.5 Failure policy fast-commit (configurable) | PASS  |
| §7 Commit / played_ms / backpressure         | PASS     |
| §8 Response lifecycle / Drain / W4 audio_end_ms on all statuses | PASS |
| §9 Barge-in / delay / suppression / W7 played_ms gate | PASS |
| §10 Wire protocol / W1..W8 / errors          | PASS     |
| §11 Session config / atomic update / hard timeout | PASS |
| §11.2.1 parse-validate-commit pattern        | PASS     |
| §12 WebRTC + WebSocket                       | PASS     |
| §13 IdSource                                 | PASS     |
| §14.1 Inspector lane (`/v1/inspect` WebSocket fan-out + slog) | PASS |
| §14.4 Capabilities endpoint (`GET /v1/realtime/capabilities`) | PASS |
| §15 Testability -- fuzzer + 12 conformance fixtures + §15.7 regression suite | PASS |
| §15.2 Canonical lib bridge via subprocess (`canonical_bridge_test.go`) | PASS |
| §15.3 Synthetic audio fixture corpus -- 12 wire-replay fixtures + 4 runtime tests | PASS |
| §17 Configuration reference                  | PASS     |

## 18. Build and run

CGo deps (Nix shells):

```
nix develop
go build -tags 'whisper_cgo opus_cgo kokoro_cgo' ./cmd/server
```

Tests:

```
nix develop
go test ./internal/...
go test -race ./internal/realtime/... ./internal/eou/...
```

Property fuzzer (multi-seed):

```
go test ./internal/realtime/ -run Fuzz -v
```

Conformance fixtures:

```
go test ./internal/realtime/ -run TestConformanceCorpus -v
```

Server CLI summary (subset; `--help` for the full list):

```
./bin/server \
  --addr :8000 \
  --stt-backend ct2 \
  --kokoro-model $KOKORO_MODEL --kokoro-voices $KOKORO_VOICES \
  --silero-vad $SILERO_VAD_MODEL \
  --chat-base $CHAT_COMPLETION_BASE_URL --chat-key $CHAT_COMPLETION_API_KEY \
  --eou-model $SPEACHES_EOU_MODEL \
  --eou-audio-model $EOU_AUDIO_MODEL_PATH \
  --eou-min-delay-ms 500 --eou-max-delay-ms 3000 \
  --min-speech-ms 100 --min-speech-for-response-ms 600 \
  --session-max-duration-s 1800
```

## 19. Source quirks

Per §0, source files are comment-free. The non-obvious WHYs that
otherwise wouldn't survive a re-read live here.

### 19.1 `phaseState.withLock` releases `s.mu` before firing hooks

In `internal/realtime/state.go`, `withLock` takes `s.mu`, runs the
caller's function, snapshots transitions + invariant violations, then
releases the lock **before** invoking `transitionHook` /
`violationHook`. Hooks call into the wire layer; data-channel writes are
synchronous and can block on a slow client. Holding `s.mu` across that
path would deadlock every other state transition while the SCTP buffer
drains. Hooks therefore always run lock-free.

### 19.2 `sessionPipeline.close` uses `sync.Once`

In `internal/realtime/session.go`, `close()` guards the
`close(p.closed)` call with `sync.Once`. A prior select-default pattern
(`select { case <-p.closed: default: close(p.closed) }`) was non-atomic
under concurrent callers: two goroutines could both observe the channel
unclosed, both fall into `default`, both call `close`, and panic with
"close of closed channel". `TestSessionPipelineCloseIsConcurrencySafe`
(64 goroutines x 200 trials, `-race`) is the regression gate.

### 19.3 Pointer-typed Whisper gate thresholds -- three-state shape

`sessionConfig.NoSpeechProbThreshold` and `AvgLogprobThreshold` are
`*float32`. `nil` means **gate disabled** (matches Python's `None`
default). `session.update` carries a parallel `*Null bool` per field
because Go's `encoding/json` collapses *absent* and *explicit `null`*
into the same `nil` pointer -- the bool restores the third state so a
client can disable a previously-enabled gate. The Rust side encodes the
same `Option<Option<f32>>` shape as NaN bits in `AtomicU32` (see Rust
§ "NaN-encoded Option<f32>").

`VADNegThreshold == 0` is also a sentinel: "auto", meaning
`max(VADThreshold - 0.15, 0.01)`. Any positive value is explicit.

### 19.4 Duration-aware Whisper noise gate -- `internal/stt/noise_gate.go`

Port of speaches' `_effective_avg_logprob_threshold` plus the
`nsp_fail || logprob_fail` decision in `realtime/input_audio_buffer.py`
lines 296-302. The threshold lerps linearly from the base value toward
`gateLooseFloor = -3.0` between `gateFullMs = 1500` and
`gateOffMs = 5000`; outside that window the base applies as-is (short
utterances) or the gate is off (long utterances). NSP is checked first
(Python's reject order). `nil`/NaN stats from the backend are "no
signal" -- that half of the gate degrades to "accept", so a backend
without `avg_logprob` doesn't make every transcript fail.

### 19.5 Whisper backend stat surfacing

`stt.Transcriber` is the minimal interface; `FullTranscriber` is the
optional extension that surfaces `avg_logprob` + `no_speech_prob`;
`SegmentTranscriber` additionally surfaces per-segment timings. The
realtime pipeline type-asserts on `FullTranscriber` and falls back to
plain `Transcribe` when the backend can't provide stats, so the
duration-aware gate degrades gracefully. The diarized-transcription
endpoint requires `SegmentTranscriber`.

The CGo glue (`internal/stt/whisper_cgo.go`,
`internal/stt/ct2_cgo.go`) emits NaN whenever the model couldn't
compute a stat; the Go side converts NaN -> `nil` pointer so the gate
treats it as "no signal" rather than rejecting on a bogus value. For
CT2, `avg_logprob` comes from `WhisperGenerationResult.scores[0]` --
the sequence-averaged log-probability of the top hypothesis, matching
faster-whisper's `segment.avg_logprob`.

`CT2.TranscribeSegments` reuses one decode pass with timestamps
enabled and parses `<|t.tt|>` markers in Go; sequence-level
`avg_logprob`/`no_speech_prob` are surfaced but per-segment stats stay
`nil` because CT2 only emits one stat per generation.

### 19.6 Diarized transcription -- `serveDiarized` in `internal/stt/http.go`

Single Whisper + diarizer pass; each Whisper segment is assigned to the
diarizer cluster whose time range contains its **midpoint** (with
inclusive end). Ties pick the first cluster; segments past the last
cluster fall to the nearest. When the diarizer is unavailable or the
input is too short for it to emit clusters, the response still carries
Whisper's per-segment text without speaker labels rather than failing.

The `segment` shape (with `type`/`id`/`start`/`end`/`text`/`speaker`)
matches OpenAI's `diarized_json` per
`docs/book/02.1-model-compat-matrix.md` § "diarized_json response shape". `type`
and `id` are the OpenAI-mandated keys client SDKs key off; the other
fields are project extensions that clients ignore if unknown.
`aggregateSegmentStats` averages per-segment `avg_logprob`/
`no_speech_prob` weighted by segment duration so a 50 ms aside doesn't
shift the verdict as much as a 1 s segment.

### 19.7 Silero VAD -- ring + LSTM state propagation trade-off

`internal/vad/silero.go` runs Silero v5 **incrementally** -- one forward
pass per 512-sample window -- and applies Python's
`silero_vad_v5.get_speech_timestamps` hysteresis state machine over a
trailing probability ring (3 s = `MaxProbRing ~ 94` entries). The
deviation from a literal port: Python resets the LSTM state for each
sweep; this driver lets state propagate across pushes within an
utterance. Empirically equivalent because the LSTM saturates within a
few windows and the state machine still examines the same window of
history. The trade-off is one forward per push vs ~94 in a per-push
full-sweep design.

`Process` returns at most one `Decision` (`None` / `SpeechStart` /
`SpeechEnd`) per call; frame indices in returned offsets are
window-aligned. The prob ring tracks frame indices in ring-local
coordinates; absolute offsets are reconstructed via `durationSpl`.

### 19.8 VAD constants provenance -- `internal/vad/constants.go`

`MaxVadWindowSamples` mirrors Python's
`MAX_VAD_WINDOW_SIZE_SAMPLES = 3 s`. `MinSilenceAtMaxSpeechMs = 98` is
Python's `min_silence_samples_at_max_speech` (98 ms @ 16 kHz).
`NegThresholdDelta = 0.15` + `NegThresholdFloor = 0.01` implement
Python's `neg_threshold = max(threshold - 0.15, 0.01)` hysteresis
default. Keep these in lockstep with the Rust `vad_window` constants
in `rust/src/defaults.rs` and the Python upstream.

## 20. Module selection and allocation discipline (from the spike analysis)

Module choices, with rationale:

- `net/http` + `github.com/go-chi/chi/v5`: stdlib plus minimal routing
  sugar.
- `github.com/pion/webrtc/v4`: production-tested outside browsers
  (Twitch; LiveKit is built on pion, proving Go realtime audio at
  scale). `OnTrack` hands a `*webrtc.TrackRemote` read via `ReadRTP()`
  in a tight loop, mirroring the aiortc flow the reference client was
  verified against; pion interop is less picky than aiortc.
- `github.com/yalue/onnxruntime_go`: community ORT wrapper. More
  verbose than Rust's `ort` (manual input/output tensor allocation),
  links against whatever `libonnxruntime.{so,dylib}` you ship, no
  first-party Microsoft endorsement -- but widely used. The cgo
  overhead is not the bottleneck: a 32 ms VAD window is ~200 us of ONNX
  compute vs ~50-200 ns per cgo round-trip.
- whisper.cpp via an own thin cgo shim
  (`internal/stt/whisper_cgo.{c,go}`), not upstream's Go binding: the
  binding lags the C++ API and the maintainer focuses on the C++ and
  Python paths; a ~150 LOC shim over `whisper.h` gives full control of
  parameter tuning and streaming hooks, both of which matter for
  realtime.
- `log/slog` for structured logging, `go.opentelemetry.io/otel` for
  tracing.
- Spike-plan modules the as-built tree dropped: `hraban/opus`
  (pion/opus pure-Go decode + cgo libopus encode instead),
  `gosamplerate` (hand-rolled linear + polyphase resamplers),
  `goccy/go-json` / `easyjson` (stdlib `encoding/json` suffices).

Allocation discipline (day-one requirements, not later optimizations):

- `runtime.Pinner` pins `[]float32` audio buffers handed to cgo (the
  whisper/ct2 shims): the Go slice goes to C without copying -- same
  memcopy count as Rust+`cxx`. Needs Go >= 1.22 (Pinner race fixes);
  go.mod pins 1.25.
- `sync.Pool` on the outbound emit hot path (audio_out.go). Without
  pooling, ~20 events/s x N sessions creates GC pressure visible as
  p99 latency variance; if the realtime test shows >50 ms GC pauses,
  suspect per-event JSON allocation in the data-channel sender first.
- Commit session audio buffers to the worst case up front
  (30 min @ 16 kHz ~= 115 MB/session) rather than doubling like
  speaches' `np.empty` growth -- fine for a server not running
  thousands of concurrent sessions.
- Deploy with `GOMEMLIMIT` sized to what is left after model memory so
  the GC paces against the limit instead of doubling on the way up;
  `-pgo=cpu.pprof` after collecting a profile is worth ~10% on hot
  paths.
- Expected performance is within ~5% of the Rust port -- the heavy
  lifting (whisper.cpp, ORT, opus) is cgo either way; the variable is
  GC pause variance.

Speaches files whose semantics are mirrored exactly:
`speaches/realtime/input_audio_buffer.py:34-37` (sample-rate constants)
and `_INITIAL_CAPACITY` (buffer growth),
`speaches/routers/realtime_rtc.py:90-121` (fragmentation rules,
`MAX_FRAGMENT_SIZE = 900`, base64 inner payload) and `:355-365`
(opus-only codec filter), `speaches/realtime/session.py:48-75` (default
Session shape), `speaches/types/realtime.py` (full event type list).
