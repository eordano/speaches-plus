# `realtime/` -- implementation notes

Python port of `speaches-plus/rust/src/realtime/`. Source files are
comment-free per project policy; "why" context lives here. The design
rationale (state machine, RFC section mapping, barge-in, EOU, pacing,
backpressure) is documented once in `speaches-plus/rust/IMPLEMENTATION.md`
-- this file records only the Python-specific deltas.

## Layout

| File | Role |
|---|---|
| `__init__.py` | defaults registry + capabilities + route registration |
| `state.py` | sealed sum-type phases, transitions, invariants I1..I11 |
| `events.py` | client/server event types, JSON DTOs, brackets |
| `wire.py` | `OutboundEvent` enum, ResponsePayload, ErrorPayload |
| `framing.py` | `full_message` / `partial_message` envelope (RFC §10.4) |
| `errors.py` | reserved error-code registry (RFC §10.5) |
| `sdp_filter.py` | aiortc -> webrtc-rs SDP normalizer (regex-free) |
| `transport.py` | `EventSink` (DC vs WS) + `OutboundAudioSpec` |
| `audio_in.py` | opus -> 48 kHz mono -> 16 kHz mono f32 |
| `audio_in_ws.py` | WebSocket inbound audio (PCM/G.711) |
| `audio_out.py` | 24 kHz -> 48 kHz -> opus -> outbound track + queue gate |
| `audio_out_ws.py` | WebSocket outbound (PCM/G.711) |
| `session.py` | per-connection orchestration, client-event dispatch |
| `session_update.py` | parse-all -> validate-all -> commit-all (§11.2.1) |
| `pipeline.py` | LLM -> TTS -> wire (commit/bargein/process_utterance) |
| `eou_eager.py` | speculative LLM dispatch (Appendix C/D) |
| `eou_predicted.py` | PredictedTokenBuffer + STT/LLM runners |
| `eou_integrated.py` | IntegratedVerdictAction enum |
| `diarization.py` | per-utterance diarization integration |
| `websocket.py` | HTTP->WS upgrade, wsTransport, idle/ping cadence |
| `fuzz.py` | random-walk property fuzzer over SessionState (§15.5) |

`inspector.rs` deliberately excluded.

## Invariants I1..I11

Statements and rationale: `speaches-plus/rust/IMPLEMENTATION.md` § "Invariant enforcement".
Implemented in `state.check_invariants` and `state.check_state`. Wire-level
invariants (W1..W8) belong to a future canonical assertor shared with the
upstream conformance fixtures. Python enforcement sites:

| ID | Enforcement |
|---|---|
| I1 (speaking => no active response) | `check_invariants` raises `SpeakingWithActiveResponse` |
| I2 (buffer rotated only on commit) | wire-side, fixture-checked |
| I3 (no append after seal) | typestate: `OpenBuffer.seal` consumes |
| I4 (commit => at most one `BufferCommitted`) | wire-side |
| I5 (epochs strictly monotonic) | `last_epoch + 1` on every start |
| I6 (no phantom `in_progress` user item after cancelled commit) | append at `commit_after_eou`, not `vad.speech_stopped` |
| I7 (Predicted => wire emit suppressed) | structural: `RespPhase` has no `wire_emitted` flag; `Session.emit` short-circuits when topic == Response and phase == Predicted |
| I8 (`played_ms` snapshot under-lock at cancel) | `cancel_current_response` reads under `_state_lock` after `resp_retire_to_none` |
| I9 (`Predicted -> Created` preserves id+epoch) | `resp_promote_predicted_to_created` carries previous fields |
| I10 (`audio_end_ms` on every `response.done`) | `ResponsePayload.audio_end_ms` non-optional |
| I11 (`Stopped.audio_end_ms >= audio_start_ms`) | `check_invariants` raises `StoppedWithoutEnd` |

Sealed-buffer retention (RFC v3 §3.4): `SessionState.store_sealed_buffer`
runs FIFO eviction at `sealed_buffer_retention_count` (default 4);
`drop_sealed_buffer` on the matching `input_audio_transcription.completed`
(rule 1).

## Wire shapes

`OutboundEvent` factory methods produce Python dicts via `to_json()`
matching upstream's `serde tag = "type"` shape verbatim. Notable:
`ResponsePayload.audio_end_ms` is required for all four statuses
including `completed` (W4); `ResponseStatusReason` is a closed enum.

## Backpressure (RFC §7.4)

Mechanics as upstream (`speaches-plus/rust/IMPLEMENTATION.md` § "Backpressure"):
`audio_out.QueueGate.try_push` rejects with `QueueFull` when projected
`queued_ms` exceeds `cap_ms`. Cap sourced from `OUTBOUND_QUEUE_CAP_MS`
env or `wire_defaults.OUTBOUND_QUEUE_CAP_MS`.

## What's stubbed (TODO)

- `pipeline.run_response`: LLM-stream -> SentenceChunker -> Kokoro TTS ->
  `OutboundPacer` loop is wired end-to-end. Per-sentence flow: mint a
  `phrase_id` via `ids.next_phrase_id()` -> `Session.set_phrase_id(...)`
  (forwards to `InspectorRelay.set_phrase_id`); emit
  `response.output_audio_transcript.delta` carrying the sentence; iterate
  `ctx.models.kokoro.stream(sentence, voice, speed=, lang=)` and
  `await pacer.play(samples)` per chunk. `response.created` is emitted
  before the first sentence ships. At the end (or on `cancel.is_set()` /
  `CancelledError`), `pacer.flush()` runs (or `pacer.cancel()` and the
  error re-raises) and both `phrase_id` and `turn_id` are cleared.

  Pacer wiring lives in `pipeline._build_pacer_for_session(session,
  kokoro)`: WebRTC sessions get a real `OutboundPacer` over the
  `OutboundOpusTrack` from `Session.outbound_audio.track` plus
  `pacer.attach_capture(session.capture_outbound_f32)`; WebSocket
  sessions get a `WsAudioPacer.start(...)` over
  `Session.outbound_audio.ws_send`. If `RealtimeContext.models.kokoro`
  is `None` (or `RealtimeContext` is unset), the pipeline drops the
  audio path and still emits `response.created` + transcript deltas +
  `response.done`.

  Not wired yet: promotion of an eager `Predicted` response to `Created`,
  and the `complete_stream_messages` SSE forwarder beyond the current
  `LlmConfig.from_env()` HTTP path.
- `eou_eager.try_eager_dispatch`: throttling and predicted-response
  state install are present but the speculative STT/LLM spawn calls
  rely on a yet-to-exist `eou` package + `vad` package.
- `audio_in.AudioIngest`: opus -> 48 kHz mono -> 16 kHz mono f32.
  Hard dependency on `opuslib` (pyproject + flake.nix). Construction
  raises `RuntimeError` if opuslib (or its native `libopus`) can't
  load -- silent no-ops would mask wire-level bugs (mis-negotiated
  SDP, broken decoder install).

  Decode pipeline (parity with `rust/src/realtime/audio_in.rs`):

  1. `process_opus(payload)` -- `opuslib.Decoder.decode` with
     `frame_size=MAX_DECODE_FRAMES` (5760 = 120 ms at 48 kHz; opus
     returns the actual frame count). Defaults configurable via the
     `AudioIngest(channels, frame_samples=...)` ctor so the SDP-
     negotiated frame size (10/20/40/60 ms) can be plumbed in.
  2. `process_av_frame(frame)` -- accept aiortc's decoded
     `av.AudioFrame` (the typical `on_track` path bypasses our opus
     decoder because aiortc has already decoded). Honors planar +
     interleaved layouts, normalises i16/i32/u8/f32 -> f32 mono.
  3. Downmix: stereo -> mono via channel mean (matches Rust's
     `(l + r) / 2 / 32768`).
  4. Resample 48 kHz -> 16 kHz with polyphase decimation (down=3).
     Preferred backend: `scipy.signal.resample_poly` with a Kaiser
     window (beta=8.6). Fallback: project-local polyphase FIR with a
     Kaiser-windowed sinc kernel (65 taps per phase). Linear
     `np.interp` is wrong for audio -- replaced.
  5. State carried across chunks: a small 48 kHz "tail" buffer keeps
     resampler boundary samples so concatenating consecutive
     `process_*` calls matches one-shot resampling (within filter
     ringing).

  Threading: `process_opus` / `process_av_frame` may be called from
  the aiortc receiver task or a worker thread; `take()` / `take_array()`
  from the VAD/STT task. A `threading.Lock` guards `_buf`, the tail
  buffer, and the sample counters. `take()` returns a copy and resets
  the buffer atomically. `get_total_samples_consumed()` returns
  cumulative 16 kHz output samples (for VAD/EOU end-of-utterance math);
  `get_total_input_samples()` cumulative 48 kHz input samples.

  nixpkgs note: as of the bundled mirror, `opuslib` is the canonical
  pure-Python ctypes wrapper around `libopus`. If a future bump drops
  it, alternatives in order of preference: `pyogg` (vendored libopus,
  broader scope), then `python-opus`. Either way the system `libopus`
  must be on `LD_LIBRARY_PATH`; on NixOS it lives at
  `${pkgs.libopus}/lib`.
- `transport.maybe_handle_offer`: performs the full upstream
  `handle_offer` orchestration: builds an `OutboundOpusTrack` and
  `pc.addTrack`s it BEFORE `createAnswer` (aiortc requires the
  answerer's media track present before SDP renegotiation), registers
  `pc.on("track")` to spawn an inbound pump calling
  `session.audio_in.process_av_frame(frame)`, registers
  `pc.on("datachannel")` to wire `EventSink.data_channel_sink(dc)`
  and forward `dc.on("message")` into `session.handle_client_event`,
  registers `pc.on("connectionstatechange")` to drop the session and
  cancel pumps on `failed`/`closed`/`disconnected`, and adds the
  session to a module-local registry so `live_session_count()` covers
  WebRTC sessions too. Dependencies (models view, instructions, ...)
  are injected via a `RealtimeContext` set once during FastAPI
  lifespan startup with `transport.set_context(ctx)` -- avoids
  broadening the `maybe_handle_offer(offer_sdp, query)` signature
  while letting the route construct a `Session` with the right
  resources.
- `transport._inbound_consumer` (WebRTC mic_in drain): per-track,
  spawned alongside `_inbound_track_pump` from `pc.on("track")`.
  Two-task design -- the pump owns aiortc decode (`track.recv()` +
  `audio_in.process_av_frame(frame)`); the consumer owns Session-side
  dispatch. Polls `audio_in.take_array()` every
  `_INBOUND_CONSUMER_POLL_S = 0.02` s (matches the 20 ms VAD frame
  cadence) and for each non-empty chunk calls
  `session.capture_inbound_f32(samples)` -- the same shim the WS
  ingest path (`audio_in_ws.handle_audio_append`) uses, so mic_in
  inspect capture and any future VAD/EOU dispatch fire identically
  across transports. Empty buffers skipped (no spurious inspect
  writes). When the pump task is `done()` the consumer performs one
  final `take_array()`/dispatch before exiting; `CancelledError`
  during the sleep triggers the same final flush then re-raises. The
  consumer joins the same `pumps` list as the pump so
  `_teardown_session` cancels both on `connectionstatechange` in
  {failed, closed, disconnected}. `AudioIngest.take_array()` is
  internally locked; no extra lock needed. To bound growth when the
  consumer is blocked (heavy LLM/TTS work), `AudioIngest` caps its
  16 kHz buffer at `_MAX_BUFFER_SAMPLES_16K = 16_000 * 30` (30 s
  mono); on overflow the oldest samples are dropped and
  `dropped_samples` incremented (logged at WARNING).
- `pipeline.commit_after_eou`: full predicted-runner await, transcript
  mismatch check, and predicted-LLM promotion are skeletal -- the
  await-and-promote logic mirrors upstream but the runners are stubs
  pending the `eou`/`vad` packages.

## Concurrency model

Upstream uses a `tokio::sync::Mutex<SessionState>`; we mirror with
`asyncio.Lock` (`Session._state_lock`). Every transition flows through
it; `check_or_react(session, state)` runs after each mutation. Unlike
Rust's `JoinHandle::abort()` + `await` happens-before, Python
`asyncio.Task.cancel()` is cooperative -- `played_ms` is read while
still holding `_state_lock` so the snapshot reflects exactly what
shipped (RFC §C.7).

### Locking discipline

* `transport._sessions` (WebRTC registry) -- `transport._sessions_lock`
  (`asyncio.Lock`); all mutation via `_register_session(...)` /
  `_drop_session(...)`. Off-loop callers (e.g. `connectionstatechange`
  callbacks on aiortc internal threads) use
  `_schedule_drop_session(sid, loop=...)` which hops back via
  `loop.call_soon_threadsafe`. The synchronous `_drop_session_sync`
  helper was removed; it bypassed the lock and raced
  `_register_session`.
* `websocket._ws_sessions` (WS registry) -- `websocket._ws_sessions_lock`
  (`asyncio.Lock`); mutate via `_register_ws_session(...)` /
  `_drop_ws_session(...)`. Read paths that iterate must call
  `snapshot_ws_sessions()` first; single-key `dict.get` is safe under
  the GIL but unsnapshotted iteration can raise `RuntimeError:
  dictionary changed size during iteration`.
* `OutboundOpusTrack._pending` -> `_queue` transfer -- `_pending_lock`
  (`threading.Lock`, since `push_nowait` is called from off-loop
  encoder/pacer threads). `_ensure_queue` drains `_pending` to a
  freshly created `asyncio.Queue` atomically: pending list cleared,
  queue assigned, lock released, in that order.
* `_dc_tasks` (per-session bag of data-channel-handler tasks) -- owned
  by the session loop. `_attach_data_channel_handlers` appends each
  `asyncio.create_task` (the `attach()` coroutine and every per-message
  dispatch) and `_teardown_session` cancels + gathers them before
  terminating, so dispatched coroutines cannot outlive the session.

## OutboundPacer (audio_out.py) -- real opus

Real-opus port of `speaches-plus/rust/src/realtime/audio_out.rs`
(pacing rationale there). Python deltas:

- Resample 24k -> 48k: prefers `scipy.signal.resample_poly(arr, 2, 1)`
  (polyphase, anti-aliased), falls back to `librosa.resample`, then
  naive `np.interp` as last resort. Naive interp aliases above ~12 kHz
  which is audible at opus voip bitrates.
- Opus encoder: `opuslib.Encoder(48000, 1, 'voip')` matching upstream
  Rust (`Application::Voip`). `bitrate=64_000`, `complexity=5`,
  `inband_fec=1` mirror the WebRTC convention used by the Rust path
  (sdp_fmtp_line `useinbandfec=1`). Encoder construction is deferred
  to first encode so `import realtime.audio_out` succeeds without
  opuslib installed.
- PCM conversion: `_f32_to_s16le_bytes` does scale-then-clip
  (`rint(s * 32767.0)` then clip to `[-32768, 32767]`), byte-for-byte
  equivalent to the inner step of `audio/g711.py`'s `f32_to_*_bytes`
  helpers -- matters because the encoder is sensitive to dither at the
  LSB.
- Track interface: contracts with the transport's
  `OutboundOpusTrack(MediaStreamTrack)` via `push_opus_frame(payload,
  duration_ms)`. Optional `end_of_stream()` and `drop_queued()` hooks
  are called by `flush()` and `cancel()` respectively. If
  `push_opus_frame` is absent, falls back to a `track.queue` attribute
  (Queue.put or list.append) so a custom test fake can opt out.
- Pacing: wall-clock target `start + frames_written * FRAME_MS`, same
  as upstream. Cancellable: `asyncio.CancelledError` during the pacing
  sleep flips `_cancelled` and re-raises. `cancel()` is idempotent:
  drops the gate, snapshots `played_ms_ref` to
  `frames_written * FRAME_MS`, and asks the track to drop unsent
  payloads.
- Cancel race: `play()` checks `self._cancelled` at entry, before each
  per-frame loop iteration, and `_write_encoded_frame` re-checks at its
  own entry -- so a `cancel()` from a sibling task cannot get
  sandwiched between the loop's gate check and a subsequent
  `gate.on_frame_sent()`. `QueueGate.on_frame_sent()` clamps at zero
  (`max(0, queued_ms - FRAME_MS)`) as a defensive backstop.
- Flush: encodes one tail-silence frame to drain the encoder's
  lookahead (Celt/Silk both carry ~6.5 ms of look-ahead at 48k mono
  voip), then calls `track.end_of_stream()`. No-op if no frames were
  ever encoded.

The `audio_out_ws.py` sibling intentionally stays separate: it speaks
PCM/G.711 over a WebSocket JSON envelope and never touches opus, so
the resample-then-encode-then-pace pipeline does not collapse into a
shared base class without forcing transport leakage.

## Capacity caps

- `WS_MAX_CONCURRENT_SESSIONS`: enforced in `websocket.realtime_ws_endpoint`
  via a global async-protected counter (`_active_sessions`).
- `WS_IDLE_TIMEOUT_S`: `asyncio.wait_for(websocket.receive(), timeout=...)`.
- `WS_OUTBOUND_QUEUE_CAP`: bounded `asyncio.Queue` for the WS writer.
- `OUTBOUND_QUEUE_CAP_MS`: per-pacer wall-clock cap on queued audio.
- `PING_INTERVAL_S`: writer task sends a periodic empty text frame
  (Starlette's WebSocket has no native ping in the same way).

## Inverted dependency via SessionObserver

The realtime layer does not import `inspect_api`. `realtime/observer.py`
defines a `SessionObserver` Protocol (with a `NullObserver` no-op default);
`Session.__init__` accepts `observer: SessionObserver | None = None` or an
`observer_factory: Callable[[str], SessionObserver] | None = None` (called
with the freshly-minted `self.id`). The session forwards every wire event,
audio chunk, and correlation update through the observer without knowing
whether inspect is wired. Hook sites, correlation derivation, audio-capture
wiring, and the try/except policy are documented in
`../inspect_api/IMPLEMENTATION.md` § "Inverted dependency via
SessionObserver". Realtime-side facts:

* `turn_id` (`turn_<hex>`) -- minted in `pipeline.commit_after_eou` right
  before `input_audio_buffer.committed` ships, via `ids.next_turn_id()` ->
  `Session.set_turn_id(...)` -> `observer.on_correlation(turn_id=...)`.
  Cleared (`None`) at `response.done`.
* `phrase_id` (`phrase_<hex>`) -- minted in `pipeline.run_response` per TTS
  sentence, via `ids.next_phrase_id()` -> `Session.set_phrase_id(...)` ->
  `observer.on_correlation(phrase_id=...)`. Cleared at `response.done` and
  on cancel/error.
* `Session.transition_to_terminated_with` invokes
  `self._observer.on_session_end(sid)`.
* Lookup helpers: `realtime.lookup_session_pub(sid)` returns the active
  `Session` from either registry (`transport._sessions` or
  `websocket._ws_sessions`); `realtime.lookup_session_relay(sid)` returns
  the session's observer-owned relay or `None`.
  `inspect_api.routes._try_live_slice` uses neither -- it queries
  `inspect_api.registry.get_audio_store(sid)` directly.

## Capabilities advertisement (`capabilities_json_with_models`)

`features.eou_kinds` and `extensions.eou_kinds` partition the universe of
end-of-utterance kinds defined in `eou/types.py`:

* `features.eou_kinds` -- driven by `EouKind.V3_SPEC` (currently
  `("vad", "text", "audio", "fusion")`). RFC v3 canonical kinds.
* `extensions.eou_kinds` -- driven by `EouKind.EXTENSIONS` (currently
  `("heuristic", "integrated")`). Non-spec kinds we add on top. The two
  lists are disjoint by construction; never hardcode either side.

Capability flags in `extensions` are gated on environment so
`/v1/realtime/capabilities` reflects what the running server can actually
do, not what the code base ships:

| Flag | Truthy when |
|---|---|
| `eager_eou` | `EOU_EAGERNESS` set non-empty (any of `low/medium/high/auto`) |
| `integrated_eou` | `EOU_KIND` unset (default) or its value contains the substring `integrated` |
| `audio_eou` | `EOU_AUDIO_MODEL_PATH` set non-empty |
| `predicted_resp_phase` | unconditionally `True` (structural) |

Operators flipping the env vars at deploy time get an honest capability map
without restarting any model loaders; clients gating on `extensions.<flag>`
won't try a kind that has no model behind it.
