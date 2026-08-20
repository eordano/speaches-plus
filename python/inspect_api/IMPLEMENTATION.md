# inspect_api -- port of `speaches-plus/rust/src/inspect/`

Renamed from upstream `inspect` to `inspect_api` to avoid shadowing the
Python stdlib `inspect` module. All upstream symbol names (`SessionMeta`,
`SessionHistoryEntry`, `AudioQuery`, `InspectorRelay`, `AudioStore`,
`register`, `unregister`, `inspect_sessions`, `inspect_history`,
`inspect_history_stream`, `inspect_audio`, `inspect_stream_ws`, `Channel`,
`wav_header`, `is_error_kind`, `is_known_lane`, `cleanup_on_startup`,
`session_dir`, `retention_count`, `retention_bytes`, `retention_days`,
`run_startup_cleanup`) are preserved.

## Source map

| Rust file (`speaches-plus/rust/src/inspect/`) | Python file (`inspect_api/`) |
| --- | --- |
| `mod.rs`        | `__init__.py` + helpers in `retention.py` (`expand_home`, `session_dir`, `retention_*`, `run_startup_cleanup`) |
| `constants.rs`  | `constants.py` |
| `types.rs`      | `types.py`     |
| `registry.rs`   | `registry.py`  |
| `relay.rs`      | `relay.py`     |
| `retention.rs`  | `retention.py` |
| `audio_store.rs`| `audio_store.py` |
| `routes.rs`     | `routes.py`    |

## Concurrency mapping

* Rust `Mutex<...>` -> `threading.Lock()` (the relay is called from sync hot
  paths; `asyncio.Lock` would force every callsite to be `async`).
* Rust `tokio::sync::broadcast` -> per-subscriber `asyncio.Queue`. Each
  `subscribe()` returns its own queue plus a snapshot of the replay buffer.
  Publish is sync; it fans out to all queues with `loop.call_soon_threadsafe`
  when a subscriber's loop is running, otherwise direct `put_nowait`.
* Rust `RwLock<HashMap<...>>` (registry) -> `threading.Lock()` + `dict`.
* Rust `OnceLock` -> module-level singletons in `registry.py` and
  `retention.py`.

### Locking discipline

* `InspectorRelay._lock` (`threading.Lock`) -- guards `_seq`,
  `_replay_buffer`, `_subscribers`, `_sub_loops`, `_dropped_count`,
  `_turn_count`, `_last_event_ts`, the `_*_id` correlation fields, and
  `_ndjson` writes. Hot path: `publish()` snapshots
  `list(zip(_subscribers, _sub_loops))` under lock and fans out *outside*
  the lock so a slow subscriber cannot block other publishers.
* `InspectorRelay.subscribe(loop=None)` -- must run in an async context.
  The loop is captured via `asyncio.get_running_loop()` (or passed
  explicitly); calling from outside any loop raises `RuntimeError` with a
  clear message. The previous behaviour (`get_event_loop()` falling back to
  `new_event_loop()`) silently installed a never-running loop, which made
  `_fanout` take the wrong branch and silently drop events.
* Registry (`registry._reg_lock`) -- `threading.Lock`; same shape as
  upstream `RwLock<HashMap<...>>`.

## Retention

`retention.py` exposes both `cleanup_on_startup(...)` (synchronous, invoked
at app startup and from tests) and `retention_loop(interval_s=300)` (async
background task re-running the same cleanup; the server lifespan creates it
on startup and cancels it on shutdown).

## Audio store

PCM-16 little-endian, mono. `Channel.MIC_IN` = 16 kHz, `Channel.TTS_OUT` =
24 kHz. Two append-only `.raw` files per session; on `close()` a
`.audio.json` sidecar records sample counts and `offset_ms` per track. The
`/v1/inspect/sessions/{sid}/audio` route reads either the live
`AudioStore.slice(...)` (session still in the registry) or re-reads the
`.raw` files plus sidecar.

## Routes

All five endpoints from `routes.rs` are mounted on a `fastapi.APIRouter`
named `router`, included by `server.py`:

* `GET  /v1/inspect/sessions`
* `GET  /v1/inspect/sessions/history`
* `GET  /v1/inspect/sessions/history/{sid}`
* `GET  /v1/inspect/sessions/{sid}/audio`
* `WS   /v1/inspect/{sid}/stream`

## Inverted dependency via SessionObserver

`realtime/session.py::Session` knows nothing about `inspect_api`. It accepts
a `SessionObserver` Protocol (defined in `realtime/observer.py`; default
`NullObserver` no-op) and forwards every wire event, audio chunk, and
correlation update through it.

`inspect_api/observer.py::InspectObserver` implements the protocol -- it
owns an `InspectorRelay` and `AudioStore` (constructed in
`on_session_start`) and registers itself in the inspector `registry`
(storing both the relay and the audio store so `routes._try_live_slice` can
fetch the audio store directly via `registry.get_audio_store(sid)` without
ever importing `realtime`).

`server.py` lifespan injects the wiring:

```
set_context(RealtimeContext(
    models=...,
    observer_factory=inspect_api.make_observer_factory(),
))
```

so every WebSocket and WebRTC session gets an `InspectObserver` bound to its
sid. `Session.transition_to_terminated_with` calls
`observer.on_session_end(sid)`, which closes `audio_store` (writes the
`<sid>.audio.json` sidecar) and `registry.unregister(sid)` (closes the
relay, flushes ndjson, drains subscribers).

Hook sites -- every event the session emits is forwarded to the observer:

* `Session.emit(OutboundEvent)` -> after `sink.send_value(ev)`,
  `observer.on_outbound_event(ev)`. `InspectObserver` derives correlation
  updates (`set_response_id`, `set_item_id`) from the canonical bracket
  events (`response.created`, `response.done`, `conversation.item.added`,
  `input_audio_buffer.speech_started`, `input_audio_buffer.committed`) in
  `_update_corr_from_event` before publishing
  `relay.publish("wire", "out", ...)`.
* `Session.emit_event(dict)` -> `observer.on_outbound_event_dict(ev)`.
* `Session._emit_session_created` -> `observer.on_outbound_event(ev)`.
* `Session._emit_error` -> `observer.on_outbound_event(ev)` plus
  `observer.on_error(code, message, event_id, param)`. `InspectObserver`
  publishes the dedicated `lane="error", kind="raised"` payload; the relay
  also auto-mirrors `ERR_KINDS` kinds via `_build_error_mirror`.
* `Session.handle_client_event(...)` (inbound) ->
  `observer.on_inbound_event(kind, payload, raw_text)` ->
  `relay.publish("wire", "in", None, parsed_event)`.

All observer calls are wrapped in `try/except` so a failing observer (file
write errors, full subscriber queues, relay failure) never aborts a session.

Lookup contract (used by `routes.py::_try_live_slice`):

* `inspect_api.registry.get_audio_store(sid)` -- returns the `AudioStore`
  registered by `InspectObserver.on_session_start`, or `None` if the
  session has terminated. `_try_live_slice` uses this directly without
  importing `realtime`.
* `realtime.lookup_session_pub(sid)` and `realtime.lookup_session_relay(sid)`
  remain available for callers that need the live `Session` or its
  observer's relay (the latter pulls `session._observer.relay`).

Audio capture:

* `mic_in` (16 kHz f32) -- wired in
  `realtime/audio_in_ws.py::handle_audio_append` via
  `Session.capture_inbound_f32(samples)` ->
  `observer.on_inbound_audio_f32(samples)` -> `audio_store.append_mic_in_f32`.
* `tts_out` (24 kHz f32) -- wired in
  `realtime/pipeline.py::_build_pacer_for_session`: for the WebRTC path the
  per-response `OutboundPacer` immediately receives
  `pacer.attach_capture(session.capture_outbound_f32)` (runs at the head of
  `play(...)` with the raw 24 kHz buffer, before resample-to-48k), which
  forwards to `observer.on_outbound_audio_f32(samples)` so every TTS chunk
  is also written to `${INSPECT_SESSION_DIR}/<sid>/tts_out.raw`. The
  WebSocket path (`audio_out_ws.WsAudioPacer`) does not capture tts_out --
  the PCM/G.711 it emits is the wire format, not Kokoro's 24 kHz native
  output, and the upstream Rust path does not capture WS-shaped output
  either.

Correlation IDs: `turn_id` and `phrase_id` are minted by the realtime
pipeline (sites and lifecycle in `../realtime/IMPLEMENTATION.md`) and pushed
through `Session.set_turn_id` / `set_phrase_id` ->
`observer.on_correlation(...)` -> `InspectorRelay.set_turn_id` /
`set_phrase_id`.

No static UI files are included. The original `inspect/static/` tree from
upstream is intentionally not ported.
