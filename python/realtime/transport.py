from __future__ import annotations

import asyncio
import enum
import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .framing import frame_event
from .observer import SessionObserver

log = logging.getLogger("realtime.transport")

class _SinkKind(enum.Enum):
    DATA_CHANNEL = "data_channel"
    WEB_SOCKET = "web_socket"

@dataclass
class EventSink:
    kind: _SinkKind
    data_channel: Any = None
    ws_send: asyncio.Queue | None = None

    @classmethod
    def data_channel_sink(cls, dc: Any) -> EventSink:
        return cls(kind=_SinkKind.DATA_CHANNEL, data_channel=dc)

    @classmethod
    def websocket_sink(cls, ws_send: asyncio.Queue) -> EventSink:
        return cls(kind=_SinkKind.WEB_SOCKET, ws_send=ws_send)

    async def send_text(self, text: str) -> None:
        if self.kind is _SinkKind.DATA_CHANNEL:
            dc = self.data_channel
            if dc is not None:
                send = getattr(dc, "send", None)
                if send is not None:
                    res = send(text)
                    if asyncio.iscoroutine(res):
                        await res
        else:
            if self.ws_send is not None:
                await self.ws_send.put(text)

    async def send_value(self, ev: Any) -> None:
        if self.kind is _SinkKind.DATA_CHANNEL:
            try:
                frames = frame_event(ev)
            except (TypeError, ValueError) as err:
                log.warning("framing failed: %s", err)
                return
            for frame in frames:
                try:
                    await self.send_text(frame)
                except Exception as err:
                    log.warning("event sink send failed: %s", err)
                    break
        else:
            if hasattr(ev, "to_json"):
                payload = ev.to_json()
            else:
                payload = ev
            try:
                text = json.dumps(payload, separators=(",", ":"))
            except (TypeError, ValueError) as err:
                log.warning("ws json serialize failed: %s", err)
                return
            try:
                await self.send_text(text)
            except Exception as err:
                log.warning("ws sink send failed: %s", err)

class _SpecKind(enum.Enum):
    WEBRTC = "webrtc"
    WEBSOCKET = "websocket"

@dataclass
class OutboundAudioSpec:
    kind: _SpecKind
    track: Any = None
    ws_send: asyncio.Queue | None = None
    format: str = ""

    @classmethod
    def webrtc(cls, track: Any) -> OutboundAudioSpec:
        return cls(kind=_SpecKind.WEBRTC, track=track)

    @classmethod
    def websocket(cls, ws_send: asyncio.Queue, format: str) -> OutboundAudioSpec:
        return cls(kind=_SpecKind.WEBSOCKET, ws_send=ws_send, format=format)

    def is_webrtc(self) -> bool:
        return self.kind is _SpecKind.WEBRTC

    def is_websocket(self) -> bool:
        return self.kind is _SpecKind.WEBSOCKET

@dataclass
class RealtimeContext:
    models: Any = None
    pipeline_factory: Any = None
    instructions: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    observer_factory: Callable[[str], SessionObserver] | None = None
    vad_model: Any = None
    transcribe_factory: Callable[["Any"], Any] | None = None

_context: RealtimeContext | None = None

def set_context(ctx: RealtimeContext | None) -> None:
    global _context
    _context = ctx

def get_context() -> RealtimeContext | None:
    return _context

_sessions: dict[str, Any] = {}
_sessions_lock = asyncio.Lock()

def webrtc_session_count() -> int:
    return len(_sessions)

async def _register_session(session: Any) -> None:
    async with _sessions_lock:
        _sessions[session.id] = session

async def _drop_session(session_id: str) -> Any | None:
    async with _sessions_lock:
        return _sessions.pop(session_id, None)

def _schedule_drop_session(session_id: str, loop: asyncio.AbstractEventLoop | None = None) -> None:
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
    coro = _drop_session(session_id)
    if loop.is_running():
        try:
            loop.call_soon_threadsafe(lambda: loop.create_task(coro))
            return
        except RuntimeError:
            pass
    coro.close()

def _try_import_aiortc():
    try:
        from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
        from aiortc.mediastreams import MediaStreamError

        return MediaStreamTrack, RTCPeerConnection, RTCSessionDescription, MediaStreamError
    except ImportError:
        return None

def _try_import_av():
    try:
        import av

        return av
    except ImportError:
        return None

def _build_outbound_track(queue_maxsize: int = 256) -> "OutboundOpusTrack":
    return OutboundOpusTrack(queue_maxsize=queue_maxsize)

class _BaseTrack:
    kind = "audio"

    def stop(self) -> None:
        pass

def _outbound_track_base():
    bundle = _try_import_aiortc()
    if bundle is None:
        return _BaseTrack
    return bundle[0]

class OutboundOpusTrack(_outbound_track_base()):
    kind = "audio"

    def __init__(self, queue_maxsize: int = 256, sample_rate: int = 48_000, frame_ms: int = 20):
        try:
            super().__init__()
        except TypeError:
            pass
        self._queue_maxsize = queue_maxsize
        self._queue: asyncio.Queue[Any] | None = None
        self._closed_flag = False
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._samples_per_frame = sample_rate * frame_ms // 1000
        self._pts: int = 0
        self._time_base = None
        self._pending: list[Any] = []
        self._pending_lock = threading.Lock()
        self._end_of_stream_pending = False

    def push_opus_frame(self, payload: Any, frame_ms: int = 20) -> bool:
        """Push one 20ms PCM frame (or opaque payload) to the outbound queue.

        Keeps the legacy name (`push_opus_frame`) because audio_out.OutboundPacer
        and tests target it. The payload is the bytes that `recv()` will wrap into
        an AudioFrame; aiortc handles opus encoding for the wire. Returns False if
        the queue was full and we had to drop.
        """
        return self.push_nowait(payload)

    def drop_queued(self) -> None:
        """Drain the outbound queue on barge-in cancellation."""
        with self._pending_lock:
            self._pending = []
            queue = self._queue
        if queue is not None:
            while True:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    break

    def end_of_stream(self) -> None:
        """Mark the track as drained; consumer will see EOF on the next `recv`.

        Used by the pacer's `flush()` path so the wire knows playback finished.
        For long-lived conversation tracks we usually do NOT want to close the
        track at end-of-utterance (the next utterance will reuse it), so this is
        a soft "tail-of-stream" marker -- implemented as a no-op so subsequent
        pushes keep working. The pacer treats it as best-effort.
        """
        return None

    def _ensure_queue(self) -> asyncio.Queue[Any]:
        with self._pending_lock:
            if self._queue is None:
                q: asyncio.Queue[Any] = asyncio.Queue(maxsize=self._queue_maxsize)
                pending = self._pending
                self._pending = []
                for item in pending:
                    try:
                        q.put_nowait(item)
                    except asyncio.QueueFull:
                        break
                self._queue = q
            return self._queue

    async def push(self, payload: Any) -> None:
        if self._closed_flag:
            return
        await self._ensure_queue().put(payload)

    def push_nowait(self, payload: Any) -> bool:
        if self._closed_flag:
            return False
        with self._pending_lock:
            queue = self._queue
            if queue is None:
                self._pending.append(payload)
                return True
        try:
            queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            return False

    def close(self) -> None:
        self._closed_flag = True
        with self._pending_lock:
            queue = self._queue
            if queue is None:
                self._pending.append(None)
                return
        try:
            queue.put_nowait(None)
        except asyncio.QueueFull:
            pass

    async def recv(self) -> Any:
        payload = await self._ensure_queue().get()
        if payload is None:
            self._closed_flag = True
            bundle = _try_import_aiortc()
            if bundle is not None:
                _, _, _, MediaStreamError = bundle
                raise MediaStreamError("OutboundOpusTrack closed")
            raise EOFError("OutboundOpusTrack closed")
        av = _try_import_av()
        if av is None:
            return payload
        if hasattr(payload, "samples") and hasattr(payload, "pts"):
            return payload
        frame = av.AudioFrame(format="s16", layout="mono", samples=self._samples_per_frame)
        frame.sample_rate = self._sample_rate
        if isinstance(payload, (bytes, bytearray, memoryview)):
            try:
                frame.planes[0].update(bytes(payload))
            except Exception:
                pass
        frame.pts = self._pts
        if self._time_base is None:
            from fractions import Fraction

            self._time_base = Fraction(1, self._sample_rate)
        frame.time_base = self._time_base
        self._pts += self._samples_per_frame
        return frame

async def _inbound_track_pump(track: Any, session: Any) -> None:
    audio_in_obj = getattr(session, "audio_in", None)
    while True:
        try:
            frame = await track.recv()
        except Exception as err:
            log.debug("inbound track recv ended: %s", err)
            return
        if frame is None:
            return
        if audio_in_obj is None:
            audio_in_obj = getattr(session, "audio_in", None)
        if audio_in_obj is None:
            continue
        handler = getattr(audio_in_obj, "process_av_frame", None)
        if handler is None:
            handler = getattr(audio_in_obj, "process_frame", None)
        if handler is None:
            continue
        try:
            res = handler(frame)
            if asyncio.iscoroutine(res):
                await res
        except Exception as err:
            log.warning("audio frame handler failed: %s", err)

_INBOUND_CONSUMER_POLL_S = 0.02

def _dispatch_inbound_samples(session: Any, samples: Any) -> None:
    if samples is None:
        return
    size = getattr(samples, "size", None)
    if size is not None:
        if size == 0:
            return
    elif not samples:
        return
    capture = getattr(session, "capture_inbound_f32", None)
    if capture is not None:
        try:
            capture(samples)
        except Exception as err:
            log.warning("inbound audio capture failed: %s", err)
    runner = getattr(session, "vad_runner", None)
    if runner is not None:
        try:
            runner.push_samples(samples)
        except Exception as err:
            log.warning("vad_runner.push_samples failed: %s", err)

async def _inbound_consumer(session: Any, pump_task: asyncio.Task) -> None:
    audio_in_obj = getattr(session, "audio_in", None)
    while True:
        if audio_in_obj is None:
            audio_in_obj = getattr(session, "audio_in", None)
        if audio_in_obj is not None:
            take = getattr(audio_in_obj, "take_array", None)
            if take is not None:
                try:
                    samples = take()
                except Exception as err:
                    log.warning("audio_in.take_array failed: %s", err)
                    samples = None
                _dispatch_inbound_samples(session, samples)
        if pump_task.done():
            if audio_in_obj is not None:
                take = getattr(audio_in_obj, "take_array", None)
                if take is not None:
                    try:
                        tail = take()
                    except Exception as err:
                        log.warning("audio_in.take_array final flush failed: %s", err)
                        tail = None
                    _dispatch_inbound_samples(session, tail)
            return
        try:
            await asyncio.sleep(_INBOUND_CONSUMER_POLL_S)
        except asyncio.CancelledError:
            if audio_in_obj is not None:
                take = getattr(audio_in_obj, "take_array", None)
                if take is not None:
                    try:
                        tail = take()
                    except Exception:
                        tail = None
                    _dispatch_inbound_samples(session, tail)
            raise

def _track_dc_task(session: Any, task: asyncio.Task) -> None:
    bag = getattr(session, "_dc_tasks", None)
    if bag is None:
        bag = []
        try:
            setattr(session, "_dc_tasks", bag)
        except (AttributeError, TypeError):
            return
    bag.append(task)
    task.add_done_callback(lambda t, _b=bag: _b.remove(t) if t in _b else None)

def _attach_data_channel_handlers(session: Any, dc: Any) -> None:
    sink = EventSink.data_channel_sink(dc)
    attach = getattr(session, "attach_data_channel", None)
    attach_task: asyncio.Task | None = None
    if attach is not None:
        coro = attach(dc)
        if asyncio.iscoroutine(coro):
            attach_task = asyncio.create_task(coro)
            _track_dc_task(session, attach_task)
    else:
        try:
            session.state.event_sink = sink
        except AttributeError:
            pass

    emit_created = getattr(session, "emit_session_created", None)
    emitted = [False]

    async def _emit_session_created_after_attach() -> None:
        if attach_task is not None and not attach_task.done():
            try:
                await attach_task
            except Exception:
                pass
        try:
            await emit_created()
        except Exception as err:
            log.warning("emit_session_created failed: %s", err)

    def _emit_once() -> None:
        if emitted[0] or emit_created is None:
            return
        emitted[0] = True
        _track_dc_task(
            session, asyncio.create_task(_emit_session_created_after_attach())
        )

    @dc.on("open")
    def _on_open() -> None:
        _emit_once()

    ready = getattr(dc, "readyState", None)
    if ready == "open":
        _emit_once()

    @dc.on("message")
    def _on_message(message: Any) -> None:
        handler = getattr(session, "handle_client_event", None)
        if handler is None:
            return
        try:
            res = handler("dc", message)
            if asyncio.iscoroutine(res):
                _track_dc_task(session, asyncio.create_task(res))
        except Exception as err:
            log.warning("dc client event dispatch failed: %s", err)

def _build_session(query: Any, ctx: RealtimeContext | None, outbound_spec: OutboundAudioSpec | None) -> Any:
    from .session import Intent, Session

    intent = Intent.from_query(query)
    instructions = ctx.instructions if ctx is not None else None
    factory = ctx.observer_factory if ctx is not None else None
    session = Session(
        query=query,
        intent=intent,
        outbound_audio=outbound_spec,
        instructions=instructions,
        observer_factory=factory,
    )
    _attach_vad_runner(session, ctx)
    return session

def _attach_vad_runner(session: Any, ctx: RealtimeContext | None) -> None:
    if ctx is None or ctx.vad_model is None or ctx.transcribe_factory is None:
        return
    try:
        transcribe = ctx.transcribe_factory(session)
    except Exception as err:
        log.warning("transcribe_factory failed: %s", err)
        return
    if transcribe is None:
        return
    try:
        from .audio_in import AudioIngest

        if getattr(session, "audio_in", None) is None:
            session.audio_in = AudioIngest(channels=1)
    except Exception as err:
        log.warning("AudioIngest setup failed: %s", err)
        return
    try:
        from .vad_runner import VadRunner

        session.vad_runner = VadRunner(session, ctx.vad_model, transcribe)
        session.vad_runner.start()
    except Exception as err:
        log.warning("VadRunner setup failed: %s", err)

async def _teardown_session(pc: Any, session: Any, tasks: list[asyncio.Task]) -> None:
    for t in tasks:
        if not t.done():
            t.cancel()
    dc_tasks = list(getattr(session, "_dc_tasks", []) or [])
    for t in dc_tasks:
        if not t.done():
            t.cancel()
    if dc_tasks:
        try:
            await asyncio.gather(*dc_tasks, return_exceptions=True)
        except Exception:
            pass
    runner = getattr(session, "vad_runner", None)
    if runner is not None:
        try:
            await runner.stop()
        except Exception as err:
            log.debug("vad_runner stop failed: %s", err)
    drop_handler = getattr(session, "transition_to_terminated", None)
    if drop_handler is not None:
        try:
            await drop_handler()
        except Exception as err:
            log.debug("session terminate failed: %s", err)
    abort = getattr(session, "abort_timeout_task", None)
    if abort is not None:
        try:
            await abort()
        except Exception:
            pass
    sid = getattr(session, "id", None)
    if sid is not None:
        await _drop_session(sid)
    if pc is not None:
        try:
            await pc.close()
        except Exception:
            pass

async def maybe_handle_offer(offer_sdp: str, query: Any) -> str | None:
    bundle = _try_import_aiortc()
    if bundle is None:
        return None
    _, RTCPeerConnection, RTCSessionDescription, _ = bundle
    from .sdp_filter import normalize_offer
    from .session import Intent

    ctx = get_context()
    pc = RTCPeerConnection()

    intent = Intent.from_query(query)
    outbound_track: OutboundOpusTrack | None = None
    outbound_spec: OutboundAudioSpec | None = None
    if intent is Intent.CONVERSATION:
        outbound_track = _build_outbound_track()
        try:
            pc.addTrack(outbound_track)
        except Exception as err:
            log.warning("addTrack failed: %s", err)
        outbound_spec = OutboundAudioSpec.webrtc(outbound_track)

    session = _build_session(query, ctx, outbound_spec)

    pumps: list[asyncio.Task] = []

    @pc.on("track")
    def _on_track(track: Any) -> None:
        log.info("track received: kind=%s id=%s", getattr(track, "kind", None), getattr(track, "id", None))
        task = asyncio.create_task(_inbound_track_pump(track, session))
        pumps.append(task)
        consumer = asyncio.create_task(_inbound_consumer(session, task))
        pumps.append(consumer)

    @pc.on("datachannel")
    def _on_datachannel(dc: Any) -> None:
        log.info("data channel: label=%s", getattr(dc, "label", None))
        _attach_data_channel_handlers(session, dc)

    @pc.on("connectionstatechange")
    async def _on_state_change() -> None:
        state = getattr(pc, "connectionState", None)
        log.debug("PC state: %s session=%s", state, getattr(session, "id", None))
        if state in ("failed", "closed", "disconnected"):
            await _teardown_session(pc, session, pumps)

    normalized = normalize_offer(offer_sdp)
    await pc.setRemoteDescription(RTCSessionDescription(sdp=normalized, type="offer"))
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    await _register_session(session)
    activate = getattr(session, "transition_to_active", None)
    if activate is not None:
        try:
            await activate()
        except Exception as err:
            log.debug("session activate failed: %s", err)
    spawn_timeout = getattr(session, "spawn_max_duration_timeout", None)
    if spawn_timeout is not None:
        try:
            await spawn_timeout(getattr(session, "session_max_duration_s", 1800))
        except Exception:
            pass

    local = pc.localDescription
    return local.sdp if local is not None else None

__all__ = [
    "EventSink",
    "OutboundAudioSpec",
    "OutboundOpusTrack",
    "RealtimeContext",
    "get_context",
    "maybe_handle_offer",
    "set_context",
    "webrtc_session_count",
    "_inbound_consumer",
    "_inbound_track_pump",
]
