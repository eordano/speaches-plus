from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from . import ws_defaults
from .session import Intent, RealtimeQuery, Session
from .state import TerminationReason
from .transport import OutboundAudioSpec, get_context

if TYPE_CHECKING:
    from starlette.websockets import WebSocket

log = logging.getLogger("realtime.websocket")

_active_sessions = 0
_active_sessions_lock = asyncio.Lock()
_ws_sessions: dict[str, Any] = {}
_ws_sessions_lock = asyncio.Lock()

async def _register_ws_session(session: Any) -> None:
    async with _ws_sessions_lock:
        _ws_sessions[session.id] = session

async def _drop_ws_session(session_id: str) -> None:
    async with _ws_sessions_lock:
        _ws_sessions.pop(session_id, None)

def snapshot_ws_sessions() -> dict[str, Any]:
    return dict(_ws_sessions)

def _read_env_int(key: str, fallback: int) -> int:
    import env as _env

    return _env.read_int(key, fallback)

async def _try_acquire_active_slot(cap: int) -> bool:
    global _active_sessions
    async with _active_sessions_lock:
        if _active_sessions >= cap:
            return False
        _active_sessions += 1
        return True

async def _release_active_slot() -> None:
    global _active_sessions
    async with _active_sessions_lock:
        if _active_sessions > 0:
            _active_sessions -= 1

def active_session_count() -> int:
    return _active_sessions

async def realtime_ws_endpoint(websocket: "WebSocket") -> None:
    import env as env_mod

    cap = _read_env_int(env_mod.WS_MAX_CONCURRENT_SESSIONS, ws_defaults.MAX_CONCURRENT_SESSIONS)
    if not await _try_acquire_active_slot(cap):
        await websocket.close(code=1013, reason="concurrent session cap exceeded")
        return

    try:
        await websocket.accept()
        params = dict(websocket.query_params)
        query = RealtimeQuery(
            intent=params.get("intent"),
            voice=params.get("voice"),
            model=params.get("model"),
            transcription_model=params.get("transcription_model"),
            language=params.get("language"),
        )
        intent = Intent.from_query(query)

        queue_cap = _read_env_int(env_mod.WS_OUTBOUND_QUEUE_CAP, ws_defaults.OUTBOUND_QUEUE_CAP)
        idle_timeout = _read_env_int(env_mod.WS_IDLE_TIMEOUT_S, ws_defaults.IDLE_TIMEOUT_S)

        ws_send_q: asyncio.Queue[str] = asyncio.Queue(maxsize=queue_cap)

        outbound: OutboundAudioSpec | None = None
        if intent is Intent.CONVERSATION:
            from . import AUDIO_FORMAT_DEFAULT

            outbound = OutboundAudioSpec.websocket(ws_send_q, AUDIO_FORMAT_DEFAULT)

        ctx = get_context()
        observer_factory = ctx.observer_factory if ctx is not None else None
        session = Session(
            query=query,
            intent=intent,
            outbound_audio=outbound,
            observer_factory=observer_factory,
        )
        from .transport import _attach_vad_runner
        _attach_vad_runner(session, ctx)
        await _register_ws_session(session)
        await session.attach_websocket(ws_send_q)
        await session.transition_to_active()
        await session.spawn_max_duration_timeout(session.session_max_duration_s)

        async def writer():
            ping_every = ws_defaults.PING_INTERVAL_S
            while True:
                try:
                    text = await asyncio.wait_for(ws_send_q.get(), timeout=ping_every)
                except asyncio.TimeoutError:
                    try:
                        await websocket.send_text("")
                        continue
                    except Exception:
                        break
                if text is None:
                    break
                try:
                    await websocket.send_text(text)
                except Exception as err:
                    log.warning("ws send failed: %s", err)
                    break

        writer_task = asyncio.create_task(writer())

        try:
            while True:
                try:
                    msg = await asyncio.wait_for(websocket.receive(), timeout=idle_timeout)
                except asyncio.TimeoutError:
                    log.warning("ws idle timeout: %s", session.id)
                    break
                msg_type = msg.get("type")
                if msg_type == "websocket.disconnect":
                    break
                text = msg.get("text")
                if text is not None:
                    try:
                        await session.handle_client_event("ws", text)
                    except Exception as err:
                        log.warning("ws client event handler failed: %s", err)
        finally:
            runner = getattr(session, "vad_runner", None)
            if runner is not None:
                try:
                    await runner.stop()
                except Exception as err:
                    log.debug("vad_runner stop failed: %s", err)
            await session.emit_session_done("client_closed")
            await session.transition_to_terminated_with(TerminationReason.CLIENT_CLOSED)
            await session.abort_timeout_task()
            await _drop_ws_session(session.id)
            try:
                ws_send_q.put_nowait(None)
            except asyncio.QueueFull:
                pass
            writer_task.cancel()
            try:
                await writer_task
            except (asyncio.CancelledError, Exception):
                pass
            try:
                await websocket.close()
            except Exception:
                pass
    finally:
        await _release_active_slot()
