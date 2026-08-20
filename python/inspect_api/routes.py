from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

import oapi
from oapi import kind as oapi_kind

from . import registry
from .audio_store import Channel, wav_header
from .retention import session_dir
from .types import AudioQuery, SessionHistoryEntry, SessionMeta

logger = logging.getLogger(__name__)

router = APIRouter()

def sanitize_sid(sid: str) -> str | None:
    if not sid or len(sid) > 64:
        return None
    for ch in sid:
        if not (ch.isascii() and (ch.isalnum() or ch in ("_", "-"))):
            return None
    return sid

@router.get("/v1/inspect/sessions", response_model=list[SessionMeta])
async def inspect_sessions() -> list[SessionMeta]:
    return registry.list_meta()

@router.get("/v1/inspect/sessions/history", response_model=list[SessionHistoryEntry])
async def inspect_history() -> list[SessionHistoryEntry]:
    out: list[SessionHistoryEntry] = []
    sd = session_dir()
    if sd is None or not sd.exists():
        return out
    try:
        entries = list(sd.iterdir())
    except OSError as err:
        logger.warning("read session dir %s: %s", sd, err)
        return out
    for p in entries:
        if not p.is_file():
            continue
        if p.suffix != ".ndjson":
            continue
        stem = p.stem
        try:
            st = p.stat()
        except OSError:
            continue
        out.append(
            SessionHistoryEntry(
                id=stem,
                size_bytes=st.st_size,
                mtime=st.st_mtime,
            )
        )
    out.sort(key=lambda e: e.mtime, reverse=True)
    return out

@router.get("/v1/inspect/sessions/history/{sid}")
async def inspect_history_stream(sid: str) -> Response:
    clean = sanitize_sid(sid)
    if clean is None:
        oapi.raise_openai_error(
            400, "invalid session id", oapi_kind.INVALID_REQUEST,
            code="invalid_session_id",
        )
    sd = session_dir()
    if sd is None:
        oapi.raise_openai_error(
            404, "session not found", oapi_kind.NOT_FOUND,
            code="session_not_found",
        )
    path = sd / f"{clean}.ndjson"
    if not path.is_file():
        oapi.raise_openai_error(
            404, "session not found", oapi_kind.NOT_FOUND,
            code="session_not_found",
        )
    try:
        body = path.read_bytes()
    except OSError as err:
        logger.warning("read history ndjson %s: %s", path, err)
        oapi.raise_openai_error(
            500, "audio read failed", oapi_kind.SERVER,
            code="audio_read_failed",
        )
    return Response(content=body, media_type="application/x-ndjson")

def _try_live_slice(sid: str, channel: Channel, from_ms: int, to_ms: int) -> bytes | None:
    audio_store = registry.get_audio_store(sid)
    if audio_store is None:
        return None
    try:
        return audio_store.slice(channel, from_ms, to_ms)
    except Exception:
        return None

def _try_disk_slice(sid: str, channel: Channel, from_ms: int, to_ms: int) -> bytes | None:
    sd = session_dir()
    if sd is None:
        return None
    raw = sd / f"{sid}.audio_{channel.as_str()}.raw"
    if not raw.is_file():
        return None
    sidecar = sd / f"{sid}.audio.json"
    offset_ms = 0
    try:
        if sidecar.is_file():
            data = json.loads(sidecar.read_text())
            tracks = data.get("tracks", {})
            entry = tracks.get(channel.as_str(), {})
            offset_ms = int(entry.get("offset_ms", 0))
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        offset_ms = 0
    adj_from = max(0, from_ms - offset_ms)
    adj_to = max(0, to_ms - offset_ms) if to_ms > 0 else 0
    sr = channel.sample_rate()
    try:
        all_bytes = raw.read_bytes()
    except OSError:
        return None
    if adj_to == 0:
        start = (adj_from * sr * 2) // 1000
        if start >= len(all_bytes):
            return b""
        return all_bytes[start:]
    start = (adj_from * sr * 2) // 1000
    end = (adj_to * sr * 2) // 1000
    end = min(end, len(all_bytes))
    if start >= end:
        return b""
    return all_bytes[start:end]

@router.get("/v1/inspect/sessions/{sid}/audio")
async def inspect_audio(
    sid: str,
    channel: str = Query(...),
    from_ms: int = Query(0),
    to_ms: int = Query(0),
) -> Response:
    clean = sanitize_sid(sid)
    if clean is None:
        oapi.raise_openai_error(
            400, "invalid session id", oapi_kind.INVALID_REQUEST,
            code="invalid_session_id",
        )
    chan = Channel.parse(channel)
    if chan is None:
        oapi.raise_openai_error(
            400, "channel must be mic_in or tts_out", oapi_kind.INVALID_REQUEST,
            code="invalid_channel",
        )
    pcm = _try_live_slice(clean, chan, from_ms, to_ms)
    if pcm is None:
        pcm = _try_disk_slice(clean, chan, from_ms, to_ms)
    if pcm is None:
        oapi.raise_openai_error(
            404, "no audio for session", oapi_kind.NOT_FOUND,
            code="session_not_found",
        )
    num_samples = len(pcm) // 2
    body = wav_header(num_samples, chan.sample_rate()) + pcm
    return Response(content=body, media_type="audio/wav")

async def _replay_history_to_socket(socket: WebSocket, sid: str) -> None:
    sd = session_dir()
    if sd is None:
        return
    path = sd / f"{sid}.ndjson"
    try:
        bytes_data = path.read_bytes()
    except OSError:
        return
    for line in bytes_data.split(b"\n"):
        if not line:
            continue
        try:
            await socket.send_bytes(line)
        except (WebSocketDisconnect, RuntimeError):
            return

@router.websocket("/v1/inspect/{sid}/stream")
async def inspect_stream_ws(websocket: WebSocket, sid: str) -> None:
    if sanitize_sid(sid) is None:
        await websocket.close(code=1008)
        return
    await websocket.accept()
    relay = registry.get_relay(sid)
    if relay is None:
        await _replay_history_to_socket(websocket, sid)
        await websocket.close()
        return
    sub = relay.subscribe(loop=asyncio.get_running_loop())
    try:
        for line in sub.snapshot:
            try:
                await websocket.send_bytes(_strip_trailing_nl(line))
            except (WebSocketDisconnect, RuntimeError):
                return
        while True:
            recv_task = asyncio.create_task(sub.queue.get())
            sock_task = asyncio.create_task(websocket.receive())
            done, pending = await asyncio.wait(
                {recv_task, sock_task}, return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            if recv_task in done:
                line = recv_task.result()
                if line is None:
                    return
                try:
                    await websocket.send_bytes(_strip_trailing_nl(line))
                except (WebSocketDisconnect, RuntimeError):
                    return
            if sock_task in done:
                try:
                    msg = sock_task.result()
                except (WebSocketDisconnect, RuntimeError):
                    return
                if msg.get("type") == "websocket.disconnect":
                    return
    finally:
        relay.unsubscribe(sub.queue)
        try:
            await websocket.close()
        except RuntimeError:
            pass

def _strip_trailing_nl(line: bytes) -> bytes:
    if line.endswith(b"\n"):
        return line[:-1]
    return line

def _ensure_session_dir() -> Path | None:
    return session_dir()
