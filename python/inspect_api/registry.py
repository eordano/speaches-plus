from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from .audio_store import AudioStore
from .relay import InspectorRelay
from .types import SessionMeta

@dataclass
class _Entry:
    relay: InspectorRelay
    created_at: float
    model: str
    state_fn: Callable[[], str]
    audio_store: AudioStore | None = None

_REGISTRY: dict[str, _Entry] = {}
_LOCK = threading.Lock()

def register(
    session_id: str,
    relay: InspectorRelay,
    model: str,
    state_fn: Callable[[], str],
    audio_store: AudioStore | None = None,
) -> None:
    entry = _Entry(
        relay=relay,
        created_at=time.time(),
        model=str(model),
        state_fn=state_fn,
        audio_store=audio_store,
    )
    with _LOCK:
        _REGISTRY[session_id] = entry

def unregister(session_id: str) -> None:
    with _LOCK:
        entry = _REGISTRY.pop(session_id, None)
    if entry is not None:
        entry.relay.close()

def get_relay(session_id: str) -> InspectorRelay | None:
    with _LOCK:
        entry = _REGISTRY.get(session_id)
    return entry.relay if entry is not None else None

def get_audio_store(session_id: str) -> AudioStore | None:
    with _LOCK:
        entry = _REGISTRY.get(session_id)
    return entry.audio_store if entry is not None else None

def list_meta() -> list[SessionMeta]:
    with _LOCK:
        items = list(_REGISTRY.items())
    out: list[SessionMeta] = []
    for sid, e in items:
        try:
            state = e.state_fn()
        except Exception:
            state = "unknown"
        out.append(
            SessionMeta(
                id=sid,
                created_at=e.created_at,
                model=e.model,
                state=state,
                turn_count=e.relay.turn_count(),
                last_event_ts=e.relay.last_event_ts(),
            )
        )
    return out

def clear() -> None:
    with _LOCK:
        items = list(_REGISTRY.values())
        _REGISTRY.clear()
    for entry in items:
        entry.relay.close()
