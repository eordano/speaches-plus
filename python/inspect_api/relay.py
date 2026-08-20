from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from .constants import RELAY_CAP, REPLAY_CAP, is_error_kind
from .types import Corr, WireEvent

logger = logging.getLogger(__name__)

_MONO_START_NS = time.monotonic_ns()

def _monotonic_offset_ns() -> int:
    return time.monotonic_ns() - _MONO_START_NS

@dataclass
class InspectorSubscription:
    snapshot: list[bytes]
    queue: asyncio.Queue[bytes | None]

class InspectorRelay:
    def __init__(self, session_id: str, session_dir: Path | None) -> None:
        self.session_id = session_id
        self.session_dir = session_dir
        self._seq = 0
        self._lock = threading.Lock()
        self._replay_buffer: deque[bytes] = deque(maxlen=REPLAY_CAP)
        self._ndjson: TextIO | None = None
        self._turn_count: int = 0
        self._last_event_ts: float | None = None
        self._dropped_count: int = 0
        self._turn_id: str | None = None
        self._item_id: str | None = None
        self._response_id: str | None = None
        self._phrase_id: str | None = None
        self._subscribers: list[asyncio.Queue[bytes | None]] = []
        self._sub_loops: list[asyncio.AbstractEventLoop] = []
        self._closed = False

        if session_dir is not None:
            try:
                session_dir.mkdir(parents=True, exist_ok=True)
                ndjson_path = session_dir / f"{session_id}.ndjson"
                self._ndjson = open(ndjson_path, "ab")
            except OSError as err:
                logger.warning("open inspector ndjson failed: %s", err)
                self._ndjson = None

    def next_seq(self) -> int:
        with self._lock:
            s = self._seq
            self._seq += 1
            return s

    def turn_count(self) -> int:
        with self._lock:
            return self._turn_count

    def last_event_ts(self) -> float | None:
        with self._lock:
            return self._last_event_ts

    def dropped_count(self) -> int:
        with self._lock:
            return self._dropped_count

    def corr(self) -> Corr:
        with self._lock:
            return Corr(
                turn_id=self._turn_id,
                item_id=self._item_id,
                response_id=self._response_id,
                phrase_id=self._phrase_id,
            )

    def set_turn_id(self, v: str | None) -> None:
        with self._lock:
            self._turn_id = v

    def set_item_id(self, v: str | None) -> None:
        with self._lock:
            self._item_id = v

    def set_response_id(self, v: str | None) -> None:
        with self._lock:
            self._response_id = v

    def set_phrase_id(self, v: str | None) -> None:
        with self._lock:
            self._phrase_id = v

    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)

    def subscribe(self, loop: asyncio.AbstractEventLoop | None = None) -> InspectorSubscription:
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError as err:
                raise RuntimeError(
                    "InspectorRelay.subscribe() requires a running asyncio event loop; "
                    "call from an async context or pass loop= explicitly"
                ) from err
        queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=RELAY_CAP)
        with self._lock:
            snapshot = list(self._replay_buffer)
            self._subscribers.append(queue)
            self._sub_loops.append(loop)
        return InspectorSubscription(snapshot=snapshot, queue=queue)

    def unsubscribe(self, queue: asyncio.Queue[bytes | None]) -> None:
        with self._lock:
            for i, q in enumerate(self._subscribers):
                if q is queue:
                    del self._subscribers[i]
                    del self._sub_loops[i]
                    break

    def publish(
        self,
        lane: str,
        kind: str,
        corr_override: Corr | None,
        payload: dict[str, Any],
    ) -> None:
        ts_wall = time.time()
        ts_mono_ns = _monotonic_offset_ns()
        seq = self.next_seq()
        merged = self._merge_corr(corr_override)

        try:
            from otel import current_span_id_hex  # type: ignore[import-not-found]
            span_id = current_span_id_hex()
        except Exception:
            span_id = None

        event = WireEvent(
            session_id=self.session_id,
            seq=seq,
            ts_mono_ns=ts_mono_ns,
            ts_wall=ts_wall,
            lane=lane,
            kind=kind,
            corr=merged,
            span_id=span_id,
            payload=dict(payload),
        )

        try:
            line = json.dumps(event.to_dict(), separators=(",", ":")).encode("utf-8") + b"\n"
        except (TypeError, ValueError) as err:
            logger.warning("serialize inspector event: %s", err)
            return

        emit_mirror = lane != "error" and is_error_kind(kind)

        with self._lock:
            if lane == "turn" and kind == "turn_end":
                self._turn_count += 1
            self._last_event_ts = ts_wall
            self._replay_buffer.append(line)
            if self._ndjson is not None:
                try:
                    self._ndjson.write(line)
                    self._ndjson.flush()
                except OSError as err:
                    logger.warning("write inspector ndjson: %s", err)
            subs = list(zip(self._subscribers, self._sub_loops, strict=True))

        self._fanout(line, subs)

        if emit_mirror:
            mirror_event = self._build_error_mirror(event)
            mirror_event.seq = self.next_seq()
            try:
                mline = (
                    json.dumps(mirror_event.to_dict(), separators=(",", ":")).encode("utf-8")
                    + b"\n"
                )
            except (TypeError, ValueError):
                return
            with self._lock:
                self._replay_buffer.append(mline)
                if self._ndjson is not None:
                    try:
                        self._ndjson.write(mline)
                        self._ndjson.flush()
                    except OSError:
                        pass
                subs = list(zip(self._subscribers, self._sub_loops, strict=True))
            self._fanout(mline, subs)

    def _fanout(
        self,
        line: bytes,
        subs: list[tuple[asyncio.Queue[bytes | None], asyncio.AbstractEventLoop]],
    ) -> None:
        for q, loop in subs:
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(self._queue_put_nowait, q, line)
                else:
                    try:
                        q.put_nowait(line)
                    except asyncio.QueueFull:
                        with self._lock:
                            self._dropped_count += 1
            except Exception:
                with self._lock:
                    self._dropped_count += 1

    @staticmethod
    def _queue_put_nowait(q: asyncio.Queue[bytes | None], line: bytes) -> None:
        try:
            q.put_nowait(line)
        except asyncio.QueueFull:
            pass

    def _merge_corr(self, override: Corr | None) -> Corr:
        base = self.corr()
        if override is None:
            return base
        return Corr(
            turn_id=override.turn_id if override.turn_id is not None else base.turn_id,
            item_id=override.item_id if override.item_id is not None else base.item_id,
            response_id=override.response_id if override.response_id is not None else base.response_id,
            phrase_id=override.phrase_id if override.phrase_id is not None else base.phrase_id,
        )

    def _build_error_mirror(self, origin: WireEvent) -> WireEvent:
        error_text_raw = origin.payload.get("error") or origin.payload.get("reason")
        if isinstance(error_text_raw, str):
            error_text = error_text_raw
        elif error_text_raw is None:
            error_text = origin.kind
        else:
            error_text = json.dumps(error_text_raw)

        payload: dict[str, Any] = {
            "lane": origin.lane,
            "origin_seq": origin.seq,
            "origin_kind": origin.kind,
            "error": error_text,
            "severity": "error",
        }
        return WireEvent(
            session_id=origin.session_id,
            seq=0,
            ts_mono_ns=origin.ts_mono_ns,
            ts_wall=origin.ts_wall,
            lane="error",
            kind="raised",
            corr=Corr(
                turn_id=origin.corr.turn_id,
                item_id=origin.corr.item_id,
                response_id=origin.corr.response_id,
                phrase_id=origin.corr.phrase_id,
            ),
            span_id=origin.span_id,
            payload=payload,
        )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._ndjson is not None:
                try:
                    self._ndjson.flush()
                    self._ndjson.close()
                except OSError:
                    pass
                self._ndjson = None
            subs = list(zip(self._subscribers, self._sub_loops, strict=True))
            self._subscribers.clear()
            self._sub_loops.clear()
        for q, loop in subs:
            try:
                if loop.is_running():
                    loop.call_soon_threadsafe(self._queue_put_nowait, q, None)
                else:
                    try:
                        q.put_nowait(None)
                    except asyncio.QueueFull:
                        pass
            except Exception:
                pass
