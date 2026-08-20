from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from realtime.observer import SessionObserver

from . import registry
from .audio_store import AudioStore
from .relay import InspectorRelay
from .retention import session_dir

log = logging.getLogger(__name__)

_UNSET = object()

class InspectObserver:
    def __init__(self, sid: str = "") -> None:
        self.sid = sid
        self.relay: InspectorRelay | None = None
        self.audio_store: AudioStore | None = None
        self._terminated = False
        self._started = False

    def on_session_start(self, sid: str, meta: dict[str, Any]) -> None:
        if self._started:
            return
        self._started = True
        self.sid = sid
        try:
            sdir = session_dir()
            self.relay = InspectorRelay(sid, sdir)
            self.audio_store = AudioStore(sid, sdir) if sdir is not None else None
            model_label = str(meta.get("intent_label") or "")
            state_fn = meta.get("state_fn")
            if not callable(state_fn):
                state_fn = lambda: "unknown"
            registry.register(sid, self.relay, model_label, state_fn, self.audio_store)
        except Exception as err:
            log.warning("inspect_api register failed: %s", err)
            self.relay = None
            self.audio_store = None

    def on_session_end(self, sid: str) -> None:
        if self._terminated:
            return
        self._terminated = True
        try:
            if self.audio_store is not None:
                self.audio_store.close()
        except Exception as err:
            log.warning("inspect audio_store close failed: %s", err)
        try:
            registry.unregister(sid)
        except Exception as err:
            log.warning("inspect unregister failed: %s", err)
        self.relay = None
        self.audio_store = None

    def on_outbound_event(self, ev: Any) -> None:
        relay = self.relay
        if relay is None:
            return
        try:
            payload = ev.to_json() if hasattr(ev, "to_json") else (ev if isinstance(ev, dict) else {"raw": ev})
            kind = ev.type_name() if hasattr(ev, "type_name") else (payload.get("type") if isinstance(payload, dict) else "unknown")
            if isinstance(payload, dict) and isinstance(kind, str):
                self._update_corr_from_event(kind, payload)
            relay.publish("wire", "out", None, payload if isinstance(payload, dict) else {"raw": payload})
        except Exception as err:
            log.warning("inspect publish outbound failed: %s", err)

    def on_outbound_event_dict(self, ev: dict[str, Any]) -> None:
        relay = self.relay
        if relay is None:
            return
        try:
            data = ev if isinstance(ev, dict) else {"raw": ev}
            kind = data.get("type") if isinstance(data, dict) else "unknown"
            if isinstance(kind, str):
                self._update_corr_from_event(kind, data)
            relay.publish("wire", "out", None, dict(data))
        except Exception as err:
            log.warning("inspect publish outbound_dict failed: %s", err)

    def on_inbound_event(self, ev_kind: str, payload: dict[str, Any], raw_text: str) -> None:
        relay = self.relay
        if relay is None:
            return
        try:
            kind = ev_kind if ev_kind else "unknown"
            data = dict(payload) if isinstance(payload, dict) else {"raw": raw_text}
            data.setdefault("type", kind)
            relay.publish("wire", "in", None, data)
        except Exception as err:
            log.warning("inspect publish inbound failed: %s", err)

    def on_error(self, code: str, message: str, event_id: str | None, param: str | None) -> None:
        relay = self.relay
        if relay is None:
            return
        try:
            relay.publish(
                "error",
                "raised",
                None,
                {"code": code, "error": message, "param": param, "event_id": event_id},
            )
        except Exception as err:
            log.warning("inspect publish error lane failed: %s", err)

    def on_inbound_audio_pcm16(self, pcm: bytes) -> None:
        store = self.audio_store
        if store is None or not pcm:
            return
        try:
            store.append_mic_in_pcm16(pcm)
        except Exception as err:
            log.warning("inspect mic_in capture failed: %s", err)

    def on_outbound_audio_pcm16(self, pcm: bytes) -> None:
        store = self.audio_store
        if store is None or not pcm:
            return
        try:
            store.append_tts_out_pcm16(pcm)
        except Exception as err:
            log.warning("inspect tts_out capture failed: %s", err)

    def on_inbound_audio_f32(self, samples: Any) -> None:
        store = self.audio_store
        if store is None:
            return
        try:
            seq = samples if isinstance(samples, (list, bytes, bytearray)) else list(samples)
            if isinstance(seq, list) and not seq:
                return
            store.append_mic_in_f32(seq)
        except Exception as err:
            log.warning("inspect mic_in f32 capture failed: %s", err)

    def on_outbound_audio_f32(self, samples: Any) -> None:
        store = self.audio_store
        if store is None:
            return
        try:
            seq = samples if isinstance(samples, (list, bytes, bytearray)) else list(samples)
            if isinstance(seq, list) and not seq:
                return
            store.append_tts_out_f32(seq)
        except Exception as err:
            log.warning("inspect tts_out f32 capture failed: %s", err)

    def on_correlation(
        self,
        *,
        response_id: Any = _UNSET,
        item_id: Any = _UNSET,
        turn_id: Any = _UNSET,
        phrase_id: Any = _UNSET,
    ) -> None:
        relay = self.relay
        if relay is None:
            return
        try:
            if response_id is not _UNSET:
                relay.set_response_id(response_id)
            if item_id is not _UNSET:
                relay.set_item_id(item_id)
            if turn_id is not _UNSET:
                relay.set_turn_id(turn_id)
            if phrase_id is not _UNSET:
                relay.set_phrase_id(phrase_id)
        except Exception as err:
            log.warning("inspect set correlation failed: %s", err)

    def _update_corr_from_event(self, kind: str, payload: dict[str, Any]) -> None:
        relay = self.relay
        if relay is None:
            return
        try:
            if kind == "response.created":
                resp = payload.get("response")
                if isinstance(resp, dict):
                    rid = resp.get("id")
                    if isinstance(rid, str):
                        relay.set_response_id(rid)
            elif kind == "response.done":
                relay.set_response_id(None)
                relay.set_item_id(None)
            elif kind == "conversation.item.added":
                item = payload.get("item")
                if isinstance(item, dict):
                    iid = item.get("id")
                    if isinstance(iid, str):
                        relay.set_item_id(iid)
            elif kind == "input_audio_buffer.speech_started":
                iid = payload.get("item_id")
                if isinstance(iid, str):
                    relay.set_item_id(iid)
            elif kind == "input_audio_buffer.committed":
                iid = payload.get("item_id")
                if isinstance(iid, str):
                    relay.set_item_id(iid)
        except Exception as err:
            log.warning("inspect corr update failed: %s", err)

def make_observer_factory() -> Callable[[str], SessionObserver]:
    def _factory(sid: str) -> SessionObserver:
        return InspectObserver(sid)

    return _factory

__all__ = ["InspectObserver", "make_observer_factory"]
