from __future__ import annotations

import asyncio
import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import (
    AUDIO_FORMAT_DEFAULT,
    TURN_DETECTION_NONE,
    TURN_DETECTION_SERVER_VAD,
    audio_defaults,
    buffer_defaults,
    eou_defaults,
    response_defaults,
    session_defaults,
    turn_detection,
    wire_defaults,
)
from eou.integrated import IntegratedEouBackend, IntegratedVerdict
from eou.loader import EouConfig
from eou.types import EouKind, EouModel, StubEouModel

from .eou_integrated import IntegratedVerdictAction
from .errors import code as errcode
from .events import (
    item_to_json,
    make_error_event,
    make_response_done,
    parse_client_event,
)
from .state import (
    ConversationItem,
    InvariantViolation,
    ItemContent,
    ItemRole,
    ItemStatus,
    PendingBargein,
    PredictedRunner,
    PredictedSharedState,
    RespPhase,
    SealedBuffer,
    SessionPhase,
    SessionState,
    TerminationReason,
    Topic,
    VadPhase,
    _AtomicU64,
    apply_truncate_to_conversation,
    check_state,
)
from .observer import NullObserver, SessionObserver
from .transport import EventSink, OutboundAudioSpec
from .wire import OutboundEvent, ResponsePayload, ResponseStatus, ResponseStatusReason

log = logging.getLogger("realtime.session")

class Intent(enum.Enum):
    TRANSCRIPTION = "transcription"
    CONVERSATION = "conversation"

    @classmethod
    def from_query(cls, q: RealtimeQuery) -> Intent:
        if q.intent == "conversation":
            return cls.CONVERSATION
        return cls.TRANSCRIPTION

@dataclass
class RealtimeQuery:
    intent: str | None = None
    voice: str | None = None
    model: str | None = None
    transcription_model: str | None = None
    language: str | None = None

class TurnDetectionKind(enum.Enum):
    SERVER_VAD = "server_vad"
    NONE = "none"

    def as_str(self) -> str:
        return self.value

    @classmethod
    def parse(cls, s: str) -> TurnDetectionKind | None:
        if s == TURN_DETECTION_SERVER_VAD:
            return cls.SERVER_VAD
        if s == TURN_DETECTION_NONE:
            return cls.NONE
        return None

@dataclass
class TurnDetectionConfig:
    kind: TurnDetectionKind = TurnDetectionKind.SERVER_VAD
    threshold: float = turn_detection.THRESHOLD
    neg_threshold: float | None = None
    min_speech_duration_ms: int = 100
    prefix_padding_ms: int = turn_detection.PREFIX_PADDING_MS
    silence_duration_ms: int = turn_detection.SILENCE_DURATION_MS
    barge_in_delay_ms: int = turn_detection.BARGE_IN_DELAY_MS
    create_response: bool = turn_detection.CREATE_RESPONSE

    @classmethod
    def from_env(cls) -> TurnDetectionConfig:
        import env as env_mod

        raw = env_mod.read_int(env_mod.BARGE_IN_DELAY_MS, turn_detection.BARGE_IN_DELAY_MS)
        bidm = min(raw, turn_detection.BARGE_IN_DELAY_MS_MAX)
        return cls(barge_in_delay_ms=bidm)

    def snapshot(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": self.kind.as_str(),
            "threshold": float(self.threshold),
            "min_speech_duration_ms": int(self.min_speech_duration_ms),
            "prefix_padding_ms": int(self.prefix_padding_ms),
            "silence_duration_ms": int(self.silence_duration_ms),
            "barge_in_delay_ms": int(self.barge_in_delay_ms),
            "create_response": bool(self.create_response),
        }
        if self.neg_threshold is not None:
            out["neg_threshold"] = float(self.neg_threshold)
        return out

def validate_session_max_duration_s(n: int) -> int:
    if n < 1 or n > session_defaults.MAX_DURATION_HARD_CAP_S:
        raise ValueError(f"session_max_duration_s must be in [1, {session_defaults.MAX_DURATION_HARD_CAP_S}]")
    return n

def compute_remaining_timeout_ms(new_secs: int, created_at_ms: int, now_ms: int) -> int:
    elapsed = max(0, now_ms - created_at_ms)
    remaining_ms = new_secs * 1000 - elapsed
    return max(0, remaining_ms)

class _CancelReason(enum.Enum):
    CLIENT_CANCELLED = "client_cancelled"
    BARGE_IN = "barge_in"

@dataclass
class CancelledSnapshot:
    response_id: str
    assistant_item_id: str
    transcript: str
    played_ms: int
    transcript_done_emitted: bool = False
    audio_done_emitted: bool = False

class FailReason(enum.Enum):
    LLM_ERROR = "llm_error"
    TTS_ERROR = "tts_error"
    CLIENT_TOO_SLOW = "client_too_slow"

    def to_status_reason(self) -> ResponseStatusReason:
        return {
            FailReason.LLM_ERROR: ResponseStatusReason.LLM_ERROR,
            FailReason.TTS_ERROR: ResponseStatusReason.TTS_ERROR,
            FailReason.CLIENT_TOO_SLOW: ResponseStatusReason.CLIENT_TOO_SLOW,
        }[self]

def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"

def _clip01(v: float) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return 0.0
    if x != x:
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x

@dataclass
class _IdSource:
    def session(self) -> str:
        return _gen_id("sess")

    def item(self) -> str:
        return _gen_id("item")

    def response(self) -> str:
        return _gen_id("resp")

    def event(self) -> str:
        return _gen_id("evt")

class Session:
    def __init__(
        self,
        query: RealtimeQuery,
        intent: Intent,
        outbound_audio: OutboundAudioSpec | None = None,
        instructions: str | None = None,
        observer: SessionObserver | None = None,
        observer_factory: Any = None,
    ):
        self.id: str = _gen_id("sess")
        self.query = query
        self.intent = intent
        self.outbound_audio = outbound_audio
        self.id_source = _IdSource()
        self.turn_detection: TurnDetectionConfig = TurnDetectionConfig.from_env()

        self.outbound_queue_cap_ms = wire_defaults.OUTBOUND_QUEUE_CAP_MS
        self.outbound_queue_cap_events = wire_defaults.OUTBOUND_QUEUE_CAP_EVENTS
        self._outbound_inflight = 0
        self.session_max_duration_s = session_defaults.MAX_DURATION_S
        self.min_speech_ms = buffer_defaults.MIN_SPEECH_MS
        self.min_speech_for_response_ms = buffer_defaults.MIN_SPEECH_FOR_RESPONSE_MS
        self.no_speech_prob_threshold: float | None = None
        self.avg_logprob_threshold: float | None = None
        self.input_audio_format = AUDIO_FORMAT_DEFAULT
        self.output_audio_format = AUDIO_FORMAT_DEFAULT
        self.voice: str | None = query.voice
        self.model: str | None = query.model
        self.transcription_model: str | None = query.transcription_model

        self._instructions = instructions
        self.last_eager_dispatch_at: float | None = None

        self.eou_config: EouConfig = EouConfig.from_env()
        self.eou_model: EouModel = StubEouModel()
        self._integrated_backend: IntegratedEouBackend | None = None
        self._transcribe: Any = None

        self.played_ms_ref: list[int] = [0]
        self._turn_id: str | None = None
        self._phrase_id: str | None = None

        self._timeout_task: asyncio.Task | None = None
        self.state = SessionState()
        self.state.instructions = instructions
        self.state.sealed_buffer_retention_count = buffer_defaults.SEALED_BUFFER_RETENTION_COUNT
        self._state_lock = asyncio.Lock()
        self._created_at_ms: int = 0

        resolved_observer: SessionObserver
        if observer is not None:
            resolved_observer = observer
        elif observer_factory is not None:
            try:
                resolved_observer = observer_factory(self.id)
            except Exception as err:
                log.warning("observer_factory failed: %s", err)
                resolved_observer = NullObserver()
        else:
            resolved_observer = NullObserver()
        self._observer = resolved_observer
        self._session_ended = False
        try:
            intent_label = (
                "conversation" if intent is Intent.CONVERSATION else "transcription"
            )
            self._observer.on_session_start(
                self.id,
                {"intent_label": intent_label, "state_fn": self._state_label},
            )
        except Exception as err:
            log.warning("observer on_session_start failed: %s", err)

    async def transition_to_active(self) -> None:
        async with self._state_lock:
            if self.state.session.is_pending():
                now_ms = int(time.time() * 1000)
                self._created_at_ms = now_ms
                self.state.session = SessionPhase.active(now_ms)
                _check_or_react(self, self.state)

    async def transition_to_terminated(self) -> None:
        await self.transition_to_terminated_with(TerminationReason.CLIENT_CLOSED)

    async def transition_to_terminated_with(self, reason: TerminationReason) -> None:
        async with self._state_lock:
            if self.state.session.is_terminated():
                return
            self.state.session = SessionPhase.terminated(reason)
            if self.state.timeout_task is not None:
                self.state.timeout_task.cancel()
                self.state.timeout_task = None
            if self.state.commit_timer is not None:
                self.state.commit_timer.cancel()
                self.state.commit_timer = None
            if self.state.bargein_task is not None:
                self.state.bargein_task.cancel()
                self.state.bargein_task = None
            if self.state.current_response is not None and self.state.current_response.handle is not None:
                self.state.current_response.handle.cancel()
        self._notify_session_end()

    async def attach_data_channel(self, dc: Any) -> None:
        async with self._state_lock:
            self.state.event_sink = EventSink.data_channel_sink(dc)

    async def attach_websocket(self, ws_send: asyncio.Queue) -> None:
        async with self._state_lock:
            self.state.event_sink = EventSink.websocket_sink(ws_send)
        await self.emit_session_created()

    async def emit_session_created(self) -> None:
        sink = await self.event_sink()
        if sink is None:
            return
        snapshot = await self.current_session_view()
        ev = OutboundEvent.session_created(snapshot)
        await sink.send_value(ev)
        self._publish_outbound(ev)

    async def event_sink(self) -> EventSink | None:
        async with self._state_lock:
            return self.state.event_sink

    async def current_session_view(self) -> dict[str, Any]:
        async with self._state_lock:
            modalities = ["audio", "text"] if self.intent is Intent.CONVERSATION else ["text"]
            input_audio_transcription: dict[str, Any] = {
                "model": self.transcription_model or self.model or "",
            }
            if self.query.language:
                input_audio_transcription["language"] = self.query.language
            else:
                input_audio_transcription["language"] = None
            from .v2_compat import enrich_session_view
            view = {
                "id": self.id,
                "object": "realtime.session",
                "model": self.model or "",
                "voice": self.voice,
                "instructions": self.state.instructions,
                "modalities": modalities,
                "input_audio_format": self.input_audio_format,
                "output_audio_format": self.output_audio_format,
                "input_audio_transcription": input_audio_transcription,
                "turn_detection": self.turn_detection.snapshot(),
                "min_speech_ms": int(self.min_speech_ms),
                "min_speech_for_response_ms": int(self.min_speech_for_response_ms),
                "session_max_duration_s": int(self.session_max_duration_s),
                "sealed_buffer_retention_count": int(self.state.sealed_buffer_retention_count),
            }
            return enrich_session_view(view)

    async def emit(self, ev: OutboundEvent) -> None:
        sink = await self.event_sink()
        if sink is None:
            return
        async with self._state_lock:
            if ev.topic() is Topic.RESPONSE and self.state.resp.is_predicted():
                return
        await sink.send_value(ev)
        self._publish_outbound(ev)

    async def emit_event(self, ev: dict[str, Any]) -> None:
        sink = await self.event_sink()
        if sink is None:
            return
        await sink.send_value(ev)
        self._publish_outbound_dict(ev)

    async def emit_session_done(self, reason: str) -> None:
        await self.emit(OutboundEvent.session_done(reason))

    async def spawn_max_duration_timeout(self, timeout_s: int) -> None:
        async def _runner():
            try:
                await asyncio.sleep(timeout_s)
            except asyncio.CancelledError:
                return
            await self.emit_session_done("max_duration")
            await self.transition_to_terminated_with(TerminationReason.MAX_DURATION)

        async with self._state_lock:
            if self.state.timeout_task is not None:
                self.state.timeout_task.cancel()
            self.state.timeout_task = asyncio.create_task(_runner())

    async def abort_timeout_task(self) -> None:
        async with self._state_lock:
            if self.state.timeout_task is not None:
                self.state.timeout_task.cancel()
                self.state.timeout_task = None

    async def reschedule_max_duration_timeout(self) -> None:
        now_ms = int(time.time() * 1000)
        remaining_ms = compute_remaining_timeout_ms(
            self.session_max_duration_s, self._created_at_ms, now_ms
        )
        await self.spawn_max_duration_timeout(max(0, remaining_ms // 1000))

    async def instructions(self) -> str | None:
        async with self._state_lock:
            return self.state.instructions

    async def set_instructions(self, instructions: str | None) -> None:
        async with self._state_lock:
            self.state.instructions = instructions

    async def build_chat_messages(self, instructions: str | None) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if instructions:
            msgs.append({"role": "system", "content": instructions})
        async with self._state_lock:
            for item in self.state.conversation:
                t = item.transcript()
                if not t:
                    continue
                role = item.role.as_str()
                msgs.append({"role": role, "content": t})
        return msgs

    async def build_eou_context(self, k: int) -> str:
        async with self._state_lock:
            tail: list[str] = []
            count = 0
            for item in reversed(self.state.conversation):
                t = item.transcript()
                if not t:
                    continue
                tail.append(t)
                count += 1
                if count >= k:
                    break
            tail.reverse()
            return " ".join(tail)

    async def append_assistant_item(self, item_id: str, transcript: str, audio_ms: int) -> None:
        async with self._state_lock:
            self.state.conversation.append(
                ConversationItem.new_assistant_audio(item_id, transcript, int(audio_ms))
            )

    async def complete_user_item_transcript(self, item_id: str, transcript: str) -> None:
        async with self._state_lock:
            for item in self.state.conversation:
                if item.id == item_id and item.content.is_user_audio():
                    item.status = ItemStatus.COMPLETED
                    item.content.transcript = transcript
                    return

    async def mark_user_item_incomplete(self, item_id: str) -> None:
        async with self._state_lock:
            for item in self.state.conversation:
                if item.id == item_id and item.content.is_user_audio():
                    item.status = ItemStatus.INCOMPLETE
                    return

    async def apply_truncate_to_assistant_item(self, snap: CancelledSnapshot) -> None:
        async with self._state_lock:
            apply_truncate_to_conversation(
                self.state.conversation,
                snap.assistant_item_id,
                snap.played_ms,
                snap.transcript,
            )

    async def cancel_current_response(self) -> CancelledSnapshot | None:
        async with self._state_lock:
            if not self.state.resp.is_active() or self.state.resp.is_predicted():
                return None
            response_id = self.state.resp.id or ""
            assistant_item_id = self.state.resp.item_id or ""
            played_ms = self.state.resp.played_ms.load() if self.state.resp.played_ms else 0
            transcript = ""
            runtime = self.state.current_response
            transcript_done = False
            audio_done = False
            if runtime is not None:
                async with runtime.transcript_lock:
                    transcript = "".join(runtime.transcript_so_far)
                transcript_done = bool(getattr(runtime, "_transcript_done_emitted", False))
                audio_done = bool(getattr(runtime, "_audio_done_emitted", False))
            try:
                self.state.resp_retire_to_none()
            except InvariantViolation:
                pass
            if runtime is not None and runtime.handle is not None:
                runtime.handle.cancel()
        return CancelledSnapshot(
            response_id=response_id,
            assistant_item_id=assistant_item_id,
            transcript=transcript,
            played_ms=int(played_ms),
            transcript_done_emitted=transcript_done,
            audio_done_emitted=audio_done,
        )

    async def clear_response_if_matches(self, response_id: str) -> None:
        async with self._state_lock:
            if self.state.resp.id == response_id and self.state.resp.is_active():
                try:
                    self.state.resp_retire_to_none()
                except InvariantViolation:
                    pass

    async def register_response(
        self,
        response_id: str,
        handle: asyncio.Task | None,
        played_ms: _AtomicU64,
        assistant_item_id: str,
        transcript_so_far: list[str],
    ) -> None:
        from .state import ResponseRuntime

        async with self._state_lock:
            runtime = ResponseRuntime(handle=handle, transcript_so_far=transcript_so_far)
            try:
                self.state.resp_create_from_none(response_id, assistant_item_id, runtime)
            except InvariantViolation as v:
                log.warning("register_response invariant: %s", v)

    async def mark_streaming(self, response_id: str) -> None:
        async with self._state_lock:
            if self.state.resp.id == response_id and self.state.resp.tag.value == "Created":
                played = self.state.resp.played_ms or _AtomicU64()
                try:
                    self.state.resp_advance_to_streaming(played)
                except InvariantViolation as v:
                    log.warning("mark_streaming invariant: %s", v)

    async def install_commit_timer(self, task: asyncio.Task) -> None:
        async with self._state_lock:
            if self.state.commit_timer is not None:
                self.state.commit_timer.cancel()
            self.state.commit_timer = task

    async def clear_commit_timer(self) -> None:
        async with self._state_lock:
            if self.state.commit_timer is not None:
                self.state.commit_timer.cancel()
                self.state.commit_timer = None

    async def set_pending_bargein(self, pending: PendingBargein) -> None:
        async with self._state_lock:
            self.state.pending_bargein = pending

    async def take_pending_bargein_if(self, item_id: str) -> PendingBargein | None:
        async with self._state_lock:
            if self.state.pending_bargein and self.state.pending_bargein.item_id == item_id:
                pending = self.state.pending_bargein
                self.state.pending_bargein = None
                return pending
            return None

    async def handle_client_event(self, source: str, raw: bytes | str) -> None:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        ev_type, obj = parse_client_event(text)
        self._publish_inbound(ev_type, obj, text)
        if ev_type is None and not obj:
            await self._emit_error(errcode.INVALID_REQUEST_ERROR, "invalid_request: malformed JSON", None, None)
            return
        if ev_type is None:
            t = obj.get("type") if isinstance(obj, dict) else None
            from .v2_compat import is_known_v2_noop_event
            if isinstance(t, str) and is_known_v2_noop_event(t):
                return
            await self._emit_error(
                errcode.UNKNOWN_EVENT_TYPE,
                f"unknown event type: {t!r}",
                obj.get("event_id") if isinstance(obj, dict) else None,
                None,
            )
            return
        await _dispatch_client_event(self, ev_type, obj)

    async def set_integrated_backend(self, backend: IntegratedEouBackend) -> None:
        async with self._state_lock:
            old = self._integrated_backend
            self._integrated_backend = backend
        if old is not None:
            try:
                old.reset()
            except Exception as err:
                log.warning("integrated_backend reset failed: %s", err)

    async def handle_integrated_verdict(
        self, verdict: IntegratedVerdict
    ) -> IntegratedVerdictAction:
        cfg = self.eou_config
        if cfg.kind is not EouKind.INTEGRATED:
            return IntegratedVerdictAction.IGNORED

        p_eot = _clip01(verdict.p_eot)
        p_eager = _clip01(verdict.p_eager_eot)

        try:
            self._observer.on_eou_scored(
                self.id,
                kind=cfg.kind.as_str(),
                score=p_eot,
                eager_score=p_eager,
                threshold=cfg.eot_threshold,
                language=self.query.language,
                input_chars=len(verdict.transcript_so_far),
                input_audio_ms=None,
                delay_ms=0,
                elapsed_ms=0,
                cancelled_by="none",
                hard_cap_fired=False,
            )
        except Exception as err:
            log.warning("observer on_eou_scored failed: %s", err)

        if p_eot >= cfg.eot_threshold:
            return IntegratedVerdictAction.COMMIT

        if p_eager >= cfg.eager_eot_threshold:
            predicted_id = self.id_source.response()
            predicted_item_id = self.id_source.item()
            async with self._state_lock:
                if not self.state.resp.is_active():
                    try:
                        epoch = self.state.resp_start_predicted(
                            predicted_id, predicted_item_id, p_eager, None
                        )
                    except InvariantViolation:
                        return IntegratedVerdictAction.NONE
                    try:
                        self._observer.on_predicted_promoted(
                            self.id, response_id=predicted_id, score=p_eager
                        )
                    except Exception as err:
                        log.warning("observer on_predicted_promoted failed: %s", err)
                    _ = epoch
                    return IntegratedVerdictAction.STARTED_PREDICTED
        return IntegratedVerdictAction.NONE

    async def _emit_error(
        self, code: str, message: str, event_id: str | None, param: str | None
    ) -> None:
        sink = await self.event_sink()
        if sink is None:
            return
        ev = make_error_event(code, message, event_id, param)
        await sink.send_value(ev)
        self._publish_outbound(ev)
        try:
            self._observer.on_error(code, message, event_id, param)
        except Exception as err:
            log.warning("observer on_error failed: %s", err)

    def _state_label(self) -> str:
        try:
            phase = self.state.session
            if phase.is_pending():
                return "pending"
            if phase.is_active():
                return "active"
            if phase.is_terminated():
                return "terminated"
        except Exception:
            return "unknown"
        return "unknown"

    def _publish_outbound(self, ev: OutboundEvent) -> None:
        try:
            self._observer.on_outbound_event(ev)
        except Exception as err:
            log.warning("observer on_outbound_event failed: %s", err)

    def _publish_outbound_dict(self, payload: Any) -> None:
        try:
            self._observer.on_outbound_event_dict(payload if isinstance(payload, dict) else {"raw": payload})
        except Exception as err:
            log.warning("observer on_outbound_event_dict failed: %s", err)

    def _publish_inbound(self, ev_type: Any, obj: dict[str, Any], raw_text: str) -> None:
        try:
            kind = ev_type.value if ev_type is not None else "unknown"
            payload = dict(obj) if isinstance(obj, dict) else {}
            self._observer.on_inbound_event(kind, payload, raw_text)
        except Exception as err:
            log.warning("observer on_inbound_event failed: %s", err)

    def _notify_session_end(self) -> None:
        if self._session_ended:
            return
        self._session_ended = True
        try:
            self._observer.on_session_end(self.id)
        except Exception as err:
            log.warning("observer on_session_end failed: %s", err)

    def capture_inbound_pcm16(self, pcm: bytes) -> None:
        if not pcm:
            return
        try:
            self._observer.on_inbound_audio_pcm16(pcm)
        except Exception as err:
            log.warning("observer on_inbound_audio_pcm16 failed: %s", err)

    def capture_inbound_f32(self, samples: Any) -> None:
        try:
            self._observer.on_inbound_audio_f32(samples)
        except Exception as err:
            log.warning("observer on_inbound_audio_f32 failed: %s", err)

    def capture_outbound_pcm16(self, pcm: bytes) -> None:
        if not pcm:
            return
        try:
            self._observer.on_outbound_audio_pcm16(pcm)
        except Exception as err:
            log.warning("observer on_outbound_audio_pcm16 failed: %s", err)

    def capture_outbound_f32(self, samples: Any) -> None:
        try:
            self._observer.on_outbound_audio_f32(samples)
        except Exception as err:
            log.warning("observer on_outbound_audio_f32 failed: %s", err)

    def set_turn_id(self, value: str | None) -> None:
        self._turn_id = value
        try:
            self._observer.on_correlation(turn_id=value)
        except Exception as err:
            log.warning("observer on_correlation(turn_id) failed: %s", err)

    def set_phrase_id(self, value: str | None) -> None:
        self._phrase_id = value
        try:
            self._observer.on_correlation(phrase_id=value)
        except Exception as err:
            log.warning("observer on_correlation(phrase_id) failed: %s", err)

def _check_or_react(session: Session, state: SessionState) -> None:
    try:
        check_state(state)
    except InvariantViolation as v:
        log.error("invariant violation: %s", v)

check_or_react = _check_or_react

async def _dispatch_client_event(
    session: Session, ev_type: Any, obj: dict[str, Any]
) -> None:
    from .events import ClientEventType
    from .session_update import handle_session_update

    event_id = obj.get("event_id") if isinstance(obj, dict) else None

    if ev_type is ClientEventType.SESSION_UPDATE:
        await handle_session_update(session, obj, event_id)
        return

    if ev_type is ClientEventType.INPUT_AUDIO_BUFFER_CLEAR:
        await session.emit(OutboundEvent.buffer_cleared())
        return

    if ev_type is ClientEventType.INPUT_AUDIO_BUFFER_COMMIT:
        await session._emit_error(
            errcode.INPUT_AUDIO_BUFFER_COMMIT_EMPTY,
            "manual commit not supported in VAD-driven session",
            event_id,
            None,
        )
        return

    if ev_type is ClientEventType.INPUT_AUDIO_BUFFER_APPEND:
        b64 = obj.get("audio")
        if not isinstance(b64, str):
            await session._emit_error(
                errcode.INVALID_REQUEST_ERROR, "input_audio_buffer.append: audio missing", event_id, "audio"
            )
            return
        from .audio_in_ws import handle_audio_append

        await handle_audio_append(session, b64)
        return

    if ev_type is ClientEventType.RESPONSE_CANCEL:
        snap = await session.cancel_current_response()
        if snap is None:
            await session._emit_error(
                errcode.RESPONSE_CANCEL_NOT_ACTIVE,
                "response.cancel: no active response",
                event_id,
                None,
            )
            return
        from .events import make_cancelled_brackets

        brackets, done = make_cancelled_brackets(
            snap.response_id, snap.assistant_item_id, snap.transcript, snap.played_ms,
            ResponseStatusReason.CLIENT_CANCELLED,
            transcript_done_emitted=snap.transcript_done_emitted,
            audio_done_emitted=snap.audio_done_emitted,
        )
        for b in brackets:
            await session.emit(b)
        await session.emit(done)
        return

    if ev_type is ClientEventType.CONVERSATION_ITEM_CREATE:
        item = obj.get("item")
        if not isinstance(item, dict):
            await session._emit_error(
                errcode.INVALID_REQUEST_ERROR, "conversation.item.create: missing item", event_id, "item"
            )
            return
        item_id = item.get("id") or session.id_source.item()
        role_str = item.get("role", "user")
        from .events import extract_text_from_content

        role = ItemRole.parse(role_str) or ItemRole.USER
        text = extract_text_from_content(item.get("content")) or ""
        new_item = ConversationItem.new_text(item_id, role, ItemStatus.COMPLETED, text)
        async with session._state_lock:
            session.state.conversation.append(new_item)
        await session.emit(OutboundEvent.item_added(item_to_json(new_item)))
        return

    if ev_type is ClientEventType.CONVERSATION_ITEM_DELETE:
        item_id = obj.get("item_id")
        if not isinstance(item_id, str):
            await session._emit_error(
                errcode.INVALID_REQUEST_ERROR,
                "conversation.item.delete: missing item_id",
                event_id,
                "item_id",
            )
            return
        removed = False
        async with session._state_lock:
            before = len(session.state.conversation)
            session.state.conversation = [
                x for x in session.state.conversation if x.id != item_id
            ]
            removed = len(session.state.conversation) != before
        if not removed:
            await session._emit_error(
                errcode.INVALID_REQUEST_ERROR,
                f"conversation.item.delete: item {item_id} not found",
                event_id,
                "item_id",
            )
            return
        await session.emit(OutboundEvent.item_deleted(item_id))
        return

    if ev_type is ClientEventType.CONVERSATION_ITEM_TRUNCATE:
        item_id = obj.get("item_id")
        content_index = int(obj.get("content_index", 0))
        audio_end_ms = int(obj.get("audio_end_ms", 0))
        if not isinstance(item_id, str):
            await session._emit_error(
                errcode.INVALID_REQUEST_ERROR,
                "conversation.item.truncate: missing item_id",
                event_id,
                "item_id",
            )
            return
        async with session._state_lock:
            for it in session.state.conversation:
                if it.id == item_id and it.content.is_assistant_audio():
                    cur = it.content.audio_ms or 0
                    it.content.audio_ms = min(cur, audio_end_ms)
                    break
        await session.emit(
            OutboundEvent.item_truncated_client_ack(item_id, content_index, audio_end_ms)
        )
        return

    if ev_type is ClientEventType.CONVERSATION_ITEM_RETRIEVE:
        await session._emit_error(
            errcode.INVALID_REQUEST_ERROR,
            "conversation.item.retrieve is not yet implemented",
            event_id,
            None,
        )
        return

    if ev_type is ClientEventType.RESPONSE_CREATE:
        async with session._state_lock:
            already = session.state.resp.is_active()
        if already:
            await session._emit_error(
                errcode.RESPONSE_ALREADY_ACTIVE,
                "response.create: response already active",
                event_id,
                None,
            )
            return
        body = obj.get("response") if isinstance(obj.get("response"), dict) else {}
        override_instructions: str | None = None
        if isinstance(body, dict):
            instr = body.get("instructions")
            if isinstance(instr, str) and instr:
                override_instructions = instr
        override_user_text: str | None = None
        if isinstance(body, dict):
            inp = body.get("input")
            if isinstance(inp, list):
                from .events import extract_text_from_content

                for entry in reversed(inp):
                    if not isinstance(entry, dict):
                        continue
                    if entry.get("role") != "user":
                        continue
                    t = extract_text_from_content(entry.get("content"))
                    if t:
                        override_user_text = t
                        break
        fallback_user_text = override_user_text or ""
        if not fallback_user_text:
            async with session._state_lock:
                for it in reversed(session.state.conversation):
                    if it.role is ItemRole.USER:
                        t = it.transcript() or ""
                        if t:
                            fallback_user_text = t
                            break
        from .pipeline import commit_after_eou_with_response

        async def _run_manual_response() -> None:
            try:
                await commit_after_eou_with_response(
                    session,
                    "",
                    fallback_user_text,
                    instructions_override=override_instructions,
                )
            except Exception as err:
                log.warning("manual response.create failed: %s", err)

        asyncio.create_task(_run_manual_response(), name="manual-response-create")
        return

__all__ = [
    "CancelledSnapshot",
    "FailReason",
    "Intent",
    "RealtimeQuery",
    "Session",
    "TurnDetectionConfig",
    "TurnDetectionKind",
    "check_or_react",
    "compute_remaining_timeout_ms",
    "validate_session_max_duration_s",
]
