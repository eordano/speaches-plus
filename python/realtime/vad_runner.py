"""Per-session VAD + STT driver.

Bridges the inbound 16 kHz mono PCM stream into the Silero VAD model and the
realtime pipeline:

  inbound f32 samples ──► VadProcessor.push ──► SpeechStarted / SpeechCommitted
                                                       │
                                                       ▼
                          OutboundEvent.speech_started / speech_stopped /
                          buffer_committed / conversation.item.added /
                          conversation.item.input_audio_transcription.completed

The runner is per-session, owns its own asyncio.Queue, and processes events
in its own coroutine so the sync `_dispatch_inbound_samples` path stays
non-blocking. Set on `RealtimeContext.vad_model` + `RealtimeContext.transcribe_factory`
in `server.py` lifespan to enable; sessions that don't have a runner attached
still negotiate and stay open, they just never emit transcription events
(the pre-this-patch behavior).
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

import numpy as np

from .events import item_to_json
from .state import ConversationItem
from .wire import OutboundEvent

if TYPE_CHECKING:
    from vad.silero import SileroVad

    from .session import Session

log = logging.getLogger(__name__)

TranscribeCallable = Callable[[np.ndarray], Awaitable[str]]

_QUEUE_CAP = 256
_STOP_TIMEOUT_S = 2.0

class _SessionTurnDetectionAdapter:
    """Read-through adapter exposing the session's ``TurnDetectionConfig`` as a
    :class:`vad.silero.TurnDetectionRead`. The adapter holds a reference to the
    session, so subsequent ``session.update`` mutations are picked up on the
    next ``VadProcessor._options()`` call without any explicit notification.
    """

    def __init__(self, session: "Session"):
        from vad.constants import NEG_THRESHOLD_DELTA, NEG_THRESHOLD_FLOOR, SAMPLE_RATE

        self._session = session
        self._sample_rate = SAMPLE_RATE
        self._neg_delta = NEG_THRESHOLD_DELTA
        self._neg_floor = NEG_THRESHOLD_FLOOR

    def _cfg(self):
        return self._session.turn_detection

    def threshold(self) -> float:
        return float(self._cfg().threshold)

    def prefix_padding_samples(self) -> int:
        return int(self._cfg().prefix_padding_ms) * self._sample_rate // 1000

    def silence_duration_samples(self) -> int:
        return int(self._cfg().silence_duration_ms) * self._sample_rate // 1000

    def neg_threshold(self) -> float:
        cfg = self._cfg()
        if cfg.neg_threshold is not None:
            return float(cfg.neg_threshold)
        return max(self.threshold() - self._neg_delta, self._neg_floor)

    def min_speech_duration_ms(self) -> int:
        return int(self._cfg().min_speech_duration_ms)

    def max_speech_duration_s(self) -> float:
        from vad.constants import MAX_SPEECH_DURATION_S

        return MAX_SPEECH_DURATION_S

class VadRunner:
    def __init__(
        self,
        session: "Session",
        vad_model: "SileroVad",
        transcribe: TranscribeCallable,
    ):
        from vad.silero import VadProcessor

        self.session = session
        self.processor = VadProcessor(vad_model).with_turn_detection(
            _SessionTurnDetectionAdapter(session)
        )
        self.transcribe = transcribe
        session._transcribe = transcribe
        self._queue: asyncio.Queue[np.ndarray | None] = asyncio.Queue(maxsize=_QUEUE_CAP)
        self._task: asyncio.Task | None = None
        self._inflight: list[asyncio.Task] = []
        self._stt_lock = asyncio.Lock()

    def start(self) -> None:
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._run(), name=f"vad-runner-{self.session.id}")

    def push_samples(self, samples: np.ndarray) -> None:
        if samples is None or samples.size == 0:
            return
        try:
            self._queue.put_nowait(np.ascontiguousarray(samples, dtype=np.float32))
        except asyncio.QueueFull:
            log.warning("vad_runner queue full; dropping %d samples", samples.size)

    async def stop(self) -> None:
        if self._task is None:
            return
        try:
            self._queue.put_nowait(None)
        except asyncio.QueueFull:
            self._task.cancel()
        try:
            await asyncio.wait_for(self._task, timeout=_STOP_TIMEOUT_S)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._task.cancel()
        self._task = None
        for t in self._inflight:
            if not t.done():
                t.cancel()

    async def _run(self) -> None:
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    return
                try:
                    self.processor.push(item)
                except Exception as err:
                    log.warning("vad push failed: %s", err)
                    continue
                events = self.processor.take_events()
                for ev in events:
                    await self._handle_event(ev)
        except asyncio.CancelledError:
            return

    async def _handle_event(self, ev) -> None:
        from vad.silero import Failed, SpeechCommitted, SpeechStarted

        if isinstance(ev, SpeechStarted):
            await self._on_speech_started(ev.item_id, ev.audio_start_ms)
        elif isinstance(ev, SpeechCommitted):
            self._inflight.append(
                asyncio.create_task(
                    self._on_speech_committed(
                        ev.item_id, ev.audio_end_ms, ev.audio[: ev.speech_samples]
                    ),
                    name=f"vad-stt-{ev.item_id}",
                )
            )
        elif isinstance(ev, Failed):
            log.warning("vad failed: %s", ev.reason)

    async def _on_speech_started(self, item_id: str, audio_start_ms: int) -> None:
        from .session import Intent

        async with self.session._state_lock:
            resp_active = (
                self.session.state.resp.is_active()
                and not self.session.state.resp.is_predicted()
            )
        if resp_active and self.session.intent is Intent.CONVERSATION:
            from .pipeline import commit_bargein

            new_item = ConversationItem.new_user_audio(item_id)
            async with self.session._state_lock:
                self.session.state.conversation.append(new_item)
            await self.session.emit(OutboundEvent.item_added(item_to_json(new_item)))
            await commit_bargein(self.session, item_id, audio_start_ms)
            return

        new_item = ConversationItem.new_user_audio(item_id)
        async with self.session._state_lock:
            self.session.state.conversation.append(new_item)
        await self.session.emit(OutboundEvent.item_added(item_to_json(new_item)))
        await self.session.emit(OutboundEvent.speech_started(item_id, audio_start_ms))

    async def _on_speech_committed(
        self, item_id: str, audio_end_ms: int, audio: np.ndarray,
    ) -> None:
        from .pipeline import run_eou_dispatch

        await self.session.emit(OutboundEvent.speech_stopped(item_id, audio_end_ms))
        audio_list = audio.tolist() if isinstance(audio, np.ndarray) else list(audio)
        await run_eou_dispatch(
            self.session,
            item_id,
            audio_list,
            int(audio_end_ms),
            suppress_response=False,
        )
