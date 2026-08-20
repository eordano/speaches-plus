from __future__ import annotations

import asyncio
import base64
import logging
import time
from typing import TYPE_CHECKING, Any, AsyncIterator, Awaitable, Callable

import numpy as np

from conversation.llm import (
    ChatMessage,
    LlmConfig,
    LlmStreamError,
    SentenceChunker,
    complete_stream_messages,
)
from eou.types import sigmoid_lerp
from ids import next_phrase_id, next_turn_id

from . import response_defaults
from . import eou_defaults
from .errors import code as errcode
from . import eou_eager, eou_predicted
from .events import (
    assistant_audio_item_json,
    make_cancelled_brackets,
    make_error_event,
    make_response_done,
)
from .state import (
    ConversationItem,
    InvariantViolation,
    SealedBuffer,
    VadPhase,
    _AtomicU64,
)
from .wire import OutboundEvent, ResponseStatusReason

if TYPE_CHECKING:
    from .session import Session

log = logging.getLogger("realtime.pipeline")

_SENTENCE_FORCE_FLUSH_CHARS = 200

async def _race_hard_cap(
    hard_cap_deadline: float, fut: Awaitable[Any]
) -> tuple[bool, Any]:
    """Race ``fut`` against an absolute monotonic deadline.

    Returns ``(hard_cap_fired, result)``. ``result`` is ``None`` if the cap
    fires first. The pending future is cancelled in either case.
    """
    loop = asyncio.get_running_loop()
    remaining = max(0.0, hard_cap_deadline - loop.time())
    task = asyncio.ensure_future(fut)
    try:
        result = await asyncio.wait_for(task, timeout=remaining)
        return False, result
    except asyncio.TimeoutError:
        if not task.done():
            task.cancel()
        return True, None

async def run_eou_dispatch(
    session: "Session",
    item_id: str,
    audio: Any,
    audio_ms: int,
    suppress_response: bool,
) -> None:
    """EOU scoring stage: score -> maybe eager-dispatch -> race delay -> commit.

    Mirrors ``speaches-plus/rust/src/realtime/pipeline.rs::run_eou_dispatch``.
    """
    cfg = session.eou_config
    started = time.monotonic()
    loop = asyncio.get_running_loop()
    hard_cap_deadline = loop.time() + max(0, cfg.silence_hard_cap_ms) / 1000.0
    audio_list = audio if isinstance(audio, list) else list(np.asarray(audio).tolist())

    try:
        context = await session.build_eou_context(cfg.context_turns)
    except Exception as err:
        log.debug("build_eou_context failed: %s", err)
        context = ""
    input_chars = len(context)

    p: float = 1.0
    cancelled_by = "none"
    hard_cap_during_eou = False
    if cfg.kind.calls_classifier() and session.eou_model is not None:
        score_fut = asyncio.wait_for(
            asyncio.to_thread(
                session.eou_model.score_with_audio, context, audio_list, 16000
            ),
            timeout=max(1, cfg.inference_timeout_ms) / 1000.0,
        )
        hard_cap_during_eou, p_val = await _race_hard_cap(hard_cap_deadline, score_fut)
        if hard_cap_during_eou:
            cancelled_by = "hard_cap"
            p = cfg.failure_p_default
        else:
            try:
                p = float(p_val) if p_val is not None else cfg.failure_p_default
            except (TypeError, ValueError):
                p = cfg.failure_p_default

    threshold = cfg.threshold_for_language(getattr(session.query, "language", None))
    delay_ms = sigmoid_lerp(p, threshold, 1.0, cfg.max_delay_ms, cfg.min_delay_ms)
    elapsed_ms = int((time.monotonic() - started) * 1000)

    try:
        session._observer.on_eou_scored(
            session.id,
            kind=cfg.kind.as_str(),
            score=p,
            eager_score=None,
            threshold=threshold,
            language=getattr(session.query, "language", None),
            input_chars=input_chars,
            input_audio_ms=audio_ms,
            delay_ms=delay_ms,
            elapsed_ms=elapsed_ms,
            cancelled_by=cancelled_by,
            hard_cap_fired=hard_cap_during_eou,
        )
    except Exception as err:
        log.warning("observer on_eou_scored failed: %s", err)

    if hard_cap_during_eou:
        try:
            session._observer.on_eou_hard_cap_fired(
                session.id, item_id=item_id, phase="during_eou", score=None
            )
        except Exception as err:
            log.warning("observer on_eou_hard_cap_fired failed: %s", err)

    if (
        not cfg.eager_disabled()
        and p >= cfg.eager_p_threshold
        and not suppress_response
    ):
        try:
            await eou_eager.try_eager_dispatch(session, cfg, p, audio_list)
        except Exception as err:
            log.warning("try_eager_dispatch failed: %s", err)

    if not hard_cap_during_eou and delay_ms > 0:
        sleep_fut = asyncio.sleep(delay_ms / 1000.0)
        hard_cap_during_wait, _ = await _race_hard_cap(hard_cap_deadline, sleep_fut)
        if hard_cap_during_wait:
            try:
                session._observer.on_eou_hard_cap_fired(
                    session.id, item_id=item_id, phase="during_wait", score=p
                )
            except Exception as err:
                log.warning("observer on_eou_hard_cap_fired failed: %s", err)

    await session.clear_commit_timer()
    await commit_after_eou(session, item_id, audio_list, audio_ms, suppress_response)

async def commit_after_eou(
    session: "Session",
    item_id: str,
    audio: list[float],
    audio_ms: int,
    suppress_response: bool,
) -> None:
    predicted_id_for_log: str | None = None
    predicted_runner = None
    predicted_llm_handle = None
    async with session._state_lock:
        already = any(i.id == item_id for i in session.state.conversation)
        if not already:
            session.state.conversation.append(ConversationItem.new_user_audio(item_id))
        if session.state.vad.is_stopped():
            session.state.vad = VadPhase.silent()
        if session.state.resp.is_predicted():
            predicted_id_for_log = session.state.resp.id
            try:
                predicted_runner, predicted_llm_handle = (
                    session.state.resp_retire_predicted_full()
                )
            except InvariantViolation:
                predicted_runner, predicted_llm_handle = None, None

    session.last_eager_dispatch_at = None

    try:
        session.set_turn_id(next_turn_id())
    except Exception as err:
        log.debug("set_turn_id failed: %s", err)

    predicted_transcript: str | None = None
    if predicted_runner is not None:
        try:
            ok, err = await eou_predicted.await_predicted_stt(predicted_runner)
            if err is None and ok:
                predicted_transcript = ok
            elif err is not None:
                log.warning("speculative STT failed: %s", err)
        except Exception as err:
            log.warning("await_predicted_stt failed: %s", err)

    predicted_llm_text: str | None = None
    if predicted_llm_handle is not None:
        runner = predicted_llm_handle.into_runner()
        timeout = max(1, session.eou_config.silence_hard_cap_ms) / 1000.0
        try:
            await asyncio.wait_for(runner.wait_finished(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        except Exception as err:
            log.debug("predicted_llm wait_finished failed: %s", err)
        rid = predicted_id_for_log or ""
        if runner.overflowed():
            try:
                session._observer.on_predicted_overflow(
                    session.id, response_id=rid, dropped_tokens=runner.dropped_count()
                )
            except Exception as err:
                log.warning("observer on_predicted_overflow failed: %s", err)
            try:
                session._observer.on_predicted_rollback(
                    session.id,
                    response_id=rid,
                    reason="predicted_overflow",
                    llm_chars_thrown=runner.chars_seen(),
                )
            except Exception as err:
                log.warning("observer on_predicted_rollback failed: %s", err)
            runner.abort()
            predicted_llm_text = None
        else:
            text = await runner.snapshot_text()
            runner.abort()
            predicted_llm_text = text or None

    if predicted_llm_text and predicted_transcript:
        if eou_predicted.transcripts_materially_differ(
            predicted_transcript,
            predicted_transcript,
            eou_defaults.EAGER_TRANSCRIPT_MISMATCH_RATIO,
        ):
            rid = predicted_id_for_log or ""
            try:
                session._observer.on_predicted_rollback(
                    session.id,
                    response_id=rid,
                    reason="transcript_mismatch",
                    llm_chars_thrown=len(predicted_llm_text),
                )
            except Exception as err:
                log.warning("observer on_predicted_rollback failed: %s", err)
            predicted_llm_text = None

    sink = await session.event_sink()
    if sink is not None:
        await sink.send_value(OutboundEvent.buffer_committed(item_id))
        from .events import item_to_json

        item = ConversationItem.new_user_audio(item_id)
        await sink.send_value(OutboundEvent.item_added(item_to_json(item)))

    from .session import Intent

    if (
        session.intent is Intent.CONVERSATION
        and not suppress_response
        and getattr(session.turn_detection, "create_response", True)
    ):
        asyncio.create_task(
            _drive_response_after_eou(
                session,
                item_id,
                audio,
                audio_ms,
                cached_transcript=predicted_transcript,
                predicted_llm_text=predicted_llm_text,
            ),
            name=f"resp-after-eou-{item_id}",
        )

async def _drive_response_after_eou(
    session: "Session",
    item_id: str,
    audio: list[float],
    audio_ms: int,
    *,
    cached_transcript: str | None,
    predicted_llm_text: str | None,
) -> None:
    transcribe = getattr(session, "_transcribe", None)
    transcript: str | None = cached_transcript
    if transcript is None:
        if transcribe is None:
            log.warning("no transcribe callable on session; skipping response")
            return
        try:
            raw = await transcribe(audio)
            transcript = (raw or "").strip()
        except Exception as err:
            log.warning("STT failed for %s: %s", item_id, err)
            await session.emit(
                OutboundEvent.transcription_failed(
                    item_id, 0, {"code": "stt_failed", "message": str(err)}
                )
            )
            await session.mark_user_item_incomplete(item_id)
            return
    if not transcript:
        await session.mark_user_item_incomplete(item_id)
        return
    await session.complete_user_item_transcript(item_id, transcript)
    sink = await session.event_sink()
    if sink is not None:
        await sink.send_value(
            OutboundEvent.transcription_completed(item_id, 0, transcript)
        )
    await commit_after_eou_with_response(
        session,
        item_id,
        transcript,
        predicted_llm_text=predicted_llm_text,
    )

async def commit_bargein(session: "Session", item_id: str, audio_start_ms: int) -> None:
    async with session._state_lock:
        prior_handle = None
        if session.state.current_response is not None:
            prior_handle = session.state.current_response.handle
    snap = await session.cancel_current_response()
    if prior_handle is not None and not prior_handle.done():
        try:
            await asyncio.wait_for(prior_handle, timeout=2.0)
        except asyncio.TimeoutError:
            log.warning("prior response task didn't terminate in 2s; continuing")
        except (asyncio.CancelledError, Exception):
            pass
    if snap is not None:
        brackets, done = make_cancelled_brackets(
            snap.response_id,
            snap.assistant_item_id,
            snap.transcript,
            snap.played_ms,
            ResponseStatusReason.BARGE_IN,
            transcript_done_emitted=snap.transcript_done_emitted,
            audio_done_emitted=snap.audio_done_emitted,
        )
        for b in brackets:
            await session.emit(b)
        await session.emit(done)
        await session.apply_truncate_to_assistant_item(snap)
        from .events import make_server_truncate_event

        ev = make_server_truncate_event(
            session.id_source.event(), snap.assistant_item_id, snap.played_ms, snap.transcript
        )
        if ev is not None:
            sink = await session.event_sink()
            if sink is not None:
                await sink.send_value(ev)

    async with session._state_lock:
        session.state.vad = VadPhase.speaking(item_id, audio_start_ms)

    await session.emit(OutboundEvent.speech_started(item_id, audio_start_ms))

async def process_utterance(
    session: "Session",
    response_id: str,
    item_id: str,
    audio: list[float],
    played_ms: _AtomicU64,
    assistant_item_id: str,
    transcript_so_far: list[str],
    audio_ms: int,
    suppress_response: bool,
    cached_transcript: str | None,
    predicted_llm_text: str | None,
    transcribe: Callable[[list[float]], Awaitable[str]],
) -> None:
    sealed = SealedBuffer(item_id=item_id, audio=list(audio), audio_start_ms=0, audio_end_ms=audio_ms)
    async with session._state_lock:
        session.state.store_sealed_buffer(sealed)

    if cached_transcript is not None:
        transcript = cached_transcript
    else:
        try:
            transcript = await transcribe(audio)
        except Exception as err:
            log.warning("STT failed: %s (item_id=%s)", err, item_id)
            await session.emit(
                OutboundEvent.transcription_failed(
                    item_id, 0, {"code": "stt_failed", "message": str(err)}
                )
            )
            await session.mark_user_item_incomplete(item_id)
            async with session._state_lock:
                session.state.drop_sealed_buffer(item_id)
            return

    if not transcript:
        async with session._state_lock:
            session.state.drop_sealed_buffer(item_id)
        return

    await session.complete_user_item_transcript(item_id, transcript)
    sink = await session.event_sink()
    if sink is not None:
        await sink.send_value(OutboundEvent.transcription_completed(item_id, 0, transcript))
    async with session._state_lock:
        session.state.drop_sealed_buffer(item_id)

async def drain_pacer(
    session: "Session",
    response_id: str,
    pacer,
    played_ms: _AtomicU64,
    planned_ms: int,
) -> str:
    if pacer is None:
        return "completed"
    if played_ms.load() >= planned_ms:
        try:
            await pacer.flush()
        except Exception as err:
            log.warning("outbound audio flush failed: %s", err)
        return "completed"

    drain_cap_ms = max(
        response_defaults.DRAIN_CAP_FLOOR_MS,
        min(response_defaults.DRAIN_CAP_CEILING_MS, 2 * planned_ms),
    )
    try:
        await asyncio.wait_for(pacer.flush(), timeout=drain_cap_ms / 1000.0)
        return "completed"
    except asyncio.TimeoutError:
        log.warning(
            "drain cap expired; planned=%d played=%d cap=%d",
            planned_ms,
            played_ms.load(),
            drain_cap_ms,
        )
        return "incomplete"
    except Exception as err:
        log.warning("flush failed: %s", err)
        return "completed"

async def handle_client_too_slow(
    session: "Session",
    response_id: str,
    queued_ms: int,
    cap_ms: int,
    audio_end_ms: int,
    transcript_so_far: str,
) -> None:
    msg = f"outbound queue cap exceeded ({queued_ms} ms buffered)"
    await session.emit(make_error_event(errcode.CLIENT_TOO_SLOW, msg, None, None))
    transcript = transcript_so_far if transcript_so_far else None
    await session.emit(
        make_response_done(
            response_id, "", "failed", transcript, ResponseStatusReason.CLIENT_TOO_SLOW, audio_end_ms
        )
    )

async def stream_llm_sentences(
    cfg: LlmConfig,
    instructions: str | None,
    user_text: str,
    cancel: asyncio.Event | None = None,
) -> AsyncIterator[str]:
    """Yield sentence-sized chunks; retained for test compatibility.

    The live `run_response` path uses `complete_stream_messages` directly so
    transcript deltas can be emitted per SSE chunk (W12) instead of per
    sentence. This helper is still imported by older tests.
    """
    chunker = SentenceChunker()
    messages: list[ChatMessage] = []
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))
    messages.append(ChatMessage(role="user", content=user_text))
    async for delta in complete_stream_messages(cfg, messages, cancel=cancel):
        for sentence in chunker.feed(delta):
            yield sentence
    tail = chunker.flush()
    if tail:
        yield tail

def _resolve_voice(session: "Session") -> str:
    from tts.kokoro import DEFAULT_VOICE as _KOKORO_DEFAULT_VOICE

    v = getattr(session, "voice", None)
    if isinstance(v, str) and v:
        return v
    return _KOKORO_DEFAULT_VOICE

def _resolve_speed(session: "Session") -> float:
    from tts.kokoro.text import DEFAULT_SPEED as _KOKORO_DEFAULT_SPEED

    s = getattr(session, "speed", None)
    if isinstance(s, (int, float)):
        return float(s)
    return float(_KOKORO_DEFAULT_SPEED)

def _resolve_language(session: "Session") -> str:
    from tts.kokoro import DEFAULT_LANGUAGE as _KOKORO_DEFAULT_LANGUAGE

    q = getattr(session, "query", None)
    lang = getattr(q, "language", None) if q is not None else None
    if isinstance(lang, str) and lang:
        return lang
    return _KOKORO_DEFAULT_LANGUAGE

def _build_pacer_for_session(session: "Session", kokoro: Any):
    from .audio_out import OutboundPacer, read_queue_cap_ms_from_env
    from .audio_out_ws import WsAudioPacer

    spec = getattr(session, "outbound_audio", None)
    cap_ms = read_queue_cap_ms_from_env()
    if spec is None:
        return None
    if spec.is_webrtc():
        track = spec.track
        if track is None:
            return None
        pacer = OutboundPacer(
            track=track,
            played_ms_ref=session.played_ms_ref,
            queue_cap_ms=cap_ms,
        )
        try:
            pacer.attach_capture(session.capture_outbound_f32)
        except Exception as err:
            log.debug("attach_capture failed: %s", err)
        return pacer
    if spec.is_websocket():
        ws_send = spec.ws_send
        if ws_send is None:
            return None
        return WsAudioPacer.start(
            ws_send=ws_send,
            id_event_factory=session.id_source.event,
            played_ms_ref=session.played_ms_ref,
            format=spec.format or "",
        )
    return None

def _is_webrtc_session(session: "Session") -> bool:
    spec = getattr(session, "outbound_audio", None)
    return spec is not None and getattr(spec, "is_webrtc", lambda: False)()

_LLM_STREAM_END = object()

class _SentenceQueue:
    """Sentence chunker that lives on the TTS-feed side of the fanout.

    Independent from the transcript-emit task so that TTS lag never gates the
    text stream (the W12 anti-pattern).
    """

    def __init__(self) -> None:
        self._chunker = SentenceChunker()
        self._buf_chars = 0

    def feed(self, delta: str) -> list[str]:
        self._buf_chars += len(delta)
        sentences = self._chunker.feed(delta)
        if sentences:
            self._buf_chars = len(self._chunker.buf)
            return sentences
        if len(self._chunker.buf) >= _SENTENCE_FORCE_FLUSH_CHARS:
            forced = self._chunker.buf.strip()
            self._chunker.buf = ""
            self._buf_chars = 0
            return [forced] if forced else []
        return []

    def flush(self) -> str | None:
        return self._chunker.flush()

async def _build_messages_from_session(
    session: "Session", instructions: str | None, fallback_user_text: str
) -> list[ChatMessage]:
    """Build chat messages from the session's conversation history.

    Uses the session's full transcript so multi-turn context is preserved;
    falls back to `fallback_user_text` when the session has no recorded turns.
    """
    messages: list[ChatMessage] = []
    sys = instructions if instructions is not None else await session.instructions()
    if sys:
        messages.append(ChatMessage(role="system", content=sys))
    try:
        hist = await session.build_chat_messages(None)
    except Exception as err:
        log.debug("build_chat_messages failed: %s", err)
        hist = []
    for m in hist:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant", "system") and isinstance(content, str) and content:
            messages.append(ChatMessage(role=role, content=content))
    if not any(m.role == "user" for m in messages) and fallback_user_text:
        messages.append(ChatMessage(role="user", content=fallback_user_text))
    return messages

async def _emit_audio_delta_event(
    session: "Session", response_id: str, item_id: str, pcm_bytes: bytes
) -> None:
    """Emit a `response.output_audio.delta` event on the data channel.

    Mandatory per §8.2 for WebRTC, because the data channel is a separate plane
    from the audio track and clients without the audio track still need the
    base64 PCM to render. For WebSocket transport the WsAudioPacer already
    pushes audio.delta events directly to the ws_send queue at frame
    granularity, so this would be a duplicate; we skip in that case.
    """
    if not pcm_bytes:
        return
    spec = getattr(session, "outbound_audio", None)
    if spec is not None and getattr(spec, "is_websocket", lambda: False)():
        return
    delta_b64 = base64.b64encode(pcm_bytes).decode("ascii")
    await session.emit(
        OutboundEvent.response_output_audio_delta(response_id, item_id, 0, 0, delta_b64)
    )

async def _iter_llm_deltas(
    cfg: LlmConfig,
    messages: list[ChatMessage],
    cancel: asyncio.Event | None,
    predicted_text: str | None = None,
) -> AsyncIterator[str]:
    """Default LLM delta source -- one delta per upstream SSE chunk.

    Module-level so tests can monkey-patch this seam to inject deterministic
    deltas without exercising the real chat-completions HTTP path. The realtime
    fanout (`_llm_pump`) calls this; both the transcript-emit task and the
    TTS-feed task see the same delta granularity.

    Yielding per SSE chunk (vs per sentence) is what makes W12 pass: the wire's
    inter-delta gap on `response.output_audio_transcript.delta` stays bounded
    by the LLM's token cadence, not by the TTS synth cadence.

    If ``predicted_text`` is set, the cached speculative LLM output replaces
    the upstream call: yield it once and return. Mirrors
    ``pipeline.rs:584-594`` in speaches-plus.
    """
    if predicted_text is not None:
        if predicted_text:
            yield predicted_text
        return
    async for delta in complete_stream_messages(cfg, messages, cancel=cancel):
        yield delta

def _put_sentinels_nowait(*qs: asyncio.Queue) -> None:
    for q in qs:
        try:
            q.put_nowait(_LLM_STREAM_END)
        except asyncio.QueueFull:
            pass

async def _llm_pump(
    cfg: LlmConfig,
    messages: list[ChatMessage],
    text_q: asyncio.Queue,
    tts_q: asyncio.Queue,
    cancel: asyncio.Event | None,
    predicted_text: str | None = None,
) -> None:
    """Pump LLM SSE deltas into two consumer queues.

    The W12 invariant requires that the transcript-delta stream NOT be gated on
    TTS. We satisfy that by fanout: both queues are written to immediately as
    each SSE chunk arrives, and the transcript-emit task drains `text_q`
    independently of how slow the TTS pipeline is.
    """
    try:
        async for delta in _iter_llm_deltas(cfg, messages, cancel, predicted_text=predicted_text):
            if cancel is not None and cancel.is_set():
                break
            await text_q.put(delta)
            await tts_q.put(delta)
    except LlmStreamError:
        _put_sentinels_nowait(text_q, tts_q)
        raise
    except asyncio.CancelledError:
        _put_sentinels_nowait(text_q, tts_q)
        raise
    except Exception as err:
        log.warning("llm pump failed: %s", err)
        _put_sentinels_nowait(text_q, tts_q)
        raise
    await text_q.put(_LLM_STREAM_END)
    await tts_q.put(_LLM_STREAM_END)

async def _transcript_emit_task(
    session: "Session",
    response_id: str,
    item_id: str,
    text_q: asyncio.Queue,
    transcript_parts: list[str],
    transcript_lock: asyncio.Lock,
) -> None:
    """Consume each LLM delta and emit `response.output_audio_transcript.delta`
    immediately. This is the W12 hot path -- never block on TTS here.
    """
    while True:
        item = await text_q.get()
        if item is _LLM_STREAM_END:
            return
        delta: str = item  # type: ignore[assignment]
        if not delta:
            continue
        async with transcript_lock:
            transcript_parts.append(delta)
        await session.emit(
            OutboundEvent.response_output_audio_transcript_delta(
                response_id, item_id, 0, 0, delta
            )
        )

_AUDIO_STREAM_END = object()

async def _tts_synth_task(
    sentence_q: asyncio.Queue,
    audio_q: asyncio.Queue,
    kokoro: Any,
    voice: str,
    speed: float,
    language: str,
    cancel: asyncio.Event | None,
) -> None:
    """Background synth pump: drain sentences, push f32 audio chunks.

    Runs kokoro.stream in a worker thread per sentence so ONNX doesn't block
    the event loop. The pacer-side consumer (`_tts_consume_task`) drains
    `audio_q` independently, so synth for sentence N+1 can proceed while
    sentence N is still being paced -- closing the inter-delta gap that would
    otherwise blow W12 between sentences.
    """
    loop = asyncio.get_event_loop()
    try:
        while True:
            sentence = await sentence_q.get()
            if sentence is _AUDIO_STREAM_END:
                await audio_q.put(_AUDIO_STREAM_END)
                return
            if cancel is not None and cancel.is_set():
                await audio_q.put(_AUDIO_STREAM_END)
                return
            if not sentence or kokoro is None:
                continue

            def _producer() -> None:
                try:
                    for audio_f32, _sr in kokoro.stream(
                        sentence, voice, speed=speed, lang=language
                    ):
                        if cancel is not None and cancel.is_set():
                            break
                        try:
                            asyncio.run_coroutine_threadsafe(
                                audio_q.put(audio_f32), loop
                            ).result()
                        except Exception:
                            break
                except Exception as err:
                    log.warning("kokoro.stream raised: %s", err)

            await asyncio.to_thread(_producer)
    except asyncio.CancelledError:
        try:
            audio_q.put_nowait(_AUDIO_STREAM_END)
        except asyncio.QueueFull:
            pass
        raise

async def _tts_consume_task(
    session: "Session",
    response_id: str,
    item_id: str,
    audio_q: asyncio.Queue,
    pacer,
    cancel: asyncio.Event | None,
    played_ms_atomic: _AtomicU64 | None,
) -> None:
    """Drain pre-synthesized audio chunks and push to pacer / wire."""
    try:
        while True:
            item = await audio_q.get()
            if item is _AUDIO_STREAM_END:
                return
            if cancel is not None and cancel.is_set():
                return
            audio_f32 = item
            if pacer is not None:
                try:
                    await pacer.play(audio_f32)
                except Exception as err:
                    log.warning("pacer.play failed: %s", err)
                if played_ms_atomic is not None:
                    played_ms_atomic.store(
                        int(session.played_ms_ref[0]) if session.played_ms_ref else 0
                    )
            else:
                arr = np.asarray(audio_f32, dtype=np.float32)
                frame_n = 24_000 * 20 // 1000
                cursor = 0
                while cursor + frame_n <= arr.shape[0]:
                    pcm_bytes = _f32_to_pcm16(arr[cursor : cursor + frame_n])
                    await _emit_audio_delta_event(
                        session, response_id, item_id, pcm_bytes
                    )
                    cursor += frame_n
    except asyncio.CancelledError:
        raise

async def _tts_feed_task(
    session: "Session",
    response_id: str,
    item_id: str,
    tts_q: asyncio.Queue,
    pacer,
    kokoro: Any,
    voice: str,
    speed: float,
    language: str,
    cancel: asyncio.Event | None,
    played_ms_atomic: _AtomicU64 | None = None,
) -> None:
    """Drain the TTS-side queue, accumulate sentences, synth + push audio.

    Architecture: this task feeds sentences into a sentence_q. A background
    synth task pulls from sentence_q, runs kokoro.stream in a thread, and
    pushes audio chunks into audio_q. A background consume task pulls from
    audio_q and feeds the pacer. The pipelining means TTS synth for sentence
    N+1 runs while pacer plays sentence N, eliminating the inter-sentence
    audio.delta gap that would otherwise violate W12.

    Per chunk we emit `response.output_audio.delta` (base64 PCM on the data
    channel / WS sink) AND push PCM into the pacer (which feeds the WebRTC
    audio track or the WS audio pacer).
    """
    sq = _SentenceQueue()

    if pacer is not None and _is_webrtc_session(session):
        async def _on_frame(pcm_bytes: bytes) -> None:
            await _emit_audio_delta_event(session, response_id, item_id, pcm_bytes)

        try:
            pacer.attach_frame_callback(_on_frame)
        except Exception as err:
            log.debug("attach_frame_callback failed: %s", err)

    sentence_q: asyncio.Queue = asyncio.Queue(maxsize=64)
    audio_q: asyncio.Queue = asyncio.Queue(maxsize=16)

    synth_task = asyncio.create_task(
        _tts_synth_task(sentence_q, audio_q, kokoro, voice, speed, language, cancel),
        name=f"tts-synth-{response_id}",
    )
    consume_task = asyncio.create_task(
        _tts_consume_task(
            session, response_id, item_id, audio_q, pacer, cancel, played_ms_atomic,
        ),
        name=f"tts-consume-{response_id}",
    )

    cancelled = False
    try:
        while True:
            item = await tts_q.get()
            if item is _LLM_STREAM_END:
                tail = sq.flush()
                if tail:
                    try:
                        session.set_phrase_id(next_phrase_id())
                    except Exception as err:
                        log.debug("set_phrase_id failed: %s", err)
                    await sentence_q.put(tail)
                await sentence_q.put(_AUDIO_STREAM_END)
                break
            delta: str = item  # type: ignore[assignment]
            should_break = False
            for sentence in sq.feed(delta):
                if cancel is not None and cancel.is_set():
                    await sentence_q.put(_AUDIO_STREAM_END)
                    should_break = True
                    break
                try:
                    session.set_phrase_id(next_phrase_id())
                except Exception as err:
                    log.debug("set_phrase_id failed: %s", err)
                await sentence_q.put(sentence)
            if should_break:
                break

        await synth_task
        await consume_task
    except asyncio.CancelledError:
        cancelled = True
    except Exception as err:
        log.warning("tts_feed_task body raised: %s", err)
    finally:
        if cancelled:
            for t in (synth_task, consume_task):
                if not t.done():
                    t.cancel()
            for t in (synth_task, consume_task):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass

    if cancelled:
        raise asyncio.CancelledError()

def _f32_to_pcm16(audio_f32: Any) -> bytes:
    arr = np.asarray(audio_f32, dtype=np.float32)
    if arr.size == 0:
        return b""
    arr = np.clip(arr, -1.0, 1.0)
    v = np.rint(arr * 32_767.0).astype("<i2")
    return v.tobytes()

async def commit_after_eou_with_response(
    session: "Session",
    user_item_id: str,
    user_transcript: str,
    *,
    instructions_override: str | None = None,
    predicted_llm_text: str | None = None,
) -> None:
    """Drive the LLM -> TTS pipeline after an utterance's STT completes.

    Called by VadRunner after `transcription_completed` for conversation-intent
    sessions whose `turn_detection.create_response` is true. Mints the response
    id and assistant item id, registers the response with the session for
    barge-in tracking, and spawns `run_response` as a tracked task.

    Ordering note: we register the response (state.resp -> Created) BEFORE the
    task starts emitting wire events, so `commit_bargein` and the invariant
    checks see a coherent state. The task does its own state transitions
    (Created -> Streaming) once it gets the LLM stream going.
    """
    async with session._state_lock:
        prior_active = (
            session.state.resp.is_active() and not session.state.resp.is_predicted()
        )
        prior_handle = None
        if session.state.current_response is not None:
            prior_handle = session.state.current_response.handle
    if prior_active:
        snap = await session.cancel_current_response()
        if prior_handle is not None and not prior_handle.done():
            try:
                await asyncio.wait_for(prior_handle, timeout=2.0)
            except asyncio.TimeoutError:
                log.warning("prior response task didn't terminate in 2s; continuing")
            except (asyncio.CancelledError, Exception):
                pass
        if snap is not None:
            brackets, done = make_cancelled_brackets(
                snap.response_id,
                snap.assistant_item_id,
                snap.transcript,
                snap.played_ms,
                ResponseStatusReason.BARGE_IN,
                transcript_done_emitted=snap.transcript_done_emitted,
                audio_done_emitted=snap.audio_done_emitted,
            )
            for b in brackets:
                await session.emit(b)
            await session.emit(done)
            await session.apply_truncate_to_assistant_item(snap)

    response_id = session.id_source.response()
    assistant_item_id = session.id_source.item()

    cancel_evt = asyncio.Event()
    transcript_parts: list[str] = []
    played_ms_atomic = _AtomicU64()
    started_gate = asyncio.Event()

    async def _runner() -> None:
        await started_gate.wait()
        try:
            await run_response(
                session,
                response_id=response_id,
                instructions=instructions_override,
                user_text=user_transcript,
                cancel=cancel_evt,
                assistant_item_id=assistant_item_id,
                transcript_parts=transcript_parts,
                played_ms_atomic=played_ms_atomic,
                predicted_llm_text=predicted_llm_text,
            )
        except asyncio.CancelledError:
            cancel_evt.set()
            raise
        except Exception as err:
            log.warning("run_response runner failed: %s", err)
        finally:
            await session.clear_response_if_matches(response_id)

    task = asyncio.create_task(_runner(), name=f"resp-{response_id}")
    try:
        await session.register_response(
            response_id, task, played_ms_atomic, assistant_item_id, transcript_parts
        )
    except Exception as err:
        log.warning("register_response failed: %s", err)

    try:
        async with session._state_lock:
            rt = session.state.current_response
            if rt is not None:
                setattr(rt, "_cancel_event", cancel_evt)
    except Exception as err:
        log.debug("attach cancel event failed: %s", err)

    started_gate.set()

async def run_response(
    session: "Session",
    response_id: str,
    instructions: str | None,
    user_text: str,
    cancel: asyncio.Event | None = None,
    *,
    assistant_item_id: str | None = None,
    transcript_parts: list[str] | None = None,
    played_ms_atomic: _AtomicU64 | None = None,
    predicted_llm_text: str | None = None,
) -> None:
    from . import transport as _transport_mod

    ctx = _transport_mod.get_context()
    kokoro = None
    if ctx is not None and getattr(ctx, "models", None) is not None:
        kokoro = getattr(ctx.models, "kokoro", None)

    pacer = None
    if kokoro is not None:
        try:
            pacer = _build_pacer_for_session(session, kokoro)
        except Exception as err:
            log.warning("pacer build failed: %s", err)
            pacer = None

    voice = _resolve_voice(session)
    speed = _resolve_speed(session)
    language = _resolve_language(session)

    if assistant_item_id is None:
        assistant_item_id = session.id_source.item()
    if transcript_parts is None:
        transcript_parts = []

    cfg = LlmConfig.from_env()
    if cfg is None:
        await session.emit(
            make_error_event(
                errcode.INVALID_REQUEST_ERROR,
                "LLM not configured (CHAT_COMPLETION_BASE_URL unset)",
                None,
                None,
            )
        )
        await session.emit(
            OutboundEvent.response_created(
                {"id": response_id, "object": "realtime.response", "status": "in_progress"}
            )
        )
        await session.emit(make_response_done(
            response_id, assistant_item_id, "failed", None, ResponseStatusReason.LLM_ERROR, 0,
        ))
        return

    await session.emit(
        OutboundEvent.response_created(
            {"id": response_id, "object": "realtime.response", "status": "in_progress"}
        )
    )
    assistant_item_open_json = {
        "id": assistant_item_id,
        "object": "realtime.item",
        "type": "message",
        "role": "assistant",
        "status": "in_progress",
        "content": [],
    }
    await session.emit(
        OutboundEvent.response_output_item_added(response_id, 0, assistant_item_open_json)
    )
    await session.emit(
        OutboundEvent.response_content_part_added(
            response_id,
            assistant_item_id,
            0,
            0,
            {"type": "audio", "transcript": ""},
        )
    )

    try:
        await session.mark_streaming(response_id)
    except Exception as err:
        log.debug("mark_streaming failed: %s", err)

    if played_ms_atomic is not None:
        try:
            async with session._state_lock:
                if (
                    session.state.resp.id == response_id
                    and session.state.resp.played_ms is not None
                ):
                    session.state.resp.played_ms = played_ms_atomic
        except Exception as err:
            log.debug("wire resp.played_ms failed: %s", err)

    try:
        session.set_turn_id(next_turn_id())
    except Exception as err:
        log.debug("set_turn_id failed: %s", err)

    text_q: asyncio.Queue = asyncio.Queue(maxsize=128)
    tts_q: asyncio.Queue = asyncio.Queue(maxsize=128)

    messages = await _build_messages_from_session(session, instructions, user_text)
    transcript_lock = asyncio.Lock()

    llm_task = asyncio.create_task(
        _llm_pump(cfg, messages, text_q, tts_q, cancel, predicted_text=predicted_llm_text),
        name=f"llm-pump-{response_id}",
    )
    transcript_task = asyncio.create_task(
        _transcript_emit_task(
            session, response_id, assistant_item_id, text_q, transcript_parts, transcript_lock
        ),
        name=f"transcript-emit-{response_id}",
    )
    tts_task = asyncio.create_task(
        _tts_feed_task(
            session, response_id, assistant_item_id, tts_q,
            pacer, kokoro, voice, speed, language, cancel,
            played_ms_atomic,
        ),
        name=f"tts-feed-{response_id}",
    )

    cancelled = False
    failed_reason: ResponseStatusReason | None = None
    fail_message: str | None = None
    transcript_done_emitted = False

    try:
        await llm_task
        await transcript_task
        transcript = "".join(transcript_parts)
        await session.emit(
            OutboundEvent.response_output_audio_transcript_done(
                response_id, assistant_item_id, 0, 0, transcript
            )
        )
        transcript_done_emitted = True
        try:
            async with session._state_lock:
                rt = session.state.current_response
                if rt is not None:
                    setattr(rt, "_transcript_done_emitted", True)
        except Exception as err:
            log.debug("flag transcript_done on runtime failed: %s", err)
        await tts_task
    except asyncio.CancelledError:
        cancelled = True
    except LlmStreamError as err:
        failed_reason = ResponseStatusReason.LLM_ERROR
        fail_message = str(err)
        log.warning("LLM stream error: %s", err)
    except Exception as err:
        failed_reason = ResponseStatusReason.LLM_ERROR
        fail_message = str(err)
        log.warning("LLM stream failed: %s", err)
    finally:
        if cancel is not None and cancel.is_set():
            cancelled = True
        for t in (llm_task, transcript_task, tts_task):
            if not t.done():
                t.cancel()
        for t in (llm_task, transcript_task, tts_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        if cancelled and pacer is not None:
            try:
                pacer.cancel()
            except Exception as err:
                log.debug("pacer cancel failed: %s", err)

    try:
        session.set_phrase_id(None)
        session.set_turn_id(None)
    except Exception:
        pass

    if cancelled:
        raise asyncio.CancelledError()

    if failed_reason is not None:
        await session.emit(
            make_error_event(errcode.INVALID_REQUEST_ERROR, fail_message or "LLM error", None, None)
        )
        played_ms_val = int(session.played_ms_ref[0] if session.played_ms_ref else 0)
        if played_ms_atomic is not None:
            played_ms_atomic.store(played_ms_val)
        await session.emit(
            make_response_done(
                response_id, assistant_item_id, "failed",
                "".join(transcript_parts) if transcript_parts else None,
                failed_reason,
                played_ms_val,
            )
        )
        return

    if pacer is not None:
        try:
            await pacer.flush()
        except Exception as err:
            log.warning("pacer flush failed: %s", err)

    transcript = "".join(transcript_parts)

    if not transcript_done_emitted:
        await session.emit(
            OutboundEvent.response_output_audio_transcript_done(
                response_id, assistant_item_id, 0, 0, transcript
            )
        )
    await session.emit(
        OutboundEvent.response_output_audio_done(response_id, assistant_item_id, 0, 0)
    )
    try:
        async with session._state_lock:
            rt = session.state.current_response
            if rt is not None:
                setattr(rt, "_audio_done_emitted", True)
    except Exception as err:
        log.debug("flag audio_done on runtime failed: %s", err)
    await session.emit(
        OutboundEvent.response_content_part_done(
            response_id, assistant_item_id, 0, 0,
            {"type": "audio", "transcript": transcript},
        )
    )

    played_ms_val = int(session.played_ms_ref[0] if session.played_ms_ref else 0)
    if played_ms_atomic is not None:
        played_ms_atomic.store(played_ms_val)

    await session.emit(
        OutboundEvent.response_output_item_done(
            response_id, 0,
            assistant_audio_item_json(assistant_item_id, transcript, "completed"),
        )
    )

    try:
        await session.append_assistant_item(assistant_item_id, transcript, played_ms_val)
    except Exception as err:
        log.debug("append_assistant_item failed: %s", err)

    await session.emit(
        make_response_done(response_id, assistant_item_id, "completed", transcript, None, played_ms_val)
    )
