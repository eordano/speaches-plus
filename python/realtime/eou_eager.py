from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .eou_predicted import spawn_predicted_llm, spawn_predicted_stt
from .state import InvariantViolation, PredictedLlmRunnerHandle

if TYPE_CHECKING:
    from eou.loader import EouConfig

    from .session import Session

log = logging.getLogger("realtime.eou_eager")

async def try_eager_dispatch(
    session: "Session",
    cfg: "EouConfig",
    score: float,
    audio: Any,
) -> None:
    now = time.monotonic()
    if session.last_eager_dispatch_at is not None:
        throttle_s = max(0.001, cfg.eager_interval_ms / 1000.0)
        if now - session.last_eager_dispatch_at < throttle_s:
            try:
                session._observer.on_predicted_suppressed(
                    session.id, score=score, inflight=1
                )
            except Exception as err:
                log.warning("observer on_predicted_suppressed failed: %s", err)
            return
    session.last_eager_dispatch_at = now

    predicted_id = session.id_source.response()
    predicted_item_id = session.id_source.item()

    transcribe = getattr(session, "_transcribe", None)
    if transcribe is None:
        log.warning("eager dispatch skipped: session has no transcribe callable")
        session.last_eager_dispatch_at = None
        return

    stt_runner = spawn_predicted_stt(transcribe, _to_audio_list(audio))

    llm_handle: PredictedLlmRunnerHandle | None = None
    from .session import Intent

    if session.intent is Intent.CONVERSATION:
        llm_handle = await _spawn_predicted_llm_for_session(
            session, cfg, stt_runner, trajectory_tag=predicted_id
        )

    started_ok = False
    epoch_out: int | None = None
    async with session._state_lock:
        if not session.state.resp.is_active():
            try:
                epoch_out = session.state.resp_start_predicted_with_llm(
                    predicted_id,
                    predicted_item_id,
                    score,
                    stt_runner,
                    llm_handle,
                )
                started_ok = True
            except InvariantViolation:
                started_ok = False
        if started_ok and epoch_out is not None:
            try:
                session._observer.on_predicted_promoted(
                    session.id, response_id=predicted_id, score=score
                )
            except Exception as err:
                log.warning("observer on_predicted_promoted failed: %s", err)
            try:
                session._observer.on_eou_eager_dispatch(
                    session.id,
                    response_id=predicted_id,
                    item_id=predicted_item_id,
                    score=score,
                    threshold=cfg.eager_p_threshold,
                    epoch=int(epoch_out),
                )
            except Exception as err:
                log.warning("observer on_eou_eager_dispatch failed: %s", err)

    if not started_ok:
        session.last_eager_dispatch_at = None
        try:
            session._observer.on_predicted_suppressed(
                session.id, score=score, inflight=1
            )
        except Exception as err:
            log.warning("observer on_predicted_suppressed failed: %s", err)
        if stt_runner.task is not None:
            stt_runner.task.cancel()
        if llm_handle is not None:
            llm_handle.into_runner().abort()

def _to_audio_list(audio: Any) -> list[float]:
    if isinstance(audio, list):
        return audio
    try:
        return [float(x) for x in audio]
    except TypeError:
        return list(audio)

async def _spawn_predicted_llm_for_session(
    session: "Session",
    cfg: "EouConfig",
    stt_runner: Any,
    trajectory_tag: str | None = None,
) -> PredictedLlmRunnerHandle | None:
    try:
        from conversation.llm import (
            ChatMessage,
            LlmConfig,
            complete_stream_messages,
        )
    except Exception as err:
        log.warning("predicted-llm import failed: %s", err)
        return None

    llm_cfg = LlmConfig.from_env()
    if llm_cfg is None:
        return None

    try:
        instructions = await session.instructions()
        msgs = await session.build_chat_messages(instructions)
    except Exception as err:
        log.warning("predicted-llm message build failed: %s", err)
        return None

    base_msgs = [
        ChatMessage(role=m.get("role", "user"), content=m.get("content", ""))
        for m in list(msgs)
    ]

    shared_ref: list[Any] = []

    async def _stream():
        from .eou_predicted import await_predicted_stt

        ok, err = await await_predicted_stt(stt_runner)
        partial = (ok or "").strip()
        if err is not None or not partial:
            return
        if shared_ref:
            shared_ref[0].prompt_user_text = partial
        chat_msgs = base_msgs + [ChatMessage(role="user", content=partial)]
        async for delta in complete_stream_messages(
            llm_cfg, chat_msgs, trajectory_tag=trajectory_tag
        ):
            yield delta

    runner = spawn_predicted_llm(_stream, cfg.predicted_token_buffer_cap)
    shared_ref.append(runner.shared)
    return PredictedLlmRunnerHandle.from_runner(runner)
