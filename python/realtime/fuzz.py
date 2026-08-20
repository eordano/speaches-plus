from __future__ import annotations

import enum
from dataclasses import dataclass

from .state import (
    InvariantViolation,
    SealedBuffer,
    SessionPhase,
    SessionState,
    TerminationReason,
    VadPhase,
    apply_truncate_to_conversation,
    check_state,
)

class FuzzOp(enum.Enum):
    SESSION_ACTIVATE = "SessionActivate"
    VAD_SPEECH_START = "VadSpeechStart"
    VAD_SPEECH_STOP = "VadSpeechStop"
    START_PREDICTED = "StartPredicted"
    START_PREDICTED_WITH_LLM = "StartPredictedWithLlm"
    PROMOTE_PREDICTED = "PromotePredicted"
    CREATE_FROM_NONE = "CreateFromNone"
    ADVANCE_TO_STREAMING = "AdvanceToStreaming"
    DRAIN = "Drain"
    RETIRE_TO_NONE = "RetireToNone"
    RETIRE_PREDICTED = "RetirePredicted"
    RETIRE_PREDICTED_FULL = "RetirePredictedFull"
    STORE_SEALED_BUFFER = "StoreSealedBuffer"
    DROP_SEALED_BUFFER = "DropSealedBuffer"
    TRUNCATE_CONVERSATION = "TruncateConversation"
    TERMINATE = "Terminate"

_ALL_OPS = list(FuzzOp)

@dataclass
class Lcg:
    state: int

    @classmethod
    def new(cls, seed: int) -> "Lcg":
        return cls(state=(seed + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF)

    def next(self) -> int:
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state

    def pick(self, seq):
        i = self.next() % len(seq)
        return seq[i]

def _dummy_runtime():
    from .state import ResponseRuntime

    return ResponseRuntime(handle=None)

def _apply(op: FuzzOp, state: SessionState, idx: int) -> None:
    try:
        if op is FuzzOp.SESSION_ACTIVATE:
            if state.session.is_pending():
                state.session = SessionPhase.active(idx)
        elif op is FuzzOp.TERMINATE:
            if not state.session.is_terminated():
                state.session = SessionPhase.terminated(TerminationReason.CLIENT_CLOSED)
        elif op is FuzzOp.VAD_SPEECH_START:
            if state.session.is_active() and state.vad.is_silent() and not state.resp.is_active():
                state.vad = VadPhase.speaking(f"item_{idx}", idx * 100)
        elif op is FuzzOp.VAD_SPEECH_STOP:
            if state.vad.is_speaking():
                state.vad = VadPhase.stopped(
                    state.vad.item_id or "",
                    state.vad.audio_start_ms or 0,
                    (state.vad.audio_start_ms or 0) + 1000,
                )
        elif op is FuzzOp.START_PREDICTED:
            if state.session.is_active():
                try:
                    state.resp_start_predicted(f"resp_{idx}", f"item_{idx}", 0.9, None)
                except InvariantViolation:
                    pass
        elif op is FuzzOp.START_PREDICTED_WITH_LLM:
            if state.session.is_active():
                try:
                    state.resp_start_predicted_with_llm(
                        f"resp_{idx}", f"item_{idx}", 0.9, None, _dummy_llm_handle()
                    )
                except InvariantViolation:
                    pass
        elif op is FuzzOp.PROMOTE_PREDICTED:
            try:
                state.resp_promote_predicted_to_created(_dummy_runtime())
            except InvariantViolation:
                pass
        elif op is FuzzOp.CREATE_FROM_NONE:
            if state.session.is_active() and not state.vad.is_speaking():
                try:
                    state.resp_create_from_none(f"resp_{idx}", f"item_{idx}", _dummy_runtime())
                except InvariantViolation:
                    pass
        elif op is FuzzOp.ADVANCE_TO_STREAMING:
            try:
                state.resp_advance_to_streaming(_AtomicU64_new())
            except InvariantViolation:
                pass
        elif op is FuzzOp.DRAIN:
            try:
                state.resp_drain(1500)
            except InvariantViolation:
                pass
        elif op is FuzzOp.RETIRE_TO_NONE:
            try:
                state.resp_retire_to_none()
            except InvariantViolation:
                pass
        elif op is FuzzOp.RETIRE_PREDICTED:
            try:
                state.resp_retire_predicted()
            except InvariantViolation:
                pass
        elif op is FuzzOp.RETIRE_PREDICTED_FULL:
            try:
                state.resp_retire_predicted_full()
            except InvariantViolation:
                pass
        elif op is FuzzOp.STORE_SEALED_BUFFER:
            slot = idx % 8
            start = idx * 50
            state.store_sealed_buffer(
                SealedBuffer(item_id=f"buf_item_{slot}", audio=[], audio_start_ms=start, audio_end_ms=start + 100)
            )
        elif op is FuzzOp.DROP_SEALED_BUFFER:
            slot = idx % 8
            state.drop_sealed_buffer(f"buf_item_{slot}")
        elif op is FuzzOp.TRUNCATE_CONVERSATION:
            slot = idx % 8
            apply_truncate_to_conversation(
                state.conversation, f"buf_item_{slot}", idx % 2_000, "fuzz transcript"
            )
    except InvariantViolation:
        pass

def _AtomicU64_new():
    from .state import _AtomicU64

    return _AtomicU64()

def _dummy_llm_handle():
    from .eou_predicted import PredictedLlmShared
    from .state import PredictedLlmRunnerHandle

    return PredictedLlmRunnerHandle(task=None, shared=PredictedLlmShared(), cap=16)

def run_random_walk(seed: int, steps: int) -> tuple[int, FuzzOp, str] | None:
    state = SessionState()
    rng = Lcg.new(seed)
    for i in range(steps):
        op = rng.pick(_ALL_OPS)
        _apply(op, state, i)
        try:
            check_state(state)
        except InvariantViolation as v:
            return (i, op, str(v))
    return None
