from __future__ import annotations

import asyncio
import enum
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    from aiortc import RTCPeerConnection

    from .eou_predicted import PredictedLlmRunner, PredictedLlmShared
    from .transport import EventSink

SEALED_BUFFER_RETENTION_COUNT_DEFAULT = 4

class TerminationReason(enum.Enum):
    CLIENT_CLOSED = "client_closed"
    MAX_DURATION = "max_duration"
    INTERNAL_STATE_ERROR = "internal_state_error"
    VAD_FAILED = "vad_failed"
    STT_FAILED = "stt_failed"
    MODEL_LOAD_FAILED = "model_load_failed"

    def as_str(self) -> str:
        return self.value

class _SessionPhaseTag(enum.Enum):
    PENDING = "pending"
    ACTIVE = "active"
    TERMINATED = "terminated"

@dataclass
class SessionPhase:
    tag: _SessionPhaseTag = _SessionPhaseTag.PENDING
    created_at_ms: int | None = None
    reason: TerminationReason | None = None

    @classmethod
    def pending(cls) -> SessionPhase:
        return cls(tag=_SessionPhaseTag.PENDING)

    @classmethod
    def active(cls, created_at_ms: int) -> SessionPhase:
        return cls(tag=_SessionPhaseTag.ACTIVE, created_at_ms=created_at_ms)

    @classmethod
    def terminated(cls, reason: TerminationReason) -> SessionPhase:
        return cls(tag=_SessionPhaseTag.TERMINATED, reason=reason)

    def is_active(self) -> bool:
        return self.tag is _SessionPhaseTag.ACTIVE

    def is_terminated(self) -> bool:
        return self.tag is _SessionPhaseTag.TERMINATED

    def is_pending(self) -> bool:
        return self.tag is _SessionPhaseTag.PENDING

class _VadPhaseTag(enum.Enum):
    SILENT = "silent"
    SPEAKING = "speaking"
    STOPPED = "stopped"

@dataclass
class VadPhase:
    tag: _VadPhaseTag = _VadPhaseTag.SILENT
    item_id: str | None = None
    audio_start_ms: int | None = None
    audio_end_ms: int | None = None

    @classmethod
    def silent(cls) -> VadPhase:
        return cls(tag=_VadPhaseTag.SILENT)

    @classmethod
    def speaking(cls, item_id: str, audio_start_ms: int) -> VadPhase:
        return cls(
            tag=_VadPhaseTag.SPEAKING,
            item_id=item_id,
            audio_start_ms=audio_start_ms,
        )

    @classmethod
    def stopped(cls, item_id: str, audio_start_ms: int, audio_end_ms: int) -> VadPhase:
        return cls(
            tag=_VadPhaseTag.STOPPED,
            item_id=item_id,
            audio_start_ms=audio_start_ms,
            audio_end_ms=audio_end_ms,
        )

    def is_silent(self) -> bool:
        return self.tag is _VadPhaseTag.SILENT

    def is_speaking(self) -> bool:
        return self.tag is _VadPhaseTag.SPEAKING

    def is_stopped(self) -> bool:
        return self.tag is _VadPhaseTag.STOPPED

@dataclass
class PredictedSharedState:
    user_transcript: tuple[bool, str | None, str | None] = (False, None, None)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@dataclass
class PredictedRunner:
    task: asyncio.Task | None
    shared: PredictedSharedState

@dataclass
class PredictedLlmRunnerHandle:
    task: asyncio.Task | None
    shared: "PredictedLlmShared"
    cap: int

    @classmethod
    def from_runner(cls, runner: "PredictedLlmRunner") -> "PredictedLlmRunnerHandle":
        return cls(task=runner.task, shared=runner.shared, cap=runner.cap)

    def into_runner(self) -> "PredictedLlmRunner":
        from .eou_predicted import PredictedLlmRunner

        return PredictedLlmRunner(task=self.task, shared=self.shared, cap=self.cap)

class _RespPhaseTag(enum.Enum):
    NONE = "None"
    PREDICTED = "Predicted"
    CREATED = "Created"
    STREAMING = "Streaming"
    DRAIN = "Drain"

@dataclass
class _AtomicU64:
    _value: int = 0

    def load(self) -> int:
        return self._value

    def store(self, v: int) -> None:
        self._value = v

@dataclass
class RespPhase:
    tag: _RespPhaseTag = _RespPhaseTag.NONE
    id: str | None = None
    item_id: str | None = None
    epoch: int | None = None
    eou_score: float | None = None
    runner: PredictedRunner | None = None
    llm_runner: PredictedLlmRunnerHandle | None = None
    played_ms: _AtomicU64 | None = None
    planned_ms: int | None = None

    @classmethod
    def none(cls) -> RespPhase:
        return cls(tag=_RespPhaseTag.NONE)

    @classmethod
    def predicted(
        cls,
        id: str,
        item_id: str,
        epoch: int,
        eou_score: float,
        runner: PredictedRunner | None,
        llm_runner: PredictedLlmRunnerHandle | None,
    ) -> RespPhase:
        return cls(
            tag=_RespPhaseTag.PREDICTED,
            id=id,
            item_id=item_id,
            epoch=epoch,
            eou_score=eou_score,
            runner=runner,
            llm_runner=llm_runner,
        )

    @classmethod
    def created(cls, id: str, item_id: str, epoch: int) -> RespPhase:
        return cls(tag=_RespPhaseTag.CREATED, id=id, item_id=item_id, epoch=epoch)

    @classmethod
    def streaming(
        cls,
        id: str,
        item_id: str,
        epoch: int,
        played_ms: _AtomicU64,
        planned_ms: int | None,
    ) -> RespPhase:
        return cls(
            tag=_RespPhaseTag.STREAMING,
            id=id,
            item_id=item_id,
            epoch=epoch,
            played_ms=played_ms,
            planned_ms=planned_ms,
        )

    @classmethod
    def drain(
        cls,
        id: str,
        item_id: str,
        epoch: int,
        played_ms: _AtomicU64,
        planned_ms: int,
    ) -> RespPhase:
        return cls(
            tag=_RespPhaseTag.DRAIN,
            id=id,
            item_id=item_id,
            epoch=epoch,
            played_ms=played_ms,
            planned_ms=planned_ms,
        )

    def is_active(self) -> bool:
        return self.tag is not _RespPhaseTag.NONE

    def is_predicted(self) -> bool:
        return self.tag is _RespPhaseTag.PREDICTED

    def kind(self) -> str:
        return self.tag.value

def _phase_kind(p: RespPhase) -> str:
    return p.kind()

@dataclass
class OpenBuffer:
    id: str
    audio: list[float] = field(default_factory=list)
    audio_start_ms: int | None = None

    def append(self, samples: Iterable[float]) -> None:
        self.audio.extend(float(s) for s in samples)

    def seal(self, audio_end_ms: int) -> SealedBuffer:
        start = self.audio_start_ms if self.audio_start_ms is not None else 0
        return SealedBuffer(
            item_id=self.id,
            audio=list(self.audio),
            audio_start_ms=int(start),
            audio_end_ms=int(audio_end_ms),
        )

@dataclass
class SealedBuffer:
    item_id: str
    audio: list[float]
    audio_start_ms: int
    audio_end_ms: int

@dataclass
class ResponseRuntime:
    handle: asyncio.Task | None
    transcript_so_far: list[str] = field(default_factory=list)
    transcript_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@dataclass
class PendingBargein:
    item_id: str
    audio_start_ms: int

class ItemRole(enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

    def as_str(self) -> str:
        return self.value

    @classmethod
    def parse(cls, s: str) -> ItemRole | None:
        for member in cls:
            if member.value == s:
                return member
        return None

class ItemStatus(enum.Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    INCOMPLETE = "incomplete"

    def as_str(self) -> str:
        return self.value

class _ItemContentTag(enum.Enum):
    USER_AUDIO = "user_audio"
    ASSISTANT_AUDIO = "assistant_audio"
    TEXT = "text"

@dataclass
class ItemContent:
    tag: _ItemContentTag
    transcript: str | None = None
    audio_end_ms: int | None = None
    audio_ms: int | None = None
    text: str | None = None

    @classmethod
    def user_audio(cls, transcript: str | None = None, audio_end_ms: int | None = None) -> ItemContent:
        return cls(tag=_ItemContentTag.USER_AUDIO, transcript=transcript, audio_end_ms=audio_end_ms)

    @classmethod
    def assistant_audio(cls, transcript: str, audio_ms: int) -> ItemContent:
        return cls(tag=_ItemContentTag.ASSISTANT_AUDIO, transcript=transcript, audio_ms=audio_ms)

    @classmethod
    def text_(cls, text: str) -> ItemContent:
        return cls(tag=_ItemContentTag.TEXT, text=text)

    def is_user_audio(self) -> bool:
        return self.tag is _ItemContentTag.USER_AUDIO

    def is_assistant_audio(self) -> bool:
        return self.tag is _ItemContentTag.ASSISTANT_AUDIO

    def is_text(self) -> bool:
        return self.tag is _ItemContentTag.TEXT

@dataclass
class ConversationItem:
    id: str
    role: ItemRole
    status: ItemStatus
    content: ItemContent

    @classmethod
    def new_user_audio(cls, id: str) -> ConversationItem:
        return cls(
            id=id,
            role=ItemRole.USER,
            status=ItemStatus.IN_PROGRESS,
            content=ItemContent.user_audio(),
        )

    @classmethod
    def new_assistant_audio(cls, id: str, transcript: str, audio_ms: int) -> ConversationItem:
        return cls(
            id=id,
            role=ItemRole.ASSISTANT,
            status=ItemStatus.COMPLETED,
            content=ItemContent.assistant_audio(transcript, audio_ms),
        )

    @classmethod
    def new_text(cls, id: str, role: ItemRole, status: ItemStatus, text: str) -> ConversationItem:
        return cls(id=id, role=role, status=status, content=ItemContent.text_(text))

    def transcript(self) -> str | None:
        if self.content.is_user_audio():
            return self.content.transcript
        if self.content.is_assistant_audio():
            return self.content.transcript
        if self.content.is_text():
            return self.content.text
        return None

def apply_truncate_to_conversation(
    conversation: list[ConversationItem],
    assistant_item_id: str,
    played_ms: int,
    transcript: str,
) -> None:
    for item in conversation:
        if item.id == assistant_item_id:
            item.status = ItemStatus.INCOMPLETE
            if item.content.is_assistant_audio():
                cur = item.content.audio_ms or 0
                item.content.audio_ms = min(cur, played_ms)
                item.content.transcript = transcript
            else:
                item.content = ItemContent.assistant_audio(transcript, played_ms)
            return
    item = ConversationItem.new_assistant_audio(assistant_item_id, transcript, played_ms)
    item.status = ItemStatus.INCOMPLETE
    conversation.append(item)

class _IVKind(enum.Enum):
    SPEAKING_WITH_ACTIVE_RESPONSE = "SpeakingWithActiveResponse"
    STOPPED_WITHOUT_END = "StoppedWithoutEnd"
    DRAIN_WITHOUT_PLANNED_MS = "DrainWithoutPlannedMs"
    EPOCH_REGRESSION = "EpochRegression"
    SEALED_BUFFER_APPEND = "SealedBufferAppend"
    RESPONSE_RUNTIME_MISMATCH = "ResponseRuntimeMismatch"
    ILLEGAL_RESP_TRANSITION = "IllegalRespTransition"
    EMPTY_RESPONSE_ID = "EmptyResponseId"
    ROTATION_BEFORE_COMMIT = "RotationBeforeCommit"
    CONV_HAS_VOICED = "ConvHasVoiced"

@dataclass
class InvariantViolation(Exception):
    kind: _IVKind
    from_: str | None = None
    to: str | None = None
    message: str = ""

    def __post_init__(self) -> None:
        Exception.__init__(self, self._render())

    def _render(self) -> str:
        if self.kind is _IVKind.ILLEGAL_RESP_TRANSITION:
            return f"IllegalRespTransition({self.from_} -> {self.to})"
        return f"{self.kind.value}: {self.message}" if self.message else self.kind.value

    def __str__(self) -> str:
        return self._render()

def _make_speaking_active_resp() -> InvariantViolation:
    return InvariantViolation(_IVKind.SPEAKING_WITH_ACTIVE_RESPONSE)

def _make_stopped_without_end() -> InvariantViolation:
    return InvariantViolation(_IVKind.STOPPED_WITHOUT_END)

def _make_runtime_mismatch() -> InvariantViolation:
    return InvariantViolation(_IVKind.RESPONSE_RUNTIME_MISMATCH)

def _make_illegal_transition(from_: str, to: str) -> InvariantViolation:
    return InvariantViolation(_IVKind.ILLEGAL_RESP_TRANSITION, from_=from_, to=to)

def _make_empty_response_id() -> InvariantViolation:
    return InvariantViolation(_IVKind.EMPTY_RESPONSE_ID)

def _make_rotation_before_commit(item_id: str) -> InvariantViolation:
    return InvariantViolation(_IVKind.ROTATION_BEFORE_COMMIT, message=item_id)

def _make_conv_has_voiced(item_id: str) -> InvariantViolation:
    return InvariantViolation(_IVKind.CONV_HAS_VOICED, message=item_id)

_WIRE_ACTIVE_RESP_TAGS = (_RespPhaseTag.CREATED, _RespPhaseTag.STREAMING, _RespPhaseTag.DRAIN)

def check_invariants(session: SessionPhase, vad: VadPhase, resp: RespPhase) -> None:
    if session.is_terminated():
        return
    if vad.is_speaking() and resp.tag in _WIRE_ACTIVE_RESP_TAGS:
        raise _make_speaking_active_resp()
    if vad.is_stopped():
        if vad.audio_end_ms is not None and vad.audio_start_ms is not None:
            if vad.audio_end_ms < vad.audio_start_ms:
                raise _make_stopped_without_end()

def check_state(state: SessionState) -> None:
    check_invariants(state.session, state.vad, state.resp)
    if state.resp.tag is _RespPhaseTag.NONE and state.current_response is not None:
        raise _make_runtime_mismatch()
    if state.resp.tag in _WIRE_ACTIVE_RESP_TAGS:
        if state.current_response is None:
            raise _make_runtime_mismatch()
        if not state.resp.id:
            raise _make_empty_response_id()
    if state.vad.is_stopped() and state.vad.item_id is not None:
        if state.sealed_buffer(state.vad.item_id) is not None:
            raise _make_rotation_before_commit(state.vad.item_id)
    if state.vad.is_speaking() and state.vad.item_id is not None:
        for it in state.conversation:
            if (
                it.id == state.vad.item_id
                and it.role is ItemRole.USER
                and it.status is ItemStatus.IN_PROGRESS
            ):
                raise _make_conv_has_voiced(state.vad.item_id)

class Topic(enum.Enum):
    SESSION = "session"
    ITEM = "item"
    BUFFER = "buffer"
    RESPONSE = "response"
    ERROR = "error"
    OTHER = "other"

    @classmethod
    def classify(cls, event_type: str) -> Topic:
        if event_type.startswith("response."):
            return cls.RESPONSE
        if event_type.startswith("session."):
            return cls.SESSION
        if event_type.startswith("conversation.item."):
            return cls.ITEM
        if event_type.startswith("input_audio_buffer."):
            return cls.BUFFER
        if event_type == "error":
            return cls.ERROR
        return cls.OTHER

class ResponseGate:
    @classmethod
    def open(cls, resp: RespPhase) -> ResponseGate | None:
        if resp.is_predicted():
            return None
        return cls()

@dataclass
class SessionState:
    session: SessionPhase = field(default_factory=SessionPhase.pending)
    vad: VadPhase = field(default_factory=VadPhase.silent)
    resp: RespPhase = field(default_factory=RespPhase.none)
    instructions: str | None = None
    pc: RTCPeerConnection | None = None
    event_sink: EventSink | None = None
    current_response: ResponseRuntime | None = None
    last_epoch: int = 0
    timeout_task: asyncio.Task | None = None
    commit_timer: asyncio.Task | None = None
    bargein_task: asyncio.Task | None = None
    pending_bargein: PendingBargein | None = None
    conversation: list[ConversationItem] = field(default_factory=list)
    current_speech_item: str | None = None
    sealed_buffers: deque[SealedBuffer] = field(default_factory=deque)
    sealed_buffer_retention_count: int = SEALED_BUFFER_RETENTION_COUNT_DEFAULT

    def store_sealed_buffer(self, buf: SealedBuffer) -> None:
        self.sealed_buffers = deque([b for b in self.sealed_buffers if b.item_id != buf.item_id])
        self.sealed_buffers.append(buf)
        while len(self.sealed_buffers) > self.sealed_buffer_retention_count:
            self.sealed_buffers.popleft()

    def drop_sealed_buffer(self, item_id: str) -> bool:
        before = len(self.sealed_buffers)
        self.sealed_buffers = deque([b for b in self.sealed_buffers if b.item_id != item_id])
        return before != len(self.sealed_buffers)

    def sealed_buffer(self, item_id: str) -> SealedBuffer | None:
        for b in self.sealed_buffers:
            if b.item_id == item_id:
                return b
        return None

    def resp_create_from_none(
        self,
        id: str,
        item_id: str,
        runtime: ResponseRuntime,
    ) -> int:
        if self.resp.tag is not _RespPhaseTag.NONE:
            raise _make_illegal_transition(_phase_kind(self.resp), "Created")
        if self.vad.is_speaking():
            raise _make_speaking_active_resp()
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.resp = RespPhase.created(id, item_id, epoch)
        self.current_response = runtime
        check_state(self)
        return epoch

    def resp_start_predicted(
        self,
        id: str,
        item_id: str,
        eou_score: float,
        runner: PredictedRunner | None,
    ) -> int:
        return self.resp_start_predicted_with_llm(id, item_id, eou_score, runner, None)

    def resp_start_predicted_with_llm(
        self,
        id: str,
        item_id: str,
        eou_score: float,
        runner: PredictedRunner | None,
        llm_runner: PredictedLlmRunnerHandle | None,
    ) -> int:
        if self.resp.tag is not _RespPhaseTag.NONE:
            raise _make_illegal_transition(_phase_kind(self.resp), "Predicted")
        if self.vad.is_speaking():
            raise _make_speaking_active_resp()
        epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self.resp = RespPhase.predicted(id, item_id, epoch, eou_score, runner, llm_runner)
        check_state(self)
        return epoch

    def resp_attach_predicted_llm(self, llm_runner: PredictedLlmRunnerHandle) -> None:
        if not self.resp.is_predicted():
            raise _make_illegal_transition(_phase_kind(self.resp), "Predicted(attach_llm)")
        if self.resp.llm_runner is not None:
            raise _make_illegal_transition("Predicted(with-llm)", "Predicted(with-llm)")
        self.resp.llm_runner = llm_runner

    def resp_promote_predicted_to_created(
        self, runtime: ResponseRuntime
    ) -> tuple[str, str, int, PredictedRunner | None, PredictedLlmRunnerHandle | None]:
        if not self.resp.is_predicted():
            raise _make_illegal_transition(_phase_kind(self.resp), "Created (from Predicted)")
        prev = self.resp
        self.resp = RespPhase.created(prev.id or "", prev.item_id or "", prev.epoch or 0)
        self.current_response = runtime
        check_state(self)
        return (prev.id or "", prev.item_id or "", prev.epoch or 0, prev.runner, prev.llm_runner)

    def resp_advance_to_streaming(self, played_ms: _AtomicU64) -> None:
        if self.resp.tag is not _RespPhaseTag.CREATED:
            raise _make_illegal_transition(_phase_kind(self.resp), "Streaming")
        prev = self.resp
        self.resp = RespPhase.streaming(prev.id or "", prev.item_id or "", prev.epoch or 0, played_ms, None)
        check_state(self)

    def resp_drain(self, planned_ms: int) -> None:
        if self.resp.tag is not _RespPhaseTag.STREAMING:
            raise _make_illegal_transition(_phase_kind(self.resp), "Drain")
        prev = self.resp
        if prev.played_ms is None:
            raise _make_illegal_transition(_phase_kind(self.resp), "Drain")
        self.resp = RespPhase.drain(
            prev.id or "", prev.item_id or "", prev.epoch or 0, prev.played_ms, planned_ms
        )
        check_state(self)

    def resp_retire_to_none(self) -> ResponseRuntime | None:
        self.resp = RespPhase.none()
        runtime = self.current_response
        self.current_response = None
        check_state(self)
        return runtime

    def resp_retire_predicted(self) -> PredictedRunner | None:
        runner, _llm = self.resp_retire_predicted_full()
        return runner

    def resp_retire_predicted_full(self) -> tuple[PredictedRunner | None, PredictedLlmRunnerHandle | None]:
        if not self.resp.is_predicted():
            raise _make_illegal_transition(_phase_kind(self.resp), "None (from Predicted)")
        prev = self.resp
        self.resp = RespPhase.none()
        check_state(self)
        return prev.runner, prev.llm_runner
