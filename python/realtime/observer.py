from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class SessionObserver(Protocol):
    def on_session_start(self, sid: str, meta: dict[str, Any]) -> None: ...

    def on_session_end(self, sid: str) -> None: ...

    def on_outbound_event(self, ev: Any) -> None: ...

    def on_outbound_event_dict(self, ev: dict[str, Any]) -> None: ...

    def on_inbound_event(self, ev_kind: str, payload: dict[str, Any], raw_text: str) -> None: ...

    def on_error(self, code: str, message: str, event_id: str | None, param: str | None) -> None: ...

    def on_inbound_audio_pcm16(self, pcm: bytes) -> None: ...

    def on_outbound_audio_pcm16(self, pcm: bytes) -> None: ...

    def on_inbound_audio_f32(self, samples: Any) -> None: ...

    def on_outbound_audio_f32(self, samples: Any) -> None: ...

    def on_correlation(
        self,
        *,
        response_id: str | None = ...,
        item_id: str | None = ...,
        turn_id: str | None = ...,
        phrase_id: str | None = ...,
    ) -> None: ...

    def on_eou_scored(
        self,
        sid: str,
        *,
        kind: str,
        score: float,
        eager_score: float | None,
        threshold: float,
        language: str | None,
        input_chars: int | None,
        input_audio_ms: int | None,
        delay_ms: int,
        elapsed_ms: int,
        cancelled_by: str,
        hard_cap_fired: bool,
    ) -> None: ...

    def on_eou_hard_cap_fired(
        self,
        sid: str,
        *,
        item_id: str,
        phase: str,
        score: float | None,
    ) -> None: ...

    def on_eou_eager_dispatch(
        self,
        sid: str,
        *,
        response_id: str,
        item_id: str,
        score: float,
        threshold: float,
        epoch: int,
    ) -> None: ...

    def on_predicted_suppressed(
        self,
        sid: str,
        *,
        score: float,
        inflight: int,
    ) -> None: ...

    def on_predicted_promoted(
        self,
        sid: str,
        *,
        response_id: str,
        score: float,
    ) -> None: ...

    def on_predicted_overflow(
        self,
        sid: str,
        *,
        response_id: str,
        dropped_tokens: int,
    ) -> None: ...

    def on_predicted_rollback(
        self,
        sid: str,
        *,
        response_id: str,
        reason: str,
        llm_chars_thrown: int,
    ) -> None: ...

class NullObserver:
    def on_session_start(self, sid: str, meta: dict[str, Any]) -> None:
        pass

    def on_session_end(self, sid: str) -> None:
        pass

    def on_outbound_event(self, ev: Any) -> None:
        pass

    def on_outbound_event_dict(self, ev: dict[str, Any]) -> None:
        pass

    def on_inbound_event(self, ev_kind: str, payload: dict[str, Any], raw_text: str) -> None:
        pass

    def on_error(self, code: str, message: str, event_id: str | None, param: str | None) -> None:
        pass

    def on_inbound_audio_pcm16(self, pcm: bytes) -> None:
        pass

    def on_outbound_audio_pcm16(self, pcm: bytes) -> None:
        pass

    def on_inbound_audio_f32(self, samples: Any) -> None:
        pass

    def on_outbound_audio_f32(self, samples: Any) -> None:
        pass

    def on_correlation(
        self,
        *,
        response_id: Any = ...,
        item_id: Any = ...,
        turn_id: Any = ...,
        phrase_id: Any = ...,
    ) -> None:
        pass

    def on_eou_scored(
        self,
        sid: str,
        *,
        kind: str,
        score: float,
        eager_score: float | None,
        threshold: float,
        language: str | None,
        input_chars: int | None,
        input_audio_ms: int | None,
        delay_ms: int,
        elapsed_ms: int,
        cancelled_by: str,
        hard_cap_fired: bool,
    ) -> None:
        pass

    def on_eou_hard_cap_fired(
        self,
        sid: str,
        *,
        item_id: str,
        phase: str,
        score: float | None,
    ) -> None:
        pass

    def on_eou_eager_dispatch(
        self,
        sid: str,
        *,
        response_id: str,
        item_id: str,
        score: float,
        threshold: float,
        epoch: int,
    ) -> None:
        pass

    def on_predicted_suppressed(
        self,
        sid: str,
        *,
        score: float,
        inflight: int,
    ) -> None:
        pass

    def on_predicted_promoted(
        self,
        sid: str,
        *,
        response_id: str,
        score: float,
    ) -> None:
        pass

    def on_predicted_overflow(
        self,
        sid: str,
        *,
        response_id: str,
        dropped_tokens: int,
    ) -> None:
        pass

    def on_predicted_rollback(
        self,
        sid: str,
        *,
        response_id: str,
        reason: str,
        llm_chars_thrown: int,
    ) -> None:
        pass

__all__ = ["SessionObserver", "NullObserver"]
