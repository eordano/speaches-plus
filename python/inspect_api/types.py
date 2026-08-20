from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, Field

@dataclass
class Corr:
    turn_id: str | None = None
    item_id: str | None = None
    response_id: str | None = None
    phrase_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.turn_id is not None:
            out["turn_id"] = self.turn_id
        if self.item_id is not None:
            out["item_id"] = self.item_id
        if self.response_id is not None:
            out["response_id"] = self.response_id
        if self.phrase_id is not None:
            out["phrase_id"] = self.phrase_id
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> Corr:
        if not data:
            return cls()
        return cls(
            turn_id=data.get("turn_id"),
            item_id=data.get("item_id"),
            response_id=data.get("response_id"),
            phrase_id=data.get("phrase_id"),
        )

@dataclass
class WireEvent:
    session_id: str
    seq: int
    ts_mono_ns: int
    ts_wall: float
    lane: str
    kind: str
    corr: Corr
    span_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "session_id": self.session_id,
            "seq": self.seq,
            "ts_mono_ns": self.ts_mono_ns,
            "ts_wall": self.ts_wall,
            "lane": self.lane,
            "kind": self.kind,
            "corr": self.corr.to_dict(),
            "payload": dict(self.payload),
        }
        if self.span_id is not None:
            out["span_id"] = self.span_id
        return out

class SessionMeta(BaseModel):
    id: str
    created_at: float
    model: str
    state: str
    turn_count: int
    last_event_ts: float | None = None

class SessionHistoryEntry(BaseModel):
    id: str
    size_bytes: int
    mtime: float

class AudioQuery(BaseModel):
    channel: str
    from_ms: int = Field(default=0)
    to_ms: int = Field(default=0)
