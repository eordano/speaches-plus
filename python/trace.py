from __future__ import annotations

import functools
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, TypeVar

import otel

F = TypeVar("F", bound=Callable[..., Any])

TS_FIELDS: tuple[str, ...] = ("ts_ms", "created_at", "audio_start_ms", "audio_end_ms")
FLOAT_FIELDS: tuple[str, ...] = (
    "score",
    "eou.score",
    "eou.eager_score",
    "eou.threshold",
    "vad.probability",
)
ID_PREFIXES: tuple[tuple[str, str], ...] = (
    ("sess_", "sess"),
    ("item_", "item"),
    ("resp_", "resp"),
    ("evt_", "evt"),
)

@dataclass
class CanonicalTrace:
    events: list[Any] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.events)

    def is_empty(self) -> bool:
        return not self.events

def canonicalize_trace(trace: list[Any]) -> CanonicalTrace:
    id_map: dict[str, str] = {}
    counters: dict[str, int] = {}
    out: list[Any] = []
    for ev in trace:
        cloned = _deepcopy_jsonish(ev)
        _canonicalize_node(cloned, id_map, counters, parent_key=None)
        out.append(cloned)
    return CanonicalTrace(events=out)

def _deepcopy_jsonish(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _deepcopy_jsonish(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_deepcopy_jsonish(x) for x in v]
    return v

def _canonicalize_node(
    v: Any,
    id_map: dict[str, str],
    counters: dict[str, int],
    parent_key: str | None,
) -> Any:
    if isinstance(v, dict):
        for k in list(v.keys()):
            child = v[k]
            if k in TS_FIELDS:
                if isinstance(child, int) and not isinstance(child, bool) and child >= 0:
                    v[k] = 0
                continue
            if k in FLOAT_FIELDS:
                if isinstance(child, (int, float)) and not isinstance(child, bool):
                    v[k] = round(float(child), 3)
                continue
            if isinstance(child, str):
                canon = _canon_id(child, id_map, counters)
                if canon is not None:
                    v[k] = canon
                elif k in ("audio", "data"):
                    v[k] = f"<{len(child)} bytes>"
                continue
            _canonicalize_node(child, id_map, counters, parent_key=k)
        return v
    if isinstance(v, list):
        for i, item in enumerate(v):
            if isinstance(item, str):
                canon = _canon_id(item, id_map, counters)
                if canon is not None:
                    v[i] = canon
            else:
                _canonicalize_node(item, id_map, counters, parent_key=parent_key)
        return v
    if isinstance(v, str):
        canon = _canon_id(v, id_map, counters)
        if canon is not None:
            return canon
    return v

def _canon_id(
    s: str,
    id_map: dict[str, str],
    counters: dict[str, int],
) -> str | None:
    for prefix, kind in ID_PREFIXES:
        if s.startswith(prefix):
            existing = id_map.get(s)
            if existing is not None:
                return existing
            n = counters.get(kind, 0) + 1
            counters[kind] = n
            canon = f"{kind}_{n}"
            id_map[s] = canon
            return canon
    return None

def trace_diff(a: CanonicalTrace, b: CanonicalTrace) -> int | None:
    n = min(len(a.events), len(b.events))
    for i in range(n):
        if a.events[i] != b.events[i]:
            return i
    if len(a.events) != len(b.events):
        return n
    return None

def init() -> bool:
    return otel.init()

@contextmanager
def span(name: str, **attributes: Any) -> Iterator[Any]:
    if not otel.is_enabled():
        yield None
        return
    try:
        from opentelemetry import trace as ot_trace

        tracer = ot_trace.get_tracer(otel.TRACER_NAME)
        with tracer.start_as_current_span(name) as sp:
            for k, val in attributes.items():
                try:
                    sp.set_attribute(k, val)
                except Exception:
                    pass
            yield sp
    except ImportError:
        yield None
    except Exception:
        yield None

def traced(name: str | None = None, **span_attributes: Any) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        span_name = name if name is not None else f"{fn.__module__}.{fn.__qualname__}"

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with span(span_name, **span_attributes):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
