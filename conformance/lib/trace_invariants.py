#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

KNOWN_EVENT_TYPES: set[str] = {
    "session.created",
    "session.updated",
    "session.done",
    "input_audio_buffer.speech_started",
    "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "input_audio_buffer.cleared",
    "input_audio_buffer.partial_transcription",
    "conversation.item.create",
    "conversation.item.added",
    "conversation.item.done",
    "conversation.item.deleted",
    "conversation.item.truncate",
    "conversation.item.truncated",
    "conversation.item.assistant_truncated",
    "conversation.item.retrieve",
    "conversation.item.retrieved",
    "conversation.item.input_audio_transcription.completed",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.failed",
    "response.created",
    "response.done",
    "response.cancelled",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_text.delta",
    "response.output_text.done",
    "response.output_audio.delta",
    "response.output_audio.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "response.function_call_arguments.delta",
    "response.function_call_arguments.done",
    "response.tool_progress",
    "output_audio_buffer.clear",
    "output_audio_buffer.cleared",
    "output_audio_buffer.started",
    "output_audio_buffer.stopped",
    "rate_limits.updated",
    "error",
    "conversation.item.created",
    "response.audio.delta",
    "response.audio.done",
    "response.audio_transcript.delta",
    "response.audio_transcript.done",
    "response.text.delta",
    "response.text.done",
}

ALIAS_MAP: dict[str, str] = {
    "conversation.item.created": "conversation.item.added",
    "response.audio.delta": "response.output_audio.delta",
    "response.audio.done": "response.output_audio.done",
    "response.audio_transcript.delta": "response.output_audio_transcript.delta",
    "response.audio_transcript.done": "response.output_audio_transcript.done",
    "response.text.delta": "response.output_text.delta",
    "response.text.done": "response.output_text.done",
}

def canonicalize_event_type(t: str) -> str:
    """Return the canonical v3 name for ``t`` (post-§0.3 rename), or ``t`` itself."""
    return ALIAS_MAP.get(t, t)

CANCELLED_STATUSES: set[str] = {"cancelled", "incomplete"}

def load_trace(path: Path) -> tuple[dict, list[dict], dict | None]:
    config: dict | None = None
    events: list[dict] = []
    result: dict | None = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kind = rec.get("kind")
        if kind == "config":
            config = rec
        elif kind == "event":
            ev = rec["event"]
            t = ev.get("type")
            if isinstance(t, str) and t in ALIAS_MAP:
                ev["type"] = ALIAS_MAP[t]
            events.append(ev)
        elif kind == "result":
            result = rec
    if config is None:
        raise SystemExit(f"{path}: no 'config' line")
    return config, events, result

def get_event_id(ev: dict, key: str) -> str | None:
    if key in ev:
        v = ev[key]
        if isinstance(v, str):
            return v
    if isinstance(ev.get("response"), dict):
        v = ev["response"].get(key.removeprefix("response."))
        if isinstance(v, str):
            return v
    if isinstance(ev.get("session"), dict) and key == "id":
        v = ev["session"].get("id")
        if isinstance(v, str):
            return v
    return None

def check_session_created_first(events: list[dict]) -> list[str]:
    if not events:
        return ["no events at all"]
    first = events[0].get("type")
    if first != "session.created":
        return [f"first event was {first!r}, expected 'session.created'"]
    return []

def check_speech_response_atomicity(events: list[dict]) -> list[str]:
    violations: list[str] = []
    speaking = False
    speaking_item: str | None = None
    active_response: str | None = None
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "input_audio_buffer.speech_started":
            speaking = True
            speaking_item = ev.get("item_id") or get_event_id(ev, "item_id")
            if active_response is not None:
                violations.append(
                    f"event {i}: speech_started while response {active_response!r} still active "
                    f"(no response.done between them) -- missing barge-in"
                )
        elif t in ("input_audio_buffer.speech_stopped",
                   "input_audio_buffer.committed"):
            ev_item = ev.get("item_id")
            if speaking_item is not None and (ev_item is None or ev_item == speaking_item):
                speaking = False
                speaking_item = None
        elif t == "response.created":
            rid = get_event_id(ev, "id") or get_event_id(ev, "response.id")
            if speaking:
                violations.append(
                    f"event {i}: response.created during Speaking phase "
                    f"(item={speaking_item!r}) -- would race the inbound utterance"
                )
            active_response = rid
        elif t == "response.done":
            active_response = None
    return violations

def check_no_stuck_user_items(events: list[dict]) -> list[str]:
    started: dict[str, int] = {}
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "input_audio_buffer.speech_started":
            iid = ev.get("item_id")
            if iid:
                started[iid] = i
        elif t in ("conversation.item.input_audio_transcription.completed",
                   "input_audio_buffer.committed"):
            iid = ev.get("item_id")
            if iid in started:
                started.pop(iid)
    return [
        f"item {iid!r} (speech_started @ event {idx}) never received transcription.completed or committed"
        for iid, idx in started.items()
    ]

def check_response_done_per_created(events: list[dict]) -> list[str]:
    created: dict[str, int] = {}
    terminals: dict[str, int] = {}
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "response.created":
            rid = get_event_id(ev, "id") or "<missing>"
            if rid in created:
                return [f"event {i}: duplicate response.created for {rid!r} (first @ {created[rid]})"]
            created[rid] = i
        elif t == "response.done":
            rid = get_event_id(ev, "id") or "<missing>"
            if rid in terminals:
                return [f"event {i}: duplicate response.done for {rid!r} (first @ {terminals[rid]})"]
            terminals[rid] = i
    violations: list[str] = []
    for rid in created:
        if rid not in terminals:
            violations.append(f"response.created {rid!r} (event {created[rid]}) has no response.done")
    for rid in terminals:
        if rid not in created:
            violations.append(f"response.done for {rid!r} (event {terminals[rid]}) has no matching response.created")
    return violations

def check_delta_only_between_created_and_done(events: list[dict]) -> list[str]:
    open_ids: set[str] = set()
    violations: list[str] = []
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "response.created":
            rid = get_event_id(ev, "id") or "<missing>"
            open_ids.add(rid)
            continue
        if t == "response.done":
            rid = get_event_id(ev, "id") or "<missing>"
            open_ids.discard(rid)
            continue
        if t.startswith("response.") and t != "response.tool_progress":
            rid = get_event_id(ev, "id") or get_event_id(ev, "response_id") or None
            if rid is not None and rid not in open_ids:
                violations.append(
                    f"event {i}: {t} for response {rid!r} fired outside of an open response window"
                )
    return violations

def check_committed_after_stopped_before_created(events: list[dict]) -> list[str]:
    stopped_at: dict[str, int] = {}
    committed_at: dict[str, int] = {}
    item_created_at: dict[str, int] = {}
    violations: list[str] = []
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        iid = ev.get("item_id")
        if t == "input_audio_buffer.speech_stopped" and iid:
            stopped_at[iid] = i
        elif t == "input_audio_buffer.committed" and iid:
            committed_at[iid] = i
            if iid in stopped_at and stopped_at[iid] > i:
                violations.append(
                    f"event {i}: committed({iid!r}) preceded speech_stopped @ {stopped_at[iid]}"
                )
        elif t == "conversation.item.added":
            item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
            item_id = item.get("id") if isinstance(item, dict) else None
            if item_id:
                item_created_at[item_id] = i
                if item_id in committed_at and committed_at[item_id] > i:
                    violations.append(
                        f"event {i}: conversation.item.added({item_id!r}) "
                        f"preceded committed @ {committed_at[item_id]}"
                    )
    return violations

def check_response_done_carries_audio_end_ms(events: list[dict]) -> list[str]:
    violations: list[str] = []
    for i, ev in enumerate(events):
        if ev.get("type") != "response.done":
            continue
        resp = ev.get("response") if isinstance(ev.get("response"), dict) else {}
        status = resp.get("status") if isinstance(resp, dict) else ev.get("status")
        audio_end_ms = resp.get("audio_end_ms") if isinstance(resp, dict) else ev.get("audio_end_ms")
        if not isinstance(audio_end_ms, (int, float)):
            rid = get_event_id(ev, "id") or "<missing>"
            violations.append(
                f"event {i}: response.done {rid!r} status={status!r} missing numeric audio_end_ms"
            )
    return violations

def check_no_events_after_response_done(events: list[dict]) -> list[str]:
    done_at: dict[str, int] = {}
    violations: list[str] = []
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "response.done":
            rid = get_event_id(ev, "id") or "<missing>"
            done_at[rid] = i
        elif t.startswith("response.") and t != "response.created" and t != "response.done":
            rid = get_event_id(ev, "id") or get_event_id(ev, "response_id") or "<missing>"
            if rid in done_at:
                violations.append(
                    f"event {i}: {t} for response {rid!r} after response.done @ {done_at[rid]}"
                )
    return violations

W6_BRACKET_TYPES: frozenset[str] = frozenset({
    "response.created",
    "response.output_item.added",
    "response.output_item.done",
    "response.content_part.added",
    "response.content_part.done",
    "response.output_audio_transcript.delta",
    "response.output_audio_transcript.done",
    "response.output_audio.delta",
    "response.output_audio.done",
})

def W6_no_response_events_after_done(events: list[dict]) -> list[str]:
    done_at: dict[str, int] = {}
    for i, ev in enumerate(events):
        if ev.get("type") != "response.done":
            continue
        rid = get_event_id(ev, "id")
        if rid is None:
            continue
        done_at.setdefault(rid, i)

    violations: list[str] = []
    for j, ev in enumerate(events):
        t = ev.get("type", "")
        if t in ("conversation.item.assistant_truncated",
                 "conversation.item.truncate"):
            continue
        if t not in W6_BRACKET_TYPES:
            continue
        rid = (
            get_event_id(ev, "id")
            or get_event_id(ev, "response_id")
            or None
        )
        if rid is None or rid not in done_at:
            continue
        if j > done_at[rid]:
            violations.append(
                f"event {j}: {t} for response_id={rid!r} fired after "
                f"response.done @ event {done_at[rid]} (W6)"
            )
    return violations

def check_cancelled_emits_truncate(events: list[dict]) -> list[str]:
    violations: list[str] = []
    for i, ev in enumerate(events):
        if ev.get("type") != "response.done":
            continue
        resp = ev.get("response") if isinstance(ev.get("response"), dict) else {}
        status = resp.get("status") if isinstance(resp, dict) else ev.get("status")
        if status != "cancelled":
            continue
        rid = get_event_id(ev, "id") or "<missing>"
        cancelled_audio_end_ms = (
            resp.get("audio_end_ms") if isinstance(resp, dict) else ev.get("audio_end_ms")
        )
        output = resp.get("output") if isinstance(resp, dict) else None
        assistant_item_id: str | None = None
        if isinstance(output, list) and output and isinstance(output[0], dict):
            iid = output[0].get("id")
            if isinstance(iid, str):
                assistant_item_id = iid
        found = False
        for j in range(i + 1, len(events)):
            nxt = events[j]
            nt = nxt.get("type", "")
            if nt == "response.created":
                break
            if nt != "conversation.item.assistant_truncated":
                continue
            t_iid = nxt.get("item_id")
            t_aem = nxt.get("audio_end_ms")
            if assistant_item_id is not None and t_iid != assistant_item_id:
                continue
            if not isinstance(t_aem, (int, float)):
                violations.append(
                    f"event {j}: conversation.item.assistant_truncated after response.done {rid!r} "
                    f"missing numeric audio_end_ms"
                )
                found = True
                break
            if isinstance(cancelled_audio_end_ms, (int, float)) and t_aem != cancelled_audio_end_ms:
                violations.append(
                    f"event {j}: conversation.item.assistant_truncated audio_end_ms={t_aem} "
                    f"does not match response.done {rid!r} audio_end_ms={cancelled_audio_end_ms}"
                )
            found = True
            break
        if not found:
            violations.append(
                f"event {i}: response.done(cancelled) {rid!r} (item={assistant_item_id!r}) "
                f"not followed by conversation.item.assistant_truncated before next response.created / EOF"
            )
    return violations

def check_W7_assistant_truncated_paired(events: list[dict]) -> list[str]:
    violations: list[str] = []
    last_done_status: str | None = None
    last_done_index: int | None = None
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "response.done":
            resp = ev.get("response") if isinstance(ev.get("response"), dict) else {}
            last_done_status = (
                resp.get("status") if isinstance(resp, dict) else ev.get("status")
            )
            last_done_index = i
            continue
        if t != "conversation.item.assistant_truncated":
            continue
        if last_done_status is None:
            violations.append(
                f"event {i}: assistant_truncated emitted before any response.done"
            )
            continue
        if last_done_status not in CANCELLED_STATUSES:
            violations.append(
                f"event {i}: assistant_truncated paired with response.done @ "
                f"{last_done_index} status={last_done_status!r} (must be cancelled/incomplete)"
            )
    return violations

def check_W8_client_create_paired(events: list[dict]) -> list[str]:
    requested: dict[str, int] = {}
    answered: dict[str, int] = {}
    violations: list[str] = []
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "conversation.item.create":
            item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
            iid = item.get("id") if isinstance(item, dict) else None
            if isinstance(iid, str):
                if iid in requested:
                    violations.append(
                        f"event {i}: duplicate conversation.item.create for {iid!r}"
                    )
                requested[iid] = i
        elif t == "conversation.item.added":
            item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
            iid = item.get("id") if isinstance(item, dict) else None
            if isinstance(iid, str) and iid in requested:
                if iid in answered:
                    violations.append(
                        f"event {i}: duplicate conversation.item.added for {iid!r}"
                    )
                answered[iid] = i
    for iid, idx in requested.items():
        if iid not in answered:
            violations.append(
                f"event {idx}: conversation.item.create({iid!r}) "
                f"never followed by conversation.item.added"
            )
    return violations

def check_W9_retrieve_paired_with_retrieved(events: list[dict]) -> list[str]:
    """v3 §10.3 W9: every accepted conversation.item.retrieve(id=A) is paired
    with exactly one conversation.item.retrieved(item.id=A). Rejected
    retrievals emit `error` referencing event_id instead.
    """
    requested: dict[str, int] = {}
    request_event_ids: dict[str, str] = {}
    error_event_ids: set[str] = set()
    answered: dict[str, int] = {}
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "conversation.item.retrieve":
            iid = ev.get("item_id")
            eid = ev.get("event_id")
            if isinstance(iid, str):
                if iid in requested:
                    pass
                else:
                    requested[iid] = i
                    if isinstance(eid, str):
                        request_event_ids[iid] = eid
        elif t == "conversation.item.retrieved":
            item = ev.get("item") if isinstance(ev.get("item"), dict) else {}
            iid = item.get("id") if isinstance(item, dict) else None
            if isinstance(iid, str) and iid in requested:
                if iid in answered:
                    return [f"event {i}: duplicate conversation.item.retrieved for {iid!r}"]
                answered[iid] = i
        elif t == "error":
            eid = ev.get("event_id")
            if isinstance(eid, str):
                error_event_ids.add(eid)
    violations: list[str] = []
    for iid, idx in requested.items():
        if iid in answered:
            continue
        if request_event_ids.get(iid) in error_event_ids:
            continue
        violations.append(
            f"event {idx}: conversation.item.retrieve({iid!r}) never followed "
            "by conversation.item.retrieved (and no matching error)"
        )
    return violations

def check_W10_output_audio_started_once_per_response(events: list[dict]) -> list[str]:
    """v3 §10.3 W10: output_audio_buffer.started fires at most once per
    response, only after the first response.output_audio.delta for that
    response, and never after response.done(id=R).
    """
    first_audio_delta_at: dict[str, int] = {}
    response_done_at: dict[str, int] = {}
    started_at: dict[str, list[int]] = {}
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "response.output_audio.delta":
            rid = ev.get("response_id")
            if isinstance(rid, str) and rid not in first_audio_delta_at:
                first_audio_delta_at[rid] = i
        elif t == "response.done":
            r = ev.get("response") if isinstance(ev.get("response"), dict) else {}
            rid = r.get("id") if isinstance(r, dict) else None
            if isinstance(rid, str) and rid not in response_done_at:
                response_done_at[rid] = i
        elif t == "output_audio_buffer.started":
            rid = ev.get("response_id")
            if isinstance(rid, str):
                started_at.setdefault(rid, []).append(i)
    violations: list[str] = []
    for rid, indices in started_at.items():
        if len(indices) > 1:
            violations.append(
                f"event {indices[1]}: output_audio_buffer.started fired twice for response {rid!r}"
            )
        first = indices[0]
        first_delta = first_audio_delta_at.get(rid)
        if first_delta is None:
            violations.append(
                f"event {first}: output_audio_buffer.started({rid!r}) "
                "fired but no response.output_audio.delta for this response"
            )
        elif first <= first_delta:
            violations.append(
                f"event {first}: output_audio_buffer.started({rid!r}) "
                f"fired before first response.output_audio.delta @ event {first_delta}"
            )
        done = response_done_at.get(rid)
        if done is not None and first > done:
            violations.append(
                f"event {first}: output_audio_buffer.started({rid!r}) "
                f"fired after response.done @ event {done}"
            )
    return violations

def check_W11_output_audio_started_stopped_paired(events: list[dict]) -> list[str]:
    """v3 §10.3 W11: every output_audio_buffer.started(response_id=R) is
    paired with exactly one output_audio_buffer.stopped(response_id=R)
    before session.done. The two MAY appear in either order relative to
    response.done(id=R) but MUST reference the same played_ms snapshot.
    """
    started: dict[str, dict] = {}
    stopped: dict[str, dict] = {}
    response_done_audio_end: dict[str, int | None] = {}
    session_done_at: int | None = None
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "output_audio_buffer.started":
            rid = ev.get("response_id")
            if isinstance(rid, str) and rid not in started:
                started[rid] = {"idx": i}
        elif t == "output_audio_buffer.stopped":
            rid = ev.get("response_id")
            if isinstance(rid, str):
                if rid in stopped:
                    return [f"event {i}: duplicate output_audio_buffer.stopped for response {rid!r}"]
                stopped[rid] = {"idx": i, "audio_end_ms": ev.get("audio_end_ms")}
        elif t == "response.done":
            r = ev.get("response") if isinstance(ev.get("response"), dict) else {}
            rid = r.get("id") if isinstance(r, dict) else None
            if isinstance(rid, str):
                response_done_audio_end[rid] = (
                    r.get("audio_end_ms") if isinstance(r, dict) else ev.get("audio_end_ms")
                )
        elif t == "session.done":
            session_done_at = i
    violations: list[str] = []
    for rid in started:
        if rid not in stopped:
            violations.append(
                f"event {started[rid]['idx']}: output_audio_buffer.started({rid!r}) "
                "never paired with output_audio_buffer.stopped"
            )
            continue
        if session_done_at is not None and stopped[rid]["idx"] > session_done_at:
            violations.append(
                f"event {stopped[rid]['idx']}: output_audio_buffer.stopped({rid!r}) "
                "fired after session.done"
            )
        stop_aem = stopped[rid].get("audio_end_ms")
        done_aem = response_done_audio_end.get(rid)
        if isinstance(stop_aem, int) and isinstance(done_aem, int) and stop_aem != done_aem:
            violations.append(
                f"event {stopped[rid]['idx']}: "
                f"output_audio_buffer.stopped({rid!r}).audio_end_ms={stop_aem} "
                f"!= response.done.audio_end_ms={done_aem}"
            )
    return violations

_W12_STREAM_TYPES = (
    "response.output_audio.delta",
    "response.output_audio_transcript.delta",
    "response.output_text.delta",
)

_W12_DONE_TYPES = {
    "response.output_audio.delta": "response.output_audio.done",
    "response.output_audio_transcript.delta": "response.output_audio_transcript.done",
    "response.output_text.delta": "response.output_text.done",
}

W12_DEFAULTS = {
    "max_inter_delta_ms": 1500,
    "max_terminal_stall_ms": 500,
    "predicted_flush_grace_ms": 200,
}

def _has_live_timings(events: list[dict]) -> bool:
    """W12 is live-trace-only -- skip when no event carries `_t_ms`."""
    for ev in events:
        t_ms = ev.get("_t_ms")
        if isinstance(t_ms, (int, float)):
            return True
    return False

def check_W12_deltas_flushed_as_produced(
    events: list[dict],
    *,
    max_inter_delta_ms: int = W12_DEFAULTS["max_inter_delta_ms"],
    max_terminal_stall_ms: int = W12_DEFAULTS["max_terminal_stall_ms"],
    predicted_flush_grace_ms: int = W12_DEFAULTS["predicted_flush_grace_ms"],
) -> list[str]:
    """v3 §10.3 W12: deltas of the three stream types
    (output_audio.delta, output_audio_transcript.delta, output_text.delta)
    must arrive on the wire as the server produces them. Two patterns
    are flagged:

      W12a -- max inter-delta gap > max_inter_delta_ms after the
             predicted-flush window (§8.1.2). This catches mid-stream
             accumulate-and-dump.
      W12b -- terminal stall: the gap between a stream's last delta and
             its `.done` event > max_terminal_stall_ms. This catches
             holding the tail of the stream until the producer signals
             end.

    The predicted-flush burst -- buffered tokens from a `Predicted`-phase
    LLM call that all become send-able at response.created -- is exempt
    from W12a within `predicted_flush_grace_ms` after
    `response.created._t_ms`.

    This check is **live-trace only**. Canonicalized fixtures (no
    `_t_ms`) return zero violations.
    """
    if not _has_live_timings(events):
        return []

    response_created_t: dict[str, int] = {}
    streams: dict[str, dict[str, list[tuple[int, int]]]] = {}
    stream_dones: dict[str, dict[str, tuple[int, int]]] = {}

    for i, ev in enumerate(events):
        t = ev.get("type", "")
        rid_raw = ev.get("response_id")
        t_ms = ev.get("_t_ms")
        if not isinstance(t_ms, (int, float)):
            continue
        t_ms_int = int(t_ms)
        if t == "response.created":
            r = ev.get("response") if isinstance(ev.get("response"), dict) else {}
            rid = r.get("id") if isinstance(r, dict) else None
            if isinstance(rid, str) and rid not in response_created_t:
                response_created_t[rid] = t_ms_int
            continue
        if t in _W12_STREAM_TYPES:
            if isinstance(rid_raw, str):
                streams.setdefault(rid_raw, {}).setdefault(t, []).append((i, t_ms_int))
            continue
        if t in _W12_DONE_TYPES.values() and isinstance(rid_raw, str):
            for delta_t, done_t in _W12_DONE_TYPES.items():
                if t == done_t:
                    stream_dones.setdefault(rid_raw, {})[delta_t] = (i, t_ms_int)
                    break

    violations: list[str] = []
    for rid, by_type in streams.items():
        created_t = response_created_t.get(rid)
        flush_end_t = (
            (created_t + predicted_flush_grace_ms) if created_t is not None else None
        )
        for stream_t, arrivals in by_type.items():
            post_flush = arrivals
            if flush_end_t is not None:
                post_flush = [
                    (idx, t_ms) for (idx, t_ms) in arrivals if t_ms > flush_end_t
                ]
            if len(post_flush) >= 2:
                for j in range(len(post_flush) - 1):
                    gap = post_flush[j + 1][1] - post_flush[j][1]
                    if gap > max_inter_delta_ms:
                        violations.append(
                            f"response {rid!r} stream {stream_t}: "
                            f"inter-delta gap {gap}ms (event {post_flush[j][0]} -> "
                            f"{post_flush[j + 1][0]}) exceeds "
                            f"max_inter_delta_ms={max_inter_delta_ms} (W12a)"
                        )
            done_info = stream_dones.get(rid, {}).get(stream_t)
            if done_info is not None and arrivals:
                last_delta_idx, last_delta_t = arrivals[-1]
                done_idx, done_t = done_info
                if done_idx > last_delta_idx:
                    stall = done_t - last_delta_t
                    if stall > max_terminal_stall_ms:
                        violations.append(
                            f"response {rid!r} stream {stream_t}: "
                            f"terminal stall {stall}ms (last delta @ event "
                            f"{last_delta_idx} -> .done @ event {done_idx}) exceeds "
                            f"max_terminal_stall_ms={max_terminal_stall_ms} (W12b)"
                        )
    return violations

check_W12_streams_parallel = check_W12_deltas_flushed_as_produced

def check_known_event_types(events: list[dict]) -> list[str]:
    seen_unknown: dict[str, int] = {}
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t and t not in KNOWN_EVENT_TYPES:
            if t not in seen_unknown:
                seen_unknown[t] = i
    return [f"unknown event type {t!r} (first @ event {i})" for t, i in seen_unknown.items()]

def check_result_consistent_with_events(events: list[dict], result: dict | None) -> list[str]:
    if result is None:
        return []
    violations: list[str] = []
    has_transcription = any(
        ev.get("type") == "conversation.item.input_audio_transcription.completed"
        for ev in events
    )
    has_response_done_completed = any(
        ev.get("type") == "response.done"
        and (
            ev.get("response", {}).get("status") == "completed"
            if isinstance(ev.get("response"), dict)
            else ev.get("status") == "completed"
        )
        for ev in events
    )
    if result.get("transcription_pass") and not has_transcription:
        violations.append("result.transcription_pass=true but no transcription.completed event")
    if result.get("tts_pass") and not has_response_done_completed:
        violations.append("result.tts_pass=true but no response.done(status=completed) event")
    return violations

CHECKS = {
    "session_created_first": lambda e, r: check_session_created_first(e),
    "W1_response_done_per_created": lambda e, r: check_response_done_per_created(e),
    "W2_delta_only_between_created_and_done": lambda e, r: check_delta_only_between_created_and_done(e),
    "W3_committed_after_stopped_before_created": lambda e, r: check_committed_after_stopped_before_created(e),
    "W4_response_done_carries_audio_end_ms": lambda e, r: check_response_done_carries_audio_end_ms(e),
    "W6_no_response_events_after_done": lambda e, r: W6_no_response_events_after_done(e),
    "W7_assistant_truncated_paired_with_cancelled_done": lambda e, r: check_W7_assistant_truncated_paired(e),
    "W8_client_create_paired_with_server_created": lambda e, r: check_W8_client_create_paired(e),
    "W9_retrieve_paired_with_retrieved": lambda e, r: check_W9_retrieve_paired_with_retrieved(e),
    "W10_output_audio_started_once_per_response": lambda e, r: check_W10_output_audio_started_once_per_response(e),
    "W11_output_audio_started_stopped_paired": lambda e, r: check_W11_output_audio_started_stopped_paired(e),
    "W12_deltas_flushed_as_produced": lambda e, r: check_W12_deltas_flushed_as_produced(e),
    "speech_started_response_atomicity": lambda e, r: check_speech_response_atomicity(e),
    "no_stuck_user_items": lambda e, r: check_no_stuck_user_items(e),
    "no_events_after_response_done": lambda e, r: check_no_events_after_response_done(e),
    "cancelled_emits_truncate": lambda e, r: check_cancelled_emits_truncate(e),
    "known_event_types": lambda e, r: check_known_event_types(e),
    "result_consistent_with_events": lambda e, r: check_result_consistent_with_events(e, r),
}

_W6_VIOLATING_FIXTURE: list[dict] = [
    {"type": "session.created", "session": {"id": "sess_test"}},
    {"type": "response.created", "response": {"id": "resp_x", "status": "in_progress"}},
    {"type": "response.output_audio.delta", "response_id": "resp_x", "delta": "AAA"},
    {"type": "response.done", "response": {"id": "resp_x", "status": "completed", "audio_end_ms": 1000}},
    {"type": "response.output_audio.delta", "response_id": "resp_x", "delta": "BBB"},
]

_W6_CLEAN_FIXTURE: list[dict] = [
    {"type": "session.created", "session": {"id": "sess_test"}},
    {"type": "response.created", "response": {"id": "resp_x", "status": "in_progress"}},
    {"type": "response.output_audio.delta", "response_id": "resp_x", "delta": "AAA"},
    {"type": "response.output_audio.done", "response_id": "resp_x"},
    {"type": "response.done", "response": {"id": "resp_x", "status": "cancelled", "audio_end_ms": 200}},
    {"type": "conversation.item.assistant_truncated", "item_id": "item_x", "audio_end_ms": 200},
]

_W7_VIOLATING_FIXTURE: list[dict] = [
    {"type": "session.created", "session": {"id": "sess_test"}},
    {"type": "response.created", "response": {"id": "resp_x"}},
    {"type": "response.done", "response": {"id": "resp_x", "status": "completed", "audio_end_ms": 100}},
    {"type": "conversation.item.assistant_truncated", "item_id": "item_x", "audio_end_ms": 50},
]

_W8_VIOLATING_FIXTURE: list[dict] = [
    {"type": "session.created", "session": {"id": "sess_test"}},
    {"type": "conversation.item.create", "item": {"id": "item_q", "type": "message", "role": "user"}},
]

_W8_CLEAN_FIXTURE: list[dict] = [
    {"type": "session.created", "session": {"id": "sess_test"}},
    {"type": "conversation.item.create", "item": {"id": "item_q", "type": "message", "role": "user"}},
    {"type": "conversation.item.added", "item": {"id": "item_q"}},
]

def _run_self_test() -> int:
    failures: list[str] = []
    cases = [
        ("W6 violating", W6_no_response_events_after_done, _W6_VIOLATING_FIXTURE, True),
        ("W6 clean", W6_no_response_events_after_done, _W6_CLEAN_FIXTURE, False),
        ("W7 violating", check_W7_assistant_truncated_paired, _W7_VIOLATING_FIXTURE, True),
        ("W7 clean", check_W7_assistant_truncated_paired, _W6_CLEAN_FIXTURE, False),
        ("W8 violating", check_W8_client_create_paired, _W8_VIOLATING_FIXTURE, True),
        ("W8 clean", check_W8_client_create_paired, _W8_CLEAN_FIXTURE, False),
    ]
    for name, fn, fixture, expect_violation in cases:
        violations = fn(fixture)
        if expect_violation and not violations:
            failures.append(f"{name}: expected violation, got none")
        elif not expect_violation and violations:
            failures.append(f"{name}: unexpected violation: {violations!r}")
        else:
            print(f"  [PASS] {name}")
    if failures:
        for f in failures:
            print(f"  [FAIL] {f}")
        return 1
    return 0

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace", type=Path, nargs="?")
    p.add_argument("--skip", default="",
                   help="Comma-separated check names to skip (see source for list).")
    p.add_argument("--only", default="",
                   help="Comma-separated check names to run; others are skipped.")
    p.add_argument("--list", action="store_true", help="Print check names and exit.")
    p.add_argument("--self-test", action="store_true",
                   help="Run inline synthetic-fixture self-tests for the W6 check.")
    args = p.parse_args()

    if args.list:
        for name in CHECKS:
            print(name)
        return 0

    if args.self_test:
        return _run_self_test()

    if args.trace is None:
        p.error("trace path required (or pass --self-test / --list)")

    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    _, events, result = load_trace(args.trace)

    any_failed = False
    for name, fn in CHECKS.items():
        if name in skip:
            print(f"  [SKIP] {name}")
            continue
        if only and name not in only:
            continue
        violations = fn(events, result)
        if violations:
            any_failed = True
            print(f"  [FAIL] {name}")
            for v in violations:
                print(f"    - {v}")
        else:
            print(f"  [PASS] {name}")
    return 1 if any_failed else 0

if __name__ == "__main__":
    sys.exit(main())
