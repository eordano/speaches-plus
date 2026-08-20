#!/usr/bin/env python3
"""Drive the Python realtime SessionState from each conformance fixture's
input.jsonl and diff the emitted canonical trace against expected.jsonl --
the Python counterpart of go/internal/realtime/conformance_test.go::replay.

Run:  python3 conformance_replay.py            # all fixtures
      python3 conformance_replay.py 002-...     # one fixture

Exit 0 iff every fixture's canonical trace matches expected.jsonl.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "conformance" / "fixtures"
sys.path.insert(0, str(ROOT / "conformance" / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from trace_diff import canonicalise_events  # noqa: E402
import realtime.state as st  # noqa: E402

CREATED = st._RespPhaseTag.CREATED
STREAMING = st._RespPhaseTag.STREAMING
DRAIN = st._RespPhaseTag.DRAIN
ACTIVE = (CREATED, STREAMING, DRAIN)

def _runtime() -> st.ResponseRuntime:
    return st.ResponseRuntime(handle=None)

def replay(ops: list[dict]) -> list[dict]:
    s = st.SessionState()
    trace: list[dict] = []
    started = False
    planned: dict[int, int] = {}
    played: dict[int, int] = {}

    def emit(ev: dict) -> None:
        trace.append(ev)

    def maybe_start() -> None:
        nonlocal started
        if started:
            return
        s.session = st.SessionPhase.active(0)
        emit({"type": "session.created", "session": {"id": "sess_1"}})
        started = True

    for i, op in enumerate(ops):
        name = op.get("op")
        if name == "markActive":
            maybe_start()
        elif name == "session_update":
            maybe_start()
            emit({"type": "session.updated", "session": {"id": "sess_1"}})
        elif name == "session_update_invalid":
            maybe_start()
            emit({"type": "error", "code": op.get("code") or "session_update_invalid",
                  "message": op.get("message", "")})
            emit({"type": "session.updated", "session": {"id": "sess_1"}})
        elif name == "vad_speech_start":
            maybe_start()
            item = op.get("item_id", "")
            start_ms = int(op.get("start_ms", 0))
            if s.resp.tag in ACTIVE:
                ep = s.resp.epoch or 0
                pl = played.get(ep, 0)
                emit({"type": "response.done", "response": {
                    "id": s.resp.id or "", "status": "cancelled",
                    "audio_end_ms": pl, "output": [{"id": s.resp.item_id or ""}]}})
                emit({"type": "conversation.item.assistant_truncated",
                      "item_id": s.resp.item_id or "", "audio_end_ms": pl})
                s.resp_retire_to_none()
            if s.vad.is_speaking():
                pass
            elif s.vad.is_stopped():
                s.vad = st.VadPhase.speaking(s.vad.item_id or "", s.vad.audio_start_ms or 0)
            else:
                s.vad = st.VadPhase.speaking(item, start_ms)
                emit({"type": "input_audio_buffer.speech_started",
                      "item_id": item, "audio_start_ms": start_ms})
        elif name == "vad_speech_end":
            maybe_start()
            end_ms = int(op.get("end_ms", 0))
            if s.vad.is_speaking():
                item = s.vad.item_id or ""
                start = s.vad.audio_start_ms or 0
                s.vad = st.VadPhase.stopped(item, start, end_ms)
                emit({"type": "input_audio_buffer.speech_stopped",
                      "item_id": item, "audio_end_ms": end_ms})
        elif name == "commit_fire":
            maybe_start()
            if s.vad.is_stopped():
                item = s.vad.item_id or ""
                s.vad = st.VadPhase.silent()
                s.conversation.append(st.ConversationItem.new_user_audio(item))
                emit({"type": "input_audio_buffer.committed", "item_id": item})
                emit({"type": "conversation.item.added",
                      "item": {"id": item, "role": "user"}})
        elif name == "transcription_complete":
            maybe_start()
            emit({"type": "conversation.item.input_audio_transcription.completed",
                  "item_id": op.get("item_id", ""), "transcript": op.get("transcript", "")})
        elif name == "response_create":
            maybe_start()
            rid = op.get("resp_id", "")
            iid = op.get("item_id", "")
            s.resp_create_from_none(rid, iid, _runtime())
            ev = {"type": "response.created", "response": {"id": rid}}
            if "instructions" in op:
                ev["response"]["instructions"] = op["instructions"]
            emit(ev)
        elif name == "audio_delta":
            ab = int(op.get("audio_bytes", 0)) or 1024
            emit({"type": "response.output_audio.delta",
                  "response_id": op.get("resp_id", ""), "audio": {"audio_bytes": ab}})
        elif name == "llm_complete":
            ep = int(op.get("epoch", 0))
            planned[ep] = planned.get(ep, 0) + int(op.get("planned_ms", 0))
            if s.resp.tag is CREATED:
                s.resp_advance_to_streaming(st._AtomicU64(0))
            if s.resp.tag is STREAMING:
                s.resp_drain(planned[ep])
        elif name == "audio_drained":
            ep = int(op.get("epoch", 0))
            pl = int(op.get("played_ms", 0))
            played[ep] = pl
            if s.resp.tag is DRAIN and pl >= planned.get(ep, 0):
                rid = s.resp.id or ""
                emit({"type": "response.output_audio.done",
                      "response": {"id": rid, "audio_end_ms": pl}})
                emit({"type": "response.done",
                      "response": {"id": rid, "status": "completed", "audio_end_ms": pl}})
                s.resp_retire_to_none()
        elif name == "response_failed":
            pl = int(op.get("played_ms", 0))
            reason = op.get("reason") or "llm_error"
            if s.resp.tag not in ACTIVE:
                raise RuntimeError(f"op {i} response_failed: no in-flight response")
            emit({"type": "response.done", "response": {
                "id": s.resp.id or "", "status": "failed", "audio_end_ms": pl,
                "status_details": {"reason": reason},
                "output": [{"id": s.resp.item_id or ""}]}})
            s.resp_retire_to_none()
        elif name == "response_drain_cap_expired":
            ep = int(op.get("epoch", 0))
            pl = int(op.get("played_ms", 0))
            planned[ep] = planned.get(ep, 0) + int(op.get("planned_ms", 0))
            if s.resp.tag is CREATED:
                s.resp_advance_to_streaming(st._AtomicU64(0))
            if s.resp.tag is STREAMING:
                s.resp_drain(planned[ep])
            played[ep] = pl
            if s.resp.tag is DRAIN:
                emit({"type": "response.done", "response": {
                    "id": s.resp.id or "", "status": "incomplete", "audio_end_ms": pl,
                    "status_details": {"reason": "drain_cap"},
                    "output": [{"id": s.resp.item_id or ""}]}})
                s.resp_retire_to_none()
        else:
            raise RuntimeError(f"op {i}: unknown op {name!r}")
    return trace

def load_jsonl(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(("//", "#")):
            continue
        out.append(json.loads(line))
    return out

def diff(expected: list[dict], actual: list[dict]) -> int:
    ce = canonicalise_events(expected)
    ca = canonicalise_events(actual)
    n = max(len(ce), len(ca))
    for i in range(n):
        e = ce[i] if i < len(ce) else None
        a = ca[i] if i < len(ca) else None
        if e != a:
            return i
    return -1

def main() -> int:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    dirs = sorted(d for d in FIXTURES.iterdir() if d.is_dir() and not d.name.startswith("."))
    fails = []
    n = 0
    for d in dirs:
        if only and only not in d.name:
            continue
        inp = d / "input.jsonl"
        exp = d / "expected.jsonl"
        if not inp.is_file() or not exp.is_file():
            continue
        n += 1
        expected = load_jsonl(exp)
        try:
            actual = replay(load_jsonl(inp))
        except Exception as e:  # noqa: BLE001
            fails.append((d.name, f"replay raised: {e}"))
            continue
        idx = diff(expected, actual)
        if idx < 0:
            print(f"[PASS] {d.name}")
        else:
            ce = canonicalise_events(expected)
            ca = canonicalise_events(actual)
            e = ce[idx] if idx < len(ce) else None
            a = ca[idx] if idx < len(ca) else None
            fails.append((d.name, f"diverge at {idx}:\n    expected={json.dumps(e)}\n    actual  ={json.dumps(a)}"))
            print(f"[FAIL] {d.name}")
    print(f"\n{n - len(fails)}/{n} fixtures match")
    for name, msg in fails:
        print(f"--- {name} ---\n  {msg}")
    return 0 if not fails else 1

if __name__ == "__main__":
    raise SystemExit(main())
