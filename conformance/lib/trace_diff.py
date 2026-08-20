#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

UUID_FIELDS = {
    "id",
    "item_id",
    "response_id",
    "event_id",
    "session_id",
    "previous_item_id",
}

ID_PATTERN = re.compile(
    r"^(sess|item|resp|event|msg)_[A-Za-z0-9]{6,}$"
)

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
            events.append(rec)
        elif kind == "result":
            result = rec
    if config is None:
        raise SystemExit(f"{path}: no 'config' line")
    return config, events, result

class IdCanonicaliser:
    def __init__(self) -> None:
        self.next_idx: dict[str, int] = {}
        self.mapping: dict[str, str] = {}

    def canon(self, value: str) -> str:
        if value in self.mapping:
            return self.mapping[value]
        m = ID_PATTERN.match(value)
        if not m:
            return value
        prefix = m.group(1)
        idx = self.next_idx.get(prefix, 0)
        self.next_idx[prefix] = idx + 1
        placeholder = f"<{prefix}:{idx}>"
        self.mapping[value] = placeholder
        return placeholder

    def walk(self, obj):
        if isinstance(obj, dict):
            return {k: self.walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.walk(v) for v in obj]
        if isinstance(obj, str):
            return self.canon(obj)
        return obj

def canonicalise_events(events: list[dict]) -> list[dict]:
    can = IdCanonicaliser()
    out = []
    for rec in events:
        ev = rec.get("event", rec)
        out.append(can.walk(ev))
    return out

def event_summary(ev: dict) -> str:
    t = ev.get("type", "?")
    extra = []
    for k in ("transcript", "delta", "status", "code"):
        if k in ev:
            v = ev[k]
            extra.append(f"{k}={v!r}")
        elif isinstance(ev.get("response"), dict) and k in ev["response"]:
            v = ev["response"][k]
            extra.append(f"response.{k}={v!r}")
    return t + (" " + " ".join(extra) if extra else "")

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("trace_a", type=Path)
    p.add_argument("trace_b", type=Path)
    p.add_argument("--strict-text", action="store_true",
                   help="Require transcript/delta strings to match byte-for-byte. "
                        "Default: types and structure must match; text fields only need to be present.")
    p.add_argument("--ignore-types", default="",
                   help="Comma-separated event types to skip on both sides "
                        "(e.g. response.output_audio_transcript.delta which is high-volume noise).")
    args = p.parse_args()

    cfg_a, ev_a, res_a = load_trace(args.trace_a)
    cfg_b, ev_b, res_b = load_trace(args.trace_b)

    config_diff = []
    for k in sorted(set(cfg_a) | set(cfg_b)):
        if k == "kind" or k.startswith("target"):
            continue
        if cfg_a.get(k) != cfg_b.get(k):
            config_diff.append((k, cfg_a.get(k), cfg_b.get(k)))
    if config_diff:
        print(f"[FAIL] config differs between {args.trace_a} and {args.trace_b}")
        for k, va, vb in config_diff:
            print(f"  {k}: A={va!r}  B={vb!r}")
        return 1

    ignore = {s.strip() for s in args.ignore_types.split(",") if s.strip()}
    can_a = [e for e in canonicalise_events(ev_a) if e.get("type") not in ignore]
    can_b = [e for e in canonicalise_events(ev_b) if e.get("type") not in ignore]

    if not args.strict_text:
        for e in (*can_a, *can_b):
            for k in ("transcript", "delta"):
                if k in e and isinstance(e[k], str):
                    e[k] = "<text>"

    if can_a == can_b:
        print(f"[PASS] {len(can_a)} events match (canonicalised IDs"
              f"{'; loose text' if not args.strict_text else ''})")
        return 0

    print(f"[FAIL] traces diverge")
    print(f"  A: {args.trace_a} ({len(can_a)} events)")
    print(f"  B: {args.trace_b} ({len(can_b)} events)")
    for i, (a, b) in enumerate(zip(can_a, can_b)):
        if a != b:
            print(f"\n  first divergence at event index {i}:")
            print(f"    A: {event_summary(a)}")
            print(f"    B: {event_summary(b)}")
            print(f"\n  full A: {json.dumps(a, indent=2)}")
            print(f"\n  full B: {json.dumps(b, indent=2)}")
            return 1
    if len(can_a) != len(can_b):
        longer, name = (can_a, "A") if len(can_a) > len(can_b) else (can_b, "B")
        n_extra = abs(len(can_a) - len(can_b))
        print(f"\n  {name} has {n_extra} trailing event(s):")
        for e in longer[len(can_a) if name == "B" else len(can_b):]:
            print(f"    + {event_summary(e)}")
        return 1
    return 1

if __name__ == "__main__":
    sys.exit(main())
