#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""run_fixture -- RFC v3 §15.6 conformance fixture runner.

Drives a fixture under <repo>/conformance/fixtures/<name>/ against the
canonical assertion library at <repo>/conformance/lib/. Two modes:

  1. validate-only:  run_fixture.py <fixture-dir>
       Validates fixture contents (input.jsonl, expected.jsonl), runs the
       expected.jsonl trace through the canonical assertions, and reports
       any wire-invariant violations. This is the cheap CI gate -- no impl
       needed.

  2. live-impl (W7+W8 of §18 conformance):
       run_fixture.py --impl <impl-url> <fixture-dir>
       (Live-impl driving is intentionally a stub right now; client/
        test_e2e.py and test_e2e_full.py drive the impl over WebRTC. The
        runner is the intended single CLI entry point per §15.6.)

Exit code: 0 on pass, 1 on any violation, 2 on usage error.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

def fail(msg: str, code: int = 1) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return code

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def lib_path() -> Path:
    return repo_root() / "conformance" / "lib" / "trace_invariants.py"

def diff_path() -> Path:
    return repo_root() / "conformance" / "lib" / "trace_diff.py"

def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    for ln, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise SystemExit(f"{path}:{ln}: invalid JSON: {e}")
    return out

def wrap_as_e2e_trace(events: list[dict], source: str) -> str:
    """Wrap a raw events list in the {kind:config|event} envelope the
    canonical loader expects."""
    lines = [json.dumps({"kind": "config", "source": source})]
    for ev in events:
        lines.append(json.dumps({"kind": "event", "event": ev}))
    return "\n".join(lines) + "\n"

def run_canonical(trace_text: str, only: str | None) -> tuple[int, str]:
    """Run conformance/lib/trace_invariants.py on a trace string.
    Returns (exit_code, combined_output)."""
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
        fh.write(trace_text)
        path = fh.name
    cmd = [sys.executable, str(lib_path())]
    if only:
        cmd.extend(["--only", only])
    cmd.append(path)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr

W_INVARIANTS = ",".join([
    "W1_response_done_per_created",
    "W2_delta_only_between_created_and_done",
    "W3_committed_after_stopped_before_created",
    "W4_response_done_carries_audio_end_ms",
    "W6_no_response_events_after_done",
    "W7_assistant_truncated_paired_with_cancelled_done",
    "W8_client_create_paired_with_server_created",
])

def cmd_validate(fixture_dir: Path, *, only_w: bool) -> int:
    if not fixture_dir.is_dir():
        return fail(f"not a directory: {fixture_dir}", 2)
    input_path = fixture_dir / "input.jsonl"
    expected_path = fixture_dir / "expected.jsonl"
    if not input_path.exists():
        return fail(f"missing {input_path}", 2)
    if not expected_path.exists():
        return fail(f"missing {expected_path}", 2)
    events = load_jsonl(expected_path)
    if not events:
        return fail(f"empty expected trace: {expected_path}")
    only = W_INVARIANTS if only_w else None
    rc, out = run_canonical(wrap_as_e2e_trace(events, source=fixture_dir.name), only)
    print(out, end="")
    if rc != 0:
        return fail(f"{fixture_dir.name}: canonical reported violations (exit={rc})")
    print(f"[PASS] {fixture_dir.name}")
    return 0

def cmd_validate_all(fixtures_dir: Path, *, only_w: bool) -> int:
    if not fixtures_dir.is_dir():
        return fail(f"not a directory: {fixtures_dir}", 2)
    failures = 0
    ran = 0
    for entry in sorted(fixtures_dir.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not (entry / "expected.jsonl").exists():
            continue
        ran += 1
        rc = cmd_validate(entry, only_w=only_w)
        if rc != 0:
            failures += 1
    if ran == 0:
        return fail(f"no fixtures found under {fixtures_dir}", 2)
    if failures:
        return fail(f"{failures}/{ran} fixtures failed")
    print(f"[OK] {ran} fixtures passed")
    return 0

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fixture", nargs="?", type=Path,
                   help="Path to a single fixture directory. If omitted, --all "
                        "validates every fixture under conformance/fixtures/.")
    p.add_argument("--all", action="store_true",
                   help="Validate every fixture under conformance/fixtures/.")
    p.add_argument("--impl", default=None, metavar="URL",
                   help="(reserved) drive a live impl at URL; not yet implemented.")
    p.add_argument("--strict", action="store_true",
                   help="Run all canonical checks, not just the W1..W8 wire "
                        "invariants. Strict mode also exercises Python-side "
                        "consistency checks (no_stuck_user_items, etc.) that "
                        "assume a complete trace.")
    args = p.parse_args()

    if args.impl:
        return fail("--impl driving is not yet implemented; "
                    "use client/test_e2e_*.py for live runs", 2)

    only_w = not args.strict

    if args.all or args.fixture is None:
        return cmd_validate_all(repo_root() / "conformance" / "fixtures", only_w=only_w)
    return cmd_validate(args.fixture, only_w=only_w)

if __name__ == "__main__":
    sys.exit(main())
