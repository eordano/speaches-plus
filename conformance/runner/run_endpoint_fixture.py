#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""run_endpoint_fixture -- PRD §11.4 manifest-fixture runner.

Companion to `run_fixture.py` (which drives the Realtime corpus). This
runner picks up directories under `conformance/fixtures/<NNN>-<family>-<name>/`
that contain a `fixture.json` (NOT `input.jsonl` + `expected.jsonl`) and
performs structural validation:

  1. Schema-validate fixture.json (required top-level fields, band-derived
     family naming, name-matches-directory).
  2. Verify referenced input artifacts (input_artifacts entries and
     request_multipart `field@path` entries) exist on disk and, for WAVs,
     have a valid RIFF header with the declared sample rate / channels /
     duration.
  3. (Future) Drive a live target with `--target URL` and a model loaded.

Two manifest kinds share the schema. `endpoint` fixtures declare an
`endpoint` or `steps` block and are drivable over HTTP. `declarative`
fixtures (050 diarization, 060 eou, 070/071 ocr) carry gates consumed by
in-process Go/Rust tests instead; this runner validates their shape and
reports them as skipped rather than driving them.

The fixture.json schema is *intentionally* loose -- placeholder mode is
about locking the shape of the assertions before the model lands. Once
real models are wired, the per-fixture `ref_outputs` block is consumed by
the live-driving path to assert byte/regex/envelope equality.

Exit code:
  0   all fixtures structurally valid (or skipped because they're not
      drivable over HTTP)
  1   one or more fixtures failed validation
  2   usage / missing files
"""
from __future__ import annotations

import argparse
import json
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED_TOP_LEVEL = {"name", "family", "description"}
FAMILY_BANDS = {
    "020": "020-chat-completions",
    "030": "030-voice-clone",
    "040": "040-align",
    "050": "050-diarization",
    "060": "060-eou",
    "070": "070-ocr",
    "071": "071-ocr-layout",
}
ENDPOINT_FAMILIES = {
    "020-chat-completions",
    "030-voice-clone",
    "040-align",
}
ENDPOINT_OR_STEPS = {"endpoint", "steps"}
BAND_RE = re.compile(r"^(\d{3})-")

@dataclass
class FixtureResult:
    path: Path
    name: str
    family: str
    ok: bool
    skipped: bool
    skip_reason: str = ""
    violations: list[str] = None

    def __post_init__(self) -> None:
        if self.violations is None:
            self.violations = []

def fail(msg: str, code: int = 1) -> int:
    print(f"[FAIL] {msg}", file=sys.stderr)
    return code

def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent

def fixtures_root() -> Path:
    return repo_root() / "conformance" / "fixtures"

def validate_wav_header(path: Path, expected_sr: int | None,
                        expected_channels: int | None,
                        expected_duration_ms: int | None) -> list[str]:
    """Validate a RIFF/WAVE header. Returns a list of violation strings;
    empty list means OK. Implementation is intentionally minimal -- we
    only assert what the fixtures declare."""
    viols: list[str] = []
    try:
        data = path.read_bytes()
    except OSError as e:
        return [f"cannot read {path}: {e}"]
    if len(data) < 44:
        return [f"{path}: too short ({len(data)} bytes) for a WAV header"]
    if data[0:4] != b"RIFF":
        viols.append(f"{path}: missing RIFF tag")
    if data[8:12] != b"WAVE":
        viols.append(f"{path}: missing WAVE tag")
    if data[12:16] != b"fmt ":
        viols.append(f"{path}: missing fmt chunk")
    if viols:
        return viols
    sr = struct.unpack("<I", data[24:28])[0]
    channels = struct.unpack("<H", data[22:24])[0]
    bps = struct.unpack("<H", data[34:36])[0]
    if expected_sr is not None and sr != expected_sr:
        viols.append(f"{path}: sample_rate={sr}, expected {expected_sr}")
    if expected_channels is not None and channels != expected_channels:
        viols.append(f"{path}: channels={channels}, expected {expected_channels}")
    if expected_duration_ms is not None:
        data_idx = data.find(b"data", 36)
        if data_idx < 0:
            viols.append(f"{path}: missing data chunk")
        else:
            data_size = struct.unpack("<I", data[data_idx + 4:data_idx + 8])[0]
            bytes_per_sample = (bps // 8) * channels
            n_samples = data_size // max(bytes_per_sample, 1)
            actual_ms = int(round(n_samples * 1000 / max(sr, 1)))
            if abs(actual_ms - expected_duration_ms) > 50:
                viols.append(
                    f"{path}: duration_ms={actual_ms}, expected ~{expected_duration_ms}"
                )
    return viols

def find_multipart_file_refs(node: object) -> list[str]:
    """Walk a JSON tree, returning all values of keys ending in '@' (the
    fixture convention for 'this multipart field is a file path relative
    to the fixture dir'). Also accepts the dict form {"path": "..."}
    inside input_artifacts."""
    out: list[str] = []
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(k, str) and k.endswith("@") and isinstance(v, str):
                out.append(v)
            elif isinstance(v, (dict, list)):
                out.extend(find_multipart_file_refs(v))
    elif isinstance(node, list):
        for item in node:
            out.extend(find_multipart_file_refs(item))
    return out

def validate_fixture(fdir: Path) -> FixtureResult:
    fpath = fdir / "fixture.json"
    name = fdir.name
    if not fpath.is_file():
        return FixtureResult(fdir, name, "?", ok=True, skipped=True,
                             skip_reason="no fixture.json")
    try:
        data = json.loads(fpath.read_text())
    except json.JSONDecodeError as e:
        return FixtureResult(fdir, name, "?", ok=False, skipped=False,
                             violations=[f"invalid JSON: {e}"])

    viols: list[str] = []
    missing = REQUIRED_TOP_LEVEL - set(data.keys())
    if missing:
        viols.append(f"missing top-level fields: {sorted(missing)}")

    family = data.get("family", "?")
    band_m = BAND_RE.match(name)
    band = band_m.group(1) if band_m else ""
    expected_family = FAMILY_BANDS.get(band)
    if expected_family is None:
        viols.append(
            f"directory band {band!r} is not registered; known bands are "
            f"{sorted(FAMILY_BANDS)}"
        )
    elif family != expected_family:
        viols.append(
            f"family {family!r} does not match band {band}; expected "
            f"{expected_family!r}"
        )

    fixture_name = data.get("name", "")
    if fixture_name != name:
        viols.append(
            f"name field {fixture_name!r} does not match directory {name!r}"
        )

    is_endpoint = family in ENDPOINT_FAMILIES
    if is_endpoint and not (ENDPOINT_OR_STEPS & set(data.keys())):
        viols.append(
            f"missing both `endpoint` and `steps` -- fixture must declare one"
        )

    file_refs = find_multipart_file_refs(data)
    for ref in file_refs:
        target = fdir / ref
        if not target.exists():
            viols.append(f"referenced artifact not found: {ref}")
            continue

    for art in data.get("input_artifacts", []) or []:
        path = art.get("path")
        if not path:
            viols.append("input_artifacts entry missing `path`")
            continue
        target = fdir / path
        if not target.exists():
            viols.append(f"input_artifacts: {path} not found on disk")
            continue
        kind = art.get("kind", "")
        if kind == "wav":
            v = validate_wav_header(
                target,
                expected_sr=art.get("sample_rate"),
                expected_channels=art.get("channels"),
                expected_duration_ms=art.get("duration_ms"),
            )
            viols.extend(v)

    if not is_endpoint:
        return FixtureResult(fdir, name, family, ok=not viols, skipped=True,
                             skip_reason="declarative fixture (no endpoint)",
                             violations=viols)

    skip_no_model = bool(data.get("skip_when_no_model", False))

    return FixtureResult(fdir, name, family, ok=not viols, skipped=False,
                         skip_reason="skip_when_no_model" if skip_no_model else "",
                         violations=viols)

def discover_fixtures(root: Path) -> list[Path]:
    out: list[Path] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        if not BAND_RE.match(entry.name):
            continue
        if (entry / "fixture.json").is_file():
            out.append(entry)
    return out

def cmd_validate(target: Path | None) -> int:
    root = fixtures_root()
    if not root.is_dir():
        return fail(f"not a directory: {root}", 2)

    if target is not None:
        fixtures = [target] if (target / "fixture.json").is_file() else []
        if not fixtures:
            return fail(f"no fixture.json under {target}", 2)
    else:
        fixtures = discover_fixtures(root)

    if not fixtures:
        return fail("no manifest fixtures discovered (looked for "
                    "NNN-* directories with fixture.json under "
                    f"{root})", 2)

    by_family: dict[str, list[FixtureResult]] = {}
    failed = 0
    for fdir in fixtures:
        res = validate_fixture(fdir)
        by_family.setdefault(res.family, []).append(res)
        marker = (
            "[PASS]" if res.ok and not res.skipped
            else "[SKIP]" if res.skipped
            else "[FAIL]"
        )
        suffix = ""
        if res.skipped and res.skip_reason:
            suffix = f"  ({res.skip_reason})"
        print(f"{marker} {res.name}{suffix}")
        if res.violations:
            for v in res.violations:
                print(f"    - {v}")
            failed += 1

    print()
    for fam in sorted(by_family.keys()):
        oks = sum(1 for r in by_family[fam] if r.ok)
        total = len(by_family[fam])
        print(f"  {fam}: {oks}/{total} ok")
    if failed:
        return fail(f"{failed} fixture(s) failed structural validation")
    print(f"\n[OK] {len(fixtures)} fixtures structurally valid")
    return 0

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("fixture", nargs="?", type=Path,
                   help="Path to a single fixture directory. If omitted, every "
                        "NNN-* fixture.json under conformance/fixtures/ is "
                        "validated.")
    args = p.parse_args()

    return cmd_validate(args.fixture)

if __name__ == "__main__":
    sys.exit(main())
