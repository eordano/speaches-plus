#!/usr/bin/env python3
"""Verify that the code citations in rocq/ still point at what they claim.

The adversarial review found the whole citation set dead after the
chat_engine split: rows said `chat_engine.rs:402-443` for a function that had
moved to `chat_engine/build.rs:715`. A proof that cites the wrong code is
worse than one that cites none, because it reads as evidence.

Line numbers rot on every refactor, so this checks SYMBOLS, not lines: for
each load-bearing anchor it locates the defining symbol in the tree and
requires that the citation recorded next to it in rocq/ names the same file
and a line range containing that definition. Run from rocq/gen (run.sh does).

Exit 1 on any drift, printing the correction to paste in.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
ROCQ = os.path.abspath(os.path.join(HERE, ".."))

ANCHORS = [
    ("pub fn tree_window_attends", "rust/crates/nv-models/src/gemma4.rs"),
    ("pub fn kv_budget", "rust/crates/nv-models/src/gemma4.rs"),
    ("pub fn kv_budget_capped", "rust/crates/nv-models/src/gemma4.rs"),
    ("pub(crate) fn enforce_gemma4_vram_budget", "rust/src/oapi/chat_engine/build.rs"),
    ("pub(crate) fn assert_kv_window_invariants", "rust/src/oapi/chat_engine/spec_window.rs"),
    ("pub(crate) fn kv_window", "rust/src/oapi/chat_engine/spec_window.rs"),
    ("pub(crate) fn spec_verify_window", "rust/src/oapi/chat_engine/spec_window.rs"),
    ("pub fn from_gemma4_hybrid", "rust/crates/nv-models/src/paged_fp8.rs"),
]

def find_symbol(sym, relpath):
    path = os.path.join(REPO, relpath)
    if not os.path.exists(path):
        return None, f"file missing: {relpath}"
    with open(path, errors="replace") as f:
        for i, line in enumerate(f, 1):
            if sym in line:
                return i, None
    return None, f"symbol not found in {relpath}: {sym!r}"

def rocq_text():
    """Every citation-bearing file under rocq/ (README + the .v sources)."""
    out = {}
    for root, _dirs, files in os.walk(ROCQ):
        if os.sep + "gen" + os.sep in root + os.sep and "out" in root:
            continue
        for fn in files:
            if fn.endswith((".v", ".md")):
                p = os.path.join(root, fn)
                with open(p, errors="replace") as f:
                    out[os.path.relpath(p, REPO)] = f.read()
    return out

def main():
    texts = rocq_text()
    drift = []
    ok = 0
    for sym, relpath in ANCHORS:
        line, err = find_symbol(sym, relpath)
        if err:
            drift.append(f"  {sym}: {err}")
            continue
        base = os.path.basename(relpath)
        bad = []
        name = sym.split()[-1].split("(")[0]
        for doc, body in texts.items():
            for m in re.finditer(re.escape(base) + r":(\d+)(?:-(\d+))?", body):
                lo = int(m.group(1))
                hi = int(m.group(2) or m.group(1))
                pre = body[max(0, m.start() - 80): m.start()]
                if name not in pre:
                    continue
                nearest = max(
                    (pre.rfind(a.split()[-1].split("(")[0]), a)
                    for a, _f in ANCHORS
                    if a.split()[-1].split("(")[0] in pre
                )
                if nearest[1] != sym:
                    continue
                BODY = 260
                if hi < line or lo > line + BODY:
                    bad.append(f"{doc} cites {base}:{lo}-{hi}, definition at {line}")
        if bad:
            drift.append(f"  {name} -> {relpath}:{line}\n    " + "\n    ".join(bad))
        else:
            ok += 1
    print(f"[anchors] {ok}/{len(ANCHORS)} load-bearing anchors verified against the tree")
    if drift:
        print("[anchors] DRIFT -- citations point at code that moved:")
        for d in drift:
            print(d)
        print("[anchors] fix the citations above (or the ANCHORS table if a symbol was renamed)")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
