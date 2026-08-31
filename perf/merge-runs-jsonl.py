#!/usr/bin/env python3
import json
import sys
from pathlib import Path

ENABLE_ONCE_PER_CLONE_BECAUSE_GIT_CONFIG_DOES_NOT_TRAVEL = (
    "git config merge.union-jsonl.driver "
    f"'python3 {Path(__file__).resolve()} %O %A %B'"
)

GITATTRIBUTES_LINE_ALREADY_COMMITTED_NEXT_TO_THIS_SCRIPT = "perf/runs.jsonl merge=union-jsonl"

CONFLICT_MARKER_PREFIXES = ("<<<<<<<", "=======", ">>>>>>>", "|||||||")

USAGE = f"""merge-runs-jsonl.py -- union merge for append-only JSONL row stores

  merge-runs-jsonl.py %O %A %B    git merge driver: union of both sides into %A
  merge-runs-jsonl.py FILE        repair a conflict-marker-contaminated store in place

Concurrent sessions appending rows produce textual conflicts in an append-only
store whose true resolution is always the union of both sides. Enable per clone
(.gitattributes already names the driver for perf/runs.jsonl; git runs the
driver from the repo root, so the command below carries this file's resolved
path -- re-print it from each checkout rather than copying it between clones):

  {ENABLE_ONCE_PER_CLONE_BECAUSE_GIT_CONFIG_DOES_NOT_TRAVEL}
"""

def fail(msg):
    print(f"merge-runs-jsonl: REFUSING: {msg}", file=sys.stderr)
    sys.exit(1)

def canonical_rows(text, source, tolerate_markers):
    rows = []
    markers = 0
    for n, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(CONFLICT_MARKER_PREFIXES):
            if tolerate_markers:
                markers += 1
                continue
            fail(f"{source}:{n} is a conflict marker; a driver input must be one clean side "
                 f"-- for an already-conflicted working-tree file run the 1-arg repair mode")
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            fail(f"{source}:{n} is neither a JSON row nor a conflict marker; refusing to "
                 f"guess -- fix or delete that line by hand first: {stripped[:120]!r}")
        if not isinstance(obj, dict) or "run_id" not in obj:
            fail(f"{source}:{n} parses but is not a run row (no run_id); this store holds "
                 f"only parse_and_append.py rows")
        rows.append(json.dumps(obj, sort_keys=True))
    return rows, markers

def dedupe_keeping_first(lines):
    seen = set()
    out = []
    dupes = 0
    for ln in lines:
        if ln in seen:
            dupes += 1
            continue
        seen.add(ln)
        out.append(ln)
    return out, dupes

def warn_on_diverged_run_ids(lines):
    by_id = {}
    for ln in lines:
        rid = json.loads(ln)["run_id"]
        by_id.setdefault(rid, []).append(ln)
    for rid, variants in by_id.items():
        if len(variants) > 1:
            print(f"merge-runs-jsonl: WARNING: run_id {rid} appears {len(variants)}x with "
                  f"differing content; keeping all -- adjudicate by hand", file=sys.stderr)

def write_lines(path, lines):
    Path(path).write_text("".join(ln + "\n" for ln in lines))

def merge_driver(base_path, ours_path, theirs_path):
    canonical_rows(Path(base_path).read_text(), base_path, tolerate_markers=False)
    ours, _ = canonical_rows(Path(ours_path).read_text(), ours_path, tolerate_markers=False)
    theirs, _ = canonical_rows(Path(theirs_path).read_text(), theirs_path, tolerate_markers=False)
    ours_set = set(ours)
    added = [ln for ln in dedupe_keeping_first(theirs)[0] if ln not in ours_set]
    merged, dupes = dedupe_keeping_first(ours + added)
    warn_on_diverged_run_ids(merged)
    write_lines(ours_path, merged)
    print(f"merge-runs-jsonl: union merged {len(merged)} rows "
          f"({len(ours)} ours + {len(added)} theirs-only, {dupes} exact duplicates dropped)")

def repair_in_place(path):
    text = Path(path).read_text()
    rows, markers = canonical_rows(text, path, tolerate_markers=True)
    merged, dupes = dedupe_keeping_first(rows)
    warn_on_diverged_run_ids(merged)
    write_lines(path, merged)
    print(f"merge-runs-jsonl: repaired {path}: {len(merged)} rows kept, "
          f"{markers} conflict-marker lines stripped, {dupes} exact duplicates dropped")

def main(argv):
    if len(argv) == 4:
        merge_driver(argv[1], argv[2], argv[3])
    elif len(argv) == 2:
        repair_in_place(argv[1])
    else:
        print(USAGE, end="", file=sys.stderr)
        sys.exit(2)

if __name__ == "__main__":
    main(sys.argv)
