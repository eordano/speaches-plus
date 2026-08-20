#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
LOGDIR="${BT_LOGDIR:-$HOME/.cache/agent-logs/batch-attr}"

for arm in eager-traced graph-traced; do
  t="$LOGDIR/$arm.tsv"
  [ -s "$t" ] || {
    echo "== $arm: no trace"
    continue
  }
  {
    echo "############ $arm ############"
    echo "--- captured graph shapes from the server log ---"
    grep -a "cuda-graph.*captured shape_token" "$LOGDIR/$arm.log" 2>/dev/null |
      sort | uniq -c || echo "(none: eager batched decode)"
    grep -a "CUDA-graph family active" "$LOGDIR/$arm.log" 2>/dev/null || true
    echo
    echo "--- analyze.py graphs (per-replay) ---"
    python3 rust/scripts/profiling/analyze.py "$t" graphs || true
    echo
    echo "--- analyze_batch.py (per batched decode step) ---"
    python3 rust/scripts/profiling/analyze_batch.py "$t" 262144 16 || true
  } >"$LOGDIR/$arm.analysis.txt" 2>&1
  echo "== wrote $LOGDIR/$arm.analysis.txt"
done

python3 - "$LOGDIR" <<'PY'
import glob, re, sys, collections
for p in sorted(glob.glob(sys.argv[1] + "/*-traced.log")):
    toks = collections.Counter(
        int(m) for m in re.findall(r"captured shape_token=(\d+)",
                                   open(p, errors="ignore").read()))
    if not toks:
        continue
    print(f"{p}:")
    for t, n in toks.most_common():
        print(f"  shape_token={t}  b_bucket={t >> 32}  ctx_bucket={t & 0xffffffff}  captures={n}")
PY
