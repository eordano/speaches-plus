#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

LANE="${AB_LANE:-wf30}"
PORT="${AB_PORT:-8379}"
BOOT_TIMEOUT="${AB_BOOT_TIMEOUT:-600}"
LOGDIR="${AB_LOGDIR:-$HOME/.cache/agent-logs/lmi8-serving-ab}"
NVK="rust/scripts/nvk.sh"
mkdir -p "$LOGDIR"

line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
util=$(echo "$line" | cut -d, -f1 | tr -d ' ')
mem=$(echo "$line" | cut -d, -f2 | tr -d ' ')
echo "GPU gate: util=${util}% mem=${mem}MiB"
if [ "$util" -gt 5 ] || [ "$mem" -gt 1024 ]; then
  echo "GPU-GATE-FAIL: contended" >&2
  exit 1
fi

run_arm() {
  local arm="$1"
  shift
  local log="$LOGDIR/$arm.log"
  echo "=== arm $arm ==="
  setsid env "$@" \
    NV_PROF_CHAT=1 UVICORN_PORT="$PORT" RUST_LOG=info \
    NVK_LANE="$LANE" NVK_PKG=speaches-plus NVK_JOBS=8 \
    "$NVK" run --release --bin speaches-plus >"$log" 2>&1 &
  local pg=$!
  trap 'kill -TERM -- "-'"$pg"'" 2>/dev/null || true' EXIT

  local pick_chat='import sys,json
ids=[m["id"] for m in json.load(sys.stdin)["data"]]
bad=("embed","tts","whisper","wespeaker","voice","speaker")
chat=[i for i in ids if not any(b in i.lower() for b in bad)]
print((chat or ids)[0])'
  local t=0 model_id=""
  until model_id=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
    python3 -c "$pick_chat" 2>/dev/null) && [ -n "$model_id" ]; do
    t=$((t + 5))
    if [ "$t" -ge "$BOOT_TIMEOUT" ] || ! kill -0 "$pg" 2>/dev/null; then
      echo "ARM-FAIL [$arm]: no boot in ${t}s" >&2
      tail -n 20 "$log" >&2
      kill -TERM -- "-$pg" 2>/dev/null || true
      trap - EXIT
      return 1
    fi
    sleep 5 &
    wait $!
  done

  ask() {
    curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c "import json;print(json.dumps({'model':'$model_id','temperature':0,'max_tokens':256,'messages':[{'role':'user','content':'$1'}]}))")" \
      >/dev/null
  }
  ask "Say OK." # prime: graph capture + allocator warm, discarded
  ask "Explain in detail how a four-stroke engine works, covering intake, compression, combustion and exhaust."
  ask "Describe the water cycle from evaporation to precipitation, step by step, in detail."

  kill -TERM -- "-$pg" 2>/dev/null || true
  trap - EXIT
  local i
  for i in 1 2 3 4 5 6; do
    kill -0 "$pg" 2>/dev/null || break
    sleep 5 &
    wait $!
  done
  kill -KILL -- "-$pg" 2>/dev/null || true

  echo "--- $arm summaries:"
  grep "GRAPHED SUMMARY" "$log" | tail -n 2 | tee -a "$LOGDIR/RESULTS.txt"
}

: >"$LOGDIR/RESULTS.txt"
echo "# lmi8-serving-ab $(date -Is) HEAD=$(git -C ../.. rev-parse --short HEAD 2>/dev/null || echo '?')" >>"$LOGDIR/RESULTS.txt"
run_arm on-1 NV_VERIFY_LMHEAD_INT8=1
run_arm off-1
run_arm on-2 NV_VERIFY_LMHEAD_INT8=1
run_arm off-2

echo
echo "=== RESULTS ($LOGDIR/RESULTS.txt) ==="
cat "$LOGDIR/RESULTS.txt"
