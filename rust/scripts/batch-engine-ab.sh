#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

LANE="${BA_LANE:-wf30}"
PORT="${BA_PORT:-8381}"
BOOT_TIMEOUT="${BA_BOOT_TIMEOUT:-600}"
CONCS="${BA_CONCS:-1 4 8 16}"
MAXTOK="${BA_MAXTOK:-128}"
LOGDIR="${BA_LOGDIR:-$HOME/.cache/agent-logs/batch-engine-ab}"
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
  echo "=== arm $arm ($*) ==="
  setsid env "$@" \
    UVICORN_PORT="$PORT" RUST_LOG=info \
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

  BA_PORT="$PORT" BA_MODEL="$model_id" BA_MAXTOK="$MAXTOK" BA_ARM="$arm" \
    python3 rust/scripts/batch_engine_load.py $CONCS | tee -a "$LOGDIR/RESULTS.txt"

  grep -c "continuous-batching engine active" "$log" >/dev/null 2>&1 &&
    echo "[$arm] batch engine ACTIVE per boot log" | tee -a "$LOGDIR/RESULTS.txt" ||
    echo "[$arm] batch engine NOT active per boot log" | tee -a "$LOGDIR/RESULTS.txt"

  kill -TERM -- "-$pg" 2>/dev/null || true
  trap - EXIT
  local i
  for i in 1 2 3 4 5 6; do
    kill -0 "$pg" 2>/dev/null || break
    sleep 5 &
    wait $!
  done
  kill -KILL -- "-$pg" 2>/dev/null || true
}

: >"$LOGDIR/RESULTS.txt"
{
  echo "# batch-engine-ab $(date -Is) HEAD=$(git -C ../.. rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "# concurrencies: $CONCS   max_tokens=$MAXTOK   both arms NV_NO_SPEC=1"
} >>"$LOGDIR/RESULTS.txt"

run_arm nospec-perreq NV_NO_SPEC=1
run_arm nospec-batched NV_NO_SPEC=1 NV_BATCH_ENGINE=1

if [ "${BA_GRAPH_ARM:-0}" = "1" ]; then
  run_arm nospec-batched-graphed NV_NO_SPEC=1 NV_BATCH_ENGINE=1 NV_KV_RING=0 NV_BATCH_GRAPH=1
fi

if [ "${BA_RING_ARM:-0}" = "1" ]; then
  run_arm nospec-batched-noring NV_NO_SPEC=1 NV_BATCH_ENGINE=1 NV_KV_RING=0
fi

if [ "${BA_RINGFP8_ARM:-0}" = "1" ]; then
  run_arm nospec-batched-ringfp8 NV_NO_SPEC=1 NV_BATCH_ENGINE=1 NV_PAGED_ATTN_FP8_RING=1
fi

echo
echo "=== RESULTS ($LOGDIR/RESULTS.txt) ==="
cat "$LOGDIR/RESULTS.txt"
