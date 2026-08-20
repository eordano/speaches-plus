#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

LANE="${KV30_LANE:-wf30}"
PORT="${KV30_PORT:-8377}"
KV_MAX="${KV30_KV_MAX:-32768}"
WINDOW="${KV30_WINDOW:-2048}"
MAX_UTIL="${KV30_MAX_UTIL:-5}"
MAX_MEM_MIB="${KV30_MAX_MEM_MIB:-1024}"
BOOT_TIMEOUT="${KV30_BOOT_TIMEOUT:-600}"
LOGDIR="${KV30_LOGDIR:-$HOME/.cache/agent-logs/kv30-ab}"
OUTDIR="${KV30_OUTDIR:-docs/measurements/$(date +%F)-kv30-drafter-cap-ab}"
NVK="rust/scripts/nvk.sh"

mkdir -p "$LOGDIR" "$OUTDIR"

gate() {
  local line util mem
  line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
  util=$(echo "$line" | cut -d, -f1 | tr -d ' ')
  mem=$(echo "$line" | cut -d, -f2 | tr -d ' ')
  echo "GPU gate reading: util=${util}% mem_used=${mem}MiB (limits: ${MAX_UTIL}% / ${MAX_MEM_MIB}MiB)"
  if [ "$util" -gt "$MAX_UTIL" ] || [ "$mem" -gt "$MAX_MEM_MIB" ]; then
    echo "GPU-GATE-FAIL: contended; record this run as deferred-contended" >&2
    return 1
  fi
  echo "GPU-GATE-CLEARED"
}

build() {
  echo "building speaches-plus (lane $LANE) ..."
  NVK_LANE="$LANE" NVK_PKG=speaches-plus NVK_JOBS=8 \
    "$NVK" build --release >"$LOGDIR/build.log" 2>&1
  echo "build ok"
}

run_arm() {
  local arm="$1"
  shift
  local log="$LOGDIR/$arm.log"
  echo "=== arm $arm ==="
  gate

  setsid env "$@" \
    UVICORN_PORT="$PORT" \
    NV_KV_MAX_SEQ_LEN="$KV_MAX" \
    RUST_LOG=info \
    NVK_LANE="$LANE" NVK_PKG=speaches-plus NVK_JOBS=8 \
    "$NVK" run --release --bin speaches-plus >"$log" 2>&1 &
  SERVER_PG=$!
  local pg=$SERVER_PG
  trap 'kill -TERM -- "-${SERVER_PG:-0}" 2>/dev/null || true' EXIT

  local pick_chat='import sys,json
ids=[m["id"] for m in json.load(sys.stdin)["data"]]
bad=("embed","tts","whisper","wespeaker","voice","speaker")
chat=[i for i in ids if not any(b in i.lower() for b in bad)]
print((chat or ids)[0])'
  local t=0 model_id=""
  until model_id=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
    python3 -c "$pick_chat" 2>/dev/null) &&
    [ -n "$model_id" ]; do
    t=$((t + 5))
    if [ "$t" -ge "$BOOT_TIMEOUT" ] || ! kill -0 "$pg" 2>/dev/null; then
      echo "ARM-FAIL [$arm]: server did not come up in ${t}s; last log lines:" >&2
      tail -n 30 "$log" >&2
      kill -TERM -- "-$pg" 2>/dev/null || true
      trap - EXIT
      return 1
    fi
    sleep 5 &
    wait $!
  done
  echo "server up, model id: $model_id"

  curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json;print(json.dumps({'model':'$model_id','temperature':0,'max_tokens':64,'messages':[{'role':'user','content':'Count from 1 to 20, one number per line.'}]}))")" \
    >"$LOGDIR/$arm.response.json"
  echo "request done"

  local pids proc_mem=""
  pids=$(ps -o pid= -g "$pg" | tr -d ' ')
  while read -r pid mem; do
    for p in $pids; do
      [ "$pid" = "$p" ] && proc_mem="$mem"
    done
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | tr -d ' ' | tr ',' ' ')
  echo "$arm per-process used_memory: ${proc_mem:-unknown} MiB"

  kill -TERM -- "-$pg" 2>/dev/null || true
  trap - EXIT
  local i
  for i in 1 2 3 4 5 6; do
    kill -0 "$pg" 2>/dev/null || break
    sleep 5 &
    wait $!
  done
  kill -KILL -- "-$pg" 2>/dev/null || true

  {
    echo "== arm $arm =="
    echo "model_id: $model_id"
    echo "per_process_used_memory_mib: ${proc_mem:-unknown}"
    grep -E "loading .* from" "$log" | sed -n '1p' || true
    grep -E "gemma4 VRAM budget at kv_max" "$log" || echo "MISSING: VRAM budget line"
    grep -E "draft-chain CUDA graph allocated" "$log" || echo "MISSING: draft-chain graph line (spec path not taken?)"
    grep -E "drafter" "$LOGDIR/$arm.response.json" >/dev/null 2>&1 || true
  } | tee -a "$OUTDIR/RESULTS.txt"
}

: >"$OUTDIR/RESULTS.txt"
{
  echo "# kv30-ab $(date -Is)  HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "# kv_max=$KV_MAX window(arm B)=$WINDOW port=$PORT lane=$LANE"
  uptime
} >>"$OUTDIR/RESULTS.txt"

build
run_arm A
run_arm B NV_DRAFTER_KV_WINDOW="$WINDOW"

echo
echo "=== summary ($OUTDIR/RESULTS.txt) ==="
cat "$OUTDIR/RESULTS.txt"
