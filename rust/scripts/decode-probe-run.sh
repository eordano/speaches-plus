#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

LANE="${DP_LANE:-wf6ser}"
PORT="${DP_PORT:-8393}"
BOOT_TIMEOUT="${DP_BOOT_TIMEOUT:-900}"
CONCS="${DP_CONCS:-1 4 16}"
MAXTOK="${DP_MAXTOK:-128}"
LOGDIR="${DP_LOGDIR:-$HOME/.cache/agent-logs/serprobe}"
NVK="${DP_NVK:-rust/scripts/nvk.sh}"
mkdir -p "$LOGDIR"

line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
util=$(echo "$line" | cut -d, -f1 | tr -d ' ')
mem=$(echo "$line" | cut -d, -f2 | tr -d ' ')
echo "GPU gate: util=${util}% mem=${mem}MiB"
if [ "$util" -gt 5 ] || [ "$mem" -gt 1024 ]; then
  echo "GPU-GATE-FAIL: contended" >&2
  exit 1
fi

SRV="$LOGDIR/server.log"
DMON="$LOGDIR/dmon.log"
: >"$SRV"
: >"$DMON"

setsid env \
  NV_NO_SPEC=1 \
  NV_PROF_DECODE_PHASES=1 \
  NV_PROF_DECODE_EVERY="${DP_EVERY:-256}" \
  NV_DEBUG_GRAPH_MEM=1 \
  NV_PROF_CHAT=1 \
  UVICORN_PORT="$PORT" RUST_LOG=info \
  NVKC_REPO="${DP_REPO:-$PWD}" \
  NVK_LANE="$LANE" NVK_PKG=speaches-plus NVK_FEATURES=cuda NVK_JOBS=8 \
  bash "$NVK" run --release --bin speaches-plus >"$SRV" 2>&1 &
PG=$!
cleanup() {
  kill -TERM -- "-$PG" 2>/dev/null || true
  kill "$DMON_PID" 2>/dev/null || true
}
trap cleanup EXIT

pick_chat='import sys,json
ids=[m["id"] for m in json.load(sys.stdin)["data"]]
bad=("embed","tts","whisper","wespeaker","voice","speaker")
chat=[i for i in ids if not any(b in i.lower() for b in bad)]
print((chat or ids)[0])'
t=0
MODEL=""
until MODEL=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
  python3 -c "$pick_chat" 2>/dev/null) && [ -n "$MODEL" ]; do
  t=$((t + 5))
  if [ "$t" -ge "$BOOT_TIMEOUT" ] || ! kill -0 "$PG" 2>/dev/null; then
    echo "BOOT-FAIL after ${t}s" >&2
    exit 1
  fi
  sleep 5 &
  wait $!
done
echo "booted in ${t}s model=$MODEL"

nvidia-smi dmon -s pucm -o T -d 1 >"$DMON" 2>&1 &
DMON_PID=$!

MARKS="$LOGDIR/marks.log"
: >"$MARKS"
for n in $CONCS; do
  echo "MARK conc=$n start $(date +%H:%M:%S)" >>"$MARKS"
  BA_PORT="$PORT" BA_MODEL="$MODEL" BA_MAXTOK="$MAXTOK" BA_ARM="probe-c$n" \
    python3 rust/scripts/batch_engine_load.py "$n" | tee -a "$LOGDIR/RESULTS.txt"
  echo "MARK conc=$n end   $(date +%H:%M:%S)" >>"$MARKS"
done

kill "$DMON_PID" 2>/dev/null || true
kill -TERM -- "-$PG" 2>/dev/null || true
trap - EXIT
for i in 1 2 3 4 5 6 7 8; do
  kill -0 "$PG" 2>/dev/null || break
  sleep 5 &
  wait $!
done
kill -KILL -- "-$PG" 2>/dev/null || true
sleep 5 &
wait $!
echo "GPU after teardown: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
