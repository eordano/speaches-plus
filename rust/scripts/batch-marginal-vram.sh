#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

LANE="${MV_LANE:-admit}"
PORT="${MV_PORT:-8392}"
BOOT_TIMEOUT="${MV_BOOT_TIMEOUT:-600}"
CONCS="${MV_CONCS:-1 2 4 8 16}"
MAXTOK="${MV_MAXTOK:-192}"
SEQS="${MV_SEQS:-16}"
LOGDIR="${MV_LOGDIR:-$HOME/.cache/agent-logs/batch-marginal-vram}"
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

LOG="$LOGDIR/server.log"
setsid env NV_NO_SPEC=1 NV_BATCH_ENGINE=1 NV_BATCH_MAX_SEQS="$SEQS" \
  NV_ADMIT_DISABLE=1 NV_CHAT_CONCURRENCY=64 \
  UVICORN_PORT="$PORT" RUST_LOG=info \
  NVK_LANE="$LANE" NVK_PKG=speaches-plus NVK_JOBS=8 \
  "$NVK" run --release --bin speaches-plus >"$LOG" 2>&1 &
PG=$!
trap 'kill -TERM -- "-$PG" 2>/dev/null || true' EXIT

pick_chat='import sys,json
ids=[m["id"] for m in json.load(sys.stdin)["data"]]
bad=("embed","tts","whisper","wespeaker","voice","speaker")
chat=[i for i in ids if not any(b in i.lower() for b in bad)]
print((chat or ids)[0])'
t=0 model_id=""
until model_id=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
  python3 -c "$pick_chat" 2>/dev/null) && [ -n "$model_id" ]; do
  t=$((t + 5))
  if [ "$t" -ge "$BOOT_TIMEOUT" ] || ! kill -0 "$PG" 2>/dev/null; then
    echo "no boot in ${t}s" >&2
    tail -n 20 "$LOG" >&2
    exit 1
  fi
  sleep 5 &
  wait $!
done
echo "server up: $model_id"

SPID=$(pgrep -g "$PG" -f 'speaches-plus$' | head -n 1 || true)
[ -n "$SPID" ] || SPID=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -n 1)
echo "server pid: $SPID"

vram_now() {
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits |
    awk -F', *' -v p="$SPID" '$1==p {print $2}' | head -n 1
}

BA_PORT="$PORT" BA_MODEL="$model_id" BA_MAXTOK=32 BA_ARM=warm \
  python3 rust/scripts/batch_engine_load.py 1 >/dev/null
idle_mib=$(vram_now)
echo "idle_after_warm_mib=$idle_mib" | tee "$LOGDIR/RESULTS.txt"

for n in $CONCS; do
  : >"$LOGDIR/samples-$n.txt"
  (while :; do
    vram_now >>"$LOGDIR/samples-$n.txt"
    sleep 0.2 &
    wait $!
  done) &
  SAMPLER=$!
  BA_PORT="$PORT" BA_MODEL="$model_id" BA_MAXTOK="$MAXTOK" BA_ARM="conc$n" \
    python3 rust/scripts/batch_engine_load.py "$n" >"$LOGDIR/load-$n.txt" 2>&1 || true
  kill "$SAMPLER" 2>/dev/null || true
  wait "$SAMPLER" 2>/dev/null || true
  peak=$(sort -n "$LOGDIR/samples-$n.txt" | tail -n 1)
  med=$(sort -n "$LOGDIR/samples-$n.txt" | awk '{a[NR]=$1} END{print a[int(NR/2)+1]}')
  echo "conc=$n peak_mib=$peak median_mib=$med delta_vs_idle_mib=$((peak - idle_mib))" |
    tee -a "$LOGDIR/RESULTS.txt"
  grep -hE "^\[conc$n\]" "$LOGDIR/load-$n.txt" | tee -a "$LOGDIR/RESULTS.txt" || true
done

kill -TERM -- "-$PG" 2>/dev/null || true
trap - EXIT
for i in 1 2 3 4 5 6; do
  kill -0 "$PG" 2>/dev/null || break
  sleep 5 &
  wait $!
done
kill -KILL -- "-$PG" 2>/dev/null || true

echo
echo "=== RESULTS ($LOGDIR/RESULTS.txt) ==="
cat "$LOGDIR/RESULTS.txt"
