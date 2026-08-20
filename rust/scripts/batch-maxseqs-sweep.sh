#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

LANE="${MS_LANE:-wf30}"
PORT="${MS_PORT:-8382}"
BOOT_TIMEOUT="${MS_BOOT_TIMEOUT:-600}"
LOAD_TIMEOUT="${MS_LOAD_TIMEOUT:-420}"
CONCS="${MS_CONCS:-16 32}"
MAXTOK="${MS_MAXTOK:-128}"
SEQS="${MS_SEQS:-8 16 24 32 48}"
EXTRA_ENV="${MS_EXTRA_ENV:-}"
TAG="${MS_TAG:-}"
LOGDIR="${MS_LOGDIR:-$HOME/.cache/agent-logs/batch-maxseqs-sweep}"
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

RESULTS="$LOGDIR/RESULTS${TAG:+-$TAG}.txt"
: >"$RESULTS"
{
  echo "# batch-maxseqs-sweep $(date -Is) HEAD=$(git -C ../.. rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "# NV_BATCH_MAX_SEQS arms: $SEQS   concurrencies: $CONCS   max_tokens=$MAXTOK"
  echo "# extra server env: ${EXTRA_ENV:-<none>}"
} >>"$RESULTS"

for seqs in $SEQS; do
  arm="seqs$seqs${TAG:+-$TAG}"
  log="$LOGDIR/$arm.log"
  echo "=== arm NV_BATCH_MAX_SEQS=$seqs ==="
  # shellcheck disable=SC2086
  setsid env NV_NO_SPEC=1 NV_BATCH_ENGINE=1 NV_BATCH_MAX_SEQS="$seqs" $EXTRA_ENV \
    UVICORN_PORT="$PORT" RUST_LOG=info \
    NVK_LANE="$LANE" NVK_PKG=speaches-plus NVK_JOBS=8 \
    "$NVK" run --release --bin speaches-plus >"$log" 2>&1 &
  pg=$!
  trap 'kill -TERM -- "-'"$pg"'" 2>/dev/null || true' EXIT

  pick_chat='import sys,json
ids=[m["id"] for m in json.load(sys.stdin)["data"]]
bad=("embed","tts","whisper","wespeaker","voice","speaker")
chat=[i for i in ids if not any(b in i.lower() for b in bad)]
print((chat or ids)[0])'
  t=0 model_id=""
  until model_id=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
    python3 -c "$pick_chat" 2>/dev/null) && [ -n "$model_id" ]; do
    t=$((t + 5))
    if [ "$t" -ge "$BOOT_TIMEOUT" ] || ! kill -0 "$pg" 2>/dev/null; then
      echo "ARM-FAIL [$arm]: no boot in ${t}s (KV pool may not fit at this cap)" |
        tee -a "$RESULTS" >&2
      grep -hoE 'sizing paged KV pool.*|memory allocation of .*|CUDA_ERROR[A-Z_]*|out of memory' "$log" |
        tail -n 3 | sed "s/^/# [$arm] /" >>"$RESULTS" || true
      kill -TERM -- "-$pg" 2>/dev/null || true
      trap - EXIT
      continue 2
    fi
    sleep 5 &
    wait $!
  done

  grep -hoE 'sizing paged KV pool from free VRAM.*' "$log" | sed "s/^/# [$arm] /" >>"$RESULTS" || true
  grep -hoE 'max_concurrent_upper_bound[^ ]*' "$log" | head -n 1 | sed "s/^/# [$arm] admission /" >>"$RESULTS" || true

  rc=0
  BA_PORT="$PORT" BA_MODEL="$model_id" BA_MAXTOK="$MAXTOK" BA_ARM="$arm" \
    timeout "$LOAD_TIMEOUT" python3 rust/scripts/batch_engine_load.py $CONCS |
    tee -a "$RESULTS" || rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "ARM-STALL [$arm]: load driver exited rc=$rc (timeout ${LOAD_TIMEOUT}s)" | tee -a "$RESULTS" >&2
  fi

  kill -TERM -- "-$pg" 2>/dev/null || true
  trap - EXIT
  for i in 1 2 3 4 5 6; do
    kill -0 "$pg" 2>/dev/null || break
    sleep 5 &
    wait $!
  done
  kill -KILL -- "-$pg" 2>/dev/null || true
  for i in 1 2 3 4 5 6 7 8; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$used" -le 1024 ] && break
    sleep 5 &
    wait $!
  done
done

echo
echo "=== RESULTS ($RESULTS) ==="
cat "$RESULTS"
