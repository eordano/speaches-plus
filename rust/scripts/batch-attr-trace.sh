#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

LANE="${BT_LANE:-wf30}"
PORT="${BT_PORT:-8386}"
BOOT_TIMEOUT="${BT_BOOT_TIMEOUT:-600}"
CONC="${BT_CONC:-16}"
MAXTOK="${BT_MAXTOK:-64}"
LOGDIR="${BT_LOGDIR:-$HOME/.cache/agent-logs/batch-attr}"
SHIM="${BT_SHIM:-$HOME/tmp/roofline/libnvtrace.so}"
ARMS="${BT_ARMS:-eager-plain graph-plain eager-traced graph-traced}"
NVK="${BT_NVK:-rust/scripts/nvk.sh}"
mkdir -p "$LOGDIR"

line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
util=$(echo "$line" | cut -d, -f1 | tr -d ' ')
mem=$(echo "$line" | cut -d, -f2 | tr -d ' ')
echo "GPU gate: util=${util}% mem=${mem}MiB"
if [ "${BT_SKIP_GATE:-0}" != "1" ] && { [ "$util" -gt 5 ] || [ "$mem" -gt 1024 ]; }; then
  echo "GPU-GATE-FAIL: contended" >&2
  exit 1
fi
[ -f "$SHIM" ] || {
  echo "missing shim $SHIM" >&2
  exit 1
}

GRAPH_ENV=(NV_KV_RING=0 NV_BATCH_GRAPH=1
  "NV_BATCH_GRAPH_SIZES=${BT_GRAPH_SIZES:-1,2,4,8,16}")
BASE_ENV=(NV_NO_SPEC=1 NV_BATCH_ENGINE=1 NV_BATCH_MAX_SEQS=16)

run_arm() {
  local arm="$1" traced="$2"
  shift 2
  local log="$LOGDIR/$arm.log"
  local trace="$LOGDIR/$arm.tsv"
  local -a shim_env=()
  if [ "$traced" = "1" ]; then
    rm -f "$trace"
    shim_env=(CUDA_INJECTION64_PATH="$SHIM" NVTRACE_OUT="$trace")
  fi
  echo "=== arm $arm (traced=$traced) env: $* ==="
  setsid env "${shim_env[@]}" "$@" \
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
    python3 rust/scripts/batch_engine_load.py $CONC | tee -a "$LOGDIR/RESULTS.txt"

  grep -c "continuous-batching engine active" "$log" >/dev/null 2>&1 &&
    echo "[$arm] batch engine ACTIVE" | tee -a "$LOGDIR/RESULTS.txt" ||
    echo "[$arm] batch engine NOT ACTIVE" | tee -a "$LOGDIR/RESULTS.txt"
  grep -c "CUDA-graph family active" "$log" >/dev/null 2>&1 &&
    echo "[$arm] batch CUDA-graph family ACTIVE" | tee -a "$LOGDIR/RESULTS.txt" ||
    echo "[$arm] batch CUDA-graph family absent (eager batched decode)" |
    tee -a "$LOGDIR/RESULTS.txt"

  kill -TERM -- "-$pg" 2>/dev/null || true
  trap - EXIT
  local i
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    if [ "$traced" = "1" ]; then [ -s "$trace" ] && break; fi
    kill -0 "$pg" 2>/dev/null || break
    sleep 5 &
    wait $!
  done
  kill -KILL -- "-$pg" 2>/dev/null || true
  sleep 3 &
  wait $!
  echo "[$arm] post-teardown GPU: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  [ "$traced" = "1" ] && wc -l "$trace"
  return 0
}

: >"$LOGDIR/RESULTS.txt"
{
  echo "# batch-attr-trace $(date -Is) HEAD=$(git rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "# concurrency=$CONC max_tokens=$MAXTOK arms='$ARMS'"
} >>"$LOGDIR/RESULTS.txt"

for arm in $ARMS; do
  case "$arm" in
    eager-plain) run_arm "$arm" 0 "${BASE_ENV[@]}" ;;
    graph-plain) run_arm "$arm" 0 "${BASE_ENV[@]}" "${GRAPH_ENV[@]}" ;;
    eager-traced) run_arm "$arm" 1 "${BASE_ENV[@]}" ;;
    graph-traced) run_arm "$arm" 1 "${BASE_ENV[@]}" "${GRAPH_ENV[@]}" ;;
    *)
      echo "unknown arm $arm" >&2
      exit 1
      ;;
  esac
done

echo
echo "=== RESULTS ($LOGDIR/RESULTS.txt) ==="
cat "$LOGDIR/RESULTS.txt"
