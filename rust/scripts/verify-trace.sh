#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/.."
cd ..

LANE="${VT_LANE:-wf30}"
PORT="${VT_PORT:-8378}"
BOOT_TIMEOUT="${VT_BOOT_TIMEOUT:-600}"
LOGDIR="${VT_LOGDIR:-$HOME/.cache/agent-logs/verify-trace}"
SHIM="${VT_SHIM:-$HOME/tmp/roofline/libnvtrace.so}"
WIDTHS="${VT_WIDTHS:-5}"
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

[ -f "$SHIM" ] || {
  echo "missing shim $SHIM (sh rust/scripts/profiling/build-nvtrace.sh)" >&2
  exit 1
}

TRACE="$LOGDIR/verify-trace.tsv"
rm -f "$TRACE"

setsid env \
  CUDA_INJECTION64_PATH="$SHIM" NVTRACE_OUT="$TRACE" \
  UVICORN_PORT="$PORT" RUST_LOG=info \
  NVK_LANE="$LANE" NVK_PKG=speaches-plus NVK_JOBS=8 \
  "$NVK" run --release --bin speaches-plus >"$LOGDIR/server.log" 2>&1 &
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
    echo "server did not come up in ${t}s; tail:" >&2
    tail -n 30 "$LOGDIR/server.log" >&2
    exit 1
  fi
  sleep 5 &
  wait $!
done
echo "server up, model: $model_id"

ask() {
  local prompt="$1" toks="$2" out="$3"
  curl -fsS "http://127.0.0.1:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$(python3 -c "import json;print(json.dumps({'model':'$model_id','temperature':0,'max_tokens':$toks,'messages':[{'role':'user','content':'$prompt'}]}))")" \
    >"$LOGDIR/$out"
}

ask "Say OK." 8 warm.json
echo "warmup done"
ask "Explain in detail how a four-stroke engine works, covering intake, compression, combustion and exhaust." 256 long.json
echo "long request done"
grep -E "draft-chain CUDA graph allocated" "$LOGDIR/server.log" ||
  echo "WARNING: no draft-chain graph line; spec path may be off"

kill -TERM -- "-$PG" 2>/dev/null || true
trap - EXIT
for i in 1 2 3 4 5 6 7 8; do
  [ -s "$TRACE" ] && break
  kill -0 "$PG" 2>/dev/null || break
  sleep 5 &
  wait $!
done
kill -KILL -- "-$PG" 2>/dev/null || true

[ -s "$TRACE" ] || {
  echo "TRACE MISSING: shim did not flush" >&2
  exit 1
}
wc -l "$TRACE"
python3 rust/scripts/profiling/analyze.py "$TRACE" graphs | tee "$LOGDIR/analysis.txt"
python3 rust/scripts/profiling/analyze.py "$TRACE" "$WIDTHS" 262144 >>"$LOGDIR/analysis.txt" 2>&1 || true
