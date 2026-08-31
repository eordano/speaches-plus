#!/usr/bin/env bash
set -euo pipefail

ARM="${VL_ARM:?VL_ARM=<arm name>}"
OUT="${VL_OUT:?VL_OUT=<artifact dir>}"
PORT="${VL_PORT:-8399}"
BIN="${VL_BIN:-$HOME/.cache/cargo-tmp/tgt-ledger/release/speaches-plus}"
BOOT_TIMEOUT="${VL_BOOT_TIMEOUT:-1200}"
REQ="${VL_REQ:-chat}"
MODEL_FILTER="${VL_MODEL_FILTER:-}"
QUIET_MIB="${VL_QUIET_MIB:-500}"
LOCK="${VL_LOCK:-$HOME/tmp/wf/GPUQ_LOCK}"

mkdir -p "$OUT"
LOG="$OUT/$ARM.server.log"
SAMPLES="$OUT/$ARM.samples.txt"
RESULT="$OUT/$ARM.result.json"

mem_now() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' '; }

median_of() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }

exec 9>"$LOCK"
flock 9

t0=$SECONDS
until [ "$(mem_now)" -lt "$QUIET_MIB" ]; do
  if [ $((SECONDS - t0)) -gt 900 ]; then
    echo "IDLE-GATE-FAIL: card busy ($(mem_now) MiB) after 900s" >&2
    exit 1
  fi
  sleep 5 & wait $!
done
BASELINE=$(mem_now)

DEVENV_MODEL_VARS="NV_CHAT_MODEL_DIR NV_CHAT_MODEL_DIRS NV_EAGLE3_DRAFT_DIR NV_EMBEDDING_MODEL_DIR NV_GEMMA4_VERIFIER_DIR NV_SPEAKER_MODEL_DIR NV_TTS_TALKER_DIR"
declare -A CALLER_ENV
for v in $DEVENV_MODEL_VARS; do
  if [ -n "${!v+x}" ]; then CALLER_ENV[$v]="${!v}"; fi
done

set +u
source "$HOME/.cache/cargo-tmp/devenv-__cuda.sh"
set -u

for v in $DEVENV_MODEL_VARS; do
  if [ -n "${CALLER_ENV[$v]+x}" ]; then
    export "$v"="${CALLER_ENV[$v]}"
  else
    unset "$v"
  fi
done
export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
case ":$PATH:" in
*:/run/current-system/sw/bin:*) ;;
*) export PATH="$PATH:/run/current-system/sw/bin" ;;
esac

: >"$SAMPLES"
BOOT_T0=$(date +%s.%N)
setsid env UVICORN_PORT="$PORT" RUST_LOG=info,speaches_plus=debug "$BIN" >"$LOG" 2>&1 &
PG=$!
trap 'kill -TERM -- "-$PG" 2>/dev/null || true' EXIT

( while kill -0 "$PG" 2>/dev/null; do mem_now >>"$SAMPLES"; sleep 0.5 & wait $!; done ) &
SAMPLER=$!

t=0
until curl -fsS "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; do
  t=$((t + 3))
  if [ "$t" -ge "$BOOT_TIMEOUT" ] || ! kill -0 "$PG" 2>/dev/null; then
    echo "BOOT-FAIL after ${t}s; log tail:" >&2
    tail -n 30 "$LOG" >&2
    kill "$SAMPLER" 2>/dev/null || true
    exit 1
  fi
  sleep 3 & wait $!
done
BOOT_ELAPSED=$(echo "$(date +%s.%N) $BOOT_T0" | awk '{printf "%.1f", $1-$2}')
PEAK_BOOT=$(sort -n "$SAMPLES" | tail -n1)

sleep 4 & wait $!
STEADY_READY=$(tail -n 6 "$SAMPLES" | median_of)

MODEL_ID=""
PEAK_REQ=null
STEADY_AFTER=null
REQ_OK=false
if [ "$REQ" = "chat" ]; then
  MODEL_ID=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" | python3 -c "
import sys, json
ids = [m['id'] for m in json.load(sys.stdin)['data']]
flt = '$MODEL_FILTER'.lower()
cand = [i for i in ids if flt in i.lower()] if flt else ids
print(cand[0] if cand else '')")
  MARK=$(grep -c '' "$SAMPLES")
  if [ -n "$MODEL_ID" ] && curl -fsS -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'content-type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Name three rivers.\"}]}" \
      >"$OUT/$ARM.req.json" 2>>"$LOG"; then
    REQ_OK=true
  fi
  sleep 4 & wait $!
  PEAK_REQ=$(tail -n +"$MARK" "$SAMPLES" | sort -n | tail -n1)
  STEADY_AFTER=$(tail -n 6 "$SAMPLES" | median_of)
elif [ "$REQ" = "embed" ]; then
  MARK=$(grep -c '' "$SAMPLES")
  if curl -fsS -m 120 "http://127.0.0.1:$PORT/v1/embeddings" \
      -H 'content-type: application/json' \
      -d '{"model":"qwen3-embedding","input":"a small measurement probe"}' \
      >"$OUT/$ARM.req.json" 2>>"$LOG"; then
    REQ_OK=true
  fi
  sleep 4 & wait $!
  PEAK_REQ=$(tail -n +"$MARK" "$SAMPLES" | sort -n | tail -n1)
  STEADY_AFTER=$(tail -n 6 "$SAMPLES" | median_of)
fi

kill -TERM -- "-$PG" 2>/dev/null || true
trap - EXIT
for _ in 1 2 3 4 5 6 7 8; do
  kill -0 "$PG" 2>/dev/null || break
  sleep 5 & wait $!
done
kill -KILL -- "-$PG" 2>/dev/null || true
kill "$SAMPLER" 2>/dev/null || true
wait "$SAMPLER" 2>/dev/null || true

t0=$SECONDS
until [ "$(mem_now)" -le $((BASELINE + 100)) ] || [ $((SECONDS - t0)) -gt 120 ]; do sleep 3 & wait $!; done
FINAL=$(mem_now)

cat >"$RESULT" <<EOF
{"arm":"$ARM","baseline_mib":$BASELINE,"peak_boot_mib":$PEAK_BOOT,
"steady_ready_mib":$STEADY_READY,"peak_req_mib":$PEAK_REQ,"steady_after_req_mib":$STEADY_AFTER,
"boot_elapsed_s":$BOOT_ELAPSED,"req_ok":$REQ_OK,"model_id":"$MODEL_ID","final_after_kill_mib":$FINAL,
"delta_steady_mib":$((STEADY_READY - BASELINE)),"log":"$LOG"}
EOF
cat "$RESULT"
