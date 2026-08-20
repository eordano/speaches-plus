#!/usr/bin/env bash
set -euo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
PERF=$(dirname "$SELF")
REPO=$(cd "$PERF/.." && pwd)
NVK="$REPO/rust/scripts/nvk.sh"
LOGDIR="${PERFROOM_LOGDIR:-$HOME/.cache/perfroom/logs}"
GGUF_SHA_CACHE="$HOME/.cache/perfroom/gguf-sha"
LLAMA_BENCH_VULKAN="$HOME/.cache/llama-bench-build/build-vulkan/bin/llama-bench"
LLAMA_BENCH_CUDA="$HOME/.cache/llama-bench-build/build/bin/llama-bench"

usage() {
  cat <<'EOF'
run-harness.sh -- measure one arm and append validated rows to perf/runs.jsonl

  run-harness.sh ours --build-name S --model S --backend wgpu-vulkan|cuda \
      --suite TESTFILE_STEM --exact TEST_FN --parse ctx-scaling|prefill|gen-arm|ppl \
      [--pkg nv-models] [--features cuda,wgpu] [--env K=V]... [--timeout SEC] \
      [--max-seq N] [--corpus PATH] [--sampling S] [--batch N] [--notes S]

  run-harness.sh llamacpp --build-name S --model S --bin vulkan|cuda --gguf PATH \
      [--bench-args "-p 2048 -n 256"] [--timeout SEC] [--notes S]

  run-harness.sh exec --build-name S --model S --backend cuda --engine-name vllm \
      --parse vllm-bench --cmd "shell command" [--flags-json '{"k":"v"}'] \
      [--checkpoint-repo R] [--timeout SEC] [--notes S]

Every ours run goes through nvk.sh probe (release, exclusive GPU flock,
NVK_LANE=perfroom). Rows missing required schema-v1 fields are refused by
parse_and_append.py. NV_WGPU_PROFILE runs are refused: profiling doubles step
time so profiled absolute numbers must never enter the store.
EOF
  exit 1
}

[ $# -ge 1 ] || usage
ENGINE_MODE="$1"
shift

BUILD_NAME="" MODEL="" BACKEND="" SUITE="" EXACT="" PARSE="" PKG="nv-models"
FEATURES="cuda,wgpu" TIMEOUT=1800 MAX_SEQ="" CORPUS="" NOTES="" SAMPLING="greedy"
BATCH=1 BIN="" GGUF="" BENCH_ARGS="" CHECKPOINT_REPO="" STATUS_OVERRIDE="" BENCH_ENV_RECORD=""
CMD="" ENGINE_NAME="" FLAGS_JSON=""
declare -a ENVS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --build-name) BUILD_NAME="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --suite) SUITE="$2"; shift 2 ;;
    --exact) EXACT="$2"; shift 2 ;;
    --parse) PARSE="$2"; shift 2 ;;
    --pkg) PKG="$2"; shift 2 ;;
    --features) FEATURES="$2"; shift 2 ;;
    --env) ENVS+=("$2"); shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --max-seq) MAX_SEQ="$2"; shift 2 ;;
    --corpus) CORPUS="$2"; shift 2 ;;
    --sampling) SAMPLING="$2"; shift 2 ;;
    --batch) BATCH="$2"; shift 2 ;;
    --notes) NOTES="$2"; shift 2 ;;
    --bin) BIN="$2"; shift 2 ;;
    --gguf) GGUF="$2"; shift 2 ;;
    --bench-args) BENCH_ARGS="$2"; shift 2 ;;
    --checkpoint-repo) CHECKPOINT_REPO="$2"; shift 2 ;;
    --status-override) STATUS_OVERRIDE="$2"; shift 2 ;;
    --cmd) CMD="$2"; shift 2 ;;
    --engine-name) ENGINE_NAME="$2"; shift 2 ;;
    --flags-json) FLAGS_JSON="$2"; shift 2 ;;
    *) echo "run-harness: unknown arg $1" >&2; usage ;;
  esac
done

[ -n "$BUILD_NAME" ] && [ -n "$MODEL" ] || usage
for kv in ${ENVS[@]+"${ENVS[@]}"}; do
  case "$kv" in
    NV_WGPU_PROFILE=*) echo "run-harness: REFUSING: NV_WGPU_PROFILE doubles step time; profiled absolutes never enter the store" >&2; exit 1 ;;
  esac
done
if [ -n "${NV_WGPU_PROFILE:-}" ]; then
  echo "run-harness: REFUSING: NV_WGPU_PROFILE is set in the caller environment" >&2
  exit 1
fi

mkdir -p "$LOGDIR" "$GGUF_SHA_CACHE"
RUN_ID="pr-$(date +%Y%m%d-%H%M%S)-$$"
LOG="$LOGDIR/$RUN_ID.log"
META="$LOGDIR/$RUN_ID.meta.json"
VRAM_SAMPLES="$LOGDIR/$RUN_ID.vram"

COMMIT_HASH=$(git -C "$REPO" rev-parse HEAD)
COMMIT_DATE=$(git -C "$REPO" show -s --format=%cI HEAD)
STARTED_AT=$(date -Iseconds)

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
POWER_LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1)
VRAM_BEFORE=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

(
  while true; do
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >>"$VRAM_SAMPLES" 2>/dev/null || true
    sleep 5
  done
) &
VRAM_PID=$!
trap 'kill $VRAM_PID 2>/dev/null || true' EXIT

T0=$(date +%s.%N)
EXIT_CODE=0

if [ "$ENGINE_MODE" = "ours" ]; then
  [ -n "$SUITE" ] && [ -n "$EXACT" ] && [ -n "$PARSE" ] && [ -n "$BACKEND" ] || usage
  ENGINE="ours"
  INSTRUMENT="$EXACT"
  SOURCE_FILE=$(grep -rl "fn $EXACT" "$REPO/rust/crates/$PKG/tests" "$REPO/rust/tests" 2>/dev/null | head -1 || true)
  set +e
  env ${ENVS[@]+"${ENVS[@]}"} \
    NVK_LANE=perfroom NVK_PKG="$PKG" NVK_FEATURES="$FEATURES" \
    timeout "$TIMEOUT" "$NVK" probe --test "$SUITE" -- --ignored --exact "$EXACT" --nocapture \
    >"$LOG" 2>&1
  EXIT_CODE=$?
  set -e
  SUITE_US=$(printf '%s' "$SUITE" | tr - _)
  BIN_PATH=$(ls -t "$HOME/.cache/cargo-tmp/tgt-perfroom/release/deps/$SUITE_US"-* 2>/dev/null | grep -v '\.d$' | head -1 || true)
  if [ -n "$BIN_PATH" ]; then
    BUILD_HASH=$(sha256sum "$BIN_PATH" | cut -c1-16)
  else
    BUILD_HASH="unbuilt"
  fi
elif [ "$ENGINE_MODE" = "llamacpp" ]; then
  [ -n "$BIN" ] && [ -n "$GGUF" ] || usage
  ENGINE="llama.cpp"
  case "$BIN" in
    vulkan)
      BENCH="$LLAMA_BENCH_VULKAN"
      BACKEND="${BACKEND:-vulkan}"
      export GGML_VK_VISIBLE_DEVICES="${GGML_VK_VISIBLE_DEVICES:-1}"
      BENCH_ENV_RECORD="GGML_VK_VISIBLE_DEVICES=$GGML_VK_VISIBLE_DEVICES"
      ;;
    cuda) BENCH="$LLAMA_BENCH_CUDA"; BACKEND="${BACKEND:-cuda}" ;;
    *) usage ;;
  esac
  INSTRUMENT="llama-bench"
  SOURCE_FILE="$BENCH"
  BUILD_HASH=$(sha256sum "$BENCH" | cut -c1-16)
  GGUF_BASE=$(basename "$GGUF")
  SHA_FILE="$GGUF_SHA_CACHE/$GGUF_BASE.sha256"
  if [ ! -s "$SHA_FILE" ]; then
    sha256sum "$GGUF" | cut -d' ' -f1 >"$SHA_FILE"
  fi
  GGUF_SHA=$(cat "$SHA_FILE")
  CHECKPOINT_REPO="${CHECKPOINT_REPO:-gguf:$GGUF_BASE}"
  set +e
  LD_LIBRARY_PATH="/run/opengl-driver/lib:$(dirname "$BENCH")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    flock "${NVK_GPU_LOCK:-/tmp/nvk-gpu0.lock}" \
    timeout "$TIMEOUT" "$BENCH" -m "$GGUF" $BENCH_ARGS -o json \
    >"$LOG" 2>&1
  EXIT_CODE=$?
  set -e
  PARSE="llama-bench"
elif [ "$ENGINE_MODE" = "exec" ]; then
  [ -n "$CMD" ] && [ -n "$PARSE" ] && [ -n "$BACKEND" ] || usage
  ENGINE="${ENGINE_NAME:-vllm}"
  INSTRUMENT="${INSTRUMENT_OVERRIDE:-exec:$PARSE}"
  SOURCE_FILE=""
  BUILD_HASH=$(printf '%s' "$CMD" | sha256sum | cut -c1-16)
  set +e
  flock "${NVK_GPU_LOCK:-/tmp/nvk-gpu0.lock}" \
    timeout "$TIMEOUT" bash -c "$CMD" >"$LOG" 2>&1
  EXIT_CODE=$?
  set -e
else
  usage
fi

T1=$(date +%s.%N)
WALL_S=$(python3 -c "print(round($T1 - $T0, 1))")
kill $VRAM_PID 2>/dev/null || true
trap - EXIT
VRAM_AFTER=$(sort -n "$VRAM_SAMPLES" 2>/dev/null | tail -1)
VRAM_AFTER="${VRAM_AFTER:-$VRAM_BEFORE}"
THROTTLE=$(nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader | head -1)

if [ "$EXIT_CODE" = "124" ]; then
  NOTES="${NOTES:+$NOTES | }timed out after ${TIMEOUT}s (30-min single-probe cap)"
fi

CORPUS_SHA=""
if [ -n "$CORPUS" ]; then
  CORPUS_SHA=$(sha256sum "$CORPUS" | cut -d' ' -f1)
fi

RUN_ID_BASE="$RUN_ID" LOG_PATH="$LOG" META_ENGINE="$ENGINE" META_BACKEND="$BACKEND" \
META_MODEL="$MODEL" META_BUILD_NAME="$BUILD_NAME" META_BUILD_HASH="$BUILD_HASH" \
META_COMMIT_HASH="$COMMIT_HASH" META_COMMIT_DATE="$COMMIT_DATE" META_STARTED_AT="$STARTED_AT" \
META_GPU_NAME="$GPU_NAME" META_DRIVER="$DRIVER" META_POWER_LIMIT="$POWER_LIMIT" \
META_THROTTLE="$THROTTLE" META_VRAM_BEFORE="$VRAM_BEFORE" META_VRAM_AFTER="$VRAM_AFTER" \
META_WALL_S="$WALL_S" META_INSTRUMENT="$INSTRUMENT" META_SOURCE_FILE="${SOURCE_FILE:-}" \
META_NOTES="$NOTES" META_SAMPLING="$SAMPLING" META_BATCH="$BATCH" META_MAX_SEQ="$MAX_SEQ" \
META_CORPUS="$CORPUS" META_CORPUS_SHA="$CORPUS_SHA" META_CHECKPOINT_REPO="$CHECKPOINT_REPO" \
META_GGUF_SHA="${GGUF_SHA:-}" META_BENCH_ARGS="$BENCH_ARGS${BENCH_ENV_RECORD:+ [env] $BENCH_ENV_RECORD}" META_STATUS_OVERRIDE="$STATUS_OVERRIDE" \
META_FLAGS_JSON="$FLAGS_JSON" \
META_ENVS=$(printf '%s\n' ${ENVS[@]+"${ENVS[@]}"}) \
python3 - "$META" <<'PYEOF'
import json, os, sys
flags = {}
for kv in os.environ.get("META_ENVS", "").splitlines():
    if kv:
        k, _, v = kv.partition("=")
        flags[k] = v
if os.environ["META_ENGINE"] == "llama.cpp":
    flags = {"llama_bench_args": os.environ.get("META_BENCH_ARGS", "")}
if os.environ.get("META_FLAGS_JSON"):
    flags = json.loads(os.environ["META_FLAGS_JSON"])
meta = {
    "run_id_base": os.environ["RUN_ID_BASE"],
    "started_at": os.environ["META_STARTED_AT"],
    "engine": os.environ["META_ENGINE"],
    "backend": os.environ["META_BACKEND"],
    "model": os.environ["META_MODEL"],
    "build_name": os.environ["META_BUILD_NAME"],
    "build_hash": os.environ["META_BUILD_HASH"],
    "commit_hash": os.environ["META_COMMIT_HASH"],
    "commit_date": os.environ["META_COMMIT_DATE"],
    "flags": flags,
    "device": {
        "gpu_name": os.environ["META_GPU_NAME"],
        "driver": os.environ["META_DRIVER"],
        "power_limit_w": float(os.environ["META_POWER_LIMIT"]),
        "throttle_flags": os.environ["META_THROTTLE"],
    },
    "instrument": os.environ["META_INSTRUMENT"],
    "source_file": os.environ.get("META_SOURCE_FILE") or None,
    "inference_args": {"sampling": os.environ["META_SAMPLING"], "batch": int(os.environ["META_BATCH"])},
    "max_seq_allocated": int(os.environ["META_MAX_SEQ"]) if os.environ.get("META_MAX_SEQ") else None,
    "vram_mb_before": int(os.environ["META_VRAM_BEFORE"]),
    "vram_mb_after": int(os.environ["META_VRAM_AFTER"]),
    "wall_s": float(os.environ["META_WALL_S"]),
    "log_path": os.environ["LOG_PATH"],
    "notes": os.environ.get("META_NOTES") or "",
    "corpus_path": os.environ.get("META_CORPUS") or None,
    "corpus_sha256": os.environ.get("META_CORPUS_SHA") or None,
    "status_override": os.environ.get("META_STATUS_OVERRIDE") or None,
}
if os.environ.get("META_CHECKPOINT_REPO"):
    rev = os.environ.get("META_GGUF_SHA", "")
    meta["checkpoint_repo"] = os.environ["META_CHECKPOINT_REPO"]
    if rev:
        meta["checkpoint"] = {"repo": os.environ["META_CHECKPOINT_REPO"], "revision": rev[:16]}
json.dump(meta, open(sys.argv[1], "w"), indent=1)
PYEOF

python3 "$PERF/parse_and_append.py" --meta "$META" --log "$LOG" --mode "$PARSE" --exit-code "$EXIT_CODE"
echo "run-harness: run_id=$RUN_ID exit=$EXIT_CODE wall=${WALL_S}s log=$LOG"
