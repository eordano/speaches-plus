#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
lora_accept_curve_runbook.sh -- the REAL acceptance-curve measurement, staged end to end.

Trains a genuine rank-8 adapter on captured trajectories via POST
/v1/fine_tuning/jobs?wait=true (no promote, no runtime load), then measures MTP
acceptance adapter-on vs adapter-off over >= 9 independent held-out prompt
windows with per-round KL(base||adapter) from verify-chain logit rows
(rust/crates/nv-specdecode/tests/lora_qwen38_wgpu_real_adapter_accept_curve_perfb.rs).

Required env:
  MODEL_ID   served teacher model id exactly as /v1/models lists it
  BASE_DIR   Qwen3.8-27B-NVFP4 snapshot dir (config.json + model_mtp.safetensors)
  TRAJ_DIR   NV_TRAJECTORY_DIR holding trajectories*.jsonl
Optional env:
  SERVER (http://127.0.0.1:1327)  OUT_DIR  STEPS (20)  LR (1e-4; 0.05 and 0.005
  measurably diverge on self-distillation corpora, 1e-4 trains clean)
  EMITTED (128)  NVK_LANE (gape)  TRAIN_GATE_MAX_MIB (2000; raise when the
  serving engine is legitimately resident during the training stage -- it gates
  stage 2 only)  STAGE3_IDLE_WAIT_S (900)
  SKIP_TRAIN_ADAPTER_DIR (reuse an already-trained adapter dir and skip stage 2)

GPU blocks: stage 2 trains inside the running server process, gated by
TRAIN_GATE_MAX_MIB; stage 3 boots two ~22.6 GB wgpu engines and needs the card
idle at 2000 MiB. One invocation runs 0-2 against the live server, then stage 3
WAITS up to STAGE3_IDLE_WAIT_S seconds for you to stop the server instead of
exiting 2, and only a wait timeout refuses. Both stages belong in negotiated
windows.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
SERVER=${SERVER:-http://127.0.0.1:1327}
MODEL_ID=${MODEL_ID:?usage: MODEL_ID=<served model id> (see --help)}
BASE_DIR=${BASE_DIR:?usage: BASE_DIR=<qwen38 snapshot dir> (see --help)}
TRAJ_DIR=${TRAJ_DIR:?usage: TRAJ_DIR=<NV_TRAJECTORY_DIR> (see --help)}
OUT_DIR=${OUT_DIR:-$HOME/.cache/lora-accept-curve/$(date +%Y%m%d-%H%M%S)}
STEPS=${STEPS:-20}
LR=${LR:-1e-4}
EMITTED=${EMITTED:-128}
TRAIN_GATE_MAX_MIB=${TRAIN_GATE_MAX_MIB:-2000}
TWO_ENGINE_STAGE_IDLE_GATE_MIB=2000
STAGE3_IDLE_WAIT_S=${STAGE3_IDLE_WAIT_S:-900}
mkdir -p "$OUT_DIR"

stage() { printf '\n=== %s ===\n' "$*"; }

gpu_used_mib() {
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '
}

gpu_gate() {
  local limit=$1 what=$2 used
  used=$(gpu_used_mib)
  if [ "$used" -ge "$limit" ]; then
    echo "REFUSING $what: GPU memory.used=${used}MiB >= ${limit}MiB (another process is resident; run this stage only in its negotiated window)" >&2
    exit 2
  fi
  echo "gpu gate ok before $what: memory.used=${used}MiB < ${limit}MiB"
}

gpu_wait_idle() {
  local limit=$1 what=$2 deadline used
  deadline=$(( $(date +%s) + STAGE3_IDLE_WAIT_S ))
  used=$(gpu_used_mib)
  if [ "$used" -ge "$limit" ]; then
    echo "waiting for $what: memory.used=${used}MiB >= ${limit}MiB -- stop the serving process now (stages 0-2 needed it live, stage 3 needs the card); polling up to ${STAGE3_IDLE_WAIT_S}s"
  fi
  while [ "$used" -ge "$limit" ]; do
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "REFUSING $what: still memory.used=${used}MiB >= ${limit}MiB after ${STAGE3_IDLE_WAIT_S}s" >&2
      exit 2
    fi
    sleep 10
    used=$(gpu_used_mib)
  done
  echo "gpu idle for $what: memory.used=${used}MiB < ${limit}MiB"
}

stage "0 preflight"
test -f "$BASE_DIR/config.json"
test -f "$BASE_DIR/model_mtp.safetensors"
ls "$TRAJ_DIR"/trajectories*.jsonl >/dev/null
curl -fsS --max-time 10 "$SERVER/v1/models" >"$OUT_DIR/models.json"
python3 -c "
import json,sys
ids=[m['id'] for m in json.load(open('$OUT_DIR/models.json'))['data']]
assert '$MODEL_ID' in ids, f'$MODEL_ID not served; /v1/models lists {ids}'
print('teacher model served:', '$MODEL_ID')
"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee "$OUT_DIR/nvidia-smi-preflight.csv"

stage "1 extract >=9 independent held-out prompt windows plus train/validation jsonl"
python3 "$SCRIPT_DIR/extract_holdout_prompt_windows.py" \
  --trajectories "$TRAJ_DIR" \
  --model "$MODEL_ID" \
  --out-dir "$OUT_DIR" | tee "$OUT_DIR/extract.log"

if [ -n "${SKIP_TRAIN_ADAPTER_DIR:-}" ]; then
  stage "2 SKIPPED: reusing $SKIP_TRAIN_ADAPTER_DIR"
  ADAPTER_DIR=$SKIP_TRAIN_ADAPTER_DIR
else
  stage "2 GPU BLOCK: train the genuine rank-8 adapter via POST /v1/fine_tuning/jobs?wait=true"
  gpu_gate "$TRAIN_GATE_MAX_MIB" "training"
  python3 - "$SERVER" "$BASE_DIR" "$OUT_DIR" "$STEPS" "$LR" <<'EOF' | tee "$OUT_DIR/job.log"
import json, sys, urllib.request
server, base_dir, out_dir, steps, lr = sys.argv[1:6]
body = {
    "model": base_dir,
    "training_file": f"{out_dir}/train.jsonl",
    "validation_file": f"{out_dir}/validation.jsonl",
    "suffix": "acceptcurve",
    "hyperparameters": {"rank": 8, "steps": int(steps), "learning_rate": float(lr)},
}
req = urllib.request.Request(
    f"{server}/v1/fine_tuning/jobs?wait=true",
    data=json.dumps(body).encode(),
    headers={"content-type": "application/json"},
)
job = json.load(urllib.request.urlopen(req, timeout=7200))
open(f"{out_dir}/job.json", "w").write(json.dumps(job, indent=1))
assert job.get("status") == "succeeded", f"job did not succeed: {job}"
assert job.get("result_files"), f"no result_files: {job}"
m = job.get("metrics") or {}
print("fine_tuned_model:", job.get("fine_tuned_model"))
print("adapter_dir:", job["result_files"][0])
print("eval_loss_adapter:", m.get("eval_loss_adapter"), "eval_loss_base:", m.get("eval_loss_base"))
EOF
  ADAPTER_DIR=$(python3 -c "import json;print(json.load(open('$OUT_DIR/job.json'))['result_files'][0])")
fi
test -f "$ADAPTER_DIR/adapter_model.safetensors"
test -f "$ADAPTER_DIR/adapter_config.json"
echo "adapter: $ADAPTER_DIR"

stage "3 GPU BLOCK: acceptance curve, base vs real adapter, per-round verify-chain KL"
gpu_wait_idle "$TWO_ENGINE_STAGE_IDLE_GATE_MIB" "the two-engine wgpu measurement"
cd "$PROJECT_ROOT/rust"
NV_QWEN38_WGPU_TEST=1 \
NV_QWEN38_DIR="$BASE_DIR" \
NV_LORA_CURVE_ADAPTER_DIR="$ADAPTER_DIR" \
NV_LORA_CURVE_PROMPTS="$OUT_DIR/prompt_windows.json" \
NV_LORA_CURVE_EMITTED="$EMITTED" \
NVK_LANE=${NVK_LANE:-gape} NVK_PKG=nv-specdecode NVK_FEATURES=cuda,wgpu \
  scripts/nvk.sh probe --test lora_qwen38_wgpu_real_adapter_accept_curve_perfb -- \
  --ignored --nocapture 2>&1 | tee "$OUT_DIR/accept_curve.log"

stage "4 rows (08.2-shaped: paired windows, n named, basis on every row)"
grep -E 'LORA-CURVE-(BOOT|WINDOW|KL-WINDOW|PAIRED)' "$OUT_DIR/accept_curve.log" | tee "$OUT_DIR/rows.txt"
echo
echo "verdict input: the LORA-CURVE-PAIRED row above (n, mean_delta_accept, t, signs);"
echo "per-round KL rows: grep LORA-CURVE-KL $OUT_DIR/accept_curve.log"
echo "artifacts: $OUT_DIR"
