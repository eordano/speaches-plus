#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
train_resident_ab.sh -- STAGED ppl A/B for NV_TRAIN_QUANT_RESIDENT, run only in a
negotiated GPU window.

Two windowed Qwen3.8-27B training runs with identical data/seed/steps/hyperparameters:
  arm host      NV_TRAIN_QUANT_RESIDENT unset  (host-dequant bf16 base, today's default)
  arm resident  NV_TRAIN_QUANT_RESIDENT=1      (nvfp4/fp8 checkpoint modules stay
                resident; forwards run the serving quant GEMM, backward is dx-only
                through QuantLinearOp; LoRA deltas carry the trainable parameters)

Per arm it reports train tok/s (parsed from the trainer's per-step stderr rows),
peak VRAM (1 Hz nvidia-smi sampler), and eval_loss_adapter / eval_loss_base per
row-count-normalised nats. These are the ppl rows that gate any future default flip;
the default stays host-dequant until rows land per the repo's
never-flip-a-performance-default rule.

Required env:
  BASE       Qwen3.8-27B-NVFP4 snapshot dir (config.json + *.safetensors)
  DATA       train jsonl
  EVAL_DATA  held-out jsonl
Optional env:
  OUT_ROOT ($HOME/.cache/nv-train/resident-ab/<ts>)  STEPS (4)  GRAD_LAYERS (2)
  RANK (8)  ALPHA (8.0)  LR (1e-4)  SEED (7)  NVK_LANE (resab)  GPU_GATE_MIB (2000)
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
BASE=${BASE:?usage: BASE=<qwen38 packed snapshot dir> (see --help)}
DATA=${DATA:?usage: DATA=<train jsonl> (see --help)}
EVAL_DATA=${EVAL_DATA:?usage: EVAL_DATA=<held-out jsonl> (see --help)}
OUT_ROOT=${OUT_ROOT:-$HOME/.cache/nv-train/resident-ab/$(date +%Y%m%d-%H%M%S)}
STEPS=${STEPS:-4}
GRAD_LAYERS=${GRAD_LAYERS:-2}
RANK=${RANK:-8}
ALPHA=${ALPHA:-8.0}
LR=${LR:-1e-4}
SEED=${SEED:-7}
NVK_LANE=${NVK_LANE:-resab}
GPU_GATE_MIB=${GPU_GATE_MIB:-2000}
mkdir -p "$OUT_ROOT"

gpu_gate() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$used" -ge "$GPU_GATE_MIB" ]; then
    echo "REFUSING $1: GPU memory.used=${used}MiB >= ${GPU_GATE_MIB}MiB (run only in a negotiated idle window)" >&2
    exit 2
  fi
  echo "gpu gate ok before $1: memory.used=${used}MiB < ${GPU_GATE_MIB}MiB"
}

run_arm() {
  local arm=$1
  local out_dir=$OUT_ROOT/$arm
  local log=$OUT_ROOT/$arm.log
  local vram=$OUT_ROOT/$arm.vram.csv
  gpu_gate "arm $arm"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 >"$vram" &
  local sampler=$!
  local rc=0
  (
    cd "$PROJECT_ROOT/rust"
    if [ "$arm" = "resident" ]; then export NV_TRAIN_QUANT_RESIDENT=1; else unset NV_TRAIN_QUANT_RESIDENT; fi
    NV_Q38_RESIDENT_AB_TEST=1 \
    NV_Q38_AB_BASE="$BASE" \
    NV_Q38_AB_DATA="$DATA" \
    NV_Q38_AB_EVAL_DATA="$EVAL_DATA" \
    NV_Q38_AB_OUT="$out_dir" \
    NV_Q38_AB_STEPS="$STEPS" \
    NV_Q38_AB_GRAD_LAYERS="$GRAD_LAYERS" \
    NV_Q38_AB_RANK="$RANK" \
    NV_Q38_AB_ALPHA="$ALPHA" \
    NV_Q38_AB_LR="$LR" \
    NV_Q38_AB_SEED="$SEED" \
    NVK_LANE="$NVK_LANE" NVK_PKG=nv-train \
      scripts/nvk.sh test --no-fail-fast --test qwen38_quant_resident -- \
      --ignored --nocapture
  ) >"$log" 2>&1 || rc=$?
  kill "$sampler" 2>/dev/null || true
  wait "$sampler" 2>/dev/null || true
  if [ "$rc" -ne 0 ]; then
    echo "arm $arm failed rc=$rc; tail of $log:" >&2
    tail -30 "$log" >&2
    exit 1
  fi
  test -f "$out_dir/resident_train_summary.json" || {
    echo "arm $arm: no $out_dir/resident_train_summary.json; the ignored test silently ran nothing" >&2
    exit 1
  }
}

for arm in host resident; do
  run_arm "$arm"
done

GIT_SHA=$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
python3 - "$OUT_ROOT" "$BASE" "$STEPS" "$GRAD_LAYERS" "$RANK" "$LR" "$SEED" "$GIT_SHA" <<'EOF'
import json, pathlib, re, sys

out_root, base, steps, grad_layers, rank, lr, seed, sha = sys.argv[1:9]
root = pathlib.Path(out_root)

def arm_row(arm):
    s = json.loads((root / arm / "resident_train_summary.json").read_text())
    log = (root / f"{arm}.log").read_text()
    step_secs = [float(m) for m in re.findall(r"\[q38-lora-train\] step\s+\d+\s+loss\s+\S+\s+(\d+\.\d+)s", log)]
    toks = re.search(r"train_tokens_per_step (\d+)", log)
    toks = int(toks.group(1)) if toks else 0
    tok_s = (toks * len(step_secs) / sum(step_secs)) if step_secs and sum(step_secs) > 0 else float("nan")
    vram = [int(x) for x in (root / f"{arm}.vram.csv").read_text().split() if x.strip().isdigit()]
    n = max(s["num_eval_examples"], 1)
    return {
        "arm": arm,
        "resident_base": s["resident_base"],
        "base_dtype": s["base_dtype"],
        "train_tok_s": tok_s,
        "peak_vram_mib": max(vram) if vram else 0,
        "eval_loss_base": s["eval_loss_base"] / n if s["eval_loss_base"] is not None else None,
        "eval_loss_adapter": s["eval_loss_adapter"] / n if s["eval_loss_adapter"] is not None else None,
        "train_losses": s["train_losses"],
        "wall_seconds": s["wall_seconds"],
        "n_eval": s["num_eval_examples"],
    }

rows = [arm_row("host"), arm_row("resident")]
assert not rows[0]["resident_base"] and rows[1]["resident_base"], (
    "arm identity mismatch: the summaries do not carry the arms the env selected"
)
hdr = "arm       tok/s    peakMiB  base_ppl_nats  adpt_ppl_nats  gain_nats"
print(hdr)
for r in rows:
    gain = (r["eval_loss_adapter"] - r["eval_loss_base"]) if r["eval_loss_adapter"] is not None else float("nan")
    r["gain"] = gain
    print(f"{r['arm']:<8}  {r['train_tok_s']:7.1f}  {r['peak_vram_mib']:7d}  "
          f"{r['eval_loss_base']:13.5f}  {r['eval_loss_adapter']:13.5f}  {gain:+9.5f}")
host, res = rows
print(f"resident-vs-host: base_delta {res['eval_loss_base']-host['eval_loss_base']:+.5f} nats  "
      f"adapter_delta {res['eval_loss_adapter']-host['eval_loss_adapter']:+.5f} nats  "
      f"vram_delta {res['peak_vram_mib']-host['peak_vram_mib']:+d} MiB  "
      f"tok_s_ratio {res['train_tok_s']/host['train_tok_s']:.2f}x")
basis = {
    "checkpoint": base,
    "harness": "nv-train tests/qwen38_quant_resident.rs resident_arm_on_a_real_packed_base_trains_and_writes_the_ab_sidecar",
    "backend": "cuda",
    "steps": int(steps),
    "grad_layers": int(grad_layers),
    "rank": int(rank),
    "lr": float(lr),
    "seed": int(seed),
    "git_sha": sha,
    "logs": [str(root / "host.log"), str(root / "resident.log")],
    "units": "nats per eval row (masked-mean CE per row, summed loss / n_eval)",
}
(root / "verdict.json").write_text(json.dumps({"rows": rows, "basis": basis}, indent=2))
print("basis:", json.dumps(basis))
print("artifacts:", out_root)
EOF
