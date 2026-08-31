#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
quant_linear_release_bench.sh -- the fp8/nvfp4 QuantLinearOp requantification, staged.

Runs nv-layers/tests/quant_linear_train_bench_perfc.rs in --release under the
probe GPU flock: fwd_serve at the training m=4 shape AND the serving m=1 gemv
shape, plus fwd+bwd, each quant arm priced against its host-dequant dense twin.
The earlier numbers were debug-profile; quote only rows from this invocation.
Optional env: NVK_LANE (gape)  OUT_DIR
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
OUT_DIR=${OUT_DIR:-$HOME/.cache/quant-linear-bench/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT_DIR"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$used" -ge 2000 ]; then
  echo "REFUSING: GPU memory.used=${used}MiB >= 2000MiB (wall-clock micro-bench needs a quiet card)" >&2
  exit 2
fi
echo "gpu idle gate ok: memory.used=${used}MiB"

cd "$PROJECT_ROOT/rust"
NV_QUANT_LINEAR_TRAIN_BENCH=1 \
NVK_LANE=${NVK_LANE:-gape} NVK_PKG=nv-layers \
  scripts/nvk.sh probe --test quant_linear_train_bench_perfc -- \
  --ignored --nocapture 2>&1 | tee "$OUT_DIR/quant_linear_release.log"

grep -F '[quant-linear-train-bench]' "$OUT_DIR/quant_linear_release.log" | tee "$OUT_DIR/rows.txt"
echo "artifacts: $OUT_DIR"
