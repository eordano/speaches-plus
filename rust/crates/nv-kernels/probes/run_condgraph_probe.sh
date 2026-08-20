#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CACHE="${NVK_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/cargo-tmp}"
ENVFILE="$CACHE/devenv-__cuda.sh"
OUTDIR="${TMPDIR:-$HOME/.cache/nvk-tmp}"
mkdir -p "$OUTDIR"
OUT="$OUTDIR/condgraph_probe"

if [ -s "$ENVFILE" ]; then
  # shellcheck disable=SC1090
  source "$ENVFILE" 2>/dev/null || true
fi
command -v nvcc >/dev/null || {
  echo "nvcc not on PATH (source the cuda devshell)"
  exit 2
}

echo "== compile =="
nvcc -arch=sm_120 -O2 -std=c++17 -o "$OUT" "$HERE/condgraph_probe.cu"
echo "compiled: $OUT"

echo "== idle-gate =="
for _ in $(seq 1 90); do
  u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$u" -lt 20 ] && [ "$m" -lt 12000 ]; then break; fi
  echo "gpu busy u=${u}% m=${m}MiB, waiting"
  sleep 2
done

echo "== run =="
"$OUT"
