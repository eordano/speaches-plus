#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
lora_deep_serving_row_172k.sh -- the missing 172k-depth adapter-vs-base serving row, staged.

VRAM arithmetic on the RTX PRO 6000 Blackwell (97887 MiB total), per wgpu engine
at NV_CTX_TOKENS=172k (depth 176128, max_seq 177152 with the 1024 headroom):
  weights ~22.6 GB resident; per-token full-attn KV under the default arm stack
  (f16 K+V 4096 B + default-on NV_Q3D_KV_FP8 pair 2048 B + scales 32 B +
  default-on NV_Q3D_KV_FP8_TP K twin 1024 B = 7200 B/layer x 16 layers =
  115200 B) -> 20.4 GB, plus the drafter's own KV layer ~1.3 GB, plus prefill/
  verify/logits buffers ~1 GB: ~45 GB (~42 GiB) per engine. The earlier ~40 GiB
  per-engine note is confirmed to first order. TWO engines = ~90 GB against the
  102.6 GB card: on-paper fit with <10% headroom, inside Vulkan budget noise --
  so the SEQUENTIAL single-engine run is the default here and the same-session
  two-engine suite is the opt-in (TWO_ENGINE=1) for when the card is fully idle
  and the attempt is worth one boot to find out.

Default path: two processes, one engine each, same depth, rows labeled
pairing=cross_process (clock drift sits inside the pair; quote it as such).
Optional env: DEPTH (172k)  PAIRS (3)  NVK_LANE (gape)  OUT_DIR  TWO_ENGINE=1
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
DEPTH=${DEPTH:-172k}
PAIRS=${PAIRS:-3}
OUT_DIR=${OUT_DIR:-$HOME/.cache/lora-deep-row/$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT_DIR"

stage() { printf '\n=== %s ===\n' "$*"; }

gpu_idle_gate() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "$used" -ge 2000 ]; then
    echo "REFUSING $1: GPU memory.used=${used}MiB >= 2000MiB (this depth needs the whole card)" >&2
    exit 2
  fi
  echo "gpu idle gate ok before $1: memory.used=${used}MiB"
}

run_probe() {
  NV_QWEN38_WGPU_TEST=1 NV_CTX_TOKENS="$DEPTH" NV_LORA_AB_PAIRS="$PAIRS" \
    NVK_LANE=${NVK_LANE:-gape} NVK_PKG=nv-specdecode NVK_FEATURES=cuda,wgpu \
    scripts/nvk.sh probe "$@"
}

cd "$PROJECT_ROOT/rust"

if [ "${TWO_ENGINE:-0}" = "1" ]; then
  stage "GPU BLOCK: two-engine same-session interleaved pairs at $DEPTH (preferred basis when it fits)"
  gpu_idle_gate "the two-engine attempt"
  run_probe --test lora_qwen38_wgpu_mtp_ab_perfb -- --ignored --nocapture \
    2>&1 | tee "$OUT_DIR/two_engine_$DEPTH.log"
  grep -E 'LORA-AB-(BOOT|PLAIN|MTP|SKIP)' "$OUT_DIR/two_engine_$DEPTH.log" | tee "$OUT_DIR/rows.txt"
  echo "artifacts: $OUT_DIR"
  exit 0
fi

stage "GPU BLOCK 1/2: single-engine arm=base at $DEPTH"
gpu_idle_gate "arm=base"
NV_LORA_AB_ARM=base run_probe --test lora_qwen38_wgpu_deep_single_engine_perfb -- \
  --ignored --nocapture 2>&1 | tee "$OUT_DIR/se_base_$DEPTH.log"

stage "GPU BLOCK 2/2: single-engine arm=lora at $DEPTH (back to back for the cross-process pair)"
gpu_idle_gate "arm=lora"
NV_LORA_AB_ARM=lora run_probe --test lora_qwen38_wgpu_deep_single_engine_perfb -- \
  --ignored --nocapture 2>&1 | tee "$OUT_DIR/se_lora_$DEPTH.log"

stage "rows (pair base/lora by pair index; pairing=cross_process is named on every row)"
grep -hE 'LORA-AB-SE-BOOT|LORA-AB-(PLAIN|MTP)' \
  "$OUT_DIR/se_base_$DEPTH.log" "$OUT_DIR/se_lora_$DEPTH.log" | tee "$OUT_DIR/rows.txt"
echo "artifacts: $OUT_DIR"
