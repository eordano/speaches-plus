#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANE="${NVK_LANE:-gapb}"
LOG_DIR="${NV_LORA_PK_AB_LOG_DIR:-$HOME/.cache/nv-lora-pk-ab}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"

echo "=== pinning graph env arms to defaults (inherited fold/fuse/tail arms break the"
echo "    pass-delta==ledger asserts and FUSE_GELU=1 refuses the dense gate_up adapter)"
for v in $(compgen -v | grep -E '^(NV_E4B_WGPU_|NV_WGPU_NA|NV_E4B_W4_|NV_LAGUNA_WGPU_|NV_WGPU_PREFILL_M)' || true); do
  echo "    unset $v (was ${!v})"
  unset "$v"
done

echo "=== prebuild (CPU only; safe while the GPU is busy)"
NVK_LANE="$LANE" NVK_PKG=nv-models NVK_FEATURES=cuda,wgpu,laguna-wip \
  "$REPO/scripts/nvk.sh" test --release --no-run \
  --test lora_e4b_wgpu --test lora_laguna_wgpu \
  >"$LOG_DIR/lora-pk-ab-$STAMP-prebuild.log" 2>&1

used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
if [ "$used_mib" -ge 2000 ]; then
  echo "GPU busy: memory.used=${used_mib} MiB >= 2000; refusing to start" >&2
  exit 2
fi

echo "=== fixture-scale gates first (seconds; a red here voids the perf arms)"
GATE_LOG="$LOG_DIR/lora-pk-ab-$STAMP-gates.log"
NVK_LANE="$LANE" NVK_PKG=nv-models NVK_FEATURES=cuda,wgpu,laguna-wip \
  "$REPO/scripts/nvk.sh" test --no-fail-fast \
  --test lora_e4b_wgpu --test lora_laguna_wgpu --test lora_q3d_wgpu -- --test-threads=1 \
  >"$GATE_LOG" 2>&1 || { tail -40 "$GATE_LOG"; exit 3; }
grep -E "test result" "$GATE_LOG"

PERF_FAIL=0

E4B_LOG="$LOG_DIR/lora-pk-ab-$STAMP-e4b.log"
echo "=== e4b real-weight adapter decode A/B log=$E4B_LOG"
NV_E4B_WGPU_TEST=1 \
  NVK_LANE="$LANE" NVK_PKG=nv-models NVK_FEATURES=cuda,wgpu,laguna-wip \
  "$REPO/scripts/nvk.sh" test --release --test lora_e4b_wgpu -- \
  --ignored --nocapture real_e4b_adapter_decode_ab_perfb \
  >"$E4B_LOG" 2>&1 || PERF_FAIL=1
grep -E "LORA-E4B-AB|adapter: rank|panicked" "$E4B_LOG" || tail -20 "$E4B_LOG"
grep -q "LORA-E4B-AB summary" "$E4B_LOG" || {
  echo "e4b perf arm produced no LORA-E4B-AB summary row (failed or silently skipped)"
  PERF_FAIL=1
}

LAG_LOG="$LOG_DIR/lora-pk-ab-$STAMP-laguna.log"
echo "=== laguna real-weight adapter decode A/B log=$LAG_LOG"
NV_LAGUNA_WGPU_TEST=1 \
  NVK_LANE="$LANE" NVK_PKG=nv-models NVK_FEATURES=cuda,wgpu,laguna-wip \
  "$REPO/scripts/nvk.sh" test --release --test lora_laguna_wgpu -- \
  --ignored --nocapture real_laguna_adapter_decode_ab_perfb \
  >"$LAG_LOG" 2>&1 || PERF_FAIL=1
grep -E "LORA-LAGUNA-AB|adapter: rank|panicked" "$LAG_LOG" || tail -20 "$LAG_LOG"
grep -q "LORA-LAGUNA-AB summary" "$LAG_LOG" || {
  echo "laguna perf arm produced no LORA-LAGUNA-AB summary row (failed or silently skipped)"
  PERF_FAIL=1
}

echo "=== expectation ledger (predictions on record before this run) ==="
echo "e4b adapter passes = base + 8*n_layers (pk pair per qkv/o/gate_up/down site);"
echo "  the legacy widen/fused/repack ledger was base + 16*(n_layers-shared) + 12*shared,"
echo "  so the adapter dispatch tax halves (16->8) on non-shared layers and drops a third on shared"
echo "laguna adapter passes = base + 2*(3 sites attn + 2 sites on dense-mlp layers);"
echo "  first laguna lora numbers on record, no prior baseline to beat -- gate is delta_ms per pass"
echo "  landing near the ~7us wgpu dispatch floor times the pass count"

if [ "$PERF_FAIL" -ne 0 ]; then
  echo "RESULT: perf arm failure or silent skip above; this window produced no usable row" >&2
  exit 4
fi
echo "RESULT: all arms green"
