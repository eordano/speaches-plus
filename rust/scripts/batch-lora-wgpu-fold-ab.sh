#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LANE="${NVK_LANE:-gapa}"
DEPTH="${NV_CTX_TOKENS:-256}"
LOG_DIR="${NV_LORA_FOLD_AB_LOG_DIR:-$HOME/.cache/nv-lora-fold-ab}"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d-%H%M%S)"

echo "=== prebuild (CPU only; safe while the GPU is busy)"
NVK_LANE="$LANE" NVK_PKG=nv-specdecode NVK_FEATURES=cuda,wgpu \
  "$REPO/scripts/nvk.sh" test --release --no-run --test lora_qwen38_wgpu_mtp_ab_perfb \
  >"$LOG_DIR/lora-fold-ab-$STAMP-prebuild.log" 2>&1

refuse_phase_unless_gpu_idle_because_a_ladder_can_start_mid_batch() {
  local phase="$1"
  local used_mib
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"
  echo "idle-gate phase=$phase memory.used=${used_mib}MiB uptime:$(uptime)"
  if [ "$used_mib" -ge 2000 ]; then
    echo "GPU busy at phase=$phase: memory.used=${used_mib} MiB >= 2000; refusing this phase (earlier arms' rows remain valid)" >&2
    exit 2
  fi
}

run_arm() {
  local arm="$1"
  shift
  local log="$LOG_DIR/lora-fold-ab-$STAMP-d$DEPTH-$arm.log"
  refuse_phase_unless_gpu_idle_because_a_ladder_can_start_mid_batch "arm-$arm"
  echo "=== arm=$arm depth=$DEPTH log=$log"
  env "$@" \
    NV_QWEN38_WGPU_TEST=1 NV_CTX_TOKENS="$DEPTH" \
    NVK_LANE="$LANE" NVK_PKG=nv-specdecode NVK_FEATURES=cuda,wgpu \
    "$REPO/scripts/nvk.sh" test --release --test lora_qwen38_wgpu_mtp_ab_perfb -- \
    --ignored --nocapture \
    qwen38_wgpu_lora_adapter_vs_base_mtp_decode_and_acceptance_interleaved_pairs \
    >"$log" 2>&1
  grep -E "LORA-AB-(BOOT|PASSES|PLAIN|MTP|KL|SKIP)" "$log" | sed "s/^/[$arm] /"
}

echo "=== fixture-scale gates first (seconds; a red here voids the perf arms)"
refuse_phase_unless_gpu_idle_because_a_ladder_can_start_mid_batch gates
GATE_LOG="$LOG_DIR/lora-fold-ab-$STAMP-gates.log"
NVK_LANE="$LANE" NVK_PKG=nv-models NVK_FEATURES=cuda,wgpu \
  "$REPO/scripts/nvk.sh" test --no-fail-fast \
  --test lora_q3d_w4a8_fold --test lora_q3d_wgpu --test q3d_mlp_w4a8 -- --test-threads=1 \
  >"$GATE_LOG" 2>&1 || {
  tail -40 "$GATE_LOG"
  exit 3
}
grep -E "test result" "$GATE_LOG"

run_arm fold_default
run_arm attn_dn_fold NV_Q3D_LORA_GEMV_FOLD_ATTN_DN=1
run_arm fold_off NV_Q3D_LORA_GEMV_FOLD=0
run_arm w4a8_off NV_Q3D_MLP_W4A8=0

E4B_LOG="$LOG_DIR/lora-fold-ab-$STAMP-e4b.log"
echo "=== e4b real-weight adapter decode A/B (control arm, no e4b fold shipped) log=$E4B_LOG"
refuse_phase_unless_gpu_idle_because_a_ladder_can_start_mid_batch arm-e4b
NV_E4B_WGPU_TEST=1 \
  NVK_LANE="$LANE" NVK_PKG=nv-models NVK_FEATURES=cuda,wgpu \
  "$REPO/scripts/nvk.sh" test --release --test lora_e4b_wgpu -- \
  --ignored --nocapture real_e4b_adapter_decode_ab_perfb \
  >"$E4B_LOG" 2>&1 || true
grep -E "LORA-E4B-AB|adapter: rank|panicked" "$E4B_LOG" || tail -20 "$E4B_LOG"

echo "=== expectation ledger (qwen3.8 = 64 layers: 16 full-attn + 48 DN; 2 MLP sites per layer) ==="
echo "fold_default  adapter_decode_passes = base + 384: 128 MLP shrink (w4a8 fold, the shipped"
echo "              default) + 256 two-pass attn/DN sites; the recorded default basis is"
echo "              19.8 base / 22.4 adapter ms/tok at d256"
echo "attn_dn_fold  adapter_decode_passes = base + 256: 128 MLP shrink + 32 attn shrink (qkv+o"
echo "              expands fold into q3w_gemv_fp8_rowscale_lora_fold) + 96 DN shrink (dn_in into"
echo "              the merged DN gemv fold entry, dn_out into the rowscale fold entry)."
echo "              MEASURED 2026-08-29 d256 (logs lora-fold-ab-20260829-101806-*): 23.66 ms/tok"
echo "              vs 23.02 all-folds-off vs the 22.4 default -- the pass cut is a ~1 ms LOSS"
echo "              because every 2-row fp8 workgroup restages h (4x denser than the w4a8 8-row"
echo "              template) and the shrink becomes a critical-path predecessor of the widest"
echo "              gemvs, forfeiting the free overlap the standalone expand enjoyed; opt-in only,"
echo "              re-measure only after a per-site h-reduce stage replaces per-workgroup restaging"
echo "fold_off      adapter_decode_passes = base + 512 (measured 23.02 ms/tok at d256)"
echo "w4a8_off      adapter_decode_passes = base' + 576 with base' = base + 64: every site two-pass"
echo "              plus the adapter-forced silu/dquant defusion; attribution anchor"
echo "MTP serving: real-prefill acceptance parity must hold on every arm (measured 0.605 vs 0.598)"
echo "e4b control: 336 pk shrink/expand passes, measured 10.09 -> 11.19 ms/tok (+10.9%); no e4b fold"
echo "  shipped -- its decode gemvs are the generated sg16 kpre/rowsplit/pk family (5 routed variants"
echo "  + shared w4a16 kernel lane), ~5 entry twins for ~0.14 ms, and the q3d attn/DN measurement"
echo "  above shows the same recipe regressing on 2-row-per-workgroup geometry"
