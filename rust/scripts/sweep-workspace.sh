#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

PACKAGES_WITHOUT_A_CUDA_FEATURE="nv-config nv-grammar nv-imgdec nv-lookup nv-ocr nv-ocr-web nv-punkt nv-tokenizer"
THE_ONLY_PACKAGE_THAT_ALSO_NEEDS_WGPU="nv-kernels"
PACKAGES_WITH_A_WGPU_FEATURE="speaches-plus nv-kernels nv-layers nv-models nv-specdecode"
BACKEND="${NVK_SWEEP_BACKEND:-cuda}"
case "$BACKEND" in cuda|wgpu) ;; *) echo "NVK_SWEEP_BACKEND must be cuda or wgpu, got '$BACKEND'" >&2; exit 2 ;; esac
LANE="${NVK_LANE:-sweep-$BACKEND}"
CACHE="${NVK_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/cargo-tmp}"
mkdir -p "$CACHE/sweep-logs" || exit 1
LOG_DIR=$(mktemp -d "$CACHE/sweep-logs/$LANE.XXXXXX") || exit 1
echo "sweep backend: $BACKEND; logs: $LOG_DIR"

pass_total=0
fail_total=0
ignored_total=0
broken=""

for manifest in Cargo.toml crates/*/Cargo.toml; do
  if [ "$manifest" = Cargo.toml ]; then
    crate="speaches-plus"
  else
    crate=$(basename "$(dirname "$manifest")")
  fi
  if [ "$BACKEND" = wgpu ]; then
    features=""
    case " $PACKAGES_WITH_A_WGPU_FEATURE " in *" $crate "*) features="wgpu" ;; esac
  else
    features="cuda"
    case " $PACKAGES_WITHOUT_A_CUDA_FEATURE " in *" $crate "*) features="" ;; esac
    [ "$crate" = "$THE_ONLY_PACKAGE_THAT_ALSO_NEEDS_WGPU" ] && features="cuda,wgpu"
  fi
  synthetic_batch_gates=()
  if [ "$crate" = nv-models ]; then
    if [ "$BACKEND" = wgpu ]; then
      synthetic_batch_gates=(NV_G4BD=1 NV_G4D_CHUNKED_PREFILL=1 NV_G4D_HOST_SHAPE=1 NV_G4PFW=1)
    else
      synthetic_batch_gates=(NV_Q38_BATCH_TEST=1 NV_LAGBAT_TEST=1)
    fi
  fi

  log="$LOG_DIR/$crate.log"
  if env "${synthetic_batch_gates[@]}" NVK_LANE="$LANE" NVK_PKG="$crate" NVK_FEATURES="$features" \
    ./scripts/nvk.sh test --tests --no-fail-fast -- --nocapture --test-threads=1 >"$log" 2>&1; then
    status=0
  else
    status=$?
  fi
  read -r summaries p f ignored <<<"$(awk '/^test result: (ok|FAILED)\./ {
    n++; p+=$4; f+=$6; i+=$8
  } END {print n+0, p+0, f+0, i+0}' "$log")"
  pass_total=$((pass_total + p))
  fail_total=$((fail_total + f))
  ignored_total=$((ignored_total + ignored))
  printf '%-16s %5d passed %3d failed %3d ignored; exit=%d; log=%s' \
    "$crate" "$p" "$f" "$ignored" "$status" "$log"
  if [ "$status" -ne 0 ] || [ "$f" -gt 0 ]; then
    printf ' FAILED\n'
    broken="$broken $crate"
  elif [ "$summaries" -eq 0 ]; then
    printf ' NO RESULT SUMMARY\n'
    broken="$broken $crate"
  elif [ "$p" -eq 0 ]; then
    printf ' NO TESTS EXECUTED\n'
  else
    printf '\n'
  fi
done

echo
echo "workspace: $pass_total passed, $fail_total failed, $ignored_total ignored (reported by test harnesses)"
if [ -n "$broken" ]; then
  echo "NOT GREEN:$broken"
  exit 1
fi
