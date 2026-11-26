#!/usr/bin/env bash
set -eu
CUDA_13_DROPPED_COMPUTE_53_FROM_CTRANSLATE2_COMMON_SO_AN_ALL_SM_10_PLUS_LIST_FALLS_BACK_TO_THE_FLEETS_8_9="8.9"
found=0
for f in $HOME/.cargo/registry/src/index.crates.io-*/ct2rs-*/build.rs; do
  [ -f "$f" ] || continue
  found=1
  grep -qF 'cmake_tokens.push("Common".to_string());' "$f" || continue
  chmod u+w "$f" 2>/dev/null || true
  sed -i "s|cmake_tokens.push(\"Common\".to_string());|cmake_tokens.push(\"$CUDA_13_DROPPED_COMPUTE_53_FROM_CTRANSLATE2_COMMON_SO_AN_ALL_SM_10_PLUS_LIST_FALLS_BACK_TO_THE_FLEETS_8_9\".to_string());|" "$f"
done
[ "$found" = 1 ] || {
  echo "ct2rs build.rs not found under ~/.cargo/registry; run cargo fetch first" >&2
  exit 1
}
