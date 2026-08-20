#!/usr/bin/env bash
set -euo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
DIR=$(dirname "$SELF")
NVK="$DIR/nvk.sh"

cmd="${1:-}"
shift || true

case "$cmd" in
  train)
    export NV_DETERMINISTIC="${NV_DETERMINISTIC:-1}"
    NVK_PKG=nv-models NVK_FEATURES="${NVK_FEATURES:-}" NVK_LANE="${NVK_LANE:-loratrain}" \
      "$NVK" run --bin nvk-train -- "$@"
    ;;
  check)
    adir="${1:-}"
    if [ -z "$adir" ]; then
      echo "usage: nvk-lora.sh check <adapter-dir>" >&2
      exit 2
    fi
    adir=$(readlink -f "$adir")
    if [ ! -f "$adir/adapter_config.json" ]; then
      echo "no adapter_config.json in $adir" >&2
      exit 2
    fi
    NV_LORA_REAL_ADAPTER_DIR="$adir" \
      NVK_PKG=nv-weights NVK_FEATURES="${NVK_FEATURES:-}" NVK_LANE="${NVK_LANE:-loratrain}" \
      "$NVK" test --test lora_adapter_real -- --ignored --nocapture --test-threads=1
    ;;
  *)
    grep '^#' "$SELF" | sed 's/^# \{0,1\}//' | head -12
    exit 1
    ;;
esac
