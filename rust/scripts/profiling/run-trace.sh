#!/bin/sh
set -u
R="${NVTRACE_ROOF:-$HOME/tmp/roofline}"
BIN="${NVTRACE_BIN:-$HOME/.cache/cargo-tmp/tgt-roof/release/ocr-bsn-bench}"
PAGES=${PAGES:-$HOME/tmp/ocr-round/final/pages20.txt}
ITERS=${ITERS:-64}
set +u
. "$HOME/.cache/cargo-tmp/devenv-__cuda.sh" 2>/dev/null || true
set -u
export LD_LIBRARY_PATH="/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TMPDIR="$HOME/.cache/cargo-tmp"

occ() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; }

echo "=== occupancy at start: $(occ)"

run_arm() {
  tag=$1
  batch=$2
  trace=$3
  echo "### ARM tag=$tag NV_DSOCR_ATTN_BATCH=$batch trace=$trace $(date -Is)"
  if [ "$trace" = "1" ]; then
    env NV_DSOCR_ATTN_BATCH="$batch" \
      CUDA_INJECTION64_PATH="$R/libnvtrace.so" NVTRACE_OUT="$R/nvtrace-$tag.tsv" \
      "$BIN" --micro-step 1,8 --micro-iters "$ITERS" --pages-from "$PAGES" \
      --json "$R/micro-$tag.json" >"$R/$tag.out" 2>"$R/$tag.err"
  else
    env NV_DSOCR_ATTN_BATCH="$batch" \
      "$BIN" --micro-step 1,8 --micro-iters "$ITERS" --pages-from "$PAGES" \
      --json "$R/micro-$tag.json" >"$R/$tag.out" 2>"$R/$tag.err"
  fi
  echo "rc=$?"
  grep -hE '^micro' "$R/$tag.out" 2>/dev/null
  grep -hE 'nvtrace|cuda-graph captured|\[micro\]' "$R/$tag.err" 2>/dev/null | head -n 12
  echo "--- occupancy after $tag: $(occ)"
}

run_arm base-untraced 0 0
run_arm batch-untraced 1 0
run_arm base-traced 0 1
run_arm batch-traced 1 1

echo "=== occupancy at arm end: $(occ)"
ls -la "$R"/nvtrace-*.tsv 2>/dev/null
