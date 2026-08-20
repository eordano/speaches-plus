#!/usr/bin/env bash
set -eu

CT2_HELPERS="${CT2_HELPERS_OVERRIDE:-$(echo $HOME/.cargo/registry/src/index.crates.io-*/ct2rs-*/CTranslate2/src/cuda/helpers.h | awk '{print $1}')}"
[ -f "$CT2_HELPERS" ] || {
  echo "helpers.h not found ($CT2_HELPERS); run cargo fetch first" >&2
  exit 1
}
MARKER='// speaches-plus: thrust includes for cuda 12.8+'
grep -qF "$MARKER" "$CT2_HELPERS" && exit 0

chmod u+w "$CT2_HELPERS" 2>/dev/null || true

awk -v marker="$MARKER" '
{ print }
/^#include <cuda_bf16\.h>$/ {
  print marker
  print "#include <thrust/iterator/counting_iterator.h>"
  print "#include <thrust/iterator/permutation_iterator.h>"
  print "#include <thrust/iterator/transform_iterator.h>"
  print "#include <thrust/reduce.h>"
  print "#include <thrust/extrema.h>"
  print "#include <thrust/transform.h>"
  print "#include <thrust/copy.h>"
  print "#include <thrust/fill.h>"
}
' "$CT2_HELPERS" >"$CT2_HELPERS.tmp"

if ! grep -qF "$MARKER" "$CT2_HELPERS.tmp"; then
  rm -f "$CT2_HELPERS.tmp"
  echo "anchor '#include <cuda_bf16.h>' not found in $CT2_HELPERS" >&2
  exit 1
fi

mv "$CT2_HELPERS.tmp" "$CT2_HELPERS"
