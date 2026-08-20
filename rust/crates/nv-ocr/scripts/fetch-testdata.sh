#!/usr/bin/env bash
set -euo pipefail

DEST="${NV_OCR_TESSDATA:-$HOME/.cache/ocr-testdata}"
BEST_SHA=8280aed0782fe27257a68ea10fe7ef324ca0f8d85bd2fd145d1c2b560bcb66ba
FAST_SHA=7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2

fetch() {
  local repo=$1 sha=$2 out="$DEST/$1/eng.traineddata"
  if [[ -f $out ]] && echo "$sha  $out" | sha256sum -c --quiet - 2>/dev/null; then
    echo "ok: $out"
    return
  fi
  mkdir -p "$DEST/$repo"
  curl -fL --retry 3 -o "$out.tmp" \
    "https://github.com/tesseract-ocr/$repo/raw/main/eng.traineddata"
  echo "$sha  $out.tmp" | sha256sum -c --quiet -
  mv "$out.tmp" "$out"
  echo "fetched: $out"
}

fetch tessdata_best "$BEST_SHA"
fetch tessdata_fast "$FAST_SHA"
