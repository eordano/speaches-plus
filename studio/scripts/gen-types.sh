#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
out_dir="$here/../src/api/generated"
rust_dir="$here/../../rust"
mkdir -p "$out_dir"
out_dir="$(cd "$out_dir" && pwd)"
stamp="$(mktemp)"

devenv="${SPEACHES_DEVENV:-}"
if [ -z "$devenv" ]; then
  devenv="$(mktemp)"
  trap 'rm -f "$devenv"' EXIT
  (cd "$rust_dir/.." && nix print-dev-env --offline .#no-models >"$devenv")
fi

cd "$rust_dir"
# shellcheck disable=SC1090
source "$devenv"
sdk="$(/usr/bin/xcrun --show-sdk-path 2>/dev/null || true)"
if [ -n "$sdk" ]; then
  export LIBRARY_PATH="$sdk/usr/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
export TS_RS_EXPORT_DIR="$out_dir"
export TS_RS_LARGE_INT=number

cargo test -p speaches-plus --lib --features ts-bindings export_bindings
cargo test -p nv-ocr --lib --features ts-bindings export_bindings

find "$out_dir" -name '*.ts' -newer "$stamp" | grep -q . \
  || { echo "no bindings regenerated — the export_bindings test filter matched nothing" >&2; rm -f "$stamp"; exit 1; }
find "$out_dir" -name '*.ts' ! -newer "$stamp" -delete
rm -f "$stamp"
echo "bindings written to $out_dir"
