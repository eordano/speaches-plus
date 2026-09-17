#!/usr/bin/env bash
set -euo pipefail

check=0
if [ "${1:-}" = "--check" ]; then
  check=1
fi

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

[ -n "$(find "$out_dir" -name '*.ts' -newer "$stamp" -print -quit)" ] ||
  {
    echo "no bindings regenerated — the export_bindings test filter matched nothing" >&2
    rm -f "$stamp"
    exit 1
  }
find "$out_dir" -name '*.ts' ! -newer "$stamp" -delete
rm -f "$stamp"

if [ "$check" = 1 ]; then
  drift=0
  git diff --exit-code -- "$out_dir" || drift=1
  untracked="$(git ls-files --others --exclude-standard -- "$out_dir")"
  if [ -n "$untracked" ]; then
    echo "untracked generated bindings:" >&2
    echo "$untracked" >&2
    drift=1
  fi
  if [ "$drift" = 1 ]; then
    echo "DRIFT: studio ts-rs bindings are stale; run studio/scripts/gen-types.sh and commit" >&2
    exit 1
  fi
  echo "studio ts-rs bindings are current"
else
  echo "bindings written to $out_dir"
fi
