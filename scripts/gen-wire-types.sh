#!/usr/bin/env bash
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
sp="$(dirname "$here")"
repo_root="$(cd "$sp/../.." && pwd)"
garden="$(cd "$sp/../new-garden" && pwd)"

typeshare_bin="${TYPESHARE_BIN:-}"
if [ -z "$typeshare_bin" ]; then
  typeshare_bin="$(nix build --inputs-from "$repo_root" 'nixpkgs#typeshare' --no-link --print-out-paths)/bin/typeshare"
fi

cfg="$sp/rust/typeshare.toml"
src="$sp/rust/src"

ts_out="$garden/lib/generated/wire/speaches.ts"
kt_out="$garden/android-2/app/src/main/java/garden/decent/rock/wire/SpeachesWire.kt"
go_out="$sp/go/internal/wire/wire.go"

emit_all() {
  "$typeshare_bin" "$src" --lang=typescript --config-file="$cfg" --output-file="$1" 2>/dev/null
  "$typeshare_bin" "$src" --lang=kotlin --config-file="$cfg" \
    --java-package=garden.decent.rock.wire --module-name=speachesWire --output-file="$2" 2>/dev/null
  "$typeshare_bin" "$src" --lang=go --config-file="$cfg" --go-package=wire --output-file="$3" 2>/dev/null
}

if [ "${1:-}" = "--check" ]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  emit_all "$tmp/speaches.ts" "$tmp/SpeachesWire.kt" "$tmp/wire.go"
  fail=0
  for pair in "$ts_out:$tmp/speaches.ts" "$kt_out:$tmp/SpeachesWire.kt" "$go_out:$tmp/wire.go"; do
    committed="${pair%%:*}"
    fresh="${pair##*:}"
    if ! diff -u "$committed" "$fresh"; then
      echo "DRIFT: $committed is stale; run scripts/gen-wire-types.sh" >&2
      fail=1
    fi
  done
  exit "$fail"
fi

mkdir -p "$(dirname "$ts_out")" "$(dirname "$kt_out")" "$(dirname "$go_out")"
emit_all "$ts_out" "$kt_out" "$go_out"
echo "wire types written:"
echo "  $ts_out"
echo "  $kt_out"
echo "  $go_out"
