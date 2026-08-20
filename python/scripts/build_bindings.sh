#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"

die() {
  printf 'build_bindings: %s\n' "$*" >&2
  exit 1
}

probe_header() {
  local header="$1"
  local hint_dir="${2:-}"
  if [ -n "$hint_dir" ] && [ -f "$hint_dir/$header" ]; then
    return 0
  fi
  for d in /usr/include /usr/local/include /opt/homebrew/include; do
    [ -f "$d/$header" ] && return 0
  done
  return 1
}

if ! probe_header "ctranslate2/translator.h" "${CT2_INCLUDE_DIR:-}"; then
  die "ctranslate2/translator.h not found. Enter \`nix develop\` first, or set CT2_INCLUDE_DIR / CT2_LIBRARY_DIR."
fi

if ! probe_header "whisper.h" "${WHISPER_INCLUDE_DIR:-}"; then
  die "whisper.h not found. Enter \`nix develop\` first, or set WHISPER_INCLUDE_DIR / WHISPER_LIBRARY_DIR."
fi

build_one() {
  local dir="$1"
  local mod="$2"
  [ -d "$dir" ] || die "directory $dir does not exist"
  [ -f "$dir/setup.py" ] || die "$dir/setup.py missing -- nothing to build"
  printf '\n=== building %s (%s) ===\n' "$dir" "$mod"
  (
    cd "$dir"
    "$PYTHON" setup.py build_ext --inplace
  )
  local so_glob
  so_glob=$(find "$dir" -maxdepth 1 -name "${mod}*.so" -print -quit)
  [ -n "$so_glob" ] || die "no ${mod}*.so produced under $dir"
  printf '  built: %s\n' "$so_glob"
}

build_one "ct2_bindings" "_ct2"
build_one "whisper_bindings" "_whisper"

printf '\n=== import smoke test ===\n'
"$PYTHON" - <<'PY'
import importlib
import sys

failures = []
for modname in ("ct2_bindings._ct2", "whisper_bindings._whisper"):
    try:
        importlib.import_module(modname)
        print(f"  OK  import {modname}")
    except Exception as exc:
        failures.append((modname, exc))
        print(f"  FAIL import {modname}: {exc!r}")

if failures:
    sys.exit(1)
PY

printf '\nbuild_bindings: all extensions built and import-clean.\n'
