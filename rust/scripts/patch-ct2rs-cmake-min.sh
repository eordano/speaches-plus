#!/usr/bin/env bash
set -eu

ROOT="$(echo $HOME/.cargo/registry/src/index.crates.io-*/ct2rs-*/CTranslate2/third_party 2>/dev/null | awk '{print $1}')"
[ -d "$ROOT" ] || {
  echo "ct2rs third_party not found ($ROOT); run cargo fetch first" >&2
  exit 1
}

while IFS= read -r f; do
  if grep -qE "cmake_minimum_required\(VERSION [0-2]\." "$f" 2>/dev/null ||
    grep -qE "cmake_minimum_required\(VERSION 3\.[01234]([^0-9]|$)" "$f" 2>/dev/null; then
    chmod u+w "$f" 2>/dev/null || true
    sed -i -E 's/cmake_minimum_required\(VERSION [0-9]+\.[0-9]+(\.[0-9]+)?( FATAL_ERROR)?\)/cmake_minimum_required(VERSION 3.5)/' "$f"
  fi
done < <(find "$ROOT" -name CMakeLists.txt -type f)
