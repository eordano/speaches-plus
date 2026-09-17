#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HOME/tmp/nvktrace}
mkdir -p "$OUT"
CUPTI_INC=$(ls -d /nix/store/*cuda_cupti*-include/include | head -1)
CUPTI_LIB=$(ls -d /nix/store/*cuda_cupti*-lib/lib | head -1)
. "$HOME/.cache/cargo-tmp/devenv-__cuda.sh" >/dev/null 2>&1 || true
g++ -shared -fPIC -O2 -std=c++17 -I"$CUPTI_INC" "$HERE/nvktrace.cc" -o "$OUT/libnvktrace.so" -L"$CUPTI_LIB" -lcupti -Wl,-rpath,"$CUPTI_LIB"
echo "$OUT/libnvktrace.so"
echo "usage: LD_PRELOAD=$OUT/libnvktrace.so <binary>   (kernel time by name to stderr at exit; no perf-counter permission needed)"
