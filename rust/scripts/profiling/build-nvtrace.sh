#!/bin/sh
set -eu
R="${NVTRACE_ROOF:-$HOME/tmp/roofline}"
set +u
. "$HOME/.cache/cargo-tmp/devenv-__cuda.sh" 2>/dev/null || true
set -u
C="${CUDA_PATH:?set CUDA_PATH to a CUDA 12.x toolkit root}"
g++ -std=c++17 -O2 -fPIC -shared \
  -I"$C/include" \
  -o "$R/libnvtrace.so" \
  "$R/nvtrace.cpp" \
  -L"$C/lib" -lcupti -Wl,-rpath,"$C/lib"
echo "built: $(ls -la "$R/libnvtrace.so")"
