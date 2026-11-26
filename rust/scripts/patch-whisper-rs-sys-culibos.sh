#!/usr/bin/env bash
set -eu

if [ -n "${WHISPER_RS_SYS_BUILD_RS_OVERRIDE:-}" ]; then
  set -- "$WHISPER_RS_SYS_BUILD_RS_OVERRIDE"
else
  set -- $HOME/.cargo/registry/src/index.crates.io-*/whisper-rs-sys-*/build.rs
fi

for build_rs in "$@"; do
  [ -f "$build_rs" ] || {
    echo "whisper-rs-sys build.rs not found ($build_rs); run cargo fetch first" >&2
    exit 1
  }
  grep -qF 'cargo:rustc-link-lib=culibos' "$build_rs" || continue
  chmod u+w "$build_rs" 2>/dev/null || true
  sed -i '/cargo:rustc-link-lib=culibos/d' "$build_rs"
done
