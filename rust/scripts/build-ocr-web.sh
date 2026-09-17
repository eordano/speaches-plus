#!/usr/bin/env bash
set -euo pipefail

RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${OCR_WEB_OUT:-$RUST_DIR/../../portal/static/ocr-web}"
LANE="${NVK_LANE:-ocr-web}"

DEVENV=""
for c in "$HOME/.cache/cargo-tmp/devenv-__cuda.sh" \
  "$HOME/.cache/cargo-tmp/devenv-__default.sh" \
  "$HOME/.cache/cargo-tmp/devenv-__cuda_no_models.sh" \
  "$HOME/.cache/cargo-tmp/devenv-__no_models.sh"; do
  if [ -f "$c" ] && grep -oE '/nix/store/[^/ ]+-source/[^ "]*cmake-install-shim.sh' "$c" | head -1 | xargs -r test -e; then
    DEVENV="$c"
    break
  elif [ -f "$c" ]; then
    echo "build-ocr-web: $c bakes a source path that no longer exists; trying the next cached devenv" >&2
  fi
done
if [ -z "$DEVENV" ]; then
  echo "build-ocr-web: no cached devenv under ~/.cache/cargo-tmp -- run scripts/nvk.sh once first" >&2
  exit 1
fi

source "$DEVENV"

export CARGO_TARGET_DIR="$HOME/.cache/cargo-tmp/tgt-$LANE"
export RUSTFLAGS='--cfg getrandom_backend="wasm_js"'

rustc --print target-list | grep -qx wasm32-unknown-unknown || {
  echo "build-ocr-web: $DEVENV rustc has no wasm32-unknown-unknown target" >&2
  exit 1
}

if ! command -v lld >/dev/null; then
  LLD_OUT="$(nix build --no-link --print-out-paths --inputs-from "$RUST_DIR/.." nixpkgs#lld)"
  export PATH="$LLD_OUT/bin:$PATH"
fi
command -v lld >/dev/null || {
  echo "build-ocr-web: the devenv rustc links wasm via PATH lld and none is available" >&2
  exit 1
}

cd "$RUST_DIR"
cargo build -p nv-ocr-web --release --no-default-features --features browser \
  --target wasm32-unknown-unknown

WASM="$CARGO_TARGET_DIR/wasm32-unknown-unknown/release/nv_ocr_web.wasm"
if [ ! -f "$WASM" ]; then
  echo "build-ocr-web: cargo succeeded but $WASM is missing" >&2
  exit 1
fi

WB_VER="$(awk '$0 == "name = \"wasm-bindgen\"" { getline; gsub(/[version ="]/, ""); print; exit }' Cargo.lock)"
if [ -z "$WB_VER" ]; then
  echo "build-ocr-web: wasm-bindgen not found in Cargo.lock" >&2
  exit 1
fi
BINDGEN="nix run --inputs-from $RUST_DIR/.. nixpkgs#wasm-bindgen-cli --"
if [ "$($BINDGEN --version | awk '{print $2}')" != "$WB_VER" ]; then
  CLI_ROOT="$HOME/.cache/cargo-tmp/wasm-bindgen-cli-$WB_VER"
  if [ ! -x "$CLI_ROOT/bin/wasm-bindgen" ]; then
    echo "build-ocr-web: nixpkgs wasm-bindgen-cli != $WB_VER (the schema must match Cargo.lock exactly); cargo-installing $WB_VER once into $CLI_ROOT" >&2
    cargo install wasm-bindgen-cli --version "$WB_VER" --locked --root "$CLI_ROOT"
  fi
  BINDGEN="$CLI_ROOT/bin/wasm-bindgen"
fi

mkdir -p "$OUT_DIR"
$BINDGEN --target web --no-typescript --out-dir "$OUT_DIR" "$WASM"

grep -q 'export function engine_init' "$OUT_DIR/nv_ocr_web.js" || {
  echo "build-ocr-web: bindgen output has no engine_init export -- the crate's wasm surface (src/wasm.rs) did not make it into the module" >&2
  exit 1
}

echo "build-ocr-web: wrote $(ls "$OUT_DIR" | tr '\n' ' ')to $OUT_DIR"
ls -la "$OUT_DIR/nv_ocr_web.js" "$OUT_DIR/nv_ocr_web_bg.wasm"
