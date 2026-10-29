#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tp=$here/../third-party
cache=$tp/.cache
mkdir -p "$cache"

crates=(
  "wgpu 30.0.0 6d8f4bd44d92da5270f03409dba9f952dab24f128e05d6a554926101d1bf9114"
  "wgpu-hal 30.0.0 cf765132d8d5f50e192e7880464890c13f4e7457aafe8e5466e8174586e9f101"
  "candle-core 0.11.0 5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706"
  "candle-metal-kernels 0.11.0 242e83c6acf639bb273c929d73c67a882bb4dd08a140f121096e19ba2f213d3e"
)
mlx_tag=v0.32.2
forge=${FORGE_URL:-https://git.decent.foundation}
upstream=${THIRD_PARTY_UPSTREAM:-0}
[ "${1:-}" = --upstream ] && upstream=1

crate_url() {
  if [ "${3:-$upstream}" = 1 ]; then
    echo "https://static.crates.io/crates/$1/$1-$2.crate"
  else
    echo "$forge/api/packages/mirrors/generic/$1/$2/$1-$2.crate"
  fi
}

mlx_url() {
  if [ "${2:-$upstream}" = 1 ]; then
    echo "https://raw.githubusercontent.com/ml-explore/mlx/$mlx_tag/$1"
  else
    echo "$forge/mirrors/mlx/raw/tag/$mlx_tag/$1"
  fi
}

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

fetch() {
  local url=$1 dest=$2 want=$3 fallback=$4 got
  if [ -f "$dest" ] && [ "$(sha256 "$dest")" = "$want" ]; then
    return
  fi
  mkdir -p "$(dirname "$dest")"
  if ! curl -fsSL --retry 3 -o "$dest.tmp" "$url"; then
    [ "$url" != "$fallback" ] || exit 1
    echo "fetch: $url is unavailable; trying $fallback (the pinned sha256 still decides)" >&2
    curl -fsSL --retry 3 -o "$dest.tmp" "$fallback"
  fi
  got=$(sha256 "$dest.tmp")
  if [ "$got" != "$want" ]; then
    rm -f "$dest.tmp"
    echo "sha256 mismatch for $url: got $got, want $want" >&2
    exit 1
  fi
  mv "$dest.tmp" "$dest"
}

for entry in "${crates[@]}"; do
  read -r name ver sha <<<"$entry"
  dir=$tp/$name-$ver
  patchfile=$tp/patches/$name-$ver.patch
  stamp="$sha $(sha256 "$patchfile")"
  if [ -f "$dir/.fetch-stamp" ] && [ "$(cat "$dir/.fetch-stamp")" = "$stamp" ]; then
    echo "$name-$ver: up to date"
    continue
  fi
  fetch "$(crate_url "$name" "$ver")" "$cache/$name-$ver.crate" "$sha" "$(crate_url "$name" "$ver" 1)"
  rm -rf "$dir"
  tar -xzf "$cache/$name-$ver.crate" -C "$tp"
  patch -d "$dir" -p1 -s < "$patchfile"
  echo "$stamp" > "$dir/.fetch-stamp"
  echo "$name-$ver: fetched and patched"
done

mlx=$tp/mlx-steel-attn
while read -r sha path; do
  fetch "$(mlx_url "$path")" "$mlx/$path" "$sha" "$(mlx_url "$path" 1)"
done < "$mlx/FILES"
echo "mlx-steel-attn: $(wc -l < "$mlx/FILES") headers verified"
