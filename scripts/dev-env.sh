# shellcheck shell=bash

_sp_profile="${1:-cuda}"
_sp_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
_sp_cache="$_sp_root/.direnv/devenv-$_sp_profile.env"
mkdir -p "$_sp_root/.direnv"

if [ ! -s "$_sp_cache" ] ||
  [ "$_sp_root/flake.lock" -nt "$_sp_cache" ] ||
  [ "$_sp_root/flake.nix" -nt "$_sp_cache" ]; then
  echo "dev-env: building '$_sp_profile' snapshot (first run or flake changed)..." >&2
  if ! nix print-dev-env "$_sp_root#$_sp_profile" >"$_sp_cache.tmp"; then
    echo "dev-env: 'nix print-dev-env $_sp_root#$_sp_profile' failed" >&2
    rm -f "$_sp_cache.tmp"
    return 1 2>/dev/null || exit 1
  fi
  mv "$_sp_cache.tmp" "$_sp_cache"
fi

# shellcheck disable=SC1090
. "$_sp_cache"
echo "dev-env: '$_sp_profile' ready -- run cargo/go/python directly (incremental)." >&2
