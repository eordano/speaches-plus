#!/usr/bin/env bash
set -e

real="${REAL_CMAKE:?REAL_CMAKE not set}"

if [ -z "$CMAKE_ROOT" ]; then
  cmake_prefix="${real%/bin/cmake}"
  for d in "$cmake_prefix"/share/cmake-*; do
    if [ -d "$d" ]; then
      export CMAKE_ROOT="$d"
      break
    fi
  done
fi

is_configure=true
for a in "$@"; do
  case "$a" in
    --build | -P | --install | -E) is_configure=false ;;
  esac
done

if ! $is_configure; then
  exec "$real" "$@"
fi

cli_prefix=""
for a in "$@"; do
  case "$a" in
    -DCMAKE_INSTALL_PREFIX=*) cli_prefix="${a#-DCMAKE_INSTALL_PREFIX=}" ;;
  esac
done

"$real" "$@"

prefix="$cli_prefix"
if [ -z "$prefix" ]; then
  cache="$PWD/CMakeCache.txt"
  if [ -f "$cache" ]; then
    prefix="$(awk -F= '/^CMAKE_INSTALL_PREFIX:PATH=/ { sub(/^CMAKE_INSTALL_PREFIX:PATH=/, ""); print; exit }' "$cache")"
  fi
fi
if [ "$prefix" = "/usr/local" ] || [ -z "$prefix" ]; then
  case "$PWD" in
    */out/build) prefix="${PWD%/build}" ;;
  esac
fi

if [ -n "$prefix" ] && [ "$prefix" != "/usr/local" ]; then
  while IFS= read -r f; do
    if ! head -3 "$f" | grep -q "speaches-cmake-shim-prefix"; then
      tmp="$(mktemp)"
      {
        printf '# speaches-cmake-shim-prefix\n'
        printf 'set(CMAKE_INSTALL_PREFIX "%s")\n' "$prefix"
        cat "$f"
      } >"$tmp"
      mv "$tmp" "$f"
    fi
  done < <(find "$PWD" -name cmake_install.cmake -type f)
fi
