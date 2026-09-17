#!/usr/bin/env bash
set -euo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd)
CACHE="${NVK_CACHE:-${XDG_CACHE_HOME:-$HOME/.cache}/cargo-tmp}"

LANE="${NVK_LANE:-base}"
PKG="${NVK_PKG:-nv-kernels}"
if [ -z "${NVK_FEATURES+x}" ]; then
  case "$PKG" in
    nv-kernels) FEATURES="cuda,wgpu" ;;
    *) FEATURES="cuda" ;;
  esac
else
  FEATURES="$NVK_FEATURES"
fi
JOBS="${NVK_JOBS:-8}"

usage() {
  cat <<'EOF'
nvk.sh - build/test wrapper with a cached nix dev env

  rust/scripts/nvk.sh test --test parity_rope -- --nocapture
  NVK_LANE=rope rust/scripts/nvk.sh test --test parity_rope
  NVK_FEATURES=wgpu rust/scripts/nvk.sh test --test wgpu_rope
  NVK_PKG=nv-models rust/scripts/nvk.sh test --release --test laguna_smoke
  NVK_LANE=agent2 rust/scripts/nvk.sh check
  rust/scripts/nvk.sh test --no-run

  gate  = test (correctness only: measurement probes are #[ignore] and stay out)
  probe = test in --release under an exclusive GPU flock (NVK_GPU_LOCK), for the
          ignored measurement ladders: nvk.sh probe --lib 'batch' -- --ignored
          decode_batch_ladder --nocapture. Never run a probe in debug; never run
          two probes concurrently -- the lock enforces the second rule for you.
  lanes = lane housekeeping for the tgt-* target dirs under the cache:
          nvk.sh lanes                        list lanes by size, idle days, live consumers
          nvk.sh lanes --reap DAYS --dry-run  list what a reap would delete, delete nothing
          nvk.sh lanes --reap DAYS            delete lanes idle > DAYS; never tgt-base,
          never the current NVK_LANE, never a lane a live process references (checked via
          pgrep -f on the target-dir path and NVK_LANE= token, plus /proc/*/environ)

Sources a CACHED nix dev environment instead of re-entering `nix develop`.
Re-entering it yields a slightly different env each time, which trips
build.rs's rerun-if-env-changed and recompiles every .cu on EVERY run
(~6 min). The cache is keyed on flake.nix + flake.lock and refreshes
automatically when either changes. It ALSO refreshes when any store path
baked into the cached env has vanished (GC) or a baked ${./script} has
drifted from the working tree -- the key cannot see a repo-layout change
or a GC, and each once left every build failing on a missing
cmake-install-shim.sh.

env:
  NVK_LANE      target-dir suffix under $HOME/.cache/cargo-tmp (default: base)
                each concurrent agent MUST use its own lane
  NVK_PKG       cargo package (default: nv-kernels; use nv-models for laguna tests)
  NVK_FEATURES  cargo features (default: cuda,wgpu for nv-kernels, cuda otherwise;
                empty string for none)
                use "wgpu" to skip every nvcc invocation - far faster edit loop
  NVK_JOBS      cargo -j (default: 8)
  NVK_SHELL     nix devshell attr (default: .#cuda, or .#default when no cuda)
  NVK_REFRESH   set to 1 to force-regenerate the cached dev env
  NVK_VULKAN_LOADER  dir holding libvulkan.so.1 (default: newest in /nix/store);
                needed or every wgpu GPU test SKIPs while still printing "ok"
  NVK_ORT_CUDA  cuda shells only: the devshell pins a CPU-only onnxruntime, so
                after sourcing, ORT_DYLIB_PATH is repointed at a CUDA-EP-capable
                libonnxruntime.so (default: a /nix/store lib of the SAME
                onnxruntime version carrying libonnxruntime_providers_cuda.so).
                Set a dir to pin one, 0 to keep the devshell CPU dylib. GPU
                STT/kokoro/diarization key off this: without the CUDA EP the
                STT default stays CPU whisper
  NVK_CCACHE    1 (default) to route nvcc + cmake CUDA through ccache so
                concurrent lanes share compiled objects; 0 to disable
  NVK_CCACHE_DIR    ccache storage (default: $CACHE/ccache)
  NVK_CCACHE_SIZE   ccache cap (default: 20G; a full nv-kernels + whisper-rs-sys
                + ct2rs build costs ~300 MB)
  NVK_SCCACHE   1 to route rustc (RUSTC_WRAPPER) through sccache; sets
                CARGO_INCREMENTAL=0, so a warm in-lane edit loop keeps
                incremental only with this off. Default 0, and it does NOT
                share rust artifacts across lanes: the cache key includes the
                per-lane target-dir path, so a fresh lane hits 0% (nv-models
                --release: 5m52s vs 5m31s bare -- a net loss). It pays only
                when a lane is rebuilt under the SAME name after a reap
                (100% hits, 4m39s vs 5m31s, ~16% -- the wall is linking,
                build scripts and proc-macro crates, none cacheable).
                nvcc stays on ccache even here: sccache 0.17 cannot parse the
                '#$ compiler-bindir=' line nixpkgs cuda_nvcc's nvcc.profile adds
                to --dryrun output (is_envvar_line_re wants [_A-Z]+=) and dies
                with 'cannot find binary path'
  NVK_SCCACHE_DIR   sccache storage (default: $HOME/build/sccache -- /tank is
                root-owned and root is tmpfs, so the usr-owned build area is the
                only writable pool; capped to keep zroot headroom)
  NVK_SCCACHE_SIZE  sccache cap (default: 20G)
  NVK_SCCACHE_BIN   sccache binary (default: PATH, else newest /nix/store match;
                provision via: nix build --inputs-from . nixpkgs#sccache)
EOF
}

lane_last_used_epoch() {
  python3 - "$1" <<'PYEOF' 2>/dev/null || true
import os, sys
root = sys.argv[1]
best = 0
for cur, dirs, files in os.walk(root):
    depth = os.path.relpath(cur, root).count(os.sep)
    if depth >= 2:
        dirs[:] = []
        continue
    for n in dirs + files:
        try:
            m = os.lstat(os.path.join(cur, n)).st_mtime
        except OSError:
            continue
        if m > best:
            best = m
print(int(best), end="")
PYEOF
}

lane_live_consumers() {
  local dir="$1" lane="$2" pids=""
  pids=$(pgrep -f -- "$dir" 2>/dev/null || true)
  pids="$pids $(pgrep -f "NVK_LANE=${lane}([^A-Za-z0-9._-]|\$)" 2>/dev/null || true)"
  pids="$pids $(grep -slzF -- "$dir" /proc/[0-9]*/environ 2>/dev/null | grep -oE '[0-9]+' || true)"
  printf '%s\n' "$pids" | tr ' ' '\n' | grep -Ex '[0-9]+' | grep -vx "$$" | sort -un | tr '\n' ' ' || true
}

lanes_cmd() {
  local reap_days="" dry=0 now epoch idle size live lane dir doomed="" keep
  while [ $# -gt 0 ]; do
    case "$1" in
      --reap)
        reap_days="${2:?nvk: lanes --reap needs DAYS}"
        shift 2
        ;;
      --dry-run)
        dry=1
        shift
        ;;
      *)
        echo "nvk: unknown lanes arg '$1' (usage: lanes [--reap DAYS [--dry-run]])" >&2
        exit 1
        ;;
    esac
  done
  now=$(date +%s)
  printf '%-14s %10s %9s  %-16s %s\n' LANE SIZE 'IDLE(d)' LAST_USED LIVE_PIDS
  for dir in "$CACHE"/tgt-*; do
    [ -d "$dir" ] || continue
    lane="${dir##*/tgt-}"
    size=$(du -sh -- "$dir" 2>/dev/null | awk '{print $1}' || true)
    epoch=$(lane_last_used_epoch "$dir")
    [ -n "$epoch" ] || epoch=0
    idle=$(((now - epoch) / 86400))
    live=$(lane_live_consumers "$dir" "$lane")
    printf '%-14s %10s %9s  %-16s %s\n' "$lane" "$size" "$idle" \
      "$(date -d "@$epoch" +'%Y-%m-%d %H:%M')" "${live:-}"
    if [ -n "$reap_days" ] && [ "$idle" -gt "$reap_days" ] &&
      [ "$lane" != "base" ] && [ "$lane" != "$LANE" ] && [ -z "$live" ]; then
      doomed="$doomed$dir"$'\n'
    fi
  done
  [ -n "$reap_days" ] || return 0
  if [ -z "$doomed" ]; then
    keep="tgt-base"
    [ "$LANE" != "base" ] && keep="$keep, tgt-$LANE"
    echo "nvk: nothing idle > $reap_days days ($keep and live lanes are never reaped)"
    return 0
  fi
  if [ "$dry" = "1" ]; then
    echo "nvk: DRY RUN -- lanes --reap $reap_days would delete:"
    printf '%s' "$doomed"
    return 0
  fi
  echo "nvk: reaping lanes idle > $reap_days days:"
  printf '%s' "$doomed"
  printf '%s' "$doomed" | xargs -r -d '\n' rm -rf --
}

case "${1:-}" in
  '')
    usage
    exit 1
    ;;
  -h | --help)
    usage
    exit 0
    ;;
  lanes)
    shift
    lanes_cmd "$@"
    exit 0
    ;;
esac

case "$FEATURES" in
  *cuda*) SHELL_ATTR="${NVK_SHELL:-.#cuda}" ;;
  *) SHELL_ATTR="${NVK_SHELL:-.#default}" ;;
esac

SUB="$1"
shift
if [ $# -eq 0 ]; then ARGS=""; else ARGS=$(printf '%q ' "$@"); fi
TGT="$CACHE/tgt-$LANE"

cd "$REPO"

mkdir -p "$CACHE"
SLUG=$(printf '%s' "$SHELL_ATTR" | tr -c 'A-Za-z0-9' '_')
ENVFILE="$CACHE/devenv-$SLUG.sh"
STAMPFILE="$ENVFILE.stamp"
NIXHUG="${NVK_NIXHUG:-$REPO/../nix-hug}"
if [ -d "$NIXHUG" ]; then NIX_ARGS=(--override-input nix-hug "path:$NIXHUG"); else NIX_ARGS=(); fi
STAMP=$({
  cat flake.nix flake.lock 2>/dev/null
  printf '%s' "${NIX_ARGS[*]}"
} | md5sum | cut -d' ' -f1)

STALE_SELF=0
if [ -s "$ENVFILE" ]; then
  # Every store path the env names must still exist: a GC'd one (package or a
  # bare ${./file} like cmake-install-shim.sh, which is NOT a -source/ subpath)
  # fails the build with a confusing "cannot stat" instead of a refresh.
  for p in $(grep -oE "/nix/store/[a-z0-9]{32}-[A-Za-z0-9._+/@~-]*" "$ENVFILE" | grep -v '\.drv$' | sort -u); do
    if [ ! -e "$p" ]; then
      echo "nvk: cached dev env points at a vanished store path ($p); regenerating" >&2
      STALE_SELF=1
      break
    fi
    case "$p" in
      */*-source/*|/nix/store/*-*.sh|/nix/store/*-*.py) ;;
      *) continue ;;
    esac
    base=$(basename "$p")
    for local in "$REPO/rust/scripts/$base" "$REPO/scripts/$base"; do
      if [ -e "$local" ] && ! cmp -s "$p" "$local"; then
        echo "nvk: cached dev env bakes a stale $base (differs from $local); regenerating" >&2
        STALE_SELF=1
        break 2
      fi
    done
  done
fi

if [ "${NVK_REFRESH:-0}" = "1" ] || [ ! -s "$ENVFILE" ] || [ "$STALE_SELF" = "1" ] || [ "$(cat "$STAMPFILE" 2>/dev/null || true)" != "$STAMP" ]; then
  echo "nvk: regenerating dev env for $SHELL_ATTR (this takes a minute)" >&2
  TMP=$(mktemp "$ENVFILE.XXXXXX")
  trap 'rm -f "$TMP"' EXIT INT TERM HUP
  nix print-dev-env "$SHELL_ATTR" "${NIX_ARGS[@]}" >"$TMP"
  mv -f "$TMP" "$ENVFILE"
  trap - EXIT INT TERM HUP
  printf '%s\n' "$STAMP" >"$STAMPFILE"
fi

CALLER_MODEL_ENV_WINS_OVER_CACHED_DEVENV="NV_CHAT_MODEL_DIR NV_CHAT_MODEL_DIRS NV_WGPU_CHAT_MODEL_DIRS HF_HUB_CACHE"
for v in $CALLER_MODEL_ENV_WINS_OVER_CACHED_DEVENV; do
  if [ -n "${!v+x}" ]; then eval "NVK_CALLER_$v=\${$v}"; fi
done

set +u
# shellcheck disable=SC1090
. "$ENVFILE"
set -u

for v in $CALLER_MODEL_ENV_WINS_OVER_CACHED_DEVENV; do
  saved="NVK_CALLER_$v"
  if [ -n "${!saved+x}" ]; then eval "export $v=\${$saved}"; fi
done

export TMPDIR="$TGT/tmp"
mkdir -p "$TMPDIR"
export CARGO_TARGET_DIR="$TGT"
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
export CUDA_ARCH_LIST="${NVK_CUDA_ARCH:-12.0}"
CMAKE_CUDA_ARCHITECTURES=$(printf '%s' "$CUDA_ARCH_LIST" | tr -d '.')
export CMAKE_CUDA_ARCHITECTURES
export NV_KERNELS_PARITY_REQUIRE="${NV_KERNELS_PARITY_REQUIRE:-1}"

if [ -z "${VK_ICD_FILENAMES:-}" ] && [ -e /run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
fi
VKLOADER="${NVK_VULKAN_LOADER:-$(printf '%s\n' /nix/store/*-vulkan-loader-*/lib | sort | tail -1 || true)}"
if [ -n "$VKLOADER" ] && [ -e "$VKLOADER/libvulkan.so.1" ]; then
  export LD_LIBRARY_PATH="$VKLOADER${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if [ "${NVK_ORT_CUDA:-1}" != "0" ] && [ -n "${ORT_DYLIB_PATH:-}" ]; then
  case "$SHELL_ATTR" in
    *cuda*)
      if [ ! -e "$(dirname "$ORT_DYLIB_PATH")/libonnxruntime_providers_cuda.so" ]; then
        ORTCUDA="${NVK_ORT_CUDA:-1}"
        if [ "$ORTCUDA" = "1" ]; then
          ORTNAME=$(basename "$(dirname "$(dirname "$ORT_DYLIB_PATH")")")
          ORTNAME="${ORTNAME#*-}"
          ORTCUDA=$(for d in /nix/store/*-"$ORTNAME"/lib; do
            if [ -e "$d/libonnxruntime_providers_cuda.so" ] && [ -e "$d/libonnxruntime.so" ]; then
              printf '%s\n' "$d"
            fi
          done | sort | tail -1 || true)
        fi
        if [ -n "$ORTCUDA" ] && [ -e "$ORTCUDA/libonnxruntime.so" ]; then
          export ORT_DYLIB_PATH="$ORTCUDA/libonnxruntime.so"
          echo "nvk: devshell onnxruntime lacks the CUDA EP; ORT_DYLIB_PATH -> $ORT_DYLIB_PATH (NVK_ORT_CUDA=0 restores the CPU dylib)" >&2
        fi
      fi
      ;;
  esac
fi

if [ "${NVK_SCCACHE:-0}" != "0" ]; then
  SCCACHE_BIN="${NVK_SCCACHE_BIN:-$(command -v sccache 2>/dev/null || true)}"
  if [ -z "$SCCACHE_BIN" ]; then
    SCCACHE_BIN=$(printf '%s\n' /nix/store/*-sccache-*/bin/sccache | sort | awk 'END{print}' || true)
  fi
  if [ -n "$SCCACHE_BIN" ] && [ -x "$SCCACHE_BIN" ]; then
    export SCCACHE_DIR="${NVK_SCCACHE_DIR:-$HOME/build/sccache}"
    export SCCACHE_CACHE_SIZE="${NVK_SCCACHE_SIZE:-20G}"
    mkdir -p "$SCCACHE_DIR"
    export RUSTC_WRAPPER="$SCCACHE_BIN"
    export CARGO_INCREMENTAL=0
  else
    echo "nvk: NVK_SCCACHE=1 but no sccache binary found (provision: nix build --inputs-from . nixpkgs#sccache); rustc uncached" >&2
  fi
fi

if [ "${NVK_CCACHE:-1}" != "0" ]; then
  CCACHE_BIN="${NVK_CCACHE_BIN:-$(command -v ccache 2>/dev/null || true)}"
  if [ -n "$CCACHE_BIN" ]; then
    export NVK_NVCC_WRAPPER="$CCACHE_BIN"
    export CCACHE_DIR="${NVK_CCACHE_DIR:-$CACHE/ccache}"
    export CCACHE_MAXSIZE="${NVK_CCACHE_SIZE:-20G}"
    if [ "${NVK_CCACHE_BASEDIR:-1}" != "0" ]; then
      export CCACHE_BASEDIR="$REPO"
    fi
    export CMAKE_CUDA_COMPILER_LAUNCHER="$CCACHE_BIN"
    export CMAKE_C_COMPILER_LAUNCHER="$CCACHE_BIN"
    export CMAKE_CXX_COMPILER_LAUNCHER="$CCACHE_BIN"
    mkdir -p "$CCACHE_DIR"
  fi
else
  unset NVK_NVCC_WRAPPER
fi

cd rust
LOCKWRAP=""
case "$SUB" in
  gate)
    SUB='test'
    ;;
  probe)
    SUB='test'
    ARGS="--release $ARGS"
    GPU_LOCK="${NVK_GPU_LOCK:-/tmp/claude-$(id -u)/nvk-gpu0.lock}"
    mkdir -p "$(dirname "$GPU_LOCK")"
    if command -v flock >/dev/null 2>&1; then
      LOCKWRAP="flock $GPU_LOCK"
    else
      LOCKWRAP="python3 $REPO/rust/scripts/gpu-flock.py $GPU_LOCK"
    fi
    echo "nvk: probe mode -- release profile, exclusive GPU lock $GPU_LOCK (waits if another probe holds it)" >&2
    PROBE_PREBUILD=1
    ;;
esac
if [ -n "${PROBE_PREBUILD:-}" ]; then
  PRE_ARGS="${ARGS%% -- *}"
  PRE_ARGS="${PRE_ARGS% --}"
  echo "nvk: probe prebuild -- compiling outside the GPU lock" >&2
  if [ -n "$FEATURES" ]; then
    eval "cargo $SUB -j $JOBS -p $PKG --features $FEATURES $PRE_ARGS --no-run" || exit $?
  else
    eval "cargo $SUB -j $JOBS -p $PKG $PRE_ARGS --no-run" || exit $?
  fi
fi
if [ -n "$FEATURES" ]; then
  eval "exec $LOCKWRAP cargo $SUB -j $JOBS -p $PKG --features $FEATURES $ARGS"
else
  eval "exec $LOCKWRAP cargo $SUB -j $JOBS -p $PKG $ARGS"
fi
