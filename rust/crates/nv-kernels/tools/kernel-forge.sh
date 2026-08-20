#!/usr/bin/env bash
set -euo pipefail

SUITE=kernel_forge_gemv_w4a16
CRATE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUST_DIR="$(cd "$CRATE_DIR/../../.." && pwd)/rust"
NVK="$RUST_DIR/scripts/nvk.sh"

mode=gate
candidates=""
while [ $# -gt 0 ]; do
    case "$1" in
        gate|bench|all) mode="$1" ;;
        -c|--candidates) shift; candidates="$1" ;;
        *) printf 'kernel-forge.sh [gate|bench|all] [-c stem,stem]\n' >&2; exit 2 ;;
    esac
    shift
done

export NVK_LANE="${NVK_LANE:-wkforge}"
export NVK_PKG=nv-kernels
export NVK_FEATURES="${NVK_FEATURES:-wgpu}"

LOG_DIR="${KERNEL_FORGE_LOG_DIR:-${TMPDIR:-/tmp}/kernel-forge}"
mkdir -p "$LOG_DIR"
case "$LOG_DIR" in
    "$RUST_DIR"*) printf 'kernel-forge: log dir %s is inside the source tree\n' "$LOG_DIR" >&2; exit 2 ;;
esac

export NV_KERNEL_FORGE_FAILURE_LOG="${NV_KERNEL_FORGE_FAILURE_LOG:-$LOG_DIR/failures.tsv}"
[ -n "$candidates" ] && export NV_KERNEL_FORGE_CANDIDATES="$candidates"

IDLE_MIB="${KERNEL_FORGE_IDLE_MIB:-500}"
IDLE_TIMEOUT_S="${KERNEL_FORGE_IDLE_TIMEOUT_S:-1200}"

gpu_free_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -n1
}

wait_for_idle_gpu() {
    local waited=0 used
    used="$(gpu_free_mib)"
    until [ -n "$used" ] && [ "$used" -lt "$IDLE_MIB" ]; do
        if [ "$waited" -ge "$IDLE_TIMEOUT_S" ]; then
            printf 'kernel-forge: GPU still at %s MiB after %ss; NOT-RUN\n' "${used:-unknown}" "$waited" >&2
            return 1
        fi
        sleep 15
        waited=$((waited + 15))
        used="$(gpu_free_mib)"
    done
    printf 'kernel-forge: GPU idle at %s MiB after %ss\n' "$used" "$waited"
}

run_suite() {
    local tag="$1"; shift
    local log="$LOG_DIR/$SUITE-$tag-$(date +%s).log"
    printf 'kernel-forge: %s -> %s\n' "$tag" "$log"
    if "$NVK" test --test "$SUITE" "$@" >"$log" 2>&1; then
        grep -E '^\[forge\]|^ +gemv|^ +\S+ +[0-9]+\.[0-9]+ us|^=== |^test result:' "$log" || true
        return 0
    fi
    printf 'kernel-forge: %s FAILED, log %s\n' "$tag" "$log" >&2
    grep -E '^error|panicked at|assertion|^test result:' "$log" || true
    return 1
}

status=0
if [ "$mode" = gate ] || [ "$mode" = all ]; then
    wait_for_idle_gpu || exit 3
    run_suite gate -- --nocapture --test-threads=1 || status=1
fi
if [ "$mode" = bench ] || [ "$mode" = all ]; then
    wait_for_idle_gpu || exit 3
    run_suite bench -- --ignored --nocapture --test-threads=1 || status=1
fi

if [ -s "$NV_KERNEL_FORGE_FAILURE_LOG" ]; then
    printf 'kernel-forge: recycled failures for the next generation attempt (%s):\n' "$NV_KERNEL_FORGE_FAILURE_LOG"
    cat "$NV_KERNEL_FORGE_FAILURE_LOG"
fi
exit $status
