#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."

PACKAGES_WITHOUT_A_CUDA_FEATURE="nv-config nv-grammar nv-lookup nv-ocr nv-punkt nv-tokenizer nv-train"
THE_ONLY_PACKAGE_THAT_ALSO_NEEDS_WGPU="nv-kernels"
LANE="${NVK_LANE:-sweep}"

pass_total=0
fail_total=0
broken=""

for manifest in crates/*/Cargo.toml; do
    crate=$(basename "$(dirname "$manifest")")
    features="cuda"
    case " $PACKAGES_WITHOUT_A_CUDA_FEATURE " in *" $crate "*) features="" ;; esac
    [ "$crate" = "$THE_ONLY_PACKAGE_THAT_ALSO_NEEDS_WGPU" ] && features="cuda,wgpu"

    out=$(NVK_LANE="$LANE" NVK_PKG="$crate" NVK_FEATURES="$features" \
        ./scripts/nvk.sh test --tests --no-fail-fast 2>&1)
    read -r p f <<<"$(grep -E '^test result:' <<<"$out" |
        awk '{p+=$4; f+=$6} END {print p+0, f+0}')"

    if grep -qE '^error\[|^error: could not compile|^error: the package' <<<"$out"; then
        printf '%-16s COMPILE FAILURE\n' "$crate"
        grep -E '^error' <<<"$out" | head -2 | sed 's/^/    /'
        broken="$broken $crate"
        continue
    fi
    printf '%-16s %5d passed %3d failed%s\n' "$crate" "$p" "$f" \
        "$( [ "$p" -eq 0 ] && [ "$f" -eq 0 ] && echo '   <- declares no tests' )"
    pass_total=$((pass_total + p))
    fail_total=$((fail_total + f))
    [ "$f" -gt 0 ] && broken="$broken $crate"
done

echo
echo "workspace: $pass_total passed, $fail_total failed"
if [ -n "$broken" ]; then
    echo "NOT GREEN:$broken"
    exit 1
fi
