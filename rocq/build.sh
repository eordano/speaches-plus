#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -z "${COQBIN:-}" ] && command -v coqc >/dev/null 2>&1; then
  COQBIN="$(dirname "$(command -v coqc)")"
fi
if [ -z "${COQBIN:-}" ]; then
  echo "== realizing coq from nixpkgs (no COQBIN, none on PATH)"
  COQBIN="$(nix build --impure --expr 'with import <nixpkgs> {}; coq' \
    --no-link --print-out-paths)/bin"
fi
if [ -z "${ROCQPATH:-}" ]; then
  echo "== realizing rocq stdlib from nixpkgs"
  ROCQPATH="$(nix build --impure --expr 'with import <nixpkgs> {}; coqPackages.stdlib' \
    --no-link --print-out-paths)/lib/coq/9.1/user-contrib"
fi
export ROCQPATH
echo "== toolchain: COQBIN=$COQBIN"

# Order is load-bearing: KvBudget needs WindowClamp, StreamK needs Roofline,
# GenRoofline needs Roofline + GenTraffic. Do not sort or glob this list.
MODS="WindowClamp KvBudget KvBudgetMerged AcceptSoundness Roofline StreamK ChunkedPrefill RoPE PdlOrder GenPdl PdlKernels LaunchGeometry GenLaunch GenClaims"
if [ -f GenTraffic.v ]; then MODS="$MODS GenTraffic GenRoofline"; fi

# A build list that cannot notice a new module is a gate that silently shrinks.
# RoPE.v sat in the tree from 2026-08-10 with a .vo built by hand and was never
# re-checked here, because it was absent from this list; PdlOrder.v hit the same
# hole on 2026-08-11 and reported rc=0 having compiled nothing. Fail loudly
# instead.
missing=""
for f in *.v; do
  case " $MODS " in
    *" ${f%.v} "*) ;;
    *) missing="$missing ${f%.v}" ;;
  esac
done
if [ -n "$missing" ]; then
  echo "BUILD-LIST-INCOMPLETE: .v files not in MODS:$missing" >&2
  echo "Add them to MODS in dependency order (and to _CoqProject)." >&2
  exit 1
fi

for f in $MODS; do
  echo "== coqc $f.v"
  "$COQBIN/coqc" -R . SpeachesPlus "$f.v"
done

echo "== rocqchk (independent kernel re-check)"
CHK=""
for f in $MODS; do CHK="$CHK SpeachesPlus.$f"; done
"$COQBIN/rocqchk" -silent -R . SpeachesPlus $CHK

echo "ALL CHECKED"
