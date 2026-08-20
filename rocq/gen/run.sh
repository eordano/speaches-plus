#!/usr/bin/env bash
# Regenerate every generated Rocq module and re-check the whole development.
#
# Stages run in order, cheapest and most hermetic first.  Every stage runs even
# if an earlier one failed, and the run exits non-zero if ANY stage did: a stage
# that cannot run is a finding, not a reason to stop looking at the others.  The
# old `set -e` form stopped at the first failure, which is why the CUDA
# extractors are ordered ahead of gen.py -- gen.py needs GPU-produced
# measurement logs, and a missing log must not be able to hide source drift.
#
#   ./run.sh                 all stages
#   ./run.sh extract build   only those stages
#   ./run.sh --check         extractors verify GenLaunch.v against the CUDA and
#                            write nothing; the form a pre-commit hook wants
set -uo pipefail
cd "$(dirname "$0")"

CHECK=""
STAGES=""
for a in "$@"; do
  case "$a" in
    --check) CHECK="--check" ;;
    anchors | extract | selftest | claims | gen | build) STAGES="$STAGES $a" ;;
    *)
      echo "usage: $0 [--check] [anchors] [extract] [selftest] [claims] [gen] [build]" >&2
      exit 2
      ;;
  esac
done
[ -n "$STAGES" ] || STAGES="anchors extract selftest claims gen build"

rc_anchors=skipped
rc_extract=skipped
rc_selftest=skipped
rc_claims=skipped
rc_gen=skipped
rc_build=skipped
failed=0

run_stage() {
  local name="$1"
  shift
  echo "== stage $name: $*"
  "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "[run] STAGE $name FAILED (rc=$rc)" >&2
    failed=1
  fi
  # A stage that runs more than one command keeps the FAILURE, never the last
  # result: with two extractors under `extract`, a plain assignment let a green
  # second run overwrite a red first one and the summary line reported
  # extract=0 while the run exited 1.
  eval "prev=\$rc_$name"
  if [ "$prev" = "skipped" ] || [ "$rc" -ne 0 ]; then
    eval "rc_$name=$rc"
  fi
}

for s in $STAGES; do
  case "$s" in
    anchors) run_stage anchors python3 check_anchors.py ;;
    extract)
      for x in extract/launch_geometry.py extract/pdl_sites.py; do
        run_stage extract python3 "$x" $CHECK
      done
      ;;
    selftest)
      # ~4.5 minutes: each case re-parses the whole 44-file corpus.  It is in the
      # default list on purpose -- an extractor whose refusal paths are never
      # exercised reports a coverage number nobody can contradict.
      run_stage selftest python3 extract/selftest.py
      run_stage selftest python3 extract/pdl_sites.py --self-test
      ;;
    claims)
      # Expected to be RED while docs/measurements/2026-08-10-rocq-repoint/ is
      # deleted: six paper claims are bounded only by constants whose artifact
      # is gone, and claims.py refuses to compute a bound from the literals
      # transcribed into GenRoofline.v. Unlike gen.py this stage IS covered by
      # --check: its output carries no timestamp.
      run_stage claims python3 claims.py $CHECK
      ;;
    gen)
      if [ -n "$CHECK" ]; then
        # gen.py stamps datetime.now() into its output, so its bytes are not a
        # function of its inputs and --check cannot mean anything for it.  Say
        # so rather than let a green --check imply more coverage than it has.
        echo "[run] --check does NOT cover GenTraffic.v / GenRoofline.v:" \
          "gen.py embeds a wall-clock timestamp, so its output is not" \
          "reproducible byte for byte." >&2
        rc_gen="not-checkable"
      else
        run_stage gen python3 gen.py
      fi
      ;;
    build) run_stage build ../build.sh ;;
  esac
done

echo
echo "[run] stage results: anchors=$rc_anchors extract=$rc_extract selftest=$rc_selftest claims=$rc_claims gen=$rc_gen build=$rc_build"
exit $failed
