#!/usr/bin/env bash
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
      run_stage selftest python3 extract/selftest.py
      run_stage selftest python3 extract/pdl_sites.py --self-test
      ;;
    claims)
      run_stage claims python3 claims.py $CHECK
      ;;
    gen)
      if [ -n "$CHECK" ]; then
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
