#!/usr/bin/env bash
set -uo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd)
NVK="$(dirname "$SELF")/nvk.sh"
LANE="${PARITY_AUDIT_LANE:-parity}"
OUT="${PARITY_AUDIT_LOGDIR:-$HOME/.cache/agent-logs/parity-audit}"
mkdir -p "$OUT"
cd "$REPO"

CORPUS_RULE="What this audit ratchets is task #50's cross-backend parity coverage, and membership \
is a DECISION RECORDED IN THIS LIST -- not a filename convention, and not 'every suite with an \
independent oracle'. Two facts force that. The name was already a bad proxy in both directions: \
parity_verify_fused.rs is cfg(cuda)-only and holds CUDA to a host oracle with no second backend \
in sight, while gemv_w4a16_cpu_ref.rs, marlin_w4a16_cpu_ref.rs and wgpu_flash_attn_fp8_oracle.rs \
do the same kind of work and are deliberately OUTSIDE this ratchet. And 409372806 swapped one \
member for another -- parity_quantize_nvfp4_bf16.rs out, wgpu_quantize_nvfp4_bf16_cpu_ref.rs in, \
same kernel, same commit -- which a parity_*.rs glob cannot see, so the audit silently ran four \
suites against a floor written for five. Widening the ratchet to the other oracle suites is a \
real option and someone has to price it; it is not something a glob should decide."

BITEXACT_REFERENCE_GATES=(
  rust/crates/nv-kernels/tests/parity_gdn.rs
  rust/crates/nv-kernels/tests/parity_gemv_bf16_i8.rs
  rust/crates/nv-kernels/tests/parity_kv_fp8_paged.rs
  rust/crates/nv-kernels/tests/parity_verify_fused.rs
  rust/crates/nv-kernels/tests/wgpu_quantize_nvfp4_bf16_cpu_ref.rs
)

EXECUTABLE_TESTS_IN_THE_LISTED_SUITES=63

vanished=()
for f in "${BITEXACT_REFERENCE_GATES[@]}"; do
  [ -f "$f" ] || vanished+=("$f")
done
if [ ${#vanished[@]} -ne 0 ]; then
  echo "parity audit FAILED: ${#vanished[@]} named suite(s) are gone from the tree:" >&2
  printf '  %s\n' "${vanished[@]}" >&2
  echo "$CORPUS_RULE" >&2
  echo "Membership is a list here, not a glob, so a deleted suite is named rather than silently \
subtracted from a count. If the removal was deliberate, drop it from BITEXACT_REFERENCE_GATES and \
lower EXECUTABLE_TESTS_IN_THE_LISTED_SUITES in the same commit, and say why." >&2
  exit 1
fi

unlisted=()
for f in rust/crates/nv-kernels/tests/parity_*.rs; do
  case " ${BITEXACT_REFERENCE_GATES[*]} " in
    *" $f "*) ;;
    *) unlisted+=("$f") ;;
  esac
done
if [ ${#unlisted[@]} -ne 0 ]; then
  echo "parity audit FAILED: ${#unlisted[@]} parity_*.rs suite(s) exist but are not audited:" >&2
  printf '  %s\n' "${unlisted[@]}" >&2
  echo "$CORPUS_RULE" >&2
  echo "Naming a suite parity_* is a claim that it belongs to this ratchet, so adding it to \
BITEXACT_REFERENCE_GATES and raising EXECUTABLE_TESTS_IN_THE_LISTED_SUITES by its test count is \
the whole fix -- or rename it if it was never meant to join. Going red for a NEW suite is \
deliberate: a suite that joins the corpus without joining the floor can leave it again without \
anyone noticing, which is the exact defect this list replaced." >&2
  exit 1
fi

bad=0 suites=0 tests=0 failed_suites=0 vacuous_suites=0
printf '%-42s %6s %6s %6s %8s  %s\n' SUITE PASS FAIL IGN SECS VERDICT
for f in "${BITEXACT_REFERENCE_GATES[@]}"; do
  s=$(basename "$f" .rs)
  log="$OUT/$s.log"
  if [ -n "${PARITY_AUDIT_FEATURES:-}" ]; then
    NVK_LANE="$LANE" NVK_FEATURES="$PARITY_AUDIT_FEATURES" "$NVK" test --test "$s" >"$log" 2>&1
  else
    NVK_LANE="$LANE" "$NVK" test --test "$s" >"$log" 2>&1
  fi
  line=$(grep -E '^test result:' "$log" | tail -1)
  pass=$(grep -oE '[0-9]+ passed' <<<"$line" | grep -oE '^[0-9]+')
  fail=$(grep -oE '[0-9]+ failed' <<<"$line" | grep -oE '^[0-9]+')
  ign=$(grep -oE '[0-9]+ ignored' <<<"$line" | grep -oE '^[0-9]+')
  secs=$(sed -n 's/.*finished in \([0-9.]*\)s.*/\1/p' <<<"$line")
  pass=${pass:-0}
  fail=${fail:-0}
  ign=${ign:-0}
  secs=${secs:-0}
  skips=$(grep -ciE '(^|[^a-z])skip' "$log")
  suites=$((suites + 1))
  tests=$((tests + pass))
  if [ -z "$line" ]; then
    v="NO RESULT LINE -- see $log"
    bad=$((bad + 1))
  elif [ "$fail" != "0" ]; then
    v="FAILED"
    bad=$((bad + 1))
    failed_suites=$((failed_suites + 1))
  elif [ "$pass" = "0" ] && [ "$ign" = "0" ]; then
    v="VACUOUS: compiled to nothing"
    bad=$((bad + 1))
    vacuous_suites=$((vacuous_suites + 1))
  elif [ "$skips" -gt 0 ]; then
    v="SKIPPED x$skips (env/dep gate)"
    bad=$((bad + 1))
  else
    v="ran"
  fi
  printf '%-42s %6s %6s %6s %8s  %s\n' "$s" "$pass" "$fail" "$ign" "$secs" "$v"
done

echo
echo "$suites suite(s), $tests test(s) executed, $bad suite(s) not credited"

if [ $bad -ne 0 ]; then
  if [ $failed_suites -ne 0 ]; then
    echo "parity audit FAILED: $failed_suites suite(s) have a FAILING test -- a real cross-backend \
disagreement, not a skip. See the per-suite logs in $OUT." >&2
    echo "This is reported apart from VACUOUS on purpose: the two shared one message once, and a \
genuinely failing test in parity_gemv_bf16_i8 was printed as 'green without running anything', \
which sends the reader hunting a cfg problem that is not there." >&2
  fi
  if [ $vacuous_suites -ne 0 ]; then
    echo "parity audit FAILED: $vacuous_suites suite(s) ran nothing (0 passed, 0 ignored)." >&2
    echo "If PARITY_AUDIT_FEATURES was set to one feature, that is the trap being demonstrated." >&2
  fi
  if [ $((failed_suites + vacuous_suites)) -ne $bad ]; then
    echo "parity audit FAILED: $((bad - failed_suites - vacuous_suites)) suite(s) skipped or \
produced no result line." >&2
  fi
  exit 1
fi
MIN_TESTS="${PARITY_AUDIT_MIN_TESTS:-$EXECUTABLE_TESTS_IN_THE_LISTED_SUITES}"
if [ "$tests" -lt "$MIN_TESTS" ]; then
  echo "parity audit FAILED: corpus shrank. The $suites listed suite(s) ran $tests test(s), floor \
is $MIN_TESTS." >&2
  echo "Every listed file is still present -- checked above -- so tests went missing INSIDE them: \
deleted, renamed away, or newly #[ignore]d. An #[ignore] is the quiet one; it lands in the \
'ignored' column, trips neither FAILED nor VACUOUS, and only this floor sees it." >&2
  echo "The floor is the exact executable count, not a slack number: parity_gdn 16 + \
parity_gemv_bf16_i8 20 + parity_kv_fp8_paged 18 + parity_verify_fused 7 (8 tests, 1 #[ignore]d) + \
wgpu_quantize_nvfp4_bf16_cpu_ref 2. Slack was tried and was wrong -- 8d44f1842 set the floor one \
under the count so a flaky suite would report FAILED instead of corpus-shrank, but a failing test \
sets bad and exits above, before this check is ever reached, so the slack bought nothing and hid \
one deletable test." >&2
  echo "Cross-backend parity is the thinnest coverage in this repo (task #50), so losing a test is \
a bigger deal than a test going red -- a red test is visible and a deleted one is not. If the \
removal was deliberate, lower EXECUTABLE_TESTS_IN_THE_LISTED_SUITES in the same commit that \
removes it and say why." >&2
  exit 1
fi
python3 "$(dirname "$0")/kernel-parity-census.py" --check || exit 1
echo "parity audit OK"
