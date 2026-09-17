#!/usr/bin/env bash
set -euo pipefail

RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${Q38WEB_WORK:?set Q38WEB_WORK to the serve root holding web/ and slice-n<N>/}"
PORT="${Q38WEB_PORT:-8643}"
CHROME="${Q38WEB_CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
NS="${Q38WEB_NS:-2 4 8}"
WAIT_S="${Q38WEB_WAIT_S:-900}"

[ -f "$WORK/web/nv_ocr_web_bg.wasm" ] || {
  echo "q38-browser-splice: no wasm at $WORK/web -- run OCR_WEB_OUT=$WORK/web build-ocr-web.sh" >&2
  exit 1
}
[ -x "$CHROME" ] || {
  echo "q38-browser-splice: no chrome at $CHROME" >&2
  exit 1
}

cp "$RUST_DIR/crates/nv-ocr-web/harness/q38_splice.html" "$WORK/q38_splice.html"

python3 "$RUST_DIR/crates/nv-ocr-web/harness/serve.py" "$PORT" "$WORK" &>"$WORK/http-splice.log" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 1

fail=0
for n in $NS; do
  [ -f "$WORK/slice-n$n/ref-splice.json" ] || {
    echo "q38-browser-splice: no native reference at $WORK/slice-n$n -- run the q38_splice_native suite for N=$n first" >&2
    fail=1
    continue
  }
  rm -f "$WORK/q38-result-splice-n$n.json"
  prof="$WORK/chrome-profile-splice-n$n"
  rm -rf "$prof"
  log="$WORK/console-splice-n$n.log"
  "$CHROME" --headless=new --no-first-run --user-data-dir="$prof" \
    --enable-unsafe-webgpu --enable-logging=stderr --v=0 \
    --js-flags=--max-old-space-size=8192 \
    "http://127.0.0.1:$PORT/q38_splice.html?n=$n" &>"$log" &
  CPID=$!
  waited=0
  until [ -s "$WORK/q38-result-splice-n$n.json" ]; do
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge "$WAIT_S" ]; then break; fi
    kill -0 $CPID 2>/dev/null || break
  done
  sleep 1
  kill $CPID 2>/dev/null || true
  wait $CPID 2>/dev/null || true
  if [ ! -s "$WORK/q38-result-splice-n$n.json" ]; then
    echo "Q38WEB-SPLICE n=$n NO RESULT (console tail follows)"
    tail -5 "$log" || true
    fail=1
    continue
  fi
  if ! python3 - "$WORK/q38-result-splice-n$n.json" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
print(f"Q38WEB-SPLICE n={j.get('n_layers')} done={j.get('done')} verdict={j.get('verdict')}")
if j.get('census'):
    print(f"  census={json.dumps(j['census'])}")
    print(f"  token_match={j.get('token_match')} labels_match={j.get('labels_match')}")
print(f"  staged_bytes={j.get('staged_bytes')} shard_load_ms={round(j.get('shard_load_ms') or -1)} "
      f"boot_ms={round(j.get('boot_ms') or -1)} prefill_ms={round(j.get('prefill_ms') or -1, 1)}")
if j.get('quiet_step_ms_mean') is not None:
    print(f"  quiet_step_ms mean={j['quiet_step_ms_mean']:.2f} median={j['quiet_step_ms_median']:.2f} "
          f"tok_per_s={j['tok_per_s']:.1f}")
for key in ('mem_after_boot', 'mem_device'):
    m = j.get(key)
    if m:
        print(f"  {key}: live={m.get('live_bytes')} peak={m.get('peak_live_bytes')} "
              f"buffers={m.get('buffers_live')} wasm={m.get('wasm_memory_bytes')}")
if j.get('error'):
    print(f"  ERROR: {j['error'][:600]}")
if j.get('stage_error'):
    print(f"  STAGE_ERROR: {j['stage_error'][:300]}")
ok = j.get('done') in ('splice-bit-equal', 'splice-characterized')
sys.exit(0 if ok else 3)
EOF
  then fail=1; fi
done
exit $fail
