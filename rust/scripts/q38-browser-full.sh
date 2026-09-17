#!/usr/bin/env bash
set -euo pipefail

RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${Q38WEB_WORK:?set Q38WEB_WORK to the serve root holding web/ and full-model/}"
PORT="${Q38WEB_PORT:-8643}"
CHROME="${Q38WEB_CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
WAIT_S="${Q38WEB_WAIT_S:-3600}"
STRICT="${Q38WEB_STRICT:-0}"
DEEP="${Q38WEB_DEEP:-4096}"
EXTRA="${Q38WEB_QUERY:-}"

[ -f "$WORK/web/nv_ocr_web_bg.wasm" ] || {
  echo "q38-browser-full: no wasm at $WORK/web -- run OCR_WEB_OUT=$WORK/web build-ocr-web.sh" >&2
  exit 1
}
for stem in ref-full ref-full-refute; do
  [ -f "$WORK/full-model/$stem.json" ] && [ -f "$WORK/full-model/$stem-logits.bin" ] || {
    echo "q38-browser-full: no native reference $stem at $WORK/full-model -- run the q38_full_native suite first" >&2
    exit 1
  }
done
[ -x "$CHROME" ] || {
  echo "q38-browser-full: no chrome at $CHROME" >&2
  exit 1
}

cp "$RUST_DIR/crates/nv-ocr-web/harness/q38_full.html" "$WORK/q38_full.html"

python3 "$RUST_DIR/crates/nv-ocr-web/harness/serve.py" "$PORT" "$WORK" &>"$WORK/http-full.log" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 1

run_fixture() {
  local stem="$1" result="$2" deep="$3" mode="$4"
  rm -f "$WORK/$result"
  local prof="$WORK/chrome-profile-full-$stem"
  rm -rf "$prof"
  local log="$WORK/console-full-$stem.log"
  local query="?ref=$stem&result=$result&deep=$deep${EXTRA:+&$EXTRA}"
  "$CHROME" --headless=new --no-first-run --user-data-dir="$prof" \
    --enable-unsafe-webgpu --enable-logging=stderr --v=0 \
    --js-flags=--max-old-space-size=8192 \
    "http://127.0.0.1:$PORT/q38_full.html$query" &>"$log" &
  local cpid=$!
  local waited=0
  until [ -s "$WORK/$result" ]; do
    sleep 5
    waited=$((waited + 5))
    if [ "$waited" -ge "$WAIT_S" ]; then break; fi
    kill -0 $cpid 2>/dev/null || break
  done
  sleep 1
  kill $cpid 2>/dev/null || true
  wait $cpid 2>/dev/null || true
  if [ ! -s "$WORK/$result" ]; then
    echo "Q38WEB-FULL[$stem] NO RESULT (console tail follows)"
    tail -8 "$log" || true
    return 1
  fi
  python3 - "$WORK/$result" "$stem" "$mode" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
stem, mode = sys.argv[2], sys.argv[3]
print(f"Q38WEB-FULL[{stem}] done={j.get('done')} mode={mode}")
print(f"  verdict={j.get('verdict')}")
crit = j.get('census_criteria') or {}
det = j.get('census_determinism') or {}
print(f"  criteria: envelope_ok={crit.get('envelope_ok')} argmax_match={crit.get('argmax_match')} "
      f"labels_match={crit.get('labels_match')} determinism_ok={crit.get('determinism_ok')} "
      f"max_abs_ceiling={crit.get('max_abs_ceiling')}")
print(f"  determinism={json.dumps(det)}")
if j.get('census'):
    print(f"  census={json.dumps(j['census'])}")
print(f"  token_match={j.get('token_match')} labels_match={j.get('labels_match')}")
if j.get('token_divergence'):
    d = j['token_divergence']
    print(f"  TOKEN-DIVERGENCE step={d.get('step')} browser={d.get('browser_id')} "
          f"native={d.get('native_id')} (reported; not a failure when census criteria hold)")
print(f"  staged_bytes={j.get('staged_bytes')} shard_load_ms={round(j.get('shard_load_ms') or -1)} "
      f"opfs={json.dumps({k: v for k, v in (j.get('opfs') or {}).items() if k != 'dir'})}")
print(f"  boot_ms={round(j.get('boot_ms') or -1)} prefill_ms={round(j.get('prefill_ms') or -1, 1)}")
if j.get('quiet_step_ms_mean') is not None:
    print(f"  quiet_step_ms mean={j['quiet_step_ms_mean']:.2f} median={j['quiet_step_ms_median']:.2f} "
          f"tok_per_s={j['tok_per_s']:.1f}")
if j.get('deep'):
    d = j['deep']
    print(f"  deep d{d['target_pos']}: steps={d['steps']} tail64 mean={d['tail64_ms_mean']:.2f} "
          f"median={d['tail64_ms_median']:.2f} tok_per_s={d['tail64_tok_per_s']:.1f}")
for key in ('mem_after_boot', 'mem_device'):
    m = j.get(key)
    if m:
        print(f"  {key}: live={m.get('live_bytes')} peak={m.get('peak_live_bytes')} "
              f"buffers={m.get('buffers_live')} wasm={m.get('wasm_memory_bytes')} "
              f"js_shards={m.get('js_shard_bytes_live')}")
for key in ('error', 'stage_error', 'boot_error', 'deep_error', 'page_error'):
    if j.get(key):
        print(f"  {key.upper()}: {j[key][:600]}")
passing = {'full-bit-equal', 'full-characterized', 'full-characterized-token-divergence'}
if mode == 'strict':
    passing = {'full-bit-equal', 'full-characterized'}
sys.exit(0 if j.get('done') in passing else 3)
EOF
}

MODE=characterized
[ "$STRICT" = "1" ] && MODE=strict
rc=0
run_fixture ref-full q38-result-full.json "$DEEP" "$MODE" || rc=1
run_fixture ref-full-refute q38-result-full-refute.json 0 characterized || rc=1
exit $rc
