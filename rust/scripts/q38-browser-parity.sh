#!/usr/bin/env bash
set -euo pipefail

RUST_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${Q38WEB_WORK:?set Q38WEB_WORK to the serve root holding web/ and slice/}"
PORT="${Q38WEB_PORT:-8643}"
CHROME_DEFAULT="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
[ -x "$CHROME_DEFAULT" ] || CHROME_DEFAULT="$(command -v chromium || command -v google-chrome || echo "$CHROME_DEFAULT")"
CHROME="${Q38WEB_CHROME:-$CHROME_DEFAULT}"
ARMS="${Q38WEB_ARMS:-sg nosg}"
WAIT_S="${Q38WEB_WAIT_S:-600}"

if [ "${Q38WEB_DISPLAY:-}" = xvfb ] && [ -z "${Q38WEB_UNDER_XVFB:-}" ]; then
  XVFB_RUN="$(command -v xvfb-run || true)"
  [ -n "$XVFB_RUN" ] || XVFB_RUN="$(nix build --no-link --print-out-paths --inputs-from "$RUST_DIR/.." nixpkgs#xvfb-run)/bin/xvfb-run"
  export Q38WEB_UNDER_XVFB=1 Q38WEB_HEADLESS=0
  export Q38WEB_CHROME_FLAGS="${Q38WEB_CHROME_FLAGS:---no-sandbox --disable-gpu-sandbox --ignore-gpu-blocklist --enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE --use-gl=angle --use-angle=vulkan --ozone-platform=x11}"
  exec "$XVFB_RUN" -a -s "-screen 0 1600x900x24" "$0" "$@"
fi

[ -f "$WORK/web/nv_ocr_web_bg.wasm" ] || {
  echo "q38-browser-parity: no wasm at $WORK/web -- run OCR_WEB_OUT=$WORK/web build-ocr-web.sh" >&2
  exit 1
}
case " $ARMS " in
  *" sg "*|*" nosg "*)
    [ -f "$WORK/slice/ref-sg.json" ] || {
      echo "q38-browser-parity: no native reference at $WORK/slice -- run the q38_layer_parity_native suite per arm first (the tint arm needs no slice)" >&2
      exit 1
    } ;;
esac
[ -x "$CHROME" ] || {
  echo "q38-browser-parity: no chrome at $CHROME" >&2
  exit 1
}

cp "$RUST_DIR/crates/nv-ocr-web/harness/q38_layer_parity.html" "$WORK/q38_layer_parity.html"
cp "$RUST_DIR/crates/nv-ocr-web/harness/voxtral_tts.html" "$RUST_DIR/crates/nv-ocr-web/harness/voxtral_tts_worker.js" "$WORK/"
case " $ARMS " in
  *" tts "*)
    [ -e "$WORK/tts-model" ] || ln -s "${VOXTRAL_TTS_MODEL_DIR:?tts arm: set VOXTRAL_TTS_MODEL_DIR to the Voxtral-4B-TTS-2603 snapshot or link it at $WORK/tts-model}" "$WORK/tts-model"
    [ -e "$WORK/tts-fixtures" ] || ln -s "$RUST_DIR/tests/data" "$WORK/tts-fixtures"
    [ -f "$WORK/tts-model/consolidated.safetensors" ] || { echo "q38-browser-parity: $WORK/tts-model holds no consolidated.safetensors" >&2; exit 1; } ;;
esac

python3 "$RUST_DIR/crates/nv-ocr-web/harness/serve.py" "$PORT" "$WORK" &>"$WORK/http.log" &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 1

if [ "${Q38WEB_GPU_INFO:-0}" = 1 ]; then
  GPU_FLAGS="${Q38WEB_CHROME_FLAGS:---ignore-gpu-blocklist --use-gl=angle --use-angle=vulkan --enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE}"
  NVIDIA_ICD=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
  [ -f "$NVIDIA_ICD" ] && export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$NVIDIA_ICD}"
  rm -rf "$WORK/chrome-profile-gpuinfo"
  DBG_PORT="${Q38WEB_DEBUG_PORT:-9333}"
  # shellcheck disable=SC2086
  "$CHROME" --headless=new --no-first-run --user-data-dir="$WORK/chrome-profile-gpuinfo" \
    --enable-unsafe-webgpu --enable-logging=stderr --v=0 $GPU_FLAGS \
    --remote-debugging-port="$DBG_PORT" about:blank 2>"$WORK/gpu-info.log" &
  GPID=$!
  for _ in $(seq 1 40); do
    curl -sf "http://127.0.0.1:$DBG_PORT/json/version" >/dev/null 2>&1 && break
    sleep 0.5
  done
  python3 "$RUST_DIR/crates/nv-ocr-web/harness/gpu_info.py" "$DBG_PORT" | tee "$WORK/gpu-info.txt" || true
  kill $GPID 2>/dev/null || true
  wait $GPID 2>/dev/null || true
fi

fail=0
for arm in $ARMS; do
  rm -f "$WORK/q38-result-$arm.json"
  prof="$WORK/chrome-profile-$arm"
  rm -rf "$prof"
  log="$WORK/console-$arm.log"
  NVIDIA_ICD=/run/opengl-driver/share/vulkan/icd.d/nvidia_icd.json
  [ -f "$NVIDIA_ICD" ] && export VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-$NVIDIA_ICD}"
  GPU_FLAGS="${Q38WEB_CHROME_FLAGS:---ignore-gpu-blocklist --use-gl=angle --use-angle=vulkan --enable-features=Vulkan,DefaultANGLEVulkan,VulkanFromANGLE}"
  HEADLESS_FLAG="--headless=new"
  [ "${Q38WEB_HEADLESS:-1}" = 0 ] && HEADLESS_FLAG=""
  PAGE=q38_layer_parity.html
  [ "$arm" = tts ] && PAGE=voxtral_tts.html
  # shellcheck disable=SC2086
  "$CHROME" $HEADLESS_FLAG --no-first-run --user-data-dir="$prof" \
    --enable-unsafe-webgpu --enable-logging=stderr --v=0 $GPU_FLAGS \
    "http://127.0.0.1:$PORT/$PAGE?arm=$arm${Q38WEB_QUERY:+&$Q38WEB_QUERY}" &>"$log" &
  CPID=$!
  waited=0
  until [ -s "$WORK/q38-result-$arm.json" ]; do
    sleep 2
    waited=$((waited + 2))
    if [ "$waited" -ge "$WAIT_S" ]; then break; fi
    kill -0 $CPID 2>/dev/null || break
  done
  sleep 1
  kill $CPID 2>/dev/null || true
  wait $CPID 2>/dev/null || true
  if [ ! -s "$WORK/q38-result-$arm.json" ]; then
    echo "Q38WEB-BROWSER arm=$arm NO RESULT (console tail follows)"
    tail -5 "$log" || true
    fail=1
    continue
  fi
  if ! python3 - "$WORK/q38-result-$arm.json" <<'EOF'
import json, sys
j = json.load(open(sys.argv[1]))
print(f"Q38WEB-BROWSER arm={j.get('arm')} done={j.get('done')} verdict={j.get('verdict')}")
if j.get('tint'):
    t = j['tint']
    print(f"  adapter={t.get('adapter')}")
    print(f"  limits={json.dumps(t.get('limits'))}")
    print(f"  over-limit entries (routed around at runtime)={len(t.get('limited') or [])} tint rejections={len(t.get('failures') or [])}")
    for f in (t.get('failures') or [])[:12]:
        print(f"    REJECTED {f.get('tag')}: {str(f.get('err')).splitlines()[0][:160]}")
if j.get('census'):
    print(f"  census={json.dumps(j['census'])}")
    print(f"  token_match={j.get('token_match')} labels_match={j.get('labels_match')}")
print(f"  wasm_bytes={j.get('wasm_bytes')} staged_bytes={j.get('staged_bytes')} "
      f"shard_load_ms={round(j.get('shard_load_ms') or -1)} boot_ms={round(j.get('boot_ms') or -1)} "
      f"prefill_ms={round(j.get('prefill_ms') or -1, 1)} step_ms={[round(x,1) for x in (j.get('step_ms') or [])]}")
if j.get('error'):
    print(f"  ERROR: {j['error'][:600]}")
if j.get('train'):
    t = j['train']
    print(f"  train samples={t.get('samples')} losses={[round(x, 6) for x in (t.get('losses') or [])]} grads_finite={t.get('all_grads_finite')} adapter_bytes={t.get('adapter_safetensors_bytes')} ms={round(j.get('train_ms') or 0)} verdict={j.get('verdict')}")
    if j.get('e4b_window'):
        print(f"  e4b window={json.dumps(j['e4b_window'])} staged_bytes={j.get('staged_bytes')} sample_bytes={j.get('sample_bytes')} stage_ms={round(j.get('stage_ms') or 0)}")
if j.get('init') is not None and j.get('style') is not None:
    print(f"  kokoro init={json.dumps(j['init'])} init_ms={round(j.get('init_ms') or 0)} synth_ms={round(j.get('synth_ms') or 0)} synth_samples={j.get('synth_samples')}")
    print(f"  style={json.dumps(j['style'])} style_ms={round(j.get('style_ms') or 0)}")
    print(f"  adapter={json.dumps({k: v for k, v in (j.get('adapter') or {}).items() if k != 'epochs'})} adapter_ms={round(j.get('adapter_ms') or 0)} adapter_bytes={j.get('adapter_safetensors_bytes')} verdict={j.get('verdict')}")
if j.get('tts'):
    print(f"  adapter={json.dumps(j.get('adapter'))}")
    print(f"  wasm_init_ms={round(j.get('wasm_init_ms') or 0)} open_ms={round(j.get('open_ms') or 0)} verdict={j.get('verdict')}")
    for tier, t in j['tts'].items():
        print(f"  {tier}: boot_ms={round(t.get('boot_ms') or 0)} verdict={t.get('verdict')}")
        for case, r in (t.get('cases') or {}).items():
            print(f"    {case}: {json.dumps(r)}")
if j.get('probe_adapter'):
    print(f"  probe adapter={json.dumps(j['probe_adapter'])[:400]}")
ok = j.get('done') in ('sg-bit-equal', 'sg-characterized', 'nosg-refusal-matched', 'tint-clean', 'probe-ok', 'train-ok', 'e4b-ok', 'kokoro-ok', 'tts-ok')
sys.exit(0 if ok else 3)
EOF
  then fail=1; fi
done
exit $fail
