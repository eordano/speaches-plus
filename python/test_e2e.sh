#!/usr/bin/env bash
set -u

PORT=${PORT:-1329}
ROOT=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$(command -v python)}
LOG=/tmp/nano-e2e.log
OUT=/tmp/nano-e2e
HEALTH_TIMEOUT_SECONDS=90
REQUEST_TIMEOUT_SECONDS=900
MIN_OUTPUT_BYTES=50000
mkdir -p "$OUT"

teardown() {
  if [ -n "${SERVER_PID:-}" ]; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap teardown EXIT

boot() {
  local model=$1 device=${2:-mps}
  teardown
  : >"$LOG"
  PYTHONPATH="$ROOT" \
    QWEN3_TTS_MODEL="$model" QWEN3_TTS_DEVICE="$device" QWEN3_TTS_DTYPE=bfloat16 \
    "$PY" -m uvicorn --app-dir "$ROOT" server:app --port "$PORT" >"$LOG" 2>&1 &
  SERVER_PID=$!
  for _ in $(seq 1 "$HEALTH_TIMEOUT_SECONDS"); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      return 0
    fi
    if grep -qE "Application startup failed|Traceback" "$LOG"; then
      tail -30 "$LOG"
      return 1
    fi
    sleep 1
  done
  echo "  TIMED OUT waiting for server" >&2
  tail -30 "$LOG" >&2
  return 1
}

assert_wav() {
  local path=$1 minbytes=${2:-$MIN_OUTPUT_BYTES}
  if [ ! -s "$path" ]; then
    echo "  ❌ $path: empty"
    return 1
  fi
  local size
  size=$(stat -f%z "$path" 2>/dev/null || stat -c%s "$path")
  if [ "$size" -lt "$minbytes" ]; then
    echo "  ❌ $path: only $size bytes (< $minbytes)"
    return 1
  fi
  if ! file "$path" | grep -q "WAVE audio"; then
    echo "  ❌ $path: not WAV"
    return 1
  fi
  local duration
  duration=$(ffprobe -v error -show_entries format=duration -of default=nokey=1:noprint_wrappers=1 "$path" 2>/dev/null)
  printf "  ✅ %s  %d B  %.2fs audio\n" "$path" "$size" "$duration"
}

post_speech() {
  local data_arg=$1 output_path=$2
  curl -fsS -m "$REQUEST_TIMEOUT_SECONDS" -X POST "http://127.0.0.1:$PORT/v1/audio/speech" \
    -H 'content-type: application/json' \
    "$data_arg" \
    -o "$output_path" -w "  HTTP %{http_code}  %{size_download}B  %{time_total}s\n"
}

echo "===== 1. CustomVoice (preset speaker) ====="
boot "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" || exit 1
post_speech \
  "-d {\"input\":\"Custom voice. Ryan reading a sentence.\",\"voice\":\"Ryan\",\"task_type\":\"CustomVoice\"}" \
  "$OUT/1-custom.wav"
assert_wav "$OUT/1-custom.wav"

echo
echo "===== 2. VoiceDesign (text-described voice) ====="
boot "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign" || exit 1
post_speech \
  "-d {\"input\":\"Voice design. Warm British male reading a sentence.\",\"voice\":\"warm British male, calm pace\",\"task_type\":\"VoiceDesign\"}" \
  "$OUT/2-design.wav"
assert_wav "$OUT/2-design.wav"

echo
echo "===== 3. Voice cloning (Base + reference audio) ====="
python3 -c "
import base64, json, pathlib
ref = base64.b64encode(pathlib.Path('$OUT/1-custom.wav').read_bytes()).decode()
pathlib.Path('$OUT/3-clone.json').write_text(json.dumps({
  'input': 'Voice clone. This is a different sentence in the cloned voice.',
  'task_type': 'Base',
  'language': 'English',
  'ref_audio': 'data:audio/wav;base64,' + ref,
  'ref_text': 'Custom voice. Ryan reading a sentence.',
  'x_vector_only_mode': False,
}))
"
boot "Qwen/Qwen3-TTS-12Hz-0.6B-Base" || exit 1
post_speech "--data-binary @$OUT/3-clone.json" "$OUT/3-clone.wav"
assert_wav "$OUT/3-clone.wav"

echo
echo "===== outputs ====="
ls -l "$OUT/"
echo
echo "play with:  for f in $OUT/*.wav; do open \"\$f\"; sleep 4; done"
