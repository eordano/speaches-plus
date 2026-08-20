#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS_NIX="nix/models.nix"
LOG="${LOG:-/tmp/nano-omni-fetch-models.log}"
: >"$LOG"

ENTRIES=(
  "kokoroOnnxCommunity onnx-community/Kokoro-82M-v1.0-ONNX"
  "qwenTts17Cv         Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
  "qwenTts06Cv         Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
  "qwenTts17Vd         Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
  "qwenTts06B          Qwen/Qwen3-TTS-12Hz-0.6B-Base"
  "qwenAligner         Qwen/Qwen3-ForcedAligner-0.6B"
  "qwenOmni30B         Qwen/Qwen3-Omni-30B-A3B-Instruct"
  "gemma4E4B           google/gemma-4-E4B-it"
  "gemma431B           google/gemma-4-31B-it"
  "whisperGgml         ggerganov/whisper.cpp"
  "whisperCt2          deepdml/faster-whisper-large-v3-turbo-ct2"
  "kokoroSpeaches      speaches-ai/Kokoro-82M-v1.0-ONNX"
  "smartTurnV3         pipecat-ai/smart-turn-v3"
  "diarizen            BUT-FIT/diarizen-wavlm-large-s80-md-v2"
  "wespeaker           Wespeaker/wespeaker-voxceleb-resnet293-LM"
)

SILERO_URL="https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"

if [[ ! -f "$MODELS_NIX" ]]; then
  echo "error: $MODELS_NIX does not exist. Bootstrap with the stub from the plan." >&2
  exit 1
fi

extract() {
  local field=$1 text=$2
  printf '%s\n' "$text" | sed -nE "s/.*${field} = \"([^\"]+)\".*/\1/p" | head -n1
}

patch_attr() {
  local attr=$1 rev=$2 hash=$3
  python3 - "$MODELS_NIX" "$attr" "$rev" "$hash" <<'PY'
import re, sys
path, attr, rev, hash_ = sys.argv[1:]
src = open(path).read()
# Match: <attr> = nix-hug-lib.fetchModel { ... rev = "TODO"; fileTreeHash = "TODO"; ... };
pat = re.compile(
    r'(\b' + re.escape(attr) + r'\s*=\s*nix-hug-lib\.fetchModel\s*\{[^}]*?rev\s*=\s*")[^"]*("\s*;[^}]*?fileTreeHash\s*=\s*")[^"]*("\s*;[^}]*?\};)',
    re.DOTALL,
)
new, n = pat.subn(rf'\g<1>{rev}\g<2>{hash_}\g<3>', src)
if n != 1:
    print(f"warn: no match for {attr} (n={n}), skipping", file=sys.stderr)
    sys.exit(0)
open(path, 'w').write(new)
print(f"  patched {attr}")
PY
}

is_already_pinned() {
  local attr=$1
  python3 - "$MODELS_NIX" "$attr" <<'PY'
import re, sys
path, attr = sys.argv[1:]
src = open(path).read()
m = re.search(
    r'\b' + re.escape(attr) + r'\s*=\s*nix-hug-lib\.fetchModel\s*\{[^}]*?rev\s*=\s*"([^"]+)"[^}]*?fileTreeHash\s*=\s*"([^"]+)"',
    src, re.DOTALL,
)
if m and m.group(1) != "TODO" and m.group(2) != "TODO":
    sys.exit(0)
sys.exit(1)
PY
}

for entry in "${ENTRIES[@]}"; do
  read -r attr url <<<"$entry"
  echo "==> $attr  ($url)" | tee -a "$LOG"
  if is_already_pinned "$attr"; then
    echo "  [skip] already pinned" | tee -a "$LOG"
    continue
  fi
  out=$(nix-hug fetch "$url" 2>&1) || {
    echo "  fetch failed:"
    printf '%s\n' "$out" | tail -20
    exit 1
  }
  rev=$(printf '%s\n' "$out" | sed -nE 's/.*rev = "([^"]+)".*/\1/p' | head -n1)
  hash=$(printf '%s\n' "$out" | sed -nE 's/.*fileTreeHash = "([^"]+)".*/\1/p' | head -n1)
  if [[ -z "$rev" || -z "$hash" ]]; then
    echo "  could not parse rev/hash from output:" >&2
    printf '%s\n' "$out" | tail -20 >&2
    exit 1
  fi
  echo "  rev=$rev hash=$hash" | tee -a "$LOG"
  patch_attr "$attr" "$rev" "$hash"
done

echo "==> sileroVad  (GitHub raw)" | tee -a "$LOG"
silero_hash=$(nix-prefetch-url "$SILERO_URL")
echo "  sha256=$silero_hash" | tee -a "$LOG"
python3 - "$MODELS_NIX" "$silero_hash" <<'PY'
import re, sys
path, hash_ = sys.argv[1:]
src = open(path).read()
pat = re.compile(r'(sileroVad\s*=\s*pkgs\.fetchurl\s*\{[^}]*?sha256\s*=\s*")[^"]*("\s*;[^}]*?\};)', re.DOTALL)
new, n = pat.subn(rf'\g<1>{hash_}\g<2>', src)
if n != 1:
    print(f"warn: no match for sileroVad (n={n})", file=sys.stderr); sys.exit(0)
open(path, 'w').write(new)
print("  patched sileroVad")
PY

echo
echo "All models pinned in $MODELS_NIX. Next: \`nix develop\`."
