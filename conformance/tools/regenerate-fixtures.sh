#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CONF="$(cd "$HERE/.." && pwd)"
FIXTURES="$CONF/fixtures"
SCALE="$CONF/fixtures-scale"
NIX_HINT="nix-shell -p imagemagick fontconfig dejavu_fonts liberation_ttf 'python3.withPackages (ps: [ps.pillow ps.numpy])' --run $0"

die() {
  echo "regenerate-fixtures: $*" >&2
  echo "  try: $NIX_HINT" >&2
  exit 1
}

font_file() {
  local base="$1" f d
  f="$(fc-list -f '%{file}\n' 2>/dev/null | grep -F -- "/$base" | sort | sed -n 1p || true)"
  if [ -z "$f" ]; then
    for d in ${buildInputs:-} /run/current-system/sw/share/X11/fonts /usr/share/fonts /usr/local/share/fonts "$HOME/.local/share/fonts"; do
      [ -d "$d" ] || continue
      f="$(find "$d" -maxdepth 4 -name "$base" 2>/dev/null | sort | sed -n 1p || true)"
      [ -n "$f" ] && break
    done
  fi
  [ -n "$f" ] || die "font $base not found; set the env var that names it"
  echo "$f"
}

command -v magick >/dev/null || die "magick (ImageMagick 7) missing"
command -v python3 >/dev/null || die "python3 missing"
python3 -c 'import PIL, numpy' 2>/dev/null || die "python3 needs pillow and numpy"

export FONT="${FONT:-$(font_file DejaVuSans.ttf)}"
export FONTS="${FONTS:-$(dirname "$FONT")}"
export FONT_REG="${FONT_REG:-$(font_file LiberationSans-Regular.ttf)}"
export FONT_BOLD="${FONT_BOLD:-$(font_file LiberationSans-Bold.ttf)}"
export FONT_MONO="${FONT_MONO:-$(font_file LiberationMono-Regular.ttf)}"
for f in "$FONT" "$FONTS/DejaVuSans-Bold.ttf" "$FONTS/DejaVuSerif.ttf" "$FONTS/DejaVuSansMono.ttf" \
         "$FONT_REG" "$FONT_BOLD" "$FONT_MONO"; do
  [ -f "$f" ] || die "font file missing: $f"
done
echo "fonts: DejaVu=$FONTS Liberation=$FONT_REG $FONT_BOLD $FONT_MONO"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "== 070-ocr (magick)"
OUT="$TMP/070" bash "$HERE/generate-070-ocr-fixtures.sh" >/dev/null
for d in "$TMP"/070/*/; do
  n="$(basename "$d")"
  cp "$d/input.png" "$FIXTURES/$n/input.png"
  python3 - "$d/fixture.json" "$FIXTURES/$n/fixture.json" <<'EOF'
import json, sys
new, tracked = (json.load(open(p)) for p in sys.argv[1:3])
if new != tracked:
    print(f"warning: {sys.argv[2]} no longer matches the generator output (word_rects/gates); "
          f"rerun OUT=<dir> generate-070-ocr-fixtures.sh and review", file=sys.stderr)
EOF
  echo "  $n/input.png"
done

echo "== 071-ocr-layout (PIL)"
OUT="$TMP/071" python3 "$HERE/generate-071-ocr-layout-fixtures.py" >/dev/null
python3 - "$TMP/071" "$FIXTURES" <<'EOF'
import glob, json, os, shutil, sys
rendered, fixtures = sys.argv[1:3]
for fj in sorted(glob.glob(os.path.join(fixtures, "071-*", "fixture.json"))):
    fx = json.load(open(fj))
    case = fx["render"]["case"]
    d = os.path.dirname(fj)
    shutil.copyfile(os.path.join(rendered, case + ".png"), os.path.join(d, "input.png"))
    text = open(os.path.join(rendered, case + ".txt")).read().rstrip("\n")
    if text != fx["expected_text"]:
        print(f"warning: {fj} expected_text differs from the generator's {case}.txt", file=sys.stderr)
    print(f"  {os.path.basename(d)}/input.png <- {case}")
EOF

echo "== 090-scale (PIL; input.png + fixture.json)"
python3 "$HERE/generate-090-scale-fixtures.py" "$SCALE" | sed 's/^/  /'

echo "== sine WAVs"
OUT="$FIXTURES" python3 "$HERE/generate-audio.py" | sed 's/^/  /'

echo "== 050-diarization-multispeaker (Kokoro TTS)"
DIAR="$FIXTURES/050-diarization-multispeaker"
if [ -n "${DIAR_FIXTURE_KOKORO_MODEL:-}" ]; then
  export DIAR_FIXTURE_KOKORO_VOICES="${DIAR_FIXTURE_KOKORO_VOICES:-$(dirname "$DIAR_FIXTURE_KOKORO_MODEL")/voices.bin}"
  export DIAR_FIXTURE_BUILD=1 DIAR_FIXTURE_OUT="$DIAR"
  NVK_PKG=speaches-plus NVK_FEATURES= "$CONF/../rust/scripts/nvk.sh" test --lib \
    diarization::hop_sweep::diar_build_multispeaker_fixture -- --ignored --nocapture
else
  cat <<EOF
  skipped: audio.wav is synthesized by Kokoro-82M v1.0 ONNX (rust/src/diarization/hop_sweep.rs).
  K=\$HOME/.cache/huggingface/hub/models--speaches-ai--Kokoro-82M-v1.0-ONNX/snapshots/*/
  DIAR_FIXTURE_KOKORO_MODEL=\$K/model.onnx DIAR_FIXTURE_KOKORO_VOICES=\$K/voices.bin $0
  (only 050 reruns the Rust build; see $DIAR/README.md)
EOF
fi

echo "done"
