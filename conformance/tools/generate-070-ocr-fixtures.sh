#!/usr/bin/env bash
set -euo pipefail

MAGICK="${MAGICK:-magick}"
FONT="${FONT:-/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf}"
OUT="${OUT:-$(cd "$(dirname "$0")/../fixtures" && pwd)}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

PARA_L1='The quick brown fox jumps over 13 lazy dogs.'
PARA_L2='Pack my box with five dozen liquor jugs; costs 42.'
PARA_L3='How vexingly quick daft zebras jump!'
PARA_L4='Sphinx of black quartz, judge my vow.'
PARA_L5='Amazingly few discotheques provide 8 jukeboxes.'
PARA_L6='Mr. Jock, TV quiz PhD, bags few lynx.'
PARA="$PARA_L1
$PARA_L2
$PARA_L3
$PARA_L4
$PARA_L5
$PARA_L6"
PARA_JSON="$PARA_L1\\n$PARA_L2\\n$PARA_L3\\n$PARA_L4\\n$PARA_L5\\n$PARA_L6"

emit() {
  local dir="$1" json="$2" readme="$3"
  mkdir -p "$OUT/$dir"
  printf '%s\n' "$json" >"$OUT/$dir/fixture.json"
  printf '%s\n' "$readme" >"$OUT/$dir/README.md"
}

name=070-ocr-clean-line
mkdir -p "$OUT/$name"
"$MAGICK" -background white -fill black -font "$FONT" -pointsize 44 \
  label:'Quick brown foxes jump' -bordercolor white -border 30 -depth 8 \
  "$OUT/$name/input.png"
emit "$name" "$(
  cat <<EOF
{
  "name": "$name",
  "family": "070-ocr",
  "description": "One clean rendered line, ~32 px cap height, black on white. Gates exact text (CER = 0), 1 line, 4 words. Requires the tessdata_best eng.traineddata cache (NV_OCR_TESSDATA).",
  "input": "input.png",
  "expected_text": "Quick brown foxes jump",
  "render": {"font": "DejaVuSans", "pointsize": 44, "extra_ops": ["-border 30"]},
  "gates": {"cer_max": 0.0, "line_count": 1, "word_count": 4},
  "oracle": {"enabled": true, "psm": 7},
  "comparison_strategy": "cer+layout_gates",
  "skip_when_no_model": true
}
EOF
)" 'One clean line, ~32 px cap height, black on white. Gate: exact text match (CER = 0) with tessdata_best.'

name=070-ocr-paragraph
mkdir -p "$OUT/$name"
"$MAGICK" -background white -fill black -font "$FONT" -pointsize 44 \
  -interline-spacing 10 label:"$PARA" -bordercolor white -border 40 -depth 8 \
  "$OUT/$name/input.png"
emit "$name" "$(
  cat <<EOF
{
  "name": "$name",
  "family": "070-ocr",
  "description": "Six rendered lines with mixed case, digits and punctuation. Gates CER <= 0.5%, exact line count and exact reading order. Requires the tessdata_best eng.traineddata cache (NV_OCR_TESSDATA).",
  "input": "input.png",
  "expected_text": "$PARA_JSON",
  "render": {"font": "DejaVuSans", "pointsize": 44, "extra_ops": ["-interline-spacing 10", "-border 40"]},
  "gates": {"cer_max": 0.005, "line_count": 6, "word_counts": [9, 10, 6, 7, 6, 8]},
  "oracle": {"enabled": true, "psm": 6},
  "comparison_strategy": "cer+layout_gates",
  "skip_when_no_model": true
}
EOF
)" 'Six lines, mixed case + digits + punctuation. Gates: CER <= 0.5%, line count exact, reading order exact.'

name=070-ocr-skew-2deg
mkdir -p "$OUT/$name"
"$MAGICK" "$OUT/070-ocr-paragraph/input.png" -background white -rotate 2 +repage -depth 8 \
  "$OUT/$name/input.png"
emit "$name" "$(
  cat <<EOF
{
  "name": "$name",
  "family": "070-ocr",
  "description": "The paragraph page rotated +2.0 deg (imagemagick -rotate 2, clockwise). Gates the deskew estimate within 0.2 deg of 2.0 in magnitude and CER <= 1%. Requires the tessdata_best eng.traineddata cache (NV_OCR_TESSDATA).",
  "input": "input.png",
  "expected_text": "$PARA_JSON",
  "render": {"font": "DejaVuSans", "pointsize": 44, "extra_ops": ["-interline-spacing 10", "-border 40", "-rotate 2"]},
  "applied_rotation_deg": 2.0,
  "gates": {"cer_max": 0.01, "line_count": 6, "deskew_abs_err_max": 0.2},
  "oracle": {"enabled": true, "psm": 6},
  "comparison_strategy": "cer+layout_gates",
  "skip_when_no_model": true
}
EOF
)" 'The paragraph fixture rotated +2.0 degrees (imagemagick -rotate 2, clockwise). Gates: deskew estimate within 0.2 deg of 2.0 in magnitude; CER <= 1%.'

name=070-ocr-skew-12deg
mkdir -p "$OUT/$name"
"$MAGICK" "$OUT/070-ocr-paragraph/input.png" -background white -rotate 12 +repage -depth 8 \
  "$OUT/$name/input.png"
emit "$name" "$(
  cat <<EOF
{
  "name": "$name",
  "family": "070-ocr",
  "description": "The paragraph page rotated +12.0 deg, outside the legacy +/-5 deg deskew sweep. Gates the deskew estimate within 0.5 deg of 12.0 in magnitude, 6 lines and CER <= 2%. Requires the tessdata_best eng.traineddata cache (NV_OCR_TESSDATA).",
  "input": "input.png",
  "expected_text": "$PARA_JSON",
  "render": {"font": "DejaVuSans", "pointsize": 44, "extra_ops": ["-interline-spacing 10", "-border 40", "-rotate 12"]},
  "applied_rotation_deg": 12.0,
  "gates": {"cer_max": 0.02, "line_count": 6, "deskew_abs_err_max": 0.5},
  "oracle": {"enabled": true, "psm": 6},
  "comparison_strategy": "cer+layout_gates",
  "skip_when_no_model": true
}
EOF
)" 'The paragraph fixture rotated +12.0 degrees, outside the legacy +/-5 deg deskew sweep. Gates: deskew estimate within 0.5 deg of 12.0 in magnitude; CER <= 2%; the rotated canvas must be expanded so no line is clipped.'

name=070-ocr-photo-noise-surround
mkdir -p "$OUT/$name"
"$MAGICK" "$OUT/070-ocr-paragraph/input.png" -resize 150% -depth 8 "$TMP/page15.png"
"$MAGICK" -size 128x128 xc:'gray(45%)' -attenuate 1.2 +noise Gaussian -colorspace Gray -depth 8 \
  "$TMP/tile.png"
PW=$("$MAGICK" "$TMP/page15.png" -format '%w' info:)
PH=$("$MAGICK" "$TMP/page15.png" -format '%h' info:)
CW=$((PW + 384))
CH=$((PH + 384))
"$MAGICK" -size "${CW}x${CH}" "tile:$TMP/tile.png" -colorspace Gray \
  "$TMP/page15.png" -geometry "+192+192" -composite \
  \( -size "${CW}x${CH}" gradient:'gray(100%)'-'gray(78%)' \) -compose Multiply -composite \
  -depth 8 "$OUT/$name/input.png"
emit "$name" "$(
  cat <<EOF
{
  "name": "$name",
  "family": "070-ocr",
  "description": "The paragraph page at 150% scale composited onto a grainy mid-grey surround under a vertical shadow gradient. Gates at least 4 of the 6 expected lines recovered with per-line CER <= 0.1. Requires the tessdata_best eng.traineddata cache (NV_OCR_TESSDATA).",
  "input": "input.png",
  "expected_text": "$PARA_JSON",
  "render": {"font": "DejaVuSans", "pointsize": 44, "extra_ops": ["-resize 150%", "tile:128x128 gray(45%) +noise Gaussian surround", "page composited at +192+192", "gradient gray(100%)-gray(78%) multiply"]},
  "gates": {"lines_recovered_min": 4, "line_cer_max": 0.1},
  "oracle": {"enabled": false, "psm": 6},
  "comparison_strategy": "cer+layout_gates",
  "skip_when_no_model": true
}
EOF
)" 'The paragraph page at 150% scale on a grainy mid-grey surround under a shadow gradient - the Real5 capture mode where a document is photographed on a textured surface. The surround binarizes to a speckle field that outnumbers the text 100:1, collapsing whole-width line banding to ONE band. Recovery needs both grey-level page-region suppression and the speckle filter. Gate is per-line (a 64 px rim survives the cell mask and emits junk lines): at least 4 of 6 expected lines with per-line CER <= 0.1.'

name=070-ocr-lowcontrast-gradient
mkdir -p "$OUT/$name"
"$MAGICK" -size 140x760 gradient:'gray(32%)'-'gray(100%)' -rotate 90 \
  -font "$FONT" -pointsize 44 -fill black -annotate +40+95 'Gradient shadows persist' -depth 8 \
  "$OUT/$name/input.png"
emit "$name" "$(
  cat <<EOF
{
  "name": "$name",
  "family": "070-ocr",
  "description": "Black text over a horizontal luminance gradient (grey 32% to 100%). Sauvola local binarization must beat global Otsu, which a unit test asserts separately. Gates CER <= 2%. Requires the tessdata_best eng.traineddata cache (NV_OCR_TESSDATA).",
  "input": "input.png",
  "expected_text": "Gradient shadows persist",
  "render": {"font": "DejaVuSans", "pointsize": 44, "extra_ops": ["gradient:gray(32%)-gray(100%) -rotate 90", "-annotate +40+95"]},
  "gates": {"cer_max": 0.02, "line_count": 1, "word_count": 3, "otsu_must_lose": true},
  "oracle": {"enabled": true, "psm": 7},
  "comparison_strategy": "cer+layout_gates",
  "skip_when_no_model": true
}
EOF
)" 'Black text over a horizontal luminance gradient (gray 32% to 100%). Sauvola local binarization must win where global Otsu fails; a unit test asserts Otsu is strictly worse on this image. Gate: CER <= 2%.'

name=070-ocr-noise-gauss
mkdir -p "$OUT/$name"
"$MAGICK" -background white -fill black -font "$FONT" -pointsize 44 \
  label:'Noise resistant text' -bordercolor white -border 40 \
  -attenuate 0.6 +noise Gaussian -depth 8 \
  "$OUT/$name/input.png"
emit "$name" "$(
  cat <<EOF
{
  "name": "$name",
  "family": "070-ocr",
  "description": "Clean render plus additive Gaussian noise, sigma ~12 grey levels. Gates CER <= 2%. Requires the tessdata_best eng.traineddata cache (NV_OCR_TESSDATA).",
  "input": "input.png",
  "expected_text": "Noise resistant text",
  "render": {"font": "DejaVuSans", "pointsize": 44, "extra_ops": ["-attenuate 0.6 +noise Gaussian"]},
  "noise_sigma": 12,
  "gates": {"cer_max": 0.02, "line_count": 1, "word_count": 3},
  "oracle": {"enabled": true, "psm": 7},
  "comparison_strategy": "cer+layout_gates",
  "skip_when_no_model": true
}
EOF
)" 'Clean render plus additive Gaussian noise, sigma ~12 grey levels (magick -attenuate 0.6 +noise Gaussian, calibrated 2026-08-06). Gate: CER <= 2%.'

name=070-ocr-small-font
mkdir -p "$OUT/$name"
"$MAGICK" -background white -fill black -font "$FONT" -pointsize 22 \
  label:'small print reads fine' -bordercolor white -border 24 -depth 8 \
  "$OUT/$name/input.png"
emit "$name" "$(
  cat <<EOF
{
  "name": "$name",
  "family": "070-ocr",
  "description": "One rendered line at ~16 px cap height (pointsize 22). Gates CER <= 3%. Requires the tessdata_best eng.traineddata cache (NV_OCR_TESSDATA).",
  "input": "input.png",
  "expected_text": "small print reads fine",
  "render": {"font": "DejaVuSans", "pointsize": 22, "extra_ops": ["-border 24"]},
  "gates": {"cer_max": 0.03, "line_count": 1, "word_count": 4},
  "oracle": {"enabled": true, "psm": 7},
  "comparison_strategy": "cer+layout_gates",
  "skip_when_no_model": true
}
EOF
)" 'One line at ~16 px cap height (pointsize 22). Gate: CER <= 3%.'

name=070-ocr-multiword-boxes
mkdir -p "$OUT/$name"
words=(alpha bravo charlie delta)
margin_x=40
margin_y=40
gap=28
x=$margin_x
rects=""
compose_args=()
lh=0
for i in "${!words[@]}"; do
  w="${words[$i]}"
  "$MAGICK" -background white -fill black -font "$FONT" -pointsize 44 label:"$w" -depth 8 "$TMP/w$i.png"
  lw=$("$MAGICK" "$TMP/w$i.png" -format '%w' info:)
  lh=$("$MAGICK" "$TMP/w$i.png" -format '%h' info:)
  geom=$("$MAGICK" "$TMP/w$i.png" -format '%@' info:)
  wh=${geom%%+*}
  W=${wh%x*}
  H=${wh#*x}
  rest=${geom#*+}
  X=${rest%%+*}
  Y=${rest##*+}
  left=$((x + X))
  top=$((margin_y + Y))
  right=$((left + W))
  bottom=$((top + H))
  rects="$rects{\"left\": $left, \"top\": $top, \"right\": $right, \"bottom\": $bottom}"
  if [ "$i" -lt $((${#words[@]} - 1)) ]; then
    rects="$rects, "
  fi
  compose_args+=("$TMP/w$i.png" -geometry "+$x+$margin_y" -composite)
  x=$((x + lw + gap))
done
cw=$((x - gap + margin_x))
ch=$((margin_y * 2 + lh))
"$MAGICK" -size "${cw}x${ch}" xc:white "${compose_args[@]}" -depth 8 "$OUT/$name/input.png"
emit "$name" "$(
  cat <<EOF
{
  "name": "$name",
  "family": "070-ocr",
  "description": "Four words composed at measured offsets; word_rects are exact ink bounding boxes from magick trim geometry. Gates every reported word rect at IoU >= 0.6 against word_rects. Requires the tessdata_best eng.traineddata cache (NV_OCR_TESSDATA).",
  "input": "input.png",
  "expected_text": "alpha bravo charlie delta",
  "render": {"font": "DejaVuSans", "pointsize": 44, "extra_ops": ["composed word labels at known offsets, inter-word gap ${gap}px"]},
  "word_rects": [$rects],
  "gates": {"cer_max": 0.0, "line_count": 1, "word_count": 4, "iou_min": 0.6},
  "oracle": {"enabled": true, "psm": 7},
  "comparison_strategy": "cer+layout_gates",
  "skip_when_no_model": true
}
EOF
)" 'Four words composed at measured offsets; word_rects in fixture.json are exact ink bounding boxes from magick %@ trim geometry. Gates: every reported word rect IoU >= 0.6 against word_rects; token offsets slice expected_text to the exact word.'

echo "fixtures written to $OUT"
