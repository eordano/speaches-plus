#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models models/whisper-ct2

fetch() {
  local dest=$1 url=$2
  [[ -f "$dest" ]] && {
    echo "  [skip] $dest"
    return
  }
  echo "  [fetch] $url -> $dest"
  curl -fL --progress-bar -o "$dest" "$url"
}

fetch models/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx

fetch models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin

WHISPER_REPO=deepdml/faster-whisper-large-v3-turbo-ct2
for f in config.json model.bin tokenizer.json vocabulary.json preprocessor_config.json; do
  fetch "models/whisper-ct2/$f" \
    "https://huggingface.co/${WHISPER_REPO}/resolve/main/${f}"
done

fetch models/kokoro-v1.0.onnx \
  https://huggingface.co/speaches-ai/Kokoro-82M-v1.0-ONNX/resolve/main/model.onnx
fetch models/voices.bin \
  https://huggingface.co/speaches-ai/Kokoro-82M-v1.0-ONNX/resolve/main/voices.bin

fetch models/smart-turn-v3.onnx \
  https://huggingface.co/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx

DIARIZEN_DIR="models/diarizen-large-s80-v2"
mkdir -p "$DIARIZEN_DIR"
for f in pytorch_model.bin config.toml; do
  fetch "$DIARIZEN_DIR/$f" \
    "https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2/resolve/main/$f"
done
if [[ ! -f models/diarizen-segmentation.onnx ]]; then
  echo "  [info] models/diarizen-segmentation.onnx not present"
  echo "  [info] run: python3 scripts/export-diarizen-onnx.py $DIARIZEN_DIR models/diarizen-segmentation.onnx"
fi

fetch models/wespeaker-resnet293-LM.onnx \
  https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM/resolve/main/voxceleb_resnet293_LM.onnx

ls -lh models/ models/whisper-ct2/
