#!/usr/bin/env bash
# ============================================================================
# ============================================================================
set -euo pipefail

SELF=$(readlink -f "${BASH_SOURCE[0]}")
REPO=$(cd "$(dirname "$SELF")/../.." && pwd) # repo root (parent of rust/)
cd "$REPO"

export NVK_LANE="${NVK_LANE:-loratrain6}"
export NVK_FEATURES="${NVK_FEATURES:-}"

echo "== 1. prepare data =========================================================="
cat >examples/lora/data.jsonl <<'JSONL'
{"text": "apple pie is good"}
{"prompt": "Bravo", "completion": " zulu nine"}
{"text": "cat dog fox run"}
JSONL
cat examples/lora/data.jsonl

echo "== 2. make a tiny example base model (torch-free) =========================="
python3 examples/lora/_make_example_base.py "$PWD/examples/lora/example-base"

echo "== 3. train a servable LoRA adapter ========================================"
rust/scripts/nvk-lora.sh train --base "$PWD/examples/lora/example-base" --data "$PWD/examples/lora/data.jsonl" --out "$PWD/examples/lora/out" --rank 8 --alpha 16 --target q,k,v,o,gate,up,down --steps 100 --lr 0.05 --seed 7

echo "== adapter produced ========================================================"
ls -l examples/lora/out/adapter_model.safetensors examples/lora/out/adapter_config.json

echo "== 4. check the adapter loads & routes ====================================="
rust/scripts/nvk-lora.sh check "$PWD/examples/lora/out"

echo "== DONE. Trained adapter -> examples/lora/out (serve via NV_LORA_ADAPTER_DIRS) =="
