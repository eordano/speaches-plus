#!/usr/bin/env bash

set -euo pipefail
cd "$(dirname "$0")/../.."

LANE="${RB_LANE:-wf30}"
PORT="${RB_PORT:-8380}"
BOOT_TIMEOUT="${RB_BOOT_TIMEOUT:-600}"
OUTDIR="${RB_OUTDIR:-docs/measurements/$(date +%F)-rocq-repoint}"
NVK="rust/scripts/nvk.sh"
mkdir -p "$OUTDIR"

line=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits)
util=$(echo "$line" | cut -d, -f1 | tr -d ' ')
mem=$(echo "$line" | cut -d, -f2 | tr -d ' ')
echo "GPU gate: util=${util}% mem=${mem}MiB"
if [ "$util" -gt 5 ] || [ "$mem" -gt 1024 ]; then
  echo "GPU-GATE-FAIL: contended" >&2
  exit 1
fi

LOG="$OUTDIR/server-prof.txt" # .txt, not .log: .gitignore excludes *.log and
setsid env NV_PROF_CHAT=1 UVICORN_PORT="$PORT" RUST_LOG=info \
  NVK_LANE="$LANE" NVK_PKG=speaches-plus NVK_JOBS=8 \
  "$NVK" run --release --bin speaches-plus >"$LOG" 2>&1 &
PG=$!
trap 'kill -TERM -- "-$PG" 2>/dev/null || true' EXIT

pick_chat='import sys,json
ids=[m["id"] for m in json.load(sys.stdin)["data"]]
bad=("embed","tts","whisper","wespeaker","voice","speaker")
chat=[i for i in ids if not any(b in i.lower() for b in bad)]
print((chat or ids)[0])'
t=0 model_id=""
until model_id=$(curl -fsS "http://127.0.0.1:$PORT/v1/models" 2>/dev/null |
  python3 -c "$pick_chat" 2>/dev/null) && [ -n "$model_id" ]; do
  t=$((t + 5))
  if [ "$t" -ge "$BOOT_TIMEOUT" ] || ! kill -0 "$PG" 2>/dev/null; then
    echo "server did not boot" >&2
    tail -n 20 "$LOG" >&2
    exit 1
  fi
  sleep 5 &
  wait $!
done
echo "server up: $model_id"

python3 - "$PORT" "$model_id" "$OUTDIR/bench.json" <<'EOF'
import json, sys, time, urllib.request

port, model, out = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{port}/v1/chat/completions"

PROMPTS = [
    "Explain in detail how a four-stroke engine works, covering intake, compression, combustion and exhaust.",
    "Describe the water cycle from evaporation to precipitation, step by step, in detail.",
    "Explain how public-key cryptography enables two strangers to communicate securely over an open channel.",
    "Walk through how a compiler turns source code into machine code, phase by phase.",
]

def ask(prompt, max_tokens):
    body = json.dumps({
        "model": model, "temperature": 0, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as r:
        v = json.load(r)
    return time.monotonic() - t0, v["usage"]["completion_tokens"]

# prime: graph capture + allocator warm, excluded from the bench
ask("Say OK.", 8)

rounds = []
for _ in range(2):
    per_prompt = []
    for p in PROMPTS:
        wall_one, _ = ask(p, 1)
        wall_full, toks = ask(p, 256)
        per_prompt.append({
            "completion_tokens": toks - 1,
            "decode_wall_s": round(wall_full - wall_one, 4),
        })
    rounds.append({"per_prompt": per_prompt})

json.dump({"rounds": rounds,
           "method": "client-side wall; decode_wall_s = wall(full) - wall(max_tokens=1); "
                     "tokens = completion_tokens - 1 (conservative for the SOL tripwire)"},
          open(out, "w"), indent=1)
print("bench.json written:", out)
EOF

kill -TERM -- "-$PG" 2>/dev/null || true
trap - EXIT
for i in 1 2 3 4 5 6; do
  kill -0 "$PG" 2>/dev/null || break
  sleep 5 &
  wait $!
done
kill -KILL -- "-$PG" 2>/dev/null || true

grep -c "GRAPHED SUMMARY" "$LOG" || true
echo "artifacts: $LOG , $OUTDIR/bench.json"
echo "now point rocq/gen/measured.json at them and re-run rocq/gen/run.sh"
