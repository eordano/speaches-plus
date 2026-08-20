#!/usr/bin/env python3
"""End-to-end chat throughput benchmark.

Runs N requests against the running server, reports per-run TTFT,
sustained decode tok/s, and aggregate stats.

Usage:
    python3 scripts/bench.py                          # default: 3 runs, 256 tok each
    python3 scripts/bench.py --runs 5 --max-tokens 512
    python3 scripts/bench.py --temp 0                 # engages Eagle3 spec-decode
    SPEACHES_URL=http://host:18080 python3 scripts/bench.py
"""

import argparse, json, os, statistics, sys, time
import requests

BASE = os.environ.get("SPEACHES_URL", "http://127.0.0.1:8000")

PROMPT = (
    "Write a detailed technical essay about the history of GPU compute, "
    "starting from CUDA 1.0 and going through modern Blackwell architecture. "
    "Include specific architectural details, memory hierarchies, and the "
    "evolution of tensor cores. Be thorough and complete."
)

def run_once(model, temperature, max_tokens):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
    }
    t_send = time.perf_counter()
    resp = requests.post(f"{BASE}/v1/chat/completions",
                         json=body, stream=True, timeout=600)
    resp.raise_for_status()

    first_token_t = None
    last_token_t = None
    delta_count = 0
    char_count = 0
    completion_tokens = None
    for line in resp.iter_lines(decode_unicode=True, chunk_size=64):
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        usage = chunk.get("usage") or {}
        if usage.get("completion_tokens") is not None:
            completion_tokens = usage["completion_tokens"]
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
        if delta:
            now = time.perf_counter()
            if first_token_t is None:
                first_token_t = now
            last_token_t = now
            delta_count += 1
            char_count += len(delta)

    total = (last_token_t or time.perf_counter()) - t_send
    ttft = (first_token_t - t_send) if first_token_t else None
    decode_time = (last_token_t - first_token_t) if first_token_t and last_token_t and delta_count > 1 else 0.0
    tokens = completion_tokens or delta_count
    return {
        "ttft": ttft,
        "total": total,
        "decode_time": decode_time,
        "tokens": tokens,
        "deltas": delta_count,
        "chars": char_count,
        "tok_per_s_decode": (tokens / decode_time) if decode_time > 0 else 0.0,
        "tok_per_s_overall": tokens / total if total > 0 else 0.0,
    }

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temp", type=float, default=0.7)
    p.add_argument("--model", default="qwen3.6")
    p.add_argument("--warmup", type=int, default=1, help="number of warmup runs (not reported)")
    args = p.parse_args()

    label = "Eagle3 spec-decode" if args.temp == 0 else "autoregressive"
    print(f"benchmark: {label}  model={args.model}  temp={args.temp}  max_tokens={args.max_tokens}")
    print(f"           url={BASE}  warmup={args.warmup}  runs={args.runs}\n")

    for i in range(args.warmup):
        print(f"  warmup {i+1}/{args.warmup}...", end="", flush=True)
        try:
            r = run_once(args.model, args.temp, args.max_tokens)
            print(f"  {r['tokens']} tok in {r['total']:.2f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            return

    results = []
    for i in range(args.runs):
        try:
            r = run_once(args.model, args.temp, args.max_tokens)
        except Exception as e:
            print(f"run {i+1}: FAILED: {e}")
            return
        results.append(r)
        print(f"  run {i+1}: ttft={r['ttft']*1000:6.0f}ms  decode={r['decode_time']:5.2f}s  "
              f"tok={r['tokens']:3d}  rate={r['tok_per_s_decode']:5.1f}tok/s  "
              f"overall={r['tok_per_s_overall']:5.1f}tok/s")

    print()
    print("aggregate:")
    print(f"  TTFT:           {statistics.median(r['ttft']*1000 for r in results):.0f}ms median   "
          f"({min(r['ttft']*1000 for r in results):.0f} - {max(r['ttft']*1000 for r in results):.0f}ms)")
    print(f"  decode tok/s:   {statistics.median(r['tok_per_s_decode'] for r in results):.1f} median   "
          f"({min(r['tok_per_s_decode'] for r in results):.1f} - {max(r['tok_per_s_decode'] for r in results):.1f})")
    print(f"  overall tok/s:  {statistics.median(r['tok_per_s_overall'] for r in results):.1f} median")
    print(f"  tokens:         {statistics.median(r['tokens'] for r in results):.0f} median per run")

if __name__ == "__main__":
    main()
