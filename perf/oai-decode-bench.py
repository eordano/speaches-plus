#!/usr/bin/env python3
import argparse
import json
import time
import urllib.request


def post(url, body, timeout=None):
    req = urllib.request.Request(
        url, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return urllib.request.urlopen(req, timeout=timeout)


def tokenize(base, text, model):
    for path, body, key in (
        ("/tokenize", {"content": text}, "tokens"),
        ("/tokenize", {"model": model, "prompt": text}, "tokens"),
    ):
        try:
            with post(base + path, body, timeout=120) as r:
                out = json.load(r)
                toks = out.get(key)
                if isinstance(toks, list) and toks and isinstance(toks[0], dict):
                    toks = [t["id"] for t in toks]
                if toks:
                    return toks
        except Exception:
            continue
    raise SystemExit("no tokenize endpoint answered; cannot build an exact-depth prompt")


def main():
    ap = argparse.ArgumentParser(
        description="uniform decode bench against any OpenAI-compatible server: "
        "exact-token-depth prompt via the server's own /tokenize, greedy streaming "
        "completion, decode tok/s measured first-token to last so prefill is excluded; "
        "prints VLLM-BENCH lines for run-harness exec --parse vllm-bench"
    )
    ap.add_argument("--base", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--depth", type=int, required=True)
    ap.add_argument("--emit", type=int, default=512)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--paragraph", default=(
        "The quick study of performance is the study of where time hides. "
        "Every engine pays for its memory traffic, its launch overhead, and its "
        "synchronization, and the craft is in knowing which of the three is the "
        "bill on this workload on this machine. "
    ))
    ap.add_argument("--corpus")
    ap.add_argument("--tokenizer-json")
    ap.add_argument("--prompt-file")
    args = ap.parse_args()

    if args.prompt_file:
        prompt = open(args.prompt_file).read()
        run_reps(args, prompt)
        return

    text = open(args.corpus).read() if args.corpus else args.paragraph * (
        2 + args.depth // max(1, len(args.paragraph) // 6)
    )
    if args.tokenizer_json:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(args.tokenizer_json)
        ids = tok.encode(text).ids
        if len(ids) < args.depth:
            raise SystemExit(f"prompt source holds {len(ids)} tokens < depth {args.depth}")
        prompt_ids = tok.decode(ids[: args.depth], skip_special_tokens=False)
    else:
        ids = tokenize(args.base, text, args.model)
        if len(ids) < args.depth:
            raise SystemExit(f"prompt source holds {len(ids)} tokens < depth {args.depth}")
        prompt_ids = ids[: args.depth]

    run_reps(args, prompt_ids)


def run_reps(args, prompt):
    for rep in range(args.reps):
        body = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.emit,
            "temperature": 0,
            "stream": True,
        }
        t_first = None
        t_last = None
        n = 0
        with post(args.base + "/v1/completions", body) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                now = time.perf_counter()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
        if n < 2:
            raise SystemExit(f"rep {rep}: only {n} stream events; nothing to time")
        tok_s = (n - 1) / (t_last - t_first)
        tpot = 1e3 * (t_last - t_first) / (n - 1)
        print(f"VLLM-ARM input_len={args.depth} output_len={n}", flush=True)
        print(f"Median TPOT (ms):    {tpot:.2f}", flush=True)
        print(f"Output token throughput (tok/s):    {tok_s:.1f}", flush=True)


if __name__ == "__main__":
    main()
