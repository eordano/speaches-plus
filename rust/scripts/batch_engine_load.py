#!/usr/bin/env python3
"""Concurrency load driver for batch-engine-ab.sh.

Fires N chat completions at once and reports AGGREGATE decode throughput --
the metric the batched roofline ceiling bounds. Per-request latency is
reported alongside so a throughput win that comes purely from queueing
(each request slower, more of them in flight) is visible rather than hidden.
"""
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT = os.environ["BA_PORT"]
MODEL = os.environ["BA_MODEL"]
MAXTOK = int(os.environ.get("BA_MAXTOK", "128"))
ARM = os.environ.get("BA_ARM", "?")
PROMPT_TOKENS = int(os.environ.get("BA_PROMPT_TOKENS", "0"))
TIMEOUT_S = int(os.environ.get("BA_TIMEOUT", "600"))
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"

PROMPTS = [
    "Explain how a four-stroke engine works, covering all four strokes.",
    "Describe the water cycle from evaporation to precipitation.",
    "Explain how public-key cryptography lets strangers communicate securely.",
    "Walk through how a compiler turns source code into machine code.",
    "Explain why the sky is blue, in terms a curious ten-year-old would follow.",
    "Describe how a refrigerator moves heat out of an insulated box.",
    "Explain what a hash table is and why lookups are fast.",
    "Describe how vaccines train the immune system.",
]

FILLER_SENTENCES = [
    "The survey team recorded {n} distinct readings before the instrument drifted.",
    "In case {n}, the reviewer noted an inconsistency between the log and the ledger.",
    "Section {n} of the appendix lists the calibration constants used that season.",
    "Observation {n} was discarded because the reference channel had saturated.",
    "The {n}th trial repeated the protocol with the temperature held ten degrees lower.",
    "Footnote {n} clarifies which of the two conventions the authors adopted.",
    "Batch {n} arrived mislabelled and had to be re-identified from its spectra.",
]

def build_prompt(i):
    """Short prompt when BA_PROMPT_TOKENS=0, else that prompt preceded by
    roughly PROMPT_TOKENS tokens of varied filler. Length is approximate: we
    count words * 1.3, which is close enough for a context-regime sweep and
    avoids making the load driver depend on a tokenizer."""
    base = PROMPTS[i % len(PROMPTS)]
    if PROMPT_TOKENS <= 0:
        return base
    out, approx, n = [], 0, i * 1000
    while approx < PROMPT_TOKENS:
        s = FILLER_SENTENCES[n % len(FILLER_SENTENCES)].format(n=n)
        out.append(s)
        approx += int(len(s.split()) * 1.3)
        n += 1
    return (
        "Read the following record, then answer the question at the end.\n\n"
        + " ".join(out)
        + "\n\nQuestion: "
        + base
    )

def ask(i):
    """Returns (wall_s, tokens) or (wall_s, -1) when the server sheds (503).
    Admission shedding is real serving behavior, not a harness failure: at
    high concurrency the engine's waiting queue fills and returns 503, and a
    driver that crashes there would hide the capacity limit it just found."""
    body = json.dumps({
        "model": MODEL,
        "temperature": 0,
        "max_tokens": MAXTOK,
        "messages": [{"role": "user", "content": build_prompt(i)}],
    }).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            v = json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 503:
            return time.monotonic() - t0, -1, -1
        raise
    dt = time.monotonic() - t0
    u = v["usage"]
    return dt, u["completion_tokens"], u.get("prompt_tokens", -1)

def run(n):
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        res = list(ex.map(ask, range(n)))
    wall = time.monotonic() - t0
    shed = sum(1 for _, t, _ in res if t < 0)
    toks = sum(t for _, t, _ in res if t >= 0)
    lats = sorted(d for d, t, _ in res if t >= 0) or [0.0]
    ptoks = [p for _, t, p in res if t >= 0 and p >= 0]
    return wall, toks, lats, shed, (max(ptoks) if ptoks else -1)

SWEEP = [int(x) for x in os.environ.get("BA_PROMPT_SWEEP", "").split()]

run(1)

print(f"[{ARM}] req_ptok  concurrency  aggregate_tok/s   per_req_tok/s   p50_lat_s  p100_lat_s  tokens  shed  prompt_tok")
for want in (SWEEP or [PROMPT_TOKENS]):
    PROMPT_TOKENS = want
    for n in [int(x) for x in sys.argv[1:]]:
        wall, toks, lats, shed, ptok = run(n)
        agg = toks / wall
        served = n - shed
        per = agg / served if served else 0.0
        print(f"[{ARM}] {want:>8d}  {n:>11d}  {agg:>14.1f}  {per:>13.1f}  {lats[len(lats)//2]:>10.2f}  "
              f"{lats[-1]:>10.2f}  {toks:>6d}  {shed:>4d}  {ptok:>10d}", flush=True)
