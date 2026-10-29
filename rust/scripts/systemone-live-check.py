import argparse
import json
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "crates/nv-laya/tests/fixtures"
CHECKPOINTS = {"english": "laya", "multilingual": "laya-multilingual"}
UPSTREAM_FP16_PROBABILITY_TOLERANCE = 0.02
BURST_AGAINST_SEQUENTIAL_TOLERANCE = 0.02

def post(base, route, body, key):
    request = urllib.request.Request(base + route, json.dumps(body).encode(), {"Content-Type": "application/json"})
    if key:
        request.add_header("Authorization", f"Bearer {key}")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read()), (time.perf_counter() - started) * 1e3

def probabilities(answer):
    if "probabilities" in answer:
        return list(answer["probabilities"].values())
    return [1 - answer["noul"], answer["noul"]]

def decision(answer):
    p = probabilities(answer)
    return max(range(len(p)), key=lambda i: (p[i], -i))

def agreement(reference, candidate):
    same = total = 0
    gap = 0.0
    for qid, ref in reference.items():
        total += 1
        same += decision(ref) == decision(candidate[qid])
        gap = max(gap, max(abs(a - b) for a, b in zip(probabilities(ref), probabilities(candidate[qid]))))
    return same, total, gap

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("--key", default=None)
    parser.add_argument("--burst-repeats", type=int, default=4)
    parser.add_argument("--burst-threads", type=int, default=32)
    args = parser.parse_args()
    cases = json.loads((FIXTURES / "upstream_parity_cases.json").read_text())["cases"]
    torch = json.loads((FIXTURES / "upstream_reference_answers.json").read_text())["answers"]
    failures = []
    for model, name in CHECKPOINTS.items():
        bodies = [{"state": c["state"], "questions": c["questions"], "model": name} for c in cases]
        post(args.base, "/v1/systemone/detailed", bodies[0], args.key)
        sequential, latencies = [], []
        for body in bodies:
            result, ms = post(args.base, "/v1/systemone/detailed", body, args.key)
            sequential.append(result)
            latencies.append(ms)
        same = total = mismatched_tokens = 0
        gap = 0.0
        for case, result in zip(cases, sequential):
            reference = torch[model][case["case"]]
            if "reference_raised" in reference:
                continue
            mismatched_tokens += result["usage"]["input_tokens"] != reference["usage"]["input_tokens"]
            s, t, g = agreement(reference["answers"], result["answers"])
            same, total, gap = same + s, total + t, max(gap, g)
        print(f"{model} sequential-vs-torch: decisions {same}/{total} widest-gap {gap:.4f} token-mismatches {mismatched_tokens} p50-ms {statistics.median(latencies):.2f} served-by {sequential[0]['model']}")
        if same != total or mismatched_tokens or gap > UPSTREAM_FP16_PROBABILITY_TOLERANCE:
            failures.append(f"{model}: sequential answers leave the torch reference")

        jobs = [(i, body) for _ in range(args.burst_repeats) for i, body in enumerate(bodies)]
        started = time.perf_counter()
        with ThreadPoolExecutor(args.burst_threads) as pool:
            burst = list(pool.map(lambda job: (job[0], post(args.base, "/v1/systemone/detailed", job[1], args.key)[0]), jobs))
        wall = time.perf_counter() - started
        same = total = 0
        gap = 0.0
        for i, result in burst:
            s, t, g = agreement(sequential[i]["answers"], result["answers"])
            same, total, gap = same + s, total + t, max(gap, g)
        questions = sum(len(b["questions"]) for _, b in jobs)
        print(f"{model} burst-vs-sequential: requests {len(jobs)} threads {args.burst_threads} decisions {same}/{total} widest-gap {gap:.4f} wall-s {wall:.2f} questions-per-s {questions / wall:.0f}")
        if same != total or gap > BURST_AGAINST_SEQUENTIAL_TOLERANCE:
            failures.append(f"{model}: merged concurrent passes answer differently from sequential ones")

        strict, _ = post(args.base, "/v1/systemone", bodies[0], args.key)
        leaked = [qid for qid, a in strict["answers"].items() if "action" in a or (a["type"] == "noul" and "confidence" in a)]
        listed, _ = post(args.base, "/v1/systemone/detailed", {"requests": bodies[:3]}, args.key)
        print(f"{model} jev-shape: keys {sorted(strict)} leaked-fields {leaked} list-results {len(listed['results'])}")
        if leaked or sorted(strict) != ["answers", "model", "usage"] or len(listed["results"]) != 3:
            failures.append(f"{model}: the Jev wire shape is wrong")
    for failure in failures:
        print("FAIL", failure)
    sys.exit(1 if failures else 0)

if __name__ == "__main__":
    main()
