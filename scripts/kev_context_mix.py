#!/usr/bin/env python3
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re


TARGETS = (512, 1024, 2048, 4096, 6144, 7552)
SHARES = (10, 15, 20, 20, 15, 20)
SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")
MARKER = "recent:\n"


def token_ids(tokenizer, value):
    return tokenizer(SPECIAL.sub(r"<¦\1¦>", value), add_special_tokens=False).input_ids


def state_parts(state):
    prefix, marker, recent = state.partition(MARKER)
    if not marker:
        raise ValueError("CCCP Kev state has no recent-history boundary")
    return prefix + marker, recent


def clipped_state(tokenizer, prefix, recent, full_length, target):
    if full_length <= target:
        return prefix + recent
    prefix_length = len(token_ids(tokenizer, prefix)) + 1
    if prefix_length > target:
        return None
    recent_ids = token_ids(tokenizer, recent)
    keep = min(len(recent_ids), target - prefix_length)
    while keep >= 0:
        tail = tokenizer.decode(recent_ids[-keep:] if keep else [], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
        result = prefix + tail
        if len(token_ids(tokenizer, result)) + 1 <= target:
            return result
        keep -= 1
    return None


def band(length):
    for target in TARGETS:
        if length <= target:
            return str(target)
    return "over"


def json_line(record):
    return json.dumps(record, ensure_ascii=True) + "\n"


def mix(records, tokenizer, seed=0, forbidden=frozenset()):
    prepared = []
    strata = defaultdict(list)
    for index, record in enumerate(records):
        prefix, recent = state_parts(record["state"])
        label = next(iter(record["questions"].values()))["label"]
        agent = prefix.split("\n", 1)[0]
        item = {"index": index, "record": record, "prefix": prefix, "recent": recent,
                "label": label, "prefix_length": len(token_ids(tokenizer, prefix)) + 1,
                "full_length": len(token_ids(tokenizer, record["state"])) + 1}
        prepared.append(item)
        strata[(agent, label == "no_tool")].append(item)

    assigned = {}
    for key in sorted(strata):
        pool = strata[key][:]
        random.Random(f"{seed}/{key}").shuffle(pool)
        total = len(pool)
        for target, share in zip(TARGETS[:-1], SHARES[:-1]):
            count = round(total * share / 100)
            eligible = [item for item in pool if item["prefix_length"] <= target
                        and item["full_length"] >= target]
            chosen = eligible[:count]
            if len(chosen) < count:
                chosen += [item for item in pool if item["prefix_length"] <= target
                           and item not in chosen][:count - len(chosen)]
            for item in chosen:
                assigned[item["index"]] = target
            chosen_ids = {item["index"] for item in chosen}
            pool = [item for item in pool if item["index"] not in chosen_ids]
        for item in pool:
            assigned[item["index"]] = TARGETS[-1]

    output = []
    seen = set()
    counts = Counter()
    label_counts = defaultdict(Counter)
    raised = 0
    skipped = 0
    for item in prepared:
        requested = assigned[item["index"]]
        for target in TARGETS[TARGETS.index(requested):]:
            state = clipped_state(tokenizer, item["prefix"], item["recent"],
                                  item["full_length"], target)
            if state is None or state in seen or state in forbidden:
                continue
            seen.add(state)
            record = dict(item["record"], state=state)
            output.append(record)
            actual = len(token_ids(tokenizer, state)) + 1
            counts[band(actual)] += 1
            label_counts[band(actual)][item["label"]] += 1
            raised += target != requested
            break
        else:
            skipped += 1
    report = {"count": len(output), "bands": dict(counts),
              "labels_by_band": {key: dict(value) for key, value in label_counts.items()},
              "raised_to_avoid_duplicate": raised, "skipped_unresolvable": skipped}
    return output, report


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", type=Path)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen3.5-0.8B-Base",
        revision="dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68",
        local_files_only=True,
    )
    rows = [json.loads(line) for line in args.input.open() if line.strip()]
    if args.limit is not None:
        rows = random.Random(args.seed).sample(rows, args.limit)
    forbidden = set()
    if args.exclude:
        forbidden = {json.loads(line)["state"] for line in args.exclude.open() if line.strip()}
    output, report = mix(rows, tokenizer, args.seed, forbidden)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as stream:
        for row in output:
            stream.write(json_line(row))
    report["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"count": report["count"], "bands": report["bands"],
                      "raised_to_avoid_duplicate": report["raised_to_avoid_duplicate"],
                      "skipped_unresolvable": report["skipped_unresolvable"]}))


if __name__ == "__main__":
    main()
