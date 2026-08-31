#!/usr/bin/env python3
"""Materialize the acceptance-curve inputs from captured trajectories.

Reuses consolidate_trajectories.py's row selection and leakage-free prompt-group
holdout split (same filters, same seed-keyed ordering) and writes, into --out-dir:
train.jsonl and validation.jsonl (the {prompt, completion} payload rows a direct
POST /v1/fine_tuning/jobs takes via training_file/validation_file), plus
prompt_windows.json (one representative prompt per independent held-out window,
the shortest group row clearing the consuming suite's 8-token prompt floor via
a normalized-byte proxy -- windows with no clearing row are dropped, never
handed to the suite to panic on -- for NV_LORA_CURVE_PROMPTS) and meta.json
(window count, split digest, row counts). Exits 3 UNRESOLVED when the corpus
seats fewer than 9 floor-clearing independent held-out windows, because
docs/book/08.2-quality-verdicts.md says fewer cannot adjudicate.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import consolidate_trajectories as ct

REP_FLOOR_128_NORMALIZED_BYTES_CLEARS_THE_SUITE_8_TOKEN_PROMPT_FLOOR_AT_16_BYTES_PER_TOKEN = 128

def clears_the_suite_prompt_token_floor(row):
    normalized_bytes = len(ct.whitespace_and_case_normalized(row["prompt"]).encode())
    return (
        normalized_bytes
        >= REP_FLOOR_128_NORMALIZED_BYTES_CLEARS_THE_SUITE_8_TOKEN_PROMPT_FLOOR_AT_16_BYTES_PER_TOKEN
    )

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--min-completion-chars", type=int, default=16)
    ap.add_argument("--max-records", type=int, default=512)
    ap.add_argument("--keep-reasoning-rows", action="store_true")
    ap.add_argument("--require-label", action="store_true")
    ap.add_argument("--holdout-frac", type=float, default=0.1)
    ap.add_argument("--split-seed", default="holdout-v1")
    ap.add_argument(
        "--min-windows",
        type=int,
        default=ct.HOLDOUT_NEEDS_NINE_WINDOWS_BECAUSE_08_2_SAYS_FEWER_INDEPENDENT_WINDOWS_CANNOT_ADJUDICATE,
    )
    args = ap.parse_args(argv)

    rows = ct.read_trajectories(args.trajectories)
    patches = ct.read_label_patches(args.trajectories)
    data = ct.select_rows(rows, patches, args)
    print(f"{len(rows)} trajectory records -> {len(data)} training rows after filtering")
    if not data:
        print("nothing to train on", file=sys.stderr)
        return 1
    holdout, train, split_digest, holdout_group_sizes = (
        ct.split_holdout_without_prompt_leakage(data, args.holdout_frac, args.split_seed)
    )
    windows = []
    windows_dropped_under_prompt_floor = 0
    start = 0
    for size in holdout_group_sizes:
        group = holdout[start : start + size]
        start += size
        floored = [r for r in group if clears_the_suite_prompt_token_floor(r)]
        if not floored:
            windows_dropped_under_prompt_floor += 1
            continue
        windows.append(min(floored, key=lambda r: len(r["prompt"]))["prompt"])
    if len(windows) < args.min_windows:
        print(
            f"UNRESOLVED: {len(data)} deduped rows seat only {len(windows)} independent "
            f"held-out windows clearing the suite prompt floor "
            f"({windows_dropped_under_prompt_floor} of {len(holdout_group_sizes)} windows "
            f"dropped as sub-floor), under the {args.min_windows}-window minimum "
            f"(08.2: fewer independent windows cannot adjudicate)",
            file=sys.stderr,
        )
        return 3
    if not train:
        print("empty train split after the holdout draw", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)

    def write_jsonl(name, payload):
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for r in payload:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return path

    train_path = write_jsonl("train.jsonl", ct.payload_rows(train))
    val_path = write_jsonl("validation.jsonl", ct.payload_rows(holdout))

    prompts_path = os.path.join(args.out_dir, "prompt_windows.json")
    with open(prompts_path, "w", encoding="utf-8") as f:
        json.dump(windows, f, ensure_ascii=False, indent=1)

    meta = {
        "windows": len(windows),
        "windows_dropped_under_prompt_floor": windows_dropped_under_prompt_floor,
        "split_digest": split_digest,
        "split_seed": args.split_seed,
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "holdout_group_sizes": holdout_group_sizes,
        "trajectories": os.path.abspath(args.trajectories),
        "model": args.model,
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    print(
        f"{len(train)} train rows -> {train_path}; {len(holdout)} held-out rows in "
        f"{len(holdout_group_sizes)} independent windows -> {val_path}; "
        f"{len(windows)} floor-clearing window prompts "
        f"({windows_dropped_under_prompt_floor} dropped sub-floor) -> {prompts_path} "
        f"(split {split_digest})"
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
