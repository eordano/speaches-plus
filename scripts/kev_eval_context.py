#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from kev.benchmark import evaluate_records
from kev.checkpoint import LoadOptions
from kev.data import load_records
from kev.model import training_context
from kev.predictors import LocalPredictor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-state", type=int, default=7552)
    args = parser.parse_args()
    records = load_records(args.data)
    predictor = LocalPredictor(args.run, "cuda", LoadOptions.from_env(),
                               context=training_context(args.max_state))
    report, _ = evaluate_records(records, predictor, args.out, skip_overlong=True)
    print(json.dumps({"run": args.run, "clean": report["clean"],
                      "coverage": report["coverage"]}), flush=True)


if __name__ == "__main__":
    main()
