"""Export a Kev checkpoint for speaches-plus's /v1/systemone Kev backend.

Kev stores a LoRA adapter and a pointer head over a Qwen3.5 base. speaches-plus
loads a plain Qwen3.5 checkpoint, so the adapter is merged into the base
weights here (W + (alpha/r) * B @ A, in fp32, cast back to the base dtype --
the same order kev.serve merges in), and the head and its calibration
temperature travel beside it as kev_head.safetensors and kev.json.

    python scripts/kev_export.py --run RUN_DIR --out OUT_DIR [--oracle records.jsonl]

--oracle scores the records with Kev's own fp32 predictor and writes
kev_oracle.json, the reference the Rust parity suite compares against. Run it
inside Kev's environment (kev/.venv, the repo on sys.path via --kev)."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
SERVE_MAX_STATE = SERVE_MAX_BRANCH = 8192
ADAPTER_PREFIX, BASE_PREFIX = "base_model.model.", "model.language_model."

def base_dir(name, revision):
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(name, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"]))

def merged_weights(base, run, scale):
    adapter = safe_open(run / "adapter_model.safetensors", "pt")
    pairs = {}
    for key in adapter.keys():
        stem, part = key.rsplit(".lora_", 1)
        assert stem.startswith(ADAPTER_PREFIX), f"unexpected adapter key {key}"
        pairs.setdefault(BASE_PREFIX + stem[len(ADAPTER_PREFIX):] + ".weight", {})[part[0]] = adapter.get_tensor(key)
    out, merged = {}, 0
    for shard in sorted(base.glob("*.safetensors")):
        with safe_open(shard, "pt") as f:
            for key in f.keys():
                w = f.get_tensor(key)
                if key in pairs:
                    a, b = pairs[key]["A"].float(), pairs[key]["B"].float()
                    w = (w.float() + scale * (b @ a)).to(w.dtype)
                    merged += 1
                out[key] = w.contiguous()
    assert merged == len(pairs), f"merged {merged} of {len(pairs)} adapted weights: base names do not match the adapter"
    return out, merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", help="model id served on /v1/systemone (default: the run directory's name)")
    ap.add_argument("--oracle", type=Path, help="labelled or unlabelled request jsonl to score with Kev for the parity suite")
    ap.add_argument("--kev", type=Path, help="Kev repository root, for --oracle")
    a = ap.parse_args()
    head = torch.load(a.run / "head.pt", map_location="cpu", weights_only=False)
    cfg = json.loads((a.run / "adapter_config.json").read_text())
    assert not cfg.get("use_dora") and not cfg.get("use_rslora") and not cfg.get("trainable_token_indices"), \
        "only plain LoRA merges into the base weights; this adapter needs Kev's own runtime"
    assert not head.get("option_isolation") and not head.get("special_embeddings"), \
        "option isolation and trained special embeddings are not part of the exported format"
    base = base_dir(head["base"], head.get("base_revision"))
    a.out.mkdir(parents=True, exist_ok=True)
    weights, merged = merged_weights(base, a.run, cfg["lora_alpha"] / cfg["r"])
    save_file(weights, a.out / "model.safetensors", metadata={"format": "pt"})
    for leaf in ("config.json", "generation_config.json"):
        if (base / leaf).exists():
            shutil.copy(base / leaf, a.out / leaf)
    for leaf in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(a.run / leaf, a.out / leaf)
    save_file({k: v.float().contiguous() for k, v in head["head"].items()}, a.out / "kev_head.safetensors")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(a.run / "tokenizer.json"))
    ids = [tok.token_to_id(t) for t in SPECIAL]
    assert None not in ids, f"a Kev delimiter is missing from the tokenizer: {dict(zip(SPECIAL, ids))}"
    meta = {"name": a.name or a.run.name, "base": head["base"], "base_revision": head.get("base_revision"),
            "temperature": float(head.get("temperature", 1.0)), "head_dim": int(head["head_dim"]),
            "special": dict(zip(["state", "question", "option", "option_end", "decide"], ids)),
            "max_state": SERVE_MAX_STATE, "max_branch": SERVE_MAX_BRANCH, "merged_weights": merged}
    (a.out / "kev.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("name", "base", "temperature", "merged_weights")}))
    if a.oracle:
        sys.path.insert(0, str(a.kev))
        from kev.checkpoint import LoadOptions
        from kev.predictors import LocalPredictor
        from kev.model import SERVE_MAX_BRANCH as MB, SERVE_MAX_STATE as MS, SERVE_MAX_PACKED as MP
        pred = LocalPredictor(str(a.run), "cuda" if torch.cuda.is_available() else "cpu", LoadOptions(dtype=torch.float32),
                              context={"max_state": MS, "max_branch": MB, "max_packed": MP})
        cases = []
        for line in a.oracle.read_text().splitlines():
            r = json.loads(line)
            req = {"state": r["state"], "model": meta["name"], "questions": {
                qid: {k: v for k, v in q.items() if k in ("type", "instructions", "criteria")} for qid, q in r["questions"].items()}}
            placeholder = {"choice": lambda q: next(iter(q["criteria"])), "noul": lambda q: False, "score": lambda q: 0}
            got = pred({"state": req["state"], "questions": {
                qid: {**q, "label": placeholder[q["type"]](q), "src": "oracle"} for qid, q in req["questions"].items()}})
            cases.append({"request": req, "probabilities": got["probabilities"]})
        (a.out / "kev_oracle.json").write_text(json.dumps(cases, ensure_ascii=False) + "\n")
        print(f"oracle: {len(cases)} requests")

if __name__ == "__main__":
    main()
