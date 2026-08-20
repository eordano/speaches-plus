#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path

SCHEMA_VERSION = 1

STATUS_ENUM = {"ok", "refused", "oom", "crashed", "skipped"}
ENGINE_ENUM = {"ours", "llama.cpp", "vllm"}
BACKEND_ENUM = {"wgpu-vulkan", "cuda", "vulkan"}

REQUIRED_ALWAYS = [
    "run_id", "schema_version", "started_at", "status", "commit_hash", "commit_date",
    "build_hash", "build_name", "engine", "backend", "model", "flags", "checkpoint",
    "device", "instrument", "inference_args", "vram_mb_before", "vram_mb_after",
    "wall_s", "log_path", "timing_basis",
]

REQUIRED_NULLABLE_METRICS = [
    "max_seq_allocated", "prefill_tokens", "generated_tokens", "ttft_s",
    "prefill_tok_s_avg", "prefill_tok_s_last_256", "decode_ms_per_tok_median",
    "decode_tok_s", "steps", "warmup_steps",
]

OPTIONAL_FIELDS = ["ppl", "null_delta_pct", "notes", "measure_lines"]

REQUIRED_DEVICE = ["gpu_name", "driver", "power_limit_w", "throttle_flags"]
REQUIRED_CHECKPOINT = ["repo", "revision"]
REQUIRED_INFERENCE = ["sampling", "batch"]
REQUIRED_PPL = ["instrument", "source_file", "corpus_path", "corpus_sha256", "tokens", "value", "accuracy"]

MEASURE_LINE_RE = re.compile(
    r"^\s*(NV-MEASURE v1|CTX-SCALING|CTX-PREFILL|GEN-ARM|PPL|PPL-SHUFFLED-CONTROL|"
    r"QWEN38-WGPU-PREFILL|QWEN38-WGPU-DECODE)\b.*"
)

KV_RE = re.compile(r"(\w[\w.]*)=([^\s\"]+|\"[^\"]*\")")


def fail(msg):
    print(f"parse_and_append: REFUSING to append: {msg}", file=sys.stderr)
    sys.exit(1)


def validate(row):
    for k in REQUIRED_ALWAYS:
        if k not in row or row[k] is None:
            fail(f"missing required field {k!r} in row {row.get('run_id')}")
    for k in REQUIRED_NULLABLE_METRICS:
        if k not in row:
            fail(f"metric key {k!r} absent (must be present, null allowed) in row {row.get('run_id')}")
    if row["status"] not in STATUS_ENUM:
        fail(f"status {row['status']!r} not in {sorted(STATUS_ENUM)}")
    if row["engine"] not in ENGINE_ENUM:
        fail(f"engine {row['engine']!r} not in {sorted(ENGINE_ENUM)}")
    if row["backend"] not in BACKEND_ENUM:
        fail(f"backend {row['backend']!r} not in {sorted(BACKEND_ENUM)}")
    if row["schema_version"] != SCHEMA_VERSION:
        fail(f"schema_version {row['schema_version']} != {SCHEMA_VERSION}")
    if not isinstance(row["flags"], dict):
        fail("flags must be an object of the NV_*/CLI flags exactly as passed")
    for k in REQUIRED_DEVICE:
        if k not in row["device"]:
            fail(f"device.{k} missing")
    for k in REQUIRED_CHECKPOINT:
        if k not in row["checkpoint"]:
            fail(f"checkpoint.{k} missing")
    for k in REQUIRED_INFERENCE:
        if k not in row["inference_args"]:
            fail(f"inference_args.{k} missing")
    if row.get("ppl") is not None:
        for k in REQUIRED_PPL:
            if k not in row["ppl"]:
                fail(f"ppl.{k} missing")
    if row["status"] == "ok":
        has_decode = row["decode_tok_s"] is not None
        has_prefill = row["prefill_tok_s_avg"] is not None or row["ttft_s"] is not None
        has_ppl = row.get("ppl") is not None
        if not (has_decode or has_prefill or has_ppl):
            fail("status=ok but the row carries no decode, prefill or ppl measurement")
    for k in ["ttft_s", "prefill_tok_s_avg", "prefill_tok_s_last_256",
              "decode_ms_per_tok_median", "decode_tok_s", "wall_s", "null_delta_pct"]:
        v = row.get(k)
        if v is not None and (not isinstance(v, (int, float)) or not math.isfinite(v)):
            fail(f"{k}={v!r} is not a finite number")
    return row


def parse_kv(line):
    out = {}
    for k, v in KV_RE.findall(line):
        out[k] = v.strip('"')
    return out


def fnum(d, *keys):
    for k in keys:
        if k in d:
            try:
                return float(d[k])
            except ValueError:
                return None
    return None


def fint(d, *keys):
    v = fnum(d, *keys)
    return int(v) if v is not None else None


def extract_measure_lines(log_text):
    return [ln.rstrip() for ln in log_text.splitlines() if MEASURE_LINE_RE.match(ln)]


def checkpoint_from_lines(lines, meta):
    for ln in lines:
        kv = parse_kv(ln)
        label = kv.get("checkpoint") or kv.get("model")
        if label and "@" in label:
            repo, _, rev = label.partition("@")
            if repo.startswith("models--"):
                repo = repo[len("models--"):].replace("--", "/")
            return {"repo": repo, "revision": rev}
    for ln in lines:
        m = re.search(r"(\S+/\S+)@([0-9a-f]{6,40})", ln)
        if m:
            return {"repo": m.group(1), "revision": m.group(2)}
    return {"repo": meta.get("checkpoint_repo", "unknown"), "revision": meta.get("checkpoint_revision", "unknown")}


def rows_ctx_scaling(lines, meta):
    rows = []
    for ln in lines:
        if not (ln.startswith("CTX-SCALING") or ln.startswith("CTX-PREFILL")):
            continue
        kv = parse_kv(ln)
        depth = fint(kv, "depth")
        if depth is None:
            continue
        median = fnum(kv, "median_ms_tok", "decode_median_ms_tok")
        tok_s = fnum(kv, "tok_s")
        if tok_s is None and median:
            tok_s = 1000.0 / median
        steps = fint(kv, "steps")
        r = {
            "prefill_tokens": depth,
            "generated_tokens": steps,
            "steps": steps,
            "warmup_steps": fint(kv, "warmup_steps"),
            "decode_ms_per_tok_median": median,
            "decode_tok_s": tok_s,
            "ttft_s": fnum(kv, "prefill_s") if "prefill_s" in kv else None,
            "prefill_tok_s_avg": fnum(kv, "prefill_tok_s", "prime_tok_s"),
            "prefill_tok_s_last_256": None,
            "timing_basis": "median",
            "measure_lines": [ln],
        }
        if r["prefill_tok_s_avg"] is None and "prefill_s" in kv and fnum(kv, "prefill_s"):
            r["prefill_tok_s_avg"] = depth / fnum(kv, "prefill_s")
        rows.append(r)
    return rows


def rows_prefill(lines, meta):
    rows = []
    for ln in lines:
        if not ln.startswith("QWEN38-WGPU-PREFILL"):
            continue
        kv = parse_kv(ln)
        n = fint(kv, "prompt_tokens")
        s = fnum(kv, "prefill_s")
        rows.append({
            "prefill_tokens": n,
            "generated_tokens": 1,
            "steps": None,
            "warmup_steps": None,
            "decode_ms_per_tok_median": None,
            "decode_tok_s": None,
            "ttft_s": s,
            "prefill_tok_s_avg": fnum(kv, "tok_s"),
            "prefill_tok_s_last_256": None,
            "timing_basis": "single_prefill_wall",
            "measure_lines": [ln],
        })
    return rows


def rows_gen_arm(lines, meta):
    rows = []
    for ln in lines:
        if not ln.startswith("GEN-ARM"):
            continue
        kv = parse_kv(ln)
        rows.append({
            "prefill_tokens": fint(kv, "prefill_tokens"),
            "generated_tokens": fint(kv, "gen"),
            "steps": fint(kv, "decode_steps"),
            "warmup_steps": 0,
            "decode_ms_per_tok_median": fnum(kv, "decode_median_ms"),
            "decode_tok_s": fnum(kv, "decode_tok_s"),
            "ttft_s": fnum(kv, "ttft_s"),
            "prefill_tok_s_avg": fnum(kv, "prefill_tok_s"),
            "prefill_tok_s_last_256": None,
            "timing_basis": "median",
            "measure_lines": [ln],
        })
    return rows


def rows_ppl(lines, meta):
    rows = []
    controls = {}
    for ln in lines:
        if ln.startswith("PPL-SHUFFLED-CONTROL"):
            kv = parse_kv(ln)
            parts = ln.split()
            controls[parts[1]] = fnum(kv, "ppl")
    for ln in lines:
        if not ln.startswith("PPL ") and not (ln.startswith("PPL") and not ln.startswith("PPL-")):
            continue
        parts = ln.split()
        if len(parts) < 3 or parts[0] != "PPL":
            continue
        kv = parse_kv(ln)
        family = parts[1]
        val = fnum(kv, "ppl")
        if val is None:
            continue
        acc = fnum(kv, "acc")
        rows.append({
            "prefill_tokens": None,
            "generated_tokens": None,
            "steps": None,
            "warmup_steps": None,
            "decode_ms_per_tok_median": None,
            "decode_tok_s": None,
            "ttft_s": None,
            "prefill_tok_s_avg": None,
            "prefill_tok_s_last_256": None,
            "timing_basis": "teacher_forced_nll",
            "ppl": {
                "instrument": meta.get("instrument", "unknown"),
                "source_file": meta.get("source_file", "unknown"),
                "corpus_path": meta.get("corpus_path"),
                "corpus_sha256": meta.get("corpus_sha256"),
                "tokens": fint(kv, "tokens"),
                "value": val,
                "accuracy": acc,
                "family": family,
                "shuffled_control_ppl": controls.get(family),
            },
            "measure_lines": [ln] + [c for c in lines if c.startswith("PPL-SHUFFLED-CONTROL")],
        })
    return rows


def rows_llama_bench(log_text, meta):
    rows = []
    entries = None
    for m in re.finditer(r"^\[\s*$.*?^\]\s*$", log_text, re.DOTALL | re.MULTILINE):
        try:
            entries = json.loads(m.group(0))
            break
        except json.JSONDecodeError:
            continue
    if entries is None:
        m = re.search(r"\[\s*\{.*\}\s*\]", log_text, re.DOTALL)
        if not m:
            return rows
        try:
            entries = json.loads(m.group(0))
        except json.JSONDecodeError:
            return rows
    for e in entries:
        n_prompt = int(e.get("n_prompt", 0))
        n_gen = int(e.get("n_gen", 0))
        depth = int(e.get("n_depth", 0) or 0)
        avg_ts = e.get("avg_ts")
        if avg_ts is None:
            continue
        line = (f"LLAMA-BENCH model={e.get('model_filename', '')} n_prompt={n_prompt} n_gen={n_gen} "
                f"n_depth={depth} avg_ts={avg_ts:.2f} stddev_ts={e.get('stddev_ts', 0):.2f} "
                f"backend={e.get('backends', '')} ngl={e.get('n_gpu_layers', '')}")
        r = {
            "prefill_tokens": (depth + n_prompt) if n_prompt else depth,
            "generated_tokens": n_gen if n_gen else (1 if n_prompt else None),
            "steps": n_gen if n_gen else None,
            "warmup_steps": None,
            "decode_ms_per_tok_median": (1000.0 / avg_ts) if n_gen else None,
            "decode_tok_s": avg_ts if n_gen else None,
            "ttft_s": (n_prompt / avg_ts) if (n_prompt and not n_gen) else None,
            "prefill_tok_s_avg": avg_ts if (n_prompt and not n_gen) else None,
            "prefill_tok_s_last_256": None,
            "timing_basis": "mean",
            "measure_lines": [line],
        }
        if n_prompt and n_gen:
            r["decode_tok_s"] = None
            r["decode_ms_per_tok_median"] = None
            r["notes_extra"] = "pp+tg combined arm; avg_ts mixes phases; stored raw only"
        rows.append(r)
    return rows


def rows_vllm_bench(log_text, meta):
    rows = []
    arm = None
    block = {}
    def flush():
        if arm is None or "median_tpot" not in block:
            return
        tpot = block["median_tpot"]
        rows.append({
            "prefill_tokens": arm.get("input_len"),
            "generated_tokens": arm.get("output_len"),
            "steps": arm.get("output_len"),
            "warmup_steps": None,
            "decode_ms_per_tok_median": tpot,
            "decode_tok_s": 1000.0 / tpot if tpot else None,
            "ttft_s": block.get("median_ttft") / 1000.0 if block.get("median_ttft") is not None else None,
            "prefill_tok_s_avg": (arm.get("input_len") / (block["median_ttft"] / 1000.0))
                if (arm.get("input_len") and block.get("median_ttft")) else None,
            "prefill_tok_s_last_256": None,
            "timing_basis": "median",
            "measure_lines": [f"VLLM-BENCH input_len={arm.get('input_len')} output_len={arm.get('output_len')} "
                              f"median_ttft_ms={block.get('median_ttft')} median_tpot_ms={block.get('median_tpot')} "
                              f"median_itl_ms={block.get('median_itl')} output_tok_s={block.get('output_throughput')}"],
        })
    for ln in log_text.splitlines():
        m = re.match(r"VLLM-ARM\s+input_len=(\d+)\s+output_len=(\d+)", ln)
        if m:
            flush()
            arm = {"input_len": int(m.group(1)), "output_len": int(m.group(2))}
            block = {}
            continue
        for key, pat in [("median_ttft", r"Median TTFT \(ms\):\s+([\d.]+)"),
                         ("median_tpot", r"Median TPOT \(ms\):\s+([\d.]+)"),
                         ("median_itl", r"Median ITL \(ms\):\s+([\d.]+)"),
                         ("output_throughput", r"Output token throughput \(tok/s\):\s+([\d.]+)")]:
            m = re.search(pat, ln)
            if m:
                block[key] = float(m.group(1))
    flush()
    return rows


PARSERS = {
    "ctx-scaling": rows_ctx_scaling,
    "prefill": rows_prefill,
    "gen-arm": rows_gen_arm,
    "ppl": rows_ppl,
}


def classify_failure(log_text, exit_code):
    if re.search(r"out of memory|OutOfMemory|CUDA_ERROR_OUT_OF_MEMORY|\bOOM\b|allocation of .* failed|DeviceLost", log_text):
        return "oom"
    if re.search(r"refuses|refusing|must never silently skip|unset it|not supported|no [^\n]{0,40}snapshot|snapshots dir [^\n]* missing|is not a directory|compile_fail|error\[E", log_text):
        return "refused"
    if exit_code != 0:
        return "crashed"
    return "skipped"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--mode", required=True, choices=list(PARSERS) + ["llama-bench", "vllm-bench"])
    ap.add_argument("--exit-code", type=int, required=True)
    ap.add_argument("--runs", default=str(Path(__file__).parent / "runs.jsonl"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    log_text = Path(args.log).read_text(errors="replace")
    lines = extract_measure_lines(log_text)

    if args.mode == "llama-bench":
        parsed = rows_llama_bench(log_text, meta)
    elif args.mode == "vllm-bench":
        parsed = rows_vllm_bench(log_text, meta)
    else:
        parsed = PARSERS[args.mode](lines, meta)

    ts = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    base_id = meta.get("run_id_base") or f"pr-{time.strftime('%Y%m%d-%H%M%S')}"
    checkpoint = meta.get("checkpoint") or checkpoint_from_lines(lines, meta)

    rows = []
    if not parsed:
        status = meta.get("status_override") or classify_failure(log_text, args.exit_code)
        tail = "\n".join(log_text.splitlines()[-15:])
        rows.append({
            "status": status,
            "prefill_tokens": meta.get("prefill_tokens"),
            "generated_tokens": None, "steps": None, "warmup_steps": None,
            "decode_ms_per_tok_median": None, "decode_tok_s": None, "ttft_s": None,
            "prefill_tok_s_avg": None, "prefill_tok_s_last_256": None,
            "timing_basis": "none",
            "notes": (meta.get("notes", "") + f" | no measurement lines parsed (exit={args.exit_code}); log tail: {tail}")[:2000],
        })
    else:
        for r in parsed:
            r["status"] = meta.get("status_override") or "ok"
            extra = r.pop("notes_extra", None)
            note = meta.get("notes", "")
            if extra:
                note = f"{note} | {extra}" if note else extra
            if note:
                r["notes"] = note
            rows.append(r)

    out = []
    for i, r in enumerate(rows):
        row = {
            "run_id": f"{base_id}-{i}",
            "schema_version": SCHEMA_VERSION,
            "started_at": meta.get("started_at", ts),
            "commit_hash": meta["commit_hash"],
            "commit_date": meta["commit_date"],
            "build_hash": meta["build_hash"],
            "build_name": meta["build_name"],
            "engine": meta["engine"],
            "backend": meta["backend"],
            "model": meta["model"],
            "flags": meta["flags"],
            "checkpoint": checkpoint,
            "device": meta["device"],
            "instrument": meta["instrument"],
            "inference_args": meta.get("inference_args", {"sampling": "greedy", "batch": 1}),
            "max_seq_allocated": meta.get("max_seq_allocated"),
            "vram_mb_before": meta["vram_mb_before"],
            "vram_mb_after": meta["vram_mb_after"],
            "wall_s": meta["wall_s"],
            "log_path": meta["log_path"],
            "ppl": None,
            "null_delta_pct": meta.get("null_delta_pct"),
        }
        row.update(r)
        validate(row)
        out.append(row)

    if args.dry_run:
        for row in out:
            print(json.dumps(row))
        return

    runs = Path(args.runs)
    runs.parent.mkdir(parents=True, exist_ok=True)
    with open(runs, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        for row in out:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
    print(f"parse_and_append: appended {len(out)} row(s) to {runs} "
          f"(statuses: {sorted({r['status'] for r in out})})")


if __name__ == "__main__":
    main()
