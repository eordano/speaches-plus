#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
NVK = REPO / "rust" / "scripts" / "nvk.sh"

GPU_BUSY_MIB = 2000

def gpu_memory_used_mib():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [int(x) for x in out.split() if x.strip()]

def refuse_if_gpu_busy(gate_mib):
    used = gpu_memory_used_mib()
    if any(u > gate_mib for u in used):
        print(f"GPU busy: memory.used={used} MiB exceeds gate {gate_mib} MiB", file=sys.stderr)
        sys.exit(2)

def run_arm(args, seed, out_dir, log_path):
    env = dict(os.environ)
    env.update(
        {
            "NVK_LANE": args.lane,
            "NVK_PKG": "nv-train",
            "NV_Q38_W4A4_AB_TEST": "1",
            "NV_Q38_AB_BASE": str(args.base),
            "NV_Q38_AB_DATA": str(args.data),
            "NV_Q38_AB_EVAL_DATA": str(args.eval_data),
            "NV_Q38_AB_OUT": str(out_dir),
            "NV_Q38_AB_STEPS": str(args.steps),
            "NV_Q38_AB_GRAD_LAYERS": str(args.grad_layers),
            "NV_Q38_AB_RANK": str(args.rank),
            "NV_Q38_AB_ALPHA": str(args.alpha),
            "NV_Q38_AB_LR": str(args.lr),
            "NV_Q38_AB_SEED": str(seed),
        }
    )
    cmd = [
        str(NVK),
        "test",
        "--no-fail-fast",
        "--test",
        "w4a4_eval_ab",
        "--",
        "--ignored",
        "--nocapture",
    ]
    with open(log_path, "w") as log:
        rc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        print(f"seed {seed}: nvk.sh test failed rc={rc}; tail of {log_path}:", file=sys.stderr)
        print("".join(open(log_path).readlines()[-30:]), file=sys.stderr)
        sys.exit(1)
    sidecar = pathlib.Path(out_dir) / "w4a4_eval.json"
    if not sidecar.is_file():
        print(f"seed {seed}: no {sidecar}; the eval arm silently ran nothing", file=sys.stderr)
        sys.exit(1)
    return json.loads(sidecar.read_text())

def per_row(v, n):
    return v / n

def main():
    p = argparse.ArgumentParser(
        description="A/B: score a Qwen3.8 LoRA adapter's held-out loss under the training "
        "forward (host-dequant bf16 dense GEMM) vs the serving forward (NVFP4-resident W4A4 "
        "GEMM with per-call activation quantization). One training run per seed yields both "
        "eval arms, so data/seed/hyperparameters are identical by construction. Verdict gates "
        "on aggregate eval-loss deltas in nats against the seed-to-seed variance band; the "
        "fixed-batch logit max-abs-diff is descriptive only."
    )
    p.add_argument("--base", required=True, type=pathlib.Path)
    p.add_argument("--data", required=True, type=pathlib.Path)
    p.add_argument("--eval-data", required=True, type=pathlib.Path)
    p.add_argument(
        "--out-root",
        type=pathlib.Path,
        default=pathlib.Path.home() / ".cache" / "nv-train" / "w4a4-ab" / time.strftime("%Y%m%d-%H%M%S"),
    )
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--grad-layers", type=int, default=2)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 8])
    p.add_argument("--lane", default="abarm")
    p.add_argument("--gpu-gate-mib", type=int, default=GPU_BUSY_MIB)
    args = p.parse_args()

    refuse_if_gpu_busy(args.gpu_gate_mib)
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        out_dir = args.out_root / f"seed-{seed}"
        log_path = args.out_root / f"seed-{seed}.log"
        s = run_arm(args, seed, out_dir, log_path)
        n = s["num_eval_examples"]
        r = {
            "seed": seed,
            "n_eval": n,
            "train_losses": s["train_losses"],
            "base_bf16": per_row(s["eval_loss_base_bf16"], n),
            "adapter_bf16": per_row(s["eval_loss_adapter_bf16"], n),
            "base_w4a4": per_row(s["eval_loss_base_w4a4"], n),
            "adapter_w4a4": per_row(s["eval_loss_adapter_w4a4"], n),
            "row0_logit_maxabsdiff": s["row0_logit_max_abs_diff_bf16_vs_w4a4"],
        }
        r["gain_bf16"] = r["adapter_bf16"] - r["base_bf16"]
        r["gain_w4a4"] = r["adapter_w4a4"] - r["base_w4a4"]
        r["serving_tax_base"] = r["base_w4a4"] - r["base_bf16"]
        r["serving_tax_adapter"] = r["adapter_w4a4"] - r["adapter_bf16"]
        r["excess_tax_on_adapter"] = r["serving_tax_adapter"] - r["serving_tax_base"]
        rows.append(r)

    band = max(abs(a["gain_bf16"] - b["gain_bf16"]) for a in rows for b in rows)
    mean_excess = sum(r["excess_tax_on_adapter"] for r in rows) / len(rows)

    hdr = (
        "seed  base_bf16  adpt_bf16  base_w4a4  adpt_w4a4  gain_bf16  gain_w4a4  "
        "excess_tax  logit_maxdiff"
    )
    print(hdr)
    for r in rows:
        print(
            f"{r['seed']:>4}  {r['base_bf16']:9.5f}  {r['adapter_bf16']:9.5f}  "
            f"{r['base_w4a4']:9.5f}  {r['adapter_w4a4']:9.5f}  {r['gain_bf16']:+9.5f}  "
            f"{r['gain_w4a4']:+9.5f}  {r['excess_tax_on_adapter']:+10.5f}  "
            f"{r['row0_logit_maxabsdiff']:.4e}"
        )
    print(
        f"units: nats per eval row (masked-mean CE per row, summed loss / {rows[0]['n_eval']} rows)"
    )
    print(f"seed band on gain_bf16 (reference-arm variance): {band:.5f} nats")
    print(f"mean excess serving tax on the adapter vs the base: {mean_excess:+.5f} nats")
    floor = 0.005
    gate = max(band, floor)
    verdict = (
        f"adapter gain SURVIVES the serving W4A4 forward (|excess| {abs(mean_excess):.5f} <= "
        f"gate {gate:.5f} = max(seed band, {floor} nats floor))"
        if abs(mean_excess) <= gate
        else f"adapter gain DIVERGES under the serving W4A4 forward (|excess| "
        f"{abs(mean_excess):.5f} > gate {gate:.5f} = max(seed band, {floor} nats floor))"
    )
    print(f"verdict: {verdict}")
    summary = {"rows": rows, "band_gain_bf16": band, "mean_excess_tax": mean_excess, "verdict": verdict}
    (args.out_root / "verdict.json").write_text(json.dumps(summary, indent=2))
    print(f"artifacts: {args.out_root}")

if __name__ == "__main__":
    main()
