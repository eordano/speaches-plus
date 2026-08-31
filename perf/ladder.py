#!/usr/bin/env python3
import argparse
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

PERF = Path(__file__).parent
DEFAULT_RUNS = PERF / "runs.jsonl"
HARNESS = PERF / "run-harness.sh"

IDENTICAL_WITHIN_EPS_MEANS_THE_ARM_NEVER_ENGAGED = "SUSPECT-INERT"

END_SAMPLE_IS_PRINTED_NOT_STORED = (
    "rows are append-only and notes are fixed at invocation; the end-of-run "
    "clock/power sample lives in this ladder transcript, not in the row"
)

ARM_IN_NOTES_RE = re.compile(r"\barm=([^\s;|,]+)")

SMI_SAMPLE_QUERY = "clocks.sm,power.draw,temperature.gpu"

def fail(msg):
    print(f"ladder: REFUSING: {msg}", file=sys.stderr)
    sys.exit(1)

def load_spec(path):
    spec = json.loads(Path(path).read_text())
    for k in ("ladder", "base_args", "arms", "depths"):
        if k not in spec:
            fail(f"spec is missing {k!r}")
    if "build-name" not in spec["base_args"]:
        fail("base_args must carry build-name; tabulation joins rows on it")
    names = [a["name"] for a in spec["arms"]]
    if len(names) != len(set(names)):
        fail(f"duplicate arm names in {names}")
    spec.setdefault("mode", "ours")
    spec.setdefault("base_env", {})
    spec.setdefault("depth_env", "NV_CTX_TOKENS")
    spec.setdefault("depths_per_run", False)
    spec.setdefault("notes", "")
    spec.setdefault("baseline_arm", names[0])
    if spec["baseline_arm"] not in names:
        fail(f"baseline_arm {spec['baseline_arm']!r} is not one of {names}")
    for arm in spec["arms"]:
        env = arm.get("env", {})
        if spec["depth_env"] in env or spec["depth_env"] in spec["base_env"]:
            fail(f"{spec['depth_env']} belongs to the depths axis, not to arm/base env")
        if "NV_WGPU_PROFILE" in env or "NV_WGPU_PROFILE" in spec["base_env"]:
            fail("NV_WGPU_PROFILE doubles step time; profiled absolutes never enter the store")
    if len(spec["arms"]) > 1 and not spec.get("null_pair") and not spec.get("null_pair_waived_because"):
        fail(
            "a multi-arm spec must name a null_pair -- two arms whose env resolves to the SAME code "
            "path -- or set null_pair_waived_because to say why not. Without one, an arm delta is "
            "read against no measured floor: identical-routing arms have been measured spreading "
            "5.5% run to run, ~17x the 0.32% two-identical-graphs figure this project used to quote, "
            "and a published +6.4% win once evaporated entirely under cooldown gating (83d396003)"
        )
    if spec.get("null_pair"):
        pair = spec["null_pair"]
        if not isinstance(pair, list) or len(pair) != 2:
            fail(f"null_pair must be two arm names, got {pair!r}")
        for n in pair:
            if n not in names:
                fail(f"null_pair names {n!r}, which is not one of {names}")
        if pair[0] == pair[1]:
            fail(
                f"null_pair names {pair[0]!r} twice; a null pair is two SEPARATELY RUN arms that "
                "resolve to the same path, not one arm compared with itself"
            )
    return spec

def smi_sample():
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={SMI_SAMPLE_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    parts = [p.strip() for p in out.stdout.strip().splitlines()[0].split(",")]
    if len(parts) != 3:
        return None
    return f"sm={parts[0]}MHz p={parts[1]}W t={parts[2]}C"

def smi_power_w():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip().splitlines()[0])
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError):
        return None

def wait_until_gpu_cools(threshold_w, timeout_s, interval_s):
    deadline = time.monotonic() + timeout_s
    while True:
        draw = smi_power_w()
        if draw is None:
            print("ladder: no nvidia-smi power reading; cooldown gate skipped")
            return
        if draw <= threshold_w:
            print(f"ladder: cooldown gate passed at {draw:.1f}W <= {threshold_w}W")
            return
        if time.monotonic() >= deadline:
            print(f"ladder: cooldown TIMEOUT after {timeout_s}s at {draw:.1f}W > {threshold_w}W; "
                  f"proceeding -- this arm may read heat-soaked, order-bias applies")
            return
        time.sleep(interval_s)

def invocations(spec):
    depth_values = ([str(d) for d in spec["depths"]] if spec["depths_per_run"]
                    else [",".join(str(d) for d in spec["depths"])])
    for arm in spec["arms"]:
        for dv in depth_values:
            env = dict(spec["base_env"])
            env.update(arm.get("env", {}))
            env[spec["depth_env"]] = dv
            yield arm["name"], dv, env

def build_cmd(spec, env, notes):
    cmd = [str(HARNESS), spec["mode"]]
    for k, v in spec["base_args"].items():
        cmd += [f"--{k}", str(v)]
    for k, v in sorted(env.items()):
        cmd += ["--env", f"{k}={v}"]
    cmd += ["--notes", notes]
    return cmd

def compose_notes(spec, arm_name, start_sample):
    parts = []
    if spec["notes"]:
        parts.append(spec["notes"])
    tag = f"ladder={spec['ladder']} arm={arm_name}"
    if start_sample:
        tag += f" [start {start_sample}]"
    parts.append(tag)
    return "; ".join(parts)

def run_ladder(spec, args):
    failures = []
    for arm_name, dv, env in invocations(spec):
        if args.cooldown_below_watts:
            wait_until_gpu_cools(args.cooldown_below_watts, args.cooldown_timeout,
                                 args.cooldown_interval)
        start_sample = smi_sample()
        cmd = build_cmd(spec, env, compose_notes(spec, arm_name, start_sample))
        print(f"\nladder: arm={arm_name} {spec['depth_env']}={dv}"
              + (f" start[{start_sample}]" if start_sample else ""))
        print("  " + " ".join(shlex.quote(c) for c in cmd))
        rc = subprocess.run(cmd).returncode
        end_sample = smi_sample()
        if end_sample:
            print(f"ladder: arm={arm_name} end[{end_sample}] ({END_SAMPLE_IS_PRINTED_NOT_STORED})")
        if rc != 0:
            failures.append((arm_name, dv, rc))
            print(f"ladder: arm={arm_name} {spec['depth_env']}={dv} exited {rc}; "
                  f"the harness recorded whatever status it could -- continuing the ladder")
    if failures:
        print(f"\nladder: {len(failures)} invocation(s) failed: {failures}")
    return failures

def print_dry_run(spec, args):
    plan = list(invocations(spec))
    print(f"ladder: DRY RUN -- {len(plan)} serial invocation(s), nothing executed")
    if args.cooldown_below_watts:
        print(f"ladder: each preceded by a cooldown gate: power.draw <= "
              f"{args.cooldown_below_watts}W (timeout {args.cooldown_timeout}s, "
              f"poll {args.cooldown_interval}s)")
    for arm_name, dv, env in plan:
        cmd = build_cmd(spec, env, compose_notes(spec, arm_name, start_sample=None))
        print("\n" + " ".join(shlex.quote(c) for c in cmd))

def tabulate(spec, args):
    match = args.tabulate_match or f"ladder={spec['ladder']}"
    build = spec["base_args"]["build-name"]
    metric = args.metric
    cells = {}
    arm_order = []
    with open(args.runs) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                fail(f"{args.runs}:{n} is not valid JSON -- repair the store first: "
                     f"python3 perf/merge-runs-jsonl.py {args.runs}")
            notes = str(r.get("notes") or "")
            if r.get("build_name") != build or match not in notes:
                continue
            m = ARM_IN_NOTES_RE.search(notes)
            depth = r.get("prefill_tokens")
            val = r.get(metric)
            if not m or depth is None or not isinstance(val, (int, float)):
                continue
            arm = m.group(1)
            if arm not in arm_order:
                arm_order.append(arm)
            key = (arm, int(depth))
            prev = cells.get(key)
            if prev is None or str(r.get("started_at")) > str(prev[1]):
                cells[key] = (float(val), r.get("started_at"))
    if not cells:
        print(f"ladder: no rows match build_name={build!r} notes~{match!r} with a numeric "
              f"{metric}; nothing to tabulate")
        return
    spec_order = [a["name"] for a in spec["arms"]]
    arms = [a for a in spec_order if a in arm_order] + [a for a in arm_order if a not in spec_order]
    depths = sorted({d for _, d in cells})
    baseline = spec["baseline_arm"] if spec["baseline_arm"] in arms else arms[0]
    header = ["arm"] + [str(d) for d in depths]
    body = []
    for arm in arms:
        line = [arm + (" (baseline)" if arm == baseline else "")]
        for d in depths:
            got = cells.get((arm, d))
            base = cells.get((baseline, d))
            if got is None:
                line.append("-")
            elif arm == baseline or base is None:
                line.append(f"{got[0]:g}")
            else:
                delta = (got[0] - base[0]) / base[0] * 100.0
                mark = (f" {IDENTICAL_WITHIN_EPS_MEANS_THE_ARM_NEVER_ENGAGED}"
                        if abs(delta) <= args.epsilon_pct else "")
                line.append(f"{got[0]:g} ({delta:+.1f}%){mark}")
        body.append(line)
    widths = [max(len(row[i]) for row in [header] + body) for i in range(len(header))]
    print(f"\nladder: {metric} by arm x depth (rows: build_name={build!r}, notes~{match!r}; "
          f"latest per cell)")
    for row in [header] + body:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())

def main():
    ap = argparse.ArgumentParser(
        description="declarative arm-ladder runner: {arms x depths} serially through "
                    "run-harness.sh, then a baseline-relative table from runs.jsonl")
    ap.add_argument("spec", help="JSON: ladder, base_args, base_env, arms:[{name,env}], "
                                 "depths, depth_env, depths_per_run, baseline_arm, notes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned harness invocations and exit")
    ap.add_argument("--tabulate-only", action="store_true",
                    help="skip execution; print the table from existing rows")
    ap.add_argument("--tabulate-match", metavar="SUBSTR",
                    help="notes substring selecting rows for the table "
                         "(default: this spec's ladder=<id> tag)")
    ap.add_argument("--metric", default="decode_tok_s")
    ap.add_argument("--epsilon-pct", type=float, default=0.1,
                    help="baseline-identical cells within this read SUSPECT-INERT")
    ap.add_argument("--cooldown-below-watts", type=float, metavar="W",
                    help="between invocations, wait until nvidia-smi power.draw is at or "
                         "under W: arm ORDER thermally biases late arms on a heat-soaked GPU")
    ap.add_argument("--cooldown-timeout", type=float, default=300)
    ap.add_argument("--cooldown-interval", type=float, default=5)
    ap.add_argument("--runs", default=str(DEFAULT_RUNS))
    args = ap.parse_args()

    spec = load_spec(args.spec)
    if args.dry_run:
        print_dry_run(spec, args)
        return
    if args.tabulate_only:
        tabulate(spec, args)
        return
    if not HARNESS.exists():
        fail(f"{HARNESS} is missing; ladders land rows only through run-harness.sh")
    failures = run_ladder(spec, args)
    tabulate(spec, args)
    if failures:
        sys.exit(1)

if __name__ == "__main__":
    main()
