#!/usr/bin/env python3
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

GROUP_ON_WHAT_THE_ROW_RECORDS_NOT_ON_WHAT_SOMEONE_TYPED = """
build_name is free text. It carries hundreds of distinct values, one model wears
several of them, and a single prefix mixes our rows with rival rows -- grouping on it
has twice produced a published figure that was wrong in sign or in kind. Every row
already records what actually identifies a measurement: checkpoint {repo, revision},
engine, backend, and the resolved flags. This tool groups on those and never on
build_name, so the convenient query is the correct one.

A COMPARISON IS ONLY A COMPARISON IF THE AXES MATCH. Two cells differing in backend,
or in whether a drafter ran, are not a result -- they are two different experiments.
Speculative decoding in particular is worth a large multiple and is invisible in a
tok/s number, so this tool refuses to print a cross-speculation margin unless asked,
and labels every margin it does print with the axes that were held fixed.
"""

SPECULATION_OFF_VALUES = ("", "none", "off", "0", "false")

def speculating(row):
    drafter = str((row.get("flags") or {}).get("NV_DRAFTER", "")).strip().lower()
    return drafter not in SPECULATION_OFF_VALUES

def bucket_of(row):
    ck = row.get("checkpoint") or {}
    return (
        str(ck.get("repo") or "?"),
        str(row.get("engine") or "?"),
        str(row.get("backend") or "?"),
        "spec" if speculating(row) else "plain",
    )

def load(path, model_filter):
    cells = defaultdict(list)
    malformed = 0
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            malformed += 1
            continue
        depth, rate = row.get("prefill_tokens"), row.get("decode_tok_s")
        if depth is None or not isinstance(rate, (int, float)):
            continue
        b = bucket_of(row)
        if model_filter and model_filter.lower() not in b[0].lower():
            continue
        cells[(b, int(depth))].append(
            (float(rate), str(row.get("build_name") or "?"), str(row.get("started_at") or "")[:10],
             (row.get("device") or {}).get("power_limit_w"))
        )
    return cells, malformed

def render(cells, depths, show_provenance):
    buckets = sorted({b for b, _ in cells})
    label = lambda b: f"{b[1]}/{b[2]}/{b[3]}"
    width = max([len(label(b)) for b in buckets] + [6])
    by_model = defaultdict(list)
    for b in buckets:
        by_model[b[0]].append(b)
    for model in sorted(by_model):
        print(f"\n=== {model}")
        print(f"{'engine/backend/spec':<{width}}" + "".join(f"{d:>10}" for d in depths))
        for b in sorted(by_model[model], key=lambda x: (x[1] != "ours", x[1], x[2], x[3])):
            line = f"{label(b):<{width}}"
            for d in depths:
                got = cells.get((b, d))
                line += f"{max(got)[0]:>10.2f}" if got else f"{'-':>10}"
            print(line)
            if show_provenance:
                for d in depths:
                    got = cells.get((b, d))
                    if got:
                        rate, name, when, pl = max(got)
                        watts = "" if pl is None else f"  @{pl:g}W"
                        print(f"{'':<{width}}  {d}: {rate:.2f}  {name}  {when}{watts}  ({len(got)} rows)")

A_SHALLOW_CELL_SLOWER_THAN_A_DEEPER_ONE_IS_PHYSICALLY_IMPOSSIBLE_AND_NAMES_A_STALE_ROW = """
Decode rate falls monotonically with depth: a deeper context reads strictly more KV per
step. So a bucket whose shallow cell is SLOWER than its own deeper cell is not a curve,
it is a stale or broken row that nothing in the store marks as superseded. This is how
`gemma4-26b-a4b moe wgpu kv-ring` reads 1.5 tok/s at 122880 while the same model reads
42.5 at the deeper 196608 -- an honest measurement of a since-fixed arm that still reads
as current. The check costs nothing and needs no schema change, so it runs always.

A HIT IS A PROMPT TO LOOK, NOT A VERDICT. The bucket does not include the arm, so a
deliberately slow arm and a fast one land together and read as an inversion -- the
gemma4-31b pair below is cuda-eager against cuda-serving on the same day, which is a real
arm difference and not a stale row. Read the two build_names the report prints before
concluding anything.
"""

UNKNOWN_CHECKPOINT_REPOS = ("unknown", "?", "none", "")

def report_unattributed(cells):
    unknown = sum(len(v) for (b, _), v in cells.items() if b[0].lower() in UNKNOWN_CHECKPOINT_REPOS)
    total = sum(len(v) for v in cells.values())
    if not unknown:
        return 0
    print(f"\nstandings: {unknown} of {total} rows ({100 * unknown / total:.1f}%) record no checkpoint "
          f"and are bucketed as 'unknown'. They CANNOT be compared against a named checkpoint, and two "
          f"eras of the same model land in different buckets -- so an inversion or a regression spanning "
          f"the boundary is invisible here. The emitters are chiefly the synthetic-fill depth ladders; "
          f"fixing them upstream is worth more than any query built on top.")
    return unknown

AN_INVERSION_INSIDE_THE_NULL_FLOOR_IS_NOISE_NOT_A_DEFECT = """
Identically-routed arms have been measured spreading 5.5% run to run (83d396003), and
cells from different eras drift further, so a shallow cell a few percent under its deeper
neighbour says nothing. Only inversions well outside that floor are physically impossible
rather than merely unlucky, so the default threshold sits above it and hits are ranked by
severity -- otherwise a 28x defect prints below ninety-odd 2% coincidences and is missed.
"""

INVERSION_THRESHOLD_PCT_SITS_ABOVE_THE_MEASURED_5_5_PCT_NULL_SPREAD = 10.0

def impossible_depth_inversions(cells, threshold_pct):
    by_bucket = defaultdict(dict)
    for (b, d), rows in cells.items():
        by_bucket[b][d] = max(rows)
    found = []
    for b, series in by_bucket.items():
        ds = sorted(series)
        for i, shallow in enumerate(ds):
            for deeper in ds[i + 1:]:
                srate, sname, swhen, _spl = series[shallow]
                drate, dname, dwhen, _dpl = series[deeper]
                excess = (drate / srate - 1.0) * 100.0 if srate > 0 else float("inf")
                if excess > threshold_pct:
                    found.append((excess, b, shallow, srate, sname, swhen, deeper, drate, dname, dwhen))
    return sorted(found, reverse=True)

def report_inversions(cells, threshold_pct):
    found = impossible_depth_inversions(cells, threshold_pct)
    if not found:
        return 0
    print(f"\nstandings: {len(found)} IMPOSSIBLE depth inversion(s) beyond {threshold_pct:g}% -- a "
          f"shallower cell slower than a deeper one in the same bucket, by more than run-to-run spread "
          f"explains. Suspect a stale or broken row before trusting either cell:")
    for excess, b, sd, sr, sn, sw, dd, dr, dn, dw in found[:10]:
        print(f"  {excess:7.0f}%  {b[1]}/{b[2]}/{b[3]} [{b[0]}]")
        print(f"           {sd:>7} {sr:8.2f}  {sn}  {sw}   <-- slower than")
        print(f"           {dd:>7} {dr:8.2f}  {dn}  {dw}")
    return len(found)

WGPU_OVER_AN_API_IS_THE_SAME_API_FOR_FAIRNESS_BUT_NOT_THE_SAME_PATH_FOR_BLAME = """
Our `wgpu-vulkan` and llama.cpp's `vulkan` are the same graphics API on the same card,
so refusing to compare them would rule out the backend-fair comparison the campaign
exists to make. They are NOT the same code path -- ours crosses wgpu and naga first --
so the raw backend of each side stays in the printed label and only the API family is
held fixed.
"""

def backend_family(backend):
    b = backend.lower()
    for api in ("vulkan", "metal", "cuda", "rocm", "dx12"):
        if api in b:
            return api
    return b

def compare(cells, depths, ours_key, rival_key, allow_cross_axis):
    fixed, differing = [], []
    for axis, a, b in (("backend", backend_family(ours_key[2]), backend_family(rival_key[2])),
                       ("speculation", ours_key[3], rival_key[3])):
        (fixed if a == b else differing).append(axis)
    if differing and not allow_cross_axis:
        print(f"\nstandings: REFUSING to print a margin across differing {' and '.join(differing)} "
              f"({label_pair(ours_key, rival_key)}). Pass --allow-cross-axis to see it anyway, and if "
              f"you publish it, say in the same sentence as the number that {differing[0]} differs.",
              file=sys.stderr)
        return 1
    held = ", ".join(fixed) if fixed else "nothing"
    warn = "" if not differing else f"  [UNMATCHED on {' and '.join(differing)} -- not an engine comparison]"
    print(f"\n{label_pair(ours_key, rival_key)}   holding {held} fixed{warn}")
    for d in depths:
        o, r = cells.get((ours_key, d)), cells.get((rival_key, d))
        if not o or not r:
            print(f"  {d:>7}: ours={'-' if not o else f'{max(o)[0]:.2f}'}  "
                  f"rival={'-' if not r else f'{max(r)[0]:.2f}'}   NO CELL")
            continue
        (ov, _, _, opl), (rv, _, _, rpl) = max(o), max(r)
        if opl is not None and rpl is not None and opl != rpl:
            if not allow_cross_axis:
                print(f"  {d:>7}: ours {ov:7.2f} @{opl:g}W  rival {rv:7.2f} @{rpl:g}W   MARGIN WITHHELD "
                      f"-- {THE_CARDS_POWER_LIMIT_FLAPS_BETWEEN_250W_CONFIG_AND_300W_DEFAULT_SO_LIMITS_MUST_MATCH}")
                continue
            print(f"  {d:>7}: ours {ov:7.2f}  rival {rv:7.2f}   {(ov / rv - 1) * 100:+7.1f}%"
                  f"   [UNMATCHED power limit {opl:g}W vs {rpl:g}W -- not the same card]")
            continue
        print(f"  {d:>7}: ours {ov:7.2f}  rival {rv:7.2f}   {(ov / rv - 1) * 100:+7.1f}%")
    return 0

def label_pair(a, b):
    return f"{a[1]}/{a[2]}/{a[3]} [{a[0]}]  vs  {b[1]}/{b[2]}/{b[3]} [{b[0]}]"

THE_CARDS_POWER_LIMIT_FLAPS_BETWEEN_250W_CONFIG_AND_300W_DEFAULT_SO_LIMITS_MUST_MATCH = (
    "a 50W limit change moves sustained clocks by more than most optimizations, so a margin "
    "across it measures the flap, not the code; giga's gpu-power-cap.nix intends 250W (PSU OCP) "
    "but Xid recovery and service-ordering gaps have left long 300W windows, and every row "
    "records which one it got (device.power_limit_w)"
)

def report_power_limits(cells):
    limits = defaultdict(int)
    for rows in cells.values():
        for row in rows:
            limits[row[3]] += 1
    known = {k: v for k, v in limits.items() if k is not None}
    if len(known) <= 1:
        return 0
    counts = ", ".join(f"{v} @{k:g}W" for k, v in sorted(known.items()))
    print(f"\nstandings: the store spans MULTIPLE power limits ({counts}). "
          f"{THE_CARDS_POWER_LIMIT_FLAPS_BETWEEN_250W_CONFIG_AND_300W_DEFAULT_SO_LIMITS_MUST_MATCH}. "
          f"Best-cell picks above may be irreproducible under the currently enforced limit.")
    return len(known)

A_RIVAL_COMPARISON_ALWAYS_CROSSES_CHECKPOINTS_BECAUSE_THEIR_4BIT_FILE_IS_NOT_OUR_4BIT_FILE = """
Our NVFP4 checkpoint and llama.cpp's Q4_K_M GGUF are different repos by construction, so
requiring a single checkpoint would forbid the only comparison anyone wants. Both are
printed in full instead, because WHICH 4-bit scheme each side ran is part of the result:
the schemes differ in bits per weight and therefore in the bytes a memory-bound decode
has to move.
"""

def resolve(cells, spec):
    parts = spec.split("/")
    if len(parts) != 3:
        print(f"standings: --ours/--rival take engine/backend/spec, got {spec!r}", file=sys.stderr)
        sys.exit(2)
    matches = {b for b, _ in cells if (b[1], b[2], b[3]) == tuple(parts)}
    if not matches:
        print(f"standings: no rows for {spec!r}. Available: "
              f"{sorted({f'{b[1]}/{b[2]}/{b[3]}' for b, _ in cells})}", file=sys.stderr)
        sys.exit(2)
    if len(matches) > 1:
        print(f"standings: {spec!r} spans several checkpoints {sorted({b[0] for b in matches})}; "
              f"narrow with --model so the row behind each cell is unambiguous", file=sys.stderr)
        sys.exit(2)
    return sorted(matches)[0]

def main():
    ap = argparse.ArgumentParser(
        description="standings from perf/runs.jsonl grouped on checkpoint/engine/backend/speculation")
    ap.add_argument("--runs", default=str(Path(__file__).parent / "runs.jsonl"))
    ap.add_argument("--model", help="substring of the checkpoint repo, e.g. Qwen3.8")
    ap.add_argument("--depths", default="256,8192,32768,172032")
    ap.add_argument("--ours", help="engine/backend/spec, e.g. ours/wgpu-vulkan/plain")
    ap.add_argument("--rival", help="engine/backend/spec, e.g. llama.cpp/vulkan/plain")
    ap.add_argument("--allow-cross-axis", action="store_true",
                    help="print a margin even when backend or speculation differs between the two")
    ap.add_argument("--provenance", action="store_true", help="show the row behind every cell")
    ap.add_argument("--inversion-threshold-pct", type=float,
                    default=INVERSION_THRESHOLD_PCT_SITS_ABOVE_THE_MEASURED_5_5_PCT_NULL_SPREAD,
                    help="report a shallow-slower-than-deep cell only beyond this excess")
    args = ap.parse_args()

    depths = [int(d) for d in args.depths.split(",") if d.strip()]
    cells, malformed = load(args.runs, args.model)
    if malformed:
        print(f"standings: {malformed} unparseable line(s) skipped; repair with "
              f"python3 perf/merge-runs-jsonl.py {args.runs}", file=sys.stderr)
    if not cells:
        print("standings: no rows with a numeric decode_tok_s matched", file=sys.stderr)
        return 1
    render(cells, depths, args.provenance)
    report_unattributed(cells)
    report_inversions(cells, args.inversion_threshold_pct)
    report_power_limits(cells)
    if args.ours and args.rival:
        return compare(cells, depths, resolve(cells, args.ours), resolve(cells, args.rival),
                       args.allow_cross_axis)
    if args.ours or args.rival:
        print("standings: --ours and --rival must be given together", file=sys.stderr)
        return 2
    return 0

if __name__ == "__main__":
    sys.exit(main())
