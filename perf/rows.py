#!/usr/bin/env python3
import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_RUNS = Path(__file__).parent / "runs.jsonl"

IDENTICAL_WITHIN_EPS_MEANS_THE_ARM_NEVER_ENGAGED = "SUSPECT-INERT"

VERDICT_WORDS = [
    "NEGATIVE", "INERT", "NULL", "banked", "neutral", "wins", "champion",
    "baseline", "regress", "tie", "knee", "gated", "crash", "flip", "timed out",
]

ARM_IN_NOTES_RE = re.compile(r"\barm=([^\s;|,]+)")

WHERE_RE = re.compile(r"^([\w.]+?)(!~|>=|<=|!=|~|=|>|<)(.*)$", re.S)

def fail(msg):
    print(f"rows: REFUSING: {msg}", file=sys.stderr)
    sys.exit(1)

def load_rows(path):
    rows = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                fail(
                    f"{path}:{n} is not valid JSON -- a half-merged store answers no "
                    f"query truthfully; repair it first: python3 perf/merge-runs-jsonl.py {path}"
                )
    return rows

def field(row, path):
    cur = row
    for part in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur

def as_num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def compile_where(token):
    m = WHERE_RE.match(token)
    if not m:
        fail(f"where token {token!r} does not parse; grammar: field(=|!=|~|!~|>=|<=|>|<)value")
    path, op, val = m.groups()

    def pred(row):
        v = field(row, path)
        if op == "~":
            return v is not None and re.search(val, str(v)) is not None
        if op == "!~":
            return v is None or re.search(val, str(v)) is None
        if op in ("=", "!="):
            hit = v is not None and (
                str(v) == val
                or (as_num(v) is not None and as_num(val) is not None and as_num(v) == as_num(val))
            )
            return hit if op == "=" else not hit
        if v is None:
            return False
        a, b = as_num(v), as_num(val)
        if a is None or b is None:
            a, b = str(v), val
        if op == ">=":
            return a >= b
        if op == "<=":
            return a <= b
        if op == ">":
            return a > b
        return a < b

    return pred

def add_filter_args(p):
    p.add_argument("--runs", default=str(DEFAULT_RUNS))
    p.add_argument("--build", help="regex on build_name")
    p.add_argument("--since", help="ISO date/datetime lower bound on started_at")
    p.add_argument("--note", help="substring of notes")
    p.add_argument("--flag", action="append", default=[], metavar="K[=V]",
                   help="flags.K present, or flags.K equal to V")
    p.add_argument("-w", "--where", action="append", default=[], metavar="EXPR",
                   help="field(=|!=|~|!~|>=|<=|>|<)value with dot paths like flags.NV_CTX_TOKENS or ppl.value")

def predicates(args, extra_tokens=()):
    preds = [compile_where(t) for t in list(args.where) + list(extra_tokens)]
    if args.build:
        rx = re.compile(args.build)
        preds.append(lambda r: rx.search(str(r.get("build_name") or "")) is not None)
    if args.since:
        preds.append(lambda r: str(field(r, "started_at") or "") >= args.since)
    if args.note:
        preds.append(lambda r: args.note in str(r.get("notes") or ""))
    for f in args.flag:
        k, eq, v = f.partition("=")
        if eq:
            preds.append(lambda r, k=k, v=v: field(r, "flags." + k) is not None
                         and str(field(r, "flags." + k)) == v)
        else:
            preds.append(lambda r, k=k: field(r, "flags." + k) is not None)
    return preds

def select(args, extra_tokens=()):
    rows = load_rows(args.runs)
    preds = predicates(args, extra_tokens)
    return [r for r in rows if all(p(r) for p in preds)]

def cell_text(v, cap):
    if v is None:
        s = "-"
    elif isinstance(v, float):
        s = f"{v:g}"
    elif isinstance(v, (dict, list)):
        s = json.dumps(v)
    else:
        s = str(v)
    s = s.replace("\n", " ")
    return s if len(s) <= cap else s[: cap - 3] + "..."

def print_table(headers, body):
    widths = [len(h) for h in headers]
    for row in body:
        for i, c in enumerate(row):
            widths[i] = max(widths[i], len(c))
    for row in [headers] + body:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())

def cmd_filter(args):
    for r in select(args):
        print(json.dumps(r, sort_keys=True))

def cmd_table(args):
    rows = select(args)
    cols = [c.strip() for c in args.cols.split(",") if c.strip()]
    if args.sort:
        rows.sort(key=lambda r: (field(r, args.sort) is None,
                                 as_num(field(r, args.sort))
                                 if as_num(field(r, args.sort)) is not None
                                 else str(field(r, args.sort) or "")))
    if args.limit:
        rows = rows[: args.limit]
    body = [[cell_text(field(r, c), args.max_width) for c in cols] for r in rows]
    print_table(cols, body)
    print(f"({len(rows)} rows)")

def latest_cell_per_key(rows, key, metric):
    cells = {}
    for r in sorted(rows, key=lambda r: str(r.get("started_at") or "")):
        k = field(r, key)
        m = field(r, metric)
        if k is None or as_num(m) is None:
            continue
        prev = cells.get(str(k))
        cells[str(k)] = {"key": k, "metric": float(m), "n": (prev["n"] + 1 if prev else 1),
                         "run_id": r.get("run_id"), "started_at": r.get("started_at")}
    return cells

def sorted_keys(cells):
    ks = list(cells.values())
    if all(as_num(c["key"]) is not None for c in ks):
        ks.sort(key=lambda c: as_num(c["key"]))
    else:
        ks.sort(key=lambda c: str(c["key"]))
    return [str(c["key"]) for c in ks]

def crossover_report(points, metric):
    signed = [(k, d, s) for k, d, s in points if s != 0]
    reports = []
    for (k1, d1, s1), (k2, d2, s2) in zip(signed, signed[1:]):
        if s1 != s2:
            reports.append(
                f"crossover: {metric} leader flips between {k1} ({d1:+.1f}%) and {k2} ({d2:+.1f}%)"
            )
    return reports

def cmd_compare(args):
    a_rows = select(args, args.a)
    b_rows = select(args, args.b)
    if not a_rows:
        fail(f"side A matched no rows (a={args.a})")
    if not b_rows:
        fail(f"side B matched no rows (b={args.b})")
    a_cells = latest_cell_per_key(a_rows, args.key, args.metric)
    b_cells = latest_cell_per_key(b_rows, args.key, args.metric)
    shared = [k for k in sorted_keys({**a_cells, **b_cells}) if k in a_cells and k in b_cells]
    if not shared:
        fail(f"no shared {args.key} values carry a non-null {args.metric} on both sides "
             f"(A has {sorted_keys(a_cells)}, B has {sorted_keys(b_cells)})")
    body = []
    points = []
    for k in shared:
        a, b = a_cells[k], b_cells[k]
        d = (b["metric"] - a["metric"]) / a["metric"] * 100.0
        inert = abs(d) <= args.epsilon_pct
        sign = 0 if inert else (1 if d > 0 else -1)
        points.append((k, d, sign))
        verdict = IDENTICAL_WITHIN_EPS_MEANS_THE_ARM_NEVER_ENGAGED if inert else (
            "b leads" if d > 0 else "a leads")
        body.append([str(k), f"{a['metric']:g}", f"{b['metric']:g}", f"{d:+.1f}%",
                     verdict, f"{a['n']}/{b['n']}"])
    print_table([args.key, f"a:{args.metric}", f"b:{args.metric}", "delta", "verdict", "n(a/b)"],
                body)
    inert_keys = [k for k, _, s in points if s == 0]
    if inert_keys == shared:
        print(f"{IDENTICAL_WITHIN_EPS_MEANS_THE_ARM_NEVER_ENGAGED} at EVERY shared {args.key}: "
              f"identical medians mean the config never switched the pipeline -- verify the flag "
              f"reached the builder before reading this as a kernel verdict")
    for line in crossover_report(points, args.metric):
        print(line)

def arm_label(row):
    m = ARM_IN_NOTES_RE.search(str(row.get("notes") or ""))
    if m:
        return m.group(1)
    flags = row.get("flags") or {}
    return ",".join(f"{k}={v}" for k, v in sorted(flags.items())) or "(no-flags)"

def verdicts_in(text):
    found = []
    for w in VERDICT_WORDS:
        if w.isupper() or " " in w:
            hit = re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", text)
        else:
            hit = re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", text, re.IGNORECASE)
        if hit:
            found.append(w)
    return found

def cmd_ledger(args):
    if not args.build:
        fail("ledger needs --build (a build_name regex naming the campaign family)")
    rows = sorted(select(args), key=lambda r: str(r.get("started_at") or ""))
    if not rows:
        fail(f"no rows match build_name regex {args.build!r}")
    arms = {}
    order = []
    for r in rows:
        label = arm_label(r)
        if label not in arms:
            arms[label] = []
            order.append(label)
        arms[label].append(r)
    print(f"ledger: build~{args.build}  {len(rows)} rows, {len(order)} arms, "
          f"{rows[0].get('started_at')} .. {rows[-1].get('started_at')}")
    for label in order:
        group = arms[label]
        vals = [as_num(field(r, args.metric)) for r in group]
        vals = [v for v in vals if v is not None]
        span = f"{min(vals):g}..{max(vals):g}" if vals else "-"
        notes_blob = " | ".join(str(r.get("notes") or "") for r in group)
        vwords = verdicts_in(notes_blob)
        statuses = sorted({r.get("status") for r in group})
        print(f"\narm={label}")
        print(f"  rows={len(group)}  {group[0].get('started_at')} .. {group[-1].get('started_at')}"
              f"  {args.metric}={span}  status={','.join(str(s) for s in statuses)}")
        print(f"  verdict words: {', '.join(vwords) if vwords else '(none in notes)'}")
        print(f"  first: {cell_text(group[0].get('notes'), args.max_width)}")
        if len(group) > 1:
            print(f"  last:  {cell_text(group[-1].get('notes'), args.max_width)}")

def main():
    ap = argparse.ArgumentParser(
        description="query, tabulate, compare and adjudicate perf/runs.jsonl rows")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("filter", help="print matching rows as JSONL")
    add_filter_args(p)
    p.set_defaults(fn=cmd_filter)

    p = sub.add_parser("table", help="print selected columns of matching rows")
    add_filter_args(p)
    p.add_argument("-c", "--cols",
                   default="started_at,build_name,model,backend,prefill_tokens,decode_tok_s,notes")
    p.add_argument("--sort")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-width", type=int, default=60)
    p.set_defaults(fn=cmd_table)

    p = sub.add_parser("compare",
                       help="join two filtered sides on a key; delta pct, inert tell, crossover")
    add_filter_args(p)
    p.add_argument("--a", action="append", default=[], metavar="EXPR", required=True,
                   help="extra where tokens selecting side A (the baseline)")
    p.add_argument("--b", action="append", default=[], metavar="EXPR", required=True,
                   help="extra where tokens selecting side B (the candidate)")
    p.add_argument("--key", default="prefill_tokens")
    p.add_argument("--metric", default="decode_tok_s")
    p.add_argument("--epsilon-pct", type=float, default=0.1,
                   help="|delta| at or under this reads SUSPECT-INERT: identical medians "
                        "mean the arm silently never engaged")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("ledger",
                       help="chronological adjudication story of a build_name family, by arm")
    add_filter_args(p)
    p.add_argument("--metric", default="decode_tok_s")
    p.add_argument("--max-width", type=int, default=110)
    p.set_defaults(fn=cmd_ledger)

    args = ap.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
