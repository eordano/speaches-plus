#!/usr/bin/env python3
"""Segment an nvtrace.tsv into decode steps and produce a per-kernel table.

Usage: analyze.py <trace.tsv> <batch-sizes,comma-sep> [vocab]

A decode step is delimited by the logits DtoH memcpy (b * vocab * 4 bytes).
Vocab defaults to 129280 (DSOCR); pass 262144 (or NVTRACE_VOCAB=262144) for
Gemma4 verify traces, where b is the verify width k (5 for k=4 spec rounds).
Marks are clustered into "arms" (separated by long gaps = prefill between rounds).
"""
import os, sys, collections, statistics

VOCAB = 129280

def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 12:
                continue
            recs.append(dict(
                kind=p[0], start=int(p[1]), end=int(p[2]), dur=int(p[3]),
                stream=int(p[4]), graph=int(p[5]), corr=int(p[6]),
                grid=p[7], block=p[8], bytes=int(p[9]), copy=p[10],
                name=p[11]))
    recs.sort(key=lambda r: r["start"])
    return recs

def short(n):
    n = n.split("(")[0]
    if "<" in n:
        n = n.split("<")[0]
    return n[:74]

def clusters(recs, marks):
    """Split mark indices into runs separated by unusually long gaps."""
    if len(marks) < 4:
        return [marks]
    gaps = [recs[b]["end"] - recs[a]["end"] for a, b in zip(marks, marks[1:])]
    med = statistics.median(gaps)
    out, cur = [], [marks[0]]
    for (a, b), g in zip(zip(marks, marks[1:]), gaps):
        if g > max(5 * med, med + 2_000_000):
            out.append(cur)
            cur = [b]
        else:
            cur.append(b)
    out.append(cur)
    return [c for c in out if len(c) >= 8]

def analyze_cluster(recs, marks, label, out):
    lo = max(1, len(marks) // 5)
    steps = list(zip(marks[lo:-1], marks[lo + 1:]))
    if not steps:
        return None

    per_name = collections.defaultdict(lambda: [0.0, 0])
    tot_busy = tot_wall = tot_dtoh = tot_gap_before = 0.0
    nsteps = len(steps)
    for a, c in steps:
        seg = recs[a + 1:c + 1]
        if not seg:
            continue
        tot_wall += recs[c]["end"] - recs[a]["end"]
        ivs = sorted((r["start"], r["end"]) for r in seg)
        busy = 0
        cs, ce = ivs[0]
        for s, e in ivs[1:]:
            if s > ce:
                busy += ce - cs
                cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += ce - cs
        tot_busy += busy
        tot_gap_before += ivs[0][0] - recs[a]["end"]
        for r in seg:
            key = short(r["name"]) if r["kind"] == "kernel" else \
                f"[{r['kind']} {r['copy']}] {r['bytes']//1024}KB"
            per_name[key][0] += r["dur"]
            per_name[key][1] += 1
            if r["kind"] == "memcpy" and r["copy"] == "DtoH":
                tot_dtoh += r["dur"]

    ms = lambda ns: ns / 1e6 / nsteps
    out.write(f"\n## {label}  ({nsteps} steady steps)\n\n")
    out.write(f"```\n")
    out.write(f"step wall (logits-copy end -> next)  : {ms(tot_wall):8.3f} ms   100.0%\n")
    out.write(f"GPU busy (union of all GPU activity) : {ms(tot_busy):8.3f} ms   "
              f"{tot_busy/tot_wall*100:5.1f}%\n")
    out.write(f"   - logits DtoH memcpy              : {ms(tot_dtoh):8.3f} ms   "
              f"{tot_dtoh/tot_wall*100:5.1f}%\n")
    out.write(f"   - compute kernels                 : {ms(tot_busy-tot_dtoh):8.3f} ms   "
              f"{(tot_busy-tot_dtoh)/tot_wall*100:5.1f}%\n")
    out.write(f"GPU IDLE inside the step             : {ms(tot_wall-tot_busy):8.3f} ms   "
              f"{(tot_wall-tot_busy)/tot_wall*100:5.1f}%\n")
    out.write(f"   - dead gap before first kernel    : {ms(tot_gap_before):8.3f} ms\n")
    out.write(f"```\n")

    rows = sorted(per_name.items(), key=lambda kv: -kv[1][0])
    out.write(f"\n| # | kernel | calls/step | ms/step | % busy | % wall |\n")
    out.write("|---|---|---|---|---|---|\n")
    for i, (k, (d, n)) in enumerate(rows[:28], 1):
        out.write(f"| {i} | `{k}` | {n/nsteps:.1f} | {ms(d):.4f} | "
                  f"{d/tot_busy*100:.1f}% | {d/tot_wall*100:.1f}% |\n")
    if len(rows) > 28:
        other = sum(d for _, (d, _) in rows[28:])
        oc = sum(n for _, (_, n) in rows[28:])
        out.write(f"| .. | (other {len(rows)-28} names) | {oc/nsteps:.1f} | {ms(other):.4f} | "
                  f"{other/tot_busy*100:.1f}% | {other/tot_wall*100:.1f}% |\n")
    nk = sum(n for _, (_, n) in rows)
    out.write(f"\nlaunches/step: {nk/nsteps:.1f}\n")
    return dict(wall=ms(tot_wall), busy=ms(tot_busy), dtoh=ms(tot_dtoh),
                idle=ms(tot_wall - tot_busy), launches=nk / nsteps, n=nsteps)

def graph_mode(recs, out):
    """Segment by CUDA graphId instead of logits-DtoH marks. Serving verify
    does device-side accept and never copies b*vocab logits to host, so the
    DtoH segmentation finds nothing there; every in-graph kernel carries its
    graphId, which is segmentation enough. Clusters replays on >200us gaps."""
    import statistics as st
    from collections import Counter
    by_graph = Counter(r["graph"] for r in recs)
    for g, _ in by_graph.most_common(8):
        if g == 0:
            continue
        ks = sorted((r for r in recs if r["graph"] == g), key=lambda r: r["start"])
        cl, cur = [], [ks[0]]
        for r in ks[1:]:
            if r["start"] - cur[-1]["end"] > 200_000:
                cl.append(cur)
                cur = [r]
            else:
                cur.append(r)
        cl.append(cur)
        if len(cl) < 4:
            continue
        spans = [(c[-1]["end"] - c[0]["start"]) / 1e6 for c in cl]
        busy = [sum(x["dur"] for x in c) / 1e6 for c in cl]
        idle = [s - b for s, b in zip(spans, busy)]
        out.write(
            f"\n# graph {g}: {len(ks)} kernels, {len(cl)} replays; span med "
            f"{st.median(spans):.3f} ms, busy med {st.median(busy):.3f} ms, "
            f"IDLE-WITHIN med {st.median(idle):.3f} ms\n"
        )
        agg = Counter()
        for r in ks:
            agg[r["name"]] += r["dur"]
        for name, tot in agg.most_common(8):
            out.write(f"  {tot / 1e6 / len(cl):8.3f} ms/replay  {name[:80]}\n")

if __name__ == "__main__":
    path = sys.argv[1]
    if sys.argv[2] == "graphs":
        recs = load(path)
        sys.stdout.write(f"# trace {path}: {len(recs)} records (graph mode)\n")
        graph_mode(recs, sys.stdout)
        sys.exit(0)
    bs = [int(x) for x in sys.argv[2].split(",")]
    if len(sys.argv) > 3:
        VOCAB = int(sys.argv[3])
    else:
        VOCAB = int(os.environ.get("NVTRACE_VOCAB", VOCAB))
    recs = load(path)
    out = sys.stdout
    out.write(f"# trace {path}: {len(recs)} records (vocab={VOCAB})\n")
    res = {}
    for b in bs:
        want = b * VOCAB * 4
        marks = [i for i, r in enumerate(recs)
                 if r["kind"] == "memcpy" and r["copy"] == "DtoH" and r["bytes"] == want]
        cls = clusters(recs, marks)
        out.write(f"\n# b={b}: {len(marks)} logits-DtoH marks in {len(cls)} arm(s)\n")
        for ci, c in enumerate(cls):
            r = analyze_cluster(recs, c, f"b={b} arm{ci}", out)
            if r:
                res[(b, ci)] = r
