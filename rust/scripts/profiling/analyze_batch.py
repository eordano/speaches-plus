#!/usr/bin/env python3
"""Segment an nvtrace.tsv from a LOADED batched server into decode steps.

analyze.py's DtoH segmentation assumes ONE logits copy per step (b*vocab*4).
The batched stepper copies logits row by row -- src/oapi/batch_chat.rs:339-344
`row()` does `logits.i((i,..)).to_vec1::<f32>()` per sequence -- so a B-way
decode step ends with B separate vocab*4-byte DtoH memcpys back to back. This
script clusters those bursts, so one "step" is one batched decode, and B is
measured (marks per burst) rather than assumed.

Usage: analyze_batch.py <trace.tsv> [vocab=262144] [top=12] [only_B]

`only_B` restricts the steady-state window to steps whose burst size equals
that B. batch_engine_load.py always fires a discarded warm run(1) first, so a
single trace carries both a B=1 and a B=16 population; contrasting them gives
the marginal cost of a sequence directly, with no second server boot.

Reports per steady-state step: wall, GPU-busy union, idle-within, the biggest
idle gaps, per-kernel ms/step, and the per-stream split.
"""
import sys
import collections
import statistics as st

def load(path, vocab_bytes):
    names = {}
    recs = []
    marks = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) < 12:
                continue
            kind = p[0]
            start = int(p[1]); end = int(p[2]); dur = int(p[3])
            stream = int(p[4]); graph = int(p[5])
            nbytes = int(p[9]); copy = p[10]; name = p[11]
            nid = names.get(name)
            if nid is None:
                nid = len(names)
                names[name] = nid
            recs.append((start, end, dur, stream, graph, kind, nbytes, copy, nid))
    recs.sort()
    for i, r in enumerate(recs):
        if r[5] == "memcpy" and r[7] == "DtoH" and r[6] == vocab_bytes:
            marks.append(i)
    inv = [None] * len(names)
    for n, i in names.items():
        inv[i] = n
    return recs, marks, inv

def short(n):
    n = n.split("(")[0]
    if "<" in n:
        n = n.split("<")[0]
    return n[:72]

def bursts(recs, marks):
    """Group logits-row DtoH marks into per-step bursts."""
    if len(marks) < 4:
        return []
    gaps = [recs[b][1] - recs[a][1] for a, b in zip(marks, marks[1:])]
    med = st.median(gaps)
    thr = max(10 * med, 200_000)
    out, cur = [], [marks[0]]
    for (a, b), g in zip(zip(marks, marks[1:]), gaps):
        if g > thr:
            out.append(cur)
            cur = [b]
        else:
            cur.append(b)
    out.append(cur)
    return out

def union_busy(seg):
    if not seg:
        return 0, []
    ivs = sorted((r[0], r[1]) for r in seg)
    busy = 0
    gaps = []
    cs, ce = ivs[0]
    for s, e in ivs[1:]:
        if s > ce:
            busy += ce - cs
            gaps.append((s - ce, ce))
            cs, ce = s, e
        else:
            ce = max(ce, e)
    busy += ce - cs
    return busy, gaps

def main():
    path = sys.argv[1]
    vocab = int(sys.argv[2]) if len(sys.argv) > 2 else 262144
    top = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    only_b = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    recs, marks, names = load(path, vocab * 4)
    print(f"# trace {path}: {len(recs)} records, "
          f"{len(marks)} logits-row DtoH marks of {vocab*4} B")
    if not recs:
        return
    bs = bursts(recs, marks)
    print(f"# {len(bs)} decode-step bursts; burst sizes (=B) histogram: "
          f"{dict(collections.Counter(len(b) for b in bs).most_common(8))}")
    if len(bs) < 6:
        print("# too few bursts to segment")
        return

    lo = max(1, len(bs) // 5)
    steps = list(zip(bs[lo:-1], bs[lo + 1:]))
    if only_b:
        steps = [(a, c) for a, c in steps if len(a) == only_b and len(c) == only_b]
        print(f"# filtered to B={only_b}: {len(steps)} steps")
    nsteps = len(steps)
    if nsteps == 0:
        return

    per_name = collections.defaultdict(lambda: [0.0, 0])
    per_stream = collections.defaultdict(float)
    tot_wall = tot_busy = tot_dtoh = tot_gap_before = 0.0
    all_gaps = []
    bsizes = []
    for a, c in steps:
        i0, i1 = a[-1], c[-1]
        seg = recs[i0 + 1:i1 + 1]
        if not seg:
            continue
        bsizes.append(len(c))
        tot_wall += recs[i1][1] - recs[i0][1]
        busy, gaps = union_busy(seg)
        tot_busy += busy
        all_gaps.extend(g for g, _ in gaps)
        tot_gap_before += min(r[0] for r in seg) - recs[i0][1]
        for r in seg:
            key = short(names[r[8]]) if r[5] == "kernel" else \
                f"[{r[5]} {r[7]}] {r[6]//1024}KB"
            per_name[key][0] += r[2]
            per_name[key][1] += 1
            per_stream[r[3]] += r[2]
            if r[5] == "memcpy" and r[7] == "DtoH":
                tot_dtoh += r[2]

    def ms(ns):
        return ns / 1e6 / nsteps

    print(f"\n## steady-state batched decode step ({nsteps} steps, "
          f"median B={st.median(bsizes):.0f})\n")
    print(f"step wall (last logits copy -> next)  : {ms(tot_wall):9.3f} ms  100.0%")
    print(f"GPU busy (union of all GPU activity)  : {ms(tot_busy):9.3f} ms  "
          f"{tot_busy/tot_wall*100:5.1f}%")
    print(f"   - logits DtoH memcpy               : {ms(tot_dtoh):9.3f} ms  "
          f"{tot_dtoh/tot_wall*100:5.1f}%")
    print(f"   - compute kernels                  : {ms(tot_busy-tot_dtoh):9.3f} ms  "
          f"{(tot_busy-tot_dtoh)/tot_wall*100:5.1f}%")
    print(f"GPU IDLE inside the step              : {ms(tot_wall-tot_busy):9.3f} ms  "
          f"{(tot_wall-tot_busy)/tot_wall*100:5.1f}%")
    print(f"   - dead gap before first kernel     : {ms(tot_gap_before):9.3f} ms")
    if all_gaps:
        all_gaps.sort(reverse=True)
        print(f"   - idle gaps/step: {len(all_gaps)/nsteps:.1f}  "
              f"median {st.median(all_gaps)/1e3:.2f} us  "
              f"max {all_gaps[0]/1e3:.1f} us  "
              f"top-10 sum {sum(all_gaps[:10*nsteps])/1e6/nsteps:.3f} ms/step")
    tps = 1e9 / (tot_wall / nsteps) * st.median(bsizes)
    print(f"implied aggregate decode rate         : {tps:.1f} tok/s "
          f"(B={st.median(bsizes):.0f} / step wall)")

    rows = sorted(per_name.items(), key=lambda kv: -kv[1][0])
    print(f"\n| # | kernel | calls/step | ms/step | % busy | % wall |")
    print("|---|---|---|---|---|---|")
    for i, (k, (d, n)) in enumerate(rows[:top], 1):
        print(f"| {i} | `{k}` | {n/nsteps:.1f} | {ms(d):.4f} | "
              f"{d/tot_busy*100:.1f}% | {d/tot_wall*100:.1f}% |")
    if len(rows) > top:
        other = sum(d for _, (d, _) in rows[top:])
        oc = sum(n for _, (_, n) in rows[top:])
        print(f"| .. | (other {len(rows)-top} names) | {oc/nsteps:.1f} | "
              f"{ms(other):.4f} | {other/tot_busy*100:.1f}% | {other/tot_wall*100:.1f}% |")
    nk = sum(n for _, (_, n) in rows)
    print(f"\nlaunches/step: {nk/nsteps:.1f}")
    print("per-stream busy ms/step: " + "  ".join(
        f"s{sid}={v/1e6/nsteps:.3f}" for sid, v in
        sorted(per_stream.items(), key=lambda kv: -kv[1])[:8]))

if __name__ == "__main__":
    main()
