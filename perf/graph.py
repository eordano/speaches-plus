#!/usr/bin/env python3
import argparse
import html
import json
import math
import re
from pathlib import Path

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"

PANELS = [
    {
        "title": "qwen3.8-27B NVFP4 -- effective decode tok/s vs context depth (CUDA, batch 1)",
        "series": [
            {"label": "ours MTP agentic", "color": BLUE,
             "build": r"q38 (armv$|graph-refresh)", "engine": "ours",
             "note_filter": r"class=agentic|arm=mtp-champion(?!-)", "drafter": "mtp"},
            {"label": "ours MTP freeform", "color": AQUA,
             "build": r"q38 (armv$|graph-refresh)", "engine": "ours",
             "note_filter": r"class=freeform|arm=mtp-champion-freeform", "drafter": "mtp"},
            {"label": "ours MTP deep (tctile)", "color": BLUE,
             "build": r"q38 tctile$", "engine": "ours", "drafter": "mtp",
             "note_filter": r"tctile campaign treatment"},
            {"label": "llama.cpp CUDA Q4_K_M", "color": ORANGE,
             "build": r"llamacpp|llama\.cpp", "engine": "llama.cpp",
             "model": r"qwen38|Qwen3\.8", "backend": "cuda"},
            {"label": "vLLM NVFP4", "color": YELLOW,
             "build": r"vllm", "engine": "vllm", "model": r"qwen38"},
            {"label": "ours SERVED full stack", "color": BLUE,
             "build": r"qwen3\.8 ours served mtp", "engine": "ours",
             "note_filter": r"full arm stack|full current stack|engine-best refresh"},
            {"label": "ours SERVED pending flips", "color": YELLOW,
             "build": r"qwen3\.8 ours served pending-flips", "engine": "ours"},
            {"label": "llama.cpp SERVED draft-mtp", "color": ORANGE,
             "build": r"qwen3\.8 llamacpp mtp", "engine": "llama.cpp",
             "note_filter": r"draft-mtp"},
        ],
        "ppl_build": r"q38 (gdnflip|rotkv|tctile-cert|nows12288-cert)",
    },
    {
        "title": "gpt-oss-20B MXFP4 -- decode tok/s vs context depth (wgpu-Vulkan, batch 1)",
        "series": [
            {"label": "ours wgpu defaults", "color": BLUE,
             "build": r"gptoss (flip-verify|shallow-r1|fuse)", "engine": "ours"},
            {"label": "ours chunk32+lmhead (pending flips)", "color": YELLOW,
             "build": r"gptoss subchunk", "engine": "ours",
             "note_filter": r"arm=chunk32-lmhead"},
            {"label": "llama.cpp Vulkan", "color": ORANGE,
             "build": r"llamacpp", "engine": "llama.cpp",
             "model": r"gpt-oss", "backend": "vulkan"},
            {"label": "llama.cpp CUDA", "color": AQUA,
             "build": r"llamacpp", "engine": "llama.cpp",
             "model": r"gpt-oss", "backend": "cuda"},
        ],
        "ppl_build": r"gptoss",
    },
    {
        "title": "gemma4-26B-A4B -- decode tok/s vs context depth (wgpu-Vulkan, batch 1)",
        "series": [
            {"label": "ours wgpu defaults", "color": BLUE,
             "build": r"g4moe (flip-verify|gemvq|graph-refresh)", "engine": "ours",
             "note_filter": r"flip-verify|ladder=g4moe-graph-refresh|arm=champ [\[(]|arm=champ-again"},
            {"label": "ours chunk32+v2 (pending flips)", "color": YELLOW,
             "build": r"g4moe subchunk", "engine": "ours",
             "note_filter": r"arm=chunk32-v2"},
            {"label": "llama.cpp Vulkan", "color": ORANGE,
             "build": r"llamacpp|llama\.cpp", "engine": "llama.cpp",
             "model": r"gemma-?4|g4moe", "backend": "vulkan"},
        ],
        "ppl_build": r"g4moe",
    },
    {
        "title": "gemma4-E4B -- decode tok/s vs context depth (CUDA serving, batch 1; no on-box comparator)",
        "series": [
            {"label": "ours cuda defaults", "color": BLUE,
             "build": r"e4b (cupf|graph-deep)", "engine": "ours",
             "note_filter": r"serving-default|ladder=e4b-graph-deep"},
            {"label": "ours boot-warm prefill (opt-in)", "color": AQUA,
             "build": r"e4b cupf", "engine": "ours",
             "note_filter": r"arm: NV_E4B_BOOT_WARM_PREFILL"},
        ],
        "ppl_build": r"e4b",
    },
]


def row_depths_and_tok_s(row):
    out = []
    flags = row.get("flags") or {}
    ctx = str(flags.get("NV_CTX_TOKENS") or "")
    dec = row.get("decode_tok_s")
    if ctx and "," not in ctx and dec:
        try:
            out.append((int(ctx), float(dec)))
            return out
        except ValueError:
            pass
    for line in row.get("measure_lines") or []:
        m = re.search(r"n_depth=(\d+).*?avg_ts=([0-9.]+)", line)
        if m and "n_prompt=0" in line:
            out.append((int(m.group(1)), float(m.group(2))))
        m = re.search(r"VLLM-BENCH input_len=(\d+).*?output_tok_s=([0-9.]+)", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
        m = re.search(r"CTX-SCALING \S+ depth=(\d+).*?tok_s=([0-9.]+)", line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    if not out and dec:
        m = re.search(r"depth=(\d+)", str(row.get("notes") or ""))
        if m:
            out.append((int(m.group(1)), float(dec)))
    return out


def series_match(spec, row):
    if not re.search(spec["build"], row.get("build_name") or ""):
        return False
    if spec.get("engine") and (row.get("engine") or "") != spec["engine"]:
        return False
    if spec.get("model") and not re.search(spec["model"], row.get("model") or ""):
        return False
    if spec.get("backend") and (row.get("backend") or "") != spec["backend"]:
        return False
    if spec.get("drafter") and (row.get("flags") or {}).get("NV_DRAFTER") != spec["drafter"]:
        return False
    nf = spec.get("note_filter")
    if nf and not re.search(nf, str(row.get("notes") or "")):
        return False
    if row.get("status") not in (None, "ok"):
        return False
    if (row.get("flags") or {}).get("NV_WGPU_PROFILE") == "1":
        return False
    return True


def collect(rows, spec):
    best = {}
    for row in rows:
        if not series_match(spec, row):
            continue
        ts = row.get("started_at") or ""
        for depth, tok_s in row_depths_and_tok_s(row):
            cur = best.get(depth)
            if cur is None or ts > cur[0]:
                best[depth] = (ts, tok_s, row.get("run_id") or "")
    return {d: v for d, v in best.items()}


def svg_panel(panel, rows):
    width, height = 760, 340
    ml, mr, mt, mb = 64, 190, 24, 44
    pw, ph = width - ml - mr, height - mt - mb
    per_series = [(spec, collect(rows, spec)) for spec in panel["series"]]
    per_series = [(s, pts) for s, pts in per_series if pts]
    if not per_series:
        return "<p>no rows matched</p>", []
    depths = sorted({d for _, pts in per_series for d in pts})
    ymax = max(v[1] for _, pts in per_series for v in pts.values()) * 1.12
    x0, x1 = math.log2(min(depths)), math.log2(max(max(depths), min(depths) * 2))

    def X(d):
        return ml + pw * (math.log2(d) - x0) / (x1 - x0)

    def Y(v):
        return mt + ph * (1 - v / ymax)

    parts = [f'<svg viewBox="0 0 {width} {height}" font-family="system-ui" font-size="12">']
    step = max(1, int(ymax // 5 // 10) * 10) or 1
    y = step
    while y < ymax:
        parts.append(f'<line x1="{ml}" y1="{Y(y):.1f}" x2="{ml+pw}" y2="{Y(y):.1f}" stroke="{GRID}"/>')
        parts.append(f'<text x="{ml-8}" y="{Y(y)+4:.1f}" text-anchor="end" fill="{MUTED}">{y}</text>')
        y += step
    parts.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="{AXIS}"/>')
    for d in depths:
        parts.append(f'<text x="{X(d):.1f}" y="{mt+ph+18}" text-anchor="middle" fill="{MUTED}">{d if d < 1000 else str(round(d/1024))+"k"}</text>')
    table = []
    for spec, pts in per_series:
        pl = sorted(pts.items())
        path = " ".join(f'{"M" if i == 0 else "L"}{X(d):.1f},{Y(v[1]):.1f}' for i, (d, v) in enumerate(pl))
        parts.append(f'<path d="{path}" fill="none" stroke="{spec["color"]}" stroke-width="2"/>')
        for d, (ts, v, rid) in pl:
            parts.append(
                f'<circle cx="{X(d):.1f}" cy="{Y(v):.1f}" r="4" fill="{spec["color"]}" stroke="{SURFACE}" stroke-width="2">'
                f"<title>{html.escape(spec['label'])} @{d}: {v:.1f} tok/s ({html.escape(rid)} {html.escape(ts[:16])})</title></circle>")
        ld, (lts, lv, _) = pl[-1]
        parts.append(f'<text x="{X(ld)+10:.1f}" y="{Y(lv)+4:.1f}" fill="{INK2}">{html.escape(spec["label"])} {lv:.0f}</text>')
        table.append((spec["label"], pl))
    parts.append("</svg>")
    return "".join(parts), table


def ppl_rows(rows, pat):
    latest = {}
    for row in rows:
        p = row.get("ppl") or {}
        if not p.get("value"):
            continue
        if not re.search(pat, row.get("build_name") or ""):
            continue
        key = (row.get("build_name"), p.get("family") or "")
        ts = row.get("started_at") or ""
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, p)
    out = sorted(latest.items(), key=lambda kv: kv[1][0], reverse=True)[:6]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(Path(__file__).parent / "runs.jsonl"))
    ap.add_argument("-o", "--out", default=str(Path(__file__).parent / "graph.html"))
    args = ap.parse_args()
    rows = []
    for line in open(args.runs):
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    body = [f"<html><head><meta charset='utf-8'><title>speaches-plus perf frontier</title>"
            f"<style>body{{background:{SURFACE};color:{INK};font-family:system-ui;max-width:820px;margin:24px auto}}"
            f"table{{border-collapse:collapse;font-size:12px;margin:8px 0 28px}}"
            f"td,th{{border:1px solid {GRID};padding:3px 8px;text-align:right}}"
            f"th{{color:{INK2}}}td:first-child,th:first-child{{text-align:left}}</style></head><body>"]
    newest = max((r.get("started_at") or "" for r in rows), default="")
    body.append(f"<h2>speaches-plus decode frontier</h2><p style='color:{INK2}'>generated from perf/runs.jsonl "
                f"(newest row {html.escape(newest[:19])}); each point is the latest ok, unprofiled row per series and depth; "
                f"hover a marker for run id.</p>")
    for panel in PANELS:
        svg, table = svg_panel(panel, rows)
        body.append(f"<h3>{html.escape(panel['title'])}</h3>")
        body.append(svg)
        body.append("<table><tr><th>series</th><th>depth</th><th>tok/s</th><th>run</th><th>when</th></tr>")
        for label, pl in table:
            for d, (ts, v, rid) in pl:
                body.append(f"<tr><td>{html.escape(label)}</td><td>{d}</td><td>{v:.1f}</td>"
                            f"<td>{html.escape(rid)}</td><td>{html.escape(ts[:16])}</td></tr>")
        body.append("</table>")
        pp = ppl_rows(rows, panel["ppl_build"])
        if pp:
            body.append("<table><tr><th>quality row</th><th>family</th><th>ppl</th><th>acc</th><th>tokens</th></tr>")
            for (bn, fam), (ts, p) in pp:
                body.append(f"<tr><td>{html.escape(str(bn))}</td><td>{html.escape(str(fam))}</td>"
                            f"<td>{p.get('value')}</td><td>{p.get('accuracy') or ''}</td><td>{p.get('tokens') or ''}</td></tr>")
            body.append("</table>")
    body.append("</body></html>")
    Path(args.out).write_text("".join(body))
    print(f"graph: wrote {args.out} from {len(rows)} rows")


if __name__ == "__main__":
    main()
