#!/usr/bin/env python3
import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BACKEND_TOKENS = {"cuda", "wgpu", "metal", "gguf"}
ARCH_TOKENS = {"bf16", "fp8", "fp4", "e4m3", "int8", "w4a8", "w4a16"}
EXCLUDED_DIFF_TOKENS = BACKEND_TOKENS | ARCH_TOKENS

RUST_FN_RE = re.compile(
    r"^\s*(pub(\([^)]*\))?\s+)?(async\s+)?(unsafe\s+)?fn\s+([A-Za-z0-9_]+)"
)
CU_KERNEL_RE = re.compile(r"__global__\s+\w+\s+([A-Za-z0-9_]+)")
WGSL_FN_RE = re.compile(r"^\s*fn\s+([A-Za-z0-9_]+)")

MIN_NAME_LEN = 10
LEV_THRESHOLD = 0.82
JACCARD_THRESHOLD = 0.75
BODY_CAP_LINES = 400


def levenshtein(a, b):
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def lev_similarity(a, b):
    m = max(len(a), len(b))
    if m == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / m


def tokens_of(name):
    return [t for t in name.lower().split("_") if t]


def jaccard(name_a, name_b):
    sa, sb = set(tokens_of(name_a)), set(tokens_of(name_b))
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def backend_only_difference(name_a, name_b):
    ta = [t for t in tokens_of(name_a) if t not in EXCLUDED_DIFF_TOKENS]
    tb = [t for t in tokens_of(name_b) if t not in EXCLUDED_DIFF_TOKENS]
    if name_a == name_b:
        return False
    return ta == tb


def scan_body_len(lines, start_idx):
    depth = 0
    opened = False
    count = 0
    for i in range(start_idx, min(len(lines), start_idx + BODY_CAP_LINES)):
        count = i - start_idx + 1
        for ch in lines[i]:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
                if opened and depth <= 0:
                    return count
        if not opened and ";" in lines[i]:
            return count
    return count


def collect_defs(root, include_tests):
    defs = []
    scan_dirs = []
    crates_dir = os.path.join(root, "rust", "crates")
    if os.path.isdir(crates_dir):
        for crate in sorted(os.listdir(crates_dir)):
            src = os.path.join(crates_dir, crate, "src")
            if os.path.isdir(src):
                scan_dirs.append(src)
    top_src = os.path.join(root, "rust", "src")
    if os.path.isdir(top_src):
        scan_dirs.append(top_src)

    rust_files = []
    gpu_files = []
    for d in scan_dirs:
        for dirpath, _, filenames in os.walk(d):
            for fname in filenames:
                if fname.endswith(".rs"):
                    rust_files.append(os.path.join(dirpath, fname))
    if os.path.isdir(crates_dir):
        for dirpath, _, filenames in os.walk(crates_dir):
            for fname in filenames:
                if fname.endswith((".cu", ".wgsl")):
                    gpu_files.append(os.path.join(dirpath, fname))

    for path in rust_files + gpu_files:
        rel = os.path.relpath(path, root)
        if not include_tests and "/tests/" in "/" + rel.replace(os.sep, "/") + "/":
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines):
            name = None
            if path.endswith(".rs") or path.endswith(".wgsl"):
                m = RUST_FN_RE.match(line) if path.endswith(".rs") else WGSL_FN_RE.match(line)
                if m:
                    name = m.group(5) if path.endswith(".rs") else m.group(1)
            elif path.endswith(".cu"):
                m = CU_KERNEL_RE.search(line)
                if m:
                    name = m.group(1)
            if name:
                defs.append(
                    {
                        "name": name,
                        "path": rel,
                        "line": idx + 1,
                        "body_lines": scan_body_len(lines, idx),
                    }
                )
    return defs


def char_bag(name):
    bag = {}
    for ch in name:
        bag[ch] = bag.get(ch, 0) + 1
    return bag


def char_bag_dist_lower_bound(bag_a, bag_b):
    diff = 0
    for ch in set(bag_a) | set(bag_b):
        diff += abs(bag_a.get(ch, 0) - bag_b.get(ch, 0))
    return (diff + 1) // 2


def candidate_pairs(names):
    pairs = []
    toksets = {n: set(tokens_of(n)) for n in names}
    bags = {n: char_bag(n) for n in names}
    ordered = sorted(names)
    max_lev_frac = 1 - LEV_THRESHOLD
    for i in range(len(ordered)):
        a = ordered[i]
        for j in range(i + 1, len(ordered)):
            b = ordered[j]
            la, lb = len(a), len(b)
            m = max(la, lb)
            lev_possible = abs(la - lb) / m <= max_lev_frac
            if lev_possible:
                lev_possible = (
                    char_bag_dist_lower_bound(bags[a], bags[b]) / m <= max_lev_frac
                )
            sa, sb = toksets[a], toksets[b]
            jac = len(sa & sb) / len(sa | sb) if (sa or sb) else 1.0
            if not lev_possible and jac < JACCARD_THRESHOLD:
                continue
            sim_lev = lev_similarity(a, b) if lev_possible else 0.0
            if sim_lev < LEV_THRESHOLD and jac < JACCARD_THRESHOLD:
                continue
            if backend_only_difference(a, b):
                continue
            pairs.append((a, b, max(sim_lev, jac), sim_lev, jac))
    return pairs


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_clusters(defs, include_same_name):
    by_name = {}
    for d in defs:
        by_name.setdefault(d["name"], []).append(d)
    names = [n for n in by_name if len(n) >= MIN_NAME_LEN]
    pairs = candidate_pairs(names)
    uf = UnionFind()
    for a, b, _, _, _ in pairs:
        uf.union(a, b)
    if include_same_name:
        for n in names:
            if len(by_name[n]) > 1:
                uf.find(n)
    groups = {}
    for a, b, sim, sl, jc in pairs:
        groups.setdefault(uf.find(a), {"names": set(), "pair_sims": []})
        g = groups[uf.find(a)]
        g["names"].update((a, b))
        g["pair_sims"].append(sim)
    if include_same_name:
        for n in names:
            if len(by_name[n]) > 1 and uf.find(n) not in groups:
                groups[uf.find(n)] = {"names": {n}, "pair_sims": [1.0]}

    clusters = []
    for g in groups.values():
        members = []
        for n in sorted(g["names"]):
            members.extend(by_name[n])
        mean_sim = sum(g["pair_sims"]) / len(g["pair_sims"])
        total_body = sum(m["body_lines"] for m in members)
        clusters.append(
            {
                "names": sorted(g["names"]),
                "members": sorted(
                    members, key=lambda m: (m["path"], m["line"])
                ),
                "member_count": len(members),
                "mean_similarity": round(mean_sim, 4),
                "total_body_lines": total_body,
            }
        )
    clusters.sort(
        key=lambda c: (c["member_count"], c["mean_similarity"], c["total_body_lines"]),
        reverse=True,
    )
    return clusters


def print_report(clusters, top):
    print(f"namesim: {len(clusters)} clusters total; showing top {min(top, len(clusters))}")
    print()
    for rank, c in enumerate(clusters[:top], 1):
        print(
            f"#{rank}  members={c['member_count']}  mean_sim={c['mean_similarity']}  "
            f"body_lines={c['total_body_lines']}"
        )
        for m in c["members"]:
            print(f"    {m['name']:<52} {m['path']}:{m['line']}  ({m['body_lines']}L)")
        print()


def self_test():
    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    check(levenshtein("kitten", "sitting") == 3, "levenshtein kitten/sitting != 3")
    check(levenshtein("abc", "abc") == 0, "levenshtein identical != 0")
    check(levenshtein("", "abcd") == 4, "levenshtein empty/abcd != 4")
    check(
        abs(lev_similarity("decode_step_cuda", "decode_step_wgpu") - (1 - 4 / 16)) < 1e-9,
        "lev_similarity decode_step pair wrong",
    )

    check(
        backend_only_difference("matmul_kernel_cuda", "matmul_kernel_wgpu"),
        "backend exclusion missed cuda/wgpu pair",
    )
    check(
        backend_only_difference("gemm_launch_bf16", "gemm_launch_fp8"),
        "arch exclusion missed bf16/fp8 pair",
    )
    check(
        not backend_only_difference("decode_tokens_fast", "decode_tokens_slow"),
        "exclusion wrongly fired on fast/slow pair",
    )
    check(
        not backend_only_difference("prefill_chunked_run", "prefill_chunked_run"),
        "identical names must not be excluded-by-backend",
    )

    fixture = [
        "fn alpha_beta_gamma(x: u32) -> u32 {",
        "    if x > 0 {",
        "        return x;",
        "    }",
        "    0",
        "}",
        "fn decl_only();",
    ]
    check(scan_body_len(fixture, 0) == 6, "brace scan on fixture != 6 lines")
    check(scan_body_len(fixture, 6) == 1, "decl-only scan != 1 line")

    check(jaccard("load_weights_from_disk", "from_disk_load_weights") == 1.0,
          "jaccard reordered tokens != 1.0")

    pairs = candidate_pairs(
        ["matmul_kernel_cuda", "matmul_kernel_wgpu", "apply_rotary_embedding",
         "apply_rotary_embeddings", "decode_tokens_fast", "decode_tokens_slow"]
    )
    names_in_pairs = {frozenset((a, b)) for a, b, *_ in pairs}
    check(
        frozenset(("matmul_kernel_cuda", "matmul_kernel_wgpu")) not in names_in_pairs,
        "candidate_pairs kept a backend-twin pair",
    )
    check(
        frozenset(("apply_rotary_embedding", "apply_rotary_embeddings"))
        in names_in_pairs,
        "candidate_pairs dropped a real near-duplicate pair",
    )
    check(
        frozenset(("decode_tokens_fast", "decode_tokens_slow")) not in names_in_pairs,
        "candidate_pairs kept a below-threshold pair",
    )

    if failures:
        for f in failures:
            print(f"FAIL: {f}", file=sys.stderr)
        return 1
    print("self-test: all checks passed")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--include-tests", action="store_true")
    ap.add_argument("--root", default=REPO_ROOT)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(self_test())

    defs = collect_defs(args.root, args.include_tests)
    clusters = build_clusters(defs, include_same_name=True)

    if args.json:
        out = {
            "total_defs": len(defs),
            "total_clusters": len(clusters),
            "clusters": clusters[: args.top],
        }
        print(json.dumps(out, indent=1))
    else:
        print_report(clusters, args.top)


if __name__ == "__main__":
    main()
