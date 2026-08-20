#!/usr/bin/env python3
"""Decide the paper-claims ledger (rocq/gen/claims.json) against artifacts on disk.

WHY THIS EXISTS. Six research notes proposed importing ~30 quantitative claims
into this stack. Read as prose they all sound plausible; the only way to tell an
importable number from an un-importable one is to name the artifact each side is
measured against and do the arithmetic. This does that, and REFUSES -- exit
non-zero, naming the construct and the file -- on anything it cannot decide.

WHAT IT WILL NOT DO. It will not fall back to a literal when the artifact behind
that literal is gone. The measured constants in GenRoofline.v (verify 27.86 ms,
draft 3.23 ms, host 0.1213 ms) are real and were really measured, but the
corpus they were parsed from -- docs/measurements/2026-08-10-rocq-repoint/ --
was committed on 2026-08-10 expressly so the tripwire could not go dark and
deleted 21h later by 6557eba90. Every bound that needs one of those constants is
REFUSED here, by name, rather than computed from the transcribed literal. A
ledger that trusts an orphaned number is the failure mode it exists to prevent.

WHAT IT DOES DECIDE. The claims that are assertions about OUR OWN pinned
artifacts -- checkpoint byte formats, tensor censuses, declared context windows,
M=1 compulsory-read shares. Those are decidable from disk with no GPU and no
measurement round, and they are where four of the five contradictions live.

    ./claims.py            decide, emit ../GenClaims.v, print counts
    ./claims.py --check    decide, verify ../GenClaims.v matches, write nothing

Exit codes: 0 all decided and nothing refused; 1 one or more REFUSED (or, under
--check, GenClaims.v is stale); 2 the ledger itself is malformed.
"""
import json
import os
import re
import struct
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "claims.json")
OUT_V = os.path.abspath(os.path.join(HERE, "..", "GenClaims.v"))

SAFETENSORS_MIN_HEADER = 2
SAFETENSORS_MAX_HEADER = 100 * 1024 * 1024
LFS_MAGIC = b"version https://git-lfs.github.com/spec/v1"

CLAIM_KEYS = {
    "id", "note", "paper", "note_artifact", "claim_text",
    "stated_operating_point", "our_context", "lever_class", "claimed",
    "check", "check_args", "reason_code",
}
CONTEXT_KEYS = {"model", "phase", "drafter", "k"}

OUTCOMES = ("CONTRADICTED", "BOUNDED_BELOW_CLAIM", "CONSISTENT",
            "NOT_COMPARABLE", "REFUSED")


class Refusal(Exception):
    """Raised by a check that cannot decide. Carries the construct and the file."""

    def __init__(self, what, where):
        super().__init__(what)
        self.what = what
        self.where = where


def fatal(msg, where):
    print(f"[claims] LEDGER MALFORMED: {msg}\n         at {where}", file=sys.stderr)
    sys.exit(2)


def frac(v, where):
    if not (isinstance(v, list) and len(v) == 2
            and all(isinstance(x, int) for x in v) and v[1] != 0):
        fatal(f"expected a [num, den] integer pair, got {v!r}", where)
    return Fraction(v[0], v[1])


def resolve_model(name, models):
    if name not in models:
        fatal(f"claim names model {name!r}, absent from the models table", LEDGER)
    m = models[name]
    for c in m["dir_candidates"]:
        expanded = os.path.expanduser(os.path.expandvars(c))
        if os.path.exists(os.path.join(expanded, "config.json")):
            return dict(m, dir=expanded, name=name)
    raise Refusal(
        f"no snapshot for model {name!r}; tried " + ", ".join(m["dir_candidates"]),
        LEDGER,
    )


def read_config(m):
    with open(os.path.join(m["dir"], "config.json")) as f:
        c = json.load(f)
    sec = m.get("config_section")
    cfg = c.get(sec, c) if sec else c
    for k, want in m["expect_config"].items():
        if k not in cfg:
            raise Refusal(f"config.json has no {k!r} for {m['name']}",
                          os.path.join(m["dir"], "config.json"))
        if cfg[k] != want:
            raise Refusal(
                f"pin drift: {m['name']} config {k}={cfg[k]!r}, ledger expects "
                f"{want!r}; re-derive the ledger before trusting any bound",
                os.path.join(m["dir"], "config.json"),
            )
    return cfg


def safetensors_header(path):
    """Header dict, or a Refusal naming exactly why the file is not weights."""
    if not os.path.exists(path):
        raise Refusal(f"weights file absent: {os.path.basename(path)}", path)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if head.startswith(LFS_MAGIC[:8]):
            f.seek(0)
            body = f.read(512).decode("utf-8", "replace")
            oid = re.search(r"oid sha256:(\w+)", body)
            declared = re.search(r"size (\d+)", body)
            raise Refusal(
                f"UNRESOLVED GIT-LFS POINTER, not weights: {size} B on disk, "
                f"declares {declared.group(1) if declared else '?'} B "
                f"(oid {oid.group(1)[:16] if oid else '?'}...)",
                path,
            )
        if len(head) < 8:
            raise Refusal(f"file shorter than a safetensors header ({size} B)", path)
        n = struct.unpack("<Q", head)[0]
        if not (SAFETENSORS_MIN_HEADER <= n <= SAFETENSORS_MAX_HEADER) or 8 + n > size:
            raise Refusal(
                f"not a safetensors file: header length {n} against {size} B on disk",
                path,
            )
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr


def tensor_bytes(t):
    lo, hi = t["data_offsets"]
    return hi - lo


def group_bytes(m, hdr):
    """Total bytes per m1_groups bucket. First matching pattern wins; the last
    pattern must be a catch-all so no tensor can fall out of the accounting."""
    groups = m["m1_groups"]
    if not groups:
        raise Refusal(f"model {m['name']} declares no m1_groups", LEDGER)
    if groups[-1]["re"] != ".":
        fatal(f"m1_groups for {m['name']} has no catch-all last bucket", LEDGER)
    pats = [(g["name"], re.compile(g["re"]), g["per_step"]) for g in groups]
    tot = {g["name"]: 0 for g in groups}
    seen = 0
    for k, v in hdr.items():
        for name, rx, _ps in pats:
            if rx.search(k):
                tot[name] += tensor_bytes(v)
                seen += 1
                break
        else:
            fatal(f"tensor {k!r} matched no bucket", LEDGER)
    if seen != len(hdr):
        fatal(f"bucket accounting lost tensors: {seen} of {len(hdr)}", LEDGER)
    return tot, {g["name"]: g["per_step"] for g in groups}


def m1_compulsory(m, cfg, hdr):
    tot, per_step = group_bytes(m, hdr)
    active = {}
    for name, b in tot.items():
        ps = per_step[name]
        if ps == "none":
            continue
        if ps == "all":
            active[name] = b
        elif ps == "topk_of_num_experts":
            ne, tk = cfg["num_experts"], cfg["num_experts_per_tok"]
            if b % ne != 0:
                raise Refusal(
                    f"group {name} is {b} B over {ne} experts, not divisible; the "
                    f"uniform-expert assumption behind the top-k share does not hold",
                    m["dir"],
                )
            active[name] = b // ne * tk
        else:
            fatal(f"unknown per_step {ps!r}", LEDGER)
    return sum(active.values()), active, tot


# ---------------------------------------------------------------- checks

def check_block_format_bytes(c, ctx):
    a = c["check_args"]
    m = resolve_model(a["model"], ctx["models"])
    read_config(m)
    hdr = safetensors_header(os.path.join(m["dir"], m["weights"]))
    for t in (a["packed_tensor"], a["scale_tensor"]):
        if t not in hdr:
            raise Refusal(f"tensor absent from the checkpoint: {t}",
                          os.path.join(m["dir"], m["weights"]))
    got = Fraction(tensor_bytes(hdr[a["packed_tensor"]])
                   + tensor_bytes(hdr[a["scale_tensor"]]), a["logical_elems"])
    base = frac(a["baseline_bytes_per_elem"], c["id"])
    claimed_cut = frac(c["claimed"]["value"], c["id"])
    actual_cut = 1 - got / base
    ok = actual_cut >= claimed_cut
    return {
        "outcome": "CONSISTENT" if ok else "CONTRADICTED",
        "detail": (f"checkpoint bytes/elem = {got} ({float(got)}) vs baseline "
                   f"{base}; actual cut {float(actual_cut)*100:.2f}%, "
                   f"claimed {float(claimed_cut)*100:.2f}%"),
        "rocq": {
            "kind": "format",
            "packed": tensor_bytes(hdr[a["packed_tensor"]]),
            "scale": tensor_bytes(hdr[a["scale_tensor"]]),
            "elems": a["logical_elems"],
            "baseline": a["baseline_bytes_per_elem"],
            "claimed_cut": [claimed_cut.numerator, claimed_cut.denominator],
        },
    }


def check_declared_context_ceiling(c, ctx):
    a = c["check_args"]
    m = resolve_model(a["model"], ctx["models"])
    cfg = read_config(m)
    mpe = cfg["max_position_embeddings"]
    want = c["claimed"]["value"]
    return {
        "outcome": "CONTRADICTED" if want > mpe else "CONSISTENT",
        "detail": (f"claimed operating point {want} tokens vs declared "
                   f"max_position_embeddings {mpe} in {m['name']} config.json; "
                   f"the claimed point lies outside the declared context"),
        "rocq": {"kind": "context", "mpe": mpe, "claimed": want},
    }


def check_checkpoint_tensor_file_real(c, ctx):
    a = c["check_args"]
    m = resolve_model(a["model"], ctx["models"])
    path = os.path.join(m["dir"], a["file"])
    try:
        hdr = safetensors_header(path)
    except Refusal as r:
        return {
            "outcome": "CONTRADICTED",
            "detail": f"{r.what} [{r.where}]",
            "rocq": None,
        }
    return {
        "outcome": "CONSISTENT",
        "detail": f"{a['file']} holds {len(hdr)} real tensors",
        "rocq": None,
    }


def check_tensor_prefix_census(c, ctx):
    a = c["check_args"]
    m = resolve_model(a["model"], ctx["models"])
    cfg = read_config(m)
    hdr = safetensors_header(os.path.join(m["dir"], m["weights"]))
    counts = {}
    for name, pat in a["patterns"].items():
        rx = re.compile(pat)
        counts[name] = len({mm.group(1) for k in hdr for mm in [rx.search(k)] if mm})
    bad = [n for n in a["expect_zero"] if counts[n] != 0]
    total = sum(counts.values())
    layers = cfg["num_hidden_layers"]
    if total != layers:
        raise Refusal(
            f"census covers {total} layers of {layers}; the pattern set is "
            f"incomplete and a zero count would be meaningless",
            os.path.join(m["dir"], m["weights"]),
        )
    return {
        "outcome": "CONTRADICTED" if bad else "CONSISTENT",
        "detail": ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
                  + f" (of {layers} layers)",
        "rocq": {"kind": "census", "counts": counts, "layers": layers,
                 "expect_zero": a["expect_zero"]},
    }


def check_m1_weight_amdahl(c, ctx):
    a = c["check_args"]
    m = resolve_model(a["model"], ctx["models"])
    cfg = read_config(m)
    hdr = safetensors_header(os.path.join(m["dir"], m["weights"]))
    m1, active, _tot = m1_compulsory(m, cfg, hdr)
    grp = a["cut_group"]
    if grp not in active:
        raise Refusal(f"cut_group {grp!r} is not an active M=1 group "
                      f"({sorted(active)})", LEDGER)
    cut = Fraction(active[grp]) * frac(a["cut_fraction"], c["id"])
    if cut >= m1:
        raise Refusal(f"cut {cut} B >= compulsory {m1} B", LEDGER)
    bound = Fraction(m1) / (Fraction(m1) - cut)
    claimed = frac(c["claimed"]["value"], c["id"])
    share = Fraction(active[grp], m1)
    return {
        "outcome": "BOUNDED_BELOW_CLAIM" if claimed > bound else "CONSISTENT",
        "detail": (f"M=1 compulsory read {m1:,} B; {grp} active {active[grp]:,} B "
                   f"= {float(share)*100:.2f}%; removing {int(cut):,} B caps the "
                   f"speedup at {float(bound):.4f}x, claim is {float(claimed):.4f}x"),
        "rocq": {"kind": "amdahl", "m1": m1, "cut": int(cut),
                 "claimed": [claimed.numerator, claimed.denominator],
                 "group": grp, "active": active[grp]},
    }


def _measured_refusal(ctx, why):
    missing = [p for p in ctx["measured"]["artifacts"]
               if not os.path.exists(os.path.join(HERE, p))]
    if missing:
        raise Refusal(
            f"{why}; the measurement corpus is absent: "
            + ", ".join(os.path.normpath(os.path.join(HERE, p)) for p in missing)
            + ". The constants are transcribed in rocq/GenRoofline.v but this "
              "ledger will not compute a bound from an orphaned literal. "
              "Re-arm with rocq/gen/bench.sh (needs the GPU) and re-run.",
            ctx["measured"]["constants_recorded_in"],
        )
    raise Refusal(
        f"{why}; the artifacts are present again but this check was never "
        f"implemented against them -- implement it or delete the claim",
        LEDGER,
    )


def check_verify_weight_floor_ratio(c, ctx):
    _measured_refusal(ctx, "verify-phase ratio needs measured_verify_hi_s")


def check_phase_share_amdahl(c, ctx):
    _measured_refusal(
        ctx, f"phase-share Amdahl on {c['check_args']['phase']} needs the "
             f"measured round split")


def check_accept_rate_ceiling(c, ctx):
    _measured_refusal(ctx, "acceptance-rate ceiling needs measured tokens/round")


CHECKS = {
    "block_format_bytes": check_block_format_bytes,
    "declared_context_ceiling": check_declared_context_ceiling,
    "checkpoint_tensor_file_real": check_checkpoint_tensor_file_real,
    "tensor_prefix_census": check_tensor_prefix_census,
    "m1_weight_amdahl": check_m1_weight_amdahl,
    "verify_weight_floor_ratio": check_verify_weight_floor_ratio,
    "phase_share_amdahl": check_phase_share_amdahl,
    "accept_rate_ceiling": check_accept_rate_ceiling,
}


# ---------------------------------------------------------------- Rocq

def emit_rocq(rows, ledger):
    L = []
    w = L.append
    w("(* AUTO-GENERATED by rocq/gen/claims.py -- DO NOT EDIT; re-run gen/run.sh.")
    w("")
    w("   Every constant below was read out of a pinned checkpoint at generation")
    w("   time, not transcribed by hand: re-pin the model and the constants move,")
    w("   and any theorem that stops holding fails this build instead of rotting.")
    w("")
    w("   Each theorem is a REFUTATION stated so that it is false if the claim is")
    w("   importable. `2 * (m1 - cut) > m1` says: a lever that removes only `cut`")
    w("   bytes from the M=1 compulsory read cannot reach 2x, because Amdahl caps")
    w("   it at m1 / (m1 - cut). The claims themselves, their sources and the")
    w("   ones this file does NOT bound are in rocq/gen/claims.json. *)")
    w("")
    w("From Stdlib Require Import ZArith Lia.")
    w("Open Scope Z_scope.")
    w("")
    for r in rows:
        k = r["rocq"]
        cid = re.sub(r"[^A-Za-z0-9]", "_", r["id"])
        w(f"(* {r['id']} -- {r['paper']} *)")
        w(f"(* claim: {r['claim_text']} *)")
        w(f"(* {r['detail']} *)")
        if k["kind"] == "amdahl":
            cn, cd = k["claimed"]
            w(f"Definition {cid}_m1_bytes : Z := {k['m1']}.")
            w(f"Definition {cid}_cut_bytes : Z := {k['cut']}.")
            w(f"Theorem {cid}_unattainable :")
            w(f"  {cn} * ({cid}_m1_bytes - {cid}_cut_bytes) > {cd} * {cid}_m1_bytes.")
            w(f"Proof. unfold {cid}_m1_bytes, {cid}_cut_bytes; lia. Qed.")
        elif k["kind"] == "format":
            cn, cd = k["claimed_cut"]
            if k["baseline"] != [1, 1]:
                fatal("the Z form of the format refutation assumes a 1 B/elem "
                      f"baseline; got {k['baseline']}", LEDGER)
            tot = k["packed"] + k["scale"]
            w(f"Definition {cid}_quant_bytes : Z := {tot}.")
            w(f"Definition {cid}_logical_elems : Z := {k['elems']}.")
            w(f"(* claimed cut {cn}/{cd} against a 1 B/elem baseline; refuted iff")
            w(f"   the surviving bytes exceed the claimed survivor share *)")
            w(f"Theorem {cid}_cut_overstated :")
            w(f"  {cd} * {cid}_quant_bytes > ({cd} - {cn}) * {cid}_logical_elems.")
            w(f"Proof. unfold {cid}_quant_bytes, {cid}_logical_elems; lia. Qed.")
        elif k["kind"] == "context":
            w(f"Definition {cid}_max_position_embeddings : Z := {k['mpe']}.")
            w(f"Theorem {cid}_outside_declared_context :")
            w(f"  {k['claimed']} > {cid}_max_position_embeddings.")
            w(f"Proof. unfold {cid}_max_position_embeddings; lia. Qed.")
        elif k["kind"] == "census":
            for n, v in sorted(k["counts"].items()):
                w(f"Definition {cid}_{n} : Z := {v}.")
            terms = " + ".join(f"{cid}_{n}" for n in sorted(k["counts"]))
            w(f"Theorem {cid}_census_total : {terms} = {k['layers']}.")
            w(f"Proof. unfold {', '.join(cid + '_' + n for n in sorted(k['counts']))}; lia. Qed.")
            for n in k["expect_zero"]:
                w(f"Theorem {cid}_{n}_nonzero : {cid}_{n} > 0.")
                w(f"Proof. unfold {cid}_{n}; lia. Qed.")
        else:
            fatal(f"emit_rocq has no case for {k['kind']!r}", LEDGER)
        w("")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- main

def main(argv):
    check_only = "--check" in argv[1:]
    for a in argv[1:]:
        if a != "--check":
            print(f"usage: {argv[0]} [--check]", file=sys.stderr)
            return 2
    with open(LEDGER) as f:
        led = json.load(f)
    for k in ("models", "claims", "measured_basis", "_reason_codes", "_outcomes"):
        if k not in led:
            fatal(f"missing top-level key {k!r}", LEDGER)
    ctx = {"models": led["models"], "measured": led["measured_basis"]}
    reasons = set(led["_reason_codes"])

    print("[claims] STANDING REFUSAL NOTE-CORPUS-ABSENT: the six source notes are")
    print("[claims]   not in this repo; every claim_text is RELAYED, not quoted")
    print("[claims]   from an artifact, and note_artifact is null for all entries.")
    print("[claims]   Claims that assert something about OUR artifacts are still")
    print("[claims]   decidable and are decided below; nothing else is.")

    seen = set()
    rows = []
    for c in led["claims"]:
        where = f"{LEDGER}: claim {c.get('id', '<unnamed>')!r}"
        extra = set(c) - CLAIM_KEYS
        missing = CLAIM_KEYS - set(c)
        if extra or missing:
            fatal(f"key mismatch: unexpected {sorted(extra)}, missing {sorted(missing)}",
                  where)
        if c["id"] in seen:
            fatal(f"duplicate claim id {c['id']!r}", where)
        seen.add(c["id"])
        if set(c["our_context"]) != CONTEXT_KEYS:
            fatal(f"our_context keys must be exactly {sorted(CONTEXT_KEYS)}", where)
        if c["note_artifact"] is not None:
            fatal("note_artifact is non-null but no note corpus is in-tree; "
                  "add the file and teach this checker to quote it", where)
        if (c["check"] is None) == (c["reason_code"] is None):
            fatal("exactly one of check / reason_code must be set -- a claim with "
                  "neither is a silent skip, a claim with both hides which one ran",
                  where)
        row = dict(c)
        if c["check"] is None:
            if c["reason_code"] not in reasons:
                fatal(f"reason_code {c['reason_code']!r} not in _reason_codes", where)
            row.update(outcome="NOT_COMPARABLE", detail=c["reason_code"], rocq=None)
        else:
            fn = CHECKS.get(c["check"])
            if fn is None:
                fatal(f"no implementation for check {c['check']!r}", where)
            try:
                res = fn(c, ctx)
                if res is None:
                    fatal(f"check {c['check']!r} returned without deciding", where)
                row.update(res)
            except Refusal as r:
                row.update(outcome="REFUSED", detail=f"{r.what} [{r.where}]",
                           rocq=None)
        if row["outcome"] not in OUTCOMES:
            fatal(f"check produced unknown outcome {row['outcome']!r}", where)
        rows.append(row)

    by = {o: [r for r in rows if r["outcome"] == o] for o in OUTCOMES}
    print()
    for r in rows:
        print(f"[claims] {r['outcome']:<20} {r['id']}")
        print(f"[claims]   {r['detail']}")
    print()
    print("[claims] " + " | ".join(f"{o} {len(by[o])}" for o in OUTCOMES)
          + f" | total {len(rows)}")

    for o in ("NOT_COMPARABLE", "REFUSED"):
        if by[o]:
            tally = {}
            for r in by[o]:
                tally[r["detail"].split(";")[0][:60]] = tally.get(
                    r["detail"].split(";")[0][:60], 0) + 1
            print(f"[claims] {o} by reason:")
            for k, v in sorted(tally.items(), key=lambda kv: -kv[1]):
                print(f"[claims]   {v:>2}  {k}")

    rocq_rows = [r for r in rows if r.get("rocq")]
    text = emit_rocq(rocq_rows, led)
    if check_only:
        cur = open(OUT_V).read() if os.path.exists(OUT_V) else None
        if cur != text:
            print(f"[claims] STALE: {OUT_V} does not match the ledger; re-run "
                  f"claims.py", file=sys.stderr)
            return 1
        print(f"[claims] {OUT_V} matches the ledger ({len(rocq_rows)} theorems' "
              f"worth of constants)")
    else:
        with open(OUT_V, "w") as f:
            f.write(text)
        print(f"[claims] wrote {OUT_V} from {len(rocq_rows)} artifact-derived checks")

    if by["REFUSED"]:
        print(f"[claims] REFUSING: {len(by['REFUSED'])} claim(s) could not be "
              f"decided; see the REFUSED lines above", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
