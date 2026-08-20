#!/usr/bin/env python3
"""Derive the PDL happens-before facts from the CUDA and emit rocq/GenPdl.v.

PdlOrder.v is the RULE (a monotone schedule, epilog-dominates-writes,
prolog-precedes-reads, the PDL edge).  PdlKernels.v is the PAIRING (which
producer feeds which consumer on the captured decode path).  This is the
middle layer neither of them can hold honestly: WHERE, in today's source, the
prolog, the epilog, every store and every dereference of each wired kernel sit.

Why it exists.  PdlKernels.v used to carry those positions as integer literals
-- `k_prolog := 41`, `k_writes := [54]`.  Rocq cannot read a .cu, so nothing
ever compared them to the file, and all twelve had rotted: residual_scale.cu's
prolog had moved 41 -> 17, flash_decode.cu's 687 -> 1008.  The proofs still
checked, because a proof about integer literals is a proof about integer
literals.  Here the literals are OUTPUTS, regenerated from the sha256'd sources
recorded in gen/out/pdl_sites.json, and `gen/run.sh --check` fails when they
and the CUDA have drifted apart.

WHAT IS DERIVED (re-read on every run, never declared):

  prolog / epilog   the unique NVK_PDL_PROLOG() and at-most-one
                    NVK_PDL_EPILOG() inside the kernel body.
  store points      every line of the body holding an assignment whose target
                    is a subscript (`p[i] = `, `p[i] += `) or an explicit
                    dereference (`*p = `).
  dereference       every line of the body holding a subscript or an explicit
  points            dereference at all, in any position.

Both point sets are deliberate OVER-approximations: they include shared memory,
local arrays and buffers this model does not track.  Over-approximating is the
safe direction for both obligations -- "every store precedes the epilog" and
"the prolog precedes every dereference" are STRONGER than the per-buffer claims
PdlOrder needs, and imply them.  It also removes the step that the proof notes
record as having failed twice in review: nobody hand-picks a "last write" any
more, so nobody can pick the wrong one.  The names carry this: the emitted
lists are `<symbol>_every_store_line` and `<symbol>_every_dereference_line`.

WHAT IS DECLARED (gen/pdl_kernels.json, with a witness apiece): which
__global__ function each Rocq record models, and which logical buffer each
pointer parameter carries -- pinned to the exact source text of the parameter
declaration, required to occur exactly once inside that kernel's parameter
list.  A rename, a retype or a reorder is a refusal.

CENSUS, in both directions, because a model that cannot notice a kernel is the
failure this replaces:

  every NVK_PDL_PROLOG / NVK_PDL_EPILOG in the declared corpus must fall inside
  a declared kernel body -- that is how flash_splitk_fused_fp8_derivev_kernel,
  a second prolog the old table did not mention, becomes visible;
  every cudaLaunchKernelEx callee in the corpus must be a declared kernel --
  cudaLaunchKernelEx is how this repo launches with
  cudaLaunchAttributeProgrammaticStreamSerialization, both through NVK_PDL_ATTR
  and hand-rolled (gemv_bf16.cu builds the attribute inline), so keying on the
  call rather than on the macro catches both.

REFUSE vs EMIT.  A structural fact this cannot represent -- symbol absent or
ambiguous, pin not found, braces unbalanced, an undeclared wired kernel, inline
asm that could touch memory without a subscript -- is a refusal: rc=1, nothing
written, in the style of extract/launch_geometry.py.  An ORDERING violation is
representable, so it is emitted and left for Rocq to fail on: a store after the
epilog, or a dereference before the prolog, becomes a GenPdl.v that does not
compile.  Two loud failures beat one silent pass.
"""

import bisect
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cudatext  # noqa: E402
import cudatu  # noqa: E402

TAG = "pdl"
GEN = cudatu.GEN
ROCQ = cudatu.ROCQ
REPO = cudatu.REPO
OUT_V = os.path.join(ROCQ, "GenPdl.v")
OUT_JSON = os.path.join(GEN, "out", "pdl_sites.json")

PROLOG = "NVK_PDL_PROLOG"
EPILOG = "NVK_PDL_EPILOG"

SUBSCRIPT = re.compile(r"\b([A-Za-z_]\w*)\s*\[")
DEREF = re.compile(r"\*\s*[A-Za-z_]")
LAUNCH_EX = re.compile(r"cudaLaunchKernelEx\s*\(\s*&\s*\w+\s*,\s*([A-Za-z_][\w:]*)")
ASM = re.compile(r"\basm\b|__pipeline_memcpy_async|\bmemcpy\b")
ASSIGN = re.compile(r"(<<=|>>=|[-+*/%&|^]=|=(?![=]))")
NOT_UNARY_BEFORE = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_)]")

class Refusal(Exception):
    def __init__(self, where, kind, detail):
        super().__init__(detail)
        self.where = where
        self.kind = kind
        self.detail = detail

def line_starts(src):
    out = [0]
    for m in re.finditer("\n", src):
        out.append(m.end())
    return out

def line_of(starts, off):
    return bisect.bisect_right(starts, off)

def match_pair(s, i, opener, closer):
    depth = 0
    while i < len(s):
        if s[i] == opener:
            depth += 1
        elif s[i] == closer:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None

def kernel_extent(blanked, rel, symbol):
    """(param_open, param_close, body_open, body_close) byte offsets.

    Anchored on the SYMBOL, which is what a citation should name.  Exactly one
    definition is required: two overloads, or a definition plus a stray
    forward declaration written the same way, mean this cannot say which body
    the model describes.
    """
    pat = re.compile(r"__global__\s+void\s+" + re.escape(symbol) + r"\s*\(")
    ms = list(pat.finditer(blanked))
    if len(ms) != 1:
        raise Refusal(
            rel,
            "symbol-not-unique",
            "`__global__ void %s(` occurs %d times; the model names a symbol, "
            "so exactly one definition must answer to it." % (symbol, len(ms)),
        )
    popen = ms[0].end() - 1
    pclose = match_pair(blanked, popen, "(", ")")
    if pclose is None:
        raise Refusal(rel, "unbalanced-parens", "parameter list of %s never closes" % symbol)
    bopen = blanked.find("{", pclose)
    if bopen < 0:
        raise Refusal(rel, "no-body", "%s has no body after its parameter list" % symbol)
    bclose = match_pair(blanked, bopen, "{", "}")
    if bclose is None:
        raise Refusal(rel, "unbalanced-braces", "body of %s never closes" % symbol)
    return popen, pclose, bopen, bclose

def prev_significant(s, i):
    while i >= 0 and s[i] in " \t\r\n":
        i -= 1
    return s[i] if i >= 0 else ""

def next_significant(s, i):
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i

def deref_and_store_sites(blanked, rel, symbol, lo, hi):
    """([deref offsets], [store offsets]) inside the half-open body (lo, hi).

    A subscript `p[...]` or a unary `*p` is a dereference; either one directly
    left of an assignment operator is also a store.  Multiplication is told
    from a unary star by the character before it -- an identifier, `)` or `]`
    ends an operand, anything else opens one -- which also excludes the `*` of
    a pointer declaration, since a type name ends in an identifier character.
    """
    body = blanked[lo:hi]
    if ASM.search(body):
        raise Refusal(
            "%s:%d" % (rel, 0),
            "opaque-memory-op",
            "%s contains inline asm or a memcpy-like intrinsic, which can load "
            "or store without a subscript this scan would see. The dereference "
            "set would no longer be an over-approximation, so the "
            "prolog-precedes-reads theorem would stop implying the real claim."
            % symbol,
        )
    derefs = set()
    stores = set()

    def classify(off, after):
        derefs.add(off)
        j = next_significant(blanked, after)
        while j < hi and blanked[j] == "[":
            end = match_pair(blanked, j, "[", "]")
            if end is None:
                return
            j = next_significant(blanked, end + 1)
        m = ASSIGN.match(blanked, j)
        if m:
            stores.add(off)

    for m in SUBSCRIPT.finditer(body):
        start = lo + m.start(1)
        obr = lo + m.end() - 1
        end = match_pair(blanked, obr, "[", "]")
        if end is None or end >= hi:
            raise Refusal(rel, "unbalanced-subscript", "subscript in %s never closes" % symbol)
        classify(start, end + 1)
    for m in DEREF.finditer(body):
        star = lo + m.start()
        before = prev_significant(blanked, star - 1)
        if before in NOT_UNARY_BEFORE:
            continue
        if before == "*":
            raise Refusal(
                rel,
                "double-star",
                "`**` in %s: this scan cannot tell a pointer-to-pointer "
                "declaration from a double dereference, and guessing either way "
                "corrupts the point sets. There were zero of these when it was "
                "written." % symbol,
            )
        j = star + 1
        while j < hi and blanked[j] in " \t\r\n":
            j += 1
        k = j
        while k < hi and (blanked[k].isalnum() or blanked[k] == "_"):
            k += 1
        classify(star, k)
    return sorted(derefs), sorted(stores)

def lines(starts, offs):
    return sorted(set(line_of(starts, o) for o in offs))

def split_params(raw, rel, symbol, popen, pclose):
    """[(offset, text)] per parameter, split on the commas between them.

    Whole parameters, never substrings: a pin matched by `in` accepts
    `T* __restrict__ y` against a parameter that has since been renamed
    `y_out`, which is the same silent-drift failure this file exists to end.
    Planted and confirmed.
    """
    sig = raw[popen + 1: pclose]
    if "<" in sig:
        raise Refusal(
            rel,
            "template-in-parameter-list",
            "the parameter list of %s contains `<`; this splits parameters on "
            "commas and cannot tell a template argument comma from a parameter "
            "separator. There were zero of these when it was written." % symbol,
        )
    out = []
    depth = 0
    start = 0
    for i, ch in enumerate(sig + ","):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append((popen + 1 + start, sig[start:i]))
            start = i + 1
    return out

def param_pin_line(params, rel, symbol, pin, starts):
    hits = [off for off, text in params if text.strip() == pin]
    if len(hits) != 1:
        raise Refusal(
            rel,
            "parameter-pin-not-unique",
            "the text pin %r is the whole text of %d parameters of %s (the "
            "parameters are %r); a buffer witness must name exactly one, or the "
            "Rocq record is modelling a parameter nobody checked."
            % (pin, len(hits), symbol, [t.strip() for _o, t in params]),
        )
    return line_of(starts, hits[0])

def scan_file(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        raw = fh.read()
    return raw, cudatext.blank_out(raw), line_starts(raw)

def collect(cfg, decl):
    files = {}
    refusals = []
    corpus = cudatu.corpus(cfg, refusals)
    if refusals:
        return None, refusals
    for rel, path, _ent in corpus:
        files[rel] = scan_file(path)

    kernels = []
    claimed = []
    for idx, k in enumerate(decl["kernels"]):
        rel = k["file"]
        if rel not in files:
            refusals.append(
                (
                    rel,
                    "kernel-file-not-in-corpus",
                    "gen/pdl_kernels.json models %s here, but gen/sources.json "
                    "does not declare this file." % k["symbol"],
                )
            )
            continue
        raw, blanked, starts = files[rel]
        try:
            popen, pclose, bopen, bclose = kernel_extent(blanked, rel, k["symbol"])
            derefs, stores = deref_and_store_sites(blanked, rel, k["symbol"], bopen, bclose)
        except Refusal as e:
            refusals.append((e.where, e.kind, e.detail))
            continue
        body = blanked[bopen:bclose]
        pro = [bopen + m.start() for m in re.finditer(re.escape(PROLOG), body)]
        epi = [bopen + m.start() for m in re.finditer(re.escape(EPILOG), body)]
        if len(pro) != 1:
            refusals.append(
                (
                    "%s:%d" % (rel, line_of(starts, bopen)),
                    "prolog-count",
                    "%s holds %d %s() sites; the model gives each kernel one "
                    "prolog point." % (k["symbol"], len(pro), PROLOG),
                )
            )
            continue
        if len(epi) > 1:
            refusals.append(
                (
                    "%s:%d" % (rel, line_of(starts, bopen)),
                    "epilog-count",
                    "%s holds %d %s() sites; the model gives each kernel one "
                    "epilog point." % (k["symbol"], len(epi), EPILOG),
                )
            )
            continue
        claimed.append((rel, bopen, bclose))
        prolog_line = line_of(starts, pro[0])
        explicit = bool(epi)
        epilog_line = line_of(starts, epi[0]) if explicit else line_of(starts, bclose)
        try:
            params = split_params(raw, rel, k["symbol"], popen, pclose)
            bufs = {}
            for kind in ("produces", "consumes"):
                for buf, pins in sorted(k.get(kind, {}).items()):
                    if buf not in decl["buffers"]:
                        raise Refusal(rel, "unknown-buffer", "%r is not in the buffers list" % buf)
                    for pin in pins:
                        bufs.setdefault((kind, buf), []).append(
                            (pin, param_pin_line(params, rel, k["symbol"], pin, starts))
                        )
        except Refusal as e:
            refusals.append((e.where, e.kind, e.detail))
            continue
        kernels.append(
            dict(
                rocq=k["rocq"],
                file=rel,
                symbol=k["symbol"],
                k_id=idx,
                signature_line=line_of(starts, popen),
                body_open_line=line_of(starts, bopen),
                body_close_line=line_of(starts, bclose),
                prolog_line=prolog_line,
                epilog_line=epilog_line,
                explicit_epilog=explicit,
                deref_lines=lines(starts, derefs),
                store_lines=lines(starts, stores),
                produces=sorted(k.get("produces", {})),
                consumes=sorted(k.get("consumes", {})),
                witnesses={
                    "%s %s" % (kind, buf): [{"pin": p, "line": ln} for p, ln in v]
                    for (kind, buf), v in sorted(bufs.items())
                },
            )
        )

    modelled_symbols = set(k["symbol"] for k in kernels)
    macro_sites = []
    launch_sites = []
    for rel, (raw, blanked, starts) in sorted(files.items()):
        for macro in (PROLOG, EPILOG):
            for m in re.finditer(re.escape(macro) + r"\s*\(", blanked):
                off = m.start()
                inside = any(r == rel and lo < off < hi for r, lo, hi in claimed)
                macro_sites.append((rel, line_of(starts, off), macro, inside))
                if not inside:
                    refusals.append(
                        (
                            "%s:%d" % (rel, line_of(starts, off)),
                            "unmodelled-pdl-kernel",
                            "a %s() outside every kernel gen/pdl_kernels.json "
                            "declares. A kernel carrying the intrinsics that the "
                            "model does not mention is exactly the blind spot "
                            "this census exists to close: declare it." % macro,
                        )
                    )
        for m in LAUNCH_EX.finditer(blanked):
            callee = m.group(1).split("::")[-1]
            launch_sites.append((rel, line_of(starts, m.start()), callee))
            if callee not in modelled_symbols:
                refusals.append(
                    (
                        "%s:%d" % (rel, line_of(starts, m.start())),
                        "unmodelled-pdl-launch",
                        "cudaLaunchKernelEx launches %r, which "
                        "gen/pdl_kernels.json does not declare. This repo uses "
                        "cudaLaunchKernelEx only to pass "
                        "cudaLaunchAttributeProgrammaticStreamSerialization, so "
                        "every callee is a PDL consumer and needs a model." % callee,
                    )
                )
    launched = set(c for _r, _l, c in launch_sites)
    for k in kernels:
        k["pdl_launched"] = k["symbol"] in launched
    return dict(kernels=kernels, macro_sites=macro_sites, launch_sites=launch_sites), refusals

def vlist(xs):
    return "[" + "; ".join(str(x) for x in xs) + "]"

def emit(decl, data, files_sha):
    L = []
    w = L.append
    w("(* AUTO-GENERATED by rocq/gen/extract/pdl_sites.py -- DO NOT EDIT;")
    w("   re-run rocq/gen/run.sh.  `run.sh --check` fails when this file and the")
    w("   CUDA it describes have drifted apart, which is the whole reason it")
    w("   exists: the hand-written table it replaces carried twelve source line")
    w("   numbers as hypotheses, Rocq never compared one of them to a .cu, and")
    w("   all twelve were wrong while the proofs stayed green.")
    w("")
    w("   PdlOrder.v is the rule, PdlKernels.v is the pairing, this is the")
    w("   position data.  Every number below is an OUTPUT, read out of the file")
    w("   named beside it on the run that wrote this, at the sha256s recorded in")
    w("   gen/out/pdl_sites.json.")
    w("")
    w("   DERIVED: the prolog and epilog sites, and the two point sets --")
    w("   `<symbol>_every_store_line` (every assignment through a subscript or a")
    w("   dereference anywhere in the body) and `<symbol>_every_dereference_line`")
    w("   (every subscript or dereference at all).  Both are over-approximations")
    w("   on purpose: they cover shared memory, local arrays and untracked")
    w("   buffers, so `every store precedes the epilog` and `the prolog precedes")
    w("   every dereference` are strictly stronger than the per-buffer")
    w("   obligations PdlOrder states, and imply them.  Nobody hand-picks a last")
    w("   write, so nobody can pick the wrong one.")
    w("")
    w("   DECLARED, in gen/pdl_kernels.json: which __global__ symbol each record")
    w("   models, and which buffer each parameter carries -- each witnessed by")
    w("   the exact source text of the parameter declaration, required to occur")
    w("   exactly once in that kernel's parameter list.")
    w("*)")
    w("")
    w("From Stdlib Require Import List Arith.")
    w("Import ListNotations.")
    w("From SpeachesPlus Require Import PdlOrder.")
    w("")
    w("Lemma every_member_below :")
    w("  forall n l, forallb (fun p => Nat.ltb p n) l = true -> forall p, In p l -> p < n.")
    w("Proof.")
    w("  intros n l. induction l as [| a l IH]; intros H p Hp; [contradiction |].")
    w("  simpl in H. destruct (Nat.ltb a n) eqn:Ha; simpl in H; [| discriminate].")
    w("  destruct Hp as [Hp | Hp];")
    w("  [subst; apply Nat.ltb_lt; exact Ha | apply IH; assumption].")
    w("Qed.")
    w("")
    w("Lemma every_member_above :")
    w("  forall n l, forallb (fun p => Nat.ltb n p) l = true -> forall p, In p l -> n < p.")
    w("Proof.")
    w("  intros n l. induction l as [| a l IH]; intros H p Hp; [contradiction |].")
    w("  simpl in H. destruct (Nat.ltb n a) eqn:Ha; simpl in H; [| discriminate].")
    w("  destruct Hp as [Hp | Hp];")
    w("  [subst; apply Nat.ltb_lt; exact Ha | apply IH; assumption].")
    w("Qed.")
    w("")
    for i, b in enumerate(decl["buffers"]):
        w("Definition %s : buffer := %d." % (b, i))
    w("")

    for k in data["kernels"]:
        w("(* %s  %s:%d" % (k["symbol"], k["file"], k["signature_line"]))
        w("   body %d-%d, prolog %d, %s %d" % (
            k["body_open_line"], k["body_close_line"], k["prolog_line"],
            "epilog" if k["explicit_epilog"] else "no explicit epilog; kernel exit at",
            k["epilog_line"]))
        for role, ws in sorted(k["witnesses"].items()):
            for it in ws:
                w("   %s: `%s` at :%d" % (role, it["pin"], it["line"]))
        w("*)")
        w("Definition %s_every_store_line : list point := %s." % (k["symbol"], vlist(k["store_lines"])))
        w("Definition %s_every_dereference_line : list point := %s." % (k["symbol"], vlist(k["deref_lines"])))
        if k["explicit_epilog"]:
            w("Definition %s_epilog_line : point := %d." % (k["symbol"], k["epilog_line"]))
        else:
            w("Definition %s_epilog_is_kernel_exit_no_explicit_trigger : point := %d."
              % (k["symbol"], k["epilog_line"]))
        w("Definition %s : kernel :=" % k["rocq"])
        w("  {| k_id := %d; k_prolog := %d;" % (k["k_id"], k["prolog_line"]))
        if k["explicit_epilog"]:
            w("     k_epilog := %s_epilog_line;" % k["symbol"])
        else:
            w("     k_epilog := %s_epilog_is_kernel_exit_no_explicit_trigger;" % k["symbol"])
        w("     k_writes := fun b =>")
        body = "[]"
        for b in reversed(k["produces"]):
            body = "if Nat.eqb b %s then %s_every_store_line else %s" % (b, k["symbol"], body)
        w("       %s;" % body)
        w("     k_reads := fun b =>")
        body = "[]"
        for b in reversed(k["consumes"]):
            body = "if Nat.eqb b %s then %s_every_dereference_line else %s" % (b, k["symbol"], body)
        w("       %s |}." % body)
        w("")
        w("Theorem %s_prolog_precedes_every_dereference :" % k["rocq"])
        w("  forall p, In p %s_every_dereference_line -> k_prolog %s < p."
          % (k["symbol"], k["rocq"]))
        w("Proof. apply every_member_above. vm_compute. reflexivity. Qed.")
        w("")
        w("Theorem %s_epilog_follows_every_store :" % k["rocq"])
        w("  forall p, In p %s_every_store_line -> p < k_epilog %s."
          % (k["symbol"], k["rocq"]))
        w("Proof. apply every_member_below. vm_compute. reflexivity. Qed.")
        w("")
        for b in k["produces"]:
            w("Lemma %s_writes_%s_before_epilog : epilog_after_writes %s %s."
              % (k["rocq"], b, k["rocq"], b))
            w("Proof. unfold epilog_after_writes. apply every_member_below.")
            w("       vm_compute. reflexivity. Qed.")
            w("")
        for b in k["consumes"]:
            w("Lemma %s_reads_%s_after_prolog : prolog_before_reads %s %s."
              % (k["rocq"], b, k["rocq"], b))
            w("Proof. unfold prolog_before_reads. apply every_member_above.")
            w("       vm_compute. reflexivity. Qed.")
            w("")
        w("Definition %s_launched_with_programmatic_serialization : bool := %s."
          % (k["symbol"], "true" if k["pdl_launched"] else "false"))
        w("")

    pro = [s for s in data["macro_sites"] if s[2] == PROLOG]
    epi = [s for s in data["macro_sites"] if s[2] == EPILOG]
    w("(* Census, in both directions, as data rather than as a claim in a commit")
    w("   message.  Every prolog and epilog in the declared corpus sits inside a")
    w("   modelled kernel, and every cudaLaunchKernelEx callee is a modelled")
    w("   kernel -- that call is how this repo passes")
    w("   cudaLaunchAttributeProgrammaticStreamSerialization, through NVK_PDL_ATTR")
    w("   and hand-rolled alike. *)")
    for rel, line, macro, _inside in data["macro_sites"]:
        w("(*   %s:%d %s *)" % (rel, line, macro))
    for rel, line, callee in data["launch_sites"]:
        w("(*   %s:%d launches %s *)" % (rel, line, callee))
    w("Definition pdl_prolog_sites_in_corpus : nat := %d." % len(pro))
    w("Definition pdl_epilog_sites_in_corpus : nat := %d." % len(epi))
    w("Definition pdl_kernels_modelled : nat := %d." % len(data["kernels"]))
    w("Definition pdl_launch_sites_in_corpus : nat := %d." % len(data["launch_sites"]))
    w("")
    w("Theorem every_wired_kernel_is_modelled :")
    w("  pdl_prolog_sites_in_corpus = pdl_kernels_modelled.")
    w("Proof. reflexivity. Qed.")
    w("")
    unwired = [k for k in data["kernels"] if not k["explicit_epilog"] or not k["pdl_launched"]]
    if unwired:
        w("(* Modelled kernels whose PDL wiring is INCOMPLETE in the CUDA.  Each is")
        w("   safe -- without an explicit trigger the launch completes at kernel")
        w("   exit, and a kernel nothing launches with programmatic serialization")
        w("   is serialized the ordinary way (PdlOrder.unwired_neighbour_is_safe)")
        w("   -- but each is also a prolog that buys nothing today. *)")
        for k in unwired:
            w("(*   %s: explicit epilog %s, launched with programmatic serialization %s *)"
              % (k["symbol"], str(k["explicit_epilog"]).lower(), str(k["pdl_launched"]).lower()))
    w("Definition pdl_kernels_with_incomplete_wiring : nat := %d." % len(unwired))
    w("")
    w("(* The sources this file was read out of. *)")
    for rel in sorted(files_sha):
        w("(*   %s %s *)" % (rel, files_sha[rel]))
    w("")
    return "\n".join(L)

SELFTEST_SOURCES = {
    "multiplication is not a dereference, and a pointer declaration is not one either": (
        """
__global__ void k(const float* __restrict__ x, float* __restrict__ y) {
    const float* row = x + blockIdx.x * 4;
    float s = (1.f + 2.f) * 3.f;
    NVK_PDL_PROLOG();
    y[0] = row[1] * s;
    NVK_PDL_EPILOG();
}
""",
        {"derefs": [6], "stores": [6], "prolog": 5, "epilog": 7},
    ),
    "an explicit dereference read is a dereference, and a dereference store is a store": (
        """
__global__ void k(const int* __restrict__ n, int* __restrict__ y) {
    NVK_PDL_PROLOG();
    int total = *n;
    *y = total;
    NVK_PDL_EPILOG();
}
""",
        {"derefs": [4, 5], "stores": [5], "prolog": 3, "epilog": 6},
    ),
    "compound assignment through a subscript is a store; a comparison is not": (
        """
__global__ void k(int* __restrict__ y) {
    NVK_PDL_PROLOG();
    if (y[0] == 3) y[1] += 2;
    y[2][3] = 4;
    NVK_PDL_EPILOG();
}
""",
        {"derefs": [4, 5], "stores": [4, 5], "prolog": 3, "epilog": 6},
    ),
    "a store after the epilog is EMITTED, not refused, so Rocq is what fails": (
        """
__global__ void k(int* __restrict__ y) {
    NVK_PDL_PROLOG();
    NVK_PDL_EPILOG();
    y[0] = 1;
}
""",
        {"derefs": [5], "stores": [5], "prolog": 3, "epilog": 4},
    ),
    "a dereference before the prolog is EMITTED, not refused, for the same reason": (
        """
__global__ void k(const int* __restrict__ x, int* __restrict__ y) {
    int v = x[0];
    NVK_PDL_PROLOG();
    y[0] = v;
    NVK_PDL_EPILOG();
}
""",
        {"derefs": [3, 5], "stores": [5], "prolog": 4, "epilog": 6},
    ),
}

SELFTEST_PINS = {
    "a pin must be a WHOLE parameter, never a substring of a renamed one": (
        "__global__ void k(float* __restrict__ y_out, float* __restrict__ z) { }",
        "float* __restrict__ y",
        None,
    ),
    "a pin matches the parameter it names, whitespace and all": (
        "__global__ void k(float*   __restrict__ y,\n                  float* z) { }",
        "float*   __restrict__ y",
        1,
    ),
}

SELFTEST_REFUSALS = {
    "symbol-not-unique": """
__global__ void k(int* y) { NVK_PDL_PROLOG(); y[0] = 1; }
__global__ void k(int* y, int n) { NVK_PDL_PROLOG(); y[0] = n; }
""",
    "opaque-memory-op": """
__global__ void k(int* y) { NVK_PDL_PROLOG(); asm volatile("st.global.u32 [%0], %1;"); }
""",
    "double-star": """
__global__ void k(int** y) { NVK_PDL_PROLOG(); int v = **y; }
""",
}

def self_test():
    """Assert the classifier's edges and every refusal path, hermetically.

    An extractor whose refusals are never exercised reports a coverage number
    nobody can contradict.  The two EMITTED cases are the load-bearing ones:
    an ordering violation must reach GenPdl.v so the proof fails, because a
    refusal there would leave the previous, still-green file on disk.
    """
    bad = 0
    for name, (src, want) in sorted(SELFTEST_SOURCES.items()):
        blanked = cudatext.blank_out(src)
        starts = line_starts(src)
        _po, _pc, bopen, bclose = kernel_extent(blanked, "<selftest>", "k")
        derefs, stores = deref_and_store_sites(blanked, "<selftest>", "k", bopen, bclose)
        body = blanked[bopen:bclose]
        got = {
            "derefs": lines(starts, derefs),
            "stores": lines(starts, stores),
            "prolog": line_of(starts, bopen + body.index(PROLOG)),
            "epilog": line_of(starts, bopen + body.index(EPILOG)),
        }
        if got != want:
            bad += 1
            print("[%s] SELFTEST FAIL %s: %r != %r" % (TAG, name, got, want), file=sys.stderr)
        else:
            print("[%s] selftest ok: %s" % (TAG, name))
    for name, (src, pin, want) in sorted(SELFTEST_PINS.items()):
        blanked = cudatext.blank_out(src)
        starts = line_starts(src)
        popen, pclose, _bo, _bc = kernel_extent(blanked, "<selftest>", "k")
        params = split_params(src, "<selftest>", "k", popen, pclose)
        try:
            got = param_pin_line(params, "<selftest>", "k", pin, starts)
        except Refusal as e:
            got = None if e.kind == "parameter-pin-not-unique" else e.kind
        if got != want:
            bad += 1
            print("[%s] SELFTEST FAIL %s: %r != %r" % (TAG, name, got, want), file=sys.stderr)
        else:
            print("[%s] selftest ok: %s" % (TAG, name))
    for kind, src in sorted(SELFTEST_REFUSALS.items()):
        blanked = cudatext.blank_out(src)
        try:
            _po, _pc, bopen, bclose = kernel_extent(blanked, "<selftest>", "k")
            deref_and_store_sites(blanked, "<selftest>", "k", bopen, bclose)
        except Refusal as e:
            if e.kind == kind:
                print("[%s] selftest ok: refuses %s" % (TAG, kind))
                continue
            bad += 1
            print("[%s] SELFTEST FAIL: expected %s, got %s" % (TAG, kind, e.kind), file=sys.stderr)
            continue
        bad += 1
        print("[%s] SELFTEST FAIL: %s did not refuse" % (TAG, kind), file=sys.stderr)
    print("[%s] selftest: %d case(s) failed" % (TAG, bad))
    return 1 if bad else 0

def main(argv):
    if "--self-test" in argv:
        return self_test()
    check_only = "--check" in argv
    with open(os.path.join(GEN, "pdl_kernels.json")) as fh:
        decl = json.load(fh)
    cfg = cudatu.load_sources(TAG)
    data, refusals = collect(cfg, decl)
    if refusals:
        cudatu.report_refusals(TAG, refusals)
        return 1
    files_sha = {}
    for k in data["kernels"]:
        files_sha[k["file"]] = cudatu.sha256_file(os.path.join(REPO, k["file"]))
    text = emit(decl, data, files_sha)
    if check_only:
        cur = open(OUT_V).read() if os.path.exists(OUT_V) else ""
        if cur != text:
            print(
                "[%s] STALE: %s does not match a regeneration from the current "
                "sources." % (TAG, OUT_V),
                file=sys.stderr,
            )
            import difflib

            for ln in list(
                difflib.unified_diff(
                    cur.splitlines(), text.splitlines(), "on-disk", "regenerated", lineterm=""
                )
            )[:80]:
                print(ln, file=sys.stderr)
            return 1
        print("[%s] --check: GenPdl.v matches its sources." % TAG)
        return 0
    cudatu.write_atomic(OUT_V, text)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    cudatu.write_atomic(
        OUT_JSON,
        json.dumps(
            {
                "generator": "rocq/gen/extract/pdl_sites.py",
                "files": files_sha,
                "kernels": data["kernels"],
                "macro_sites": [
                    {"file": r, "line": l, "macro": m, "inside_modelled_kernel": i}
                    for r, l, m, i in data["macro_sites"]
                ],
                "launch_sites": [
                    {"file": r, "line": l, "callee": c} for r, l, c in data["launch_sites"]
                ],
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
    )
    print(
        "[%s] %d kernels modelled from %d files; %d prolog and %d epilog sites in "
        "the corpus, %d cudaLaunchKernelEx sites, all accounted for"
        % (
            TAG,
            len(data["kernels"]),
            len(files_sha),
            len([s for s in data["macro_sites"] if s[2] == PROLOG]),
            len([s for s in data["macro_sites"] if s[2] == EPILOG]),
            len(data["launch_sites"]),
        )
    )
    for k in data["kernels"]:
        print(
            "[%s] %-38s %s:%d prolog %d %s %d, %d stores, %d dereferences, "
            "programmatic serialization %s"
            % (
                TAG,
                k["symbol"],
                os.path.basename(k["file"]),
                k["signature_line"],
                k["prolog_line"],
                "epilog" if k["explicit_epilog"] else "EXIT(no trigger)",
                k["epilog_line"],
                len(k["store_lines"]),
                len(k["deref_lines"]),
                k["pdl_launched"],
            )
        )
    return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
