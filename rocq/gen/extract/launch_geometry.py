#!/usr/bin/env python3
"""Extract every CUDA launch geometry in nv-kernels and emit rocq/GenLaunch.v.

LaunchGeometry.v is the RULE (grid, launchable, the CUDA maxGridDim constants,
the historical dequant_old existence theorem).  This emits the INSTANCES: one
`grid`-valued definition per launch site, parameterised over the free
identifiers that reach an axis, plus a `launchable` theorem discharged from
the bounds declared in gen/launch_symbols.json.

WHAT IS MECHANISED AND WHAT IS DECLARED -- read this before trusting a number.

  Mechanised from the CUDA, re-derived on every run: which launches exist,
  which kernel each one calls, which expression reaches gridDim.{x,y,z}, and
  the resolution chain (DeclRefExpr -> VarDecl -> dim3 constructor argument,
  or ParmDecl straight through).  Nothing here is transcribed by hand and
  nothing is keyed on a line number.

  DECLARED, not derived: the numeric range of each free identifier.  Whether
  `n_tokens` is a whole context or one prefill chunk is a property of the
  CALLER, in Rust, not of the .cu file -- no amount of CUDA parsing recovers
  it.  Those ranges live in gen/launch_symbols.json with a witness apiece.
  The extractor REFUSES on any identifier that reaches an axis and is not in
  that table, which is what makes "someone adds a launcher with a sequence
  length on y" a build failure instead of a silently larger corpus.

The anti-silent-skip gate.  Two independent parsers must agree:
  A. a text scan (extract/cudatext.py) counts every `<<<` after blanking
     comments and literals -- 187 sites today;
  B. the libclang walk yields launch instantiations, each carrying the
     SPELLING offset of its `<<<` -- 275 today, because macro bodies expand.
Every textual site must receive at least one AST launch and every AST launch
must land on a textual site, or this refuses.  That gate is not decoration:
using clang_Location_isFromMainFile as the main-file predicate (the obvious
choice) drops every macro-expanded launch, measured at 93 of 275, and the only
symptom is a smaller, still-green number.
"""

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import clangbind as CB  # noqa: E402
import cudatext  # noqa: E402
import cudatu  # noqa: E402
from clangbind import CXCursor, CHILD_CONTINUE, CHILD_RECURSE  # noqa: E402

TAG = "launch"
GEN = cudatu.GEN
ROCQ = cudatu.ROCQ
REPO = cudatu.REPO
OUT_V = os.path.join(ROCQ, "GenLaunch.v")
OUT_JSON = os.path.join(GEN, "out", "launch_geometry.json")

MAX_GRID = {"x": 2147483647, "y": 65535, "z": 65535}
AXES = ("x", "y", "z")

CAST_KINDS = {
    "ParenExpr",
    "UnexposedExpr",
    "CStyleCastExpr",
    "CXXStaticCastExpr",
    "CXXFunctionalCastExpr",
    "CXXConstCastExpr",
    "FirstExpr",
}
NONVALUE_KINDS = {
    "TypeRef",
    "TemplateRef",
    "NamespaceRef",
    "OverloadedDeclRef",
    "LabelRef",
    "MemberRef",
    "VariableRef",
    "UnexposedAttr",
}
BINOPS = {"+": "+", "-": "-", "*": "*", "/": "/"}

CMP_OPS = ("<", "<=", ">", ">=")
MIRROR = {"<": ">", "<=": ">=", ">": "<", ">=": "<="}
NEGATE = {"<": ">=", "<=": ">", ">": "<=", ">=": "<"}
DECL_KINDS = ("VarDecl", "ParmDecl", "NonTypeTemplateParameter")

CONTROL_HAZARDS = ("GotoStmt", "IndirectGotoStmt", "LabelStmt")
SWITCH_CUT = ("SwitchStmt", "CaseStmt", "DefaultStmt")
PATH_HAZARDS = ("LambdaExpr", "StmtExpr", "BlockExpr")

PROOF_BY = {
    "+": "lia",
    "-": "lia",
    "*": "nia",
    "/": "split; [apply Z.div_le_lower_bound; nia | apply Z.div_le_upper_bound; nia]",
}

def poly_add(a, b):
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, 0) + c
        if out[m] == 0:
            del out[m]
    return out

def poly_scale(a, k):
    return {m: c * k for m, c in a.items() if c * k != 0}

def poly_mul(a, b):
    out = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            m = tuple(sorted(ma + mb))
            out[m] = out.get(m, 0) + ca * cb
            if out[m] == 0:
                del out[m]
    return out

def poly_bounds(p, env):
    lo = hi = 0
    for m, c in p.items():
        mlo, mhi = 1, 1
        for v in m:
            vlo, vhi = env[v]
            if vlo < 0:
                raise Refusable("identifier %r has a negative declared lower bound" % v)
            mlo *= vlo
            mhi *= vhi
        if c >= 0:
            lo += c * mlo
            hi += c * mhi
        else:
            lo += c * mhi
            hi += c * mlo
    return (lo, hi)

class Term:
    """Grid-axis expression: a literal, an identifier, or a binary node."""

    def __init__(self, op, a=None, b=None, val=None, name=None):
        self.op = op
        self.a = a
        self.b = b
        self.val = val
        self.name = name

    @staticmethod
    def lit(v):
        return Term("lit", val=v)

    @staticmethod
    def var(n):
        return Term("var", name=n)

    def idents(self):
        if self.op == "var":
            return {self.name}
        if self.op in BINOPS:
            return self.a.idents() | self.b.idents()
        return set()

    def rocq(self, top=True):
        if self.op == "lit":
            return str(self.val)
        if self.op == "var":
            return self.name
        inner = "%s %s %s" % (self.a.rocq(False), self.op, self.b.rocq(False))
        return inner if top else "(" + inner + ")"

    def c_like(self):
        if self.op == "lit":
            return str(self.val)
        if self.op == "var":
            return self.name
        return "(%s %s %s)" % (self.a.c_like(), self.op, self.b.c_like())

    def interval(self, env):
        """Bounds via a POLYNOMIAL normal form, not naive interval propagation.

        Naive propagation loses the tiling idiom: `(T + tile - 1) / tile` has
        numerator [1+32-1, 262144+256-1] and divisor [32, 256], so the quotient
        floor is 32/256 = 0 and the axis looks as if it could be zero.  It
        cannot -- the two occurrences of `tile` are the same variable.
        Normalising to a polynomial cancels them, and that is also exactly the
        form `nia` sees when it checks the emitted assertion, so the bound this
        computes and the bound Rocq can prove are the same bound by
        construction.
        """
        return poly_bounds(self.poly(env), env)

    def poly(self, env):
        """{monomial (sorted tuple of names) -> integer coefficient}."""
        if self.op == "lit":
            return {(): self.val}
        if self.op == "var":
            return {(self.name,): 1}
        pa = self.a.poly(env)
        pb = self.b.poly(env)
        if self.op == "+":
            return poly_add(pa, pb)
        if self.op == "-":
            return poly_add(pa, poly_scale(pb, -1))
        if self.op == "*":
            return poly_mul(pa, pb)
        if self.op == "/":
            return {(self.div_atom(env),): 1}
        raise Refusable("unhandled operator %r" % self.op)

    def div_atom(self, env):
        """Register `a / b` as an opaque atom with proof-checkable bounds.

        L is the largest integer with `b * L <= a` provable from the declared
        ranges, U the smallest with `a <= b * U`.  Those are literally the side
        conditions Z.div_le_lower_bound / Z.div_le_upper_bound leave behind, so
        an L or U found here is one nia can discharge.
        """
        key = "d_" + self.rocq()
        if key in env:
            return key
        pa = self.a.poly(env)
        pb = self.b.poly(env)
        lb, hb = poly_bounds(pb, env)
        la, ha = poly_bounds(pa, env)
        if lb < 1:
            raise Refusable(
                "%s: the divisor's declared range starts at %d, so the quotient "
                "is neither bounded nor safe (a zero extent is an invalid launch)"
                % (self.c_like(), lb)
            )
        if la < 0:
            raise Refusable("%s: the numerator may be negative" % self.c_like())

        def ok_lower(q):
            return poly_bounds(poly_add(pa, poly_scale(pb, -q)), env)[0] >= 0

        def ok_upper(q):
            return poly_bounds(poly_add(poly_scale(pb, q), poly_scale(pa, -1)), env)[0] >= 0

        lo, hi = 0, max(1, ha // lb + 2)
        while not ok_lower(lo):
            raise Refusable("%s: cannot even show the quotient is nonnegative" % self.c_like())
        while ok_lower(hi):
            hi *= 2
            if hi > 1 << 62:
                raise Refusable("%s: quotient lower bound diverges" % self.c_like())
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if ok_lower(mid):
                lo = mid
            else:
                hi = mid
        L = lo
        ulo, uhi = 0, max(1, -((-ha) // lb) + 2)
        while not ok_upper(uhi):
            uhi *= 2
            if uhi > 1 << 62:
                raise Refusable("%s: quotient upper bound diverges" % self.c_like())
        while ulo + 1 < uhi:
            mid = (ulo + uhi) // 2
            if ok_upper(mid):
                uhi = mid
            else:
                ulo = mid
        U = uhi if ok_upper(uhi) else ulo
        if not ok_upper(U):
            raise Refusable("%s: no provable upper bound for the quotient" % self.c_like())
        env[key] = (L, U)
        return key

    def subterms(self):
        """Post-order over non-leaf nodes: the order the proof asserts them."""
        if self.op in BINOPS:
            return self.a.subterms() + self.b.subterms() + [self]
        return []

    def eval_at(self, point):
        if self.op == "lit":
            return self.val
        if self.op == "var":
            return point[self.name]
        a = self.a.eval_at(point)
        b = self.b.eval_at(point)
        if self.op == "+":
            return a + b
        if self.op == "-":
            return a - b
        if self.op == "*":
            return a * b
        if self.op == "/":
            return a // b
        raise Refusable("unhandled operator %r" % self.op)

class Refusable(Exception):
    pass

ONE = Term.lit(1)

class Analyzer:
    def __init__(self, cl, path, rel, raw, table):
        self.cl = cl
        self.path = path
        self.rel = rel
        self.raw = raw
        self.notes = []
        self.identdecl = {}
        self.table = table

    def where(self, cur):
        name, line, _col, _off = self.cl.spelling_loc(cur)
        base = os.path.relpath(name, REPO) if name.startswith(REPO) else name
        return "%s:%d" % (base, line)

    def src(self, cur):
        a, b = self.cl.spelling_extent(cur)
        if a[0] != b[0] or not a[0]:
            return "<multi-file extent>"
        try:
            with open(a[0], "rb") as fh:
                blob = fh.read()
        except OSError:
            return "<unreadable>"
        return " ".join(blob[a[3] : b[3]].decode(errors="replace").split())

    def valkids(self, cur):
        return [k for k in self.cl.children(cur) if self.cl.kind(k) not in NONVALUE_KINDS]

    def unwrap(self, cur):
        seen = 0
        while self.cl.kind(cur) in CAST_KINDS:
            kids = self.valkids(cur)
            if len(kids) != 1:
                return cur
            cur = kids[0]
            seen += 1
            if seen > 32:
                raise Refusable("cast/paren nesting deeper than 32")
        return cur

    def reassigned(self, vardecl, fn_extent):
        """True if the variable is stored to anywhere in the enclosing function.

        Reading a `dim3 grid(...)` constructor is a lie about the launched
        geometry if anything assigns to grid (or to a scalar that feeds it)
        between the declaration and the launch.  Rather than model control
        flow, refuse on any store at all.
        """
        hits = []
        vd_loc = self.cl.spelling_loc(vardecl)

        def visit(cc, _pp):
            k = self.cl.kind(cc)
            if k == "BinaryOperator":
                op = self.cl.binop(cc)
                if op in ("=", "+=", "-=", "*=", "/=", "%=", "<<=", ">>=", "&=", "|=", "^="):
                    kids = self.valkids(cc)
                    if kids and self._refs(kids[0], vd_loc):
                        hits.append(cc)
            elif k == "CompoundAssignOperator":
                kids = self.valkids(cc)
                if kids and self._refs(kids[0], vd_loc):
                    hits.append(cc)
            elif k == "UnaryOperator":
                if self.cl.unop(cc) in ("++", "--"):
                    kids = self.valkids(cc)
                    if kids and self._refs(kids[0], vd_loc):
                        hits.append(cc)
            elif k == "MemberRefExpr":
                pass
            return CHILD_RECURSE

        self.cl.walk(fn_extent, visit)
        return hits

    def address_taken(self, decl, fn_extent):
        """`&ident` anywhere in the function.

        `reassigned` only sees stores written through the name itself, so a
        guard on a value whose address escapes proves nothing: `*p = 262144`
        after `int* p = &m` is invisible to it.  Enforcement therefore drops
        any bound on a declaration whose address is taken at all, rather than
        try to decide what the pointer was used for.
        """
        hits = []
        loc = self.cl.spelling_loc(decl)

        def visit(cc, _pp):
            if self.cl.kind(cc) == "UnaryOperator" and self.cl.unop(cc) == "&":
                kids = self.valkids(cc)
                if kids and self._refs(kids[0], loc):
                    hits.append(cc)
            return CHILD_RECURSE

        self.cl.walk(fn_extent, visit)
        return hits

    def _refs(self, cur, vd_loc):
        cur = self.unwrap(cur)
        if self.cl.kind(cur) == "MemberRefExpr":
            kids = self.valkids(cur)
            if kids:
                cur = self.unwrap(kids[0])
        if self.cl.kind(cur) != "DeclRefExpr":
            return False
        ref = self.cl.referenced(cur)
        if self.cl.is_null(ref):
            return False
        return self.cl.spelling_loc(ref) == vd_loc

    def _var(self, ref, name):
        """Term.var, remembering WHICH declaration this name came from.

        Guard matching compares declarations, not spellings: two `int m` in one
        translation unit are two identifiers, and a guard on one says nothing
        about the other.  A name seen resolving to more than one declaration is
        recorded as such and can never be matched.
        """
        self.identdecl.setdefault(name, set()).add(self.cl.spelling_loc(ref))
        return Term.var(name)

    def translate(self, cur, fn, depth=0):
        """CUDA expression -> Term.  Refuses on anything it cannot name."""
        if depth > 12:
            raise Refusable("expression resolution deeper than 12 levels")
        folded = self.cl.eval_int(cur)
        if folded is not None:
            return Term.lit(folded)
        cur = self.unwrap(cur)
        k = self.cl.kind(cur)
        if k == "IntegerLiteral":
            raise Refusable("integer literal that libclang would not constant-fold: %r" % self.src(cur))
        if k == "BinaryOperator":
            op = self.cl.binop(cur)
            if op not in BINOPS:
                raise Refusable(
                    "binary operator %r in a grid axis (only + - * / are modelled): %r"
                    % (op, self.src(cur))
                )
            kids = self.valkids(cur)
            if len(kids) != 2:
                raise Refusable("binary operator with %d value operands" % len(kids))
            return Term(BINOPS[op], self.translate(kids[0], fn, depth + 1), self.translate(kids[1], fn, depth + 1))
        if k == "DeclRefExpr":
            ref = self.cl.referenced(cur)
            if self.cl.is_null(ref):
                raise Refusable("unresolved identifier %r" % self.src(cur))
            rk = self.cl.kind(ref)
            name = self.cl.spelling(ref)
            if rk == "VarDecl":
                if lookup_bound(self.table, self.rel, name)[1] is not None:
                    self.notes.append(
                        "%s: range DECLARED in launch_symbols.json, overriding its "
                        "initialiser" % name
                    )
                    return self._var(ref, name)
                stores = self.reassigned(ref, fn) if fn is not None else []
                if stores:
                    self.notes.append(
                        "%s assigned at %s: value DECLARED, not derived"
                        % (name, ",".join(self.where(s) for s in stores))
                    )
                    return self._var(ref, name)
                inits = self.valkids(ref)
                if len(inits) == 1:
                    try:
                        return self.translate(inits[0], fn, depth + 1)
                    except Refusable:
                        return self._var(ref, name)
                return self._var(ref, name)
            if rk in ("ParmDecl", "NonTypeTemplateParameter"):
                return self._var(ref, name)
            raise Refusable("grid axis reaches a %s (%r), which has no value this extractor can name" % (rk, name))
        if k == "CallExpr":
            ref = self.cl.referenced(cur)
            nm = self.cl.spelling(ref) if not self.cl.is_null(ref) else "<unresolved>"
            raise Refusable(
                "call to %r in a grid axis that is not constexpr-evaluable: %r"
                % (nm, self.src(cur))
            )
        raise Refusable("unmodelled expression kind %s in a grid axis: %r" % (k, self.src(cur)))

    def split_logic(self, cur, op):
        """Flatten a top-level `&&` or `||` chain; anything else is one part."""
        cur = self.unwrap(cur)
        if self.cl.kind(cur) == "BinaryOperator" and self.cl.binop(cur) == op:
            kids = self.valkids(cur)
            if len(kids) == 2:
                return self.split_logic(kids[0], op) + self.split_logic(kids[1], op)
        return [cur]

    def cmp_upper(self, cur, negate):
        """`cur` (negated if asked) being TRUE implies `ident <= K`.

        Returns (name, decl-location, K, decl-cursor, where, source) or None.
        One side must constant-fold and the other must be a plain reference to
        a variable, parameter or non-type template parameter; anything else --
        a member, a call, a deref, both sides constant -- yields None, because
        this cannot say which runtime quantity was bounded.
        """
        cur = self.unwrap(cur)
        if self.cl.kind(cur) != "BinaryOperator":
            return None
        op = self.cl.binop(cur)
        if op not in CMP_OPS:
            return None
        kids = self.valkids(cur)
        if len(kids) != 2:
            return None
        lv = self.cl.eval_int(kids[0])
        rv = self.cl.eval_int(kids[1])
        if (lv is None) == (rv is None):
            return None
        if rv is None:
            op, k, side = MIRROR[op], lv, kids[1]
        else:
            k, side = rv, kids[0]
        if negate:
            op = NEGATE[op]
        if op == "<=":
            upper = k
        elif op == "<":
            upper = k - 1
        else:
            return None
        side = self.unwrap(side)
        if self.cl.kind(side) != "DeclRefExpr":
            return None
        ref = self.cl.referenced(side)
        if self.cl.is_null(ref) or self.cl.kind(ref) not in DECL_KINDS:
            return None
        return (
            self.cl.spelling(ref),
            self.cl.spelling_loc(ref),
            upper,
            CXCursor(ref.kind, ref.xdata, ref.data),
            self.where(cur),
            self.src(cur),
        )

    def cond_uppers(self, cond, held):
        """Upper bounds implied by a condition that HELD (or that FAILED).

        held=True is the guarded-block shape (`if (i <= K) { launch }`): the
        whole condition is true, so every top-level `&&` conjunct is true.
        held=False is the early-return and else-branch shape: the condition is
        false, so by De Morgan every top-level `||` disjunct is false.  The
        other two pairings -- a conjunct of a failed `&&`, a disjunct of a held
        `||` -- imply nothing about any single operand and are not split.
        """
        parts = self.split_logic(cond, "&&" if held else "||")
        out = []
        for p in parts:
            b = self.cmp_upper(p, not held)
            if b is not None:
                out.append(b)
        return out

    def dim3_axes(self, cur, fn, depth=0):
        """Grid argument -> (Term, Term, Term) plus the resolution chain."""
        if depth > 4:
            raise Refusable("dim3 resolution deeper than 4 levels")
        cur = self.unwrap(cur)
        k = self.cl.kind(cur)
        if k == "DeclRefExpr":
            ref = self.cl.referenced(cur)
            if self.cl.is_null(ref):
                raise Refusable("grid argument resolves to no declaration")
            if self.cl.kind(ref) == "ParmDecl" and "dim3" in self.cl.type_spelling(ref):
                nm = self.cl.spelling(ref)
                return (
                    tuple(Term.var("%s_%s" % (nm, a)) for a in AXES),
                    ["dim3 parameter %s of %s: the geometry is the CALLER's, so all "
                     "three extents are free identifiers requiring declaration"
                     % (nm, self.where(ref))],
                )
            if self.cl.kind(ref) != "VarDecl":
                raise Refusable(
                    "grid argument resolves to %s, not a dim3 variable"
                    % self.cl.kind(ref)
                )
            stores = self.reassigned(ref, fn) if fn is not None else []
            if stores:
                raise Refusable(
                    "dim3 %r is assigned to at %s after construction; reading the "
                    "constructor would misreport the launched geometry"
                    % (self.cl.spelling(ref), ", ".join(self.where(s) for s in stores))
                )
            inits = self.valkids(ref)
            if len(inits) != 1:
                raise Refusable("dim3 %r has %d initialisers" % (self.cl.spelling(ref), len(inits)))
            sub, subchain = self.dim3_axes(inits[0], fn, depth + 1)
            return sub, ["%s (%s)" % (self.cl.spelling(ref), self.where(ref))] + subchain
        if k in ("CallExpr", "CXXFunctionalCastExpr", "CXXTemporaryObjectExpr", "CXXConstructExpr"):
            kids = self.valkids(cur)
            if len(kids) == 1 and self.cl.kind(self.unwrap(kids[0])) == "DeclRefExpr":
                inner = self.unwrap(kids[0])
                ref = self.cl.referenced(inner)
                if not self.cl.is_null(ref) and "dim3" in self.cl.type_spelling(ref):
                    sub, subchain = self.dim3_axes(inner, fn, depth + 1)
                    return sub, ["copy of dim3"] + subchain
            if not kids or len(kids) > 3:
                raise Refusable("dim3 constructor with %d arguments" % len(kids))
            terms = [self.translate(a, fn) for a in kids]
            while len(terms) < 3:
                terms.append(ONE)
            return tuple(terms), ["dim3(%s)" % ", ".join(t.c_like() for t in terms)]
        raise Refusable("grid argument is a %s, not a dim3 construction or variable: %r" % (k, self.src(cur)))

def always_returns(cl, cur):
    """The statement leaves the function on every path through it."""
    k = cl.kind(cur)
    if k == "ReturnStmt":
        return True
    if k == "CompoundStmt":
        kids = cl.children(cur)
        return bool(kids) and always_returns(cl, kids[-1])
    return False

def stmt_path(cl, fn_cur, off):
    """Cursor chain from the function down to the innermost node holding `off`.

    Offsets are EXPANSION offsets on both sides, so a launch written inside a
    macro body is located at its invocation site, which is where its enclosing
    statements actually are.
    """
    out = []
    cur = fn_cur
    for _ in range(64):
        nxt = None
        for c in cl.children(cur):
            a, b = cl.expansion_extent(c)
            if a[0] and a[3] <= off <= b[3]:
                nxt = c
                break
        if nxt is None:
            return out
        out.append(nxt)
        cur = nxt
    return out

def same_cursor(cl, a, b):
    return cl.kind(a) == cl.kind(b) and cl.expansion_extent(a) == cl.expansion_extent(b)

def collect_guards(an, fn_cur, off):
    """Upper bounds this launch may ASSUME because a guard dominates it.

    Only two shapes are read, and both are checked structurally rather than
    textually:

      early return   a sibling `if (C) <stmt that always returns>` with NO else,
                     lexically before the statement holding the launch in an
                     enclosing compound statement.  Reaching the launch means C
                     was false.
      branch         the launch sits in the then-branch (C held) or the
                     else-branch (C failed) of an enclosing `if`.

    Domination is textual precedence inside nested compound statements, which
    is sound only because a function containing any goto, label or switch is
    rejected outright, and because a launch reached through a lambda or
    statement-expression body is rejected too.  A bound is dropped if anything
    in the function assigns to the guarded declaration: `reassigned` is
    whole-function, so "the store happened after the guard" never has to be
    decided.
    """
    if fn_cur is None:
        return {}, "the launch is not inside a function this extractor located", {}

    bad = []

    def scan(cc, _pp):
        if an.cl.kind(cc) in CONTROL_HAZARDS:
            bad.append(an.cl.kind(cc))
        return CHILD_RECURSE

    an.cl.walk(fn_cur, scan)
    if bad:
        return {}, "host function contains %s; lexical order does not imply domination" % (
            ", ".join(sorted(set(bad)))
        ), {}

    path = stmt_path(an.cl, fn_cur, off)
    if not path:
        return {}, "could not locate the launch inside the function body", {}
    for c in path:
        if an.cl.kind(c) in PATH_HAZARDS:
            return {}, "the launch is inside a %s, whose execution point is not its definition point" % an.cl.kind(c), {}

    raw = {}

    def add(hits, why):
        for name, loc, upper, decl, where, src in hits:
            raw.setdefault(name, []).append((upper, loc, decl, where, src, why))

    cut = len(path)
    for i, c in enumerate(path):
        if an.cl.kind(c) in SWITCH_CUT:
            cut = i
            break

    for i, child in enumerate(path):
        if i > cut:
            break
        parent = fn_cur if i == 0 else path[i - 1]
        pk = an.cl.kind(parent)
        if pk == "CompoundStmt":
            cstart = an.cl.expansion_extent(child)[0][3]
            for s in an.cl.children(parent):
                if an.cl.kind(s) != "IfStmt":
                    continue
                if an.cl.expansion_extent(s)[1][3] >= cstart:
                    continue
                kids = an.cl.children(s)
                if len(kids) != 2 or not always_returns(an.cl, kids[1]):
                    continue
                add(an.cond_uppers(kids[0], False), "early return at %s" % an.where(s))
        elif pk == "IfStmt":
            kids = an.cl.children(parent)
            if len(kids) < 2:
                continue
            if same_cursor(an.cl, child, kids[1]):
                add(an.cond_uppers(kids[0], True), "then-branch of the if at %s" % an.where(parent))
            elif len(kids) > 2 and same_cursor(an.cl, child, kids[2]):
                add(an.cond_uppers(kids[0], False), "else-branch of the if at %s" % an.where(parent))

    guards = {}
    rejected = {}

    def reject(name, where, src, reason):
        rejected.setdefault(name, "the guard at %s (%s) does not count: %s" % (where, src, reason))

    for name, locs in sorted(an.identdecl.items()):
        if len(locs) != 1:
            for upper, gloc, decl, where, src, why in raw.get(name, []):
                reject(
                    name,
                    where,
                    src,
                    "%d different declarations named %r reach this axis, so which one "
                    "the guard bounds is ambiguous" % (len(locs), name),
                )
            continue
        loc = next(iter(locs))
        best = None
        for upper, gloc, decl, where, src, why in raw.get(name, []):
            if gloc != loc:
                reject(
                    name,
                    where,
                    src,
                    "it bounds a different declaration of %r than the one reaching "
                    "the axis" % name,
                )
                continue
            if an.reassigned(decl, fn_cur):
                reject(name, where, src, "%r is assigned again in this function after it" % name)
                continue
            if an.address_taken(decl, fn_cur):
                reject(
                    name,
                    where,
                    src,
                    "%r has its address taken, so it may be written through a pointer" % name,
                )
                continue
            if best is None or upper < best["upper"]:
                best = {"upper": upper, "where": where, "cond": src, "why": why}
        if best is not None:
            guards[name] = best
            rejected.pop(name, None)
    return guards, None, rejected

def enclosing_functions(cl, path):
    """[(lo, hi, name, cursor)] over top-level function bodies in this file."""
    out = []
    root = cl.lib.clang_getTranslationUnitCursor(path[1])
    fnkinds = {"FunctionDecl", "CXXMethod", "FunctionTemplate", "Constructor", "Destructor"}

    def visit(cc, _pp):
        if cl.kind(cc) in fnkinds:
            a, b = cl.expansion_extent(cc)
            if a[0] == path[0]:
                out.append((a[3], b[3], cl.spelling(cc), CXCursor(cc.kind, cc.xdata, cc.data)))
            return CHILD_CONTINUE
        return CHILD_RECURSE

    cl.walk(root, visit)
    return out

def collect_launches(cl, path, tu):
    launches = []

    def visit(cc, _pp):
        name, _ln, _col, _off = cl.expansion_loc(cc)
        if name != path:
            return CHILD_CONTINUE
        if cl.kind(cc) == "CallExpr":
            kids = cl.children(cc)
            if len(kids) >= 2:
                ref = cl.referenced(kids[1])
                if not cl.is_null(ref) and cl.spelling(ref) == "__cudaPushCallConfiguration":
                    launches.append(CXCursor(cc.kind, cc.xdata, cc.data))
        return CHILD_RECURSE

    cl.walk(cl.lib.clang_getTranslationUnitCursor(tu), visit)
    return launches

def kernel_identity(cl, callee_expr):
    """(name, is_global, detail) for the thing being launched."""
    ref = cl.referenced(callee_expr)
    if cl.is_null(ref):
        return None, False, "callee does not resolve to any declaration"
    k = cl.kind(ref)
    name = cl.spelling(ref)
    if k == "OverloadedDeclRef":
        cands = cl.overloads(ref)
        if not cands:
            return name, False, "overload set with no candidates"
        allg = all(any(cl.kind(c) == "attribute(global)" for c in cl.children(d)) for d in cands)
        return name, allg, "template/overload set of %d, all __global__: %s" % (len(cands), allg)
    if k in ("FunctionDecl", "FunctionTemplate"):
        isg = any(cl.kind(c) == "attribute(global)" for c in cl.children(ref))
        return name, isg, k
    return name, False, "callee is a %s, not a function" % k

def report_coverage(live, records, ast_sites, text_sites, refusals):
    """The one line a reader is allowed to quote.  Denominators are the totals
    the parsers found, never the totals that happened to succeed."""
    nexcl = len([r for r in records if r["excluded"]])
    kinds = {}
    for _where, kind, _detail in refusals:
        kinds[kind] = kinds.get(kind, 0) + 1
    print(
        "[%s] coverage: parsed %d of %d launch instantiations (%d excluded by "
        "declaration) at %d of %d textual `<<<` sites; refused %d%s"
        % (
            TAG,
            len(live),
            len(live) + nexcl + len(refusals),
            nexcl,
            len(ast_sites),
            len(text_sites),
            len(refusals),
            (" -- " + ", ".join("%s x%d" % (k, v) for k, v in sorted(kinds.items()))) if kinds else "",
        )
    )

def load_symbols():
    with open(os.path.join(GEN, "launch_symbols.json")) as fh:
        return json.load(fh)

def sanitize(s):
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in s)

def lookup_bound(table, rel, ident):
    for key in ("%s:%s" % (rel, ident), "%s:%s" % (os.path.basename(rel), ident), ident):
        if key in table:
            return key, table[key]
    return None, None

def main(argv):
    check_only = "--check" in argv
    refusals = []
    cfg = cudatu.load_sources(TAG)
    tc = cudatu.resolve_toolchain(cfg, refusals)
    if refusals:
        cudatu.report_refusals(TAG, refusals)
        return 1
    files = cudatu.corpus(cfg, refusals)
    symcfg = load_symbols()
    table = symcfg["symbols"]
    exclusions = symcfg.get("excluded_launches", [])
    excl_hits = {i: 0 for i in range(len(exclusions))}

    args = cudatu.compile_args(tc, cfg)
    cl = CB.Clang(tc["libclang"])

    text_sites = {}
    for rel, path, ent in files:
        with open(path, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        for off, line in cudatext.launch_sites(src):
            text_sites[(rel, off)] = line
        for off, line, txt in cudatext.grid_member_assignments(src):
            refusals.append(
                (
                    "%s:%d" % (rel, line),
                    "grid-member-assignment",
                    "%r assigns a launch-extent member after construction; the "
                    "constructor this extractor reads would be a lie. There were "
                    "zero of these when the extractor was written." % txt.strip(),
                )
            )

    records = []
    ast_sites = set()
    per_file = {}
    for rel, path, ent in files:
        if ent.get("extract") == "none":
            per_file[rel] = {"launches": 0, "skipped_by_declaration": ent.get("why", "")}
            continue
        tu = cudatu.parse_or_refuse(cl, path, rel, args, refusals)
        if tu is None:
            continue
        with open(path, "rb") as fh:
            raw = fh.read()
        an = Analyzer(cl, path, rel, raw, table)
        fns = enclosing_functions(cl, (path, tu))
        launches = collect_launches(cl, path, tu)
        per_file[rel] = {"launches": len(launches)}
        for L in launches:
            kids = cl.children(L)
            cfg_call = kids[1]
            sp = cl.spelling_extent(cfg_call)[0]
            spfile = os.path.relpath(sp[0], REPO) if sp[0].startswith(REPO) else sp[0]
            site = (spfile, sp[3])
            ast_sites.add(site)
            exp = cl.expansion_loc(L)
            where = "%s:%d" % (spfile, sp[1])
            kname, isglobal, kdetail = kernel_identity(cl, kids[0])
            fn_cur = None
            fn_name = "<file scope>"
            for lo, hi, nm, cur in fns:
                if lo <= exp[3] <= hi:
                    fn_cur, fn_name = cur, nm
            excluded = None
            for i, ex in enumerate(exclusions):
                if ex["file"] == spfile and ex["kernel_expr"] == (kname or ""):
                    excl_hits[i] += 1
                    excluded = ex
            if excluded is not None:
                records.append(
                    dict(
                        excluded=True,
                        file=spfile,
                        line=sp[1],
                        offset=sp[3],
                        kernel=kname,
                        why=excluded["why"],
                    )
                )
                continue
            if not isglobal:
                refusals.append(
                    (
                        where,
                        "callee-not-a-kernel",
                        "launch of %r: %s. This extractor cannot state a geometry "
                        "obligation for a launch whose callee is not a __global__ "
                        "function. Add an excluded_launches entry with a reason if "
                        "it is genuinely unanalyzable." % (kname, kdetail),
                    )
                )
                continue
            cfgkids = cl.children(cfg_call)
            if len(cfgkids) < 2:
                refusals.append((where, "no-grid-argument", "launch configuration has %d arguments" % len(cfgkids)))
                continue
            an.notes = []
            an.identdecl = {}
            try:
                axes, chain = an.dim3_axes(cfgkids[1], fn_cur)
            except Refusable as e:
                refusals.append((where, "grid-unresolvable", "launch of %r in %s: %s" % (kname, fn_name, e)))
                continue
            notes = list(an.notes)
            an.notes = []
            guards, noguard, rejected = collect_guards(an, fn_cur, exp[3])
            an.identdecl = {}
            records.append(
                dict(
                    excluded=False,
                    guards=guards,
                    no_guard_reason=noguard,
                    rejected_guards=rejected,
                    file=spfile,
                    line=sp[1],
                    offset=sp[3],
                    kernel=kname,
                    host_fn=fn_name,
                    chain=chain + notes,
                    axes={a: axes[i] for i, a in enumerate(AXES)},
                    macro_expanded=(exp[3] != sp[3]) or (exp[0] != sp[0]),
                    expansion_line=exp[1],
                )
            )
        cl.dispose(tu)

    for i, ex in enumerate(exclusions):
        if excl_hits[i] == 0:
            refusals.append(
                (
                    "gen/launch_symbols.json",
                    "stale-exclusion",
                    "excluded_launches entry (%s, %s) matched nothing. An exclusion "
                    "must not outlive the code it excused." % (ex["file"], ex["kernel_expr"]),
                )
            )

    for site in sorted(set(text_sites) - ast_sites):
        refusals.append(
            (
                "%s:%d" % (site[0], text_sites[site]),
                "textual-site-not-in-ast",
                "the text scan found a `<<<` here that the libclang walk never "
                "reported. The AST walk is dropping launches; do not trust its count.",
            )
        )
    for site in sorted(ast_sites - set(text_sites)):
        refusals.append(
            (
                "%s@%d" % (site[0], site[1]),
                "ast-launch-not-in-text",
                "libclang reported a launch at a byte offset the text scan does not "
                "see as a `<<<` site; the two parsers disagree about the corpus.",
            )
        )

    live = [r for r in records if not r["excluded"]]
    used = {}
    for r in live:
        r["env"] = {}
        env = r["env"]
        for ax in AXES:
            for ident in sorted(r["axes"][ax].idents()):
                key, ent = lookup_bound(table, r["file"], ident)
                if ent is None:
                    refusals.append(
                        (
                            "%s:%d" % (r["file"], r["line"]),
                            "undeclared-symbol",
                            "identifier %r reaches gridDim.%s of %r and is absent from "
                            "gen/launch_symbols.json. Its runtime range is a property of "
                            "the CALLER and cannot be read out of the CUDA: declare it "
                            "with a bound and a witness, or with input_scaled plus the "
                            "bound the caller actually enforces."
                            % (ident, ax, r["kernel"]),
                        )
                    )
                    continue
                env[ident] = (ent["lo"], ent["hi"])
                used.setdefault(key, []).append("%s:%d %s.%s" % (r["file"], r["line"], r["kernel"], ax))

    if refusals:
        cudatu.report_refusals(TAG, refusals)
        report_coverage(live, records, ast_sites, text_sites, refusals)
        return 1

    for r in live:
        env = r["env"]
        for ax in AXES:
            t = r["axes"][ax]
            try:
                lo, hi = t.interval(env)
            except Refusable as e:
                refusals.append(("%s:%d" % (r["file"], r["line"]), "unboundable-axis", str(e)))
                continue
            if lo < 1 or hi > MAX_GRID[ax]:
                refusals.append(
                    (
                        "%s:%d" % (r["file"], r["line"]),
                        "axis-out-of-range",
                        "gridDim.%s of %r is %s, which over the DECLARED ranges of its "
                        "identifiers spans [%d, %d]; CUDA allows [1, %d]. Either the "
                        "launch is unsafe or a declared bound in launch_symbols.json is "
                        "wrong. This is the kv_fp8_paged shape."
                        % (ax, r["kernel"], t.c_like(), lo, hi, MAX_GRID[ax]),
                    )
                )
    if refusals:
        cudatu.report_refusals(TAG, refusals)
        report_coverage(live, records, ast_sites, text_sites, refusals)
        return 1

    text = emit(cfg, tc, files, live, records, table, used, text_sites, ast_sites, symcfg)
    if check_only:
        cur = open(OUT_V).read() if os.path.exists(OUT_V) else ""
        if cur != text:
            print("[%s] STALE: %s does not match a regeneration from the current sources." % (TAG, OUT_V), file=sys.stderr)
            import difflib

            for ln in list(difflib.unified_diff(cur.splitlines(), text.splitlines(), "on-disk", "regenerated", lineterm=""))[:80]:
                print(ln, file=sys.stderr)
            return 1
        print("[%s] --check: GenLaunch.v matches its sources." % TAG)
        return 0

    cudatu.write_atomic(OUT_V, text)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    cudatu.write_atomic(
        OUT_JSON,
        json.dumps(
            {
                "generator": "rocq/gen/extract/launch_geometry.py",
                "arch": cfg["offload_arch"],
                "files": {rel: cudatu.sha256_file(p) for rel, p, _ in files},
                "textual_launch_sites": len(text_sites),
                "ast_launch_instantiations": len(live) + len([r for r in records if r["excluded"]]),
                "extracted": len(live),
                "excluded": [r for r in records if r["excluded"]],
                "refused": [],
                "per_file": per_file,
                "symbol_uses": used,
                "launches": [
                    {
                        "file": r["file"],
                        "line": r["line"],
                        "kernel": r["kernel"],
                        "host_fn": r["host_fn"],
                        "macro_expanded": r["macro_expanded"],
                        "chain": r["chain"],
                        "axes": {a: r["axes"][a].c_like() for a in AXES},
                        "guards": r["guards"],
                        "no_guard_reason": r["no_guard_reason"],
                        "enforced_axes": [
                            {"axis": ax, "ident": i, "guard": g, "why": why}
                            for ax, i, g, why in r.get("enforced", [])
                        ],
                        "unenforced_axes": [
                            {"axis": ax, "ident": i, "unlaunchable_at": v, "why": why}
                            for ax, i, v, _pt, why in r.get("hazard_info", [])
                        ],
                    }
                    for r in live
                ],
            },
            indent=1,
            sort_keys=True,
        )
        + "\n",
    )
    report_coverage(live, records, ast_sites, text_sites, [])
    print(
        "[%s] %d files parsed; coverage declared in gen/out/launch_geometry.json"
        % (TAG, len([f for f in files if f[2].get("extract") != "none"]))
    )
    for r in live:
        for ax, ident, g, _why in r.get("enforced", []):
            print(
                "[%s] ENFORCED BOUND %s:%d %s gridDim.%s depends on %r, bounded "
                "<= %d by the %s (%s)"
                % (TAG, r["file"], r["line"], r["kernel"], ax, ident, g["upper"], g["why"], g["cond"])
            )
        for ax, ident, v, _pt, _why in r.get("hazard_info", []):
            print(
                "[%s] UNENFORCED BOUND %s:%d %s gridDim.%s depends on %r, declared "
                "<= %d by the caller with no assertion; unlaunchable at %d"
                % (TAG, r["file"], r["line"], r["kernel"], ax, ident, r["env"][ident][1], v)
            )
    return 0

def hazard_witness(term, env, ident, axmax, params):
    """Smallest value of `ident` past its declared range that breaks the axis."""
    point = {i: env[i][0] for i in params}
    hi = env[ident][1]
    v = hi
    for _ in range(64):
        v = v * 2 + 1
        point[ident] = v
        if term.eval_at(point) > axmax:
            lo, high = hi, v
            while lo + 1 < high:
                mid = (lo + high) // 2
                point[ident] = mid
                if term.eval_at(point) > axmax:
                    high = mid
                else:
                    lo = mid
            point[ident] = high
            return high, point
        if v > 1 << 40:
            break
    return None, None

def guard_retires(term, env, ident, ax, guard):
    """Does `guard` discharge the hazard `ident` puts on gridDim.<ax>?

    Returns (True, why) only when all four hold, and (False, why-not) on the
    first that does not.  Any doubt is a False: a bound wrongly called enforced
    silently retires a live hazard, which is strictly worse than carrying a
    counterexample for a site that is in fact safe.

      1. a guard was found at all, on the SAME declaration that reaches the
         axis (checked in collect_guards) and dominating this launch;
      2. the constant it enforces is itself within the CUDA limit for the axis;
      3. it does not contradict the declared range;
      4. re-running the axis interval with the identifier's ceiling lowered to
         the guard's constant leaves the whole axis inside [1, limit].
    """
    if guard is None:
        return False, None
    u = guard["upper"]
    if u > MAX_GRID[ax]:
        return False, "the guard still admits %s = %d, above the gridDim.%s limit of %d" % (
            ident, u, ax, MAX_GRID[ax],
        )
    lo, hi = env[ident]
    if u < lo:
        return False, "the guard admits nothing in the declared range [%d, %d]" % (lo, hi)
    env2 = {k: v for k, v in env.items() if not k.startswith("d_")}
    env2[ident] = (lo, min(hi, u))
    try:
        nlo, nhi = term.interval(env2)
    except Refusable as e:
        return False, str(e)
    if nlo < 1 or nhi > MAX_GRID[ax]:
        return False, "even with %s <= %d the axis spans [%d, %d], not [1, %d]" % (
            ident, u, nlo, nhi, MAX_GRID[ax],
        )
    return True, "with %s <= %d, gridDim.%s = %s spans [%d, %d], inside [1, %d]" % (
        ident, u, ax, term.c_like(), nlo, nhi, MAX_GRID[ax],
    )

def classify_axes(live, table):
    """Split every input-scaled (axis, identifier) pair into enforced and not.

    Runs before anything is written, because whether a symbol's `unenforced`
    note in launch_symbols.json still describes the code is a fact about all of
    that symbol's launch sites at once.
    """
    hazards = []
    enforced = []
    for r in live:
        env = r["env"]
        params = sorted(set().union(*[r["axes"][a].idents() for a in AXES]))
        r["hazards"] = []
        r["hazard_info"] = []
        r["enforced"] = []
        for ax in ("y", "z"):
            t = r["axes"][ax]
            for ident in sorted(t.idents()):
                key = lookup_bound(table, r["file"], ident)[0]
                if not table[key].get("input_scaled"):
                    continue
                v, point = hazard_witness(t, env, ident, MAX_GRID[ax], params)
                if v is None:
                    continue
                g = r["guards"].get(ident)
                ok, why = guard_retires(t, env, ident, ax, g)
                if not ok and why is None:
                    why = r.get("rejected_guards", {}).get(ident)
                if ok:
                    r["enforced"].append((ax, ident, g, why))
                    enforced.append((r, ax, ident, g, why, key, params))
                    continue
                r["hazards"].append((ax, ident, v))
                r["hazard_info"].append((ax, ident, v, point, why))
                hazards.append((r, ax, ident, v, point, key, params))
    return hazards, enforced

def emit(cfg, tc, files, live, records, table, used, text_sites, ast_sites, symcfg):
    hazards, enforced = classify_axes(live, table)
    hazard_keys = {key for _r, _ax, _i, _v, _pt, key, _ps in hazards}
    enforced_by = {}
    for r, ax, ident, g, _why, key, _ps in enforced:
        enforced_by.setdefault(key, set()).add("%s, %s" % (g["why"], g["cond"]))
    L = []
    w = L.append
    w("(* AUTO-GENERATED by rocq/gen/extract/launch_geometry.py -- DO NOT EDIT;")
    w("   re-run rocq/gen/run.sh.  `run.sh --check` fails if this file and the")
    w("   CUDA it describes have drifted apart, so nothing here can rot silently.")
    w("")
    w("   LaunchGeometry.v is the rule; this file is the instances.  Every record")
    w("   below was read out of the AST of the .cu named beside it, at the byte")
    w("   offset of its `<<<`, on the run that wrote this file.  No line number")
    w("   here is an input: they are all outputs, regenerated from the sha256s")
    w("   recorded in gen/out/launch_geometry.json.")
    w("")
    w("   MECHANISED: which launches exist, which kernel each calls, and which")
    w("   expression reaches gridDim.{x,y,z} through which resolution chain.")
    w("   DECLARED: the numeric range of every free identifier, in")
    w("   gen/launch_symbols.json.  Whether a count is a whole context or one")
    w("   prefill chunk is a fact about the Rust caller, not about the CUDA, and")
    w("   this generator cannot derive it.  It REFUSES on any identifier reaching")
    w("   an axis that the table does not declare, which is the only reason the")
    w("   declared half is safe.")
    w("")
    w("   ENFORCEMENT is mechanised too, and separately.  A declared range is one")
    w("   thing; a guard in the host launcher that makes the range true is")
    w("   another.  An axis gets NO counterexample theorem below only when a")
    w("   guard was found on the same DECLARATION that reaches the axis, that")
    w("   guard dominates the launch, its constant is inside the CUDA limit, and")
    w("   narrowing the identifier to that constant keeps the whole axis inside")
    w("   [1, limit].  Every accepted guard is named with the file:line and the")
    w("   condition it was read from, so the acceptance can be checked by hand.")
    w("   Guards are read from the AST -- an early return, a then-branch or an")
    w("   else-branch -- never from text, and a host function containing a goto,")
    w("   label or switch gets none recognised, because in one lexical order no")
    w("   longer implies domination.")
    w("")
    w("   arch fixed at %s (matches CUDA_ARCH_LIST=12.0); host pass," % cfg["offload_arch"])
    w("   --cuda-host-only, %s, with extract/shim.h supplying the three `using`" % cfg["cxx_std"])
    w("   declarations nvcc injects implicitly for host code (std::min, std::max,")
    w("   std::isfinite).  Toolchain:")
    for k in sorted(tc):
        w("     %-11s %s" % (k, re.sub(r"/nix/store/[a-z0-9]{32}-", "/nix/store/<hash>-", tc[k])))
    w("")
    w("   corpus: %d declared .cu files, %d textual `<<<` sites, %d AST launch" % (len(files), len(text_sites), len(live)))
    w("   instantiations after macro expansion, %d excluded by declaration," % len([r for r in records if r["excluded"]]))
    w("   0 refused.  Both parsers agreed on all %d sites." % len(ast_sites))
    w("*)")
    w("")
    w("From Stdlib Require Import ZArith Lia.")
    w("From SpeachesPlus Require Import LaunchGeometry.")
    w("Open Scope Z_scope.")
    w("")
    w("Ltac launch_absurd :=")
    w("  unfold launchable, max_grid_x, max_grid_y, max_grid_z;")
    w("  cbn [gx gy gz]; try vm_compute; intuition lia.")
    w("")
    w("Ltac launch_ok :=")
    w("  unfold launchable, max_grid_x, max_grid_y, max_grid_z;")
    w("  cbn [gx gy gz]; repeat split;")
    w("  solve [ lia | vm_compute; discriminate ].")
    w("")
    w("(* Declared identifier ranges, from gen/launch_symbols.json.  Each is an")
    w("   ASSUMPTION about the caller.  An ENFORCED line means the assumption is")
    w("   also checked by the CUDA at the site named, at every launch that puts")
    w("   the identifier on a limited axis; an UNENFORCED line means it is not")
    w("   checked anywhere and is discharged nowhere in this file. *)")
    for key in sorted(used):
        ent = table[key]
        tagtxt = "input_scaled" if ent.get("input_scaled") else "bounded"
        w("(* %s in [%d, %d] -- %s; %s *)" % (key, ent["lo"], ent["hi"], tagtxt, ent["why"]))
        if key in enforced_by and key not in hazard_keys:
            for g in sorted(enforced_by[key]):
                w("(*   ENFORCED in the CUDA by the %s *)" % g)
            if ent.get("unenforced"):
                w("(*   that guard retires the AXIS hazard only; what the code still does")
                w("     not check, per launch_symbols.json: %s *)" % ent["unenforced"])
        elif ent.get("unenforced"):
            if key in enforced_by:
                for g in sorted(enforced_by[key]):
                    w("(*   enforced at SOME sites by the %s, but not at all of them *)" % g)
            w("(*   UNENFORCED: nothing in the code asserts this. %s *)" % ent["unenforced"])
    w("")

    names = {}
    counts = {}
    for r in live:
        base = "l_" + sanitize(r["kernel"])
        n = counts.get(base, 0)
        counts[base] = n + 1
        names[id(r)] = "%s_%d" % (base, n)

    for r in live:
        env = r["env"]
        nm = names[id(r)]
        params = sorted(set().union(*[r["axes"][a].idents() for a in AXES]))
        w("(* %s:%d  %s%s" % (r["file"], r["line"], r["kernel"], "  [macro-expanded at :%d]" % r["expansion_line"] if r["macro_expanded"] else ""))
        w("   host fn %s; grid resolved via %s *)" % (r["host_fn"], " -> ".join(r["chain"])))
        sig = (" (" + " ".join(params) + " : Z)") if params else ""
        w("Definition %s%s : grid :=" % (nm, sig))
        w("  {| gx := %s; gy := %s; gz := %s |}." % tuple(r["axes"][a].rocq() for a in AXES))
        w("")
        w("Theorem %s_launchable :" % nm)
        if params:
            w("  forall %s," % " ".join(params))
            for p in params:
                w("    %d <= %s <= %d ->" % (env[p][0], p, env[p][1]))
            w("    launchable (%s %s)." % (nm, " ".join(params)))
            w("Proof.")
            w("  intros %s." % " ".join(params + ["H%d" % i for i in range(len(params))]))
        else:
            w("  launchable %s." % nm)
            w("Proof.")
        seen = set()
        n = 0
        for ax in AXES:
            for sub in r["axes"][ax].subterms():
                txt = sub.rocq()
                if txt in seen:
                    continue
                seen.add(txt)
                lo, hi = sub.interval(env)
                w("  assert (Hs%d : %d <= %s <= %d)" % (n, lo, txt, hi))
                w("    by (%s)." % PROOF_BY[sub.op])
                n += 1
        w("  unfold launchable, %s, max_grid_x, max_grid_y, max_grid_z." % nm)
        w("  cbn [gx gy gz]. nia.")
        w("Qed.")
        w("")
        for ax, ident, g, why in r["enforced"]:
            w("(* %s is input-scaled and lands on gridDim.%s, but the launcher" % (ident, ax))
            w("   ENFORCES the range, so no counterexample is stated here. Accepted")
            w("   because: the %s bounds the same declaration that reaches the axis," % g["why"])
            w("   it dominates this launch, and %s." % why)
            w("   Guard read from the AST at %s: `%s` *)" % (g["where"], g["cond"]))
            ceiling = {p: env[p][0] for p in params}
            ceiling[ident] = min(env[ident][1], g["upper"])
            vals = {a: r["axes"][a].eval_at(ceiling) for a in AXES}
            if all(1 <= vals[a] <= MAX_GRID[a] for a in AXES):
                w("Theorem %s_%s_launchable_at_the_%s_ceiling_%d :" % (nm, ax, sanitize(ident), ceiling[ident]))
                w("  launchable (%s %s)." % (nm, " ".join(str(ceiling[p]) for p in params)))
                w("Proof. unfold %s. launch_ok. Qed." % nm)
                w("")
        for ax, ident, v, point, why in r["hazard_info"]:
            ent = table[lookup_bound(table, r["file"], ident)[0]]
            if why:
                last = "A guard on %s WAS found and rejected: %s." % (ident, why)
            elif r["no_guard_reason"]:
                last = "No guard was looked for here: %s." % r["no_guard_reason"]
            else:
                last = "No guard dominating this launch bounds %s in %s." % (ident, r["host_fn"])
            w("(* %s is input-scaled and lands on gridDim.%s. Inside the declared" % (ident, ax))
            w("   range this launches; at %s = %d it does not, and NOTHING IN THE CODE" % (ident, v))
            w("   ENFORCES THE RANGE. %s" % ent.get("unenforced", ""))
            w("   %s *)" % last)
            w("Theorem %s_%s_unlaunchable_at_%s_%d :" % (nm, ax, sanitize(ident), v))
            w("  ~ launchable (%s %s)." % (nm, " ".join(str(point[p]) for p in params)))
            w("Proof. unfold %s. launch_absurd. Qed." % nm)
            w("")

    w("(* Coverage, as data rather than as a claim in a commit message. *)")
    w("Definition textual_launch_sites : Z := %d." % len(text_sites))
    w("Definition ast_launch_sites : Z := %d." % len(ast_sites))
    w("Definition launches_extracted : Z := %d." % len(live))
    w("Definition launches_excluded : Z := %d." % len([r for r in records if r["excluded"]]))
    w("Definition launches_refused : Z := 0.")
    w("")
    w("Theorem both_parsers_saw_the_same_corpus :")
    w("  textual_launch_sites = ast_launch_sites.")
    w("Proof. reflexivity. Qed.")
    w("")
    w("Theorem nothing_was_skipped :")
    w("  launches_refused = 0 /\\ launches_extracted = %d." % len(live))
    w("Proof. split; reflexivity. Qed.")
    w("")
    if hazards:
        w("(* Launches whose safety rests on an UNENFORCED caller bound.  Each has an")
        w("   _unlaunchable_at_ theorem above: the arithmetic that kills it is proved,")
        w("   not asserted. *)")
        for r, ax, ident, v, _pt, _key, _ps in hazards:
            w("(*   %s:%d %s gridDim.%s = %s, dies at %s = %d *)" % (r["file"], r["line"], r["kernel"], ax, r["axes"][ax].c_like(), ident, v))
    w("Definition unenforced_bound_launches : Z := %d." % len(hazards))
    w("")
    if enforced:
        w("(* Launches where the same hazard exists in shape but the launcher refuses")
        w("   the out-of-range call itself.  These carry no counterexample: the guard")
        w("   named beside each was read out of the AST, dominates the launch, and")
        w("   bounds the very declaration that reaches the axis. *)")
        for r, ax, ident, g, _why, _key, _ps in enforced:
            w("(*   %s:%d %s gridDim.%s = %s, %s <= %d by the %s *)" % (r["file"], r["line"], r["kernel"], ax, r["axes"][ax].c_like(), ident, g["upper"], g["why"]))
    w("Definition enforced_bound_launches : Z := %d." % len(enforced))
    w("")
    w("(* Every input-scaled axis that could leave the legal grid is in exactly one")
    w("   of the two buckets: no axis is both enforced and counterexampled, and")
    w("   none is silently in neither. *)")
    w("Definition input_scaled_hazard_axes : Z := %d." % (len(hazards) + len(enforced)))
    w("")
    w("Theorem every_hazard_axis_is_classified :")
    w("  unenforced_bound_launches + enforced_bound_launches = input_scaled_hazard_axes.")
    w("Proof. reflexivity. Qed.")
    w("")
    return "\n".join(L)

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
