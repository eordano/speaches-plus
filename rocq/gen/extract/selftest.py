#!/usr/bin/env python3
"""Prove that launch_geometry.py's refusals actually fire.

An extractor whose refusal paths are never exercised is the same artifact as a
test suite that asserts nothing: it reports a coverage number nobody can
contradict.  Each case below injects one defect, runs the extractor in-process
with its writer stubbed out, and requires BOTH a non-zero exit and the named
refusal kind on stderr.  A case that passes without producing its refusal is a
failure of this file, not a success of the extractor.

Every injection is in-memory.  Nothing under rust/ is touched and nothing is
written to disk -- cudatu.write_atomic is replaced for the duration.
"""

import contextlib
import copy
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cudatext  # noqa: E402
import cudatu  # noqa: E402
import launch_geometry as LG  # noqa: E402

TAG = "selftest"
CU = "rust/crates/nv-kernels/cuda/"


class Wrote(Exception):
    pass


def _no_write(path, text):
    raise Wrote("extractor wrote %s during a refusal case" % path)


def run_case(name, expect_kind, patch):
    """patch() installs the defect and returns a list of undo callables."""
    undo = patch()
    real_write = cudatu.write_atomic
    cudatu.write_atomic = _no_write
    err = io.StringIO()
    out = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
            rc = LG.main([])
    except Wrote as e:
        rc = 0
        err.write(str(e))
    finally:
        cudatu.write_atomic = real_write
        for fn in reversed(undo):
            fn()
    text = err.getvalue()
    ok = rc != 0 and ("REFUSED %s" % expect_kind) in text
    print("[%s] %-26s %s" % (TAG, name, "PASS" if ok else "FAIL"))
    if not ok:
        print("[%s]   expected rc!=0 and 'REFUSED %s'; got rc=%s" % (TAG, expect_kind, rc))
        for line in text.strip().splitlines()[:6]:
            print("[%s]   stderr: %s" % (TAG, line))
    return ok


def patch_symbols(mutate):
    def go():
        base = LG.load_symbols()
        d = copy.deepcopy(base)
        mutate(d)
        LG.load_symbols = lambda: d
        real = LG.load_symbols
        return [lambda: setattr(LG, "load_symbols", _orig_load_symbols)]

    return go


def patch_sources(mutate):
    def go():
        def loader(tag):
            cfg = _orig_load_sources(tag)
            mutate(cfg)
            return cfg

        cudatu.load_sources = loader
        return [lambda: setattr(cudatu, "load_sources", _orig_load_sources)]

    return go


_orig_load_symbols = LG.load_symbols
_orig_load_sources = cudatu.load_sources
_orig_sites = cudatext.launch_sites
_orig_members = cudatext.grid_member_assignments

# extract/guard_fixture.cu, one host function per shape.  None means "must NOT
# be recognised": for enforcement, a miss is a carried hazard and a false hit is
# a retired one, so the second half of this table is the half that matters.
GUARD_EXPECT = {
    "case_early_return_gt": 65535,
    "case_early_return_ge": 65535,
    "case_constant_on_the_left": 65535,
    "case_named_constant": 65535,
    "case_or_chain": 65535,
    "case_early_return_in_a_block": 65535,
    "case_guarded_then_branch": 65535,
    "case_guarded_then_branch_and_chain": 65535,
    "case_guarded_else_branch": 65535,
    "case_constant_on_the_left_then_branch": 65535,
    "case_nested_dominating_blocks": 65535,
    "case_and_chain_early_return": None,
    "case_guard_after_the_launch": None,
    "case_guard_on_another_identifier": None,
    "case_guard_that_does_not_return": None,
    "case_guard_with_an_else": None,
    "case_guarded_then_reassigned": None,
    "case_address_of_the_guarded_value": None,
    "case_guard_inside_a_conditional_block": None,
    # A guard before a switch dominates the launch after it; a guard inside one
    # case does not dominate a launch in another.
    "case_guard_before_a_switch": 65535,
    "case_guard_inside_a_switch_case": None,
    "case_function_contains_a_label": None,
    "case_guard_is_not_a_constant": None,
    "case_lower_bound_only": None,
}


def guard_fixture():
    """Read every guard shape out of extract/guard_fixture.cu and check it.

    One parse, not one per case.  The fixture is outside the corpus, so this
    asserts about the guard reader without any dependence on what the CUDA
    under rust/ happens to look like today.
    """
    path = os.path.join(HERE, "guard_fixture.cu")
    rel = os.path.relpath(path, cudatu.REPO)
    refusals = []
    cfg = cudatu.load_sources(TAG)
    tc = cudatu.resolve_toolchain(cfg, refusals)
    if refusals:
        print("[%s] guard fixture: toolchain unavailable: %s" % (TAG, refusals[0][2]))
        return False
    cl = LG.CB.Clang(tc["libclang"])
    tu = cudatu.parse_or_refuse(cl, path, rel, cudatu.compile_args(tc, cfg), refusals)
    if tu is None:
        print("[%s] guard fixture: %s" % (TAG, refusals[0][2]))
        return False
    an = LG.Analyzer(cl, path, rel, open(path, "rb").read(), {})
    fns = LG.enclosing_functions(cl, (path, tu))
    got = {}
    for L in LG.collect_launches(cl, path, tu):
        exp = cl.expansion_loc(L)
        fn_cur, fn_name = None, "<file scope>"
        for lo, hi, nm, cur in fns:
            if lo <= exp[3] <= hi:
                fn_cur, fn_name = cur, nm
        cfg_call = cl.children(L)[1]
        an.identdecl = {}
        an.notes = []
        try:
            an.dim3_axes(cl.children(cfg_call)[1], fn_cur)
        except LG.Refusable as e:
            got[fn_name] = "refused: %s" % e
            continue
        guards, _why, _rejected = LG.collect_guards(an, fn_cur, exp[3])
        got[fn_name] = guards.get("m", {}).get("upper")
    cl.dispose(tu)
    ok = True
    for name in sorted(set(GUARD_EXPECT) | set(got)):
        want = GUARD_EXPECT.get(name, "<case not declared in GUARD_EXPECT>")
        have = got.get(name, "<no launch found in the fixture>")
        if want != have:
            ok = False
            print("[%s] guard fixture %-40s want %r, got %r" % (TAG, name, want, have))
    print("[%s] %-26s %s" % (TAG, "guard shapes", "PASS" if ok else "FAIL"))
    return ok


def main():
    results = [guard_fixture()]

    def drop_symbol(d):
        del d["symbols"][CU + "kv_fp8_paged.cu:n_tokens"]

    results.append(
        run_case(
            "undeclared symbol",
            "undeclared-symbol",
            patch_symbols(drop_symbol),
        )
    )

    def widen_bound(d):
        # n_kv is what kv_fp8_paged.cu puts on gridDim.y today.  This case used
        # to widen n_tokens, and went red when n_tokens moved to gridDim.x --
        # where 262144 is a legal extent, so no refusal was owed. The case was
        # right to fail; it is repointed here at the axis that is still limited.
        d["symbols"][CU + "kv_fp8_paged.cu:n_kv"]["hi"] = 262144

    results.append(
        run_case(
            "axis out of range",
            "axis-out-of-range",
            patch_symbols(widen_bound),
        )
    )

    def stale_exclusion(d):
        d["excluded_launches"][0]["kernel_expr"] = "a_kernel_that_no_longer_exists"

    results.append(
        run_case(
            "exclusion outlived code",
            "stale-exclusion",
            patch_symbols(stale_exclusion),
        )
    )

    def undeclare_file(cfg):
        cfg["files"] = [f for f in cfg["files"] if f["path"] != CU + "kv_fp8_paged.cu"]

    results.append(
        run_case(
            "corpus grew, undeclared",
            "undeclared-file",
            patch_sources(undeclare_file),
        )
    )

    def vanish_file(cfg):
        cfg["files"] = cfg["files"] + [{"path": CU + "a_kernel_that_was_deleted.cu", "extract": "launches"}]

    results.append(
        run_case(
            "corpus shrank, declared",
            "declared-file-missing",
            patch_sources(vanish_file),
        )
    )

    def break_toolchain(cfg):
        cfg["toolchain"]["cutlass"]["path"] = "/nonexistent/cutlass"

    results.append(
        run_case(
            "include root missing",
            "toolchain-missing",
            patch_sources(break_toolchain),
        )
    )

    def move_pin(cfg):
        cfg["toolchain"]["cuda"]["expect_substring"] = "cuda-merged-99.9"

    results.append(
        run_case(
            "toolchain pin moved",
            "toolchain-unexpected-version",
            patch_sources(move_pin),
        )
    )

    def hide_one_site():
        def fewer(src):
            got = _orig_sites(src)
            return got[1:] if len(got) > 1 else got

        cudatext.launch_sites = fewer
        return [lambda: setattr(cudatext, "launch_sites", _orig_sites)]

    results.append(
        run_case(
            "parsers disagree (text<ast)",
            "ast-launch-not-in-text",
            hide_one_site,
        )
    )

    def invent_a_site():
        def more(src):
            return _orig_sites(src) + [(len(src) + 7, 1)]

        cudatext.launch_sites = more
        return [lambda: setattr(cudatext, "launch_sites", _orig_sites)]

    results.append(
        run_case(
            "parsers disagree (ast<text)",
            "textual-site-not-in-ast",
            invent_a_site,
        )
    )

    def invent_member_assign():
        def some(src):
            return _orig_members(src) + [(0, 1, "grid.y =")]

        cudatext.grid_member_assignments = some
        return [lambda: setattr(cudatext, "grid_member_assignments", _orig_members)]

    results.append(
        run_case(
            "grid.y assigned after ctor",
            "grid-member-assignment",
            invent_member_assign,
        )
    )

    npass = sum(1 for r in results if r)
    print("[%s] %d/%d refusal paths fire" % (TAG, npass, len(results)))
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
