#!/usr/bin/env python3
"""Shared CUDA translation-unit front end for the rocq/gen extractors.

Owns three things that must not diverge between extractors:

  * the toolchain declaration in gen/sources.json (nix store paths, with env
    overrides) and the refusal when any of it is absent -- a missing include
    root must never turn into a skipped file;
  * the corpus declaration, checked in BOTH directions (declared-but-missing
    and present-but-undeclared);
  * the compile arguments, including the three-`using` shim that stands in for
    the min/max/isfinite injection nvcc does implicitly for host code.
"""

import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.dirname(HERE)
ROCQ = os.path.dirname(GEN)
REPO = os.path.dirname(ROCQ)

sys.path.insert(0, HERE)
import clangbind as CB  # noqa: E402


class Refusal(Exception):
    """A construct the extractor will not guess at.  Always names file:line."""

    def __init__(self, where, kind, detail):
        super().__init__("%s: %s: %s" % (where, kind, detail))
        self.where = where
        self.kind = kind
        self.detail = detail


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sources(tag):
    with open(os.path.join(GEN, "sources.json")) as fh:
        return json.load(fh)


def resolve_toolchain(cfg, refusals):
    tc = {}
    for name, ent in sorted(cfg["toolchain"].items()):
        path = os.environ.get(ent["env"], "") or ent["path"]
        origin = "env %s" % ent["env"] if os.environ.get(ent["env"]) else "sources.json"
        if not os.path.exists(path):
            refusals.append(
                (
                    "gen/sources.json",
                    "toolchain-missing",
                    "%s resolved to %s (%s) which does not exist; "
                    "set $%s or repin sources.json. Refusing rather than "
                    "parsing without it." % (name, path, origin, ent["env"]),
                )
            )
            continue
        want = ent.get("expect_substring")
        if want and want not in path:
            refusals.append(
                (
                    "gen/sources.json",
                    "toolchain-unexpected-version",
                    "%s resolved to %s (%s) which does not contain %r; the pin "
                    "moved. Re-verify the parse and update sources.json."
                    % (name, path, origin, want),
                )
            )
            continue
        tc[name] = path
    return tc


def compile_args(tc, cfg):
    a = [
        "-x",
        "cuda",
        "--cuda-host-only",
        "--cuda-path=" + tc["cuda"],
        "--offload-arch=" + cfg["offload_arch"],
        "-std=" + cfg["cxx_std"],
        "-w",
    ]
    for inc in cfg["include_roots"]:
        a.append("-I" + inc.replace("$REPO", REPO).replace("$CUTLASS", tc["cutlass"]).replace("$FLASHINFER", tc["flashinfer"]))
    a += ["-include", os.path.join(HERE, "shim.h")]
    for inc in cfg["system_include_roots"]:
        a += ["-isystem", inc.replace("$GCC", tc["gcc"]).replace("$GLIBC", tc["glibc"])]
    return a


def device_args(tc, cfg):
    a = compile_args(tc, cfg)
    return ["--cuda-device-only" if x == "--cuda-host-only" else x for x in a]


def corpus(cfg, refusals):
    """Declared file list, checked against what is actually on disk."""
    root = os.path.join(REPO, cfg["corpus_root"])
    declared = {}
    for ent in cfg["files"]:
        declared[ent["path"]] = ent
    on_disk = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d in cfg["corpus_subdirs"] or dirpath == root]
        for fn in filenames:
            if fn.endswith(".cu"):
                on_disk.add(
                    os.path.relpath(os.path.join(dirpath, fn), REPO)
                )
    for rel in sorted(set(declared) - on_disk):
        refusals.append(
            (
                rel,
                "declared-file-missing",
                "gen/sources.json declares this file but it is not on disk. "
                "A census that cannot notice its own corpus vanishing is the "
                "failure this extractor exists to prevent.",
            )
        )
    for rel in sorted(on_disk - set(declared)):
        refusals.append(
            (
                rel,
                "undeclared-file",
                "a .cu file under %s that gen/sources.json does not declare. "
                "Add it, or declare it with \"extract\": \"none\" and a reason."
                % cfg["corpus_root"],
            )
        )
    return [
        (rel, os.path.join(REPO, rel), declared[rel])
        for rel in sorted(set(declared) & on_disk)
    ]


def parse_or_refuse(cl, path, rel, args, refusals):
    tu = cl.parse(path, args)
    if not tu:
        refusals.append((rel, "parse-null", "libclang returned no translation unit"))
        return None
    errs = cl.errors(tu)
    if errs:
        cl.dispose(tu)
        refusals.append(
            (
                rel,
                "parse-error",
                "%d clang error(s), first: %s" % (len(errs), errs[0].strip().replace("\n", " ")[:300]),
            )
        )
        return None
    return tu


def git_rev():
    try:
        out = subprocess.run(
            ["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def write_atomic(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)


def report_refusals(tag, refusals):
    for where, kind, detail in refusals:
        print("[%s] REFUSED %s at %s: %s" % (tag, kind, where, detail), file=sys.stderr)
    print(
        "[%s] REFUSED: %d construct(s) this extractor will not guess at. "
        "Nothing was written." % (tag, len(refusals)),
        file=sys.stderr,
    )
