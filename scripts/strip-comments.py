#!/usr/bin/env python3
"""Strip comments from Go, Rust, and Python source files while preserving:

- Cgo declaration blocks (any /* ... */ containing #cgo, #include, or extern)
- Go compiler directives and cgo exports: //go:build gates which file
  compiles (stripping it merges both build-tag arms into one package) and
  //export is how cgo finds callbacks
- All string and char/rune literals (regular, raw, byte, byte-raw)
- Rust lifetime tokens ('a) vs char literals ('x')
- Nested Rust block comments
- Rust `///` / `//!` blocks that contain a fence cargo compiles or runs as a
  doctest: a doctest is a TEST, and a formatter that deletes tests is a
  coverage loss no test count reports. Prose-only `///` and `//!` are still
  stripped -- a doc comment is not a place to keep rationale, because this
  script is entitled to delete it. Fences tagged `text` or `ignore` are prose:
  cargo neither compiles nor runs them. Block-form `/** */` and `/*! */` doc
  comments get no such reprieve; write the line form.
- Python shebang on line 1 and PEP 723 inline-script metadata
  (`# /// script` ... `# ///`)
- Python docstrings (they are string literals, not comments)
- Python tool directives (# noqa, # type:, # pragma, # ruff:, # mypy:,
  # isort:, # fmt:) -- they change ruff/mypy output, not readers; stripping
  a # noqa flips the tree from lint-clean to lint-dirty
- Blank lines and trailing whitespace inside multi-line string literals:
  blank-run collapse skips lines the strippers report as literal interior,
  so an embedded shader/script/docstring is never rewritten

Usage:
    strip-comments [--check] [--quiet] [PATH ...]

PATHs default to the current directory. With --check, no files are written and
the exit code is 1 if any file would be modified.
"""

from __future__ import annotations

import argparse
import bisect
import io
import re
import sys
import tokenize
from pathlib import Path

SKIP_DIRS = {".git", "target", "vendor", "node_modules", ".direnv", ".venv", "__pycache__"}

GO_DIRECTIVE_RE = re.compile(r"^//(go:|export[ \t])")
PY_DIRECTIVE_RE = re.compile(r"^#\s*(noqa|type:|pragma|mypy:|ruff|isort:|fmt:|yapf)")

RUST_DOC_LINE_RE = re.compile(r"^[ \t]*//[/!]")
RUST_DOC_FENCE_RE = re.compile(r"^[ \t]*//[/!][ \t]*(?:```|~~~)[ \t]*(.*)$")

FENCE_TAGS_CARGO_BUILDS_OR_RUNS = {
    "",
    "rust",
    "should_panic",
    "no_run",
    "compile_fail",
    "allow_fail",
    "edition2015",
    "edition2018",
    "edition2021",
    "edition2024",
}

def _fence_is_a_doctest(info: str) -> bool:
    tags = [t.strip() for t in info.strip().replace(" ", ",").split(",")]
    return all(t in FENCE_TAGS_CARGO_BUILDS_OR_RUNS for t in tags)

def doc_block_lines_carrying_a_doctest(src: str) -> set[int]:
    """0-based line numbers of every contiguous `///`/`//!` run whose fences
    cargo builds or runs. Deleting one of those deletes a test."""
    lines = src.split("\n")
    keep: set[int] = set()
    i = 0
    while i < len(lines):
        if not RUST_DOC_LINE_RE.match(lines[i]):
            i += 1
            continue
        start = i
        while i < len(lines) and RUST_DOC_LINE_RE.match(lines[i]):
            i += 1
        block = range(start, i)
        inside = False
        for row in block:
            m = RUST_DOC_FENCE_RE.match(lines[row])
            if not m:
                continue
            if inside:
                inside = False
                continue
            inside = True
            if _fence_is_a_doctest(m.group(1)):
                keep.update(block)
                break
    return keep

def _line_starts(src: str) -> list[int]:
    starts = [0]
    for idx, ch in enumerate(src):
        if ch == "\n":
            starts.append(idx + 1)
    return starts

def _consume_string(src: str, i: int, quote: str) -> int:
    n = len(src)
    j = i + 1
    while j < n:
        ch = src[j]
        if ch == "\\" and j + 1 < n:
            j += 2
            continue
        if ch == quote:
            return j + 1
        if ch == "\n" and quote == "'":
            return j
        j += 1
    return n

def _is_directive_block(body: str) -> bool:
    return "#cgo" in body or "#include" in body or "extern " in body

def _ident_char(c: str) -> bool:
    return bool(c) and (c.isalnum() or c == "_")

class _Emitter:
    """Accumulates output text and the 0-based output line numbers that fall
    inside emitted literals, so blank-run collapse can leave them alone."""

    def __init__(self) -> None:
        self.out: list[str] = []
        self.line = 0
        self.protected: set[int] = set()

    def emit(self, s: str, literal: bool = False) -> None:
        nl = s.count("\n")
        if literal and nl:
            self.protected.update(range(self.line, self.line + nl + 1))
        self.line += nl
        self.out.append(s)

    def pop_trailing_ws(self) -> None:
        while self.out and self.out[-1] in (" ", "\t"):
            self.out.pop()

    def text(self) -> str:
        return "".join(self.out)

def _close_block_comment_e(em: _Emitter, src: str, i: int) -> None:
    """Drop the whitespace that preceded a block comment without gluing tokens.

    `a /* c */b` must not become `ab`: an inline block comment is a token
    separator, so re-emit one space when both neighbours are identifier
    characters.
    """
    em.pop_trailing_ws()
    if em.out and _ident_char(em.out[-1][-1:]) and i < len(src) and _ident_char(src[i]):
        em.emit(" ")

def strip_go(src: str) -> tuple[str, set[int]]:
    em = _Emitter()
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        if c == '"':
            end = _consume_string(src, i, '"')
            em.emit(src[i:end], literal=True)
            i = end
            continue

        if c == "'":
            end = _consume_string(src, i, "'")
            em.emit(src[i:end], literal=True)
            i = end
            continue

        if c == "`":
            end = src.find("`", i + 1)
            if end == -1:
                em.emit(src[i:], literal=True)
                return em.text(), em.protected
            em.emit(src[i:end + 1], literal=True)
            i = end + 1
            continue

        if c == "/" and nxt == "/":
            j = i
            while j < n and src[j] != "\n":
                j += 1
            if GO_DIRECTIVE_RE.match(src[i:j]):
                em.emit(src[i:j])
                i = j
                continue
            i = j
            em.pop_trailing_ws()
            continue

        if c == "/" and nxt == "*":
            j = i + 2
            while j + 1 < n and not (src[j] == "*" and src[j + 1] == "/"):
                j += 1
            end = min(j + 2, n)
            if _is_directive_block(src[i + 2:j]):
                em.emit(src[i:end], literal=True)
                i = end
                continue
            i = end
            _close_block_comment_e(em, src, i)
            continue

        em.emit(c)
        i += 1

    return em.text(), em.protected

def strip_rust(src: str) -> tuple[str, set[int]]:
    em = _Emitter()
    doctest_lines = doc_block_lines_carrying_a_doctest(src)
    starts = _line_starts(src) if doctest_lines else []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""

        raw_at = -1
        if c == "r" and (nxt == '"' or nxt == "#"):
            raw_at = i
        elif c == "b" and nxt == "r" and i + 2 < n and (src[i + 2] == '"' or src[i + 2] == "#"):
            raw_at = i + 1
        if raw_at >= 0:
            j = raw_at + 1
            hashes = 0
            while j < n and src[j] == "#":
                hashes += 1
                j += 1
            if j < n and src[j] == '"':
                terminator = '"' + "#" * hashes
                end = src.find(terminator, j + 1)
                if end == -1:
                    em.emit(src[i:], literal=True)
                    return em.text(), em.protected
                end += len(terminator)
                em.emit(src[i:end], literal=True)
                i = end
                continue

        if c == "b" and nxt == '"':
            end = _consume_string(src, i + 1, '"')
            em.emit(src[i:end], literal=True)
            i = end
            continue

        if c == "b" and nxt == "'":
            end = _consume_string(src, i + 1, "'")
            em.emit(src[i:end], literal=True)
            i = end
            continue

        if c == '"':
            end = _consume_string(src, i, '"')
            em.emit(src[i:end], literal=True)
            i = end
            continue

        if c == "'":
            if i + 1 < n:
                a = src[i + 1]
                if a == "\\":
                    j = i + 2
                    if j < n and src[j] == "u" and j + 1 < n and src[j + 1] == "{":
                        close = src.find("}", j + 2)
                        if close == -1:
                            em.emit(src[i:], literal=True)
                            return em.text(), em.protected
                        j = close + 1
                    else:
                        j = j + 1 if j < n else j
                    if j < n and src[j] == "'":
                        em.emit(src[i:j + 1], literal=True)
                        i = j + 1
                        continue
                    em.emit(c)
                    i += 1
                    continue
                if i + 2 < n and src[i + 2] == "'":
                    em.emit(src[i:i + 3], literal=True)
                    i += 3
                    continue
                em.emit(c)
                i += 1
                continue
            em.emit(c)
            i += 1
            continue

        if c == "/" and nxt == "/":
            if doctest_lines and (bisect.bisect_right(starts, i) - 1) in doctest_lines:
                j = i
                while j < n and src[j] != "\n":
                    j += 1
                em.emit(src[i:j])
                i = j
                continue
            while i < n and src[i] != "\n":
                i += 1
            em.pop_trailing_ws()
            continue

        if c == "/" and nxt == "*":
            depth = 1
            i += 2
            while i < n and depth > 0:
                if i + 1 < n and src[i] == "/" and src[i + 1] == "*":
                    depth += 1
                    i += 2
                elif i + 1 < n and src[i] == "*" and src[i + 1] == "/":
                    depth -= 1
                    i += 2
                else:
                    i += 1
            _close_block_comment_e(em, src, i)
            continue

        em.emit(c)
        i += 1

    return em.text(), em.protected

def _find_pep723_block(src: str) -> tuple[int, int] | None:
    lines = src.split("\n")
    start = None
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped == "# /// script":
            start = i
            break
        if stripped and not stripped.startswith("#"):
            return None
    if start is None:
        return None
    for j in range(start + 1, len(lines)):
        if lines[j].strip() == "# ///":
            return (start, j)
    return None

def _python_string_lines(src: str) -> set[int]:
    protected: set[int] = set()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenizeError, SyntaxError, IndentationError):
        return protected
    for tok in tokens:
        if tok.type == tokenize.STRING and tok.end[0] > tok.start[0]:
            protected.update(range(tok.start[0] - 1, tok.end[0]))
    return protected

def strip_python(src: str) -> tuple[str, set[int]]:
    lines = src.split("\n")
    shebang_end = 0
    if lines and lines[0].startswith("#!"):
        shebang_end = 1

    pep = _find_pep723_block(src)
    if pep is not None:
        keep_start, keep_end = pep
    else:
        keep_start = keep_end = -1

    keep_lines: set[int] = set()
    for i in range(shebang_end):
        keep_lines.add(i)
    if pep is not None:
        for i in range(keep_start, keep_end + 1):
            keep_lines.add(i)

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenizeError, SyntaxError, IndentationError):
        return src, _python_string_lines(src)

    comment_spans: list[tuple[int, int, int, int]] = []
    for tok in tokens:
        if tok.type == tokenize.COMMENT:
            srow, scol = tok.start
            erow, ecol = tok.end
            if (srow - 1) in keep_lines:
                continue
            if PY_DIRECTIVE_RE.match(tok.string):
                continue
            comment_spans.append((srow, scol, erow, ecol))

    if not comment_spans:
        return src, _python_string_lines(src)

    new_lines = list(lines)
    by_line: dict[int, list[tuple[int, int]]] = {}
    for srow, scol, erow, ecol in comment_spans:
        by_line.setdefault(srow, []).append((scol, ecol))

    drop_lines: set[int] = set()
    for row, spans in by_line.items():
        idx = row - 1
        line = new_lines[idx]
        spans.sort(reverse=True)
        for scol, ecol in spans:
            line = line[:scol] + line[ecol:]
        stripped_line = line.rstrip()
        if stripped_line == "":
            drop_lines.add(idx)
        else:
            new_lines[idx] = stripped_line

    result_lines = [ln for i, ln in enumerate(new_lines) if i not in drop_lines]
    new_src = "\n".join(result_lines) + ("\n" if src.endswith("\n") else "")
    return new_src, _python_string_lines(new_src)

def collapse_blank_lines(src: str, protected: set[int]) -> str:
    out: list[tuple[str, bool]] = []
    blank_run = 0
    for i, raw in enumerate(src.split("\n")):
        if i in protected:
            out.append((raw, True))
            blank_run = 0
            continue
        ln = raw.rstrip()
        if ln == "":
            blank_run += 1
            if blank_run <= 1:
                out.append((ln, False))
        else:
            blank_run = 0
            out.append((ln, False))
    while out and out[-1] == ("", False):
        out.pop()
    return "\n".join(ln for ln, _ in out) + "\n"

def process_file(path: Path, *, check: bool) -> tuple[bool, bool]:
    src = path.read_text()
    if path.suffix == ".go":
        new, protected = strip_go(src)
    elif path.suffix == ".rs":
        new, protected = strip_rust(src)
    elif path.suffix == ".py":
        new, protected = strip_python(src)
    else:
        return False, False
    new = collapse_blank_lines(new, protected)
    if new == src:
        return True, False
    if not check:
        path.write_text(new)
    return True, True

def _strip_rust_text(src: str) -> str:
    out, protected = strip_rust(src)
    return collapse_blank_lines(out, protected)

def self_test() -> int:
    runnable = "/// d\n///\n/// ```\n/// assert_eq!(1, 1);\n/// ```\npub fn a() {}\n"
    assert _strip_rust_text(runnable) == runnable, (
        "a doctest is a test cargo runs; a formatter that deletes it removes "
        "coverage that no test count reports missing"
    )
    for tag in ("no_run", "compile_fail", "should_panic"):
        src = f"/// ```{tag}\n/// let _ = 1;\n/// ```\npub fn a() {{}}\n"
        assert _strip_rust_text(src) == src, f"cargo builds ```{tag}, so it is a test"
    for tag in ("text", "ignore"):
        src = f"/// ```{tag}\n/// prose\n/// ```\npub fn a() {{}}\n"
        out = _strip_rust_text(src)
        assert "///" not in out and "pub fn a() {}" in out, (
            f"cargo neither builds nor runs ```{tag}, so it is prose and prose is not durable"
        )
    for src in ("/// why\npub fn a() {}\n", "//! why\npub fn a() {}\n"):
        out = _strip_rust_text(src)
        assert "why" not in out and "pub fn a() {}" in out, (
            "prose-only doc comments are stripped: rationale belongs in a name, "
            "an assertion message, or docs/"
        )
    assert _strip_rust_text("/// why\npub fn a() {}\n").startswith("\n"), (
        "a comment stripped from line 1 leaves the blank line behind: "
        "collapse_blank_lines trims trailing blank runs, never leading ones"
    )
    mixed = "/// ```text\n/// prose\n/// ```\npub fn a() {}\n\n/// ```\n/// assert!(true);\n/// ```\npub fn b() {}\n"
    out = _strip_rust_text(mixed)
    assert "prose" not in out and "assert!(true)" in out, (
        "a closing fence carries an empty info string; classifying it as an opener "
        "would spare every ```text block in the tree"
    )
    once = _strip_rust_text(mixed)
    assert _strip_rust_text(once) == once, "stripping is idempotent"
    print("strip-comments self-test: ok")
    return 0

def iter_files(root: Path):
    if root.is_file():
        yield root
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix in (".go", ".rs", ".py"):
            yield p

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="strip-comments",
        description="Strip comments from Go and Rust source files (preserves cgo blocks).",
    )
    parser.add_argument("paths", nargs="*", default=["."], help="files or directories (default: .)")
    parser.add_argument("--check", action="store_true", help="exit 1 if any file would change; do not write")
    parser.add_argument("--quiet", "-q", action="store_true", help="suppress per-file output")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="assert which doc comments survive; writes nothing",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    total = 0
    changed = 0
    for raw in args.paths:
        root = Path(raw)
        if not root.exists():
            print(f"strip-comments: {raw}: no such path", file=sys.stderr)
            return 2
        for path in iter_files(root):
            total += 1
            considered, modified = process_file(path, check=args.check)
            if not considered:
                continue
            if modified:
                changed += 1
                if not args.quiet:
                    verb = "would strip" if args.check else "stripped"
                    print(f"{verb} {path}")

    if not args.quiet:
        suffix = "would be modified" if args.check else "modified"
        print(f"processed {total} files, {changed} {suffix}", file=sys.stderr)

    if args.check and changed:
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
