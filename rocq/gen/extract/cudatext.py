#!/usr/bin/env python3
"""Independent textual scan of CUDA sources: enumerate every `<<<` launch site.

This exists ONLY as a cross-check on the AST walk, and it is deliberately
dumb: blank comments and string/char literals, then record the byte offset of
every remaining `<<<`.  It shares no code and no assumptions with libclang.

The reason it exists: the AST route silently dropped 34% of this corpus
(93 of 275 launches) from one wrong main-file predicate, and the only symptom
was a smaller number.  A count nobody can contradict is not evidence.  Any
textual site with zero AST launches mapped to it, or any AST launch with no
textual site, is a refusal in launch_geometry.py.
"""

import re


def blank_out(src):
    """Replace comment and literal bytes with spaces, preserving offsets and newlines."""
    out = list(src)
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            j = i
            while j < n:
                if src[j] == "\n":
                    if j > 0 and src[j - 1] == "\\":
                        j += 1
                        continue
                    break
                out[j] = " "
                j += 1
            i = j
        elif ch == "/" and nxt == "*":
            j = i
            while j < n and not (src[j] == "*" and j + 1 < n and src[j + 1] == "/"):
                if src[j] != "\n":
                    out[j] = " "
                j += 1
            for k in range(j, min(j + 2, n)):
                out[k] = " "
            i = j + 2
        elif ch in ('"', "'"):
            q = ch
            out[i] = " "
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    if src[j] != "\n":
                        out[j] = " "
                    if j + 1 < n and src[j + 1] != "\n":
                        out[j + 1] = " "
                    j += 2
                    continue
                if src[j] == q:
                    out[j] = " "
                    j += 1
                    break
                if src[j] == "\n":
                    break
                out[j] = " "
                j += 1
            i = j
        else:
            i += 1
    return "".join(out)


_SHIFT3 = re.compile(r"<<<")


def launch_sites(src):
    """[(offset, line)] of every `<<<` outside comments and literals.

    `<<<` is not a C++ token outside a launch: `a << <b` is not valid and
    `a << (b<c)` needs the paren.  A false positive here costs a refusal
    (loud), never a silent skip.
    """
    blanked = blank_out(src)
    line_starts = [0]
    for m in re.finditer("\n", blanked):
        line_starts.append(m.end())
    out = []
    for m in _SHIFT3.finditer(blanked):
        off = m.start()
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        out.append((off, lo + 1))
    return out


_MEMBER_ASSIGN = re.compile(r"\b([A-Za-z_]\w*)\s*\.\s*([xyz])\s*=(?!=)")


def grid_member_assignments(src):
    """[(offset, line, text)] of any `<id>.x = ...` -- a constructor read after
    one of these is a lie about the launched geometry.  There are zero today."""
    blanked = blank_out(src)
    line_starts = [0]
    for m in re.finditer("\n", blanked):
        line_starts.append(m.end())
    out = []
    for m in _MEMBER_ASSIGN.finditer(blanked):
        off = m.start()
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= off:
                lo = mid
            else:
                hi = mid - 1
        out.append((off, lo + 1, m.group(0)))
    return out
