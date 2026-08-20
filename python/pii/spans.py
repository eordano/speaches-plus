from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

@dataclass
class PiiSpan:
    start: int
    endExclusive: int
    label: str

def assemble_spans(
    labels: Sequence[str],
    offsets: Sequence[Tuple[int, int]],
    attention_mask: Sequence[int],
) -> List[PiiSpan]:
    out: List[PiiSpan] = []
    open_label: str | None = None
    open_start = -1
    open_end = -1

    def close():
        nonlocal open_label, open_start, open_end
        if open_label is not None and open_start >= 0 and open_end > open_start:
            out.append(PiiSpan(open_start, open_end, open_label))
        open_label = None
        open_start = -1
        open_end = -1

    for i, tag in enumerate(labels):
        if i >= len(attention_mask) or attention_mask[i] == 0:
            continue
        s, e = offsets[i]
        if e <= s:
            continue
        if tag == "O":
            close()
            continue
        dash = tag.find("-")
        if dash < 0:
            close()
            continue
        prefix = tag[:dash]
        cls = tag[dash + 1 :]

        if prefix == "B":
            close()
            open_label, open_start, open_end = cls, s, e
        elif prefix == "I":
            if open_label == cls:
                open_end = e
            else:
                close()
                open_label, open_start, open_end = cls, s, e
        elif prefix == "E":
            if open_label == cls:
                open_end = e
                close()
            else:
                close()
                out.append(PiiSpan(s, e, cls))
        elif prefix == "S":
            close()
            out.append(PiiSpan(s, e, cls))
        else:
            close()

    close()
    return out
