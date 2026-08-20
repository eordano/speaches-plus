from __future__ import annotations

from dataclasses import dataclass

from pii.classifier import PiiSpan
from pii.ocr import OcrToken

@dataclass
class LabeledRect:
    left: int
    top: int
    right: int
    bottom: int
    label: str

def _overlaps(token: OcrToken, span: PiiSpan) -> bool:
    return token.start < span.end_exclusive and token.end_exclusive > span.start

def _vertical_overlap_ratio(r1_top: int, r1_bottom: int, r2_top: int, r2_bottom: int) -> float:
    overlap_top = max(r1_top, r2_top)
    overlap_bottom = min(r1_bottom, r2_bottom)
    if overlap_bottom <= overlap_top:
        return 0.0
    overlap_height = overlap_bottom - overlap_top
    min_height = min(r1_bottom - r1_top, r2_bottom - r2_top)
    if min_height <= 0:
        return 0.0
    return overlap_height / min_height

def map_spans(
    tokens: list[OcrToken],
    spans: list[PiiSpan],
    image_width: int,
    image_height: int,
) -> list[LabeledRect]:
    rects: list[LabeledRect] = []

    for span in spans:
        matching_tokens = [t for t in tokens if _overlaps(t, span)]
        if not matching_tokens:
            continue

        token_rects = [
            (t.left, t.top, t.right, t.bottom)
            for t in matching_tokens
        ]
        token_rects.sort(key=lambda r: (r[1], r[0]))

        merged: list[list[int]] = []
        for left, top, right, bottom in token_rects:
            if merged:
                prev = merged[-1]
                h_gap = left - prev[2]
                v_ratio = _vertical_overlap_ratio(prev[1], prev[3], top, bottom)
                if h_gap < 8 and v_ratio > 0.5:
                    prev[0] = min(prev[0], left)
                    prev[1] = min(prev[1], top)
                    prev[2] = max(prev[2], right)
                    prev[3] = max(prev[3], bottom)
                    continue
            merged.append([left, top, right, bottom])

        for m in merged:
            pad = 4
            r_left = max(0, m[0] - pad)
            r_top = max(0, m[1] - pad)
            r_right = min(image_width, m[2] + pad)
            r_bottom = min(image_height, m[3] + pad)
            rects.append(LabeledRect(
                left=r_left,
                top=r_top,
                right=r_right,
                bottom=r_bottom,
                label=span.label,
            ))

    return rects
