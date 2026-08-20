from __future__ import annotations

import io
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw

@dataclass
class Rect:
    left: int
    top: int
    right: int
    bottom: int

def _parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    hex_str = hex_str.lstrip("#")
    if len(hex_str) == 3:
        hex_str = "".join(c * 2 for c in hex_str)
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (r, g, b)

def _shuffle_fill(image: Image.Image, rect: Rect) -> None:
    pixels = image.load()
    left, top, right, bottom = rect.left, rect.top, rect.right, rect.bottom
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return

    buckets: dict[tuple[int, int, int], list[tuple[int, int]]] = {}
    for y in range(top, bottom):
        for x in range(left, right):
            px = pixels[x, y]
            if isinstance(px, int):
                px = (px, px, px)
            elif len(px) == 4:
                px = px[:3]
            qr = (px[0] >> 4) << 4
            qg = (px[1] >> 4) << 4
            qb = (px[2] >> 4) << 4
            key = (qr, qg, qb)
            if key not in buckets:
                buckets[key] = []
            buckets[key].append((x, y))

    sorted_buckets = sorted(buckets.items(), key=lambda item: -len(item[1]))
    if len(sorted_buckets) < 2:
        color1 = sorted_buckets[0][0] if sorted_buckets else (0, 0, 0)
        color2 = (255, 255, 255)
    else:
        positions1 = sorted_buckets[0][1]
        positions2 = sorted_buckets[1][1]
        c1_r = sum(pixels[p[0], p[1]][0] if not isinstance(pixels[p[0], p[1]], int) else pixels[p[0], p[1]] for p in positions1) // len(positions1)
        c1_g = sum(pixels[p[0], p[1]][1] if not isinstance(pixels[p[0], p[1]], int) else pixels[p[0], p[1]] for p in positions1) // len(positions1)
        c1_b = sum(pixels[p[0], p[1]][2] if not isinstance(pixels[p[0], p[1]], int) else pixels[p[0], p[1]] for p in positions1) // len(positions1)
        color1 = (c1_r, c1_g, c1_b)
        c2_r = sum(pixels[p[0], p[1]][0] if not isinstance(pixels[p[0], p[1]], int) else pixels[p[0], p[1]] for p in positions2) // len(positions2)
        c2_g = sum(pixels[p[0], p[1]][1] if not isinstance(pixels[p[0], p[1]], int) else pixels[p[0], p[1]] for p in positions2) // len(positions2)
        c2_b = sum(pixels[p[0], p[1]][2] if not isinstance(pixels[p[0], p[1]], int) else pixels[p[0], p[1]] for p in positions2) // len(positions2)
        color2 = (c2_r, c2_g, c2_b)

    seed = left ^ top ^ right
    rng = random.Random(seed)

    for y in range(top, bottom):
        for x in range(left, right):
            color = color1 if rng.random() < 0.5 else color2
            pixels[x, y] = color

def render(image: Image.Image, rects: list[Rect], fill_mode: str = "solid", fill_color: str = "#000000") -> bytes:
    image = image.convert("RGB")

    if fill_mode == "shuffle":
        for rect in rects:
            _shuffle_fill(image, rect)
    else:
        draw = ImageDraw.Draw(image)
        color = _parse_hex_color(fill_color)
        for rect in rects:
            draw.rectangle(
                [rect.left, rect.top, rect.right, rect.bottom],
                fill=color,
            )

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()
