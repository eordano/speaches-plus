from __future__ import annotations

from typing import Optional

def _build_byte_to_char() -> list[str]:
    keep = [False] * 256
    for b in range(256):
        k = (
            (ord("!") <= b <= ord("~"))
            or (0xA1 <= b <= 0xAC)
            or (0xAE <= b <= 0xFF)
        )
        keep[b] = k
    bs: list[int] = [b for b in range(256) if keep[b]]
    cs: list[int] = list(bs)
    n = 0
    for b in range(256):
        if not keep[b]:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    m = ["\x00"] * 256
    for i, b in enumerate(bs):
        m[b] = chr(cs[i])
    return m

_BYTE_TO_CHAR: Optional[list[str]] = None
_CHAR_TO_BYTE: Optional[dict[str, int]] = None

def byte_to_char_table() -> list[str]:
    global _BYTE_TO_CHAR
    if _BYTE_TO_CHAR is None:
        _BYTE_TO_CHAR = _build_byte_to_char()
    return _BYTE_TO_CHAR

def _char_to_byte_map() -> dict[str, int]:
    global _CHAR_TO_BYTE
    if _CHAR_TO_BYTE is None:
        table = byte_to_char_table()
        _CHAR_TO_BYTE = {ch: b for b, ch in enumerate(table)}
    return _CHAR_TO_BYTE

def char_to_byte(c: str) -> Optional[int]:
    return _char_to_byte_map().get(c)

def bytes_to_bpe_chars(s: str) -> str:
    table = byte_to_char_table()
    raw = s.encode("utf-8")
    return "".join(table[b] for b in raw)

def bpe_chars_to_bytes(s: str) -> str:
    out = bytearray()
    for c in s:
        b = char_to_byte(c)
        if b is not None:
            out.append(b)
    return out.decode("utf-8", errors="replace")
