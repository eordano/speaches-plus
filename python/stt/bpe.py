from __future__ import annotations

def _build_bpe_rune_to_byte() -> dict[str, int]:
    bs: list[int] = []
    for c in range(ord("!"), ord("~") + 1):
        bs.append(c)
    for c in range(ord("¡"), ord("¬") + 1):
        bs.append(c)
    for c in range(ord("®"), ord("ÿ") + 1):
        bs.append(c)
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    m: dict[str, int] = {}
    for i, codepoint in enumerate(cs):
        m[chr(codepoint)] = bs[i]
    return m

_BPE_RUNE_TO_BYTE: dict[str, int] = _build_bpe_rune_to_byte()
_BPE_BYTE_TO_RUNE: dict[int, str] = {v: k for k, v in _BPE_RUNE_TO_BYTE.items()}

def decode_bpe(s: str) -> str:
    out = bytearray()
    for r in s:
        b = _BPE_RUNE_TO_BYTE.get(r)
        if b is not None:
            out.append(b)
        else:
            out.extend(r.encode("utf-8"))
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return out.decode("utf-8", errors="replace")

def encode_bpe(s: str) -> str:
    out: list[str] = []
    for b in s.encode("utf-8"):
        out.append(_BPE_BYTE_TO_RUNE[b])
    return "".join(out)

class BPETokenizer:
    def __init__(self) -> None:
        self.rune_to_byte = dict(_BPE_RUNE_TO_BYTE)
        self.byte_to_rune = dict(_BPE_BYTE_TO_RUNE)

    def decode(self, s: str) -> str:
        return decode_bpe(s)

    def encode(self, s: str) -> str:
        return encode_bpe(s)
