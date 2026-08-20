from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class SpecialPiece:
    text: str
    id: int
    special: bool

class SpecialNode:
    __slots__ = ("children", "terminal", "id", "content")

    def __init__(self) -> None:
        self.children: dict[int, "SpecialNode"] = {}
        self.terminal: bool = False
        self.id: int = 0
        self.content: str = ""

    def insert(self, s: str, id: int) -> None:
        cur = self
        for b in s.encode("utf-8"):
            nxt = cur.children.get(b)
            if nxt is None:
                nxt = SpecialNode()
                cur.children[b] = nxt
            cur = nxt
        cur.terminal = True
        cur.id = id
        cur.content = s

    def split(self, text: str) -> list[SpecialPiece]:
        data = text.encode("utf-8")
        out: list[SpecialPiece] = []
        plain = bytearray()
        i = 0
        n = len(data)

        def flush() -> None:
            if plain:
                out.append(
                    SpecialPiece(
                        text=plain.decode("utf-8", errors="replace"),
                        id=-1,
                        special=False,
                    )
                )
                plain.clear()

        while i < n:
            mid, mlen, mtext = self._match_at(data, i)
            if mlen > 0:
                flush()
                out.append(SpecialPiece(text=mtext, id=mid, special=True))
                i += mlen
            else:
                plain.append(data[i])
                i += 1
        flush()
        return out

    def _match_at(self, data: bytes, start: int) -> tuple[int, int, str]:
        cur = self
        match_id = -1
        match_end = 0
        match_text = ""
        i = start
        n = len(data)
        while i < n:
            nxt = cur.children.get(data[i])
            if nxt is None:
                break
            cur = nxt
            if cur.terminal:
                match_id = cur.id
                match_end = i + 1
                match_text = cur.content
            i += 1
        if match_end == 0:
            return (-1, 0, "")
        return (match_id, match_end - start, match_text)
