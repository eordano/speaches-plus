from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import constants
from .byte_map import bpe_chars_to_bytes, bytes_to_bpe_chars
from .special_trie import SpecialNode

class Tokenizer:
    def __init__(self) -> None:
        self.vocab: dict[str, int] = {}
        self.id_to_token: dict[int, str] = {}
        self.merges: dict[tuple[str, str], int] = {}
        self.added_tokens: dict[str, int] = {}
        self.special_trie: SpecialNode = SpecialNode()
        self._im_start_id: int = -1
        self._im_end_id: int = -1
        self._has_im_tokens: bool = False

    @classmethod
    def load_from_path(cls, path) -> "Tokenizer":
        raw = Path(path).read_text(encoding="utf-8")
        return cls.load_from_json(raw)

    @classmethod
    def load_from_json(cls, raw: str) -> "Tokenizer":
        try:
            tj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"parse tokenizer json: {e}") from e
        model = tj.get("model") or {}
        ty = model.get("type", "")
        if ty and ty != "BPE":
            raise ValueError(
                f"unsupported tokenizer model type {ty!r} (only BPE supported)"
            )
        t = cls()
        vocab = model.get("vocab") or {}
        for tok, tid in vocab.items():
            tid_int = int(tid)
            t.vocab[tok] = tid_int
            t.id_to_token[tid_int] = tok
        merges_list = model.get("merges") or []
        for rank, m in enumerate(merges_list):
            parsed = _parse_merge(m)
            if parsed is not None:
                t.merges[parsed] = rank
        added = tj.get("added_tokens") or []
        for at in added:
            content = at.get("content")
            tid = at.get("id")
            if content is None or tid is None:
                continue
            tid_int = int(tid)
            t.added_tokens[content] = tid_int
            t.id_to_token[tid_int] = content
            if content not in t.vocab:
                t.vocab[content] = tid_int
            t.special_trie.insert(content, tid_int)
            if content == constants.IM_START:
                t._im_start_id = tid_int
            elif content == constants.IM_END:
                t._im_end_id = tid_int
        if t._im_end_id < 0:
            v = t.vocab.get(constants.IM_END)
            if v is not None:
                t._im_end_id = int(v)
        if t._im_start_id < 0:
            v = t.vocab.get(constants.IM_START)
            if v is not None:
                t._im_start_id = int(v)
        t._has_im_tokens = t._im_end_id >= 0
        return t

    def im_end_id(self) -> int:
        return self._im_end_id

    def im_start_id(self) -> int:
        return self._im_start_id

    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def has_im_tokens(self) -> bool:
        return self._has_im_tokens

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
        pieces = self.special_trie.split(text)
        out: list[int] = []
        for p in pieces:
            if p.special:
                out.append(p.id)
                continue
            out.extend(self._encode_plain(p.text))
        return out

    def _encode_plain(self, text: str) -> list[int]:
        if not text:
            return []
        splits = gpt2_pre_split(text)
        out: list[int] = []
        for s in splits:
            bpe_input = bytes_to_bpe_chars(s)
            merged = self._bpe_merges(bpe_input)
            for tok in merged:
                tid = self.vocab.get(tok)
                if tid is not None:
                    out.append(tid)
                else:
                    for r in tok:
                        sub = self.vocab.get(r)
                        if sub is not None:
                            out.append(sub)
        return out

    def _bpe_merges(self, s: str) -> list[str]:
        if not s:
            return []
        tokens: list[str] = list(s)
        while True:
            best_rank = -1
            best_idx = -1
            for i in range(len(tokens) - 1):
                key = (tokens[i], tokens[i + 1])
                rank = self.merges.get(key)
                if rank is None:
                    continue
                if best_rank < 0 or rank < best_rank:
                    best_rank = rank
                    best_idx = i
            if best_idx < 0:
                break
            merged = tokens[best_idx] + tokens[best_idx + 1]
            tokens[best_idx : best_idx + 2] = [merged]
        return tokens

    def decode(self, ids: list[int]) -> str:
        joined: list[str] = []
        for i in ids:
            tok = self.id_to_token.get(i)
            if tok is not None:
                joined.append(tok)
        return bpe_chars_to_bytes("".join(joined))

def _parse_merge(raw) -> Optional[tuple[str, str]]:
    if isinstance(raw, str):
        idx = raw.find(" ")
        if idx <= 0 or idx >= len(raw) - 1:
            return None
        return (raw[:idx], raw[idx + 1 :])
    if isinstance(raw, list):
        if len(raw) != 2:
            return None
        a, b = raw[0], raw[1]
        if not isinstance(a, str) or not isinstance(b, str):
            return None
        return (a, b)
    return None

def gpt2_pre_split(text: str) -> list[str]:
    chars = list(text)
    n = len(chars)
    out: list[str] = []
    i = 0
    while i < n:
        if chars[i] == "'" and i + 1 < n:
            nxt = chars[i + 1]
            if nxt in ("s", "d", "m", "t"):
                out.append("".join(chars[i : i + 2]))
                i += 2
                continue
            if i + 2 < n:
                two = "".join(chars[i + 1 : i + 3])
                if two in ("ll", "ve", "re"):
                    out.append("".join(chars[i : i + 3]))
                    i += 3
                    continue
        start = i
        leading_space = chars[i] == " "
        probe = i + 1 if leading_space else i
        if probe < n:
            c = chars[probe]
            if _is_letter(c):
                j = probe
                while j < n and _is_letter(chars[j]):
                    j += 1
                if j > probe:
                    out.append("".join(chars[start:j]))
                    i = j
                    continue
            if _is_number(c):
                j = probe
                while j < n and _is_number(chars[j]):
                    j += 1
                if j > probe:
                    out.append("".join(chars[start:j]))
                    i = j
                    continue
            if not c.isspace() and not _is_letter(c) and not _is_number(c):
                j = probe
                while (
                    j < n
                    and not chars[j].isspace()
                    and not _is_letter(chars[j])
                    and not _is_number(chars[j])
                ):
                    j += 1
                if j > probe:
                    out.append("".join(chars[start:j]))
                    i = j
                    continue
        if chars[i].isspace():
            j = i
            while j < n and chars[j].isspace():
                j += 1
            if j < n:
                take_to = j
                if take_to - 1 > i:
                    out.append("".join(chars[i : take_to - 1]))
                out.append("".join(chars[take_to - 1 : j + 1]))
                i = j + 1
                continue
            else:
                out.append("".join(chars[i:j]))
                i = j
                continue
        out.append(chars[i])
        i += 1
    return out

def _is_letter(c: str) -> bool:
    return c.isalpha()

def _is_number(c: str) -> bool:
    return c.isnumeric()
