from __future__ import annotations

from .types import EouModel

HESITATION = ("uh", "um", "uhh", "umm", "er", "erm", "hmm", "like", "so")
CONTINUATIONS = (
    "and",
    "or",
    "but",
    "with",
    "the",
    "a",
    "an",
    "to",
    "of",
    "for",
    "is",
    "was",
    "are",
    "were",
    "because",
    "since",
    "if",
    "when",
    "while",
    "as",
    "than",
    "that",
    "which",
    "who",
    "whom",
    "whose",
)

_PUNCT_TRIM = (".", "!", "?", ",", ";", ":")

def _last_word(s: str) -> str:
    end = len(s)
    while end > 0:
        c = s[end - 1]
        if c.isspace() or c in _PUNCT_TRIM:
            end -= 1
        else:
            break
    trimmed = s[:end]
    out_chars: list[str] = []
    j = len(trimmed)
    while j > 0:
        c = trimmed[j - 1]
        if c.isalnum() or c == "'" or c == "-":
            out_chars.append(c)
            j -= 1
        else:
            break
    return "".join(reversed(out_chars))

class HeuristicEouModel(EouModel):
    @staticmethod
    def score_text(s: str) -> float:
        s = s.strip()
        if not s:
            return 0.1
        last_non_ws_char = s[-1] if s else " "
        last = _last_word(s).lower()
        if last_non_ws_char in (".", "!", "?"):
            return 0.95
        if last_non_ws_char in (",", ";", ":", "-"):
            return 0.25
        if not last:
            return 0.3
        if last in HESITATION:
            return 0.15
        if last in CONTINUATIONS:
            return 0.2
        return 0.6

    def score(self, context: str) -> float:
        return HeuristicEouModel.score_text(context)
