"""Port of the rule-based text-EOU heuristic.

rust/src/eou/heuristic.rs::HeuristicEouModel::score_text is CANONICAL.
The scores here and in go/internal/eou/constants.go were realigned to it
on 2026-08-04; before that the three implementations disagreed on 5 of 6
cases, and the Rust one is the production fallback (realtime/session.rs
falls back to HeuristicEouModel when no EOU model loads).

The SCORES are now in lockstep. The rest is NOT, deliberately:
  - Go carries per-language hesitation/continuation tables (es/fr/de/it/pt);
    Rust and Python are English-only.
  - Go also treats CJK punctuation as terminators.
  - The three HESITATIONS word lists still differ ("well", "ah", "ugh",
    "ehm", "uhm" are in some and not others).
So identical text can still score differently across implementations via a
different BRANCH, even though each branch now returns the same number.
"""
from __future__ import annotations

HESITATIONS = frozenset({
    "uh", "um", "hmm", "er", "erm", "ugh", "uhh", "umm", "ehm", "like", "so",
})

CONTINUATIONS = frozenset({
    "and", "or", "but", "with", "the", "a", "an", "to", "of", "for",
    "is", "was", "are", "were", "because", "since", "if", "when",
    "while", "as", "than", "that", "which", "who", "whom", "whose",
})

SCORE_EMPTY = 0.1
SCORE_STRONG_TERMINATOR = 0.95
SCORE_SOFT_TERMINATOR = 0.25
SCORE_EMPTY_LAST_WORD = 0.3
SCORE_HESITATION = 0.15
SCORE_CONTINUATION = 0.2
SCORE_DEFAULT = 0.6

def heuristic_score(text: str) -> float:
    """Score a partial transcript on the [0, 1] EOU probability scale."""
    s = (text or "").strip()
    if not s:
        return SCORE_EMPTY
    last = s[-1]
    if last in {".", "!", "?"}:
        return SCORE_STRONG_TERMINATOR
    if last in {",", ";", ":", "-"}:
        return SCORE_SOFT_TERMINATOR
    last_word = "".join(c for c in s.split()[-1].lower() if c.isalnum() or c in "'-")
    if not last_word:
        return SCORE_EMPTY_LAST_WORD
    if last_word in HESITATIONS:
        return SCORE_HESITATION
    if last_word in CONTINUATIONS:
        return SCORE_CONTINUATION
    return SCORE_DEFAULT

def ends_strong_terminator(text: str) -> bool:
    s = (text or "").strip()
    return bool(s) and s[-1] in {".", "!", "?"}

def ends_soft_terminator(text: str) -> bool:
    s = (text or "").strip()
    return bool(s) and s[-1] in {",", ";", ":", "-"}

def last_word_is_continuation(text: str) -> bool:
    words = (text or "").strip().split()
    if not words:
        return False
    last = "".join(c for c in words[-1].lower() if c.isalnum() or c in "'-")
    return last in CONTINUATIONS
