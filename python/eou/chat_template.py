from __future__ import annotations

from dataclasses import dataclass

from . import constants

@dataclass
class Turn:
    role: str
    content: str

    @classmethod
    def user(cls, content: str) -> "Turn":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> "Turn":
        return cls(role="assistant", content=content)

def format_qwen_chat(turns: list[Turn], partial: str) -> str:
    parts: list[str] = []
    for t in turns:
        role = t.role if t.role else "user"
        parts.append(constants.IM_START)
        parts.append(role)
        parts.append("\n")
        parts.append(t.content)
        parts.append(constants.IM_END)
        parts.append("\n")
    if partial:
        parts.append(constants.IM_START)
        parts.append("user\n")
        parts.append(partial)
    return "".join(parts)

def rolling_history(turns: list[Turn], max_turns: int) -> list[Turn]:
    if max_turns == 0 or len(turns) <= max_turns:
        return turns
    return turns[len(turns) - max_turns :]
