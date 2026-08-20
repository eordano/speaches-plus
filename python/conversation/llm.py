from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator

import httpx

import env

log = logging.getLogger("conversation.llm")

class PredictedTokenBuffer:
    def __init__(self, cap: int) -> None:
        self.cap = max(1, cap)
        self.inner: deque[str] = deque()
        self.dropped: int = 0
        self.chars_seen: int = 0

    def push(self, token: str) -> bool:
        self.chars_seen = min(self.chars_seen + len(token), 0xFFFFFFFF)
        overflowed = False
        while len(self.inner) >= self.cap:
            self.inner.popleft()
            self.dropped = min(self.dropped + 1, 0xFFFFFFFF)
            overflowed = True
        self.inner.append(token)
        return overflowed

    def drain(self) -> list[str]:
        out = list(self.inner)
        self.inner.clear()
        return out

    def len(self) -> int:
        return len(self.inner)

    def __len__(self) -> int:
        return len(self.inner)

    def is_empty(self) -> bool:
        return len(self.inner) == 0

    def dropped_count(self) -> int:
        return self.dropped

@dataclass
class ChatMessage:
    role: str
    content: str

@dataclass
class LlmConfig:
    base_url: str
    api_key: str | None
    model: str

    @classmethod
    def from_env(cls) -> "LlmConfig | None":
        base_url = os.environ.get(env.CHAT_COMPLETION_BASE_URL)
        if base_url is None:
            return None
        api_key = os.environ.get(env.CHAT_COMPLETION_API_KEY)
        model = env.read_str(env.DEFAULT_REALTIME_CONVERSATION_MODEL, "default") or "default"
        return cls(base_url=base_url, api_key=api_key, model=model)

class LlmStreamError(Exception):
    pass

async def complete(
    cfg: LlmConfig,
    instructions: str | None,
    user_text: str,
    cancel: asyncio.Event | None = None,
) -> str:
    text_parts: list[str] = []
    last_err: Exception | None = None
    async for item in complete_stream(cfg, instructions, user_text, cancel=cancel):
        if isinstance(item, Exception):
            last_err = item
        else:
            text_parts.append(item)
    if last_err is not None:
        raise last_err
    text = "".join(text_parts)
    if not text:
        raise LlmStreamError("LLM stream ended with no content")
    return text

def _build_messages(instructions: str | None, user_text: str) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if instructions:
        messages.append(ChatMessage(role="system", content=instructions))
    messages.append(ChatMessage(role="user", content=user_text))
    return messages

async def complete_stream(
    cfg: LlmConfig,
    instructions: str | None,
    user_text: str,
    cancel: asyncio.Event | None = None,
) -> AsyncIterator[str]:
    async for delta in complete_stream_messages(
        cfg, _build_messages(instructions, user_text), cancel=cancel
    ):
        yield delta

async def complete_stream_messages(
    cfg: LlmConfig,
    messages: list[ChatMessage],
    cancel: asyncio.Event | None = None,
) -> AsyncIterator[str]:
    queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue(maxsize=64)
    task = asyncio.create_task(_stream_messages_into(cfg, messages, queue, cancel))
    try:
        while True:
            kind, payload = await queue.get()
            if kind == "delta":
                yield payload  # type: ignore[misc]
            elif kind == "error":
                raise payload  # type: ignore[misc]
            elif kind == "end":
                return
    finally:
        if not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

async def _stream_messages_into(
    cfg: LlmConfig,
    messages: list[ChatMessage],
    queue: asyncio.Queue[tuple[str, object]],
    cancel: asyncio.Event | None,
) -> None:
    try:
        url = f"{cfg.base_url.rstrip('/')}/chat/completions"
        body = {
            "model": cfg.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"

        emitted_any = False
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", url, json=body, headers=headers) as resp:
                if resp.status_code >= 400:
                    body_text = ""
                    try:
                        body_text = (await resp.aread()).decode("utf-8", errors="replace")
                    except Exception:
                        pass
                    await queue.put(
                        ("error", LlmStreamError(f"LLM upstream {resp.status_code}: {body_text}"))
                    )
                    return

                sse_buf = ""
                async for chunk in resp.aiter_text():
                    if cancel is not None and cancel.is_set():
                        return
                    sse_buf += chunk
                    while True:
                        idx = sse_buf.find("\n\n")
                        if idx < 0:
                            break
                        event = sse_buf[: idx + 2]
                        sse_buf = sse_buf[idx + 2 :]
                        for line in event.splitlines():
                            if not line.startswith("data: "):
                                continue
                            payload = line[len("data: ") :]
                            if payload == "[DONE]":
                                if not emitted_any:
                                    await queue.put(
                                        ("error", LlmStreamError("LLM stream ended with no content"))
                                    )
                                    return
                                await queue.put(("end", None))
                                return
                            try:
                                parsed = json.loads(payload)
                            except json.JSONDecodeError as err:
                                log.debug("skip unparseable SSE chunk: %s line=%s", err, payload)
                                continue
                            for choice in parsed.get("choices", []) or []:
                                delta = choice.get("delta") or {}
                                content = delta.get("content")
                                if content:
                                    emitted_any = True
                                    await queue.put(("delta", content))
                                msg = choice.get("message") or {}
                                mcontent = msg.get("content")
                                if mcontent:
                                    emitted_any = True
                                    await queue.put(("delta", mcontent))
        if not emitted_any:
            await queue.put(("error", LlmStreamError("LLM stream ended with no content")))
            return
        await queue.put(("end", None))
    except asyncio.CancelledError:
        raise
    except Exception as err:
        await queue.put(("error", err))

class SentenceChunker:
    def __init__(self) -> None:
        self.buf: str = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        terminators = (".", "!", "?", "\n")
        while True:
            idx = -1
            for i, ch in enumerate(self.buf):
                if ch in terminators:
                    idx = i
                    break
            if idx < 0:
                break
            end = idx + 1
            sentence = self.buf[:end].strip()
            self.buf = self.buf[end:]
            if sentence:
                out.append(sentence)
        return out

    def flush(self) -> str | None:
        trimmed = self.buf.strip()
        self.buf = ""
        if not trimmed:
            return None
        return trimmed

__all__ = [
    "ChatMessage",
    "LlmConfig",
    "LlmStreamError",
    "PredictedTokenBuffer",
    "SentenceChunker",
    "complete",
    "complete_stream",
    "complete_stream_messages",
]
