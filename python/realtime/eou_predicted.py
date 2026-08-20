from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from .state import PredictedRunner, PredictedSharedState

class PredictedTokenBuffer:
    def __init__(self, cap: int):
        self.cap = max(1, cap)
        self.inner: deque[str] = deque(maxlen=self.cap)
        self._dropped = 0

    def push(self, token: str) -> bool:
        overflowed = len(self.inner) == self.cap
        if overflowed:
            self._dropped += 1
        self.inner.append(token)
        return overflowed

    def __len__(self) -> int:
        return len(self.inner)

    def is_empty(self) -> bool:
        return not self.inner

    def dropped_count(self) -> int:
        return self._dropped

    def drain(self) -> list[str]:
        out = list(self.inner)
        self.inner.clear()
        return out

def spawn_predicted_stt(
    transcribe_async: Callable[[list[float]], Awaitable[str]],
    audio: list[float],
) -> PredictedRunner:
    shared = PredictedSharedState()

    async def _runner():
        try:
            text = await transcribe_async(audio)
            async with shared._lock:
                shared.user_transcript = (True, text, None)
        except Exception as err:
            async with shared._lock:
                shared.user_transcript = (True, None, str(err))
        shared.done.set()

    task = asyncio.create_task(_runner())
    return PredictedRunner(task=task, shared=shared)

async def await_predicted_stt(runner: PredictedRunner) -> tuple[str | None, str | None]:
    while True:
        async with runner.shared._lock:
            done, ok, err = runner.shared.user_transcript
            if done:
                return ok, err
        await runner.shared.done.wait()

@dataclass
class PredictedLlmShared:
    buffer: list[str] = field(default_factory=list)
    overflowed: bool = False
    dropped_tokens: int = 0
    chars_seen: int = 0
    done: bool = False
    cancelled: bool = False
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    progress: asyncio.Event = field(default_factory=asyncio.Event)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

@dataclass
class PredictedLlmRunner:
    task: asyncio.Task | None
    shared: PredictedLlmShared
    cap: int

    def abort(self) -> None:
        self.shared.cancelled = True
        self.shared.finished.set()
        self.shared.progress.set()
        if self.task is not None:
            self.task.cancel()

    async def snapshot(self) -> list[str]:
        async with self.shared._lock:
            return list(self.shared.buffer)

    async def snapshot_text(self) -> str:
        async with self.shared._lock:
            return "".join(self.shared.buffer)

    def dropped_count(self) -> int:
        return self.shared.dropped_tokens

    def overflowed(self) -> bool:
        return self.shared.overflowed

    def is_done(self) -> bool:
        return self.shared.done

    def is_cancelled(self) -> bool:
        return self.shared.cancelled

    def chars_seen(self) -> int:
        return self.shared.chars_seen

    async def wait_finished(self) -> None:
        while not (self.is_done() or self.is_cancelled()):
            await self.shared.finished.wait()

def spawn_predicted_llm(
    stream_factory: Callable[[], Any],
    cap: int,
) -> PredictedLlmRunner:
    shared = PredictedLlmShared()
    cap_n = max(1, cap)

    async def _runner():
        try:
            stream = stream_factory()
            async for delta in stream:
                if shared.cancelled:
                    break
                if not delta:
                    continue
                shared.chars_seen += len(delta)
                async with shared._lock:
                    shared.buffer.append(delta)
                    if len(shared.buffer) > cap_n:
                        drop_n = len(shared.buffer) - cap_n
                        del shared.buffer[:drop_n]
                        shared.dropped_tokens += drop_n
                        shared.overflowed = True
                shared.progress.set()
                shared.progress.clear()
        except Exception:
            pass
        finally:
            shared.done = True
            shared.finished.set()
            shared.progress.set()

    task = asyncio.create_task(_runner())
    return PredictedLlmRunner(task=task, shared=shared, cap=cap_n)

def transcripts_materially_differ(predicted: str, finalized: str, ratio: float) -> bool:
    p = predicted.strip().lower()
    f = finalized.strip().lower()
    if not p or not f:
        return p != f
    if p == f:
        return False
    pset = {c for c in p if not c.isspace()}
    fset = {c for c in f if not c.isspace()}
    inter = len(pset & fset)
    union = max(1, len(pset | fset))
    jaccard = inter / float(union)
    threshold = max(0.0, min(1.0, 1.0 - ratio))
    return jaccard < threshold
