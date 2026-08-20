from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path

import env

from .constants import (
    DEFAULT_RETENTION_BYTES,
    DEFAULT_RETENTION_COUNT,
    DEFAULT_RETENTION_DAYS,
)

logger = logging.getLogger(__name__)

_ALLOWED_EXTENSIONS = {".ndjson", ".raw", ".json"}

def expand_home(raw: str) -> Path:
    if raw.startswith("~/"):
        home = os.environ.get("HOME")
        if home:
            return Path(home) / raw[2:]
    if raw == "~":
        home = os.environ.get("HOME")
        if home:
            return Path(home)
    return Path(raw)

def session_dir() -> Path | None:
    raw = env.read_str_or_none(env.INSPECT_SESSION_DIR)
    if raw is None:
        return None
    return expand_home(raw)

def retention_count() -> int:
    return env.read_int(env.INSPECT_RETENTION_COUNT, DEFAULT_RETENTION_COUNT)

def retention_bytes() -> int:
    return env.read_int(env.INSPECT_RETENTION_BYTES, DEFAULT_RETENTION_BYTES)

def retention_days() -> int:
    return env.read_int(env.INSPECT_RETENTION_DAYS, DEFAULT_RETENTION_DAYS)

def _file_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0

def _file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0

def _unlink(p: Path) -> None:
    try:
        p.unlink()
    except OSError as err:
        logger.warning("delete inspector artifact %s: %s", p, err)

def cleanup_on_startup(
    session_path: Path,
    max_count: int,
    max_bytes: int,
    max_days: int,
) -> None:
    if not session_path.exists():
        return
    try:
        entries = list(session_path.iterdir())
    except OSError as err:
        logger.warning("read session dir %s: %s", session_path, err)
        return

    sessions: dict[str, list[Path]] = {}
    for path in entries:
        if not path.is_file():
            continue
        ext = path.suffix
        if ext not in _ALLOWED_EXTENSIONS:
            continue
        stem = path.name.split(".", 1)[0]
        if not stem:
            continue
        sessions.setdefault(stem, []).append(path)

    now = time.time()
    if max_days > 0:
        max_age_s = max_days * 86_400
        to_remove: list[str] = []
        for sid, paths in sessions.items():
            mt = max((_file_mtime(p) for p in paths), default=0.0)
            if now - mt > max_age_s:
                to_remove.append(sid)
        for sid in to_remove:
            for p in sessions.pop(sid, []):
                _unlink(p)

    if max_count > 0 and len(sessions) > max_count:
        ordered: list[tuple[str, list[Path]]] = sorted(
            sessions.items(),
            key=lambda kv: max((_file_mtime(p) for p in kv[1]), default=0.0),
            reverse=True,
        )
        keep = ordered[:max_count]
        drop = ordered[max_count:]
        for _, paths in drop:
            for p in paths:
                _unlink(p)
        sessions = dict(keep)

    if max_bytes > 0:
        ordered = sorted(
            sessions.items(),
            key=lambda kv: max((_file_mtime(p) for p in kv[1]), default=0.0),
            reverse=True,
        )
        running = 0
        for _, paths in ordered:
            size = sum(_file_size(p) for p in paths)
            if running + size > max_bytes:
                for p in paths:
                    _unlink(p)
            else:
                running += size

async def retention_loop(interval_s: float = 300.0) -> None:
    while True:
        try:
            sd = session_dir()
            if sd is not None:
                cleanup_on_startup(
                    sd,
                    retention_count(),
                    retention_bytes(),
                    retention_days(),
                )
        except Exception as err:
            logger.warning("retention sweep failed: %s", err)
        try:
            await asyncio.sleep(interval_s)
        except asyncio.CancelledError:
            return

def run_startup_cleanup() -> None:
    sd = session_dir()
    if sd is None:
        return
    cleanup_on_startup(sd, retention_count(), retention_bytes(), retention_days())
