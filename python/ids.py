from __future__ import annotations

import threading
import uuid
from typing import Protocol

class IdSource(Protocol):
    def session(self) -> str: ...
    def item(self) -> str: ...
    def response(self) -> str: ...
    def event(self) -> str: ...
    def turn(self) -> str: ...
    def phrase(self) -> str: ...

class RandomIdSource:
    def session(self) -> str:
        return f"sess_{uuid.uuid4().hex}"

    def item(self) -> str:
        return f"item_{uuid.uuid4().hex}"

    def response(self) -> str:
        return f"resp_{uuid.uuid4().hex}"

    def event(self) -> str:
        return f"evt_{uuid.uuid4().hex}"

    def turn(self) -> str:
        return f"turn_{uuid.uuid4().hex}"

    def phrase(self) -> str:
        return f"phrase_{uuid.uuid4().hex}"

class CounterIdSource:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session = 0
        self._item = 0
        self._response = 0
        self._event = 0
        self._turn = 0
        self._phrase = 0

    def session(self) -> str:
        with self._lock:
            n = self._session
            self._session += 1
        return f"sess_{n:024d}"

    def item(self) -> str:
        with self._lock:
            n = self._item
            self._item += 1
        return f"item_{n:024d}"

    def response(self) -> str:
        with self._lock:
            n = self._response
            self._response += 1
        return f"resp_{n:024d}"

    def event(self) -> str:
        with self._lock:
            n = self._event
            self._event += 1
        return f"evt_{n:024d}"

    def turn(self) -> str:
        with self._lock:
            n = self._turn
            self._turn += 1
        return f"turn_{n:024d}"

    def phrase(self) -> str:
        with self._lock:
            n = self._phrase
            self._phrase += 1
        return f"phrase_{n:024d}"

_DEFAULT: IdSource = RandomIdSource()

def default_source() -> IdSource:
    return _DEFAULT

def next_session_id() -> str:
    return _DEFAULT.session()

def next_item_id() -> str:
    return _DEFAULT.item()

def next_response_id() -> str:
    return _DEFAULT.response()

def next_event_id() -> str:
    return _DEFAULT.event()

def next_turn_id() -> str:
    return _DEFAULT.turn()

def next_phrase_id() -> str:
    return _DEFAULT.phrase()
