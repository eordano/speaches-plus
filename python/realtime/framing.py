from __future__ import annotations

import base64
import json
import uuid
from typing import Any

from . import wire_defaults

MAX_FRAGMENT_SIZE = wire_defaults.DATA_CHANNEL_FRAGMENT_MAX

def frame_event(event: Any) -> list[str]:
    if hasattr(event, "to_json"):
        payload_obj = event.to_json()
    else:
        payload_obj = event
    raw = json.dumps(payload_obj, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(raw).decode("ascii")

    msg_id = uuid.uuid4().hex
    if len(encoded) <= MAX_FRAGMENT_SIZE:
        env = {"type": "full_message", "id": msg_id, "data": encoded}
        return [json.dumps(env, separators=(",", ":"))]

    total = (len(encoded) + MAX_FRAGMENT_SIZE - 1) // MAX_FRAGMENT_SIZE
    frames: list[str] = []
    for i in range(total):
        chunk = encoded[i * MAX_FRAGMENT_SIZE : (i + 1) * MAX_FRAGMENT_SIZE]
        env = {
            "type": "partial_message",
            "id": msg_id,
            "fragment_index": i,
            "total_fragments": total,
            "data": chunk,
        }
        frames.append(json.dumps(env, separators=(",", ":")))
    return frames
