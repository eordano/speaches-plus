"""OpenAI Realtime GA / v2 leniency layer.

The Realtime API moved from beta to GA on 2025-08-28 and the beta namespace
was removed on 2026-05-07. The wire shape changed in three places that
affect us:

  • `session.type` is now a required discriminator (`"realtime"` vs
    `"transcription"`) -- beta sessions did not carry it.
  • `audio.input.{format,transcription,turn_detection,...}` and
    `audio.output.{format,voice,...}` replace the flat top-level keys
    `input_audio_format`, `input_audio_transcription`, `turn_detection`,
    `output_audio_format`, `voice`.
  • `output_modalities` replaces `modalities`. Same enum (`"audio"`,
    `"text"`), same semantics.

We're lenient: accept either shape on input, emit both on output, so old
and new clients both work. Unknown v2-only event types (DTMF, MCP-tool,
output_audio_buffer.*, classifier safety events) are accepted as no-ops
instead of returning UNKNOWN_EVENT_TYPE -- beta clients may not know about
them but it's wrong to 400 just because a client is more current than us.
"""
from __future__ import annotations

from typing import Any

KNOWN_V2_NOOP_EVENTS: frozenset[str] = frozenset({
    "output_audio_buffer.clear",
    "output_audio_buffer.append",
    "input_audio_buffer.dtmf.received",
    "transcription_session.update",
    "response.cancel_audio",
})

def normalize_session_object(session_obj: dict[str, Any]) -> dict[str, Any]:
    """Flatten v2 nested shape into the flat beta shape used internally.

    Idempotent: if the object is already flat, returns it unchanged (after
    a shallow copy). Never raises -- invalid types are left for the
    downstream parser to reject with its proper error code.
    """
    if not isinstance(session_obj, dict):
        return session_obj
    out: dict[str, Any] = dict(session_obj)

    audio = out.get("audio")
    if isinstance(audio, dict):
        ai = audio.get("input")
        if isinstance(ai, dict):
            if "format" in ai and "input_audio_format" not in out:
                out["input_audio_format"] = ai["format"]
            if "transcription" in ai and "input_audio_transcription" not in out:
                out["input_audio_transcription"] = ai["transcription"]
            if "turn_detection" in ai and "turn_detection" not in out:
                out["turn_detection"] = ai["turn_detection"]
        ao = audio.get("output")
        if isinstance(ao, dict):
            if "format" in ao and "output_audio_format" not in out:
                out["output_audio_format"] = ao["format"]
            if "voice" in ao and "voice" not in out:
                out["voice"] = ao["voice"]

    if "output_modalities" in out and "modalities" not in out:
        out["modalities"] = out["output_modalities"]

    return out

def enrich_session_view(view: dict[str, Any]) -> dict[str, Any]:
    """Add the v2 nested keys alongside the flat beta keys for emission.

    Mutates and returns `view`. Always ensures `session.type` is present
    (defaults to `"realtime"`).
    """
    if not isinstance(view, dict):
        return view

    view.setdefault("type", "realtime")

    audio = view.get("audio")
    if not isinstance(audio, dict):
        audio = {}
        view["audio"] = audio

    inp = audio.get("input")
    if not isinstance(inp, dict):
        inp = {}
        audio["input"] = inp
    if "input_audio_format" in view:
        inp.setdefault("format", view["input_audio_format"])
    if "input_audio_transcription" in view:
        inp.setdefault("transcription", view["input_audio_transcription"])
    if "turn_detection" in view:
        inp.setdefault("turn_detection", view["turn_detection"])

    outp = audio.get("output")
    if not isinstance(outp, dict):
        outp = {}
        audio["output"] = outp
    if "output_audio_format" in view:
        outp.setdefault("format", view["output_audio_format"])
    if "voice" in view:
        outp.setdefault("voice", view["voice"])

    if "modalities" in view:
        view.setdefault("output_modalities", view["modalities"])

    return view

def is_known_v2_noop_event(event_type: str) -> bool:
    """True for v2 event types we accept but do not act on."""
    return event_type in KNOWN_V2_NOOP_EVENTS
