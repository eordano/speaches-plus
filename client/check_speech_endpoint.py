#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx>=0.28",
# ]
# ///
"""
Smoke-test the /v1/audio/speech endpoint behavior against an arbitrary
target server, comparing it to the Python speaches reference contract.

Cases exercised:

1. Missing required field           -> 422 + FastAPI {"detail":[...]} shape
2. Out-of-range speed               -> 200 + empty body (Python validates
                                       inside the streaming generator,
                                       so headers are already flushed)
3. OpenAI voice alias (alloy/etc.)  -> 200 + non-empty audio (server
                                       falls back to default Kokoro voice)
4. Each response_format             -> right MIME type + non-empty body
5. SSE stream_format                -> text/event-stream, ends with
                                       a `speech.audio.done` event

Usage
-----
    ./client/check_speech_endpoint.py --target http://localhost:1327
    ./client/check_speech_endpoint.py --target http://localhost:8765
    ./client/check_speech_endpoint.py --target http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass

import httpx

DEFAULT_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
DEFAULT_VOICE = "af_heart"
ALL_FORMATS = ("pcm", "mp3", "wav", "flac", "opus", "aac")

EXPECTED_MIME = {
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "opus": "audio/opus",
    "aac": "audio/aac",
}

@dataclass
class Case:
    name: str
    passed: bool
    detail: str = ""

def post(client: httpx.Client, url: str, body: dict, timeout: float = 30.0) -> httpx.Response:
    return client.post(url, json=body, timeout=timeout)

def case_missing_voice(client: httpx.Client, base: str) -> Case:
    r = post(client, f"{base}/v1/audio/speech", {"model": DEFAULT_MODEL, "input": "hello"})
    if r.status_code != 422:
        return Case("missing voice -> 422", False, f"got {r.status_code}")
    try:
        body = r.json()
    except Exception as exc:
        return Case("missing voice -> 422", False, f"non-JSON body: {exc}")
    detail = body.get("detail")
    if not isinstance(detail, list) or not detail:
        return Case("missing voice -> 422", False, "detail not a list")
    entry = detail[0]
    if entry.get("type") != "missing":
        return Case("missing voice -> 422", False, f"wrong type: {entry.get('type')}")
    if entry.get("loc") != ["body", "voice"]:
        return Case("missing voice -> 422", False, f"wrong loc: {entry.get('loc')}")
    return Case("missing voice -> 422", True, "FastAPI-shaped detail")

def case_bad_speed(client: httpx.Client, base: str, allow_python_quirk: bool) -> Case:
    """Out-of-range speed.

    Our Go and Rust impls return 400 with an OpenAI-style error
    envelope (`{"error":{"message":...,"type":"invalid_request_error",
    "param":"speed","code":"out_of_range"}}`). Python speaches instead
    validates inside the streaming generator and closes the chunked
    response mid-flight with 200 + zero bytes -- `--allow-python-quirk`
    accepts that as a pass when probing the Python reference.
    """
    body = {
        "model": DEFAULT_MODEL,
        "input": "hello",
        "voice": DEFAULT_VOICE,
        "speed": 3.0,
        "response_format": "pcm",
    }
    name = "speed=3.0 -> 400"
    try:
        r = post(client, f"{base}/v1/audio/speech", body)
    except httpx.RemoteProtocolError as exc:
        if allow_python_quirk:
            return Case(name, True, f"Python-style early close: {exc}")
        return Case(name, False, f"got mid-stream close: {exc}")
    if r.status_code == 200 and len(r.content) == 0 and allow_python_quirk:
        return Case(name, True, "Python-style 200 + empty body")
    if r.status_code != 400:
        return Case(name, False, f"got {r.status_code}")
    try:
        body_obj = r.json()
    except Exception as exc:
        return Case(name, False, f"non-JSON body: {exc}")
    err = body_obj.get("error")
    if not isinstance(err, dict):
        return Case(name, False, "missing 'error' object")
    if err.get("type") != "invalid_request_error" or err.get("param") != "speed":
        return Case(name, False, f"wrong envelope: {err}")
    return Case(name, True, f"OpenAI-style error: {err.get('message','')!r}")

def case_openai_alias(client: httpx.Client, base: str) -> Case:
    body = {
        "model": DEFAULT_MODEL,
        "input": "Hello there.",
        "voice": "alloy",
        "response_format": "wav",
    }
    r = post(client, f"{base}/v1/audio/speech", body, timeout=60.0)
    if r.status_code != 200:
        return Case("voice=alloy -> 200 audio", False, f"got {r.status_code}")
    if len(r.content) < 1024:
        return Case("voice=alloy -> 200 audio", False, f"only {len(r.content)} bytes")
    if not r.content[:4] == b"RIFF":
        return Case("voice=alloy -> 200 audio", False, "not a RIFF/WAV header")
    return Case("voice=alloy -> 200 audio", True, f"{len(r.content)} bytes WAV")

def case_format(client: httpx.Client, base: str, fmt: str) -> Case:
    body = {
        "model": DEFAULT_MODEL,
        "input": "Quick brown fox.",
        "voice": DEFAULT_VOICE,
        "response_format": fmt,
    }
    r = post(client, f"{base}/v1/audio/speech", body, timeout=60.0)
    name = f"format={fmt}"
    if r.status_code != 200:
        return Case(name, False, f"got {r.status_code}")
    ctype = r.headers.get("content-type", "")
    expected = EXPECTED_MIME[fmt]
    if not ctype.startswith(expected):
        return Case(name, False, f"got Content-Type {ctype!r}, want {expected!r}")
    if len(r.content) < 100:
        return Case(name, False, f"got only {len(r.content)} bytes")
    return Case(name, True, f"{len(r.content)} bytes, {ctype}")

def case_sse(client: httpx.Client, base: str) -> Case:
    body = {
        "model": DEFAULT_MODEL,
        "input": "First sentence. Second sentence.",
        "voice": DEFAULT_VOICE,
        "stream_format": "sse",
    }
    r = post(client, f"{base}/v1/audio/speech", body, timeout=60.0)
    if r.status_code != 200:
        return Case("stream_format=sse", False, f"got {r.status_code}")
    ctype = r.headers.get("content-type", "")
    if "text/event-stream" not in ctype:
        return Case("stream_format=sse", False, f"got Content-Type {ctype!r}")
    text = r.text
    events = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip().startswith("data:")]
    if not events:
        return Case("stream_format=sse", False, "no events")
    delta_count = 0
    saw_done = False
    for ev in events:
        payload = ev[len("data:"):].strip()
        try:
            obj = json.loads(payload)
        except Exception:
            return Case("stream_format=sse", False, f"non-JSON event: {payload}")
        t = obj.get("type")
        if t == "speech.audio.delta":
            delta_count += 1
            if "audio" not in obj or not isinstance(obj["audio"], str):
                return Case("stream_format=sse", False, "delta missing audio (base64) field")
        elif t == "speech.audio.done":
            saw_done = True
            if "token_usage" not in obj:
                return Case("stream_format=sse", False, "done missing token_usage")
        else:
            return Case("stream_format=sse", False, f"unknown event type: {t}")
    if not saw_done:
        return Case("stream_format=sse", False, "no speech.audio.done event")
    return Case("stream_format=sse", True, f"{delta_count} deltas + done")

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default="http://localhost:1327")
    p.add_argument("--skip-formats", action="store_true",
                   help="Skip the per-format matrix (faster smoke test)")
    p.add_argument("--skip-alias", action="store_true",
                   help="Skip the OpenAI voice alias test")
    p.add_argument("--skip-sse", action="store_true",
                   help="Skip the SSE stream_format test")
    p.add_argument("--allow-python-quirk", action="store_true",
                   help="Treat Python speaches' 200+empty-body response on "
                        "out-of-range speed as a pass. Off by default -- Go "
                        "and Rust deliberately diverge to return 400.")
    args = p.parse_args()

    cases: list[Case] = []
    with httpx.Client() as client:
        cases.append(case_missing_voice(client, args.target))
        cases.append(case_bad_speed(client, args.target, args.allow_python_quirk))
        if not args.skip_alias:
            cases.append(case_openai_alias(client, args.target))
        if not args.skip_formats:
            for fmt in ALL_FORMATS:
                cases.append(case_format(client, args.target, fmt))
        if not args.skip_sse:
            cases.append(case_sse(client, args.target))

    width = max(len(c.name) for c in cases)
    passed = 0
    for c in cases:
        marker = "PASS" if c.passed else "FAIL"
        line = f"  [{marker}] {c.name:<{width}}"
        if c.detail:
            line += f"   {c.detail}"
        print(line)
        if c.passed:
            passed += 1

    print()
    print(f"{passed}/{len(cases)} cases passed against {args.target}")
    return 0 if passed == len(cases) else 1

if __name__ == "__main__":
    sys.exit(main())
