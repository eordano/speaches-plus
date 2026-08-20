#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "aiortc>=1.13.0,<1.14",
#   "httpx>=0.28",
#   "numpy>=2.0",
#   "soundfile>=0.13",
#   "av>=14.0,<14.3",  # 14.0-14.2 have macOS arm64 wheels; 14.3+ are sdist-only
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import re
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
from aiortc import (
    RTCConfiguration,
    RTCPeerConnection,
    RTCSessionDescription,
)
from aiortc.contrib.media import MediaStreamTrack
from av.audio.frame import AudioFrame

logger = logging.getLogger("e2e")

DEFAULT_TARGET = "http://localhost:8000"
DEFAULT_TTS_BASE = "http://localhost:8000"
DEFAULT_TTS_MODEL = "kokoro"
DEFAULT_VOICE = "af_heart"
DEFAULT_REALTIME_MODEL = "llm-default"
DEFAULT_TRANSCRIPTION_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
DEFAULT_AUTH = "Bearer dummy"

REFERENCE_TEXT = "the quick brown fox jumps over the lazy dog"

PUSH_SAMPLE_RATE = 48000
PUSH_CHANNELS = 1

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""
    duration_s: float = 0.0

@dataclass
class CollectedEvents:
    events: list[dict] = field(default_factory=list)
    event_timestamps_ms: list[int] = field(default_factory=list)
    transcriptions: list[str] = field(default_factory=list)
    audio_chunks: list[bytes] = field(default_factory=list)
    fragments: dict[str, dict[int, str]] = field(default_factory=dict)

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def text_match(expected: str, actual: str, *, max_extra_words: int = 2) -> bool:
    e = normalize(expected).split()
    a = normalize(actual).split()
    i = 0
    for word in a:
        if i < len(e) and word == e[i]:
            i += 1
    matched = i
    return matched >= len(e) - max_extra_words

async def generate_reference_audio(
    *,
    base_url: str,
    auth: str,
    text: str,
    model: str = DEFAULT_TTS_MODEL,
    voice: str = DEFAULT_VOICE,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, int]:
    if cache_path is not None and cache_path.exists():
        logger.info(f"Using cached reference audio: {cache_path}")
        data, sr = sf.read(str(cache_path), dtype="float32")
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.astype(np.float32), int(sr)

    logger.info(f"Requesting TTS for: {text!r}")
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{base_url}/v1/audio/speech",
            headers={"Authorization": auth, "Content-Type": "application/json"},
            json={
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": "wav",
            },
        )
        resp.raise_for_status()
        wav_bytes = resp.content

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(wav_bytes)
        logger.info(f"Cached reference audio: {cache_path}")

    data, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if data.ndim == 2:
        data = data.mean(axis=1)
    logger.info(f"Reference audio: sr={sr} duration={len(data) / sr:.2f}s samples={len(data)}")
    return data.astype(np.float32), int(sr)

class BufferAudioTrack(MediaStreamTrack):

    kind = "audio"

    def __init__(self, samples_f32: np.ndarray, sample_rate: int) -> None:
        super().__init__()
        if sample_rate != PUSH_SAMPLE_RATE:
            samples_f32 = _resample_linear(samples_f32, sample_rate, PUSH_SAMPLE_RATE)
        clipped = np.clip(samples_f32, -1.0, 1.0)
        self._samples_i16 = (clipped * 32767.0).astype(np.int16)
        self._frame_samples = PUSH_SAMPLE_RATE // 50
        self._cursor = 0
        self._pts = 0
        self._silence_pad = PUSH_SAMPLE_RATE * 4
        self._start_wall: float | None = None

    @property
    def total_samples(self) -> int:
        return len(self._samples_i16) + self._silence_pad

    async def recv(self) -> AudioFrame:
        if self._start_wall is None:
            self._start_wall = time.monotonic()

        target_wall = self._start_wall + (self._pts / PUSH_SAMPLE_RATE)
        delay = target_wall - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)

        end = self._cursor + self._frame_samples
        if self._cursor < len(self._samples_i16):
            slice_ = self._samples_i16[self._cursor : end]
            if len(slice_) < self._frame_samples:
                slice_ = np.concatenate(
                    [slice_, np.zeros(self._frame_samples - len(slice_), dtype=np.int16)]
                )
        elif self._cursor < self.total_samples:
            slice_ = np.zeros(self._frame_samples, dtype=np.int16)
        else:
            self.stop()
            raise asyncio.CancelledError("track exhausted")

        frame = AudioFrame.from_ndarray(slice_.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = PUSH_SAMPLE_RATE
        frame.pts = self._pts
        frame.time_base = __import__("fractions").Fraction(1, PUSH_SAMPLE_RATE)

        self._cursor = end
        self._pts += self._frame_samples
        return frame

def _resample_linear(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data
    duration = len(data) / sr_in
    n_out = int(duration * sr_out)
    x_in = np.arange(len(data))
    x_out = np.linspace(0, len(data) - 1, n_out)
    return np.interp(x_out, x_in, data).astype(np.float32)

def _try_parse_event(raw: str, fragments: dict[str, dict[int, str]]) -> dict | None:
    try:
        envelope = json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(raw)
        except Exception:
            return None

    etype = envelope.get("type")
    if etype == "full_message":
        payload_b64 = envelope.get("data", "")
        try:
            inner = json.loads(base64.b64decode(payload_b64).decode("utf-8"))
            return inner
        except Exception as exc:
            logger.warning(f"Failed to decode full_message: {exc}")
            return None
    if etype == "partial_message":
        eid = envelope.get("id", "")
        idx = envelope.get("fragment_index", 0)
        total = envelope.get("total_fragments", 1)
        bucket = fragments.setdefault(eid, {})
        bucket[idx] = envelope.get("data", "")
        if len(bucket) >= total:
            joined = "".join(bucket[i] for i in range(total))
            try:
                inner = json.loads(base64.b64decode(joined).decode("utf-8"))
            except Exception as exc:
                logger.warning(f"Failed to decode reassembled fragments: {exc}")
                fragments.pop(eid, None)
                return None
            fragments.pop(eid, None)
            return inner
        return None
    return envelope

async def run_realtime_session(
    *,
    target: str,
    auth: str,
    audio_f32: np.ndarray,
    sample_rate: int,
    model: str,
    transcription_model: str,
    intent: str,
    timeout_s: float,
    instructions: str | None = None,
) -> CollectedEvents:
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    collected = CollectedEvents()
    transcription_done = asyncio.Event()
    received_audio_done = asyncio.Event()
    t0 = time.monotonic()

    track = BufferAudioTrack(audio_f32, sample_rate)
    pc.addTrack(track)

    channel = pc.createDataChannel("oai-events")

    @channel.on("open")
    def _on_open() -> None:
        logger.info("Data channel opened (client side)")
        if instructions is not None:
            channel.send(json.dumps({
                "type": "session.update",
                "session": {"instructions": instructions},
            }))

    @channel.on("message")
    def _on_message(message: str) -> None:
        event = _try_parse_event(message, collected.fragments)
        if event is None:
            return
        etype = event.get("type", "?")
        collected.events.append(event)
        collected.event_timestamps_ms.append(int((time.monotonic() - t0) * 1000))
        logger.debug(f"<- {etype}")
        if etype == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            collected.transcriptions.append(transcript)
            logger.info(f"transcription.completed: {transcript!r}")
            transcription_done.set()
        elif etype == "conversation.item.input_audio_transcription.delta":
            logger.debug(f"transcription.delta: {event.get('delta', '')!r}")
        elif etype == "error":
            logger.error(f"Server error event: {json.dumps(event, indent=2)}")

    @pc.on("track")
    def _on_track(remote: MediaStreamTrack) -> None:
        logger.info(f"Inbound track: kind={remote.kind}")
        if remote.kind != "audio":
            return

        async def _drain() -> None:
            try:
                while True:
                    frame = await remote.recv()
                    arr = frame.to_ndarray()
                    collected.audio_chunks.append(arr.tobytes())
            except Exception as exc:
                logger.debug(f"Inbound track drain ended: {exc}")
            finally:
                received_audio_done.set()

        asyncio.create_task(_drain())

    @pc.on("connectionstatechange")
    def _on_state() -> None:
        logger.info(f"PC connection state: {pc.connectionState}")

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    logger.info(f"Local SDP offer ready ({len(pc.localDescription.sdp)} bytes)")

    url = f"{target}/v1/realtime?model={model}&intent={intent}"
    if intent == "transcription":
        url += f"&transcription_model={transcription_model}"
    logger.info(f"POST {url}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            headers={"Authorization": auth, "Content-Type": "application/sdp"},
            content=pc.localDescription.sdp,
        )
        if resp.status_code != 200:
            await pc.close()
            raise RuntimeError(
                f"Realtime POST failed: HTTP {resp.status_code}\n{resp.text[:500]}"
            )
        answer_sdp = resp.text

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))
    logger.info("Remote SDP answer applied -- peer connection negotiating")

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if transcription_done.is_set():
            await asyncio.sleep(0.5)
            break
        await asyncio.sleep(0.1)

    await pc.close()
    return collected

async def test_transcription_only(args: argparse.Namespace) -> tuple[TestResult, CollectedEvents]:
    t0 = time.monotonic()
    cache = Path(args.fixtures_dir) / f"ref_{abs(hash(args.text)) % (10**8)}.wav"
    audio, sr = await generate_reference_audio(
        base_url=args.tts_base,
        auth=args.auth,
        text=args.text,
        cache_path=cache,
    )

    collected = await run_realtime_session(
        target=args.target,
        auth=args.auth,
        audio_f32=audio,
        sample_rate=sr,
        model=args.model,
        transcription_model=args.transcription_model,
        intent="transcription",
        timeout_s=args.timeout,
    )

    duration = time.monotonic() - t0
    if not collected.transcriptions:
        return TestResult(
            name="transcription_only",
            passed=False,
            detail=f"no transcription event received (got {len(collected.events)} events; types={sorted({e.get('type') for e in collected.events})})",
            duration_s=duration,
        ), collected

    final = collected.transcriptions[-1]
    if text_match(args.text, final):
        return TestResult(
            name="transcription_only",
            passed=True,
            detail=f"expected={args.text!r} got={final!r}",
            duration_s=duration,
        ), collected
    return TestResult(
        name="transcription_only",
        passed=False,
        detail=f"text mismatch: expected={args.text!r} got={final!r}",
        duration_s=duration,
    ), collected

def write_trace_jsonl(
    *,
    path: str,
    config: dict,
    collected: CollectedEvents,
    result: TestResult,
) -> None:
    with open(path, "w") as fh:
        fh.write(json.dumps({"kind": "config", **config}) + "\n")
        for ts_ms, event in zip(collected.event_timestamps_ms, collected.events):
            fh.write(json.dumps({"kind": "event", "ts_ms": ts_ms, "event": event}) + "\n")
        fh.write(json.dumps({
            "kind": "result",
            "name": result.name,
            "passed": result.passed,
            "detail": result.detail,
            "duration_s": result.duration_s,
            "transcription": collected.transcriptions[-1] if collected.transcriptions else None,
        }) + "\n")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", default=os.environ.get("SPEACHES_TARGET", DEFAULT_TARGET),
                   help="Realtime target base URL (default: %(default)s)")
    p.add_argument("--tts-base", default=os.environ.get("SPEACHES_TTS_BASE", DEFAULT_TTS_BASE),
                   help="TTS base URL for fixture generation (default: %(default)s)")
    p.add_argument("--auth", default=os.environ.get("SPEACHES_AUTH", DEFAULT_AUTH),
                   help="Authorization header value (default: %(default)s)")
    p.add_argument("--model", default=DEFAULT_REALTIME_MODEL,
                   help="Realtime conversation model id (default: %(default)s)")
    p.add_argument("--transcription-model", default=DEFAULT_TRANSCRIPTION_MODEL,
                   help="STT model id (default: %(default)s)")
    p.add_argument("--text", default=REFERENCE_TEXT,
                   help="Reference text to TTS and round-trip (default: %(default)r)")
    p.add_argument("--timeout", type=float, default=30.0,
                   help="How long to wait for transcription event (default: %(default)s)")
    p.add_argument("--fixtures-dir", default=str(Path(__file__).parent / "fixtures"))
    p.add_argument(
        "--record-trace",
        default=None,
        metavar="PATH",
        help="Write a JSONL trace (config + outbound events + result) to PATH. "
             "Compare two such traces with conformance/lib/trace_diff.py.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()

async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        logging.getLogger("aiortc").setLevel(logging.WARNING)
        logging.getLogger("aioice").setLevel(logging.WARNING)

    print(f"target           = {args.target}")
    print(f"tts_base         = {args.tts_base}")
    print(f"model            = {args.model}")
    print(f"transcription    = {args.transcription_model}")
    print(f"reference text   = {args.text!r}")
    print()

    results: list[TestResult] = []
    collected: CollectedEvents | None = None
    try:
        result, collected = await test_transcription_only(args)
        results.append(result)
    except Exception as exc:
        logger.exception("Test crashed")
        results.append(TestResult(name="transcription_only", passed=False, detail=f"crash: {exc}"))

    if args.record_trace and collected is not None:
        write_trace_jsonl(
            path=args.record_trace,
            config={
                "phase": "phase1",
                "target": args.target,
                "intent": "transcription",
                "text": args.text,
                "model": args.model,
                "transcription_model": args.transcription_model,
            },
            collected=collected,
            result=results[0],
        )
        logger.info(f"trace written: {args.record_trace}")

    print("\n=== Results ===")
    for r in results:
        flag = "PASS" if r.passed else "FAIL"
        print(f"  [{flag}] {r.name} ({r.duration_s:.1f}s) -- {r.detail}")
    print()
    return 0 if all(r.passed for r in results) else 1

def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(main_async(args)))

def test_e2e_default_target():
    """Pytest entry point: skips unless SPEACHES_E2E_TARGET is set + reachable.

    Without this, pytest discovers zero test functions in this file and reports
    green even though the script asserts nothing via pytest semantics. With it,
    pytest runs the same orchestration as `python client/test_e2e.py` and fails
    loudly on non-zero exit.
    """
    target = os.environ.get("SPEACHES_E2E_TARGET")
    if not target:
        import pytest

        pytest.skip("set SPEACHES_E2E_TARGET=http://host:port to run this test")
    args = argparse.Namespace(
        target=target,
        tts_base=os.environ.get("SPEACHES_TTS_BASE", target),
        model=os.environ.get("SPEACHES_E2E_MODEL", "gpt-4o-realtime-preview"),
        transcription_model=os.environ.get(
            "SPEACHES_E2E_TRANSCRIPTION_MODEL", "whisper-large-v3-turbo"
        ),
        text=os.environ.get(
            "SPEACHES_E2E_TEXT", "the quick brown fox jumps over the lazy dog"
        ),
        record_trace=None,
        verbose=False,
    )
    rc = asyncio.run(main_async(args))
    assert rc == 0, f"test_e2e main_async returned non-zero exit code {rc}"

if __name__ == "__main__":
    main()
