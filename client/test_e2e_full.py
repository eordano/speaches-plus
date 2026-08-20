#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "aiortc>=1.13.0,<1.14",
#   "httpx>=0.28",
#   "numpy>=2.0",
#   "soundfile>=0.13",
#   "av>=14.0,<14.3",
# ]
# ///

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from test_e2e import (
    BufferAudioTrack,
    DEFAULT_AUTH,
    DEFAULT_TTS_BASE,
    DEFAULT_TTS_MODEL,
    DEFAULT_VOICE,
    REFERENCE_TEXT,
    _try_parse_event,
    generate_reference_audio,
    normalize,
    text_match,
)

logger = logging.getLogger("e2e-full")

DEFAULT_RESPONSE_TEXT = "acknowledged"
DEFAULT_REALTIME_MODEL = "fake-llm-model"
DEFAULT_TRANSCRIPTION_MODEL = "deepdml/faster-whisper-large-v3-turbo-ct2"
DEFAULT_VOICE_OUT = "af_heart"
DEFAULT_SPEECH_MODEL = "speaches-ai/Kokoro-82M-v1.0-ONNX"
SPEACHES_HF_CACHE = os.environ.get(
    "HF_HUB_CACHE", str(Path.home() / ".cache/huggingface/hub")
)

def free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def find_speaches_binary() -> str:
    cand = subprocess.run(
        ["/bin/launchctl", "print", "system/org.nixos.speaches"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    m = re.search(r"(/nix/store/[a-z0-9]+-speaches-start)", cand)
    if not m:
        raise RuntimeError("could not locate speaches start-script via launchctl")
    start_script = Path(m.group(1)).read_text()
    m = re.search(r"(/nix/store/[a-z0-9]+-speaches-0\.\d+\.\d+)/bin/speaches", start_script)
    if not m:
        raise RuntimeError("could not parse speaches binary path from start-script")
    return f"{m.group(1)}/bin/speaches"

async def wait_for_health(url: str, timeout_s: float, name: str) -> None:
    deadline = time.monotonic() + timeout_s
    last_err = "timeout"
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(url)
                if r.status_code < 500:
                    elapsed = timeout_s - (deadline - time.monotonic())
                    logger.info(f"{name} ready after {elapsed:.1f}s ({url})")
                    return
                last_err = f"HTTP {r.status_code}"
            except Exception as exc:
                last_err = type(exc).__name__
            await asyncio.sleep(0.5)
    raise RuntimeError(f"{name} did not come up at {url} within {timeout_s}s ({last_err})")

class Subprocess:

    def __init__(self, name: str, cmd: list[str], env: dict[str, str] | None = None, cwd: str | None = None) -> None:
        self.name = name
        self.cmd = cmd
        self.env = env
        self.cwd = cwd
        self.proc: subprocess.Popen | None = None

    def start(self) -> None:
        logger.info(f"spawning {self.name}: {' '.join(self.cmd[:3])} ... [{len(self.cmd)} args]")
        log_path = Path(f"/tmp/test_e2e_full.{self.name}.log")
        self.log_fh = log_path.open("w")
        logger.info(f"  -> logs: {log_path}")
        self.proc = subprocess.Popen(
            self.cmd,
            env=self.env,
            cwd=self.cwd,
            stdout=self.log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                self.proc.wait(timeout=2)
        self.log_fh.close()

async def run_conversation_session(
    *,
    target: str,
    audio_f32: np.ndarray,
    sample_rate: int,
    model: str,
    transcription_model: str,
    timeout_s: float,
    voice: str,
    speech_model: str,
) -> dict[str, Any]:
    from aiortc import (
        RTCConfiguration,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.contrib.media import MediaStreamTrack

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    fragments: dict[str, dict[int, str]] = {}
    transcriptions: list[str] = []
    all_events: list[dict] = []
    event_ts_ms: list[int] = []
    response_audio_chunks: list[bytes] = []
    transcription_done = asyncio.Event()
    response_done = asyncio.Event()
    t0 = time.monotonic()

    track = BufferAudioTrack(audio_f32, sample_rate)
    pc.addTrack(track)

    channel = pc.createDataChannel("oai-events")

    @channel.on("message")
    def _on_message(message: str) -> None:
        event = _try_parse_event(message, fragments)
        if event is None:
            return
        etype = event.get("type", "?")
        all_events.append(event)
        event_ts_ms.append(int((time.monotonic() - t0) * 1000))
        logger.debug(f"<- {etype}")
        if etype == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "")
            transcriptions.append(transcript)
            logger.info(f"transcription.completed: {transcript!r}")
            transcription_done.set()
        elif etype == "response.done":
            response_done.set()
        elif etype == "error":
            logger.error(f"server error: {json.dumps(event, indent=2)}")

    @pc.on("track")
    def _on_track(remote: MediaStreamTrack) -> None:
        if remote.kind != "audio":
            return

        async def _drain() -> None:
            try:
                while True:
                    frame = await remote.recv()
                    response_audio_chunks.append(frame.to_ndarray().tobytes())
            except Exception as exc:
                logger.debug(f"inbound track ended: {exc}")

        asyncio.create_task(_drain())

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    url = f"{target}/v1/realtime?model={model}&intent=conversation&transcription_model={transcription_model}&voice={voice}&speech_model={speech_model}"
    logger.info(f"POST {url}")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            url,
            headers={"Content-Type": "application/sdp"},
            content=pc.localDescription.sdp,
        )
        if resp.status_code != 200:
            await pc.close()
            raise RuntimeError(f"realtime POST failed: HTTP {resp.status_code}\n{resp.text[:500]}")
        answer_sdp = resp.text

    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if response_done.is_set():
            await asyncio.sleep(0.5)
            break
        await asyncio.sleep(0.1)

    await pc.close()
    return {
        "transcription": transcriptions[-1] if transcriptions else None,
        "response_audio_pcm16_48k_stereo": b"".join(response_audio_chunks),
        "transcription_done": transcription_done.is_set(),
        "response_done": response_done.is_set(),
        "events": all_events,
        "event_ts_ms": event_ts_ms,
    }

async def transcribe_via_speaches(
    speaches_url: str, pcm16_48k_stereo: bytes, model: str
) -> str:
    arr = np.frombuffer(pcm16_48k_stereo, dtype=np.int16)
    arr = arr.reshape(-1, 2).astype(np.float32) / 32768.0
    mono = arr.mean(axis=1)
    target_sr = 16000
    n_out = int(len(mono) * target_sr / 48000)
    x_in = np.arange(len(mono))
    x_out = np.linspace(0, len(mono) - 1, n_out)
    mono_16k = np.interp(x_out, x_in, mono).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, mono_16k, target_sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    async with httpx.AsyncClient(timeout=60.0) as client:
        files = {"file": ("response.wav", buf.getvalue(), "audio/wav")}
        data = {"model": model, "response_format": "text"}
        r = await client.post(f"{speaches_url}/v1/audio/transcriptions", files=files, data=data)
        r.raise_for_status()
        return r.text.strip()

async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if not args.verbose:
        logging.getLogger("aiortc").setLevel(logging.WARNING)
        logging.getLogger("aioice").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    fake_llm_port = args.fake_llm_port or free_port()
    speaches_port = args.speaches_port or free_port()
    fake_llm_url = f"http://127.0.0.1:{fake_llm_port}"
    speaches_url = f"http://127.0.0.1:{speaches_port}"

    print(f"reference text   = {args.text!r}")
    print(f"response text    = {args.response!r}")
    print(f"fake LLM         = {fake_llm_url}")
    print(f"speaches         = {speaches_url}")
    print()

    fake_llm = Subprocess(
        name="fake_llm",
        cmd=[
            shutil.which("uv") or "uv",
            "run",
            "--script",
            str(Path(__file__).parent / "fake_llm.py"),
            "--port",
            str(fake_llm_port),
            "--response-text",
            args.response,
        ],
    )
    fake_llm.start()

    speaches_binary = args.speaches_binary or find_speaches_binary()
    speaches_share = Path(speaches_binary).parent.parent / "share/speaches"
    speaches_cwd = str(speaches_share) if speaches_share.exists() else None
    speaches_env = {
        **os.environ,
        "UVICORN_HOST": "127.0.0.1",
        "UVICORN_PORT": str(speaches_port),
        "HF_HUB_CACHE": SPEACHES_HF_CACHE,
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "DO_NOT_TRACK": "1",
        "WHISPER__INFERENCE_DEVICE": "cpu",
        "WHISPER__COMPUTE_TYPE": "float32",
        "DEFAULT_REALTIME_STT_MODEL": args.transcription_model,
        "CHAT_COMPLETION_BASE_URL": f"{fake_llm_url}/v1",
        "CHAT_COMPLETION_API_KEY": "fake-key",
        "DEFAULT_REALTIME_CONVERSATION_MODEL": args.model,
        "LOOPBACK_HOST_URL": speaches_url,
        "WARMUP_ALL_LOCAL_MODELS": "false",
        "LOG_LEVEL": "INFO",
    }
    speaches = Subprocess(
        name="speaches",
        cmd=[speaches_binary],
        env=speaches_env,
        cwd=speaches_cwd,
    )
    speaches.start()

    try:
        await wait_for_health(f"{fake_llm_url}/health", 15.0, "fake_llm")
        await wait_for_health(f"{speaches_url}/health", args.startup_timeout, "speaches")

        cfg: dict[str, Any] = {"reset": True, "response_text": args.response}
        if args.inject_llm_fail:
            cfg["fail_status"] = args.inject_llm_fail
            logger.info("injecting fake-LLM fail_status=%s", args.inject_llm_fail)
        if args.inject_llm_delay_ms:
            cfg["delay_ms"] = args.inject_llm_delay_ms
            logger.info("injecting fake-LLM delay_ms=%s", args.inject_llm_delay_ms)
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(f"{fake_llm_url}/test/configure", json=cfg)

        cache = Path(args.fixtures_dir) / f"ref_{abs(hash(args.text)) % (10**8)}.wav"
        audio, sr = await generate_reference_audio(
            base_url=args.tts_base,
            auth=args.auth,
            text=args.text,
            cache_path=cache,
            model=DEFAULT_TTS_MODEL,
            voice=DEFAULT_VOICE,
        )

        result = await run_conversation_session(
            target=speaches_url,
            audio_f32=audio,
            sample_rate=sr,
            model=args.model,
            transcription_model=args.transcription_model,
            voice=DEFAULT_VOICE_OUT,
            speech_model=DEFAULT_SPEECH_MODEL,
            timeout_s=args.timeout,
        )

        async with httpx.AsyncClient(timeout=5.0) as client:
            state = (await client.get(f"{fake_llm_url}/test/state")).json()
        forwarded_user_texts = [r["user_text"] for r in state.get("received", []) if r.get("user_text")]

        response_text_decoded: str | None = None
        if result["response_audio_pcm16_48k_stereo"]:
            try:
                response_text_decoded = await transcribe_via_speaches(
                    speaches_url, result["response_audio_pcm16_48k_stereo"], args.transcription_model
                )
            except Exception:
                logger.exception("response transcription failed")

        print("\n=== Results ===")
        ok_transcription = result["transcription"] is not None and text_match(args.text, result["transcription"])
        ok_fake_saw_text = any(text_match(args.text, t) for t in forwarded_user_texts)
        ok_response_audio = response_text_decoded is not None and text_match(args.response, response_text_decoded)

        rows = [
            ("[transcription leg] speaches -> text", ok_transcription, f"got={result['transcription']!r}"),
            (
                "[fake-llm leg]      speaches -> fake-LLM",
                ok_fake_saw_text,
                f"forwarded={forwarded_user_texts!r}",
            ),
            ("[tts leg]           fake-LLM -> audio -> text", ok_response_audio, f"got={response_text_decoded!r}"),
        ]
        for label, ok, detail in rows:
            flag = "PASS" if ok else "FAIL"
            print(f"  [{flag}] {label} -- {detail}")
        passed = all(ok for _, ok, _ in rows)
        print(f"\n  {'OVERALL PASS' if passed else 'OVERALL FAIL'}")

        if args.record_trace:
            with open(args.record_trace, "w") as fh:
                fh.write(json.dumps({
                    "kind": "config",
                    "phase": "phase2",
                    "intent": "conversation",
                    "text": args.text,
                    "response": args.response,
                    "voice": DEFAULT_VOICE_OUT,
                    "model": args.model,
                    "transcription_model": args.transcription_model,
                    "speech_model": DEFAULT_SPEECH_MODEL,
                    "inject_llm_fail": args.inject_llm_fail,
                    "inject_llm_delay_ms": args.inject_llm_delay_ms,
                }) + "\n")
                for ts_ms, ev in zip(result.get("event_ts_ms", []), result.get("events", [])):
                    fh.write(json.dumps({"kind": "event", "ts_ms": ts_ms, "event": ev}) + "\n")
                fh.write(json.dumps({
                    "kind": "result",
                    "passed": passed,
                    "transcription_pass": ok_transcription,
                    "fakellm_pass": ok_fake_saw_text,
                    "tts_pass": ok_response_audio,
                    "transcription": result["transcription"],
                    "response_decode": response_text_decoded,
                    "fakellm_user_texts": forwarded_user_texts,
                }) + "\n")
            logger.info(f"trace written: {args.record_trace}")

        if args.expect_fail:
            return 1 if passed else 0
        return 0 if passed else 1

    finally:
        speaches.stop()
        fake_llm.stop()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text", default=REFERENCE_TEXT, help="Reference text to TTS and round-trip")
    p.add_argument("--response", default=DEFAULT_RESPONSE_TEXT, help="Fixed text the fake LLM emits")
    p.add_argument("--model", default=DEFAULT_REALTIME_MODEL, help="Conversation model id stamped on Session")
    p.add_argument("--transcription-model", default=DEFAULT_TRANSCRIPTION_MODEL, help="STT model id")
    p.add_argument("--tts-base", default=os.environ.get("SPEACHES_TTS_BASE", DEFAULT_TTS_BASE))
    p.add_argument("--auth", default=os.environ.get("SPEACHES_AUTH", DEFAULT_AUTH))
    p.add_argument("--fake-llm-port", type=int, default=0, help="0 = pick free port")
    p.add_argument("--speaches-port", type=int, default=0, help="0 = pick free port")
    p.add_argument(
        "--speaches-binary",
        default=None,
        help="Path to the speaches launcher; default = autodiscover via launchctl",
    )
    p.add_argument(
        "--startup-timeout", type=float, default=180.0, help="Seconds to wait for spawned speaches to come up"
    )
    p.add_argument("--timeout", type=float, default=60.0, help="Seconds to wait for response.done event")
    p.add_argument("--fixtures-dir", default=str(Path(__file__).parent / "fixtures"))
    p.add_argument(
        "--inject-llm-fail",
        type=int,
        default=0,
        metavar="STATUS",
        help="Make the fake LLM return this HTTP status (e.g. 500) instead of streaming a response. "
             "Use to verify the realtime path stays alive when the upstream errors out.",
    )
    p.add_argument(
        "--inject-llm-delay-ms",
        type=int,
        default=0,
        help="Make the fake LLM sleep this many ms before responding. Use to exercise client timeouts.",
    )
    p.add_argument(
        "--expect-fail",
        action="store_true",
        help="Invert the exit code -- succeed if the run produces a [FAIL]. Useful for failure-injection probes.",
    )
    p.add_argument(
        "--record-trace",
        default=None,
        metavar="PATH",
        help="Write a JSONL trace (config + outbound events + result) to PATH. "
             "Compare two such traces with conformance/lib/trace_diff.py.",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(main_async(args)))

def test_e2e_full_default():
    """Pytest entry point. Skips unless SPEACHES_E2E_FULL=1.

    Without the gate, pytest discovers nothing here and reports green even
    though the script asserts nothing via pytest semantics. Setting the env
    var opts a CI runner with the prereq tooling (Nix shell, models) into
    actually running it.
    """
    if os.environ.get("SPEACHES_E2E_FULL") != "1":
        import pytest

        pytest.skip("set SPEACHES_E2E_FULL=1 to run this test")
    saved = sys.argv
    try:
        sys.argv = [sys.argv[0]]
        args = parse_args()
    finally:
        sys.argv = saved
    rc = asyncio.run(main_async(args))
    assert rc == 0, f"test_e2e_full main_async returned non-zero exit code {rc}"

if __name__ == "__main__":
    main()
