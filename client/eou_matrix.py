#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "aiortc>=1.13.0,<1.14",
#   "httpx>=0.28",
#   "numpy>=2.0",
#   "soundfile>=0.13",
#   "av>=14.0,<14.3",
#   "websockets>=13",
# ]
# ///
"""eou_matrix -- drive a speaches-plus binary (Go OR Rust) through the
realtime API against a fixture x EOU_KIND x fusion-rule matrix and
collect the per-session inspector ndjson + audio sidecars.

Two transports:

  ws    -- open `/v1/realtime` as a WebSocket (only the Go binary speaks
          this; the Rust server is WebRTC-only). Sends `pcm16_16k`
          input_audio_buffer.append frames at realtime cadence.
  rtc   -- open `/v1/realtime` as an SDP offer, send Opus-encoded audio
          via aiortc. Works for both Go and Rust binaries.

Replaces:

  go/cmd/eou-runner          (Go-side WS driver -- deleted)
  client/eou_inspector_matrix_rust.py (Rust-side WebRTC driver -- deleted)

Single Python script means the Go and Rust matrices use the same code
path and emit comparable inspector files.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import io
import json
import logging
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

logger = logging.getLogger("matrix")

DEFAULT_HF_HUB_CACHE = str(Path.home() / ".cache/huggingface/hub")
DEFAULT_SMART_TURN = str(Path(__file__).resolve().parents[1] / "rust/models/smart-turn-v3.onnx")

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

async def wait_health(url: str, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(url)
                if r.status_code < 500:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.3)
    raise RuntimeError(f"server did not become healthy at {url} in {timeout_s}s")

def spawn_fake_llm(port: int, log_path: Path) -> subprocess.Popen:
    cmd = [
        shutil.which("uv") or "uv", "run", "--script",
        str(Path(__file__).parent / "fake_llm.py"),
        "--port", str(port),
        "--response-text", "OK got it.",
    ]
    fh = log_path.open("w")
    return subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)

def spawn_server(binary: str, env: dict, log_path: Path) -> subprocess.Popen:
    fh = log_path.open("w")
    return subprocess.Popen([binary], env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)

def kill_proc(p: subprocess.Popen | None) -> None:
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        p.wait(timeout=5)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)

def f32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    s = np.clip(samples, -1.0, 1.0)
    return (s * 32767.0).astype("<i2").tobytes()

async def run_ws(target: str, audio_f32: np.ndarray, sr: int,
                  timeout_s: float) -> tuple[bool, list[str]]:
    import websockets
    if sr != 16000:
        duration = len(audio_f32) / sr
        n_out = int(duration * 16000)
        audio_f32 = np.interp(np.linspace(0, len(audio_f32) - 1, n_out),
                               np.arange(len(audio_f32)), audio_f32).astype(np.float32)

    url = (f"{target}/v1/realtime?model=eou-runner&intent=conversation"
            f"&transcription_model=ct2&voice=af_heart&speech_model=kokoro")
    url = url.replace("http://", "ws://").replace("https://", "wss://")

    events: list[dict] = []

    async with websockets.connect(url, subprotocols=["realtime"],
                                    max_size=8 * 1024 * 1024) as ws:
        async def reader():
            try:
                async for msg in ws:
                    try:
                        events.append(json.loads(msg))
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass

        reader_task = asyncio.create_task(reader())
        try:
            await ws.send(json.dumps({
                "type": "session.update",
                "session": {
                    "input_audio_format": "pcm16_16k",
                    "output_audio_format": "pcm16",
                },
            }))
            chunk = 1600
            for i in range(0, len(audio_f32), chunk):
                pcm = f32_to_pcm16_bytes(audio_f32[i:i + chunk])
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }))
                await asyncio.sleep(0.1)

            tail = np.zeros(1600, dtype=np.float32)
            for _ in range(20):
                pcm = f32_to_pcm16_bytes(tail)
                await ws.send(json.dumps({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                }))
                await asyncio.sleep(0.1)

            end_at = time.monotonic() + timeout_s
            last_change = time.monotonic()
            last_count = 0
            while time.monotonic() < end_at:
                await asyncio.sleep(0.5)
                if len(events) != last_count:
                    last_change = time.monotonic()
                    last_count = len(events)
                types_seen = {e.get("type") for e in events if isinstance(e, dict)}
                terminal = {"response.done", "response.audio_transcript.done", "session.done"}
                if time.monotonic() - last_change > 8.0 and (terminal & types_seen):
                    break
        finally:
            reader_task.cancel()
            with contextlib.suppress(BaseException):
                await reader_task

    types = [e.get("type") for e in events if isinstance(e, dict)]
    return True, [t for t in types if t]

async def run_rtc(target: str, audio_f32: np.ndarray, sr: int,
                   timeout_s: float) -> tuple[bool, list[str]]:
    from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
    sys.path.insert(0, str(Path(__file__).parent))
    from test_e2e import BufferAudioTrack, _try_parse_event

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    fragments: dict[str, dict[int, str]] = {}
    events: list[dict] = []
    chan_open = asyncio.Event()

    track = BufferAudioTrack(audio_f32, sr)
    pc.addTrack(track)
    channel = pc.createDataChannel("oai-events")

    @channel.on("open")
    def _opened() -> None:
        chan_open.set()

    @channel.on("message")
    def _on_message(message: str) -> None:
        ev = _try_parse_event(message, fragments)
        if ev is not None:
            events.append(ev)

    offer = await pc.createOffer()
    await pc.setLocalDescription(offer)
    url = (f"{target}/v1/realtime?model=eou-runner&intent=conversation"
            f"&transcription_model=deepdml/faster-whisper-large-v3-turbo-ct2"
            f"&voice=af_heart&speech_model=kokoro")
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers={"Content-Type": "application/sdp"},
                                  content=pc.localDescription.sdp)
        if resp.status_code != 200:
            await pc.close()
            raise RuntimeError(f"POST {url} -> {resp.status_code}: {resp.text}")
        answer_sdp = resp.text
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

    try:
        await asyncio.wait_for(chan_open.wait(), 10.0)
    except asyncio.TimeoutError:
        pass

    end_at = time.monotonic() + timeout_s
    last_change = time.monotonic()
    last_count = 0
    while time.monotonic() < end_at:
        await asyncio.sleep(0.5)
        if len(events) != last_count:
            last_change = time.monotonic()
            last_count = len(events)
        types_seen = {e.get("type") for e in events if isinstance(e, dict)}
        terminal = {"response.done", "response.audio_transcript.done", "session.done"}
        if time.monotonic() - last_change > 10.0 and (terminal & types_seen):
            break

    types = [e.get("type") for e in events if isinstance(e, dict)]
    await pc.close()
    return True, [t for t in types if t]

def build_env(*, transport: str, port: int, fake_url: str, inspect_dir: Path,
               kind: str, fusion_rule: str | None) -> dict:
    env_var = "INSPECT_SESSION_DIR" if transport == "rtc" else "SPEACHES_INSPECT_SESSION_DIR"
    env = {
        **os.environ,
        "UVICORN_HOST": "127.0.0.1",
        "UVICORN_PORT": str(port),
        "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", DEFAULT_HF_HUB_CACHE),
        "HF_HUB_OFFLINE": "1",
        "WARMUP_ALL_LOCAL_MODELS": "false",
        "LOG_LEVEL": "info",
        "RUST_LOG": "info",
        "CHAT_COMPLETION_BASE_URL": f"{fake_url}/v1",
        "CHAT_COMPLETION_API_KEY": "fake-key",
        env_var: str(inspect_dir),
        "SPEACHES_EOU_KIND": kind,
        "EOU_KIND": kind,
    }
    if kind in ("audio", "fusion"):
        env["EOU_AUDIO_MODEL_PATH"] = DEFAULT_SMART_TURN
    if kind == "fusion" and fusion_rule:
        env["EOU_FUSION_RULE"] = fusion_rule
    return env

async def run_one_combo(args, fx_path: Path, kind: str, fusion_rule: str | None,
                         fake_url: str) -> tuple[str, str, bool, str]:
    fx = fx_path.stem
    slot = kind if not (kind == "fusion" and fusion_rule and fusion_rule != "noisy_or") \
            else f"{kind}-{fusion_rule}"
    run_dir = Path(args.out) / slot / fx
    inspector_dir = run_dir / "inspector"
    inspector_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== {slot} x {fx} ===")

    audio_f32, sr = sf.read(str(fx_path), dtype="float32")
    if audio_f32.ndim > 1:
        audio_f32 = audio_f32[:, 0]

    port = free_port()
    env = build_env(transport=args.transport, port=port, fake_url=fake_url,
                     inspect_dir=inspector_dir, kind=kind, fusion_rule=fusion_rule)
    log_path = run_dir / "server.log"
    proc = spawn_server(args.binary, env, log_path)
    try:
        await wait_health(f"http://127.0.0.1:{port}/health", timeout_s=120)
    except Exception as exc:
        kill_proc(proc)
        return slot, fx, False, f"unhealthy: {exc}"

    target = f"http://127.0.0.1:{port}"
    try:
        if args.transport == "ws":
            ok, types = await run_ws(target, audio_f32, sr, args.timeout)
        else:
            ok, types = await run_rtc(target, audio_f32, sr, args.timeout)
        uniq, seen = [], set()
        for t in types:
            if t and t not in seen:
                uniq.append(t)
                seen.add(t)
        last = types[-1] if types else "(none)"
        note = f"{len(types)} events, last={last}"
        logger.info(f"  events ({len(types)}): {', '.join(uniq)}")
        return slot, fx, True, note
    except Exception as exc:
        logger.exception(f"  {slot} x {fx} failed")
        return slot, fx, False, str(exc)
    finally:
        kill_proc(proc)
        await asyncio.sleep(0.5)

async def matrix_async(args) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    for noisy in ("aiortc", "aioice", "httpx", "websockets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    wavs = sorted(Path(args.fixtures).glob("*.wav"))
    if not wavs:
        logger.error(f"no fixtures in {args.fixtures}")
        return 1
    if not Path(args.binary).exists():
        logger.error(f"binary missing: {args.binary}")
        return 1

    fake_port = free_port()
    fake = spawn_fake_llm(fake_port, out_root / "fake_llm.log")
    try:
        await wait_health(f"http://127.0.0.1:{fake_port}/health", 15.0)
    except Exception as exc:
        logger.error(f"fake_llm did not come up: {exc}")
        kill_proc(fake)
        return 1
    fake_url = f"http://127.0.0.1:{fake_port}"

    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    rules = [r.strip() for r in args.fusion_rules.split(",") if r.strip()] or ["noisy_or"]

    results: list[tuple[str, str, bool, str]] = []
    try:
        for wav in wavs:
            for kind in kinds:
                if kind == "fusion":
                    for rule in rules:
                        results.append(await run_one_combo(args, wav, kind, rule, fake_url))
                else:
                    results.append(await run_one_combo(args, wav, kind, None, fake_url))
    finally:
        kill_proc(fake)

    print("\n=== matrix summary ===")
    pass_n = fail_n = 0
    for slot, fx, ok, note in results:
        flag = "PASS" if ok else "FAIL"
        if ok:
            pass_n += 1
        else:
            fail_n += 1
        print(f"  [{flag}]  {slot:18s}  {fx:14s}  {note}")
    print(f"total: {pass_n} pass, {fail_n} fail")
    return 0 if fail_n == 0 else 1

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--binary", required=True,
                   help="path to a speaches-plus binary (Go or Rust)")
    p.add_argument("--transport", choices=("ws", "rtc"), required=True,
                   help="ws (Go binary) or rtc (Go or Rust)")
    p.add_argument("--fixtures", default="/tmp/eou-fixtures",
                   help="directory with *.wav fixtures (16 kHz mono)")
    p.add_argument("--kinds", default="vad,heuristic,audio,fusion",
                   help="comma-separated EOU kinds to run")
    p.add_argument("--fusion_rules", default="noisy_or",
                   help="comma-separated fusion rules (only used for kind=fusion)")
    p.add_argument("--out", default="/tmp/eou-runs",
                   help="output bundle root (per-(slot,fixture) inspector dir)")
    p.add_argument("--timeout", type=float, default=90.0,
                   help="per-(kind,fixture) wall-time cap in seconds")
    args = p.parse_args()
    sys.exit(asyncio.run(matrix_async(args)))

if __name__ == "__main__":
    main()
