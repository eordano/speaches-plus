#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "httpx>=0.28",
#   "numpy>=2.0",
#   "soundfile>=0.13",
# ]
# ///
"""eou_fixture -- render the canonical EOU stress fixtures by hitting a
speaches-plus binary's /v1/audio/speech endpoint, then concatenating the
returned audio with calibrated silence gaps.

Three fixtures:

  canonical (~21 s)   "I was thinking we could maybe go to the store later
                       this afternoon, what do you think." -> 5 s sil ->
                       "hmm." -> 5 s sil -> "yes, that's right." -> 3 s sil
  continuation (~11 s) "I think we should," -> 2.5 s pause ->
                       "go to the park if the weather holds, that sounds
                       nice."
  hesitation (~11 s)  "um," / "uh," / "so," with short gaps + 5 s tail

Output: 16 kHz mono WAV files. Linear-resample whatever sample rate the
TTS endpoint returns; the EOU pipeline runs at 16 kHz.

Replaces the older Go cmd/eou-fixture -- same outputs, just speaks to a
server over HTTP instead of pulling the Kokoro Python pipeline in-process.
The server can be either Go or Rust; both expose /v1/audio/speech.
"""
from __future__ import annotations

import argparse
import io
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

logger = logging.getLogger("fixture")

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port

def wait_health(url: str, timeout_s: float = 90.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code < 500:
                return
        except Exception:
            pass
        time.sleep(0.3)
    raise RuntimeError(f"server did not become healthy at {url} in {timeout_s}s")

def render_one(base_url: str, text: str, voice: str = "af_heart",
                model: str = "kokoro") -> tuple[np.ndarray, int]:
    """POST one TTS request; return (mono_f32, sample_rate)."""
    body = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "wav",
        "speed": 1.0,
    }
    r = httpx.post(f"{base_url}/v1/audio/speech", json=body, timeout=60.0)
    r.raise_for_status()
    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(sr)

def resample_to_16k(samples: np.ndarray, sr_in: int) -> np.ndarray:
    if sr_in == 16000:
        return samples
    duration = len(samples) / sr_in
    n_out = int(duration * 16000)
    x_in = np.arange(len(samples))
    x_out = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(x_out, x_in, samples).astype(np.float32)

def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * 16000), dtype=np.float32)

def render_fixture(base_url: str, kind: str, voice: str) -> np.ndarray:
    def fetch(text: str) -> np.ndarray:
        audio, sr = render_one(base_url, text, voice)
        return resample_to_16k(audio, sr)

    if kind == "canonical":
        opener = fetch("I was thinking we could maybe go to the store later "
                        "this afternoon, what do you think.")
        hmm = fetch("hmm.")
        yes = fetch("yes, that's right.")
        return np.concatenate([
            opener, silence(5.0), hmm, silence(5.0), yes, silence(3.0)
        ])
    if kind == "continuation":
        first = fetch("I think we should,")
        second = fetch("go to the park if the weather holds, that sounds nice.")
        return np.concatenate([
            silence(0.5), first, silence(2.5), second, silence(3.0)
        ])
    if kind == "hesitation":
        um = fetch("um,")
        uh = fetch("uh,")
        so = fetch("so,")
        return np.concatenate([
            silence(0.5), um, silence(1.0), uh, silence(1.0), so, silence(5.0)
        ])
    raise ValueError(f"unknown fixture kind: {kind}")

def write_wav(path: Path, samples: np.ndarray) -> None:
    int16 = np.clip(samples, -1.0, 1.0)
    int16 = (int16 * 32767.0).astype(np.int16)
    sf.write(path, int16, 16000, subtype="PCM_16")
    logger.info(f"  wrote {path} ({len(samples)} samples, "
                f"{len(samples) / 16000:.2f}s @ 16 kHz)")

def maybe_spawn_server(binary: str, log_path: Path) -> tuple[subprocess.Popen | None, str]:
    """If binary is set, spawn it on a free port and wait for /health. Returns
    (process, base_url). If binary is None or empty, returns (None, "")."""
    if not binary:
        return None, ""
    port = free_port()
    env = {
        **os.environ,
        "UVICORN_HOST": "127.0.0.1",
        "UVICORN_PORT": str(port),
        "WARMUP_ALL_LOCAL_MODELS": "false",
        "LOG_LEVEL": "info",
        "RUST_LOG": "info",
    }
    fh = log_path.open("w")
    proc = subprocess.Popen([binary], env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
    base = f"http://127.0.0.1:{port}"
    try:
        wait_health(f"{base}/health")
    except Exception:
        proc.kill()
        raise
    return proc, base

def kill_proc(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="/tmp/eou-fixtures",
                   help="output directory for canonical/continuation/hesitation.wav")
    p.add_argument("--one", choices=("canonical", "continuation", "hesitation", "all"),
                   default="all")
    p.add_argument("--voice", default="af_heart", help="Kokoro voice id")
    p.add_argument("--server-url", default=os.environ.get("SPEACHES_PLUS_URL", ""),
                   help="base URL of a running speaches-plus server "
                        "(env: SPEACHES_PLUS_URL). If unset and --binary given, "
                        "this script spawns + tears down a server itself.")
    p.add_argument("--binary", default="",
                   help="path to a speaches-plus binary; spawned if --server-url empty")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = args.server_url
    proc = None
    if not base_url:
        if not args.binary:
            logger.error("no --server-url and no --binary supplied")
            return 2
        proc, base_url = maybe_spawn_server(args.binary, out_dir / "server.log")
        logger.info(f"spawned server at {base_url}")

    try:
        kinds = ["canonical", "continuation", "hesitation"] if args.one == "all" else [args.one]
        for k in kinds:
            logger.info(f"rendering {k} ...")
            samples = render_fixture(base_url, k, args.voice)
            write_wav(out_dir / f"{k}.wav", samples)
    finally:
        kill_proc(proc)

    return 0

if __name__ == "__main__":
    sys.exit(main())
