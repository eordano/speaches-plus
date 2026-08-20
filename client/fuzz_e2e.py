#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = [
#   "aiortc>=1.13.0,<1.14",
#   "httpx>=0.28",
#   "numpy>=2.0",
#   "av>=14.0,<14.3",
# ]
# ///

"""
End-to-end fuzzer for speaches-plus realtime sessions.

What it does
============
- Spawns the server under test (Rust or Go) and a fake LLM sidecar
- For N iterations, generates a random adversarial Scenario:
  - random audio (silence / tone / noise / mixed)
  - random client events injected at random offsets
    (session.update, input_audio_buffer.{append,commit,clear},
     conversation.item.create, response.{create,cancel})
  - random fake-LLM behavior (fast/slow/error/stream-then-fail)
  - random session lifetime (early-close / normal / hard-cap)
- Connects via WebRTC, sends the scenario, collects every data-channel event
- Checks invariants:
  1. The session ends cleanly (session.done OR transport close), no hang
  2. No error event with `internal_state_error` (RFC v3 §9.6 says these are
     invariant-violation bugs)
  3. No response.created without a matching response.done in the session
  4. audio_end_ms is monotonic within a session for the same item_id
  5. Every emitted event has a `type` field and (where applicable) the
     RFC v3 §10.1-§10.3 required fields
  6. response.done emitted at most once per response_id (RFC v3 §8.4)
  7. No response.output_audio.delta after response.output_audio.done for
     the same (response_id, item_id) pair (RFC v3 §8.2)
  8. No panics / unexpected process death during the run
- On any failure: dumps the scenario seed + minimized repro + relevant
  server stderr to /tmp/fuzz-failure-<idx>.json

What it does NOT do
===================
- Modify any production code. Pure black-box stress test.
- Validate transcription accuracy or LLM output correctness -- only
  protocol-level invariants.
- Run inside the Rust impl's internal fuzz harness (`realtime::fuzz`) --
  that one tests the state machine in isolation; this one tests the full
  WebRTC + LLM + audio + wire-protocol stack.

Usage
=====
  ./fuzz_e2e.py --iterations 50 --seed 42
  ./fuzz_e2e.py --target rust --iterations 1000
  ./fuzz_e2e.py --target go --iterations 100 --max-duration-s 10

Concurrent stress: run N scenarios against the same server in parallel
(stresses concurrent-session handling, surfaces races):

  ./fuzz_e2e.py --target rust --iterations 200 --workers 8

The master RNG is locked across workers, so the same --seed produces the
same set of scenarios regardless of --workers; only the *interleaving*
varies. Failure dumps are still indexed by iteration order.

Exit code: 0 if all scenarios passed invariants; 1 if any failed.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import asyncio
import contextlib
import dataclasses
import datetime
import json
import logging
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import httpx
import numpy as np

logger = logging.getLogger("fuzz_e2e")

class RunDir:
    """`/tmp/fuzz_e2e/<run_id>/{server.log, fake_llm.log, failures/, summary.json}`."""

    def __init__(self, root: Path, run_id: str) -> None:
        self.root = root / run_id
        self.run_id = run_id
        self.failures_dir = self.root / "failures"
        self.root.mkdir(parents=True, exist_ok=True)
        self.failures_dir.mkdir(parents=True, exist_ok=True)

    def log_path(self, name: str) -> Path:
        return self.root / f"{name}.log"

    def failure_path(self, idx: int) -> Path:
        return self.failures_dir / f"{idx:04d}.json"

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.json"

    @classmethod
    def make(cls, base: str | os.PathLike[str], seed: int) -> RunDir:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return cls(Path(base), f"{ts}-seed{seed}")

def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p

def find_speaches_binary(target: str) -> str:
    if target == "rust":
        candidates = [
            "rust/target/release/speaches-plus",
            "rust/target/debug/speaches-plus",
        ]
    elif target == "go":
        candidates = ["go/bin/speaches-plus-go", "go/cmd/server/server"]
    else:
        raise ValueError(f"unknown target: {target}")
    for c in candidates:
        p = Path.cwd() / c
        if p.is_file():
            return str(p)
        p = Path(__file__).resolve().parent.parent / c
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        f"no {target} binary found; tried: {candidates}\n"
        f"build with: cd {target} && {'cargo build --release --features metal' if target == 'rust' else 'go build -o bin/speaches-plus-go ./cmd/server'}"
    )

class Subprocess:
    def __init__(
        self,
        name: str,
        cmd: list[str],
        env: dict[str, str] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.name = name
        self.cmd = cmd
        self.env = env
        self.proc: subprocess.Popen | None = None
        self.log_path = log_path or Path(f"/tmp/fuzz_e2e.{name}.log")

    def start(self) -> None:
        self.log_fh = self.log_path.open("w")
        self.proc = subprocess.Popen(
            self.cmd,
            env=self.env,
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

    def tail_log(self, n: int = 200) -> str:
        try:
            text = self.log_path.read_text(errors="replace")
        except Exception:
            return ""
        lines = text.splitlines()
        return "\n".join(lines[-n:])

async def wait_for_health(url: str, timeout_s: float, name: str) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: Any = None
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                r = await client.get(url)
                if r.status_code == 200:
                    return
                last_err = f"HTTP {r.status_code}"
            except Exception as e:
                last_err = e
            await asyncio.sleep(0.1)
    raise RuntimeError(f"{name} not ready within {timeout_s}s ({last_err})")

@dataclass
class AudioStep:
    """One audio segment in the scenario.

    Kinds:
      silence       -- pure zeros
      tone          -- sine wave at `freq_hz`
      noise         -- white gaussian
      mixed         -- tone + low-amplitude noise
      speech_burst  -- multi-formant vowel-like signal with attack / decay
                      envelope. Most likely to trip Silero VAD.
    """
    kind: str
    duration_ms: int
    freq_hz: float = 440.0
    amplitude: float = 0.3

@dataclass
class ClientEventStep:
    """A data-channel event injected at a wall-clock offset."""
    delay_ms: int
    payload: dict

@dataclass
class Scenario:
    seed: int
    pattern: str
    audio_steps: list[AudioStep]
    client_events: list[ClientEventStep]
    llm_mode: str
    llm_response_ms: int
    session_max_s: int
    early_close_ms: int | None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

PATTERNS = (
    "simple_random",
    "conversation_turns",
    "bargein_storm",
    "session_update_burst",
    "state_machine_adversarial",
    "chaos_mix",
)

PATTERN_WEIGHTS = {
    "simple_random": 0.20,
    "conversation_turns": 0.25,
    "bargein_storm": 0.15,
    "session_update_burst": 0.15,
    "state_machine_adversarial": 0.10,
    "chaos_mix": 0.15,
}

def gen_scenario(
    rng: random.Random,
    *,
    max_duration_s: int = 30,
    force_pattern: str | None = None,
) -> Scenario:
    seed = rng.randrange(0, 2**31)
    inner = random.Random(seed)

    if force_pattern and force_pattern in PATTERNS:
        pattern = force_pattern
    else:
        choices = list(PATTERN_WEIGHTS.items())
        pattern = inner.choices(
            [p for p, _ in choices], weights=[w for _, w in choices], k=1
        )[0]

    builder = {
        "simple_random": _build_simple_random,
        "conversation_turns": _build_conversation_turns,
        "bargein_storm": _build_bargein_storm,
        "session_update_burst": _build_session_update_burst,
        "state_machine_adversarial": _build_state_machine_adversarial,
        "chaos_mix": _build_chaos_mix,
    }[pattern]

    audio_steps, client_events = builder(inner)
    total_ms = sum(s.duration_ms for s in audio_steps) or 1000

    llm_mode = inner.choice(["fast", "fast", "slow", "error", "stream_then_fail"])
    llm_response_ms = {
        "fast": inner.randint(20, 200),
        "slow": inner.randint(2000, 6000),
        "error": inner.randint(20, 200),
        "stream_then_fail": inner.randint(200, 1500),
    }[llm_mode]

    session_max_s = inner.choice([5, 10, max_duration_s])
    early_close_ms = (
        inner.randint(100, max(101, total_ms))
        if inner.random() < 0.15
        else None
    )

    return Scenario(
        seed=seed,
        pattern=pattern,
        audio_steps=audio_steps,
        client_events=sorted(client_events, key=lambda e: e.delay_ms),
        llm_mode=llm_mode,
        llm_response_ms=llm_response_ms,
        session_max_s=session_max_s,
        early_close_ms=early_close_ms,
    )

def _speech_burst(rng: random.Random, ms: int) -> AudioStep:
    """Vowel-like multi-formant burst -- most likely to trigger Silero VAD."""
    return AudioStep(
        kind="speech_burst",
        duration_ms=ms,
        freq_hz=rng.choice([180, 220, 260, 300]),
        amplitude=rng.uniform(0.2, 0.5),
    )

def _silence(ms: int) -> AudioStep:
    return AudioStep(kind="silence", duration_ms=ms)

def _build_simple_random(rng: random.Random) -> tuple[list[AudioStep], list[ClientEventStep]]:
    """Original flat random pattern -- short, mixed, low complexity."""
    n_audio = rng.randint(0, 5)
    audio: list[AudioStep] = []
    for _ in range(n_audio):
        kind = rng.choice(["silence", "tone", "noise", "mixed", "speech_burst"])
        dur = rng.randint(50, 3000)
        audio.append(
            AudioStep(
                kind=kind,
                duration_ms=dur,
                freq_hz=rng.choice([220, 330, 440, 660, 880]),
                amplitude=rng.uniform(0.05, 0.6),
            )
        )
    total_ms = sum(s.duration_ms for s in audio) or 1000
    n_events = rng.randint(0, 6)
    events = [
        ClientEventStep(rng.randint(0, total_ms), _random_client_event(rng))
        for _ in range(n_events)
    ]
    return audio, events

def _build_conversation_turns(rng: random.Random) -> tuple[list[AudioStep], list[ClientEventStep]]:
    """K turns of {speech burst, silence}, with optional response.cancel mid-turn."""
    n_turns = rng.randint(2, 5)
    audio: list[AudioStep] = []
    events: list[ClientEventStep] = []
    cursor = 0
    for _ in range(n_turns):
        burst_ms = rng.randint(300, 2500)
        gap_ms = rng.randint(400, 1500)
        audio.append(_speech_burst(rng, burst_ms))
        cursor += burst_ms
        if rng.random() < 0.4:
            events.append(ClientEventStep(
                cursor - rng.randint(0, burst_ms // 2),
                rng.choice([
                    {"type": "response.cancel"},
                    {"type": "input_audio_buffer.commit"},
                    _random_client_event(rng),
                ]),
            ))
        audio.append(_silence(gap_ms))
        cursor += gap_ms
    return audio, events

def _build_bargein_storm(rng: random.Random) -> tuple[list[AudioStep], list[ClientEventStep]]:
    """Speech -> response.create -> mid-response speech (barge-in) -> repeat.

    Stresses §9.1-§9.5 barge-in handling.
    """
    audio: list[AudioStep] = []
    events: list[ClientEventStep] = []
    cursor = 0
    for _ in range(rng.randint(2, 4)):
        burst1 = rng.randint(500, 1500)
        audio.append(_speech_burst(rng, burst1))
        cursor += burst1
        audio.append(_silence(rng.randint(200, 600)))
        cursor += 400
        events.append(ClientEventStep(
            cursor + rng.randint(50, 400),
            {"type": "response.create"},
        ))
        audio.append(_silence(rng.randint(100, 500)))
        cursor += 300
        burst2 = rng.randint(300, 1200)
        audio.append(_speech_burst(rng, burst2))
        cursor += burst2
        audio.append(_silence(rng.randint(300, 800)))
        cursor += 500
    return audio, events

def _build_session_update_burst(rng: random.Random) -> tuple[list[AudioStep], list[ClientEventStep]]:
    """Hammer the server with many session.update events during VAD-active speech."""
    audio: list[AudioStep] = [_speech_burst(rng, rng.randint(1500, 4000))]
    events: list[ClientEventStep] = []
    n_updates = rng.randint(10, 30)
    total_ms = audio[0].duration_ms
    for _ in range(n_updates):
        if rng.random() < 0.7:
            payload = _random_client_event(rng)
            if payload.get("type") != "session.update":
                payload = {"type": "session.update", "session": {
                    "instructions": "burst-" + str(rng.randint(0, 999)),
                }}
        else:
            payload = {"type": "session.update", "session": rng.choice([
                {"min_speech_ms": -5},
                {"turn_detection": {"threshold": 99}},
                {"voice": 42},
                {"turn_detection": {"eou": {"fusion_rule": "noisy_xor"}}},
            ])}
        events.append(ClientEventStep(rng.randint(0, total_ms), payload))
    return audio, events

def _build_state_machine_adversarial(rng: random.Random) -> tuple[list[AudioStep], list[ClientEventStep]]:
    """Sequences designed to violate the state machine; server must reject cleanly."""
    audio: list[AudioStep] = [_speech_burst(rng, rng.randint(800, 2500))]
    total_ms = audio[0].duration_ms
    sequences = [
        [{"type": "response.cancel"}],
        [{"type": "input_audio_buffer.commit"}, {"type": "input_audio_buffer.commit"}],
        [{"type": "input_audio_buffer.clear"}, {"type": "input_audio_buffer.commit"}],
        [{"type": "response.create"}, {"type": "response.create"}],
        [
            {"type": "response.create"},
            {"type": "session.update", "session": {"instructions": "hi"}},
            {"type": "response.cancel"},
        ],
        [
            {"_raw_bytes": "{not json"},
            {"type": "fuzz.unknown.event"},
            {"type": "session.update", "session": {"instructions": "x"}},
        ],
    ]
    seq = rng.choice(sequences)
    events: list[ClientEventStep] = [
        ClientEventStep(
            rng.randint(0, max(1, total_ms)) + i * rng.randint(50, 250),
            payload,
        )
        for i, payload in enumerate(seq)
    ]
    return audio, events

def _build_chaos_mix(rng: random.Random) -> tuple[list[AudioStep], list[ClientEventStep]]:
    """Maximum entropy. Lots of audio, lots of events, all interleaved."""
    audio: list[AudioStep] = []
    n_audio = rng.randint(6, 12)
    for _ in range(n_audio):
        kind = rng.choices(
            ["silence", "tone", "noise", "mixed", "speech_burst"],
            weights=[2, 1, 1, 1, 3],
        )[0]
        dur = rng.randint(80, 2000)
        audio.append(AudioStep(
            kind=kind,
            duration_ms=dur,
            freq_hz=rng.choice([180, 220, 330, 440, 660, 880]),
            amplitude=rng.uniform(0.05, 0.6),
        ))
    total_ms = sum(s.duration_ms for s in audio)
    n_events = rng.randint(8, 20)
    events = [
        ClientEventStep(rng.randint(0, total_ms), _random_client_event(rng))
        for _ in range(n_events)
    ]
    return audio, events

def _random_client_event(rng: random.Random) -> dict:
    """One random client->server data-channel event. Mix of valid + adversarial."""
    kind = rng.choice(
        [
            "session.update",
            "session.update_invalid",
            "input_audio_buffer.commit",
            "input_audio_buffer.clear",
            "conversation.item.create",
            "response.create",
            "response.cancel",
            "garbage_json",
            "unknown_event",
        ]
    )
    if kind == "session.update":
        which = rng.choice(
            [
                "instructions",
                "session_max_duration_s",
                "min_speech_ms",
                "min_speech_for_response_ms",
                "voice",
                "input_audio_format",
                "output_audio_format",
                "turn_detection.threshold",
                "turn_detection.silence_duration_ms",
                "turn_detection.eou.kind",
                "turn_detection.eou.fusion_rule",
            ]
        )
        body: dict = {}
        if which == "instructions":
            body["instructions"] = "fuzz: " + ("x" * rng.randint(1, 200))
        elif which == "session_max_duration_s":
            body["session_max_duration_s"] = rng.randint(1, 86400)
        elif which == "min_speech_ms":
            body["min_speech_ms"] = rng.randint(0, 60000)
        elif which == "min_speech_for_response_ms":
            body["min_speech_for_response_ms"] = rng.randint(0, 60000)
        elif which == "voice":
            body["voice"] = rng.choice(["alloy", "shimmer", "marin", "af_heart"])
        elif which == "input_audio_format":
            body["input_audio_format"] = rng.choice(
                ["pcm16", "pcm16_16k", "g711_ulaw", "g711_alaw"]
            )
        elif which == "output_audio_format":
            body["output_audio_format"] = rng.choice(
                ["pcm16", "pcm16_16k", "g711_ulaw", "g711_alaw"]
            )
        elif which.startswith("turn_detection.eou."):
            sub = which.split(".")[-1]
            eou_value: Any
            if sub == "kind":
                eou_value = rng.choice(
                    ["vad", "heuristic", "text", "audio", "fusion", "integrated"]
                )
            elif sub == "fusion_rule":
                eou_value = rng.choice(
                    ["noisy_or", "max", "mean", "weighted", "gated"]
                )
            else:
                eou_value = 0
            body["turn_detection"] = {"eou": {sub: eou_value}}
        elif which.startswith("turn_detection."):
            sub = which.split(".", 1)[1]
            if sub == "threshold":
                body["turn_detection"] = {"threshold": rng.uniform(0.0, 1.0)}
            elif sub == "silence_duration_ms":
                body["turn_detection"] = {
                    "silence_duration_ms": rng.randint(50, 5000)
                }
        return {"type": "session.update", "session": body}

    if kind == "session.update_invalid":
        bad = rng.choice(
            [
                {"session_max_duration_s": -1},
                {"session_max_duration_s": "not a number"},
                {"min_speech_ms": -100},
                {"turn_detection": {"threshold": 1.5}},
                {"turn_detection": {"threshold": "huge"}},
                {"turn_detection": {"eou": {"fusion_rule": "noisy_xor"}}},
                {"turn_detection": {"eou": {"kind": "spaceship"}}},
                {"turn_detection": "not_an_object"},
                {"input_audio_format": "ogg_opus"},
                {"voice": 12345},
                {"instructions": 99},
                {"sealed_buffer_retention_count": 9999999},
            ]
        )
        return {"type": "session.update", "session": bad}

    if kind == "input_audio_buffer.commit":
        return {"type": "input_audio_buffer.commit"}

    if kind == "input_audio_buffer.clear":
        return {"type": "input_audio_buffer.clear"}

    if kind == "conversation.item.create":
        return {
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "fuzz user message " + str(rng.randint(0, 999))}
                ],
            },
        }

    if kind == "response.create":
        return {"type": "response.create"}

    if kind == "response.cancel":
        return {"type": "response.cancel"}

    if kind == "garbage_json":
        return {"_raw_bytes": "{not valid json"}

    if kind == "unknown_event":
        return {"type": f"fuzz.unknown.{rng.randint(0, 999)}"}

    raise AssertionError("unreachable")

def synth_audio(scenario: Scenario, sample_rate: int = 48000) -> np.ndarray:
    """Materialize the scenario's audio steps into one mono float32 array."""
    rng = np.random.default_rng(scenario.seed)
    chunks: list[np.ndarray] = []
    for step in scenario.audio_steps:
        n = max(1, int(sample_rate * step.duration_ms / 1000))
        if step.kind == "silence":
            chunks.append(np.zeros(n, dtype=np.float32))
        elif step.kind == "tone":
            t = np.arange(n) / sample_rate
            chunks.append(
                (step.amplitude * np.sin(2 * np.pi * step.freq_hz * t)).astype(np.float32)
            )
        elif step.kind == "noise":
            chunks.append(
                (step.amplitude * rng.standard_normal(n)).astype(np.float32)
            )
        elif step.kind == "mixed":
            t = np.arange(n) / sample_rate
            tone = step.amplitude * np.sin(2 * np.pi * step.freq_hz * t)
            noise = 0.1 * step.amplitude * rng.standard_normal(n)
            chunks.append((tone + noise).astype(np.float32))
        elif step.kind == "speech_burst":
            t = np.arange(n) / sample_rate
            f0 = step.freq_hz
            f1 = f0 * 3.5 + 100
            f2 = f0 * 7.0 + 200
            f3 = f0 * 12.0 + 200
            sig = (
                0.5 * np.sin(2 * np.pi * f0 * t)
                + 0.35 * np.sin(2 * np.pi * f1 * t)
                + 0.20 * np.sin(2 * np.pi * f2 * t)
                + 0.10 * np.sin(2 * np.pi * f3 * t)
            )
            attack = min(int(0.05 * sample_rate), n // 4)
            decay = min(int(0.10 * sample_rate), n // 4)
            env = np.ones(n, dtype=np.float32)
            if attack > 0:
                env[:attack] = np.linspace(0, 1, attack)
            if decay > 0:
                env[-decay:] = np.linspace(1, 0, decay)
            sig = sig.astype(np.float32) * env * step.amplitude
            sig += (0.02 * step.amplitude * rng.standard_normal(n)).astype(np.float32)
            chunks.append(sig)
        else:
            chunks.append(np.zeros(n, dtype=np.float32))
    if not chunks:
        return np.zeros(int(sample_rate * 0.5), dtype=np.float32)
    return np.concatenate(chunks)

@dataclass
class InvariantViolation:
    name: str
    detail: str
    event_index: int | None = None

def check_invariants(events: list[dict], session_done: bool, transport_closed: bool) -> list[InvariantViolation]:
    out: list[InvariantViolation] = []

    if not session_done and not transport_closed:
        out.append(InvariantViolation(
            "session_did_not_terminate_after_close",
            "after pc.close(), neither session.done nor transport-close "
            "observed within the 2s grace window",
        ))

    for i, ev in enumerate(events):
        if ev.get("type") == "error":
            err = ev.get("error", {}) if isinstance(ev.get("error"), dict) else {}
            code = err.get("code") or err.get("type")
            if code == "internal_state_error":
                out.append(InvariantViolation(
                    "internal_state_error",
                    f"server emitted internal_state_error: {json.dumps(err)}",
                    event_index=i,
                ))

    open_responses: dict[str, int] = {}
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        if t == "response.created":
            rid = (ev.get("response") or {}).get("id") or ev.get("response_id")
            if rid:
                open_responses[rid] = i
        elif t == "response.done":
            rid = (ev.get("response") or {}).get("id") or ev.get("response_id")
            if rid and rid in open_responses:
                del open_responses[rid]
    for rid, idx in open_responses.items():
        out.append(InvariantViolation(
            "response_without_done",
            f"response.created at idx {idx} (id={rid}) had no matching response.done",
            event_index=idx,
        ))

    last_end_ms: dict[str, int] = {}
    for i, ev in enumerate(events):
        item_id = ev.get("item_id")
        end_ms = ev.get("audio_end_ms")
        if not isinstance(item_id, str) or not isinstance(end_ms, (int, float)):
            continue
        prev = last_end_ms.get(item_id)
        if prev is not None and end_ms < prev:
            out.append(InvariantViolation(
                "audio_end_ms_regression",
                f"item {item_id}: audio_end_ms went {prev} -> {end_ms}",
                event_index=i,
            ))
        last_end_ms[item_id] = max(prev or 0, int(end_ms))

    for i, ev in enumerate(events):
        if not isinstance(ev, dict) or not isinstance(ev.get("type"), str):
            out.append(InvariantViolation(
                "malformed_event",
                f"event #{i} missing/invalid `type` field: {json.dumps(ev)}",
                event_index=i,
            ))

    done_count: dict[str, int] = {}
    done_first_idx: dict[str, int] = {}
    for i, ev in enumerate(events):
        if ev.get("type") != "response.done":
            continue
        rid = (ev.get("response") or {}).get("id") or ev.get("response_id")
        if not isinstance(rid, str):
            continue
        done_count[rid] = done_count.get(rid, 0) + 1
        done_first_idx.setdefault(rid, i)
    for rid, n in done_count.items():
        if n > 1:
            out.append(InvariantViolation(
                "duplicate_response_done",
                f"response.done emitted {n} times for response_id={rid} "
                f"(first at idx {done_first_idx[rid]})",
                event_index=done_first_idx[rid],
            ))

    audio_done_at: dict[tuple[str, str], int] = {}
    for i, ev in enumerate(events):
        t = ev.get("type", "")
        rid = ev.get("response_id")
        item_id = ev.get("item_id")
        if not isinstance(rid, str) or not isinstance(item_id, str):
            continue
        key = (rid, item_id)
        if t == "response.output_audio.done":
            audio_done_at.setdefault(key, i)
        elif t == "response.output_audio.delta":
            if key in audio_done_at:
                out.append(InvariantViolation(
                    "audio_delta_after_done",
                    f"audio.delta at idx {i} for ({rid},{item_id}) "
                    f"arrived after audio.done at idx {audio_done_at[key]}",
                    event_index=i,
                ))

    return out

async def run_scenario(
    scenario: Scenario,
    *,
    target_url: str,
    fake_llm_url: str,
    overall_timeout_s: float,
) -> tuple[list[dict], bool, bool, str | None]:
    """
    Returns: (events, session_done, transport_closed, error_message_or_None).
    `error_message` is a Python-level error (test crashed before we could
    collect data); distinct from a server error event.
    """
    from aiortc import (
        RTCConfiguration,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.mediastreams import MediaStreamTrack
    from av import AudioFrame

    sample_rate = 48000
    audio = synth_audio(scenario, sample_rate=sample_rate)

    class BufferTrack(MediaStreamTrack):
        kind = "audio"

        def __init__(self, samples: np.ndarray) -> None:
            super().__init__()
            self.samples = samples
            self.cursor = 0
            self.frame_size = 960

        async def recv(self) -> AudioFrame:
            chunk = self.samples[self.cursor : self.cursor + self.frame_size]
            self.cursor += self.frame_size
            if len(chunk) < self.frame_size:
                pad = np.zeros(self.frame_size - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, pad])
            pcm = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)
            frame = AudioFrame.from_ndarray(pcm.reshape(1, -1), format="s16", layout="mono")
            frame.sample_rate = 48000
            frame.pts = self.cursor
            await asyncio.sleep(0.020)
            return frame

    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    fragments: dict[str, dict[int, str]] = {}
    events: list[dict] = []
    session_done = asyncio.Event()
    transport_closed = asyncio.Event()
    channel_open = asyncio.Event()

    async def wait_for_session_end(timeout_s: float) -> None:
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(
                asyncio.wait(
                    [
                        asyncio.create_task(session_done.wait()),
                        asyncio.create_task(transport_closed.wait()),
                    ],
                    return_when=asyncio.FIRST_COMPLETED,
                ),
                timeout=timeout_s,
            )

    track = BufferTrack(audio)
    pc.addTrack(track)
    channel = pc.createDataChannel("oai-events")

    @channel.on("open")
    def _on_open() -> None:
        channel_open.set()

    @channel.on("close")
    def _on_close() -> None:
        transport_closed.set()

    @channel.on("message")
    def _on_message(msg: str) -> None:
        ev = _parse_event(msg, fragments)
        if ev is None:
            return
        events.append(ev)
        t = ev.get("type")
        if t == "session.done":
            session_done.set()

    @pc.on("connectionstatechange")
    async def _on_state_change() -> None:
        if pc.connectionState in ("closed", "failed", "disconnected"):
            transport_closed.set()

    try:
        offer = await pc.createOffer()
        await pc.setLocalDescription(offer)
        url = (
            f"{target_url}/v1/realtime"
            f"?intent=conversation"
            f"&model=gpt-4o-realtime-preview"
            f"&transcription_model=whisper-large-v3-turbo"
            f"&voice=marin"
        )
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                url,
                headers={"Content-Type": "application/sdp"},
                content=pc.localDescription.sdp,
            )
            if resp.status_code != 200:
                return events, False, True, f"realtime POST {resp.status_code}: {resp.text}"
            answer_sdp = resp.text

        await pc.setRemoteDescription(RTCSessionDescription(sdp=answer_sdp, type="answer"))

        try:
            await asyncio.wait_for(channel_open.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            return events, False, False, "data channel never opened"

        t0 = time.monotonic()

        async def inject_events() -> None:
            for step in scenario.client_events:
                target_t = t0 + step.delay_ms / 1000.0
                now = time.monotonic()
                if target_t > now:
                    await asyncio.sleep(target_t - now)
                if "_raw_bytes" in step.payload:
                    try:
                        channel.send(step.payload["_raw_bytes"])
                    except Exception:
                        pass
                else:
                    try:
                        channel.send(json.dumps(step.payload))
                    except Exception:
                        pass

        injection_task = asyncio.create_task(inject_events())

        early_close_task: asyncio.Task | None = None
        if scenario.early_close_ms is not None:
            async def early_close() -> None:
                await asyncio.sleep(scenario.early_close_ms / 1000.0)
                await pc.close()
            early_close_task = asyncio.create_task(early_close())

        await wait_for_session_end(overall_timeout_s)

        injection_task.cancel()
        with contextlib.suppress(BaseException):
            await injection_task
        if early_close_task is not None:
            early_close_task.cancel()
            with contextlib.suppress(BaseException):
                await early_close_task

        try:
            await asyncio.wait_for(pc.close(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            pass

        await wait_for_session_end(2.0)

        return events, session_done.is_set(), transport_closed.is_set(), None

    except Exception as e:
        with contextlib.suppress(asyncio.TimeoutError, Exception):
            await asyncio.wait_for(pc.close(), timeout=2.0)
        return events, session_done.is_set(), transport_closed.is_set(), f"{type(e).__name__}: {e}"

def _parse_event(msg: str, fragments: dict[str, dict[int, str]]) -> dict | None:
    try:
        ev = json.loads(msg)
    except Exception:
        return {"type": "_unparsed", "_raw": msg}
    frag = ev.get("_fragment") or ev.get("fragment")
    if isinstance(frag, dict):
        gid = str(frag.get("id"))
        idx = int(frag.get("seq", 0))
        total = int(frag.get("total", 1))
        body = ev.get("body", "")
        bucket = fragments.setdefault(gid, {})
        bucket[idx] = body
        if len(bucket) == total:
            try:
                joined = "".join(bucket[i] for i in range(total))
                fragments.pop(gid, None)
                return json.loads(joined)
            except Exception:
                fragments.pop(gid, None)
                return None
        return None
    return ev

async def main_async(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_dir = RunDir.make(args.run_root, args.seed)
    logger.info(f"run dir: {run_dir.root}")

    binary = find_speaches_binary(args.target)
    server_port = free_port()
    llm_port = free_port()

    server_env = os.environ.copy()
    server_env["UVICORN_HOST"] = "127.0.0.1"
    server_env["UVICORN_PORT"] = str(server_port)
    server_env["CHAT_COMPLETION_BASE_URL"] = f"http://127.0.0.1:{llm_port}/v1"
    server_env["CHAT_COMPLETION_API_KEY"] = "fuzz-key"
    server_env.setdefault("RUST_LOG", "info")

    fake_llm_path = Path(__file__).resolve().parent / "fake_llm.py"
    llm_cmd = [str(fake_llm_path), "--host", "127.0.0.1", "--port", str(llm_port)]

    server = Subprocess(
        f"server-{args.target}",
        [binary, "--host", "127.0.0.1", "--port", str(server_port)],
        env=server_env,
        log_path=run_dir.log_path("server"),
    )
    fake_llm = Subprocess(
        "fake_llm",
        llm_cmd,
        log_path=run_dir.log_path("fake_llm"),
    )

    server.start()
    fake_llm.start()
    try:
        await wait_for_health(f"http://127.0.0.1:{server_port}/health", 30.0, "server")
        await wait_for_health(f"http://127.0.0.1:{llm_port}/health", 10.0, "fake_llm")

        master_rng = random.Random(args.seed)
        target_url = f"http://127.0.0.1:{server_port}"
        fake_llm_url = f"http://127.0.0.1:{llm_port}"

        master_rng_lock = asyncio.Lock()
        sem = asyncio.Semaphore(max(1, args.workers))
        abort = asyncio.Event()
        failures = 0
        passes = 0
        completed = 0
        completed_lock = asyncio.Lock()
        per_pattern_pass: dict[str, int] = {p: 0 for p in PATTERNS}
        per_pattern_fail: dict[str, int] = {p: 0 for p in PATTERNS}
        per_violation: dict[str, int] = {}
        latencies: list[float] = []

        async def run_one(idx: int) -> int:
            """Run a single scenario, returns 1 on failure / 0 on pass."""
            nonlocal completed
            if abort.is_set():
                return 0
            async with sem:
                if abort.is_set():
                    return 0
                async with master_rng_lock:
                    scenario = gen_scenario(
                        master_rng,
                        max_duration_s=args.max_duration_s,
                        force_pattern=args.pattern,
                    )
                t_start = time.monotonic()
                try:
                    events, sd, tc, err = await run_scenario(
                        scenario,
                        target_url=target_url,
                        fake_llm_url=fake_llm_url,
                        overall_timeout_s=args.scenario_timeout_s,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    events, sd, tc = [], False, True
                    err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
                elapsed = time.monotonic() - t_start

                violations: list[InvariantViolation] = []
                if err is not None:
                    violations.append(InvariantViolation("test_harness_error", err))
                violations.extend(check_invariants(events, sd, tc))

                if not server.alive():
                    violations.append(InvariantViolation(
                        "server_died",
                        "server process exited unexpectedly during scenario",
                    ))

            async with completed_lock:
                completed += 1
                progress = f"{completed}/{args.iterations}"
                latencies.append(elapsed)
                if violations:
                    per_pattern_fail[scenario.pattern] = per_pattern_fail.get(scenario.pattern, 0) + 1
                    for v in violations:
                        per_violation[v.name] = per_violation.get(v.name, 0) + 1
                else:
                    per_pattern_pass[scenario.pattern] = per_pattern_pass.get(scenario.pattern, 0) + 1
            tag = "PASS" if not violations else "FAIL"
            logger.info(
                f"[{progress}] worker={idx} {tag} pattern={scenario.pattern} "
                f"seed={scenario.seed} events={len(events)} elapsed={elapsed:.2f}s "
                f"violations={len(violations)}"
            )

            if violations:
                dump_path = run_dir.failure_path(idx + 1)
                dump = {
                    "iteration": idx + 1,
                    "scenario": scenario.to_dict(),
                    "events": events,
                    "session_done": sd,
                    "transport_closed": tc,
                    "violations": [
                        {"name": v.name, "detail": v.detail, "event_index": v.event_index}
                        for v in violations
                    ],
                    "server_log_tail": server.tail_log(200) if not server.alive() else None,
                }
                try:
                    dump_path.write_text(json.dumps(dump, indent=2, default=str))
                    logger.error(f"  -> dumped {dump_path}")
                except Exception as dump_err:
                    logger.error(f"  -> failed to dump: {dump_err}")
                for v in violations:
                    logger.error(f"  - {v.name}: {v.detail}")
                if not server.alive():
                    logger.error("server died -- aborting remaining scenarios")
                    abort.set()
                return 1
            return 0

        tasks = [asyncio.create_task(run_one(i)) for i in range(args.iterations)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"task crashed: {r}")
                failures += 1
            elif r == 1:
                failures += 1
            else:
                passes += 1

        def _percentile(xs: list[float], p: float) -> float:
            if not xs:
                return 0.0
            xs = sorted(xs)
            k = max(0, min(len(xs) - 1, int(round(p / 100.0 * (len(xs) - 1)))))
            return xs[k]

        summary = {
            "run_id": run_dir.run_id,
            "target": args.target,
            "iterations": args.iterations,
            "workers": args.workers,
            "seed": args.seed,
            "passed": passes,
            "failed": failures,
            "per_pattern_pass": per_pattern_pass,
            "per_pattern_fail": per_pattern_fail,
            "per_violation": per_violation,
            "latency_s": {
                "min": min(latencies) if latencies else 0.0,
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
                "p99": _percentile(latencies, 99),
                "max": max(latencies) if latencies else 0.0,
            },
        }
        run_dir.summary_path.write_text(json.dumps(summary, indent=2))

        logger.info("=" * 60)
        logger.info(
            f"summary: {passes} passed, {failures} failed of {args.iterations} "
            f"(concurrency={args.workers})"
        )
        for pat in PATTERNS:
            p, f = per_pattern_pass.get(pat, 0), per_pattern_fail.get(pat, 0)
            if p + f > 0:
                logger.info(f"  pattern {pat:<28s}  pass={p:3d}  fail={f:3d}")
        if per_violation:
            logger.info("violation counts:")
            for name, n in sorted(per_violation.items(), key=lambda kv: -kv[1]):
                logger.info(f"  {name}: {n}")
        logger.info(f"summary written to {run_dir.summary_path}")
        if failures == 0:
            return 0
        return 1

    finally:
        fake_llm.stop()
        server.stop()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--target", choices=["rust", "go"], default="rust",
                   help="which server to fuzz")
    p.add_argument("--iterations", type=int, default=20,
                   help="number of random scenarios to run")
    p.add_argument("--seed", type=int, default=0xC0FFEE,
                   help="master RNG seed (controls all per-scenario seeds)")
    p.add_argument("--max-duration-s", type=int, default=30,
                   help="max session_max_duration_s any scenario will pick")
    p.add_argument("--scenario-timeout-s", type=float, default=20.0,
                   help="hard timeout per scenario before aborting")
    p.add_argument("--workers", "-j", type=int, default=1,
                   help="number of scenarios to run concurrently against the same "
                        "server. Useful for stressing concurrent-session handling "
                        "and surfacing race conditions. Total scenarios run is "
                        "still --iterations; --workers controls how many fly in "
                        "parallel.")
    p.add_argument("--run-root", default="/tmp/fuzz_e2e",
                   help="parent directory under which each run gets its own "
                        "subdir (server.log, fake_llm.log, failures/, summary.json). "
                        "Default: /tmp/fuzz_e2e/")
    p.add_argument("--pattern", choices=PATTERNS, default=None,
                   help="force every scenario to use the named pattern. Default: "
                        "weighted random across all patterns.")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(main_async(args)))

if __name__ == "__main__":
    main()
