from __future__ import annotations

import os
import threading
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from .constants import (
    CONTEXT_SAMPLES,
    INPUT_AUDIO_KEY,
    INPUT_SR_KEY,
    INPUT_STATE_KEY,
    MAX_PROB_RING,
    NEG_THRESHOLD_DELTA,
    NEG_THRESHOLD_FLOOR,
    OUTPUT_PROB_KEY,
    OUTPUT_STATE_KEY,
    PREFIX_PADDING_MS,
    SAMPLE_RATE,
    SILENCE_DURATION_MS,
    SPEECH_THRESHOLD,
    WINDOW_SAMPLES,
)
from .segmenter import (
    SpeechTimestamp,
    VadOptions,
    speech_timestamps_from_probs,
    to_ms_speech_timestamps,
)

if TYPE_CHECKING:
    import onnxruntime as ort

class SileroVad:
    def __init__(self, session: "ort.InferenceSession"):
        self._session = session
        self._lock = threading.Lock()
        names = {i.name for i in session.get_inputs()}
        self._legacy_hc = "h" in names and "c" in names
        self._feed_sr = "sr" in names
        units = 64 if self._legacy_hc else 128
        for i in session.get_inputs():
            last = i.shape[-1] if i.name in ("state", "h") and i.shape else None
            if isinstance(last, int) and last > 0:
                units = last
        self._units = units
        self.state = np.zeros(2 * 1 * units, dtype=np.float32)
        self.state_c = np.zeros(2 * 1 * units, dtype=np.float32) if self._legacy_hc else None
        self.context = np.zeros(
            0 if self._legacy_hc else CONTEXT_SAMPLES, dtype=np.float32
        )
        self.sr = np.array([SAMPLE_RATE], dtype=np.int64)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        providers: list[str] | None = None,
    ) -> SileroVad:
        import onnxruntime as ort
        path = Path(model_path)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1
        session = ort.InferenceSession(
            str(path),
            sess_options=sess_options,
            providers=providers or ["CPUExecutionProvider"],
        )
        return cls(session)

    def process_window(self, window: np.ndarray) -> float:
        if window.shape[0] != WINDOW_SAMPLES:
            raise ValueError(
                f"expected window of {WINDOW_SAMPLES} samples, got {window.shape[0]}"
            )
        window_f32 = np.ascontiguousarray(window, dtype=np.float32)
        full_input = np.concatenate([self.context, window_f32])
        if self.context.size:
            self.context[:] = window_f32[WINDOW_SAMPLES - self.context.size :]

        audio_in = full_input.reshape(1, -1)
        state_in = self.state.reshape(2, 1, self._units)
        if self._legacy_hc:
            feed = {
                INPUT_AUDIO_KEY: audio_in,
                INPUT_SR_KEY: self.sr,
                "h": state_in,
                "c": self.state_c.reshape(2, 1, self._units),
            }
            fetch = [OUTPUT_PROB_KEY, "hn", "cn"]
        else:
            feed = {INPUT_AUDIO_KEY: audio_in, INPUT_STATE_KEY: state_in}
            if self._feed_sr:
                feed[INPUT_SR_KEY] = self.sr
            fetch = [OUTPUT_PROB_KEY, OUTPUT_STATE_KEY]
        with self._lock:
            outputs = self._session.run(fetch, feed)
        prob_out = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
        if prob_out.size == 0:
            raise ValueError("empty output")
        new_state = np.asarray(outputs[1], dtype=np.float32).reshape(-1)
        if new_state.size != self.state.size:
            raise ValueError(
                f"state length mismatch: got {new_state.size}, expected {self.state.size}"
            )
        self.state[:] = new_state
        if self._legacy_hc:
            new_c = np.asarray(outputs[2], dtype=np.float32).reshape(-1)
            if new_c.size != self.state_c.size:
                raise ValueError(
                    f"cn length mismatch: got {new_c.size}, expected {self.state_c.size}"
                )
            self.state_c[:] = new_c
        return float(prob_out[0])

    def reset(self) -> None:
        self.state.fill(0.0)
        if self.state_c is not None:
            self.state_c.fill(0.0)
        self.context.fill(0.0)

@dataclass
class SpeechStarted:
    item_id: str
    audio_start_ms: int

@dataclass
class SpeechCommitted:
    item_id: str
    audio_end_ms: int
    audio: np.ndarray
    speech_samples: int = 0

    def __post_init__(self) -> None:
        if self.speech_samples <= 0:
            self.speech_samples = int(self.audio.shape[0])

@dataclass
class Failed:
    reason: str

VadEvent = SpeechStarted | SpeechCommitted | Failed

class TurnDetectionRead(Protocol):
    def threshold(self) -> float: ...
    def prefix_padding_samples(self) -> int: ...
    def silence_duration_samples(self) -> int: ...
    def neg_threshold(self) -> float: ...
    def min_speech_duration_ms(self) -> int: ...
    def max_speech_duration_s(self) -> float: ...

class _DefaultTd:
    def threshold(self) -> float:
        return SPEECH_THRESHOLD

    def prefix_padding_samples(self) -> int:
        return PREFIX_PADDING_MS * SAMPLE_RATE // 1000

    def silence_duration_samples(self) -> int:
        return SILENCE_DURATION_MS * SAMPLE_RATE // 1000

    def neg_threshold(self) -> float:
        return max(self.threshold() - NEG_THRESHOLD_DELTA, NEG_THRESHOLD_FLOOR)

    def min_speech_duration_ms(self) -> int:
        from .constants import MIN_SPEECH_DURATION_MS
        return MIN_SPEECH_DURATION_MS

    def max_speech_duration_s(self) -> float:
        from .constants import MAX_SPEECH_DURATION_S
        return MAX_SPEECH_DURATION_S

class VadProcessor:
    def __init__(self, model):
        self.model = model
        self.buffer = np.empty(0, dtype=np.float32)
        self.pending_audio = np.empty(0, dtype=np.float32)
        self.probs: deque[float] = deque(maxlen=MAX_PROB_RING)
        self.probs_start_window = 0
        self.duration_samples = 0
        self.audio_start_ms: int | None = None
        self.audio_end_ms: int | None = None
        self.current_item: str | None = None
        self.pending: list[VadEvent] = []
        self.td: TurnDetectionRead | None = None

    def with_turn_detection(self, td: TurnDetectionRead) -> VadProcessor:
        self.td = td
        return self

    def _options(self) -> VadOptions:
        td = self.td
        if td is not None:
            return VadOptions(
                threshold=td.threshold(),
                neg_threshold=td.neg_threshold(),
                min_speech_duration_ms=td.min_speech_duration_ms(),
                max_speech_duration_s=td.max_speech_duration_s(),
                min_silence_duration_ms=(td.silence_duration_samples() * 1000 // SAMPLE_RATE),
                speech_pad_ms=(td.prefix_padding_samples() * 1000 // SAMPLE_RATE),
            )
        return VadOptions()

    def _duration_ms(self) -> int:
        return self.duration_samples * 1000 // SAMPLE_RATE

    def current_speech_audio(self) -> tuple[str, np.ndarray] | None:
        if self.audio_start_ms is None or self.current_item is None:
            return None
        start_sample = self.audio_start_ms * (SAMPLE_RATE // 1000)
        if start_sample >= self.buffer.shape[0]:
            return None
        return (self.current_item, self.buffer[start_sample:].copy())

    def push(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        samples_f32 = np.ascontiguousarray(samples, dtype=np.float32)
        self.buffer = np.concatenate([self.buffer, samples_f32])
        self.duration_samples = self.buffer.shape[0]

        if self.audio_end_ms is not None:
            return

        self.pending_audio = np.concatenate([self.pending_audio, samples_f32])
        while self.pending_audio.shape[0] >= WINDOW_SAMPLES:
            window = self.pending_audio[:WINDOW_SAMPLES]
            prob = self.model.process_window(window)
            self.pending_audio = self.pending_audio[WINDOW_SAMPLES:]
            if len(self.probs) == MAX_PROB_RING:
                self.probs_start_window += 1
            self.probs.append(prob)

        if not self.probs:
            return

        opts = self._options()
        ring_samples = len(self.probs) * WINDOW_SAMPLES
        probs_list = list(self.probs)
        timestamps_samples = speech_timestamps_from_probs(
            probs_list, ring_samples, opts, SAMPLE_RATE
        )
        timestamps = to_ms_speech_timestamps(timestamps_samples)
        ring_ms = ring_samples * 1000 // SAMPLE_RATE
        duration_ms = self._duration_ms()
        last: SpeechTimestamp | None = timestamps[-1] if timestamps else None

        if self.audio_start_ms is None:
            if last is not None:
                audio_start_ms = max(duration_ms - ring_ms, 0) + last.start
                item_id = f"item_{uuid.uuid4().hex}"
                self.audio_start_ms = audio_start_ms
                self.current_item = item_id
                self.pending.append(
                    SpeechStarted(item_id=item_id, audio_start_ms=audio_start_ms)
                )
            return

        stop_at_ms: int | None
        if last is None:
            stop_at_ms = duration_ms
        else:
            trailing = max(ring_ms - last.end, 0)
            if trailing >= opts.min_silence_duration_ms:
                stop_at_ms = max(duration_ms - trailing, 0)
            else:
                stop_at_ms = None

        if stop_at_ms is not None:
            self.audio_end_ms = stop_at_ms
            assert self.audio_start_ms is not None
            start_sample = self.audio_start_ms * (SAMPLE_RATE // 1000)
            end_sample = stop_at_ms * (SAMPLE_RATE // 1000)
            end_sample = min(end_sample, self.buffer.shape[0])
            start_sample = min(start_sample, end_sample)
            tail_end = end_sample if os.environ.get("VAD_COMMIT_TAIL") == "0" else self.buffer.shape[0]
            utterance = self.buffer[start_sample:tail_end].copy()
            item_id = self.current_item or f"item_{uuid.uuid4().hex}"
            self.pending.append(
                SpeechCommitted(
                    item_id=item_id,
                    audio_end_ms=stop_at_ms,
                    audio=utterance,
                    speech_samples=end_sample - start_sample,
                )
            )

    def take_events(self) -> list[VadEvent]:
        evs = self.pending
        self.pending = []
        if self.audio_end_ms is not None:
            self.buffer = np.empty(0, dtype=np.float32)
            self.pending_audio = np.empty(0, dtype=np.float32)
            self.probs.clear()
            self.probs_start_window = 0
            self.duration_samples = 0
            self.audio_start_ms = None
            self.audio_end_ms = None
            self.current_item = None
            self.model.reset()
        return evs
