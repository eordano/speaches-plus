# stt/ -- Whisper noise gate

The speech pipeline as a whole (backends, mel chain, long-audio tiling,
gate placement) is documented in `docs/book/07-speech-stack.md`; this
file keeps the Python-specific decisions and the rationale that chapter
cites.

## noise_gate.py

Faithful Python port of `speaches-plus/rust/src/stt/noise_gate.rs` (itself a
port of speaches' `_effective_avg_logprob_threshold` plus the
`nsp_fail || logprob_fail` decision in
`realtime/input_audio_buffer.py:296-302`).

Why the gate exists: Whisper hallucinates short, plausible-looking
transcripts on coughs, breath, lip smacks, and ambient noise. Two stats from
the model bound the lie: `no_speech_prob` (Whisper's own estimate that the
audio is not speech) and `avg_logprob` (lower = less confident decode).

Why the threshold is duration-aware: for the `avg_logprob` gate, *short
audio is the riskiest* -- a cough is ~200 ms and a real word might also be
~200 ms, but the longer the audio runs the more likely it's real speech. So
the threshold relaxes linearly once the utterance passes `FULL_MS` (1.5 s),
lerping toward `LOOSE_FLOOR` (-3.0), and disables outright at `OFF_MS` (5 s).

`evaluate()` checks `no_speech_prob` before `avg_logprob` to match Python's
reason-priority ordering. Missing stats (`None`) are treated as "no signal"
so the gate degrades gracefully when a backend cannot surface them.

Function and constant names match upstream Rust verbatim
(`effective_avg_logprob_threshold`, `evaluate`, `GateThresholds.disabled`,
`FULL_MS`, `OFF_MS`, `LOOSE_FLOOR`); the Rust `NoiseRejection` enum becomes
a Python `Enum` with the same string values via `as_str()`.

Wired through `stt.segments.transcribe_long`, which the
`/v1/audio/transcriptions` route delegates to when `STT_BACKEND=whisper`.
Per-chunk rejections are dropped; surviving chunks have their
`avg_logprob` / `no_speech_prob` aggregated weighted by chunk duration.
The qwen3_omni path does not use the gate (different model, own
hallucination handling).

## constants.py

Whisper's encoder is shape-frozen at 30 s x 16 kHz -> 80 (or 128) mels x
3000 frames. Constants live in one module so `mel.py`, `segments.py`,
`http.py` and the sibling `ct2.py` can't drift on values like
`WHISPER_NB_FRAMES = 3000` or `WHISPER_PAD_SAMPLES = 480000`. Names mirror
Go's `whisperNbFrames` / `whisperPadSamples` etc. for cross-port
greppability. `LANGUAGE_CODES` is the canonical Whisper language list, used
for HTTP request validation.

## mel.py

Faithful Python port of `speaches-plus/rust/src/stt/mel.rs` (which mirrors
`go/internal/stt/mel.go`, which mirrors faster-whisper's preprocessing).
The chain (pad-to-30s, reflect-pad, Hann/FFT, Slaney mel, log/clamp/
normalize) is spelled out in `docs/book/07-speech-stack.md` § "The mel
frontend". Output is `(n_mel, 3000)` row-major. The shape and the
eps/clamp/normalize constants are byte-identical to faster-whisper because
that's the input distribution Whisper was trained on; deviating breaks
accuracy in non-obvious ways.

Python specifics: `numpy.fft.rfft` runs batched (vectorized over the 3000
frames) -- faster than per-frame loops while staying within the numpy
dependency budget; the mel matmul is a single `(n_mels, 201) @ (201, 3000)`
GEMM.

## whisper.py

Hosts the *backend-agnostic* surface callers code against: the `Backend`
enum, `WhisperBackend` Protocol, `TranscriptionResult` and `TimedSegment`
dataclasses, and the timestamp/segmentation helpers
(`parse_timestamp_token`, `classify_timestamp`, `split_ct2_segments`,
`peak_amplitude`, `join_segments`, `strip_special_tokens`).

`TranscriptionResult` + `TimedSegment` are the *canonical* shape --
`stt/ct2.py` re-exports `Segment` as an alias for `TimedSegment` and
imports `TranscriptionResult` from this module so all backends return the
same type. Field names + semantics match upstream Rust.
`compression_ratio` is reserved (faster-whisper surfaces it; ct2/
whisper.cpp don't expose it directly here).

### WhisperBackend Protocol contract

`@runtime_checkable` Protocol with two members:

```
class WhisperBackend(Protocol):
    model_id: str
    def transcribe(
        self,
        samples: np.ndarray,
        sample_rate: int = 16000,
        *,
        language: str | None = None,
        prompt: str | None = None,
        with_timestamps: bool = False,
        task: str = "transcribe",
    ) -> TranscriptionResult: ...
```

Decision: callers usually have raw PCM, not a precomputed mel -- both
`Ct2WhisperBackend` and `WhisperCppBackend` already accepted samples and
computed their own mel internally (whisper.cpp owns its mel inside the C++
binding; ct2 lazily computes via librosa). Routing the Protocol through
`samples` matches that and lets `stt.segments.transcribe_long` chunk audio
into 30 s windows without a redundant `MelExtractor.log_mel(...)` step.

`model_id` is an attribute (not a method) so `getattr(backend, "model_id")`
returns a string with no call overhead, and so HTTP responses can include
the actual model id (file basename of the on-disk model path).

### Translations contract

`task="translate"` instructs the backend to *translate* the input audio to
English (Whisper's standard task), as opposed to transcribing in the source
language. Setting `language="en"` and calling transcribe -- the earlier
behaviour -- was wrong: Whisper's language token is the *source* language,
not the target, so that produced gibberish for non-English audio. Per
backend:

- `Ct2WhisperBackend`: passes `<|translate|>` instead of `<|transcribe|>`
  in the decoder prompt tokens
  (`<|startoftranscript|><|<lang>|><|translate|>`). The PyPI ctranslate2
  fallback path supports this directly. The native `ct2_bindings._ct2`
  extension currently hardcodes `<|transcribe|>` in C++, so
  `task="translate"` raises `NotImplementedError` on that path until the
  binding is rebuilt with a task-token override.
- `WhisperCppBackend`: forwards to `WhisperCppBackend.translate(...)`,
  which calls the binding's `translate_full(...)` entrypoint (whisper.cpp
  sets `wparams.translate = true`). If the bound extension doesn't expose
  `translate_full` yet (current binding does not), the method raises
  `NotImplementedError`.

`stt.http.translations_post` calls `_run_backend(..., task="translate")`,
which threads the task down through `transcribe_long` ->
`backend.transcribe(..., task="translate")`.

`split_ct2_segments` reproduces the Rust splitter exactly: classify by ID
when the tokenizer exposes `<|0.00|>` as a known token, else fall back to
string-pattern parsing; clamp segment times against the real audio length;
drop inverted (`ts_end < ts_start`) pairs.

## segments.py

`transcribe_long` is the Python addition for audio longer than Whisper's
30 s window. Upstream Rust `transcribe_full` only handles single utterances
<=30 s; the Python port surfaces a chunking wrapper because the HTTP route
naturally accepts arbitrary-length uploads. Pipeline: silence pre-gate via
`peak_amplitude` (same threshold as Rust, `SILENCE_PEAK_THRESHOLD =
0.005`); chunk into 30 s windows, pad each non-silent chunk to exactly 30 s
and run the backend; run the duration-aware noise gate per chunk (rejected
chunks dropped silently); time-shift each chunk's segments by the chunk's
offset_ms so segment times stay absolute; aggregate avg_logprob /
no_speech_prob across surviving chunks weighted by chunk duration -- same
recipe as `aggregateSegmentStats` in `go/internal/stt/http.go`.

## http.py

FastAPI handlers `transcriptions_post` and `translations_post` -- match the
OpenAI `/v1/audio/transcriptions` / `/v1/audio/translations` shape, and the
response_format taxonomy from `go/internal/stt/http.go` (text / json /
verbose_json / srt / vtt / diarized_json).

### Response shape

For BC with the existing qwen3_omni branch, the `json` response_format
includes `text`, `language`, `model`, and `task` keys (`task` is
`"transcribe"` for transcriptions, `"translate"` for translations); `model`
comes from `backend.model_id`. `verbose_json` includes `task / language /
duration / text / segments / words` to match the qwen3_omni verbose-json
shape (`words` is always `[]` -- the Whisper backends here don't emit
word-level alignment). SRT/VTT remain text-only.

`translations_post` does NOT set `language="en"`; it threads
`task="translate"` down (see "Translations contract" above).

Wired into `server.py` behind the `STT_BACKEND` env var. Default is
`STT_BACKEND=qwen3_omni` (or unset), preserving the existing Qwen3-Omni
codepath. `STT_BACKEND=whisper` (aliases: `ct2`, `ctranslate2`,
`faster-whisper`) causes `lifespan` to load a `Ct2WhisperBackend` from
`stt.ct2`, taking the model path from the first entry of
`SPEACHES_PLUS_MODELS` (comma-separated, mirroring `QWEN3_TTS_MODELS`). On
load failure (extension not built, model missing) the server logs a warning
and falls back to qwen3_omni.

### Dispatch decision: Option B (parse-once-then-dispatch)

The route `/v1/audio/transcriptions` keeps its original FastAPI signature
(`File()`, `Form()` for multipart parsing). When `_stt_backend` is non-None
and the request does not explicitly target the qwen3-omni model id, the
route delegates to `stt.http.transcriptions_post` with the parsed args
(`file`, `response_format`, `language`, `prompt`), seeking the upload back
to 0 first. Otherwise it falls through to the existing qwen3_omni path.
Same shape for `/v1/audio/translations`.

Option A (refactor `transcriptions_post` to accept `request: Request` and
parse multipart inside, matching upstream Rust's `(State, Multipart) ->
Response`) was rejected: the existing route already parses multipart for
the qwen3_omni path, and parsing twice (or moving the parse into
`stt.http`) would force the server.py route to read the body twice or
re-stream it. Option B keeps multipart parsing at the FastAPI layer
(single source of truth) and treats `stt.http` as a backend-agnostic
helper taking already-decoded inputs.

`_decode_audio_bytes` first tries the project's `audio.codecs.decode_any`
(Opus / WAV / G.711 with a single entrypoint per the Audio-Codecs port),
then falls back to `soundfile + librosa.resample`. Mirrors how the Go
server delegates to its `internal/audio` package and resamples to 16 kHz
before STT.

### Model registry

When the whisper backend is loaded, `_build_models()` adds an entry for the
model path under `task.ASR`, so it shows up in `/v1/models` alongside the
qwen3-omni ASR entry. Mirrors how Kokoro / aligner register themselves only
when their load succeeded.

## bpe.py

Port of `go/internal/stt/bpe.go` -- reproduces the GPT-2 / Whisper
byte-to-Unicode mapping (`bytes_to_unicode` from the canonical OpenAI GPT-2
tokenizer). The vocab is 256 entries: printable ASCII (`!`-`~`), two
Latin-1 ranges, and synthetic codepoints `256+n` for the bytes that fell
through. `decode_bpe` reverses the rune-to-byte mapping then UTF-8-decodes;
`encode_bpe` does the inverse and round-trips for any UTF-8 string.
`BPETokenizer` is a thin OO wrapper. Used by Whisper for prompt encoding
when the caller wants to bias decoding toward a context (vocabulary,
names, etc).
