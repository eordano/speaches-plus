# The Speech Stack

Everything between a microphone and a speaker: audio decoding, speech
recognition, voice activity detection, turn detection, speaker diarization,
forced alignment, sentence segmentation, speech synthesis, and the realtime
conversational loop chaining them. It is the one large subsystem that is not
the LLM engine of `01-architecture.md`.

Three implementations exist. `rust/` is primary and the only one consuming
the `nv-*` crates. `go/` is a self-contained server (pion, whisper.cpp,
Kokoro) replaying the same fixture corpus. `python/` is a reference port
whose per-module `IMPLEMENTATION.md` files carry the rationale the Rust tree
strips from its comments. The three share constant and function names on
purpose, so a value like `WHISPER_NB_FRAMES` greps across languages; where
they diverge, the divergence is usually a defect and is called out below.

Routes are enumerated in `06-serving-surface.md`; this chapter is what
happens behind them. The normative contract for the realtime protocol is
`docs/book/07.1-barge-turn-spec-rfc-v3.md`, the RFC below: where it states a
rule, this chapter names the code implementing it rather than restating it.
Rates and accuracies live in `perf/runs.jsonl` per `08.4-PERFORMANCE.md`.

## One canonical audio format

Every input path converges on **16 kHz mono f32** (RFC §3.0).
`TARGET_SAMPLE_RATE` is defined once per language (`rust/src/audio/types.rs`,
`go/internal/audio/types.go`, `python/audio/types.py`) because Silero,
Whisper, WeSpeaker, DiariZen and the audio EOU head all require f32 at that
rate at the call boundary.

`rust/src/audio/decode_any.rs::decode_any_to_16k_mono` is the single ingest
entry point, sniffing in a fixed order: a MIME shortcut (`audio/pcm` /
`audio/raw` is raw s16le already at 16 kHz); WAV via `audio/wav.rs`, treated
as "not WAV" on failure, handling int16/24/32 and float and rewriting RIFF
and `data` chunk sizes a streaming writer left as `0xFFFFFFFF`; Ogg/Opus,
magic-sniffed in `audio/ogg_opus.rs` (`OggS` + `OpusHead`, decoded at native
48 kHz with `pre_skip` frames trimmed, stereo mean-downmixed, then
resampled); WebM/Matroska + Opus via the hand-written EBML parser in
`audio/webm_opus.rs` (VINT id/size decoding, `SimpleBlock` and
`BlockGroup>Block` payloads, `A_OPUS` only, `CodecPrivate` for `pre_skip`);
and everything else through symphonia in `audio/avdecode.rs`.

Resampling on the decode path is linear interpolation
(`rust/src/audio/resample.rs`), guarded by minimum and maximum input rates. A
size cap from `SPEACHES_MAX_AUDIO_SECONDS` is enforced before conversion, not
after.

G.711 (`rust/src/audio/g711.rs`) is deliberately *not* in `decode_any`: µ-law
and A-law appear only as realtime wire formats, encoded and decoded in
`rust/src/realtime/audio_in_ws.rs` and `audio_out_ws.rs`. The accepted
realtime formats are `pcm16`, `pcm16_16k`, `g711_ulaw`, `g711_alaw`
(`rust/src/defaults.rs`).

Go takes a different route for the tail case: `go/internal/audio/avdecode.go`
is a cgo FFmpeg decoder with a memory-backed `AVIOContext` and `swr_convert`,
so Go covers any libav container natively where Rust hand-rolls Ogg and WebM.
Python's `python/audio/decode_any.py` tries `soundfile` then shells out to
`ffmpeg`. Go also carries a windowed-sinc `PolyphaseUpsampler`
(`go/internal/audio/polyphase.go`), used on TTS output rather than ingest.

## Speech to text: three backends, and why each exists

`rust/src/stt/mod.rs` defines `enum Backend { Ct2, WhisperCpp, Parakeet }`
and one `WhisperEngine` wrapping any of them. `Backend::from_env` reads
`STT_BACKEND`: `ct2` / `ctranslate2` / `faster-whisper` select CTranslate2;
`parakeet` / `parakeet-tdt` / `tdt` select the Parakeet-TDT ONNX path; `""` /
`whisper-cpp` / `whisper_cpp` / `cpp` / `ggml` select whisper.cpp. **The Rust
default is whisper.cpp; the Go default is ct2** (`go/cmd/server/main.go`) — a
real divergence, not a documented choice.

**Parakeet-TDT** (`rust/src/stt/parakeet.rs`) is the CPU-fast English lane:
NVIDIA parakeet-tdt-0.6b-v2 via the istupakov ONNX export (nemo128 mel
preprocessor + conformer encoder + TDT decoder_joint greedy loop over 1024
sentencepiece tokens, blank 1024, five duration bins), on the same `ort`
runtime as VAD/EOU. Transcribe-only English (`/v1/audio/translations` refuses
via `TranslateUnsupported`), one proportional-word segment per request, and
conformer full-attention cost bounded by 60-second decode windows
(`WINDOW_SAMPLES_BOUNDS_FULL_ATTENTION_COST`). Checkpoint resolution:
`STT_PARAKEET_DIR`, else the newest hub snapshot of
`istupakov/parakeet-tdt-0.6b-v2-onnx`. The suite covering it is
`stt_parakeet_real` (gated `NV_PARAKEET_TEST=1`): it transcribes the
committed espeak clip `rust/tests/data/parakeet_fox_16k.wav` exactly and
asserts faster-than-realtime CPU decode including the 128-mel frontend, with
the model's own inverse text normalization ("seventeen" -> "17"). The
python-side onnx-asr baseline runs far faster on CPU and orders faster on
CUDA, so the CUDA EP via `ORT_DYLIB_PATH` is the headroom lane when
English-only STT becomes the bottleneck.

Model resolution is by path convention, not model id: `ct2_model_dir`
requires `<models>/whisper-ct2/model.bin`; `whisper_cpp_model_path` probes
`ggml-large-v3-turbo.bin`, `ggml-large-v3.bin`, `ggml-tiny.en.bin` in order.
One engine is pinned per process; the multipart `model` field is parsed and
ignored.

The bindings differ per language and are all vendored:

| | CTranslate2 | whisper.cpp |
|---|---|---|
| Rust | `ct2rs` crate | `whisper_rs` crate |
| Go | cgo shim `go/internal/stt/ct2_cgo.cc` | cgo shim `go/internal/stt/whisper_cgo.c` |
| Python | `python/ct2_bindings/_ct2.cpp` (pybind11) | `python/whisper_bindings/_whisper.cpp` (pybind11) |

The C++ call sites are vendored rather than pulled from wheels so all three
languages share one `extern "C"` ABI (`sp_ct2_open`,
`sp_ct2_generate_segmented`, `sp_whisper_transcribe_full`, …) and therefore
identical prompt assembly, identical `avg_logprob` computation, and the same
`-2 → resize buffer` contract (`python/ct2_bindings/README.md`,
`python/whisper_bindings/README.md`).

Why two Whisper backends at all: **kernel coverage** — CTranslate2 has no
Metal kernels so Apple silicon uses whisper.cpp, while CTranslate2 is the
path with a real CUDA build (`ct2_config()` is `#[cfg(feature = "cuda")]`
gated to `Ct2Device::CUDA` + `Ct2ComputeType::FLOAT16`); **quantization
control** — Go threads `CT2_COMPUTE_TYPE` into
`ctranslate2::str_to_compute_type` so `int8` / `float16` / `int8_float16` are
reachable, with no whisper.cpp analogue; **word timestamps** — whisper.cpp
exposes per-token timings, ct2rs does not; **artifact format** — ggml `.bin`
and CT2 `model.bin` + `tokenizer.json` are disjoint, so the checkpoint forces
the backend; **concurrency** — whisper.cpp needs `ctx.create_state()` per
request while CTranslate2 is internally thread-safe and shares one
`Arc<Ct2State>`.

A third backend, `qwen3_omni`, exists only in `python/` and is the default
there; the Rust tree has the `nv-omni` encoders but does not route
transcription through them (see "What is scaffolding").

Translation capability is decided by parsing the checkpoint name:
`checkpoint_is_transcribe_only` treats any id containing `turbo`, or a `.en`
token not followed by an alphanumeric, as translation-incapable, and for ct2
the tokenizer must additionally expose `<|translate|>`.
`/v1/audio/translations` then refuses with the typed `TranslateUnsupported`
error unless `STT_ALLOW_UNSUPPORTED_TRANSLATE=1`.

## The mel frontend

Whisper's encoder is shape-frozen at 30 s × 16 kHz, so the frontend is fixed
rather than parameterized. It is implemented three times —
`rust/src/stt/mel.rs`, `go/internal/stt/mel.go`, `python/stt/mel.py` — and
only for the ct2 backend; whisper.cpp computes its own mel in C++.

`N_FFT = 400` (25 ms), `HOP_LENGTH = 160` (10 ms), `N_FRAMES = 3000`,
`TARGET_SAMPLES = 480000`. `n_mels` is read from the model
(`model.n_mels()`), 80 for most checkpoints and 128 for large-v3. The chain:
pad or truncate to exactly 30 s, reflect-pad by `N_FFT/2`, Hann-window each
frame, real FFT to a power spectrum, apply a **Slaney**-scale mel filterbank,
`log10(max(v, 1e-10))`, clamp to `global_max - 8`, normalize `(v + 4) / 4`.
These constants are frozen rather than tuned because they reproduce
faster-whisper's preprocessing exactly, which is the input distribution the
weights were trained on (`python/stt/IMPLEMENTATION.md`).

## Decoding, long audio, and the silence pre-gate

Decoding is greedy everywhere in Rust: ct2 with `beam_size: 1`, whisper.cpp
with `SamplingStrategy::Greedy { best_of: 1 }`. Go's ct2 path defaults to
beam 5. There is no temperature-fallback ladder anywhere; the HTTP layer
rejects `temperature != 0` outright with `unsupported_parameter`
(`check_stt_params` in `rust/src/main.rs`) and rejects `prompt` for the same
reason — conditioning on previous text is disabled (`set_no_context(true)`,
and the ct2 prompt is rebuilt per chunk as
`["<|startoftranscript|>", <lang>, <task>]`). `suppress_blank` is on for
whisper.cpp. `<|notimestamps|>` is omitted from the segmented ct2 prompt so
timestamp tokens are emitted. The compression-ratio threshold upstream
faster-whisper uses is not implemented:
`TranscriptionResult.compression_ratio` exists but is `None` at every
construction site.

Language detection is per-backend: ct2 calls `detect_language` on the encoder
output and uses the winning language token in the prompt; whisper.cpp sets
`"auto"` and reads back `full_lang_id_from_state`; Go bakes `<|en|>` into the
prompt.

Long audio is handled by **fixed 30 s non-overlapping tiling**, not a sliding
window or a seek-token loop. `go/internal/stt/long.go` (`chunkAudio`,
`shiftSegments`) and the Rust `transcribe_ct2_long` both chunk, skip chunks
whose peak amplitude is below `0.005`, run the model, shift each chunk's
segment and word times by the chunk offset, join surviving texts with one
space, and recombine `avg_logprob` / `no_speech_prob` as a
**duration-weighted mean** over surviving chunks. In Rust only the ct2 arm is
chunked; `transcribe_whisper_cpp` hands the whole buffer to `whisper_full`,
which windows internally. The peak-amplitude gate at `0.005` is the cheapest
hallucination filter and runs before the model is invoked at all.

## The noise gate

Whisper produces short, plausible transcripts for coughs, breaths and ambient
noise. `rust/src/stt/noise_gate.rs` bounds that with the two statistics the
model emits: `no_speech_prob` and `avg_logprob`. `evaluate()` checks
`no_speech_prob` first (reason-priority ordering matching upstream), then
`avg_logprob` against a **duration-aware** threshold:
`effective_avg_logprob_threshold` keeps the base threshold up to `FULL_MS`,
lerps toward `LOOSE_FLOOR` between `FULL_MS` and `OFF_MS`, and disables
entirely past `OFF_MS` — a cough and a real word are indistinguishable at
short durations, while a longer utterance is more likely real speech
(`python/stt/IMPLEMENTATION.md`). Missing
statistics are treated as "no signal" and pass, so the gate degrades
gracefully across backends.

Placement differs by language. In Rust and Go the gate sits **between STT and
the LLM in the realtime pipeline** (`rust/src/realtime/pipeline.rs`, in
`process_utterance`, immediately after `run_stt_full`), with thresholds read
per-session from `Session::noise_gate_thresholds` — NaN meaning off, so the
gate is disabled by default and enabled through `session.update`. Python
additionally runs it per 30 s chunk inside `python/stt/segments.py`, dropping
rejected chunks before statistics are aggregated.

The complete filter chain: peak-amplitude pre-gate → per-chunk peak gate →
decode → empty-text drop → duration-aware noise gate (realtime only) →
empty-transcript drop.

## Segments, word timestamps, and output shapes

Segment boundaries for ct2 come from Whisper's own timestamp tokens.
`split_ct2_segments` walks the token sequence pairing consecutive timestamp
tokens; `classify_timestamp` prefers token-id arithmetic when the tokenizer
exposes `<|0.00|>` as an added token and falls back to string parsing.
Inverted pairs are dropped and times clamped to the real audio length; if
nothing valid survives, `whole_clip_segment` emits one segment spanning the
clip.

Word timestamps have two mechanisms, neither DTW nor cross-attention
alignment. Under **whisper.cpp** the backend provides token times
(`set_token_timestamps(true)`) and
`group_whisper_tokens_into_words` opens a new word on any token whose text
starts with a space, taking the first token's `t0` and the last token's `t1`;
pseudo-tokens (`[_BEG_]`, `[BLANK_AUDIO]`, `<|…|>`) are filtered first by
`is_whisper_pseudo_token` so they cannot become words. Under **ct2**,
`proportional_word_timings` splits on whitespace and apportions the segment
span by cumulative character count — also the fallback when whisper.cpp
yields no usable token times. Go has no word timestamps at all; its `Segment`
carries only start, end, text and statistics.

`rust/src/oapi/transcriptions.rs` holds the response serializers and only
those — routing and multipart parsing live in
`rust/src/main.rs::do_stt_post`, which both `/v1/audio/transcriptions` and
`/v1/audio/translations` delegate to with a `WhisperTask`. The shared adapter
`timed_segments_to_aligned` converts `TimedSegment` into
`nv_aligner::AlignedSegment` (milliseconds to seconds, `speaker: None`).

| `response_format` | Shape |
|---|---|
| `""` / `text` | `text/plain`, bare transcript |
| `json` | `{"text": …}` |
| `verbose_json` | `{task, language, duration, text, segments[], words[]}`; each segment carries `id, start, end, text, words[], avg_logprob, no_speech_prob`, and words are also flattened to a top-level array |
| `srt` | `nv_aligner::to_srt` — 1-based indices, `HH:MM:SS,mmm` |
| `vtt` | `nv_aligner::to_vtt` — `WEBVTT` header, `HH:MM:SS.mmm` |
| `diarized_json` | speaker-attributed segments, below |

There is **no streaming on the HTTP transcription surface** — no `stream`
field, no SSE, one buffered body. Incremental transcription exists only over
the realtime socket as `input_audio_buffer.partial_transcription` and the
`conversation.item.input_audio_transcription.*` events. Go supports only
`text`, `json` and `diarized_json`; Python covers all six.

## Speaker diarization

`rust/src/diarization/mod.rs::Diarizer` orchestrates the pipeline, with a numpy
mirror in `python/diarization/`. Stages, all inside one call:

1. `framing::slide_chunks` cuts overlapping fixed-length windows.
   `DiarConfig::from_env` defaults `chunk_seconds = 16.0`, `hop_ratio = 0.1`, so
   windows overlap heavily. Only full chunks are emitted — a trailing partial
   tail is dropped.
2. `segmentation::SegmentationModel::run_batch` runs the **DiariZen** WavLM +
   Conformer EEND model as ONNX (input `waveform`, `[B, 1, N]`), exported by
   `rust/scripts/export-diarizen-onnx.py`. The raw-waveform layer norm makes the
   sample axis static, so short chunks are zero-padded rather than shape-varied.
   `FRAME_RATE_HZ = 50`, `SAMPLES_PER_FRAME = 320`.
3. `powerset::PowersetDecoder::to_multilabel_hard` decodes a softmax over
   *subsets* of the local speakers capped at `max_speakers_per_frame`, not
   independent per-speaker sigmoids — so overlap is a first-class class (a
   two-element set) rather than a thresholding artifact. `build_mapping`
   enumerates subsets by size then lexicographically; decoding is a hard argmax.
4. `framing::median_filter_multihot` smooths; `framing::extract_spans` converts
   contiguous runs into `Span`s, dropping runs shorter than `min_span_frames`
   and marking spans whose frames are mostly overlap.
5. Unique spans are featurized once each with Kaldi-style log-mel filterbanks
   (`rust/src/diarization/fbank.rs`): 80 mels, 25 ms frames at 10 ms shift,
   pre-emphasis 0.97, **Povey** window, triangular mel filters over
   20 Hz–7600 Hz, natural log, mean-only cepstral normalization over the
   utterance.
6. `embedding::EmbeddingModel` runs **WeSpeaker ResNet293-LM** as ONNX (`feats`
   → `embs`, 256-d, L2-normalized). Spans shorter than one second are dropped.
   A windowed GPU path tiles each span and mean-pools; a CPU path
   length-buckets spans to bound the number of distinct ONNX shapes.
7. `clustering::OnlineClusterer::assign` is **online greedy nearest-centroid
   with EMA centroid update**, not agglomerative or spectral: cosine similarity
   against unit-norm centroids, a new centroid below `clustering_threshold`
   while under `max_speakers`, and a forced merge into the nearest centroid once
   the cap is reached. **There is no permutation-alignment stage between
   chunks** — global identity rests entirely on the embedding space, which is
   why the windows overlap so heavily.
8. `framing::coalesce_segments` merges adjacent same-speaker segments across a
   small gap tolerance.

`rust/src/diarization/embeddings_http.rs` is, despite the name, a local endpoint
serving `/v1/audio/embeddings` in the OpenAI-embeddings shape. The standalone
`/v1/audio/diarization` route is `rust/src/diarization/http.rs` and can also emit
NIST RTTM; `build_speaker_label_map` maps clusters onto supplied
`known_speaker_references` by concatenating each cluster's audio, embedding it,
and taking the argmax cosine.

**Merging with transcription** happens in
`rust/src/main.rs::build_diarized_segments_json`: diarization segments are the
output rows and Whisper segments are bucketed *into* them by midpoint
(containing segment, else nearest), texts joined and statistics
duration-weighted. Missing diarization models degrade to a passthrough with
every `speaker` set to `null` — a silent failure rather than an error. Python
does the same join at word granularity using forced-aligner items.

In the realtime path `rust/src/realtime/diarization.rs` runs `diarize_utterance`
per committed utterance and emits `conversation.item.diarization`. The only
"online" property is that the `Diarizer`, and so the clusterer's centroids, is
owned by the session, keeping speaker ids stable across turns; the offline HTTP
path builds a throwaway `Diarizer` per request. Gated off by default
(`REALTIME_DIARIZATION`), with `rust/src/realtime/order_harness.rs` asserting a
diarization event never precedes its item's transcript.

## Voice activity detection

`rust/src/vad/mod.rs::VadModel` runs **Silero VAD** as ONNX through `ort`,
pinned to the v6.2 release but tolerant of every export generation: the
loader reads the session's input signature and adapts. The classic layout
(v5/v6, 16k-op15, op18-ifless) takes 16 kHz mono f32, a 512-sample inference
window with a **64-sample context window** prepended from the tail of the
previous chunk, an `sr` scalar, and a `[2, 1, N]` recurrent `state` (N read
from the model, 128 today) carried across calls and zeroed on speech end.
Omitting the context window on classic exports makes the model emit its idle
baseline (~5e-4) regardless of input — a failure that looks like a silent
microphone. The fp16-weights export is the same minus `sr`; the v4 legacy
export instead takes `h`/`c` LSTM pairs (`[2, 1, 64]`, outputs `hn`/`cn`) and
no context window. A signature matching none of these is assumed classic and
fails loudly at first inference with the model's own input names. Python
(`vad/silero.py`) mirrors the same detection; **Go supports the classic
layout only**. `silero_zoo_every_signature_detects_speech`
(`SILERO_ZOO_DIR=<dir of exports>`) drives real speech through every export
and is the gate for any future silero release.

Hysteresis lives in `speech_timestamps_from_probs`: a segment opens at
`prob >= threshold` and closes only at `prob < neg_threshold`, where
`neg_threshold` defaults to `max(threshold - 0.15, 0.01)`, and closing
additionally requires a run of sub-threshold windows worth
`min_silence_samples`. Segments shorter than `min_speech_duration_ms` are
discarded, survivors are padded by `prefix_padding_ms` on both edges
(splitting the gap when neighbours are close), and `max_speech_duration_s`
force-cuts runaway segments.

`VadProcessor` is a sweep, not a per-window state machine: it keeps a ring of
recent window probabilities covering `MAX_VAD_WINDOW_MS` and re-runs the
segmenter over the whole ring on each push, emitting `SpeechStarted`,
`SpeechCommitted` and `Failed`. Session-level phase is separate —
`VadPhase::{Silent, Speaking, Stopped}` in `rust/src/realtime/state.rs`,
driven by `Session::handle_vad_event`.

VAD is not optional. If the model fails to load the server does not start; if
per-frame inference fails `VAD_FAILURE_THRESHOLD` times consecutively the
session terminates with `vad_failed`.

## End of utterance, and why barge-in needs it

VAD answers "is there speech right now?"; end-of-utterance answers "did the
user finish their thought?". Silence alone misfires on natural pauses,
enumerations, addresses and hesitations, so a system responding on
`speech_stopped` would interrupt constantly — and a system that interrupts
constantly cannot support barge-in, because every user pause becomes a turn
the assistant tries to take.

`rust/src/eou/mod.rs` defines `EouKind::{Vad, Text, Audio, Fusion, Heuristic,
Integrated}` (RFC §6.2). All heads implement one trait:

```rust
pub trait EouModel: Send + Sync {
    fn score(&self, context: &str) -> f32;
    fn score_with_audio(&self, context: &str, audio: &[f32], sample_rate: u32) -> f32;
}
```

- **`vad`** calls no classifier; the wait is `silence_duration_ms`.
- **`heuristic`** (`eou/heuristic.rs`) is a fixed table over the tail of the
  partial transcript: terminal punctuation high, soft punctuation low,
  hesitation markers and continuation words lower still.
- **`text`** (`eou/onnx.rs`) is a distilled LM.
  `eou/chat_template.rs::format_qwen_chat` renders the conversation into
  `<|im_start|>role\ncontent<|im_end|>` turns with no trailing `<|im_end|>`
  on the partial, left-truncated to `MAX_CONTEXT_TOKENS`; the score is
  `softmax(logits[-1])[<|im_end|>]` via `extract_im_end_prob`. The tokenizer
  is reimplemented in-tree — `eou/bpe.rs`, `byte_map.rs` (GPT-2
  byte↔unicode), `special_trie.rs` for atomic special-token splitting.
- **`audio`** (`eou/audio.rs`) is a smart-turn ONNX model over the trailing
  `audio_window_ms`. `prepare_audio` zero-pads with **leading** alignment
  because the model expects speech at the *end* of its window; the frontend
  is an 80-mel Whisper-style log-mel in the same file. Its text-only
  `score()` deliberately returns `NaN`, so it can never be used without
  audio.
- **`fusion`** runs both heads and combines them.
- **`integrated`** is the RFC's predicted-response path, advertised in the
  capabilities document but inert in Rust (see "What is scaffolding").

**Fusion** (`rust/src/eou/fusion.rs`) first handles degradation, identically
in all three languages: both heads garbage → `1.0` (commit fast), one head
garbage → the other's probability, with `is_garbage_prob` catching
non-finite and out-of-range values. Then
`FusionRule::{NoisyOr, Max, Mean, Weighted, Gated, Logit, Banded}` combines
them. `Gated` is the default and is a learned logistic mixing weight, not a
boolean predicate:

```
r = g·p_audio + (1 − g)·p_text,   g = σ(W · x)
```

over the eight-feature vector `[1, p_text, p_audio, ln(1+audio_s),
ln(1+partial_chars), strong_terminator, soft_terminator,
last_word_is_continuation]`. `DEFAULT_GATED_FUSION_WEIGHTS` is a compiled-in
constant, byte-identical in `go/internal/eou/gated_fusion.go` and
`python/eou/fusion.py`, with a Rust↔Go parity test pinning the outputs.

The verdict feeds a delay curve rather than a binary decision: RFC §6.4 and
`eou/mod.rs::sigmoid_lerp` map `p` through a normalized logistic onto
`[min_delay_ms, max_delay_ms]`, with `curve_k` controlling steepness and the
wait pinned to the maximum below `p_threshold`.

`rust/src/realtime/pipeline.rs::run_eou_dispatch` implements the RFC's
mandatory hard-cap pattern (RFC §6.3.1): a `hard_cap_deadline` derived
**once**, raced by the classifier call, then raced again by the derived wait.
A timeout, an error or a garbage probability all collapse to `p = 1.0` and
the minimum delay (RFC §6.5). `race_hard_cap` uses a biased `select!` so the
deadline wins ties.

Two speculative extensions sit on top. **Eager** dispatch
(`realtime/eou_eager.rs`) fires when the score clears `eager_p_threshold`,
spawning speculative STT and, in conversation mode, a speculative LLM call,
and moving the response phase to `Predicted`. **Predicted**
(`realtime/eou_predicted.rs`) buffers the speculative LLM deltas in a capped
ring; the entire `response.*` family is refused while the phase is
`Predicted`, both at the emit site and structurally via `ResponseGate::open`,
so a wrong speculation rolls back with nothing on the wire.

Configuration comes from `eou/loader.rs::EouConfig::from_env` (`EOU_KIND`,
`EOU_FUSION_RULE`, `EOU_P_THRESHOLD`, `EOU_AUDIO_WINDOW_MS`,
`EOU_THRESHOLDS` as a per-language map, …) and is partly runtime-mutable
through `session.update`. `realtime/session_update.rs` accepts
`turn_detection.type` of only `server_vad` or `none` — `semantic_vad` is not
implemented in Rust or Go; semantic turn-taking is expressed through the
`eou` sub-object. Thresholds are mutable mid-session, but `eou.kind` and
`eou.fusion_rule` are bound at session creation and rejected if changed.

### What the fusion rules are worth, and which one ships

Every rule below is backed by a harness under `client/perf/` --
`fusion_extract.py`, `fusion_train.py`, `fusion_verify.py`,
`text_head_probe.py`, `text_head_fnv.py`, `context_probe.py`,
`eot_pause_exposure.py`, `eot_silence_ab.py` -- and rates and accuracies live in
`perf/runs.jsonl`. The probes hash features, so reproducing one needs
`PYTHONHASHSEED=0`.

**The committed utterance carries its silence tail.** smart-turn is trained on
clips ending in a natural pause, so audio hard-cut at the last speech-positive
window is out of distribution and costs several points of complete-detection.
The VAD seal therefore includes the buffered trailing silence in
`SpeechCommitted.audio` and reports the speech-only length as `speech_samples`;
the EOU classifier scores the full audio while STT, diarization and eager
partials consume only the `speech_samples` prefix (whisper hallucinates on
trailing silence). `VAD_COMMIT_TAIL=0` restores the hard-cut for A/B.

**A convex blend of text and audio cannot beat audio alone, and the ceiling is
the blend rather than the text head's quality.** Retrained on the full
smart-turn corpus against the official held-out split (`fusion_extract.py`,
`fusion_train.py`, `fusion_verify.py`), the unconstrained optimum wins only by
routing to the text head exactly when audio is confident, which flips confident
audio verdicts under adversarial transcripts and fails the anti-flip and
monotonicity invariants; constrained to those invariants the gain is zero. A
far stronger text head does not rescue it: `client/perf/text_head_probe.py`
trains a hashed char/word-n-gram logistic head that beats the shipped `p_text`
heuristic by ~+0.29 AUC and un-inverts the whisper trap (incomplete turns
whisper ends with a strong terminator), yet blending it globally with `p_audio`
still LOSES to audio alone — the channels' errors correlate and a convex mix
dilutes the stronger one. `DEFAULT_GATED_FUSION_WEIGHTS` stays the
default. Text-channel value lives in policy (capping the Incomplete fallback
timer, eager-prefill decisions), not the blend, and the fallback-cap policy is
refuted on both the corpus and on real conversational pauses (`context_probe.py`
over LiveKit eot-bench), where the text head is *anti-correlated* because the
pauses that fool smart-turn fool the transcript the same way.

**Text fusion must be band-gated or capped, never a global reweight.** A global
text weight large enough to matter re-opens the invariant-violating
cross-routing the fusion tests refuse. Two combiners satisfy this and both are
opt-in:

- `FusionRule::Logit` (`EOU_FUSION_RULE=logit`, `combine_fusion_logit`) passes
  audio's log-odds through a Platt term and clamps the text channel's total
  contribution to `[-cap_hold, +cap_cut] = [-2.0, +0.4]`. The caps — not the
  fit — enforce the safety invariants, so adversarial text can never flip a
  confident-audio verdict in either direction; the invariant tests in
  `fusion.rs` are structural, not golden. `DEFAULT_LOGIT_FUSION_WEIGHTS` is
  fitted against the probabilities of the model the deployment ships and **must
  be refit whenever that model changes**, because the audio model's output
  distribution is part of the calibration; stale weights on a retrained audio
  model are over-conservative, which is a latency regression, and the stack's
  margin shrinks as the audio model improves. `fusion_extract.py` plus the
  logit trainer is the refit path;
  `logit_fusion_matches_the_python_trainer_goldens` pins Rust to the trainer's
  arithmetic.
- `FusionRule::Banded` (`EOU_FUSION_RULE=banded`, also per-session) consults
  text only inside the audio-uncertain band, where the text head's value is
  confined. Outside the band the audio verdict passes through untouched;
  inside, the fused score is clamped strictly into the band so out-of-band
  ordering cannot change. Weights are
  `DEFAULT_BANDED_FUSION_WEIGHTS_FIT_ON_CONTEXT_PROBE_BAND`, pinned bit-for-bit
  by goldens in the fusion suite. Under the real-pause protocol it cuts false
  cutoffs meaningfully at the cost of a few points of complete-detection that
  fall through to the silence-timeout fallback as added latency, not errors;
  veto-only gating (text may only hold a cut) is dominated, because text's
  in-band value includes promoting holds to cuts.

The default stays `gated`: turning either on is a false-cutoff-vs-latency
product trade, not a pure win.

**Quote a false-cutoff rate only next to its hold latency**, since any fusion
can buy false-cutoff by holding longer. Both are quoted under the production
delay policy (`sigmoid_lerp`, `MIN_DELAY_MS` / `MAX_DELAY_MS` /
`SILENCE_HARD_CAP_MS`). Mean hold on true completes is dominated by the miss
tail — the few percent of completes scored below `EOU_P_THRESHOLD` ride to the
hard cap — so the top latency lever here is shrinking that tail with a better
audio model, not reshaping the delay curve for the majority already committed
near `MIN_DELAY_MS`.

**Recalibrate under the real-pause protocol, not the corpus.** The logit stack
is operating-point-positive but ranking-negative against the retrained audio
model on eot-bench: it lowers false-cutoff at the 0.5 threshold with identical
eot-detection, yet AUC drops, because the capped text term reorders scores away
from the corpus distribution it was fit on. Any `EOU_P_THRESHOLD` recalibration
must be measured under `context_probe.py`.

**The trained FNV text head is a committed artifact.**
`rust/tests/data/eou_text_head_fnv_v1.bin` (FNV-1a hashed n-gram logistic
weights, DIM 2^18) is what `EOU_TEXT_HEAD_PATH` and `FusionRule::Banded`
consume; `rust/tests/eou_text_head_golden.rs` pins the Rust loader and hash
against training-time probabilities so drift on either side fails loudly.
Regenerate with `client/perf/text_head_fnv.py` (`TH_FEATURES` = a
`fusion_extract.py` shard dir; deterministic seed). The head is calibrated to
the smart-turn corpus — a bare "yes." scores near 1 while a politeness-final
like "That's all I needed, thanks." scores near 0 — so treat it as a
corpus-shaped signal, not a general finality oracle.

**Silence-end and smart-turn are independent knobs.** smart-turn's verdict is
driven by speech content, not by how much trailing silence the window carries,
so shortening `turn_detection::SILENCE_DURATION_MS` is a pure latency choice
*at the smart-turn layer*; what it changes is how often mid-utterance pauses
reach the classifier at all, and shortening it raises mid-utterance false-cut
exposure substantially. The eager two-threshold rescue — commit at the shorter
silence only above confidence `t`, else wait for the longer one — has no
working operating point, because confidence at the shorter silence does not
separate "pausing" from "done". Endpointing-tail latency must come from a
signal other than audio-only smart-turn at a shorter silence.
`eot_pause_exposure.py` enumerates real intra-speech pauses via silero and
scores smart-turn at the exact moment a candidate silence-end would consult it;
`eot_silence_ab.py` is the harness for the model-layer claim and self-checks by
reproducing published whole-clip accuracy, failing loudly when its feature
replica drifts from `rust/src/eou/audio.rs`.

## Barge-in

Barge-in is the path from `vad.speech_started` while a response is active
(RFC §9). Its correctness is why the state machine exists at all.

`Session::handle_vad_event` on `SpeechStarted` cancels any in-flight commit
timer, rolls back a `Predicted` response (aborting both speculative tasks and
emitting nothing to the client), then either commits the barge-in immediately
or — when `barge_in_delay_ms > 0` — buffers the outbound `speech_started` and
sleeps. If `speech_stopped` arrives during that sleep the barge-in is
**suppressed**: the task is cancelled, no event is published, the active
response continues. The pending record is a single slot, so a second
`speech_started` retires the first.

`pipeline::commit_bargein` is the commitment sequence:

1. `Session::cancel_current_response` snapshots `response_id`,
   `assistant_item_id` and `played_ms` under the session mutex, takes the
   runtime, and sets the response phase to `None`. Outside the lock it aborts
   the response task **and** `TtsAbort::cancel`, which stops synthesis, not
   just delivery.
2. If the response's brackets were ever opened, the close cascade is emitted
   in order — transcript done, audio done, content part done, output item
   done with `status: "incomplete"` — then `response.done` with
   `status: "cancelled"`, `audio_end_ms = played_ms`, reason `barge_in`. If
   nothing was opened the cascade is suppressed; you cannot close a response
   the client never saw.
3. The assistant conversation item is amended in place: status becomes
   incomplete, `audio_ms` is clamped to `played_ms`, and its transcript is
   overwritten with the text actually heard.
4. `conversation.item.assistant_truncated` is emitted **only when
   `played_ms > 0`** (RFC §9.3).
5. Only then does the VAD phase advance to `Speaking` and the buffered
   `input_audio_buffer.speech_started` go out.

The ordering is forced by invariant I1 in `rust/src/realtime/state.rs`: a
`Speaking` VAD phase concurrent with an active response is a violation that
terminates the session with `internal_state_error`. Cancel-then-speak is what
keeps the transition legal. `played_ms` is what makes the truncation honest —
it is incremented in the outbound pacer *after* the frame is written to the
transport, so the snapshot taken under the mutex reflects exactly what the
user heard.

## Forced alignment: `nv-aligner`

`rust/crates/nv-aligner` implements classical **CTC forced alignment by Viterbi
over the blank-interleaved label trellis**. `dp.rs::viterbi_align` takes a flat
`[T, V]` log-probability matrix, builds the extended target
`blank, tok0, blank, tok1, …, blank`, and runs a max-product DP with stay /
advance-one / advance-two transitions, the skip legal only when the skipped
state is a blank between two *different* labels. `pipeline.rs::align_with_logprobs`
validates the vocabulary size, converts frame spans with `frame_to_time_ms`, and
groups words into `AlignedSegment`s; `output.rs` holds `to_srt`, `to_vtt` and
`to_diarized_json`; `lib.rs` declares the `Aligner` trait.

There is **no acoustic model in the crate** — it consumes emissions computed
elsewhere. On the serving path only the output formatters are used
(`rust/src/oapi/transcriptions.rs` imports `to_srt`, `to_vtt`, `AlignedSegment`,
`WordTiming` to render Whisper's already-timestamped segments); the DP has no
callers outside its own tests, and `to_diarized_json` is unreachable because
`main.rs` builds a different diarized shape by hand.

`tests/paper_ctc.rs` compares the DP against an exhaustive enumeration of every
legal CTC path over randomized inputs, asserts repeated labels are separated by
at least one blank frame, and pins a known deviation from the paper — the
implementation requires `T >= 2L+1` and so rejects otherwise-valid alignments,
with a test that fails if the deviation is ever fixed without updating the
ledger.

The working forced aligner is the Python one. `python/aligner/` is a fork of
Qwen3-ASR whose `Qwen3ForcedAligner.align` is not CTC at all: it interleaves a
`<timestamp>` token between every word, runs one forward pass, takes the argmax
at the timestamp positions, and scales by the model's timestamp quantum.
`fix_timestamp` repairs non-monotonicity via a longest-increasing-subsequence,
snapping short anomalous runs to a neighbour and interpolating longer ones.
`python/server.py` consumes it for captions and word-level speaker attribution.

## Sentence segmentation: why a Punkt is vendored

`rust/crates/nv-punkt` is a dependency-free port of Kiss & Strunk's unsupervised
sentence boundary detection: whether a period ends a sentence or marks an
abbreviation, an initial or a decimal is learnable from raw unannotated text by
scoring the abbreviation-period collocation with a Dunning log-likelihood ratio.
`trainer.rs::PunktTrainer` learns abbreviation types, orthographic context
(per-type uppercase/lowercase and sentence-initial/mid-sentence bitflags),
sentence starters and collocations; `params.rs::PunktParameters` is the model
plus a hand-written `CURATED_ABBREVS` list; `segmenter.rs` applies it in two
passes (table lookup, then orthographic-heuristic and collocation revision).
`sentences()` returns byte ranges into the original text, so segmentation is
zero-copy.

The pre-trained models are NLTK's upstream `punkt_tab` distribution — 19
languages, each a directory of four plain-text tables (`abbrev_types.txt`,
`sent_starters.txt`, `collocations.tab`, `ortho_context.tab`; the port shares
NLTK's ortho bitflag encoding and `##number##` sentinel, so the tables load
as-is). `flake.nix` pins the set with `fetchzip` from a fixed `nltk_data` commit
and the dev shells export its store path as `NV_PUNKT_DATA`; both servers parse
`$NV_PUNKT_DATA/<lang>/` at first use, so Rust and Go get byte-identical
sentence boundaries from one artifact with no NLTK and no Python. English merges
`CURATED_ABBREVS` on top; other languages load pure upstream via
`Segmenter::for_lang` / `punkt.Trained`. `PunktParameters::english()` /
`punkt.EnglishParams()` fall back to `CURATED_ABBREVS` alone with a loud stderr
line when the data is missing; `english_trained()` / `EnglishTrained()` are the
strict accessors the tests assert on, so a missing file cannot skip silently.
`bin/punkt_train.rs` trains a custom model from a cleaned corpus, `--out DIR`
emitting the same four punkt_tab tables.

The single consumer is TTS chunking. `rust/src/tts/chunk.rs::plan` holds a
lazily-initialized `Segmenter::english`, packs whole sentences greedily into
chunks of at most `max_chars`, and splits an oversize sentence on whitespace,
tagging those boundaries with `INTRA_SENTENCE_SPLIT` so a caller can tell a
clean seam from a forced cut. Its tests pin exactly the traps a regex splitter
fails — `Dr. Smith`, decimal numbers, initials — each asserted against a naive
baseline that gets them wrong. Go's `go/internal/punkt` backs `go/internal/tts`
for the same purpose. Neither the EOU path nor `nv-aligner` uses it;
`nv-aligner` has its own naive terminal-punctuation rule.

## Text to speech: the codec-token talker

`rust/crates/nv-tts` implements Qwen3-TTS-12Hz. A second, unused `Talker` lives
in `rust/crates/nv-omni/src/talker.rs` (the Qwen3-Omni thinker/talker split,
where the thinker is an LLM backbone whose *hidden states* condition a small
autoregressive speech head). Only the `nv-tts` talker is wired to HTTP.

**Codec tokens.** One frame is 1920 samples at 24 kHz carrying **16 codebook
indices** of 2048 entries each (`nv_omni::vocoder::NUM_CODEBOOKS`,
`nv_tts::CODEC_CODEBOOK_SIZE`) — residual-VQ levels, not parallel streams. The
talker's vocabulary is wider than the codebook because codebook-0 space also
carries control ids: pad, BOS, EOS and the think markers
(`nv-tts/src/talker.rs`).

**Dual-stream prefill.** `Qwen3TtsTalker::build_nonstreaming_prefill` builds one
residual sequence by *element-wise adding* a text-stream and a codec-stream
embedding at every position: role prefix, a text side of pads then the TTS BOS
against a codec side of think tokens, an optional speaker-embedding row,
pad/BOS, then the projected body tokens plus a codec pad everywhere. During
decode the addition continues — the next input is the codebook-0 embedding plus
the summed embeddings of the 15 extra codebooks plus the text-stream pad, the
text stream having finished and contributing pad forever.

**`code_predictor.rs` is a depth transformer over codebooks, not a multi-token
predictor over time** (the RQ-Transformer pattern): fifteen embedding tables and
fifteen output heads, recurrence along the codebook axis *within a single
frame*, each step keeping its own small KV cache.

**Sampling** (`nv-tts/src/sampling.rs`) applies, in order: a sign-aware
repetition penalty, an allowed-set mask, a greedy shortcut if sampling is
disabled, temperature, top-k, softmax, top-p, then inverse-CDF sampling from a
seeded xorshift PRNG. The allowed-set mask does double duty — narrowing the
wider head to real codebook entries so control ids cannot be emitted, and
implementing a minimum-length constraint by hiding EOS until enough frames
exist. The base sampler's repetition penalty sees every base token emitted so
far in the utterance; the sub-sampler for codebooks 1–15 uses no penalty and no
mask. There is no classifier-free guidance anywhere in the Rust tree.

**The vocoder** is `nv-omni/src/vocoder.rs`, consumed through
`nv-tts/src/codec_decoder.rs`: an RVQ dequantizer (codebook 0 plus summed
residual codebooks, reconstructed at load time from EMA statistics), a causal
Conv1d, a windowed-causal transformer with LayerScale and RoPE, two
ConvNeXt-block upsample stages, and four BigVGAN-style decoder stages of
transposed convolutions with dilated residual blocks, activated by snakebeta.
Every convolution is causal — `CausalConv1d` left-pads only, `CausalConvT1d`
trims from the right — so there is **zero lookahead**, and that is what makes
streaming exact. `VocoderStreamer` re-decodes `left_context` already-emitted
frames purely to warm the receptive field, then discards exactly that many
frames' worth of samples from the front; with no lookahead and the transformer's
window equal to the configured left context, streamed output matches batch
output. A test asserts that agreement and separately gates seam discontinuity
against the batch signal's own maximum sample-to-sample jump. `Vocoder::decode`
is itself implemented through the streamer, so batch and stream share one path.

`vocoder_loader.rs` is deliberately paranoid: `scan_vocoder_dir` counts keys per
weight subtree into a `VocoderInventory`, `is_real_qwen3_decoder` requires the
expected subtrees to be non-empty, and `VocoderLoadReport` records whether
loading silently degraded to zero-initialized weights. A zero-init vocoder
decodes every frame to silence, answering 200 with a correctly-sized WAV of
nothing, so the route refuses to serve unless `NV_TTS_ALLOW_SILENT_VOCODER` is
set; the synthesis entry point likewise bails rather than emit audio if the
talker, code predictor or text embedding is missing.

`rust/src/oapi/audio_speech_nvtts.rs` runs three stages concurrently — frame
generation on a blocking task, vocoder decode on another, and an async forwarder
that awaits the *first* PCM chunk before returning the receiver, so a
first-chunk failure surfaces as an error status rather than a truncated 200.
Every codebook index is range-checked before it reaches the vocoder.

## Voices: built-in speakers, profiles, custom voice, voice design

Three conditioning mechanisms exist and the checkpoint decides which are
available: `tts_model_type` is read from the checkpoint's `config.json`, and
`profiles_supported` is true only for `base`.

**Built-in speakers.** `resolve_speaker_embed` looks the requested voice up
in `talker.config().spk_id`; the "embedding" is the talker's codec embedding
row for that speaker's token id. OpenAI voice aliases (`alloy`, `echo`, …)
fall back to a default speaker with a warning.

**Voice profiles (custom voice).** `nv-tts/src/speaker_encoder.rs` is an
ECAPA-TDNN: SE-Res2Net blocks with dilated TDNN convolutions, multi-layer
feature aggregation, attentive statistics pooling over `[x, mean, std]`, and
a final projection to a 1024-d x-vector. Its frontend, `spk_mel.rs`, is a
hand-rolled 128-mel log-magnitude spectrogram at 24 kHz with a Slaney
filterbank and center-mode reflect padding. The weights live under
`speaker_encoder.` in the checkpoint; loading errors explicitly say
CustomVoice and VoiceDesign checkpoints carry no encoder and that enrollment
needs a Base checkpoint.

`rust/src/oapi/voice_profiles.rs` serves enrollment. `POST /v1/voice-profiles`
takes `name`, reference audio and an opaque `design_params` JSON blob;
`is_safe_name` is the path-traversal gate; audio is decoded to 16 kHz,
required to exceed a minimum length, resampled to 24 kHz and encoded. If no
encoder is loaded the route answers 503 rather than storing a zero vector.
The response never returns the embedding — only its dimension and an
`embedding_state` of `encoded` or `no_encoder`. Storage is one pretty-printed
JSON file per profile (`nv-tts/src/voice_profile.rs`). At synthesis time a
profile embedding must match the talker's hidden size, must not be all-zero,
and the checkpoint must be `base`; each failure is a distinct 400 with a
message naming the fix. The resulting `[1, 1, hidden]` tensor occupies the
single speaker slot in the codec-stream prefill.

**Voice design** — a voice described in words rather than demonstrated — is
implemented only in Python.
`python/tts/qwen3/inference/qwen3_tts_model.py`'s `generate_voice_design`
builds an instruction as a chat-formatted *prompt*, not a vector, and
`python/server.py` sources it from the OpenAI `instructions` field, falling
back to the voice name string itself. In Rust `design_params` is stored,
round-tripped and never read; `conformance/fixtures/030-voice-clone-design-warm`
says so and asserts only the structural round-trip.

## The Kokoro ONNX path

The second TTS engine is Kokoro-82M as a single ONNX graph, served from
`rust/src/tts/` and `go/internal/tts/`. Its graph takes phoneme ids
zero-padded on both ends, a 256-wide `style` vector and a `speed` scalar, and
emits mono f32 at 24 kHz.

**Execution-provider policy is per model, not global.** The GPU-worthy
sessions ride a ladder or an explicit pin: Kokoro (`KOKORO_ONNX_PROVIDER`,
auto ladder) and both diarization models, segmentation and speaker embedding
sharing `diarization/ep.rs` (`DIAR_EP`). The latency-critical or
trivially-cheap sessions are pinned CPU on purpose and must stay there:
silero VAD (sub-millisecond frames; a GPU round-trip costs more than the
model), smart-turn (`eou/audio.rs`), the text EOU (`eou/onnx.rs`) and the PII
classifier. Every CPU-pinned session works under any dylib because the CPU EP
is always present — verified against the CUDA build and the AMD MIGraphX
build alike. On AMD hosts (`ortAccel = "rocm"`) the unit pins
`KOKORO_ONNX_PROVIDER=migraphx` and `DIAR_EP=migraphx`; a GPU session that
registers but cannot run fails loudly per request (warmup logs the first
failure at boot), never silently on CPU.

**Provider selection** (Rust and Python; Go is CPU-only): the session walks a
GPU provider ladder — cuda, rocm, coreml, webgpu, dml — and runs on the first
one the loaded onnxruntime exposes and registers, falling back to CPU when
none does. `KOKORO_ONNX_PROVIDER=<name>|cpu` forces one (Python spellings
like `CUDAExecutionProvider` are accepted), and a forced provider that cannot
register is a load error, not a silent CPU fallback. OpenVINO is force-only,
never auto: CPU-only onnxruntime builds still list it as available, and its
CPU plugin rejects Kokoro's dynamic-rank shapes at session commit — after
registration has already succeeded, where no ladder fallback is possible.

That distinction is where realtime headroom comes from: CPU inference
saturates at a low multiple of realtime regardless of `KOKORO_INTRA_THREADS`
while the CUDA provider runs an order of magnitude past it.
`rust/tests/kokoro_rtf.rs` measures the served path (`KOKORO_RTF_TEST=1`,
`KOKORO_RTF_MODEL_DIR=<dir with kokoro-v1.0.onnx + voices.bin>`, floor
asserted via `KOKORO_RTF_FLOOR`). The dylib behind `ORT_DYLIB_PATH` — not any
crate feature — decides what is available: the dev shells pin a CPU-only
onnxruntime while the deployed service links a CUDA-enabled build. The two
providers are interchangeable on quality — the same tokens produce spectrally
identical audio, with duration allowed to differ by one duration-predictor
frame — so **tests comparing provider outputs must compare spectra, not
samples**: a few-millisecond timing drift destroys sample-level correlation
while the audio stays identical to the ear.

**Phonemization** is espeak-ng through a hand-written C shim that is
logically identical in both languages (`rust/src/tts/phonemize_glue.c`,
`go/internal/tts/phonemize_cgo.c`): it initializes espeak in retrieval mode,
sets the voice by name, and loops `espeak_TextToPhonemes` with the IPA flag,
returning a negative sentinel with the required size when the caller's buffer
is too small. Both bindings implement the grow-and-retry and cache the
current voice so it is re-set only when it changes.

**Voice packs** are `.npz` files parsed by hand (`rust/src/tts/npz.rs`): a
zip of `.npy` arrays, one per voice, with a header scanner rejecting Fortran
order and any dtype other than little-endian f32. The crucial detail is that
**the style row is selected by phoneme count** — `voice.row(n)` where `n` is
the number of tokens — which is why the pack's leading dimension and
`MAX_PHONEME_LENGTH` must agree, and why exceeding it is an error in Rust
(Go truncates instead, a real divergence).

**Vocabulary** (`rust/src/tts/vocab.rs`) concatenates a pad character, a
punctuation set, the ASCII letters and an IPA inventory, assigning sequential
ids. `clean_phonemes` applies two literal fixes for the word "kokoro", maps a
handful of IPA characters onto in-vocabulary ones, then drops every rune not
in the vocabulary.

**Text handling** is where Kokoro differs most from nv-tts.
`rust/src/tts/text.rs` strips emoji by Unicode range, strips markdown
emphasis and collapses whitespace; `chunk::plan` then splits with Punkt and
packs sentences into chunks bounded by `KOKORO_CHUNK_CHARS`, capped at
`MAX_CHUNK_CHARS`. Chunks are stitched with a fixed silence prefix on every
chunk after the first to mask the concatenation seam — the Kokoro analogue of
the vocoder's left-context trick. The first chunk is synthesized before the
response headers are chosen, so a first-chunk failure maps to a status code
while later failures can only truncate the stream; each subsequent chunk
races client disconnect so a dropped client cancels the work.

**Output encoding**: `pcm` streams raw s16le directly; everything else is
piped through a spawned `ffmpeg` (`rust/src/tts/http.rs`,
`go/internal/tts/ffmpeg.go`), with `ffmpeg` availability probed once and
surfaced as a 400 when missing, and an abort channel so a mid-stream
synthesis failure kills the encoder rather than producing a silently short
file. `stream_format=sse` emits `speech.audio.delta` events carrying base64
s16le, terminated by `speech.audio.done`.

By contrast the nv-tts path applies **no text normalization and no chunking**
at all: raw input goes through a byte-level BPE with no normalizer and no
post-processor, into one prefill bounded only by a maximum frame count.

## Route resolution for `/v1/audio/speech`

There is exactly one route, registered in `rust/src/main.rs`.
`rust/src/oapi/audio_speech.rs` is the router and validator; nv-tts and
Kokoro are two implementations of one trait:

```rust
#[async_trait] pub trait AudioSpeech {
    async fn synthesize(&self, text: &str, voice: &str) -> Result<mpsc::Receiver<Vec<f32>>>;
    fn sample_rate(&self) -> u32;
    fn model_id(&self) -> Option<String>;
}
```

`resolve_speech_route` is a pure function with an exhaustive test block. A
blank `model` picks nv-tts when a talker is loaded, 503s when the talker's
bootstrap *failed* (recorded in a `OnceLock` so "not configured" and
"configured but broken" stay distinguishable), and otherwise falls back to
Kokoro. A model containing "kokoro" always routes to Kokoro. Anything else is
matched against the loaded talker id by `model_id_matches`, which strips both
sides to lowercase alphanumerics and requires a prefix match against the full
id or its basename. A non-match is a **404** whose message says the request
was refused rather than rendered by whichever engine happened to be loaded.

`unsupported_nvtts_params` follows the same philosophy for parameters nv-tts
cannot honour — `speed`, a differing `sample_rate`, `stream_format=sse` —
400ing with the counterfactual stated explicitly ("the parameter would have
been ignored and … returned"). This is a deliberate inversion of the usual
accept-and-ignore convention, and the same instinct appears in the STT
route's rejection of `temperature` and `prompt`.

## The realtime loop

`rust/src/realtime/` ties all of the above into one conversational loop. The
normative contract is the RFC; the conformance corpus under `conformance/`
replays the same fixtures against both the Rust and Go implementations.

**Ingress.** WebRTC decodes Opus and resamples to 16 kHz in
`realtime/audio_in.rs`, carrying leftover samples across calls so chunking
stays exact; WebSocket decodes the session's `input_audio_format` in
`audio_in_ws.rs`. Both push `Vec<f32>` onto one bounded channel and tee into
the inspector's audio store.

**VAD lane.** `spawn_vad_task` owns the only `VadProcessor` and consumes that
channel, rebuilding the processor when `input_audio_buffer.clear` bumps the
clear epoch. Optionally it also runs **partial transcription**: at a fixed
interval, with at most one in flight, it takes the current speech audio and
runs a full STT pass, emitting `input_audio_buffer.partial_transcription`.
Partials are client hints only — they feed neither the LLM nor the EOU
classifier.

**Turn.** On `SpeechCommitted` with enough speech, the session builds the EOU
context and installs `run_eou_dispatch` as an abortable commit timer. When it
fires, `commit_after_eou` pushes the user conversation item, harvests any
speculative work, and emits `input_audio_buffer.committed` before
`conversation.item.added` (RFC §10.3 W3).

**Transcribe.** `process_utterance` stores a sealed buffer, uses the
speculative transcript if one was promoted or runs STT on a blocking task,
applies the noise gate, and emits
`conversation.item.input_audio_transcription.completed` (or `.failed`). Short
utterances are treated as backchannels and suppress the response entirely.

**Respond.** `run_response` opens the response brackets and sets a
`wire_opened` flag — the exact flag barge-in later consults to decide whether
the close cascade is legal. The LLM is reached over **HTTP**, not internally:
`rust/src/conversation/llm.rs` POSTs an OpenAI-shaped
`{model, messages, stream: true}` to `CHAT_COMPLETION_BASE_URL` and
hand-parses the SSE stream, reading `choices[].delta.content`. It can point
at this same binary's `/v1/chat/completions` or anything else. There is no
tool calling in this path: the SSE structs deserialize only `content`, and
the `response.function_call_arguments.*` events declared in `wire.rs` are
unused outside the ordering harness.

Deltas fan out three ways: appended to the response text, appended to a
shared `transcript_so_far` (the string barge-in snapshots when it truncates),
and fed to `SentenceChunker`, which cuts at the first terminal punctuation or
newline. Each sentence is one Kokoro synthesis call on a blocking task,
played through an outbound pacer. The realtime path uses **Kokoro only** —
nv-tts is not wired into it.

The main loop is a biased `select!` over the TTS worker joining and the LLM
stream, so LLM streaming, synthesis and audio pacing run concurrently. The
queues are the sentence channel, the LLM delta channel, and the pacer's own
gate. `QueueGate` tracks queued audio against `outbound_queue_cap_ms` and, if
`played_ms` stops advancing, fails the response with `client_too_slow` rather
than buffering unboundedly. On completion `drain_pacer` waits out the
remaining planned audio against a cap derived from the planned duration,
downgrading to `incomplete` with reason `drain_cap` if the client never
drains it.

**Egress.** The input pipeline is symmetric — everything at 16 kHz — but the
output pipeline deliberately is not (RFC §12.4): TTS produces audio at the
model's native rate and there is a single resample-and-encode step at egress.
WebRTC always emits Opus at 48 kHz on the track regardless of
`output_audio_format`, with the data channel carrying the configured format
in parallel; WebSocket resamples and encodes to PCM16 or G.711 into
`response.output_audio.delta`.

**Ordering.** `Session::deliver_to_sink` holds one lock across
stamp-then-record-then-write, because unsynchronized steps let two tasks
emitting concurrently take sequence numbers `N` and `N+1` and land on the
wire in the opposite order. Events are bucketed by topic, and only the
response topic is gated by the `Predicted` phase.

**Cancellation** has three layers: a session-wide `CancellationToken` with an
inflight-lane counter and a quiesce wait (`realtime/cancel.rs`); a
single-slot `TtsAbort` registry keyed by response id, which stops *synthesis*
rather than just delivery; and the response task's own abort handle.

**Transports** differ only at the edges. The data channel fragments every
event — base64 the JSON, wrap as a full message under a size limit or as
contiguous indexed partials above it (`realtime/framing.rs`); the WebSocket
sends raw JSON. Both converge on the same `Session`, state machine and event
alphabet, which is what lets one fixture corpus cover both.

## What is scaffolding

The audio tree contains substantial built-but-unwired code, named here so the
chapter is not read as a description of live behaviour.

- **`nv-omni` beyond the vocoder.** `Thinker`, `Talker`, `AudioEncoder` and
  `Qwen3VisionTower` are architecturally complete but nothing in `rust/src/`
  imports them; the only `nv_omni::` imports outside the crate are for the
  vocoder. The omni ASR/TTS path exists in `python/` only.
- **`nv-omni/src/codec_vel.rs`.** A FiLM-conditioned flow-matching velocity
  field with no caller anywhere, including inside its own crate — a scaffold
  for a latent decoder the shipped Qwen3-TTS path does not use.
- **`nv-aligner`'s DP.** Fully implemented and validated against brute-force
  path enumeration, with no callers outside its tests.
- **`EouKind::Integrated`.** Advertised in the capabilities document, but it
  resolves to a stub returning `1.0`, the verdict handler has no non-test
  caller, and the VAD handler explicitly excludes it — so selecting it
  commits immediately with no classifier.
- **`conversation.item.retrieve`.** Parsed on the wire and answered with a
  not-implemented error in `rust/src/realtime/session.rs`; RFC §7.5 and W9
  are unmet in the Rust tree.
- **The DiariZen segmentation ONNX is not shipped.** Until
  `rust/scripts/export-diarizen-onnx.py` is run, `/v1/audio/diarization`
  correctly answers 503, but `response_format=diarized_json` returns every
  `speaker` as `null` with no failure signal. The speaker-embedding half
  works independently at `/v1/audio/embeddings`.
- **The Rust and Python diarization configs disagree** on
  `max_speakers_per_frame` and `chunk_seconds`, which changes the powerset
  class count — the two would not decode the same ONNX export.
- **The EOU classifier never sees the current utterance's text.** The context
  is built at `SpeechCommitted`, before the user item is pushed to the
  conversation, so the text head scores only prior turns and the gated-fusion
  text features derive from history rather than the partial. The audio head
  is unaffected.
- **The predicted-transcript mismatch rollback cannot fire**, because both
  sides of its comparison derive from the same binding.

Licensing is load-bearing here: per `NOTICE`, the DiariZen segmentation model
is CC-BY-NC-4.0, so commercial deployments must replace it. Every other
shipped weight in the audio path permits commercial use.
