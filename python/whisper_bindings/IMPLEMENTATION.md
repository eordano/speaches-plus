# whisper_bindings -- port mapping

Maps each upstream symbol from `speaches-plus/go/internal/stt/whisper_cgo.c`
(the C side of the Go cgo wrapper) to its equivalent in
`whisper_bindings/_whisper.cpp`, and from there to the pybind11 surface.

## C-API symbols (extern "C", ABI-compatible with upstream)

Kept verbatim so a future binary diff against the Go cgo build produces
zero deltas.

| Upstream `whisper_cgo.c` symbol | Local definition in `_whisper.cpp` | Notes |
| --- | --- | --- |
| `sp_whisper_init(const char* path)` | line ~22 | identical: calls `whisper_init_from_file_with_params` with default `whisper_context_params`. |
| `sp_whisper_free(struct whisper_context*)` | line ~28 | identical: NULL-guard then `whisper_free`. |
| `sp_whisper_run(...)` (static helper) | line ~38 | identical: `whisper_full_default_params(WHISPER_SAMPLING_GREEDY)`, hardcoded `language="en"`, `n_threads=4`, the `<...>` special-token filter for `avg_logprob` accumulation, the `nsp_sum/count -> mean` segment-level `no_speech_prob`, and the `-2 -> resize buffer` contract. |
| `sp_whisper_transcribe(...)` | line ~134 | identical: thin wrapper calling `sp_whisper_run` with NULL stat pointers. |
| `sp_whisper_transcribe_full(...)` | line ~144 | identical: same wrapper, with caller-supplied `avg_logprob_out` / `no_speech_prob_out`. |

`sp_whisper_transcribe_segmented` is not ported in this first cut -- it is
only used by the diarized-transcription endpoint in the Go server, and the
equivalent feature on the Python side is currently routed through
`Ct2WhisperBackend` (which has its own `generate_segmented` path). When
diarized transcription is wanted on the whisper.cpp backend, port that
function next; the structure parallels `sp_whisper_run`.

## C++ class (bound via pybind11)

| `_whisper.cpp` member | Upstream equivalent | Python signature |
| --- | --- | --- |
| `WhisperContext(model_path)` | `sp_whisper_init` + null-check | `WhisperContext(model_path: str)` -- raises `RuntimeError` on load failure (mirrors Go's `NewWhisper` error path). |
| `WhisperContext::open(model_path)` (static) | factory wrapper | `WhisperContext.open(model_path) -> WhisperContext` |
| `WhisperContext::close()` | `sp_whisper_free` | `WhisperContext.close() -> None` |
| `WhisperContext::is_open` | bool view of `ctx_` | property |
| `WhisperContext::transcribe(samples, language)` | `sp_whisper_transcribe` (or local `sp_whisper_transcribe_with_language` shim when `language` is set) | `transcribe(samples: ndarray[float32], language: str | None = None) -> {"text": str}` |
| `WhisperContext::transcribe_full(samples)` | `sp_whisper_transcribe_full` | `transcribe_full(samples: ndarray[float32]) -> {"text": str, "avg_logprob": float|None, "no_speech_prob": float|None}` |
| `__enter__` / `__exit__` | n/a | context-manager sugar calling `close()` on exit. |

### The `language` override

Upstream's `sp_whisper_run` hardcodes `wparams.language = "en"` and the Go
cgo wrapper accepts no language argument. The Python side wants a per-call
`language` kwarg (matching CT2's per-call language token), so
`transcribe(..., language="ja")` re-implements the text-only path inside
`sp_whisper_transcribe_with_language` -- the only difference from upstream
`sp_whisper_run` is the `wparams.language = language` line. When
`language=None` (the default) we fall through to the verbatim upstream
`sp_whisper_transcribe`, preserving byte-for-byte ABI parity on the default
path.

### Buffer growth contract

Both `transcribe` and `transcribe_full` implement the same loop the Go cgo
wrapper does:

1. Allocate `buf_size` bytes (start at 64 KiB, matching Go).
2. Call into the C symbol with `*out_size = buf_size`.
3. On `rc == 0`: success -- copy `buf[..size]` into the Python `dict`.
4. On `rc == -2`: the C side wrote the *required* size into `*out_size`
   without writing any payload -- bump `buf_size = size + 1` and retry.
5. On any other `rc`: raise `RuntimeError`.

The retry loop is unbounded (matching Go), since `-2` is monotonic (each
retry sets a strictly larger required size, and the C side caps the
eventual size at `n_samples * something_bounded`).

### GIL handling

Each call into the C entry points releases the GIL via
`py::gil_scoped_release` for the duration of the `whisper_full` call, so
concurrent Python threads keep working while the model runs. Matches the Go
cgo behavior (cgo calls release the goroutine's GMP slot).

## Layout

The package: `_whisper.cpp` (pybind11 module + verbatim upstream C
symbols), `setup.py` (`Pybind11Extension('_whisper', ['_whisper.cpp'],
libraries=['whisper'])`, with `WHISPER_INCLUDE_DIR` / `WHISPER_LIBRARY_DIR`
env-var overrides), `pyproject.toml` (PEP 517, `pybind11>=2.13`),
`__init__.py` (graceful loader exposing `EXTENSION_AVAILABLE`,
`EXTENSION_IMPORT_ERROR`, `WhisperContext`), `README.md` (build + fallback
policy). The Python wrapper that turns `WhisperContext` into the
`WhisperBackend` Protocol implementation lives at `stt/whisper_cpp.py`,
sibling to `stt/ct2.py`.
