# whisper_bindings -- pybind11 wrapper around whisper.cpp

Python equivalent of the upstream Go cgo wrapper at
`speaches-plus/go/internal/stt/whisper_cgo.c` and
`speaches-plus/go/internal/stt/whisper_cgo.go`. It is intentionally
vendored here rather than pulled from a PyPI wheel -- whisper.cpp does not
ship a canonical Python wheel, and we want a single C++ call site that
mirrors upstream's `extern "C"` ABI symbol names (`sp_whisper_init`,
`sp_whisper_free`, `sp_whisper_run`, `sp_whisper_transcribe`,
`sp_whisper_transcribe_full`) so behavior matches the Rust/Go callers
exactly -- including the hardcoded
`whisper_full_default_params(WHISPER_SAMPLING_GREEDY)` shape, the
special-token filtering for `avg_logprob`, the segment-mean
`no_speech_prob`, and the `-2 -> resize buffer` contract on the C side.

The pybind11 layer exposes a `WhisperContext` class with methods
`open(model_path)`, `transcribe(samples, language=None)`,
`transcribe_full(samples)`, `close()`. The samples input is a 1-D
`numpy.ndarray` of `float32` 16 kHz PCM (other dtypes auto-cast via
`py::array::forcecast`). `transcribe` returns `{"text": str}`;
`transcribe_full` returns `{"text", "avg_logprob", "no_speech_prob"}` with
`None` for stats the model couldn't compute (NaN on the C side).

## Build

Requires Python >= 3.10, `pybind11 >= 2.13` (build-time only; declared in
`pyproject.toml`), the `libwhisper` shared library + `whisper.h` header
(and its dependency `ggml.h`, pulled in transitively), and a C++17
compiler.

```sh
cd whisper_bindings
pip install .                         # installs as a separate package
# -- or --
python setup.py build_ext --inplace   # produces _whisper.cpython-*.so here
```

Either path produces an extension importable as
`whisper_bindings._whisper`. If headers/library aren't on the default
search path, set `WHISPER_INCLUDE_DIR` / `WHISPER_LIBRARY_DIR`
(colon-separated like `PATH`). `setup.py` also probes `/usr/include`,
`/usr/local/include`, `/opt/homebrew/include` (and the matching `lib/`
paths), so Homebrew installs of `whisper-cpp` are found without extra
config.

## Nix integration

The headers and `libwhisper.so` come from the nixpkgs `whisper-cpp`
package: add `pkgs.whisper-cpp` to the dev shell's `packages` /
`buildInputs` plus `pkgs.python3Packages.pybind11`. `whisper.h` lands at
`${pkgs.whisper-cpp}/include/whisper.h` and the shared library at
`${pkgs.whisper-cpp}/lib/libwhisper.so`, both picked up automatically by
`nix develop`'s `NIX_CFLAGS_COMPILE` / `NIX_LDFLAGS`.

## Fallback

The sibling `stt/whisper_cpp.py` `WhisperCppBackend` checks in this order:

1. `from whisper_bindings import _whisper` (the local extension built here).
2. *(no PyPI fallback)* -- whisper.cpp has no canonical PyPI wheel. The
   closest is `pywhispercpp`, which carries a different ABI and would need
   a separate adapter; we don't bundle it.
3. Otherwise raises an informative `RuntimeError` pointing back here.

Casual users on a dev shell without `libwhisper` therefore can't use the
whisper.cpp backend -- build this extension or fall back to
`Ct2WhisperBackend` (which does have a PyPI fallback via
`pip install '.[ct2]'`).

## Why a separate buildable

Same reasoning as `../ct2_bindings/README.md` § "Why a separate buildable":
the main package must install cleanly without `libwhisper` headers, and
isolating the C++ extension means a broken `whisper_bindings` build can
never break a `pip install .` of `speaches-plus-python`.
