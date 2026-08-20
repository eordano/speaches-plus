# ct2_bindings -- pybind11 wrapper around CTranslate2's Whisper model

Python equivalent of the upstream Go cgo wrapper at
`speaches-plus/go/internal/stt/ct2_cgo.cc`. It is intentionally vendored
here rather than pulled from the PyPI `ctranslate2` wheel: we want a single
C++ call site that mirrors upstream's `extern "C"` ABI symbol names
(`sp_ct2_open`, `sp_ct2_close`, `sp_ct2_n_mels`, `sp_ct2_generate`,
`sp_ct2_generate_segmented`) so behavior matches the Rust/Go callers
exactly -- including the `<|notimestamps|>` prompt assembly, the
no_speech_prob / avg_logprob plumbing, and the `-2 -> resize buffer`
contract on the C side.

The pybind11 layer exposes a `Ct2Whisper` class with methods
`open(model_path, device, compute_type)`, `n_mels`, `generate`,
`generate_segmented`, `close`. The mel input is a `numpy.ndarray` with
shape `(n_mels, n_frames)` and dtype `float32` (other dtypes auto-cast via
`py::array::forcecast`). Both `generate` methods return a `dict` with keys
`text`, `tokens_blob` (segmented only), `no_speech_prob`, `avg_logprob` --
`None` when not requested or unavailable.

## Build

Requires Python >= 3.10, `pybind11 >= 2.13` (build-time only; declared in
`pyproject.toml`), the `libctranslate2` shared library + `ctranslate2/*.h`
headers, and a C++17 compiler.

```sh
cd ct2_bindings
pip install .                         # installs as a separate package
# -- or --
python setup.py build_ext --inplace   # produces _ct2.cpython-*.so here
```

Either path produces an extension importable as `ct2_bindings._ct2`. If
headers/library aren't on the default search path, set `CT2_INCLUDE_DIR` /
`CT2_LIBRARY_DIR` (colon-separated like `PATH`). `setup.py` also probes
`/usr/include`, `/usr/local/include`, `/opt/homebrew/include` (and the
matching `lib/` paths), so Homebrew installs of `ctranslate2` are found
without extra config.

## Nix integration

The dev shell in `flake.nix` provides `pkgs.ctranslate2` (the C++ library,
not the PyPI wheel) plus `pkgs.python3Packages.pybind11` and the C++
toolchain, and pre-sets `CT2_INCLUDE_DIR` / `CT2_LIBRARY_DIR` to the store
paths. Build with `nix develop` then `bash scripts/build_bindings.sh`
(builds both ct2_bindings and whisper_bindings) or
`cd ct2_bindings && python setup.py build_ext --inplace`.

If `pkgs.ctranslate2` isn't reachable (building outside `nix develop`),
fall back to `pip install ctranslate2` in the venv and the PyPI path in
`stt/ct2.py` (below).

## Fallback

The sibling `stt/ct2.py` `Ct2WhisperBackend` checks in this order:

1. `from ct2_bindings import _ct2` (the local extension built here).
2. `import ctranslate2` (PyPI wheel) -- used only if (1) failed.
3. Otherwise raises an informative `RuntimeError` pointing back here.

So casual users on a dev shell without `libctranslate2` can still satisfy
the dep with `pip install '.[ct2]'` from the repo root (see the `[ct2]`
extra in the top-level `pyproject.toml`).

## Why a separate buildable

`ct2_bindings/` is intentionally a *sibling* project, not part of the main
`speaches-plus-python` package's `[tool.setuptools.packages.find]`. Two
reasons: the main package must install cleanly on machines without
`libctranslate2` headers (most dev shells today), and a C++ extension
forces PEP 517 `build_ext` territory -- isolating it means a broken
`ct2_bindings` build can never break a `pip install .` of
`speaches-plus-python`.
