from __future__ import annotations

import ctypes.util
import warnings
from pathlib import Path

import env

def _resolve_espeak_library_path() -> str | None:
    explicit = env.read_str_or_none(env.PHONEMIZER_ESPEAK_LIBRARY)
    if explicit:
        return explicit
    return ctypes.util.find_library("espeak-ng") or ctypes.util.find_library("espeak")

def configure_espeak() -> None:
    library_path = _resolve_espeak_library_path()
    if not library_path:
        return
    if not Path(library_path).is_file():
        warnings.warn(
            f"espeak library path {library_path!r} does not exist; skipping",
            stacklevel=2,
        )
        return
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
    except ImportError:
        return
    EspeakWrapper.set_library(library_path)
    data_path = env.read_str_or_none(env.ESPEAK_DATA_PATH)
    if data_path:
        if not Path(data_path).is_dir():
            warnings.warn(
                f"espeak data path {data_path!r} is not a directory; skipping",
                stacklevel=2,
            )
            return
        setter = getattr(EspeakWrapper, "set_data_path", None)
        if callable(setter):
            setter(data_path)
        else:
            EspeakWrapper.data_path = data_path

def phonemize(text: str, language: str) -> str:
    import phonemizer
    return phonemizer.phonemize(
        text.strip(), language, preserve_punctuation=True, with_stress=True,
    )
