from __future__ import annotations

import os
import sys

from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError as e:
    raise RuntimeError(
        "pybind11 is required to build whisper_bindings. "
        "Install it with `pip install pybind11>=2.13`."
    ) from e

def _split_env(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [p for p in raw.split(os.pathsep) if p]

def _maybe_default(path: str) -> list[str]:
    return [path] if os.path.isdir(path) else []

_extra_includes: list[str] = []
_extra_includes += _split_env("WHISPER_INCLUDE_DIR")
_extra_includes += _maybe_default("/usr/include")
_extra_includes += _maybe_default("/usr/local/include")
_extra_includes += _maybe_default("/opt/homebrew/include")

_extra_lib_dirs: list[str] = []
_extra_lib_dirs += _split_env("WHISPER_LIBRARY_DIR")
_extra_lib_dirs += _maybe_default("/usr/lib")
_extra_lib_dirs += _maybe_default("/usr/local/lib")
_extra_lib_dirs += _maybe_default("/opt/homebrew/lib")

_extra_compile_args = ["-std=c++17", "-O2"]
_extra_link_args: list[str] = []

if sys.platform == "darwin":
    _extra_compile_args += ["-stdlib=libc++"]
    _extra_link_args += ["-stdlib=libc++"]

ext_modules = [
    Pybind11Extension(
        "_whisper",
        ["_whisper.cpp"],
        include_dirs=_extra_includes,
        library_dirs=_extra_lib_dirs,
        libraries=["whisper"],
        extra_compile_args=_extra_compile_args,
        extra_link_args=_extra_link_args,
        cxx_std=17,
    ),
]

setup(
    name="whisper_bindings",
    version="0.1.0",
    description="pybind11 bindings around whisper.cpp's whisper_full API "
                "(port of speaches-plus/go/internal/stt/whisper_cgo.c).",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.10",
)
