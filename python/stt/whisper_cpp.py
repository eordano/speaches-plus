from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .whisper import TimedSegment, TranscriptionResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_BACKEND_NATIVE = "native"
_BACKEND_NONE = "none"

def _select_backend() -> tuple[str, Any, str | None]:
    try:
        from whisper_bindings import _whisper as _native_mod

        if _native_mod is not None:
            return _BACKEND_NATIVE, _native_mod, None
        try:
            from whisper_bindings import EXTENSION_IMPORT_ERROR as _err

            native_err = _err
        except ImportError:
            native_err = "whisper_bindings._whisper attribute is None"
    except ImportError as e:
        native_err = str(e)

    msg = (
        "The local whisper_bindings._whisper extension is not importable, "
        "and whisper.cpp has no canonical PyPI wheel to fall back to.\n"
        f"  whisper_bindings._whisper import error: {native_err}\n"
        "Build the local extension (see whisper_bindings/README.md): "
        "`cd whisper_bindings && pip install .` once libwhisper + whisper.h "
        "are on the include/library search path (or set WHISPER_INCLUDE_DIR "
        "and WHISPER_LIBRARY_DIR)."
    )
    return _BACKEND_NONE, msg, native_err

@dataclass(frozen=True)
class WhisperCppConfig:
    model_path: str
    language: str = "en"

class WhisperCppBackend:
    model_id: str = ""

    def __init__(self, model_path: str, *, language: str = "en") -> None:
        if not model_path:
            raise RuntimeError(
                "WhisperCppBackend: model_path must be a non-empty string"
            )
        self._model_path = model_path
        self._language = language
        import os as _os
        self.model_id = _os.path.basename(model_path.rstrip("/")) or "whisper"
        self._backend, self._mod_or_err, self._native_err = _select_backend()
        if self._backend == _BACKEND_NONE:
            raise RuntimeError(self._mod_or_err)
        self._handle: Any = None
        self._open()

    def _open(self) -> None:
        if self._backend == _BACKEND_NATIVE:
            try:
                self._handle = self._mod_or_err.WhisperContext(self._model_path)
            except Exception as e:
                raise RuntimeError(
                    f"WhisperCppBackend: failed to load whisper.cpp model at "
                    f"{self._model_path!r}: {e}"
                ) from e
        else:
            raise RuntimeError("WhisperCppBackend: unreachable backend state")

    @property
    def backend_kind(self) -> str:
        return self._backend

    def close(self) -> None:
        if self._handle is None:
            return
        try:
            self._handle.close()
        except Exception:
            pass
        self._handle = None

    def __enter__(self) -> WhisperCppBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def transcribe(
        self,
        samples: NDArray[np.float32],
        sample_rate: int = 16000,
        *,
        language: str | None = None,
        prompt: str | None = None,
        with_timestamps: bool = False,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        del prompt
        del with_timestamps
        if self._handle is None:
            raise RuntimeError("WhisperCppBackend: handle is closed")
        if sample_rate != 16000:
            raise ValueError(
                f"WhisperCppBackend: expected 16 kHz sample rate, got {sample_rate}"
            )
        if task not in ("transcribe", "translate"):
            raise ValueError(
                f"WhisperCppBackend.transcribe: task must be 'transcribe' or "
                f"'translate'; got {task!r}"
            )
        if task == "translate":
            return self.translate(samples, sample_rate=sample_rate, language=language)

        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError(
                "WhisperCppBackend.transcribe: numpy is required to pass "
                "samples to the whisper.cpp extension."
            ) from e

        if samples.size == 0:
            return TranscriptionResult(text="")

        if samples.ndim != 1:
            raise ValueError(
                f"WhisperCppBackend.transcribe: samples must be 1-D float32 "
                f"PCM at 16 kHz; got shape {samples.shape}"
            )

        samples_f32 = np.ascontiguousarray(samples, dtype=np.float32)
        lang = language if language is not None else self._language

        res = self._handle.transcribe_full(samples_f32)
        text = res.get("text", "")
        avg_logprob = res.get("avg_logprob")
        no_speech_prob = res.get("no_speech_prob")
        del lang
        return TranscriptionResult(
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
        )

    def translate(
        self,
        samples: NDArray[np.float32],
        sample_rate: int = 16000,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        if self._handle is None:
            raise RuntimeError("WhisperCppBackend: handle is closed")
        if sample_rate != 16000:
            raise ValueError(
                f"WhisperCppBackend: expected 16 kHz sample rate, got {sample_rate}"
            )
        translate_fn = getattr(self._handle, "translate_full", None)
        if translate_fn is None:
            raise NotImplementedError(
                "WhisperCppBackend.translate: the underlying whisper.cpp binding "
                "does not expose a translate path (rebuild whisper_bindings with "
                "wparams.translate=true wired through, or call /v1/audio/transcriptions "
                "instead)."
            )
        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError(
                "WhisperCppBackend.translate: numpy is required."
            ) from e
        if samples.size == 0:
            return TranscriptionResult(text="")
        if samples.ndim != 1:
            raise ValueError(
                f"WhisperCppBackend.translate: samples must be 1-D float32 PCM "
                f"at 16 kHz; got shape {samples.shape}"
            )
        samples_f32 = np.ascontiguousarray(samples, dtype=np.float32)
        lang = language if language is not None else self._language
        res = translate_fn(samples_f32, lang)
        text = res.get("text", "")
        avg_logprob = res.get("avg_logprob")
        no_speech_prob = res.get("no_speech_prob")
        return TranscriptionResult(
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
        )

    def transcribe_segmented(
        self,
        samples: NDArray[np.float32],
        sample_rate: int = 16000,
        *,
        language: str | None = None,
    ) -> TranscriptionResult:
        if self._handle is None:
            raise RuntimeError("WhisperCppBackend: handle is closed")
        if sample_rate != 16000:
            raise ValueError(
                f"WhisperCppBackend: expected 16 kHz sample rate, got {sample_rate}"
            )
        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError(
                "WhisperCppBackend.transcribe_segmented: numpy is required to "
                "pass samples to the whisper.cpp extension."
            ) from e
        if samples.size == 0:
            return TranscriptionResult(text="")
        if samples.ndim != 1:
            raise ValueError(
                f"WhisperCppBackend.transcribe_segmented: samples must be 1-D "
                f"float32 PCM at 16 kHz; got shape {samples.shape}"
            )
        samples_f32 = np.ascontiguousarray(samples, dtype=np.float32)
        lang = language if language is not None else self._language
        res = self._handle.transcribe_segmented(samples_f32, lang)
        text = res.get("text", "")
        avg_logprob = res.get("avg_logprob")
        no_speech_prob = res.get("no_speech_prob")
        raw_segments = res.get("segments", []) or []
        segments: list[TimedSegment] = []
        for seg in raw_segments:
            t0 = int(seg.get("t_start_ms", 0))
            t1 = int(seg.get("t_end_ms", 0))
            seg_text = seg.get("text", "")
            seg_lp = seg.get("avg_logprob")
            seg_nsp = seg.get("no_speech_prob")
            segments.append(
                TimedSegment(
                    t_start_ms=t0,
                    t_end_ms=t1,
                    text=seg_text,
                    avg_logprob=seg_lp,
                    no_speech_prob=seg_nsp,
                )
            )
        return TranscriptionResult(
            text=text,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
            segments=segments,
        )

    def transcribe_text(
        self,
        samples: NDArray[np.float32],
        *,
        language: str | None = None,
    ) -> str:
        if self._handle is None:
            raise RuntimeError("WhisperCppBackend: handle is closed")
        try:
            import numpy as np
        except ImportError as e:
            raise RuntimeError(
                "WhisperCppBackend.transcribe_text: numpy is required."
            ) from e
        if samples.size == 0:
            return ""
        if samples.ndim != 1:
            raise ValueError(
                f"WhisperCppBackend.transcribe_text: samples must be 1-D; "
                f"got shape {samples.shape}"
            )
        samples_f32 = np.ascontiguousarray(samples, dtype=np.float32)
        lang = language if language is not None else self._language
        res = self._handle.transcribe(samples_f32, lang)
        return res.get("text", "")

__all__ = [
    "WhisperCppBackend",
    "WhisperCppConfig",
]
