from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .whisper import TimedSegment as Segment, TranscriptionResult

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

_BACKEND_NATIVE = "native"
_BACKEND_PYPI = "pypi"
_BACKEND_NONE = "none"

def _select_backend() -> tuple[str, Any, str | None]:
    try:
        from ct2_bindings import _ct2 as _native_mod

        if _native_mod is not None:
            return _BACKEND_NATIVE, _native_mod, None
        try:
            from ct2_bindings import EXTENSION_IMPORT_ERROR as _err

            native_err = _err
        except ImportError:
            native_err = "ct2_bindings._ct2 attribute is None"
    except ImportError as e:
        native_err = str(e)

    try:
        import ctranslate2 as _pypi_mod

        return _BACKEND_PYPI, _pypi_mod, None
    except ImportError as pypi_err:
        msg = (
            "Neither the local ct2_bindings._ct2 extension nor the PyPI "
            "ctranslate2 package is importable.\n"
            f"  ct2_bindings._ct2 import error: {native_err}\n"
            f"  ctranslate2 import error:        {pypi_err}\n"
            "Build the local extension (see ct2_bindings/README.md) or "
            "install the PyPI fallback with: pip install '.[ct2]'"
        )
        return _BACKEND_NONE, msg, native_err

@dataclass(frozen=True)
class Ct2WhisperConfig:
    model_path: str
    device: str = "cpu"
    compute_type: str = "default"
    language: str = "en"
    beam_size: int = 5
    no_speech_prob_threshold: float | None = 0.6

_BPE_RUNE_TO_BYTE: dict[str, int] = {}

def _build_bpe_table() -> None:
    if _BPE_RUNE_TO_BYTE:
        return
    bs: list[int] = []
    for c in range(ord("!"), ord("~") + 1):
        bs.append(c)
    for c in range(ord("¡"), ord("¬") + 1):
        bs.append(c)
    for c in range(ord("®"), ord("ÿ") + 1):
        bs.append(c)
    cs = list(bs)
    seen = set(bs)
    n = 0
    for b in range(256):
        if b not in seen:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    for codepoint, byte in zip(cs, bs, strict=True):
        _BPE_RUNE_TO_BYTE[chr(codepoint)] = byte

def decode_bpe(s: str) -> str:
    _build_bpe_table()
    out = bytearray()
    for ch in s:
        b = _BPE_RUNE_TO_BYTE.get(ch)
        if b is not None:
            out.append(b)
        else:
            out.extend(ch.encode("utf-8"))
    return out.decode("utf-8", errors="replace")

def _parse_timestamp_token(tok: str) -> int | None:
    if len(tok) < 5 or tok[0] != "<" or tok[1] != "|" or tok[-2] != "|" or tok[-1] != ">":
        return None
    inner = tok[2:-2]
    dot = inner.find(".")
    if dot < 1 or dot >= len(inner) - 1:
        return None
    whole = inner[:dot]
    frac = inner[dot + 1 :]
    if not whole.isdigit() or not frac.isdigit():
        return None
    secs = int(whole)
    frac_val = int(frac)
    if len(frac) == 1:
        frac_ms = frac_val * 100
    elif len(frac) == 2:
        frac_ms = frac_val * 10
    elif len(frac) == 3:
        frac_ms = frac_val
    else:
        div = 10 ** (len(frac) - 3)
        frac_ms = frac_val // div
    return secs * 1000 + frac_ms

def _parse_segments_from_tokens(blob: str) -> list[Segment]:
    if not blob:
        return []
    tokens = blob.rstrip("\n").split("\n")
    segs: list[Segment] = []
    cur_start: int | None = None
    cur_tokens: list[str] = []

    def _flush(end_ms: int) -> None:
        nonlocal cur_tokens
        if cur_start is None:
            return
        text = decode_bpe("".join(cur_tokens)).strip()
        if text:
            segs.append(Segment(t_start_ms=cur_start, t_end_ms=end_ms, text=text))
        cur_tokens = []

    for tok in tokens:
        ms = _parse_timestamp_token(tok)
        if ms is not None:
            if cur_start is None:
                cur_start = ms
            else:
                _flush(ms)
                cur_start = ms
            continue
        if cur_start is None:
            continue
        cur_tokens.append(tok)
    return segs

class Ct2WhisperBackend:
    model_id: str = ""

    def __init__(self, config: Ct2WhisperConfig) -> None:
        self._config = config
        self.model_id = os.path.basename(config.model_path.rstrip("/")) or "whisper"
        self._backend, self._mod_or_err, self._native_err = _select_backend()
        if self._backend == _BACKEND_NONE:
            raise RuntimeError(self._mod_or_err)
        self._handle: Any = None
        self._open()

    def _open(self) -> None:
        cfg = self._config
        if self._backend == _BACKEND_NATIVE:
            try:
                self._handle = self._mod_or_err.Ct2Whisper(
                    cfg.model_path, cfg.device, cfg.compute_type
                )
            except Exception as e:
                raise RuntimeError(
                    f"Ct2WhisperBackend: failed to open model at "
                    f"{cfg.model_path!r} (device={cfg.device!r}, "
                    f"compute_type={cfg.compute_type!r}): {e}"
                ) from e
        elif self._backend == _BACKEND_PYPI:
            try:
                self._handle = self._mod_or_err.models.Whisper(
                    cfg.model_path,
                    device=cfg.device,
                    compute_type=cfg.compute_type,
                )
            except Exception as e:
                raise RuntimeError(
                    f"Ct2WhisperBackend: failed to open model at "
                    f"{cfg.model_path!r} via PyPI ctranslate2 "
                    f"(device={cfg.device!r}, compute_type={cfg.compute_type!r}): {e}"
                ) from e
        else:
            raise RuntimeError("Ct2WhisperBackend: unreachable backend state")

    @property
    def n_mels(self) -> int:
        if self._handle is None:
            raise RuntimeError("Ct2WhisperBackend: handle is closed")
        if self._backend == _BACKEND_NATIVE:
            return int(self._handle.n_mels)
        return int(self._handle.n_mels)

    @property
    def backend_kind(self) -> str:
        return self._backend

    def close(self) -> None:
        if self._handle is None:
            return
        if self._backend == _BACKEND_NATIVE:
            try:
                self._handle.close()
            except Exception:
                pass
        self._handle = None

    def __enter__(self) -> Ct2WhisperBackend:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def transcribe_mel(
        self,
        mel: NDArray[np.float32],
        *,
        language: str | None = None,
        with_timestamps: bool = False,
        return_no_speech_prob: bool = True,
        return_avg_logprob: bool = True,
        task: str = "transcribe",
    ) -> TranscriptionResult:
        if self._handle is None:
            raise RuntimeError("Ct2WhisperBackend: handle is closed")
        if mel.ndim != 2:
            raise ValueError(
                f"Ct2WhisperBackend.transcribe_mel: mel must be 2-D "
                f"(n_mels, n_frames); got shape {mel.shape}"
            )
        lang = language if language is not None else self._config.language
        lang_token = lang if lang.startswith("<|") else f"<|{lang}|>"
        task_token = "<|translate|>" if task == "translate" else "<|transcribe|>"

        if self._backend == _BACKEND_NATIVE:
            return self._transcribe_native(
                mel,
                lang_token,
                task_token,
                with_timestamps,
                return_no_speech_prob,
                return_avg_logprob,
            )
        return self._transcribe_pypi(
            mel,
            lang_token,
            task_token,
            with_timestamps,
            return_no_speech_prob,
            return_avg_logprob,
        )

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
        if sample_rate != 16000:
            raise ValueError(
                f"Ct2WhisperBackend: expected 16 kHz sample rate, got {sample_rate}"
            )
        if samples.size == 0:
            return TranscriptionResult(text="")
        mel = self._compute_mel(samples)
        return self.transcribe_mel(
            mel,
            language=language,
            with_timestamps=with_timestamps,
            task=task,
        )

    def _compute_mel(self, samples: NDArray[np.float32]) -> NDArray[np.float32]:
        from .mel import MelExtractor
        extractor = getattr(self, "_mel_extractor", None)
        if extractor is None or extractor.n_mels != self.n_mels:
            extractor = MelExtractor(n_mels=self.n_mels)
            self._mel_extractor = extractor
        return extractor.extract(samples)

    def _transcribe_native(
        self,
        mel: NDArray[np.float32],
        lang_token: str,
        task_token: str,
        with_timestamps: bool,
        return_no_speech_prob: bool,
        return_avg_logprob: bool,
    ) -> TranscriptionResult:
        if task_token == "<|translate|>":
            raise NotImplementedError(
                "Ct2WhisperBackend native backend does not support task='translate'; "
                "use the PyPI ctranslate2 fallback (uninstall ct2_bindings._ct2 or "
                "set environment to force PyPI) or rebuild ct2_bindings with a "
                "task-token override."
            )
        if with_timestamps:
            res = self._handle.generate_segmented(
                mel,
                lang_token,
                self._config.beam_size,
                return_no_speech_prob,
                return_avg_logprob,
            )
            text = decode_bpe(res["text"])
            segments = _parse_segments_from_tokens(res.get("tokens_blob", ""))
            return TranscriptionResult(
                text=text,
                avg_logprob=res.get("avg_logprob"),
                no_speech_prob=res.get("no_speech_prob"),
                segments=segments,
            )
        res = self._handle.generate(
            mel,
            lang_token,
            self._config.beam_size,
            return_no_speech_prob,
            return_avg_logprob,
        )
        return TranscriptionResult(
            text=decode_bpe(res["text"]),
            avg_logprob=res.get("avg_logprob"),
            no_speech_prob=res.get("no_speech_prob"),
        )

    def _transcribe_pypi(
        self,
        mel: NDArray[np.float32],
        lang_token: str,
        task_token: str,
        with_timestamps: bool,
        return_no_speech_prob: bool,
        return_avg_logprob: bool,
    ) -> TranscriptionResult:
        ct2 = self._mod_or_err
        import numpy as np

        mel_3d = mel.astype(np.float32)[None, :, :]
        features = ct2.StorageView.from_array(mel_3d)
        prompt = ["<|startoftranscript|>", lang_token, task_token]
        if not with_timestamps:
            prompt.append("<|notimestamps|>")
        results = self._handle.generate(
            features,
            [prompt],
            beam_size=max(1, self._config.beam_size),
            return_no_speech_prob=return_no_speech_prob,
            return_scores=return_avg_logprob,
        )
        if not results:
            return TranscriptionResult(text="")
        result = results[0]
        nsp = getattr(result, "no_speech_prob", None) if return_no_speech_prob else None
        scores = getattr(result, "scores", None)
        alp: float | None
        if return_avg_logprob and scores:
            alp = float(scores[0])
        else:
            alp = None
        sequences = getattr(result, "sequences", None) or []
        if not sequences or not sequences[0]:
            return TranscriptionResult(text="", no_speech_prob=nsp, avg_logprob=alp)
        tokens = sequences[0]
        text_parts: list[str] = []
        for tok in tokens:
            if (
                len(tok) >= 4
                and tok[0] == "<"
                and tok[1] == "|"
                and tok[-2] == "|"
                and tok[-1] == ">"
            ):
                continue
            text_parts.append(tok)
        text = decode_bpe("".join(text_parts))
        segments: list[Segment] = []
        if with_timestamps:
            blob = "\n".join(tokens) + "\n"
            segments = _parse_segments_from_tokens(blob)
        return TranscriptionResult(
            text=text,
            avg_logprob=alp,
            no_speech_prob=nsp,
            segments=segments,
        )

__all__ = [
    "Ct2WhisperBackend",
    "Ct2WhisperConfig",
    "Segment",
    "TranscriptionResult",
    "decode_bpe",
]
