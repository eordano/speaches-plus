from __future__ import annotations

import json
from typing import Any

import numpy as np

from .constants import WHISPER_SAMPLING_HZ
from .segments import transcribe_long
from .whisper import TranscriptionResult, WhisperBackend, join_segments

MULTIPART_PARSE_LIMIT = 100 * 1024 * 1024
FILE_READ_LIMIT = 200 * 1024 * 1024
FORMAT_TEXT = "text"
FORMAT_JSON = "json"
FORMAT_VERBOSE_JSON = "verbose_json"
FORMAT_DIARIZED_JSON = "diarized_json"
FORMAT_SRT = "srt"
FORMAT_VTT = "vtt"
CONTENT_TYPE_TEXT = "text/plain; charset=utf-8"
CONTENT_TYPE_JSON = "application/json"

def _decode_audio_bytes(raw: bytes, content_type: str | None) -> np.ndarray:
    try:
        from audio import codecs as audio_codecs  # type: ignore

        if hasattr(audio_codecs, "decode_any"):
            samples = audio_codecs.decode_any(raw, content_type or "")
            return np.asarray(samples, dtype=np.float32)
    except Exception:
        pass
    try:
        import io
        import soundfile as sf

        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if sr != WHISPER_SAMPLING_HZ:
            try:
                import librosa

                data = librosa.resample(data, orig_sr=sr, target_sr=WHISPER_SAMPLING_HZ)
            except Exception:
                pass
        return np.asarray(data, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"audio decode failed: {e}") from e

def _format_srt_time(ms: int) -> str:
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1_000
    msec = ms % 1_000
    return f"{h:02d}:{m:02d}:{s:02d},{msec:03d}"

def _format_vtt_time(ms: int) -> str:
    h = ms // 3_600_000
    m = (ms % 3_600_000) // 60_000
    s = (ms % 60_000) // 1_000
    msec = ms % 1_000
    return f"{h:02d}:{m:02d}:{s:02d}.{msec:03d}"

def _result_to_srt(res: TranscriptionResult) -> str:
    lines: list[str] = []
    for i, seg in enumerate(res.segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_srt_time(seg.t_start_ms)} --> {_format_srt_time(seg.t_end_ms)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)

def _result_to_vtt(res: TranscriptionResult) -> str:
    lines: list[str] = ["WEBVTT", ""]
    for seg in res.segments:
        lines.append(f"{_format_vtt_time(seg.t_start_ms)} --> {_format_vtt_time(seg.t_end_ms)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)

def _result_to_verbose_json(
    res: TranscriptionResult,
    language: str | None,
    duration_s: float,
    task: str = "transcribe",
) -> dict[str, Any]:
    return {
        "task": task,
        "language": language or "en",
        "duration": duration_s,
        "text": res.text,
        "segments": [
            {
                "id": i,
                "start": s.t_start_ms / 1000.0,
                "end": s.t_end_ms / 1000.0,
                "text": s.text,
                "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob,
            }
            for i, s in enumerate(res.segments)
        ],
        "words": [],
    }

def _backend_model_id(backend: WhisperBackend) -> str:
    mid = getattr(backend, "model_id", None)
    if callable(mid):
        try:
            return str(mid())
        except Exception:
            return "whisper"
    if mid is None:
        return "whisper"
    return str(mid)

async def _read_upload(file: Any) -> tuple[bytes, str | None]:
    raw = await file.read()
    if len(raw) > FILE_READ_LIMIT:
        from fastapi import HTTPException

        raise HTTPException(status_code=413, detail=f"file exceeds maximum size of {FILE_READ_LIMIT} bytes")
    content_type = getattr(file, "content_type", None)
    return raw, content_type

async def _run_backend(
    backend: WhisperBackend,
    file: Any,
    response_format: str,
    language: str | None,
    prompt: str | None,
    task: str,
):
    from fastapi import HTTPException
    from fastapi.responses import JSONResponse, PlainTextResponse

    if backend is None:
        raise HTTPException(status_code=503, detail="transcriber not configured")
    raw, content_type = await _read_upload(file)
    samples = _decode_audio_bytes(raw, content_type)
    duration_s = len(samples) / float(WHISPER_SAMPLING_HZ)
    res = transcribe_long(
        backend, samples, WHISPER_SAMPLING_HZ,
        language=language, prompt=prompt, task=task,
    )
    model_id = _backend_model_id(backend)
    out_language = language if language else ("en" if task == "translate" else "en")
    fmt = (response_format or FORMAT_JSON).lower()
    if fmt == FORMAT_TEXT:
        return PlainTextResponse(res.text, media_type=CONTENT_TYPE_TEXT)
    if fmt == FORMAT_JSON:
        return JSONResponse({
            "text": res.text,
            "language": out_language,
            "model": model_id,
            "task": task,
        })
    if fmt == FORMAT_VERBOSE_JSON:
        return JSONResponse(_result_to_verbose_json(res, out_language, duration_s, task=task))
    if fmt == FORMAT_SRT:
        return PlainTextResponse(_result_to_srt(res), media_type="application/x-subrip")
    if fmt == FORMAT_VTT:
        return PlainTextResponse(_result_to_vtt(res), media_type="text/vtt")
    if fmt == FORMAT_DIARIZED_JSON:
        body = {
            "text": res.text,
            "language": out_language,
            "model": model_id,
            "task": task,
            "avg_logprob": res.avg_logprob,
            "no_speech_prob": res.no_speech_prob,
            "segments": [
                {
                    "type": "transcript.text.segment",
                    "id": f"seg_{i+1:03d}",
                    "speaker": None,
                    "start": s.t_start_ms / 1000.0,
                    "end": s.t_end_ms / 1000.0,
                    "duration": (s.t_end_ms - s.t_start_ms) / 1000.0,
                    "text": s.text,
                    "avg_logprob": s.avg_logprob,
                    "no_speech_prob": s.no_speech_prob,
                    "confidence": None,
                }
                for i, s in enumerate(res.segments)
            ],
        }
        return JSONResponse(body)
    raise HTTPException(
        status_code=400,
        detail=f"unsupported response_format: {response_format!r} (supported: text, json, verbose_json, srt, vtt, diarized_json)",
    )

async def transcriptions_post(
    backend: WhisperBackend,
    file: Any,
    response_format: str = FORMAT_JSON,
    language: str | None = None,
    prompt: str | None = None,
    temperature: float = 0.0,
):
    del temperature
    return await _run_backend(
        backend=backend,
        file=file,
        response_format=response_format,
        language=language,
        prompt=prompt,
        task="transcribe",
    )

async def translations_post(
    backend: WhisperBackend,
    file: Any,
    response_format: str = FORMAT_JSON,
    prompt: str | None = None,
    temperature: float = 0.0,
):
    del temperature
    return await _run_backend(
        backend=backend,
        file=file,
        response_format=response_format,
        language=None,
        prompt=prompt,
        task="translate",
    )
